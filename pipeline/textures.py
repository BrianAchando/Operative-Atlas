"""Tissue textures from intraoperative photographs.

    python pipeline/textures.py            (reads pipeline/photos/<tissue>/*.jpg|png, writes public/data/textures/)

Put de-identified close-ups of one tissue per folder: pipeline/photos/myocardium/, fat/, artery/, pulm-artery/, vein/,
pericardium/, muscle/, bone/, cartilage/, lung/, valve/, organ/, liver/, drape/, fabric/ (the names are the app's tissue
classes). A close-up that fills the frame with that tissue, in focus, under the operating light, is best; no faces,
tattoos, labels, monitors or anything else that could identify the patient, and consent under your institution's rules.

For each folder: the sharpest photo is cropped to its central square, specular highlights are removed (they would be
baked in and fight the real-time lighting), the image is made seamless (tileable), and a normal map is derived from
its fine detail. The app uses the result with triplanar mapping (the meshes have no UVs), keeping each structure's own
teaching colour and taking the photograph's detail and texture.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
SRC = HERE / 'photos'
OUT = HERE.parent / 'public' / 'data' / 'textures'
SIZE = 1024
# physical size of one texture tile in mm (how large a photographed patch appears on the model)
TILE_MM = {'myocardium': 30, 'fat': 40, 'artery': 25, 'pulm-artery': 25, 'vein': 25, 'pericardium': 35, 'muscle': 30, 'bone': 25,
           'cartilage': 20, 'lung': 40, 'valve': 15, 'organ': 35, 'liver': 40, 'drape': 60, 'fabric': 12}


def sharpness(img: np.ndarray) -> float:
    g = img.mean(2); return float(ndimage.laplace(g).var())


def load_best(folder: Path) -> np.ndarray | None:
    best, score = None, -1.0
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'): continue
        im = Image.open(f).convert('RGB'); w, h = im.size; s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((SIZE, SIZE), Image.LANCZOS)
        a = np.asarray(im, np.float32) / 255.0; sc = sharpness(a)
        if sc > score: best, score = a, sc
    return best


def remove_highlights(a: np.ndarray) -> np.ndarray:
    """specular glare (bright, desaturated) filled from its surroundings by normalised convolution"""
    mx, mn = a.max(2), a.min(2); sat = (mx - mn) / (mx + 1e-6)
    glare = ndimage.binary_dilation((mx > 0.88) & (sat < 0.3), iterations=3)
    if not glare.any(): return a
    w = (~glare).astype(np.float32); out = a.copy()
    for c in range(3):
        num = ndimage.gaussian_filter(a[..., c] * w, 6); den = ndimage.gaussian_filter(w, 6) + 1e-6
        out[..., c] = np.where(glare, num / den, a[..., c])
    return out


def seamless(a: np.ndarray) -> np.ndarray:
    """blend the image with itself shifted by half a tile: the shifted copy covers the edges, so tiles meet without seams"""
    n = a.shape[0]; r = np.roll(a, (n // 2, n // 2), axis=(0, 1))
    t = np.linspace(-1, 1, n); d = np.maximum(np.abs(t)[:, None], np.abs(t)[None, :])       # 0 at the centre, 1 at the edges
    w = np.clip((d - 0.55) / 0.4, 0, 1)[..., None]; w = w * w * (3 - 2 * w)
    return a * (1 - w) + r * w


def flatten(a: np.ndarray) -> np.ndarray:
    """take out large-scale shading (the lighting in the photograph), keep the tissue's own colour and fine detail"""
    low = np.stack([ndimage.gaussian_filter(a[..., c], SIZE / 12, mode='wrap') for c in range(3)], 2)
    return np.clip(a / (low + 1e-3) * a.reshape(-1, 3).mean(0), 0, 1)


def normal_map(a: np.ndarray, strength: float = 3.0) -> np.ndarray:
    h = a.mean(2); h = h - ndimage.gaussian_filter(h, 12, mode='wrap')                        # fine relief only
    gx = ndimage.sobel(h, 1, mode='wrap'); gy = ndimage.sobel(h, 0, mode='wrap')
    n = np.stack([-gx * strength, -gy * strength, np.ones_like(h)], 2); n /= np.linalg.norm(n, axis=2, keepdims=True)
    return (n * 0.5 + 0.5)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); man = {}
    if not SRC.exists():
        SRC.mkdir(parents=True); print(f'  made {SRC}; add a folder per tissue with photographs, then run again'); return
    credits_md = ['# Texture sources', '', 'Tissue textures are derived from these photographs (cropped, glare removed, made seamless). '
                  'Textures from CC BY-SA photographs are shared under CC BY-SA.', '']
    for folder in sorted(p for p in SRC.iterdir() if p.is_dir() and not p.name.startswith('_')):
        a = load_best(folder)
        if a is None: continue
        a = seamless(flatten(remove_highlights(a))); nm = normal_map(a)
        Image.fromarray((a * 255).astype(np.uint8)).save(OUT / f'{folder.name}.jpg', quality=88)
        Image.fromarray((nm * 255).astype(np.uint8)).save(OUT / f'{folder.name}_n.jpg', quality=92)
        man[folder.name] = {'albedo': f'textures/{folder.name}.jpg', 'normal': f'textures/{folder.name}_n.jpg',
                            'mean': [round(float(x), 4) for x in a.reshape(-1, 3).mean(0)], 'tile': TILE_MM.get(folder.name, 30)}
        cf = folder / 'credits.json'
        if cf.exists():
            cr = json.loads(cf.read_text()); man[folder.name]['credit'] = '; '.join(f"{c['title'][5:]} ({c['author']}, {c['licence']})" for c in cr)
            credits_md += [f"- **{folder.name}**: " + '; '.join(f"[{c['title'][5:]}]({c['page']}), {c['author']}, {c['licence']}" for c in cr)]
        else:
            credits_md += [f'- **{folder.name}**: own photographs']
        print(f'  {folder.name}: done')
    (OUT / 'manifest.json').write_text(json.dumps(man, indent=1))
    (OUT / 'ATTRIBUTION.md').write_text('\n'.join(credits_md) + '\n')
    print(f'  {len(man)} tissue textures -> {OUT}')


if __name__ == '__main__':
    main()
