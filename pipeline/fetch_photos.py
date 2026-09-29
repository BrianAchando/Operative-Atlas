"""Openly licensed tissue photographs from Wikimedia Commons, for pipeline/textures.py.

    python pipeline/fetch_photos.py                      search and download candidates, write a contact sheet
    (open pipeline/photos/_candidates/index.html, choose)
    python pipeline/fetch_photos.py --keep myocardium=3 fat=1,4 artery=2:0.35/0.2/0.4
                                                         copy the chosen candidates into pipeline/photos/<tissue>/
    python pipeline/textures.py                          make the textures (with an attribution file)

Only files whose Commons licence metadata is public domain, CC0, CC BY or CC BY-SA are downloaded; the licence, author
and source page travel with each photo into public/data/textures/ATTRIBUTION.md (CC BY-SA also asks that the textures
made from it are shared under the same licence). Commons photographs of surgery are already published; still, choose
close-ups of tissue only, with nothing that could identify a patient.

A crop can follow the candidate number, as fractions of the image: x,y,size (the top-left corner and the side of a
square), e.g. 2:0.35/0.2/0.4, to take just the patch of tissue from a wider photograph.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHOTOS = HERE / 'photos'; CAND = PHOTOS / '_candidates'
API = 'https://commons.wikimedia.org/w/api.php'
UA = 'OperativeAtlas/0.1 (surgical teaching atlas; texture sourcing) python-urllib'
OK_LICENCES = re.compile(r'^(public domain|pd|cc0|cc[ -]by(-sa)?[ -]?[0-9.]*)', re.I)
QUERIES = {
    'myocardium': ['open heart surgery', 'coronary artery bypass surgery', 'beating heart surgery', 'heart surgery epicardium'],
    'fat': ['epicardial fat', 'adipose tissue surgery', 'lipoma excision'],
    'artery': ['aorta surgery', 'aortic aneurysm surgery', 'carotid endarterectomy'],
    'pulm-artery': ['pulmonary artery surgery', 'pulmonary embolectomy'],
    'vein': ['vena cava surgery', 'saphenous vein harvest', 'vein surgery'],
    'pericardium': ['pericardium surgery', 'pericardiectomy'],
    'muscle': ['skeletal muscle surgery', 'muscle flap surgery', 'pectoralis major surgery'],
    'bone': ['sternotomy', 'bone surgery exposed', 'rib resection'],
    'cartilage': ['tracheal surgery', 'costal cartilage', 'tracheostomy'],
    'valve': ['heart valve surgery', 'mitral valve surgery', 'aortic valve surgery'],
    'organ': ['stomach surgery', 'bowel surgery laparotomy', 'esophagectomy'],
    'liver': ['liver surgery', 'hepatectomy'],
    'drape': ['surgical drape', 'operating field draped'],
    'fabric': ['vascular graft dacron', 'dacron graft'],
}


def api(params: dict) -> dict:
    q = urllib.parse.urlencode({**params, 'format': 'json'})
    req = urllib.request.Request(f'{API}?{q}', headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=60) as r: return json.loads(r.read())


def strip(s: str) -> str: return html.unescape(re.sub(r'<[^>]+>', '', s or '')).strip()


def search(term: str, n: int = 25) -> list[dict]:
    d = api({'action': 'query', 'generator': 'search', 'gsrsearch': f'{term} filetype:bitmap', 'gsrnamespace': 6, 'gsrlimit': n,
             'prop': 'imageinfo', 'iiprop': 'url|size|mime|extmetadata', 'iiurlwidth': 1600,
             'iiextmetadatafilter': 'LicenseShortName|Artist|Credit|ImageDescription'})
    out = []
    for p in (d.get('query', {}).get('pages', {}) or {}).values():
        ii = (p.get('imageinfo') or [{}])[0]; md = ii.get('extmetadata', {})
        lic = strip(md.get('LicenseShortName', {}).get('value', ''))
        if not OK_LICENCES.match(lic) or ii.get('mime') not in ('image/jpeg', 'image/png') or ii.get('width', 0) < 900: continue
        out.append({'title': p['title'], 'licence': lic, 'author': strip(md.get('Artist', {}).get('value', ''))[:200],
                    'page': ii.get('descriptionurl', ''), 'url': ii.get('thumburl') or ii.get('url'), 'size': [ii.get('width'), ii.get('height')],
                    'description': strip(md.get('ImageDescription', {}).get('value', ''))[:300]})
    return out


def fetch_candidates(per_tissue: int) -> None:
    CAND.mkdir(parents=True, exist_ok=True); sheet = []
    for tissue, terms in QUERIES.items():
        seen, found = set(), []
        for t in terms:
            try: res = search(t)
            except Exception as e: print(f'  {tissue}: search "{t}" failed: {e}'); continue  # noqa: BLE001
            for r in res:
                if r['title'] not in seen: seen.add(r['title']); found.append(r)
            time.sleep(0.5)
        found = found[:per_tissue]; d = CAND / tissue; d.mkdir(exist_ok=True)
        for i, r in enumerate(found, 1):
            f = d / f'{i}.jpg'
            if not f.exists():
                try:
                    req = urllib.request.Request(r['url'], headers={'User-Agent': UA})
                    with urllib.request.urlopen(req, timeout=120) as resp: f.write_bytes(resp.read())
                    time.sleep(0.5)
                except Exception as e: print(f'  {tissue} {i}: download failed: {e}'); continue  # noqa: BLE001
            r['file'] = f'{tissue}/{i}.jpg'
        (d / 'credits.json').write_text(json.dumps(found, indent=1))
        print(f'  {tissue}: {len(found)} candidates')
        sheet.append((tissue, found))
    rows = []
    for tissue, found in sheet:
        cells = ''.join(f'<figure><img src="{html.escape(r["file"])}" loading="lazy"><figcaption><b>{i}</b> {html.escape(r["licence"])} · '
                        f'<a href="{html.escape(r["page"])}" target="_blank">source</a><br>{html.escape(r["title"][5:80])}</figcaption></figure>'
                        for i, r in enumerate(found, 1) if 'file' in r)
        rows.append(f'<h2>{tissue}</h2><div class="g">{cells or "<p>none found</p>"}</div>')
    (CAND / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Tissue photo candidates</title><style>'
                                     'body{font:14px system-ui;background:#111;color:#ddd;margin:20px}.g{display:flex;flex-wrap:wrap;gap:10px}'
                                     'figure{margin:0;width:260px}img{width:260px;height:200px;object-fit:cover;border-radius:6px}a{color:#7cc}</style>'
                                     '<h1>Tissue photo candidates</h1><p>Pick close-ups that show only the tissue. Then run: '
                                     '<code>python pipeline/fetch_photos.py --keep myocardium=3 fat=1,4 ...</code> (a crop may follow a number: 3:x/y/size as fractions)</p>'
                                     + ''.join(rows))
    print(f'  contact sheet: {CAND / "index.html"}')


def keep(specs: list[str]) -> None:
    from PIL import Image
    for spec in specs:
        tissue, picks = spec.split('=', 1)
        src = CAND / tissue; credits = json.loads((src / 'credits.json').read_text())
        dst = PHOTOS / tissue
        if dst.exists(): shutil.rmtree(dst)
        dst.mkdir(parents=True); kept = []
        for p in picks.split(','):
            num, _, crop = p.partition(':')
            r = credits[int(num) - 1]; im = Image.open(CAND / r['file']).convert('RGB')
            if crop:
                try: x, y, s = (float(v) for v in crop.split('/'))
                except ValueError: raise SystemExit('crop as x/y/size, e.g. 2:0.35/0.2/0.4')
                w, h = im.size; side = int(s * min(w, h)); im = im.crop((int(x * w), int(y * h), int(x * w) + side, int(y * h) + side))
            im.save(dst / f'{num}.jpg', quality=95); kept.append(r)
        (dst / 'credits.json').write_text(json.dumps(kept, indent=1))
        print(f'  {tissue}: kept {len(kept)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--keep', nargs='*', help='tissue=n[,n...] (optionally n:x/y/size) to copy into pipeline/photos/<tissue>/')
    ap.add_argument('--per', type=int, default=12, help='candidates per tissue')
    a = ap.parse_args()
    if a.keep: keep(a.keep)
    else: fetch_candidates(a.per)
