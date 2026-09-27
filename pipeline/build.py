"""Build the Hilum atlas data from one contrast chest CT.

    uv run --project ../thoracic-atlas/pipeline python pipeline/build.py

    python pipeline/build.py --work work/          (after pipeline/segment.py has filled work/)

Inputs in the work folder (written by pipeline/segment.py):
    ct.nii.gz          the CT, RAS
    total.nii.gz       TotalSegmentator 'total' (lobes, heart, great vessels, bones)
    vessels.nii.gz     TotalSegmentator 'lung_vessels' on the left hemithorax (airways, arteries, veins)

Outputs in public/data/: atlas.json, procedures.json, ct.hu8.gz, labels.u8.gz, meshes/*.glb.
World frame: RAS millimetres with the carina at the origin.
"""
from __future__ import annotations

import gzip, json, sys
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import meshing  # noqa: E402  (from the neuro atlas: SDF -> marching cubes -> Taubin -> decimation -> meshopt)
from hilum import Tree, lobe_origins, owner_map  # noqa: E402

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument('--work', type=Path, default=HERE.parent / 'work')
WORK = _ap.parse_args().work
OUT = HERE.parent / 'public' / 'data'
MESH = OUT / 'meshes'
MESH.mkdir(parents=True, exist_ok=True)

TS = {n: int(k) for k, n in json.loads((HERE / 'config' / 'totalsegmentator_total_v2.json').read_text()).items()}

ct_img = nib.as_closest_canonical(nib.load(WORK / 'ct.nii.gz'))
tot_img = nib.as_closest_canonical(nib.load(WORK / 'total.nii.gz'))
lv_img = nib.load(WORK / 'vessels.nii.gz')
CT = np.asanyarray(ct_img.dataobj).astype(np.int16)
T = np.asanyarray(tot_img.dataobj).astype(np.int16)
LV = np.asanyarray(lv_img.dataobj)
AT, AL = tot_img.affine, lv_img.affine
assert np.allclose(ct_img.affine, AT)


def mm(vox: np.ndarray, aff: np.ndarray) -> np.ndarray:
    return vox @ aff[:3, :3].T + aff[:3, 3]


def pull(mask: np.ndarray, src_aff: np.ndarray, dst_shape, dst_aff: np.ndarray) -> np.ndarray:
    m = np.linalg.inv(src_aff) @ dst_aff
    return ndimage.affine_transform(mask.astype(np.uint8), m[:3, :3], offset=m[:3, 3], output_shape=tuple(dst_shape), order=0) > 0


def biggest(msk: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(msk)
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n else msk


ts = lambda *names: np.isin(T, [TS[n] for n in names])

# ------------------------------------------------------------------ lobes and trees on the vessel grid
U = pull(ts('lung_upper_lobe_left'), AT, LV.shape, AL)
D = pull(ts('lung_lower_lobe_left'), AT, LV.shape, AL)
heart_lv = pull(ts('heart'), AT, LV.shape, AL)

air = biggest((LV == 1) | (LV == 2))
art = biggest(LV == 3)
vein_all = LV == 4

wv = np.argwhere(air)
air_t = Tree(air, AL, mm(wv[np.argmax(wv[:, 2])], AL))
air_terr = air_t.territory({'U': U, 'D': D})
size = air_t.subtree_sizes()
# carina: walking down the largest branch from the top of the trachea, the first node with two big children
cur = air_t.root
while True:
    ch = sorted(air_t.children.get(cur, []), key=lambda c: -size[c])
    if len(ch) >= 2 and size[ch[1]] > 40:
        break
    if not ch:
        break
    cur = ch[0]
CARINA_NODE = cur
CARINA = air_t.mm[cur].copy()          # scanner RAS
print('carina (scanner)', CARINA.round(1))


def W(p):
    """scanner RAS -> world (carina at origin)"""
    return (np.asarray(p, float) - CARINA)


def shifted(aff: np.ndarray) -> np.ndarray:
    a = aff.copy(); a[:3, 3] -= CARINA; return a


structures: list[dict] = []
labels_out: dict[str, np.ndarray] = {}      # id -> mask on the lv grid or the total grid (resampled later)


def emit(id_, name, group, colour, mask, aff, faces=8000, opacity=1.0, visible=True, division=None, note=None, sigma=None, label=True):
    if mask is None or not mask.any():
        print('  skip', id_); return
    sig = sigma if sigma is not None else (0.6 if mask.sum() < 20000 else 1.0)
    mesh = meshing.mesh_from_mask(mask, shifted(aff), faces, sigma=sig, taubin_iterations=25)
    if mesh is None:
        print('  skip (no surface)', id_); return
    nbytes = meshing.export_glb(mesh, MESH / f'{id_}.glb', id_)
    b = mesh.bounds
    rec = {'id': id_, 'name': name, 'group': group, 'colour': colour, 'opacity': opacity, 'file': f'meshes/{id_}.glb',
           'centroid': [round(float(x), 1) for x in mesh.vertices.mean(0)], 'bbox': [[round(float(x), 1) for x in b[0]], [round(float(x), 1) for x in b[1]]]}
    if not visible: rec['visible'] = False
    if division: rec['division'] = division
    if note: rec['note'] = note
    structures.append(rec)
    if label:
        # keep only the mask's bounding box (full-size masks for ~30 structures exhaust memory)
        ii = np.argwhere(mask); lo_, hi_ = ii.min(0), ii.max(0) + 1
        a2_ = aff.copy(); a2_[:3, 3] = aff[:3, :3] @ lo_ + aff[:3, 3]
        labels_out[id_] = (mask[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]].copy(), a2_)
    print(f'  {id_:28s} {len(mesh.faces):6d} tris {nbytes / 1024:6.1f} kB')


def emit_mesh(id_, name, group, colour, mesh: trimesh.Trimesh, opacity=1.0, visible=True, note=None):
    nbytes = meshing.export_glb(mesh, MESH / f'{id_}.glb', id_)
    b = mesh.bounds
    rec = {'id': id_, 'name': name, 'group': group, 'colour': colour, 'opacity': opacity, 'file': f'meshes/{id_}.glb', 'schematic': True,
           'centroid': [round(float(x), 1) for x in mesh.vertices.mean(0)], 'bbox': [[round(float(x), 1) for x in b[0]], [round(float(x), 1) for x in b[1]]]}
    if not visible: rec['visible'] = False
    if note: rec['note'] = note
    structures.append(rec)
    print(f'  {id_:28s} schematic {nbytes / 1024:6.1f} kB')


edt_art = ndimage.distance_transform_edt(art, sampling=np.abs(np.diag(AL)[:3]))
edt_air = ndimage.distance_transform_edt(air, sampling=np.abs(np.diag(AL)[:3]))


def division_at(t: Tree, origin: int, edt: np.ndarray, into=7.0, look=14.0) -> dict:
    """Staple line `into` mm down the branch from its origin, pointing proximal -> distal."""
    a = t.along(origin, into); b = t.along(origin, look)
    p = t.mm[a]; d = t.mm[b] - t.mm[origin]
    if np.linalg.norm(d) < 1e-3: d = t.mm[a] - t.mm[t.pred[origin]]
    d = d / (np.linalg.norm(d) + 1e-9)
    r = float(edt[tuple(t.vox[a])])
    return {'point': [round(float(x), 1) for x in W(p)], 'dir': [round(float(x), 3) for x in d], 'radius': round(max(r, 2.5), 1)}


def split_hilar(t: Tree, origin: int, reach: float) -> tuple[np.ndarray, np.ndarray]:
    sub = t.subtree(origin)
    near = sub[t.dist[sub] - t.dist[origin] <= reach]
    far = sub[t.dist[sub] - t.dist[origin] > reach]
    return near, far


def owned(t: Tree, mask: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    vox, own = owner_map(t, mask)
    sel = np.isin(own, nodes)
    out = np.zeros(mask.shape, bool); out[tuple(vox[sel].T)] = True
    return out


# surgical-atlas convention: pulmonary arteries blue (deoxygenated), pulmonary veins red-violet, systemic arteries red
ART, VEIN, BRONCH = '#3f6fd8', '#c2476f', '#e3dac6'
SYSV = '#5d6f9c'
print('== arteries')
av = np.argwhere(art)
art_t = Tree(art, AL, mm(av[np.argmax(av[:, 0])], AL))
art_terr = art_t.territory({'U': U, 'D': D})
u_or = sorted(lobe_origins(art_t, art_terr, 'U', min_size=40), key=lambda i: art_t.dist[i])
d_or = lobe_origins(art_t, art_terr, 'D', min_size=40)
u_c = mm(np.argwhere(U), AL).mean(0)
info = []
for o in u_or:
    sub = art_t.subtree(o); c = art_t.mm[sub].mean(0)
    info.append((o, c))
names: dict[int, tuple[str, str]] = {}
if info:
    names[info[0][0]] = ('pa-truncus-anterior', 'Truncus anterior (A1+2a,b and A3)')
    rest = info[1:]
    ling = [x for x in rest if x[1][2] < u_c[2] - 5]
    if ling:
        lg = min(ling, key=lambda x: x[1][2])
        names[lg[0]] = ('pa-lingular', 'Lingular artery (A4+5)')
    post = [x for x in rest if x[0] not in names]
    for k, (o, _) in enumerate(sorted(post, key=lambda x: art_t.dist[x[0]])):
        names[o] = (f'pa-posterior-{k + 1}', f'Posterior segmental artery {k + 1} (A1+2c)' if len(post) > 1 else 'Posterior segmental artery (A1+2c)')
d_info = [(o, art_t.mm[art_t.subtree(o)].mean(0)) for o in d_or]
if d_info:
    basal = max(d_info, key=lambda x: art_terr['_size'][x[0]])
    names[basal[0]] = ('pa-basal-trunk', 'Basal trunk (lower lobe)')
    sup = [x for x in d_info if x[0] != basal[0]]
    if sup:
        a6 = max(sup, key=lambda x: x[1][2])
        names[a6[0]] = ('pa-a6', 'Superior segmental artery (A6, lower lobe)')
intra_u, intra_d, branch_nodes = [], [], []
for o, (id_, nm) in sorted(names.items(), key=lambda kv: art_t.dist[kv[0]]):
    near, far = split_hilar(art_t, o, 32.0)
    branch_nodes.append(art_t.subtree(o))
    (intra_u if id_ in ('pa-truncus-anterior', 'pa-lingular') or id_.startswith('pa-posterior') else intra_d).append(far)
    emit(id_, nm, 'arteries', ART, owned(art_t, art, near), AL, faces=5000, division=division_at(art_t, o, edt_art))
all_branch = np.concatenate(branch_nodes) if branch_nodes else np.array([], int)
main_nodes = np.setdiff1d(np.flatnonzero(art_t.keep), all_branch)
emit('pa-left', 'Left pulmonary artery', 'arteries', ART, owned(art_t, art, main_nodes), AL, faces=9000)
emit('lul-arteries', 'Upper lobe segmental arteries', 'lul-intra', ART, owned(art_t, art, np.concatenate(intra_u)) if intra_u else None, AL, faces=12000, opacity=0.85, label=False)
emit('lll-arteries', 'Lower lobe segmental arteries', 'lll-intra', ART, owned(art_t, art, np.concatenate(intra_d)) if intra_d else None, AL, faces=12000, opacity=0.6, visible=False, label=False)

print('== veins')
la = ndimage.binary_dilation(heart_lv, iterations=2)
vein = biggest(vein_all & ~la)
# The superior and inferior veins usually touch somewhere in the lung, so they come out as one component. Split it
# at the atrium: every place the tree meets the atrium is an ostium, and each skeleton node belongs to the ostium it
# is closest to along the vessel. The ostium whose territory lies in the upper lobe is the superior vein.
touch = vein & ndimage.binary_dilation(la, iterations=2)
tl, tn = ndimage.label(touch, structure=np.ones((3, 3, 3)))
tc = np.bincount(tl.ravel()); tc[0] = 0
ostia = [int(i) for i in np.argsort(-tc)[:4] if tc[i] >= 20]
vt = Tree(vein, AL, mm(np.argwhere(tl == ostia[0]).mean(0), AL))
kd_v = cKDTree(vt.vox)
src_nodes, src_lab = [], []
for k, o in enumerate(ostia):
    nn = np.unique(kd_v.query(np.argwhere(tl == o))[1]); src_nodes.append(nn); src_lab.append(np.full(len(nn), k))
from scipy.sparse.csgraph import dijkstra as _dij
allsrc = np.concatenate(src_nodes); srclab = dict(zip(allsrc.tolist(), np.concatenate(src_lab).tolist()))
_d, _p, _s = _dij(vt.g, indices=allsrc, min_only=True, return_predecessors=True)
part = np.array([srclab.get(int(s), -1) if np.isfinite(dd) else -1 for s, dd in zip(_s, _d)])
inU, inD = U[tuple(vt.vox.T)], D[tuple(vt.vox.T)]
fracU = [inU[part == k].sum() / max(1, (inU | inD)[part == k].sum()) for k in range(len(ostia))]
print('  ostia', [(int(tc[o]), round(float(f), 2)) for o, f in zip(ostia, fracU)])
vox_v, own_v = owner_map(vt, vein)
spv, ipv = np.zeros(vein.shape, bool), np.zeros(vein.shape, bool)
roots = {}
for k, o in enumerate(ostia):
    if (part == k).sum() < 40: continue
    tgt = spv if fracU[k] >= 0.5 else ipv
    tgt[tuple(vox_v[np.isin(own_v, np.flatnonzero(part == k))].T)] = True
    roots.setdefault('pv-superior' if fracU[k] >= 0.5 else 'pv-inferior', np.argwhere(tl == o).mean(0))
for id_, nm, msk in (('pv-superior', 'Superior pulmonary vein', spv), ('pv-inferior', 'Inferior pulmonary vein', ipv)):
    if not msk.any(): print('  no', id_); continue
    msk = biggest(msk)
    t = Tree(msk, AL, mm(roots[id_], AL))
    edt_v = ndimage.distance_transform_edt(msk, sampling=np.abs(np.diag(AL)[:3]))
    near = t.order[t.dist[t.order] <= 38.0]; far = t.order[t.dist[t.order] > 38.0]
    emit(id_, nm, 'veins', VEIN, owned(t, msk, near), AL, faces=7000, division=division_at(t, t.root, edt_v, into=16.0, look=26.0))
    emit(f'{"lul" if id_ == "pv-superior" else "lll"}-veins', f'{"Upper" if id_ == "pv-superior" else "Lower"} lobe segmental veins', 'lul-intra' if id_ == 'pv-superior' else 'lll-intra', VEIN,
         owned(t, msk, far), AL, faces=12000, opacity=0.85 if id_ == 'pv-superior' else 0.6, visible=id_ == 'pv-superior', label=False)

print('== airways')
u_ao = lobe_origins(air_t, air_terr, 'U', min_size=30); d_ao = lobe_origins(air_t, air_terr, 'D', min_size=30)
lul_b, lll_b = u_ao[0], d_ao[0]
path = air_t.path_to_root(lul_b)
lmb_nodes = np.array([p for p in path if air_t.dist[p] > air_t.dist[CARINA_NODE] - 1])
trachea_nodes = np.array([p for p in air_t.order if air_t.dist[p] <= air_t.dist[CARINA_NODE]])
near_u, far_u = split_hilar(air_t, lul_b, 22.0)
near_d, far_d = split_hilar(air_t, lll_b, 22.0)
emit('br-lul', 'Upper lobe bronchus', 'airway', BRONCH, owned(air_t, air, near_u), AL, faces=5000, division=division_at(air_t, lul_b, edt_air, into=6.0, look=12.0))
# lower lobe bronchus: staple just beyond the secondary carina, proximal to the superior segmental (B6) origin
emit('br-lll', 'Lower lobe bronchus', 'airway', BRONCH, owned(air_t, air, near_d), AL, faces=5000, division=division_at(air_t, lll_b, edt_air, into=5.0, look=11.0))
emit('br-left-main', 'Left main bronchus', 'airway', BRONCH, owned(air_t, air, lmb_nodes), AL, faces=5000)
emit('trachea', 'Trachea and carina', 'airway', BRONCH, owned(air_t, air, trachea_nodes), AL, faces=6000)
emit('lul-bronchi', 'Upper lobe segmental bronchi', 'lul-intra', BRONCH, owned(air_t, air, far_u), AL, faces=8000, opacity=0.8, label=False)
emit('lll-bronchi', 'Lower lobe segmental bronchi', 'lll-intra', BRONCH, owned(air_t, air, far_d), AL, faces=8000, opacity=0.6, visible=False, label=False)
SEC_CARINA = air_t.mm[air_t.pred[lul_b]]

print('== lobes, heart, great vessels')
LUNG = '#e9b2a6'
emit('lul', 'Left upper lobe', 'lungs', LUNG, ts('lung_upper_lobe_left'), AT, faces=20000, opacity=0.3, sigma=1.3)
emit('lll', 'Left lower lobe', 'lungs', '#d9a69a', ts('lung_lower_lobe_left'), AT, faces=20000, opacity=0.3, sigma=1.3)
for n_, nm in (('lung_upper_lobe_right', 'Right upper lobe'), ('lung_middle_lobe_right', 'Right middle lobe'), ('lung_lower_lobe_right', 'Right lower lobe')):
    emit('r' + n_.split('_')[1][0] + 'l', nm, 'lungs', LUNG, ts(n_), AT, faces=10000, opacity=0.18, visible=False, sigma=1.3, label=False)
Ut, Dt = ts('lung_upper_lobe_left'), ts('lung_lower_lobe_left')
fis = ndimage.binary_dilation(Ut, iterations=1) & ndimage.binary_dilation(Dt, iterations=1)
emit('fissure', 'Oblique fissure (left)', 'lungs', '#f4e3a1', fis, AT, faces=9000, opacity=0.55, sigma=0.8, label=False,
     note='Where the upper and lower lobe segmentations meet. Real fissures are often incomplete; this shows the plane, not its completeness.')
fis_mm = (np.argwhere(fis) @ AT[:3, :3].T + AT[:3, 3])
FIS_C = fis_mm.mean(0)
_u, _s, _vt = np.linalg.svd(fis_mm - FIS_C, full_matrices=False)
FIS_N = _vt[2] * np.sign(np.dot(_vt[2], (np.argwhere(Ut) @ AT[:3, :3].T).mean(0) - (np.argwhere(Dt) @ AT[:3, :3].T).mean(0)))
emit('heart', 'Heart and pericardium', 'mediastinum', '#9a5a52', ts('heart'), AT, faces=20000, sigma=1.2)
emit('laa', 'Left atrial appendage', 'mediastinum', '#b35c50', ts('atrial_appendage_left'), AT, faces=4000)
emit('aorta', 'Aorta', 'mediastinum', '#d0433a', ts('aorta'), AT, faces=18000, sigma=1.1)
for n_, id_, nm in (('brachiocephalic_trunk', 'bct', 'Brachiocephalic trunk'), ('common_carotid_artery_left', 'lcca', 'Left common carotid artery'),
                    ('subclavian_artery_left', 'lsca', 'Left subclavian artery')):
    emit(id_, nm, 'mediastinum', '#d0433a', ts(n_), AT, faces=4000)
emit('svc', 'Superior vena cava', 'mediastinum', SYSV, ts('superior_vena_cava'), AT, faces=5000)
emit('lbcv', 'Left brachiocephalic vein', 'mediastinum', SYSV, ts('brachiocephalic_vein_left'), AT, faces=4000)
emit('esophagus', 'Oesophagus', 'mediastinum', '#cf9459', ts('esophagus'), AT, faces=7000)
for v in [f'T{i}' for i in range(2, 11)]:
    emit(f'vert-{v.lower()}', f'{v} vertebra', 'chest-wall', '#e6dcc6', ts(f'vertebrae_{v}'), AT, faces=4000, label=False)
for i in range(1, 11):
    emit(f'rib-{i}-l', f'Left rib {i}', 'chest-wall', '#e6dcc6', ts(f'rib_left_{i}'), AT, faces=2500, opacity=0.9, visible=False, label=False)
emit('sternum', 'Sternum', 'chest-wall', '#e6dcc6', ts('sternum'), AT, faces=5000, visible=False, label=False)

# ------------------------------------------------------------------ schematic structures on landmarks
print('== schematic')


def tube(points, radius, seg=14) -> trimesh.Trimesh:
    P = np.array(points, float)
    # Catmull-Rom through the control points, ~1.5 mm steps
    pts = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)]
        n = max(2, int(np.linalg.norm(p2 - p1) / 1.5))
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    pts.append(P[-1]); pts = np.array(pts)
    tang = np.gradient(pts, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    ref = np.array([0, 0, 1.0]) if abs(tang[0][2]) < 0.9 else np.array([1.0, 0, 0])
    verts, faces = [], []
    nrm = np.cross(tang[0], ref); nrm /= np.linalg.norm(nrm)
    for i, (p, t) in enumerate(zip(pts, tang)):
        nrm = nrm - t * np.dot(nrm, t); nrm /= np.linalg.norm(nrm) + 1e-9
        b = np.cross(t, nrm)
        for k in range(seg):
            a = 2 * np.pi * k / seg
            verts.append(p + radius * (np.cos(a) * nrm + np.sin(a) * b))
    for i in range(len(pts) - 1):
        for k in range(seg):
            a, b = i * seg + k, i * seg + (k + 1) % seg
            faces += [[a, b + seg, b], [a, a + seg, b + seg]]
    m = trimesh.Trimesh(np.array(verts), np.array(faces), process=False)   # keep ring indexing for the caps
    for end, sign in ((0, -1), (len(pts) - 1, 1)):   # caps
        c = len(m.vertices); ring = np.arange(end * seg, end * seg + seg)
        m = trimesh.Trimesh(np.vstack([m.vertices, pts[end]]), np.vstack([m.faces, [[c, ring[k], ring[(k + 1) % seg]] if sign < 0 else [c, ring[(k + 1) % seg], ring[k]] for k in range(seg)]]), process=False)
    m = trimesh.Trimesh(m.vertices, m.faces, process=True)
    m.fix_normals()
    return m


def sphere(c, r) -> trimesh.Trimesh:
    s = trimesh.creation.icosphere(subdivisions=2, radius=r); s.apply_translation(c); return s


def vox_mm(mask, aff=AT):
    return mm(np.argwhere(mask), aff)


aorta_mm = vox_mm(ts('aorta')); heart_mm = vox_mm(ts('heart')); eso_mm = vox_mm(ts('esophagus'))
lsca_mm = vox_mm(ts('subclavian_artery_left')); lcca_mm = vox_mm(ts('common_carotid_artery_left')); tr_mm = vox_mm(ts('trachea'))
arch_top = aorta_mm[:, 2].max()
lpa_mm = mm(np.argwhere(owned(art_t, art, main_nodes)), AL)
lmb_mm = mm(np.argwhere(owned(air_t, air, lmb_nodes)), AL)
spv_div = next(s for s in structures if s['id'] == 'pv-superior')['division']['point'] if any(s['id'] == 'pv-superior' for s in structures) else [0, 0, 0]
hilum_z = spv_div[2] + CARINA[2]


def slab(pts, z, h=4.0):
    s = pts[np.abs(pts[:, 2] - z) < h]
    return s if len(s) else pts


def extreme(pts, axis, fn):
    return pts[fn(pts[:, axis])]


# ligamentum arteriosum: shortest link between the arch underside and the left PA
kd = cKDTree(aorta_mm[aorta_mm[:, 2] > arch_top - 45])
dists, idx = kd.query(lpa_mm)
p_lig = lpa_mm[np.argmin(dists)]; a_lig = kd.data[idx[np.argmin(dists)]]
LEFT = np.array([-1.0, 0, 0]); ANT = np.array([0, 1.0, 0]); SUP = np.array([0, 0, 1.0])

# phrenic: between subclavian artery and vein -> anterolateral arch -> lateral pericardium anterior to the hilum -> diaphragm
ph = [extreme(slab(lsca_mm, arch_top + 38), 1, np.argmax) + ANT * 6 + LEFT * 4,
      extreme(slab(aorta_mm, arch_top - 8), 0, np.argmin) + ANT * 10 + LEFT * 5]
for dz in (0, -35, -70):
    hs = slab(heart_mm, hilum_z + dz); hs = hs[hs[:, 1] > np.median(hs[:, 1])]
    ph.append(extreme(hs, 0, np.argmin) + LEFT * 3)
hb = heart_mm[heart_mm[:, 2] < heart_mm[:, 2].min() + 15]; hb = hb[hb[:, 1] > np.median(hb[:, 1])]
ph.append(extreme(hb, 0, np.argmin) + LEFT * 3 - SUP * 4)
emit_mesh('n-phrenic', 'Left phrenic nerve', 'nerves', '#f2d24b', tube([W(p) for p in ph], 1.4),
          note='Schematic, drawn on landmarks: crosses the arch anterolaterally and runs on the pericardium anterior to the hilum with the pericardiophrenic vessels.')
# vagus: between carotid and subclavian -> posterolateral arch (gives the recurrent nerve at its lower border) -> behind the hilum -> oesophagus
vag = [(slab(lcca_mm, arch_top + 38).mean(0) + slab(lsca_mm, arch_top + 38).mean(0)) / 2 - ANT * 4]
arch_s = slab(aorta_mm, arch_top - 10); post = arch_s[arch_s[:, 1] < np.median(arch_s[:, 1])]
vag.append(extreme(post, 0, np.argmin) + LEFT * 4)
rln_origin = a_lig + LEFT * 7 - SUP * 2
vag.append(rln_origin)
vag.append(lmb_mm.mean(0) - ANT * 13 + LEFT * 4)
for dz in (-45, -95):
    es = slab(eso_mm, hilum_z + dz); vag.append(extreme(es, 0, np.argmin) + LEFT * 2 + ANT * 3)
emit_mesh('n-vagus', 'Left vagus nerve', 'nerves', '#f2d24b', tube([W(p) for p in vag], 1.6),
          note='Schematic: crosses the arch posterior to the phrenic nerve, gives the recurrent laryngeal nerve at the arch, then passes behind the hilum onto the oesophagus.')
trs = slab(tr_mm, arch_top); groove = trs[np.argmin(trs[:, 0] - trs[:, 1])]
trs2 = slab(tr_mm, arch_top + 35); groove2 = trs2[np.argmin(trs2[:, 0] - trs2[:, 1])]
rln = [rln_origin, a_lig - SUP * 6 + np.array([6, -4, 0]), a_lig - SUP * 3 + np.array([14, -6, 4]), groove + LEFT * 2 - ANT * 3, groove2 + LEFT * 2 - ANT * 3]
emit_mesh('n-rln', 'Left recurrent laryngeal nerve', 'nerves', '#f2d24b', tube([W(p) for p in rln], 1.0),
          note='Schematic: hooks under the arch lateral to the ligamentum arteriosum and climbs in the tracheo-oesophageal groove. At risk clearing station 5.')
emit_mesh('lig-art', 'Ligamentum arteriosum', 'nerves', '#c9b79c', tube([W(p_lig), W((p_lig + a_lig) / 2 + LEFT * 2), W(a_lig)], 2.2),
          note='Schematic: from the top of the left PA to the underside of the arch. The recurrent laryngeal nerve hooks around it.')

nodes = {
    'ln-5': ('Station 5 (subaortic, AP window)', (a_lig + p_lig) / 2 + LEFT * 9, 6.5),
    'ln-6': ('Station 6 (para-aortic)', extreme(slab(aorta_mm, arch_top - 6), 1, np.argmax) + ANT * 8 + LEFT * 6, 6.0),
    'ln-7': ('Station 7 (subcarinal)', CARINA - SUP * 18 - ANT * 6, 8.0),
    'ln-10l': ('Station 10L (hilar)', lmb_mm[np.argmin(lmb_mm[:, 0])] + SUP * 9 - ANT * 2, 5.5),
    'ln-11l': ('Station 11L (interlobar)', SEC_CARINA + LEFT * 9 - SUP * 3, 5.0),
}
# inferior pulmonary ligament: the double pleural fold from the lower border of the inferior vein down the posterior
# mediastinal surface of the lower lobe to the diaphragm (schematic: drawn on the lobe's medial edge)
lll_mm = vox_mm(Dt)
pvi = np.array(next(s_['division']['point'] for s_ in structures if s_['id'] == 'pv-inferior')) + CARINA
edge, med = [], []
for k, z in enumerate(np.linspace(pvi[2] - 6, lll_mm[:, 2].min() + 10, 9)):
    sl = slab(lll_mm, z, 3.0)
    band = sl[(sl[:, 1] > pvi[1] - 32 - 2 * k) & (sl[:, 1] < pvi[1] + 6)]   # stays behind and below the vein, drifting back
    sl = band if len(band) > 20 else sl
    e = sl[np.argmax(sl[:, 0])] + LEFT * -1.0                                 # medial-most point (+x is medial on the left)
    if k == 0: e = pvi - SUP * 7 - LEFT * 1.0                                 # starts at the lower border of the vein
    edge.append(e)
edge = np.array(edge)
edge[1:-1] = (edge[:-2] + 2 * edge[1:-1] + edge[2:]) / 4          # smooth the lung edge so the fold hangs cleanly
edge = list(edge); med = []
for e in edge:                                                      # the fold attaches to the mediastinum along the oesophagus
    es = slab(eso_mm, e[2], 4.0); tgt = es.mean(0) if len(es) else e - LEFT * 12
    d = tgt - e; d[2] = 0; L_ = np.linalg.norm(d)
    med.append(e + d / (L_ + 1e-9) * min(L_ - 4.0, 16.0) if L_ > 6 else e - LEFT * 6)
V_ = np.array([W(p_) for p_ in edge + med]); n_ = len(edge)
F_ = [[i, i + 1, n_ + i] for i in range(n_ - 1)] + [[i + 1, n_ + i + 1, n_ + i] for i in range(n_ - 1)]
F_ += [f[::-1] for f in F_]
emit_mesh('ipl', 'Inferior pulmonary ligament', 'pleura', '#e8d9c9', trimesh.Trimesh(V_, np.array(F_), process=False),
          note='Schematic: the pleural fold below the inferior pulmonary vein, divided first in a lower lobectomy. Station 9 nodes lie in it.')
nodes['ln-9l'] = ('Station 9L (pulmonary ligament)', np.mean(edge[2:5], axis=0) - LEFT * 4, 4.5)
for id_, (nm, c, r) in nodes.items():
    emit_mesh(id_, nm, 'nodes', '#8fc79a', sphere(W(c), r), note='Schematic node station placed on landmarks (IASLC map).')

# ------------------------------------------------------------------ skin and ports
body = biggest(ndimage.binary_opening(CT > -400, iterations=2))
body_ds = body[::2, ::2, ::2]
a2 = AT.copy(); a2[:3, :3] *= 2
emit('skin', 'Chest wall (skin)', 'chest-wall', '#d9b8a4', body_ds, a2, faces=24000, opacity=0.1, visible=False, sigma=1.5, label=False)
skin_mm = mm(np.argwhere(body_ds & ~ndimage.binary_erosion(body_ds)), a2)
lung_c = vox_mm(Ut | Dt).mean(0)


def rib_z(n, az):
    r = vox_mm(ts(f'rib_left_{n}'))
    ang = np.degrees(np.arctan2(r[:, 1] - lung_c[1], -(r[:, 0] - lung_c[0])))
    s = r[np.abs(ang - az) < 8]
    return float(np.median(s[:, 2])) if len(s) else None


def port(ics, az):
    """skin point in the ics-th intercostal space (between ribs ics and ics+1) at azimuth az (0 = lateral, + = anterior)"""
    z1, z2 = rib_z(ics, az), rib_z(ics + 1, az)
    z = (z1 + z2) / 2 if z1 and z2 else lung_c[2]
    s = slab(skin_mm, z, 3.0); s = s[s[:, 0] < lung_c[0]]
    ang = np.degrees(np.arctan2(s[:, 1] - lung_c[1], -(s[:, 0] - lung_c[0])))
    return s[np.argmin(np.abs(ang - az))]


PORTS = {
    # standardized anterior approach (Hansen & Petersen 2012; McElnay et al 2014): utility incision anteriorly over the
    # hilum, a low anterior camera port at the level of the diaphragm, a working port at the same level further back
    'anterior': [('utility', 'Utility incision, anterior, over the hilum (4th space)', port(4, 45)),
                 ('camera', 'Camera port, low anterior (diaphragm level)', port(7, 40)),
                 ('posterior', 'Working port, same level, posterior', port(7, -35))],
    # a fissure-based layout: utility incision in line with the fissure, camera and working ports posterior
    'posterior': [('utility', 'Utility incision, in line with the fissure (5th space)', port(5, 10)),
                  ('camera', 'Camera port, low, posterior axillary', port(7, -25)),
                  ('posterior', 'Working port, posterior', port(6, -65))],
}
for appr, ps in PORTS.items():
    for k, nm, p in ps:
        emit_mesh(f'port-{appr}-{k}', nm, f'ports-{appr}', '#46c2c7', sphere(W(p), 5.5 if k == 'utility' else 3.5), visible=False)

# ------------------------------------------------------------------ CT and label volumes on a 1.25 mm grid, cropped to the thorax
vox = 1.0
thorax = ts('lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right', 'heart', 'trachea')
tm = vox_mm(thorax)
lo = np.maximum(tm.min(0) - [25, 25, 20], vox_mm(body).min(0)); hi = np.minimum(tm.max(0) + [25, 25, 25], vox_mm(body).max(0))
shape = (np.floor((hi - lo) / vox) + 1).astype(int)
g_aff = np.diag([vox, vox, vox, 1.0]); g_aff[:3, 3] = lo
m = np.linalg.inv(AT) @ g_aff
hu = ndimage.affine_transform(CT.astype(np.float32), m[:3, :3], offset=m[:3, 3], output_shape=tuple(shape), order=1, cval=-1024)
HU0, STEP = -1024, 8
ct8 = np.clip(np.round((hu - HU0) / STEP), 0, 255).astype(np.uint8)
lab8 = np.zeros(shape, np.uint8); lut = {}
for i, (id_, (msk, aff)) in enumerate(labels_out.items(), start=1):
    r = pull(msk, aff, shape, g_aff)
    lab8[r & (lab8 == 0) if id_ in ('lul', 'lll', 'heart') else r] = i
    lut[str(i)] = id_


def write(arr, name):
    raw = np.ascontiguousarray(arr.transpose(2, 1, 0)).tobytes()   # x fastest
    (OUT / name).write_bytes(gzip.compress(raw, 9))
    return (OUT / name).stat().st_size


print('ct', shape, write(ct8, 'ct.hu8.gz') / 1e6, 'MB; labels', write(lab8, 'labels.u8.gz') / 1e6, 'MB')
w_aff = shifted(g_aff)
landmarks = {'carina': [0.0, 0.0, 0.0], 'secondary-carina': [round(float(x), 1) for x in W(SEC_CARINA)],
             'ap-window': [round(float(x), 1) for x in W((a_lig + p_lig) / 2)], 'lung-centre': [round(float(x), 1) for x in W(lung_c)],
             'fissure-centre': [round(float(x), 1) for x in W(FIS_C)], 'fissure-normal': [round(float(x), 3) for x in FIS_N]}
for appr, ps in PORTS.items():
    for k, _, p in ps: landmarks[f'port-{appr}-{k}'] = [round(float(x), 1) for x in W(p)]
atlas = {
    'ct': {'file': 'ct.hu8.gz', 'dims': [int(x) for x in shape], 'affine': [[round(float(x), 4) for x in row] for row in w_aff[:3]], 'scale': STEP, 'offset': HU0, 'spacing': vox},
    'labels': {'file': 'labels.u8.gz', 'lut': lut},
    'groups': [{'id': 'lungs', 'name': 'Lungs and fissure', 'open': True}, {'id': 'arteries', 'name': 'Pulmonary arteries', 'open': True},
               {'id': 'veins', 'name': 'Pulmonary veins', 'open': True}, {'id': 'airway', 'name': 'Airway', 'open': True},
               {'id': 'nerves', 'name': 'Nerves (schematic)', 'open': True}, {'id': 'pleura', 'name': 'Pleura and ligament (schematic)'}, {'id': 'nodes', 'name': 'Lymph node stations (schematic)'},
               {'id': 'mediastinum', 'name': 'Heart, great vessels, oesophagus'}, {'id': 'lul-intra', 'name': 'Upper lobe, intrapulmonary'},
               {'id': 'lll-intra', 'name': 'Lower lobe, intrapulmonary'}, {'id': 'chest-wall', 'name': 'Chest wall and spine'},
               {'id': 'ports-anterior', 'name': 'Ports, anterior approach'}, {'id': 'ports-posterior', 'name': 'Ports, posterior approach'}],
    'structures': structures,
    'landmarks': landmarks,
    'source': {'name': 'Reference CT: 3D Slicer sample CTA (CTA-cardio)', 'licence': 'unstated',
               'note': 'Segmented with TotalSegmentator (total and lung_vessels). Branch names are assigned from the geometry of this one scan and need a surgeon\'s check.'},
}
(OUT / 'atlas.json').write_text(json.dumps(atlas))
print('structures', len(structures))
print('names', {v[0]: round(float(art_t.dist[k]), 1) for k, v in names.items()})
