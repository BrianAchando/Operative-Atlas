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
_present = set(np.unique(T).tolist())
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


def div_before(t: Tree, node: int, back: float, edt: np.ndarray) -> dict:
    """staple line on the trunk `back` mm proximal to `node`, pointing proximal -> distal"""
    path = t.path_to_root(node); p = path[-1]
    for q in path:
        if t.dist[q] <= t.dist[node] - back: p = q; break
    d = t.mm[node] - t.mm[p]; d = d / (np.linalg.norm(d) + 1e-9)
    return {'point': [round(float(x), 1) for x in W(t.mm[p])], 'dir': [round(float(x), 3) for x in d], 'radius': round(max(float(edt[tuple(t.vox[p])]), 3.0), 1)}


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
trunc_o = next((o for o, v in names.items() if v[0] == 'pa-truncus-anterior'), None)
emit('pa-left', 'Left pulmonary artery', 'arteries', ART, owned(art_t, art, main_nodes), AL, faces=9000, division=div_before(art_t, trunc_o, 12.0, edt_art) if trunc_o is not None else None)
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
_lmb_div = None
if len(lmb_nodes):
    _q = min(lmb_nodes, key=lambda q: abs(air_t.dist[q] - (air_t.dist[CARINA_NODE] + 9.0)))
    _dd = air_t.mm[lul_b] - air_t.mm[_q]; _dd /= np.linalg.norm(_dd) + 1e-9
    _lmb_div = {'point': [round(float(x), 1) for x in W(air_t.mm[_q])], 'dir': [round(float(x), 3) for x in _dd], 'radius': round(max(float(edt_air[tuple(air_t.vox[_q])]), 4.0), 1)}
emit('br-left-main', 'Left main bronchus', 'airway', BRONCH, owned(air_t, air, lmb_nodes), AL, faces=5000, division=_lmb_div)
emit('trachea', 'Trachea and carina', 'airway', BRONCH, owned(air_t, air, trachea_nodes), AL, faces=6000)
emit('lul-bronchi', 'Upper lobe segmental bronchi', 'lul-intra', BRONCH, owned(air_t, air, far_u), AL, faces=8000, opacity=0.8, label=False)
emit('lll-bronchi', 'Lower lobe segmental bronchi', 'lll-intra', BRONCH, owned(air_t, air, far_d), AL, faces=8000, opacity=0.6, visible=False, label=False)
SEC_CARINA = air_t.mm[air_t.pred[lul_b]]

Ut = ts('lung_upper_lobe_left')

print('== left segments')
SEGC = {'seg-lul-upper': '#d9b36a', 'seg-lingula': '#7fb38f', 'seg-s6': '#b58fc9', 'seg-lll-basal': '#8fa7c9'}
sizes = air_t.subtree_sizes()


def first_split(t, start, min_size=25):
    """walk down the largest branch from `start` to the first node with two substantial children"""
    cur = start
    while True:
        ch = sorted(t.children.get(cur, []), key=lambda c: -sizes[c])
        if len(ch) >= 2 and sizes[ch[1]] >= min_size: return cur, ch
        if not ch: return cur, []
        cur = ch[0]


cen = lambda n: air_t.mm[air_t.subtree(n)].mean(0)
sp_u, ch_u = first_split(air_t, lul_b)
b_ling, b_updiv = (ch_u[0], ch_u[1]) if cen(ch_u[0])[2] < cen(ch_u[1])[2] else (ch_u[1], ch_u[0])
# B6: walking down the lower lobe bronchus, the first sizeable branch that heads up and back
b6, cur = None, lll_b
for _ in range(80):
    ch = sorted(air_t.children.get(cur, []), key=lambda c: -sizes[c])
    if not ch: break
    for c in ch[1:]:
        v_ = cen(c) - air_t.mm[cur]
        if sizes[c] >= 15 and v_[2] > -3 and v_[1] < 0: b6 = c; break
    if b6 is not None: break
    cur = ch[0]
b_basal = cur if b6 is not None else lll_b
print('  bronchi: lingula', b_ling, 'upper division', b_updiv, 'B6', b6)


def territories(lobe_mask, parts):
    """label each lobe voxel (TS grid) by the nearest airway node of each part's subtree"""
    nodes, labs = [], []
    for k, n in enumerate(parts): sub = air_t.subtree(n); nodes.append(sub); labs.append(np.full(len(sub), k))
    kd = cKDTree(air_t.mm[np.concatenate(nodes)]); lab = np.concatenate(labs)
    vox = np.argwhere(lobe_mask); _, nn = kd.query(mm(vox, AT))
    out = [np.zeros(lobe_mask.shape, bool) for _ in parts]
    for k in range(len(parts)): out[k][tuple(vox[lab[nn] == k].T)] = True
    return [biggest(ndimage.binary_opening(o, iterations=1)) for o in out]


S_UPDIV, S_LING = territories(Ut if 'Ut' in dir() else ts('lung_upper_lobe_left'), [b_updiv, b_ling])
SEG = {'seg-lul-upper': ('Upper division (S1+2, S3)', S_UPDIV), 'seg-lingula': ('Lingula (S4, S5)', S_LING)}
if b6 is not None:
    S_S6, S_BAS = territories(ts('lung_lower_lobe_left'), [b6, b_basal])
    SEG.update({'seg-s6': ('Superior segment (S6)', S_S6), 'seg-lll-basal': ('Basal segments (S7-S10)', S_BAS)})
for id_, (nm, m_) in SEG.items():
    emit(id_, nm, 'segments', SEGC[id_], m_, AT, faces=12000, opacity=0.35, visible=False, sigma=1.3)
d1l = lambda m: ndimage.binary_dilation(m, iterations=1)
ISP = {'isp-lingula': ('Intersegmental plane: lingula / upper division', d1l(S_LING) & d1l(S_UPDIV), S_UPDIV, S_LING)}
if b6 is not None: ISP['isp-s6'] = ('Intersegmental plane: S6 / basal', d1l(S_S6) & d1l(S_BAS), S_S6, S_BAS)
ISP_LM = {}
for id_, (nm, m_, pos, neg) in ISP.items():
    emit(id_, nm, 'segments', '#f4e3a1', m_, AT, faces=6000, opacity=0.5, visible=False, sigma=0.8, label=False)
    P_ = mm(np.argwhere(m_), AT); c_ = P_.mean(0); _, _, vt_ = np.linalg.svd(P_ - c_, full_matrices=False)
    n_ = vt_[2] * np.sign(np.dot(vt_[2], mm(np.argwhere(pos), AT).mean(0) - mm(np.argwhere(neg), AT).mean(0)))
    ISP_LM[id_] = (c_, n_, vt_[0])

# segmental bronchi
for id_, nm, n in (('br-lingular', 'Lingular bronchus (B4+5)', b_ling), ('br-upper-div', 'Upper division bronchus (B1+2, B3)', b_updiv), ('br-b6', 'Superior segmental bronchus (B6)', b6)):
    if n is None: continue
    emit(id_, nm, 'airway', BRONCH, owned(air_t, air, split_hilar(air_t, n, 12.0)[0]), AL, faces=3000, division=division_at(air_t, n, edt_air, into=3.0, look=8.0))

# segmental veins: the lingular vein and V6, by the segment territories they drain
segLV = {k: pull(v[1], AT, LV.shape, AL) for k, v in SEG.items()}
for vid, root_key, mine, other, nm in (('pv-lingular', 'pv-superior', 'seg-lingula', 'seg-lul-upper', 'Lingular vein (V4+5)'),
                                       ('pv-upper-div', 'pv-superior', 'seg-lul-upper', 'seg-lingula', 'Upper division veins (V1+2, V3)'),
                                       ('pv-v6', 'pv-inferior', 'seg-s6', 'seg-lll-basal', 'Superior segmental vein (V6)')):
    if mine not in segLV: continue
    vm = biggest(spv if root_key == 'pv-superior' else ipv)
    tv = Tree(vm, AL, mm(roots[root_key], AL)); tt = tv.territory({'A': segLV[mine], 'B': segLV[other]})
    fA = tt['A'] / (tt['A'] + tt['B'] + 1e-9)
    cand = [i for i in tv.order if tv.pred[i] >= 0 and fA[i] >= 0.85 and fA[tv.pred[i]] < 0.85 and tt['_size'][i] >= 20]
    if not cand: print('  no', vid); continue
    o = max(cand, key=lambda i: tt['_size'][i])
    edt_sv = ndimage.distance_transform_edt(vm, sampling=np.abs(np.diag(AL)[:3]))
    emit(vid, nm, 'veins', VEIN, owned(tv, vm, split_hilar(tv, o, 14.0)[0]), AL, faces=3000, division=division_at(tv, o, edt_sv, into=4.0, look=10.0))

print('== lobes, heart, great vessels')
LUNG = '#e9b2a6'
emit('lul', 'Left upper lobe', 'lungs', LUNG, ts('lung_upper_lobe_left'), AT, faces=20000, opacity=0.3, sigma=1.3)
emit('lll', 'Left lower lobe', 'lungs', '#d9a69a', ts('lung_lower_lobe_left'), AT, faces=20000, opacity=0.3, sigma=1.3)
emit('rul', 'Right upper lobe', 'lungs', LUNG, ts('lung_upper_lobe_right'), AT, faces=16000, opacity=0.3, sigma=1.3)
emit('rml', 'Right middle lobe', 'lungs', '#e2ab9e', ts('lung_middle_lobe_right'), AT, faces=10000, opacity=0.3, sigma=1.3)
emit('rll', 'Right lower lobe', 'lungs', '#d9a69a', ts('lung_lower_lobe_right'), AT, faces=16000, opacity=0.3, sigma=1.3)
RU_t, RM_t, RD_t = ts('lung_upper_lobe_right'), ts('lung_middle_lobe_right'), ts('lung_lower_lobe_right')
d1 = lambda m: ndimage.binary_dilation(m, iterations=1)
emit('fissure-h', 'Horizontal fissure (right)', 'lungs', '#f4e3a1', d1(RU_t) & d1(RM_t), AT, faces=6000, opacity=0.55, sigma=0.8, label=False,
     note='Between the right upper and middle lobes; often incomplete.')
emit('fissure-r', 'Oblique fissure (right)', 'lungs', '#f4e3a1', d1(RU_t | RM_t) & d1(RD_t), AT, faces=9000, opacity=0.55, sigma=0.8, label=False,
     note='Between the lower lobe and the upper and middle lobes.')
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
    emit(f'rib-{i}-r', f'Right rib {i}', 'chest-wall', '#e6dcc6', ts(f'rib_right_{i}'), AT, faces=2500, opacity=0.9, visible=False, label=False)
emit('sternum', 'Sternum', 'chest-wall', '#e6dcc6', ts('sternum'), AT, faces=5000, visible=False, label=False)


# ------------------------------------------------------------------ right hilum (TotalSegmentator lung_vessels on the right hemithorax)
RIGHT_IDS: set[str] = set()
if (WORK / 'vessels_right.nii.gz').exists():
    print('== right hilum')
    n0 = len(structures)
    rv_img = nib.load(WORK / 'vessels_right.nii.gz'); RV = np.asanyarray(rv_img.dataobj); AR = rv_img.affine
    rU, rM, rD = (pull(m, AT, RV.shape, AR) for m in (RU_t, RM_t, RD_t))
    lobes_r = {'U': rU, 'M': rM, 'D': rD}
    r_air = biggest((RV == 1) | (RV == 2)); r_art = biggest(RV == 3); r_vein_all = RV == 4
    edt_rair = ndimage.distance_transform_edt(r_air, sampling=np.abs(np.diag(AR)[:3]))
    edt_rart = ndimage.distance_transform_edt(r_art, sampling=np.abs(np.diag(AR)[:3]))

    # ---- airway: right main bronchus -> upper lobe bronchus, bronchus intermedius -> middle and lower lobe bronchi
    # root in the trachea just above the carina (the top of this crop can be an apical bronchus, not the trachea)
    ra = Tree(r_air, AR, CARINA + np.array([0, 0, 25.0]))
    ra_terr = ra.territory(lobes_r)
    rul_b = sorted(lobe_origins(ra, ra_terr, 'U', min_size=30), key=lambda i: -ra_terr['_size'][i])[0]
    rml_b = sorted(lobe_origins(ra, ra_terr, 'M', min_size=20), key=lambda i: -ra_terr['_size'][i])[0]
    rll_b = sorted(lobe_origins(ra, ra_terr, 'D', min_size=30), key=lambda i: -ra_terr['_size'][i])[0]
    up_path = ra.path_to_root(rul_b)
    # the main bronchus: from the carina (nearest node to the left tree's carina) to the upper lobe take-off
    car_r = int(cKDTree(ra.mm).query(CARINA)[1])
    rmb_nodes = np.array([q for q in up_path[1:] if ra.dist[q] >= ra.dist[car_r] - 1])
    bi_nodes = np.array([q for q in ra.path_to_root(rml_b)[1:] if ra.dist[q] > ra.dist[up_path[1]]])
    nu, fu = split_hilar(ra, rul_b, 20.0)
    emit('br-rul', 'Right upper lobe bronchus', 'airway', BRONCH, owned(ra, r_air, nu), AR, faces=5000, division=division_at(ra, rul_b, edt_rair, into=5.0, look=11.0))
    emit('br-intermedius', 'Bronchus intermedius', 'airway', BRONCH, owned(ra, r_air, bi_nodes), AR, faces=5000)
    emit('br-rml', 'Middle lobe bronchus', 'airway', BRONCH, owned(ra, r_air, split_hilar(ra, rml_b, 14.0)[0]), AR, faces=4000, division=division_at(ra, rml_b, edt_rair, into=4.0, look=9.0))
    emit('br-rll', 'Right lower lobe bronchus', 'airway', BRONCH, owned(ra, r_air, split_hilar(ra, rll_b, 18.0)[0]), AR, faces=5000, division=division_at(ra, rll_b, edt_rair, into=4.0, look=10.0))
    _rq = min(rmb_nodes, key=lambda q: abs(ra.dist[q] - (ra.dist[car_r] + 7.0))) if len(rmb_nodes) else None
    _rdiv = None
    if _rq is not None:
        _dd = ra.mm[rul_b] - ra.mm[_rq]; _dd /= np.linalg.norm(_dd) + 1e-9
        _rdiv = {'point': [round(float(x), 1) for x in W(ra.mm[_rq])], 'dir': [round(float(x), 3) for x in _dd], 'radius': round(max(float(edt_rair[tuple(ra.vox[_rq])]), 4.0), 1)}
    emit('br-right-main', 'Right main bronchus', 'airway', BRONCH, owned(ra, r_air, rmb_nodes), AR, faces=5000, division=_rdiv)
    emit('rul-bronchi', 'Right upper lobe segmental bronchi', 'rul-intra', BRONCH, owned(ra, r_air, fu), AR, faces=8000, opacity=0.8, label=False)
    R_SEC = ra.mm[up_path[1]]                         # upper lobe take-off (the "secondary carina" of the right)
    R_RMB = ra.mm[rmb_nodes].mean(0) if len(rmb_nodes) else CARINA

    # ---- arteries: right PA -> truncus anterior (truncus superior), ascending posterior (A2), middle lobe, A6, basal trunk
    rav = np.argwhere(r_art)
    rt = Tree(r_art, AR, mm(rav[np.argmin(rav[:, 0])], AR))          # root at the most medial point: the right PA origin
    rt_terr = rt.territory(lobes_r)
    ru_or = sorted(lobe_origins(rt, rt_terr, 'U', min_size=40), key=lambda i: rt.dist[i])
    rm_or = sorted(lobe_origins(rt, rt_terr, 'M', min_size=25), key=lambda i: rt.dist[i])
    rd_or = lobe_origins(rt, rt_terr, 'D', min_size=40)
    lim = min([rt.dist[o] for o in rm_or + list(rd_or)], default=np.inf) + 12.0
    ru_or = [o for o in ru_or if rt.dist[o] <= lim]      # an 'upper lobe origin' distal to the basal trunk is a lobe-boundary artefact
    rnames: dict[int, tuple[str, str]] = {}
    if ru_or:
        rnames[ru_or[0]] = ('rpa-truncus', 'Truncus anterior (right; A1 + A3)')
        # beyond the truncus: branches heading back are ascending posterior (A2), heading forward ascending anterior (A3)
        na = npo = 0
        for o in ru_or[1:]:
            v_ = rt.mm[rt.subtree(o)].mean(0) - rt.mm[o]
            if v_[1] > 0.35 * np.linalg.norm(v_): na += 1; rnames[o] = (f'rpa-a3-{na}', 'Ascending anterior artery (A3)')
            else: npo += 1; rnames[o] = (f'rpa-a2-{npo}', 'Ascending posterior artery (A2)')
    for k, o in enumerate(rm_or[:2]):
        rnames[o] = (f'rpa-ml-{k + 1}', 'Middle lobe artery (A4+5)' if len(rm_or) == 1 else f'Middle lobe artery {k + 1}')
    rd_info = [(o, rt.mm[rt.subtree(o)].mean(0)) for o in rd_or]
    if rd_info:
        rbas = max(rd_info, key=lambda x: rt_terr['_size'][x[0]]); rnames[rbas[0]] = ('rpa-basal', 'Basal trunk (right lower lobe)')
        rsup = [x for x in rd_info if x[0] != rbas[0]]
        if rsup: rnames[max(rsup, key=lambda x: x[1][2])[0]] = ('rpa-a6', 'Superior segmental artery (A6, right)')
    r_intra, r_branch = [], []
    for o, (id_, nm) in sorted(rnames.items(), key=lambda kv: rt.dist[kv[0]]):
        near, far = split_hilar(rt, o, 30.0); r_branch.append(rt.subtree(o))
        if id_.startswith(('rpa-truncus', 'rpa-a2', 'rpa-a3')): r_intra.append(far)
        emit(id_, nm, 'arteries', ART, owned(rt, r_art, near), AR, faces=5000, division=division_at(rt, o, edt_rart))
    rmain = np.setdiff1d(np.flatnonzero(rt.keep), np.concatenate(r_branch) if r_branch else np.array([], int))
    rtr_o = next((o for o, v in rnames.items() if v[0] == 'rpa-truncus'), None)
    emit('rpa', 'Right pulmonary artery', 'arteries', ART, owned(rt, r_art, rmain), AR, faces=9000, division=div_before(rt, rtr_o, 10.0, edt_rart) if rtr_o is not None else None)
    if r_intra: emit('rul-arteries', 'Right upper lobe segmental arteries', 'rul-intra', ART, owned(rt, r_art, np.concatenate(r_intra)), AR, faces=12000, opacity=0.85, label=False)
    print('  right artery names', {v[0]: round(float(rt.dist[k]), 1) for k, v in rnames.items()})

    # ---- veins: superior vein (upper + middle lobe) and inferior vein, split at their atrial ostia; in the superior
    #      vein the upper lobe trunk and the middle lobe vein are separated by territory
    rheart = pull(ts('heart'), AT, RV.shape, AR); rla = ndimage.binary_dilation(rheart, iterations=2)
    rvein = biggest(r_vein_all & ~rla)
    # one tree from the atrium; the two veins often share the atrial contact, so they are split by what they drain
    hdr = ndimage.distance_transform_edt(~rla)
    vv_ = np.argwhere(rvein); vroot_ = vv_[np.argmin(hdr[tuple(vv_.T)])]
    rvt = Tree(rvein, AR, mm(vroot_, AR)); vte = rvt.territory(lobes_r)
    tot_ = vte['U'] + vte['M'] + vte['D'] + 1e-9
    fr = {k: vte[k] / tot_ for k in 'UMD'}
    first = lambda k, frac, size, pool=None: min((i for i in (rvt.order if pool is None else pool) if rvt.pred[i] >= 0 and fr[k][i] >= frac and fr[k][rvt.pred[i]] < frac and vte['_size'][i] >= size),
                                                   key=lambda i: rvt.dist[i], default=None)
    ipv_o = first('D', 0.9, 150)
    ipv_nodes = rvt.subtree(ipv_o) if ipv_o is not None else np.array([], int)
    spv_pool = np.setdiff1d(rvt.order, ipv_nodes)
    u_o = max((i for i in spv_pool if rvt.pred[i] >= 0 and fr['U'][i] >= 0.9 and fr['U'][rvt.pred[i]] < 0.9 and vte['_size'][i] >= 40), key=lambda i: vte['_size'][i], default=None)
    m_o = max((i for i in spv_pool if rvt.pred[i] >= 0 and fr['M'][i] >= 0.9 and fr['M'][rvt.pred[i]] < 0.9 and vte['_size'][i] >= 20), key=lambda i: vte['_size'][i], default=None)
    edt_v = ndimage.distance_transform_edt(rvein, sampling=np.abs(np.diag(AR)[:3]))
    taken = [ipv_nodes]
    if u_o is not None:
        nr, fr_ = split_hilar(rvt, u_o, 25.0); taken.append(rvt.subtree(u_o))
        emit('rpv-rul', 'Upper lobe veins (superior vein, upper trunk)', 'veins', VEIN, owned(rvt, rvein, nr), AR, faces=5000, division=division_at(rvt, u_o, edt_v, into=6.0, look=14.0))
        emit('rul-veins', 'Right upper lobe segmental veins', 'rul-intra', VEIN, owned(rvt, rvein, fr_), AR, faces=10000, opacity=0.85, label=False)
    if m_o is not None:
        taken.append(rvt.subtree(m_o))
        emit('rpv-ml', 'Middle lobe vein', 'veins', VEIN, owned(rvt, rvein, split_hilar(rvt, m_o, 20.0)[0]), AR, faces=4000, division=division_at(rvt, m_o, edt_v, into=5.0, look=12.0))
    trunk = np.setdiff1d(spv_pool[rvt.dist[spv_pool] <= 40.0], np.concatenate(taken))
    emit('rpv-superior', 'Right superior pulmonary vein', 'veins', VEIN, owned(rvt, rvein, trunk), AR, faces=5000,
         division=div_before(rvt, u_o, 8.0, edt_v) if u_o is not None else None)
    if ipv_o is not None:
        emit('rpv-inferior', 'Right inferior pulmonary vein', 'veins', VEIN, owned(rvt, rvein, ipv_nodes[rvt.dist[ipv_nodes] - rvt.dist[ipv_o] <= 30.0]), AR, faces=6000,
             division=division_at(rvt, ipv_o, edt_v, into=6.0, look=16.0))
    print('  right veins: upper', u_o is not None, 'middle', m_o is not None, 'inferior', ipv_o is not None)
    RIGHT_IDS |= {q['id'] for q in structures[n0:]}

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
def lig_sheet(lobe_mm, pvi, med):
    """edge points and a triangulated sheet for the inferior pulmonary ligament; med = +1 if medial is +x (left lung), -1 (right)"""
    edge = []
    for k, z in enumerate(np.linspace(pvi[2] - 6, lobe_mm[:, 2].min() + 10, 9)):
        sl = slab(lobe_mm, z, 3.0)
        band = sl[(sl[:, 1] > pvi[1] - 32 - 2 * k) & (sl[:, 1] < pvi[1] + 6)]   # behind and below the vein, drifting back
        sl = band if len(band) > 20 else sl
        e = sl[np.argmax(med * sl[:, 0])] + np.array([med * 1.0, 0, 0])      # medial-most point of the lobe
        if k == 0: e = pvi - SUP * 7 + np.array([med * 1.0, 0, 0])            # starts at the lower border of the vein
        edge.append(e)
    edge = np.array(edge)
    edge[1:-1] = (edge[:-2] + 2 * edge[1:-1] + edge[2:]) / 4                  # smooth the lung edge so the fold hangs cleanly
    edge = list(edge); medp = []
    for e in edge:                                                          # attaches to the mediastinum along the oesophagus
        es = slab(eso_mm, e[2], 4.0); tgt = es.mean(0) if len(es) else e + np.array([med * 12.0, 0, 0])
        d = tgt - e; d[2] = 0; L_ = np.linalg.norm(d)
        medp.append(e + d / (L_ + 1e-9) * min(L_ - 4.0, 16.0) if L_ > 6 else e + np.array([med * 6.0, 0, 0]))
    V_ = np.array([W(p_) for p_ in edge + medp]); n_ = len(edge)
    F_ = [[i, i + 1, n_ + i] for i in range(n_ - 1)] + [[i + 1, n_ + i + 1, n_ + i] for i in range(n_ - 1)]
    F_ += [f[::-1] for f in F_]
    return edge, trimesh.Trimesh(V_, np.array(F_), process=False)


pvi = np.array(next(s_['division']['point'] for s_ in structures if s_['id'] == 'pv-inferior')) + CARINA
edge, lig_mesh = lig_sheet(vox_mm(Dt), pvi, +1)
emit_mesh('ipl', 'Inferior pulmonary ligament', 'pleura', '#e8d9c9', lig_mesh,
          note='Schematic: the pleural fold below the inferior pulmonary vein, divided first in a lower lobectomy. Station 9 nodes lie in it.')
nodes['ln-9l'] = ('Station 9L (pulmonary ligament)', np.mean(edge[2:5], axis=0) - LEFT * 4, 4.5)
for id_, (nm, c, r) in nodes.items():
    emit_mesh(id_, nm, 'nodes', '#8fc79a', sphere(W(c), r), note='Schematic node station placed on landmarks (IASLC map).')

# ---- right side, schematic on landmarks
RIGHT = -LEFT
if RIGHT_IDS:
    n0 = len(structures)
    svc_mm = vox_mm(ts('superior_vena_cava'))
    def ra_point(v, off):
        pts = vox_mm(ts(f'vertebrae_{v}')); return pts[np.argmax(pts[:, 0] + pts[:, 1])] + off
    arch_z = R_RMB[2] + 14
    svc_s = slab(svc_mm, arch_z, 3.0)
    az = [ra_point(v, RIGHT * 3 + ANT * 3) for v in ('T12', 'T10') if TS.get(f'vertebrae_{v}') in _present] + \
         [ra_point('T8', RIGHT * 3 + ANT * 3), ra_point('T6', RIGHT * 3 + ANT * 3), ra_point('T5', RIGHT * 4 + ANT * 2 + SUP * 4),
          R_RMB + RIGHT * 9 - ANT * 10 + SUP * 14, R_RMB + RIGHT * 10 + SUP * 16, svc_s[np.argmin(svc_s[:, 1])] - ANT * 1]
    emit_mesh('azygos', 'Azygos vein (arch)', 'mediastinum', SYSV, tube([W(p_) for p_ in az], 4.5),
              note='Schematic: ascends on the right of the vertebral bodies and arches forward over the right main bronchus into the back of the SVC.')
    AZ_ARCH = az[-3]; _d = az[-2] - az[-3]; _d /= np.linalg.norm(_d)
    structures[-1]['division'] = {'point': [round(float(x), 1) for x in W(AZ_ARCH + (az[-2] - az[-3]) * 0.4)], 'dir': [round(float(x), 3) for x in _d], 'radius': 5.0}
    rph = []
    svc_top = svc_mm[:, 2].max()
    for z in (svc_top - 10, arch_z, R_RMB[2] - 10):
        ss = slab(svc_mm, z, 3.0); rph.append(ss[np.argmax(ss[:, 0])] + RIGHT * 3 + ANT * 2)
    for dz in (-35, -70):
        hs = slab(heart_mm, hilum_z + dz); hs = hs[hs[:, 1] > np.median(hs[:, 1])]; rph.append(hs[np.argmax(hs[:, 0])] + RIGHT * 3)
    emit_mesh('n-phrenic-r', 'Right phrenic nerve', 'nerves', '#f2d24b', tube([W(p_) for p_ in rph], 1.4),
              note='Schematic: on the lateral surface of the SVC, then the pericardium anterior to the hilum.')
    trs_ = slab(tr_mm, arch_z + 30); t1 = trs_[np.argmax(trs_[:, 0] - trs_[:, 1])]
    trs_ = slab(tr_mm, arch_z + 5); t2 = trs_[np.argmax(trs_[:, 0] - trs_[:, 1])]
    rvag = [t1 + RIGHT * 3 - ANT * 3, t2 + RIGHT * 3 - ANT * 4, R_RMB - ANT * 13 + RIGHT * 4]
    for dz in (-45, -95):
        es = slab(eso_mm, hilum_z + dz); rvag.append(es[np.argmax(es[:, 0])] + RIGHT * 2 - ANT * 3)
    emit_mesh('n-vagus-r', 'Right vagus nerve', 'nerves', '#f2d24b', tube([W(p_) for p_ in rvag], 1.6),
              note='Schematic: descends on the right of the trachea, passes medial to the azygos arch and behind the hilum onto the oesophagus.')
    trs_ = slab(tr_mm, arch_z + 12); svc_s2 = slab(svc_mm, arch_z + 12, 3.0)
    rnodes = {'ln-4r': ('Station 4R (lower paratracheal)', (trs_[np.argmax(trs_[:, 0])] + svc_s2[np.argmin(svc_s2[:, 0])]) / 2, 6.5),
              'ln-10r': ('Station 10R (hilar)', R_SEC + SUP * 9 + RIGHT * 3, 5.5),
              'ln-11r': ('Station 11R (interlobar)', R_SEC + RIGHT * 8 - SUP * 8, 5.0)}
    rpvi_s = next((s_ for s_ in structures if s_['id'] == 'rpv-inferior'), None)
    if rpvi_s:
        redge, rlig = lig_sheet(vox_mm(RD_t), np.array(rpvi_s['division']['point']) + CARINA, -1)
        emit_mesh('ipl-r', 'Inferior pulmonary ligament (right)', 'pleura', '#e8d9c9', rlig,
                  note='Schematic: the pleural fold below the right inferior pulmonary vein, divided first in a lower lobectomy. Station 9R nodes lie in it.')
        rnodes['ln-9r'] = ('Station 9R (pulmonary ligament)', np.mean(redge[2:5], axis=0) - RIGHT * 4, 4.5)
    for id_, (nm, c, r) in rnodes.items():
        emit_mesh(id_, nm, 'nodes', '#8fc79a', sphere(W(c), r), note='Schematic node station placed on landmarks (IASLC map).')
    RIGHT_IDS |= {q['id'] for q in structures[n0:]}

# ------------------------------------------------------------------ skin and ports
body = biggest(ndimage.binary_opening(CT > -400, iterations=2))
body_ds = body[::2, ::2, ::2]
# the skin is the outer surface: fill the lungs slice by slice so their surfaces are not mistaken for skin
body_fill = np.stack([ndimage.binary_fill_holes(body_ds[:, :, k]) for k in range(body_ds.shape[2])], axis=2)
a2 = AT.copy(); a2[:3, :3] *= 2
emit('skin', 'Chest wall (skin)', 'chest-wall', '#d9b8a4', body_ds, a2, faces=24000, opacity=0.1, visible=False, sigma=1.5, label=False)
skin_mm = mm(np.argwhere(body_fill & ~ndimage.binary_erosion(body_fill)), a2)
lung_c = vox_mm(Ut | Dt).mean(0)
lung_cr = vox_mm(RU_t | RM_t | RD_t).mean(0)
lung_pts = {'left': vox_mm(Ut | Dt)[::7], 'right': vox_mm(RU_t | RM_t | RD_t)[::7]}


def rib_z(n, az, side='left'):
    c = lung_c if side == 'left' else lung_cr; sg = -1 if side == 'left' else 1
    r = vox_mm(ts(f'rib_{side}_{n}'))
    ang = np.degrees(np.arctan2(r[:, 1] - c[1], sg * (r[:, 0] - c[0])))
    s = r[np.abs(ang - az) < 8]
    return float(np.median(s[:, 2])) if len(s) else None


def port(ics, az, side='left'):
    # skin point in the ics-th intercostal space (between ribs ics and ics+1) at azimuth az (0 = lateral, + = anterior)
    c = lung_c if side == 'left' else lung_cr
    z1, z2 = rib_z(ics, az, side), rib_z(ics + 1, az, side)
    return skin_at((z1 + z2) / 2 if z1 and z2 else c[2], az, side)


def skin_at(z, az, side='left'):
    # skin point at height z and azimuth az about that side's lung centre, on the chest wall rather than an arm beside it
    c = lung_c if side == 'left' else lung_cr; sg = -1 if side == 'left' else 1
    s = slab(skin_mm, z, 3.0); s = s[(s[:, 0] < c[0]) if side == 'left' else (s[:, 0] > c[0])]
    ang = np.degrees(np.arctan2(s[:, 1] - c[1], sg * (s[:, 0] - c[0])))
    near = s[np.abs(ang - az) < 4]
    # an arm lying against the chest can merge with it: keep skin within 45 mm of the lung edge along that direction
    lp = slab(lung_pts[side], z, 4.0)
    if len(lp) < 20: lp = lung_pts[side]            # below the lung base: use the whole lung's outline in that direction
    if len(lp) and len(near):
        la = np.degrees(np.arctan2(lp[:, 1] - c[1], sg * (lp[:, 0] - c[0]))); lp = lp[np.abs(la - az) < 6]
        if len(lp):
            rl = np.linalg.norm(lp[:, :2] - c[:2], axis=1).max()
            ok = np.linalg.norm(near[:, :2] - c[:2], axis=1) <= rl + 45
            if ok.any(): near = near[ok]
            else:   # the arm is fused to the chest wall in this slice: place the port on the chest wall estimate
                a_ = np.radians(az); d_ = np.array([sg * np.cos(a_), np.sin(a_)])
                return np.array([c[0] + d_[0] * (rl + 22), c[1] + d_[1] * (rl + 22), z])
    if len(near):   # the chest wall, not an arm lying beside it: the nearest skin along that direction
        return near[np.argmin(np.linalg.norm(near[:, :2] - c[:2], axis=1))]
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
    'r-anterior': [('utility', 'Utility incision, anterior, over the hilum (4th space)', port(4, 45, 'right')),
                   ('camera', 'Camera port, low anterior (diaphragm level)', port(7, 40, 'right')),
                   ('posterior', 'Working port, same level, posterior', port(7, -35, 'right'))],
    'r-posterior': [('utility', 'Utility incision, in line with the fissure (5th space)', port(5, 10, 'right')),
                    ('camera', 'Camera port, low, posterior axillary', port(7, -25, 'right')),
                    ('posterior', 'Working port, posterior', port(6, -65, 'right'))],
}
# posterolateral thoracotomy: a curved incision along the 5th intercostal space, anterior axillary line round below the
# scapular tip; the chest is entered over the upper border of the 6th rib
THOR = {}
for sd in ('left', 'right'):
    pts = np.array([port(5, az, sd) for az in range(50, -121, -10)])
    pts[1:-1] = (pts[:-2] + 2 * pts[1:-1] + pts[2:]) / 4
    c_ = lung_c if sd == 'left' else lung_cr
    _r = pts[:, :2] - c_[:2]; pts[:, :2] += _r / np.linalg.norm(_r, axis=1, keepdims=True) * 2.5     # drawn on the skin, not in it
    inward = lambda q: q + (c_ - q) * np.array([1, 1, 0]) / (np.linalg.norm((c_ - q)[:2]) + 1e-9) * 14.0
    THOR[sd] = {'path': pts, 'centre': inward(port(5, -25, sd))}
    emit_mesh(f'incision-{sd[0]}', 'Posterolateral thoracotomy incision (5th space)', f'ports-open-{sd}', '#d0433a', tube([W(q) for q in pts], 1.8), visible=False,
              note='Schematic: from the anterior axillary line, curving below the tip of the scapula; latissimus dorsi divided, serratus anterior spared, chest entered over the 6th rib.')
# ------------------------------------------------------------------ trauma: anterolateral and clamshell incisions, internal
# mammary arteries, pericardiotomy line, a right ventricular stab wound, a missile tract through the left lower lobe
print('== trauma')
TRLM = {}
st_mm = vox_mm(ts('sternum'))


def rec_of(i):
    return next(q for q in structures if q['id'] == i)


AL = {}
st_z0, st_z1 = float(st_mm[:, 2].min()), float(st_mm[:, 2].max())
raw = {}
for sd in ('left', 'right'):
    # anterolateral thoracotomy: along the 5th space from the sternal edge to the mid-axillary line
    ok_az = [az for az in range(105, -21, -10) if rib_z(5, az, sd) and rib_z(6, az, sd)]
    raw[sd] = np.array([port(5, az, sd) for az in ok_az])
# the transverse sternotomy lies in the sternal body at the level of the 4th/5th costal cartilages: the 5th space ends
# at the costal margin laterally, so the cut level is clamped into the sternal body and both incisions curve up to it
z_cs = float(np.clip((raw['left'][0][2] + raw['right'][0][2]) / 2, st_z0 + 30, st_z1 - 60))
print('  sternum z', round(st_z0, 1), round(st_z1, 1), 'transverse sternotomy z', round(z_cs, 1))
sts = slab(st_mm, z_cs, 4.0)
for sd in ('left', 'right'):
    pts = raw[sd]; a0 = pts[0]
    edge_x = sts[:, 0].min() - 8 if sd == 'left' else sts[:, 0].max() + 8
    front = sts[:, 1].max() - 2
    pts = np.vstack([[edge_x, front, z_cs], [(2 * edge_x + a0[0]) / 3, (2 * front + a0[1]) / 3 + 3, (2 * z_cs + a0[2]) / 3],
                     [(edge_x + 2 * a0[0]) / 3, (front + 2 * a0[1]) / 3 + 3, (z_cs + 2 * a0[2]) / 3], pts])
    pts[1:-1] = (pts[:-2] + 2 * pts[1:-1] + pts[2:]) / 4
    AL[sd] = pts
    c_ = lung_c if sd == 'left' else lung_cr
    q = port(5, 45, sd); ctr = q + (c_ - q) * np.array([1, 1, 0]) / (np.linalg.norm((c_ - q)[:2]) + 1e-9) * 14.0
    TRLM[f'thor-a{sd[0]}'] = ctr
    emit_mesh(f'incision-a{sd[0]}', 'Anterolateral thoracotomy incision (5th space)', 'trauma', '#d0433a', tube([W(q_) for q_ in pts], 1.8), visible=False,
              note='Schematic: along the 5th intercostal space (below the nipple, in the inframammary fold in women), from the sternal edge to the mid-axillary line.')
# clamshell: both anterolateral incisions joined across the sternum
mid_ = sts[np.abs(sts[:, 0] - np.median(sts[:, 0])) < 6]
st_front = np.array([np.median(sts[:, 0]), mid_[:, 1].max(), z_cs])
bridge = [AL['left'][0], (AL['left'][0] + st_front) / 2 + ANT * 6, st_front + ANT * 8, (AL['right'][0] + st_front) / 2 + ANT * 6, AL['right'][0]]
cs_path = list(AL['left'][::-1]) + bridge[1:-1] + list(AL['right'])
emit_mesh('incision-cs', 'Clamshell incision (bilateral anterolateral + transverse sternotomy)', 'trauma', '#d0433a', tube([W(q_) for q_ in cs_path], 1.8), visible=False,
          note='Schematic: both 5th-space anterolateral incisions joined across the sternum.')
TRLM['sternotomy'] = st_front
# the sternum is divided transversely at that level: the upper half goes up with the chest wall "lid"
st_c = sts.mean(0)
rec_of('sternum')['division'] = {'point': [round(float(x), 1) for x in W(st_c)], 'dir': [0.0, 0.0, 1.0], 'radius': 12.0}
# internal mammary (thoracic) arteries: from the subclavian, 1-2 cm lateral to the sternal edge behind the costal cartilages
for sd, sub_name in (('l', 'subclavian_artery_left'), ('r', 'subclavian_artery_right')):
    sub = vox_mm(ts(sub_name)); sgn = -1 if sd == 'l' else 1
    o = sub[np.argmin(np.abs(sub[:, 0] - (st_mm[:, 0].min() if sd == 'l' else st_mm[:, 0].max()) - sgn * 10) + 0.3 * sub[:, 2])]
    path = [o]
    for z in np.linspace(st_mm[:, 2].max() - 5, z_cs - 45, 7):
        ss = slab(st_mm, z, 4.0)
        if not len(ss): continue
        edge = ss[:, 0].min() if sd == 'l' else ss[:, 0].max()
        path.append(np.array([edge + sgn * 13, ss[:, 1].min() + 2, z]))
    path = np.array(path)
    emit_mesh(f'ima-{sd}', f'{"Left" if sd == "l" else "Right"} internal mammary artery', 'trauma', '#c24a3e', tube([W(q_) for q_ in path], 1.3), visible=False,
              note='Schematic: descends behind the costal cartilages 1-2 cm from the sternal edge. Divided by a clamshell; ligate both ends once there is a circulation.')
    k = int(np.argmin(np.abs(path[:, 2] - z_cs)))
    rec_of(f'ima-{sd}')['division'] = {'point': [round(float(x), 1) for x in W(np.array([path[k][0], path[k][1], z_cs]))], 'dir': [0.0, 0.0, 1.0], 'radius': 2.2}
# pericardiotomy: longitudinal, anterior to and parallel with the left phrenic nerve
heart_surf = cKDTree(heart_mm[::3])
pc = []
for p_ in [ph[2] + SUP * 22, ph[2], ph[3], ph[4]]:
    q_ = p_ + ANT * 16 - LEFT * 6
    _, i_ = heart_surf.query(q_); pc.append(heart_surf.data[i_] + ANT * 1.5)
emit_mesh('pericardiotomy', 'Pericardiotomy line', 'trauma', '#2a0d10', tube([W(q_) for q_ in pc], 1.1), visible=False,
          note='Opened longitudinally, anterior to and parallel with the phrenic nerve, from the base to the apex.')
TRLM['pericardiotomy-a'], TRLM['pericardiotomy-b'] = pc[0], pc[-1]
# a stab wound of the right ventricle: anterior surface, mid-ventricle
hs = slab(heart_mm, hilum_z - 45, 4.0); wpt = hs[np.argmax(hs[:, 1])] + ANT * 1.0
wm = trimesh.creation.icosphere(subdivisions=2, radius=1.0); wm.apply_scale([8.0, 2.2, 2.6]); wm.apply_translation(W(wpt))
emit_mesh('wound-rv', 'Stab wound, right ventricle', 'trauma', '#2a0507', wm, visible=False, note='Teaching example of a ventricular stab wound.')
TRLM['wound-rv'] = wpt
# a through-and-through missile tract in the left lower lobe, away from the hilum
lll_mm = vox_mm(Dt)
lz = float(np.median(lll_mm[:, 2])); ls_ = slab(lll_mm, lz, 3.0)
ta = ls_[np.argmin(ls_[:, 0])] + np.array([6.0, 0, 0])
hil_ = np.mean([np.array(rec_of(i)['division']['point']) + CARINA for i in ('pa-left', 'pv-superior', 'pv-inferior') if 'division' in rec_of(i)], axis=0)
tb_s = slab(lll_mm, lz - 12, 3.0); far_ok = tb_s[np.linalg.norm(tb_s - hil_, axis=1) > 50]
tb = far_ok[np.argmax(np.linalg.norm(far_ok - ta, axis=1))]
tb = tb + (ta - tb) / np.linalg.norm(ta - tb) * 6.0            # stop just inside the far surface
emit_mesh('tract-l', 'Missile tract, left lower lobe', 'trauma', '#4a0f14', tube([W(ta), W((ta + tb) / 2 + np.array([4.0, 0, 3])), W(tb)], 3.2), visible=False,
          note='Teaching example: a bleeding through-and-through tract away from the hilum, treated by tractotomy.')
TRLM['tract-a'], TRLM['tract-b'] = ta, tb
# descending thoracic aorta just above the diaphragm: where it is cross-clamped
z_ac = float(lll_mm[:, 2].min() + 35); a_s = slab(aorta_mm, z_ac, 4.0)
a_s = a_s[a_s[:, 1] < np.median(aorta_mm[:, 1])] if len(a_s[a_s[:, 1] < np.median(aorta_mm[:, 1])]) else a_s
TRLM['aorta-clamp'] = a_s.mean(0)
TRLM['lower-lung-lz'] = np.array([lung_c[0], lung_c[1], z_ac])
# ------------------------------------------------------------------ chest wall: girdle bones, muscle layers, surface landmarks, VATS incisions
import chestwall  # noqa: E402
_has = set(np.unique(T).tolist())
CW_L, CW_R, CWLM = chestwall.build(dict(emit=emit, emit_mesh=emit_mesh, W=W, tube=tube, sphere=sphere, ts=ts, AT=AT, has=lambda n: n in TS and TS[n] in _has,
                                        vox_mm=vox_mm, lung_c=lung_c, lung_cr=lung_cr, rib_z=rib_z, port=port, skin_at=skin_at, skin_mm=skin_mm, CARINA=CARINA))
TRLM.update(CWLM)
# ------------------------------------------------------------------ batch 4: thymus, oesophagectomy, thoracic duct, empyema
import mediastinum  # noqa: E402
MDIV = {}
_hp4 = [np.array(rec_of(i)['division']['point']) + CARINA for i in ('pa-left', 'pv-superior', 'pv-inferior', 'br-lul') if 'division' in rec_of(i)]
MDLM, MD_L, MD_R = mediastinum.build(dict(emit=emit, emit_mesh=emit_mesh, W=W, tube=tube, sphere=sphere, ts=ts, AT=AT, has=lambda n: n in TS and TS[n] in _has,
                                          vox_mm=vox_mm, CT=CT, T=T, TS=TS, skin_mm=skin_mm, division=MDIV,
                                          azygos_arch=AZ_ARCH if RIGHT_IDS else CARINA + np.array([15.0, -10, 5]), hilum_l_scanner=np.mean(_hp4, axis=0)))
for _i, _d in MDIV.items(): rec_of(_i)['division'] = _d
TRLM.update(MDLM)
# ------------------------------------------------------------------ the neck: tracheal resection
import neck  # noqa: E402
NKDIR, NKSC = {}, {}
NKLM = neck.build(dict(emit=emit, emit_mesh=emit_mesh, W=W, tube=tube, sphere=sphere, ts=ts, AT=AT, has=lambda n: n in TS and TS[n] in _has, vox_mm=vox_mm,
                       skin_mm=skin_mm, CARINA=CARINA, dirs=NKDIR, scalars=NKSC))
TRLM.update(NKLM)
# ------------------------------------------------------------------ the heart: mitral valve replacement (needs work/heart.nii.gz from segment_heart.py)
import cardiac  # noqa: E402
CDLM, CDDIR, CDSC = cardiac.build(dict(emit=emit, emit_mesh=emit_mesh, W=W, tube=tube, sphere=sphere, work=Path(WORK), aorta_mm=aorta_mm,
                                       bct_mm=vox_mm(ts('brachiocephalic_trunk')), svc_mm=vox_mm(ts('superior_vena_cava')), port=port, lung_cr=lung_cr, CARINA=CARINA))
TRLM.update(CDLM)
CW_L |= MD_L; CW_R |= MD_R
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


def plane_of(mask, pos, neg):
    # centre and unit normal (pointing from `neg` toward `pos`) of a fissure surface
    P = vox_mm(mask); c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2] * np.sign(np.dot(vt[2], vox_mm(pos).mean(0) - vox_mm(neg).mean(0))); return c, n


for id_, (c_, n_, a_) in ISP_LM.items():
    landmarks[f'{id_}-centre'] = [round(float(x), 1) for x in W(c_)]; landmarks[f'{id_}-normal'] = [round(float(x), 3) for x in n_]; landmarks[f'{id_}-axis'] = [round(float(x), 3) for x in a_]
if RIGHT_IDS:
    landmarks['r-secondary-carina'] = [round(float(x), 1) for x in W(R_SEC)]
    for nm_, (c_, n_) in (('fissure-h', plane_of(d1(RU_t) & d1(RM_t), RU_t, RM_t)), ('fissure-r', plane_of(d1(RU_t | RM_t) & d1(RD_t), RU_t | RM_t, RD_t))):
        landmarks[f'{nm_}-centre'] = [round(float(x), 1) for x in W(c_)]; landmarks[f'{nm_}-normal'] = [round(float(x), 3) for x in n_]
# which side a structure belongs to: each operation shows one side's hilum
LEFT_IDS = {'lul', 'lll', 'fissure', 'ipl', 'seg-lul-upper', 'seg-lingula', 'seg-s6', 'seg-lll-basal', 'isp-lingula', 'isp-s6', 'br-lingular', 'br-upper-div', 'br-b6', 'lig-art', 'n-phrenic', 'n-vagus', 'n-rln', 'ln-5', 'ln-6', 'ln-10l', 'ln-11l', 'ln-9l', 'br-lul', 'br-lll', 'br-left-main'}
RIGHT_IDS |= {'rul', 'rml', 'rll', 'fissure-h', 'fissure-r'} | CW_R
LEFT_IDS |= CW_L
for q in structures:
    i = q['id']
    if i in RIGHT_IDS or i.startswith('port-r-') or i == 'incision-r' or (i.startswith('rib-') and i.endswith('-r')):
        q['side'] = 'right'; q['sideVisible'] = q.get('visible', True); q['visible'] = False
    elif i in LEFT_IDS or i == 'incision-l' or i.startswith(('pa-', 'pv-', 'lul-', 'lll-', 'port-anterior', 'port-posterior', 'rib-')):
        q['side'] = 'left'
for appr, ps in PORTS.items():
    for k, _, p in ps: landmarks[f'port-{appr}-{k}'] = [round(float(x), 1) for x in W(p)]
for sd, th in THOR.items():
    landmarks[f'thor-{sd[0]}'] = [round(float(x), 1) for x in W(th['centre'])]
for k, v in TRLM.items():
    landmarks[k] = [round(float(x), 3) for x in v] if k.endswith('-axis') else [round(float(x), 1) for x in W(v)]
for k, v in NKDIR.items(): landmarks[k] = [round(float(x), 3) for x in v]
for k, v in CDDIR.items(): landmarks[k] = [round(float(x), 3) for x in v]
if CDSC: landmarks['mv-dims'] = [round(CDSC['mv-radius'], 1), 0.0, 0.0]
landmarks['trach-dims'] = [round(NKSC.get('trach-radius', 9.0), 1), round(NKSC.get('stenosis-length', 20.0), 1), 0.0]
# the left hilum as a pivot (hilar clamp and twist): the centre of its four staple lines, and the axis out into the lung
_hp = [np.array(rec_of(i)['division']['point']) for i in ('pa-left', 'pv-superior', 'pv-inferior', 'br-lul') if 'division' in rec_of(i)]
landmarks['hilum-l'] = [round(float(x), 1) for x in np.mean(_hp, axis=0)]
_ax = W(lung_c) - np.mean(_hp, axis=0); _ax[2] = 0; _ax /= np.linalg.norm(_ax)
landmarks['hilum-l-axis'] = [round(float(x), 3) for x in _ax]
atlas = {
    'ct': {'file': 'ct.hu8.gz', 'dims': [int(x) for x in shape], 'affine': [[round(float(x), 4) for x in row] for row in w_aff[:3]], 'scale': STEP, 'offset': HU0, 'spacing': vox},
    'labels': {'file': 'labels.u8.gz', 'lut': lut},
    'groups': [{'id': 'lungs', 'name': 'Lungs and fissure', 'open': True}, {'id': 'arteries', 'name': 'Pulmonary arteries', 'open': True},
               {'id': 'veins', 'name': 'Pulmonary veins', 'open': True}, {'id': 'airway', 'name': 'Airway', 'open': True},
               {'id': 'nerves', 'name': 'Nerves (schematic)', 'open': True}, {'id': 'pleura', 'name': 'Pleura and ligament (schematic)'}, {'id': 'nodes', 'name': 'Lymph node stations (schematic)'},
               {'id': 'mediastinum', 'name': 'Heart, great vessels, oesophagus'}, {'id': 'lul-intra', 'name': 'Upper lobe, intrapulmonary'},
               {'id': 'lll-intra', 'name': 'Lower lobe, intrapulmonary'}, {'id': 'chest-wall', 'name': 'Chest wall and spine'},
               {'id': 'ports-anterior', 'name': 'Ports, anterior approach'}, {'id': 'ports-posterior', 'name': 'Ports, posterior approach'},
               {'id': 'rul-intra', 'name': 'Right upper lobe, intrapulmonary'},
               {'id': 'ports-r-anterior', 'name': 'Ports, right anterior approach'}, {'id': 'ports-r-posterior', 'name': 'Ports, right posterior approach'},
               {'id': 'segments', 'name': 'Segments (from bronchial territories)'}, {'id': 'trauma', 'name': 'Trauma (schematic)'}, {'id': 'ports-open-left', 'name': 'Thoracotomy, left'}, {'id': 'ports-open-right', 'name': 'Thoracotomy, right'},
               {'id': 'abdomen', 'name': 'Upper abdomen and conduit'}, {'id': 'incisions', 'name': 'Incisions (sternotomy, neck, abdomen)'},
               {'id': 'cardiac', 'name': 'Heart: chambers, valves, cannulas'}, {'id': 'muscles', 'name': 'Chest wall muscles (schematic)'}, {'id': 'landmarks', 'name': 'Surface landmarks'}, {'id': 'ports-vats', 'name': 'VATS incisions, uni- and biportal'}],
    'structures': structures,
    'landmarks': landmarks,
    'source': {'name': 'Reference CT: 3D Slicer sample CTA (CTA-cardio)', 'licence': 'unstated',
               'note': 'Segmented with TotalSegmentator (total and lung_vessels). Branch names are assigned from the geometry of this one scan and need a surgeon\'s check.'},
}
(OUT / 'atlas.json').write_text(json.dumps(atlas))
print('structures', len(structures))
print('names', {v[0]: round(float(art_t.dist[k]), 1) for k, v in names.items()})
