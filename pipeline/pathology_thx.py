"""Thoracic pathology for the case scenarios (schematic, placed on this patient's own lobes and mediastinum).

  lung cancer        a spiculated peripheral tumour in each lobe; a central (hilar) tumour on each side
  post-TB lung       an apical cavity with a fungal ball (aspergilloma), right upper lobe; cavities in a destroyed left lung
  congenital         congenital lobar emphysema (an over-distended left upper lobe); CPAM cysts in the left lower lobe
  mediastinum        a thymoma on the thymus; oesophageal tumours (mid and lower third)
  after right pneumonectomy: the post-pneumonectomy space (air above, infected fluid below), the bronchial stump with a
                     bronchopleural fistula, an open window (Eloesser) on the chest wall, and a latissimus dorsi flap
                     brought in through the window to the stump
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

ANT, SUP, RIGHT = np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
LOBES = {'lul': 'lung_upper_lobe_left', 'lll': 'lung_lower_lobe_left', 'rul': 'lung_upper_lobe_right', 'rml': 'lung_middle_lobe_right', 'rll': 'lung_lower_lobe_right'}


def _blob(C, c, r, seed, lumps=8, spread=0.7, spikes=0):
    """an irregular mass: overlapping spheres, optionally with spicules (thin cones) as in a lung carcinoma"""
    rng = np.random.default_rng(seed); parts = []
    for k in range(lumps):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        s = trimesh.creation.icosphere(subdivisions=3, radius=r * (0.8 if k == 0 else rng.uniform(0.45, 0.8)))
        s.apply_translation(np.asarray(c, float) + (0 if k == 0 else d * r * spread * rng.uniform(0.3, 1.0)) - C); parts.append(s)
    for _ in range(spikes):
        d = rng.normal(size=3); d /= np.linalg.norm(d); L = r * rng.uniform(0.7, 1.3)
        cone = trimesh.creation.cone(radius=r * 0.12, height=L, sections=8)
        T = trimesh.geometry.align_vectors([0, 0, 1], d); T[:3, 3] = np.asarray(c, float) + d * r * 0.6 - C; cone.apply_transform(T); parts.append(cone)
    return trimesh.util.concatenate(parts)


def _shell(C, c, r, t=2.5):
    """a thick-walled cavity: an outer sphere (drawn translucent) with an inner, inverted sphere"""
    o = trimesh.creation.icosphere(subdivisions=3, radius=r); i = trimesh.creation.icosphere(subdivisions=3, radius=r - t); i.invert()
    m = trimesh.util.concatenate([o, i]); m.apply_translation(np.asarray(c, float) - C); return m


def build(ctx):
    ts, vox_mm, emit, emit_mesh, W, tube, sphere, C = ctx['ts'], ctx['vox_mm'], ctx['emit'], ctx['emit_mesh'], ctx['W'], ctx['tube'], ctx['sphere'], ctx['CARINA']
    AT = ctx['AT']; sp = np.abs(np.diag(AT)[:3]); LM = {}
    carina = np.asarray(C, float)
    print('== thoracic pathology')
    # ---------------------------------------------------------------- lung cancer: a peripheral tumour in each lobe
    masks = {}
    for k, nm in LOBES.items():
        m = ts(nm)
        if not m.any(): continue
        masks[k] = m
        idx = np.argwhere(m); lo, hi = idx.min(0), idx.max(0) + 1; sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        edt = ndimage.distance_transform_edt(sub, sampling=sp)
        cand = np.argwhere((edt > 14) & (edt < 22))
        if not len(cand): cand = np.argwhere(edt >= edt.max() * 0.7)
        pts = (cand + lo) @ AT[:3, :3].T + AT[:3, 3]
        cen = pts.mean(0)
        # peripheral, in the middle of the lobe's extent: far from the carina but near the lobe's own centre
        score = np.linalg.norm(pts - carina, axis=1) * 0.6 - np.linalg.norm(pts - cen, axis=1)
        p = pts[int(np.argmax(score))]
        r = 11.0 if k != 'rml' else 9.0
        emit_mesh(f'tumour-{k}', f'Lung carcinoma, {k.upper()} (about {round(2 * r * 1.3 / 10, 1)} cm)', 'pathology', '#e6e0d0', _blob(C, p, r, {'lul': 11, 'lll': 12, 'rul': 13, 'rml': 14, 'rll': 15}[k], lumps=7, spikes=9),
                  visible=False, note='Schematic: a spiculated peripheral tumour. Its size, distance from the hilum and any nodes decide the T and N stage and the resection.')
        LM[f'tumour-{k}'] = p
    # central tumours: at the lobar hilum, next to the main bronchus
    for side, lobes_, bk in (('l', ('lul', 'lll'), 'lmb'), ('r', ('rul', 'rll'), 'rmb')):
        b = ctx.get(bk)
        if b is None or not all(x in masks for x in lobes_): continue
        lung = masks[lobes_[0]] | masks[lobes_[1]]; pts = vox_mm(lung)[::5]
        lat = RIGHT if side == 'r' else -RIGHT
        q = pts[np.argmin(np.linalg.norm(pts - (b + lat * 22), axis=1))]
        emit_mesh(f'tumour-central-{side}', f'Central (hilar) carcinoma, {"right" if side == "r" else "left"}', 'pathology', '#e6e0d0', _blob(C, q, 16.0, 7 if side == 'r' else 8, lumps=9, spikes=6),
                  visible=False, note='Schematic: a central tumour at the hilum, involving the main or lobar bronchus and the pulmonary artery: pneumonectomy territory.')
        LM[f'tumour-central-{side}'] = q
    # ---------------------------------------------------------------- post-TB: aspergilloma in a right upper lobe apical cavity
    if 'rul' in masks:
        m = masks['rul']; idx = np.argwhere(m); zt = np.percentile(idx[:, 2], 88); top = idx[idx[:, 2] >= zt]
        c = top.mean(0) @ AT[:3, :3].T + AT[:3, 3]
        c = c - SUP * 6
        emit_mesh('asp-cavity', 'Post-tuberculous cavity (thick wall)', 'pathology', '#c9b8a0', _shell(C, c, 17.0, 3.0), opacity=0.55, visible=False,
                  note='Schematic: a healed TB cavity at the apex, its wall thickened, lined by damaged bronchi and fed by enlarged bronchial and intercostal arteries (the source of haemoptysis).')
        emit_mesh('asp-ball', 'Fungal ball (aspergilloma)', 'pathology', '#6f7a4a', _blob(C, c - ANT * 5 - SUP * 3, 9.5, 21, lumps=6, spread=0.5), visible=False,
                  note='A mass of Aspergillus hyphae, mucus and debris lying free in the cavity; it moves with posture (the dependent part of the cavity when supine is posterior).')
        emit_mesh('asp-pleura', 'Thickened apical pleura and adhesions', 'pathology', '#d8cfb8', _shell(C, c + SUP * 6, 26.0, 4.0), opacity=0.3, visible=False,
                  note='Dense, vascular adhesions to the apex and chest wall: extrapleural dissection is often needed.')
        LM['aspergilloma'] = c
    # destroyed left lung: several cavities
    if 'lul' in masks and 'lll' in masks:
        rng = np.random.default_rng(4); cav = []
        for k, n_ in (('lul', 3), ('lll', 2)):
            idx = np.argwhere(ndimage.binary_erosion(masks[k], iterations=12)); idx = idx[rng.choice(len(idx), size=min(n_, len(idx)), replace=False)] if len(idx) else idx
            for q in idx: cav.append(_shell(C, q @ AT[:3, :3].T + AT[:3, 3], float(rng.uniform(9, 15)), 2.2))
        if cav:
            emit_mesh('tb-cavities-l', 'Cavities and bronchiectasis (TB-destroyed lung)', 'pathology', '#c9b8a0', trimesh.util.concatenate(cav), opacity=0.6, visible=False,
                      note='Schematic: a lung destroyed by tuberculosis: cavities, bronchiectasis and fibrosis, with volume loss and pleural symphysis; a reservoir for infection and haemoptysis.')
    # ---------------------------------------------------------------- congenital: CLE (left upper lobe), CPAM (left lower lobe)
    if 'lul' in masks:
        big = ndimage.binary_dilation(masks['lul'], iterations=int(round(7 / sp.mean())))
        emit('cle-lul', 'Congenital lobar emphysema: over-distended left upper lobe', 'pathology', '#f1d6d0', big, AT, faces=5000, opacity=0.35, visible=False,
             note='Schematic: a ball-valve defect in the lobar bronchus (deficient cartilage) lets air in but not out; the lobe over-distends, compresses the lower lobe and pushes the mediastinum across.')
    if 'lll' in masks:
        m = masks['lll']; idx = np.argwhere(ndimage.binary_erosion(m, iterations=10)); rng = np.random.default_rng(12)
        if len(idx):
            base = idx[idx[:, 2] <= np.percentile(idx[:, 2], 45)]
            cy = []
            for q in base[rng.choice(len(base), size=min(8, len(base)), replace=False)]:
                cy.append(_shell(C, q @ AT[:3, :3].T + AT[:3, 3], float(rng.uniform(6, 16)), 1.0))
            emit_mesh('cpam-lll', 'CPAM: air-filled cysts, left lower lobe', 'pathology', '#cfe0ea', trimesh.util.concatenate(cy), opacity=0.55, visible=False,
                      note='Schematic: thin-walled cysts of a congenital pulmonary airway malformation (Stocker type 1: one or more cysts over 2 cm). They connect poorly with the airway and become infected.')
    # ---------------------------------------------------------------- thymoma
    st = ctx.get('sternum_mm')
    if st is not None and len(st):
        zt = np.percentile(st[:, 2], 55); up = st[st[:, 2] >= zt]; c = up.mean(0) - ANT * 26 - SUP * 12
        emit_mesh('thymoma', 'Thymoma (about 5 cm), anterior mediastinum', 'pathology', '#d9b98c', _blob(C, c, 20.0, 31, lumps=9, spread=0.55), visible=False,
                  note='Schematic: an encapsulated, lobulated thymoma arising in the thymus, behind the sternum and in front of the pericardium and great veins.')
        LM['thymoma'] = c
    # ---------------------------------------------------------------- oesophageal carcinoma (mid and lower third)
    eso = vox_mm(ts('esophagus')) if ts('esophagus').any() else None
    if eso is not None and len(eso) > 100:
        zs = np.arange(np.percentile(eso[:, 2], 2), np.percentile(eso[:, 2], 98), 4.0)
        cl = [eso[np.abs(eso[:, 2] - z) < 2.5].mean(0) for z in zs if (np.abs(eso[:, 2] - z) < 2.5).sum() > 5]
        cl = np.array(cl)
        def seg(z0, z1, seed):
            P = cl[(cl[:, 2] >= z0) & (cl[:, 2] <= z1)]
            if len(P) < 3: return None
            rng = np.random.default_rng(seed); parts = [tube([W(p) for p in P], 12.5, seg=18)]
            for p in P[::2]:
                d = rng.normal(size=3); d[2] = 0; d /= np.linalg.norm(d) + 1e-9
                parts.append(sphere(W(p + d * 8), float(rng.uniform(6, 9))))
            return trimesh.util.concatenate(parts), P[len(P) // 2]
        zc = carina[2]
        mid = seg(zc - 20, zc + 25, 3)
        lo_z = cl[:, 2].min()
        low = seg(lo_z + 15, lo_z + 65, 5)
        if mid is not None:
            emit_mesh('eso-tumour-mid', 'Oesophageal carcinoma, middle third (squamous)', 'pathology', '#9c4a3a', mid[0], visible=False,
                      note='Schematic: a circumferential squamous cell carcinoma at the level of the carina and the left main bronchus: check the airway (bronchoscopy) before resection.')
            LM['eso-tumour-mid'] = mid[1]
        if low is not None:
            emit_mesh('eso-tumour-low', 'Oesophageal carcinoma, lower third', 'pathology', '#9c4a3a', low[0], visible=False,
                      note='Schematic: a lower-third tumour above the gastro-oesophageal junction.')
            LM['eso-tumour-low'] = low[1]
    # ---------------------------------------------------------------- after a right pneumonectomy: space, stump, fistula, window, flap
    rl = [masks[k] for k in ('rul', 'rml', 'rll') if k in masks]
    rmb = ctx.get('rmb'); port = ctx.get('port')
    if len(rl) == 3 and rmb is not None:
        lung = rl[0] | rl[1] | rl[2]
        space = ndimage.binary_erosion(lung, iterations=int(round(6 / sp.mean())))   # the space contracts as the mediastinum shifts
        idx = np.argwhere(space); z_lvl = np.percentile(idx[:, 2], 38)
        k_ = int(z_lvl) + 1
        fluid = space.copy(); fluid[..., k_:] = False
        air = space.copy(); air[..., :k_] = False
        emit('ppe-fluid', 'Infected fluid (post-pneumonectomy empyema)', 'pathology', '#b3a64a', fluid, AT, faces=6000, opacity=0.7, visible=False,
             note='Schematic: after pneumonectomy the space fills with serous fluid over weeks; infected, it becomes empyema. A falling fluid level, or a new air-fluid level, suggests a bronchopleural fistula.')
        emit('ppe-air', 'Air in the post-pneumonectomy space', 'pathology', '#cfe6f2', air, AT, faces=5000, opacity=0.12, visible=False)
        LM['ppe-level'] = vox_mm(fluid).mean(0) if fluid.any() else rmb
        # the stump: 5-10 mm of right main bronchus beyond the carina, and the fistula at its end
        d = rmb - carina; d[2] = min(d[2], 0); d /= np.linalg.norm(d) + 1e-9
        s0 = carina + d * 8; s1 = carina + d * 20
        emit_mesh('bronchial-stump', 'Right main bronchial stump', 'pathology', '#e7d9a8', tube([W(s0), W(s1)], 7.0, seg=16), visible=False,
                  note='The right main bronchus divided flush with the carina: a long stump pools secretions and is more prone to break down.')
        f1 = s1 + d * 10 - SUP * 6
        emit_mesh('bpf', 'Bronchopleural fistula', 'pathology', '#d0433a', tube([W(s1 - d * 2), W(s1 + d * 4), W(f1)], 2.6, seg=12), visible=False,
                  note='A breakdown of the stump: the airway opens into the infected space. Fluid is coughed up and can spill into the remaining lung.')
        LM['bronchial-stump'] = s1; LM['bpf'] = f1
        if port is not None:
            # an open window: a rectangle of chest wall over the lowest part of the cavity, laterally (ribs 6-7 resected)
            lz = LM['ppe-level'][2] - 10                                     # over the lowest part of the cavity, laterally
            ic = min(range(3, 10), key=lambda i: abs(port(i, 10, 'right')[2] - lz))
            ring = [port(ic - 1, a, 'right') for a in np.linspace(-5, 30, 5)] + [port(ic + 1, a, 'right') for a in np.linspace(30, -5, 5)]
            ring = np.array(ring)
            emit_mesh('eloesser', 'Open window thoracostomy (Eloesser flap)', 'incisions', '#d0433a', tube([W(p) for p in [*ring, ring[0]]], 2.2), visible=False,
                      note='Segments of two ribs removed over the dependent part of the cavity; the skin is sutured to the parietal pleura so the window stays open and drains.')
            wc = ring.mean(0); LM['eloesser'] = wc
            # latissimus dorsi flap: from its posterior origin, rotated in through the window, laid on the stump
            base_ = port(5, -70, 'right')
            path = [base_, base_ + (wc - base_) * 0.5 - RIGHT * 10, wc - RIGHT * 12, (wc + s1) / 2, s1 + RIGHT * 6]
            emit_mesh('flap-lat', 'Latissimus dorsi flap (to the stump)', 'pathology', '#a4453d', tube([W(p) for p in path], 9.0, seg=16), visible=False,
                      note='A pedicled muscle flap on the thoracodorsal vessels brought into the chest (through a window or a resected rib bed) to cover the closed stump and fill part of the space.')
    return LM
