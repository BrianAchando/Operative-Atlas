"""Organic rebuilds of the post-tuberculous thoracic pathology (replaces the earlier sphere-and-tube schematics by id):

  asp-cavity     an irregular, thick-walled apical cavity (wall thicker where it abuts the pleura), with its draining bronchus
  asp-ball       a lobulated fungal ball lying in the dependent (posterior) part of the cavity, leaving an air crescent in front
  asp-pleura     an apical pleural cap that follows the lung surface, with fibrous adhesion bands to the chest wall
  bx-lul         a continuous bronchial tree: thick walls, cylindrical, varicose (beaded) and cystic (saccular) segments
  tb-cavities-l  irregular cavities in the destroyed left lung

Atlas frame (scanner millimetres minus the carina). Uses the lobe masks from TotalSegmentator.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

import meshing
from sdfmesh import Vessel, Ellipsoid, mesh, frame

ANT, SUP, RIGHT = np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
V = lambda *a: np.array(a, float)


def _unit(v):
    v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-9)


def _lumpy(c, r, n, rng, spread=0.35, squash=(1.0, 1.0, 1.0)):
    """a lumpy body: a core ellipsoid and n satellite ellipsoids on its surface, all smoothly blended"""
    c = np.asarray(c, float); out = [Ellipsoid(c, np.asarray(squash) * r)]
    for _ in range(n):
        d = _unit(rng.normal(size=3)); rr = r * rng.uniform(0.45, 0.75)
        out.append(Ellipsoid(c + d * r * spread * rng.uniform(0.7, 1.2) * np.asarray(squash), [rr, rr * rng.uniform(0.75, 1.0), rr * rng.uniform(0.7, 1.0)], frame(d)))
    return out


def _cavity(c, r, wall, rng, toward=None, lumps=7):
    """outer lumpy body minus a lumpy lumen; the lumen is shifted away from `toward` so the wall is thickest there"""
    shift = -_unit(toward) * wall * 0.45 if toward is not None else 0
    outer = _lumpy(c, r * 0.85, lumps, rng, 0.75, (1.0, 0.92, 1.08))
    inner = _lumpy(c + shift, (r - wall) * 0.85, lumps - 2, rng, 0.45, (1.0, 0.9, 1.05))
    return outer, inner


def _cap(mask, AT, C, region, thick=4.0, faces=7000):
    """a shell of thickness `thick` mm on the outside of `mask`, limited to `region` (a boolean of the same shape)"""
    sp = np.abs(np.diag(AT)[:3])
    out = ndimage.distance_transform_edt(~mask, sampling=sp)
    shell = (out > 0.3) & (out <= thick) & region
    if not shell.any(): return None
    a = AT.copy(); a[:3, 3] -= C
    return meshing.mesh_from_mask(shell, a, faces, sigma=1.2, taubin_iterations=20)


def build(ctx):
    emit_mesh, ts, AT = ctx['emit_mesh'], ctx['ts'], ctx['AT']
    C = np.asarray(ctx['CARINA'], float); LMW = ctx['LMW']; LM = {}
    sp = np.abs(np.diag(AT)[:3])
    to_w = lambda idx: idx @ AT[:3, :3].T + AT[:3, 3] - C
    print('== thoracic pathology (organic)')

    # ------------------------------------------------------------ aspergilloma in a right upper lobe apical cavity
    RU = ts('lung_upper_lobe_right')
    if RU.any() and 'aspergilloma' in LMW:
        c = np.asarray(LMW['aspergilloma'], float); rng = np.random.default_rng(11)
        # the nearest pleura: the direction from the cavity to the closest point on the lobe surface
        surf = to_w(np.argwhere(RU & ~ndimage.binary_erosion(RU, iterations=2)))[::3]
        near = surf[np.argsort(np.linalg.norm(surf - c, axis=1))[:60]].mean(0); pl = _unit(near - c)
        outer, inner = _cavity(c, 17.0, 4.2, rng, toward=pl, lumps=8)
        # the draining (apical segmental) bronchus: enters the cavity from below and medially
        hil = np.asarray(LMW.get('hilum-r', c - SUP * 45 - RIGHT * 25), float)
        b0 = c + _unit(hil - c) * 14; b1 = c + _unit(hil - c) * 36
        br = Vessel([b0 - _unit(hil - c) * 4, (b0 + b1) / 2 + V(1.5, -2, 0), b1], [3.4, 3.0, 2.8], step=0.8)
        lum = Vessel([c, b0 + _unit(hil - c) * 2, b1 + _unit(hil - c) * 2], [1.9, 1.9, 1.7], step=0.8)
        cav = mesh(outer + [br], voxel=0.55, blend=2.5, subtract=inner + [lum], density=1.6, smooth=10)
        emit_mesh('asp-cavity', 'Post-tuberculous cavity (thick, irregular wall)', 'pathology', '#c9b8a0', cav, opacity=0.5, visible=False,
                  note='Schematic: a healed TB cavity at the apex. The wall is irregular and thickest against the pleura; a damaged segmental bronchus drains it, and enlarged bronchial and intercostal arteries in the wall are the source of haemoptysis.')
        # fungal ball: lobulated, resting in the dependent (posterior when supine) part, leaving an air crescent anteriorly
        bc = c - ANT * 4.5 - SUP * 1.0
        ball = _lumpy(bc, 8.2, 14, rng, 0.55, (1.05, 0.9, 1.0))
        ball += [Ellipsoid(bc + _unit(rng.normal(size=3)) * 7.5, [2.2, 2.2, 2.2]) for _ in range(18)]       # surface knobs
        emit_mesh('asp-ball', 'Fungal ball (aspergilloma)', 'pathology', '#6f7a4a', mesh(ball, voxel=0.4, blend=1.6, density=2.5, smooth=6), visible=False,
                  note='A mass of Aspergillus hyphae, mucus and debris lying free in the cavity, in its dependent part; the air crescent above it (Monod sign) is the CT clue. It moves with posture.')
        # apical pleural cap: a thickened rind over the lobe surface near the cavity, plus adhesion bands to the chest wall
        sl = tuple(slice(max(0, int(a) - 2), int(b) + 3) for a, b in zip(np.argwhere(RU).min(0), np.argwhere(RU).max(0)))
        sub = RU[sl]; lo = np.array([s.start for s in sl])
        g = np.stack(np.meshgrid(*[np.arange(n) for n in sub.shape], indexing='ij'), -1)
        Wg = (g + lo) @ AT[:3, :3].T + AT[:3, 3] - C
        region = (np.linalg.norm(Wg - c, axis=-1) < 42) & (Wg[..., 2] > c[2] - 14)
        a2 = AT.copy(); a2[:3, 3] = AT[:3, 3] + AT[:3, :3] @ lo
        cap = _cap(sub, a2, C, region, thick=4.5)
        parts = [cap] if cap is not None else []
        cs = to_w(np.argwhere(sub & ~ndimage.binary_erosion(sub, iterations=2)) + lo)
        cs = cs[(np.linalg.norm(cs - c, axis=1) < 38) & (cs[:, 2] > c[2] - 8)]
        if len(cs):
            mid = cs.mean(0); bands = []
            for s in cs[rng.choice(len(cs), size=min(14, len(cs)), replace=False)]:
                d = _unit(s - mid + _unit(s - c) * 10); L = rng.uniform(7, 12)
                bands.append(Vessel([s - d * 1.0, s + d * L * 0.5 + rng.normal(0, 1.5, 3), s + d * L], [1.2, rng.uniform(0.6, 1.0), 1.6], step=0.7))
            parts.append(mesh(bands, voxel=0.4, blend=1.2, density=1.6, smooth=6))
        if parts:
            emit_mesh('asp-pleura', 'Thickened apical pleura and adhesions', 'pathology', '#d8cfb8', trimesh.util.concatenate(parts), opacity=0.35, visible=False,
                      note='A thick, vascular pleural rind over the apex, tethered to the chest wall by fibrous bands: extrapleural dissection is often needed.')
        LM['aspergilloma'] = c

    # ------------------------------------------------------------ TB-destroyed left lung: irregular cavities
    LU, LL = ts('lung_upper_lobe_left'), ts('lung_lower_lobe_left')
    if LU.any() and LL.any():
        rng = np.random.default_rng(4); outer, inner = [], []
        for m_, n_ in ((LU, 3), (LL, 2)):
            idx = np.argwhere(ndimage.binary_erosion(m_, iterations=12)); idx = idx[rng.choice(len(idx), size=min(n_, len(idx)), replace=False)] if len(idx) else idx
            for q in idx:
                o, i = _cavity(to_w(q), float(rng.uniform(9, 15)), 2.6, rng, toward=rng.normal(size=3), lumps=6)
                outer += o; inner += i
        if outer:
            emit_mesh('tb-cavities-l', 'Cavities and bronchiectasis (TB-destroyed lung)', 'pathology', '#c9b8a0', mesh(outer, voxel=0.6, blend=2.0, subtract=inner, density=1.0, smooth=8),
                      opacity=0.6, visible=False,
                      note='Schematic: a lung destroyed by tuberculosis: cavities, bronchiectasis and fibrosis, with volume loss and pleural symphysis; a reservoir for infection and haemoptysis.')

    # ------------------------------------------------------------ post-TB bronchiectasis, left upper lobe: one continuous, thick-walled tree
    if LU.any() and 'hilum-l' in LMW:
        er = ndimage.binary_erosion(LU, iterations=max(2, int(8 / sp.mean())))
        q = to_w(np.argwhere(er))
        o = np.asarray(LMW['hilum-l'], float) + V(-6, 0, 6)
        rng = np.random.default_rng(7); tips = []; walls = []; lumens = []
        toward = lambda target: q[np.argmin(np.linalg.norm(q - target, axis=1))]
        def add(P, r, wall):
            walls.append(Vessel(P, r, step=0.7))
            lumens.append(Vessel(P, [max(0.6, x - wall) for x in r], step=0.7))
        add([o + V(12, 0, -7), o], [4.4, 4.0], 1.4)
        for target in (V(-60, 30, 40), V(-90, 10, 20), V(-70, -20, 30), V(-100, 40, -10), V(-55, 5, 55)):
            t = toward(target); tips.append(t)
            mid = o + (t - o) * 0.45 + rng.normal(0, 4, 3)
            add([o, o + (mid - o) * 0.5, mid], [3.1, 3.0, 3.2], 1.2)
            for j in range(3):
                end = toward(t + rng.normal(0, 14, 3))
                ctrl = [mid, mid + (end - mid) * 0.5 + rng.normal(0, 3, 3), end]
                kind = (j + len(tips)) % 3
                if kind == 0: rf = lambda s: 1.9 + 1.7 * s                                               # cylindrical: widening toward the periphery
                elif kind == 1: rf = lambda s: 1.9 + 1.0 * s + 1.4 * np.sin(s * np.pi * 5) ** 2           # varicose: beaded
                else: rf = lambda s: 1.9 + 0.6 * s
                walls.append(Vessel(ctrl, rf, step=0.6)); lumens.append(Vessel(ctrl, lambda s, rf=rf: max(0.7, rf(s) - 1.1), step=0.6))
                if kind == 2:                                                                             # cystic (saccular): a cluster of thin-walled sacs
                    R = float(rng.uniform(4.5, 7.0)) * 0.8; d = _unit(end - mid)
                    for k in range(3):
                        cc = end + d * R * 0.55 + _unit(rng.normal(size=3)) * R * (0.0 if k == 0 else 0.6); rr = R * (1.0 if k == 0 else 0.65)
                        walls.append(Ellipsoid(cc, [rr, rr, rr])); lumens.append(Ellipsoid(cc, [rr - 0.9] * 3))
        tree = mesh(walls, voxel=0.45, blend=0.9, subtract=lumens, density=0.9, smooth=8)
        emit_mesh('bx-lul', 'Bronchiectasis: dilated, varicose and cystic bronchi (left upper lobe)', 'pathology', '#e7d9a8', tree, opacity=0.8, visible=False,
                  note='Traction and post-infective bronchiectasis after tuberculosis: thick-walled bronchi wider than their artery (signet ring), cylindrical, varicose and cystic, pooling secretions; recurrent infection and haemoptysis.')
        LM['bx-lul'] = tips[0]
    return LM
