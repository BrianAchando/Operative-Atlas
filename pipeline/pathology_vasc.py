"""The aorta for the vascular module: its abdominal branches, the kidneys, the iliac and femoral arteries, aneurysms,
occlusive disease, grafts and stent grafts, and the accesses (laparotomy, left flank, groins, left thoracotomy, axilla).

From the CT: the aorta (thoracic and abdominal, down to the 4th lumbar vertebra, the lower edge of this scan), the
kidneys, the vertebrae (for levels), the skin. Schematic, on those landmarks: the coeliac trunk, SMA, renal arteries and
veins, IMA, the infrarenal IVC; and BELOW THE SCAN (which ends at L4) the iliac and femoral vessels and the groins,
extrapolated to typical adult dimensions and marked as such.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

ANT, SUP, RIGHT = np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
ART, VEIN, GRAFT, STENT = '#c0392b', '#4b5fa8', '#f1f1ea', '#c9d1d9'


def _lathe_path(C, P, radii, sections=36):
    """a tube of varying radius along a polyline (an aneurysm sac, a graft of changing calibre)"""
    P = np.asarray(P, float); n = len(P); T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    ref = np.array([1.0, 0, 0]) if abs(T[0][0]) < 0.9 else np.array([0, 1.0, 0]); V, F = [], []
    u = np.cross(T[0], ref); u /= np.linalg.norm(u)
    for i in range(n):
        u = u - T[i] * np.dot(u, T[i]); u /= np.linalg.norm(u) + 1e-9; v = np.cross(T[i], u)
        for a in np.linspace(0, 2 * np.pi, sections, endpoint=False): V.append(P[i] + (u * np.cos(a) + v * np.sin(a)) * radii[i])
    for i in range(n - 1):
        for j in range(sections):
            a, b = i * sections + j, i * sections + (j + 1) % sections; F += [[a, a + sections, b + sections], [a, b + sections, b]]
    V.append(P[0]); V.append(P[-1]); c0, c1 = len(V) - 2, len(V) - 1
    for j in range(sections):
        F.append([c0, (j + 1) % sections, j]); F.append([c1, (n - 1) * sections + j, (n - 1) * sections + (j + 1) % sections])
    return trimesh.Trimesh(np.array(V) - C, np.array(F), process=True)


def _resample(P, step=3.0):
    P = np.asarray(P, float); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.arange(0, d[-1] + 1e-6, step); return np.array([np.interp(t, d, P[:, k]) for k in range(3)]).T


def _cr(P, step=1.5):
    """Catmull-Rom through control points (mm), sampled about every `step` mm"""
    P = np.asarray(P, float); out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)]
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1]); return np.array(out)


def _frames(Q):
    T = np.gradient(Q, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    ref = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(T[0], ref); u /= np.linalg.norm(u); U, Vv = [], []
    for t in T:
        u = u - t * np.dot(u, t); u /= np.linalg.norm(u) + 1e-9; U.append(u); Vv.append(np.cross(t, u))
    return T, np.array(U), np.array(Vv)


def build(ctx):
    ts, vox_mm, emit, emit_mesh, W, tube, sphere, C = ctx['ts'], ctx['vox_mm'], ctx['emit'], ctx['emit_mesh'], ctx['W'], ctx['tube'], ctx['sphere'], ctx['CARINA']
    AT = ctx['AT']; LM = {}
    def ring(p, d, r, rr=1.2):
        t_ = trimesh.creation.torus(major_radius=r, minor_radius=rr, major_sections=40, minor_sections=8)
        T_ = trimesh.geometry.align_vectors([0, 0, 1], d / np.linalg.norm(d)); T_[:3, 3] = W(p); t_.apply_transform(T_); return t_
    ao = ts('aorta')
    if not ao.any(): return LM
    print('== vascular')
    # ---------------------------------------------------------------- the aortic centreline, slice by slice (two blobs above the arch base: ascending, descending)
    ks = sorted(set(np.argwhere(ao)[:, 2].tolist()))
    asc, desc = [], []
    for k in ks:
        lab, n = ndimage.label(ao[:, :, k])
        if n == 0: continue
        cs = []
        for i in range(n):
            ij = np.argwhere(lab == i + 1)
            if len(ij) < 20: continue
            p = np.c_[ij, np.full(len(ij), k)] @ AT[:3, :3].T + AT[:3, 3]
            cs.append((p.mean(0), float(np.sqrt(len(ij) * abs(AT[0, 0] * AT[1, 1]) / np.pi))))
        if len(cs) == 2:
            cs.sort(key=lambda c: -c[0][1]); asc.append(cs[0]); desc.append(cs[1])     # the more anterior is ascending
        elif len(cs) == 1:
            desc.append(cs[0])
    desc.sort(key=lambda c: -c[0][2]); asc.sort(key=lambda c: c[0][2])
    D = np.array([c[0] for c in desc]); Dr = np.array([c[1] for c in desc])
    # keep the descending centreline below the arch only
    arch_z = max(c[0][2] for c in asc) if asc else D[:, 2].max()
    keep = D[:, 2] <= arch_z - 5; D, Dr = D[keep], Dr[keep]
    k_ = 5; D = np.array([D[max(0, i - k_):i + k_ + 1].mean(0) for i in range(len(D))])
    zlo = float(D[:, 2].min())
    at_z = lambda z: np.array([np.interp(z, D[::-1, 2], D[::-1, j]) for j in range(3)])
    r_at = lambda z: float(np.interp(z, D[::-1, 2], Dr[::-1]))
    # levels from the vertebrae
    def vz(name, f=0.5):
        m = ts(name)
        if not m.any(): return None
        z = vox_mm(m)[:, 2]; return float(z.min() + (z.max() - z.min()) * f)
    z_t12, z_l1, z_l2, z_l3 = vz('vertebrae_T12', 0.35), vz('vertebrae_L1'), vz('vertebrae_L2', 0.7), vz('vertebrae_L3')
    z_l1top = vz('vertebrae_L1', 0.95)
    z_coel = z_l1top if z_l1top is not None else (z_t12 if z_t12 is not None else zlo + 100)   # coeliac at T12/L1, SMA at L1, renals at L1/L2
    z_sma = z_coel - 12
    # kidneys and the renal arteries (to each hilum: the medial, middle part of the kidney)
    kid = {}
    for sd, nm in (('l', 'kidney_left'), ('r', 'kidney_right')):
        m = ts(nm)
        if not m.any(): continue
        emit(f'kidney-{sd}', f'Kidney, {"left" if sd == "l" else "right"}', 'vascular', '#8a3a33', m, AT, faces=5000, visible=False)
        p = vox_mm(m); zc = np.median(p[:, 2]); mid = p[np.abs(p[:, 2] - zc) < 8]
        hil = mid[np.argmin(np.abs(mid[:, 0] - (D[:, 0].mean())))]                                 # the medial edge
        kid[sd] = (hil, zc)
    z_ren = {sd: z_sma - (13 if sd == 'l' else 17) for sd in kid}                                    # renal origins just below the SMA (the left usually a little higher)
    z_ren_lo = min(z_ren.values()) if z_ren else z_sma - 20
    for sd in kid:
        o = at_z(z_ren[sd]); h = kid[sd][0]
        side = -RIGHT if sd == 'l' else RIGHT
        path = [o + side * r_at(z_ren[sd]) * 0.8, o + side * 20 - ANT * 4, (o + h) / 2 - ANT * 6, h]
        emit_mesh(f'renal-a-{sd}', f'Renal artery, {"left" if sd == "l" else "right"}', 'vascular', ART, tube([W(p) for p in path], 2.8), visible=False,
                  note='Schematic on this patient\'s aorta and kidney. The right renal artery passes behind the IVC.' if sd == 'r' else 'Schematic on this patient\'s aorta and kidney.')
        LM[f'renal-{sd}'] = o
    # coeliac trunk and SMA: anterior branches
    oc = at_z(z_coel); os_ = at_z(z_sma)
    emit_mesh('coeliac', 'Coeliac trunk', 'vascular', ART, tube([W(oc + ANT * r_at(z_coel) * 0.8), W(oc + ANT * 18 + SUP * 3), W(oc + ANT * 26 + SUP * 2 - RIGHT * 10), W(oc + ANT * 30 + RIGHT * 14)], 3.2), visible=False,
              note='Schematic: from the front of the aorta at T12/L1, dividing into the left gastric, splenic and common hepatic arteries.')
    emit_mesh('sma', 'Superior mesenteric artery', 'vascular', ART, tube([W(os_ + ANT * r_at(z_sma) * 0.8), W(os_ + ANT * 16 - SUP * 4), W(os_ + ANT * 24 - SUP * 25), W(os_ + ANT * 28 - SUP * 60 + RIGHT * 10)], 3.2), visible=False,
              note='Schematic: about 1 cm below the coeliac trunk, running down in front of the left renal vein and the uncinate process.')
    LM['coeliac'] = oc; LM['sma'] = os_
    z_ima = z_l3 if z_l3 is not None else zlo + 25
    oi = at_z(z_ima)
    emit_mesh('ima', 'Inferior mesenteric artery', 'vascular', ART, tube([W(oi + ANT * r_at(z_ima) * 0.7 - RIGHT * 4), W(oi + ANT * 10 - RIGHT * 14 - SUP * 6), W(oi + ANT * 12 - RIGHT * 32 - SUP * 40)], 1.8), visible=False,
              note='Schematic: from the left front of the aorta at L3, often patent into an aneurysm; its reimplantation is considered when pelvic perfusion is doubtful.')
    LM['ima'] = oi
    # the IVC below the liver and the left renal vein across the front of the aorta
    ivc = ts('inferior_vena_cava'); ivc_p = vox_mm(ivc) if ivc.any() else None
    if ivc_p is not None and 'l' in kid:
        z0 = float(ivc_p[:, 2].min()); base = ivc_p[ivc_p[:, 2] < z0 + 6].mean(0)
        ivc_path = [base + SUP * 30, base, base - SUP * 30, np.array([base[0], base[1], zlo - 12])]
        emit_mesh('ivc-infra', 'Inferior vena cava (infrarenal)', 'vascular', VEIN, tube([W(p) for p in ivc_path], 10.0), opacity=0.9, visible=False,
                  note='Schematic below the segmented IVC: to the right of the aorta, the confluence of the iliac veins at L5.')
        lz = z_ren['l'] - 6; o = at_z(lz); h = kid['l'][0]
        iv = np.array([base[0], base[1], lz])
        lrv = [h, (h + o) / 2 + ANT * 4, o + ANT * (r_at(lz) + 6), iv]
        emit_mesh('renal-v-l', 'Left renal vein (in front of the aorta)', 'vascular', VEIN, tube([W(p) for p in lrv], 4.5), visible=False,
                  note='Crosses in front of the aorta just below the SMA, receiving the gonadal (below), adrenal (above) and lumbar veins on the left. It marks the upper limit of an infrarenal neck; dividing it near the IVC gives access to the juxtarenal aorta.')
        LM['renal-v-l'] = o + ANT * (r_at(lz) + 6); LM['ivc'] = base
    # ---------------------------------------------------------------- below the scan: aortic bifurcation, iliac and femoral arteries (schematic)
    bif = at_z(zlo) - SUP * 6
    LM['bifurcation'] = bif
    il = {}
    for sd, s in (('l', -1), ('r', 1)):
        lat = RIGHT * s
        cia_end = bif + lat * 32 - SUP * 55 - ANT * 4
        eia_end = cia_end + lat * 38 - SUP * 95 + ANT * 45
        cfa_end = eia_end - SUP * 40 + ANT * 8
        iia = [cia_end, cia_end + lat * 8 - SUP * 20 - ANT * 22, cia_end + lat * 14 - SUP * 45 - ANT * 30]
        sfa = [cfa_end, cfa_end - SUP * 60 - lat * 6 + ANT * 2, cfa_end - SUP * 120 - lat * 12]
        pfa = [cfa_end, cfa_end - SUP * 20 + lat * 6 - ANT * 10, cfa_end - SUP * 70 + lat * 10 - ANT * 18]
        nm = 'left' if sd == 'l' else 'right'
        emit_mesh(f'cia-{sd}', f'Common iliac artery, {nm} (schematic, below the scan)', 'vascular', ART, tube([W(bif + lat * 4), W((bif + cia_end) / 2 + lat * 4), W(cia_end)], 5.0), visible=False,
                  note='Below the lower edge of this CT: drawn to typical adult dimensions.')
        emit_mesh(f'eia-{sd}', f'External iliac artery, {nm} (schematic)', 'vascular', ART, tube([W(cia_end), W((cia_end + eia_end) / 2 + ANT * 10), W(eia_end)], 4.0), visible=False)
        emit_mesh(f'iia-{sd}', f'Internal iliac artery, {nm} (schematic)', 'vascular', ART, tube([W(p) for p in iia], 3.2), visible=False)
        emit_mesh(f'cfa-{sd}', f'Common femoral artery, {nm} (schematic)', 'vascular', ART, tube([W(eia_end), W(cfa_end)], 4.2), visible=False,
                  note='Under the inguinal ligament at the mid-inguinal point, bifurcating into the superficial and profunda femoris arteries.')
        emit_mesh(f'sfa-{sd}', f'Superficial femoral artery, {nm} (schematic)', 'vascular', ART, tube([W(p) for p in sfa], 3.2), visible=False)
        emit_mesh(f'pfa-{sd}', f'Profunda femoris artery, {nm} (schematic)', 'vascular', ART, tube([W(p) for p in pfa], 3.0), visible=False)
        il[sd] = dict(cia=cia_end, eia=eia_end, cfa=cfa_end, cfa_mid=(eia_end + cfa_end) / 2)
        LM[f'cfa-{sd}'] = (eia_end + cfa_end) / 2; LM[f'cia-{sd}'] = (bif + cia_end) / 2
        # the groin incision, over the femoral bifurcation, 1 cm lateral... schematic skin depth 25 mm
        g = (eia_end + cfa_end) / 2 + ANT * 28
        emit_mesh(f'incision-groin-{sd}', f'Groin incision, {nm} (vertical, over the femoral artery)', 'incisions', '#d0433a', tube([W(g + SUP * 30), W(g - SUP * 50)], 1.8), visible=False,
                  note='Schematic (below this scan): a vertical incision over the femoral pulse, from above the inguinal ligament down over the femoral bifurcation.')
        LM[f'groin-{sd}'] = g
        # iliac veins
        iv0 = np.array([LM['ivc'][0], LM['ivc'][1], zlo - 12]) if 'ivc' in LM else bif + RIGHT * 20
        emit_mesh(f'civ-{sd}', f'Common and external iliac veins, {nm} (schematic)', 'vascular', VEIN, tube([W(iv0), W(cia_end - ANT * 10 - lat * 2), W(eia_end - ANT * 5 - lat * 10)], 5.5), opacity=0.85, visible=False,
                  note='Behind and then medial to the arteries; the left common iliac vein crosses behind the right common iliac artery.')
        emit_mesh(f'fv-{sd}', f'Common femoral vein, {nm} (schematic)', 'vascular', VEIN, tube([W(eia_end - ANT * 5 - lat * 10), W(cfa_end - lat * 11 - ANT * 2), W(cfa_end - SUP * 60 - lat * 12)], 5.0), opacity=0.85, visible=False,
                  note='Medial to the femoral artery in the femoral sheath (nerve, artery, vein, empty space, lymphatics: NAVEL from lateral to medial).')
        asis = eia_end + lat * 60 + SUP * 40 + ANT * 22; tub = eia_end - lat * 38 - SUP * 12 + ANT * 22
        emit_mesh(f'inguinal-lig-{sd}', f'Inguinal ligament, {nm} (schematic)', 'vascular', '#d9cfae', tube([W(asis), W((asis + tub) / 2 + ANT * 2 - SUP * 6), W(tub)], 1.6), visible=False,
                  note='From the anterior superior iliac spine to the pubic tubercle; the external iliac artery becomes the common femoral artery beneath it, at the mid-inguinal point.')
        LM[f'asis-{sd}'] = asis
    for sd, s_ in (('l', -1), ('r', 1)):
        if sd not in kid: continue
        lat = RIGHT * s_; h = kid[sd][0]; q = il[sd]['cia']
        up = [h + ANT * 6 - lat * 4, h - SUP * 40 + ANT * 4 - lat * 8, (h + q) / 2 + ANT * 6 - lat * 4, q + ANT * 16 + SUP * 12, q + ANT * 14 - SUP * 14, q - SUP * 50 - ANT * 10 - lat * 6]
        emit_mesh(f'ureter-{sd}', f'Ureter, {"left" if sd == "l" else "right"} (schematic)', 'vascular', '#e8d27a', tube([W(p) for p in up], 2.2), visible=False,
                  note='Down on the psoas, crossing in front of the iliac bifurcation into the pelvis. The limbs of an aortobifemoral graft are tunnelled BEHIND it.')
        LM[f'ureter-{sd}'] = q + ANT * 15
    # ---------------------------------------------------------------- aneurysms (fusiform) on the abdominal aorta
    def sac(z_top, z_bot, rmax, name, id_, note):
        zs = np.linspace(z_top, z_bot, 40); P = [at_z(z) for z in zs[:-4]] + [bif + SUP * (6 - 6 * i / 3) for i in range(4)]
        P = np.array(P); t = np.linspace(0, 1, len(P))
        base = np.array([r_at(z) for z in zs]); prof = base + (rmax - base) * np.sin(np.pi * np.clip((t - 0.05) / 0.9, 0, 1)) ** 0.8
        P[:, 1] += (prof - base) * 0.35                                                             # aneurysms bulge forward and to the left
        P[:, 0] -= (prof - base) * 0.15
        emit_mesh(id_, name, 'vascular', '#c96a5a', _lathe_path(C, P, prof), opacity=0.6, visible=False, note=note)
        return P, prof
    zr = z_ren_lo
    SAC_I = sac(zr - 18, zlo, 30.0, 'Infrarenal abdominal aortic aneurysm (6 cm)', 'aaa-infra',
        'Schematic: a fusiform aneurysm starting 1.5-2 cm below the lowest renal artery (a good neck), to the bifurcation.')
    sac(zr - 2, zlo, 30.0, 'Juxtarenal aneurysm (no infrarenal neck)', 'aaa-juxta',
        'Schematic: the aneurysm begins at the renal arteries without involving them; there is no room for an infrarenal clamp.')
    sac(z_sma - 6, zlo, 31.0, 'Suprarenal (pararenal/paravisceral) aneurysm', 'aaa-supra',
        'Schematic: the aneurysm involves the renal artery origins and extends up to the SMA; the clamp goes above the coeliac trunk or between it and the SMA.')
    LM['aaa'] = at_z((zr + zlo) / 2) + ANT * 10
    # mural thrombus and calcified plaque are not drawn; the sac is shown translucent
    # ---------------------------------------------------------------- aorto-iliac occlusive disease
    occ = [at_z(z) for z in np.linspace(z_ima + 10, zlo, 12)] + [bif]
    parts = [tube([W(p) for p in occ], 7.5)]
    for sd in il: parts.append(tube([W(bif), W((bif + il[sd]['cia']) / 2), W(il[sd]['cia'])], 5.4))
    emit_mesh('aiod-occlusion', 'Aorto-iliac occlusion (thrombus and calcified plaque)', 'vascular', '#e8dcb0', trimesh.util.concatenate(parts), visible=False,
              note='Schematic: the distal aorta and both common iliac arteries occluded (Leriche syndrome, TASC II D), the aorta blocked up to near the IMA.')
    col = []
    for sd, s in (('l', -1), ('r', 1)):
        lat = RIGHT * s
        a = at_z(zr - 25) - ANT * 6 + lat * 8
        col.append(tube([W(a), W(a + lat * 30 - ANT * 20 - SUP * 10), W(il[sd]['cia'] + lat * 30 - ANT * 20), W(il[sd]['eia'] + lat * 20 - ANT * 4)], 1.2))   # lumbar -> iliolumbar / circumflex
        ep = il[sd]['cfa_mid'] + ANT * 12
        col.append(tube([W(ep), W(ep + SUP * 90 + ANT * 12 - lat * 20), W(ep + SUP * 170 + ANT * 14 - lat * 30)], 1.1))  # inferior epigastric -> internal thoracic
    col.append(tube([W(LM['sma'] + ANT * 20 - SUP * 30), W(LM['sma'] + ANT * 30 - SUP * 60 - RIGHT * 30), W(oi + ANT * 12 - RIGHT * 32 - SUP * 40)], 1.3))   # arc of Riolan
    emit_mesh('aiod-collaterals', 'Collaterals around the occlusion', 'vascular', '#d9665a', trimesh.util.concatenate(col), visible=False,
              note='Schematic: lumbar to iliolumbar and circumflex iliac, internal thoracic to inferior epigastric (Winslow pathway), SMA to IMA (arc of Riolan). Do not divide them carelessly.')
    # ---------------------------------------------------------------- grafts and stent grafts
    ntop = at_z(zr - 12); nlow = at_z(zlo + 5)
    tg = [at_z(z) for z in np.linspace(zr - 16, zlo + 2, 20)]
    emit_mesh('graft-tube', 'Tube graft (Dacron), infrarenal', 'vascular', GRAFT, tube([W(p) for p in tg], 9.0, seg=24), visible=False,
              note='Sewn inside the opened sac, from the neck to the bifurcation (or a bifurcated graft to the iliacs), then the sac closed over it.')
    tj = [at_z(z) for z in np.linspace(zr + 2, zlo + 2, 20)]
    emit_mesh('graft-juxta', 'Tube graft, anastomosis at the renal arteries', 'vascular', GRAFT, tube([W(p) for p in tj], 9.0, seg=24), visible=False)
    ts_ = [at_z(z) for z in np.linspace(z_sma - 2, zlo + 2, 24)]
    parts = [tube([W(p) for p in ts_], 9.5, seg=24)]
    if 'l' in kid:
        parts.append(tube([W(at_z(z_ren['l']) - RIGHT * 8), W(at_z(z_ren['l']) - RIGHT * 24), W((at_z(z_ren['l']) + kid['l'][0]) / 2)], 3.2))
    emit_mesh('graft-supra', 'Graft with a bevelled proximal anastomosis incorporating the visceral and right renal origins; left renal side-arm', 'vascular', GRAFT, trimesh.util.concatenate(parts), visible=False,
              note='Schematic: the proximal suture line is bevelled to include the SMA and right renal origins as a patch; the left renal artery is reimplanted or bypassed with a side-arm.')
    # EVAR: main body from below the lowest renal, limbs into both CIAs; stent rings
    body = [at_z(z) for z in np.linspace(zr - 8, zlo + 20, 14)]
    ev = [tube([W(p) for p in body], 11.0, seg=24)]
    for sd in il:
        lp = [at_z(zlo + 20), at_z(zlo + 5), bif, (bif + il[sd]['cia']) / 2, il[sd]['cia'] - SUP * 4]
        ev.append(tube([W(p) for p in lp], 6.5, seg=18))
    rings = []
    for p in body[::2]:
        r_ = trimesh.creation.torus(major_radius=11.5, minor_radius=0.9, major_sections=36, minor_sections=6); r_.apply_translation(W(p)); rings.append(r_)
    emit_mesh('evar-graft', 'Bifurcated stent graft (EVAR)', 'vascular', STENT, trimesh.util.concatenate([*ev, *rings]), opacity=0.8, visible=False,
              note='Schematic: sealing in the infrarenal neck (at least 10-15 mm of healthy, parallel aorta) and in both common iliac arteries; the sac is excluded, not removed.')
    LM['evar-neck'] = at_z(zr - 8)
    # aortobifemoral: end-to-side on the front of the infrarenal aorta just below the renal vein, a short body, limbs tunnelled on the front of
    # the iliac arteries (behind the ureters), under the inguinal ligaments, end-to-side onto the front of each common femoral artery
    def crimped(P, r, seg=20, pitch=3.0):
        Q = _cr(P); parts = [tube([W(p) for p in P], r, seg=seg)]
        T_, _, _ = _frames(Q); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        for L in np.arange(pitch, d[-1] - pitch, pitch):
            k = int(np.searchsorted(d, L)); t_ = trimesh.creation.torus(major_radius=r, minor_radius=0.45, major_sections=max(seg, 16), minor_sections=5)
            M = trimesh.geometry.align_vectors([0, 0, 1], T_[k]); M[:3, 3] = W(Q[k]); t_.apply_transform(M); parts.append(t_)
        return trimesh.util.concatenate(parts)
    top = at_z(zr - 12); ra = r_at(zr - 12)
    hood = top + ANT * (ra + 1.5)                                                                     # on the front wall of the aorta
    bb = at_z(zr - 58) + ANT * (r_at(zr - 58) + 14)                                                  # the graft bifurcation, in front of the occluded aorta
    body = [hood - SUP * 2, hood + ANT * 5 - SUP * 12, top + ANT * (ra + 11) - SUP * 28, bb]
    abf = [crimped(body, 7.5)]
    for sd, s_ in (('l', -1), ('r', 1)):
        lat = RIGHT * s_; q = il[sd]; cdir = (q['cfa'] - q['eia']) / np.linalg.norm(q['cfa'] - q['eia'])
        limb = [bb, bb + lat * 14 - SUP * 18 + ANT * 2, (bif + q['cia']) / 2 + ANT * 13 + lat * 3, q['cia'] + ANT * 11 + lat * 4,
                (q['cia'] + q['eia']) / 2 + ANT * 20 + lat * 3, q['eia'] + ANT * 12, q['eia'] + cdir * 10 + ANT * 8, q['cfa_mid'] + ANT * 4.5]
        abf.append(crimped(limb, 4.0, seg=16))
        LM[f'abf-fem-{sd}'] = q['cfa_mid']
    emit_mesh('graft-abf', 'Aortobifemoral bypass graft (bifurcated, crimped Dacron)', 'vascular', GRAFT, trimesh.util.concatenate(abf), visible=False,
              note='End-to-side on the front of the infrarenal aorta below the left renal vein; a short body; each limb in a retroperitoneal tunnel on the front of the iliac arteries, behind the ureter, under the inguinal ligament, end-to-side onto the common femoral artery.')
    LM['abf-prox'] = top
    # kissing stents: two balloon-expandable stents side by side from the distal aorta into both common iliacs, their tops level (the new carina)
    def lattice(P, r, cell=6.0, zig=10, strut=0.35):
        Q = _cr(P, 1.0); T_, U_, V_ = _frames(Q); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        at = lambda L: int(min(np.searchsorted(d, L), len(Q) - 1))
        pt = lambda L, a: (lambda k: Q[k] + (U_[k] * np.cos(a) + V_[k] * np.sin(a)) * r)(at(L))
        segs = []; rows = np.arange(0, d[-1] + 1e-6, cell / 2)
        for m, L in enumerate(rows):
            for n in range(zig):
                a0 = 2 * np.pi * n / zig; a1 = 2 * np.pi * (n + 0.5) / zig; a2 = 2 * np.pi * (n + 1) / zig
                if m + 1 < len(rows):
                    up = rows[m + 1]
                    if m % 2 == 0: segs += [(pt(L, a0), pt(up, a1)), (pt(up, a1), pt(L, a2))]
                    else: segs += [(pt(L, a1), pt(up, a0 if n else 0)), (pt(L, a1), pt(up, a2))]
        cyl = []
        for a, b in segs:
            L = np.linalg.norm(b - a)
            if L < 0.2: continue
            c = trimesh.creation.cylinder(radius=strut, height=L, sections=5)
            M = trimesh.geometry.align_vectors([0, 0, 1], (b - a) / L); M[:3, 3] = W((a + b) / 2); c.apply_transform(M); cyl.append(c)
        return trimesh.util.concatenate(cyl)
    ks_, kb = [], []
    for sd, s_ in (('l', -1), ('r', 1)):
        q = il[sd]; off = RIGHT * s_ * 4.2
        P = [at_z(zlo + 14) + off, at_z(zlo + 4) + off, bif + off * 1.2, (bif + q['cia']) / 2, q['cia'] + SUP * 10]
        ks_.append(lattice(P, 4.0)); ks_.append(tube([W(p) for p in P], 3.85, seg=18))            # struts over a thin PTFE cover (covered stent)
        kb.append(tube([W(P[0] + SUP * 5)] + [W(p) for p in P] + [W(q['cia'] + SUP * 4)], 4.3, seg=20))
    emit_mesh('stents-kissing', 'Kissing stents at the aortic bifurcation (covered, balloon-expandable)', 'vascular', STENT, trimesh.util.concatenate(ks_), visible=False,
              note='Two balloon-expandable covered stents side by side from the distal aorta into both common iliac arteries, their tops level about 1 cm above the old bifurcation (the new carina), inflated together.')
    emit_mesh('kissing-balloons', 'Two balloons inflated together (kissing inflation)', 'vascular', '#6fa8dc', trimesh.util.concatenate(kb), opacity=0.45, visible=False,
              note='Simultaneous inflation so that neither stent crushes the other at the carina.')
    LM['kiss-top'] = at_z(zlo + 14)
    # axillobifemoral: end-to-side on the first part of the right axillary artery, along it for a few cm, then down the mid-axillary line
    # under the skin of the chest and flank (outside the ribs), in front of the iliac crest, to the right common femoral; a suprapubic
    # femorofemoral limb to the left
    sca = ts('subclavian_artery_right')
    if sca.any():
        emit('sca-r', 'Subclavian artery, right', 'vascular', ART, sca, AT, faces=2500, visible=False)
        s0 = vox_mm(sca); e = s0[np.argmax(s0[:, 0])]                                                 # the lateral end of the subclavian (outer border of the first rib)
        axa = [e, e + RIGHT * 22 - SUP * 6 + ANT * 2, e + RIGHT * 45 - SUP * 16, e + RIGHT * 70 - SUP * 32 - ANT * 4]
        emit_mesh('axillary-a-r', 'Axillary artery, right (first part medial to pectoralis minor)', 'vascular', ART, tube([W(p) for p in axa], 3.6), visible=False,
                  note='Schematic continuation of the subclavian from the outer border of the first rib; its first part, behind the clavipectoral fascia, takes the graft.')
        ax = e + RIGHT * 24 - SUP * 7 + ANT * 2
        LM['axillary-r'] = ax
        rib = {}
        for n in range(3, 12):
            m = ts(f'rib_right_{n}')
            if m.any():
                q = vox_mm(m); rib[n] = q
        wall = np.concatenate(list(rib.values())) if rib else None
        def chest_wall(z):
            sl = wall[np.abs(wall[:, 2] - z) < 4]
            if not len(sl): return None
            q = sl[np.argmax(sl[:, 0])]; return q + RIGHT * 11 + ANT * 4                               # just outside the ribs: the subcutaneous plane
        pts = [ax - RIGHT * 4, ax + RIGHT * 14 - SUP * 2 + ANT * 6, ax + RIGHT * 30 - SUP * 18 + ANT * 8]
        zc = [z for z in np.linspace(ax[2] - 45, zlo + 5, 9)]
        for z in zc:
            q = chest_wall(z) if wall is not None else None
            if q is not None and (not len(pts) or q[2] < pts[-1][2] - 5): pts.append(q)
        last = pts[-1]; qr = il['r']; asis = LM['asis-r']
        pts += [np.array([last[0] - 6, last[1] + 12, (last[2] + asis[2]) / 2]), asis + SUP * 18 - RIGHT * 16 + ANT * 6, qr['eia'] + ANT * 24 + RIGHT * 10, qr['cfa_mid'] + ANT * 14, qr['cfa_mid'] + ANT * 4.5]
        pts = [p for k_, p in enumerate(pts) if k_ == 0 or np.linalg.norm(p - pts[k_ - 1]) > 4]
        cross = [qr['cfa_mid'] + ANT * 4.5, qr['cfa_mid'] + ANT * 16 - RIGHT * 6 + SUP * 6, (qr['cfa_mid'] + il['l']['cfa_mid']) / 2 + ANT * 34 + SUP * 22,
                 il['l']['cfa_mid'] + ANT * 16 + RIGHT * 6 + SUP * 6, il['l']['cfa_mid'] + ANT * 4.5]
        def ringed(P, r):
            Q = _cr(P); T_, _, _ = _frames(Q); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]; parts = [tube([W(p) for p in P], r, seg=16)]
            for L in np.arange(8, d[-1] - 8, 8.0):
                k = int(np.searchsorted(d, L)); t_ = trimesh.creation.torus(major_radius=r + 0.3, minor_radius=0.6, major_sections=18, minor_sections=5)
                M = trimesh.geometry.align_vectors([0, 0, 1], T_[k]); M[:3, 3] = W(Q[k]); t_.apply_transform(M); parts.append(t_)
            return trimesh.util.concatenate(parts)
        emit_mesh('graft-axbf', 'Axillobifemoral bypass (8 mm ringed PTFE), with a femorofemoral cross-over', 'vascular', GRAFT, trimesh.util.concatenate([ringed(pts, 4.0), ringed(cross, 4.0)]), visible=False,
                  note='Extra-anatomic: end-to-side on the first part of the axillary artery, running along it before turning down, subcutaneous in the mid-axillary line outside the ribs, in front of the iliac crest to the right groin; a suprapubic subcutaneous femorofemoral limb to the left groin.')
        axr = [ring(ax + ANT * 0.5, ANT * 0.3 - SUP * 1.0, 4.6)]
        for sd in il: axr.append(ring(il[sd]['cfa_mid'] + ANT * 4.8, ANT + SUP * 0.5, 4.8))
        emit_mesh('anast-axbf', 'Suture lines: axillary and both femoral anastomoses', 'vascular', '#3fa7d6', trimesh.util.concatenate(axr), visible=False)
        emit_mesh('incision-axillary-r', 'Infraclavicular incision (right axillary artery)', 'incisions', '#d0433a',
                  tube([W(ax + ANT * 30 - RIGHT * 28 + SUP * 12), W(ax + ANT * 32 + SUP * 10), W(ax + ANT * 30 + RIGHT * 28 + SUP * 6)], 1.8), visible=False)
        LM['axbf-mid'] = pts[len(pts) // 2]
    skin = ctx.get('skin_mm')
    # ---------------------------------------------------------------- incisions: midline laparotomy, left flank (retroperitoneal)
    if skin is not None:
        mid_x = float(D[:, 0].mean()) + 30                                                         # the body's midline (the aorta lies to the left of it)
        xm = float(np.median(skin[:, 0]))
        lap = []
        for z in np.linspace(z_coel + 30, zlo - 5, 10):
            sl = skin[(np.abs(skin[:, 2] - z) < 3) & (np.abs(skin[:, 0] - xm) < 12)]
            if len(sl): lap.append(sl[np.argmax(sl[:, 1])] + ANT * 1)
        if len(lap) > 3:
            emit_mesh('incision-laparotomy', 'Midline laparotomy', 'incisions', '#d0433a', tube([W(p) for p in lap], 1.8), visible=False)
            LM['laparotomy'] = lap[len(lap) // 2]
        fl = []
        for i, z in enumerate(np.linspace(z_t12 if z_t12 else z_coel, zlo + 10, 7)):
            sl = skin[np.abs(skin[:, 2] - z) < 3]
            if not len(sl): continue
            th = np.radians(200 - i * 22)                                                           # from the posterior flank (11th rib) round to the left lower quadrant
            d = np.array([np.cos(th), np.sin(th), 0])
            c0 = np.array([xm, float(np.median(sl[:, 1])), z])
            q = sl[np.argmax((sl - c0) @ d)]; fl.append(q)
        if len(fl) > 3:
            emit_mesh('incision-flank-l', 'Left flank incision (retroperitoneal approach, through the 11th rib bed)', 'incisions', '#d0433a', tube([W(p) for p in fl], 1.8), visible=False,
                      note='From the tip of the 11th rib posteriorly, obliquely down to the lateral edge of the rectus sheath; the peritoneum and the left kidney swept forward (or the kidney left behind).')
            LM['flank-l'] = fl[len(fl) // 2]
    # ---------------------------------------------------------------- thoracic aorta: descending aneurysm, TEVAR, open graft; ascending aneurysm
    dz0 = float(D[:, 2].max())
    z_hi, z_lo = dz0 - 45, dz0 - 135
    zs = np.linspace(z_hi, z_lo, 36); P = np.array([at_z(z) for z in zs]); base = np.array([r_at(z) for z in zs]); t = np.linspace(0, 1, len(zs))
    prof = base + (32.0 - base) * np.sin(np.pi * t) ** 0.9
    P2 = P.copy(); P2[:, 0] -= (prof - base) * 0.3; P2[:, 1] += (prof - base) * 0.1
    emit_mesh('taa-desc', 'Descending thoracic aortic aneurysm (6.4 cm)', 'vascular', '#c96a5a', _lathe_path(C, P2, prof), opacity=0.6, visible=False,
              note='Schematic: a fusiform aneurysm of the mid-descending thoracic aorta, beginning several centimetres beyond the left subclavian artery.')
    LM['taa-desc'] = at_z((z_hi + z_lo) / 2)
    zt = np.linspace(z_hi + 30, z_lo - 30, 26); tp = [at_z(z) for z in zt]
    tv = [tube([W(p) for p in tp], float(np.median(base)) + 1.5, seg=24)]
    for p in tp[::3]:
        r_ = trimesh.creation.torus(major_radius=float(np.median(base)) + 2.0, minor_radius=0.9, major_sections=36, minor_sections=6)
        k = np.argmin(np.linalg.norm(np.array(tp) - p, axis=1)); d_ = tp[min(k + 1, len(tp) - 1)] - tp[max(k - 1, 0)]
        T_ = trimesh.geometry.align_vectors([0, 0, 1], d_ / np.linalg.norm(d_)); T_[:3, 3] = W(p); r_.apply_transform(T_); tv.append(r_)
    emit_mesh('tevar-graft', 'Thoracic stent graft (TEVAR)', 'vascular', STENT, trimesh.util.concatenate(tv), opacity=0.8, visible=False,
              note='Schematic: at least 20 mm of healthy aorta above and below (the landing zones); the sac excluded.')
    emit_mesh('graft-taa', 'Interposition graft (Dacron), descending thoracic aorta', 'vascular', GRAFT, tube([W(at_z(z)) for z in np.linspace(z_hi + 6, z_lo - 6, 20)], float(np.median(base)) + 1, seg=24), visible=False)
    LM['taa-prox'] = at_z(z_hi + 12); LM['taa-dist'] = at_z(z_lo - 12)
    # spinal cord supply: an intercostal at T9-T11 on the left giving the artery of Adamkiewicz; a lumbar CSF drain
    zi = (z_t12 + 45) if z_t12 else z_lo - 30
    o = at_z(zi); v = ts('vertebrae_T10'); vc = vox_mm(v).mean(0) if v.any() else o - ANT * 30 + RIGHT * 30
    canal = vc - ANT * 12
    emit_mesh('adamkiewicz', 'Left intercostal artery and the artery of Adamkiewicz (arteria radicularis magna)', 'vascular', ART,
              tube([W(o - ANT * 6), W(o - ANT * 18 - RIGHT * 6), W(canal + SUP * 25 - RIGHT * 6), W(canal + SUP * 40), W(canal + SUP * 20)], 1.3), visible=False,
              note='Schematic: most often from a left intercostal artery between T9 and T12; it joins the anterior spinal artery with a hairpin turn and is the main supply to the lower cord.')
    LM['adamkiewicz'] = o
    l3 = ts('vertebrae_L3')
    if l3.any():
        c3 = vox_mm(l3); top3 = c3[c3[:, 2] > np.percentile(c3[:, 2], 85)].mean(0); skn = top3 - ANT * 70
        emit_mesh('csf-drain', 'Lumbar CSF drain (L2-L3 or L3-L4)', 'vascular', '#e6e2da', tube([W(skn), W(top3 - ANT * 18), W(top3 - ANT * 16 + SUP * 40)], 1.0), visible=False,
                  note='Drains CSF to keep the pressure at about 10-12 mmHg (not higher than 15), raising spinal cord perfusion pressure (MAP minus CSF pressure).')
    # ascending aneurysm
    if len(asc) > 5:
        A_ = np.array([c[0] for c in asc]); Ar = np.array([c[1] for c in asc])
        A_ = np.array([A_[max(0, i - 3):i + 4].mean(0) for i in range(len(A_))])
        t = np.linspace(0, 1, len(A_)); prof = Ar + (28.0 - Ar) * np.sin(np.pi * np.clip(t * 1.1, 0, 1)) ** 0.7
        emit_mesh('taa-asc', 'Ascending aortic aneurysm (5.6 cm)', 'vascular', '#c96a5a', _lathe_path(C, A_, prof), opacity=0.55, visible=False,
                  note='Schematic: a supracoronary ascending aneurysm with a normal root and sinotubular junction, ending at the innominate artery.')
        emit_mesh('graft-asc', 'Ascending aorta and hemiarch graft (Dacron)', 'vascular', GRAFT, tube([W(p) for p in A_[1:]], float(np.median(Ar)) + 1.5, seg=24), visible=False)
        LM['taa-asc'] = A_[len(A_) // 2]

    # ================================================================ for depicting the operations
    # the native aorta with the diseased segment taken out, so that a sac, graft or stent graft is not drawn around a normal aorta
    kz = lambda z: int(round((z - AT[2, 3]) / AT[2, 2]))
    def cut(id_, name, drop):
        m = ao.copy(); drop(m)
        emit(id_, name, 'vascular', '#d0433a', m, AT, faces=9000, visible=False, note='The aorta from the CT, the diseased segment removed (it is drawn separately).')
    def below(z):
        def f(m):
            k = kz(z)
            if AT[2, 2] > 0: m[..., :max(k, 0)] = False
            else: m[..., k:] = False
        return f
    cut('aorta-cut-infra', 'Aorta above the infrarenal neck', below(zr - 16))
    cut('aorta-cut-juxta', 'Aorta above the renal arteries', below(zr - 1))
    cut('aorta-cut-supra', 'Aorta above the visceral segment', below(z_sma - 5))
    ij = np.argwhere(ao); xyz = ij @ AT[:3, :3].T + AT[:3, 3]
    def near_line(Q, rad, zmin, zmax):
        sel = (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
        idx = np.where(sel)[0]
        Qz = Q[np.argsort(Q[:, 2])]
        cx = np.interp(xyz[idx, 2], Qz[:, 2], Qz[:, 0]); cy = np.interp(xyz[idx, 2], Qz[:, 2], Qz[:, 1])
        close = np.hypot(xyz[idx, 0] - cx, xyz[idx, 1] - cy) < rad
        return ij[idx[close]]
    def drop_vox(v):
        def f(m):
            m[v[:, 0], v[:, 1], v[:, 2]] = False
        return f
    cut('aorta-cut-desc', 'Aorta with the aneurysmal descending segment removed', drop_vox(near_line(D, 28.0, z_lo + 2, z_hi - 2)))
    if len(asc) > 5:
        A0 = np.array([c[0] for c in asc])
        cut('aorta-cut-asc', 'Aorta with the ascending segment removed', drop_vox(near_line(A0, 26.0, A0[:, 2].min() + 3, A0[:, 2].max() - 3)))
    # the infrarenal sac opened, its laminated thrombus, and the lumbar arteries that back-bleed into it
    Ps, prof = SAC_I
    Tn = np.gradient(Ps, axis=0); Tn /= np.linalg.norm(Tn, axis=1, keepdims=True) + 1e-9
    def lathe_sector(P, radii, a0, a1, secs=28):
        V, F = [], []
        for i in range(len(P)):
            t_ = Tn[i]; u = ANT - t_ * np.dot(ANT, t_); u /= np.linalg.norm(u) + 1e-9; v = np.cross(t_, u)
            for a in np.linspace(a0, a1, secs): V.append(P[i] + (u * np.cos(a) + v * np.sin(a)) * radii[i])
        for i in range(len(P) - 1):
            for j in range(secs - 1):
                a, b = i * secs + j, i * secs + j + 1
                F += [[a, a + secs, b + secs], [a, b + secs, b], [a, b + secs, a + secs], [a, b, b + secs]]    # both sides
        return trimesh.Trimesh(np.array(V) - C, np.array(F), process=True)
    emit_mesh('aaa-sac-open', 'Aneurysm sac, opened longitudinally', 'vascular', '#c96a5a', lathe_sector(Ps, prof, np.radians(55), np.radians(305)), opacity=0.75, visible=False,
              note='The front of the sac opened (to the right of the IMA), the thrombus scooped out; the walls fall back and later close over the graft.')
    emit_mesh('aaa-thrombus', 'Laminated mural thrombus in the sac', 'vascular', '#7d3a2c', _lathe_path(C, Ps, np.maximum(prof * 0.84, [r_at(min(max(z, zlo), zr)) + 1.0 for z in Ps[:, 2]])), opacity=0.9, visible=False,
              note='Layers of old thrombus line most aneurysms, leaving a channel of normal calibre: the sac is far larger than the lumen seen on angiography.')
    lum = []
    for z in np.linspace(zr - 30, zlo + 12, 3):
        o = at_z(z)
        for s_ in (-1, 1):
            lum.append(tube([W(o - ANT * r_at(z) * 0.8 + RIGHT * s_ * 4), W(o - ANT * 18 + RIGHT * s_ * 14), W(o - ANT * 26 + RIGHT * s_ * 30)], 1.4))
    emit_mesh('lumbar-arteries', 'Lumbar arteries (back-bleeding into the opened sac)', 'vascular', ART, trimesh.util.concatenate(lum), visible=False,
              note='Paired from the back of the aorta; after the sac is opened they are oversewn from inside with figure-of-eight sutures.')
    # suture lines (anastomoses)
    dvec = lambda z: at_z(z + 3) - at_z(z - 3)
    emit_mesh('anast-aaa', 'Suture lines: proximal (neck) and distal (bifurcation) anastomoses', 'vascular', '#3fa7d6',
              trimesh.util.concatenate([ring(at_z(zr - 16), dvec(zr - 16), 10.0), ring(at_z(zlo + 2), dvec(zlo + 2), 10.0)]), visible=False,
              note='Running 3-0 polypropylene, the back wall first from inside the sac, taking the full thickness of the aorta (and a strip of felt if it is friable).')
    emit_mesh('anast-juxta', 'Suture line at the renal arteries', 'vascular', '#3fa7d6', ring(at_z(zr + 1), dvec(zr + 1), 10.0), visible=False)
    emit_mesh('anast-supra', 'Bevelled proximal suture line (visceral patch)', 'vascular', '#3fa7d6', ring(at_z(z_sma - 3), dvec(z_sma - 3) + ANT * 3, 11.0), visible=False)
    ab = [ring(hood + ANT * 0.5 - SUP * 4, ANT + SUP * 0.6, 8.5)]
    for sd in il: ab.append(ring(il[sd]['cfa_mid'] + ANT * 4.8, ANT + SUP * 0.5, 4.8))
    emit_mesh('anast-abf', 'Suture lines: aortic (end-to-side) and both femoral anastomoses', 'vascular', '#3fa7d6', trimesh.util.concatenate(ab), visible=False)
    emit_mesh('anast-taa', 'Suture lines: proximal and distal thoracic anastomoses', 'vascular', '#3fa7d6',
              trimesh.util.concatenate([ring(at_z(z_hi + 6), dvec(z_hi + 6), float(np.median(base)) + 2), ring(at_z(z_lo - 6), dvec(z_lo - 6), float(np.median(base)) + 2)]), visible=False)
    # guidewires and sheaths: from both groins, up the iliacs into the aorta
    def up_path(sd, z_end):
        q = il[sd]; P_ = [LM[f'groin-{sd}'] - SUP * 10, q['cfa_mid'], q['eia'], q['cia'], (bif + q['cia']) / 2, bif]
        P_ += [at_z(z) for z in np.linspace(zlo + 5, z_end, 12)]
        return P_
    wires = []
    for sd in il:
        P_ = up_path(sd, z_coel + 40)
        wires.append(tube([W(p + RIGHT * (1.5 if sd == 'r' else -1.5)) for p in P_], 0.6, seg=8))
        wires.append(tube([W(p) for p in P_[:4]], 3.2, seg=12))                                     # the sheath in the femoral and iliac
    emit_mesh('evar-wires', 'Stiff guidewires and sheaths (both femoral arteries)', 'vascular', '#9aa6b2', trimesh.util.concatenate(wires), visible=False,
              note='Percutaneous (pre-closure sutures) or open femoral access; stiff wires up to the thoracic aorta; the main body travels up the ipsilateral side, the contralateral limb through a gate cannulated from the other groin.')
    Dtop = D[np.argmax(D[:, 2])]
    tw = up_path('r', float(D[:, 2].max()) - 5) + [Dtop + SUP * 10 + ANT * 10]
    emit_mesh('tevar-wire', 'Stiff guidewire and sheath (right femoral) to the arch', 'vascular', '#9aa6b2',
              trimesh.util.concatenate([tube([W(p) for p in tw], 0.7, seg=8), tube([W(p) for p in tw[:4]], 3.6, seg=12)]), visible=False)
    kw = [tube([W(p + RIGHT * (1.2 if sd == 'r' else -1.2)) for p in up_path(sd, zlo + 70)], 0.9, seg=8) for sd in il]
    kw += [tube([W(p) for p in up_path(sd, zlo + 70)[:3]], 2.6, seg=12) for sd in il]                  # 7F sheaths in both femorals
    emit_mesh('kissing-wires', 'Sheaths in both femorals; guidewires crossing both iliac occlusions', 'vascular', '#9aa6b2', trimesh.util.concatenate(kw), visible=False)
    LM['aaa-neck'] = at_z(zr - 16); LM['aaa-bottom'] = at_z(zlo + 2)
    return LM
