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


def build(ctx):
    ts, vox_mm, emit, emit_mesh, W, tube, sphere, C = ctx['ts'], ctx['vox_mm'], ctx['emit'], ctx['emit_mesh'], ctx['W'], ctx['tube'], ctx['sphere'], ctx['CARINA']
    AT = ctx['AT']; LM = {}
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
        emit_mesh(f'civ-{sd}', f'Common iliac vein, {nm} (schematic)', 'vascular', VEIN, tube([W(iv0), W(cia_end - ANT * 10 + lat * 4), W(eia_end - ANT * 6 + lat * 12)], 5.5), opacity=0.85, visible=False)
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
    # aortobifemoral: end-to-side high on the infrarenal aorta, limbs through retroperitoneal tunnels to the femorals
    top = at_z(zr - 12); bb = at_z(zr - 40) + ANT * 16
    abf = [tube([W(top + ANT * r_at(zr - 12) * 0.9), W(top + ANT * 16 - SUP * 12), W(bb)], 7.0, seg=20)]
    for sd, s in (('l', -1), ('r', 1)):
        lat = RIGHT * s
        abf.append(tube([W(bb), W(bb + lat * 20 - SUP * 30 + ANT * 4), W(il[sd]['cia'] + ANT * 18 + lat * 6), W(il[sd]['eia'] + ANT * 8), W(il[sd]['cfa_mid'] + ANT * 6)], 4.5, seg=16))
    emit_mesh('graft-abf', 'Aortobifemoral bypass graft (bifurcated Dacron)', 'vascular', GRAFT, trimesh.util.concatenate(abf), visible=False,
              note='Proximal anastomosis just below the renal arteries (end-to-end or end-to-side), limbs tunnelled behind the peritoneum and under the inguinal ligaments along the iliac arteries, anterior to them, to the common femoral arteries.')
    LM['abf-prox'] = top
    # kissing stents (endovascular reconstruction of the bifurcation)
    ks_ = []
    for sd in il:
        ks_.append(tube([W(at_z(zlo + 20) + RIGHT * (3 if sd == 'r' else -3)), W(bif + RIGHT * (3 if sd == 'r' else -3)), W((bif + il[sd]['cia']) / 2), W(il[sd]['cia'])], 4.2, seg=14))
    emit_mesh('stents-kissing', 'Kissing stents at the aortic bifurcation', 'vascular', STENT, trimesh.util.concatenate(ks_), opacity=0.85, visible=False,
              note='Two balloon-expandable (often covered) stents deployed side by side from the distal aorta into both common iliac arteries, inflated together.')
    # axillobifemoral: right axillary artery, down the lateral chest wall under the skin, to the right groin, cross-over to the left
    skin = ctx.get('skin_mm'); sca = ts('subclavian_artery_right')
    if skin is not None and sca.any():
        ax0 = vox_mm(sca); ax = ax0[np.argmax(ax0[:, 0])]                                            # the lateral end of the subclavian: the first part of the axillary
        ax = ax + RIGHT * 22 - SUP * 6
        pts = [ax]
        for z in np.linspace(ax[2] - 40, zlo, 8):
            sl = skin[np.abs(skin[:, 2] - z) < 4]
            if not len(sl): continue
            q = sl[np.argmax(sl[:, 0] + sl[:, 1] * 0.3)]                                              # the right anterolateral surface
            pts.append(q - (q - np.array([D[:, 0].mean(), D[:, 1].mean(), q[2]])) / np.linalg.norm(q - np.array([D[:, 0].mean(), D[:, 1].mean(), q[2]])) * 10)
        pts += [il['r']['cfa_mid'] + ANT * 20 + RIGHT * 14, il['r']['cfa_mid'] + ANT * 6]
        axbf = [tube([W(p) for p in pts], 4.0, seg=16),
                tube([W(il['r']['cfa_mid'] + ANT * 18), W((il['r']['cfa_mid'] + il['l']['cfa_mid']) / 2 + ANT * 30 + SUP * 25), W(il['l']['cfa_mid'] + ANT * 6)], 3.8, seg=16)]
        emit_mesh('graft-axbf', 'Axillobifemoral bypass (ringed PTFE), with a femorofemoral cross-over', 'vascular', GRAFT, trimesh.util.concatenate(axbf), visible=False,
                  note='Extra-anatomic: from the first part of the axillary artery, tunnelled subcutaneously down the mid-axillary line to the groin, with a suprapubic femorofemoral limb.')
        emit_mesh('incision-axillary-r', 'Infraclavicular incision (right axillary artery)', 'incisions', '#d0433a',
                  tube([W(ax + ANT * 26 - RIGHT * 30 + SUP * 4), W(ax + ANT * 28 + SUP * 2), W(ax + ANT * 26 + RIGHT * 25)], 1.8), visible=False)
        LM['axillary-r'] = ax
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
    def ring(p, d, r, rr=1.2):
        t_ = trimesh.creation.torus(major_radius=r, minor_radius=rr, major_sections=40, minor_sections=8)
        T_ = trimesh.geometry.align_vectors([0, 0, 1], d / np.linalg.norm(d)); T_[:3, 3] = W(p); t_.apply_transform(T_); return t_
    dvec = lambda z: at_z(z + 3) - at_z(z - 3)
    emit_mesh('anast-aaa', 'Suture lines: proximal (neck) and distal (bifurcation) anastomoses', 'vascular', '#3fa7d6',
              trimesh.util.concatenate([ring(at_z(zr - 16), dvec(zr - 16), 10.0), ring(at_z(zlo + 2), dvec(zlo + 2), 10.0)]), visible=False,
              note='Running 3-0 polypropylene, the back wall first from inside the sac, taking the full thickness of the aorta (and a strip of felt if it is friable).')
    emit_mesh('anast-juxta', 'Suture line at the renal arteries', 'vascular', '#3fa7d6', ring(at_z(zr + 1), dvec(zr + 1), 10.0), visible=False)
    emit_mesh('anast-supra', 'Bevelled proximal suture line (visceral patch)', 'vascular', '#3fa7d6', ring(at_z(z_sma - 3), dvec(z_sma - 3) + ANT * 3, 11.0), visible=False)
    ab = [ring(top + ANT * (r_at(zr - 12) * 0.9 + 1), ANT + SUP * 0.3, 7.5)]
    for sd in il: ab.append(ring(il[sd]['cfa_mid'] + ANT * 6, ANT + SUP * 0.2, 5.0))
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
    kw = [tube([W(p + RIGHT * (1.2 if sd == 'r' else -1.2)) for p in up_path(sd, zlo + 70)], 0.6, seg=8) for sd in il]
    emit_mesh('kissing-wires', 'Guidewires crossing both iliac occlusions (from both groins)', 'vascular', '#9aa6b2', trimesh.util.concatenate(kw), visible=False)
    LM['aaa-neck'] = at_z(zr - 16); LM['aaa-bottom'] = at_z(zlo + 2)
    return LM
