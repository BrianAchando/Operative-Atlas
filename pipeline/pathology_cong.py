"""Meshes for the congenital cardiac series: secundum atrial septal defect, perimembranous ventricular septal defect,
patent ductus arteriosus and coarctation of the aorta, with left posterolateral thoracotomy incisions in the 3rd and 4th
intercostal spaces.

The heart, aorta and pulmonary arteries are the adult reference CT. The defects, the duct, the coarctation and the repairs
are SCHEMATIC, placed on that anatomy (the fossa ovalis, the membranous septum, the ligamentum arteriosum, the aortic
isthmus) and drawn to typical dimensions. A child's heart is smaller, but the relations are the same.
Coordinates: meshes in the atlas frame (millimetres, carina at the origin); returned landmarks in scanner millimetres.
"""
from __future__ import annotations

import numpy as np
import trimesh

from pathology_new import lathe, ring, disc, spline

ANT, SUP, RIGHT = np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
LEFT = -RIGHT
ART, DEFECT, PATCH, DACRON, SUTURE, SILK, STEEL, FIBRE, INC = '#c0392b', '#f2b84b', '#e8dcc0', '#f1f1ea', '#3fa7d6', '#2d2d2d', '#9aa6b2', '#d8c9ae', '#d0433a'
U = lambda v: np.asarray(v, float) / (np.linalg.norm(v) + 1e-9)


def cone(base, tip, r):
    """an arrow head: a cone from base to tip (atlas frame)"""
    base, tip = np.asarray(base, float), np.asarray(tip, float); h = float(np.linalg.norm(tip - base))
    c = trimesh.creation.cone(radius=r, height=h, sections=24)
    M = trimesh.geometry.align_vectors([0, 0, 1], U(tip - base)); M[:3, 3] = base; c.apply_transform(M); return c


def arrow(a, b, r):
    """a flow arrow from a to b: shaft and head"""
    a, b = np.asarray(a, float), np.asarray(b, float); d = U(b - a); L = np.linalg.norm(b - a)
    shaft = trimesh.creation.cylinder(radius=r * 0.45, segment=[a, a + d * L * 0.62], sections=16)
    return trimesh.util.concatenate([shaft, cone(a + d * L * 0.6, b, r)])


def bar(p, d, length, w=3.0, t=2.0, up=SUP):
    """a clamp jaw: a flat bar centred on p, long axis d"""
    b = trimesh.creation.box(extents=[length, w, t])
    x = U(d); z = U(np.cross(x, up)) if abs(np.dot(x, U(up))) < 0.95 else U(np.cross(x, ANT)); y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, p; b.apply_transform(M); return b


def build(ctx):
    emit_mesh, W, tube = ctx['emit_mesh'], ctx['W'], ctx['tube']
    S = ctx['S']; C = np.asarray(ctx['CARINA'], float); LM = {}
    c = lambda i: np.array(S[i]['centroid'], float)
    print('== congenital: ASD, VSD, PDA, coarctation')

    # ============================================================ secundum ASD (fossa ovalis)
    if all(i in S for i in ('septal-incision', 'ra', 'la')):
        n = U(c('la') - c('ra'))                                # septal normal, right atrium to left
        ctr = c('septal-incision')
        up = U(SUP - n * np.dot(SUP, n))
        emit_mesh('asd-defect', 'Secundum atrial septal defect (fossa ovalis, about 18 mm)', 'congenital', DEFECT,
                  trimesh.util.concatenate([ring(ctr, n, 9.0, 1.4), disc(ctr, n, 8.0, 8.0, th=0.6)]), opacity=0.9, visible=False,
                  note='Schematic. A deficiency of the floor of the fossa ovalis (septum primum). The rim: superior toward the SVC and aorta, inferior toward the IVC, posterior toward the right pulmonary veins.')
        emit_mesh('asd-shunt', 'Left-to-right shunt across the ASD (mostly in late diastole)', 'congenital', ART,
                  arrow(ctr + n * 16 + up * 2, ctr - n * 16 - up * 2, 3.2), opacity=0.85, visible=False,
                  note='Schematic. Flow follows the compliance of the ventricles: the thinner, more compliant RV fills preferentially, so blood crosses from LA to RA.')
        emit_mesh('asd-patch', 'Autologous pericardial patch (ASD closure)', 'congenital', PATCH,
                  disc(ctr - n * 1.2, n, 12.0, 11.0, th=0.8), visible=False,
                  note='Schematic. Fresh or glutaraldehyde-treated autologous pericardium, sewn to the rim with running 5-0 polypropylene; small defects may be closed directly.')
        emit_mesh('asd-suture', 'Running polypropylene suture line (ASD patch)', 'congenital', SUTURE,
                  ring(ctr - n * 1.6, n, 10.4, 0.55), visible=False)
        LM['asd-c'] = ctr + C

    # ============================================================ perimembranous VSD (below the aortic valve, under the septal leaflet)
    if all(i in S for i in ('his-bundle', 'lv', 'rv', 'lvot')):
        n = U(c('rv') - c('lv'))                                # septal normal, LV to RV
        ctr = c('his-bundle') + n * 2.0 - SUP * 1.5
        emit_mesh('vsd-defect', 'Perimembranous ventricular septal defect (about 10 mm)', 'congenital', DEFECT,
                  trimesh.util.concatenate([ring(ctr, n, 5.5, 1.2), disc(ctr, n, 4.6, 4.6, th=0.6)]), opacity=0.9, visible=False,
                  note='Schematic. In the membranous septum, below the right and non-coronary cusps, partly under the septal leaflet of the tricuspid valve. The His bundle runs along its posteroinferior rim.')
        emit_mesh('vsd-shunt', 'Left-to-right shunt across the VSD (systole)', 'congenital', ART,
                  arrow(ctr - n * 18, ctr + n * 16, 3.0), opacity=0.85, visible=False,
                  note='Schematic. LV systolic pressure drives flow into the RV and on to the lungs: the LA and LV take the volume load.')
        emit_mesh('vsd-patch', 'Dacron patch (VSD closure)', 'congenital', DACRON, disc(ctr + n * 1.6, n, 7.5, 7.0, th=0.8), visible=False,
                  note='Schematic. Dacron, PTFE or treated pericardium, on the RV side, with interrupted pledgeted or running 5-0/6-0 polypropylene.')
        emit_mesh('vsd-suture', 'Suture line: shallow bites on the RV side of the posteroinferior rim', 'congenital', SUTURE, ring(ctr + n * 2.0, n, 6.4, 0.5), visible=False)
        LM['vsd-c'] = ctr + C
        # where the other types sit on the septum (markers, seen from the RV)
        lv, rv, lvot, tvs = c('lv'), c('rv'), c('lvot'), c('tv-septal') if 'tv-septal' in S else c('rv')
        midsep = (lv + rv) / 2
        LOC = {
            'vsd-loc-1': ('Type 1, subarterial (outlet, doubly committed): under the pulmonary and aortic valves', ctr + SUP * 13 + ANT * 7 + LEFT * 8, '#3fa7d6'),
            'vsd-loc-3': ('Type 3, inlet (AV canal type): under the septal tricuspid leaflet, behind the perimembranous zone', tvs + (lv - tvs) * 0.25 - SUP * 6, '#8fbf7f'),
            'vsd-loc-4': ('Type 4, muscular (trabecular): surrounded by muscle; mid, apical, anterior or posterior; often multiple', midsep + U(midsep - lvot) * 22, '#c9a36b'),
            'vsd-loc-g': ('Gerbode defect: left ventricle to right atrium, through the atrioventricular membranous septum', ctr + U(c('ra') - ctr) * 11 + SUP * 7, '#b07fa8'),
        }
        for k, (nm, q, col) in LOC.items():
            emit_mesh(k, nm, 'congenital', col, trimesh.util.concatenate([ring(q, n, 4.2, 1.0), disc(q, n, 3.4, 3.4, th=0.6)]), visible=False,
                      note='Schematic marker of the typical position of this type (STS / Congenital Heart Surgery Nomenclature types 1 to 4 and the Gerbode defect).')
            LM[k] = q + C

    # ============================================================ coarctation geometry first: the isthmus and the descending aorta
    aorta_mm = np.asarray(ctx['aorta_mm'], float); lsca_mm = np.asarray(ctx['lsca_mm'], float); lpa_mm = np.asarray(ctx['lpa_mm'], float)
    top_z = aorta_mm[:, 2].max()
    # the posterior-left part of each slice is the descending aorta (the ascending aorta is anterior and to the right)
    cl, rad = [], []
    for z in np.arange(top_z - 8, top_z - 125, -5.0):
        s = aorta_mm[(np.abs(aorta_mm[:, 2] - z) < 2.5) & (aorta_mm[:, 0] < C[0] - 12) & (aorta_mm[:, 1] < C[1] + 8)]
        if len(s) < 30: continue
        cc = s.mean(0); cc[2] = z
        rad.append(float(np.percentile(np.linalg.norm(s[:, :2] - cc[:2], axis=1), 90))); cl.append(cc)
    cl = np.array(cl); rad = np.array(rad)
    ls0 = lsca_mm[np.argmin(lsca_mm[:, 2])]
    arch_pt = aorta_mm[np.argmin(np.linalg.norm(aorta_mm - (ls0 - SUP * 6), axis=1))]
    top = (arch_pt + cl[0]) / 2 + SUP * 2

    # ============================================================ patent ductus arteriosus: arch concavity at the isthmus to the top of the left PA
    cand = aorta_mm[(aorta_mm[:, 2] > top_z - 32) & (aorta_mm[:, 2] < top_z - 6) & (aorta_mm[:, 0] < C[0] - 12) & (aorta_mm[:, 1] < C[1] + 10)]
    from scipy.spatial import cKDTree
    d_, ix = cKDTree(lpa_mm).query(cand)
    a_d = cand[np.argmin(d_)]; p_d = lpa_mm[ix[np.argmin(d_)]]
    A, P_ = W(a_d), W(p_d)
    if np.linalg.norm(A - P_) < 8: P_ = A + U(P_ - A) * 8          # a duct is about 5-10 mm long
    mid = (A + P_) / 2 + LEFT * 2.5
    duct = spline([A + U(A - P_) * 2.5, mid, P_ + U(P_ - A) * 2.5], 1.0)
    emit_mesh('pda', 'Patent ductus arteriosus (about 8 mm)', 'congenital', ART, tube(list(duct), 4.0), visible=False,
              note='Schematic. From the concavity of the arch at the isthmus, opposite the left subclavian origin, to the top of the left pulmonary artery near its origin. The left recurrent laryngeal nerve hooks under it.')
    L_ = len(duct)
    t1, t2 = duct[int(L_ * 0.3)], duct[int(L_ * 0.7)]; dd = U(P_ - A)
    emit_mesh('pda-ligatures', 'Two heavy ligatures (aortic end first)', 'congenital', SILK,
              trimesh.util.concatenate([ring(t1, dd, 4.4, 0.9), ring(t2, dd, 4.4, 0.9)]), visible=False,
              note='Schematic. Non-absorbable (e.g. 2-0 or heavier braided) ligatures; aortic end first. A clip suits small infants; division between clamps suits a short, wide or older duct.')
    emit_mesh('pda-clip', 'Titanium clip across the ductus (infant alternative)', 'congenital', STEEL,
              bar(duct[int(L_ * 0.35)], np.cross(dd, SUP), 11.0, 2.0, 1.4), visible=False)
    LM['pda-c'] = (A + P_) / 2 + C; LM['pda-a'] = a_d; LM['pda-p'] = p_d
    # Krichenko types A-E: five small specimens in a row lateral to the real duct, each with stubs of aorta (above) and PA (below)
    ax = SUP                                         # as on a lateral angiogram: aorta above, PA below
    side = -ANT                                      # the row runs front to back, seen from the left
    base = (A + P_) / 2 + LEFT * 45 + SUP * 18 + ANT * 44
    t = np.linspace(0, 1, 40)
    TYPES = {
        'a': ('Type A, conical', 12.0, lambda u: 1.6 + 3.8 * u ** 1.6, 0.0),
        'b': ('Type B, window', 4.0, lambda u: 5.2 + 0 * u, 0.0),
        'c': ('Type C, tubular', 12.0, lambda u: 2.8 + 0 * u, 0.0),
        'd': ('Type D, complex', 12.0, lambda u: 3.0 - 1.4 * np.exp(-((u - 0.3) / 0.08) ** 2) - 1.4 * np.exp(-((u - 0.7) / 0.08) ** 2), 0.0),
        'e': ('Type E, elongated', 20.0, lambda u: 1.4 + 2.2 * (1 - np.exp(-((u - 0.0) / 0.25) ** 2)) * 0 + np.where(u < 0.25, 1.4 + 6 * u, 2.9), 5.0),
    }
    for k, (key, (nm, L, prof, bow)) in enumerate(TYPES.items()):
        o = base + side * (k * 30.0) - side * 15.0
        p0, p1 = o - ax * L / 2, o + ax * L / 2           # pulmonary end, aortic end
        pts = np.array([p0 + (p1 - p0) * u + LEFT * bow * np.sin(np.pi * u) for u in t])
        duct_k = lathe(pts, prof(t))
        aorta_k = trimesh.creation.cylinder(radius=5.5, height=14.0, sections=24)
        Ma = trimesh.geometry.align_vectors([0, 0, 1], side); Ma[:3, 3] = p1 + ax * 5.5; aorta_k.apply_transform(Ma)
        pa_k = trimesh.creation.cylinder(radius=4.5, height=13.0, sections=24)
        Mp = trimesh.geometry.align_vectors([0, 0, 1], side); Mp[:3, 3] = p0 - ax * 4.5; pa_k.apply_transform(Mp)
        emit_mesh(f'pda-type-{key}', nm, 'congenital', ART, trimesh.util.concatenate([duct_k, aorta_k, pa_k]), visible=False,
                  note='Schematic specimen (Krichenko angiographic classification, lateral view): aorta above, pulmonary artery below.')
        LM[f'pda-type-{key}'] = o + C

    # ============================================================ coarctation (juxtaductal, just beyond the duct)
    path = np.vstack([top, cl])
    rr = np.concatenate([[rad[:3].mean()], rad])
    rr = np.convolve(np.pad(rr, 2, mode='edge'), np.ones(5) / 5, mode='valid')
    Pw = np.array([W(p) for p in path])
    fine = spline(Pw, 1.5)
    s_ = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(fine, axis=0), axis=1))])
    s_raw = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pw, axis=0), axis=1))])
    r_fine = np.interp(s_ * s_raw[-1] / s_[-1], s_raw, rr)
    s_waist = float(s_[np.argmin(np.linalg.norm(fine - A, axis=1))]) + 4.0
    prof = 1 - 0.68 * np.exp(-((s_ - s_waist) / 3.2) ** 2) + 0.16 * np.exp(-((s_ - (s_waist + 20)) / 9.0) ** 2)
    emit_mesh('coa-segment', 'Coarctation of the aorta: juxtaductal shelf, post-stenotic dilatation', 'congenital', ART,
              lathe(fine, r_fine * prof), visible=False,
              note='Schematic. A posterior shelf of thickened media and intima at the isthmus, opposite the duct; the lumen here is a third of normal. The aorta dilates beyond it.')
    iw = int(np.argmin(np.abs(s_ - s_waist))); wpt = fine[iw]; td = U(fine[min(iw + 2, len(fine) - 1)] - fine[max(iw - 2, 0)])
    emit_mesh('coa-shelf', 'Coarctation shelf (ridge of intima and media)', 'congenital', FIBRE, ring(wpt, td, r_fine[iw] * 0.55, r_fine[iw] * 0.28), visible=False)
    emit_mesh('coa-repaired', 'Descending aorta after resection and extended end-to-end anastomosis', 'congenital', ART,
              lathe(fine, r_fine * (1 - 0.08 * np.exp(-((s_ - s_waist) / 6) ** 2))), visible=False,
              note='Schematic. The narrowed segment and ductal tissue excised; the descending aorta brought up to an incision extended along the underside of the arch.')
    emit_mesh('coa-anastomosis', 'Anastomosis: running polypropylene (oblique, extended under the arch)', 'congenital', SUTURE,
              ring(wpt, U(td + ANT * 0.35), r_fine[iw] * 1.02, 0.55), visible=False)
    col = []
    for k, off in enumerate((28, 46, 64, 82)):
        j = int(np.argmin(np.abs(s_ - (s_waist + off))))
        if j >= len(fine) - 1: continue
        p0 = fine[j]; out = U(LEFT * 0.8 - ANT * 0.6)
        pts = [p0 + out * r_fine[j]] + [p0 + out * (r_fine[j] + 8 * q) + SUP * (2.5 * np.sin(q * 1.9 + k)) + ANT * (2.0 * np.cos(q * 1.4)) - SUP * q * 1.5 for q in range(1, 8)]
        col.append(tube(pts, 1.5))
    if col:
        emit_mesh('coa-collaterals', 'Enlarged, tortuous intercostal collaterals', 'congenital', ART, trimesh.util.concatenate(col), visible=False,
                  note='Schematic. Collaterals from the subclavian branches (internal thoracic, thyrocervical) fill the intercostals, which run backwards into the aorta below the coarctation. They cause rib notching and bleed when the chest is opened.')
    # clamps: proximal across the distal arch (taking the left subclavian origin), distal on the descending aorta
    jd = int(np.argmin(np.abs(s_ - (s_waist + 32)))); pd = fine[jd]
    side = U(np.cross(td, ANT))
    emit_mesh('coa-clamps', 'Clamps: distal arch (with the subclavian origin) and descending aorta', 'congenital', STEEL,
              trimesh.util.concatenate([bar(W(top) + SUP * 3, side, 34, 3.0, 2.2), bar(pd, side, 30, 3.0, 2.2)]), visible=False,
              note='Schematic. Proximal clamp across the distal arch beyond the left carotid, including the subclavian origin; distal clamp on the descending aorta below the intercostals that have been controlled.')
    LM['coa-c'] = wpt + C; LM['coa-top'] = W(top) + C

    # ============================================================ left posterolateral thoracotomy: 3rd and 4th intercostal spaces
    port, lung_c = ctx['port'], np.asarray(ctx['lung_c'], float)
    for ics in (3, 4):
        pts = np.array([port(ics, az, 'left') for az in range(30, -131, -10)])
        pts[1:-1] = (pts[:-2] + 2 * pts[1:-1] + pts[2:]) / 4
        r_ = pts[:, :2] - lung_c[:2]; pts[:, :2] += r_ / np.linalg.norm(r_, axis=1, keepdims=True) * 2.5
        emit_mesh(f'incision-l{ics}', f'Left posterolateral thoracotomy, {ics}{"rd" if ics == 3 else "th"} intercostal space', 'incisions', INC,
                  tube([W(q) for q in pts], 1.8), visible=False,
                  note=f'Schematic: along the {ics}{"rd" if ics == 3 else "th"} space, from below the axilla, round the tip of the scapula to the paraspinal muscles. '
                       + ('Higher: straight onto the distal arch, the subclavian and the duct; cramped under the scapula in a bigger child.' if ics == 3 else
                          'The standard space: the isthmus and the duct lie at its level, with room to reach the arch above and the descending aorta below.'))
        q = port(ics, -30, 'left'); LM[f'thor{ics}-l'] = q + (lung_c - q) * np.array([1, 1, 0]) / (np.linalg.norm((lung_c - q)[:2]) + 1e-9) * 14.0
    return LM
