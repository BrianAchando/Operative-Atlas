"""The heart for the cardiac module: mitral, aortic and tricuspid valve surgery, aortic root replacement (Bentall, Ross), coronary bypass, and their accesses.

From the CT (TotalSegmentator licensed tasks, run by segment_heart.py): the four chambers and the myocardium
(heartchambers_highres), the LV outflow tract and the three aortic cusps (aortic_sinuses), the coronaries.
Derived from them: the mitral annulus (the LA-LV interface, fitted to a plane), Sondergaard's (interatrial) groove,
the atrial septum and fossa ovalis, the right atrial wall.
Schematic, on those landmarks: the mitral leaflets (anterior and posterior), papillary muscles and chordae, the
coronary sinus and circumflex artery in the posterior AV groove, the AV node region, the cannulas (aortic, SVC, IVC,
antegrade and retrograde cardioplegia), the atriotomy lines, a mechanical bileaflet prosthesis, the right
mini-thoracotomy incision and the transthoracic clamp site.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage

ANT, SUP, RIGHT = np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
TUBE_BLUE, CANNULA = '#8fc8e8', '#dfe8ee'


def build(ctx):
    emit, emit_mesh, W, tube, sphere = ctx['emit'], ctx['emit_mesh'], ctx['W'], ctx['tube'], ctx['sphere']
    work: Path = ctx['work']; LM, DIRS, SC = {}, {}, {}
    if not (work / 'heart.nii.gz').exists():
        print('== cardiac: no work/heart.nii.gz (run segment_heart.py); skipped'); return LM, DIRS, SC
    print('== cardiac')
    himg = nib.as_closest_canonical(nib.load(str(work / 'heart.nii.gz'))); H = np.asanyarray(himg.dataobj); A = himg.affine
    sp = np.abs(np.diag(A)[:3])
    mm = lambda m: np.argwhere(m) @ A[:3, :3].T + A[:3, 3]
    names = {1: ('myocardium', 'Myocardium', '#8b3a33'), 2: ('la', 'Left atrium', '#b8705f'), 3: ('lv', 'Left ventricle (cavity)', '#a2463b'),
             4: ('ra', 'Right atrium', '#7a5a86'), 5: ('rv', 'Right ventricle (cavity)', '#8e5a78'), 7: ('pa-trunk', 'Pulmonary trunk', '#3b52a8')}
    for lab, (id_, name, col) in names.items():
        m = H == lab
        if m.any(): emit(id_, name, 'cardiac', col, m, A, faces=12000 if lab in (1, 2, 3) else 8000, visible=False, label=True)
    if (work / 'sinuses.nii.gz').exists():
        S = np.asanyarray(nib.as_closest_canonical(nib.load(str(work / 'sinuses.nii.gz'))).dataobj)
        for lab, (id_, name, col) in {1: ('lvot', 'LV outflow tract', '#c0564a'), 2: ('cusp-r', 'Right coronary cusp', '#e8d9c9'),
                                       3: ('cusp-l', 'Left coronary cusp', '#e8d9c9'), 4: ('cusp-n', 'Non-coronary cusp', '#e8d9c9')}.items():
            if (S == lab).any(): emit(id_, name, 'cardiac', col, S == lab, A, faces=3000, visible=False, label=True)
    if (work / 'coronary.nii.gz').exists():
        C = np.asanyarray(nib.as_closest_canonical(nib.load(str(work / 'coronary.nii.gz'))).dataobj) > 0
        if C.sum() > 50: emit('coronaries', 'Coronary arteries (segmented)', 'cardiac', '#d0433a', C, A, faces=4000, visible=False, label=True,
                              note='From the CT; small vessels are incomplete on this scan.')

    LA, LV, RA, RV, AO = H == 2, H == 3, H == 4, H == 5, H == 6
    la_mm, lv_mm, ra_mm = mm(LA), mm(LV), mm(RA)
    # ---------------------------------------------------------------- mitral annulus: where the LA meets the LV
    iface = LV & ndimage.binary_dilation(LA, iterations=2)
    P = mm(iface)
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2] * np.sign(np.dot(vt[2], la_mm.mean(0) - c))                 # the annular plane's normal, toward the LA
    lvot = mm(np.asanyarray(nib.as_closest_canonical(nib.load(str(work / 'sinuses.nii.gz'))).dataobj) == 1) if (work / 'sinuses.nii.gz').exists() else mm(AO)
    u = lvot.mean(0) - c; u -= n * np.dot(u, n); u /= np.linalg.norm(u)   # toward the aortic valve: the anterior leaflet side
    v = np.cross(n, u)
    rel = P - c; ang = np.arctan2(rel @ v, rel @ u); rad = np.hypot(rel @ u, rel @ v)
    bins = np.linspace(-np.pi, np.pi, 49); ring = []
    for a0, a1 in zip(bins[:-1], bins[1:]):
        s = (ang >= a0) & (ang < a1)
        r_ = np.percentile(rad[s], 80) if s.sum() > 3 else np.nan
        h_ = np.median(rel[s] @ n) if s.sum() > 3 else 0.0
        ring.append(((a0 + a1) / 2, r_, h_))
    ring = np.array(ring); ok = ~np.isnan(ring[:, 1])
    ring[:, 1] = np.interp(ring[:, 0], ring[ok, 0], ring[ok, 1], period=2 * np.pi)
    k = np.ones(5) / 5
    ring[:, 1] = np.convolve(np.r_[ring[-2:, 1], ring[:, 1], ring[:2, 1]], k, 'valid'); ring[:, 2] = np.convolve(np.r_[ring[-2:, 2], ring[:, 2], ring[:2, 2]], k, 'valid')
    R = float(np.clip(np.median(ring[:, 1]), 11, 20))
    ring[:, 1] = np.clip(ring[:, 1], R * 0.8, R * 1.2)
    ann = np.array([c + u * np.cos(t) * r + v * np.sin(t) * r + n * h for t, r, h in ring])
    emit_mesh('mitral-annulus', 'Mitral annulus', 'cardiac', '#e9dec6', tube([W(p) for p in [*ann, ann[0]]], 1.6), visible=False,
              note='The LA-LV junction on this scan, fitted to a ring. Anteriorly it is continuous with the aortic valve (aortomitral curtain).')
    LM['mv-centre'] = c; DIRS['mv-normal'] = n; DIRS['mv-anterior'] = u; SC['mv-radius'] = R
    # ---------------------------------------------------------------- leaflets (closed), papillary muscles, chordae
    apex = lv_mm[np.argmax((lv_mm - c) @ -n)]
    SC['_ann'] = ann; SC['_apex'] = apex
    coapt = lambda t, r: c + v * (np.sin(t) * r * 0.82) - u * (0.28 * R) - n * 9.0          # the coaptation line, a third of the way from the back
    def leaflet(sel):
        V, F = [], []; rows = 7; cols = [i for i in range(len(ring)) if sel(ring[i, 0])]
        if len(cols) < 2: return None
        for j, i in enumerate(cols):
            t, r, h = ring[i]; a = ann[i]; b = coapt(t, r)
            for q in range(rows):
                s = q / (rows - 1); p = a + (b - a) * s - n * 3.0 * np.sin(np.pi * s)       # bellied toward the ventricle
                V.append(p)
        for j in range(len(cols) - 1):
            for q in range(rows - 1):
                a_, b_ = j * rows + q, (j + 1) * rows + q; F += [[a_, b_, b_ + 1], [a_, b_ + 1, a_ + 1]]
        return trimesh.Trimesh(np.array(V) - ctx['CARINA'], np.array(F), process=True)
    antl = leaflet(lambda t: abs(t) <= np.radians(62)); post = leaflet(lambda t: abs(t) >= np.radians(62))
    if antl is not None: emit_mesh('mv-ant-leaflet', 'Anterior mitral leaflet', 'cardiac', '#efe3cf', antl, visible=False, note='Schematic: the larger, anterior leaflet, in continuity with the aortic valve.')
    if post is not None: emit_mesh('mv-post-leaflet', 'Posterior mitral leaflet', 'cardiac', '#efe3cf', post, visible=False, note='Schematic: the posterior (mural) leaflet, P1-P3; its chordae are kept in a chordal-sparing replacement.')
    pm_base = c + (apex - c) * 0.62
    pms = {'al': pm_base + v * (-R * 0.55) + u * (R * 0.25), 'pm': pm_base + v * (R * 0.55) - u * (R * 0.35)}
    # anterolateral is toward the patient's left (lateral); choose the side of v that points left
    if np.dot(v, RIGHT) < 0: pms = {'al': pms['pm'], 'pm': pms['al']}
    parts, chords = [], []
    for key, b in pms.items():
        tip = b + (c - b) * 0.25
        cone = trimesh.creation.cone(radius=5.5, height=np.linalg.norm(tip - b), sections=16)
        z = (tip - b) / np.linalg.norm(tip - b); T = trimesh.geometry.align_vectors([0, 0, 1], z); T[:3, 3] = W(b); cone.apply_transform(T); parts.append(cone)
        for t in np.linspace(-np.pi, np.pi, 13):
            edge = coapt(t, R)
            if np.dot(edge - c, tip - c) > 0: chords.append(tube([W(tip), W(edge)], 0.35, seg=6))
        LM[f'pm-{key}'] = tip
    emit_mesh('papillary', 'Papillary muscles (anterolateral, posteromedial)', 'cardiac', '#8b3a33', trimesh.util.concatenate(parts), visible=False, note='Schematic.')
    if chords: emit_mesh('chordae', 'Chordae tendineae', 'cardiac', '#f3ecd9', trimesh.util.concatenate(chords), visible=False, note='Schematic: from the papillary muscles to both leaflets.')
    # ---------------------------------------------------------------- the posterior AV groove: coronary sinus and circumflex
    post_idx = [i for i in range(len(ring)) if abs(ring[i, 0]) > np.radians(80)]
    ordered = sorted(post_idx, key=lambda i: (ring[i, 0] % (2 * np.pi)))
    cs = [ann[i] + (ann[i] - c) / np.linalg.norm(ann[i] - c) * 7 + n * 6 for i in ordered]
    cx = [ann[i] + (ann[i] - c) / np.linalg.norm(ann[i] - c) * 6 - n * 2 for i in ordered]
    emit_mesh('coronary-sinus', 'Coronary sinus', 'cardiac', '#5e6a92', tube([W(p) for p in cs], 4.0), visible=False,
              note='Schematic: in the posterior AV groove, about 1 cm on the atrial side of the posterior annulus, opening into the right atrium. Retrograde cardioplegia goes in here.')
    emit_mesh('circumflex', 'Circumflex artery (AV groove)', 'cardiac', '#d0433a', tube([W(p) for p in cx], 1.6), visible=False,
              note='Schematic: close to the posterior annulus near the anterolateral commissure, especially if left-dominant. A deep suture there can catch it.')
    # AV node region: at the posteromedial commissure, toward the right atrium
    pmc = min(range(len(ring)), key=lambda i: np.linalg.norm(ann[i] - ra_mm.mean(0)))
    avn = ann[pmc] + (ra_mm.mean(0) - ann[pmc]) / np.linalg.norm(ra_mm.mean(0) - ann[pmc]) * 9
    emit_mesh('av-node', 'AV node region (Koch\'s triangle)', 'cardiac', '#f2d24b', sphere(W(avn), 4.0), visible=False,
              note='Schematic: near the posteromedial commissure and the right fibrous trigone. Deep bites here cause heart block.')
    LM['av-node'] = avn
    # ---------------------------------------------------------------- Sondergaard's groove, the atrial septum, the right atrium
    dRA = ndimage.distance_transform_edt(~RA, sampling=sp)                 # the septum (myocardium) lies between the two atria
    groove = mm(LA & (dRA < 12))
    g_right = groove[(groove[:, 0] > np.percentile(groove[:, 0], 40))]
    zs = np.arange(g_right[:, 2].min() + 5, g_right[:, 2].max() - 5, 6.0)
    line = []
    for z in zs:
        s = g_right[np.abs(g_right[:, 2] - z) < 3]
        if len(s): line.append(s[np.argmax(s[:, 0] - 0.3 * s[:, 1])] + RIGHT * 2)
    line = np.array(line)
    if len(line) > 3:
        line[1:-1] = (line[:-2] + 2 * line[1:-1] + line[2:]) / 4
        emit_mesh('la-incision', "Left atriotomy (Sondergaard's groove)", 'cardiac', '#d0433a', tube([W(p) for p in line], 1.4), visible=False,
                  note="Through the interatrial groove, in front of the right pulmonary veins, after developing Sondergaard's plane.")
        LM['la-incision'] = line[len(line) // 2]
    sep = mm(LA & (dRA < 9)); sc_ = sep.mean(0)
    sline = [p for p in sep if np.linalg.norm((p - sc_)[:2]) < 6]
    sline = np.array(sorted(sline, key=lambda p: p[2]))[::max(1, len(sline) // 8)] if len(sline) > 8 else np.array([sc_ - SUP * 12, sc_, sc_ + SUP * 12])
    emit_mesh('septal-incision', 'Transseptal incision (through the fossa ovalis)', 'cardiac', '#d0433a', tube([W(p) for p in sline], 1.2), visible=False,
              note='Vertical incision through the fossa ovalis, extended up if needed (the superior septal approach carries it across the dome of the LA).')
    LM['septum'] = sc_
    rl = ra_mm[ra_mm[:, 0] > np.percentile(ra_mm[:, 0], 90)]
    ra_line = np.array([rl[np.argmin(np.abs(rl[:, 2] - z))] + RIGHT * 2 for z in np.linspace(rl[:, 2].max() - 8, rl[:, 2].min() + 10, 6)])
    emit_mesh('ra-incision', 'Right atriotomy', 'cardiac', '#d0433a', tube([W(p) for p in ra_line], 1.4), visible=False,
              note='Oblique, from the appendage toward the IVC, parallel to the AV groove (away from the sinus node at the SVC junction).')
    LM['ra-incision'] = ra_line[len(ra_line) // 2]
    # ---------------------------------------------------------------- cannulas and cardioplegia
    ao = ctx['aorta_mm']; bct = ctx['bct_mm']
    z_can = float(bct[:, 2].min()) - 12 if len(bct) else float(np.percentile(ao[:, 2], 75))
    s_ = ao[(np.abs(ao[:, 2] - z_can) < 3) & (ao[:, 1] > np.percentile(ao[:, 1], 60))]
    ao_can = s_[np.argmax(s_[:, 1])] if len(s_) else ao[np.argmax(ao[:, 1])]
    cusps = mm(np.asanyarray(nib.as_closest_canonical(nib.load(str(work / 'sinuses.nii.gz'))).dataobj) >= 2) if (work / 'sinuses.nii.gz').exists() else lvot
    root_top = cusps[np.argmax(cusps[:, 2])]
    s2 = ao[(np.abs(ao[:, 2] - (root_top[2] + 12)) < 3) & (ao[:, 1] > np.percentile(ao[:, 1], 50))]
    cp = s2[np.argmax(s2[:, 1])] if len(s2) else root_top + ANT * 12
    clamp = (ao_can + cp) / 2
    s3 = ao[(np.abs(ao[:, 2] - clamp[2]) < 3)]
    clamp_c = s3.mean(0) if len(s3) else clamp
    svc = ctx['svc_mm']; svc_hi = svc[svc[:, 2] > np.percentile(svc[:, 2], 60)]; svc_p = svc_hi.mean(0)
    ra_low = ra_mm[ra_mm[:, 2] < np.percentile(ra_mm[:, 2], 12)].mean(0)
    cans = [('can-aortic', 'Aortic cannula', [ao_can + ANT * 2, ao_can + ANT * 30 + SUP * 20, ao_can + ANT * 90 + SUP * 40], 3.2),
            ('can-svc', 'SVC cannula', [svc_p, svc_p + RIGHT * 18 + ANT * 20, svc_p + RIGHT * 30 + ANT * 90 + SUP * 20], 3.6),
            ('can-ivc', 'IVC cannula (tip in the IVC)', [ra_low - SUP * 20, ra_low, ra_low + RIGHT * 22 + ANT * 20, ra_low + RIGHT * 40 + ANT * 90 - SUP * 10], 4.0),
            ('can-cp', 'Antegrade cardioplegia needle (aortic root)', [cp + ANT * 1, cp + ANT * 25 + SUP * 10, cp + ANT * 80 + SUP * 25], 1.6)]
    for id_, name, pts, r in cans:
        emit_mesh(id_, name, 'cardiac', CANNULA, tube([W(p) for p in pts], r), opacity=0.9, visible=False, note='Schematic.')
    csos = cs[0] if np.linalg.norm(cs[0] - ra_mm.mean(0)) < np.linalg.norm(cs[-1] - ra_mm.mean(0)) else cs[-1]
    emit_mesh('can-retro', 'Retrograde cardioplegia cannula (coronary sinus)', 'cardiac', CANNULA,
              tube([W(p) for p in [csos, (csos + LM['ra-incision']) / 2 + RIGHT * 10, LM['ra-incision'] + RIGHT * 40 + ANT * 60]], 1.8), opacity=0.9, visible=False,
              note='Schematic: passed through a purse-string in the right atrium into the coronary sinus.')
    tricuspid(ctx, RA, RV, mm, LM, DIRS, SC, ra_mm, lv_mm, csos, svc_p, ra_low)
    LM['clamp-ao'] = clamp_c; DIRS['ao-axis'] = np.array([0, 0.25, 1.0]) / np.linalg.norm([0, 0.25, 1.0])
    LM['can-aortic'] = ao_can; LM['cp'] = cp; LM['can-svc'] = svc_p; LM['can-ivc'] = ra_low
    # ---------------------------------------------------------------- the prosthesis: a mechanical bileaflet valve, seated in the annulus
    rp = float(np.clip(R * 0.92, 12.5, 16.0))
    parts = [trimesh.creation.torus(major_radius=rp + 1.5, minor_radius=2.0, major_sections=48, minor_sections=10),
             trimesh.creation.annulus(r_min=rp - 1.2, r_max=rp, height=5.0, sections=48)]
    for sx in (-1, 1):
        leaf = trimesh.creation.box(extents=[rp * 0.95, 0.8, 5.5]); leaf.apply_translation([sx * rp * 0.45, 0, -1.5])
        leaf.apply_transform(trimesh.transformations.rotation_matrix(np.radians(80) * sx, [0, 1, 0], point=[sx * rp * 0.9, 0, 0])); parts.append(leaf)
    pv = trimesh.util.concatenate(parts)
    T = trimesh.geometry.align_vectors([0, 0, 1], n); T[:3, 3] = W(c + n * 1.5); pv.apply_transform(T)
    emit_mesh('mv-prosthesis', f'Mechanical bileaflet prosthesis (~{round(2 * rp)} mm)', 'cardiac', '#c9d1d9', pv, visible=False,
              note='Schematic: a bileaflet valve on its sewing ring, seated on the annulus.')
    # ---------------------------------------------------------------- right mini-thoracotomy and the transthoracic clamp
    if ctx.get('port') is not None:
        pts = np.array([ctx['port'](4, a, 'right') for a in np.linspace(20, 55, 5)])
        cr = ctx['lung_cr']; r_ = pts[:, :2] - cr[:2]; pts[:, :2] += r_ / np.linalg.norm(r_, axis=1, keepdims=True) * 2.5
        emit_mesh('incision-mics', 'Right mini-thoracotomy (4th space)', 'incisions', '#d0433a', tube([W(p) for p in pts], 1.8), visible=False,
                  note='4-6 cm in the right 4th space, in the inframammary fold, anterior to the anterior axillary line.')
        LM['mics'] = pts[2]
        cl = ctx['port'](3, 0, 'right'); LM['chitwood'] = cl
        emit_mesh('port-chitwood', 'Transthoracic (Chitwood) clamp site, 3rd space', 'ports-vats', '#46c2c7', sphere(W(cl), 3.5), visible=False)
    # ================================================================ the aortic root, for aortic valve replacement
    if (work / 'sinuses.nii.gz').exists():
        aortic_root(ctx, work, mm, LM, DIRS, SC, ra_mm, la_mm, lv_mm)
        root_repl(ctx, mm, H, LM, DIRS, SC)
        coronary_tree(ctx, H, mm, LM, DIRS, SC, A)
    import pathology                                                        # valve pathology for the case scenarios
    pathology.build({**ctx, 'la_c': la_mm.mean(0)}, LM, DIRS, SC)
    print(f'  annulus radius {R:.1f} mm; prosthesis ~{2 * rp:.0f} mm')
    return LM, DIRS, SC


def aortic_root(ctx, work, mm, LM, DIRS, SC, ra_mm, la_mm, lv_mm):
    """The aortic root from the sinus segmentation: the scalloped (crown) annulus, the three cusps (schematic, calcified),
    commissures, coronary ostia, the His bundle under the right-non-coronary commissure, the aortotomy, a stented
    bioprosthesis, and the accesses for AVR (upper hemisternotomy, right anterior mini-thoracotomy)."""
    emit_mesh, W, tube, sphere = ctx['emit_mesh'], ctx['W'], ctx['tube'], ctx['sphere']
    S = np.asanyarray(nib.as_closest_canonical(nib.load(str(work / 'sinuses.nii.gz'))).dataobj)
    lvot = S == 1; sin = {k: mm(S == lab) for k, lab in (('r', 2), ('l', 3), ('n', 4)) if (S == lab).sum() > 30}
    if len(sin) < 3 or lvot.sum() < 30: print('  aortic root: sinuses incomplete; skipped'); return
    allc = np.vstack(list(sin.values()))
    iface = mm(lvot & ndimage.binary_dilation(S >= 2, iterations=2))
    if len(iface) < 20: iface = mm(lvot)[np.argsort(-(mm(lvot) @ (allc.mean(0) - mm(lvot).mean(0))))[:200]]
    c = iface.mean(0); _, _, vt = np.linalg.svd(iface - c, full_matrices=False)
    n = vt[2] * np.sign(np.dot(vt[2], allc.mean(0) - c))              # the root axis, from the LV into the aorta
    # the basal ring: the lowest points of the three sinuses define it better than the interface alone
    nadir = {k: p[np.argmin((p - c) @ n)] for k, p in sin.items()}
    c = c + n * float(np.mean([(nadir[k] - c) @ n for k in sin]))
    e1 = nadir['r'] - c; e1 -= n * np.dot(e1, n); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    ang = lambda p: float(np.arctan2((p - c) @ e2, (p - c) @ e1))
    ctr = {k: np.arctan2(np.mean(np.sin([ang(q) for q in p[::5]])), np.mean(np.cos([ang(q) for q in p[::5]]))) for k, p in sin.items()}
    h = {k: (p - c) @ n for k, p in sin.items()}; hs = float(np.percentile(np.concatenate(list(h.values())), 96))   # sinotubular junction
    rad = lambda p: np.linalg.norm((p - c) - np.outer((p - c) @ n, n), axis=1)
    R = float(np.clip(np.percentile(rad(iface), 95), 9.5, 14.0))
    hs = float(np.clip(hs, 17, 26)); hc = hs * 0.85                         # commissures just below the STJ
    at = lambda t, r, z: c + e1 * np.cos(t) * r + e2 * np.sin(t) * r + n * z
    def mid(a, b):
        d = (b - a + np.pi) % (2 * np.pi) - np.pi; return a + d / 2
    order = sorted(sin, key=lambda k: ctr[k] % (2 * np.pi))
    pairs = [(order[i], order[(i + 1) % 3]) for i in range(3)]
    com = {frozenset(p): mid(ctr[p[0]], ctr[p[1]]) for p in pairs}
    # the crown: each cusp's hinge runs from one commissure (high) down to its nadir (low) and back
    crown, cusp_rows = [], {}
    for k in order:
        a0 = [com[f] for f in com if k in f]
        t0, t1 = sorted(a0, key=lambda t: ((t - ctr[k] + np.pi) % (2 * np.pi)) - np.pi)
        d0 = ((t0 - ctr[k] + np.pi) % (2 * np.pi)) - np.pi; d1 = ((t1 - ctr[k] + np.pi) % (2 * np.pi)) - np.pi
        ts = ctr[k] + np.linspace(d0, d1, 17)
        s = np.linspace(-1, 1, 17); z = hc * s ** 2; r_ = R * (1 + 0.18 * s ** 2)     # wider toward the commissures
        pts = [at(t, rr, zz) for t, rr, zz in zip(ts, r_, z)]; crown += pts[:-1]; cusp_rows[k] = (ts, r_, z)
    crown = np.array(crown)
    emit_mesh('aortic-annulus', 'Aortic annulus (crown-shaped hinge line)', 'cardiac', '#e9dec6', tube([W(p) for p in [*crown, crown[0]]], 1.3), visible=False,
              note='The cusps hinge on a three-pronged crown: low at each nadir (the virtual basal ring), high at the commissures just below the sinotubular junction. Sutures follow this line.')
    stj = np.array([at(t, R * 1.1, hs) for t in np.linspace(-np.pi, np.pi, 40)])
    emit_mesh('stj', 'Sinotubular junction', 'cardiac', '#b9c2c9', tube([W(p) for p in [*stj, stj[0]]], 0.9), visible=False)
    # the cusps, closed: each a curved sheet from its hinge to the centre, bellied toward the ventricle
    nm = {'r': 'Right coronary cusp (leaflet)', 'l': 'Left coronary cusp (leaflet)', 'n': 'Non-coronary cusp (leaflet)'}
    calc = []
    rng = np.random.default_rng(3)
    for k in order:
        ts, r_, z = cusp_rows[k]; Vv, F = [], []; rows = 7
        for t, rr, zz in zip(ts, r_, z):
            a = at(t, rr, zz); b = c + n * (hc * 0.55)
            for q in range(rows):
                s_ = q / (rows - 1); Vv.append(a + (b - a) * s_ - n * 3.5 * np.sin(np.pi * s_) * (1 - abs(t - ctr[k]) / np.pi))
        for j in range(len(ts) - 1):
            for q in range(rows - 1):
                a_, b_ = j * rows + q, (j + 1) * rows + q; F += [[a_, b_, b_ + 1], [a_, b_ + 1, a_ + 1]]
        m = trimesh.Trimesh(np.array(Vv) - ctx['CARINA'], np.array(F), process=True)
        emit_mesh(f'av-cusp-{k}', nm[k], 'cardiac', '#e3d6bf', m, visible=False, note='Schematic, thickened and calcified as in degenerative or rheumatic stenosis.')
        Vv = np.array(Vv)
        for _ in range(4):
            j = rng.integers(3, len(ts) - 3); q = rng.integers(1, rows - 2)
            calc.append(sphere(W(Vv[j * rows + q] - n * 1.0), float(rng.uniform(1.6, 2.8))))
        calc.append(sphere(W(at(ctr[k], R * 0.95, 1.5)), 2.4))                    # at the nadir, into the annulus
    emit_mesh('av-calcium', 'Calcium on the cusps and annulus', 'cardiac', '#f4f0e0', trimesh.util.concatenate(calc), visible=False,
              note='Schematic: nodular calcium on the cusps, extending into the annulus at the nadirs. Debride it all, catching every fragment.')
    # coronary ostia: where the segmented coronaries leave the left and right sinuses, else high in the sinus wall
    C = np.asanyarray(nib.as_closest_canonical(nib.load(str(work / 'coronary.nii.gz'))).dataobj) > 0 if (work / 'coronary.nii.gz').exists() else None
    cmm = mm(C) if C is not None and C.sum() > 50 else np.zeros((0, 3))
    for k in ('r', 'l'):
        guess = at(ctr[k], R * 1.28, hs * 0.62)
        p = guess
        if len(cmm):
            d = np.linalg.norm(cmm - guess, axis=1); j = np.argmin(d)
            if d[j] < 14: p = cmm[j]
        LM[f'ostium-{k}'] = p
        emit_mesh(f'ostium-{k}', 'Left main coronary ostium' if k == 'l' else 'Right coronary ostium', 'cardiac', '#d0433a',
                  trimesh.util.concatenate([sphere(W(p), 2.6), tube([W(p), W(p + (p - c - n * np.dot(p - c, n)) / np.linalg.norm(p - c - n * np.dot(p - c, n)) * 14)], 1.6)]), visible=False,
                  note='About 1-1.5 cm above the annulus in its sinus. Keep prosthesis posts and pledgets clear of it, and look into it before closing.')
    # the His bundle: in the membranous septum under the right-non-coronary commissure
    rn = com[frozenset(('r', 'n'))]; his = at(rn, R * 0.95, -4.0)
    emit_mesh('his-bundle', 'Membranous septum and His bundle', 'cardiac', '#f2d24b', sphere(W(his), 3.6), visible=False,
              note='Below the commissure between the right and non-coronary cusps. Deep bites or aggressive debridement here cause complete heart block.')
    LM['his'] = his
    ln = com[frozenset(('l', 'n'))]; LM['aml-curtain'] = at(ln, R, -5.0)
    # the aortotomy: oblique (hockey-stick), 1-1.5 cm above the right coronary ostium, curving down into the non-coronary sinus
    ao = ctx['aorta_mm']; z_top = hs + 16
    def ao_r(z):
        s = ao[np.abs((ao - c) @ n - z) < 2.5]
        return float(np.median(rad(s))) if len(s) > 20 else R * 1.3
    ra_ = ao_r(z_top); tR, tN = ctr['r'], ctr['n']
    d_ = ((tN - tR + np.pi) % (2 * np.pi)) - np.pi
    leg1 = [at(tR - 0.9 * np.sign(d_), ra_ + 1.5, z_top + 3), at(tR, ra_ + 1.5, z_top), at(tR + d_ * 0.5, ra_ + 1.5, z_top - 3)]
    leg2 = [at(tR + d_ * (0.5 + 0.5 * s), R * 1.3 + (ra_ - R * 1.3) * (1 - s) + 1.5, z_top - 3 - (z_top - 3 - hs * 0.45) * s) for s in np.linspace(0.2, 1, 5)]
    line = np.array([*leg1, *leg2])
    emit_mesh('aortotomy', 'Aortotomy (oblique, into the non-coronary sinus)', 'cardiac', '#d0433a', tube([W(p) for p in line], 1.4), visible=False,
              note='Oblique ("hockey-stick"): across the front of the aorta 1-1.5 cm above the right coronary, then down into the non-coronary sinus toward its nadir. Stay above the right coronary ostium.')
    for i, q in enumerate(line): LM[f'aot-{i}'] = q
    LM['aortotomy'] = line[2]; LM['av-centre'] = c; DIRS['av-axis'] = n; DIRS['av-e1'] = e1; SC['av-radius'] = R; SC['av-stj'] = hs
    for i, (a_, b_) in enumerate(pairs): LM[f'comm-{i}'] = at(com[frozenset((a_, b_))], R * 1.08, hc)
    # a stented bioprosthesis: sewing ring, three posts aligned with the native commissures, three leaflets
    rp = float(np.clip(R * 0.95, 9.5, 13.5)); base = c + n * 1.5; post_h = hs * 0.8
    parts = []
    ring = trimesh.creation.torus(major_radius=rp + 1.3, minor_radius=1.8, major_sections=48, minor_sections=10)
    T = trimesh.geometry.align_vectors([0, 0, 1], n); T[:3, 3] = W(base); ring.apply_transform(T); parts.append(ring)
    tcs = [com[frozenset(p)] for p in pairs]
    for t in tcs:
        parts.append(tube([W(base + e1 * np.cos(t) * rp + e2 * np.sin(t) * rp), W(base + e1 * np.cos(t) * rp * 0.93 + e2 * np.sin(t) * rp * 0.93 + n * post_h)], 1.1, seg=8))
    for i in range(3):
        t0, t1 = tcs[i], tcs[(i + 1) % 3]; dd = ((t1 - t0) % (2 * np.pi)); Vv, F = [], []
        ts = t0 + np.linspace(0, dd, 13); rows = 6
        for t in ts:
            s = (t - t0) / dd; zz = post_h * (1 - np.sin(np.pi * s)) * 0.95 + 1.0
            a = base + e1 * np.cos(t) * rp * 0.95 + e2 * np.sin(t) * rp * 0.95 + n * zz; b = base + n * post_h * 0.7
            for q in range(rows):
                u_ = q / (rows - 1); Vv.append(a + (b - a) * u_ - n * 2.0 * np.sin(np.pi * u_))
        for j in range(len(ts) - 1):
            for q in range(rows - 1):
                a_, b_ = j * rows + q, (j + 1) * rows + q; F += [[a_, b_, b_ + 1], [a_, b_ + 1, a_ + 1]]
        parts.append(trimesh.Trimesh(np.array(Vv) - ctx['CARINA'], np.array(F), process=True))
    emit_mesh('av-prosthesis', f'Stented bioprosthesis (~{round(2 * rp)} mm)', 'cardiac', '#d9dcc8', trimesh.util.concatenate(parts), visible=False,
              note='Schematic: sewing ring on the annulus, three posts at the native commissures so none faces a coronary ostium.')
    # cannulas for AVR: two-stage venous through the right atrial appendage, LV vent through the right superior pulmonary vein, ostial cardioplegia
    ra_top = ra_mm[np.argmax(ra_mm @ (SUP * 0.6 + ANT * 0.8))]; ra_low = LM['can-ivc']
    emit_mesh('can-2stage', 'Two-stage venous cannula (RA appendage to IVC)', 'cardiac', CANNULA,
              tube([W(p) for p in [ra_low - SUP * 20, ra_low, (ra_top + ra_low) / 2 + RIGHT * 4, ra_top, ra_top + ANT * 40 + SUP * 25, ra_top + ANT * 90 + SUP * 40]], 4.2), opacity=0.9, visible=False,
              note='Schematic: through a purse-string in the right atrial appendage, the tip in the IVC, the side holes in the atrium.')
    rspv = la_mm[np.argmax(la_mm @ (RIGHT * 1.0 + SUP * 0.4 - ANT * 0.2))]
    emit_mesh('can-lvvent', 'LV vent (right superior pulmonary vein)', 'cardiac', CANNULA,
              tube([W(p) for p in [LM['mv-centre'] - DIRS['mv-normal'] * 20, LM['mv-centre'], rspv, rspv + RIGHT * 25 + ANT * 30, rspv + RIGHT * 40 + ANT * 90 + SUP * 20]], 1.7), opacity=0.9, visible=False,
              note='Schematic: through the right superior pulmonary vein, across the mitral valve into the LV. Keeps the arrested ventricle empty, above all with aortic regurgitation.')
    LM['rspv'] = rspv
    osts = []
    for k in ('l', 'r'):
        p = LM[f'ostium-{k}']; osts.append(tube([W(p), W(c + n * (hs + 8)), W(c + n * (hs + 60) + ANT * 40)], 1.3))
    emit_mesh('can-ostial', 'Hand-held ostial cardioplegia cannulas', 'cardiac', CANNULA, trimesh.util.concatenate(osts), opacity=0.9, visible=False,
              note='Schematic: soft-tipped cannulas held in the left main and right coronary ostia once the aorta is open.')
    # accesses: upper J-hemisternotomy and right anterior mini-thoracotomy (2nd space)
    st = ctx.get('sternum_mm')
    if st is not None and len(st):
        mid_x = float(np.median(st[:, 0])); top = float(st[:, 2].max()) - 3
        j_end = ctx['port'](3, 80, 'right') if ctx.get('port') is not None else None
        zj = float(j_end[2]) if j_end is not None else top - 60
        zs = np.linspace(top, zj + 6, 8); cut = []
        for z in zs:
            s = st[np.abs(st[:, 2] - z) < 2.5]; s = s[np.abs(s[:, 0] - mid_x) < 4] if len(s) else s
            cut.append(s[np.argmax(s[:, 1])] + ANT * 1.5 if len(s) else np.array([mid_x, st[:, 1].max(), z]))
        s = st[np.abs(st[:, 2] - zj) < 3]
        edge = s[np.argmax(s[:, 0])] + ANT * 1.5 if len(s) else cut[-1] + RIGHT * 15
        cut += [cut[-1] * 0.5 + edge * 0.5 - SUP * 3, edge]
        emit_mesh('hemi-cut', 'Upper hemisternotomy (J into the right 3rd/4th space)', 'incisions', '#d0433a', tube([W(p) for p in cut], 1.6), visible=False,
                  note='From the sternal notch down the midline to the 3rd or 4th space, then out to the right. The lower sternum stays whole; watch the right internal thoracic vessels at the J.')
        LM['hemi'] = cut[len(cut) // 2]
        sk = [p + ANT * 9 for p in cut[1:7]]
        emit_mesh('incision-hemi', 'Skin incision for upper hemisternotomy (about 6-8 cm)', 'incisions', '#ff6a5a', tube([W(p) for p in sk], 1.6), visible=False)
    if ctx.get('port') is not None:
        pts = np.array([ctx['port'](2, a, 'right') for a in np.linspace(82, 50, 5)])
        cr = ctx['lung_cr']; r_ = pts[:, :2] - cr[:2]; pts[:, :2] += r_ / np.linalg.norm(r_, axis=1, keepdims=True) * 3.0
        emit_mesh('incision-ramt', 'Right anterior mini-thoracotomy (2nd space)', 'incisions', '#d0433a', tube([W(p) for p in pts], 1.8), visible=False,
                  note='5-6 cm in the right 2nd space from the sternal edge; the right internal thoracic vessels ligated or kept, the 3rd costal cartilage divided if more room is needed.')
        LM['ramt'] = pts[1]
    print(f'  aortic root: radius {R:.1f} mm, STJ {hs:.0f} mm above the annulus, prosthesis ~{2 * rp:.0f} mm')


def tricuspid(ctx, RA, RV, mm, LM, DIRS, SC, ra_mm, lv_mm, csos, svc_p, ra_low):
    """The tricuspid valve from the RA-RV junction: annulus, the three leaflets (schematic), Koch's triangle with the AV
    node at its apex, the right coronary in the AV groove, an incomplete annuloplasty ring, a De Vega suture line, a
    bioprosthesis, and caval snares."""
    emit_mesh, W, tube, sphere = ctx['emit_mesh'], ctx['W'], ctx['tube'], ctx['sphere']
    iface = mm(RV & ndimage.binary_dilation(RA, iterations=2))
    if len(iface) < 50: print('  tricuspid: no RA-RV interface; skipped'); return
    c = iface.mean(0); _, _, vt = np.linalg.svd(iface - c, full_matrices=False)
    n = vt[2] * np.sign(np.dot(vt[2], ra_mm.mean(0) - c))                   # the annular plane's normal, toward the RA
    s = lv_mm.mean(0) - c; s -= n * np.dot(s, n); s /= np.linalg.norm(s)     # toward the septum (the LV lies beyond it)
    a_ = ANT - n * np.dot(ANT, n); a_ -= s * np.dot(a_, s); a_ /= np.linalg.norm(a_)
    v = np.cross(n, s)
    if np.dot(v, a_) < 0: v = -v                                             # angles increase from the septum toward the front
    rel = iface - c; ang = np.degrees(np.arctan2(rel @ v, rel @ s)) % 360; rad = np.hypot(rel @ s, rel @ v)
    ring = []
    for a0 in range(0, 360, 10):
        m_ = (ang >= a0) & (ang < a0 + 10)
        ring.append((a0 + 5, np.percentile(rad[m_], 80) if m_.sum() > 3 else np.nan, np.median(rel[m_] @ n) if m_.sum() > 3 else 0.0))
    ring = np.array(ring); ok = ~np.isnan(ring[:, 1])
    ring[:, 1] = np.interp(ring[:, 0], ring[ok, 0], ring[ok, 1], period=360)
    k5 = np.ones(5) / 5
    ring[:, 1] = np.convolve(np.r_[ring[-2:, 1], ring[:, 1], ring[:2, 1]], k5, 'valid'); ring[:, 2] = np.convolve(np.r_[ring[-2:, 2], ring[:, 2], ring[:2, 2]], k5, 'valid')
    R = float(np.clip(np.median(ring[:, 1]), 15, 25)); ring[:, 1] = np.clip(ring[:, 1], R * 0.8, R * 1.2)
    rr = lambda t: float(np.interp(t % 360, ring[:, 0], ring[:, 1], period=360)); hh = lambda t: float(np.interp(t % 360, ring[:, 0], ring[:, 2], period=360))
    at = lambda t, r=None, dn=0.0: c + s * np.cos(np.radians(t)) * (rr(t) if r is None else r) + v * np.sin(np.radians(t)) * (rr(t) if r is None else r) + n * (hh(t) + dn)
    ann = np.array([at(t) for t in range(0, 360, 6)])
    emit_mesh('tricuspid-annulus', 'Tricuspid annulus', 'cardiac', '#e9dec6', tube([W(p) for p in [*ann, ann[0]]], 1.6), visible=False,
              note='The RA-RV junction on this scan. In functional TR it dilates mainly along the anterior and posterior leaflets; the septal part is fixed by the fibrous skeleton.')
    # leaflets by sector (angle 0 = the septum, increasing toward the front): septal, anterior (the largest), posterior
    AS, AP, PS = 60.0, 200.0, 300.0                                            # commissures
    coapt = c - n * 10.0
    def leaflet(t0, t1):
        ts = np.linspace(t0, t1, max(4, int((t1 - t0) / 6))); Vv, F = [], []; rows = 7
        for t in ts:
            a = at(t); b = coapt + (a - coapt) * 0.12
            for q in range(rows):
                u_ = q / (rows - 1); Vv.append(a + (b - a) * u_ - n * 3.0 * np.sin(np.pi * u_))
        for j in range(len(ts) - 1):
            for q in range(rows - 1):
                a2, b2 = j * rows + q, (j + 1) * rows + q; F += [[a2, b2, b2 + 1], [a2, b2 + 1, a2 + 1]]
        return trimesh.Trimesh(np.array(Vv) - ctx['CARINA'], np.array(F), process=True)
    for id_, nm, t0, t1 in (('tv-septal', 'Septal leaflet (tricuspid)', PS - 360, AS), ('tv-anterior', 'Anterior leaflet (tricuspid)', AS, AP), ('tv-posterior', 'Posterior leaflet (tricuspid)', AP, PS)):
        emit_mesh(id_, nm, 'cardiac', '#efe3cf', leaflet(t0, t1), visible=False, note='Schematic.')
    # Koch's triangle: coronary sinus ostium, tendon of Todaro, septal leaflet hinge; the AV node at its apex near the anteroseptal commissure
    cso = csos if csos is not None else at(-35, dn=6)
    apex = at(AS - 20, dn=5.0); base = at(-10, dn=3.0)
    tod = cso + (apex - cso) * 1.0 + n * 3.0
    emit_mesh('koch', "Koch's triangle", 'cardiac', '#f2d24b', tube([W(p) for p in [cso, apex, base, cso]], 0.9), visible=False,
              note='Bounded by the coronary sinus ostium, the tendon of Todaro and the septal leaflet hinge. The AV node lies at its apex, near the anteroseptal commissure: no deep bites there.')
    emit_mesh('tv-avnode', 'AV node and His bundle (apex of Koch\'s triangle)', 'cardiac', '#f2d24b', sphere(W(apex), 3.8), visible=False)
    emit_mesh('cs-ostium', 'Coronary sinus ostium', 'cardiac', '#5e6a92', sphere(W(cso), 3.2), visible=False)
    LM['tv-avnode'] = apex; LM['cs-ostium'] = cso
    # the right coronary in the right AV groove, round the anterior and posterior annulus
    rca = [at(t, rr(t) + 8, dn=-3) for t in np.linspace(AS + 25, PS - 10, 16)]
    emit_mesh('rca-groove', 'Right coronary artery (right AV groove)', 'cardiac', '#d0433a', tube([W(p) for p in rca], 1.8), visible=False,
              note='Schematic: a few millimetres outside the anterior and posterior annulus, closest near the anteroposterior commissure. Deep bites or a downsized rigid ring can kink it.')
    # an incomplete annuloplasty ring (downsized), open at the AV node; a De Vega double running suture
    rr_ring = lambda t: rr(t) * 0.82
    rg = [c + s * np.cos(np.radians(t)) * rr_ring(t) + v * np.sin(np.radians(t)) * rr_ring(t) + n * (hh(t) + 1.2) for t in np.linspace(AS + 10, 360 + 15, 50)]
    emit_mesh('tv-ring', 'Incomplete annuloplasty ring', 'cardiac', '#cfd6dc', tube([W(p) for p in rg], 2.0), visible=False,
              note='Schematic: open at the anteroseptal commissure and the AV node; it reshapes and reduces the anterior and posterior annulus.')
    dv = [trimesh.util.concatenate([tube([W(at(t, rr(t) + o, dn=0.8)) for t in np.linspace(AS, PS, 30)], 0.45, seg=6)]) for o in (-1.2, 1.2)]
    emit_mesh('tv-devega', 'De Vega suture annuloplasty (double running)', 'cardiac', '#3fa7d6', trimesh.util.concatenate(dv), visible=False,
              note='Schematic: two parallel running polypropylene sutures from the anteroseptal to the posteroseptal commissure, tied over pledgets to shorten the anterior and posterior annulus.')
    # a stented bioprosthesis in the tricuspid position: posts at the commissures, none into the outflow tract
    rp = float(np.clip(R * 0.85, 13.5, 16.0)); base_ = c + n * 1.0; post_h = 16.0; parts = []
    tor = trimesh.creation.torus(major_radius=rp + 1.5, minor_radius=2.0, major_sections=48, minor_sections=10)
    T = trimesh.geometry.align_vectors([0, 0, 1], -n); T[:3, 3] = W(base_); tor.apply_transform(T); parts.append(tor)
    posts = (AS, AP, PS)
    for t in posts:
        d_ = s * np.cos(np.radians(t)) + v * np.sin(np.radians(t))
        parts.append(tube([W(base_ + d_ * rp), W(base_ + d_ * rp * 0.92 - n * post_h)], 1.2, seg=8))
    for i in range(3):
        t0 = posts[i]; t1 = posts[(i + 1) % 3] + (360 if i == 2 else 0); ts = np.linspace(t0, t1, 13); Vv, F = [], []; rows = 6
        for t in ts:
            f = (t - t0) / (t1 - t0); d_ = s * np.cos(np.radians(t)) + v * np.sin(np.radians(t))
            a = base_ + d_ * rp * 0.95 - n * (post_h * (1 - np.sin(np.pi * f)) * 0.95 + 1.0); b = base_ - n * post_h * 0.7
            for q in range(rows):
                u_ = q / (rows - 1); Vv.append(a + (b - a) * u_ + n * 2.0 * np.sin(np.pi * u_))
        for j in range(len(ts) - 1):
            for q in range(rows - 1):
                a2, b2 = j * rows + q, (j + 1) * rows + q; F += [[a2, b2, b2 + 1], [a2, b2 + 1, a2 + 1]]
        parts.append(trimesh.Trimesh(np.array(Vv) - ctx['CARINA'], np.array(F), process=True))
    emit_mesh('tv-prosthesis', f'Stented bioprosthesis, tricuspid (~{round(2 * rp)} mm)', 'cardiac', '#d9dcc8', trimesh.util.concatenate(parts), visible=False,
              note='Schematic: posts at the commissures, none pointing into the right ventricular outflow tract.')
    # caval snares
    sn = [trimesh.creation.torus(major_radius=11, minor_radius=1.2, major_sections=32, minor_sections=8) for _ in range(2)]
    for m_, p in zip(sn, (svc_p - SUP * 12, ra_low - SUP * 14)):
        m_.apply_translation(W(p))
    emit_mesh('snares', 'Caval snares (SVC and IVC)', 'cardiac', '#e8e3a0', trimesh.util.concatenate(sn), visible=False,
              note='Tapes round both cavae, tightened over the cannulas so that the right atrium can be opened without air entering the venous line.')
    SC['_rca'] = rca
    LM['tv-centre'] = c; DIRS['tv-normal'] = n; DIRS['tv-septal'] = s; SC['tv-radius'] = R
    print(f'  tricuspid annulus radius {R:.1f} mm; ring ~{2 * R * 0.82:.0f} mm; prosthesis ~{2 * rp:.0f} mm')


def _lathe(c, n, e1, prof, sections=40):
    """a surface of revolution about axis n through c: prof = [(height, radius), ...]"""
    e2 = np.cross(n, e1); V_, F = [], []
    ts = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    for h, r in prof:
        for t in ts: V_.append(c + n * h + (e1 * np.cos(t) + e2 * np.sin(t)) * r)
    m = len(ts)
    for i in range(len(prof) - 1):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m; F += [[a, b, b + m], [a, b + m, a + m]]
    return np.array(V_), np.array(F)


def root_repl(ctx, mm, H, LM, DIRS, SC):
    """Aortic root replacement: a schematic aneurysmal root, the composite valved graft (Bentall), coronary buttons on the
    sinuses and reimplanted on the graft; for the Ross: the pulmonary root (autograft) from the RV-PA junction, the first
    septal perforator, the autograft in the aortic position, and a pulmonary homograft."""
    emit_mesh, W, tube, sphere = ctx['emit_mesh'], ctx['W'], ctx['tube'], ctx['sphere']
    if 'av-centre' not in LM: return
    c, n, e1, R = LM['av-centre'], DIRS['av-axis'], DIRS['av-e1'], SC['av-radius']
    CA = ctx['CARINA']
    mk = lambda V_, F: trimesh.Trimesh(V_ - CA, F, process=True)
    # the aneurysmal root (schematic, about 55 mm at the sinuses), to show the indication and to be excised
    prof = [(-1, R + 1.5), (6, R + 9), (14, 27.0), (24, 26.0), (34, 22.0), (44, 18.0)]
    emit_mesh('root-aneurysm', 'Aortic root aneurysm (schematic, ~55 mm at the sinuses)', 'cardiac', '#c96a5a', mk(*_lathe(c, n, e1, prof)), opacity=0.45, visible=False,
              note='Schematic: sinuses of Valsalva dilated, effacing the sinotubular junction. The CT root itself is normal size.')
    # coronary buttons: a cuff of sinus wall round each ostium; their new positions on the graft
    gr = R + 2.5; gtop = 46.0
    for k in ('l', 'r'):
        if f'ostium-{k}' not in LM: continue
        p = LM[f'ostium-{k}']; d_ = (p - c) - n * np.dot(p - c, n); d_ /= np.linalg.norm(d_)
        tor = trimesh.creation.torus(major_radius=5.5, minor_radius=1.1, major_sections=28, minor_sections=8)
        T = trimesh.geometry.align_vectors([0, 0, 1], d_); T[:3, 3] = W(p); tor.apply_transform(T)
        emit_mesh(f'button-{k}', f'{"Left main" if k == "l" else "Right coronary"} button', 'cardiac', '#e38b6f', tor, visible=False,
                  note='A 5-8 mm cuff of sinus wall round the ostium, mobilised just enough to reach the graft without tension.')
        h_new = 14.0 if k == 'l' else 17.0                                     # the right a little higher: it kinks if placed low
        q = c + n * h_new + d_ * gr
        tor2 = trimesh.creation.torus(major_radius=5.0, minor_radius=0.9, major_sections=28, minor_sections=8)
        T2 = trimesh.geometry.align_vectors([0, 0, 1], d_); T2[:3, 3] = W(q); tor2.apply_transform(T2)
        emit_mesh(f'button-{k}-graft', f'{"Left main" if k == "l" else "Right coronary"} button, reimplanted', 'cardiac', '#e38b6f',
                  trimesh.util.concatenate([tor2, tube([W(q), W(q + d_ * 6), W(p + d_ * 4)], 1.8)]), visible=False,
                  note='Sewn end-to-side to a hole in the graft with running 5-0 polypropylene. The right button is placed with the heart filled, to avoid kinking.')
        LM[f'button-{k}-graft'] = q
    # the composite valved graft: a crimped polyester tube with a valve at its base
    prof_g = [(h, gr) for h in np.linspace(0, gtop, 24)]
    gv, gf = _lathe(c, n, e1, prof_g, 36)
    parts = [mk(gv, gf)]
    for h in np.linspace(3, gtop - 3, 14):
        rg = trimesh.creation.torus(major_radius=gr + 0.3, minor_radius=0.5, major_sections=36, minor_sections=6)
        T = trimesh.geometry.align_vectors([0, 0, 1], n); T[:3, 3] = W(c + n * h); rg.apply_transform(T); parts.append(rg)
    sw = trimesh.creation.torus(major_radius=gr + 1.0, minor_radius=1.8, major_sections=40, minor_sections=8)
    T = trimesh.geometry.align_vectors([0, 0, 1], n); T[:3, 3] = W(c + n * 0.5); sw.apply_transform(T); parts.append(sw)
    for sx in (-1, 1):
        leaf = trimesh.creation.box(extents=[gr * 0.95, 0.8, 5.0]); leaf.apply_translation([sx * gr * 0.45, 0, 2.5])
        leaf.apply_transform(trimesh.transformations.rotation_matrix(np.radians(75) * sx, [0, 1, 0], point=[sx * gr * 0.9, 0, 2.5]))
        Tl = trimesh.geometry.align_vectors([0, 0, 1], n); Tl[:3, 3] = W(c); leaf.apply_transform(Tl); parts.append(leaf)
    emit_mesh('cvg', f'Composite valved graft (~{round(2 * gr)} mm tube, mechanical valve)', 'cardiac', '#f1f1ea', trimesh.util.concatenate(parts), opacity=0.85, visible=False,
              note='Schematic: a crimped polyester graft with the valve sewn into its base (or a tissue valve: a "bio-Bentall").')
    dist = c + n * gtop
    rd = trimesh.creation.torus(major_radius=gr + 1.2, minor_radius=1.3, major_sections=40, minor_sections=8)
    T = trimesh.geometry.align_vectors([0, 0, 1], n); T[:3, 3] = W(dist); rd.apply_transform(T)
    emit_mesh('root-distal', 'Distal anastomosis (graft to ascending aorta)', 'cardiac', '#3fa7d6', rd, visible=False)
    LM['root-distal'] = dist; LM['root-top'] = c + n * 30
    # ---------------------------------------------------------------- David reimplantation: the valve kept, inside a polyester graft
    hs = SC.get('av-stj', 18.0); dr = R + 4.0; dbase = -5.0
    dv, df = _lathe(c, n, e1, [(h, dr) for h in np.linspace(dbase, gtop, 26)], 36)
    parts = [mk(dv, df)]
    for h in np.linspace(dbase + 3, gtop - 3, 15):
        rg = trimesh.creation.torus(major_radius=dr + 0.3, minor_radius=0.5, major_sections=36, minor_sections=6)
        T = trimesh.geometry.align_vectors([0, 0, 1], n); T[:3, 3] = W(c + n * h); rg.apply_transform(T); parts.append(rg)
    emit_mesh('david-graft', f'Reimplantation graft (~{round(2 * dr)} mm), the native valve inside', 'cardiac', '#f1f1ea', trimesh.util.concatenate(parts), opacity=0.55, visible=False,
              note='Schematic: a straight or Valsalva-shaped polyester graft, its base below the valve at the ventriculo-aortic junction; the valve is sewn inside it.')
    e2 = np.cross(n, e1); ring = []
    for t in np.linspace(0, 2 * np.pi, 13)[:-1]:
        d = e1 * np.cos(t) + e2 * np.sin(t); q = c + n * dbase + d * (R - 1.0); o = c + n * dbase + d * (dr + 1.5)
        ring.append(tube([W(q), W(o)], 0.45, seg=6)); ring.append(sphere(W(o), 1.3))
    ring.append(tube([W(c + n * dbase + (e1 * np.cos(t) + e2 * np.sin(t)) * (dr + 1.5)) for t in np.linspace(0, 2 * np.pi, 49)], 0.6, seg=6))
    emit_mesh('david-subannular', 'Subannular sutures (horizontal mattress, inside out)', 'cardiac', '#3fa7d6', trimesh.util.concatenate(ring), visible=False,
              note='A single horizontal plane just below the nadirs of the cusps, passed from inside the LVOT out through the base of the graft; shallow (or through the fibrous tissue) under the membranous septum.')
    posts = []
    for k in range(3):
        if f'comm-{k}' not in LM: continue
        q = LM[f'comm-{k}']; d = (q - c) - n * np.dot(q - c, n); d /= np.linalg.norm(d)
        top = c + n * (hs * 0.95) + d * (dr - 0.4)
        posts.append(tube([W(q), W(top)], 0.9, seg=8)); posts.append(sphere(W(top), 1.6))
    if posts:
        emit_mesh('david-commissures', 'Commissures resuspended inside the graft', 'cardiac', '#3fa7d6', trimesh.util.concatenate(posts), visible=False,
                  note='Each commissure pulled up vertically and fixed to the graft, at a height that lets the cusps coapt well above the annulus.')
    LM['david-base'] = c + n * dbase
    # ---------------------------------------------------------------- the pulmonary root: the RV-PA junction
    RV, PA = H == 5, H == 7
    if RV.sum() < 50 or PA.sum() < 50: print('  root: no pulmonary artery; Ross skipped'); return
    iface = mm(PA & ndimage.binary_dilation(RV, iterations=2))
    if len(iface) < 20: print('  root: no RV-PA junction; Ross skipped'); return
    pc = iface.mean(0); _, _, vt = np.linalg.svd(iface - pc, full_matrices=False)
    pa_mm = mm(PA); pn = vt[2] * np.sign(np.dot(vt[2], pa_mm.mean(0) - pc))
    rel = iface - pc; pr = float(np.clip(np.percentile(np.linalg.norm(rel - np.outer(rel @ pn, pn), axis=1), 85), 10, 14))
    pe1 = np.cross(pn, SUP); pe1 = pe1 / np.linalg.norm(pe1) if np.linalg.norm(pe1) > 0.1 else e1
    prof_a = [(-5, pr + 1.0), (0, pr + 1.2), (6, pr + 2.8), (13, pr + 3.0), (20, pr + 1.2), (24, pr + 0.8)]
    av_, af_ = _lathe(pc, pn, pe1, prof_a, 36)
    emit_mesh('pa-root', 'Pulmonary root (the autograft)', 'cardiac', '#6f86c9', mk(av_, af_), opacity=0.8, visible=False,
              note='The pulmonary valve with 3-5 mm of RV muscle below it and the trunk above it, harvested as a whole root.')
    # harvest lines: the trunk just below the bifurcation; the RVOT a few millimetres below the valve
    ring = lambda cc, nn, ee, r, h: [cc + nn * h + (ee * np.cos(t) + np.cross(nn, ee) * np.sin(t)) * r for t in np.linspace(0, 2 * np.pi, 33)]
    emit_mesh('pa-harvest', 'Autograft harvest lines (PA trunk above, RVOT below)', 'cardiac', '#d0433a',
              trimesh.util.concatenate([tube([W(p) for p in ring(pc, pn, pe1, pr + 1.6, 24)], 1.0), tube([W(p) for p in ring(pc, pn, pe1, pr + 1.8, -5)], 1.0)]), visible=False,
              note='Transect the trunk below the bifurcation; open the RVOT 3-5 mm below the valve and free the root off the septum, staying shallow posteriorly.')
    # the first septal perforator: from the LAD, runs just below and behind the pulmonary root into the septum
    lv = mm(H == 3); toward = lv.mean(0) - pc; toward -= pn * np.dot(toward, pn); toward /= np.linalg.norm(toward)
    s0 = pc - pn * 7 + toward * (pr + 4) + ANT * 3
    sp_ = [s0, s0 - pn * 4 + toward * 6, s0 - pn * 9 + toward * 14 - ANT * 4, s0 - pn * 13 + toward * 22 - ANT * 8]
    emit_mesh('septal-perforator', 'First septal perforator (LAD)', 'cardiac', '#d0433a', tube([W(p) for p in sp_], 1.3), visible=False,
              note='Schematic: leaves the LAD and runs into the septum just beneath the posterior RVOT. Deep dissection here during harvest divides it (septal infarction).')
    LM['pa-root'] = pc; LM['septal-perforator'] = sp_[1]; DIRS['pa-axis'] = pn; SC['pa-radius'] = pr
    # the autograft in the aortic position, and the homograft in the pulmonary position
    prof_ao = [(h - 5 * 0 , r) for h, r in [(-1, R + 0.8), (4, R + 2.6), (12, R + 3.0), (20, R + 1.6), (40, gr)]]
    emit_mesh('autograft-ao', 'Pulmonary autograft in the aortic position', 'cardiac', '#6f86c9', mk(*_lathe(c, n, e1, prof_ao, 36)), opacity=0.8, visible=False,
              note='Sewn to the aortic annulus with interrupted sutures, the coronary buttons reimplanted into its sinuses, then joined to the ascending aorta. Often reinforced (inclusion in a polyester graft, or annular and STJ stabilisation) to prevent dilatation.')
    hv, hf = _lathe(pc, pn, pe1, [(-4, pr + 1.4), (2, pr + 2.6), (12, pr + 2.6), (24, pr + 1.4)], 36)
    emit_mesh('homograft', 'Pulmonary homograft (RVOT reconstruction)', 'cardiac', '#b7c6a2', mk(hv, hf), opacity=0.85, visible=False,
              note='A cryopreserved pulmonary homograft sewn to the RVOT below and the PA trunk above.')
    LM['autograft-ao'] = c + n * 10
    print(f'  root: graft ~{2 * gr:.0f} mm; pulmonary root radius {pr:.1f} mm')


def _surf_path(pts, S, out_c, lift=1.6):
    """snap a path to the epicardial surface (nearest surface point), lifted a little outward from the heart centre"""
    from scipy.spatial import cKDTree
    t = cKDTree(S); res = []
    for p in pts:
        q = S[t.query(p)[1]]; d = q - out_c; res.append(q + d / (np.linalg.norm(d) + 1e-6) * lift)
    return np.array(res)


def _smooth(P, k=2):
    P = np.array(P, float)
    for _ in range(k): P[1:-1] = (P[:-2] + 2 * P[1:-1] + P[2:]) / 4
    return P


def coronary_tree(ctx, H, mm, LM, DIRS, SC, A):
    """The coronary tree for CABG: left main, LAD with diagonals, circumflex with obtuse marginals, right coronary with an
    acute marginal and the PDA (right dominant). The LAD and PDA follow the interventricular grooves, found on the
    epicardium where the LV and RV territories meet; the circumflex and right coronary follow the AV grooves (the
    mitral and tricuspid annuli). The proximal left system is anchored on the segmented coronaries; the rest is
    schematic. Also: three typical lesions, the distal targets, conduits (LIMA, saphenous vein grafts), proximal
    anastomoses on the aorta, and an off-pump stabiliser."""
    emit_mesh, W, tube, sphere = ctx['emit_mesh'], ctx['W'], ctx['tube'], ctx['sphere']
    need = ('ostium-l', 'ostium-r', 'mv-centre')
    if any(k not in LM for k in need) or '_ann' not in SC or '_rca' not in SC: print('  coronary tree: missing landmarks; skipped'); return
    sp = np.abs(np.diag(A)[:3])
    LVc, RVc, MY = H == 3, H == 5, H == 1
    V = LVc | RVc | MY
    V = ndimage.binary_closing(V, iterations=2)
    S_mask = V & ~ndimage.binary_erosion(V)
    S = mm(S_mask)
    dLV = ndimage.distance_transform_edt(~LVc, sampling=sp)[S_mask]; dRV = ndimage.distance_transform_edt(~RVc, sampling=sp)[S_mask]
    hc = mm(V).mean(0)
    mvc, mvn = LM['mv-centre'], DIRS['mv-normal']; apex = SC['_apex']; L = float(np.linalg.norm(apex - mvc))
    tpar = lambda p: float(np.dot(np.asarray(p) - mvc, -mvn) / L)
    ts = (S - mvc) @ -mvn / L
    # the interventricular grooves: where the LV and RV lie equally deep beneath the epicardium
    sep = np.abs(dLV - dRV) < 3.0
    g = S[sep]; gt = ts[sep]
    front = (g - hc) @ ANT > 0
    def groove(sel, t0, t1):
        pts = g[sel]; tt = gt[sel]; line = []
        for a in np.linspace(t0, t1, 14):
            m = np.abs(tt - a) < 0.05
            if m.sum() > 3: line.append(np.median(pts[m], 0))
        return np.array(line)
    aivg = groove(front, 0.12, 1.0); pivg = groove(~front, 0.12, 0.85)
    if len(aivg) < 5 or len(pivg) < 4: print('  coronary tree: grooves not found; skipped'); return
    aivg = _surf_path(_smooth(aivg), S, hc); pivg = _surf_path(_smooth(pivg), S, hc)
    ol, orr = LM['ostium-l'], LM['ostium-r']
    # left main: from the ostium toward the top of the anterior groove, behind the pulmonary trunk; bifurcates after ~12 mm
    lm_dir = aivg[0] - ol; B = ol + lm_dir / np.linalg.norm(lm_dir) * min(12.0, 0.4 * np.linalg.norm(lm_dir))
    lad = np.array([ol, B, *aivg, apex + (apex - hc) / np.linalg.norm(apex - hc) * 2 - mvn * 4])
    lad = _smooth(lad, 1)
    # the circumflex: into the left AV groove (the lateral and posterior mitral annulus, 7 mm out, a little toward the LA)
    ann = SC['_ann']; ring_c = ann.mean(0)
    out = lambda p: p + (p - ring_c) / np.linalg.norm(p - ring_c) * 7 + mvn * 2
    k0 = int(np.argmin(np.linalg.norm(ann - B, axis=1))); lat = np.cross(mvn, ANT)                  # toward the patient's left? check sign below
    if np.dot(lat, -RIGHT) < 0: lat = -lat
    step = 1 if np.dot(ann[(k0 + 1) % len(ann)] - ann[k0], lat) > 0 else -1                           # go round toward the left, then back
    lcx_ring = [out(ann[(k0 + step * i) % len(ann)]) for i in range(1, int(len(ann) * 0.42))]
    lcx = _smooth(np.array([ol, B, *lcx_ring]), 2)
    # the right coronary: from its ostium into the right AV groove, round to the crux, then the PDA down the posterior groove
    rca_g = np.array(SC['_rca'])
    if np.linalg.norm(rca_g[0] - orr) > np.linalg.norm(rca_g[-1] - orr): rca_g = rca_g[::-1]
    crux = rca_g[-1]; pd = pivg if np.linalg.norm(pivg[0] - crux) < np.linalg.norm(pivg[-1] - crux) else pivg[::-1]
    rca = _smooth(np.array([orr, orr + (rca_g[0] - orr) * 0.5, *rca_g]), 2)
    pda = _smooth(np.array([crux, *pd]), 1)
    # branches: diagonals and obtuse marginals toward the apex over the LV free wall, an acute marginal over the RV
    lv_s = S[(dLV < dRV - 4)]; rv_s = S[(dRV < dLV - 4)]
    def branch(start, pool, t_end, away):
        t0 = tpar(start); cand = pool[np.abs((pool - mvc) @ -mvn / L - t_end) < 0.05]
        if len(cand) < 5: return None
        score = (cand - start) @ away - 0.02 * np.linalg.norm(cand - start, axis=1) ** 1.0
        end = cand[np.argmax(score)]
        mids = [start + (end - start) * f for f in (0.33, 0.66)]
        return _smooth(np.array([start, *_surf_path(mids, S, hc), _surf_path([end], S, hc)[0]]), 1)
    left = -RIGHT
    idx = lambda P, f: P[int(f * (len(P) - 1))]
    d1 = branch(idx(lad, 0.22), lv_s, 0.55, left + ANT * 0.2); d2 = branch(idx(lad, 0.42), lv_s, 0.75, left)
    om1 = branch(idx(lcx, 0.45), lv_s, 0.65, -mvn + left * 0.3); om2 = branch(idx(lcx, 0.7), lv_s, 0.7, -mvn - ANT * 0.3)
    am = branch(idx(rca, 0.55), rv_s, 0.7, -mvn + RIGHT * 0.2)
    cor = [('cor-lm', 'Left main coronary artery', np.array([ol, B]), 2.1), ('cor-lad', 'Left anterior descending (LAD)', lad, 1.6),
           ('cor-lcx', 'Circumflex (LCx)', lcx, 1.5), ('cor-rca', 'Right coronary artery (RCA)', rca, 1.7), ('cor-pda', 'Posterior descending (PDA)', pda, 1.2)]
    for id_, nm, P, r in cor:
        emit_mesh(id_, nm, 'cardiac', '#c0392b', tube([W(p) for p in P], r), visible=False, note='Epicardial course schematic along the grooves of this heart; proximal left system from the CT.')
    for id_, nm, P in (('cor-d1', 'First diagonal (D1)', d1), ('cor-d2', 'Second diagonal (D2)', d2), ('cor-om1', 'First obtuse marginal (OM1)', om1),
                       ('cor-om2', 'Second obtuse marginal (OM2)', om2), ('cor-am', 'Acute marginal', am)):
        if P is not None: emit_mesh(id_, nm, 'cardiac', '#c0392b', tube([W(p) for p in P], 1.0), visible=False, note='Schematic branch.')
    # lesions (a typical three-vessel pattern) and distal targets
    def lesion(P, f, r, id_, nm):
        p = idx(P, f); q = idx(P, min(1.0, f + 0.03)); d = (q - p) / (np.linalg.norm(q - p) + 1e-6)
        tor = trimesh.creation.torus(major_radius=r, minor_radius=r * 0.55, major_sections=24, minor_sections=8)
        T = trimesh.geometry.align_vectors([0, 0, 1], d); T[:3, 3] = W(p); tor.apply_transform(T)
        emit_mesh(id_, nm, 'cardiac', '#f1e3b0', tor, visible=False, note='Schematic plaque: a severe (70% or more) stenosis.')
        LM[id_] = p
    lesion(lad, 0.12, 2.0, 'lesion-lad', 'Proximal LAD stenosis')
    if om1 is not None: lesion(om1, 0.15, 1.4, 'lesion-om', 'OM1 stenosis')
    lesion(rca, 0.45, 2.1, 'lesion-rca', 'Mid RCA stenosis')
    tg = {'lad': idx(lad, 0.62), 'om': idx(om1, 0.5) if om1 is not None else idx(lcx, 0.6), 'pda': idx(pda, 0.35)}
    for k, p in tg.items():
        LM[f'target-{k}'] = p
        emit_mesh(f'target-{k}', f'Distal anastomosis site ({k.upper()})', 'cardiac', '#3fa7d6', sphere(W(p), 2.6), visible=False)
    # conduits: in-situ LIMA behind the chest wall; LIMA to LAD; vein grafts from the ascending aorta to OM and PDA
    st = ctx.get('sternum_mm')
    if st is None or not len(st): return
    lb = st[:, 0].min() - 12; back = np.percentile(st[:, 1], 10) - 4
    zs = np.linspace(st[:, 2].max() + 20, st[:, 2].min() + 25, 9)
    lima_in = np.array([[lb - (4 if i == 0 else 0), back, z] for i, z in enumerate(zs)])
    emit_mesh('lima-insitu', 'Left internal mammary (thoracic) artery, in situ', 'cardiac', '#c0392b', tube([W(p) for p in lima_in], 1.3), visible=False,
              note='About 1-2 cm from the sternal edge, on the back of the chest wall, with its two veins; harvested from the subclavian origin to the bifurcation at the 6th space.')
    T_ = tg['lad']; mid = (lima_in[3] + T_) / 2 + ANT * 12 - RIGHT * 8
    lima_g = _smooth(np.array([*lima_in[:3], lima_in[3] * 0.6 + mid * 0.4, mid, T_ + ANT * 6 - RIGHT * 3, T_]), 2)
    emit_mesh('graft-lima', 'LIMA to LAD graft (pedicle)', 'cardiac', '#c0392b', tube([W(p) for p in lima_g], 1.6), visible=False,
              note='The in-situ LIMA, divided distally and brought down, lateral to the pulmonary artery, to the LAD. The best-proven graft.')
    ao = ctx['aorta_mm']; zc = float(np.percentile(ao[:, 2], 55))
    s_ = ao[(np.abs(ao[:, 2] - zc) < 6) & (ao[:, 1] > np.percentile(ao[:, 1], 70))]
    pa1 = s_[np.argmax(s_ @ (ANT + RIGHT * 0.4))] if len(s_) else LM['cp'] - SUP * 10
    pa2 = pa1 - SUP * 9 + RIGHT * 3
    for id_, p in (('prox-om', pa1), ('prox-pda', pa2)):
        rg = trimesh.creation.torus(major_radius=3.0, minor_radius=0.8, major_sections=24, minor_sections=8)
        T = trimesh.geometry.align_vectors([0, 0, 1], ANT); T[:3, 3] = W(p); rg.apply_transform(T)
        emit_mesh(id_, 'Proximal anastomosis on the ascending aorta', 'cardiac', '#3fa7d6', rg, visible=False)
        LM[id_] = p
    # vein to OM: leftward over the pulmonary trunk, round the left side of the heart
    to = tg['om']; via = hc + (to - hc) * 1.35 + ANT * 12 + SUP * 20
    svg_om = _smooth(np.array([pa1, pa1 + ANT * 12 - RIGHT * 10 + SUP * 6, (pa1 + via) / 2 + ANT * 18, via, to + (to - hc) / np.linalg.norm(to - hc) * 6, to]), 2)
    tp = tg['pda']; via2 = orr + RIGHT * 28 + ANT * 14 - SUP * 10; via3 = hc + (tp - hc) * 1.3 + RIGHT * 10
    svg_pda = _smooth(np.array([pa2, pa2 + ANT * 12 + RIGHT * 8, via2, via3, tp + (tp - hc) / np.linalg.norm(tp - hc) * 6, tp]), 2)
    emit_mesh('graft-svg-om', 'Saphenous vein graft: aorta to OM1', 'cardiac', '#7d5a8c', tube([W(p) for p in svg_om], 2.2), visible=False,
              note='Reversed long saphenous vein; the course round the left side of the heart is judged with the heart full.')
    emit_mesh('graft-radial-om', 'Radial artery graft: aorta to OM1', 'cardiac', '#b8453a', tube([W(p) for p in svg_om], 1.5), visible=False,
              note='The radial artery (non-dominant arm), proximal end on the aorta; for a target with a severe stenosis.')
    # MIDCAB: a short left anterior thoracotomy over the LAD, 4th or 5th space
    if ctx.get('port') is not None:
        mid = np.array([ctx['port'](4, a, 'left') for a in np.linspace(78, 45, 5)])
        cr = mid.mean(0); mid[:, 1] += 2.5
        emit_mesh('incision-midcab', 'Left anterior mini-thoracotomy (MIDCAB, 4th space)', 'incisions', '#d0433a', tube([W(p) for p in mid], 1.8), visible=False,
                  note='6-8 cm in the 4th (or 5th) space, from near the sternal edge laterally; the LIMA is taken down under direct vision or thoracoscopically.')
        LM['midcab'] = mid[2]
    emit_mesh('graft-svg-pda', 'Saphenous vein graft: aorta to PDA', 'cardiac', '#7d5a8c', tube([W(p) for p in svg_pda], 2.2), visible=False,
              note='Reversed long saphenous vein round the acute margin to the inferior wall.')
    # off-pump stabiliser on the LAD target: two pads either side, on an arm from the sternal retractor
    tgt = tg['lad']; d = idx(lad, 0.66) - idx(lad, 0.58); d /= np.linalg.norm(d); o = np.cross(d, tgt - hc); o /= np.linalg.norm(o)
    nrm = (tgt - hc) / np.linalg.norm(tgt - hc)
    pads = [tube([W(tgt + o * s * 6 - d * 9 + nrm * 1.5), W(tgt + o * s * 6 + d * 9 + nrm * 1.5)], 1.6) for s in (-1, 1)]
    arm = tube([W(tgt + nrm * 4), W(tgt + nrm * 25 + ANT * 20), W(tgt + nrm * 60 + ANT * 70 + RIGHT * 30)], 1.5)
    emit_mesh('stabilizer', 'Off-pump stabiliser (suction pads on the target)', 'cardiac', '#b8c2cc', trimesh.util.concatenate([*pads, arm]), visible=False,
              note='Immobilises a few centimetres of epicardium around the target; an intracoronary shunt or snare keeps the field dry.')
    DIRS['lad-dir'] = d; LM['cor-lad-mid'] = idx(lad, 0.5)
    print(f'  coronary tree: LAD {len(lad)} pts, PDA {len(pda)} pts, branches ' + ', '.join(k for k, v in (('D1', d1), ('D2', d2), ('OM1', om1), ('OM2', om2), ('AM', am)) if v is not None))
