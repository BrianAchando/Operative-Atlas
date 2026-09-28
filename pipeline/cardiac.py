"""The heart for the cardiac module: mitral valve replacement and its accesses.

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
            ('can-ivc', 'IVC cannula', [ra_low, ra_low + RIGHT * 22 + ANT * 20, ra_low + RIGHT * 40 + ANT * 90 - SUP * 10], 4.0),
            ('can-cp', 'Antegrade cardioplegia needle (aortic root)', [cp + ANT * 1, cp + ANT * 25 + SUP * 10, cp + ANT * 80 + SUP * 25], 1.6)]
    for id_, name, pts, r in cans:
        emit_mesh(id_, name, 'cardiac', CANNULA, tube([W(p) for p in pts], r), opacity=0.9, visible=False, note='Schematic.')
    csos = cs[0] if np.linalg.norm(cs[0] - ra_mm.mean(0)) < np.linalg.norm(cs[-1] - ra_mm.mean(0)) else cs[-1]
    emit_mesh('can-retro', 'Retrograde cardioplegia cannula (coronary sinus)', 'cardiac', CANNULA,
              tube([W(p) for p in [csos, (csos + LM['ra-incision']) / 2 + RIGHT * 10, LM['ra-incision'] + RIGHT * 40 + ANT * 60]], 1.8), opacity=0.9, visible=False,
              note='Schematic: passed through a purse-string in the right atrium into the coronary sinus.')
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
    print(f'  annulus radius {R:.1f} mm; prosthesis ~{2 * rp:.0f} mm')
    return LM, DIRS, SC
