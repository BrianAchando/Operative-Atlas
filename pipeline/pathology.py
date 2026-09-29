"""Valve pathology for the case scenarios and the pathophysiology steps (schematic, on this heart's landmarks).

Mitral:  the chronic rheumatic valve (fused commissures, a thickened funnel with a fish-mouth orifice, calcium, short
         thick fused chordae), the acute rheumatic lesion (anterior leaflet prolapse on elongated chordae), a vegetation,
         a left atrial appendage thrombus, and the jets (mitral stenosis inflow, mitral regurgitation).
Aortic:  the rheumatic valve (retracted, thickened cusps with rolled edges, a central gap, one fused commissure), the
         aortic regurgitant jet, a vegetation, and a paravalvular (root) abscess.
Called from cardiac.build once the mitral annulus, papillary muscles and aortic root are known.
"""
from __future__ import annotations

import numpy as np
import trimesh


def _sheet(ctx, rows_pts):
    """a quad sheet from a list of rows (each a list of points, same length)"""
    V = np.array([p for row in rows_pts for p in row]); nr, nc = len(rows_pts), len(rows_pts[0]); F = []
    for i in range(nr - 1):
        for j in range(nc - 1):
            a, b = i * nc + j, (i + 1) * nc + j; F += [[a, b, b + 1], [a, b + 1, a + 1]]
    return trimesh.Trimesh(V - ctx['CARINA'], np.array(F), process=True)


def _blob(ctx, c, r, seed, lumps=7, spread=0.7, elong=None):
    """an irregular mass (thrombus, vegetation): overlapping spheres around c"""
    rng = np.random.default_rng(seed); parts = []
    for k in range(lumps):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        if elong is not None: d = d * 0.5 + elong * rng.uniform(-1, 1)
        s = trimesh.creation.icosphere(subdivisions=3, radius=r * rng.uniform(0.45, 0.8) if k else r * 0.8)
        s.apply_translation(np.asarray(c, float) + (d * r * spread * rng.uniform(0.2, 1.0) if k else 0) - ctx['CARINA']); parts.append(s)
    return trimesh.util.concatenate(parts)


def _cone(ctx, start, end, r0, r1, sections=28):
    """a jet: narrow at start (the vena contracta), widening to end"""
    start, end = np.asarray(start, float), np.asarray(end, float); ax = end - start; L = float(np.linalg.norm(ax)); z = ax / L
    t = np.linspace(0, 1, 10); ang = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    ref = np.array([0, 0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0, 0]); x = np.cross(z, ref); x /= np.linalg.norm(x); y = np.cross(z, x)
    V = [start + z * L * s + (x * np.cos(a) + y * np.sin(a)) * (r0 + (r1 - r0) * s ** 0.8) for s in t for a in ang]
    F = []
    for i in range(len(t) - 1):
        for j in range(sections):
            a, b = i * sections + j, i * sections + (j + 1) % sections; F += [[a, a + sections, b + sections], [a, b + sections, b]]
    cap = len(V); V.append(end)
    for j in range(sections): F.append([(len(t) - 1) * sections + j, (len(t) - 1) * sections + (j + 1) % sections, cap])
    return trimesh.Trimesh(np.array(V) - ctx['CARINA'], np.array(F), process=True)


def build(ctx, LM, DIRS, SC):
    emit_mesh, W, tube, sphere = ctx['emit_mesh'], ctx['W'], ctx['tube'], ctx['sphere']
    if 'mv-centre' not in LM or '_ann' not in SC: return
    c, n, u = LM['mv-centre'], DIRS['mv-normal'], DIRS['mv-anterior']; v = np.cross(n, u); R = SC['mv-radius']; ann = SC['_ann']
    t_of = lambda p: float(np.arctan2((p - c) @ v, (p - c) @ u))
    order = sorted(range(len(ann)), key=lambda i: t_of(ann[i])); A = ann[order]; T = np.array([t_of(p) for p in A])
    # ---------------------------------------------------------------- chronic rheumatic mitral stenosis
    oc = c - n * 13.0 - u * R * 0.12                                          # the orifice, displaced into the LV (diastolic doming)
    oa, ob = min(8.0, R * 0.55), 3.4                                          # fish-mouth: long axis commissure to commissure (v); ~0.9 cm2
    orif = lambda t: oc + v * np.sin(t) * oa + u * np.cos(t) * ob
    rows = []
    for s in np.linspace(0, 1, 9):
        rows.append([a + (orif(t) - a) * s - n * 3.2 * np.sin(np.pi * s) + (u * np.cos(t) + v * np.sin(t)) * 1.2 * np.sin(np.pi * s) for a, t in zip([*A, A[0]], [*T, T[0] + 2 * np.pi])])
    emit_mesh('rh-mv', 'Rheumatic mitral valve: fused, thickened, fish-mouth orifice', 'cardiac', '#dcc9a6', _sheet(ctx, rows), visible=False,
              note='Schematic chronic rheumatic mitral stenosis: commissural fusion turns the two leaflets into a funnel that domes into the LV in diastole, with a small fish-mouth orifice (here about 0.9 cm2; normal 4-6 cm2).')
    rim = [orif(t) for t in np.linspace(-np.pi, np.pi, 40)]
    fused = [tube([W(A[int(np.argmin(np.abs(((T - t0 + np.pi) % (2 * np.pi)) - np.pi)))]), W(orif(t0) + n * 1.0)], 1.6) for t0 in (np.pi / 2, -np.pi / 2)]
    emit_mesh('rh-mv-edge', 'Thickened, rolled leaflet edges and fused commissures', 'cardiac', '#c9b27f',
              trimesh.util.concatenate([tube([W(p) for p in [*rim, rim[0]]], 1.5), *fused]), visible=False,
              note='The fused commissures (the ridges running up to the annulus) are what a balloon or a surgical commissurotomy splits.')
    LM['rh-mv'] = oc; LM['rh-comm'] = orif(np.pi / 2)
    rng = np.random.default_rng(11); calc = []
    for t0 in (np.pi / 2, -np.pi / 2):                                      # calcium clusters at the commissures, and in the leaflet body
        for _ in range(3):
            s = rng.uniform(0.35, 0.95); t = t0 + rng.normal(0, 0.18); a = A[int(np.argmin(np.abs(((T - t + np.pi) % (2 * np.pi)) - np.pi)))]
            calc.append(sphere(W(a + (orif(t) - a) * s - n * 3.2 * np.sin(np.pi * s) + n * 0.8), float(rng.uniform(1.4, 2.4))))
    for t in rng.uniform(-np.pi, np.pi, 4):
        a = A[int(np.argmin(np.abs(((T - t + np.pi) % (2 * np.pi)) - np.pi)))]; s = rng.uniform(0.3, 0.7)
        calc.append(sphere(W(a + (orif(t) - a) * s - n * 3.2 * np.sin(np.pi * s) + n * 0.8), float(rng.uniform(1.2, 2.0))))
    emit_mesh('rh-mv-calcium', 'Calcium in the rheumatic mitral valve', 'cardiac', '#f4f0e0', trimesh.util.concatenate(calc), visible=False,
              note='Commissural calcium is the strongest single predictor of a poor balloon result (Wilkins); heavy calcium pushes toward surgery.')
    ch = []
    for key in ('pm-al', 'pm-pm'):
        if key not in LM: continue
        tip = LM[key] + (oc - LM[key]) * 0.25                               # papillary tips drawn up: the chordae are short
        side = np.sign((LM[key] - c) @ v) or 1.0
        for k, t in enumerate(np.linspace(side * np.pi / 2 - 1.1, side * np.pi / 2 + 1.1, 5)):
            ch.append(tube([W(LM[key]), W(tip + (orif(t) - tip) * 0.4), W(orif(t))], 1.1 if k % 2 else 0.8, seg=8))
    if ch: emit_mesh('rh-chordae', 'Shortened, thickened, fused chordae', 'cardiac', '#e6dcc2', trimesh.util.concatenate(ch), visible=False,
                     note='Subvalvular disease: fused, short chordae tether the leaflet tips; severe subvalvular fusion makes a balloon or a repair less likely to succeed.')
    # the jets
    emit_mesh('jet-ms', 'Mitral stenosis: high-velocity inflow jet (diastole)', 'cardiac', '#ff7a1a',
              _cone(ctx, oc + n * 1.5, oc + (SC['_apex'] - oc) * 0.55, 2.2, 8.0), opacity=0.62, visible=False,
              note='Diastole: blood forced through the small orifice at high velocity; the gradient rises with flow and with heart rate (shorter diastole).')
    cp_ = c - n * 8.5 - u * R * 0.25                                         # the coaptation point
    emit_mesh('jet-mr', 'Mitral regurgitation jet (systole)', 'cardiac', '#2f7fff',
              _cone(ctx, cp_, cp_ + n * 34 - u * 12, 2.0, 11.0), opacity=0.62, visible=False,
              note='Systole: blood driven back into the LA. With anterior leaflet prolapse the jet points posteriorly, away from the prolapsing leaflet.')
    # acute rheumatic mitral regurgitation: anterior leaflet prolapse on elongated chordae
    rowsA = []
    sel = [(a, t) for a, t in zip(A, T) if abs(t) <= np.radians(62)]
    coapt = lambda t: c + v * (np.sin(t) * R * 0.82) - u * (0.28 * R) - n * 9.0
    for s in np.linspace(0, 1, 7):
        rowsA.append([a + (coapt(t) + n * 12.0 + u * 2 - a) * s + n * 5.5 * np.sin(np.pi * s) for a, t in sel])
    if len(sel) > 2:
        emit_mesh('mv-ant-prolapse', 'Anterior leaflet prolapsing into the LA (acute rheumatic MR)', 'cardiac', '#efe3cf', _sheet(ctx, rowsA), visible=False,
                  note='Acute rheumatic carditis: inflamed chordae elongate and the anterior leaflet prolapses (Carpentier type II); the annulus dilates (type I).')
        el = []
        for key in ('pm-al', 'pm-pm'):
            if key not in LM: continue
            tip = LM[key]
            side = np.sign((tip - c) @ v) or 1.0                                # each papillary muscle to its own half of the leaflet
            for t in np.linspace(0.1, np.radians(55), 4) * side:
                e = coapt(t) + n * 12.0 + u * 2
                el.append(tube([W(tip), W(e)], 0.35, seg=6))
        if el: emit_mesh('chordae-long', 'Elongated chordae (acute carditis)', 'cardiac', '#f3ecd9', trimesh.util.concatenate(el), visible=False)
        LM['mv-prolapse'] = coapt(0) + n * 12.0
    # vegetation on the atrial face of the anterior leaflet, near its closing edge
    a0 = A[int(np.argmin(np.abs(T)))]; vp = a0 + (coapt(0.2) - a0) * 0.72 - n * 3.2 * np.sin(np.pi * 0.72) + n * 3.5
    emit_mesh('mv-vegetation', 'Vegetation (infective endocarditis), about 12 mm', 'cardiac', '#8f5a2a', _blob(ctx, vp, 5.5, 5, lumps=9), visible=False,
              note='On the atrial side of the anterior leaflet, the low-pressure side of the regurgitant jet. 10 mm or more with an embolic event is an indication for urgent surgery (ESC 2023).')
    LM['mv-vegetation'] = vp
    # left atrial appendage thrombus
    laa = ctx.get('laa_mm')
    if laa is not None and len(laa) > 50:
        la_c = ctx.get('la_c', c + n * 25)
        d = np.linalg.norm(laa - la_c, axis=1); far = laa[d > np.percentile(d, 55)]
        tc = far.mean(0); r = float(np.clip(np.percentile(np.linalg.norm(far - tc, axis=1), 60) * 0.7, 4.0, 8.0))
        emit_mesh('laa-thrombus', 'Left atrial appendage thrombus', 'cardiac', '#5b1f1c', _blob(ctx, tc, r, 9, lumps=8, spread=0.6), visible=False,
                  note='Stasis in a large, fibrillating left atrium. A contraindication to balloon commissurotomy; removed, and the appendage closed, at surgery.')
        LM['laa-thrombus'] = tc
    # ---------------------------------------------------------------- aortic valve
    if 'av-centre' not in LM: return
    ac, an, e1 = LM['av-centre'], DIRS['av-axis'], DIRS['av-e1']; e2 = np.cross(an, e1); AR = SC['av-radius']; hs = SC['av-stj']; hc = hs * 0.85
    at = lambda t, r, z: ac + e1 * np.cos(t) * r + e2 * np.sin(t) * r + an * z
    comm = [LM[f'comm-{i}'] for i in range(3) if f'comm-{i}' in LM]
    if len(comm) == 3:
        ct_ = sorted([float(np.arctan2((q - ac) @ e2, (q - ac) @ e1)) for q in comm])
        cusps, edges = [], []
        for i in range(3):
            t0, t1 = ct_[i], ct_[(i + 1) % 3] + (2 * np.pi if i == 2 else 0)
            ts = np.linspace(t0, t1, 17); s_ = np.linspace(-1, 1, 17); z = hc * s_ ** 2; r_ = AR * (1 + 0.18 * s_ ** 2)
            rowsC, edge = [], []
            for k, (t, rr, zz) in enumerate(zip(ts, r_, z)):
                a = at(t, rr, zz)
                reach = 0.9 if (i == 0 and k >= 13) or (i == 1 and k <= 3) else 0.62               # retracted: the free edge stops short of the centre
                b = ac + an * (hc * 0.5) + (a - ac - an * ((a - ac) @ an)) * (1 - reach)
                rowsC.append([a + (b - a) * q - an * 3.0 * np.sin(np.pi * q) * (1 - abs(s_[k])) for q in np.linspace(0, 1, 7)]); edge.append(rowsC[-1][-1])
            cusps.append(_sheet(ctx, rowsC)); edges.append(tube([W(p) for p in edge], 1.3))
        emit_mesh('av-rheum', 'Rheumatic aortic valve: retracted, thickened cusps', 'cardiac', '#dcc9a6', trimesh.util.concatenate(cusps), visible=False,
                  note='Fibrosis retracts the cusps so that they no longer meet in the centre (a central regurgitant gap); one commissure is fused. Later, fusion and calcium add stenosis (mixed disease).')
        emit_mesh('av-rheum-edge', 'Rolled, thickened free edges', 'cardiac', '#c9b27f', trimesh.util.concatenate(edges), visible=False)
    emit_mesh('jet-ar', 'Aortic regurgitation jet (diastole)', 'cardiac', '#2f7fff',
              _cone(ctx, ac + an * (hc * 0.45), ac - an * 32 + (LM['mv-centre'] - ac) * 0.15, 2.4, 10.0), opacity=0.62, visible=False,
              note='Diastole: blood falls back from the aorta into the LV, which must eject it again: volume and pressure overload, a wide pulse pressure and lower diastolic coronary perfusion.')
    # vegetation on the ventricular side of the non-coronary / right cusp, and a paravalvular abscess under the non-coronary sinus
    t_n = float(np.arctan2((LM['aml-curtain'] - ac) @ e2, (LM['aml-curtain'] - ac) @ e1)) if 'aml-curtain' in LM else 0.0
    vv = at(t_n + 0.9, AR * 0.5, hc * 0.2) - an * 5.0
    emit_mesh('av-vegetation', 'Vegetation on the aortic valve', 'cardiac', '#8f5a2a', _blob(ctx, vv, 4.8, 7, lumps=8), visible=False,
              note='On the ventricular side of the cusp. Cusp perforation and acute severe regurgitation follow.')
    ab = [at(t, AR * 1.25, -2.0 + 2.5 * np.cos((t - t_n) * 2)) for t in np.linspace(t_n - 0.9, t_n + 0.9, 9)]
    emit_mesh('root-abscess', 'Paravalvular (aortic root) abscess', 'cardiac', '#7a4a32', tube([W(p) for p in ab], 3.6), opacity=0.85, visible=False,
              note='In the aortomitral curtain and under the non-coronary sinus: uncontrolled infection. New PR prolongation or heart block means it has reached the conduction tissue. An indication for urgent surgery; radical debridement, then patch or root replacement.')
    LM['av-vegetation'] = vv; LM['root-abscess'] = ab[4]
