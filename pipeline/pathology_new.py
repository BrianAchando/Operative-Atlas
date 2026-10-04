"""Meshes for the newer modules: constrictive pericarditis (thick, calcified pericardium and the pericardiectomy),
the lower limb (acute limb ischaemia, fem-pop bypass, amputation levels), the arm (arteriovenous fistulas) and
post-tuberculous bronchiectasis of the left upper lobe.

The heart, lungs, aorta and the right humerus come from the reference CT. The scan ends at the 4th lumbar vertebra and
at the right elbow, so the leg below the groin and the forearm are SCHEMATIC, drawn to typical adult dimensions, and are
labelled as such. Coordinates here are in the atlas frame (millimetres, carina at the origin) unless they come from a mask.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

ANT, SUP, RIGHT = np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
ART, VEIN, BONE, SKIN, GRAFTV = '#c0392b', '#4b5fa8', '#e9e2cf', '#d9b8a0', '#8c7fb8'
CLOT, INC, RING, STEEL = '#5a1a1a', '#d0433a', '#3fa7d6', '#9aa6b2'
V = lambda *a: np.array(a, float)


def lathe(P, radii, sections=28):
    """a tube of varying radius along a polyline, closed at both ends (atlas frame)"""
    P = np.asarray(P, float); n = len(P); T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    ref = RIGHT if abs(T[0][0]) < 0.9 else ANT; u = np.cross(T[0], ref); u /= np.linalg.norm(u); Vv, F = [], []
    for i in range(n):
        u = u - T[i] * np.dot(u, T[i]); u /= np.linalg.norm(u) + 1e-9; v = np.cross(T[i], u)
        for a in np.linspace(0, 2 * np.pi, sections, endpoint=False): Vv.append(P[i] + (u * np.cos(a) + v * np.sin(a)) * radii[i])
    for i in range(n - 1):
        for j in range(sections):
            a, b = i * sections + j, i * sections + (j + 1) % sections; F += [[a, a + sections, b + sections], [a, b + sections, b]]
    Vv += [P[0], P[-1]]; c0, c1 = len(Vv) - 2, len(Vv) - 1
    for j in range(sections):
        F.append([c0, (j + 1) % sections, j]); F.append([c1, (n - 1) * sections + j, (n - 1) * sections + (j + 1) % sections])
    return trimesh.Trimesh(np.array(Vv), np.array(F), process=True)


def spline(P, step=2.0):
    P = np.asarray(P, float); out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)]
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1]); return np.array(out)


def ring(p, d, r, rr=1.0):
    t = trimesh.creation.torus(major_radius=r, minor_radius=rr, major_sections=36, minor_sections=8)
    M = trimesh.geometry.align_vectors([0, 0, 1], np.asarray(d, float) / np.linalg.norm(d)); M[:3, 3] = p; t.apply_transform(M); return t


def disc(p, d, rx, ry, th=1.2, upx=RIGHT):
    """a thin elliptical disc (a cutting plane) at p, normal d"""
    c = trimesh.creation.cylinder(radius=1.0, height=th, sections=48); c.apply_scale([rx, ry, 1])
    M = trimesh.geometry.align_vectors([0, 0, 1], np.asarray(d, float) / np.linalg.norm(d)); M[:3, 3] = p; c.apply_transform(M); return c


def prism(poly2d, z0, z1, scale_bottom=0.75, origin=(0, 0)):
    """extrude a polygon (x, y) from z0 (top) down to z1, tapering toward origin"""
    from shapely.geometry import Polygon
    m = trimesh.creation.extrude_polygon(Polygon(poly2d), height=1.0)
    v = m.vertices.copy(); t = v[:, 2]                                    # 0 at z1 (bottom) .. 1 at z0 (top)
    s = scale_bottom + (1 - scale_bottom) * t
    ox, oy = origin
    v[:, 0] = ox + (v[:, 0] - ox) * s; v[:, 1] = oy + (v[:, 1] - oy) * s; v[:, 2] = z1 + t * (z0 - z1)
    m.vertices = v; return m


def build(ctx):
    ts, emit, emit_mesh, tube, AT = ctx['ts'], ctx['emit'], ctx['emit_mesh'], ctx['tube'], ctx['AT']
    C = np.asarray(ctx['CARINA'], float); LMW = ctx['LMW']; LM = {}
    sp = np.abs(np.diag(AT)[:3])
    tubeW = lambda pts, r, seg=14: tube([np.asarray(p, float) for p in pts], r, seg)
    print('== new modules: pericardium, leg, arm, bronchiectasis')

    # ============================================================ constrictive pericarditis
    H = ts('heart')
    if H.any():
        it_in = max(1, int(round(1.0 / sp.mean()))); it_out = max(2, int(round(6.0 / sp.mean())))
        shell = ndimage.binary_dilation(H, iterations=it_out) & ~ndimage.binary_dilation(H, iterations=it_in)
        emit('peri-thick', 'Thickened pericardium (constrictive, about 5–6 mm)', 'pathology', '#cbb79a', shell, AT, faces=14000, opacity=0.85, visible=False,
             note='Tuberculous constriction: a thick, fibrous, often calcified pericardium encases the heart and limits diastolic filling of all four chambers.')
        idx = np.argwhere(shell); pts = idx @ AT[:3, :3].T + AT[:3, 3]
        hp = np.argwhere(H) @ AT[:3, :3].T + AT[:3, 3]
        zlo, zhi = float(hp[:, 2].min()), float(hp[:, 2].max()); cy = float(np.median(hp[:, 1]))
        # calcified plaques: over the atrioventricular grooves and the diaphragmatic surface
        rng = np.random.default_rng(11)
        base = pts[(pts[:, 2] < zlo + 0.35 * (zhi - zlo))]
        pick = base[rng.choice(len(base), size=min(320, len(base)), replace=False)]
        hc = hp.mean(0); calc = []
        for q in pick:
            r = float(rng.uniform(1.2, 2.8)); e = trimesh.creation.icosphere(subdivisions=1, radius=1.0); e.apply_scale([r * 1.6, r, 0.45 * r])      # flat plaques
            n = (q - hc) / (np.linalg.norm(q - hc) + 1e-9)
            M = trimesh.geometry.align_vectors([0, 0, 1], n); M[:3, 3] = q - C; e.apply_transform(M); calc.append(e)
        emit_mesh('peri-calcium', 'Pericardial calcification', 'pathology', '#f2efe6', trimesh.util.concatenate(calc), visible=False,
                  note='Calcium is seen in about a quarter to a half of chronic constriction, typically over the atrioventricular grooves and the diaphragmatic surface: it marks the plane that is hardest to free.')
        # the pericardiectomy: everything in front of the phrenic nerves (lateral borders of the heart at each level) and the diaphragmatic surface removed
        keep = np.zeros(len(idx), bool)
        for k in np.unique(idx[:, 2]):
            sel = idx[:, 2] == k
            hz = hp[np.abs(hp[:, 2] - pts[sel][0, 2]) < sp[2]]
            if not len(hz): keep[sel] = True; continue
            x0, x1 = hz[:, 0].min() + 6, hz[:, 0].max() - 6
            p = pts[sel]
            lateral = (p[:, 0] < x0) | (p[:, 0] > x1)                      # the strips behind the phrenic nerves stay
            posterior = p[:, 1] < cy - 0.30 * np.ptp(hp[:, 1])
            keep[sel] = lateral | posterior
        keep &= pts[:, 2] > zlo + 9                                         # the diaphragmatic surface is freed too
        left = shell.copy(); left[tuple(idx[~keep].T)] = False
        emit('peri-left', 'Pericardium remaining after phrenic-to-phrenic pericardiectomy', 'pathology', '#cbb79a', left, AT, faces=9000, opacity=0.85, visible=False,
             note='Total (radical) pericardiectomy removes the pericardium from phrenic nerve to phrenic nerve, the diaphragmatic surface and over the cavae; a pedicle of pericardium is left with each phrenic nerve.')
        LM['heart-ant'] = hp[np.argmax(hp[:, 1])] - C; LM['heart-c'] = hp.mean(0) - C

    # ============================================================ the right leg (schematic below the groin)
    if 'cfa-r' not in LMW: print('   no femoral landmark: leg skipped'); LMW = {**LMW, 'cfa-r': V(46.8, 81.5, -482.1)}
    cfa_mid = np.asarray(LMW['cfa-r'], float); cfa_end = cfa_mid + V(0, 4, -20)
    X = cfa_end[0]
    K = V(X + 2, 35, -900)                                # the knee joint centre
    A_ = V(X + 2, 40, -1290)                              # the ankle joint
    LM['knee-r'] = K; LM['ankle-r'] = A_
    # bones
    femur = [V(X + 22, 45, -478), V(X + 52, 38, -515), V(X + 40, 36, -700), K + V(0, 0, 18)]
    emit_mesh('leg-femur', 'Femur, right (schematic)', 'leg', BONE, trimesh.util.concatenate([tubeW(femur[1:], 12.0), trimesh.creation.icosphere(2, 19.0).apply_translation(femur[0] + V(4, -10, 0))]), visible=False)
    tib_top = K + V(-4, 6, -20)
    tp = spline([K + V(-4, 6, -6), tib_top, tib_top + V(-2, 4, -150), A_ + V(-6, 6, 10)], 4.0)
    tr = np.interp(np.arange(len(tp)), [0, 6, 12, len(tp) - 1], [34, 18, 12, 14])
    emit_mesh('leg-tibia', 'Tibia, right (schematic)', 'leg', BONE, lathe(tp, tr), visible=False,
              note='Subcutaneous medial surface and anterior crest: the landmarks for fasciotomy and below-knee amputation.')
    fib = [K + V(28, -6, -30), K + V(30, -8, -200), A_ + V(26, -6, 4)]
    emit_mesh('leg-fibula', 'Fibula, right (schematic)', 'leg', BONE, tubeW(fib, 6.0), visible=False)
    foot = trimesh.creation.box(extents=[70, 210, 34]); foot.apply_translation(A_ + V(0, 70, -40))
    emit_mesh('leg-foot', 'Foot (schematic)', 'leg', SKIN, foot, opacity=0.25, visible=False)
    # skin of the limb
    axis = spline([cfa_end + V(10, -10, 20), V(X + 10, 40, -700), K + V(0, 5, 0), K + V(0, 10, -200), A_ + V(0, 5, -10)], 6.0)
    zz = axis[:, 2]
    rad = np.interp(zz, [-1300, -1100, -950, -900, -700, -480], [34, 46, 52, 50, 64, 78])
    emit_mesh('leg-skin', 'Skin of the right leg (schematic)', 'leg', SKIN, lathe(axis, rad, 40), opacity=0.25, visible=False)
    LM['leg-thigh'] = V(X + 10, 40, -700); LM['leg-calf'] = K + V(0, 10, -200)
    # arteries
    hiatus = V(X - 12, 40, -800)
    sfa = [cfa_end, cfa_end + V(-4, -6, -60), V(X - 8, 60, -650), hiatus]
    pop = [hiatus, K + V(-2, -12, 60), K + V(0, -16, 0), K + V(2, -14, -60)]
    tpt0 = K + V(2, -14, -60); ata0 = tpt0 + V(10, 4, -4)
    ata = [ata0, ata0 + V(14, 30, -10), ata0 + V(12, 34, -120), A_ + V(4, 40, 30), A_ + V(0, 52, -10)]
    dpa = [A_ + V(0, 52, -10), A_ + V(-6, 80, -24), A_ + V(-8, 130, -30)]
    tpt = [tpt0, tpt0 + V(-2, 0, -30)]
    pta = [tpt0 + V(-2, 0, -30), tpt0 + V(-4, 4, -150), A_ + V(-30, 0, 10), A_ + V(-30, 20, -24), A_ + V(-20, 60, -40)]
    per = [tpt0 + V(-2, 0, -30), tpt0 + V(12, -2, -150), A_ + V(18, -10, 30)]
    parts = {'leg-sfa': (sfa, 3.4, 'Superficial femoral artery, right (to the adductor hiatus)'), 'leg-pop': (pop, 3.2, 'Popliteal artery, right'),
             'leg-ata': (ata, 2.0, 'Anterior tibial artery, right'), 'leg-dpa': (dpa, 1.6, 'Dorsalis pedis artery'), 'leg-tpt': (tpt, 2.6, 'Tibioperoneal trunk'),
             'leg-pta': (pta, 1.9, 'Posterior tibial artery (behind the medial malleolus)'), 'leg-per': (per, 1.7, 'Peroneal artery')}
    for k, (P, r, nm) in parts.items():
        emit_mesh(k, nm + ' (schematic)', 'leg', ART, tubeW(P, r), visible=False)
    LM['adductor-hiatus'] = hiatus; LM['pop-ak'] = K + V(-2, -12, 60); LM['pop-bk'] = K + V(2, -14, -50); LM['trifurcation'] = tpt0
    LM['ankle-pt'] = A_ + V(-30, 0, 10); LM['ankle-dp'] = A_ + V(-6, 80, -24)
    # the great saphenous vein: groin to the front of the medial malleolus
    sfj = cfa_mid + V(-16, 10, -12)
    gsv = [sfj, sfj + V(-14, -4, -40), V(X - 38, 50, -650), K + V(-46, -8, 10), K + V(-40, 10, -150), A_ + V(-34, 34, 10), A_ + V(-24, 60, -20)]
    emit_mesh('leg-gsv', 'Great saphenous vein (schematic)', 'leg', VEIN, tubeW(gsv, 2.6), visible=False,
              note='From the saphenofemoral junction (about 3 cm below and lateral to the pubic tubercle) down the medial thigh, behind the medial femoral condyle, to in front of the medial malleolus. The best conduit for infrainguinal bypass.')
    LM['sfj'] = sfj
    # acute limb ischaemia: a saddle embolus at the femoral bifurcation, propagated clot down the popliteal
    ep = spline([cfa_mid + V(0, 1, 6), cfa_end + V(0, 0, 2), cfa_end + V(-2, -3, -18)], 1.0)
    emb = lathe(ep, np.full(len(ep), 3.6))
    pfa0 = cfa_end + V(6, -12, -12)
    emb2 = tubeW([cfa_end + V(1, -2, 0), pfa0], 2.9)
    clot = tubeW(spline([hiatus, *pop[1:], tpt[1]], 4.0), 2.8)
    emit_mesh('ali-embolus', 'Embolus at the femoral bifurcation (saddle)', 'leg', CLOT, trimesh.util.concatenate([emb, emb2]), visible=False,
              note='An embolus lodges where the artery divides and narrows: the common femoral bifurcation is the commonest site in the leg.')
    emit_mesh('ali-clot', 'Propagated thrombus (popliteal and trifurcation)', 'leg', CLOT, clot, opacity=0.9, visible=False)
    LM['ali-embolus'] = cfa_end
    art = cfa_mid + V(0, 4.2, -2)
    emit_mesh('ali-arteriotomy', 'Transverse arteriotomy, common femoral artery', 'incisions', INC, ring(art, ANT, 3.6, 0.7), visible=False)
    fog = spline([art + V(0, 30, 6), art + V(0, 8, 2), art, cfa_end, *sfa[1:], *pop[1:], tpt[1]], 3.0)
    emit_mesh('fogarty', 'Fogarty balloon catheter (4F) passed down to the trifurcation', 'leg', STEEL,
              trimesh.util.concatenate([tubeW(fog, 0.8, 8), trimesh.creation.icosphere(2, 3.4).apply_scale([1, 1, 1.8]).apply_translation(fog[-1])]), visible=False,
              note='Passed beyond the clot, the balloon inflated gently with saline and withdrawn steadily, pulling the clot out of the arteriotomy; repeated until two clean passes and good back-bleeding.')
    LM['fogarty-tip'] = fog[-1]
    # leg compartments (cross-section about the tibia and fibula), the fasciotomy incisions
    O = (K[0] + 6, K[1] + 4)
    cx, cy0 = K[0], K[1]
    poly = {
        'comp-ant': [(cx - 2, cy0 + 26), (cx + 14, cy0 + 12), (cx + 27, cy0 + 4), (cx + 40, cy0 + 20), (cx + 26, cy0 + 44), (cx + 6, cy0 + 44)],
        'comp-lat': [(cx + 27, cy0 + 4), (cx + 40, cy0 + 20), (cx + 50, cy0 + 4), (cx + 46, cy0 - 14), (cx + 32, cy0 - 10)],
        'comp-deep': [(cx - 10, cy0 + 18), (cx + 14, cy0 + 12), (cx + 27, cy0 + 4), (cx + 32, cy0 - 10), (cx + 18, cy0 - 20), (cx - 2, cy0 - 14), (cx - 14, cy0 + 2)],
        'comp-sup': [(cx - 14, cy0 + 2), (cx - 2, cy0 - 14), (cx + 18, cy0 - 20), (cx + 32, cy0 - 10), (cx + 46, cy0 - 14), (cx + 38, cy0 - 36), (cx + 12, cy0 - 50), (cx - 18, cy0 - 38), (cx - 30, cy0 - 14)]}
    names = {'comp-ant': ('Anterior compartment (tibialis anterior; deep peroneal nerve, anterior tibial artery)', '#d98c7a'),
             'comp-lat': ('Lateral compartment (peronei; superficial peroneal nerve)', '#c9a36b'),
             'comp-deep': ('Deep posterior compartment (tibialis posterior, long flexors; tibial nerve, posterior tibial and peroneal vessels)', '#b07fa8'),
             'comp-sup': ('Superficial posterior compartment (gastrocnemius, soleus)', '#a3876b')}
    for k, pl in poly.items():
        m = prism(pl, K[2] - 40, A_[2] + 70, 0.62, O)
        emit_mesh(k, names[k][0], 'leg', names[k][1], m, opacity=0.45, visible=False)
    LM['comp-ant'] = V(cx + 18, cy0 + 26, -1000); LM['comp-sup'] = V(cx + 10, cy0 - 30, -1000)
    # a slab cut across mid-calf: the four compartments in cross-section, for the fasciotomy step
    zs = -1085.0
    for k, pl in poly.items():
        m = prism(pl, zs + 6, zs - 6, 1.0, O)
        emit_mesh('xs-' + k[5:], names[k][0].split(' (')[0] + ' (cross-section)', 'leg', names[k][1], m, visible=False)
    sec = [trimesh.creation.cylinder(radius=11, height=13, sections=24).apply_translation(V(K[0] - 6, K[1] + 10, zs)),
           trimesh.creation.cylinder(radius=6, height=13, sections=20).apply_translation(V(K[0] + 26, K[1] - 6, zs))]
    emit_mesh('xs-bones', 'Tibia and fibula (cross-section)', 'leg', BONE, trimesh.util.concatenate(sec), visible=False)
    for k_, q_ in (('xs-ant', (cx + 14, cy0 + 28)), ('xs-lat', (cx + 40, cy0 + 2)), ('xs-deep', (cx + 10, cy0 - 4)), ('xs-sup', (cx + 6, cy0 - 32))):
        LM[k_] = V(q_[0], q_[1], zs + 7)
    LM['xs-centre'] = V(cx + 10, cy0, zs)
    al = [V(cx + 38, cy0 + 40, -960), V(cx + 38, cy0 + 40, -1085), V(cx + 32, cy0 + 32, -1200)]
    pm = [V(cx - 32, cy0 + 4, -960), V(cx - 32, cy0 + 4, -1085), V(cx - 22, cy0 + 4, -1200)]
    emit_mesh('fasc-al', 'Anterolateral fasciotomy incision (anterior and lateral compartments)', 'incisions', INC, tubeW(al, 1.8), visible=False,
              note='Midway between the tibial crest and the fibula, over the septum between the anterior and lateral compartments; protect the superficial peroneal nerve in the lower third.')
    emit_mesh('fasc-pm', 'Posteromedial fasciotomy incision (superficial and deep posterior compartments)', 'incisions', INC, tubeW(pm, 1.8), visible=False,
              note='1–2 cm behind the posteromedial border of the tibia; protect the great saphenous vein and saphenous nerve; detach soleus from the tibia to reach the deep compartment.')
    LM['fasc-al'] = al[1]; LM['fasc-pm'] = pm[1]
    # fem-pop: an occluded superficial femoral artery, a reversed vein graft to the below-knee popliteal
    occ = tubeW(spline([cfa_end + V(-3, -6, -70), V(X - 8, 60, -650), hiatus], 4.0), 3.7)
    emit_mesh('sfa-occlusion', 'Superficial femoral artery occlusion (atherosclerotic, about 20 cm)', 'leg', '#e8dcb0', occ, visible=False,
              note='The commonest site of lower limb occlusive disease, at the adductor canal; collaterals from the profunda reconstitute the popliteal.')
    gp = [cfa_mid + V(-3, 4.5, 0), cfa_mid + V(-8, 18, -40), V(X - 22, 66, -620), K + V(-24, -6, 60), K + V(-14, -18, -20), LMW_get(LM, 'pop-bk') + V(-2, -3, 0)]
    emit_mesh('graft-fempop', 'Reversed great saphenous vein graft: common femoral to below-knee popliteal', 'leg', GRAFTV, tubeW(gp, 2.8, 16), visible=False,
              note='Reversed so that its valves do not obstruct flow; tunnelled deep to sartorius and behind the knee; end-to-side at both ends.')
    emit_mesh('anast-fempop', 'Suture lines: femoral and below-knee popliteal anastomoses', 'leg', RING,
              trimesh.util.concatenate([ring(gp[0], ANT + SUP * 0.4, 3.6), ring(gp[-1], -ANT + RIGHT * 0.3, 3.4)]), visible=False)
    emit_mesh('inc-gsv', 'Incisions over the saphenous vein (harvest) and the below-knee popliteal', 'incisions', INC,
              trimesh.util.concatenate([tubeW([gsv[1] + V(-2, 6, 0), gsv[2] + V(-6, 6, 0)], 1.6), tubeW([K + V(-50, 2, -40), K + V(-46, 6, -110)], 1.6)]), visible=False)
    LM['graft-fempop'] = gp[2]
    # amputation levels
    lev = {'amp-aka': (V(X + 8, 42, -720), 'Above-knee amputation level (mid-thigh; equal anterior and posterior flaps)', 62, 58),
           'amp-tka': (K + V(0, 8, 8), 'Through-knee amputation level', 52, 50),
           'amp-bka': (K + V(0, 10, -130), 'Below-knee amputation level (about 12–15 cm below the knee joint; long posterior flap)', 50, 50),
           'amp-tma': (A_ + V(0, 110, -38), 'Transmetatarsal amputation level', 36, 20),
           'amp-ray': (A_ + V(-18, 190, -42), 'Toe or ray amputation', 10, 10)}
    for k, (p, nm, rx, ry) in lev.items():
        d = ANT if k in ('amp-tma', 'amp-ray') else SUP
        emit_mesh(k, nm, 'leg', INC, disc(p, d, rx, ry, 1.6), opacity=0.55, visible=False)
        LM[k] = p
    flap = [K + V(-40, 30, -130), K + V(-46, -10, -140), K + V(-30, -40, -230), K + V(0, -48, -265), K + V(30, -40, -230), K + V(46, -10, -140), K + V(40, 30, -130)]
    emit_mesh('amp-bka-flap', 'Long posterior myocutaneous flap (Burgess)', 'incisions', INC, tubeW(spline(flap, 4.0), 1.6), visible=False,
              note='The anterior incision at the level of bone section; a posterior flap about as long as the leg is wide at that level, folded forward over the bone end.')

    # ============================================================ the right arm: the humerus from the CT, the forearm schematic
    hm = ts('humerus_right')
    if hm.any():
        p = np.argwhere(hm) @ AT[:3, :3].T + AT[:3, 3] - C
        c = p.mean(0); _, _, vt = np.linalg.svd(p - c, full_matrices=False); ax = vt[0]
        if ax[2] > 0: ax = -ax
        pr = (p - c) @ ax; top, elb = c + ax * pr.min(), c + ax * pr.max()
    else:
        top, elb, ax = V(176, 17, 56), V(215, -48, -282), V(0.11, -0.19, -0.98)
    ax = ax / np.linalg.norm(ax)
    lat = np.cross(ax, ANT); lat = lat if lat[0] > 0 else -lat; lat /= np.linalg.norm(lat)          # lateral (radial side) for the right arm
    ant = np.cross(lat, ax); ant = ant if ant[1] > 0 else -ant; ant /= np.linalg.norm(ant)
    H_ = lambda t: top + (elb - top) * t
    E = elb + ant * 12; Wr = E + ax * 250                                     # elbow crease, wrist
    LM['elbow-r'] = E; LM['wrist-r'] = Wr
    off = lambda q, l, a: q + lat * l + ant * a
    if not hm.any(): emit_mesh('arm-humerus', 'Humerus, right', 'arm', BONE, tubeW([top, elb], 10.0), visible=False)
    emit_mesh('arm-radius', 'Radius (schematic)', 'arm', BONE, tubeW([off(E, 12, -6), off(E + ax * 120, 12, -4), off(Wr, 18, 0)], 4.5), visible=False)
    emit_mesh('arm-ulna', 'Ulna (schematic)', 'arm', BONE, tubeW([off(E, -14, -16), off(E + ax * 120, -16, -10), off(Wr, -16, -4)], 4.5), visible=False)
    sk = spline([top + ax * 30, H_(0.5), E, E + ax * 125, Wr, Wr + ax * 70], 6.0)
    d_ = (sk - top) @ ax
    emit_mesh('arm-skin', 'Skin of the right arm and forearm (forearm schematic)', 'arm', SKIN, lathe(sk, np.interp(d_, [0, 170, 350, 470, 600, 680], [48, 42, 38, 32, 24, 26]), 36), opacity=0.25, visible=False)
    axa_end = V(154.6, 4, -3.8)
    brach = [axa_end, off(H_(0.3), -22, 16), off(H_(0.7), -18, 20), off(E, -8, 14), off(E + ax * 22, -4, 10)]
    bif = brach[-1]
    rad = [bif, off(E + ax * 60, 10, 12), off(Wr - ax * 40, 18, 12), off(Wr, 19, 12)]
    uln = [bif, off(E + ax * 60, -14, 6), off(Wr, -18, 10)]
    emit_mesh('arm-brachial', 'Brachial artery (medial to the humerus; medial to the biceps tendon at the elbow)', 'arm', ART, tubeW(brach, 2.6), visible=False)
    emit_mesh('arm-radial', 'Radial artery (schematic)', 'arm', ART, tubeW(rad, 1.6), visible=False)
    emit_mesh('arm-ulnar', 'Ulnar artery (schematic)', 'arm', ART, tubeW(uln, 1.6), visible=False)
    ceph = [off(Wr, 22, 16), off(Wr - ax * 60, 24, 18), off(E + ax * 40, 26, 22), off(E, 26, 26), off(H_(0.55), 24, 30), off(H_(0.2), 18, 28), V(118, 30, 28)]
    bas = [off(Wr, -22, 12), off(E + ax * 60, -26, 12), off(E, -30, 14), off(H_(0.65), -30, 8), off(H_(0.45), -26, -4), off(H_(0.15), -26, -2)]
    mcv = [off(E + ax * 22, 25, 24), off(E, 0, 24), off(E - ax * 18, -29, 15)]
    emit_mesh('arm-cephalic', 'Cephalic vein (radial side; lateral arm; deltopectoral groove)', 'arm', VEIN, tubeW(ceph, 1.8), visible=False)
    emit_mesh('arm-basilic', 'Basilic vein (ulnar side; deep to the fascia in the upper arm)', 'arm', VEIN, tubeW(bas, 2.0), visible=False)
    emit_mesh('arm-mcv', 'Median cubital vein', 'arm', VEIN, tubeW(mcv, 1.6), visible=False)
    for k_, q in (('wrist-radial', off(Wr - ax * 15, 19, 12)), ('elbow-brachial', off(E, -8, 14)), ('cephalic-wrist', off(Wr - ax * 15, 23, 17)), ('arm-basilic-mid', off(H_(0.65), -30, 8))):
        LM[k_] = q
    # fistulas: radiocephalic (wrist), brachiocephalic (elbow), transposed brachiobasilic
    a_rc = off(Wr - ax * 25, 19, 13); a_bc = off(E, -7, 15)
    rc = [off(Wr - ax * 2, 22, 17), off(Wr - ax * 18, 21, 16), a_rc + lat * 1.6]
    emit_mesh('avf-rc', 'Radiocephalic fistula (Brescia–Cimino): cephalic vein end-to-side to the radial artery', 'arm', VEIN, tubeW(rc, 1.9), visible=False)
    emit_mesh('avf-rc-mature', 'Matured forearm cephalic vein (about 6 mm, superficial)', 'arm', VEIN, tubeW([a_rc + lat * 1.6, off(Wr - ax * 60, 24, 18), off(E + ax * 40, 26, 22), off(E, 26, 26)], 3.2), visible=False)
    bc = [off(E + ax * 10, 24, 25), off(E + ax * 2, 10, 22), a_bc + ant * 1.5]
    emit_mesh('avf-bc', 'Brachiocephalic fistula: cephalic vein (or median cubital) to the brachial artery at the elbow', 'arm', VEIN, tubeW(bc, 2.2), visible=False)
    emit_mesh('avf-bc-mature', 'Matured cephalic vein in the upper arm', 'arm', VEIN, tubeW([a_bc + ant * 1.5, off(E - ax * 20, 20, 28), off(H_(0.55), 24, 30), off(H_(0.2), 18, 28)], 3.4), visible=False)
    bb = [off(E + ax * 4, -8, 17), off(E - ax * 40, -14, 30), off(H_(0.6), -16, 34), off(H_(0.3), -18, 30), off(H_(0.15), -24, 2)]
    emit_mesh('avf-bb', 'Transposed brachiobasilic fistula: the basilic vein brought to the surface', 'arm', VEIN, tubeW(bb, 3.0), visible=False,
              note='The basilic vein is deep and medial, beside the median and medial cutaneous nerves: mobilised along the arm and tunnelled superficially and laterally so it can be needled.')
    emit_mesh('anast-avf', 'Anastomoses: radial (wrist) and brachial (elbow)', 'arm', RING, trimesh.util.concatenate([ring(a_rc + lat * 1.6, lat, 2.6, 0.6), ring(a_bc + ant * 1.5, ant, 3.0, 0.6)]), visible=False)
    emit_mesh('inc-avf', 'Incisions: longitudinal at the wrist; transverse in the antecubital fossa; medial arm (basilic transposition)', 'incisions', INC,
              trimesh.util.concatenate([tubeW([off(Wr - ax * 5, 22, 22), off(Wr - ax * 45, 22, 22)], 1.4), tubeW([off(E, -24, 30), off(E, 22, 32)], 1.4),
                                        tubeW([off(E - ax * 10, -26, 30), off(H_(0.3), -30, 26)], 1.4)]), visible=False)
    LM['avf-rc'] = a_rc; LM['avf-bc'] = a_bc

    # ============================================================ post-tuberculous bronchiectasis, left upper lobe
    L = ts('lung_upper_lobe_left')
    if L.any() and 'hilum-l' in LMW:
        er = ndimage.binary_erosion(L, iterations=max(2, int(8 / sp.mean())))
        q = np.argwhere(er) @ AT[:3, :3].T + AT[:3, 3] - C
        o = np.asarray(LMW['hilum-l'], float) + V(-6, 0, 6)
        rng = np.random.default_rng(7); tips = []; br = []
        def toward(target):
            return q[np.argmin(np.linalg.norm(q - target, axis=1))]
        for target in (V(-60, 30, 40), V(-90, 10, 20), V(-70, -20, 30), V(-100, 40, -10), V(-55, 5, 55)):
            t = toward(target); tips.append(t)
            mid = o + (t - o) * 0.45 + rng.normal(0, 4, 3)
            P0 = spline([o, o + (mid - o) * 0.5, mid], 2.0)
            br.append(lathe(P0, np.linspace(3.2, 3.8, len(P0)), 18))
            for j in range(3):                                                     # each segmental bronchus divides; the subsegmental ones are dilated
                end = toward(t + rng.normal(0, 14, 3))
                P1 = spline([mid, mid + (end - mid) * 0.5 + rng.normal(0, 3, 3), end], 1.5)
                s_ = np.linspace(0, 1, len(P1))
                kind = (j + len(tips)) % 3
                if kind == 0: r = 2.2 + 2.3 * s_                                                  # cylindrical: wider toward the periphery
                elif kind == 1: r = 2.2 + 1.6 * s_ + 1.4 * np.sin(s_ * np.pi * 5) ** 2           # varicose: beaded
                else: r = 2.2 + 1.0 * s_
                br.append(lathe(P1, r, 16))
                if kind == 2: br.append(trimesh.creation.icosphere(2, float(rng.uniform(4.5, 7.0))).apply_translation(end))   # cystic (saccular) end
        emit_mesh('bx-lul', 'Bronchiectasis: dilated, varicose and cystic bronchi (left upper lobe)', 'pathology', '#e7d9a8', trimesh.util.concatenate(br), opacity=0.8, visible=False,
                  note='Traction and post-infective bronchiectasis after tuberculosis: thick-walled, dilated bronchi pooling secretions, with recurrent infection and haemoptysis.')
        LM['bx-lul'] = tips[0]
        # hypertrophied bronchial artery from the descending aorta to the left hilum
        a0 = np.asarray(LMW.get('taa-prox', o + V(18, -20, 12)), float) + V(-6, 12, -10)
        ba = spline([a0, a0 + V(-6, 10, 6), o + V(4, -14, -4), o + V(-4, -6, 2), o + V(-14, 2, 8), tips[1] * 0.4 + o * 0.6], 2.0)
        ba = ba + np.c_[np.sin(np.arange(len(ba)) / 3.0) * 2.0, np.cos(np.arange(len(ba)) / 3.0) * 2.0, np.zeros(len(ba))]
        emit_mesh('bx-bronchial-a', 'Hypertrophied, tortuous bronchial artery (the source of haemoptysis)', 'pathology', ART, tubeW(ba, 2.2), visible=False,
                  note='Chronic inflammation enlarges the bronchial arteries (systemic pressure): they bleed into the bronchi. Bronchial artery embolisation is the first treatment for massive haemoptysis; surgery removes the source.')
        # adhesions: bands from the lobe surface to the chest wall
        surf = np.argwhere(L & ~ndimage.binary_erosion(L, iterations=2)) @ AT[:3, :3].T + AT[:3, 3] - C
        lat_s = surf[(surf[:, 0] < np.percentile(surf[:, 0], 12)) | (surf[:, 2] > np.percentile(surf[:, 2], 90))]
        sel = lat_s[rng.choice(len(lat_s), size=min(26, len(lat_s)), replace=False)]
        ad = [tubeW([s - V(-2, 0, 1), s + V(-14, 0, 4) + rng.normal(0, 3, 3)], float(rng.uniform(1.4, 2.6)), 8) for s in sel]
        emit_mesh('bx-adhesions', 'Dense pleural adhesions (pleural symphysis)', 'pathology', '#e6dcc8', trimesh.util.concatenate(ad), visible=False,
                  note='After tuberculosis the lung is often fused to the chest wall: an extrapleural plane may be safer than fighting through the adhesions.')
        LM['bx-adhesions'] = sel[0]
    return {k: np.asarray(v, float) + C for k, v in LM.items()}      # returned in scanner millimetres, like the other builders


def LMW_get(LM, k):
    return np.asarray(LM[k], float)
