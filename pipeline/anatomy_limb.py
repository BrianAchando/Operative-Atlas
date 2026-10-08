"""Second-generation shapes for the schematic anatomy below the scan: the pelvic and right-leg arterial tree, the iliac
and leg veins, the right arm vessels, and the bones of the leg, knee and foot and of the forearm.

The centrelines are the same as pathology_vasc.py and pathology_new.py drew (so the grafts, stents, clots and catheters
built on them still sit inside the vessels); what changes is the surface: every vessel tapers, branches flare into their
parent with a smooth fillet, ends are rounded, and the bones have heads, condyles, malleoli and a foot. Each tree is meshed
as one signed-distance surface and then cut back into the named structures (faces go to the nearest vessel), so the
joins are seamless and each part can still be highlighted on its own. Runs after those two modules and replaces their
meshes by id. Atlas frame (mm, carina at the origin) throughout.
"""
from __future__ import annotations

import numpy as np
import trimesh

from sdfmesh import Vessel, Ellipsoid, mesh, frame

V = lambda *a: np.array(a, float)
U = lambda v: np.asarray(v, float) / (np.linalg.norm(v) + 1e-9)
RIGHT, ANT, SUP = V(1, 0, 0), V(0, 1, 0), V(0, 0, 1)
ART, VEIN, BONE = '#c0392b', '#4b5fa8', '#e9e2cf'


def split(m, owners):
    """cut a tree mesh into parts: each face goes to the owner whose primitives lie nearest its centre"""
    c = m.triangles_center
    D = np.stack([np.min([p.eval(c) for p in prims], axis=0) for prims in owners.values()], 1)
    lab = np.argmin(D, 1); out = {}
    for i, k in enumerate(owners):
        f = np.where(lab == i)[0]
        if len(f): s = m.submesh([f], append=True); s.fix_normals(); out[k] = s
    return out


def build(ctx):
    emit_mesh = ctx['emit_mesh']; L = {k: np.asarray(v, float) for k, v in ctx['LMW'].items()}
    out = {}
    if not all(k in L for k in ('bifurcation', 'cia-l', 'cia-r', 'cfa-l', 'cfa-r')):
        print('  anatomy_limb: iliac landmarks missing, skipped'); return out
    print('== second-generation limb anatomy (vessel trees, bones)')
    bif = L['bifurcation']
    names = {}

    # ---------------------------------------------------------------- the pelvic arteries and both femoral bifurcations
    tree = {}
    for sd, s in (('l', -1), ('r', 1)):
        lat = RIGHT * s
        cia0 = bif + lat * 4; cia_end = 2 * L[f'cia-{sd}'] - bif
        eia_end = L[f'cfa-{sd}'] + SUP * 20 - ANT * 4; cfa_end = eia_end - SUP * 40 + ANT * 8
        iia = [cia_end, cia_end + lat * 8 - SUP * 20 - ANT * 22, cia_end + lat * 14 - SUP * 45 - ANT * 30]
        sfa = [cfa_end, cfa_end - SUP * 60 - lat * 6 + ANT * 2, cfa_end - SUP * 120 - lat * 12]
        pfa = [cfa_end, cfa_end - SUP * 20 + lat * 6 - ANT * 10, cfa_end - SUP * 70 + lat * 10 - ANT * 18]
        nm = 'left' if sd == 'l' else 'right'
        tree[f'cia-{sd}'] = ([Vessel([bif + SUP * 4, cia0, (cia0 + cia_end) / 2 + lat * 0.5, cia_end], [6.2, 5.4, 5.0, 4.6])], f'Common iliac artery, {nm}')
        tree[f'eia-{sd}'] = ([Vessel([cia_end, (cia_end + eia_end) / 2 + ANT * 10, eia_end], [4.1, 3.9, 3.8])], f'External iliac artery, {nm}')
        # internal iliac with its anterior and posterior divisions (superior gluteal behind, the visceral trunk in front)
        div = iia[1]
        tree[f'iia-{sd}'] = ([Vessel(iia, [3.3, 2.9, 2.4]),
                              Vessel([div, div + lat * 6 - SUP * 10 - ANT * 22, div + lat * 12 - SUP * 24 - ANT * 38], [2.2, 1.9, 1.6], seed=1),
                              Vessel([iia[2], iia[2] - SUP * 20 + ANT * 6 - lat * 4, iia[2] - SUP * 44 + ANT * 14 - lat * 10], [1.9, 1.6, 1.3], seed=2)],
                             f'Internal iliac artery, {nm} (anterior and posterior divisions)')
        tree[f'cfa-{sd}'] = ([Vessel([eia_end, cfa_end], [3.9, 3.8])], f'Common femoral artery, {nm}')
        tree[f'pfa-{sd}'] = ([Vessel(pfa, [3.0, 2.7, 2.2]),
                              Vessel([pfa[0] - SUP * 6 + lat * 3, pfa[0] - SUP * 12 + lat * 22 + ANT * 4, pfa[0] - SUP * 26 + lat * 40 + ANT * 2], [1.8, 1.6, 1.3], seed=3),   # lateral circumflex femoral
                              Vessel([pfa[0] - SUP * 8 - lat * 2 - ANT * 4, pfa[0] - SUP * 10 - lat * 20 - ANT * 10, pfa[0] - SUP * 16 - lat * 30 - ANT * 22], [1.6, 1.4, 1.2], seed=4)],  # medial circumflex femoral
                             f'Profunda femoris artery, {nm}, with the lateral and medial circumflex femoral arteries')
        tree[f'sfa-{sd}'] = ([Vessel(sfa, [3.3, 3.2, 3.1])], f'Superficial femoral artery, {nm}')
        out[f'cfa-end-{sd}'] = cfa_end
    m = mesh([p for prims, _ in tree.values() for p in prims], voxel=0.4, blend=1.6, smooth=6, density=0.9)
    parts = split(m, {k: v[0] for k, v in tree.items()})
    for k, s in parts.items():
        emit_mesh(k, tree[k][1] + ' (schematic, below the scan)', 'vascular', ART, s, visible=False,
                  note='Below the lower edge of this CT: drawn to typical adult calibre, tapering and branching.')
        names[k] = len(s.faces)

    # ---------------------------------------------------------------- the right leg arteries: one tree from the femoral bifurcation to the foot
    if 'knee-r' in L and 'ankle-r' in L:
        cfa_mid = L['cfa-r']; cfa_end = cfa_mid + V(0, 4, -20); X = cfa_end[0]
        K = V(X + 2, 35, -900); A_ = V(X + 2, 40, -1290); hiatus = V(X - 12, 40, -800)
        sfa = [cfa_end, cfa_end + V(-4, -6, -60), V(X - 8, 60, -650), hiatus]
        pop = [hiatus, K + V(-2, -12, 60), K + V(0, -16, 0), K + V(2, -14, -60)]
        tpt0 = K + V(2, -14, -60); ata0 = tpt0 + V(10, 4, -4)
        ata = [ata0, ata0 + V(14, 30, -10), ata0 + V(12, 34, -120), A_ + V(4, 40, 30), A_ + V(0, 52, -10)]
        dpa = [A_ + V(0, 52, -10), A_ + V(-6, 80, -24), A_ + V(-8, 130, -30)]
        tpt = [tpt0, tpt0 + V(-2, 0, -30)]
        pta = [tpt0 + V(-2, 0, -30), tpt0 + V(-4, 4, -150), A_ + V(-30, 0, 10), A_ + V(-30, 20, -24), A_ + V(-20, 60, -40)]
        per = [tpt0 + V(-2, 0, -30), tpt0 + V(12, -2, -150), A_ + V(18, -10, 30)]
        gen = K + V(-2, -14, 30)
        leg = {
            'leg-sfa': ([Vessel(sfa, [3.4, 3.2, 3.0, 2.9])], 'Superficial femoral artery, right (to the adductor hiatus)'),
            'leg-pop': ([Vessel(pop, [2.9, 2.8, 2.7, 2.6]),
                         Vessel([gen, gen + V(-16, 6, 4), gen + V(-30, 18, -6)], [1.0, 0.9, 0.8]), Vessel([gen, gen + V(16, 6, 4), gen + V(30, 18, -6)], [1.0, 0.9, 0.8]),
                         Vessel([K + V(0, -15, -20), K + V(-10, -26, -40), K + V(-14, -34, -90)], [1.1, 1.0, 0.8], wiggle=1.5, seed=5),
                         Vessel([K + V(1, -15, -22), K + V(10, -26, -42), K + V(14, -34, -92)], [1.1, 1.0, 0.8], wiggle=1.5, seed=6)],
                        'Popliteal artery, right, with the genicular and sural branches'),
            'leg-tpt': ([Vessel(tpt, [2.5, 2.3])], 'Tibioperoneal trunk'),
            'leg-ata': ([Vessel(ata, [2.0, 1.9, 1.7, 1.6, 1.5], wiggle=1.2, seed=7)], 'Anterior tibial artery, right'),
            'leg-dpa': ([Vessel(dpa, [1.5, 1.3, 1.0]), Vessel([dpa[1], dpa[1] + V(16, 10, -4), dpa[1] + V(36, 26, -8)], [0.9, 0.8, 0.7])], 'Dorsalis pedis artery, with the arcuate artery'),
            'leg-pta': ([Vessel(pta, [1.9, 1.8, 1.6, 1.4, 1.2], wiggle=1.2, seed=8)], 'Posterior tibial artery (behind the medial malleolus)'),
            'leg-per': ([Vessel(per, [1.7, 1.5, 1.2], wiggle=1.0, seed=9)], 'Peroneal artery'),
        }
        # a short piece of the common femoral and its bifurcation so the superficial femoral flares out of it
        anchor = {'_cfa': [Vessel([cfa_mid + SUP * 12, cfa_end], [3.9, 3.8])]}
        m = mesh([p for prims, _ in leg.values() for p in prims] + anchor['_cfa'], voxel=0.35, blend=1.2, smooth=6, density=1.2)
        parts = split(m, {**{k: v[0] for k, v in leg.items()}, **anchor})
        for k, s in parts.items():
            if k.startswith('_'): continue
            emit_mesh(k, leg[k][1] + ' (schematic)', 'leg', ART, s, visible=False); names[k] = len(s.faces)
        out['knee-r'] = K; out['ankle-r'] = A_

        # ------------------------------------------------------------ bones of the right leg
        fz = lambda a, b: frame(U(b - a))
        hip = V(X + 22, 45, -478) + V(4, -10, 0)
        shaft = [V(X + 52, 38, -515), V(X + 46, 40, -600), V(X + 40, 38, -700), V(X + 30, 36, -820), K + V(0, 0, 30)]
        femur = [Ellipsoid(hip, [21, 21, 21]),                                                   # head
                 Vessel([hip, hip + V(18, 2, -24), shaft[0]], [12, 11.5, 14]),                    # neck
                 Ellipsoid(shaft[0] + V(14, -6, 14), [16, 14, 20]),                               # greater trochanter
                 Ellipsoid(shaft[0] + V(-6, -10, -26), [9, 9, 10]),                               # lesser trochanter
                 Vessel(shaft, [15, 13, 12, 13, 17]),                                             # shaft, flaring to the condyles
                 Ellipsoid(K + V(-18, -8, 22), [17, 25, 20]), Ellipsoid(K + V(18, -8, 22), [16, 24, 19]),   # medial and lateral condyles
                 Ellipsoid(K + V(0, 14, 30), [20, 10, 18])]                                       # trochlea
        fm = mesh(femur, voxel=0.8, blend=4.0, faces=14000, smooth=10)
        emit_mesh('leg-femur', 'Femur, right (schematic)', 'leg', BONE, fm, visible=False)
        pat = mesh([Ellipsoid(K + V(0, 36, 30), [20, 9, 24])], voxel=0.6, blend=0, faces=2500)
        emit_mesh('leg-patella', 'Patella (schematic)', 'leg', BONE, pat, visible=False)
        tib_top = K + V(-4, 6, -20)
        tib = [Ellipsoid(K + V(-16, 2, -12), [20, 24, 11]), Ellipsoid(K + V(16, 0, -12), [18, 22, 11]),     # medial and lateral plateaus
               Ellipsoid(K + V(-2, 22, -40), [9, 7, 14]),                                                   # tibial tuberosity
               Vessel([K + V(-4, 4, -18), tib_top + V(-2, 4, -60), tib_top + V(-2, 6, -150), tib_top + V(-4, 6, -280), A_ + V(-6, 6, 20)], [24, 15, 12, 12, 17]),
               Ellipsoid(A_ + V(-22, 4, 6), [7, 12, 16])]                                                   # medial malleolus
        tm = mesh(tib, voxel=0.8, blend=4.0, faces=14000, smooth=10)
        emit_mesh('leg-tibia', 'Tibia, right (schematic)', 'leg', BONE, tm, visible=False,
                  note='Subcutaneous medial surface and anterior crest: the landmarks for fasciotomy and below-knee amputation.')
        fib = [Ellipsoid(K + V(30, -6, -32), [9, 9, 10]), Vessel([K + V(30, -6, -32), K + V(31, -8, -200), A_ + V(26, -6, 14)], [6.5, 5.5, 6.5]),
               Ellipsoid(A_ + V(27, -6, -2), [8, 11, 18])]                                                  # lateral malleolus (lower than the medial)
        emit_mesh('leg-fibula', 'Fibula, right (schematic)', 'leg', BONE, mesh(fib, voxel=0.6, blend=3.0, faces=8000, smooth=8), visible=False)
        # the foot: talus, calcaneus, the midfoot and five metatarsals and toes (one bony block, schematic)
        ft = [Ellipsoid(A_ + V(0, 8, -12), [18, 22, 13]),                                        # talus
              Ellipsoid(A_ + V(2, -18, -36), [20, 34, 18]),                                      # calcaneus
              Ellipsoid(A_ + V(2, 44, -32), [26, 22, 13])]                                       # navicular, cuboid, cuneiforms
        for i, dx in enumerate((-22, -10, 1, 12, 23)):
            b0 = A_ + V(dx * 0.7, 60, -34 - i * 0.6); b1 = A_ + V(dx, 130 - abs(i - 1.2) * 6, -44)
            ft.append(Vessel([b0, b1], [5.0 if i == 0 else 3.6, 4.2 if i == 0 else 3.0]))
            ft.append(Vessel([b1, b1 + V(dx * 0.05, 30 - i * 2, -2)], [4.0 if i == 0 else 2.8, 3.4 if i == 0 else 2.2]))
        emit_mesh('leg-foot', 'Bones of the foot (schematic)', 'leg', BONE, mesh(ft, voxel=0.7, blend=3.0, faces=14000, smooth=8), opacity=0.9, visible=False)
        # the skin of the limb: thigh, knee, the calf bulging behind the shin, ankle and the foot (schematic, shaped)
        thigh = Vessel([V(X + 16, 50, -470), V(X + 12, 46, -560), V(X + 8, 40, -700), K + V(0, 10, 60), K + V(0, 6, 0)], [80, 72, 62, 52, 48])
        shin = Vessel([K + V(0, 6, 0), K + V(2, 10, -120), K + V(2, 14, -260), A_ + V(0, 8, 40), A_ + V(0, 4, 0)], [46, 40, 33, 28, 30])
        calf = Ellipsoid(K + V(2, -24, -150), [44, 38, 105])
        heel = Ellipsoid(A_ + V(0, -20, -36), [30, 36, 30]); mid = Ellipsoid(A_ + V(0, 50, -32), [38, 52, 24])
        fore = Ellipsoid(A_ + V(2, 120, -40), [46, 40, 15]); toes = Ellipsoid(A_ + V(0, 168, -42), [44, 22, 12])
        sk = mesh([thigh, shin, calf, heel, mid, fore, toes], voxel=2.0, blend=18.0, smooth=10, density=0.06)
        emit_mesh('leg-skin', 'Skin of the right leg (schematic)', 'leg', '#d9b8a0', sk, opacity=0.25, visible=False)

        # ------------------------------------------------------------ the leg veins: great saphenous and the deep veins
        sfj = cfa_mid + V(-16, 10, -12)
        gsv = [sfj, sfj + V(-14, -4, -40), V(X - 38, 50, -650), K + V(-46, -8, 10), K + V(-40, 10, -150), A_ + V(-34, 34, 10), A_ + V(-24, 60, -20)]
        emit_mesh('leg-gsv', 'Great saphenous vein (schematic)', 'leg', VEIN, mesh([Vessel(gsv, [3.4, 3.0, 2.8, 2.6, 2.3, 2.0, 1.8], wiggle=2.0, seed=11)], voxel=0.4, blend=0, faces=10000, smooth=6), visible=False,
                  note='From the saphenofemoral junction (about 3 cm below and lateral to the pubic tubercle) down the medial thigh, behind the medial femoral condyle, to in front of the medial malleolus. The best conduit for infrainguinal bypass.')

    # ---------------------------------------------------------------- the right arm: brachial, radial and ulnar as one tree
    if 'elbow-brachial' in L and 'wrist-radial' in L and 'elbow-r' in L and 'wrist-r' in L:
        E = L['elbow-r']; Wr = L['wrist-r']; ax = U(Wr - E)
        lat = np.cross(ax, ANT); lat = lat if lat[0] > 0 else -lat; lat = U(lat)
        ant = np.cross(lat, ax); ant = ant if ant[1] > 0 else -ant; ant = U(ant)
        off = lambda q, l, a: q + lat * l + ant * a
        elb = E - ant * 12                                                                       # pathology_new: E = elbow + 12 mm forward
        h65 = L['arm-basilic-mid'] + lat * 30 - ant * 8 if 'arm-basilic-mid' in L else elb - ax * 120   # = H_(0.65) on the humeral line
        H_ = lambda t: elb + (h65 - elb) * (1 - t) / 0.35
        axa_end = V(154.6, 4, -3.8)
        brach = [axa_end, off(H_(0.3), -22, 16), off(H_(0.7), -18, 20), off(E, -8, 14), off(E + ax * 22, -4, 10)]
        b = brach[-1]
        rad = [b, off(E + ax * 60, 10, 12), off(Wr - ax * 40, 18, 12), off(Wr, 19, 12)]
        uln = [b, off(E + ax * 60, -14, 6), off(Wr, -18, 10)]
        arm = {'arm-brachial': ([Vessel(brach, [2.9, 2.8, 2.7, 2.6, 2.5])], 'Brachial artery (medial to the humerus; medial to the biceps tendon at the elbow)'),
               'arm-radial': ([Vessel(rad, [1.9, 1.7, 1.5, 1.4])], 'Radial artery (schematic)'),
               'arm-ulnar': ([Vessel(uln, [2.0, 1.7, 1.5])], 'Ulnar artery (schematic)')}
        m = mesh([p for prims, _ in arm.values() for p in prims], voxel=0.35, blend=1.0, smooth=6, density=1.2)
        for k, s in split(m, {k: v[0] for k, v in arm.items()}).items():
            emit_mesh(k, arm[k][1], 'arm', ART, s, visible=False); names[k] = len(s.faces)
        ceph = [off(Wr, 22, 16), off(Wr - ax * 60, 24, 18), off(E + ax * 40, 26, 22), off(E, 26, 26), off(H_(0.55), 24, 30), off(H_(0.2), 18, 28), V(118, 30, 28)]
        bas = [off(Wr, -22, 12), off(E + ax * 60, -26, 12), off(E, -30, 14), off(H_(0.65), -30, 8), off(H_(0.45), -26, -4), off(H_(0.15), -26, -2)]
        mcv = [off(E + ax * 22, 25, 24), off(E, 0, 24), off(E - ax * 18, -29, 15)]
        veins = {'arm-cephalic': ([Vessel(ceph, [1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8], wiggle=1.2, seed=21)], 'Cephalic vein (radial side; lateral arm; deltopectoral groove)'),
                 'arm-basilic': ([Vessel(bas, [1.6, 1.9, 2.2, 2.6, 3.0, 3.3], wiggle=1.2, seed=22)], 'Basilic vein (ulnar side; deep to the fascia in the upper arm)'),
                 'arm-mcv': ([Vessel(mcv, [1.6, 1.6, 1.7])], 'Median cubital vein')}
        m = mesh([p for prims, _ in veins.values() for p in prims], voxel=0.35, blend=1.0, smooth=6, density=1.2)
        for k, s_ in split(m, {k: v[0] for k, v in veins.items()}).items():
            emit_mesh(k, veins[k][1], 'arm', VEIN, s_, visible=False); names[k] = len(s_.faces)
        # forearm bones and the skin of the arm (schematic): radius and ulna with their heads and styloids; arm, forearm, hand
        rd = [Ellipsoid(off(E, 12, -6), [7, 7, 6]), Vessel([off(E, 12, -6), off(E + ax * 120, 12, -4), off(Wr, 18, 0)], [4.0, 4.6, 8.0]), Ellipsoid(off(Wr + ax * 4, 21, 0), [5, 6, 7])]
        ul = [Ellipsoid(off(E - ax * 8, -14, -18), [9, 10, 12]), Vessel([off(E, -14, -16), off(E + ax * 120, -16, -10), off(Wr, -16, -4)], [7.0, 4.6, 3.6]), Ellipsoid(off(Wr + ax * 4, -17, -4), [4, 4, 6])]
        emit_mesh('arm-radius', 'Radius (schematic)', 'arm', BONE, mesh(rd, voxel=0.5, blend=2.0, smooth=8, density=0.5), visible=False)
        emit_mesh('arm-ulna', 'Ulna (schematic)', 'arm', BONE, mesh(ul, voxel=0.5, blend=2.0, smooth=8, density=0.5), visible=False)
        top = H_(0.0)
        upper = Vessel([top + ax * 30, H_(0.5), E], [48, 42, 37]); fore = Vessel([E, E + ax * 125, Wr], [37, 31, 24])
        flex = Ellipsoid(off(E + ax * 70, -6, 4), [34, 30, 80]); hand = Ellipsoid(Wr + ax * 60 + ant * 4, [42, 16, 52], axes=frame(ax, lat))
        emit_mesh('arm-skin', 'Skin of the right arm and forearm (forearm schematic)', 'arm', '#d9b8a0',
                  mesh([upper, fore, flex, hand], voxel=1.6, blend=12.0, smooth=10, density=0.08), opacity=0.25, visible=False)
    print('  limb anatomy:', names)
    return {k: v for k, v in out.items()}
