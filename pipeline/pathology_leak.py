"""Anastomotic leak after esophagectomy: the defect, the mediastinal and pleural collections, the neck leak, and the
tools that treat them (nasogastric tube, chest drain, endoscopic vacuum sponge, covered stent, cervical esophagostomy),
and necrosis of the conduit tip.

All schematic, placed on the reference CT: the esophageal centerline, the intrathoracic and cervical anastomoses
(from mediastinum.py), the base of the right pleural space and the right lateral chest wall.
Works in carina-centred coordinates (W); returns landmarks in scanner coordinates.
"""
from __future__ import annotations

import numpy as np
import trimesh

U = lambda v: np.asarray(v, float) / (np.linalg.norm(v) + 1e-9)
RIGHT, ANT, SUP = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])


def ellipsoid(c, axes, radii, sub=3):
    s = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
    M = np.column_stack([U(a) * r for a, r in zip(axes, radii)])
    s.vertices = s.vertices @ M.T + np.asarray(c, float)
    s.fix_normals(); return s


def build(ctx):
    emit_mesh, W, tube, CARINA = ctx['emit_mesh'], ctx['W'], ctx['tube'], ctx['CARINA']
    L = ctx['LMW']                       # landmarks already in W
    if not all(k in L for k in ('anast-chest', 'anast-neck', 'eso-neck')):
        print('  leak: anastomosis landmarks missing, skipped'); return {}
    eso = W(ctx['eso_mm'])               # esophagus voxels in W
    zs = np.arange(eso[:, 2].min() + 4, eso[:, 2].max() - 2, 6.0)
    cl = np.array([eso[np.abs(eso[:, 2] - z) < 3].mean(0) for z in zs if (np.abs(eso[:, 2] - z) < 3).sum() > 5])
    a_ch, a_nk, e_nk = np.asarray(L['anast-chest'], float), np.asarray(L['anast-neck'], float), np.asarray(L['eso-neck'], float)
    along = lambda z0, z1: [p for p in cl if z0 <= p[2] <= z1] if z1 > z0 else [p for p in cl[::-1] if z1 <= p[2] <= z0]
    out = {}

    # ---------------------------------------------------------------- intrathoracic leak: the defect faces the right pleura and the back
    face = U(RIGHT * 0.75 - ANT * 0.65)
    defect = a_ch + face * 14.0
    emit_mesh('leak-defect', 'Anastomotic defect (intrathoracic)', 'leak', '#7a1414', ellipsoid(defect, [face, np.cross(face, SUP), SUP], [3.0, 7.0, 5.0]),
              visible=False, note='Schematic: a full-thickness defect in the right posterior wall of the intrathoracic anastomosis. Size it at endoscopy as a fraction of the circumference.')
    coll = defect + face * 16.0 - SUP * 6.0
    emit_mesh('leak-collection', 'Mediastinal collection', 'leak', '#b9a33a', ellipsoid(coll, [face, np.cross(face, SUP), SUP], [13.0, 11.0, 26.0]), opacity=0.7,
              visible=False, note='Schematic: saliva, gastric juice and pus tracking beside the conduit in the posterior mediastinum.')
    # the right pleural collection: posterior and basal in the right chest
    rl = W(ctx['rlung_mm'])
    base = rl[(rl[:, 2] < np.percentile(rl[:, 2], 25)) & (rl[:, 1] < np.percentile(rl[:, 1], 35))]
    eff = base.mean(0) - ANT * 8.0
    emit_mesh('leak-effusion', 'Right pleural collection (empyema)', 'leak', '#b9a33a', ellipsoid(eff, [RIGHT, ANT, SUP], [30.0, 22.0, 42.0]), opacity=0.5,
              visible=False, note='Schematic: the leak has broken into the right pleural space: a layering, infected effusion.')
    # chest drain: right lateral chest wall, 5th-6th space mid-axillary line, down to the effusion and up toward the collection
    sk = W(ctx['skin_mm']); zt = eff[2] + 45.0
    band = sk[(np.abs(sk[:, 2] - zt) < 6) & (np.abs(sk[:, 1] - eff[1] - 25) < 25) & (sk[:, 0] < rl[:, 0].max() + 30)]   # the chest wall, not the arm
    entry = band[np.argmax(band[:, 0])] if len(band) else eff + RIGHT * 90 + SUP * 45
    drain = [entry + RIGHT * 25, entry, entry - RIGHT * 25 - SUP * 10, eff + RIGHT * 10, eff - SUP * 10 + RIGHT * 5, eff + SUP * 25 - RIGHT * 10, coll + RIGHT * 14 - SUP * 12]
    emit_mesh('leak-drain', 'Chest drain (right)', 'leak', '#e9edf0', tube(drain, 3.6, seg=12), visible=False,
              note='Schematic: image-guided or surgical tube drainage of the pleural collection, its tip toward the mediastinal leak.')
    out['leak-drain-entry'] = entry

    # ---------------------------------------------------------------- nasogastric tube: down the esophagus, through the anastomosis, into the conduit
    up = along(a_ch[2], e_nk[2] + 40)
    mouth = e_nk + SUP * 75 + ANT * 55
    ng = [mouth + ANT * 20 + SUP * 10, mouth, *[p + RIGHT * 3 for p in up[::-1]], a_ch + RIGHT * 3, a_ch - SUP * 40 + RIGHT * 2, a_ch - SUP * 90 + ANT * 4]
    emit_mesh('leak-ngt', 'Nasogastric tube (conduit decompression)', 'leak', '#f5d76e', tube(ng, 2.4, seg=10), visible=False,
              note='Schematic: placed under vision past the anastomosis to decompress the conduit. Never passed blind after an esophagectomy.')

    # ---------------------------------------------------------------- endoscopic vacuum therapy: sponge through the defect into the cavity
    sp0 = defect + face * 4.0; sp1 = coll - face * 2.0 - SUP * 6.0
    emit_mesh('leak-evt', 'Endoscopic vacuum sponge (EVT)', 'leak', '#262626', tube([sp0, (sp0 + sp1) / 2, sp1], 6.5, seg=16), visible=False,
              note='Schematic: open-pore polyurethane sponge on a drainage tube, placed endoscopically through the defect into the cavity (intracavitary) or across it in the lumen (intraluminal); continuous suction -100 to -125 mmHg, changed every 3-4 days.')
    evt_tube = [mouth + ANT * 20 + SUP * 18 + RIGHT * 8, mouth + RIGHT * 7, *[p - RIGHT * 4 for p in up[::-1]], a_ch - RIGHT * 2 + face * 4, sp0]
    emit_mesh('leak-evt-tube', 'EVT suction tube (out through the nose)', 'leak', '#c8ccd0', tube(evt_tube, 2.4, seg=10), visible=False,
              note='Schematic: the sponge\'s tube leaves through the nose to a continuous suction unit.')

    # ---------------------------------------------------------------- covered self-expanding stent across the anastomosis
    st = along(a_ch[2] + 45, a_ch[2] + 1) or [a_ch + SUP * 45]
    st = [*st, a_ch, a_ch - SUP * 30, a_ch - SUP * 60]
    emit_mesh('leak-stent', 'Fully covered self-expanding metal stent', 'leak', '#a9b8c4', tube(st, 10.5, seg=24), opacity=0.55, visible=False,
              note='Schematic: a fully covered stent bridging the anastomosis, about 4 cm either side of the defect. Usually removed or exchanged by 4-8 weeks.')

    # ---------------------------------------------------------------- cervical leak: the collection under the neck wound, the wound opened and packed
    inc = np.asarray(ctx['S']['incision-neck']['centroid'], float) if 'incision-neck' in ctx['S'] else a_nk + ANT * 35 - RIGHT * 40
    nk = a_nk + U(inc - a_nk) * 14.0
    emit_mesh('leak-neck-collection', 'Cervical leak: collection under the neck wound', 'leak', '#b9a33a',
              ellipsoid(nk, [U(inc - a_nk), np.cross(U(inc - a_nk), SUP), SUP], [11.0, 8.0, 15.0]), opacity=0.7, visible=False,
              note='Schematic: saliva and pus beside the cervical anastomosis, deep to the sternocleidomastoid.')
    emit_mesh('leak-neck-pack', 'Neck wound opened and packed', 'leak', '#f2efe6',
              tube([inc + U(inc - a_nk) * 6, (inc + nk) / 2, nk + U(inc - a_nk) * 4], 5.0, seg=14), visible=False,
              note='Schematic: the incision opened at the bedside, the space drained and packed with gauze.')

    # ---------------------------------------------------------------- conduit necrosis: the top of the conduit, farthest from the right gastroepiploic artery
    tip = along(a_ch[2] + 6, a_ch[2] - 40)
    emit_mesh('leak-necrosis', 'Necrosis of the conduit tip', 'leak', '#3d2433', tube([a_ch + SUP * 6, *tip[1:]] if len(tip) > 1 else [a_ch + SUP * 6, a_ch - SUP * 40], 15.5, seg=22),
              visible=False, note='Schematic: the ischemic tip of the gastric conduit, the part farthest from its blood supply. Limited: continuity-preserving treatment; extensive: resection and diversion.')

    # ---------------------------------------------------------------- diversion: cervical esophagostomy (spit fistula) to the left neck skin
    stump = along(a_ch[2] + 50, e_nk[2])
    stoma = inc - SUP * 30 + ANT * 4
    eso_t = [*(stump if stump else [a_ch + SUP * 50]), e_nk, e_nk + U(stoma - e_nk) * 25 - SUP * 4, stoma]
    emit_mesh('leak-esophagostomy', 'Cervical esophagostomy (spit fistula)', 'leak', '#cf9459', tube(eso_t, 8.0, seg=18), visible=False,
              note='Schematic: after the conduit is taken down, the esophageal remnant is brought out on the left neck so saliva drains to a bag.')
    out['leak-stoma'] = stoma

    out.update({'leak-defect': defect, 'leak-collection': coll, 'leak-effusion': eff, 'leak-neck': nk})
    return {k: np.asarray(v, float) + CARINA for k, v in out.items()}
