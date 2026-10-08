"""Vascular, part two: common iliac artery aneurysm and its repairs (iliac branch device; internal iliac coils with an
external iliac extension; open interposition graft), and the ankle-brachial index (cuffs and the Doppler probe).

Schematic, drawn on the iliac and limb geometry that pathology_vasc.py and pathology_new.py lay out below the scan.
Works in carina-centred coordinates (W landmarks in, meshes out); returns landmarks in scanner coordinates.
"""
from __future__ import annotations

import numpy as np
import trimesh

U = lambda v: np.asarray(v, float) / (np.linalg.norm(v) + 1e-9)
RIGHT, ANT, SUP = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
ART, GRAFT, STENT, ANEUR = '#c0392b', '#f1f1ea', '#c9d1d9', '#c96a5a'


def lathe(P, radii, sections=40):
    """a tube of varying radius along the polyline P (closed at both ends)"""
    P = np.asarray(P, float); R = np.asarray(radii, float)
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    ref = np.array([1.0, 0, 0]) if abs(T[0][0]) < 0.9 else np.array([0, 1.0, 0])
    n = np.cross(T[0], ref); n /= np.linalg.norm(n)
    V, F = [], []
    for i, (p, t, r) in enumerate(zip(P, T, R)):
        n = n - t * np.dot(n, t); n /= np.linalg.norm(n) + 1e-9; b = np.cross(t, n)
        for k in range(sections):
            a = 2 * np.pi * k / sections; V.append(p + r * (np.cos(a) * n + np.sin(a) * b))
    for i in range(len(P) - 1):
        for k in range(sections):
            a, b_ = i * sections + k, i * sections + (k + 1) % sections
            F += [[a, b_ + sections, b_], [a, a + sections, b_ + sections]]
    c0, c1 = len(V), len(V) + 1; V += [P[0], P[-1]]
    last = (len(P) - 1) * sections
    for k in range(sections):
        F.append([c0, (k + 1) % sections, k]); F.append([c1, last + k, last + (k + 1) % sections])
    m = trimesh.Trimesh(np.array(V), np.array(F), process=True); m.fix_normals(); return m


def bez(a, b, c, n=24):
    t = np.linspace(0, 1, n)[:, None]; return (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c


def build(ctx):
    emit_mesh, tube, CARINA = ctx['emit_mesh'], ctx['tube'], ctx['CARINA']
    L = {k: np.asarray(v, float) for k, v in ctx['LMW'].items()}
    out = {}
    if all(k in L for k in ('bifurcation', 'cia-r', 'cfa-r')):
        lat = RIGHT
        bif = L['bifurcation']; cia_mid = L['cia-r']
        cia0 = bif + lat * 4; cia_end = 2 * cia_mid - bif               # as laid out in pathology_vasc.py
        eia_end = L['cfa-r'] + SUP * 20 - ANT * 4
        eia_mid = (cia_end + eia_end) / 2 + ANT * 10
        iia = [cia_end, cia_end + lat * 8 - SUP * 20 - ANT * 22, cia_end + lat * 14 - SUP * 45 - ANT * 30]
        # ------------------------------------------------ the aneurysm: fusiform, 42 mm, the whole right common iliac
        P = bez(cia0, (cia0 + cia_end) / 2 + lat * 3, cia_end, 40)
        t = np.linspace(0, 1, len(P)); base = 5.2
        prof = base + (21.0 - base) * np.sin(np.pi * np.clip((t - 0.08) / 0.84, 0, 1)) ** 0.9
        P = P + np.outer((prof - base) * 0.25, ANT) + np.outer((prof - base) * 0.2, lat)       # bulges forward and outward
        emit_mesh('ciaa-r', 'Common iliac artery aneurysm, right (42 mm)', 'vascular', ANEUR, lathe(P, prof), opacity=0.6, visible=False,
                  note='Schematic: a fusiform 4.2 cm aneurysm of the right common iliac artery, the whole length from the aortic bifurcation to the iliac bifurcation. Repair threshold 40 mm (ESVS 2024).')
        out['ciaa-r'] = P[len(P) // 2]
        emit_mesh('ciaa-thrombus-r', 'Mural thrombus in the iliac aneurysm', 'vascular', '#7d3a2c', lathe(P, np.maximum(prof * 0.82, base + 0.6)), opacity=0.85, visible=False)
        # ------------------------------------------------ iliac branch device: external limb into the EIA, side branch bridged into the IIA
        ib = [*bez(cia0, (cia0 + cia_end) / 2 + lat * 3, cia_end, 14), *bez(cia_end, cia_end + (eia_mid - cia_end) * 0.6, eia_mid, 10)[1:]]
        side = [cia_end - SUP * 6 + lat * 2, *iia[1:2], iia[1] + (iia[2] - iia[1]) * 0.5]
        rings = []
        for p in ib[::2]:
            r_ = trimesh.creation.torus(major_radius=7.0, minor_radius=0.6, major_sections=32, minor_sections=6); r_.apply_translation(p); rings.append(r_)
        emit_mesh('ibd-r', 'Iliac branch device, right (with a covered bridging stent into the internal iliac)', 'vascular', STENT,
                  trimesh.util.concatenate([tube(ib, 6.6, seg=20), tube(side, 3.6, seg=14), *rings]), opacity=0.8, visible=False,
                  note='Schematic: the iliac branch device lands in the external iliac artery; its side branch is bridged into the internal iliac artery with a covered stent, so pelvic flow is preserved. Joined to a standard bifurcated EVAR.')
        # ------------------------------------------------ the alternative: coil the internal iliac, extend the limb into the external iliac
        coils = []
        c0 = iia[1] + (iia[2] - iia[1]) * 0.15
        ax = U(iia[2] - iia[1]); n1 = U(np.cross(ax, ANT)); n2 = np.cross(ax, n1)
        for k in range(46):
            a = k * 0.55; p = c0 + ax * (k * 0.45) + 2.6 * (np.cos(a) * n1 + np.sin(a) * n2)
            s = trimesh.creation.icosphere(subdivisions=1, radius=0.9); s.apply_translation(p); coils.append(s)
        emit_mesh('iia-coils-r', 'Coils in the right internal iliac artery', 'vascular', '#5d6670', trimesh.util.concatenate(coils), visible=False,
                  note='Schematic: the internal iliac origin is embolised so the limb can be extended into the external iliac without a type II endoleak from the pelvis. Buttock claudication follows in a quarter to a third.')
        ext = [*bez(cia0, (cia0 + cia_end) / 2 + lat * 3, cia_end, 14), *bez(cia_end, cia_end + (eia_mid - cia_end) * 0.6, eia_mid, 10)[1:]]
        emit_mesh('eia-ext-r', 'EVAR limb extended into the right external iliac', 'vascular', STENT, tube(ext, 6.2, seg=20), opacity=0.8, visible=False)
        # ------------------------------------------------ open repair: an interposition graft to the iliac bifurcation, with the internal iliac re-implanted
        g = [*bez(cia0, (cia0 + cia_end) / 2 + lat * 3, cia_end + (eia_mid - cia_end) * 0.25, 18)]
        jump = [g[len(g) * 3 // 4], g[len(g) * 3 // 4] - SUP * 10 - ANT * 12 + lat * 4, iia[1]]
        emit_mesh('graft-cia-r', 'Open repair: interposition graft, right common iliac, internal iliac re-implanted', 'vascular', GRAFT,
                  trimesh.util.concatenate([tube(g, 5.6, seg=22), tube(jump, 3.4, seg=14)]), visible=False,
                  note='Schematic: through a retroperitoneal or midline approach, a 10-12 mm Dacron graft from the aortic bifurcation to the external iliac origin, with a side limb (or the native vessel) to the internal iliac.')
        out['cia-end-r'] = cia_end; out['iia-r'] = iia[1]
    # ---------------------------------------------------------------- ankle-brachial index: cuffs and the Doppler pen
    if 'ankle-r' in L and 'ankle-dp' in L:
        a = L['ankle-r'] + SUP * 70 + ANT * 6
        cuff = trimesh.creation.cylinder(radius=46.0, height=60.0, sections=48); cuff.apply_translation(a)
        inner = trimesh.creation.cylinder(radius=40.0, height=62.0, sections=48); inner.apply_translation(a)
        try:
            cm = cuff.difference(inner)
        except Exception:
            cm = trimesh.creation.annulus(r_min=40.0, r_max=46.0, height=60.0, sections=48); cm.apply_translation(a)
        emit_mesh('abi-cuff-ankle', 'Blood pressure cuff above the ankle', 'leg', '#3d5a80', cm, opacity=0.9, visible=False,
                  note='Schematic: a 12 cm cuff just above the malleoli; inflate until the Doppler signal goes, deflate until it returns.')
        dp = L['ankle-dp']
        pen = [dp + ANT * 8 + SUP * 4, dp + ANT * 40 + SUP * 40, dp + ANT * 70 + SUP * 76]
        emit_mesh('doppler-pen', 'Hand-held Doppler probe on the dorsalis pedis', 'leg', '#e6e6e6', tube(pen, 5.0, seg=16), visible=False,
                  note='8 MHz pencil probe at 45-60 degrees to the artery, with gel; repeat on the posterior tibial behind the medial malleolus.')
        out['abi-ankle'] = a
    if 'elbow-brachial' in L:
        e = L['elbow-brachial'] + SUP * 110 + RIGHT * 4
        cuff = trimesh.creation.annulus(r_min=44.0, r_max=50.0, height=110.0, sections=48); cuff.apply_translation(e)
        emit_mesh('abi-cuff-arm', 'Blood pressure cuff on the upper arm', 'arm', '#3d5a80', cuff, opacity=0.9, visible=False,
                  note='Schematic: brachial systolic pressure by Doppler in both arms; use the higher of the two.')
        out['abi-arm'] = e
    return {k: np.asarray(v, float) + CARINA for k, v in out.items()}
