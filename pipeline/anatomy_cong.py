"""Organic congenital models (replace the ring-and-disc schematics of pathology_cong.py by id):

  vsd-defect        a window of interventricular septum with a perimembranous defect: thin fibrous rim above, muscle below
  vsd-patch         a slightly domed patch, larger than the defect
  vsd-suture        interrupted pledgeted mattress sutures around the rim (RV side)
  tof-vsd           the large malalignment VSD, with the anteriorly deviated outlet septum as a muscular ridge over it
  tof-infundibulum  hypertrophied septal and parietal bands and trabeculations narrowing the RV outflow (also tof-resection)
  tof-pv            a small, thickened, bicuspid, domed pulmonary valve with a narrow slit orifice
  tof-vsd-patch     a domed patch baffling the LV to the aorta

Frames (centre, normal, radius) are recovered from the meshes pathology_cong.py has just written, so this follows them.
"""
from __future__ import annotations

import numpy as np
import trimesh

from sdfmesh import Vessel, Ellipsoid, mesh, frame
from anatomy_heart import sheet
from anatomy_neck import Field

U = lambda v: np.asarray(v, float) / (np.linalg.norm(v) + 1e-9)
DEFECT, DACRON, SUTURE, PLEDGET, FIBRE, MUSCLE = '#f2b84b', '#f1f1ea', '#3fa7d6', '#f6f6f2', '#d8c9ae', '#8a2f2f'


def _frame(P, toward):
    """centre, unit normal (pointing toward `toward`) and in-plane axes of a flat ring or disc"""
    c = P.mean(0); w, V = np.linalg.eigh(np.cov((P - c).T)); n = V[:, 0]
    if (np.asarray(toward) - c) @ n < 0: n = -n
    return c, n


def _read(root, ids):
    import json, subprocess, tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / 'm.json'
        r = subprocess.run(['node', 'scripts/glb-dump.mjs', str(out), *ids], cwd=root, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists(): print('  congenital: could not read meshes', r.stderr[-300:]); return {}
        D = json.loads(out.read_text())
    return {k: np.array(v['P']).reshape(-1, 3) for k, v in D.items()}


def septal_window(c, n, up, hole, R=15.0, thick=5.0, rim_thin=1.6, ridge=None):
    """a disc of septum around a defect. hole=(a, b): semi-axes along `up` and across. Thin (membranous) above the hole,
    muscular elsewhere. ridge: optional (height, offset along n) for a malaligned outlet septum over the upper rim"""
    c, n, up = np.asarray(c, float), U(n), U(up); up = U(up - n * (up @ n)); ac = np.cross(n, up)
    a, b = hole
    def fn(X):
        q = X - c; x, y, z = q @ ac, q @ up, q @ n
        rho = np.hypot(x, y)
        th = thick - (thick - rim_thin) * np.clip((y - a * 0.4) / (a * 1.2), 0, 1) * np.exp(-(x / (b * 1.6)) ** 2)
        d = np.maximum(np.abs(z) - th / 2, rho - R)
        e = np.sqrt((x / b) ** 2 + (y / a) ** 2)
        d = np.maximum(d, -(e - 1.0) * min(a, b))
        return d
    prims = [Field(c - R - 4, c + R + 4, fn)]
    if ridge is not None:
        h, off = ridge
        pts = [c + up * (a + 1.5) + ac * (b * x_) + n * off * (1 - 0.3 * x_ * x_) for x_ in np.linspace(-1.4, 1.4, 7)]
        prims.append(Vessel(pts, [2.2, 3.0, 3.6, 3.8, 3.6, 3.0, 2.2], step=0.6))
    return mesh(prims, voxel=0.3, blend=1.5, smooth=8, density=1.6)


def patch(c, n, up, a, b, dome=1.6, thick=0.8):
    c, n, up = np.asarray(c, float), U(n), U(up); up = U(up - n * (up @ n)); ac = np.cross(n, up)
    lens = mesh([Ellipsoid(c + n * dome * 0.5, [a, b, thick + dome * 0.5], np.array([up, ac, n]))], voxel=0.2, blend=0, smooth=4, density=1.5)
    rim = [c + up * np.cos(t) * a + ac * np.sin(t) * b for t in np.linspace(0, 2 * np.pi, 49)]
    return trimesh.util.concatenate([lens, mesh([Vessel(rim, [0.7, 0.7], step=0.5)], voxel=0.2, blend=0, smooth=3, density=2.0)])


def pledgets(c, n, up, a, b, k=10, off=1.2):
    c, n, up = np.asarray(c, float), U(n), U(up); up = U(up - n * (up @ n)); ac = np.cross(n, up)
    pl, su = [], []
    for t in np.linspace(0, 2 * np.pi, k, endpoint=False):
        rad = up * np.cos(t) + ac * np.sin(t); tan = np.cross(n, rad)
        p = c + up * np.cos(t) * (a + 2.2) + ac * np.sin(t) * (b + 2.2) + n * off
        pl.append(Ellipsoid(p, [1.8, 0.9, 0.5], np.array([tan, rad, n])))
        for s in (-1, 1):                                            # the two limbs of the mattress, over the patch edge
            q = p + tan * s * 1.1
            su.append(Vessel([q + n * 0.4, q - rad * 1.6 + n * 1.6, q - rad * 3.2 + n * 0.9], [0.22, 0.22, 0.22], step=0.3))
    return mesh(pl, voxel=0.2, blend=0.4, smooth=4, density=2.0), mesh(su, voxel=0.12, blend=0.1, smooth=2, density=2.0)


def bicuspid(c, ax, r, ant, emit_mesh):
    """a small thick bicuspid pulmonary valve, domed toward the trunk (systole), with a fused raphe and slit orifice"""
    c, ax = np.asarray(c, float), U(ax); e1 = U(ant - ax * (ant @ ax)); e2 = np.cross(ax, e1)
    parts = []
    for side in (1, -1):
        rows, cols = 10, 30; G = np.zeros((rows, cols, 3))
        for j, s in enumerate(np.linspace(0.04, 0.96, cols)):
            t = side * (np.pi * s - np.pi / 2) + (0 if side > 0 else np.pi)
            h = c + (e1 * np.cos(t) + e2 * np.sin(t)) * r
            f = c + ax * 4.0 + e2 * (r * 0.72 * np.sin(t)) + e1 * side * 0.5
            for i, q in enumerate(np.linspace(0, 1, rows)):
                G[i, j] = h + (f - h) * q + ax * (2.2 * np.sin(np.pi * q))
        parts.append(sheet(G, 1.8, smooth=3))
    ring_ = [c + (e1 * np.cos(t) + e2 * np.sin(t)) * r for t in np.linspace(0, 2 * np.pi, 41)]
    raphe = [c + e1 * r * 0.95, c + e1 * r * 0.4 + ax * 3.0]
    parts.append(mesh([Vessel(ring_, [1.4, 1.4], step=0.5), Vessel(raphe, [1.2, 0.8], step=0.4)], voxel=0.25, blend=0.8, smooth=4, density=1.6))
    return trimesh.util.concatenate(parts)


def infundibulum(c, ax, r, ant, rng):
    """muscle bundles around the outflow: a septal band and a parietal band crossing it, plus hypertrophied trabeculations"""
    c, ax = np.asarray(c, float), U(ax); e1 = U(ant - ax * (ant @ ax)); e2 = np.cross(ax, e1)
    prims = []
    for k in range(14):                                               # a lumpy, thick muscular collar narrowing the lumen
        t = 2 * np.pi * k / 14 + rng.normal(0, 0.1); rad = e1 * np.cos(t) + e2 * np.sin(t)
        p = c + rad * (r + 2.5) + ax * rng.normal(0, 2.0)
        prims.append(Ellipsoid(p, [3.5, 3.2 * rng.uniform(0.8, 1.2), 6.5], np.array([U(np.cross(ax, rad)), rad, ax])))
    # septal band (from the septum, left) and parietal band (free wall, right), each a thick column running obliquely
    prims.append(Vessel([c - e2 * (r + 6) - ax * 9, c - e2 * (r + 1) - ax * 2, c - e2 * (r - 0.5) + e1 * 2 + ax * 5], [4.0, 3.6, 3.0], step=0.6))
    prims.append(Vessel([c + e2 * (r + 6) - ax * 8, c + e2 * (r + 0.5) + e1 * 1.5 - ax * 1, c + e1 * (r - 0.5) + ax * 6], [3.8, 3.4, 2.8], step=0.6))
    return mesh(prims, voxel=0.4, blend=2.2, smooth=8, density=1.2)


def build(ctx):
    emit_mesh, S, root = ctx['emit_mesh'], ctx['S'], ctx['root']
    ids = [i for i in ('vsd-defect', 'tof-vsd', 'tof-pv', 'tof-infundibulum') if i in S]
    if not ids or 'rv' not in S: return {}
    print('== congenital (organic)')
    M = _read(root, ids); rv = np.array(S['rv']['centroid']); LM = {}
    rng = np.random.default_rng(3); SUPv = np.array([0, 0, 1.0])
    if 'vsd-defect' in M:
        c, n = _frame(M['vsd-defect'], rv)
        emit_mesh('vsd-defect', 'Perimembranous ventricular septal defect (about 10 mm)', 'congenital', DEFECT, septal_window(c, n, SUPv, (5.2, 4.6)), opacity=0.9, visible=False,
                  note='Schematic window of septum seen from the RV. The defect lies in the membranous septum (its thin fibrous upper rim), below the right and non-coronary cusps and partly under the septal tricuspid leaflet; muscle forms its lower rim, along which the His bundle runs (posteroinferior).')
        emit_mesh('vsd-patch', 'Dacron patch (VSD closure)', 'congenital', DACRON, patch(c + n * 2.8, n, SUPv, 7.8, 7.0), visible=False,
                  note='Schematic. Dacron, PTFE or treated pericardium, on the RV side, larger than the defect, with interrupted pledgeted or running 5-0/6-0 polypropylene.')
        pl, su = pledgets(c, n, SUPv, 7.8, 7.0, off=3.0)
        emit_mesh('vsd-suture', 'Suture line: shallow bites on the RV side of the posteroinferior rim', 'congenital', SUTURE, trimesh.util.concatenate([pl, su]), visible=False,
                  note='Interrupted pledgeted mattress sutures. Along the posteroinferior rim the bites are shallow and 3-5 mm away from the edge, on the RV side, to stay clear of the His bundle.')
        LM['vsd-c'] = c
    if 'tof-vsd' in M:
        c, n = _frame(M['tof-vsd'], rv)
        emit_mesh('tof-vsd', 'Malalignment VSD (large, subaortic, non-restrictive)', 'congenital', DEFECT,
                  septal_window(c, n, SUPv, (8.5, 7.5), R=18.0, ridge=(3.5, 4.5)), opacity=0.9, visible=False,
                  note='Schematic, from the RV. The outlet septum (the ridge above the defect) is deviated anteriorly and cephalad, leaving a large defect under the aorta, which straddles it; the same deviation narrows the RV outflow.')
        emit_mesh('tof-vsd-patch', 'VSD patch (closes the VSD, baffling LV to aorta)', 'congenital', DACRON, patch(c + n * 2.8, n, SUPv, 10.5, 9.5, dome=2.2), visible=False,
                  note='Schematic. Sewn on the RV side so that the LV ejects through the defect into the aorta.')
    ant = np.array([0, 1.0, 0])
    if 'tof-pv' in M:
        P = M['tof-pv']; pa = np.array(S['pa-trunk']['centroid']) if 'pa-trunk' in S else P.mean(0) + SUPv * 20
        c, ax = _frame(P, pa); r = float(np.percentile(np.linalg.norm((P - c) - np.outer((P - c) @ ax, ax), axis=1), 50))
        emit_mesh('tof-pv', 'Small, thickened pulmonary valve and annulus', 'congenital', FIBRE, bicuspid(c, ax, max(4.0, r), ant, emit_mesh), visible=False,
                  note='Schematic: often bicuspid, thickened and domed, with a narrow orifice and a small annulus; the annulus z-score decides whether it can be kept.')
    if 'tof-infundibulum' in M:
        P = M['tof-infundibulum']; pv = M['tof-pv'].mean(0) if 'tof-pv' in M else P.mean(0) + SUPv * 10
        c, ax = _frame(P, pv); r = float(np.percentile(np.linalg.norm((P - c) - np.outer((P - c) @ ax, ax), axis=1), 20))
        inf = infundibulum(c, ax, max(3.0, r * 0.8), ant, rng)
        note = 'Schematic. The deviated outlet septum, the septal and parietal bands and hypertrophied trabeculations narrow the RV outflow below the valve; this is the dynamic part that spasms in a tet spell.'
        emit_mesh('tof-infundibulum', 'Infundibular stenosis: hypertrophied septoparietal muscle bands', 'congenital', MUSCLE, inf, visible=False, note=note)
        emit_mesh('tof-resection', 'Muscle resected from the RV outflow', 'congenital', MUSCLE, inf.copy(), visible=False,
                  note='The obstructing parietal and septal band muscle is divided and excised through the infundibulum (or the right atrium and pulmonary valve), avoiding the conduction tissue and the tricuspid apparatus.')
    return LM
