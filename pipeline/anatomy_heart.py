"""Second-generation valve models, fitted to the annuli that cardiac.py traced on the reference CT.

Mitral: anterior and posterior leaflets with thickness, the posterior leaflet in three scallops (P1-P3) with clefts, a
zone of coaptation that curves posteriorly (a "smile") a few millimetres below the annulus, two papillary muscles with
several heads, and branching primary, secondary and strut chordae. Aortic: three semilunar cusps hinged on the crown,
free edges meeting at the centre with a nodule of Arantius on each, bellied toward the ventricle. Tricuspid: septal,
anterior and posterior leaflets closing in a three-pointed star. All in the atlas frame (mm, carina at the origin).

The annulus rings are read back from the meshes cardiac.py wrote (see ring()), so the leaflets hinge exactly on them.
"""
from __future__ import annotations

import numpy as np
import trimesh

from sdfmesh import Vessel, Ellipsoid, mesh

U = lambda v: np.asarray(v, float) / (np.linalg.norm(v) + 1e-9)
LEAF, CHORD, MUSCLE = '#efe3cf', '#f3ecd9', '#8b3a33'


def ring(P, c, n, e1, bins=72):
    """the centreline of a tube-shaped ring mesh: mean of its points per angular bin, as a function of angle"""
    e2 = np.cross(n, e1); r = P - c
    a = np.arctan2(r @ e2, r @ e1)
    edges = np.linspace(-np.pi, np.pi, bins + 1); C = []
    for a0, a1 in zip(edges[:-1], edges[1:]):
        s = (a >= a0) & (a < a1)
        C.append(P[s].mean(0) if s.sum() else np.full(3, np.nan))
    C = np.array(C); ok = ~np.isnan(C[:, 0]); mid = (edges[:-1] + edges[1:]) / 2
    for j in range(3): C[:, j] = np.interp(mid, mid[ok], C[ok, j], period=2 * np.pi)
    def at(t):                                                    # t in radians, periodic
        t = (t + np.pi) % (2 * np.pi) - np.pi
        return np.array([np.interp(t, mid, C[:, j], period=2 * np.pi) for j in range(3)])
    return at


def sheet(G, thick=1.0, smooth=0):
    """a solid leaflet from a grid of surface points (rows x cols x 3): offset both ways along the normals and closed"""
    R_, C_ = G.shape[:2]
    du = np.gradient(G, axis=1); dv = np.gradient(G, axis=0)
    N = np.cross(du, dv); N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-9
    top, bot = (G + N * thick / 2).reshape(-1, 3), (G - N * thick / 2).reshape(-1, 3)
    V = np.vstack([top, bot]); o = R_ * C_; F = []
    idx = lambda r, c: r * C_ + c
    for r in range(R_ - 1):
        for c in range(C_ - 1):
            a, b, d, e = idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)
            F += [[a, b, e], [a, e, d], [o + a, o + e, o + b], [o + a, o + d, o + e]]
    rim = [idx(0, c) for c in range(C_)] + [idx(r, C_ - 1) for r in range(1, R_)] + [idx(R_ - 1, c) for c in range(C_ - 2, -1, -1)] + [idx(r, 0) for r in range(R_ - 2, 0, -1)]
    for i in range(len(rim)):
        a, b = rim[i], rim[(i + 1) % len(rim)]; F += [[a, o + a, o + b], [a, o + b, b]]
    m = trimesh.Trimesh(V, np.array(F), process=True)
    if smooth: trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=smooth)
    m.fix_normals(); return m


def cords(pairs, r):
    out = []
    for a, b in pairs:
        a, b = np.asarray(a, float), np.asarray(b, float)
        if np.linalg.norm(b - a) < 0.5: continue
        out.append(trimesh.creation.cylinder(radius=r, segment=[a, b], sections=8))
        out.append(trimesh.creation.icosphere(subdivisions=1, radius=r * 1.15).apply_translation(b))
    return out


def mitral(L, P_ann, emit_mesh):
    c, n, u, R = L['mv-centre'], U(L['mv-normal']), U(L['mv-anterior']), float(L['mv-dims'][0])
    v = np.cross(n, u)
    A = ring(P_ann, c, n, u)
    tc = np.radians(62)                                          # the commissures: ends of the anterior hinge
    C1 = c + (A(tc) - c) * 0.9 - n * 2.0; C2 = c + (A(-tc) - c) * 0.9 - n * 2.0
    def coap(s):                                                 # the line of closure, a posterior "smile" below the annulus
        b = C1 + (C2 - C1) * s
        return b - u * (0.78 * R * np.sin(np.pi * s)) - n * (5.0 * np.sin(np.pi * s))
    rows, cols = 16, 44
    # anterior leaflet: hinge from +62 degrees to -62 degrees through the front
    G = np.zeros((rows + 4, cols, 3))
    for j, s in enumerate(np.linspace(0, 1, cols)):
        h = A(tc - 2 * tc * s); f = coap(s)
        for i, q in enumerate(np.linspace(0, 1, rows)):
            G[i, j] = h + (f - h) * q - n * (2.6 * np.sin(np.pi * q) * np.sin(np.pi * min(1, 0.15 + s * 0.85)) ** 0.3)
        for k in range(4):                                      # the rough zone: the leaflets meet face to face for a few mm
            G[rows + k, j] = f - n * (1.2 * (k + 1)) + u * 0.55
    emit_mesh('mv-ant-leaflet', 'Anterior mitral leaflet (A1-A3)', 'cardiac', LEAF, sheet(G, 1.1, smooth=4), visible=False,
              note='The larger, semicircular anterior leaflet hinged on a third of the annulus, in continuity with the aortic valve (aortomitral curtain). Schematic, fitted to this scan.')
    # posterior leaflet: hinge from +62 degrees round the back to -62 degrees; three scallops separated by clefts
    G = np.zeros((rows + 4, cols * 2, 3))
    for j, s in enumerate(np.linspace(0, 1, cols * 2)):
        h = A(tc + (2 * np.pi - 2 * tc) * s); f = coap(s)
        reach = 1 - 0.30 * np.exp(-((s - 1 / 3) / 0.03) ** 2) - 0.30 * np.exp(-((s - 2 / 3) / 0.03) ** 2)
        for i, q in enumerate(np.linspace(0, 1, rows)):
            qq = q * reach
            G[i, j] = h + (f - h) * qq - n * (1.6 * np.sin(np.pi * qq))
        fe = G[rows - 1, j]
        for k in range(4):
            G[rows + k, j] = fe - n * (1.2 * (k + 1)) * reach - u * 0.55
    emit_mesh('mv-post-leaflet', 'Posterior mitral leaflet (P1, P2, P3 scallops)', 'cardiac', LEAF, sheet(G, 1.0, smooth=4), visible=False,
              note='The posterior (mural) leaflet hinged on two-thirds of the annulus, divided by clefts into P1 (anterolateral), P2 (middle) and P3 (posteromedial). Its chordae are kept in a chordal-sparing replacement. Schematic.')
    # papillary muscles: a fleshy column with several heads; chordae fan from each head to its half of both leaflets
    pm_parts, ch = [], []
    heads_of = {}
    for key in ('pm-al', 'pm-pm'):
        tip = L[key]; base = (tip - 0.25 * c) / 0.75; root = base + (base - tip) * 0.35
        side = U(np.cross(n, tip - c)); fwd = U(np.cross(side, U(c - tip)))
        heads = [tip + side * 3.5 + U(c - tip) * 2.5, tip - side * 3.5 + U(c - tip) * 2.0, tip + fwd * 3.0 + U(c - tip) * 3.0]
        heads_of[key] = heads
        pm_parts += [Vessel([root, base, tip], [7.5, 6.0, 4.2])] + [Ellipsoid(h, [2.8, 2.8, 3.6]) for h in heads]
    emit_mesh('papillary', 'Papillary muscles (anterolateral and posteromedial), each with several heads', 'cardiac', MUSCLE,
              mesh(pm_parts, voxel=0.35, blend=2.0, smooth=6, density=1.0), visible=False,
              note='Schematic. The anterolateral muscle (often double blood supply) sends chordae to the anterolateral halves of both leaflets; the posteromedial (usually a single supply, the one that ruptures after inferior MI) to the posteromedial halves.')
    al_near_c1 = np.linalg.norm(L['pm-al'] - C1) < np.linalg.norm(L['pm-al'] - C2)
    def edge(leaf, s):                                          # a point on the free edge (just under it)
        if leaf == 'a':
            h = A(tc - 2 * tc * s); f = coap(s); return h + (f - h) * 0.97 - n * 1.2
        h = A(tc + (2 * np.pi - 2 * tc) * s); f = coap(s)
        reach = 1 - 0.30 * np.exp(-((s - 1 / 3) / 0.03) ** 2) - 0.30 * np.exp(-((s - 2 / 3) / 0.03) ** 2)
        return h + (f - h) * 0.97 * reach - n * 1.0
    def belly(leaf, s, q=0.55):
        h = A(tc - 2 * tc * s) if leaf == 'a' else A(tc + (2 * np.pi - 2 * tc) * s); f = coap(s)
        return h + (f - h) * q - n * 2.2
    primary, secondary, struts = [], [], []
    for key, near_c1 in (('pm-al', al_near_c1), ('pm-pm', not al_near_c1)):
        H = heads_of[key]; ss = np.linspace(0.04, 0.48, 6) if near_c1 else np.linspace(0.52, 0.96, 6)
        for leaf in ('a', 'p'):
            for g in range(3):                                   # three fans per leaflet half: trunk, then two or three branches
                grp = ss[g * 2:(g + 1) * 2]; tgt = [edge(leaf, s) for s in np.linspace(grp[0] - 0.02, grp[-1] + 0.02, 3)]
                hd = H[(g + (leaf == 'p')) % 3]; bp = hd + (np.mean(tgt, 0) - hd) * 0.68
                primary += [(hd, bp)]; primary += [(bp, t) for t in tgt]
            secondary += [(H[2], belly(leaf, ss[2], 0.5)), (H[1], belly(leaf, ss[4], 0.6))]
        struts += [(H[0], belly('a', ss[3], 0.45))]                # the strut chordae to the anterior leaflet belly
    parts = cords(primary, 0.28) + cords(secondary, 0.34) + cords(struts, 0.5)
    emit_mesh('chordae', 'Chordae tendineae (primary, secondary and strut)', 'cardiac', CHORD, trimesh.util.concatenate(parts), visible=False,
              note='Primary chordae fan out to the free edges and stop prolapse; secondary chordae to the ventricular surface of the leaflets keep the ventricle\'s shape (preserve them); two thick strut chordae to the anterior leaflet. Schematic.')
    return {'mv-c1': C1, 'mv-c2': C2, 'mv-coapt': coap(0.5)}


def aortic(L, P_ann, emit_mesh):
    c, a, e1, R = L['av-centre'], U(L['av-axis']), U(L['av-e1']), float(L['av-dims'][0])
    e2 = np.cross(a, e1); A = ring(P_ann, c, a, e1, bins=90)
    ts = np.radians(np.arange(0, 360, 2)); hs = np.array([(A(t) - c) @ a for t in ts])
    # commissures: the three high points of the crown, one in each third between the cusp centres
    ang = lambda p: np.arctan2((p - c) @ e2, (p - c) @ e1)
    cen = {'r': 0.0, 'l': ang(L['ostium-l']) if 'ostium-l' in L else np.radians(120)}
    cen['l'] = np.radians(120) if np.sin(cen['l']) > 0 else np.radians(-120); cen['n'] = -cen['l']
    comm = []
    for k1, k2 in (('r', 'l'), ('l', 'n'), ('n', 'r')):
        mid = np.arctan2(np.sin(cen[k1]) + np.sin(cen[k2]), np.cos(cen[k1]) + np.cos(cen[k2]))
        win = np.abs(((ts - mid) + np.pi) % (2 * np.pi) - np.pi) < np.radians(40)
        comm.append(ts[win][np.argmax(hs[win])])
    ctop = np.mean([(A(t) - c) @ a for t in comm]); cbot = min(hs)
    O = c + a * (cbot + 0.62 * (ctop - cbot))                     # where the three free edges meet
    names = {'r': 'Right coronary cusp (leaflet)', 'l': 'Left coronary cusp (leaflet)', 'n': 'Non-coronary cusp (leaflet)'}
    order = {('r', 'l'): 0, ('l', 'n'): 1, ('n', 'r'): 2}
    for k in ('r', 'l', 'n'):
        ca = comm[[i for (x, y), i in order.items() if y == k][0]]; cb = comm[[i for (x, y), i in order.items() if x == k][0]]
        span = ((cb - ca) % (2 * np.pi))
        if span > np.pi: span -= 2 * np.pi                             # go the short way round, through the cusp's own centre
        Ca, Cb = A(ca), A(cb)
        rows, cols = 14, 40; G = np.zeros((rows + 3, cols, 3))
        for j, s in enumerate(np.linspace(0.03, 0.97, cols)):
            h = A(ca + span * s)
            f = Ca + (O - Ca) * (2 * s) if s <= 0.5 else O + (Cb - O) * (2 * s - 1)
            f = f + (c + a * ((O - c) @ a) - f) * 0.0
            for i, q in enumerate(np.linspace(0, 1, rows)):
                G[i, j] = h + (f - h) * q - a * (3.2 * np.sin(np.pi * q) * np.sin(np.pi * s) ** 0.6)
            for kk in range(3):                                  # the lunula: the cusps meet face to face just below the free edge
                G[rows + kk, j] = f + a * (0.9 * (kk + 1))
        leaf = sheet(G, 1.5, smooth=4)
        nod = trimesh.creation.icosphere(subdivisions=2, radius=1.7); nod.apply_scale([1.0, 1.0, 1.3])
        mid_edge = O + (A(ca + span / 2) - O) * 0.08 + a * 1.4; nod.apply_translation(mid_edge)
        emit_mesh(f'av-cusp-{k}', names[k], 'cardiac', '#e3d6bf', trimesh.util.concatenate([leaf, nod]), visible=False,
                  note='A semilunar cusp hinged on the crown-shaped annulus; its free edge runs from commissure to the centre, with a nodule of Arantius where the three meet. Shown thickened, as in degenerative or rheumatic stenosis. Schematic.')
    return {'av-coapt': O}


def tricuspid(L, P_ann, emit_mesh):
    c, n, sd = L['tv-centre'], U(L['tv-normal']), U(L['tv-septal'])
    sd = U(sd - n * (sd @ n)); e2 = np.cross(n, sd); A = ring(P_ann, c, n, sd)
    ant = np.array([0, 1.0, 0]); sgn = 1.0 if (ant @ e2) > 0 else -1.0       # angles increase toward the front
    d = lambda deg: np.radians(deg) * sgn
    arcs = {'tv-septal': (d(-55), d(55)), 'tv-anterior': (d(55), d(205)), 'tv-posterior': (d(205), d(305))}
    O = c - n * 7.0                                             # the closure point, toward the ventricle
    comm = {k: A(t0) + (c - A(t0)) * 0.1 - n * 2.5 for k, (t0, t1) in arcs.items()}
    names = {'tv-septal': 'Septal leaflet (tricuspid)', 'tv-anterior': 'Anterior leaflet (tricuspid), the largest', 'tv-posterior': 'Posterior leaflet (tricuspid), the smallest'}
    keys = list(arcs)
    for i, k in enumerate(keys):
        t0, t1 = arcs[k]; Ca = comm[k]; Cb = comm[keys[(i + 1) % 3]]
        rows, cols = 14, 40; G = np.zeros((rows + 3, cols, 3))
        for j, s in enumerate(np.linspace(0.03, 0.97, cols)):
            h = A(t0 + (t1 - t0) * s)
            f = Ca + (O - Ca) * (2 * s) if s <= 0.5 else O + (Cb - O) * (2 * s - 1)
            for r_, q in enumerate(np.linspace(0, 1, rows)):
                G[r_, j] = h + (f - h) * q - n * (2.2 * np.sin(np.pi * q) * np.sin(np.pi * s) ** 0.6)
            for kk in range(3): G[rows + kk, j] = f - n * (1.0 * (kk + 1))
        emit_mesh(k, names[k], 'cardiac', LEAF, sheet(G, 0.9, smooth=4), visible=False, note='Schematic, fitted to the annulus on this scan.')
    return {'tv-coapt': O}


def load_rings(root):
    """read the annulus meshes back (meshopt-compressed GLBs) through scripts/glb-dump.mjs"""
    import json, subprocess, tempfile
    from pathlib import Path
    ids = ['mitral-annulus', 'aortic-annulus', 'tricuspid-annulus']
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / 'rings.json'
        r = subprocess.run(['node', 'scripts/glb-dump.mjs', str(out), *ids], cwd=root, capture_output=True, text=True, shell=False)
        if r.returncode != 0 or not out.exists(): print('  valves: could not read the annuli', r.stderr[-300:]); return {}
        D = json.loads(out.read_text())
    return {k: np.array(v['P']).reshape(-1, 3) for k, v in D.items()}


def build(ctx):
    """ctx: emit_mesh, LMW (landmarks, atlas frame), and rings {'mitral-annulus': Nx3, ...} (atlas frame) or root (to read them)"""
    L = {k: np.asarray(v, float) for k, v in ctx['LMW'].items()}; out = {}
    R = ctx.get('rings') or load_rings(ctx['root'])
    print('== second-generation valves')
    if 'mitral-annulus' in R and all(k in L for k in ('mv-centre', 'mv-normal', 'mv-anterior', 'mv-dims', 'pm-al', 'pm-pm')):
        out.update(mitral(L, R['mitral-annulus'], ctx['emit_mesh']))
    if 'aortic-annulus' in R and all(k in L for k in ('av-centre', 'av-axis', 'av-e1', 'av-dims')):
        out.update(aortic(L, R['aortic-annulus'], ctx['emit_mesh']))
    if 'tricuspid-annulus' in R and all(k in L for k in ('tv-centre', 'tv-normal', 'tv-septal')):
        out.update(tricuspid(L, R['tricuspid-annulus'], ctx['emit_mesh']))
    return out
