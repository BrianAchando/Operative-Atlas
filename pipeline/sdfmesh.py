"""Organic meshes from signed distance fields: tapered vessel trees with filleted branch points, and bones built from
blended ellipsoids and capsules. Shapes are combined with a smooth minimum (so a branch flares into its parent and a
condyle blends into the shaft), voxelised on a fine grid, meshed with marching cubes, smoothed and decimated.

All coordinates in millimetres, in whatever frame the caller uses (the atlas frame here).
"""
from __future__ import annotations

import numpy as np
import trimesh
from skimage.measure import marching_cubes


def _spline(P, step=1.0):
    P = np.asarray(P, float)
    if len(P) < 3: return np.linspace(P[0], P[-1], max(2, int(np.linalg.norm(P[-1] - P[0]) / step)))
    out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)]
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1]); return np.array(out)


class Vessel:
    """a tapered tube along a smooth curve through control points; radii given at the control points (or a callable of 0..1)"""
    def __init__(self, ctrl, radii, step=1.0, wiggle=0.0, seed=0):
        Q = _spline(ctrl, step)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]; t = s / max(s[-1], 1e-9)
        if wiggle:                                         # gentle tortuosity, zero at both ends
            rng = np.random.default_rng(seed); T = np.gradient(Q, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
            a = np.cross(T, [0.3, 0.5, 0.8]); a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9; b = np.cross(T, a)
            ph = rng.uniform(0, 6.28, 2); L = s[-1]
            w = np.sin(np.pi * t) * wiggle
            Q = Q + a * (w * np.sin(2 * np.pi * s / max(L / 3.2, 40) + ph[0]))[:, None] + b * (w * 0.6 * np.cos(2 * np.pi * s / max(L / 2.3, 55) + ph[1]))[:, None]
        if callable(radii): R = np.array([radii(x) for x in t])
        else:
            r = np.asarray(radii, float)
            R = np.interp(t, np.linspace(0, 1, len(r)), r)
        self.P, self.R = Q, R
        self.lo = Q.min(0) - R.max(); self.hi = Q.max(0) + R.max()

    def eval(self, X):
        d = np.full(len(X), np.inf)
        for i in range(len(self.P) - 1):
            a, b = self.P[i], self.P[i + 1]; ra, rb = self.R[i], self.R[i + 1]
            ab = b - a; L2 = float(ab @ ab) + 1e-12
            h = np.clip(((X - a) @ ab) / L2, 0, 1)
            q = X - (a + h[:, None] * ab)
            d = np.minimum(d, np.sqrt((q * q).sum(1)) - (ra + (rb - ra) * h))
        return d

    def segs(self):
        for i in range(len(self.P) - 1):
            r = max(self.R[i], self.R[i + 1])
            yield (np.minimum(self.P[i], self.P[i + 1]) - r, np.maximum(self.P[i], self.P[i + 1]) + r, i)


class Ellipsoid:
    def __init__(self, c, radii, axes=None):
        self.c = np.asarray(c, float); self.r = np.asarray(radii, float)
        self.A = np.eye(3) if axes is None else np.asarray(axes, float)       # rows: unit axes
        e = np.abs(self.A.T) @ self.r
        self.lo, self.hi = self.c - e, self.c + e

    def eval(self, X):
        q = (X - self.c) @ self.A.T / self.r
        k = np.linalg.norm(q, axis=1)
        return (k - 1.0) * self.r.min()


def frame(z, hint=(1.0, 0, 0)):
    """an orthonormal frame whose third axis is z"""
    z = np.asarray(z, float); z = z / np.linalg.norm(z)
    x = np.asarray(hint, float); x = x - z * (x @ z)
    if np.linalg.norm(x) < 1e-6: x = np.cross(z, [0, 1.0, 0])
    x /= np.linalg.norm(x); y = np.cross(z, x); return np.array([x, y, z])


def smin(a, b, k):
    if k <= 0: return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def mesh(prims, voxel=0.5, blend=1.5, faces=None, smooth=8, subtract=(), density=None):
    """union (smoothly blended) of the primitives, minus `subtract` primitives, as a watertight trimesh"""
    lo = np.min([p.lo for p in prims], axis=0) - 3 * voxel - blend
    hi = np.max([p.hi for p in prims], axis=0) + 3 * voxel + blend
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    F = np.full(shape, 1e3, np.float32)
    def sub_eval(prim, plo, phi, fn):
        i0 = np.clip(np.floor((plo - blend - lo) / voxel).astype(int) - 1, 0, shape - 1)
        i1 = np.clip(np.ceil((phi + blend - lo) / voxel).astype(int) + 2, 0, shape)
        if np.any(i1 <= i0): return
        g = np.stack(np.meshgrid(*[np.arange(i0[k], i1[k]) for k in range(3)], indexing='ij'), -1).reshape(-1, 3)
        X = lo + g * voxel
        sl = tuple(slice(i0[k], i1[k]) for k in range(3))
        cur = F[sl].reshape(-1)
        F[sl] = fn(cur, fn_eval(prim, X)).reshape(F[sl].shape)
    def fn_eval(prim, X): return prim.eval(X)
    for p in prims:
        if isinstance(p, Vessel):
            for a, b, i in p.segs():                       # per segment: only the voxels near it
                seg = Vessel.__new__(Vessel); seg.P = p.P[i:i + 2]; seg.R = p.R[i:i + 2]
                sub_eval(seg, a, b, lambda c, d: smin(c, d, blend))
        else:
            sub_eval(p, p.lo, p.hi, lambda c, d: smin(c, d, blend))
    for p in subtract:
        sub_eval(p, p.lo, p.hi, lambda c, d: np.maximum(c, -d))
    v, f, _, _ = marching_cubes(F, 0.0, spacing=(voxel, voxel, voxel), allow_degenerate=False)
    m = trimesh.Trimesh(v + lo, f[:, ::-1], process=True)
    if smooth: trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=smooth)
    if density: faces = max(800, int(m.area * density))             # triangles per mm² of surface
    if faces and len(m.faces) > faces:
        try: m = m.simplify_quadric_decimation(face_count=faces)
        except Exception: pass
    m.fix_normals(); return m
