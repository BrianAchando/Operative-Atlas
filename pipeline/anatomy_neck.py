"""Laryngeal cartilages as organic shapes (replaces the annulus and box schematics in neck.py):

  cricoid       a signet ring: a low anterior arch and a tall posterior lamina, the only complete ring of the airway
  thyroid-cart  two laminae meeting in front at about 90 degrees (the prominence), with the superior thyroid notch,
                the oblique line's tubercle, and superior and inferior horns (the inferior horns reach the cricoid)

Atlas frame, millimetres. Built as signed distance fields and meshed with sdfmesh.
"""
from __future__ import annotations

import numpy as np

from sdfmesh import mesh

RIGHT, ANT, SUP = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])


class Field:
    """an arbitrary distance-like field over a box"""
    def __init__(self, lo, hi, fn):
        self.lo, self.hi, self.fn = np.asarray(lo, float), np.asarray(hi, float), fn

    def eval(self, X):
        return self.fn(X)


def _sd_polygon(p, V):
    """signed distance from 2D points p (N,2) to a closed polygon V (M,2); negative inside"""
    d = np.full(len(p), np.inf); s = np.ones(len(p))
    for i in range(len(V)):
        a, b = V[i], V[i - 1]
        e = b - a; w = p - a
        h = np.clip((w @ e) / (e @ e), 0, 1)
        d = np.minimum(d, np.linalg.norm(w - h[:, None] * e, axis=1))
        c1 = p[:, 1] >= a[1]; c2 = p[:, 1] < b[1]; c3 = e[0] * w[:, 1] > e[1] * w[:, 0]
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s[flip] *= -1
    return s * d


def cricoid(c, r_in, r_out, arch=6.0, lamina=20.0):
    """c: centre of the ring's lower border (atlas frame)"""
    c = np.asarray(c, float)
    def fn(X):
        q = X - c; rho = np.hypot(q[:, 0], q[:, 1] * 1.08); th = np.arctan2(q[:, 1], q[:, 0])
        back = ((1 - np.sin(th)) / 2) ** 2.2                      # 0 in front, 1 behind
        top = arch + (lamina - arch) * back
        bot = -1.5 * (1 - back) * np.clip(np.sin(th), 0, 1)        # the arch dips a little in front
        d_r = np.maximum(r_in - rho, rho - r_out)
        d_z = np.maximum(bot - q[:, 2], q[:, 2] - top)
        return np.maximum(d_r, d_z)
    e = r_out + 2
    return mesh([Field(c - [e, e, 4], c + [e, e, lamina + 2], fn)], voxel=0.3, blend=0, smooth=10, density=2.5)


# one thyroid lamina in its own plane: u from the midline (prominence) backwards, v upwards from the lower border (mm)
LAMINA = np.array([(0, 1.5), (7, 0), (15, -1), (19, -2.2), (23, -0.5), (27, -0.5), (27.5, -11), (30.5, -11.5), (31.5, 0),
                   (32, 22), (31.8, 35), (29.3, 35.5), (28.3, 23), (22, 25), (13, 24.5), (6, 21), (2, 16), (0, 13.5)])



def _chaikin(V, n=2):
    for _ in range(n):
        V = np.concatenate([[0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b] for a, b in zip(V, np.roll(V, -1, axis=0))])
    return V


LAMINA = _chaikin(LAMINA)


def thyroid(apex, angle_deg=90.0, thick=2.4):
    """apex: the lower end of the prominence in the midline (atlas frame)"""
    apex = np.asarray(apex, float); half = np.radians(angle_deg) / 2
    parts = []
    for sx in (-1, 1):
        eu = np.array([sx * np.sin(half), -np.cos(half), 0.0]); ev = SUP; ew = np.cross(eu, ev)
        def fn(X, eu=eu, ew=ew):
            q = X - apex; uv = np.c_[q @ eu, q @ ev]
            return np.maximum(_sd_polygon(uv, LAMINA), np.abs(q @ ew) - thick / 2)
        pts = apex + LAMINA[:, :1] * eu + LAMINA[:, 1:] * ev
        parts.append(Field(pts.min(0) - 4, pts.max(0) + 4, fn))
    return mesh(parts, voxel=0.35, blend=1.2, smooth=10, density=2.0)
