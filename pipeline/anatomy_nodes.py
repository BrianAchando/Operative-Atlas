"""Second-generation lymph node stations: each station sphere becomes a small cluster of bean-shaped nodes (two to four,
touching, of unequal size) within the same space, so a station reads as nodes in fat rather than a ball. Placement and
size come from the station records already in the atlas (IASLC map positions from build.py). Atlas frame."""
from __future__ import annotations

import numpy as np

from sdfmesh import Ellipsoid, mesh, frame


def build(ctx):
    emit_mesh, S = ctx['emit_mesh'], ctx['structures']; done = []
    for s in list(S):
        if s.get('group') != 'nodes' or not s['id'].startswith('ln-'): continue
        b = np.array(s['bbox']); c = b.mean(0); r = float(np.ptp(b, 0).mean() / 2)
        rng = np.random.default_rng(__import__("zlib").crc32(s["id"].encode()))
        k = 3 if r >= 6.5 else 2
        prims = []
        for i in range(k):
            off = rng.normal(0, r * 0.32, 3) if i else np.zeros(3)
            rad = r * np.array([0.62, 0.44, 0.38]) * (1.0 if i == 0 else rng.uniform(0.6, 0.85))
            prims.append(Ellipsoid(c + off, rad, axes=frame(rng.normal(0, 1, 3), rng.normal(0, 1, 3))))
        m = mesh(prims, voxel=max(0.3, r / 18), blend=0.6, smooth=6, density=2.0)
        emit_mesh(s['id'], s['name'], 'nodes', s['colour'], m, opacity=s.get('opacity', 1.0), visible=s.get('visible', True) is not False,
                  note='Schematic: a station holds a few nodes in the mediastinal fat; placed on landmarks (IASLC map).')
        done.append(s['id'])
    print('  node stations:', done)
    return {}
