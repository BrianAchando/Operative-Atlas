"""Name the left hilar branches from TotalSegmentator's lung_vessels output (arteries / veins / airways).

Each tree is skeletonised and turned into a graph; a shortest-path tree is grown from its root (the left PA where it
enters the crop, the left atrium, the trachea). Every skeleton node then knows which lobe its downstream territory
lies in (the share of its descendants inside the left upper lobe, U, and the lower lobe, D). A branch *belongs* to a
lobe from the first node whose territory is >= 90% in that lobe. Those first nodes are the branch origins; their
order along the trunk and the position of their territory give the names:

  arteries: first upper-lobe origin = truncus anterior; later origins whose territory is inferior and anterior =
            lingular artery; the rest = posterior segmental (ascending) arteries. First lower-lobe origin with a
            superior-posterior territory = superior segmental artery (A6); the continuing trunk = basal trunk.
  veins:    upper-lobe tree = superior pulmonary vein; lower-lobe tree = inferior pulmonary vein.
  airways:  upper-lobe tree = upper lobe bronchus (upper division + lingular); lower-lobe tree = lower lobe bronchus;
            the path from the carina to their split = left main bronchus.

Every name is a teaching label from one scan's geometry: branch counts vary (3 to 7 arteries to the upper lobe), and
a surgeon should check them against the CT.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


class Tree:
    """Skeleton of a mask as a graph, with a shortest-path tree from a root point."""

    def __init__(self, mask: np.ndarray, affine: np.ndarray, root_mm: np.ndarray):
        self.mask = mask; self.aff = affine
        sk = np.argwhere(skeletonize(mask))
        self.vox = sk
        self.mm = sk @ affine[:3, :3].T + affine[:3, 3]
        kd = cKDTree(sk)
        pairs = kd.query_pairs(1.8, output_type="ndarray")
        w = np.linalg.norm(self.mm[pairs[:, 0]] - self.mm[pairs[:, 1]], axis=1)
        n = len(sk)
        self.g = coo_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])), shape=(n, n)).tocsr()
        # keep the component that contains the root
        ncomp, comp = connected_components(self.g, directed=False)
        self.root = int(cKDTree(self.mm).query(root_mm)[1])
        keep = comp == comp[self.root]
        self.keep = keep
        self.dist, self.pred = dijkstra(self.g, indices=self.root, return_predecessors=True)
        self.dist[~keep] = np.inf
        order = np.argsort(self.dist)
        self.order = order[np.isfinite(self.dist[order])]
        self.children: dict[int, list[int]] = {}
        for i in self.order:
            p = self.pred[i]
            if p >= 0:
                self.children.setdefault(int(p), []).append(int(i))

    def territory(self, lobe_masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Per node: number of descendant nodes (itself included) inside each lobe, and in total."""
        n = len(self.vox)
        inside = {k: m[tuple(self.vox.T)].astype(np.float64) for k, m in lobe_masks.items()}
        tot = np.ones(n)
        acc = {k: v.copy() for k, v in inside.items()}
        for i in self.order[::-1]:                      # leaves first
            p = self.pred[i]
            if p >= 0:
                tot[p] += tot[i]
                for k in acc: acc[k][p] += acc[k][i]
        out = {k: acc[k] / tot for k in acc}
        out["_size"] = tot
        return out

    def subtree(self, node: int) -> np.ndarray:
        out = [node]; stack = [node]
        while stack:
            for c in self.children.get(stack.pop(), []):
                out.append(c); stack.append(c)
        return np.array(out)

    def path_to_root(self, node: int) -> list[int]:
        p = [node]
        while self.pred[p[-1]] >= 0:
            p.append(int(self.pred[p[-1]]))
        return p

    def along(self, node: int, mm: float) -> int:
        """Node reached by walking `mm` down the largest subtree from `node` (used for staple lines and directions)."""
        size = self.subtree_sizes()
        cur = node; d0 = self.dist[node]
        while self.dist[cur] - d0 < mm:
            ch = self.children.get(cur)
            if not ch: break
            cur = max(ch, key=lambda c: size[c])
        return cur

    _sizes: np.ndarray | None = None

    def subtree_sizes(self) -> np.ndarray:
        if self._sizes is None:
            s = np.ones(len(self.vox))
            for i in self.order[::-1]:
                p = self.pred[i]
                if p >= 0: s[p] += s[i]
            self._sizes = s
        return self._sizes


def lobe_origins(t: Tree, terr: dict[str, np.ndarray], lobe: str, frac=0.9, min_size=25) -> list[int]:
    """Nodes where the territory first becomes >= frac in `lobe` (the parent is below frac), largest first."""
    f = terr[lobe]; out = []
    for i in t.order:
        p = t.pred[i]
        if p < 0: continue
        if f[i] >= frac and f[p] < frac and terr["_size"][i] >= min_size:
            out.append(int(i))
    return sorted(out, key=lambda i: -terr["_size"][i])


def voxels_of(t: Tree, nodes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask voxels whose nearest skeleton node (within the kept component) is in `nodes`."""
    keep_idx = np.flatnonzero(t.keep)
    kd = cKDTree(t.vox[keep_idx])
    vox = np.argwhere(mask)
    _, nn = kd.query(vox)
    owner = keep_idx[nn]
    sel = np.isin(owner, nodes)
    out = np.zeros(mask.shape, bool); out[tuple(vox[sel].T)] = True
    return out


def owner_map(t: Tree, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep_idx = np.flatnonzero(t.keep)
    kd = cKDTree(t.vox[keep_idx])
    vox = np.argwhere(mask)
    _, nn = kd.query(vox)
    return vox, keep_idx[nn]
