import sys, json, numpy as np, nibabel as nib
from scipy import ndimage
sys.path.insert(0, '.')
from hilum import Tree, lobe_origins
lv = nib.load('/home/claude/seg/lv_left.nii.gz'); L = np.asanyarray(lv.dataobj); A = lv.affine
tot = nib.load('/home/claude/seg/total.nii.gz'); T = np.asanyarray(tot.dataobj)
m = np.linalg.inv(tot.affine) @ A
def pull(labels):
    src = np.isin(T, labels).astype(np.uint8)
    return ndimage.affine_transform(src, m[:3, :3], offset=m[:3, 3], output_shape=L.shape, order=0) > 0
U = pull([10]); D = pull([11]); heart = pull([51])
np.savez_compressed('/home/claude/seg/lobes_lv.npz', U=U, D=D, heart=heart)
def mm(v): return v @ A[:3, :3].T + A[:3, 3]
def biggest(msk):
    lab, n = ndimage.label(msk); 
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
art = biggest(L == 3); vein = L == 4; air = biggest((L == 1) | (L == 2))
# roots
av = np.argwhere(art); aroot = mm(av[np.argmax(av[:, 0])])
vv = np.argwhere(vein & ndimage.binary_dilation(heart, iterations=3)); vroot = mm(vv[np.argmax(vv[:, 0])]) if len(vv) else None
wv = np.argwhere(air); wroot = mm(wv[np.argmax(wv[:, 2])])
print('roots', aroot.round(), None if vroot is None else vroot.round(), wroot.round())
for name, msk, root in [('art', art, aroot), ('vein', vein, vroot), ('air', air, wroot)]:
    t = Tree(msk, A, root); terr = t.territory({'U': U, 'D': D})
    print(f'== {name}: {len(t.vox)} nodes, kept {t.keep.sum()}')
    for lobe in ('U', 'D'):
        for o in lobe_origins(t, terr, lobe)[:10]:
            sub = t.subtree(o); c = t.mm[sub].mean(0)
            print(f'  {lobe} origin node {o} at {t.mm[o].round()} rootdist {t.dist[o]:.0f} size {terr["_size"][o]:.0f} territory centroid {c.round()}')
