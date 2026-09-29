"""The operative field for open-heart steps: sterile drapes round a sternotomy window, and the opened pericardium
hitched up as a cradle round the heart. Built from the body outline and the heart mask of the reference CT.

In the app a surgeon's-eye cutaway removes whatever of these (and the sternum and skin) lies between the camera and
the structure being looked at, so side and oblique teaching views stay open while the field frames the heart.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage


def build(ctx):
    emit_mesh, meshing = ctx['emit_mesh'], ctx['meshing']
    W = ctx['W']; LM = {}
    st = ctx['sternum_mm']
    if st is None or len(st) < 50: print('== field: no sternum; skipped'); return LM
    print('== operative field')
    stW = np.array([W(p) for p in st[::4]])
    mid_x = float(np.median(stW[:, 0])); z0, z1 = float(stW[:, 2].min()), float(stW[:, 2].max())
    # ---------------------------------------------------------------- drapes: the anterior chest, a window over the sternum
    skin = meshing.mesh_from_mask(ctx['body'], ctx['body_aff'], 16000, sigma=1.5, taubin_iterations=10)
    if skin is not None:
        cen = skin.vertices.mean(0)
        fn = skin.face_normals.copy()
        if np.mean(np.einsum('ij,ij->i', fn, skin.triangles_center - cen)) < 0: fn = -fn
        fc = skin.triangles_center
        chest = (fc[:, 2] > z0 - 70) & (fc[:, 2] < z1 + 60) & (fn[:, 1] > 0.05)
        window = (np.abs(fc[:, 0] - mid_x) < 40) & (fc[:, 2] > z0 - 18) & (fc[:, 2] < z1 + 12)
        keep = np.where(chest & ~window)[0]
        if len(keep) > 100:
            d = skin.submesh([keep], append=True)
            vn = d.vertex_normals.copy()
            if np.mean(np.einsum('ij,ij->i', vn, d.vertices - cen)) < 0: vn = -vn
            v = d.vertices
            # lift off the skin, with soft folds (larger away from the window, where the drape is not tacked down)
            dist = np.clip((np.abs(v[:, 0] - mid_x) - 40) / 60, 0, 1)
            fold = 2.2 * np.sin(v[:, 0] * 0.09 + np.sin(v[:, 2] * 0.05) * 2.0) * dist + 1.2 * np.sin(v[:, 2] * 0.13 + v[:, 0] * 0.02) * dist
            d.vertices = v + vn * (5.0 + fold[:, None])
            emit_mesh('drape-sternotomy', 'Sterile drapes (sternotomy window)', 'field', '#2b5670', d, visible=False,
                      note='Adhesive-edged drapes leave a window over the sternum, from the notch to below the xiphoid.')
    # ---------------------------------------------------------------- the pericardium, opened in front and hitched up
    H = ctx['heart']; A = ctx['heart_aff']
    if H is not None and H.sum() > 500:
        sp = np.abs(np.diag(A)[:3])
        it_out = max(2, int(round(4.0 / sp.mean()))); it_in = max(1, int(round(1.5 / sp.mean())))
        shell = ndimage.binary_dilation(H, iterations=it_out) & ~ndimage.binary_dilation(H, iterations=it_in)
        idx = np.argwhere(shell); pts = idx @ A[:3, :3].T + A[:3, 3]
        hp = np.argwhere(H) @ A[:3, :3].T + A[:3, 3]
        cy, ext_y = float(np.median(hp[:, 1])), float(np.ptp(hp[:, 1]))
        front = pts[:, 1] > cy - 0.05 * ext_y                                      # opened: only the posterior bowl stays
        cut = shell.copy(); cut[tuple(idx[front].T)] = False
        lab, nl = ndimage.label(cut)
        if nl > 1: cut = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        m = meshing.mesh_from_mask(cut, ctx['heart_aff_shifted'], 12000, sigma=1.2, taubin_iterations=20)
        if m is not None:
            v = m.vertices.copy(); cx = float(np.median(v[:, 0])); cyW = cy - ctx['CARINA'][1]
            t = np.clip((v[:, 1] - (cyW - 0.35 * ext_y)) / (0.3 * ext_y), 0, 1)  # the rim flares out to the sides, hitched up a little
            v[:, 0] += np.sign(v[:, 0] - cx) * t ** 2 * 16; v[:, 1] += t ** 2 * 8
            m.vertices = v
            emit_mesh('pericardium-open', 'Pericardium, opened and hitched (cradle)', 'field', '#d8c3a2', m, opacity=0.55, visible=False,
                      note='Opened in the midline and hitched to the sternal edges or drapes with stay sutures: the heart sits in a cradle, raised toward the surgeon.')
            LM['pericardium'] = np.array([cx, cyW, float(np.median(v[:, 2]))]) + ctx['CARINA']
    return LM
