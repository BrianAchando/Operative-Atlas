"""Batch 4 anatomy: thymus, oesophagectomy, thoracic duct, empyema.

From the CT (TotalSegmentator `total`): stomach, liver, spleen, duodenum, pancreas, IVC, thyroid, and the thymus as the
fat of the anterior mediastinum (in an adult the involuted thymus is that fat pad; an extended thymectomy removes all
of it between the phrenic nerves). Schematic, on landmarks: thymic veins, the gastric conduit and anastomoses, the left
gastric artery, the thoracic duct and cisterna chyli, laparotomy and neck incisions, the visceral peel and a posterior
basal empyema collection.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
import trimesh

LEFT = np.array([-1.0, 0, 0]); RIGHT = -LEFT; ANT = np.array([0, 1.0, 0]); SUP = np.array([0, 0, 1.0])


def ring(c, n, r, k=28):
    """points on a circle of radius r about centre c, in the plane normal to n (closed)"""
    n = n / np.linalg.norm(n); a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0]); a /= np.linalg.norm(a); b = np.cross(n, a)
    t = np.linspace(0, 2 * np.pi, k + 1)
    return [c + r * (np.cos(x) * a + np.sin(x) * b) for x in t]


def build(ctx):
    ts, vm, emit, emit_mesh, W, tube, sphere, AT, CT, T, TS = (ctx[k] for k in ('ts', 'vox_mm', 'emit', 'emit_mesh', 'W', 'tube', 'sphere', 'AT', 'CT', 'T', 'TS'))
    has = ctx['has']; LM = {}; L_IDS, R_IDS = set(), set()
    spacing = np.abs(np.diag(AT)[:3])
    print('== mediastinum and abdomen (batch 4)')

    # ------------------------------------------------------------------ abdominal organs, for oesophagectomy
    for id_, lab, col, name, op, faces in (('stomach', 'stomach', '#d69a8a', 'Stomach', 1.0, 9000), ('liver', 'liver', '#8a3b2e', 'Liver', 0.5, 12000),
                                          ('spleen', 'spleen', '#7a2f3e', 'Spleen', 1.0, 5000), ('duodenum', 'duodenum', '#d9a38e', 'Duodenum', 1.0, 5000),
                                          ('pancreas', 'pancreas', '#e2c08a', 'Pancreas', 1.0, 5000), ('ivc', 'inferior_vena_cava', '#5e6a92', 'Inferior vena cava', 1.0, 5000),
                                          ('thyroid', 'thyroid_gland', '#b0645a', 'Thyroid gland', 1.0, 4000)):
        if has(lab): emit(id_, name, 'abdomen' if id_ not in ('thyroid',) else 'mediastinum', col, ts(lab), AT, faces=faces, opacity=op, visible=False)

    eso = vm(ts('esophagus')); aorta = vm(ts('aorta')); st = vm(ts('sternum')); mid_x = float(st[:, 0].mean())
    # oesophageal centre line, one point per 6 mm
    zs = np.arange(eso[:, 2].min() + 3, eso[:, 2].max() - 2, 6.0)
    cl = np.array([eso[np.abs(eso[:, 2] - z) < 3].mean(0) for z in zs if (np.abs(eso[:, 2] - z) < 3).sum() > 5])
    cl[1:-1] = (cl[:-2] + 2 * cl[1:-1] + cl[2:]) / 4
    gej = cl[0]; top = cl[-1]
    LM['gej'] = gej; LM['eso-top'] = top

    # ------------------------------------------------------------------ thymus: the fat of the anterior mediastinum between the great vessels and the sternum
    vess_ids = [TS[n] for n in ('heart', 'aorta', 'superior_vena_cava', 'brachiocephalic_vein_left', 'brachiocephalic_vein_right', 'brachiocephalic_trunk',
                                'common_carotid_artery_left', 'common_carotid_artery_right', 'pulmonary_vein', 'atrial_appendage_left', 'trachea') if has(n)]
    lung_ids = [TS[n] for n in TS if n.startswith('lung_') and has(n)]
    stern_id = TS['sternum']; cart_id = TS.get('costal_cartilages', -1)
    inv = np.linalg.inv(AT)
    k_of = lambda z: int(round((inv @ np.array([0, 0, z, 1.0]))[2]))
    heart = vm(ts('heart')); lbcv = vm(ts('brachiocephalic_vein_left'))
    z_hi = float(vm(ts('thyroid_gland'))[:, 2].min()) if has('thyroid_gland') else float(lbcv[:, 2].max() + 25)
    z_lo = float(heart[:, 2].max() - 55)
    th = np.zeros(T.shape, bool)
    i_mid = int(round((inv @ np.array([mid_x, 0, 0, 1.0]))[0])); half = int(48 / spacing[0])
    ka, kb = sorted((k_of(z_lo), k_of(z_hi)))
    for k in range(max(ka, 0), min(kb, T.shape[2] - 1) + 1):
        sl = T[:, :, k]; hu = CT[:, :, k]
        ves = np.isin(sl, vess_ids); stn = (sl == stern_id) | (sl == cart_id)
        for i in range(max(i_mid - half, 0), min(i_mid + half, T.shape[0] - 1) + 1):
            cols = slice(max(i - 2, 0), i + 3)
            vj = np.flatnonzero(ves[cols, :].any(0)); sj = np.flatnonzero(stn[cols, :].any(0))
            if not len(vj) or not len(sj): continue
            jb, jf = vj.max() + 2, sj.min() - 2                     # just in front of the vessels, just behind the sternum
            if jf - jb < 2: continue
            seg = np.arange(jb, jf)
            ok = (hu[i, seg] > -190) & (hu[i, seg] < 90) & ~np.isin(sl[i, seg], lung_ids)
            th[i, seg[ok], k] = True
    th = ndimage.binary_opening(th, iterations=1)
    lab, n = ndimage.label(th)
    if n:
        th = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        th = ndimage.binary_closing(th, iterations=2)
        emit('thymus', 'Thymus and anterior mediastinal fat', 'mediastinum', '#e8c77a', th, AT, faces=8000, visible=False,
             note='The fat of the anterior mediastinum between the great vessels and the sternum, from the thyroid to the pericardium: in an adult the involuted thymus lies in it, '
                  'and an extended thymectomy removes all of it from phrenic nerve to phrenic nerve.')
        tp = vm(th)
        # thymic veins: from the back of the gland into the left brachiocephalic vein
        from scipy.spatial import cKDTree
        kd = cKDTree(tp); veins = []
        for dx in (-12.0, 8.0):
            tgt = lbcv[np.argmin(np.abs(lbcv[:, 0] - (mid_x + dx)) + 0.2 * np.abs(lbcv[:, 2] - np.median(lbcv[:, 2])))]
            _, j = kd.query(tgt + ANT * 12); a = tp[j]
            veins.append((a, tgt))
        for m, (a, b) in enumerate(veins, 1):
            path = [a + (b - a) * t + ANT * 3 * np.sin(np.pi * t) for t in np.linspace(0, 1, 5)]
            emit_mesh(f'thymic-vein-{m}', f'Thymic vein {m}', 'mediastinum', '#5e6a92', tube([W(p) for p in path], 1.6), visible=False,
                      note='Schematic: one to three thymic veins leave the back of the gland and drain into the left brachiocephalic (innominate) vein; tie or clip them before lifting the gland off it.')
            d = (b - a) / np.linalg.norm(b - a); p = a + (b - a) * 0.55
            ctx['division'][f'thymic-vein-{m}'] = {'point': [round(float(x), 1) for x in W(p)], 'dir': [round(float(x), 3) for x in d], 'radius': 2.0}
        LM['thymus-c'] = tp.mean(0); LM['thymus-lo'] = tp[np.argmin(tp[:, 2])]; LM['thymus-hi'] = tp[np.argmax(tp[:, 2])]
        LM['thymus-horn-l'] = tp[np.argmax(tp[:, 2] - 0.4 * (tp[:, 0] - mid_x))]; LM['thymus-horn-r'] = tp[np.argmax(tp[:, 2] + 0.4 * (tp[:, 0] - mid_x))]

    # ------------------------------------------------------------------ oesophagectomy: conduit, anastomoses, left gastric artery, incisions
    if has('stomach'):
        stm = vm(ts('stomach'))
        duo = vm(ts('duodenum')) if has('duodenum') else stm[stm[:, 0] > np.percentile(stm[:, 0], 95)]
        from scipy.spatial import cKDTree
        kd = cKDTree(duo); dist, _ = kd.query(stm[::3])
        pylorus = stm[::3][np.argmin(dist)]
        # greater curvature: the outer (left, inferior) edge of the stomach, between the pylorus and the fundus
        s_ = stm[::5]; c_ = s_.mean(0)
        u = gej - pylorus; u /= np.linalg.norm(u)
        gc = []
        for t in np.linspace(0.05, 0.95, 10):
            q = pylorus + (gej - pylorus) * t
            band = s_[np.abs((s_ - q) @ u) < 6]
            if not len(band): continue
            out = (band - q); out -= np.outer(out @ u, u)
            far = band[np.argmax(out @ (LEFT * 0.5 + np.array([0, 0, -1.0]) + (c_ - q) * 0))]
            gc.append(far)
        gc = np.array(gc)
        LM['pylorus'] = pylorus; LM['hiatus'] = gej + SUP * 20
        # the conduit: tubularised greater curvature, from the pylorus up the oesophageal bed
        def conduit(top_z, id_, name, note):
            up = [p for p in cl if p[2] <= top_z]
            path = [pylorus, (pylorus + gej) / 2 + LEFT * 18 - SUP * 10, gej + SUP * 4, *up[1:]]
            path = np.array(path); path[1:-1] = (path[:-2] + 2 * path[1:-1] + path[2:]) / 4
            emit_mesh(id_, name, 'abdomen', '#d69a8a', tube([W(p) for p in path], 14.0, seg=20), visible=False, note=note)
            return path[-1]
        z_az = ctx['azygos_arch'][2]
        ivor_top = conduit(z_az + 22, 'conduit-chest', 'Gastric conduit (to the apex of the chest)', 'Schematic: the tubularised greater curvature, 4-5 cm wide, pulled up the oesophageal bed to above the azygos arch (Ivor Lewis).')
        neck_top = conduit(top[2], 'conduit-neck', 'Gastric conduit (to the neck)', 'Schematic: the conduit through the posterior mediastinum to the left neck (McKeown, transhiatal).')
        for id_, c, name in (('anast-chest', ivor_top, 'Intrathoracic anastomosis (Ivor Lewis)'), ('anast-neck', neck_top, 'Cervical anastomosis')):
            n_ = (cl[-1] - cl[-3]); emit_mesh(id_, name, 'abdomen', '#e6e2da', tube([W(p) for p in ring(c, n_, 15.5)], 2.2), visible=False,
                                              note='Schematic: circular stapled (intrathoracic) or hand-sewn / linear-stapled (neck).')
            LM[id_] = c
        # proximal division of the oesophagus: above the azygos (Ivor Lewis) or in the neck
        for key, z in (('eso-ivor', z_az + 18), ('eso-neck', top[2] - 12)):
            j = np.argmin(np.abs(cl[:, 2] - z)); LM[key] = cl[j]; LM[key + '-axis'] = (cl[min(j + 1, len(cl) - 1)] - cl[max(j - 1, 0)]); LM[key + '-axis'] /= np.linalg.norm(LM[key + '-axis'])
        # left gastric artery: from the coeliac trunk (front of the aorta at L1) up to the lesser curvature below the cardia
        if has('vertebrae_L1'):
            zc = float(vm(ts('vertebrae_L1'))[:, 2].max() + 5)
            a_s = aorta[np.abs(aorta[:, 2] - zc) < 3]
            if len(a_s):
                coel = a_s[np.argmax(a_s[:, 1])] + ANT * 2
                lc = gej - SUP * 30 + RIGHT * 10 + ANT * 8
                path = [coel, coel + ANT * 18 + SUP * 5, (coel + lc) / 2 + ANT * 14 + SUP * 10 + LEFT * 5, lc]
                emit_mesh('lga', 'Left gastric artery (from the coeliac trunk)', 'abdomen', '#b8382e', tube([W(p) for p in path], 2.4), visible=False,
                          note='Schematic: the coeliac trunk leaves the front of the aorta at T12-L1; the left gastric artery runs up to the lesser curvature. Divided at its origin with the coeliac nodes; the conduit lives on the right gastroepiploic artery.')
                d = path[2] - path[1]; d /= np.linalg.norm(d)
                ctx['division']['lga'] = {'point': [round(float(x), 1) for x in W(path[1] + (path[2] - path[1]) * 0.3)], 'dir': [round(float(x), 3) for x in d], 'radius': 3.0}
                LM['coeliac'] = coel
        # right gastroepiploic arcade along the greater curvature: the conduit's blood supply
        if len(gc) > 3:
            g = gc.copy(); g[1:-1] = (g[:-2] + 2 * g[1:-1] + g[2:]) / 4
            emit_mesh('rgea', 'Right gastroepiploic artery (greater curvature arcade)', 'abdomen', '#b8382e', tube([W(p + (p - c_) / np.linalg.norm(p - c_) * 6) for p in g], 1.8), visible=False,
                      note='Schematic: the arcade along the greater curvature that the conduit lives on. Keep 1-2 cm away from it when dividing the omentum and short gastrics.')

    # skin incisions: upper midline laparotomy and left neck
    skin = ctx['skin_mm']
    def front_skin(z, x, ymax=1e9):
        s = skin[(np.abs(skin[:, 2] - z) < 3) & (np.abs(skin[:, 0] - x) < 4) & (skin[:, 1] < ymax)]
        return s[np.argmax(s[:, 1])] if len(s) else None
    xiph = st[np.argmin(st[:, 2])]
    lap = [front_skin(z, mid_x) for z in np.arange(xiph[2] + 10, xiph[2] - 170, -12)]
    lap = np.array([p for p in lap if p is not None])
    if len(lap) > 3:
        lap[:, 1] += 2.5
        emit_mesh('incision-lap', 'Upper midline laparotomy', 'ports-open-left', '#d0433a', tube([W(p) for p in lap], 1.8), visible=False,
                  note='Schematic: xiphoid to umbilicus. For the abdominal phase of an oesophagectomy (or a roof-top incision; or laparoscopy).')
        LM['lap'] = lap[len(lap) // 3]
    st_top = st[np.argmax(st[:, 2])]
    neck = [front_skin(st_top[2] + 8 + 9 * i, mid_x - 22 - 7 * i, st_top[1] + 35) for i in range(7)]
    neck = np.array([p for p in neck if p is not None and p[1] < st_top[1] + 35])     # the neck, not the chin
    if len(neck) > 2:
        neck[:, 1] += 2.5
        emit_mesh('incision-neck', 'Left neck incision (anterior border of sternocleidomastoid)', 'ports-open-left', '#d0433a', tube([W(p) for p in neck], 1.8), visible=False,
                  note='Schematic: along the anterior border of the left sternocleidomastoid, from the sternal notch up. The carotid sheath goes laterally; the left recurrent laryngeal nerve lies in the tracheo-oesophageal groove: no metal retractor on it.')
        LM['neck'] = neck[len(neck) // 2]

    # ------------------------------------------------------------------ thoracic duct and cisterna chyli
    vert = {v: vm(ts(f'vertebrae_{v}')) for v in ('L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5', 'T4', 'T3', 'T2', 'T1') if has(f'vertebrae_{v}')}
    front = lambda v: vert[v][np.argmax(vert[v][:, 1] - 0.3 * np.abs(vert[v][:, 0] - np.median(vert[v][:, 0])))]
    td = []
    for v in ('L1', 'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6'):
        if v not in vert: continue
        f = front(v); a_s = aorta[np.abs(aorta[:, 2] - f[2]) < 4]
        ax = a_s[:, 0].mean() if len(a_s) else f[0] - 20
        # between the aorta (on the left) and the azygos (on the right), on the front of the vertebral bodies
        td.append(np.array([f[0] + (ax - f[0]) * 0.15 + RIGHT[0] * 6, f[1] + 5, f[2]]))
    cis = td[0] - SUP * 10 + RIGHT * 2
    for v in ('T5', 'T4'):
        if v not in vert: continue
        f = front(v); e_ = eso[np.abs(eso[:, 2] - f[2]) < 4]
        ex = e_[:, 0].mean() if len(e_) else f[0]
        td.append(np.array([ex + (8 if v == 'T5' else -10), max(f[1] + 4, e_[:, 1].min() - 3 if len(e_) else f[1] + 4), f[2]]))
    for v in ('T3', 'T2', 'T1'):
        if v not in vert: continue
        f = front(v); e_ = eso[np.abs(eso[:, 2] - f[2]) < 4]
        ex = e_[:, 0].min() if len(e_) else f[0] - 15
        td.append(np.array([ex - 4, (e_[:, 1].mean() if len(e_) else f[1] + 8), f[2]]))
    lb_top = lbcv[lbcv[:, 2] > np.percentile(lbcv[:, 2], 60)]
    angle = lb_top[np.argmin(lb_top[:, 0])]                                # the left venous angle (subclavian and internal jugular)
    td += [td[-1] + SUP * 22 + LEFT * 10, angle + SUP * 18 + LEFT * 6, angle + SUP * 4]
    td = np.array([cis, *td])
    td[1:-1] = (td[:-2] + 2 * td[1:-1] + td[2:]) / 4
    emit_mesh('thoracic-duct', 'Thoracic duct', 'mediastinum', '#f3eecd', tube([W(p) for p in td], 1.8), visible=False,
              note='Schematic: from the cisterna chyli (L1-L2) through the aortic hiatus, up between the aorta and the azygos on the right of the vertebral bodies, '
                   'crossing to the left behind the oesophagus at T4-T6, then up on the left of the oesophagus to the left venous angle. Variants (double, plexiform) are common.')
    cm_ = trimesh.creation.icosphere(2, 1.0); cm_.apply_scale([7, 6, 16]); cm_.apply_translation(W(cis))
    emit_mesh('cisterna', 'Cisterna chyli', 'mediastinum', '#f3eecd', cm_, visible=False,
              note='Schematic: the sac at L1-L2, behind and to the right of the aorta, deep to the right crus.')
    # mass ligation: just above the right hemidiaphragm (T9-T10), all tissue between the aorta and the azygos
    zl = vert['T10'][:, 2].mean() if 'T10' in vert else td[3][2]
    j = int(np.argmin(np.abs(td[:, 2] - zl)))
    d = td[min(j + 1, len(td) - 1)] - td[max(j - 1, 0)]; d /= np.linalg.norm(d)
    ctx['division']['thoracic-duct'] = {'point': [round(float(x), 1) for x in W(td[j])], 'dir': [round(float(x), 3) for x in d], 'radius': 3.5}
    LM['td-ligation'] = td[j]; LM['td-cross'] = td[int(np.argmin(np.abs(td[:, 2] - (vert['T5'][:, 2].mean() if 'T5' in vert else td[j][2] + 60))))]

    # ------------------------------------------------------------------ empyema: the visceral peel over the left lung and a posterior basal collection
    lungF = ts('lung_upper_lobe_left', 'lung_lower_lobe_left')
    # work in a box round the left lung (dilations on the whole volume are slow)
    ii = np.argwhere(lungF); m = int(20 / spacing[0]) + 2
    lo = np.maximum(ii.min(0) - m, 0); hi = np.minimum(ii.max(0) + m + 1, T.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    AB = AT.copy(); AB[:3, 3] = AT[:3, :3] @ lo + AT[:3, 3]
    lungL = lungF[box]; Tb = T[box]; CTb = CT[box]
    to_mm = lambda idx: idx @ AB[:3, :3].T + AB[:3, 3]
    dist_out = ndimage.distance_transform_edt(~lungL, sampling=spacing)
    shell = (dist_out > 0) & (dist_out <= 4.0)
    hil = np.array(ctx['hilum_l_scanner'])
    pi = np.argwhere(shell); far = np.linalg.norm(to_mm(pi) - hil, axis=1) > 38            # no peel over the hilum
    shell[:] = False; shell[tuple(pi[far].T)] = True
    emit('peel-l', 'Visceral peel (fibrous cortex), left', 'pleura', '#c9b98a', shell, AB, faces=12000, opacity=0.85, visible=False,
         note='Schematic: the organised fibrinous cortex on the visceral pleura that traps the lung in a stage III (chronic) empyema. Decortication peels it off so the lung re-expands.')
    L_IDS.add('peel-l')
    # the collection: pleural space behind and below the lung, inside the ribs
    solid = np.isin(Tb, [TS[n] for n in TS if n.startswith(('rib_', 'vertebrae_', 'autochthon', 'aorta', 'esophagus', 'heart', 'scapula', 'liver', 'spleen', 'stomach', 'lung_')) and has(n)])
    coll = (dist_out > 4.0) & (dist_out <= 16.0) & ~solid & (CTb > -300) & (CTb < 200)
    lp = to_mm(np.argwhere(lungL))
    # a rounded pocket in the posterior costophrenic gutter
    ec = np.array([np.percentile(lp[:, 0], 35), np.percentile(lp[:, 1], 12), np.percentile(lp[:, 2], 18)]); er = np.array([60.0, 40.0, 55.0])
    cz = np.argwhere(coll); cm = to_mm(cz)
    keep = (((cm - ec) / er) ** 2).sum(1) < 1.0
    coll[:] = False; coll[tuple(cz[keep].T)] = True
    coll = ndimage.binary_opening(coll, iterations=1)
    lab, n = ndimage.label(coll)
    if n:
        coll = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        coll = ndimage.binary_closing(coll, iterations=3)
        emit('empyema-l', 'Empyema collection (posterior, basal), left', 'pleura', '#b9a24a', coll, AB, faces=6000, opacity=0.8, visible=False,
             note='Schematic: loculated pus in the posterior costophrenic gutter, where it collects in a supine patient. Drawn in the pleural space of this scan, which is itself normal.')
        L_IDS.add('empyema-l'); LM['empyema'] = to_mm(np.argwhere(coll)).mean(0)
    return LM, L_IDS, R_IDS
