"""The chest wall a surgeon goes through, and the landmarks used to plan the incision.

From the CT itself (TotalSegmentator `total`): scapulae, clavicles, humeri, erector spinae, costal cartilages.
Schematic, laid on the patient's own rib cage: the muscle sheets (latissimus dorsi, serratus anterior, trapezius,
rhomboids, pectoralis major, intercostals), the long thoracic and thoracodorsal nerves, surface lines (anterior,
mid and posterior axillary), the scapular tip and nipple, and the VATS port and muscle-sparing incisions.

Every sheet is a surface over a (height, azimuth) grid around the hemithorax: azimuth is measured from the lateral
direction (0) toward anterior (+) and posterior (-), about that side's lung centre. The radius is the rib cage's outer
surface plus a fixed depth, so each muscle sits on this patient's ribs in its textbook layer: intercostals flush with
the ribs; serratus anterior on them; latissimus dorsi and pectoralis major over that; trapezius outermost.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
import trimesh

MUSCLE = '#a8423a'
NERVE = '#efe1a8'
LINE = '#6a3fc8'           # skin-marker violet

ZS = 2.0      # grid spacing, mm of height
AS = 2.0      # grid spacing, degrees of azimuth


class Side:
    """the hemithorax in (z, azimuth) coordinates"""

    def __init__(self, ctx, side):
        self.ctx, self.side = ctx, side
        self.L = side == 'left'
        self.sg = -1 if self.L else 1
        self.c = ctx['lung_c'] if self.L else ctx['lung_cr']
        ts, vm = ctx['ts'], ctx['vox_mm']
        s = side
        ribs = np.vstack([vm(ts(f'rib_{s}_{n}')) for n in range(1, 13) if ctx['has'](f'rib_{s}_{n}')])
        mid_x = float(vm(ts('sternum'))[:, 0].mean())
        cc = vm(ts('costal_cartilages')); cc = cc[(cc[:, 0] < mid_x) if self.L else (cc[:, 0] > mid_x)]
        st = vm(ts('sternum'))
        self.z0 = float(ribs[:, 2].min()) - 6; self.z1 = float(max(ribs[:, 2].max(), cc[:, 2].max() if len(cc) else -1e9)) + 6
        self.zg = np.arange(self.z0, self.z1, ZS)
        self.ag = np.arange(-175.0, 150.0 + 1e-6, AS)
        self.cage = self.radius_map(np.vstack([ribs, cc, st]), fill=60.0, sigma=3.0)
        self.scap = self.radius_map(vm(ts(f'scapula_{s}')), fill=16.0, sigma=3.0)
        self.erec = self.radius_map(vm(ts(f'autochthon_{s}')), fill=16.0, sigma=3.0)
        sc = vm(ts(f'scapula_{s}'))
        self.tip = sc[np.argmin(sc[:, 2])]                                   # inferior angle
        self.tip_az = self.az(self.tip)
        a = self.az(sc); self.scap_med = float(np.percentile(a, 3))           # medial border
        self.scap_top = float(sc[:, 2].max())
        v = {n: vm(ts(f'vertebrae_{n}')) for n in ('T1', 'T3', 'T5', 'T7', 'T10', 'T12') if ctx['has'](f'vertebrae_{n}')}
        self.vz = {n: float(p[:, 2].mean()) for n, p in v.items()}
        ec = vm(ts(f'autochthon_{s}')); self.spine_az = float(np.percentile(self.az(ec), 2))   # the midline behind, as seen from this side
        self.rz = {n: ctx['rib_z'](n, 0, side) for n in range(1, 13)}

    def az(self, p):
        p = np.atleast_2d(p)
        a = np.degrees(np.arctan2(p[:, 1] - self.c[1], self.sg * (p[:, 0] - self.c[0])))
        return a if len(a) > 1 else float(a[0])

    def radius_map(self, pts, fill, sigma=1.2):
        """outer radius of `pts` on the grid; gaps up to `fill` mm high (intercostal spaces) interpolated, then smoothed"""
        a = self.az(pts); r = np.hypot(pts[:, 0] - self.c[0], pts[:, 1] - self.c[1])
        iz = np.round((pts[:, 2] - self.zg[0]) / ZS).astype(int); ia = np.round((a - self.ag[0]) / AS).astype(int)
        ok = (iz >= 0) & (iz < len(self.zg)) & (ia >= 0) & (ia < len(self.ag))
        R = np.full((len(self.zg), len(self.ag)), np.nan)
        np.fmax.at(R, (iz[ok], ia[ok]), r[ok])
        # fill gaps along z, then along azimuth (short gaps only)
        for axis, lim in ((0, int(fill / ZS)), (1, 8)):
            R = np.moveaxis(R, axis, 0)
            for j in range(R.shape[1]):
                col = R[:, j]; good = np.flatnonzero(~np.isnan(col))
                if len(good) < 2: continue
                gaps = np.flatnonzero(np.diff(good) > 1)
                for g in gaps:
                    a0, a1 = good[g], good[g + 1]
                    if a1 - a0 - 1 <= lim: col[a0 + 1:a1] = np.interp(np.arange(a0 + 1, a1), [a0, a1], [col[a0], col[a1]])
            R = np.moveaxis(R, 0, axis)
        m = ~np.isnan(R)
        # the envelope over the ribs (not dipping into each space), then smoothed
        R = np.where(m, ndimage.maximum_filter(np.where(m, R, -1e9), size=(9, 3)), np.nan); Rf = np.where(m, R, 0.0)
        w = ndimage.gaussian_filter(m.astype(float), sigma); v = ndimage.gaussian_filter(Rf, sigma)
        out = np.where(m, v / np.maximum(w, 1e-6), np.nan)
        return out

    def point(self, z, a, r):
        t = np.radians(a)
        return np.stack([self.c[0] + self.sg * r * np.cos(t), self.c[1] + r * np.sin(t), z], axis=-1)

    def sheet(self, region, radius) -> trimesh.Trimesh | None:
        """a surface over the grid cells where region(z, az) holds, at radius(z, az)"""
        Z, A = np.meshgrid(self.zg, self.ag, indexing='ij')
        R = radius(Z, A); G = region(Z, A); valid = ~np.isnan(R)
        if not (G & valid).any(): return None
        M = G & ndimage.binary_fill_holes(G & valid)
        # radius in the small unsampled holes: nearest sampled value
        ii = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True); R = R[tuple(ii)]
        R = ndimage.gaussian_filter(R, 1.0)
        if M.sum() < 20: return None
        P = self.point(Z, A, R)
        idx = -np.ones(M.shape, int); idx[M] = np.arange(M.sum())
        V = P[M]
        f = []
        for i in range(M.shape[0] - 1):
            for j in range(M.shape[1] - 1):
                a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
                if min(a, b, c, d) >= 0: f += [[a, b, d], [a, d, c]]
        m = trimesh.Trimesh(V, np.array(f), process=True)
        m.remove_unreferenced_vertices()
        return m

    def on_wall(self, p):
        """move a skin point that landed on an arm (far outside the ribs) back onto the chest wall"""
        a = self.az(p); r = np.hypot(p[0] - self.c[0], p[1] - self.c[1])
        iz = int(np.clip(round((p[2] - self.zg[0]) / ZS), 0, len(self.zg) - 1)); ia = int(np.clip(round((a - self.ag[0]) / AS), 0, len(self.ag) - 1))
        rc = self.cage[iz, ia]
        if np.isnan(rc):
            col = self.cage[:, ia]; ok = np.flatnonzero(~np.isnan(col))
            if not len(ok): return p
            rc = col[ok[np.argmin(np.abs(ok - iz))]]
        return self.point(p[2], a, rc + 18.0) if r > rc + 40 else p

    def cage_at(self, z, a):
        iz = int(np.clip(round((z - self.zg[0]) / ZS), 0, len(self.zg) - 1)); ia = int(np.clip(round((a - self.ag[0]) / AS), 0, len(self.ag) - 1))
        rc = self.cage[iz, ia]
        if np.isnan(rc):
            col = self.cage[:, ia]; ok = np.flatnonzero(~np.isnan(col))
            rc = col[ok[np.argmin(np.abs(ok - iz))]] if len(ok) else np.nan
        return rc

    def skin(self, z, a):
        """the outer skin at height z and azimuth a: the outermost skin point within 60 mm of the ribs (an arm lying
        against the chest is further out than that), else the old estimate"""
        sk = self.ctx['skin_mm']; s = sk[np.abs(sk[:, 2] - z) < 2.5]
        if len(s):
            s = s[np.abs(self.az(s) - a) < 3.0] if len(s) > 1 else s
            rc = self.cage_at(z, a)
            if len(s) and np.isfinite(rc):
                r = np.hypot(s[:, 0] - self.c[0], s[:, 1] - self.c[1]); ok = r <= rc + 60
                if ok.any(): return self.point(z, a, float(r[ok].max()))
        return self.on_wall(self.ctx['skin_at'](z, a, self.side))

    def port(self, ics, a):
        z1, z2 = self.ctx['rib_z'](ics, a, self.side), self.ctx['rib_z'](ics + 1, a, self.side)
        return self.skin((z1 + z2) / 2, a) if z1 and z2 else self.on_wall(self.ctx['port'](ics, a, self.side))

    def lift(self, p, mm=2.5):
        """move surface points out from the skin a little, so a drawing on the skin is not buried in it"""
        p = np.array(p, float); one = p.ndim == 1; p = np.atleast_2d(p)
        r = p[:, :2] - self.c[:2]; p[:, :2] += r / np.linalg.norm(r, axis=1, keepdims=True) * mm
        return p[0] if one else p

    def zr(self, n, a=0.0):
        z = self.ctx['rib_z'](n, a, self.side)
        return z if z is not None else self.rz.get(n) or self.c[2]


def build(ctx):
    emit, emit_mesh, W, tube, sphere = ctx['emit'], ctx['emit_mesh'], ctx['W'], ctx['tube'], ctx['sphere']
    ts, AT = ctx['ts'], ctx['AT']
    ids_l, ids_r, LM = set(), set(), {}
    print('== chest wall')
    for s in ('left', 'right'):
        k = s[0]
        for nm_, lab, col, name in (('scapula', f'scapula_{s}', '#e9dec6', 'Scapula'), ('clavicle', f'clavicula_{s}', '#e9dec6', 'Clavicle'),
                                     ('humerus', f'humerus_{s}', '#e9dec6', 'Humerus (head)'), ('erector', f'autochthon_{s}', MUSCLE, 'Erector spinae')):
            if ctx['has'](lab):
                emit(f'{nm_}-{k}', f'{name}, {s}', 'chest-wall', col, ts(lab), AT, faces=6000, visible=False, label=True)
                (ids_l if k == 'l' else ids_r).add(f'{nm_}-{k}')
    if ctx['has']('costal_cartilages'):
        emit('cartilages', 'Costal cartilages', 'chest-wall', '#dfe6e0', ts('costal_cartilages'), AT, faces=8000, visible=False, label=True)

    for s in ('left', 'right'):
        H = Side(ctx, s); k = s[0]; ids = ids_l if k == 'l' else ids_r
        Z3, Z7, Z9 = H.zr(3), H.zr(7), H.zr(9)
        T7 = H.vz.get('T7', H.tip[2] - 10); T12 = H.vz.get('T12', H.z0); T5 = H.vz.get('T5', H.tip[2] + 60); T1 = H.vz.get('T1', H.scap_top)
        cage = lambda Zg, Ag: H.cage[np.clip(np.round((Zg - H.zg[0]) / ZS).astype(int), 0, len(H.zg) - 1), np.clip(np.round((Ag - H.ag[0]) / AS).astype(int), 0, len(H.ag) - 1)]
        look = lambda Mp: (lambda Zg, Ag: Mp[np.clip(np.round((Zg - H.zg[0]) / ZS).astype(int), 0, len(H.zg) - 1), np.clip(np.round((Ag - H.ag[0]) / AS).astype(int), 0, len(H.ag) - 1)])
        scap, erec = look(H.scap), look(H.erec)
        over = lambda Zg, Ag, d, *deep: np.fmax.reduce([cage(Zg, Ag) + d] + [np.nan_to_num(f(Zg, Ag), nan=-1) + dd for f, dd in deep])

        # serratus anterior: ribs 1-8 anterolaterally (digitations), deep to the scapula, to its medial border
        saw = lambda Zg: 7 * np.sign(np.sin(2 * np.pi * (Zg - H.zr(1)) / 22.0))
        sa_low = lambda Ag: np.interp(Ag, [H.scap_med, min(H.tip_az, -45.0), 0, 55], [H.tip[2] + 25, H.tip[2] - 5, H.zr(8), H.zr(6, 50)])
        sa = H.sheet(lambda Zg, Ag: (Ag >= H.scap_med) & (Ag <= 52 + saw(Zg)) & (Zg <= H.zr(1) - 4) & (Zg >= sa_low(Ag)),
                     lambda Zg, Ag: cage(Zg, Ag) + 4.0)
        # latissimus dorsi: from T7-T12 and the lower ribs, over the scapular tip, to the axilla; its free anterior border is the posterior axillary fold
        tip_a = min(H.tip_az, -45.0)
        ld_top = lambda Ag: np.interp(Ag, [H.spine_az, tip_a, -18], [T7, H.tip[2] + 18, Z3])
        ld_front = lambda Zg: np.interp(Zg, [H.zr(10), Z3], [4, -18])
        ld = H.sheet(lambda Zg, Ag: (Ag >= H.spine_az + 3) & (Ag <= ld_front(Zg)) & (Zg <= ld_top(Ag)) & (Zg >= max(T12, H.z0 + 6)),
                     lambda Zg, Ag: over(Zg, Ag, 12.0, (scap, 4.0), (erec, 4.0)))
        # trapezius (middle and lower fibres): outermost, from the spinous processes to the spine of the scapula
        tr_lat = lambda Zg: np.interp(Zg, [T12, H.tip[2], H.scap_top], [H.spine_az + 6, H.spine_az + 26, H.scap_med + 25])
        tr = H.sheet(lambda Zg, Ag: (Ag >= H.spine_az + 1) & (Ag <= tr_lat(Zg)) & (Zg >= T12),
                     lambda Zg, Ag: over(Zg, Ag, 18.0, (scap, 9.0), (erec, 9.0)))
        # rhomboids: spinous processes C7-T5 to the medial border of the scapula, deep to trapezius
        rh = H.sheet(lambda Zg, Ag: (Ag >= H.spine_az + 2) & (Ag <= H.scap_med + 6) & (Zg >= T5) & (Zg <= T1),
                     lambda Zg, Ag: over(Zg, Ag, 10.0, (erec, 5.0)))
        # pectoralis major: clavicle and sternum to the humerus; lateral border is the anterior axillary fold
        pm_lat = lambda Zg: np.interp(Zg, [H.zr(6, 70), Z3], [62, 28])
        pm = H.sheet(lambda Zg, Ag: (Ag >= pm_lat(Zg)) & (Ag <= 128) & (Zg >= H.zr(6, 70)),
                     lambda Zg, Ag: cage(Zg, Ag) + 12.0)
        # intercostal muscles: flush with the ribs, over the lateral chest wall
        ic = H.sheet(lambda Zg, Ag: (Ag >= -125) & (Ag <= 95) & (Zg >= Z9) & (Zg <= H.zr(2)),
                     lambda Zg, Ag: cage(Zg, Ag) - 1.5)
        side_name = s
        SHADE = {'serratus': '#cf6f63', 'latdorsi': '#8f3129', 'trapezius': '#7d3346', 'rhomboid': '#a3584c', 'pecmajor': '#b8483c', 'intercostal': '#5e2323'}
        for id_, name, m, op, note in (
                (f'mus-serratus-{k}', f'Serratus anterior, {side_name}', sa, 0.9, 'Schematic, on this patient\'s ribs. Digitations from ribs 1 to 8, passing deep to the scapula to its medial border. Spared in a muscle-sparing thoracotomy; the long thoracic nerve runs on its surface.'),
                (f'mus-latdorsi-{k}', f'Latissimus dorsi, {side_name}', ld, 0.9, 'Schematic. From T7-T12, the thoracolumbar fascia and the lower ribs, over the scapular tip to the humerus. Divided in a standard posterolateral thoracotomy; retracted in a muscle-sparing one.'),
                (f'mus-trapezius-{k}', f'Trapezius (lower fibres), {side_name}', tr, 0.85, 'Schematic. Divided only when a posterolateral incision is carried up between the scapula and the spine.'),
                (f'mus-rhomboid-{k}', f'Rhomboids, {side_name}', rh, 0.85, 'Schematic. Deep to trapezius, spine to the medial border of the scapula.'),
                (f'mus-pecmajor-{k}', f'Pectoralis major, {side_name}', pm, 0.85, 'Schematic. Its lateral border is the anterior axillary fold; anterolateral incisions and anterior ports pass at or below it.'),
                (f'mus-intercostal-{k}', f'Intercostal muscles, {side_name}', ic, 0.75, 'Schematic. External, internal and innermost layers between the ribs; divided on the upper border of the lower rib to spare the neurovascular bundle.')):
            if m is None: print('  skip', id_); continue
            m.apply_translation(-np.asarray(ctx['CARINA'], float))
            emit_mesh(id_, name, 'muscles', SHADE.get(id_.split('-')[1], MUSCLE), m, opacity=op, visible=False, note=note); ids.add(id_)
        # nerves: long thoracic on serratus in the mid-axillary line; thoracodorsal on the deep surface of latissimus near its border
        zl = np.arange(H.zr(8), H.zr(1) - 10, 6.0)
        lt = [H.point(z, -4.0, cage(np.array(z), np.array(-4.0)) + 5.5) for z in zl]
        td = [H.point(z, float(np.interp(z, [H.zr(9), Z3], [-4, -24])), cage(np.array(z), np.array(np.interp(z, [H.zr(9), Z3], [-4, -24]))) + 10.0) for z in np.arange(H.zr(9), Z3, 6.0)]
        for id_, name, pts, note in ((f'n-longthoracic-{k}', f'Long thoracic nerve, {s}', lt, 'Schematic. On the outer surface of serratus anterior in the mid-axillary line: injury gives a winged scapula.'),
                                     (f'n-thoracodorsal-{k}', f'Thoracodorsal nerve and vessels, {s}', td, 'Schematic. On the deep surface of latissimus dorsi near its anterior border: kept in a muscle-sparing thoracotomy.')):
            pts = [p for p in pts if not np.isnan(p).any()]
            if len(pts) > 3: emit_mesh(id_, name, 'nerves', NERVE, tube([W(p) for p in pts], 1.3), visible=False, note=note); ids.add(id_)
        # surface lines on the skin: anterior, mid and posterior axillary
        for id_, name, a in ((f'line-aal-{k}', 'Anterior axillary line', 35.0), (f'line-mal-{k}', 'Mid-axillary line', 0.0), (f'line-pal-{k}', 'Posterior axillary line', -30.0)):
            pts = [H.skin(z, a) for z in np.arange(H.zr(10, a) if H.zr(10, a) else Z9, Z3 + 20, 8.0)]
            pts = np.array(pts); pts[1:-1] = (pts[:-2] + 2 * pts[1:-1] + pts[2:]) / 4
            radial = pts[:, :2] - H.c[:2]; radial /= np.linalg.norm(radial, axis=1, keepdims=True); pts[:, :2] += radial * 2.5   # drawn on, not in, the skin
            emit_mesh(id_, f'{name}, {s}', 'landmarks', LINE, tube([W(p) for p in pts], 1.6), visible=False, note='Surface landmark, drawn on the skin.'); ids.add(id_)
        tip_skin = H.lift(H.skin(H.tip[2], H.tip_az), 3.0)
        emit_mesh(f'lm-scaptip-{k}', f'Tip of the scapula, {s}', 'landmarks', LINE, sphere(W(tip_skin), 5.0), visible=False,
                  note='Inferior angle of the scapula: about the 7th rib / 7th space with the arm by the side. The posterolateral incision passes 2-3 cm below it.'); ids.add(f'lm-scaptip-{k}')
        nip = H.lift(H.port(4, 72), 2.0)
        emit_mesh(f'lm-nipple-{k}', f'Nipple (4th space, male), {s}', 'landmarks', LINE, sphere(W(nip), 3.5), visible=False,
                  note='In a man the nipple lies over the 4th intercostal space in the mid-clavicular line; not a reliable landmark in a woman.'); ids.add(f'lm-nipple-{k}')
        LM[f'scaptip-{k}'] = tip_skin; LM[f'nipple-{k}'] = nip
        H_ = H
        # VATS: uniportal (4th space for upper lobes, 5th for lower) and biportal incisions, as short cuts along the space
        def cut(ics, a0, length, id_, name, note):
            span = np.degrees(length / max(1.0, float(np.nanmedian(H_.cage))))
            pts = np.array([H_.port(ics, a) for a in np.linspace(a0 - span / 2, a0 + span / 2, 5)])
            pts[1:-1] = (pts[:-2] + 2 * pts[1:-1] + pts[2:]) / 4; pts = H_.lift(pts, 2.5)
            emit_mesh(id_, name, 'ports-vats', '#d0433a', tube([W(p) for p in pts], 2.2), visible=False, note=note); ids.add(id_)
            LM[id_] = pts[len(pts) // 2]
        cut(4, 15.0, 40.0, f'uni-4-{k}', 'Uniportal incision, 4th space', 'Single 3-4 cm incision in the 4th space between the anterior and mid-axillary lines: upper lobes. Camera at the back of the wound, instruments below it.')
        cut(5, 15.0, 40.0, f'uni-5-{k}', 'Uniportal incision, 5th space', 'Single 3-4 cm incision in the 5th space between the anterior and mid-axillary lines: middle and lower lobes.')
        cut(4, 32.0, 35.0, f'bi-utility-{k}', 'Biportal utility incision, 4th space', 'Two-port VATS: a 3-4 cm utility incision in the 4th (upper lobes) or 5th space at the anterior axillary line.')
        LM[f'bi-camera-{k}'] = H.lift(H.port(7, -5), 2.0)
        emit_mesh(f'bi-camera-{k}', 'Biportal camera port, 7th space', 'ports-vats', '#46c2c7', sphere(W(LM[f'bi-camera-{k}']), 3.8), visible=False,
                  note='Two-port VATS: 1 cm camera port in the 7th (or 8th) space in the mid- to posterior axillary line.'); ids.add(f'bi-camera-{k}')
        # muscle-sparing lateral thoracotomy: vertical, along the anterior border of latissimus
        ms = np.array([H.skin(z, float(np.interp(z, [H.zr(7), Z3], [-10, -20]))) for z in np.arange(H.zr(7, -15), Z3 + 5, 8.0)])
        ms[1:-1] = (ms[:-2] + 2 * ms[1:-1] + ms[2:]) / 4; ms = H.lift(ms, 2.5)
        emit_mesh(f'incision-ms-{k}', 'Muscle-sparing lateral thoracotomy incision', 'ports-open-' + s, '#d0433a', tube([W(p) for p in ms], 1.8), visible=False,
                  note='Schematic: vertical, 10-12 cm, along the anterior border of latissimus dorsi, from the axilla down; flaps raised, latissimus retracted back, serratus retracted forward or split.'); ids.add(f'incision-ms-{k}')
        LM[f'msthor-{k}'] = ms[len(ms) // 2]
        print(f'  {s}: scapular tip z {W(H.tip)[2]:.0f}, az {H.tip_az:.0f}; medial border az {H.scap_med:.0f}; midline az {H.spine_az:.0f}')
    return ids_l, ids_r, LM
