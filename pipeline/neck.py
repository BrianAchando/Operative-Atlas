"""The cervical airway for tracheal resection.

From the CT: the cervical trachea's centreline and calibre (TotalSegmentator `trachea`), the thyroid, the carotids.
Schematic, on those landmarks: the trachea above the scan as far as the cricoid, the larynx outline (cricoid ring and
thyroid cartilage), a post-intubation stenosis 2-4 cm below the cricoid, the tracheal segments either side of it,
the lateral segmental blood supply, the recurrent laryngeal nerves in the tracheo-oesophageal grooves (the right looping
under the subclavian), the external and internal branches of the superior laryngeal nerves, the platysma, the strap
muscles, a collar incision, lateral stay sutures, and the cross-field and orotracheal tubes.
"""
from __future__ import annotations

import numpy as np
import trimesh

RIGHT, ANT, SUP = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
CART = '#e6dcc6'
NERVE = '#f2d24b'


def build(ctx):
    ts, vm, emit, emit_mesh, W, tube, sphere, AT, has = ctx['ts'], ctx['vox_mm'], ctx['emit'], ctx['emit_mesh'], ctx['W'], ctx['tube'], ctx['sphere'], ctx['AT'], ctx['has']
    LM = {}
    print('== neck (tracheal resection)')
    tr = vm(ts('trachea')); st = vm(ts('sternum')); eso = vm(ts('esophagus'))
    z_notch = float(st[:, 2].max()); z_scan = float(tr[:, 2].max())
    # centreline and radius of the trachea, from 3 cm below the notch to the top of the scan
    cl = []
    for z in np.arange(z_notch - 30, z_scan - 1, 3.0):
        s = tr[np.abs(tr[:, 2] - z) < 1.5]
        if len(s) < 20: continue
        c = s.mean(0); r = float(np.sqrt(np.ptp(s[:, 0]) * np.ptp(s[:, 1])) / 2)
        cl.append((c, r))
    C = np.array([c for c, _ in cl]); Rr = np.array([r for _, r in cl])
    r_tr = float(np.median(Rr)) + 1.5                                      # outer radius, with the wall
    top = C[-1]
    # carry the airway up to the cricoid (just above the scan here): the cricoid's lower border 12 mm above the last slice
    z_cric = top[2] + 12
    ext = [top + SUP * d for d in (4, 8, 12)]
    path = np.vstack([C, ext])
    axis = lambda z: (lambda i: (path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]) / np.linalg.norm(path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]))(int(np.argmin(np.abs(path[:, 2] - z))))
    at = lambda z: path[int(np.argmin(np.abs(path[:, 2] - z)))]
    # stenosis: 2.0 cm long, its upper edge 2.0 cm below the cricoid (a cuff-level post-intubation stricture)
    z_up = z_cric - 20.0; z_lo = z_up - 20.0
    L = z_up - z_lo
    seg = lambda z0, z1: [p for p in path if z0 <= p[2] <= z1]
    rings = lambda P, r: tube([W(p) for p in P], r, seg=28) if len(P) > 1 else None
    prox = seg(z_up + 1.5, z_cric)
    dist_ = seg(C[0][2], z_lo - 1.5)
    sten = seg(z_lo - 1.0, z_up + 1.0)
    # the stricture: an hourglass waist of scar
    sm = []
    for i, p in enumerate(sten):
        t = (p[2] - z_lo) / L; rr = r_tr * (1 - 0.28 * np.sin(np.pi * np.clip(t, 0, 1)))
        ring = trimesh.creation.annulus(r_min=max(1.0, rr - 2.2), r_max=rr, height=3.2, sections=32); ring.apply_translation(W(p)); sm.append(ring)
    emit_mesh('trach-steno', 'Post-intubation tracheal stenosis', 'airway', '#b98a82', trimesh.util.concatenate(sm), visible=False,
              note='Teaching example: a 2 cm cuff-level stricture, 2 cm below the cricoid. Resected as a sleeve.')
    # cartilage rings, every 4 mm, drawn with each segment so they move with it (C-shaped: the membranous wall behind has none)
    def with_rings(P, z0, z1):
        parts = [rings(P, r_tr)]
        for z in np.arange(z0, z1, 4.0):
            ring = trimesh.creation.torus(major_radius=r_tr + 0.2, minor_radius=0.9, major_sections=40, minor_sections=6)
            v = ring.vertices; keep = np.arctan2(v[:, 1], v[:, 0]) > -2.2
            ring = trimesh.Trimesh(v, ring.faces[keep[ring.faces].all(1)], process=True); ring.apply_translation(W(at(z))); parts.append(ring)
        return trimesh.util.concatenate([q for q in parts if q is not None])
    emit_mesh('trach-prox', 'Trachea above the stenosis (to the cricoid)', 'airway', CART, with_rings(prox, z_up + 3, z_cric), visible=False,
              note='Schematic from the CT centreline; carried up to the cricoid, just above this scan.')
    emit_mesh('trach-dist', 'Trachea below the stenosis', 'airway', CART, with_rings(dist_, C[0][2], z_lo - 2), visible=False, note='Schematic from the CT centreline.')
    # larynx outline: cricoid ring (a signet, high behind) and the thyroid cartilage laminae
    cc = at(z_cric) + SUP * 4
    import anatomy_neck
    base = W(cc) - SUP * 3.5
    cric = anatomy_neck.cricoid(base, r_tr - 1.0, r_tr + 2.2)
    emit_mesh('cricoid', 'Cricoid cartilage', 'airway', CART, cric, visible=False,
              note='Schematic, above this scan: a signet ring, low arch in front and tall lamina behind; the only complete ring of the airway, so a stricture here is the hardest to resect.')
    # thyroid cartilage: two laminae meeting in front at about 90 degrees (the prominence), above the cricothyroid membrane
    shield_mesh = anatomy_neck.thyroid(base + SUP * 15 + ANT * (r_tr + 6))
    emit_mesh('thyroid-cart', 'Thyroid cartilage', 'airway', CART, shield_mesh, visible=False, note='Schematic, above this scan: the two laminae meet in front at the laryngeal prominence; the superior horns reach up to the hyoid and the inferior horns articulate with the cricoid (the recurrent laryngeal nerve enters the larynx just behind that joint).')
    LM['cricoid'] = cc; LM['stenosis'] = at((z_up + z_lo) / 2); LM['cut-up'] = at(z_up); LM['cut-lo'] = at(z_lo)
    ctx['dirs']['trach-axis'] = axis((z_up + z_lo) / 2)
    ctx['scalars']['trach-radius'] = r_tr; ctx['scalars']['stenosis-length'] = L
    # lateral segmental vessels (from the inferior thyroid artery), entering the sides of the trachea
    ves = []
    for z in np.arange(z_lo - 30, z_cric - 2, 9.0):
        for sx in (-1, 1):
            p = at(z); a = p + RIGHT * sx * (r_tr + 1) - ANT * 2; b = a + RIGHT * sx * 14 - ANT * 4 + SUP * 3
            ves.append(tube([W(b), W((a + b) / 2 + SUP * 1.5), W(a)], 0.7))
    emit_mesh('trach-vessels', 'Tracheal blood supply (lateral, segmental)', 'airway', '#c24a3e', trimesh.util.concatenate(ves), visible=False,
              note='Schematic: small segmental branches (mostly from the inferior thyroid artery) enter the trachea laterally. Free the trachea circumferentially only for about 1 cm beyond each cut.')
    # recurrent laryngeal nerves in the tracheo-oesophageal grooves, entering the larynx behind the cricothyroid joint
    groove = lambda z, sx: at(z) + RIGHT * sx * (r_tr + 1.5) - ANT * (r_tr * 0.55)
    zs = np.arange(z_notch - 25, z_cric + 5, 6.0)
    left = [groove(z, -1) for z in zs]
    emit_mesh('n-rln-neck-l', 'Left recurrent laryngeal nerve (neck)', 'nerves', NERVE, tube([W(p) for p in left], 1.3), visible=False,
              note='Schematic: up from under the aortic arch in the left tracheo-oesophageal groove, entering the larynx behind the cricothyroid joint.')
    right = [groove(z, 1) for z in zs]
    if has('subclavian_artery_right'):
        sa = vm(ts('subclavian_artery_right')); s0 = sa[np.argmin(np.abs(sa[:, 0] - (at(z_notch)[0] + 25)) + 0.2 * np.abs(sa[:, 2] - z_notch))]
        right = [s0 - SUP * 8 + ANT * 2, s0 - SUP * 6 - ANT * 8, *right[2:]]
    emit_mesh('n-rln-neck-r', 'Right recurrent laryngeal nerve (neck)', 'nerves', NERVE, tube([W(p) for p in right], 1.3), visible=False,
              note='Schematic: loops under the right subclavian artery and runs up obliquely toward the right tracheo-oesophageal groove; more lateral low in the neck than the left.')
    # superior laryngeal nerves: external branch down on the inferior constrictor to cricothyroid, near the upper thyroid pole; internal through the thyrohyoid membrane
    for sx, k in ((-1, 'l'), (1, 'r')):
        ext_ = [cc + SUP * 55 + RIGHT * sx * 28 - ANT * 4, cc + SUP * 35 + RIGHT * sx * (r_tr + 10) - ANT * 2, cc + SUP * 12 + RIGHT * sx * (r_tr + 5) + ANT * 4, cc + SUP * 3 + RIGHT * sx * (r_tr - 2) + ANT * 7]
        emit_mesh(f'n-sln-ext-{k}', f'External branch, superior laryngeal nerve ({"left" if k == "l" else "right"})', 'nerves', NERVE, tube([W(p) for p in ext_], 0.9), visible=False,
                  note='Schematic: close to the superior thyroid artery near the upper pole, down on the inferior constrictor to cricothyroid. Injury: a weak, low-pitched voice that tires.')
        intl = [cc + SUP * 60 + RIGHT * sx * 30 - ANT * 6, cc + SUP * 52 + RIGHT * sx * 20, cc + SUP * 48 + RIGHT * sx * 12 + ANT * 4]
        emit_mesh(f'n-sln-int-{k}', f'Internal branch, superior laryngeal nerve ({"left" if k == "l" else "right"})', 'nerves', NERVE, tube([W(p) for p in intl], 1.0), visible=False,
                  note='Schematic: pierces the thyrohyoid membrane (sensation above the cords). At risk in a thyrohyoid laryngeal release: aspiration.')
    # platysma and strap muscles, from the skin and the airway
    sk = ctx['skin_mm']; x_mid = float(at(z_notch)[0])
    def skin_front(z, x):
        s = sk[(np.abs(sk[:, 2] - z) < 2.5) & (np.abs(sk[:, 0] - x) < 3)]
        return float(s[:, 1].max()) if len(s) else None
    grid_z = np.arange(z_notch - 2, z_scan - 1, 3.0); grid_x = np.arange(-58, 58.1, 4.0)
    V = []; I = -np.ones((len(grid_z), len(grid_x)), int)
    for i, z in enumerate(grid_z):
        for j, dx in enumerate(grid_x):
            y = skin_front(z, x_mid + dx)
            if y is not None and y > at(z)[1] + 8: I[i, j] = len(V); V.append([x_mid + dx, y - 3.5, z])
    F = []
    for i in range(len(grid_z) - 1):
        for j in range(len(grid_x) - 1):
            a, b, c, d = I[i, j], I[i, j + 1], I[i + 1, j], I[i + 1, j + 1]
            if min(a, b, c, d) >= 0: F += [[a, b, d], [a, d, c]]
    if F:
        pm = trimesh.Trimesh(np.array(V) - ctx['CARINA'], np.array(F), process=True)
        emit_mesh('platysma', 'Platysma', 'muscles', '#b05a4c', pm, opacity=0.85, visible=False, note='Schematic: raised with the skin as subplatysmal flaps, up to the thyroid cartilage and down to the notch.')
    thy = vm(ts('thyroid_gland')) if has('thyroid_gland') else np.zeros((0, 3))
    for sx, k in ((-1, 'l'), (1, 'r')):
        V = []; F = []
        zz = np.arange(z_notch - 4, z_cric + 30, 3.0); xx = np.arange(2.5, 27.0, 3.0)
        I = -np.ones((len(zz), len(xx)), int)
        for i, z in enumerate(zz):
            for j, dx in enumerate(xx):
                x = x_mid + sx * dx; th = thy[(np.abs(thy[:, 2] - min(z, z_scan - 2)) < 2.5) & (np.abs(thy[:, 0] - x) < 2.5)]
                base = max(at(min(z, z_cric))[1] + r_tr, float(th[:, 1].max()) if len(th) else -1e9)
                I[i, j] = len(V); V.append([x, base + 5.0, z])
        for i in range(len(zz) - 1):
            for j in range(len(xx) - 1):
                a, b, c, d = I[i, j], I[i, j + 1], I[i + 1, j], I[i + 1, j + 1]; F += [[a, b, d], [a, d, c]]
        emit_mesh(f'straps-{k}', f'Strap muscles ({"left" if k == "l" else "right"}: sternohyoid, sternothyroid)', 'muscles', '#a8423a',
                  trimesh.Trimesh(np.array(V) - ctx['CARINA'], np.array(F), process=True), opacity=0.9, visible=False,
                  note='Schematic: separated in the midline (the avascular linea alba of the neck) and retracted laterally.')
    if has('common_carotid_artery_right'):
        emit('rcca', 'Right common carotid artery', 'mediastinum', '#d0433a', ts('common_carotid_artery_right'), AT, faces=4000, visible=False, label=True)
    # collar incision: two fingerbreadths above the notch, across the anterior neck
    zc = z_notch + 20
    col = [np.array([x_mid + dx, skin_front(zc + 0.08 * dx * dx / 10, x_mid + dx) + 2.5, zc + 0.08 * dx * dx / 10]) for dx in np.arange(-45, 46, 9) if skin_front(zc + 0.08 * dx * dx / 10, x_mid + dx) is not None]
    if len(col) > 3:
        emit_mesh('incision-collar', 'Collar incision', 'incisions', '#d0433a', tube([W(p) for p in col], 1.8), visible=False, note='Low collar incision, two fingerbreadths above the sternal notch; an old stoma is included in it.')
        LM['collar'] = col[len(col) // 2]
    LM['neck-front'] = at(z_up) + ANT * 140
    # stay sutures (lateral, through the full wall a ring beyond each cut) and the tubes
    stays = []
    for z in (z_up + 6, z_lo - 6):
        for sx in (-1, 1):
            p = at(z) + RIGHT * sx * (r_tr + 0.5)
            loop = [p + ANT * 2.5 * np.cos(t) + SUP * 2.5 * np.sin(t) for t in np.linspace(0, 2 * np.pi, 14)]
            tail = [p + ANT * 2.5, p + ANT * 20 + RIGHT * sx * 6, p + ANT * 55 + RIGHT * sx * 16]
            stays += [tube([W(q) for q in loop], 0.5), tube([W(q) for q in tail], 0.5)]
    emit_mesh('stay-sutures', 'Lateral stay sutures (2-0)', 'airway', '#2d3fa0', trimesh.util.concatenate(stays), visible=False,
              note='Traction sutures through the full wall laterally, a ring beyond each cut: they hold the ends, test the tension, and are crossed and held while the anastomosis is tied.')
    lo = at(z_lo)
    xf = [lo + ANT * 120 + SUP * 25, lo + ANT * 60 + SUP * 8, lo + ANT * 12, lo - SUP * 12, lo - SUP * 55]
    emit_mesh('ett-crossfield', 'Cross-field endotracheal tube', 'airway', '#8fc8e8', tube([W(p) for p in xf], 4.2), opacity=0.8, visible=False,
              note='A sterile armoured tube passed into the distal trachea across the field and connected to a sterile circuit; out for short periods while the sutures are placed.')
    ot = [at(z_cric) + SUP * 120, at(z_cric) + SUP * 40, at(z_cric), at(z_up), *[at(z) for z in np.arange(z_up - 2, z_lo - 45, -6)]]
    emit_mesh('ett-oral', 'Orotracheal tube (advanced across the anastomosis)', 'airway', '#8fc8e8', tube([W(p) for p in ot], 4.2), opacity=0.6, visible=False,
              note='Advanced from above past the anastomosis before the anterior sutures are tied.')
    LM['anast'] = at(z_up)
    return LM
