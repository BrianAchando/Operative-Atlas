"""Write public/data/procedures.json: VATS left upper lobectomy by an anterior (hilum-first) and a posterior
(fissure-first, artery-first) approach. Cameras and CT positions are computed from atlas.json, so every view sits on
the reference anatomy; the text is a teaching summary to be reviewed by a thoracic surgeon.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / 'public' / 'data'
atlas = json.loads((OUT / 'atlas.json').read_text())
S = {s['id']: s for s in atlas['structures']}
LM = atlas['landmarks']
has = lambda i: i in S
POST = [i for i in ('pa-posterior-1', 'pa-posterior-2', 'pa-posterior-3') if has(i)]
LUL = ['lul', 'lul-arteries', 'lul-veins', 'lul-bronchi']


def pt(i):
    s = S[i]; return (s['division']['point'] if 'division' in s else s['centroid'])


def mean(ids):
    return list(np.mean([pt(i) for i in ids], axis=0).round(1))


# Surgeon's-eye directions once the lobe is retracted: lateral, tilted toward the side the approach works from.
VIEW = {'anterior': np.array([-1.0, 0.2, 0.22]), 'posterior': np.array([-1.0, -0.5, 0.18])}
V = lambda v: np.array(v, float)
ZOOM = 1.45                                     # camera distance factor: enough room to see the hilum whole
N = V(LM['fissure-normal'])                     # fissure plane normal, lower lobe -> upper lobe
R = lambda v: [round(float(x), 1) for x in v]


def scope(approach, target, back=0.0, lift=0.0, dist=120.0, side=None):
    """Look at the target from the approach's side, from outside the hilum (lung retracted and translucent)."""
    t = V(target); d = V(side) if side is not None else VIEW[approach]; d = d / np.linalg.norm(d)
    eye = t + d * (dist * ZOOM + back) + V([0, 0, lift])
    return {'eye': R(eye), 'target': R(t)}


def ct(i, plane='axial', window='mediastinum'):
    return {'focus': pt(i) if isinstance(i, str) else i, 'plane': plane, 'window': window}


def ask(q, right, why, *wrong):
    return {'question': q, 'choices': [{'text': right, 'correct': True, 'why': why}] + [{'text': w, 'correct': False, 'why': ''} for w in wrong]}


HILAR = ['pa-left', 'pa-truncus-anterior', *POST, 'pa-lingular', 'pa-a6', 'pa-basal-trunk', 'pv-superior', 'pv-inferior', 'br-lul', 'br-lll', 'br-left-main']
nodes = [i for i in ('ln-5', 'ln-6', 'ln-7', 'ln-10l', 'ln-11l') if has(i)]
ports = lambda a: [f'port-{a}-{k}' for k in ('utility', 'camera', 'posterior')]
CLEAR = {'lul': 0.35, 'lll': 0.35, 'heart': 0.45, 'laa': 0.5}

# the interlobar artery in the fissure, and the fissure plane through it
IA = np.mean([V(pt(i)) for i in ('pa-lingular', 'pa-a6', *POST[-1:])], axis=0)
FC = V(LM['fissure-centre'])
on_plane = lambda p: p - N * np.dot(p - FC, N)
AXIS = np.cross(N, [1.0, 0, 0]); AXIS /= np.linalg.norm(AXIS)   # along the fissure, toward anterior-inferior
LAT = V([-1.0, 0, 0])
SPREAD = [{'ids': ['lul', 'lul-arteries', 'lul-veins', 'lul-bronchi'], 'offset': R(N * 13)}, {'ids': ['lll', 'lll-arteries', 'lll-veins', 'lll-bronchi', 'fissure'], 'offset': R(-N * 9)}]
SPREAD_RETRACT = lambda op=0.35: [{'ids': sp['ids'], 'offset': sp['offset'], 'opacity': op} for sp in SPREAD]


def anatomy(approach):
    front = approach == 'anterior'
    return {
        'id': f'{approach[0]}-anatomy', 'phase': 'Anatomy', 'seq': 0,
        'title': 'The left hilum you are about to meet' if front else 'The hilum from behind and in the fissure',
        'body': ('<p>Look at the hilum from the front, as the anterior approach meets it. The <b>superior pulmonary vein</b> is the most anterior structure, '
                 'with the <b>truncus anterior</b> just above and behind it, the first branch off the top of the pulmonary artery.</p>'
                 '<p>Behind the vein lies the <b>upper lobe bronchus</b>. The <b>pulmonary artery</b> arches over the left main bronchus, then turns down behind the upper lobe bronchus into the fissure.</p>'
                 '<p>Order of division from the front: <b>vein → truncus → bronchus → remaining arteries → fissure</b>.</p>'
                 if front else
                 '<p>Seen from behind and from the fissure, the <b>pulmonary artery</b> is the key: it arches over the left main bronchus, then descends in the fissure, giving off its upper lobe branches one by one.</p>'
                 '<p>In the fissure: the <b>posterior segmental</b> artery up and back, the <b>lingular</b> artery forward, the <b>superior segmental (A6)</b> to the lower lobe behind, the <b>basal trunk</b> continuing down. '
                 'The <b>truncus anterior</b> is the first branch, high on the anterosuperior surface.</p>'
                 '<p>Order of division from behind: <b>fissure → posterior segmental → truncus → bronchus → vein</b>.</p>'),
        'view': {'frame': ['pa-left', 'pa-truncus-anterior', 'pv-superior', 'pv-inferior', 'br-lul', 'pa-basal-trunk'], 'dir': [-1, 0.4, 0.2] if front else [-1, -0.6, 0.25], 'pad': 0.95},
        'opacity': {**CLEAR, 'lul': 0.08, 'lll': 0.08, 'fissure': 0.1, 'heart': 0.35, 'lul-arteries': 0.3, 'lul-veins': 0.3, 'lul-bronchi': 0.3}, 'spin': True,
        'labels': ['pa-left', 'pa-truncus-anterior', 'pa-lingular', 'pa-a6', 'pv-superior', 'pv-inferior', 'br-lul'],
        'ct': ct('pa-truncus-anterior', 'coronal'),
    }


# ---------------------------------------------------------------------------------------------------- anterior
tA = V(pt('pa-truncus-anterior')); sv = V(pt('pv-superior')); iv = V(pt('pv-inferior'))
FIS_LINE = [on_plane(IA + LAT * 12 + AXIS * t) for t in (48, 18, -12, -36)]
anterior = [
    anatomy('anterior'),
    {'id': 'a-setup', 'phase': 'Setup', 'title': 'Position and ports',
     'body': '<p>Right lateral decubitus, table flexed to open the intercostal spaces, right lung ventilated through a double-lumen tube.</p>'
             '<p>Three ports: a 4–5 cm <b>utility incision anteriorly, directly over the hilum</b> and the superior vein, no rib spreading; a <b>low anterior camera port</b> at the level of the top of the diaphragm; and a working port at the same level further back.</p><p>The utility incision sits over the structures you will staple first.</p>',
     'view': {'frame': ['skin'], 'dir': [-1, 0.25, 0.15], 'pad': 1.05}, 'show': ['skin', *ports('anterior')], 'labels': ports('anterior'),
     'ct': ct(LM['port-anterior-utility'], 'axial', 'lung')},
    {'id': 'a-hilum', 'phase': 'Hilum', 'seq': 1, 'title': 'Open the pleura over the front of the hilum',
     'body': '<p>Retract the upper lobe posteriorly. With the <b>peanut</b>, sweep the mediastinal pleura off the front of the hilum, <b>behind the phrenic nerve</b>, from the top of the hilum down to the inferior vein.</p>'
             '<p>The <b>superior pulmonary vein</b> comes into view as the most anterior structure. Keep the phrenic nerve and its vessels on the pericardium.</p>',
     'view': scope('anterior', sv + V([0, 0, 8]), dist=115), 'retract': {'ids': LUL, 'offset': [-4, -16, 4], 'opacity': 0.3}, 'opacity': {'lll': 0.3, 'heart': 0.4, 'laa': 0.5},
     'highlight': ['pv-superior'], 'danger': ['n-phrenic'], 'labels': ['pa-truncus-anterior', 'pv-inferior'],
     'action': {'kind': 'dissect', 'label': 'Dissect with the peanut', 'port': 'port-anterior-utility',
                'path': [R(tA + V([-10, 14, 8])), R(sv + V([-9, 12, 8])), R(sv + V([-10, 11, -6])), R(iv + V([-10, 12, 4]))]},
     'ask': ask('Where does the left phrenic nerve run relative to the hilum?', 'Anterior to it, on the pericardium',
                'The phrenic nerve descends on the pericardium anterior to the hilum; the vagus passes behind it.', 'Posterior to it, with the vagus', 'Inside the fissure'),
     'ct': ct('pv-superior')},
    {'id': 'a-spv', 'phase': 'Vein', 'seq': 1, 'title': 'Superior pulmonary vein: staple',
     'body': '<p>Encircle the superior vein. Before any stapler goes round it, <b>see the inferior pulmonary vein</b> as a separate structure draining the lower lobe: a common trunk taken here drains the whole lung.</p>'
             '<p>The artery lies directly behind and above. Pass the vascular stapler from the working port, anvil tip in view, and fire.</p>',
     'view': scope('anterior', sv, dist=105), 'retract': {'ids': LUL, 'offset': [-4, -16, 4], 'opacity': 0.3}, 'highlight': ['pv-superior'], 'danger': ['pa-left', 'pv-inferior', 'n-phrenic'], 'opacity': {'heart': 0.4, 'laa': 0.5, 'lll': 0.1},
     'action': {'kind': 'staple', 'label': 'Fire the stapler', 'ids': ['pv-superior'], 'port': 'port-anterior-posterior', 'reload': 'vascular'},
     'ask': ask('Before stapling the superior vein, what must you confirm?', 'That the inferior pulmonary vein is separate and drains the lower lobe',
                'A common pulmonary vein (a variant) taken as the "superior vein" drains the whole lung.', 'That the lingular artery is already divided', 'That the fissure is complete'),
     'pearl': 'Staple far enough from the pericardium to leave a cuff, close enough to catch all the upper lobe tributaries.',
     'ct': ct('pv-superior')},
    {'id': 'a-truncus', 'phase': 'Artery', 'seq': 2, 'title': 'Truncus anterior: staple',
     'body': '<p>With the vein gone, the pulmonary artery comes into view behind it. Its first branch, the <b>truncus anterior</b>, leaves the anterosuperior surface for the apicoposterior and anterior segments. It is short, wide and thin-walled.</p>'
             '<p>Clear the <b>station 5 and 10</b> nodes to open the angle, then staple (or ligate) it. The recurrent laryngeal nerve hooks under the arch just above.</p>',
     'view': scope('anterior', tA, dist=100, lift=10), 'retract': {'ids': LUL, 'offset': [-4, -16, 4], 'opacity': 0.3}, 'opacity': {'heart': 0.4, 'laa': 0.5, 'lll': 0.1},
     'highlight': ['pa-truncus-anterior'], 'danger': ['pa-left', 'n-rln', 'aorta'], 'labels': ['ln-5', 'br-lul'],
     'action': {'kind': 'staple', 'label': 'Fire the stapler', 'ids': ['pa-truncus-anterior'], 'port': 'port-anterior-posterior', 'reload': 'vascular'},
     'ask': ask('Clearing station 5 above the truncus, which nerve is at risk?', 'Left recurrent laryngeal nerve',
                'It leaves the vagus at the arch and hooks under it beside the ligamentum arteriosum, in the AP window.', 'Left phrenic nerve', 'Thoracic duct'),
     'ct': ct('pa-truncus-anterior')},
    {'id': 'a-bronchus', 'phase': 'Bronchus', 'seq': 3, 'title': 'Upper lobe bronchus: clamp, inflate, staple',
     'body': '<p>The upper lobe bronchus now lies exposed behind the divided vein and below the artery. Sweep the station 10 and 11 nodes toward the specimen so the stapler sits on bronchus only.</p>'
             '<p>Close the stapler (thick-tissue reload) and ask for the lung to be inflated: <b>the lower lobe must ventilate</b>. Then fire, close to the origin.</p>',
     'view': scope('anterior', pt('br-lul'), dist=100), 'retract': {'ids': LUL, 'offset': [-4, -16, 4], 'opacity': 0.3}, 'opacity': {'heart': 0.4, 'laa': 0.5, 'lll': 0.1},
     'highlight': ['br-lul'], 'danger': ['br-lll', 'br-left-main', 'pa-left'], 'labels': ['ln-10l', 'ln-11l'],
     'action': {'kind': 'staple', 'label': 'Clamp, inflate, fire', 'ids': ['br-lul'], 'port': 'port-anterior-utility', 'reload': 'tissue'},
     'ask': ask('Before firing on the upper lobe bronchus, how do you know it is not the lower lobe bronchus?', 'Clamp, then inflate: the lower lobe must expand',
                'A test inflation with the stapler closed shows which lobe you have isolated.', 'By its diameter', 'It is always the more anterior bronchus'),
     'ct': ct('br-lul')},
    {'id': 'a-arteries', 'phase': 'Artery', 'seq': 4, 'title': 'Posterior segmental and lingular arteries',
     'body': '<p>Lift the bronchial stump forward: the artery runs down into the fissure. The remaining upper lobe branches leave it here, the <b>posterior segmental</b> above and the <b>lingular</b> lower down.</p>'
             '<p>Opposite the lingular artery, the <b>superior segmental artery of the lower lobe (A6)</b> leaves the back of the artery; the basal trunk continues below. Identify both before dividing anything.</p>',
     'view': scope('anterior', np.mean([V(pt(i)) for i in POST + ['pa-lingular']], axis=0), dist=110), 'retract': {'ids': LUL, 'offset': [-4, -16, 4], 'opacity': 0.3}, 'opacity': {'heart': 0.4, 'laa': 0.5, 'lll': 0.3},
     'highlight': POST + ['pa-lingular'], 'danger': ['pa-a6', 'pa-basal-trunk'],
     'action': {'kind': 'staple', 'label': 'Staple each artery', 'ids': POST + ['pa-lingular'], 'port': 'port-anterior-utility', 'reload': 'vascular'},
     'ask': ask('Which lower-lobe branch arises near the lingular artery and must be kept?', 'The superior segmental artery (A6)',
                'A6 leaves the posterior aspect of the interlobar artery, often opposite the lingular artery.', 'The basal trunk', 'The inferior pulmonary vein'),
     'ct': ct('pa-lingular')},
    {'id': 'a-fissure', 'phase': 'Fissure', 'seq': 5, 'title': 'Staple the fissure last, front to back',
     'body': '<p>Every hilar structure of the upper lobe is divided. Only parenchyma joins the lobes now. Staple the fissure with thick-tissue reloads <b>from front to back</b>, keeping the lower lobe artery below and behind the staple line.</p>',
     'view': scope('anterior', np.mean(FIS_LINE, axis=0), dist=150, side=[-1, 0.1, 0.35]), 'opacity': {'lul': 0.45, 'lll': 0.45, 'heart': 0.4, 'fissure': 0.2},
     'labels': ['fissure'], 'danger': ['pa-basal-trunk', 'pa-a6'],
     'action': {'kind': 'staple-fissure', 'label': 'Staple the fissure', 'port': 'port-anterior-utility', 'reload': 'tissue', 'normal': R(N), 'path': [R(p) for p in FIS_LINE], 'spread': SPREAD},
     'ct': ct('fissure', 'sagittal', 'lung')},
    {'id': 'a-specimen', 'phase': 'Close', 'seq': 6, 'title': 'Specimen out, nodes, leak test',
     'body': '<p>Bag the lobe and remove it through the utility incision. Complete the nodal dissection: <b>stations 5 and 6</b> in the AP window and along the aorta, <b>7</b> below the carina, <b>10 and 11</b> at the hilum.</p>'
             '<p>Fill the chest with saline and ventilate to test the bronchial stump and the fissure staple line.</p>',
     'view': {'frame': ['aorta', 'pa-left', 'br-left-main', 'pv-superior'], 'dir': [-0.7, 0.55, 0.45]}, 'opacity': {'heart': 0.4, 'lll': 0.2}, 'highlight': nodes, 'specimen': {'ids': LUL, 'offset': [-170, 20, -120]},
     'ct': ct(LM['ap-window'], 'coronal')},
]

# ---------------------------------------------------------------------------------------------------- posterior
ling = V(pt('pa-lingular')); a6 = V(pt('pa-a6'))
posterior = [
    anatomy('posterior'),
    {'id': 'p-setup', 'phase': 'Setup', 'title': 'Position and ports',
     'body': '<p>Right lateral decubitus, right lung ventilated. For a fissure-first approach the ports sit further back: a utility incision in the <b>5th intercostal space</b> in line with the fissure, a camera port low and posterior, and a posterior working port. Port sites vary between units.</p>',
     'view': {'frame': ['skin'], 'dir': [-1, -0.35, 0.15], 'pad': 1.05}, 'show': ['skin', *ports('posterior')], 'labels': ports('posterior'),
     'ct': ct(LM['port-posterior-utility'], 'axial', 'lung')},
    {'id': 'p-fissure', 'phase': 'Fissure', 'seq': 1, 'title': 'Open the fissure bluntly with a peanut',
     'body': '<p>Retract the upper lobe forward and up, the lower lobe back and down. Where the oblique fissure is deepest, open the visceral pleura and <b>dissect bluntly with the peanut</b>, sweeping along the line of the artery.</p>'
             '<p>The <b>interlobar pulmonary artery</b> appears in its sheath. Get onto the sheath: everything that follows is dissected on the artery.</p>',
     'view': scope('posterior', IA + LAT * 10, dist=140, side=[-1, -0.15, 0.55]), 'opacity': {**CLEAR, 'fissure': 0.2},
     'highlight': ['pa-left'], 'labels': ['fissure', 'pa-lingular', 'pa-a6', 'pa-basal-trunk'],
     'action': {'kind': 'open-fissure', 'label': 'Open the fissure', 'port': 'port-posterior-utility', 'spread': SPREAD,
                'path': [R(on_plane(IA + LAT * 38 + AXIS * 30)), R(on_plane(IA + LAT * 22 + AXIS * 10)), R(IA + LAT * 7 + AXIS * 2), R(IA + LAT * 6 - AXIS * 16)]},
     'ask': ask('A fissure-first approach works best when…', 'The fissure is complete and the artery is visible in it',
                'In a complete fissure the interlobar artery lies just under the visceral pleura where the fissures meet.', 'The fissure is fused', 'The superior vein is short'),
     'ct': ct('pa-lingular', 'sagittal', 'lung')},
    {'id': 'p-segmental', 'phase': 'Artery', 'seq': 2, 'title': 'First segmental arteries in the fissure',
     'body': '<p>Follow the artery up and back. The <b>posterior segmental artery</b> (A1+2c, sometimes two) is the first upper lobe branch you meet from the fissure, leaving the posterosuperior surface as the artery arches over the bronchus. The <b>lingular artery</b> runs forward into the lingula.</p>'
             '<p>Map the lower lobe branches first: <b>A6</b> behind, often opposite the lingular artery, and the <b>basal trunk</b> below. Then divide the upper lobe branches.</p>',
     'view': scope('posterior', np.mean([V(pt(i)) for i in POST + ['pa-lingular']], axis=0), dist=105, side=[-1, -0.35, 0.45]),
     'opacity': CLEAR,
     'highlight': POST + ['pa-lingular'], 'danger': ['pa-a6', 'pa-basal-trunk', 'n-vagus'],
     'action': {'kind': 'staple', 'label': 'Staple each artery', 'ids': POST + ['pa-lingular'], 'port': 'port-posterior-posterior', 'reload': 'vascular'},
     'ask': ask('In the fissure, which lower-lobe branch arises posteriorly, often opposite the lingular artery?', 'The superior segmental artery (A6)',
                'A6 is the first lower-lobe branch and leaves the posterior aspect of the artery; take it by mistake and the superior segment is devascularised.', 'The basal trunk', 'The truncus anterior'),
     'ct': ct(POST[0] if POST else 'pa-lingular')},
    {'id': 'p-truncus', 'phase': 'Artery', 'seq': 3, 'title': 'Truncus anterior: ligate and divide',
     'body': '<p>Follow the artery up to its first branch. Open the mediastinal pleura over the top of the hilum, in front of the vagus, and dissect the <b>truncus anterior</b> circumferentially.</p>'
             '<p><b>Ligate</b> it: two ties on the pulmonary artery side, one on the lobe side, then divide between. The truncus is short: leave enough stump for the ties.</p>'
             '<p>The aortic arch, the <b>recurrent laryngeal nerve</b> and the station 5 nodes are just above.</p>',
     'view': scope('posterior', tA, dist=100, side=[-0.65, 0.2, 1]), 'opacity': CLEAR,
     'highlight': ['pa-truncus-anterior'], 'danger': ['n-rln', 'n-vagus', 'aorta', 'pa-left'], 'labels': ['ln-5', 'br-lul'],
     'action': {'kind': 'ligate', 'label': 'Tie and divide', 'ids': ['pa-truncus-anterior'], 'port': 'port-posterior-utility'},
     'ask': ask('Where does the left recurrent laryngeal nerve leave the vagus?', 'At the aortic arch, hooking under it beside the ligamentum arteriosum',
                'On the left the nerve loops under the arch; on the right it loops under the subclavian artery.', 'Below the left main bronchus', 'At the level of the inferior pulmonary vein'),
     'pearl': 'A short, wide truncus that will not take three ties is stapled instead.',
     'ct': ct('pa-truncus-anterior')},
    {'id': 'p-bronchus', 'phase': 'Bronchus', 'seq': 4, 'title': 'Upper lobe bronchus: clamp, inflate, staple',
     'body': '<p>With the upper lobe arteries divided, the <b>upper lobe bronchus</b> lies free beneath where the artery arched over it. Clear station 11 at the secondary carina.</p>'
             '<p>Pass the endostapler (thick-tissue reload), close it, and <b>inflate: the lower lobe must ventilate</b> before firing.</p>',
     'view': scope('posterior', pt('br-lul'), dist=100, side=[-1, -0.45, 0.3]), 'opacity': CLEAR,
     'highlight': ['br-lul'], 'danger': ['br-lll', 'pa-left'], 'labels': ['ln-11l'],
     'action': {'kind': 'staple', 'label': 'Clamp, inflate, fire', 'ids': ['br-lul'], 'port': 'port-posterior-posterior', 'reload': 'tissue'},
     'ct': ct('br-lul')},
    {'id': 'p-vein', 'phase': 'Vein', 'seq': 5, 'title': 'Pulmonary veins: superior vein last',
     'body': '<p>Roll the lobe back to show the front of the hilum. Two veins: the <b>superior pulmonary vein</b> from the upper lobe, and below it the <b>inferior pulmonary vein</b> from the lower lobe, which stays.</p>'
             '<p>Keep the phrenic nerve on the pericardium, pass the vascular stapler round the superior vein and fire. The lobe is now free except for any fused fissure.</p>',
     'view': scope('anterior', (sv + iv) / 2, dist=120), 'retract': {'ids': LUL, 'offset': [-4, -14, 4], 'opacity': 0.3}, 'opacity': {'heart': 0.4, 'laa': 0.5, 'lll': 0.1},
     'highlight': ['pv-superior'], 'danger': ['pv-inferior', 'n-phrenic'],
     'action': {'kind': 'staple', 'label': 'Fire the stapler', 'ids': ['pv-superior'], 'port': 'port-posterior-utility', 'reload': 'vascular'},
     'ask': ask('Taking the vein last, what is the argument for doing it this way?', 'Arterial inflow is stopped before venous outflow',
                'Dividing arteries first avoids congesting the lobe; in cancer surgery the vein-first argument (less tumour-cell shedding) is debated.', 'The vein is easier to reach from behind', 'It avoids the phrenic nerve'),
     'ct': ct('pv-superior')},
    {'id': 'p-specimen', 'phase': 'Close', 'seq': 6, 'title': 'Specimen out, nodes, leak test',
     'body': '<p>Staple any remaining fused fissure anteriorly, bag and remove the lobe, and complete the nodal dissection: stations 5, 6, 7, 10 and 11. Leak-test the stump under saline.</p>',
     'view': {'frame': ['aorta', 'pa-left', 'br-left-main', 'pv-superior'], 'dir': [-0.7, -0.45, 0.5]}, 'opacity': {'heart': 0.4, 'lll': 0.2}, 'highlight': nodes, 'specimen': {'ids': LUL, 'offset': [-170, 20, -120]},
     'ct': ct(LM['ap-window'], 'coronal')},
]
# after the fissure is open (posterior approach) the lobes stay apart until the lobe is rolled back for the vein
for s in posterior[3:6]:
    s['retract'] = SPREAD_RETRACT()

# ==================================================================================================== left lower lobectomy
LLLS = ['lll', 'lll-arteries', 'lll-veins', 'lll-bronchi']
bas = V(pt('pa-basal-trunk')); brL = V(pt('br-lll')); LIG = V(S['ipl']['centroid']) if has('ipl') else iv + V([0, -20, -40])
LIG_LO = V(S['ipl']['bbox'][0]) if has('ipl') else LIG
UP = [{'ids': LLLS, 'offset': [-6, 6, 16], 'opacity': 0.3}]         # lower lobe retracted cephalad to open the ligament and hilum below
lnodes = [i for i in ('ln-7', 'ln-9l', 'ln-10l', 'ln-11l') if has(i)]


def lll_anatomy(op):
    return {
        'id': f'{op}-anatomy', 'phase': 'Anatomy', 'seq': 0, 'title': 'The lower lobe hilum',
        'body': '<p>Three structures leave the lower lobe, each lower and more posterior than its upper lobe counterpart. The <b>inferior pulmonary vein</b> is the lowest structure of the hilum, '
                'with the <b>inferior pulmonary ligament</b> running down from its lower border to the diaphragm.</p>'
                '<p>In the fissure the artery gives the <b>superior segmental artery (A6)</b> posteriorly, often opposite the <b>lingular artery</b>, and continues as the <b>basal trunk</b>. '
                'The <b>lower lobe bronchus</b> lies behind and between the artery and the vein; its superior segmental branch (B6) leaves early and posteriorly.</p>'
                '<p>Keep in view what stays: the <b>lingular artery</b>, the <b>posterior segmental arteries</b> and the <b>upper lobe bronchus</b>.</p>',
        'view': {'frame': ['pa-a6', 'pa-basal-trunk', 'pv-inferior', 'br-lll', 'pv-superior', 'pa-left'], 'dir': [-1, -0.55, 0.05], 'pad': 0.95},
        'opacity': {**CLEAR, 'lul': 0.08, 'lll': 0.08, 'fissure': 0.1, 'heart': 0.35}, 'spin': True,
        'labels': ['pa-a6', 'pa-basal-trunk', 'pa-lingular', 'pv-inferior', 'pv-superior', 'br-lll', 'br-lul', 'ipl'],
        'ct': ct('pv-inferior', 'coronal'),
    }


ligament = lambda op, port: {
    'id': f'{op}-ligament', 'phase': 'Ligament', 'seq': 1, 'title': 'Divide the inferior pulmonary ligament',
    'body': '<p>Retract the lower lobe up and forward. Divide the <b>inferior pulmonary ligament</b> with the diathermy hook from the diaphragm upward, close to the lung, '
            'taking the <b>station 9</b> nodes with the specimen.</p><p>Stop at the lower border of the <b>inferior pulmonary vein</b>. The oesophagus and the descending aorta are just medial: keep the hook on the lung side.</p>',
    'view': scope('posterior', (LIG + iv) / 2, dist=125, side=[-0.75, -0.8, -0.25]), 'retract': UP, 'opacity': {'lul': 0.3, 'heart': 0.4},
    'highlight': ['ipl'], 'danger': ['esophagus', 'aorta'], 'labels': ['pv-inferior', 'ln-9l'],
    'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Divide with the hook', 'port': port, 'remove': ['ipl'],
               'path': [R(LIG_LO + V([-3, 6, 4])), R((LIG_LO + LIG) / 2 + V([-3, 4, 0])), R(LIG + V([-3, 3, 0])), R(iv + V([-4, -4, -9]))]},
    'ct': ct(R(LIG), 'coronal'),
}

lll_vein = lambda op, seq, port, extra='': {
    'id': f'{op}-vein', 'phase': 'Vein', 'seq': seq, 'title': 'Inferior pulmonary vein: staple',
    'body': '<p>With the ligament divided the <b>inferior pulmonary vein</b> lies free at the bottom of the hilum. Clear it circumferentially.</p>'
            '<p>Before stapling, <b>see the superior pulmonary vein</b> as a separate structure: a common venous trunk, a recognised variant on the left, taken here would drain the whole lung.</p>' + extra,
    'view': scope('posterior', iv, dist=110, side=[-1, -0.55, -0.3]), 'retract': UP, 'opacity': {'lul': 0.3, 'heart': 0.4},
    'highlight': ['pv-inferior'], 'danger': ['pv-superior', 'esophagus'],
    'action': {'kind': 'staple', 'label': 'Fire the stapler', 'ids': ['pv-inferior'], 'port': port, 'reload': 'vascular'},
    'ask': ask('Before stapling the inferior pulmonary vein, what must you confirm?', 'That the superior pulmonary vein is separate and drains the upper lobe',
               'A common pulmonary vein taken as the "inferior vein" drains the whole lung.', 'That the fissure is complete', 'That A6 is already divided'),
    'ct': ct('pv-inferior'),
}

lll_artery = lambda op, seq, port, body: {
    'id': f'{op}-artery', 'phase': 'Artery', 'seq': seq, 'title': 'Superior segmental artery (A6) and basal trunk',
    'body': body,
    'view': scope('posterior', (a6 + bas) / 2, dist=110, side=[-1, -0.3, 0.45]), 'opacity': CLEAR,
    'highlight': ['pa-a6', 'pa-basal-trunk'], 'danger': ['pa-lingular', *POST],
    'action': {'kind': 'staple', 'label': 'Staple each artery', 'ids': ['pa-a6', 'pa-basal-trunk'], 'port': port, 'reload': 'vascular'},
    'ask': ask('Which upper lobe artery often arises opposite A6 and must be protected?', 'The lingular artery (A4+5)',
               'The lingular artery leaves the anterior aspect of the interlobar artery, often at the level of A6; stapling the basal trunk too high can take it.', 'The truncus anterior', 'The inferior pulmonary vein'),
    'pearl': 'If A6 arises high and the basal trunk is short, staple them separately; one stapler across both risks the lingular artery.',
    'ct': ct('pa-basal-trunk'),
}

lll_bronchus = lambda op, seq, port: {
    'id': f'{op}-bronchus', 'phase': 'Bronchus', 'seq': seq, 'title': 'Lower lobe bronchus: clamp, inflate, staple',
    'body': '<p>Sweep the <b>station 11</b> nodes up into the specimen and expose the <b>lower lobe bronchus</b> down to the secondary carina. Staple proximal to the <b>superior segmental bronchus (B6)</b>, which leaves early and posteriorly.</p>'
            '<p>Close the stapler (thick-tissue reload) and inflate: <b>the upper lobe must ventilate</b>. Then fire.</p>',
    'view': scope('posterior', brL, dist=105, side=[-1, -0.65, 0.1]), 'retract': UP, 'opacity': {'lul': 0.3, 'heart': 0.4},
    'highlight': ['br-lll'], 'danger': ['br-lul', 'pa-left'], 'labels': ['ln-11l'],
    'action': {'kind': 'staple', 'label': 'Clamp, inflate, fire', 'ids': ['br-lll'], 'port': port, 'reload': 'tissue'},
    'ask': ask('Stapler closed on the lower lobe bronchus: what must happen when the lung is inflated?', 'The upper lobe expands',
               'If the upper lobe does not ventilate, the stapler is across the left main or upper lobe bronchus.', 'The lower lobe expands', 'Nothing should move'),
    'ct': ct('br-lll'),
}

lll_specimen = lambda op, seq: {
    'id': f'{op}-specimen', 'phase': 'Close', 'seq': seq, 'title': 'Specimen out, nodes, leak test',
    'body': '<p>Bag the lobe and remove it. Complete the nodal dissection: <b>station 7</b> below the carina, <b>9</b> in the ligament, <b>10 and 11</b> at the hilum; for lower lobe tumours the subcarinal nodes matter most.</p>'
            '<p>Leak-test the bronchial stump under saline and check that the upper lobe fills the chest; an upper lobe that does not reach the apex may need an apical drain.</p>',
    'view': {'frame': ['pa-left', 'br-left-main', 'pv-superior', 'pv-inferior', 'heart'], 'dir': [-0.7, -0.5, 0.2], 'pad': 1.1}, 'opacity': {'heart': 0.4, 'lul': 0.3}, 'highlight': lnodes,
    'specimen': {'ids': LLLS, 'offset': [-110, -20, -70]},
    'ct': ct(LM['carina'], 'coronal'),
}

lll_fissure_first = [
    lll_anatomy('lf'),
    {**posterior[1], 'id': 'lf-setup'},
    ligament('lf', 'port-posterior-utility'),
    {**posterior[2], 'id': 'lf-fissure', 'seq': 2, 'highlight': ['pa-a6', 'pa-basal-trunk'], 'labels': ['fissure', 'pa-lingular'],
     'body': '<p>Retract the upper lobe forward and up, the lower lobe back and down. Open the visceral pleura where the fissure is deepest and <b>dissect bluntly with the peanut</b> onto the interlobar artery.</p>'
             '<p>Identify the <b>A6</b> branch behind, the <b>basal trunk</b> continuing down and the <b>lingular artery</b> in front, before dividing anything.</p>'},
    lll_artery('lf', 3, 'port-posterior-posterior',
               '<p>In the open fissure the artery is followed down. <b>A6</b> leaves its posterior surface first; the <b>basal trunk</b> continues below. Opposite A6, on the anterior surface, is the <b>lingular artery</b>, which stays.</p>'
               '<p>Divide A6 and the basal trunk separately with the vascular stapler.</p>'),
    lll_vein('lf', 4, 'port-posterior-posterior'),
    lll_bronchus('lf', 5, 'port-posterior-utility'),
    lll_specimen('lf', 6),
]
lll_fissure_first[4]['retract'] = SPREAD_RETRACT()          # after the fissure is opened the lobes stay apart for the artery

lll_hilum_first = [
    lll_anatomy('lh'),
    {**anterior[1], 'id': 'lh-setup'},
    ligament('lh', 'port-anterior-posterior'),
    lll_vein('lh', 2, 'port-anterior-posterior'),
    lll_bronchus('lh', 3, 'port-anterior-utility'),
    lll_artery('lh', 4, 'port-anterior-utility',
               '<p>With the vein and bronchus divided, lift the bronchial stump: the <b>basal trunk</b> and <b>A6</b> lie directly above and behind it.</p>'
               '<p>Identify the <b>lingular artery</b> on the front of the interlobar artery before stapling; A6 and the basal trunk are taken separately.</p>'),
    {**anterior[7], 'id': 'lh-fissure', 'seq': 5, 'danger': ['pa-lingular', *POST],
     'body': '<p>Only the fissure joins the lobes now. Staple it <b>front to back</b> with thick-tissue reloads, keeping the lingular artery and the upper lobe arteries above the staple line.</p>'},
    lll_specimen('lh', 6),
]
lll_hilum_first[5]['view'] = scope('posterior', (a6 + bas) / 2, dist=110, side=[-1, -0.55, 0.1])

# ==================================================================================================== right upper lobectomy
RUL_OK = has('rpa-truncus') and has('rpv-rul') and has('br-rul')
if RUL_OK:
    RULS = ['rul', 'rul-arteries', 'rul-veins', 'rul-bronchi']
    ASC = [i for i in ('rpa-a3-1', 'rpa-a3-2', 'rpa-a2-1', 'rpa-a2-2') if has(i)]
    ML = [i for i in ('rpa-ml-1', 'rpa-ml-2') if has(i)]
    RCLEAR = {'rul': 0.35, 'rml': 0.35, 'rll': 0.35, 'heart': 0.45}
    RLAT = V([1.0, 0, 0])
    tR = V(pt('rpa-truncus')); vU = V(pt('rpv-rul')); vM = V(pt('rpv-ml')) if has('rpv-ml') else vU + V([0, 0, -15])
    bR = V(pt('br-rul'))
    J = np.mean([V(pt(i)) for i in ASC[-1:] + ML[:1] + (['rpa-a6'] if has('rpa-a6') else [])], axis=0)   # where the fissures meet over the artery
    Nh, Nr = V(LM['fissure-h-normal']), V(LM['fissure-r-normal']); Ch, Cr = V(LM['fissure-h-centre']), V(LM['fissure-r-centre'])
    inplane = lambda v, n: (lambda w: w / np.linalg.norm(w))(v - n * np.dot(v, n))
    onp = lambda p, c, n: p - n * np.dot(p - c, n)
    AXh = inplane(V([0, 1.0, 0]), Nh)                 # along the horizontal fissure, toward the front
    AXr = inplane(V([0, -1.0, 0.6]), Nr)              # along the oblique fissure, up and back
    Jl = J + RLAT * 10
    H_LINE = [onp(Jl + AXh * t, Ch, Nh) for t in (72, 46, 22, 0)]
    O_LINE = [onp(Jl + AXr * t, Cr, Nr) for t in (0, 24, 48, 66)]
    RSPREAD = [{'ids': RULS, 'offset': [4, 6, 12]}, {'ids': ['rml', 'rll', 'fissure-h', 'fissure-r'], 'offset': [2, -4, -9]}]
    RBACK = {'ids': RULS, 'offset': [4, -16, 4], 'opacity': 0.3}          # upper lobe retracted back to open the front of the hilum
    rnodes = [i for i in ('ln-4r', 'ln-7', 'ln-10r', 'ln-11r') if has(i)]
    rports = lambda a: [f'port-{a}-{k}' for k in ('utility', 'camera', 'posterior')]

    def rul_anatomy(op, front):
        return {'id': f'{op}-anatomy', 'phase': 'Anatomy', 'seq': 0, 'title': 'The right hilum' + (' from the front' if front else ' from behind'),
                'body': ('<p>From the front, the <b>superior pulmonary vein</b> is the most anterior structure. Its upper tributaries drain the upper lobe; its lowest tributary, the <b>middle lobe vein</b>, must be kept. '
                         'Above and behind the vein, the <b>truncus anterior</b> leaves the right pulmonary artery as its first branch, just below the <b>azygos arch</b> and behind the <b>SVC</b>.</p>'
                         '<p>Order from the front: <b>upper lobe veins → truncus → bronchus → ascending arteries → fissures</b>.</p>'
                         if front else
                         '<p>From behind, the <b>upper lobe bronchus</b> leaves the right main bronchus high, above the artery (the eparterial bronchus), with the <b>bronchus intermedius</b> continuing below. '
                         'The <b>azygos arch</b> crosses above it into the SVC; the <b>vagus</b> runs down behind the hilum.</p>'
                         '<p>Where the fissures meet, the interlobar artery gives the <b>ascending posterior artery (A2)</b> up to the upper lobe, the <b>middle lobe artery</b> forward and <b>A6</b> back.</p>'
                         '<p>Order from behind: <b>fissure → ascending arteries → bronchus → truncus → upper lobe veins</b>.</p>'),
                'view': {'frame': ['rpa', 'rpa-truncus', 'rpv-superior', 'rpv-rul', 'br-rul', 'br-intermedius'], 'dir': [1, 0.45, 0.2] if front else [1, -0.6, 0.25], 'pad': 0.95},
                'opacity': {'rul': 0.08, 'rml': 0.08, 'rll': 0.08, 'fissure-h': 0.1, 'fissure-r': 0.1, 'heart': 0.35}, 'spin': True,
                'labels': ['rpa', 'rpa-truncus', *ASC[:1], *ML[:1], 'rpv-rul', 'rpv-ml', 'br-rul', 'br-intermedius', 'azygos', 'svc'],
                'ct': ct('rpa-truncus', 'coronal')}

    rv_step = lambda op, seq, port, body, view: {
        'id': f'{op}-veins', 'phase': 'Vein', 'seq': seq, 'title': 'Upper lobe veins: staple, keep the middle lobe vein',
        'body': body, 'view': view, 'retract': RBACK, 'opacity': {'heart': 0.4, 'rml': 0.3, 'rll': 0.3},
        'highlight': ['rpv-rul'], 'danger': ['rpv-ml', 'rpa', 'n-phrenic-r'],
        'action': {'kind': 'staple', 'label': 'Fire the stapler', 'ids': ['rpv-rul'], 'port': port, 'reload': 'vascular'},
        'ask': ask('Stapling the upper lobe veins, which tributary of the superior pulmonary vein must be kept?', 'The middle lobe vein',
                   'The middle lobe vein is the lowest tributary of the right superior vein; taking it with the upper lobe veins leaves the middle lobe congested and may force a bilobectomy.',
                   'The apical vein', 'The inferior pulmonary vein'),
        'ct': ct('rpv-rul')}

    truncus_step = lambda op, seq, port, kind: {
        'id': f'{op}-truncus', 'phase': 'Artery', 'seq': seq, 'title': 'Truncus anterior: ' + ('staple' if kind == 'staple' else 'ligate and divide'),
        'body': '<p>The <b>truncus anterior</b> is the first branch of the right pulmonary artery, leaving its upper surface for the apical and anterior segments. It is short and wide, just below the <b>azygos arch</b> and behind the SVC.</p>'
                '<p>Clear the <b>station 10R</b> node from the angle between the truncus and the artery; that exposes the length you need. ' + ('Pass the vascular stapler and fire.' if kind == 'staple' else 'Two ties on the artery side, one on the lobe side, then divide.') + '</p>',
        'view': scope('anterior', tR, dist=100, side=[1, 0.3, 0.6]), 'opacity': {**RCLEAR, 'rul': 0.3, 'svc': 0.3},
        'highlight': ['rpa-truncus'], 'danger': ['rpa', 'svc', 'azygos'], 'labels': ['ln-10r', 'br-rul'],
        'action': ({'kind': 'staple', 'label': 'Fire the stapler', 'ids': ['rpa-truncus'], 'port': port, 'reload': 'vascular'} if kind == 'staple'
                   else {'kind': 'ligate', 'label': 'Tie and divide', 'ids': ['rpa-truncus'], 'port': port}),
        'ask': ask('What lies immediately above the right truncus anterior as you clear it?', 'The azygos arch',
                   'The azygos arches forward over the right main bronchus into the SVC just above the truncus; it can be divided for exposure if needed.', 'The phrenic nerve', 'The inferior pulmonary vein'),
        'ct': ct('rpa-truncus')}

    bronchus_step = lambda op, seq, port, view, extra='': {
        'id': f'{op}-bronchus', 'phase': 'Bronchus', 'seq': seq, 'title': 'Upper lobe bronchus: clamp, inflate, staple',
        'body': '<p>Sweep <b>stations 10R and 11R</b> toward the specimen. The <b>upper lobe bronchus</b> leaves the right main bronchus high; below it the <b>bronchus intermedius</b> must stay intact.</p>'
                '<p>Close the stapler (thick-tissue reload) on the upper lobe bronchus and inflate: <b>the middle and lower lobes must ventilate</b>. Then fire.</p>' + extra,
        'view': view, 'opacity': {**RCLEAR, 'rul': 0.3},
        'highlight': ['br-rul'], 'danger': ['br-intermedius', 'br-right-main', 'rpa'], 'labels': ['ln-11r', 'ln-10r'],
        'action': {'kind': 'staple', 'label': 'Clamp, inflate, fire', 'ids': ['br-rul'], 'port': port, 'reload': 'tissue'},
        'ask': ask('Stapler closed on the right upper lobe bronchus: which lobes must inflate?', 'The middle and lower lobes',
                   'If the middle and lower lobes do not ventilate, the stapler is across the bronchus intermedius or the main bronchus.', 'Only the lower lobe', 'None: the lung is collapsed'),
        'ct': ct('br-rul')}

    asc_step = lambda op, seq, port, body, view, retract=None: {
        'id': f'{op}-ascending', 'phase': 'Artery', 'seq': seq, 'title': 'Ascending arteries to the upper lobe',
        'body': body, 'view': view, 'opacity': RCLEAR, **({'retract': retract} if retract else {}),
        'highlight': ASC, 'danger': ML + [i for i in ('rpa-a6', 'rpa-basal') if has(i)],
        'action': {'kind': 'staple', 'label': 'Staple each artery', 'ids': ASC, 'port': port, 'reload': 'vascular'},
        'ask': ask('Dividing the ascending posterior artery in the fissure, which branch arises close by and must be kept?', 'The superior segmental artery of the lower lobe (A6)',
                   'A2 leaves the interlobar artery at about the level of A6 and the middle lobe artery; identify all three before dividing.', 'The truncus anterior', 'The azygos vein'),
        'ct': ct(ASC[0])}

    rspec = lambda op, seq: {
        'id': f'{op}-specimen', 'phase': 'Close', 'seq': seq, 'title': 'Specimen out, nodes, leak test',
        'body': '<p>Bag the lobe and remove it. Complete the nodal dissection: <b>stations 2R and 4R</b> (between the SVC, trachea and azygos), <b>7</b> below the carina, <b>10R and 11R</b> at the hilum.</p>'
                '<p>Leak-test the stump under saline. Check that the <b>middle lobe</b> is pink, ventilating and not twisted; if it is mobile on a complete fissure, fix it to the lower lobe to prevent torsion.</p>',
        'view': {'frame': ['rpa', 'br-right-main', 'rpv-superior', 'rpv-inferior', 'svc'], 'dir': [0.8, 0.4, 0.3], 'pad': 1.1}, 'opacity': {'heart': 0.4, 'rml': 0.3, 'rll': 0.3},
        'highlight': rnodes, 'specimen': {'ids': RULS, 'offset': [120, 20, 60]}, 'ct': ct(LM['carina'], 'coronal')}

    rul_anterior = [
        rul_anatomy('ra', True),
        {'id': 'ra-setup', 'phase': 'Setup', 'title': 'Position and ports',
         'body': '<p>Left lateral decubitus, table flexed, left lung ventilated through a double-lumen tube.</p><p>Utility incision anteriorly in the <b>4th intercostal space</b> over the hilum; camera low and anterior; working port at the same level further back.</p>',
         'view': {'frame': ['skin'], 'dir': [1, 0.25, 0.15], 'pad': 1.05}, 'show': ['skin', *rports('r-anterior')], 'labels': rports('r-anterior'), 'ct': ct(LM['port-r-anterior-utility'], 'axial', 'lung')},
        {'id': 'ra-hilum', 'phase': 'Hilum', 'seq': 1, 'title': 'Open the pleura over the front of the hilum',
         'body': '<p>Retract the upper lobe back. With the peanut, sweep the mediastinal pleura off the front of the hilum <b>behind the phrenic nerve</b>, from the azygos arch down to the middle lobe vein.</p>'
                 '<p>Identify all the tributaries of the superior vein, including the <b>middle lobe vein</b> at its lower border.</p>',
         'view': scope('anterior', (vU + tR) / 2, dist=115, side=[1, 0.55, 0.2]), 'retract': RBACK, 'opacity': {'heart': 0.4, 'rml': 0.3, 'rll': 0.3},
         'highlight': ['rpv-rul'], 'danger': ['n-phrenic-r', 'svc'], 'labels': ['rpv-ml', 'rpa-truncus'],
         'action': {'kind': 'dissect', 'label': 'Dissect with the peanut', 'port': 'port-r-anterior-utility',
                    'path': [R(tR + V([9, 13, 8])), R(vU + V([8, 11, 6])), R(vU + V([9, 11, -6])), R(vM + V([8, 10, -2]))]},
         'ct': ct('rpv-rul')},
        rv_step('ra', 1, 'port-r-anterior-posterior',
                '<p>Encircle the <b>upper lobe tributaries</b> of the superior vein and pass the vascular stapler from the working port.</p><p>Before firing, trace the <b>middle lobe vein</b> into the middle lobe and keep it out of the jaws.</p>',
                scope('anterior', vU, dist=105, side=[1, 0.55, 0.15])),
        truncus_step('ra', 2, 'port-r-anterior-posterior', 'staple'),
        bronchus_step('ra', 3, 'port-r-anterior-utility', scope('anterior', bR, dist=105, side=[1, 0.2, 0.45])),
        asc_step('ra', 4, 'port-r-anterior-utility',
                 '<p>Lift the bronchial stump: the remaining upper lobe arteries rise from the interlobar artery below it. The <b>ascending posterior artery (A2)</b> and any <b>ascending anterior branch</b> go up into the upper lobe.</p>'
                 '<p>The <b>middle lobe artery</b> runs forward and <b>A6</b> back at the same level: keep both.</p>',
                 scope('anterior', np.mean([V(pt(i)) for i in ASC], axis=0), dist=110, side=[1, 0.1, 0.4])),
        {'id': 'ra-fissure-h', 'phase': 'Fissure', 'seq': 5, 'title': 'Horizontal fissure, front to back',
         'body': '<p>Only the fissures hold the upper lobe now. Staple the <b>horizontal fissure</b> from the front back to where the fissures meet, keeping the middle lobe below the staple line.</p>',
         'view': scope('anterior', np.mean(H_LINE, axis=0), dist=150, side=[1, 0.3, 0.6]), 'opacity': {'rul': 0.45, 'rml': 0.45, 'rll': 0.35, 'heart': 0.4},
         'labels': ['fissure-h'], 'danger': ML,
         'action': {'kind': 'staple-fissure', 'label': 'Staple the horizontal fissure', 'port': 'port-r-anterior-utility', 'reload': 'tissue', 'normal': R(Nh), 'path': [R(p) for p in H_LINE]},
         'ct': ct(R(Ch), 'sagittal', 'lung')},
        {'id': 'ra-fissure-o', 'phase': 'Fissure', 'seq': 5, 'title': 'Posterior oblique fissure',
         'body': '<p>Finish with the <b>posterior part of the oblique fissure</b>, from the junction up and back, keeping <b>A6</b> and the lower lobe below the line.</p>',
         'view': scope('posterior', np.mean(O_LINE, axis=0), dist=150, side=[1, -0.45, 0.5]), 'opacity': {'rul': 0.45, 'rml': 0.45, 'rll': 0.45, 'heart': 0.4},
         'labels': ['fissure-r'], 'danger': [i for i in ('rpa-a6',) if has(i)],
         'action': {'kind': 'staple-fissure', 'label': 'Staple the posterior fissure', 'port': 'port-r-anterior-posterior', 'reload': 'tissue', 'normal': R(Nr), 'path': [R(p) for p in O_LINE], 'spread': RSPREAD},
         'ct': ct(R(Cr), 'sagittal', 'lung')},
        rspec('ra', 6),
    ]

    rul_posterior = [
        rul_anatomy('rp', False),
        {'id': 'rp-setup', 'phase': 'Setup', 'title': 'Position and ports',
         'body': '<p>Left lateral decubitus, left lung ventilated. For a fissure-first approach the utility incision sits in the <b>5th intercostal space</b> over the fissures, with a posterior working port. Port sites vary between units.</p>',
         'view': {'frame': ['skin'], 'dir': [1, -0.35, 0.15], 'pad': 1.05}, 'show': ['skin', *rports('r-posterior')], 'labels': rports('r-posterior'), 'ct': ct(LM['port-r-posterior-utility'], 'axial', 'lung')},
        {'id': 'rp-fissure', 'phase': 'Fissure', 'seq': 1, 'title': 'Open the fissure junction with a peanut',
         'body': '<p>Where the horizontal and oblique fissures meet, open the visceral pleura and <b>dissect bluntly with the peanut</b> onto the interlobar artery.</p>'
                 '<p>Map its branches before dividing anything: <b>A2</b> up into the upper lobe, the <b>middle lobe artery</b> forward, <b>A6</b> back, the <b>basal trunk</b> down.</p>',
         'view': scope('posterior', J + RLAT * 10, dist=140, side=[1, -0.1, 0.55]), 'opacity': {**RCLEAR, 'fissure-h': 0.2, 'fissure-r': 0.2},
         'highlight': ['rpa'], 'labels': [*ASC[:1], *ML[:1], 'rpa-a6', 'rpa-basal'],
         'action': {'kind': 'open-fissure', 'label': 'Open the fissure', 'port': 'port-r-posterior-utility', 'spread': RSPREAD,
                    'path': [R(J + RLAT * 40 + V([0, 6, 8])), R(J + RLAT * 24 + V([0, 3, 4])), R(J + RLAT * 8), R(V(pt(ASC[-1])) + RLAT * 6)]},
         'ct': ct(ASC[-1], 'sagittal', 'lung')},
        asc_step('rp', 2, 'port-r-posterior-posterior',
                 '<p>In the open fissure, follow the interlobar artery up: the <b>ascending posterior artery (A2)</b> is the first upper lobe branch you meet from here, and any <b>ascending anterior branch</b> lies just in front.</p>'
                 '<p>Keep the <b>middle lobe artery</b> and <b>A6</b>, which leave at about the same level. Divide the ascending branches.</p>',
                 scope('posterior', np.mean([V(pt(i)) for i in ASC], axis=0), dist=105, side=[1, -0.3, 0.5]),
                 retract=[{'ids': sp['ids'], 'offset': sp['offset'], 'opacity': 0.35} for sp in RSPREAD]),
        bronchus_step('rp', 3, 'port-r-posterior-posterior', scope('posterior', bR, dist=100, side=[1, -0.7, 0.3]),
                      '<p>From behind, the bronchus is reached by opening the posterior mediastinal pleura below the azygos arch, in front of the vagus.</p>'),
        truncus_step('rp', 4, 'port-r-posterior-utility', 'ligate'),
        rv_step('rp', 5, 'port-r-posterior-utility',
                '<p>Roll the lobe back to show the front of the hilum. Only the <b>upper lobe veins</b> remain. Keep the phrenic nerve on the pericardium and the <b>middle lobe vein</b> out of the jaws, then fire.</p>',
                scope('anterior', (vU + vM) / 2, dist=115, side=[1, 0.6, 0.2])),
        rspec('rp', 6),
    ]
    rul_posterior[3]['retract'] = [{'ids': sp['ids'], 'offset': sp['offset'], 'opacity': 0.35} for sp in RSPREAD]

# ==================================================================================================== right lower and middle lobectomies
import copy
if RUL_OK:
    rA6, rBas, rIV, rBL = V(pt('rpa-a6')), V(pt('rpa-basal')), V(pt('rpv-inferior')), V(pt('br-rll'))
    RLIG = V(S['ipl-r']['centroid']) if has('ipl-r') else rIV + V([0, -20, -40]); RLIG_LO = V(S['ipl-r']['bbox'][0]) if has('ipl-r') else RLIG
    RUP = [{'ids': ['rll'], 'offset': [6, 6, 16], 'opacity': 0.3}]
    rlnodes = [i for i in ('ln-7', 'ln-9r', 'ln-10r', 'ln-11r') if has(i)]
    ROBL = [onp(Jl + AXr * t, Cr, Nr) for t in (-60, -36, -12, 8)] if False else [onp(J + RLAT * 12 + inplane(V([0, 1.0, -0.8]), Nr) * t, Cr, Nr) for t in (60, 36, 12, -8)]
    RLL_SPREAD = [{'ids': ['rul', 'rml', 'rul-arteries', 'rul-veins', 'rul-bronchi'], 'offset': R(Nr * 12)}, {'ids': ['rll', 'fissure-r'], 'offset': R(-Nr * 9)}]

    def rll_anatomy(op):
        return {'id': f'{op}-anatomy', 'phase': 'Anatomy', 'seq': 0, 'title': 'The right lower lobe hilum',
                'body': '<p>The <b>inferior pulmonary vein</b> is the lowest structure of the hilum, with the <b>inferior pulmonary ligament</b> below it. In the fissure the artery gives <b>A6</b> posteriorly and continues as the <b>basal trunk</b>; '
                        'at the same level the <b>middle lobe artery</b> leaves anteriorly and must be kept.</p>'
                        '<p>The <b>lower lobe bronchus</b> continues the bronchus intermedius; the <b>middle lobe bronchus</b> leaves its front just above, and <b>B6</b> its back at about the same level.</p>',
                'view': {'frame': ['rpa-a6', 'rpa-basal', 'rpv-inferior', 'br-rll', 'rpv-ml', 'rpa'], 'dir': [1, -0.55, 0.05], 'pad': 0.95},
                'opacity': {'rul': 0.08, 'rml': 0.08, 'rll': 0.08, 'fissure-r': 0.1, 'heart': 0.35}, 'spin': True,
                'labels': ['rpa-a6', 'rpa-basal', *ML[:1], 'rpv-inferior', 'rpv-ml', 'br-rll', 'br-rml', 'br-intermedius', 'ipl-r'], 'ct': ct('rpv-inferior', 'coronal')}

    rlig = lambda op, port: {
        'id': f'{op}-ligament', 'phase': 'Ligament', 'seq': 1, 'title': 'Divide the inferior pulmonary ligament',
        'body': '<p>Retract the lower lobe up. Divide the <b>inferior pulmonary ligament</b> with the hook from the diaphragm up to the lower border of the <b>inferior vein</b>, taking <b>station 9R</b> with the specimen. The oesophagus lies just medial.</p>',
        'view': scope('posterior', (RLIG + rIV) / 2, dist=125, side=[0.75, -0.8, -0.25]), 'retract': RUP, 'opacity': {'rul': 0.3, 'rml': 0.3, 'heart': 0.4},
        'highlight': ['ipl-r'], 'danger': ['esophagus'], 'labels': ['rpv-inferior', 'ln-9r'],
        'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Divide with the hook', 'port': port, 'remove': ['ipl-r'],
                   'path': [R(RLIG_LO + V([3, 6, 4])), R((RLIG_LO + RLIG) / 2 + V([3, 4, 0])), R(RLIG + V([3, 3, 0])), R(rIV + V([4, -4, -9]))]},
        'ct': ct(R(RLIG), 'coronal')}
    rlvein = lambda op, seq, port: {
        'id': f'{op}-vein', 'phase': 'Vein', 'seq': seq, 'title': 'Inferior pulmonary vein: staple',
        'body': '<p>With the ligament divided the <b>inferior pulmonary vein</b> lies free at the bottom of the hilum. See the <b>superior vein</b> and its <b>middle lobe tributary</b> as separate before stapling.</p>',
        'view': scope('posterior', rIV, dist=110, side=[1, -0.55, -0.3]), 'retract': RUP, 'opacity': {'rul': 0.3, 'rml': 0.3, 'heart': 0.4},
        'highlight': ['rpv-inferior'], 'danger': ['rpv-superior', 'rpv-ml', 'esophagus'],
        'action': {'kind': 'staple', 'label': 'Fire the stapler', 'ids': ['rpv-inferior'], 'port': port, 'reload': 'vascular'},
        'ask': ask('Before stapling the right inferior vein, which vein must you see separately?', 'The superior pulmonary vein with its middle lobe tributary',
                   'Taking a common trunk, or catching the middle lobe vein, drains lung you mean to keep.', 'The azygos vein', 'The inferior vena cava'),
        'ct': ct('rpv-inferior')}
    rlart = lambda op, seq, port, body: {
        'id': f'{op}-artery', 'phase': 'Artery', 'seq': seq, 'title': 'Superior segmental artery (A6) and basal trunk',
        'body': body, 'view': scope('posterior', (rA6 + rBas) / 2, dist=110, side=[1, -0.3, 0.45]), 'opacity': RCLEAR,
        'highlight': ['rpa-a6', 'rpa-basal'], 'danger': ML + ASC[-1:],
        'action': {'kind': 'staple', 'label': 'Staple each artery', 'ids': ['rpa-a6', 'rpa-basal'], 'port': port, 'reload': 'vascular'},
        'ask': ask('Stapling the right basal trunk, which artery arising at the same level must be protected?', 'The middle lobe artery',
                   'The middle lobe artery leaves the front of the interlobar artery opposite A6; a stapler placed too high on the basal trunk takes it.', 'The truncus anterior', 'The azygos vein'),
        'pearl': 'If A6 arises opposite the middle lobe artery, staple A6 and the basal trunk separately.', 'ct': ct('rpa-basal')}
    rlbr = lambda op, seq, port: {
        'id': f'{op}-bronchus', 'phase': 'Bronchus', 'seq': seq, 'title': 'Lower lobe bronchus: clamp, inflate, staple',
        'body': '<p>Clear <b>station 11R</b> and expose the <b>lower lobe bronchus</b>. The <b>middle lobe bronchus</b> leaves the front of the intermedius just above: angle the stapler so it does not narrow it, '
                'or take <b>B6</b> and the basal bronchus separately.</p><p>Close the stapler and inflate: <b>the upper and middle lobes must ventilate</b>. Then fire.</p>',
        'view': scope('posterior', rBL, dist=105, side=[1, -0.65, 0.1]), 'retract': RUP, 'opacity': {'rul': 0.3, 'rml': 0.3, 'heart': 0.4},
        'highlight': ['br-rll'], 'danger': ['br-rml', 'br-intermedius', 'rpa'], 'labels': ['ln-11r'],
        'action': {'kind': 'staple', 'label': 'Clamp, inflate, fire', 'ids': ['br-rll'], 'port': port, 'reload': 'tissue'},
        'ask': ask('Stapler closed on the right lower lobe bronchus: what must you check before firing?', 'The upper and middle lobes ventilate and the middle lobe bronchus is not narrowed',
                   'The middle lobe bronchus arises just above; a stapler across the intermedius takes both lobes.', 'Only that the lower lobe collapses', 'The azygos is divided'),
        'ct': ct('br-rll')}
    rlspec = lambda op, seq: {
        'id': f'{op}-specimen', 'phase': 'Close', 'seq': seq, 'title': 'Specimen out, nodes, leak test',
        'body': '<p>Remove the lobe and complete the nodal dissection: <b>station 7</b>, <b>9R</b>, <b>10R and 11R</b>, and the paratracheal stations. Leak-test the stump and check that the <b>middle lobe</b> is ventilating and not rotated.</p>',
        'view': {'frame': ['rpa', 'br-right-main', 'rpv-superior', 'rpv-inferior', 'heart'], 'dir': [0.7, -0.5, 0.2], 'pad': 1.1}, 'opacity': {'heart': 0.4, 'rul': 0.3, 'rml': 0.3},
        'highlight': rlnodes, 'specimen': {'ids': ['rll'], 'offset': [110, -20, -70]}, 'ct': ct(LM['carina'], 'coronal')}
    rfis_open = lambda op, seq: {
        'id': f'{op}-fissure', 'phase': 'Fissure', 'seq': seq, 'title': 'Open the oblique fissure with a peanut',
        'body': '<p>Retract the upper and middle lobes forward, the lower lobe back. Where the oblique fissure is deepest, <b>dissect bluntly with the peanut</b> onto the interlobar artery and map <b>A6</b>, the <b>basal trunk</b> and the <b>middle lobe artery</b>.</p>',
        'view': scope('posterior', J + RLAT * 10, dist=140, side=[1, -0.15, 0.5]), 'opacity': {**RCLEAR, 'fissure-r': 0.2},
        'highlight': ['rpa-a6', 'rpa-basal'], 'labels': ['fissure-r', *ML[:1]],
        'action': {'kind': 'open-fissure', 'label': 'Open the fissure', 'port': 'port-r-posterior-utility', 'spread': RLL_SPREAD,
                   'path': [R(J + RLAT * 40 + V([0, -6, -6])), R(J + RLAT * 22 + V([0, -3, -3])), R((rA6 + rBas) / 2 + RLAT * 7), R(rBas + RLAT * 6)]},
        'ask': ask('A fissure-first right lower lobectomy works best when…', 'The oblique fissure is complete over the artery',
                   'With a fused fissure the hilum-first order avoids tearing parenchyma over the artery.', 'The ligament is thick', 'The azygos is low'),
        'ct': ct('rpa-a6', 'sagittal', 'lung')}
    rsetup = lambda op, appr: {'id': f'{op}-setup', 'phase': 'Setup', 'title': 'Position and ports',
                               'body': '<p>Left lateral decubitus, table flexed, left lung ventilated through a double-lumen tube.</p>',
                               'view': {'frame': ['skin'], 'dir': [1, 0.25 if appr == 'r-anterior' else -0.35, 0.15], 'pad': 1.05}, 'show': ['skin', *rports(appr)], 'labels': rports(appr),
                               'ct': ct(LM[f'port-{appr}-utility'], 'axial', 'lung')}

    rll_fissure_first = [rll_anatomy('rf'), rsetup('rf', 'r-posterior'), rlig('rf', 'port-r-posterior-utility'), rfis_open('rf', 2),
                         rlart('rf', 3, 'port-r-posterior-posterior', '<p>In the open fissure, <b>A6</b> leaves the back of the artery and the <b>basal trunk</b> continues down; the <b>middle lobe artery</b> leaves the front at about the same level and stays.</p><p>Staple A6 and the basal trunk separately.</p>'),
                         rlvein('rf', 4, 'port-r-posterior-posterior'), rlbr('rf', 5, 'port-r-posterior-utility'), rlspec('rf', 6)]
    rll_fissure_first[4]['retract'] = [{'ids': sp['ids'], 'offset': sp['offset'], 'opacity': 0.35} for sp in RLL_SPREAD]
    rll_hilum_first = [rll_anatomy('rh'), rsetup('rh', 'r-anterior'), rlig('rh', 'port-r-anterior-posterior'), rlvein('rh', 2, 'port-r-anterior-posterior'), rlbr('rh', 3, 'port-r-anterior-utility'),
                       rlart('rh', 4, 'port-r-anterior-utility', '<p>With the vein and bronchus divided, lift the bronchial stump: <b>A6</b> and the <b>basal trunk</b> lie above and behind it. Identify the <b>middle lobe artery</b> in front first.</p>'),
                       {'id': 'rh-fissure', 'phase': 'Fissure', 'seq': 5, 'title': 'Staple the oblique fissure last',
                        'body': '<p>Only the oblique fissure joins the lower lobe now. Staple it front to back, keeping the <b>middle lobe</b> and its artery above the line.</p>',
                        'view': scope('anterior', np.mean(ROBL, axis=0), dist=150, side=[1, 0.1, 0.4]), 'opacity': {'rul': 0.45, 'rml': 0.45, 'rll': 0.45, 'heart': 0.4},
                        'labels': ['fissure-r'], 'danger': ML,
                        'action': {'kind': 'staple-fissure', 'label': 'Staple the fissure', 'port': 'port-r-anterior-utility', 'reload': 'tissue', 'normal': R(Nr), 'path': [R(p) for p in ROBL], 'spread': RLL_SPREAD},
                        'ct': ct(R(Cr), 'sagittal', 'lung')},
                       rlspec('rh', 6)]
    rll_hilum_first[5]['view'] = scope('posterior', (rA6 + rBas) / 2, dist=110, side=[1, -0.55, 0.1])

    # ---- right middle lobectomy
    vMl, bMl = V(pt('rpv-ml')), V(pt('br-rml'))
    ML_C = np.mean([V(pt(i)) for i in ML], axis=0)
    H_ML = [onp(Jl + AXh * t, Ch, Nh) for t in (72, 46, 22, 0)]
    AXa = inplane(V([0, 1.0, -0.7]), Nr); A_ML = [onp(Jl + AXa * t, Cr, Nr) for t in (0, 22, 44, 62)]
    RML_SPREAD = [{'ids': ['rml'], 'offset': R(-Nh * 9 + Nr * 6)}]
    rml_anat = lambda op: {'id': f'{op}-anatomy', 'phase': 'Anatomy', 'seq': 0, 'title': 'The middle lobe hilum',
                           'body': '<p>The middle lobe has three small hilar structures, all at the front. The <b>middle lobe vein</b> is the lowest tributary of the superior vein. Directly behind it lies the <b>middle lobe bronchus</b>, leaving the front of the intermedius, '
                                   'and behind and lateral to that, in the fissure, the <b>middle lobe artery</b> (often two branches) leaves the front of the interlobar artery opposite A6.</p>'
                                   '<p>Order from the front: <b>vein → bronchus → artery → fissures</b>.</p>',
                           'view': {'frame': ['rpv-ml', 'br-rml', *ML, 'rpv-rul'], 'dir': [1, 0.55, 0.05], 'pad': 1.0},
                           'opacity': {'rul': 0.08, 'rml': 0.08, 'rll': 0.08, 'fissure-h': 0.1, 'fissure-r': 0.1, 'heart': 0.35}, 'spin': True,
                           'labels': ['rpv-ml', 'rpv-rul', 'br-rml', 'br-intermedius', *ML, 'rpa-a6'], 'ct': ct('br-rml', 'sagittal')}
    rml_vein = lambda op, seq, port: {'id': f'{op}-vein', 'phase': 'Vein', 'seq': seq, 'title': 'Middle lobe vein: staple or ligate',
                                      'body': '<p>Retract the middle lobe back. Open the pleura over the front of the hilum behind the phrenic nerve and find the <b>lowest tributary of the superior vein</b> running into the middle lobe.</p>'
                                              '<p>Keep the <b>upper lobe veins</b> above it. Divide the middle lobe vein.</p>',
                                      'view': scope('anterior', vMl, dist=105, side=[1, 0.6, 0.1]), 'retract': {'ids': ['rml'], 'offset': [4, -12, -4], 'opacity': 0.3}, 'opacity': {'heart': 0.4, 'rul': 0.3, 'rll': 0.3},
                                      'highlight': ['rpv-ml'], 'danger': ['rpv-rul', 'rpv-inferior', 'n-phrenic-r'],
                                      'action': {'kind': 'staple', 'label': 'Fire the stapler', 'ids': ['rpv-ml'], 'port': port, 'reload': 'vascular'},
                                      'ask': ask('Which vein must be kept when dividing the middle lobe vein?', 'The upper lobe tributaries of the superior vein',
                                                 'The middle lobe vein joins the superior vein; the upper lobe veins join it just above.', 'The azygos vein', 'None'),
                                      'ct': ct('rpv-ml')}
    rml_br = lambda op, seq, port: {'id': f'{op}-bronchus', 'phase': 'Bronchus', 'seq': seq, 'title': 'Middle lobe bronchus: clamp, inflate, staple',
                                    'body': '<p>Directly behind the divided vein is the <b>middle lobe bronchus</b>. Sweep the <b>station 11R</b> node away and pass the stapler.</p>'
                                            '<p>The <b>interlobar artery</b> lies immediately behind and lateral to this bronchus: keep the anvil tip in view. Close and inflate: <b>upper and lower lobes must ventilate</b>.</p>',
                                    'view': scope('anterior', bMl, dist=100, side=[1, 0.55, 0.2]), 'retract': {'ids': ['rml'], 'offset': [4, -12, -4], 'opacity': 0.3}, 'opacity': {'heart': 0.4, 'rul': 0.3, 'rll': 0.3},
                                    'highlight': ['br-rml'], 'danger': ['br-intermedius', 'br-rll', 'rpa'], 'labels': ['ln-11r'],
                                    'action': {'kind': 'staple', 'label': 'Clamp, inflate, fire', 'ids': ['br-rml'], 'port': port, 'reload': 'tissue'},
                                    'ask': ask('Passing the stapler behind the middle lobe bronchus, what lies immediately behind it?', 'The interlobar pulmonary artery',
                                               'The artery runs just behind and lateral to the middle lobe bronchus; a blind anvil tip can injure it.', 'The azygos vein', 'The oesophagus'),
                                    'ct': ct('br-rml')}
    rml_art = lambda op, seq, port: {'id': f'{op}-artery', 'phase': 'Artery', 'seq': seq, 'title': 'Middle lobe artery (A4+5)',
                                     'body': '<p>Lift the bronchial stump: the <b>middle lobe artery</b> (often two branches) leaves the front of the interlobar artery. <b>A6</b> leaves the back at the same level and the <b>ascending posterior artery</b> goes up: keep both.</p>',
                                     'view': scope('anterior', ML_C, dist=105, side=[1, 0.35, 0.4]), 'opacity': RCLEAR,
                                     'highlight': ML, 'danger': ['rpa-a6', 'rpa-basal', *ASC[-1:]],
                                     'action': {'kind': 'staple', 'label': 'Staple each artery', 'ids': ML, 'port': port, 'reload': 'vascular'}, 'ct': ct(ML[0])}
    rml_hfis = lambda op, seq, port: {'id': f'{op}-fissure-h', 'phase': 'Fissure', 'seq': seq, 'title': 'Horizontal fissure',
                                      'body': '<p>Staple the <b>horizontal fissure</b> between the middle and upper lobes, from the front back to the fissure junction.</p>',
                                      'view': scope('anterior', np.mean(H_ML, axis=0), dist=150, side=[1, 0.3, 0.6]), 'opacity': {'rul': 0.45, 'rml': 0.45, 'rll': 0.35, 'heart': 0.4},
                                      'labels': ['fissure-h'], 'danger': ASC[:1],
                                      'action': {'kind': 'staple-fissure', 'label': 'Staple the horizontal fissure', 'port': port, 'reload': 'tissue', 'normal': R(Nh), 'path': [R(p) for p in H_ML]}, 'ct': ct(R(Ch), 'sagittal', 'lung')}
    rml_ofis = lambda op, seq, port: {'id': f'{op}-fissure-o', 'phase': 'Fissure', 'seq': seq, 'title': 'Anterior oblique fissure',
                                      'body': '<p>Finish with the <b>anterior part of the oblique fissure</b>, between the middle and lower lobes, keeping the basal trunk below the line.</p>',
                                      'view': scope('anterior', np.mean(A_ML, axis=0), dist=150, side=[1, 0.5, -0.1]), 'opacity': {'rul': 0.45, 'rml': 0.45, 'rll': 0.45, 'heart': 0.4},
                                      'labels': ['fissure-r'], 'danger': ['rpa-basal'],
                                      'action': {'kind': 'staple-fissure', 'label': 'Staple the oblique fissure', 'port': port, 'reload': 'tissue', 'normal': R(Nr), 'path': [R(p) for p in A_ML], 'spread': RML_SPREAD}, 'ct': ct(R(Cr), 'sagittal', 'lung')}
    rml_spec = lambda op, seq: {'id': f'{op}-specimen', 'phase': 'Close', 'seq': seq, 'title': 'Specimen out, nodes, leak test',
                                'body': '<p>Remove the lobe and complete the nodal dissection: stations <b>4R, 7, 10R and 11R</b>. Leak-test the stump; check the upper and lower lobes expand to fill the space.</p>',
                                'view': {'frame': ['rpa', 'br-right-main', 'rpv-superior', 'rpv-inferior', 'heart'], 'dir': [0.8, 0.5, 0.1], 'pad': 1.1}, 'opacity': {'heart': 0.4, 'rul': 0.3, 'rll': 0.3},
                                'highlight': rnodes, 'specimen': {'ids': ['rml'], 'offset': [110, 60, -40]}, 'ct': ct(LM['carina'], 'coronal')}
    rml_fisopen = {'id': 'mf-fissure', 'phase': 'Fissure', 'seq': 1, 'title': 'Open the fissure junction with a peanut',
                   'body': '<p>Where the horizontal and oblique fissures meet, <b>dissect bluntly with the peanut</b> onto the interlobar artery. The <b>middle lobe artery</b> leaves its front; <b>A6</b> its back.</p>',
                   'view': scope('posterior', J + RLAT * 10, dist=140, side=[1, 0.2, 0.55]), 'opacity': {**RCLEAR, 'fissure-h': 0.2, 'fissure-r': 0.2},
                   'highlight': ML, 'labels': ['rpa-a6', *ASC[-1:]],
                   'action': {'kind': 'open-fissure', 'label': 'Open the fissure', 'port': 'port-r-anterior-utility', 'spread': RML_SPREAD,
                              'path': [R(J + RLAT * 40 + V([0, 8, 2])), R(J + RLAT * 22 + V([0, 4, 1])), R(ML_C + RLAT * 7), R(ML_C + RLAT * 5 + V([0, 4, -4]))]},
                   'ct': ct(ML[0], 'sagittal', 'lung')}
    rml_anterior = [rml_anat('ma'), rsetup('ma', 'r-anterior'), rml_vein('ma', 1, 'port-r-anterior-posterior'), rml_br('ma', 2, 'port-r-anterior-posterior'),
                    rml_art('ma', 3, 'port-r-anterior-utility'), rml_hfis('ma', 4, 'port-r-anterior-utility'), rml_ofis('ma', 4, 'port-r-anterior-posterior'), rml_spec('ma', 5)]
    rml_fissure_first = [rml_anat('mf'), rsetup('mf', 'r-anterior'), rml_fisopen, rml_art('mf', 2, 'port-r-anterior-utility'), rml_vein('mf', 3, 'port-r-anterior-posterior'),
                         rml_br('mf', 4, 'port-r-anterior-posterior'), rml_hfis('mf', 5, 'port-r-anterior-utility'), rml_spec('mf', 6)]
    rml_fissure_first[3]['retract'] = [{'ids': sp['ids'], 'offset': sp['offset'], 'opacity': 0.35} for sp in RML_SPREAD]


# ==================================================================================================== pneumonectomy
LPA_BR = [i for i in S if i.startswith('pa-') and i != 'pa-left']
L_SPEC = [i for i in ('lul', 'lll', 'fissure', 'lul-arteries', 'lul-veins', 'lul-bronchi', 'lll-arteries', 'lll-veins', 'lll-bronchi', 'br-lul', 'br-lll',
                      'br-lingular', 'br-upper-div', 'br-b6', 'pv-lingular', 'pv-upper-div', 'pv-v6', *LPA_BR) if has(i)]
LPN_OK = all(has(i) and 'division' in S[i] for i in ('pa-left', 'br-left-main', 'pv-superior', 'pv-inferior'))


def pn_steps(side):
    L = side == 'left'
    sx = -1 if L else 1
    pa, spv_, ipv_, mb = ('pa-left', 'pv-superior', 'pv-inferior', 'br-left-main') if L else ('rpa', 'rpv-superior', 'rpv-inferior', 'br-right-main')
    lig_id, lig_node = ('ipl', 'ln-9l') if L else ('ipl-r', 'ln-9r')
    lobes = ['lul', 'lll'] if L else ['rul', 'rml', 'rll']
    spec = L_SPEC if L else [i for i in ('rul', 'rml', 'rll', 'fissure-h', 'fissure-r', 'rul-arteries', 'rul-veins', 'rul-bronchi', 'br-rul', 'br-intermedius', 'br-rml', 'br-rll',
                                         'rpv-rul', 'rpv-ml', *[k for k in S if k.startswith('rpa-')]) if has(i)]
    nerves = ['n-phrenic', 'n-vagus', 'n-rln'] if L else ['n-phrenic-r', 'n-vagus-r']
    port = lambda k: f'port-{"" if L else "r-"}anterior-{k}'
    P, SV, IV, MB = V(pt(pa)), V(pt(spv_)), V(pt(ipv_)), V(pt(mb))
    lig_c = V(S[lig_id]['centroid']) if has(lig_id) else IV + V([0, -20, -40]); lig_lo = V(S[lig_id]['bbox'][0]) if has(lig_id) else lig_c
    clear = {k: 0.3 for k in lobes} | {'heart': 0.4}
    up = [{'ids': lobes, 'offset': [sx * 6, 4, 14], 'opacity': 0.3}]
    back = [{'ids': lobes, 'offset': [sx * 4, -14, 4], 'opacity': 0.3}]
    pre = 'lp' if L else 'rp'
    st = [
        {'id': f'{pre}-anatomy', 'phase': 'Anatomy', 'seq': 0, 'title': f'The {side} hilum as a whole',
         'body': ('<p>A pneumonectomy divides four structures: the <b>main pulmonary artery</b>, the <b>superior</b> and <b>inferior pulmonary veins</b>, and the <b>main bronchus</b>. '
                  + ('On the left the artery arches over the bronchus under the <b>aortic arch</b>, tethered by the <b>ligamentum arteriosum</b>, with the <b>recurrent laryngeal nerve</b> hooking round it. The left main bronchus is long and runs under the arch to the carina.</p>'
                     if L else 'On the right the artery runs behind the <b>SVC</b>, below the <b>azygos arch</b>; the right main bronchus is short, with the upper lobe bronchus leaving it close to the carina.</p>')
                  + '<p>Before committing, assess the <b>fissure and hilum for a lobectomy</b> option, and trial-clamp the artery to see that the right heart tolerates it.</p>'),
         'view': {'frame': [pa, spv_, ipv_, mb], 'dir': [sx, 0.35, 0.2], 'pad': 1.0},
         'opacity': {**{k: 0.08 for k in lobes}, 'heart': 0.35}, 'spin': True,
         'labels': [pa, spv_, ipv_, mb, 'aorta', *(['lig-art', 'n-rln'] if L else ['azygos', 'svc'])], 'ct': ct(pa, 'coronal')},
        {'id': f'{pre}-setup', 'phase': 'Setup', 'title': 'Position and ports',
         'body': f'<p>{"Right" if L else "Left"} lateral decubitus, table flexed, {"right" if L else "left"} lung ventilated through a double-lumen tube (the tube must not sit in the bronchus you will divide). Anterior utility incision over the hilum, low camera port, posterior working port.</p>',
         'view': {'frame': ['skin'], 'dir': [sx, 0.25, 0.15], 'pad': 1.05}, 'show': ['skin', *[port(k) for k in ('utility', 'camera', 'posterior')]], 'labels': [port(k) for k in ('utility', 'camera', 'posterior')],
         'ct': ct(LM[port('utility')], 'axial', 'lung')},
        {'id': f'{pre}-ligament', 'phase': 'Ligament', 'seq': 1, 'title': 'Divide the inferior pulmonary ligament',
         'body': '<p>Retract the lower lobe up and divide the <b>inferior pulmonary ligament</b> with the hook up to the inferior vein, taking station 9. The oesophagus lies just medial.</p>',
         'view': scope('posterior', (lig_c + IV) / 2, dist=125, side=[sx * 0.75, -0.8, -0.25]), 'retract': up, 'opacity': clear,
         'highlight': [lig_id], 'danger': ['esophagus'], 'labels': [ipv_, lig_node],
         'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Divide with the hook', 'port': port('posterior'), 'remove': [lig_id],
                    'path': [R(lig_lo + V([-sx * 3, 6, 4])), R((lig_lo + lig_c) / 2 + V([-sx * 3, 4, 0])), R(lig_c + V([-sx * 3, 3, 0])), R(IV + V([-sx * 4, -4, -9]))]},
         'ct': ct(R(lig_c), 'coronal')},
        {'id': f'{pre}-ipv', 'phase': 'Vein', 'seq': 2, 'title': 'Inferior pulmonary vein: staple',
         'body': '<p>Clear the <b>inferior pulmonary vein</b> circumferentially down to the pericardium and staple it with a vascular load. For a central tumour it can be taken <b>inside the pericardium</b>.</p>',
         'view': scope('posterior', IV, dist=110, side=[sx, -0.55, -0.3]), 'retract': up, 'opacity': clear,
         'highlight': [ipv_], 'danger': ['esophagus'],
         'action': {'kind': 'staple', 'label': 'Fire the stapler', 'ids': [ipv_], 'port': port('posterior'), 'reload': 'vascular'}, 'ct': ct(ipv_)},
        {'id': f'{pre}-spv', 'phase': 'Vein', 'seq': 3, 'title': 'Superior pulmonary vein: staple',
         'body': '<p>Retract the lung back, open the pleura over the front of the hilum <b>behind the phrenic nerve</b>, and staple the <b>superior pulmonary vein</b> as a trunk, close to the pericardium.</p>',
         'view': scope('anterior', SV, dist=110, side=[sx, 0.55, 0.15]), 'retract': back, 'opacity': clear,
         'highlight': [spv_], 'danger': [nerves[0], pa],
         'action': {'kind': 'staple', 'label': 'Fire the stapler', 'ids': [spv_], 'port': port('posterior'), 'reload': 'vascular'}, 'ct': ct(spv_)},
        {'id': f'{pre}-pa', 'phase': 'Artery', 'seq': 4, 'title': 'Main pulmonary artery: clamp test, staple',
         'body': ('<p>The <b>main pulmonary artery</b> now lies free above the bronchus. Clear it proximal to its first branch'
                  + (', staying below the <b>aortic arch</b> and clear of the <b>recurrent laryngeal nerve</b> at the ligamentum.' if L else ', behind the SVC and below the azygos.')
                  + '</p><p><b>Trial-clamp</b> it for a few minutes: watch blood pressure, heart rate and the right ventricle on echo. If tolerated, staple with a vascular load.</p>'),
         'view': scope('anterior', P, dist=110, side=[sx, 0.3, 0.55]), 'retract': back, 'opacity': clear | ({'svc': 0.3} if not L else {}),
         'highlight': [pa], 'danger': (['n-rln', 'lig-art', 'aorta'] if L else ['svc', 'azygos']),
         'action': {'kind': 'staple', 'label': 'Fire the stapler', 'ids': [pa], 'port': port('posterior'), 'reload': 'vascular'},
         'ask': ask('Why trial-clamp the main pulmonary artery before dividing it?', 'To see whether the right ventricle tolerates the whole cardiac output going to one lung',
                    'Acute right heart strain or a fall in pressure on clamping warns that pneumonectomy may not be tolerated.', 'To check for bleeding from the bronchial arteries', 'To test the bronchial stump'),
         'ct': ct(pa)},
        {'id': f'{pre}-bronchus', 'phase': 'Bronchus', 'seq': 5, 'title': 'Main bronchus: staple flush with the carina',
         'body': ('<p>Clear the subcarinal nodes (<b>station 7</b>) and follow the <b>main bronchus</b> to the carina. Staple it <b>flush with the carina</b> with a thick-tissue load: a long stump pools secretions and leaks.</p>'
                  + ('<p>On the left the bronchus runs under the aortic arch: pull the lung down and out to reach the carina.</p>' if L else '<p>On the right the stump has no aortic arch to cover it and leaks more often: plan to cover it.</p>')),
         'view': scope('posterior', MB, dist=110, side=[sx, -0.5, 0.35]), 'opacity': clear, 'highlight': [mb], 'danger': ['trachea', 'esophagus', *(['aorta'] if L else ['azygos'])], 'labels': ['ln-7'],
         'action': {'kind': 'staple', 'label': 'Clamp, fire', 'ids': [mb], 'port': port('utility'), 'reload': 'tissue'},
         'ask': ask('Why staple the main bronchus flush with the carina?', 'A long stump pools secretions and is prone to breakdown (bronchopleural fistula)',
                    'The shorter the stump, the less dead space for infection and dehiscence.', 'To preserve the contralateral lung', 'To make the specimen easier to remove'),
         'ct': ct(mb)},
        {'id': f'{pre}-specimen', 'phase': 'Close', 'seq': 6, 'title': 'Specimen out, cover the stump, leak test',
         'body': '<p>Remove the lung in a bag through an enlarged incision. Leak-test the stump under saline at 20–25 cmH<sub>2</sub>O. '
                 + ('Cover it with a vascularised flap (pericardial fat, pleura or intercostal muscle), especially after induction therapy.' if L else '<b>Cover the right stump</b> with a vascularised flap (intercostal muscle, pericardial fat or azygos-pleura).')
                 + '</p><p>Complete the nodal dissection. Leave a balanced drain or none, and keep the mediastinum central; restrict fluids.</p>',
         'view': {'frame': [pa, mb, spv_, 'heart'], 'dir': [sx * 0.8, 0.3, 0.3], 'pad': 1.2}, 'opacity': {'heart': 0.4},
         'highlight': [i for i in (['ln-5', 'ln-7', 'ln-10l'] if L else ['ln-4r', 'ln-7', 'ln-10r']) if has(i)],
         'specimen': {'ids': spec, 'offset': [sx * 150, 10, -60]}, 'ct': ct(LM['carina'], 'coronal')},
    ]
    return st


PN = {}
if LPN_OK: PN['left'] = pn_steps('left')
if RUL_OK and all(has(i) and 'division' in S[i] for i in ('rpa', 'br-right-main', 'rpv-superior', 'rpv-inferior')): PN['right'] = pn_steps('right')


def artery_first(steps):
    """open pneumonectomy, artery first: ligament, main PA, superior vein, inferior vein, bronchus"""
    d = {x['id'].split('-', 1)[1]: x for x in steps}
    order = [d['anatomy'], d['setup'], d['ligament'], d['pa'], d['spv'], d['ipv'], d['bronchus'], d['specimen']]
    out = copy.deepcopy(order)
    for i, x in enumerate(out):
        if 'seq' in x: x['seq'] = i - 1 if i > 1 else 0
    return out


# ==================================================================================================== left segmentectomies
SEG_OK = all(has(i) for i in ('seg-lingula', 'seg-lul-upper', 'isp-lingula', 'br-lingular', 'pv-lingular'))
if SEG_OK:
    def plane_path(isp, hilum_pt):
        c, n, a = V(LM[f'{isp}-centre']), V(LM[f'{isp}-normal']), V(LM[f'{isp}-axis'])
        pts = [c + a * t for t in (-45, -15, 15, 45)]
        if np.linalg.norm(pts[-1] - hilum_pt) < np.linalg.norm(pts[0] - hilum_pt): pts = pts[::-1]   # staple from the hilum outward
        return pts, n

    def seg_proc(op, title, target_seg, keep_seg, isp, arteries, vein, bronchus, protect, anat_body, notes):
        showsegs = [target_seg, keep_seg]
        segop = {'lul': 0.06, 'lll': 0.06, target_seg: 0.4, keep_seg: 0.25, 'heart': 0.4}
        hil = V(pt(bronchus)); path, n = plane_path(isp, hil)
        base_show = lambda: list(showsegs)
        steps = [
            {'id': f'{op}-anatomy', 'phase': 'Anatomy', 'seq': 0, 'title': title, 'body': anat_body,
             'view': {'frame': [target_seg, *arteries, vein, bronchus], 'dir': [-1, -0.2, 0.3], 'pad': 1.0}, 'show': base_show(),
             'opacity': segop, 'spin': True, 'labels': [target_seg, keep_seg, *arteries, vein, bronchus], 'ct': ct(bronchus, 'coronal', 'lung')},
            {**copy.deepcopy(posterior[1]), 'id': f'{op}-setup'},
            {**copy.deepcopy(posterior[2]), 'id': f'{op}-fissure', 'seq': 1, 'show': base_show(), 'highlight': arteries[:1], 'labels': ['fissure', *protect[:2]]},
            {'id': f'{op}-artery', 'phase': 'Artery', 'seq': 2, 'title': 'Segmental arter' + ('ies' if len(arteries) > 1 else 'y'),
             'body': notes['artery'], 'view': scope('posterior', np.mean([V(pt(i)) for i in arteries], axis=0), dist=110, side=[-1, -0.3, 0.45]),
             'show': base_show(), 'opacity': {**segop, 'lul': 0.2, 'lll': 0.2}, 'highlight': arteries, 'danger': protect,
             'action': {'kind': 'staple', 'label': 'Staple' + (' each artery' if len(arteries) > 1 else ''), 'ids': arteries, 'port': 'port-posterior-posterior', 'reload': 'vascular'},
             'ct': ct(arteries[0])},
            {'id': f'{op}-vein', 'phase': 'Vein', 'seq': 3, 'title': 'Segmental vein', 'body': notes['vein'],
             'view': scope('anterior', V(pt(vein)), dist=105, side=notes.get('vein_side', [-1, 0.5, 0.1])), 'show': base_show(), 'opacity': {**segop, 'lul': 0.2, 'lll': 0.2},
             'highlight': [vein], 'danger': notes['vein_keep'],
             'action': {'kind': 'staple', 'label': 'Fire the stapler', 'ids': [vein], 'port': 'port-posterior-utility', 'reload': 'vascular'}, 'ct': ct(vein)},
            {'id': f'{op}-bronchus', 'phase': 'Bronchus', 'seq': 4, 'title': 'Segmental bronchus, then inflate: the plane appears', 'body': notes['bronchus'],
             'view': scope('posterior', hil, dist=100, side=[-1, -0.35, 0.3]), 'show': base_show(), 'opacity': {**segop, 'lul': 0.2, 'lll': 0.2},
             'highlight': [bronchus], 'danger': notes['bronchus_keep'],
             'action': {'kind': 'staple', 'label': 'Clamp, fire', 'ids': [bronchus], 'port': 'port-posterior-utility', 'reload': 'tissue'},
             'ask': ask('After the segmental bronchus is divided, how is the intersegmental plane shown with the inflation–deflation method?', 'Inflate the whole lung, then let it deflate: the target segment stays inflated',
                        'Air trapped behind the divided bronchus keeps the segment inflated while the rest deflates, drawing the boundary; intravenous ICG after dividing the artery is the alternative.', 'The segment deflates first', 'By palpation of the fissure'),
             'ct': ct(bronchus)},
            {'id': f'{op}-plane', 'phase': 'Plane', 'seq': 5, 'title': 'Divide the intersegmental plane',
             'body': '<p>Lift the divided bronchus and vessels with the specimen and staple along the <b>inflation–deflation line</b>, from the hilum outward, keeping the <b>intersegmental veins</b> on the side that stays.</p>'
                     '<p>A margin of at least the tumour diameter (and 2 cm) is the aim for cancer.</p>',
             'view': scope('posterior', np.mean(path, axis=0), dist=160, side=[-1, 0.0, 0.35]), 'show': [*showsegs, isp], 'opacity': {**segop, isp: 0.35},
             'labels': [isp], 'danger': protect[:2],
             'action': {'kind': 'staple-fissure', 'label': 'Staple the plane', 'port': 'port-posterior-utility', 'reload': 'tissue', 'normal': R(n), 'path': [R(p_) for p_ in path]}, 'ct': ct(R(np.mean(path, axis=0)), 'sagittal', 'lung')},
            {'id': f'{op}-specimen', 'phase': 'Close', 'seq': 6, 'title': 'Specimen out, nodes, margins',
             'body': '<p>Remove the segment in a bag. Sample <b>stations 10, 11 and 12/13</b> at its root: a positive intersegmental node means converting to lobectomy. Check the margin, then leak-test the stapled plane under saline.</p>',
             'view': {'frame': [keep_seg, bronchus, 'pa-left'], 'dir': [-1, -0.2, 0.3], 'pad': 1.3}, 'show': [keep_seg], 'opacity': {'lul': 0.06, 'lll': 0.06, keep_seg: 0.3, 'heart': 0.4},
             'highlight': [i for i in ('ln-10l', 'ln-11l') if has(i)], 'specimen': {'ids': [target_seg], 'offset': [-120, 10, -30]}, 'ct': ct(bronchus, 'coronal', 'lung')},
        ]
        return steps

    lingulectomy = seg_proc('sl', 'Lingular segmentectomy (S4+5)', 'seg-lingula', 'seg-lul-upper', 'isp-lingula', ['pa-lingular'], 'pv-lingular', 'br-lingular',
                            ['pa-a6', 'pa-basal-trunk', *POST[:1]],
                            '<p>The lingula is the upper lobe\'s lower division, supplied by its own three structures: the <b>lingular artery</b> from the front of the artery in the fissure, the <b>lingular vein</b> as the lowest tributary of the superior vein, '
                            'and the <b>lingular bronchus</b>, the lower branch of the upper lobe bronchus.</p><p>What stays: the <b>upper division</b> above, A6 and the basal trunk in the fissure.</p>',
                            {'artery': '<p>In the open fissure, the <b>lingular artery</b> leaves the front of the interlobar artery toward the lingula, often opposite <b>A6</b>. Staple it clear of the basal trunk.</p>',
                             'vein': '<p>Roll the lobe back. The <b>lingular vein</b> is the lowest tributary of the superior pulmonary vein; the <b>upper division veins</b> join above it and must be kept.</p>', 'vein_keep': ['pv-upper-div', 'pv-inferior'],
                             'bronchus': '<p>Behind the vein, the <b>lingular bronchus</b> leaves the lower side of the upper lobe bronchus. Clamp it and inflate: the <b>upper division must still ventilate</b>. Fire.</p>', 'bronchus_keep': ['br-upper-div', 'br-lll']})
    trisegmentectomy = seg_proc('su', 'Upper division segmentectomy (S1+2, S3)', 'seg-lul-upper', 'seg-lingula', 'isp-lingula', ['pa-truncus-anterior', *POST], 'pv-upper-div', 'br-upper-div',
                                ['pa-lingular', 'pa-a6', 'pv-lingular'],
                                '<p>A lingula-sparing upper lobectomy: the <b>upper division</b> (S1+2 and S3) is removed and the lingula kept. Its artery supply is the <b>truncus anterior</b> and the <b>posterior segmental arteries</b>; its veins are the upper tributaries of the superior vein; '
                                'its bronchus is the <b>upper division bronchus</b>.</p><p>What stays: the <b>lingular artery, vein and bronchus</b>.</p>',
                                {'artery': '<p>From the fissure, divide the <b>posterior segmental arteries</b>; from above and in front, the <b>truncus anterior</b>. The <b>lingular artery</b> lower down stays.</p>',
                                 'vein': '<p>Divide the <b>upper division veins</b> (V1+2, V3) where they join the superior vein, keeping the <b>lingular vein</b> below.</p>', 'vein_keep': ['pv-lingular'],
                                 'bronchus': '<p>The <b>upper division bronchus</b> is the upper branch of the upper lobe bronchus. Clamp it and inflate: the <b>lingula must ventilate</b>. Fire.</p>', 'bronchus_keep': ['br-lingular', 'br-lll']})
    s6seg = None
    if has('seg-s6') and has('pv-v6') and has('br-b6'):
        s6seg = seg_proc('s6', 'Superior segmentectomy (S6)', 'seg-s6', 'seg-lll-basal', 'isp-s6', ['pa-a6'], 'pv-v6', 'br-b6', ['pa-basal-trunk', 'pa-lingular', *POST[-1:]],
                         '<p>The superior segment is the top of the lower lobe, supplied by <b>A6</b> from the back of the artery in the fissure, drained by <b>V6</b>, the highest tributary of the inferior vein, and aerated by <b>B6</b>, the first posterior branch of the lower lobe bronchus.</p>'
                         '<p>What stays: the <b>basal trunk</b>, basal veins and basal bronchus.</p>',
                         {'artery': '<p>In the fissure, <b>A6</b> leaves the back of the interlobar artery; the <b>basal trunk</b> continues below and the <b>lingular artery</b> leaves the front opposite. Staple A6 only.</p>',
                          'vein': '<p>From behind, with the lower lobe lifted, <b>V6</b> is the highest tributary of the inferior pulmonary vein. Divide it, keeping the basal veins.</p>', 'vein_keep': ['pv-inferior'], 'vein_side': [-1, -0.6, 0.1],
                          'bronchus': '<p><b>B6</b> leaves the back of the lower lobe bronchus just below the secondary carina. Clamp it and inflate: the <b>basal segments must ventilate</b>. Fire.</p>', 'bronchus_keep': ['br-lll', 'br-lul']})

# ==================================================================================================== open thoracotomy, for every lobectomy
def thoracotomy_step(op, side, upper=True):
    sd = side[0]
    ribs = [f'rib-{i}-{sd}' for i in range(3, 9) if has(f'rib-{i}-{sd}')]
    return {'id': f'{op}-thor', 'phase': 'Setup', 'title': 'Posterolateral thoracotomy, 5th intercostal space',
            'body': f'<p>{"Right" if side == "left" else "Left"} lateral decubitus, table flexed, the arm forward. The incision curves from the <b>anterior axillary line</b> to a point midway between the <b>tip of the scapula</b> and the spine.</p>'
                    '<p>Divide <b>latissimus dorsi</b>, spare and retract <b>serratus anterior</b>, count the ribs from above under the scapula, and enter the chest over the <b>upper border of the 6th rib</b> '
                    'so the neurovascular bundle under the 5th is spared. Open the space with the <b>rib spreader</b>, slowly.</p>',
            'view': {'frame': [f'incision-{sd}', *ribs[1:5]], 'dir': [-1 if side == 'left' else 1, -0.25, 0.3], 'pad': 1.2},
            'labels': [f'incision-{sd}'],
            'action': {'kind': 'thoracotomy', 'label': 'Open the chest', 'port': f'thor-{sd}', 'incision': f'incision-{sd}', 'ribs': [f'rib-5-{sd}', f'rib-6-{sd}'], 'show': ribs},
            'ct': ct(LM[f'thor-{sd}'], 'axial', 'lung')}


def open_version(steps, op, side):
    """the fissure-first sequence done through a thoracotomy: every instrument comes through the incision, and each view
    looks in through it"""
    sd = side[0]; T = V(LM[f'thor-{sd}']); out = []
    for s_ in steps:
        if s_['phase'] == 'Setup': continue
        c = copy.deepcopy(s_); c['id'] = f'{op}-{s_["id"].split("-", 1)[1]}'
        if 'action' in c: c['action']['port'] = f'thor-{sd}'
        if 'eye' in c['view']:
            # look in along the line from outside the wound, through the spread ribs, to the target
            tgt = V(c['view']['target']); L_ = np.linalg.norm(T - tgt)
            c['view'] = {'eye': R(tgt + (T - tgt) / L_ * max(135.0, L_ + 35.0)), 'target': R(tgt)}   # through the gap between the blades
        out.append(c)
    return [out[0], thoracotomy_step(op, side), *out[1:]]

sources = [
    {'title': 'Hansen HJ, Petersen RH. Video-assisted thoracoscopic lobectomy using a standardized three-port anterior approach: the Copenhagen experience. Ann Cardiothorac Surg 2012;1(1):70-76', 'url': 'https://doi.org/10.3978/j.issn.2225-319X.2012.04.15'},
    {'title': 'McElnay P, Casali G, Batchelor T, West D. Adopting a standardized anterior approach significantly increases VATS lobectomy rates. Eur J Cardiothorac Surg 2014;46(1):100', 'url': 'https://academic.oup.com/ejcts/article/46/1/100/394433'},
    {'title': 'Rusch VW, et al. The IASLC lung cancer staging project: a proposal for a new international lymph node map. J Thorac Oncol 2009', 'url': 'https://pubmed.ncbi.nlm.nih.gov/19357537'},
    {'title': 'Lim E, et al. Video-assisted thoracoscopic versus open lobectomy in patients with early-stage lung cancer (VIOLET): a randomised controlled trial. Lancet Oncol 2022', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=VIOLET+video-assisted+thoracoscopic+versus+open+lobectomy+Lim+2022'},
    {'title': 'Saji H, et al. Segmentectomy versus lobectomy in small-sized peripheral non-small-cell lung cancer (JCOG0802/WJOG4607L). Lancet 2022;399:1607-17', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=JCOG0802+segmentectomy+versus+lobectomy+Saji+2022'},
    {'title': 'Altorki N, et al. Lobar or sublobar resection for peripheral stage IA non-small-cell lung cancer (CALGB 140503). N Engl J Med 2023;388:489-98', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=CALGB+140503+lobar+or+sublobar+resection+Altorki+2023'},
    {'title': 'Wasserthal J, et al. TotalSegmentator. Radiol Artif Intell 2023', 'url': 'https://doi.org/10.1148/ryai.230024'},
]
# every operative step: the intrapulmonary trees and the spine recede so the hilar structures read clearly
BASE = {'lul-arteries': 0.16, 'lul-veins': 0.16, 'lul-bronchi': 0.2, **{f'vert-t{i}': 0.22 for i in range(2, 11)}}
OPEN = {'lul': open_version(posterior, 'lo', 'left'), 'lll': open_version(lll_fissure_first, 'llo', 'left')}
for sd, st_ in PN.items(): OPEN[f'pn{sd[0]}'] = open_version(artery_first(st_), f'pn{sd[0]}o', sd)
SEGS = {}
if SEG_OK:
    SEGS = {'lingula': lingulectomy, 'lul-updiv': trisegmentectomy, **({'s6': s6seg} if s6seg else {})}
if RUL_OK:
    OPEN.update({'rul': open_version(rul_posterior, 'ro', 'right'), 'rll': open_version(rll_fissure_first, 'rlo', 'right'), 'rml': open_version(rml_fissure_first, 'mo', 'right')})
for steps in (anterior, posterior, lll_fissure_first, lll_hilum_first, *OPEN.values(), *PN.values(), *SEGS.values(), *((rul_anterior, rul_posterior, rll_fissure_first, rll_hilum_first, rml_anterior, rml_fissure_first) if RUL_OK else ())):
    for s in steps:
        if s['phase'] != 'Setup':
            s['opacity'] = {**BASE, **s.get('opacity', {})}
            # clean view: only what the step is about. Spine, nerves and intrapulmonary trees stay hidden unless named.
            named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', []))
            quiet = [f'vert-t{i}' for i in range(2, 11)] + [i for i in ('n-phrenic', 'n-vagus', 'n-rln', 'lig-art', 'esophagus', 'svc', 'lbcv', 'ipl', 'ipl-r', 'n-phrenic-r', 'n-vagus-r') if i not in named]
            quiet += ['lul-arteries', 'lul-veins', 'lul-bronchi', 'rul-arteries', 'rul-veins', 'rul-bronchi']
            s['hide'] = [i for i in quiet if has(i)]
        for k in ('ct', 'retract'):
            if s.get(k) is None: s.pop(k, None)
        for k in ('highlight', 'danger', 'labels', 'show'):
            if k in s: s[k] = [i for i in s[k] if has(i)]
        if 'action' in s and 'ids' in s['action']: s['action']['ids'] = [i for i in s['action']['ids'] if has(i)]
procs = {
    'lul-anterior': {'id': 'vats-lul-anterior', 'op': 'lul', 'opName': 'Left upper lobectomy', 'side': 'left', 'name': 'VATS left upper lobectomy', 'approach': 'Anterior approach', 'summary': 'Hilum first: vein, truncus, bronchus, remaining arteries, fissure last.',
                 'sequence': [{'label': 'Anatomy', 'kind': 'other'}, {'label': 'Superior vein', 'kind': 'vein'}, {'label': 'Truncus', 'kind': 'artery'}, {'label': 'Bronchus', 'kind': 'bronchus'},
                              {'label': 'A2 + lingular', 'kind': 'artery'}, {'label': 'Fissure', 'kind': 'fissure'}, {'label': 'Specimen', 'kind': 'other'}],
                 'ports': [], 'steps': anterior, 'sources': sources},
    'lul-posterior': {'id': 'vats-lul-posterior', 'op': 'lul', 'opName': 'Left upper lobectomy', 'side': 'left', 'name': 'VATS left upper lobectomy', 'approach': 'Posterior approach', 'summary': 'Fissure first: arteries in the fissure, truncus, bronchus, vein last.',
                  'sequence': [{'label': 'Anatomy', 'kind': 'other'}, {'label': 'Fissure', 'kind': 'fissure'}, {'label': 'A2 + lingular', 'kind': 'artery'}, {'label': 'Truncus', 'kind': 'artery'},
                               {'label': 'Bronchus', 'kind': 'bronchus'}, {'label': 'Vein', 'kind': 'vein'}, {'label': 'Specimen', 'kind': 'other'}],
                  'ports': [], 'steps': posterior, 'sources': sources},
}
procs['lll-fissure'] = {'id': 'vats-lll-fissure', 'op': 'lll', 'opName': 'Left lower lobectomy', 'side': 'left', 'name': 'VATS left lower lobectomy', 'approach': 'Fissure first',
                        'summary': 'Ligament, fissure, A6 and basal trunk, inferior vein, bronchus.',
                        'sequence': [{'label': 'Anatomy', 'kind': 'other'}, {'label': 'Ligament', 'kind': 'other'}, {'label': 'Fissure', 'kind': 'fissure'}, {'label': 'A6 + basal', 'kind': 'artery'},
                                     {'label': 'Inferior vein', 'kind': 'vein'}, {'label': 'Bronchus', 'kind': 'bronchus'}, {'label': 'Specimen', 'kind': 'other'}],
                        'ports': [], 'steps': lll_fissure_first, 'sources': sources}
procs['lll-hilum'] = {'id': 'vats-lll-hilum', 'op': 'lll', 'opName': 'Left lower lobectomy', 'side': 'left', 'name': 'VATS left lower lobectomy', 'approach': 'Hilum first (fissureless)',
                      'summary': 'Ligament, inferior vein, bronchus, A6 and basal trunk, fissure last.',
                      'sequence': [{'label': 'Anatomy', 'kind': 'other'}, {'label': 'Ligament', 'kind': 'other'}, {'label': 'Inferior vein', 'kind': 'vein'}, {'label': 'Bronchus', 'kind': 'bronchus'},
                                   {'label': 'A6 + basal', 'kind': 'artery'}, {'label': 'Fissure', 'kind': 'fissure'}, {'label': 'Specimen', 'kind': 'other'}],
                      'ports': [], 'steps': lll_hilum_first, 'sources': sources}
if RUL_OK:
    procs['rul-anterior'] = {'id': 'vats-rul-anterior', 'op': 'rul', 'opName': 'Right upper lobectomy', 'side': 'right', 'name': 'VATS right upper lobectomy', 'approach': 'Anterior approach',
                             'summary': 'Upper lobe veins, truncus, bronchus, ascending arteries, fissures last.',
                             'sequence': [{'label': 'Anatomy', 'kind': 'other'}, {'label': 'Upper lobe veins', 'kind': 'vein'}, {'label': 'Truncus', 'kind': 'artery'}, {'label': 'Bronchus', 'kind': 'bronchus'},
                                          {'label': 'A2 + A3', 'kind': 'artery'}, {'label': 'Fissures', 'kind': 'fissure'}, {'label': 'Specimen', 'kind': 'other'}],
                             'ports': [], 'steps': rul_anterior, 'sources': sources}
    procs['rul-posterior'] = {'id': 'vats-rul-posterior', 'op': 'rul', 'opName': 'Right upper lobectomy', 'side': 'right', 'name': 'VATS right upper lobectomy', 'approach': 'Posterior approach',
                              'summary': 'Fissure junction, ascending arteries, bronchus, truncus, upper lobe veins.',
                              'sequence': [{'label': 'Anatomy', 'kind': 'other'}, {'label': 'Fissure', 'kind': 'fissure'}, {'label': 'A2 + A3', 'kind': 'artery'}, {'label': 'Bronchus', 'kind': 'bronchus'},
                                           {'label': 'Truncus', 'kind': 'artery'}, {'label': 'Upper lobe veins', 'kind': 'vein'}, {'label': 'Specimen', 'kind': 'other'}],
                              'ports': [], 'steps': rul_posterior, 'sources': sources}
if RUL_OK:
    seq = lambda *xs: [{'label': l, 'kind': k} for l, k in xs]
    procs['rll-fissure'] = {'id': 'vats-rll-fissure', 'op': 'rll', 'opName': 'Right lower lobectomy', 'side': 'right', 'name': 'VATS right lower lobectomy', 'approach': 'Fissure first',
                            'summary': 'Ligament, fissure, A6 and basal trunk, inferior vein, bronchus.', 'ports': [], 'steps': rll_fissure_first, 'sources': sources,
                            'sequence': seq(('Anatomy', 'other'), ('Ligament', 'other'), ('Fissure', 'fissure'), ('A6 + basal', 'artery'), ('Inferior vein', 'vein'), ('Bronchus', 'bronchus'), ('Specimen', 'other'))}
    procs['rll-hilum'] = {'id': 'vats-rll-hilum', 'op': 'rll', 'opName': 'Right lower lobectomy', 'side': 'right', 'name': 'VATS right lower lobectomy', 'approach': 'Hilum first (fissureless)',
                          'summary': 'Ligament, inferior vein, bronchus, A6 and basal trunk, fissure last.', 'ports': [], 'steps': rll_hilum_first, 'sources': sources,
                          'sequence': seq(('Anatomy', 'other'), ('Ligament', 'other'), ('Inferior vein', 'vein'), ('Bronchus', 'bronchus'), ('A6 + basal', 'artery'), ('Fissure', 'fissure'), ('Specimen', 'other'))}
    procs['rml-anterior'] = {'id': 'vats-rml-anterior', 'op': 'rml', 'opName': 'Right middle lobectomy', 'side': 'right', 'name': 'VATS right middle lobectomy', 'approach': 'Anterior approach',
                             'summary': 'Middle lobe vein, bronchus, artery, fissures last.', 'ports': [], 'steps': rml_anterior, 'sources': sources,
                             'sequence': seq(('Anatomy', 'other'), ('Vein', 'vein'), ('Bronchus', 'bronchus'), ('Artery', 'artery'), ('Fissures', 'fissure'), ('Specimen', 'other'))}
    procs['rml-fissure'] = {'id': 'vats-rml-fissure', 'op': 'rml', 'opName': 'Right middle lobectomy', 'side': 'right', 'name': 'VATS right middle lobectomy', 'approach': 'Fissure first',
                            'summary': 'Fissure junction, artery, vein, bronchus, horizontal fissure.', 'ports': [], 'steps': rml_fissure_first, 'sources': sources,
                            'sequence': seq(('Anatomy', 'other'), ('Fissure', 'fissure'), ('Artery', 'artery'), ('Vein', 'vein'), ('Bronchus', 'bronchus'), ('Horizontal fissure', 'fissure'), ('Specimen', 'other'))}
seq = lambda *xs: [{'label': l, 'kind': k} for l, k in xs]
for sd, st_ in PN.items():
    procs[f'pn{sd[0]}-vats'] = {'id': f'vats-pn-{sd}', 'op': f'pn{sd[0]}', 'opName': f'{sd.capitalize()} pneumonectomy', 'side': sd, 'name': f'VATS {sd} pneumonectomy', 'approach': 'VATS (veins first)',
                                'summary': 'Ligament, inferior vein, superior vein, main artery, main bronchus.', 'ports': [], 'steps': st_, 'sources': sources,
                                'sequence': seq(('Anatomy', 'other'), ('Ligament', 'other'), ('Inferior vein', 'vein'), ('Superior vein', 'vein'), ('Main PA', 'artery'), ('Main bronchus', 'bronchus'), ('Specimen', 'other'))}
SEGNAME = {'lingula': ('Lingulectomy', 'Lingula (S4+5)'), 'lul-updiv': ('Upper division segmentectomy', 'S1+2, S3'), 's6': ('S6 segmentectomy', 'Superior segment')}
for k, st_ in SEGS.items():
    procs[f'seg-{k}'] = {'id': f'vats-seg-{k}', 'op': f'seg-{k}', 'opName': SEGNAME[k][0], 'side': 'left', 'name': SEGNAME[k][0], 'approach': 'VATS, fissure first',
                         'summary': 'Fissure, segmental artery, vein, bronchus, inflation–deflation line, intersegmental plane.', 'ports': [], 'steps': st_, 'sources': sources,
                         'sequence': seq(('Anatomy', 'other'), ('Fissure', 'fissure'), ('Artery', 'artery'), ('Vein', 'vein'), ('Bronchus', 'bronchus'), ('Plane', 'fissure'), ('Specimen', 'other'))}
# open thoracotomy, one per operation, following that operation's fissure-first sequence (artery first for pneumonectomy)
for op_, steps_ in OPEN.items():
    base = next(v for v in procs.values() if v['op'] == op_ and v['approach'] in ('Posterior approach', 'Fissure first', 'VATS (veins first)'))
    procs[f'{op_}-open'] = {**base, 'id': f'open-{op_}', 'approach': 'Open thoracotomy', 'name': base['opName'] + ', open', 'steps': steps_,
                            'summary': 'Posterolateral thoracotomy, then ' + base['summary'][0].lower() + base['summary'][1:]}
    if op_.startswith('pn'):
        procs[f'{op_}-open']['summary'] = 'Posterolateral thoracotomy; artery first: main PA, superior vein, inferior vein, bronchus.'
        procs[f'{op_}-open']['sequence'] = seq(('Anatomy', 'other'), ('Ligament', 'other'), ('Main PA', 'artery'), ('Superior vein', 'vein'), ('Inferior vein', 'vein'), ('Main bronchus', 'bronchus'), ('Specimen', 'other'))
# ==================================================================================================== chest trauma
TR_OK = all(k in LM for k in ('thor-al', 'thor-ar', 'aorta-clamp', 'wound-rv', 'tract-a', 'tract-b', 'hilum-l', 'hilum-l-axis', 'sternotomy')) and has('incision-al')
TRAUMA = {}
if TR_OK:
    T_AL, T_AR, STN = V(LM['thor-al']), V(LM['thor-ar']), V(LM['sternotomy'])
    WND, ACL = V(LM['wound-rv']), V(LM['aorta-clamp'])
    TA_, TB_ = V(LM['tract-a']), V(LM['tract-b'])
    HIL, HAX = V(LM['hilum-l']), V(LM['hilum-l-axis'])
    PCA, PCB = V(LM['pericardiotomy-a']), V(LM['pericardiotomy-b'])
    LUNG_L = [i for i in ('lul', 'lll', 'fissure') if has(i)]
    L_TWIST = [i for i in ('lul', 'lll', 'fissure', 'lul-arteries', 'lul-veins', 'lul-bronchi', 'lll-arteries', 'lll-veins', 'lll-bronchi', 'pa-truncus-anterior', *POST,
                           'pa-lingular', 'pa-a6', 'pa-basal-trunk', 'br-lingular', 'br-upper-div', 'br-b6', 'pv-lingular', 'pv-upper-div', 'pv-v6', 'tract-l') if has(i)]
    TSRC = [
        {'title': 'Seamon MJ, et al. An evidence-based approach to patient selection for emergency department thoracotomy: a practice management guideline from the Eastern Association for the Surgery of Trauma. J Trauma Acute Care Surg 2015;79(1):159-173',
         'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Seamon+emergency+department+thoracotomy+Eastern+Association+2015'},
        {'title': 'Burlew CC, et al. Western Trauma Association critical decisions in trauma: resuscitative thoracotomy. J Trauma Acute Care Surg 2012;73(6):1359-1363',
         'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Burlew+Western+Trauma+Association+resuscitative+thoracotomy'},
        {'title': 'Simms ER, et al. Bilateral anterior thoracotomy (clamshell incision) is the ideal emergency thoracotomy incision: an anatomic study. World J Surg 2013;37(6):1277-1285',
         'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Simms+clamshell+incision+ideal+emergency+thoracotomy'},
        {'title': 'Wall MJ Jr, Hirshberg A, Mattox KL. Pulmonary tractotomy with selective vascular ligation for penetrating injuries to the lung. Am J Surg 1994;168(6):665-669',
         'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Wall+Hirshberg+Mattox+pulmonary+tractotomy'},
        {'title': 'Wilson A, Wall MJ Jr, Maxson R, Mattox K. The pulmonary hilum twist as a thoracic damage control procedure. Am J Surg 2003;186(1):49-52',
         'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=pulmonary+hilum+twist+thoracic+damage+control'},
    ]

    def via(tgt, T=T_AL, mn=330.0, extra=150.0):
        """look in through the anterolateral wound (past the seated spreader) at the target"""
        t = V(tgt); L_ = np.linalg.norm(T - t); return {'eye': R(t + (T - t) / L_ * max(mn, L_ + extra)), 'target': R(t)}

    def al_entry(op, sd='l', seq=1):
        L = sd == 'l'
        ribs = [f'rib-{i}-{sd}' for i in range(3, 9) if has(f'rib-{i}-{sd}')]
        return {'id': f'{op}-entry-{sd}', 'phase': 'Entry', 'seq': seq, 'title': f'{"Left" if L else "Right"} anterolateral thoracotomy, 5th space',
                'body': ('<p>Supine, the left arm up above the head. Incise along the <b>5th intercostal space</b> from the <b>sternal edge</b> to the <b>mid-axillary line</b>: below the nipple in a man, '
                         'in the inframammary fold in a woman (lift the breast). Curve slightly upward toward the axilla, following the rib.</p>'
                         '<p>Cut through the pectoralis and serratus in one pass, then divide the intercostal muscles on the <b>upper border of the 6th rib</b> with scissors (heavy Mayo), pushing the lung away. '
                         'Place the <b>rib spreader with its handle toward the axilla</b>, so it does not block an extension across the sternum.</p>'
                         '<p>Medially the <b>internal mammary artery</b> runs 1–2 cm from the sternal edge: it is cut if the incision goes onto the sternum, and bleeds once there is a pressure.</p>'
                         if L else
                         '<p>The same incision on the right: 5th space, sternal edge to mid-axillary line, intercostals divided on the upper border of the 6th rib. '
                         'A second spreader goes in with its handle toward the right axilla.</p>'
                         '<p>The <b>right internal mammary artery</b> crosses the line of the sternal cut 1–2 cm from the edge.</p>'),
                'view': {'frame': [f'incision-a{sd}', *ribs[1:4]], 'dir': [-0.75 if L else 0.75, 0.7, 0.2], 'pad': 1.15},
                'labels': [f'incision-a{sd}'], 'danger': [f'ima-{sd}'], 'show': [f'ima-{sd}'],
                'action': {'kind': 'thoracotomy', 'label': 'Open the chest', 'port': f'thor-a{sd}', 'incision': f'incision-a{sd}', 'ribs': [f'rib-5-{sd}', f'rib-6-{sd}'], 'show': ribs},
                'ct': ct(R(T_AL if L else T_AR), 'axial', 'lung')}

    def down(tgt, d=(-0.45, 1.0, 0.12), dist=300.0):
        """the surgeon's view from above the supine patient, down through the anterolateral wound"""
        d = V(d) / np.linalg.norm(d); return {'eye': R(V(tgt) + d * dist), 'target': R(tgt)}

    LUNG_BACK = {'ids': [i for i in ('lul', 'lll', 'fissure') if has(i)], 'offset': [-10, -30, 0], 'opacity': 0.22}
    peri_path = [R(PCA + V([0, 4, 0])), R((2 * PCA + PCB) / 3 + V([0, 5, 0])), R((PCA + 2 * PCB) / 3 + V([0, 5, 0])), R(PCB + V([0, 4, 0]))]
    pericardium = lambda op, seq=2: {
        'id': f'{op}-pericardium', 'phase': 'Pericardium', 'seq': seq, 'title': 'Pericardiotomy, anterior to the phrenic nerve',
        'body': '<p>Push the collapsed lung back. The <b>phrenic nerve</b> runs down the side of the pericardium. Pick up the pericardium with toothed forceps <b>anterior to the nerve</b>, nick it with scissors, '
                'and open it longitudinally, <b>parallel to the nerve</b>, from the apex to the root of the aorta.</p>'
                '<p>A tense, blue, non-pulsatile pericardium is <b>tamponade</b>: scoop out the clot and deliver the heart into the wound.</p>',
        'view': down((PCA + PCB) / 2), 'retract': LUNG_BACK, 'opacity': {'heart': 1.0, 'laa': 1.0},
        'highlight': ['heart'], 'danger': ['n-phrenic'], 'labels': ['laa', 'aorta'],
        'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Open the pericardium', 'port': 'thor-al', 'path': peri_path, 'show': ['pericardiotomy']},
        'ask': ask('Where do you open the pericardium?', 'Anterior to the phrenic nerve, parallel to it',
                   'The nerve runs on the lateral pericardium; a longitudinal cut in front of it spares it. A transverse cut divides it.', 'Posterior to the phrenic nerve', 'Transversely across the nerve'),
        'ct': ct(R(WND), 'axial')}
    stitch_path = [R(WND + V([dx, 1.0, 0])) for dx in (-6.0, 0.0, 6.0)]
    cardiorrhaphy = lambda op, seq=3: {
        'id': f'{op}-cardio', 'phase': 'Heart', 'seq': seq, 'title': 'Cardiorrhaphy: right ventricular stab wound',
        'body': '<p>Put a <b>finger on the hole</b> first. The right ventricle is the chamber most often hit, as it lies under the sternum.</p>'
                '<p>Close with <b>3-0 or 4-0 polypropylene</b>, <b>pledgeted horizontal mattress</b> sutures, sliding the finger along as each stitch goes in. Tie gently: ventricular muscle tears.</p>'
                '<p>Bridges while you get ready: a <b>Foley catheter</b> through the hole, balloon inflated and pulled up gently; or a skin stapler across a clean wound.</p>',
        'view': down(WND, dist=240), 'retract': LUNG_BACK, 'show': ['wound-rv', 'pericardiotomy'], 'opacity': {'heart': 1.0},
        'highlight': ['wound-rv'], 'danger': ['heart'],
        'action': {'kind': 'suture', 'label': 'Place pledgeted sutures', 'port': 'thor-al', 'path': stitch_path, 'normal': [0, 1, 0], 'axis': [1, 0, 0], 'remove': ['wound-rv']},
        'pearl': 'Near a coronary artery, pass the mattress stitch <b>beneath</b> the artery so it is not tied off. Atrial wounds: a side-biting clamp, then a running suture.',
        'ct': ct(R(WND), 'axial')}

    rt = [
        {'id': 'rt-decide', 'phase': 'Decision', 'seq': 0, 'title': 'Who gets a resuscitative thoracotomy',
         'body': '<p>The goals: <b>release tamponade</b>, <b>control bleeding</b> from the heart or lung, <b>cross-clamp the descending aorta</b>, <b>internal cardiac massage</b>, and stop <b>air embolism</b>.</p>'
                 '<p>Best results after a <b>penetrating chest wound</b> with signs of life, poorest after <b>blunt</b> injury. Commonly used cut-offs: thoracotomy is futile after about <b>15 minutes of CPR</b> for penetrating injury, '
                 'and after about <b>10 minutes</b> for blunt injury, without a response. Follow your unit\'s protocol.</p>',
         'view': {'frame': ['heart', 'lul', 'lll', 'aorta'], 'dir': [-0.55, 1, 0.25], 'pad': 1.0}, 'spin': True, 'opacity': {'lul': 0.3, 'lll': 0.3, 'fissure': 0.2, 'heart': 0.9},
         'labels': ['heart', 'aorta', 'lul', 'lll'],
         'ask': ask('A man with a stab wound to the left chest loses his pulse on arrival; CPR for 6 minutes. What now?', 'Left anterolateral thoracotomy in the resuscitation room',
                    'Penetrating chest injury, pulseless, CPR well under 15 minutes: the best indication there is (EAST: strong recommendation).', 'Keep up CPR and give blood only', 'Pericardiocentesis and wait'),
         'ct': ct(R(WND), 'axial')},
        al_entry('rt'),
        pericardium('rt'),
        cardiorrhaphy('rt'),
        {'id': 'rt-aorta', 'phase': 'Aorta', 'seq': 4, 'title': 'Cross-clamp the descending aorta',
         'body': '<p>Lift the left lung <b>up and forward</b>. Just above the diaphragm, open the mediastinal pleura over the aorta and separate it from the <b>oesophagus</b> with the fingers: '
                 'the aorta is the firm tube on the spine; a <b>nasogastric tube</b> makes the oesophagus easy to feel.</p>'
                 '<p>Place a vascular clamp across the aorta. It diverts what cardiac output there is to the heart and brain and cuts bleeding below the diaphragm. '
                 '<b>Note the time</b>: visceral and spinal ischaemia build after about 30 minutes.</p>',
         'view': down(ACL, d=(-0.85, 0.75, 0.2), dist=300), 'retract': {'ids': [i for i in ('lul', 'lll', 'fissure') if has(i)], 'offset': [4, 34, 40], 'opacity': 0.15},
         'opacity': {'heart': 0.18, 'laa': 0.18}, 'show': ['esophagus'],
         'highlight': ['aorta'], 'danger': ['esophagus'], 'labels': ['heart'],
         'action': {'kind': 'clamp', 'label': 'Apply the aortic clamp', 'port': 'thor-al', 'at': R(ACL), 'axis': [0, 0, 1], 'radius': 12, 'jawLen': 55},
         'ask': ask('Where does the oesophagus lie relative to the lower descending aorta?', 'Anterior and to its right, closely applied',
                    'Low in the chest the oesophagus lies in front of and to the right of the aorta; a blind clamp can take both.', 'Behind it, on the spine', 'Lateral to it, under the lung'),
         'ct': ct(R(ACL), 'axial')},
        {'id': 'rt-massage', 'phase': 'Heart', 'seq': 5, 'title': 'Internal massage and defibrillation',
         'body': '<p><b>Two-handed</b> massage: the heart between the flat palms, compressing from the <b>apex toward the base</b>. Not the fingertips: they go through the ventricle.</p>'
                 '<p>Fill the heart first: an empty heart gains nothing from massage. For VF, internal paddles on either side of the ventricles, starting at about <b>10–20 J</b>.</p>',
         'view': down(V(S['heart']['centroid']), dist=330), 'retract': LUNG_BACK, 'opacity': {'heart': 1.0},
         'highlight': ['heart'], 'labels': ['aorta'],
         'action': {'kind': 'massage', 'label': 'Massage the heart', 'port': 'thor-al', 'ids': ['heart']},
         'ct': ct('heart', 'axial')},
        {'id': 'rt-next', 'phase': 'Next', 'seq': 6, 'title': 'Extend, or go to theatre',
         'body': '<p>Cannot reach the right side of the heart, the right lung or the great vessels? <b>Extend across the sternum</b> into a clamshell (next operation in the menu).</p>'
                 '<p>With a circulation back: go to theatre; release the aortic clamp slowly with the anaesthetist ready; <b>ligate both internal mammary arteries</b>; look for bleeding you could not see at a pressure of zero.</p>',
         'view': {'frame': ['heart', 'lul', 'lll', 'aorta'], 'dir': [-0.6, 1, 0.3], 'pad': 1.05}, 'retract': LUNG_BACK, 'labels': ['heart', 'aorta'],
         'ct': ct('heart', 'axial')},
    ]
    TRAUMA['rt'] = ('Resuscitative thoracotomy', 'Left anterolateral (ED)', 'left', 'Anterolateral thoracotomy, pericardiotomy, cardiorrhaphy, aortic cross-clamp, massage.', rt,
                    seq(('Decide', 'other'), ('Entry', 'other'), ('Pericardium', 'other'), ('Heart wound', 'vein'), ('Aortic clamp', 'artery'), ('Massage', 'other'), ('Next', 'other')))

    # ---------------------------------------------------------------- clamshell
    ribs_lid = [f'rib-{i}-{s}' for i in range(1, 6) for s in 'lr' if has(f'rib-{i}-{s}')]
    vt = min((k for k in S if k.startswith('vert-t')), key=lambda k: abs(S[k]['centroid'][2] - STN[2]))
    HINGE = R([0.0, S[vt]['centroid'][1], STN[2]])
    ima_mid = mean(['ima-l', 'ima-r'])
    cs_view = lambda tgt, d=(0, 1, -0.35), dist=260: {'eye': R(V(tgt) + V(d) / np.linalg.norm(d) * dist), 'target': R(tgt)}
    cs = [
        {'id': 'cs-anat', 'phase': 'Decision', 'seq': 0, 'title': 'Clamshell: when and what it opens',
         'body': '<p>Both 5th-space anterolateral incisions, joined by a <b>transverse sternotomy</b>. The chest wall above lifts like a lid, opening <b>both pleural cavities</b>, the <b>whole anterior heart</b>, '
                 'the pulmonary hila and, with traction, the arch and its branches.</p>'
                 '<p>Use it for a <b>wound to the right chest</b>, a <b>precordial wound</b> in an arrested patient, or when a left anterolateral thoracotomy cannot reach the injury. '
                 'Many trauma teams start with a clamshell in the arrested patient.</p>',
         'view': {'frame': ['heart', 'lul', 'rul', 'lll', 'rll'], 'dir': [0, 1, 0.25], 'pad': 1.0}, 'show': ['incision-cs', 'sternum', 'skin'], 'opacity': {'skin': 0.35, 'lul': 0.3, 'lll': 0.3, 'rul': 0.3, 'rml': 0.3, 'rll': 0.3, 'sternum': 0.8},
         'labels': ['incision-cs', 'sternum', 'heart'], 'ct': ct(R(STN), 'axial', 'bone')},
        {**al_entry('cs', 'l', 1), 'title': 'Left anterolateral thoracotomy, 5th space'},
        al_entry('cs', 'r', 2),
        {'id': 'cs-saw', 'phase': 'Sternum', 'seq': 3, 'title': 'Divide the sternum transversely',
         'body': '<p>Join the two incisions across the sternum at the same level. Divide the bone with a <b>Gigli saw</b> passed behind it, a <b>Lebsche knife</b> struck with a mallet, or heavy trauma shears.</p>'
                 '<p>Stay in the same intercostal level on both sides so the lid lifts evenly.</p>',
         'view': cs_view(STN, (0, 1, -0.5), 230), 'show': ['sternum', 'ima-l', 'ima-r'], 'opacity': {'sternum': 0.9},
         'highlight': ['sternum'], 'danger': ['ima-l', 'ima-r', 'heart'],
         'action': {'kind': 'saw', 'label': 'Divide the sternum', 'port': 'sternotomy', 'ids': ['sternum']},
         'ct': ct(R(STN), 'sagittal', 'bone')},
        {'id': 'cs-ima', 'phase': 'Sternum', 'seq': 4, 'title': 'Internal mammary arteries: ligate both ends',
         'body': '<p>Both <b>internal mammary arteries</b> are divided with the sternum. In arrest they do not bleed; once there is a pressure they bleed briskly, from <b>both ends</b>.</p>'
                 '<p>Find each end 1–2 cm from the sternal edge on the cut surface and <b>ligate or clip</b> all four. A missed mammary is a common cause of return to theatre.</p>',
         'view': cs_view(V(ima_mid), (0, 0.8, -1), 200), 'show': ['sternum', 'ima-l', 'ima-r'], 'opacity': {'sternum': 0.6},
         'highlight': ['ima-l', 'ima-r'],
         'action': {'kind': 'ligate', 'label': 'Ligate both mammaries', 'port': 'sternotomy', 'ids': ['ima-l', 'ima-r']},
         'ask': ask('After a clamshell, the heart restarts and the chest fills with blood from the wound edges. First suspect?', 'The internal mammary arteries',
                    'Both are cut with the sternum and do not bleed until there is a pressure.', 'The intercostal veins', 'The pericardiophrenic vessels'),
         'ct': ct('ima-l', 'axial')},
        {'id': 'cs-lid', 'phase': 'Exposure', 'seq': 5, 'title': 'Lift the lid',
         'body': '<p>With both spreaders open, lift the upper chest wall and sternum <b>up toward the head</b>. The whole anterior mediastinum, both lungs and both hila come into view.</p>',
         'view': cs_view(V(S['heart']['centroid']), (0, 1, -0.6), 330), 'show': ['sternum', 'ima-l', 'ima-r'], 'opacity': {'sternum': 0.8, 'lul': 0.3, 'lll': 0.3, 'rul': 0.3, 'rml': 0.3, 'rll': 0.3},
         'labels': ['heart', 'sternum'],
         'action': {'kind': 'twist', 'label': 'Lift the chest wall', 'port': 'sternotomy', 'hinge': {'ids': ['sternum', 'ima-l', 'ima-r', *ribs_lid], 'pivot': HINGE, 'axis': [1, 0, 0], 'angle': 32}},
         'ct': ct(R(STN), 'sagittal', 'bone')},
        {'id': 'cs-expose', 'phase': 'Exposure', 'seq': 6, 'title': 'What you can reach',
         'body': '<p>Open the pericardium in the midline (an inverted T), clear of both phrenic nerves. You now reach the <b>whole anterior heart</b>, <b>both hila</b> for a clamp or twist, '
                 'the <b>descending aorta</b> through the left chest, and the <b>SVC and right atrium</b> through the right.</p>'
                 '<p>For the arch branches, add a vertical upper sternotomy from the midpoint of the transverse cut.</p>',
         'view': cs_view(V(S['heart']['centroid']), (0, 1, -0.45), 330), 'show': ['sternum', 'ima-l', 'ima-r', 'svc'], 'opacity': {'sternum': 0.8, 'lul': 0.3, 'lll': 0.3, 'rul': 0.3, 'rml': 0.3, 'rll': 0.3},
         'labels': ['heart', 'aorta', 'svc', 'pa-left', 'rpa', 'n-phrenic', 'n-phrenic-r'], 'danger': ['n-phrenic', 'n-phrenic-r'],
         'ct': ct('heart', 'axial')},
    ]
    TRAUMA['clamshell'] = ('Clamshell thoracotomy', 'Bilateral anterolateral', 'both', 'Both 5th-space thoracotomies, transverse sternotomy, mammaries ligated, lid lifted.', cs,
                           seq(('Decide', 'other'), ('Left', 'other'), ('Right', 'other'), ('Sternum', 'bronchus'), ('Mammaries', 'artery'), ('Lid', 'other'), ('Exposure', 'other')))

    # ---------------------------------------------------------------- pulmonary tractotomy
    tdir = (TB_ - TA_) / np.linalg.norm(TB_ - TA_); tn = np.cross(tdir, [0, 0, 1.0]); tn /= np.linalg.norm(tn)
    tr = [
        {'id': 'tr-decide', 'phase': 'Decision', 'seq': 0, 'title': 'The bleeding lung: what to do',
         'body': '<p>Most lung wounds need only a <b>chest drain</b>. Operate for continuing bleeding: commonly about <b>1500 mL</b> at insertion, or <b>200 mL an hour</b> for several hours, or shock.</p>'
                 '<p>Choose the <b>least lung resection</b> that controls it: suture (pneumonorrhaphy) for a superficial wound, <b>tractotomy</b> for a through-and-through tract, a stapled wedge at the periphery; '
                 'lobectomy or pneumonectomy only for hilar injury. Mortality climbs steeply with each step up.</p>',
         'view': {'frame': ['lll', 'tract-l'], 'dir': [-1, 0.3, 0.2], 'pad': 1.05}, 'show': ['tract-l'], 'opacity': {'lll': 0.3, 'lul': 0.25, 'fissure': 0.2},
         'highlight': ['tract-l'], 'labels': ['lll'], 'spin': True,
         'ask': ask('A missile tract runs through the left lower lobe, away from the hilum, and bleeds. Best operation?', 'Stapled tractotomy with selective ligation',
                    'It opens the tract, controls the bleeding vessels and air leaks individually, and saves the lobe.', 'Left lower lobectomy', 'Oversew the entry and exit wounds'),
         'ct': ct(R((TA_ + TB_) / 2), 'axial', 'lung')},
        al_entry('tr'),
        {'id': 'tr-find', 'phase': 'Tract', 'seq': 2, 'title': 'Find the entry and exit wounds',
         'body': '<p>Deliver the lobe into the wound. Compress it in the hand to hold the bleeding; find both holes. Pass a finger or the jaw of a clamp along the tract to check it is a single straight path, '
                 'well away from the hilum.</p><p>A tract through the hilum is not for tractotomy: control the hilum instead.</p>',
         'view': via((TA_ + TB_) / 2), 'show': ['tract-l'], 'opacity': {'lll': 0.3, 'lul': 0.2, 'fissure': 0.15},
         'highlight': ['tract-l'], 'danger': ['pa-basal-trunk', 'pv-inferior'], 'labels': ['lll'],
         'ct': ct(R((TA_ + TB_) / 2), 'axial', 'lung')},
        {'id': 'tr-staple', 'phase': 'Tract', 'seq': 3, 'title': 'Tractotomy: staple through the tract',
         'body': '<p>Pass one jaw of a linear stapler (or two long clamps) <b>through the tract</b>, the other over the thin bridge of lung above it, and fire. The tract lies open as a trough.</p>',
         'view': via((TA_ + TB_) / 2), 'show': ['tract-l'], 'opacity': {'lll': 0.35, 'lul': 0.2, 'fissure': 0.15},
         'highlight': ['tract-l'], 'danger': ['pa-basal-trunk'],
         'action': {'kind': 'staple-fissure', 'label': 'Fire through the tract', 'port': 'thor-al', 'path': [R(TA_), R(TB_)], 'normal': R(tn), 'reload': 'tissue'},
         'ct': ct(R((TA_ + TB_) / 2), 'axial', 'lung')},
        {'id': 'tr-ligate', 'phase': 'Tract', 'seq': 4, 'title': 'Selective ligation in the tract',
         'body': '<p>In the open tract, find each <b>bleeding vessel</b> and each <b>leaking bronchus</b> and tie or suture it individually (4-0 polypropylene). '
                 'Leave the tract <b>open</b>: closing it over traps blood and risks <b>air embolism</b>.</p>',
         'view': via((TA_ + TB_) / 2), 'show': ['tract-l'], 'opacity': {'lll': 0.35, 'lul': 0.2, 'fissure': 0.15},
         'highlight': ['tract-l'], 'labels': ['lll'],
         'ask': ask('Why not simply oversew the entry and exit holes of a deep tract?', 'Bleeding continues inside, and air can enter the pulmonary veins',
                    'An oversewn tract becomes a haematoma and a route for systemic air embolism.', 'It takes longer', 'It needs a larger incision'),
         'ct': ct(R((TA_ + TB_) / 2), 'axial', 'lung')},
        {'id': 'tr-close', 'phase': 'Close', 'seq': 5, 'title': 'Test and close',
         'body': '<p>Fill the chest with warm saline and inflate the lung: <b>no bubbles</b>, no bleeding. Two drains (apical and basal). Close the ribs with pericostal sutures.</p>',
         'view': {'frame': ['lll', 'lul'], 'dir': [-1, 0.4, 0.2], 'pad': 1.0}, 'show': ['tract-l'], 'opacity': {'lll': 0.6, 'lul': 0.5},
         'labels': ['lll'], 'ct': ct(R((TA_ + TB_) / 2), 'axial', 'lung')},
    ]
    TRAUMA['tract'] = ('Pulmonary tractotomy', 'Anterolateral thoracotomy', 'left', 'Find the tract, staple through it, ligate each vessel and bronchus, leave it open.', tr,
                       seq(('Decide', 'other'), ('Entry', 'other'), ('Find', 'other'), ('Staple', 'fissure'), ('Ligate', 'artery'), ('Close', 'other')))

    # ---------------------------------------------------------------- hilar control: clamp or twist
    lig_path = [R(LIG_LO + V([-3, 6, 4])), R((LIG_LO + LIG) / 2 + V([-3, 4, 0])), R(LIG + V([-3, 3, 0])), R(iv + V([-4, -4, -9]))]

    def hil_steps(pre):
        return [
            {'id': f'{pre}-decide', 'phase': 'Decision', 'seq': 0, 'title': 'When to control the whole hilum',
             'body': '<p>Hilar control is for <b>massive bleeding from the hilum or deep lung</b> that a hand cannot hold, or for <b>systemic air embolism</b> from a lung wound '
                     '(air in the coronary arteries, sudden arrest when ventilated).</p>'
                     '<p>Two ways: a <b>clamp across the whole hilum</b>, or the <b>hilar twist</b>. Both need the <b>inferior pulmonary ligament</b> divided first. '
                     'The right ventricle then pumps into one lung: expect <b>acute right heart strain</b>.</p>',
             'view': {'frame': ['pa-left', 'pv-superior', 'pv-inferior', 'br-left-main', 'lul', 'lll'], 'dir': [-1, 0.35, 0.2], 'pad': 1.0}, 'spin': True,
             'opacity': {'lul': 0.2, 'lll': 0.2, 'fissure': 0.15, 'heart': 0.4},
             'labels': ['pa-left', 'pv-superior', 'pv-inferior', 'br-left-main', 'ipl'], 'ct': ct('pa-left', 'coronal')},
            al_entry(pre),
            {'id': f'{pre}-ligament', 'phase': 'Ligament', 'seq': 2, 'title': 'Divide the inferior pulmonary ligament',
             'body': '<p>Pull the lower lobe up. Divide the <b>inferior pulmonary ligament</b> with scissors or diathermy, close to the lung, up to the <b>inferior pulmonary vein</b>. '
                     'The oesophagus and aorta lie just medial.</p><p>Now the whole hilum is free for a clamp or a twist.</p>',
             'view': via((LIG + iv) / 2), 'opacity': {'lul': 0.3, 'lll': 0.3, 'fissure': 0.2, 'heart': 0.4}, 'show': ['esophagus'],
             'highlight': ['ipl'], 'danger': ['esophagus', 'aorta', 'pv-inferior'],
             'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Divide the ligament', 'port': 'thor-al', 'remove': ['ipl'], 'path': lig_path},
             'ct': ct(R(LIG), 'coronal')},
        ]

    hc = hil_steps('hc') + [
        {'id': 'hc-clamp', 'phase': 'Hilum', 'seq': 3, 'title': 'Clamp across the hilum',
         'body': '<p>Slide a large <b>Satinsky clamp</b> round the hilum from above, the jaws taking the <b>pulmonary artery, both veins and the bronchus</b> together, as close to the mediastinum as you can. '
                 'A hand round the hilum holds it while you position the clamp.</p>'
                 '<p>Watch the right heart: if it dilates, release partially.</p>',
         'view': via(HIL), 'opacity': {'lul': 0.25, 'lll': 0.25, 'fissure': 0.2, 'heart': 0.4},
         'highlight': ['pa-left', 'pv-superior', 'pv-inferior', 'br-left-main'], 'danger': ['n-phrenic', 'n-vagus', 'heart'],
         'action': {'kind': 'clamp', 'label': 'Clamp the hilum', 'port': 'thor-al', 'at': R(HIL + HAX * 18), 'axis': R(HAX), 'radius': 16, 'jawLen': 78},
         'ct': ct('pa-left', 'coronal')},
        {'id': 'hc-after', 'phase': 'After', 'seq': 4, 'title': 'After the clamp',
         'body': '<p>The clamp buys time. Repair the injured vessel if it is proximal and simple; otherwise a <b>stapled pneumonectomy</b> (one fire across the hilum) may be all the patient can stand.</p>'
                 '<p>Trauma pneumonectomy carries a <b>very high mortality</b>, largely from right heart failure: give fluids carefully and consider inotropes early.</p>',
         'view': via(HIL), 'opacity': {'lul': 0.3, 'lll': 0.3, 'fissure': 0.2, 'heart': 0.5}, 'labels': ['pa-left', 'heart'],
         'ct': ct('pa-left', 'coronal')},
    ]
    tw_ang = -180 if HAX[0] < 0 else 180
    tw = hil_steps('tw') + [
        {'id': 'tw-twist', 'phase': 'Hilum', 'seq': 3, 'title': 'The hilar twist',
         'body': '<p>With the ligament divided, take the lower lobe in the hand and rotate it <b>forward and up over the upper lobe</b>, 180 degrees about the hilum. '
                 'The artery, veins and bronchus <b>kink</b> on themselves: bleeding and air embolism stop.</p>'
                 '<p>No clamp in the way, nothing to slip: pack laparotomy pads round the apex to hold the lung turned.</p>',
         'view': via(HIL + V([-30, 0, 0])), 'opacity': {'lul': 0.45, 'lll': 0.45, 'fissure': 0.3, 'heart': 0.4},
         'highlight': ['lll', 'lul'], 'danger': ['pa-left', 'pv-inferior'],
         'action': {'kind': 'twist', 'label': 'Twist the lung', 'port': 'thor-al', 'hinge': {'ids': L_TWIST, 'pivot': R(HIL), 'axis': R(HAX), 'angle': tw_ang}},
         'ask': ask('What must be done before a hilar twist?', 'Divide the inferior pulmonary ligament',
                    'The ligament tethers the lower lobe to the mediastinum; the lung cannot turn until it is cut.', 'Divide the pulmonary artery', 'Open the fissure'),
         'ct': ct('pa-left', 'coronal')},
        {'id': 'tw-after', 'phase': 'After', 'seq': 4, 'title': 'Damage control, then back',
         'body': '<p>Leave the lung twisted and packed; close temporarily or pack the chest. Back in theatre once warm, not acidotic and not coagulopathic: untwist, then decide between <b>repair</b> and a <b>stapled pneumonectomy</b>.</p>',
         'view': via(HIL + V([-30, 0, 0])), 'opacity': {'lul': 0.45, 'lll': 0.45, 'fissure': 0.3, 'heart': 0.5}, 'labels': ['pa-left', 'heart'],
         'ct': ct('pa-left', 'coronal')},
    ]
    TRAUMA['hilar'] = [('Hilar control', 'Hilar clamp', 'left', 'Anterolateral thoracotomy, ligament, Satinsky clamp across the hilum.', hc,
                        seq(('Decide', 'other'), ('Entry', 'other'), ('Ligament', 'other'), ('Clamp', 'artery'), ('After', 'other'))),
                       ('Hilar control', 'Hilar twist', 'left', 'Anterolateral thoracotomy, ligament, 180 degree twist of the lung about its hilum.', tw,
                        seq(('Decide', 'other'), ('Entry', 'other'), ('Ligament', 'other'), ('Twist', 'artery'), ('After', 'other')))]

    # cardiorrhaphy on its own: the cardiac box, exposure and repair
    cr = [
        {'id': 'cr-box', 'phase': 'Anatomy', 'seq': 0, 'title': 'The cardiac box',
         'body': '<p>A wound between the <b>clavicles</b>, the <b>mid-clavicular lines</b> and the <b>costal margins</b> is a heart wound until proved otherwise.</p>'
                 '<p>The <b>right ventricle</b> is the most anterior chamber and the most often hit, then the left ventricle; atrial wounds are rarer and bleed less. '
                 'Beck\'s triad (hypotension, raised venous pressure, muffled heart sounds) is often incomplete: scan the pericardium (FAST).</p>',
         'view': {'frame': ['heart'], 'dir': [-0.2, 1, 0.15], 'pad': 1.15}, 'show': ['wound-rv', 'sternum', 'skin'], 'opacity': {'skin': 0.3, 'sternum': 0.5, 'lul': 0.2, 'lll': 0.2, 'fissure': 0.15},
         'highlight': ['wound-rv'], 'labels': ['heart', 'laa', 'aorta', 'sternum'], 'spin': True, 'ct': ct(R(WND), 'axial')},
        al_entry('cr'),
        pericardium('cr', 2),
        cardiorrhaphy('cr', 3),
        {'id': 'cr-after', 'phase': 'Close', 'seq': 4, 'title': 'After the repair',
         'body': '<p>Check the <b>back of the heart</b> for an exit wound: lift the apex gently (the pressure falls while you do). Leave the pericardium <b>open or loosely closed</b> so it cannot tamponade again.</p>'
                 '<p>Get an <b>echo</b> before discharge: septal and valve injuries are easily missed.</p>',
         'view': down(WND, dist=280), 'retract': LUNG_BACK, 'show': ['pericardiotomy'], 'opacity': {'heart': 1.0}, 'labels': ['heart'], 'ct': ct(R(WND), 'axial')},
    ]
    TRAUMA['cardio'] = ('Cardiorrhaphy', 'Left anterolateral', 'left', 'Cardiac box, anterolateral thoracotomy, pericardiotomy, pledgeted repair.', cr,
                        seq(('Box', 'other'), ('Entry', 'other'), ('Pericardium', 'other'), ('Repair', 'vein'), ('After', 'other')))

    # clean views: nerves, oesophagus, intrapulmonary trees and ligament hidden unless the step names them
    for v in TRAUMA.values():
        for entry in (v if isinstance(v, list) else [v]):
            for s in entry[4]:
                named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
                quiet = [i for i in ('n-phrenic', 'n-vagus', 'n-rln', 'lig-art', 'esophagus', 'ipl', 'ipl-r', 'n-phrenic-r', 'n-vagus-r', 'azygos',
                                     'lul-arteries', 'lul-veins', 'lul-bronchi', 'rul-arteries', 'rul-veins', 'rul-bronchi') if i not in named]
                s['hide'] = [i for i in quiet if has(i)]
                if entry[2] != 'both' and s.get('action', {}).get('kind') != 'thoracotomy':
                    s['hide'] += [f'rib-{k}-{sd_}' for k in (1, 2, 3, 8, 9, 10) for sd_ in 'lr' if has(f'rib-{k}-{sd_}')]
                ribs_faint = {} if s.get('action', {}).get('kind') == 'thoracotomy' else {q['id']: 0.16 for q in atlas['structures'] if q['id'].startswith('rib-')}
                s['opacity'] = {**{f'vert-t{i}': 0.22 for i in range(2, 11)}, **ribs_faint, **s.get('opacity', {})}
                for k in ('highlight', 'danger', 'labels', 'show'):
                    if k in s: s[k] = [i for i in s[k] if has(i)]
    for op_, v in TRAUMA.items():
        for (opName, appr, side, summ, steps_, sq) in (v if isinstance(v, list) else [v]):
            key = f'{op_}-{appr.split()[-1].lower()}' if isinstance(v, list) else op_
            procs[key] = {'id': f'trauma-{key}', 'op': op_, 'opName': opName, 'side': side, 'name': opName, 'approach': appr, 'summary': summ,
                          'ports': [], 'steps': steps_, 'sources': TSRC, 'sequence': sq, 'group': 'Trauma'}
# ==================================================================================================== access: positioning, landmarks, layers
ACC_OK = has('mus-latdorsi-l') and 'uni-4-l' in LM
ACCESS = {}
if ACC_OK:
    ASRC = [
        {'title': 'Shields TW, LoCicero J, et al. General Thoracic Surgery, 8th ed: chapters on thoracic incisions', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=thoracic+incisions+posterolateral+muscle+sparing+thoracotomy'},
        {'title': 'Gonzalez-Rivas D, et al. Uniportal video-assisted thoracoscopic lobectomy. J Thorac Dis 2013;5 Suppl 3:S234-45', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Gonzalez-Rivas+uniportal+video-assisted+thoracoscopic+lobectomy+2013'},
        {'title': 'Hansen HJ, Petersen RH. Video-assisted thoracoscopic lobectomy using a standardized three-port anterior approach. Ann Cardiothorac Surg 2012;1(1):70-76', 'url': 'https://doi.org/10.3978/j.issn.2225-319X.2012.04.15'},
        {'title': 'Laws D, Neville E, Duffy J. BTS guidelines for the insertion of a chest drain (the triangle of safety). Thorax 2003;58 Suppl 2:ii53-9', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=BTS+guidelines+insertion+chest+drain+2003'},
    ]

    def lines(k):
        return [i for i in (f'line-aal-{k}', f'line-mal-{k}', f'line-pal-{k}', f'lm-scaptip-{k}', f'lm-nipple-{k}') if has(i)]

    def outward(k):
        return V([-1.0, -0.3, 0.15]) if k == 'l' else V([1.0, -0.3, 0.15])

    def lat_view(tgt, k, dist=420.0, d=None):
        d = V(d) if d is not None else outward(k); d /= np.linalg.norm(d); return {'eye': R(V(tgt) + d * dist), 'target': R(tgt)}

    WALL = lambda k: [i for i in (f'mus-latdorsi-{k}', f'mus-serratus-{k}', f'mus-trapezius-{k}', f'mus-rhomboid-{k}', f'mus-pecmajor-{k}', f'mus-intercostal-{k}') if has(i)]
    RIBS = lambda k: [f'rib-{i}-{k}' for i in range(1, 11) if has(f'rib-{i}-{k}')]
    GIRDLE = lambda k: [i for i in (f'scapula-{k}', f'clavicle-{k}', f'humerus-{k}', 'cartilages') if has(i)]
    SIDE_NAME = {'l': 'left', 'r': 'right'}

    def positioning(k):
        sd = SIDE_NAME[k]; other = 'right' if k == 'l' else 'left'; out = outward(k)
        chest = V(S['heart']['centroid'])
        return [
            {'id': f'po{k}-surface', 'phase': 'Landmarks', 'seq': 0, 'title': 'Surface anatomy, supine',
             'body': '<p>Count the ribs from the <b>sternal angle</b> (angle of Louis): the 2nd costal cartilage joins the sternum there, and the 2nd space lies just below it. '
                     'In a man the <b>nipple</b> sits over the <b>4th space</b> in the mid-clavicular line. The <b>costal margin</b> (cartilages 7 to 10) ends at the xiphoid.</p>'
                     '<p>Laterally: the <b>anterior axillary line</b> drops from the anterior axillary fold (lower border of pectoralis major), the <b>mid-axillary line</b> from the apex of the axilla, '
                     'the <b>posterior axillary line</b> from the posterior fold (latissimus dorsi).</p>',
             'view': {'frame': ['sternum', *RIBS(k)[:8]], 'dir': [0, 1, 0.2], 'pad': 1.05}, 'show': ['skin', 'sternum', *RIBS(k), *GIRDLE(k), *lines(k)],
             'opacity': {'skin': 0.22, **{r: 0.55 for r in RIBS(k)}}, 'labels': ['sternum', f'lm-nipple-{k}', f'line-aal-{k}', f'line-mal-{k}', 'cartilages', f'clavicle-{k}'],
             'ct': ct(LM[f'nipple-{k}'], 'axial', 'bone')},
            {'id': f'po{k}-lateral', 'phase': 'Position', 'seq': 1, 'title': f'{other.capitalize()} lateral decubitus, {sd} side up',
             'body': '<p>Double-lumen tube placed and checked with the bronchoscope <b>before</b> turning; recheck after. Turn as a team, the head and neck with the trunk.</p>'
                     f'<p><b>Table broken (flexed)</b> with the break under the {other} flank, between the costal margin and the iliac crest: the {sd} intercostal spaces open and the hip drops out of the way of the instruments. '
                     '<b>Axillary roll</b> under the upper chest, a hand\'s breadth <b>below</b> the axilla (not in it), to take weight off the brachial plexus and the dependent shoulder.</p>'
                     '<p>Dependent arm forward on a board; upper arm on a rest, shoulder flexed about 90°, <b>never hyperabducted</b>. Pillow between the knees, lower leg flexed, upper leg straight; pad the fibular head (common peroneal nerve). '
                     'Supports or a bean bag front (pubis) and back (sacrum), strapping across the hip.</p>',
             'view': {'frame': ['skin'], 'dir': [out[0] * 0.9, 0.75, 0.12], 'pad': 1.6}, 'pose': 'lateral', 'show': ['skin', *lines(k)], 'opacity': {'skin': 1.0},
             'labels': [f'lm-scaptip-{k}', f'line-mal-{k}'],
             'ask': ask('Where does the axillary roll go?', 'Under the dependent chest wall, a hand\'s breadth below the axilla',
                        'It lifts the chest off the dependent shoulder; placed in the axilla it compresses the brachial plexus and axillary vessels.', 'In the dependent axilla', 'Under the upper arm'),
             'ct': ct(LM[f'scaptip-{k}'], 'axial', 'lung')},
            {'id': f'po{k}-landmarks', 'phase': 'Landmarks', 'seq': 2, 'title': 'Landmarks on the side you operate on',
             'body': '<p><b>Tip of the scapula</b>: about the 7th rib (7th space) with the arm at rest; the posterolateral incision passes 2–3 cm below it. Lifting the arm forward draws the scapula up and forward.</p>'
                     '<p><b>Mid-axillary line</b> at the 4th–5th space: the site for a uniportal incision, a utility incision and a chest drain. '
                     '<b>Triangle of safety</b> for a drain: lateral border of pectoralis major, anterior border of latissimus dorsi, a line at the level of the nipple (5th space), apex below the axilla.</p>'
                     '<p>Count ribs under the scapula from above with a hand: the highest rib felt is usually the 2nd; the 1st lies inside the curve of the 2nd.</p>',
             'view': lat_view(LM[f'scaptip-{k}'], k, 430, d=out + V([0, 0.6, 0.1])), 'pose': 'lateral', 'show': ['skin', *lines(k), *GIRDLE(k), *RIBS(k)],
             'opacity': {'skin': 0.3, **{r: 0.5 for r in RIBS(k)}},
             'highlight': [f'lm-scaptip-{k}', f'line-mal-{k}'], 'labels': [f'line-aal-{k}', f'line-pal-{k}', f'lm-nipple-{k}', f'scapula-{k}', f'rib-5-{k}', f'rib-7-{k}'],
             'ct': ct(LM[f'scaptip-{k}'], 'axial', 'bone')},
            {'id': f'po{k}-muscles', 'phase': 'Layers', 'seq': 3, 'title': 'The muscle layers of the lateral chest wall',
             'body': '<p>From outside in: <b>skin and fat</b>; <b>latissimus dorsi</b> behind and below (its anterior border is the posterior axillary fold); <b>trapezius</b> and <b>rhomboids</b> between the scapula and the spine; '
                     '<b>pectoralis major</b> in front (its border is the anterior axillary fold); <b>serratus anterior</b> on the ribs from the 1st to the 8th, running back under the scapula; '
                     'then the <b>intercostal muscles</b>, endothoracic fascia and parietal pleura.</p>'
                     '<p>The <b>long thoracic nerve</b> runs down the outer surface of serratus anterior near the mid-axillary line; the <b>thoracodorsal</b> nerve and vessels on the deep surface of latissimus dorsi.</p>'
                     '<p>The <b>auscultatory triangle</b> (latissimus, trapezius, medial border of the scapula) has no muscle over the ribs: the posterior limb of a thoracotomy enters there.</p>',
             'view': lat_view(LM[f'scaptip-{k}'], k, 470, d=out + V([0, -0.25, 0.05])), 'pose': 'lateral', 'show': [*WALL(k), *GIRDLE(k), *RIBS(k), f'n-longthoracic-{k}', f'n-thoracodorsal-{k}'],
             'opacity': {r: 0.5 for r in RIBS(k)}, 'labels': [*WALL(k)[:5], f'n-longthoracic-{k}', f'n-thoracodorsal-{k}'], 'danger': [f'n-longthoracic-{k}', f'n-thoracodorsal-{k}'],
             'ask': ask('A patient has a winged scapula after a lateral thoracotomy. Which nerve was injured?', 'The long thoracic nerve',
                        'It runs on the outer surface of serratus anterior in the mid-axillary line; serratus paralysis lets the scapula wing.', 'The thoracodorsal nerve', 'The intercostobrachial nerve'),
             'ct': ct(LM[f'scaptip-{k}'], 'axial', 'mediastinum')},
        ]

    for k in ('l', 'r'):
        ACCESS[f'position-{k}'] = ('Positioning and landmarks', f'{"Left" if k == "l" else "Right"} side up', SIDE_NAME[k], 'Surface anatomy, lateral decubitus on a broken table, landmarks, the layers of the chest wall.',
                                   positioning(k), seq(('Surface', 'other'), ('Position', 'other'), ('Landmarks', 'other'), ('Layers', 'other')))

    # ---------------------------------------------------------------- posterolateral and muscle-sparing thoracotomy
    def thor_variant(k, variant):
        sd = SIDE_NAME[k]; T = V(LM[f'thor-{k}']); out = outward(k)
        cutp = R(T); zc = [0.0, 0.0, 1.0]
        L = lambda i: f'{i}-{k}'
        skin = {'id': 'skin', 'label': 'Skin and subcutaneous fat', 'fate': 'through'}
        ic = {'id': L('mus-intercostal'), 'label': 'Intercostal muscles, on the upper border of the 6th rib', 'fate': 'divide', 'point': cutp, 'dir': zc, 'open': 10}
        if variant == 'standard':
            title, inc = 'Standard posterolateral thoracotomy (serratus-sparing)', f'incision-{k}'
            text = ('<p>From the <b>anterior axillary line</b> along the line of the 5th space, curving <b>2–3 cm below the tip of the scapula</b>, then up between the scapula and the spine as far as needed.</p>'
                    '<p><b>Latissimus dorsi is divided</b> in the line of the incision. <b>Serratus anterior is spared</b>: its posterior border is freed and the muscle lifted forward. '
                    'Trapezius and rhomboids are cut only if the incision is carried up behind the scapula.</p>')
            layers = [skin, {'id': L('mus-latdorsi'), 'label': 'Latissimus dorsi: divided', 'fate': 'divide', 'point': cutp, 'dir': zc, 'open': 18},
                      {'id': L('mus-serratus'), 'label': 'Serratus anterior: freed and retracted forward', 'fate': 'retract', 'offset': [0, 26, 6]}, ic]
        elif variant == 'classic':
            title, inc = 'Classic posterolateral thoracotomy (muscle-dividing)', f'incision-{k}'
            text = ('<p>The same curved incision below the scapular tip, but <b>both latissimus dorsi and serratus anterior are divided</b> across. Widest exposure, fastest to open and close; '
                    'the most pain and loss of shoulder strength. Keep the serratus cut low, near its rib attachments, to spare the long thoracic nerve.</p>')
            layers = [skin, {'id': L('mus-latdorsi'), 'label': 'Latissimus dorsi: divided', 'fate': 'divide', 'point': cutp, 'dir': zc, 'open': 18},
                      {'id': L('mus-serratus'), 'label': 'Serratus anterior: divided', 'fate': 'divide', 'point': cutp, 'dir': zc, 'open': 14}, ic]
        else:
            title, inc = 'Muscle-sparing lateral thoracotomy', f'incision-ms-{k}'
            text = ('<p>A <b>vertical incision of 10–12 cm</b> along the anterior border of latissimus dorsi, from the axilla down. Raise skin flaps widely.</p>'
                    '<p><b>Latissimus dorsi is retracted back</b> (the thoracodorsal bundle stays on its deep surface). <b>Serratus anterior is split along its fibres</b> over the 5th space, '
                    'or lifted forward. Divide the intercostals well beyond the skin incision so the ribs spread without breaking. Less pain and better shoulder function; seroma is common, so drain the flaps.</p>')
            layers = [skin, {'id': L('mus-latdorsi'), 'label': 'Latissimus dorsi: retracted back', 'fate': 'retract', 'offset': [0, -30, -4]},
                      {'id': L('mus-serratus'), 'label': 'Serratus anterior: split along its fibres', 'fate': 'split', 'point': cutp, 'dir': zc, 'open': 8}, ic]
        pre = f'{variant[:2]}{k}'
        steps = [
            {'id': f'{pre}-incision', 'phase': 'Incision', 'seq': 0, 'title': title,
             'body': text + ('<p>Landmarks: <b>tip of the scapula</b>, the <b>5th space</b> (count from above), the anterior and posterior axillary lines.</p>'),
             'view': lat_view(T, k, 440, d=out + V([0, 0.2, 0.25])), 'pose': 'lateral', 'show': ['skin', inc, *lines(k)], 'opacity': {'skin': 1.0},
             'highlight': [inc], 'labels': [f'lm-scaptip-{k}', f'line-aal-{k}', f'line-pal-{k}'], 'ct': ct(R(T), 'axial', 'lung')},
            {'id': f'{pre}-layers', 'phase': 'Layers', 'seq': 1, 'title': 'Through the chest wall, layer by layer',
             'body': '<p>' + ' → '.join(l['label'] for l in layers) + '.</p><p>Press play to take each layer in turn.</p>'
                     '<p>Enter the chest over the <b>upper border of the 6th rib</b>: the intercostal vein, artery and nerve run in the groove under the lower border of the rib above.</p>',
             'view': lat_view(T, k, 400, d=out + V([0, -0.35, 0.2])), 'pose': 'lateral', 'show': [inc, *WALL(k), *RIBS(k), f'scapula-{k}', f'n-longthoracic-{k}', f'n-thoracodorsal-{k}'],
             'opacity': {r: 0.55 for r in RIBS(k)}, 'hide': ['skin'],
             'labels': [L('mus-latdorsi'), L('mus-serratus'), L('mus-intercostal')], 'danger': [f'n-longthoracic-{k}', f'n-thoracodorsal-{k}'],
             'action': {'kind': 'layers', 'label': 'Go through the layers', 'port': f'thor-{k}', 'layers': [l for l in layers if l['id'] != 'skin']},
             'ct': ct(R(T), 'axial', 'mediastinum')},
            {**thoracotomy_step(pre, sd), 'seq': 2, 'id': f'{pre}-spread', 'phase': 'Entry', 'title': 'Open the pleura and spread the ribs'},
            {'id': f'{pre}-close', 'phase': 'Close', 'seq': 3, 'title': 'Closing and what each choice costs',
             'body': '<p>Two drains through separate stab incisions below the wound. <b>Pericostal sutures</b> round the 5th and 6th ribs (or intracostal through the 6th, sparing the nerve below the 5th). '
                     'Repair each muscle layer that was cut; with a muscle-sparing incision, lay the muscles back and drain the flaps.</p>'
                     '<p>Muscle-dividing: best exposure, most pain and shoulder weakness. Serratus-sparing: the usual compromise. Muscle-sparing: least pain, smaller field, seromas.</p>',
             'view': lat_view(T, k, 520), 'pose': 'lateral', 'show': ['skin', inc], 'opacity': {'skin': 1.0}, 'labels': [inc], 'ct': ct(R(T), 'axial', 'lung')},
        ]
        steps[2]['pose'] = 'lateral'; steps[2]['action'] = {**steps[2]['action'], 'incision': inc}
        return steps

    for k in ('l', 'r'):
        for variant, appr in (('standard', 'Serratus-sparing'), ('classic', 'Muscle-dividing'), ('muscle', 'Muscle-sparing')):
            ACCESS[f'plt-{variant}-{k}'] = (f'Thoracotomy incisions', f'{appr} ({"left" if k == "l" else "right"})', SIDE_NAME[k],
                                           'Landmarks, the incision, each muscle layer, then the ribs.', thor_variant(k, variant),
                                           seq(('Incision', 'other'), ('Layers', 'other'), ('Ribs', 'bronchus'), ('Close', 'other')))

    # ---------------------------------------------------------------- VATS port maps
    def vats_map(k, layout):
        sd = SIDE_NAME[k]; out = outward(k); L = lambda i: f'{i}-{k}'
        pre = 'r-' if k == 'r' else ''
        if layout == 'three-a':
            title = 'Three-port VATS, anterior approach'; pids = [f'port-{pre}anterior-{x}' for x in ('utility', 'camera', 'posterior')]; main = pids[0]
            text = ('<p><b>Utility incision</b> (4–5 cm, no rib spreading) in the <b>4th space, anteriorly</b>, directly over the hilum and the superior vein. '
                    '<b>Camera</b> low anterior, at the level of the top of the diaphragm (7th space). <b>Working port</b> at the same low level, further back.</p>'
                    '<p>Hilum first, from the front (Copenhagen). Anterior ports pass through serratus anterior and, near the fold, the edge of pectoralis major.</p>')
            through = [L('mus-serratus'), L('mus-pecmajor')]
        elif layout == 'three-p':
            title = 'Three-port VATS, posterior (fissure) approach'; pids = [f'port-{pre}posterior-{x}' for x in ('utility', 'camera', 'posterior')]; main = pids[0]
            text = ('<p><b>Utility incision</b> in the <b>5th space</b> in line with the fissure; <b>camera</b> low in the posterior axillary line; <b>working port</b> posteriorly, below the scapular tip. '
                    'Fissure first, artery first.</p><p>The posterior ports go through latissimus dorsi as well as serratus anterior.</p>')
            through = [L('mus-latdorsi'), L('mus-serratus')]
        elif layout == 'two':
            title = 'Two-port (biportal) VATS'; pids = [L('bi-utility'), L('bi-camera')]; main = pids[0]
            text = ('<p>A <b>utility incision</b> of 3–4 cm in the <b>4th space</b> (upper lobes; 5th for lower lobes) at the <b>anterior axillary line</b>, and a 1 cm <b>camera port</b> in the <b>7th or 8th space</b> in the mid- to posterior axillary line.</p>'
                    '<p>All instruments and staplers go through the utility incision; the camera looks up from below, as in three-port VATS.</p>')
            through = [L('mus-serratus')]
        else:
            title = 'Uniportal VATS'; pids = [L('uni-4'), L('uni-5')]; main = pids[0]
            text = ('<p>A single <b>3–4 cm incision</b> between the <b>anterior and mid-axillary lines</b>: <b>4th space</b> for upper lobes, <b>5th space</b> for middle and lower lobes.</p>'
                    '<p>Serratus anterior is split along its fibres; a wound protector holds it open. The <b>camera sits at the back</b> of the wound, instruments below and in front of it, all working in the same plane '
                    'as in open surgery. Staplers come in from the same incision, so angles for the superior vein and the bronchus need planning (curved-tip staplers help).</p>')
            through = [L('mus-serratus')]
        P = V(LM[main]) if main in LM else V(pt(main))
        layers = [{'id': 'skin', 'label': 'Skin and subcutaneous fat', 'fate': 'through'}] + \
                 [{'id': t, 'label': f'{S[t]["name"].split(",")[0]}: {"split along its fibres" if layout == "uni" and "serratus" in t else "passed through"}',
                   'fate': 'split' if layout == 'uni' and 'serratus' in t else 'through', 'point': R(P), 'dir': [0, 0, 1], 'open': 6} for t in through if has(t)] + \
                 [{'id': L('mus-intercostal'), 'label': 'Intercostal muscles, on the upper border of the rib below', 'fate': 'through'}]
        hil = V(pt('pa-left' if k == 'l' else 'rpa'))
        d_in = (hil - P) / np.linalg.norm(hil - P)
        return [
            {'id': f'vm{layout}{k}-map', 'phase': 'Ports', 'seq': 0, 'title': title, 'body': text,
             'view': lat_view(P, k, 470, d=out + V([0, 0.5, 0.3])), 'pose': 'lateral', 'show': ['skin', *pids, *lines(k)], 'opacity': {'skin': 1.0},
             'highlight': pids, 'labels': [*pids, f'lm-scaptip-{k}', f'line-mal-{k}'], 'ct': ct(R(P), 'axial', 'lung')},
            {'id': f'vm{layout}{k}-layers', 'phase': 'Layers', 'seq': 1, 'title': 'What the port goes through',
             'body': '<p>' + ' → '.join(l['label'] for l in layers) + '.</p><p>Enter on the <b>upper border of the lower rib</b>; open the pleura with a finger and sweep for adhesions before the first instrument goes in.</p>',
             'view': lat_view(P, k, 380, d=out + V([0, 0.2, 0.2])), 'pose': 'lateral', 'show': [*pids, *WALL(k), *RIBS(k)], 'hide': ['skin'], 'opacity': {r: 0.55 for r in RIBS(k)},
             'labels': [*through, L('mus-intercostal')], 'danger': [f'n-longthoracic-{k}'],
             'action': {'kind': 'layers', 'label': 'Go through the layers', 'port': main, 'layers': [l for l in layers if l['id'] != 'skin']}, 'ct': ct(R(P), 'axial', 'mediastinum')},
            {'id': f'vm{layout}{k}-inside', 'phase': 'Inside', 'seq': 2, 'title': 'The view from inside',
             'body': '<p>From the camera the hilum sits in the middle of the screen, the instruments entering from '
                     + ('the same incision, in line with the camera.' if layout == 'uni' else 'separate ports at an angle to the camera (triangulation).') + '</p>',
             'view': {'eye': R(P - d_in * 5), 'target': R(hil)}, 'show': [*pids], 'hide': [i for i in ('lul-arteries', 'lul-veins', 'lul-bronchi', 'lll-arteries', 'lll-veins', 'lll-bronchi', 'rul-arteries', 'rul-veins', 'rul-bronchi') if has(i)], 'opacity': {**CLEAR, **({'rul': 0.35, 'rml': 0.35, 'rll': 0.35} if k == 'r' else {})},
             'labels': ['pa-left', 'pv-superior', 'br-lul'] if k == 'l' else ['rpa', 'rpv-superior', 'br-rul'], 'ct': ct(R(hil), 'axial', 'mediastinum')},
        ]

    for k in ('l', 'r'):
        for layout, appr in (('three-a', 'Three-port, anterior'), ('three-p', 'Three-port, posterior'), ('two', 'Two-port'), ('uni', 'Uniportal')):
            ACCESS[f'vats-{layout}-{k}'] = ('VATS port placement', f'{appr} ({"left" if k == "l" else "right"})', SIDE_NAME[k],
                                           'Where the ports go, what they go through, and the view from inside.', vats_map(k, layout),
                                           seq(('Ports', 'other'), ('Layers', 'other'), ('Inside', 'other')))

    for key, (opName, appr, side, summ, steps_, sq) in ACCESS.items():
        base_op = {'Positioning and landmarks': 'position', 'Thoracotomy incisions': 'thoracotomy', 'VATS port placement': 'vats-ports'}[opName]
        if base_op != 'position':   # one menu entry per side, so each has a short row of approaches
            op_ = f'{base_op}-{side[0]}'; opName = f'{opName}, {side}'; appr = appr.rsplit(' (', 1)[0]
        else: op_ = base_op
        KEEP = {'chest-wall', 'muscles', 'landmarks', 'ports-vats', 'ports-anterior', 'ports-posterior', 'ports-r-anterior', 'ports-r-posterior', 'ports-open-left', 'ports-open-right'}
        for s in steps_:
            for kk in ('highlight', 'danger', 'labels', 'show'):
                if kk in s: s[kk] = [i for i in s[kk] if has(i) or i == 'skin']
            if s['phase'] != 'Inside':   # the chest wall only: lungs, vessels and mediastinum out of the way
                named = set(s.get('show', [])) | set(s.get('labels', [])) | set(s.get('highlight', [])) | set(s.get('danger', []))
                s['hide'] = [*s.get('hide', []), *[q['id'] for q in atlas['structures'] if q['group'] not in KEEP and q['id'] not in named]]
        procs[f'acc-{key}'] = {'id': f'access-{key}', 'op': op_, 'opName': opName, 'side': side, 'name': opName, 'approach': appr, 'summary': summ,
                               'ports': [], 'steps': steps_, 'sources': ASRC, 'sequence': sq, 'group': 'Access and positioning'}

    # ---------------------------------------------------------------- uniportal and biportal lobectomy: the fissure-first sequence through fewer ports
    def port_version(steps, op, side, layout, lobe):
        k = side[0]; upper = lobe in ('lul', 'rul')
        main = f'uni-{4 if upper else 5}-{k}' if layout == 'uni' else f'bi-utility-{k}'
        cam = main if layout == 'uni' else f'bi-camera-{k}'
        setup = {'id': f'{op}-setup', 'phase': 'Setup', 'title': 'Uniportal incision' if layout == 'uni' else 'Two ports', 'pose': 'lateral',
                 'body': (f'<p>{"Right" if k == "l" else "Left"} lateral decubitus, table broken. One 3–4 cm incision in the <b>{4 if upper else 5}th space</b> between the anterior and mid-axillary lines, '
                          'wound protector in. Camera at the back of the wound; staplers and instruments below and in front of it.</p>'
                          if layout == 'uni' else
                          f'<p>{"Right" if k == "l" else "Left"} lateral decubitus, table broken. A 3–4 cm <b>utility incision</b> in the 4th space at the anterior axillary line and a 1 cm <b>camera port</b> in the 7th–8th space, mid- to posterior axillary line. '
                          'Every instrument and stapler goes through the utility incision.</p>'),
                 'view': lat_view(V(LM[main]), k, 470, d=outward(k) + V([0, 0.5, 0.3])), 'show': ['skin', main, cam, *lines(k)], 'opacity': {'skin': 1.0},
                 'labels': [main, cam, f'line-mal-{k}', f'lm-scaptip-{k}'], 'ct': ct(R(V(LM[main])), 'axial', 'lung')}
        out = []
        for s_ in steps:
            if s_['phase'] == 'Setup': continue
            c = copy.deepcopy(s_); c['id'] = f'{op}-{s_["id"].split("-", 1)[1]}'
            if 'action' in c: c['action']['port'] = main
            out.append(c)
        return [out[0], setup, *out[1:]]

    BASE_SEQ = {'lul': ('lul-posterior', posterior), 'lll': ('lll-fissure', lll_fissure_first)}
    if RUL_OK: BASE_SEQ.update({'rul': ('rul-posterior', rul_posterior), 'rml': ('rml-fissure', rml_fissure_first), 'rll': ('rll-fissure', rll_fissure_first)})
    for lobe, (bkey, bsteps) in BASE_SEQ.items():
        base = procs[bkey]
        for layout, appr in (('uni', 'Uniportal VATS'), ('bi', 'Two-port VATS')):
            procs[f'{lobe}-{layout}'] = {**base, 'id': f'{layout}-{lobe}', 'approach': appr, 'name': base['opName'] + ', ' + appr.lower(),
                                         'steps': port_version(bsteps, f'{lobe[:2]}{layout}', base['side'], layout, lobe),
                                         'summary': ('One incision; ' if layout == 'uni' else 'Two ports; ') + base['summary'][0].lower() + base['summary'][1:]}
# ==================================================================================================== batch 4: thymectomy, oesophagectomy, thoracic duct, empyema
B4 = {}
B4_OK = has('thymus') and has('conduit-chest') and 'eso-ivor' in LM and has('thoracic-duct')
if B4_OK:
    Lm = lambda k: V(LM[k])
    LUNGS_ALL = [i for i in ('lul', 'lll', 'fissure', 'rul', 'rml', 'rll', 'fissure-h', 'fissure-r') if has(i)]
    INTRA = [i for i in S if S[i]['group'].endswith('-intra')]
    stn = S['sternum']; st_x = stn['centroid'][0]
    st_top = V([st_x, stn['bbox'][1][1] - 8, stn['bbox'][1][2] - 6]); st_bot = V([st_x, stn['bbox'][1][1] - 22, stn['bbox'][0][2] + 8])
    st_mid = (st_top + st_bot) / 2
    LM['sternum-front'] = R(st_mid + V([0, 60, 0]))
    LM['xiphoid'] = R(st_bot + V([0, 25, -30]))
    B4SRC = {
        'thymus': [{'title': 'Wolfe GI, et al. Randomized trial of thymectomy in myasthenia gravis (MGTX). N Engl J Med 2016;375:511-522', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Wolfe+randomized+trial+thymectomy+myasthenia+gravis+2016'},
                   {'title': 'Detterbeck FC, Parsons AM. Management of stage I and II thymoma. Thorac Surg Clin 2011 (Masaoka-Koga staging)', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Masaoka-Koga+thymoma+staging'},
                   {'title': 'Suda T. Subxiphoid thymectomy: single-port, dual-port, and robot-assisted. J Vis Surg 2017', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Suda+subxiphoid+thymectomy'}],
        'eso': [{'title': 'Orringer MB, et al. Two thousand transhiatal esophagectomies. Ann Surg 2007;246:363-374', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Orringer+two+thousand+transhiatal+esophagectomies'},
                {'title': 'Low DE, et al. International consensus on standardization of data collection for complications associated with esophagectomy (ECCG). Ann Surg 2015', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Esophagectomy+Complications+Consensus+Group+2015'},
                {'title': 'Hulscher JB, et al. Extended transthoracic resection compared with limited transhiatal resection for adenocarcinoma of the esophagus. N Engl J Med 2002;347:1662-9', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Hulscher+extended+transthoracic+transhiatal+2002'}],
        'duct': [{'title': 'Patterson GA, et al. Supradiaphragmatic ligation of the thoracic duct in intractable chylous fistula. Ann Thorac Surg 1981;32:44-9', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=supradiaphragmatic+ligation+thoracic+duct+chylous'},
                 {'title': 'Cope C, Kaiser LR. Management of unremitting chylothorax by percutaneous embolization of the thoracic duct. J Vasc Interv Radiol 2002', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Cope+percutaneous+embolization+thoracic+duct'}],
        'empyema': [{'title': 'Shen KR, et al. The American Association for Thoracic Surgery consensus guidelines for the management of empyema. J Thorac Cardiovasc Surg 2017;153:e129-46', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=AATS+consensus+guidelines+management+of+empyema+2017'},
                    {'title': 'Rahman NM, et al. Intrapleural use of tissue plasminogen activator and DNase in pleural infection (MIST2). N Engl J Med 2011;365:518-26', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=MIST2+tissue+plasminogen+activator+DNase+pleural+infection'}],
    }

    def front(tgt, dist=330, d=(0, 1, 0.25)):
        d = V(d) / np.linalg.norm(V(d)); return {'eye': R(V(tgt) + d * dist), 'target': R(tgt)}

    # ---------------------------------------------------------------- thymectomy
    TH = ['thymus', 'lbcv', 'svc', 'aorta', 'heart', 'n-phrenic', 'n-phrenic-r', 'thymic-vein-1', 'thymic-vein-2', 'thyroid']
    th_c, th_lo, th_hi = Lm('thymus-c'), Lm('thymus-lo'), Lm('thymus-hi')
    hl, hr = Lm('thymus-horn-l'), Lm('thymus-horn-r')
    ph_l = V(S['n-phrenic']['centroid']); ph_r = V(S['n-phrenic-r']['centroid']) if has('n-phrenic-r') else th_c + V([45, 0, 0])

    def th_steps(appr):
        pre = {'sternotomy': 'ts', 'rvats': 'tv', 'subx': 'tx'}[appr]
        port = {'sternotomy': 'sternum-front', 'rvats': 'uni-4-r', 'subx': 'xiphoid'}[appr]
        view = (lambda t, dist=300: front(t, dist)) if appr != 'rvats' else (lambda t, dist=300: front(t, dist, (0.9, 0.55, 0.15)))
        if appr == 'subx': view = lambda t, dist=300: front(t, dist, (0, 0.8, -0.6))
        entry = {
            'sternotomy': {'id': f'{pre}-entry', 'phase': 'Entry', 'seq': 1, 'title': 'Median sternotomy',
                           'body': '<p>Supine, a roll between the shoulders. Midline incision from the <b>sternal notch</b> to below the <b>xiphoid</b>. Divide the interclavicular ligament above and free the xiphoid below; '
                                   'sweep a finger behind the manubrium and the lower sternum.</p><p>Ask for the lungs to be deflated, then saw <b>exactly in the midline</b>. Wax the marrow, seat the retractor, open slowly.</p>',
                           'view': front(st_mid, 420, (0.15, 1, 0.2)), 'show': ['sternum', 'cartilages', *TH], 'opacity': {'sternum': 0.95, 'heart': 0.5},
                           'highlight': ['sternum'], 'danger': ['lbcv'],
                           'action': {'kind': 'sternotomy', 'label': 'Saw and open the sternum', 'port': port, 'ids': ['sternum'], 'at': R(st_mid), 'axis': [1, 0, 0], 'path': [R(st_top), R(st_bot)], 'radius': 70},
                           'ct': ct('sternum', 'axial', 'bone')},
            'rvats': {'id': f'{pre}-entry', 'phase': 'Entry', 'seq': 1, 'title': 'Right VATS: position and ports',
                      'body': '<p>Supine with the right side <b>raised 30°</b>, right arm down or on an arm rest. Three ports on the right: <b>5th space mid-axillary</b> (camera), <b>3rd space anterior axillary</b> and <b>5th space mid-clavicular</b>. '
                              'CO2 at <b>6–8 mmHg</b> pushes the mediastinum and lung away.</p><p>Start at the <b>right pericardiophrenic angle</b>, anterior to the right phrenic nerve, and work up.</p>',
                      'view': view(th_c, 380), 'show': TH, 'opacity': {'heart': 0.5}, 'highlight': ['thymus'], 'danger': ['n-phrenic-r', 'svc'], 'labels': ['svc'],
                      'ct': ct(R(th_c), 'axial', 'mediastinum')},
            'subx': {'id': f'{pre}-entry', 'phase': 'Entry', 'seq': 1, 'title': 'Subxiphoid approach',
                     'body': '<p>Supine, legs apart (the surgeon stands between them). A <b>3 cm incision below the xiphoid</b>; detach the rectus from the xiphoid (or excise it); '
                             'develop the plane <b>behind the sternum</b> with a finger. Wound protector, CO2 at 8 mmHg, and a sternal lifting hook if needed. Two 5 mm subcostal ports are optional.</p>'
                             '<p>Both pleurae and both phrenic nerves are seen from the midline: the best view of the <b>left side</b> and of the neck.</p>',
                     'view': view(th_c, 380), 'show': [*TH, 'sternum'], 'opacity': {'heart': 0.5, 'sternum': 0.3}, 'highlight': ['thymus'], 'danger': ['n-phrenic', 'n-phrenic-r'],
                     'ct': ct(R(th_c), 'sagittal', 'mediastinum')},
        }[appr]
        return [
            {'id': f'{pre}-anat', 'phase': 'Anatomy', 'seq': 0, 'title': 'The thymus and its boundaries',
             'body': '<p>In an adult the thymus is a fatty, bilobed organ in the <b>anterior mediastinum</b>, on the pericardium and the great vessels. Its limits: the <b>phrenic nerves</b> laterally, '
                     'the <b>thyroid</b> above (the two upper horns), the <b>diaphragm</b> below. Its veins drain into the back of the <b>left brachiocephalic (innominate) vein</b>; its arteries come from the internal mammary and inferior thyroid arteries.</p>'
                     '<p>Indications: <b>thymoma</b> (stage by Masaoka-Koga) and <b>myasthenia gravis</b> (the MGTX trial favoured extended thymectomy in AChR-antibody-positive, non-thymomatous disease). '
                     'For myasthenia, remove <b>all</b> the anterior mediastinal fat between the phrenic nerves: ectopic thymic tissue lies throughout it.</p>',
             'view': front(th_c, 380), 'show': [*TH, 'sternum'], 'opacity': {'sternum': 0.15, 'heart': 0.55}, 'spin': True,
             'highlight': ['thymus'], 'danger': ['n-phrenic', 'n-phrenic-r', 'lbcv'], 'labels': ['thyroid', 'svc', 'aorta', 'thymic-vein-1'],
             'ask': ask('What are the lateral limits of an extended thymectomy?', 'The two phrenic nerves',
                        'Everything anterior between the phrenic nerves, from the diaphragm to the thyroid, comes out; a phrenic palsy in a myasthenic is disastrous.', 'The internal mammary arteries', 'The lateral borders of the sternum'),
             'ct': ct(R(th_c), 'axial', 'mediastinum')},
            entry,
            {'id': f'{pre}-lower', 'phase': 'Dissection', 'seq': 2, 'title': 'Lower poles off the pericardium',
             'body': '<p>Start low: lift the lower poles and the pericardial fat off the pericardium with diathermy and blunt dissection, working up toward the great vessels. '
                     'Take the pericardiophrenic fat pads with the specimen.</p>',
             'view': view(th_lo, 300), 'show': TH, 'opacity': {'heart': 0.6}, 'highlight': ['thymus'], 'danger': ['n-phrenic', 'n-phrenic-r'],
             'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Lift the lower poles', 'port': port,
                        'path': [R(th_lo + V([-25, 8, 5])), R(th_lo + V([0, 10, 8])), R(th_lo + V([25, 8, 5])), R(th_c + V([0, 6, -5]))]},
             'ct': ct(R(th_lo), 'axial', 'mediastinum')},
            {'id': f'{pre}-phrenic', 'phase': 'Dissection', 'seq': 3, 'title': 'Along each phrenic nerve',
             'body': '<p>Open the mediastinal pleura <b>1 cm in front of each phrenic nerve</b> and take the fat off it, never the nerve itself: no diathermy on the nerve, no traction on it. '
                     'On the left the nerve crosses the aortic arch and is easily hidden in fat.</p>',
             'view': view(th_c, 360), 'show': TH, 'opacity': {'heart': 0.55}, 'highlight': ['thymus'], 'danger': ['n-phrenic', 'n-phrenic-r'], 'labels': ['n-phrenic', 'n-phrenic-r'],
             'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Dissect along the nerves', 'port': port,
                        'path': [R(ph_r + V([-8, 12, -30])), R(ph_r + V([-8, 12, 20])), R(th_c + V([0, 12, 30])), R(ph_l + V([10, 12, 20])), R(ph_l + V([10, 12, -30]))]},
             'ct': ct('n-phrenic', 'axial', 'mediastinum')},
            {'id': f'{pre}-veins', 'phase': 'Veins', 'seq': 4, 'title': 'Thymic veins into the innominate vein',
             'body': '<p>Lift the gland forward off the <b>left brachiocephalic vein</b>. One to three <b>thymic veins</b> enter its back: clip or tie each and divide it. '
                     'Traction tears them flush with the innominate vein: control that with a finger and a side-biting clamp, not blind clips.</p>',
             'view': view(V(pt('thymic-vein-1')), 260), 'show': TH, 'opacity': {'heart': 0.55, 'thymus': 0.55},
             'highlight': ['thymic-vein-1', 'thymic-vein-2'], 'danger': ['lbcv'],
             'action': {'kind': 'ligate', 'label': 'Tie the thymic veins', 'port': port, 'ids': ['thymic-vein-1', 'thymic-vein-2']},
             'ask': ask('A thymic vein tears flush with the innominate vein. First move?', 'Finger pressure, then a side-biting clamp and a fine suture',
                        'Blind clips or diathermy on a torn innominate vein make it bigger.', 'Clip it blindly', 'Ligate the innominate vein'),
             'ct': ct('thymic-vein-1', 'axial', 'mediastinum')},
            {'id': f'{pre}-horns', 'phase': 'Dissection', 'seq': 5, 'title': 'The upper horns from the thyroid',
             'body': '<p>Follow each upper horn into the neck with gentle traction until it narrows to a thin band ending on the thyroid; tie or clip its tip (small inferior thyroid branches). '
                     'The horns lie on the brachiocephalic veins and the innominate artery.</p>',
             'view': view((hl + hr) / 2, 280), 'show': TH, 'opacity': {'heart': 0.5}, 'highlight': ['thymus'], 'danger': ['lbcv', 'thyroid'], 'labels': ['thyroid'],
             'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Free the upper horns', 'port': port, 'path': [R(hr + V([0, 8, -15])), R(hr + V([0, 8, 0])), R(hl + V([0, 8, 0])), R(hl + V([0, 8, -15]))]},
             'ct': ct(R((hl + hr) / 2), 'coronal', 'mediastinum')},
            {'id': f'{pre}-specimen', 'phase': 'Specimen', 'seq': 6, 'title': 'En bloc, then check the fat that is left',
             'body': '<p>The gland comes out <b>in one piece with its fat</b>, in a bag. Look again at the places where ectopic thymus hides: the <b>aortopulmonary window</b>, the <b>aortocaval groove</b>, '
                     'the <b>pericardiophrenic fat</b> on both sides and the <b>neck</b>.</p><p>One drain; close the sternum with wires (sternotomy).</p>',
             'view': front(th_c, 400), 'show': TH, 'opacity': {'heart': 0.6}, 'labels': ['aorta', 'svc', 'n-phrenic', 'n-phrenic-r'],
             'specimen': {'ids': ['thymus', 'thymic-vein-1', 'thymic-vein-2'], 'offset': [0, 90, 20]}, 'ct': ct(R(th_c), 'axial', 'mediastinum')},
        ]

    for appr, name in (('sternotomy', 'Median sternotomy'), ('rvats', 'Right VATS'), ('subx', 'Subxiphoid VATS')):
        B4[f'thymectomy-{appr}'] = ('thymectomy', 'Thymectomy', name, 'both', 'Extended thymectomy: phrenic to phrenic, diaphragm to thyroid, the thymic veins tied.',
                                    th_steps(appr), seq(('Anatomy', 'other'), ('Entry', 'other'), ('Lower poles', 'other'), ('Phrenics', 'other'), ('Thymic veins', 'vein'), ('Horns', 'other'), ('Specimen', 'other')), 'Mediastinum', 'thymus')

    # ---------------------------------------------------------------- oesophagectomy
    ESO = ['esophagus', 'aorta', 'azygos', 'trachea', 'br-left-main', 'br-right-main', 'thoracic-duct', 'stomach', 'n-vagus-r', 'n-vagus', 'n-rln', 'heart']
    gej, eso_top, pyl = Lm('gej'), Lm('eso-top'), Lm('pylorus')
    mid_eso = (gej + eso_top) / 2
    az_div = V(S['azygos']['division']['point']) if 'division' in S['azygos'] else mid_eso + V([20, 0, 40])
    back = lambda t, dist=360: front(t, dist, (0.75, -0.65, 0.15))          # from the right and behind, as through a right thoracotomy
    abd = lambda t, dist=380: front(t, dist, (-0.1, 1, 0.35))

    def e_anat(pre):
        return {'id': f'{pre}-anat', 'phase': 'Anatomy', 'seq': 0, 'title': 'The oesophagus and its neighbours',
                'body': '<p><b>Neck</b>: behind the trachea, the <b>recurrent laryngeal nerves</b> in the grooves either side. <b>Upper chest</b>: behind the trachea, the <b>azygos arch</b> on its right, the aortic arch on its left. '
                        '<b>Mid chest</b>: behind the left main bronchus and the left atrium. <b>Lower chest</b>: in front of and to the right of the descending aorta, through the hiatus at T10.</p>'
                        '<p>Behind it on the right: the <b>thoracic duct</b> between the aorta and the azygos, crossing to the left at T4–T6. The <b>vagi</b> form the oesophageal plexus on its wall.</p>',
                'view': back(mid_eso, 480), 'show': ESO, 'hide': LUNGS_ALL + INTRA, 'opacity': {'heart': 0.35, 'aorta': 0.8, 'stomach': 0.8}, 'spin': True,
                'highlight': ['esophagus'], 'danger': ['thoracic-duct', 'azygos', 'trachea', 'aorta'], 'labels': ['stomach', 'n-rln', 'n-vagus-r'],
                'ct': ct(R(mid_eso), 'sagittal', 'mediastinum')}

    lap = lambda pre, seq_: [
        {'id': f'{pre}-lap', 'phase': 'Abdomen', 'seq': seq_, 'title': 'Abdomen: mobilise the stomach',
         'body': '<p>Upper midline laparotomy (or laparoscopy). Divide the gastrocolic omentum <b>well away from the right gastroepiploic arcade</b>, which the conduit will live on, then the short gastric vessels up to the left crus. '
                 'Open the lesser omentum. <b>Kocherise</b> the duodenum so the pylorus reaches the hiatus; a pyloric drainage procedure or none, by unit policy.</p>',
         'view': abd(pyl, 420), 'show': ['incision-lap', 'stomach', 'liver', 'spleen', 'duodenum', 'pancreas', 'rgea', 'lga', 'aorta', 'esophagus'], 'hide': LUNGS_ALL + INTRA,
         'opacity': {'liver': 0.25, 'heart': 0.3, 'stomach': 0.9}, 'highlight': ['stomach'], 'danger': ['rgea', 'spleen'], 'labels': ['duodenum', 'liver', 'incision-lap'],
         'ct': ct(R(pyl), 'axial', 'mediastinum')},
        {'id': f'{pre}-lga', 'phase': 'Abdomen', 'seq': seq_, 'title': 'Left gastric artery at its origin',
         'body': '<p>Lift the stomach up; the <b>left gastric pedicle</b> is taut in the lesser sac. Clear the <b>coeliac nodes</b> into the specimen and staple or tie the left gastric artery and vein <b>at their origin</b>. '
                 'Check the common hepatic and splenic arteries first.</p>',
         'view': abd(Lm('coeliac'), 300), 'show': ['stomach', 'liver', 'pancreas', 'rgea', 'lga', 'aorta', 'spleen'], 'hide': LUNGS_ALL + INTRA, 'opacity': {'liver': 0.2, 'stomach': 0.45, 'pancreas': 0.5},
         'highlight': ['lga'], 'danger': ['aorta', 'pancreas'],
         'action': {'kind': 'staple', 'label': 'Staple the left gastric pedicle', 'port': 'lap', 'ids': ['lga'], 'reload': 'vascular'},
         'ct': ct('lga', 'axial', 'mediastinum')},
    ]

    def r_chest(pre, seq_, level_top):
        return [
            {**thoracotomy_step(pre, 'right'), 'seq': seq_, 'phase': 'Chest', 'pose': 'lateral', 'title': 'Right thoracotomy (or right VATS)',
             'body': '<p>Left lateral decubitus. Right posterolateral thoracotomy through the <b>5th space</b> (or four-port VATS / prone thoracoscopy). The right lung is isolated and retracted forward.</p>'},
            {'id': f'{pre}-azygos', 'phase': 'Chest', 'seq': seq_, 'title': 'Divide the azygos arch',
             'body': '<p>Open the mediastinal pleura along the front of the azygos arch and behind it. Staple the arch (vascular load): it opens the upper mediastinum and the space for the conduit.</p>',
             'view': back(az_div, 300), 'show': ESO, 'hide': ['rul', 'rml', 'rll', 'fissure-h', 'fissure-r', *INTRA], 'opacity': {'heart': 0.3},
             'highlight': ['azygos'], 'danger': ['svc', 'trachea', 'n-vagus-r'], 'labels': ['svc', 'esophagus'],
             'action': {'kind': 'staple', 'label': 'Staple the azygos arch', 'port': 'thor-r', 'ids': ['azygos'], 'reload': 'vascular'},
             'ct': ct('azygos', 'axial', 'mediastinum')},
            {'id': f'{pre}-mobilise', 'phase': 'Chest', 'seq': seq_, 'title': 'Mobilise the oesophagus en bloc',
             'body': '<p>Open the pleura in front of the <b>aorta</b> and along the <b>pericardium</b>. Take the oesophagus with its <b>periesophageal fat and nodes</b>, including the <b>subcarinal (station 7)</b> packet, from the hiatus to '
                     + ('above the azygos arch.' if level_top == 'ivor' else 'the thoracic inlet, keeping close to the oesophagus near the trachea to spare the recurrent nerves.') +
                     '</p><p>Clip the aortic oesophageal branches. Watch the <b>membranous trachea and left main bronchus</b> in front, and the <b>thoracic duct</b> behind.</p>',
             'view': back(mid_eso, 360), 'show': ESO, 'hide': ['rul', 'rml', 'rll', 'fissure-h', 'fissure-r', *INTRA], 'opacity': {'heart': 0.3},
             'highlight': ['esophagus'], 'danger': ['thoracic-duct', 'aorta', 'trachea', 'br-left-main'],
             'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Mobilise the oesophagus', 'port': 'thor-r',
                        'path': [R(gej + V([14, -4, 25])), R(mid_eso + V([14, -6, -20])), R(mid_eso + V([14, -6, 25])), R((Lm('eso-ivor') if level_top == 'ivor' else eso_top - V([0, 0, 20])) + V([14, -4, 0]))]},
             'ct': ct('esophagus', 'axial', 'mediastinum')},
            {'id': f'{pre}-duct', 'phase': 'Chest', 'seq': seq_, 'title': 'Ligate the thoracic duct (low, above the hiatus)',
             'body': '<p>Many units ligate the duct routinely: <b>mass-ligate all the tissue between the aorta and the azygos</b> just above the hiatus, on the front of the spine. '
                     'A missed duct injury declares itself as a milky drain output once feeding starts.</p>',
             'view': back(Lm('td-ligation'), 260), 'show': ESO, 'hide': ['rul', 'rml', 'rll', 'fissure-h', 'fissure-r', *INTRA], 'opacity': {'heart': 0.3, 'esophagus': 0.5},
             'highlight': ['thoracic-duct'], 'danger': ['aorta', 'azygos'],
             'action': {'kind': 'ligate', 'label': 'Mass-ligate the duct', 'port': 'thor-r', 'ids': ['thoracic-duct'], 'keep': True},
             'ct': ct(R(Lm('td-ligation')), 'axial', 'mediastinum')},
        ]

    cut_ivor = {'kind': 'staple', 'label': 'Divide the oesophagus', 'port': 'thor-r', 'ids': ['esophagus'], 'reload': 'tissue', 'at': R(Lm('eso-ivor')), 'axis': LM['eso-ivor-axis'], 'radius': 11}
    cut_neck = {'kind': 'staple', 'label': 'Divide the oesophagus in the neck', 'port': 'neck', 'ids': ['esophagus'], 'reload': 'tissue', 'at': R(Lm('eso-neck')), 'axis': LM['eso-neck-axis'], 'radius': 11}
    neck = lambda pre, seq_: {'id': f'{pre}-neck', 'phase': 'Neck', 'seq': seq_, 'title': 'Left neck: find the cervical oesophagus',
                              'body': '<p>Head turned to the right. Incision along the <b>anterior border of the left sternocleidomastoid</b>. Retract the <b>carotid sheath laterally</b>, divide the omohyoid and the middle thyroid vein, '
                                      'and reach the oesophagus on the prevertebral fascia.</p><p>The <b>left recurrent laryngeal nerve</b> lies in the tracheo-oesophageal groove: no metal retractor against it, finger dissection only.</p>',
                              'view': front(Lm('neck'), 300, (-0.45, 1, 0.3)), 'show': ['incision-neck', 'esophagus', 'trachea', 'thyroid', 'n-rln', 'lcca', 'aorta'], 'hide': LUNGS_ALL + INTRA,
                              'highlight': ['esophagus'], 'danger': ['n-rln', 'lcca', 'thyroid'], 'labels': ['incision-neck', 'trachea'],
                              'ct': ct(R(Lm('eso-neck')), 'axial', 'mediastinum')}
    conduit = lambda pre, seq_, which: {'id': f'{pre}-conduit', 'phase': 'Conduit', 'seq': seq_, 'title': 'Make the conduit and bring it up',
                                        'body': '<p>Staple the lesser curvature from below the cardia to make a <b>4–5 cm tube</b> of greater curvature on the <b>right gastroepiploic artery</b>. '
                                                + ('Pull it up through the hiatus into the right chest, without twisting (staple line to the right).' if which == 'chest' else 'Pass it through the posterior mediastinum (the oesophageal bed) to the neck in a plastic sleeve, without twisting.') +
                                                '</p><p>Check the colour of the tip: poor perfusion there is what leaks.</p>',
                                        'view': back(mid_eso, 480) if which == 'chest' else front(mid_eso, 520, (0.3, 1, 0.2)),
                                        'show': [*ESO, 'rgea'], 'hide': LUNGS_ALL + INTRA + ['esophagus'], 'opacity': {'heart': 0.25, 'stomach': 0.35},
                                        'highlight': [f'conduit-{which}'], 'danger': ['rgea'],
                                        'action': {'kind': 'reveal', 'label': 'Bring up the conduit', 'port': 'thor-r' if which == 'chest' else 'neck', 'ids': [f'conduit-{which}']},
                                        'ct': ct(R(mid_eso), 'coronal', 'mediastinum')}
    anast = lambda pre, seq_, which: {'id': f'{pre}-anast', 'phase': 'Anastomosis', 'seq': seq_, 'title': 'The anastomosis',
                                      'body': ('<p><b>Intrathoracic</b>, at or above the azygos level: circular stapler (25–28 mm) with the anvil in the oesophagus and the gun through the conduit, or linear side-to-side. '
                                               'Close the conduit tip, wrap the anastomosis with omentum. A leak here is a mediastinitis: prevention is everything.</p>'
                                               if which == 'chest' else
                                               '<p><b>Cervical</b>: hand-sewn single layer or a linear-stapled side-to-side (Orringer). More leaks than in the chest, but a neck leak usually drains through the wound and heals.</p>')
                                              + '<p>Nasogastric tube past the anastomosis; feeding jejunostomy by unit policy.</p>',
                                      'view': back(Lm(f'anast-{which}'), 300) if which == 'chest' else front(Lm('anast-neck'), 280, (-0.4, 1, 0.3)),
                                      'show': [*ESO, f'conduit-{which}'], 'hide': LUNGS_ALL + INTRA, 'opacity': {'heart': 0.25, 'stomach': 0.3, f'conduit-{which}': 0.85},
                                      'highlight': [f'anast-{which}'], 'danger': ['trachea', 'n-rln'] if which == 'neck' else ['trachea'],
                                      'action': {'kind': 'reveal', 'label': 'Make the anastomosis', 'port': 'thor-r' if which == 'chest' else 'neck', 'ids': [f'anast-{which}']},
                                      'ct': ct(R(Lm(f'anast-{which}')), 'axial', 'mediastinum')}
    after = lambda pre, seq_: {'id': f'{pre}-after', 'phase': 'After', 'seq': seq_, 'title': 'What goes wrong',
                               'body': '<p><b>Anastomotic leak</b> and <b>conduit necrosis</b> (fever, arrhythmia, effluent in the drain: contrast study or endoscopy). <b>Chylothorax</b> (milky drain output once fed). '
                                       '<b>Recurrent laryngeal nerve palsy</b> (hoarseness, aspiration), mostly after neck dissection. Pneumonia above all: early mobilisation, physiotherapy, sitting up.</p>',
                               'view': front(mid_eso, 560, (0.35, 1, 0.2)), 'show': ESO, 'hide': LUNGS_ALL + INTRA, 'opacity': {'heart': 0.25},
                               'labels': ['thoracic-duct', 'n-rln'], 'ct': ct(R(mid_eso), 'coronal', 'mediastinum')}

    il = [e_anat('il'), *lap('il', 1), *r_chest('il', 2, 'ivor'),
          {'id': 'il-divide', 'phase': 'Chest', 'seq': 2, 'title': 'Divide the oesophagus above the azygos',
           'body': '<p>With the stomach already mobilised from below, divide the oesophagus <b>above the azygos arch</b> (tissue load, or open for a purse-string for the anvil). '
                   'Pull the specimen with the lesser curvature and its nodes into the chest.</p>',
           'view': back(Lm('eso-ivor'), 280), 'show': ESO, 'hide': ['rul', 'rml', 'rll', 'fissure-h', 'fissure-r', *INTRA], 'opacity': {'heart': 0.3},
           'highlight': ['esophagus'], 'danger': ['trachea', 'n-vagus-r'], 'action': cut_ivor, 'ct': ct(R(Lm('eso-ivor')), 'axial', 'mediastinum')},
          conduit('il', 3, 'chest'), anast('il', 4, 'chest'), after('il', 5)]
    mk = [e_anat('mk'), *r_chest('mk', 1, 'neck'), *lap('mk', 2), neck('mk', 3),
          {'id': 'mk-divide', 'phase': 'Neck', 'seq': 3, 'title': 'Divide the oesophagus in the neck',
           'body': '<p>Encircle the cervical oesophagus with a finger from the neck, meeting the thoracic dissection. Divide it low in the neck and deliver the specimen through the abdomen.</p>',
           'view': front(Lm('eso-neck'), 280, (-0.45, 1, 0.3)), 'show': ['esophagus', 'trachea', 'thyroid', 'n-rln', 'lcca'], 'hide': LUNGS_ALL + INTRA,
           'highlight': ['esophagus'], 'danger': ['n-rln', 'trachea'], 'action': cut_neck, 'ct': ct(R(Lm('eso-neck')), 'axial', 'mediastinum')},
          conduit('mk', 4, 'neck'), anast('mk', 5, 'neck'), after('mk', 6)]
    trh = [e_anat('th'), *lap('th', 1),
           {'id': 'th-hiatus', 'phase': 'Hiatus', 'seq': 2, 'title': 'Transhiatal: blunt dissection from below',
            'body': '<p>Open the hiatus widely (divide the crura forward). Under direct vision, then with the hand flat on the oesophagus, free it from the <b>aorta</b> behind, the <b>pericardium</b> in front and both pleurae, as high as the hand reaches (the carina).</p>'
                    '<p>Watch the blood pressure: the heart is compressed by the hand.</p>',
            'view': front(mid_eso, 420, (0.1, 1, -0.45)), 'show': ESO, 'hide': LUNGS_ALL + INTRA, 'opacity': {'heart': 0.25},
            'highlight': ['esophagus'], 'danger': ['aorta', 'azygos', 'trachea', 'thoracic-duct'],
            'action': {'kind': 'dissect', 'tool': 'peanut', 'label': 'Free it from below', 'port': 'lap', 'path': [R(gej + V([0, 8, 10])), R(mid_eso + V([0, 10, -30])), R(mid_eso + V([0, 10, 20]))]},
            'ask': ask('During transhiatal blunt dissection the anaesthetist reports a large air leak. What has happened?', 'A tear in the membranous trachea or left main bronchus',
                       'The membranous airway lies directly on the oesophagus; advance the tube past the tear and repair it through a right thoracotomy or via the neck.', 'A pneumothorax', 'An azygos tear'),
            'ct': ct(R(mid_eso), 'sagittal', 'mediastinum')},
           neck('th', 3),
           {'id': 'th-above', 'phase': 'Neck', 'seq': 3, 'title': 'Blunt dissection from above, then divide',
            'body': '<p>From the neck, free the upper oesophagus with a finger to meet the hand from below. Divide it in the neck and draw the specimen down into the abdomen.</p>',
            'view': front(Lm('eso-neck'), 300, (-0.45, 1, 0.3)), 'show': ['esophagus', 'trachea', 'thyroid', 'n-rln', 'lcca', 'azygos'], 'hide': LUNGS_ALL + INTRA,
            'highlight': ['esophagus'], 'danger': ['n-rln', 'trachea', 'azygos'], 'action': cut_neck, 'ct': ct(R(Lm('eso-neck')), 'axial', 'mediastinum')},
           conduit('th', 4, 'neck'), anast('th', 5, 'neck'), after('th', 6)]
    B4['eso-ivor'] = ('oesophagectomy', 'Oesophagectomy', 'Ivor Lewis', 'right', 'Abdomen, then right chest: anastomosis in the chest above the azygos.', il,
                      seq(('Anatomy', 'other'), ('Abdomen', 'other'), ('Chest', 'vein'), ('Conduit', 'other'), ('Anastomosis', 'bronchus'), ('After', 'other')), 'Oesophagus', 'eso')
    B4['eso-mckeown'] = ('oesophagectomy', 'Oesophagectomy', 'McKeown (three-stage)', 'right', 'Right chest, abdomen, then left neck: anastomosis in the neck.', mk,
                         seq(('Anatomy', 'other'), ('Chest', 'vein'), ('Abdomen', 'other'), ('Neck', 'other'), ('Conduit', 'other'), ('Anastomosis', 'bronchus'), ('After', 'other')), 'Oesophagus', 'eso')
    B4['eso-transhiatal'] = ('oesophagectomy', 'Oesophagectomy', 'Transhiatal', 'right', 'Abdomen and left neck, no thoracotomy: blunt mediastinal dissection.', trh,
                             seq(('Anatomy', 'other'), ('Abdomen', 'other'), ('Hiatus', 'other'), ('Neck', 'other'), ('Conduit', 'other'), ('Anastomosis', 'bronchus'), ('After', 'other')), 'Oesophagus', 'eso')

    # ---------------------------------------------------------------- thoracic duct ligation
    tdl, tdc = Lm('td-ligation'), Lm('td-cross')
    DUCT = ['thoracic-duct', 'cisterna', 'aorta', 'azygos', 'esophagus', 'lbcv', 'heart']
    duct = [
        {'id': 'td-anat', 'phase': 'Anatomy', 'seq': 0, 'title': 'The course of the thoracic duct',
         'body': '<p>From the <b>cisterna chyli</b> (L1–L2, behind and right of the aorta) through the <b>aortic hiatus</b>, up on the right front of the vertebral bodies <b>between the aorta and the azygos</b>, behind the oesophagus. '
                 'At <b>T4–T6 it crosses to the left</b> and climbs on the left of the oesophagus to end in the <b>left venous angle</b> (internal jugular and subclavian).</p>'
                 '<p>So an injury <b>below T5 gives a right chylothorax</b>, above it a left one. Doubled and plexiform ducts are common.</p>',
         'view': front(V(S['thoracic-duct']['centroid']), 520, (0.55, -0.8, 0.2)), 'show': DUCT, 'hide': LUNGS_ALL + INTRA, 'opacity': {'heart': 0.2, 'aorta': 0.7, 'esophagus': 0.55}, 'spin': True,
         'highlight': ['thoracic-duct'], 'labels': ['cisterna', 'azygos', 'aorta', 'esophagus', 'lbcv'],
         'ask': ask('After a left upper lobectomy, a patient has a milky left pleural effusion. At what level is the duct most likely injured?', 'Above T5, where it runs on the left',
                    'The duct crosses from right to left at T4-T6: upper injuries leak into the left chest, lower ones into the right.', 'At the cisterna chyli', 'Below T8'),
         'ct': ct(R(tdc), 'axial', 'mediastinum')},
        {'id': 'td-decide', 'phase': 'Decision', 'seq': 1, 'title': 'Chylothorax: when to operate',
         'body': '<p>Milky fluid; confirm with <b>triglycerides above 1.24 mmol/L (110 mg/dL)</b> or chylomicrons. Start with drainage, nil by mouth or a fat-free / medium-chain diet, parenteral nutrition, octreotide.</p>'
                 '<p>Operate (or embolise) for <b>high output</b> (commonly more than 1 L a day, or more than 10 mL/kg/day in a child), failure after about <b>5–7 days</b>, or nutritional and immune depletion. '
                 '<b>Thoracic duct embolisation</b> by interventional radiology is the alternative where available.</p>'
                 '<p>Give <b>cream or olive oil</b> 2–4 hours before surgery: the leak turns white and shows itself.</p>',
         'view': front(tdl, 420, (0.8, -0.5, 0.2)), 'show': DUCT, 'hide': LUNGS_ALL + INTRA, 'opacity': {'heart': 0.2}, 'labels': ['thoracic-duct'], 'ct': ct(R(tdl), 'axial', 'mediastinum')},
        {'id': 'td-setup', 'phase': 'Setup', 'seq': 2, 'title': 'Right VATS, whichever side the effusion is',
         'body': '<p>Left lateral decubitus, right VATS (three ports, low). The duct is ligated on the <b>right, just above the diaphragm</b>, where it is a single trunk in most patients, even for a left chylothorax.</p>',
         'view': {'frame': ['skin'], 'dir': [1, -0.35, 0.15], 'pad': 1.05}, 'pose': 'lateral', 'show': ['skin', *[i for i in ('port-r-posterior-utility', 'port-r-posterior-camera', 'port-r-posterior-posterior') if has(i)]],
         'labels': [i for i in ('port-r-posterior-utility', 'port-r-posterior-camera', 'port-r-posterior-posterior') if has(i)], 'ct': ct(R(tdl), 'axial', 'lung')},
        {'id': 'td-ligament', 'phase': 'Exposure', 'seq': 3, 'title': 'Divide the inferior pulmonary ligament, lift the lung',
         'body': '<p>Retract the right lower lobe up and forward and divide the ligament. Open the pleura over the <b>triangle</b> formed by the <b>aorta</b>, the <b>azygos vein</b> and the <b>vertebral bodies</b>, just above the diaphragm.</p>',
         'view': back(tdl, 300), 'show': DUCT, 'hide': ['rul', 'rml', 'fissure-h', *INTRA], 'opacity': {'rll': 0.25, 'fissure-r': 0.2, 'heart': 0.2, 'esophagus': 0.5},
         'highlight': ['ipl-r'], 'danger': ['esophagus', 'azygos'], 'labels': ['aorta', 'azygos'],
         'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Divide the ligament', 'port': 'port-r-posterior-posterior', 'remove': ['ipl-r'],
                    'path': [R(V(S['ipl-r']['bbox'][0]) + V([4, 6, 4])), R(V(S['ipl-r']['centroid']) + V([4, 4, 0])), R(V(S['ipl-r']['bbox'][1]) + V([4, 4, -4]))] if has('ipl-r') else [R(tdl)]},
         'ct': ct(R(tdl), 'axial', 'mediastinum')},
        {'id': 'td-ligate', 'phase': 'Duct', 'seq': 4, 'title': 'Mass ligation above the diaphragm',
         'body': '<p>Pass a right-angle clamp round <b>all the tissue between the aorta and the azygos</b>, on the front of the spine, 2–5 cm above the hiatus. Tie it with non-absorbable ligatures (two) or clip it. '
                 'You do not need to see the duct itself.</p><p>Watch the <b>intercostal arteries</b> behind, the <b>oesophagus</b> in front, and the azygos on the right.</p>',
         'view': back(tdl, 240), 'show': DUCT, 'hide': ['rul', 'rml', 'rll', 'fissure-h', 'fissure-r', 'ipl-r', *INTRA], 'opacity': {'heart': 0.2, 'esophagus': 0.45},
         'highlight': ['thoracic-duct'], 'danger': ['aorta', 'azygos', 'esophagus'],
         'action': {'kind': 'ligate', 'label': 'Mass-ligate', 'port': 'port-r-posterior-posterior', 'ids': ['thoracic-duct'], 'keep': True},
         'ct': ct(R(tdl), 'axial', 'mediastinum')},
        {'id': 'td-check', 'phase': 'Close', 'seq': 5, 'title': 'Check, glue, drain',
         'body': '<p>Look for any white leak above the ligature (cream helps); clip or oversew it. Fibrin glue and a mechanical or talc <b>pleurodesis</b> by preference. One drain; record the output after feeding restarts.</p>',
         'view': back(tdl, 300), 'show': DUCT, 'hide': ['rul', 'rml', 'rll', 'fissure-h', 'fissure-r', 'ipl-r', *INTRA], 'opacity': {'heart': 0.2, 'esophagus': 0.45},
         'labels': ['thoracic-duct'], 'ct': ct(R(tdl), 'axial', 'mediastinum')},
    ]
    B4['duct'] = ('duct', 'Thoracic duct ligation', 'Right VATS, supradiaphragmatic', 'right', 'Mass ligation between the aorta and the azygos above the right hemidiaphragm.', duct,
                  seq(('Anatomy', 'other'), ('Decide', 'other'), ('Setup', 'other'), ('Exposure', 'other'), ('Ligate', 'artery'), ('Check', 'other')), 'Oesophagus', 'duct')

    # ---------------------------------------------------------------- empyema
    EMP = ['lul', 'lll', 'fissure', 'peel-l', 'empyema-l', 'heart', 'aorta']
    hil = LM['hilum-l']; emp_c = Lm('empyema')
    SHR = {'ids': ['lul', 'lll', 'fissure', 'peel-l', 'lul-arteries', 'lul-veins', 'lul-bronchi', 'lll-arteries', 'lll-veins', 'lll-bronchi'], 'pivot': hil, 'scale': 0.82}
    lat = lambda t, dist=380: front(t, dist, (-1, -0.45, 0.2))
    stages = {'id': 'em-stages', 'phase': 'Anatomy', 'seq': 0, 'title': 'Empyema: three stages',
              'body': '<p><b>I, exudative</b> (days): thin fluid, the lung still expands: a chest drain and antibiotics. '
                      '<b>II, fibrinopurulent</b> (1–2 weeks): fibrin septa and loculations: drain plus intrapleural <b>tPA and DNase</b> (MIST2), or early <b>VATS debridement</b>. '
                      '<b>III, organising</b> (after 3–6 weeks): a thick <b>peel</b> on the visceral pleura traps the lung: <b>decortication</b>.</p>'
                      '<p>Here the left lung is shown <b>trapped</b>, smaller than the chest, under its peel, with pus in the posterior costophrenic gutter.</p>',
              'view': lat(emp_c + V([10, 30, 60]), 460), 'show': EMP, 'shrink': SHR, 'hide': INTRA, 'opacity': {'lul': 0.55, 'lll': 0.55, 'heart': 0.3},
              'highlight': ['peel-l'], 'danger': [], 'labels': ['empyema-l', 'lll', 'lul'], 'spin': True,
              'ask': ask('A patient with a parapneumonic effusion has septated fluid on ultrasound and a pleural fluid pH of 7.0. Stage?', 'II, fibrinopurulent',
                         'Septations and a low pH or glucose mean a complicated effusion or empyema in the fibrinopurulent stage: drain it, and add fibrinolytics or VATS if it does not clear.', 'I, exudative', 'III, organising'),
              'ct': ct(R(emp_c), 'axial', 'lung')}
    vats_e = [stages,
              {'id': 'ev-setup', 'phase': 'Setup', 'seq': 1, 'title': 'Ports over the collection',
               'body': '<p>Right lateral decubitus. Place the first port where the <b>collection is largest on ultrasound or CT</b> (often low and posterior), by open finger dissection: the lung may be stuck to the wall. '
                       'Two more ports under vision.</p>',
               'view': {'frame': ['skin'], 'dir': [-1, -0.35, 0.15], 'pad': 1.05}, 'pose': 'lateral', 'show': ['skin', *[i for i in ('port-posterior-utility', 'port-posterior-camera', 'port-posterior-posterior') if has(i)]],
               'shrink': SHR, 'labels': [i for i in ('port-posterior-utility', 'port-posterior-camera', 'port-posterior-posterior') if has(i)], 'ct': ct(R(emp_c), 'axial', 'lung')},
              {'id': 'ev-debride', 'phase': 'Debride', 'seq': 2, 'title': 'Break the loculations, evacuate the pus',
               'body': '<p>With a sucker, a sponge on a holder and the camera, <b>break every septum</b> so the space becomes one cavity; take the fibrin off the lung and the diaphragm. '
                       'Send pus and pleura for culture. Irrigate with warm saline.</p>',
               'view': lat(emp_c, 320), 'show': EMP, 'shrink': SHR, 'hide': INTRA, 'opacity': {'lul': 0.45, 'lll': 0.45, 'heart': 0.3},
               'highlight': ['empyema-l'], 'danger': ['lll'],
               'action': {'kind': 'dissect', 'tool': 'peanut', 'label': 'Debride the cavity', 'port': 'port-posterior-camera' if has('port-posterior-camera') else 'thor-l', 'remove': ['empyema-l'],
                          'path': [R(emp_c + V([-10, 10, 25])), R(emp_c + V([-5, 0, 0])), R(emp_c + V([-10, -10, -20]))]},
               'ct': ct(R(emp_c), 'axial', 'lung')},
              {'id': 'ev-peel', 'phase': 'Decorticate', 'seq': 3, 'title': 'Early peel: strip it thoracoscopically',
               'body': '<p>A young peel strips with a peanut: find the plane between the peel and the visceral pleura, then ask the anaesthetist to <b>inflate</b> the lung as you go: it helps the peel lift. '
                       'A thick, old peel is a conversion to thoracotomy.</p>',
               'view': lat(V(pt('lll')), 420), 'show': EMP, 'shrink': SHR, 'hide': INTRA, 'opacity': {'lul': 0.6, 'lll': 0.6, 'heart': 0.3},
               'highlight': ['peel-l'], 'danger': ['lll', 'lul'],
               'action': {'kind': 'decorticate', 'label': 'Strip the peel', 'port': 'port-posterior-utility' if has('port-posterior-utility') else 'thor-l', 'ids': ['peel-l'],
                          'expand': {'ids': SHR['ids'], 'pivot': hil, 'from': SHR['scale']}},
               'ct': ct(R(emp_c), 'axial', 'lung')},
              {'id': 'ev-drain', 'phase': 'Close', 'seq': 4, 'title': 'Re-expansion and drains',
               'body': '<p>The lung should fill the chest. Two large drains, apical and basal-posterior. Air leak from small tears is expected; persistent space or leak means the peel was left.</p>',
               'view': lat(V(pt('lll')), 460), 'show': ['lul', 'lll', 'fissure', 'heart'], 'hide': INTRA, 'opacity': {'lul': 0.6, 'lll': 0.6}, 'labels': ['lul', 'lll'], 'ct': ct(R(emp_c), 'axial', 'lung')}]
    open_e = [stages,
              {**thoracotomy_step('eo', 'left'), 'seq': 1, 'pose': 'lateral', 'shrink': SHR,
               'body': '<p>Right lateral decubitus. Posterolateral thoracotomy, often through the <b>6th space</b> for a basal cavity; a rib may be resected. The ribs are often crowded by the contracted hemithorax: spread slowly.</p>'},
              {'id': 'eo-evacuate', 'phase': 'Debride', 'seq': 2, 'title': 'Open the cavity, evacuate',
               'body': '<p>Enter the cavity through the <b>parietal peel</b>, suck out pus and debris, and send it for culture.</p>',
               'view': lat(emp_c, 320), 'show': EMP, 'shrink': SHR, 'hide': INTRA, 'opacity': {'lul': 0.45, 'lll': 0.45, 'heart': 0.3},
               'highlight': ['empyema-l'], 'action': {'kind': 'dissect', 'tool': 'peanut', 'label': 'Evacuate the cavity', 'port': 'thor-l', 'remove': ['empyema-l'],
                                                        'path': [R(emp_c + V([-10, 10, 25])), R(emp_c), R(emp_c + V([-10, -10, -20]))]},
               'ct': ct(R(emp_c), 'axial', 'lung')},
              {'id': 'eo-peel', 'phase': 'Decorticate', 'seq': 3, 'title': 'Decorticate the lung',
               'body': '<p>Incise the visceral peel with a knife until the <b>glistening visceral pleura</b> bulges into the cut (inflation helps); then separate the peel with a peanut and scissors, lobe by lobe, and in the fissure. '
                       'Stay <b>on the peel, not in the lung</b>: tears leak air for days.</p>'
                       '<p>Free the diaphragm too, so it moves. Decorticate the parietal side only as needed for the lung to meet the wall.</p>',
               'view': lat(V(pt('lll')), 420), 'show': EMP, 'shrink': SHR, 'hide': INTRA, 'opacity': {'lul': 0.6, 'lll': 0.6, 'heart': 0.3},
               'highlight': ['peel-l'], 'danger': ['lll', 'lul', 'n-phrenic'],
               'action': {'kind': 'decorticate', 'label': 'Peel off the cortex', 'port': 'thor-l', 'ids': ['peel-l'], 'expand': {'ids': SHR['ids'], 'pivot': hil, 'from': SHR['scale']}},
               'ask': ask('The lung does not re-expand after decortication and the space persists. Options?', 'Muscle flap or thoracoplasty to fill the space, or an open window (Eloesser) in the frail',
                          'A residual space re-infects: fill it (serratus or latissimus flap, limited thoracoplasty) or leave it open to drain.', 'A second chest drain only', 'Pleurodesis'),
               'ct': ct(R(emp_c), 'axial', 'lung')},
              {'id': 'eo-close', 'phase': 'Close', 'seq': 4, 'title': 'Re-expansion, drains, close',
               'body': '<p>Two large drains. For the frail patient who cannot stand a decortication, an <b>open thoracostomy window (Eloesser flap)</b> drains a chronic cavity instead.</p>',
               'view': lat(V(pt('lll')), 460), 'show': ['lul', 'lll', 'fissure', 'heart'], 'hide': INTRA, 'opacity': {'lul': 0.6, 'lll': 0.6}, 'labels': ['lul', 'lll'], 'ct': ct(R(emp_c), 'axial', 'lung')}]
    B4['emp-vats'] = ('empyema', 'Empyema and decortication', 'VATS debridement', 'left', 'Stage II: break the loculations, evacuate, strip an early peel.', vats_e,
                      seq(('Stages', 'other'), ('Ports', 'other'), ('Debride', 'other'), ('Peel', 'fissure'), ('Drains', 'other')), 'Pleura', 'empyema')
    B4['emp-open'] = ('empyema', 'Empyema and decortication', 'Open decortication', 'left', 'Stage III: thoracotomy, evacuate, peel the cortex off the lung.', open_e,
                      seq(('Stages', 'other'), ('Thoracotomy', 'other'), ('Evacuate', 'other'), ('Decorticate', 'fissure'), ('Close', 'other')), 'Pleura', 'empyema')

    for key, (op_, opName, appr, side, summ, steps_, sq, group, src) in B4.items():
        for s in steps_:
            named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
            quiet = [i for i in ('n-phrenic', 'n-vagus', 'n-rln', 'lig-art', 'esophagus', 'ipl', 'ipl-r', 'n-phrenic-r', 'n-vagus-r', 'azygos', *[k for k in S if S[k]['group'] == 'nodes'],
                                 *[f'vert-t{i}' for i in range(2, 11)]) if i not in named]
            s['hide'] = [*s.get('hide', []), *[i for i in quiet if has(i)]]
            if s.get('action', {}).get('kind') != 'thoracotomy':   # ribs from an earlier thoracotomy step out of the way
                s['hide'] += [k for k in S if k.startswith('rib-') and k not in named]
            if op_ == 'thymectomy': s['hide'] += [i for i in LUNGS_ALL + INTRA if i not in named]
            s['opacity'] = {**{f'vert-t{i}': 0.22 for i in range(2, 11)}, **s.get('opacity', {})}
            for kk in ('highlight', 'danger', 'labels', 'show', 'hide'):
                if kk in s: s[kk] = [i for i in s[kk] if has(i) or i == 'skin']
        procs[f'b4-{key}'] = {'id': f'b4-{key}', 'op': op_, 'opName': opName, 'side': side, 'name': opName, 'approach': appr, 'summary': summ,
                              'ports': [], 'steps': steps_, 'sources': B4SRC[src], 'sequence': sq, 'group': group}
# ==================================================================================================== airway: cervical tracheal resection
TR_RES_OK = has('trach-steno') and 'cut-up' in LM
if TR_RES_OK:
    Ln = lambda k: V(LM[k])
    tlook = lambda tgt, d, dist=300.0: {'eye': R(V(tgt) + V(d) / np.linalg.norm(d) * dist), 'target': R(tgt)}
    r_tr, L_st = LM['trach-dims'][0], LM['trach-dims'][1]
    AX = V(LM['trach-axis'])
    STEN, CU, CL, CRI = Ln('stenosis'), Ln('cut-up'), Ln('cut-lo'), Ln('cricoid')
    NECK_OFF = [i for i in ('lul', 'lll', 'rul', 'rml', 'rll', 'fissure', 'fissure-h', 'fissure-r', 'trachea', 'br-left-main', 'br-right-main', 'heart', 'laa') if has(i)]
    AIR = [i for i in ('trach-prox', 'trach-dist', 'trach-steno', 'cricoid', 'thyroid-cart') if has(i)]
    NERV = [i for i in ('n-rln-neck-l', 'n-rln-neck-r', 'n-sln-ext-l', 'n-sln-ext-r') if has(i)]
    ant = V([0, 1, 0.15]); obl_r = V([0.9, 0.45, 0.2]); obl_l = V([-0.9, 0.45, 0.2])
    ring_path = lambda c, r, n=7: [R(c + V([np.sin(t) * (r + 2), np.cos(t) * (r + 2), 0])) for t in np.linspace(-1.9, 1.9, n)]
    TRSRC = [
        {'title': 'Auchincloss HG, Wright CD. Complications after tracheal resection and reconstruction: prevention and treatment. J Thorac Dis 2016; and Tracheal stenosis: resection and reconstruction. Ann Cardiothorac Surg 2018;7(2):306-308', 'url': 'https://www.annalscts.com/article/view/16464/16676'},
        {'title': 'Grillo HC. Surgery of the Trachea and Bronchi. BC Decker, 2004', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Grillo+tracheal+resection+reconstruction+postintubation'},
        {'title': 'Tracheal resection and anastomosis in postintubation tracheal stenosis: a systematic review. Eur J Cardiothorac Surg 2024', 'url': 'https://pubmed.ncbi.nlm.nih.gov/39254596/'},
        {'title': 'Tracheal Resection. StatPearls, NCBI Bookshelf', 'url': 'https://www.ncbi.nlm.nih.gov/books/NBK563234/'},
        {'title': 'Tracheal sleeve resection with suprahyoid and infrahyoid release. Iowa Head and Neck Protocols', 'url': 'https://iowaprotocols.medicine.uiowa.edu/protocols/tracheal-sleeve-resection-suprahyoid-and-infrahyoid-release'},
    ]
    DIST_UP = {'ids': ['trach-dist'], 'offset': R(AX * L_st), 'opacity': 1.0}
    tr_steps = [
        {'id': 'tr-anatomy', 'phase': 'Anatomy', 'seq': 0, 'title': 'The cervical trachea and its nerves',
         'body': '<p>About 11 cm long in an adult, 18–22 C-shaped rings; roughly half lies in the neck with the neck extended. The <b>thyroid isthmus</b> crosses rings 2–4; the <b>innominate artery</b> crosses the front low in the neck.</p>'
                 '<p>Blood supply is <b>segmental and lateral</b> (mostly the inferior thyroid artery): free the trachea circumferentially only for about <b>1 cm beyond each cut</b>.</p>'
                 '<p>The <b>recurrent laryngeal nerves</b> run up in the <b>tracheo-oesophageal grooves</b> and enter the larynx behind the cricothyroid joints; the right loops under the subclavian and runs more obliquely. '
                 'The <b>external branch of the superior laryngeal nerve</b> runs with the superior thyroid artery near the upper pole down to cricothyroid; the <b>internal branch</b> pierces the thyrohyoid membrane.</p>',
         'view': tlook(STEN + V([0, 0, 15]), obl_r, 260), 'spin': True, 'show': [*AIR, *NERV, 'n-sln-int-l', 'n-sln-int-r', 'trach-vessels', 'thyroid', 'esophagus', 'lcca', 'rcca', 'bct'],
         'hide': NECK_OFF, 'opacity': {'thyroid': 0.35, 'esophagus': 0.6},
         'highlight': ['trach-steno'], 'danger': NERV, 'labels': ['cricoid', 'thyroid', 'bct', 'n-rln-neck-l', 'n-rln-neck-r', 'n-sln-ext-r', 'trach-vessels'],
         'ask': ask('How do you protect the recurrent laryngeal nerves during tracheal resection?', 'Keep the dissection on the tracheal wall; do not look for the nerves',
                    'They run in the tracheo-oesophageal grooves just beside the trachea; staying on the wall (especially laterally and behind) keeps them out of harm. Searching for them in scar injures them.',
                    'Identify and sling both nerves first', 'Divide the lateral pedicles widely'),
         'ct': ct(R(STEN), 'axial')},
        {'id': 'tr-decide', 'phase': 'Decision', 'seq': 1, 'title': 'Before you resect',
         'body': '<p><b>Rigid and flexible bronchoscopy</b>: the length of the stricture, its distance below the cords and the cricoid, the state of the mucosa; CT for length and extrinsic disease. Dilate to buy time if needed.</p>'
                 '<p>Operate when the inflammation has settled, the patient is <b>off steroids</b> and off the ventilator, and any tracheostomy can be closed at the same time. '
                 'Up to about <b>half the adult trachea (4–5 cm)</b> can be resected with mobilisation and release; less in children. Subglottic disease involving the cricoid needs a <b>cricotracheal (Pearson) resection</b> instead.</p>',
         'view': tlook(STEN, obl_l, 240), 'show': AIR, 'hide': NECK_OFF, 'highlight': ['trach-steno'], 'labels': ['trach-steno', 'cricoid'],
         'ask': ask('Roughly how much adult trachea can be resected with primary anastomosis?', 'About half (4–5 cm), with mobilisation and release manoeuvres',
                    'Beyond that, tension rises steeply and dehiscence and restenosis follow.', 'Up to three quarters', 'No more than 1 cm'),
         'ct': ct(R(STEN), 'coronal')},
        {'id': 'tr-incision', 'phase': 'Access', 'seq': 2, 'title': 'Position and collar incision',
         'body': '<p>Supine, a <b>shoulder roll</b>, the neck extended, the head on a ring. Anaesthesia: a small tube passed through or above the stricture (after dilatation if needed), or spontaneous ventilation.</p>'
                 '<p>A <b>low collar incision</b> two fingerbreadths above the sternal notch; include an old stoma. A partial upper sternal split is added only for low lesions.</p>',
         'view': tlook(Ln('collar'), V([0, 1, 0.35]), 330), 'show': ['skin', 'incision-collar', 'sternum'], 'opacity': {'skin': 1.0}, 'hide': NECK_OFF,
         'highlight': ['incision-collar'], 'labels': ['incision-collar'], 'ct': ct(R(Ln('collar')), 'sagittal', 'bone')},
        {'id': 'tr-flaps', 'phase': 'Access', 'seq': 3, 'title': 'Subplatysmal flaps; straps apart; isthmus divided',
         'body': '<p>Raise <b>subplatysmal flaps</b> up to the thyroid cartilage and down to the notch (the anterior jugular veins stay down on the straps). Separate the <b>strap muscles in the midline</b> and retract them laterally. '
                 '<b>Divide the thyroid isthmus</b> and oversew it; roll the lobes off the front of the trachea.</p>',
         'view': tlook(STEN + V([0, 0, 15]), ant, 280), 'show': ['platysma', 'straps-l', 'straps-r', 'thyroid', *AIR, 'incision-collar'], 'hide': NECK_OFF,
         'labels': ['platysma', 'straps-l', 'straps-r', 'thyroid'], 'danger': [],
         'action': {'kind': 'layers', 'label': 'Open the layers', 'port': 'neck-front', 'layers': [
             {'id': 'platysma', 'label': 'Platysma: flaps raised', 'fate': 'divide', 'point': R(Ln('collar')), 'dir': [0, 0, 1], 'open': 30},
             {'id': 'straps-l', 'label': 'Left strap muscles: retracted', 'fate': 'retract', 'offset': [-16, 0, 0]},
             {'id': 'straps-r', 'label': 'Right strap muscles: retracted', 'fate': 'retract', 'offset': [16, 0, 0]},
             {'id': 'thyroid', 'label': 'Thyroid isthmus: divided', 'fate': 'divide', 'point': R(V([STEN[0], V(S['thyroid']['centroid'])[1], V(S['thyroid']['centroid'])[2]])), 'dir': [1, 0, 0], 'open': 14}]},
         'ct': ct(R(STEN), 'axial')},
        {'id': 'tr-front', 'phase': 'Trachea', 'seq': 4, 'title': 'The front of the trachea, and the stricture',
         'body': '<p>Clear the <b>anterior wall</b> from the cricoid down past the stricture: this plane is avascular. Find the stricture from outside (scarred, narrowed; a needle through the wall seen with the bronchoscope marks its limits exactly).</p>'
                 '<p>Low in the field the <b>innominate artery</b> crosses in front: keep off it.</p>',
         'view': tlook(STEN + V([0, 0, 10]), ant, 220), 'show': [*AIR, 'bct', 'thyroid'], 'hide': NECK_OFF, 'opacity': {'thyroid': 0.3},
         'highlight': ['trach-steno'], 'danger': ['bct'],
         'action': {'kind': 'dissect', 'tool': 'peanut', 'label': 'Clear the front', 'port': 'neck-front',
                    'path': [R(CRI + V([0, r_tr + 3, -6])), R(CU + V([0, r_tr + 3, 0])), R(STEN + V([0, r_tr + 2, 0])), R(CL + V([0, r_tr + 3, -8]))]},
         'ct': ct(R(STEN), 'sagittal')},
        {'id': 'tr-around', 'phase': 'Trachea', 'seq': 5, 'title': 'Circumferential dissection, on the wall',
         'body': '<p>Encircle the trachea <b>only at the stricture</b>, staying <b>on the tracheal wall</b> laterally and behind (the plane between trachea and oesophagus). '
                 'Do not look for the recurrent nerves: they lie in the grooves just outside this plane. Preserve the lateral vessels beyond the segment.</p>',
         'view': tlook(STEN, obl_r, 200), 'show': [*AIR, *NERV[:2], 'trach-vessels', 'esophagus'], 'hide': NECK_OFF, 'opacity': {'esophagus': 0.6},
         'highlight': ['trach-steno'], 'danger': [*NERV[:2], 'esophagus', 'trach-vessels'],
         'action': {'kind': 'dissect', 'tool': 'peanut', 'label': 'Encircle the segment', 'port': 'neck-front', 'path': ring_path(STEN, r_tr)},
         'ct': ct(R(STEN), 'axial')},
        {'id': 'tr-lower', 'phase': 'Resection', 'seq': 6, 'title': 'Divide below; cross-field ventilation',
         'body': '<p>Place <b>lateral stay sutures</b> (2-0) through the full wall a ring below the stricture. Divide the trachea just below it; pass a sterile <b>armoured tube into the distal trachea</b> across the field and ventilate through it.</p>',
         'view': tlook(CL, ant + V([0.3, 0, 0.2]), 220), 'show': [*AIR, 'stay-sutures', *NERV[:2]], 'hide': NECK_OFF,
         'highlight': ['trach-steno'], 'danger': NERV[:2],
         'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Divide below the stricture', 'port': 'neck-front', 'path': ring_path(CL, r_tr), 'show': ['ett-crossfield']},
         'ct': ct(R(CL), 'axial')},
        {'id': 'tr-upper', 'phase': 'Resection', 'seq': 7, 'title': 'Divide above; the segment out',
         'body': '<p>Stay sutures above, then divide at healthy airway above the stricture (bevel it if the stricture is higher in front). Send the margins. The oral tube is pulled back above the field.</p>',
         'view': tlook(CU, ant + V([0.3, 0, 0.1]), 230), 'show': [*AIR, 'stay-sutures', 'ett-crossfield', *NERV[:2]], 'hide': NECK_OFF,
         'highlight': ['trach-steno'], 'danger': NERV[:2],
         'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Divide above; remove it', 'port': 'neck-front', 'path': ring_path(CU, r_tr), 'remove': ['trach-steno']},
         'ct': ct(R(CU), 'axial')},
        {'id': 'tr-tension', 'phase': 'Release', 'seq': 8, 'title': 'Test the tension; release if needed',
         'body': '<p>Flex the neck and draw the ends together on the <b>crossed stay sutures</b>: they should meet without strain. Mobilise the <b>front of the distal trachea</b> into the mediastinum by finger (pretracheal plane, avascular).</p>'
                 '<p>If the ends will not meet: a <b>suprahyoid laryngeal release</b> (Montgomery) gives 1–2 cm; a thyrohyoid release risks the <b>internal branch of the superior laryngeal nerve</b> (aspiration). Low lesions: hilar and pericardial release through the chest.</p>',
         'view': tlook((CU + CL) / 2, obl_r, 240), 'retract': DIST_UP, 'show': [*AIR, 'stay-sutures', 'ett-crossfield', *NERV], 'hide': NECK_OFF,
         'highlight': ['trach-dist', 'trach-prox'], 'danger': ['n-sln-ext-l', 'n-sln-ext-r'], 'labels': ['n-sln-ext-r'],
         'ct': ct(R(CU), 'sagittal')},
        {'id': 'tr-anast', 'phase': 'Anastomosis', 'seq': 9, 'title': 'The anastomosis: 4-0 PDS',
         'body': '<p><b>Membranous wall first</b>, behind: a running 4-0 PDS (or interrupted), bites about <b>3–4 mm from the edge and 3–4 mm apart</b>. '
                 '<b>Cartilaginous wall</b> in front: interrupted 4-0 PDS (or 4-0 Vicryl), each round a ring on both sides, <b>all placed before any is tied</b>, knots outside the lumen.</p>'
                 '<p>Take the cross-field tube out and <b>advance the oral tube past the anastomosis</b>. Flex the neck, cross and hold the stay sutures, then tie the anterior sutures (from the sides to the front). '
                 'Leak test under saline to 20–30 cmH2O. Cover with the strap muscles or thyroid; put tissue between the suture line and the innominate artery if they touch.</p>',
         'view': tlook(CU, ant + V([0.4, 0, 0.1]), 200), 'retract': DIST_UP, 'show': [*AIR, 'ett-oral', 'stay-sutures', *NERV[:2]], 'hide': [*NECK_OFF, 'ett-crossfield'], 'opacity': {'ett-oral': 0.5},
         'highlight': ['trach-prox', 'trach-dist'], 'danger': NERV[:2],
         'action': {'kind': 'anastomose', 'label': 'Sew the anastomosis', 'port': 'neck-front', 'at': R(CU - AX * 0.5), 'axis': R(AX), 'radius': r_tr},
         'ask': ask('When are the anterior (cartilaginous) sutures tied?', 'After all are placed, the neck flexed and the stay sutures crossed',
                    'Placing every suture first keeps the view open; flexion and the stay sutures take the tension off while they are tied.', 'One by one as each is placed', 'Before the membranous wall'),
         'ct': ct(R(CU), 'axial')},
        {'id': 'tr-after', 'phase': 'After', 'seq': 10, 'title': 'Guardian stitch, extubation, bronchoscopy',
         'body': '<p>A heavy <b>guardian (chin-to-chest) suture</b> keeps the neck flexed for about 7 days. <b>Extubate in theatre</b> where possible. Bronchoscopy before discharge (around day 7).</p>'
                 '<p>Watch for: <b>stridor</b> (oedema: steroids, racemic adrenaline, a small tube; dehiscence must be excluded), <b>air or wound infection</b> (a leak), <b>voice change or aspiration</b> (nerve injury). '
                 'In large series about <b>95% have a good airway</b>; restenosis around 4–5%, and dehiscence is the complication that kills.</p>',
         'view': tlook(CU, ant, 260), 'retract': DIST_UP, 'show': [*AIR, *NERV[:2]], 'hide': [*NECK_OFF, 'ett-crossfield', 'ett-oral'], 'labels': ['trach-prox', 'trach-dist'], 'ct': ct(R(CU), 'sagittal')},
    ]
    for s in tr_steps:
        if s.get('seq', 0) >= 4:   # flaps and straps already turned back: out of the way
            s['hide'] = [*s.get('hide', []), 'platysma', 'straps-l', 'straps-r']; s['opacity'] = {'thyroid': 0.3, **s.get('opacity', {})}
        named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
        s['hide'] = [*s.get('hide', []), *[i for i in ('n-phrenic', 'n-vagus', 'n-rln', 'n-phrenic-r', 'n-vagus-r', 'azygos', 'lig-art', 'lul-arteries', 'lul-veins', 'lul-bronchi', 'rul-arteries', 'rul-veins', 'rul-bronchi') if has(i) and i not in named]]
        s['opacity'] = {**{f'vert-t{i}': 0.25 for i in range(2, 11)}, 'aorta': 0.5, **s.get('opacity', {})}
        for kk in ('highlight', 'danger', 'labels', 'show'):
            if kk in s: s[kk] = [i for i in s[kk] if has(i)]
    procs['trachea-cervical'] = {'id': 'trachea-cervical', 'op': 'trachea', 'opName': 'Tracheal resection', 'side': 'both', 'name': 'Tracheal resection and reconstruction', 'approach': 'Cervical (collar incision)',
                                 'summary': 'Collar incision, subplatysmal flaps, stricture freed on the wall, resected, end-to-end 4-0 PDS anastomosis.', 'ports': [], 'steps': tr_steps, 'sources': TRSRC, 'group': 'Airway',
                                 'sequence': seq(('Anatomy', 'other'), ('Decide', 'other'), ('Incision', 'other'), ('Layers', 'other'), ('Front', 'other'), ('Around', 'other'), ('Below', 'bronchus'), ('Above', 'bronchus'), ('Tension', 'other'), ('Anastomosis', 'fissure'), ('After', 'other'))}
# ==================================================================================================== cardiac: mitral valve replacement
MVR_OK = has('la') and has('mitral-annulus') and 'mv-centre' in LM
if MVR_OK:
    Lc = lambda k: V(LM[k])
    clook = lambda tgt, d, dist=300.0: {'eye': R(V(tgt) + V(d) / np.linalg.norm(d) * dist), 'target': R(tgt)}
    MC, MN, MU = Lc('mv-centre'), V(LM['mv-normal']), V(LM['mv-anterior']); MR_ = LM['mv-dims'][0]
    st_b = S['sternum']['bbox']; ST_MID = V([(st_b[0][0] + st_b[1][0]) / 2, (st_b[0][1] + st_b[1][1]) / 2 + 4, (st_b[0][2] + st_b[1][2]) / 2])
    HEART_OFF = [i for i in ('heart', 'laa', 'lul', 'lll', 'rul', 'rml', 'rll', 'fissure', 'fissure-h', 'fissure-r', 'thymus') if has(i)]
    VALVE = [i for i in ('mitral-annulus', 'mv-ant-leaflet', 'mv-post-leaflet', 'chordae', 'papillary') if has(i)]
    DANGER = [i for i in ('circumflex', 'av-node', 'cusp-n', 'cusp-l', 'coronary-sinus') if has(i)]
    CH = [i for i in ('la', 'lv', 'ra', 'rv', 'myocardium', 'pa-trunk', 'aorta', 'svc') if has(i)]
    CANS = [i for i in ('can-aortic', 'can-svc', 'can-ivc', 'can-cp') if has(i)]
    FAINT = {'myocardium': 0.12, 'lv': 0.3, 'rv': 0.2, 'ra': 0.3, 'la': 0.3, 'pa-trunk': 0.4, 'aorta': 0.7}
    valve_view = clook(MC, MN * 1.0 + V([0.55, 0.35, 0.1]), 190)
    ext = lambda k, d=18: [R(Lc(k) + V([0, 0, d])), R(Lc(k)), R(Lc(k) - V([0, 0, d]))]
    MVSRC = [
        {'title': 'Kouchoukos NT, Blackstone EH, Hanley FL, Kirklin JK. Kirklin/Barratt-Boyes Cardiac Surgery, 4th ed. Elsevier 2013: mitral valve replacement', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Kirklin+Barratt-Boyes+cardiac+surgery+mitral+valve'},
        {'title': 'Otto CM, Nishimura RA, et al. 2020 ACC/AHA guideline for the management of patients with valvular heart disease. Circulation 2021;143:e72-e227', 'url': 'https://pubmed.ncbi.nlm.nih.gov/33332150/'},
        {'title': 'Vahanian A, et al. 2021 ESC/EACTS guidelines for the management of valvular heart disease. Eur Heart J 2022;43:561-632', 'url': 'https://pubmed.ncbi.nlm.nih.gov/34453165/'},
        {'title': 'Guiraudon GM, et al. The superior septal approach to the mitral valve', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Guiraudon+superior+septal+approach+mitral+valve'},
        {'title': 'Chitwood WR Jr, et al. Minimally invasive video-directed mitral valve surgery; transthoracic aortic cross-clamp', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Chitwood+transthoracic+aortic+crossclamp+minimally+invasive+mitral'},
        {'title': 'Praz F, Borger MA, et al. 2025 ESC/EACTS Guidelines for the management of valvular heart disease. Eur Heart J 2025', 'url': 'https://www.escardio.org/guidelines/clinical-practice-guidelines/all-esc-practice-guidelines/valvular-heart-disease/'},
        {'title': 'Englberger L, et al. Importance of implant technique on risk of major paravalvular leak after St. Jude mechanical valve replacement (AVERT). Eur J Cardiothorac Surg 2005;28:838-43', 'url': 'https://academic.oup.com/ejcts/article/28/6/838/377180'},
        {'title': 'Svenarud P, et al. Effect of CO2 insufflation on the number and behavior of air microemboli in open-heart surgery: a randomized clinical trial. Circulation 2004', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Svenarud+carbon+dioxide+insufflation+microemboli+randomized+Circulation+2004'},
        {'title': 'Martens S, et al. Carbon dioxide field flooding reduces neurologic impairment after open heart surgery. Ann Thorac Surg 2008', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Martens+carbon+dioxide+field+flooding+neurologic+impairment+open+heart+2008'},
        {'title': 'Hahn RT, et al. Guidelines for performing a comprehensive transesophageal echocardiographic examination (ASE/SCA). J Am Soc Echocardiogr 2013', 'url': 'https://pubmed.ncbi.nlm.nih.gov/?term=Hahn+guidelines+comprehensive+transesophageal+echocardiographic+examination+2013'},
    ]

    def mv_anat(pre):
        return {'id': f'{pre}-anatomy', 'phase': 'Anatomy', 'seq': 0, 'title': 'The mitral valve and what lies around it',
                'body': '<p>The <b>mitral annulus</b> is D-shaped and saddle-shaped. The <b>anterior leaflet</b> hangs from the fibrous <b>aortomitral curtain</b>, in continuity with the <b>left and non-coronary cusps</b> of the aortic valve; the <b>posterior leaflet</b> takes the rest of the circumference. '
                        'Chordae run from both leaflets to the <b>anterolateral</b> and <b>posteromedial papillary muscles</b>.</p>'
                        '<p>Round the posterior annulus, in the AV groove: the <b>circumflex artery</b> (close near the anterolateral commissure, closer in left dominance) and the <b>coronary sinus</b>. '
                        'At the posteromedial commissure, next to the right trigone: the <b>AV node and His bundle</b>. These are what deep annular sutures injure.</p>',
                'view': clook(MC, V([0.4, 0.8, 0.45]), 260), 'spin': True, 'show': [*CH, *VALVE, *DANGER, 'cusp-r', 'lvot'], 'hide': HEART_OFF,
                'opacity': {**FAINT, 'lv': 0.2, 'la': 0.25}, 'highlight': ['mitral-annulus', 'mv-ant-leaflet', 'mv-post-leaflet'], 'danger': DANGER,
                'labels': ['mv-ant-leaflet', 'mv-post-leaflet', 'papillary', 'circumflex', 'coronary-sinus', 'av-node', 'cusp-n', 'cusp-l'],
                'ask': ask('A deep suture at the posterior annulus near the anterolateral commissure is most likely to injure…', 'The circumflex artery',
                           'It runs in the AV groove close to the annulus there, especially in a left-dominant circulation.', 'The AV node', 'The right coronary artery'),
                'ct': ct(R(MC), 'axial')}

    def mv_decide(pre):
        return {'id': f'{pre}-decide', 'phase': 'Decision', 'seq': 1, 'title': 'Replace, and by which access',
                'body': '<p><b>Repair when you can</b> (degenerative MR above all). <b>Replace</b> when repair is unlikely to last: heavily calcified or fibrotic <b>rheumatic</b> valves, extensive leaflet destruction by endocarditis, failed repair. '
                        'Mechanical valves for the young who can take warfarin reliably (mitral INR target about 2.5–3.5); tissue valves for the older, or where anticoagulation is unsafe or pregnancy is planned.</p>'
                        '<p><b>Access by scenario:</b></p><ul>'
                        '<li><b>Median sternotomy, left atriotomy through Sondergaard\'s groove</b> (the default): most patients, any concomitant aortic, tricuspid or coronary surgery.</li>'
                        '<li><b>Transseptal (right atrium, fossa ovalis)</b>: a small left atrium, concomitant tricuspid surgery, a redo where the groove is scarred. The <b>superior septal</b> extension gives the widest view but divides the sinus node artery (atrial arrhythmias).</li>'
                        '<li><b>Right mini-thoracotomy</b> with femoral cannulation: isolated mitral surgery, and a redo after sternotomy (no re-entry). Avoid with significant aortic regurgitation, iliofemoral or aortic atheroma, dense right pleural adhesions.</li></ul>',
                'view': clook(MC, V([0.4, 0.8, 0.45]), 300), 'show': [*CH, *VALVE], 'hide': HEART_OFF, 'opacity': FAINT, 'labels': ['la', 'ra', 'lv'], 'ct': ct(R(MC), 'coronal')}

    def mv_sternotomy(pre, seq_):
        return {'id': f'{pre}-sternotomy', 'phase': 'Access', 'seq': seq_, 'title': 'Median sternotomy, pericardial cradle',
                'body': '<p>Median sternotomy; open the pericardium in the midline and <b>hitch it up as a cradle</b>, more on the right, which lifts the right atrium and the interatrial groove toward you.</p>',
                'view': clook(ST_MID, V([0, 1, 0.3]), 380), 'show': ['sternum', *CH], 'hide': HEART_OFF, 'opacity': {**FAINT, 'sternum': 0.95},
                'highlight': ['sternum'],
                'action': {'kind': 'saw', 'label': 'Divide the sternum', 'port': 'sternotomy', 'ids': ['sternum'], 'at': R(ST_MID), 'axis': [1, 0, 0], 'open': 90},
                'ct': ct(R(ST_MID), 'axial', 'bone')}

    def mv_cannulate(pre, seq_, septal=False):
        return {'id': f'{pre}-cannulate', 'phase': 'Bypass', 'seq': seq_, 'title': 'Cannulation: aorta and both cavae',
                'body': '<p>Heparin (about 300–400 U/kg; ACT above 480 s). Two purse-strings on the <b>distal ascending aorta</b> for the arterial cannula. <b>Bicaval venous drainage: two separate cannulas</b>: an angled <b>SVC cannula</b> directly into the SVC (or through the appendage), and an <b>IVC cannula</b> through a purse-string low on the right atrium, its tip passed into the IVC'
                        + (', with <b>snares round both cavae</b>: the right atrium will be opened.' if septal else '; snares are needed if the right atrium is to be opened.') + '</p>'
                        '<p>An antegrade cardioplegia and root vent line in the ascending aorta; a <b>retrograde cannula</b> into the coronary sinus if wanted; an <b>LV vent</b> through the right superior pulmonary vein.</p>',
                'view': clook((V(LM['can-svc']) + V(LM['can-ivc'])) / 2, V([0.75, 0.75, 0.1]), 360), 'show': [*CH, *[i for i in ('ivc', 'snares') if has(i)]], 'hide': HEART_OFF, 'opacity': {**FAINT, 'ivc': 0.6},
                'highlight': CANS, 'labels': ['can-svc', 'can-ivc', 'can-aortic', 'can-cp', 'svc', 'ivc'],
                'action': {'kind': 'reveal', 'label': 'Place the cannulas', 'port': 'sternotomy', 'ids': CANS},
                'ct': ct(R(V(LM['can-aortic'])), 'axial')}

    def mv_clamp(pre, seq_, port='sternotomy', text=''):
        return {'id': f'{pre}-clamp', 'phase': 'Bypass', 'seq': seq_, 'title': 'Cross-clamp and cardioplegia',
                'body': text or '<p>On full bypass, cool as planned. Cross-clamp the ascending aorta between the arterial cannula and the cardioplegia line; arrest with cold blood cardioplegia <b>antegrade</b>, then <b>retrograde</b> through the coronary sinus, repeated every 15–20 minutes.</p>',
                'view': clook(Lc('clamp-ao'), V([0.3, 1, 0.3]), 280), 'show': [*CH, *CANS, 'can-retro'], 'hide': HEART_OFF, 'opacity': FAINT,
                'highlight': ['aorta'], 'labels': ['can-cp', 'can-retro'],
                'action': {'kind': 'clamp', 'label': 'Apply the cross-clamp', 'port': port, 'at': R(Lc('clamp-ao')), 'axis': R(V(LM['ao-axis'])), 'radius': 14, 'jawLen': 50},
                'ct': ct(R(Lc('clamp-ao')), 'axial')}

    def mv_la(pre, seq_, port='sternotomy'):
        return {'id': f'{pre}-atriotomy', 'phase': 'Left atrium', 'seq': seq_, 'title': "Sondergaard's groove and the left atriotomy",
                'body': "<p>Develop <b>Sondergaard's plane</b>: the fat between the right atrium and the right pulmonary veins, back toward the septum, for 1–2 cm. "
                        'Open the left atrium there, <b>in front of the right pulmonary veins</b>, extending up behind the SVC and down behind the IVC as needed. A self-retaining atrial retractor lifts the septum toward you.</p>',
                'view': clook(Lc('la-incision'), V([1, 0.5, 0.2]), 250), 'show': [*CH, *CANS, 'la-incision'], 'hide': HEART_OFF, 'opacity': {**FAINT, 'la': 0.45},
                'highlight': ['la-incision'], 'danger': ['ra'],
                'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Open the left atrium', 'port': port, 'path': ext('la-incision', 16)},
                'ct': ct(R(Lc('la-incision')), 'axial')}

    def mv_excise(pre, seq_, port='sternotomy', view=None):
        return {'id': f'{pre}-excise', 'phase': 'Valve', 'seq': seq_, 'title': 'Excise the anterior leaflet; keep the posterior chordae',
                'body': '<p>Inspect the valve. For replacement, excise the <b>anterior leaflet</b> (detach it 2–3 mm from the annulus, divide its chordae; some reattach anterior chordae to the annulus), '
                        '<b>preserve the posterior leaflet and its chordae</b>: <b>chordal sparing</b> keeps left ventricular function and guards against rupture of the ventricle.</p>'
                        '<p>Debride calcium off the annulus carefully: deep debridement posteriorly risks <b>AV groove disruption</b>. Size the annulus with the valve sizers.</p>',
                'view': view or valve_view, 'show': [*VALVE, *DANGER, 'cusp-r'], 'hide': HEART_OFF, 'opacity': {**FAINT, 'la': 0.12, 'lv': 0.25},
                'highlight': ['mv-ant-leaflet'], 'danger': DANGER,
                'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Excise the anterior leaflet', 'port': port, 'remove': ['mv-ant-leaflet'],
                           'path': [R(MC + MU * MR_ * 0.95 + V(np.cross(MN, MU)) * MR_ * 0.6 + MN), R(MC + MU * (MR_ - 1) + MN), R(MC + MU * MR_ * 0.95 - V(np.cross(MN, MU)) * MR_ * 0.6 + MN)]},
                'ct': ct(R(MC), 'axial')}

    def mv_sutures(pre, seq_, port='sternotomy', view=None):
        return {'id': f'{pre}-sutures', 'phase': 'Valve', 'seq': seq_, 'title': 'Annular sutures: which technique?',
                'body': '<p><b>2-0 braided polyester horizontal mattress</b> sutures, 12–16 of them. Where the pledgets sit decides where the valve sits, and the choice follows the <b>quality of the annulus</b>.</p>'
                        '<p><b>Ventricular pledgets</b> (non-everting: needle from the LV side up through the annulus; the valve sits <b>supra-annular</b>)<br>'
                        '<b>For:</b> the strongest hold in <b>friable, calcified, rheumatic or infected</b> tissue, so fewer paravalvular leaks and dehiscences; the pledget spreads the load; supra-annular seating often takes <b>one size larger</b>.<br>'
                        '<b>Against:</b> deeper, blinder bites (circumflex and AV groove posteriorly, AV node at the posteromedial commissure, aortic cusps anteriorly); pledgets and preserved chordae under the ring can <b>trap a leaflet</b> or seed pannus and thrombus; anterior pledgets can narrow the <b>LVOT</b>; awkward through a small LA or a MICS port.</p>'
                        '<p><b>Annulus-only</b> (everting, with <b>atrial pledgets</b>, or unpledgeted; the valve sits <b>intra-annular</b>)<br>'
                        '<b>For:</b> shallow bites you can see; nothing below the ring to catch a leaflet; LVOT clear; quicker; suits a <b>healthy annulus</b> and MICS.<br>'
                        '<b>Against:</b> sutures can <b>cut through</b> poor tissue; the prosthesis is often a size smaller; <b>unpledgeted</b> sutures are the weakest.</p>'
                        '<p>A common compromise: <b>ventricular pledgets posteriorly and at the commissures</b> (where the tissue is worst and leaks happen), <b>atrial pledgets anteriorly</b> (to protect the aortic valve and LVOT).</p>'
                        '<p>Whatever the technique, bite <b>in the annulus, not beyond it</b>.</p>'
                        '<p class="evidence"><b>Evidence:</b> no randomised trial compares everting and non-everting sutures in MVR. In the AVERT trial cohort, <b>pledgeted</b> sutures were associated with fewer major paravalvular leaks than unpledgeted ones (Englberger et al., <i>Eur J Cardiothorac Surg</i> 2005). '
                        'The ACC/AHA 2020 and ESC/EACTS 2021 valve guidelines make no recommendation on suture technique, but support <b>preserving the subvalvular apparatus</b> in MVR (for LV function), and that is one more reason to keep pledgets clear of preserved chordae.</p>',
                'view': view or valve_view, 'show': [*[i for i in VALVE if i != 'mv-ant-leaflet'], *DANGER], 'hide': [*HEART_OFF, 'mv-ant-leaflet'], 'opacity': {**FAINT, 'la': 0.12, 'lv': 0.25},
                'highlight': ['mitral-annulus'], 'danger': DANGER,
                'action': {'kind': 'annulus', 'label': 'Place the annular sutures', 'port': port, 'at': R(MC), 'axis': R(MN), 'anterior': R(MU), 'radius': MR_ - 0.5, 'count': 14},
                'ask': ask('Rheumatic mitral stenosis with a heavily calcified, small posterior annulus. Which suture technique gives the most secure seat and the largest valve?',
                           'Pledgeted mattress sutures with ventricular pledgets (non-everting, supra-annular), at least posteriorly',
                           'Ventricular pledgets hold best in poor tissue and seat the ring above the annulus, which usually allows a larger size. Keep bites in the annulus to spare the circumflex and AV groove, and check that no pledget traps a leaflet.',
                           'Unpledgeted simple interrupted sutures', 'A continuous polypropylene suture', 'Everting sutures with atrial pledgets all round, one size smaller'),
                'ct': ct(R(MC), 'axial')}

    def mv_seat(pre, seq_, port='sternotomy', view=None):
        return {'id': f'{pre}-seat', 'phase': 'Valve', 'seq': seq_, 'title': 'Seat the prosthesis and tie',
                'body': '<p>Pass the sutures through the sewing ring in order, lower the valve down the sutures onto the annulus and tie. '
                        'For a bileaflet mechanical valve, the usual orientation is <b>anti-anatomical</b> (hinges perpendicular to the natural commissures).</p>'
                        '<p>Check that <b>both leaflets open and close fully</b>: nothing trapped (preserved chordae, a long suture tail, pledget).</p>',
                'view': view or valve_view, 'show': [*[i for i in VALVE if i != 'mv-ant-leaflet'], *DANGER], 'hide': [*HEART_OFF, 'mv-ant-leaflet'], 'opacity': {**FAINT, 'la': 0.12, 'lv': 0.25},
                'highlight': ['mv-prosthesis'], 'labels': ['mv-prosthesis'],
                'action': {'kind': 'seat', 'label': 'Seat the valve', 'port': port, 'ids': ['mv-prosthesis'], 'from': R(MN * 45)},
                'ask': ask('After seating a mechanical mitral valve, the most important check before closing the atrium is…', 'That both leaflets move freely',
                           'Preserved chordae, suture tails or pledgets can trap a leaflet; fix it now, not after the clamp is off.', 'That the LA appendage is closed', 'That the sutures are cut short'),
                'ct': ct(R(MC), 'axial')}

    def mv_close(pre, seq_, septal=False, mics=False):
        return {'id': f'{pre}-close', 'phase': 'Wean', 'seq': seq_, 'title': 'Close the atrium, de-air, clamp off',
                'body': ('<p>Close the septum and the right atrium (3-0/4-0 polypropylene); release the caval snares.</p>' if septal else '<p>Close the left atriotomy with 3-0/4-0 polypropylene, leaving the LV vent across until the last suture.</p>')
                        + '<p><b>De-air</b>: CO₂ in the field, head down, fill the heart as the last sutures go in, ventilate the lungs to push air out of the pulmonary veins, aortic root vent on suction; release the clamp; rewarm. Consider closing the <b>left atrial appendage</b> if the patient is in AF.</p>'
                        '<p><b>De-air early and come off</b> (most de-airing done in the arrested heart before the clamp is released; vent out and wean soon after)<br>'
                        '<b>For:</b> the flaccid, arrested heart can be balloted and needle-vented; a shorter bypass and clamp time.<br>'
                        '<b>Against:</b> air stays trapped in the <b>pulmonary veins, LA appendage and LV trabeculae</b> until the heart beats and the lungs are ventilated. It then comes out when the root vent is already out: into the <b>RCA</b> (the most anterior ostium, uppermost with the patient supine), giving inferior ST elevation, RV failure and VF, or into the <b>brain</b>. Mitral surgery opens the left heart widely, so it carries more air than CABG.</p>'
                        '<p><b>Keep de-airing during reperfusion</b> (the beating, ejecting heart on partial bypass, root vent still on, until TOE is clear)<br>'
                        '<b>For:</b> contraction and ventilation release the trapped air while the vent can still catch it; if air does reach the RCA, the heart is still supported, so raise the perfusion pressure and let it pass. It overlaps with the rest the heart needs anyway.<br>'
                        '<b>Against:</b> longer bypass; strong root-vent suction with a low root pressure can draw air <b>in</b> through the purse-string; handling the beating heart can cause arrhythmia.</p>'
                        '<p>The end-point is <b>TOE, not the clock</b>: no bubbles in the LA, LV or pulmonary veins at near-normal filling, then vent out.</p>'
                        + ('<p>In <b>MICS</b> the heart cannot be handled or balloted, so CO₂ flooding and TOE-guided venting matter even more.</p>' if mics else '')
                        + '<p class="evidence"><b>Evidence:</b> no trial compares early with late de-airing; practice rests on physiology and TOE studies. CO₂ field flooding cut microemboli on TOE in a randomised trial (Svenarud et al., <i>Circulation</i> 2004) and reduced neurocognitive impairment in another (Martens et al., <i>Ann Thorac Surg</i> 2008); a benefit for stroke has not been shown. '
                        'Intraoperative TOE in valve surgery, including to guide de-airing, is standard practice (ASE/SCA guidelines, Hahn et al., <i>J Am Soc Echocardiogr</i> 2013).</p>'
                        '<p><b>TOE</b> before leaving theatre: no paravalvular leak, leaflets moving, no LVOT obstruction. Pacing wires, drains.</p>'
                        + ('<p>Decannulate the femoral vessels and repair them; check the right lung re-expands.</p>' if mics else '')
                        + '<p>Serious complications: <b>AV groove disruption</b> (catastrophic), circumflex injury, heart block, paravalvular leak, stroke.</p>',
                'view': clook(MC, V([0.4, 0.8, 0.45]), 300), 'show': [*CH, 'mv-prosthesis'], 'hide': [*HEART_OFF, 'mv-ant-leaflet'], 'opacity': FAINT,
                'labels': ['mv-prosthesis'],
                'ask': ask('Ten minutes after the clamp is released and the root vent is removed, the inferior leads show ST elevation and the RV dilates. The likely cause and the move?',
                           'Air in the right coronary: stay on (or go back on) bypass, raise the perfusion pressure, and let the heart beat unloaded until it clears',
                           'The RCA ostium is uppermost with the patient supine and catches retained left-heart air. Supported, higher-pressure perfusion usually clears it in minutes; weaning onto a failing RV does not. Keeping the root vent on until TOE is clear prevents it.',
                           'Circumflex injury from an annular suture: re-arrest and inspect', 'Protamine reaction: stop the protamine', 'Prosthetic leaflet stuck: re-open the atrium'),
                'ct': ct(R(MC), 'axial')}

    def mv_reperfuse(pre, seq_, mics=False):
        return {'id': f'{pre}-reperfuse', 'phase': 'Wean', 'seq': seq_, 'title': 'Reperfuse on bypass, or separate early?',
                'body': '<p>With the clamp off the heart is <b>reperfused while bypass still carries the circulation</b>. How long to rest it before weaning is a judgement, not a fixed rule. '
                        'A common rule of thumb is about <b>a third of the cross-clamp time</b> (roughly 10 minutes for each 30 of ischaemia).</p>'
                        '<p><b>Waiting for myocardial recovery</b> (a longer supported reperfusion): it washes out cardioplegia and potassium, restores energy stores, lets the rhythm settle and rewarming finish, and needs fewer inotropes. '
                        'Worth it after a <b>long clamp</b>, with a <b>poor LV or RV</b>, a hypertrophied ventricle, doubtful protection, or <b>pulmonary hypertension</b> (common in rheumatic mitral stenosis: the RV fails first).</p>'
                        '<p><b>Separating early</b>: every extra minute of bypass adds haemodilution, platelet damage, inflammation and bleeding. After a <b>short clamp</b>, a good ventricle, sound protection and a stable rhythm, wean as soon as the conditions are met.</p>'
                        '<p>Before either: temperature 36–37 °C, sinus rhythm or pacing, potassium and haemoglobin corrected, lungs ventilated, <b>de-airing confirmed on TOE</b>, and the valve checked (no paravalvular leak, leaflets moving). '
                        'The cost of weaning too early is a low-output state and a return to bypass; that is still easy while <b>the cannulas are in and protamine has not been given</b>.</p>',
                'view': clook(MC, V([0.4, 0.8, 0.45]), 300), 'show': [*CH, 'mv-prosthesis', *([] if mics else CANS)], 'hide': [*HEART_OFF, 'mv-ant-leaflet'], 'opacity': FAINT,
                'labels': ['lv', 'rv', 'mv-prosthesis'],
                'ask': ask('After MVR for rheumatic stenosis with severe pulmonary hypertension (clamp time 95 min), the heart is sluggish on first weaning attempt. Best move?',
                           'Go back to full bypass, rest the heart longer, start RV support (inotrope, pulmonary vasodilator), then wean again',
                           'The RV is failing against a high pulmonary pressure; more supported reperfusion and RV-directed support usually rescue it. Pushing on off bypass drives the RV into failure.',
                           'Give protamine and push inotropes off bypass', 'Decannulate and accept the low output'),
                'ct': ct(R(MC), 'axial')}

    def mv_decannulate(pre, seq_, mics=False):
        ids = [i for i in (('can-cp',) if mics else ('can-retro', 'can-cp', 'can-ivc', 'can-svc', 'can-aortic')) if has(i)]
        return {'id': f'{pre}-decannulate', 'phase': 'Wean', 'seq': seq_, 'title': 'Separate, then decannulate in order',
                'body': '<p>Wean slowly, watching the pressures and the TOE. Then the order that keeps a way back:</p>'
                        '<ol><li><b>Venous cannula(s) out</b> first' + (' (the femoral venous cannula)' if mics else ' (IVC, then SVC); tie the purse-strings') + '.</li>'
                        '<li><b>Protamine</b> started slowly (watch for pulmonary hypertension and hypotension); <b>stop the pump suckers</b> once it runs.</li>'
                        '<li><b>Arterial cannula out last</b>, after part of the protamine and a stable pressure' + (' (repair the femoral artery)' if mics else '') + ': while it is in, blood can be given from the pump and bypass restarted quickly.</li></ol>'
                        '<p>A fast decannulation saves pump time only if the heart is ready; a return to bypass after full protamine means re-heparinising and re-cannulating a heart that is already struggling.</p>',
                'view': clook(V(LM['can-svc']), V([0.5, 0.9, 0.35]), 330), 'show': [*CH, 'mv-prosthesis', *ids], 'hide': [*HEART_OFF, 'mv-ant-leaflet'], 'opacity': FAINT,
                'labels': ids, 'highlight': ids,
                'action': {'kind': 'decannulate', 'label': 'Clamp off and decannulate', 'port': 'sternotomy', 'ids': ids},
                'ct': ct(R(MC), 'axial')}

    std = [mv_anat('ms'), mv_decide('ms'), mv_sternotomy('ms', 2), mv_cannulate('ms', 3), mv_clamp('ms', 4), mv_la('ms', 5), mv_excise('ms', 6), mv_sutures('ms', 7), mv_seat('ms', 8), mv_close('ms', 9), mv_reperfuse('ms', 10), mv_decannulate('ms', 11)]
    sept_view = clook(MC, MN * 0.4 + V([1, 0.4, 0.0]), 200)
    ts_ = [mv_anat('mt'), mv_decide('mt'), mv_sternotomy('mt', 2), mv_cannulate('mt', 3, septal=True), mv_clamp('mt', 4),
           {'id': 'mt-ra', 'phase': 'Right atrium', 'seq': 5, 'title': 'Right atriotomy',
            'body': '<p>Snare both cavae. Open the right atrium obliquely, from the base of the appendage toward the IVC, parallel to the AV groove and <b>away from the sinus node</b> at the SVC junction.</p>',
            'view': clook(Lc('ra-incision'), V([1, 0.6, 0.2]), 240), 'show': [*CH, *CANS, 'ra-incision'], 'hide': HEART_OFF, 'opacity': {**FAINT, 'ra': 0.5},
            'highlight': ['ra-incision'], 'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Open the right atrium', 'port': 'sternotomy', 'path': ext('ra-incision', 15)},
            'ct': ct(R(Lc('ra-incision')), 'axial')},
           {'id': 'mt-septum', 'phase': 'Septum', 'seq': 6, 'title': 'Through the fossa ovalis',
            'body': '<p>Incise the <b>fossa ovalis</b> vertically and extend it up (toward the dome) or down as needed; the mitral valve lies straight behind. Stay off the <b>coronary sinus</b> orifice and Koch\'s triangle below and in front (AV node), and off the aortic root in front of the septum.</p>'
                    '<p>The <b>superior septal (Guiraudon)</b> extension carries the incision across the roof of the left atrium: widest view, but it divides the sinus node artery.</p>',
            'view': clook(Lc('septum'), V([1, 0.3, 0.1]), 200), 'show': [*CH, 'septal-incision', 'av-node', 'coronary-sinus'], 'hide': HEART_OFF, 'opacity': {**FAINT, 'ra': 0.15},
            'highlight': ['septal-incision'], 'danger': ['av-node', 'coronary-sinus'],
            'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Open the septum', 'port': 'sternotomy', 'path': ext('septum', 14)},
            'ct': ct(R(Lc('septum')), 'axial')},
           mv_excise('mt', 7, view=sept_view), mv_sutures('mt', 8, view=sept_view), mv_seat('mt', 9, view=sept_view), mv_close('mt', 10, septal=True), mv_reperfuse('mt', 11), mv_decannulate('mt', 12)]
    mics_view = clook(MC, MN * 0.8 + V([1, 0.35, 0.05]), 200)
    mi = [mv_anat('mm'), mv_decide('mm'),
          {'id': 'mm-setup', 'phase': 'Access', 'seq': 2, 'title': 'Position, femoral cannulation, the incision',
           'body': '<p>Supine with the <b>right chest raised about 30°</b>, the right arm by the side. Double-lumen tube (right lung down), external defibrillator pads, TOE.</p>'
                   '<p><b>Femoral cannulation</b> (open or percutaneous): the venous cannula guided up into the SVC under TOE, the arterial cannula into the femoral artery (check iliofemoral atheroma on CT first). '
                   'A <b>4–6 cm incision in the right 4th space</b>, in the inframammary fold; a soft-tissue retractor; camera through the incision or a port above it; CO2 flooding the field.</p>',
           'view': clook(Lc('mics'), V([1, 0.6, 0.25]), 380), 'show': ['skin', 'incision-mics', 'port-chitwood'], 'opacity': {'skin': 1.0}, 'hide': HEART_OFF,
           'highlight': ['incision-mics'], 'labels': ['incision-mics', 'port-chitwood'], 'ct': ct(R(Lc('mics')), 'axial', 'lung')},
          mv_clamp('mm', 3, port='chitwood', text='<p>A <b>transthoracic (Chitwood) clamp</b> through a stab in the 3rd space, mid-axillary line, across the ascending aorta behind the pulmonary artery (avoiding the left atrial appendage and the right PA); '
                                                 'or an endoballoon occlusion. Antegrade cardioplegia through a root needle placed through the incision.</p>'),
          mv_la('mm', 4, port='mics'), mv_excise('mm', 5, port='mics', view=mics_view), mv_sutures('mm', 6, port='mics', view=mics_view), mv_seat('mm', 7, port='mics', view=mics_view), mv_close('mm', 8, mics=True), mv_reperfuse('mm', 9, mics=True), mv_decannulate('mm', 10, mics=True)]
    mi[3]['show'] = [*CH, 'port-chitwood', 'can-cp']
    for key, appr, steps_, sq in (('mvr-std', 'Median sternotomy (Kouchoukos)', std, seq(('Anatomy', 'other'), ('Decide', 'other'), ('Sternotomy', 'other'), ('Cannulate', 'artery'), ('Clamp', 'artery'), ('Atriotomy', 'vein'), ('Excise', 'fissure'), ('Sutures', 'fissure'), ('Seat', 'bronchus'), ('Close', 'other'), ('Reperfuse', 'other'), ('Decannulate', 'artery'))),
                                  ('mvr-septal', 'Transseptal', ts_, seq(('Anatomy', 'other'), ('Decide', 'other'), ('Sternotomy', 'other'), ('Cannulate', 'artery'), ('Clamp', 'artery'), ('Right atrium', 'vein'), ('Septum', 'vein'), ('Excise', 'fissure'), ('Sutures', 'fissure'), ('Seat', 'bronchus'), ('Close', 'other'), ('Reperfuse', 'other'), ('Decannulate', 'artery'))),
                                  ('mvr-mics', 'Right mini-thoracotomy', mi, seq(('Anatomy', 'other'), ('Decide', 'other'), ('Access', 'other'), ('Clamp', 'artery'), ('Atriotomy', 'vein'), ('Excise', 'fissure'), ('Sutures', 'fissure'), ('Seat', 'bronchus'), ('Close', 'other'), ('Reperfuse', 'other'), ('Decannulate', 'artery')))):
        for s in steps_:
            named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
            # the heart alone: the lung hila, nodes, nerves and pleura out of the way; the sternum once it is open
            off_groups = {'arteries', 'veins', 'airway', 'lul-intra', 'lll-intra', 'rul-intra', 'nodes', 'nerves', 'pleura', 'segments', 'trauma', 'muscles', 'landmarks'}
            s['hide'] = [*s.get('hide', []), *[q['id'] for q in atlas['structures'] if q['group'] in off_groups and q['id'] not in named],
                         *[i for i in ('esophagus', 'thymus', 'thyroid') if has(i) and i not in named], *(['sternum'] if s.get('seq', 0) >= 3 else [])]
            if s['phase'] in ('Valve', 'Wean', 'Septum'): s['opacity'] = {**{c_: 0.3 for c_ in CANS}, 'can-retro': 0.3, **s.get('opacity', {})}
            if s['phase'] in ('Valve', 'Septum'): s['hide'] = [*s['hide'], 'svc']
            s['opacity'] = {**{f'vert-t{i}': 0.25 for i in range(2, 11)}, **s.get('opacity', {})}
            for kk in ('highlight', 'danger', 'labels', 'show'):
                if kk in s: s[kk] = [i for i in s[kk] if has(i) or i == 'skin']
        procs[key] = {'id': key, 'op': 'mvr', 'opName': 'Mitral valve replacement', 'side': 'both', 'name': 'Mitral valve replacement', 'approach': appr,
                      'summary': 'Access, bypass, left atrium, chordal-sparing excision, pledgeted annular sutures, prosthesis, de-airing.', 'ports': [], 'steps': steps_, 'sources': MVSRC,
                      'group': 'Cardiac', 'sequence': sq}
# ==================================================================================================== cardiac: aortic valve replacement
AVR_OK = MVR_OK and has('aortic-annulus') and 'av-centre' in LM
if AVR_OK:
    AC, AN, AE = Lc('av-centre'), V(LM['av-axis']), V(LM['av-e1']); AR_ = LM['av-dims'][0]
    AOT = [Lc(f'aot-{i}') for i in range(8) if f'aot-{i}' in LM]
    CUSPS = [i for i in ('av-cusp-r', 'av-cusp-l', 'av-cusp-n', 'av-calcium') if has(i)]
    ROOT = [i for i in ('aortic-annulus', 'stj', *CUSPS) if has(i)]
    AV_DANGER = [i for i in ('ostium-r', 'ostium-l', 'his-bundle', 'mv-ant-leaflet') if has(i)]
    AV_CANS = [i for i in ('can-aortic', 'can-2stage', 'can-cp', 'can-lvvent', 'can-retro') if has(i)]
    AV_FAINT = {**FAINT, 'aorta': 0.18, 'lvot': 0.25, 'cusp-r': 0.15, 'cusp-l': 0.15, 'cusp-n': 0.15}
    root_view = clook(AC, AN * 0.85 + V([0.15, 0.55, 0.0]), 150)
    open_view = clook(AC, AN * 0.45 + V([0.15, 1.0, 0.0]), 100)
    ev = lambda t: f'<p class="evidence"><b>Evidence:</b> {t}</p>'
    pm = lambda term: 'https://pubmed.ncbi.nlm.nih.gov/?term=' + term.replace(' ', '+')
    AVSRC = [
        {'title': 'Kouchoukos NT, Blackstone EH, Hanley FL, Kirklin JK. Kirklin/Barratt-Boyes Cardiac Surgery, 4th ed. Elsevier 2013: aortic valve replacement', 'url': pm('Kirklin Barratt-Boyes cardiac surgery aortic valve')},
        {'title': 'Praz F, Borger MA, et al. 2025 ESC/EACTS Guidelines for the management of valvular heart disease. Eur Heart J 2025', 'url': 'https://www.escardio.org/guidelines/clinical-practice-guidelines/all-esc-practice-guidelines/valvular-heart-disease/'},
        {'title': 'Otto CM, Nishimura RA, et al. 2020 ACC/AHA guideline for the management of patients with valvular heart disease. Circulation 2021;143:e72-e227', 'url': 'https://pubmed.ncbi.nlm.nih.gov/33332150/'},
        {'title': 'Isselbacher EM, et al. 2022 ACC/AHA guideline for the diagnosis and management of aortic disease. Circulation 2022', 'url': 'https://pubmed.ncbi.nlm.nih.gov/36322642/'},
        {'title': 'Mack MJ, et al. Transcatheter aortic-valve replacement with a balloon-expandable valve in low-risk patients (PARTNER 3). N Engl J Med 2019', 'url': pm('Mack PARTNER 3 low-risk transcatheter NEJM 2019')},
        {'title': 'Popma JJ, et al. Transcatheter aortic-valve replacement with a self-expanding valve in low-risk patients (Evolut Low Risk). N Engl J Med 2019', 'url': pm('Popma Evolut low risk self-expanding NEJM 2019')},
        {'title': 'Généreux P, et al. Valve Academic Research Consortium 3 (VARC-3): updated endpoint definitions. J Am Coll Cardiol 2021', 'url': pm('VARC-3 updated endpoint definitions Genereux 2021')},
        {'title': 'Impact of prosthesis-patient mismatch after surgical aortic valve replacement: systematic review and meta-analysis of reconstructed time-to-event data of 122 989 patients. J Am Heart Assoc 2024', 'url': 'https://www.ahajournals.org/doi/10.1161/JAHA.123.033176'},
        {'title': 'Englberger L, et al. Importance of implant technique on risk of major paravalvular leak after St. Jude mechanical valve replacement (AVERT). Eur J Cardiothorac Surg 2005;28:838-43', 'url': 'https://academic.oup.com/ejcts/article/28/6/838/377180'},
        {'title': 'Boltje JWT, et al. The use of pledget-reinforced sutures during surgical aortic valve replacement: systematic review and meta-analysis. 2024', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11387225/'},
        {'title': 'Tabata M, et al. Simple interrupted suturing increases valve performance after aortic valve replacement with a small supra-annular bioprosthesis. J Thorac Cardiovasc Surg 2014;147:321-5', 'url': 'https://www.sciencedirect.com/science/article/pii/S0022522312013979'},
        {'title': 'Fischlein T, et al. Sutureless versus conventional bioprostheses for aortic valve replacement in severe symptomatic aortic stenosis (PERSIST-AVR). J Thorac Cardiovasc Surg 2021', 'url': pm('Fischlein PERSIST-AVR sutureless conventional bioprostheses')},
        {'title': 'Yang B, et al. Early outcomes of the Y-incision technique to enlarge the aortic annulus 3 to 4 valve sizes. J Thorac Cardiovasc Surg', 'url': 'https://www.jtcvs.org/article/S0022-5223(22)00722-X/fulltext'},
        {'title': 'Miceli A, Ferrarini M, Glauber M. Right anterior minithoracotomy for aortic valve replacement. Ann Cardiothorac Surg 2015;4:91-3', 'url': 'https://www.annalscts.com/article/view/5486/6313'},
        {'title': 'Svenarud P, et al. Effect of CO2 insufflation on the number and behavior of air microemboli in open-heart surgery: a randomized clinical trial. Circulation 2004', 'url': pm('Svenarud carbon dioxide insufflation microemboli open-heart randomized Circulation 2004')},
    ]

    def av_anat(pre):
        return {'id': f'{pre}-anatomy', 'phase': 'Anatomy', 'seq': 0, 'title': 'The aortic root and what lies around it',
                'body': '<p>The cusps do not hinge on a flat ring but on a <b>three-pronged crown</b>: low at each <b>nadir</b> (the virtual basal ring, which is what echo and CT call the annulus) and high at the three <b>commissures</b>, just below the <b>sinotubular junction</b>.</p>'
                        '<p>What the sutures and the debridement can injure:</p><ul>'
                        '<li><b>Coronary ostia</b>: the left main in the left sinus, the right coronary in the right sinus, usually 1–1.5 cm above the annulus. Keep prosthesis posts and pledgets clear of them.</li>'
                        '<li><b>Membranous septum and His bundle</b>: below the commissure between the <b>right and non-coronary cusps</b>. Deep bites or aggressive decalcification here cause complete heart block.</li>'
                        '<li><b>Anterior mitral leaflet</b>: below the <b>left and non-coronary cusps</b>, through the fibrous aortomitral curtain. Deep bites there distort the mitral valve.</li></ul>'
                        + ev('the crown-shaped annulus and its relations are from anatomical and CT studies of the root; conduction injury is the main cause of the permanent pacemakers needed after surgical AVR.'),
                'view': root_view, 'spin': True, 'show': [*CH, *ROOT, *AV_DANGER, 'lvot'], 'hide': HEART_OFF, 'opacity': AV_FAINT,
                'highlight': ['aortic-annulus'], 'danger': AV_DANGER,
                'labels': ['aortic-annulus', 'stj', 'av-cusp-r', 'av-cusp-l', 'av-cusp-n', 'ostium-r', 'ostium-l', 'his-bundle', 'mv-ant-leaflet'],
                'ask': ask('A deep annular suture below the commissure between the right and non-coronary cusps is most likely to cause…', 'Complete heart block',
                           'The His bundle runs in the membranous septum just below that commissure.', 'Mitral regurgitation', 'Occlusion of the left main'),
                'ct': ct(R(AC), 'coronal')}

    def av_decide(pre):
        return {'id': f'{pre}-decide', 'phase': 'Decision', 'seq': 1, 'title': 'SAVR or TAVI; mechanical or tissue; which access',
                'body': '<p><b>Surgical or transcatheter.</b> A Heart Team decision on age, life expectancy, anatomy (bicuspid, annulus size, coronary heights, access), other lesions, and the patient\'s wishes.</p><ul>'
                        '<li><b>ESC/EACTS 2025</b>: TAVI (class I) from <b>70 years</b> with tricuspid aortic stenosis and suitable anatomy, whatever the surgical risk; SAVR (class I) <b>under 70</b> at low risk.</li>'
                        '<li><b>ACC/AHA 2020</b>: SAVR under 65 or with life expectancy over 20 years; TAVI over 80 or with life expectancy under 10 years; either between 65 and 80.</li>'
                        '<li>SAVR remains the choice for <b>bicuspid</b> or <b>rheumatic</b> valves in the young, for <b>endocarditis</b>, a <b>small annulus</b> needing enlargement, and concomitant aortic, mitral or coronary surgery.</li></ul>'
                        '<p><b>Mechanical or tissue.</b> ESC/EACTS 2025: mechanical preferred <b>under 60</b> (aortic), tissue <b>over 65</b>, individual choice between. ACC/AHA 2020: mechanical under 50, tissue over 65, choice between 50 and 65. '
                        'In the young, a tissue valve degenerates early (sooner still after rheumatic disease) and means reoperation or valve-in-valve. A mechanical valve means lifelong warfarin (aortic INR target 2.5, higher with risk factors), so it needs reliable INR monitoring. Pregnancy weighs against warfarin.</p>'
                        '<p><b>The ascending aorta.</b> ACC/AHA 2022: replacing it at AVR is reasonable at <b>5.0 cm or more</b> (4.5 cm or more in experienced centres), and at <b>4.5 cm or more</b> with a <b>bicuspid</b> valve.</p>'
                        '<p><b>Access.</b> Full sternotomy: the default, and for anything concomitant. <b>Upper hemisternotomy</b>: isolated AVR, most anatomies. <b>Right anterior mini-thoracotomy</b>: isolated AVR when CT shows a rightward aorta close to the sternum.</p>'
                        + ev('PARTNER 3 and Evolut Low Risk (NEJM 2019) showed TAVI non-inferior (PARTNER 3: superior at one year on a composite endpoint) to SAVR in low-risk patients with tricuspid stenosis, mostly in their 70s. They excluded bicuspid and rheumatic valves, and their durability beyond 10 years is not yet known.'),
                'view': clook(AC, V([0.3, 0.9, 0.4]), 300), 'show': [*CH, *ROOT], 'hide': HEART_OFF, 'opacity': AV_FAINT, 'labels': ['aorta', 'lv'],
                'ask': ask('A 34-year-old with rheumatic aortic stenosis and regurgitation, no plans for pregnancy, reliable INR clinic access. Which valve do the guidelines favour?',
                           'A mechanical valve', 'Under 60 (ESC/EACTS) or 50 (ACC/AHA) with reliable anticoagulation, a mechanical valve avoids the early degeneration of tissue valves, which is faster after rheumatic disease. TAVI trials did not include rheumatic valves.',
                           'A tissue valve, then valve-in-valve later', 'TAVI', 'A stentless tissue root'),
                'ct': ct(R(AC), 'coronal')}

    def av_cannulate(pre, seq_):
        return {'id': f'{pre}-cannulate', 'phase': 'Bypass', 'seq': seq_, 'title': 'Cannulation: aorta, two-stage venous cannula (RA to IVC), LV vent',
                'body': '<p>Heparin (ACT above 480 s). Arterial cannula in the <b>distal ascending aorta</b>, high enough to leave room for the clamp and an aortotomy below it. For isolated AVR the right atrium is not opened, so venous drainage is by a single <b>two-stage (cavoatrial) cannula</b>: in through a purse-string on the <b>right atrial appendage</b>, its tip passed down into the <b>IVC</b>. The tip drains the IVC, the side holes (the second stage) sit in the right atrium and drain the SVC return. It cannot be snared, so it is not used when the right atrium must be opened (mitral, tricuspid, septal defects: use two caval cannulas).</p>'
                        '<p>Root cardioplegia and vent line; a <b>retrograde</b> cannula in the coronary sinus; an <b>LV vent through the right superior pulmonary vein</b>, placed <b>before the heart slows</b> if there is aortic regurgitation, so the ventricle never distends.</p>'
                        + ev('this is standard practice (Kirklin/Barratt-Boyes). A regurgitant valve lets cardioplegia and bypass return fill the arrested LV; distension damages the myocardium, which is why the vent goes in early.'),
                'view': clook((V(LM['can-aortic']) + V(LM['can-ivc'])) / 2, V([0.6, 0.85, 0.2]), 380), 'show': [*CH, *[i for i in ('ivc',) if has(i)]], 'hide': HEART_OFF, 'opacity': {**FAINT, 'ivc': 0.6},
                'highlight': AV_CANS, 'labels': [*AV_CANS, 'aorta', 'ivc'],
                'action': {'kind': 'reveal', 'label': 'Place the cannulas', 'port': 'sternotomy', 'ids': AV_CANS},
                'ct': ct(R(V(LM['can-aortic'])), 'axial')}

    def av_clamp(pre, seq_, port='sternotomy'):
        return {'id': f'{pre}-clamp', 'phase': 'Bypass', 'seq': seq_, 'title': 'Cross-clamp; cardioplegia depends on the valve',
                'body': '<p>On full bypass, clamp the ascending aorta below the arterial cannula.</p><ul>'
                        '<li><b>Aortic stenosis, competent valve</b>: antegrade into the root arrests the heart.</li>'
                        '<li><b>Aortic regurgitation</b>: antegrade into the root <b>runs into the LV</b> and distends it instead of perfusing the coronaries. Give it <b>retrograde</b>, and/or open the aorta and give it <b>directly into each ostium</b> with hand-held cannulas.</li></ul>'
                        '<p>Repeat every 15–20 minutes (retrograde, or down the ostia), or as the solution in use requires.</p>'
                        + ev('the choice follows the physiology rather than trials; retrograde perfusion reaches the right ventricle less well, which is one reason many combine retrograde with direct ostial doses.'),
                'view': clook(Lc('clamp-ao'), V([0.3, 1, 0.3]), 280), 'show': [*CH, *AV_CANS], 'hide': HEART_OFF, 'opacity': FAINT,
                'highlight': ['aorta'], 'labels': ['can-cp', 'can-retro', 'can-lvvent'],
                'action': {'kind': 'clamp', 'label': 'Apply the cross-clamp', 'port': port, 'at': R(Lc('clamp-ao')), 'axis': R(V(LM['ao-axis'])), 'radius': 14, 'jawLen': 50},
                'ask': ask('Severe aortic regurgitation. After the cross-clamp, antegrade root cardioplegia is started and the LV swells while the heart keeps beating. Next?',
                           'Stop the root infusion, vent the LV, open the aorta and give cardioplegia directly into the ostia (and/or retrograde)',
                           'Root cardioplegia is going through the incompetent valve into the LV, not down the coronaries. Distension injures the myocardium.',
                           'Increase the root infusion pressure', 'Cool further and wait for arrest'),
                'ct': ct(R(Lc('clamp-ao')), 'axial')}

    def av_aortotomy(pre, seq_, port='sternotomy'):
        return {'id': f'{pre}-aortotomy', 'phase': 'Aorta', 'seq': seq_, 'title': 'Oblique aortotomy into the non-coronary sinus',
                'body': '<p>Find the <b>right coronary</b> origin first. Open the aorta <b>1–1.5 cm above it</b>, across the front, and carry the incision <b>obliquely down into the non-coronary sinus</b> toward its nadir ("hockey stick"). A transverse aortotomy above the STJ is the alternative, and is easier to close for a later root operation.</p>'
                        '<p>Stay above the right coronary and away from the <b>left main</b> on the other side. Stay sutures open the aorta; give ostial cardioplegia now if needed.</p>'
                        + ev('the placement is technique by consensus (Kirklin/Barratt-Boyes); the risks it avoids are right coronary injury and distortion of the sinuses and STJ when closing.'),
                'view': clook(AOT[3] if AOT else AC, AN * 0.3 + V([0.2, 1.0, 0.1]), 170), 'show': [*CH, *AV_CANS, 'aortotomy', 'ostium-r', 'ostium-l'], 'hide': HEART_OFF,
                'opacity': {**FAINT, 'aorta': 0.55}, 'highlight': ['aortotomy'], 'danger': ['ostium-r', 'ostium-l'], 'labels': ['aortotomy', 'ostium-r'],
                'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Open the aorta', 'port': port, 'path': [R(p) for p in AOT[::2]] or [R(AC)]},
                'ct': ct(R(AC), 'axial')}

    def av_excise(pre, seq_, port='sternotomy', view=None):
        return {'id': f'{pre}-excise', 'phase': 'Valve', 'seq': seq_, 'title': 'Excise the cusps, debride the annulus, catch every fragment',
                'body': '<p>Protect the LV: a moist sponge or gauze through the valve into the outflow tract, and the vent on. Excise the cusps 1–2 mm from their hinge, one cusp at a time.</p>'
                        '<p><b>Debride the calcium</b> back to a pliable annulus so the sewing ring can sit flat, without leaks. The danger zones:</p><ul>'
                        '<li>under the <b>right and non-coronary commissure</b>: the membranous septum and His bundle (heart block; a VSD if you go through);</li>'
                        '<li>under the <b>left and non-coronary cusps</b>: the aortomitral curtain (a hole into the left atrium; mitral damage);</li>'
                        '<li>the <b>ostia</b>: calcium near them is lifted out, not avulsed.</li></ul>'
                        '<p>Remove the sponge, <b>irrigate the LV and root</b> with saline and suction, and look for loose fragments before sizing.</p>'
                        + ev('stroke after SAVR is mostly embolic, and calcific debris is one source; catching and irrigating it is standard practice. Aggressive debridement near the membranous septum raises the risk of heart block and of annular disruption.'),
                'view': view or open_view, 'show': [*ROOT, *AV_DANGER], 'hide': HEART_OFF, 'opacity': AV_FAINT,
                'highlight': CUSPS, 'danger': AV_DANGER, 'labels': ['av-calcium', 'his-bundle', 'mv-ant-leaflet', 'ostium-l', 'ostium-r'],
                'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Excise and decalcify', 'port': port, 'remove': CUSPS,
                           'path': [R(AC + AE * AR_ * np.cos(t) + V(np.cross(AN, AE)) * AR_ * np.sin(t) + AN * 3) for t in np.linspace(0, 2 * np.pi, 7)]},
                'ask': ask('While debriding, you take calcium from beneath the commissure between the right and non-coronary cusps and see the septum bulge. What is at risk?',
                           'The His bundle (heart block) and the membranous septum (a VSD)', 'That area is the membranous septum carrying the His bundle. Take less calcium there; a little residual calcium is safer than a VSD or heart block.',
                           'The left main coronary', 'The anterior mitral leaflet'),
                'ct': ct(R(AC), 'coronal')}

    def av_size(pre, seq_, view=None, ramt=False):
        return {'id': f'{pre}-size', 'phase': 'Valve', 'seq': seq_, 'title': 'Size, and avoid patient–prosthesis mismatch',
                'body': '<p>Measure the annulus with the <b>sizers made for the chosen valve</b> (sizes differ between makers). The question is not only "what fits" but whether the valve\'s <b>effective orifice area indexed to body surface area (EOAi)</b> will be big enough for this patient.</p>'
                        '<p><b>Patient–prosthesis mismatch (VARC-3)</b>, BMI under 30: moderate at EOAi 0.85–0.66, severe at 0.65 cm²/m² or less. BMI 30 or more: moderate 0.70–0.56, severe 0.55 or less. Look up the valve\'s expected EOA before choosing.</p>'
                        '<p>If the annulus is too small:</p><ul><li>a valve with a better orifice for its size (supra-annular, thin-sewing-ring, or stentless);</li>'
                        '<li><b>annular enlargement</b>: <b>Nicks</b> (through the non-coronary sinus toward the aortomitral curtain), <b>Manouguian</b> (through the left–non-coronary commissure into the curtain and the anterior mitral leaflet), '
                        '<b>Y-incision</b> (Yang: through the left–non-coronary commissure with a Y into both trigones, allowing about 3–4 sizes larger). The <b>Konno</b> operation (into the septum) is mainly for children.</li></ul>'
                        + ('<p>Through a mini-thoracotomy, a <b>sutureless or rapid-deployment</b> valve shortens the clamp time and needs few or no annular sutures.</p>' if ramt else '')
                        + ev('VARC-3 (Généreux et al., JACC 2021) sets the definitions. In a meta-analysis of about 123,000 patients (J Am Heart Assoc 2024), mismatch after SAVR was associated with higher late mortality and earlier failure of tissue valves. The Y-incision series (Yang et al., JTCVS) reported enlargement by 3–4 sizes with low early mortality; these are single-centre, non-randomised data.'
                             + (' PERSIST-AVR (Fischlein et al., JTCVS 2021), a randomised trial, found a sutureless valve non-inferior to a stented one for major adverse events at one year, with <b>more permanent pacemakers</b>.' if ramt else '')),
                'view': view or open_view, 'show': ['aortic-annulus', *AV_DANGER], 'hide': [*HEART_OFF, *CUSPS], 'opacity': AV_FAINT,
                'highlight': ['aortic-annulus'], 'labels': ['aortic-annulus'],
                'ask': ask('A 1.9 m² patient (BMI 26). The largest valve that fits has an expected EOA of 1.2 cm². What is the predicted mismatch, and what should you consider?',
                           'EOAi 0.63: severe mismatch; consider annular enlargement or a valve with a larger orifice', '1.2 / 1.9 = 0.63 cm²/m², which is 0.65 or less: severe by VARC-3 (BMI under 30).',
                           'EOAi 0.63: acceptable', 'EOAi 1.1: no mismatch'),
                'ct': ct(R(AC), 'coronal')}

    def av_sutures(pre, seq_, port='sternotomy', view=None):
        return {'id': f'{pre}-sutures', 'phase': 'Valve', 'seq': seq_, 'title': 'Annular sutures: which technique?',
                'body': '<p>Usually 12–15 sutures of <b>2-0 braided polyester</b>, following the crown: up to the commissures, down to the nadirs. Where they sit decides where the valve sits.</p>'
                        '<p><b>Non-everting mattress, pledgets below</b> (on the LV side; the valve sits <b>supra-annular</b>)<br><b>For:</b> the usual technique for supra-annular valves; pledgets spread the load in calcified or fragile tissue; the largest valve for the annulus.<br>'
                        '<b>Against:</b> pledgets sit in the outflow tract; the bites below the right–non-coronary commissure are near the His bundle.</p>'
                        '<p><b>Everting mattress, pledgets above</b> (on the aortic side; the valve sits <b>intra-annular</b>)<br><b>For:</b> nothing below the annulus; everts the tissue for a good seal.<br><b>Against:</b> the ring sits inside the annulus, so the valve is often a size smaller: a mismatch risk in a small root.</p>'
                        '<p><b>Simple interrupted</b> (no pledgets)<br><b>For:</b> less material on the annulus, so the valve seats deeper and opens more. <b>Against:</b> no buttress in poor tissue.</p>'
                        '<p><b>Continuous polypropylene</b><br><b>For:</b> fast. <b>Against:</b> one weak point can loosen the whole line; not for calcified or infected annuli.</p>'
                        '<p>Keep pledgets and sutures clear of the <b>ostia</b>; take shallow bites under the <b>right–non-coronary commissure</b>.</p>'
                        + ev('no randomised trial has settled this. In the AVERT cohort (807 patients), major paravalvular leak occurred in 1.7% with pledgets and 5.8% without (Englberger et al., EJCTS 2005). '
                             'A 2024 meta-analysis of 9 observational SAVR studies (4,390 patients; Boltje et al.) found no clear difference in leak, gradients or mortality, and concluded that the evidence neither supports nor opposes pledgets. '
                             'In 152 patients with small (19–21 mm) supra-annular bioprostheses, simple interrupted sutures gave a larger orifice and less mismatch than mattress sutures (Tabata et al., JTCVS 2014; retrospective).'),
                'view': view or open_view, 'show': ['aortic-annulus', *AV_DANGER], 'hide': [*HEART_OFF, *CUSPS], 'opacity': AV_FAINT,
                'highlight': ['aortic-annulus'], 'danger': AV_DANGER,
                'action': {'kind': 'annulus', 'label': 'Place the annular sutures', 'port': port, 'at': R(AC + AN * 2), 'axis': R(AN), 'anterior': R(AE), 'radius': AR_ - 0.3, 'count': 12},
                'ask': ask('An elderly woman with a heavily calcified 19 mm annulus, receiving a supra-annular bioprosthesis. Which suture choice best balances sealing and orifice size?',
                           'Pledgeted non-everting mattress sutures (supra-annular), or simple interrupted where the tissue holds', 'Supra-annular seating gives the largest valve. Pledgets protect calcified tissue from cutting through; simple interrupted bites (Tabata) gain orifice where the tissue is sound. Everting sutures would push the valve intra-annular and a size down.',
                           'Everting mattress sutures all round', 'A continuous polypropylene suture'),
                'ct': ct(R(AC), 'coronal')}

    def av_seat(pre, seq_, port='sternotomy', view=None):
        return {'id': f'{pre}-seat', 'phase': 'Valve', 'seq': seq_, 'title': 'Seat the valve; check the ostia and the leaflets',
                'body': '<p>Pass the sutures through the sewing ring and lower the valve. For a stented tissue valve, <b>line the three posts up with the native commissures</b>, so that no post faces a coronary ostium. Tie, with even tension, the ring seated flat.</p>'
                        '<p>Before closing: look into <b>both ostia</b> (nothing overhanging); check for <b>gaps under the ring</b>; check that the <b>leaflets move freely</b> (for a mechanical valve, rotate it to free them of the septum).</p>'
                        + ev('coronary obstruction after SAVR is rare but often fatal; it is prevented by post orientation and direct inspection. Post-bypass TOE confirms leaflet motion, gradient and paravalvular leak (ASE/SCA guidance).'),
                'view': view or open_view, 'show': ['aortic-annulus', 'av-prosthesis', 'ostium-r', 'ostium-l', 'his-bundle'], 'hide': [*HEART_OFF, *CUSPS], 'opacity': AV_FAINT,
                'highlight': ['av-prosthesis'], 'danger': ['ostium-r', 'ostium-l'], 'labels': ['av-prosthesis', 'ostium-r', 'ostium-l'],
                'action': {'kind': 'seat', 'label': 'Seat the valve', 'port': port, 'ids': ['av-prosthesis'], 'from': R(AN * 45)},
                'ask': ask('After weaning, new ST elevation in the anterolateral leads and poor anterior wall motion on TOE. The valve looks well seated. First suspicion?',
                           'The left main ostium is partly obstructed by a post, a pledget or the sewing ring (or has embolised air/debris)', 'Coronary compromise right after SAVR is the prosthesis or embolism until proved otherwise: go back on bypass, inspect, and re-seat or graft.',
                           'Heart block', 'Patient–prosthesis mismatch'),
                'ct': ct(R(AC), 'coronal')}

    def av_close(pre, seq_, mini=False):
        return {'id': f'{pre}-close', 'phase': 'Wean', 'seq': seq_, 'title': 'Close the aorta and de-air',
                'body': '<p>Close the aortotomy in <b>two layers of 4-0 polypropylene</b> (a horizontal mattress, then over-and-over), with felt strips if the wall is thin. Before the last sutures: fill the heart, ventilate the lungs, head down; the <b>root vent on</b>; release the clamp.</p>'
                        '<p><b>Early vs late de-airing</b> is the same question as in mitral surgery, with less air because the left atrium was not opened: '
                        'venting only until the clamp is off is quicker, but air trapped in the pulmonary veins and LV comes out later and goes up the right coronary (uppermost) or to the brain; '
                        'keeping the root vent on while the heart ejects on partial bypass catches it. Stop when <b>TOE</b> shows no air, not by the clock.</p>'
                        + ('<p>Through a mini-thoracotomy the heart cannot be handled, so <b>CO₂ in the field</b> and TOE-guided venting carry more weight.</p>' if mini else '')
                        + ev('no trial compares de-airing strategies; CO₂ field flooding reduced microemboli on TOE in a randomised trial (Svenarud et al., Circulation 2004); a stroke benefit has not been shown.'),
                'view': clook(AOT[3] if AOT else AC, AN * 0.3 + V([0.2, 1.0, 0.1]), 200), 'show': [*CH, 'aortotomy', 'av-prosthesis'], 'hide': [*HEART_OFF, *CUSPS], 'opacity': {**FAINT, 'aorta': 0.5},
                'highlight': ['aortotomy'], 'labels': ['aortotomy'],
                'action': {'kind': 'suture', 'label': 'Close the aortotomy', 'port': 'sternotomy', 'path': [R(p) for p in AOT], 'normal': R(V([0.2, 1, 0.1])), 'axis': R(AOT[-1] - AOT[0] if AOT else V([1, 0, 0]))},
                'ct': ct(R(AC), 'axial')}

    def av_wean(pre, seq_, mini=False):
        ids = [i for i in ('can-lvvent', 'can-retro', 'can-cp', 'can-2stage', 'can-aortic') if has(i)]
        return {'id': f'{pre}-wean', 'phase': 'Wean', 'seq': seq_, 'title': 'Reperfuse, wean, TOE, decannulate',
                'body': '<p>Reperfuse on bypass until the heart is ready (about a third of the clamp time is a common rule of thumb; longer after a long clamp or with a thick, hypertrophied ventricle). Pace (<b>epicardial wires</b>) if there is heart block: it often recovers over days.</p>'
                        '<p><b>TOE after bypass</b>: prosthesis gradient and leaflet motion, <b>paravalvular leak</b>, new <b>regional wall-motion abnormality</b> (ostia), mitral function, and, in a small hypertrophied LV, <b>outflow obstruction with systolic anterior motion</b> of the mitral leaflet (treat with volume and beta-blockade; stop inotropes).</p>'
                        '<p>Then venous cannula out, protamine, arterial cannula out last.</p>'
                        + ev('intraoperative TOE is standard in valve surgery (ASE/SCA guidelines). A leak more than mild on TOE is usually repaired before leaving theatre, because significant paravalvular leak is associated with haemolysis, heart failure and worse survival.'),
                'view': clook(V(LM['can-aortic']), V([0.4, 0.9, 0.35]), 330), 'show': [*CH, 'av-prosthesis', *ids], 'hide': [*HEART_OFF, *CUSPS], 'opacity': FAINT,
                'labels': ids, 'highlight': ids,
                'action': {'kind': 'decannulate', 'label': 'Wean and decannulate', 'port': 'sternotomy', 'ids': ids},
                'ask': ask('After AVR for severe AS (small, thick LV), the patient becomes hypotensive on adrenaline. TOE: hyperdynamic LV, mitral leaflet touching the septum in systole, high LVOT gradient. Treatment?',
                           'Stop inotropes, give volume, a beta-blocker or phenylephrine', 'This is dynamic LVOT obstruction with systolic anterior motion of the mitral leaflet, which inotropes worsen. Fill, slow and constrict.',
                           'More adrenaline', 'Go back on bypass and replace the mitral valve'),
                'ct': ct(R(AC), 'axial')}

    std = [av_anat('as'), av_decide('as'), {**mv_sternotomy('as', 2), 'id': 'as-sternotomy'}, av_cannulate('as', 3), av_clamp('as', 4), av_aortotomy('as', 5), av_excise('as', 6),
           av_size('as', 7), av_sutures('as', 8), av_seat('as', 9), av_close('as', 10), av_wean('as', 11)]
    std[2] = {**std[2], 'body': '<p>Median sternotomy; open the pericardium and hitch it up. Look at the ascending aorta (size, calcification by palpation or epiaortic scan) before choosing the cannulation and clamp sites.</p>'
              + ev('epiaortic scanning finds atheroma that palpation misses; it changes cannulation or clamp site in a proportion of patients.')}
    hemi = [av_anat('ah'), av_decide('ah'),
            {'id': 'ah-access', 'phase': 'Access', 'seq': 2, 'title': 'Upper hemisternotomy (J into the right 3rd or 4th space)',
             'body': '<p>A 6–8 cm skin incision from just below the sternal notch. Saw the sternum in the midline from the notch down to the <b>3rd or 4th space</b> (choose from CT: the level of the root), then turn the cut out into that space on the <b>right</b> ("J"). Watch the <b>right internal thoracic vessels</b> at the J. The lower sternum stays whole.</p>'
                     '<p>Central cannulation is usually possible: the aorta directly, and venous drainage either through the right atrial appendage or <b>percutaneously from the femoral vein</b> under TOE. Convert to full sternotomy (extend the cut) if exposure is poor or there is bleeding.</p>'
                     + ev('randomised trials and meta-analyses comparing hemisternotomy with full sternotomy show similar mortality and valve results. Differences in bleeding, ventilation and stay are small and inconsistent, and cross-clamp and bypass times are slightly longer.'),
             'view': clook(V(LM['hemi']) if 'hemi' in LM else ST_MID, V([0.1, 1, 0.35]), 360), 'show': ['sternum', 'incision-hemi', 'hemi-cut', *CH], 'hide': HEART_OFF, 'opacity': {**FAINT, 'sternum': 0.95},
             'highlight': ['hemi-cut'], 'labels': ['hemi-cut'],
             'action': {'kind': 'reveal', 'label': 'Divide the upper sternum', 'port': 'sternotomy', 'ids': ['incision-hemi', 'hemi-cut']},
             'ct': ct(R(V(LM['hemi']) if 'hemi' in LM else ST_MID), 'axial', 'bone')},
            av_cannulate('ah', 3), av_clamp('ah', 4), av_aortotomy('ah', 5), av_excise('ah', 6), av_size('ah', 7), av_sutures('ah', 8), av_seat('ah', 9), av_close('ah', 10, mini=True), av_wean('ah', 11)]
    ramt_view = clook(AC, AN * 0.4 + V([0.65, 0.8, 0.0]), 130)
    rm = [av_anat('ar'), av_decide('ar'),
          {'id': 'ar-access', 'phase': 'Access', 'seq': 2, 'title': 'Right anterior mini-thoracotomy: select on CT first',
           'body': '<p><b>CT selection</b> (Glauber and colleagues): more than half of the ascending aorta lies to the <b>right of the right sternal border</b>; the aorta is <b>less than 10 cm</b> from the sternum; and the aorta\'s angle to the midline is over 45°. If not, choose hemisternotomy.</p>'
                   '<p>A 5–6 cm incision in the <b>right 2nd space</b> from the sternal edge. Ligate the right internal thoracic vessels (or keep them); divide the 3rd costal cartilage for more room if needed. '
                   '<b>Femoral venous</b> cannulation percutaneously, the arterial cannula in the <b>ascending aorta</b> directly (or femoral). CO₂ in the field; external defibrillator pads.</p>'
                   + ev('criteria and technique from Miceli, Ferrarini and Glauber (Ann Cardiothorac Surg 2015). Large series and meta-analyses report mortality similar to sternotomy with longer clamp times. The evidence is mostly observational and from experienced centres.'),
           'view': clook(V(LM['ramt']) if 'ramt' in LM else AC, V([0.45, 1.0, 0.25]), 330), 'show': ['skin', 'incision-ramt'], 'opacity': {'skin': 1.0}, 'hide': HEART_OFF,
           'highlight': ['incision-ramt'], 'labels': ['incision-ramt'], 'ct': ct(R(AC), 'axial')},
          av_clamp('ar', 3, port='ramt'), av_aortotomy('ar', 4, port='ramt'), av_excise('ar', 5, port='ramt', view=ramt_view), av_size('ar', 6, view=ramt_view, ramt=True),
          av_sutures('ar', 7, port='ramt', view=ramt_view), av_seat('ar', 8, port='ramt', view=ramt_view), av_close('ar', 9, mini=True), av_wean('ar', 10, mini=True)]
    rm[3]['show'] = [*CH, 'can-aortic', 'can-cp', 'can-lvvent']
    rm[3]['body'] = rm[3]['body'].replace('<p>On full bypass, clamp', '<p>Femoral venous and aortic cannulation are already in. On full bypass, clamp (through the incision, or a flexible clamp)')
    for key, appr, steps_, sq in (
            ('avr-std', 'Median sternotomy (Kouchoukos)', std, seq(('Anatomy', 'other'), ('Decide', 'other'), ('Sternotomy', 'other'), ('Cannulate', 'artery'), ('Clamp', 'artery'), ('Aortotomy', 'artery'), ('Excise', 'fissure'), ('Size', 'other'), ('Sutures', 'fissure'), ('Seat', 'bronchus'), ('Close', 'other'), ('Wean', 'artery'))),
            ('avr-hemi', 'Upper hemisternotomy', hemi, seq(('Anatomy', 'other'), ('Decide', 'other'), ('Hemisternotomy', 'other'), ('Cannulate', 'artery'), ('Clamp', 'artery'), ('Aortotomy', 'artery'), ('Excise', 'fissure'), ('Size', 'other'), ('Sutures', 'fissure'), ('Seat', 'bronchus'), ('Close', 'other'), ('Wean', 'artery'))),
            ('avr-ramt', 'Right anterior mini-thoracotomy', rm, seq(('Anatomy', 'other'), ('Decide', 'other'), ('Access', 'other'), ('Clamp', 'artery'), ('Aortotomy', 'artery'), ('Excise', 'fissure'), ('Size', 'other'), ('Sutures', 'fissure'), ('Seat', 'bronchus'), ('Close', 'other'), ('Wean', 'artery')))):
        for s in steps_:
            named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
            off_groups = {'arteries', 'veins', 'airway', 'lul-intra', 'lll-intra', 'rul-intra', 'nodes', 'nerves', 'pleura', 'segments', 'trauma', 'muscles', 'landmarks'}
            s['hide'] = [*s.get('hide', []), *[q['id'] for q in atlas['structures'] if q['group'] in off_groups and q['id'] not in named],
                         *[i for i in ('esophagus', 'thymus', 'thyroid') if has(i) and i not in named], *(['sternum'] if s.get('seq', 0) >= 3 else [])]
            if s['phase'] in ('Valve', 'Wean', 'Aorta'): s['opacity'] = {**{c_: 0.3 for c_ in AV_CANS}, **s.get('opacity', {})}
            if s['phase'] in ('Valve', 'Anatomy'):
                s['hide'] = [*s['hide'], 'svc', 'pa-trunk', 'ra', 'rv', 'la', 'myocardium', *[i for i in (*AV_CANS, 'can-svc', 'can-ivc', 'can-ostial') if i not in named]]
                s['opacity'] = {**s.get('opacity', {}), 'lv': 0.12, 'aorta': 0.12, 'lvot': 0.2}
                s['hide'] = [*s['hide'], 'esophagus', *[f'vert-t{i}' for i in range(1, 13)]]
            s['opacity'] = {**{f'vert-t{i}': 0.25 for i in range(2, 11)}, **s.get('opacity', {})}
            for kk in ('highlight', 'danger', 'labels', 'show'):
                if kk in s: s[kk] = [i for i in s[kk] if has(i) or i == 'skin']
        procs[key] = {'id': key, 'op': 'avr', 'opName': 'Aortic valve replacement', 'side': 'both', 'name': 'Aortic valve replacement', 'approach': appr,
                      'summary': 'Heart Team decision, access, bypass and protection by valve lesion, aortotomy, debridement, sizing against mismatch, annular sutures, prosthesis, de-airing, TOE.',
                      'ports': [], 'steps': steps_, 'sources': AVSRC, 'group': 'Cardiac', 'sequence': sq}
# ==================================================================================================== cardiac: tricuspid valve surgery
TV_OK = MVR_OK and has('tricuspid-annulus') and 'tv-centre' in LM
if TV_OK:
    TC, TN, TS_ = Lc('tv-centre'), V(LM['tv-normal']), V(LM['tv-septal']); TR_ = LM['tv-dims'][0]
    TVL = [i for i in ('tricuspid-annulus', 'tv-septal', 'tv-anterior', 'tv-posterior') if has(i)]
    TV_DANGER = [i for i in ('koch', 'tv-avnode', 'cs-ostium', 'rca-groove', 'cusp-n') if has(i)]
    TV_CANS = [i for i in ('can-aortic', 'can-svc', 'can-ivc', 'snares', 'can-cp') if has(i)]
    TV_FAINT = {**FAINT, 'ra': 0.15, 'rv': 0.22, 'lv': 0.15, 'aorta': 0.35, 'myocardium': 0.08}
    tv_view = clook(TC, TN * 1.0 + V([0.55, 0.35, 0.05]), 170)
    ev = lambda t: f'<p class="evidence"><b>Evidence:</b> {t}</p>'
    pm = lambda term: 'https://pubmed.ncbi.nlm.nih.gov/?term=' + term.replace(' ', '+')
    TVSRC = [
        {'title': 'Kouchoukos NT, Blackstone EH, Hanley FL, Kirklin JK. Kirklin/Barratt-Boyes Cardiac Surgery, 4th ed. Elsevier 2013: tricuspid valve disease', 'url': pm('Kirklin Barratt-Boyes cardiac surgery tricuspid')},
        {'title': 'Praz F, Borger MA, et al. 2025 ESC/EACTS Guidelines for the management of valvular heart disease. Eur Heart J 2025', 'url': 'https://www.escardio.org/guidelines/clinical-practice-guidelines/all-esc-practice-guidelines/valvular-heart-disease/'},
        {'title': '2025 ESC/EACTS valvular heart disease guidelines: practical updates on mitral and tricuspid regurgitation. Eur Heart J Suppl 2026', 'url': 'https://academic.oup.com/eurheartjsupp/article/28/Supplement_4/iv83/8512029'},
        {'title': 'Otto CM, Nishimura RA, et al. 2020 ACC/AHA guideline for the management of patients with valvular heart disease. Circulation 2021;143:e72-e227', 'url': 'https://pubmed.ncbi.nlm.nih.gov/33332150/'},
        {'title': 'Gammie JS, et al. Concomitant tricuspid repair in patients with degenerative mitral regurgitation (CTSN). N Engl J Med 2022;386:327-39', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa2115961'},
        {'title': 'Parolari A, Barili F, Pilozzi A, Pacini D. Ring or suture annuloplasty for tricuspid regurgitation? A meta-analysis review. Ann Thorac Surg 2014;98:2255-63', 'url': 'https://pubmed.ncbi.nlm.nih.gov/25443026/'},
        {'title': 'Ragnarsson S, et al. Pacemaker implantation following tricuspid valve annuloplasty (SWEDEHEART). JTCVS Open 2023', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10775064/'},
        {'title': 'Caldonazo T, et al. Beating versus arrested heart technique for isolated tricuspid valve surgery: meta-analysis of reconstructed time-to-event data. Innovations 2025', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12398632/'},
        {'title': 'Said SM, et al. Tricuspid valve replacement with mechanical versus biological prostheses: systematic review and meta-analysis. J Cardiothorac Surg 2024', 'url': 'https://link.springer.com/article/10.1186/s13019-024-03014-0'},
        {'title': 'Sorajja P, et al. Transcatheter repair for patients with tricuspid regurgitation (TRILUMINATE Pivotal). N Engl J Med 2023;388:1833-42', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa2300525'},
        {'title': 'Hahn RT, et al. Transcatheter valve replacement in severe tricuspid regurgitation (TRISCEND II). N Engl J Med 2025', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa2401918'},
    ]

    def tv_anat(pre):
        return {'id': f'{pre}-anatomy', 'phase': 'Anatomy', 'seq': 0, 'title': 'The tricuspid valve and what lies around it',
                'body': '<p>Three leaflets: the large <b>anterior</b>, the <b>posterior</b>, and the <b>septal</b>, hinged on the septum. In functional regurgitation the annulus dilates along the <b>anterior and posterior</b> leaflets; the <b>septal</b> part is held by the fibrous skeleton and hardly stretches. That is why rings and suture annuloplasties shorten the anterior and posterior annulus.</p>'
                        '<p>What the sutures can injure:</p><ul>'
                        '<li><b>The AV node and His bundle</b>, at the apex of <b>Koch\'s triangle</b> (coronary sinus ostium, tendon of Todaro, septal leaflet hinge), close to the <b>anteroseptal commissure</b>. Deep bites here cause heart block.</li>'
                        '<li><b>The right coronary artery</b>, a few millimetres outside the anterior and posterior annulus in the AV groove.</li>'
                        '<li><b>The non-coronary sinus</b> of the aortic root, just beyond the anteroseptal commissure.</li></ul>'
                        + ev('heart block is the commonest serious complication of tricuspid surgery. In a national Swedish registry of 1,502 annuloplasties, 14.2% needed a permanent pacemaker within 30 days, mostly for AV block (Ragnarsson et al., JTCVS Open 2023).'),
                'view': clook(TC, TN * 0.8 + V([0.6, 0.45, 0.15]), 230), 'spin': True, 'show': [*CH, *TVL, *TV_DANGER], 'hide': HEART_OFF, 'opacity': TV_FAINT,
                'highlight': ['tricuspid-annulus'], 'danger': TV_DANGER,
                'labels': ['tv-anterior', 'tv-posterior', 'tv-septal', 'koch', 'tv-avnode', 'cs-ostium', 'rca-groove', 'cusp-n'],
                'ask': ask('Where is the AV node relative to the tricuspid valve?', "At the apex of Koch's triangle, near the anteroseptal commissure",
                           'Bounded by the coronary sinus ostium, the tendon of Todaro and the septal leaflet hinge; the node sits at the apex, near the anteroseptal commissure.',
                           'Near the anteroposterior commissure', 'In the middle of the posterior annulus', 'Beside the right coronary artery'),
                'ct': ct(R(TC), 'axial')}

    def tv_decide(pre, replace=False):
        return {'id': f'{pre}-decide', 'phase': 'Decision', 'seq': 1, 'title': 'When to operate on the tricuspid, and how',
                'body': '<p><b>At left-sided valve surgery</b> (the commonest setting, often after rheumatic mitral disease):</p><ul>'
                        '<li><b>Severe TR</b>: operate on the tricuspid too. <b>Recommended</b> in both ESC/EACTS 2025 and ACC/AHA 2020 (class I).</li>'
                        '<li><b>Moderate TR</b>, or <b>milder TR with a dilated annulus</b> (40 mm or more, or 21 mm/m², on echo), or previous right heart failure: annuloplasty is <b>reasonable</b> (ACC/AHA class 2a; ESC should be considered). It stops progression but adds a pacemaker risk, so weigh atrial fibrillation, right atrial size and pulmonary pressure.</li></ul>'
                        '<p><b>Isolated severe TR</b>: surgery for symptomatic primary TR; for secondary TR, consider it before the RV fails. It is not for advanced biventricular failure or severe pulmonary hypertension. For high-risk patients, <b>transcatheter</b> repair (TEER) or replacement (TTVR) is a class IIa option (ESC 2025).</p>'
                        '<p><b>Repair or replace.</b> Annuloplasty for a dilated annulus with mobile leaflets. <b>Replace</b> when the leaflets are destroyed or tethered deep into the ventricle, or thickened and retracted (rheumatic, carcinoid, endocarditis, pacing-lead damage), or when a repair has failed.</p>'
                        + ev('CTSN trial (Gammie et al., NEJM 2022; 401 patients having mitral repair for degenerative MR, with moderate TR or a dilated annulus): adding annuloplasty cut the two-year composite of reoperation, progression or severe TR from 10.2% to 3.9%. It did not change mortality, and it raised the permanent pacemaker rate to 14.1% vs 2.5%. '
                             'TRILUMINATE (TEER, NEJM 2023) and TRISCEND II (TTVR, NEJM 2025) improved quality of life in severe TR; TTVR needed a new pacemaker in about a quarter of patients.'),
                'view': clook(TC, V([0.6, 0.7, 0.3]), 300), 'show': [*CH, *TVL], 'hide': HEART_OFF, 'opacity': TV_FAINT, 'labels': ['ra', 'rv', 'tricuspid-annulus'],
                'ask': ask('Rheumatic mitral stenosis for MVR. Mild TR, but the tricuspid annulus measures 44 mm and there has been right heart failure. What do the guidelines advise?',
                           'Tricuspid annuloplasty at the same operation is reasonable', 'An annulus of 40 mm or more, or previous right heart failure, makes concomitant annuloplasty reasonable even with less than severe TR: the annulus keeps dilating after the mitral operation, and reoperation for late TR carries high risk.',
                           'Leave it; mild TR regresses once the mitral valve is fixed', 'Replace the tricuspid valve'),
                'ct': ct(R(TC), 'axial')}

    def tv_cannulate(pre, seq_):
        return {'id': f'{pre}-cannulate', 'phase': 'Bypass', 'seq': seq_, 'title': 'Bicaval cannulation and snares',
                'body': '<p>Heparin. The aortic cannula as usual. <b>Separate SVC and IVC cannulas</b> (the IVC one low on the atrium, near the IVC junction, so the atriotomy is clear), with <b>snares</b> round both cavae. Tightened, they isolate the right atrium, so it can be opened without the venous line taking in air.</p>'
                        + ev('bicaval cannulation with snares is the standard for any right atrial opening (Kirklin/Barratt-Boyes). With vacuum-assisted drainage, some surgeons open the atrium without snaring.'),
                'view': clook((V(LM['can-svc']) + V(LM['can-ivc'])) / 2, V([0.8, 0.7, 0.1]), 360), 'show': [*CH, *[i for i in ('ivc',) if has(i)]], 'hide': HEART_OFF, 'opacity': {**FAINT, 'ivc': 0.6},
                'highlight': TV_CANS, 'labels': ['can-svc', 'can-ivc', 'snares', 'can-aortic', 'ivc'],
                'action': {'kind': 'reveal', 'label': 'Cannulate and snare', 'port': 'sternotomy', 'ids': TV_CANS},
                'ct': ct(R(V(LM['can-svc'])), 'axial')}

    def tv_beating(pre, seq_, port='sternotomy', beating_default=False):
        st = {'id': f'{pre}-heart', 'phase': 'Bypass', 'seq': seq_, 'title': 'Arrested or beating heart?',
              'body': '<p><b>Arrested</b> (cross-clamp and cardioplegia): a still, bloodless field; the usual choice when the tricuspid follows a mitral or aortic procedure under the same clamp.<br>'
                      '<b>Beating</b> (on bypass, clamp off, the right heart isolated by the snares): no cardioplegia or ischaemia for the left heart. The rhythm can be <b>watched as each suture is tied</b> near the AV node, and a stitch that causes block can be removed at once. '
                      'Any left-heart opening (a patent foramen ovale) risks air embolism, so check the septum on TOE first.</p>'
                      '<p>After a mitral operation the tricuspid is often done <b>after the clamp is off</b>, during reperfusion, which also shortens the ischaemic time.</p>'
                      + ev('a 2025 meta-analysis of 6 observational studies (767 isolated tricuspid operations; Caldonazo et al., Innovations) found <b>no difference</b> between beating and arrested hearts in permanent pacemaker rate, early or late mortality, or bypass time. The theoretical advantage in conduction safety has not been shown; randomised trials are lacking.'),
              'view': clook(Lc('clamp-ao'), V([0.3, 1, 0.3]), 280), 'show': [*CH, *TV_CANS], 'hide': HEART_OFF, 'opacity': FAINT, 'labels': ['snares'],
              'ask': ask('Isolated tricuspid ring annuloplasty on the beating heart. While tying the sutures near the anteroseptal commissure, complete heart block appears. Best move?',
                         'Cut and remove that suture, and place it more superficially or on the atrial side', 'Watching the rhythm while tying is the point of the beating-heart technique: a block that appears with one stitch often resolves when it is removed.',
                         'Carry on and put in a pacemaker later', 'Give atropine and continue'),
              'ct': ct(R(Lc('clamp-ao')), 'axial')}
        if not beating_default:
            st['action'] = {'kind': 'clamp', 'label': 'Apply the cross-clamp', 'port': port, 'at': R(Lc('clamp-ao')), 'axis': R(V(LM['ao-axis'])), 'radius': 14, 'jawLen': 50}
        return st

    def tv_ra(pre, seq_, port='sternotomy'):
        return {'id': f'{pre}-atriotomy', 'phase': 'Right atrium', 'seq': seq_, 'title': 'Right atriotomy',
                'body': '<p>Tighten the snares. Open the right atrium obliquely, from the base of the appendage toward the IVC, <b>parallel to the AV groove</b> and well above it (the right coronary), and <b>away from the sinus node</b> at the SVC junction. Stay sutures; a sucker into the coronary sinus if it floods the field.</p>'
                        + ev('the incision is placed by anatomy: the sinus node lies at the SVC-atrial junction (the crista terminalis), and the right coronary runs in the AV groove below.'),
                'view': clook(Lc('ra-incision'), V([1, 0.6, 0.2]), 240), 'show': [*CH, *TV_CANS, 'ra-incision'], 'hide': HEART_OFF, 'opacity': {**FAINT, 'ra': 0.5},
                'highlight': ['ra-incision'], 'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Open the right atrium', 'port': port, 'path': ext('ra-incision', 15)},
                'ct': ct(R(Lc('ra-incision')), 'axial')}

    def tv_assess(pre, seq_, replace=False):
        return {'id': f'{pre}-assess', 'phase': 'Valve', 'seq': seq_, 'title': 'Assess the valve and size the annulus',
                'body': '<p>Look at the leaflets (thickened, retracted, perforated, tethered), the chordae and the commissures. A <b>saline test</b> (fluid into the RV) shows where it leaks.</p>'
                        '<p><b>Size the ring</b> by the <b>septal leaflet length</b> (base of the septal leaflet, commissure to commissure) or the <b>anterior leaflet area</b>, with the maker\'s sizers. The septal annulus hardly dilates, so it guides the true size.</p>'
                        '<p><b>Signs a repair will not last</b>: severe tethering (a coaptation depth of about 8 mm or more on echo, or a large tenting area), a very dilated annulus with a failing RV, or destroyed or rheumatic leaflets. Then consider leaflet augmentation or <b>replacement</b>.</p>'
                        + ev('tethering depth and annular size predict residual and recurrent TR after annuloplasty in observational echo series; exact thresholds vary between studies.'),
                'view': tv_view, 'show': [*TVL, *TV_DANGER], 'hide': HEART_OFF, 'opacity': TV_FAINT, 'highlight': ['tricuspid-annulus'], 'danger': TV_DANGER,
                'labels': ['tv-anterior', 'tv-posterior', 'tv-septal'], 'ct': ct(R(TC), 'axial')}

    def tv_which(pre, seq_):
        return {'id': f'{pre}-which', 'phase': 'Valve', 'seq': seq_, 'title': 'Which repair: ring, De Vega, Kay, clover',
                'body': '<p><b>Ring annuloplasty</b> (incomplete or 3D ring, open at the AV node)<br><b>For:</b> fixes the annulus in a normal shape and resists late dilatation; the most durable repair. <b>Against:</b> foreign material; cost; a downsized rigid ring can kink the right coronary or cause stenosis.</p>'
                        '<p><b>De Vega</b> (double running suture, anteroseptal to posteroseptal commissure)<br><b>For:</b> quick, cheap, no prosthesis: attractive where rings are costly or scarce. <b>Against:</b> the suture can cut through or the annulus re-dilate ("guitar-string" effect): more late recurrence.</p>'
                        '<p><b>Kay</b> (bicuspidisation: obliterates the posterior leaflet annulus)<br><b>For:</b> simple, for moderate dilatation. <b>Against:</b> less durable than a ring when the annulus is very large.</p>'
                        '<p><b>Clover</b> (edge-to-edge, the three leaflets stitched at their centres, with a ring)<br>for complex or prolapsing leaflets.</p>'
                        + ev('meta-analysis of 9 studies (2 randomised; Parolari et al., Ann Thorac Surg 2014): rings protected against early mortality and late recurrence, with freedom from moderate or worse TR at 15 years of about 79% with a ring vs 60% with suture annuloplasty; late survival did not differ.'),
                'view': tv_view, 'show': [*TVL, 'tv-devega', *TV_DANGER], 'hide': HEART_OFF, 'opacity': TV_FAINT,
                'highlight': ['tv-devega'], 'labels': ['tv-devega', 'tv-avnode'], 'danger': ['tv-avnode', 'rca-groove'],
                'ask': ask('A young patient with functional TR (annulus 44 mm) at MVR. What does the evidence favour for durability?',
                           'Ring annuloplasty', 'Rings reduce late recurrence compared with suture techniques (Parolari 2014). A De Vega is a reasonable fallback where rings are not available.',
                           'De Vega', 'No tricuspid procedure', 'Replacement'),
                'ct': ct(R(TC), 'axial')}

    def tv_ring_sutures(pre, seq_, port='sternotomy', view=None):
        return {'id': f'{pre}-sutures', 'phase': 'Valve', 'seq': seq_, 'title': 'Ring sutures: round the anterior and posterior annulus',
                'body': '<p>Horizontal mattress sutures of <b>2-0 braided polyester</b> in the annulus, <b>not</b> the leaflet. Start just beyond the <b>anteroseptal commissure</b>, go round the <b>anterior and posterior</b> annulus, and finish on the <b>septal annulus</b> short of the coronary sinus. '
                        '<b>No sutures at the apex of Koch\'s triangle</b>: the ring\'s gap sits there. Bites posteriorly and anteriorly are firm but not deep (the right coronary lies just outside).</p>'
                        + ev('the incomplete ring was designed to leave the conduction tissue untouched. Heart block still occurs after annuloplasty; concomitant mitral surgery, ablation and a low-volume centre raised the risk in the Swedish registry (Ragnarsson 2023).'),
                'view': view or tv_view, 'show': [*TVL, *TV_DANGER], 'hide': HEART_OFF, 'opacity': TV_FAINT, 'highlight': ['tricuspid-annulus'], 'danger': TV_DANGER,
                'labels': ['tv-avnode', 'koch', 'rca-groove'],
                'action': {'kind': 'annulus', 'label': 'Place the annular sutures', 'port': port, 'at': R(TC + TN * 1.5), 'axis': R(TN), 'anterior': R(TS_), 'radius': TR_ - 0.5, 'count': 11},
                'ct': ct(R(TC), 'axial')}

    def tv_seat_ring(pre, seq_, port='sternotomy', view=None):
        return {'id': f'{pre}-ring', 'phase': 'Valve', 'seq': seq_, 'title': 'Seat and tie the ring; test',
                'body': '<p>Pass the sutures through the ring, lower it and tie. The ring pulls the dilated anterior and posterior annulus back to size. <b>Saline test</b>: fill the RV; the leaflets should meet along a good line of coaptation.</p>'
                        + ev('intraoperative TOE after bypass confirms the result: residual TR more than mild, or a gradient, is usually corrected before leaving theatre.'),
                'view': view or tv_view, 'show': [*TVL, 'tv-ring', 'tv-avnode', 'rca-groove'], 'hide': HEART_OFF, 'opacity': TV_FAINT,
                'highlight': ['tv-ring'], 'labels': ['tv-ring', 'tv-avnode'],
                'action': {'kind': 'seat', 'label': 'Seat the ring', 'port': port, 'ids': ['tv-ring'], 'from': R(TN * 40)},
                'ct': ct(R(TC), 'axial')}

    def tv_excise(pre, seq_, port='sternotomy'):
        return {'id': f'{pre}-excise', 'phase': 'Valve', 'seq': seq_, 'title': 'Excise the anterior and posterior leaflets; keep the septal',
                'body': '<p>Excise the <b>anterior and posterior leaflets</b>, leaving a 2–3 mm rim. Keep the <b>septal leaflet</b> (or fold it in): sutures placed <b>through the septal leaflet tissue</b>, rather than the annulus beneath it, stay clear of the AV node. Chordal preservation helps RV function.</p>'
                        + ev('sparing the septal leaflet and suturing through it is a standard way to avoid the conduction tissue (Kirklin/Barratt-Boyes). Pacemaker rates after replacement remain high in series.'),
                'view': tv_view, 'show': [*TVL, *TV_DANGER], 'hide': HEART_OFF, 'opacity': TV_FAINT, 'highlight': ['tv-anterior', 'tv-posterior'], 'danger': TV_DANGER,
                'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Excise the anterior and posterior leaflets', 'port': port, 'remove': [i for i in ('tv-anterior', 'tv-posterior') if has(i)],
                           'path': [R(TC + (TS_ * np.cos(np.radians(t)) + V(np.cross(TN, TS_)) * np.sin(np.radians(t))) * (TR_ - 1) + TN) for t in np.linspace(70, 290, 6)]},
                'ct': ct(R(TC), 'axial')}

    def tv_repl_sutures(pre, seq_, port='sternotomy'):
        return {'id': f'{pre}-sutures', 'phase': 'Valve', 'seq': seq_, 'title': 'Sutures for replacement: through the septal leaflet at the node',
                'body': '<p>Pledgeted mattress sutures of 2-0 braided polyester round the annulus. Along the septum, and especially at the apex of Koch\'s triangle, take them <b>through the base of the septal leaflet</b> rather than the annulus. '
                        'Some surgeons run the line <b>on the atrial side of the coronary sinus</b>, leaving the sinus draining into the ventricle.</p>'
                        + ev('technique by consensus; no trial compares the suture routes. Paravalvular leak and heart block are the two complications these routes are designed to avoid.'),
                'view': tv_view, 'show': ['tricuspid-annulus', 'tv-septal', *TV_DANGER], 'hide': [*HEART_OFF, 'tv-anterior', 'tv-posterior'], 'opacity': TV_FAINT, 'highlight': ['tricuspid-annulus'], 'danger': TV_DANGER,
                'labels': ['tv-avnode', 'cs-ostium'],
                'action': {'kind': 'annulus', 'label': 'Place the annular sutures', 'port': port, 'at': R(TC + TN * 1.5), 'axis': R(TN), 'anterior': R(TS_), 'radius': TR_ - 0.5, 'count': 14},
                'ct': ct(R(TC), 'axial')}

    def tv_seat_valve(pre, seq_, port='sternotomy'):
        return {'id': f'{pre}-seat', 'phase': 'Valve', 'seq': seq_, 'title': 'Seat the prosthesis: tissue or mechanical',
                'body': '<p>Lower and tie a <b>large</b> prosthesis (the right heart tolerates no gradient). Orient a stented valve so that <b>no post points into the RV outflow tract</b>, and check the leaflets move.</p>'
                        '<p><b>Tissue or mechanical?</b> In the low-pressure, low-flow right heart, a mechanical valve thromboses more readily, and a tissue valve can later take a <b>valve-in-valve</b> transcatheter valve. A transvenous pacing lead cannot pass a mechanical tricuspid valve, and will be trapped between a tissue valve and the annulus, so <b>place an epicardial lead</b> now if block is likely.</p>'
                        + ev('meta-analysis of 37 studies (8,316 prostheses; Said et al., J Cardiothorac Surg 2024): no difference in 30-day or long-term survival or reoperation; mechanical valves had about a sixfold higher risk of valve thrombosis.'),
                'view': tv_view, 'show': ['tricuspid-annulus', 'tv-prosthesis', 'tv-avnode', 'rca-groove'], 'hide': [*HEART_OFF, 'tv-anterior', 'tv-posterior', 'tv-septal'], 'opacity': TV_FAINT,
                'highlight': ['tv-prosthesis'], 'labels': ['tv-prosthesis'],
                'action': {'kind': 'seat', 'label': 'Seat the valve', 'port': port, 'ids': ['tv-prosthesis'], 'from': R(TN * 45)},
                'ask': ask('A 30-year-old woman needs tricuspid replacement for rheumatic disease; her mitral valve was repaired. Which prosthesis does the evidence favour?',
                           'A bioprosthesis', 'Survival is the same either way, but mechanical tricuspid valves thrombose far more often (about sixfold); a tissue valve also allows later valve-in-valve and avoids warfarin in pregnancy.',
                           'A mechanical valve for durability', 'A homograft'),
                'ct': ct(R(TC), 'axial')}

    def tv_close(pre, seq_, beating=False):
        return {'id': f'{pre}-close', 'phase': 'Wean', 'seq': seq_, 'title': 'Close the atrium, release the snares, wean',
                'body': '<p>Close the right atrium in two layers of 4-0 polypropylene, de-airing as the last stitches go in, and release the snares. ' + ('' if beating else 'Release the cross-clamp with the root vent on. ') +
                        'Place <b>atrial and ventricular pacing wires</b>.</p>'
                        '<p><b>TOE</b>: residual TR, gradient, RV function. The RV that has pumped against regurgitation now pumps against a competent valve (a higher afterload). With pulmonary hypertension, support it: inotropes, inhaled pulmonary vasodilators, and a <b>slow wean</b>.</p>'
                        '<p>Check the rhythm: heart block after tricuspid surgery may recover over days; persistent block needs a permanent pacemaker.</p>'
                        + ev('RV dysfunction and pulmonary hypertension are the main predictors of death after tricuspid surgery, which is why guidelines advise operating before RV failure (ESC/EACTS 2025).'),
                'view': clook(Lc('ra-incision'), V([1, 0.6, 0.2]), 260), 'show': [*CH, 'ra-incision', *TV_CANS], 'hide': HEART_OFF, 'opacity': FAINT,
                'labels': ['ra-incision', 'snares'],
                'action': {'kind': 'decannulate', 'label': 'Wean and decannulate', 'port': 'sternotomy', 'ids': [i for i in ('snares', 'can-cp', 'can-ivc', 'can-svc', 'can-aortic') if has(i)]},
                'ct': ct(R(TC), 'axial')}

    ring = [tv_anat('tr'), tv_decide('tr'), {**mv_sternotomy('tr', 2), 'id': 'tr-sternotomy'}, tv_cannulate('tr', 3), tv_beating('tr', 4), tv_ra('tr', 5), tv_assess('tr', 6), tv_which('tr', 7),
            tv_ring_sutures('tr', 8), tv_seat_ring('tr', 9), tv_close('tr', 10)]
    repl = [tv_anat('tx'), tv_decide('tx', replace=True), {**mv_sternotomy('tx', 2), 'id': 'tx-sternotomy'}, tv_cannulate('tx', 3), tv_beating('tx', 4), tv_ra('tx', 5), tv_assess('tx', 6, replace=True),
            tv_excise('tx', 7), tv_repl_sutures('tx', 8), tv_seat_valve('tx', 9), tv_close('tx', 10)]
    mics_tv = clook(TC, TN * 0.8 + V([1, 0.3, 0.05]), 180)
    mi_tv = [tv_anat('tm'), tv_decide('tm'),
             {'id': 'tm-setup', 'phase': 'Access', 'seq': 2, 'title': 'Right mini-thoracotomy, peripheral cannulation, snares',
              'body': '<p>Supine, right chest raised about 30°, external pads, TOE. <b>Femoral venous</b> drainage plus an <b>SVC</b> (right internal jugular) cannula, or a single femoral cannula with vacuum; femoral arterial return. '
                      'A 4–6 cm incision in the <b>right 4th space</b>; CO₂ in the field; snares round both cavae (or occlusion balloons).</p>'
                      '<p>An attractive route for <b>isolated</b> or <b>redo</b> tricuspid surgery after sternotomy: no re-entry, and it is often done on the <b>beating heart</b> without a clamp.</p>'
                      + ev('observational series only; outcomes depend on centre experience. For redo isolated tricuspid surgery, avoiding re-sternotomy is the main practical argument.'),
              'view': clook(Lc('mics'), V([1, 0.6, 0.25]), 380), 'show': ['skin', 'incision-mics'], 'opacity': {'skin': 1.0}, 'hide': HEART_OFF,
              'highlight': ['incision-mics'], 'labels': ['incision-mics'], 'ct': ct(R(Lc('mics')), 'axial', 'lung')},
             tv_beating('tm', 3, beating_default=True), tv_ra('tm', 4, port='mics'), tv_assess('tm', 5), tv_ring_sutures('tm', 6, port='mics', view=mics_tv), tv_seat_ring('tm', 7, port='mics', view=mics_tv), tv_close('tm', 8, beating=True)]
    for key, appr, steps_, sq in (
            ('tv-ring', 'Ring annuloplasty (sternotomy)', ring, seq(('Anatomy', 'other'), ('Decide', 'other'), ('Sternotomy', 'other'), ('Cannulate', 'artery'), ('Arrest?', 'artery'), ('Atriotomy', 'vein'), ('Assess', 'other'), ('Which repair', 'other'), ('Sutures', 'fissure'), ('Ring', 'bronchus'), ('Close', 'other'))),
            ('tv-replace', 'Replacement (sternotomy)', repl, seq(('Anatomy', 'other'), ('Decide', 'other'), ('Sternotomy', 'other'), ('Cannulate', 'artery'), ('Arrest?', 'artery'), ('Atriotomy', 'vein'), ('Assess', 'other'), ('Excise', 'fissure'), ('Sutures', 'fissure'), ('Seat', 'bronchus'), ('Close', 'other'))),
            ('tv-mics', 'Right mini-thoracotomy, beating heart', mi_tv, seq(('Anatomy', 'other'), ('Decide', 'other'), ('Access', 'other'), ('Beating', 'artery'), ('Atriotomy', 'vein'), ('Assess', 'other'), ('Sutures', 'fissure'), ('Ring', 'bronchus'), ('Close', 'other')))):
        for s in steps_:
            named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
            off_groups = {'arteries', 'veins', 'airway', 'lul-intra', 'lll-intra', 'rul-intra', 'nodes', 'nerves', 'pleura', 'segments', 'trauma', 'muscles', 'landmarks'}
            s['hide'] = [*s.get('hide', []), *[q['id'] for q in atlas['structures'] if q['group'] in off_groups and q['id'] not in named],
                         *[i for i in ('esophagus', 'thymus', 'thyroid') if has(i) and i not in named], *(['sternum'] if s.get('seq', 0) >= 3 else [])]
            if s['phase'] in ('Valve', 'Anatomy'):
                s['hide'] = [*s['hide'], 'svc', 'la', *[i for i in (*TV_CANS, 'can-retro') if i not in named], *[f'vert-t{i}' for i in range(1, 13)]]
            s['opacity'] = {**{f'vert-t{i}': 0.25 for i in range(2, 11)}, **s.get('opacity', {})}
            for kk in ('highlight', 'danger', 'labels', 'show'):
                if kk in s: s[kk] = [i for i in s[kk] if has(i) or i == 'skin']
        procs[key] = {'id': key, 'op': 'tricuspid', 'opName': 'Tricuspid valve surgery', 'side': 'both', 'name': 'Tricuspid valve surgery', 'approach': appr,
                      'summary': 'Guideline indications, bicaval snares, arrested or beating, assessment and sizing, ring vs suture repair, replacement, conduction and RCA safety.',
                      'ports': [], 'steps': steps_, 'sources': TVSRC, 'group': 'Cardiac', 'sequence': sq}
# ==================================================================================================== cardiac: aortic root replacement (Bentall, Ross)
ROOT_OK = AVR_OK and has('cvg') and 'root-distal' in LM
if ROOT_OK:
    RD_ = Lc('root-distal'); RT_ = Lc('root-top')
    BTN = [i for i in ('button-l', 'button-r') if has(i)]; BTN_G = [i for i in ('button-l-graft', 'button-r-graft') if has(i)]
    ROSS = has('pa-root') and 'pa-root' in LM
    root_view = clook(RT_, AN * 0.3 + V([0.2, 1.0, 0.1]), 190)
    RTSRC = [
        {'title': 'Kouchoukos NT, Blackstone EH, Hanley FL, Kirklin JK. Kirklin/Barratt-Boyes Cardiac Surgery, 4th ed. Elsevier 2013: aortic root replacement; Ross operation', 'url': pm('Kirklin Barratt-Boyes aortic root replacement')},
        {'title': 'Bentall H, De Bono A. A technique for complete replacement of the ascending aorta. Thorax 1968;23:338-9', 'url': pm('Bentall De Bono technique complete replacement ascending aorta Thorax 1968')},
        {'title': 'Isselbacher EM, et al. 2022 ACC/AHA guideline for the diagnosis and management of aortic disease. Circulation 2022', 'url': 'https://pubmed.ncbi.nlm.nih.gov/36322642/'},
        {'title': 'Praz F, Borger MA, et al. 2025 ESC/EACTS Guidelines for the management of valvular heart disease. Eur Heart J 2025', 'url': 'https://www.escardio.org/guidelines/clinical-practice-guidelines/all-esc-practice-guidelines/valvular-heart-disease/'},
        {'title': 'Otto CM, Nishimura RA, et al. 2020 ACC/AHA guideline for the management of patients with valvular heart disease. Circulation 2021', 'url': 'https://pubmed.ncbi.nlm.nih.gov/33332150/'},
        {'title': 'Pantaleo A, et al. Biological versus mechanical Bentall procedure for aortic root replacement: propensity score analysis of 1112 patients. Eur J Cardiothorac Surg 2017;52:143-9', 'url': 'https://academic.oup.com/ejcts/article/52/1/143/3603551'},
        {'title': 'Evolution and current applications of the Cabrol procedure and its modifications. Ann Thorac Surg 2011', 'url': 'https://www.annalsthoracicsurgery.org/article/S0003-4975(11)00251-7/fulltext'},
        {'title': 'Ross DN. Replacement of aortic and mitral valves with a pulmonary autograft. Lancet 1967;2:956-8', 'url': pm('Ross replacement aortic mitral valves pulmonary autograft Lancet 1967')},
        {'title': 'El-Hamamsy I, et al. Long-term outcomes after autograft versus homograft aortic root replacement in adults with aortic valve disease: a randomised controlled trial. Lancet 2010;376:524-31', 'url': 'https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(10)60828-8/abstract'},
        {'title': 'EACTS Expert Consensus Statement on the Ross procedure in adult patients. Eur J Cardiothorac Surg 2025;68:ezaf295', 'url': 'https://academic.oup.com/ejcts/article/68/2/ezaf295/8276889'},
        {'title': 'Ross procedure in rheumatic aortic valve disease (81 patients). Eur J Cardiothorac Surg 2006;29:156', 'url': 'https://academic.oup.com/ejcts/article/29/2/156/533641'},
        {'title': 'The Ross procedure: clinical relevance, guidelines recognition, and centers of excellence (editorial). J Am Coll Cardiol 2022', 'url': 'https://www.sciencedirect.com/science/article/pii/S073510972200064X'},
    ]

    def rt_anat(pre, ross=False):
        return {'id': f'{pre}-anatomy', 'phase': 'Anatomy', 'seq': 0, 'title': 'The aortic root as a unit' + (' and the pulmonary root beside it' if ross else ''),
                'body': '<p>The <b>root</b> runs from the ventricular attachment of the cusps (the annulus) to the <b>sinotubular junction</b>: the cusps, the three <b>sinuses of Valsalva</b>, the <b>coronary ostia</b> and the interleaflet triangles. In a root aneurysm the sinuses dilate and the STJ effaces, so replacing only the valve or only the tube above leaves diseased sinuses behind.</p>'
                        + ('<p>The <b>pulmonary root</b> lies in front and to the left, sharing a fascial plane with the aortic root. <b>Behind it</b> run the <b>left main coronary</b> and, beneath its posterior RVOT, the <b>first septal perforator</b> of the LAD. Both are at risk during harvest.</p>' if ross else '')
                        + ev('the root as a functional unit is from Anderson\'s anatomical work and the Kirklin/Barratt-Boyes descriptions; the sinuses and the STJ shape how the cusps close, which is why valve-sparing operations restore them.'),
                'view': clook(RT_, AN * 0.4 + V([0.3, 0.9, 0.2]), 230), 'spin': True,
                'show': [*CH, 'root-aneurysm', *ROOT, *AV_DANGER, *(['pa-root', 'septal-perforator'] if ross and ROSS else [])], 'hide': HEART_OFF,
                'opacity': {**AV_FAINT, 'pa-trunk': 0.2, 'rv': 0.2}, 'highlight': ['root-aneurysm'] if not ross else ['pa-root'],
                'danger': ['ostium-l', 'ostium-r', *(['septal-perforator'] if ross else [])],
                'labels': ['root-aneurysm', 'ostium-l', 'ostium-r', 'stj', *(['pa-root', 'septal-perforator'] if ross else [])],
                'ask': ask('Why is a Bentall (or valve-sparing root) needed for a root aneurysm rather than an AVR plus a supracoronary tube?',
                           'The dilated sinuses would be left behind and keep enlarging', 'In root disease the sinuses themselves are aneurysmal; a supracoronary graft leaves them in place to dilate or dissect.',
                           'The coronary ostia are always too low', 'An AVR cannot be done in a large root'),
                'ct': ct(R(AC), 'coronal')}

    def rt_decide(pre, ross=False):
        body = ('<p><b>When to replace the root</b> (ACC/AHA 2022): sporadic aneurysm at <b>5.5 cm</b> (5.0 cm is reasonable with an experienced surgeon), or with symptoms or rapid growth; <b>bicuspid</b> at 5.5 cm, at 5.0–5.4 cm with risk factors, and at <b>4.5 cm or more when the valve is being operated on anyway</b>; '
                '<b>Marfan</b> at 5.0 cm (4.5 cm with a family history of dissection or rapid growth); <b>Loeys–Dietz</b> by variant, size and growth.</p>'
                '<p><b>Which root operation:</b></p><ul>'
                '<li><b>Valve-sparing root replacement</b> (David, Yacoub) when the cusps are good: no prosthesis, no warfarin. It needs experience.</li>'
                '<li><b>Bentall</b> (composite valved graft) when the valve is diseased: the reliable standard. <b>Mechanical</b> for the young with reliable INR monitoring, <b>biological</b> ("bio-Bentall") for the older.</li>'
                '<li><b>Ross</b> (pulmonary autograft) for selected young adults with aortic valve disease, at experienced centres.</li></ul>')
        if ross:
            body += ('<p><b>Ross: good candidates</b>: young adults (typically under about 50–60) with a long life expectancy, active lives, women planning pregnancy (no warfarin). <b>Cautions</b> (EACTS consensus 2025): <b>rheumatic</b> valve disease, connective tissue disease, a <b>dilated annulus</b> or severe AR (dilatation risk), and a need for other valve surgery.</p>'
                     '<p><b>Rheumatic disease</b> matters here: in 81 rheumatic Ross patients (mean age 29.5), freedom from autograft dysfunction was 65% under 30 vs 98.5% over 30, and explanted autografts showed rheumatic valvulitis.</p>')
        ev_t = ('Ross: the one RCT (El-Hamamsy et al., Lancet 2010; 228 adults, mean age 38) found 10-year survival of 97% after the autograft vs 83% after a homograft root, and 99% vs 51% freedom from aortic valve reoperation at 13 years; survival matched the general population. '
                'ACC/AHA 2020 gives the Ross a class 2b recommendation in young adults, at experienced centres; ESC/EACTS 2025 calls it a valid alternative in well-selected young patients.' if ross else
                'no randomised trial compares mechanical and biological Bentall; in a propensity-matched series of 1,112 patients (Pantaleo et al., EJCTS 2017), 5-year survival did not differ (84% vs 87%), with more reoperation after tissue valves and more bleeding after mechanical ones.')
        return {'id': f'{pre}-decide', 'phase': 'Decision', 'seq': 1, 'title': 'Which root operation' + (': is this patient for a Ross?' if ross else ''),
                'body': body + ev(ev_t),
                'view': clook(RT_, V([0.3, 0.9, 0.4]), 300), 'show': [*CH, 'root-aneurysm'], 'hide': HEART_OFF, 'opacity': AV_FAINT, 'labels': ['root-aneurysm'],
                'ask': (ask('A 24-year-old with rheumatic aortic regurgitation and mild mitral disease asks for the Ross operation. What does the evidence suggest?',
                            'Rheumatic disease is a caution: the autograft can be affected and fail early; a mechanical valve (or a repair) is usually the better choice',
                            'In young rheumatic patients autograft dysfunction was far more common (freedom only 65%), with rheumatic changes in the explanted autografts; EACTS 2025 lists rheumatic disease among the cautions.',
                            'The Ross is ideal: young and wants to avoid warfarin', 'A homograft root') if ross else
                        ask('A 45-year-old with a 5.6 cm root aneurysm and a severely stenotic bicuspid valve. What operation fits?',
                            'A Bentall (mechanical composite graft, if INR monitoring is reliable)', 'The root is past 5.5 cm and the valve is diseased, so valve-sparing is not an option; at 45 guidelines favour a mechanical valve if anticoagulation is safe.',
                            'AVR alone', 'A valve-sparing root replacement', 'Surveillance')),
                'ct': ct(R(AC), 'coronal')}

    def rt_cannulate(pre, seq_):
        return {'id': f'{pre}-cannulate', 'phase': 'Bypass', 'seq': seq_, 'title': 'Cannulate distally, vent, protect',
                'body': '<p>The arterial cannula <b>high</b>: distal ascending aorta or arch, or the <b>right axillary artery</b> if the aorta is large up to the arch. A two-stage venous cannula; an <b>LV vent</b> through the right superior pulmonary vein; retrograde cardioplegia in the coronary sinus.</p>'
                        + ev('practice by consensus (Kirklin/Barratt-Boyes): the whole root is replaced, so the cannula and clamp must sit above the diseased segment.'),
                'view': clook(V(LM['can-aortic']), V([0.4, 0.9, 0.35]), 330), 'show': CH, 'hide': HEART_OFF, 'opacity': FAINT,
                'highlight': AV_CANS, 'labels': AV_CANS,
                'action': {'kind': 'reveal', 'label': 'Place the cannulas', 'port': 'sternotomy', 'ids': AV_CANS}, 'ct': ct(R(V(LM['can-aortic'])), 'axial')}

    def rt_excise(pre, seq_, ross=False):
        return {'id': f'{pre}-excise', 'phase': 'Root', 'seq': seq_, 'title': 'Transect, excise the sinuses, keep the coronary buttons',
                'body': '<p>Cross-clamp high; cardioplegia <b>retrograde</b> and <b>directly into the ostia</b> once open. Transect the aorta above the STJ. Excise the <b>valve</b> and the <b>sinus walls</b>, leaving a 3–5 mm rim at the annulus and a <b>button</b> of sinus wall (5–8 mm) round each coronary ostium.</p>'
                        '<p>Mobilise each button <b>just enough</b> to reach the graft: the left main is short and lies behind the pulmonary trunk; the right coronary has branches (conus, RV branches) that tether it.</p>'
                        + ev('the "open" button technique replaced the older inclusion and wrap methods, which were associated with pseudoaneurysms at the coronary suture lines. When buttons cannot be mobilised (redo, low ostia), a small interposition graft (Cabrol) is used.'),
                'view': root_view, 'show': [*ROOT, 'root-aneurysm', *BTN, *AV_DANGER], 'hide': [*HEART_OFF, 'aorta'], 'opacity': AV_FAINT,
                'highlight': BTN, 'danger': ['ostium-l', 'ostium-r'], 'labels': [*BTN, 'root-aneurysm'],
                'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Excise the root, keep the buttons', 'port': 'sternotomy', 'remove': ['root-aneurysm', *CUSPS],
                           'path': [R(AC + (AE * np.cos(t) + V(np.cross(AN, AE)) * np.sin(t)) * (AR_ + 8) + AN * 20) for t in np.linspace(0, 2 * np.pi, 7)]},
                'ct': ct(R(AC), 'coronal')}

    def rt_graft(pre, seq_):
        return {'id': f'{pre}-graft', 'phase': 'Root', 'seq': seq_, 'title': 'Sew the composite graft to the annulus',
                'body': '<p>Size the annulus. Place <b>pledgeted horizontal mattress sutures</b> round the annulus (non-everting, pledgets below, so the valve sits supra-annular), through the sewing ring of the composite graft, and tie. Shallow bites under the right–non-coronary commissure (His bundle).</p>'
                        '<p>The graft diameter is usually the annulus size plus a few millimetres, so the coronary buttons can be reached and the distal aorta matched.</p>'
                        + ev('pledgeted interrupted sutures are the usual choice at the proximal line, where bleeding after the graft is in is hard to reach; there is no trial of techniques in root replacement.'),
                'view': root_view, 'show': ['aortic-annulus', 'cvg', *BTN, 'his-bundle'], 'hide': [*HEART_OFF, 'aorta', *CUSPS], 'opacity': AV_FAINT,
                'highlight': ['cvg'], 'labels': ['cvg', *BTN],
                'action': {'kind': 'seat', 'label': 'Seat the composite graft', 'port': 'sternotomy', 'ids': ['cvg'], 'from': R(AN * 60)}, 'ct': ct(R(AC), 'coronal')}

    def rt_buttons(pre, seq_, into='cvg'):
        return {'id': f'{pre}-buttons', 'phase': 'Root', 'seq': seq_, 'title': 'Reimplant the coronary buttons',
                'body': '<p>Cut a hole in the graft (cautery) opposite each ostium. Sew each button end-to-side with running <b>5-0 polypropylene</b>, often with a felt or pericardial strip on the outside. '
                        '<b>Left main first</b> (lower, posterior). Place the <b>right</b> with the heart <b>filled</b> (release the vent briefly, or fill the root) so it is not too low, twisted or stretched.</p>'
                        '<p>Test each suture line (cardioplegia down the graft) <b>before</b> the distal anastomosis: afterwards the back of these suture lines is hard to reach.</p>'
                        + ev('coronary button problems (kinking, tension, bleeding) are the main technical causes of early death and ischaemia after root replacement in series; positioning with the heart filled is standard advice.'),
                'view': root_view, 'show': ['aortic-annulus', into, *BTN_G, 'ostium-l', 'ostium-r'], 'hide': [*HEART_OFF, 'aorta', *CUSPS, *BTN], 'opacity': AV_FAINT,
                'highlight': BTN_G, 'danger': ['ostium-l', 'ostium-r'], 'labels': [*BTN_G],
                'action': {'kind': 'reveal', 'label': 'Sew on the buttons', 'port': 'sternotomy', 'ids': BTN_G},
                'ask': ask('After a Bentall, the patient comes off bypass with inferior ST elevation and a failing RV. The left side looks fine. Most likely?',
                           'The right coronary button is kinked or under tension', 'The right button, placed too low or with the heart empty, kinks when the heart fills. Go back on bypass and redo it (or bypass the RCA).',
                           'Air in the left main', 'A paravalvular leak'),
                'ct': ct(R(AC), 'coronal')}

    def rt_distal(pre, seq_, into='cvg'):
        return {'id': f'{pre}-distal', 'phase': 'Wean', 'seq': seq_, 'title': 'Distal anastomosis, de-air, check',
                'body': '<p>Cut the graft to length and join it to the ascending aorta with running <b>4-0 polypropylene</b>, often with a felt strip. De-air through the root vent in the graft, release the clamp, and check <b>every suture line</b>, especially the backs of the buttons.</p>'
                        '<p><b>TOE</b>: valve function, <b>regional wall motion</b> in the left and right coronary territories, no leak. Bleeding is the other big risk: keep haemostatic agents and blood ready.</p>'
                        + ev('bleeding and coronary problems dominate early morbidity after root replacement in large series (Kirklin/Barratt-Boyes).'),
                'view': clook(RD_, AN * 0.3 + V([0.2, 1.0, 0.1]), 220), 'show': [*CH, into, *BTN_G, 'root-distal', 'can-aortic'], 'hide': [*HEART_OFF, *CUSPS, *BTN], 'opacity': {**FAINT, 'aorta': 0.5},
                'highlight': ['root-distal'], 'labels': ['root-distal', into],
                'action': {'kind': 'suture', 'label': 'Distal anastomosis', 'port': 'sternotomy', 'path': [R(RD_ + (AE * np.cos(t) + V(np.cross(AN, AE)) * np.sin(t)) * (AR_ + 3.5)) for t in np.linspace(0, 2 * np.pi, 13)], 'normal': R(AN), 'axis': R(AE)},
                'ct': ct(R(RD_), 'axial')}

    bentall = [rt_anat('rb'), rt_decide('rb'), {**mv_sternotomy('rb', 2), 'id': 'rb-sternotomy'}, rt_cannulate('rb', 3), {**av_clamp('rb', 4), 'id': 'rb-clamp'},
               rt_excise('rb', 5), rt_graft('rb', 6), rt_buttons('rb', 7), rt_distal('rb', 8), {**av_wean('rb', 9), 'show': [*CH, 'cvg', *BTN_G, *AV_CANS], 'hide': [*HEART_OFF, *CUSPS, *BTN]}]
    procs['root-bentall'] = {'steps': bentall, 'appr': 'Bentall (composite valved graft)',
                             'sq': seq(('Anatomy', 'other'), ('Decide', 'other'), ('Sternotomy', 'other'), ('Cannulate', 'artery'), ('Clamp', 'artery'), ('Excise', 'fissure'), ('Graft', 'bronchus'), ('Buttons', 'artery'), ('Distal', 'artery'), ('Wean', 'artery'))}
    if ROSS:
        PC_, PN_ = Lc('pa-root'), V(LM['pa-axis'])
        ross_view = clook(PC_, PN_ * 0.3 + V([0.1, 1.0, 0.3]), 200)
        ross = [rt_anat('rr', ross=True), rt_decide('rr', ross=True), {**mv_sternotomy('rr', 2), 'id': 'rr-sternotomy'}, rt_cannulate('rr', 3), {**av_clamp('rr', 4), 'id': 'rr-clamp'},
                {**rt_excise('rr', 5, ross=True), 'title': 'Transect the aorta, excise the valve, take the buttons'},
                {'id': 'rr-harvest', 'phase': 'Pulmonary root', 'seq': 6, 'title': 'Harvest the pulmonary autograft',
                 'body': '<p>Transect the <b>PA trunk</b> just below its bifurcation and look at the pulmonary valve: it must be <b>tricuspid and competent</b>, or the Ross is abandoned. Open the <b>RVOT 3–5 mm below the valve</b> (a right-angle clamp through the valve marks the level).</p>'
                         '<p>Free the root from the septum <b>posteriorly and to the left</b>, keeping the plane shallow: the <b>first septal perforator</b> runs just beneath, and the <b>left main</b> lies behind the root. Keep the muscle cuff thin but intact.</p>'
                         + ev('injury to the first septal perforator (septal infarction, ventricular arrhythmia) is a recognised harvest complication; the EACTS 2025 consensus highlights preserving it.'),
                 'view': ross_view, 'show': [*CH, 'pa-root', 'pa-harvest', 'septal-perforator', 'ostium-l'], 'hide': HEART_OFF, 'opacity': {**AV_FAINT, 'pa-trunk': 0.15, 'rv': 0.25},
                 'highlight': ['pa-harvest'], 'danger': ['septal-perforator', 'ostium-l'], 'labels': ['pa-root', 'septal-perforator', 'ostium-l'],
                 'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Harvest the autograft', 'port': 'sternotomy',
                            'path': [R(PC_ + (V(np.cross(PN_, [0, 0, 1])) * np.cos(t) + V(np.cross(PN_, np.cross(PN_, [0, 0, 1]))) * np.sin(t)) * 14 - PN_ * 5) for t in np.linspace(0, 2 * np.pi, 7)]},
                 'ask': ask('During harvest, dissection is carried deep into the muscle behind the pulmonary root. What is at risk?',
                            'The first septal perforator of the LAD', 'It runs just beneath the posterior RVOT; dividing it gives a septal infarct. Stay shallow and close to the root.',
                            'The right coronary artery', 'The AV node'),
                 'ct': ct(R(PC_), 'axial')},
                {'id': 'rr-implant', 'phase': 'Root', 'seq': 7, 'title': 'Implant the autograft as a root; stabilise it',
                 'body': '<p>Sew the autograft to the aortic annulus with <b>interrupted</b> sutures (or a running suture with a strip), <b>in the same plane</b> as the annulus so the cusps do not distort. The pulmonary sinuses are thinner than aortic ones.</p>'
                         '<p>To limit later <b>dilatation</b> (the main long-term failure mode): reduce and fix the annulus if it is large, keep the STJ at the right size, and consider <b>reinforcing</b> the autograft (inclusion inside a polyester graft, or an external wrap), especially with pre-existing AR or a big annulus.</p>'
                         + ev('the EACTS 2025 consensus recommends annular assessment and stabilisation, and considering autograft reinforcement in higher-risk roots. The full-root technique is the most widely used; the best form of reinforcement is not settled.'),
                 'view': root_view, 'show': ['aortic-annulus', 'autograft-ao', *BTN, 'his-bundle'], 'hide': [*HEART_OFF, 'aorta', 'pa-root', *CUSPS], 'opacity': AV_FAINT,
                 'highlight': ['autograft-ao'], 'labels': ['autograft-ao', *BTN],
                 'action': {'kind': 'seat', 'label': 'Move the autograft to the aortic position', 'port': 'sternotomy', 'ids': ['autograft-ao'], 'from': R(PC_ - AC)},
                 'ct': ct(R(AC), 'coronal')},
                {**rt_buttons('rr', 8, into='autograft-ao'), 'body': rt_buttons('rr', 8)['body'].replace('Cut a hole in the graft (cautery) opposite each ostium.', 'Open the autograft\'s facing sinuses opposite each ostium.')},
                {**rt_distal('rr', 9, into='autograft-ao'), 'title': 'Join the autograft to the ascending aorta'},
                {'id': 'rr-homograft', 'phase': 'Pulmonary root', 'seq': 10, 'title': 'Rebuild the RVOT with a pulmonary homograft',
                 'body': '<p>Sew a <b>pulmonary homograft</b> (or another conduit if none is available) to the PA bifurcation and to the RVOT with running 4-0 polypropylene, often on the beating heart after the clamp is off. It lies in the low-pressure circuit, so it lasts longer than a homograft would in the aortic position.</p>'
                         + ev('the homograft in the pulmonary position is the second valve at risk after a Ross; in long-term series its degeneration, rather than the autograft\'s, is a common reason for reintervention, often transcatheter.'),
                 'view': ross_view, 'show': [*CH, 'autograft-ao', 'homograft'], 'hide': [*HEART_OFF, 'pa-root'], 'opacity': {**FAINT, 'pa-trunk': 0.15},
                 'highlight': ['homograft'], 'labels': ['homograft', 'autograft-ao'],
                 'action': {'kind': 'seat', 'label': 'Sew in the homograft', 'port': 'sternotomy', 'ids': ['homograft'], 'from': R(PN_ * 40 + V([0, 30, 0]))},
                 'ct': ct(R(PC_), 'axial')},
                {**av_wean('rr', 11), 'title': 'Wean; control blood pressure', 'show': [*CH, 'autograft-ao', 'homograft', *BTN_G, *AV_CANS], 'hide': [*HEART_OFF, *CUSPS, *BTN, 'pa-root'],
                 'body': '<p>TOE: autograft competence (no AR), RVOT gradient across the homograft, regional wall motion (coronary buttons, septal perforator). '
                         'Afterwards, <b>strict blood-pressure control</b> in the first months while the autograft adapts to systemic pressure, then lifelong surveillance of the autograft and the homograft.</p>'
                         + ev('blood-pressure control after the Ross is recommended in expert consensus (EACTS 2025) to limit early autograft dilatation; its exact targets have not been tested in trials.')}]
        procs['root-ross'] = {'steps': ross, 'appr': 'Ross (pulmonary autograft)',
                              'sq': seq(('Anatomy', 'other'), ('Decide', 'other'), ('Sternotomy', 'other'), ('Cannulate', 'artery'), ('Clamp', 'artery'), ('Excise', 'fissure'), ('Harvest', 'vein'), ('Implant', 'bronchus'), ('Buttons', 'artery'), ('Distal', 'artery'), ('Homograft', 'vein'), ('Wean', 'artery'))}
    for key in [k for k in ('root-bentall', 'root-ross') if k in procs and 'appr' in procs[k]]:
        steps_, appr, sq = procs[key]['steps'], procs[key]['appr'], procs[key]['sq']
        for s in steps_:
            named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
            off_groups = {'arteries', 'veins', 'airway', 'lul-intra', 'lll-intra', 'rul-intra', 'nodes', 'nerves', 'pleura', 'segments', 'trauma', 'muscles', 'landmarks'}
            s['hide'] = [*s.get('hide', []), *[q['id'] for q in atlas['structures'] if q['group'] in off_groups and q['id'] not in named],
                         *[i for i in ('esophagus', 'thymus', 'thyroid') if has(i) and i not in named], *(['sternum'] if s.get('seq', 0) >= 3 else [])]
            if s['phase'] in ('Root', 'Anatomy', 'Pulmonary root'):
                s['hide'] = [*s['hide'], 'svc', 'ra', 'la', 'myocardium', *[i for i in (*AV_CANS, 'can-svc', 'can-ivc', 'can-ostial') if i not in named], *[f'vert-t{i}' for i in range(1, 13)]]
                s['opacity'] = {**s.get('opacity', {}), 'lv': 0.12, 'lvot': 0.2}
            if s['phase'] == 'Root': s['hide'] = [*s['hide'], 'pa-trunk', 'rv']
            s['opacity'] = {**{f'vert-t{i}': 0.25 for i in range(2, 11)}, **s.get('opacity', {})}
            for kk in ('highlight', 'danger', 'labels', 'show'):
                if kk in s: s[kk] = [i for i in s[kk] if has(i) or i == 'skin']
        procs[key] = {'id': key, 'op': 'root', 'opName': 'Aortic root replacement', 'side': 'both', 'name': 'Aortic root replacement', 'approach': appr,
                      'summary': 'Thresholds, choice of root operation, excision with coronary buttons, composite graft or pulmonary autograft, button reimplantation, distal anastomosis, homograft.',
                      'ports': [], 'steps': steps_, 'sources': RTSRC, 'group': 'Cardiac', 'sequence': sq}
# ==================================================================================================== cardiac: coronary artery bypass grafting
CABG_OK = MVR_OK and has('cor-lad') and 'target-lad' in LM
if CABG_OK:
    TL, TO, TP = Lc('target-lad'), Lc('target-om'), Lc('target-pda')
    HC_ = V(np.mean([V(LM['mv-centre']), V(LM['tv-centre'])], 0)) if 'tv-centre' in LM else V(LM['mv-centre'])
    TREE = [i for i in ('cor-lm', 'cor-lad', 'cor-d1', 'cor-d2', 'cor-lcx', 'cor-om1', 'cor-om2', 'cor-rca', 'cor-am', 'cor-pda') if has(i)]
    LES = [i for i in ('lesion-lad', 'lesion-om', 'lesion-rca') if has(i)]
    SURF = [i for i in ('myocardium', 'rv', 'ra', 'la', 'aorta', 'pa-trunk', 'svc', 'laa') if has(i)]
    SOLID = {'myocardium': 0.93, 'rv': 0.9, 'ra': 0.85, 'la': 0.85, 'aorta': 0.9, 'pa-trunk': 0.9, 'laa': 0.9}
    CB_OFF = [i for i in (*HEART_OFF, 'lv', 'circumflex', 'coronaries', 'rca-groove', 'coronary-sinus', 'lvot', 'cusp-r', 'cusp-l', 'cusp-n') if has(i)]
    look = lambda tgt, extra=V([0, 0, 0]), dist=190: clook(tgt, (V(tgt) - HC_) / np.linalg.norm(V(tgt) - HC_) + extra, dist)
    outn = lambda tgt: R((V(tgt) - HC_) / np.linalg.norm(V(tgt) - HC_))
    CBSRC = [
        {'title': 'Lawton JS, Tamis-Holland JE, et al. 2021 ACC/AHA/SCAI guideline for coronary artery revascularization. J Am Coll Cardiol 2022;79:e21-e129', 'url': 'https://pubmed.ncbi.nlm.nih.gov/34882435/'},
        {'title': 'Kouchoukos NT, Blackstone EH, Hanley FL, Kirklin JK. Kirklin/Barratt-Boyes Cardiac Surgery, 4th ed. Elsevier 2013: coronary artery bypass', 'url': pm('Kirklin Barratt-Boyes coronary artery bypass')},
        {'title': 'Farkouh ME, et al. Strategies for multivessel revascularization in patients with diabetes (FREEDOM). N Engl J Med 2012;367:2375-84', 'url': pm('FREEDOM trial Farkouh multivessel revascularization diabetes 2012')},
        {'title': 'Velazquez EJ, et al. Coronary-artery bypass surgery in patients with ischemic cardiomyopathy (STICHES, 10 years). N Engl J Med 2016;374:1511-20', 'url': pm('Velazquez STICHES ischemic cardiomyopathy 2016')},
        {'title': 'Taggart DP, et al. Bilateral versus single internal-thoracic-artery grafts at 10 years (ART). N Engl J Med 2019;380:437-46', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa1808783'},
        {'title': 'Gaudino M, et al. Radial-artery or saphenous-vein grafts in coronary-artery bypass surgery (RADIAL). N Engl J Med 2018;378:2069-77', 'url': 'https://www.acc.org/latest-in-cardiology/journal-scans/2018/04/30/14/44/radial-artery-or-saphenous-vein-grafts-in-cabg'},
        {'title': 'Zenati MA, et al. Randomized trial of endoscopic or open vein-graft harvesting (REGROUP). N Engl J Med 2019', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa1812390'},
        {'title': 'No-touch vein grafts in coronary artery bypass surgery: a registry-based randomized clinical trial (SWEDEGRAFT). Eur Heart J 2025;46:1720', 'url': 'https://academic.oup.com/eurheartj/article/46/18/1720/8023883'},
        {'title': 'Graft patency of no-touch versus conventionally harvested saphenous vein conduits: meta-analysis of 7 randomized trials. 2025', 'url': 'https://www.sciencedirect.com/science/article/pii/S2666273625000555'},
        {'title': 'Maron DJ, et al. Initial invasive or conservative strategy for stable coronary disease (ISCHEMIA). N Engl J Med 2020;382:1395-407', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa1915922'},
        {'title': 'Blazek S, et al. Comparison of bare-metal stenting with minimally invasive bypass surgery for stenosis of the LAD: 10-year follow-up of a randomized trial. JACC Cardiovasc Interv 2013;6:20-6', 'url': 'https://www.jacc.org/doi/10.1016/j.jcin.2012.09.008'},
        {'title': 'Lamy A, et al. Five-year outcomes after off-pump or on-pump coronary-artery bypass grafting (CORONARY). N Engl J Med 2016;375:2359-68', 'url': 'https://www.acc.org/Latest-in-Cardiology/Clinical-Trials/2014/06/08/17/13/CORONARY'},
        {'title': 'Shroyer AL, et al. Five-year outcomes after on-pump and off-pump coronary-artery bypass (ROOBY-FS). N Engl J Med 2017;377:623-32; Quin JA, et al. Ten-year outcomes. JAMA Surg 2022', 'url': 'https://www.tctmd.com/news/rooby-fs-10-year-data-affirm-pump-cabg-default-strategy'},
    ]

    def cb_anat(pre):
        return {'id': f'{pre}-anatomy', 'phase': 'Anatomy', 'seq': 0, 'title': 'The coronary tree and the targets',
                'body': '<p>The <b>left main</b> runs behind the pulmonary trunk and divides into the <b>LAD</b> (in the anterior interventricular groove, giving <b>diagonals</b> over the LV and septal perforators into the septum) and the <b>circumflex</b> (in the left AV groove, giving <b>obtuse marginals</b> over the lateral wall). '
                        'The <b>right coronary</b> runs in the right AV groove, gives the <b>acute marginal</b> over the RV and, in a <b>right-dominant</b> heart (about 85%), the <b>PDA</b> down the posterior groove from the crux.</p>'
                        '<p>This case: <b>three-vessel disease</b>: proximal LAD, OM1 and mid-RCA stenoses. <b>Targets</b> are chosen beyond the disease, on a vessel of good calibre (1.5 mm or more) that is soft to the touch.</p>'
                        + ev('the coronary course here follows the grooves of this heart; only the proximal left system is from the CT segmentation, the rest is schematic. Dominance proportions are from angiographic series.'),
                'view': clook(HC_, V([-0.3, 1, 0.2]), 260), 'spin': True, 'show': [*SURF, *TREE, *LES], 'hide': CB_OFF, 'opacity': SOLID,
                'highlight': TREE[:0], 'danger': LES, 'labels': ['cor-lad', 'cor-d1', 'cor-lcx', 'cor-om1', 'cor-rca', 'cor-pda', 'cor-am', *LES],
                'ask': ask('In a right-dominant circulation, which artery gives the posterior descending artery?', 'The right coronary artery',
                           'Dominance is defined by which artery gives the PDA; the RCA does in about 85% (the circumflex in left dominance).', 'The circumflex', 'The LAD'),
                'ct': ct(R(HC_), 'axial')}

    def cb_decide(pre, offpump=False):
        return {'id': f'{pre}-decide', 'phase': 'Decision', 'seq': 1, 'title': 'Why surgery, which conduits' + (', and why off-pump' if offpump else ''),
                'body': '<p><b>Surgery over PCI</b> (Heart Team, ACC/AHA/SCAI 2021): <b>left main</b> disease (CABG class I; PCI class IIa if anatomy is of low or intermediate complexity); '
                        '<b>diabetes with three-vessel disease</b> (CABG class I); <b>three-vessel disease with a normal EF</b> (CABG class IIb for survival). With <b>ischaemic cardiomyopathy</b>, CABG improves long-term survival (STICHES).</p>'
                        '<p><b>Conduits</b>: the <b>LIMA to the LAD</b> is the foundation. For the second most important target, a <b>radial artery</b> is preferred to vein (ACC/AHA/SCAI class IIa). Saphenous vein for the rest.</p>'
                        + ('<p><b>Off-pump</b> avoids bypass and aortic manipulation; it fits a heavily calcified aorta (a no-touch aorta), and it needs experience with the stabiliser and heart positioning.</p>' if offpump else '')
                        + ev('FREEDOM (NEJM 2012): in diabetes with multivessel disease, CABG reduced death, MI and stroke at 5 years compared with drug-eluting stents. STICHES (NEJM 2016): CABG reduced 10-year death in ischaemic cardiomyopathy (EF 35% or less). '
                             'RADIAL (NEJM 2018, 6 trials, 1,036 patients): radial artery grafts had fewer adverse cardiac events (HR 0.67) and graft occlusions (HR 0.44) than vein at 5 years, with no difference in death. '
                             'ART (NEJM 2019): bilateral ITA grafts did not reduce 10-year death on intention to treat (many crossovers; radial artery used in some single-ITA patients), and sternal wound complications were more frequent.'
                             + (' CORONARY (NEJM 2016, 4,752 patients): off-pump and on-pump had the same 5-year composite (23.1% vs 23.6%). ROOBY-FS (NEJM 2017): off-pump had higher 5-year mortality in VA surgeons\' hands; at 10 years (JAMA Surg 2022) the difference was no longer significant.' if offpump else '')),
                'view': clook(HC_, V([-0.3, 1, 0.3]), 280), 'show': [*SURF, *TREE, *LES], 'hide': CB_OFF, 'opacity': SOLID, 'labels': LES,
                'ask': ask('A 58-year-old with diabetes and three-vessel disease, EF 50%. What do the guidelines recommend?', 'CABG (class I)',
                           'In diabetes with multivessel disease, CABG reduced death, MI and stroke compared with PCI in FREEDOM; the 2021 guideline gives surgery class I.',
                           'PCI with drug-eluting stents', 'Medical therapy alone'),
                'ct': ct(R(HC_), 'axial')}

    def cb_lima(pre, seq_):
        return {'id': f'{pre}-lima', 'phase': 'Conduits', 'seq': seq_, 'title': 'Harvest the LIMA',
                'body': '<p>A retractor lifts the left sternal half. Open the left pleura if needed. Harvest the <b>left internal mammary artery</b> from its origin under the subclavian vein to its bifurcation at the 6th space: as a <b>pedicle</b> (with its veins, fat and muscle) or <b>skeletonised</b> (the artery alone, with low-energy diathermy and clips on each branch).</p>'
                        '<p>Heparin before dividing it distally; check the free flow; spray papaverine to prevent spasm; keep it long enough to reach the LAD without tension.</p>'
                        + ev('skeletonisation gives a longer conduit and preserves sternal blood supply (fewer sternal wound problems with bilateral ITA). In a post hoc analysis of ART, skeletonised grafts were associated with more adverse events in some analyses, so the choice remains debated and experience-dependent.'),
                'view': clook(Lc('clamp-ao') if 'clamp-ao' in LM else HC_, V([-0.6, 1, 0.3]), 330), 'show': [*SURF, 'lima-insitu', *TREE], 'hide': CB_OFF, 'opacity': SOLID,
                'highlight': ['lima-insitu'], 'labels': ['lima-insitu'],
                'action': {'kind': 'reveal', 'label': 'Harvest the LIMA', 'port': 'sternotomy', 'ids': ['lima-insitu']},
                'ct': ct(R(HC_), 'axial')}

    def cb_veins(pre, seq_):
        return {'id': f'{pre}-conduits', 'phase': 'Conduits', 'seq': seq_, 'title': 'Radial artery or long saphenous vein',
                'body': '<p><b>Radial artery</b> (non-dominant arm): check the ulnar collateral circulation first (Allen test, or pulse oximetry or Doppler with the radial compressed). Harvest as a pedicle, avoid spasm (topical vasodilators). Best to a target with a <b>severe</b> stenosis: competitive flow closes it.</p>'
                        '<p><b>Long saphenous vein</b>: open, bridging or endoscopic harvest; "<b>no-touch</b>" harvest keeps a cuff of fat round the vein. Handle it gently, distend it at low pressure, and <b>reverse</b> it (valves).</p>'
                        + ev('REGROUP (NEJM 2019, 1,150 patients): endoscopic and open vein harvest had similar major cardiac events at about 3 years, with fewer leg wound infections endoscopically. '
                             'No-touch harvest: a 2025 meta-analysis of 7 trials (3,334 patients) found fewer vein graft occlusions at about a year (RR 0.57) but more leg wound problems (RR 2.3); SWEDEGRAFT (Eur Heart J 2025, 902 patients) found no reduction in graft failure at 2 years, and more leg complications.'),
                'view': clook(HC_, V([-0.3, 1, 0.3]), 280), 'show': [*SURF, *TREE, *LES, 'lima-insitu'], 'hide': CB_OFF, 'opacity': SOLID, 'labels': LES,
                'ask': ask('The OM1 has only a 60% stenosis. Which conduit is at risk of closing from competitive flow?', 'A radial artery graft',
                           'Arterial grafts, the radial above all, need a severe proximal stenosis; with competitive native flow they string down or occlude. Vein is more forgiving here.',
                           'A saphenous vein graft', 'Neither'),
                'ct': ct(R(HC_), 'axial')}

    def cb_distal(pre, seq_, key, graft, title, body, e, view_extra=V([0, 0, 0]), port='sternotomy', offpump=False):
        tgt = Lc(f'target-{key}')
        return {'id': f'{pre}-{key}', 'phase': 'Graft', 'seq': seq_, 'title': title,
                'body': body + (('<p><b>Off-pump</b>: the stabiliser holds the target still; an intracoronary shunt or a silicone snare proximally keeps the field bloodless; a CO₂ blower clears the view. Warn the anaesthetist before lifting the heart.</p>') if offpump else '') + ev(e),
                'view': look(tgt, view_extra, 170), 'show': [*SURF, *TREE, *LES, graft, f'target-{key}', *(['stabilizer'] if offpump and key == 'lad' else [])], 'hide': CB_OFF, 'opacity': SOLID,
                'highlight': [graft, f'target-{key}'], 'labels': [graft, f'target-{key}'],
                'action': {'kind': 'anastomose', 'label': 'Sew the distal anastomosis', 'port': port, 'at': R(tgt), 'axis': outn(tgt), 'radius': 2.4, 'show': [graft]},
                'ct': ct(R(tgt), 'axial')}

    lad_body = ('<p>Bring the LIMA pedicle down lateral to the pulmonary artery, with no tension or twist and enough length for the heart to fill. On the <b>mid LAD</b> beyond the disease (after the second diagonal here): a 4–5 mm arteriotomy on the vessel\'s anterior surface; '
                'an end-to-side anastomosis with running <b>8-0 polypropylene</b>, heel and toe first; tack the pedicle to the epicardium on each side.</p>')
    lad_ev = 'LIMA to LAD gives the best long-term patency and survival of any graft and is the basis of the class I recommendation to use it; its patency is above 90% at 10 years in large series.'
    om_body = ('<p>Lift the apex (on-pump: with the heart empty; off-pump: an apical suction device and deep pericardial stitches) to expose the <b>lateral wall</b>. On <b>OM1</b> beyond the stenosis: arteriotomy, then the reversed vein end-to-side with running <b>7-0 polypropylene</b>, heel then toe.</p>'
               '<p>Judge the vein\'s length with the heart <b>filled</b>: too long kinks, too short tears.</p>')
    om_ev = 'the order of grafting (lateral and inferior wall first, LIMA to LAD last on-pump so it is not torn when the heart is lifted) is standard practice (Kirklin/Barratt-Boyes).'
    pda_body = ('<p>Lift the heart up and toward the head to expose the <b>inferior wall</b>. The <b>PDA</b> beyond the crux, or the distal RCA before the crux if it is large and soft: the vein end-to-side with running 7-0 polypropylene.</p>')
    pda_ev = 'grafting the PDA rather than a diseased distal RCA avoids the crux, where disease is common; the choice follows the angiogram and palpation, not trials.'

    def cb_cannulate(pre, seq_):
        return {**av_cannulate(pre, seq_), 'title': 'Cannulate (aorta; two-stage venous RA to IVC), arrest', 'show': [*SURF, *TREE, *LES, *[i for i in ('ivc',) if has(i)]], 'hide': CB_OFF, 'opacity': {**SOLID, 'ivc': 0.6},
                'body': '<p>Heparin (about 300–400 U/kg; ACT above 480 s). Arterial cannula in the distal ascending aorta. Venous drainage by one <b>two-stage cannula</b>: through the <b>right atrial appendage</b>, tip in the <b>IVC</b>, side holes in the atrium (the heart is not opened, so separate caval cannulas are not needed). Antegrade cardioplegia and root vent, and a retrograde cannula in the coronary sinus. Palpate or scan the aorta (epiaortic ultrasound) for calcium before cannulating or clamping.</p>'
                        '<p>Cross-clamp and arrest: <b>antegrade and retrograde</b> cardioplegia (retrograde reaches beyond the blocked arteries); after each distal anastomosis, give cardioplegia down the new graft too.</p>'
                        + ev('epiaortic scanning finds atheroma missed by palpation and changes the cannulation or clamp site in some patients; stroke after CABG is mostly embolic from the aorta.')}

    def cb_proximal(pre, seq_, offpump=False):
        ids = [i for i in ('prox-om', 'prox-pda', 'graft-svg-om', 'graft-svg-pda') if has(i)]
        return {'id': f'{pre}-proximal', 'phase': 'Graft', 'seq': seq_, 'title': 'Proximal anastomoses on the aorta',
                'body': ('<p>Either during the single cross-clamp (no second clamp on the aorta), or after release with a <b>side-biting clamp</b>. ' if not offpump else
                         '<p>With a <b>side-biting clamp</b> at low blood pressure (systolic about 90 mmHg), or a <b>clampless</b> sealing device; with a porcelain aorta, no aortic touch at all: take the vein or radial off the LIMA (a composite "Y" or "T" graft). ')
                        + 'A 4–5 mm aortotomy with a punch; the vein end-to-side with running 6-0 polypropylene; lay out the grafts so they neither kink nor stretch when the heart fills.</p>'
                        + ev('partial aortic clamping is a source of emboli; single-clamp and clampless techniques aim to reduce stroke; the comparative evidence is observational.'),
                'view': clook(Lc('prox-om') if 'prox-om' in LM else HC_, V([0.3, 1, 0.3]), 230), 'show': [*SURF, *TREE, 'graft-lima', *ids], 'hide': CB_OFF, 'opacity': SOLID,
                'highlight': [i for i in ids if i.startswith('prox')], 'labels': ids,
                'action': {'kind': 'reveal', 'label': 'Sew the proximal anastomoses', 'port': 'sternotomy', 'ids': ids},
                'ct': ct(R(Lc('prox-om') if 'prox-om' in LM else HC_), 'axial')}

    def cb_flow(pre, seq_, offpump=False):
        ids = [i for i in ('can-cp', 'can-2stage', 'can-aortic') if has(i)] if not offpump else []
        return {'id': f'{pre}-flow', 'phase': 'Wean', 'seq': seq_, 'title': 'Check every graft, then close',
                'body': '<p>' + ('De-air, release the clamp, reperfuse and wean. ' if not offpump else 'Reverse heparin with protamine once all grafts are checked. ') +
                        '<b>Transit-time flow measurement</b> on each graft: mean flow, <b>pulsatility index</b> and diastolic filling. A low flow with a high PI means a technical problem until proved otherwise: look, and redo it now. TOE for new wall-motion abnormalities.</p>'
                        '<p>Before closing: the LIMA pedicle lies without tension; the veins do not kink when the lungs inflate; a drain in each opened pleura.</p>'
                        + ev('the commonly used TTFM thresholds (for example PI below 5 and a mean flow above about 15–20 mL/min for a left-sided graft) come from observational series; in them, abnormal readings predict early graft failure, and correcting grafts intraoperatively is associated with better outcomes.'),
                'view': clook(HC_, V([-0.3, 1, 0.3]), 280), 'show': [*SURF, *TREE, 'graft-lima', 'graft-svg-om', 'graft-svg-pda', 'prox-om', 'prox-pda', *ids], 'hide': CB_OFF, 'opacity': SOLID,
                'labels': ['graft-lima', 'graft-svg-om', 'graft-svg-pda'],
                **({'action': {'kind': 'decannulate', 'label': 'Wean and decannulate', 'port': 'sternotomy', 'ids': ids}} if ids else {}),
                'ask': ask('After the LIMA–LAD, TTFM shows mean flow 6 mL/min and PI 9. Next?', 'Inspect the anastomosis and the pedicle; revise the graft now',
                           'Low flow with a high PI points to a technical problem (kink, twist, anastomotic narrowing). Fixing it before leaving theatre is far safer than finding it after an infarct.',
                           'Accept it: flows improve after weaning', 'Add a vein graft to the LAD later if needed'),
                'ct': ct(R(HC_), 'axial')}

    def cb_case(pre, title, vignette, lesions, quiz):
        return {'id': f'{pre}-case', 'phase': 'Case', 'seq': 0, 'title': title,
                'body': vignette, 'view': clook(HC_, V([-0.3, 1, 0.2]), 260), 'spin': True, 'show': [*SURF, *TREE, *lesions], 'hide': CB_OFF, 'opacity': SOLID,
                'danger': lesions, 'labels': lesions, 'ask': quiz, 'ct': ct(R(HC_), 'axial')}

    # ----------------------------------------------------------------------------- single-vessel: isolated proximal LAD, MIDCAB
    one = [cb_case('c1', 'Case: isolated proximal LAD disease',
                   '<p>A 62-year-old man, <b>angina (CCS III) despite full medical therapy</b>; stress imaging shows a large anterior ischaemic area. Angiogram: a long, calcified <b>90% ostial–proximal LAD</b> stenosis involving the diagonal origin; circumflex and RCA normal; EF 55%. '
                   'The interventional team judge it unfavourable for PCI.</p>'
                   + ev('in stable coronary disease, revascularisation relieves angina, but ISCHEMIA (NEJM 2020, over 5,000 patients with moderate or severe ischaemia) found no reduction in death or MI with an initial invasive strategy. For isolated proximal LAD disease, the randomised Leipzig trial (Blazek et al., JACC Cardiovasc Interv 2013; 220 patients, 10 years) found MIDCAB and stenting similar for death and MI, with far fewer repeat revascularisations after MIDCAB (11% vs 34%).'),
                   ['lesion-lad'],
                   ask('What is the main aim of revascularising this man?', 'Relieving angina that persists despite medical therapy',
                       'In stable single-vessel disease, trials have not shown a survival benefit; the indication is symptoms despite optimal medical therapy (and here an anatomy unfavourable for PCI).',
                       'Improving survival', 'Preventing a future myocardial infarction')),
           {'id': 'c1-access', 'phase': 'Access', 'seq': 1, 'title': 'MIDCAB: left anterior mini-thoracotomy',
            'body': '<p>Double-lumen tube, <b>left lung deflated</b>, the left chest raised about 30°, external defibrillator pads. A 6–8 cm incision in the <b>left 4th (or 5th) space</b> from near the sternal edge laterally, positioned over the LAD target (check on the angiogram or CT).</p>'
                    '<p>No sternotomy, no bypass: the operation is <b>off-pump</b>, and the only graft is the <b>LIMA to the LAD</b>.</p>'
                    + ev('MIDCAB avoids sternotomy and cardiopulmonary bypass; its results depend on centre experience, and conversion to sternotomy must always be possible.'),
            'view': clook(Lc('midcab') if 'midcab' in LM else HC_, V([-0.45, 1, 0.3]), 330), 'show': ['skin', 'incision-midcab'], 'opacity': {'skin': 1.0}, 'hide': CB_OFF,
            'highlight': ['incision-midcab'], 'labels': ['incision-midcab'], 'ct': ct(R(Lc('midcab') if 'midcab' in LM else HC_), 'axial', 'lung')},
           {**cb_lima('c1', 2), 'body': '<p>A special retractor lifts the upper ribs. The <b>LIMA</b> is harvested under direct vision (or thoracoscopically) from as high as possible down to the incision, pedicled or skeletonised, with clips on each branch. Heparin before dividing it.</p>'
                    + ev('a long LIMA is essential here: the target must be reached without tension through a small incision, and the pedicle cannot be brought round other structures as at sternotomy.')},
           {'id': 'c1-stabilise', 'phase': 'Graft', 'seq': 3, 'title': 'Open the pericardium over the LAD; stabilise',
            'body': '<p>Open the pericardium over the LAD, stitch its edges up to lift the heart. Find the target beyond the lesion (and beyond the diagonal); place the <b>stabiliser</b>, a proximal silicone snare or an intracoronary shunt, and a CO₂ blower.</p>'
                    + ev('ischaemic preconditioning (a brief trial occlusion) before the arteriotomy is practised by some surgeons; its benefit is not proven in trials.'),
            'view': look(TL, V([0.35, 1.3, 0.35]), 190), 'show': [*SURF, *TREE, 'lesion-lad', 'target-lad', 'stabilizer', 'lima-insitu'], 'hide': CB_OFF, 'opacity': SOLID,
            'highlight': ['stabilizer', 'target-lad'], 'labels': ['stabilizer', 'target-lad'],
            'action': {'kind': 'reveal', 'label': 'Place the stabiliser', 'port': 'sternotomy', 'ids': ['stabilizer']}, 'ct': ct(R(TL), 'axial')},
           {**cb_distal('c1', 4, 'lad', 'graft-lima', 'LIMA to LAD, off-pump', lad_body, lad_ev, V([0.35, 1.3, 0.35]), offpump=True), 'show': [*SURF, *TREE, 'lesion-lad', 'graft-lima', 'target-lad', 'stabilizer']},
           {'id': 'c1-check', 'phase': 'Wean', 'seq': 5, 'title': 'Check the graft, close',
            'body': '<p><b>Transit-time flow</b> on the LIMA (a good mean flow and a PI below about 5), TOE for anterior wall motion. Protamine. A chest drain, the lung re-expanded, the ribs approximated, a local anaesthetic block for pain.</p>'
                    + ev('TTFM thresholds come from observational series; a poor reading on the only graft calls for revision before closing.'),
            'view': look(TL, V([0.35, 1.3, 0.35]), 220), 'show': [*SURF, *TREE, 'graft-lima', 'target-lad'], 'hide': CB_OFF, 'opacity': SOLID, 'labels': ['graft-lima'],
            'ask': ask('The LIMA reads mean flow 5 mL/min, PI 12. What next?', 'Look for a kink, twist or anastomotic problem and revise the graft now',
                       'On a single graft, a technical failure means an anterior infarct; fix it before closing.', 'Accept: flow improves later', 'Give nitrates and close'),
            'ct': ct(R(TL), 'axial')}]

    # ----------------------------------------------------------------------------- two-vessel: LAD and OM, diabetic; LIMA + radial
    two = [cb_case('c2', 'Case: two-vessel disease (LAD and OM) with diabetes',
                   '<p>A 55-year-old woman with <b>type 2 diabetes</b>, exertional angina. Angiogram: <b>90% proximal LAD</b> and <b>85% OM1</b>; the RCA is normal; EF 50%. Good radial pulses; a normal ulnar collateral test on the left (non-dominant) arm.</p>'
                   + ev('FREEDOM (NEJM 2012) randomised patients with diabetes and multivessel disease (two or more vessels in at least two territories) to CABG or drug-eluting stents: CABG reduced the 5-year rate of death, MI or stroke, with more strokes after surgery.'),
                   ['lesion-lad', 'lesion-om'],
                   ask('For this diabetic patient with two-vessel disease involving the proximal LAD, which strategy has trial evidence of fewer deaths and MIs?', 'CABG',
                       'FREEDOM included diabetics with two- or three-vessel disease; CABG reduced death and MI at 5 years compared with drug-eluting stents (with more strokes).',
                       'PCI with drug-eluting stents', 'Medical therapy only')),
           {**cb_decide('c2'), 'seq': 1},
           {**mv_sternotomy('c2', 2), 'id': 'c2-sternotomy'}, cb_lima('c2', 3),
           {**cb_veins('c2', 4), 'title': 'Harvest the radial artery (for the OM)',
            'body': '<p><b>Radial artery</b> from the <b>non-dominant arm</b>, after confirming ulnar collateral flow (modified Allen test, oximetry or Doppler with the radial compressed). Harvest it as a pedicle with its veins, from the elbow crease to the wrist, avoiding the lateral antebrachial cutaneous and superficial radial nerves. '
                    'Topical and systemic vasodilators against spasm. Use it only for a target with a <b>severe</b> stenosis (here OM1 85%): competitive flow closes it.</p>'
                    + ev('RADIAL (NEJM 2018; 6 trials, 1,036 patients): radial artery grafts had fewer adverse cardiac events (HR 0.67) and occlusions (HR 0.44) than saphenous vein at about 5 years; the 2021 ACC/AHA/SCAI guideline prefers the radial to vein for the second most important target (class IIa).')},
           cb_cannulate('c2', 5),
           {**cb_distal('c2', 6, 'om', 'graft-radial-om', 'Distal: radial artery to OM1', om_body.replace('the reversed vein end-to-side with running <b>7-0 polypropylene</b>', 'the radial artery end-to-side with running <b>8-0 polypropylene</b>'), 'arterial grafts are sewn with finer suture; the radial must not be stretched or twisted, and its spasm is prevented with vasodilators.', V([0, 0, -0.3])),
            'show': [*SURF, *TREE, 'lesion-lad', 'lesion-om', 'graft-radial-om', 'target-om']},
           {**cb_distal('c2', 7, 'lad', 'graft-lima', 'Distal: LIMA to LAD', lad_body, lad_ev, V([0.35, 1.3, 0.35])), 'show': [*SURF, *TREE, 'lesion-lad', 'lesion-om', 'graft-lima', 'graft-radial-om', 'target-lad']},
           {**cb_proximal('c2', 8), 'show': [*SURF, *TREE, 'graft-lima', 'graft-radial-om', 'prox-om'], 'highlight': ['prox-om'], 'labels': ['prox-om', 'graft-radial-om'],
            'action': {'kind': 'reveal', 'label': 'Sew the proximal anastomosis', 'port': 'sternotomy', 'ids': ['prox-om']}},
           {**cb_flow('c2', 9), 'show': [*SURF, *TREE, 'graft-lima', 'graft-radial-om', 'prox-om', *[i for i in ('can-cp', 'can-2stage', 'can-aortic') if has(i)]], 'labels': ['graft-lima', 'graft-radial-om']}]

    # ----------------------------------------------------------------------------- three-vessel: on-pump and off-pump
    case3 = lambda pre: cb_case(pre, 'Case: three-vessel disease with impaired LV',
                                '<p>A 64-year-old man, diabetic, breathless and with angina; <b>EF 30%</b> with viable myocardium on imaging. Angiogram: <b>proximal LAD</b>, <b>OM1</b> and <b>mid-RCA</b> severe stenoses (a right-dominant system).</p>'
                                + ev('STICHES (NEJM 2016): in ischaemic cardiomyopathy with EF 35% or less, CABG plus medical therapy reduced death from any cause at 10 years compared with medical therapy alone (59% vs 66%).'),
                                ['lesion-lad', 'lesion-om', 'lesion-rca'],
                                ask('EF 30%, three-vessel disease, suitable targets. Compared with medical therapy alone, CABG…', 'reduces death at 10 years (STICHES)',
                                    'STICH, extended to 10 years (STICHES), showed lower all-cause and cardiovascular mortality with CABG in ischaemic cardiomyopathy.',
                                    'has no effect on survival', 'is contraindicated below EF 35%'))
    onp = [case3('co'), {**cb_decide('co'), 'seq': 1}, {**mv_sternotomy('co', 2), 'id': 'co-sternotomy'}, cb_lima('co', 3), cb_veins('co', 4), cb_cannulate('co', 5),
           cb_distal('co', 6, 'om', 'graft-svg-om', 'Distal: vein to OM1', om_body, om_ev, V([0, 0, -0.3])),
           cb_distal('co', 7, 'pda', 'graft-svg-pda', 'Distal: vein to the PDA', pda_body, pda_ev, V([0, 0, -0.5])),
           cb_distal('co', 8, 'lad', 'graft-lima', 'Distal: LIMA to LAD (last, on-pump)', lad_body, lad_ev, V([0.35, 1.3, 0.35])),
           cb_proximal('co', 9), cb_flow('co', 10)]
    offp = [case3('cf'), {**cb_decide('cf', offpump=True), 'seq': 1}, {**mv_sternotomy('cf', 2), 'id': 'cf-sternotomy'}, cb_lima('cf', 3), cb_veins('cf', 4),
            {'id': 'cf-position', 'phase': 'Bypass', 'seq': 5, 'title': 'Off-pump: heparin, positioning, stabiliser',
             'body': '<p>Heparin (a lower dose than for bypass, often about 150–200 U/kg, per unit practice). Keep the patient <b>warm</b>, the blood pressure up and the heart filled. <b>Deep pericardial stitches</b> (and head-down tilt) lift and rotate the heart without compressing it; an <b>apical suction</b> device helps for the lateral and inferior walls.</p>'
                     '<p>Graft the <b>LAD first</b> with the LIMA: the anterior wall is then reperfused before the heart is lifted for the other targets. Keep a perfusionist and pump ready: convert if the heart will not tolerate positioning.</p>'
                     + ev('conversion from off-pump to on-pump during surgery is associated with worse outcomes in registry data, which is why haemodynamic instability should prompt an early, controlled conversion rather than a late, emergency one.'),
             'view': look(TL, V([0.35, 1.3, 0.35]), 220), 'show': [*SURF, *TREE, *LES, 'stabilizer'], 'hide': CB_OFF, 'opacity': SOLID, 'highlight': ['stabilizer'], 'labels': ['stabilizer'],
             'action': {'kind': 'reveal', 'label': 'Place the stabiliser', 'port': 'sternotomy', 'ids': ['stabilizer']}, 'ct': ct(R(TL), 'axial')},
            cb_distal('cf', 6, 'lad', 'graft-lima', 'Distal: LIMA to LAD (first, off-pump)', lad_body, lad_ev, V([0.35, 1.3, 0.35]), offpump=True),
            cb_distal('cf', 7, 'om', 'graft-svg-om', 'Distal: vein to OM1', om_body, om_ev, V([0, 0, -0.3]), offpump=True),
            cb_distal('cf', 8, 'pda', 'graft-svg-pda', 'Distal: vein to the PDA', pda_body, pda_ev, V([0, 0, -0.5]), offpump=True),
            cb_proximal('cf', 9, offpump=True), cb_flow('cf', 10, offpump=True)]
    SCEN = {'cabg-1v': ['lesion-lad'], 'cabg-2v': ['lesion-lad', 'lesion-om'], 'cabg-onpump': LES, 'cabg-offpump': LES}
    for key, appr, steps_, sq in (
            ('cabg-1v', 'Single-vessel: LAD by MIDCAB (off-pump)', one, seq(('Case', 'other'), ('Access', 'other'), ('LIMA', 'artery'), ('Stabilise', 'other'), ('LIMA-LAD', 'artery'), ('Check', 'other'))),
            ('cabg-2v', 'Two-vessel: LAD + OM (LIMA + radial)', two, seq(('Case', 'other'), ('Decide', 'other'), ('Sternotomy', 'other'), ('LIMA', 'artery'), ('Radial', 'artery'), ('Arrest', 'artery'), ('Radial-OM', 'artery'), ('LIMA-LAD', 'artery'), ('Proximal', 'artery'), ('Check', 'other'))),
            ('cabg-onpump', 'Three-vessel, on-pump', onp, seq(('Case', 'other'), ('Decide', 'other'), ('Sternotomy', 'other'), ('LIMA', 'artery'), ('Conduits', 'vein'), ('Arrest', 'artery'), ('OM', 'vein'), ('PDA', 'vein'), ('LIMA-LAD', 'artery'), ('Proximals', 'artery'), ('Check', 'other'))),
            ('cabg-offpump', 'Three-vessel, off-pump (OPCAB)', offp, seq(('Case', 'other'), ('Decide', 'other'), ('Sternotomy', 'other'), ('LIMA', 'artery'), ('Conduits', 'vein'), ('Position', 'other'), ('LIMA-LAD', 'artery'), ('OM', 'vein'), ('PDA', 'vein'), ('Proximals', 'artery'), ('Check', 'other')))):
        keep_les = set(SCEN[key])
        for s in steps_:
            for kk in ('show', 'danger', 'labels', 'highlight'):
                if kk in s: s[kk] = [i for i in s[kk] if not (i.startswith('lesion-') and i not in keep_les)]
            named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
            off_groups = {'arteries', 'veins', 'airway', 'lul-intra', 'lll-intra', 'rul-intra', 'nodes', 'nerves', 'pleura', 'segments', 'trauma', 'muscles', 'landmarks'}
            s['hide'] = [*s.get('hide', []), *[q['id'] for q in atlas['structures'] if q['group'] in off_groups and q['id'] not in named],
                         *[i for i in ('esophagus', 'thymus', 'thyroid') if has(i) and i not in named], *(['sternum'] if s.get('seq', 0) >= 3 else []),
                         *[i for i in ('lesion-lad', 'lesion-om', 'lesion-rca') if i not in keep_les]]
            s['opacity'] = {**{f'vert-t{i}': 0.25 for i in range(2, 11)}, **s.get('opacity', {})}
            for kk in ('highlight', 'danger', 'labels', 'show'):
                if kk in s: s[kk] = [i for i in s[kk] if has(i) or i == 'skin']
        procs[key] = {'id': key, 'op': 'cabg', 'opName': 'Coronary artery bypass grafting', 'side': 'both', 'name': 'Coronary artery bypass grafting', 'approach': appr,
                      'summary': 'Case-based: single-vessel MIDCAB, two-vessel LIMA + radial, three-vessel on- and off-pump; conduits, anastomoses, flow checks.',
                      'ports': [], 'steps': steps_, 'sources': CBSRC, 'group': 'Cardiac', 'sequence': sq}
# ==================================================================================================== the operative field in open-heart steps
# after the chest is open: the drapes and their sternotomy window, the split sternum held open, the pericardial cradle
FIELD = [i for i in ('drape-sternotomy', 'pericardium-open') if has(i)]
if FIELD:
    for key, v in procs.items():
        if v.get('group') != 'Cardiac' or key in ('mvr-mics', 'tv-mics', 'avr-ramt', 'cabg-1v'): continue
        for s_ in v['steps']:
            if s_.get('seq', 0) < 3 or s_['phase'] in ('Decision', 'Anatomy'): continue
            s_['hide'] = [i for i in s_.get('hide', []) if i not in ('sternum', *FIELD)]
            inside = s_['phase'] in ('Valve', 'Root', 'Septum', 'Pulmonary root')   # looking inside the heart: the pericardium would show through
            s_['show'] = [*s_.get('show', []), 'sternum', *[i for i in FIELD if not (inside and i == 'pericardium-open')]]
            if inside: s_['hide'] = [*s_['hide'], 'pericardium-open']
            else: s_['opacity'] = {**s_.get('opacity', {}), 'pericardium-open': 0.55}
# operations appear in the menu in this order
ORDER = ['position', 'thoracotomy-l', 'thoracotomy-r', 'vats-ports-l', 'vats-ports-r', 'lul', 'lll', 'rul', 'rml', 'rll', 'pnl', 'pnr', 'seg-lingula', 'seg-lul-updiv', 'seg-s6', 'trachea', 'thymectomy', 'oesophagectomy', 'duct', 'empyema', 'rt', 'clamshell', 'cardio', 'tract', 'hilar', 'mvr', 'avr', 'root', 'tricuspid', 'cabg']
procs = dict(sorted(procs.items(), key=lambda kv: (ORDER.index(kv[1]['op']), list(procs).index(kv[0]))))
for v in procs.values():
    v['group'] = v.get('group') or ('Pneumonectomy' if v['op'].startswith('pn') else 'Segmentectomy' if v['op'].startswith('seg-') else 'Lobectomy')
# every VATS or open setup step: the patient on the side, the surface lines drawn
for v in procs.values():
    if v['group'] in ('Trauma', 'Access and positioning', 'Mediastinum', 'Oesophagus', 'Pleura'): continue
    k = v['side'][0]
    for s in v['steps']:
        if s['phase'] == 'Setup':
            s['pose'] = 'lateral'
            s['show'] = [*s.get('show', []), *[i for i in (f'line-aal-{k}', f'line-mal-{k}', f'line-pal-{k}', f'lm-scaptip-{k}') if has(i)]]
            s['opacity'] = {**s.get('opacity', {}), 'skin': 1.0}
(OUT / 'procedures.json').write_text(json.dumps(procs, indent=1))
print({k: len(v['steps']) for k, v in procs.items()})
