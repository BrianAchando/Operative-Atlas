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

sources = [
    {'title': 'Hansen HJ, Petersen RH. Video-assisted thoracoscopic lobectomy using a standardized three-port anterior approach: the Copenhagen experience. Ann Cardiothorac Surg 2012;1(1):70-76', 'url': 'https://doi.org/10.3978/j.issn.2225-319X.2012.04.15'},
    {'title': 'McElnay P, Casali G, Batchelor T, West D. Adopting a standardized anterior approach significantly increases VATS lobectomy rates. Eur J Cardiothorac Surg 2014;46(1):100', 'url': 'https://academic.oup.com/ejcts/article/46/1/100/394433'},
    {'title': 'Rusch VW, et al. The IASLC lung cancer staging project: a proposal for a new international lymph node map. J Thorac Oncol 2009', 'url': 'https://pubmed.ncbi.nlm.nih.gov/19357537'},
    {'title': 'Wasserthal J, et al. TotalSegmentator. Radiol Artif Intell 2023', 'url': 'https://doi.org/10.1148/ryai.230024'},
]
# every operative step: the intrapulmonary trees and the spine recede so the hilar structures read clearly
BASE = {'lul-arteries': 0.16, 'lul-veins': 0.16, 'lul-bronchi': 0.2, **{f'vert-t{i}': 0.22 for i in range(2, 11)}}
for steps in (anterior, posterior, lll_fissure_first, lll_hilum_first):
    for s in steps:
        if s['phase'] != 'Setup':
            s['opacity'] = {**BASE, **s.get('opacity', {})}
            # clean view: only what the step is about. Spine, nerves and intrapulmonary trees stay hidden unless named.
            named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', []))
            quiet = [f'vert-t{i}' for i in range(2, 11)] + [i for i in ('n-phrenic', 'n-vagus', 'n-rln', 'lig-art', 'esophagus', 'svc', 'lbcv', 'ipl') if i not in named]
            quiet += ['lul-arteries', 'lul-veins', 'lul-bronchi']
            s['hide'] = [i for i in quiet if has(i)]
        for k in ('ct', 'retract'):
            if s.get(k) is None: s.pop(k, None)
        for k in ('highlight', 'danger', 'labels', 'show'):
            if k in s: s[k] = [i for i in s[k] if has(i)]
        if 'action' in s and 'ids' in s['action']: s['action']['ids'] = [i for i in s['action']['ids'] if has(i)]
procs = {
    'lul-anterior': {'id': 'vats-lul-anterior', 'op': 'lul', 'opName': 'Left upper lobectomy', 'name': 'VATS left upper lobectomy', 'approach': 'Anterior approach', 'summary': 'Hilum first: vein, truncus, bronchus, remaining arteries, fissure last.',
                 'sequence': [{'label': 'Anatomy', 'kind': 'other'}, {'label': 'Superior vein', 'kind': 'vein'}, {'label': 'Truncus', 'kind': 'artery'}, {'label': 'Bronchus', 'kind': 'bronchus'},
                              {'label': 'A2 + lingular', 'kind': 'artery'}, {'label': 'Fissure', 'kind': 'fissure'}, {'label': 'Specimen', 'kind': 'other'}],
                 'ports': [], 'steps': anterior, 'sources': sources},
    'lul-posterior': {'id': 'vats-lul-posterior', 'op': 'lul', 'opName': 'Left upper lobectomy', 'name': 'VATS left upper lobectomy', 'approach': 'Posterior approach', 'summary': 'Fissure first: arteries in the fissure, truncus, bronchus, vein last.',
                  'sequence': [{'label': 'Anatomy', 'kind': 'other'}, {'label': 'Fissure', 'kind': 'fissure'}, {'label': 'A2 + lingular', 'kind': 'artery'}, {'label': 'Truncus', 'kind': 'artery'},
                               {'label': 'Bronchus', 'kind': 'bronchus'}, {'label': 'Vein', 'kind': 'vein'}, {'label': 'Specimen', 'kind': 'other'}],
                  'ports': [], 'steps': posterior, 'sources': sources},
}
procs['lll-fissure'] = {'id': 'vats-lll-fissure', 'op': 'lll', 'opName': 'Left lower lobectomy', 'name': 'VATS left lower lobectomy', 'approach': 'Fissure first',
                        'summary': 'Ligament, fissure, A6 and basal trunk, inferior vein, bronchus.',
                        'sequence': [{'label': 'Anatomy', 'kind': 'other'}, {'label': 'Ligament', 'kind': 'other'}, {'label': 'Fissure', 'kind': 'fissure'}, {'label': 'A6 + basal', 'kind': 'artery'},
                                     {'label': 'Inferior vein', 'kind': 'vein'}, {'label': 'Bronchus', 'kind': 'bronchus'}, {'label': 'Specimen', 'kind': 'other'}],
                        'ports': [], 'steps': lll_fissure_first, 'sources': sources}
procs['lll-hilum'] = {'id': 'vats-lll-hilum', 'op': 'lll', 'opName': 'Left lower lobectomy', 'name': 'VATS left lower lobectomy', 'approach': 'Hilum first (fissureless)',
                      'summary': 'Ligament, inferior vein, bronchus, A6 and basal trunk, fissure last.',
                      'sequence': [{'label': 'Anatomy', 'kind': 'other'}, {'label': 'Ligament', 'kind': 'other'}, {'label': 'Inferior vein', 'kind': 'vein'}, {'label': 'Bronchus', 'kind': 'bronchus'},
                                   {'label': 'A6 + basal', 'kind': 'artery'}, {'label': 'Fissure', 'kind': 'fissure'}, {'label': 'Specimen', 'kind': 'other'}],
                      'ports': [], 'steps': lll_hilum_first, 'sources': sources}
(OUT / 'procedures.json').write_text(json.dumps(procs, indent=1))
print({k: len(v['steps']) for k, v in procs.items()})
