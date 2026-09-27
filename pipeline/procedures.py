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
    {'title': 'Wasserthal J, et al. TotalSegmentator. Radiol Artif Intell 2023', 'url': 'https://doi.org/10.1148/ryai.230024'},
]
# every operative step: the intrapulmonary trees and the spine recede so the hilar structures read clearly
BASE = {'lul-arteries': 0.16, 'lul-veins': 0.16, 'lul-bronchi': 0.2, **{f'vert-t{i}': 0.22 for i in range(2, 11)}}
OPEN = {'lul': open_version(posterior, 'lo', 'left'), 'lll': open_version(lll_fissure_first, 'llo', 'left')}
if RUL_OK:
    OPEN.update({'rul': open_version(rul_posterior, 'ro', 'right'), 'rll': open_version(rll_fissure_first, 'rlo', 'right'), 'rml': open_version(rml_fissure_first, 'mo', 'right')})
for steps in (anterior, posterior, lll_fissure_first, lll_hilum_first, *OPEN.values(), *((rul_anterior, rul_posterior, rll_fissure_first, rll_hilum_first, rml_anterior, rml_fissure_first) if RUL_OK else ())):
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
# open thoracotomy, one per operation, following that operation's fissure-first sequence
for op_, steps_ in OPEN.items():
    base = next(v for v in procs.values() if v['op'] == op_ and v['approach'] in ('Posterior approach', 'Fissure first'))
    procs[f'{op_}-open'] = {**base, 'id': f'open-{op_}', 'approach': 'Open thoracotomy', 'name': base['opName'] + ', open', 'steps': steps_,
                            'summary': 'Posterolateral thoracotomy, then ' + base['summary'][0].lower() + base['summary'][1:]}
# operations appear in the menu in this order
ORDER = ['lul', 'lll', 'rul', 'rml', 'rll']
procs = dict(sorted(procs.items(), key=lambda kv: (ORDER.index(kv[1]['op']), list(procs).index(kv[0]))))
(OUT / 'procedures.json').write_text(json.dumps(procs, indent=1))
print({k: len(v['steps']) for k, v in procs.items()})
