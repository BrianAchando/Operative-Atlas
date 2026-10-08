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
                               'body': '<p><b>Anastomotic leak</b> and <b>conduit necrosis</b> (fever, new AF, turbid or salivary drain, rising CRP: CT with oral contrast first, then endoscopy; see <a class="link" href="#approach=eso-leak&amp;step=0">Esophagectomy: anastomotic leak</a>). <b>Chylothorax</b> (milky drain output once fed). '
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
    HEART_OFF = [i for i in ('heart', 'laa', 'lul', 'lll', 'rul', 'rml', 'rll', 'fissure', 'fissure-h', 'fissure-r', 'thymus', 'azygos', 'hemiazygos') if has(i)]
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
    # ------------------------------------------------------------------------------------------ case-based MVR
    # each approach is a patient: pathophysiology, anatomy, the case (vignette and decision), then the operation
    ev = lambda t: f'<p class="evidence"><b>Evidence:</b> {t}</p>'
    pm = lambda term: 'https://pubmed.ncbi.nlm.nih.gov/?term=' + term.replace(' ', '+')
    PROVEN, INFERRED = '<span class="grade proven">shown in human rheumatic tissue</span>', '<span class="grade inferred">inferred</span>'
    chain = lambda *xs: '<div class="chain">' + '<i>→</i>'.join(f'<span class="hot">{x[1:]}</span>' if x.startswith('!') else f'<span>{x}</span>' for x in xs) + '</div>'
    MV_SIDE = clook(MC - MN * 6, MN * 0.35 + V(np.cross(MN, MU)) * 0.95 + MU * 0.15 + V([0, 0.3, 0]), 135)
    PATHO_OP = {'myocardium': 0.0, 'la': 0.12, 'lv': 0.12, 'ra': 0.1, 'rv': 0.14, 'aorta': 0.14, 'pa-trunk': 0.3, 'laa': 0.3}
    PATHO_HIDE = ['myocardium', 'svc', *CANS]
    CHP = [i for i in CH if i not in ('svc', 'myocardium')]
    RH_MV = [i for i in ('rh-mv', 'rh-mv-edge', 'rh-mv-calcium', 'rh-chordae', 'papillary', 'mitral-annulus') if has(i)]
    NORMAL_MV = ['mv-ant-leaflet', 'mv-post-leaflet', 'chordae']
    RHD_SRC = [
        {'title': 'Carapetis JR, Beaton A, Cunningham MW, et al. Acute rheumatic fever and rheumatic heart disease. Nat Rev Dis Primers 2016;2:15084', 'url': 'https://www.nature.com/articles/nrdp201584'},
        {'title': 'Marijon E, Mirabel M, Celermajer DS, Jouven X. Rheumatic heart disease. Lancet 2012;379:953-64', 'url': pm('Marijon Mirabel Celermajer Jouven rheumatic heart disease Lancet 2012')},
        {'title': 'Kumar RK, Antunes MJ, Beaton A, et al. Contemporary diagnosis and management of rheumatic heart disease: implications for closing the gap. AHA scientific statement. Circulation 2020;142:e337-e357', 'url': 'https://www.ahajournals.org/doi/10.1161/CIR.0000000000000921'},
        {'title': 'Cunningham MW. Pathogenesis of group A streptococcal infections. Clin Microbiol Rev 2000;13:470-511', 'url': pm('Cunningham Pathogenesis of group A streptococcal infections Clin Microbiol Rev 2000')},
        {'title': 'Guilherme L, et al. Human heart-infiltrating T-cell clones from rheumatic heart disease patients recognize both streptococcal and cardiac proteins. Circulation 1995;92:415-20', 'url': pm('Guilherme heart-infiltrating T-cell clones rheumatic heart disease Circulation 1995')},
        {'title': 'Chandrashekhar Y, Westaby S, Narula J. Mitral stenosis. Lancet 2009;374:1271-83', 'url': 'https://pubmed.ncbi.nlm.nih.gov/19747723/'},
        {'title': 'Zühlke L, et al. Clinical outcomes in 3343 children and adults with rheumatic heart disease from 14 low- and middle-income countries: two-year follow-up of the Global RHD Registry (REMEDY). Circulation 2016;134:1456-66', 'url': 'https://www.ahajournals.org/doi/10.1161/circulationaha.116.024769'},
        {'title': 'Beaton A, et al. Secondary antibiotic prophylaxis for latent rheumatic heart disease (GOAL). N Engl J Med 2022;386:230-40', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa2102074'},
        {'title': 'Connolly SJ, Karthikeyan G, Ntsekhe M, et al. Rivaroxaban in rheumatic heart disease-associated atrial fibrillation (INVICTUS). N Engl J Med 2022;387:978-88', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa2209051'},
        {'title': 'Wilkins GT, Weyman AE, Abascal VM, Block PC, Palacios IF. Percutaneous balloon dilatation of the mitral valve: an analysis of echocardiographic variables related to outcome. Br Heart J 1988;60:299-308', 'url': pm('Wilkins Weyman Abascal Block Palacios percutaneous balloon dilatation mitral valve 1988')},
        {'title': 'Nunes MCP, et al. The echo score revisited: impact of incorporating commissural morphology and leaflet displacement. Circulation 2014;129:886-95', 'url': 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.113.001252'},
        {'title': 'Whitlock RP, et al. Left atrial appendage occlusion during cardiac surgery to prevent stroke (LAAOS III). N Engl J Med 2021;384:2081-91', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa2101897'},
        {'title': 'Van Gelder IC, et al. 2024 ESC Guidelines for the management of atrial fibrillation (with EACTS). Eur Heart J 2024', 'url': 'https://pubmed.ncbi.nlm.nih.gov/39210723/'},
        {'title': 'Jiang Y, et al. Clinical outcomes following surgical mitral valve repair or replacement in patients with rheumatic heart disease: a meta-analysis. Ann Transl Med 2021', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7940942/'},
        {'title': 'De Backer J, et al. 2025 ESC Guidelines for the management of cardiovascular disease and pregnancy. Eur Heart J 2025;46:4462', 'url': 'https://pubmed.ncbi.nlm.nih.gov/40878294/'},
        {'title': 'van Hagen IM, et al. Pregnancy in women with a mechanical heart valve: data of the ESC Registry of Pregnancy and Cardiac disease (ROPAC). Circulation 2015;132:132-42', 'url': 'https://www.ahajournals.org/doi/10.1161/circulationaha.115.015242'},
        {'title': 'Delgado V, et al. 2023 ESC Guidelines for the management of endocarditis. Eur Heart J 2023;44:3948-4042', 'url': 'https://academic.oup.com/eurheartj/article/44/39/3948/7243107'},
        {'title': 'Kang DH, et al. Early surgery versus conventional treatment for infective endocarditis (EASE). N Engl J Med 2012;366:2466-73', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa1112843'},
        {'title': 'Noubiap JJ, Nkeck JR, Kwondom BS, Nyaga UF. Epidemiology of infective endocarditis in Africa: a systematic review and meta-analysis. Lancet Glob Health 2022;10:e77-e86', 'url': 'https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(21)00400-9/fulltext'},
        {'title': 'Dreyfus GD, Corbi PJ, Chan KMJ, Bahrami T. Secondary tricuspid regurgitation or dilatation: which should be the criteria for surgical repair? Ann Thorac Surg 2005;79:127-32', 'url': pm('Dreyfus secondary tricuspid regurgitation or dilatation criteria surgical repair 2005')},
        {'title': 'Akowuah EF, et al. Minithoracotomy vs conventional sternotomy for mitral valve repair: a randomized clinical trial (UK Mini Mitral). JAMA 2023', 'url': 'https://www.journalslibrary.nihr.ac.uk/hta/PKOT2391'},
    ]
    RHD_EPI = ('<p><b>In Kenya and sub-Saharan Africa</b> most mitral stenosis, and most aortic regurgitation in the young, is <b>rheumatic</b>. Patients present in their 20s and 30s, often late, with atrial fibrillation, pulmonary hypertension or in pregnancy. '
               'In the REMEDY registry (14 low- and middle-income countries, median age 28 years) about 17% had died within two years.</p>')
    RHD_PRIMER = ('<p><b>From sore throat to scarred valve.</b></p><ol>'
                  f'<li><b>Group A streptococcal</b> pharyngitis (skin infection may also contribute) in a susceptible child.</li>'
                  f'<li><b>Molecular mimicry</b>: antibodies and T cells raised against the streptococcal M protein and its carbohydrate (GlcNAc) cross-react with cardiac myosin and with laminin on the valve {PROVEN}.</li>'
                  f'<li>The valve endothelium is activated (VCAM-1), T cells home into the valve along the closure line and drive a <b>valvulitis</b> (Th1/Th17) {PROVEN}. This is acute rheumatic fever with carditis.</li>'
                  f'<li>Each <b>recurrence</b> re-injures the valve. Over years, valve interstitial cells lay down scar (TGF-β-driven fibrosis) and calcium: the commissures fuse, the leaflets thicken, the chordae shorten and fuse {INFERRED} (much of this signalling is from calcific aortic valve disease).</li></ol>'
                  '<table class="mini"><tr><th></th><th>Acute carditis</th><th>Chronic rheumatic heart disease</th></tr>'
                  '<tr><td>Lesion</td><td>Mitral <b>regurgitation</b>: annular dilatation, elongated chordae, anterior leaflet prolapse (Carpentier I, II)</td><td>Mitral <b>stenosis</b> (often with regurgitation): commissural fusion, thick leaflets, fused short chordae (Carpentier IIIa)</td></tr>'
                  '<tr><td>The valve…</td><td>cannot close</td><td>cannot open</td></tr>'
                  '<tr><td>Reversible?</td><td>partly, if carditis settles and recurrences are prevented</td><td>no: scar and calcium, treated mechanically (balloon or surgery)</td></tr></table>')

    def mv_patho_ms(pre, tr=False, preg=False):
        return {'id': f'{pre}-patho', 'phase': 'Pathophysiology', 'title': 'Pathophysiology: rheumatic mitral stenosis' + (' and functional TR' if tr else ''),
                'body': RHD_EPI + RHD_PRIMER
                        + '<p><b>The haemodynamics.</b> A normal mitral orifice is 4–6 cm²; stenosis becomes clinically significant at <b>1.5 cm² or less</b>. The LV is protected (it is under-filled); the load falls on everything behind the valve:</p>'
                        + chain('Small orifice', 'LA→LV gradient', 'LA pressure ↑, LA dilates', '!Atrial fibrillation', '!LA appendage thrombus → stroke')
                        + chain('LA pressure ↑', 'Pulmonary venous pressure ↑ (breathless, oedema)', 'Pulmonary hypertension', 'RV pressure overload', '!RV dilates → TR → right heart failure')
                        + '<p><b>Why a fast heart rate decompensates MS.</b> The gradient depends on flow and on the time available for the LA to empty: diastole. Tachycardia (AF with a fast ventricular rate, exercise, fever, anaemia, <b>pregnancy</b>) shortens diastole, so the gradient and LA pressure climb and pulmonary oedema follows. Hence rate control (a β-blocker) and why MS often declares itself in pregnancy. '
                        'AF also removes the atrial kick and adds stasis: rheumatic AF carries a high embolic risk, and it is treated with <b>warfarin</b>, not a direct oral anticoagulant.</p>'
                        + ('<p><b>Functional tricuspid regurgitation.</b> Pulmonary hypertension dilates the RV; the tricuspid annulus stretches along its anterior and posterior segments (the septal segment is fixed), the leaflets are pulled apart, and TR appears on an anatomically normal valve. It may persist or progress after the mitral valve is fixed. '
                           'Rheumatic (organic) tricuspid disease also occurs, in perhaps 10–15% of advanced cases.</p>' if tr else '')
                        + ev('the immunology (mimicry, valvular endothelial activation, T-cell infiltration) rests on studies of human rheumatic valves and valve-derived T-cell clones (Guilherme et al., Circulation 1995; Cunningham, Clin Microbiol Rev 2000); the later fibrocalcific steps are largely extrapolated from calcific aortic valve disease. '
                             'Clinical picture: Carapetis et al., Nat Rev Dis Primers 2016; Chandrashekhar, Westaby and Narula, Lancet 2009. '
                             'Prevention works: in GOAL (NEJM 2022; 818 Ugandan children with latent RHD) monthly benzathine penicillin cut progression over 2 years from 8.3% to 0.8%. '
                             'INVICTUS (NEJM 2022; 4,531 patients with rheumatic AF) found more vascular deaths and ischaemic strokes with rivaroxaban than with a vitamin K antagonist; ESC/EACTS 2025 advise against DOACs in AF with rheumatic MS and a valve area of 2.0 cm² or less.'),
                'view': MV_SIDE, 'spin': True,
                'show': [*CHP, *RH_MV, 'jet-ms', 'laa', 'laa-thrombus', *(['tricuspid-annulus', 'tv-anterior', 'tv-posterior', 'tv-septal'] if tr else [])],
                'hide': [i for i in HEART_OFF if i != 'laa'] + NORMAL_MV + PATHO_HIDE, 'opacity': {**FAINT, **PATHO_OP}, 'askAfter': True,
                'highlight': ['rh-mv', 'rh-mv-edge'], 'danger': ['laa-thrombus', 'jet-ms'],
                'labels': ['rh-mv', 'rh-mv-edge', 'rh-mv-calcium', 'rh-chordae', 'jet-ms', 'laa-thrombus', 'pa-trunk', 'rv', *(['tricuspid-annulus'] if tr else [])],
                'ask': ask('A 26-year-old woman with moderate rheumatic MS, comfortable at rest, goes into AF at 150 beats per minute and within hours is in pulmonary oedema. Why?',
                           'The short diastole leaves too little time to empty the LA through the narrow valve, so the gradient and LA pressure rise',
                           'In MS the transmitral gradient rises steeply with heart rate. Slowing the rate (and cardioversion, with anticoagulation) often relieves the oedema before anything is done to the valve.',
                           'The LV has failed', 'The valve has suddenly narrowed further', 'AF has caused acute mitral regurgitation'),
                'ct': ct(R(MC), 'axial')}

    def mv_patho_mr(pre):
        return {'id': f'{pre}-patho', 'phase': 'Pathophysiology', 'title': 'Pathophysiology: rheumatic mitral regurgitation',
                'body': RHD_EPI + RHD_PRIMER
                        + '<p><b>How carditis makes the valve leak.</b> Rheumatic carditis is a pancarditis: the leaflets are oedematous and friable, the <b>chordae elongate</b> so the <b>anterior leaflet prolapses</b> (Carpentier II), the inflamed ventricle and atrium dilate and <b>stretch the annulus</b> (Carpentier I), and displaced papillary muscles tether the leaflets. '
                        'The jet points <b>away from the prolapsing leaflet</b>: posteriorly. With time, fibrosis retracts the posterior leaflet and the lesion becomes restrictive (IIIa), often with some stenosis.</p>'
                        + chain('Regurgitant volume into the LA', 'LA and LV volume overload', 'LV dilates (eccentric hypertrophy)', '!Contractility falls while the EF still looks normal', '!Irreversible LV dysfunction')
                        + chain('LA pressure ↑', 'Pulmonary congestion', 'Pulmonary hypertension', 'AF')
                        + '<p><b>Why the EF misleads.</b> In MR the LV empties partly into the low-pressure LA, so the ejection fraction overstates how well the muscle works. Guidelines therefore operate in symptomatic severe MR, and in asymptomatic patients once the <b>EF falls to 60% or less</b> or the <b>LV end-systolic diameter reaches 40 mm</b>, before damage is permanent.</p>'
                        '<p><b>Repair or replace?</b> Repair keeps the native valve and avoids warfarin, which matters for young women and where INR monitoring is hard, but rheumatic repair is less durable than repair of degenerative valves: the fibrotic process goes on, and recurrences re-injure the valve, so <b>penicillin prophylaxis continues after surgery</b>.</p>'
                        + ev('mechanisms from human pathology and the Carpentier functional classification; see Carapetis et al., Nat Rev Dis Primers 2016. Intervention thresholds from the ACC/AHA 2020 and ESC/EACTS 2021/2025 guidelines. '
                             'A meta-analysis of 16 retrospective studies (Jiang et al., Ann Transl Med 2021; 8,659 patients) found lower early mortality (OR 0.58) and better long-term survival with repair than replacement for rheumatic mitral disease, but about twice the risk of reoperation (HR 1.96); the studies are observational and repaired valves were selected. The AHA 2020 statement notes repair is feasible in most patients in expert hands, while many endemic-region centres favour replacement to avoid redo surgery.'),
                'view': MV_SIDE, 'spin': True,
                'show': [*CHP, 'mv-ant-prolapse', 'mv-post-leaflet', 'chordae-long', 'papillary', 'mitral-annulus', 'jet-mr'],
                'hide': [*HEART_OFF, 'mv-ant-leaflet', 'chordae', *PATHO_HIDE], 'opacity': {**FAINT, **PATHO_OP}, 'askAfter': True,
                'highlight': ['mv-ant-prolapse', 'chordae-long'], 'danger': ['jet-mr'],
                'labels': ['mv-ant-prolapse', 'mv-post-leaflet', 'chordae-long', 'jet-mr', 'mitral-annulus'],
                'ask': ask('A 16-year-old has severe rheumatic MR, mild breathlessness, EF 62% and LV end-systolic diameter 43 mm. Why not wait until the EF falls?',
                           'In MR the EF overstates LV function; an end-systolic diameter of 40 mm or more already signals damage, and waiting risks permanent dysfunction',
                           'Because the LV unloads into the low-pressure LA, a "normal" EF can hide falling contractility. Guidelines use EF ≤60% or LVESD ≥40 mm as triggers for surgery.',
                           'The EF is normal, so surgery can safely wait', 'Surgery is only for mitral stenosis', 'Medical therapy reverses chronic rheumatic MR'),
                'ct': ct(R(MC), 'axial')}

    def mv_patho_ie(pre):
        return {'id': f'{pre}-patho', 'phase': 'Pathophysiology', 'title': 'Pathophysiology: endocarditis on a rheumatic valve',
                'body': '<p><b>Why rheumatic valves get infected.</b> Rheumatic heart disease is the commonest substrate for endocarditis in African adults. A regurgitant jet and a scarred valve damage the endothelium on the <b>low-pressure side</b> (the atrial face of the mitral leaflets); platelets and fibrin settle there (a sterile thrombus). '
                        'A transient bacteraemia (dental sepsis, skin infection, an intravenous line, injecting) seeds it, and the organisms multiply inside the growing <b>vegetation</b>, shielded from white cells and antibiotics. This is why bactericidal antibiotics are given intravenously for weeks.</p>'
                        '<p>What the infection does:</p>'
                        + chain('Vegetation', '!Destruction: leaflet perforation, chordal rupture', 'Acute severe MR', '!Pulmonary oedema, shock')
                        + chain('Vegetation', '!Embolism: brain, spleen, kidneys, limbs', 'Risk highest with mobile vegetations ≥10 mm on the anterior mitral leaflet')
                        + chain('Vegetation', 'Local spread: annular abscess (commoner at the aortic valve)', '!Uncontrolled infection')
                        + '<p>An <b>acute</b> leak gives the LA no time to dilate: the pressure rises at once and the lungs flood, unlike the slow volume overload of chronic MR.</p>'
                        + ev('in a meta-analysis of 42 African studies (Noubiap et al., Lancet Glob Health 2022), rheumatic heart disease was the underlying condition in 52% of adults; staphylococci (41%) and streptococci (34%) dominated; only about half of blood cultures were positive (often after antibiotics); in-hospital mortality was 23%. '
                             'Surgical indications and timing: ESC 2023 endocarditis guidelines (Delgado et al.).'),
                'view': clook(MC, MN * 1.0 + V(np.cross(MN, MU)) * 0.35 + V([0, 0.35, 0]), 125), 'spin': True,
                'show': [*CHP, *VALVE, 'mv-vegetation', 'jet-mr'], 'hide': [*HEART_OFF, *PATHO_HIDE], 'opacity': {**FAINT, **PATHO_OP}, 'askAfter': True,
                'highlight': ['mv-vegetation'], 'danger': ['jet-mr'], 'labels': ['mv-vegetation', 'mv-ant-leaflet', 'mv-post-leaflet', 'jet-mr'],
                'ask': ask('Which vegetation carries the highest risk of embolism?', 'A mobile vegetation of 10 mm or more on the anterior mitral leaflet',
                           'Size (≥10 mm), mobility and the mitral (especially anterior leaflet) position predict embolism; the risk is highest in the first days to weeks of treatment, which is the case for early surgery.',
                           'A small, sessile vegetation on the aortic valve', 'Any vegetation after two weeks of antibiotics', 'A vegetation on the tricuspid valve'),
                'ct': ct(R(MC), 'axial')}

    def mv_case(pre, title, lead, body, quiz, show=None, hide=None, labels=None, view=None):
        return {'id': f'{pre}-case', 'phase': 'Case', 'title': title, 'lead': lead, 'body': body,
                'view': view or clook(MC, V([0.4, 0.8, 0.45]), 260), 'show': show or [*CH, *VALVE], 'hide': hide or HEART_OFF, 'opacity': {**FAINT, 'la': 0.25, 'lv': 0.25},
                'labels': labels or ['la', 'lv'], 'ask': quiz, 'ct': ct(R(MC), 'axial')}

    def mv_laa(pre):
        L = V(LM['laa-thrombus']) if 'laa-thrombus' in LM else Lc('la-incision')
        return {'id': f'{pre}-laa', 'phase': 'Left atrium', 'title': 'Remove the thrombus; close the appendage',
                'body': '<p>Before touching the valve, look into the <b>appendage</b>. Lift the thrombus out whole with a spoon or forceps, without fragmenting it; wash and suck out the appendage and the LA. Keep the aortic cross-clamp on until the atrium is cleared.</p>'
                        '<p>Close the appendage from inside: a double layer of running 4-0 or 5-0 polypropylene across its orifice (or excise it and close, or an external clip). Check for a residual stump: a leak into an incompletely closed appendage is itself a source of emboli.</p>'
                        '<p>With AF, add a <b>surgical ablation</b> (a left atrial lesion set or a full Cox-Maze) where the equipment and experience exist; in a very large rheumatic LA sinus rhythm is less often restored.</p>'
                        + ev('LAAOS III (NEJM 2021; 4,770 patients with AF having cardiac surgery): closing the appendage reduced ischaemic stroke or systemic embolism (4.8% vs 7.0% over 3.8 years), on top of continued anticoagulation. The 2024 ESC AF guideline recommends appendage closure at cardiac surgery in AF (class I) and concomitant surgical ablation at mitral surgery (class I). Warfarin continues after surgery.'),
                'view': clook(L, MN * 1.0 + V([0.55, 0.35, 0.1]), 190), 'show': [*CH, 'laa', 'laa-thrombus', *CANS], 'hide': [i for i in HEART_OFF if i != 'laa'], 'opacity': {**FAINT, 'la': 0.3, 'laa': 0.35},
                'highlight': ['laa-thrombus'], 'danger': ['laa-thrombus'], 'labels': ['laa', 'laa-thrombus'],
                'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Remove the thrombus', 'port': 'sternotomy', 'remove': ['laa-thrombus'], 'path': [R(L + V([0, 0, 8])), R(L), R(L - V([0, 0, 8]))]},
                'ask': ask('Why is the appendage closed even though the patient will stay on warfarin?', 'Closure adds protection against stroke on top of anticoagulation',
                           'In LAAOS III most patients stayed on anticoagulation, and closure still cut stroke and systemic embolism by about a third.', 'It is not: warfarin makes closure unnecessary', 'To shorten bypass time'),
                'ct': ct(R(L), 'axial')}

    def mv_inspect(pre):
        return {'id': f'{pre}-inspect', 'phase': 'Valve', 'title': 'Valve analysis: repair or replace?',
                'body': '<p>Analyse the valve systematically before deciding (Carpentier): the <b>annulus</b> (dilated?), each <b>leaflet</b> segment (pliable, thickened, retracted, calcified?), the <b>commissures</b> (fused?), the <b>chordae</b> (elongated, ruptured, fused?) and the papillary muscles. Test with saline under pressure.</p>'
                        '<p><b>Rheumatic repair</b>, when the tissue allows: commissurotomy, thinning (peeling) of thickened leaflets, fenestration of fused chordae, <b>augmenting a retracted leaflet with a pericardial patch</b>, chordal shortening, transfer or artificial chordae for prolapse, and a <b>complete ring</b> annuloplasty. '
                        '<b>Replace</b> when the leaflets are thick, retracted and calcified, the subvalvular apparatus is fused, or a durable repair is unlikely: a failed repair in a young patient means another sternotomy.</p>'
                        + ev('repair versus replacement in rheumatic disease has no randomised trial; observational data and meta-analysis favour repair for survival with more reoperations (Jiang et al., 2021). Intraoperative TOE after repair (residual MR, gradient, systolic anterior motion) is standard.'),
                'view': valve_view, 'show': [*VALVE, *DANGER, 'mv-ant-prolapse', 'chordae-long'], 'hide': [*HEART_OFF, 'mv-ant-leaflet', 'chordae'], 'opacity': {**FAINT, 'la': 0.12, 'lv': 0.25},
                'highlight': ['mv-ant-prolapse', 'mv-post-leaflet'], 'labels': ['mv-ant-prolapse', 'mv-post-leaflet', 'chordae-long', 'mitral-annulus'],
                'ask': ask('The anterior leaflet prolapses on long chordae, but the posterior leaflet is thick, retracted and calcified with fused chordae. The most likely durable option in this 17-year-old?',
                           'Replacement: a retracted, calcified posterior leaflet leaves no coaptation surface a repair can rely on',
                           'Prolapse alone can be repaired, but repair needs a pliable posterior leaflet to coapt against. Patch augmentation is possible in selected cases; in a heavily diseased valve, replacement is more durable.',
                           'A ring annuloplasty alone', 'Close the chest and treat medically', 'Balloon commissurotomy after surgery'),
                'ct': ct(R(MC), 'axial')}

    def mv_debride(pre):
        return {'id': f'{pre}-debride', 'phase': 'Valve', 'title': 'Radical debridement of the infected valve',
                'body': '<p>Remove the vegetation <b>whole</b>, without fragmenting it into the LV, and send it for culture, Gram stain, histology and (if cultures were negative) <b>16S PCR</b>. Excise all infected and necrotic tissue back to healthy tissue; inspect the annulus for an abscess. Irrigate; change gloves and instruments before implanting.</p>'
                        '<p>A limited lesion (a perforation, one ruptured chord) in a young patient may be <b>repaired</b> (autologous pericardial patch, chordal repair). With extensive destruction, or a rheumatic valve that would not repair anyway, replace it. A posterior annular abscess is debrided and the defect closed with a pericardial patch before the sutures go in.</p>'
                        + ev('ESC 2023: mitral repair is preferred when a durable result is likely, especially in the young; mechanical and biological prostheses have similar reinfection risk, so the choice follows age and anticoagulation, not the infection.'),
                'view': valve_view, 'show': [*VALVE, *DANGER, 'mv-vegetation'], 'hide': HEART_OFF, 'opacity': {**FAINT, 'la': 0.12, 'lv': 0.25},
                'highlight': ['mv-vegetation'], 'danger': DANGER, 'labels': ['mv-vegetation', 'mv-ant-leaflet'],
                'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Excise the vegetation', 'port': 'sternotomy', 'remove': ['mv-vegetation'],
                           'path': [R(V(LM['mv-vegetation']) + MN * 3 + V([0, 0, 6])), R(V(LM['mv-vegetation']) + MN * 2), R(V(LM['mv-vegetation']) + MN * 3 - V([0, 0, 6]))]} if 'mv-vegetation' in LM else None,
                'ask': ask('Cultures were negative after antibiotics given at a clinic. What gives the best chance of identifying the organism now?', 'Culture, histology and 16S PCR of the excised valve and vegetation',
                           'Molecular diagnosis on valve tissue identifies the organism in many culture-negative cases, and it guides the length and choice of antibiotics.', 'Repeat blood cultures on bypass', 'Swab the pericardium'),
                'ct': ct(R(MC), 'axial')}

    def mv_tv(pre):
        TC = Lc('tv-centre') if 'tv-centre' in LM else Lc('septum')
        return {'id': f'{pre}-tv', 'phase': 'Tricuspid', 'title': 'Tricuspid annuloplasty (after the mitral)',
                'body': '<p>With the mitral prosthesis in, close the septum. The tricuspid ring can then go in with the <b>cross-clamp off and the heart beating</b> (caval snares still tight), which shortens ischaemic time and shows at once if a suture has caught the conduction tissue.</p>'
                        '<p>An <b>incomplete ring</b> sized to the anterior leaflet, the sutures along the anterior and posterior annulus; leave the septal annulus near <b>Koch\'s triangle</b> (AV node) without sutures, or stay very shallow. A De Vega suture annuloplasty is the cheaper alternative. (The <b>Tricuspid</b> module covers this step by step.)</p>'
                        + ev('ESC/EACTS 2025 and ACC/AHA 2020: repair severe TR at left-sided surgery (class I); consider it for milder TR with a dilated annulus (≥40 mm or 21 mm/m² on echo). Dreyfus et al. (Ann Thorac Surg 2005) showed that annuloplasty guided by annular dilatation rather than TR grade improved function over years.'),
                'view': clook(TC, (V(LM['tv-normal']) if 'tv-normal' in LM else V([1, 0, 0])) * 1.0 + V([0.55, 0.35, 0.05]), 170), 'show': [*CH, *[i for i in ('tricuspid-annulus', 'tv-anterior', 'tv-posterior', 'tv-septal', 'tv-ring', 'koch', 'tv-avnode', 'snares') if has(i)], 'mv-prosthesis'],
                'hide': [*HEART_OFF, 'mv-ant-leaflet', 'svc'], 'opacity': {**FAINT, 'ra': 0.15, 'rv': 0.22, 'lv': 0.15, 'myocardium': 0.08},
                'highlight': ['tv-ring'], 'danger': ['tv-avnode', 'koch'], 'labels': ['tv-ring', 'tricuspid-annulus', 'tv-avnode'],
                'action': {'kind': 'reveal', 'label': 'Place the ring', 'port': 'sternotomy', 'ids': ['tv-ring']},
                'ct': ct(R(TC), 'axial')}

    def build_case(steps_by_group):
        """[(label, kind, [steps])] -> (steps with seq set, sequence)"""
        out, sq_ = [], []
        for i, (lab, kind, sts) in enumerate(steps_by_group):
            sq_.append({'label': lab, 'kind': kind})
            for s in sts:
                if s is None: continue
                s = {**s, 'seq': i}
                if s.get('action') is None: s.pop('action', None)
                out.append(s)
        return out, sq_

    def ids_(pre, xs):
        """re-prefix step ids so that each case has unique ids"""
        return [{**x, 'id': f'{pre}-{x["id"].split("-", 1)[1]}'} for x in xs]

    MS_ACCESS = '<p><b>Access</b>: median sternotomy and a left atriotomy (the default; any concomitant surgery). Transseptal for a small LA, tricuspid surgery or a redo; a right mini-thoracotomy for isolated mitral surgery in selected patients.</p>'
    # ------------------------------------------------ 1. rheumatic MS with AF and an appendage thrombus
    c1 = mv_case('mc', 'Case: rheumatic mitral stenosis, AF and an appendage thrombus',
                 '<p>A <b>32-year-old woman</b> from western Kenya, NYHA III, on monthly benzathine penicillin since a teenage episode of rheumatic fever. <b>Atrial fibrillation</b>, rate controlled. '
                 'Echo: <b>MVA 0.9 cm²</b> (planimetry), mean gradient 14 mmHg at 90/min, <b>Wilkins score 11</b> (thick, calcified commissures, fused chordae), mild MR; PA systolic pressure 60 mmHg; aortic and tricuspid valves normal. TOE: a <b>thrombus in the left atrial appendage</b>. She has two children and does not plan more; the INR clinic is at the county hospital.</p>',
                 '<p><b>Why not a balloon?</b> Percutaneous mitral commissurotomy (PMC) is the first choice for symptomatic MS with favourable anatomy. Here there are three reasons against it: an <b>LA thrombus</b> (a contraindication: the catheter crosses the LA), <b>unfavourable morphology</b> (Wilkins over 8, commissural calcium) and a poor predicted result. So: <b>surgery</b>.</p>'
                 '<p><b>Which operation?</b> Heavily calcified, fused rheumatic valves rarely repair durably: <b>replacement</b>, with removal of the thrombus, <b>closure of the appendage</b> and, where available, <b>surgical ablation</b> of the AF.</p>'
                 '<p><b>Which valve?</b> At 32 with reliable INR access: a <b>mechanical</b> valve (mitral INR target about 3.0, range 2.5–3.5). She needs warfarin for rheumatic AF anyway, so a tissue valve would not spare her anticoagulation, and it would degenerate early (sooner after rheumatic disease) and need a redo.</p>'
                 + MS_ACCESS
                 + ev('Wilkins score (Br Heart J 1988): leaflet mobility, thickening, calcification and subvalvular disease, each 1–4; 8 or less favours PMC; commissural calcium is the strongest single predictor of a poor result (Nunes et al., Circulation 2014). ESC/EACTS 2025 list LA thrombus, more than mild MR, severe or bicommissural calcification and severe concomitant aortic or tricuspid disease as contraindications to PMC.'),
                 ask('What makes surgery, not a balloon, the right choice here?', 'The left atrial appendage thrombus together with a calcified, unfavourable valve (Wilkins 11)',
                     'An LA thrombus contraindicates PMC, and the morphology predicts a poor balloon result. A tissue valve would not spare her warfarin, as rheumatic AF needs it anyway.',
                     'Her age', 'The pulmonary pressure of 60 mmHg', 'Atrial fibrillation alone'),
                 show=[*CH, *RH_MV, 'laa', 'laa-thrombus'], hide=[i for i in HEART_OFF if i != 'laa'] + NORMAL_MV, labels=['laa-thrombus', 'rh-mv', 'rh-mv-calcium'], view=clook(MC, MN * 1.0 + V([0.45, 0.35, 0.1]), 150))
    ms_steps, ms_sq = build_case([
        ('Patho', 'other', [mv_patho_ms('ms')]), ('Anatomy', 'other', [mv_anat('ms')]), ('Case', 'other', [{**c1, 'id': 'ms-case'}]),
        ('Sternotomy', 'other', [mv_sternotomy('ms', 0)]), ('Cannulate', 'artery', [mv_cannulate('ms', 0)]), ('Clamp', 'artery', [mv_clamp('ms', 0)]), ('Atriotomy', 'vein', [mv_la('ms', 0)]),
        ('Appendage', 'vein', [mv_laa('ms')]), ('Excise', 'fissure', [mv_excise('ms', 0)]), ('Sutures', 'fissure', [mv_sutures('ms', 0)]), ('Seat', 'bronchus', [mv_seat('ms', 0)]),
        ('Close', 'other', [mv_close('ms', 0)]), ('Reperfuse', 'other', [mv_reperfuse('ms', 0)]), ('Decannulate', 'artery', [mv_decannulate('ms', 0)])])
    # rheumatic excision: the whole funnel is often thick and fused
    for s in ms_steps:
        if s['id'] == 'ms-excise':
            s['body'] += ('<p><b>In this rheumatic valve</b> the leaflets are fused into a thick funnel. Split the fused commissures to see the subvalvular apparatus; keep the posterior leaflet and its chordae only if they are pliable (a thick, fused posterior apparatus can obstruct the prosthesis or trap a disc: excise it and, if you wish, resuspend a few chordae). '
                          'Remove calcium from the annulus piece by piece, catching every fragment.</p>')
    # ------------------------------------------------ 2. MS with severe functional TR: transseptal, mitral + tricuspid
    c2 = mv_case('mt', 'Case: mitral stenosis with pulmonary hypertension and severe TR',
                 '<p>A <b>38-year-old man</b> with rheumatic MS: MVA 1.0 cm², Wilkins 10, moderate MR. <b>PA systolic pressure 75 mmHg</b>, a dilated RV, <b>severe functional TR</b> with a tricuspid annulus of 44 mm; ankle oedema and a large liver. Sinus rhythm; LA 52 mm; no thrombus.</p>',
                 '<p><b>Mitral</b>: unfavourable for PMC (calcium, more than mild MR): replace. <b>Tricuspid</b>: severe TR at left-sided surgery is repaired at the same operation. Pulmonary pressure usually falls after the mitral valve is fixed, but the dilated annulus does not shrink back, and untreated TR often progresses.</p>'
                 '<p><b>Access</b>: the <b>transseptal</b> route suits two-valve surgery from the right side: bicaval cannulation with snares, one right atriotomy, the mitral through the septum and the tricuspid in the same field.</p>'
                 + ev('ESC/EACTS 2025 and ACC/AHA 2020: tricuspid repair for severe TR at left-sided valve surgery (class I); for less than severe TR with annular dilatation it should be considered. The rheumatic tricuspid valve can be organic (thickened, fused) too: look at it, not only the echo.'),
                 ask('Why repair the tricuspid now, when pulmonary pressure may fall after MVR?', 'The dilated annulus does not recover, and TR left at left-sided surgery often progresses; a redo for TR carries high risk',
                     'Functional TR is an annular problem; lowering the afterload helps but does not reverse annular dilatation. Isolated tricuspid reoperation later carries a high mortality.',
                     'It is not needed: TR always resolves', 'Only organic TR is repaired', 'Because the transseptal route requires it'),
                 show=[*CH, *RH_MV, *[i for i in ('tricuspid-annulus', 'tv-anterior', 'tv-posterior', 'tv-septal') if has(i)]], hide=HEART_OFF + NORMAL_MV, labels=['rh-mv', 'tricuspid-annulus', 'rv'])
    ts_steps, ts_sq = build_case([
        ('Patho', 'other', [mv_patho_ms('mt', tr=True)]), ('Anatomy', 'other', [mv_anat('mt')]), ('Case', 'other', [{**c2, 'id': 'mt-case'}]),
        ('Sternotomy', 'other', [mv_sternotomy('mt', 0)]), ('Cannulate', 'artery', [mv_cannulate('mt', 0, septal=True)]), ('Clamp', 'artery', [mv_clamp('mt', 0)]),
        ('Right atrium', 'vein', [ts_[5]]), ('Septum', 'vein', [ts_[6]]), ('Excise', 'fissure', [mv_excise('mt', 0, view=sept_view)]), ('Sutures', 'fissure', [mv_sutures('mt', 0, view=sept_view)]),
        ('Seat', 'bronchus', [mv_seat('mt', 0, view=sept_view)]), ('Tricuspid', 'vein', [mv_tv('mt') if has('tv-ring') else None]),
        ('Close', 'other', [mv_close('mt', 0, septal=True)]), ('Reperfuse', 'other', [mv_reperfuse('mt', 0)]), ('Decannulate', 'artery', [mv_decannulate('mt', 0)])])
    # ------------------------------------------------ 3. adolescent rheumatic MR: repair or replace
    c3 = mv_case('mr', 'Case: a teenager with severe rheumatic mitral regurgitation',
                 '<p>A <b>17-year-old girl</b>, two admissions with acute rheumatic fever, now NYHA II–III. Echo: <b>severe MR</b> with a posteriorly directed jet from <b>anterior leaflet prolapse</b>; the <b>posterior leaflet is thick and restricted</b>; LV end-diastolic diameter 64 mm, <b>end-systolic 43 mm</b>, EF 60%. Mitral valve area normal. No active carditis (ESR, CRP normal). She lives 80 km from the nearest INR clinic and hopes to have children.</p>',
                 '<p><b>Operate now</b>: symptoms, and an LV end-systolic diameter over 40 mm.</p>'
                 '<p><b>Repair first, if the valve allows</b>: in a young woman, repair avoids warfarin (teratogenic, and hard to monitor far from a clinic) and keeps the native valve for pregnancy. Its weakness in rheumatic disease is durability.</p>'
                 '<p><b>If it must be replaced</b>, the choice is hard. A <b>mechanical</b> valve lasts but needs warfarin: a high-risk pregnancy (valve thrombosis, embryopathy, bleeding). A <b>tissue</b> valve avoids warfarin and is favoured for women planning pregnancy, but in a 17-year-old it may fail within about 10 years and needs a redo (mitral valve-in-valve is possible later). Decide with her and her family, before theatre.</p>'
                 + MS_ACCESS
                 + ev('ESC 2025 pregnancy guideline: when valve replacement is needed in a woman contemplating pregnancy, a bioprosthesis is recommended (class I, upgraded from IIa). Pregnancy with a mechanical valve is high risk (modified WHO class III); in the ESC ROPAC registry only about 58% of such pregnancies were free of serious adverse events (van Hagen et al., Circulation 2015). Continue secondary prophylaxis after surgery (AHA 2020 statement).'),
                 ask('The valve is unrepairable. For this 17-year-old who wants children and lives far from an INR clinic, which valve do current guidelines favour?',
                     'A bioprosthesis, accepting a likely reoperation, after a shared decision',
                     'The ESC 2025 pregnancy guideline recommends a bioprosthesis in women contemplating pregnancy; poor access to INR monitoring adds to the case. The cost is early degeneration and a redo, which she must understand.',
                     'A mechanical valve: durability matters most', 'No valve: continue medical therapy until after pregnancy', 'A Ross-type pulmonary autograft in the mitral position'),
                 show=[*CH, 'mv-ant-prolapse', 'mv-post-leaflet', 'chordae-long', 'papillary', 'jet-mr'], hide=[*HEART_OFF, 'mv-ant-leaflet', 'chordae'], labels=['mv-ant-prolapse', 'jet-mr'])
    mr_steps, mr_sq = build_case([
        ('Patho', 'other', [mv_patho_mr('mr')]), ('Anatomy', 'other', [mv_anat('mr')]), ('Case', 'other', [{**c3, 'id': 'mr-case'}]),
        ('Sternotomy', 'other', [mv_sternotomy('mr', 0)]), ('Cannulate', 'artery', [mv_cannulate('mr', 0)]), ('Clamp', 'artery', [mv_clamp('mr', 0)]), ('Atriotomy', 'vein', [mv_la('mr', 0)]),
        ('Analyse', 'other', [mv_inspect('mr')]), ('Excise', 'fissure', [mv_excise('mr', 0)]), ('Sutures', 'fissure', [mv_sutures('mr', 0)]), ('Seat', 'bronchus', [mv_seat('mr', 0)]),
        ('Close', 'other', [mv_close('mr', 0)]), ('Reperfuse', 'other', [mv_reperfuse('mr', 0)]), ('Decannulate', 'artery', [mv_decannulate('mr', 0)])])
    for s in mr_steps:
        if s['id'] == 'mr-seat':
            s['body'] = s['body'].replace('For a bileaflet mechanical valve, the usual orientation is <b>anti-anatomical</b> (hinges perpendicular to the natural commissures).',
                                          'For a <b>bioprosthesis</b> (this patient), orient the struts so that none sits in the LV outflow tract. For a bileaflet mechanical valve, the usual orientation is <b>anti-anatomical</b>.')
    # ------------------------------------------------ 4. endocarditis on a rheumatic mitral valve
    c4 = mv_case('me', 'Case: endocarditis on a rheumatic mitral valve',
                 '<p>A <b>24-year-old man</b> with known rheumatic MR, three weeks of fever after a dental abscess. Two blood cultures grow <i>Streptococcus</i>. Today: sudden <b>left arm weakness</b>; CT head: a small ischaemic infarct, <b>no haemorrhage</b>. TOE: a <b>14 mm mobile vegetation</b> on the anterior leaflet, a leaflet perforation, <b>severe MR</b>; breathless on minimal exertion.</p>',
                 '<p><b>Indications</b> (ESC 2023): severe MR with heart failure; a vegetation of 10 mm or more <b>after an embolic event</b>. Both point to <b>urgent</b> surgery, within days, on antibiotics.</p>'
                 '<p><b>The stroke</b>: after a transient ischaemic attack or an ischaemic stroke <b>without haemorrhage or coma</b>, surgery should not be delayed when it is indicated for heart failure, uncontrolled infection or a high embolic risk. After an intracranial <b>haemorrhage</b>, surgery is generally deferred (often about 4 weeks) unless the patient is unstable.</p>'
                 + ev('EASE (NEJM 2012; 76 patients with left-sided endocarditis, severe valve disease and vegetations over 10 mm): surgery within 48 hours reduced in-hospital death or embolism at 6 weeks from 23% to 3%, mainly by preventing embolism. ESC 2023 endocarditis guidelines (Delgado et al.) define emergency (within 24 h), urgent (3–5 days) and non-urgent timing.'),
                 ask('Small ischaemic stroke without haemorrhage, a 14 mm mobile vegetation, severe MR with heart failure. When should he have surgery?',
                     'Urgently, within days, without waiting for the stroke to recover',
                     'A non-haemorrhagic stroke without coma is not a reason to delay when surgery is indicated; waiting risks a second embolus and worsening heart failure (EASE; ESC 2023).',
                     'After 6 weeks of antibiotics', 'After 4 weeks, to let the stroke settle', 'Only if a second embolus occurs'),
                 show=[*CH, *VALVE, 'mv-vegetation', 'jet-mr'], labels=['mv-vegetation', 'jet-mr'], view=clook(MC, MN * 0.9 + V(np.cross(MN, MU)) * 0.5 + V([0, 0.4, 0]), 170))
    ie_steps, ie_sq = build_case([
        ('Patho', 'other', [mv_patho_ie('me')]), ('Anatomy', 'other', [mv_anat('me')]), ('Case', 'other', [{**c4, 'id': 'me-case'}]),
        ('Sternotomy', 'other', [mv_sternotomy('me', 0)]), ('Cannulate', 'artery', [mv_cannulate('me', 0)]), ('Clamp', 'artery', [mv_clamp('me', 0)]), ('Atriotomy', 'vein', [mv_la('me', 0)]),
        ('Debride', 'fissure', [mv_debride('me')]), ('Excise', 'fissure', [mv_excise('me', 0)]), ('Sutures', 'fissure', [mv_sutures('me', 0)]), ('Seat', 'bronchus', [mv_seat('me', 0)]),
        ('Close', 'other', [mv_close('me', 0)]), ('Reperfuse', 'other', [mv_reperfuse('me', 0)]), ('Decannulate', 'artery', [mv_decannulate('me', 0)])])
    for s in ie_steps:
        if s['id'] in ('me-excise', 'me-sutures', 'me-seat', 'me-close'):
            s['hide'] = [*s.get('hide', []), 'mv-vegetation']
    # ------------------------------------------------ 5. restenosis after balloon commissurotomy: right mini-thoracotomy
    c5 = mv_case('mm', 'Case: restenosis after balloon commissurotomy',
                 '<p>A <b>41-year-old teacher</b>, balloon mitral commissurotomy 9 years ago, now breathless again. Echo: <b>MVA 1.0 cm²</b>, mean gradient 11 mmHg, <b>bicommissural calcium</b>, Wilkins 10, mild MR; <b>sinus rhythm</b>, no LA thrombus; aortic and tricuspid valves normal. CT: good femoral and iliac vessels, no aortic atheroma. She wants to return to work quickly.</p>',
                 '<p><b>A second balloon?</b> Repeat PMC works for restenosis caused by commissural refusion when the anatomy is still favourable. Bicommissural calcium makes a good result unlikely, so: <b>surgery</b>.</p>'
                 '<p><b>Access</b>: isolated mitral disease, no aortic regurgitation, good femoral vessels and no pleural adhesions: a <b>right mini-thoracotomy</b> is reasonable in a centre that does it regularly. A sternotomy would be equally correct.</p>'
                 + ev('UK Mini Mitral (Akowuah et al., JAMA 2023; 330 patients, degenerative MR repair): the minithoracotomy did not improve physical function at 12 weeks over sternotomy, but was as safe, with a shorter stay and faster early recovery. Evidence for rheumatic valves is observational, from experienced centres.'),
                 ask('What makes a second balloon commissurotomy unlikely to succeed here?', 'Calcium in both commissures',
                     'Balloon commissurotomy works by splitting fused commissures; calcified commissures do not split cleanly, and the balloon then tears the leaflets (severe MR).',
                     'The 9-year interval', 'Sinus rhythm', 'Her age'),
                 show=[*CH, *RH_MV], hide=HEART_OFF + NORMAL_MV, labels=['rh-mv', 'rh-mv-calcium'], view=clook(MC, MN * 1.0 + V([0.45, 0.35, 0.1]), 150))
    mi_steps, mi_sq = build_case([
        ('Patho', 'other', [mv_patho_ms('mm')]), ('Anatomy', 'other', [mv_anat('mm')]), ('Case', 'other', [{**c5, 'id': 'mm-case'}]),
        ('Access', 'other', [mi[2]]), ('Clamp', 'artery', [mi[3]]), ('Atriotomy', 'vein', [mi[4]]), ('Excise', 'fissure', [mi[5]]), ('Sutures', 'fissure', [mi[6]]),
        ('Seat', 'bronchus', [mi[7]]), ('Close', 'other', [mi[8]]), ('Reperfuse', 'other', [mi[9]]), ('Decannulate', 'artery', [mi[10]])])
    MV_CASES = (('mvr-std', 'Rheumatic MS, AF, LA thrombus (sternotomy)', ms_steps, ms_sq),
                ('mvr-septal', 'MS with severe TR (transseptal + tricuspid)', ts_steps, ts_sq),
                ('mvr-repair', 'Teenager with rheumatic MR: repair or replace', mr_steps, mr_sq),
                ('mvr-ie', 'Endocarditis on a rheumatic valve', ie_steps, ie_sq),
                ('mvr-mics', 'Restenosis after balloon (right mini-thoracotomy)', mi_steps, mi_sq))
    for key, appr, steps_, sq in MV_CASES:
        for s in steps_:
            named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
            # the heart alone: the lung hila, nodes, nerves and pleura out of the way; the sternum once it is open
            off_groups = {'arteries', 'veins', 'airway', 'lul-intra', 'lll-intra', 'rul-intra', 'nodes', 'nerves', 'pleura', 'segments', 'trauma', 'muscles', 'landmarks'}
            s['hide'] = [*s.get('hide', []), *[q['id'] for q in atlas['structures'] if q['group'] in off_groups and q['id'] not in named],
                         *[i for i in ('esophagus', 'thymus', 'thyroid') if has(i) and i not in named], *(['sternum'] if s['phase'] not in ('Pathophysiology', 'Anatomy', 'Case') and not s['id'].endswith(('-sternotomy', '-setup')) else []),
                         *[i for i in ('rh-mv', 'rh-mv-edge', 'rh-mv-calcium', 'rh-chordae', 'jet-ms', 'jet-mr', 'mv-ant-prolapse', 'chordae-long', 'laa-thrombus', 'mv-vegetation') if i not in named]]
            if s['phase'] in ('Valve', 'Wean', 'Septum', 'Tricuspid'): s['opacity'] = {**{c_: 0.3 for c_ in CANS}, 'can-retro': 0.3, **s.get('opacity', {})}
            if s['phase'] in ('Valve', 'Septum'): s['hide'] = [*s['hide'], 'svc']
            s['opacity'] = {**{f'vert-t{i}': 0.25 for i in range(2, 11)}, **s.get('opacity', {})}
            for kk in ('highlight', 'danger', 'labels', 'show'):
                if kk in s: s[kk] = [i for i in s[kk] if has(i) or i == 'skin']
        procs[key] = {'id': key, 'op': 'mvr', 'opName': 'Mitral valve replacement', 'side': 'both', 'name': 'Mitral valve replacement', 'approach': appr,
                      'summary': 'Case-based: pathophysiology, anatomy, the patient and the decision, then the operation (access, bypass, left atrium, excision, sutures, prosthesis, de-airing).',
                      'ports': [], 'steps': steps_, 'sources': [*RHD_SRC, *MVSRC], 'group': 'Cardiac', 'sequence': sq}
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
    # ------------------------------------------------------------------------------------------ case-based AVR
    AV_SIDE = clook(AC - AN * 8, V(np.cross(AN, AE)) * 0.9 + AN * 0.25 + V([0, 0.45, 0]), 140)
    RH_AV = [i for i in ('av-rheum', 'av-rheum-edge', 'aortic-annulus', 'stj') if has(i)]
    CALC_AV = [i for i in ('av-cusp-r', 'av-cusp-l', 'av-cusp-n', 'av-calcium', 'aortic-annulus', 'stj') if has(i)]
    AV_SRC = [
        {'title': 'Afifi A, Hosny H, Yacoub M. Rheumatic aortic valve disease: when and who to repair? Ann Cardiothorac Surg 2019;8:383-9', 'url': pm('Afifi Hosny Yacoub rheumatic aortic valve disease when and who to repair')},
        {'title': 'Best evidence topic: in young patients with rheumatic aortic regurgitation, is a Ross operation associated with more autograft failure? Interact CardioVasc Thorac Surg 2010;10:600', 'url': 'https://academic.oup.com/icvts/article/10/4/600/659372'},
        {'title': 'Mentias A, et al. Transcatheter versus surgical aortic valve replacement in patients with rheumatic aortic stenosis. J Am Coll Cardiol 2021;77:1703-13', 'url': 'https://pubmed.ncbi.nlm.nih.gov/33832596/'},
        {'title': 'Ross J Jr, Braunwald E. Aortic stenosis. Circulation 1968;38(1 Suppl):61-7', 'url': pm('Ross Braunwald aortic stenosis Circulation 1968')},
        {'title': 'Rossebø AB, et al. Intensive lipid lowering with simvastatin and ezetimibe in aortic stenosis (SEAS). N Engl J Med 2008;359:1343-56', 'url': pm('Rossebo SEAS simvastatin ezetimibe aortic stenosis 2008')},
        {'title': 'Surgical implications of the 2023 ESC endocarditis guidelines endorsed by EACTS: bridging guidelines and practice. Eur J Cardiothorac Surg 2025;67:ezaf225', 'url': 'https://academic.oup.com/ejcts/article/67/7/ezaf225/8185406'},
        {'title': 'Ribeiro HB, et al. Predictive factors, management, and clinical outcomes of coronary obstruction following TAVI. J Am Coll Cardiol 2013;62:1552-62', 'url': pm('Ribeiro coronary obstruction transcatheter aortic valve implantation predictive factors 2013')},
    ]

    def av_patho_rh(pre, multi=False):
        return {'id': f'{pre}-patho', 'phase': 'Pathophysiology', 'title': 'Pathophysiology: rheumatic ' + ('multivalve disease' if multi else 'aortic regurgitation'),
                'body': RHD_EPI
                        + ('' if multi else RHD_PRIMER)
                        + '<p><b>The pattern of rheumatic valve disease.</b> The <b>mitral</b> valve is involved, alone or with others, in about 85%; mitral plus aortic disease in roughly a fifth; <b>isolated aortic</b> disease is uncommon (under 5%) and is then nearly always <b>regurgitation</b>. Rheumatic aortic <b>stenosis</b> comes late, as fusion and calcium accumulate, and almost never without regurgitation or mitral disease. '
                        'One explanation (a hypothesis) is load: the mitral valve closes against full LV systolic pressure every beat, the aortic cusps against the lower diastolic pressure.</p>'
                        '<p><b>The rheumatic aortic valve</b>: fibrosis <b>retracts</b> the cusps so they no longer meet in the centre (a central leak), the free edges thicken and roll, and the commissures fuse. Contrast degenerative calcific stenosis, where calcium in the cusp bodies stiffens a valve whose commissures are open.</p>'
                        + chain('Regurgitant volume back into the LV', 'LV volume and pressure overload', 'Eccentric hypertrophy, dilatation (years compensated)', '!End-systolic size ↑, EF ↓', '!Irreversible LV dysfunction')
                        + chain('Low aortic diastolic pressure', 'Wide pulse pressure', '!Less coronary perfusion (diastole)', 'Angina, breathlessness')
                        + ('<p><b>Two lesions mask each other.</b> Mitral stenosis under-fills the LV, so aortic regurgitation seems milder and an aortic gradient is lower than the stenosis deserves (low flow). After the mitral valve is opened, the aortic lesion can be unmasked. Assess each valve on its own (valve area, regurgitant volume, LV size), and inspect them at surgery.</p>' if multi else '')
                        + '<p><b>Acute</b> aortic regurgitation (endocarditis, dissection) is different: a normal-sized LV cannot take the sudden volume, the diastolic pressure rises steeply, and pulmonary oedema and shock follow.</p>'
                        + ev('distribution of rheumatic valve involvement: Kumar et al., AHA scientific statement 2020; Afifi, Hosny and Yacoub, Ann Cardiothorac Surg 2019. Haemodynamics and intervention thresholds: ACC/AHA 2020, ESC/EACTS 2021 and 2025 guidelines.'),
                'view': AV_SIDE, 'spin': True,
                'show': [*CHP, *RH_AV, 'jet-ar', 'lvot', *(RH_MV if multi else []), *(['jet-ms'] if multi else [])],
                'hide': [*HEART_OFF, *CUSPS, *(NORMAL_MV if multi else []), *PATHO_HIDE], 'opacity': {**AV_FAINT, **PATHO_OP}, 'askAfter': True,
                'highlight': ['av-rheum', 'av-rheum-edge'], 'danger': ['jet-ar'],
                'labels': ['av-rheum', 'av-rheum-edge', 'jet-ar', 'lv', *(['rh-mv', 'jet-ms'] if multi else [])],
                'ask': (ask('Rheumatic MS and moderate aortic stenosis on echo (mean aortic gradient 25 mmHg). Why might the aortic stenosis be worse than it looks?',
                            'The narrow mitral valve limits LV filling and output, so the aortic gradient is low for the degree of stenosis',
                            'Gradients depend on flow. With low forward flow, a severe aortic stenosis can produce a modest gradient; valve area and direct inspection matter.',
                            'Aortic gradients are always overestimated', 'MS increases aortic flow', 'It cannot be: rheumatic aortic disease is always pure regurgitation') if multi else
                        ask('Chronic severe AR: which finding shows the LV is beginning to fail and surgery should not wait?', 'LV end-systolic diameter over 50 mm (or 25 mm/m²) or an EF of 50% or less',
                            'The end-systolic size tracks contractility. Guidelines (class I) operate at these thresholds even without symptoms, because beyond them LV function may not recover.',
                            'A wide pulse pressure', 'A diastolic murmur', 'An end-diastolic diameter of 60 mm alone')),
                'ct': ct(R(AC), 'coronal')}

    def av_patho_calc(pre, bicuspid=False):
        return {'id': f'{pre}-patho', 'phase': 'Pathophysiology', 'title': 'Pathophysiology: calcific aortic stenosis' + (' (bicuspid valve)' if bicuspid else ''),
                'body': '<p><b>How the valve calcifies.</b> Years of mechanical stress injure the endothelium on the aortic side of the cusps; lipids (LDL, lipoprotein(a)) enter and oxidise; macrophages and T cells follow; and the valve interstitial cells switch to a <b>bone-forming</b> programme (RUNX2), laying calcium nodules in the cusp bodies. The commissures stay open; the cusps become stiff. '
                        + ('A <b>bicuspid</b> valve (about 1–2% of people) carries more stress on its two cusps and calcifies a decade or two earlier, and it comes with a weaker ascending aorta (aortopathy).</p>' if bicuspid else 'It is a disease of later life, sharing risk factors with atherosclerosis.</p>')
                        + chain('Aortic valve area ↓', 'LV pressure overload', 'Concentric hypertrophy', '!O₂ demand ↑, coronary reserve ↓ → angina')
                        + chain('Fixed outflow', 'Exercise vasodilatation', '!Pressure falls → exertional syncope')
                        + chain('Stiff, thick LV', 'Filling depends on the atrial kick', '!Diastolic, then systolic failure → breathlessness')
                        + '<p><b>Once symptoms appear, the clock runs fast</b>: without valve replacement, average survival is a few years (shortest after heart failure). No drug slows the valve: lipid lowering did not change progression in trials. So the treatment of severe symptomatic AS is mechanical: <b>SAVR or TAVI</b>.</p>'
                        '<p>Severe AS: aortic valve area under 1.0 cm², mean gradient 40 mmHg or more, peak velocity 4 m/s or more (with low-flow variants that need extra testing).</p>'
                        + ev('natural history: Ross and Braunwald, Circulation 1968 (angina, syncope, heart failure as the turning points). SEAS (NEJM 2008): simvastatin plus ezetimibe did not reduce aortic valve events. Mechanisms from human valve studies of calcific aortic valve disease.'),
                'view': AV_SIDE, 'spin': True, 'show': [*CHP, *CALC_AV, 'lvot'], 'hide': [*HEART_OFF, *PATHO_HIDE], 'opacity': {**AV_FAINT, **PATHO_OP}, 'askAfter': True,
                'highlight': ['av-calcium'], 'labels': ['av-calcium', 'av-cusp-n', 'lv', 'aorta'],
                'ask': ask('Why does exertional syncope occur in severe aortic stenosis?', 'Exercise dilates peripheral vessels, but the narrowed valve cannot increase output to match, so pressure falls',
                           'Cardiac output is fixed by the valve; vasodilatation drops the blood pressure and cerebral perfusion. Arrhythmia can also contribute.',
                           'Mitral regurgitation develops on exercise', 'The coronary arteries go into spasm', 'Blood pools in the LA'),
                'ct': ct(R(AC), 'coronal')}

    def av_patho_ie(pre):
        return {'id': f'{pre}-patho', 'phase': 'Pathophysiology', 'title': 'Pathophysiology: aortic endocarditis and root abscess',
                'body': '<p>On the aortic valve the vegetation sits on the <b>ventricular</b> face of the cusps (the low-pressure side of the regurgitant jet). The infection perforates or tears the cusps (<b>acute severe AR</b>), and it can burrow into the <b>annulus</b>: a <b>paravalvular abscess</b>, typically in the aortomitral curtain and under the non-coronary sinus, next to the His bundle.</p>'
                        + chain('Vegetation', '!Cusp perforation → acute severe AR', 'Sudden volume load on a normal-sized LV', '!Pulmonary oedema, shock')
                        + chain('Annular extension', 'Abscess, false aneurysm', '!Fistula (to the RA, LA, RV)', '!Heart block (the His bundle lies just below)')
                        + '<p><b>New PR prolongation or heart block</b> in aortic endocarditis means the infection has reached the conduction tissue: an abscess until proved otherwise, and an indication for urgent surgery (uncontrolled infection). <i>Staphylococcus aureus</i> is the commonest cause of destructive, abscess-forming disease.</p>'
                        + ev('ESC 2023 endocarditis guidelines (Delgado et al.): locally uncontrolled infection (abscess, false aneurysm, fistula, enlarging vegetation) is an indication for urgent surgery. African data: Noubiap et al., Lancet Glob Health 2022.'),
                'view': AV_SIDE, 'spin': True, 'show': [*CHP, *CUSPS, 'aortic-annulus', 'av-vegetation', 'root-abscess', 'jet-ar', 'his-bundle', 'mv-ant-leaflet'], 'hide': [*HEART_OFF, 'av-calcium', *PATHO_HIDE],
                'opacity': {**AV_FAINT, **PATHO_OP, 'root-abscess': 0.85, **{c_: 0.45 for c_ in CUSPS}}, 'askAfter': True,
                'highlight': ['av-vegetation'], 'danger': ['root-abscess', 'his-bundle', 'jet-ar'], 'labels': ['av-vegetation', 'root-abscess', 'his-bundle', 'jet-ar', 'mv-ant-leaflet'],
                'ask': ask('A patient with aortic endocarditis develops a PR interval of 280 ms, then complete heart block. What does it mean?', 'The infection has extended into the annulus near the His bundle: a paravalvular abscess',
                           'The conduction tissue lies just below the commissure between the right and non-coronary cusps; heart block signals annular extension and calls for urgent surgery.',
                           'Drug toxicity from the antibiotics', 'A vegetation on the mitral valve', 'Nothing specific: heart block is common in fever'),
                'ct': ct(R(AC), 'coronal')}

    def av_case(pre, title, lead, body, quiz, show=None, hide=None, labels=None, view=None):
        return {'id': f'{pre}-case', 'phase': 'Case', 'title': title, 'lead': lead, 'body': body,
                'view': view or AV_SIDE, 'show': show or [*CH, *ROOT], 'hide': hide or HEART_OFF, 'opacity': {**AV_FAINT, 'lv': 0.2, 'aorta': 0.2},
                'labels': labels or ['aorta', 'lv'], 'ask': quiz, 'ct': ct(R(AC), 'coronal')}

    AV_EXC = [i for i in ('av-rheum', 'av-rheum-edge', 'jet-ar', 'av-vegetation', 'root-abscess', 'rh-mv', 'rh-mv-edge', 'rh-mv-calcium', 'rh-chordae', 'jet-ms') if has(i)]
    # ------------------------------------------------ 1. young man with rheumatic aortic regurgitation
    a1 = av_case('as', 'Case: a young man with rheumatic aortic regurgitation',
                 '<p>A <b>22-year-old</b> matatu driver, breathless on climbing stairs for 6 months. On benzathine penicillin since age 14. Collapsing pulse, BP 140/40. Echo: <b>severe AR</b> (central jet from retracted, thickened cusps), LV end-diastolic 72 mm, <b>end-systolic 54 mm</b>, <b>EF 48%</b>; the mitral valve is thickened with mild MR and no stenosis; tricuspid normal. Coronary angiography not needed at his age.</p>',
                 '<p><b>Operate</b>: he is symptomatic, and the LV is already failing (end-systolic diameter over 50 mm, EF 50% or less). Each alone is a class I indication.</p>'
                 '<p><b>The options.</b> <b>Repair</b>: rarely durable in rheumatic AR (retracted, fibrotic cusps; the disease goes on). <b>Ross</b>: attractive in the young (no warfarin, growth, near-normal survival in non-rheumatic series), but in rheumatic patients, particularly the young and those with mitral disease, the autograft fails more often. '
                 '<b>Tissue valve</b>: degenerates fast in a 22-year-old. <b>Mechanical AVR</b> with warfarin (aortic INR target about 2.5): the usual choice, with an INR plan agreed before surgery.</p>'
                 + ev('ESC/EACTS 2021 and 2025: surgery for symptomatic severe AR, and for asymptomatic severe AR with LVESD over 50 mm, LVESDi over 25 mm/m² or LVEF 50% or less (class I). A best-evidence review (Interact CardioVasc Thorac Surg 2010) found poorer autograft durability after the Ross in young rheumatic patients, more so with mitral disease. Rheumatic aortic repair: Afifi, Hosny and Yacoub, Ann Cardiothorac Surg 2019.'),
                 ask('Which valve suits this 22-year-old man with rheumatic AR and some mitral involvement, who can attend an INR clinic?', 'A mechanical aortic valve',
                     'Durable, and the Ross is less reliable in young rheumatic patients with mitral involvement. A tissue valve would fail early. Warfarin is manageable with a reliable INR plan.',
                     'A Ross operation', 'A bioprosthesis', 'Aortic valve repair'),
                 show=[*CH, *RH_AV, 'jet-ar'], hide=[*HEART_OFF, *CUSPS], labels=['av-rheum', 'jet-ar', 'lv'])
    std_c, std_sq = build_case([
        ('Patho', 'other', [av_patho_rh('as')]), ('Anatomy', 'other', [av_anat('as')]), ('Case', 'other', [a1]),
        ('Sternotomy', 'other', [std[2]]), ('Cannulate', 'artery', [av_cannulate('as', 0)]), ('Clamp', 'artery', [av_clamp('as', 0)]), ('Aortotomy', 'artery', [av_aortotomy('as', 0)]),
        ('Excise', 'fissure', [av_excise('as', 0)]), ('Size', 'other', [av_size('as', 0)]), ('Sutures', 'fissure', [av_sutures('as', 0)]), ('Seat', 'bronchus', [av_seat('as', 0)]),
        ('Close', 'other', [av_close('as', 0)]), ('Wean', 'artery', [av_wean('as', 0)])])
    for s in std_c:
        if s['id'] == 'as-excise':
            s['body'] += '<p><b>In this rheumatic valve</b> the cusps are thick, retracted and fused at one commissure, with little calcium: excision is easier than in calcific AS, but check the anterior mitral leaflet through the valve and the aortomitral curtain.</p>'
        if s['id'] == 'as-seat':
            s['body'] = s['body'].replace('Seat the valve', 'Seat the valve', 1) + '<p>This patient has a <b>mechanical</b> bileaflet valve: check that both leaflets open and close fully and that no suture tail or pledget can catch them.</p>'
    # ------------------------------------------------ 2. double valve: rheumatic mixed aortic disease and mitral stenosis
    a2 = av_case('ad', 'Case: rheumatic mitral stenosis with mixed aortic disease',
                 '<p>A <b>35-year-old woman</b>, NYHA III, in AF. Echo: <b>mitral stenosis</b> (MVA 1.1 cm², thick, calcified commissures) and <b>mixed aortic disease</b>: moderate-to-severe AR and aortic valve area 1.0 cm² with a mean gradient of only 28 mmHg. PA pressure 55 mmHg; mild functional TR, annulus 36 mm.</p>',
                 '<p><b>Both valves</b>: the mitral is unfavourable for a balloon, and the aortic valve is significantly diseased (a balloon would leave it). Low flow through the stenotic mitral makes the aortic gradient understate the stenosis. <b>Double valve replacement</b>, with the appendage closed and AF ablation where possible.</p>'
                 '<p><b>The order at surgery</b>: open the aorta and <b>excise the aortic valve first</b> (it opens the view of the mitral through the LA and lets you see the aortomitral curtain), then replace the <b>mitral</b>, then implant the <b>aortic</b> prosthesis: an aortic valve already in place would block access to the anterior mitral annulus, and mitral sutures pulled against it could distort it.</p>'
                 '<p><b>Valves</b>: she needs warfarin for AF anyway, so two mechanical valves are usual; a tissue option would mean two redos at a young age.</p>'
                 + ev('technique: Kirklin/Barratt-Boyes Cardiac Surgery. The AHA 2020 statement notes that in endemic regions double-valve surgery is common and replacement is usually preferred to limit redo operations.'),
                 ask('In a double valve replacement, in which order are the valves dealt with?', 'Excise the aortic valve, replace the mitral, then implant the aortic prosthesis',
                     'The mitral annulus, especially anteriorly, is reached best with the aortic valve out and no aortic prosthesis in the way; the aortic prosthesis goes in last.',
                     'Implant the aortic prosthesis, then the mitral', 'Mitral first, before opening the aorta at all', 'Either order; it makes no difference'),
                 show=[*CH, *RH_AV, *RH_MV, 'jet-ar', 'jet-ms'], hide=[*HEART_OFF, *CUSPS, *NORMAL_MV], labels=['av-rheum', 'rh-mv', 'jet-ar', 'jet-ms'])
    dv_close = {**av_close('ad', 0), 'title': 'Close the left atrium, then the aorta; de-air'}
    dv_close['body'] = '<p>Close the <b>left atriotomy</b> first (LV vent across the mitral prosthesis until the last suture), then the <b>aortotomy</b>.</p>' + dv_close['body']
    dv_c, dv_sq = build_case([
        ('Patho', 'other', [av_patho_rh('ad', multi=True)]), ('Anatomy', 'other', [av_anat('ad')]), ('Case', 'other', [a2]),
        ('Sternotomy', 'other', [{**mv_sternotomy('ad', 0)}]), ('Cannulate', 'artery', [mv_cannulate('ad', 0)]), ('Clamp', 'artery', [av_clamp('ad', 0)]),
        ('Aortotomy', 'artery', [av_aortotomy('ad', 0)]), ('Excise AV', 'fissure', [av_excise('ad', 0)]),
        ('Atriotomy', 'vein', [mv_la('ad', 0)]), ('Excise MV', 'fissure', [{**mv_excise('ad', 0), 'id': 'ad-mv-excise'}]), ('MV sutures', 'fissure', [{**mv_sutures('ad', 0), 'id': 'ad-mv-sutures'}]),
        ('MV seat', 'bronchus', [{**mv_seat('ad', 0), 'id': 'ad-mv-seat'}]), ('AV size', 'other', [av_size('ad', 0)]), ('AV sutures', 'fissure', [av_sutures('ad', 0)]), ('AV seat', 'bronchus', [av_seat('ad', 0)]),
        ('Close', 'other', [dv_close]), ('Wean', 'artery', [av_wean('ad', 0)])])
    for s in dv_c:
        if s['id'].startswith('ad-mv-') or s['id'] == 'ad-atriotomy': s['_mv'] = True
        if s['id'] in ('ad-av-seat', 'ad-seat', 'ad-close', 'ad-wean'): s['show'] = [*s.get('show', []), 'mv-prosthesis']
    # ------------------------------------------------ 3. older patient, calcific AS, TAVI unsuitable: SAVR by upper hemisternotomy
    a3 = av_case('ah', 'Case: an older patient with calcific aortic stenosis',
                 '<p>A <b>74-year-old woman</b>, retired nurse, exertional chest tightness and a presyncopal episode. Echo: <b>tricuspid calcific AS</b>, AVA 0.7 cm², mean gradient 48 mmHg, EF 60%; no other valve disease. Coronary angiography: no significant disease. '
                 'CT: annulus 20 mm, <b>left main ostium 8 mm above the annulus</b> with shallow sinuses; <b>small, calcified iliofemoral arteries</b> (4.5 mm). Surgical risk low (STS 2.4%).</p>',
                 '<p><b>The default at 74</b> with tricuspid AS and suitable anatomy is <b>TAVI</b> (ESC/EACTS 2025, class I from 70 years). But her anatomy is <b>not</b> suitable: a low left main with shallow sinuses risks <b>coronary obstruction</b> by the displaced native cusp, and the femoral route is poor. The Heart Team recommends <b>SAVR</b>, which also allows the largest valve for a small annulus (avoiding mismatch).</p>'
                 '<p><b>Valve</b>: a <b>bioprosthesis</b> at 74 (no warfarin; valve-in-valve possible later). <b>Access</b>: upper hemisternotomy, since the operation is isolated AVR.</p>'
                 + ev('ESC/EACTS 2025: TAVI (class I) at 70 or over with tricuspid AS and suitable anatomy; SAVR (class I) under 70 at low risk. PARTNER 3 and Evolut Low Risk (NEJM 2019) enrolled patients with suitable anatomy only. Coronary obstruction after TAVI is rare but often fatal; low coronary height (under about 10–12 mm) and shallow sinuses are the main CT predictors (Ribeiro et al., JACC 2013).'),
                 ask('At 74 with severe tricuspid AS, what makes SAVR preferable to TAVI for this patient?', 'Anatomy: a low left main ostium with shallow sinuses (coronary obstruction risk) and poor femoral access',
                     'Age favours TAVI, but the Heart Team decides on anatomy too; when TAVI anatomy is hostile and surgical risk is low, SAVR is the safer choice.',
                     'Her age alone', 'The aortic valve area of 0.7 cm²', 'Chest tightness'),
                 show=[*CH, *CALC_AV, 'ostium-l', 'ostium-r'], hide=HEART_OFF, labels=['av-calcium', 'ostium-l', 'aortic-annulus'], view=root_view)
    hemi_c, hemi_sq = build_case([
        ('Patho', 'other', [av_patho_calc('ah')]), ('Anatomy', 'other', [av_anat('ah')]), ('Case', 'other', [a3]),
        ('Hemisternotomy', 'other', [hemi[2]]), ('Cannulate', 'artery', [hemi[3]]), ('Clamp', 'artery', [hemi[4]]), ('Aortotomy', 'artery', [hemi[5]]), ('Excise', 'fissure', [hemi[6]]),
        ('Size', 'other', [hemi[7]]), ('Sutures', 'fissure', [hemi[8]]), ('Seat', 'bronchus', [hemi[9]]), ('Close', 'other', [hemi[10]]), ('Wean', 'artery', [hemi[11]])])
    # ------------------------------------------------ 4. bicuspid calcific AS under 70: right anterior mini-thoracotomy
    a4 = av_case('ar', 'Case: bicuspid aortic stenosis at 64',
                 '<p>A <b>64-year-old</b> businessman, breathless on exertion. Echo: <b>bicuspid</b> valve (fused right and left cusps with a raphe), heavily calcified; AVA 0.8 cm², mean gradient 55 mmHg, EF 58%. Ascending aorta <b>42 mm</b>. Coronaries normal. '
                 'CT: the ascending aorta lies mostly to the <b>right of the sternum</b>, close behind it. He wants to be back at work quickly and dislikes the idea of warfarin.</p>',
                 '<p><b>SAVR</b>: he is under 70 at low risk (class I), and his valve is <b>bicuspid</b>, the anatomy TAVI trials excluded. <b>The aorta</b> is 42 mm: below the threshold for replacing it at AVR (45 mm or more with a bicuspid valve), so it is left and followed.</p>'
                 '<p><b>Valve</b>: between 60 and 65 the choice is individual; he chooses a <b>bioprosthesis</b>, accepting a likely valve-in-valve later. <b>Access</b>: CT meets the criteria for a <b>right anterior mini-thoracotomy</b>.</p>'
                 + ev('ESC/EACTS 2025: SAVR recommended under 70 at low risk; TAVI may be considered in bicuspid stenosis only at increased surgical risk (class IIb, new). ACC/AHA 2022 aortic guideline: replacing the ascending aorta at AVR is reasonable at 4.5 cm or more with a bicuspid valve.'),
                 ask('Why is SAVR, rather than TAVI, the standard for this 64-year-old?', 'He is under 70 at low surgical risk, and the valve is bicuspid',
                     'Both favour surgery: ESC/EACTS 2025 recommend SAVR under 70 at low risk, and bicuspid valves were excluded from the low-risk TAVI trials.',
                     'TAVI is contraindicated after 60', 'His ascending aorta needs replacing', 'Bioprostheses cannot be implanted by TAVI'),
                 show=[*CH, *CALC_AV], hide=HEART_OFF, labels=['av-calcium', 'aorta'])
    ramt_c, ramt_sq = build_case([
        ('Patho', 'other', [av_patho_calc('ar', bicuspid=True)]), ('Anatomy', 'other', [av_anat('ar')]), ('Case', 'other', [a4]),
        ('Access', 'other', [rm[2]]), ('Clamp', 'artery', [rm[3]]), ('Aortotomy', 'artery', [rm[4]]), ('Excise', 'fissure', [rm[5]]), ('Size', 'other', [rm[6]]),
        ('Sutures', 'fissure', [rm[7]]), ('Seat', 'bronchus', [rm[8]]), ('Close', 'other', [rm[9]]), ('Wean', 'artery', [rm[10]])])
    # ------------------------------------------------ 5. aortic endocarditis with a root abscess
    a5 = av_case('ai', 'Case: aortic endocarditis with a root abscess',
                 '<p>A <b>30-year-old man</b> with known mild rheumatic AR, admitted with fever and rigors; three blood cultures grow <i>Staphylococcus aureus</i> despite 7 days of cloxacillin. The PR interval has lengthened from 180 to <b>300 ms</b>. TOE: a 12 mm vegetation on the non-coronary cusp, a cusp perforation with <b>severe AR</b>, and an <b>echolucent cavity in the aortomitral curtain</b>. He is breathless but not in shock.</p>',
                 '<p><b>Indications</b> (ESC 2023): <b>uncontrolled infection</b> (a paravalvular abscess; persistent bacteraemia on appropriate antibiotics) and severe AR with heart failure: <b>urgent surgery</b>, within days.</p>'
                 '<p><b>The operation</b>: radical debridement of all infected tissue, closure of the abscess cavity (autologous or bovine pericardium), then valve replacement; if the root is destroyed, <b>root replacement</b> (homograft, or a composite graft or stentless root: see the <b>Root</b> module). Pacing wires: heart block may be permanent.</p>'
                 + ev('ESC 2023: urgent surgery for locally uncontrolled infection (abscess, false aneurysm, fistula, enlarging vegetation) and for persistent positive cultures despite appropriate antibiotics. No prosthesis type (mechanical, biological, homograft) has been shown to reduce reinfection; complete debridement matters more than the choice of substitute.'),
                 ask('What makes this an urgent operation rather than completing 6 weeks of antibiotics first?', 'Uncontrolled infection: a paravalvular abscess and persistent bacteraemia, with severe AR',
                     'Abscess and persistent bacteraemia do not resolve on antibiotics alone; waiting risks fistula, complete heart block, embolism and death.',
                     'The vegetation size alone', 'His young age', 'The PR interval alone'),
                 show=[*CH, *CUSPS, 'aortic-annulus', 'av-vegetation', 'root-abscess', 'jet-ar', 'his-bundle'], hide=[*HEART_OFF, 'av-calcium'], labels=['av-vegetation', 'root-abscess', 'his-bundle'])
    RA_ = V(LM['root-abscess']) if 'root-abscess' in LM else AC
    ai_debride = {'id': 'ai-debride', 'phase': 'Valve', 'title': 'Radical debridement of the abscess',
                  'body': '<p>With the cusps out, find the abscess: here in the <b>aortomitral curtain</b> under the non-coronary and left sinuses. Unroof it, then <b>debride to healthy, bleeding tissue</b>, removing all necrotic and infected material; send it for culture, histology and PCR. Irrigate (dilute povidone-iodine or saline); change gloves and instruments.</p>'
                          '<p>Respect what lies close: the <b>His bundle</b> below the right/non-coronary commissure, the <b>anterior mitral leaflet</b> below the curtain, the left main ostium above. Incomplete debridement is the main cause of recurrent infection and paravalvular leak.</p>'
                          + ev('surgical principles from the ESC 2023 guideline and the 2025 EJCTS review of its surgical implications: complete debridement first; the reconstruction is chosen after the extent of destruction is seen.'),
                  'view': open_view, 'show': [*ROOT, 'root-abscess', 'his-bundle', 'mv-ant-leaflet', 'ostium-l', 'ostium-r'], 'hide': [*HEART_OFF, *CUSPS],
                  'opacity': {**AV_FAINT, 'root-abscess': 0.9}, 'highlight': ['root-abscess'], 'danger': ['his-bundle', 'mv-ant-leaflet', 'ostium-l'],
                  'labels': ['root-abscess', 'his-bundle', 'mv-ant-leaflet', 'ostium-l'],
                  'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Debride the abscess', 'port': 'sternotomy', 'remove': ['root-abscess'],
                             'path': [R(RA_ + AN * 4 + AE * 4), R(RA_), R(RA_ + AN * 4 - AE * 4)]},
                  'ask': ask('What is the commonest cause of recurrent infection after surgery for aortic root abscess?', 'Incomplete debridement',
                             'Residual infected tissue seeds the new prosthesis; the extent of debridement, not the choice of valve, determines reinfection.', 'Choosing a mechanical valve', 'Using pledgets', 'Short cross-clamp time'),
                  'ct': ct(R(RA_), 'coronal')}
    ai_patch = {'id': 'ai-patch', 'phase': 'Valve', 'title': 'Close the cavity; patch or replace the root',
                'body': '<p>A <b>localised</b> cavity: close it with a patch of glutaraldehyde-fixed autologous or bovine <b>pericardium</b>, sewn to healthy tissue with running 4-0 polypropylene, so that the new valve sits on the patch and healthy annulus rather than in infected space. Then size and implant the valve as usual, with pledgeted sutures in sound tissue.</p>'
                        '<p><b>Extensive</b> destruction (circumferential abscess, aorto-ventricular discontinuity, fistula): replace the <b>root</b> (an aortic homograft, whose anterior mitral leaflet can patch the curtain, or a composite valved graft or stentless root) with coronary buttons. The <b>Root</b> module shows these steps.</p>'
                        + ev('a homograft offers no proven advantage against reinfection over prosthetic roots (ESC 2023); it is chosen for its handling of destroyed tissue where available. Complete heart block after surgery for abscess often needs a permanent pacemaker; epicardial leads may be placed at the operation.'),
                'view': open_view, 'show': [*ROOT, 'his-bundle', 'mv-ant-leaflet', 'ostium-l', 'ostium-r'], 'hide': [*HEART_OFF, *CUSPS, 'root-abscess'],
                'opacity': AV_FAINT, 'highlight': ['aortic-annulus'], 'danger': ['his-bundle'], 'labels': ['aortic-annulus', 'his-bundle', 'mv-ant-leaflet'],
                'ct': ct(R(AC), 'coronal')}
    ai_c, ai_sq = build_case([
        ('Patho', 'other', [av_patho_ie('ai')]), ('Anatomy', 'other', [av_anat('ai')]), ('Case', 'other', [a5]),
        ('Sternotomy', 'other', [{**std[2], 'id': 'ai-sternotomy'}]), ('Cannulate', 'artery', [av_cannulate('ai', 0)]), ('Clamp', 'artery', [av_clamp('ai', 0)]), ('Aortotomy', 'artery', [av_aortotomy('ai', 0)]),
        ('Excise', 'fissure', [av_excise('ai', 0)]), ('Debride', 'fissure', [ai_debride]), ('Patch', 'other', [ai_patch]), ('Size', 'other', [av_size('ai', 0)]),
        ('Sutures', 'fissure', [av_sutures('ai', 0)]), ('Seat', 'bronchus', [av_seat('ai', 0)]), ('Close', 'other', [av_close('ai', 0)]), ('Wean', 'artery', [av_wean('ai', 0)])])
    for s in ai_c:
        if s['id'] == 'ai-excise':
            s['show'] = [*s.get('show', []), 'av-vegetation', 'root-abscess']; s['labels'] = [*s.get('labels', []), 'av-vegetation']
            s['body'] += '<p><b>Here</b>: lift the vegetation out whole with the non-coronary cusp and send both for culture; do not let fragments fall into the LV.</p>'
        if s['id'] == 'ai-wean':
            s['body'] += '<p><b>This patient</b>: atrioventricular pacing through epicardial wires from the start; if heart block persists, a permanent system once the infection is controlled.</p>'
    AV_CASES = (('avr-std', 'Young man with rheumatic AR (sternotomy)', std_c, std_sq),
                ('avr-dvr', 'Double valve: MS with mixed aortic disease', dv_c, dv_sq),
                ('avr-hemi', 'Older patient, calcific AS, TAVI unsuitable (hemisternotomy)', hemi_c, hemi_sq),
                ('avr-ramt', 'Bicuspid AS at 64 (right anterior mini-thoracotomy)', ramt_c, ramt_sq),
                ('avr-ie', 'Endocarditis with root abscess', ai_c, ai_sq))
    for key, appr, steps_, sq in AV_CASES:
        for s in steps_:
            mvstep = s.pop('_mv', False)
            named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
            off_groups = {'arteries', 'veins', 'airway', 'lul-intra', 'lll-intra', 'rul-intra', 'nodes', 'nerves', 'pleura', 'segments', 'trauma', 'muscles', 'landmarks'}
            opened = s['phase'] not in ('Pathophysiology', 'Anatomy', 'Case') and not s['id'].endswith(('-sternotomy', '-access'))
            s['hide'] = [*s.get('hide', []), *[q['id'] for q in atlas['structures'] if q['group'] in off_groups and q['id'] not in named],
                         *[i for i in ('esophagus', 'thymus', 'thyroid') if has(i) and i not in named], *(['sternum'] if opened else []),
                         *[i for i in AV_EXC if i not in named]]
            if s['phase'] in ('Valve', 'Wean', 'Aorta'): s['opacity'] = {**{c_: 0.3 for c_ in AV_CANS}, **s.get('opacity', {})}
            if s['phase'] in ('Valve', 'Anatomy') and not mvstep:
                s['hide'] = [*s['hide'], 'svc', 'pa-trunk', 'ra', 'rv', 'la', 'myocardium', *[i for i in (*AV_CANS, 'can-svc', 'can-ivc', 'can-ostial') if i not in named]]
                s['opacity'] = {**s.get('opacity', {}), 'lv': 0.12, 'aorta': 0.12, 'lvot': 0.2}
                s['hide'] = [*s['hide'], 'esophagus', *[f'vert-t{i}' for i in range(1, 13)]]
            s['opacity'] = {**{f'vert-t{i}': 0.25 for i in range(2, 11)}, **s.get('opacity', {})}
            for kk in ('highlight', 'danger', 'labels', 'show'):
                if kk in s: s[kk] = [i for i in s[kk] if has(i) or i == 'skin']
        procs[key] = {'id': key, 'op': 'avr', 'opName': 'Aortic valve replacement', 'side': 'both', 'name': 'Aortic valve replacement', 'approach': appr,
                      'summary': 'Case-based: pathophysiology, anatomy, the patient and the Heart Team decision, then the operation (access, bypass and protection, aortotomy, debridement, sizing, sutures, prosthesis, de-airing, TOE).',
                      'ports': [], 'steps': steps_, 'sources': [*RHD_SRC[:3], *AV_SRC, *[x for x in RHD_SRC if 'endocarditis' in x['title'].lower() or 'EASE' in x['title']], *AVSRC], 'group': 'Cardiac', 'sequence': sq}
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
    if has('david-graft'):
        DAV_CUSPS = [i for i in ('av-cusp-r', 'av-cusp-l', 'av-cusp-n') if has(i)]
        dview = clook(AC, AN * 0.55 + V([0.15, 1.0, 0.0]), 130)
        RTSRC.extend([
            {'title': 'David TE, Feindel CM. An aortic valve-sparing operation for patients with aortic incompetence and aneurysm of the ascending aorta. J Thorac Cardiovasc Surg 1992;103:617-22', 'url': pm('David Feindel aortic valve-sparing operation 1992')},
            {'title': 'David TE, et al. Aortic valve sparing operations: outcomes at 20 years. Ann Cardiothorac Surg 2013;2:24-9', 'url': 'https://www.annalscts.com/article/view/1396/html'},
            {'title': 'Mastrobuoni S, et al. Valve-sparing aortic root replacement using the reimplantation (David) technique: systematic review and meta-analysis. Ann Cardiothorac Surg 2023;12:131-9', 'url': 'https://www.annalscts.com/article/view/17025/html'},
            {'title': 'Zhou T, et al. Reimplantation versus remodeling in valve-sparing surgery for aortic root aneurysms: a meta-analysis. J Thorac Dis 2020', 'url': 'https://jtd.amegroups.org/article/view/43518/html'},
            {'title': 'Schäfers HJ, et al. Cusp height in aortic valves. J Thorac Cardiovasc Surg 2013;146:269-74', 'url': 'https://pubmed.ncbi.nlm.nih.gov/22853942/'},
        ])
        david = [
            {**rt_anat('rd'), 'title': 'The functional aortic annulus', 'show': [*CH, 'root-aneurysm', *ROOT, *AV_DANGER], 'hide': [*HEART_OFF, 'av-calcium'],
             'body': '<p>The valve works as a unit with its root: the <b>ventriculo-aortic junction</b> (the base), the <b>sinuses</b>, and the <b>sinotubular junction</b>. Together they are the <b>functional aortic annulus</b>. '
                     'In a root aneurysm with thin, normal cusps, the cusps leak only because the STJ and the base have dilated and pulled the commissures apart (<b>type I</b> regurgitation in El Khoury\'s classification). '
                     'Restore the root\'s size and shape, and the cusps meet again: <b>keep the valve</b>.</p>'
                     + ev('the functional classification of AR (type I dilatation, type II prolapse, type III restriction) guides repair; valve-sparing works best for type I with pliable cusps.'),
             'ask': ask('Root aneurysm 5.3 cm, moderate central AR, thin mobile cusps without calcium. Why does the valve leak?', 'The dilated root (STJ and base) pulls the commissures apart, so the cusps no longer meet',
                        'This is type I (functional annulus dilatation) regurgitation: the cusps are normal, which is exactly the case for keeping them.', 'The cusps are torn', 'The cusps are calcified and restricted')},
            {'id': 'rd-decide', 'phase': 'Decision', 'seq': 1, 'title': 'Who is right for a valve-sparing root?',
             'body': '<p><b>Good candidates</b>: a root aneurysm with <b>normal or near-normal cusps</b>: young patients, <b>Marfan</b> and other heritable aortopathies, some <b>bicuspid</b> valves in experienced hands, and type A dissection with a normal valve. '
                     '<b>Not suitable</b>: calcified, thickened, retracted or badly fenestrated cusps; then a Bentall or Ross.</p>'
                     '<p><b>Reimplantation (David)</b> puts the whole valve inside a graft and fixes the base, so it also treats a dilated annulus. <b>Remodelling (Yacoub)</b> replaces the sinuses with a scalloped graft but leaves the base free (often with an external ring).</p>'
                     + ev('in David\'s 20-year series (374 patients), survival was 69% at 20 years and 97% were free of reoperation at 10 years. A 2023 meta-analysis of reimplantation (44 studies, 7,878 patients) found early mortality 1.6% and 91% freedom from reoperation at 10 years, with no difference between bicuspid and tricuspid valves. '
                          'A meta-analysis of 14 cohort studies (1,672 patients) found lower late mortality and reoperation after reimplantation than after remodelling (retrospective data). No valve means no anticoagulation and very low rates of valve-related events.'),
             'view': clook(RT_, V([0.3, 0.9, 0.4]), 300), 'show': [*CH, 'root-aneurysm'], 'hide': HEART_OFF, 'opacity': AV_FAINT, 'labels': ['root-aneurysm'],
             'ask': ask('A 28-year-old with Marfan syndrome, root 5.0 cm, mild AR, normal cusps. Which operation keeps his own valve and also stabilises the annulus?',
                        'Valve-sparing reimplantation (David)', 'Reimplantation fixes the base inside the graft as well as replacing the sinuses; with normal cusps it avoids a prosthesis and anticoagulation.',
                        'Mechanical Bentall', 'Ross procedure', 'Remodelling without annuloplasty'),
             'ct': ct(R(AC), 'coronal')},
            {**mv_sternotomy('rd', 2), 'id': 'rd-sternotomy'}, rt_cannulate('rd', 3), {**av_clamp('rd', 4), 'id': 'rd-clamp'},
            {'id': 'rd-dissect', 'phase': 'Root', 'seq': 5, 'title': 'Free the root down to the base; excise the sinuses',
             'body': '<p>Transect the aorta above the STJ; inspect the cusps (thin, mobile, no fenestration or calcium: go on). Free the root <b>outside</b> down to the level of the <b>ventriculo-aortic junction</b>: off the RVOT and the roof of the left atrium, as far as the muscle beneath the right cusp allows. '
                     'Excise the sinus walls leaving a <b>4–5 mm rim</b> along the cusp attachments and <b>buttons</b> round the coronaries. Keep the cusps untouched.</p>'
                     + ev('the depth of the external dissection sets how low the graft can go; the right coronary sinus is limited by the muscular septum, which is why the graft lies higher there.'),
             'view': dview, 'show': [*DAV_CUSPS, 'aortic-annulus', 'stj', 'root-aneurysm', *BTN, *AV_DANGER], 'hide': [*HEART_OFF, 'aorta', 'av-calcium'], 'opacity': AV_FAINT,
             'highlight': ['root-aneurysm', *BTN], 'danger': ['ostium-l', 'ostium-r', 'his-bundle'], 'labels': [*BTN, *DAV_CUSPS],
             'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Excise the sinuses, keep the cusps', 'port': 'sternotomy', 'remove': ['root-aneurysm'],
                        'path': [R(AC + (AE * np.cos(t) + V(np.cross(AN, AE)) * np.sin(t)) * (AR_ + 8) + AN * 18) for t in np.linspace(0, 2 * np.pi, 7)]},
             'ct': ct(R(AC), 'coronal')},
            {'id': 'rd-subannular', 'phase': 'Root', 'seq': 6, 'title': 'Subannular sutures in one horizontal plane',
             'body': '<p>Twelve or so <b>horizontal mattress</b> sutures (2-0 polyester, often pledgeted), passed from <b>inside the LVOT to the outside</b>, just below the nadirs of the cusps, all in <b>one horizontal plane</b>. '
                     'Under the <b>right–non-coronary commissure</b> (membranous septum, His bundle) keep them shallow, or through the fibrous tissue; under the right cusp they sit higher, on the muscular septum.</p>'
                     + ev('a single horizontal plane at the base is what fixes the ventriculo-aortic junction and prevents later annular dilatation, the main reason for late failure of valve-sparing repair.'),
             'view': dview, 'show': [*DAV_CUSPS, 'aortic-annulus', 'his-bundle', 'david-subannular', *BTN], 'hide': [*HEART_OFF, 'aorta', 'av-calcium', 'root-aneurysm'], 'opacity': AV_FAINT,
             'highlight': ['david-subannular'], 'danger': ['his-bundle'], 'labels': ['david-subannular', 'his-bundle'],
             'action': {'kind': 'reveal', 'label': 'Place the subannular sutures', 'port': 'sternotomy', 'ids': ['david-subannular']},
             'ct': ct(R(AC), 'coronal')},
            {'id': 'rd-graft', 'phase': 'Root', 'seq': 7, 'title': 'Size the graft; lower it over the valve; tie',
             'body': '<p><b>Size</b> the graft to the valve, not to the aneurysm: several methods are used (the cusp height, the span between commissures, or the annulus plus a few millimetres); commonly 26–32 mm. Too large leaves the cusps without coaptation; too small crowds them.</p>'
                     '<p>Pass the subannular sutures through the base of the graft, lower it over the valve and tie them over the graft: the valve now sits <b>inside</b> the tube.</p>'
                     + ev('no trial compares sizing methods; surgeons follow their method consistently. Residual AR after reimplantation is most often due to a mismatch between graft size and cusp size, or to unrecognised cusp prolapse.'),
             'view': dview, 'show': [*DAV_CUSPS, 'david-subannular', 'david-graft', *BTN], 'hide': [*HEART_OFF, 'aorta', 'av-calcium', 'root-aneurysm'], 'opacity': AV_FAINT,
             'highlight': ['david-graft'], 'labels': ['david-graft'],
             'action': {'kind': 'seat', 'label': 'Lower the graft over the valve', 'port': 'sternotomy', 'ids': ['david-graft'], 'from': R(AN * 55)},
             'ct': ct(R(AC), 'coronal')},
            {'id': 'rd-reimplant', 'phase': 'Root', 'seq': 8, 'title': 'Resuspend the commissures; sew the valve inside the graft',
             'body': '<p>Pull each <b>commissure</b> up vertically inside the graft and fix it with a pledgeted suture at a height that lets the cusps meet. Then sew the <b>rim of sinus wall</b> to the inside of the graft with running <b>4-0 polypropylene</b>, following each cusp\'s scalloped attachment from nadir to commissure.</p>'
                     + ev('commissural height and symmetry determine coaptation; asymmetric resuspension is a common cause of residual regurgitation.'),
             'view': dview, 'show': [*DAV_CUSPS, 'david-subannular', 'david-graft', 'david-commissures', *BTN], 'hide': [*HEART_OFF, 'aorta', 'av-calcium', 'root-aneurysm'], 'opacity': {**AV_FAINT, 'david-graft': 0.35},
             'highlight': ['david-commissures'], 'labels': ['david-commissures'],
             'action': {'kind': 'reveal', 'label': 'Resuspend the commissures', 'port': 'sternotomy', 'ids': ['david-commissures']},
             'ct': ct(R(AC), 'coronal')},
            {'id': 'rd-check', 'phase': 'Root', 'seq': 9, 'title': 'Check the cusps: effective height and prolapse',
             'body': '<p>Look at the valve from above. The free margins should meet at the same level, well above the base. Measure the <b>effective height</b> (from the base of the cusp to its free margin at the centre) with a calliper: a common target is <b>about 9 mm</b>. '
                     'A cusp lying lower than the others is <b>prolapsing</b>: correct it by <b>central plication</b> of its free margin (a fine polypropylene suture), then re-check. A saline test into the graft (clamped above) shows central coaptation.</p>'
                     + ev('effective height as a target (about 9–10 mm) and the importance of the geometric (cusp) height come from the Homburg group (Schäfers et al.) and are widely used in valve repair; they are expert practice rather than trial-based thresholds.'),
             'view': dview, 'show': [*DAV_CUSPS, 'david-graft', 'david-commissures', *BTN], 'hide': [*HEART_OFF, 'aorta', 'av-calcium', 'root-aneurysm'], 'opacity': {**AV_FAINT, 'david-graft': 0.3},
             'labels': DAV_CUSPS,
             'ask': ask('After reimplantation, one cusp\'s free margin sits 3 mm lower than the other two, and the effective height is 6 mm. What next?', 'Central plication of the prolapsing cusp, then re-measure',
                        'A low-lying cusp is prolapsing; shortening its free margin (central plication) restores coaptation. Leaving it means residual AR and early failure.', 'Accept it: it will settle', 'Convert to a Bentall immediately'),
             'ct': ct(R(AC), 'coronal')},
            {**rt_buttons('rd', 10, into='david-graft'), 'body': rt_buttons('rd', 10)['body'].replace('Cut a hole in the graft (cautery) opposite each ostium.', 'Cut holes in the graft opposite each coronary ostium (the valve is already inside).')},
            {**rt_distal('rd', 11, into='david-graft'), 'title': 'Distal anastomosis; TOE of the repaired valve',
             'body': '<p>Join the graft to the ascending aorta (running 4-0 polypropylene). After bypass, <b>TOE</b>: no more than <b>trivial</b> AR, coaptation well <b>above the base</b> of the root, no cusp prolapse, normal gradients, and regional wall motion (buttons). '
                     'More than mild AR at this point: go back and repair, or replace the valve.</p>'
                     + ev('residual AR more than mild on intraoperative TOE predicts late failure of valve-sparing repair in large series; that is why it is corrected before leaving theatre.')},
        ]
        procs['root-david'] = {'steps': david, 'appr': 'Valve-sparing reimplantation (David)',
                               'sq': seq(('Anatomy', 'other'), ('Decide', 'other'), ('Sternotomy', 'other'), ('Cannulate', 'artery'), ('Clamp', 'artery'), ('Dissect', 'fissure'), ('Sutures', 'fissure'), ('Graft', 'bronchus'), ('Reimplant', 'bronchus'), ('Check', 'other'), ('Buttons', 'artery'), ('Distal', 'artery'))}
    for key in [k for k in ('root-bentall', 'root-ross', 'root-david') if k in procs and 'appr' in procs[k]]:
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
        k_ = vignette.find('<p class="evidence">'); lead_, rest_ = (vignette[:k_], vignette[k_:]) if k_ >= 0 else (vignette, '')
        return {'id': f'{pre}-case', 'phase': 'Case', 'seq': 0, 'title': title, 'lead': lead_,
                'body': rest_ or '<p>Read the case above; the steps that follow are this patient\'s operation.</p>', 'view': clook(HC_, V([-0.3, 1, 0.2]), 260), 'spin': True, 'show': [*SURF, *TREE, *lesions], 'hide': CB_OFF, 'opacity': SOLID,
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
# ==================================================================================================== thoracic case scenarios
# each operation opens with the disease (pathophysiology), then its anatomy, then a patient (the case and the decision), then
# the operation: lung cancer and its staging, post-TB lung and aspergilloma, congenital lobar emphysema and CPAM, myasthenia
# gravis and thymoma, oesophageal cancer, empyema; and a new operation for post-pneumonectomy empyema and bronchopleural fistula
TX_OK = has('tumour-rul') and has('tumour-lul')
if TX_OK:
    ev = lambda t: f'<p class="evidence"><b>Evidence:</b> {t}</p>'
    pm = lambda term: 'https://pubmed.ncbi.nlm.nih.gov/?term=' + term.replace(' ', '+')
    chain = lambda *xs: '<div class="chain">' + '<i>→</i>'.join(f'<span class="hot">{x[1:]}</span>' if x.startswith('!') else f'<span>{x}</span>' for x in xs) + '</div>'
    tl = lambda tgt, d, dist=300: {'eye': R(V(tgt) + V(d) / np.linalg.norm(V(d)) * dist), 'target': R(tgt)}
    Pt = lambda k: V(LM[k]) if k in LM else V(S[k]['centroid'])
    PATH_IDS = [s_['id'] for s_ in atlas['structures'] if s_['group'] == 'pathology']
    LOBE_IDS = [i for i in ('lul', 'lll', 'rul', 'rml', 'rll', 'fissure', 'fissure-h', 'fissure-r') if has(i)]
    NODES = [i for i in S if S[i]['group'] == 'nodes']
    LATD = {'left': (-1, -0.25, 0.25), 'right': (1, -0.25, 0.25)}

    def add_case(key, patho, case, sources=()):
        """insert a pathophysiology step at the start and a case step after the anatomy; renumber the sequence strip"""
        if key not in procs: return
        p = procs[key]; st = [dict(s_) for s_ in p['steps']]          # steps can be shared between approaches: renumber copies
        a = min((s['seq'] for s in st if s['phase'] in ('Anatomy',) and s.get('seq') is not None), default=-1)
        for s in st:
            if s.get('seq') is not None: s['seq'] = s['seq'] + (1 if s['seq'] <= a else 2)
        last_anat = max((i for i, s in enumerate(st) if s['phase'] == 'Anatomy'), default=-1)
        pat = {**patho, 'id': f'{key}-patho', 'seq': 0, 'askAfter': True}
        cas = {**case, 'id': f'{key}-case', 'seq': a + 2}
        p['steps'] = [pat, *st[:last_anat + 1], cas, *st[last_anat + 1:]]
        sq = list(p.get('sequence') or [])
        sq.insert(0, {'label': 'Patho', 'kind': 'other'}); sq.insert(a + 2, {'label': 'Case', 'kind': 'other'})
        p['sequence'] = sq
        p['sources'] = [*sources, *[x for x in p.get('sources', []) if x not in sources]]
        for s in p['steps']:
            named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
            s['hide'] = [*s.get('hide', []), *[i for i in PATH_IDS if i not in named]]
            if s['phase'] in ('Pathophysiology', 'Case'):          # the disease, uncluttered: no intrapulmonary branches, faint spine
                s['hide'] += [i for i in S if (S[i]['group'].endswith('-intra') or S[i]['group'] == 'segments') and i not in named]
                s['opacity'] = {**{f'vert-t{i}': 0.18 for i in range(1, 13)}, **s.get('opacity', {})}
            for kk in ('highlight', 'danger', 'labels', 'show', 'hide'):
                if kk in s: s[kk] = [i for i in s[kk] if has(i) or i == 'skin']

    def case(title, lead, body, quiz, view, show, hide=(), labels=(), opacity=None, danger=()):
        return {'phase': 'Case', 'title': title, 'lead': lead, 'body': body, 'ask': quiz, 'view': view, 'show': list(show), 'hide': list(hide),
                'labels': list(labels), 'danger': list(danger), 'opacity': opacity or {}, 'ct': ct(R(view['target']), 'axial', 'lung')}

    # ------------------------------------------------------------------------------------------ sources
    LC_SRC = [
        {'title': 'Rami-Porta R, et al. The IASLC Lung Cancer Staging Project: proposals for revision of the TNM stage groups in the forthcoming (ninth) edition of the TNM classification for lung cancer. J Thorac Oncol 2024;19:1007-27', 'url': 'https://www.jto.org/article/S1556-0864(24)00079-0/fulltext'},
        {'title': 'Klug M, et al. The ninth edition of TNM staging for lung cancer: what radiologists need to know. RadioGraphics 2024;44:e240057', 'url': 'https://pubs.rsna.org/doi/10.1148/rg.240057'},
        {'title': 'Brunelli A, Kim AW, Berger KI, et al. Physiologic evaluation of the patient with lung cancer being considered for resectional surgery. ACCP guidelines. Chest 2013;143(5 Suppl):e166S-e190S', 'url': 'https://journal.chestnet.org/article/S0012-3692(13)60294-9/fulltext'},
        {'title': 'De Leyn P, et al. Revised ESTS guidelines for preoperative mediastinal lymph node staging for non-small-cell lung cancer. Eur J Cardiothorac Surg 2014;45:787-98', 'url': pm('De Leyn revised ESTS guidelines preoperative mediastinal lymph node staging 2014')},
        {'title': 'Saji H, et al. Segmentectomy versus lobectomy in small-sized peripheral non-small-cell lung cancer (JCOG0802/WJOG4607L). Lancet 2022;399:1607-17', 'url': 'https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)02333-3/abstract'},
        {'title': 'Altorki N, et al. Lobar or sublobar resection for peripheral stage IA non-small-cell lung cancer (CALGB 140503). N Engl J Med 2023;388:489-98', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa2212083'},
        {'title': 'Lim E, et al. Video-assisted thoracoscopic versus open lobectomy in patients with early-stage lung cancer: the VIOLET RCT. Health Technol Assess 2022;26(48)', 'url': 'https://www.ncbi.nlm.nih.gov/books/NBK587651/'},
        {'title': 'Tsuboi M, et al. Overall survival with osimertinib in resected EGFR-mutated NSCLC (ADAURA). N Engl J Med 2023;389:137-47', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa2304594'},
        {'title': 'Forde PM, et al. Neoadjuvant nivolumab plus chemotherapy in resectable lung cancer (CheckMate 816). N Engl J Med 2022;386:1973-85', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa2202170'},
        {'title': 'Daniels J, et al. Lung cancer at Korle-Bu Teaching Hospital, Ghana. ecancermedicalscience 2025', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12221260'},
    ]
    TB_SRC = [
        {'title': 'Denning DW, et al. Chronic pulmonary aspergillosis: rationale and clinical guidelines for diagnosis and management (ESCMID/ERS). Eur Respir J 2016;47:45-68', 'url': 'https://publications.ersnet.org/content/erj/47/1/45'},
        {'title': 'Akbari JG, et al. Clinical profile and surgical outcome for pulmonary aspergilloma: a single center experience. Ann Thorac Surg 2005;80:1067-72', 'url': 'https://www.sciencedirect.com/science/article/abs/pii/S0003497505005382'},
        {'title': 'Kim YT, et al. Surgical treatment of pulmonary aspergilloma. Ann Thorac Surg 2005;79:294-8', 'url': 'https://www.sciencedirect.com/science/article/abs/pii/S0003497504011920'},
        {'title': 'Panda A, Bhalla AS, Goyal A. Bronchial artery embolization in hemoptysis: a systematic review. Diagn Interv Radiol 2017;23:307-17', 'url': 'https://dirjournal.org/articles/bronchial-artery-embolization-in-hemoptysis-a-systematic-review/dir.2017.16454'},
        {'title': 'Migliori GB, et al. Clinical standards for the assessment, management and rehabilitation of post-TB lung disease. Int J Tuberc Lung Dis 2021;25:797-813', 'url': 'https://scienceportal.msf.org/api/assets/7348/download/13290'},
        {'title': 'Ivanova O, et al. Lung function testing and prediction equations in adult population with a history of tuberculosis: a systematic review and meta-analysis. Eur Respir Rev 2023;32:220221', 'url': 'https://publications.ersnet.org/content/errev/32/168/220221'},
        {'title': 'Kim YT, et al. Pneumonectomy for tuberculous destroyed lung. Eur J Cardiothorac Surg 2003;23:833-9', 'url': 'https://academic.oup.com/ejcts/article/23/5/833/407385'},
    ]
    CONG_SRC = [
        {'title': 'Mukhtar S, Sharma S, Trovela DA. Congenital lobar emphysema. StatPearls 2024', 'url': 'https://www.ncbi.nlm.nih.gov/books/NBK560602/'},
        {'title': 'Congenital lobar emphysema: anaesthetic considerations. OpenAnesthesia 2025', 'url': 'https://www.openanesthesia.org/keywords/congenital-lobar-emphysema/'},
        {'title': 'Crombleholme TM, et al. Cystic adenomatoid malformation volume ratio predicts outcome in prenatally diagnosed CCAM. J Pediatr Surg 2002;37:331-8', 'url': 'https://www.sciencedirect.com/science/article/abs/pii/S0022346802749269'},
        {'title': 'Congenital pulmonary airway malformation. StatPearls', 'url': 'https://www.statpearls.com/point-of-care/20208'},
        {'title': 'Dehner LP, et al. Congenital pulmonary airway malformations with a reconsideration and current classification. Pediatr Dev Pathol 2023', 'url': 'https://dx.doi.org/10.1177/10935266221146823'},
    ]
    MG_SRC = [
        {'title': 'Gilhus NE. Myasthenia gravis. N Engl J Med 2016;375:2570-81', 'url': 'https://www.nejm.org/doi/abs/10.1056/NEJMra1602678'},
        {'title': 'Gilhus NE, et al. Myasthenia gravis. Nat Rev Dis Primers 2019;5:30', 'url': 'https://www.nature.com/articles/s41572-019-0079-y'},
        {'title': 'Wolfe GI, et al. Long-term effect of thymectomy plus prednisone versus prednisone alone in myasthenia gravis: 2-year extension of the MGTX trial. Lancet Neurol 2019;18:259-68', 'url': 'https://www.sciencedirect.com/science/article/abs/pii/S1474442218303922'},
        {'title': 'Narayanaswami P, et al. International consensus guidance for management of myasthenia gravis: 2020 update. Neurology 2021;96:114-22', 'url': 'https://ern-euro-nmd.eu/publication/international-consensus-guidance-for-management-of-myasthenia-gravis-2020-update/'},
        {'title': 'Sanders DB, et al. International consensus guidance for management of myasthenia gravis. Neurology 2016;87:419-25', 'url': 'https://www.neurology.org/doi/10.1212/WNL.0000000000002790'},
        {'title': 'Leuzzi G, et al. Prediction of postoperative myasthenic crisis after thymectomy. Eur J Cardiothorac Surg 2014;45:e76', 'url': 'https://academic.oup.com/ejcts/article/45/4/e76/362404'},
        {'title': 'Detterbeck FC, et al. The Masaoka-Koga stage classification for thymic malignancies. J Thorac Oncol 2011;6(7 Suppl 3):S1710-6', 'url': 'https://www.iccr-cancer.org/datasets/docs/iccr-thymic-stage/'},
        {'title': 'College of American Pathologists. Protocol for thymic epithelial tumours (AJCC/UICC 9th edition TNM), v5.0', 'url': 'https://documents.cap.org/protocols/Thymus_5.0.0.0.REL.CAPCP.pdf'},
        {'title': 'Friedant AJ, et al. Minimally invasive versus open thymectomy for thymic malignancies: systematic review and meta-analysis. J Thorac Oncol 2016;11:30-8', 'url': 'https://www.sciencedirect.com/science/article/pii/S1556086415000106'},
        {'title': 'Lee Y, et al. Minimally invasive vs open thymectomy for myasthenia gravis: meta-analysis. Surg Endosc 2023;37:3321-39', 'url': 'https://link.springer.com/article/10.1007/s00464-022-09757-y'},
    ]
    ESO_SRC = [
        {'title': 'Middleton DRS, et al. Alcohol consumption and oesophageal squamous cell cancer risk in east Africa (ESCCAPE). Lancet Glob Health 2022;10:e236-45', 'url': 'https://pure.qub.ac.uk/en/publications/alcohol-consumption-and-oesophageal-squamous-cell-cancer-risk-in-'},
        {'title': 'Middleton DRS, et al. Hot beverages and oesophageal cancer risk in western Kenya: findings from the ESCCAPE case-control study. Int J Cancer 2019;144:2669-76', 'url': 'https://pure.qub.ac.uk/en/publications/hot-beverages-and-oesophageal-cancer-risk-in-western-kenya-findin'},
        {'title': 'Oesophageal cancer in young patients, Bomet District (Tenwek), Kenya. Lancet 2002;360:462-3', 'url': 'https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(02)09639-3/abstract'},
        {'title': 'Rice TW, Patil DT, Blackstone EH. 8th edition AJCC/UICC staging of cancers of the esophagus and esophagogastric junction. Ann Cardiothorac Surg 2017;6:119-30', 'url': 'https://www.annalscts.com/article/view/14237/pdf'},
        {'title': 'van Hagen P, et al. Preoperative chemoradiotherapy for esophageal or junctional cancer (CROSS). N Engl J Med 2012;366:2074-84', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa1112088'},
        {'title': 'Shapiro J, et al. Neoadjuvant chemoradiotherapy plus surgery versus surgery alone for oesophageal cancer (CROSS long-term). Lancet Oncol 2015;16:1090-8', 'url': 'https://www.sciencedirect.com/science/article/abs/pii/S1470204515000406'},
        {'title': 'Hoeppner J, et al. Perioperative chemotherapy or preoperative chemoradiotherapy in esophageal cancer (ESOPEC). N Engl J Med 2025', 'url': 'https://www.nejm.org/doi/abs/10.1056/NEJMoa2409408'},
        {'title': 'Kelly RJ, et al. Adjuvant nivolumab in resected esophageal or gastroesophageal junction cancer (CheckMate 577). N Engl J Med 2021;384:1191-203', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa2032125'},
        {'title': 'Biere SS, et al. Minimally invasive versus open oesophagectomy (TIME). Lancet 2012;379:1887-92', 'url': pm('Biere minimally invasive versus open oesophagectomy TIME Lancet 2012')},
        {'title': 'Mariette C, et al. Hybrid minimally invasive esophagectomy for esophageal cancer (MIRO). N Engl J Med 2019;380:152-62', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa1805101'},
        {'title': 'Stahl M, et al. Chemoradiation with and without surgery in locally advanced squamous cell carcinoma of the esophagus. J Clin Oncol 2005;23:2310-7', 'url': pm('Stahl chemoradiation with and without surgery squamous esophagus 2005')},
        {'title': 'Homs MY, et al. Single-dose brachytherapy versus metal stent placement for the palliation of dysphagia (SIREC). Lancet 2004;364:1497-504', 'url': pm('Homs single-dose brachytherapy versus metal stent SIREC 2004')},
    ]
    EMP_SRC = [
        {'title': 'Roberts ME, Rahman NM, Maskell NA, et al. British Thoracic Society guideline for pleural disease. Thorax 2023;78:1143-56', 'url': 'https://www.brit-thoracic.org.uk/about-us/news/2023/british-thoracic-society-publishes-a-guideline-and-clinical-statement-on-pleural-disease/'},
        {'title': 'Davies HE, Davies RJO, Davies CWH. Management of pleural infection in adults: BTS pleural disease guideline 2010. Thorax 2010;65(Suppl 2):ii41-53', 'url': pm('Davies management of pleural infection in adults BTS 2010')},
        {'title': 'Maskell NA, et al. U.K. controlled trial of intrapleural streptokinase for pleural infection (MIST1). N Engl J Med 2005;352:865-74', 'url': pm('Maskell intrapleural streptokinase pleural infection MIST1 2005')},
        {'title': 'Rahman NM, et al. A clinical score (RAPID) to identify those at risk for poor outcome at presentation in patients with pleural infection. Chest 2014;145:848-55', 'url': 'https://discovery.ucl.ac.uk/id/eprint/1430484/'},
        {'title': 'Corcoran JP, et al. Prospective validation of the RAPID clinical risk prediction score (PILOT). Eur Respir J 2020;56:2000130', 'url': 'https://publications.ersnet.org/content/erj/56/5/2000130'},
        {'title': 'Pan H, et al. VATS versus open thoracotomy decortication for empyema: a meta-analysis. J Thorac Dis 2017;9:2006-14', 'url': 'https://jtd.amegroups.org/article/view/14673/11890'},
        {'title': 'Mwesige M, et al. Management and outcomes of thoracic empyema at Mulago National Referral Hospital, Uganda. BMC Pulm Med 2025', 'url': 'https://bmcpulmmed.biomedcentral.com/articles/10.1186/s12890-025-03861-0'},
        {'title': 'Vorster MJ, et al. Tuberculous pleural effusions: advances and controversies. J Thorac Dis 2015;7:981-91', 'url': 'https://jtd.amegroups.org/article/view/4221/4848'},
    ]
    PPE_SRC = [
        {'title': 'Deschamps C, et al. Empyema and bronchopleural fistula after pneumonectomy: factors affecting incidence. Ann Thorac Surg 2001;72:243-8', 'url': 'https://www.annalsthoracicsurgery.org/article/S0003-4975(01)02681-9/fulltext'},
        {'title': 'Wali A, Billè A. Complications of thoracic surgery: post-pneumonectomy bronchopleural fistula. Shanghai Chest 2021;5:3', 'url': 'https://shc.amegroups.org/article/view/5883/html'},
        {'title': 'Lois M, Noppen M. Bronchopleural fistulas: an overview of the problem with special focus on endoscopic management. Chest 2005;128:3955-65', 'url': 'https://www.sciencedirect.com/science/article/abs/pii/S0012369215496400'},
        {'title': 'Gritsiuta AI, Stovall A, Petrov RV. Surgical strategies in the management of postoperative bronchopleural fistula: a narrative review. AME Surg J 2025;5:10', 'url': 'https://asj.amegroups.org/article/view/99260/html'},
        {'title': 'Clagett OT, Geraci JE. A procedure for the management of postpneumonectomy empyema. J Thorac Cardiovasc Surg 1963;45:141-5', 'url': pm('Clagett Geraci procedure management postpneumonectomy empyema 1963')},
        {'title': 'Pairolero PC, et al. Postpneumonectomy empyema: the role of intrathoracic muscle transposition. J Thorac Cardiovasc Surg 1990;99:958-68', 'url': pm('Pairolero postpneumonectomy empyema intrathoracic muscle transposition 1990')},
        {'title': 'Zaheer S, et al. Postpneumonectomy empyema: results after the Clagett procedure. Ann Thorac Surg 2006;82:279-87', 'url': pm('Zaheer postpneumonectomy empyema Clagett procedure 2006')},
        {'title': 'Schneiter D, et al. Accelerated treatment of postpneumonectomy empyema: a binational long-term study. J Thorac Cardiovasc Surg 2008;136:179-85', 'url': 'https://www.sciencedirect.com/science/article/pii/S0022522308004236'},
        {'title': 'Di Maio M, et al. A meta-analysis of the impact of bronchial stump coverage on the risk of bronchopleural fistula after pneumonectomy. Eur J Cardiothorac Surg 2015;48:196-200', 'url': 'https://academic.oup.com/ejcts/article/48/2/196/445759'},
        {'title': 'Symbas PN, et al. Modified Eloesser flap. Ann Thorac Surg 1971;12:69-78', 'url': pm('Symbas Eloesser flap 1971')},
    ]

    # ------------------------------------------------------------------------------------------ lung cancer
    TNM = ('<table class="mini"><tr><th>T</th><th>Tumour (9th edition keeps the 8th-edition T)</th></tr>'
           '<tr><td>T1</td><td>≤3 cm, surrounded by lung (T1a ≤1, T1b >1–2, T1c >2–3 cm)</td></tr>'
           '<tr><td>T2</td><td>>3–5 cm (T2a >3–4, T2b >4–5), or main bronchus (not carina), visceral pleura, collapse to the hilum</td></tr>'
           '<tr><td>T3</td><td>>5–7 cm, or chest wall, phrenic nerve, parietal pericardium, a separate nodule in the same lobe</td></tr>'
           '<tr><td>T4</td><td>>7 cm, or mediastinum, heart, great vessels, trachea, carina, oesophagus, diaphragm, vertebra, a nodule in another ipsilateral lobe</td></tr>'
           '<tr><th>N</th><th>Nodes</th></tr>'
           '<tr><td>N1</td><td>ipsilateral hilar or intrapulmonary (stations 10–14)</td></tr>'
           '<tr><td>N2a / N2b</td><td>ipsilateral mediastinal or subcarinal: <b>one station</b> (N2a) or <b>several stations</b> (N2b), new in the 9th edition</td></tr>'
           '<tr><td>N3</td><td>contralateral mediastinal or hilar, or any scalene or supraclavicular</td></tr>'
           '<tr><th>M</th><th>Metastasis</th></tr>'
           '<tr><td>M1a–c</td><td>M1a pleural or pericardial spread, contralateral nodules; M1b one extrathoracic metastasis; <b>M1c1</b> several in one organ system, <b>M1c2</b> several organ systems (new)</td></tr></table>')

    def lc_patho(side, lobe, focus=''):
        tum = f'tumour-{lobe}' if has(f'tumour-{lobe}') else f'tumour-central-{side[0]}'
        return {'phase': 'Pathophysiology', 'title': 'Pathophysiology and staging: lung cancer',
                'body': '<p><b>In Kenya and across Africa</b> lung cancer usually presents late: in series from Ghana and West Africa three-quarters or more are stage III–IV at diagnosis. A cough, weight loss and a shadow are often treated first as <b>tuberculosis</b>, frequently without bacteriological confirmation; a smear- or GeneXpert-negative "TB" that does not improve needs a CT and a tissue diagnosis.</p>'
                        '<p><b>How it arises.</b> Carcinogens (tobacco above all; also biomass smoke, occupational exposures, radon) cause accumulating mutations in the airway epithelium. <b>Adenocarcinoma</b> (commonest, peripheral, and in never-smokers often driven by <b>EGFR</b> or <b>ALK</b> alterations) and <b>squamous cell carcinoma</b> (central, smokers) make up most non-small-cell cancer. Small-cell cancer is rarely surgical.</p>'
                        '<p><b>How it spreads</b> decides the stage and the operation:</p>'
                        + chain('Primary tumour (T: size, invasion)', 'Intrapulmonary and hilar nodes (N1)', '!Mediastinal nodes (N2)', '!Contralateral or supraclavicular (N3)')
                        + chain('Primary tumour', '!Blood: brain, bone, adrenal, liver (M1)')
                        + TNM
                        + '<p><b>Before an operation</b>: CT and <b>PET-CT</b>; <b>invasive mediastinal staging</b> (EBUS/EUS needle aspiration, or mediastinoscopy) when the tumour is central, over 3 cm, or the nodes are enlarged or PET-positive; brain imaging for stage II and above. Then fitness: FEV1 and DLCO, and the <b>predicted postoperative</b> values (ppo = preoperative value × segments remaining / 19; by lobe: RUL 3, RML 2, RLL 5, LUL 5 with the lingula, LLL 4).</p>'
                        + focus
                        + ev('9th-edition TNM from the IASLC (Rami-Porta et al., J Thorac Oncol 2024): N2 split into N2a (single station) and N2b (multiple stations); M1c into M1c1 and M1c2; T1N1 moves to stage IIA, T1N2a is IIB. ACCP 2013 physiological evaluation (Brunelli et al.): ppoFEV1 and ppoDLCO both over 60% is low risk; 30–60% needs a stair climb (over 22 m) or shuttle walk (over 400 m); under 30%, or a poor walk test, needs CPET (VO2max over 20 mL/kg/min low risk, under 10 high risk). ESTS 2014 guideline for invasive mediastinal staging (De Leyn et al.).'),
                'view': tl(Pt(tum), LATD[side], 320), 'spin': True,
                'show': [*LOBE_IDS, tum, *NODES, 'trachea'], 'hide': [], 'opacity': {**{i: 0.28 for i in LOBE_IDS}},
                'highlight': [tum], 'labels': [tum, *[n for n in NODES if n.endswith(side[0]) or n in ('ln-7', 'ln-5', 'ln-6', 'ln-4r')][:6]],
                'ask': ask('Under the 9th edition, a 2.6 cm tumour (T1c) with metastasis in a single mediastinal station (subcarinal, station 7) and no distant spread is stage…', 'IIB (T1 N2a)',
                           'The 9th edition splits N2: a single station (N2a) with a T1 tumour is IIB; several stations (N2b) make it IIIA. Many single-station N2 patients are now treated with neoadjuvant chemo-immunotherapy and surgery.',
                           'IIIA', 'IIIB', 'IV'),
                'ct': ct(R(Pt(tum)), 'axial', 'lung')}

    def lc_case(side, lobe, which):
        tum = f'tumour-{lobe}'
        v = tl(Pt(tum), LATD[side], 300)
        show = [*LOBE_IDS, tum, *NODES]; op_ = {i: 0.3 for i in LOBE_IDS}
        C_ = {
            'lul': case('Case: an EGFR-positive adenocarcinoma in a never-smoker',
                        '<p>A <b>58-year-old woman</b>, never a smoker, cooked over a wood fire for 30 years. Six months of cough; given anti-TB treatment twice at a health centre without a positive sputum test. CT: a <b>3.4 cm</b> mass in the left upper lobe, no enlarged nodes; biopsy: <b>adenocarcinoma, EGFR exon 19 deletion</b>. PET: no nodal or distant uptake. EBUS: stations 4L, 7 and 10L negative. FEV1 78%, DLCO 72% predicted.</p>',
                        '<p><b>Stage</b>: T2a (3–4 cm) N0 M0 = <b>IB</b>. <b>Why EBUS with a negative PET?</b> A tumour over 3 cm (or central, or cN1) carries enough risk of occult N2 to justify invasive staging. '
                        '<b>Fitness</b>: the left upper lobe (with the lingula) has 5 of the 19 segments: ppoFEV1 = 78 × 14/19 ≈ 57%, ppoDLCO = 72 × 14/19 ≈ 53%; both are in the 30–60% band, and she climbs three flights (over 22 m) without stopping: fit for lobectomy.</p>'
                        '<p><b>Plan</b>: VATS left upper lobectomy with systematic nodal dissection, then adjuvant <b>osimertinib</b> for the EGFR mutation.</p>'
                        + ev('VIOLET (HTA 2022; 503 patients): VATS lobectomy gave better physical function at 5 weeks and fewer in-hospital complications than open lobectomy, with no loss of nodal upstaging. ADAURA (NEJM 2023): adjuvant osimertinib after resection of EGFR-mutant stage IB–IIIA disease improved 5-year overall survival (88% vs 78%, HR 0.49).'),
                        ask('Why did she need EBUS when the PET showed no nodal uptake?', 'A tumour over 3 cm carries a significant risk of occult mediastinal nodes; guidelines advise invasive staging for tumours over 3 cm, central tumours or cN1',
                            'PET misses small nodal deposits. ESTS 2014 recommends invasive staging for central tumours, tumours over 3 cm, or suspected N1 even when PET is negative.',
                            'EBUS is required before every lobectomy', 'To confirm the EGFR mutation', 'Because she had been treated for TB'),
                        v, show, labels=[tum, 'ln-5', 'ln-6', 'ln-7', 'ln-10l'], opacity=op_),
            'lll': case('Case: a squamous carcinoma with a hilar node',
                        '<p>A <b>66-year-old man</b>, smoker (45 pack-years). A <b>4.6 cm</b> squamous cell carcinoma in the left lower lobe; PET: uptake in an <b>interlobar node (station 11L)</b>, mediastinum clear; EBUS of stations 4L and 7 negative; no distant disease. FEV1 70%, DLCO 64%.</p>',
                        '<p><b>Stage</b>: T2b (4–5 cm) N1 M0 = <b>IIB</b>. Resectable by lobectomy (a sleeve if the node is fused to the lower lobe bronchus origin).</p>'
                        '<p><b>Sequence</b>: for stage II–IIIA disease, <b>neoadjuvant chemo-immunotherapy</b> (platinum doublet plus nivolumab, three cycles) before surgery is now an evidence-based option where available; otherwise surgery then adjuvant chemotherapy.</p>'
                        + ev('CheckMate 816 (NEJM 2022; 358 patients, stage IB–IIIA): three cycles of nivolumab plus chemotherapy before surgery gave a pathological complete response in 24% vs 2.2% with chemotherapy alone, and longer event-free survival (median 31.6 vs 20.8 months).'),
                        ask('What stage is a 4.6 cm tumour with an interlobar (station 11) node and a negative mediastinum?', 'IIB (T2b N1 M0)',
                            'T2b is over 4 up to 5 cm; station 11 is N1 (intrapulmonary/hilar). T2 N1 is stage IIB.', 'IIA', 'IIIA', 'IB'),
                        v, show, labels=[tum, 'ln-11l', 'ln-10l', 'ln-7'], opacity=op_),
            'rul': case('Case: a small peripheral adenocarcinoma: segment or lobe?',
                        '<p>A <b>54-year-old woman</b>. An incidental <b>1.8 cm</b> part-solid nodule in the right upper lobe (consolidation-to-tumour ratio 0.8), growing over 6 months; PET: mild uptake, nodes clear. It lies <b>across the plane between the apical (S1) and anterior (S3) segments</b>, 12 mm from it. FEV1 92%.</p>',
                        '<p><b>Stage</b>: cT1b N0 = <b>IA2</b>. For a peripheral tumour of 2 cm or less with confirmed node-negative disease, an anatomical <b>segmentectomy</b> is now equivalent or better than lobectomy for survival. But it needs a <b>margin</b> at least as wide as the tumour (2 cm, or the tumour diameter). Straddling the S1/S3 plane, a single segment would not give that margin: a bisegmentectomy or, as here, a <b>lobectomy</b>.</p>'
                        + ev('JCOG0802/WJOG4607L (Lancet 2022; tumours ≤2 cm, C/T ratio >0.5): 5-year overall survival 94.3% after segmentectomy vs 91.1% after lobectomy (HR 0.66), with more local recurrence (10.5% vs 5.4%). CALGB 140503 (NEJM 2023; ≤2 cm, node-negative on frozen section): sublobar resection was non-inferior for disease-free survival (63.6% vs 64.1%).'),
                        ask('Which trial showed better overall survival with segmentectomy than lobectomy for peripheral tumours of 2 cm or less?', 'JCOG0802/WJOG4607L',
                            '5-year OS 94.3% vs 91.1%, attributed to preserved lung function and fewer deaths from other causes, despite more local recurrence.', 'CALGB 140503', 'VIOLET', 'ADAURA'),
                        v, show, labels=[tum, 'ln-4r', 'ln-10r', 'ln-7'], opacity=op_),
            'rml': case('Case: is he fit? Predicting lung function',
                        '<p>A <b>71-year-old man</b>, ex-smoker with COPD, a <b>2.6 cm</b> adenocarcinoma in the middle lobe, cT1c N0 (EBUS negative). <b>FEV1 55%</b>, <b>DLCO 50%</b> predicted. He walks 2 km a day.</p>',
                        '<p><b>Predicted postoperative values</b>: the middle lobe has <b>2</b> of the 19 segments. ppoFEV1 = 55 × (19 − 2)/19 ≈ <b>49%</b>; ppoDLCO = 50 × 17/19 ≈ <b>45%</b>. Both lie between 30 and 60%: a <b>low-technology exercise test</b> decides. He climbs 25 m of stairs without stopping: <b>proceed</b> (middle lobectomy, VATS).</p>'
                        + ev('ACCP 2013: ppoFEV1 and ppoDLCO >60% low risk; either 30–60%: stair climb >22 m or shuttle walk >400 m is satisfactory, otherwise CPET; either <30%: CPET, with VO2max <10 mL/kg/min (or <35% predicted) high risk.'),
                        ask('FEV1 55% and DLCO 50%; a middle lobectomy removes 2 of 19 segments. What next?', 'Both ppo values are 30–60%: do a stair climb or shuttle walk test',
                            'ppoFEV1 ≈ 49% and ppoDLCO ≈ 45%. In the 30–60% band, a simple exercise test (stairs >22 m, shuttle >400 m) separates those who can proceed from those who need CPET.',
                            'Operate: both are over 40%', 'He is inoperable: DLCO is under 60%', 'Go straight to pneumonectomy work-up'),
                        v, show, labels=[tum, 'rml', 'ln-10r'], opacity=op_),
            'rll': case('Case: single-station N2 disease',
                        '<p>A <b>60-year-old man</b>, smoker. A <b>5.8 cm</b> squamous carcinoma in the right lower lobe; PET: uptake in the <b>subcarinal node (station 7)</b> only; EBUS: station 7 positive, 4R and 4L negative; brain MRI clear. FEV1 80%.</p>',
                        '<p><b>Stage</b>: T3 (5–7 cm) <b>N2a</b> (one mediastinal station) M0 = <b>IIIA</b>. N2 disease is treated with <b>multimodality</b> therapy: neoadjuvant chemo-immunotherapy, then restaging and lobectomy with nodal dissection if the disease responds and a lobectomy suffices; or definitive chemoradiotherapy followed by durvalumab. Multi-station (N2b) or bulky N2 favours the non-surgical route. The <b>Tumour Board</b> decides.</p>'
                        + ev('9th-edition staging (IASLC 2024): T3 N2a is IIIA; T3 N2b is IIIB. CheckMate 816: neoadjuvant nivolumab plus chemotherapy improved pCR (24% vs 2.2%) and event-free survival in resectable IB–IIIA disease; about two-thirds of patients were stage IIIA.'),
                        ask('What distinguishes N2a from N2b in the 9th edition?', 'The number of mediastinal stations involved: one (N2a) or several (N2b)',
                            'The count is by station, not by the number of nodes, and it carries prognostic weight: T3 N2a is IIIA, T3 N2b IIIB.',
                            'Node size over 1 cm', 'Ipsilateral versus contralateral nodes', 'PET uptake intensity'),
                        v, show, labels=[tum, 'ln-7', 'ln-4r', 'ln-9r'], opacity=op_, danger=['ln-7']),
        }
        return C_[lobe]

    def seg_case(lobe, segname):
        tum = f'tumour-{lobe}'
        return case(f'Case: a small peripheral tumour: {segname}',
                    f'<p>A <b>63-year-old woman</b> with a <b>1.6 cm</b> solid-predominant adenocarcinoma in the {segname}, well inside the segment (margin to the intersegmental plane over 2 cm). PET: nodes clear; <b>FEV1 62%</b> (COPD).</p>',
                    '<p><b>Stage</b>: cT1b N0 = IA2. An anatomical <b>segmentectomy</b> with sampling of hilar and mediastinal nodes (frozen section: if a node is positive, convert to lobectomy). It saves lung she needs, and in JCOG0802 it gave better overall survival than lobectomy.</p>'
                    '<p>The margin must be at least 2 cm or the tumour\'s diameter; the intersegmental plane is found by inflation–deflation or indocyanine green after the segmental artery is divided.</p>'
                    + ev('JCOG0802 (Lancet 2022): 5-year OS 94.3% segmentectomy vs 91.1% lobectomy; local recurrence 10.5% vs 5.4%. CALGB 140503 (NEJM 2023): sublobar resection non-inferior for DFS and OS; FEV1 about 2 percentage points better at 6 months.'),
                    ask('During segmentectomy the frozen section of a hilar node (station 12) is positive. What now?', 'Convert to lobectomy with systematic nodal dissection',
                        'Both trials required node-negative disease; N1 disease needs a lobectomy (and adjuvant or perioperative systemic therapy).', 'Continue the segmentectomy', 'Close and refer for radiotherapy'),
                    tl(Pt(tum), LATD['left'], 280), [*LOBE_IDS, tum, *NODES], labels=[tum, 'ln-10l', 'ln-11l'], opacity={i: 0.3 for i in LOBE_IDS})

    # ------------------------------------------------------------------------------------------ post-TB lung, aspergilloma
    def tb_patho(destroyed=False):
        tgt = Pt('aspergilloma') if not destroyed and 'aspergilloma' in LM else Pt('lul')
        return {'phase': 'Pathophysiology', 'title': 'Pathophysiology: post-tuberculous lung ' + ('destruction' if destroyed else 'and aspergilloma'),
                'body': '<p><b>Cured is not healed.</b> Up to half of people who complete TB treatment are left with lung damage (post-TB lung disease): cavities, bronchiectasis, fibrosis and pleural thickening, with obstruction, restriction or both.</p>'
                        + chain('Caseous necrosis in the upper lobe', 'Liquefaction, discharged through a bronchus', 'Cavity', 'Healing by fibrosis: thick wall, traction bronchiectasis')
                        + chain('Cavity + bronchial and non-bronchial systemic arteries hypertrophy', '!Haemoptysis (from the systemic circulation, at systemic pressure)')
                        + chain('Cavity colonised by <i>Aspergillus</i>', 'Fungal ball (aspergilloma)', '!Erosion of the vascular wall → haemoptysis', 'Chronic cavitary aspergillosis if it progresses')
                        + ('<p><b>The destroyed lung</b>: a whole lung reduced to cavities, bronchiectasis and fibrosis, contracted, with the pleura fused to the chest wall. It is a reservoir of infection (TB, non-tuberculous mycobacteria, <i>Aspergillus</i>, bacteria), a source of recurrent haemoptysis, and it contributes no gas exchange, often only shunt.</p>' if destroyed else '')
                        + '<p><b>Simple aspergilloma</b>: a single cavity with a fungal ball, few symptoms, no progression over 3 months. <b>Chronic cavitary pulmonary aspergillosis</b>: one or more cavities that enlarge or multiply over months, with symptoms and a positive <i>Aspergillus</i> IgG; treated with long-term oral azoles. Surgery is for simple aspergilloma, and for complex disease with haemoptysis once medically optimised.</p>'
                        '<p><b>Massive haemoptysis</b> kills by asphyxia, not blood loss. First: lie the patient <b>bleeding side down</b>, secure the airway (a large tube, selective intubation of the good side or a bronchial blocker), then <b>bronchial artery embolisation</b> as a bridge; operate once the bleeding has settled and the patient is optimised.</p>'
                        + ev('post-TB lung disease: Migliori et al. clinical standards (Int J Tuberc Lung Dis 2021): up to 50% have problems after treatment; a meta-analysis (Ivanova et al., Eur Respir Rev 2023; 14,621 people) found mean FEV1 77% predicted with obstruction in 22% and restriction in 23%. ESCMID/ERS guideline (Denning et al., Eur Respir J 2016): excise simple aspergilloma if technically possible. Bronchial artery embolisation stops haemoptysis in 70–99%, but it recurs in 10–57% (Panda et al., 2017).'),
                'view': tl(tgt, (1, 0.6, 0.45) if not destroyed else LATD['left'], 260), 'spin': True,
                'show': [*LOBE_IDS, *(['asp-cavity', 'asp-ball', 'asp-pleura'] if not destroyed else ['tb-cavities-l'])], 'opacity': {i: 0.25 for i in LOBE_IDS},
                'highlight': ['asp-ball'] if not destroyed else ['tb-cavities-l'], 'labels': ['asp-cavity', 'asp-ball', 'asp-pleura'] if not destroyed else ['tb-cavities-l', 'lul', 'lll'],
                'ask': ask('Where does the blood come from in haemoptysis from a post-TB cavity?', 'Hypertrophied bronchial and non-bronchial systemic arteries, at systemic pressure',
                           'That is why it can be massive, and why bronchial (and intercostal, phrenic) artery embolisation controls it; the pulmonary artery is the source in a minority (Rasmussen aneurysm).',
                           'The pulmonary veins', 'The fungal ball itself', 'Capillaries in the cavity wall only'),
                'ct': ct(R(tgt), 'axial', 'lung')}

    asp_case = case('Case: aspergilloma with haemoptysis',
                    '<p>A <b>42-year-old man</b>, treated for pulmonary TB 8 years ago (cured). Three episodes of haemoptysis in 2 months, the last about 300 mL, controlled by <b>bronchial artery embolisation</b> 10 days ago. CT: a <b>thick-walled right apical cavity with a mobile fungal ball</b> (air crescent), the rest of the lung nearly normal; <i>Aspergillus</i> IgG positive. FEV1 72%. Sputum smear and GeneXpert negative.</p>',
                    '<p><b>Simple aspergilloma</b> in a fit patient with recurrent haemoptysis: <b>resection</b> (right upper lobectomy). Embolisation bought time; bleeding recurs in a large proportion. Exclude active TB first.</p>'
                    '<p><b>Why open</b> (or experienced VATS only): dense, vascular apical adhesions; an <b>extrapleural</b> plane may be needed; bleeding from the chest wall collaterals; the cavity must not be entered (spillage). An antifungal (voriconazole) around surgery is reasonable if spillage is likely. Plan a <b>muscle flap</b> (serratus or intercostal) if a residual space is expected.</p>'
                    + ev('surgical series: Akbari et al. (Mayo, 2005): no deaths or major complications after resection of simple aspergilloma vs 4.3% mortality and 26% major complications for complex disease; Kim et al. (Korea, 2005): mortality 1.1%, morbidity 27%. ESCMID/ERS 2016: excise simple aspergilloma if technically possible.'),
                    ask('After successful bronchial artery embolisation, why operate on this simple aspergilloma?', 'Haemoptysis often recurs after embolisation, and resection of a simple aspergilloma is curative with low risk',
                        'Recurrence after embolisation ranges from 10% to over 50%; a simple aspergilloma in a fit patient is best removed, electively, once bleeding has settled.', 'Embolisation is curative; surgery is not needed', 'To obtain tissue for TB culture only', 'Only if itraconazole fails for 2 years'),
                    tl(Pt('aspergilloma') if 'aspergilloma' in LM else Pt('rul'), (1, 0.6, 0.45), 240), [*LOBE_IDS, 'asp-cavity', 'asp-ball', 'asp-pleura'],
                    labels=['asp-cavity', 'asp-ball'], opacity={i: 0.25 for i in LOBE_IDS})
    destroyed_case = case('Case: a TB-destroyed left lung',
                          '<p>A <b>29-year-old woman</b>, treated twice for TB (the second time for multidrug-resistant TB, now culture-negative after treatment). Recurrent haemoptysis and purulent sputum; CT: the <b>left lung destroyed</b> (cavities, bronchiectasis, volume loss, pleural thickening), the right lung clear. Perfusion scan: left lung 8% of total. FEV1 1.4 L (48%).</p>',
                          '<p><b>Pneumonectomy</b> removes a lung that adds almost nothing to gas exchange (ppoFEV1 ≈ 1.4 × 0.92 ≈ 1.3 L) and is the source of her symptoms. It is still a high-risk operation: dense adhesions (extrapleural dissection), bleeding, and the highest risk of <b>bronchopleural fistula</b> and <b>empyema</b> of any pneumonectomy (benign, infected, often malnourished). Plan: nutrition first, culture conversion, a short bronchial stump <b>covered with a flap</b>.</p>'
                          + ev('Kim et al. (Eur J Cardiothorac Surg 2003; 94 pneumonectomies for TB-destroyed lung): mortality 1.1%, bronchopleural fistula 7.5%, empyema 15.9%; low FEV1, positive sputum after surgery and aspergilloma predicted fistula.'),
                          ask('Why does a perfusion scan matter before this pneumonectomy?', 'It shows how little the destroyed lung contributes, so the predicted postoperative function is close to the current function',
                              'ppoFEV1 is calculated from the fraction of perfusion to the lung left behind; here 92% of perfusion goes to the right lung.', 'To look for pulmonary emboli', 'To locate the bleeding vessel', 'It is not needed'),
                          tl(Pt('lul'), LATD['left'], 360), [*LOBE_IDS, 'tb-cavities-l'], labels=['tb-cavities-l'], opacity={i: 0.25 for i in LOBE_IDS})

    # ------------------------------------------------------------------------------------------ congenital: CLE vs CPAM
    def cong_patho(which):
        tgt = Pt('lul') if which == 'cle' else Pt('lll')
        return {'phase': 'Pathophysiology', 'title': 'Pathophysiology: congenital lobar emphysema versus CPAM',
                'body': '<table class="mini"><tr><th></th><th>Congenital lobar emphysema (CLE)</th><th>Congenital pulmonary airway malformation (CPAM)</th></tr>'
                        '<tr><td>What it is</td><td>A normal-structured lobe that <b>over-distends</b>: deficient bronchial cartilage (or compression) makes a <b>ball valve</b>; air enters, cannot leave</td><td>A <b>hamartomatous</b> lesion of cysts and abnormal airways, usually one lobe; blood supply from the pulmonary artery</td></tr>'
                        '<tr><td>Where</td><td>Left upper lobe (about 43%), middle lobe (32%), right upper (21%); lower lobes rare</td><td>Any lobe; lower lobes often</td></tr>'
                        '<tr><td>When</td><td>Neonatal respiratory distress; half at birth, most by 6 months</td><td>Most found on antenatal ultrasound; some present with infection later</td></tr>'
                        '<tr><td>Imaging</td><td>A hyperlucent lobe <b>with vascular markings</b>, the other lobes compressed, the mediastinum shifted</td><td>Air-filled cysts of varying size; solid in microcystic types</td></tr>'
                        '<tr><td>Risks</td><td>Progressive compression, tension physiology</td><td>Infection; hydrops in the fetus (CVR over 1.6); malignancy (mucinous adenocarcinoma with type 1; type 4 now regarded as cystic pleuropulmonary blastoma)</td></tr></table>'
                        + chain('Deficient bronchial cartilage', 'Airway collapses in expiration (ball valve)', 'Air trapping, lobe over-distends', '!Compresses the other lobes, shifts the mediastinum', '!Respiratory distress, falling venous return')
                        + '<p><b>The trap</b>: CLE looks like a tension pneumothorax. A chest drain into an emphysematous lobe makes a large air leak and can kill; look for lung markings in the lucent area before inserting one.</p>'
                        '<p><b>Anaesthesia for CLE</b>: avoid nitrous oxide (it expands the lobe) and high positive-pressure ventilation before the chest is open (spontaneous breathing or gentle ventilation); the surgeon scrubbed at induction, ready to open the chest and deliver the lobe.</p>'
                        '<p><b>Stocker types of CPAM</b>: 0 (acinar dysplasia, lethal), 1 (large cysts over 2 cm, 50–70%), 2 (small cysts, associated anomalies), 3 (solid-appearing, alveolar), 4 (peripheral cysts; now considered cystic pleuropulmonary blastoma).</p>'
                        + ev('CLE: StatPearls 2024 (lobe distribution, presentation, pneumothorax pitfall) and OpenAnesthesia 2025 (anaesthetic management). CPAM: CPAM volume ratio over 1.6 predicted hydrops in 75% (Crombleholme et al., J Pediatr Surg 2002); resection of asymptomatic lesions at 6–12 months versus surveillance remains debated (StatPearls). Pathology update: Dehner et al., Pediatr Dev Pathol 2023.'),
                'view': tl(tgt, LATD['left'], 330), 'spin': True,
                'show': [*LOBE_IDS, 'cle-lul' if which == 'cle' else 'cpam-lll'], 'hide': ['lul'] if which == 'cle' else [], 'opacity': {i: 0.25 for i in LOBE_IDS},
                'highlight': ['cle-lul' if which == 'cle' else 'cpam-lll'], 'labels': ['cle-lul' if which == 'cle' else 'cpam-lll', 'lll' if which == 'cle' else 'lul'],
                'ask': ask('A 3-week-old with tachypnoea has a hyperlucent left upper zone and mediastinal shift. Vascular markings are visible in the lucent area. What must you avoid?', 'Inserting a chest drain for a presumed pneumothorax',
                           'Vascular markings mean over-distended lung, not free air: this is CLE. A drain would enter the lobe and cause a large air leak. The treatment is lobectomy.',
                           'A CT scan', 'Oxygen', 'Surgical consultation'),
                'ct': ct(R(tgt), 'axial', 'lung')}

    cle_case = case('Case: a neonate with congenital lobar emphysema',
                    '<p>A <b>5-week-old boy</b>, increasing tachypnoea and feeding difficulty; SpO₂ 90% in air. Chest X-ray: a <b>hyperlucent left upper zone</b> with faint vascular markings, the left lower lobe compressed, the mediastinum pushed to the right. CT: an over-distended left upper lobe; no mass or vascular sling compressing the bronchus.</p>',
                    '<p>Symptomatic CLE: <b>left upper lobectomy</b> (in a neonate by thoracotomy through the 4th or 5th space, or thoracoscopy in experienced hands). Mild, stable cases can be observed.</p>'
                    '<p><b>In theatre</b>: gentle or spontaneous ventilation until the chest is open; no nitrous oxide. Once the chest is open the lobe <b>herniates</b> out of the incision and the child improves at once. Then the hilum as in the adult: the lingular and upper lobe arteries, the superior pulmonary vein, the upper lobe bronchus.</p>'
                    '<p><i>The model shows an adult chest; the neonatal anatomy is the same in arrangement, much smaller in scale.</i></p>'
                    + ev('StatPearls 2024: lobectomy for symptomatic CLE; conservative follow-up for mild cases. OpenAnesthesia 2025: avoid N₂O, minimise positive pressure, surgeon ready at induction.'),
                    ask('At induction the child desaturates and becomes hypotensive with bag ventilation. Best immediate action?', 'Open the chest quickly and let the lobe decompress out of the wound',
                        'Positive pressure inflates the trapped lobe further (tension physiology). Opening the chest decompresses it at once; this is why the surgeon is scrubbed at induction.', 'Increase the ventilation pressure', 'Give nitrous oxide', 'Insert a chest drain'),
                    tl(Pt('lul'), LATD['left'], 330), [*LOBE_IDS, 'cle-lul'], hide=['lul'], labels=['cle-lul', 'lll'], opacity={i: 0.25 for i in LOBE_IDS})
    cpam_case = case('Case: an infected CPAM in a child',
                     '<p>A <b>6-year-old girl</b>, three admissions for "left lower lobe pneumonia" in a year. Antenatal scans were not done. CT after treatment: <b>multiple air-filled cysts up to 3 cm</b> in the left lower lobe, the cyst walls thick; the arterial supply from the pulmonary artery (no systemic feeder).</p>',
                     '<p>A symptomatic, recurrently infected <b>type 1 CPAM</b>: <b>left lower lobectomy</b> once the infection has settled (a segmentectomy only if the lesion is small and clearly confined). Complete excision matters: type 1 lesions carry a risk of mucinous adenocarcinoma, especially if incompletely removed. Check the CT for a systemic artery (a hybrid lesion with sequestration): an unseen feeder from the aorta in the inferior ligament bleeds.</p>'
                     + ev('StatPearls (CPAM): resection is indicated for symptomatic lesions; for asymptomatic ones, elective resection at 6–12 months versus surveillance is debated. Malignancy association: Dehner et al., Pediatr Dev Pathol 2023.'),
                     ask('Before dividing the inferior pulmonary ligament in a lower lobe cystic lesion, what must the CT be checked for?', 'A systemic arterial feeder from the aorta (a hybrid lesion or sequestration)',
                         'Sequestrations and hybrid lesions are supplied from the aorta, often through the inferior ligament; an unrecognised feeder retracts into the abdomen when cut.', 'A pulmonary vein anomaly', 'An enlarged subcarinal node', 'A pericardial cyst'),
                     tl(Pt('lll'), LATD['left'], 330), [*LOBE_IDS, 'cpam-lll'], labels=['cpam-lll', 'lll'], opacity={i: 0.3 for i in LOBE_IDS})

    # ------------------------------------------------------------------------------------------ myasthenia gravis and thymoma
    TH_C = Pt('thymoma') if 'thymoma' in LM else Pt('thymus')
    th_view = tl(TH_C, (0, 1, 0.25), 300)
    TH_SHOW = ['thymus', 'thymoma', 'lbcv', 'svc', 'aorta', 'heart', 'n-phrenic', 'n-phrenic-r', 'pa-trunk']
    mg_patho = {'phase': 'Pathophysiology', 'title': 'Pathophysiology: myasthenia gravis and the thymus',
                'body': '<p><b>An antibody attack on the neuromuscular junction.</b> In about 85% of generalised myasthenia, IgG1/IgG3 antibodies against the <b>acetylcholine receptor (AChR)</b> bind the endplate, fix <b>complement</b> and destroy the postsynaptic folds; fewer receptors means a smaller endplate potential, which fails with repeated firing: <b>fatigable weakness</b>. Other subtypes: anti-MuSK (IgG4, no complement; the thymus is normal), anti-LRP4, and seronegative.</p>'
                        + chain('Thymus: myoid cells express AChR', 'Germinal centres: autoreactive B cells (thymic follicular hyperplasia)', 'Anti-AChR antibodies', '!Complement destroys the endplate', 'Fatigable weakness: eyes, bulbar, limbs, breathing')
                        + '<p><b>Why remove the thymus</b>: in early-onset AChR-positive disease, about 70% of thymuses show germinal-centre hyperplasia, a factory for the antibodies. Removing it (all of it, including ectopic thymic fat) lowers the drive. <b>Thymoma</b> occurs in about 10–20% of patients with myasthenia, and roughly 20–25% of thymoma patients have myasthenia: every thymoma is removed.</p>'
                        '<p><b>Crisis</b>: respiratory or bulbar failure (MGFA class V: intubation). Triggers: infection, surgery, certain drugs (aminoglycosides, fluoroquinolones, magnesium, some anaesthetic agents), steroid initiation. A falling vital capacity warns before the gases change.</p>'
                        '<p><b>MGFA classes</b>: I ocular only; II mild, III moderate, IV severe generalised (a: limb and axial, b: oropharyngeal and respiratory); V intubated.</p>'
                        + ev('mechanisms and subtypes: Gilhus, NEJM 2016 and Nat Rev Dis Primers 2019; Fichtner et al., Front Immunol 2020 (germinal centres in about 70% of early-onset AChR MG). MG in thymoma: Lucchi et al., Eur J Cardiothorac Surg 2009.'),
                'view': th_view, 'spin': True, 'show': TH_SHOW, 'hide': [], 'opacity': {'heart': 0.35, 'aorta': 0.5, 'thymus': 0.8},
                'highlight': ['thymus'], 'labels': ['thymus', 'thymoma', 'lbcv', 'n-phrenic', 'n-phrenic-r'],
                'ask': ask('Why is thymectomy not recommended for anti-MuSK myasthenia?', 'The thymus is usually normal in MuSK disease, and the antibodies (IgG4) are not driven by thymic germinal centres',
                           'Thymic hyperplasia is a feature of AChR-positive, early-onset disease; guidance (2020) finds no evidence of benefit in MuSK MG.', 'MuSK patients are too weak for surgery', 'It is recommended for all subtypes', 'The thymus is always malignant in MuSK MG'),
                'ct': ct(R(TH_C), 'axial')}
    th_cases = {
        'sternotomy': case('Case: myasthenia gravis with a thymoma',
                           '<p>A <b>47-year-old man</b>: ptosis, diplopia, then dysarthria and difficulty swallowing over 4 months; <b>AChR antibodies positive</b>. CT: a <b>5 cm</b> smooth, lobulated anterior mediastinal mass, abutting but with a fat plane to the pericardium and left brachiocephalic vein. On pyridostigmine; vital capacity 2.6 L.</p>',
                           '<p><b>Resect the thymoma with the whole thymus and surrounding fat</b> (extended thymectomy), en bloc and without breaching the capsule: complete (R0) resection is the strongest prognostic factor. A median sternotomy gives the safest R0 resection for a 5 cm tumour, with the phrenic nerves in view and the option to take pericardium, lung or the innominate vein if invaded (send frozen sections).</p>'
                           '<p><b>Prepare the myasthenia first</b>: bulbar weakness is a risk for postoperative crisis, so give <b>IVIG or plasma exchange</b> before surgery; avoid long-acting relaxants.</p>'
                           '<p><b>Staging</b>: Masaoka-Koga (I encapsulated; IIa microscopic, IIb macroscopic capsular invasion into fat; III neighbouring organs; IVa pleural or pericardial, IVb distant) and the 9th-edition TNM (T1a ≤5 cm and T1b >5 cm confined to thymus or fat; T2 pericardium, lung or phrenic; T3 innominate vein, SVC, chest wall; T4 aorta, pulmonary artery, myocardium, trachea, oesophagus).</p>'
                           + ev('International consensus (Sanders et al., Neurology 2016): IVIG or plasma exchange before surgery in patients with significant bulbar dysfunction. Leuzzi et al. (2014): postoperative crisis in 12.4%; generalised, bulbar-predominant disease raised the risk. Staging: Masaoka-Koga (Detterbeck et al., J Thorac Oncol 2011) and the 9th-edition TNM (CAP protocol).'),
                           ask('Before thymectomy, this patient with bulbar weakness should receive…', 'IVIG or plasma exchange to optimise him and reduce the risk of postoperative crisis',
                               'Bulbar and respiratory weakness predict crisis; preoperative immunomodulation is recommended. Operate when stable.', 'A high loading dose of steroids the day before', 'Nothing: surgery improves the myasthenia', 'Neostigmine infusion only'),
                           th_view, TH_SHOW, labels=['thymoma', 'lbcv', 'n-phrenic', 'n-phrenic-r'], opacity={'heart': 0.4, 'aorta': 0.5}, danger=['n-phrenic', 'n-phrenic-r', 'lbcv']),
        'rvats': case('Case: generalised AChR-positive myasthenia, no thymoma',
                      '<p>A <b>28-year-old teacher</b>, generalised myasthenia for 14 months (MGFA IIa: ptosis, arm and leg fatigue, no bulbar symptoms), <b>AChR antibodies positive</b>; CT: normal-sized thymus, no thymoma. On pyridostigmine and prednisolone 30 mg; weakness recurs when the steroid is tapered.</p>',
                      '<p><b>Thymectomy</b>: early in the disease, for AChR-positive generalised non-thymomatous myasthenia in adults up to about 50 (MGTX included 18 to 65). The aim is less steroid, fewer relapses and admissions, and a chance of remission; benefit accrues over months to years. A minimally invasive <b>extended</b> thymectomy (right VATS, subxiphoid or robotic) removes the whole thymus and the fat from phrenic to phrenic, from the thyroid to the diaphragm.</p>'
                      + ev('MGTX (Wolfe et al., NEJM 2016; 126 patients): at 3 years, thymectomy lowered the time-weighted QMG score (6.15 vs 8.99), the prednisone requirement (44 vs 60 mg alternate days) and admissions for exacerbation (9% vs 37%). The benefit persisted at 5 years (Lancet Neurol 2019). 2020 international guidance: thymectomy should be considered early for AChR-positive generalised MG aged 18–50; minimally invasive thymectomy has a good safety record in experienced centres. Meta-analysis (Lee et al., Surg Endosc 2023): no difference in remission between minimally invasive and open thymectomy.'),
                      ask('In MGTX, which outcome improved with thymectomy plus prednisone versus prednisone alone?', 'Clinical score, prednisone dose and admissions for exacerbation, all at 3 years',
                          'QMG 6.15 vs 8.99; alternate-day prednisone 44 vs 60 mg; hospitalisation 9% vs 37%. The effect lasted to 5 years.', 'Only the antibody titre', 'Mortality', 'Nothing: the trial was negative'),
                      th_view, TH_SHOW, hide=['thymoma'], labels=['thymus', 'n-phrenic-r', 'svc'], opacity={'heart': 0.4, 'aorta': 0.5}),
        'subx': case('Case: a young woman with bulbar myasthenia',
                     '<p>A <b>19-year-old student</b>, 8 months of generalised myasthenia with <b>nasal speech and choking on fluids</b> (MGFA IIIb), AChR antibodies positive; CT: thymic hyperplasia. Vital capacity 2.1 L (55% predicted).</p>',
                     '<p><b>Thymectomy is indicated</b> (young, AChR-positive, generalised). <b>Not yet</b>: bulbar weakness and a reduced vital capacity mean a high risk of postoperative crisis. Optimise first: <b>plasma exchange or IVIG</b>, pyridostigmine adjusted, infections treated, then operate when stable. A <b>subxiphoid</b> approach avoids an intercostal incision and gives a symmetric view of both phrenic nerves and both cervical horns.</p>'
                     + ev('International consensus (2016): IVIG or plasma exchange before surgery with significant bulbar dysfunction. Leuzzi et al. (Eur J Cardiothorac Surg 2014; 177 patients): postoperative crisis in 12.4%, higher with Osserman IIB and III–IV disease.'),
                     ask('Which feature most raises her risk of myasthenic crisis after thymectomy?', 'Bulbar weakness with a reduced vital capacity',
                         'Bulbar-predominant, more severe disease predicts crisis; optimise with IVIG or plasma exchange, and extubate only when strength allows.', 'Her young age', 'Thymic hyperplasia on CT', 'Being AChR-positive'),
                     th_view, TH_SHOW, hide=['thymoma'], labels=['thymus', 'n-phrenic', 'n-phrenic-r'], opacity={'heart': 0.4, 'aorta': 0.5}),
    }

    # ------------------------------------------------------------------------------------------ oesophageal cancer
    ESO_T = {'mid': 'eso-tumour-mid', 'low': 'eso-tumour-low'}
    def eso_patho(level):
        tum = ESO_T[level]; tgt = Pt(tum) if tum in LM else Pt('esophagus')
        return {'phase': 'Pathophysiology', 'title': 'Pathophysiology and staging: oesophageal cancer',
                'body': '<p><b>East Africa carries one of the world\'s highest rates of oesophageal squamous cell carcinoma</b>, along a corridor from Ethiopia and Kenya to Malawi. About nine in ten cases here are <b>squamous</b>, and patients are young: in Bomet (Tenwek) 11% were 30 or younger. Risk factors studied in Kenya: <b>very hot tea</b>, alcohol (especially home-brewed spirits), tobacco, household smoke, and poor oral health. In high-income countries <b>adenocarcinoma</b> of the lower third and junction (from reflux and Barrett\'s oesophagus) predominates.</p>'
                        + chain('Chronic thermal and chemical injury to the squamous mucosa', 'Dysplasia', 'Invasive squamous carcinoma', 'Circumferential growth', '!Progressive dysphagia (solids, then liquids), weight loss')
                        + chain('No serosa: early spread', 'Lymphatics along the length of the oesophagus (neck to coeliac)', '!Airway (mid third): tracheo-oesophageal fistula')
                        + '<table class="mini"><tr><th>T (AJCC 8th)</th><th></th></tr>'
                        '<tr><td>T1a / T1b</td><td>lamina propria or muscularis mucosae / submucosa</td></tr><tr><td>T2</td><td>muscularis propria</td></tr><tr><td>T3</td><td>adventitia</td></tr>'
                        '<tr><td>T4a / T4b</td><td>pleura, pericardium, azygos, diaphragm, peritoneum (resectable) / aorta, vertebra, trachea (unresectable)</td></tr>'
                        '<tr><th>N</th><th>by number of nodes: N1 1–2, N2 3–6, N3 7 or more</th></tr></table>'
                        '<p>Separate clinical, pathological and post-neoadjuvant (yp) stage groups; squamous and adenocarcinoma are grouped differently. <b>Work-up</b>: endoscopy and biopsy, CT chest and abdomen, PET-CT for curative candidates, EUS for T and N, <b>bronchoscopy</b> for tumours at or above the carina, staging laparoscopy for junctional adenocarcinoma.</p>'
                        '<p><b>Most patients here present with advanced disease</b>: palliation of dysphagia (a self-expanding metal stent works fastest; brachytherapy lasts longer) is the commonest intervention.</p>'
                        + ev('ESCCAPE Kenya/Tanzania/Malawi (Middleton et al., Lancet Glob Health 2022): alcohol accounted for 65% of cases in Kenyan men; very hot drinks OR 3.7 in western Kenya (Int J Cancer 2019). Young patients: Lancet 2002 (Bomet). Staging: Rice et al., Ann Cardiothorac Surg 2017. SIREC (Lancet 2004): stents relieved dysphagia faster, brachytherapy gave more dysphagia-free days (115 vs 82) with fewer complications.'),
                'view': tl(tgt, (0.9, -0.5, 0.2), 300), 'spin': True,
                'show': ['esophagus', tum, 'trachea', 'aorta', 'azygos', 'heart', 'br-left-main', 'br-right-main', 'thoracic-duct'], 'hide': [*LOBE_IDS], 'opacity': {'heart': 0.25, 'aorta': 0.5, 'esophagus': 0.6},
                'highlight': [tum], 'labels': [tum, 'trachea', 'aorta', 'azygos'],
                'ask': ask('A squamous carcinoma of the middle third lies at the level of the carina. Which test must precede resection?', 'Bronchoscopy, to exclude invasion of the trachea or left main bronchus (T4b)',
                           'Mid-third tumours sit against the membranous trachea and left main bronchus; airway invasion makes the tumour unresectable and changes the plan.', 'Colonoscopy', 'A barium enema', 'Bone marrow biopsy'),
                'ct': ct(R(tgt), 'axial')}
    eso_cases = {
        'ivor': case('Case: lower-third squamous carcinoma after chemoradiotherapy',
                     '<p>A <b>52-year-old farmer</b> from Bomet, 3 months of dysphagia to solids, 6 kg weight loss. Endoscopy: a lower-third ulcerated tumour at 36–40 cm, biopsy <b>squamous cell carcinoma</b>; CT/EUS: <b>cT3 N1</b>; no distant disease; bronchoscopy normal. After <b>CROSS</b> chemoradiotherapy (carboplatin/paclitaxel with 41.4 Gy), restaging shows a good response. Fit, BMI 19.</p>',
                     '<p><b>Oesophagectomy</b> after neoadjuvant chemoradiotherapy: Ivor Lewis (abdomen, then right thoracotomy or thoracoscopy, anastomosis in the chest) suits a lower-third tumour with a good proximal margin. Nutrition before surgery (a feeding jejunostomy is often placed). Minimally invasive or hybrid access lowers pulmonary complications.</p>'
                     '<p><b>If there is residual disease</b> in the specimen, adjuvant nivolumab is an option.</p>'
                     + ev('CROSS (NEJM 2012; Lancet Oncol 2015): chemoradiotherapy then surgery improved R0 resection (92% vs 69%) and survival; pCR 49% in squamous vs 23% in adenocarcinoma; median OS for squamous 81.6 vs 21.1 months. TIME (Lancet 2012): minimally invasive oesophagectomy cut 2-week pulmonary infection (9% vs 29%). MIRO (NEJM 2019): hybrid access halved major complications (36% vs 64%). CheckMate 577 (NEJM 2021): adjuvant nivolumab doubled DFS (22.4 vs 11.0 months) with residual disease after CRT.'),
                     ask('In CROSS, which histology responded best to chemoradiotherapy?', 'Squamous cell carcinoma (pCR 49% vs 23% in adenocarcinoma)',
                         'Squamous tumours are more radiosensitive; the survival gain was largest in squamous carcinoma (median OS 81.6 vs 21.1 months).', 'Adenocarcinoma', 'Both equally', 'Neither: CROSS was negative'),
                     tl(Pt(ESO_T['low']) if ESO_T['low'] in LM else Pt('esophagus'), (0.9, -0.5, 0.2), 300), ['esophagus', ESO_T['low'], 'aorta', 'azygos', 'heart', 'trachea'], hide=LOBE_IDS, labels=[ESO_T['low'], 'azygos', 'aorta'], opacity={'heart': 0.25, 'aorta': 0.5, 'esophagus': 0.6}),
        'mckeown': case('Case: a young man with mid-oesophageal squamous carcinoma',
                        '<p>A <b>27-year-old man</b> from western Kenya, dysphagia for 4 months. Endoscopy: a tumour at <b>25–30 cm</b> (mid third, at the carina), squamous; CT/EUS cT3 N1; <b>bronchoscopy: no airway invasion</b>; PET: no distant disease. Good performance status.</p>',
                        '<p><b>Mid-third tumours</b> need a long proximal margin: a <b>McKeown</b> (three-stage) oesophagectomy with a <b>neck anastomosis</b>, after neoadjuvant chemoradiotherapy (CROSS). The thoracic dissection is close to the membranous trachea and the left main bronchus: injury there is a disaster.</p>'
                        '<p><b>Definitive chemoradiotherapy</b> is an alternative for squamous carcinoma: surgery adds local control but not clearly survival, at higher treatment mortality; it suits patients who respond clinically, the frail, or those declining surgery, with salvage surgery for residual disease.</p>'
                        + ev('Stahl et al. (J Clin Oncol 2005; 172 patients, squamous): adding surgery after chemoradiotherapy improved 2-year local control (64% vs 41%) but not overall survival (40% vs 35%); treatment mortality 12.8% vs 3.5%. CROSS long-term (Lancet Oncol 2015).'),
                        ask('Why a neck anastomosis (McKeown) for this tumour?', 'A mid-third tumour needs a long proximal margin that a chest anastomosis may not give',
                            'Taking the oesophagus into the neck gives a longer margin above the tumour and a complete thoracic lymphadenectomy; a cervical leak is also easier to manage than an intrathoracic one.', 'It is always required for squamous carcinoma', 'Because the stomach is too short', 'To avoid the abdomen'),
                        tl(Pt(ESO_T['mid']) if ESO_T['mid'] in LM else Pt('esophagus'), (0.9, -0.5, 0.2), 300), ['esophagus', ESO_T['mid'], 'trachea', 'br-left-main', 'br-right-main', 'aorta', 'azygos', 'heart'], hide=LOBE_IDS,
                        labels=[ESO_T['mid'], 'br-left-main', 'trachea', 'azygos'], opacity={'heart': 0.25, 'aorta': 0.5, 'esophagus': 0.6}, danger=['trachea', 'br-left-main']),
        'transhiatal': case('Case: junctional adenocarcinoma in a patient with poor lung function',
                            '<p>A <b>68-year-old man</b>, long-standing reflux, dysphagia; endoscopy: an <b>adenocarcinoma at the gastro-oesophageal junction</b> (Siewert type I–II) on Barrett\'s mucosa; staging laparoscopy negative; cT3 N1. COPD with <b>FEV1 45%</b>.</p>',
                            '<p><b>Neoadjuvant therapy</b> first: for adenocarcinoma, perioperative <b>FLOT</b> chemotherapy now outperforms CROSS. Then a <b>transhiatal</b> oesophagectomy (abdomen and neck, no thoracotomy) spares his lungs; the price is a less complete mediastinal lymphadenectomy.</p>'
                            + ev('ESOPEC (NEJM 2025; 438 patients with adenocarcinoma): perioperative FLOT improved 3-year overall survival over CROSS (57.4% vs 50.7%; HR 0.70). Hulscher et al. (NEJM 2002) and 5-year follow-up (Omloo, Ann Surg 2007): no significant overall survival difference between transhiatal and extended transthoracic resection (34% vs 36%), with fewer pulmonary complications after transhiatal resection.'),
                            ask('For resectable oesophageal adenocarcinoma, which neoadjuvant strategy did ESOPEC favour?', 'Perioperative FLOT chemotherapy over CROSS chemoradiotherapy',
                                '3-year OS 57.4% vs 50.7% (HR 0.70). For squamous carcinoma, CROSS-type chemoradiotherapy remains standard.', 'CROSS over FLOT', 'Surgery alone', 'Definitive chemoradiotherapy'),
                            tl(Pt(ESO_T['low']) if ESO_T['low'] in LM else Pt('esophagus'), (0.9, -0.5, 0.2), 300), ['esophagus', ESO_T['low'], 'aorta', 'heart', 'trachea'], hide=LOBE_IDS, labels=[ESO_T['low'], 'aorta'], opacity={'heart': 0.25, 'aorta': 0.5, 'esophagus': 0.6}),
    }

    # ------------------------------------------------------------------------------------------ empyema
    EMPV = tl(Pt('empyema') if 'empyema' in LM else Pt('lll'), (-1, -0.45, 0.2), 400)
    emp_patho = {'phase': 'Pathophysiology', 'title': 'Pathophysiology: parapneumonic effusion to empyema',
                 'body': '<p><b>From pneumonia to peel.</b> Inflammation next to the pleura makes the pleural capillaries leak: a sterile <b>exudate</b>. If bacteria cross, neutrophils and bacteria consume glucose and produce lactate and CO₂ (pH falls, LDH rises), fibrin is laid down and the fluid <b>loculates</b>. Fibroblasts then organise the fibrin into a <b>peel</b> that traps the lung.</p>'
                         + chain('I exudative (days): free-flowing, sterile', 'II fibrinopurulent (1–2 weeks): infected, pH and glucose fall, septations', 'III organising (3–6 weeks): fibrous peel, trapped lung')
                         + '<p><b>Drain when</b>: pus, organisms on Gram stain or culture, <b>pH 7.2 or less</b>, or (if pH unavailable) glucose under 3.3 mmol/L; loculation on ultrasound supports it. A pH of 7.2–7.4 is intermediate (LDH over 900 IU/L supports drainage).</p>'
                         '<p><b>Treatment follows the stage</b>: antibiotics and a small-bore drain; intrapleural <b>tPA plus DNase</b> for a residual collection; <b>surgery</b> (VATS debridement, or decortication) when sepsis and collection persist, or the lung is trapped. <b>Tuberculous</b> empyema is chronic from the start, with a thick peel, often a trapped lung, and needs TB treatment before and after decortication.</p>'
                         '<p><b>Risk at presentation: the RAPID score</b> (renal: urea; age; purulence; infection source; dietary: albumin) sorts patients into low, medium and high risk of death at 3 months (about 2%, 9% and 29%).</p>'
                         + ev('BTS 2023 guideline (Roberts et al., Thorax 2023): small-bore drains, tPA 10 mg + DNase 5 mg twice daily for 3 days for residual collections, not streptokinase or either drug alone; VATS preferred when surgery is needed. MIST1 (NEJM 2005): streptokinase gave no benefit. MIST2 (NEJM 2011): tPA+DNase improved drainage and cut surgical referral (4% vs 16%). RAPID validated in PILOT (Eur Respir J 2020). Uganda (Mwesige et al., BMC Pulm Med 2025; 200 adults): 6.5% of empyemas were tuberculous; in-hospital mortality 10.5%.'),
                 'view': EMPV, 'spin': True, 'show': ['lul', 'lll', 'fissure', 'peel-l', 'empyema-l', 'heart', 'aorta'], 'hide': [], 'opacity': {'lul': 0.5, 'lll': 0.5, 'heart': 0.3},
                 'highlight': ['empyema-l', 'peel-l'], 'labels': ['empyema-l', 'peel-l'],
                 'ask': ask('After 5 days of a small-bore drain and antibiotics a loculated collection remains and the patient is still febrile. Next step?', 'Intrapleural tPA plus DNase, or surgical referral (VATS) if unsuitable or it fails',
                            'MIST2: the combination (not either drug alone, nor streptokinase) improves drainage and reduces surgery; persistent sepsis despite drainage is the trigger for surgery.', 'Intrapleural streptokinase', 'Continue and wait 2 more weeks', 'Intrapleural DNase alone'),
                 'ct': ct(R(EMPV['target']), 'axial', 'lung')}
    emp_cases = {
        'vats': case('Case: a fibrinopurulent empyema that did not drain',
                     '<p>A <b>38-year-old man</b>, 10 days of community-acquired pneumonia; febrile, CRP 280. Ultrasound: a <b>septated</b> left effusion; pleural fluid <b>pH 7.0</b>, glucose 1.8 mmol/L, turbid; <i>Streptococcus</i> on culture. A 12F drain and 5 days of antibiotics, then <b>tPA/DNase</b> for 3 days: a posterior basal collection persists and he is still febrile. HIV-negative; GeneXpert on the fluid negative. RAPID score 2 (low risk).</p>',
                     '<p><b>Stage II (fibrinopurulent)</b> not controlled by drainage and fibrinolytics: <b>VATS debridement</b>, breaking the loculations, evacuating the pus and stripping the early peel so the lung re-expands.</p>'
                     + ev('VATS vs open decortication meta-analysis (Pan et al., J Thorac Dis 2017; 918 patients, retrospective): lower mortality (4.1% vs 6.2%) and morbidity, shorter stay, no difference in relapse. BTS 2023: prefer VATS when surgery is needed.'),
                     ask('What finding on pleural fluid most clearly mandated chest drainage here?', 'pH 7.0 (with positive culture and turbid fluid)',
                         'A pleural pH of 7.2 or less indicates complicated parapneumonic effusion or empyema; culture positivity and pus are independent indications.', 'Protein over 30 g/L', 'A lymphocytic effusion', 'Fluid volume over 1 L'),
                     EMPV, ['lul', 'lll', 'fissure', 'peel-l', 'empyema-l'], labels=['empyema-l'], opacity={'lul': 0.5, 'lll': 0.5}),
        'open': case('Case: a chronic tuberculous empyema with a trapped lung',
                     '<p>A <b>24-year-old woman</b>, HIV-positive (on ART, CD4 380), 3 months of cough and weight loss; a left pleural collection drained 6 weeks ago grew nothing on routine culture, but <b>GeneXpert: <i>M. tuberculosis</i>, rifampicin-sensitive</b>. On TB treatment for 5 weeks. CT: a thick, enhancing pleural peel, a residual posterior collection, <b>the left lung trapped</b>, ribs crowded.</p>',
                     '<p><b>Stage III (organising)</b>, tuberculous: continue TB treatment, and <b>decortication</b> through a thoracotomy (a thick, old peel rarely comes off thoracoscopically) once she is on treatment and nutritionally supported. The goal: free the lung so it fills the chest and the space is gone.</p>'
                     '<p>If the lung is destroyed underneath, decortication will not re-expand it: plan for space-filling (muscle flap, limited thoracoplasty) or an open window.</p>'
                     + ev('TB pleural disease and empyema: Vorster et al., J Thorac Dis 2015. BTS 2023 guideline on pleural infection. In a Ugandan series of 200 empyemas, most were treated by chest tube alone; 10 had decortication (Mwesige et al., 2025).'),
                     ask('Why continue TB treatment before and after decortication?', 'Surgery removes the peel and the space, but only drugs treat the mycobacterial infection',
                         'Operating on a patient already established on effective TB treatment lowers the risk of bronchopleural fistula and recurrent infection; treatment completes the course afterwards.', 'It is not needed after decortication', 'Only for MDR-TB', 'To prevent bleeding'),
                     EMPV, ['lul', 'lll', 'fissure', 'peel-l', 'empyema-l'], labels=['peel-l', 'empyema-l'], opacity={'lul': 0.5, 'lll': 0.5}),
    }

    # ------------------------------------------------------------------------------------------ attach the cases
    for k in ('anterior', 'posterior', 'uni', 'bi'):
        add_case(f'lul-{k}', lc_patho('left', 'lul'), lc_case('left', 'lul', k), LC_SRC)
        add_case(f'rul-{k}', lc_patho('right', 'rul'), lc_case('right', 'rul', k), LC_SRC)
        add_case(f'rml-{k}', lc_patho('right', 'rml'), lc_case('right', 'rml', k), LC_SRC)
    add_case('rml-fissure', lc_patho('right', 'rml'), lc_case('right', 'rml', 'f'), LC_SRC)
    add_case('rml-open', lc_patho('right', 'rml'), lc_case('right', 'rml', 'o'), LC_SRC)
    for k in ('fissure', 'hilum', 'uni', 'bi'):
        add_case(f'lll-{k}', lc_patho('left', 'lll'), lc_case('left', 'lll', k), LC_SRC)
        add_case(f'rll-{k}', lc_patho('right', 'rll'), lc_case('right', 'rll', k), LC_SRC)
    add_case('rll-open', lc_patho('right', 'rll'), lc_case('right', 'rll', 'o'), LC_SRC)
    # ---- congenital lung lesions have entries of their own: each is cloned from the raw open-lobectomy skeleton, then given its own pathophysiology and case.
    # The lobectomies themselves carry the general lung-cancer pathophysiology and staging, like the other approaches.
    import copy as _cp
    def cong_entry(key, src, op, opName, approach, summary, patho, case_, op_title, op_body, group='Congenital lung lesions', srcs=CONG_SRC):
        if src not in procs: return
        q = _cp.deepcopy(procs[src])
        q.update({'id': f'open-{op}', 'op': op, 'opName': opName, 'name': opName, 'approach': approach, 'summary': summary, 'group': group})
        for s in q['steps']: s['id'] = f"{key}-{s['id']}"
        procs[key] = q
        add_case(key, patho, case_, srcs)
        # the operation itself is the lobectomy: keep one step that points to it, instead of repeating its steps
        st = q['steps']; teach = ('Pathophysiology', 'Anatomy', 'Case', 'Decision')
        o = dict(next(s for s in st if s['phase'] not in teach))
        for kk in ('lead', 'pearl', 'ask', 'askAfter', 'pose'): o.pop(kk, None)
        o.update({'id': f'{key}-operation', 'phase': 'Operation', 'title': op_title, 'body': op_body, 'seq': 2})
        pat_ = next(s for s in st if s['phase'] == 'Pathophysiology'); cas_ = next(s for s in st if s['phase'] == 'Case')
        pat_['seq'] = 0; cas_['seq'] = 1
        q['steps'] = [pat_, cas_, o]
        q['sequence'] = [{'label': 'Patho', 'kind': 'other'}, {'label': 'Case', 'kind': 'other'}, {'label': 'Operation', 'kind': 'other'}]
    cong_entry('cle-open', 'lul-open', 'cle', 'Congenital lobar emphysema', 'Left upper lobectomy, open (the commonest site)',
               'An over-distended lobe from a ball-valve bronchus: where it sits, how it presents, the drain trap, the anesthetic, and lobectomy.',
               {**cong_patho('cle'), 'opacity': {'rul': 0.45, 'rml': 0.55, 'lll': 0.2, 'rll': 0.2}, 'highlight': ['cle-lul', 'rml', 'rul'], 'labels': ['cle-lul', 'rml', 'rul']}, cle_case,
               'The operation: lobectomy', '<p><b>Symptomatic CLE is treated by lobectomy.</b> A left upper lesion is a <a class="link" href="#approach=lul-open&step=4">left upper lobectomy</a> through a thoracotomy (thoracoscopy in experienced hands); a right middle or right upper lesion is the <a class="link" href="#approach=rml-open&step=4">right middle</a> or <a class="link" href="#approach=rul-open&step=4">right upper</a> lobectomy. The steps are those of the adult operation in a much smaller chest: fissure, arteries, bronchus, veins.</p><p><b>What differs in an infant</b>: gentle or spontaneous ventilation until the chest is open, no nitrous oxide, the surgeon scrubbed at induction; the lobe herniates out of the incision and the child improves at once.</p>')
    cong_entry('cpam-open', 'lll-open', 'cpam', 'Congenital pulmonary airway malformation (CPAM)', 'Left lower lobectomy, open (any lobe is possible)',
               'A cystic or solid malformation of one lobe: Stocker types, hydrops and the CVR, the infection and malignancy risk, timing, and lobectomy.',
               {**cong_patho('cpam'), 'highlight': ['cpam-lll'], 'labels': ['cpam-lll', 'lll']}, cpam_case,
               'The operation: lobectomy', '<p><b>A symptomatic CPAM is treated by lobectomy</b> (segmentectomy only in selected cases). A left lower lesion is a <a class="link" href="#approach=lll-open&step=4">left lower lobectomy</a>; other lobes follow the matching lobectomy, for example the <a class="link" href="#approach=rll-open&step=4">right lower</a>. The steps are those of the adult operation in a smaller chest: ligament, fissure, arteries, vein, bronchus.</p><p><b>What matters here</b>: complete excision, because some lesions carry a malignancy risk; check the CT for a systemic feeder from the aorta before dividing the inferior ligament; operate once an infection has settled.</p>')
    cong_entry('asp-open', 'rul-open', 'asp', 'Aspergilloma (post-TB cavity)', 'Right upper lobectomy, open (upper lobes most often)',
               'A fungal ball in a post-TB cavity: simple versus complex disease, the hemoptysis pathway, embolization as a bridge, and resection.',
               tb_patho(), asp_case,
               'The operation: resection', '<p><b>A simple aspergilloma is resected, most often by lobectomy</b>: the <a class="link" href="#approach=rul-open&step=4">right upper lobectomy</a> is the model for an upper lobe cavity; a left upper cavity follows the <a class="link" href="#approach=lul-open&step=4">left upper lobectomy</a>. The steps are fissure, arteries, bronchus, veins.</p><p><b>What differs here</b>: dense, vascular apical adhesions, so an extrapleural plane may be needed; bleeding from chest wall collaterals; do not enter the cavity (spillage of the fungal ball); plan a muscle flap if a space will remain. Open surgery, or VATS only in experienced hands.</p>', group='Lung infection and cavities', srcs=TB_SRC)
    add_case('rul-open', lc_patho('right', 'rul'), lc_case('right', 'rul', 'o'), LC_SRC)
    add_case('lul-open', lc_patho('left', 'lul'), lc_case('left', 'lul', 'o'), LC_SRC)
    add_case('lll-open', lc_patho('left', 'lll'), lc_case('left', 'lll', 'o'), LC_SRC)
    for k, nm in (('lingula', 'lingula'), ('lul-updiv', 'upper division of the left upper lobe'), ('s6', 'superior segment (S6) of the left lower lobe')):
        add_case(f'seg-{k}', lc_patho('left', 'lll' if k == 's6' else 'lul'), seg_case('lll' if k == 's6' else 'lul', nm), LC_SRC)
    add_case('pnl-open', tb_patho(destroyed=True), destroyed_case, TB_SRC)
    for sd, key in (('left', 'pnl-vats'), ('right', 'pnr-vats'), ('right', 'pnr-open')):
        tumc = f'tumour-central-{sd[0]}'
        add_case(key, lc_patho(sd, 'x' + sd[0]),
                 case('Case: a central squamous carcinoma needing pneumonectomy',
                      f'<p>A <b>63-year-old man</b>, smoker, haemoptysis. Bronchoscopy: a squamous carcinoma at the origin of the {"left upper and lower lobe bronchi, extending into the left main bronchus" if sd == "left" else "right upper lobe bronchus, extending to the bronchus intermedius and the right main bronchus"}; CT: a 5 cm hilar mass encasing the {"left" if sd == "left" else "right"} pulmonary artery. EBUS: 4{"L" if sd == "left" else "R"} and 7 negative, 10{sd[0].upper()} positive (N1). PET: no distant disease. FEV1 2.4 L (82%), DLCO 75%; perfusion to the {sd} lung 45%.</p>',
                      '<p><b>Stage</b>: T3 (5 cm) N1 = IIIA. A sleeve lobectomy cannot clear the artery and both lobar bronchi: <b>pneumonectomy</b>. ppoFEV1 = 2.4 × 0.55 ≈ 1.3 L (≈45%): exercise testing and CPET before committing. Neoadjuvant chemo-immunotherapy is an option. '
                      + ('<b>Right pneumonectomy</b> carries the highest mortality of any lung resection and the highest risk of <b>bronchopleural fistula</b> (a poorly covered stump with a single bronchial artery): keep the stump short and <b>cover it</b> (intercostal muscle, pericardial fat, pleura), avoid fluid overload (post-pneumonectomy pulmonary oedema). See the <b>post-pneumonectomy empyema and BPF</b> module.' if sd == 'right' else 'The left main bronchus retracts under the aortic arch after division: keep the stump short, flush with the carina, before it disappears.')
                      + '</p>' + ev('BPF after pneumonectomy: 0–6.3% after left and 1.1–22.9% after right pneumonectomy across series (Wali and Billè, Shanghai Chest 2021); Deschamps et al. (Mayo, 713 pneumonectomies): empyema 7.5%, fistula 4.5%.'),
                      ask('Why is bronchopleural fistula commoner after right than left pneumonectomy?', 'The right main bronchial stump has little mediastinal cover and usually a single bronchial artery',
                          'The left stump retracts under the aortic arch into the mediastinum and usually has two bronchial arteries; the exposed, less well perfused right stump breaks down more often.', 'The right bronchus is narrower', 'The left lung is smaller', 'It is not: left is commoner'),
                      tl(Pt(tumc), LATD[sd], 320), [*LOBE_IDS, tumc, *NODES], labels=[tumc, f'ln-10{sd[0]}', 'ln-7'], opacity={i: 0.3 for i in LOBE_IDS}), LC_SRC)
        procs[key]['steps'][0]['show'] = [*LOBE_IDS, tumc, *NODES, 'trachea']; procs[key]['steps'][0]['highlight'] = [tumc]
        procs[key]['steps'][0]['labels'] = [tumc]; procs[key]['steps'][0]['view'] = tl(Pt(tumc), LATD[sd], 320)
    for k in ('sternotomy', 'rvats', 'subx'):
        add_case(f'b4-thymectomy-{k}', mg_patho, th_cases[k], MG_SRC)
    for k, lvl in (('ivor', 'low'), ('mckeown', 'mid'), ('transhiatal', 'low')):
        add_case(f'b4-eso-{k}', eso_patho(lvl), eso_cases[k], ESO_SRC)
    add_case('b4-emp-vats', emp_patho, emp_cases['vats'], EMP_SRC)
    add_case('b4-emp-open', emp_patho, emp_cases['open'], EMP_SRC)
    # the thymoma is shown only where the case has one; lung-cancer patho for a central tumour points at it
    for k in ('b4-thymectomy-rvats', 'b4-thymectomy-subx'):
        if k in procs:
            for s in procs[k]['steps']:
                s['show'] = [i for i in s.get('show', []) if i != 'thymoma']; s['labels'] = [i for i in s.get('labels', []) if i != 'thymoma']
                s['hide'] = [*s.get('hide', []), 'thymoma']

    # ------------------------------------------------------------------------------------------ new: post-pneumonectomy empyema and bronchopleural fistula
    if has('ppe-fluid') and has('bronchial-stump'):
        STUMP, BPF_ = Pt('bronchial-stump'), Pt('bpf')
        PPE_OFF = [i for i in ('rul', 'rml', 'rll', 'fissure-h', 'fissure-r', 'ipl-r', 'br-right-main', 'lesion-rca') if has(i)]
        PPE_OFF += [i for i in S if S[i]['group'] in ('airway', 'arteries', 'veins', 'rul-intra', 'segments') and S[i]['centroid'][0] > 12]
        PPE_OFF += [i for i in S if S[i]['group'] in ('rul-intra',)]
        PPE_SHOW = ['ppe-fluid', 'ppe-air', 'bronchial-stump', 'bpf', 'trachea', 'br-left-main', 'heart', 'aorta', 'svc', 'azygos', 'esophagus', 'lul', 'lll', 'fissure']
        pv = tl(STUMP, (1, 0.1, 0.35), 280)
        side_v = tl(Pt('ppe-level') if 'ppe-level' in LM else STUMP, (1, -0.2, 0.25), 420)
        W_ = Pt('eloesser') if 'eloesser' in LM else Pt('ppe-level')
        ppe_patho = {'id': 'ppe-patho', 'phase': 'Pathophysiology', 'seq': 0, 'askAfter': True, 'title': 'Pathophysiology: the post-pneumonectomy space, empyema and fistula',
                     'body': '<p><b>The empty hemithorax.</b> After pneumonectomy the space fills with serous fluid over days to weeks, while the mediastinum shifts across, the diaphragm rises and the ribs crowd in. The fluid is an ideal culture medium: seeded at operation, from the blood, or through a <b>bronchopleural fistula (BPF)</b>, it becomes <b>post-pneumonectomy empyema (PPE)</b>, reported in 2–16%.</p>'
                             + chain('Stump ischaemia (stripped bronchial arteries), tension, a long stump, residual tumour, radiation, infection', '!Stump dehiscence: BPF', 'Air enters the space; fluid drains into the airway', '!Aspiration into the only lung: pneumonia, ARDS, death')
                             + chain('Infected fluid in a rigid space', 'Sepsis', '!Erodes the stump from outside: late BPF')
                             + '<p><b>Early BPF</b> (first days to about 2 weeks) is usually technical; <b>late BPF</b> follows infection or ischaemia and can appear months later. <b>Right</b> pneumonectomy is worst: the right stump lies uncovered in the pleural space, usually on a single bronchial artery, while the left retracts under the aortic arch and has two.</p>'
                             '<p><b>Signs</b>: fever, <b>coughing up large volumes of serous or brownish fluid</b> (the pleural fluid), a <b>falling fluid level</b> or a new air-fluid level on the X-ray, subcutaneous emphysema, contralateral aspiration pneumonia.</p>'
                             '<p><b>First move</b>: sit up or lie <b>operated side down</b> (so the fluid cannot run into the good lung), then a chest drain into the space, antibiotics, and bronchoscopy.</p>'
                             + ev('incidence: Deschamps et al. (Mayo, 713 pneumonectomies, Ann Thorac Surg 2001): empyema 7.5%, BPF 4.5%; BPF 0–6.3% after left vs 1.1–22.9% after right pneumonectomy, mortality 18–71% (Wali and Billè, 2021; Gritsiuta et al., 2025). Anatomy of right-sided risk: Wali and Billè 2021.'),
                     'view': side_v, 'spin': True, 'show': PPE_SHOW, 'hide': PPE_OFF, 'opacity': {'heart': 0.35, 'lul': 0.45, 'lll': 0.45, 'aorta': 0.6},
                     'highlight': ['bpf'], 'danger': ['ppe-fluid'], 'labels': ['ppe-fluid', 'ppe-air', 'bronchial-stump', 'bpf', 'lul'],
                     'ask': ask('Ten days after right pneumonectomy, a patient suddenly coughs up 300 mL of thin brown fluid and becomes breathless. How should he be positioned?', 'Operated (right) side down, or sitting up, so the space fluid cannot flood the left lung',
                                'This is a bronchopleural fistula: the pneumonectomy space is draining into the airway. Positioning protects the remaining lung until a drain is in.', 'Left side down', 'Flat and supine', 'Head down'),
                     'ct': ct(R(STUMP), 'axial', 'lung')}
        ppe_anat = {'id': 'ppe-anatomy', 'phase': 'Anatomy', 'seq': 1, 'title': 'The right stump and its neighbours',
                    'body': '<p>The right main bronchial stump sits at the <b>carina</b>, behind the <b>SVC</b> and the stump of the right pulmonary artery, under the <b>azygos arch</b>, with the <b>oesophagus</b> behind and medial. Re-exposing it through the infected space means dense, friable tissue; the <b>transsternal, transpericardial</b> route reaches the carina through clean tissue, between the SVC and the aorta, when the pleural route is hostile.</p>'
                            '<p>The space itself is bounded by the mediastinum, the raised diaphragm and the chest wall; its lowest point laterally (about the 6th–8th ribs) is where an open window drains it.</p>',
                    'view': pv, 'spin': True, 'show': PPE_SHOW, 'hide': PPE_OFF, 'opacity': {'heart': 0.35, 'lul': 0.35, 'lll': 0.35, 'ppe-fluid': 0.35, 'aorta': 0.6},
                    'highlight': ['bronchial-stump'], 'danger': ['svc', 'azygos', 'esophagus', 'aorta'], 'labels': ['bronchial-stump', 'trachea', 'svc', 'azygos', 'esophagus', 'aorta'],
                    'ct': ct(R(STUMP), 'axial')}
        ppe_case = {'id': 'ppe-case', 'phase': 'Case', 'seq': 2, 'title': 'Case: fever and a falling fluid level after right pneumonectomy',
                    'lead': '<p>A <b>61-year-old man</b>, <b>right pneumonectomy</b> 11 days ago for a central squamous carcinoma after neoadjuvant chemotherapy; the stump was stapled and not covered. Now: fever 38.9 °C, WCC 21, and he coughs up <b>thin brown fluid</b>. X-ray: the fluid level in the right hemithorax has <b>fallen by 4 cm</b> since yesterday; a patchy shadow in the left lower zone. Bronchoscopy: a <b>5 mm dehiscence</b> at the stump; the pleural fluid is turbid.</p>',
                    'body': '<p><b>Early BPF with post-pneumonectomy empyema</b>. Immediate: operated side down, a <b>chest drain</b> into the space, broad-spectrum antibiotics, ICU. Then two paths:</p>'
                            '<ul><li><b>Fit, early fistula, clean enough</b>: <b>re-operate early</b>: debride the space, re-amputate and close the stump, <b>cover it with a vascularised flap</b> (latissimus, serratus, intercostal muscle or omentum), then sterilise and close the space (Clagett-type fill, or repeated debridement with negative pressure and closure).</li>'
                            '<li><b>Unfit, or late, established infection</b>: <b>open window thoracostomy</b> (Eloesser) to drain the space for weeks, dressing changes, then a later closure (Clagett) once the cavity is clean, with or without muscle transposition.</li></ul>'
                            '<p>Small fistulas (under 3–5 mm) in unfit patients can sometimes be closed endoscopically (glue, occluder devices, stents), usually as a bridge.</p>'
                            + ev('Clagett and Geraci (1963): open window drainage then filling the cavity with antibiotic solution and closing it; Zaheer et al. (Mayo 2006): success 81% after the first attempt. Pairolero et al. (JTCVS 1990): muscle transposition to close the fistula and fill the space, success 84%. Accelerated treatment (Schneiter et al., JTCVS 2008; 75 patients, 59% with BPF): repeated debridement, negative pressure and antibiotic-filled closure healed 97% with 4% 90-day mortality. Stump coverage (Di Maio et al., meta-analysis 2015): coverage was selective in most series; the one trial, in diabetics, favoured coverage.'),
                    'ask': ask('What is the first priority when this patient starts coughing up space fluid?', 'Protect the left lung: position operated side down and drain the space',
                               'Death in BPF comes from aspiration of the infected space fluid into the remaining lung; positioning and a chest drain stop it, before bronchoscopy or definitive surgery.',
                               'Urgent bronchoscopic glue', 'CT scan first', 'Start a diuretic'),
                    'view': side_v, 'show': PPE_SHOW, 'hide': PPE_OFF, 'opacity': {'heart': 0.35, 'lul': 0.45, 'lll': 0.45, 'aorta': 0.6},
                    'labels': ['ppe-fluid', 'bpf', 'lll'], 'danger': ['lll', 'lul'], 'ct': ct(R(STUMP), 'axial', 'lung')}
        ppe_drain = {'id': 'ppe-drain', 'phase': 'Drain', 'seq': 3, 'title': 'Position, drain the space, bronchoscopy',
                     'body': '<p>Sit him up, <b>operated side down</b>. A large-bore drain into the space in the mid-axillary line, above the diaphragm (which has risen: check the level on ultrasound), connected to an underwater seal <b>without suction</b> (suction pulls the mediastinum across). Send the fluid for culture, including TB.</p>'
                             '<p><b>Bronchoscopy</b>: the size and site of the defect, and the viability of the stump; clear aspirated secretions from the left lung.</p>',
                     'view': side_v, 'show': PPE_SHOW, 'hide': PPE_OFF, 'opacity': {'heart': 0.35, 'lul': 0.4, 'lll': 0.4, 'aorta': 0.6}, 'highlight': ['ppe-fluid'], 'labels': ['ppe-fluid', 'ppe-air'],
                     'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Drain the space', 'port': 'thor-r', 'remove': ['ppe-fluid'],
                                'path': [R(Pt('ppe-level') + V([30, 0, 20])), R(Pt('ppe-level') + V([10, 0, 0])), R(Pt('ppe-level') + V([30, 0, -20]))]},
                     'ct': ct(R(Pt('ppe-level')), 'axial', 'lung')}
        win_steps = [ppe_patho, ppe_anat, ppe_case, ppe_drain,
                     {'id': 'ppe-window', 'phase': 'Window', 'seq': 4, 'title': 'Open window thoracostomy (Eloesser)',
                      'body': '<p>Over the <b>lowest part of the cavity</b>, laterally (confirm with CT or a needle), make a U-shaped or inverted-U skin flap with its base at the dependent edge. Resect <b>segments of two ribs</b> (about 8–10 cm each) under the flap, open the thickened parietal pleura, and <b>suture the skin to the pleura</b> all round so the window lines itself and stays open.</p>'
                              '<p>The cavity now drains by gravity and can be packed and inspected; the mediastinum is already fixed, so an open chest is tolerated.</p>',
                      'view': tl(W_, (1, 0.1, 0.2), 320), 'show': [*PPE_SHOW, 'eloesser', *[f'rib-{i}-r' for i in range(5, 10) if has(f'rib-{i}-r')]], 'hide': PPE_OFF,
                      'opacity': {'heart': 0.3, 'lul': 0.3, 'lll': 0.3, 'ppe-fluid': 0.4}, 'highlight': ['eloesser'], 'labels': ['eloesser', 'ppe-fluid'],
                      'action': {'kind': 'reveal', 'label': 'Open the window', 'port': 'thor-r', 'ids': ['eloesser']},
                      'ask': ask('Where should the open window be placed?', 'Over the most dependent part of the cavity, laterally',
                                 'Gravity drainage needs the lowest point; too high and pus pools below it.', 'Anteriorly in the 2nd space', 'Over the stump', 'Posteriorly at the apex'),
                      'ct': ct(R(W_), 'axial', 'lung')},
                     {'id': 'ppe-pack', 'phase': 'Window', 'seq': 5, 'title': 'Debride, pack or apply negative pressure',
                      'body': '<p>Debride the fibrin and necrotic tissue; irrigate. Pack with gauze soaked in dilute povidone-iodine or antibiotic solution, changed daily or every 2 days, or apply <b>negative-pressure therapy</b> (a sponge in the cavity) if there is <b>no open fistula</b> or it has been closed and covered. The fistula often closes as the space granulates; if not, it is closed and covered with muscle later.</p>'
                              + ev('accelerated treatment (Schneiter et al., JTCVS 2008): repeated debridement in theatre every 48 hours, povidone-iodine packing and negative pressure, then closure with the cavity filled with antibiotic solution: 97% healed, chest closed within 8 days in 95%.'),
                      'view': tl(W_, (1, 0.1, 0.2), 300), 'show': [*PPE_SHOW, 'eloesser'], 'hide': [*PPE_OFF, 'ppe-fluid'], 'opacity': {'heart': 0.3, 'lul': 0.3, 'lll': 0.3},
                      'highlight': ['ppe-air'], 'labels': ['eloesser', 'bpf'], 'ct': ct(R(W_), 'axial', 'lung')},
                     {'id': 'ppe-clagett', 'phase': 'Close', 'seq': 6, 'title': 'Clagett closure: fill the clean cavity and close',
                      'body': '<p>When the cavity is clean and granulating, and the fistula closed (weeks to months later, typically 6–8 weeks), <b>fill the space with antibiotic solution</b> (for example DAB: neomycin, polymyxin B and gentamicin per litre), and close the window in layers, watertight. The sterile fluid then obliterates slowly, like a normal post-pneumonectomy space.</p>'
                              '<p>A persistent fistula must first be closed and <b>covered with muscle</b> (latissimus or serratus through the window), or the Clagett fails.</p>'
                              + ev('Clagett and Geraci (JTCVS 1963); success 81% at first attempt and 88% after a second (Zaheer et al., Ann Thorac Surg 2006).'),
                      'view': tl(W_, (1, 0.1, 0.2), 340), 'show': [*PPE_SHOW, 'eloesser'], 'hide': [*PPE_OFF, 'ppe-fluid', 'bpf'], 'opacity': {'heart': 0.3, 'lul': 0.3, 'lll': 0.3},
                      'labels': ['eloesser'], 'ct': ct(R(W_), 'axial', 'lung')}]
        FLAP_ = Pt('bronchial-stump')
        flap_steps = [{**ppe_patho, 'id': 'pf-patho'}, {**ppe_anat, 'id': 'pf-anatomy'}, {**ppe_case, 'id': 'pf-case'}, {**ppe_drain, 'id': 'pf-drain'},
                      {'id': 'pf-rethor', 'phase': 'Space', 'seq': 4, 'title': 'Re-thoracotomy: evacuate and debride the space',
                       'body': '<p>Through the old incision (or a fresh one through the rib bed), evacuate the infected fluid and fibrin; debride the parietal pleura; irrigate generously. Send tissue for culture. The mediastinum is fixed and the heart lies close under the fibrin: take care over the pericardium and the SVC.</p>',
                       'view': side_v, 'show': PPE_SHOW, 'hide': PPE_OFF, 'opacity': {'heart': 0.35, 'lul': 0.35, 'lll': 0.35, 'aorta': 0.6}, 'highlight': ['ppe-fluid'], 'labels': ['ppe-fluid'],
                       'action': {'kind': 'dissect', 'tool': 'peanut', 'label': 'Evacuate and debride', 'port': 'thor-r', 'remove': ['ppe-fluid'],
                                  'path': [R(Pt('ppe-level') + V([25, 10, 15])), R(Pt('ppe-level')), R(Pt('ppe-level') + V([25, -10, -15]))]},
                       'ct': ct(R(Pt('ppe-level')), 'axial', 'lung')},
                      {'id': 'pf-stump', 'phase': 'Stump', 'seq': 5, 'title': 'Re-amputate and close the stump',
                       'body': '<p>Free the stump from the fibrin without stripping its remaining blood supply. Trim back to healthy, bleeding cartilage, <b>flush with the carina</b>, and close it again: interrupted absorbable (or polypropylene) sutures, membranous to cartilaginous wall, or a stapler if there is length. Test under saline at 25–30 cmH₂O.</p>'
                               '<p>If the pleural field is too hostile, the stump can be reached and re-amputated <b>transsternally and transpericardially</b>, between the SVC and the aorta, in clean tissue.</p>',
                       'view': pv, 'show': PPE_SHOW, 'hide': [*PPE_OFF, 'ppe-fluid'], 'opacity': {'heart': 0.3, 'lul': 0.3, 'lll': 0.3, 'aorta': 0.6},
                       'highlight': ['bronchial-stump'], 'danger': ['svc', 'azygos', 'esophagus'], 'labels': ['bronchial-stump', 'bpf', 'svc', 'azygos'],
                       'action': {'kind': 'dissect', 'tool': 'hook', 'label': 'Close the stump', 'port': 'thor-r', 'remove': ['bpf'],
                                  'path': [R(STUMP + V([0, 8, 6])), R(STUMP), R(STUMP + V([0, -8, -6]))]},
                       'ask': ask('Why must a re-closed stump in an infected field be covered?', 'A sutured stump in an infected space breaks down again unless a vascularised flap brings blood supply and seals it',
                                  'Muscle (latissimus, serratus, intercostal), omentum or pericardial fat buttresses the closure and helps clear infection; it is the key step in Pairolero\'s approach.', 'Cover is optional', 'To prevent bleeding', 'To lengthen the stump'),
                       'ct': ct(R(STUMP), 'axial')},
                      {'id': 'pf-flap', 'phase': 'Flap', 'seq': 6, 'title': 'Latissimus dorsi flap onto the stump',
                       'body': '<p>Raise the <b>latissimus dorsi</b> on its <b>thoracodorsal</b> pedicle (it was divided at the first thoracotomy? then use serratus anterior, or an intercostal muscle, or omentum through the diaphragm). Bring it into the chest through a window made by resecting a 5–6 cm segment of the 2nd or 3rd rib, and <b>suture it over the closed stump</b> so it seals and supplies it; the bulk fills part of the space.</p>',
                       'view': tl(FLAP_, (1, -0.3, 0.3), 320), 'show': [*PPE_SHOW, 'flap-lat'], 'hide': [*PPE_OFF, 'ppe-fluid', 'bpf'], 'opacity': {'heart': 0.3, 'lul': 0.3, 'lll': 0.3, 'aorta': 0.6},
                       'highlight': ['flap-lat'], 'labels': ['flap-lat', 'bronchial-stump'],
                       'action': {'kind': 'reveal', 'label': 'Transpose the flap', 'port': 'thor-r', 'ids': ['flap-lat']},
                       'ct': ct(R(FLAP_), 'axial')},
                      {'id': 'pf-close', 'phase': 'Close', 'seq': 7, 'title': 'Fill the space and close; or open window',
                       'body': '<p>If the space is clean after debridement: fill it with antibiotic solution and close (a Clagett-type closure), or use the accelerated approach (repeat debridement every 48 hours with negative pressure, then antibiotic-filled closure). If it is not clean: leave an <b>open window</b> and close later.</p>'
                               + ev('Pairolero et al. (JTCVS 1990): muscle transposition closed the fistula and controlled the empyema in 84%. Schneiter et al. (JTCVS 2008): accelerated treatment healed 97%.'),
                       'view': side_v, 'show': [*PPE_SHOW, 'flap-lat'], 'hide': [*PPE_OFF, 'ppe-fluid', 'bpf'], 'opacity': {'heart': 0.35, 'lul': 0.35, 'lll': 0.35, 'aorta': 0.6},
                       'labels': ['flap-lat', 'ppe-air'], 'ct': ct(R(STUMP), 'axial', 'lung')}]
        for key, appr, steps_, sq in (('ppe-window', 'Open window, then Clagett closure', win_steps, seq(('Patho', 'other'), ('Anatomy', 'other'), ('Case', 'other'), ('Drain', 'other'), ('Window', 'other'), ('Pack', 'other'), ('Clagett', 'other'))),
                                      ('ppe-flap', 'Stump closure with a muscle flap', flap_steps, seq(('Patho', 'other'), ('Anatomy', 'other'), ('Case', 'other'), ('Drain', 'other'), ('Debride', 'other'), ('Stump', 'bronchus'), ('Flap', 'other'), ('Close', 'other')))):
            for s in steps_:
                if s['id'] == 'pf-flap':
                    s['body'] = s['body'].replace(' (it was divided at the first thoracotomy? then use serratus anterior, or an intercostal muscle, or omentum through the diaphragm)',
                                                  ' (if it was divided at the first thoracotomy, use serratus anterior, an intercostal muscle, or omentum through the diaphragm instead)')
                named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
                s['hide'] = [*s.get('hide', []), *[i for i in PATH_IDS if i not in named], *[k for k in S if S[k]['group'] == 'nodes' and k not in named],
                             *[i for i in ('n-phrenic', 'n-vagus', 'n-rln', 'n-phrenic-r', 'n-vagus-r', 'thymus', 'thyroid') if has(i) and i not in named]]
                s['opacity'] = {**{f'vert-t{i}': 0.22 for i in range(2, 11)}, **s.get('opacity', {})}
                for kk in ('highlight', 'danger', 'labels', 'show', 'hide'):
                    if kk in s: s[kk] = [i for i in s[kk] if has(i) or i == 'skin']
            procs[key] = {'id': key, 'op': 'ppe', 'opName': 'Post-pneumonectomy empyema and BPF', 'side': 'right', 'name': 'Post-pneumonectomy empyema and bronchopleural fistula', 'approach': appr,
                          'summary': 'The infected post-pneumonectomy space and a stump fistula: protect the other lung, drain, then open window and Clagett closure, or stump re-closure with a muscle flap.',
                          'ports': [], 'steps': steps_, 'sources': PPE_SRC, 'sequence': sq, 'group': 'Pleura'}
# ==================================================================================================== vascular: AAA, aorto-iliac occlusive disease, thoracic aortic aneurysm
VASC_OK = has('aaa-infra') and has('cia-l') and 'bifurcation' in LM
if VASC_OK:
    ev = lambda t: f'<p class="evidence"><b>Evidence:</b> {t}</p>'
    pm = lambda term: 'https://pubmed.ncbi.nlm.nih.gov/?term=' + term.replace(' ', '+')
    chain = lambda *xs: '<div class="chain">' + '<i>→</i>'.join(f'<span class="hot">{x[1:]}</span>' if x.startswith('!') else f'<span>{x}</span>' for x in xs) + '</div>'
    tl = lambda tgt, d, dist=300: {'eye': R(V(tgt) + V(d) / np.linalg.norm(V(d)) * dist), 'target': R(tgt)}
    Pv = lambda k: V(LM[k]) if k in LM else V(S[k]['centroid'])
    hv = lambda xs: [i for i in xs if has(i)]
    BELOW = '<p class="evidence"><b>Note:</b> the reference CT ends at the 4th lumbar vertebra; the iliac and femoral vessels and the groins are drawn to typical adult dimensions below it.</p>'
    VESS = hv(['aorta', 'coeliac', 'sma', 'renal-a-l', 'renal-a-r', 'ima', 'renal-v-l', 'ivc-infra', 'ivc', 'kidney-l', 'kidney-r',
               *[f'{v}-{s}' for v in ('cia', 'eia', 'iia', 'cfa', 'sfa', 'pfa', 'civ') for s in ('l', 'r')]])
    VPATH = [s_['id'] for s_ in atlas['structures'] if s_['group'] == 'vascular' and s_['id'] not in VESS] + hv(['incision-laparotomy', 'incision-flank-l', 'incision-groin-l', 'incision-groin-r', 'incision-axillary-r'])
    V_OFF = [i for i in S if S[i]['group'] in ('lungs', 'nodes', 'nerves', 'airway', 'arteries', 'veins', 'pleura', 'segments', 'lul-intra', 'lll-intra', 'rul-intra', 'trauma', 'cardiac', 'pathology')
             or S[i]['group'].startswith('ports')] + hv(['lul', 'lll', 'rul', 'rml', 'rll', 'fissure', 'fissure-h', 'fissure-r', 'esophagus', 'thymus', 'thyroid', 'stomach', 'duodenum', 'pancreas', 'spleen', 'liver', 'conduit-chest', 'conduit-neck', 'lga', 'rgea', 'thoracic-duct', 'cisterna', 'azygos'])
    ABD = Pv('aaa'); front = lambda t, dist=380, d=(0.1, 1, 0.3): tl(t, d, dist)
    AB_OP = {'aorta': 0.55, 'kidney-l': 0.5, 'kidney-r': 0.5, 'ivc-infra': 0.6, 'ivc': 0.6, **{f'vert-{x}': 0.25 for x in ('t10', 't11', 't12')}}
    AAA_SRC = [
        {'title': 'Wanhainen A, et al. European Society for Vascular Surgery (ESVS) 2024 clinical practice guidelines on the management of abdominal aorto-iliac artery aneurysms. Eur J Vasc Endovasc Surg 2024;67:192-331', 'url': 'https://www.sciencedirect.com/science/article/pii/S1078588423008894'},
        {'title': 'Oderich GS, et al. Reporting standards for endovascular aortic repair of aneurysms involving the renal-mesenteric arteries (SVS). J Vasc Surg 2021;73(1 Suppl):4S-52S', 'url': 'https://www.sciencedirect.com/science/article/pii/S0741521420314178'},
        {'title': 'Lederle FA, et al. Rupture rate of large abdominal aortic aneurysms in patients refusing or unfit for elective repair. JAMA 2002;287:2968-72', 'url': 'https://pure.johnshopkins.edu/en/publications/rupture-rate-of-large-abdominal-aortic-aneurysms-in-patients-refu-6/'},
        {'title': 'Powell JT, et al. Final 12-year follow-up of surgery versus surveillance in the UK Small Aneurysm Trial. Br J Surg 2007;94:702-8', 'url': 'https://academic.oup.com/bjs/article/94/6/702/6142549'},
        {'title': 'Lederle FA, et al. Immediate repair compared with surveillance of small abdominal aortic aneurysms (ADAM). N Engl J Med 2002;346:1437-44', 'url': 'https://www.acc.org/latest-in-cardiology/clinical-trials/2010/02/22/19/20/adam'},
        {'title': 'Patel R, et al. Endovascular versus open repair of abdominal aortic aneurysm in 15 years\' follow-up of the UK EVAR trial 1. Lancet 2016;388:2366-74', 'url': 'https://www.ncbi.nlm.nih.gov/books/NBK476568/'},
        {'title': 'Lederle FA, et al. Open versus endovascular repair of abdominal aortic aneurysm (OVER long-term). N Engl J Med 2019;380:2126-35', 'url': 'https://www.tctmd.com/news/over-trial-long-term-mortality-similar-after-endovascular-and-open-aaa-repair'},
        {'title': 'De Bruin JL, et al. Long-term outcome of open or endovascular repair of abdominal aortic aneurysm (DREAM). N Engl J Med 2010;362:1881-9', 'url': 'https://www.acc.org/latest-in-cardiology/clinical-trials/2010/05/25/16/28/dream'},
        {'title': 'IMPROVE trial investigators. Comparative clinical effectiveness and cost effectiveness of endovascular strategy v open repair for ruptured AAA: three year results. BMJ 2017;359:j4859', 'url': 'https://evtoday.com/news/three-year-improve-results-compare-treatment-strategies-for-ruptured-aaa'},
        {'title': 'Jongkind V, et al. Juxtarenal aortic aneurysm repair: a systematic review. J Vasc Surg 2010;52:760-7', 'url': 'https://www.ncbi.nlm.nih.gov/books/NBK80665/'},
        {'title': 'Sicard GA, et al. Transabdominal versus retroperitoneal incision for abdominal aortic surgery: report of a prospective randomized trial. J Vasc Surg 1995;21:174-83', 'url': 'https://www.sciencedirect.com/science/article/pii/S0741521495702601'},
        {'title': 'Ashton HA, et al. The Multicentre Aneurysm Screening Study (MASS). Lancet 2002;360:1531-9; 13-year results, Br J Surg 2012;99:1649-56', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC3569614/'},
        {'title': 'Nair R, Abdool-Carrim ATO, Chetty R, Robbs JV. Arterial aneurysms in patients infected with human immunodeficiency virus. J Vasc Surg 1999;29:600-7', 'url': 'https://www.sciencedirect.com/science/article/pii/S0741521499703046'},
        {'title': 'Wu L. Abdominal aortic aneurysm in Africa (editorial). J West Afr Coll Surg 2022', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9067627/'},
    ]
    AIOD_SRC = [
        {'title': 'Norgren L, et al. Inter-Society Consensus for the Management of Peripheral Arterial Disease (TASC II). J Vasc Surg 2007;45(Suppl S):S5-67', 'url': 'https://radcalculator.com/calc/tasc-ii'},
        {'title': 'Nordanstig J, et al. ESVS 2024 clinical practice guidelines on the management of asymptomatic lower limb peripheral arterial disease and intermittent claudication. Eur J Vasc Endovasc Surg 2024;67:9-96', 'url': 'https://www.sciencedirect.com/science/article/pii/S1078588423007414'},
        {'title': 'Conte MS, et al. Global vascular guidelines on the management of chronic limb-threatening ischemia. J Vasc Surg 2019;69(6S):3S-125S', 'url': 'https://angiolsurgery.org/library/recommendations/2019/recommendations_chronic_limb-threatening_ischemia_2019.pdf'},
        {'title': 'de Vries SO, Hunink MG. Results of aortic bifurcation grafts for aortoiliac occlusive disease: a meta-analysis. J Vasc Surg 1997;26:558-69', 'url': 'https://www.ncbi.nlm.nih.gov/books/NBK66938/'},
        {'title': 'Management of extensive aorto-iliac disease: a systematic review and meta-analysis of 9,319 patients. Cardiovasc Intervent Radiol 2021', 'url': 'https://link.springer.com/article/10.1007/s00270-021-02785-6'},
        {'title': 'Mwipatayi BP, et al. A comparison of covered vs bare expandable stents for the treatment of aortoiliac occlusive disease (COBEST). J Vasc Surg 2011;54:1561-70; 5-year results J Vasc Surg 2016;64:83-94', 'url': 'https://www.sciencedirect.com/science/article/pii/S0741521411015862'},
        {'title': 'Taeymans K, et al. Three-year outcome of the covered endovascular reconstruction of the aortic bifurcation technique for aortoiliac occlusive disease. J Vasc Surg 2018;67:1438-47', 'url': 'https://www.sciencedirect.com/science/article/abs/pii/S0741521417322978'},
        {'title': 'Martin D, Katz SG. Axillofemoral bypass for aortoiliac occlusive disease. Am J Surg 2000;180:100-3', 'url': 'https://www.sciencedirect.com/science/article/abs/pii/S0002961000004268'},
        {'title': 'Robbs JV, Paruk N. Management of HIV vasculopathy: a South African experience. Eur J Vasc Endovasc Surg 2010;39(Suppl 1):S25-31', 'url': 'https://www.sciencedirect.com/science/article/pii/S1078588410000055'},
        {'title': 'Van Marle J, Mistry PP, Botes K. HIV-occlusive vascular disease. S Afr J Surg', 'url': 'https://www.ajol.info/index.php/sajs/article/view/50473'},
        {'title': 'Genga E, Oyoo O, Adebajo A. Vasculitis in Africa. Curr Rheumatol Rep 2018;20:4', 'url': 'https://link.springer.com/article/10.1007/s11926-018-0711-y'},
        {'title': 'Leriche R, Morel A. The syndrome of thrombotic obliteration of the aortic bifurcation. Ann Surg 1948;127:193-206', 'url': 'https://www.ccjm.org/content/88/9/482'},
    ]
    TAA_SRC = [
        {'title': 'Isselbacher EM, et al. 2022 ACC/AHA guideline for the diagnosis and management of aortic disease. Circulation 2022;146:e334-e482', 'url': 'https://www.acc.org/latest-in-cardiology/ten-points-to-remember/2022/11/01/12/21/2022-guideline-on-aortic-disease-2-gl-ad'},
        {'title': 'Czerny M, et al. 2024 EACTS/STS guidelines for diagnosing and treating acute and chronic syndromes of the aortic organ. Eur J Cardiothorac Surg 2024;65:ezad426', 'url': 'https://academic.oup.com/ejcts/article/65/2/ezad426/7614462'},
        {'title': 'Coady MA, et al. What is the appropriate size criterion for resection of thoracic aortic aneurysms? J Thorac Cardiovasc Surg 1997;113:476-91', 'url': 'https://www.sciencedirect.com/science/article/pii/S002252239770360X'},
        {'title': 'Coselli JS, et al. Cerebrospinal fluid drainage reduces paraplegia after thoracoabdominal aortic aneurysm repair: a randomized clinical trial. J Vasc Surg 2002;35:631-9', 'url': 'https://www.sciencedirect.com/science/article/pii/S0741521402791477'},
        {'title': 'Coselli JS, et al. Outcomes of 3309 thoracoabdominal aortic aneurysm repairs. J Thorac Cardiovasc Surg 2016;151:1323-38', 'url': 'https://vascsurg.me/wp-content/uploads/2016/05/open-taaa-repair-coselli.pdf'},
        {'title': 'Fairman RM, et al. Pivotal results of the Medtronic Vascular Talent thoracic stent graft system: the VALOR trial. J Vasc Surg 2008;48:546-54', 'url': 'https://www.sciencedirect.com/science/article/pii/S0741521408005119'},
        {'title': 'Cheng D, et al. Endovascular aortic repair versus open surgical repair for descending thoracic aortic disease: a systematic review and meta-analysis. J Am Coll Cardiol 2010;55:986-1001', 'url': 'https://www.ncbi.nlm.nih.gov/books/NBK79689/'},
        {'title': 'Matsumura JS, et al. The Society for Vascular Surgery practice guidelines: management of the left subclavian artery with thoracic endovascular aortic repair. J Vasc Surg 2009;50:1155-8', 'url': 'https://www.sciencedirect.com/science/article/pii/S0741521409018230'},
        {'title': 'Spinal cord injury after open and endovascular repair of descending thoracic and thoracoabdominal aortic aneurysms: a meta-analysis. Ann Cardiothorac Surg 2023;12:409-17', 'url': 'https://www.annalscts.com/article/view/17051/html'},
        {'title': 'Frederick JR, Woo YJ. Thoracoabdominal aortic aneurysm (Crawford/Safi classification). Ann Cardiothorac Surg 2012;1:277-85', 'url': 'https://www.annalscts.com/article/view/1070/html'},
        {'title': 'Yan TD, et al. Consensus on hypothermia in aortic arch surgery. Ann Cardiothorac Surg 2013;2:163-8', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC3741830/'},
        {'title': 'Syphilitic aortitis (review). Interact CardioVasc Thorac Surg 2012;14:223', 'url': 'https://academic.oup.com/icvts/article/14/2/223/646377'},
    ]

    def vstep(id_, phase, title, body, view, show=(), hide=(), highlight=(), danger=(), labels=(), opacity=None, action=None, quiz=None, lead=None, after=False, ctp=None, spin=False):
        s = {'id': id_, 'phase': phase, 'title': title, 'body': body, 'view': view, 'show': list(show), 'hide': list(hide), 'highlight': list(highlight),
             'danger': list(danger), 'labels': list(labels), 'opacity': opacity or {}, 'ct': ct(R(ctp if ctp is not None else view['target']), 'axial')}
        if action: s['action'] = action
        if quiz: s['ask'] = quiz
        if lead: s['lead'] = lead
        if after: s['askAfter'] = True
        if spin: s['spin'] = True
        return s

    def build_v(groups):
        out, sq_ = [], []
        for i, (lab, kind, sts) in enumerate(groups):
            sq_.append({'label': lab, 'kind': kind})
            for s in sts:
                if s is None: continue
                out.append({**s, 'seq': i})
        return out, sq_

    # ------------------------------------------------------------------------------------------ depicting the operations, and the reference tables
    CUTS = [({'aaa-infra', 'graft-tube', 'evar-graft', 'aaa-sac-open', 'aaa-thrombus', 'anast-aaa'}, 'aorta-cut-infra'),
            ({'aaa-juxta', 'graft-juxta', 'anast-juxta'}, 'aorta-cut-juxta'),
            ({'aaa-supra', 'graft-supra', 'anast-supra'}, 'aorta-cut-supra'),
            ({'taa-desc', 'graft-taa', 'tevar-graft', 'anast-taa'}, 'aorta-cut-desc'),
            ({'taa-asc', 'graft-asc'}, 'aorta-cut-asc')]
    SAC_OF = {'aaa-infra': 'aaa-infra', 'aaa-juxta': 'aaa-juxta', 'aaa-supra': 'aaa-supra'}
    ANAST_OF = {'aaa-infra': 'anast-aaa', 'aaa-juxta': 'anast-juxta', 'aaa-supra': 'anast-supra', 'aiod-abf': 'anast-abf', 'taa-open': 'anast-taa'}
    SIZES = ('<p><b>Normal sizes and when to act.</b> An aneurysm is a permanent dilatation to at least <b>1.5 times</b> the expected normal diameter.</p>'
             '<table class="mini"><tr><th>Artery</th><th>Normal adult (typical)</th><th>Aneurysmal</th><th>Usual threshold to repair</th></tr>'
             '<tr><td>Ascending aorta</td><td>about 2.2–3.6 cm</td><td>dilated above 22 mm/m² (ESC)</td><td>5.5 cm; 5.0 cm with risk factors, bicuspid root phenotype or Marfan; 4.5 cm if the aortic valve is being operated on</td></tr>'
             '<tr><td>Descending thoracic</td><td>about 2.0–3.0 cm</td><td>dilated above 16 mm/m² (ESC)</td><td>TEVAR at 5.5 cm with suitable anatomy (ESC 2024); 6.0 cm (ACC/AHA 2022)</td></tr>'
             '<tr><td>Infrarenal aorta</td><td>about 1.5 cm (women), 1.7 cm (men) over 50</td><td>3.0 cm or more</td><td>5.5 cm (men), 5.0 cm (women); growth of about 1 cm a year; symptoms</td></tr>'
             '<tr><td>Common iliac</td><td>about 1 cm</td><td>about 1.8–2.0 cm and more (definitions vary)</td><td>4.0 cm (ESVS 2024, raised from 3.5 cm)</td></tr>'
             '<tr><td>Popliteal</td><td>about 0.5–0.9 cm</td><td>over 1.5 cm or 1.5 × normal</td><td>2.0 cm, or smaller with thrombus and embolism or poor run-off (SVS 2022)</td></tr></table>'
             + ev('definition of aneurysm as 1.5 × normal: SVS/ISCVS reporting standards (Johnston et al., J Vasc Surg 1991); normal infrarenal diameters (AAFP 2006). Thoracic indexing and thresholds: ESC 2024 (Mazzolai et al.) and ACC/AHA 2022 (Isselbacher et al.). Iliac threshold: ESVS 2024 (Recommendation 135). Popliteal: SVS 2022 (Farber et al.). Normal ranges for the iliac and popliteal arteries are typical values, not guideline cut-offs.'))
    RUTH = ('<p><b>Grading the ischaemia</b> (chronic):</p>'
            '<table class="mini"><tr><th>Fontaine</th><th>Rutherford (category)</th><th>Clinical picture</th><th>Objective (Rutherford)</th></tr>'
            '<tr><td>I</td><td>0</td><td>asymptomatic</td><td>normal treadmill test</td></tr>'
            '<tr><td>IIa (over 200 m)</td><td>1 mild</td><td>claudication</td><td>completes treadmill; ankle pressure after exercise over 50 mmHg but at least 20 mmHg below resting</td></tr>'
            '<tr><td>IIb (under 200 m)</td><td>2 moderate, 3 severe</td><td>claudication</td><td>category 3: cannot complete treadmill; ankle pressure after exercise under 50 mmHg</td></tr>'
            '<tr><td>III</td><td>4</td><td>ischaemic rest pain</td><td>resting ankle pressure under 40 mmHg, toe pressure under 30 mmHg</td></tr>'
            '<tr><td>IV</td><td>5 minor, 6 major tissue loss</td><td>ulcer, gangrene</td><td>resting ankle pressure under 60 mmHg, toe pressure under 40 mmHg</td></tr></table>'
            '<p>Rutherford 4–6 is <b>chronic limb-threatening ischaemia (CLTI)</b>. <b>Acute</b> limb ischaemia has its own Rutherford grading: I viable; IIa marginally threatened (no or minimal sensory loss, no weakness); IIb immediately threatened (sensory loss beyond the toes, rest pain, mild or moderate weakness); III irreversible (anaesthetic, paralysed, no venous Doppler signal).</p>'
            '<p><b>Ankle-brachial index</b>: 0.90 or less is PAD; 1.40 or more is non-compressible (calcified, common in diabetes and renal failure): use toe pressures.</p>'
            '<p><b>Best medical therapy is for everyone with PAD</b>, and the first treatment for claudication:</p><ul>'
            '<li><b>Stop smoking</b>: counselling at every visit, with varenicline (or bupropion, nicotine replacement).</li>'
            '<li><b>Antithrombotic</b>: a single antiplatelet (aspirin or clopidogrel); consider low-dose rivaroxaban 2.5 mg twice daily with aspirin in symptomatic PAD and after revascularisation if bleeding risk is low.</li>'
            '<li><b>High-intensity statin</b>: LDL under 1.4 mmol/L (55 mg/dL) and at least a 50% fall (ESVS 2024).</li>'
            '<li><b>Blood pressure</b> under 130/80 mmHg (ACE inhibitor or ARB); <b>diabetes</b> control (SGLT2 inhibitors, GLP-1 agonists); foot care.</li>'
            '<li><b>Supervised exercise</b>: 30–45 minutes of walking to near-maximal pain, at least 3 times a week, for at least 12 weeks.</li></ul>'
            '<p><b>When to revascularise</b>: <b>claudication</b> only when it stays lifestyle-limiting despite best medical therapy and exercise, decided with the patient (never for asymptomatic disease); <b>CLTI</b> (rest pain, ulcer, gangrene): prompt assessment for revascularisation to save the limb; <b>acute ischaemia</b> IIa urgently, IIb as an emergency, III amputation.</p>'
            + ev('Rutherford 1997 standards (J Vasc Surg 1997;26:517-38) and Fontaine, as tabulated in current reviews. ACC/AHA 2024 PAD guideline (Gornik et al.): single antiplatelet, rivaroxaban 2.5 mg twice daily plus aspirin, high-intensity statin, BP under 130/80, supervised exercise at least 3 times a week for 12 weeks; revascularisation for claudication only if lifestyle-limiting despite therapy. ESVS 2024 (Nordanstig et al.): LDL under 1.4 mmol/L; varenicline first-line; individualised decisions. COMPASS (Lancet 2018) and VOYAGER PAD (NEJM 2020) for low-dose rivaroxaban.'))
    opt = lambda name, choose, avoid, pat, risk: (f'<div class="opt"><p><b>{name}</b></p><ul><li><b>Choose for:</b> {choose}</li><li><b>Avoid when:</b> {avoid}</li>'
                                                   f'<li><b>Patency at 5 years:</b> {pat}</li><li><b>Main risks:</b> {risk}</li></ul></div>')
    WHICH = (opt('Aortobifemoral bypass', 'a fit patient with extensive bilateral aorto-iliac disease or aortic occlusion (TASC C/D); failed endovascular treatment',
                 'unfit for general anaesthesia and laparotomy; a hostile abdomen; (relative) retroperitoneal fibrosis, horseshoe kidney, severe cardiac disease',
                 'about 86–91% per limb', 'operative mortality about 4% in older series; sexual dysfunction; graft infection; aortoenteric fistula')
             + opt('Axillobifemoral bypass', 'bilateral aorto-iliac occlusion in a patient unfit for laparotomy; a hostile abdomen (several laparotomies, a stoma, radiation); an infected aortic graft or aortoenteric fistula after the graft is removed',
                   '<b>inflow disease</b> in the subclavian or axillary artery (compare both arm pressures, image the arch branches, use the arm with the higher pressure); a fit claudicant (poorer patency)',
                   'about 50–75% (63% primary in one series)', 'graft thrombosis; disruption of the axillary anastomosis with arm strain; infection along the tunnel')
             + opt('Femorofemoral cross-over', '<b>unilateral</b> iliac occlusion with a healthy donor iliac (or one stented first); high risk or hostile abdomen; failed iliac stenting',
                   'untreated disease in the donor iliac; bilateral disease; (relative) marked obesity', 'about 70% primary, 85% secondary', 'groin infection; steal from the donor leg (rare)')
             + opt('Endovascular (kissing stents, CERAB)', 'most TASC A–C lesions, and increasingly D in experienced centres',
                   'heavy calcified occlusion into the common femoral (hybrid: femoral endarterectomy plus stent); occlusion up to the renal arteries; no access',
                   'covered stents about 75% (COBEST); CERAB 82% at 3 years', 'access complications; stent occlusion; reintervention')
             + ev('de Vries and Hunink (J Vasc Surg 1997): aortic bifurcation grafts, 5-year limb patency 91% (claudication) and 87.5% (CLI), mortality 4.4%. Martin and Katz (Am J Surg 2000): axillofemoral primary patency 63% at 5 years. Park et al. (Vasc Specialist Int 2017): femorofemoral 70% primary, 85% secondary at 5 years; donor-iliac disease was treated first when present. COBEST (J Vasc Surg 2016); CERAB (Taeymans, J Vasc Surg 2018). Indications and contraindications: StatPearls/Medscape reviews of each operation.'))

    def depict(key, steps_):
        """show the operation: take out the native aortic segment the lesion replaces; thrombus, lumbar arteries, the opened
        sac, suture lines and wires where they belong; add the reference tables and the bypass choice"""
        for s in steps_:
            sid = s['id']; sh = s.setdefault('show', [])
            def add(*xs):
                for x in xs:
                    if has(x) and x not in sh: sh.append(x)
            act = s.get('action') or {}
            if key in SAC_OF and sid.endswith('-sac'):
                if key == 'aaa-infra':
                    add('aaa-thrombus', 'lumbar-arteries'); s['labels'] = [*s.get('labels', []), 'aaa-thrombus', 'lumbar-arteries']
                    act.update({'label': 'Open the sac and clear the thrombus', 'remove': ['aaa-thrombus']}); s['action'] = act
                    s['danger'] = [*s.get('danger', []), 'lumbar-arteries']
            if key in SAC_OF and sid.endswith('-graft'):
                an = ANAST_OF[key]; add(an)
                if key == 'aaa-infra':
                    s['show'] = [i for i in sh if i != 'aaa-infra']; add('aaa-sac-open', 'lumbar-arteries'); s['opacity'] = {**s.get('opacity', {}), 'aaa-sac-open': 0.55}
                else:
                    add(SAC_OF[key]); s['opacity'] = {**s.get('opacity', {}), SAC_OF[key]: 0.2}
                if act.get('ids'): act['ids'] = [*act['ids'], an]
                s['labels'] = [*s.get('labels', []), an]
            if key in SAC_OF and sid.endswith('-close'):
                add(SAC_OF[key], ANAST_OF[key]); s['opacity'] = {**s.get('opacity', {}), SAC_OF[key]: 0.25}
            if key == 'aaa-evar':
                if sid.endswith('-access'):
                    add('evar-wires'); s['highlight'] = [*s.get('highlight', []), 'evar-wires']; s['labels'] = [*s.get('labels', []), 'evar-wires']
                    s['action'] = {'kind': 'reveal', 'label': 'Pass the wires and sheaths', 'port': 'laparotomy', 'ids': ['evar-wires']}
                    s['view'] = front((Pv('cfa-r') + Pv('cfa-l')) / 2 * 0.5 + Pv('evar-neck') * 0.5, 820, (0.1, 1, 0.25))
                if sid.endswith('-deploy'): add('evar-wires'); s['opacity'] = {**s.get('opacity', {}), 'evar-wires': 0.6}
            if key == 'aiod-endo':
                if sid.endswith('-access'):
                    add('kissing-wires'); s['highlight'] = [*s.get('highlight', []), 'kissing-wires']; s['labels'] = [*s.get('labels', []), 'kissing-wires']
                    s['action'] = {'kind': 'reveal', 'label': 'Cross the occlusions', 'port': 'laparotomy', 'ids': ['kissing-wires']}
                    s['view'] = front((Pv('cfa-r') + Pv('cfa-l')) / 2 * 0.55 + Pv('bifurcation') * 0.45, 700, (0.1, 1, 0.25))
                if sid.endswith('-stents'): add('kissing-wires'); s['opacity'] = {**s.get('opacity', {}), 'kissing-wires': 0.7}
                if sid.endswith(('-access', '-stents', '-after')):
                    s['opacity'] = {**s.get('opacity', {}), 'aorta': 0.35, **{f'{v}-{x}': 0.35 for v in ('cia', 'eia', 'cfa') for x in 'lr'}}
            if key == 'taa-tevar' and sid.endswith('-deploy'):
                add('tevar-wire'); s['labels'] = [*s.get('labels', []), 'tevar-wire']
                if act.get('ids'): act['ids'] = ['tevar-wire', *act['ids']]
            if key in ('aiod-abf', 'taa-open') and sid.endswith('-graft') and act.get('ids'):
                an = ANAST_OF[key]; add(an); act['ids'] = [*act['ids'], an]; s['labels'] = [*s.get('labels', []), an]
            if key in ('aiod-abf', 'taa-open') and sid.endswith(('-close', '-after')): add(ANAST_OF[key])
            # the native aorta without the segment that the lesion or its replacement occupies
            shown = set(s['show'])
            for ids, cut_ in CUTS:
                if shown & ids and has(cut_):
                    for kk in ('show', 'highlight', 'labels', 'danger'):
                        if kk in s: s[kk] = [cut_ if i == 'aorta' else i for i in s[kk]]
                    if cut_ not in s['show']: s['show'].append(cut_)
                    op_ = s.get('opacity', {})
                    if 'aorta' in op_: op_[cut_] = op_.pop('aorta')
                    s['hide'] = [*s.get('hide', []), 'aorta']
                    if (s.get('action') or {}).get('kind') == 'clamp' and 'ids' in s['action']: s['action']['ids'] = [cut_ if i == 'aorta' else i for i in s['action']['ids']]
                    break
            # reference tables
            if s['phase'] == 'Pathophysiology':
                if key.startswith('aaa') or key.startswith('taa'):
                    if SIZES not in s['body']: s['body'] = s['body'] + SIZES
                if key.startswith('aiod') and RUTH not in s['body']: s['body'] = s['body'] + RUTH
        if key.startswith('aiod'):
            i = next((j for j, s in enumerate(steps_) if s['phase'] == 'Case'), None)
            if i is not None:
                c = steps_[i]
                steps_.insert(i + 1, {'id': c['id'].replace('-case', '-which'), 'phase': 'Decision', 'seq': c['seq'], 'title': 'Which reconstruction? Aortobifemoral, axillobifemoral, femorofemoral or endovascular',
                                      'body': '<p>The choice rests on <b>the extent of disease</b> (unilateral or bilateral, how far up the aorta, the common femorals), <b>the patient</b> (fitness for laparotomy, the abdomen, life expectancy, infection) and <b>the inflow</b> for an extra-anatomic graft.</p>' + WHICH,
                                      'view': c['view'], 'show': list(c.get('show', [])), 'hide': list(c.get('hide', [])), 'opacity': dict(c.get('opacity', {})),
                                      'labels': [i for i in ('aiod-occlusion', 'cia-l', 'cia-r', 'cfa-l', 'cfa-r') if has(i)], 'ct': c.get('ct'),
                                      'ask': ask('A 70-year-old with a right common iliac occlusion, a normal left iliac on CT, and severe COPD has rest pain in the right foot. Stenting has failed. Which bypass?', 'Femorofemoral cross-over from the left groin',
                                                 'Unilateral disease with a healthy donor iliac is the classic indication; it avoids a laparotomy and has better patency than an axillofemoral graft.', 'Aortobifemoral bypass', 'Axillobifemoral bypass', 'Primary amputation')})

    def finish(key, op_, opName, appr, summ, steps_, sq, src, side='both'):
        depict(key, steps_)
        for s in steps_:
            named = set(s.get('highlight', [])) | set(s.get('danger', [])) | set(s.get('labels', [])) | set(s.get('show', []))
            s['hide'] = [*s.get('hide', []), *[i for i in V_OFF if i not in named], *[i for i in VPATH if i not in named],
                         *[k for k in S if k.startswith('rib-') and k not in named]]
            s['opacity'] = {**{f'vert-t{i}': 0.22 for i in range(2, 13)}, **s.get('opacity', {})}
            for kk in ('highlight', 'danger', 'labels', 'show', 'hide'):
                if kk in s: s[kk] = [i for i in s[kk] if has(i) or i == 'skin']
        procs[key] = {'id': key, 'op': op_, 'opName': opName, 'side': side, 'name': opName, 'approach': appr, 'summary': summ,
                      'ports': [], 'steps': steps_, 'sources': src, 'sequence': sq, 'group': 'Vascular'}

    # ================================================================================================ abdominal aortic aneurysm
    RUP = ('<table class="mini"><tr><th>Diameter</th><th>1-year risk of (probable) rupture, untreated</th></tr>'
           '<tr><td>5.5–5.9 cm</td><td>9.4%</td></tr><tr><td>6.0–6.9 cm</td><td>10.2% (19.1% at 6.5–6.9)</td></tr><tr><td>7.0 cm or more</td><td>32.5%</td></tr></table>')
    NECK = ('<table class="mini"><tr><th>Type (SVS 2021)</th><th>Where the aneurysm starts</th><th>Proximal clamp and repair</th></tr>'
            '<tr><td><b>Infrarenal</b></td><td>a healthy neck of 10–15 mm or more below the lowest renal artery</td><td>infrarenal clamp; EVAR usually possible</td></tr>'
            '<tr><td>Short neck</td><td>neck 4–10 mm</td><td>often suprarenal clamp; EVAR marginal</td></tr>'
            '<tr><td><b>Juxtarenal</b></td><td>at the renal arteries (neck 4 mm or less), not involving them</td><td>suprarenal clamp, anastomosis at the renal origins; or fenestrated EVAR</td></tr>'
            '<tr><td><b>Pararenal / paravisceral</b> ("suprarenal")</td><td>involving the renal arteries, up to (pararenal) or including (paravisceral) the SMA</td><td>supracoeliac clamp, renal and visceral reconstruction; or fenestrated/branched EVAR</td></tr>'
            '<tr><td>Thoracoabdominal extent IV</td><td>up to the coeliac axis or diaphragmatic hiatus</td><td>thoracoabdominal repair</td></tr></table>')
    aaa_patho = vstep('aaa-patho', 'Pathophysiology', 'Pathophysiology: abdominal aortic aneurysm',
        '<p><b>What fails is the media.</b> Elastin is broken down (matrix metalloproteinases from macrophages and smooth muscle cells), smooth muscle cells die, and chronic inflammation involves the adventitia. The infrarenal aorta has the fewest vasa vasorum and the least elastin: most aneurysms start there. Risk factors: <b>age, male sex, smoking</b> (the strongest modifiable), family history, hypertension; diabetes is, oddly, protective.</p>'
        + chain('Elastin degradation, loss of smooth muscle', 'Dilatation', 'Wall tension rises with radius (Laplace: T = P × r)', '!More dilatation, faster growth', '!Rupture')
        + '<p>An aneurysm is a diameter of <b>3.0 cm or more</b>. Growth accelerates as it enlarges; rupture risk climbs steeply above 5.5 cm:</p>' + RUP
        + '<p><b>Repair thresholds</b>: 5.5 cm in men, 5.0 cm in women (smaller aortas rupture at smaller diameters), rapid growth (about 1 cm a year, confirmed), or symptoms (tenderness, back pain). Below that, <b>surveillance</b>: early repair of 4.0–5.5 cm aneurysms did not improve survival.</p>'
        '<p><b>The neck</b> decides the operation:</p>' + NECK
        + '<p><b>In Africa</b> aortic aneurysms present younger and more often from <b>infection or HIV-associated vasculitis</b> (saccular, at atypical sites, sometimes multiple); many present ruptured. Always consider an infective (mycotic) aneurysm: fever, raised inflammatory markers, a saccular or lobulated shape, periaortic gas or fluid.</p>'
        + ev('rupture rates: Lederle et al., JAMA 2002 (198 patients refusing or unfit for repair). Small aneurysms: UK Small Aneurysm Trial 12-year follow-up (Powell et al., Br J Surg 2007: no survival difference, HR 0.90) and ADAM (NEJM 2002: RR 1.21). Thresholds and surveillance: ESVS 2024 (Wanhainen et al.). Anatomical definitions: SVS reporting standards (Oderich et al., J Vasc Surg 2021). HIV-associated aneurysms: Nair et al. (J Vasc Surg 1999; mean age 30, often multiple, atypical sites). African AAA: Wu, J West Afr Coll Surg 2022.'),
        front(ABD), show=[*VESS, 'aaa-infra'], highlight=['aaa-infra'], labels=['aaa-infra', 'renal-a-l', 'renal-v-l', 'sma', 'ima', 'cia-r'], opacity={**AB_OP, 'aaa-infra': 0.6},
        quiz=ask('Why does an aneurysm grow faster as it gets bigger?', 'Wall tension rises with the radius (Laplace\'s law), so the larger the aneurysm, the greater the stress on an already weakened wall',
                 'T = P × r: at the same blood pressure a 6 cm aneurysm carries twice the wall tension of a 3 cm aorta. That is why growth and rupture risk accelerate with size.', 'Blood flow slows', 'Thrombus lining the sac weakens it', 'Blood pressure rises with size'),
        after=True, spin=True)
    aaa_anat = vstep('aaa-anatomy', 'Anatomy', 'The infrarenal aorta and its neighbours',
        '<p>From above down on the front of the aorta: the <b>coeliac trunk</b> (T12/L1), the <b>SMA</b> about 1 cm lower, the <b>renal arteries</b> (L1/L2, the right passing behind the IVC), the <b>left renal vein</b> crossing in front of the aorta just below the SMA (the upper limit of the infrarenal neck), the <b>IMA</b> from the left front at L3, and the bifurcation at L4.</p>'
        '<p>What to protect: the <b>left renal vein</b> above the neck (and a retroaortic or circumaortic variant, which tears when clamping); the <b>duodenum</b> (3rd and 4th parts) over the aneurysm; the <b>IVC</b> to the right and the <b>iliac veins</b> behind the iliac arteries (the left common iliac vein runs behind the right common iliac artery): encircling the iliac arteries tears them; the <b>ureters</b> crossing the iliac bifurcations; the <b>superior hypogastric plexus</b> on the left of the aorta and over the left common iliac (dissect on the right to preserve ejaculation); the <b>lumbar arteries</b> behind, which back-bleed into the opened sac.</p>'
        + BELOW,
        front(ABD, 340), show=VESS, highlight=['renal-v-l'], danger=hv(['renal-v-l', 'ivc-infra', 'civ-l', 'civ-r']), labels=['coeliac', 'sma', 'renal-a-l', 'renal-a-r', 'renal-v-l', 'ima', 'ivc-infra', 'civ-l'], opacity=AB_OP,
        quiz=ask('Which vein lies behind the right common iliac artery and is torn by careless encircling of that artery?', 'The left common iliac vein (near the confluence of the IVC)',
                 'The iliac veins lie behind and slightly to the right of the arteries; the left common iliac vein crosses behind the right common iliac artery to reach the IVC. Clamp the iliacs without encircling them.', 'The left renal vein', 'The inferior mesenteric vein', 'The gonadal vein'),
        spin=True)
    LAP = hv(['incision-laparotomy'])
    lap_step = lambda pre: vstep(f'{pre}-lap', 'Access', 'Midline laparotomy; expose the aorta',
        '<p>Supine, arms out; an epidural or good analgesia; arterial line, central line, cell salvage, warming; prophylactic antibiotics. Prepare from nipples to knees (the groins in the field).</p>'
        '<p><b>Midline laparotomy</b> from the xiphoid to the pubis. Lift the transverse colon up, the small bowel to the right in a bag; incise the <b>posterior peritoneum</b> to the right of the duodenojejunal flexure, mobilise the <b>4th part of the duodenum</b> to the right (divide the ligament of Treitz and the inferior mesenteric vein if needed), and open the peritoneum down over the aneurysm to the bifurcation, staying to the <b>right</b> of the IMA and the hypogastric plexus.</p>',
        front(ABD, 520, (0.05, 1, 0.45)), show=['skin', *LAP, *VESS], highlight=LAP, labels=LAP, opacity={**AB_OP, 'skin': 0.35},
        action={'kind': 'reveal', 'label': 'Open the abdomen', 'port': 'laparotomy', 'ids': LAP} if LAP else None)
    def neck_step(pre, sac_id, level='infra'):
        tgt = Pv('evar-neck')
        body = {'infra': '<p>Find the <b>left renal vein</b> crossing the neck and retract it upward; dissect the neck below it on both sides down to the vertebral body, enough for a clamp front-to-back (no need to encircle). Identify the <b>renal arteries</b> above.</p>',
                'juxta': '<p>The aneurysm reaches the renal arteries: the clamp must go <b>above</b> them. Mobilise the <b>left renal vein</b>; if it prevents access, <b>divide it close to the IVC</b>, keeping its gonadal, adrenal and lumbar tributaries (they drain the kidney afterwards). Expose the aorta between the renal arteries and the SMA, taking care of the SMA origin and the renal ostia.</p>',
                'supra': '<p>Through the left retroperitoneum: sweep the <b>left kidney</b> (or leave it posterior), the colon and the spleen forward; divide the <b>left crus of the diaphragm</b> to reach the supracoeliac aorta. The left renal artery is the landmark: follow it to the aorta. Expose the coeliac, SMA and both renal origins.</p>'}[level]
        return vstep(f'{pre}-neck', 'Exposure', 'Expose the neck' + (' and the renal vein' if level != 'supra' else ' and the visceral aorta'), body,
                     front(tgt, 260, (0.2, 1, 0.5)), show=[*VESS, sac_id], highlight=hv(['renal-v-l']) if level != 'supra' else hv(['coeliac', 'sma']),
                     danger=hv(['renal-v-l', 'renal-a-l', 'renal-a-r', 'sma']), labels=hv(['renal-v-l', 'renal-a-l', 'renal-a-r', 'sma', 'coeliac']), opacity={**AB_OP, sac_id: 0.55})
    iliac_step = lambda pre, sac_id: vstep(f'{pre}-iliacs', 'Exposure', 'Control the iliac arteries (do not encircle them)',
        '<p>Open the peritoneum over both <b>common iliac arteries</b>, see the <b>ureters</b> crossing their bifurcations, and prepare clamp sites on healthy artery. <b>Do not encircle</b> the iliac arteries: the iliac veins are stuck behind them. If the iliacs are aneurysmal too, go to the external and internal iliac arteries.</p>'
        + BELOW, front(Pv('bifurcation'), 300, (0.1, 1, 0.4)), show=[*VESS, sac_id], highlight=hv(['cia-l', 'cia-r']), danger=hv(['civ-l', 'civ-r', 'ivc-infra']), labels=hv(['cia-l', 'cia-r', 'civ-l', 'civ-r', 'iia-l']), opacity={**AB_OP, sac_id: 0.5})
    def clamp_step(pre, level, sac_id):
        z = {'infra': Pv('evar-neck'), 'juxta': (Pv('sma') + Pv('renal-l')) / 2, 'supra': Pv('coeliac') + V([0, 0, 14])}[level]
        txt = {'infra': 'the <b>infrarenal neck</b>, below the renal arteries',
               'juxta': 'the aorta <b>between the renal arteries and the SMA</b> (suprarenal), or above the SMA if there is no room; the renal arteries are now ischaemic: note the time',
               'supra': 'the <b>supracoeliac</b> aorta at the hiatus: the liver, bowel and kidneys are all ischaemic; work fast, and <b>perfuse the kidneys with cold crystalloid</b>'}[level]
        return vstep(f'{pre}-clamp', 'Clamp', 'Heparin, clamp distally, then proximally',
            f'<p>Heparin (about 70–100 U/kg). Clamp the <b>iliac arteries first</b> (so that debris from the sac goes nowhere), then {txt}. Tell the anaesthetist before clamping: afterload rises; before releasing, fluid and vasoconstrictor ready.</p>'
            + (ev('ESVS 2024: consider cold renal perfusion when the suprarenal clamp time is expected to exceed about 25 minutes. In a systematic review of 1,256 open juxtarenal repairs (Jongkind et al., J Vasc Surg 2010), 30-day mortality was 2.9% and 3.3% needed new dialysis.') if level != 'infra' else ''),
            front(z, 240, (0.2, 1, 0.5)), show=[*VESS, sac_id], highlight=['aorta'], danger=hv(['renal-a-l', 'renal-a-r', 'sma', 'renal-v-l']), labels=hv(['renal-a-l', 'sma', 'renal-v-l']), opacity={**AB_OP, sac_id: 0.45},
            action={'kind': 'clamp', 'label': 'Clamp the aorta', 'port': 'laparotomy' if level != 'supra' else 'flank-l', 'at': R(z), 'axis': [0, 0.3, 1], 'radius': 13, 'jawLen': 60})
    open_sac = lambda pre, sac_id: vstep(f'{pre}-sac', 'Sac', 'Open the sac, clear the thrombus, stop the back-bleeding',
        '<p>Open the aneurysm longitudinally on its right anterior surface (away from the IMA), T-ing the incision at the neck and at the bifurcation. Scoop out the laminated <b>thrombus</b>. <b>Oversew the lumbar arteries</b> from inside with figure-of-eight 2-0 or 3-0 sutures; back-bleeding from the <b>IMA</b>: if it bleeds briskly (good collaterals) oversew its origin from inside; if the flow is poor and the colon may depend on it, keep a button for reimplantation.</p>'
        + ev('ESVS 2024: routine IMA reimplantation is not recommended; consider it when pelvic and colonic perfusion is doubtful (both internal iliacs diseased or excluded, previous colectomy, poor back-bleeding). Colonic ischaemia after open repair presents with bloody diarrhoea, acidosis or distension: sigmoidoscopy early.'),
        front(ABD, 260, (0.3, 1, 0.3)), show=[*VESS, sac_id], highlight=[sac_id], danger=hv(['ima']), labels=hv([sac_id, 'ima']), opacity={**AB_OP, sac_id: 0.35},
        action={'kind': 'dissect', 'tool': 'hook', 'label': 'Open the sac', 'port': 'laparotomy', 'path': [R(Pv('evar-neck') + V([0, 25, -10])), R(ABD + V([0, 30, 0])), R(Pv('bifurcation') + V([0, 25, 15]))]})
    def graft_step(pre, graft, level):
        body = {'infra': '<p><b>Proximal anastomosis</b> first: a straight <b>Dacron tube</b> (or bifurcated if the iliacs are aneurysmal), running 3-0 polypropylene, full thickness of the neck with generous bites (buttressing with felt strips if the wall is poor). Test it by releasing the clamp briefly with the graft clamped.</p>'
                        '<p><b>Distal anastomosis</b> to the bifurcation (tube) or to the iliac arteries (bifurcated), flushing before the last sutures (release the iliac clamps briefly, then the aortic) to wash out air and debris.</p>',
                'juxta': '<p><b>Proximal anastomosis at the level of the renal arteries</b>, the back wall sewn from inside, the suture line just below the renal ostia (so neither is narrowed). Then <b>move the clamp below the renal arteries onto the graft</b> as soon as the proximal suture line is done: renal ischaemia ends. The rest as for an infrarenal repair.</p>',
                'supra': '<p><b>Bevelled proximal anastomosis</b>: the graft cut obliquely so that its suture line incorporates the <b>coeliac, SMA and right renal origins</b> as one patch (a Carrel patch), sewn with 3-0 polypropylene. The <b>left renal artery</b> is reimplanted as a button or bypassed with a side-arm. Move the clamp down onto the graft below the visceral origins in sequence: the viscera, then each kidney.</p>'}[level]
        return vstep(f'{pre}-graft', 'Graft', 'Sew in the graft', body, front(ABD, 280, (0.2, 1, 0.35)),
                     show=[*VESS, graft], highlight=[graft], danger=hv(['renal-a-l', 'renal-a-r', 'sma']), labels=hv([graft, 'renal-a-l', 'renal-a-r', 'sma']), opacity=AB_OP,
                     action={'kind': 'reveal', 'label': 'Sew in the graft', 'port': 'laparotomy' if level != 'supra' else 'flank-l', 'ids': [graft]},
                     quiz=ask('Before tying the distal anastomosis, why flush the graft?', 'To wash out air, thrombus and debris that would otherwise embolise to the legs or pelvis',
                              'Brief release of the iliac, then aortic clamps flushes the graft; debris left in it embolises to the feet ("trash foot") or the pelvis.', 'To test the proximal anastomosis only', 'It is not needed', 'To reduce heparin effect'))
    close_aaa = lambda pre, graft, extra='': vstep(f'{pre}-close', 'Close', 'Release, close the sac over the graft, check the colon and the feet',
        '<p>Release the clamps <b>one leg at a time</b>, slowly, with the anaesthetist (declamping hypotension). Protamine if needed. <b>Close the sac over the graft</b> and then the posterior peritoneum, so the graft never touches the duodenum (aortoenteric fistula). Look at the <b>sigmoid colon</b> (pink, peristalsing, a mesenteric pulse) and the <b>feet</b> (pulses, colour) before closing.</p>' + extra,
        front(ABD, 340, (0.1, 1, 0.35)), show=[*VESS, graft], highlight=[], danger=[], labels=[graft], opacity=AB_OP,
        quiz=ask('Why close the aneurysm sac and peritoneum over the graft?', 'To keep the graft away from the duodenum and prevent an aortoenteric fistula',
                 'Direct contact between the graft suture line and the duodenum can erode into the bowel months or years later: a catastrophic bleed.', 'To stop back-bleeding', 'To help the graft endothelialise', 'For cosmetic reasons'))
    # --------------------------------------------------------------------- infrarenal open
    c_inf = vstep('ai-case', 'Case', 'Case: a 6.2 cm infrarenal aneurysm in a fit 64-year-old',
        '<p><b>Read the case above.</b> The steps that follow are his operation.</p>'
        '<p><b>Decision</b>: a 6.2 cm aneurysm (1-year rupture risk about 10%) in a fit man: <b>repair</b>. Open or EVAR? EVAR has lower early mortality, but the benefit is lost over the years, it needs <b>lifelong imaging surveillance</b> and more reinterventions, and late rupture occurs. For a fit 64-year-old with a long life expectancy and limited access to CT surveillance, <b>open repair</b> is a sound choice; EVAR is equally valid where surveillance is reliable.</p>'
        + ev('EVAR-1 (Lancet 2016; mean follow-up 12.7 years): EVAR lower total and aneurysm-related mortality in the first 6 months, but after 8 years higher aneurysm-related mortality (mainly from sac rupture) and more reinterventions. OVER (NEJM 2019) and DREAM (NEJM 2010): long-term survival similar. ESVS 2024: EVAR remains the preferred modality in most patients, weighing durability, life expectancy, and patient preference.'),
        front(ABD), show=[*VESS, 'aaa-infra'], highlight=['aaa-infra'], labels=['aaa-infra', 'renal-v-l'], opacity={**AB_OP, 'aaa-infra': 0.6},
        lead='<p>A <b>64-year-old man</b>, a retired teacher from Nakuru, hypertensive, ex-smoker. An abdominal pulsation felt at a medical check; ultrasound, then CT angiography: a <b>6.2 cm infrarenal aneurysm</b>, neck 25 mm long and 22 mm wide, iliacs 14 mm, no iliac aneurysm. Creatinine normal; echo EF 60%; he walks 3 km a day. He lives 4 hours from the nearest CT scanner.</p>',
        quiz=ask('What is the main long-term drawback of EVAR compared with open repair?', 'The need for lifelong imaging surveillance and reinterventions, with a risk of late rupture',
                 'EVAR excludes the sac without removing it; endoleaks and migration can re-pressurise it. EVAR-1 showed more reinterventions and higher late aneurysm mortality after 8 years.', 'Higher 30-day mortality', 'Longer hospital stay', 'More blood loss'))
    inf_steps, inf_sq = build_v([('Patho', 'other', [aaa_patho]), ('Anatomy', 'other', [aaa_anat]), ('Case', 'other', [c_inf]), ('Laparotomy', 'other', [lap_step('ai')]),
                                 ('Neck', 'other', [neck_step('ai', 'aaa-infra')]), ('Iliacs', 'artery', [iliac_step('ai', 'aaa-infra')]), ('Clamp', 'artery', [clamp_step('ai', 'infra', 'aaa-infra')]),
                                 ('Sac', 'artery', [open_sac('ai', 'aaa-infra')]), ('Graft', 'artery', [graft_step('ai', 'graft-tube', 'infra')]), ('Close', 'other', [close_aaa('ai', 'graft-tube')])])
    for s in inf_steps:
        s['id'] = s['id'].replace('aaa-', 'ai-', 1) if s['id'].startswith('aaa-') else s['id']
    finish('aaa-infra', 'aaa', 'Abdominal aortic aneurysm', 'Infrarenal, open repair', 'Transperitoneal open repair of an infrarenal aneurysm: neck, iliacs, clamps, sac, tube graft, closure.', inf_steps, inf_sq, AAA_SRC)
    # --------------------------------------------------------------------- juxtarenal open
    c_jx = vstep('aj-case', 'Case', 'Case: a juxtarenal aneurysm with no neck',
        '<p><b>No infrarenal neck</b> (under 4 mm): a standard EVAR cannot seal. The options are <b>open repair with a suprarenal clamp</b>, or a <b>fenestrated</b> stent graft (FEVAR, custom-made or off-the-shelf, where available). At standard surgical risk, either is reasonable; chimney grafts are for emergencies or bailout.</p>'
        '<p><b>Protect the kidneys</b>: preoperative hydration, avoid nephrotoxins, keep suprarenal clamp time short, cold renal perfusion if it will exceed about 25 minutes.</p>'
        + ev('ESVS 2024: for complex AAA at standard risk, open or endovascular repair based on fitness, anatomy and preference; fenestrated/branched EVAR first line at high surgical risk; parallel (chimney) grafts only in emergencies or as bailout. Jongkind et al. (J Vasc Surg 2010; 1,256 open juxtarenal repairs): 30-day mortality 2.9%, new dialysis 3.3%.'),
        front(ABD), show=[*VESS, 'aaa-juxta'], highlight=['aaa-juxta'], labels=['aaa-juxta', 'renal-a-l', 'renal-a-r', 'renal-v-l'], opacity={**AB_OP, 'aaa-juxta': 0.6},
        lead='<p>A <b>70-year-old man</b> with a <b>6.0 cm</b> aneurysm on CT; the sac begins <b>2 mm below the renal arteries</b>; the renal arteries and SMA are free of disease; eGFR 68. Fit (climbs two flights), no previous abdominal surgery. Fenestrated devices are not available in-country.</p>',
        quiz=ask('Where must the proximal clamp go for this juxtarenal aneurysm?', 'Above the renal arteries (between the renals and the SMA, or above the SMA)',
                 'There is no healthy infrarenal aorta to clamp; the anastomosis is made at the level of the renal ostia, then the clamp is moved onto the graft below them to restore renal flow.', 'Infrarenal, as usual', 'On the iliac arteries only', 'Supracoeliac is always required'))
    jx_steps, jx_sq = build_v([('Patho', 'other', [{**aaa_patho, 'id': 'aj-patho', 'show': [*VESS, 'aaa-juxta'], 'highlight': ['aaa-juxta'], 'labels': ['aaa-juxta', 'renal-a-l', 'renal-v-l', 'sma'], 'opacity': {**AB_OP, 'aaa-juxta': 0.6}}]),
                               ('Anatomy', 'other', [{**aaa_anat, 'id': 'aj-anatomy'}]), ('Case', 'other', [c_jx]), ('Laparotomy', 'other', [lap_step('aj')]),
                               ('Renal vein', 'vein', [neck_step('aj', 'aaa-juxta', 'juxta')]), ('Iliacs', 'artery', [iliac_step('aj', 'aaa-juxta')]), ('Clamp', 'artery', [clamp_step('aj', 'juxta', 'aaa-juxta')]),
                               ('Sac', 'artery', [open_sac('aj', 'aaa-juxta')]), ('Graft', 'artery', [graft_step('aj', 'graft-juxta', 'juxta')]),
                               ('Close', 'other', [close_aaa('aj', 'graft-juxta', '<p>If the left renal vein was divided, check the kidney\'s colour; reconstruction may be considered if its collaterals were sacrificed.</p>')])])
    finish('aaa-juxta', 'aaa', 'Abdominal aortic aneurysm', 'Juxtarenal, open (suprarenal clamp)', 'No infrarenal neck: renal vein, suprarenal clamp, anastomosis at the renal arteries, clamp moved down.', jx_steps, jx_sq, AAA_SRC)
    # --------------------------------------------------------------------- suprarenal (pararenal/paravisceral), left retroperitoneal
    FL = hv(['incision-flank-l'])
    c_sp = vstep('as-case', 'Case', 'Case: a pararenal aneurysm through the left flank',
        '<p><b>The renal arteries arise from the aneurysm</b>: the proximal anastomosis must be above them, incorporating the visceral and renal origins. The <b>left retroperitoneal</b> approach gives the best access to the aorta above the renal arteries and up to the hiatus, and avoids a hostile abdomen (her previous laparotomy).</p>'
        '<p>Alternatives: fenestrated or branched EVAR at high risk, where available.</p>'
        + ev('Sicard et al. (J Vasc Surg 1995, randomised, 145 patients): the retroperitoneal approach had less ileus and shorter ICU stay; Cambria et al. (1990, 113 patients) found no important difference. ESVS 2024: the approach is chosen on patient factors and surgeon preference. Jongkind et al. 2010: renal dysfunction after complex open repair was common (median 18%).'),
        front(ABD + V([0, 0, 30])), show=[*VESS, 'aaa-supra'], highlight=['aaa-supra'], labels=['aaa-supra', 'renal-a-l', 'renal-a-r', 'sma', 'coeliac'], opacity={**AB_OP, 'aaa-supra': 0.6},
        lead='<p>A <b>66-year-old woman</b>, hypertensive, a previous laparotomy for a perforated duodenal ulcer. CT angiography: a <b>6.5 cm</b> aneurysm that <b>involves both renal artery origins</b> and reaches 1 cm below the SMA; the coeliac and SMA origins are healthy; eGFR 55.</p>',
        quiz=ask('Why the left flank (retroperitoneal) approach here?', 'It exposes the aorta above the renal arteries up to the hiatus and avoids her scarred abdomen',
                 'The left retroperitoneal route gives direct access to the para- and supravisceral aorta (dividing the left crus) and stays out of the peritoneal cavity; the right renal artery is harder to reach from this side.', 'It gives better access to the right iliac artery', 'It is quicker for infrarenal aneurysms', 'It avoids the need for a suprarenal clamp'))
    flank = vstep('as-flank', 'Access', 'Left flank incision; the retroperitoneal plane',
        '<p>Right side down, the left shoulder rotated back about 60°, the pelvis flatter (the "corkscrew" position), the table broken at the flank. Incision from the tip of the <b>11th rib</b> posteriorly, obliquely to the lateral edge of the rectus sheath below the umbilicus; resect or enter the bed of the 11th rib. Divide the abdominal muscles; stay <b>outside the peritoneum</b>, sweeping it, the colon and the spleen forward. Take the <b>left kidney forward</b> with them (exposing the posterior aorta and the left renal artery) or leave it in place.</p>',
        tl(Pv('flank-l') if 'flank-l' in LM else ABD, (-1, 0.3, 0.3), 460), show=['skin', *FL, *VESS, 'aaa-supra'], highlight=FL, labels=FL, opacity={**AB_OP, 'skin': 0.35, 'aaa-supra': 0.4},
        action={'kind': 'reveal', 'label': 'Open the flank', 'port': 'flank-l', 'ids': FL} if FL else None)
    renal_cool = vstep('as-renal', 'Kidneys', 'Cold renal perfusion; reimplant the left renal',
        '<p>With the supracoeliac clamp on and the sac opened, place <b>balloon perfusion catheters</b> into both renal ostia and infuse <b>cold crystalloid</b> (about 4 °C, commonly Ringer\'s lactate with mannitol, per unit protocol) intermittently; visceral ischaemia time is the other clock: keep it short.</p>'
        '<p>After the bevelled proximal suture line, the <b>left renal artery</b> goes onto a side-arm or as a button reimplanted into the graft; restore flow to the viscera and right kidney first by moving the clamp down onto the graft.</p>'
        + ev('ESVS 2024 (Rec 125): consider cold renal perfusion when suprarenal clamp time exceeds about 25 minutes. The perfusate temperature and composition vary by centre.'),
        front(Pv('renal-l'), 220, (-0.4, 1, 0.4)), show=[*VESS, 'graft-supra'], highlight=hv(['renal-a-l', 'renal-a-r']), danger=hv(['sma', 'coeliac']), labels=hv(['renal-a-l', 'renal-a-r', 'sma', 'graft-supra']), opacity=AB_OP)
    sp_steps, sp_sq = build_v([('Patho', 'other', [{**aaa_patho, 'id': 'as-patho', 'show': [*VESS, 'aaa-supra'], 'highlight': ['aaa-supra'], 'labels': ['aaa-supra', 'renal-a-l', 'renal-a-r', 'sma'], 'opacity': {**AB_OP, 'aaa-supra': 0.6}}]),
                               ('Anatomy', 'other', [{**aaa_anat, 'id': 'as-anatomy'}]), ('Case', 'other', [c_sp]), ('Flank', 'other', [flank]),
                               ('Visceral aorta', 'artery', [neck_step('as', 'aaa-supra', 'supra')]), ('Clamp', 'artery', [clamp_step('as', 'supra', 'aaa-supra')]),
                               ('Sac', 'artery', [{**open_sac('as', 'aaa-supra'), 'action': {**open_sac('as', 'aaa-supra')['action'], 'port': 'flank-l'}}]),
                               ('Graft', 'artery', [graft_step('as', 'graft-supra', 'supra')]), ('Kidneys', 'artery', [renal_cool]),
                               ('Close', 'other', [close_aaa('as', 'graft-supra', '<p>Check both kidneys (colour, a pulse in each renal artery) and the bowel; urine output and lactate overnight.</p>')])])
    finish('aaa-supra', 'aaa', 'Abdominal aortic aneurysm', 'Suprarenal (pararenal), left retroperitoneal', 'Left flank approach, supracoeliac clamp, bevelled visceral patch, left renal reimplant, cold renal perfusion.', sp_steps, sp_sq, AAA_SRC)
    # --------------------------------------------------------------------- EVAR
    c_ev = vstep('ae-case', 'Case', 'Case: EVAR in a man with COPD',
        '<p><b>Anatomy suits EVAR</b>: a neck of at least 10–15 mm, not too angled or conical, of a diameter the device can seal (usually 18–32 mm), and iliac access vessels wide enough for the sheath (about 7 mm or more for most devices). <b>His lungs</b> make a laparotomy and a long aortic clamp riskier: EVAR avoids both.</p>'
        '<p><b>The contract</b>: lifelong surveillance (CT or duplex) for endoleaks and migration, and a greater chance of a secondary procedure.</p>'
        + ev('EVAR-1 (Lancet 2016): lower early mortality with EVAR, lost over time; more reinterventions. ESVS 2024: EVAR is the preferred modality in most patients with suitable anatomy; for a ruptured AAA, EVAR is the first option when anatomy allows (Class I). IMPROVE (BMJ 2014 and 2017): 30-day mortality 35.4% vs 37.4% for an endovascular strategy vs open repair in rupture, with better 3-year survival (48% vs 56% mortality) and quality of life.'),
        front(ABD), show=[*VESS, 'aaa-infra'], highlight=['aaa-infra'], labels=['aaa-infra', 'renal-a-l', 'cia-l', 'cia-r'], opacity={**AB_OP, 'aaa-infra': 0.6},
        lead='<p>A <b>77-year-old man</b>, COPD (FEV1 45%), ischaemic heart disease with a stent 3 years ago. A <b>5.9 cm</b> infrarenal aneurysm; neck <b>24 mm long</b>, 23 mm wide, angle 35°; common iliacs 13 mm and not aneurysmal; external iliacs 8 mm. Lives in Nairobi near a hospital with CT.</p>',
        quiz=ask('Which anatomical feature is essential for a standard EVAR?', 'A healthy infrarenal neck long enough for a seal (usually at least 10–15 mm), not too angled or conical',
                 'The stent graft seals by radial force against normal aorta below the renal arteries; a short, wide, angled or thrombus-lined neck causes type I endoleak and failure.', 'A small aneurysm sac', 'A patent IMA', 'Calcified iliac arteries'))
    ev_access = vstep('ae-access', 'Access', 'Femoral access; mark the renal arteries',
        '<p>Ultrasound-guided puncture of both <b>common femoral arteries</b> (over the femoral head), <b>preclosure</b> sutures placed (or a surgical cut-down in calcified or small arteries). Heparin. A stiff wire to the thoracic aorta; an angiogram with a marker catheter, the image angled to open the neck, to <b>mark the lowest renal artery</b>.</p>' + BELOW,
        front(Pv('cfa-r'), 360, (0.15, 1, 0.2)), show=[*VESS, 'aaa-infra', *hv(['incision-groin-l', 'incision-groin-r'])], highlight=hv(['cfa-l', 'cfa-r']), labels=hv(['cfa-r', 'cfa-l', 'eia-r']), opacity={**AB_OP, 'aaa-infra': 0.5})
    ev_deploy = vstep('ae-deploy', 'Deploy', 'Deploy the main body below the lowest renal; cannulate the gate; the limbs',
        '<p>Introduce the <b>main body</b> through one groin, align its top marker just below the lowest renal artery, and deploy (suprarenal fixation, where the device has it, sits across the renal arteries without covering them). <b>Cannulate the contralateral gate</b> from the other groin, confirm you are inside the graft, and deploy the <b>contralateral limb</b>, then the ipsilateral limb, landing both in the common iliac arteries above the internal iliac origins. Balloon the seal zones and junctions.</p>',
        front(ABD, 300, (0.15, 1, 0.3)), show=[*VESS, 'aaa-infra', 'evar-graft'], highlight=['evar-graft'], danger=hv(['renal-a-l', 'renal-a-r', 'iia-l', 'iia-r']), labels=hv(['evar-graft', 'renal-a-l', 'iia-r']), opacity={**AB_OP, 'aaa-infra': 0.35},
        action={'kind': 'reveal', 'label': 'Deploy the stent graft', 'port': 'groin-r', 'ids': ['evar-graft']},
        quiz=ask('The completion angiogram shows contrast filling the sac from around the top of the graft. What is it, and what now?', 'A type I (proximal seal) endoleak: treat it now (balloon, proximal cuff or anchors)',
                 'Type I and III endoleaks pressurise the sac and must be fixed; a type II (from lumbars or IMA) is usually observed.', 'A type II endoleak: observe', 'Normal: it will thrombose', 'Convert to open repair immediately'))
    ev_end = vstep('ae-surv', 'After', 'Completion angiogram, closure, lifelong surveillance',
        '<p>Completion angiogram: both renal arteries patent, no <b>type I</b> (seal) or <b>type III</b> (junction) endoleak, both internal iliacs perfused. Close the arteries. <b>Surveillance</b>: CT angiography (or contrast ultrasound) at about 30 days, then by protocol, lifelong: sac growth means an endoleak until proven otherwise.</p>'
        '<table class="mini"><tr><th>Endoleak</th><th>Source</th><th>Action</th></tr><tr><td>I</td><td>seal zone (proximal a, distal b)</td><td>treat</td></tr><tr><td>II</td><td>lumbar arteries or IMA back-filling</td><td>observe; treat if the sac grows</td></tr>'
        '<tr><td>III</td><td>junction or fabric tear</td><td>treat</td></tr><tr><td>IV</td><td>graft porosity</td><td>resolves</td></tr><tr><td>V</td><td>endotension (growth without visible leak)</td><td>investigate</td></tr></table>',
        front(ABD, 340), show=[*VESS, 'aaa-infra', 'evar-graft'], labels=['evar-graft'], opacity={**AB_OP, 'aaa-infra': 0.35})
    ev_steps, ev_sq = build_v([('Patho', 'other', [{**aaa_patho, 'id': 'ae-patho'}]), ('Anatomy', 'other', [{**aaa_anat, 'id': 'ae-anatomy'}]), ('Case', 'other', [c_ev]),
                               ('Access', 'other', [ev_access]), ('Deploy', 'artery', [ev_deploy]), ('After', 'other', [ev_end])])
    finish('aaa-evar', 'aaa', 'Abdominal aortic aneurysm', 'Infrarenal, EVAR', 'Femoral access, marking the renal arteries, main body, gate, limbs, completion angiogram, surveillance.', ev_steps, ev_sq, AAA_SRC)

    # ================================================================================================ aorto-iliac occlusive disease
    TASC = ('<table class="mini"><tr><th>TASC II</th><th>Aorto-iliac lesions (summary)</th><th>Usual first choice</th></tr>'
            '<tr><td>A</td><td>CIA stenoses; a short (≤3 cm) EIA stenosis</td><td>endovascular</td></tr>'
            '<tr><td>B</td><td>short infrarenal aortic stenosis; unilateral CIA occlusion; EIA stenoses 3–10 cm; unilateral EIA occlusion sparing the internal iliac and CFA</td><td>endovascular</td></tr>'
            '<tr><td>C</td><td>bilateral CIA occlusions; bilateral EIA stenoses 3–10 cm; EIA disease into the CFA; heavily calcified EIA occlusion</td><td>open in fit patients; endovascular increasingly</td></tr>'
            '<tr><td>D</td><td><b>infrarenal aorto-iliac occlusion</b>; diffuse disease of the aorta and both iliacs; unilateral CIA + EIA occlusion; bilateral EIA occlusions</td><td>open (aortobifemoral); CERAB in experienced hands</td></tr></table>')
    aiod_patho = vstep('ao-patho', 'Pathophysiology', 'Pathophysiology: aorto-iliac occlusive disease',
        '<p><b>Atherosclerosis at the aortic bifurcation</b>, where flow divides and shear stress is low: plaque, then stenosis, then thrombosis of the distal aorta and iliac arteries, often in smokers in their 40s–60s. <b>Collaterals</b> (lumbar to iliolumbar and circumflex iliac; internal thoracic to inferior epigastric; SMA to IMA through the arc of Riolan) keep the legs alive.</p>'
        + chain('Plaque at the bifurcation', 'Stenosis', 'Thrombosis: occluded distal aorta and iliacs', 'Collaterals', 'Claudication (buttock, thigh, calf)')
        + chain('Collaterals insufficient', '!Rest pain, ulcers, gangrene (chronic limb-threatening ischaemia)')
        + '<p><b>Leriche syndrome</b>: buttock and thigh claudication, absent femoral pulses, erectile dysfunction. <b>Ankle–brachial index</b>: 0.90 or less is peripheral arterial disease; 1.40 or more means non-compressible (calcified) arteries.</p>'
        + TASC
        + '<p><b>In Africa</b> think also of <b>HIV-associated vasculopathy</b> (young patients, acute thrombosis or occlusions, often presenting late with critical ischaemia) and <b>Takayasu arteritis</b> (young women, aorta and its branches; control inflammation before surgery).</p>'
        '<p><b>Treatment</b>: for everyone, best medical therapy: stop smoking (varenicline), antiplatelet, statin, blood pressure and diabetes control; supervised exercise for claudication. Revascularise <b>chronic limb-threatening ischaemia</b>, and claudication that still limits life after exercise and medical therapy, by shared decision. Treat the inflow (aorto-iliac) before the outflow.</p>'
        + ev('TASC II (Norgren et al., J Vasc Surg 2007). ESVS 2024 claudication guideline (Nordanstig et al.): ABI thresholds, smoking cessation, individualised revascularisation for claudication. Global Vascular Guidelines 2019 for CLTI (Conte et al.): endovascular-first for moderate-to-severe aorto-iliac disease; open reconstruction for extensive disease in average-risk patients. HIV vasculopathy: Robbs and Paruk (Eur J Vasc Endovasc Surg 2010; 226 patients, mean age 36) and Van Marle et al. (S Afr J Surg; over 90% presented with Fontaine III/IV, primary amputation 32%). Takayasu in Africa: Genga, Oyoo and Adebajo (Curr Rheumatol Rep 2018).'),
        front(Pv('bifurcation'), 420, (0.1, 1, 0.3)), show=[*VESS, 'aiod-occlusion', 'aiod-collaterals'], highlight=['aiod-occlusion'], labels=['aiod-occlusion', 'aiod-collaterals', 'ima', 'cfa-l'], opacity=AB_OP,
        quiz=ask('A 50-year-old smoker has buttock claudication, absent femoral pulses and erectile dysfunction. Where is the disease?', 'The distal aorta and both iliac arteries (Leriche syndrome)',
                 'Buttock claudication and absent femoral pulses place the obstruction above the groins; erectile dysfunction reflects poor internal iliac flow.', 'The superficial femoral arteries', 'The popliteal arteries', 'The tibial arteries'),
        after=True, spin=True)
    aiod_anat = vstep('ao-anatomy', 'Anatomy', 'The aortic bifurcation, iliacs and femoral arteries',
        '<p>The aorta divides at <b>L4</b> (the level of the iliac crests) into the <b>common iliac arteries</b>, each dividing over the sacroiliac joint into the <b>internal iliac</b> (pelvis, buttock) and the <b>external iliac</b>, which becomes the <b>common femoral artery</b> under the inguinal ligament at the mid-inguinal point and divides 3–5 cm lower into the <b>superficial femoral</b> and <b>profunda femoris</b>. The profunda is the key collateral to the leg when the SFA is occluded.</p>'
        '<p>Protect: the <b>ureters</b> (crossing the iliac bifurcations: graft tunnels pass behind them), the <b>iliac veins</b> behind the arteries, the <b>hypogastric plexus</b> (sexual function), the <b>IMA</b>, and in the groin the <b>femoral vein</b> medially and the femoral nerve laterally.</p>' + BELOW,
        front(Pv('bifurcation'), 440, (0.1, 1, 0.25)), show=VESS, highlight=hv(['cfa-l', 'cfa-r', 'pfa-l', 'pfa-r']), danger=hv(['civ-l', 'civ-r']), labels=hv(['cia-l', 'eia-l', 'iia-l', 'cfa-l', 'sfa-l', 'pfa-l', 'civ-r']), opacity=AB_OP, spin=True)
    GR = hv(['incision-groin-l', 'incision-groin-r'])
    GRX = hv(['fv-l', 'fv-r', 'inguinal-lig-l', 'inguinal-lig-r'])
    URE = hv(['ureter-l', 'ureter-r'])
    OBL = lambda t, dist=430: front(t, dist, (-0.45, 1, 0.3))                     # front-left oblique: the graft limbs stand off the front of the iliacs
    groins = lambda pre: vstep(f'{pre}-groins', 'Groins', 'Expose both femoral bifurcations',
        '<p>Vertical incisions over the femoral pulses (or where they should be: the mid-inguinal point). Expose the <b>common femoral artery</b> from the inguinal ligament down, and control the <b>SFA</b> and the <b>profunda</b> (its first branches too). Tie the lymphatics (lymph leaks and infection are the groin\'s complications). Feel the CFA for plaque: an endarterectomy or <b>profundaplasty</b> may be needed where the graft is sewn.</p>' + BELOW,
        front(Pv('cfa-r'), 300, (0.1, 1, 0.3)), show=['skin', *GR, *VESS, *GRX, 'aiod-occlusion'], highlight=hv(['cfa-l', 'cfa-r']), danger=hv(['fv-r', 'fv-l']), labels=hv(['cfa-r', 'sfa-r', 'pfa-r', 'fv-r', 'inguinal-lig-r', 'incision-groin-r']), opacity={**AB_OP, 'skin': 0.3},
        action={'kind': 'reveal', 'label': 'Open the groins', 'port': 'groin-r', 'ids': GR} if GR else None)
    c_abf = vstep('ab-case', 'Case', 'Case: Leriche syndrome with rest pain',
        '<p><b>TASC II D</b> (infrarenal aorto-iliac occlusion) with <b>chronic limb-threatening ischaemia</b> in a fit patient: <b>aortobifemoral bypass</b>, the most durable reconstruction. Endovascular reconstruction (CERAB) is an alternative in expert centres.</p>'
        '<p><b>Before surgery</b>: stop smoking, statin and antiplatelet, check the coronaries and the renal function; confirm the femoral run-off (profunda patent) on CT angiography. Test for HIV.</p>'
        + ev('de Vries and Hunink (J Vasc Surg 1997, meta-analysis): aortic bifurcation graft 5-year limb patency 91% for claudication and 87.5% for critical ischaemia; operative mortality 4.4%, systemic morbidity 12.1%. A 2021 meta-analysis of extensive aorto-iliac disease (Cardiovasc Intervent Radiol; 9,319 patients): open repair 5-year primary patency 88% vs 71% for standard endovascular treatment, with higher 30-day mortality (3% vs 0.8%).'),
        front(Pv('bifurcation'), 420, (0.1, 1, 0.3)), show=[*VESS, 'aiod-occlusion', 'aiod-collaterals'], highlight=['aiod-occlusion'], labels=['aiod-occlusion', 'cfa-l', 'cfa-r'], opacity=AB_OP,
        lead='<p>A <b>54-year-old man</b>, a matatu owner who smokes 20 a day. Two years of buttock and thigh claudication at 100 m, now <b>rest pain</b> in the right foot at night and erectile dysfunction. <b>No femoral pulses.</b> ABI 0.38 right, 0.52 left. CT angiography: the <b>aorta occluded from just below the IMA</b>, both common iliacs occluded, the external iliacs reconstituting; common femoral and profunda arteries patent; SFA patent. HIV-negative; creatinine normal; echo normal.</p>',
        quiz=ask('What is the most durable reconstruction for this fit patient with a TASC II D aorto-iliac occlusion?', 'Aortobifemoral bypass',
                 'Five-year limb patency is about 90% (de Vries and Hunink, 1997); endovascular results for TASC D are improving (CERAB) but standard stenting is less durable (about 71% at 5 years).', 'Bilateral iliac angioplasty without stents', 'Axillobifemoral bypass', 'Lumbar sympathectomy'))
    abf_lap = vstep('ab-aorta', 'Aorta', 'Laparotomy; the infrarenal aorta below the left renal vein',
        '<p>Midline laparotomy (or left retroperitoneal). Expose the infrarenal aorta <b>just below the left renal vein</b>, where it is usually soft enough to clamp: the occlusion often reaches the IMA, and above it the aorta is patent.</p>'
        '<p><b>End-to-end</b> anastomosis (divide the aorta, oversew the distal stump) is better haemodynamically and lies flatter behind the duodenum; <b>end-to-side</b> keeps flow to the IMA and internal iliacs when the external iliacs are occluded (pelvic and colonic perfusion).</p>',
        front(Pv('abf-prox'), 300, (0.15, 1, 0.4)), show=['skin', *LAP, *VESS, 'aiod-occlusion'], highlight=['aorta'], danger=hv(['renal-v-l', 'renal-a-l']), labels=hv(['renal-v-l', 'aiod-occlusion', 'ima']), opacity={**AB_OP, 'skin': 0.2},
        action={'kind': 'clamp', 'label': 'Clamp the infrarenal aorta', 'port': 'laparotomy', 'at': R(Pv('abf-prox')), 'axis': [0, 0.3, 1], 'radius': 12, 'jawLen': 55})
    abf_tunnel = vstep('ab-tunnel', 'Tunnels', 'Retroperitoneal tunnels to the groins, behind the ureters',
        '<p>From the groin and the aorta, a finger (then a tunnelling instrument) creates a tunnel on the front of the iliac arteries, <b>behind the ureter</b>, under the inguinal ligament, to meet in the groin. A graft limb placed in front of the ureter can compress it (hydronephrosis).</p>',
        OBL(Pv('ureter-l') if 'ureter-l' in LM else Pv('cia-l'), 300), show=[*VESS, *GRX, *URE, 'aiod-occlusion', 'graft-abf'], highlight=['graft-abf'], danger=hv(['ureter-l', 'ureter-r', 'civ-l', 'civ-r']), labels=hv(['graft-abf', 'ureter-l', 'cia-l', 'inguinal-lig-l']), opacity={**AB_OP, 'graft-abf': 0.6},
        quiz=ask('Why must the limbs of an aortobifemoral graft pass behind the ureters?', 'A graft in front of the ureter can compress it and cause hydronephrosis',
                 'Tunnel along the front of the iliac arteries, under (behind) the ureter; the ureter then lies in front of the graft.', 'To shorten the graft', 'To avoid the iliac veins', 'To prevent infection'))
    abf_graft = vstep('ab-graft', 'Graft', 'Proximal anastomosis, then the femoral anastomoses',
        '<p>Heparin; clamp. A <b>bifurcated Dacron graft</b> (commonly 16 × 8 mm or 14 × 7 mm, matched to the aorta), the body kept <b>short</b> so the bifurcation sits low and the limbs do not kink. Proximal anastomosis (3-0 polypropylene), pass the limbs through the tunnels, then each <b>femoral anastomosis</b> end-to-side onto the CFA (5-0 polypropylene), extended onto the profunda (profundaplasty) if the SFA is occluded or the profunda origin narrowed. Flush before completing each.</p>',
        OBL(Pv('bifurcation') - V([0, 0, 60]), 560), show=[*VESS, *GRX, *URE, 'aiod-occlusion', 'graft-abf'], highlight=['graft-abf'], danger=URE, labels=hv(['graft-abf', 'aiod-occlusion', 'ureter-l', 'pfa-l', 'cfa-r']), opacity=AB_OP,
        action={'kind': 'reveal', 'label': 'Sew in the graft', 'port': 'laparotomy', 'ids': ['graft-abf']})
    abf_close = vstep('ab-close', 'Close', 'Release one limb at a time; check the feet and the colon',
        '<p>Release each limb slowly (declamping hypotension). Feel for pulses in the grafts and at the feet (Doppler signals), look at the sigmoid colon. Close the retroperitoneum over the graft (away from the duodenum), then the groins in layers without dead space.</p>'
        '<p><b>Early complications</b>: bleeding, limb thrombosis, distal embolism (trash foot), colonic ischaemia, groin infection or lymph leak. <b>Late</b>: anastomotic false aneurysm at the groin, graft infection, limb occlusion from outflow disease.</p>',
        front(Pv('bifurcation') - V([0, 0, 60]), 560, (0.1, 1, 0.3)), show=[*VESS, *GRX, *URE, 'aiod-occlusion', 'graft-abf'], labels=hv(['graft-abf', 'aiod-occlusion', 'ureter-r']), opacity=AB_OP)
    abf_steps, abf_sq = build_v([('Patho', 'other', [aiod_patho]), ('Anatomy', 'other', [aiod_anat]), ('Case', 'other', [c_abf]), ('Groins', 'artery', [groins('ab')]),
                                 ('Aorta', 'artery', [abf_lap]), ('Tunnels', 'other', [abf_tunnel]), ('Graft', 'artery', [abf_graft]), ('Close', 'other', [abf_close])])
    finish('aiod-abf', 'aiod', 'Aorto-iliac occlusive disease', 'Aortobifemoral bypass', 'Both groins, the infrarenal aorta, retroperitoneal tunnels behind the ureters, bifurcated graft, femoral anastomoses.', abf_steps, abf_sq, AIOD_SRC)
    # --------------------------------------------------------------------- axillobifemoral
    AX = hv(['incision-axillary-r'])
    AXA = hv(['axillary-a-r', 'sca-r', 'bct'])
    RIBR = hv([f'rib-{n}-r' for n in range(1, 11)])
    AXV = lambda: tl((Pv('axillary-r') + Pv('cfa-r')) / 2 + V([10, 0, 0]), (0.55, 1, 0.12), 1050)
    AXSHOW = ['skin', *AX, *GR, *GRX, *AXA, *RIBR, 'aorta', *VESS, 'aiod-occlusion']
    AXOP = {**AB_OP, 'skin': 0.15, **{r: 0.45 for r in RIBR}}
    c_ax = vstep('ax-case', 'Case', 'Case: critical ischaemia with a hostile abdomen and poor lungs',
        '<p>An aortic operation is too risky (her lungs, the frozen abdomen): an <b>extra-anatomic</b> bypass from the <b>axillary artery</b> avoids the abdomen and the aortic clamp. Choose the side with the better arm pressure (and no subclavian stenosis). Less durable than an aortobifemoral graft.</p>'
        + ev('Martin and Katz (Am J Surg 2000): primary patency 86%, 72% and 63% at 1, 3 and 5 years; 30-day mortality 4.9%. Contemporary series report better 5-year patency with externally supported (ringed) grafts; patient selection makes comparison with aortobifemoral bypass unfair.'),
        front(Pv('bifurcation') + V([0, 0, 120]), 700, (0.3, 1, 0.3)), show=[*VESS, 'aiod-occlusion'], highlight=['aiod-occlusion'], labels=['aiod-occlusion'], opacity=AB_OP,
        lead='<p>A <b>71-year-old woman</b>, rest pain and a heel ulcer on the right, ABI 0.3 both sides; aorto-iliac occlusion on CT. <b>Severe COPD</b> on home oxygen; <b>three previous laparotomies</b> and a colostomy. Brachial pressures equal on both arms; no subclavian stenosis on CT.</p>',
        quiz=ask('Before an axillobifemoral bypass, what must be checked in the donor arm?', 'Equal arm pressures and no subclavian or axillary stenosis on the chosen side',
                 'A stenosed donor artery starves the graft and the arm; a pressure difference over about 20 mmHg suggests subclavian stenosis.', 'The ABI of the arm', 'The radial pulse only', 'Nothing: any arm will do'))
    ax_exp = vstep('ax-axilla', 'Axilla', 'Expose the first part of the axillary artery',
        '<p>A horizontal incision 2 cm below the middle third of the clavicle. Split the <b>pectoralis major</b> in the line of its fibres, divide the clavipectoral fascia (and the <b>pectoralis minor</b> if needed); the axillary vein lies in front and below, the <b>brachial plexus</b> cords above and behind. Control the artery medial to pectoralis minor.</p>'
        '<p>The anastomosis goes on the <b>first part</b> of the artery, the graft running along the artery for a few centimetres before turning down (so arm abduction does not tear it off).</p>',
        tl(Pv('axillary-r'), (0.35, 1, 0.45), 240), show=['skin', *AX, *AXA, *hv(['clavicle-r', 'rib-1-r', 'rib-2-r', 'rib-3-r'])], hide=['heart', 'aorta'], highlight=hv(['axillary-a-r']), labels=hv(['axillary-a-r', 'sca-r', 'clavicle-r', *AX]), opacity={'skin': 0.3},
        action={'kind': 'reveal', 'label': 'Open the axilla', 'port': 'axillary-r', 'ids': AX} if AX else None)
    ax_tunnel = vstep('ax-tunnel', 'Tunnel', 'Subcutaneous tunnel down the mid-axillary line; the cross-over',
        '<p>A long tunneller, through one or two counter-incisions, <b>subcutaneously in the mid-axillary line</b> (not over the costal margin, where it kinks), down to the right groin. A ringed (externally supported) PTFE graft, 8 mm. Then a <b>femorofemoral</b> limb in a suprapubic subcutaneous tunnel to the left groin.</p>',
        AXV(), show=[*AXSHOW, 'graft-axbf', 'anast-axbf'], hide=['heart'], highlight=['graft-axbf'], labels=hv(['graft-axbf', 'anast-axbf', 'axillary-a-r', 'rib-6-r', 'aiod-occlusion', 'inguinal-lig-r']), opacity=AXOP,
        action={'kind': 'reveal', 'label': 'Tunnel and sew the graft', 'port': 'axillary-r', 'ids': hv(['graft-axbf', 'anast-axbf'])})
    ax_steps, ax_sq = build_v([('Patho', 'other', [{**aiod_patho, 'id': 'ax-patho'}]), ('Anatomy', 'other', [{**aiod_anat, 'id': 'ax-anatomy'}]), ('Case', 'other', [c_ax]),
                               ('Groins', 'artery', [groins('ax')]), ('Axilla', 'artery', [ax_exp]), ('Graft', 'artery', [ax_tunnel]),
                               ('Close', 'other', [vstep('ax-close', 'Close', 'Release, check the flow, close',
                                    '<p>Complete the femoral anastomoses, flush, release. A graft pulse along the chest wall and at both groins; Doppler signals at the feet. The patient must not lie on the graft side or wear tight belts over it.</p>',
                                    AXV(), show=[*AXSHOW, 'graft-axbf', 'anast-axbf'], hide=['heart'], labels=hv(['graft-axbf', 'axillary-a-r', 'aiod-occlusion']), opacity=AXOP)])])
    finish('aiod-axbf', 'aiod', 'Aorto-iliac occlusive disease', 'Axillobifemoral bypass (extra-anatomic)', 'For the high-risk patient: axillary artery, subcutaneous tunnel, femoral anastomoses and a femorofemoral cross-over.', ax_steps, ax_sq, AIOD_SRC)
    # --------------------------------------------------------------------- endovascular: kissing stents
    c_en = vstep('ae2-case', 'Case', 'Case: bilateral common iliac occlusions and disabling claudication',
        '<p><b>TASC II C</b> (bilateral common iliac occlusions) with <b>lifestyle-limiting claudication</b> after 6 months of supervised walking and best medical therapy. <b>Endovascular first</b>: <b>kissing stents</b> at the bifurcation (covered balloon-expandable stents do better in occlusions and complex lesions), or a CERAB reconstruction. Open surgery stays available if it fails.</p>'
        + ev('COBEST (J Vasc Surg 2011 and 2016): covered stents gave better freedom from restenosis than bare stents, especially in TASC C/D lesions (5-year patency 74.7% vs 62.5%). CERAB (Taeymans et al., J Vasc Surg 2018; 89% TASC D): 3-year primary patency 82%. ESVS 2024: primary stenting for iliac occlusions; covered stents may be considered for TASC C/D.'),
        front(Pv('bifurcation'), 380, (0.1, 1, 0.3)), show=[*VESS, 'aiod-occlusion'], highlight=['aiod-occlusion'], labels=['aiod-occlusion', 'cia-l', 'cia-r'], opacity=AB_OP,
        lead='<p>A <b>61-year-old man</b>, ex-smoker (stopped a year ago), diabetic. <b>Buttock and calf claudication at 80 m</b> despite 6 months of supervised exercise, statin and aspirin; he is a farmer and cannot work. ABI 0.62 and 0.58. CT: <b>both common iliac arteries occluded</b> (5 cm each), the aorta above and the external iliacs below healthy; not heavily calcified.</p>',
        quiz=ask('Why treat this claudicant at all, and why endovascular first?', 'Symptoms still limit his life after exercise and best medical therapy; for iliac occlusions, stenting is effective and less invasive, with surgery in reserve',
                 'Revascularisation for claudication is individualised after conservative therapy fails; for common iliac occlusions (TASC B–C), primary stenting has good patency and low morbidity.', 'All claudicants need surgery', 'To prevent amputation, which is likely within a year', 'Because exercise is harmful'))
    en_access = vstep('ae2-access', 'Access', 'Bilateral femoral access; cross the occlusions',
        '<p>Ultrasound-guided retrograde puncture of both CFAs; sheaths; heparin. Cross each occlusion with a hydrophilic wire and catheter, preferably <b>intraluminally</b>; if subintimal, <b>re-enter</b> the true lumen in the aorta below the renal arteries (a re-entry device, or a brachial approach from above). Confirm the true lumen with contrast before dilating.</p>' + BELOW,
        front(Pv('bifurcation'), 360, (0.1, 1, 0.3)), show=[*VESS, 'aiod-occlusion', *GR], highlight=hv(['cfa-l', 'cfa-r']), labels=hv(['cfa-l', 'cfa-r', 'aiod-occlusion']), opacity=AB_OP)
    en_stent = vstep('ae2-stents', 'Stents', 'Kissing stents: deploy together, inflate together',
        '<p>Predilate. Position two <b>balloon-expandable (covered) stents</b> side by side, their tops level just above the bifurcation (the "new carina"), and <b>inflate them simultaneously</b> so neither crushes the other. Extend with further stents distally if needed, stopping short of the internal iliac origins where possible. Completion angiogram: no residual stenosis, no dissection, no rupture (have a covered stent and an occlusion balloon ready).</p>',
        front(Pv('bifurcation') - V([0, 0, 20]), 230, (0.1, 1, 0.3)), show=[*VESS, 'stents-kissing', 'kissing-balloons'], highlight=['stents-kissing'], danger=hv(['iia-l', 'iia-r']), labels=hv(['stents-kissing', 'kissing-balloons', 'cia-l', 'iia-r']), opacity=AB_OP,
        action={'kind': 'reveal', 'label': 'Inflate both balloons together', 'port': 'groin-r', 'ids': hv(['kissing-balloons', 'stents-kissing'])},
        quiz=ask('During iliac stenting the patient suddenly has back pain and hypotension. Most likely cause and first move?', 'Iliac rupture: inflate a balloon at the site to control bleeding, then place a covered stent',
                 'Rupture is rare but lethal; always have an occlusion balloon and covered stents on the shelf.', 'Vasovagal: give atropine', 'Contrast reaction: give steroids', 'Embolism: give heparin'))
    en_steps, en_sq = build_v([('Patho', 'other', [{**aiod_patho, 'id': 'ae2-patho'}]), ('Anatomy', 'other', [{**aiod_anat, 'id': 'ae2-anatomy'}]), ('Case', 'other', [c_en]),
                               ('Access', 'other', [en_access]), ('Stents', 'artery', [en_stent]),
                               ('After', 'other', [vstep('ae2-after', 'After', 'Closure, antiplatelet, surveillance',
                                    '<p>Closure devices or manual pressure. Dual antiplatelet therapy for a period, then single; statin; stay off tobacco. Walking resumes the next day; duplex surveillance of the stents. Recurrence is treated endovascularly again, or by aortobifemoral bypass.</p>',
                                    front(Pv('bifurcation') - V([0, 0, 20]), 260, (0.1, 1, 0.3)), show=[*VESS, 'stents-kissing'], labels=hv(['stents-kissing', 'cia-l', 'cia-r']), opacity=AB_OP)])])
    finish('aiod-endo', 'aiod', 'Aorto-iliac occlusive disease', 'Endovascular: kissing stents', 'Bilateral femoral access, crossing and re-entry, simultaneous covered stents at the bifurcation.', en_steps, en_sq, AIOD_SRC)

    # ================================================================================================ thoracic aortic aneurysm
    TD = Pv('taa-desc'); lat_l = lambda t, dist=420: tl(t, (-1, -0.2, 0.25), dist)
    T_SHOW = hv(['aorta', 'heart', 'lsca', 'lcca', 'bct', 'esophagus', 'adamkiewicz', *[f'vert-t{i}' for i in range(4, 13)]])
    taa_patho = vstep('ta-patho', 'Pathophysiology', 'Pathophysiology: thoracic aortic aneurysm',
        '<p><b>Medial degeneration</b> (loss of smooth muscle cells, fragmented elastic fibres, pooled proteoglycans) weakens the wall. Causes: <b>heritable</b> (Marfan syndrome, Loeys-Dietz, familial, bicuspid aortic valve), <b>degenerative/atherosclerotic</b> (the descending aorta in older smokers with hypertension), and <b>inflammatory or infective</b>: <b>syphilitic aortitis</b> (ascending and arch; obliterative endarteritis of the vasa vasorum) and <b>Takayasu arteritis</b> (young women), both seen in Africa.</p>'
        + chain('Medial degeneration', 'Dilatation', 'Laplace: tension rises with radius', '!Dissection or rupture')
        + '<p><b>Hinge points</b>: the risk of rupture or dissection jumps at about <b>6.0 cm in the ascending</b> and <b>7.0 cm in the descending</b> aorta, so repair is advised before them: <b>ascending 5.5 cm</b> (5.0 cm in Marfan, or with risk features, or in experienced teams), <b>descending 5.5 cm</b> for TEVAR with suitable anatomy (ESC 2024; ACC/AHA 2022 uses 6.0 cm for descending and thoracoabdominal), <b>thoracoabdominal 6.0 cm</b>, or growth of 0.5 cm in a year (0.3 cm a year for 2 years).</p>'
        '<p><b>Crawford extent</b> of thoracoabdominal aneurysm: I (left subclavian to above the renals), II (left subclavian to the bifurcation: the largest, highest spinal risk), III (distal thoracic to the bifurcation), IV (below the diaphragm), V (distal thoracic to above the renals, Safi).</p>'
        '<p><b>The spinal cord</b> depends on the anterior spinal artery, fed at the lower thoracic level mainly by the <b>artery of Adamkiewicz</b> (usually from a left intercostal artery at T9–T12). Covering or clamping that segment risks <b>paraplegia</b>: protect the cord with perfusion pressure (MAP), CSF drainage and, in open repair, distal perfusion and intercostal reattachment.</p>'
        + ev('natural history: Coady et al. (J Thorac Cardiovasc Surg 1997): hinge points 6.0 cm ascending, 7.0 cm descending; median size at rupture or dissection 6.0 and 7.2 cm. Thresholds: 2022 ACC/AHA guideline (Isselbacher et al.) and 2024 EACTS/STS guideline (Czerny et al.). Crawford/Safi extents: Frederick and Woo, Ann Cardiothorac Surg 2012. Syphilitic aortitis: ascending 50%, arch 35%, descending 15% (ICVTS 2012).'),
        lat_l(TD, 460), show=[*T_SHOW, 'taa-desc'], highlight=['taa-desc'], danger=hv(['adamkiewicz']), labels=hv(['taa-desc', 'lsca', 'adamkiewicz', 'esophagus']), opacity={'heart': 0.35, 'aorta': 0.6, 'taa-desc': 0.6, 'esophagus': 0.5},
        quiz=ask('At which diameter does the risk of rupture or dissection of the descending thoracic aorta rise sharply ("hinge point")?', 'About 7.0 cm',
                 'Coady et al. found hinge points at 6.0 cm (ascending) and 7.0 cm (descending); guidelines advise repair of the descending aorta at 5.5 cm, before that point.', '4.0 cm', '5.0 cm', '9.0 cm'),
        after=True, spin=True)
    taa_anat = vstep('ta-anatomy', 'Anatomy', 'The descending thoracic aorta and what lies around it',
        '<p>From the left subclavian artery (zone 3) down to the hiatus at T12. On its left: the <b>vagus</b> and, at the arch, the <b>left recurrent laryngeal nerve</b> hooking round the ligamentum arteriosum; medially the <b>oesophagus</b> (and the thoracic duct behind); the <b>intercostal arteries</b> leave its back in pairs; the <b>hemiazygos</b> veins cross behind. The <b>artery of Adamkiewicz</b> usually arises between T9 and T12 on the left.</p>'
        '<p>Aortic zones for TEVAR (Ishimaru): 0 ascending to the innominate; 1 to the left carotid; 2 to the left subclavian; 3 the proximal descending; 4 the rest of the descending.</p>',
        lat_l(TD, 400), show=[*T_SHOW, 'taa-desc'], highlight=['aorta'], danger=hv(['esophagus', 'adamkiewicz', 'lsca']), labels=hv(['lsca', 'lcca', 'esophagus', 'adamkiewicz', 'taa-desc']), opacity={'heart': 0.35, 'taa-desc': 0.35, 'esophagus': 0.6}, spin=True)
    c_to = vstep('to-case', 'Case', 'Case: a descending aneurysm in Marfan syndrome',
        '<p><b>Marfan syndrome</b> with a 5.8 cm descending aneurysm: <b>open repair</b>. TEVAR is not advised in heritable aortopathies (the stent graft\'s radial force on fragile tissue, progressive dilatation of the landing zones, retrograde dissection) except as a bridge or in emergencies.</p>'
        '<p><b>Plan</b>: left thoracotomy, <b>left heart bypass</b> (distal aortic perfusion), <b>CSF drainage</b>, reattachment of critical intercostals, mild hypothermia.</p>'
        + ev('2022 ACC/AHA: TEVAR is preferred for descending aneurysms with suitable anatomy in the absence of Marfan, Loeys-Dietz or vascular Ehlers-Danlos syndrome. Coselli et al. (J Vasc Surg 2002, randomised, extent I/II): CSF drainage reduced paraplegia or paraparesis from 13.0% to 2.6%.'),
        lat_l(TD, 460), show=[*T_SHOW, 'taa-desc'], highlight=['taa-desc'], labels=['taa-desc'], opacity={'heart': 0.35, 'taa-desc': 0.6, 'esophagus': 0.5},
        lead='<p>A <b>34-year-old woman</b> with <b>Marfan syndrome</b>, a valve-sparing root replacement 6 years ago. CT: the descending thoracic aorta has grown from 4.9 to <b>5.8 cm</b> in 18 months, from 4 cm beyond the left subclavian to T10. FEV1 85%; creatinine normal.</p>',
        quiz=ask('Why open repair rather than TEVAR in this patient?', 'Marfan syndrome: stent grafts in heritable aortopathy risk landing-zone dilatation, endoleak and retrograde dissection',
                 'Guidelines reserve TEVAR in Marfan and related syndromes for emergencies or as a bridge; open repair is durable.', 'The aneurysm is too small for TEVAR', 'TEVAR causes more paraplegia in young patients', 'Her lung function'))
    TH = hv(['incision-l'])
    to_thor = vstep('to-thor', 'Access', 'CSF drain, double-lumen tube, left thoracotomy',
        '<p>Before induction: a <b>lumbar CSF drain</b> (L3–L4), keeping the pressure at 10–15 mmHg or less during and after the repair. A double-lumen tube (left lung deflated), arterial lines in the right arm and a leg (for distal pressure), motor evoked potentials where available. Right lateral decubitus, hips rotated back (femoral access).</p>'
        '<p><b>Left posterolateral thoracotomy</b> through the <b>5th or 6th space</b> (a second, lower space or rib section for a long aneurysm).</p>',
        lat_l(TD, 480), show=['skin', *TH, *T_SHOW, 'taa-desc', 'csf-drain'], highlight=[*TH, 'csf-drain'], labels=hv(['csf-drain', *TH]), opacity={'skin': 0.3, 'taa-desc': 0.5},
        action={'kind': 'reveal', 'label': 'Place the drain; open the chest', 'port': 'thor-l', 'ids': hv(['csf-drain'])})
    to_lhb = vstep('to-lhb', 'Bypass', 'Left heart bypass; sequential clamping',
        '<p><b>Left heart bypass</b>: drain oxygenated blood from the <b>left inferior pulmonary vein</b> (or LA appendage) and return it by a centrifugal pump to the <b>distal aorta or a femoral artery</b>, with low-dose heparin. The kidneys, gut and the cord below the clamps are perfused while the aneurysm is repaired, and the heart is unloaded when the aorta is clamped.</p>'
        '<p>Clamp proximally (between the left carotid and subclavian, or distal to the subclavian if there is a neck), and distally in the mid-descending aorta. Keep the proximal MAP 80–100 mmHg and the distal pressure above about 60 mmHg.</p>'
        + ev('Coselli et al. (Ann Cardiothorac Surg 2023) protection bundle: CSF pressure below 15 mmHg during clamping, left heart bypass, reattachment of T7/8 to L1/2 intercostals, MAP 80–100 mmHg, spinal perfusion pressure at least 60 mmHg, haemoglobin at least 10 g/dL, mild hypothermia 32–34 °C.'),
        lat_l(TD, 380), show=[*T_SHOW, 'taa-desc'], highlight=['aorta'], danger=hv(['lsca', 'esophagus']), labels=hv(['lsca', 'taa-desc']), opacity={'heart': 0.35, 'taa-desc': 0.45},
        action={'kind': 'clamp', 'label': 'Clamp proximally', 'port': 'thor-l', 'at': R(Pv('taa-prox')), 'axis': [0, 0.2, 1], 'radius': 14, 'jawLen': 60})
    to_graft = vstep('to-graft', 'Graft', 'Open the aneurysm, reattach intercostals, sew in the graft',
        '<p>Open the aneurysm longitudinally, oversew back-bleeding upper intercostals, and <b>reattach the critical lower intercostals</b> (T8–T12, especially large, back-bleeding pairs) to the graft as an island or with a small side graft. <b>Proximal anastomosis</b> with 3-0 or 4-0 polypropylene (felt strip if fragile, as in Marfan), then move the clamp down; then the <b>distal anastomosis</b>. Flush, de-air, release slowly.</p>',
        lat_l(TD, 340), show=[*T_SHOW, 'graft-taa'], highlight=['graft-taa'], danger=hv(['adamkiewicz', 'esophagus']), labels=hv(['graft-taa', 'adamkiewicz']), opacity={'heart': 0.35},
        action={'kind': 'reveal', 'label': 'Sew in the graft', 'port': 'thor-l', 'ids': ['graft-taa']},
        quiz=ask('After an extensive descending repair the patient wakes with weak legs. First moves?', 'Raise the MAP (above about 90 mmHg), drain CSF to below 10 mmHg, correct anaemia and hypoxia',
                 'Delayed spinal cord ischaemia often recovers if spinal perfusion pressure (MAP minus CSF pressure) is restored quickly.', 'Wait and reassess in 24 hours', 'Give steroids only', 'Lower the blood pressure to protect the anastomoses'))
    to_steps, to_sq = build_v([('Patho', 'other', [taa_patho]), ('Anatomy', 'other', [taa_anat]), ('Case', 'other', [c_to]), ('Access', 'other', [to_thor]),
                               ('Bypass', 'artery', [to_lhb]), ('Graft', 'artery', [to_graft]),
                               ('After', 'other', [vstep('to-after', 'After', 'Spinal cord watch; the drain',
                                    '<p>Hourly leg checks for 48–72 hours; MAP targets, CSF drainage by protocol (watch for headache and for blood in the CSF: subdural haematoma), then clamp and remove the drain. Other complications: bleeding, renal failure, left recurrent laryngeal nerve palsy (hoarseness), chylothorax, pneumonia.</p>',
                                    lat_l(TD, 420), show=[*T_SHOW, 'graft-taa', 'csf-drain'], labels=hv(['graft-taa', 'csf-drain']), opacity={'heart': 0.35})])])
    finish('taa-open', 'taa', 'Thoracic aortic aneurysm', 'Descending, open (left heart bypass)', 'Left thoracotomy, CSF drainage, left heart bypass, intercostal reattachment, interposition graft.', to_steps, to_sq, TAA_SRC, side='left')
    # --------------------------------------------------------------------- TEVAR
    c_tv = vstep('tv2-case', 'Case', 'Case: a degenerative descending aneurysm in an older man',
        '<p><b>TEVAR</b>: a 6.4 cm degenerative aneurysm in a 71-year-old with COPD, with <b>landing zones of at least 20 mm</b> of healthy aorta at both ends and iliac access that takes the sheath. The proximal landing needs coverage of the <b>left subclavian artery</b> (zone 2): <b>revascularise it first</b> (carotid-subclavian bypass or transposition), which lowers stroke, arm ischaemia and spinal cord risk.</p>'
        + ev('VALOR (J Vasc Surg 2008): 30-day mortality 2.1% with TEVAR vs 7.9% with open repair (historical controls), paraplegia 1.5% and stroke 3.6% after TEVAR. Cheng et al. (JACC 2010, 5,888 patients): TEVAR lower 30-day mortality (OR 0.44) and paraplegia (OR 0.42), no long-term survival difference. SVS 2009 (Matsumura et al.): routine revascularisation before elective TEVAR that covers the left subclavian (weak recommendation); strongly recommended with a LIMA graft or dominant left vertebral. Spinal cord injury meta-analysis (Ann Cardiothorac Surg 2023; 61,962 patients): descending aneurysms 2.0% after TEVAR and after open repair.'),
        lat_l(TD, 460), show=[*T_SHOW, 'taa-desc'], highlight=['taa-desc'], labels=hv(['taa-desc', 'lsca']), opacity={'heart': 0.35, 'taa-desc': 0.6, 'esophagus': 0.5},
        lead='<p>A <b>71-year-old man</b>, smoker, COPD (FEV1 48%), hypertension. CT: a <b>6.4 cm</b> fusiform aneurysm of the mid-descending aorta, starting <b>12 mm</b> beyond the left subclavian origin, ending 4 cm above the coeliac trunk. External iliac arteries 8 mm. A dominant left vertebral artery.</p>',
        quiz=ask('The landing zone requires covering his left subclavian, and his left vertebral is dominant. What should be done?', 'Revascularise the left subclavian (carotid-subclavian bypass or transposition) before or at the TEVAR',
                 'Covering the subclavian with a dominant left vertebral risks posterior-circulation stroke, arm ischaemia and spinal cord ischaemia; revascularisation is strongly recommended in this setting.', 'Cover it without revascularisation', 'Abandon TEVAR', 'Embolise the vertebral artery'))
    tv_deploy = vstep('tv2-deploy', 'Deploy', 'Access, angiography, deploy, completion',
        '<p>A CSF drain if the coverage is long (over about 20 cm), the previous abdominal aorta has been repaired, or the subclavian and hypogastric supply is compromised. Femoral access (percutaneous with preclosure, or a cut-down; an iliac conduit if the vessels are small). A pigtail from the other groin or the left arm; an angiogram in the left anterior oblique projection to open the arch.</p>'
        '<p>Advance the stent graft over a stiff wire, place its covered edge at the planned zone, lower the blood pressure (systolic about 90–100 mmHg or rapid pacing) and <b>deploy</b>; balloon the seal zones only if needed. Completion angiogram: no type I endoleak, the carotids patent.</p>',
        lat_l(TD, 380), show=[*T_SHOW, 'taa-desc', 'tevar-graft'], highlight=['tevar-graft'], danger=hv(['lsca', 'lcca', 'adamkiewicz']), labels=hv(['tevar-graft', 'lsca']), opacity={'heart': 0.35, 'taa-desc': 0.3},
        action={'kind': 'reveal', 'label': 'Deploy the stent graft', 'port': 'thor-l', 'ids': ['tevar-graft']})
    tv_steps, tv_sq = build_v([('Patho', 'other', [{**taa_patho, 'id': 'tv2-patho'}]), ('Anatomy', 'other', [{**taa_anat, 'id': 'tv2-anatomy'}]), ('Case', 'other', [c_tv]),
                               ('Deploy', 'artery', [tv_deploy]),
                               ('After', 'other', [vstep('tv2-after', 'After', 'Neurology checks; surveillance',
                                    '<p>Leg and arm checks, MAP above 80–90 mmHg for the first days, drain management. <b>Surveillance</b> imaging at about 1 month, then yearly: endoleak, migration, sac growth, retrograde dissection.</p>',
                                    lat_l(TD, 420), show=[*T_SHOW, 'taa-desc', 'tevar-graft'], labels=['tevar-graft'], opacity={'heart': 0.35, 'taa-desc': 0.3})])])
    finish('taa-tevar', 'taa', 'Thoracic aortic aneurysm', 'Descending, TEVAR', 'Landing zones, left subclavian management, spinal cord protection, deployment, surveillance.', tv_steps, tv_sq, TAA_SRC, side='left')
    # --------------------------------------------------------------------- ascending aneurysm with hemiarch (syphilitic aortitis)
    if has('taa-asc'):
        TA = Pv('taa-asc'); fr = lambda t, dist=360: tl(t, (0.25, 1, 0.35), dist)
        A_SHOW = hv(['aorta', 'heart', 'bct', 'lcca', 'lsca', 'svc', 'pa-trunk', 'lbcv'])
        asc_patho = {**taa_patho, 'id': 'tas-patho', 'view': fr(TA, 380), 'show': [*A_SHOW, 'taa-asc'], 'highlight': ['taa-asc'], 'labels': hv(['taa-asc', 'bct', 'svc']), 'danger': [], 'opacity': {'heart': 0.4, 'taa-asc': 0.6}}
        asc_anat = vstep('tas-anatomy', 'Anatomy', 'The ascending aorta and the arch',
            '<p>The ascending aorta runs from the sinotubular junction to the <b>innominate artery</b>, inside the pericardium, with the <b>SVC</b> and right atrium on its right, the <b>pulmonary trunk</b> in front and to the left, the right pulmonary artery behind. The <b>left brachiocephalic vein</b> crosses in front of the arch branches. A <b>hemiarch</b> repair replaces the underside of the arch (the lesser curve) without reimplanting the head vessels.</p>',
            fr(TA, 340), show=[*A_SHOW, 'taa-asc'], highlight=['taa-asc'], danger=hv(['svc', 'lbcv', 'pa-trunk']), labels=hv(['bct', 'lcca', 'lsca', 'svc', 'lbcv', 'pa-trunk']), opacity={'heart': 0.4, 'taa-asc': 0.45}, spin=True)
        c_as = vstep('tas-case', 'Case', 'Case: syphilitic aortitis with an ascending aneurysm',
            '<p><b>A 5.6 cm ascending aneurysm</b> reaching the arch, symptomatic (chest pain), from <b>syphilitic aortitis</b>: <b>replace the ascending aorta and the hemiarch</b> under hypothermic circulatory arrest with antegrade cerebral perfusion. Treat the syphilis (intravenous penicillin), check the <b>coronary ostia</b> (ostial stenosis is typical) and the aortic valve (regurgitation from root dilatation).</p>'
            + ev('2022 ACC/AHA: ascending repair at 5.5 cm or more, or when symptomatic; hemiarch replacement when the aneurysm extends into the arch. Syphilitic aortitis involves the ascending aorta in about half and the arch in a third; untreated, symptomatic disease has a high mortality (ICVTS 2012). Hypothermia classification (Yan et al. 2013): moderate 20.1–28 °C, used with antegrade cerebral perfusion.'),
            fr(TA, 380), show=[*A_SHOW, 'taa-asc'], highlight=['taa-asc'], labels=['taa-asc'], opacity={'heart': 0.4, 'taa-asc': 0.6},
            lead='<p>A <b>52-year-old man</b>, 3 months of central chest pain and a hoarse voice. CT: a <b>5.6 cm</b> ascending aneurysm with wall calcification, extending to the proximal arch; aortic root 3.9 cm; mild aortic regurgitation. <b>TPHA positive, VDRL 1:32</b>. Coronary angiography: a 70% ostial stenosis of the left main.</p>',
            quiz=ask('Which coronary lesion is characteristic of syphilitic aortitis?', 'Ostial stenosis of the coronary arteries',
                     'Aortitis thickens the intima at the root, narrowing the ostia; it needs to be addressed at surgery (endarterectomy, patch or bypass).', 'Mid-LAD plaque', 'Coronary aneurysms', 'Coronary spasm'))
        as_cpb = vstep('tas-cpb', 'Bypass', 'Sternotomy; axillary cannulation; cool',
            '<p>Median sternotomy. Arterial cannulation of the <b>right axillary artery</b> (through an 8 mm graft sewn end-to-side), which later provides <b>antegrade cerebral perfusion</b>; venous drainage from the right atrium; an LV vent. Cool on bypass to <b>moderate hypothermia</b> (about 24–28 °C). Cross-clamp below the innominate; cardioplegia (antegrade, or into the ostia if there is aortic regurgitation).</p>',
            fr(TA, 360), show=['sternum', *A_SHOW, 'taa-asc'], highlight=['taa-asc'], danger=hv(['bct', 'lbcv']), labels=hv(['bct', 'svc', 'taa-asc']), opacity={'heart': 0.4, 'taa-asc': 0.45, 'sternum': 0.4},
            action={'kind': 'clamp', 'label': 'Cross-clamp', 'port': 'sternotomy', 'at': R(TA + V([0, 0, 25])), 'axis': [0, 0.3, 1], 'radius': 16, 'jawLen': 60})
        as_graft = vstep('tas-graft', 'Graft', 'Proximal anastomosis; circulatory arrest and the hemiarch',
            '<p>Excise the aneurysm; <b>proximal anastomosis</b> to the sinotubular junction (4-0 polypropylene, felt if needed). At target temperature, stop the pump, clamp the innominate, and perfuse the brain <b>antegrade</b> through the axillary graft (about 10 mL/kg/min, right radial pressure 40–60 mmHg). Remove the clamp, bevel the <b>open distal anastomosis</b> along the underside of the arch. Restart perfusion through the graft, de-air, clamp the graft, rewarm; complete the proximal work.</p>',
            fr(TA, 320), show=[*A_SHOW, 'graft-asc'], highlight=['graft-asc'], danger=hv(['bct', 'lcca', 'lsca']), labels=hv(['graft-asc', 'bct']), opacity={'heart': 0.4},
            action={'kind': 'reveal', 'label': 'Sew in the graft', 'port': 'sternotomy', 'ids': ['graft-asc']})
        as_steps, as_sq = build_v([('Patho', 'other', [asc_patho]), ('Anatomy', 'other', [asc_anat]), ('Case', 'other', [c_as]), ('Bypass', 'artery', [as_cpb]), ('Graft', 'artery', [as_graft]),
                                   ('After', 'other', [vstep('tas-after', 'After', 'Rewarm, wean, and treat the syphilis',
                                        '<p>Rewarm slowly (no more than 10 °C gradient, not above 37 °C), wean, protamine, haemostasis (coagulopathy after circulatory arrest: platelets, fibrinogen). Neurological assessment on waking. <b>Penicillin</b> for tertiary syphilis (14 days intravenously for cardiovascular syphilis, per local protocol), and follow-up imaging of the remaining arch and descending aorta.</p>',
                                        fr(TA, 360), show=[*A_SHOW, 'graft-asc'], labels=['graft-asc'], opacity={'heart': 0.4})])])
        finish('taa-asc', 'taa', 'Thoracic aortic aneurysm', 'Ascending and hemiarch (circulatory arrest)', 'Sternotomy, axillary cannulation, moderate hypothermia, antegrade cerebral perfusion, open distal hemiarch anastomosis.', as_steps, as_sq, TAA_SRC)
    # the case vignette of the infrarenal case (its lead is on c_inf); patho and case steps shared by several approaches get unique ids
    for key in ('aaa-infra', 'aaa-juxta', 'aaa-supra', 'aaa-evar', 'aiod-abf', 'aiod-axbf', 'aiod-endo', 'taa-open', 'taa-tevar', 'taa-asc'):
        if key in procs:
            seen = set()
            for s in procs[key]['steps']:
                if s['id'] in seen: s['id'] = s['id'] + '-2'
                seen.add(s['id'])
# ==================================================================================================== the operative field in open-heart steps
# after the chest is open: the drapes and their sternotomy window, the split sternum held open, the pericardial cradle
FIELD = [i for i in ('drape-sternotomy', 'pericardium-open') if has(i)]
if FIELD:
    for key, v in procs.items():
        if v.get('group') != 'Cardiac' or key in ('mvr-mics', 'tv-mics', 'avr-ramt', 'cabg-1v'): continue
        for s_ in v['steps']:
            if s_.get('seq', 0) < 3 or s_['phase'] in ('Decision', 'Anatomy', 'Pathophysiology', 'Case') or s_['id'].endswith(('-sternotomy', '-access', '-setup', '-laa')): continue
            s_['hide'] = [i for i in s_.get('hide', []) if i not in ('sternum', *FIELD)]
            inside = s_['phase'] in ('Valve', 'Root', 'Septum', 'Pulmonary root', 'Tricuspid')   # looking inside the heart: the pericardium would show through
            s_['show'] = [*s_.get('show', []), 'sternum', *[i for i in FIELD if not (inside and i == 'pericardium-open')]]
            if inside: s_['hide'] = [*s_['hide'], 'pericardium-open']
            else: s_['opacity'] = {**s_.get('opacity', {}), 'pericardium-open': 0.55}
# consent and ICU steps in every operation, and the CTICU protocol hub
import modules_new
modules_new.add(procs, ask, has, LM, S)
import modules_cong
modules_cong.add(procs, ask, has, LM, S)
import postop
postop.apply(procs, ask, has)
import modules_leak   # after postop: the leak module carries its own consent and ICU steps
modules_leak.add(procs, ask, has, LM, S)
import modules_vasc2  # PAD best medical therapy, ALI Rutherford IIa, AAA management, common iliac aneurysm
modules_vasc2.add(procs, ask, has, LM, S)
# operations appear in the menu in this order
ORDER = ['position', 'thoracotomy-l', 'thoracotomy-r', 'vats-ports-l', 'vats-ports-r', 'lul', 'lll', 'rul', 'rml', 'rll', 'pnl', 'pnr', 'bronchiectasis', 'asp', 'seg-lingula', 'seg-lul-updiv', 'seg-s6', 'trachea', 'thymectomy', 'oesophagectomy', 'duct', 'empyema', 'ppe', 'cle', 'cpam', 'rt', 'clamshell', 'cardio', 'tract', 'hilar', 'mvr', 'avr', 'root', 'tricuspid', 'cabg', 'pericardium', 'asd', 'vsd', 'pda', 'coa', 'tof', 'palliation', 'pad', 'aiod', 'aaa', 'iliac', 'taa', 'ali', 'infrainguinal', 'bka', 'aka', 'avf', 'cticu']
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
# American spelling in what the reader sees (esophagus, esophagectomy); identifiers and the titles of cited papers are left as they are
import re as _re
from us_spelling import fix as _us_fix
def _us(x, key=None):
    if isinstance(x, str): return x if key in ('id', 'op', 'url') else _us_fix(_re.sub(r'Oesophag', 'Esophag', _re.sub(r'oesophag', 'esophag', x)))[0]
    if isinstance(x, list): return [_us(i, key) for i in x]
    if isinstance(x, dict): return {k: (v if k == 'sources' else _us(v, k)) for k, v in x.items()}
    return x
procs = {k: _us(v) for k, v in procs.items()}
# the 3D skeleton of every operation; the words are edited in content/procedures/*.md and merged by scripts/content-build.mjs
_BASE = Path(__file__).resolve().parent.parent / 'content' / '_base'; _BASE.mkdir(parents=True, exist_ok=True)
(_BASE / 'procedures.base.json').write_text(json.dumps(procs, indent=1))
print({k: len(v['steps']) for k, v in procs.items()})
