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
# operations appear in the menu in this order
ORDER = ['position', 'thoracotomy-l', 'thoracotomy-r', 'vats-ports-l', 'vats-ports-r', 'lul', 'lll', 'rul', 'rml', 'rll', 'pnl', 'pnr', 'seg-lingula', 'seg-lul-updiv', 'seg-s6', 'rt', 'clamshell', 'cardio', 'tract', 'hilar']
procs = dict(sorted(procs.items(), key=lambda kv: (ORDER.index(kv[1]['op']), list(procs).index(kv[0]))))
for v in procs.values():
    v['group'] = v.get('group') or ('Pneumonectomy' if v['op'].startswith('pn') else 'Segmentectomy' if v['op'].startswith('seg-') else 'Lobectomy')
# every VATS or open setup step: the patient on the side, the surface lines drawn
for v in procs.values():
    if v['group'] in ('Trauma', 'Access and positioning'): continue
    k = v['side'][0]
    for s in v['steps']:
        if s['phase'] == 'Setup':
            s['pose'] = 'lateral'
            s['show'] = [*s.get('show', []), *[i for i in (f'line-aal-{k}', f'line-mal-{k}', f'line-pal-{k}', f'lm-scaptip-{k}') if has(i)]]
            s['opacity'] = {**s.get('opacity', {}), 'skin': 1.0}
(OUT / 'procedures.json').write_text(json.dumps(procs, indent=1))
print({k: len(v['steps']) for k, v in procs.items()})
