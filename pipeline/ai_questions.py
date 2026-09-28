"""Write public/data/ai_questions.json: the questions put to NV-Reason-CT about the reference CT, per operation.

Each question has
  id      stable key (answers are stored under it by ai_read.py)
  op      the operation it belongs to ('general' = the scan as a whole, shown in Explore)
  kind    report | preop (what the scan shows before this operation) | anatomy (structural anatomy) | approach
  q       the prompt sent to the model, phrased as a radiology question it can answer from the volume
  focus   structure or landmark id: the 3D view and CT jump there
  key     the atlas answer: counts and relations measured from this scan's segmentation, or what to look for.
          It is what the trainee and the model are both compared against.

The model reads the scan; it has not seen the operation. Its answers are research output, unverified.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'public' / 'data'
atlas = json.loads((OUT / 'atlas.json').read_text())
S = {s['id']: s for s in atlas['structures']}
has = lambda i: i in S
nm = lambda i: S[i]['name'] if has(i) else i

L_UP = [i for i in ('pa-truncus-anterior', 'pa-posterior-1', 'pa-posterior-2', 'pa-posterior-3', 'pa-lingular') if has(i)]
R_UP = [i for i in ('rpa-truncus', 'rpa-a3-1', 'rpa-a2-1') if has(i)]

Q = []


def q(op, kind, id_, text, focus, key):
    Q.append({'id': f'{op}:{id_}', 'op': op, 'kind': kind, 'q': text, 'focus': focus if (has(focus) or focus in atlas['landmarks']) else 'carina', 'key': key})


# ---------------------------------------------------------------- the scan as a whole
q('general', 'report', 'report', 'write a structured chest CT report', 'carina',
  'Read it yourself first: lungs (nodules, emphysema, consolidation), pleura, mediastinum and nodes, heart and pericardium, great vessels, bones, upper abdomen.')
q('general', 'preop', 'nodes', 'Is there mediastinal or hilar lymphadenopathy? If so, which nodal stations and what size?', 'ln-7',
  'Short axis over 10 mm is enlarged. Check stations 2R/4R, 2L/4L, 5, 6, 7 (subcarinal), 10 and 11 on both sides.')
q('general', 'preop', 'nodules', 'Are there any pulmonary nodules or masses? Give the lobe, size and features.', 'lung-centre',
  'Scroll every lobe on lung windows; a nodule is under 3 cm, a mass over 3 cm.')
q('general', 'preop', 'parenchyma', 'Is there emphysema, fibrosis or other chronic lung disease?', 'lung-centre',
  'Emphysema and fibrosis lower the reserve for resection: they push toward a smaller resection and formal lung function testing.')
q('general', 'preop', 'pleura', 'Is there a pleural effusion, pleural thickening, calcification or pneumothorax?', 'lung-centre',
  'Pleural thickening or calcification warns of adhesions and a harder VATS.')

# ---------------------------------------------------------------- lobectomies and pneumonectomies
LOBE = {'lul': ('left upper lobe', 'lul', 'left'), 'lll': ('left lower lobe', 'lll', 'left'), 'rul': ('right upper lobe', 'rul', 'right'),
        'rml': ('right middle lobe', 'rml', 'right'), 'rll': ('right lower lobe', 'rll', 'right')}
for op, (lobe, focus, side) in LOBE.items():
    q(op, 'preop', 'lobe', f'Describe the {lobe}: is there any mass, nodule, consolidation, collapse or bronchiectasis?', focus,
      f'Find the lesion and its relation to the fissure and the hilum: a central lesion or one crossing the fissure may need more than a lobectomy.')
    q(op, 'preop', 'hilar-nodes', f'Are the {side} hilar and interlobar lymph nodes enlarged or calcified?', 'ln-10l' if side == 'left' else 'ln-10r',
      'Calcified or matted hilar nodes (old TB is common in Kenya) stick to the artery and make VATS dissection dangerous: consider an open approach or proximal control first.')
    q(op, 'approach', 'fissure', f'Is the {"left oblique" if side == "left" else "right oblique and horizontal"} fissure complete on this CT?', 'fissure' if side == 'left' else 'fissure-r',
      'Complete fissure: artery-first in the fissure is easy (posterior / fissure-first approach). Incomplete fissure: hilum-first (anterior / fissureless), fissure stapled last.')
    q(op, 'approach', 'vats', f'Is there anything on this CT that would make a VATS {lobe} lobectomy difficult, such as a central tumour, calcified nodes, pleural thickening or chest wall invasion?', focus,
      'Central tumour, calcified hilar nodes, chest wall invasion or a frozen pleura favour thoracotomy or a planned conversion.')
q('lul', 'anatomy', 'arteries', 'How many arterial branches arise from the left pulmonary artery to the left upper lobe?', 'pa-left',
  f'On this scan the atlas finds {len(L_UP)}: ' + ', '.join(nm(i) for i in L_UP) + '. The usual range is 3 to 7; the truncus anterior comes first.')
q('lul', 'anatomy', 'veins', 'Are the left superior and inferior pulmonary veins separate, or is there a common trunk?', 'pv-superior',
  'Separate on this scan (two left atrial ostia). A common trunk is a recognised variant: never staple a "superior vein" until the inferior vein is seen.')
q('lll', 'anatomy', 'a6', 'Describe the superior segmental artery of the left lower lobe and its origin relative to the lingular artery.', 'pa-a6',
  'A6 leaves the back of the interlobar artery, usually at or above the level of the lingular artery on the front: take A6 and the basal trunk separately if the lingular artery is close.')
q('rul', 'anatomy', 'arteries', 'Describe the right pulmonary artery branches to the right upper lobe, including the truncus anterior and any posterior ascending artery.', 'rpa',
  f'On this scan: ' + ', '.join(nm(i) for i in R_UP) + '. The truncus anterior is first and short; the posterior ascending artery (A2) arises in the fissure.')
q('rul', 'anatomy', 'azygos', 'Describe the azygos vein and its relation to the right main bronchus.', 'azygos',
  'The azygos arches over the right main bronchus into the SVC: it is the roof of the dissection for the RUL bronchus and station 4R.')
q('rml', 'anatomy', 'veins', 'Does the middle lobe vein drain into the right superior pulmonary vein?', 'rpv-ml',
  'Usually yes, as its lowest tributary: preserve the upper lobe veins above it.')
q('rll', 'anatomy', 'a6', 'Where does the superior segmental artery of the right lower lobe arise relative to the middle lobe artery?', 'rpa-a6',
  'A6 often arises opposite or above the middle lobe artery: a single staple across the "basal trunk plus A6" can take the middle lobe artery.')
for op, side in (('pnl', 'left'), ('pnr', 'right')):
    q(op, 'preop', 'central', f'Is there a central {side} lung tumour involving the main bronchus, main pulmonary artery or pulmonary veins?', 'pa-left' if side == 'left' else 'rpa',
      'Pneumonectomy is for central disease a lobectomy or sleeve cannot clear. Check the length of main bronchus and artery free of tumour.')
    q(op, 'preop', 'other-lung', f'Is the {"right" if side == "left" else "left"} lung normal? Any nodules, emphysema or fibrosis?', 'lung-centre',
      'After pneumonectomy the other lung is all there is: any disease there may rule the operation out.')
    q(op, 'anatomy', 'mainpa', f'Describe the {side} main pulmonary artery: its length before the first branch and its diameter.', 'pa-left' if side == 'left' else 'rpa',
      'The right main PA is longer (runs behind the SVC); the left is short: the truncus anterior comes off early, so clamp-space on the left is tight.')
    q(op, 'approach', 'pericardium', f'Is there tumour near the pericardium or left atrium on the {side} side that would need an intrapericardial approach?', 'heart',
      'Tumour at the vein ostia or on the proximal PA: open the pericardium and control the vessels inside it.')

# ---------------------------------------------------------------- segmentectomies
q('seg-lingula', 'preop', 'lesion', 'Is there a nodule in the lingula? What is its size and distance from the intersegmental plane?', 'seg-lingula',
  'Segmentectomy suits a peripheral nodule of 2 cm or less with a margin at least equal to its diameter.')
q('seg-lul-updiv', 'preop', 'lesion', 'Is there a nodule in the apicoposterior or anterior segments of the left upper lobe? Size and location?', 'seg-lul-upper',
  'Upper division segmentectomy: lesion in S1+2 or S3, small and peripheral, margin clear of the lingula.')
q('seg-s6', 'preop', 'lesion', 'Is there a nodule in the superior segment of the left lower lobe? Size and location?', 'seg-s6',
  'S6 segmentectomy: the superior segment has its own artery, vein and bronchus and a relatively flat plane.')
for op in ('seg-lingula', 'seg-lul-updiv', 'seg-s6'):
    q(op, 'anatomy', 'bronchus', 'Describe the segmental bronchi of the left lung and the division of the left upper lobe bronchus.', 'br-lul',
      'The left upper lobe bronchus divides into an upper division (B1+2, B3) and the lingular bronchus (B4, B5); B6 leaves the back of the lower lobe bronchus.')

# ---------------------------------------------------------------- trauma
q('rt', 'preop', 'tamponade', 'Is there a pericardial effusion or haemopericardium?', 'heart',
  'Fluid in the pericardium in a shocked patient with a precordial wound is tamponade until proved otherwise: thoracotomy or sternotomy, not a needle.')
q('rt', 'preop', 'haemothorax', 'Is there a haemothorax or pneumothorax? Which side and roughly how large?', 'lung-centre',
  'Operate for about 1500 mL on drain insertion, or 200 mL an hour for several hours, or instability.')
q('rt', 'anatomy', 'aorta', 'Describe the descending thoracic aorta and the oesophagus just above the diaphragm and their relationship.', 'aorta-clamp',
  'Low in the chest the oesophagus lies anterior and to the right of the aorta: separate them with the fingers before clamping.')
q('clamshell', 'anatomy', 'mammary', 'Identify the internal mammary (internal thoracic) arteries and their distance from the sternal edge.', 'ima-l',
  'About 1 to 2 cm from the sternal edge behind the costal cartilages: divided in a clamshell, ligate both ends.')
q('clamshell', 'preop', 'mediastinum', 'Is there a mediastinal haematoma or any sign of injury to the aorta or great vessels?', 'aorta',
  'Mediastinal blood around the arch suggests great vessel injury: a clamshell plus upper sternal extension reaches the arch branches.')
q('cardio', 'preop', 'pericardium', 'Is there pericardial fluid, and which cardiac chamber lies directly behind the sternum?', 'heart',
  'The right ventricle is the most anterior chamber: the one most often hit by a precordial stab.')
q('cardio', 'anatomy', 'coronary', 'Describe the course of the left anterior descending and right coronary arteries on this CT.', 'heart',
  'LAD in the anterior interventricular groove, RCA in the right atrioventricular groove: pass mattress sutures beneath them, never tie them off.')
q('tract', 'preop', 'contusion', 'Is there pulmonary contusion, laceration or a pneumatocoele in the left lung?', 'lll',
  'A tract away from the hilum: tractotomy. Near the hilum: control the hilum and consider anatomical resection.')
q('hilar', 'anatomy', 'hilum', 'Describe the arrangement of the left pulmonary artery, pulmonary veins and main bronchus at the hilum.', 'pa-left',
  'Artery highest, veins anterior and lowest, bronchus behind: a clamp across the whole hilum takes all three; the inferior ligament must be free first.')
q('hilar', 'preop', 'air', 'Is there air in the cardiac chambers, aorta or pulmonary veins?', 'heart',
  'Air in the left heart or coronaries after lung injury is systemic air embolism: clamp or twist the hilum, head down, aspirate the left ventricle and aorta.')

# ---------------------------------------------------------------- batch 4
q('thymectomy', 'preop', 'mass', 'Is there an anterior mediastinal mass? Describe its size, margins, and any invasion of the pericardium, great vessels or lung.', 'thymus',
  'Thymoma: a well-defined anterior mediastinal mass; loss of fat planes with the great vessels or pericardium suggests invasion (Masaoka-Koga III).')
q('thymectomy', 'anatomy', 'innominate', 'Describe the left brachiocephalic (innominate) vein and its relation to the thymus.', 'lbcv',
  'It crosses behind the upper thymus from left to right to join the SVC; the thymic veins drain into its back.')
q('thymectomy', 'approach', 'side', 'Is the anterior mediastinal tissue predominantly to the right or the left of the midline?', 'thymus',
  'Right VATS gives the best view of the SVC and right phrenic; left-sided disease or the aortopulmonary window favours left VATS, subxiphoid or sternotomy.')
q('oesophagectomy', 'preop', 'tumour', 'Is there oesophageal wall thickening or a mass? Give its level, length and relation to the carina, aorta and left main bronchus.', 'esophagus',
  'Level decides the operation: upper third (above the carina) favours McKeown with a neck anastomosis; lower third and GE junction suit Ivor Lewis or transhiatal.')
q('oesophagectomy', 'preop', 'nodes', 'Are there enlarged paraoesophageal, subcarinal, coeliac or left gastric lymph nodes?', 'esophagus',
  'Coeliac or distant nodes change staging; subcarinal and paraoesophageal nodes come out with a transthoracic en bloc resection.')
q('oesophagectomy', 'anatomy', 'stomach', 'Describe the stomach and whether it is suitable as a conduit (any mass, previous surgery or hiatus hernia).', 'stomach',
  'A healthy stomach with an intact right gastroepiploic arcade makes the conduit; if not, colon or jejunum.')
q('duct', 'preop', 'chyle', 'Is there a pleural effusion, and on which side?', 'thoracic-duct',
  'A duct injury below T5 gives a right chylothorax, above it a left one; the ligation is still done low on the right.')
q('duct', 'anatomy', 'retrocrural', 'Describe the space between the descending aorta and the azygos vein just above the diaphragm.', 'thoracic-duct',
  'The thoracic duct runs here on the front of the vertebral bodies: the target of mass ligation.')
q('empyema', 'preop', 'collection', 'Is there a pleural collection? Is it loculated, is there pleural thickening or enhancement (split pleura sign), and is the underlying lung collapsed?', 'empyema-l',
  'Split pleura sign and loculation mean empyema; thick pleura with a trapped lung means stage III and decortication.')
q('empyema', 'preop', 'lung', 'Is there consolidation, abscess or a bronchopleural fistula in the adjacent lung?', 'lll',
  'A lung abscess or fistula changes the operation (resection, muscle flap) and the prognosis.')
(OUT / 'ai_questions.json').write_text(json.dumps({'model': 'nvidia/NV-Reason-CT', 'questions': Q}, indent=1))
print(len(Q), 'questions;', {k: sum(1 for x in Q if x['op'] == k) for k in dict.fromkeys(x['op'] for x in Q)})
