// The TCVS logbook catalogue: operations and their component skills, from the KNH/UoN MMed TCVS logbook (version 1.0,
// curriculum 2, 2022), with the same skill counted wherever it recurs (a thoracotomy is a thoracotomy in a lobectomy or a
// decortication), so progress can be followed skill by skill. Core procedures (*) must be done independently by the end.

export type Level = 'O' | 'A2' | 'A' | 'S-TS' | 'S-TU' | 'P' | 'T';
export const LEVELS: { code: Level; name: string; def: string; knh: 'P' | '1A' | '2A' | '' }[] = [
  { code: 'O', name: 'Observed', def: 'Present, unscrubbed, for at least 70% of it', knh: '' },
  { code: 'A2', name: '2nd assistant', def: 'Supplementary assistant', knh: '2A' },
  { code: 'A', name: '1st assistant', def: 'The trainer did the key parts; you assisted (may have opened or closed)', knh: '1A' },
  { code: 'S-TS', name: 'Supervised, trainer scrubbed', def: 'You did the key parts with the trainer scrubbed', knh: 'P' },
  { code: 'S-TU', name: 'Supervised, trainer unscrubbed', def: 'You did the key parts; trainer in theatre at least 70% of the time', knh: 'P' },
  { code: 'P', name: 'Performed', def: 'You did the key parts; trainer unscrubbed and in theatre under 70%', knh: 'P' },
  { code: 'T', name: 'Trained a junior', def: 'You trained a more junior trainee through the key parts', knh: 'P' },
];
export const RANK: Record<string, number> = { O: 0, A2: 1, A: 2, 'S-TS': 3, 'S-TU': 4, P: 5, T: 6 };

/** skills that recur across operations share an id */
export const SKILLS: Record<string, string> = {
  sternotomy: 'Median sternotomy*', redo: 'Re-do sternotomy', thoracotomy: 'Thoracotomy', vats: 'VATS port placement and assessment', laparotomy: 'Laparotomy', neck: 'Neck dissection',
  aocann: 'Aortic cannulation for bypass*', vcann: 'Venous cannulation for bypass*', cscann: 'Coronary sinus cannula', femcann: 'Femoral cannulation (for bypass)', axcann: 'Axillary cannulation (for bypass)',
  wean: 'Wean off cardiopulmonary bypass', decann: 'De-cannulate', sternclose: 'Sternal closure*', chestclose: 'Other chest closure*', abdclose: 'Closure of abdomen*',
  svg: 'Saphenous vein harvest', radial: 'Radial artery harvest', ima: 'Internal mammary artery harvest', distal: 'Distal coronary anastomosis', proximal: 'Aorto-coronary (proximal) anastomosis', tgraft: 'Other proximal anastomosis (e.g. T-graft)',
  aortotomy: 'Aortotomy and leaflet excision', avsut: 'Aortic annular sutures', avseat: 'Parachute aortic valve down and tie', avclose: 'Close aortotomy',
  atriotomy: 'Atriotomy and leaflet excision', mvsut: 'Mitral annular sutures', mvseat: 'Parachute mitral valve down and tie', mvclose: 'Close atriotomy',
  aoprox: 'Aortic graft: proximal anastomosis', aodist: 'Aortic graft: distal anastomosis', repair: 'Key repair (the defining step)',
  duct: 'Ductus: dissection and ligation or division',
  pa: 'Ligation and division of pulmonary artery branch', pv: 'Ligation and division of pulmonary vein tributary', bronchus: 'Division and repair of bronchus', fissure: 'Fissure completion',
  nodes: 'Mediastinal lymph node dissection', specimen: 'Specimen delivery',
  peelwall: 'Peel pleura off chest wall', peellung: 'Peel pleura off lung',
  esotumor: 'Dissection and mobilization of tumor in chest', conduit: 'Prepare stomach conduit', pullup: 'Mobilize stomach into chest', egpart: 'Esophagogastric anastomosis (part)',
  egfull: 'Esophagogastric anastomosis (full)', pyloro: 'Gastric drainage procedure',
  proxctl: 'Proximal control', distctl: 'Distal control', paspart: 'Proximal anastomosis (part)', pasfull: 'Proximal anastomosis (full)', dapart: 'Distal anastomosis (part)', dafull: 'Distal anastomosis (full)',
  incision: 'Skin incision', vascctl: 'Vascular control', bone: 'Bone transection', skinclose: 'Skin closure',
  arteriotomy: 'Arteriotomy', embol: 'Balloon embolectomy (Fogarty)', artclose: 'Arteriotomy closure / patch',
};

export interface Op { id: string; name: string; core?: boolean; skills: string[] }
export interface Section { id: string; name: string; ops: Op[] }
const cardiac = (key: string[]) => ['sternotomy', 'aocann', 'vcann', ...key, 'wean', 'decann', 'sternclose'];
const cong = (approach = 'sternotomy') => approach === 'thoracotomy' ? ['thoracotomy', 'repair', 'chestclose'] : ['sternotomy', 'aocann', 'vcann', 'repair', 'wean', 'decann', 'sternclose'];
const simple = (name: string, id: string, skills: string[] = []): Op => ({ id, name, skills });

export const CATALOG: Section[] = [
  { id: 'adult', name: 'Adult / acquired cardiac', ops: [
    { id: 'avr', name: 'Aortic valve replacement', core: true, skills: cardiac(['aortotomy', 'avsut', 'avseat', 'avclose']) },
    { id: 'mvr', name: 'Mitral valve replacement', core: true, skills: cardiac(['atriotomy', 'mvsut', 'mvseat', 'mvclose']) },
    { id: 'mvrep', name: 'Mitral valve repair', skills: cardiac(['atriotomy', 'repair', 'mvclose']) },
    { id: 'cabg', name: 'Coronary bypass, on pump', skills: ['sternotomy', 'ima', 'svg', 'radial', 'aocann', 'vcann', 'distal', 'proximal', 'tgraft', 'wean', 'decann', 'sternclose'] },
    { id: 'opcab', name: 'Coronary bypass, off pump', skills: ['sternotomy', 'ima', 'svg', 'distal', 'proximal', 'sternclose'] },
    { id: 'tv', name: 'Tricuspid valve procedure', skills: cardiac(['repair']) },
    { id: 'dvr', name: 'Double valve procedure', skills: cardiac(['aortotomy', 'avsut', 'avseat', 'avclose', 'atriotomy', 'mvsut', 'mvseat', 'mvclose']) },
    { id: 'tvr3', name: 'Triple valve procedure', skills: cardiac(['repair']) },
    { id: 'asc', name: 'Ascending aorta replacement', skills: cardiac(['aoprox', 'aodist']) },
    { id: 'root', name: 'Aortic root replacement', skills: cardiac(['aoprox', 'aodist']) },
    { id: 'arch', name: 'Aortic arch replacement', skills: cardiac(['aoprox', 'aodist']) },
    { id: 'desc', name: 'Descending aorta replacement', skills: ['thoracotomy', 'femcann', 'aoprox', 'aodist', 'chestclose'] },
    { id: 'taaa', name: 'Thoracoabdominal aorta replacement', skills: ['thoracotomy', 'laparotomy', 'aoprox', 'aodist', 'chestclose', 'abdclose'] },
    { id: 'ivsd', name: 'Post-infarct VSD / rupture', skills: cardiac(['repair']) },
    { id: 'myxoma', name: 'Cardiac tumor: myxoma', skills: cardiac(['repair']) },
    { id: 'tumor', name: 'Cardiac tumor: other', skills: cardiac(['repair']) },
    { id: 'htx', name: 'Cardiac transplantation', skills: cardiac(['repair']) },
    { id: 'vad', name: 'VAD insertion', skills: ['sternotomy', 'repair', 'sternclose'] },
    { id: 'pemb', name: 'Pulmonary embolectomy', skills: cardiac(['repair']) },
    { id: 'af', name: 'Surgery for atrial fibrillation', skills: cardiac(['repair']) },
    { id: 'peri', name: 'Pericardiectomy / pericardial window', skills: ['sternotomy', 'repair', 'sternclose'] },
    { id: 'cardother', name: 'Other major cardiac procedure', skills: cardiac(['repair']) },
    { id: 'iabp', name: 'IABP insertion', skills: [] },
    { id: 'femcannop', name: 'Femoral cannulation (for bypass)', skills: ['femcann'] },
    { id: 'axcannop', name: 'Axillary cannulation (for bypass)', skills: ['axcann'] },
    { id: 'bleed', name: 'Re-operation for bleeding (intrathoracic)', skills: ['sternotomy', 'repair', 'sternclose'] },
    { id: 'redo', name: 'Re-do sternotomy', skills: ['redo'] },
  ] },
  { id: 'cong', name: 'Pediatric and congenital cardiac', ops: [
    { id: 'pda', name: 'Patent ductus arteriosus (open ligation)', core: true, skills: ['thoracotomy', 'duct', 'chestclose'] },
    ...([['asdp', 'ASD repair: primum'], ['asds', 'ASD repair: secundum'], ['papvd', 'Partial anomalous pulmonary venous drainage'], ['vsd', 'VSD repair'],
      ['tof', 'Tetralogy of Fallot repair: without transannular patch'], ['toftap', 'Tetralogy of Fallot repair: transannular patch'], ['tofc', 'Tetralogy of Fallot repair: RV-PA conduit'],
      ['bdg', 'Bidirectional cavopulmonary connection'], ['fontan', 'Fontan completion'], ['aso', 'Arterial switch operation'], ['truncus', 'Truncus arteriosus repair'],
      ['rvpa', 'RVOT conduit replacement'], ['avsdp', 'AVSD repair: partial'], ['avsdc', 'AVSD repair: complete'], ['apw', 'Aortopulmonary window repair'], ['dorv', 'DORV repair'],
      ['tapvd', 'TAPVD repair'], ['dswitch', 'Double switch operation'], ['norwood', 'Norwood stage I'], ['dks', 'Damus-Kaye-Stansel procedure'], ['konno', 'Konno procedure'],
      ['ross', 'Ross procedure'], ['pavr', 'Aortic valve repair (pediatric)'], ['pavrr', 'Aortic valve replacement (pediatric)'], ['proot', 'Aortic root replacement (pediatric)'],
      ['pmvrep', 'Mitral valve repair (pediatric)'], ['pmvr', 'Mitral valve replacement (pediatric)'], ['ebstein', "Ebstein's anomaly repair"], ['pvr', 'Pulmonary valve replacement'],
      ['alcapa', 'Anomalous coronary artery repair'], ['lvoto', 'LVOT obstruction relief'], ['iaa', 'Interrupted aortic arch repair'], ['unifoc', 'Unifocalization'],
      ['pvad', 'VAD insertion (pediatric)'], ['ptx', 'Transplant (pediatric)']] as [string, string][]).map(([id, name]) => ({ id, name, skills: cong() })),
    ...([['mbts', 'Blalock-Taussig shunt: modified'], ['cbts', 'Blalock-Taussig shunt: classic'], ['coa', 'Coarctation of the aorta repair'], ['ring', 'Vascular ring division'],
      ['sling', 'Pulmonary artery sling repair']] as [string, string][]).map(([id, name]) => ({ id, name, skills: cong('thoracotomy') })),
    { id: 'ecmo', name: 'ECMO cannulation', skills: [] },
  ] },
  { id: 'thor', name: 'General thoracic', ops: [
    { id: 'eso', name: 'Esophageal resection', core: true, skills: ['laparotomy', 'thoracotomy', 'neck', 'esotumor', 'conduit', 'pullup', 'egpart', 'egfull', 'pyloro', 'chestclose', 'abdclose'] },
    { id: 'pneum', name: 'Pneumonectomy', core: true, skills: ['thoracotomy', 'pa', 'pv', 'bronchus', 'nodes', 'chestclose'] },
    { id: 'lob', name: 'Lobectomy (open)', core: true, skills: ['thoracotomy', 'pa', 'pv', 'bronchus', 'fissure', 'nodes', 'specimen', 'chestclose'] },
    { id: 'lobv', name: 'Lobectomy (VATS)', skills: ['vats', 'pa', 'pv', 'bronchus', 'fissure', 'nodes', 'specimen', 'chestclose'] },
    { id: 'seg', name: 'Segmentectomy', skills: ['thoracotomy', 'pa', 'pv', 'bronchus', 'fissure', 'chestclose'] },
    { id: 'decort', name: 'Decortication', core: true, skills: ['thoracotomy', 'peelwall', 'peellung', 'chestclose'] },
    { id: 'medtum', name: 'Mediastinal tumor resection', skills: ['sternotomy', 'repair', 'sternclose'] },
    { id: 'esoinj', name: 'Repair of esophageal injury', skills: ['thoracotomy', 'repair', 'chestclose'] },
    { id: 'trinj', name: 'Repair of tracheal / bronchial injury', skills: ['thoracotomy', 'repair', 'chestclose'] },
    { id: 'tracres', name: 'Tracheal resection and reconstruction', skills: ['neck', 'repair'] },
    { id: 'lvrs', name: 'Lung volume reduction / bullectomy', skills: ['vats', 'repair', 'chestclose'] },
    { id: 'ltx', name: 'Lung transplantation', skills: ['thoracotomy', 'repair', 'chestclose'] },
    { id: 'thorother', name: 'Other major thoracic (thoracotomy)', skills: ['thoracotomy', 'repair', 'chestclose'] },
    { id: 'esoother', name: 'Other esophageal procedure', skills: ['repair'] },
    simple('VATS diagnostic', 'vatsdx', ['vats']), simple('VATS pleurodesis', 'vatspl', ['vats']), simple('VATS wedge resection', 'vatsw', ['vats']), simple('Other VATS', 'vatso', ['vats']),
    simple('Mediastinoscopy', 'medsc'), simple('Bronchoscopy: rigid', 'brrig'), simple('Bronchoscopy: flexible', 'brflex'), simple('Bronchoscopy: therapeutic (stent, laser)', 'brther'),
    simple('Esophagoscopy: rigid', 'esrig'), simple('Esophagoscopy: flexible', 'esflex'), simple('Esophagoscopy: therapeutic (stent, dilatation)', 'esther'),
    simple('Chest drain insertion', 'drain'), simple('Other minor thoracic (not thoracotomy)', 'thorminor'),
  ] },
  { id: 'vasc', name: 'Peripheral vascular', ops: [
    { id: 'aaa', name: 'Abdominal aortic aneurysm repair (open)', core: true, skills: ['laparotomy', 'proxctl', 'distctl', 'paspart', 'pasfull', 'dapart', 'dafull', 'abdclose'] },
    { id: 'aka', name: 'Above-knee amputation', core: true, skills: ['incision', 'vascctl', 'bone', 'skinclose'] },
    { id: 'bka', name: 'Below-knee amputation', core: true, skills: ['incision', 'vascctl', 'bone', 'skinclose'] },
    { id: 'cea', name: 'Carotid endarterectomy', skills: ['incision', 'proxctl', 'distctl', 'arteriotomy', 'artclose', 'skinclose'] },
    { id: 'cbt', name: 'Carotid body tumor excision', skills: ['incision', 'proxctl', 'repair', 'skinclose'] },
    simple('EVAR (endovascular aneurysm repair)', 'evar'),
    { id: 'bypa', name: 'Lower limb bypass: anatomical', skills: ['incision', 'proxctl', 'distctl', 'paspart', 'pasfull', 'dapart', 'dafull', 'skinclose'] },
    { id: 'bypx', name: 'Lower limb bypass: extra-anatomical', skills: ['incision', 'proxctl', 'distctl', 'paspart', 'pasfull', 'dapart', 'dafull', 'skinclose'] },
    { id: 'anothr', name: 'Other aneurysm surgery', skills: ['proxctl', 'distctl', 'repair'] },
    { id: 'artinj', name: 'Repair of injured artery', skills: ['incision', 'proxctl', 'distctl', 'repair', 'skinclose'] },
    { id: 'veininj', name: 'Repair of injured vein', skills: ['incision', 'repair', 'skinclose'] },
    { id: 'emb', name: 'Embolectomy', skills: ['incision', 'arteriotomy', 'embol', 'artclose', 'skinclose'] },
    { id: 'avf', name: 'Arteriovenous fistula creation', skills: ['incision', 'repair', 'skinclose'] },
    simple('Varicose veins: stripping', 'vvs', ['incision', 'skinclose']), simple('Varicose veins: perforator ligation', 'vvp', ['incision', 'skinclose']),
    simple('Other peripheral vascular procedure', 'vascother'),
  ] },
  { id: 'gen', name: 'General / pediatric surgery / ORL (rotations)', ops: [
    simple('Laparotomy', 'glap', ['laparotomy', 'abdclose']), simple('Bowel anastomosis', 'bowel'), simple('Gastrostomy tube insertion', 'gtube'), simple('Gastrectomy', 'gastrect'),
    simple('Jejunostomy tube insertion', 'jtube'), simple('Colostomy', 'colost'), simple('Colostomy closure', 'colclose'), simple('Incision and drainage of abscess', 'id'),
    simple('Esophagogastroduodenoscopy', 'ogd'), simple('Repair of tracheoesophageal fistula (pediatric)', 'tef', ['thoracotomy', 'repair', 'chestclose']),
    simple('Rigid bronchoscopy: diagnostic', 'rbdx'), simple('Rigid bronchoscopy: foreign body', 'rbfb'), simple('Flexible bronchoscopy: diagnostic', 'fbdx'),
    simple('Flexible bronchoscopy: foreign body', 'fbfb'), simple('Tracheostomy', 'trach'), simple('Neck dissection (other)', 'gneck', ['neck']),
  ] },
];

export const OPS: Record<string, Op & { section: string }> = Object.fromEntries(CATALOG.flatMap((s) => s.ops.map((o) => [o.id, { ...o, section: s.id }])));
/** the atlas procedure that best matches each catalogue operation, to preselect it */
export const ATLAS_TO_OP: [RegExp, string][] = [
  [/^avr-/, 'avr'], [/^mvr-repair/, 'mvrep'], [/^mvr-/, 'mvr'], [/^cabg-offpump/, 'opcab'], [/^cabg/, 'cabg'], [/^tv-/, 'tv'], [/^root-/, 'root'], [/^taa-asc/, 'asc'], [/^taa-open/, 'desc'],
  [/^peri/, 'peri'], [/^pda/, 'pda'], [/^asd/, 'asds'], [/^vsd/, 'vsd'], [/^tof/, 'toftap'], [/^pal-bt/, 'mbts'], [/^coa/, 'coa'],
  [/^(b4-eso|eso-)/, 'eso'], [/^pn[lr]-/, 'pneum'], [/-(uni|bi|anterior|posterior|fissure|hilum)$/, 'lobv'], [/^(lul|lll|rul|rml|rll)-open/, 'lob'], [/^seg-/, 'seg'],
  [/^(b4-emp|ppe)/, 'decort'], [/^b4-thym/, 'medtum'], [/^trachea/, 'tracres'], [/^bulla/, 'lvrs'], [/^bx-|^asp-|^cle-|^cpam-/, 'lob'],
  [/^aaa-(infra|juxta|supra)/, 'aaa'], [/^aaa-evar/, 'evar'], [/^aka/, 'aka'], [/^bka/, 'bka'], [/^ali-/, 'emb'], [/^aiod-abf|^fp-/, 'bypa'], [/^aiod-axbf/, 'bypx'], [/^avf/, 'avf'], [/^ciaa/, 'anothr'],
];
export const opForAtlas = (k: string) => ATLAS_TO_OP.find(([re]) => re.test(k))?.[1] ?? '';
