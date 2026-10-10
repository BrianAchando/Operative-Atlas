// On-call quick cards: the first minutes of the emergencies a cardiothoracic and vascular resident is called to.
// Each card: how to recognize it, what to do now, the numbers that decide, when to call, what not to do, the atlas
// modules to open and the sources. Bundled with the app, so the cards work offline once COVA has loaded.
// Evidence rule: every threshold is from the cited source; KNH protocol items are labeled as local.

type H = (tag: string, attrs?: Record<string, unknown>, ...kids: (Node | string | null | undefined)[]) => HTMLElement;
interface Card { id: string; title: string; tag: 'Trauma' | 'Cardiac' | 'Aorta' | 'Vascular' | 'Thoracic' | 'Airway'; line: string;
  spot: string[]; now: string[]; nums?: [string, string][]; call: string[]; avoid?: string[]; links: string[]; src: string[] }

export const CARDS: Card[] = [
  { id: 'pen-chest', tag: 'Trauma', title: 'Penetrating chest injury, losing or lost output', line: 'Resuscitative thoracotomy: who, and the first four moves',
    spot: ['Stab or gunshot of the chest or "box" (between the nipples, clavicles to costal margin) with shock or arrest',
      'Signs of life: pupils reacting, any spontaneous movement or breathing, organized ECG activity, palpable pulse or measurable pressure',
      'Tamponade: hypotension, distended neck veins, muffled sounds; pericardial fluid on FAST'],
    now: ['Massive transfusion protocol; tranexamic acid 1 g IV within 3 hours of injury, then 1 g over 8 hours',
      'Losing pulse or arrested with signs of life: left anterolateral thoracotomy in the emergency room, extend to a clamshell if needed',
      'Open the pericardium anterior to the phrenic nerve, release tamponade, finger on the cardiac wound, then suture',
      'Cross-clamp the descending aorta if there is no output; internal cardiac massage; control the lung hilum if bleeding from the lung',
      'Still has output with tamponade: straight to theatre (sternotomy or thoracotomy)'],
    nums: [['Penetrating chest, signs of life', 'Thoracotomy: strong recommendation'], ['Penetrating chest, no signs of life', 'Conditional recommendation'],
      ['Blunt, no signs of life', 'Recommended against'], ['Tranexamic acid', 'Within 3 h of injury']],
    call: ['Call the consultant and theatre at the same time as you open the chest'],
    avoid: ['Do not delay the thoracotomy for imaging in a patient losing output', 'Pericardiocentesis does not treat traumatic tamponade; it only buys minutes'],
    links: ['rt', 'clamshell', 'cardio', 'hilar-clamp', 'hilar-twist'],
    src: ['Seamon MJ et al. EAST practice management guideline: emergency department thoracotomy. J Trauma Acute Care Surg 2015;79:159–73.',
      'CRASH-2 collaborators. Lancet 2010;376:23–32 (tranexamic acid within 3 h).', 'ATLS 10th edition, American College of Surgeons, 2018.'] },

  { id: 'hemothorax', tag: 'Trauma', title: 'Massive hemothorax', line: 'Drain, measure, and when it means theatre',
    spot: ['Shock with a dull, silent hemithorax; white-out on the chest X-ray; fluid on ultrasound'],
    now: ['Two large IV lines, cross-match, massive transfusion protocol; tranexamic acid within 3 hours',
      'Large-bore chest drain (28–32 Fr), 5th or 6th intercostal space, mid-axillary line',
      'Measure the first drainage and then every hour; collect for autotransfusion if a cell-saver bag is available',
      'Repeat the chest X-ray after the drain'],
    nums: [['Thoracotomy', 'Initial drainage over 1500 mL'], ['or', 'Over 200 mL/h for 2–4 hours'], ['or', 'Repeated transfusion needed, or instability']],
    call: ['Any thoracotomy threshold met', 'Retained clot after the drain: VATS within days'],
    avoid: ['Do not clamp the drain to "tamponade" the bleeding'],
    links: ['tract', 'hilar-clamp', 'rt'],
    src: ['Forrester JD. Hemothorax. MSD Manual Professional (reviewed May 2026).', 'ATLS 10th edition, 2018.', 'CRASH-2, Lancet 2010.'] },

  { id: 'tension', tag: 'Trauma', title: 'Tension pneumothorax', line: 'Clinical diagnosis: decompress before the X-ray',
    spot: ['Hypotension or arrest with absent breath sounds on one side, hyper-resonance, distended neck veins, rising airway pressures on the ventilator',
      'Tracheal deviation is late'],
    now: ['High-flow oxygen', 'Needle decompression in adults: 4th or 5th intercostal space, just anterior to the mid-axillary line',
      'Or finger thoracostomy at the same site (ventilated patient)', 'Then a chest drain'],
    call: ['No improvement after decompression: think of a misplaced needle, the other side, or tamponade'],
    links: ['rt'],
    src: ['ATLS 10th edition, American College of Surgeons, 2018 (adult needle site moved to the 4th/5th space, anterior to the mid-axillary line).'] },

  { id: 'arrest-cs', tag: 'Cardiac', title: 'Arrest after cardiac surgery (CALS)', line: 'Shocks or pacing first; resternotomy within 5 minutes',
    spot: ['Arrest in the ICU after cardiac surgery: check the arterial line, ECG and the patient'],
    now: ['VF or pulseless VT: three stacked shocks before compressions (compressions may wait up to 1 minute for the defibrillator)',
      'Asystole or severe bradycardia: connect the epicardial wires, DDD 80–100/min at maximal outputs',
      'Airway: 100% oxygen, remove PEEP, bag by hand, check the tube and look for a pneumothorax',
      'Not resolved after shocks or pacing: compressions and prepare for emergency resternotomy',
      'Amiodarone 300 mg IV after three failed shocks'],
    nums: [['Resternotomy', 'Within 5 minutes, up to postoperative day 10'], ['Compressions', '100–120/min, aim for arterial systolic over 60 mmHg'],
      ['Systolic under 60 with good compressions', 'Tamponade or hypovolemia: reopen']],
    call: ['Call the consultant and the resternotomy team at the start'],
    avoid: ['No routine 1 mg epinephrine: the rebound hypertension tears suture lines (only in small doses, by a senior)', 'No atropine for asystole', 'Do not use an AED when a manual defibrillator is there'],
    links: ['cticu-cardiac', 'cticu-core'],
    src: ['Dunning J et al. The Society of Thoracic Surgeons expert consensus for the resuscitation of patients who arrest after cardiac surgery. Ann Thorac Surg 2017;103:1005–20.'] },

  { id: 'bleed-cs', tag: 'Cardiac', title: 'Bleeding or tamponade after cardiac surgery', line: 'Drains, CVP and echo: when to reopen',
    spot: ['Brisk drain output, or drains that suddenly stop with a rising CVP and falling pressure',
      'Tamponade: hypotension, tachycardia, rising and equalizing filling pressures, low urine output; echo may show a clot (often localized)'],
    now: ['Warm, correct calcium, check ACT/heparin, platelets, fibrinogen; protamine if heparin is residual',
      'Milk the drains; urgent bedside echo', 'Keep the blood pressure controlled (no hypertension)', 'Prepare for re-exploration'],
    nums: [['Call the consultant (KNH protocol)', 'Over 400 mL in the first hour'], ['or', 'Over 200 mL/h for several hours'], ['or', 'A sudden gush, or drains stop with a rising CVP']],
    call: ['Any threshold above; tamponade physiology; arrest (see CALS card)'],
    links: ['cticu-cardiac', 'cticu-core'],
    src: ['KNH CTICU protocol, cardiac core (local protocol, in COVA).', 'Dunning J et al. Ann Thorac Surg 2017;103:1005–20 (resternotomy in arrest).'] },

  { id: 'dissection', tag: 'Aorta', title: 'Acute aortic dissection', line: 'Impulse control, CT, and who goes to theatre',
    spot: ['Tearing chest or back pain, pulse or BP difference between arms, new aortic regurgitation, stroke or limb ischemia with chest pain',
      'CT angiography of the whole aorta to the groins'],
    now: ['Two large IV lines, arterial line, analgesia (opioids)',
      'IV beta-blocker first; add a vasodilator only once the heart rate is controlled',
      'Look for malperfusion: coronary (ECG), brain, gut (lactate, pain), kidneys, legs',
      'Type A: emergency surgery', 'Type B: medical; complicated (rupture, malperfusion, extension, enlarging, pain or hypertension that will not settle): endovascular repair'],
    nums: [['Systolic pressure', 'Under 120 mmHg, or the lowest that keeps organs perfused'], ['Heart rate', '60–80/min'],
      ['Complicated type B', 'TEVAR first-line (ESC 2024: class I, level A)']],
    call: ['Every type A: the consultant now', 'Type B with any complication'],
    avoid: ['No vasodilator before the beta-blocker (reflex tachycardia raises the shear)', 'No thrombolysis for "MI" before the aorta is excluded'],
    links: ['taa-asc', 'root-bentall', 'root-david', 'taa-tevar'],
    src: ['Isselbacher EM et al. 2022 ACC/AHA guideline for the diagnosis and management of aortic disease. Circulation 2022;146:e334–e482.',
      'Mazzolai L et al. 2024 ESC guidelines for the management of peripheral arterial and aortic diseases. Eur Heart J 2024;45:3538–700.'] },

  { id: 'raaa', tag: 'Vascular', title: 'Ruptured abdominal aortic aneurysm', line: 'Permissive hypotension, CT if stable, EVAR first',
    spot: ['Abdominal or back pain with collapse; known or palpable aneurysm; older man; flank bruising'],
    now: ['Permissive hypotension: just enough blood to keep the patient conscious; avoid large volumes of crystalloid',
      'Massive transfusion protocol; tranexamic acid', 'Stable enough: CT angiography of the whole aorta and access vessels at once',
      'EVAR first if the anatomy is suitable (local anesthesia where possible); otherwise open repair',
      'After repair: watch for abdominal compartment syndrome (bladder pressure)'],
    nums: [['Permissive hypotension', 'ESVS 2024 Rec 72 (I-C)'], ['Prompt CTA', 'Rec 70'], ['EVAR first, suitable anatomy', 'Rec 80 (I-A)']],
    call: ['Consultant, theatre and the endovascular team at once'],
    avoid: ['Do not chase a normal blood pressure', 'Do not transfer an unstable patient for imaging they do not need'],
    links: ['aaa-mgmt', 'aaa-evar', 'aaa-infra'],
    src: ['Wanhainen A et al. ESVS 2024 clinical practice guidelines on the management of abdominal aorto-iliac artery aneurysms. Eur J Vasc Endovasc Surg 2024;67:192–331.', 'IMPROVE trial investigators, BMJ 2014.'] },

  { id: 'ali', tag: 'Vascular', title: 'Acute limb ischemia', line: 'Heparin, Rutherford class, and the clock',
    spot: ['Sudden pain, pallor, pulselessness, paresthesia, paralysis, cold limb; atrial fibrillation or a known aneurysm',
      'Compare with the other leg: normal pulses there suggest an embolus'],
    now: ['Heparin 5000 IU (or 70–100 IU/kg) IV bolus, then infusion; analgesia', 'Classify (below) and examine sensation and movement',
      'CT angiography if it will not delay treatment', 'IIa: embolectomy or catheter-directed thrombolysis; IIb: immediate revascularization',
      'After revascularization: check the compartments; fasciotomy for compartment syndrome'],
    nums: [['I, viable', 'No sensory loss or weakness; Doppler signals present'], ['IIa, marginally threatened', 'Minimal sensory loss (toes); no weakness'],
      ['IIb, immediately threatened', 'Sensory loss beyond the toes, rest pain, mild to moderate weakness'], ['III, irreversible', 'Profound anesthesia, paralysis, rigor: amputation']],
    call: ['IIb: the consultant now; theatre', 'III: discuss primary amputation'],
    avoid: ['No IV (systemic) thrombolysis (ESVS: class III)', 'Do not decide on creatine kinase or myoglobin alone'],
    links: ['ali-iia', 'ali-emb'],
    src: ['Björck M et al. ESVS 2020 clinical practice guidelines on the management of acute limb ischaemia. Eur J Vasc Endovasc Surg 2020;59:173–218.', 'Rutherford RB et al. J Vasc Surg 1997;26:517–38.'] },

  { id: 'hemoptysis', tag: 'Thoracic', title: 'Massive hemoptysis', line: 'It kills by drowning: protect the good lung',
    spot: ['Large-volume hemoptysis, or any amount with hypoxia or airway compromise; old TB cavity, aspergilloma, bronchiectasis'],
    now: ['Lie the patient bleeding side down (if known)', 'Oxygen; large-bore IV; cross-match; correct coagulopathy',
      'Airway: intubate with a large tube (allows bronchoscopy); selective intubation of the good side or a bronchial blocker if needed',
      'CT angiography if stable to find the side and the vessel', 'Bronchial artery embolization as the first definitive step; surgery when it fails or for the cause (aspergilloma, destroyed lobe)'],
    call: ['Any massive hemoptysis: the consultant, anesthesia and interventional radiology together'],
    avoid: ['Do not lie the patient bleeding side up', 'Do not send an unstable patient to CT without a secured airway'],
    links: ['asp-open', 'bx-lul'],
    src: ['Davidson K, Shojaee S. Managing massive hemoptysis. Chest 2020;157:77–88.', 'Denning DW et al. ESCMID/ERS guideline on chronic pulmonary aspergillosis. Eur Respir J 2016;47:45–68.'] },

  { id: 'eso-perf', tag: 'Thoracic', title: 'Esophageal perforation', line: 'Nil by mouth, antibiotics, contrast, and the 24-hour mark',
    spot: ['Chest or epigastric pain after vomiting (Boerhaave) or after endoscopy or dilatation; subcutaneous emphysema; sepsis',
      'Pneumomediastinum, effusion or pneumothorax on X-ray or CT'],
    now: ['Nil by mouth; broad-spectrum IV antibiotics; fluids', 'Water-soluble contrast swallow or CT with oral water-soluble contrast',
      'Drain any pleural collection', 'Contained, early, no sepsis: non-operative or endoscopic (stent or vacuum therapy if contamination is minimal)',
      'Free perforation, sepsis or a large collection: surgery (repair and drainage) as early as possible'],
    nums: [['Treated within 24 h', 'Mortality about 14%'], ['Treated after 24 h', 'Mortality about 27%']],
    call: ['Every suspected perforation: the consultant now'],
    avoid: ['Do not give barium first: use water-soluble contrast'],
    links: ['eso-leak', 'b4-eso-ivor'],
    src: ['Nachira D et al. Management of esophageal perforations and postoperative leaks. Ann Esophagus 2023;6:10 (citing Brinster et al. and the Altorjay criteria).'] },

  { id: 'pe', tag: 'Cardiac', title: 'High-risk pulmonary embolism', line: 'Shock or arrest: reperfuse now',
    spot: ['Cardiac arrest; or systolic under 90 mmHg (or needing vasopressors) with poor perfusion; or a drop of 40 mmHg for over 15 minutes',
      'Bedside echo: a dilated, struggling right ventricle'],
    now: ['Unfractionated heparin bolus', 'Systemic thrombolysis', 'Thrombolysis contraindicated or failed: surgical embolectomy, or catheter-directed treatment'],
    nums: [['Systemic thrombolysis', 'High-risk PE (class I)'], ['Surgical embolectomy', 'Thrombolysis contraindicated or failed (class I)']],
    call: ['The consultant early if thrombolysis is contraindicated (recent surgery): embolectomy needs bypass'],
    links: ['cticu-cardiac'],
    src: ['Konstantinides SV et al. 2019 ESC guidelines for the diagnosis and management of acute pulmonary embolism. Eur Heart J 2020;41:543–603 (also Eur Respir J 2019;54:1901647).'] },

  { id: 'trach', tag: 'Airway', title: 'Tracheostomy emergency', line: 'Blocked or displaced tube: the NTSP sequence',
    spot: ['Breathing difficulty, no end-tidal CO2, high airway pressures, the suction catheter will not pass'],
    now: ['Call for help (anesthesia); oxygen to the face and the stoma', 'Remove the inner tube; pass a suction catheter',
      'Catheter will not pass: deflate the cuff; still not breathing: remove the tracheostomy tube',
      'Ventilate by the face (cover the stoma) or the stoma (small face mask)', 'Re-intubate orally, or through the stoma for a mature stoma'],
    call: ['Fresh tracheostomy (under 7 days) displaced: the stoma is not formed; oral intubation is usually safer'],
    avoid: ['Laryngectomy patients cannot be oxygenated or intubated from the mouth: use the stoma only'],
    links: ['trachea-cervical'],
    src: ['McGrath BA et al. Multidisciplinary guidelines for the management of tracheostomy and laryngectomy airway emergencies. Anaesthesia 2012;67:1025–41.'] },
];

const TAGS = ['All', 'Trauma', 'Cardiac', 'Aorta', 'Vascular', 'Thoracic', 'Airway'] as const;

export function createOnCall(h: H, names: (key: string) => string | null) {
  const btn = h('button', { class: 'btn oncall-btn', onclick: () => open(), title: 'Emergency quick cards: first minutes, numbers, when to call' }, 'On call') as HTMLButtonElement;
  const close = () => document.getElementById('oncall-panel')?.remove();
  const shell = (title: string, back: (() => void) | null, ...kids: (Node | null)[]) => {
    close();
    const panel = h('div', { id: 'oncall-panel', class: 'overlay', role: 'dialog', 'aria-label': title },
      h('div', { class: 'overlay-box acct-box oc-box' }, h('div', { class: 'ov-head' }, back ? h('button', { class: 'btn ghost', onclick: back }, '← Cards') : null, h('b', {}, title),
        h('button', { class: 'btn ghost', onclick: close }, 'Close')), ...kids));
    panel.addEventListener('click', (e) => { if (e.target === panel) close(); });
    document.body.append(panel);
  };
  let tag: string = 'All'; let q = '';
  function open(): void {
    const list = h('div', { class: 'oc-list' });
    const paint = () => {
      const t = q.trim().toLowerCase();
      const cs = CARDS.filter((c) => (tag === 'All' || c.tag === tag) && (!t || [c.title, c.line, ...c.spot, ...c.now].join(' ').toLowerCase().includes(t)));
      list.replaceChildren(...(cs.length ? cs.map((c) => h('button', { class: `oc-item t-${c.tag.toLowerCase()}`, onclick: () => card(c) }, h('small', {}, c.tag), h('b', {}, c.title), h('span', {}, c.line)))
        : [h('p', { class: 'foot' }, 'No card matches.')]));
    };
    const search = h('input', { type: 'search', placeholder: 'Search: tamponade, hemoptysis, AAA…', value: q, oninput: (e: Event) => { q = (e.target as HTMLInputElement).value; paint(); } }) as HTMLInputElement;
    const chips = h('div', { class: 'oc-tags' }, ...TAGS.map((t) => h('button', { class: `chip${t === tag ? ' on' : ''}`, onclick: (e: Event) => { tag = t;
      chips.querySelectorAll('.chip').forEach((x) => x.classList.remove('on')); (e.currentTarget as HTMLElement).classList.add('on'); paint(); } }, t)));
    paint();
    shell('On call', null, search, chips, list,
      h('p', { class: 'foot' }, 'First-minutes aids for residents, not a substitute for senior review or your hospital\'s protocols. Every threshold is from the source on its card. They work offline once COVA has loaded.'));
    setTimeout(() => search.focus(), 50);
  }
  function card(c: Card): void {
    const li = (xs: string[]) => xs.map((x) => h('li', {}, x));
    const mods = c.links.map((k) => { const n = names(k); return n ? h('a', { href: `#approach=${k}&step=0`, onclick: close }, n) : null; }).filter(Boolean) as HTMLElement[];
    shell(c.title, open,
      h('p', { class: 'oc-line' }, h('span', { class: `oc-tag t-${c.tag.toLowerCase()}` }, c.tag), ' ', c.line),
      h('h4', {}, 'Recognize'), h('ul', {}, ...li(c.spot)),
      h('h4', { class: 'oc-now' }, 'Do now'), h('ol', { class: 'oc-steps' }, ...li(c.now)),
      c.nums ? h('div', {}, h('h4', {}, 'The numbers'), h('table', { class: 'oc-nums' }, ...c.nums.map(([a, b]) => h('tr', {}, h('th', {}, a), h('td', {}, b))))) : null,
      h('div', { class: 'oc-call' }, h('h4', {}, 'Call the consultant'), h('ul', {}, ...li(c.call))),
      c.avoid ? h('div', { class: 'oc-avoid' }, h('h4', {}, 'Do not'), h('ul', {}, ...li(c.avoid))) : null,
      mods.length ? h('div', {}, h('h4', {}, 'In the atlas'), h('p', { class: 'oc-mods' }, ...mods.flatMap((m, i) => (i ? [' · ', m] : [m])))) : null,
      h('details', { class: 'oc-src' }, h('summary', {}, `Sources (${c.src.length})`), h('ol', {}, ...li(c.src))));
  }
  return { button: btn, open };
}
