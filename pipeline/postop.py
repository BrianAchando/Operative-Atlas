"""Consent and post-operative (ICU) care for every operation, and the CTICU protocol hub.

Each operation gains two steps: Consent (after the case / decision, before the first incision) and ICU (the last step).
The hub is a separate "operation" (group "CTICU protocol") holding the shared core, labs, escalation, doses and
governance; the per-operation ICU steps link to it.

Every line is tagged by where it comes from: KNH practice (what the unit does), guideline (cited), proposed (awaiting
unit agreement). Doses: Medscape Reference (the unit standard) and product labels, prepared from KNH stock.
"""
from __future__ import annotations

KNH = ' <span class="tag knh">KNH practice</span>'
GL = ' <span class="tag gl">guideline</span>'
PROP = ' <span class="tag prop">proposed</span>'
link = lambda appr, text, step=0: f'<a class="link" href="#approach={appr}&amp;step={step}">{text}</a>'
ul = lambda *xs: '<ul>' + ''.join(f'<li>{x}</li>' for x in xs) + '</ul>'
h4 = lambda t: f'<h4>{t}</h4>'
ev = lambda t: f'<p class="evidence"><b>Evidence:</b> {t}</p>'
red = lambda *xs: '<div class="box red"><h4>Call the consultant</h4>' + ul(*xs) + '</div>'

# ---------------------------------------------------------------------------------------------------- shared tables
LABS = ('<table class="mini"><tr><th style="width:32%">When</th><th>What</th></tr>'
        '<tr><td>On arrival</td><td>ABG (lactate, K⁺, ionised Ca, glucose, Hb), ACT or coagulation screen, chest X-ray, ECG</td></tr>'
        f'<tr><td>First 24 h</td><td>ABG every 4–6 h{KNH}</td></tr>'
        f'<tr><td>Day 1 morning (all)</td><td>FBC, UEC, LFTs, Ca, Mg{KNH}</td></tr>'
        f'<tr><td>Day 1 morning, add</td><td>PO₄ for all; troponin after CABG if ischaemia is suspected{KNH}</td></tr>'
        f'<tr><td>Warfarin started (valves)</td><td>INR 72 h after the first dose, then as the dose is adjusted{KNH}</td></tr>'
        f'<tr><td>Oesophagectomy, day 1 and day 3</td><td>FBC, UEC/LFT, CRP, Ca, Mg, PCT: the trend{KNH}</td></tr>'
        f'<tr><td>Open AAA</td><td>Serial haematocrit 6-hourly for 24 h, then daily{KNH}</td></tr></table>')

ROUND = ('<table class="mini"><tr><th style="width:34%">System</th><th>Review every day</th></tr>'
         '<tr><td>Neuro, analgesia</td><td>GCS or CAM-ICU (delirium), pain score, block working, sedation</td></tr>'
         '<tr><td>Respiratory</td><td>Ventilation or extubation, chest X-ray, air leak, sputum, physiotherapy</td></tr>'
         '<tr><td>Cardiovascular</td><td>MAP, vasoactive drugs and their trend, rhythm, pacing, lactate</td></tr>'
         '<tr><td>Renal, fluids, electrolytes</td><td>Urine output, balance, creatinine, K⁺, Mg²⁺, Ca²⁺, PO₄</td></tr>'
         '<tr><td>Haematology</td><td>Hb or Hct trend, drain output, coagulation, anticoagulation plan</td></tr>'
         '<tr><td>GI, nutrition</td><td>Route and target, NG, bowels, glucose</td></tr>'
         '<tr><td>Infection</td><td>Temperature, CRP or PCT, wounds, lines, antibiotics</td></tr>'
         '<tr><td>Lines, drains, wounds</td><td>What can come out today</td></tr>'
         '<tr><td>Plan</td><td>Step-down criteria met? Family updated?</td></tr></table>')

ESCALATE = ('<div class="box red"><h4>Call the consultant</h4><table class="mini"><tr><th style="width:52%">Trigger</th><th>Think of</th></tr>'
            '<tr><td>Drain over 400 mL in the first hour, over 200 mL/h for several hours, or a sudden gush</td><td>Surgical bleeding: re-exploration</td></tr>'
            '<tr><td>Drains stop, CVP rises, BP and urine fall</td><td>Tamponade</td></tr>'
            '<tr><td>Rising vasoactive need; lactate rising</td><td>Low output, bleeding, tamponade, sepsis</td></tr>'
            '<tr><td>New ST changes after CABG</td><td>Graft failure</td></tr>'
            '<tr><td>New neurological deficit; leg weakness after aortic surgery</td><td>Stroke; spinal cord ischaemia: act within minutes</td></tr>'
            '<tr><td>Urine under 0.5 mL/kg/h for 6 h</td><td>AKI; after AAA, compartment syndrome</td></tr>'
            '<tr><td>Rising girth or tense abdomen after AAA</td><td>Bleeding or compartment syndrome</td></tr>'
            '<tr><td>After oesophagectomy: new AF, fever, rising CRP or PCT, turbid or salivary drain</td><td>Anastomotic leak until proven otherwise</td></tr>'
            '<tr><td>Spreading surgical emphysema, falling saturations, new air–fluid level change</td><td>Air leak, bronchopleural fistula</td></tr></table></div>')

DOSES_E = (
    '<p>Doses as in <b>Medscape Reference</b>, the unit standard' + KNH + ', prepared from KNH stock. Targets are the unit\'s; limits are the product label\'s.</p>'
    + h4('Potassium: KCl 15% (2 mmol/mL; 10 mL = 20 mmol)')
    + '<p>Target after cardiac surgery <b>4.0–5.0 mmol/L</b>' + GL + '. Never undiluted, never as a push.</p>'
    '<table class="mini"><tr><th>Serum K⁺</th><th>Maximum rate</th><th>Max / 24 h</th></tr>'
    '<tr><td>2.5 or more</td><td>10 mmol/h, at no more than 40 mmol/L</td><td>200 mmol</td></tr>'
    '<tr><td>Under 2.0 with ECG changes or paralysis</td><td>up to 40 mmol/h, continuous ECG</td><td>400 mmol</td></tr></table>'
    '<p><b>Peripheral bag</b>: 10 mL KCl 15% (20 mmol) in 500 mL 0.9% saline = 40 mmol/L; 250 mL/h gives 10 mmol/h. Use saline, not dextrose, when urgent. Hold if urine is under 0.5 mL/kg/h or creatinine is rising. Concentrated central-line syringes: unit to decide' + PROP + '</p>'
    + h4('Magnesium: MgSO₄')
    + '<p>Target after cardiac surgery 1.0 mmol/L or more' + PROP + '</p>'
    '<table class="mini"><tr><th>Deficiency</th><th>Dose</th></tr><tr><td>Mild</td><td>1 g IM every 6 h for 4 doses</td></tr>'
    '<tr><td>Severe</td><td>5 g (about 20 mmol) in 1 L 5% dextrose or 0.9% saline IV over 3 h</td></tr></table>'
    '<p>Dilute 50% MgSO₄ to 20% or less for IV use; no faster than 150 mg/min. Severe renal impairment: no more than 20 g in 48 h. Pharmacy to confirm the vial strength stocked.</p>'
    + h4('Calcium: calcium gluconate 10% (10 mL = 1 g, about 2.3 mmol Ca)')
    + '<p>Target ionised Ca 1.1–1.3 mmol/L.</p>'
    '<table class="mini"><tr><th>Ionised Ca</th><th>Dose</th></tr><tr><td>1.0–1.2 (mild)</td><td>1–2 g IV over 2 h</td></tr>'
    '<tr><td>Under 1.0, no tetany</td><td>0.5 mg/kg/h IV, up to 2 mg/kg/h; no more than 3–4 g over 4 h</td></tr>'
    '<tr><td>Tetany or seizures</td><td>about 3 g (30 mL) IV over 5–10 min, then 0.5 mg/kg/h</td></tr></table>'
    '<p>Maximum 50–100 mg/min as a push, 200 mg/min as an infusion. Not in the same line as bicarbonate or phosphate. Pharmacy to confirm whether the mg/kg/h rate is elemental calcium.</p>'
    + h4('Phosphate')
    + '<table class="mini"><tr><th>Serum PO₄</th><th>Dose, IV over 4–6 h</th></tr><tr><td>Under 0.16 mmol/L (0.5 mg/dL)</td><td>0.5 mmol/kg</td></tr>'
    '<tr><td>0.16–0.32 mmol/L (0.5–1 mg/dL)</td><td>0.25 mmol/kg</td></tr></table>'
    '<p>Potassium phosphate if K⁺ is low, sodium phosphate otherwise; caution in renal impairment. Milder deficits: unit to set' + PROP + '</p>'
    + ev('Medscape Reference (calcium gluconate, sodium phosphates IV); potassium chloride and magnesium sulfate prescribing information (product labels, as followed by Medscape).'))

DOSES_V = (
    '<p><b>One standard syringe for both</b>: 4 mL of the 1 mg/mL vial (4 mg) + 46 mL 5% dextrose = 50 mL at <b>80 µg/mL</b>, central line only. Label clearly: the two syringes look the same. The choice of drug and dose is the anaesthetist\'s' + KNH + '</p>'
    + h4('Noradrenaline')
    + '<p>Start 8–12 µg/min and titrate to MAP; maintenance usually 2–4 µg/min. <b>mL/h = µg/min × 0.75</b></p>'
    '<table class="mini"><tr><th>µg/min</th><td>2</td><td>4</td><td>8</td><td>12</td><td>20</td><td>30</td></tr><tr><th>mL/h</th><td>1.5</td><td>3</td><td>6</td><td>9</td><td>15</td><td>22.5</td></tr></table>'
    + h4('Adrenaline')
    + '<p>0.05–2 µg/kg/min, titrated to MAP by 0.05–0.2 µg/kg/min every 10–15 min; wean every 30 min over 12–24 h once stable. <b>mL/h = µg/kg/min × kg × 0.75</b></p>'
    '<table class="mini"><tr><th>kg</th><th>0.02</th><th>0.05</th><th>0.1</th><th>0.2</th><th>0.5</th></tr>'
    '<tr><td>50</td><td>0.8</td><td>1.9</td><td>3.8</td><td>7.5</td><td>18.8</td></tr><tr><td>60</td><td>0.9</td><td>2.3</td><td>4.5</td><td>9.0</td><td>22.5</td></tr>'
    '<tr><td>70</td><td>1.1</td><td>2.6</td><td>5.3</td><td>10.5</td><td>26.3</td></tr><tr><td>80</td><td>1.2</td><td>3.0</td><td>6.0</td><td>12.0</td><td>30.0</td></tr>'
    '<tr><td>90</td><td>1.4</td><td>3.4</td><td>6.8</td><td>13.5</td><td>33.8</td></tr></table>'
    + red('Rising noradrenaline need: look for bleeding, tamponade, sepsis', 'Rising adrenaline need: echo for tamponade, RV or graft failure; consider an IABP')
    + ev('Medscape Reference: norepinephrine (initial 8–12 µg/min, maintenance 2–4 µg/min) and epinephrine (0.05–2 µg/kg/min in shock).'))

CONSENT_T = ('<table class="mini"><tr><th style="width:32%">Part</th><th>Content</th></tr>'
             '<tr><td>1. The patient\'s own risk</td><td>EuroSCORE II or STS PROM (cardiac); ppoFEV1, ppoDLCO and Thoracoscore (thoracic); fitness and cardiac risk (vascular)</td></tr>'
             '<tr><td>2. Common risks</td><td>Occur often; discussed with everyone</td></tr>'
             '<tr><td>3. Serious risks</td><td>Rare but life-changing: death, stroke, paraplegia, amputation</td></tr>'
             '<tr><td>4. Specific to this operation</td><td>The choices and risks only this operation has</td></tr>'
             '<tr><td>5. Alternatives</td><td>Other operations, endovascular, medical therapy, no operation</td></tr>'
             '<tr><td>6. Recovery</td><td>ICU and ward days, drains, return to work, long-term medicines, follow-up</td></tr>'
             '<tr><td>7. Kenya-specific</td><td>INR access, pregnancy on warfarin, penicillin prophylaxis, blood availability, cost, distance to follow-up</td></tr></table>'
             '<p>A structured aid for the conversation: it does not replace the signed KNH consent form. Material risks are those this patient would want to know, not only the frequent ones.</p>')

GOV = ('<table class="mini"><tr><th style="width:32%">Field</th><th>Value</th></tr>'
       '<tr><td>Title</td><td>KNH Thoracic and Cardiovascular ICU Protocol</td></tr><tr><td>Version</td><td>0.1 (draft)</td></tr>'
       '<tr><td>Authors</td><td>Department of Thoracic and Cardiovascular Surgery, KNH/UoN</td></tr>'
       '<tr><td>Sign-off</td><td>Thoracic and cardiovascular surgery; anaesthesia and critical care (vasoactive drugs); pharmacy (concentrations)</td></tr>'
       '<tr><td>Review</td><td>12 months after adoption</td></tr></table>'
       '<div class="box amber"><h4>Still open</h4>' + ul('Bladder pressure after AAA (trigger-based)', 'Refeeding start rate after oesophagectomy', 'Mg²⁺ target 1.0 mmol/L', 'INR in range on two consecutive days before stopping enoxaparin',
                                                        'Pharmacy: MgSO₄ vial strength; calcium infusion units; concentrated central KCl') + '</div>'
       + h4('Audit measures') + ul('Re-exploration for bleeding', 'Time to extubation', 'AKI (KDIGO)', 'Post-operative AF', 'Anastomotic leak after oesophagectomy',
                                   'Compartment syndrome and colonic ischaemia after AAA', 'Spinal cord ischaemia after aortic surgery'))

# ---------------------------------------------------------------------------------------------------- cores
CARD_1 = (h4('Arrival (first 30 min)') + ul('Structured handover: surgeon, anaesthetist, nurse: the operation, bypass and clamp times, how the heart came off bypass, drugs running, pacing, drains',
                                             'ABG, Hb, K⁺, glucose, lactate, ACT or coagulation screen, chest X-ray, ECG; test the epicardial pacing wires')
          + h4('Haemodynamics') + ul('MAP 65–80 mmHg; higher after CABG with carotid disease, lower with a friable aortic suture line',
                                     'Noradrenaline for vasodilatation, adding adrenaline for low output: the anaesthetist\'s choice' + KNH + ' (' + link('cticu-doses', 'doses', 1) + ')',
                                     'Low output: preload, rhythm and pacing first; then inotrope; then IABP. <b>Never miss tamponade</b>')
          + h4('Bleeding') + ul('Re-explore for more than 400 mL in the first hour, more than 300 mL/h for 2–3 h, more than 200 mL/h for 4 h, a sudden gush, or tamponade physiology',
                                'Correct temperature, protamine (by ACT), platelets, fibrinogen, calcium first; do not strip the drains',
                                'Transfuse restrictively: Hb under 7.5 g/dL (TRICS III)' + GL)
          + h4('Ventilation') + ul('Extubate within 6 h when warm, awake, not bleeding, stable, with good gases' + GL)
          + ev('ERAS Cardiac Society guidelines (Engelman et al., JAMA Surg 2019): early extubation, glucose, multimodal analgesia, AKI prevention. TRICS III (Mazer et al., NEJM 2017): restrictive threshold 7.5 g/dL non-inferior.'))
CARD_2 = (h4('Rhythm') + ul('K⁺ 4.0–5.0 mmol/L; keep Mg²⁺ replete (' + link('cticu-doses', 'doses') + ')', 'AF in 20–40%, peaking on days 2–3: beta-blocker prophylaxis, rate or rhythm control; anticoagulate if it lasts beyond 48 h, once bleeding allows')
          + h4('Kidneys, glucose, analgesia, delirium') + ul('AKI prevention: perfusion pressure, no nephrotoxins, urine output (KDIGO)', 'Glucose under 10 mmol/L (180 mg/dL)' + GL,
                                                          'Paracetamol-based multimodal analgesia; avoid NSAIDs', 'Screen for delirium (CAM-ICU); mobilise early')
          + h4('Day 1') + ul('Day-1 labs: FBC, UEC, LFTs, Ca, Mg, PO₄' + KNH, 'Drains out when output is low and serous', 'Sit out, walk, physiotherapy')
          + ev('KDIGO AKI guideline 2012; ERAS Cardiac 2019 (glucose under 180 mg/dL, AF prophylaxis).'))
CARD_3 = (h4('Mechanical valve') + ul('Day 1 morning, or once bleeding has stopped: enoxaparin (Clexane), bridging to warfarin' + KNH,
                                      'First INR 72 h after the first warfarin dose, then as the dose is adjusted' + KNH,
                                      'Stop enoxaparin when the INR is in the target range for the valve type, position and risk factors' + GL + '; in range on two consecutive days' + PROP)
          + '<table class="mini"><tr><th>Prosthesis thrombogenicity</th><th>No risk factor</th><th>Risk factor*</th></tr>'
            '<tr><td>Low (e.g. St Jude, On-X, Carbomedics, Medtronic Hall, ATS)</td><td>2.5</td><td>3.0</td></tr><tr><td>Medium (other bileaflet valves)</td><td>3.0</td><td>3.5</td></tr>'
            '<tr><td>High (tilting disc such as Bj\u00f6rk-Shiley; caged ball such as Starr-Edwards)</td><td>3.5</td><td>4.0</td></tr></table>'
            '<p>Target INR. *Risk factors: mitral or tricuspid replacement, previous thromboembolism, AF, mitral stenosis of any degree, LVEF under 35%.</p>'
          + h4('Tissue valve, repair, rheumatic AF') + ul('Tissue mitral valve or mitral repair: warfarin for 3 months' + GL, 'Rheumatic AF: warfarin, not rivaroxaban (INVICTUS)' + GL,
                                                         'Continue secondary penicillin prophylaxis after rheumatic valve surgery')
          + ev('ESC/EACTS 2021 valvular heart disease guideline (INR targets by prosthesis and patient risk factors; VKA after bioprosthetic MVR or repair for 3 months). INVICTUS (Connolly et al., NEJM 2022).'))
THX_1 = (h4('Arrival') + ul('Chest X-ray: expansion, drain position, mediastinum, surgical emphysema; ABG; Hb', 'Check that the regional block works: paravertebral, erector spinae or epidural')
         + h4('Analgesia is lung function') + ul('Block + paracetamol ± NSAID (kidneys permitting), low opioid', 'Sit up and walk on day 0–1; physiotherapy, incentive spirometry',
                                                 'Sputum retention: physio, then bedside bronchoscopy, then minitracheostomy')
         + h4('Fluids, rhythm, VTE') + ul('Euvolaemic to restrictive; no liberal crystalloid', 'AF: continue beta-blockers, keep K⁺ and Mg²⁺ replete', 'LMWH + mechanical prophylaxis; extended prophylaxis for high-risk cancer resections')
         + ev('ERAS Society / ESTS guidelines for enhanced recovery after lung surgery (Batchelor et al., Eur J Cardiothorac Surg 2019).'))
THX_2 = (h4('Drains') + ul('One drain after lobectomy; no routine suction' + GL, 'Record the air leak every shift: none, on cough, continuous',
                           'Remove when there is no air leak and non-chylous output is up to 450 mL/24 h' + GL, 'Prolonged air leak: more than 5 days')
         + h4('Read the drain') + ul('Serous: expected', 'Blood: over the thresholds, re-explore', 'Turbid or salivary after oesophagectomy: leak', 'Milky: chyle (send triglycerides)')
         + h4('Pneumonectomy') + ul('<b>No suction</b> on the pneumonectomy drain (clamped or balanced)', 'Strict fluids: post-pneumonectomy pulmonary oedema is rare but often fatal',
                                     'BPF at days 7–14 (right side, after chemoradiation, TB stump): lie operated side down, drain the space, bronchoscopy')
         + ev('Batchelor 2019 (drain removal up to 450 mL/24 h, no routine suction). Post-pneumonectomy pulmonary oedema: Parquin et al. (Eur J Cardiothorac Surg 1996); Campisi (Shanghai Chest).'))
VASC_1 = (h4('Open aortic surgery') + ul('Hourly urine output; foot pulses and Doppler; wounds', 'Serial haematocrit 6-hourly for 24 h, then daily' + KNH,
                                         'Abdominal girth at the umbilicus against a marked line, every 4–6 h' + KNH,
                                         'When girth rises, the abdomen is tense, after rupture or massive transfusion, or with oliguria despite filling: bladder pressure' + PROP)
          + '<p><b>Why bladder pressure</b>: clinical examination misses raised abdominal pressure (sensitivity about 40–60%), and oliguria after a clamp may be hypovolaemia, AKI or compartment syndrome: fluid worsens the last. Foley, 25 mL saline, zero at the mid-axillary line, read at end-expiration. Intra-abdominal hypertension is 12 mmHg or more; compartment syndrome over 20 mmHg with new organ failure: decompress.</p>'
          + ul('Colonic ischaemia: bloody diarrhoea or rising lactate: sigmoidoscopy', 'Trash foot: distal emboli', 'AKI after juxta- or suprarenal clamping')
          + ev('WSACS 2013 definitions (Kirkpatrick et al., Intensive Care Med 2013). Kirkpatrick et al. (Can J Surg 2000): clinical examination sensitivity 40–56%. ACS after endovascular repair of rupture: 17–21% in studies with proper definitions, mortality 47% (J Vasc Surg meta-analysis). ESVS 2024 AAA guideline.'))
VASC_2 = (h4('Spinal cord (TAA, TEVAR)') + ul('MAP target 80–90 mmHg or more', 'CSF drain at 10–12 mmHg; hourly leg checks', 'New weakness: raise MAP, drain CSF, keep Hb up, call immediately')
          + h4('Bypass and endovascular') + ul('Graft and foot Doppler signals hourly at first; groin wounds (lymph leak, infection)',
                                               'Axillobifemoral: no BP cuff on the donor arm; do not lie on the graft side', 'Puncture sites: haematoma; back pain and a falling Hb: retroperitoneal bleed',
                                               'EVAR and TEVAR: contrast AKI; post-implantation fever')
          + ev('ESVS 2024 AAA guideline; ESC 2024 and ACC/AHA 2022 aortic guidelines (spinal cord protection, CSF drainage).'))

# ---------------------------------------------------------------------------------------------------- per operation
# kind: which core; consent: parts 1–7 as lists; icu: specific items; q1 consent MCQ, q2 ICU MCQ (ask args)
LUNG_C = dict(risk='ppoFEV1 and ppoDLCO (stair climb or CPET if low); Thoracoscore',
              common=['Prolonged air leak', 'AF', 'Pneumonia and sputum retention', 'Bleeding', 'Wound infection', 'Chronic chest wall pain (more after thoracotomy)'],
              serious=['Death', 'Respiratory failure and ventilation', 'Bronchopleural fistula', 'Return to theatre'],
              kenya=['Post-TB adhesions and calcified nodes raise bleeding and conversion risk', 'Exclude TB before calling a mass cancer', 'Cost and access to adjuvant treatment'])
OPS = {
    'lobe': dict(kind='thx', c=dict(**LUNG_C, specific=['Conversion from VATS to open', 'Final staging on the specimen may change the plan (adjuvant therapy)', 'Recurrence'],
                                    alt=['Segmentectomy for small peripheral tumours', 'Stereotactic radiotherapy (SBRT) if unfit', 'Surveillance for indeterminate nodules']),
                 icu=['Air leak: record every shift; prolonged beyond 5 days', 'Lobar torsion (especially the middle lobe): a lobe that opacifies on X-ray', 'Sputum retention'],
                 q1=('Which of these belongs in consent for a VATS lobectomy in a patient with healed TB?', 'A higher chance of conversion to open and of bleeding from adhesions and calcified nodes',
                     'Post-TB pleural and nodal changes make VATS dissection harder; the conversion risk is a material risk.', 'A guaranteed shorter stay', 'No chest drain', 'No risk of air leak'),
                 q2=('Day 2 after right upper lobectomy: the middle lobe has opacified and the patient is febrile. Most important diagnosis to exclude?', 'Middle lobe torsion',
                     'Torsion is rare but causes infarction; bronchoscopy and return to theatre if suspected.', 'Atelectasis only: physiotherapy', 'Pleural effusion', 'Normal after resection')),
    'seg': dict(kind='thx', c=dict(**LUNG_C, specific=['Local recurrence somewhat higher than lobectomy, survival not worse for small peripheral tumours (JCOG0802)', 'Conversion to lobectomy if margins or nodes demand it'],
                                   alt=['Lobectomy', 'SBRT if unfit']),
                icu=['Air leak along the intersegmental plane is common', 'Sputum retention'],
                q1=('For a 1.8 cm peripheral NSCLC, what does JCOG0802 let you tell the patient about segmentectomy versus lobectomy?', 'Overall survival was better with segmentectomy, although local recurrence was higher',
                    'JCOG0802 (Lancet 2022): 5-year OS 94.3% vs 91.1%; local recurrence 10.5% vs 5.4%.', 'Segmentectomy has no recurrence', 'Lobectomy has better survival', 'They are identical in every respect'),
                q2=('Day 3 after segmentectomy: small continuous air leak, lung up, patient well. Next?', 'Keep the drain, no suction, reassess daily; consider a portable valve if it persists',
                    'Intersegmental-plane leaks usually seal; prolonged leak is defined beyond 5 days.', 'Return to theatre today', 'Clamp the drain', 'High suction')),
    'pn': dict(kind='thx', c=dict(**LUNG_C, specific=['Higher mortality than lobectomy (right more than left)', 'Permanent reduction in exercise tolerance', 'Post-pneumonectomy pulmonary oedema', 'Cardiac herniation (intrapericardial)'],
                                  alt=['Sleeve lobectomy where anatomy allows', 'Chemoradiation']),
               icu=['<b>No suction</b> on the pneumonectomy drain; daily chest X-ray for mediastinal shift', 'Strict fluid limits', 'Sudden collapse after an intrapericardial resection: cardiac herniation: operated side up, back to theatre',
                    'BPF (days 7–14): fever, serosanguinous sputum, falling fluid level: lie operated side <b>down</b>'],
               q1=('Why is sleeve lobectomy offered, where possible, instead of pneumonectomy?', 'Similar cancer control with lower mortality and better lung function',
                   'Parenchyma-sparing: pneumonectomy carries higher mortality and permanent loss of function.', 'It is quicker', 'It needs no bronchial anastomosis', 'It avoids a thoracotomy'),
               q2=('Day 9 after right pneumonectomy: coughing serosanguinous fluid, fluid level has fallen. First action?', 'Lie the patient operated side down, then drain the space and arrange bronchoscopy',
                   'Bronchopleural fistula: protect the remaining lung from spill-over first.', 'Lie operated side up', 'Suction on the drain', 'Start a diuretic')),
    'trachea': dict(kind='thx', c=dict(risk='Airway assessment, fitness, prior tracheostomy or steroid use', common=['Sore throat, hoarseness', 'Wound infection', 'Swallowing difficulty for a few days'],
                                       serious=['Anastomotic dehiscence (airway emergency)', 'Restenosis', 'Recurrent laryngeal nerve injury', 'Need for tracheostomy', 'Death'],
                                       specific=['Chin-to-chest guardian stitch for 7 days', 'Check bronchoscopy before discharge'], alt=['Dilatation, stent', 'Tracheostomy or T-tube'],
                                       kenya=['Post-intubation stenosis is common: ICU follow-up of the cause', 'Access to repeat bronchoscopy']),
                    icu=['Extubate in theatre if possible', 'Guardian stitch 7 days; neck flexed, no hyperextension', 'Stridor or neck emphysema: dehiscence: urgent bronchoscopy'],
                    q1=('Which instruction must the patient understand before tracheal resection?', 'The neck will be held flexed by a chin-to-chest stitch for about a week',
                        'It protects the anastomosis from tension; understanding it prevents the patient fighting it.', 'Early neck extension exercises', 'No voice rest needed', 'A permanent tracheostomy is routine'),
                    q2=('Day 2 after tracheal resection: new stridor and neck surgical emphysema. Next?', 'Call the consultant and prepare urgent bronchoscopy in theatre',
                        'Suspect anastomotic dehiscence; secure the airway under direct vision.', 'Nebulised adrenaline and observe', 'Remove the guardian stitch', 'Increase humidification only')),
    'thymectomy': dict(kind='thx', c=dict(risk='Myasthenia severity and control; FVC', common=['Pain', 'Wound problems', 'Pneumothorax (VATS)'],
                                          serious=['Myasthenic crisis needing ventilation', 'Phrenic nerve injury', 'Bleeding', 'Deep sternal wound infection (sternotomy)'],
                                          specific=['Benefit accrues over months to years; fewer steroids (MGTX)', 'Medicines continue around surgery'], alt=['Medical therapy alone'],
                                          kenya=['Access to IVIG or plasma exchange if crisis occurs', 'Pyridostigmine supply']),
                       icu=['Continue pyridostigmine and steroids', 'Avoid aggravating drugs: aminoglycosides, fluoroquinolones, magnesium boluses, excess neuromuscular blockers',
                            'Myasthenic crisis: <b>20/30/40 rule</b> (FVC under 20 mL/kg, NIF weaker than −30, PEF under 40): intubate; IVIG or plasma exchange', 'Diaphragm on chest X-ray: phrenic injury'],
                       q1=('What benefit can a patient with generalised MG expect from thymectomy (MGTX)?', 'Better clinical scores and less prednisone over 3 years',
                           'MGTX (NEJM 2016): improved Quantitative MG score and lower prednisone dose at 3 years; benefit is gradual.', 'Immediate cure', 'Stopping all medicines the next day', 'No benefit'),
                       q2=('After thymectomy for MG: FVC falls to 15 mL/kg with a weak cough. Next?', 'Call the consultant; prepare to intubate and start IVIG or plasma exchange',
                           'Myasthenic crisis: the 20/30/40 rule; do not wait for hypercapnia.', 'Give magnesium', 'Increase opioids', 'Start gentamicin')),
    'oesophagectomy': dict(kind='thx', c=dict(risk='Fitness, nutrition (weight loss, albumin), cardiorespiratory reserve', common=['Pneumonia', 'AF', 'Wound infection', 'Hoarse voice (neck anastomosis)'],
                                              serious=['Anastomotic leak', 'Conduit necrosis', 'Chylothorax', 'Recurrent laryngeal nerve palsy', 'Death'],
                                              specific=['Long-term: reflux (sleep head-up), early satiety, dumping, stricture needing dilatation, weight loss', 'Feeding jejunostomy for weeks'],
                                              alt=['Definitive chemoradiation', 'Palliation: stent, radiotherapy'],
                                              kenya=['Most present late and malnourished: pre-operative nutrition', 'The realistic chance of cure, and palliation as an option']),
                           icu=['Labs day 1 and day 3: FBC, UEC/LFT, CRP, Ca, Mg, PCT (trend)' + KNH, 'POD 3 CRP under 17.6 mg/dL (176 mg/L), POD 5 under 13.2 mg/dL (132 mg/L): leak unlikely' + GL,
                                'Jejunostomy feeds from day 1–2' + KNH + '; refeeding precautions in the malnourished (start at no more than 10 kcal/kg/day; PO₄, K⁺, Mg²⁺ daily for 3 days)' + PROP,
                                'Oral sips day 6–7 with a clean drain, falling CRP and a well patient' + KNH, 'NG on free drainage, never re-passed blind; head up 30°',
                                'New AF, fever, rising CRP or PCT, turbid or salivary drain: CT with oral contrast, then endoscopy'],
                           q1=('Which long-term effect must be discussed before oesophagectomy?', 'Reflux, early satiety, dumping and possible anastomotic stricture',
                               'These shape life after surgery and are part of informed consent.', 'Better appetite than before', 'No dietary change', 'Hair loss'),
                           q2=('POD 3 after Ivor Lewis: new AF, CRP 24 mg/dL and cloudy drain fluid. Next?', 'Treat as an anastomotic leak until proven otherwise: CT with oral contrast, then endoscopy',
                               'New AF and a CRP above about 17.6 mg/dL on POD 3 with turbid drain fluid point to a leak.', 'Rate control only', 'Start oral sips', 'Remove the chest drain'),
                           ev='ERAS Society oesophagectomy guidelines (Low et al., World J Surg 2019); ECCG definitions (Low et al., Ann Surg 2015); Aiolfi et al. (PLoS One 2018) CRP meta-analysis; NICE CG32 (refeeding).'),
    'duct': dict(kind='thx', c=dict(risk='Nutritional and immune state after chyle loss', common=['Pain', 'Wound infection'], serious=['Persistent chyle leak', 'Bleeding'],
                                    specific=['Output should fall within 24–48 h'], alt=['Conservative: fat-free diet or TPN (± octreotide)', 'Thoracic duct embolisation where available'],
                                    kenya=['Access to TPN and embolisation']),
                 icu=['Chest drain output should fall within 24–48 h', 'Replace protein, fluid and lymphocyte losses; nutrition'],
                 q1=('When is thoracic duct ligation offered?', 'When a chylothorax persists or has a high output despite conservative treatment',
                     'Surgery follows a failed trial of fat-free feeding or TPN, especially with high output.', 'For every chylothorax on day 1', 'Never', 'Only for malignancy'),
                 q2=('24 h after duct ligation, output is unchanged at 1.5 L of milky fluid. Next?', 'Call the consultant: ligation may have failed',
                     'Output should fall within 24–48 h; persistence suggests a missed or accessory duct.', 'Start a normal diet', 'Remove the drain', 'Give furosemide')),
    'empyema': dict(kind='thx', c=dict(risk='Sepsis, nutrition, lung trapped or not', common=['Air leak', 'Bleeding from the decorticated surface', 'Pain'],
                                       serious=['Persistent space or recurrence', 'Sepsis', 'Conversion to open'], specific=['Antibiotics continue for weeks'],
                                       alt=['Chest drain with intrapleural tPA and DNase (MIST2)', 'Open drainage'], kenya=['Test for TB (GeneXpert) and treat', 'HIV testing']),
                    icu=['Expect air leak and ooze', 'Ongoing sepsis: undrained collection: CT', 'Culture-directed antibiotics; TB screen'],
                    q1=('Before decortication for empyema, which alternative should the patient hear about?', 'A chest drain with intrapleural tPA and DNase',
                        'MIST2 (NEJM 2011): improved drainage and fewer surgical referrals.', 'Antibiotics by mouth only', 'Observation', 'Pneumonectomy'),
                    q2=('Day 3 after decortication: persistent fever, new loculated collection on CT. Next?', 'Drain the collection and review cultures, including TB',
                        'Undrained pus is the usual cause of persistent sepsis.', 'Change antibiotics only', 'Remove all drains', 'Discharge')),
    'ppe': dict(kind='thx', c=dict(risk='Sepsis, nutrition, fistula', common=['Long course of dressings', 'Pain'], serious=['Persistent bronchopleural fistula', 'Death'],
                                   specific=['Open window for weeks to months', 'Later closure (Clagett or flap)'], alt=['Drainage alone', 'Muscle flap and thoracoplasty'],
                                   kenya=['Dressing supplies and follow-up distance']),
                icu=['Window dressings; nutrition', 'Watch for fistula closure or reopening', 'Plan the Clagett or flap'],
                q1=('What must a patient know before an open window thoracostomy?', 'The chest stays open for weeks to months, with regular dressings, before closure',
                    'Expectations about a long course prevent distress and non-attendance.', 'It closes in a day', 'No dressings are needed', 'It cures a fistula immediately'),
                q2=('An open window patient coughs up dressing fluid. What does it suggest?', 'A bronchopleural fistula is open',
                    'Fluid from the space entering the airway means the fistula is patent: review and plan closure.', 'Normal healing', 'Pneumonia only', 'Wound infection only')),
    'trauma': dict(kind='thx', c=dict(risk='Emergency', common=[], serious=[], specific=[], alt=[], kenya=[]), emergency=True,
                   icu=['Damage control: correct the lethal triad (hypothermia, acidosis, coagulopathy)', '1:1:1 transfusion; TXA within 3 h of injury (CRASH-2)' + GL,
                        'Repeat and tertiary survey: missed injuries', 'After hilar twist or pneumonectomy for trauma: RV failure: avoid overload, inodilator', 'Air embolism after lung injury',
                        'Cardiorrhaphy: echo for residual VSD, valve injury, effusion'],
                   q1=('An unconscious patient needs a resuscitative thoracotomy. How is consent handled?', 'Proceed on necessity to save life, document it, and inform the next of kin as soon as possible',
                       'Emergency treatment without consent is justified when the patient cannot consent and delay risks life.', 'Wait for the family', 'Do not operate', 'Ask the police'),
                   q2=('After a clamshell for penetrating trauma: temperature 34 °C, pH 7.1, oozing. Priority?', 'Correct the lethal triad: warm, transfuse 1:1:1, give TXA if within 3 h',
                       'Damage-control resuscitation before further surgery.', 'Return to theatre immediately for definitive repair', 'Crystalloid boluses', 'Extubate'),
                   ev='CRASH-2 (Lancet 2010): TXA within 3 h reduces death from bleeding. Damage-control resuscitation principles (ATLS 10th edition).'),
    # ------------------------------------------------------------------ cardiac
    'mvr': dict(kind='card', c=dict(risk='EuroSCORE II (or STS PROM)', common=['Bleeding and return to theatre', 'AF', 'Chest and wound infection', 'Kidney injury'],
                                    serious=['Stroke', 'Death', 'Deep sternal wound infection', 'Heart block needing a pacemaker'],
                                    specific=['Mechanical: lifelong warfarin, INR checks, bleeding, valve thrombosis if INR lapses', 'Pregnancy on warfarin: embryopathy and valve thrombosis; plan before conception',
                                              'Tissue: degenerates faster in the young; reoperation likely', 'Repair where possible'],
                                    alt=['Balloon mitral valvotomy for pliable MS without LA thrombus or significant MR', 'Medical therapy'],
                                    kenya=['INR testing near home', 'Continue benzathine penicillin prophylaxis', 'Blood availability', 'Cost of follow-up']),
                icu=['Pulmonary hypertension and RV failure: avoid hypoxia, hypercapnia, acidosis; inodilator if the RV fails', 'AF: rate or rhythm control',
                     'Anticoagulation: ' + link('cticu-cardiac', 'enoxaparin day 1, bridge to warfarin, INR at 72 h', 2) + KNH],
                q1=('A 24-year-old woman planning children needs MVR. Which topic must her consent cover that a 65-year-old\'s would not?', 'Pregnancy on warfarin: embryopathy, valve thrombosis, and a plan before conception',
                    'Valve choice in young women is decided with pregnancy in mind (ESC 2018 pregnancy guideline).', 'The risk of AF', 'Wound infection', 'Return to theatre for bleeding'),
                q2=('Day 1 after mechanical MVR, drains dry. When is the first INR checked?', '72 h after the first warfarin dose, while enoxaparin continues',
                    'Warfarin takes 2–3 days to move the INR; enoxaparin bridges until the INR is in range.', 'Two hours after the first dose', 'Never: enoxaparin alone', 'Only at discharge'),
                ev='ESC/EACTS 2021 valve guideline; ESC 2018 pregnancy guideline; INVICTUS (NEJM 2022).'),
    'avr': dict(kind='card', c=dict(risk='EuroSCORE II (or STS PROM)', common=['Bleeding', 'AF', 'Wound infection', 'Kidney injury'],
                                    serious=['Stroke', 'Death', 'Complete heart block needing a permanent pacemaker', 'Paravalvular leak'],
                                    specific=['Mechanical versus tissue valve: warfarin against durability', 'Patient–prosthesis mismatch with a small annulus'],
                                    alt=['TAVI where available and suitable', 'Balloon valvotomy as a bridge'], kenya=['INR access', 'Penicillin prophylaxis if rheumatic', 'Cost']),
                icu=['Heart block (conduction tissue beside the annulus): test wires; pacemaker if block persists', 'Hypertrophied ventricle: keep filled, avoid tachycardia',
                     'Mechanical valve: ' + link('cticu-cardiac', 'enoxaparin, warfarin, INR at 72 h', 2) + KNH],
                q1=('Which AVR risk is specific to the valve\'s position next to the conduction system?', 'Complete heart block needing a permanent pacemaker',
                    'The His bundle runs beneath the non-coronary/right-coronary commissure.', 'Mitral stenosis', 'Chylothorax', 'Paraplegia'),
                q2=('After AVR, complete heart block: the epicardial wires are not capturing. Next?', 'Call the consultant; increase output, check connections, prepare transcutaneous pacing',
                    'Loss of capture in complete heart block is an emergency.', 'Observe', 'Give a beta-blocker', 'Remove the wires')),
    'root': dict(kind='card', c=dict(risk='EuroSCORE II', common=['Bleeding', 'AF', 'Wound infection'], serious=['Stroke', 'Death', 'Coronary button ischaemia', 'Heart block'],
                                     specific=['Bentall: mechanical or tissue conduit', 'David: aortic regurgitation may recur', 'Ross: two valves at risk (autograft dilatation, homograft degeneration)'],
                                     alt=['Separate valve and ascending replacement', 'Surveillance below the threshold'], kenya=['INR access for mechanical conduits', 'Imaging follow-up']),
                 icu=['Strict blood pressure control to protect suture lines', 'New ST change or ventricular arrhythmia: coronary button ischaemia', 'Ross: control autograft pressure'],
                 q1=('What should a Ross patient understand about reoperation?', 'Both the autograft and the pulmonary homograft may need later intervention',
                     'The Ross converts single-valve disease into two-valve disease.', 'Reoperation never occurs', 'Only the mitral valve is at risk', 'Warfarin is lifelong'),
                 q2=('After a Bentall, new ST elevation in the inferior leads. Think of?', 'Right coronary button kinking or ischaemia',
                     'Coronary button problems present early with ST change or arrhythmia: echo and angiography.', 'Pericarditis only', 'Normal after bypass', 'Hypokalaemia')),
    'tricuspid': dict(kind='card', c=dict(risk='EuroSCORE II; RV function; liver and kidneys', common=['Bleeding', 'AF', 'Kidney injury'], serious=['Heart block needing a pacemaker', 'RV failure', 'Death'],
                                          specific=['Ring repair versus replacement'], alt=['Medical therapy (diuretics)'], kenya=['Pacemaker cost and follow-up']),
                      icu=['Heart block: test wires; permanent pacemaker if it persists', 'RV failure: inodilator, avoid overload, keep the pulmonary resistance low'],
                      q1=('Which risk is higher after tricuspid surgery than after most valve operations?', 'Heart block needing a permanent pacemaker',
                          'The AV node lies close to the septal leaflet annulus.', 'Paraplegia', 'Chylothorax', 'Colonic ischaemia'),
                      q2=('After tricuspid repair: CVP rising, low output, RV dilated on echo. Priority?', 'Treat RV failure: inodilator, avoid more fluid, lower pulmonary resistance',
                          'The RV fails with overload and high afterload.', 'Fluid bolus', 'Increase PEEP a lot', 'Beta-blocker')),
    'cabg': dict(kind='card', c=dict(risk='EuroSCORE II; SYNTAX score for the decision', common=['Bleeding', 'AF', 'Leg wound problems (saphenous vein)', 'Kidney injury'],
                                     serious=['Stroke', 'MI', 'Death', 'Deep sternal wound infection (more with both mammaries in diabetics)'],
                                     specific=['Radial artery harvest: hand ischaemia (Allen test)', 'On-pump versus off-pump'],
                                     alt=['PCI', 'Medical therapy'], kenya=['Secondary prevention and cost of medicines', 'Access to follow-up']),
                 icu=['Aspirin within 6 h once bleeding settles; statin; beta-blocker' + GL, 'New ST change or ventricular arrhythmia: graft failure until proven otherwise', 'Radial graft: vasospasm prophylaxis per unit'],
                 q1=('For a diabetic with three-vessel disease, what does the evidence let you say about CABG versus PCI?', 'CABG reduces death and MI compared with PCI (FREEDOM)',
                     'FREEDOM (NEJM 2012): lower all-cause mortality and MI with CABG in diabetics, with more stroke.', 'PCI is always better', 'They are equal', 'Medical therapy is best'),
                 q2=('Four hours after CABG, bleeding has settled. What should start within 6 h?', 'Aspirin',
                     'Early aspirin after CABG reduces death and ischaemic complications (Mangano, NEJM 2002).', 'Warfarin', 'Clopidogrel loading only', 'Nothing until day 3')),
    # ------------------------------------------------------------------ vascular
    'aaa-open': dict(kind='vasc', c=dict(risk='Cardiac, respiratory and renal fitness', common=['Ileus', 'Chest infection', 'Wound problems, incisional hernia'],
                                         serious=['MI', 'Kidney failure (higher with juxta- or suprarenal clamping)', 'Colonic ischaemia', 'Limb ischaemia', 'Death'],
                                         specific=['Sexual dysfunction (retrograde ejaculation)', 'Graft infection'], alt=['EVAR if anatomy suits', 'Surveillance below the threshold'],
                                         kenya=['Blood availability', 'Cost']),
                     icu=['Serial haematocrit and abdominal girth' + KNH + '; ' + link('cticu-vascular', 'bladder pressure when triggered') + PROP, 'Colonic ischaemia: bloody diarrhoea, rising lactate: sigmoidoscopy',
                          'Feet: trash foot', 'Kidneys after supra- or juxtarenal clamps'],
                     q1=('Which open AAA risk should be discussed with a sexually active man?', 'Sexual dysfunction from injury to the hypogastric plexus',
                         'The autonomic nerves over the left common iliac and aortic bifurcation can be injured.', 'Blindness', 'Hoarseness', 'Paraplegia is common'),
                     q2=('Day 1 after rupture repair: girth rising, urine 10 mL/h despite filling, airway pressures rising. Next?', 'Measure bladder pressure and call the consultant: compartment syndrome',
                         'Oliguria despite filling with a tense abdomen: more fluid makes it worse.', 'More fluid boluses', 'Furosemide', 'Observe until morning')),
    'evar': dict(kind='vasc', c=dict(risk='Anatomy (neck, access), renal function', common=['Groin haematoma', 'Post-implantation fever'],
                                     serious=['Endoleak and late rupture', 'Reintervention', 'Contrast kidney injury', 'Limb occlusion'],
                                     specific=['Lifelong imaging surveillance'], alt=['Open repair'], kenya=['Access to CT surveillance', 'Cost of the device']),
                 icu=['Puncture sites; distal pulses', 'Contrast AKI: creatinine', 'Post-implantation fever is common and sterile'],
                 q1=('What commitment does EVAR need from the patient?', 'Lifelong imaging surveillance for endoleak and migration',
                     'EVAR-1: more reinterventions over time; surveillance detects them.', 'None after discharge', 'Weekly INR', 'Annual chest X-ray only'),
                 q2=('Day 1 after EVAR: fever 38.4 °C, well, wounds clean. Most likely?', 'Post-implantation syndrome',
                     'A sterile inflammatory response is common after EVAR; examine and culture if unwell.', 'Graft infection', 'Endoleak', 'Colonic ischaemia')),
    'taa-open': dict(kind='vasc', c=dict(risk='Cardiac, respiratory and renal fitness', common=['Chest infection', 'Pain'], serious=['Paraplegia', 'Stroke', 'Kidney failure', 'Left vocal cord palsy', 'Death'],
                                         specific=['CSF drain and its risks (headache, bleeding)'], alt=['TEVAR if anatomy suits', 'Surveillance below the threshold'], kenya=['Blood and ICU availability']),
                     icu=[link('cticu-vascular', 'Spinal cord protection', 1) + ': MAP 80–90 or more, CSF drain 10–12 mmHg, hourly leg checks', 'Kidneys, lungs (thoracotomy)'],
                     q1=('Which risk must be discussed before open descending thoracic aneurysm repair?', 'Paraplegia from spinal cord ischaemia',
                         'A defining risk of thoracic aortic surgery.', 'Hand ischaemia', 'Mitral regurgitation', 'Chylothorax only'),
                     q2=('Six hours after thoracic aortic repair: new bilateral leg weakness. First action?', 'Raise MAP, drain CSF to target, keep Hb up, call immediately',
                         'Spinal cord ischaemia can recover if perfusion pressure is restored quickly.', 'Wait for MRI tomorrow', 'Lower the blood pressure', 'Remove the CSF drain')),
    'tevar': dict(kind='vasc', c=dict(risk='Anatomy (landing zones), access', common=['Groin haematoma', 'Post-implantation fever'],
                                      serious=['Paraplegia (lower than open)', 'Stroke', 'Endoleak', 'Left arm ischaemia if the subclavian is covered'],
                                      specific=['Lifelong imaging'], alt=['Open repair'], kenya=['Access to CT surveillance', 'Device cost']),
                  icu=['Spinal cord protection as for open repair', 'Puncture site; left arm pulses if the subclavian was covered'],
                  q1=('What must a TEVAR patient accept?', 'Lifelong imaging surveillance for endoleak',
                      'Endovascular repair needs follow-up imaging.', 'Weekly INR', 'Nothing after discharge', 'A permanent drain'),
                  q2=('After TEVAR with left subclavian coverage: cold left hand. Next?', 'Assess and call: arm ischaemia may need revascularisation',
                      'Covering the subclavian without revascularisation can cause arm ischaemia.', 'Normal: ignore', 'Raise the arm', 'Heparin only')),
    'taa-asc': dict(kind='card', c=dict(risk='EuroSCORE II', common=['Bleeding', 'AF'], serious=['Stroke (circulatory arrest)', 'Death', 'Kidney injury'],
                                        specific=['Hypothermic circulatory arrest', 'Treatment of syphilis if aortitis'], alt=['Surveillance below the threshold'], kenya=['Syphilis treatment and partner testing']),
                    icu=['Neurological assessment on waking', 'Coagulopathy after circulatory arrest: platelets, fibrinogen', 'Penicillin for cardiovascular syphilis'],
                    q1=('Which risk is specific to hemiarch repair under circulatory arrest?', 'Stroke and neurological injury',
                        'Brain protection (antegrade cerebral perfusion, hypothermia) reduces but does not remove it.', 'Paraplegia is the main risk', 'Chylothorax', 'Heart block'),
                    q2=('After hemiarch repair: oozing, platelets 60, fibrinogen 1.0 g/L. Next?', 'Give platelets and fibrinogen (cryoprecipitate)',
                        'Correct the coagulopathy of circulatory arrest before re-exploring.', 'Re-open immediately', 'More protamine only', 'Observe')),
    'abf': dict(kind='vasc', c=dict(risk='Cardiac and respiratory fitness', common=['Ileus', 'Groin lymph leak or infection'], serious=['Graft infection', 'Limb occlusion', 'Colonic ischaemia', 'Aortoenteric fistula (late)', 'Death'],
                                    specific=['Sexual dysfunction'], alt=['Endovascular (kissing stents, CERAB)', 'Axillobifemoral if unfit'], kenya=['Cost; follow-up']),
                icu=['Graft and foot Doppler hourly at first', 'Groin wounds', 'Colon: bloody diarrhoea'],
                q1=('Which late complication should the ABF patient know to report?', 'Gastrointestinal bleeding (aortoenteric fistula) or a groin lump or infection',
                    'Late graft complications present this way.', 'Hoarseness', 'Headache', 'Hair loss'),
                q2=('Day 1 after ABF: right foot cold, no Doppler signal. Next?', 'Call the consultant: limb occlusion needs thrombectomy',
                    'Early limb thrombosis is a technical problem until proven otherwise.', 'Elevate the leg', 'Observe', 'Start aspirin only')),
    'axbf': dict(kind='vasc', c=dict(risk='Arm pressures; subclavian inflow', common=['Wound problems'], serious=['Graft thrombosis', 'Graft infection', 'Axillary anastomosis disruption'],
                                     specific=['Lower patency than aortobifemoral'], alt=['Aortobifemoral if fit', 'Endovascular'], kenya=['Follow-up']),
                 icu=['No BP cuff on the donor arm; do not lie on the graft side', 'Graft pulse along the chest wall; feet Doppler'],
                 q1=('Why is axillobifemoral chosen despite poorer patency?', 'It avoids a laparotomy and aortic clamp in a high-risk patient',
                     'Extra-anatomic bypass trades durability for lower operative risk.', 'It lasts longer', 'It is cheaper to maintain', 'It needs no anaesthetic'),
                 q2=('Which instruction protects an axillobifemoral graft?', 'No BP cuff on the donor arm and do not lie on the graft side',
                     'External compression can thrombose the graft.', 'Tight belt support', 'Sleep on the graft side', 'Arm raised constantly')),
    'endo': dict(kind='vasc', c=dict(risk='Lesion anatomy, access', common=['Groin haematoma'], serious=['Iliac rupture', 'Stent occlusion', 'Distal embolism', 'Conversion to open'],
                                     specific=['Restenosis and repeat procedures'], alt=['Aortobifemoral bypass'], kenya=['Stent cost']),
                 icu=['Puncture sites; distal pulses', 'Back pain and falling Hb: retroperitoneal bleed'],
                 q1=('What should a patient know about kissing stents compared with bypass?', 'Less invasive, with a higher chance of restenosis and repeat procedures',
                     'Durability is lower than aortobifemoral bypass for extensive disease.', 'It never needs repeating', 'It is riskier than bypass', 'It needs a laparotomy'),
                 q2=('Two hours after iliac stenting: back pain, BP falling, Hb down. Next?', 'Suspect a retroperitoneal bleed or rupture: call, resuscitate, CT or angiography',
                     'Access or iliac injury bleeds into the retroperitoneum.', 'Analgesia and observe', 'Give furosemide', 'Discharge')),
}

KEY_OF = {'lul': 'lobe', 'lll': 'lobe', 'rul': 'lobe', 'rml': 'lobe', 'rll': 'lobe', 'pnl': 'pn', 'pnr': 'pn', 'seg-lingula': 'seg', 'seg-lul-updiv': 'seg', 'seg-s6': 'seg',
          'trachea': 'trachea', 'thymectomy': 'thymectomy', 'oesophagectomy': 'oesophagectomy', 'duct': 'duct', 'empyema': 'empyema', 'ppe': 'ppe',
          'rt': 'trauma', 'clamshell': 'trauma', 'cardio': 'trauma', 'tract': 'trauma', 'hilar': 'trauma',
          'mvr': 'mvr', 'avr': 'avr', 'root': 'root', 'tricuspid': 'tricuspid', 'cabg': 'cabg'}
APPR_OF = {'aaa-infra': 'aaa-open', 'aaa-juxta': 'aaa-open', 'aaa-supra': 'aaa-open', 'aaa-evar': 'evar', 'taa-open': 'taa-open', 'taa-tevar': 'tevar', 'taa-asc': 'taa-asc',
           'aiod-abf': 'abf', 'aiod-axbf': 'axbf', 'aiod-endo': 'endo'}

OPS.update({
    'peri': dict(kind='card', c=dict(risk='EuroSCORE II; NYHA class (class IV carries the highest risk), nutrition, liver function', common=['Bleeding', 'AF', 'Chest infection'],
                                     serious=['Low cardiac output needing inotropes', 'Injury to a heart chamber or coronary artery', 'Phrenic nerve injury', 'Death (about 1 in 10 in African series)'],
                                     specific=['Symptoms may take weeks to improve as the heart muscle recovers', 'Bypass may be needed if a chamber tears'],
                                     alt=['Continued anti-TB therapy and diuretics (when constriction may still resolve)'], kenya=['Complete anti-TB therapy and continue antiretrovirals', 'Blood availability']),
                 icu=['Low output from an atrophied myocardium: inotropes, cautious filling, diuresis; avoid overload', 'Bleeding from raw epicardium', 'Continue anti-TB and antiretroviral therapy; send pericardium for histology and TB culture'],
                 q1=('What should a patient expect after pericardiectomy for long-standing constriction?', 'Improvement over weeks, with a risk of low cardiac output early on',
                     'The myocardium has atrophied under the shell and needs time and support.', 'Instant cure on the table', 'No risk of bleeding', 'Lifelong warfarin'),
                 q2=('Day 1 after pericardiectomy: low output, CVP 14, a dilated thin RV. Next?', 'Inotropes and diuresis, avoid more fluid',
                     'The freed but atrophied ventricle dilates with volume; support contractility rather than fill.', 'Fluid boluses', 'Re-open for residual constriction', 'Beta-blocker'),
                 ev='Africa meta-analysis 2026: low cardiac output the commonest cause of perioperative death.'),
    'ali': dict(kind='vasc', c=dict(risk='Cardiac state (AF, recent MI), renal function, time since onset', common=['Groin wound problems', 'Swelling of the leg'],
                                    serious=['Amputation despite surgery', 'Death', 'Reperfusion injury: high potassium, kidney injury', 'Compartment syndrome needing fasciotomy'],
                                    specific=['Fasciotomy wounds left open, closed or grafted later', 'Lifelong anticoagulation if embolic from AF'], alt=['Catheter-directed thrombolysis (IIa, no contraindication)', 'Primary amputation if irreversible'],
                                    kenya=['INR monitoring access for warfarin', 'HIV testing in the young']),
                icu=['Hourly foot Doppler and compartment checks (pain on passive stretch, tense calf)', 'Reperfusion: K⁺, CK, myoglobinuria; urine output; alkalinise urine per unit', 'Heparin infusion, then anticoagulation for AF; echo for the source'],
                q1=('Which risk is specific to revascularising a limb ischaemic for many hours?', 'Reperfusion injury: compartment syndrome, high potassium and kidney injury',
                    'Restoring flow to dead or injured muscle releases potassium and myoglobin.', 'Paraplegia', 'Stroke', 'Chylothorax'),
                q2=('Six hours after embolectomy: severe calf pain on passive toe movement, tense calf. Next?', 'Four-compartment fasciotomy now',
                    'Clinical compartment syndrome needs decompression without waiting for pressure readings.', 'More analgesia', 'Elevate the leg high', 'Repeat embolectomy')),
    'fempop': dict(kind='vasc', c=dict(risk='Cardiac and renal risk, frailty, WIfI stage', common=['Wound problems along the vein harvest', 'Leg swelling'],
                                       serious=['Graft occlusion', 'Amputation', 'MI', 'Death', 'Graft infection'], specific=['Duplex surveillance of the graft', 'Further procedures to keep the graft open'],
                                       alt=['Angioplasty or stenting (BASIL-2, anatomy permitting)', 'Medical therapy alone', 'Primary amputation'], kenya=['Access to duplex surveillance', 'Wound care at home']),
                   icu=['Graft pulse and pedal Doppler hourly at first', 'Leg wound and swelling (reperfusion oedema)', 'Antiplatelet and statin; glucose control'],
                   q1=('Which commitment comes with a vein bypass?', 'Graft surveillance with duplex scans and possible further procedures',
                       'Vein graft stenoses are found and fixed before occlusion.', 'None after discharge', 'Warfarin for life in all', 'Weekly angiograms'),
                   q2=('Day 1 after fem-pop bypass: the graft pulse is lost and the foot is cold. Next?', 'Call the consultant: early graft thrombosis needs return to theatre',
                       'Early failure is usually technical: inflow, twist, or the distal anastomosis.', 'Elevate and observe', 'Aspirin only', 'Wait for the morning duplex')),
    'amp': dict(kind='vasc', c=dict(risk='Cardiac and renal risk; sepsis', common=['Wound healing problems', 'Phantom sensations and pain'],
                                    serious=['Revision to a higher level', 'Death', 'Loss of the other leg later'], specific=['Rehabilitation and a prosthesis over months', 'Staged operation in sepsis'],
                                    alt=['Revascularisation if the limb is salvageable', 'Palliative care'], kenya=['Access to and cost of a prosthesis', 'Home adaptations']),
                icu=['Sepsis control: glucose, antibiotics, fluids', 'Stump wound; knee kept straight (prevent flexion contracture)', 'Phantom pain: early, multimodal analgesia'],
                q1=('What should a patient know before a below-knee amputation?', 'Healing may need revision; walking with a prosthesis takes months of rehabilitation',
                    'Realistic expectations and the risk to the other leg are part of consent.', 'Walking normally in a week', 'No pain afterwards', 'The other leg is never at risk'),
                q2=('After BKA the patient lies with the knee bent on a pillow. Why correct it?', 'A knee flexion contracture prevents prosthetic walking',
                    'Keep the knee straight and start physiotherapy early.', 'It causes phantom pain', 'It raises blood pressure', 'It is harmless')),
    'avf': dict(kind='vasc', c=dict(risk='Vessel quality, diabetes, age, previous lines', common=['Failure to mature', 'Bruising, swelling'],
                                    serious=['Steal syndrome (hand ischaemia)', 'Thrombosis', 'Infection', 'Nerve injury', 'High-output heart failure (rare, large upper-arm fistulas)'],
                                    specific=['Weeks before it can be used; further procedures to help maturation'], alt=['Graft', 'Tunnelled catheter', 'Peritoneal dialysis', 'Conservative kidney care'],
                                    kenya=['Protect the arm: no cannulas or blood pressure cuffs', 'Dialysis access and cost']),
                icu=['Check the thrill and bruit every few hours', 'Hand: colour, warmth, movement, sensation (steal)', 'No BP cuff, cannula or blood tests on the fistula arm'],
                q1=('Which serious early complication must be explained before a brachial fistula?', 'Steal syndrome: hand ischaemia needing urgent correction',
                    'Commoner with brachial inflow, diabetes and age.', 'Paraplegia', 'Stroke', 'Chylothorax'),
                q2=('Four hours after a radiocephalic fistula the thrill has gone. Next?', 'Call the surgeon: early thrombosis needs urgent exploration',
                    'Early loss of thrill is usually technical and salvageable if acted on quickly.', 'Wait 6 weeks', 'Massage the arm', 'Start dialysis through it')),
    'bx': dict(kind='thx', c=dict(risk='ppoFEV1, ppoDLCO; nutrition; sputum TB status', common=['Air leak', 'Bleeding (adhesions, collaterals)', 'Chest infection', 'Chronic chest wall pain'],
                                  serious=['Bronchopleural fistula and empyema', 'Major bleeding', 'Death'], specific=['Open surgery likely (adhesions); conversion if VATS', 'Symptoms may persist if disease remains elsewhere'],
                                  alt=['Medical therapy and airway clearance', 'Bronchial artery embolisation for haemoptysis'], kenya=['Exclude and treat active TB; HIV testing']),
               icu=['Bleeding from adhesion beds: drain output closely', 'Air leak and residual space', 'Physiotherapy; culture-directed antibiotics', 'Stump: BPF risk in an infected field'],
               q1=('Which risk is higher after resection for post-TB bronchiectasis than after lobectomy for a small cancer?', 'Bleeding from adhesions and collaterals, and bronchopleural fistula',
                   'Inflamed, adherent fields bleed and heal poorly.', 'Lower risk overall', 'Phrenic injury only', 'None'),
               q2=('Day 1 after lobectomy for bronchiectasis: 250 mL/h of blood for 3 hours. Next?', 'Call the consultant: re-exploration for bleeding',
                   'Adhesion beds and bronchial collaterals bleed; sustained output above the thresholds needs theatre.', 'Strip the drain', 'Clamp the drain', 'Give furosemide')),
})
KEY_OF.update({'pericardium': 'peri', 'ali': 'ali', 'avf': 'avf', 'bronchiectasis': 'bx'})
APPR_OF.update({'fp-gsv': 'fempop', 'amp-levels': 'amp'})

CORE_LINK = {'card': ('cticu-cardiac', 'cardiac core'), 'thx': ('cticu-thoracic', 'thoracic core'), 'vasc': ('cticu-vascular', 'vascular core')}
TEACH = ('Pathophysiology', 'Anatomy', 'Case', 'Decision')


def consent_body(o):
    c = o['c']
    if o.get('emergency'):
        return ('<p><b>Life-saving emergency surgery</b>: when the patient cannot consent and delay risks life, proceed on necessity, document why, and inform the next of kin as soon as possible. '
                'If the patient can talk, a short explanation still matters.</p>' + h4('Discuss with the family afterwards')
                + ul('Survival after resuscitative thoracotomy is low overall; best for penetrating cardiac injury with signs of life', 'Re-operation, bleeding, infection', 'Missed injuries', 'Long ICU stay')
                + '<p>The seven-part template: ' + link('cticu-consent', 'CTICU protocol, consent') + '.</p>')
    b = '<p><b>1. The patient\'s own risk</b>: ' + c['risk'] + '. Quote the figure, not a textbook average.</p>'
    if c['common']: b += h4('2. Common') + ul(*c['common'])
    if c['serious']: b += h4('3. Serious') + ul(*c['serious'])
    if c['specific']: b += h4('4. Specific to this operation') + ul(*c['specific'])
    if c['alt']: b += h4('5. Alternatives') + ul(*c['alt'], 'No operation, and what that means')
    b += h4('6. Recovery') + ul('ICU, then ward; drains; walking from day 1', 'Return to normal activity over weeks', 'Follow-up in clinic')
    if c['kenya']: b += h4('7. Kenya-specific') + ul(*c['kenya'])
    b += '<p>Template and how to use it: ' + link('cticu-consent', 'CTICU protocol, consent') + '.</p>'
    return b


def icu_body(o):
    appr, name = CORE_LINK[o['kind']]
    b = f'<p>Start with the {link(appr, name)}, the {link("cticu-core", "lab schedule")} and the {link("cticu-core", "escalation table", 2)}; then this operation\'s own points.</p>'
    b += h4('Specific to this operation') + ul(*o['icu'])
    b += h4('Labs') + ul('Day 1 morning: FBC, UEC, LFTs, Ca, Mg, PO₄' + KNH) + '<p>Doses: ' + link('cticu-doses', 'electrolytes') + ', ' + link('cticu-doses', 'vasoactive drugs', 1) + '.</p>'
    if o.get('ev'): b += ev(o['ev'])
    return b


def apply(procs, ask, has):
    """insert Consent and ICU into every operation; add the hub"""
    import copy
    for p in procs.values(): p['steps'] = copy.deepcopy(p['steps']); p['sequence'] = copy.deepcopy(p.get('sequence', []))   # steps are shared between approaches
    for key, p in procs.items():
        if p.get('group') == 'Access and positioning' or p['op'] == 'cticu': continue
        k = APPR_OF.get(key) or KEY_OF.get(p['op'])
        if not k: continue
        o = OPS[k]; steps = p['steps']
        i = 0
        while i < len(steps) and steps[i]['phase'] in TEACH: i += 1
        ref = steps[max(i - 1, 0)]; last = steps[-1]
        seq_ins = ref.get('seq', 0) + 1
        for s in steps:
            if s.get('seq', 0) >= seq_ins: s['seq'] = s['seq'] + 1
        seq = p.setdefault('sequence', [])
        seq.insert(min(seq_ins, len(seq)), {'label': 'Consent', 'kind': 'other'})
        pre = key
        view_of = lambda s: {kk: (list(s[kk]) if isinstance(s.get(kk), list) else dict(s[kk]) if isinstance(s.get(kk), dict) else s[kk]) for kk in ('view', 'show', 'hide', 'opacity', 'ct', 'pose') if kk in s}
        cons = {'id': f'{pre}-consent', 'phase': 'Consent', 'seq': seq_ins, 'title': 'Consent: what to discuss with this patient', 'body': consent_body(o), 'askAfter': True,
                'ask': ask(*o['q1']), **view_of(ref), 'labels': [], 'highlight': [], 'danger': []}
        steps.insert(i, cons)
        seq.append({'label': 'ICU', 'kind': 'other'})
        icu = {'id': f'{pre}-icu', 'phase': 'ICU', 'seq': len(seq) - 1, 'title': 'ICU and post-operative care', 'body': icu_body(o), 'askAfter': True,
               'ask': ask(*o['q2']), **view_of(last), 'labels': list(last.get('labels', [])), 'highlight': [], 'danger': []}
        steps.append(icu)
    # ------------------------------------------------------------------ the hub
    def borrow(key, idx=0):
        p = procs.get(key)
        if not p: return {'view': {'frame': ['heart'], 'dir': [0, 1, 0.3]}}
        s = p['steps'][idx]
        return {kk: s[kk] for kk in ('view', 'show', 'hide', 'opacity', 'ct', 'pose') if kk in s}
    CV = borrow('mvr-std', 2); TV = borrow('lul-anterior', 1); VV = borrow('aaa-infra', 2)
    def hub(id_, approach, steps_, vv):
        out = []
        for n, (phase, title, body, q) in enumerate(steps_):
            st = {'id': f'{id_}-{n}', 'phase': phase, 'seq': n, 'title': title, 'body': body, 'askAfter': True, **vv, 'labels': [], 'highlight': [], 'danger': []}
            if q: st['ask'] = ask(*q)
            out.append(st)
        procs[id_] = {'id': id_, 'op': 'cticu', 'opName': 'CTICU protocol (KNH)', 'side': 'both', 'name': 'CTICU protocol', 'approach': approach, 'group': 'CTICU protocol',
                      'summary': 'The unit protocol: labs, rounds, escalation, cardiac, thoracic and vascular cores, doses, consent and governance.', 'ports': [],
                      'steps': out, 'sequence': [{'label': s_[0], 'kind': 'other'} for s_ in steps_], 'sources': HUB_SRC}
    hub('cticu-core', 'Labs, rounds, escalation', [
        ('Labs', 'Lab schedule: all operations', '<p>One schedule for every cardiac, thoracic and vascular operation. Every row is unit practice.</p>' + LABS, None),
        ('Round', 'The daily ICU round: by system', '<p>Present every patient in this order on the ward round.</p>' + ROUND, None),
        ('Escalate', 'Call the consultant', ESCALATE, ('Two hours after surgery the drains stop suddenly, CVP rises, BP and urine fall. Diagnosis?', 'Tamponade',
                                                       'Clot in the drains with falling output and rising filling pressures: echo and prepare to re-open.', 'Hypovolaemia', 'Vasoplegia', 'Normal recovery'))], CV)
    hub('cticu-cardiac', 'Cardiac core', [
        ('0–6 h', 'Arrival, haemodynamics, bleeding, ventilation', CARD_1, ('When is re-exploration indicated for bleeding?', 'More than 400 mL in the first hour, or sustained high output, or tamponade',
                                                                             'Correct coagulopathy and temperature, but these thresholds call for the surgeon.', 'Any drainage at all', 'Only when Hb is under 5', 'Never in the first 24 h')),
        ('6–24 h', 'Rhythm, kidneys, glucose, analgesia, day 1', CARD_2, ('Target K⁺ after cardiac surgery?', '4.0–5.0 mmol/L',
                                                                          'Low K⁺ and Mg²⁺ provoke AF and ventricular arrhythmias.', '3.0–3.5 mmol/L', '5.5–6.0 mmol/L', 'Any value')),
        ('Valves', 'Anticoagulation after valve surgery', CARD_3, ('A mechanical mitral valve in a patient with AF: INR target (medium-thrombogenicity bileaflet)?', '3.5',
                                                                    'ESC/EACTS 2021: medium thrombogenicity 3.0, plus 0.5 with a risk factor such as mitral position or AF.', '2.0', '2.5', '4.5'))], CV)
    hub('cticu-thoracic', 'Thoracic core', [
        ('0–24 h', 'Arrival, analgesia, lungs, fluids', THX_1, ('Why is regional analgesia central after thoracotomy?', 'Pain prevents coughing, leading to sputum retention and pneumonia',
                                                                 'Analgesia is lung function.', 'It shortens the operation', 'It prevents air leak', 'It replaces physiotherapy')),
        ('Drains', 'Drains, air leak, pneumonectomy', THX_2, ('When can a chest drain come out after lobectomy?', 'No air leak and non-chylous output up to 450 mL/24 h',
                                                               'ERAS lung 2019.', 'Only when output is zero', 'After 7 days routinely', 'When the patient asks'))], TV)
    hub('cticu-vascular', 'Vascular core', [
        ('Open aorta', 'Open aortic surgery: Hct, girth, compartment syndrome', VASC_1, ('Why can oliguria after AAA repair be dangerous to treat with fluid alone?', 'It may be abdominal compartment syndrome, which fluid worsens',
                                                                                        'Measure bladder pressure when girth rises or oliguria persists despite filling.', 'Fluid is always right', 'Oliguria is normal', 'It is always AKI')),
        ('Cord, grafts', 'Spinal cord, bypass grafts, endovascular', VASC_2, None)], VV)
    hub('cticu-doses', 'Doses (Medscape)', [
        ('Electrolytes', 'Electrolytes: K⁺, Mg²⁺, Ca²⁺, PO₄', DOSES_E, ('Maximum KCl rate for K⁺ 2.8 mmol/L without ECG changes?', '10 mmol/h, at no more than 40 mmol/L',
                                                                     'The label limit for K⁺ 2.5 or more; up to 40 mmol/h only with ECG monitoring for severe hypokalaemia with ECG changes.', '40 mmol/h peripherally', 'A 20 mmol push', '100 mmol/h')),
        ('Vasoactive', 'Noradrenaline and adrenaline infusions', DOSES_V, ('Noradrenaline 4 mg in 50 mL: what rate gives 8 µg/min?', '6 mL/h',
                                                                           '80 µg/mL: mL/h = µg/min × 0.75.', '8 mL/h', '0.6 mL/h', '60 mL/h'))], CV)
    hub('cticu-consent', 'Consent and governance', [
        ('Consent', 'Consent: seven parts for every operation', CONSENT_T, ('What makes a risk material?', 'It is one this patient would want to know, not only a frequent one',
                                                                            'Serious rare risks (paraplegia, stroke) are material even when uncommon.', 'Only risks above 10%', 'Only risks the surgeon worries about', 'None if the form is signed')),
        ('Governance', 'Governance, open decisions and audit', GOV, None)], CV)


HUB_SRC = [
    {'title': 'Engelman DT, et al. Guidelines for perioperative care in cardiac surgery: ERAS Society recommendations. JAMA Surg 2019;154:755-66', 'url': 'https://jamanetwork.com/journals/jamasurgery/fullarticle/2732511'},
    {'title': 'Batchelor TJP, et al. Guidelines for enhanced recovery after lung surgery (ERAS Society and ESTS). Eur J Cardiothorac Surg 2019;55:91-115', 'url': 'https://academic.oup.com/ejcts/article/55/1/91/5124324'},
    {'title': 'Low DE, et al. Guidelines for perioperative care in esophagectomy: ERAS Society recommendations. World J Surg 2019;43:299-330', 'url': 'https://link.springer.com/article/10.1007/s00268-018-4786-4'},
    {'title': 'Aiolfi A, et al. Use of C-reactive protein for the early prediction of anastomotic leak after esophagectomy: systematic review and Bayesian meta-analysis. PLoS One 2018', 'url': 'https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0209272'},
    {'title': 'Connolly SJ, et al. Rivaroxaban in rheumatic heart disease-associated atrial fibrillation (INVICTUS). N Engl J Med 2022;387:978-88', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa2209051'},
    {'title': 'Kirkpatrick AW, et al. Is clinical examination an accurate indicator of raised intra-abdominal pressure? Can J Surg 2000;43:207-11', 'url': 'https://www.canjsurg.ca/content/43/3/207'},
    {'title': 'World Society of the Abdominal Compartment Syndrome: 2013 consensus definitions and guidelines', 'url': 'https://www.wsacs.org/wp-content/uploads/2021/04/2013-Guidelines-slide-set.pdf'},
    {'title': 'Abdominal compartment syndrome after endovascular repair of ruptured AAA: systematic review and meta-analysis. J Vasc Surg', 'url': 'https://www.sciencedirect.com/science/article/pii/S074152141302212X'},
    {'title': 'Medscape Reference: calcium gluconate', 'url': 'https://reference.medscape.com/drug/calcium-gluconate-344434'},
    {'title': 'Medscape Reference: norepinephrine', 'url': 'https://reference.medscape.com/drug/levarterenol-levophed-norepinephrine-342443'},
    {'title': 'Medscape Reference: epinephrine', 'url': 'https://reference.medscape.com/drug/epipen-jr-epinephrine-342437'},
    {'title': 'Medscape Reference: sodium phosphates IV', 'url': 'https://reference.medscape.com/drug/sodium-phosphates-iv-999713'},
    {'title': 'Potassium chloride dosing (product label)', 'url': 'https://www.drugs.com/dosage/potassium-chloride.html'},
    {'title': 'Magnesium sulfate prescribing information', 'url': 'https://www.drugs.com/pro/magnesium-sulfate.html'},
]
