"""Five modules chosen for their volume at KNH: pericardiectomy for tuberculous constriction, embolectomy and fasciotomy for
acute limb ischaemia, arteriovenous fistulas for dialysis, femoropopliteal bypass and amputation levels, and surgery for
post-tuberculous bronchiectasis. Each follows the atlas pattern (pathophysiology, anatomy, case, decision, operation);
consent and ICU steps are added for every operation by postop.py.
"""
from __future__ import annotations

import copy

import numpy as np

V = lambda *a: np.array(a[0] if len(a) == 1 else a, float)
R = lambda v: [round(float(x), 1) for x in v]
ev = lambda t: f'<p class="evidence"><b>Evidence:</b> {t}</p>'
ul = lambda *xs: '<ul>' + ''.join(f'<li>{x}</li>' for x in xs) + '</ul>'
h4 = lambda t: f'<h4>{t}</h4>'
chain = lambda *xs: '<div class="chain">' + '<i>→</i>'.join(f'<span class="hot">{x[1:]}</span>' if x.startswith('!') else f'<span>{x}</span>' for x in xs) + '</div>'


def add(procs, ask, has, LM, S):
    P = lambda k: V(LM[k]) if k in LM else V(S[k]['centroid'])
    hv = lambda xs: [i for i in xs if has(i)]
    DEFAULT_ON = [i for i, s in S.items() if s.get('visible', True) is not False]

    def view(t, d, dist):
        d = V(d); d = d / np.linalg.norm(d); t = V(t); return {'eye': R(t + d * dist), 'target': R(t)}

    def step(id_, phase, title, body, v, show=(), highlight=(), danger=(), labels=(), opacity=None, action=None, quiz=None, lead=None, after=False, ct=None, spin=False, hide_extra=()):
        show = hv(show); named = set(show) | set(highlight) | set(danger) | set(labels)
        s = {'id': id_, 'phase': phase, 'title': title, 'body': body, 'view': v, 'show': show,
             'hide': [i for i in DEFAULT_ON if i not in named] + hv(hide_extra), 'highlight': hv(highlight), 'danger': hv(danger), 'labels': hv(labels), 'opacity': opacity or {}}
        if ct is not None: s['ct'] = {'focus': R(ct), 'plane': 'axial', 'window': 'mediastinum'}
        if action: s['action'] = action
        if quiz: s['ask'] = quiz
        if lead: s['lead'] = lead
        if after: s['askAfter'] = True
        if spin: s['spin'] = True
        return s

    def proc(key, op, opName, approach, summary, groups, sources, group, side='both'):
        steps, sq = [], []
        for i, (lab, kind, sts) in enumerate(groups):
            sq.append({'label': lab, 'kind': kind})
            for s in sts: steps.append({**s, 'seq': i})
        procs[key] = {'id': key, 'op': op, 'opName': opName, 'side': side, 'name': opName, 'approach': approach, 'summary': summary,
                      'ports': [], 'steps': steps, 'sources': sources, 'group': group, 'sequence': sq}

    # =================================================================================================== pericardiectomy
    if has('peri-thick'):
        HC = P('heart-c') if 'heart-c' in LM else P('heart'); FR = (0.15, 1, 0.25)
        HEART = hv(['heart', 'ra', 'rv', 'la', 'lv', 'svc', 'ivc', 'aorta', 'n-phrenic', 'n-phrenic-r'])
        PERI = ['peri-thick', 'peri-calcium']
        OP = {'peri-thick': 0.8, 'heart': 0.9}
        patho = step('pc-patho', 'Pathophysiology', 'Pathophysiology: tuberculous constrictive pericarditis',
            '<p>In Kenya and across sub-Saharan Africa most constrictive pericarditis follows <b>tuberculous pericarditis</b>, often with HIV. The pericardium thickens and fibroses, then often calcifies, into a rigid shell.</p>'
            + chain('TB pericardial effusion', 'Fibrinous, then fibrotic pericardium', 'Rigid shell (± calcium)', '!All four chambers cannot fill in late diastole')
            + '<p><b>Physiology</b>: early diastolic filling is rapid, then stops abruptly when the heart meets the shell (the <b>dip-and-plateau</b> or "square root" sign). Diastolic pressures equalise in all chambers. Filling of the two ventricles is <b>interdependent</b>: on inspiration the septum shifts left. The JVP <b>rises on inspiration (Kussmaul\'s sign)</b>; there is a pericardial knock, ascites and hepatomegaly out of proportion to oedema, and a small, quiet heart.</p>'
            '<p><b>Diagnosis</b>: echocardiography (respiratory septal shift, preserved or raised medial e\', hepatic vein expiratory flow reversal; the Mayo criteria); CT for thickness and calcium; catheter pressures when unclear. <b>Restrictive cardiomyopathy</b> is the main differential.</p>'
            '<p><b>Prevention</b>: anti-tuberculous therapy for all; adjunctive prednisolone reduced progression to constriction in IMPI (4.4% vs 7.8%) but raised HIV-associated cancers, so it is not given routinely to people living with HIV.</p>'
            + ev('IMPI (Mayosi et al., NEJM 2014): prednisolone did not reduce the combined outcome (23.8% vs 24.5%) but reduced constriction (4.4% vs 7.8%) and hospitalisation; cancers 1.05 vs 0.32 per 100 person-years. Mayo echo criteria (Welch et al., Circ Cardiovasc Imaging 2014). ESC 2015 pericardial guideline (Adler et al., Eur Heart J 2015).'),
            view(HC, FR, 360), show=[*HEART, *PERI], highlight=['peri-thick'], labels=['peri-thick', 'peri-calcium', 'n-phrenic'], opacity=OP, ct=HC, after=True, spin=True,
            quiz=ask('Which bedside sign points to constriction rather than tamponade?', 'The JVP rises on inspiration (Kussmaul\'s sign)',
                     'In constriction the rigid shell stops the right heart accepting the extra inspiratory venous return; pulsus paradoxus is the hallmark of tamponade.', 'Pulsus paradoxus alone', 'A large heart on chest X-ray', 'Bilateral basal crackles'))
        anat = step('pc-anatomy', 'Anatomy', 'The pericardium, the phrenic nerves and what to free',
            '<p>The <b>fibrous pericardium</b> is fused to the diaphragm below and the great vessels above. The <b>phrenic nerves</b> run on its lateral surfaces, in front of the hila, with the pericardiophrenic vessels: they mark the lateral limits of a <b>total (radical) pericardiectomy</b>, which frees the anterior surface <b>from phrenic nerve to phrenic nerve</b>, the <b>diaphragmatic surface</b>, and the bands around the <b>cavae</b>.</p>'
            '<p>The <b>coronary arteries</b> run under the epicardium in the atrioventricular and interventricular grooves, where calcium is often densest: peel around them, not through them. The right atrium and the cavae are thin-walled and tear easily.</p>',
            view(HC, (0.1, 1, 0.35), 330), show=[*HEART, *PERI], highlight=['n-phrenic', 'n-phrenic-r'], danger=hv(['n-phrenic', 'n-phrenic-r', 'ra']), labels=['n-phrenic', 'n-phrenic-r', 'peri-thick', 'ra', 'lv'], opacity={**OP, 'peri-thick': 0.55}, ct=HC, spin=True)
        case = step('pc-case', 'Case', 'Case: constriction after 8 weeks of anti-TB treatment',
            '<p><b>Constrictive pericarditis</b> persisting after 8 weeks of anti-tuberculous therapy: refer for <b>pericardiectomy</b>. She is NYHA class III; with class IV, cachexia and liver dysfunction, risk rises sharply, so do not wait for her to deteriorate.</p>'
            + ev('Africa meta-analysis (Oluwajuyigbe et al., The Cardiothoracic Surgeon 2026; 11 studies, 604 patients): perioperative mortality 10.0% (East Africa 12.9%); low cardiac output the commonest cause of death; NYHA IV the most consistent predictor.'),
            view(HC, FR, 380), show=[*HEART, *PERI], highlight=['peri-thick'], labels=['peri-thick', 'peri-calcium'], opacity=OP, ct=HC,
            lead='<p>A <b>28-year-old woman</b>, HIV-positive on antiretrovirals, treated for tuberculous pericardial effusion 3 months ago. Now: abdominal swelling, breathlessness on walking 100 m. <b>JVP 12 cm, rising on inspiration</b>, ascites, hepatomegaly, mild ankle oedema. Echo: septal bounce, respiratory variation in mitral inflow, dilated IVC. CT: pericardium 6 mm with calcium over the AV grooves.</p>',
            quiz=ask('She has completed 8 weeks of anti-TB therapy and is not improving. What next?', 'Pericardiectomy, before she reaches NYHA class IV',
                     'Pericardiectomy is indicated when constriction persists or worsens after several weeks of anti-TB therapy; advanced functional class is the strongest predictor of death.', 'Wait 6 more months', 'Pericardiocentesis', 'Diuretics alone indefinitely'))
        dec = step('pc-decision', 'Decision', 'Timing, extent and approach',
            '<table class="mini"><tr><th>Question</th><th>Answer</th></tr>'
            '<tr><td>When?</td><td>Constriction that persists or worsens after about 4–8 weeks of anti-TB therapy; early, before NYHA IV, cachexia and liver failure</td></tr>'
            '<tr><td>How much?</td><td><b>Total</b>: phrenic to phrenic, the diaphragmatic surface, the cavae. Better long-term results than partial (anterior) resection</td></tr>'
            '<tr><td>Approach?</td><td><b>Median sternotomy</b> (access to both sides and the cavae, bypass if needed); a left anterolateral thoracotomy is an alternative when the disease is left-sided or for a redo</td></tr>'
            '<tr><td>Bypass?</td><td>Off-pump; <b>bypass on standby</b> (groin prepped) for bleeding or a torn chamber</td></tr></table>'
            '<p><b>Before surgery</b>: optimise with diuretics, drain large ascites or effusions, correct albumin and nutrition, continue anti-TB and antiretroviral therapy; cross-match.</p>'
            + ev('ESC 2015 pericardial guideline: pericardiectomy for persistent constriction despite anti-TB therapy (expert timing 4–8 weeks). Total versus partial pericardiectomy: better long-term survival after total (reviews, e.g. IntechOpen 2023 "Constrictive pericarditis: surgical management").'),
            view(HC, FR, 380), show=[*HEART, *PERI], labels=['peri-thick'], opacity=OP, ct=HC)
        stern = step('pc-sternotomy', 'Access', 'Median sternotomy; groin prepped for bypass',
            '<p>Median sternotomy with the groins prepped and draped: femoral cannulation is the fastest route to bypass if a chamber tears. Heparin is not given unless bypass is needed. A retractor opened gently: the stiff heart tolerates traction poorly.</p>',
            view(HC, (0.05, 1, 0.3), 400), show=['sternum', *HEART, *PERI], highlight=['sternum'], labels=['sternum', 'peri-thick'], opacity={**OP, 'sternum': 0.6}, ct=HC)
        plane = step('pc-plane', 'Plane', 'Find the plane over the outflow tract; free the left ventricle first',
            '<p>Incise the thick pericardium over the aorta and right ventricular outflow until the <b>epicardium bulges</b> through: that is the plane. Develop it with scissors and blunt dissection, a sponge on a stick, and sharp division of bands.</p>'
            '<p><b>Free the left ventricle before the right</b>: releasing the right ventricle first lets it fill and pump into a still-constrained left ventricle and lungs, causing <b>pulmonary oedema</b>. Work from the left phrenic nerve, over the apex, down to the diaphragm, keeping a strip of pericardium on the nerve.</p>',
            view(HC + V(-30, 0, -10), (-0.35, 1, 0.3), 300), show=[*HEART, *PERI], highlight=['lv'], danger=hv(['n-phrenic']), labels=['lv', 'n-phrenic', 'peri-thick'], opacity={**OP, 'peri-thick': 0.6}, ct=HC,
            quiz=ask('Why is the left ventricle freed before the right?', 'Releasing the right side first can flood a still-constrained left ventricle and cause pulmonary oedema',
                     'The traditional order protects the lungs: the left side is decompressed first.', 'The left side is easier', 'To avoid the phrenic nerve', 'It shortens the operation'))
        peel = step('pc-peel', 'Peel', 'Right ventricle, right atrium and the cavae; calcium and the coronaries',
            '<p>Then the <b>right ventricle</b>, the <b>right atrium</b> (thin: take care) and the bands around the <b>inferior and superior venae cavae</b>, as far as the right phrenic nerve. Where calcium is welded to the myocardium, leave an island rather than tear the heart. A thick, adherent <b>epicardial peel</b> can be scored in a grid (<b>waffle</b> procedure) to let the muscle expand.</p>'
            '<p>Do not chase the plane over a coronary artery in the atrioventricular groove.</p>',
            view(HC + V(25, 0, -10), (0.45, 1, 0.25), 300), show=[*HEART, *PERI], highlight=hv(['rv', 'ra']), danger=hv(['ra', 'n-phrenic-r', 'ivc']), labels=['rv', 'ra', 'n-phrenic-r', 'peri-calcium'], opacity={**OP, 'peri-thick': 0.6}, ct=HC,
            action={'kind': 'decorticate', 'label': 'Remove the pericardium, phrenic to phrenic', 'port': 'sternotomy', 'ids': PERI, 'show': ['peri-left'],
                    'expand': {'ids': hv(['heart', 'ra', 'rv', 'la', 'lv']), 'pivot': R(HC), 'from': 0.94}})
        after_ = step('pc-after', 'After', 'The heart fills: check, haemostasis, close',
            '<p>The freed ventricles visibly expand; the CVP falls. Haemostasis over the raw epicardium (warm packs, topical agents, cautery away from the coronaries). Drains to both pleural spaces if opened. Send the pericardium for <b>histology and TB culture</b> (GeneXpert).</p>'
            '<p><b>Expect low cardiac output</b> in the first days: the myocardium has been atrophied by months of constriction. Inotropes, careful filling and diuresis; continue anti-TB therapy.</p>',
            view(HC, FR, 360), show=[*HEART, 'peri-left'], labels=['peri-left', 'n-phrenic', 'n-phrenic-r'], opacity={'peri-left': 0.7}, ct=HC,
            quiz=ask('After pericardiectomy, the CVP is 8 but the cardiac output is low and the ventricle looks thin and hypokinetic. Most likely?', 'Myocardial atrophy from long-standing constriction: inotropic support and cautious filling',
                     'Low output syndrome is the commonest cause of death after pericardiectomy in African series.', 'Residual constriction: re-operate now', 'Hypovolaemia: give 2 L fluid', 'Tamponade'))
        proc('peri-tb', 'pericardium', 'Pericardiectomy', 'TB constrictive pericarditis (sternotomy)',
             'Tuberculous constriction: physiology, timing after anti-TB therapy, total pericardiectomy from phrenic to phrenic, left ventricle first.',
             [('Patho', 'other', [patho]), ('Anatomy', 'other', [anat]), ('Case', 'other', [case]), ('Decide', 'other', [dec]), ('Sternotomy', 'other', [stern]),
              ('LV first', 'artery', [plane]), ('RV, RA, cavae', 'vein', [peel]), ('After', 'other', [after_])], PERI_SRC, 'Cardiac')

    # =================================================================================================== the leg
    if has('leg-sfa'):
        LEG = hv(['leg-skin', 'leg-femur', 'leg-tibia', 'leg-fibula', 'leg-foot', 'cfa-r', 'eia-r', 'pfa-r', 'leg-sfa', 'leg-pop', 'leg-ata', 'leg-tpt', 'leg-pta', 'leg-per', 'leg-dpa', 'fv-r', 'inguinal-lig-r'])
        LOP = {'leg-skin': 0.1, 'leg-foot': 0.15, 'leg-femur': 0.3, 'leg-tibia': 0.4, 'leg-fibula': 0.4, 'fv-r': 0.5}
        G, K, A_ = P('cfa-r'), P('knee-r'), P('ankle-r')
        WHOLE = view(K + V(0, 0, 60), (0.25, 1, 0.1), 1550)
        THIGH = view(V(G[0], G[1], (G[2] + K[2]) / 2 - 40), (0.3, 1, 0.12), 950)
        # ------------------------------------------------------------ acute limb ischaemia
        ali_patho = step('ali-patho', 'Pathophysiology', 'Pathophysiology: acute limb ischaemia',
            '<p>A sudden loss of limb perfusion threatening viability, presenting within 2 weeks. <b>Embolus</b> (from the heart: AF, mitral stenosis, a mural thrombus after MI, endocarditis) lodges at a bifurcation, most often the <b>common femoral</b>; <b>thrombosis in situ</b> occludes a diseased artery or bypass graft, and the limb often has collaterals. In Kenya add <b>rheumatic AF</b> and <b>HIV-associated arterial thrombosis</b> in the young.</p>'
            + chain('Occlusion', 'Muscle and nerve ischaemia', '!Irreversible after about 6 hours of severe ischaemia', 'Reperfusion: swelling, compartment syndrome, K⁺ and myoglobin release')
            + '<p><b>The 6 Ps</b>: pain, pallor, pulselessness, perishing cold, paraesthesia, paralysis. Sensory loss and weakness are the ones that decide urgency.</p>'
            '<table class="mini"><tr><th>Rutherford</th><th>Sensory loss</th><th>Weakness</th><th>Doppler (art / vein)</th><th>Action</th></tr>'
            '<tr><td>I viable</td><td>none</td><td>none</td><td>audible / audible</td><td>urgent imaging, revascularise</td></tr>'
            '<tr><td>IIa marginally threatened</td><td>toes or none</td><td>none</td><td>inaudible / audible</td><td>revascularise urgently</td></tr>'
            '<tr><td>IIb immediately threatened</td><td>beyond the toes, rest pain</td><td>mild to moderate</td><td>inaudible / audible</td><td><b>emergency</b> revascularisation</td></tr>'
            '<tr><td>III irreversible</td><td>profound, anaesthetic</td><td>paralysis, rigor</td><td>inaudible / inaudible</td><td>amputation</td></tr></table>'
            + ev('Rutherford et al. (J Vasc Surg 1997) categories, as adopted by the ESVS 2020 acute limb ischaemia guideline (Björck et al., Eur J Vasc Endovasc Surg 2020).'),
            THIGH, show=[*LEG, 'ali-embolus', 'ali-clot'], highlight=['ali-embolus'], labels=['ali-embolus', 'ali-clot', 'leg-sfa', 'pfa-r'], opacity=LOP, after=True,
            quiz=ask('Which finding moves acute limb ischaemia from Rutherford IIa to IIb?', 'Sensory loss beyond the toes, rest pain or muscle weakness',
                     'IIb is immediately threatened: emergency revascularisation.', 'An absent foot pulse', 'A cold foot', 'Pallor'))
        ali_anat = step('ali-anatomy', 'Anatomy', 'The femoral bifurcation to the trifurcation',
            '<p>The <b>common femoral artery</b> lies at the mid-inguinal point and divides 3–5 cm lower into the <b>superficial femoral</b> (down the adductor canal to the hiatus) and the <b>profunda femoris</b>. The <b>popliteal artery</b> lies deep in the popliteal fossa, then divides into the <b>anterior tibial</b> (through the interosseous membrane into the anterior compartment, becoming the dorsalis pedis) and the <b>tibioperoneal trunk</b> (posterior tibial, behind the medial malleolus, and peroneal).</p>'
            '<p>A <b>single transverse femoral arteriotomy</b> reaches all of these with a Fogarty catheter: proximally into the iliac, distally down the superficial femoral and profunda, and to the trifurcation.</p>',
            WHOLE, show=LEG, highlight=hv(['cfa-r', 'leg-sfa', 'pfa-r', 'leg-pop', 'leg-ata', 'leg-pta', 'leg-per']), labels=hv(['cfa-r', 'pfa-r', 'leg-sfa', 'leg-pop', 'leg-ata', 'leg-pta', 'leg-per', 'leg-dpa']), opacity=LOP, spin=True)
        ali_case = step('ali-case', 'Case', 'Case: a cold right leg in rheumatic AF',
            '<p><b>Rutherford IIb</b> embolic occlusion at the femoral bifurcation. <b>Heparin now</b>, then <b>emergency femoral embolectomy</b>; the history and the normal left leg make an embolus likely, so imaging need not delay theatre. Plan a <b>fasciotomy</b>: ischaemia beyond 6 hours with motor loss.</p>'
            + ev('Heparin on diagnosis: 70–100 IU/kg (or 5000 IU) bolus then an infusion (ESVS 2020, as summarised in Endovascular Today 2026). Clinical decision without imaging for a clear embolus with a normal contralateral limb is accepted practice.'),
            view(G + V(0, 0, -60), (0.2, 1, 0.2), 480), show=[*LEG, 'ali-embolus', 'ali-clot', 'laa-thrombus'], highlight=['ali-embolus'], labels=['ali-embolus', 'leg-sfa', 'pfa-r'], opacity=LOP,
            lead='<p>A <b>34-year-old woman</b> with rheumatic mitral stenosis and AF, off warfarin for 2 months (could not afford INR checks). <b>8 hours</b> of a painful, cold, white right leg. No femoral pulse on the right; a normal left leg. Numb to the ankle; cannot move her toes well. No Doppler signal at the ankle; venous signal present.</p>',
            quiz=ask('What is the first treatment, given in the emergency department?', 'Unfractionated heparin: a bolus then an infusion',
                     'Heparin stops propagation of clot while theatre is arranged.', 'Aspirin only', 'Thrombolysis on the ward', 'Warm the leg and observe'))
        emb_exp = step('ali-expose', 'Expose', 'Expose the common femoral bifurcation; control; transverse arteriotomy',
            '<p>A vertical groin incision over the femoral artery (local anaesthesia is possible in a frail patient). Control the <b>common femoral, superficial femoral and profunda</b> with slings. Heparin is already running. A <b>transverse arteriotomy</b> just above the bifurcation (closed primarily without narrowing); a longitudinal one if the artery is diseased and will need a patch.</p>',
            view(G + V(0, 0, -15), (0.15, 1, 0.3), 200), show=[*LEG, 'ali-embolus', 'incision-groin-r'], highlight=hv(['cfa-r', 'leg-sfa', 'pfa-r']), danger=hv(['fv-r']), labels=hv(['cfa-r', 'pfa-r', 'leg-sfa', 'fv-r', 'ali-embolus']), opacity=LOP,
            action={'kind': 'reveal', 'label': 'Open the artery', 'port': 'groin-r', 'ids': ['ali-arteriotomy']})
        emb_fog = step('ali-fogarty', 'Fogarty', 'Fogarty embolectomy: proximal, then distal; check back-bleeding',
            '<p>The embolus often extrudes. <b>Proximally</b>: a 5F Fogarty catheter into the iliac until pulsatile inflow returns. <b>Distally</b>: a 3–4F catheter down the superficial femoral and the profunda, passed beyond the clot, the balloon inflated gently with saline and withdrawn steadily while the inflation is adjusted to the vessel. Repeat until <b>two clean passes</b> and good <b>back-bleeding</b>. Flush with heparinised saline.</p>'
            '<p>Over-inflation damages the intima, causing dissection and later stenosis. If the catheter will not pass or flow stays poor, image on the table (angiogram) and explore the below-knee popliteal to direct the catheter into each tibial artery.</p>',
            view(K + V(0, 0, 150), (0.3, 1, 0.15), 900), show=[*LEG, 'ali-embolus', 'ali-clot', 'ali-arteriotomy'], highlight=['fogarty'], labels=['fogarty', 'ali-clot', 'leg-pop'], opacity=LOP,
            action={'kind': 'reveal', 'label': 'Pass the Fogarty catheter', 'port': 'groin-r', 'ids': ['fogarty']},
            quiz=ask('After embolectomy, which sign shows the distal arteries are clear?', 'Brisk back-bleeding and two clean catheter passes, confirmed by a completion angiogram or Doppler signals',
                     'Back-bleeding alone can come from collaterals; confirm with imaging or pedal Doppler.', 'The patient says the pain is better', 'The foot is warm within a minute', 'A femoral pulse alone'))
        emb_clear = step('ali-clear', 'Clear', 'Clot out; close the arteriotomy; check the foot',
            '<p>The clot is withdrawn. Close the transverse arteriotomy with interrupted or running 5-0 polypropylene, flushing before the last sutures. Check Doppler signals at the posterior tibial and dorsalis pedis; a <b>completion angiogram</b> where available.</p>'
            '<p>Send the clot for histology (myxoma, infection) and look for the source: <b>ECG, echocardiogram</b> (left atrial thrombus, valves, vegetations).</p>',
            WHOLE, show=[*LEG, 'ali-embolus', 'ali-clot', 'fogarty'], highlight=hv(['leg-pop', 'leg-pta', 'leg-ata']), labels=hv(['leg-pta', 'leg-dpa', 'leg-pop']), opacity=LOP,
            action={'kind': 'decorticate', 'label': 'Withdraw the clot', 'port': 'groin-r', 'ids': ['ali-embolus', 'ali-clot', 'fogarty']})
        fasc = step('ali-fasciotomy', 'Fasciotomy', 'Four-compartment fasciotomy through two incisions',
            '<p>After revascularisation of severe or prolonged ischaemia, reperfusion swells the muscle inside unyielding fascia. <b>Prophylactic fasciotomy</b> when ischaemia exceeded about 6 hours with neuromotor deficit; therapeutic when compartment syndrome appears (pain on passive stretch, tense calf, falling sensation).</p>'
            + ul('<b>Anterolateral incision</b>: midway between the tibial crest and the fibula; open the anterior and lateral compartments either side of the intermuscular septum; protect the <b>superficial peroneal nerve</b> in the lower third',
                 '<b>Posteromedial incision</b>: 1–2 cm behind the posteromedial border of the tibia; protect the great saphenous vein and saphenous nerve; open the superficial posterior compartment, then detach soleus from the tibia to open the <b>deep posterior</b> compartment')
            + '<p>Long incisions through skin and fascia; leave open, dress (or negative pressure); close or graft at 3–7 days. Dead muscle is debrided.</p>'
            + ev('Fasciotomy for severe ischaemia lasting more than 6 hours (compartment syndrome in up to 30% of revascularisations; Endovascular Today 2026 summary of ESVS 2020). Two-incision technique and the diastolic pressure minus compartment pressure under 30 mmHg threshold: AO Surgery Reference.'),
            view(P('xs-centre') if 'xs-centre' in LM else K, (0.15, 0.75, 1), 300), show=[*[i for i in LEG if i not in ('leg-tibia', 'leg-fibula')], 'xs-ant', 'xs-lat', 'xs-deep', 'xs-sup', 'xs-bones'],
            highlight=['xs-deep'], danger=hv(['leg-gsv']), labels=['xs-ant', 'xs-lat', 'xs-sup', 'xs-deep', 'fasc-al', 'fasc-pm'], opacity={**LOP, 'leg-skin': 0.08},
            action={'kind': 'reveal', 'label': 'Make the two incisions', 'port': 'groin-r', 'ids': ['fasc-al', 'fasc-pm']},
            quiz=ask('Which compartment is most often missed through a single lateral incision?', 'The deep posterior compartment',
                     'It lies behind the interosseous membrane and soleus; reach it through the posteromedial incision by detaching soleus from the tibia.', 'The anterior compartment', 'The lateral compartment', 'The superficial posterior compartment'))
        proc('ali-emb', 'ali', 'Acute limb ischaemia', 'Embolectomy and fasciotomy',
             'Rutherford grading, heparin, femoral embolectomy with a Fogarty catheter, source search, and four-compartment fasciotomy.',
             [('Patho', 'other', [ali_patho]), ('Anatomy', 'other', [ali_anat]), ('Case', 'other', [ali_case]), ('Expose', 'artery', [emb_exp]), ('Fogarty', 'artery', [emb_fog]),
              ('Clear', 'artery', [emb_clear]), ('Fasciotomy', 'other', [fasc])], ALI_SRC, 'Vascular')

        # ------------------------------------------------------------ fem-pop bypass
        fp_patho = step('fp-patho', 'Pathophysiology', 'Pathophysiology: femoropopliteal disease and chronic limb-threatening ischaemia',
            '<p>The <b>superficial femoral artery at the adductor canal</b> is the commonest site of leg atherosclerosis. Collaterals from the profunda reconstitute the popliteal: alone, an SFA occlusion usually causes calf claudication. Add tibial disease, diabetes or a wound, and the limb becomes <b>threatened</b> (rest pain, ulcer, gangrene: Rutherford 4–6).</p>'
            '<p><b>Stage the limb</b> (WIfI: wound, ischaemia, foot infection) and the anatomy (GLASS); decide by the PLAN: <b>P</b>atient risk, <b>L</b>imb severity, <b>AN</b>atomy. Best medical therapy for every patient.</p>'
            + ev('Global Vascular Guidelines on CLTI (Conte et al., Eur J Vasc Endovasc Surg / J Vasc Surg 2019): WIfI, GLASS, PLAN; great saphenous vein the preferred conduit.'),
            THIGH, show=[*LEG, 'sfa-occlusion'], highlight=['sfa-occlusion'], labels=hv(['sfa-occlusion', 'pfa-r', 'leg-pop']), opacity=LOP, after=True,
            quiz=ask('An isolated SFA occlusion usually causes which symptom?', 'Calf claudication',
                     'Profunda collaterals reconstitute the popliteal; rest pain or tissue loss needs more extensive disease.', 'Buttock claudication', 'Rest pain at night', 'Gangrene of the toes'))
        fp_case = step('fp-case', 'Case', 'Case: rest pain and a heel ulcer; good saphenous vein',
            '<p>CLTI with a long SFA occlusion, a patent popliteal and good run-off, a fit patient and a good <b>single-segment great saphenous vein</b>: <b>femoral to below-knee popliteal bypass with reversed vein</b>.</p>'
            + ev('BEST-CLI (Farber et al., NEJM 2022): with adequate single-segment GSV, surgery reduced major adverse limb events or death (42.6% vs 57.4%). BASIL-2 (Bradbury et al., Lancet 2023): for infrapopliteal disease, an endovascular-first strategy gave better amputation-free survival (death or major amputation 53% vs 63%). The two are complementary: anatomy and conduit decide.'),
            THIGH, show=[*LEG, 'sfa-occlusion', 'leg-gsv'], highlight=['sfa-occlusion'], labels=['sfa-occlusion', 'leg-gsv', 'leg-pop'], opacity=LOP,
            lead='<p>A <b>66-year-old man</b>, diabetic, ex-smoker. 3 months of <b>rest pain</b> in the right foot, a 2 cm heel ulcer. ABI 0.35; toe pressure 25 mmHg. CT angiogram: <b>SFA occluded from its origin to the adductor hiatus</b>, the popliteal reconstituted above the knee, two-vessel run-off. Duplex: <b>great saphenous vein 3.5–4 mm</b> throughout.</p>',
            quiz=ask('Which conduit gives the best patency to the below-knee popliteal?', 'The patient\'s own great saphenous vein',
                     'Autologous vein outperforms prosthetic grafts below the knee.', 'PTFE', 'Dacron', 'Cephalic vein from the arm, always'))
        fp_vein = step('fp-vein', 'Vein', 'Harvest the great saphenous vein; reverse it',
            '<p>Mark the vein with duplex. Harvest through continuous or skip incisions, ligating tributaries with fine ties without narrowing the vein. Gently distend with heparinised saline and check for leaks. <b>Reverse</b> the vein so its valves do not obstruct flow (or leave it in situ and cut the valves with a valvulotome).</p>',
            view(K + V(-30, 0, 150), (-0.5, 1, 0.15), 800), show=[*LEG, 'leg-gsv', 'sfa-occlusion'], highlight=['leg-gsv'], labels=['leg-gsv'], opacity=LOP,
            action={'kind': 'reveal', 'label': 'Open over the vein', 'port': 'groin-r', 'ids': ['inc-gsv']})
        fp_graft = step('fp-graft', 'Graft', 'Femoral and below-knee popliteal anastomoses; tunnel',
            '<p>Expose the <b>common femoral bifurcation</b> (inflow) and the <b>below-knee popliteal</b> through a medial incision behind the tibia (retract gastrocnemius back). Heparin. Tunnel the vein deep to sartorius and behind the knee (anatomic tunnel) without twisting. <b>End-to-side</b> anastomoses at both ends (6-0 polypropylene distally), the toe of the distal anastomosis cut to avoid narrowing.</p>',
            THIGH, show=[*LEG, 'sfa-occlusion'], highlight=['graft-fempop'], labels=['graft-fempop', 'anast-fempop', 'sfa-occlusion'], opacity=LOP,
            action={'kind': 'reveal', 'label': 'Sew in the vein graft', 'port': 'groin-r', 'ids': ['graft-fempop', 'anast-fempop']})
        fp_check = step('fp-check', 'Check', 'Completion imaging; graft surveillance',
            '<p>Check a <b>pulse in the graft</b> and pedal Doppler signals; a completion angiogram or duplex for kinks, residual valves and the distal anastomosis. <b>Duplex surveillance</b> of vein grafts (for example at 1, 3, 6 and 12 months) finds stenoses before they occlude. Antiplatelet and statin for life.</p>',
            THIGH, show=[*LEG, 'sfa-occlusion', 'graft-fempop', 'anast-fempop'], labels=['graft-fempop'], opacity=LOP,
            quiz=ask('Why are vein grafts followed with duplex scanning?', 'To detect stenoses in the graft before it occludes, when they can be fixed',
                     'Most vein graft failures after the first month come from intimal hyperplasia at anastomoses or valve sites.', 'To measure the ABI only', 'Because they always infect', 'It is not needed'))
        proc('fp-gsv', 'infrainguinal', 'Infrainguinal bypass and amputation', 'Femoral to below-knee popliteal (reversed vein)',
             'CLTI staging (WIfI, GLASS, PLAN), BEST-CLI and BASIL-2, vein harvest, femoral and below-knee popliteal anastomoses, surveillance.',
             [('Patho', 'other', [fp_patho]), ('Case', 'other', [fp_case]), ('Vein', 'vein', [fp_vein]), ('Graft', 'artery', [fp_graft]), ('Check', 'other', [fp_check])], FP_SRC, 'Vascular')

        # ------------------------------------------------------------ amputation levels
        AMP = ['amp-ray', 'amp-tma', 'amp-bka', 'amp-tka', 'amp-aka']
        am_lv = step('am-levels', 'Pathophysiology', 'Choosing the level: as distal as will heal',
            '<p>The level is chosen by <b>perfusion</b> (will it heal?), <b>infection</b> (is all dead and infected tissue removed?) and <b>function</b> (will this patient walk?).</p>'
            '<table class="mini"><tr><th>Level</th><th>Use</th><th>Note</th></tr>'
            '<tr><td>Toe or ray</td><td>dry gangrene or osteomyelitis of one or two toes</td><td>needs adequate foot perfusion; revascularise first</td></tr>'
            '<tr><td>Transmetatarsal</td><td>forefoot loss, three or more toes</td><td>preserves a weight-bearing foot; more revisions than BKA but better walking</td></tr>'
            '<tr><td><b>Below-knee</b></td><td>unreconstructable foot, heel or midfoot sepsis</td><td>the knee makes prosthetic walking far easier</td></tr>'
            '<tr><td>Through-knee</td><td>BKA not healable, long stump wanted</td><td>end-bearing; for non-walkers it balances well in sitting</td></tr>'
            '<tr><td>Above-knee</td><td>no healable distal level, flexion contracture, bed-bound</td><td>most reliable healing; walking with a prosthesis much harder</td></tr></table>'
            '<p>More proximal levels cost more energy to walk, so a vascular patient is far more likely to walk on a below-knee than an above-knee prosthesis. In sepsis, a <b>guillotine</b> amputation first (ankle level), then a definitive level once the infection settles.</p>'
            + ev('Global Vascular Guidelines 2019 (amputation in CLTI); Waters et al. (J Bone Joint Surg Am 1976): energy cost of walking rises with the level of amputation; transmetatarsal versus below-knee: more revisions but higher ambulation after TMA (published series).'),
            WHOLE, show=[*LEG, *AMP], highlight=['amp-bka'], labels=AMP, opacity={**LOP, **{a: 0.5 for a in AMP}}, after=True,
            quiz=ask('Why is a below-knee amputation preferred to an above-knee one when both would heal?', 'Keeping the knee makes walking with a prosthesis much more likely and less tiring',
                     'Energy cost and prosthetic success fall steeply with an above-knee level.', 'It is quicker to perform', 'It never needs revision', 'It avoids anaesthesia'))
        am_case = step('am-case', 'Case', 'Case: wet gangrene of the foot in a diabetic',
            '<p><b>Septic, unsalvageable foot</b>: resuscitate, antibiotics, and a <b>guillotine ankle amputation</b> today to control sepsis; a <b>below-knee amputation</b> in 3–5 days once the sepsis settles, if the popliteal pulse and skin at that level allow healing.</p>',
            view(A_ + V(0, 60, 40), (0.3, 1, 0.3), 520), show=[*LEG, 'amp-bka'], labels=['amp-bka', 'leg-foot'], opacity=LOP,
            lead='<p>A <b>58-year-old man</b>, poorly controlled diabetes. Wet gangrene of the forefoot spreading to the midfoot, crepitus, foul smell; temperature 39 °C, glucose 24 mmol/L. Popliteal pulse palpable; no foot pulses.</p>',
            quiz=ask('What is the first operation?', 'A guillotine amputation at the ankle to control sepsis, then a definitive level later',
                     'Staging avoids closing a stump in infected tissue.', 'Immediate above-knee amputation', 'Fem-pop bypass', 'Toe amputation only'))
        am_bka = step('am-bka', 'BKA', 'Below-knee amputation: long posterior flap',
            '<p>Mark the <b>anterior incision</b> at the level of bone section, about <b>12–15 cm below the knee joint</b> (or 10 cm below the tibial tuberosity); the <b>long posterior flap</b> (Burgess) is about as long as the leg is wide at that level. Divide the tibia, bevel its anterior edge, cut the <b>fibula 1–2 cm shorter</b>. Ligate the anterior tibial, posterior tibial and peroneal vessels; pull down, divide and let retract the <b>tibial nerve</b>. Trim the soleus bulk; fold the gastrocnemius flap forward over the bone; close without tension; a soft dressing or rigid removable dressing to prevent knee flexion contracture.</p>',
            view(K + V(0, 0, -150), (0.4, 0.8, 0.2), 480), show=[*LEG, 'amp-bka'], highlight=['amp-bka'], labels=['amp-bka', 'leg-tibia', 'leg-fibula'], opacity={**LOP, 'amp-bka': 0.6},
            action={'kind': 'reveal', 'label': 'Mark the flap', 'port': 'groin-r', 'ids': ['amp-bka-flap']},
            quiz=ask('Why is the fibula cut shorter than the tibia?', 'So the fibular end does not press into the stump and the socket',
                     'A long fibula is prominent and painful in the prosthesis.', 'To save time', 'To protect the peroneal artery', 'It is not cut'))
        am_after = step('am-after', 'After', 'Stump care, rehabilitation, the other leg',
            '<p>Pain control (including phantom pain), glucose and nutrition, wound checks. <b>Prevent knee flexion contracture</b>: lie prone, keep the knee straight, early physiotherapy. Stump shaping, then prosthetic fitting at about 6–8 weeks. Examine and protect the <b>other foot</b>: the risk to the second leg is high. Secondary prevention.</p>',
            WHOLE, show=[*LEG, 'amp-bka', 'amp-bka-flap'], labels=['amp-bka'], opacity=LOP)
        proc('amp-levels', 'infrainguinal', 'Infrainguinal bypass and amputation', 'Amputation levels',
             'Choosing the level, staged guillotine amputation in sepsis, below-knee amputation with a long posterior flap, rehabilitation.',
             [('Levels', 'other', [am_lv]), ('Case', 'other', [am_case]), ('BKA', 'other', [am_bka]), ('After', 'other', [am_after])], FP_SRC, 'Vascular')

    # =================================================================================================== arteriovenous fistulas
    if has('arm-brachial'):
        ARM = hv(['arm-skin', 'humerus-r', 'arm-humerus', 'arm-radius', 'arm-ulna', 'arm-brachial', 'arm-radial', 'arm-ulnar', 'arm-cephalic', 'arm-basilic', 'arm-mcv', 'axillary-a-r', 'sca-r'])
        AOP = {'arm-skin': 0.12, 'humerus-r': 0.3, 'arm-humerus': 0.3, 'arm-radius': 0.3, 'arm-ulna': 0.3}
        E, Wr = P('elbow-r'), P('wrist-r')
        ARMV = view((E + Wr) / 2 + V(0, 0, 60), (0.55, 1, 0.15), 640)
        av_patho = step('av-patho', 'Pathophysiology', 'Why a fistula: access for haemodialysis',
            '<p>Haemodialysis needs blood flow of about 300–400 mL/min, three times a week, for years. A <b>fistula</b> joins an artery to a superficial vein: the vein <b>arterialises</b> (dilates and thickens) over weeks until it can be needled. Compared with a <b>graft</b> (PTFE) it has fewer infections and interventions; compared with a <b>tunnelled catheter</b>, far less bacteraemia and central vein stenosis.</p>'
            '<p><b>KDOQI 2019</b> replaced "fistula first" with <b>"patient first"</b>: an <b>ESKD Life-Plan</b> for every patient with progressive CKD (eGFR 15–20), choosing the access that fits that patient\'s expected course: forearm first, distal to proximal, non-dominant arm, protecting veins (no cannulas or blood tests in the planned arm).</p>'
            '<p><b>Maturation</b>: the old <b>rule of 6s</b> (at 6 weeks: flow 600 mL/min, diameter 6 mm, no more than 6 mm deep). KDOQI 2019 defines maturity by function: dialysis with two needles in more than two-thirds of sessions over 4 consecutive weeks.</p>'
            + ev('KDOQI Clinical Practice Guideline for Vascular Access: 2019 Update (Lok et al., Am J Kidney Dis 2020): ESKD Life-Plan; no absolute minimum vessel diameter (vessels under 2 mm need careful evaluation); selective pre-operative ultrasound for high-risk patients; maturation assessed at 4–6 weeks; rope-ladder cannulation preferred. Rule of 6s: earlier KDOQI guidance.'),
            ARMV, show=ARM, highlight=hv(['arm-cephalic', 'arm-radial', 'arm-brachial']), labels=hv(['arm-cephalic', 'arm-basilic', 'arm-radial', 'arm-brachial', 'arm-mcv']), opacity=AOP, after=True, spin=True,
            quiz=ask('What did KDOQI 2019 change about access planning?', 'From "fistula first" to a patient-first ESKD Life-Plan',
                     'The access is chosen for the patient\'s expected life course, not a fistula at any cost.', 'Catheters first for everyone', 'Grafts are preferred to fistulas', 'Fistulas only in the leg'))
        av_case = step('av-case', 'Case', 'Case: planning access before dialysis starts',
            '<p>Progressive CKD with an eGFR of 14: create access now so it matures before dialysis. A good forearm cephalic vein and radial artery: <b>radiocephalic fistula</b> on the non-dominant arm. If it fails or the vein is poor: <b>brachiocephalic</b>, then a <b>transposed brachiobasilic</b>, then a graft.</p>',
            ARMV, show=ARM, highlight=hv(['arm-cephalic', 'arm-radial']), labels=hv(['arm-cephalic', 'arm-radial', 'arm-basilic']), opacity=AOP,
            lead='<p>A <b>45-year-old man</b>, hypertensive nephropathy, eGFR 14 and falling, not yet on dialysis; <b>left-handed</b>. Allen test normal. Ultrasound of the right arm: radial artery 2.4 mm, forearm cephalic vein 2.8 mm and continuous to the elbow; no previous cannulas.</p>',
            quiz=ask('Which access is first for this man?', 'A radiocephalic fistula in the right (non-dominant) forearm',
                     'Distal first preserves proximal sites for later; the non-dominant arm leaves the dominant hand free on dialysis.', 'A tunnelled jugular catheter', 'A brachiobasilic transposition', 'A thigh graft'))
        rc = step('av-rc', 'Radiocephalic', 'Radiocephalic fistula at the wrist (Brescia–Cimino)',
            '<p>Local or regional anaesthesia. A <b>longitudinal incision</b> between the radial artery and the cephalic vein at the wrist. Mobilise the vein, ligate its distal end and divide it; dilate gently. Expose the radial artery under the deep fascia, control it, open 6–8 mm. Swing the vein to the artery: <b>vein end to artery side</b>, 7-0 polypropylene. A thrill should be felt at once.</p>',
            view(Wr + V(0, 0, 40), (0.7, 0.9, 0.2), 230), show=ARM, highlight=hv(['arm-radial', 'arm-cephalic']), labels=hv(['arm-radial', 'arm-cephalic', 'avf-rc']), opacity=AOP,
            action={'kind': 'reveal', 'label': 'Make the anastomosis', 'port': 'wrist-r', 'ids': ['inc-avf', 'avf-rc', 'anast-avf']})
        rc_m = step('av-rc-mature', 'Mature', 'Maturation: the forearm vein arterialises',
            '<p>Over 4–6 weeks the cephalic vein dilates and thickens. Examine it at 4–6 weeks: a continuous thrill, a soft pulse (not hammering), a straight segment long enough for two needles, superficial enough to feel.</p>'
            '<table class="mini"><tr><th colspan="2">The rule of 6s: at 6 weeks a mature fistula has</th></tr>'
            '<tr><td><b>Flow</b></td><td>600 mL/min or more</td></tr><tr><td><b>Diameter</b></td><td>6 mm or more</td></tr>'
            '<tr><td><b>Depth</b></td><td>no more than 6 mm under the skin</td></tr><tr><td><b>When</b></td><td>assessed at 6 weeks</td></tr></table>'
            '<p><b>KDOQI 2019</b> defines maturity by use: the fistula supports dialysis with two needles in more than two-thirds of sessions over 4 consecutive weeks. The rule of 6s stays a useful bedside and duplex check.</p>'
            '<p><b>Not maturing by 6 weeks</b>: duplex ultrasound. A <b>juxta-anastomotic stenosis</b> (the commonest cause in the forearm): balloon angioplasty or revision. <b>Competing side branches</b> stealing flow: ligate. A <b>deep vein</b>: superficialise. A small, diseased artery: a more proximal fistula.</p>'
            + ev('Rule of 6s: earlier KDOQI vascular access guidance (as summarised by the Renal Fellow Network). KDOQI 2019 update (Lok et al., Am J Kidney Dis 2020): functional definition of maturation; assess at 4–6 weeks.'),
            ARMV, show=[*ARM, 'avf-rc', 'anast-avf'], highlight=['avf-rc-mature'], labels=['avf-rc-mature'], opacity=AOP,
            action={'kind': 'reveal', 'label': 'Six weeks later', 'port': 'wrist-r', 'ids': ['avf-rc-mature']},
            quiz=ask('At 6 weeks a radiocephalic fistula has a weak thrill that fades just above the anastomosis. Most likely?', 'A juxta-anastomotic stenosis: ultrasound, then balloon angioplasty or revision',
                     'The commonest cause of failure to mature in the forearm.', 'Normal: wait 6 months', 'Heart failure', 'Steal syndrome'))
        bc = step('av-bc', 'Brachiocephalic', 'Brachiocephalic fistula at the elbow',
            '<p>A <b>transverse incision</b> in the antecubital fossa. The cephalic vein (or the median cubital vein, giving outflow to both cephalic and basilic) to the <b>brachial artery</b>, end to side, with an arteriotomy limited to 4–6 mm to reduce steal. The bicipital aponeurosis is divided; the median nerve lies medial to the artery.</p>',
            view(E + V(0, 0, 30), (0.4, 1, 0.2), 230), show=ARM, highlight=hv(['arm-brachial', 'arm-cephalic', 'arm-mcv']), danger=[], labels=hv(['arm-brachial', 'arm-cephalic', 'arm-mcv', 'avf-bc']), opacity=AOP,
            action={'kind': 'reveal', 'label': 'Make the anastomosis', 'port': 'elbow-r', 'ids': ['inc-avf', 'avf-bc', 'avf-bc-mature', 'anast-avf']},
            quiz=ask('Six hours after a brachiocephalic fistula the hand is cold, painful and numb. Diagnosis and action?', 'Access-related hand ischaemia (steal): urgent assessment; ligation or a flow-reducing or DRIL procedure',
                     'Severe steal with neurological signs is an emergency; it is commoner with brachial fistulas, in diabetics and the elderly.', 'Normal swelling', 'Venous hypertension', 'Thrombosis of the fistula'))
        bb = step('av-bb', 'Basilic', 'Transposed brachiobasilic fistula',
            '<p>When the cephalic vein is unusable. The <b>basilic vein</b> runs deep to the fascia in the upper arm beside the <b>medial cutaneous nerve of the forearm</b> and the median nerve: mobilise it along the arm through a long medial incision (or several), then <b>tunnel it superficially and laterally</b> so it can be needled, and join it to the brachial artery. One or two stages.</p>',
            view(E + V(-20, 0, 120), (-0.2, 1, 0.2), 420), show=ARM, highlight=hv(['arm-basilic', 'arm-brachial']), labels=hv(['arm-basilic', 'arm-brachial', 'avf-bb']), opacity=AOP,
            action={'kind': 'reveal', 'label': 'Transpose the basilic vein', 'port': 'elbow-r', 'ids': ['inc-avf', 'avf-bb', 'anast-avf']})
        AVS = AVF_SRC
        proc('avf-rc', 'avf', 'Arteriovenous fistula (dialysis access)', 'Radiocephalic (wrist)',
             'Access planning (KDOQI 2019 Life-Plan), radiocephalic fistula, maturation and its failure.',
             [('Patho', 'other', [av_patho]), ('Case', 'other', [av_case]), ('Radiocephalic', 'artery', [rc]), ('Mature', 'vein', [rc_m])], AVS, 'Vascular')
        proc('avf-bc', 'avf', 'Arteriovenous fistula (dialysis access)', 'Brachiocephalic (elbow)',
             'Brachiocephalic fistula at the elbow; steal syndrome.',
             [('Patho', 'other', [{**av_patho, 'id': 'avb-patho'}]), ('Brachiocephalic', 'artery', [bc])], AVS, 'Vascular')
        proc('avf-bb', 'avf', 'Arteriovenous fistula (dialysis access)', 'Brachiobasilic transposition',
             'Transposed basilic vein fistula for when the cephalic vein is unusable.',
             [('Patho', 'other', [{**av_patho, 'id': 'avt-patho'}]), ('Transpose', 'vein', [bb])], AVS, 'Vascular')

    # =================================================================================================== post-TB bronchiectasis (left upper lobe)
    if has('bx-lul') and 'lul-open' in procs:
        base = copy.deepcopy(procs['lul-open']); st = base['steps']
        keep = [s for s in st if s['phase'] not in ('Pathophysiology', 'Case', 'Consent', 'ICU')]
        for s in keep: s['id'] = 'bx-' + s['id']
        anat_ = next((s for s in keep if s['phase'] == 'Anatomy'), None)
        thor = next((s for s in keep if s['phase'] == 'Setup'), None)
        rest = [s for s in keep if s not in (anat_, thor)]
        BX = ['bx-lul', 'bx-bronchial-a', 'bx-adhesions']
        LV = anat_['view'] if anat_ else view(P('hilum-l'), (-1, 0.2, 0.2), 300)
        tgt = P('bx-lul') if 'bx-lul' in LM else P('hilum-l')
        bxv = view(tgt + V(10, 0, -20), (-1, 0.35, 0.25), 330)
        bx_p = step('bx-patho', 'Pathophysiology', 'Pathophysiology: post-tuberculous bronchiectasis and the destroyed lobe',
            '<p>Tuberculosis heals with <b>fibrosis, cavities and traction bronchiectasis</b>, most often in the upper lobes. Dilated, thick-walled bronchi pool secretions: a <b>vicious cycle</b> of infection, inflammation and further damage. Chronic inflammation enlarges the <b>bronchial arteries</b> (systemic pressure), which bleed: <b>haemoptysis</b>, sometimes massive. Cavities may host an <b>aspergilloma</b>. The pleura fuses to the chest wall.</p>'
            + chain('Healed TB: fibrosis, cavities', 'Traction bronchiectasis', 'Pooling, infection, inflammation', '!Recurrent infection, haemoptysis, destroyed lobe')
            + '<p><b>Treat medically first</b>: exclude active TB (sputum GeneXpert and culture) and non-tuberculous mycobacteria; airway clearance, treatment of exacerbations. <b>Surgery</b> for localised disease with failed medical therapy, recurrent or massive haemoptysis (after bronchial artery embolisation, or when it fails), a destroyed lobe or lung as a septic focus, or an aspergilloma.</p>'
            + ev('Surgery for bronchiectasis-destroyed lung (Interdiscip Cardiovasc Thorac Surg 2024; 143 patients): no 30- or 90-day deaths; major complications 19.6% overall, 50% after pneumonectomy versus 13.4% after lobectomy; 76.2% asymptomatic at a median of 79 months; VATS feasible in selected patients (14% converted for adhesions or a frozen hilum). Breathe (ERS) review on surgery in bronchiectasis and TB.'),
            bxv, show=['lul', 'lll', 'heart', 'aorta', 'br-lul', 'br-left-main', 'trachea', *BX], highlight=['bx-lul'], danger=['bx-bronchial-a'], labels=['bx-lul', 'bx-bronchial-a', 'bx-adhesions'], opacity={'lul': 0.3, 'lll': 0.25, 'heart': 0.4},
            ct=tgt, after=True, spin=True,
            quiz=ask('What is the first treatment for massive haemoptysis from post-TB bronchiectasis?', 'Protect the airway (bleeding side down), resuscitate, and bronchial artery embolisation; surgery if it fails or recurs',
                     'Embolisation controls most bleeding; resection removes the source in localised disease.', 'Emergency pneumonectomy for all', 'Tranexamic acid alone', 'Bronchoscopy and wait'))
        bx_c = step('bx-case', 'Case', 'Case: recurrent haemoptysis from a destroyed left upper lobe',
            '<p>Localised, symptomatic disease in a fit patient with a healthy remaining lung, bleeding again after embolisation: <b>left upper lobectomy</b>. Open thoracotomy is safer than VATS with dense adhesions and calcified hilar nodes.</p>'
            '<p><b>Before surgery</b>: sputum negative for TB; treat infection; physiotherapy; nutrition; spirometry and a perfusion scan if borderline; bronchoscopy to exclude an endobronchial lesion and to see the left lower lobe bronchus is clean.</p>',
            bxv, show=['lul', 'lll', 'heart', 'aorta', 'br-lul', 'br-left-main', 'trachea', *BX], highlight=['bx-lul'], labels=['bx-lul', 'bx-bronchial-a'], opacity={'lul': 0.3, 'lll': 0.25, 'heart': 0.4}, ct=tgt,
            lead='<p>A <b>32-year-old woman</b>, TB treated 6 years ago. Two years of daily purulent sputum and <b>three episodes of haemoptysis</b> (the last 300 mL), recurring 4 months after bronchial artery embolisation. CT: <b>a shrunken left upper lobe</b> with cystic bronchiectasis and a small cavity; left lower lobe and right lung clear. Sputum GeneXpert negative twice. FEV1 72% predicted.</p>',
            quiz=ask('Which finding makes her a good candidate for resection?', 'Disease confined to one lobe with healthy remaining lung',
                     'Localised disease with adequate reserve; diffuse bronchiectasis is managed medically.', 'Bilateral disease', 'Active TB on sputum', 'FEV1 25%'))
        adh = step('bx-adhesions', 'Adhesions', 'Through the adhesions: the extrapleural plane',
            '<p>After TB the lung is often <b>fused to the chest wall</b>. Where the adhesions are filmy, divide them with cautery close to the chest wall. Where the pleura is thick and symphysed, the <b>extrapleural plane</b> (outside the parietal pleura, against the endothoracic fascia) is often bloodless and quicker; at the apex beware the <b>subclavian vessels</b>, and over the spine the sympathetic chain. Expect bleeding from hypertrophied <b>bronchial and intercostal collaterals</b>: control them as you go.</p>',
            view(tgt + V(-20, 0, 0), (-1, 0.2, 0.25), 300), show=['lul', 'lll', 'heart', 'aorta', *BX], highlight=['bx-adhesions'], danger=['bx-bronchial-a'], labels=['bx-adhesions', 'bx-bronchial-a'], opacity={'lul': 0.45, 'lll': 0.3, 'heart': 0.4}, ct=tgt,
            action={'kind': 'decorticate', 'label': 'Free the lobe', 'port': 'thoracotomy-l', 'ids': ['bx-adhesions']},
            quiz=ask('With a densely symphysed apex, which plane is often safest?', 'The extrapleural plane, outside the parietal pleura',
                     'It avoids tearing into the lung and its cavities; watch the subclavian vessels at the apex.', 'Straight through the lung', 'The intrapericardial plane', 'Leave the lobe stuck'))
        notes = ('<p class="evidence"><b>In bronchiectasis</b>: hilar nodes are often calcified and stuck to the arteries; consider proximal control of the pulmonary artery before dissecting a frozen hilum. Close the bronchus carefully; in a heavily infected field, cover the stump with a pedicled flap (intercostal muscle or pericardial fat).</p>')
        for s in rest:
            s['show'] = [*s.get('show', []), *[i for i in BX if i != 'bx-adhesions']]
            s['opacity'] = {**s.get('opacity', {}), 'bx-lul': 0.6}
            if s['phase'] == 'Bronchus': s['body'] = s['body'] + notes
        groups = [('Patho', 'other', [bx_p]), *([('Anatomy', 'other', [anat_])] if anat_ else []), ('Case', 'other', [bx_c])]
        steps = []
        sq = []
        for i, (lab, kind, sts) in enumerate(groups):
            sq.append({'label': lab, 'kind': kind}); steps += [{**s, 'seq': i} for s in sts]
        if thor: steps.append({**thor, 'seq': None}); steps[-1].pop('seq')
        n0 = len(sq); sq.append({'label': 'Adhesions', 'kind': 'other'}); steps.append({**adh, 'seq': n0})
        labels = {s['seq']: q for s in rest if s.get('seq') is not None for q in [base['sequence'][s['seq']]]}
        seen = {}
        for s in rest:
            old = s.get('seq')
            if old is None: steps.append(s); continue
            if old not in seen: seen[old] = len(sq); sq.append(dict(base['sequence'][old]))
            steps.append({**s, 'seq': seen[old]})
        procs['bx-lul'] = {**{k: base[k] for k in ('side', 'ports')}, 'id': 'bx-lul', 'op': 'bronchiectasis', 'opName': 'Bronchiectasis (post-TB)', 'name': 'Left upper lobectomy for bronchiectasis',
                           'approach': 'Destroyed left upper lobe (open)', 'summary': 'Post-tuberculous bronchiectasis: indications, embolisation, extrapleural dissection, a frozen hilum, protecting the stump.',
                           'steps': steps, 'sequence': sq, 'sources': BX_SRC + base.get('sources', [])[:4], 'group': 'Lobectomy'}


PERI_SRC = [
    {'title': 'Mayosi BM, et al. Prednisolone and Mycobacterium indicus pranii in tuberculous pericarditis (IMPI). N Engl J Med 2014;371:1121-30', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa1407380'},
    {'title': 'Perioperative outcomes and predictors of mortality after pericardiectomy for constrictive pericarditis in Africa: systematic review and meta-analysis. The Cardiothoracic Surgeon 2026', 'url': 'https://link.springer.com/article/10.1186/s43057-026-00221-4'},
    {'title': 'Adler Y, et al. 2015 ESC Guidelines for the diagnosis and management of pericardial diseases. Eur Heart J 2015;36:2921-64', 'url': 'https://pubmed.ncbi.nlm.nih.gov/26320112/'},
    {'title': 'Constrictive pericarditis: surgical management (IntechOpen)', 'url': 'https://www.intechopen.com/chapters/85949'},
    {'title': 'Pericardiectomy for constrictive pericarditis in a resource-constrained setting', 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8525283/'},
]
ALI_SRC = [
    {'title': 'Björck M, et al. ESVS 2020 clinical practice guidelines on the management of acute limb ischaemia. Eur J Vasc Endovasc Surg 2020;59:173-218', 'url': 'https://www.researchgate.net/publication/338287618_European_Society_for_Vascular_Surgery_ESVS_2020_Clinical_Practice_Guidelines_on_the_Management_of_Acute_Limb_Ischaemia'},
    {'title': 'Managing acute limb ischemia: a contemporary workflow. Endovascular Today 2026', 'url': 'https://evtoday.com/articles/2026-feb/managing-acute-limb-ischemia-a-contemporary-workflow'},
    {'title': 'AO Surgery Reference: compartment syndrome (lower limb), two-incision fasciotomy', 'url': 'https://surgeryreference.aofoundation.org/orthopedic-trauma/pediatric-trauma/tibial-shaft/further-reading/compartment-syndrome'},
]
FP_SRC = [
    {'title': 'Conte MS, et al. Global vascular guidelines on the management of chronic limb-threatening ischemia. Eur J Vasc Endovasc Surg 2019;58:S1-109', 'url': 'https://www.ejves.com/article/S1078-5884(19)30380-6/fulltext'},
    {'title': 'Farber A, et al. Surgery or endovascular therapy for chronic limb-threatening ischemia (BEST-CLI). N Engl J Med 2022;387:2305-16', 'url': 'https://www.nejm.org/doi/full/10.1056/NEJMoa2207899'},
    {'title': 'Bradbury AW, et al. BASIL-2: vein bypass first versus best endovascular treatment first for infrapopliteal CLTI. Lancet 2023;401:1798-809', 'url': 'https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(23)00462-2/fulltext'},
    {'title': 'Waters RL, et al. Energy cost of walking of amputees: the influence of level of amputation. J Bone Joint Surg Am 1976;58:42-6', 'url': 'https://www.semanticscholar.org/paper/Energy-cost-of-walking-of-amputees:-the-influence-Waters-Perry/9d6ff8917e238b65755cb13e73f85482eb4dd9b1'},
]
AVF_SRC = [
    {'title': 'Lok CE, et al. KDOQI clinical practice guideline for vascular access: 2019 update. Am J Kidney Dis 2020;75(4 Suppl 2):S1-164', 'url': 'https://www.ajkd.org/article/S0272-6386(19)31137-0/fulltext'},
    {'title': 'A Korean perspective on the 2019 KDOQI vascular access guideline (Kidney Res Clin Pract)', 'url': 'https://www.krcp-ksn.org/m/journal/view.php?number=6004'},
    {'title': 'Rule of 6s for dialysis access (Renal Fellow Network)', 'url': 'https://www.renalfellow.org/2011/09/02/from-rfn-archive-rule-of-6s-for/'},
]
BX_SRC = [
    {'title': 'Surgery for bronchiectasis-destroyed lung: feasibility of VATS, and surgical outcomes. Interdiscip Cardiovasc Thorac Surg 2024;38:ivad175', 'url': 'https://academic.oup.com/icvts/article/38/2/ivad175/7334460'},
    {'title': 'Surgical intervention in bronchiectasis and pulmonary tuberculosis. Breathe (ERS)', 'url': 'https://publications.ersnet.org/content/breathe/22/3/250370'},
    {'title': 'Surgical management of tuberculosis-related hemoptysis. Ann Thorac Surg', 'url': 'https://www.annalsthoracicsurgery.org/article/S0003-4975(04)01055-0/fulltext'},
]
