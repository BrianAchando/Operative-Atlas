"""Vascular, part two: best medical therapy for peripheral arterial disease; Rutherford IIa acute limb ischemia
(embolectomy with sensation and movement preserved); AAA management and surveillance (ESVS 2024); and common iliac
artery aneurysm (iliac branch device, internal iliac coils with an external iliac extension, open repair).

Added after postop.apply, so each entry carries its own counselling or consent and ICU steps.
"""
from __future__ import annotations

import copy

import numpy as np

V = lambda *a: np.array(a[0] if len(a) == 1 else a, float)
R = lambda v: [round(float(x), 1) for x in v]
U = lambda v: np.asarray(v, float) / (np.linalg.norm(v) + 1e-9)
ev = lambda t: f'<p class="evidence"><b>Evidence:</b> {t}</p>'
ul = lambda *xs: '<ul>' + ''.join(f'<li>{x}</li>' for x in xs) + '</ul>'
h4 = lambda t: f'<h4>{t}</h4>'
chain = lambda *xs: '<div class="chain">' + '<i>→</i>'.join(f'<span class="hot">{x[1:]}</span>' if x.startswith('!') else f'<span>{x}</span>' for x in xs) + '</div>'
tbl = lambda head, *rows: ('<table class="mini"><tr>' + ''.join(f'<th>{h}</th>' for h in head) + '</tr>'
                           + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows) + '</table>')
tag = lambda k, t: f' <span class="tag {k}">{t}</span>'
link = lambda href, t: f'<a class="link" href="#{href}">{t}</a>'

# ---------------------------------------------------------------------------------------------------- sources
ACC24 = {'title': 'Gornik HL, et al. 2024 ACC/AHA/AACVPR/APMA/ABC/SCAI/SVM/SVN/SVS/SIR/VESS guideline for the management of lower extremity peripheral artery disease. Circulation 2024. doi:10.1161/CIR.0000000000001251',
         'url': 'https://www.acc.org/latest-in-cardiology/articles/2024/05/14/13/40/new-acc-aha-guideline-focused-on-management-of-le-pad-gl-pad'}
AHA_TOP = {'title': 'American Heart Association. Top things to know: 2024 guideline for the management of lower extremity peripheral artery disease',
           'url': 'https://professional.heart.org/en/science-news/2024-guideline-for-the-management-of-lower-extremity-peripheral-artery-disease/top-things-to-know'}
AAFP = {'title': 'Arnold MJ. Management of lower extremity peripheral artery disease: guidelines from the ACC/AHA. Am Fam Physician 2025;112:571-3',
        'url': 'https://www.aafp.org/pubs/afp/issues/2025/1100/practice-guidelines-lower-extremity-peripheral-artery.html'}
ESVS_PAD = {'title': 'Nordanstig J, et al. ESVS 2024 clinical practice guidelines on the management of asymptomatic lower limb peripheral arterial disease and intermittent claudication. Eur J Vasc Endovasc Surg 2024;67:9-96',
            'url': 'https://www.sciencedirect.com/science/article/pii/S1078588423007414'}
GVG = {'title': 'Conte MS, et al. Global vascular guidelines on the management of chronic limb-threatening ischemia. Eur J Vasc Endovasc Surg 2019;58:S1-109',
       'url': 'https://www.ejves.com/article/S1078-5884(19)30380-6/fulltext'}
COMPASS = {'title': 'Anand SS, et al. Rivaroxaban with or without aspirin in patients with stable peripheral or carotid artery disease (COMPASS). Lancet 2018;391:219-29',
           'url': 'https://evtoday.com/news/rivaroxaban-with-or-without-aspirin-evaluated-in-patients-with-stable-peripheral-or-carotid-artery-disease'}
VOYAGER = {'title': 'Bonaca MP, et al. Rivaroxaban in peripheral artery disease after revascularization (VOYAGER PAD). N Engl J Med 2020;382:1994-2004',
           'url': 'https://www.iscpcardio.org/clinical-trials/rivaroxaban-pad/'}
TASC = {'title': 'Norgren L, et al. Inter-Society Consensus for the Management of Peripheral Arterial Disease (TASC II). J Vasc Surg 2007;45(Suppl S):S5-67', 'url': 'https://radcalculator.com/calc/tasc-ii'}
HIV = {'title': 'Robbs JV, Paruk N. Management of HIV vasculopathy: a South African experience. Eur J Vasc Endovasc Surg 2010;39(Suppl 1):S25-31',
       'url': 'https://www.sciencedirect.com/science/article/pii/S1078588410000055'}
ESVS_ALI = {'title': 'Björck M, et al. ESVS 2020 clinical practice guidelines on the management of acute limb ischaemia. Eur J Vasc Endovasc Surg 2020;59:173-218',
            'url': 'https://www.researchgate.net/publication/338287618_European_Society_for_Vascular_Surgery_ESVS_2020_Clinical_Practice_Guidelines_on_the_Management_of_Acute_Limb_Ischaemia'}
EVT26 = {'title': 'Sridharan N, Alimohamadi S. Managing acute limb ischemia: a contemporary workflow. Endovascular Today, February 2026',
         'url': 'https://evtoday.com/articles/2026-feb/managing-acute-limb-ischemia-a-contemporary-workflow'}
INVICTUS = {'title': 'Connolly SJ, et al. Rivaroxaban in rheumatic heart disease-associated atrial fibrillation (INVICTUS). N Engl J Med 2022;387:978-88',
            'url': 'https://www.acc.org/Latest-in-Cardiology/Clinical-Trials/2022/08/27/04/07/INVICTUS'}
ESVS_AAA = {'title': 'Wanhainen A, et al. ESVS 2024 clinical practice guidelines on the management of abdominal aorto-iliac artery aneurysms. Eur J Vasc Endovasc Surg 2024;67:192-331',
            'url': 'https://www.sciencedirect.com/science/article/pii/S1078588423008894'}
ESVS_AAA_NEW = {'title': 'ESVS 2024 guidelines on abdominal aorto-iliac artery aneurysms: what is new? Vascular News 2024',
                'url': 'https://vascularnews.com/esvs-2024-clinical-practice-guidelines-on-the-management-of-abdominal-aortoiliac-artery-aneurysms-whats-new/'}
ESVS_AAA_HL = {'title': 'Highlights from the ESVS 2024 guidelines on abdominal aorto-iliac artery aneurysms. Endovascular Today, March 2024',
               'url': 'https://evtoday.com/articles/2024-mar/highlights-from-the-esvs-2024-clinical-practice-guidelines-on-the-management-of-abdominal-aortoiliac-artery-aneurysms'}
NTA3CT = {'title': 'Baxter BT, et al. Effect of doxycycline on aneurysm growth among patients with small infrarenal abdominal aortic aneurysms (N-TA3CT). JAMA 2020;323:2029-38',
          'url': 'https://www.unmc.edu/newsroom/2020/05/26/landmark-study-shows-no-benefit-of-drug-to-slow-growth-of-aneurysms/'}
LEDERLE = {'title': 'Lederle FA, et al. Rupture rate of large abdominal aortic aneurysms in patients refusing or unfit for elective repair. JAMA 2002;287:2968-72',
           'url': 'https://pure.johnshopkins.edu/en/publications/rupture-rate-of-large-abdominal-aortic-aneurysms-in-patients-refu-6/'}
UKSAT = {'title': 'Powell JT, et al. Final 12-year follow-up of surgery versus surveillance in the UK Small Aneurysm Trial. Br J Surg 2007;94:702-8',
         'url': 'https://academic.oup.com/bjs/article/94/6/702/6142549'}
IMPROVE = {'title': 'IMPROVE trial investigators. Endovascular strategy versus open repair for ruptured AAA: three-year results. BMJ 2017',
           'url': 'https://evtoday.com/news/three-year-improve-results-compare-treatment-strategies-for-ruptured-aaa'}
NAIR = {'title': 'Nair R, Abdool-Carrim ATO, Chetty R, Robbs JV. Arterial aneurysms in patients infected with human immunodeficiency virus. J Vasc Surg 1999;29:600-7',
        'url': 'https://www.sciencedirect.com/science/article/pii/S0741521499703046'}
STEEN = {'title': 'Steenberge S, et al. Natural history and surveillance of isolated common iliac artery aneurysms (Cleveland Clinic). J Vasc Surg 2022 (summary)',
         'url': 'https://consultqd.clevelandclinic.org/guidance-for-surveillance-of-isolated-common-iliac-artery-and-small-abdominal-aortic-aneurysms'}
RAYT = {'title': 'Rayt HS, et al. Buttock claudication and erectile dysfunction after internal iliac artery embolization in patients prior to EVAR. Cardiovasc Intervent Radiol 2008;31:728-34',
        'url': 'https://dx.doi.org/10.1007/s00270-008-9319-3'}
BRESLER = {'title': 'Bresler AM, et al. Image-based assessment of aortoiliac aneurysm anatomy in the global iliac branch study. Langenbecks Arch Surg 2024;409:135',
           'url': 'https://link.springer.com/article/10.1007/s00423-024-03326-8'}


def add(procs, ask, has, LM, S):
    need = ('ciaa-r', 'ibd-r', 'abi-cuff-ankle', 'leg-sfa', 'ali-embolus', 'aaa-infra')
    if not all(has(i) for i in need) or not all(k in procs for k in ('ali-emb', 'aaa-infra', 'aiod-abf', 'aaa-evar')):
        print('  vascular 2: meshes or source operations missing, skipped'); return
    hv = lambda xs: [i for i in xs if has(i)]
    DEFAULT_ON = [i for i, s in S.items() if s.get('visible', True) is not False or s.get('sideVisible')]
    NEW = ['ciaa-r', 'ciaa-thrombus-r', 'ibd-r', 'iia-coils-r', 'eia-ext-r', 'graft-cia-r', 'abi-cuff-ankle', 'abi-cuff-arm', 'doppler-pen']

    def borrow(key, sid):
        s = next(x for x in procs[key]['steps'] if x['id'] == sid)
        return {k: copy.deepcopy(s[k]) for k in ('view', 'show', 'opacity', 'ct', 'pose') if s.get(k) is not None}

    def step(id_, phase, title, body, base, show=(), drop=(), highlight=(), danger=(), labels=(), opacity=None, action=None, quiz=None, after=False, view=None):
        b = copy.deepcopy(base)
        acts = set(action['ids']) if action else set()
        sh = [i for i in hv([*b.get('show', []), *show]) if i not in drop and i not in acts]
        named = set(sh) | set(highlight) | set(danger) | set(labels) | acts
        s = {'id': id_, 'phase': phase, 'title': title, 'body': body, 'view': view or b['view'], 'show': list(dict.fromkeys(sh)),
             'hide': [i for i in DEFAULT_ON if i not in named] + [i for i in NEW if i not in named and has(i)] + [i for i in drop if i not in named],
             'highlight': hv(highlight), 'danger': hv(danger), 'labels': hv(labels), 'opacity': {**b.get('opacity', {}), **(opacity or {})}}
        if b.get('ct'): s['ct'] = b['ct']
        if b.get('pose'): s['pose'] = b['pose']
        if action: s['action'] = action
        if quiz: s['ask'] = quiz
        if after: s['askAfter'] = True
        return s

    def view(t, d, dist):
        d = U(V(d)); t = V(t); return {'eye': R(t + d * dist), 'target': R(t)}

    def proc(key, op, opName, approach, summary, groups, sources, before=None):
        steps, sq = [], []
        for i, (lab, kind, sts) in enumerate(groups):
            sq.append({'label': lab, 'kind': kind})
            for s in sts: steps.append({**s, 'seq': i})
        p = {'id': key, 'op': op, 'opName': opName, 'side': 'both', 'name': opName, 'approach': approach, 'summary': summary,
             'ports': [], 'steps': steps, 'sources': sources, 'group': 'Vascular', 'sequence': sq}
        items = [(k, v) for k, v in procs.items() if k != key]
        if before and before in procs:
            i = [k for k, _ in items].index(before); items.insert(i, (key, p))
        else:
            items.append((key, p))
        procs.clear(); procs.update(items)

    consent = lambda risk, common, serious, specific, alts, recovery, kenya: (
        f'<p><b>1. The patient\'s own risk</b>: {risk} Quote the figure, not a textbook average.</p>'
        + h4('2. Common') + ul(*common) + h4('3. Serious') + ul(*serious) + h4('4. Specific to this operation') + ul(*specific)
        + h4('5. Alternatives') + ul(*alts) + h4('6. Recovery') + ul(*recovery) + h4('7. Kenya-specific') + ul(*kenya)
        + f'<p>Template and how to use it: {link("approach=cticu-consent&amp;step=0", "CTICU protocol, consent")}.</p>')
    icu_head = (f'<p>Start with the {link("approach=cticu-vascular&amp;step=0", "vascular core")}, the {link("approach=cticu-core&amp;step=0", "lab schedule")} '
                f'and the {link("approach=cticu-core&amp;step=2", "escalation table")}; then this operation\'s own points.</p>')
    doses = f'<p>Doses: {link("approach=cticu-doses&amp;step=0", "electrolytes")}, {link("approach=cticu-doses&amp;step=1", "vasoactive drugs")}.</p>'

    # ======================================================================================== 1. PAD: best medical therapy
    LEG = borrow('ali-emb', 'ali-anatomy')
    AO = borrow('aiod-abf', 'ao-anatomy')
    TREE = {'view': view(V(30.0, 60.0, -770.0), (0.22, 1.0, 0.1), 1650), 'show': [*AO['show'], *LEG['show'], 'aorta'], 'opacity': {**AO['opacity'], **LEG['opacity']}, 'ct': AO.get('ct')}
    ankle = V(LM['abi-ankle']) if 'abi-ankle' in LM else V(48.8, 46.0, -1220.0)
    arm = V(LM['abi-arm']) if 'abi-arm' in LM else V(211.0, -22.0, -177.0)
    ABI_V = {'view': view(ankle + V(0, 20, 30), (0.35, 1.0, 0.3), 380), 'show': LEG['show'], 'opacity': LEG['opacity']}

    pad_patho = step('pad-patho', 'Pathophysiology', 'Peripheral arterial disease: a marker of the whole arterial tree',
        '<p><b>Atherosclerosis of the leg arteries</b> narrows or blocks the inflow; at rest the collaterals cope, on walking the muscle outgrows its supply and becomes ischemic: <b>claudication</b>, '
        'cramping pain in the muscle below the block, relieved by standing still. When even resting flow fails: <b>rest pain, ulcers, gangrene</b> (chronic limb-threatening ischemia, CLTI).</p>'
        + chain('Smoking, diabetes, hypertension, dyslipidemia, CKD, age', 'Plaque in the leg arteries', 'Exercise outstrips supply', '!Claudication', 'Rest pain, tissue loss (CLTI)')
        + '<p><b>The same plaque is in the coronary and carotid arteries.</b> A claudicant is far more likely to have a heart attack or stroke than to lose the leg: '
          'the first aim of treatment is to keep the patient alive, the second to let them walk. ACC/AHA 2024 describes four clinical subsets: <b>asymptomatic, chronic symptomatic (claudication), CLTI and acute limb ischemia</b>.</p>'
        + '<p>In Kenya also think of <b>HIV-associated vasculopathy</b> (young, occlusive or aneurysmal, presents late) and <b>Takayasu arteritis</b> in young women.</p>'
        + ev('ACC/AHA 2024 (Gornik et al.): four clinical subsets; PAD care is aimed at major adverse cardiovascular and limb events. ESVS 2024 (Nordanstig et al.): claudication rarely progresses to limb loss; cardiovascular risk dominates. HIV vasculopathy: Robbs and Paruk 2010.'),
        TREE, highlight=['leg-sfa', 'cia-r', 'cia-l'], labels=['aorta', 'cia-r', 'cfa-r', 'leg-sfa', 'leg-pop', 'leg-ata'], after=True,
        quiz=ask('What is the commonest cause of death in a patient with intermittent claudication?', 'Myocardial infarction or stroke',
                 'PAD is a marker of systemic atherosclerosis; cardiovascular events, not limb loss, dominate the prognosis. That is why every patient needs best medical therapy.',
                 'Sepsis from an infected, gangrenous foot', 'Ruptured aortic aneurysm', 'Pulmonary embolism'))

    pad_anat = step('pad-anat', 'Anatomy', 'Where is the block? Read it from the pain',
        tbl(['Level of disease', 'Where it hurts on walking', 'Pulses'],
            ['Aorto-iliac (inflow)', 'Buttock, hip and thigh; erectile dysfunction (Leriche)', 'Weak or absent femoral pulses'],
            ['Femoropopliteal (the commonest: the superficial femoral in the adductor canal)', '<b>Calf</b>', 'Femoral present, popliteal and foot pulses absent'],
            ['Tibial (diabetes, renal failure)', 'Foot (often silent until ulcer or gangrene; neuropathy masks pain)', 'Popliteal present, foot pulses absent'])
        + '<p>Claudication is felt <b>one level below</b> the block, in the muscle supplied beyond it. Treat the <b>inflow before the outflow</b> if both are diseased.</p>',
        TREE, show=['sfa-occlusion'], highlight=['sfa-occlusion'], labels=['cia-r', 'cfa-r', 'leg-sfa', 'leg-pop', 'leg-ata', 'leg-pta', 'sfa-occlusion'])

    pad_case = step('pad-case', 'Case', 'Calf pain at 150 metres',
        '<p>A <b>58-year-old matatu driver</b>, smokes 15 cigarettes a day, type 2 diabetes for 8 years, blood pressure 156/94. Right calf pain after about <b>150 m</b>, gone within 2 minutes of standing. '
        'No rest pain, no ulcers. Right femoral pulse present; popliteal and foot pulses absent. <b>ABI 0.62 on the right</b>, 0.95 on the left. LDL 3.8 mmol/L, HbA1c 8.9%.</p>',
        TREE, show=['sfa-occlusion'], highlight=['sfa-occlusion'], labels=['sfa-occlusion', 'leg-pop'],
        quiz=ask('What is the first-line management?', 'Best medical therapy and a structured walking programme',
                 'Claudication without CLTI is treated medically first; revascularization is considered only if symptoms remain lifestyle-limiting despite therapy and exercise.',
                 'Femoropopliteal bypass with reversed saphenous vein this month', 'Angioplasty and stenting of the superficial femoral artery', 'Pentoxifylline and review in one year'))

    pad_abi = step('pad-abi', 'Decision', 'The ankle-brachial index: how to do it, how to read it',
        '<p>Patient supine for 5–10 minutes. With a cuff and an 8 MHz hand-held Doppler, take the systolic pressure in <b>both brachial arteries</b>, then the <b>dorsalis pedis and posterior tibial</b> of each leg.</p>'
        + '<p><b>ABI for each leg = the higher ankle pressure of that leg ÷ the higher of the two brachial pressures.</b></p>'
        + tbl(['ABI', 'Meaning (ACC/AHA 2024)'],
              ['1.00–1.40', 'Normal'], ['0.91–0.99', 'Borderline'], ['<b>0.90 or less</b>', '<b>PAD</b>'],
              ['Over 1.40', '<b>Non-compressible</b> (calcified: diabetes, renal failure): measure the <b>toe-brachial index</b>; 0.70 or less is PAD'])
        + ul('Normal resting ABI but typical symptoms: an <b>exercise ABI</b> (a fall after walking confirms PAD)',
             'ABI does not diagnose CLTI: in suspected CLTI use toe pressures, TcPO₂ and imaging',
             'Imaging (duplex, CTA) only when it will change management, usually when revascularization is being planned')
        + ev('ACC/AHA 2024 via AAFP 2025: ABI ≤0.90 abnormal, 0.91–0.99 borderline, 1.00–1.40 normal, >1.40 non-compressible; TBI ≤0.70 diagnostic; ABI and TBI are not useful to diagnose CLTI.'),
        ABI_V, show=['abi-cuff-ankle', 'leg-dpa', 'leg-pta'], highlight=['leg-dpa', 'leg-pta'], labels=['doppler-pen', 'leg-dpa', 'leg-pta', 'abi-cuff-ankle'],
        action={'kind': 'reveal', 'label': 'Place the Doppler on the dorsalis pedis', 'port': 'groin-r', 'ids': ['doppler-pen']}, after=True,
        quiz=ask('Brachial pressures 150 and 140 mmHg; right ankle: dorsalis pedis 80, posterior tibial 96 mmHg. What is the right ABI?', '0.64',
                 'Higher ankle pressure (96) over the higher brachial pressure (150) = 0.64: moderate PAD.', '0.53', '0.69', '0.80'))

    pad_drugs = step('pad-drugs', 'Treatment', 'Best medical therapy: the drugs and the targets',
        tbl(['What', 'How', 'Target or note'],
            ['<b>Stop smoking</b>', 'Advice at every visit; <b>varenicline</b> (or bupropion, nicotine replacement); no tobacco in any form', 'The single most effective treatment for the leg'],
            ['<b>Antiplatelet</b>', 'Single agent: <b>clopidogrel 75 mg</b> (preferred in symptomatic PAD) or aspirin 75–100 mg', 'Lifelong'],
            ['<b>Low-dose rivaroxaban</b>', '<b>2.5 mg twice daily with aspirin 75–100 mg</b>, if bleeding risk is low', 'Fewer MACE and major limb events (COMPASS; VOYAGER PAD after revascularization)'],
            ['<b>High-intensity statin</b>', 'Atorvastatin 40–80 mg or rosuvastatin 20–40 mg; add ezetimibe (then a PCSK9 inhibitor) if not at target', '<b>LDL under 1.4 mmol/L (55 mg/dL) and a fall of at least 50%</b> (ESVS 2024)'],
            ['<b>Blood pressure</b>', 'ACE inhibitor or ARB first', 'Under 130/80 mmHg'],
            ['<b>Diabetes</b>', 'Metformin; add an <b>SGLT2 inhibitor</b> or <b>GLP-1 agonist</b> for cardiovascular benefit', 'Individualized HbA1c; daily foot checks'],
            ['<b>Cilostazol</b> (for claudication)', '100 mg twice daily; benefit judged at 3 months', '<b>Contraindicated in heart failure</b>; improves walking distance, not survival'])
        + ul('Vaccination (influenza, pneumococcal) and foot care are part of the package',
             'Kenya: aspirin, clopidogrel, atorvastatin, ACE inhibitors and metformin are available as lower-cost generics; rivaroxaban, ezetimibe, SGLT2 inhibitors and GLP-1 agonists may not be affordable: prioritize stopping smoking, an antiplatelet and a statin' + tag('prop', 'proposed'))
        + ev('ACC/AHA 2024: single antiplatelet (clopidogrel), rivaroxaban 2.5 mg twice daily with low-dose aspirin for those not at increased bleeding risk, high-intensity statin, antihypertensive and diabetes therapy, smoking cessation (AHA top things to know; AAFP 2025). ESVS 2024: LDL <1.4 mmol/L and ≥50% reduction; varenicline first-line. COMPASS (Lancet 2018), VOYAGER PAD (NEJM 2020). Cilostazol reduces claudication symptoms (ACC/AHA 2024); it is contraindicated in heart failure (US label).'),
        TREE, show=['sfa-occlusion'], highlight=['sfa-occlusion'], labels=['sfa-occlusion'], after=True,
        quiz=ask('A claudicant has heart failure with an ejection fraction of 30%. Which drug for claudication must be avoided?', 'Cilostazol',
                 'Cilostazol, a phosphodiesterase-3 inhibitor, is contraindicated in heart failure of any severity.', 'Atorvastatin', 'Clopidogrel', 'Ramipril'))

    pad_ex = step('pad-exercise', 'Treatment', 'Structured exercise: the treatment that works for the walking',
        '<p><b>Supervised exercise therapy</b>: walking (treadmill or track) to <b>moderate or near-maximal claudication pain</b>, rest until it settles, walk again. '
        '<b>30–45 minutes, at least 3 times a week, for at least 12 weeks.</b> Walking distance improves by building collaterals and muscle efficiency.</p>'
        + ul('Where supervised programmes are not available, a <b>structured home or community walking programme</b> with a target and a log', 'Review at 3 months: walking distance, ABI, quality of life',
             'Combined with best medical therapy, exercise is the standard of care before any revascularization for claudication')
        + ev('ACC/AHA 2024 via AAFP 2025: structured exercise at least 3 times a week for 12 weeks improves walking distance, function and quality of life. ESVS 2024 claudication guideline (Nordanstig et al.).'),
        TREE, show=['sfa-occlusion'], labels=['leg-sfa', 'leg-pop'])

    pad_revasc = step('pad-revasc', 'Decision', 'When to refer for revascularization',
        tbl(['Situation', 'What to do'],
            ['Asymptomatic PAD', 'Best medical therapy only: <b>never revascularize to prevent progression</b>'],
            ['Claudication', 'Best medical therapy and exercise for at least 3 months; revascularize only if <b>still lifestyle-limiting</b>, by shared decision, with favourable anatomy'],
            ['<b>CLTI</b> (rest pain, ulcer, gangrene)', '<b>Prompt vascular referral</b>: imaging and revascularization to save the limb; WIfI staging; multidisciplinary foot care'],
            ['<b>Acute limb ischemia</b>', 'Heparin and emergency referral: ' + link('approach=ali-iia&amp;step=0', 'Rutherford IIa') + ', ' + link('approach=ali-emb&amp;step=0', 'IIb')])
        + f'<p>The operations: {link("approach=aiod-abf&amp;step=0", "aorto-iliac")}, {link("approach=fp-gsv&amp;step=0", "femoropopliteal bypass")}, {link("approach=amp-levels&amp;step=0", "amputation levels")}.</p>'
        + ev('ACC/AHA 2024: revascularization prevents limb loss in CLTI; for claudication only if lifestyle-limiting despite therapy. ESVS 2024: individualized revascularization for claudication; none for asymptomatic disease. Global Vascular Guidelines 2019 for CLTI (WIfI).'),
        TREE, show=['sfa-occlusion'], highlight=['sfa-occlusion'], labels=['sfa-occlusion', 'cia-r'], after=True,
        quiz=ask('A 70-year-old with an ABI of 0.75 has no leg symptoms. What do you offer?', 'Best medical therapy alone',
                 'Asymptomatic PAD carries cardiovascular risk but no limb threat; prophylactic revascularization is not indicated.', 'Angioplasty to prevent progression to rest pain', 'Bypass with the saphenous vein', 'Nothing, since there are no symptoms'))

    pad_counsel = step('pad-counsel', 'Counselling', 'What to tell the patient, and follow-up',
        ul('This is a disease of all the arteries: the tablets are for the heart and brain as much as the leg, and are for life',
           'Stopping smoking is the most important thing you can do for your leg',
           'Walk every day, through the pain to a strong ache, rest, then walk again',
           'Check your feet every day; any wound, colour change or pain at night: come back the same week',
           'Follow-up: every 6–12 months, with ABI, blood pressure, lipids, HbA1c, smoking status and adherence') + tag('prop', 'proposed'),
        TREE, labels=['leg-sfa'])

    proc('pad-bmt', 'pad', 'Peripheral arterial disease', 'Best medical therapy (ACC/AHA and ESVS 2024)',
         'Peripheral arterial disease as a marker of the whole arterial tree: ABI, best medical therapy, exercise, and when to revascularize.',
         [('Patho', 'other', [pad_patho]), ('Anatomy', 'other', [pad_anat]), ('Case', 'other', [pad_case]), ('ABI', 'other', [pad_abi]),
          ('Drugs', 'other', [pad_drugs]), ('Exercise', 'other', [pad_ex]), ('Refer', 'other', [pad_revasc]), ('Counsel', 'other', [pad_counsel])],
         [ACC24, AHA_TOP, AAFP, ESVS_PAD, GVG, COMPASS, VOYAGER, TASC, HIV])

    # ======================================================================================== 2. Acute limb ischemia, Rutherford IIa
    A = {s['id']: s for s in procs['ali-emb']['steps']}
    CASE = borrow('ali-emb', 'ali-case')
    def reuse(src, id_, title=None, body=None, quiz=None, after=False):
        s = copy.deepcopy(A[src]); s['id'] = id_
        if title: s['title'] = title
        if body: s['body'] = body
        s.pop('ask', None); s.pop('askAfter', None); s.pop('lead', None)
        if quiz: s['ask'] = quiz
        if after: s['askAfter'] = True
        return s

    ali_patho = reuse('ali-patho', 'iia-patho', 'Pathophysiology: the Rutherford grades and the clock',
        '<p>Acute limb ischemia is a sudden loss of perfusion (within 2 weeks) that threatens the leg. <b>Nerve and muscle tolerate severe ischemia for about 6 hours</b>; '
        'what separates the grades is whether <b>sensation and movement</b> are still there.</p>'
        + tbl(['Rutherford', 'Sensory loss', 'Weakness', 'Doppler (artery / vein)', 'Action'],
              ['I viable', 'none', 'none', 'audible / audible', 'urgent imaging, revascularize'],
              ['<b>IIa marginally threatened</b>', '<b>none, or the toes only</b>', '<b>none</b>', 'inaudible / audible', '<b>revascularize urgently</b> (same day, within hours)'],
              ['IIb immediately threatened', 'beyond the toes, rest pain', 'mild to moderate', 'inaudible / audible', 'emergency revascularization'],
              ['III irreversible', 'anesthetic', 'paralysis, rigor', 'inaudible / inaudible', 'amputation'])
        + '<p><b>IIa is salvageable with a little time</b>: time to image if needed and to choose the best tool, but not to wait overnight. Re-examine sensation and movement every hour: '
          'any new numbness above the toes or weakness makes it IIb.</p>'
        + ev('Rutherford 1997 categories as adopted by ESVS 2020 (Björck et al.).'),
        quiz=ask('A cold, pulseless foot: the patient feels light touch on the toes and moves the ankle and toes normally. Arterial Doppler absent, venous present. Grade?', 'Rutherford IIa',
                 'No (or toe-only) sensory loss and no weakness with an inaudible arterial signal is IIa, marginally threatened: urgent revascularization.', 'Rutherford I', 'Rutherford IIb', 'Rutherford III'), after=True)
    ali_anat = reuse('ali-anatomy', 'iia-anatomy')
    ali_case = step('iia-case', 'Case', 'A cold foot, still moving and feeling',
        '<p>A <b>46-year-old woman</b> with rheumatic mitral stenosis and AF, INR 1.3 (warfarin stopped 3 weeks ago). Sudden pain and coldness of the right leg <b>4 hours</b> ago. '
        'Right foot pale and cool; <b>no femoral pulse</b>; left leg normal. <b>Light touch and pinprick normal; she moves her toes and ankle fully.</b> No arterial Doppler signal at the ankle; venous signal present.</p>',
        CASE, highlight=['ali-embolus'], labels=['ali-embolus', 'cfa-r', 'leg-sfa', 'pfa-r'],
        quiz=ask('What is the best treatment?', 'Heparin, then femoral Fogarty embolectomy today',
                 'An abrupt onset, AF, a normal opposite leg and a lost femoral pulse mean an embolus at the femoral bifurcation in a healthy artery: balloon embolectomy is the standard treatment. IIa allows imaging but must not wait overnight.',
                 'Catheter-directed thrombolysis over the next 24–48 hours', 'Heparin alone and reassess the leg tomorrow morning', 'Primary above-knee amputation'))
    ali_decide = step('iia-decide', 'Decision', 'Rutherford IIa: what is the best therapy?',
        ul('<b>Heparin at once</b>: unfractionated heparin 70–100 IU/kg (or 5000 IU) bolus, then an infusion by aPTT',
           '<b>Imaging if it does not delay treatment</b>: CTA is first-line. In IIa there is usually time for it; a clear embolus (AF, normal other leg) can go to theatre on clinical grounds',
           'Then match the tool to the cause:')
        + tbl(['', '<b>Embolus</b>', '<b>Thrombosis in situ</b> (or a blocked graft or stent)'],
              ['Clues', 'Sudden onset; AF, mitral stenosis, recent MI; <b>normal pulses in the other leg</b>; no claudication before', 'Claudication before; diseased arteries; weak pulses in the other leg; a previous bypass'],
              ['Best therapy', '<b>Surgical balloon (Fogarty) embolectomy</b>, usually through the femoral bifurcation under local or regional anesthesia; completion angiogram', '<b>Catheter-directed thrombolysis</b> (an alternative to surgery in IIa) or percutaneous thrombectomy, then treat the culprit lesion; or bypass'],
              ['Watch for', 'Residual clot in the tibial vessels (completion imaging)', 'Bleeding (major bleeding about 8–10% with thrombolysis); thrombolysis takes 24–48 hours, too slow for IIb with motor loss'])
        + ul('<b>Completion angiography</b> after open or endovascular treatment', 'Fasciotomy is <b>not routine</b> in IIa: short ischemia, no motor loss. Examine the compartments after reperfusion and open them if they are tense or painful on stretch',
             'Intravenous (systemic) thrombolysis has no place')
        + ev('ESVS 2020: heparin while awaiting revascularization (Rec 9, I-C); imaging recommended if it does not delay treatment (Rec 5, I-C); CTA first-line (Rec 6, I-B); catheter-directed thrombolysis as an alternative to surgery in Rutherford IIa (Rec 24, I-A); IV thrombolysis not recommended (Rec 22, III-A); completion angiography (Rec 18, I-C); balloon thrombo-embolectomy is the standard treatment of embolic ALI. Endovascular Today 2026: heparin 70–100 IU/kg or 5000 IU; open embolectomy optimal for acute occlusion of relatively normal arteries; CDT takes 24–48 h with 8–10% major bleeding.'),
        CASE, highlight=['ali-embolus'], labels=['ali-embolus', 'leg-pop', 'leg-tpt'], after=True,
        quiz=ask('A man with a femoropopliteal vein bypass done 2 years ago has a cold foot for 10 hours, normal sensation and movement, no Doppler signal. CTA shows an occluded graft. Best option?',
                 'Catheter-directed thrombolysis, then fix the culprit stenosis',
                 'A thrombosed graft in Rutherford IIa with time available suits catheter-directed thrombolysis (ESVS Rec 24, I-A), which also reveals the culprit lesion to fix.',
                 'Fogarty embolectomy through the groin, with no imaging of the graft', 'Intravenous alteplase on the ward with heparin', 'Above-knee amputation, as the limb is not viable'))
    ali_cons = step('ali-iia-consent', 'Consent', 'Consent: what to discuss with this patient',
        consent('The cause (embolus or thrombosis), the heart (AF, valve, recent MI), renal function, time since onset.',
                ['Groin wound problems', 'Swelling of the leg after the blood returns'],
                ['Clot left behind needing more surgery or thrombolysis', 'Compartment syndrome needing fasciotomy', 'Bleeding (more with thrombolysis)', 'Kidney injury from reperfusion or contrast', 'Loss of the leg', 'Death (the heart is often the cause)'],
                ['Lifelong anticoagulation afterwards, because the source is usually the heart', 'Echo and a search for the source'],
                ['Thrombolysis (for thrombosis in situ)', 'Amputation if the leg cannot be saved', 'No operation, and what that means'],
                ['Walking within days if the leg recovers', 'Follow-up for the heart and the anticoagulation'],
                ['Warfarin needs INR checks: plan where and how she will afford them', 'Rheumatic AF: warfarin, not a DOAC']),
        CASE, labels=[], after=True,
        quiz=ask('Why must the consent mention long-term anticoagulation?', 'Most emboli come from the heart and recur without it',
                 'Recurrent embolism is common when the cardiac source is not treated; warfarin is the standard in rheumatic AF.', 'The arteriotomy needs anticoagulation to stay open', 'It is only needed if a stent is placed during surgery', 'It is routine for every operation on an artery'))
    ali_expose = reuse('ali-expose', 'iia-expose', body='<p>Under <b>local anesthesia</b> (often enough in IIa; anesthetist on standby), a vertical groin incision over the femoral artery. Control the <b>common femoral, superficial femoral and profunda</b> with slings. '
                       'Heparin is already running. A <b>transverse arteriotomy</b> just above the bifurcation; longitudinal if the artery is diseased and will need a patch.</p>')
    ali_fog = reuse('ali-fogarty', 'iia-fogarty')
    ali_clear = reuse('ali-clear', 'iia-clear', body='<p>Inflow: brisk pulsatile flow after the proximal pass. Back-bleeding from the superficial femoral and profunda is not proof of a clear run-off: '
                      '<b>completion angiography</b> on the table. Close the arteriotomy, release the clamps and check the foot: colour, capillary refill, Doppler signals at the ankle.</p>'
                      + ev('ESVS 2020 Rec 18 (I-C): completion angiography after open or endovascular treatment of ALI.'))
    ali_comp = step('iia-compartments', 'Compartments', 'Fasciotomy: not routine in IIa, but look',
        '<p>With short ischemia and no motor loss, <b>prophylactic fasciotomy is not needed</b>. After reperfusion, examine the leg hourly for 24 hours: '
        '<b>pain out of proportion, pain on passive stretch</b>, a tense swollen calf, numbness in the first web space (deep peroneal nerve). Any of these: a four-compartment fasciotomy through two incisions, without delay.</p>'
        + f'<p>How to do it: {link("approach=ali-emb&amp;step=7", "four-compartment fasciotomy")}.</p>'
        + ev('Endovascular Today 2026: compartment syndrome in up to 30% of revascularizations; prophylactic fasciotomy for severe ischemia over 6 hours; otherwise close compartment checks.'),
        borrow('ali-emb', 'ali-fasciotomy'), labels=['xs-ant', 'xs-lat', 'xs-sup', 'xs-deep'], highlight=['xs-ant'],
        quiz=ask('Six hours after an embolectomy the calf is tense and dorsiflexing the toes causes severe pain. Next?', 'Four-compartment fasciotomy now',
                 'Pain on passive stretch and a tense calf mean compartment syndrome; fasciotomy should not wait for pressure measurements when the signs are clear.', 'Elevate the leg and give morphine', 'Duplex scan in the morning', 'Restart heparin'))
    ali_icu = step('ali-iia-icu', 'ICU', 'ICU and post-operative care',
        icu_head + h4('Specific to this operation')
        + ul('Leg checks hourly for 24 hours: pulses or Doppler signals, colour, sensation, movement, compartments',
             'Reperfusion: potassium, CK, creatinine and urine colour (myoglobin); keep urine output up',
             'Continue heparin after surgery unless there is bleeding, then <b>long-term anticoagulation</b>',
             '<b>Rheumatic AF or mitral stenosis: warfarin (INR 2–3)</b>. Do not use rivaroxaban: it was worse than a vitamin K antagonist in INVICTUS',
             '<b>Find the source</b>: ECG, echocardiography (left atrial thrombus, valve, LV function); if no cardiac source, image the aorta and screen for thrombophilia or cancer in the young',
             'HIV test if no other cause in a young patient' + tag('prop', 'proposed'))
        + h4('Labs') + ul('Day 1: FBC, UEC, CK, K⁺, coagulation; repeat K⁺ and CK at 6–12 hours') + doses
        + ev('INVICTUS (Connolly et al., NEJM 2022; 4531 patients): rivaroxaban 8.2% vs VKA 6.5% for the primary composite, with more ischemic strokes and vascular deaths; VKA remains standard in rheumatic AF. Endovascular Today 2026: echo for new AF; full aortic imaging if aortic source suspected; thrombophilia or cancer work-up in selected patients.'),
        borrow('ali-emb', 'ali-case'), labels=['cfa-r'], after=True,
        quiz=ask('After embolectomy, a woman with rheumatic mitral stenosis and AF needs long-term anticoagulation. Which drug?', 'Warfarin, target INR 2–3',
                 'In rheumatic heart disease-associated AF, rivaroxaban was inferior to a VKA (INVICTUS); DOACs are not used in moderate-severe mitral stenosis.', 'Rivaroxaban 20 mg once daily with food', 'Aspirin 75 mg daily alone, lifelong', 'Apixaban 5 mg twice daily, no monitoring'))
    proc('ali-iia', 'ali', 'Acute limb ischemia', 'Rutherford IIa: embolectomy (sensation and movement preserved)',
         'Marginally threatened acute limb ischemia: heparin, imaging if it does not delay, embolectomy for embolus or thrombolysis for thrombosis, completion imaging, the source and anticoagulation.',
         [('Patho', 'other', [ali_patho]), ('Anatomy', 'other', [ali_anat]), ('Case', 'other', [ali_case]), ('Decision', 'other', [ali_decide]), ('Consent', 'other', [ali_cons]),
          ('Expose', 'other', [ali_expose]), ('Fogarty', 'other', [ali_fog]), ('Clear', 'other', [ali_clear]), ('Compartments', 'other', [ali_comp]), ('ICU', 'other', [ali_icu])],
         [ESVS_ALI, EVT26, INVICTUS, *procs['ali-emb']['sources'][2:]], before='ali-emb')

    # ======================================================================================== 3. AAA: management and surveillance
    AP = borrow('aaa-infra', 'ai-patho'); AN = borrow('aaa-infra', 'ai-anatomy'); EVR = borrow('aaa-evar', 'ae-deploy')
    aaa_patho = step('aaam-patho', 'Pathophysiology', 'Who gets an AAA, and why it grows',
        '<p>An abdominal aortic aneurysm is an infrarenal aorta of <b>3.0 cm or more</b> (or 1.5 times normal). The media fails: elastin is broken down, smooth muscle cells are lost, the adventitia is inflamed. '
        'By Laplace\'s law the wall tension rises with the radius, so <b>growth and rupture risk accelerate with size</b>.</p>'
        + tbl(['Risk factors for AAA', 'Risk factors for faster growth or rupture'],
              ['<b>Age, male sex, smoking</b> (the strongest modifiable), family history (first-degree relative), hypertension, atherosclerosis elsewhere, white ethnicity', '<b>Current smoking</b>, larger diameter, <b>female sex</b> (rupture at smaller sizes), hypertension, COPD'],
              ['Diabetes is <b>protective</b>', 'Infection or HIV (saccular, fast-growing, multiple)'])
        + '<p><b>Early onset (under 60) or a family history</b>: offer genetic evaluation. In Africa, consider <b>infective (mycotic) and HIV-associated aneurysms</b>: younger patients, saccular, atypical sites.</p>'
        + ev('ESVS 2024 (Wanhainen et al.): definition, risk factor management (Rec 16, I); genetic evaluation for early onset or positive family history (Rec 154, I). HIV-associated aneurysms: Nair et al., J Vasc Surg 1999.'),
        AP, highlight=['aaa-infra'], labels=['aaa-infra', 'renal-a-l', 'cia-r'], after=True,
        quiz=ask('Which risk factor is associated with a LOWER prevalence of AAA?', 'Diabetes mellitus',
                 'Diabetes is consistently associated with fewer and slower-growing aneurysms, possibly through glycation stiffening the wall.', 'Cigarette smoking', 'Male sex over 65', 'A brother with an AAA'))
    aaa_anat = step('aaam-anat', 'Anatomy', 'Measuring the aorta',
        tbl(['Diameter (infrarenal)', 'Meaning'], ['About 1.5 cm (women), 1.7 cm (men) over 50', 'Normal'], ['2.5–2.9 cm', 'Sub-aneurysmal (ectatic)'], ['<b>3.0 cm or more</b>', '<b>Aneurysm</b>'],
            ['5.5 cm men, 5.0 cm women', 'Usual repair threshold'])
        + ul('<b>Ultrasound</b> for diagnosis and surveillance: maximum anteroposterior diameter, perpendicular to the aortic axis', 'CTA when the threshold is reached, for planning (neck length and angle, iliacs, access)',
             'Look at the <b>iliac arteries</b> too: up to 40% of AAA patients have a common iliac aneurysm (' + link('approach=ciaa-repair&amp;step=0', 'common iliac aneurysm') + ')')
        + ev('ESVS 2024: diameter primarily by ultrasound; CTA for planning once the threshold is met (Endovascular Today 2024). Common iliac aneurysm in up to 40% of AAA: Bresler et al. 2024.'),
        AN, show=['aaa-infra'], highlight=['aaa-infra'], labels=['aaa-infra', 'renal-a-l', 'cia-r', 'cia-l'], opacity={'aaa-infra': 0.55})
    aaa_case = step('aaam-case', 'Case', 'A 4.6 cm aneurysm found on an ultrasound',
        '<p>A <b>66-year-old man</b>, ex-smoker of 30 pack-years (still smokes 5 a day), hypertension on amlodipine, BP 152/90. An abdominal ultrasound for gallstones shows an <b>infrarenal aneurysm of 4.6 cm</b>. '
        'He has no abdominal or back pain; the aorta is not tender. His brother had a "burst artery" at 70.</p>',
        AP, highlight=['aaa-infra'], labels=['aaa-infra'],
        quiz=ask('What is the plan?', 'Yearly ultrasound and risk factor control',
                 'A 4.0–4.9 cm AAA in a man is followed yearly (ESVS 2024 Rec 13) with full cardiovascular risk management (Rec 16); repair below 5.5 cm in an asymptomatic man is not recommended (Rec 20).',
                 'Elective repair now because of his family history', 'Doxycycline to slow growth, scan in two years', 'Repeat ultrasound in five years'))
    aaa_surv = step('aaam-surveil', 'Decision', 'Surveillance: how often to scan',
        tbl(['Diameter', 'Men (Rec 13)', 'Women (Rec 14)'],
            ['25–29 mm (sub-aneurysmal)', 'every 5 years', 'every 5 years'], ['30–39 mm', 'every 3 years', 'every 3 years'],
            ['40–44 mm', 'every year', 'every year'], ['45–49 mm', 'every year', '<b>every 6 months</b>'], ['50–54 mm', '<b>every 6 months</b>', '(at threshold: plan repair)'])
        + ul('Consider stopping surveillance in patients who would never be offered repair (age, comorbidity, preference)', 'Screening: ultrasound of <b>high-risk groups</b> (Class I, Level A); which groups is set locally. There is no support for screening women')
        + ev('ESVS 2024 Rec 13 and 14 (IIa): intervals by sex and diameter, taking into account life expectancy, suitability for repair and preference; Rec 11 (I): screening in high-risk populations; stopping surveillance IIa-C (Endovascular Today 2024).'),
        AP, highlight=['aaa-infra'], labels=['aaa-infra'], after=True,
        quiz=ask('A 72-year-old woman has a 4.7 cm AAA. How often should she be scanned?', 'Every 6 months',
                 'Women rupture at smaller diameters: ESVS 2024 recommends 6-monthly surveillance from 45 mm (Rec 14), with repair considered at 50 mm.', 'Every 12 months', 'Every 3 years', 'Repair now at 4.7 cm'))
    aaa_bmt = step('aaam-bmt', 'Treatment', 'The controls: medical management of a small AAA',
        tbl(['What', 'Why'],
            ['<b>Stop smoking</b>', 'Smoking speeds growth and rupture'],
            ['<b>Blood pressure</b> control', 'Cardiovascular risk; wall stress'],
            ['<b>Statin</b> and <b>antiplatelet</b>', 'Most patients die of heart disease, not the aneurysm'],
            ['Lifestyle advice', 'Exercise and sexual activity need <b>not</b> be restricted'])
        + ul('<b>No drug slows aneurysm growth</b>: doxycycline failed in a randomized trial; no recommendation on metformin yet',
             'Fluoroquinolones are not contraindicated by a small AAA',
             'Tell the patient the symptoms of rupture (sudden abdominal or back pain, collapse) and to call for help at once')
        + ev('ESVS 2024: Rec 16 (I) cardiovascular risk factor management with smoking cessation, BP control, statin and antiplatelet therapy and lifestyle advice; Rec 18 (III) fluoroquinolones not contraindicated; Rec 19 (III) no restriction of exercise or sexual activity; no recommendation on metformin. N-TA3CT (JAMA 2020; 254 patients): doxycycline 100 mg twice daily for 2 years did not slow growth.'),
        AP, highlight=['aaa-infra'], labels=['aaa-infra'], after=True,
        quiz=ask('Which drug has been shown in a randomized trial NOT to slow AAA growth?', 'Doxycycline',
                 'N-TA3CT randomized 254 patients with small AAA to doxycycline or placebo for 2 years: no reduction in growth.', 'Atorvastatin', 'Aspirin', 'Amlodipine'))
    aaa_repair = step('aaam-repair', 'Decision', 'When to repair, and how',
        tbl(['Indication', 'Detail'],
            ['<b>Diameter</b>', '<b>5.5 cm in men, 5.0 cm in women</b>; repair below these is <b>not</b> recommended for an asymptomatic fusiform AAA'],
            ['<b>Rapid growth</b>', 'About <b>1 cm a year</b> or more, confirmed'],
            ['<b>Symptoms</b>', 'Tenderness, abdominal or back pain: a brief period of assessment and optimisation, then <b>urgent repair</b> (ideally in working hours)'],
            ['<b>Rupture</b>', 'Emergency (below)'],
            ['Saccular, infective (mycotic)', 'Repair at smaller sizes; mycotic aneurysms need antibiotics and in-situ or extra-anatomic repair'])
        + h4('EVAR or open?')
        + ul('<b>EVAR is the preferred modality for most patients</b> with suitable anatomy (lower early mortality; similar long-term survival); only inside the device instructions for use, with lifelong surveillance (CTA at 30 days)',
             '<b>Open repair</b> for young fit patients with long life expectancy, hostile necks, or when follow-up cannot be guaranteed',
             'Centres should do at least 30 repairs a year (15 open, 15 EVAR)')
        + f'<p>The operations: {link("approach=aaa-infra&amp;step=0", "open infrarenal repair")}, {link("approach=aaa-evar&amp;step=0", "EVAR")}, {link("approach=aaa-juxta&amp;step=0", "juxtarenal")}.</p>'
        + ev('ESVS 2024: thresholds 55 mm men, 50 mm women, downgraded but maintained; Rec 20 (III) no repair of asymptomatic AAA <55 mm in men; Rec 85 (IIb) symptomatic AAA: rapid assessment then urgent repair; EVAR preferred in most patients (LoE B), not outside IFU; ≥30 repairs per centre. Rupture risk by size: Lederle 2002. No survival benefit of early repair of 4.0–5.5 cm AAA: UK Small Aneurysm Trial.'),
        EVR, highlight=['evar-graft'], labels=['evar-graft', 'aaa-infra'], after=True,
        quiz=ask('A 58-year-old woman has a 5.2 cm infrarenal AAA that has grown 0.3 cm in a year. She is asymptomatic. What do you advise?', 'Plan elective repair now',
                 'ESVS 2024 keeps the threshold at 50 mm for women because they rupture at smaller diameters.', 'Scan every 6 months until it reaches 5.5 cm', 'Start doxycycline and rescan in a year', 'Repair only once it becomes painful'))
    aaa_rupt = step('aaam-rupture', 'Emergency', 'Ruptured AAA: the first hour',
        ul('<b>Permissive hypotension</b>: give just enough blood or fluid to keep the patient conscious; avoid large volumes of crystalloid',
           'Activate the massive transfusion protocol' + tag('prop', 'proposed'),
           'Stable enough? <b>CTA of the whole aorta and access vessels</b> at once',
           '<b>EVAR first</b> if the anatomy is suitable (under local anesthesia where possible); open repair otherwise',
           'Unstable with no time: straight to theatre; proximal control first (supracoeliac clamp or aortic balloon)',
           'After repair: watch for <b>abdominal compartment syndrome</b>, colonic ischemia and renal failure')
        + f'<p>Post-operative care: {link("approach=aaa-infra&amp;step=11", "ICU after open repair")}.</p>'
        + ev('ESVS 2024: permissive hypotension recommended (Rec 72, I-C); prompt CTA (Rec 70, I); EVAR first for suitable anatomy (Rec 80, I-A); aortic balloon occlusion downgraded. IMPROVE trial.'),
        EVR, highlight=['aaa-infra'], labels=['aaa-infra', 'evar-graft'], after=True,
        quiz=ask('A 74-year-old with a known 6 cm AAA collapses with back pain; BP 78/40, alert. What is the blood pressure strategy before control?', 'Permissive hypotension, keeping him conscious',
                 'Raising the pressure with large volumes worsens bleeding; ESVS 2024 recommends permissive hypotension (Rec 72).', 'Two litres of saline to reach a systolic of 120 mmHg', 'Noradrenaline infusion to a mean arterial pressure of 80', 'Nitrates to lower the pressure further before theatre'))
    aaa_counsel = step('aaam-counsel', 'Counselling', 'What to tell the patient',
        ul('Your aorta is enlarged; at this size the risk of it bursting is lower than the risk of an operation, so we watch it',
           'The scans are how we keep you safe: do not miss them',
           'Stop smoking; take the statin, aspirin and blood pressure tablets',
           'Sudden severe abdominal or back pain, or collapse: emergency, say you have an aneurysm',
           'Your brothers and sisters, especially men over 50, should have an ultrasound') + tag('prop', 'proposed'),
        AP, labels=['aaa-infra'])
    proc('aaa-mgmt', 'aaa', 'Abdominal aortic aneurysm', 'Management and surveillance (ESVS 2024)',
         'Risk factors, measuring, surveillance intervals by sex and size, the controls (medical therapy), indications for repair, EVAR or open, and rupture.',
         [('Patho', 'other', [aaa_patho]), ('Anatomy', 'other', [aaa_anat]), ('Case', 'other', [aaa_case]), ('Surveillance', 'other', [aaa_surv]), ('Controls', 'other', [aaa_bmt]),
          ('Repair', 'other', [aaa_repair]), ('Rupture', 'vein', [aaa_rupt]), ('Counsel', 'other', [aaa_counsel])],
         [ESVS_AAA, ESVS_AAA_NEW, ESVS_AAA_HL, LEDERLE, UKSAT, NTA3CT, IMPROVE, NAIR, BRESLER], before='aaa-infra')

    # ======================================================================================== 4. Common iliac artery aneurysm
    IL = {'view': view(V(LM['ciaa-r']) if 'ciaa-r' in LM else V(0, 38, -340), (0.3, 1.0, 0.35), 330),
          'show': [i for i in AO['show'] if i not in ('aorta',)] + ['aorta'], 'opacity': {**AO['opacity'], 'ivc-infra': 0.35, 'civ-r': 0.35, 'civ-l': 0.35}, 'ct': AO.get('ct')}
    ILo = {**IL, 'opacity': {**IL['opacity'], 'ciaa-r': 0.45}}
    ci_patho = step('ciaa-patho', 'Pathophysiology', 'Iliac artery aneurysm',
        '<p>Most iliac aneurysms are <b>common iliac</b>, usually <b>with an AAA</b> (up to 40% of AAA patients have one); isolated iliac aneurysms are uncommon. The internal iliac is the next commonest; the external iliac is rarely aneurysmal. '
        'Same disease as AAA: degeneration of the media, smoking, hypertension, age, male sex. Deep in the pelvis they are <b>hard to feel</b> and often present large, or ruptured.</p>'
        + tbl(['Diameter (common iliac)', 'Meaning (ESVS 2024)'], ['About 1 cm', 'Normal'], ['20–24 mm', 'Aneurysmal: ultrasound every 3 years'], ['25–29 mm', 'Every 2 years'], ['30 mm or more', 'Every year'],
              ['<b>40 mm</b>', '<b>Consider elective repair</b> (threshold raised from 35 mm)'])
        + ul('Pressure effects: ureteric obstruction (hydronephrosis), iliac vein compression (leg swelling, DVT), rarely rupture into the iliac vein (arteriovenous fistula)', 'Isolated common iliac aneurysms grow slowly (about 1 mm a year or less in most); a few grow fast')
        + ev('ESVS 2024 Rec 134 (IIa): surveillance every 3 years for 20–24 mm, 2 years for 25–29 mm, yearly from 30 mm; Rec 135 (IIa): elective repair at 40 mm (raised from 35 mm in 2019). Steenberge et al. (J Vasc Surg 2022): growth of isolated CIA aneurysms. Bresler et al. 2024: CIA aneurysm in up to 40% of AAA.'),
        ILo, show=['ciaa-r', 'ciaa-thrombus-r'], highlight=['ciaa-r'], labels=['ciaa-r', 'iia-r', 'eia-r', 'ureter-r', 'civ-r'], after=True,
        quiz=ask('At what diameter does ESVS 2024 suggest elective repair of a common iliac artery aneurysm?', '40 mm', 'The threshold was raised from 35 mm (2019) to 40 mm in 2024 (Rec 135, IIa).', '25 mm', '30 mm', '55 mm'))
    ci_anat = step('ciaa-anat', 'Anatomy', 'The iliac bifurcation and what lies on it',
        '<p>The common iliac runs from the aortic bifurcation (L4) to the pelvic brim in front of the sacroiliac joint, where it divides into the <b>external iliac</b> (to the groin) and the <b>internal iliac</b> (to the pelvis: buttock, bowel, bladder, erectile tissue).</p>'
        + ul('<b>The ureter crosses the iliac bifurcation</b>: it can be stuck to an aneurysm, and must be seen and protected', 'The <b>iliac veins</b> lie behind; the left common iliac vein crosses behind the right common iliac artery: a tear here bleeds torrentially',
             'The <b>internal iliac</b> supplies the gluteal muscles, the rectum and the pelvic organs: keep at least one patent', 'A <b>landing zone</b> for a stent graft needs a healthy segment: 10–15 mm of normal external (and internal) iliac')
        + ev('Preserving at least one internal iliac artery is recommended (ESVS, as cited by Bresler et al. 2024).'),
        IL, show=['ciaa-r', 'ureter-r'], highlight=['iia-r', 'eia-r'], danger=['ureter-r', 'civ-r', 'civ-l'], labels=['ciaa-r', 'iia-r', 'eia-r', 'ureter-r', 'civ-l'], opacity={'ciaa-r': 0.4})
    ci_case = step('ciaa-case', 'Case', 'A 4.2 cm right common iliac aneurysm',
        '<p>A <b>69-year-old man</b>, smoker, had a CT for haematuria: a <b>4.2 cm right common iliac aneurysm</b>, a normal-calibre aorta (2.2 cm), a normal external iliac and a healthy right internal iliac of 8 mm. '
        'Mild right hydronephrosis. He walks 2 km a day and is sexually active.</p>',
        ILo, show=['ciaa-r', 'ciaa-thrombus-r', 'ureter-r'], highlight=['ciaa-r'], labels=['ciaa-r', 'iia-r', 'ureter-r'],
        quiz=ask('Which repair best preserves his pelvic circulation?', 'EVAR with an iliac branch device into the right internal iliac',
                 'He is at the 40 mm threshold, active and sexually active: an iliac branch device keeps antegrade flow to the internal iliac and avoids buttock claudication and erectile dysfunction from coiling it.',
                 'Coil the internal iliac and extend the limb into the external iliac', 'Surveillance in 3 years', 'Ligate the common iliac'))
    ci_decide = step('ciaa-decide', 'Decision', 'Which repair?',
        tbl(['Option', 'Suits', 'Price'],
            ['<b>EVAR with an iliac branch device</b>', 'Suitable anatomy (landing zones in the external and internal iliac, not too tortuous); preferred to keep pelvic flow', 'Cost, device availability, branch occlusion'],
            ['EVAR with <b>internal iliac coil embolization</b> and an extension into the external iliac', 'Anatomy unsuitable for a branch; the other internal iliac is patent', '<b>Buttock claudication about 28–31%; new erectile dysfunction about 17%</b>; rarely colonic or spinal ischemia, especially if both sides are lost'],
            ['<b>Open repair</b> (interposition graft, internal iliac re-implanted)', 'Young and fit, unsuitable anatomy, infection, no access to devices', 'A big operation; ureter and iliac vein at risk'])
        + ul('Choose by patient and lesion (ESVS Rec 136)', '<b>Never sacrifice both internal iliacs</b> if it can be avoided', 'Isolated internal iliac aneurysm: coil the outflow branches and the aneurysm, or a branch device')
        + ev('ESVS 2024 Rec 136 (IIa): technique by patient and lesion characteristics; preserve at least one internal iliac (ESVS, cited by Bresler 2024). Rayt et al. 2008 (pooled 634 patients): buttock claudication 28% (unilateral 31%, bilateral 35%), new erectile dysfunction 17% after internal iliac embolization.'),
        IL, show=['ciaa-r', 'evar-graft'], highlight=['iia-r'], labels=['ciaa-r', 'iia-r', 'iia-l'], opacity={'ciaa-r': 0.35, 'evar-graft': 0.5}, after=True,
        quiz=ask('After coil embolization of one internal iliac artery before EVAR, roughly how many patients get buttock claudication?', 'About 3 in 10',
                 'A pooled analysis of 634 patients found buttock claudication in 28% (31% after unilateral embolization).', 'Fewer than 1 in 50', 'About 9 in 10', 'None, if the other side is open'))
    ci_cons = step('ciaa-consent', 'Consent', 'Consent: what to discuss with this patient',
        consent('Fitness for open repair, renal function (contrast), the anatomy on CTA.',
                ['Groin wound problems (EVAR)', 'Lifelong scans after a stent graft'],
                ['Endoleak and further procedures', 'Branch occlusion (internal iliac)', 'Buttock claudication, erectile dysfunction (if the internal iliac is blocked)', 'Bowel ischemia', 'Ureteric injury (open)', 'Death'],
                ['With a branch device: a second puncture (arm or the other groin)', 'With coiling: possible pain in the buttock on walking, which often improves'],
                ['Open repair', 'Surveillance below 40 mm', 'No operation, and what that means'],
                ['Home in 1–3 days after EVAR; 1–2 weeks after open repair', 'CTA at 30 days, then surveillance'],
                ['Branch devices may need importing: cost and waiting time', 'Who will do and pay for lifelong CT surveillance']),
        IL, show=['ciaa-r'], labels=[], opacity={'ciaa-r': 0.4}, after=True,
        quiz=ask('Why does a patient with an iliac branch device need lifelong imaging?', 'Endoleaks and branch occlusion can appear years later',
                 'Like any EVAR, the sac is excluded, not removed; late failures are silent and found only by imaging.', 'To check the groin scars for infection each year', 'Only for the first year, after which the device is stable', 'It is not needed after an iliac branch device'))
    ci_ibd = step('ciaa-ibd', 'Endovascular', 'EVAR with an iliac branch device',
        '<p>Bilateral femoral access (and often a brachial or contralateral up-and-over route). The <b>iliac branch device</b> is deployed first, landing in the <b>external iliac</b>; through its side gate the <b>internal iliac is cannulated and bridged with a covered stent</b>. '
        'Then the bifurcated aortic body, with its limb docking into the branch device; the contralateral limb as usual. Completion angiogram: no endoleak, both iliac outflows open.</p>',
        IL, show=['evar-graft', 'ciaa-r'], highlight=['ibd-r'], labels=['ibd-r', 'iia-r', 'evar-graft'], opacity={'evar-graft': 0.55, 'ciaa-r': 0.25},
        action={'kind': 'reveal', 'label': 'Deploy the iliac branch device', 'port': 'groin-r', 'ids': ['ibd-r']})
    ci_coil = step('ciaa-coil', 'Endovascular', 'Alternative: coil the internal iliac, extend into the external iliac',
        '<p>When a branch device is not possible: <b>coil the main internal iliac trunk</b> (proximal to its division, to keep the collaterals between the gluteal branches) a few days before or at the EVAR, '
        'then extend the iliac limb into the <b>external iliac</b>, sealing below the aneurysm. The coils stop back-bleeding into the sac (a type II endoleak).</p>'
        + ul('Coil proximally, not into the gluteal branches', 'Stage bilateral embolization if both sides must be lost, and preserve flow on one side wherever possible'),
        IL, show=['evar-graft', 'ciaa-r'], highlight=['iia-coils-r'], labels=['iia-coils-r', 'eia-ext-r', 'iia-r'], opacity={'evar-graft': 0.55, 'ciaa-r': 0.25},
        action={'kind': 'reveal', 'label': 'Coil, then extend the limb', 'port': 'groin-r', 'ids': ['iia-coils-r', 'eia-ext-r']})
    ci_open = step('ciaa-open', 'Open', 'Open repair: interposition graft',
        '<p>Retroperitoneal (flank) or midline transperitoneal approach. Identify and sling the <b>ureter</b> first. Control the distal aorta or contralateral common iliac, the external iliac and the internal iliac. '
        'Open the sac, oversew back-bleeding branches from inside, and sew in a 10–12 mm Dacron <b>interposition graft</b> from the common iliac origin to the iliac bifurcation or external iliac, '
        '<b>re-implanting the internal iliac</b> (or a side limb to it). Close the sac over the graft.</p>',
        IL, show=['ureter-r'], highlight=['graft-cia-r'], danger=['ureter-r', 'civ-r', 'civ-l'], labels=['graft-cia-r', 'ureter-r', 'civ-l'],
        action={'kind': 'reveal', 'label': 'Sew in the graft', 'port': 'groin-r', 'ids': ['graft-cia-r']})
    ci_icu = step('ciaa-icu', 'ICU', 'ICU and post-operative care',
        icu_head + h4('Specific to this operation')
        + ul('Leg pulses and groin wounds hourly for 6 hours (EVAR access)', 'Buttock pain on walking: internal iliac occlusion (expected after coiling; a new symptom after a branch device suggests branch occlusion: duplex or CT)',
             '<b>Bowel</b>: bloody diarrhea or rising lactate after internal iliac loss: sigmoidoscopy for colonic ischemia', 'Urine output and creatinine (contrast); the ureter after open repair (flank pain, rising creatinine, urine in the drain)',
             'CTA at 30 days, then surveillance as for EVAR; antiplatelet and statin lifelong')
        + h4('Labs') + ul('Day 1: FBC, UEC, lactate if bowel concern') + doses
        + ev('ESVS 2024: early (30-day) CTA after EVAR; iliac diameter over 20 mm is a high-risk feature for annual imaging (Endovascular Today 2024).'),
        IL, show=['evar-graft', 'ibd-r'], labels=['ibd-r'], opacity={'evar-graft': 0.55}, after=True,
        quiz=ask('Two days after EVAR with bilateral internal iliac coiling, a patient has bloody diarrhea and a lactate of 4. Most likely?', 'Ischemic colitis from loss of pelvic flow',
                 'Bilateral internal iliac loss removes the collateral supply to the sigmoid and rectum; ischemic colitis needs urgent endoscopy and may need a laparotomy.', 'Antibiotic-associated colitis from cefazolin', 'Acute diverticulitis of the sigmoid colon', 'A delayed reaction to the iodinated contrast'))
    proc('ciaa-repair', 'iliac', 'Iliac artery aneurysm', 'Common iliac aneurysm: iliac branch device, coils or open repair',
         'Common iliac artery aneurysm: sizes and surveillance, the 40 mm threshold, keeping the internal iliac, and the three repairs.',
         [('Patho', 'other', [ci_patho]), ('Anatomy', 'other', [ci_anat]), ('Case', 'other', [ci_case]), ('Decision', 'other', [ci_decide]), ('Consent', 'other', [ci_cons]),
          ('Branch device', 'vein', [ci_ibd]), ('Coils', 'other', [ci_coil]), ('Open', 'other', [ci_open]), ('ICU', 'other', [ci_icu])],
         [ESVS_AAA, ESVS_AAA_NEW, ESVS_AAA_HL, STEEN, BRESLER, RAYT])
    print('  vascular 2: pad-bmt, ali-iia, aaa-mgmt, ciaa-repair')
