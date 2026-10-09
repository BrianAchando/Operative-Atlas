"""Bullous lung disease: giant bulla (vanishing lung), how to tell it from a pneumothorax, who benefits from bullectomy,
the alternatives (LVRS, endobronchial valves, intracavitary drainage), VATS bullectomy with a buttressed staple line, and
air leak care. Added after postop.apply, so it carries its own consent and ICU steps.
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
SCHIPPER = {'title': 'Schipper PH, Meyers BF, Battafarano RJ, Guthrie TJ, Patterson GA, Cooper JD. Outcomes after resection of giant emphysematous bullae. Ann Thorac Surg 2004;78:976-82',
            'url': 'https://profiles.wustl.edu/en/publications/outcomes-after-resection-of-giant-emphysematous-bullae/'}
BENDITT = {'title': 'Benditt JO. Surgical options for patients with COPD: sorting out the choices. Respir Care 2006;51:173-82',
           'url': 'https://rc.rcjournal.com/content/51/2/173.full.pdf'}
VENN = {'title': 'Venn GE, Williams PR, Goldstraw P. Intracavity drainage for bullous, emphysematous lung disease: experience with the Brompton technique. Thorax 1988;43:998-1002',
        'url': 'https://thorax.bmj.com/content/43/12/998'}
BUERO = {'title': 'Buero A, et al. Entirely thoracoscopic resection of a giant emphysematous bulla. Pan Afr Med J 2018;30:247',
         'url': 'https://www.panafrican-med-journal.com/content/article/30/247/full/'}
NETT = {'title': 'Fishman A, et al. A randomized trial comparing lung-volume-reduction surgery with medical therapy for severe emphysema (NETT). N Engl J Med 2003;348:2059-73',
        'url': 'https://wikijournalclub.org/wiki/NETT'}
GOLD = {'title': 'Global Initiative for Chronic Obstructive Lung Disease. 2025 GOLD report: interventional therapy (summary)',
        'url': 'https://pulmonx.com/wp-content/uploads/2025/06/2025-GOLD-Report-Summary.pdf'}
BTS = {'title': 'Roberts ME, et al. British Thoracic Society guideline for pleural disease. Thorax 2023;78:1143-56',
       'url': 'https://thorax.bmj.com/content/78/11/1143.full.pdf'}
DEVALLA = {'title': 'Devalla L, et al. Apparent "double wall sign" in emphysematous bullae of the lung. Pan Afr Med J 2026;53:154',
           'url': 'https://www.panafrican-med-journal.com/content/article/53/154/full'}
EBV_CASE = {'title': 'Bronchoscopic lung volume reduction with an endobronchial valve for huge emphysematous bullae: a case report. BMC Pulm Med 2019',
            'url': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6518705/'}


def add(procs, ask, has, LM, S):
    need = ('bulla-giant', 'bulla-staple', 'bulla-compressed', 'rul', 'port-r-anterior-utility')
    if not all(has(i) for i in need) or 'rul-anterior' not in procs or 'bulla-staple-0' not in LM:
        print('  bullous disease: meshes or source operation missing, skipped'); return
    hv = lambda xs: [i for i in xs if has(i)]
    DEFAULT_ON = [i for i, s in S.items() if s.get('visible', True) is not False or s.get('sideVisible')]
    NEW = ['bulla-giant', 'bulla-septa', 'bulla-compressed', 'bulla-staple', 'bulla-drains']

    def borrow(key, sid):
        s = next(x for x in procs[key]['steps'] if x['id'] == sid)
        return {k: copy.deepcopy(s[k]) for k in ('view', 'show', 'opacity', 'ct', 'pose') if s.get(k) is not None}

    def step(id_, phase, title, body, base, show=(), drop=(), highlight=(), danger=(), labels=(), opacity=None, action=None, quiz=None, after=False, view=None, ct=None):
        b = copy.deepcopy(base)
        acts = set(action['ids']) if action and 'ids' in action else set()
        sh = [i for i in hv([*b.get('show', []), *show]) if i not in drop and i not in acts]
        named = set(sh) | set(highlight) | set(danger) | set(labels) | acts
        s = {'id': id_, 'phase': phase, 'title': title, 'body': body, 'view': view or b['view'], 'show': list(dict.fromkeys(sh)),
             'hide': [i for i in DEFAULT_ON if i not in named] + [i for i in NEW if i not in named and has(i)] + [i for i in drop if i not in named],
             'highlight': hv(highlight), 'danger': hv(danger), 'labels': hv(labels), 'opacity': {**b.get('opacity', {}), **(opacity or {})}}
        if ct or b.get('ct'): s['ct'] = ct or b['ct']
        if b.get('pose'): s['pose'] = b['pose']
        if action: s['action'] = action
        if quiz: s['ask'] = quiz
        if after: s['askAfter'] = True
        return s

    def view(t, d, dist):
        d = U(V(d)); t = V(t); return {'eye': R(t + d * dist), 'target': R(t)}

    consent = lambda risk, common, serious, specific, alts, recovery, kenya: (
        f'<p><b>1. The patient\'s own risk</b>: {risk} Quote the figure, not a textbook average.</p>'
        + h4('2. Common') + ul(*common) + h4('3. Serious') + ul(*serious) + h4('4. Specific to this operation') + ul(*specific)
        + h4('5. Alternatives') + ul(*alts) + h4('6. Recovery') + ul(*recovery) + h4('7. Kenya-specific') + ul(*kenya)
        + f'<p>Template and how to use it: {link("approach=cticu-consent&amp;step=0", "CTICU protocol, consent")}.</p>')

    B = V(LM['bulla']); BASE = V(LM.get('bulla-base', LM['bulla'])); NRM = U(V(LM.get('bulla-normal', (0, 0, 1))))
    path = [R(LM[f'bulla-staple-{i}']) for i in range(4) if f'bulla-staple-{i}' in LM]
    LUNGS = hv(['rul', 'rml', 'rll', 'lul', 'lll', 'trachea', 'br-right-main'])
    CHEST = {'view': view(B, (1.0, 0.35, 0.3), 330), 'show': LUNGS, 'opacity': {'rul': 0.25, 'rml': 0.35, 'rll': 0.3, 'lul': 0.25, 'lll': 0.25},
             'ct': {'focus': R(B), 'plane': 'coronal', 'window': 'lung'}}
    CLOSE = {'view': view(B, (1.0, 0.45, 0.45), 210), 'show': hv(['rul', 'rml', 'rll']), 'opacity': {'rul': 0.2, 'rml': 0.35, 'rll': 0.3},
             'ct': {'focus': R(B), 'plane': 'axial', 'window': 'lung'}}
    SETUP = borrow('rul-anterior', 'ra-setup')
    BUL = ['bulla-giant', 'bulla-septa', 'bulla-compressed']

    # ------------------------------------------------------------------------------------------------ teaching steps
    patho = step('bul-patho', 'Pathophysiology', 'Bullae, blebs and the vanishing lung',
        tbl(['Term', 'What it is'],
            ['<b>Bulla</b>', 'An air space over 1 cm across in the lung, from destroyed alveolar walls; its wall is thin (under 1 mm)'],
            ['<b>Bleb</b>', 'A small collection of air <b>within the visceral pleura</b>, usually at the apex; the source of primary spontaneous pneumothorax'],
            ['<b>Giant bulla</b>', 'A bulla occupying <b>at least a third of the hemithorax</b>; "vanishing lung syndrome" when giant bullae fill one or both upper lobes and compress the lung (Roberts criteria)'])
        + chain('Smoking (paraseptal emphysema), post-TB scarring, alpha-1 antitrypsin deficiency, HIV, connective tissue disease', 'Destroyed septa', 'Check-valve air trapping: the bulla grows',
                '!Compression of better lung', 'Dyspnea, then pneumothorax, infection, hemoptysis')
        + '<p>The bulla is <b>dead space</b>: it fills and empties poorly and takes no part in gas exchange, but it <b>squeezes the lung around it</b> and flattens the diaphragm. '
          'Removing it lets the compressed lung re-expand, which is why the benefit depends on <b>how good that compressed lung is</b>.</p>'
        + ul('<b>Complications</b>: secondary pneumothorax, infection of the bulla (an air-fluid level), hemoptysis, and an association with lung cancer in the bulla wall',
             '<b>In Kenya</b>: post-tuberculous bullae and cavities are common, often in smokers; think of TB first when a young adult has upper lobe bullae, and confirm the TB is treated')
        + ev('Definitions and the one-third rule: Buero et al. 2018 (Roberts criteria for vanishing lung); Benditt 2006. Complications (dyspnea, chest pain, infection, pneumothorax, rarely malignancy): Buero et al. 2018.'),
        CHEST, show=BUL, highlight=['bulla-giant'], labels=['bulla-giant', 'bulla-compressed', 'rml', 'rll'], opacity={'bulla-giant': 0.4}, after=True,
        quiz=ask('What makes a bulla "giant"?', 'It fills at least a third of the hemithorax',
                 'Giant bullae occupy at least one third of the hemithorax (Roberts criteria); most surgeons want a third, preferably a half, before offering bullectomy for dyspnea.',
                 'It is larger than 5 cm across on a plain radiograph of the chest', 'Its wall has thickened to more than 3 mm from chronic inflammation', 'It lies within the visceral pleura over the apex of the lung'))

    anat = step('bul-anat', 'Anatomy', 'Reading the CT: bulla or pneumothorax?',
        tbl(['Feature', 'Giant bulla', 'Pneumothorax'],
            ['Inner border', 'Curved, <b>concave toward the chest wall</b>', 'Visceral pleural line <b>parallel</b> to the chest wall'],
            ['Inside the space', 'Thin <b>strands</b> (septa, vessels) crossing it', 'No lung markings at all'],
            ['Lung', 'Compressed lung at the <b>base</b> of the bulla; the rest of the lung stays against the wall', 'The whole lung collapses toward the hilum'],
            ['Air on both sides of a thin wall', 'Not seen', 'The <b>double wall sign</b>: a pneumothorax next to a bulla'])
        + '<p><b>Never put a drain in on a plain film alone</b> when bullous disease is possible: a drain in a bulla makes a bronchopleural fistula. <b>CT first</b> if the patient is stable.</p>'
        + '<p>On the model: the bulla (pale), the strands crossing it, and the compressed lung at its base (dark). The rest of the lobe and the middle lobe sit below it.</p>'
        + ev('Double wall sign (air on both sides of the bulla wall, parallel to the chest wall) points to a pneumothorax beside a bulla; two adjacent bullae can mimic it; telling the two apart prevents an unnecessary drain (Devalla et al. 2026). CT is the best test to define bullous disease (Buero et al. 2018; Benditt 2006).'),
        CLOSE, show=BUL, highlight=['bulla-compressed'], labels=['bulla-giant', 'bulla-septa', 'bulla-compressed', 'rml'], opacity={'bulla-giant': 0.35},
        ct={'focus': R(B), 'plane': 'axial', 'window': 'lung'}, after=True,
        quiz=ask('A smoker with severe COPD is breathless; the radiograph shows a large lucent area at the right apex. He is stable. What next?', 'CT of the chest before any drain',
                 'A giant bulla can mimic a pneumothorax; a drain placed into a bulla creates a bronchopleural fistula. In a stable patient, CT decides.',
                 'An immediate large-bore chest drain in the 2nd space, mid-clavicular line', 'Needle aspiration through the 2nd space, then a repeat radiograph', 'High-flow oxygen and a repeat radiograph in 6 hours'))

    case = step('bul-case', 'Case', 'Breathless, with a "pneumothorax" that is not',
        '<p>A <b>52-year-old fisherman from Kisumu</b>, 35 pack-years, treated for pulmonary TB in 2016 (cured). Breathless walking 100 m on the flat (<b>mMRC 3</b>) despite inhalers. '
        'A district hospital radiograph was reported as a <b>right pneumothorax</b>; he was referred for a drain.</p>'
        + '<p><b>CT</b>: a <b>giant bulla</b> filling about <b>half of the right hemithorax</b>, with crowded vessels and dense lung at its base; the left lung shows only mild emphysema. '
        '<b>FEV1 1.1 L (35% predicted)</b>, DLCO 55%, PaCO2 5.6 kPa, echo: no pulmonary hypertension. GeneXpert on sputum negative. He stopped smoking 6 months ago.</p>',
        CHEST, show=BUL, highlight=['bulla-giant'], labels=['bulla-giant', 'bulla-compressed'], opacity={'bulla-giant': 0.4},
        quiz=ask('Which finding most favors a good result from bullectomy here?', 'Bulla filling half the hemithorax over compressed lung',
                 'The benefit is greatest when the bulla is large (a half of the hemithorax or more) and compresses relatively normal lung; diffuse emphysema elsewhere predicts a poor result.',
                 'The FEV1 of 35% predicted, which signals that he has the most to gain from surgery', 'His previous tuberculosis, since post-TB bullae are thick-walled and easy to staple', 'The DLCO of 55%, which shows that diffuse emphysema is severe throughout both lungs'))

    decide = step('bul-decide', 'Decision', 'Who benefits, and the alternatives',
        tbl(['Favors bullectomy', 'Against'],
            ['Bulla <b>at least a third, preferably half,</b> of the hemithorax', 'Many small bullae; diffuse (homogeneous) emphysema'],
            ['Well-defined <b>compressed lung</b> at its base on CT', 'No compressed lung visible'],
            ['Near-normal DLCO; normal PaCO2', 'Low DLCO; <b>hypercapnia</b>'],
            ['No pulmonary hypertension or cor pulmonale', 'Pulmonary hypertension, cor pulmonale, major comorbidity'],
            ['Stopped smoking; completed pulmonary rehabilitation', 'Still smoking; recurrent infection, chronic bronchitis'])
        + ul('<b>Other indications</b>: pneumothorax from the bulla (recurrent or with a persistent leak), infection, hemoptysis, or an enlarging bulla',
             '<b>Lung volume reduction surgery</b> is for diffuse upper-lobe emphysema, not a single giant bulla: NETT excluded bullae of a third of the lung or more',
             '<b>Endobronchial valves</b> (GOLD 2025: an option with FEV1 15–45% and hyperinflation) target a hyperinflated lobe with an intact fissure; for giant bullae only case reports',
             '<b>Too sick for resection</b>: intracavitary drainage (Monaldi, Brompton technique) through a small incision, under local or light general anesthesia')
        + ev('Selection: Benditt 2006 (bulla at least a third, preferably half, of the hemithorax; diffuse emphysema, low FEV1 and comorbidity weigh against). GOLD 2025: bullectomy in selected patients reduces dyspnea and improves lung function and exercise tolerance (Evidence C); LVRS improves survival in upper-lobe emphysema with low post-rehabilitation exercise capacity (Evidence A); EBV for FEV1 15–45% with hyperinflation (Evidence A). NETT 2003: high-risk group (FEV1 under 20% with homogeneous emphysema or DLCO under 20%) had higher mortality with surgery. Brompton technique: Venn et al. 1988.'),
        CHEST, show=BUL, highlight=['bulla-compressed'], labels=['bulla-giant', 'bulla-compressed', 'lul'], opacity={'bulla-giant': 0.4}, after=True,
        quiz=ask('A patient with a giant bulla has FEV1 0.35 L, PaCO2 7.8 kPa and is bed-bound. The bulla is infected and enlarging. Which option fits?', 'Intracavitary drainage (Brompton)',
                 'For patients too frail for resection, decompressing and draining the bulla through a small incision (Monaldi, Brompton technique) gives relief at lower risk.',
                 'VATS bullectomy with buttressed staplers and an apical pleurectomy', 'Bilateral lung volume reduction surgery through a median sternotomy', 'Endobronchial valves to the right upper lobe after a Chartis assessment'))

    cons = step('bul-consent', 'Consent', 'Consent: what to discuss with this patient',
        consent('in this case, a stable non-smoker with a large bulla over good compressed lung: low operative mortality in experienced centres (1 death in 43 in one series), but a <b>prolonged air leak is more likely than not</b>.',
                ['<b>Air leak lasting more than 7 days</b> (about half of patients in one series): the drain stays longer', 'Pain at the port sites and along the ribs; numbness under the breast',
                 'Chest drains for several days; physiotherapy every day'],
                ['Pneumonia, respiratory failure and a return to the ventilator', 'Bleeding needing return to theatre', 'Death: low in selected patients, higher with hypercapnia, very low FEV1 or pulmonary hypertension'],
                ['Conversion to a thoracotomy if adhesions are dense or the leak cannot be controlled', 'A persistent leak may need a blood patch, a one-way valve on the drain at home, valves by bronchoscopy, or a second operation',
                 'The benefit is greatest at about 6 months and declines slowly over the years', 'Smoking again undoes the operation'],
                ['Continue inhalers and pulmonary rehabilitation (no operation)', 'Intracavitary drainage (Brompton) if resection is too risky', 'Endobronchial valves in selected anatomy; lung transplantation is not available locally'],
                ['Ward bed after a short stay in high dependency; walking on day 1', 'Home when the lung is up and the drains are out, often 5–10 days', 'Pulmonary rehabilitation again at 4–6 weeks; lung function at 3–6 months'],
                ['Cost of staplers and buttress strips: discuss NHIF/SHA cover and the family contribution before the day', 'Ambulatory drain with a one-way valve if the leak persists, with a plan for review',
                 'Treated TB documented (GeneXpert negative) before an elective operation'])
        + tag('prop', 'proposed') + ev('Air leak over 7 days in 23 of 43 (53%), one operative death, FEV1 34% to 55% predicted at 6 months and 49% at 3 years: Schipper et al. 2004. Surgical mortality 0–22.5% across older series: Benditt 2006.'),
        CHEST, show=BUL, labels=['bulla-giant'], opacity={'bulla-giant': 0.4})

    # ------------------------------------------------------------------------------------------------ the operation
    setup = step('bul-setup', 'Setup', 'Anesthesia, position and ports',
        '<p><b>Anesthesia is the dangerous part.</b> Positive pressure can blow up the bulla or rupture it into a <b>tension pneumothorax</b>. '
        'Keep airway pressures low, allow a longer expiration, <b>no nitrous oxide</b>; isolate the lung early with a <b>double-lumen tube</b>; the surgeon is scrubbed at induction, ready to decompress.</p>'
        + ul('Lateral decubitus, the bulla side up', 'VATS: camera low, utility incision over the 4th space anteriorly, a posterior working port (the anterior approach ports)',
             'Paravertebral or erector spinae catheter for analgesia: good analgesia is what keeps the patient off the ventilator')
        + ev('Avoid nitrous oxide and high airway pressures in bullous disease; one-lung ventilation is a technical advantage (Benditt 2006).') + tag('prop', 'proposed'),
        SETUP, labels=['port-r-anterior-utility', 'port-r-anterior-camera', 'port-r-anterior-posterior'],
        quiz=ask('During induction the airway pressure climbs, the saturation falls and the blood pressure drops. What is the likely cause?', 'Tension pneumothorax from a ruptured bulla',
                 'Positive pressure can rupture a giant bulla; decompress the chest at once (needle or incision) and isolate the lung.',
                 'Bronchospasm from the double-lumen tube, which needs deeper anesthesia and a bronchodilator', 'Anaphylaxis to the muscle relaxant, which needs adrenaline and fluid boluses', 'Endobronchial intubation of the right main bronchus by the tube'))

    explore = step('bul-explore', 'Explore', 'Free the lung; find the base of the bulla',
        '<p>Divide any adhesions. The bulla collapses as soon as the lung is isolated, or it can be <b>opened</b> to find its base: look inside for the <b>strands and vessels</b> crossing it, '
        'and follow the wall down to where it meets <b>firm, pink, compressed lung</b>. That junction is where the staple line goes: in healthy-feeling lung, not across the thin bulla wall.</p>',
        CLOSE, show=BUL, highlight=['bulla-giant'], labels=['bulla-giant', 'bulla-septa', 'bulla-compressed'], opacity={'bulla-giant': 0.45},
        action={'kind': 'dissect', 'label': 'Open the bulla and find its base', 'port': 'port-r-anterior-utility',
                'path': [R(B + NRM * 20), R(B + NRM * 6), R(BASE + NRM * 8), R(BASE + NRM * 3)]})

    staple = step('bul-staple', 'Staple', 'Staple across the base, with buttress strips',
        '<p>Fire <b>linear staplers</b> (tissue-thickness reloads) across the base in sequence, each jaw loaded with a <b>buttress strip</b> (bovine pericardium or synthetic) '
        'so the staples hold in thin emphysematous lung. Preserve every bit of compressed lung: this is not a lobectomy.</p>'
        + ul('Do not staple through the bulla wall itself: it tears and leaks', 'Overlap the firings; check each end of the line', 'Remove the bulla in a bag')
        + ev('Buttressing, pericardial strips, fibrin glue and pleurectomy are used to reduce air leak, a common and difficult problem after bullectomy (Benditt 2006; Buero et al. 2018).'),
        CLOSE, show=['bulla-giant', 'bulla-compressed'], highlight=['bulla-compressed'], labels=['bulla-compressed', 'bulla-giant'], opacity={'bulla-giant': 0.35},
        action={'kind': 'staple-fissure', 'label': 'Staple across the base of the bulla', 'port': 'port-r-anterior-utility', 'reload': 'tissue', 'normal': R(NRM), 'path': path},
        after=True,
        quiz=ask('Where should the staple line lie?', 'Just into firm lung at the base of the bulla',
                 'Staples hold in lung tissue, not in the paper-thin bulla wall; staying close to the base preserves the compressed lung the operation is meant to release.',
                 'Across the middle of the bulla wall, where it is thinnest and easiest to compress', 'Across the lobar bronchus and vessels, so the whole upper lobe comes away', 'Along the fissure, leaving the bulla attached to the lobe'))

    close = step('bul-close', 'Close', 'Leak test, pleurodesis and drains',
        '<p><b>Leak test</b> under saline at modest pressure; reinforce any bubbling. Re-expand the lung and watch the compressed lung fill the space. '
        '<b>Pleurodesis</b>: mechanical abrasion or an apical pleurectomy, especially when the bulla caused a pneumothorax. <b>Two drains</b>: apical and basal.</p>'
        + ev('BTS 2023: surgical pleurodesis and/or bullectomy should be considered for spontaneous pneumothorax (conditional). Two drains and pleural abrasion after VATS bullectomy: Buero et al. 2018.') + tag('prop', 'proposed'),
        CHEST, show=['bulla-staple'], highlight=['bulla-staple'], labels=['bulla-staple', 'rml', 'rll'], opacity={'rul': 0.45},
        action={'kind': 'reveal', 'label': 'Place the apical and basal drains', 'port': 'port-r-anterior-camera', 'ids': ['bulla-drains']})

    icu = step('bul-icu', 'ICU', 'After bullectomy: the air leak, the lungs and the drains',
        f'<p>Start with the {link("approach=cticu-thoracic&amp;step=0", "thoracic core")} and the {link("approach=cticu-core&amp;step=2", "escalation table")}; then these points.</p>'
        + tbl(['Problem', 'What to do'],
              ['<b>Air leak</b> (the rule, not the exception)', 'Water seal if the lung stays up; low suction only if it does not. Record the leak daily (grade, or digital flow). Most settle in days'],
              ['<b>Persistent leak</b> (beyond 5–7 days)', 'One-way valve and home with the drain; <b>autologous blood patch</b>; endobronchial valve to the leaking segment; re-operation if the lung will not stay up'],
              ['<b>Subcutaneous emphysema</b>', 'Check the drain is patent and not kinked; more suction or a second drain if the lung is down'],
              ['<b>Ventilation</b>', 'Extubate in theatre if possible; if ventilated: low pressures and volumes, long expiratory time (the air leak and the remaining bullae)'],
              ['<b>Sputum and pain</b>', 'Regional analgesia, physiotherapy, early walking; bronchodilators; treat infection early'],
              ['<b>Hypercapnia</b>', 'Controlled oxygen (target 88–92%); NIV with low pressures if needed, watching the leak'])
        + ev('Autologous blood pleurodesis or endobronchial therapies should be considered for persistent air leak when surgery is not possible (BTS 2023, good practice point). Prolonged air leak in about half after giant bullectomy (Schipper et al. 2004). Other points: unit practice.') + tag('prop', 'proposed'),
        CHEST, show=['bulla-staple', 'bulla-drains'], labels=['bulla-drains', 'bulla-staple'], opacity={'rul': 0.45}, after=True,
        quiz=ask('Day 6: the lung is fully up on water seal, a small air leak persists, and he feels well. What is a reasonable plan?', 'One-way valve on the drain and home, with review',
                 'With the lung expanded and a small leak, an ambulatory one-way valve lets most leaks seal at home; blood patch, valves or surgery are for leaks that persist or a lung that will not stay up.',
                 'Return to theatre today for a muscle flap over the staple line', 'Clamp the drain for 24 hours and remove it if the radiograph is unchanged', 'High suction at −40 cmH2O until the leak stops completely'))

    steps_ = [(lab, kind, sts) for lab, kind, sts in [
        ('Patho', 'other', [patho]), ('Anatomy', 'other', [anat]), ('Case', 'other', [case]), ('Decision', 'other', [decide]), ('Consent', 'other', [cons]),
        ('Setup', 'other', [setup]), ('Explore', 'other', [explore]), ('Staple', 'fissure', [staple]), ('Close', 'other', [close]), ('ICU', 'other', [icu])]]
    steps, sq = [], []
    for i, (lab, kind, sts) in enumerate(steps_):
        sq.append({'label': lab, 'kind': kind})
        for s in sts: steps.append({**s, 'seq': i})
    p = {'id': 'bulla-vats', 'op': 'bullous', 'opName': 'Bullous lung disease', 'side': 'right', 'name': 'VATS bullectomy for a giant bulla', 'approach': 'VATS bullectomy (giant bulla)',
         'summary': 'Bulla, bleb or pneumothorax; who benefits from bullectomy; LVRS, valves and drainage; a buttressed staple line; the air leak.',
         'ports': [], 'steps': steps, 'sources': [SCHIPPER, BENDITT, BUERO, GOLD, NETT, VENN, BTS, DEVALLA, EBV_CASE], 'group': 'Emphysema and bullous disease', 'sequence': sq}
    procs['bulla-vats'] = p
    print('  bullous disease: bulla-vats')
