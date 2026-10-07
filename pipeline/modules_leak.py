"""Anastomotic leak after esophagectomy: recognition, classification and treatment, built on the International Society for
Diseases of the Esophagus (ISDE) consensus (Stuart et al., Dis Esophagus 2026) and the Esophagectomy Complications
Consensus Group (ECCG) definitions. Listed under Esophagectomy as its own approach, with its own consent and ICU steps
(added after postop.apply, which would otherwise give it the esophagectomy consent).
"""
from __future__ import annotations

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

ISDE = {'title': 'Stuart SK, Lemmens JMG, Nieuwenhuijzen GAP, et al. International Society for Diseases of the Esophagus consensus on the diagnosis and treatment of anastomotic leak after esophagectomy. Dis Esophagus 2026;39(1):doag006',
        'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC13078160/'}
ISDE_OUP = {'title': 'The same consensus at the publisher (Diseases of the Esophagus, Oxford Academic). doi:10.1093/dote/doag006', 'url': 'https://academic.oup.com/dote/article/39/1/doag006/8464945'}
ECCG = {'title': 'Low DE, Alderson D, Cecconello I, et al. International consensus on standardization of data collection for complications associated with esophagectomy: Esophagectomy Complications Consensus Group (ECCG). Ann Surg 2015;262:286-94',
        'url': 'https://repub.eur.nl/pub/92096'}
FABBI = {'title': 'Fabbi M, Hagens ERC, van Berge Henegouwen MI, Gisbertz SS. Anastomotic leakage after esophagectomy for esophageal cancer: definitions, diagnostics, and treatment. Dis Esophagus 2021;34(1):doaa039',
         'url': 'https://academic.oup.com/dote/article/34/1/doaa039/5849135'}
MUCHA = {'title': 'Mucha AW, Strandby RB, Nerup NA, Achiam MP. Treatment of intrathoracic anastomotic leakage following esophagectomy for gastroesophageal cancer: a systematic review. Dis Esophagus 2025;38(2):doaf016',
         'url': 'https://academic.oup.com/dote/article/38/2/doaf016/8058652?login=false'}
ATHAN = {'title': 'Athanasiou A, Hennessy M, Spartalis E, Tan BHL, Griffiths EA. Conduit necrosis following esophagectomy: an up-to-date literature review. World J Gastrointest Surg 2019;11:155-68 (ECCG conduit necrosis types, Table 1)',
         'url': 'https://www.wjgnet.com/1948-9366/tables/v11/i3/155.htm'}
SARAIVA = {'title': 'Saraiva S, Mão-de-Ferro S, Rosa I, Pereira ÁD. Endoscopic vacuum therapy for esophageal anastomotic leak: combining guidewire with overtube technique. Endoscopy 2020;52:E92-3',
           'url': 'https://www.thieme-connect.com/products/ejournals/abstract/10.1055/a-1011-3646'}
CHEN = {'title': 'Chen JC, Raj V, Husain S. Instruction video for endosponge construction (nasogastric tube and wound-vac foam, 125 mmHg). Videoscopy 2025',
        'url': 'https://journals.sagepub.com/doi/abs/10.1177/23733063251380425'}
SOURCES = [ISDE, ISDE_OUP, ECCG, FABBI, MUCHA, ATHAN, SARAIVA, CHEN]


def add(procs, ask, has, LM, S):
    need = ('leak-defect', 'leak-collection', 'leak-evt', 'leak-stent', 'conduit-chest', 'anast-chest')
    if not all(has(i) for i in need) or 'b4-eso-ivor' not in procs:
        print('  leak: meshes missing, skipped'); return
    P = lambda k: V(LM[k]) if k in LM else V(S[k]['centroid'])
    hv = lambda xs: [i for i in xs if has(i)]
    DEFAULT_ON = [i for i, s in S.items() if s.get('visible', True) is not False or s.get('sideVisible')]
    VERT = {f'vert-t{i}': 0.22 for i in range(2, 11)}
    LEAK = [i for i in S if i.startswith('leak-')]

    def view(t, d, dist):
        d = U(V(d)); t = V(t); return {'eye': R(t + d * dist), 'target': R(t)}

    def step(id_, phase, title, body, v, show=(), highlight=(), danger=(), labels=(), opacity=None, action=None, quiz=None, after=False, ct=None, spin=False):
        # tools revealed in earlier steps (EVT, stent, diversion are alternatives) do not carry over
        acts = set(action['ids']) if action else set()
        show = [i for i in hv(show) if i not in acts]          # what this step reveals starts hidden and is animated in
        named = set(show) | set(highlight) | set(danger) | set(labels) | acts
        s = {'id': id_, 'phase': phase, 'title': title, 'body': body, 'view': v, 'show': show,
             'hide': [i for i in DEFAULT_ON if i not in named] + [i for i in LEAK if i not in named],
             'highlight': hv(highlight), 'danger': hv(danger), 'labels': hv(labels), 'opacity': {**VERT, **(opacity or {})}}
        if ct is not None: s['ct'] = {'focus': R(ct), 'plane': 'axial', 'window': 'mediastinum'}
        if action: s['action'] = action
        if quiz: s['ask'] = quiz
        if after: s['askAfter'] = True
        if spin: s['spin'] = True
        return s

    A, D, C, E = P('anast-chest'), P('leak-defect'), P('leak-collection'), P('leak-effusion') if 'leak-effusion' in LM else P('leak-collection')
    BACK = (0.72, -0.62, 0.3)              # from the right and behind, as through a right thoracotomy
    CH = ['esophagus', 'aorta', 'azygos', 'trachea', 'br-left-main', 'br-right-main', 'heart', 'conduit-chest', 'anast-chest', 'vert-t4', 'vert-t5', 'vert-t6', 'vert-t7', 'vert-t8']
    OP = {'heart': 0.22, 'conduit-chest': 0.8, 'esophagus': 0.75, 'aorta': 0.7}
    v_close = view(D, BACK, 190); v_chest = view((A + E) / 2, BACK, 420)

    # ------------------------------------------------------------------------------------------------ teaching
    patho = step('leak-patho', 'Pathophysiology', 'Why anastomoses leak',
        '<p>The gastric conduit lives on <b>one artery, the right gastroepiploic</b>. Its tip, now at the anastomosis, is the point farthest from that artery and drains through the thinnest veins: '
        '<b>ischemia of the tip</b> is the common root of leak and of conduit necrosis. Tension, a poor staple line, and the patient (malnutrition, low albumin, neoadjuvant chemoradiation, diabetes, smoking, vascular disease) add to it.</p>'
        + chain('Tip ischemia, tension, technique', 'Full-thickness defect', '!Saliva and gastric juice into the mediastinum', 'Mediastinitis, empyema, sepsis', 'Fistula to the airway or aorta (late)')
        + '<p><b>Definition (ECCG):</b> a full-thickness gastrointestinal defect involving the esophagus, anastomosis, staple line or conduit, irrespective of presentation or method of identification. '
          'It complicates roughly <b>1 in 9 to 1 in 5</b> esophagectomies.</p>'
        + ev('ECCG definition (Low 2015). Incidence 11.4–21.2% and leak-related mortality 7.2–35% across series (Fabbi 2021).'),
        v_close, show=[*CH, 'leak-defect'], highlight=['leak-defect'], labels=['anast-chest', 'conduit-chest'], opacity=OP, ct=D, spin=True,
        quiz=ask('Why is the tip of the gastric conduit the part most at risk?', 'It is the point farthest from its only feeding artery, the right gastroepiploic',
                 'The conduit is perfused along the greater curvature from the right gastroepiploic artery; the tip, now at the anastomosis, is the most distal and least perfused.',
                 'It is crushed by the circular stapler', 'It is supplied by the left gastric artery, which is divided', 'It lies against the aorta'), after=True)

    anat = step('leak-anat', 'Anatomy', 'Where the leak goes',
        '<p>An <b>intrathoracic anastomosis</b> (Ivor Lewis) sits above the azygos arch, behind the trachea and right main bronchus, beside the descending aorta. A leak spills into the <b>posterior mediastinum</b> '
        'and, through the opened mediastinal pleura, into the <b>right pleural space</b>: mediastinitis and empyema, the reason intrathoracic leaks kill more often.</p>'
        '<p>A <b>cervical anastomosis</b> (McKeown, transhiatal) leaks under the neck wound and usually drains through it; it can still track down the conduit bed into the chest.</p>'
        + ul('<b>Behind the trachea:</b> a persistent leak can erode into the membranous airway (conduit–airway fistula: coughing on swallowing, air in the drain, recurrent pneumonia)',
             '<b>Beside the aorta:</b> rarely an aorto-conduit fistula, heralded by a small "sentinel" bleed before the catastrophic one',
             '<b>In the pleura:</b> a layering collection at the back and base of the right chest'),
        v_chest, show=[*CH, 'leak-defect', 'leak-collection', 'leak-effusion', 'rll', 'rul'], highlight=['leak-collection', 'leak-effusion'], danger=['trachea', 'aorta'],
        labels=['leak-defect', 'azygos'], opacity={**OP, 'rll': 0.15, 'rul': 0.15}, ct=C)

    case = step('leak-case', 'Case', 'Day 5 after an Ivor Lewis',
        '<p>A 62-year-old man had an Ivor Lewis esophagectomy for a lower-third squamous carcinoma after neoadjuvant chemoradiation. On day 5 he develops <b>new atrial fibrillation</b>, a temperature of 38.4 °C and '
        'a respiratory rate of 26. CRP has climbed from 150 to 240 mg/L between days 3 and 5. The right chest drain, clear yesterday, is <b>turbid</b>. He is on jejunostomy feeds.</p>',
        v_close, show=[*CH, 'leak-defect'], labels=['anast-chest'], opacity=OP, ct=D,
        quiz=ask('What is the first investigation?', 'CT of the chest with oral contrast',
                 'For suspected intrathoracic leak the ISDE consensus recommends CT with oral contrast first (84% agreement): it shows the leak and the collections that need draining.',
                 'Barium swallow', 'Repeat CRP in 24 hours', 'Bedside methylene blue by mouth'))

    diag = step('leak-diagnose', 'Decision', 'Making the diagnosis',
        tbl(['Clue', 'Weight (ISDE 2026)'],
            ['Change in the drain fluid (turbid, salivary, bilious, food)', '<b>Suggestive</b> (92%)'],
            ['Fever, respiratory distress, tachycardia (new AF)', 'Take into account (81–85%)'],
            ['Neck wound red and swollen (cervical anastomosis)', '<b>Suggestive</b> (87%)'],
            ['CRP: follow the <b>trend from day 3 to 7</b>; white count', 'Useful (86–90%)'],
            ['Drain amylase, if drains were left', 'Might help (74%)'])
        + h4('Which test, in which order')
        + ul('<b>Intrathoracic</b>: <b>CT with oral contrast first</b>',
             '<b>Cervical, no local infection</b>: CT with oral contrast; <b>with local infection</b>: CT or simply <b>open the incision at the bedside</b>. If septic, add a CT anyway (89%): a neck leak can track into the chest',
             'Leak found at endoscopy: CT with oral contrast to map the contamination (90%)',
             'Leak found on CT: <b>endoscopy</b> to size the defect and judge the conduit (98% agree it assesses the conduit)',
             '<b>Negative CT but still suspicious: endoscopy</b>',
             'No routine screening of well patients')
        + ev('ISDE 2026: statements 1–12, all evidence level C (no randomized trials). Fabbi 2021: a CRP around 17 mg/dL (170 mg/L) on day 3 marks a higher risk of leak.'),
        v_close, show=[*CH, 'leak-defect', 'leak-collection'], highlight=['leak-defect'], labels=['leak-collection'], opacity=OP, ct=D, after=True,
        quiz=ask('After a McKeown esophagectomy, a patient on day 6 has a red, swollen neck wound with pus, and is septic. Next?',
                 'Open the neck wound at the bedside and also get a CT with oral contrast',
                 'With local signs, opening the incision is acceptable first; in a septic patient the ISDE panel recommends a CT as well (89%), because a neck leak can extend into the mediastinum.',
                 'Contrast swallow only', 'Antibiotics and review in 48 hours', 'Return to theatre for a thoracotomy'))

    decide = step('leak-decide', 'Decision', 'Grade it, then match the treatment',
        tbl(['ECCG type', 'Leak', 'Conduit necrosis'],
            ['I', 'Local defect needing no change in therapy, or only medical treatment or diet change', 'Focal, seen at endoscopy: monitoring or non-surgical therapy'],
            ['II', 'Needs <b>intervention but not surgery</b> (radiological drain, stent, bedside opening and packing of the incision)', 'Focal, without free leak: surgery <b>not</b> involving diversion'],
            ['III', 'Needs <b>surgery</b>', '<b>Extensive</b>: conduit resection with diversion'])
        + h4('The ISDE algorithm')
        + tbl(['Situation', 'Treatment'],
              ['<b>Every leak</b>', 'Supportive care: broad-spectrum antibiotics, feeding support, nil by mouth, nasogastric decompression'],
              ['Confirmed leak, <b>no collection</b>, mild signs', 'Supportive care alone may be enough'],
              ['<b>Mediastinal or pleural collection</b>', '<b>Drain it</b> (plus supportive care); endoscopic closure (EVT, stent, or both) can be added'],
              ['<b>Uncontrolled sepsis</b> or failure of primary treatment', '<b>Surgical washout and drainage</b>; closure with sutures or tissue may be considered'],
              ['Limited ischemia of the conduit tip, not septic', 'Continuity-preserving treatment'],
              ['<b>Substantial conduit necrosis</b>', '<b>Primary diversion</b>'])
        + ev('ECCG types: Low 2015; necrosis types as tabulated by Athanasiou 2019. Algorithm: ISDE 2026 statements 13–23 (consensus ≥80% except where stated).'),
        v_chest, show=[*CH, 'leak-defect', 'leak-collection', 'leak-effusion'], highlight=['leak-defect'], labels=['leak-collection', 'leak-effusion'], opacity=OP, ct=C, after=True,
        quiz=ask('A leak is treated with a CT-guided drain and an endoscopic stent, without returning to theatre. Which ECCG type is it?', 'Type II',
                 'ECCG type II covers leaks needing interventional but not surgical therapy: radiological drains, stents, bedside opening of the neck wound.', 'Type I', 'Type III', 'It cannot be typed without the defect size'))

    consent = step('eso-leak-consent', 'Consent', 'Consent: what to discuss with this patient',
        '<p><b>1. The patient\'s own risk</b>: How septic, the size of the defect and the state of the conduit at endoscopy, nutrition. Quote the figure, not a textbook average.</p>'
        + h4('2. Common') + ul('Several procedures: endoscopy and sponge change every few days, or a stent and its removal', 'Weeks without eating by mouth; feeding through the jejunostomy', 'Drains for weeks')
        + h4('3. Serious') + ul('Spreading sepsis needing a return to theatre', 'Bleeding (sponge, stent erosion)', 'Fistula to the airway', 'Loss of the conduit and a neck stoma (diversion)', 'Death')
        + h4('4. Specific to this treatment') + ul('Stent: migration, chest pain, erosion; it must come out', 'Narrowing (stricture) of the anastomosis after healing, needing dilatation', 'After a diversion, reconstruction is a further major operation months later')
        + h4('5. Alternatives') + ul('Drainage and supportive care alone (small, contained leaks)', 'Surgery first, if sepsis is not controlled', 'Comfort care, and what that means')
        + h4('6. Recovery') + ul('Long stay: weeks, sometimes months', 'Oral intake only after the leak has healed (endoscopy or CT)', 'Follow-up for stricture')
        + h4('7. Kenya-specific') + ul('Cost of repeated endoscopy, sponges and stents; what is available here, and when transfer is better', 'Nutrition: the jejunostomy feed the family will need to buy and give')
        + f'<p>Template and how to use it: {link("approach=cticu-consent&amp;step=0", "CTICU protocol, consent")}.</p>',
        v_close, show=[*CH, 'leak-defect'], labels=[], opacity=OP, ct=D, after=True,
        quiz=ask('Why must a stent placed for a leak be discussed as temporary?', 'Covered stents migrate and can erode; they are removed or exchanged, usually by 4–8 weeks',
                 'Fully covered stents are left in for a median of 4–8 weeks and then removed; leaving them risks erosion, fistula and bleeding.', 'They dissolve', 'Because they block the airway', 'They are not: stents are permanent'))

    # ------------------------------------------------------------------------------------------------ treatment
    support = step('leak-support', 'Supportive care', 'Supportive care for every leak',
        ul('<b>Nil by mouth</b> (84%)', '<b>Broad-spectrum antibiotics</b> (99%), chosen by your unit\'s antibiogram; consider <b>antifungal</b> cover (74%, near consensus)',
           '<b>Feeding support</b> (98%): the jejunostomy, or a nasojejunal tube placed at endoscopy; parenteral only if the gut cannot be used',
           '<b>Nasogastric decompression</b> of the conduit (76%): placed or repositioned <b>under endoscopic vision</b>, never blind', 'Continue a <b>proton pump inhibitor</b> (79%)')
        + '<p>A confirmed leak with <b>no collection and only mild signs</b> can be treated this way alone. Anything more needs source control.</p>'
        + ev('ISDE 2026 statements 13–15. Conservative treatment of intrathoracic leaks: pooled success 82%, mortality 9%, but a mean of 70 days to heal (Mucha 2025, 133 patients).'),
        v_close, show=[*CH, 'leak-defect', 'leak-ngt'], highlight=['leak-ngt'], labels=['leak-defect'], opacity={**OP, 'conduit-chest': 0.45, 'esophagus': 0.45}, ct=A,
        action={'kind': 'reveal', 'label': 'Pass the NG tube under vision', 'port': 'thor-r', 'ids': ['leak-ngt']})

    drain = step('leak-drainage', 'Drain', 'Drain the collections',
        '<p><b>Any mediastinal or pleural collection is drained</b> (91–96%) as well as the supportive care.</p>'
        + tbl(['Where', 'How (ISDE 2026)'],
              ['Mediastinal collection', '<b>EVT</b> through the defect (82%), a <b>naso-mediastinal tube</b>, or <b>radiological</b> drainage (81%)'],
              ['Pleural collection', '<b>Radiological</b> tube drainage (81%) or a surgical chest drain (79%)'],
              ['Neck', 'Open the cervical incision at the bedside (81%)'])
        + '<p><b>Naso-mediastinal (naso-fistula) tube drainage</b> uses only a tube placed endoscopically through the defect into the cavity, on suction or with irrigation: cheap and widely available.</p>'
        + ev('Mucha 2025 (38 studies, 899 patients, all observational): naso-fistula tube drainage success 94% and mortality 5% (201 patients), versus EVT 82% and 11%, stents 75% and 14%; the authors could not recommend one treatment over another.'),
        v_chest, show=[*CH, 'leak-defect', 'leak-collection', 'leak-effusion', 'leak-drain'], highlight=['leak-drain'], labels=['leak-collection', 'leak-effusion'], opacity=OP, ct=E,
        action={'kind': 'reveal', 'label': 'Insert the chest drain', 'port': 'thor-r', 'ids': ['leak-drain']},
        quiz=ask('CT shows a 5 cm mediastinal collection and a right pleural collection after an Ivor Lewis. The patient is stable. Best plan?',
                 'Supportive care plus drainage of both collections, with endoscopic drainage or closure of the defect',
                 'Collections must be drained (ISDE 91–96%); endoscopic closure (EVT, stent) can be added. Surgery is for uncontrolled sepsis or failure.',
                 'Supportive care alone', 'Immediate rethoracotomy and redo anastomosis', 'Stent alone, without drains'))

    neck = step('leak-neck', 'Drain', 'The cervical leak: open the wound',
        '<p>Remove the skin sutures over the leak, open the wound down to the collection with a finger, wash it out and <b>pack it</b>; repack daily. Most cervical leaks then heal without further intervention, '
        'and the patient can often swallow saliva into the dressing.</p>'
        + ul('If septic, or not settling: <b>CT</b>, because the leak may extend into the mediastinum', 'Endoscopy to check the conduit if there is any doubt about its viability', 'Watch for a stricture later: cervical leaks often heal with one')
        + ev('ISDE 2026 statements 7, 8 and 17 (bedside opening 81%).'),
        view(P('leak-neck'), (-0.5, 1.0, 0.25), 210), show=['esophagus', 'trachea', 'thyroid', 'lcca', 'n-rln', 'conduit-neck', 'anast-neck', 'incision-neck', 'leak-neck-collection', 'leak-neck-pack'],
        highlight=['leak-neck-collection'], danger=['n-rln', 'lcca'], labels=['anast-neck', 'incision-neck'], opacity={'conduit-neck': 0.8, 'esophagus': 0.7}, ct=P('leak-neck'),
        action={'kind': 'reveal', 'label': 'Open and pack the wound', 'port': 'neck', 'ids': ['leak-neck-pack']})

    evt = step('leak-evt', 'Close', 'Endoscopic vacuum therapy (EVT)',
        '<p>An open-pore polyurethane sponge on a tube is placed endoscopically <b>through the defect into the cavity</b> (intracavitary) or <b>across the defect in the lumen</b> (intraluminal), and the tube brought out through the nose to '
        '<b>continuous suction of −100 to −125 mmHg</b>. It drains the cavity, removes the infected fluid and draws the walls together. The sponge is changed <b>every 3–4 days</b> (up to a week if intraluminal); the cavity shrinks with each change.</p>'
        + ul('Typical course: about 4–5 sponges over 2–3 weeks', 'Complications: bleeding, sponge displacement, fistula',
             '<b>Without a commercial kit:</b> a sponge cut from wound-vacuum foam, tied to a nasogastric tube and pushed into place over a guidewire, on a standard wound-vacuum pump at 125 mmHg' + tag('prop', 'proposed'))
        + ev('ISDE 2026 statement 20: EVT 82%, stent 71%, EVT with stent 77%. Fabbi 2021: −100 to −125 mmHg, change every 3–4 days, success 86–100% for intrathoracic leaks. Mucha 2025: mean 4.5 sponges, 19 days, success 82%, reintervention 26%. Home-made sponges: Saraiva 2020 (foam on a nasogastric tube, changed every 3–4 days), Chen 2025 (wound-vac foam, 125 mmHg).'),
        v_close, show=[*CH, 'leak-defect', 'leak-collection', 'leak-evt', 'leak-evt-tube'], highlight=['leak-evt'], labels=['leak-collection'],
        opacity={**OP, 'conduit-chest': 0.4, 'esophagus': 0.4, 'leak-collection': 0.45}, ct=C,
        action={'kind': 'reveal', 'label': 'Place the sponge in the cavity', 'port': 'thor-r', 'ids': ['leak-evt', 'leak-evt-tube']},
        quiz=ask('What suction is applied to an esophageal EVT sponge?', 'Continuous −100 to −125 mmHg', 'The usual EVT setting is continuous negative pressure of about −100 to −125 mmHg, with sponge changes every 3–4 days.',
                 'Intermittent −20 mmHg', 'Free drainage only', '−400 mmHg wall suction'))

    stent = step('leak-stent', 'Close', 'Covered stent',
        '<p>A <b>fully covered self-expanding metal stent</b> is placed across the anastomosis, landing about 4 cm above and below the defect, to seal it from the lumen. It lets the patient swallow early, '
        'but it does <b>not drain</b> a collection: drain any collection as well.</p>'
        + ul('Works best for a <b>small defect</b> found <b>early</b>', 'Migration: higher with plastic stents, lower with fully covered metal ones; check its position on X-ray',
             'Erosion into the airway or aorta, bleeding', '<b>Remove or exchange by 4–8 weeks</b>', 'EVT and a stent can be combined (a stent with a sponge built in, or sequentially)')
        + ev('ISDE 2026 statement 20 (stent 71%, combined 77%). Fabbi 2021: median stenting 4–8 weeks; smaller defects and earlier diagnosis predict success. Mucha 2025: stents 75% success, 14% mortality, 21% reintervention.'),
        v_close, show=[*CH, 'leak-defect', 'leak-stent'], highlight=['leak-stent'], labels=['leak-defect'], opacity={**OP, 'conduit-chest': 0.35, 'esophagus': 0.35}, ct=A,
        action={'kind': 'reveal', 'label': 'Deploy the stent across the leak', 'port': 'thor-r', 'ids': ['leak-stent']})

    reop = step('leak-reop', 'Close', 'Surgery for uncontrolled sepsis',
        '<p>If sepsis is <b>not controlled</b>, or drainage and endoscopic treatment have failed: <b>reopen the right chest, wash out and drain</b> (84–85%). Decorticate the lung if the empyema has organised. '
        'If the defect is small and the tissues healthy, it can be <b>closed with sutures</b> and buttressed with tissue (an intercostal muscle or omental flap): a weak recommendation (IIb).</p>'
        + ul('Re-look at the conduit: dusky or black means necrosis, which changes the operation to a diversion', 'Leave wide drains near the defect; place a feeding jejunostomy if there is none')
        + ev('ISDE 2026 statements 18 and 21.'),
        v_chest, show=[*CH, 'leak-defect', 'leak-collection', 'leak-effusion', 'leak-drain', 'rll', 'rul'], highlight=['leak-collection', 'leak-effusion'], danger=['aorta', 'trachea'], labels=['leak-defect'],
        opacity={**OP, 'rll': 0.15, 'rul': 0.15}, ct=C)

    nec = step('leak-necrosis', 'Necrosis', 'Conduit necrosis and diversion',
        '<p>At endoscopy the mucosa of the conduit tip is <b>dusky, black or sloughed</b>. Limited ischemia in a patient who is not septic: <b>preserve continuity</b> (drain, EVT, close watch). '
        '<b>Substantial necrosis: primary diversion</b>, whether the patient is septic or not.</p>'
        + h4('Diversion')
        + ul('Resect the necrotic conduit; return the viable stomach to the abdomen or resect it', 'Bring the esophageal remnant out as a <b>cervical esophagostomy</b> (spit fistula) on the left neck',
             'Drain the chest widely; feeding jejunostomy', 'Reconstruction later, when the patient has recovered, usually with colon')
        + ev('ISDE 2026 statements 22 (84%) and 23 (89%). ECCG conduit necrosis type III: extensive, treated with conduit resection and diversion.'),
        view(A + V(0, 0, -10), BACK, 260), show=[*CH, 'leak-necrosis', 'leak-esophagostomy'], highlight=['leak-necrosis'], danger=['leak-necrosis'], labels=['leak-esophagostomy'],
        opacity={**OP, 'conduit-chest': 0.6}, ct=A,
        action={'kind': 'reveal', 'label': 'Bring out the esophagostomy', 'port': 'neck', 'ids': ['leak-esophagostomy']},
        quiz=ask('Endoscopy on day 4 shows black mucosa over the top 6 cm of the conduit; the patient is on noradrenaline. Plan?', 'Return to theatre: resect the conduit and divert (cervical esophagostomy, feeding jejunostomy)',
                 'Substantial conduit necrosis needs primary diversion (ISDE 89%); a stent or sponge cannot treat dead tissue.', 'Covered stent across the anastomosis', 'EVT and review in 72 hours', 'Antibiotics alone'))

    icu = step('eso-leak-icu', 'ICU', 'ICU and post-operative care',
        f'<p>Start with the {link("approach=cticu-thoracic&amp;step=0", "thoracic core")}, the {link("approach=cticu-core&amp;step=0", "lab schedule")} and the {link("approach=cticu-core&amp;step=2", "escalation table")}; then this treatment\'s own points.</p>'
        + h4('Specific to the leak')
        + ul('Sepsis: cultures (blood, drain, pleural fluid) before antibiotics; narrow them on the results',
             'Drains: volume and character daily. Saliva, food or bile in the drain means the leak is not controlled',
             'CRP every 1–2 days: a falling trend is the best sign of control; if not falling, <b>repeat CT</b> for an undrained collection',
             'EVT: keep the suction continuous, check the seal and the canister; book the sponge change every 3–4 days',
             'Stent: chest X-ray for position; new chest pain or a fall in hemoglobin means migration or erosion',
             '<b>Feeding</b>: full enteral feeding through the jejunostomy; refeeding precautions in the malnourished',
             '<b>Red flags</b>: coughing on swallowing saliva or air in the drain (airway fistula); <b>any fresh blood from the NG or drain</b> (sentinel bleed of an aorto-conduit fistula: CT angiography now)',
             'Oral intake only after healing is shown (endoscopy or contrast study)')
        + h4('Labs') + ul('FBC, UEC, CRP, albumin, Mg, PO₄ at least every 48 hours while the leak is active')
        + f'<p>Doses: {link("approach=cticu-doses&amp;step=0", "electrolytes")}, {link("approach=cticu-doses&amp;step=1", "vasoactive drugs")}.</p>',
        v_chest, show=[*CH, 'leak-defect', 'leak-collection', 'leak-effusion', 'leak-drain', 'leak-ngt'], labels=['leak-drain'], opacity=OP, ct=C, after=True,
        quiz=ask('Ten days into EVT, the patient\'s CRP has risen again and the drain shows food particles. Next?', 'Repeat CT with oral contrast to look for an undrained collection, and re-endoscope',
                 'A rising CRP with gastrointestinal content in the drain means the leak is not controlled; image for collections and reassess the defect and conduit.', 'Stop antibiotics', 'Start oral feeding', 'Discharge to the ward'))

    groups = [('Patho', 'other', [patho]), ('Anatomy', 'other', [anat]), ('Case', 'other', [case]), ('Decision', 'other', [diag, decide]), ('Consent', 'other', [consent]),
              ('Support', 'other', [support]), ('Drain', 'other', [drain, neck]), ('Close', 'bronchus', [evt, stent, reop]), ('Necrosis', 'vein', [nec]), ('ICU', 'other', [icu])]
    steps, sq = [], []
    for i, (lab, kind, sts) in enumerate(groups):
        sq.append({'label': lab, 'kind': kind})
        for s in sts: steps.append({**s, 'seq': i})
    procs['eso-leak'] = {'id': 'eso-leak', 'op': 'oesophagectomy', 'opName': 'Esophagectomy', 'side': 'right', 'name': 'Esophagectomy', 'approach': 'Anastomotic leak (ISDE 2026)',
                         'summary': 'Recognizing, grading and treating a leak after esophagectomy: supportive care, drainage, EVT, stent, surgery and diversion (ISDE consensus 2026).',
                         'ports': [], 'steps': steps, 'sources': SOURCES, 'group': 'Esophagus', 'sequence': sq}
