"""Congenital cardiac series: secundum ASD closure, perimembranous VSD closure, PDA ligation (through the 3rd or the 4th
intercostal space) and coarctation repair (resection and extended end-to-end anastomosis).

The bypass steps (sternotomy, cannulation, cross-clamp, right atriotomy, closure) reuse the views and actions of the
transseptal mitral operation, with their text rewritten for these defects. Consent and ICU steps are added by postop.py.
The reference anatomy is an adult CT: the relations are those of a child's heart, the sizes are not.
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
ADULT = '<p class="note">The model is an adult heart: in a child the relations are the same and everything is smaller.</p>'

ESC = {'title': 'Baumgartner H, et al. 2020 ESC Guidelines for the management of adult congenital heart disease. Eur Heart J 2021;42:563-645', 'url': 'https://academic.oup.com/eurheartj/article/42/6/563/5898606'}
ESC_ACC = {'title': 'American College of Cardiology. 2020 ESC Guidelines for adult congenital heart disease: key points', 'url': 'https://www.acc.org/latest-in-cardiology/ten-points-to-remember/2020/08/29/13/17/2020-esc-guidelines-for-adult-chd-esc-2020'}
ESC_REV = {'title': 'Comments on the 2020 ESC guidelines for the management of adult congenital heart disease. Rev Esp Cardiol 2021', 'url': 'https://www.revespcardiol.org/en-comments-on-2020-esc-guidelines-articulo-S1885585721000773'}
TSRA = {'title': 'AATS / TSRA primer: surgical techniques 1. ASD, VSD, PDA, coarctation', 'url': 'https://www.aats.org/tsra-primer-surgical-techniques-1-asd-vsd-pda-coarctation'}
KNH = {'title': 'Osano et al. One-year outcomes and intervention waiting time of patients admitted with congenital heart disease at Kenyatta National Hospital, Kenya. Preprint (Research Square) 2025', 'url': 'https://www.researchsquare.com/article/rs-7386594/v1'}
ASD_SRC = [ESC, ESC_ACC, TSRA, KNH,
           {'title': 'Haleem SM, Kanmanthareddy A. Catheter management of atrial septal defect. StatPearls, updated 2025', 'url': 'https://www.ncbi.nlm.nih.gov/books/NBK536908/'}]
VSD_SRC = [ESC, ESC_ACC, TSRA, KNH,
           {'title': 'Society of Thoracic Surgeons. Ventricular septal defects (VSD): STS Cardiothoracic Surgery Consult (types 1-4, size by aortic annulus, indications)', 'url': 'https://consult.sts.org/sts/view/Cardiac-and-Congenital/1864080/all/Ventricular_Septal_Defects__VSD_'},
           {'title': 'Jacobs JP, et al. Congenital Heart Surgery Nomenclature and Database Project: ventricular septal defect. Ann Thorac Surg 2000;69:S25-35', 'url': 'https://www.sciencedirect.com/science/article/abs/pii/S0003497599012709'},
           {'title': 'Lopez L, et al. Classification of ventricular septal defects for ICD-11. Ann Thorac Surg 2018', 'url': 'https://ipccc.net/wp-content/uploads/2024/01/2018-11-ANNALS-Lopez-2018-VSD-Classification.pdf'},
           {'title': 'WFSA Anaesthesia Tutorial of the Week 316: ventricular septal defects (size relative to the aortic annulus)', 'url': 'https://resources.wfsahq.org/wp-content/uploads/316_english.pdf'},
           {'title': 'Azab S, et al. Permanent complete heart block following surgical closure of isolated ventricular septal defect. Egypt J Chest Dis Tuberc 2013;62:529-33', 'url': 'https://www.sciencedirect.com/science/article/pii/S0422763813000332'},
           {'title': 'Yoneyama F, et al. Conduction disorders after perimembranous ventricular septal defect closure: continuous versus interrupted suturing. Eur J Cardiothorac Surg 2022;62:ezab407', 'url': 'https://academic.oup.com/ejcts/article/62/1/ezab407/6373863'}]
PDA_SRC = [ESC, ESC_ACC, TSRA, KNH,
           {'title': 'Krichenko A, et al. Angiographic classification of the isolated, persistently patent ductus arteriosus. Am J Cardiol 1989 (summary: Pediatric Echocardiography library)', 'url': 'https://pedecho.org/library/chd/pda'},
           {'title': 'Fernando R, et al. PDA classification based on size and haemodynamic significance (table), 2013', 'url': 'https://www.researchgate.net/figure/Patent-ductus-arteriosus-PDA-classification-based-on-size-and-hemodynamic-significance_tbl2_259111641'},
           {'title': 'Sathanandam S, et al. Scoring system for post-ligation cardiac syndrome after transcatheter and surgical PDA closure in extremely low birthweight infants. Circulation 2019', 'url': 'https://www.researchgate.net/publication/342702564_Scoring_System_for_Post_Ligation_Cardiac_Syndrome_and_Its_Utility_After_Transcatheter_and_Surgical_Patent_Ductus_Arteriosus_Ligation_in_Extremely_Low_Birthweight_Infants'},
           {'title': 'Surgical management of PDA in the very preterm infant and postligation cardiac compromise. Thoracic Key', 'url': 'https://thoracickey.com/surgical-management-of-patent-ductus-arteriosus-in-the-very-preterm-infant-and-postligation-cardiac-compromise/'},
           {'title': 'Patent ductus arteriosus: surgical technique. Thoracic Key (from a cardiac surgery textbook)', 'url': 'https://thoracickey.com/patent-ductus-arteriosus/'},
           {'title': 'Subbian S, Winn MMA, Kiraly L. Surgical ligation of patent ductus arteriosus in pre-term infants: a narrative review. Pediatr Med 2024', 'url': 'https://pm.amegroups.org/article/view/7773/html'}]
COA_SRC = [ESC, ESC_REV, TSRA, KNH,
           {'title': 'Farag ES, et al. Aortic coarctation repair through left thoracotomy: results in the modern era. Eur J Cardiothorac Surg 2019;55:331-7', 'url': 'https://academic.oup.com/ejcts/article/55/2/331/5079307'},
           {'title': 'Brewer LA, et al. Spinal cord complications following surgery for coarctation of the aorta. J Thorac Cardiovasc Surg 1972', 'url': 'https://pubmed.ncbi.nlm.nih.gov/5054875/'}]


def add(procs, ask, has, LM, S):
    P = lambda k: V(LM[k]) if k in LM else V(S[k]['centroid'])
    hv = lambda xs: [i for i in xs if has(i)]
    DEFAULT_ON = [i for i, s in S.items() if s.get('visible', True) is not False or s.get('sideVisible')]
    if not all(has(i) for i in ('asd-defect', 'vsd-defect', 'pda', 'coa-segment')) or 'mvr-septal' not in procs:
        print('  congenital: meshes missing, skipped'); return
    VERT = {f'vert-t{i}': 0.25 for i in range(2, 11)}

    def view(t, d, dist):
        d = U(V(d)); t = V(t); return {'eye': R(t + d * dist), 'target': R(t)}

    def step(id_, phase, title, body, v, show=(), highlight=(), danger=(), labels=(), opacity=None, action=None, quiz=None, lead=None, after=False, spin=False, ct=None):
        show = hv(show); named = set(show) | set(highlight) | set(danger) | set(labels)
        s = {'id': id_, 'phase': phase, 'title': title, 'body': body, 'view': v, 'show': show,
             'hide': [i for i in DEFAULT_ON if i not in named], 'highlight': hv(highlight), 'danger': hv(danger), 'labels': hv(labels), 'opacity': {**VERT, **(opacity or {})}}
        if ct is not None: s['ct'] = {'focus': R(ct), 'plane': 'axial', 'window': 'mediastinum'}
        if action: s['action'] = action
        if quiz: s['ask'] = quiz
        if lead: s['lead'] = lead
        if after: s['askAfter'] = True
        if spin: s['spin'] = True
        return s

    def proc(key, op, opName, approach, summary, groups, sources):
        steps, sq = [], []
        for i, (lab, kind, sts) in enumerate(groups):
            sq.append({'label': lab, 'kind': kind})
            for s in sts: steps.append({**copy.deepcopy(s), 'seq': i})
        procs[key] = {'id': key, 'op': op, 'opName': opName, 'side': 'both', 'name': opName, 'approach': approach, 'summary': summary,
                      'ports': [], 'steps': steps, 'sources': sources, 'group': 'Congenital cardiac', 'sequence': sq}

    # ------------------------------------------------------------ bypass steps borrowed from the transseptal mitral operation
    MS = {s['id']: s for s in procs['mvr-septal']['steps']}
    DROP = {'mv-prosthesis', 'rh-mv', 'rh-mv-edge', 'rh-mv-calcium', 'rh-chordae'}

    def borrow(src, id_, title, body, extra_show=(), quiz=None):
        s = copy.deepcopy(MS[src]); s.pop('seq', None)
        s.update({'id': id_, 'title': title, 'body': body})
        s['show'] = [i for i in s['show'] if i not in DROP] + [i for i in hv(extra_show) if i not in s['show']]
        s['hide'] = [i for i in s.get('hide', []) if i not in s['show']]
        s['labels'] = [i for i in s.get('labels', []) if i not in DROP]
        for k in ('highlight', 'danger'): s[k] = [i for i in s.get(k, []) if i not in DROP]
        if quiz: s['ask'] = quiz; s['askAfter'] = True
        else: s.pop('ask', None); s.pop('askAfter', None)
        return s

    def bypass(pre, what):
        return [
            borrow('mt-sternotomy', f'{pre}-sternotomy', 'Median sternotomy',
                   '<p>Median sternotomy; in a small child keep a <b>thymic remnant</b> (or excise part of it) and harvest a piece of <b>anterior pericardium</b> for the patch before heparin. '
                   'Fix it in 0.6% glutaraldehyde for a few minutes if you want it stiffer and easier to handle.</p>'),
            borrow('mt-cannulate', f'{pre}-cannulate', 'Cannulation: aorta and both cavae',
                   '<p>Heparin; <b>aortic</b> cannula high on the ascending aorta, <b>bicaval</b> venous drainage (SVC directly or through the appendage, IVC low on the right atrium), with <b>snares</b> around both cavae so the right atrium can be opened without air entering the venous line.</p>'
                   f'<p>Bicaval cannulation is the rule for any operation inside the right atrium, as here for {what}.</p>'),
            borrow('mt-clamp', f'{pre}-clamp', 'Cross-clamp and cardioplegia',
                   '<p>Mild hypothermia, cross-clamp and <b>antegrade cold cardioplegia</b> into the aortic root; tighten the caval snares. '
                   'A <b>left heart vent</b> (through the right superior pulmonary vein or across the defect) keeps the field dry from bronchial return.</p>'
                   + ev('AATS/TSRA primer: median sternotomy, aortic and bicaval cannulation, cold cardioplegia after cross-clamping; for VSD an LA or LV vent through the right superior pulmonary vein and repeat doses every 20–30 minutes.')),
            borrow('mt-ra', f'{pre}-ra', 'Right atriotomy',
                   '<p>Open the right atrium <b>parallel to the atrioventricular groove</b>, about 1 cm from it, from the appendage toward the IVC cannula; '
                   'stay away from the <b>crista terminalis and the SA node</b> at the SVC–atrial junction. Retract the edges with stay sutures.</p>'),
        ]

    def finish(pre, extra, extra_show=()):
        return [
            borrow('mt-close', f'{pre}-close', 'De-air, close the atrium, release the clamp',
                   '<p>Before the last sutures: <b>fill the left heart</b> (anaesthetist inflates the lungs), let air escape through the defect or the patch edge, then tie. '
                   'Close the right atriotomy in two layers of running polypropylene, release the snares, <b>vent the aortic root</b> and remove the cross-clamp.</p>' + extra, extra_show=extra_show),
            borrow('mt-decannulate', f'{pre}-decannulate', 'Separate from bypass; check the repair',
                   '<p>Rewarm, separate from bypass, <b>transoesophageal echo</b> for a residual shunt and valve function, then decannulate and give protamine. '
                   'Temporary <b>atrial and ventricular pacing wires</b>; one or two drains; close the pericardium loosely or leave it open.</p>', extra_show=extra_show),
        ]

    # common anatomy groups
    CH = hv(['ra', 'la', 'rv', 'lv', 'myocardium', 'aorta', 'svc', 'pa-trunk'])
    CH_OP = {'myocardium': 0.06, 'ra': 0.12, 'la': 0.22, 'rv': 0.14, 'lv': 0.14, 'pa-trunk': 0.35, 'aorta': 0.45, 'svc': 0.5}
    TV = hv(['tricuspid-annulus', 'tv-anterior', 'tv-posterior', 'tv-septal'])
    KOCH = hv(['koch', 'tv-avnode', 'cs-ostium', 'coronary-sinus'])

    # =================================================================================================== ASD
    AC = P('asd-c'); ASD = ['asd-defect', 'asd-shunt']
    a_dir = V(0.8, 0.5, 0.3)
    asd_v = lambda d=a_dir, dist=160: view(AC, d, dist)
    patho = step('asd-patho', 'Pathophysiology', 'Pathophysiology: atrial septal defect',
        '<p>A hole in the atrial septum lets blood cross from the <b>left atrium to the right</b>. The direction and size of the shunt are set less by the hole than by the <b>compliance of the two ventricles</b>: the thin RV fills more easily, so blood follows it, mostly in late diastole.</p>'
        + chain('Left-to-right shunt', 'RA, RV and pulmonary arteries carry 1.5–3 times the systemic flow', 'RA and RV dilate', '!AF, RV failure, and in a minority pulmonary vascular disease')
        + h4('Types') + ul('<b>Secundum</b> (fossa ovalis): the commonest; the only type suitable for a device',
                           '<b>Primum</b>: a partial atrioventricular septal defect, with a cleft mitral valve',
                           '<b>Sinus venosus</b>: superior (with anomalous right upper pulmonary veins) or inferior',
                           '<b>Coronary sinus</b> (unroofed): rare')
        + '<p><b>Clinical</b>: often silent in childhood. A <b>wide, fixed split S2</b>, a pulmonary flow murmur, RV heave; ECG with incomplete RBBB and right axis (left axis suggests primum). Adults present with breathlessness, palpitations from AF/flutter, or paradoxical embolism.</p>'
        + ADULT,
        asd_v(), show=[*CH, *ASD], highlight=['asd-defect'], labels=['asd-defect', 'asd-shunt', 'ra', 'la'], opacity=CH_OP, after=True, spin=True, ct=AC,
        quiz=ask('What mainly decides how much blood crosses a large ASD?', 'The relative compliance (filling) of the two ventricles',
                 'With a large defect the atria share one pressure; flow goes to the ventricle that fills more easily, the RV.', 'The size of the hole alone', 'Systolic LV pressure', 'The heart rate'))
    anat = step('asd-anatomy', 'Anatomy', 'The septum seen from the right atrium',
        '<p>Through a right atriotomy the <b>fossa ovalis</b> is the target. Know its neighbours before you place a stitch:</p>'
        + ul('<b>Superior / anterosuperior</b>: the SVC orifice and, behind the septum, the <b>aortic root</b> (the "aortic rim")',
             '<b>Inferior</b>: the <b>IVC</b> orifice with the <b>Eustachian valve</b> in front of it, a trap: sew the patch to it and the IVC drains into the left atrium',
             '<b>Posterior</b>: the right pulmonary veins (a sinus venosus defect sits here, by the SVC)',
             '<b>Anteroinferior</b>: the <b>coronary sinus</b> and <b>Koch\'s triangle</b> (tendon of Todaro, CS ostium, septal tricuspid annulus); the <b>AV node</b> lies at its apex')
        + '<p>A secundum defect can be stitched anywhere on its rim except where the rim is deficient; never let sutures stray into Koch\'s triangle.</p>' + ADULT,
        asd_v(dist=150), show=[*CH, 'asd-defect', *TV, *KOCH], highlight=['asd-defect'], danger=hv(['tv-avnode', 'koch']), labels=hv(['asd-defect', 'koch', 'tv-avnode', 'cs-ostium', 'svc', 'tv-septal']),
        opacity={**CH_OP, 'ra': 0.08}, spin=True, ct=AC)
    case = step('asd-case', 'Case', 'Case: a large secundum defect in a young woman',
        '<p>Right heart volume overload from a large secundum ASD: closure is indicated. The question is <b>how</b>.</p>'
        + ev('ESC 2020: with Qp:Qs over 1.5 and PVR under 3 WU closure is class I; PVR 3–5 WU, IIa; PVR 5 WU or more despite treatment, closure is contraindicated (III), except a fenestrated closure when PVR falls below 5 WU on targeted therapy (IIb). Device closure needs rims of more than 5 mm and a defect no larger than about 38 mm (StatPearls).'),
        asd_v(), show=[*CH, *ASD], highlight=['asd-defect'], labels=['asd-defect', 'ra', 'rv'], opacity=CH_OP, ct=AC,
        lead='<p>A <b>26-year-old woman</b> referred from a county hospital with breathlessness on climbing stairs and palpitations. Fixed split S2, pulmonary flow murmur. ECG: incomplete RBBB. '
             'Echo: <b>secundum ASD 34 mm</b>, absent aortic rim and a <b>deficient (3 mm) inferior rim</b>, dilated RA and RV, estimated PA systolic pressure 40 mmHg. Right heart catheter: Qp:Qs 2.3, PVR 2 WU.</p>',
        quiz=ask('What is the best option?', 'Surgical patch closure: the inferior rim is too deficient for a device',
                 'Shunt and normal PVR make closure class I; a deficient inferior (IVC) rim makes device closure unsafe, while an absent aortic rim alone is often tolerated by devices.', 'Device closure', 'Observe: she is young', 'Sildenafil and review'))
    dec = step('asd-decision', 'Decision', 'Close or not; device or surgery',
        tbl(['Question', 'Answer'],
            ['Close?', 'RV volume overload with Qp:Qs over 1.5: yes when PVR is under 3 WU (I) and usually at 3–5 WU (IIa). PVR 5 WU or more despite therapy: no (III)'],
            ['Device?', 'Secundum only, rims over 5 mm (the inferior/IVC and posterior rims matter most), defect up to about 38 mm'],
            ['Surgery?', 'Primum, sinus venosus, coronary sinus defects; secundum with deficient rims, very large, or with another lesion to fix (TR, anomalous veins)'],
            ['When, in a child?', 'Usually electively at 2–5 years (device or surgery); earlier if there is heart failure'],
            ['Patch or stitch?', 'Small defects with good rims: direct closure. Larger: autologous pericardial (or PTFE/Dacron) patch, so the septum is not distorted'])
        + ev('ESC 2020 adult congenital guideline (PVR thresholds); StatPearls 2025 (device criteria); AATS/TSRA primer (pericardium or Gore-Tex patch, running polypropylene).'),
        asd_v(), show=[*CH, *ASD], highlight=['asd-defect'], labels=['asd-defect'], opacity=CH_OP, ct=AC)
    inspect = step('asd-inspect', 'Septum', 'Inspect the defect and its rims',
        '<p>With the atrium open and the field vented: confirm a <b>secundum</b> defect, measure it, and look at every rim. Check the <b>pulmonary veins</b> drain into the left atrium (sinus venosus defects carry anomalous right upper veins), look for a <b>left SVC</b> (a large coronary sinus) and for other fenestrations.</p>'
        '<p>Identify the <b>IVC orifice and the Eustachian valve</b> before the first stitch: the inferior rim of the patch goes to the true atrial septum, not the Eustachian valve.</p>',
        asd_v(dist=140), show=[*CH, 'asd-defect', *TV, *KOCH, 'ra-incision'], highlight=['asd-defect'], danger=hv(['tv-avnode']), labels=hv(['asd-defect', 'cs-ostium', 'tv-avnode']),
        opacity={**CH_OP, 'ra': 0.06}, ct=AC, after=True,
        quiz=ask('Sewing the inferior edge of the patch to the Eustachian valve causes what?', 'IVC blood diverted into the left atrium: cyanosis after the operation',
                 'The Eustachian valve guards the IVC orifice; the patch must go to the septal rim, or the IVC is baffled to the left side.', 'Complete heart block', 'Tricuspid stenosis', 'Nothing'))
    patch = step('asd-patch', 'Septum', 'Sew in the pericardial patch',
        '<p>Trim the pericardium a little larger than the defect. Start at the <b>inferior (IVC) end</b>, where exposure is hardest, with a running <b>5-0 polypropylene</b> suture, and run both arms up each side to meet superiorly. '
        'Stay superficial near the aortic rim (the aortic root is just behind) and away from Koch\'s triangle.</p>'
        '<p>Before tying: <b>de-air the left atrium</b>: the anaesthetist inflates the lungs and blood and air are allowed out at the last stitch.</p>'
        + ev('AATS/TSRA primer: autologous pericardium or Gore-Tex, running polypropylene; left side de-aired through the aortic root vent before the suture line is completed.'),
        asd_v(dist=140), show=[*CH, 'asd-defect', 'asd-patch', 'asd-suture', *TV, *KOCH], highlight=['asd-patch'], danger=hv(['tv-avnode']), labels=['asd-patch', 'asd-suture'],
        opacity={**CH_OP, 'ra': 0.06}, ct=AC, action={'kind': 'reveal', 'label': 'Sew in the patch', 'port': 'sternotomy', 'ids': ['asd-patch', 'asd-suture']})
    proc('asd-patch', 'asd', 'Atrial septal defect closure', 'Secundum ASD: pericardial patch (sternotomy, bypass)',
         'Surgical patch closure of a secundum ASD with deficient rims: indications, device or surgery, the rims and Koch\'s triangle, bicaval bypass, patch, de-airing.',
         [('Patho', 'other', [patho]), ('Anatomy', 'other', [anat]), ('Case', 'other', [case]), ('Decide', 'other', [dec]),
          ('Sternotomy', 'other', bypass('asd', 'an ASD')[:1]), ('Bypass', 'other', bypass('asd', 'an ASD')[1:3]), ('Atrium', 'other', bypass('asd', 'an ASD')[3:]),
          ('Inspect', 'other', [inspect]), ('Patch', 'other', [patch]), ('Close', 'other', finish('asd', '', ['asd-patch', 'asd-suture']))], ASD_SRC)

    # =================================================================================================== VSD
    VC = P('vsd-c'); VSD = ['vsd-defect', 'vsd-shunt']
    v_dir = V(0.85, 0.4, 0.2)
    vsd_v = lambda d=v_dir, dist=140: view(VC, d, dist)
    CUSPS = hv(['cusp-r', 'cusp-n', 'cusp-l'])
    VOP = {**CH_OP, 'ra': 0.06, 'rv': 0.1, 'tv-septal': 0.5, 'tv-anterior': 0.5, 'tv-posterior': 0.5}
    patho = step('vsd-patho', 'Pathophysiology', 'Pathophysiology: ventricular septal defect',
        '<p>The commonest congenital lesion, and the commonest at KNH. Systolic LV pressure drives blood into the RV and on to the lungs, so the <b>left atrium and left ventricle</b> carry the volume load.</p>'
        + chain('Large left-to-right shunt', 'High pulmonary flow and pressure', 'Heart failure, poor growth, chest infections in infancy', '!Pulmonary vascular disease, then reversal (Eisenmenger)')
        + h4('Types (by position, seen from the RV)') + ul('<b>Perimembranous</b>: by far the commonest; below the aortic valve, under the septal tricuspid leaflet',
                                                           '<b>Muscular</b>: often multiple; many close spontaneously', '<b>Outlet (doubly committed, subarterial)</b>: below both semilunar valves; aortic cusp prolapse and regurgitation',
                                                           '<b>Inlet</b>: beneath the septal leaflet, as in AV septal defects')
        + '<p><b>Size decides the course</b>: a small (restrictive) defect gives a loud murmur and little else (risk: endocarditis); a large one causes heart failure in the first months and, if left, irreversible pulmonary vascular disease, often by about 1–2 years.</p>' + ADULT,
        vsd_v(), show=[*CH, *VSD, *TV, *CUSPS], highlight=['vsd-defect'], labels=['vsd-defect', 'vsd-shunt', 'lv', 'rv'], opacity=VOP, after=True, spin=True, ct=VC,
        quiz=ask('Which chambers carry the volume load of a VSD?', 'The left atrium and left ventricle',
                 'Shunted blood passes through the lungs and returns to the LA and LV; the RV ejects it straight on in systole.', 'The right atrium and right ventricle', 'The right atrium alone', 'None'))
    anat = step('vsd-anatomy', 'Anatomy', 'The perimembranous defect and the conduction tissue',
        '<p>Through the tricuspid valve, a perimembranous defect lies under the <b>septal leaflet</b>, at the junction of its septal and anterior leaflets. Its rims:</p>'
        + ul('<b>Superior / anterosuperior</b>: the <b>aortic valve</b> (right and non-coronary cusps) just on the other side: deep bites here catch a cusp',
             '<b>Posteroinferior</b>: the <b>His bundle</b> runs along this rim, on its left ventricular side, from the apex of Koch\'s triangle: the danger zone',
             '<b>Septal leaflet and chordae</b>: often attached to the rim; may need to be detached and resuspended')
        + '<p>Rule: in the danger zone, <b>shallow bites on the RV side, 3–5 mm back from the edge</b>, or use the tricuspid leaflet tissue itself.</p>' + ADULT,
        vsd_v(dist=125), show=[*CH, 'vsd-defect', *TV, *KOCH, 'his-bundle', *CUSPS], highlight=['vsd-defect'], danger=hv(['his-bundle', 'tv-avnode', 'cusp-r', 'cusp-n']),
        labels=hv(['vsd-defect', 'his-bundle', 'tv-septal', 'cusp-n', 'cusp-r', 'tv-avnode']), opacity=VOP, spin=True, ct=VC)
    case = step('vsd-case', 'Case', 'Case: an infant with a large perimembranous VSD',
        '<p>A large unrestrictive defect with heart failure and poor growth: close it now, before pulmonary vascular disease sets in.</p>'
        + ev('KNH (Osano et al., preprint 2025): 1,703 admissions with congenital heart disease 2016–2021, VSD the commonest; median wait for surgery 59 days; only 37.1% had the recommended operation within a year; one-year mortality 36.1%. Delay costs lives and operability.'),
        vsd_v(), show=[*CH, *VSD, *TV], highlight=['vsd-defect'], labels=['vsd-defect', 'vsd-shunt'], opacity=VOP, ct=VC,
        lead='<p>A <b>7-month-old boy, 5.2 kg</b> (below the 3rd centile), on furosemide and spironolactone. Fast breathing, sweating with feeds, two admissions with pneumonia. Pansystolic murmur, hepatomegaly. '
             'Echo: <b>perimembranous VSD 9 mm</b>, unrestrictive (low gradient), dilated LA and LV, no aortic cusp prolapse. Systolic PA pressure near systemic, but the shunt is <b>left to right</b> throughout and he is not cyanosed.</p>',
        quiz=ask('What next?', 'Surgical closure soon, on bypass',
                 'Heart failure and failure to thrive despite medical therapy with a large left-to-right shunt: close it in infancy, before the pulmonary vessels remodel.', 'Wait for spontaneous closure', 'Pulmonary artery band as routine', 'Increase diuretics and review at 2 years'))
    LOCS = hv(['vsd-loc-1', 'vsd-loc-3', 'vsd-loc-4', 'vsd-loc-g'])
    types = step('vsd-types', 'Anatomy', 'Five types of VSD: where they sit',
        tbl(['Type', 'Other names', 'Where', 'Conduction tissue', 'Usual surgical route'],
            ['<b>1. Subarterial</b>', 'Outlet, supracristal, conal, doubly committed juxta-arterial', 'Under the pulmonary and aortic valves; roof is the two valves', 'Away from the defect', 'Through the pulmonary trunk (or RV outflow); watch aortic cusp prolapse'],
            ['<b>2. Perimembranous</b>', 'Paramembranous, conoventricular, membranous', 'Membranous septum, under the septal tricuspid leaflet and the right/non-coronary cusps', '<b>Posteroinferior rim</b>, left side of the septum', 'Right atrium, through the tricuspid valve'],
            ['<b>3. Inlet</b>', 'AV canal type', 'Under the septal tricuspid leaflet, behind the perimembranous zone; often with AV septal defect', 'Inferior and posterior (from a posterior AV node)', 'Right atrium, through the tricuspid valve'],
            ['<b>4. Muscular</b>', 'Trabecular', 'Wholly surrounded by muscle: mid, apical, anterior, posterior; often multiple ("Swiss cheese")', 'Usually remote', 'Right atrium; apical ones by LV apex, device, or hybrid'],
            ['<b>Gerbode</b>', 'LV to right atrial communication', 'Atrioventricular part of the membranous septum, above the tricuspid hinge', 'Close by: the AV node region', 'Right atrium'])
        + '<p>Type 2 is about 80% of operated defects. Muscular defects are common at birth (about 20% overall) but many close on their own, so fewer reach surgery.</p>'
        + ev('STS Cardiothoracic Surgery Consult: types 1–4; operated proportions type 2 80%, type 1 7%, type 4 4%, type 3 2%, multiple 5%; conduction in perimembranous defects along the posteroinferior rim, on the left side. Congenital Heart Surgery Nomenclature project (Jacobs 2000) adds the Gerbode (LV–RA) type. ICD-11 (Lopez 2018) renames them outlet, central perimembranous, inlet and trabecular muscular.'),
        vsd_v(dist=150), show=[*CH, 'vsd-defect', *LOCS, *TV, *CUSPS, 'his-bundle'], highlight=['vsd-defect', *LOCS], danger=hv(['his-bundle']),
        labels=['vsd-defect', *LOCS], opacity={**VOP, 'tv-septal': 0.3, 'tv-anterior': 0.3, 'tv-posterior': 0.3}, spin=True, ct=VC, after=True,
        quiz=ask('Which VSD type is most likely to cause aortic regurgitation?', 'Type 1, subarterial (doubly committed)',
                 'The unsupported right coronary cusp prolapses into the defect; close it even when the shunt is small.', 'Type 4, muscular', 'Type 3, inlet', 'Gerbode defect'))
    size = step('vsd-size', 'Anatomy', 'Size, restriction and what the heart shows',
        tbl(['Size', 'Defect vs aortic annulus', 'Physiology', 'Qp:Qs', 'Echo / clinical'],
            ['<b>Small</b>', 'Under 1/3', 'Restrictive: high LV–RV gradient, normal RV pressure', 'Under 1.5', 'Loud murmur, normal LA and LV size'],
            ['<b>Moderate</b>', '1/3 to 2/3', 'Partly restrictive', '1.5–3', '<b>LA and LV dilated</b>, RV pressure mildly raised'],
            ['<b>Large</b>', 'Over 2/3 (about the size of the annulus)', 'Non-restrictive: LV and RV pressures equal', 'Over 3 (falls as PVR rises)', 'Heart failure, failure to thrive, pulmonary hypertension'])
        + '<p><b>LA and LV dilatation are the echo signature of a haemodynamically important shunt</b>: the shunted blood returns through the lungs to the left heart. A restrictive defect with normal left heart size is followed, not closed.</p>'
        + '<p>Beware a large defect with a <b>small heart and little murmur</b> in an older child: the shunt has fallen because pulmonary resistance has risen.</p>'
        + ev('Size relative to the aortic annulus: WFSA tutorial 316 (Rolo 2015). Qp:Qs bands and LA/LV enlargement as evidence of a significant shunt: STS Consult.'),
        vsd_v(), show=[*CH, *VSD], highlight=['vsd-defect'], labels=['vsd-defect', 'la', 'lv'], opacity={**VOP, 'la': 0.35, 'lv': 0.3}, ct=VC, after=True,
        quiz=ask('A 6-year-old with a perimembranous VSD has a loud murmur, LV–RV gradient of 80 mmHg and normal LA and LV size. Plan?', 'Follow up; no closure (small restrictive defect)',
                 'No left heart volume load and normal RV pressure: watch for aortic cusp prolapse and endocarditis.', 'Close on bypass now', 'PA banding', 'Device closure now'))
    dec = step('vsd-decision', 'Decision', 'Who needs closure, and how',
        tbl(['Situation', 'Plan'],
            ['Large VSD, heart failure, poor growth', 'Surgical closure in infancy (usually within the first 6 months to a year); PA banding only for multiple muscular defects or a very unwell infant'],
            ['<b>LA and LV dilatation</b> (left heart volume load), Qp:Qs 1.5 or more', 'Close: the key echo indication at any age, provided PVR allows'],
            ['Small restrictive VSD, normal heart size', 'Observe; endocarditis prevention (dental care)'],
            ['Adult or older child, LV volume overload', 'Close if PVR is under 3 WU (I) or 3–5 WU (IIa); PVR 5 WU or more: individual decision in an expert centre (IIb)'],
            ['Outlet VSD with aortic cusp prolapse or AR', 'Close even when the shunt is small, to protect the valve'],
            ['Device?', 'Muscular defects, and selected perimembranous ones in experienced hands; heart block after device closure of perimembranous defects limits its use'])
        + ev('ESC 2020 (PVR thresholds for VSD as for ASD and PDA; >5 WU individual decision, IIb). Conduction risk: permanent complete heart block in 3.5% of 400 surgical closures (Azab 2013), more with large perimembranous defects and small infants.'),
        vsd_v(), show=[*CH, *VSD], highlight=['vsd-defect'], labels=['vsd-defect'], opacity=VOP, ct=VC)
    expose = step('vsd-expose', 'Defect', 'Expose the defect through the tricuspid valve',
        '<p>Retract the anterior and septal leaflets with fine stay sutures. Look for the defect under the septal leaflet; tethering chordae may hide its edges. If they do, <b>detach the septal leaflet</b> 2 mm from the annulus and resuspend it at the end.</p>'
        '<p>Name the rims aloud: aortic valve above, conduction tissue posteroinferiorly, the muscular septum below.</p>'
        + ev('AATS/TSRA primer: anterior and septal tricuspid leaflets retracted with 6-0 polypropylene stay sutures.'),
        vsd_v(dist=115), show=[*CH, 'vsd-defect', *TV, *KOCH, 'his-bundle', *CUSPS, 'ra-incision'], highlight=['vsd-defect'], danger=hv(['his-bundle', 'cusp-r', 'cusp-n']),
        labels=hv(['vsd-defect', 'tv-septal', 'his-bundle']), opacity=VOP, ct=VC)
    patch = step('vsd-patch', 'Defect', 'Patch the defect; protect the His bundle',
        '<p>A <b>Dacron, PTFE or treated pericardial patch</b> a little larger than the defect, on the RV side. Interrupted pledgeted sutures or a running 5-0/6-0 polypropylene.</p>'
        + ul('<b>Posteroinferior rim</b>: shallow bites on the RV side, 3–5 mm from the edge (the bundle runs on the LV side of this rim)',
             '<b>Superior rim</b>: bites through the fibrous tissue at the aortic valve hinge, without catching a cusp',
             '<b>Across the tricuspid annulus</b>: transition sutures through the base of the septal leaflet')
        + '<p>Test the tricuspid valve with saline and resuspend a detached leaflet. Watch the rhythm when the heart beats again.</p>'
        + ev('TSRA primer: the conduction system runs close to the posteroinferior edge: partial-thickness bites. Yoneyama 2022: in perimembranous outlet defects the conduction tissue deviates toward the LV side, which favours shallow continuous suturing.'),
        vsd_v(dist=115), show=[*CH, 'vsd-defect', 'vsd-patch', 'vsd-suture', *TV, *KOCH, 'his-bundle', *CUSPS], highlight=['vsd-patch'], danger=hv(['his-bundle']),
        labels=['vsd-patch', 'vsd-suture', 'his-bundle'], opacity=VOP, ct=VC, after=True,
        action={'kind': 'reveal', 'label': 'Sew in the patch', 'port': 'sternotomy', 'ids': ['vsd-patch', 'vsd-suture']},
        quiz=ask('Where along a perimembranous VSD is the His bundle at risk?', 'The posteroinferior rim, near the apex of Koch\'s triangle',
                 'The bundle penetrates at the apex of Koch\'s triangle and runs along the posteroinferior margin of the defect.', 'The superior rim, under the aortic valve', 'The anterior muscular rim', 'Only in muscular VSDs'))
    proc('vsd-pm', 'vsd', 'Ventricular septal defect closure', 'Perimembranous VSD: transatrial patch closure',
         'Closure of a large perimembranous VSD through the right atrium and tricuspid valve: timing, operability (PVR), the His bundle, the aortic valve, patch technique.',
         [('Patho', 'other', [patho]), ('Types', 'other', [types]), ('Size', 'other', [size]), ('Anatomy', 'other', [anat]), ('Case', 'other', [case]), ('Decide', 'other', [dec]),
          ('Sternotomy', 'other', bypass('vsd', 'a VSD')[:1]), ('Bypass', 'other', bypass('vsd', 'a VSD')[1:3]), ('Atrium', 'other', bypass('vsd', 'a VSD')[3:]),
          ('Expose', 'other', [expose]), ('Patch', 'other', [patch]),
          ('Close', 'other', finish('vsd', '<p>Check the rhythm off bypass: <b>complete heart block</b> needs temporary pacing; if it persists beyond about 7–10 days, a permanent pacemaker.</p>', ['vsd-patch', 'vsd-suture']))], VSD_SRC)

    # =================================================================================================== left thoracotomy common to PDA and coarctation
    LCH = hv(['aorta', 'lsca', 'lcca', 'pa-trunk', 'pa-left', 'n-vagus', 'n-rln', 'n-phrenic', 'esophagus', 'thoracic-duct'])
    RIBS = hv([f'rib-{i}-l' for i in range(2, 7)] + ['scapula-l'])
    THX_OP = {'lul': 0.12, 'lll': 0.12, 'aorta': 0.85, 'pa-trunk': 0.5, 'pa-left': 0.6, 'esophagus': 0.5, 'scapula-l': 0.35, **{r: 0.6 for r in RIBS if r.startswith('rib')}}
    left = lambda t, dist=170, d=(-1, -0.35, 0.25): view(t, d, dist)

    def thoracotomy(pre, ics):
        nth = '3rd' if ics == 3 else '4th'
        ribs = [f'rib-{ics}-l', f'rib-{ics + 1}-l']
        body = ('<p><b>Right lateral decubitus</b>, left arm forward and up. A posterolateral incision from below the axilla, curving 2 cm below the tip of the scapula. Divide latissimus dorsi (or retract it in a muscle-sparing approach), spare serratus anterior where possible.</p>'
                f'<p><b>Count the ribs</b> from inside or under the scapula (the first rib you can feel under the scapula is usually the 2nd) and enter the <b>{nth} intercostal space</b> on the upper border of the {ics + 1}th rib. '
                'Retract the lung forward and down with a moist pack.</p>')
        return step(f'{pre}-thoracotomy', 'Access', f'Left posterolateral thoracotomy, {nth} space', body,
                    left(P(f'thor{ics}-l') * 0.5 + P('pda-c') * 0.5, 260), show=[*RIBS, f'incision-l{ics}', 'lul', 'lll', *LCH], highlight=[f'incision-l{ics}'], labels=[f'incision-l{ics}', *ribs],
                    opacity=THX_OP, action={'kind': 'thoracotomy', 'label': 'Open the chest', 'port': f'thor{ics}-l', 'incision': f'incision-l{ics}', 'ribs': ribs, 'show': ribs})

    def which_space(pre, target, extra_show=()):
        return step(f'{pre}-space', 'Decision', 'Which space: 3rd or 4th?',
            tbl(['', '3rd space', '4th space'],
                ['Lies over', 'The distal arch, subclavian origin and the top of the duct', 'The isthmus, the duct and the upper descending aorta'],
                ['Suits', 'Neonates and small infants; a high duct; repair that must reach the distal arch', 'Most children; the standard space for PDA ligation and coarctation repair'],
                ['Drawback', 'Cramped under the scapula in a bigger child; the descending aorta is far below', 'The distal arch is further away: retract the lung down, the aorta up'],
                ['If you are wrong', 'Too high: working down a tunnel', 'Too low (5th): the duct is out of reach behind the hilum'])
            + '<p>Both are accepted; the duct and isthmus lie behind the 3rd–4th spaces, and the choice follows the child\'s size and how much arch must be exposed. Counting accurately matters more than the choice.</p>'
            + ev('AATS/TSRA primer: "ribs are counted accurately to enter the third or fourth intercostal space" for PDA; coarctation through the third or fourth. Thoracic Key: a limited posterolateral thoracotomy through the fourth space is the one more commonly used for PDA. Subbian 2024: preterm ligation through the 3rd or 4th space. Farag 2019: coarctation through the 4th space in 292 children.'),
            left(target, 300), show=[*RIBS, 'incision-l3', 'incision-l4', *LCH, *extra_show], highlight=['incision-l3', 'incision-l4'], labels=['incision-l3', 'incision-l4', 'rib-3-l', 'rib-4-l', 'rib-5-l'],
            opacity={**THX_OP, 'lul': 0.08, 'lll': 0.08}, after=True,
            quiz=ask('Why does accurate rib counting matter more than whether you choose the 3rd or 4th space?', 'Entering the 5th space by mistake puts the duct and isthmus out of reach',
                     'The target lies behind the 3rd–4th spaces; one space too low and you work up behind the hilum.', 'The 3rd space is always wrong', 'The 4th space damages the phrenic nerve', 'It does not matter at all'))

    # =================================================================================================== PDA
    DC = P('pda-c'); PDA = ['pda']
    patho = step('pda-patho', 'Pathophysiology', 'Pathophysiology: patent ductus arteriosus',
        '<p>The fetal duct carries RV output past the lungs into the descending aorta. After birth, rising oxygen and falling prostaglandin E2 close it within days. If it stays open, flow reverses: <b>aorta to pulmonary artery</b>, in systole and diastole.</p>'
        + chain('Continuous left-to-right shunt', 'Pulmonary over-circulation, LA and LV volume load', 'Diastolic run-off: wide pulse pressure, bounding pulses', '!Heart failure; pulmonary vascular disease; endarteritis')
        + ul('<b>Preterm</b>: common; a hemodynamically significant duct may be treated with ibuprofen or paracetamol, or ligated when this fails',
             '<b>Term and older</b>: rarely closes spontaneously; continuous "machinery" murmur under the left clavicle',
             '<b>Eisenmenger PDA</b>: reversed flow to the descending aorta gives <b>differential cyanosis</b>: pink fingers, blue toes')
        + ADULT,
        left(DC, 150), show=[*LCH, *PDA], highlight=['pda'], labels=['pda', 'aorta', 'pa-left', 'n-rln'], opacity={**THX_OP}, after=True, spin=True, ct=DC,
        quiz=ask('A large PDA with reversed shunt causes which sign?', 'Differential cyanosis: blue toes, pink fingers',
                 'Desaturated pulmonary blood enters the aorta beyond the left subclavian origin, so the lower body is cyanosed.', 'Central cyanosis of the lips only', 'Clubbing of the fingers only', 'A wide fixed split S2'))
    anat = step('pda-anatomy', 'Anatomy', 'The duct, the nerves and the wrong vessels',
        '<p>From the left thoracotomy you look down on:</p>'
        + ul('The <b>descending aorta</b> and distal arch, the <b>left subclavian artery</b> rising from it',
             'The <b>vagus nerve</b> crossing the arch and the duct; the <b>recurrent laryngeal nerve</b> leaves it and hooks <b>under the duct</b> to climb medially: at risk at every ligation',
             'The <b>phrenic nerve</b> further anterior, on the pericardium',
             'The <b>left pulmonary artery</b> below the duct; the <b>thoracic duct</b> behind, to the left of the oesophagus')
        + '<p>The classic disasters are ligating the <b>left pulmonary artery</b> or the <b>descending aorta</b> instead of the duct, especially in a small baby where the duct is as big as the aorta. See all three vessels before tying anything.</p>' + ADULT,
        left(DC, 130), show=[*LCH, *PDA], highlight=['pda'], danger=hv(['n-rln', 'n-vagus', 'pa-left']), labels=hv(['pda', 'n-vagus', 'n-rln', 'n-phrenic', 'pa-left', 'lsca', 'aorta']),
        opacity=THX_OP, spin=True, ct=DC)
    case = step('pda-case', 'Case', 'Case: a large duct in a 14-month-old',
        '<p>A large symptomatic duct with left heart dilatation in a child too small for the available device: ligate it surgically.</p>'
        + ev('ESC 2020 (adults): closure is indicated on haemodynamic grounds (LV volume overload) with PVR under 3 WU; device closure is preferred when feasible. TSRA primer: in adults with a calcified duct, closure may need bypass.'),
        left(DC, 150), show=[*LCH, *PDA], highlight=['pda'], labels=['pda'], opacity=THX_OP, ct=DC,
        lead='<p>A <b>14-month-old girl, 7.4 kg</b>, with poor weight gain and recurrent chest infections. Bounding pulses, continuous murmur at the left upper sternal edge. '
             'Echo: <b>PDA 6 mm</b>, continuous left-to-right flow, dilated LA and LV, normal arch. The cath laboratory cannot offer a device of the right size for months.</p>',
        quiz=ask('What is the plan?', 'Surgical ligation through a left thoracotomy',
                 'A large symptomatic duct with LV volume overload should be closed; if a device is not available, ligation is safe and definitive.', 'Ibuprofen course', 'Wait for spontaneous closure', 'Sternotomy and bypass'))
    dec = step('pda-decision', 'Decision', 'Medicines, device or surgery',
        tbl(['Patient', 'Usual plan'],
            ['Preterm, significant duct', 'Ibuprofen or paracetamol; ligation (or clip) when this fails or is contraindicated'],
            ['Term infant or child, large duct', 'Device closure where available and the child is big enough; surgical ligation otherwise'],
            ['Small, silent duct', 'No closure needed'],
            ['Adult, calcified or aneurysmal duct', 'Device if feasible; otherwise surgery, often on bypass through a sternotomy'],
            ['PVR 5 WU or more', 'Individual decision in an expert centre (ESC IIb)']),
        left(DC, 150), show=[*LCH, *PDA], highlight=['pda'], labels=['pda'], opacity=THX_OP, ct=DC)
    pleura = step('pda-pleura', 'Duct', 'Open the pleura behind the vagus; find the nerve',
        '<p>Incise the mediastinal pleura over the descending aorta, <b>behind (posterior to) the vagus</b>, and lift the pleural flap with the vagus forward and medially on stay sutures. '
        'The <b>recurrent laryngeal nerve</b> comes into view hooking under the duct. Keep it in the flap; do not dissect it bare.</p>'
        '<p>Identify the aorta above and below the duct, the left subclavian artery and the left pulmonary artery.</p>'
        + ev('Thoracic Key: the parietal pleura is divided longitudinally behind the vagus (or between vagus and phrenic). TSRA primer: the vagus and recurrent laryngeal nerve are seen coursing over the ductus.'),
        left(DC, 80), show=[*LCH, *PDA], highlight=['n-vagus', 'n-rln'], danger=hv(['n-rln']), labels=hv(['n-vagus', 'n-rln', 'pda', 'lsca']), opacity=THX_OP, ct=DC,
        action={'kind': 'dissect', 'label': 'Open the pleura', 'port': 'thor4-l', 'path': [R(P('pda-c') + V(-6, -14, 30)), R(P('pda-c') + V(-6, -12, 10)), R(P('pda-c') + V(-6, -10, -12))]})
    test = step('pda-test', 'Duct', 'Encircle the duct; test occlusion',
        '<p>Dissect the <b>upper and lower borders</b> of the duct with fine scissors, then pass a blunt right-angled clamp <b>gently behind it</b>, from below up, staying on the duct wall: the back wall is where it tears.</p>'
        '<p><b>Test-occlude</b> it (a vascular clamp or a ligature held, not tied) for a minute: the diastolic pressure should rise and the saturation and ECG stay steady. A fall in leg pressure or saturation means you are on the wrong vessel.</p>',
        left(DC, 75), show=[*LCH, *PDA], highlight=['pda'], danger=hv(['n-rln', 'pa-left']), labels=['pda', 'aorta', 'pa-left'], opacity=THX_OP, ct=DC, after=True,
        quiz=ask('During test occlusion the femoral (or foot) saturation probe signal disappears. What has happened?', 'The descending aorta has been clamped, not the duct',
                 'Lower-body flow stops if the clamp is on the aorta: release and re-identify the three vessels.', 'The duct is very large', 'Normal response', 'Pulmonary hypertension crisis'))
    tie = step('pda-tie', 'Duct', 'Ligate: aortic end first',
        '<p>Tie two heavy non-absorbable ligatures, the <b>aortic end first</b>, then the pulmonary end, slowly while the anaesthetist lowers the pressure a little. '
        'In a premature or small infant a <b>titanium clip</b> is enough; a short, wide or older duct is safer <b>divided between clamps and oversewn</b>.</p>'
        '<p>Check the RLN, haemostasis and the lung; close the pleura over the aorta; one chest drain (often removed early).</p>'
        + ev('Thoracic Key: two heavy Ethibond sutures; finer suture can cut through a friable duct; division between clamps and oversewing for others. TSRA primer: a clip over the aortic end, or double or triple ligation with 3-0 non-absorbable suture; older children divided between ligatures.'),
        left(DC, 75), show=[*LCH, *PDA, 'pda-ligatures'], highlight=['pda-ligatures'], danger=hv(['n-rln']), labels=['pda-ligatures', 'pda', 'n-rln'], opacity=THX_OP, ct=DC,
        action={'kind': 'reveal', 'label': 'Tie the duct', 'port': 'thor4-l', 'ids': ['pda-ligatures']})
    KT = hv([f'pda-type-{k}' for k in 'abcde'])
    lineup = (P('pda-type-a') + P('pda-type-e')) / 2
    ptypes = step('pda-types', 'Anatomy', 'Morphology: Krichenko types A to E',
        tbl(['Type', 'Shape', 'What it means for the surgeon'],
            ['<b>A, conical</b>', 'Wide aortic ampulla, narrowest at the pulmonary end; the commonest', 'Room for two ligatures; ideal for a device (it seats in the ampulla)'],
            ['<b>B, window</b>', 'Very short and wide; the aorta almost touches the PA', '<b>No length to tie</b>: a ligature can tear it or cut through. Divide between clamps and oversew; in adults, often bypass. Hard for devices'],
            ['<b>C, tubular</b>', 'Long, no constriction', 'Easy to encircle and ligate; devices may slip'],
            ['<b>D, complex</b>', 'Several constrictions', 'Measure carefully; tie or clip at the longest straight segment'],
            ['<b>E, elongated</b>', 'Long, constriction far from the aortic end (on lateral angiography)', 'Plenty of length to ligate; device sizing needs care'])
        + '<p>The classification comes from lateral angiography, but echo and CT show the same shapes. <b>Measure the narrowest point and the length before choosing ligation, division, clip or device.</b></p>'
        + ev('Krichenko et al., Am J Cardiol 1989: types A (conical), B (window), C (tubular), D (complex), E (elongated), described for planning catheter occlusion.'),
        view(lineup, (-1, 0, 0.12), 280), show=KT, highlight=KT, labels=KT, opacity={}, after=True,
        quiz=ask('Which duct is most dangerous to ligate with simple ties?', 'Type B, window: very short and wide',
                 'There is no length for two ligatures; division between clamps (or bypass in adults) is safer.', 'Type A, conical', 'Type C, tubular', 'Type E, elongated'))
    psize = step('pda-size', 'Anatomy', 'Size and haemodynamic significance',
        tbl(['Grade', 'Narrowest diameter (child or adult)', 'Clinical picture'],
            ['<b>Silent</b>', 'Tiny; found only on echo', 'No murmur, no volume load'],
            ['<b>Very small</b>', 'Under 1.5 mm', 'Often no symptoms'],
            ['<b>Small</b>', '1.5–3 mm', 'Continuous murmur, normal heart size'],
            ['<b>Moderate</b>', 'Over 3 to 5 mm', '<b>LA and LV dilated</b>, wide pulse pressure; may be symptomatic'],
            ['<b>Large</b>', 'Over 5 mm', 'Heart failure, poor growth, pulmonary hypertension; risk of pulmonary vascular disease'])
        + '<p>In a preterm baby millimetres matter less than <b>haemodynamic significance</b>: the ratio of the duct to the baby\'s size, left heart dilatation, flow reversal in the descending aorta, and the clinical state.</p>'
        + '<p>As with a VSD, <b>LA and LV dilatation</b> is the sign that the shunt matters.</p>'
        + ev('Size bands from the classification table in Fernando et al. 2013 (silent; very small under 1.5 mm; small 1.5–3 mm; moderate over 3–5 mm; large over 5 mm). ESC 2020: closure in adults on haemodynamic grounds (LV volume overload, PVR under 3 WU).'),
        left(DC, 150), show=[*LCH, *PDA], highlight=['pda'], labels=['pda', 'aorta', 'pa-left'], opacity=THX_OP, ct=DC)
    plcs = step('pda-plcs', 'Post-op', 'Post-ligation cardiac syndrome (preterm)',
        '<p>Hours after ligation in a preterm infant: <b>falling blood pressure, worsening oxygenation and ventilation</b>, often needing inotropes, typically <b>6–12 hours</b> after surgery.</p>'
        + chain('Duct tied', '!LV afterload rises suddenly', 'Preload falls (no more ductal return)', 'Immature LV cannot cope', 'Low output, hypotension, pulmonary oedema')
        + h4('Recognise and prevent') + ul('<b>Echo within 1 hour</b>: left ventricular output under about 200 mL/kg/min predicts the syndrome',
                                         '<b>Targeted milrinone</b> for low output: lowers afterload and supports contractility',
                                         'Avoid escalating pure vasoconstrictors, which raise afterload further; hydrocortisone for refractory hypotension',
                                         'Less likely after transcatheter closure, which is one reason it is growing in small preterms')
        + '<p>In older children the opposite is seen: a transient rise in blood pressure after the diastolic run-off stops.</p>'
        + ev('Thoracic Key: onset 6–12 h, related to increased LV afterload; LVO under 200 mL/kg/min within 1 h as predictor; early targeted milrinone may ameliorate it. Sathanandam et al., Circulation 2019 (ELBW infants): clinically significant PLCS in 42% after surgical ligation vs 4% after transcatheter closure.'),
        left(DC, 140, (-1, -0.15, 0.55)), show=[*LCH, *PDA, 'pda-ligatures'], highlight=['pda-ligatures'], labels=['pda-ligatures', 'aorta'], opacity=THX_OP, ct=DC, after=True,
        quiz=ask('Eight hours after PDA ligation, a 900 g preterm becomes hypotensive and needs more oxygen. Echo: LV output low. Best first treatment?', 'Milrinone (afterload reduction and inotropy), with cautious volume',
                 'Post-ligation cardiac syndrome is an afterload problem for an immature LV; vasoconstrictors make it worse.', 'High-dose noradrenaline', 'Re-open the duct', 'Fluid boluses until BP normal'))
    teach = [('Patho', 'other', [patho]), ('Types', 'other', [ptypes]), ('Size', 'other', [psize]), ('Anatomy', 'other', [anat]), ('Case', 'other', [case]), ('Decide', 'other', [dec])]
    for ics in (3, 4):
        key = f'pda-ics{ics}'; nth = '3rd' if ics == 3 else '4th'
        sp = which_space(key, DC, PDA); th = thoracotomy(key, ics)
        st = [copy.deepcopy(x) for x in (pleura, test, tie)]
        for x in st:
            x['id'] = f'{key}-{x["id"].split("-", 1)[1]}'
            if x.get('action', {}).get('port') == 'thor4-l': x['action']['port'] = f'thor{ics}-l'
        if ics == 3:
            st[2]['show'] = [*st[2]['show'], 'pda-clip']; st[2]['action'] = {'kind': 'reveal', 'label': 'Apply the clip', 'port': 'thor3-l', 'ids': ['pda-clip']}
            st[2]['labels'] = ['pda-clip', 'pda', 'n-rln']; st[2]['highlight'] = ['pda-clip']
        proc(key, 'pda', 'Patent ductus arteriosus ligation', f'PDA: left thoracotomy, {nth} space' + (' (infant, clip)' if ics == 3 else ' (standard, ligation)'),
             'Closure of a large PDA through a left posterolateral thoracotomy: indications, the 3rd or 4th space, the vagus and recurrent laryngeal nerve, test occlusion, ligation or clip.',
             [*teach, ('Which space', 'other', [sp]), ('Thoracotomy', 'other', [th]), ('Nerves', 'other', st[:1]), ('Test', 'other', st[1:2]), ('Ligate', 'other', st[2:]), ('After', 'other', [plcs])], PDA_SRC)

    # =================================================================================================== coarctation
    CC = P('coa-c'); COA = hv(['coa-segment', 'coa-shelf'])
    CO_LCH = [i for i in LCH if i != 'aorta']
    C_OP = {**THX_OP, 'aorta': 0.3}
    patho = step('coa-patho', 'Pathophysiology', 'Pathophysiology: coarctation of the aorta',
        '<p>A narrowing at the <b>isthmus</b>, opposite the duct: a shelf of thickened media and intima on the posterior wall, often with ductal tissue that constricts as the duct closes.</p>'
        + h4('Two presentations') + ul('<b>Neonate, duct-dependent</b>: lower body perfused through the duct; when it closes, <b>shock, acidosis and absent femoral pulses</b> in the first weeks. Start <b>prostaglandin E1</b> to reopen the duct, then repair.',
                                        '<b>Child or adult</b>: upper-body <b>hypertension</b>, weak and delayed femoral pulses (radio-femoral delay), headaches, cold legs; <b>collaterals</b> through the intercostals cause rib notching')
        + chain('Fixed obstruction at the isthmus', 'LV pressure load, upper-body hypertension', 'Collaterals around the block', '!Heart failure, stroke, aortic dissection, early coronary disease')
        + '<p>Associations: <b>bicuspid aortic valve</b> (common), intracranial aneurysms, Turner syndrome. Even after a perfect repair, hypertension and recoarctation need lifelong follow-up.</p>' + ADULT,
        left(CC, 170), show=[*CO_LCH, 'aorta', *COA, 'coa-collaterals'], highlight=['coa-segment'], labels=['coa-segment', 'coa-shelf', 'coa-collaterals', 'lsca'], opacity=C_OP, after=True, spin=True, ct=CC,
        quiz=ask('A 3-week-old is brought in shocked with absent femoral pulses. First drug?', 'Prostaglandin E1 infusion',
                 'Reopening the duct restores lower-body perfusion in duct-dependent coarctation; repair follows once the baby is resuscitated.', 'Adrenaline bolus only', 'Ibuprofen', 'Furosemide'))
    anat = step('coa-anatomy', 'Anatomy', 'The isthmus, the collaterals and the spinal cord',
        '<p>Through the left chest: the <b>distal arch</b> with the left subclavian artery, the <b>ligamentum</b> (or duct), the coarctation, and the dilated aorta below it. '
        'The <b>vagus and recurrent laryngeal nerve</b> cross the arch; the <b>thoracic duct</b> lies behind and to the right.</p>'
        + ul('<b>Collaterals</b>: enlarged, thin-walled intercostal arteries and chest wall vessels; they bleed from the incision onward and need control, not wholesale ligation',
             '<b>Spinal cord</b>: in a child with poor collaterals, clamping the aorta can leave the cord ischaemic; protect it (short clamp time, avoid hyperthermia, keep distal pressure)')
        + ADULT,
        left(CC, 140), show=[*CO_LCH, 'aorta', *COA, 'coa-collaterals', *hv(['adamkiewicz'])], highlight=['coa-segment'], danger=hv(['n-rln', 'n-vagus', 'coa-collaterals', 'thoracic-duct']),
        labels=hv(['coa-segment', 'coa-collaterals', 'n-vagus', 'n-rln', 'lsca', 'thoracic-duct']), opacity=C_OP, spin=True, ct=CC)
    case = step('coa-case', 'Case', 'Case: hypertension in a 9-year-old',
        '<p>Discrete juxtaductal coarctation with a significant gradient and hypertension: repair it.</p>'
        + ev('ESC 2020: in adults, repair is indicated for hypertension with an increased gradient; percutaneous stenting is preferred to surgery when technically feasible (Rev Esp Cardiol commentary). Children and small patients: surgery, since a stent cannot be dilated to adult size.'),
        left(CC, 170), show=[*CO_LCH, 'aorta', *COA, 'coa-collaterals'], highlight=['coa-segment'], labels=['coa-segment', 'coa-collaterals'], opacity=C_OP, ct=CC,
        lead='<p>A <b>9-year-old boy, 24 kg</b>, referred with headaches and a blood pressure of <b>150/95 mmHg in the right arm</b>; femoral pulses weak and delayed. '
             'Echo: <b>discrete coarctation</b> beyond the left subclavian artery, continuous Doppler flow with a diastolic tail, bicuspid aortic valve without stenosis. CT angiogram: isthmus 4 mm, normal transverse arch, intercostal collaterals.</p>',
        quiz=ask('Best treatment for this boy?', 'Surgical resection with extended end-to-end anastomosis',
                 'A child of 24 kg would outgrow a stent; a discrete coarctation with a normal arch is repaired through a left thoracotomy.', 'Stent now', 'Antihypertensives only', 'Balloon angioplasty as definitive treatment'))
    dec = step('coa-decision', 'Decision', 'Which repair?',
        tbl(['Option', 'When'],
            ['Resection and <b>extended end-to-end</b> anastomosis', 'Neonates, infants and children; removes all ductal tissue and enlarges a mildly hypoplastic distal arch'],
            ['Simple end-to-end', 'Discrete coarctation with a normal arch'],
            ['Subclavian flap aortoplasty', 'Some infants; sacrifices the left subclavian artery'],
            ['Patch or interposition graft', 'Long segments, re-operation, adults where ends cannot be brought together'],
            ['Stent', 'Adolescents and adults when anatomy allows (ESC: preferred when feasible)'])
        + ev('Farag et al., EJCTS 2019 (292 children, 4th space): extended end-to-end 42%, end-to-end 50%; perioperative mortality 2%, reintervention for recoarctation 9.9% (21% in neonates); RLN injury 2%, chylothorax 2%, paraplegia 0%; 14% on antihypertensives at follow-up.'),
        left(CC, 170), show=[*CO_LCH, 'aorta', *COA], highlight=['coa-segment'], labels=['coa-segment'], opacity=C_OP, ct=CC)
    sp = which_space('coa-eea', CC, COA)
    th = thoracotomy('coa-eea', 4)
    th['body'] += '<p>In an older child the <b>chest wall collaterals</b> bleed: control them as you go; the intercostal muscle may be several millimetres thick with dilated vessels.</p>'
    th['show'] = [i for i in th['show'] if i != 'pda'] + COA
    mob = step('coa-mobilise', 'Aorta', 'Mobilise the arch, the subclavian and the descending aorta',
        '<p>Open the pleura over the aorta behind the vagus. Mobilise the <b>distal transverse arch</b> (to beyond the left carotid), the <b>left subclavian artery</b>, and the descending aorta well below the coarctation. '
        '<b>Ligate and divide the ductus or ligamentum</b>. Control one or two pairs of intercostal arteries with loops if needed; avoid dividing them.</p>'
        '<p>Mobility is what makes a tension-free anastomosis.</p>',
        left(CC, 140, (-1, -0.15, 0.55)), show=[*CO_LCH, 'aorta', *COA, 'coa-collaterals'], highlight=['coa-segment', 'lsca'], danger=hv(['n-rln', 'n-vagus', 'coa-collaterals']),
        labels=hv(['coa-segment', 'lsca', 'n-vagus', 'n-rln', 'coa-collaterals']), opacity=C_OP, ct=CC,
        action={'kind': 'dissect', 'label': 'Mobilise the aorta', 'port': 'thor4-l', 'path': [R(P('coa-top') + V(-8, -6, 8)), R(CC + V(-10, -6, 0)), R(CC + V(-10, -6, -30))]})
    clamp = step('coa-clamp', 'Aorta', 'Clamp: distal arch and descending aorta',
        '<p>Small dose of heparin. <b>Proximal clamp</b> across the distal arch, beyond the left carotid, taking the left subclavian origin; <b>distal clamp</b> on the descending aorta below the dilated segment. '
        'Note the time: the cord depends on collaterals while the aorta is clamped.</p>'
        + ul('Keep the patient mildly cool (about 35 °C), never hyperthermic', 'Keep the clamp time short (aim for under about 20–30 minutes)', 'Poor collaterals or a long repair: consider distal perfusion (left heart bypass)')
        + ev('Spinal cord injury after coarctation repair is rare (0% in Farag 2019; historically about 0.4% in a large survey, Brewer 1972) but devastating; risk rises with long clamp times and poor collaterals.'),
        left(CC, 140, (-1, -0.15, 0.55)), show=[*CO_LCH, 'aorta', *COA, 'coa-clamps'], highlight=['coa-clamps'], labels=['coa-clamps', 'coa-segment', 'lsca'], opacity=C_OP, ct=CC, after=True,
        action={'kind': 'reveal', 'label': 'Apply the clamps', 'port': 'thor4-l', 'ids': ['coa-clamps']},
        quiz=ask('Which patient is at most risk of paraplegia during coarctation clamping?', 'One with few collaterals and a long clamp time',
                 'Good collaterals carry blood to the cord around the clamps; without them, time and temperature decide.', 'One with large intercostal collaterals', 'An infant repaired in 10 minutes', 'Any patient with hypertension'))
    resect = step('coa-resect', 'Aorta', 'Excise the coarctation and the ductal tissue',
        '<p>Divide the aorta above and below the narrowing and <b>excise the whole coarctation with all ductal tissue</b> (left behind, it constricts and causes recoarctation). '
        'For an <b>extended</b> repair, incise the <b>underside of the arch</b> proximally, toward or beyond the left carotid origin, and cut the descending aorta obliquely to match.</p>',
        left(CC, 130, (-1, -0.15, 0.55)), show=[*CO_LCH, 'aorta', *COA, 'coa-clamps'], highlight=['coa-segment'], labels=['coa-segment', 'coa-shelf'], opacity=C_OP, ct=CC,
        action={'kind': 'decorticate', 'label': 'Excise the coarctation', 'port': 'thor4-l', 'ids': COA})
    anast = step('coa-anastomose', 'Aorta', 'Extended end-to-end anastomosis',
        '<p>Bring the descending aorta up to the arch. Running <b>polypropylene</b> (7-0 in a neonate, 5-0 or 4-0 in an older child), back wall first from inside, then the front wall. '
        'Release the <b>distal clamp first</b> (to de-air and check the suture line), then the proximal clamp slowly.</p>'
        '<p>Measure <b>arm and leg pressures</b>: the gradient should be gone. Check the RLN and the thoracic duct region for chyle; drain, close.</p>'
        + ev('TSRA primer: end-to-end anastomosis with running polypropylene, 7-0 in neonates to 4-0 in young adults; extended resection with an arch incision for a hypoplastic distal arch.'),
        left(CC, 130, (-1, -0.15, 0.55)), show=[*CO_LCH, 'aorta', 'coa-repaired', 'coa-anastomosis'], highlight=['coa-anastomosis'], labels=['coa-anastomosis', 'coa-repaired', 'lsca'], opacity=C_OP, ct=CC,
        action={'kind': 'reveal', 'label': 'Sew the anastomosis', 'port': 'thor4-l', 'ids': ['coa-repaired', 'coa-anastomosis']})
    proc('coa-eea', 'coa', 'Coarctation of the aorta repair', 'Coarctation: resection and extended end-to-end anastomosis (4th space)',
         'Repair of a juxtaductal coarctation through a left thoracotomy: neonatal and later presentations, stent or surgery, the 3rd or 4th space, collaterals and the spinal cord, extended end-to-end anastomosis.',
         [('Patho', 'other', [patho]), ('Anatomy', 'other', [anat]), ('Case', 'other', [case]), ('Decide', 'other', [dec]), ('Which space', 'other', [sp]),
          ('Thoracotomy', 'other', [th]), ('Mobilise', 'other', [mob]), ('Clamp', 'other', [clamp]), ('Resect', 'other', [resect]), ('Anastomose', 'other', [anast])], COA_SRC)
    print('  congenital: asd-patch, vsd-pm, pda-ics3, pda-ics4, coa-eea')
