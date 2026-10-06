"""British to American spelling for what the reader sees (tumor, ischemia, hemoptysis, anesthesia ...). Used by procedures.py; the same rules were run once over content/procedures/*.md."""
import re, glob, sys, collections
# generic families
FAM = [
  (r'[Hh]aem', lambda m: m.group(0)[0]+'em'),
  (r'[Oo]edem', lambda m: m.group(0)[0]+'dem'),   # placeholder, fixed below
]
def case(src, rep):
    return rep[0].upper()+rep[1:] if src[0].isupper() else rep
rules = []
def add(pat, rep): rules.append((re.compile(pat), rep))
add(r'\b(H|h)aem', lambda m: m.group(1)+'em')
add(r'\b(O|o)edem', lambda m: ('E' if m.group(1)=='O' else 'e')+'dem')
add(r'\b(I|i)schaem', lambda m: m.group(1)+'schem')
add(r'\b(O|o)esoph', lambda m: ('E' if m.group(1)=='O' else 'e')+'soph')
add(r'\b(A|a)naesth', lambda m: m.group(1)+'nesth')
add(r'\b(P|p)aediatr', lambda m: m.group(1)+'ediatr')
add(r'\b(P|p)araesthes', lambda m: m.group(1)+'aresthes')
add(r'(?<=[a-z])pnoea', 'pnea')
add(r'\b(D|d)iarrhoea', lambda m: m.group(1)+'iarrhea')
for w,r in [('bacteraemia','bacteremia'),('anaemia','anemia'),('hypovolaemia','hypovolemia'),('hypokalaemia','hypokalemia'),('oligaemia','oligemia'),
            ('polycythaemia','polycythemia'),('euvolaemic','euvolemic'),('hyperkalaemia','hyperkalemia'),('hypernatraemia','hypernatremia'),('hyponatraemia','hyponatremia'),
            ('anaemic','anemic'),('septicaemia','septicemia'),('hypercalcaemia','hypercalcemia'),('hypocalcaemia','hypocalcemia'),('hypomagnesaemia','hypomagnesemia'),
            ('hyperglycaemia','hyperglycemia'),('hypoglycaemia','hypoglycemia'),('leukaemia','leukemia')]:
    add(r'\b'+w+r'\b', r)
# -ise/-isation families, listed stems only
STEMS = 'randomis revascularis mobilis embolis generalis catheteris skeletonis recognis organis vascularis individualis heparinis optimis standardis localis stabilis ionis kocheris endothelialis oxidis hospitalis minimis colonis sterilis pressuris summaris nebulis equalis alkalinis superficialis bicuspidis arterialis'.split()
for st in STEMS:
    base = st[:-2] if st.endswith('is') else st
    add(r'\b((?:un|de|re|non|sub)?)(%s)(is)(e|ed|es|ing|ation|ations|ational|er|ers)\b' % re.escape(base), lambda m: m.group(1)+m.group(2)+'iz'+m.group(4))
    add(r'\b((?:un|de|re|non|sub)?)(%s)(ys)(e|ed|es|ing)\b' % re.escape(base), lambda m: m.group(1)+m.group(2)+'yz'+m.group(4))
for w,r in [('tumour','tumor'),('tumours','tumors'),('centre','center'),('centres','centers'),('colour','color'),('colours','colors'),('coloured','colored'),
            ('favour','favor'),('favours','favors'),('favoured','favored'),('favourable','favorable'),('unfavourable','unfavorable'),('favourably','favorably'),
            ('neighbour','neighbor'),('neighbours','neighbors'),('neighbouring','neighboring'),('armoured','armored'),
            ('fibre','fiber'),('fibres','fibers'),('millimetres','millimeters'),('millimetre','millimeter'),('centimetres','centimeters'),('centimetre','centimeter'),
            ('litre','liter'),('litres','liters'),('titre','titer'),('titres','titers'),('judgement','judgment'),('programme','program'),('programmes','programs'),
            ('manoeuvre','maneuver'),('manoeuvres','maneuvers'),('analyse','analyze'),('analysed','analyzed'),('practised','practiced'),('behaviour','behavior'),
            ('labour','labor'),('odour','odor'),('honour','honor'),('metre','meter'),('metres','meters'),('anaesthetise','anesthetize'),('leucocyte','leukocyte'),('oestrogen','estrogen')]:
    add(r'\b'+w+r'\b', r)
for w in ('Tumour','Centre','Colour','Favour'):
    pass

def fix(text):
    cnt = collections.Counter()
    # protect link targets and html hrefs
    prot=[]
    def keep(m): prot.append(m.group(0)); return '\x00%d\x00'%(len(prot)-1)
    text = re.sub(r'\]\([^)]*\)|href="[^"]*"|https?://\S+', keep, text)
    for rx, rep in rules:
        def sub(m, rep=rep):
            r = rep(m) if callable(rep) else rep
            if not callable(rep) and m.group(0)[0].isupper(): r = r[0].upper()+r[1:]
            cnt[(m.group(0), r)] += 1
            return r
        text = rx.sub(sub, text)
    text = re.sub('\x00(\d+)\x00', lambda m: prot[int(m.group(1))], text)
    return text, cnt

