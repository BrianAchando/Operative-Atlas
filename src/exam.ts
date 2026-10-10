// The exam track: timed mock MCQ papers from the atlas question bank, mock vivas built from each module's own
// case, decision, consent, operation and ICU steps, a level tag for each module (proposed mapping to the MMed
// years, edit LEVEL_OF below), and a weak-areas table from every answer the trainee has given.
// No new clinical content here: questions and model answers are the modules' own, with their sources.
import type { Procedure, Step } from './procedure.ts';
import type { Progress, User } from './account.ts';
import { trainingYear } from './logstats.ts';

type H = (tag: string, attrs?: Record<string, unknown>, ...kids: (Node | string | null | undefined)[]) => HTMLElement;
export interface ExamLog { mocks?: { at: string; n: number; right: number; secs: number; areas: string[]; level: string }[];
  vivas?: { at: string; proc: string; score: number; max: number; mode: 'self' | 'examiner' }[] }
type PX = Progress & { exam?: ExamLog };
interface Deps { h: H; procedures: Record<string, Procedure>; progress: PX; save: () => void; user: () => User | null }

// ------------------------------------------------------------------ areas, levels, question types
export const AREA_OF: Record<string, string> = {
  Cardiac: 'Adult cardiac', 'Congenital cardiac': 'Congenital', Vascular: 'Vascular', Trauma: 'Trauma and ICU', 'CTICU protocol': 'Trauma and ICU',
  'Access and positioning': 'Trauma and ICU', Lobectomy: 'Thoracic', Pneumonectomy: 'Thoracic', Segmentectomy: 'Thoracic', Pleura: 'Thoracic', Mediastinum: 'Thoracic',
  'Congenital lung lesions': 'Thoracic', 'Lung infection and cavities': 'Thoracic', 'Emphysema and bullous disease': 'Thoracic', Airway: 'Thoracic', Esophagus: 'Esophagus',
};
export const AREAS = ['Thoracic', 'Esophagus', 'Adult cardiac', 'Congenital', 'Vascular', 'Trauma and ICU'];
/** Proposed: which stage of training each group is first examined at. 1 = years 1-2, 2 = years 3-4, 3 = year 5 on. */
export const LEVEL_OF: Record<string, 1 | 2 | 3> = {
  'Access and positioning': 1, Trauma: 1, 'CTICU protocol': 1, Pleura: 1, 'Lung infection and cavities': 1, 'Emphysema and bullous disease': 1,
  Lobectomy: 2, Segmentectomy: 2, Pneumonectomy: 2, Mediastinum: 2, Airway: 2, Esophagus: 2, Cardiac: 2, Vascular: 2,
  'Congenital cardiac': 3, 'Congenital lung lesions': 3,
};
export const LEVEL_NAME = { 1: 'Foundation (years 1–2)', 2: 'Core (years 3–4)', 3: 'Advanced (year 5 on)' } as const;
const KIND = (phase: string) => (['Pathophysiology', 'Anatomy'].includes(phase) ? 'Science' : ['Case', 'Decision', 'Consent', 'Counselling', 'Treatment'].includes(phase) ? 'Judgment'
  : ['ICU', 'After', 'Wean'].includes(phase) ? 'Post-op' : 'Operative');
const KINDS = ['Science', 'Judgment', 'Operative', 'Post-op'];

const shuffle = <T,>(a: T[]) => { const r = [...a]; for (let i = r.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [r[i], r[j]] = [r[j]!, r[i]!]; } return r; };
const iso = () => new Date().toISOString();
const pct = (a: number, b: number) => (b ? Math.round((100 * a) / b) : 0);
const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

export function createExam(D: Deps) {
  const { h, procedures, progress } = D;
  const btn = h('button', { class: 'btn exam-btn', onclick: () => home(), title: 'Mock exams, mock vivas and your weak areas' }, 'Exam') as HTMLButtonElement;
  const close = () => { stopTimer(); document.getElementById('exam-panel')?.remove(); };
  const shell = (title: string, back: (() => void) | null, ...kids: (Node | null)[]) => {
    document.getElementById('exam-panel')?.remove();
    const panel = h('div', { id: 'exam-panel', class: 'overlay', role: 'dialog', 'aria-label': title },
      h('div', { class: 'overlay-box acct-box ex-box' }, h('div', { class: 'ov-head' }, back ? h('button', { class: 'btn ghost', onclick: () => { stopTimer(); back(); } }, '← Exam') : null,
        h('b', {}, title), h('button', { class: 'btn ghost', onclick: close }, 'Close')), ...kids));
    document.body.append(panel); return panel;
  };
  const log = (): ExamLog => (progress.exam ??= {});
  const myLevel = (): 1 | 2 | 3 => { const u = D.user(); if (!u || u.role !== 'resident') return 3; const y = trainingYear(u.start, new Date().toISOString().slice(0, 10)) || 2; return y <= 2 ? 1 : y <= 4 ? 2 : 3; };

  // the question bank, deduplicated by question text
  type Q = { key: string; proc: string; idx: number; step: Step; area: string; level: 1 | 2 | 3; kind: string };
  let bank: Q[] | null = null;
  const getBank = () => {
    if (bank) return bank; const seen = new Set<string>(); bank = [];
    for (const [k, p] of Object.entries(procedures)) {
      const area = AREA_OF[p.group ?? ''] ?? 'Thoracic'; const level = LEVEL_OF[p.group ?? ''] ?? 2;
      p.steps.forEach((s, i) => { if (!s.ask) return; const t = s.ask.question.trim(); if (seen.has(t)) return; seen.add(t);
        bank!.push({ key: `${k}/${s.id}`, proc: k, idx: i, step: s, area, level, kind: KIND(s.phase) }); });
    }
    return bank;
  };
  const answered = (q: Q) => progress.q[q.proc]?.[q.step.id];

  // ------------------------------------------------------------------ home
  function home(): void {
    const L = log(); const lv = myLevel();
    const mocks = (L.mocks ?? []).slice(-6).reverse(); const vivas = (L.vivas ?? []).slice(-6).reverse();
    shell('Exam', null,
      h('div', { class: 'ex-cards' },
        h('button', { class: 'ex-card', onclick: () => setup() }, h('b', {}, 'Mock MCQ paper'), h('span', {}, 'Timed, no feedback until the end; scored by area and by type of question')),
        h('button', { class: 'ex-card', onclick: () => vivaSetup() }, h('b', {}, 'Mock viva'), h('span', {}, 'Case, science, decision, consent, the operation and the ICU, from one module; alone or with an examiner')),
        h('button', { class: 'ex-card', onclick: () => weak() }, h('b', {}, 'Weak areas'), h('span', {}, 'Every answer you have given, by area and type, with a paper on your weakest'))),
      h('p', { class: 'foot' }, `Your level: ${LEVEL_NAME[lv]}. Module levels are a proposed mapping to the MMed years, for your programme to confirm.`),
      mocks.length ? h('div', {}, h('h4', {}, 'Recent papers'), h('table', { class: 'oc-nums' }, ...mocks.map((m) => h('tr', {}, h('th', {}, m.at.slice(0, 10)),
        h('td', {}, `${pct(m.right, m.n)}% (${m.right}/${m.n}) · ${mmss(m.secs)} · ${m.areas.join(', ') || 'All areas'}`))))) : null,
      vivas.length ? h('div', {}, h('h4', {}, 'Recent vivas'), h('table', { class: 'oc-nums' }, ...vivas.map((v) => h('tr', {}, h('th', {}, v.at.slice(0, 10)),
        h('td', {}, `${procedures[v.proc]?.opName ?? v.proc}: ${pct(v.score, v.max)}%${v.mode === 'examiner' ? ' (examiner)' : ''}`))))) : null);
  }

  // ------------------------------------------------------------------ mock MCQ paper
  let timer = 0; const stopTimer = () => { clearInterval(timer); timer = 0; };
  const fill = (el: HTMLElement, ...kids: (Node | null)[]) => el.replaceChildren(...(kids.filter(Boolean) as Node[]));
  function setup(pre?: { areas?: string[]; only?: (q: Q) => boolean; title?: string }): void {
    const areas = new Set<string>(pre?.areas ?? []); let n = 40; let level: 1 | 2 | 3 = myLevel(); let timed = true;
    const chip = (label: string, on: boolean, f: (b: HTMLElement) => void) => { const b = h('button', { class: `chip${on ? ' on' : ''}` }, label); b.onclick = () => f(b); return b; };
    const areaChips = h('div', { class: 'oc-tags' }, ...AREAS.map((a) => chip(a, areas.has(a), (b) => { if (areas.has(a)) areas.delete(a); else areas.add(a); b.classList.toggle('on'); count(); })));
    const lenChips = h('div', { class: 'oc-tags' }, ...[20, 40, 60, 100].map((x) => chip(`${x} questions`, x === n, (b) => { n = x; lenChips.querySelectorAll('.chip').forEach((c) => c.classList.remove('on')); b.classList.add('on'); count(); })));
    const lvSel = h('select', { onchange: (e: Event) => { level = Number((e.target as HTMLSelectElement).value) as 1 | 2 | 3; count(); } },
      ...([1, 2, 3] as const).map((l) => h('option', { value: String(l), ...(l === level ? { selected: true } : {}) }, l === 3 ? 'All levels' : `Up to ${LEVEL_NAME[l]}`))) as HTMLSelectElement;
    const tChk = h('input', { type: 'checkbox', checked: true, onchange: (e: Event) => { timed = (e.target as HTMLInputElement).checked; } });
    const info = h('p', { class: 'foot' });
    const pool = () => getBank().filter((q) => q.level <= level && (!areas.size || areas.has(q.area)) && (!pre?.only || pre.only(q)));
    const count = () => { const p = pool().length; info.textContent = `${p} questions available; the paper takes ${Math.min(n, p)}${timed ? `, ${Math.round(Math.min(n, p) * 1.5)} minutes (90 s each)` : ''}.`; };
    count();
    shell(pre?.title ?? 'Mock MCQ paper', home, h('h4', {}, 'Areas (none chosen = all)'), areaChips, h('h4', {}, 'Length'), lenChips,
      h('label', { class: 'af' }, h('span', {}, 'Level'), lvSel), h('label', { class: 'ex-chk' }, tChk, ' Timed (90 seconds a question)'), info,
      h('div', { class: 'lb-actions' }, h('button', { class: 'btn primary', onclick: () => { const p = shuffle(pool()).slice(0, n); if (p.length) paper(p, timed, [...areas], level); } }, 'Start the paper')));
  }

  function paper(qs: Q[], timed: boolean, areas: string[], level: number): void {
    const picks: (string | null)[] = qs.map(() => null); const flags = new Set<number>(); let i = 0; const t0 = Date.now(); const limit = qs.length * 90;
    const order = qs.map((q) => shuffle(q.step.ask!.choices));
    const clock = h('span', { class: 'ex-clock' }); const nav = h('div', { class: 'ex-nav' }); const body = h('div', { class: 'ex-q' });
    const finish = () => { stopTimer(); result(qs, picks, Math.round((Date.now() - t0) / 1000), areas, level); };
    const tick = () => { const el = (Date.now() - t0) / 1000; clock.textContent = timed ? `${mmss(Math.max(0, limit - el))} left` : mmss(el); if (timed && el >= limit) finish(); };
    const paintNav = () => nav.replaceChildren(...qs.map((_, j) => h('button', { class: `ex-dot${picks[j] ? ' done' : ''}${flags.has(j) ? ' flag' : ''}${j === i ? ' cur' : ''}`, onclick: () => { i = j; show(); } }, String(j + 1))));
    const show = () => {
      const q = qs[i]!; const p = procedures[q.proc]!;
      fill(body, h('p', { class: 'dq-n' }, `Question ${i + 1} of ${qs.length} · ${q.area} · ${q.kind}`),
        q.step.lead ? h('div', { class: 'body lead', html: q.step.lead }) : null,
        h('p', { class: 'dq-q' }, q.step.ask!.question),
        h('div', { class: 'dq-opts' }, ...order[i]!.map((c) => h('button', { class: `choice${picks[i] === c.text ? ' picked' : ''}`, onclick: () => { picks[i] = c.text; if (i < qs.length - 1) i++; show(); } }, c.text))),
        h('div', { class: 'lb-actions' }, h('button', { class: 'btn', disabled: i === 0, onclick: () => { i--; show(); } }, '← Back'),
          h('button', { class: `btn${flags.has(i) ? ' primary' : ''}`, onclick: () => { if (flags.has(i)) flags.delete(i); else flags.add(i); show(); } }, flags.has(i) ? 'Flagged' : 'Flag'),
          h('button', { class: 'btn', disabled: i === qs.length - 1, onclick: () => { i++; show(); } }, 'Next →'),
          h('button', { class: 'btn primary', onclick: () => { const left = picks.filter((x) => !x).length; if (!left || confirm(`${left} unanswered. Submit the paper?`)) finish(); } }, 'Submit')));
      void p; paintNav();
    };
    shell('Mock MCQ paper', home, h('div', { class: 'ex-top' }, clock, h('small', {}, `${qs.length} questions`)), nav, body);
    show(); tick(); stopTimer(); timer = window.setInterval(tick, 1000);
  }

  function result(qs: Q[], picks: (string | null)[], secs: number, areas: string[], level: number): void {
    const right = qs.map((q, j) => q.step.ask!.choices.find((c) => c.correct)!.text === picks[j]);
    const nr = right.filter(Boolean).length;
    // first answers count toward progress, as everywhere else in the atlas
    qs.forEach((q, j) => { if (picks[j] == null) return; const pq = (progress.q[q.proc] ??= {}); if (!(q.step.id in pq)) pq[q.step.id] = right[j] ? 1 : 0; });
    const L = log(); (L.mocks ??= []).push({ at: iso(), n: qs.length, right: nr, secs, areas, level: String(level) }); if (L.mocks.length > 60) L.mocks.splice(0, L.mocks.length - 60);
    D.save();
    const by = (f: (q: Q) => string, keys: string[]) => keys.map((k) => { const idx = qs.map((q, j) => [q, j] as const).filter(([q]) => f(q) === k); return [k, idx.filter(([, j]) => right[j]).length, idx.length] as const; }).filter(([, , n]) => n);
    const bars = (rows: readonly (readonly [string, number, number])[]) => h('div', { class: 'ex-bars' }, ...rows.map(([k, r, n]) => h('div', { class: 'ex-bar' }, h('span', {}, k),
      h('i', {}, h('b', { style: `width:${pct(r, n)}%`, class: pct(r, n) >= 70 ? 'ok' : pct(r, n) >= 50 ? 'mid' : 'low' })), h('small', {}, `${r}/${n}`))));
    const wrong = qs.map((q, j) => ({ q, j })).filter(({ j }) => !right[j]);
    shell('Paper result', home,
      h('p', { class: 'dq-score' }, `${pct(nr, qs.length)}%`), h('p', {}, `${nr} of ${qs.length} right · ${mmss(secs)}`),
      h('h4', {}, 'By area'), bars(by((q) => q.area, AREAS)), h('h4', {}, 'By type of question'), bars(by((q) => q.kind, KINDS)),
      wrong.length ? h('h4', {}, `Review: ${wrong.length} to go over`) : h('p', {}, 'All right.'),
      ...wrong.map(({ q, j }) => { const r = q.step.ask!.choices.find((c) => c.correct)!;
        return h('div', { class: 'ex-rev' }, h('p', { class: 'dq-q' }, q.step.ask!.question), picks[j] ? h('p', { class: 'ex-no' }, `Your answer: ${picks[j]}`) : h('p', { class: 'ex-no' }, 'Not answered'),
          h('p', { class: 'ex-ok' }, `Right: ${r.text}`), r.why ? h('p', { class: 'foot' }, r.why) : null,
          h('a', { href: `#approach=${q.proc}&step=${q.idx}`, onclick: close }, `Open the step: ${procedures[q.proc]!.opName}, ${q.step.title}`)); }),
      h('div', { class: 'lb-actions' }, h('button', { class: 'btn primary', onclick: () => setup() }, 'Another paper'), h('button', { class: 'btn', onclick: () => weak() }, 'Weak areas')));
  }

  // ------------------------------------------------------------------ weak areas
  function weak(): void {
    const B = getBank(); const cell = (a: string, k: string) => { const qs = B.filter((q) => q.area === a && q.kind === k); const ans = qs.filter((q) => answered(q) !== undefined);
      return { n: qs.length, done: ans.length, right: ans.filter((q) => answered(q) === 1).length }; };
    let worst: { a: string; k: string; acc: number } | null = null;
    const rows = AREAS.map((a) => h('tr', {}, h('th', {}, a), ...KINDS.map((k) => { const c = cell(a, k); if (!c.n) return h('td', {}, '–');
      const acc = pct(c.right, c.done); if (c.done >= 3 && (!worst || acc < worst.acc)) worst = { a, k, acc };
      return h('td', { class: c.done ? (acc >= 70 ? 'ok' : acc >= 50 ? 'mid' : 'low') : '' }, c.done ? `${acc}%` : '·', h('small', {}, ` ${c.done}/${c.n}`)); })));
    const w = worst as { a: string; k: string; acc: number } | null;
    shell('Weak areas', home, h('p', { class: 'foot' }, 'Accuracy of your first answers (anywhere in the atlas), and how many of the questions you have answered. Green 70% and above, amber 50–69%, red below 50%.'),
      h('div', { class: 'tbl-wrap' }, h('table', { class: 'lb-sum ex-heat' }, h('tr', {}, h('th', {}, ''), ...KINDS.map((k) => h('th', {}, k))), ...rows)),
      h('div', { class: 'lb-actions' },
        w ? h('button', { class: 'btn primary', onclick: () => setup({ areas: [w.a], only: (q) => q.kind === w.k && answered(q) !== 1, title: `Practice: ${w.a}, ${w.k}` }) }, `Practice your weakest: ${w.a}, ${w.k} (${w.acc}%)`) : null,
        h('button', { class: 'btn', onclick: () => setup({ only: (q) => answered(q) === undefined, title: 'Questions you have not seen' }) }, 'A paper of questions you have not seen'),
        h('button', { class: 'btn', onclick: () => setup({ only: (q) => answered(q) === 0, title: 'Questions you got wrong' }) }, 'A paper of the ones you got wrong')));
  }

  // ------------------------------------------------------------------ mock viva
  function vivaSetup(): void {
    const lv = myLevel();
    const groups = new Map<string, [string, Procedure][]>();
    for (const [k, p] of Object.entries(procedures)) { if (p.group === 'CTICU protocol' || p.group === 'Access and positioning') continue; if (!p.steps.some((s) => s.phase === 'Case')) continue;
      const g = p.group ?? 'Other'; (groups.get(g) ?? groups.set(g, []).get(g)!).push([k, p]); }
    const sel = h('select', {}, h('option', { value: '' }, `Random, up to ${LEVEL_NAME[lv]}`),
      ...[...groups].map(([g, l]) => h('optgroup', { label: `${g} · ${LEVEL_NAME[LEVEL_OF[g] ?? 2].split(' (')[0]}` }, ...l.map(([k, p]) => h('option', { value: k }, `${p.opName} · ${p.approach}`))))) as HTMLSelectElement;
    let mode: 'self' | 'examiner' = 'self';
    const modes = h('div', { class: 'rt-pick' }, h('label', {}, h('input', { type: 'radio', name: 'vm', checked: true, onchange: () => { mode = 'self'; } }), ' On my own: answer aloud, then reveal'),
      h('label', {}, h('input', { type: 'radio', name: 'vm', onchange: () => { mode = 'examiner'; } }), ' With an examiner: the model answers show at once for them to score'));
    shell('Mock viva', home, h('label', { class: 'af' }, h('span', {}, 'Module'), sel), h('h4', {}, 'Mode'), modes,
      h('p', { class: 'foot' }, 'Seven stations: the case, the science, the anatomy, the decision, consent, the operation and the ICU. Score each 0–3. The model answers are the module\'s own text, with its sources.'),
      h('div', { class: 'lb-actions' }, h('button', { class: 'btn primary', onclick: () => {
        let k = sel.value; if (!k) { const l = [...groups].filter(([g]) => (LEVEL_OF[g] ?? 2) <= lv).flatMap(([, x]) => x.map(([kk]) => kk)); k = l[Math.floor(Math.random() * l.length)]!; }
        viva(k, mode); } }, 'Start the viva')));
  }

  function viva(key: string, mode: 'self' | 'examiner'): void {
    const p = procedures[key]!; const S = p.steps.map((s, i) => ({ s, i }));
    const one = (ph: string) => S.find(({ s }) => s.phase === ph);
    const ops = S.filter(({ s }) => KIND(s.phase) === 'Operative');
    type St = { title: string; ask: string; lead?: string; model: () => HTMLElement; idx: number };
    const html = (x: string) => h('div', { class: 'body', html: x });
    const stations: St[] = [];
    const c = one('Case'); if (c) stations.push({ title: 'The case', ask: 'Present this patient: your assessment, the stage or severity, and your plan.', lead: c.s.lead, model: () => html(c.s.body), idx: c.i });
    const pa = one('Pathophysiology'); if (pa) stations.push({ title: 'The science', ask: `Tell me about ${pa.s.title.replace(/^Pathophysiology( and [a-z]+)?:?\s*/i, '').toLowerCase() || 'the pathophysiology'}.`, model: () => html(pa.s.body), idx: pa.i });
    const an = one('Anatomy'); if (an) stations.push({ title: 'Anatomy', ask: `Describe ${an.s.title.charAt(0).toLowerCase() + an.s.title.slice(1)}, and the structures at risk.`, model: () => html(an.s.body), idx: an.i });
    const de = one('Decision'); if (de) stations.push({ title: 'The decision', ask: 'What are the options for this patient, and what would you do?', model: () => html(de.s.body), idx: de.i });
    const co = one('Consent'); if (co) stations.push({ title: 'Consent', ask: `Consent this patient for ${p.opName.toLowerCase()}.`, model: () => html(co.s.body), idx: co.i });
    if (ops.length) stations.push({ title: 'The operation', ask: `Talk me through ${p.opName.toLowerCase()} (${p.approach.toLowerCase()}), step by step, with the danger points.`,
      model: () => h('ol', { class: 'prep-steps' }, ...ops.map(({ s }) => h('li', {}, h('details', {}, h('summary', {}, s.title), html(s.body))))), idx: ops[0]!.i });
    const icu = one('ICU'); if (icu) stations.push({ title: 'After the operation', ask: 'What do you watch for in the first 48 hours, and what can go wrong?', model: () => html(icu.s.body), idx: icu.i });
    const scores: (number | null)[] = stations.map(() => null); let i = 0;
    const DESC = ['0: not covered or unsafe', '1: gaps that matter', '2: safe and adequate', '3: complete and prioritized'];
    const body = h('div', { class: 'ex-q' });
    const show = () => {
      if (i >= stations.length) return done();
      const st = stations[i]!; const model = h('div', { class: 'ex-model' }, h('h4', {}, 'Model answer'), st.model(),
        h('a', { href: `#approach=${key}&step=${st.idx}`, onclick: close }, 'Open in the atlas'));
      const reveal = mode === 'examiner';
      if (!reveal) model.style.display = 'none';
      const score = h('div', { class: 'ex-score' }, h('span', {}, mode === 'examiner' ? 'Examiner\'s score:' : 'Your score:'),
        ...DESC.map((d, n) => h('button', { class: `chip${scores[i] === n ? ' on' : ''}`, title: d, onclick: () => { scores[i] = n; i++; show(); } }, String(n))), h('small', { class: 'foot' }, DESC.join(' · ')));
      if (!reveal) score.style.display = 'none';
      const rv = h('button', { class: 'btn primary', onclick: () => { model.style.display = ''; score.style.display = ''; rv.remove(); } }, 'Reveal the model answer');
      fill(body, h('p', { class: 'dq-n' }, `Station ${i + 1} of ${stations.length} · ${st.title} · ${p.opName}`),
        st.lead ? h('div', { class: 'body lead', html: st.lead }) : null, h('p', { class: 'ex-ask' }, st.ask), reveal ? null : rv, model, score,
        h('div', { class: 'lb-actions' }, i ? h('button', { class: 'btn ghost', onclick: () => { i--; show(); } }, '← Previous station') : null));
    };
    const done = () => {
      const got = scores.reduce<number>((a, b) => a + (b ?? 0), 0); const max = stations.length * 3;
      const L = log(); (L.vivas ??= []).push({ at: iso(), proc: key, score: got, max, mode }); if (L.vivas.length > 60) L.vivas.splice(0, L.vivas.length - 60); D.save();
      body.replaceChildren(h('p', { class: 'dq-score' }, `${pct(got, max)}%`), h('p', {}, `${got} of ${max} · ${p.opName}, ${p.approach}${mode === 'examiner' ? ' · scored by an examiner' : ''}`),
        h('table', { class: 'oc-nums' }, ...stations.map((s, j) => h('tr', {}, h('th', {}, s.title), h('td', { class: (scores[j] ?? 0) >= 2 ? 'ok' : 'low' }, `${scores[j] ?? 0} / 3`)))),
        h('p', { class: 'foot' }, 'A 2 in every station is a safe pass (proposed). Stations under 2: open them in the atlas and repeat the viva.'),
        h('div', { class: 'lb-actions' }, h('button', { class: 'btn primary', onclick: () => vivaSetup() }, 'Another viva'), h('button', { class: 'btn', onclick: () => viva(key, mode) }, 'Repeat this one')));
    };
    shell(`Viva: ${p.opName}`, home, body); show();
  }

  return { button: btn, open: home };
}
