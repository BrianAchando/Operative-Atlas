// The resident's day: tomorrow's theatre list with a prep pack for each case, one-tap logging after theatre, and the
// daily five questions (spaced repetition, weighted to the year of training) with a streak.
import type { Procedure, Step } from './procedure.ts';
import { api, type Progress, type Prefill, type User } from './account.ts';
import { CATALOG, OPS } from './logcat.ts';
import { trainingYear } from './logstats.ts';

type H = (tag: string, attrs?: Record<string, unknown>, ...kids: (Node | string | null | undefined)[]) => HTMLElement;
interface Plan { id: number; date: string; proc: string; op: string; note: string; log_id: number | null }
interface Deps { h: H; procedures: Record<string, Procedure>; progress: Progress; save: () => void; user: () => User | null; logCase: (p: Prefill) => void;
  onUser: (f: (u: User | null) => void) => void; nameOf: (id: string) => string }

const TEACH = new Set(['Pathophysiology', 'Anatomy', 'Case', 'Decision', 'Consent', 'ICU', 'Counselling', 'Treatment']);
const iso = (d: Date) => new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
const addDays = (s: string, n: number) => { const d = new Date(s + 'T12:00:00'); d.setDate(d.getDate() + n); return iso(d); };
const BOX_DAYS = [0, 1, 3, 7, 14, 30];                                         // Leitner boxes 1-5: days until the next review
const shuffle = <T,>(a: T[], seed: number) => { const r = [...a]; let s = seed || 1; for (let i = r.length - 1; i > 0; i--) { s = (s * 16807) % 2147483647; const j = s % (i + 1); [r[i], r[j]] = [r[j]!, r[i]!]; } return r; };
const hash = (t: string) => { let x = 2166136261; for (let i = 0; i < t.length; i++) x = Math.imul(x ^ t.charCodeAt(i), 16777619); return Math.abs(x); };

export function createToday(D: Deps) {
  const { h, procedures, progress } = D;
  const btn = h('button', { class: 'btn today-btn', onclick: () => { void open(); }, title: 'Your theatre list, prep packs and the daily questions', style: 'display:none' }, 'Today') as HTMLButtonElement;
  const dot = h('span', { class: 'today-dot' }); btn.append(dot);
  D.onUser((u) => { btn.style.display = u && u.role ? '' : 'none'; paintDot(); });

  const overlay = (title: string, ...kids: (Node | null)[]) => {
    document.getElementById('today-panel')?.remove();
    const close = () => panel.remove();
    const panel = h('div', { id: 'today-panel', class: 'overlay', role: 'dialog', 'aria-label': title },
      h('div', { class: 'overlay-box acct-box' }, h('div', { class: 'ov-head' }, h('b', {}, title), h('button', { class: 'btn ghost', onclick: close }, 'Close')), ...kids));
    panel.addEventListener('click', (e) => { if (e.target === panel) close(); });
    document.body.append(panel); return { panel, close };
  };
  const procLabel = (p: Plan) => (p.proc && procedures[p.proc] ? `${procedures[p.proc]!.opName} · ${procedures[p.proc]!.approach}` : p.op && OPS[p.op] ? OPS[p.op]!.name : p.note || 'Case');

  // ------------------------------------------------------------------ the daily five
  type Q = { key: string; proc: string; idx: number; step: Step; phase: string };
  let pool: Q[] | null = null;
  const getPool = () => {
    if (pool) return pool; const seen = new Set<string>(); pool = [];
    for (const [k, p] of Object.entries(procedures)) {
      if (p.op === 'cticu') continue;
      p.steps.forEach((s, i) => { if (!s.ask) return; const t = s.ask.question.trim(); if (seen.has(t)) return; seen.add(t); pool!.push({ key: `${k}/${s.id}`, proc: k, idx: i, step: s, phase: s.phase }); });
    }
    return pool;
  };
  const year = () => { const u = D.user(); if (!u || u.role === 'student') return 0; return trainingYear(u.start, iso(new Date())) || 2; };
  /** early years lean on pathophysiology, anatomy and cases; later years on decisions, the operation and the ICU */
  const weight = (q: Q, y: number) => { const early = ['Pathophysiology', 'Anatomy', 'Case'].includes(q.phase); return y <= 2 ? (early ? 3 : 1) : y >= 4 ? (early ? 1 : 3) : 2; };
  function todaySet(): { day: string; keys: string[]; done: Record<string, 0 | 1> } {
    const day = iso(new Date()); const d = (progress.daily ??= {});
    if (d.set?.day === day) return d.set;
    const all = getPool(); const sr = (progress.sr ??= {}); const y = year(); const seed = hash(day + (D.user()?.id ?? 0));
    const due = all.filter((q) => sr[q.key] && sr[q.key]![1] <= day).sort((a, b) => sr[a.key]![0] - sr[b.key]![0]).slice(0, 3);
    const fresh = shuffle(all.filter((q) => !sr[q.key]).flatMap((q) => Array(weight(q, y)).fill(q) as Q[]), seed);
    const keys = [...due.map((q) => q.key)];
    for (const q of fresh) { if (keys.length >= 5) break; if (!keys.includes(q.key)) keys.push(q.key); }
    for (const q of shuffle(all, seed + 1)) { if (keys.length >= 5) break; if (!keys.includes(q.key)) keys.push(q.key); }
    d.set = { day, keys, done: {} }; D.save(); return d.set;
  }
  function record(q: Q, right: boolean): void {
    const day = iso(new Date()); const sr = (progress.sr ??= {}); const box = right ? Math.min(5, (sr[q.key]?.[0] ?? 0) + 1) : 1;
    sr[q.key] = [box, addDays(day, BOX_DAYS[box]!)];
    const set = todaySet(); set.done[q.key] = right ? 1 : 0;
    const pq = (progress.q[q.proc] ??= {}); if (!(q.step.id in pq)) pq[q.step.id] = right ? 1 : 0;
    const d = progress.daily!;
    if (set.keys.every((k) => k in set.done)) {
      const score = set.keys.filter((k) => set.done[k] === 1).length;
      (d.days ??= {})[day] = score;
      if (d.last !== day) { d.streak = d.last === addDays(day, -1) ? (d.streak ?? 0) + 1 : 1; d.last = day; d.best = Math.max(d.best ?? 0, d.streak); }
    }
    D.save(); paintDot();
  }
  function paintDot(): void { const s = progress.daily?.set; const day = iso(new Date()); dot.textContent = s?.day === day && s.keys.every((k) => k in s.done) ? '✓' : '5'; }

  function quiz(): void {
    const set = todaySet(); const all = getPool(); const qs = set.keys.map((k) => all.find((q) => q.key === k)).filter(Boolean) as Q[];
    let i = qs.findIndex((q) => !(q.key in set.done)); if (i < 0) i = qs.length;
    const body = h('div', { class: 'dq' });
    const show = () => {
      if (i >= qs.length) {
        const score = qs.filter((q) => set.done[q.key] === 1).length; const d = progress.daily ?? {};
        body.replaceChildren(h('p', { class: 'dq-score' }, `${score} / ${qs.length} today`), h('p', {}, `Streak: ${d.streak ?? 0} day${(d.streak ?? 0) === 1 ? '' : 's'} · best ${d.best ?? 0}`),
          week(), h('p', { class: 'foot' }, 'Wrong answers come back tomorrow; right ones return after 3, 7, 14 and 30 days. Come back tomorrow for five more.'),
          ...qs.map((q) => h('p', { class: `dq-rev ${set.done[q.key] ? 'ok' : 'no'}` }, set.done[q.key] ? '✓ ' : '✗ ', h('a', { href: `#approach=${q.proc}&step=${q.idx}`, onclick: () => document.getElementById('today-panel')?.remove() }, q.step.ask!.question.length > 110 ? q.step.ask!.question.slice(0, 107).replace(/\s+\S*$/, '') + '…' : q.step.ask!.question)))); return;
      }
      const q = qs[i]!; const p = procedures[q.proc]!; const ch = shuffle(q.step.ask!.choices, hash(q.key + set.day));
      const why = h('div', { class: 'dq-why' }); let answered = false;
      const opts = ch.map((c) => h('button', { class: 'choice', onclick: (ev: Event) => {
        if (answered) return; answered = true; record(q, c.correct);
        (ev.currentTarget as HTMLElement).classList.add(c.correct ? 'right' : 'wrong');
        opts.forEach((o, j) => { if (ch[j]!.correct) o.classList.add('right'); });
        const r = ch.find((x) => x.correct)!;
        why.replaceChildren(h('p', {}, h('b', {}, c.correct ? 'Right. ' : 'Not quite. '), r.why || ''),
          h('p', {}, h('a', { href: `#approach=${q.proc}&step=${q.idx}`, onclick: () => document.getElementById('today-panel')?.remove() }, `Open the step: ${p.opName}, ${q.step.title}`)),
          h('button', { class: 'btn primary', onclick: () => { i++; show(); } }, i + 1 < qs.length ? 'Next question' : 'See today\'s score'));
      } }, c.text));
      body.replaceChildren(h('p', { class: 'dq-n' }, `Question ${i + 1} of ${qs.length} · ${p.opName} · ${q.phase}`), h('p', { class: 'dq-q' }, q.step.ask!.question), h('div', { class: 'dq-opts' }, ...opts), why);
    };
    show(); overlay('Daily five', body);
  }
  /** the last 7 days as small bars */
  function week(): HTMLElement {
    const days = progress.daily?.days ?? {}; const today = iso(new Date());
    return h('div', { class: 'dq-week' }, ...Array.from({ length: 7 }, (_, j) => { const d = addDays(today, j - 6); const sc = days[d];
      return h('span', { title: `${d}: ${sc ?? '–'}/5` }, h('i', { style: `height:${sc == null ? 2 : 4 + sc * 7}px;${sc == null ? 'opacity:.3' : ''}` }), h('small', {}, new Date(d + 'T12:00').toLocaleDateString('en', { weekday: 'narrow' }))); }));
  }

  // ------------------------------------------------------------------ prep pack
  function prep(key: string): void {
    const p = procedures[key]; if (!p) return;
    const steps = p.steps.map((s, i) => ({ s, i }));
    const ops = steps.filter(({ s }) => !TEACH.has(s.phase));
    const danger = [...new Set(p.steps.flatMap((s) => s.danger ?? []))].map(D.nameOf).filter(Boolean).slice(0, 10);
    const go = (i: number) => h('a', { href: `#approach=${key}&step=${i}`, onclick: () => document.getElementById('today-panel')?.remove() });
    const find = (ph: string) => steps.find(({ s }) => s.phase === ph);
    const qs = shuffle(steps.filter(({ s }) => s.ask), hash(key + iso(new Date()))).slice(0, 3);
    const mcq = qs.map(({ s, i }) => {
      const ch = shuffle(s.ask!.choices, hash(s.id)); const why = h('div', { class: 'dq-why' }); let done = false;
      const opts = ch.map((c) => h('button', { class: 'choice', onclick: (ev: Event) => { if (done) return; done = true;
        (ev.currentTarget as HTMLElement).classList.add(c.correct ? 'right' : 'wrong'); opts.forEach((o, j) => { if (ch[j]!.correct) o.classList.add('right'); });
        const pq = (progress.q[key] ??= {}); if (!(s.id in pq)) { pq[s.id] = c.correct ? 1 : 0; D.save(); }
        why.replaceChildren(h('p', {}, ch.find((x) => x.correct)!.why || ''), h('p', {}, (() => { const a = go(i); a.textContent = 'Open the step'; return a; })())); } }, c.text));
      return h('div', { class: 'prep-q' }, h('p', { class: 'dq-q' }, s.ask!.question), h('div', { class: 'dq-opts' }, ...opts), why);
    });
    const link = (ph: string, t: string) => { const f = find(ph); if (!f) return null; const a = go(f.i); a.textContent = t; return h('li', {}, a); };
    overlay(`Prep: ${p.opName}`,
      h('p', { class: 'pg-sum' }, p.approach, ' · ', (() => { const a = go(0); a.textContent = 'Open the whole module'; return a; })()),
      h('h4', {}, 'Before you scrub'), h('ul', {}, link('Anatomy', 'Anatomy to revise'), link('Decision', 'The decision and its alternatives'), link('Consent', 'Consent points for this patient'), link('ICU', 'ICU plan and what can go wrong')),
      h('h4', {}, `The operation: ${ops.length} steps`), h('ol', { class: 'prep-steps' }, ...ops.map(({ s, i }) => h('li', {}, (() => { const a = go(i); a.textContent = s.title; return a; })()))),
      danger.length ? h('div', {}, h('h4', {}, 'Danger points'), h('p', { class: 'prep-danger' }, danger.join(' · '))) : null,
      mcq.length ? h('h4', {}, 'Three questions') : null, ...mcq);
  }

  // ------------------------------------------------------------------ the Today screen
  async function open(): Promise<void> {
    const today = iso(new Date()); const tomorrow = addDays(today, 1);
    const r = await api<Plan[]>(`api/today?from=${addDays(today, -2)}&to=${addDays(today, 7)}`); const plans = r.ok ? r.data : [];
    const set = todaySet(); const done = set.keys.filter((k) => k in set.done).length; const d = progress.daily ?? {};
    const u = D.user();
    const dayList = (day: string, label: string) => {
      const ps = plans.filter((p) => p.date === day);
      return h('div', { class: 'td-day' }, h('h4', {}, label, h('small', {}, ` ${new Date(day + 'T12:00').toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })}`)),
        ...(ps.length ? ps.map((p) => h('div', { class: `td-case${p.log_id ? ' logged' : ''}` },
          h('span', { class: 'td-name' }, procLabel(p)),
          p.proc ? h('button', { class: 'btn', onclick: () => prep(p.proc) }, 'Prep pack') : null,
          p.log_id ? h('span', { class: 'td-ok' }, 'Logged ✓') : day <= today ? h('button', { class: 'btn primary', onclick: () => D.logCase({ date: day, proc: p.proc, op: p.op,
            onSaved: async (id) => { await api('api/today', 'PATCH', { id: p.id, log_id: id }); void open(); } }) }, 'Log this case') : null,
          h('button', { class: 'link', onclick: async () => { await fetch(`api/today?id=${p.id}`, { method: 'DELETE', headers: { 'x-cova': '1' } }); void open(); } }, 'Remove')))
          : [h('p', { class: 'foot' }, 'Nothing listed.')]));
    };
    // add to the list: an atlas operation (with its prep pack) or a logbook operation
    const when = h('select', {}, h('option', { value: tomorrow }, 'Tomorrow'), h('option', { value: today }, 'Today'), h('option', { value: addDays(today, 2) }, 'In 2 days')) as HTMLSelectElement;
    const groups = new Map<string, [string, Procedure][]>();
    for (const [k, p] of Object.entries(procedures)) { if (p.op === 'cticu' || p.group === 'Access and positioning') continue; const g = p.group ?? 'Other'; (groups.get(g) ?? groups.set(g, []).get(g)!).push([k, p]); }
    const pick = h('select', {}, h('option', { value: '' }, 'Add an operation…'),
      ...[...groups].map(([g, l]) => h('optgroup', { label: `Atlas: ${g}` }, ...l.map(([k, p]) => h('option', { value: `p:${k}` }, `${p.opName} · ${p.approach}`)))),
      ...CATALOG.map((s) => h('optgroup', { label: `Logbook: ${s.name}` }, ...s.ops.map((o) => h('option', { value: `o:${o.id}` }, o.name))))) as HTMLSelectElement;
    const add = h('button', { class: 'btn', onclick: async () => {
      if (!pick.value) return; const [t, v] = [pick.value.slice(0, 1), pick.value.slice(2)];
      const rr = await api('api/today', 'POST', { date: when.value, ...(t === 'p' ? { proc: v } : { op: v }) }); if (rr.ok) void open(); } }, 'Add');
    const name = (u?.name ?? '').split(/\s+/).slice(0, 2).join(' ');
    overlay(`Today${name ? `, ${name}` : ''}`,
      h('div', { class: 'td-quiz' },
        h('div', {}, h('b', {}, 'Daily five'), h('p', { class: 'foot' }, done >= 5 ? `Done today: ${set.keys.filter((k) => set.done[k] === 1).length}/5` : `${done}/5 answered`,
          ` · streak ${d.streak ?? 0} · best ${d.best ?? 0}`)), week(),
        h('button', { class: 'btn primary', onclick: quiz }, done >= 5 ? 'Review' : done ? 'Continue' : 'Start')),
      h('div', { class: 'td-add' }, when, pick, add),
      dayList(today, 'Today'), dayList(tomorrow, 'Tomorrow'),
      ...[...new Set(plans.map((p) => p.date))].filter((x) => x > tomorrow).map((x) => dayList(x, 'Coming up')),
      ...[...new Set(plans.filter((p) => p.date < today && !p.log_id).map((p) => p.date))].map((x) => dayList(x, 'Not logged yet')),
      h('p', { class: 'foot' }, 'List tomorrow\'s cases tonight: open the prep pack for each, then log each case in a minute after theatre. Install COVA on your phone: browser menu, "Add to home screen".'));
  }
  return { button: btn, open };
}
