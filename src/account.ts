// Accounts: sign in with an emailed code, a profile with role and institution, progress kept with the account, the
// operative logbook (residents; no consultant sign-off), trainee progress, monthly reviews and assigned modules
// (consultants) and user approval (administrator).
// The atlas stays open to everyone; signing in adds these features.
import type { Procedure } from './procedure.ts';
import { CATALOG, LEVELS as CAT_LEVELS, OPS, SKILLS, opForAtlas } from './logcat.ts';
import { analysis, type LogEntry } from './logstats.ts';

type H = (tag: string, attrs?: Record<string, unknown>, ...kids: (Node | string | null | undefined)[]) => HTMLElement;
export type Progress = { v: Record<string, Record<string, 1>>; q: Record<string, Record<string, 0 | 1>>; sr?: Record<string, [number, string]>;
  daily?: { streak?: number; best?: number; last?: string; days?: Record<string, number>; set?: { day: string; keys: string[]; done: Record<string, 0 | 1> } };
  exam?: { mocks?: { at: string; n: number; right: number; secs: number; areas: string[]; level: string }[]; vivas?: { at: string; proc: string; score: number; max: number; mode: 'self' | 'examiner' }[] } };
export interface Prefill { date: string; proc?: string; op?: string; onSaved?: (id: number) => void }
export interface User { id: number; email: string; name: string | null; role: 'student' | 'resident' | 'consultant' | null; status: 'new' | 'active' | 'pending' | 'suspended';
  institution: string | null; hospital: string | null; year: string | null; reg_no: string | null; start?: string | null; admin: boolean }
interface Dir { institutions: string[]; hospitals: string[]; consultants: { id: number; name: string; hospital: string }[] }

const LEVELS: [string, string][] = CAT_LEVELS.map((l) => [l.code, `${l.name}: ${l.def}`]);
const AREAS = ['Adult cardiac', 'Congenital cardiac', 'General thoracic', 'Esophageal', 'Airway', 'Vascular (open)', 'Endovascular', 'Trauma', 'Access / other'];
const CARDIAC = new Set(['Adult cardiac', 'Congenital cardiac']);
const ROLE_NAME = { student: 'Student', resident: 'Resident', consultant: 'Consultant' } as const;

export async function api<T = unknown>(path: string, method = 'GET', body?: unknown): Promise<{ ok: boolean; status: number; data: T & { error?: string } }> {
  try {
    const r = await fetch(path, { method, credentials: 'same-origin', headers: body !== undefined ? { 'content-type': 'application/json', 'x-cova': '1' } : { 'x-cova': '1' },
      body: body !== undefined ? JSON.stringify(body) : undefined });
    const data = await r.json().catch(() => ({})) as T & { error?: string };
    return { ok: r.ok, status: r.status, data };
  } catch { return { ok: false, status: 0, data: { error: 'No connection. Try again.' } as T & { error?: string } }; }
}

export function createAccount(h: H, procedures: Record<string, Procedure>, progress: Progress, saveLocal: () => void) {
  let user: User | null = null; let enabled = true; let dir: Dir | null = null; const onUser: ((u: User | null) => void)[] = [];
  const btn = h('button', { class: 'btn acct', onclick: () => (user && user.role ? menu() : signIn()), title: 'Sign in to keep your progress and use the logbook' }, 'Sign in') as HTMLButtonElement;

  const paint = () => {
    if (!enabled) { btn.style.display = 'none'; return; }
    btn.style.display = '';
    btn.textContent = user ? (user.name ? (user.name.split(/\s+/).find((w) => !/^(dr|prof|mr|mrs|ms|miss)\.?$/i.test(w)) ?? user.name) : user.email) : 'Sign in';
    if (user?.role) btn.append(h('span', { class: `role-chip r-${user.role}` }, user.admin ? 'Admin' : ROLE_NAME[user.role]));
    for (const f of onUser) f(user);
  };
  const overlay = (title: string, ...kids: (Node | null)[]) => {
    document.getElementById('acct-panel')?.remove();
    const close = () => panel.remove();
    const panel = h('div', { id: 'acct-panel', class: 'overlay', role: 'dialog', 'aria-label': title },
      h('div', { class: 'overlay-box acct-box' }, h('div', { class: 'ov-head' }, h('b', {}, title), h('button', { class: 'btn ghost', onclick: close }, 'Close')), ...kids));
    panel.addEventListener('click', (e) => { if (e.target === panel) close(); });
    document.body.append(panel); return { panel, close };
  };
  const field = (label: string, input: HTMLElement, hint?: string) => h('label', { class: 'af' }, h('span', {}, label), input, hint ? h('small', {}, hint) : null);
  const select = (opts: string[], value = '', blank = 'Choose…') => { const s = h('select', {}, h('option', { value: '' }, blank), ...opts.map((o) => h('option', { value: o, ...(o === value ? { selected: true } : {}) }, o))) as HTMLSelectElement; return s; };
  const msg = () => h('p', { class: 'acct-msg', role: 'status' });
  const getDir = async () => { const r = await api<Dir>('api/directory'); dir = r.ok ? r.data : { institutions: [], hospitals: [], consultants: [] }; return dir; };

  // ------------------------------------------------------------------ progress kept with the account
  let pushT = 0;
  const push = () => { if (!user) return; clearTimeout(pushT); pushT = window.setTimeout(() => { void api('api/progress', 'PUT', progress); }, 2500); };
  async function pull() {
    const r = await api<{ data: Progress | null }>('api/progress'); if (!r.ok) return;
    const s = r.data.data; if (s) {
      for (const [p, m] of Object.entries(s.v ?? {})) Object.assign((progress.v[p] ??= {}), m);
      for (const [p, m] of Object.entries(s.q ?? {})) { const q = (progress.q[p] ??= {}); for (const [k, v] of Object.entries(m)) if (!(k in q)) q[k] = v; }
      // spaced-repetition boxes: keep the later review of each question; daily record: union of days, best streak
      const sr = (progress.sr ??= {}); for (const [k, v] of Object.entries(s.sr ?? {})) if (!sr[k] || sr[k]![1] < v[1]) sr[k] = v;
      if (s.daily) { const d = (progress.daily ??= {}); d.days = { ...(s.daily.days ?? {}), ...(d.days ?? {}) };
        if ((s.daily.last ?? '') > (d.last ?? '')) { d.last = s.daily.last; d.streak = s.daily.streak; if (!d.set || (s.daily.set?.day ?? '') > d.set.day) d.set = s.daily.set; }
        d.best = Math.max(d.best ?? 0, s.daily.best ?? 0); }
      // exam history: union of papers and vivas by time
      if (s.exam) { const e = (progress.exam ??= {}); for (const k of ['mocks', 'vivas'] as const) { const have = new Set((e[k] ?? []).map((x) => x.at));
        (e[k] as { at: string }[] | undefined) = [...(e[k] ?? []), ...((s.exam[k] ?? []) as { at: string }[]).filter((x) => !have.has(x.at))].sort((a, b) => a.at.localeCompare(b.at)).slice(-60) as never; } }
      saveLocal();
    }
    void api('api/progress', 'PUT', progress);
  }

  // ------------------------------------------------------------------ sign in: email, code, profile
  function signIn(): void {
    const email = h('input', { type: 'email', autocomplete: 'email', placeholder: 'you@example.com', required: true }) as HTMLInputElement;
    const m = msg(); const go = h('button', { class: 'btn primary', type: 'submit' }, 'Email me a code') as HTMLButtonElement;
    const form = h('form', { class: 'acct-form' },
      h('p', {}, 'The atlas is open to everyone. Sign in to keep your progress on any device, keep a logbook (residents) and follow your trainees (consultants).'),
      field('Email', email), go, m,
      h('p', { class: 'foot' }, 'No password: we email a 6-digit code each time. Use the email you check most.'));
    const { panel } = overlay('Sign in', form); email.focus();
    form.addEventListener('submit', async (e) => {
      e.preventDefault(); go.disabled = true; m.textContent = 'Sending…';
      const r = await api<{ dev_code?: string }>('api/auth/start', 'POST', { email: email.value });
      go.disabled = false;
      if (!r.ok) { m.textContent = r.data.error ?? 'Could not send the code.'; return; }
      codeStep(panel, email.value.trim(), r.data.dev_code);
    });
  }

  function codeStep(panel: HTMLElement, email: string, dev?: string): void {
    const box = panel.querySelector('.acct-box')!; box.querySelectorAll('.acct-form').forEach((x) => x.remove());
    const code = h('input', { inputmode: 'numeric', autocomplete: 'one-time-code', maxlength: 6, placeholder: '6-digit code', class: 'code-in' }) as HTMLInputElement;
    const m = msg(); const go = h('button', { class: 'btn primary', type: 'submit' }, 'Sign in') as HTMLButtonElement;
    const again = h('button', { class: 'link', type: 'button', disabled: true }, 'Send a new code') as HTMLButtonElement;
    setTimeout(() => { again.disabled = false; }, 60000);
    again.onclick = async () => { again.disabled = true; const r = await api('api/auth/start', 'POST', { email }); m.textContent = r.ok ? 'A new code is on its way.' : (r.data.error ?? ''); setTimeout(() => { again.disabled = false; }, 60000); };
    const form = h('form', { class: 'acct-form' }, h('p', {}, `We sent a code to `, h('b', {}, email), '. It expires in 10 minutes.'),
      dev ? h('p', { class: 'dev' }, `Test mode: the code is ${dev}`) : null, field('Code', code), go, m, h('p', { class: 'foot' }, 'Not arrived? Check spam, or ', again, '.'));
    box.append(form); code.focus();
    form.addEventListener('submit', async (e) => {
      e.preventDefault(); go.disabled = true; m.textContent = 'Checking…';
      const r = await api<{ user: User }>('api/auth/verify', 'POST', { email, code: code.value });
      go.disabled = false;
      if (!r.ok) { m.textContent = r.data.error ?? 'Could not sign in.'; return; }
      user = r.data.user; paint(); void pull();
      if (!user.role) void profile(panel); else { panel.remove(); welcome(); }
    });
  }

  async function profile(panel?: HTMLElement): Promise<void> {
    const d = dir ?? await getDir();
    const name = h('input', { type: 'text', autocomplete: 'name', value: user?.name ?? '', placeholder: 'Dr Jane Wanjiru' }) as HTMLInputElement;
    let role = user?.role ?? '';
    const inst = select(d.institutions, user?.institution ?? ''); const hosp = select(d.hospitals, user?.hospital ?? '');
    const other = h('input', { type: 'text', placeholder: 'Name of your institution or hospital' }) as HTMLInputElement;
    const year = h('input', { type: 'text', value: user?.year ?? '', placeholder: 'e.g. Year 5 MBChB, or Year 3 MMed' }) as HTMLInputElement;
    const reg = h('input', { type: 'text', value: user?.reg_no ?? '', placeholder: 'KMPDC number (optional)' }) as HTMLInputElement;
    const start = h('input', { type: 'month', value: user?.start ?? '' }) as HTMLInputElement;
    const roles = h('div', { class: 'role-cards' }, ...(['student', 'resident', 'consultant'] as const).map((r) => {
      const desc = { student: 'Atlas, quizzes and your progress.', resident: 'All of the above, plus your operative logbook and progress analysis.', consultant: 'Follow your trainees\' progress, give monthly reviews and assign modules. Approved by the administrator.' }[r];
      return h('button', { type: 'button', class: `role-card${role === r ? ' on' : ''}`, 'data-r': r, onclick: () => { role = r; sync(); } }, h('b', {}, ROLE_NAME[r]), h('span', {}, desc));
    }));
    const instF = field('University or training institution', inst); const hospF = field('Hospital', hosp); const otherF = field('Other', other);
    const yearF = field('Year', year); const regF = field('Registration', reg); const startF = field('Training start (month)', start, 'Used to split your logbook by year of training');
    const sync = () => {
      roles.querySelectorAll<HTMLElement>('.role-card').forEach((c) => c.classList.toggle('on', c.dataset['r'] === role));
      instF.style.display = role === 'student' || role === 'resident' ? '' : 'none'; yearF.style.display = instF.style.display;
      hospF.style.display = role === 'consultant' ? '' : 'none'; startF.style.display = role === 'resident' ? '' : 'none'; regF.style.display = role === 'consultant' || role === 'resident' ? '' : 'none';
      otherF.style.display = (role === 'consultant' ? hosp.value : inst.value) === 'Other' ? '' : 'none';
    };
    inst.onchange = sync; hosp.onchange = sync;
    const m = msg(); const go = h('button', { class: 'btn primary', type: 'submit' }, 'Save') as HTMLButtonElement;
    const form = h('form', { class: 'acct-form' }, field('Full name', name), h('div', { class: 'af' }, h('span', {}, 'I am a'), roles), instF, hospF, otherF, yearF, startF, regF, go, m);
    if (panel) { const box = panel.querySelector('.acct-box')!; box.querySelectorAll('.acct-form').forEach((x) => x.remove()); box.append(form); }
    else overlay('Your profile', form);
    sync();
    form.addEventListener('submit', async (e) => {
      e.preventDefault(); go.disabled = true;
      const o = other.value.trim();
      const r = await api<{ user: User }>('api/me', 'POST', { name: name.value, role, institution: inst.value === 'Other' ? o : inst.value, hospital: hosp.value === 'Other' ? o : hosp.value, year: year.value, reg_no: reg.value, start: start.value });
      go.disabled = false;
      if (!r.ok) { m.textContent = r.data.error ?? 'Could not save.'; return; }
      user = r.data.user; paint(); document.getElementById('acct-panel')?.remove(); welcome();
    });
  }

  function welcome(): void {
    if (!user) return;
    const pending = user.role === 'consultant' && user.status === 'pending';
    overlay(`Welcome, ${user.name ?? ''}`,
      h('p', {}, pending ? 'Your consultant account is waiting for the administrator\'s approval. Until then you have the student and resident features.'
        : user.role === 'resident' ? 'Your progress is now kept with your account. Open your logbook from the account menu (your name, top right).'
        : user.role === 'consultant' ? 'Residents who log cases with you as supervisor appear under "Trainees" in the account menu.'
        : 'Your progress is now kept with your account, on any device.'),
      h('button', { class: 'btn primary', onclick: () => document.getElementById('acct-panel')?.remove() }, 'Continue'));
  }

  // ------------------------------------------------------------------ account menu
  function menu(): void {
    if (!user) return signIn();
    const items: (HTMLElement | null)[] = [
      h('p', { class: 'acct-who' }, h('b', {}, user.name ?? ''), ` · ${user.admin ? 'Administrator, ' : ''}${user.role ? ROLE_NAME[user.role] : ''}${user.status === 'pending' ? ' (awaiting approval)' : ''}`,
        h('br'), h('small', {}, [user.institution, user.hospital, user.email].filter(Boolean).join(' · '))),
      h('button', { class: 'btn', onclick: () => { void profile(); } }, 'Edit profile'),
      user.role === 'resident' || user.role === 'consultant' || user.admin ? h('button', { class: 'btn', onclick: () => { void logbook(); } }, 'Logbook') : null,
      (user.role === 'consultant' && user.status === 'active') || user.admin ? h('button', { class: 'btn', onclick: () => { void trainees(); } }, 'Trainees: progress and reviews') : null,
      (user.role === 'consultant' && user.status === 'active') || user.admin ? h('button', { class: 'btn', onclick: () => { void assignPanel(); } }, 'Assign modules') : null,
      user.admin ? h('button', { class: 'btn', onclick: () => { void admin(); } }, 'Users') : null,
      h('button', { class: 'btn ghost', onclick: async () => { await api('api/auth/logout', 'POST', {}); user = null; paint(); document.getElementById('acct-panel')?.remove(); } }, 'Sign out'),
    ];
    overlay('Account', h('div', { class: 'acct-menu' }, ...items));
  }

  // ------------------------------------------------------------------ logbook
  type Entry = { id: number; status: string; review_comment?: string; reviewer?: string; data: Record<string, string | number | null> };
  const procList = () => Object.entries(procedures).filter(([, p]) => p.op !== 'cticu' && p.group !== 'Access and positioning')
    .map(([k, p]) => [k, `${p.opName} · ${p.approach}`] as [string, string]).sort((a, b) => a[1].localeCompare(b[1]));
  const procName = (d: Entry['data']) => (d['proc'] && procedures[String(d['proc'])] ? `${procedures[String(d['proc'])]!.opName} · ${procedures[String(d['proc'])]!.approach}`
    : String(d['proc_name'] || ((d['ops'] as unknown as { op: string }[] | undefined) ?? []).map((x) => OPS[x.op]?.name ?? x.op).join(' + ')));
  const areaOf = (k: string) => { const g = procedures[k]?.group ?? ''; return /Cardiac$/.test(g) && !/Congenital/.test(g) ? 'Adult cardiac' : /Congenital cardiac/.test(g) ? 'Congenital cardiac'
    : g === 'Vascular' ? 'Vascular (open)' : g === 'Esophagus' ? 'Esophageal' : g === 'Airway' ? 'Airway' : g === 'Trauma' ? 'Trauma' : g ? 'General thoracic' : ''; };

  // cases saved while offline: kept on the device and sent when the connection returns
  const QKEY = 'cova-queue';
  const queue = (o: Record<string, unknown>, _tag: string) => { try { const a = JSON.parse(localStorage.getItem(QKEY) ?? '[]'); a.push(o); localStorage.setItem(QKEY, JSON.stringify(a)); } catch { /* storage off */ } };
  async function flush(): Promise<void> {
    let a: Record<string, unknown>[] = []; try { a = JSON.parse(localStorage.getItem(QKEY) ?? '[]'); } catch { return; }
    if (!a.length || !user) return;
    const rest: Record<string, unknown>[] = [];
    for (const o of a) { const r = await api('api/logbook', 'POST', o); if (!r.ok && r.status === 0) rest.push(o); }
    try { localStorage.setItem(QKEY, JSON.stringify(rest)); } catch { /* storage off */ }
  }
  window.addEventListener('online', () => { void flush(); });

  async function logbook(): Promise<void> {
    const [r, rv] = await Promise.all([api<Entry[]>('api/logbook'), user?.role === 'resident' ? api<Rev[]>('api/reviews') : Promise.resolve(null), dir ? null : getDir()]);
    const sname = (id: unknown) => dir?.consultants.find((c) => c.id === id)?.name ?? '';
    const list = r.ok ? r.data : [];
    const counts = new Map<string, Record<string, number>>();
    for (const e of list) { const k = procName(e.data); const c = counts.get(k) ?? {}; c[String(e.data['level'])] = (c[String(e.data['level'])] ?? 0) + 1; counts.set(k, c); }
    const summary = h('details', { class: 'pg-group' }, h('summary', {}, `Consolidated summary: ${list.length} cases`),
      h('table', { class: 'lb-sum' }, h('tr', {}, h('th', {}, 'Procedure'), ...LEVELS.map(([c]) => h('th', {}, c)), h('th', {}, 'Total')),
        ...[...counts].sort((a, b) => a[0].localeCompare(b[0])).map(([k, c]) => h('tr', {}, h('td', {}, k), ...LEVELS.map(([l]) => h('td', {}, String(c[l] ?? ''))),
          h('td', {}, String(Object.values(c).reduce((a, b) => a + b, 0)))))));
    const rows = list.map((e) => h('div', { class: `lb-row st-${e.status}` },
      h('span', { class: 'lb-date' }, String(e.data['date'] ?? '')), h('span', {}, procName(e.data)), h('span', { class: 'lb-lev' }, String(e.data['level'] ?? '')),
      h('span', { class: 'lb-st' }, e.data['supervisor_name'] ? String(e.data['supervisor_name']) : e.data['supervisor_id'] ? sname(e.data['supervisor_id']) : ''),
      h('button', { class: 'link', onclick: () => { void entryForm(e); } }, 'Edit')));
    overlay('Logbook',
      h('div', { class: 'lb-actions' }, h('button', { class: 'btn primary', onclick: () => { void entryForm(); } }, 'Add a case'),
        h('button', { class: 'btn', onclick: () => overlay('Progress analysis', analysis(h, list as unknown as LogEntry[], user?.start)) }, 'Progress analysis'),
        h('button', { class: 'btn', onclick: () => exportCsv(list) }, 'Export (CSV for Excel)'),
        h('a', { class: 'btn', href: 'media/TCVS-resident-logbook.xlsx', download: '' }, 'Paper / Excel template')),
      rv && rv.ok && rv.data.length ? h('details', { class: 'pg-group', open: true }, h('summary', {}, `Supervisor reviews: latest ${rv.data[0]!.rating}`), reviewList(rv.data)) : null,
      summary, ...(rows.length ? rows : [h('p', { class: 'pg-sum' }, 'No cases yet. Add your first case: it takes a minute after theatre.')]),
      h('p', { class: 'foot' }, 'Codes follow the intercollegiate eLogbook (ISCP): O observed, A assisted, S-TS supervised with trainer scrubbed, S-TU supervised with trainer unscrubbed, P performed, T training a junior. Record no patient names or hospital numbers.'));
  }

  async function entryForm(e?: Entry, pre?: Prefill): Promise<void> {
    const d = dir && dir.consultants.length ? dir : await getDir();
    let last: Record<string, string> = {}; try { last = JSON.parse(localStorage.getItem('cova-last') ?? '{}'); } catch { /* none */ }
    const quick = !!pre && !e;
    const v: Entry['data'] = e?.data ?? (pre ? { date: pre.date, proc: pre.proc ?? '', hospital: last['hospital'] ?? user?.hospital ?? '', supervisor_id: last['supervisor_id'] ?? '',
      area: pre.proc ? areaOf(pre.proc) : '', ops: (pre.op || (pre.proc && opForAtlas(pre.proc)) ? [{ op: pre.op || opForAtlas(pre.proc!), comps: {} }] : []) as unknown as string } : {});
    const sv = (k: string) => (v[k] == null ? '' : String(v[k]));
    const inp = (k: string, type = 'text', attrs: Record<string, unknown> = {}) => h('input', { type, value: sv(k), name: k, ...attrs }) as HTMLInputElement;
    const sel = (k: string, opts: string[], blank = 'Choose…') => { const s = select(opts, sv(k), blank); s.name = k; return s; };
    const date = inp('date', 'date', { max: new Date().toISOString().slice(0, 10) }); if (!e && !pre) date.value = new Date().toISOString().slice(0, 10);
    const proc = h('select', { name: 'proc' }, h('option', { value: '' }, 'Choose from the atlas…'), ...procList().map(([k, t]) => h('option', { value: k, ...(k === sv('proc') ? { selected: true } : {}) }, t)),
      h('option', { value: '__other' }, 'Other (type below)')) as HTMLSelectElement;
    const pother = inp('proc_name', 'text', { placeholder: 'Procedure name, if not in the atlas' });
    const area = sel('area', AREAS);
    proc.onchange = () => { if (proc.value && proc.value !== '__other') { if (!area.value) area.value = areaOf(proc.value); ensureOp(proc.value); } perf(); };
    const lev = h('div', { class: 'lev-cards' }, ...LEVELS.map(([c, t]) => h('label', { class: 'lev' }, h('input', { type: 'radio', name: 'level', value: c, ...(sv('level') === c ? { checked: true } : {}) }), h('b', {}, c), h('span', {}, t))));
    const sup = h('select', { name: 'supervisor_id' }, h('option', { value: '' }, 'Choose the consultant…'), ...d.consultants.filter((c) => c.id !== user?.id)
      .map((c) => h('option', { value: String(c.id), ...(String(c.id) === sv('supervisor_id') ? { selected: true } : {}) }, `${c.name} · ${c.hospital}`))) as HTMLSelectElement;
    // the TCVS catalogue: one block per operation (combined cases, e.g. AVR with CABG, get two), a level for each component skill
    const opsBox = h('div', { class: 'ops-box' });
    const lvlSel = (val = '') => h('select', { class: 'lv-sel' }, h('option', { value: '' }, '–'), ...CAT_LEVELS.map((l) => h('option', { value: l.code, ...(l.code === val ? { selected: true } : {}), title: l.def }, `${l.code} · ${l.name}`))) as HTMLSelectElement;
    const opBlock = (init?: { op: string; comps: Record<string, string> }) => {
      const sel = h('select', { class: 'op-sel' }, h('option', { value: '' }, 'Choose the operation…'),
        ...CATALOG.map((sec) => h('optgroup', { label: sec.name }, ...sec.ops.map((o) => h('option', { value: o.id, ...(o.id === init?.op ? { selected: true } : {}) }, `${o.name}${o.core ? ' *' : ''}`))))) as HTMLSelectElement;
      const comps = h('div', { class: 'comp-list' });
      const fill = (keep?: Record<string, string>) => {
        const op = OPS[sel.value]; comps.replaceChildren();
        if (!op) return;
        if (!op.skills.length) { comps.append(h('p', { class: 'foot' }, 'No separate components: your overall level (below) is used.')); return; }
        comps.append(h('div', { class: 'comp-row head' }, h('span', {}, 'Component skill'), h('span', {}, 'Your level'), h('button', { type: 'button', class: 'link', onclick: () => {
          const v = (comps.querySelector('.lv-sel') as HTMLSelectElement | null)?.value ?? ''; comps.querySelectorAll<HTMLSelectElement>('.lv-sel').forEach((x) => { x.value = v; }); } }, 'same for all')));
        for (const k of op.skills) { const r = h('div', { class: 'comp-row', 'data-k': k }, h('span', {}, SKILLS[k] ?? k), lvlSel(keep?.[k] ?? '')); comps.append(r); }
      };
      sel.onchange = () => fill();
      const blk = h('div', { class: 'op-block' }, h('div', { class: 'op-head' }, sel, h('button', { type: 'button', class: 'link', onclick: () => blk.remove() }, 'Remove')), comps);
      opsBox.append(blk); fill(init?.comps); return blk;
    };
    const savedOps = (v['ops'] as unknown as { op: string; comps: Record<string, string> }[] | undefined) ?? [];
    savedOps.forEach((o) => opBlock(o));
    const ensureOp = (k: string) => { const id = opForAtlas(k); if (id && !opsBox.querySelector('.op-sel')) { const b = opBlock(); const s_ = b.querySelector('.op-sel') as HTMLSelectElement; s_.value = id; s_.dispatchEvent(new Event('change')); } };
    const opsFs = h('fieldset', { class: 'lb-fs' }, h('legend', {}, 'TCVS logbook: component skills'),
      h('p', { class: 'foot' }, 'Record your level for each step. This is what the progress analysis follows, case by case.'), opsBox,
      h('button', { type: 'button', class: 'btn', onclick: () => opBlock() }, 'Add an operation (combined case)'));
    const perfBox = h('fieldset', { class: 'lb-fs' }, h('legend', {}, 'Bypass (cardiac)'),
      h('div', { class: 'af-row' }, field('CPB (min)', inp('cpb_min', 'number', { min: 0 })), field('Cross-clamp (min)', inp('xclamp_min', 'number', { min: 0 })), field('Circulatory arrest (min)', inp('arrest_min', 'number', { min: 0 }))));
    const perf = () => { perfBox.style.display = CARDIAC.has(area.value) ? '' : 'none'; };
    area.onchange = perf;
    const m = msg(); const go = h('button', { class: 'btn primary', type: 'submit' }, e ? 'Save changes' : 'Add case') as HTMLButtonElement;
    const caseFs = h('fieldset', { class: 'lb-fs' }, h('legend', {}, 'Case'),
        h('div', { class: 'af-row' }, quick ? null : field('Date', date), field('Hospital', sel('hospital', d.hospitals)), field('Case code', inp('case_code', 'text', { placeholder: 'Your own code, e.g. 2026-041' }), 'Never a name or hospital number')),
        h('div', { class: 'af-row' }, field('Age', inp('age', 'number', { min: 0 })), field('Unit', sel('age_unit', ['years', 'months', 'days'], 'years')), field('Sex', sel('sex', ['F', 'M'], '–')), field('Urgency', sel('urgency', ['Elective', 'Urgent', 'Emergency', 'Salvage']))));
    const opFs = h('fieldset', { class: 'lb-fs' }, h('legend', {}, 'Operation'),
        field('Diagnosis / indication', inp('diagnosis', 'text', { placeholder: 'e.g. Severe rheumatic MS, Wilkins 10' })),
        quick ? null : field('Procedure', proc), quick ? null : pother, h('div', { class: 'af-row' }, field('Area', area), field('Approach', inp('approach', 'text', { placeholder: 'e.g. Median sternotomy, VATS 3-port' }))));
    const supFs = h('fieldset', { class: 'lb-fs' }, h('legend', {}, quick ? 'Supervisor' : 'Overall level and supervisor'), quick ? null : lev, field('Supervising consultant', sup, 'Recorded with the case; they follow your progress and give the monthly review. Not listed? Ask them to create a consultant account.'), inp('supervisor_name', 'text', { placeholder: 'Or type the name' }));
    const outFs = h('fieldset', { class: 'lb-fs' }, h('legend', {}, 'Outcome'),
        h('div', { class: 'af-row' }, field('Complication', sel('complication', ['None', 'Minor', 'Major'])), field('Clavien-Dindo', sel('clavien', ['I', 'II', 'IIIa', 'IIIb', 'IVa', 'IVb', 'V'], '–')),
          field('Back to theatre', sel('return_theatre', ['No', 'Yes'], '–')), field('30-day', sel('outcome_30d', ['Alive', 'Died', 'Unknown'], '–')), field('Stay (days)', inp('los_days', 'number', { min: 0 })), field('Blood loss (ml)', inp('blood_ml', 'number', { min: 0 }))));
    const refFs = h('fieldset', { class: 'lb-fs' }, h('legend', {}, 'Reflection'), h('textarea', { name: 'notes', rows: 3, placeholder: 'One learning point, or what you would do differently' }, sv('notes')));
    const actions = h('div', { class: 'lb-actions' }, go, e ? h('button', { class: 'btn ghost', type: 'button', onclick: async () => { if (!confirm('Delete this case?')) return; await fetch(`api/logbook?id=${e.id}`, { method: 'DELETE', headers: { 'x-cova': '1' } }); void logbook(); } }, 'Delete') : null);
    const form = quick
      ? h('form', { class: 'acct-form lb-form' }, h('p', { class: 'quick-head' }, h('b', {}, procName({ proc: pre!.proc ?? '', ops: v['ops'] } as Entry['data']) || 'Case'), ` · ${pre!.date}`), date,
          opsFs, supFs, perfBox, h('details', { class: 'more' }, h('summary', {}, 'More details (optional): case, outcome, reflection'), caseFs, opFs, outFs, refFs), actions, m,
          h('input', { type: 'hidden', name: 'proc', value: pre!.proc ?? '' }))
      : h('form', { class: 'acct-form lb-form' }, caseFs,
          h('fieldset', { class: 'lb-fs' }, h('legend', {}, 'Procedure'), field('Procedure', proc), pother), opFs, opsFs, supFs, perfBox, outFs, refFs, actions, m);
    if (quick) { date.type = 'hidden'; date.value = pre!.date; }
    overlay(e ? 'Edit case' : quick ? 'Log this case' : 'Add a case', form); perf();
    if (sv('proc_name') && !sv('proc')) proc.value = '__other';
    form.addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const fd = new FormData(form as HTMLFormElement); const o: Record<string, unknown> = {};
      fd.forEach((val, k) => { o[k] = String(val); });
      for (const k of ['age', 'cpb_min', 'xclamp_min', 'arrest_min', 'blood_ml', 'los_days']) o[k] = o[k] === '' ? null : Number(o[k]);
      o['supervisor_id'] = o['supervisor_id'] ? Number(o['supervisor_id']) : null;
      if (o['proc'] === '__other') o['proc'] = '';
      if (o['proc']) o['proc_name'] = procName({ proc: String(o['proc']) });
      o['ops'] = [...opsBox.querySelectorAll('.op-block')].map((b) => ({ op: (b.querySelector('.op-sel') as HTMLSelectElement).value,
        comps: Object.fromEntries([...b.querySelectorAll<HTMLElement>('.comp-row[data-k]')].map((r) => [r.dataset['k']!, (r.querySelector('.lv-sel') as HTMLSelectElement).value]).filter(([, lv]) => lv)) })).filter((x) => x.op);
      if (!o['level']) delete o['level'];
      const ops_ = o['ops'] as { op: string }[];
      if (!o['proc'] && !o['proc_name'] && ops_.length) o['proc_name'] = ops_.map((x) => OPS[x.op]?.name ?? x.op).join(' + ');
      if (e) o['id'] = e.id;
      try { localStorage.setItem('cova-last', JSON.stringify({ hospital: o['hospital'] ?? '', supervisor_id: o['supervisor_id'] ?? '' })); } catch { /* storage off */ }
      go.disabled = true; const r = await api<{ id: number }>('api/logbook', e ? 'PUT' : 'POST', o); go.disabled = false;
      if (r.status === 0 && !e) { queue(o, pre?.onSaved ? 'plan' : ''); m.textContent = 'No connection: saved on this device, it will be sent when you are back online.'; return; }
      if (!r.ok) { m.textContent = r.data.error ?? 'Could not save.'; return; }
      if (pre?.onSaved) { pre.onSaved(r.data.id); document.getElementById('acct-panel')?.remove(); } else void logbook();
    });
  }

  function exportCsv(list: Entry[]): void {
    const cols = ['date', 'hospital', 'case_code', 'age', 'age_unit', 'sex', 'urgency', 'area', 'diagnosis', 'proc_name', 'approach', 'level', 'supervisor_name', 'cpb_min', 'xclamp_min',
      'arrest_min', 'blood_ml', 'complication', 'clavien', 'return_theatre', 'outcome_30d', 'los_days', 'notes'];
    const q = (x: unknown) => `"${String(x ?? '').replace(/"/g, '""')}"`;
    const sname = (id: unknown) => dir?.consultants.find((c) => c.id === id)?.name ?? '';
    const lines = [cols.join(',')].concat(list.map((e) => [...cols.map((c) => q(c === 'proc_name' ? procName(e.data) : c === 'supervisor_name' ? (e.data[c] || sname(e.data['supervisor_id'])) : e.data[c]))].join(',')));
    const a = h('a', { href: URL.createObjectURL(new Blob(['﻿' + lines.join('\r\n')], { type: 'text/csv' })), download: `cova-logbook-${new Date().toISOString().slice(0, 10)}.csv` }) as HTMLAnchorElement;
    document.body.append(a); a.click(); a.remove();
  }

  // ------------------------------------------------------------------ consultant: trainees' progress, monthly review, assigned modules
  type Rev = { id: number; month: string; rating: string; comment: string; plan: string; consultant: string; consultant_id: number; created: string };
  type Asg = { id: number; proc: string; name: string; note: string; due: string | null; done: string | null; score: string | null; resident?: string; by_name?: string; created: string };
  const RATINGS = ['On track', 'Needs support', 'Concern'];
  const monthLabel = (m: string) => new Date(m + '-15T12:00').toLocaleDateString('en-GB', { month: 'long', year: 'numeric' });
  const asgName = (a: Asg) => a.name || (procedures[a.proc] ? `${procedures[a.proc]!.opName} · ${procedures[a.proc]!.approach}` : a.proc);
  const reviewList = (rs: Rev[]) => rs.length ? h('div', { class: 'rv-list' }, ...rs.map((r) => h('div', { class: `rv-item rt-${r.rating.replace(/\s+/g, '-').toLowerCase()}` },
    h('b', {}, monthLabel(r.month)), ` · ${r.rating} · ${r.consultant}`, r.comment ? h('p', {}, r.comment) : null, r.plan ? h('p', {}, h('i', {}, 'Plan: '), r.plan) : null)))
    : h('p', { class: 'foot' }, 'No monthly reviews yet.');
  const asgRow = (a: Asg, extra?: HTMLElement | null) => h('div', { class: `lb-row as-row${a.done ? ' st-verified' : a.due && a.due < new Date().toISOString().slice(0, 10) ? ' st-returned' : ''}` },
    h('span', {}, h('a', { href: `#approach=${a.proc}&step=0`, onclick: () => document.getElementById('acct-panel')?.remove() }, asgName(a)), a.note ? h('small', {}, ` · ${a.note}`) : null),
    h('span', { class: 'lb-st' }, a.done ? `Done ${a.done.slice(0, 10)}${a.score ? `, questions ${a.score}` : ''}` : a.due ? `Due ${a.due}` : 'Open'), extra ?? null);

  async function trainee(t: { id: number; name: string }): Promise<void> {
    const [q, rv, as] = await Promise.all([api<{ resident: { start: string }; entries: LogEntry[] }>(`api/logbook?resident=${t.id}`), api<Rev[]>(`api/reviews?resident=${t.id}`), api<Asg[]>(`api/assign?by=me&resident=${t.id}`)]);
    const now = new Date(); const months = Array.from({ length: 4 }, (_, i) => new Date(now.getFullYear(), now.getMonth() - i, 15).toISOString().slice(0, 7));
    const mine = (rv.ok ? rv.data : []).filter((r) => r.consultant_id === user?.id);
    const month = select(months.map(monthLabel), monthLabel(now.getDate() <= 7 ? months[1]! : months[0]!), 'Month…');
    const rating = h('div', { class: 'rt-pick' }, ...RATINGS.map((r) => h('label', {}, h('input', { type: 'radio', name: 'rt', value: r }), ` ${r}`)));
    const comment = h('textarea', { rows: 3, placeholder: 'What went well, what to work on (seen by the resident)' }) as HTMLTextAreaElement;
    const plan = h('input', { type: 'text', placeholder: 'Plan for next month, e.g. lead 2 lobectomies, revise the ICU module' }) as HTMLInputElement;
    const fill = () => { const m = months[months.map(monthLabel).indexOf(month.value)]; const r = mine.find((x) => x.month === m);
      comment.value = r?.comment ?? ''; plan.value = r?.plan ?? ''; rating.querySelectorAll('input').forEach((i) => { (i as HTMLInputElement).checked = (i as HTMLInputElement).value === r?.rating; }); };
    month.onchange = fill; fill();
    const m1 = msg();
    const save = h('button', { class: 'btn primary', onclick: async () => {
      const rt = (rating.querySelector('input:checked') as HTMLInputElement | null)?.value; const m = months[months.map(monthLabel).indexOf(month.value)];
      if (!m || !rt) { m1.textContent = 'Choose the month and a rating.'; return; }
      const r = await api('api/reviews', 'POST', { resident_id: t.id, month: m, rating: rt, comment: comment.value, plan: plan.value });
      if (r.ok) void trainee(t); else m1.textContent = r.data.error ?? 'Could not save.';
    } }, 'Save review');
    const asgs = as.ok ? as.data : [];
    const del = (a: Asg) => h('button', { class: 'link', onclick: async () => { await fetch(`api/assign?id=${a.id}`, { method: 'DELETE', headers: { 'x-cova': '1' } }); void trainee(t); } }, 'Remove');
    overlay(`${t.name}`,
      h('details', { class: 'pg-group', open: true }, h('summary', {}, 'Monthly review'),
        h('div', { class: 'rv-form' }, field('Month', month), field('Rating', rating), field('Comment', comment), field('Plan', plan), h('div', { class: 'lb-actions' }, save), m1),
        h('h4', {}, 'Earlier reviews'), reviewList(rv.ok ? rv.data : [])),
      h('details', { class: 'pg-group' }, h('summary', {}, `Modules you assigned: ${asgs.filter((a) => a.done).length}/${asgs.length} done`),
        ...asgs.map((a) => asgRow(a, del(a))), assignForm([t.id], () => { void trainee(t); })),
      q.ok ? analysis(h, q.data.entries, q.data.resident.start) : h('p', {}, q.data.error ?? ''));
  }

  async function trainees(): Promise<void> {
    type T = { id: number; name: string; institution: string; year: string; start: string; cases: number };
    const r = await api<T[]>('api/logbook?trainees=1'); const list = r.ok ? r.data : [];
    overlay('Trainees', h('p', { class: 'pg-sum' }, h('a', { href: 'api/digest?preview=1', target: '_blank', rel: 'noopener' }, 'Preview your Monday email')), ...(list.length ? list.map((t) => h('div', { class: 'lb-row tr-row' }, h('span', {}, h('b', {}, t.name ?? '–')), h('span', {}, `${t.institution ?? ''} ${t.year ?? ''}`),
      h('span', { class: 'lb-st' }, `${t.cases} cases`),
      h('button', { class: 'btn', onclick: () => { void trainee(t); } }, 'Open'))) : [h('p', {}, 'No trainees yet: residents appear here once they log a case with you as supervisor, or once you assign them a module.')]));
  }

  /** choose a module and a due date, then assign it to the given residents */
  function assignForm(ids: number[] | (() => number[]), done: () => void): HTMLElement {
    const proc = h('select', {}, h('option', { value: '' }, 'Choose a module…'), ...procList().map(([k, n]) => h('option', { value: k }, n))) as HTMLSelectElement;
    const due = h('input', { type: 'date', value: new Date(Date.now() + 7 * 864e5).toISOString().slice(0, 10) }) as HTMLInputElement;
    const note = h('input', { type: 'text', placeholder: 'Note, e.g. before Thursday\'s list' }) as HTMLInputElement;
    const m = msg();
    return h('div', { class: 'as-form' }, h('h4', {}, 'Assign a module'), field('Module', proc), h('div', { class: 'af-row' }, field('Due', due), field('Note', note)),
      h('div', { class: 'lb-actions' }, h('button', { class: 'btn primary', onclick: async () => {
        const rid = typeof ids === 'function' ? ids() : ids;
        if (!proc.value || !rid.length) { m.textContent = 'Choose a module and at least one resident.'; return; }
        const r = await api<{ n: number }>('api/assign', 'POST', { resident_ids: rid, proc: proc.value, name: proc.selectedOptions[0]?.textContent ?? '', due: due.value, note: note.value });
        if (r.ok) { m.textContent = `Assigned to ${r.data.n}.`; done(); } else m.textContent = r.data.error ?? 'Could not assign.';
      } }, 'Assign')), m);
  }

  async function assignPanel(): Promise<void> {
    type R = { id: number; name: string; institution: string; year: string };
    const [rs, as] = await Promise.all([api<R[]>('api/assign?residents=1'), api<Asg[]>('api/assign?by=me')]);
    const res = rs.ok ? rs.data : []; const picked = new Set<number>();
    const groups = new Map<string, R[]>(); for (const r of res) { const g = r.institution || 'Other'; (groups.get(g) ?? groups.set(g, []).get(g)!).push(r); }
    const pick = h('div', { class: 'as-pick' }, ...[...groups].map(([g, l]) => h('fieldset', {}, h('legend', {}, g, ' ',
      h('button', { class: 'link', onclick: (e: Event) => { (e.currentTarget as HTMLElement).closest('fieldset')!.querySelectorAll('input').forEach((i) => { (i as HTMLInputElement).checked = true; picked.add(Number((i as HTMLInputElement).value)); }); } }, 'all')),
      ...l.map((r) => h('label', {}, h('input', { type: 'checkbox', value: String(r.id), onchange: (e: Event) => { const i = e.currentTarget as HTMLInputElement; if (i.checked) picked.add(r.id); else picked.delete(r.id); } }), ` ${r.name ?? '–'}${r.year ? ` (${r.year})` : ''}`)))));
    const made = as.ok ? as.data : [];
    overlay('Assign modules',
      res.length ? pick : h('p', {}, 'No active residents yet.'), assignForm(() => [...picked], () => { void assignPanel(); }),
      h('h4', {}, `Assigned: ${made.filter((a) => a.done).length}/${made.length} done`),
      ...made.map((a) => asgRow({ ...a, note: [a.resident, a.note].filter(Boolean).join(' · ') })));
  }

  // ------------------------------------------------------------------ administrator: users
  async function admin(): Promise<void> {
    type AU = User & { created: string; last_login: string; cases: number };
    const r = await api<AU[]>('api/admin/users'); const list = r.ok ? r.data : [];
    const set = async (id: number, body: Record<string, unknown>) => { await api('api/admin/users', 'PATCH', { id, ...body }); void admin(); };
    const dm = msg();
    const sendNow = h('button', { class: 'btn', onclick: async () => {
      if (!confirm('Send the weekly email to every consultant with something to report now?')) return;
      const r = await api<{ sent: number; quiet: number; failed: number }>('api/digest', 'POST', {});
      dm.textContent = r.ok ? `Sent ${r.data.sent}, nothing to report for ${r.data.quiet}${r.data.failed ? `, failed ${r.data.failed} (check the email settings)` : ''}.` : r.data.error ?? 'Failed.';
    } }, 'Send weekly emails now');
    overlay('Users', h('p', { class: 'pg-sum' }, `${list.length} accounts · ${list.filter((u) => u.status === 'pending').length} awaiting approval`), h('div', { class: 'lb-actions' }, sendNow), dm,
      h('table', { class: 'lb-sum adm' }, h('tr', {}, ...['Name', 'Role', 'Where', 'Email', 'Cases', 'Status', ''].map((x) => h('th', {}, x))),
        ...list.map((u) => h('tr', { class: `st-${u.status}` }, h('td', {}, u.name ?? '–'), h('td', {}, u.role ? ROLE_NAME[u.role] : '–'), h('td', {}, u.institution || u.hospital || ''),
          h('td', {}, u.email), h('td', {}, String(u.cases)), h('td', {}, u.status),
          h('td', {}, u.status === 'pending' ? h('button', { class: 'btn primary', onclick: () => set(u.id, { status: 'active' }) }, 'Approve') : null,
            u.status !== 'suspended' ? h('button', { class: 'link', onclick: () => { if (confirm(`Suspend ${u.email}?`)) void set(u.id, { status: 'suspended' }); } }, 'Suspend')
              : h('button', { class: 'link', onclick: () => set(u.id, { status: 'active' }) }, 'Restore'))))));
  }

  async function init(): Promise<void> {
    const r = await api<{ user: User | null; enabled: boolean }>('api/me');
    enabled = r.ok ? r.data.enabled !== false : false;
    user = r.ok ? r.data.user : null; paint();
    if (user) { void pull(); void flush(); if (!user.role) void profile(); }
  }
  paint(); void init();
  return { button: btn, progressChanged: push, signedIn: () => !!user, user: () => user, logCase: (pre: Prefill) => { void entryForm(undefined, pre); },
    onUser: (f: (u: User | null) => void) => { onUser.push(f); f(user); } };
}
