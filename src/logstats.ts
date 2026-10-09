// Logbook analysis: how a trainee's level of involvement in each component skill changes over repeated cases
// (observed, assisted, then doing it under supervision, then independently), plus the KNH annual summary (P / 1A / 2A).
import { CATALOG, LEVELS, OPS, RANK, SKILLS } from './logcat.ts';

type H = (tag: string, attrs?: Record<string, unknown>, ...kids: (Node | string | null | undefined)[]) => HTMLElement;
export type LogEntry = { id: number; status: string; data: Record<string, unknown> & { ops?: { op: string; comps: Record<string, string> }[] } };
interface Pt { date: string; rank: number; level: string; op: string; verified: boolean }

const NS = 'http://www.w3.org/2000/svg';
const svg = (tag: string, a: Record<string, string | number>, ...kids: Node[]) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, String(v)); kids.forEach((k) => e.append(k)); return e; };
const COL = ['#6b8592', '#7d8fa6', '#5aa0c8', '#2ec4b6', '#43d17e', '#9be15d', '#f2c46d'];   // O .. T
const SUP = 3;                                                                                   // S-TS: doing the key parts

/** every component skill (and every operation without listed components) as a series of levels over time */
export function series(entries: LogEntry[]): Map<string, { label: string; pts: Pt[] }> {
  const out = new Map<string, { label: string; pts: Pt[] }>();
  const add = (key: string, label: string, p: Pt) => { const s = out.get(key) ?? { label, pts: [] }; s.pts.push(p); out.set(key, s); };
  for (const e of [...entries].sort((a, b) => String(a.data['date']).localeCompare(String(b.data['date'])) || a.id - b.id)) {
    const date = String(e.data['date'] ?? ''); const verified = e.status === 'verified';
    for (const o of e.data.ops ?? []) {
      const op = OPS[o.op]; const name = op?.name ?? o.op; const comps = Object.entries(o.comps ?? {});
      if (!comps.length && e.data['level']) add(`op:${o.op}`, `${name} (whole operation)`, { date, rank: RANK[String(e.data['level'])] ?? 0, level: String(e.data['level']), op: name, verified });
      for (const [k, lv] of comps) add(k, SKILLS[k] ?? k, { date, rank: RANK[lv] ?? 0, level: lv, op: name, verified });
    }
    if (!(e.data.ops ?? []).length && e.data['level']) {
      const nm = String(e.data['proc_name'] || e.data['proc'] || 'Operation'); add(`proc:${nm}`, nm, { date, rank: RANK[String(e.data['level'])] ?? 0, level: String(e.data['level']), op: nm, verified });
    }
  }
  return out;
}

/** a step chart of one skill: one point per case, y = level of involvement */
function chart(pts: Pt[], w = 620, hgt = 230): SVGElement {
  const L = 64, R = 12, T = 12, B = 34; const iw = w - L - R, ih = hgt - T - B; const n = pts.length;
  const x = (i: number) => L + (n === 1 ? iw / 2 : (i * iw) / (n - 1)); const y = (r: number) => T + ih - (r * ih) / 6;
  const g = svg('svg', { viewBox: `0 0 ${w} ${hgt}`, class: 'lg-chart', role: 'img' });
  g.append(svg('rect', { x: L, y: y(6), width: iw, height: y(SUP) - y(6), fill: 'rgba(46,196,182,0.07)' }));   // the "doing it" band
  LEVELS.forEach((lv, i) => {
    g.append(svg('line', { x1: L, x2: L + iw, y1: y(i), y2: y(i), stroke: 'rgba(255,255,255,0.07)' }));
    const t = svg('text', { x: L - 8, y: y(i) + 4, 'text-anchor': 'end', 'font-size': 11, fill: COL[i]! }); t.textContent = lv.code; g.append(t);
  });
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.rank).toFixed(1)}`).join(' ');
  g.append(svg('path', { d, fill: 'none', stroke: '#2ec4b6', 'stroke-width': 2, 'stroke-linejoin': 'round' }));
  // trend: least-squares line through the ranks
  if (n >= 3) {
    const mx = (n - 1) / 2, my = pts.reduce((a, p) => a + p.rank, 0) / n;
    const k = pts.reduce((a, p, i) => a + (i - mx) * (p.rank - my), 0) / pts.reduce((a, _p, i) => a + (i - mx) ** 2, 0);
    g.append(svg('line', { x1: x(0), y1: y(my - k * mx), x2: x(n - 1), y2: y(my + k * (n - 1 - mx)), stroke: '#f2c46d', 'stroke-dasharray': '5 4', 'stroke-width': 1.2 }));
  }
  pts.forEach((p, i) => {
    const c = svg('circle', { cx: x(i), cy: y(p.rank), r: 5, fill: p.verified ? COL[Math.round(p.rank)]! : '#0b1a22', stroke: COL[Math.round(p.rank)]!, 'stroke-width': 2 });
    const tt = svg('title', {}); tt.textContent = `${p.date} · ${p.level} · ${p.op}${p.verified ? ' · verified' : ' · awaiting verification'}`; c.append(tt); g.append(c);
  });
  const lab = (i: number) => { const t = svg('text', { x: x(i), y: hgt - 12, 'text-anchor': n > 1 && i === n - 1 ? 'end' : n > 1 && i === 0 ? 'start' : 'middle', 'font-size': 10, fill: '#8aa0ab' }); t.textContent = pts[i]!.date.slice(2); g.append(t); };
  if (n) { lab(0); if (n > 1) lab(n - 1); if (n > 4) lab(Math.floor((n - 1) / 2)); }
  const cap = svg('text', { x: L + iw, y: y(6) + 12, 'text-anchor': 'end', 'font-size': 10, fill: '#2ec4b6' }); cap.textContent = 'doing the key parts'; g.append(cap);
  return g;
}

function spark(pts: Pt[], w = 120, hgt = 30): SVGElement {
  const n = pts.length; const x = (i: number) => 3 + (n === 1 ? (w - 6) / 2 : (i * (w - 6)) / (n - 1)); const y = (r: number) => hgt - 3 - (r * (hgt - 6)) / 6;
  const g = svg('svg', { viewBox: `0 0 ${w} ${hgt}`, width: w, height: hgt, class: 'lg-spark' });
  g.append(svg('line', { x1: 0, x2: w, y1: y(SUP), y2: y(SUP), stroke: 'rgba(46,196,182,0.35)', 'stroke-dasharray': '2 3' }));
  g.append(svg('path', { d: pts.map((p, i) => `${i ? 'L' : 'M'}${x(i)},${y(p.rank)}`).join(' '), fill: 'none', stroke: '#2ec4b6', 'stroke-width': 1.5 }));
  pts.forEach((p, i) => g.append(svg('circle', { cx: x(i), cy: y(p.rank), r: 2.2, fill: COL[p.rank]! })));
  return g;
}

/** months of training from the start date to a case date, as a training year (1-based) */
export const trainingYear = (start: string | null | undefined, date: string) => {
  if (!start || !/^\d{4}-\d{2}/.test(start) || !date) return 0;
  const [sy, sm] = start.split('-').map(Number) as [number, number]; const [y, m] = date.split('-').map(Number) as [number, number];
  return Math.floor(((y - sy) * 12 + (m - sm)) / 12) + 1;
};

export function analysis(h: H, entries: LogEntry[], start?: string | null): HTMLElement {
  const S = series(entries);
  const list = [...S].map(([k, s]) => {
    const first = s.pts[0]!, last = s.pts[s.pts.length - 1]!;
    const firstDoing = s.pts.findIndex((p) => p.rank >= SUP);
    return { k, ...s, first, last, firstDoing, best: Math.max(...s.pts.map((p) => p.rank)) };
  }).sort((a, b) => b.pts.length - a.pts.length || a.label.localeCompare(b.label));
  if (!list.length) return h('p', { class: 'pg-sum' }, 'No component skills logged yet. Add cases and record your level for each step; this page then shows how each skill progresses.');

  // overall: the average level of everything done each month
  const byMonth = new Map<string, number[]>();
  for (const s of list) for (const p of s.pts) { const m = p.date.slice(0, 7); (byMonth.get(m) ?? byMonth.set(m, []).get(m)!).push(p.rank); }
  const months = [...byMonth].sort((a, b) => a[0].localeCompare(b[0])).map(([m, r]) => ({ date: m, rank: r.reduce((a, b) => a + b, 0) / r.length, level: '', op: `${r.length} steps`, verified: true }));
  const overall = h('div', { class: 'lg-card wide' }, h('b', {}, 'Overall: average level per month'), months.length > 1 ? chart(months as Pt[], 620, 180) : h('p', { class: 'pg-sum' }, 'Needs cases in at least two months.'));

  const detail = h('div', { class: 'lg-detail' });
  const show = (s: typeof list[number]) => {
    detail.replaceChildren(h('b', {}, s.label), h('p', { class: 'pg-sum' },
      `${s.pts.length} case${s.pts.length > 1 ? 's' : ''}: from ${s.first.level} (${s.first.date}) to ${s.last.level} (${s.last.date}). `,
      s.firstDoing >= 0 ? `Doing the key parts from case ${s.firstDoing + 1}.` : 'Not yet doing the key parts.'), chart(s.pts),
      h('p', { class: 'foot' }, 'Filled dots: verified by the consultant; hollow: awaiting verification. Dashed line: the trend. Shaded band: S-TS and above.'));
    detail.scrollIntoView({ block: 'nearest' });
  };
  const cards = h('div', { class: 'lg-grid' }, ...list.map((s) => {
    const up = s.last.rank - s.first.rank;
    return h('button', { class: 'lg-card', onclick: () => show(s), title: 'Show the chart' },
      h('span', { class: 'lg-name' }, s.label), spark(s.pts),
      h('span', { class: 'lg-meta' }, `${s.pts.length}× · ${s.first.level} → `, h('b', { style: `color:${COL[s.last.rank]}` }, s.last.level),
        up > 0 ? h('i', { class: 'up' }, ' ▲') : up < 0 ? h('i', { class: 'down' }, ' ▼') : null));
  }));
  show(list[0]!);
  return h('div', { class: 'lg' }, monthly(h, entries), overall, h('h4', {}, 'Each skill (tap one for its chart)'), cards, detail, knhSummary(h, entries, start));
}

/** cases per month, stacked by the overall level of involvement, with a month-by-area table */
function monthly(h: H, entries: LogEntry[]): HTMLElement {
  const months = new Map<string, LogEntry[]>();
  for (const e of entries) { const m = String(e.data['date'] ?? '').slice(0, 7); if (m.length === 7) (months.get(m) ?? months.set(m, []).get(m)!).push(e); }
  if (!months.size) return h('div', {});
  // every month from the first to the last, so quiet months show as gaps
  const keys = [...months.keys()].sort(); const all: string[] = [];
  for (let [y, m] = keys[0]!.split('-').map(Number) as [number, number]; ; m++) { if (m > 12) { m = 1; y++; } const k = `${y}-${String(m).padStart(2, '0')}`; all.push(k); if (k >= keys[keys.length - 1]!) break; }
  const w = 640, hgt = 210, L = 30, R = 8, T = 14, B = 40; const iw = w - L - R, ih = hgt - T - B;
  const max = Math.max(...all.map((k) => months.get(k)?.length ?? 0), 1); const bw = iw / all.length;
  const g = svg('svg', { viewBox: `0 0 ${w} ${hgt}`, class: 'lg-chart', role: 'img' });
  for (let v = 0; v <= max; v += Math.max(1, Math.ceil(max / 4))) {
    const yy = T + ih - (v * ih) / max; g.append(svg('line', { x1: L, x2: L + iw, y1: yy, y2: yy, stroke: 'rgba(255,255,255,0.07)' }));
    const t = svg('text', { x: L - 6, y: yy + 4, 'text-anchor': 'end', 'font-size': 10, fill: '#8aa0ab' }); t.textContent = String(v); g.append(t);
  }
  all.forEach((k, i) => {
    const es = months.get(k) ?? []; let y0 = T + ih;
    for (const lv of LEVELS) {
      const n = es.filter((e) => e.data['level'] === lv.code).length; if (!n) continue;
      const hh = (n * ih) / max; y0 -= hh;
      const r = svg('rect', { x: L + i * bw + bw * 0.15, y: y0, width: bw * 0.7, height: hh, fill: COL[RANK[lv.code]!]!, rx: 2 });
      const tt = svg('title', {}); tt.textContent = `${k}: ${n} × ${lv.code} (${lv.name})`; r.append(tt); g.append(r);
    }
    if (es.length) { const t = svg('text', { x: L + i * bw + bw / 2, y: y0 - 3, 'text-anchor': 'middle', 'font-size': 10, fill: '#cfe3ea' }); t.textContent = String(es.length); g.append(t); }
    if (all.length <= 14 || i % Math.ceil(all.length / 12) === 0) {
      const t = svg('text', { x: L + i * bw + bw / 2, y: hgt - 24, 'text-anchor': 'middle', 'font-size': 10, fill: '#8aa0ab' }); t.textContent = k.slice(2); g.append(t);
    }
  });
  LEVELS.forEach((lv, i) => {
    g.append(svg('rect', { x: L + i * 82, y: hgt - 12, width: 9, height: 9, fill: COL[i]!, rx: 2 }));
    const t = svg('text', { x: L + i * 82 + 13, y: hgt - 4, 'font-size': 10, fill: '#8aa0ab' }); t.textContent = lv.code; g.append(t);
  });
  const areas = [...new Set(entries.map((e) => String(e.data['area'] || 'Not set')))].sort();
  const tbl = h('table', { class: 'lb-sum' }, h('tr', {}, h('th', {}, 'Month'), h('th', {}, 'Cases'), h('th', {}, 'Doing key parts'), h('th', {}, 'Verified'), ...areas.map((a) => h('th', {}, a))),
    ...[...all].reverse().filter((k) => months.has(k)).map((k) => { const es = months.get(k)!;
      return h('tr', {}, h('td', {}, k), h('td', {}, String(es.length)), h('td', {}, String(es.filter((e) => (RANK[String(e.data['level'])] ?? 0) >= SUP).length)),
        h('td', {}, String(es.filter((e) => e.status === 'verified').length)), ...areas.map((a) => h('td', {}, String(es.filter((e) => String(e.data['area'] || 'Not set') === a).length || '')))); }),
    h('tr', { class: 'op' }, h('td', {}, 'Total'), h('td', {}, String(entries.length)), h('td', {}, String(entries.filter((e) => (RANK[String(e.data['level'])] ?? 0) >= SUP).length)),
      h('td', {}, String(entries.filter((e) => e.status === 'verified').length)), ...areas.map((a) => h('td', {}, String(entries.filter((e) => String(e.data['area'] || 'Not set') === a).length)))));
  return h('div', { class: 'lg-card wide' }, h('b', {}, 'Cases per month (colored by the highest level you reached in each case)'), g, h('div', { class: 'tbl-wrap' }, tbl));
}

/** the KNH annual summary sheet: operations and component skills with P / 1A / 2A counts, by training year */
function knhSummary(h: H, entries: LogEntry[], start?: string | null): HTMLElement {
  const years = [...new Set(entries.map((e) => trainingYear(start, String(e.data['date'] ?? ''))))].filter((y) => y > 0).sort();
  const sel = h('select', {}, h('option', { value: '0' }, 'All years (grand summary)'), ...years.map((y) => h('option', { value: String(y) }, `Year ${y} of training`))) as HTMLSelectElement;
  const box = h('div', {});
  const knh = (lv: string) => LEVELS.find((l) => l.code === lv)?.knh ?? '';
  const render = () => {
    const yr = Number(sel.value);
    const es = entries.filter((e) => !yr || trainingYear(start, String(e.data['date'] ?? '')) === yr);
    const cnt = new Map<string, Record<string, number>>();
    const bump = (key: string, lv: string) => { const k = knh(lv); if (!k) return; const c = cnt.get(key) ?? {}; c[k] = (c[k] ?? 0) + 1; cnt.set(key, c); };
    for (const e of es) for (const o of e.data.ops ?? []) {
      const lvs = Object.values(o.comps ?? {}); const top = lvs.length ? lvs.reduce((a, b) => (RANK[b]! > RANK[a]! ? b : a)) : String(e.data['level'] ?? '');
      bump(`op:${o.op}`, top); for (const [k, lv] of Object.entries(o.comps ?? {})) bump(`${o.op}:${k}`, lv);
    }
    const row = (label: string, c: Record<string, number> | undefined, cls = '') => h('tr', { class: cls }, h('td', {}, label), ...['P', '1A', '2A'].map((k) => h('td', {}, String(c?.[k] ?? ''))),
      h('td', {}, String(c ? (c['P'] ?? 0) + (c['1A'] ?? 0) + (c['2A'] ?? 0) : '')));
    box.replaceChildren(...CATALOG.map((sec) => {
      const ops = sec.ops.filter((o) => cnt.has(`op:${o.id}`));
      if (!ops.length) return null;
      return h('table', { class: 'lb-sum knh' }, h('tr', {}, h('th', {}, sec.name), h('th', {}, 'P'), h('th', {}, '1A'), h('th', {}, '2A'), h('th', {}, 'Total')),
        ...ops.flatMap((o) => [row(`${o.name}${o.core ? '*' : ''}`, cnt.get(`op:${o.id}`), 'op'), ...o.skills.filter((k) => cnt.has(`${o.id}:${k}`)).map((k) => row(`   ${SKILLS[k]}`, cnt.get(`${o.id}:${k}`)))]));
    }).filter(Boolean) as HTMLElement[]);
    if (!box.children.length) box.append(h('p', { class: 'pg-sum' }, 'No catalogue operations logged in this period.'));
  };
  sel.onchange = render; render();
  return h('details', { class: 'pg-group', open: false }, h('summary', {}, 'KNH annual summary sheet (P / 1A / 2A)'),
    h('div', { class: 'lb-actions' }, sel, h('button', { class: 'btn', onclick: () => printSummary(box) }, 'Print for signature')),
    h('p', { class: 'foot' }, 'Crosswalk to the KNH codes: S-TS, S-TU, P and T count as P (primary); A as 1A; 2nd assistant as 2A; observed cases are not counted. * core procedure.'
      + (start ? '' : ' Add your training start date to your profile to split by year.')), box);
}

function printSummary(box: HTMLElement): void {
  const w = window.open('', '_blank'); if (!w) return;
  w.document.write(`<!doctype html><meta charset="utf-8"><title>TCVS annual summary</title><style>body{font:12px Arial;margin:24px}table{border-collapse:collapse;width:100%;margin:10px 0}
td,th{border:1px solid #999;padding:3px 6px;text-align:left}tr.op td{font-weight:bold}.sig{margin-top:30px;line-height:2.4}</style><h2>TCVS surgical logbook: annual summary</h2>${box.innerHTML}
<div class="sig">Learner's signature: ____________________ Date: __________<br>Supervising consultant: ____________________ Date: __________<br>Head of division: ____________________ Date: __________</div>`);
  w.document.close(); w.focus(); w.print();
}
