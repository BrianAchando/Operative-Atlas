// The operative logbook. Residents (and consultants, for their own cases) record cases; the named consultant verifies them.
// Supervision codes follow the intercollegiate eLogbook (ISCP): O, A, S-TS, S-TU, P, T. No patient identifiers are stored.
import { J, ready, currentUser, cut, sameSite } from '../../server/auth.js';

const LEVELS = ['O', 'A2', 'A', 'S-TS', 'S-TU', 'P', 'T'];
const RANK = { O: 0, A2: 1, A: 2, 'S-TS': 3, 'S-TU': 4, P: 5, T: 6 };
const AREAS = ['Adult cardiac', 'Congenital cardiac', 'General thoracic', 'Esophageal', 'Airway', 'Vascular (open)', 'Endovascular', 'Trauma', 'Access / other'];
const URG = ['Elective', 'Urgent', 'Emergency', 'Salvage'];
const num = (x, lo, hi) => { const n = Number(x); return Number.isFinite(n) && n >= lo && n <= hi ? n : null; };
const pick = (x, list) => (list.includes(x) ? x : '');

/** keep only known fields, bounded */
function clean(b) {
  const d = {
    date: /^\d{4}-\d{2}-\d{2}$/.test(b.date ?? '') ? b.date : '',
    hospital: cut(b.hospital, 160), case_code: cut(b.case_code, 40),
    age: num(b.age, 0, 120), age_unit: pick(b.age_unit, ['days', 'months', 'years']) || 'years', sex: pick(b.sex, ['F', 'M']),
    urgency: pick(b.urgency, URG), area: pick(b.area, AREAS), diagnosis: cut(b.diagnosis, 300),
    proc: cut(b.proc, 80), proc_name: cut(b.proc_name, 200), approach: cut(b.approach, 120),
    level: pick(b.level, LEVELS), supervisor_id: Number.isInteger(b.supervisor_id) ? b.supervisor_id : null, supervisor_name: cut(b.supervisor_name, 120),
    cpb_min: num(b.cpb_min, 0, 2000), xclamp_min: num(b.xclamp_min, 0, 2000), arrest_min: num(b.arrest_min, 0, 600),
    blood_ml: num(b.blood_ml, 0, 50000), complication: pick(b.complication, ['None', 'Minor', 'Major']), clavien: pick(b.clavien, ['', 'I', 'II', 'IIIa', 'IIIb', 'IVa', 'IVb', 'V']),
    return_theatre: pick(b.return_theatre, ['No', 'Yes']), outcome_30d: pick(b.outcome_30d, ['Alive', 'Died', 'Unknown']), los_days: num(b.los_days, 0, 400),
    notes: cut(b.notes, 1500),
    // operations from the TCVS catalogue, each with its component skills and the level for each
    ops: (Array.isArray(b.ops) ? b.ops : []).slice(0, 6).map((o) => ({
      op: cut(o?.op, 40),
      comps: Object.fromEntries(Object.entries(o?.comps ?? {}).slice(0, 40).filter(([k, v]) => /^[a-z0-9:-]{1,40}$/.test(k) && LEVELS.includes(v))),
    })).filter((o) => o.op),
  };
  // the overall level defaults to the highest component level
  if (!d.level) { let best = -1; for (const o of d.ops) for (const v of Object.values(o.comps)) if (RANK[v] > best) { best = RANK[v]; d.level = v; } }
  return d;
}

const can = (u) => u && u.status === 'active' && (u.role === 'resident' || u.role === 'consultant' || u.admin);

export async function onRequestGet({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  await ready(env); const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  const qs = new URL(request.url).searchParams;
  const review = qs.get('review');
  // a consultant's trainees (residents with a case naming them), or every resident for the administrator
  if (qs.get('trainees')) {
    if (!(u.role === 'consultant' && u.status === 'active') && !u.admin) return J({ error: 'Consultants only.' }, 403);
    const q = u.admin
      ? env.DB.prepare(`SELECT u.id, u.name, u.institution, u.year, u.start, COUNT(l.id) cases, SUM(l.status = 'pending') pending FROM users u LEFT JOIN logbook l ON l.user_id = u.id
          WHERE u.role = 'resident' GROUP BY u.id ORDER BY u.name`)
      : env.DB.prepare(`SELECT u.id, u.name, u.institution, u.year, u.start, COUNT(l.id) cases, SUM(l.status = 'pending') pending FROM users u JOIN logbook l ON l.user_id = u.id
          WHERE u.id IN (SELECT user_id FROM logbook WHERE supervisor_id = ?) GROUP BY u.id ORDER BY u.name`).bind(u.id);
    return J((await q.all()).results);
  }
  // one resident's whole logbook, for their supervising consultant or the administrator (read only)
  const rid = Number(qs.get('resident'));
  if (rid) {
    const sup = u.admin || (u.role === 'consultant' && u.status === 'active' &&
      await env.DB.prepare('SELECT 1 FROM logbook WHERE user_id = ? AND supervisor_id = ? LIMIT 1').bind(rid, u.id).first());
    if (!sup) return J({ error: 'Only a consultant who supervises this resident can see the logbook.' }, 403);
    const who = await env.DB.prepare('SELECT id, name, institution, year, start FROM users WHERE id = ?').bind(rid).first();
    const rows = (await env.DB.prepare(`SELECT l.*, s.name reviewer FROM logbook l LEFT JOIN users s ON s.id = l.reviewer_id WHERE l.user_id = ? ORDER BY l.date, l.id`).bind(rid).all()).results;
    return J({ resident: who, entries: rows.map((r) => ({ ...r, data: JSON.parse(r.data) })) });
  }
    if (review) {
    if (!(u.role === 'consultant' && u.status === 'active') && !u.admin) return J({ error: 'Consultants only.' }, 403);
    const q = u.admin && review === 'all'
      ? env.DB.prepare(`SELECT l.*, u.name resident, u.institution FROM logbook l JOIN users u ON u.id = l.user_id ORDER BY CASE l.status WHEN 'pending' THEN 0 ELSE 1 END, l.date DESC LIMIT 1000`)
      : env.DB.prepare(`SELECT l.*, u.name resident, u.institution FROM logbook l JOIN users u ON u.id = l.user_id WHERE l.supervisor_id = ?
          ORDER BY CASE l.status WHEN 'pending' THEN 0 ELSE 1 END, l.date DESC LIMIT 1000`).bind(u.id);
    return J((await q.all()).results.map((r) => ({ ...r, data: JSON.parse(r.data) })));
  }
  const rows = (await env.DB.prepare(`SELECT l.*, s.name reviewer FROM logbook l LEFT JOIN users s ON s.id = l.reviewer_id WHERE l.user_id = ? ORDER BY l.date DESC, l.id DESC`).bind(u.id).all()).results;
  return J(rows.map((r) => ({ ...r, data: JSON.parse(r.data) })));
}

export async function onRequestPost({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); const u = await currentUser(request, env);
  if (!can(u)) return J({ error: 'The logbook is for residents and consultants with an active account.' }, 403);
  const d = clean(await request.json().catch(() => ({})) ?? {});
  if (!d.date || !d.level || !(d.proc || d.proc_name || d.ops.length)) return J({ error: 'Date, procedure and your level of involvement are needed.' }, 400);
  const now = new Date().toISOString();
  const r = await env.DB.prepare('INSERT INTO logbook (user_id, data, proc, date, level, supervisor_id, status, created, updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)')
    .bind(u.id, JSON.stringify(d), d.proc || d.proc_name || d.ops[0]?.op, d.date, d.level, d.supervisor_id, 'pending', now, now).run();
  return J({ ok: true, id: r.meta.last_row_id });
}

export async function onRequestPut({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); const u = await currentUser(request, env); if (!can(u)) return J({ error: 'Not allowed.' }, 403);
  const b = await request.json().catch(() => null); if (!b || !Number.isInteger(b.id)) return J({ error: 'Bad request.' }, 400);
  const d = clean(b);
  if (!d.date || !d.level || !(d.proc || d.proc_name || d.ops.length)) return J({ error: 'Date, procedure and your level of involvement are needed.' }, 400);
  // an edit sends the case back for verification
  const r = await env.DB.prepare(`UPDATE logbook SET data = ?, proc = ?, date = ?, level = ?, supervisor_id = ?, status = 'pending', reviewer_id = NULL, review_comment = NULL, reviewed = NULL, updated = ?
    WHERE id = ? AND user_id = ?`).bind(JSON.stringify(d), d.proc || d.proc_name || d.ops[0]?.op, d.date, d.level, d.supervisor_id, new Date().toISOString(), b.id, u.id).run();
  return r.meta.changes ? J({ ok: true }) : J({ error: 'Not found.' }, 404);
}

export async function onRequestDelete({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  const id = Number(new URL(request.url).searchParams.get('id'));
  const r = await env.DB.prepare('DELETE FROM logbook WHERE id = ? AND user_id = ?').bind(id, u.id).run();
  return r.meta.changes ? J({ ok: true }) : J({ error: 'Not found.' }, 404);
}

/** the supervising consultant (or an administrator) verifies or returns a case */
export async function onRequestPatch({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  const b = await request.json().catch(() => null);
  if (!b || !Number.isInteger(b.id) || !['verified', 'returned', 'pending'].includes(b.status)) return J({ error: 'Bad request.' }, 400);
  const row = await env.DB.prepare('SELECT * FROM logbook WHERE id = ?').bind(b.id).first(); if (!row) return J({ error: 'Not found.' }, 404);
  const mine = u.role === 'consultant' && u.status === 'active' && row.supervisor_id === u.id;
  if (!mine && !u.admin) return J({ error: 'Only the named supervisor can verify this case.' }, 403);
  if (row.user_id === u.id && !u.admin) return J({ error: 'You cannot verify your own case.' }, 403);
  await env.DB.prepare('UPDATE logbook SET status = ?, reviewer_id = ?, review_comment = ?, reviewed = ? WHERE id = ?')
    .bind(b.status, u.id, cut(b.comment, 600), new Date().toISOString(), b.id).run();
  return J({ ok: true });
}
