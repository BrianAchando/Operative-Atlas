// The theatre list: operations a trainee expects to do on a given day, for the Today screen and quick logging.
import { J, ready, currentUser, cut, sameSite } from '../../server/auth.js';

const day = (x) => (/^\d{4}-\d{2}-\d{2}$/.test(x ?? '') ? x : '');

export async function onRequestGet({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  await ready(env); const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  const qs = new URL(request.url).searchParams; const from = day(qs.get('from')), to = day(qs.get('to'));
  if (!from || !to) return J({ error: 'Bad dates.' }, 400);
  const rows = (await env.DB.prepare('SELECT * FROM plans WHERE user_id = ? AND date BETWEEN ? AND ? ORDER BY date, id').bind(u.id, from, to).all()).results;
  return J(rows);
}

export async function onRequestPost({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  const b = await request.json().catch(() => null) ?? {};
  const date = day(b.date); if (!date || !(b.proc || b.op || b.note)) return J({ error: 'Choose a date and an operation.' }, 400);
  const n = await env.DB.prepare('SELECT COUNT(*) n FROM plans WHERE user_id = ? AND date = ?').bind(u.id, date).first();
  if (n.n >= 15) return J({ error: 'That list is full (15 cases).' }, 400);
  const r = await env.DB.prepare('INSERT INTO plans (user_id, date, proc, op, note, created) VALUES (?, ?, ?, ?, ?, ?)')
    .bind(u.id, date, cut(b.proc, 80), cut(b.op, 40), cut(b.note, 200), new Date().toISOString()).run();
  return J({ ok: true, id: r.meta.last_row_id });
}

/** mark a planned case as logged (links it to the logbook entry) */
export async function onRequestPatch({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  const b = await request.json().catch(() => null);
  if (!b || !Number.isInteger(b.id)) return J({ error: 'Bad request.' }, 400);
  await env.DB.prepare('UPDATE plans SET log_id = ? WHERE id = ? AND user_id = ?').bind(Number.isInteger(b.log_id) ? b.log_id : null, b.id, u.id).run();
  return J({ ok: true });
}

export async function onRequestDelete({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  await env.DB.prepare('DELETE FROM plans WHERE id = ? AND user_id = ?').bind(Number(new URL(request.url).searchParams.get('id')), u.id).run();
  return J({ ok: true });
}
