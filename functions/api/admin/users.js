// Administrator: list accounts, approve consultants, change role or suspend.
import { J, ready, currentUser, cut, sameSite, pub } from '../../../server/auth.js';

const guard = async (request, env) => { await ready(env); const u = await currentUser(request, env); return u && u.admin ? u : null; };

export async function onRequestGet({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!(await guard(request, env))) return J({ error: 'Administrators only.' }, 403);
  const rows = (await env.DB.prepare(`SELECT u.*, (SELECT COUNT(*) FROM logbook l WHERE l.user_id = u.id) cases FROM users u
    ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, created DESC LIMIT 2000`).all()).results;
  return J(rows.map((r) => ({ ...pub(r, env), created: r.created, last_login: r.last_login, cases: r.cases })));
}

export async function onRequestPatch({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  if (!(await guard(request, env))) return J({ error: 'Administrators only.' }, 403);
  const b = await request.json().catch(() => null);
  if (!b || !Number.isInteger(b.id)) return J({ error: 'Bad request.' }, 400);
  const sets = [], vals = [];
  if (['active', 'pending', 'suspended'].includes(b.status)) { sets.push('status = ?'); vals.push(b.status); }
  if (['student', 'resident', 'consultant'].includes(b.role)) { sets.push('role = ?'); vals.push(b.role); }
  if (!sets.length) return J({ error: 'Nothing to change.' }, 400);
  await env.DB.prepare(`UPDATE users SET ${sets.join(', ')} WHERE id = ?`).bind(...vals, b.id).run();
  if (b.status === 'suspended') await env.DB.prepare('DELETE FROM sessions WHERE user_id = ?').bind(b.id).run();
  return J({ ok: true });
}
