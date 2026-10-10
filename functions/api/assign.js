// Modules a consultant assigns to residents (an atlas operation to work through by a date). The resident sees them on
// the Today screen and marks them done; the consultant sees completion and the question score.
import { J, ready, currentUser, cut, sameSite, activeConsultant } from '../../server/auth.js';

const day = (x) => (/^\d{4}-\d{2}-\d{2}$/.test(x ?? '') ? x : null);

export async function onRequestGet({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  await ready(env); const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  const qs = new URL(request.url).searchParams;
  // residents a consultant can assign to: every active resident (names and institution only)
  if (qs.get('residents')) {
    if (!activeConsultant(u)) return J({ error: 'Consultants only.' }, 403);
    return J((await env.DB.prepare(`SELECT id, name, institution, year, start FROM users WHERE role = 'resident' AND status = 'active' ORDER BY institution, name`).all()).results);
  }
  // what this consultant has assigned (optionally to one resident)
  if (qs.get('by') === 'me') {
    if (!activeConsultant(u)) return J({ error: 'Consultants only.' }, 403);
    const rid = Number(qs.get('resident'));
    const q = rid
      ? env.DB.prepare('SELECT a.*, r.name resident FROM assignments a JOIN users r ON r.id = a.resident_id WHERE a.by_id = ? AND a.resident_id = ? ORDER BY a.created DESC').bind(u.id, rid)
      : env.DB.prepare('SELECT a.*, r.name resident FROM assignments a JOIN users r ON r.id = a.resident_id WHERE a.by_id = ? ORDER BY a.created DESC LIMIT 300').bind(u.id);
    return J((await q.all()).results);
  }
  // the resident's own assignments
  return J((await env.DB.prepare('SELECT a.*, c.name by_name FROM assignments a JOIN users c ON c.id = a.by_id WHERE a.resident_id = ? ORDER BY a.done IS NOT NULL, a.due, a.id').bind(u.id).all()).results);
}

/** assign one atlas module to one or more residents */
export async function onRequestPost({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); const u = await currentUser(request, env); if (!activeConsultant(u)) return J({ error: 'Consultants only.' }, 403);
  const b = await request.json().catch(() => null) ?? {};
  const ids = (Array.isArray(b.resident_ids) ? b.resident_ids : []).filter(Number.isInteger).slice(0, 60);
  const proc = cut(b.proc, 80);
  if (!ids.length || !proc) return J({ error: 'Choose the residents and a module.' }, 400);
  const ok = (await env.DB.prepare(`SELECT id FROM users WHERE role = 'resident' AND status = 'active' AND id IN (${ids.map(() => '?').join(',')})`).bind(...ids).all()).results.map((r) => r.id);
  const now = new Date().toISOString();
  await env.DB.batch(ok.map((rid) => env.DB.prepare('INSERT INTO assignments (resident_id, by_id, proc, name, note, due, created) VALUES (?, ?, ?, ?, ?, ?, ?)')
    .bind(rid, u.id, proc, cut(b.name, 200), cut(b.note, 300), day(b.due), now)));
  return J({ ok: true, n: ok.length });
}

/** the resident marks an assignment done (or not), with the question score from the module */
export async function onRequestPatch({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  const b = await request.json().catch(() => null);
  if (!b || !Number.isInteger(b.id)) return J({ error: 'Bad request.' }, 400);
  await env.DB.prepare('UPDATE assignments SET done = ?, score = ? WHERE id = ? AND resident_id = ?')
    .bind(b.done ? new Date().toISOString() : null, b.done ? cut(b.score, 20) : null, b.id, u.id).run();
  return J({ ok: true });
}

export async function onRequestDelete({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); const u = await currentUser(request, env); if (!activeConsultant(u)) return J({ error: 'Consultants only.' }, 403);
  const id = Number(new URL(request.url).searchParams.get('id'));
  const r = u.admin ? await env.DB.prepare('DELETE FROM assignments WHERE id = ?').bind(id).run()
    : await env.DB.prepare('DELETE FROM assignments WHERE id = ? AND by_id = ?').bind(id, u.id).run();
  return r.meta.changes ? J({ ok: true }) : J({ error: 'Not found.' }, 404);
}
