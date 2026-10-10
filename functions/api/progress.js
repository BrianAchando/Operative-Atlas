// Steps seen and first answers, kept with the account. The client merges: a step seen anywhere stays seen; the first answer wins.
import { J, ready, currentUser, sameSite } from '../../server/auth.js';

export async function onRequestGet({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  await ready(env); const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  const r = await env.DB.prepare('SELECT data FROM progress WHERE user_id = ?').bind(u.id).first();
  return J({ data: r ? JSON.parse(r.data) : null });
}

export async function onRequestPut({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  const txt = await request.text(); if (txt.length > 900000) return J({ error: 'Too large.' }, 413);
  let d; try { d = JSON.parse(txt); } catch { return J({ error: 'Bad JSON.' }, 400); }
  if (!d || typeof d.v !== 'object' || typeof d.q !== 'object') return J({ error: 'Bad data.' }, 400);
  await env.DB.prepare('INSERT INTO progress (user_id, data, updated) VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET data = excluded.data, updated = excluded.updated')
    .bind(u.id, JSON.stringify({ v: d.v, q: d.q, sr: typeof d.sr === 'object' ? d.sr : {}, daily: typeof d.daily === 'object' ? d.daily : {}, exam: typeof d.exam === 'object' ? d.exam : {} }), new Date().toISOString()).run();
  return J({ ok: true });
}
