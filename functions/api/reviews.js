// The monthly supervisor review: one rating, comment and plan per resident per month from each supervising consultant.
import { J, ready, currentUser, cut, sameSite, supervises, activeConsultant } from '../../server/auth.js';

export const RATINGS = ['On track', 'Needs support', 'Concern'];

export async function onRequestGet({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  await ready(env); const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  const rid = Number(new URL(request.url).searchParams.get('resident')) || u.id;
  const allowed = rid === u.id || u.admin || (activeConsultant(u) && await supervises(env, u.id, rid));
  if (!allowed) return J({ error: 'Not allowed.' }, 403);
  return J((await env.DB.prepare('SELECT r.*, c.name consultant FROM reviews r JOIN users c ON c.id = r.consultant_id WHERE r.resident_id = ? ORDER BY r.month DESC, r.id DESC').bind(rid).all()).results);
}

export async function onRequestPost({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); const u = await currentUser(request, env); if (!activeConsultant(u)) return J({ error: 'Consultants only.' }, 403);
  const b = await request.json().catch(() => null) ?? {};
  const rid = b.resident_id; const month = /^\d{4}-(0[1-9]|1[0-2])$/.test(b.month ?? '') ? b.month : '';
  if (!Number.isInteger(rid) || !month || !RATINGS.includes(b.rating)) return J({ error: 'Choose the month and a rating.' }, 400);
  if (!u.admin && !(await supervises(env, u.id, rid))) return J({ error: 'Only a supervising consultant can review this resident.' }, 403);
  await env.DB.prepare(`INSERT INTO reviews (resident_id, consultant_id, month, rating, comment, plan, created) VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (resident_id, consultant_id, month) DO UPDATE SET rating = excluded.rating, comment = excluded.comment, plan = excluded.plan, created = excluded.created`)
    .bind(rid, u.id, month, b.rating, cut(b.comment, 1500), cut(b.plan, 800), new Date().toISOString()).run();
  return J({ ok: true });
}
