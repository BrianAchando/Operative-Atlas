// Step 2 of sign-in: check the code, create the account on first sign-in, start a 30-day session.
import { J, ready, normEmail, hmac, same, newSession, sessionCookie, pub, sameSite } from '../../../server/auth.js';

export async function onRequestPost({ request, env }) {
  if (!env.DB || !env.AUTH_SECRET) return J({ error: 'Sign-in is not switched on yet.' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env);
  const b = await request.json().catch(() => null);
  const email = normEmail(b?.email); const code = String(b?.code ?? '').replace(/\D/g, '');
  const row = await env.DB.prepare('SELECT rowid, * FROM otps WHERE email = ? ORDER BY created DESC LIMIT 1').bind(email).first();
  if (!row || row.expires < Date.now()) return J({ error: 'The code has expired. Ask for a new one.' }, 400);
  if (row.attempts >= 5) return J({ error: 'Too many wrong tries. Ask for a new code.' }, 429);
  const ok = code.length === 6 && same(row.code_hash, await hmac(env.AUTH_SECRET, `otp:${email}:${code}`));
  if (!ok) {
    await env.DB.prepare('UPDATE otps SET attempts = attempts + 1 WHERE rowid = ?').bind(row.rowid).run();
    return J({ error: 'That code is not right. Check the latest email.' }, 400);
  }
  await env.DB.prepare('DELETE FROM otps WHERE email = ?').bind(email).run();
  const now = new Date().toISOString();
  let user = await env.DB.prepare('SELECT * FROM users WHERE email = ?').bind(email).first();
  if (!user) {
    await env.DB.prepare("INSERT INTO users (email, status, created, last_login) VALUES (?, 'new', ?, ?)").bind(email, now, now).run();
    user = await env.DB.prepare('SELECT * FROM users WHERE email = ?').bind(email).first();
  } else await env.DB.prepare('UPDATE users SET last_login = ? WHERE id = ?').bind(now, user.id).run();
  if (user.status === 'suspended') return J({ error: 'This account is suspended. Contact the atlas administrator.' }, 403);
  const tok = await newSession(env, user.id);
  return J({ ok: true, user: pub(user, env) }, 200, { 'set-cookie': sessionCookie(tok, 30 * 86400, request) });
}
