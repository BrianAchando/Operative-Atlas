// Step 1 of sign-in: email a 6-digit code. Limits: 5 codes per email and 20 per address per hour.
import { J, ready, normEmail, validEmail, hmac, sendCode, sameSite } from '../../../server/auth.js';

export async function onRequestPost({ request, env }) {
  if (!env.DB || !env.AUTH_SECRET) return J({ error: 'Sign-in is not switched on yet.' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env);
  const b = await request.json().catch(() => null);
  const email = normEmail(b?.email);
  if (!validEmail(email)) return J({ error: 'Enter a valid email address.' }, 400);
  const ip = request.headers.get('cf-connecting-ip') ?? '';
  const hour = Date.now() - 3600e3;
  const byEmail = await env.DB.prepare('SELECT COUNT(*) n FROM otps WHERE email = ? AND created > ?').bind(email, hour).first();
  const byIp = ip ? await env.DB.prepare('SELECT COUNT(*) n FROM otps WHERE ip = ? AND created > ?').bind(ip, hour).first() : { n: 0 };
  if (byEmail.n >= 5 || byIp.n >= 20) return J({ error: 'Too many codes requested. Wait an hour and try again.' }, 429);
  const code = String(crypto.getRandomValues(new Uint32Array(1))[0] % 1e6).padStart(6, '0');
  await env.DB.prepare('DELETE FROM otps WHERE email = ? OR expires < ?').bind(email, Date.now() - 86400e3).run();
  await env.DB.prepare('INSERT INTO otps (email, code_hash, expires, attempts, created, ip) VALUES (?, ?, ?, 0, ?, ?)')
    .bind(email, await hmac(env.AUTH_SECRET, `otp:${email}:${code}`), Date.now() + 600e3, Date.now(), ip).run();
  if (env.DEV_CODES === '1') return J({ ok: true, dev_code: code });
  const sent = await sendCode(env, email, code);
  if (!sent) return J({ error: 'Could not send the email. Try again shortly.' }, 502);
  return J({ ok: true });
}
