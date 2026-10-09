import { J, ready, endSession, sessionCookie, sameSite } from '../../../server/auth.js';

export async function onRequestPost({ request, env }) {
  if (!env.DB) return J({ ok: true });
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env); await endSession(request, env);
  return J({ ok: true }, 200, { 'set-cookie': sessionCookie('', 0, request) });
}
