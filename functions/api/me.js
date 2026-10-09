// The signed-in user's profile. Students and residents are active at once; consultants wait for the administrator.
import { J, ready, currentUser, pub, cut, ROLES, sameSite, isAdmin } from '../../server/auth.js';

export async function onRequestGet({ request, env }) {
  if (!env.DB || !env.AUTH_SECRET) return J({ user: null, enabled: false });
  await ready(env);
  return J({ user: pub(await currentUser(request, env), env), enabled: true });
}

export async function onRequestPost({ request, env }) {
  if (!env.DB || !env.AUTH_SECRET) return J({ error: 'Sign-in is not switched on yet.' }, 503);
  if (!sameSite(request)) return J({ error: 'Bad request.' }, 400);
  await ready(env);
  const u = await currentUser(request, env); if (!u) return J({ error: 'Sign in first.' }, 401);
  const b = await request.json().catch(() => null) ?? {};
  const name = cut(b.name, 100); const role = ROLES.includes(b.role) ? b.role : null;
  if (!name || !role) return J({ error: 'Add your name and choose a role.' }, 400);
  const institution = cut(b.institution, 160); const hospital = cut(b.hospital, 160);
  if ((role === 'student' || role === 'resident') && !institution) return J({ error: 'Choose your university or training institution.' }, 400);
  if (role === 'consultant' && !hospital) return J({ error: 'Choose your hospital.' }, 400);
  // a role change away from consultant is free; becoming a consultant needs approval (admins are approved by their email)
  let status = u.status;
  if (role === 'consultant' && (u.role !== 'consultant' || u.status === 'new')) status = isAdmin(env, u) ? 'active' : 'pending';
  else if (role !== 'consultant' && u.status !== 'suspended') status = 'active';
  const start = /^\d{4}-\d{2}$/.test(b.start ?? '') ? b.start : '';
  await env.DB.prepare('UPDATE users SET name = ?, role = ?, institution = ?, hospital = ?, year = ?, reg_no = ?, start = ?, status = ? WHERE id = ?')
    .bind(name, role, institution, hospital, cut(b.year, 20), cut(b.reg_no, 40), start, status, u.id).run();
  const nu = await env.DB.prepare('SELECT * FROM users WHERE id = ?').bind(u.id).first();
  return J({ ok: true, user: pub(nu, env) });
}
