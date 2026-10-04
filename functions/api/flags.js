// Review flags: anyone using the atlas can raise one; listing and resolving them needs the review key.
// Bindings (Cloudflare Pages > Settings): D1 database as DB; secret REVIEW_KEY.
const J = (d, s = 200) => new Response(JSON.stringify(d), { status: s, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } });
const cut = (x, n) => String(x ?? '').slice(0, n);
const KINDS = ['wrong', 'outdated', 'unclear', 'missing', 'typo', 'other'];
// the table creates itself on first use, so no separate schema step is needed
const ready = (env) => env.DB.prepare(`CREATE TABLE IF NOT EXISTS flags (id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT NOT NULL, proc TEXT NOT NULL, approach TEXT,
  step TEXT, step_title TEXT, kind TEXT, comment TEXT NOT NULL, name TEXT, status TEXT NOT NULL DEFAULT 'open', resolution TEXT, resolved TEXT)`).run();
const allowed = (request, env) => !!env.REVIEW_KEY && request.headers.get('x-review-key') === env.REVIEW_KEY;

export async function onRequestPost({ request, env }) {
  if (!env.DB) return J({ error: 'Review is not switched on yet.' }, 503);
  await ready(env);
  const b = await request.json().catch(() => null);
  if (!b || b.website) return J({ ok: true });                       // honeypot: bots fill the hidden field
  if (!b.proc || !cut(b.comment, 2000).trim()) return J({ error: 'Add a comment.' }, 400);
  await env.DB.prepare('INSERT INTO flags (created, proc, approach, step, step_title, kind, comment, name, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)')
    .bind(new Date().toISOString(), cut(b.proc, 80), cut(b.approach, 160), cut(b.step, 80), cut(b.stepTitle, 240), KINDS.includes(b.kind) ? b.kind : 'other',
          cut(b.comment, 2000).trim(), cut(b.name, 80).trim(), 'open').run();
  return J({ ok: true });
}

export async function onRequestGet({ request, env }) {
  if (!env.DB) return J({ error: 'Review is not switched on yet.' }, 503);
  await ready(env);
  if (!allowed(request, env)) return J({ error: 'Enter the review key.' }, 403);
  const st = new URL(request.url).searchParams.get('status');
  const q = st ? env.DB.prepare('SELECT * FROM flags WHERE status = ? ORDER BY id DESC LIMIT 1000').bind(st) : env.DB.prepare('SELECT * FROM flags ORDER BY id DESC LIMIT 1000');
  return J((await q.all()).results);
}

export async function onRequestPatch({ request, env }) {
  if (!env.DB) return J({ error: 'Review is not switched on yet.' }, 503);
  await ready(env);
  if (!allowed(request, env)) return J({ error: 'Enter the review key.' }, 403);
  const b = await request.json().catch(() => null);
  if (!b || !Number.isInteger(b.id) || !['open', 'fixed', 'rejected'].includes(b.status)) return J({ error: 'Bad request.' }, 400);
  await env.DB.prepare('UPDATE flags SET status = ?, resolution = ?, resolved = ? WHERE id = ?')
    .bind(b.status, cut(b.resolution, 1000), b.status === 'open' ? null : new Date().toISOString(), b.id).run();
  return J({ ok: true });
}
