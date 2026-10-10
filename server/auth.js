// Accounts for COVA: email one-time codes, sessions, roles. Shared by the Pages Functions in functions/api.
// Bindings (Cloudflare Pages > Settings > Functions): D1 database DB; secrets AUTH_SECRET (long random string),
// RESEND_KEY + MAIL_FROM (verified domain) or BREVO_KEY + MAIL_FROM (verified sender address); ADMIN_EMAILS (comma list).
// DEV_CODES=1 returns the code in the response: local testing only, never in production.

export const J = (d, s = 200, extra = {}) => new Response(JSON.stringify(d), { status: s, headers: { 'content-type': 'application/json', 'cache-control': 'no-store', ...extra } });
export const cut = (x, n) => String(x ?? '').trim().slice(0, n);
export const ROLES = ['student', 'resident', 'consultant'];
const COOKIE = 'cova_s';
const DAY = 86400e3;

const enc = new TextEncoder();
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
export async function hmac(secret, text) {
  const k = await crypto.subtle.importKey('raw', enc.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  return hex(await crypto.subtle.sign('HMAC', k, enc.encode(text)));
}
export const rand = (n = 32) => hex(crypto.getRandomValues(new Uint8Array(n)));
const same = (a, b) => { if (a.length !== b.length) return false; let x = 0; for (let i = 0; i < a.length; i++) x |= a.charCodeAt(i) ^ b.charCodeAt(i); return x === 0; };
export const normEmail = (e) => cut(e, 200).toLowerCase();
export const validEmail = (e) => /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(e);

let readyOnce = null;
export function ready(env) {
  if (!env.DB) throw new Error('no-db');
  readyOnce ??= env.DB.batch([
    env.DB.prepare(`CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL, name TEXT, role TEXT, status TEXT NOT NULL DEFAULT 'new',
      institution TEXT, hospital TEXT, year TEXT, reg_no TEXT, start TEXT, created TEXT NOT NULL, last_login TEXT)`),
    env.DB.prepare(`CREATE TABLE IF NOT EXISTS otps (email TEXT NOT NULL, code_hash TEXT NOT NULL, expires INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, created INTEGER NOT NULL, ip TEXT)`),
    env.DB.prepare(`CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created INTEGER NOT NULL, expires INTEGER NOT NULL)`),
    env.DB.prepare(`CREATE TABLE IF NOT EXISTS progress (user_id INTEGER PRIMARY KEY, data TEXT NOT NULL, updated TEXT NOT NULL)`),
    env.DB.prepare(`CREATE TABLE IF NOT EXISTS logbook (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, data TEXT NOT NULL, proc TEXT, date TEXT, level TEXT,
      supervisor_id INTEGER, status TEXT NOT NULL DEFAULT 'pending', reviewer_id INTEGER, review_comment TEXT, reviewed TEXT, created TEXT NOT NULL, updated TEXT NOT NULL)`),
    env.DB.prepare(`CREATE TABLE IF NOT EXISTS plans (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, date TEXT NOT NULL, proc TEXT, op TEXT, note TEXT,
      log_id INTEGER, created TEXT NOT NULL)`),
    env.DB.prepare(`CREATE TABLE IF NOT EXISTS assignments (id INTEGER PRIMARY KEY AUTOINCREMENT, resident_id INTEGER NOT NULL, by_id INTEGER NOT NULL, proc TEXT NOT NULL, name TEXT,
      note TEXT, due TEXT, created TEXT NOT NULL, done TEXT, score TEXT)`),
    env.DB.prepare(`CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, resident_id INTEGER NOT NULL, consultant_id INTEGER NOT NULL, month TEXT NOT NULL,
      rating TEXT, comment TEXT, plan TEXT, created TEXT NOT NULL, UNIQUE (resident_id, consultant_id, month))`),
    env.DB.prepare('CREATE INDEX IF NOT EXISTS as_res ON assignments (resident_id)'),
    env.DB.prepare('CREATE INDEX IF NOT EXISTS as_by ON assignments (by_id)'),
    env.DB.prepare('CREATE INDEX IF NOT EXISTS pl_user ON plans (user_id, date)'),
    env.DB.prepare('CREATE INDEX IF NOT EXISTS lb_user ON logbook (user_id)'),
    env.DB.prepare('CREATE INDEX IF NOT EXISTS lb_sup ON logbook (supervisor_id, status)'),
  ]).then(() => env.DB.prepare('ALTER TABLE users ADD COLUMN start TEXT').run().catch(() => undefined))   // older tables: add the training start date
    .then(() => env.DB.prepare("UPDATE logbook SET status = 'recorded' WHERE status != 'recorded'").run())   // cases no longer need consultant sign-off
    .catch((e) => { readyOnce = null; throw e; });
  return readyOnce;
}

const admins = (env) => String(env.ADMIN_EMAILS ?? '').toLowerCase().split(',').map((s) => s.trim()).filter(Boolean);
export const isAdmin = (env, user) => !!user && admins(env).includes(user.email);

/** Secure on https (always, once deployed); plain http only for local testing */
export function sessionCookie(token, maxAgeSec, request) {
  const secure = !request || new URL(request.url).protocol === 'https:' ? ' Secure;' : '';
  return `${COOKIE}=${token}; Path=/; HttpOnly;${secure} SameSite=Lax; Max-Age=${maxAgeSec}`;
}
const readCookie = (request) => (request.headers.get('cookie') ?? '').split(/;\s*/).map((c) => c.split('=')).find(([k]) => k === COOKIE)?.[1] ?? '';

/** the signed-in user, or null */
export async function currentUser(request, env) {
  const tok = readCookie(request); if (!tok || !env.AUTH_SECRET) return null;
  const th = await hmac(env.AUTH_SECRET, 'session:' + tok);
  const row = await env.DB.prepare('SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ? AND s.expires > ?').bind(th, Date.now()).first();
  if (!row) return null;
  if (isAdmin(env, row)) row.admin = true;
  return row;
}

export async function newSession(env, userId) {
  const tok = rand(32); const th = await hmac(env.AUTH_SECRET, 'session:' + tok);
  await env.DB.prepare('INSERT INTO sessions (token_hash, user_id, created, expires) VALUES (?, ?, ?, ?)').bind(th, userId, Date.now(), Date.now() + 30 * DAY).run();
  return tok;
}
export async function endSession(request, env) {
  const tok = readCookie(request); if (!tok || !env.AUTH_SECRET) return;
  await env.DB.prepare('DELETE FROM sessions WHERE token_hash = ?').bind(await hmac(env.AUTH_SECRET, 'session:' + tok)).run();
}

/** a mutating request must be JSON from our own pages (SameSite cookie plus this header blocks cross-site forms) */
export const sameSite = (request) => request.headers.get('x-cova') === '1';

/** the public view of a user */
export const pub = (u, env) => u && ({ id: u.id, email: u.email, name: u.name, role: u.role, status: u.status, institution: u.institution, hospital: u.hospital,
  year: u.year, reg_no: u.reg_no, start: u.start, admin: isAdmin(env, u) });

export const esc = (t) => String(t ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

/** send one email through Resend or Brevo; false when neither is set up or the provider refuses */
export async function sendMail(env, to, subject, text, html) {
  if (env.RESEND_KEY) {
    const r = await fetch('https://api.resend.com/emails', { method: 'POST', headers: { authorization: `Bearer ${env.RESEND_KEY}`, 'content-type': 'application/json' },
      body: JSON.stringify({ from: env.MAIL_FROM, to: [to], subject, text, html }) });
    return r.ok;
  }
  if (env.BREVO_KEY) {
    const m = /^(.*)<(.+)>$/.exec(env.MAIL_FROM ?? '');
    const sender = m ? { name: m[1].trim() || 'COVA', email: m[2].trim() } : { name: 'COVA', email: env.MAIL_FROM };
    const r = await fetch('https://api.brevo.com/v3/smtp/email', { method: 'POST', headers: { 'api-key': env.BREVO_KEY, 'content-type': 'application/json', accept: 'application/json' },
      body: JSON.stringify({ sender, to: [{ email: to }], subject, textContent: text, htmlContent: html }) });
    return r.ok;
  }
  return false;
}

export async function sendCode(env, email, code) {
  const subject = `Your COVA sign-in code: ${code}`;
  const text = `Your COVA sign-in code is ${code}\n\nIt expires in 10 minutes. If you did not ask for it, ignore this email.\n\nCOVA, Cardiothoracic Operative and Vascular Atlas`;
  const html = `<p>Your COVA sign-in code is</p><p style="font-size:28px;font-weight:700;letter-spacing:4px">${code}</p><p>It expires in 10 minutes. If you did not ask for it, ignore this email.</p><p style="color:#667">COVA, Cardiothoracic Operative and Vascular Atlas</p>`;
  return sendMail(env, email, subject, text, html);
}

/** does this consultant supervise this resident (a case naming them, or a module they assigned)? */
export async function supervises(env, cid, rid) {
  return !!(await env.DB.prepare('SELECT 1 FROM logbook WHERE user_id = ? AND supervisor_id = ? UNION SELECT 1 FROM assignments WHERE resident_id = ? AND by_id = ? LIMIT 1')
    .bind(rid, cid, rid, cid).first());
}
export const activeConsultant = (u) => !!u && ((u.role === 'consultant' && u.status === 'active') || u.admin);

export { same, DAY };
