// The consultant's weekly email: what each trainee logged this week (by level), modules done and overdue, and in the first week of a month the monthly reviews that are due.
// POST by the weekly cron worker (header x-cron-key = CRON_KEY) or by the administrator ("Send weekly emails now").
// GET ?preview=1 shows a signed-in consultant their own email.
import { J, ready, currentUser, sameSite, sendMail, esc, activeConsultant } from '../../server/auth.js';

const iso = (d) => d.toISOString().slice(0, 10);
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

export async function build(env, origin, c) {
  const now = new Date(); const weekAgo = iso(new Date(now.getTime() - 7 * 864e5)); const today = iso(now);
  // cases that name this consultant as supervisor, this week (for information; no sign-off needed)
  const cases = (await env.DB.prepare(`SELECT l.date, l.level, l.data, r.name resident FROM logbook l JOIN users r ON r.id = l.user_id
    WHERE l.supervisor_id = ? AND l.user_id != ? AND l.date >= ? ORDER BY r.name, l.date LIMIT 60`).bind(c.id, c.id, weekAgo).all()).results;
  const trainees = (await env.DB.prepare(`SELECT u.id, u.name,
      (SELECT COUNT(*) FROM logbook WHERE user_id = u.id AND date >= ?) wk,
      (SELECT COUNT(*) FROM assignments WHERE resident_id = u.id AND by_id = ? AND done IS NULL AND due < ?) overdue,
      (SELECT COUNT(*) FROM assignments WHERE resident_id = u.id AND by_id = ? AND done >= ?) done_wk
    FROM users u WHERE u.id IN (SELECT user_id FROM logbook WHERE supervisor_id = ? UNION SELECT resident_id FROM assignments WHERE by_id = ?) ORDER BY u.name`)
    .bind(weekAgo, c.id, today, c.id, weekAgo, c.id, c.id).all()).results;
  // monthly reviews: in the first 7 days of a month, for trainees with cases last month and no review from this consultant
  let due = []; let lastMonth = '';
  if (now.getUTCDate() <= 7) {
    const lm = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() - 1, 1)); lastMonth = lm.toISOString().slice(0, 7);
    due = (await env.DB.prepare(`SELECT DISTINCT u.id, u.name FROM logbook l JOIN users u ON u.id = l.user_id WHERE l.supervisor_id = ? AND substr(l.date, 1, 7) = ?
      AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.resident_id = u.id AND r.consultant_id = ? AND r.month = ?) ORDER BY u.name`).bind(c.id, lastMonth, c.id, lastMonth).all()).results;
  }
  const active = trainees.filter((t) => t.wk || t.overdue || t.done_wk);
  if (!cases.length && !active.length && !due.length) return null;

  const name = (c.name ?? '').trim() || 'Doctor';
  const rows = cases.map((p) => ({ p, d: JSON.parse(p.data) }));
  const td = 'style="padding:6px 8px;border-bottom:1px solid #e3e6ea;vertical-align:top"';
  const btn = (href, t) => `<a href="${href}" style="display:inline-block;background:#1f9e8f;color:#fff;text-decoration:none;padding:6px 12px;border-radius:4px;font-weight:600">${t}</a>`;
  const html = [`<div style="font-family:Segoe UI,Arial,sans-serif;color:#1d2733;max-width:680px">`,
    `<h2 style="margin:0 0 4px">COVA weekly summary</h2><p style="color:#667;margin:0 0 16px">For ${esc(name)} · week to ${today}</p>`,
    rows.length ? `<h3>Cases with you this week</h3><table style="border-collapse:collapse;width:100%;font-size:14px">
      <tr><th align="left" ${td}>Resident</th><th align="left" ${td}>Date</th><th align="left" ${td}>Operation</th><th align="left" ${td}>Their level</th></tr>
      ${rows.map(({ p, d }) => `<tr><td ${td}>${esc(p.resident)}</td><td ${td}>${esc(d.date)}</td><td ${td}>${esc(d.proc_name || d.proc || '')}${d.diagnosis ? `<br><small style="color:#667">${esc(d.diagnosis)}</small>` : ''}</td><td ${td}><b>${esc(p.level || d.level || '')}</b></td></tr>`).join('')}
      </table><p style="color:#667;font-size:13px">For information only: cases do not need your sign-off. Use the monthly review for feedback.</p>` : '',
    active.length ? `<h3>Your trainees this week</h3><table style="border-collapse:collapse;width:100%;font-size:14px">
      <tr><th align="left" ${td}>Resident</th><th align="left" ${td}>Cases logged</th><th align="left" ${td}>Modules done</th><th align="left" ${td}>Modules overdue</th></tr>
      ${active.map((t) => `<tr><td ${td}>${esc(t.name)}</td><td ${td}>${t.wk}</td><td ${td}>${t.done_wk}</td><td ${td}>${t.overdue ? `<b style="color:#b3261e">${t.overdue}</b>` : '0'}</td></tr>`).join('')}</table>` : '',
    due.length ? `<h3>Monthly review due: ${MONTHS[Number(lastMonth.slice(5)) - 1]}</h3><p>${due.map((t) => esc(t.name)).join(', ')}</p><p style="font-size:13px;color:#667">In COVA: your name, then Trainees, Open, Monthly review.</p>` : '',
    `<p style="margin-top:20px">${btn(origin + '/', 'Open COVA')}</p>`,
    `<p style="color:#99a;font-size:12px">You get this because residents name you as supervisor, or you assigned them modules, in COVA. No patient identifiers are stored or sent.</p></div>`].join('');
  const text = [`COVA weekly summary for ${name}, week to ${today}`, '',
    rows.length ? 'Cases with you this week (no sign-off needed):' : '',
    ...rows.map(({ p, d }) => `- ${p.resident}, ${d.date}, ${d.proc_name || d.proc || ''}, ${p.level || d.level || ''}`), '',
    active.length ? 'Trainees this week:' : '', ...active.map((t) => `- ${t.name}: ${t.wk} cases, ${t.done_wk} modules done, ${t.overdue} overdue`), '',
    due.length ? `Monthly review due (${lastMonth}): ${due.map((t) => t.name).join(', ')}` : '', '', `${origin}/`].filter((x, i, a) => x || a[i - 1]).join('\n');
  const subject = due.length ? 'COVA: monthly reviews due' : 'COVA: your trainees this week';
  return { subject, html, text };
}

export async function onRequestGet({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  await ready(env); const u = await currentUser(request, env); if (!activeConsultant(u)) return J({ error: 'Consultants only.' }, 403);
  const m = await build(env, new URL(request.url).origin, u);
  return new Response(m ? m.html : '<p style="font-family:sans-serif">Nothing to report this week: no cases waiting, no trainee activity, no reviews due.</p>',
    { headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' } });
}

export async function onRequestPost({ request, env }) {
  if (!env.DB) return J({ error: 'off' }, 503);
  await ready(env);
  const cron = env.CRON_KEY && request.headers.get('x-cron-key') === env.CRON_KEY;
  if (!cron) { if (!sameSite(request)) return J({ error: 'Bad request.' }, 400); const u = await currentUser(request, env); if (!u?.admin) return J({ error: 'Administrator only.' }, 403); }
  const origin = new URL(request.url).origin;
  const cs = (await env.DB.prepare(`SELECT * FROM users WHERE role = 'consultant' AND status = 'active' ORDER BY id`).all()).results;
  let sent = 0, quiet = 0, failed = 0;
  for (const c of cs.slice(0, 45)) {                                  // stays inside the per-request limit on outgoing calls
    const m = await build(env, origin, c); if (!m) { quiet++; continue; }
    if (env.DEV_CODES === '1') { sent++; continue; }                  // local testing: build, do not send
    (await sendMail(env, c.email, m.subject, m.text, m.html)) ? sent++ : failed++;
  }
  return J({ ok: true, sent, quiet, failed, consultants: cs.length });
}
