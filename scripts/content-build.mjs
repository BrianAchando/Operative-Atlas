// Merge the 3D skeleton (content/_base/procedures.base.json) with the editable text (content/procedures/**/*.md)
// into public/data/procedures.json, and write the search index. Fails (exit 1) on content errors, so CI catches them.
import fs from 'node:fs';
import path from 'node:path';
import { marked } from 'marked';
import { BASE, DIR, ROOT, walk, parse } from './content-lib.mjs';

marked.setOptions({ gfm: true, breaks: false });
const html = (m) => {
  let h = marked.parse(m || '').trim();
  h = h.replace(/<table>/g, '<table class="mini">').replace(/<a href="#/g, '<a class="link" href="#');
  h = h.replace(/<blockquote>\s*<p>([\s\S]*?)<\/p>\s*<\/blockquote>/g, (m, inner) => `<p class="evidence">${inner.replace(/<\/?strong>/g, (t) => t.replace('strong', 'b'))}</p>`);
  const splitOutside = (t) => { const out = []; let depth = 0, cur = '';
    for (let i = 0; i < t.length; i++) {
      if (t.startsWith('<strong>', i)) depth++; else if (t.startsWith('</strong>', i)) depth--;
      if (depth === 0 && t.startsWith(' → ', i)) { out.push(cur); cur = ''; i += 2; continue; }
      cur += t[i];
    }
    out.push(cur); return out; };
  h = h.replace(/<p>Chain: ([\s\S]*?)<\/p>/g, (m, inner) => '<div class="chain">' + splitOutside(inner).map((x) => {
    const hot = /^<strong>([\s\S]*)<\/strong>$/.exec(x.trim());
    return hot ? `<span class="hot">${hot[1]}</span>` : `<span>${x.trim()}</span>`;
  }).join('<i>→</i>') + '</div>');
  return h;
};
const base = JSON.parse(fs.readFileSync(BASE, 'utf8'));
const errors = [];
const files = walk(DIR);
const seen = new Set();
for (const f of files) {
  const { meta, steps, sources, errors: e } = parse(fs.readFileSync(f, 'utf8'), path.relative(ROOT, f));
  errors.push(...e);
  const p = base[meta.id];
  if (!p) { errors.push(`${path.relative(ROOT, f)}: no operation with id "${meta.id}" in the 3D skeleton`); continue; }
  seen.add(meta.id);
  if (meta.operation) p.opName = meta.operation;
  if (meta.approach) p.approach = meta.approach;
  if (meta.summary) p.summary = meta.summary;
  const byId = new Map(p.steps.map((s) => [s.id, s]));
  for (const [id, t] of steps) {
    const s = byId.get(id);
    if (!s) { errors.push(`${path.relative(ROOT, f)}: step [${id}] is not in this operation's skeleton (renamed or deleted?)`); continue; }
    s.title = t.title; s.body = html(t.body);
    if (t.lead !== undefined) s.lead = html(t.lead); else delete s.lead;
    if (t.pearl !== undefined) s.pearl = marked.parseInline(t.pearl); else delete s.pearl;
    if (t.ask) s.ask = t.ask; else if (s.ask && !t.ask) delete s.ask;
    if (!s.ask) delete s.askAfter;
  }
  if (sources) p.sources = sources;
}
// the operations' shared names: keep every approach of one operation under one menu name
const opNames = {};
for (const p of Object.values(base)) opNames[p.op] ??= p.opName;
for (const p of Object.values(base)) p.opName = opNames[p.op];
for (const id of Object.keys(base)) if (!seen.has(id)) console.warn(`content-build: no text file for ${id} (using the skeleton's own text)`);
if (errors.length) { console.error('content-build: errors\n  ' + errors.join('\n  ')); process.exit(1); }
const out = path.join(ROOT, 'public', 'data');
fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(path.join(out, 'procedures.json'), JSON.stringify(base));
// search index: one row per step
const strip = (h) => String(h || '').replace(/<[^>]+>/g, ' ').replace(/&[a-z#0-9]+;/gi, ' ').replace(/\s+/g, ' ').trim();
const idx = [];
for (const [key, p] of Object.entries(base)) p.steps.forEach((s, i) => {
  idx.push({ p: key, i, o: p.opName, a: p.approach, g: p.group || '', ph: s.phase, t: s.title, x: strip([s.body, s.lead, s.ask?.question, ...(s.ask?.choices ?? []).map((c) => c.text + ' ' + c.why)].join(' ')).slice(0, 4000) });
});
fs.writeFileSync(path.join(out, 'search.json'), JSON.stringify(idx));
console.log(`content-build: ${files.length} text files, ${Object.keys(base).length} operations, ${idx.length} steps -> public/data/procedures.json, search.json`);
