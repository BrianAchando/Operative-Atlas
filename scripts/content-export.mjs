// Write the text of every operation from the base JSON into editable Markdown files.
// Default: only operations (and steps) that have no file/entry yet, so edits are never overwritten.
// --force rewrites every file from the base (use once, at the start).
import fs from 'node:fs';
import path from 'node:path';
import TurndownService from 'turndown';
import { gfm } from 'turndown-plugin-gfm';
import { BASE, DIR, fileFor, parse, writeAsk } from './content-lib.mjs';

const force = process.argv.includes('--force');
const td = new TurndownService({ headingStyle: 'atx', bulletListMarker: '-', emDelimiter: '*', strongDelimiter: '**' });
td.use(gfm);
// HTML the Markdown cannot express (the coloured boxes, chains, evidence notes, tags, complex tables) is kept as HTML
td.keep((n) => {
  if (n.nodeName === 'TABLE') return [...n.querySelectorAll('th,td')].some((c) => c.getAttribute('colspan') || c.getAttribute('rowspan') || c.getAttribute('style'))
    || [...n.querySelectorAll('td,th')].some((c) => /<(ul|ol|p|br|div)\b/i.test(c.innerHTML));
  if (n.nodeName === 'DIV' || n.nodeName === 'SUB') return true;
  if (n.nodeName === 'SPAN' && n.getAttribute('class')) return true;
  if (n.nodeName === 'P' && n.getAttribute('class')) return true;
  return false;
});
// stray "<" in text (e.g. "<30%") is escaped first; blocks Markdown cannot hold are swapped for placeholders and restored as HTML
const KEEP_BLOCK = [/<figure[^>]*>[\s\S]*?<\/figure>/g, /<p class="(?!evidence)[^"]*">[\s\S]*?<\/p>/g, /<div class="[^"]*">[\s\S]*?<\/div>/g, /<table[^>]*>[\s\S]*?<\/table>/g];
const complexTable = (t) => /colspan|rowspan|style=/.test(t) || /<t[dh][^>]*>[^]*?<(ul|ol|p|br|div)\b/i.test(t);
const md = (html) => {
  if (!html) return '';
  html = html.replace(/<(?![a-zA-Z\/!])/g, '&lt;');
  // evidence notes become "> **Evidence:** ..." and causal chains "Chain: A → **B** → C" (bold = the key link)
  html = html.replace(/<p class="evidence">([\s\S]*?)<\/p>/g, '<blockquote><p>$1</p></blockquote>');
  html = html.replace(/<div class="chain">([\s\S]*?)<\/div>/g, (m, inner) => {
    const parts = [...inner.matchAll(/<span( class="hot")?>([\s\S]*?)<\/span>/g)].map((x) => (x[1] ? `<b>${x[2]}</b>` : x[2]));
    return parts.length ? `<p>Chain: ${parts.join(' → ')}</p>` : m;
  });
  const kept = [];
  for (const re of KEEP_BLOCK) html = html.replace(re, (m) => (m.startsWith('<table') && !complexTable(m) ? m : (kept.push(m), `<p>KEEPBLOCK${kept.length - 1}X</p>`)));
  let out = td.turndown(html).replace(/\n{3,}/g, '\n\n').trim();
  out = out.replace(/KEEPBLOCK(\d+)X/g, (_, i) => kept[+i]);
  return out;
};

const base = JSON.parse(fs.readFileSync(BASE, 'utf8'));
let wrote = 0, added = 0;
for (const [key, p] of Object.entries(base)) {
  const file = fileFor(key, p);
  const exists = fs.existsSync(file);
  const have = exists && !force ? parse(fs.readFileSync(file, 'utf8'), file).steps : new Map();
  const block = (s) => {
    let b = `## [${s.id}] ${s.title}\n\n${md(s.body)}\n`;
    if (s.lead) b += `\n### Case\n\n${md(s.lead)}\n`;
    if (s.pearl) b += `\n### Pearl\n\n${md(s.pearl)}\n`;
    if (s.ask) b += `\n### Question\n\n${writeAsk(s.ask)}`;
    return b + '\n';
  };
  if (exists && !force) {
    const missing = p.steps.filter((s) => !have.has(s.id));
    if (!missing.length) continue;
    let text = fs.readFileSync(file, 'utf8');
    const at = text.search(/^## Sources/m);
    const ins = missing.map(block).join('');
    text = at >= 0 ? text.slice(0, at) + ins + text.slice(at) : text + '\n' + ins;
    fs.writeFileSync(file, text); added += missing.length;
    continue;
  }
  let out = `---\nid: ${key}\noperation: ${p.opName}\napproach: ${p.approach}\nsummary: ${(p.summary || '').replace(/\n/g, ' ')}\n---\n\n`
    + `<!-- Edit the words freely. Keep each "## [step-id]" line as it is: it ties the text to its step in the 3D atlas. -->\n\n`;
  out += p.steps.map(block).join('');
  if (p.sources?.length) out += '## Sources\n\n' + p.sources.map((s) => `- [${String(s.title).replace(/[\[\]]/g, '')}](${s.url})`).join('\n') + '\n';
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, out); wrote++;
}
console.log(`content-export: ${wrote} files written, ${added} new steps added (dir ${path.relative(process.cwd(), DIR)})`);
