// Shared helpers for the content layer: the Python pipeline writes the 3D skeleton of every operation
// (views, structures, actions) to content/_base/procedures.base.json; the words (titles, text, case vignettes,
// questions, sources) live in content/procedures/**/*.md, which anyone can edit. content-build.mjs merges them.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const BASE = path.join(ROOT, 'content', '_base', 'procedures.base.json');
export const DIR = path.join(ROOT, 'content', 'procedures');

export const slug = (s) => String(s || 'other').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
export const fileFor = (key, p) => path.join(DIR, slug(p.group || 'other'), `${key}.md`);

export function walk(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(dir, e.name)) : e.name.endsWith('.md') ? [path.join(dir, e.name)] : []));
}

/** parse one procedure file into { meta, steps: Map(id -> {title, body, lead, pearl, ask}), sources } */
export function parse(text, file = '') {
  text = text.replace(/\r\n?/g, '\n'); // Windows line endings (git autocrlf, Notepad)
  const errors = [];
  let meta = {}; let rest = text;
  const fm = text.match(/^---\n([\s\S]*?)\n---\n?/);
  if (fm) {
    rest = text.slice(fm[0].length);
    for (const line of fm[1].split('\n')) {
      const m = line.match(/^([a-zA-Z]+):\s?(.*)$/);
      if (m) meta[m[1]] = m[2].trim();
    }
  }
  const steps = new Map(); let sources = null;
  const blocks = rest.split(/^(?=## )/m);
  for (const b of blocks) {
    const head = b.match(/^## \[([^\]]+)\]\s*(.*)\n/);
    if (head) {
      const id = head[1].trim(); const title = head[2].trim();
      const parts = b.slice(head[0].length).split(/^### (Case|Question|Pearl)\s*\n/m);
      const st = { title, body: parts[0].trim() };
      for (let i = 1; i < parts.length; i += 2) {
        const kind = parts[i], val = (parts[i + 1] || '').trim();
        if (kind === 'Case') st.lead = val;
        else if (kind === 'Pearl') st.pearl = val;
        else st.ask = parseAsk(val, `${file} [${id}]`, errors);
      }
      if (steps.has(id)) errors.push(`${file}: step [${id}] appears twice`);
      steps.set(id, st);
    } else if (/^## Sources/.test(b)) {
      sources = [...b.matchAll(/^- \[(.+?)\]\((\S+?)\)\s*$/gm)].map((m) => ({ title: m[1], url: m[2] }));
    }
  }
  return { meta, steps, sources, errors };
}

function parseAsk(text, where, errors) {
  const q = text.match(/^\*\*Q:\*\*\s*([\s\S]*?)\n(?=- \[)/);
  if (!q) { errors.push(`${where}: question must start with **Q:**`); return null; }
  const choices = [];
  for (const m of text.slice(q[0].length).matchAll(/^- \[( |x)\] (.+)\n?((?:  > .*\n?)*)/gm)) {
    const why = m[3].split('\n').map((l) => l.replace(/^  > ?/, '')).join(' ').trim();
    choices.push({ text: m[2].trim(), correct: m[1] === 'x', why });
  }
  const nRight = choices.filter((c) => c.correct).length;
  if (choices.length < 2) errors.push(`${where}: a question needs at least two answers`);
  if (nRight !== 1) errors.push(`${where}: a question needs exactly one answer marked [x] (found ${nRight})`);
  return { question: q[1].trim(), choices };
}

export function writeAsk(a) {
  let s = `**Q:** ${a.question}\n`;
  for (const c of a.choices) {
    s += `- [${c.correct ? 'x' : ' '}] ${c.text}\n`;
    if (c.why) s += `  > ${c.why}\n`;
  }
  return s;
}
