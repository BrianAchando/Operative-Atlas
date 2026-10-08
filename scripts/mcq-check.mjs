// Reports multiple-choice questions whose correct answer is the strictly longest option (a test-wise cue).
// Usage: node scripts/mcq-check.mjs [--list]
import fs from 'node:fs'; import path from 'node:path'; import { fileURLToPath } from 'node:url';
import { parse } from './content-lib.mjs';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dir = path.join(ROOT, 'content', 'procedures'); const bad = []; let n = 0;
for (const sub of fs.readdirSync(dir)) for (const f of fs.readdirSync(path.join(dir, sub))) {
  if (!f.endsWith('.md')) continue;
  const rel = `${sub}/${f}`; const { steps } = parse(fs.readFileSync(path.join(dir, sub, f), 'utf8'), rel);
  for (const [id, st] of steps) {
    if (!st.ask) continue; n++;
    const L = st.ask.choices.map((c) => c.text.length); const r = st.ask.choices.findIndex((c) => c.correct);
    if (L[r] > Math.max(...L.filter((_, i) => i !== r))) bad.push(`${rel} [${id}] ${L[r]} vs ${Math.max(...L.filter((_, i) => i !== r))}`);
  }
}
if (process.argv.includes('--list')) console.log(bad.join('\n'));
console.log(`${bad.length}/${n} questions have the correct answer as the longest option`);
process.exitCode = 0;
