import './style.css';
import { CTView, WINDOWS, type Plane, type Vec3 } from './ctview.ts';
import { Scene3D, type StructureMeta } from './scene.ts';
import { fetchGunzip, invertAffine, type Volume } from './volume.ts';
import { filesFromDrop, loadImages } from './dicom.ts';
import type { Procedure, Step } from './procedure.ts';

const DATA = 'data/';

interface Atlas {
  ct: { file: string; dims: [number, number, number]; affine: number[][]; scale: number; offset: number; spacing: number };
  labels: { file: string; lut: Record<string, string> };
  groups: { id: string; name: string; open?: boolean }[];
  structures: StructureMeta[];
  landmarks: Record<string, Vec3>;
  source: { name: string; licence: string; note: string };
}

const $ = <T extends HTMLElement = HTMLElement>(sel: string) => document.querySelector(sel) as T;
const h = (tag: string, attrs: Record<string, unknown> = {}, ...kids: (Node | string | null | undefined)[]): HTMLElement => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v as EventListener);
    else if (k === 'html') el.innerHTML = String(v);
    else if (v === true) el.setAttribute(k, '');
    else if (v !== false && v != null) el.setAttribute(k, String(v));
  }
  for (const c of kids) if (c != null) el.append(c);
  return el;
};
const hex2rgb = (c: string): [number, number, number] => { const n = parseInt(c.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };

// ------------------------------------------------------------------ state
const state = {
  mode: 'procedure' as 'explore' | 'procedure',
  approach: 'lul-anterior' as string,
  step: 0,
  answered: new Set<string>(),
  acted: new Set<string>(),
  playing: false,
  plane: 'axial' as Plane,
  focus: [0, 0, 0] as Vec3,
  window: 'mediastinum',
  source: 'reference' as 'reference' | 'upload',
  aligning: false,
  selected: null as string | null,
  showAllLabels: false,
};

let atlas: Atlas;
let procedures: Record<string, Procedure> = {};
let refVol: Volume; let upVol: Volume | null = null;
let scene3d: Scene3D;
const views: Record<Plane, CTView> = {} as Record<Plane, CTView>;
const labelOf = new Map<string, number>();           // structure id -> label id
const idOfLabel = new Map<number, string>();

async function boot(): Promise<void> {
  const status = $('#status');
  status.textContent = 'Loading the reference CT…';
  atlas = await (await fetch(DATA + 'atlas.json')).json() as Atlas;
  procedures = await (await fetch(DATA + 'procedures.json')).json() as Record<string, Procedure>;
  for (const [k, v] of Object.entries(atlas.labels.lut)) { labelOf.set(v, Number(k)); idOfLabel.set(Number(k), v); }
  const [ctBuf, labBuf] = await Promise.all([
    fetchGunzip(DATA + atlas.ct.file, (f) => { status.textContent = `Loading the reference CT… ${Math.round(f * 100)}%`; }),
    fetchGunzip(DATA + atlas.labels.file),
  ]);
  const c = atlas.ct;
  refVol = { name: 'Reference CT', dims: c.dims, data: new Uint8Array(ctBuf), scale: c.scale, offset: c.offset, affine: c.affine, inverse: invertAffine(c.affine), shift: [0, 0, 0], spacing: c.spacing, labels: new Uint8Array(labBuf) };

  // ---- CT views
  const onFocus = (p: Vec3, from: CTView | null) => {
    if (state.aligning && from && state.source === 'upload' && upVol) { finishAlign(p); return; }
    setFocus(p);
    if (from && state.mode === 'explore') { const id = idOfLabel.get(from.labelAt(p)); if (id && state.source === 'reference') select(id, false); }
  };
  for (const p of ['axial', 'coronal', 'sagittal'] as Plane[]) views[p] = new CTView(p, refVol, onFocus);
  buildCTPanel();

  // ---- 3D
  scene3d = new Scene3D($('#view3d'));
  (window as unknown as { hilum: unknown }).hilum = { get scene() { return scene3d; }, views };
  status.textContent = 'Loading the 3D anatomy…';
  await scene3d.load(DATA, atlas.structures, (f) => { status.textContent = `Loading the 3D anatomy… ${Math.round(f * 100)}%`; });
  scene3d.planeSource = () => { const v = views[state.plane]; return { canvas: v.canvas, corners: v.corners, version: v.version, visible: planeIn3d && (state.source === 'reference' || !!upVol?.shift.some((x) => x !== 0)) }; };
  scene3d.onPick = (id, p) => { if (p) setFocus(p); if (id) select(id, false); };
  scene3d.onHover = (id) => { $('#hover').textContent = id ? (scene3d.items.get(id)?.meta.name ?? '') : ''; };
  status.textContent = '';
  $('#status').hidden = true;

  buildTopbar();
  wireUpload();
  window.addEventListener('keydown', key);
  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.get('mode') === 'explore') state.mode = 'explore';
  if (hash.get('approach')) { const a = hash.get('approach')!; state.approach = procedures[a] ? a : procedures[`lul-${a}`] ? `lul-${a}` : state.approach; }
  if (hash.get('step')) state.step = Number(hash.get('step'));
  setFocus(atlas.landmarks['carina'] ?? [0, 0, 0]);
  if (state.mode === 'procedure') goStep(state.step); else setMode('explore');
}

/** the CT plane drawn in 3D: on while exploring, off during the operation unless the reader turns it on */
let planeIn3d = false;

// ------------------------------------------------------------------ focus and CT
function setFocus(p: Vec3): void {
  state.focus = [...p];
  for (const v of Object.values(views)) { v.focus = [...p]; }
  scene3d?.setFocus(p);
  renderCT();
}

function overlayFor(): { tint: Map<number, [number, number, number, number]>; danger: Set<number>; showAll: boolean; allColours: Map<number, [number, number, number]> } {
  const tint = new Map<number, [number, number, number, number]>(); const danger = new Set<number>();
  const st = currentStep();
  const hi = state.mode === 'procedure' && st ? st.highlight ?? [] : state.selected ? [state.selected] : [];
  const dg = state.mode === 'procedure' && st ? st.danger ?? [] : [];
  for (const id of hi) { const l = labelOf.get(id); if (l) tint.set(l, [70, 194, 199, 0.32]); }
  for (const id of dg) { const l = labelOf.get(id); if (l) { danger.add(l); if (!tint.has(l)) tint.set(l, [255, 70, 80, 0.16]); } }
  const allColours = new Map<number, [number, number, number]>();
  for (const s of atlas.structures) { const l = labelOf.get(s.id); if (l) allColours.set(l, hex2rgb(s.colour)); }
  return { tint, danger, showAll: state.showAllLabels, allColours };
}

function renderCT(): void {
  const vol = state.source === 'upload' && upVol ? upVol : refVol;
  const w = WINDOWS[state.window]!;
  for (const [p, v] of Object.entries(views) as [Plane, CTView][]) {
    if (v.vol !== vol) v.setVolume(vol);
    v.window = { width: w.width, level: w.level };
    v.overlay = vol === refVol ? overlayFor() : null;
    v.render();
    const r = document.querySelector(`[data-readout="${p}"]`); if (r) r.textContent = v.readout();
  }
  const main = views[state.plane];
  const [t, rt, b, l] = main.edgeLabels();
  $('#edge-t').textContent = t; $('#edge-r').textContent = rt; $('#edge-b').textContent = b; $('#edge-l').textContent = l;
  const lab = vol === refVol ? idOfLabel.get(main.labelAt(state.focus)) : undefined;
  $('#ct-under').textContent = lab ? scene3d?.items.get(lab)?.meta.name ?? '' : '';
}

function buildCTPanel(): void {
  const host = $('#ct');
  const tabs = h('div', { class: 'seg', role: 'tablist', 'aria-label': 'CT plane' });
  for (const p of ['axial', 'coronal', 'sagittal'] as Plane[]) tabs.append(h('button', { 'data-plane': p, role: 'tab', onclick: () => { state.plane = p; mountViews(); render(); } }, p[0]!.toUpperCase() + p.slice(1)));
  const wins = h('div', { class: 'seg small', 'aria-label': 'Window' });
  for (const [k, w] of Object.entries(WINDOWS)) wins.append(h('button', { 'data-window': k, onclick: () => { state.window = k; render(); } }, w.label));
  const labelsBtn = h('button', { class: 'chip-toggle', id: 'all-labels', onclick: () => { state.showAllLabels = !state.showAllLabels; render(); } }, 'All labels');
  const planeBtn = h('button', { class: 'chip-toggle on', id: 'plane3d', onclick: () => { planeIn3d = !planeIn3d; render(); } }, 'Slice in 3D');
  const main = h('div', { class: 'ct-main', id: 'ct-main' },
    h('span', { class: 'edge t', id: 'edge-t' }), h('span', { class: 'edge r', id: 'edge-r' }), h('span', { class: 'edge b', id: 'edge-b' }), h('span', { class: 'edge l', id: 'edge-l' }),
    h('div', { class: 'ct-hud' }, h('span', { id: 'ct-src' }), h('span', { 'data-readout': 'main', id: 'ct-readout' }), h('span', { id: 'ct-under', class: 'under' })),
    h('div', { class: 'drop-hint', id: 'drop-hint' }, 'Drop a DICOM folder or a .nii file'),
  );
  const minis = h('div', { class: 'ct-minis', id: 'ct-minis' });
  host.append(h('div', { class: 'ct-bar' }, tabs, wins), main, h('div', { class: 'ct-bar lower' }, labelsBtn, planeBtn, h('span', { class: 'hint' }, 'Scroll = slice · click = move crosshair')), minis);
  mountViews();
}

function mountViews(): void {
  const main = $('#ct-main'); const minis = $('#ct-minis');
  main.querySelectorAll('canvas').forEach((c) => c.remove());
  minis.replaceChildren();
  main.prepend(views[state.plane].canvas);
  for (const p of ['axial', 'coronal', 'sagittal'] as Plane[]) if (p !== state.plane) {
    minis.append(h('button', { class: 'mini', 'aria-label': `Show ${p}`, onclick: () => { state.plane = p; mountViews(); render(); } }, views[p].canvas, h('span', { class: 'mini-lab' }, p), h('span', { class: 'mini-ro', 'data-readout': p })));
  }
}

// ------------------------------------------------------------------ top bar
function buildTopbar(): void {
  const bar = $('#topbar');
  // operations (each with its approaches), grouped in one menu, and the free anatomy explorer
  const opSel = h('select', { id: 'op-select', class: 'op-select', 'aria-label': 'Operation', onchange: (e: Event) => pickOp((e.target as HTMLSelectElement).value) }) as HTMLSelectElement;
  const groups = new Map<string, Map<string, string>>();
  for (const p of Object.values(procedures)) { const g = p.group ?? 'Operations'; if (!groups.has(g)) groups.set(g, new Map()); groups.get(g)!.set(p.op, p.opName); }
  for (const [g, ops] of groups) { const og = h('optgroup', { label: g }); for (const [op, name] of ops) og.append(h('option', { value: op }, name)); opSel.append(og); }
  const modes = h('div', { class: 'seg', 'aria-label': 'Mode' },
    h('button', { 'data-mode': 'procedure', onclick: () => pickOp(opSel.value) }, 'Operate'),
    h('button', { 'data-mode': 'explore', onclick: () => setMode('explore') }, 'Explore anatomy'));
  const approach = h('div', { class: 'seg', id: 'approach', 'aria-label': 'Approach' });
  for (const [k, p] of Object.entries(procedures)) approach.append(h('button', { 'data-approach': k, 'data-of': p.op, onclick: () => { state.approach = k; state.step = 0; state.answered.clear(); state.acted.clear(); goStep(0, true); } }, p.approach));
  const src = h('div', { class: 'seg', 'aria-label': 'CT source' },
    h('button', { 'data-src': 'reference', onclick: () => { state.source = 'reference'; state.aligning = false; render(); } }, 'Reference CT'),
    h('button', { 'data-src': 'upload', id: 'src-upload', onclick: () => { if (upVol) { state.source = 'upload'; render(); } else $('#file').click(); } }, 'Your CT'));
  const up = h('label', { class: 'btn', for: 'file' }, 'Load DICOM…');
  bar.append(h('div', { class: 'brand' }, h('b', {}, 'Operative Atlas'), h('span', {}, 'thoracic')), modes, opSel, approach, h('div', { class: 'spacer' }), src, up);
}

function pickOp(op: string): void {
  if (procedures[state.approach]?.op !== op) { state.approach = Object.keys(procedures).find((k) => procedures[k]!.op === op)!; state.step = 0; state.answered.clear(); state.acted.clear(); }
  if (state.mode !== 'procedure') setMode('procedure'); else goStep(state.step, true);
}

function setMode(m: 'explore' | 'procedure'): void {
  state.mode = m; planeIn3d = m === 'explore';
  scene3d.resetOperative(); state.acted.clear();
  if (m === 'procedure') goStep(state.step, true);
  else { scene3d.highlight.clear(); scene3d.danger.clear(); scene3d.invalidate(); scene3d.setLabels([], () => 'plain'); for (const s of atlas.structures) scene3d.setVisible(s.id, s.visible !== false); scene3d.frame(['lul', 'lll', 'heart', 'aorta'], [-1, 0.35, 0.25]); }
  render();
}

// ------------------------------------------------------------------ procedure
function currentProc(): Procedure | undefined { return procedures[state.approach]; }
function currentStep(): Step | undefined { return currentProc()?.steps[state.step]; }

/** Rebuild the operative scene up to step n: every earlier action is done, retractions applied. */
function goStep(n: number, fly = true): void {
  const proc = currentProc(); if (!proc) return;
  state.step = Math.max(0, Math.min(proc.steps.length - 1, n));
  state.playing = false;
  scene3d.resetOperative();
  // one side's hilum per operation: the other side's structures stay hidden
  for (const s of atlas.structures) scene3d.setVisible(s.id, s.side ? (s.side === proc.side && (s.side === 'right' ? s.sideVisible !== false : s.visible !== false)) : s.visible !== false);
  for (let i = 0; i <= state.step; i++) {
    const st = proc.steps[i]!;
    // show / hide belong to their own step (the skin and ports of the setup step do not stay on)
    if (i === state.step) { for (const id of st.show ?? []) scene3d.setVisible(id, true); for (const id of st.hide ?? []) scene3d.setVisible(id, false); }
    for (const [id, op] of Object.entries(st.opacity ?? {})) scene3d.setOpacity(id, op);
    for (const r of st.retract ? (Array.isArray(st.retract) ? st.retract : [st.retract]) : []) for (const id of r.ids) scene3d.retract(id, r.offset, r.opacity, i === state.step ? 900 : 1);
    if (i < state.step && st.action) { scene3d.applyDone(st.action, portOf(st.action.port)); state.acted.add(st.id); }
    if (st.specimen && i <= state.step) scene3d.moveSpecimen(st.specimen.ids, st.specimen.offset, 0.45, i === state.step ? 1800 : 1);
  }
  for (const k of [...state.acted]) if (proc.steps.findIndex((x) => x.id === k) >= state.step) state.acted.delete(k);
  const st = proc.steps[state.step]!;
  // lymph node stations appear only where the step names them
  const named0 = new Set([...(st.highlight ?? []), ...(st.danger ?? []), ...(st.labels ?? [])]);
  for (const s of atlas.structures) if (s.group === 'nodes' && !named0.has(s.id)) scene3d.setVisible(s.id, false);
  if (st.action) scene3d.ready(st.action, portOf(st.action.port));
  scene3d.spin(!!st.spin);
  scene3d.highlight = new Set(st.highlight ?? []); scene3d.danger = new Set(st.danger ?? []); scene3d.invalidate(4000);
  const named = [...new Set([...(st.highlight ?? []), ...(st.danger ?? []), ...(st.labels ?? [])])];
  scene3d.setLabels(named, (id) => (st.highlight?.includes(id) ? 'hi' : st.danger?.includes(id) ? 'danger' : 'plain'));
  if (fly) {
    if ('eye' in st.view) scene3d.flyTo(st.view.eye, st.view.target);
    else scene3d.frame(st.view.frame, st.view.dir, st.view.pad);
  }
  if (st.ct) {
    const f = typeof st.ct.focus === 'string' ? focusOf(st.ct.focus) : st.ct.focus;
    state.plane = st.ct.plane; if (st.ct.window) state.window = st.ct.window;
    mountViews(); setFocus(f);
  }
  history.replaceState(null, '', `#approach=${state.approach}&step=${state.step}`);
  render();
}

function portOf(id: string): Vec3 { return atlas.landmarks[id] ?? atlas.landmarks['lung-centre'] ?? [-150, 0, -60]; }

function focusOf(id: string): Vec3 {
  const s = scene3d.items.get(id)?.meta; if (!s) return atlas.landmarks[id] ?? state.focus;
  return s.division ? s.division.point : s.centroid;
}

function act(): void {
  const st = currentStep(); if (!st?.action || state.playing) return;
  const id = st.id; state.playing = true; render();
  void scene3d.play(st.action, portOf(st.action.port)).then((ok) => {
    if (currentStep()?.id !== id) return;
    state.playing = false; if (ok) state.acted.add(id); render();
  });
}

/** play the step's action again from the start */
function replay(): void { goStep(state.step, false); setTimeout(act, 60); }

function key(e: KeyboardEvent): void {
  if ((e.target as HTMLElement).tagName === 'INPUT') return;
  if (e.key === 'ArrowRight' && state.mode === 'procedure') { next(); e.preventDefault(); }
  else if (e.key === 'ArrowLeft' && state.mode === 'procedure') { goStep(state.step - 1); e.preventDefault(); }
  else if (e.key === 'ArrowUp') { views[state.plane].scroll(1); e.preventDefault(); }
  else if (e.key === 'ArrowDown') { views[state.plane].scroll(-1); e.preventDefault(); }
  else if (e.key === 's' || e.key === 'S') act();
}

function next(): void {
  const st = currentStep(); if (!st) return;
  if (st.ask && !state.answered.has(st.id)) { $('.ask')?.classList.add('nudge'); setTimeout(() => $('.ask')?.classList.remove('nudge'), 500); return; }
  if (state.playing) return;
  if (st.action && !state.acted.has(st.id)) { act(); return; }
  goStep(state.step + 1);
}

// ------------------------------------------------------------------ explore
function select(id: string, fly = true): void {
  state.selected = id;
  if (state.mode === 'explore') {
    scene3d.highlight = new Set([id]); scene3d.setLabels([id], () => 'hi'); scene3d.invalidate(2500);
    if (fly) { const m = scene3d.items.get(id)!.meta; setFocus(m.centroid); scene3d.frame([id], [-1, 0.4, 0.3], 2.2); }
  }
  render();
}

// ------------------------------------------------------------------ upload
function wireUpload(): void {
  const input = h('input', { type: 'file', id: 'file', multiple: true, hidden: true }) as HTMLInputElement;
  input.setAttribute('webkitdirectory', '');
  document.body.append(input);
  const single = h('input', { type: 'file', id: 'file-one', multiple: true, accept: '.dcm,.nii,.gz,application/dicom', hidden: true }) as HTMLInputElement;
  document.body.append(single);
  input.addEventListener('change', () => { if (input.files?.length) void ingest(Array.from(input.files)); input.value = ''; });
  single.addEventListener('change', () => { if (single.files?.length) void ingest(Array.from(single.files)); single.value = ''; });
  const zone = $('#ct');
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragging'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', (e) => { e.preventDefault(); zone.classList.remove('dragging'); if (e.dataTransfer) void filesFromDrop(e.dataTransfer).then(ingest); });
}

async function ingest(files: File[]): Promise<void> {
  const note = $('#upload-note');
  note.hidden = false; note.className = 'upload-note';
  note.replaceChildren(h('b', {}, 'Reading your scan…'), h('span', {}, ` ${files.length} files, in this browser only`));
  try {
    const rep = await loadImages(files, (f) => { note.lastElementChild!.textContent = ` ${Math.round(f * 100)}% of ${files.length} files`; });
    upVol = rep.volume; state.source = 'upload'; state.aligning = true;
    // start the crosshair in the middle of the scan
    const [a, b, c] = upVol.dims; const m = upVol.affine;
    setFocus([0, 1, 2].map((d) => m[d]![0]! * a / 2 + m[d]![1]! * b / 2 + m[d]![2]! * c / 2 + m[d]![3]!) as Vec3);
    state.plane = 'coronal'; mountViews();
    note.replaceChildren(
      h('b', {}, `${rep.volume.name} · ${rep.slices} slices`),
      h('span', {}, rep.note.join(' ')),
      h('span', { class: 'align-cta' }, 'Align it with the 3D: scroll to the carina and click it.'),
      h('button', { class: 'link', onclick: () => { state.aligning = false; render(); } }, 'Skip'),
    );
    render();
  } catch (e) {
    note.className = 'upload-note err';
    note.replaceChildren(h('b', {}, 'Could not open that scan.'), h('span', {}, (e as Error).message), h('button', { class: 'link', onclick: () => { note.hidden = true; } }, 'Dismiss'));
  }
}

/** The reference template's origin is its carina: shifting the upload so its carina lands there aligns the two. */
function finishAlign(p: Vec3): void {
  if (!upVol) return;
  const c = atlas.landmarks['carina'] ?? [0, 0, 0];
  const raw: Vec3 = [p[0] - upVol.shift[0], p[1] - upVol.shift[1], p[2] - upVol.shift[2]];
  upVol.shift = [c[0] - raw[0], c[1] - raw[1], c[2] - raw[2]];
  state.aligning = false;
  for (const v of Object.values(views)) v.setVolume(upVol);
  setFocus(c);
  const note = $('#upload-note');
  note.replaceChildren(h('b', {}, `${upVol.name} aligned at the carina`), h('span', {}, 'The 3D is the reference anatomy; your scan is placed against it by one point, so expect centimetres of difference away from the hilum.'), h('button', { class: 'link', onclick: () => { state.aligning = true; render(); } }, 'Re-align'), h('button', { class: 'link', onclick: () => { note.hidden = true; } }, 'Hide'));
  render();
}

// ------------------------------------------------------------------ render the panels
function render(): void {
  document.body.classList.toggle('proc', state.mode === 'procedure');
  document.querySelectorAll<HTMLElement>('[data-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset['mode'] === state.mode)));
  document.querySelectorAll<HTMLElement>('[data-approach]').forEach((b) => { b.setAttribute('aria-pressed', String(b.dataset['approach'] === state.approach)); b.hidden = b.dataset['of'] !== procedures[state.approach]?.op; });
  const sel = document.getElementById('op-select') as HTMLSelectElement | null; if (sel) { sel.value = procedures[state.approach]?.op ?? ''; sel.disabled = state.mode !== 'procedure'; }
  document.querySelectorAll<HTMLElement>('[data-plane]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset['plane'] === state.plane)));
  document.querySelectorAll<HTMLElement>('[data-window]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset['window'] === state.window)));
  document.querySelectorAll<HTMLElement>('[data-src]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset['src'] === state.source)));
  $('#approach').hidden = state.mode !== 'procedure';
  $('#all-labels').classList.toggle('on', state.showAllLabels);
  $('#plane3d').classList.toggle('on', planeIn3d);
  $('#ct-src').textContent = state.source === 'upload' && upVol ? upVol.name : 'Reference CTA';
  $('#ct').classList.toggle('aligning', state.aligning && state.source === 'upload');
  renderCT();
  const panel = $('#panel');
  panel.replaceChildren(state.mode === 'procedure' ? procPanel() : explorePanel());
}

function chip(id: string, kind: 'hi' | 'danger' | 'plain'): HTMLElement {
  const m = scene3d.items.get(id)?.meta;
  return h('button', { class: `chip ${kind}`, onclick: () => { const f = focusOf(id); setFocus(f); } }, m?.name ?? id, m?.schematic ? h('i', {}, ' schematic') : null);
}

function procPanel(): HTMLElement {
  const proc = currentProc()!; const st = currentStep()!;
  const dots = h('ol', { class: 'steps', 'aria-label': 'Steps' });
  proc.steps.forEach((s, i) => dots.append(h('li', { class: i === state.step ? 'cur' : i < state.step ? 'done' : '' }, h('button', { title: s.title, onclick: () => goStep(i) }, h('span', { class: 'n' }, String(i + 1)), h('span', { class: 'lbl' }, s.title)))));
  const answered = state.answered.has(st.id);
  let ask: HTMLElement | null = null;
  if (st.ask) {
    ask = h('div', { class: 'ask' + (answered ? ' done' : '') }, h('div', { class: 'q' }, st.ask.question));
    const list = h('div', { class: 'choices' });
    st.ask.choices.forEach((c) => {
      const picked = answered && pickedChoice.get(st.id) === c.text;
      list.append(h('button', { class: 'choice' + (answered ? (c.correct ? ' right' : picked ? ' wrong' : ' dim') : ''), disabled: answered, onclick: () => { pickedChoice.set(st.id, c.text); state.answered.add(st.id); render(); } }, c.text));
    });
    ask.append(list);
    if (answered) { const c = st.ask.choices.find((x) => x.text === pickedChoice.get(st.id))!; const right = st.ask.choices.find((x) => x.correct)!; ask.append(h('p', { class: 'why ' + (c.correct ? 'ok' : 'no') }, h('b', {}, c.correct ? 'Yes. ' : `Not quite: ${right.text}. `), right.why)); }
  }
  const locked = !!st.ask && !answered;
  const needAct = !!st.action && !state.acted.has(st.id);
  const body = h('div', { class: 'body' + (locked ? ' veiled' : '') });
  body.innerHTML = st.body;
  const structs = h('div', { class: 'chips' }, ...(st.highlight ?? []).map((id) => chip(id, 'hi')), ...(st.danger ?? []).map((id) => chip(id, 'danger')));
  const nav = h('div', { class: 'nav' },
    h('button', { class: 'btn ghost', disabled: state.step === 0, onclick: () => goStep(state.step - 1) }, '← Back'),
    st.action && !locked ? (needAct
      ? h('button', { class: 'btn staple', disabled: state.playing, onclick: act }, state.playing ? 'In progress…' : st.action.label)
      : h('button', { class: 'btn ghost', onclick: replay }, '↻ Replay')) : null,
    h('button', { class: 'btn primary', disabled: locked || state.playing || state.step === proc.steps.length - 1, onclick: next }, needAct && !locked ? 'Skip →' : 'Next →'));
  const strip = h('ol', { class: 'seqstrip', 'aria-label': 'Sequence' });
  proc.sequence?.forEach((q, i) => {
    const firstStep = proc.steps.findIndex((x) => x.seq === i);
    const done = proc.steps.some((x, j) => x.seq === i && (j < state.step || (j === state.step && state.acted.has(x.id))));
    strip.append(h('li', { class: `sq ${q.kind}${st.seq === i ? ' cur' : ''}${done ? ' done' : ''}` }, h('button', { disabled: firstStep < 0, onclick: () => goStep(firstStep) }, q.label)));
  });
  return h('div', { class: 'proc' },
    proc.sequence?.length ? strip : null,
    h('div', { class: 'proc-head' }, h('div', { class: 'eyebrow' }, `${proc.approach} · step ${state.step + 1} of ${proc.steps.length} · ${st.phase}`), h('h2', {}, st.title)),
    ask,
    locked ? h('p', { class: 'veil-note' }, 'Answer to reveal the step.') : null,
    body,
    !locked && (st.highlight?.length || st.danger?.length) ? h('div', { class: 'legend' }, h('span', { class: 'k hi' }, 'Working on'), h('span', { class: 'k danger' }, 'Protect')) : null,
    !locked ? structs : null,
    st.pearl && !locked ? h('p', { class: 'pearl' }, st.pearl) : null,
    nav,
    h('details', { class: 'outline' }, h('summary', {}, 'All steps'), dots),
    h('p', { class: 'foot' }, 'Teaching model on one reference CT. Not for planning an operation on a patient.'),
  );
}
const pickedChoice = new Map<string, string>();

function explorePanel(): HTMLElement {
  const groups = h('div', { class: 'tree' });
  for (const g of atlas.groups) {
    const members = atlas.structures.filter((s) => s.group === g.id);
    if (!members.length) continue;
    const list = h('div', { class: 'tree-items' });
    for (const s of members) {
      const it = scene3d.items.get(s.id)!;
      const cb = h('input', { type: 'checkbox', checked: it.mesh.visible, onchange: (e: Event) => { scene3d.setVisible(s.id, (e.target as HTMLInputElement).checked); } });
      list.append(h('div', { class: 'tree-row' + (state.selected === s.id ? ' sel' : '') }, cb, h('span', { class: 'sw', style: `background:#${it.mat.color.getHexString()}` }), h('button', { class: 'name', onclick: () => select(s.id) }, s.name, s.schematic ? h('i', {}, ' schematic') : null)));
    }
    const allOn = members.every((s) => scene3d.items.get(s.id)!.mesh.visible);
    groups.append(h('details', { open: g.open ?? false }, h('summary', {}, h('span', {}, g.name), h('button', { class: 'link', onclick: (e: Event) => { e.preventDefault(); for (const s of members) scene3d.setVisible(s.id, !allOn); render(); } }, allOn ? 'hide' : 'show')), list));
  }
  const sel = state.selected ? scene3d.items.get(state.selected)?.meta : null;
  return h('div', { class: 'explore' },
    h('div', { class: 'proc-head' }, h('div', { class: 'eyebrow' }, 'Explore'), h('h2', {}, sel ? sel.name : 'Click anything in 3D or on the CT')),
    sel ? h('p', { class: 'body' }, sel.note ?? (sel.schematic ? 'Drawn from landmarks, not segmented from the scan.' : 'Segmented from the reference CT.')) : h('p', { class: 'body' }, 'Every structure is linked: pick it here, in 3D or on the CT, and the other two follow.'),
    groups,
    h('p', { class: 'foot' }, `${atlas.source.name}. ${atlas.source.note}`));
}

boot().catch((e) => { console.error(e); const s = $('#status'); s.hidden = false; s.textContent = `Could not start: ${(e as Error).message}`; });
