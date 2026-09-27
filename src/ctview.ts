import { type Volume, worldBounds, worldToVoxel } from './volume.ts';

export type Plane = 'axial' | 'coronal' | 'sagittal';
export type Vec3 = [number, number, number];

export interface Overlay {
  /** label id -> [r, g, b, fillAlpha]; outlined, and filled with the given alpha */
  tint: Map<number, [number, number, number, number]>;
  /** label ids drawn with a dashed danger outline */
  danger: Set<number>;
  showAll: boolean;
  allColours: Map<number, [number, number, number]>;
}

export interface Window { width: number; level: number }
export const WINDOWS: Record<string, Window & { label: string }> = {
  mediastinum: { label: 'Mediastinum', width: 400, level: 40 },
  lung: { label: 'Lung', width: 1500, level: -600 },
  bone: { label: 'Bone', width: 1800, level: 400 },
  vascular: { label: 'Angio', width: 700, level: 150 },
};

/**
 * One CT plane drawn into a canvas in radiological convention: axial and coronal put the patient's right on the
 * viewer's left; sagittal puts anterior on the left. Every pixel is sampled through the volume's inverse affine,
 * so a DICOM series and the reference CT go through the same code.
 */
export class CTView {
  readonly canvas = document.createElement('canvas');
  private ctx = this.canvas.getContext('2d')!;
  private img: ImageData | null = null;
  /** world coordinates of the image corners: top-left, top-right, bottom-left, bottom-right */
  corners: [Vec3, Vec3, Vec3, Vec3] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
  private px = 1;
  private cols = 1; private rows = 1;
  private b = { min: [0, 0, 0] as Vec3, max: [0, 0, 0] as Vec3 };
  version = 0;

  constructor(public plane: Plane, public vol: Volume, public onFocus: (p: Vec3, from: CTView) => void) {
    this.canvas.className = 'ct-canvas';
    this.canvas.addEventListener('pointerdown', (e) => this.pointer(e));
    this.canvas.addEventListener('pointermove', (e) => { if (e.buttons & 1) this.pointer(e); });
    this.canvas.addEventListener('wheel', (e) => { e.preventDefault(); this.scroll(e.deltaY > 0 ? -1 : 1, e.shiftKey ? 5 : 1); }, { passive: false });
    this.setVolume(vol);
  }

  focus: Vec3 = [0, 0, 0];
  window: Window = { ...WINDOWS['mediastinum']! };
  overlay: Overlay | null = null;
  crosshair = true;

  setVolume(v: Volume): void {
    this.vol = v;
    this.b = worldBounds(v);
    this.px = v.spacing;
    const ext = (d: number) => this.b.max[d]! - this.b.min[d]!;
    const [cu, cv] = this.axes();
    this.cols = Math.max(1, Math.round(ext(cu) / this.px) + 1);
    this.rows = Math.max(1, Math.round(ext(cv) / this.px) + 1);
    this.canvas.width = this.cols; this.canvas.height = this.rows;
    this.img = this.ctx.createImageData(this.cols, this.rows);
  }

  /** in-plane world axes (column axis, row axis) and the through-plane axis */
  axes(): [number, number, number] {
    return this.plane === 'axial' ? [0, 1, 2] : this.plane === 'coronal' ? [0, 2, 1] : [1, 2, 0];
  }

  /** world point of pixel (c, r): columns and rows both run from the max end of their axis */
  private worldAt(c: number, r: number): Vec3 {
    const [cu, cv, n] = this.axes();
    const p: Vec3 = [0, 0, 0];
    p[cu] = this.b.max[cu]! - c * this.px;
    p[cv] = this.b.max[cv]! - r * this.px;
    p[n] = this.focus[n]!;
    return p;
  }

  sliceRange(): [number, number] { const n = this.axes()[2]; return [this.b.min[n]!, this.b.max[n]!]; }

  scroll(dir: number, n = 1): void {
    const ax = this.axes()[2];
    const [lo, hi] = this.sliceRange();
    const p: Vec3 = [...this.focus];
    p[ax] = Math.min(hi, Math.max(lo, p[ax]! + dir * n * this.throughStep()));
    this.onFocus(p, this);
  }

  /** the volume's own spacing along the through-plane axis, so one wheel tick is one acquired slice */
  private throughStep(): number {
    const ax = this.axes()[2];
    const a = this.vol.affine; let best = this.px;
    for (let d = 0; d < 3; d++) { const s = Math.abs(a[ax]![d]!); if (s > 1e-3) best = Math.max(0.4, s); }
    return best;
  }

  private pointer(e: PointerEvent): void {
    const r = this.canvas.getBoundingClientRect();
    // the canvas is letterboxed with object-fit: contain; undo that to get the image pixel
    const s = Math.min(r.width / this.cols, r.height / this.rows);
    const ox = (r.width - this.cols * s) / 2, oy = (r.height - this.rows * s) / 2;
    const c = (e.clientX - r.left - ox) / s, row = (e.clientY - r.top - oy) / s;
    if (c < 0 || row < 0 || c > this.cols || row > this.rows) return;
    this.onFocus(this.worldAt(c, row), this);
  }

  /** label id under a world point (reference CT only) */
  labelAt(p: Vec3): number {
    const v = this.vol; if (!v.labels) return 0;
    const [i, j, k] = worldToVoxel(v, p[0], p[1], p[2]).map(Math.round) as Vec3;
    if (i < 0 || j < 0 || k < 0 || i >= v.dims[0] || j >= v.dims[1] || k >= v.dims[2]) return 0;
    return v.labels[i + v.dims[0] * (j + v.dims[1] * k)]!;
  }

  render(): void {
    const v = this.vol, img = this.img!, out = img.data;
    const [nx, ny, nz] = v.dims;
    const lo = this.window.level - this.window.width / 2, inv = 255 / this.window.width;
    const labels = v.labels, ov = this.overlay;
    const lab = labels && ov ? new Uint8Array(this.cols * this.rows) : null;
    const m = v.inverse;
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const w = this.worldAt(c, r);
        const x = w[0] - v.shift[0], y = w[1] - v.shift[1], z = w[2] - v.shift[2];
        const i = Math.round(m[0]![0]! * x + m[0]![1]! * y + m[0]![2]! * z + m[0]![3]!);
        const j = Math.round(m[1]![0]! * x + m[1]![1]! * y + m[1]![2]! * z + m[1]![3]!);
        const k = Math.round(m[2]![0]! * x + m[2]![1]! * y + m[2]![2]! * z + m[2]![3]!);
        const o = (r * this.cols + c) * 4;
        if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) { out[o] = out[o + 1] = out[o + 2] = 0; out[o + 3] = 255; continue; }
        const idx = i + nx * (j + ny * k);
        const hu = v.data[idx]! * v.scale + v.offset;
        const g = Math.max(0, Math.min(255, (hu - lo) * inv));
        out[o] = out[o + 1] = out[o + 2] = g; out[o + 3] = 255;
        if (lab) lab[r * this.cols + c] = labels![idx]!;
      }
    }
    if (lab && ov) this.paint(lab, out, ov);
    this.ctx.putImageData(img, 0, 0);
    if (this.crosshair) this.drawCross();
    this.updateCorners();
    this.version++;
  }

  private paint(lab: Uint8Array, out: Uint8ClampedArray, ov: Overlay): void {
    const W = this.cols, H = this.rows;
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const id = lab[r * W + c]!; const o = (r * W + c) * 4;
      if (!id) continue;
      const t = ov.tint.get(id);
      if (t) {
        out[o] = out[o]! * (1 - t[3]) + t[0] * t[3]; out[o + 1] = out[o + 1]! * (1 - t[3]) + t[1] * t[3]; out[o + 2] = out[o + 2]! * (1 - t[3]) + t[2] * t[3];
      } else if (ov.showAll) {
        const a = ov.allColours.get(id); if (a) { out[o] = out[o]! * 0.6 + a[0] * 0.4; out[o + 1] = out[o + 1]! * 0.6 + a[1] * 0.4; out[o + 2] = out[o + 2]! * 0.6 + a[2] * 0.4; }
      }
      // outline: a pixel whose right or lower neighbour belongs to another label
      const edge = (c + 1 < W && lab[r * W + c + 1] !== id) || (r + 1 < H && lab[(r + 1) * W + c] !== id) || (c > 0 && lab[r * W + c - 1] !== id) || (r > 0 && lab[(r - 1) * W + c] !== id);
      if (!edge) continue;
      if (ov.danger.has(id) && ((r + c) >> 1) % 2 === 0) { out[o] = 255; out[o + 1] = 70; out[o + 2] = 80; }
      else if (t) { out[o] = t[0]; out[o + 1] = t[1]; out[o + 2] = t[2]; }
    }
  }

  private drawCross(): void {
    const [cu, cv] = this.axes();
    const c = (this.b.max[cu]! - this.focus[cu]!) / this.px, r = (this.b.max[cv]! - this.focus[cv]!) / this.px;
    const g = this.ctx; g.save(); g.strokeStyle = 'rgba(70, 194, 199, 0.55)'; g.lineWidth = Math.max(0.6, this.cols / 500); g.setLineDash([4, 3]);
    g.beginPath(); g.moveTo(c, 0); g.lineTo(c, r - 6); g.moveTo(c, r + 6); g.lineTo(c, this.rows); g.moveTo(0, r); g.lineTo(c - 6, r); g.moveTo(c + 6, r); g.lineTo(this.cols, r); g.stroke(); g.restore();
  }

  private updateCorners(): void {
    this.corners = [this.worldAt(0, 0), this.worldAt(this.cols - 1, 0), this.worldAt(0, this.rows - 1), this.worldAt(this.cols - 1, this.rows - 1)];
  }

  /** orientation letters for the four edges: [top, right, bottom, left] */
  edgeLabels(): [string, string, string, string] {
    if (this.plane === 'axial') return ['A', 'L', 'P', 'R'];
    if (this.plane === 'coronal') return ['S', 'L', 'I', 'R'];
    return ['S', 'P', 'I', 'A'];
  }

  /** position of the slice along its axis, as the reader reads it (mm, and the slice index of the volume) */
  readout(): string {
    const n = this.axes()[2]; const name = ['x', 'y', 'z'][n]!;
    const [lo, hi] = this.sliceRange(); const step = this.throughStep();
    const idx = Math.round((this.focus[n]! - lo) / step) + 1, total = Math.round((hi - lo) / step) + 1;
    return `${name} ${this.focus[n]!.toFixed(0)} mm · ${idx}/${total}`;
  }
}
