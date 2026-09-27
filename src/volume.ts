/**
 * A CT volume in world (RAS millimetre) space.
 *
 * Voxel (i, j, k) lives at world = origin + i*col + j*row + k*slice, where col/row/slice are the affine's columns.
 * Intensities are stored raw (Uint8 for the packed reference CT, Int16 for DICOM) and decoded to Hounsfield units
 * as hu = raw * scale + offset, so one sampler serves both sources.
 */
export interface Volume {
  name: string;
  dims: [number, number, number];
  data: Uint8Array | Int16Array;
  scale: number;
  offset: number;
  /** voxel -> world, column-major 4x4 as three rows [r0, r1, r2] of [a, b, c, t] */
  affine: number[][];
  inverse: number[][];
  /** world-space translation applied at display time (the carina alignment of an uploaded scan) */
  shift: [number, number, number];
  labels?: Uint8Array;          // reference CT only: structure label per voxel
  spacing: number;              // smallest voxel edge, used as the display pixel size
}

export function invertAffine(a: number[][]): number[][] {
  const [r0, r1, r2] = a as [number[], number[], number[]];
  const m = [[r0[0]!, r0[1]!, r0[2]!], [r1[0]!, r1[1]!, r1[2]!], [r2[0]!, r2[1]!, r2[2]!]];
  const det = m[0]![0]! * (m[1]![1]! * m[2]![2]! - m[1]![2]! * m[2]![1]!) - m[0]![1]! * (m[1]![0]! * m[2]![2]! - m[1]![2]! * m[2]![0]!) + m[0]![2]! * (m[1]![0]! * m[2]![1]! - m[1]![1]! * m[2]![0]!);
  const inv = [
    [(m[1]![1]! * m[2]![2]! - m[1]![2]! * m[2]![1]!) / det, (m[0]![2]! * m[2]![1]! - m[0]![1]! * m[2]![2]!) / det, (m[0]![1]! * m[1]![2]! - m[0]![2]! * m[1]![1]!) / det],
    [(m[1]![2]! * m[2]![0]! - m[1]![0]! * m[2]![2]!) / det, (m[0]![0]! * m[2]![2]! - m[0]![2]! * m[2]![0]!) / det, (m[0]![2]! * m[1]![0]! - m[0]![0]! * m[1]![2]!) / det],
    [(m[1]![0]! * m[2]![1]! - m[1]![1]! * m[2]![0]!) / det, (m[0]![1]! * m[2]![0]! - m[0]![0]! * m[2]![1]!) / det, (m[0]![0]! * m[1]![1]! - m[0]![1]! * m[1]![0]!) / det],
  ];
  const t = [r0[3]!, r1[3]!, r2[3]!];
  return inv.map((row) => [row[0]!, row[1]!, row[2]!, -(row[0]! * t[0]! + row[1]! * t[1]! + row[2]! * t[2]!)]);
}

export function worldToVoxel(v: Volume, x: number, y: number, z: number): [number, number, number] {
  const m = v.inverse; x -= v.shift[0]; y -= v.shift[1]; z -= v.shift[2];
  return [m[0]![0]! * x + m[0]![1]! * y + m[0]![2]! * z + m[0]![3]!, m[1]![0]! * x + m[1]![1]! * y + m[1]![2]! * z + m[1]![3]!, m[2]![0]! * x + m[2]![1]! * y + m[2]![2]! * z + m[2]![3]!];
}

export function voxelToWorld(v: Volume, i: number, j: number, k: number): [number, number, number] {
  const m = v.affine;
  return [m[0]![0]! * i + m[0]![1]! * j + m[0]![2]! * k + m[0]![3]! + v.shift[0], m[1]![0]! * i + m[1]![1]! * j + m[1]![2]! * k + m[1]![3]! + v.shift[1], m[2]![0]! * i + m[2]![1]! * j + m[2]![2]! * k + m[2]![3]! + v.shift[2]];
}

/** World-space axis-aligned bounds of the volume (voxel centres), shift included. */
export function worldBounds(v: Volume): { min: [number, number, number]; max: [number, number, number] } {
  const [a, b, c] = v.dims;
  const min: [number, number, number] = [Infinity, Infinity, Infinity], max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const i of [0, a - 1]) for (const j of [0, b - 1]) for (const k of [0, c - 1]) {
    const p = voxelToWorld(v, i, j, k);
    for (let d = 0; d < 3; d++) { min[d] = Math.min(min[d]!, p[d]!); max[d] = Math.max(max[d]!, p[d]!); }
  }
  return { min, max };
}

/** Fetch a gzip payload and inflate it in the browser. */
export async function fetchGunzip(url: string, onProgress?: (f: number) => void): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok || !r.body) throw new Error(`${url}: ${r.status}`);
  const total = Number(r.headers.get('content-length') ?? 0);
  const reader = r.body.getReader(); const chunks: Uint8Array[] = []; let seen = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); seen += value.byteLength; if (total && onProgress) onProgress(seen / total); }
  const raw = new Uint8Array(seen); let o = 0; for (const c of chunks) { raw.set(c, o); o += c.byteLength; }
  if (raw[0] !== 0x1f || raw[1] !== 0x8b) return raw.buffer;
  const ds = new DecompressionStream('gzip') as unknown as ReadableWritablePair<Uint8Array, Uint8Array>;
  return new Response(new Blob([raw]).stream().pipeThrough(ds)).arrayBuffer();
}
