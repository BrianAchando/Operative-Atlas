import dicomParser from 'dicom-parser';
import { invertAffine, type Volume } from './volume.ts';

/**
 * DICOM (and NIfTI) read entirely in the browser: nothing is uploaded anywhere. A series is assembled from the
 * slices that share the largest SeriesInstanceUID, ordered along the slice normal, rescaled to Hounsfield units and
 * placed in RAS millimetres (DICOM's patient frame is LPS, so x and y change sign).
 */

const NATIVE = new Set(['1.2.840.10008.1.2', '1.2.840.10008.1.2.1', '1.2.840.10008.1.2.2']);
const SYNTAX_NAMES: Record<string, string> = {
  '1.2.840.10008.1.2.4.50': 'JPEG baseline', '1.2.840.10008.1.2.4.57': 'JPEG lossless', '1.2.840.10008.1.2.4.70': 'JPEG lossless (SV1)',
  '1.2.840.10008.1.2.4.80': 'JPEG-LS lossless', '1.2.840.10008.1.2.4.81': 'JPEG-LS', '1.2.840.10008.1.2.4.90': 'JPEG 2000 lossless',
  '1.2.840.10008.1.2.4.91': 'JPEG 2000', '1.2.840.10008.1.2.5': 'RLE', '1.2.840.10008.1.2.1.99': 'deflated',
};

interface Slice { pos: number[]; iop: number[]; ps: number[]; rows: number; cols: number; pixels: Int16Array; }

export interface LoadReport { volume: Volume; slices: number; skipped: number; series: string; note: string[] }

export async function filesFromDrop(dt: DataTransfer): Promise<File[]> {
  const out: File[] = [];
  const walk = async (entry: FileSystemEntry): Promise<void> => {
    if (entry.isFile) out.push(await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej)));
    else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const e of batch) await walk(e);
      }
    }
  };
  const entries = Array.from(dt.items).map((i) => i.webkitGetAsEntry()).filter((e): e is FileSystemEntry => !!e);
  if (entries.length) { for (const e of entries) await walk(e); return out; }
  return Array.from(dt.files);
}

export async function loadImages(files: File[], onProgress?: (f: number) => void): Promise<LoadReport> {
  const nii = files.find((f) => /\.nii(\.gz)?$/i.test(f.name));
  if (nii) return loadNifti(nii);
  const bySeries = new Map<string, { slices: Slice[]; desc: string }>();
  const syntaxes = new Set<string>();
  let skipped = 0, seen = 0;
  for (const f of files) {
    seen++; if (onProgress && seen % 20 === 0) onProgress(seen / files.length);
    if (f.size < 256) { skipped++; continue; }
    let ds: dicomParser.DataSet;
    const bytes = new Uint8Array(await f.arrayBuffer());
    try { ds = dicomParser.parseDicom(bytes); } catch { skipped++; continue; }
    const ts = ds.string('x00020010') ?? '1.2.840.10008.1.2';
    const pd = ds.elements['x7fe00010'];
    if (!pd) { skipped++; continue; }
    if (!NATIVE.has(ts)) { syntaxes.add(ts); skipped++; continue; }
    const ipp = ds.string('x00200032'), iop = ds.string('x00200037'), ps = ds.string('x00280030');
    if (!ipp || !iop || !ps) { skipped++; continue; }
    const rows = ds.uint16('x00280010')!, cols = ds.uint16('x00280011')!;
    const bits = ds.uint16('x00280100') ?? 16, signed = (ds.uint16('x00280103') ?? 0) === 1;
    if (bits !== 16 || (ds.uint16('x00280002') ?? 1) !== 1) { skipped++; continue; }
    const slope = Number(ds.string('x00281053') ?? 1), icpt = Number(ds.string('x00281052') ?? 0);
    const n = rows * cols;
    const src = signed ? new Int16Array(bytes.buffer.slice(pd.dataOffset, pd.dataOffset + n * 2)) : new Uint16Array(bytes.buffer.slice(pd.dataOffset, pd.dataOffset + n * 2));
    const px = new Int16Array(n);
    for (let i = 0; i < n; i++) px[i] = Math.max(-32768, Math.min(32767, Math.round(src[i]! * slope + icpt)));
    const uid = ds.string('x0020000e') ?? 'series';
    const s = bySeries.get(uid) ?? { slices: [], desc: ds.string('x0008103e') ?? '' };
    s.slices.push({ pos: ipp.split('\\').map(Number), iop: iop.split('\\').map(Number), ps: ps.split('\\').map(Number), rows, cols, pixels: px });
    bySeries.set(uid, s);
  }
  if (!bySeries.size) {
    const names = [...syntaxes].map((s) => SYNTAX_NAMES[s] ?? s);
    throw new Error(names.length
      ? `These images are compressed (${names.join(', ')}), which this viewer cannot decode in the browser yet. Export the series uncompressed ("Explicit VR Little Endian") and drop it again.`
      : 'No CT slices found. Drop the folder of a DICOM series (or a .nii / .nii.gz file).');
  }
  const [uid, best] = [...bySeries.entries()].sort((a, b) => b[1].slices.length - a[1].slices.length)[0]!;
  const sl = best.slices.filter((s) => s.rows === best.slices[0]!.rows && s.cols === best.slices[0]!.cols);
  const iop = sl[0]!.iop;
  const r = iop.slice(0, 3), c = iop.slice(3, 6);
  const nrm = [r[1]! * c[2]! - r[2]! * c[1]!, r[2]! * c[0]! - r[0]! * c[2]!, r[0]! * c[1]! - r[1]! * c[0]!];
  const along = (s: Slice) => s.pos[0]! * nrm[0]! + s.pos[1]! * nrm[1]! + s.pos[2]! * nrm[2]!;
  sl.sort((a, b) => along(a) - along(b));
  // drop duplicate positions (a series sometimes carries a scout or a repeated slice)
  const uniq: Slice[] = [];
  for (const s of sl) if (!uniq.length || Math.abs(along(s) - along(uniq[uniq.length - 1]!)) > 1e-3) uniq.push(s);
  if (uniq.length < 2) throw new Error('This series has a single image. Drop a CT series with many axial slices.');
  const dz = (along(uniq[uniq.length - 1]!) - along(uniq[0]!)) / (uniq.length - 1);
  const { rows, cols } = uniq[0]!; const [rowSp, colSp] = uniq[0]!.ps as [number, number];
  const data = new Int16Array(rows * cols * uniq.length);
  uniq.forEach((s, k) => data.set(s.pixels, k * rows * cols));
  const p0 = uniq[0]!.pos;
  // LPS -> RAS: negate the x and y rows
  const flip = [-1, -1, 1];
  const affine = [0, 1, 2].map((d) => [flip[d]! * r[d]! * colSp, flip[d]! * c[d]! * rowSp, flip[d]! * nrm[d]! * dz, flip[d]! * p0[d]!]);
  const note: string[] = [];
  if (Math.abs(Math.abs(nrm[2]!) - 1) > 0.05) note.push('The series is not axial; the planes are resampled from it.');
  if (syntaxes.size) note.push(`${[...syntaxes].map((s) => SYNTAX_NAMES[s] ?? s).join(', ')} files were skipped.`);
  const volume: Volume = { name: best.desc || 'Uploaded CT', dims: [cols, rows, uniq.length], data, scale: 1, offset: 0, affine, inverse: invertAffine(affine), shift: [0, 0, 0], spacing: Math.min(colSp, rowSp, Math.abs(dz)) };
  return { volume, slices: uniq.length, skipped, series: uid, note };
}

async function loadNifti(f: File): Promise<LoadReport> {
  let buf = await f.arrayBuffer();
  const b0 = new Uint8Array(buf);
  if (b0[0] === 0x1f && b0[1] === 0x8b) {
    const ds = new DecompressionStream('gzip') as unknown as ReadableWritablePair<Uint8Array, Uint8Array>;
    buf = await new Response(new Blob([b0]).stream().pipeThrough(ds)).arrayBuffer();
  }
  const dv = new DataView(buf);
  const le = dv.getInt32(0, true) === 348;
  const i16 = (o: number) => dv.getInt16(o, le), f32 = (o: number) => dv.getFloat32(o, le);
  const dims: [number, number, number] = [i16(42), i16(44), i16(46)];
  const dtype = i16(70), voxOffset = f32(108);
  const slope = f32(112) || 1, inter = f32(116);
  const n = dims[0] * dims[1] * dims[2];
  const readers: Record<number, (o: number) => number> = { 2: (o) => dv.getUint8(o), 4: (o) => dv.getInt16(o, le), 8: (o) => dv.getInt32(o, le), 16: (o) => dv.getFloat32(o, le), 512: (o) => dv.getUint16(o, le), 256: (o) => dv.getInt8(o) };
  const size: Record<number, number> = { 2: 1, 4: 2, 8: 4, 16: 4, 512: 2, 256: 1 };
  const rd = readers[dtype]; if (!rd) throw new Error(`NIfTI data type ${dtype} is not supported.`);
  const data = new Int16Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.max(-32768, Math.min(32767, Math.round(rd(voxOffset + i * size[dtype]!) * slope + inter)));
  const sform = i16(254) > 0;
  let affine: number[][];
  if (sform) affine = [0, 1, 2].map((r) => [f32(280 + r * 16), f32(284 + r * 16), f32(288 + r * 16), f32(292 + r * 16)]);
  else { const px = [f32(80), f32(84), f32(88)]; affine = [[px[0]!, 0, 0, 0], [0, px[1]!, 0, 0], [0, 0, px[2]!, 0]]; }
  const sp = Math.min(...[0, 1, 2].map((d) => Math.hypot(affine[0]![d]!, affine[1]![d]!, affine[2]![d]!)));
  return { volume: { name: f.name.replace(/\.nii(\.gz)?$/i, ''), dims, data, scale: 1, offset: 0, affine, inverse: invertAffine(affine), shift: [0, 0, 0], spacing: sp }, slices: dims[2], skipped: 0, series: f.name, note: sform ? [] : ['No sform in this NIfTI; the scan is shown unoriented.'] };
}
