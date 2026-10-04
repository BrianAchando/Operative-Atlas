import { unzipSync } from 'fflate';

/**
 * Decoders for compressed DICOM pixel data, loaded only when a compressed study is opened.
 * JPEG 2000 (what most web PACS and teleradiology viewers send) uses OpenJPEG compiled to WebAssembly;
 * JPEG lossless (process 14, common on CT archives) uses a small pure-JS decoder.
 */

type J2KModule = { J2KDecoder: new () => {
  getEncodedBuffer(n: number): Uint8Array; decode(): void; getDecodedBuffer(): Uint8Array;
  getFrameInfo(): { width: number; height: number; bitsPerSample: number; componentCount: number; isSigned: boolean };
} };
let j2k: Promise<J2KModule> | null = null;

async function openjpeg(): Promise<J2KModule> {
  j2k ??= (async () => {
    const [{ default: factory }, { default: wasmUrl }] = await Promise.all([
      import('@cornerstonejs/codec-openjpeg/decodewasmjs') as Promise<{ default: (o: object) => Promise<J2KModule> }>,
      import('@cornerstonejs/codec-openjpeg/decodewasm?url') as Promise<{ default: string }>,
    ]);
    return factory({ locateFile: (f: string) => (f.endsWith('.wasm') ? wasmUrl : f) });
  })();
  return j2k;
}

export const J2K = new Set(['1.2.840.10008.1.2.4.90', '1.2.840.10008.1.2.4.91']);
export const JPEG_LOSSLESS = new Set(['1.2.840.10008.1.2.4.57', '1.2.840.10008.1.2.4.70']);
export const DECODABLE = new Set([...J2K, ...JPEG_LOSSLESS]);

/** Decode one compressed frame to raw 16-bit samples (signed or unsigned as stored). */
export async function decodeFrame(ts: string, frame: Uint8Array, signed: boolean): Promise<Int16Array | Uint16Array> {
  if (J2K.has(ts)) {
    const mod = await openjpeg();
    const d = new mod.J2KDecoder();
    d.getEncodedBuffer(frame.length).set(frame);
    d.decode();
    const out = d.getDecodedBuffer();
    const copy = out.slice().buffer;
    return signed ? new Int16Array(copy) : new Uint16Array(copy);
  }
  if (JPEG_LOSSLESS.has(ts)) {
    const { Decoder } = await import('jpeg-lossless-decoder-js');
    const buf = frame.slice().buffer;
    const r = new Decoder().decode(buf, 0, buf.byteLength, 2) as ArrayLike<number> & { buffer: ArrayBuffer };
    return signed ? new Int16Array(r.buffer) : new Uint16Array(r.buffer);
  }
  throw new Error(`Unsupported transfer syntax ${ts}`);
}

/** Expand any .zip among the dropped files (Export from most web viewers gives a zip of the study). */
export async function expandZips(files: File[]): Promise<File[]> {
  const out: File[] = [];
  for (const f of files) {
    if (!/\.zip$/i.test(f.name)) { out.push(f); continue; }
    const entries = unzipSync(new Uint8Array(await f.arrayBuffer()));
    for (const [name, data] of Object.entries(entries)) {
      if (name.endsWith('/') || /(^|\/)(__MACOSX|\.)/.test(name) || !data.length) continue;
      out.push(new File([data], name.split('/').pop()!));
    }
  }
  return out;
}
