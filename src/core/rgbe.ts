/**
 * Radiance RGBE (.hdr) decoding to half-float RGBA, the same texels three's HDRLoader makes (value =
 * mantissa * 2^(e - 128) / 255, alpha 1). Shared by the sky decode worker and its main-thread fallback.
 * Reads new-style RLE and flat scanlines (Poly Haven and OpenCV write RLE, the old prep tool wrote flat), and the
 * packed skies the game ships (`.rgbe.gz`, scripts/pack-sky.ts): the same RGBE bytes as the .hdr, stored as four
 * planes (R, G, B, E) of rows, each byte the difference from its left neighbour, gzipped. That is lossless and
 * about 45% smaller on the wire than the RLE file gzipped by the server.
 */
export interface DecodedHDR {
  width: number;
  height: number;
  /** RGBA half floats, rows top (zenith) first, as HDRLoader returns them (the texture uses flipY) */
  data: Uint16Array;
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** float -> IEEE half (round toward zero, as three's DataUtils.toHalfFloat); inputs are >= 0 and <= 65504 */
function toHalf(v: number): number {
  f32[0] = v;
  const x = u32[0];
  const e = (x >>> 23) & 0xff;
  if (e < 103) return 0; // < 2^-24: zero
  if (e < 113) return (0x0400 | (x & 0x7fffff) >>> 13) >>> (113 - e); // subnormal half
  if (e > 142) return 0x7bff; // clamp to the largest finite half
  return ((e - 112) << 10) | ((x >>> 13) & 0x3ff);
}

/** raw RGBE bytes (4 per texel, rows top first) */
export interface RawRGBE {
  width: number;
  height: number;
  rgbe: Uint8Array;
}

/** "RGBD": the header of a packed sky after gunzip, then u32 width, u32 height (little endian) and the planes */
const PACKED_MAGIC = 0x44424752;

export function packRGBE(raw: RawRGBE): Uint8Array {
  const { width: w, height: h, rgbe } = raw;
  const out = new Uint8Array(12 + w * h * 4);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, PACKED_MAGIC, true);
  dv.setUint32(4, w, true);
  dv.setUint32(8, h, true);
  let o = 12;
  for (let c = 0; c < 4; c++) {
    for (let y = 0; y < h; y++) {
      let prev = 0;
      for (let x = 0, j = y * w * 4 + c; x < w; x++, j += 4) {
        out[o++] = (rgbe[j] - prev) & 0xff;
        prev = rgbe[j];
      }
    }
  }
  return out;
}

function unpackRGBE(bytes: Uint8Array): RawRGBE {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const w = dv.getUint32(4, true), h = dv.getUint32(8, true);
  if (bytes.length !== 12 + w * h * 4) throw new Error('rgbe: packed sky has the wrong size');
  const rgbe = new Uint8Array(w * h * 4);
  let o = 12;
  for (let c = 0; c < 4; c++) {
    for (let y = 0; y < h; y++) {
      let v = 0;
      for (let x = 0, j = y * w * 4 + c; x < w; x++, j += 4) {
        v = (v + bytes[o++]) & 0xff;
        rgbe[j] = v;
      }
    }
  }
  return { width: w, height: h, rgbe };
}

/** a Radiance .hdr or an (already gunzipped) packed sky, as raw RGBE bytes */
export function readRGBE(buf: ArrayBuffer): RawRGBE {
  const bytes = new Uint8Array(buf);
  if (bytes.length >= 12 && new DataView(buf).getUint32(0, true) === PACKED_MAGIC) return unpackRGBE(bytes);
  return readRadiance(bytes);
}

/** fetch a sky file (.hdr or .rgbe.gz) and decode it to half floats */
export async function fetchSky(url: string): Promise<DecodedHDR> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  let buf = await r.arrayBuffer();
  // GitHub Pages sends a .gz as it is; Vite's dev server sends it with Content-Encoding: gzip, already unzipped
  const head = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  if (head[0] === 0x1f && head[1] === 0x8b) {
    buf = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  return decodeRGBE(buf);
}

function readRadiance(bytes: Uint8Array): RawRGBE {
  let pos = 0;
  const line = () => {
    let s = '';
    while (pos < bytes.length) {
      const c = bytes[pos++];
      if (c === 0x0a) break;
      s += String.fromCharCode(c);
    }
    return s;
  };
  if (!line().startsWith('#?')) throw new Error('rgbe: not a Radiance file');
  for (let guard = 0; ; guard++) {
    if (pos >= bytes.length || guard > 64) throw new Error('rgbe: no end of header');
    if (line() === '') break;
  }
  const m = /^-Y (\d+) \+X (\d+)$/.exec(line().trim());
  if (!m) throw new Error('rgbe: unsupported orientation');
  const h = Number(m[1]), w = Number(m[2]);
  const rgbe = new Uint8Array(w * h * 4);
  const scan = new Uint8Array(w * 4);
  for (let y = 0; y < h; y++) {
    if (pos + 4 > bytes.length) throw new Error('rgbe: truncated');
    const rle = w >= 8 && w < 32768 && bytes[pos] === 2 && bytes[pos + 1] === 2 && ((bytes[pos + 2] << 8) | bytes[pos + 3]) === w;
    if (rle) {
      pos += 4;
      for (let c = 0; c < 4; c++) {
        let x = 0;
        while (x < w) {
          let n = bytes[pos++];
          if (n > 128) {
            n -= 128;
            if (n === 0 || x + n > w) throw new Error('rgbe: bad run');
            const v = bytes[pos++];
            for (let k = 0; k < n; k++) scan[(x++) * 4 + c] = v;
          } else {
            if (n === 0 || x + n > w) throw new Error('rgbe: bad dump');
            for (let k = 0; k < n; k++) scan[(x++) * 4 + c] = bytes[pos++];
          }
        }
      }
      rgbe.set(scan, y * w * 4);
    } else {
      // flat scanline (old-style run-length scanlines are not supported)
      rgbe.set(bytes.subarray(pos, pos + w * 4), y * w * 4);
      pos += w * 4;
    }
  }
  return { width: w, height: h, rgbe };
}

export function decodeRGBE(buf: ArrayBuffer): DecodedHDR {
  const { width: w, height: h, rgbe } = readRGBE(buf);
  const scale = new Float32Array(256);
  for (let e = 1; e < 256; e++) scale[e] = 2 ** (e - 128) / 255;
  const n = w * h;
  const out = new Uint16Array(n * 4);
  const F16 = (globalThis as unknown as { Float16Array?: new (b: ArrayBuffer) => ArrayLike<number> & { [i: number]: number } }).Float16Array;
  if (F16) {
    // native conversion (round to nearest) where the browser has Float16Array
    const f = new F16(out.buffer);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const s = scale[rgbe[j + 3]];
      f[j] = Math.min(rgbe[j] * s, 65504);
      f[j + 1] = Math.min(rgbe[j + 1] * s, 65504);
      f[j + 2] = Math.min(rgbe[j + 2] * s, 65504);
      f[j + 3] = 1;
    }
  } else {
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const s = scale[rgbe[j + 3]];
      out[j] = toHalf(Math.min(rgbe[j] * s, 65504));
      out[j + 1] = toHalf(Math.min(rgbe[j + 1] * s, 65504));
      out[j + 2] = toHalf(Math.min(rgbe[j + 2] * s, 65504));
      out[j + 3] = 0x3c00;
    }
  }
  return { width: w, height: h, data: out };
}
