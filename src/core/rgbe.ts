/**
 * Radiance RGBE (.hdr) decoding to half-float RGBA, the same texels three's HDRLoader makes (value =
 * mantissa * 2^(e - 128) / 255, alpha 1). Shared by the sky decode worker and its main-thread fallback.
 * Reads new-style RLE and flat scanlines (Poly Haven and OpenCV write RLE, the old prep tool wrote flat).
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

export function decodeRGBE(buf: ArrayBuffer): DecodedHDR {
  const bytes = new Uint8Array(buf);
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
