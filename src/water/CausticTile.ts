import * as THREE from 'three/webgpu';

/**
 * Metres of bed covered by one tile of the caustic net (deeper water; the shallows use it at half size, the riffle
 * 1.25x larger). (water fix 2: at 2 m its cells were 25-50 cm, a few lines per stone field that read as glints on the
 * stone tops, C2, W3; in sunlit shallows the web's cells are about a hand across.)
 */
export const CAUSTIC_TILE = 1.1;
/** R stores caustic irradiance / CAUSTIC_RANGE (flat water = 1). */
export const CAUSTIC_RANGE = 12;

/** Tone curve of the baked net: irradiance^power, then a soft shoulder above `knee` (x flat water). */
export interface CausticTone {
  power: number;
  knee: number;
  soft: number;
}
export const CAUSTIC_TONE: CausticTone = { power: 2.0, knee: 5, soft: 0.1 };

/** In-place radix-2 complex FFT of n values spaced `stride` apart starting at `off` (inverse: e^{+i}). */
export function fft(re: Float64Array, im: Float64Array, n: number, off: number, stride: number, inverse: boolean) {
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const a = off + i * stride, b = off + j * stride;
      let t = re[a]; re[a] = re[b]; re[b] = t;
      t = im[a]; im[a] = im[b]; im[b] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = off + (i + k) * stride, b = a + half * stride;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
}

export interface CausticBake {
  size: number;
  /** RGBA8: R caustic irradiance / CAUSTIC_RANGE after the tone curve (mean 1), G and B smooth noise, A 255 */
  data: Uint8Array;
  /** fraction of texels brighter than 2x and darker than 0.5x flat water */
  bright: number;
  dark: number;
}

/**
 * Caustic net baked on the CPU at load (~60-90 ms), tileable, so no asset has to be downloaded or licensed:
 * sunlight refracted through a rippled surface lands on a flat bed past its first focus, where the bright
 * lines join into a fine net around darker cells. The surface is 120 waves on integer lattice vectors (so it
 * wraps), random phases, wavelengths 2.7-30 cm, amplitude ~ k^-2.3: every scale bends the light about equally,
 * so the cells come in many sizes. Slopes come from one inverse FFT (gx + i gy); 4 photons per texel follow the
 * interpolated slope and are splatted bilinearly, then lightly blurred so the thinnest lines do not alias.
 * `focus`: displacement x RMS curvature (about 1 = first focus; larger = denser, finer net).
 * The net is sharpened (power 2: dark cells, bright thin lines, so it still reads over the busy pebble texture
 * after the footprint blur and tone mapping; at power 1.4 it raised the bed by at most ~1.8x in the frame), given
 * a soft shoulder above 5x flat water (bright, sun-coloured lines, never white blobs) and renormalised so the mean
 * stays 1 (the net only redistributes light).
 */
export function bakeCausticData(N = 512, seed = 11, focus = 1.45, tone: CausticTone = CAUSTIC_TONE): CausticBake {
  let st = seed >>> 0;
  const rnd = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
  const re = new Float64Array(N * N), im = new Float64Array(N * N);
  const w = (2 * Math.PI) / N;
  let curv2 = 0;
  const used = new Set<number>();
  for (let n = 0; n < 120;) {
    // |k| log-uniform in [3.3, 33] cycles per tile
    const kmag = 3.3 * Math.pow(10, rnd()), th = rnd() * Math.PI * 2;
    const kx = Math.round(kmag * Math.cos(th)), ky = Math.round(kmag * Math.sin(th));
    const k = Math.hypot(kx, ky);
    const key = (kx + N) * 4 * N + (ky + N);
    if (k < 3 || used.has(key) || used.has((-kx + N) * 4 * N + (-ky + N))) continue;
    used.add(key);
    n++;
    const a = Math.pow(k, -2.3), ph = rnd() * Math.PI * 2;
    const kxp = w * kx, kyp = w * ky; // radians per texel
    curv2 += (a * a * (kxp * kxp + kyp * kyp) ** 2) / 2;
    // Z = gx + i gy = sum (i a / 2)(kx' + i ky')(e^{i(k.x + ph)} - e^{-i(k.x + ph)})
    const cr = -0.5 * a * kyp, ci = 0.5 * a * kxp; // (i a / 2)(kx' + i ky')
    const er = Math.cos(ph), ei = Math.sin(ph);
    const p = ((ky + N) % N) * N + ((kx + N) % N), q = ((-ky + N) % N) * N + ((-kx + N) % N);
    re[p] += cr * er - ci * ei;
    im[p] += cr * ei + ci * er;
    re[q] -= cr * er + ci * ei;
    im[q] -= ci * er - cr * ei;
  }
  for (let r = 0; r < N; r++) fft(re, im, N, r * N, 1, true);
  for (let c = 0; c < N; c++) fft(re, im, N, c, N, true);

  // displacement per unit slope (texels): `focus` / RMS curvature
  const D = focus / Math.sqrt(curv2);
  const acc = new Float32Array(N * N);
  const m = N - 1;
  for (let j = 0; j < N; j++) {
    for (let sj = 0; sj < 2; sj++) {
      const v = j + 0.25 * (2 * sj - 1); // sub-texel photon row, in texel-centre coordinates
      const j0 = Math.floor(v), fy = v - j0, ja = ((j0 % N) + N) % N, jb = (ja + 1) & m;
      for (let i = 0; i < N; i++) {
        for (let si = 0; si < 2; si++) {
          const u = i + 0.25 * (2 * si - 1);
          const i0 = Math.floor(u), fx = u - i0, ia = ((i0 % N) + N) % N, ib = (ia + 1) & m;
          const k00 = ja * N + ia, k10 = ja * N + ib, k01 = jb * N + ia, k11 = jb * N + ib;
          const sx = (re[k00] * (1 - fx) + re[k10] * fx) * (1 - fy) + (re[k01] * (1 - fx) + re[k11] * fx) * fy;
          const sy = (im[k00] * (1 - fx) + im[k10] * fx) * (1 - fy) + (im[k01] * (1 - fx) + im[k11] * fx) * fy;
          const x = u + D * sx, y = v + D * sy;
          const x0 = Math.floor(x), y0 = Math.floor(y), tx = x - x0, ty = y - y0;
          const xa = ((x0 % N) + N) % N, xb = (xa + 1) & m, ya = ((y0 % N) + N) % N, yb = (ya + 1) & m;
          acc[ya * N + xa] += 0.25 * (1 - tx) * (1 - ty);
          acc[ya * N + xb] += 0.25 * tx * (1 - ty);
          acc[yb * N + xa] += 0.25 * (1 - tx) * ty;
          acc[yb * N + xb] += 0.25 * tx * ty;
        }
      }
    }
  }
  // binomial blur [1 2 1] / 4, separable, wrapped
  const tmp = new Float32Array(N * N);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) tmp[j * N + i] = 0.25 * acc[j * N + ((i - 1) & m)] + 0.5 * acc[j * N + i] + 0.25 * acc[j * N + ((i + 1) & m)];
  }
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) acc[j * N + i] = 0.25 * tmp[((j - 1) & m) * N + i] + 0.5 * tmp[j * N + i] + 0.25 * tmp[((j + 1) & m) * N + i];
  }

  // two smooth tileable noises (period = tile and tile / 2 mixed) for the warp
  const noise = (sd: number) => {
    let s2 = sd >>> 0;
    const r2 = () => ((s2 = (s2 * 1664525 + 1013904223) >>> 0) / 4294967296);
    const ws: Array<{ kx: number; ky: number; a: number; ph: number }> = [];
    while (ws.length < 9) {
      const kx = Math.round((r2() * 2 - 1) * 4), ky = Math.round((r2() * 2 - 1) * 4);
      const k = Math.hypot(kx, ky);
      if (k < 1) continue;
      ws.push({ kx, ky, a: 1 / k, ph: r2() * Math.PI * 2 });
    }
    const out = new Float32Array(N * N);
    const cx = new Float32Array(N), sx2 = new Float32Array(N);
    for (const wv of ws) {
      for (let i = 0; i < N; i++) {
        cx[i] = Math.cos(w * wv.kx * i);
        sx2[i] = Math.sin(w * wv.kx * i);
      }
      for (let j = 0; j < N; j++) {
        const b = w * wv.ky * j + wv.ph, cb = Math.cos(b), sb = Math.sin(b);
        for (let i = 0; i < N; i++) out[j * N + i] += wv.a * (cx[i] * cb - sx2[i] * sb);
      }
    }
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < N * N; k++) {
      lo = Math.min(lo, out[k]);
      hi = Math.max(hi, out[k]);
    }
    for (let k = 0; k < N * N; k++) out[k] = (out[k] - lo) / (hi - lo);
    return out;
  };
  const n1 = noise(seed * 7 + 1), n2 = noise(seed * 13 + 5);

  let fMean = 0;
  for (let k = 0; k < N * N; k++) {
    const e = Math.pow(acc[k], tone.power);
    acc[k] = e / (1 + Math.max(e - tone.knee, 0) * tone.soft);
    fMean += acc[k];
  }
  fMean /= N * N;
  const data = new Uint8Array(N * N * 4);
  let bright = 0, dark = 0;
  for (let k = 0; k < N * N; k++) {
    const c = acc[k] / fMean;
    if (c > 2) bright++;
    if (c < 0.5) dark++;
    data[k * 4] = Math.min(255, Math.round((c / CAUSTIC_RANGE) * 255));
    data[k * 4 + 1] = Math.round(n1[k] * 255);
    data[k * 4 + 2] = Math.round(n2[k] * 255);
    data[k * 4 + 3] = 255;
  }
  return { size: N, data, bright: bright / (N * N), dark: dark / (N * N) };
}

let cached: THREE.DataTexture | null = null;

/** The shared caustic texture (baked on first use). */
export function causticTexture(): THREE.DataTexture {
  if (cached) return cached;
  const b = bakeCausticData();
  const t = new THREE.DataTexture(b.data, b.size, b.size, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.colorSpace = THREE.NoColorSpace;
  t.needsUpdate = true;
  cached = t;
  return t;
}
