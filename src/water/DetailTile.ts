import * as THREE from 'three/webgpu';
import { fft } from './CausticTile';

/** Decoded slopes (RG x 2 - 1) times this give slopes with an RMS of 1. */
export const DETAIL_SLOPE = 3;

/**
 * Tileable detail texture for the water surface, built on the CPU at load (~50 ms), so no water asset has to be
 * downloaded or licensed:
 *  RG = surface slope (d/du, d/dv) as 0.5 + 0.5 s / (3 RMS) (clipped beyond 3 sigma)
 *  B  = height 0..1
 *  A  = foam lace: 1 on the borders of a wrapped Worley cell grid (where F2 - F1 is small), 0 in the cells
 * 180 waves on integer lattice vectors (so it wraps), random directions and phases, |k| log-uniform over 2-48
 * cycles per tile, slope amplitude ~ k^-0.8: an irregular wobble with cells of many sizes and no visible grid
 * (the 40-wave tile of round 1 read as privacy glass).
 */
export function bakeWaterDetail(size = 512, seed = 7): THREE.DataTexture {
  const N = size;
  let st = seed >>> 0;
  const rnd = () => ((st = (st * 1664525 + 1013904223) >>> 0) / 4294967296);
  const sr = new Float64Array(N * N), si = new Float64Array(N * N); // slopes: gx + i gy
  const hr = new Float64Array(N * N), hi = new Float64Array(N * N); // height (real)
  const w = (2 * Math.PI) / N;
  const used = new Set<number>();
  for (let n = 0; n < 180;) {
    const kmag = 2 * Math.pow(24, rnd()), th = rnd() * Math.PI * 2;
    const kx = Math.round(kmag * Math.cos(th)), ky = Math.round(kmag * Math.sin(th));
    const k = Math.hypot(kx, ky);
    if (k < 1.5 || used.has(kx * 1000 + ky) || used.has(-kx * 1000 - ky)) continue;
    used.add(kx * 1000 + ky);
    n++;
    const a = Math.pow(k, -1.8), ph = rnd() * Math.PI * 2;
    const kxp = w * kx, kyp = w * ky;
    const p = ((ky + N) % N) * N + ((kx + N) % N), q = ((-ky + N) % N) * N + ((-kx + N) % N);
    const er = Math.cos(ph), ei = Math.sin(ph);
    // slopes (see CausticTile): C+ = (i a / 2)(kx' + i ky') e^{i ph}, C- = -(i a / 2)(kx' + i ky') e^{-i ph}
    const cr = -0.5 * a * kyp, ci = 0.5 * a * kxp;
    sr[p] += cr * er - ci * ei;
    si[p] += cr * ei + ci * er;
    sr[q] -= cr * er + ci * ei;
    si[q] -= ci * er - cr * ei;
    // height a cos(theta) = (a / 2)(e^{i theta} + e^{-i theta})
    hr[p] += 0.5 * a * er;
    hi[p] += 0.5 * a * ei;
    hr[q] += 0.5 * a * er;
    hi[q] -= 0.5 * a * ei;
  }
  for (const [re, im] of [[sr, si], [hr, hi]] as const) {
    for (let r = 0; r < N; r++) fft(re, im, N, r * N, 1, true);
    for (let c = 0; c < N; c++) fft(re, im, N, c, N, true);
  }
  let s2 = 0, hMin = Infinity, hMax = -Infinity;
  for (let k = 0; k < N * N; k++) {
    s2 += sr[k] * sr[k] + si[k] * si[k];
    hMin = Math.min(hMin, hr[k]);
    hMax = Math.max(hMax, hr[k]);
  }
  // RMS of one slope component
  const rms = Math.sqrt(s2 / (2 * N * N));

  const C = 9;
  const pts = new Float32Array(C * C * 2);
  for (let c = 0; c < C * C * 2; c++) pts[c] = rnd();
  const lace = new Float32Array(N * N);
  let lMax = 1e-6;
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const u = (i / N) * C, v = (j / N) * C, cu = Math.floor(u), cv = Math.floor(v);
      let f1 = 9, f2 = 9;
      for (let dv = -1; dv <= 1; dv++) {
        for (let du = -1; du <= 1; du++) {
          const gu = cu + du, gv = cv + dv;
          const wu = ((gu % C) + C) % C, wv = ((gv % C) + C) % C;
          const px = gu + pts[(wv * C + wu) * 2], pz = gv + pts[(wv * C + wu) * 2 + 1];
          const d = Math.hypot(px - u, pz - v);
          if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) f2 = d;
        }
      }
      const k = j * N + i;
      lace[k] = f2 - f1;
      lMax = Math.max(lMax, lace[k]);
    }
  }

  const enc = (s: number) => Math.max(0, Math.min(255, Math.round((0.5 + (0.5 * s) / (DETAIL_SLOPE * rms)) * 255)));
  const data = new Uint8Array(N * N * 4);
  for (let k = 0; k < N * N; k++) {
    data[k * 4] = enc(sr[k]);
    data[k * 4 + 1] = enc(si[k]);
    data[k * 4 + 2] = Math.round(((hr[k] - hMin) / (hMax - hMin)) * 255);
    data[k * 4 + 3] = Math.round((1 - lace[k] / lMax) * 255);
  }
  const t = new THREE.DataTexture(data, N, N, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.anisotropy = 8;
  t.colorSpace = THREE.NoColorSpace;
  t.needsUpdate = true;
  return t;
}
