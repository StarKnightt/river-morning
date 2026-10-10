import * as THREE from 'three/webgpu';
import { river, newSample } from '../world/River';
import { fbm, smoothstep } from '../world/noise';
import { centreAt, sAtX } from './RiverRibbon';
import type { Obstacle, RiverField } from './RiverField';
import { texture, vec2, type N } from '../gfx/tsl';

/**
 * Fast-water features of the riffle and the pool head, baked once at load on a river-aligned grid (arc length s
 * along the centreline, signed lateral distance d, + = south: the ribbon's own uv), 0.1 m texels. Everything is
 * placed from the real geometry: the bed under the water (groundY), the baked current and depth (RiverField) and
 * the rocks (the obstacles the field already knows), never from a random texture.
 *
 *  A (RGBA16F): surface slope x, z (world: the standing waves and the water heaped on and around the rocks),
 *               breaking crests 0..1 (white curls on the steepest crests), glassy 0..1 (smooth fast water)
 *  B (RGBA16F): froth 0..1 (white water at the rocks and where the riffle plunges into the pool), aeration 0..1
 *               (bubbles in the water column; the pool-head plume), streaks 0..1 (V wakes behind the rocks and the
 *               foam lines along the jet's seams into the pool), coverage 0..1 (where these replace the generic
 *               fast-water terms of the water material)
 *
 *  - Rows of standing waves: wherever the bed drops downstream faster than its own fall over ~4 m (the risers of
 *    the cobble ledges, the lee of bed humps) the water forms a train of stationary crests whose wavelength follows from the
 *    local current (stationary gravity waves, lambda ~ 2 pi U² / g, shortened for a cobble riffle) and which dies
 *    out over a few crests. Integrated along s with a damped complex oscillator, so the phase follows the local
 *    wavelength; the crests are sharpened (narrow crests, broad flat troughs).
 *  - Rocks: water heaped against the upstream face (a pillow ~U²/2g high), a hole behind it, the V of the two
 *    diverging arms (half angle from the Froude number), a short train of curved transverse crests in the wedge,
 *    a white collar where the pillow breaks round the flanks, white water in the hole, a white roller behind rocks
 *    just under the surface, streaks along the arms.
 *  - Glassy tongues: the water converging through the gap between two neighbouring rocks (a V pointing
 *    downstream), the deep thread of the current outside every rock's wake, the smooth pour over each ledge lip.
 *  - Pool head: where the section speed falls off (the jet plunging into deep water), a heavy white mass that breaks
 *    up within a few metres, bubbles under the surface that fade over ~5 m, and thin foam lines along both seams of
 *    the jet that drift ~16 m down the pool.
 *
 * ~0.12 M texels; the loops only visit wet texels near the features (load cost ~20-30 ms in the browser).
 */
export interface RiffleField {
  s0: number;
  s1: number;
  d0: number;
  d1: number;
  texel: number;
  w: number;
  h: number;
  texA: THREE.DataTexture;
  texB: THREE.DataTexture;
  /** where the riffle's jet plunges into the pool (debug and tests) */
  plunge: { s: number; d: number; x: number; z: number; half: number } | null;
  ms: number;
}

const G = 9.81;
/** crest shape ((1 + cos psi) / 2)³ minus its mean (5/16): narrow crests, broad flat troughs */
const P_MEAN = 5 / 16;

/**
 * Wavelength of the stationary waves for a current U (m/s). Deep-water gravity waves standing in a current U have
 * lambda = 2 pi U² / g (2.5 m at 2 m/s); over cobbles the crests come at about 0.6 of that (shallower local
 * depths, three-dimensional bed), so 0.6 x, kept within 0.75-1.9 m.
 */
const standingLambda = (U: number) => Math.min(1.9, Math.max(0.75, (0.6 * 2 * Math.PI * U * U) / G));

/** Bilinear lookup of the baked field into out: current x, z, water level, section speed. False outside it. */
function fieldAt(f: RiverField, x: number, z: number, out: Float64Array): boolean {
  const fi = (x - f.x0) / f.texel, fj = (z - f.z0) / f.texel;
  if (fi < 0 || fj < 0 || fi > f.w - 1 || fj > f.h - 1) return false;
  const i = Math.min(f.w - 2, Math.floor(fi)), j = Math.min(f.h - 2, Math.floor(fj));
  const tx = fi - i, ty = fj - j;
  const k0 = (j * f.w + i) * 4, k1 = k0 + f.w * 4;
  const w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
  const a = f.a, b = f.b;
  out[0] = a[k0] * w00 + a[k0 + 4] * w10 + a[k1] * w01 + a[k1 + 4] * w11;
  out[1] = a[k0 + 1] * w00 + a[k0 + 5] * w10 + a[k1 + 1] * w01 + a[k1 + 5] * w11;
  out[2] = a[k0 + 3] * w00 + a[k0 + 7] * w10 + a[k1 + 3] * w01 + a[k1 + 7] * w11;
  out[3] = b[k0 + 1] * w00 + b[k0 + 5] * w10 + b[k1 + 1] * w01 + b[k1 + 5] * w11;
  return true;
}

export function bakeRiffle(o: {
  field: RiverField;
  groundY: (x: number, z: number) => number;
  obstacles: Obstacle[];
  /** centreline x range */
  x0?: number;
  x1?: number;
  /** lateral half range (m) */
  dMax?: number;
  texel?: number;
}): RiffleField {
  const t0 = performance.now();
  const texel = o.texel ?? 0.1;
  const s0 = sAtX(o.x0 ?? -46), s1 = sAtX(o.x1 ?? 12);
  const dMax = o.dMax ?? 10;
  const w = Math.round((s1 - s0) / texel) + 1, h = Math.round((2 * dMax) / texel) + 1;
  const d0 = -dMax;
  const N = w * h;
  const S = newSample();
  const fs = new Float64Array(4);

  // ---- the river frame of every column (same construction as the ribbon: tangent from a +-1.5 m chord), and the
  // rows that can hold water (the channel's half width plus a margin; the banks beyond stay dry)
  const cx = new Float32Array(w), cz = new Float32Array(w), tx = new Float32Array(w), tz = new Float32Array(w);
  const jLo = new Int32Array(w), jHi = new Int32Array(w), secC = new Float32Array(w);
  const A2 = new THREE.Vector2(), B2 = new THREE.Vector2();
  for (let i = 0; i < w; i++) {
    const s = s0 + i * texel;
    centreAt(s - 1.5, A2);
    centreAt(s + 1.5, B2);
    const l = Math.hypot(B2.x - A2.x, B2.y - A2.y);
    tx[i] = (B2.x - A2.x) / l;
    tz[i] = (B2.y - A2.y) / l;
    centreAt(s, A2);
    cx[i] = A2.x;
    cz[i] = A2.y;
    river.sample(A2.x, A2.y, S);
    const half = S.half + 3;
    jLo[i] = Math.max(1, Math.floor((-half - d0) / texel));
    jHi[i] = Math.min(h - 2, Math.ceil((half - d0) / texel));
    secC[i] = S.speed;
  }
  const px = (i: number, j: number) => cx[i] - tz[i] * (d0 + j * texel);
  const pz = (i: number, j: number) => cz[i] + tx[i] * (d0 + j * texel);

  // ---- current, depth and section speed under every texel
  const U = new Float32Array(N), H = new Float32Array(N), SEC = new Float32Array(N), GR = new Float32Array(N);
  for (let i = 0; i < w; i++) {
    for (let j = jLo[i]; j <= jHi[i]; j++) {
      const k = j * w + i, x = px(i, j), z = pz(i, j);
      const g = o.groundY(x, z);
      GR[k] = g;
      if (!fieldAt(o.field, x, z, fs)) continue;
      U[k] = Math.sqrt(fs[0] * fs[0] + fs[1] * fs[1]);
      H[k] = fs[2] - g;
      SEC[k] = fs[3];
    }
  }

  const eta = new Float32Array(N); // surface height (m) relative to the mean level
  const crestW = new Float32Array(N), glassy = new Float32Array(N);
  const froth = new Float32Array(N), aer = new Float32Array(N), streak = new Float32Array(N);
  const wake = new Float32Array(N); // how much of a rock's wake covers the texel (no tongue there)
  const lipQ = new Float32Array(N); // wave source strength (bed drop), for the smooth pour over each lip

  // ---- rows of standing waves behind every drop of the bed (cobble ledges, the lee of humps), over the fast
  // stretch only. The oscillator's rotation and damping per texel come from a table over the current.
  let iF0 = w, iF1 = -1;
  for (let i = 0; i < w; i++) if (secC[i] > 0.95) { iF0 = Math.min(iF0, i); iF1 = i; }
  iF1 = Math.min(w - 1, iF1 + Math.round(10 / texel));
  let jF0 = h, jF1 = 0;
  for (let i = iF0; i <= iF1; i++) { jF0 = Math.min(jF0, jLo[i]); jF1 = Math.max(jF1, jHi[i]); }
  // the pool head: where the jet leaves the riffle and plunges into deep water (the section speed falls through
  // 1.5 m/s while the bed drops away). From a metre above it the trains die fast: the plunge is churned white water.
  let ip = -1;
  for (let i = 1; i < w; i++) if (secC[i - 1] >= 1.5 && secC[i] < 1.5 && cx[i] > -24) { ip = i; break; }
  const iDamp = ip > 0 ? ip - Math.round(1 / texel) : w;
  const LUT = 96, uLo = 0.4, uHi = 2.6;
  const lutC = new Float32Array(LUT), lutS = new Float32Array(LUT), lutR = new Float32Array(LUT), lutK = new Float32Array(LUT);
  for (let n = 0; n < LUT; n++) {
    const lam = standingLambda(uLo + ((uHi - uLo) * n) / (LUT - 1));
    const th = (2 * Math.PI * texel) / lam;
    lutC[n] = Math.cos(th);
    lutS[n] = Math.sin(th);
    lutR[n] = Math.exp(-texel / (1.5 * lam + 0.7));
    lutK[n] = (2 * Math.PI) / lam;
  }
  const smoothB = new Float32Array(w);
  for (let j = Math.max(1, jF0); j <= Math.min(h - 2, jF1); j++) {
    const row = j * w;
    // the bed along s, box-smoothed over 0.7 m (pebbles do not make waves, ledges and humps do): a running sum
    const bLo = Math.max(0, iF0 - 20), bHi = Math.min(w - 1, iF1 + 20);
    let sum = 0, cnt = 0;
    for (let ii = Math.max(0, bLo - 3); ii <= Math.min(w - 1, bLo + 3); ii++) { sum += GR[row + ii]; cnt++; }
    for (let i = bLo; i <= bHi; i++) {
      smoothB[i] = sum / cnt;
      const add = i + 4, rem = i - 3;
      if (add < w) { sum += GR[row + add]; cnt++; }
      if (rem >= 0) { sum -= GR[row + rem]; cnt--; }
    }
    let zr = 0, zi = 0;
    const d = d0 + j * texel;
    for (let i = iF0; i <= iF1; i++) {
      const k = row + i;
      const depth = H[k];
      if (depth <= 0.03) {
        // dry: whatever train was running dies
        zr *= 0.5;
        zi *= 0.5;
        continue;
      }
      // only a drop steeper than the bed's own fall over ~4 m forces a wave (the riffle falls ~2.5 %): the short
      // risers of the cobble ledges and the lee of humps. The long even slope down into the pool forced a regular
      // train of a dozen breaking crests over its last 7 m. In fast, not too shallow or deep water.
      const drop = (smoothB[Math.max(0, i - 3)] - smoothB[Math.min(w - 1, i + 3)]) / (6 * texel);
      const dropL = (smoothB[Math.max(0, i - 20)] - smoothB[Math.min(w - 1, i + 20)]) / (40 * texel);
      const ex = drop - Math.max(dropL, 0.025) - 0.015;
      const fastF = smoothstep(1.0, 1.65, SEC[k]);
      let q = 0;
      if (ex > 0 && fastF > 0) {
        // crests come in short segments, not ruler lines across the river
        const seg = 0.4 + 0.6 * smoothstep(-0.35, 0.35, fbm((s0 + i * texel) * 0.25 + 3.1, d * 0.3 - 1.7, 2));
        q = ex * fastF * smoothstep(0.05, 0.15, depth) * smoothstep(1.1, 0.6, depth) * seg;
        lipQ[k] = q;
      }
      // (water fix 2) the cobbles under fast, shallow water force waves too: a random forcing, smooth across the
      // current over ~1 m, gives every stretch of the riffle stacked rows of short standing crests (the ledges
      // alone left most of the riffle a gently rippled sheet seen low and upstream, W2)
      // (slope RMS ~0.11 at 1.8 m/s, the steepest crests ~0.5: the bigger ones get white tips)
      if (fastF > 0 && depth > 0.08 && depth < 0.9 && i <= iDamp) {
        const nz = fbm((s0 + i * texel) * 1.6 + 11.3, d * 0.85 + 4.2, 2);
        q += 0.12 * nz * fastF * smoothstep(0.08, 0.18, depth) * (1 - smoothstep(0.55, 0.9, depth));
      }
      const n = Math.max(0, Math.min(LUT - 1, Math.round(((U[k] - uLo) / (uHi - uLo)) * (LUT - 1))));
      const rho = lutR[n], c = lutC[n], sn = lutS[n];
      const nr = rho * (zr * c - zi * sn), ni = rho * (zr * sn + zi * c);
      zr = nr + q * texel * 3.2;
      zi = ni;
      if (i > iDamp) {
        zr *= 0.9;
        zi *= 0.9;
      }
      const amp = Math.sqrt(zr * zr + zi * zi);
      if (amp < 1e-5) continue;
      // phase since the source, the first crest a quarter wavelength below the drop: cos(arg z - pi/2) = zi / |z|
      const cw = (1 + zi / amp) / 2;
      const c3 = cw * cw * cw;
      eta[k] += amp * (c3 - P_MEAN);
      // white curls only on the steepest crests in shallow-enough water (elsewhere a crest is a bright glassy line)
      crestW[k] = Math.max(crestW[k], c3 * c3 * smoothstep(0.35, 0.7, amp * lutK[n]) * smoothstep(0.7, 0.3, depth));
    }
  }

  // ---- rocks
  const rk: Array<{ x: number; z: number; R: number; s: number; d: number; wgt: number; U0: number; dx: number; dz: number }> = [];
  for (const ob of o.obstacles) {
    river.sample(ob.x, ob.z, S);
    if (S.s < s0 - 4 || S.s > s1 + 4 || Math.abs(S.d) > dMax + 3) continue;
    const R = ob.r;
    // the approach flow, upstream of the rock
    const ax0 = ob.x - S.tx * (R + 0.6), az0 = ob.z - S.tz * (R + 0.6);
    if (!fieldAt(o.field, ax0, az0, fs)) continue;
    const U0 = Math.sqrt(fs[0] * fs[0] + fs[1] * fs[1]);
    // nothing in slow water: a rock or log lying in the calm pool (0.2-0.35 m/s) makes no pillow, froth or wake
    // worth drawing (its collar drew white froth round the fallen spruce)
    const live = smoothstep(0.3, 0.5, U0);
    if (live < 0.01) continue;
    const dx = fs[0] / U0, dz = fs[1] / U0;
    const h0 = Math.max(0.05, fs[2] - o.groundY(ax0, az0));
    const waterY = fieldAt(o.field, ob.x, ob.z, fs) ? fs[2] : S.waterY;
    const above = ob.topY - waterY;
    const em0 = smoothstep(-0.05, 0.03, above);
    const emerge = em0 * live;
    const nearTop = smoothstep(-0.25, -0.03, above) * (1 - em0) * live;
    if (emerge + nearTop < 0.02) continue;
    rk.push({ x: ob.x, z: ob.z, R, s: S.s, d: S.d, wgt: emerge + 0.7 * nearTop, U0, dx, dz });
    const head = (U0 * U0) / (2 * G);
    const Fr = U0 / Math.sqrt(G * h0);
    // half angle of the V: a Mach-like wedge at high Froude, wider in slower water
    const alpha = Math.min(0.75, Math.max(0.38, Math.asin(Math.min(1, 0.9 / Math.max(Fr, 0.05)))));
    const ta = Math.tan(alpha);
    const lamR = standingLambda(U0);
    const fastV = smoothstep(0.15, 0.8, U0);
    const reachUp = R + 1.2, reachDown = R + 2.5 + 2.5 * lamR;
    const reachSide = R + 0.5 + reachDown * ta;
    // the window in the river frame: the box round the rock's (along, across) rectangle turned by the angle between
    // the current and the centreline, and per column only the rows the wedge can reach (margins for the bend)
    const cosT = dx * S.tx + dz * S.tz, sinT = -dx * S.tz + dz * S.tx;
    let sLo = 1e9, sHi = -1e9, dLo = 1e9, dHi = -1e9;
    for (const ca of [-reachUp, reachDown]) {
      for (const cc of [-reachSide, reachSide]) {
        const es = ca * cosT - cc * sinT, ed = ca * sinT + cc * cosT;
        sLo = Math.min(sLo, es); sHi = Math.max(sHi, es); dLo = Math.min(dLo, ed); dHi = Math.max(dHi, ed);
      }
    }
    const i0 = Math.max(0, Math.floor((S.s + sLo - 1 - s0) / texel)), i1 = Math.min(w - 1, Math.ceil((S.s + sHi + 1 - s0) / texel));
    const jA = Math.max(1, Math.floor((S.d + dLo - 1 - d0) / texel)), jB = Math.min(h - 2, Math.ceil((S.d + dHi + 1 - d0) / texel));
    const wedge = cosT > 0.5;
    // Gaussian widths and gains of the terms (each is only evaluated where it is not negligible)
    // (water fix 2: a taller, wider pillow and a longer, steeper V. At 0.5 head and 0.14 head they did not read: the
    // rocks sat in the current like flat pucks with a white skirt, W1, C3)
    const sP = 0.12 + 0.3 * R, sGl = 0.15 + 0.45 * R, gP = 0.8 * head * emerge;
    const sTa = 0.55 * R + 0.12, sTc = 0.75 * R, sNa = 0.5 * R + 0.1, sNc = 0.9 * R;
    const sHa = 1.1 * R + 0.35, sHc = 0.7 * R + 0.05, sRa = 0.45 * R + 0.15, sRc = 0.95 * R;
    const vvG = (emerge + 0.6 * nearTop) * fastV;
    const armDecay = 1.2 + 1.4 * R, streakDecay = 0.7 + 0.9 * R, wakeDecay = 3 + 3 * R, tearDecay = 0.5 + 0.9 * R;
    for (let i = i0; i <= i1; i++) {
      let j0 = jA, j1 = jB;
      if (wedge) {
        const dS = s0 + i * texel - S.s, slack = 1 + 0.15 * Math.abs(dS);
        const aMax = dS * cosT + Math.abs(sinT) * (Math.abs(dLo) + Math.abs(dHi)) + slack;
        const cMax = Math.min(reachSide, Math.max(0.8 * R + Math.max(0, aMax) * ta + 0.5, R + 0.7));
        j0 = Math.max(jA, Math.floor((S.d + (-cMax + dS * sinT) / cosT - slack - d0) / texel));
        j1 = Math.min(jB, Math.ceil((S.d + (cMax + dS * sinT) / cosT + slack - d0) / texel));
      }
      for (let j = j0; j <= j1; j++) {
        const k = j * w + i;
        const rx = px(i, j) - ob.x, rz = pz(i, j) - ob.z;
        const a = rx * dx + rz * dz, c = -rx * dz + rz * dx;
        if (a < -reachUp || a > reachDown) continue;
        const ac = Math.abs(c);
        if (ac > reachSide) continue;
        const ap = a > 0 ? a : 0;
        const r = Math.sqrt(a * a + c * c);
        const arm = 0.8 * R + ap * ta;
        // nothing outside the wedge and away from the rock
        if (ac > arm + 0.5 && r > R + 0.7) continue;
        let e = 0;
        if (a < 0 && r > R * 0.85) {
          // water heaped against the upstream face, smooth (glassy) where it faces the current
          const up = -a / Math.max(r, 1e-3);
          const q1 = (r - R) / sP, q2 = (r - R) / sGl;
          if (emerge > 0 && q1 < 3.5) e += gP * Math.exp(-q1 * q1) * Math.pow(up, 1.2);
          if (emerge > 0 && q2 < 3.5) glassy[k] = Math.max(glassy[k], emerge * Math.exp(-q2 * q2) * up);
        }
        if (nearTop > 0) {
          // a rock just under the surface lifts a smooth hump and pours into a trough behind it
          const qa = a - 0.25 * R, qq = (qa * qa + c * c) / (R * R);
          if (qq < 12) e += 0.25 * head * nearTop * Math.exp(-qq);
        }
        if (a > 0) {
          const ta1 = (a - R) / sTa, tc1 = c / sTc, qt = ta1 * ta1 + tc1 * tc1;
          if (emerge > 0 && qt < 12) e -= 0.3 * head * emerge * Math.exp(-qt);
          const na = (a - 1.3 * R) / sNa, nc = c / sNc, qn = na * na + nc * nc;
          if (nearTop > 0 && qn < 12) e -= 0.2 * head * nearTop * Math.exp(-qn);
        }
        // the V: two diverging ridges from the flanks, and curved transverse crests inside the wedge
        // (water fix 2) the arms wander and break into pieces as they run off: as ruled Gaussian ridges they drew
        // straight lines across the whole riffle (W1)
        const side = c >= 0 ? 1 : -1;
        const wob = ap > 0 && Math.abs(ac - arm) < 0.8
          ? (0.04 + 0.09 * ap) * fbm(ap * 0.9 + ob.x * 3.1, side * 5.3 + ob.z * 2.7, 2) : 0;
        const dArm = ac - arm - wob;
        const vv = vvG > 0 ? vvG * smoothstep(-0.2 * R, 0.5 * R, a) : 0;
        if (vv > 0 && dArm > -0.5 && dArm < 0.5) {
          const qd = dArm / 0.12;
          const brk = smoothstep(-0.35, 0.25, fbm(ap * 1.4 + ob.z * 1.9, side * 3.1 - ob.x * 1.3, 2));
          const ridge = vv * Math.exp(-qd * qd) * (1 - Math.min(1, ap / (2.5 * R + 1.5)) * (1 - brk));
          e += 0.26 * head * ridge * Math.exp(-ap / armDecay);
          // the arms break white where they leave the flanks
          crestW[k] = Math.max(crestW[k], 0.6 * ridge * Math.exp(-ap / (0.5 + 0.8 * R)) * emerge);
        }
        if (a > 0.8 * R && ac < arm - 0.1) {
          const inside = smoothstep(arm - 0.1, arm - 0.45, ac);
          const cw = (1 + Math.sin((2 * Math.PI * (a - 0.8 * R - (c * c) / 4)) / lamR)) / 2;
          const env = inside * Math.exp(-(a - 0.8 * R) / (0.8 * lamR + 0.3)) * (emerge * 0.8 + nearTop * 1.2) * fastV;
          const c3 = cw * cw * cw;
          e += 0.3 * head * env * (c3 - P_MEAN);
          crestW[k] = Math.max(crestW[k], c3 * c3 * env * smoothstep(0.08, 0.2, (0.25 * head * env * 2 * Math.PI) / lamR));
        }
        if (e !== 0) eta[k] += e;
        // white water: a collar where the pillow breaks round the flanks (the face itself stays smooth), the hole
        // behind the rock (drawn out downstream: churned water trailing the rock), a roller behind a rock just
        // under the surface
        let collar = 0, hole = 0, roller = 0, tear = 0;
        if (emerge > 0 && r < R + 0.13) {
          // (water fix 2) the upstream face stays a smooth glassy pillow and the back is the hole: white water only where
          // the pillow breaks round the flanks (all round the waterline it read as a white skirt)
          const flank = a < 0 ? smoothstep(0.35, 0.85, ac / Math.max(r, 1e-3)) : 1 - smoothstep(0.2 * R, 0.9 * R, a);
          collar = emerge * smoothstep(R + 0.13, R - 0.02, r) * flank * (0.45 + 0.55 * fastV);
        }
        if (emerge > 0 && vv > 0 && a > -0.3 * R && dArm > -0.35 && dArm < 0.35) {
          // and tears off both sides along the arms
          const qt2 = dArm / 0.09;
          tear = emerge * fastV * Math.exp(-qt2 * qt2) * Math.exp(-ap / tearDecay) * smoothstep(-0.3 * R, 0.1 * R, a);
        }
        if (emerge > 0 && a > -0.2 * R) {
          const ha = (a - 1.1 * R) / sHa, hc = c / sHc, qh = ha * ha + hc * hc;
          if (qh < 12) hole = emerge * fastV * Math.exp(-qh);
        }
        if (nearTop > 0 && a > 0) {
          const ra = (a - 1.4 * R) / sRa, rc = c / sRc, qr = ra * ra + rc * rc;
          if (qr < 12) roller = nearTop * fastV * Math.exp(-qr);
        }
        if (collar + hole + roller + tear > 0) {
          // (at full strength the masks filled solid: flat white decals round every rock)
          froth[k] = Math.max(froth[k], collar * 0.75, hole * 0.55, roller * 0.75, tear * 0.85);
          aer[k] = Math.max(aer[k], hole * 0.8, roller * 0.7, collar * 0.4, tear * 0.5);
          // the churned water trailing the rock strings out along the current
          if (hole > 0 && a > 0.6 * R) streak[k] = Math.max(streak[k], hole * 0.4 * smoothstep(0.6 * R, 1.6 * R, a));
        }
        // bubbly streaks peel off along both arms
        if (vv > 0 && a > 0 && dArm > -0.4 && dArm < 0.4) {
          const qs = dArm / 0.1;
          streak[k] = Math.max(streak[k], vv * Math.exp(-qs * qs) * Math.exp(-ap / streakDecay) * smoothstep(0, 0.4 * R, a));
        }
        if (ac < arm + 0.3 && a > -0.5 * R) {
          wake[k] = Math.max(wake[k], (emerge + nearTop) * smoothstep(arm + 0.3, arm - 0.1, ac) * smoothstep(-0.5 * R, 0.2 * R, a)
            * Math.exp(-ap / wakeDecay));
        }
      }
    }
  }

  // ---- tongues: the water converging through the gap between two neighbouring rocks runs smooth and dark in a V
  // that points downstream and closes where their wakes meet; a short train of standing waves starts at its tip
  for (let p = 0; p < rk.length; p++) {
    for (let q2 = p + 1; q2 < rk.length; q2++) {
      const A = rk[p], B = rk[q2];
      let fx = A.dx + B.dx, fz = A.dz + B.dz;
      const fl = Math.hypot(fx, fz);
      if (fl < 1e-3) continue;
      fx /= fl;
      fz /= fl;
      const bx = B.x - A.x, bz = B.z - A.z;
      const along = bx * fx + bz * fz, across = -bx * fz + bz * fx;
      const hw = Math.abs(across) / 2 - 0.5 * (A.R + B.R);
      if (Math.abs(along) > 1.8 || hw < 0.25 || hw > 1.8) continue;
      const U0 = (A.U0 + B.U0) / 2;
      const str = Math.min(A.wgt, B.wgt) * smoothstep(0.9, 1.5, U0);
      if (str < 0.05) continue;
      const mx = (A.x + B.x) / 2, mz = (A.z + B.z) / 2;
      // the gap line sits at the downstream rock's centre; the V closes ~3 gap half-widths below it
      const aFront = Math.abs(along) / 2, L = 3.2 * hw;
      const lam = standingLambda(U0), head = (U0 * U0) / (2 * G);
      river.sample(mx, mz, S);
      const reachS = aFront + L + 3 * lam + 1, reachD = hw + 2.5;
      const i0 = Math.max(1, Math.floor((S.s - 1.5 - s0) / texel)), i1 = Math.min(w - 2, Math.ceil((S.s + reachS - s0) / texel));
      const j0 = Math.max(1, Math.floor((S.d - reachD - 1 - d0) / texel)), j1 = Math.min(h - 2, Math.ceil((S.d + reachD + 1 - d0) / texel));
      for (let j = j0; j <= j1; j++) {
        for (let i = i0; i <= i1; i++) {
          const k = j * w + i;
          if (H[k] <= 0.03) continue;
          const rx = px(i, j) - mx, rz = pz(i, j) - mz;
          const am = rx * fx + rz * fz - aFront, cm = Math.abs(-rx * fz + rz * fx);
          const half = hw * Math.max(0, Math.min(1, 1 - am / L));
          const v = smoothstep(half + 0.05, half - 0.25, cm) * smoothstep(-0.9, -0.2, am) * (1 - smoothstep(0.75 * L, L, am));
          // (a pair with a rock just under the surface still parts the current: no tongue fainter than ~0.35 + str)
          glassy[k] = Math.max(glassy[k], v * Math.min(1, 0.3 + str));
          wake[k] = Math.max(wake[k], v * str * 0.5);
          const a2 = am - 0.85 * L;
          if (a2 > 0) {
            const spread = hw * 0.9 + a2 * 0.35;
            const cw = (1 + Math.sin((2 * Math.PI * (a2 - (cm * cm) / 5)) / lam)) / 2;
            const env = str * smoothstep(spread + 0.2, spread - 0.4, cm) * Math.exp(-a2 / (1.1 * lam + 0.4));
            const c3 = cw * cw * cw;
            eta[k] += 0.3 * head * env * (c3 - P_MEAN);
            crestW[k] = Math.max(crestW[k], c3 * c3 * env * smoothstep(0.25, 0.5, (0.3 * head * env * 2 * Math.PI) / lam));
          }
        }
      }
    }
  }

  // ---- glassy water elsewhere: the deep thread of the riffle outside the wakes, the glide where the run speeds up
  // into the riffle, the smooth pour over each ledge lip (fast stretch only)
  for (let i = 0; i < w; i++) {
    if (secC[i] < 0.85) continue;
    let hMax = 0;
    for (let j = jLo[i]; j <= jHi[i]; j++) hMax = Math.max(hMax, H[j * w + i]);
    // (over +-2 m: the section speed is piecewise linear, a short window stepped at its knots into a hard edge)
    const accel = (secC[Math.min(w - 1, i + 20)] - secC[Math.max(0, i - 20)]) / (40 * texel);
    const glide = smoothstep(0.01, 0.05, accel) * smoothstep(1.0, 1.25, secC[i]) * smoothstep(1.8, 1.55, secC[i]);
    for (let j = jLo[i]; j <= jHi[i]; j++) {
      const k = j * w + i;
      if (H[k] <= 0.03 || SEC[k] < 0.95) continue;
      // the pour: the half metre above a riser, not the riser itself (a band over the whole riser glazed a 2 m strip
      // across the riffle at every ledge)
      let pour = 0;
      for (let n = 1; n <= 5; n++) pour += lipQ[j * w + Math.min(w - 1, i + n)] * texel;
      const fastF = smoothstep(1.0, 1.6, SEC[k]);
      const deepR = smoothstep(0.84, 0.96, H[k] / Math.max(hMax, 0.05));
      const thread = fastF * deepR * (1 - wake[k]) * 0.35;
      const lip = smoothstep(0.004, 0.015, pour) * smoothstep(0.03, 0.005, lipQ[k]) * fastF * 0.4;
      const g2 = glide * (0.3 + 0.4 * deepR) * smoothstep(0.08, 0.25, H[k]);
      glassy[k] = Math.max(glassy[k], thread, lip, g2) * (1 - Math.min(1, froth[k] * 1.5 + crestW[k]));
      // broken, choppy water carries a few bubbles: the bed only shows in glimpses
      aer[k] = Math.max(aer[k], fastF * (1 - glassy[k]) * 0.18 * smoothstep(0.05, 0.2, H[k]));
    }
  }

  // ---- pool head (ip, found above): a heavy white mass, bubbles under the surface, foam lines along the seams
  let plunge: RiffleField['plunge'] = null;
  if (ip > 0) {
    const sp = s0 + ip * texel;
    // the jet core one metre above the plunge: where the current runs fastest across the section
    const ic = Math.max(0, ip - Math.round(1 / texel));
    let best = 0, bj = Math.round(-d0 / texel);
    for (let j = jLo[ic]; j <= jHi[ic]; j++) if (U[j * w + ic] > best) { best = U[j * w + ic]; bj = j; }
    let lo = bj, hi = bj;
    while (lo > jLo[ic] && U[(lo - 1) * w + ic] > 0.75 * best) lo--;
    while (hi < jHi[ic] && U[(hi + 1) * w + ic] > 0.75 * best) hi++;
    const dc = d0 + ((lo + hi) / 2) * texel;
    const w0 = Math.max(0.8, ((hi - lo) * texel) / 2);
    const jc = Math.round((dc - d0) / texel);
    plunge = { s: sp, d: dc, x: px(ip, jc), z: pz(ip, jc), half: w0 };
    const seamC = [0, 0], seamW = [0, 0], seamG = [0, 0];
    const lamE = standingLambda(1.3);
    for (let i = 0; i < w; i++) {
      const ds = s0 + i * texel - sp;
      if (ds < -4.6 || ds > 22) continue;
      const dsp = Math.max(0, ds);
      const wj = w0 + dsp * 0.17;
      // (water fix 2: the heavy white body holds for ~3 m before it breaks up; within ~1.5 m it read as a pale oval, W3)
      const massS = smoothstep(-1.4, 0.0, ds) * Math.exp(-dsp / 4.0);
      const aerS = smoothstep(-1.2, 0.8, ds) * Math.exp(-dsp / 5.5) * 0.95;
      const env = smoothstep(1.0, 3.0, ds) * Math.exp(-Math.max(0, ds - 2) / 9);
      // (water fix 2) above the plunge the jet runs in as a smooth dark tongue (the riffle's crests die out on it);
      // along the white mass's edges, where the jet shears past the slack water, a short train of standing waves
      const tongueS = smoothstep(-4.6, -3.3, ds) * (1 - smoothstep(-0.9, -0.15, ds));
      const edgeS = smoothstep(-0.4, 0.4, ds) * (1 - smoothstep(3.0, 5.0, ds));
      const ew = (1 + Math.sin((2 * Math.PI * ds) / lamE + 0.6)) / 2, ew3 = ew * ew * ew;
      // the seams wander a little as they drift down the pool
      for (let n = 0; n < 2; n++) {
        const side = n === 0 ? -1 : 1;
        // (water fix 2: they wander and kink as they drift and run wider and fainter, strings of bubble rafts; thin
        // and nearly straight they read as hairline scratches across the pool, C2, W4, W5)
        seamC[n] = dc + side * (w0 * 0.95 + dsp * 0.2) + 0.45 * Math.sin(ds * 0.7 + side * 1.3) + 0.55 * fbm(ds * 0.3, side * 3.7, 2)
          + 0.1 * Math.sin(ds * 2.9 + side);
        seamW[n] = 0.17 + 0.02 * dsp;
        // strands break and reform as they drift
        seamG[n] = smoothstep(-0.1, 0.35, fbm(ds * 0.8 + 7.1, side * 2.3, 2));
      }
      for (let j = jLo[i]; j <= jHi[i]; j++) {
        const k = j * w + i;
        if (H[k] <= 0.03) continue;
        const d = d0 + j * texel;
        const u = (d - dc) / wj;
        if (tongueS > 0) {
          const ut = (d - dc) / (w0 * 0.95);
          const tg = tongueS * Math.exp(-ut * ut * 1.4);
          glassy[k] = Math.max(glassy[k], 0.95 * tg);
          froth[k] *= 1 - tg;
          crestW[k] *= 1 - tg;
          // smooth and slightly crowned: the riffle's chop and crests flatten out on it
          eta[k] = eta[k] * (1 - 0.85 * tg) + 0.02 * tg;
        }
        if (edgeS > 0) {
          const ue = Math.abs(u) - 1.05, ge = edgeS * Math.exp(-(ue * ue) / 0.06);
          if (ge > 0.01) {
            eta[k] += 0.035 * ge * (ew3 - P_MEAN);
            crestW[k] = Math.max(crestW[k], 0.75 * ew3 * ew3 * ge);
          }
        }
        // a heavy white mass right where it plunges, breaking up within a few metres (solid over its first ~2 m: at
        // mass x 0.95 only a core of ~0.5 m reached the heavy white water of the material)
        if (massS > 0) {
          const mass = massS * Math.exp(-u * u * 1.43);
          froth[k] = Math.max(froth[k], Math.min(0.95, mass * 1.7));
          glassy[k] *= 1 - mass;
        }
        // bubbles carried down under the surface: white, then pale turquoise as they sink and spread
        if (aerS > 0) aer[k] = Math.max(aer[k], aerS * Math.exp(-u * u * 0.64));
        // thin foam lines along both seams of the jet, drifting far down the pool
        if (env > 0.01) {
          let line = 0;
          for (let n = 0; n < 2; n++) {
            const side = n === 0 ? -1 : 1;
            const e1 = (d - seamC[n]) / seamW[n], e2 = (d - seamC[n] - side * 0.42) / (seamW[n] * 0.8);
            if (e1 * e1 < 16) line = Math.max(line, Math.exp(-e1 * e1) * seamG[n]);
            if (e2 * e2 < 16) line = Math.max(line, 0.45 * Math.exp(-e2 * e2) * (1 - seamG[n]));
          }
          if (line > 0) streak[k] = Math.max(streak[k], 0.75 * line * env);
        }
      }
    }
  }

  // ---- slopes from the height, rotated into world axes; packed straight into half floats (zeros stay zero)
  const ha = new Uint16Array(N * 4), hb = new Uint16Array(N * 4);
  const toH = THREE.DataUtils.toHalfFloat;
  for (let i = 1; i < w - 1; i++) {
    // fade everything out toward the ends of the grid (the generic terms take over there)
    const s = s0 + i * texel;
    const cov = smoothstep(s0, s0 + 3, s) * smoothstep(s1, s1 - 3, s);
    if (cov <= 0) continue;
    const covH = toH(cov);
    for (let j = 1; j < h - 1; j++) {
      const k = j * w + i, k4 = k * 4;
      hb[k4 + 3] = covH;
      if (H[k] <= 0) continue;
      const es = (eta[k + 1] - eta[k - 1]) / (2 * texel);
      const ed = (eta[k + w] - eta[k - w]) / (2 * texel);
      if (es !== 0 || ed !== 0) {
        ha[k4] = toH((es * tx[i] - ed * tz[i]) * cov);
        ha[k4 + 1] = toH((es * tz[i] + ed * tx[i]) * cov);
      }
      if (crestW[k] > 0) ha[k4 + 2] = toH(Math.min(1, crestW[k]) * cov);
      if (glassy[k] > 0) ha[k4 + 3] = toH(Math.min(1, glassy[k]) * cov);
      if (froth[k] > 0) hb[k4] = toH(Math.min(1, froth[k]) * cov);
      if (aer[k] > 0) hb[k4 + 1] = toH(Math.min(1, aer[k]) * cov);
      if (streak[k] > 0) hb[k4 + 2] = toH(Math.min(1, streak[k]) * cov);
    }
  }
  return {
    s0, s1: s0 + (w - 1) * texel, d0, d1: d0 + (h - 1) * texel, texel, w, h,
    texA: halfTexture(ha, w, h), texB: halfTexture(hb, w, h), plunge, ms: Math.round(performance.now() - t0),
  };
}

function halfTexture(data: Uint16Array, w: number, h: number): THREE.DataTexture {
  const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.HalfFloatType);
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = false;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.colorSpace = THREE.NoColorSpace;
  t.needsUpdate = true;
  return t;
}

/** TSL accessors at a river coordinate (s, d), i.e. the ribbon's uv. Texel i sits at s0 + i * texel. */
export function riffleNodes(r: RiffleField) {
  const lo = vec2(r.s0, r.d0), size = vec2(r.w, r.h);
  const uvAt = (sd: N) => sd.sub(lo).div(r.texel).add(0.5).div(size);
  return {
    A: (sd: N) => texture(r.texA, uvAt(sd)),
    B: (sd: N) => texture(r.texB, uvAt(sd)),
  };
}
