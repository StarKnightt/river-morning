import * as THREE from 'three/webgpu';
import { fbm, smoothstep } from './noise';

/**
 * The river is a centreline polyline with attributes. Flow runs west (-X) to east (+X).
 * Signed lateral distance d > 0 is the south bank (+Z), where the player, dock and cabin are.
 * The bend bulges north around the pool, so the deep water hugs the north (outer) bank and
 * the south (inner) bank is a gravel point bar.
 */

// x, z, halfWidth, depth (at thalweg), waterY, speed (m/s), asym bias (thalweg offset, -1 north .. +1 south).
// The channel meanders inside a smoother valley floor: bends alternate (pool bend bulging north, then
// south above the riffle and below the pool), so point bars and cut banks swap sides along the river.
const CTRL: number[][] = [
  // far ends keep the valley continuous well past the view (the projection must never clamp nearby)
  [-760, 60.0, 4.6, 0.9, 6.0, 1.0, 0.0],
  [-400, 32.0, 4.4, 0.85, 3.6, 1.0, 0.0],
  [-220, 13.0, 4.2, 0.8, 2.2, 1.0, 0.0],
  [-150, 3.0, 4.6, 0.9, 1.75, 0.9, 0.0],
  [-112, -5.0, 5.2, 1.1, 1.45, 0.8, 0.0],
  [-86, -5.5, 4.4, 0.8, 1.22, 1.0, 0.0],
  [-64, 1.5, 3.8, 0.6, 1.0, 1.3, 0.0],
  [-47, 8.0, 4.6, 1.3, 0.86, 0.9, 0.0],
  [-34, 7.5, 3.4, 0.5, 0.66, 1.6, 0.0],
  [-24, 4.6, 3.0, 0.38, 0.42, 2.1, 0.0],
  [-14, 1.6, 3.6, 0.5, 0.16, 1.9, 0.0],
  [-4, -1.2, 6.0, 1.5, 0.03, 0.6, 0.0],
  [6, -3.3, 7.3, 3.0, 0.0, 0.25, 0.14],
  [16, -3.1, 7.5, 3.0, 0.0, 0.25, 0.14],
  [26, -1.6, 5.6, 1.2, -0.04, 0.7, 0.0],
  [38, 2.8, 4.2, 0.6, -0.18, 1.3, 0.0],
  [52, 7.6, 4.0, 0.7, -0.34, 1.1, 0.0],
  [66, 8.6, 4.6, 1.0, -0.48, 0.9, 0.0],
  [84, 4.0, 4.2, 0.7, -0.66, 1.1, 0.0],
  [104, -2.5, 4.4, 0.8, -0.85, 1.0, 0.0],
  [130, -3.5, 4.4, 0.8, -1.1, 1.0, 0.0],
  [220, 8.0, 4.4, 0.85, -1.7, 1.0, 0.0],
  [420, 26.0, 4.6, 0.9, -3.0, 1.0, 0.0],
  [780, 40.0, 4.8, 0.9, -5.4, 1.0, 0.0],
];

/** Smooth valley axis (x, z): the walls follow this, the channel wanders across the floor between them. */
const VALLEY: number[][] = [
  [-760, 60], [-400, 32], [-220, 13], [-140, 3], [-80, 0], [-30, 2.5], [10, 1.5], [60, 3], [110, 1],
  [220, 8], [420, 26], [780, 40],
];

const ATTRS = 6; // half, depth, waterY, speed, asym, curvature

export interface RiverSample {
  /** arc length along the centreline (m) */
  s: number;
  /** signed lateral distance from the centreline (m), + = south */
  d: number;
  half: number;
  depth: number;
  waterY: number;
  speed: number;
  asym: number;
  /** signed curvature (1/m): > 0 when the river turns toward the south, i.e. the south bank is the inner bank */
  curv: number;
  /** unit flow direction in XZ */
  tx: number;
  tz: number;
}

class RiverPath {
  readonly xs: Float32Array;
  readonly zs: Float32Array;
  readonly ss: Float32Array;
  readonly attrs: Float32Array;
  readonly count: number;
  readonly length: number;

  constructor() {
    const curve = new THREE.CatmullRomCurve3(
      CTRL.map((c) => new THREE.Vector3(c[0], 0, c[1])),
      false,
      'centripetal',
    );
    const n = 2400;
    this.count = n + 1;
    this.xs = new Float32Array(this.count);
    this.zs = new Float32Array(this.count);
    this.ss = new Float32Array(this.count);
    this.attrs = new Float32Array(this.count * ATTRS);
    const p = new THREE.Vector3();
    let s = 0;
    for (let i = 0; i <= n; i++) {
      const u = i / n;
      curve.getPoint(u, p);
      if (i > 0) s += Math.hypot(p.x - this.xs[i - 1], p.z - this.zs[i - 1]);
      this.xs[i] = p.x;
      this.zs[i] = p.z;
      this.ss[i] = s;
      // attributes: smooth interpolation between control points in parameter space
      const f = u * (CTRL.length - 1);
      const k = Math.min(CTRL.length - 2, Math.floor(f));
      const t = smoothstep(0, 1, f - k);
      for (let a = 0; a < 5; a++) {
        this.attrs[i * ATTRS + a] = CTRL[k][2 + a] + (CTRL[k + 1][2 + a] - CTRL[k][2 + a]) * t;
      }
    }
    this.length = s;

    // signed curvature d(heading)/ds, smoothed over about +-10 m so bars follow whole bends, not wiggles
    const head = new Float32Array(this.count);
    for (let i = 0; i < this.count; i++) {
      const a = Math.max(0, i - 1), b = Math.min(this.count - 1, i + 1);
      head[i] = Math.atan2(this.zs[b] - this.zs[a], this.xs[b] - this.xs[a]);
    }
    const raw = new Float32Array(this.count);
    for (let i = 0; i < this.count; i++) {
      const a = Math.max(0, i - 4), b = Math.min(this.count - 1, i + 4);
      let dh = head[b] - head[a];
      if (dh > Math.PI) dh -= 2 * Math.PI;
      if (dh < -Math.PI) dh += 2 * Math.PI;
      raw[i] = dh / Math.max(1e-3, this.ss[b] - this.ss[a]);
    }
    for (let i = 0; i < this.count; i++) {
      let sum = 0, wsum = 0;
      for (let j = Math.max(0, i - 40); j <= Math.min(this.count - 1, i + 40); j++) {
        const ds = (this.ss[j] - this.ss[i]) / 10;
        const w = Math.exp(-ds * ds);
        sum += raw[j] * w;
        wsum += w;
      }
      const k = sum / wsum;
      this.attrs[i * ATTRS + 5] = k;
      // the thalweg swings toward the outer bank of each bend
      const j = i * ATTRS + 4;
      this.attrs[j] = Math.max(-0.6, Math.min(0.6, this.attrs[j] - k * 16));
    }
  }

  /** nearest polyline index by x (the centreline is monotonic in x) */
  private indexForX(x: number) {
    let lo = 0, hi = this.count - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (this.xs[mid] < x) lo = mid; else hi = mid;
    }
    return lo;
  }

  /** water surface height at a given x (clamped to the ends), for far-field approximations */
  waterYAt(x: number): number {
    if (x <= this.xs[0]) return this.attrs[2];
    if (x >= this.xs[this.count - 1]) return this.attrs[(this.count - 1) * ATTRS + 2];
    const i = this.indexForX(x);
    const t = (x - this.xs[i]) / (this.xs[i + 1] - this.xs[i]);
    return this.attrs[i * ATTRS + 2] + (this.attrs[(i + 1) * ATTRS + 2] - this.attrs[i * ATTRS + 2]) * t;
  }

  /** centreline z at a given x (clamped to the ends), for far-field approximations */
  zAt(x: number): number {
    if (x <= this.xs[0]) return this.zs[0];
    if (x >= this.xs[this.count - 1]) return this.zs[this.count - 1];
    const i = this.indexForX(x);
    const t = (x - this.xs[i]) / (this.xs[i + 1] - this.xs[i]);
    return this.zs[i] + (this.zs[i + 1] - this.zs[i]) * t;
  }

  sample(x: number, z: number, out: RiverSample): RiverSample {
    const i0 = this.indexForX(x);
    const a = Math.max(0, i0 - 140), b = Math.min(this.count - 2, i0 + 140);
    let best = Infinity, bi = a, bt = 0;
    for (let i = a; i <= b; i++) {
      const x0 = this.xs[i], z0 = this.zs[i];
      const dx = this.xs[i + 1] - x0, dz = this.zs[i + 1] - z0;
      const l2 = dx * dx + dz * dz;
      let t = ((x - x0) * dx + (z - z0) * dz) / l2;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const px = x0 + dx * t - x, pz = z0 + dz * t - z;
      const d2 = px * px + pz * pz;
      if (d2 < best) { best = d2; bi = i; bt = t; }
    }
    const x0 = this.xs[bi], z0 = this.zs[bi];
    let tx = this.xs[bi + 1] - x0, tz = this.zs[bi + 1] - z0;
    const tl = Math.hypot(tx, tz);
    tx /= tl; tz /= tl;
    const cx = x0 + (this.xs[bi + 1] - x0) * bt, cz = z0 + (this.zs[bi + 1] - z0) * bt;
    // right of flow (+X flow -> +Z) is positive
    const side = (x - cx) * -tz + (z - cz) * tx;
    out.d = side >= 0 ? Math.sqrt(best) : -Math.sqrt(best);
    // beyond the ends the projection clamps; extend the valley straight along the tangent
    out.s = this.ss[bi] + tl * bt;
    const A = this.attrs;
    const j = bi * ATTRS, k = (bi + 1) * ATTRS;
    out.half = (A[j] + (A[k] - A[j]) * bt) * (1 + fbm(out.s * 0.045 + 11.0, 2.5, 3) * 0.16);
    out.depth = A[j + 1] + (A[k + 1] - A[j + 1]) * bt;
    out.waterY = A[j + 2] + (A[k + 2] - A[j + 2]) * bt;
    out.speed = A[j + 3] + (A[k + 3] - A[j + 3]) * bt;
    out.asym = A[j + 4] + (A[k + 4] - A[j + 4]) * bt;
    out.curv = A[j + 5] + (A[k + 5] - A[j + 5]) * bt;
    out.tx = tx;
    out.tz = tz;
    return out;
  }
}

export const river = new RiverPath();

/** The valley axis as a z(x) lookup (monotonic in x). */
class ValleyAxis {
  private xs: Float32Array;
  private zs: Float32Array;
  constructor() {
    const curve = new THREE.CatmullRomCurve3(VALLEY.map((c) => new THREE.Vector3(c[0], 0, c[1])), false, 'centripetal');
    const n = 1600;
    this.xs = new Float32Array(n + 1);
    this.zs = new Float32Array(n + 1);
    const p = new THREE.Vector3();
    for (let i = 0; i <= n; i++) {
      curve.getPoint(i / n, p);
      this.xs[i] = p.x;
      this.zs[i] = p.z;
    }
  }

  zAt(x: number): number {
    const xs = this.xs, n = xs.length;
    if (x <= xs[0]) return this.zs[0];
    if (x >= xs[n - 1]) return this.zs[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] < x) lo = mid; else hi = mid;
    }
    const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    return this.zs[lo] + (this.zs[hi] - this.zs[lo]) * t;
  }
}

export const valley = new ValleyAxis();

export function newSample(): RiverSample {
  return { s: 0, d: 0, half: 1, depth: 0, waterY: 0, speed: 0, asym: 0, curv: 0, tx: 1, tz: 0 };
}

/** Pool water level; the planar reflection plane sits here. */
export const POOL_LEVEL = 0.0;
