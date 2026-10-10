import * as THREE from 'three/webgpu';
import { river, newSample } from '../world/River';
import { fbm, smoothstep } from '../world/noise';
import { min, smoothstep as tslSmoothstep, texture, vec2, type N } from '../gfx/tsl';

/**
 * Top-down river field baked once at load over the corridor (world XZ, 0.25 m texels), shared by the
 * water surface, the bed/rock caustics, particles and CPU queries so they all agree on the current.
 *
 *  A (RGBA16F): flow x, flow z (m/s), foam trail 0..1, water surface y (m)   <- the only texture receivers sample
 *  B (RGBA16F): bed y (m), section speed (m/s), bed bump (m, + over humps), surface relief (m: water piled up
 *               against rocks, the trough and V-shaped ridges behind them; the water normals use its gradient)
 *
 * Foam trails are the strongest whitewater source within ~5.6 m upstream (Waterways' foam pass).
 */
export interface RiverField {
  x0: number;
  z0: number;
  x1: number;
  z1: number;
  texel: number;
  w: number;
  h: number;
  a: Float32Array;
  b: Float32Array;
  texA: THREE.DataTexture;
  texB: THREE.DataTexture;
}

/** A rock in or beside the channel. */
export interface Obstacle {
  x: number;
  z: number;
  /** footprint radius (m) */
  r: number;
  /** highest point (m) */
  topY: number;
}

export interface FieldSample {
  flowX: number;
  flowZ: number;
  foam: number;
  waterY: number;
  bedY: number;
  speed: number;
}

export function bakeRiverField(o: {
  x0: number; z0: number; x1: number; z1: number; texel?: number;
  groundY: (x: number, z: number) => number;
  obstacles?: Obstacle[];
}): RiverField {
  const texel = o.texel ?? 0.25;
  const w = Math.round((o.x1 - o.x0) / texel) + 1, h = Math.round((o.z1 - o.z0) / texel) + 1;
  const N = w * h;
  const a = new Float32Array(N * 4), b = new Float32Array(N * 4);
  const bed = new Float32Array(N), src = new Float32Array(N), seamSrc = new Float32Array(N);
  const dx = new Float32Array(N), dz = new Float32Array(N);
  const S = newSample();

  for (let j = 0; j < h; j++) {
    const z = o.z0 + j * texel;
    for (let i = 0; i < w; i++) {
      const x = o.x0 + i * texel, k = j * w + i;
      river.sample(x, z, S);
      const gy = o.groundY(x, z);
      const depth = S.waterY - gy;
      // lateral distribution (Manning-like, v ~ h^2/3): the thalweg runs fastest, shallows and margins crawl
      const rel = depth > 0 ? Math.min(1.35, Math.max(0.06, Math.pow(depth / Math.max(0.6 * S.depth, 0.05), 0.67))) : 0;
      const v = S.speed * rel;
      a[k * 4] = S.tx * v;
      a[k * 4 + 1] = S.tz * v;
      a[k * 4 + 3] = S.waterY;
      b[k * 4] = gy;
      b[k * 4 + 1] = S.speed;
      bed[k] = gy;
      dx[k] = S.tx;
      dz[k] = S.tz;
      // the seam where the riffle pours into the pool: the jet still runs fast as the bed drops away, and its
      // bubbles are carried far down the slack water (rocks are the other source; open gravel makes none)
      if (depth > 0.2) {
        const u = Math.abs(S.d) / S.half;
        const seam = smoothstep(1.55, 1.1, S.speed) * smoothstep(0.55, 0.9, S.speed) * smoothstep(0.3, 0.8, depth)
          * smoothstep(0.75, 0.25, u) * (0.45 + 0.55 * smoothstep(-0.2, 0.4, fbm(x * 0.9, z * 0.9, 2)));
        seamSrc[k] = seam * 0.55;
      }
    }
  }

  // bed bumps: the bed minus a 1.5 m box blur, positive over humps (drives chop and standing-wave foam)
  const smooth = boxBlur(bed, w, h, 6);
  for (let k = 0; k < N; k++) b[k * 4 + 2] = a[k * 4 + 3] > bed[k] ? Math.max(0, bed[k] - smooth[k]) : 0;

  for (const ob of o.obstacles ?? []) stampObstacle(ob, o.x0, o.z0, texel, w, h, a, b, src);

  // foam trails: the strongest rock source within ~5.6 m upstream, fading with distance; the jet's bubbles drift
  // ~14 m down the pool, thinning out
  let si0 = w, si1 = -1, sj0 = h, sj1 = -1;
  for (let k = 0; k < N; k++) {
    if (seamSrc[k] > 0) {
      const i = k % w, j = (k - i) / w;
      si0 = Math.min(si0, i); si1 = Math.max(si1, i); sj0 = Math.min(sj0, j); sj1 = Math.max(sj1, j);
    }
  }
  const reach = Math.ceil(14 / texel);
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      const k = j * w + i;
      let f = 0;
      for (let n = 0; n < 16; n++) {
        const d = (n * 0.35) / texel;
        f = Math.max(f, bilinear(src, w, h, i - dx[k] * d, j - dz[k] * d) * (1 - n / 16) ** 1.3);
      }
      if (i >= si0 - reach && i <= si1 + reach && j >= sj0 - reach && j <= sj1 + reach) {
        for (let n = 0; n < 28; n++) {
          const d = (n * 0.5) / texel;
          f = Math.max(f, bilinear(seamSrc, w, h, i - dx[k] * d, j - dz[k] * d) * (1 - n / 28) ** 1.6);
        }
      }
      a[k * 4 + 2] = Math.min(1, f);
    }
  }

  return {
    x0: o.x0, z0: o.z0, x1: o.x0 + (w - 1) * texel, z1: o.z0 + (h - 1) * texel, texel, w, h, a, b,
    texA: halfTexture(a, w, h), texB: halfTexture(b, w, h),
  };
}

/**
 * Potential flow around a cylinder of radius R (u = U(1 - R²/r² cos 2θ), v = -U R²/r² sin 2θ in the flow
 * frame), a dome-shaped bed, foam ringing emergent rocks, a short wake and a standing wave just downstream
 * of rocks that sit a few centimetres under the surface.
 */
function stampObstacle(ob: Obstacle, x0: number, z0: number, texel: number, w: number, h: number,
  a: Float32Array, b: Float32Array, src: Float32Array) {
  const R = ob.r, reachM = 3 * R + 1.5;
  const i0 = Math.max(0, Math.floor((ob.x - reachM - x0) / texel)), i1 = Math.min(w - 1, Math.ceil((ob.x + reachM - x0) / texel));
  const j0 = Math.max(0, Math.floor((ob.z - reachM - z0) / texel)), j1 = Math.min(h - 1, Math.ceil((ob.z + reachM - z0) / texel));
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i <= i1; i++) {
      const k = j * w + i;
      const rx = x0 + i * texel - ob.x, rz = z0 + j * texel - ob.z, r = Math.hypot(rx, rz);
      const waterY = a[k * 4 + 3];
      if (r < R) b[k * 4] = Math.max(b[k * 4], ob.topY - R * (1 - Math.sqrt(1 - (r / R) ** 2)) * 0.8);
      const fx = a[k * 4], fz = a[k * 4 + 1], U = Math.hypot(fx, fz);
      let along = 0, across = 0;
      if (U > 1e-3) {
        const ux = fx / U, uz = fz / U;
        along = rx * ux + rz * uz;
        across = -rx * uz + rz * ux;
        if (r < R) {
          a[k * 4] = 0;
          a[k * 4 + 1] = 0;
        } else {
          const th = Math.atan2(across, along), q = (R * R) / (r * r);
          const va = U * (1 - q * Math.cos(2 * th)), vc = -U * q * Math.sin(2 * th);
          const fade = 1 - smoothstep(2 * R, 3 * R, r);
          a[k * 4] = fx + (va * ux - vc * uz - fx) * fade;
          a[k * 4 + 1] = fz + (va * uz + vc * ux - fz) * fade;
        }
      }
      const emerge = smoothstep(-0.05, 0.03, ob.topY - waterY);
      const nearTop = smoothstep(-0.22, -0.05, ob.topY - waterY) * (1 - emerge);
      const fast = smoothstep(0.25, 1.15, U);
      // even the gentle current at the pool head (~0.6 m/s) leaves a V and a wake behind an emergent rock
      const fastV = smoothstep(0.15, 0.8, U);
      const up = along < 0 ? -along / Math.max(r, 1e-3) : 0, down = along > 0 ? along / Math.max(r, 1e-3) : 0;
      // white collar all round the waterline, heaviest where the water piles against the upstream face
      const ring = emerge * smoothstep(R + 0.14, R - 0.02, r) * (0.3 + 0.45 * up) * (0.45 + 0.55 * fast);
      // a V of bubbly streaks peeling off both flanks and spreading downstream, and a short white wake
      const arm = Math.abs(Math.abs(across) - (R * 0.75 + along * 0.32));
      // strong enough to stand out of the pool-head seam's own bubble trail
      const vee = along > 0 ? emerge * fastV * Math.exp(-(arm * arm) / 0.012) * smoothstep(5 * R + 1.8, R, along) * 0.7 : 0;
      const wake = along > 0 ? emerge * fastV * smoothstep(R * 0.7, R * 0.2, Math.abs(across)) * smoothstep(2.5 * R + 0.4, R, along) * 0.6 : 0;
      const wave = along > 0 ? nearTop * fast * smoothstep(1.4 * R, 0.4 * R, Math.hypot(along - 0.6 * R, across)) * 0.45 : 0;
      // (water fix 2) nothing in the slack pool: a log or branch lying in 0.2-0.35 m/s water makes no froth; its collar,
      // marched downstream as a trail, drew straight white hairlines across the pool beside the fallen spruce (C4)
      const live = smoothstep(0.3, 0.5, b[k * 4 + 1]);
      src[k] = Math.max(src[k], Math.max(ring, vee, wake, wave) * live);
      // surface relief: water heaps up against the upstream face (stagnation head U²/2g, damped), drops into a
      // trough behind the rock and rises in two ridges along the V; a submerged rock lifts a standing wave
      if (r > R * 0.9) {
        const head = (U * U) / (2 * 9.81) * 0.45;
        const near = Math.exp(-(((r - R) / (0.22 + 0.3 * R)) ** 2));
        const pillow = emerge * head * near * up ** 1.5;
        const trough = -emerge * head * 0.6 * Math.exp(-(((r - R) / (0.35 + 0.4 * R)) ** 2)) * down;
        const ridge = along > 0 ? emerge * fastV * 0.025 * Math.exp(-(arm * arm) / 0.03) * smoothstep(5 * R + 1.8, R, along) : 0;
        const stand = nearTop * head * 0.8 * (along > 0 ? Math.exp(-(((Math.hypot(along - 0.8 * R, across)) / (0.6 * R + 0.2)) ** 2)) : 0);
        b[k * 4 + 3] += pillow + trough + ridge + stand;
      }
    }
  }
}

function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      let s = 0, c = 0;
      for (let d = -r; d <= r; d++) {
        const x = i + d;
        if (x >= 0 && x < w) { s += src[j * w + x]; c++; }
      }
      tmp[j * w + i] = s / c;
    }
  }
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      let s = 0, c = 0;
      for (let d = -r; d <= r; d++) {
        const y = j + d;
        if (y >= 0 && y < h) { s += tmp[y * w + i]; c++; }
      }
      out[j * w + i] = s / c;
    }
  }
  return out;
}

function bilinear(src: Float32Array, w: number, h: number, fi: number, fj: number): number {
  const i = Math.max(0, Math.min(w - 2, Math.floor(fi))), j = Math.max(0, Math.min(h - 2, Math.floor(fj)));
  const tx = Math.max(0, Math.min(1, fi - i)), ty = Math.max(0, Math.min(1, fj - j));
  const k = j * w + i;
  const top = src[k] + (src[k + 1] - src[k]) * tx, bot = src[k + w] + (src[k + w + 1] - src[k + w]) * tx;
  return top + (bot - top) * ty;
}

function halfTexture(src: Float32Array, w: number, h: number): THREE.DataTexture {
  const data = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) data[i] = THREE.DataUtils.toHalfFloat(src[i]);
  const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.HalfFloatType);
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = false;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.colorSpace = THREE.NoColorSpace;
  t.needsUpdate = true;
  return t;
}

/** CPU lookup (bobber, lure, fish AI, splash drift). Returns false outside the baked rectangle. */
export function sampleField(f: RiverField, x: number, z: number, out: FieldSample): boolean {
  const fi = (x - f.x0) / f.texel, fj = (z - f.z0) / f.texel;
  if (fi < 0 || fj < 0 || fi > f.w - 1 || fj > f.h - 1) return false;
  const i = Math.min(f.w - 2, Math.floor(fi)), j = Math.min(f.h - 2, Math.floor(fj));
  const tx = fi - i, ty = fj - j;
  const lerp4 = (arr: Float32Array, c: number) => {
    const k = (j * f.w + i) * 4 + c, kw = f.w * 4;
    const top = arr[k] + (arr[k + 4] - arr[k]) * tx, bot = arr[k + kw] + (arr[k + kw + 4] - arr[k + kw]) * tx;
    return top + (bot - top) * ty;
  };
  out.flowX = lerp4(f.a, 0);
  out.flowZ = lerp4(f.a, 1);
  out.foam = lerp4(f.a, 2);
  out.waterY = lerp4(f.a, 3);
  out.bedY = lerp4(f.b, 0);
  out.speed = lerp4(f.b, 1);
  return true;
}

/** TSL accessors. Texel i sits at x0 + i * texel, i.e. at uv (i + 0.5) / w. */
export function fieldNodes(f: RiverField) {
  const lo = vec2(f.x0, f.z0), hi = vec2(f.x1, f.z1), size = vec2(f.w, f.h);
  const uvAt = (xz: N) => xz.sub(lo).div(f.texel).add(0.5).div(size);
  return {
    A: (xz: N) => texture(f.texA, uvAt(xz)),
    B: (xz: N) => texture(f.texB, uvAt(xz)),
    /** 1 inside the baked rectangle, fading to 0 over its outer metre (the ribbon's vertex flow takes over) */
    inside: (xz: N) => {
      const d = min(xz.sub(lo), hi.sub(xz));
      return tslSmoothstep(0, 1, min(d.x, d.y));
    },
  };
}
