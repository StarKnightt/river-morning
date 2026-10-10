import { clamp, exp, float, floor, int, max, min, rtt, screenUV, select, sqrt, uniform, vec2, vec3, vec4, type N } from './tsl';

/** proportional to view depth for the reversed depth buffer (near 0.05 m, far 20 km); the sky (0) is the farthest */
const lin = (d: N): N => float(1).div(d.add(2.5e-6));
/** relative depth difference of two `lin` values: 0 the same depth, 1 one twice as far as the other */
const relDepth = (a: N, b: N): N => max(a, b).div(min(a, b)).sub(1);

/** the old Gaussian's spread (three's gaussianBlur sigma 1.6 = 2.07 texels), in half-resolution texels */
const SIGMA = 2.07;
/** relative depth differences that count as an edge in the blur and in the upsample */
const BLUR_DEPTH = 0.25, UP_DEPTH = 0.1;
/** transmittance difference that counts as an edge (the jittered march's grain is smaller) */
const BLUR_T = 0.16;

/**
 * The half-resolution mist (Mist.ts: in-scattered light rgb, transmittance a) brought to full resolution without
 * spreading it across edges. A Gaussian of 4 px and a bilinear upsample smeared every mist edge 8-12 px both ways:
 * nothing on a near bank hundreds of pixels tall, most of a distant one (tree silhouettes against the hazy valley,
 * the far bank rising out of the river's mist sheet). Here a separable bilateral blur at half resolution, whose
 * weights fall with depth and transmittance differences, still hides the jittered march, and a 2x2 joint-bilateral
 * upsample against the full-resolution depth keeps silhouettes to the pixel.
 * `depth` is the scene pass's depth texture; Mist.ts samples it at each texel's top-left full-resolution pixel.
 */
export function mistFilter(mistRT: N, depth: N): N {
  const blur = (src: N, dx: number, dy: number): N => {
    const hs = vec2(src.size(int(0)));
    const q = vec2(0.25).div(hs);
    const step = vec2(dx, dy).div(hs);
    // the RTT node itself (not a sampled copy): that is what makes it render, before the pass that reads it
    const c = src;
    const zc = lin(depth.sample(screenUV.sub(q)).x);
    let sum: N = c;
    let wsum: N = float(1);
    for (let k = -4; k <= 4; k++) {
      if (k === 0) continue;
      const uk = screenUV.add(step.mul(k));
      const s = src.sample(uk);
      const r = relDepth(lin(depth.sample(uk.sub(q)).x), zc).div(BLUR_DEPTH);
      const dT = s.a.sub(c.a).div(BLUR_T);
      const w = float(Math.exp(-(k * k) / (2 * SIGMA * SIGMA))).div(r.mul(r).add(1)).mul(exp(dT.mul(dT).negate()));
      sum = sum.add(s.mul(w));
      wsum = wsum.add(w);
    }
    return sum.div(wsum);
  };
  const h = rtt(blur(mistRT, 1, 0), null, null, { resolutionScale: 0.5 });
  const v = rtt(blur(h, 0, 1), null, null, { resolutionScale: 0.5 });
  const hs = vec2(v.size(int(0)));
  const q = vec2(0.25).div(hs);
  const t = screenUV.mul(hs).sub(0.5);
  const i0 = floor(t);
  const f = t.sub(i0);
  const zc = lin(depth.sample(screenUV).x);
  // a whisper of the plain bilinear sample keeps the sum defined (and, read directly, makes `v` render)
  let sum: N = v.mul(1e-4);
  let wsum: N = float(1e-4);
  for (const [ox, oy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
    const uk = i0.add(vec2(ox, oy)).add(0.5).div(hs);
    const r = relDepth(lin(depth.sample(uk.sub(q)).x), zc).div(UP_DEPTH);
    const wb = (ox ? f.x : f.x.oneMinus()).mul(oy ? f.y : f.y.oneMinus());
    // a pixel unlike all four (a trunk thinner than a texel) takes the nearest in depth
    const w = wb.add(0.02).div(r.mul(r).add(1));
    sum = sum.add(v.sample(uk).mul(w));
    wsum = wsum.add(w);
  }
  return sum.div(wsum);
}

/**
 * Contrast-adaptive sharpening of the display image (the algorithm of AMD FidelityFX CAS, MIT): a negative-lobe
 * cross filter whose strength falls where the 3x3 neighbourhood is already near black or white, so edges do not ring
 * and flat areas stay clean. `amount` 0..1, live in `sharpenU` (0 passes the image through).
 */
export const sharpenU = uniform(0);

export function sharpen(tex: N, amount: number): N {
  sharpenU.value = amount;
  const px = vec2(1).div(vec2(tex.size(int(0))));
  const at = (x: number, y: number): N => tex.sample(screenUV.add(px.mul(vec2(x, y)))).rgb;
  const e = at(0, 0), b = at(0, -1), d = at(-1, 0), f = at(1, 0), h = at(0, 1);
  const a = at(-1, -1), c = at(1, -1), g = at(-1, 1), i = at(1, 1);
  const mnX = min(min(min(d, e), min(f, b)), h);
  const mxX = max(max(max(d, e), max(f, b)), h);
  const mn = mnX.add(min(min(min(mnX, a), min(c, g)), i));
  const mx = mxX.add(max(max(max(mxX, a), max(c, g)), i));
  const amp = sqrt(clamp(min(mn, vec3(2).sub(mx)).div(mx.max(1e-4)), 0.0, 1.0));
  const w = amp.mul(select(sharpenU.greaterThan(0), float(-1).div(float(8).sub(sharpenU.mul(3))), float(0)));
  return vec4(clamp(b.add(d).add(f).add(h).mul(w).add(e).div(w.mul(4).add(1)), 0.0, 1.0), 1);
}
