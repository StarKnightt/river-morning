import * as THREE from 'three/webgpu';
import {
  abs, cross, dot, exp, float, floor, fwidth, hash, If, length, max, min, mix, mx_noise_float, normalize, pow, select,
  sign, sin, smoothstep, sqrt, step, uniform, vec2, vec3, type N,
} from '../gfx/tsl';

/**
 * Night sky on top of the HDRI: crisp procedural stars (the 4k plate's stars are soft 1-2 texel blobs, ~3x
 * magnified at 1080p) and the moon disc with its aureole. Drawn in the scene background node, so the pool
 * mirror (which renders the background from the mirrored camera) reflects them too.
 */
export const nightU = {
  /** 0 by day, 1 at night: scales stars and moon */
  amount: uniform(0),
  /** star brightness (pre-exposed night units: the night sky is ~0.01-0.04). (Part 8 round 2, critic r1: "an empty,
   * flat smoky grey with a handful of faint specks": 1.0 -> 2.2, more faint ones outside the Milky Way) */
  starGain: uniform(2.2),
  /** toward the moon, unit, world space */
  moonDir: uniform(new THREE.Vector3(0, 0.45, 0.89).normalize()),
  /** angular radius (rad): 0.43 deg, a little larger than the real 0.26 so it reads at 1080p */
  moonRadius: uniform(0.0075),
  /** radiance of the lit disc (blooms: the bloom threshold is 2.2) */
  moonRadiance: uniform(new THREE.Color(28, 30, 33)),
  /** moonlit sky around the disc (aureole) */
  moonGlow: uniform(new THREE.Color(0.05, 0.065, 0.09)),
  /** frozen in shot mode: set from the engine clock */
  time: uniform(0),
};

/** radians per pixel at 1080p and a 50 deg vertical field of view: the brightness reference for the stars */
const PIX_REF = ((50 * Math.PI) / 180) / 1080;

/** fixed tilt of the star cube so its faces do not line up with the valley axes (rows of stars along a seam) */
const tilt = (d: N) => {
  const a = 0.61, b = 0.37;
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  const x1 = d.x.mul(ca).sub(d.z.mul(sa)), z1 = d.x.mul(sa).add(d.z.mul(ca));
  const y2 = d.y.mul(cb).sub(z1.mul(sb)), z2 = d.y.mul(sb).add(z1.mul(cb));
  return vec3(x1, y2, z2);
};

/**
 * One layer of stars: one candidate per cell of a cube map with n x n cells a face, kept inside the middle
 * 60 % of its cell (no neighbour lookups needed: a star is a few pixels wide, a cell ~90). Gaussian of about a
 * pixel, peak scaled with (PIX_REF / pixel)^2 so a star keeps its flux at any resolution (the half-resolution
 * mirror draws it twice as wide and a quarter as bright).
 */
function starLayer(d: N, wy: N, pix: N, n: number, seed: number, prob: N, magPow: number, gain: number): N {
  const a = abs(d);
  const isX = a.x.greaterThanEqual(a.y).and(a.x.greaterThanEqual(a.z));
  const isY = isX.not().and(a.y.greaterThanEqual(a.z));
  const major = select(isX, a.x, select(isY, a.y, a.z));
  const fuv = select(isX, vec2(d.z, d.y), select(isY, vec2(d.x, d.z), vec2(d.x, d.y))).div(major);
  const sgn = select(isX, sign(d.x), select(isY, sign(d.y), sign(d.z)));
  const face = select(isX, sgn.mul(0.5).add(0.5), select(isY, sgn.mul(0.5).add(2.5), sgn.mul(0.5).add(4.5)));
  const g = fuv.mul(0.5).add(0.5).mul(n);
  const cell = floor(g);
  const id = face.mul(n * n).add(cell.y.mul(n)).add(cell.x).mul(8).add(seed * 8 * 6 * n * n);
  const r1 = hash(id), r2 = hash(id.add(1)), r3 = hash(id.add(2)), r4 = hash(id.add(3)), r5 = hash(id.add(4));
  const sp = cell.add(vec2(r1, r2).mul(0.6).add(0.2)).div(n).mul(2).sub(1);
  const s3 = select(isX, vec3(sgn, sp.y, sp.x), select(isY, vec3(sp.x, sgn, sp.y), vec3(sp.x, sp.y, sgn)));
  const ang = length(d.sub(normalize(s3)));
  const present = step(float(1).sub(prob), r3);
  const mag = pow(r4, magPow);
  const sig = pix.mul(0.62).add(mag.mul(PIX_REF * 0.35));
  const prof = exp(ang.mul(ang).div(sig.mul(sig).mul(-2)));
  const res = min(float(PIX_REF).div(pix).pow(2), 2.0);
  // scintillation: faster and deeper toward the horizon (more air)
  const tw = sin(nightU.time.mul(r5.mul(7).add(4)).add(r1.mul(6.283))).mul(mix(0.12, 0.45, smoothstep(0.6, 0.0, wy))).add(1);
  // colour temperature: warm K stars to blue-white B stars, faint ones nearly white
  const col = mix(vec3(1.0, 0.8, 0.62), vec3(0.74, 0.84, 1.0), r2).mix(vec3(1), smoothstep(0.3, 0.0, mag).mul(0.6));
  return col.mul(present.mul(mag).mul(gain).mul(prof).mul(res).mul(tw));
}

/**
 * Stars along a world direction (unit). `pixAngle`: angle of one pixel (rad), from the direction's screen
 * derivatives. `milky`: 0..1 brightness of the Milky Way in the HDRI there (more faint stars inside it).
 */
export function starField(dir: N, milky: N): N {
  const d = tilt(dir);
  const pix = max(length(fwidth(dir)), PIX_REF * 0.25);
  const wy = abs(dir.y);
  const bright = starLayer(d, wy, pix, 34, 1, float(0.62), 7.0, 3.2);
  const mid = starLayer(d, wy, pix, 70, 2, float(0.4), 3.2, 0.5);
  const faint = starLayer(d, wy, pix, 130, 3, mix(float(0.24), float(0.6), milky), 2.0, 0.09);
  // extinction near the horizon; nothing below it (the mirror sees only the upper hemisphere anyway)
  const ext = smoothstep(-0.01, 0.22, dir.y);
  return bright.add(mid).add(faint).mul(ext).mul(nightU.starGain).mul(nightU.amount);
}

/**
 * The moon: a full disc with soft maria and a slight limb darkening, anti-aliased over a pixel, plus the
 * moonlit aureole around it (a narrow and a wide lobe). Stars fade out in the glare (returned as `.w`).
 */
export function moonSky(dir: N): { color: N; starFade: N } {
  const m = nightU.moonDir;
  const v = dir.sub(m);
  const ang = length(v);
  // toVar: emitted here, in uniform control flow (fwidth), not inside the per-pixel disc branch below
  const pix = max(length(fwidth(dir)), PIX_REF * 0.25).toVar();
  const right = normalize(cross(m, vec3(0, 1, 0)));
  const up = cross(right, m);
  const p = vec2(dot(v, right), dot(v, up)).div(nightU.moonRadius);
  const r = length(p);
  const aa = pix.div(nightU.moonRadius).max(0.02);
  const disc = smoothstep(float(1).add(aa), float(1).sub(aa), r).toVar();
  // the maria noise only on and right around the disc (no derivatives in here, so the branch may diverge)
  const discCol = vec3(0).toVar();
  If(r.lessThan(float(1).add(aa)), () => {
    const mu = sqrt(max(float(1).sub(r.mul(r)), 0));
    const maria = mx_noise_float(vec3(p.mul(1.7), 2.3)).add(mx_noise_float(vec3(p.mul(4.1), 7.9)).mul(0.45));
    const albedo = mix(float(1.0), float(0.6), smoothstep(-0.05, 0.4, maria)).mul(mu.mul(0.18).add(0.82));
    discCol.assign(vec3(nightU.moonRadiance).mul(albedo).mul(disc));
  });
  const glow = vec3(nightU.moonGlow).mul(exp(ang.div(-0.018)).mul(2.2).add(exp(ang.div(-0.16)).mul(0.6)));
  const fade = smoothstep(0.0, 0.25, ang);
  return { color: discCol.add(glow).mul(nightU.amount), starFade: mix(float(1), fade, nightU.amount) };
}
