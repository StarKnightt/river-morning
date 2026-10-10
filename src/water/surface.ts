import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, abs, atan, cos, dFdx, dFdy, dot, exp, float, floor, fract, int, length, mix, mx_noise_float, mx_noise_vec3, sin,
  smoothstep, sqrt, step, uniform, uniformArray, vec2, vec3, vec4, type N,
} from '../gfx/tsl';

/**
 * The analytic part of the water surface, shared by the water normals, the bed/rock caustics and the CPU
 * height query, so the light patterns on the bed always move with the ripples above them.
 *
 *  - 12 flow ripples (5.7-42 cm) with directions spread around downstream (+x) and 8 capillary ripples
 *    (2.6-4.6 cm, every direction), advected by the local current with a two-phase flow map (CatLikeCoding
 *    FlowUVW as used in Arnklit_Waterways' river shader).
 *  - Flow ripples follow an equal-curvature rule (a·k² equal per band, slope capped, riffle RMS curvature
 *    ~7 1/m) and fade to 3 % on the calm pool. Capillary ripples never go away (45 % on calm water): they
 *    shatter the sun into sparkles, give the calm pool a faint wobble and focus the fine caustic net in the
 *    shallows (RMS curvature ~22 1/m, so even 20 cm of calm water is past the focus).
 *  - Three wave groups are scaled by slowly drifting noise, so the sum never settles into a repeating pattern.
 *  - Rings from splashes and other disturbances (analytic, 32 events: swells, ripple trains, boils, swirls),
 *    drifting with the current.
 * Directions are fixed in world space; the flow map carries the pattern along the bends.
 */

export const FLOW_PERIOD = 1.25;

/** [direction (deg from +x, i.e. downstream), wavelength (m), weight]: N_MAIN flow ripples, then capillaries */
export const WAVES: Array<[number, number, number]> = [
  [4, 0.31, 1.0], [-23, 0.19, 0.9], [31, 0.13, 0.85], [-47, 0.088, 0.8], [58, 0.24, 0.6], [-71, 0.15, 0.55],
  [12, 0.071, 0.7], [-8, 0.42, 0.5], [83, 0.11, 0.45], [-36, 0.057, 0.6], [44, 0.36, 0.4], [-60, 0.27, 0.35],
  [17, 0.046, 1.0], [-101, 0.039, 0.9], [133, 0.033, 0.85], [-152, 0.042, 0.8], [71, 0.029, 0.75],
  [-29, 0.035, 0.9], [168, 0.026, 0.7], [-74, 0.031, 0.8],
];
export const N_MAIN = 12;
const RIFFLE_CURVATURE = 7;
const SLOPE_CAP = 0.09;
const CAP_CURVATURE = 22;
const CAP_SLOPE = 0.1;

export const WAVE_K = WAVES.map(([, l]) => (2 * Math.PI) / l);
export const WAVE_DIR = WAVES.map(([deg]) => [Math.cos((deg * Math.PI) / 180), Math.sin((deg * Math.PI) / 180)] as const);
// capillary-gravity dispersion, slowed: river ripples mostly ride the current rather than run across it
export const WAVE_OMEGA = WAVE_K.map((k) => 0.6 * Math.sqrt(9.81 * k + 7.3e-5 * k ** 3));
const rms = (ws: number[]) => Math.sqrt(ws.reduce((s, w) => s + w * w, 0) / 2);
const wMain = rms(WAVES.slice(0, N_MAIN).map((w) => w[2])), wCap = rms(WAVES.slice(N_MAIN).map((w) => w[2]));
export const WAVE_AMP = WAVES.map(([, , w], i) => i < N_MAIN
  ? Math.min((RIFFLE_CURVATURE * w) / wMain / WAVE_K[i] ** 2, SLOPE_CAP / WAVE_K[i])
  : Math.min((CAP_CURVATURE * w) / wCap / WAVE_K[i] ** 2, CAP_SLOPE / WAVE_K[i]));

export const surfaceU = {
  /** engine time (s); frozen in shot mode so captures repeat exactly */
  time: uniform(0),
  /** flow-map cycles, wrapped on the CPU (600 is a whole number of jump periods) */
  flowCycle: uniform(0),
  /** per-wave phase -ω t, wrapped to [0, 2π) on the CPU in double precision */
  wavePhase: WAVES.map(() => uniform(0)),
  /** chop floor on the calm pool: a light breeze */
  poolChop: uniform(0.05),
};

export function updateSurfaceUniforms(t: number) {
  surfaceU.time.value = t;
  surfaceU.flowCycle.value = (t / FLOW_PERIOD) % 600;
  for (let i = 0; i < WAVES.length; i++) surfaceU.wavePhase[i].value = -((WAVE_OMEGA[i] * t) % (2 * Math.PI));
}

/** Per-place phase offset so the two flow phases never reset in sync across the river. */
export const phaseNoise = (xz: N) => mx_noise_float(vec3(xz.mul(0.31), 1.7)).mul(0.5).add(0.5);

/** 0 glassy .. 1 broken riffle, from the local current only (the caustic receivers must get the same value). */
export const chopFromFlow = (flow: N) => smoothstep(0.25, 1.6, length(flow)).max(surfaceU.poolChop);
export const chopAmp = (chop: N) => mix(0.03, 1.0, chop);

/** Amplitude scales of the wave set at a point: flow ripples, capillaries, and the three group gains. */
export interface Amps {
  main: N;
  cap: N;
  gains: N;
}

/** Group gains 0.35..1.65 from slowly drifting noise (sampled at the advected position, so patches ride the current). */
export const waveGains = (p: N) => smoothstep(-0.45, 0.55, mx_noise_vec3(vec3(p.mul(0.42), 3.1))).mul(1.3).add(0.35);

export const ampsFor = (chop: N, gains: N): Amps => ({ main: chopAmp(chop), cap: mix(0.45, 1.0, chop), gains });

export interface Phase {
  pos: N;
  w: N;
}

/** One phase of the two-phase flow map: the advected position (m) and its blend weight. */
export function advect(xz: N, flow: N, noise: N, offset: 0 | 0.5): Phase {
  const tau = surfaceU.flowCycle.add(noise).add(offset);
  const p = fract(tau);
  // the jump only changes at the reset, where this phase's weight is zero
  const jump = floor(tau).mod(24);
  const pos = xz.sub(flow.mul(FLOW_PERIOD).mul(p.sub(0.5))).add(vec2(0.25, 5 / 24).mul(jump)).add(offset * 0.731);
  return { pos, w: float(1).sub(abs(p.mul(2).sub(1))) };
}

/**
 * Waves fade out where the pixel (or the sun disc) blurs them: each keeps 1 - smoothstep(lo λ, hi λ, x).
 * The water uses x = 2 x footprint, (0.3, 0.7); the caustics x = max(sun-disc blur, footprint), (0.25, 0.6).
 * Unrolled (12 copies): measured ~0.3-1 ms faster per frame than a uniform-array loop, at the same load time.
 */
const keep = (lambda: number, x: N, lo: number, hi: number) => float(1).sub(smoothstep(lambda * lo, lambda * hi, x));

/** amplitude scale of wave i: its band (flow ripple or capillary) times its group's drifting gain */
const ampOf = (a: Amps, i: number): N => (i < N_MAIN ? a.main : a.cap).mul([a.gains.x, a.gains.y, a.gains.z][i % 3]);

/**
 * Slope (dη/dx, dη/dz) of the analytic waves at an advected position. η_i = a_i sin(k_i d_i·p + φ_i).
 * `count` limits the sum to the first waves (the water's normals use the flow ripples only: as visible pixels the
 * regular capillary sinusoids alias into lines, so the surface takes its fine ripples from the irregular tile).
 */
export function waveSlope(p: N, a: Amps, x: N, lo: number, hi: number, count = WAVES.length): N {
  let sx: N = float(0), sz: N = float(0);
  WAVES.slice(0, count).forEach(([, lambda], i) => {
    const k = WAVE_K[i], [dx, dz] = WAVE_DIR[i];
    const ak = ampOf(a, i).mul(WAVE_AMP[i] * k).mul(keep(lambda, x, lo, hi));
    const c = cos(p.x.mul(k * dx).add(p.y.mul(k * dz)).add(surfaceU.wavePhase[i]));
    sx = sx.add(ak.mul(c).mul(dx));
    sz = sz.add(ak.mul(c).mul(dz));
  });
  return vec2(sx, sz);
}

/**
 * Hessian (hxx, hxz, hzz) of the analytic waves in xyz, and in w c2 = Σ (a k² · dᵀQd)², the per-wave
 * caustic strength used to normalise brightness (Underwater.ts). Q is (Qxx, Qxz, Qzx, Qzz).
 */
export function waveHessian(p: N, a: Amps, x: N, lo: number, hi: number, Q: N): N {
  let hxx: N = float(0), hxz: N = float(0), hzz: N = float(0), c2: N = float(0);
  WAVES.forEach(([, lambda], i) => {
    const k = WAVE_K[i], [dx, dz] = WAVE_DIR[i];
    const ak2 = ampOf(a, i).mul(WAVE_AMP[i] * k * k).mul(keep(lambda, x, lo, hi));
    const m = ak2.mul(sin(p.x.mul(k * dx).add(p.y.mul(k * dz)).add(surfaceU.wavePhase[i]))).negate();
    hxx = hxx.add(m.mul(dx * dx));
    hxz = hxz.add(m.mul(dx * dz));
    hzz = hzz.add(m.mul(dz * dz));
    const cq = ak2.mul(Q.x.mul(dx * dx).add(Q.y.add(Q.z).mul(dx * dz)).add(Q.w.mul(dz * dz)));
    c2 = c2.add(cq.mul(cq));
  });
  return vec4(hxx, hxz, hzz, c2);
}

// ---------------------------------------------------------------------------------------------------------------
// Rings: analytic packets from anything that disturbs the surface (a lure or float landing, a bite, a fish rising,
// boiling or swirling under the surface, the line cutting the water, hands held in the water)

export const MAX_EVENTS = 32;
/** x, z, start time (s), amplitude (m) */
export const eventA = Array.from({ length: MAX_EVENTS }, () => new THREE.Vector4(0, 0, -1e5, 0));
/** drift x, z (m/s, the current where it landed), life (s), splash foam radius (m; negative: white water, denser froth) */
export const eventB = Array.from({ length: MAX_EVENTS }, () => new THREE.Vector4(0, 0, 1, 0));
/**
 * Shape: kind (RING_KIND), size (m: a train's wavelength, a boil's or swirl's radius; unused by the swell), speed
 * (m/s: a train's front, a boil's spread), start radius of a train's front (m) or a swirl's spin (+-1).
 */
export const eventC = Array.from({ length: MAX_EVENTS }, () => new THREE.Vector4(0, 0, 0, 0));
/**
 * swell: one soft rounded crest with a shallower trough (splashes); train: a crisp train of capillary ripples, the
 * rings a small object pushes out; boil: a smooth dome of upwelling water with a disturbed rim, spreading as it
 * flattens (a fish turning under the surface); swirl: a vortex dimple with spiral arms of broken water.
 */
export const RING_KIND = { swell: 0, train: 1, boil: 2, swirl: 3 } as const;
export const eventsU = {
  a: uniformArray(eventA, 'vec4'),
  b: uniformArray(eventB, 'vec4'),
  c: uniformArray(eventC, 'vec4'),
  /** slots 0..live-1 may hold a ring that is alive now: the shaders loop over those only (usually none) */
  live: uniform(0, 'int'),
};

/** Called every frame after the events changed or time advanced. */
export function updateLiveEvents(t: number) {
  let live = 0;
  for (let i = 0; i < MAX_EVENTS; i++) if (t - eventA[i].z < eventB[i].z) live = i + 1;
  eventsU.live.value = live;
}

/**
 * A swell is one soft, rounded crest (a Gaussian) with a shallower trough behind it, running out at 0.42 m/s and
 * widening as it spreads. Its two flanks tilt opposite ways, so one catches a bright line of sky and the other a
 * dark line; its curvature focuses a caustic ring on the bed below.
 */
const SWELL = { speed: 0.42, width: 0.075, spread: 0.075, trough: 0.4, lag: 1.8 };

const ringEnvelope = (A: N, B: N, tau: N, r: N) =>
  A.w.mul(exp(tau.negate().div(B.z.mul(0.4)))).mul(sqrt(float(0.12).div(r.add(0.12))));

/** d/dr and d²/dr² of the swell profile (crest minus trailing trough), without the envelope */
const swellProfile = (r: N, tau: N) => {
  const w = tau.mul(SWELL.spread).add(SWELL.width);
  const w2 = w.mul(w);
  const x1 = r.sub(tau.mul(SWELL.speed)), x2 = x1.add(w.mul(SWELL.lag));
  const g1 = exp(x1.mul(x1).div(w2).negate()), g2 = exp(x2.mul(x2).div(w2).negate()).mul(SWELL.trough);
  const d1 = x1.mul(g1).sub(x2.mul(g2)).mul(-2).div(w2);
  const d2 = x1.mul(x1).mul(4).div(w2).sub(2).mul(g1).sub(x2.mul(x2).mul(4).div(w2).sub(2).mul(g2)).div(w2);
  return { d1, d2 };
};

/**
 * A train's wavelength stretches as the ripples disperse; its front runs out at the event's speed; the packet holds
 * about two crests (the leading one tallest) and widens slowly. Shared by the shader and the CPU height.
 */
const TRAIN = { stretch: 0.35, width: 1.1, widen: 0.015, decay: 0.45, fade: 40 };

/**
 * Slope (xy), splash foam (z) and the churned or roughened water (w: 0..1, the water treats it as broken water) of
 * every live ring at xz. The foam is a scatter of bubbles that thins out from the middle. A train's crests finer
 * than about two pixels along their radius would alias into noise: they fade into a band of roughened water, which
 * is how a distant ring reads (a ring of duller water in the mirror, catching a few glints).
 */
export const ringField = Fn(([xz]: [N]) => {
  const out = vec4(0, 0, 0, 0).toVar();
  const dX = dFdx(xz).toVar(), dY = dFdy(xz).toVar();
  Loop({ start: int(0), end: eventsU.live, type: 'int' }, ({ i }: { i: N }) => {
    const A = eventsU.a.element(i), B = eventsU.b.element(i), C = eventsU.c.element(i);
    const tau = surfaceU.time.sub(A.z);
    If(tau.greaterThan(0).and(tau.lessThan(B.z)), () => {
      const rel = xz.sub(A.xy.add(B.xy.mul(tau)));
      const r = length(rel).max(1e-3);
      const n = rel.div(r);
      If(C.x.lessThan(0.5), () => {
        const env = ringEnvelope(A, B, tau, r);
        const d1 = swellProfile(r, tau).d1;
        const radius = abs(B.w).max(1e-3);
        const has = step(1e-3, abs(B.w));
        // a negative radius is white water (a trout thrashing on top): half again the froth
        const dense = mix(float(0.5), float(0.75), step(B.w, -1e-3));
        // (falling edges as 1 - smoothstep: reversed edges are undefined in WGSL and returned 0 on this GPU, so no
        // splash ever showed its foam or its churned water)
        const foam = float(1).sub(smoothstep(radius.mul(0.1), radius, r)).mul(exp(tau.negate().div(B.z.mul(0.15)))).mul(has).mul(dense);
        const churn = float(1).sub(smoothstep(radius.mul(0.3), radius.mul(1.6), r)).mul(exp(tau.negate().div(B.z.mul(0.3)))).mul(has);
        out.addAssign(vec4(n.mul(d1.mul(env)), foam, churn));
      }).ElseIf(C.x.lessThan(1.5), () => {
        const lam = C.y.mul(tau.mul(TRAIN.stretch).add(1));
        const front = C.w.add(C.z.mul(tau));
        const W = lam.mul(TRAIN.width).add(tau.mul(TRAIN.widen));
        If(r.lessThan(front.add(W.mul(1.6))).and(r.greaterThan(front.sub(W.mul(3.6)))), () => {
          const kk = float(2 * Math.PI).div(lam);
          const x = r.sub(front), xc = x.add(W);
          const g = exp(xc.mul(xc).div(W.mul(W)).negate());
          const amp = A.w.mul(exp(tau.negate().div(B.z.mul(TRAIN.decay)))).mul(sqrt(float(0.1).div(r.add(0.1))))
            .mul(float(1).sub(exp(tau.mul(-TRAIN.fade))));
          const ph = x.mul(kk);
          const d = amp.mul(g).mul(kk.mul(sin(ph)).negate().sub(xc.mul(2).div(W.mul(W)).mul(cos(ph))));
          const fpR = abs(dot(n, dX)).add(abs(dot(n, dY)));
          const keepT = float(1).sub(smoothstep(lam.mul(0.3), lam.mul(0.65), fpR));
          const far = smoothstep(lam.mul(0.45), lam.mul(0.9), fpR);
          out.addAssign(vec4(n.mul(d.mul(keepT)), 0, amp.mul(kk).mul(g).mul(far).mul(3.0).min(0.7)));
        });
      }).ElseIf(C.x.lessThan(2.5), () => {
        const W = C.y.add(C.z.mul(tau));
        If(r.lessThan(W.mul(2.6)), () => {
          const decay = exp(tau.negate().div(B.z.mul(0.4)));
          const g = exp(r.mul(r).div(W.mul(W)).negate());
          const d = A.w.mul(decay).mul(float(1).sub(exp(tau.mul(-12)))).mul(g).mul(r.mul(-2).div(W.mul(W)));
          const rim = r.sub(W.mul(1.2)).div(W.mul(0.35));
          out.addAssign(vec4(n.mul(d), 0, exp(rim.mul(rim).negate()).mul(decay).mul(0.4)));
        });
      }).Else(() => {
        const W = C.y.mul(tau.mul(0.3).add(1));
        If(r.lessThan(W.mul(2.4)), () => {
          const decay = exp(tau.negate().div(B.z.mul(0.45)));
          const g = exp(r.mul(r).div(W.mul(W)).negate());
          // a dimple (eta = -A g), its slope turned part way round the vortex by the spin
          const d = A.w.mul(decay).mul(g).mul(r.mul(2).div(W.mul(W)));
          const t = vec2(n.y.negate(), n.x);
          const arms = cos(atan(rel.y, rel.x).mul(2).sub(r.div(W).mul(C.w.mul(5))).add(tau.mul(C.w.mul(5)))).mul(0.5).add(0.5);
          const churn = smoothstep(0.5, 0.95, arms).mul(exp(r.mul(r).div(W.mul(W).mul(3)).negate())).mul(decay).mul(0.55);
          out.addAssign(vec4(n.mul(d).add(t.mul(abs(d).mul(C.w).mul(0.9))), 0, churn));
        });
      });
    });
  });
  return out;
});

/**
 * Hessian (hxx, hxz, hzz) of the live swells and boils at xz, for the caustics under them. Trains and swirls are
 * left out: their short ripples come to a focus within centimetres and are blurred out by any depth worth drawing.
 */
export const ringHessian = Fn(([xz]: [N]) => {
  const H = vec3(0, 0, 0).toVar();
  Loop({ start: int(0), end: eventsU.live, type: 'int' }, ({ i }: { i: N }) => {
    const A = eventsU.a.element(i), B = eventsU.b.element(i), C = eventsU.c.element(i);
    const tau = surfaceU.time.sub(A.z);
    If(tau.greaterThan(0).and(tau.lessThan(B.z)), () => {
      const rel = xz.sub(A.xy.add(B.xy.mul(tau)));
      const r = length(rel).max(1e-3);
      // radial profile: H = eta_rr n nT + (eta_r / r)(I - n nT)
      const n = rel.div(r);
      const nxx = n.x.mul(n.x), nzz = n.y.mul(n.y), nxz = n.x.mul(n.y);
      If(C.x.lessThan(0.5), () => {
        const env = ringEnvelope(A, B, tau, r);
        const prof = swellProfile(r, tau);
        const t = prof.d1.div(r);
        H.addAssign(vec3(prof.d2.mul(nxx).add(t.mul(nzz)), prof.d2.sub(t).mul(nxz), prof.d2.mul(nzz).add(t.mul(nxx))).mul(env));
      }).ElseIf(C.x.greaterThan(1.5).and(C.x.lessThan(2.5)), () => {
        const W = C.y.add(C.z.mul(tau));
        const W2 = W.mul(W);
        const e = A.w.mul(exp(tau.negate().div(B.z.mul(0.4)))).mul(float(1).sub(exp(tau.mul(-12)))).mul(exp(r.mul(r).div(W2).negate()));
        const d1 = e.mul(r.mul(-2).div(W2)), d2 = e.mul(r.mul(r).mul(4).div(W2).sub(2).div(W2));
        const t = d1.div(r);
        H.addAssign(vec3(d2.mul(nxx).add(t.mul(nzz)), d2.sub(t).mul(nxz), d2.mul(nzz).add(t.mul(nxx))));
      });
    });
  });
  return H;
});

/** CPU ring height at (x, z), same packets as the shader (bobber and lure ride it). */
export function ringHeight(x: number, z: number, t: number): number {
  let y = 0;
  for (let i = 0; i < MAX_EVENTS; i++) {
    const A = eventA[i], B = eventB[i], C = eventC[i], tau = t - A.z;
    if (tau <= 0 || tau >= B.z) continue;
    const r = Math.max(1e-3, Math.hypot(x - (A.x + B.x * tau), z - (A.y + B.y * tau)));
    if (C.x < 0.5) {
      const env = A.w * Math.exp(-tau / (B.z * 0.4)) * Math.sqrt(0.12 / (r + 0.12));
      const w = SWELL.width + SWELL.spread * tau;
      const x1 = r - SWELL.speed * tau, x2 = x1 + SWELL.lag * w;
      y += env * (Math.exp(-(x1 * x1) / (w * w)) - SWELL.trough * Math.exp(-(x2 * x2) / (w * w)));
    } else if (C.x < 1.5) {
      const lam = C.y * (1 + TRAIN.stretch * tau), front = C.w + C.z * tau, W = lam * TRAIN.width + tau * TRAIN.widen;
      const xx = r - front, xc = xx + W;
      if (xx > W * 1.6 || xx < -W * 3.6) continue;
      const amp = A.w * Math.exp(-tau / (B.z * TRAIN.decay)) * Math.sqrt(0.1 / (r + 0.1)) * (1 - Math.exp(-tau * TRAIN.fade));
      y += amp * Math.exp(-(xc * xc) / (W * W)) * Math.cos((2 * Math.PI * xx) / lam);
    } else if (C.x < 2.5) {
      const W = C.y + C.z * tau;
      y += A.w * Math.exp(-tau / (B.z * 0.4)) * (1 - Math.exp(-tau * 12)) * Math.exp(-(r * r) / (W * W));
    } else {
      const W = C.y * (1 + 0.3 * tau);
      y -= A.w * Math.exp(-tau / (B.z * 0.45)) * Math.exp(-(r * r) / (W * W));
    }
  }
  return y;
}
