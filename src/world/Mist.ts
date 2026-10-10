import * as THREE from 'three/webgpu';
import {
  abs, dot, exp, float, Fn, getViewPosition, interleavedGradientNoise, lightShadowMatrix, Loop, max, min, mix,
  pow, screenCoordinate, screenSize, screenUV, smoothstep, step, texture, texture3D, uniform, vec2, vec3, vec4, type N,
} from '../gfx/tsl';
import type { SkyLighting } from './SkyLighting';
import type { CoreSampler, CorePoint } from './BankProps';
import type { CanopyField } from './ForestLayout';

/**
 * Morning mist and the near air, a half-resolution post pass before bloom: each pixel's view ray is marched from the
 * camera (or where it enters the layer) to the scene depth (the water surface writes depth), 12 jittered steps
 * crowding toward the camera. Two media:
 *  - the mist: a sheet lying on the water a few metres thick (its top wanders between ~0.6 and 2.6 m), thickest over
 *    the calm pool, thinner over fast water, a thin film over the low banks; drifting noise breaks it into sheets,
 *    gaps and wisps, and in places it lifts off as steam plumes that fray as they rise (ref_04, ref_05);
 *  - the forest air: a thin aerosol up to the crowns, lit only where the shadow map lets the sun through, so light
 *    comes down in shafts at the sun's angle and the shade between the trunks stays dark (the analytic valley haze
 *    in SkyLighting starts beyond this range).
 * Light: the low sun through the real shadow map with forward-scattering phases (warm toward the sun), plus sky light
 * where the canopy is open. Composited as scene x T + L. Clear right in front of the lens (the C2 close-up).
 */

export const mistU = {
  /** extinction at the sheet's base where the base map says 1 (1/m); part 8 scales it per time of day (scene fix: 0.072
   * when the sheet was 0.3-1.5 m deep, 0.2 for the thin 0.15-0.6 m one) */
  density: uniform(0.2),
  /** extinction of the thin sunlit air under the crowns (1/m) */
  air: uniform(0.0016),
  /** drift (m/s): down the valley, slowly rising */
  drift: uniform(new THREE.Vector3(0.22, 0.05, 0.04)),
  time: uniform(0),
  /** forward-scattering anisotropy of the mist and of the air */
  g: uniform(0.6),
  gAir: uniform(0.78),
  /** multipliers on the sun's and the sky's in-scatter (calibration) */
  sunGain: uniform(0.4),
  skyGain: uniform(1.0),
  /**
   * Colour hooks for part 8 (time of day): the in-scattered sunlight is the sun's colour desaturated by `sunSat`
   * (0 white .. 1 the sun's own colour) times `sunTint`; the sky part is tinted by `skyTint`. Morning: cream-white
   * to pale gold against the sun (round 2's sun x haze-warm product read as brown wood smoke over the dark forest).
   */
  sunSat: uniform(0.45),
  sunTint: uniform(new THREE.Color(1.0, 0.97, 0.9)),
  skyTint: uniform(new THREE.Color(1.0, 1.0, 1.0)),
  /** metres in front of the lens kept clear (round 2: 4.5-13 m, so the pool in front of C4 had no mist) */
  clearNear: uniform(1.2),
  clearFar: uniform(4.5),
  /** depth of the sheet: x its 0.15-0.6 m scale height (part 8: the dawn's river mist lies deeper) */
  thick: uniform(1.0),
  /** share of the steam plumes (part 8: at dawn they read as separate puffs from above) */
  plume: uniform(1.0),
};

const TOP = 30.0;
const TEX = { x0: -120, z0: -44, w: 232, h: 100, step: 1 };

/**
 * Per metre of the corridor: R floor height ((y + 2) / 12), G mist amount, B open sky above (1 - crown cover), A the
 * forest air (none over open water and the bar); zero amount on the border texels.
 */
function bakeBase(core: CoreSampler, canopy: CanopyField | null): THREE.DataTexture {
  const { w, h } = TEX;
  const data = new Uint8Array(w * h * 4);
  const cp: CorePoint = { waterY: 0, edge: 0, apron: 0, rock: 0, belowTop: 3, gravel: 0, y: 0 };
  const cv: [number, number] = [0, 0];
  const cover = new Float32Array(w * h);
  if (canopy) {
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) cover[j * w + i] = canopy.sample(TEX.x0 + i * TEX.step, TEX.z0 + j * TEX.step, cv)[0];
  }
  // the crowns shade the air a few metres around them: a 5 m box blur of the cover
  const blur = new Float32Array(w * h);
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    let s = 0, n = 0;
    for (let b = -2; b <= 2; b++) for (let a = -2; a <= 2; a++) {
      const ii = i + a, jj = j + b;
      if (ii < 0 || jj < 0 || ii >= w || jj >= h) continue;
      s += cover[jj * w + ii];
      n++;
    }
    blur[j * w + i] = s / n;
  }
  for (let j = 1; j < h - 1; j++) {
    for (let i = 1; i < w - 1; i++) {
      const x = TEX.x0 + i * TEX.step, z = TEX.z0 + j * TEX.step;
      let floor = 0, amt = 0, air = 0;
      if (core.at(x, z, cp)) {
        const water = cp.y < cp.waterY;
        floor = water ? cp.waterY : cp.y;
        const above = cp.y - cp.waterY;
        // the mist lies on the water and only laps onto the lowest margins (round 2 wip: a sheet up to 1.9 m over the
        // banks greyed the far bank face in C4 into one flat band; forest polish: it laps at most a metre onto them)
        const near = water ? 1 : Math.max(0, 1 - Math.max(cp.edge, 0) / 1.2) * Math.max(0, 1 - Math.max(above - 0.05, 0) / 0.35);
        // fast water stirs the air: less mist over the riffle
        const calm = cp.edge < 0 ? 1 - 0.45 * Math.min(1, Math.max(0, -cp.edge / 3)) : 1;
        // the shallows over the bars cooled most in the night and steam least (it also keeps C2's near water clear)
        const deep = water ? 0.12 + 0.88 * Math.min(1, Math.max(0, (cp.waterY - cp.y - 0.12) / 0.6)) : 0.12;
        amt = near * 0.9 * calm * deep;
        // (scene fix, critic polish r1: "the haze is densest in the forest on the right bank, at mid-height; the water
        // itself is clear": the air under the crowns no denser than over the water)
        air = water ? 0.45 : Math.min(1, 0.35 + 0.25 * Math.min(1, cp.edge / 8));
      }
      const k = (j * w + i) * 4;
      data[k] = Math.round(Math.min(1, Math.max(0, (floor + 2) / 12)) * 255);
      data[k + 1] = Math.round(Math.min(1, amt) * 255);
      data[k + 2] = Math.round(Math.max(0, 1 - blur[j * w + i] * 1.2) * 255);
      data[k + 3] = Math.round(air * 255);
    }
  }
  const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.needsUpdate = true;
  return t;
}

/**
 * Tileable 3D value noise (32³, two octaves in R and G, a third in B for the steam plumes): a texture fetch per
 * octave instead of procedural Perlin noise, whose inlined calls per march step cost seconds of shader compilation.
 */
function noiseVolume(): THREE.Data3DTexture {
  const N = 32;
  const lattice = (period: number, seed: number) => {
    const v = new Float32Array(period * period * period);
    let r = seed;
    for (let i = 0; i < v.length; i++) v[i] = ((r = (Math.imul(r, 1664525) + 1013904223) >>> 0) / 4294967296);
    return v;
  };
  const sample = (lat: Float32Array, period: number, x: number, y: number, z: number) => {
    const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
    const fx = x - xi, fy = y - yi, fz = z - zi;
    const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy), sz = fz * fz * (3 - 2 * fz);
    const at = (a: number, b: number, c: number) => lat[(((zi + c) % period) * period + ((yi + b) % period)) * period + ((xi + a) % period)];
    const l = (a: number, b: number, t: number) => a + (b - a) * t;
    return l(l(l(at(0, 0, 0), at(1, 0, 0), sx), l(at(0, 1, 0), at(1, 1, 0), sx), sy),
      l(l(at(0, 0, 1), at(1, 0, 1), sx), l(at(0, 1, 1), at(1, 1, 1), sx), sy), sz);
  };
  const A = lattice(8, 97), B = lattice(16, 211), C = lattice(4, 331);
  const data = new Uint8Array(N * N * N * 4);
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const k = ((z * N + y) * N + x) * 4;
    data[k] = Math.round(sample(A, 8, (x * 8) / N, (y * 8) / N, (z * 8) / N) * 255);
    data[k + 1] = Math.round(sample(B, 16, (x * 16) / N, (y * 16) / N, (z * 16) / N) * 255);
    data[k + 2] = Math.round(sample(C, 4, (x * 4) / N, (y * 4) / N, (z * 4) / N) * 255);
    data[k + 3] = 255;
  }
  const t = new THREE.Data3DTexture(data, N, N, N);
  t.format = THREE.RGBAFormat;
  t.type = THREE.UnsignedByteType;
  t.wrapS = t.wrapT = t.wrapR = THREE.RepeatWrapping;
  t.magFilter = t.minFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  return t;
}

/** the calm pool is where the mist lies thickest (x from -6 to 22 around z -3) */
const poolBoost = (p: N) => smoothstep(18.0, 7.0, vec2(p.x.sub(8.0).mul(0.7), p.z.add(3.0)).length()).mul(0.8).add(1);

export class Mist {
  private base: THREE.DataTexture;
  private vol = noiseVolume();
  /** comparison-ready placeholder until the sun's shadow map exists (it is created on the first shadow render) */
  private shadowTex: N;
  private ready = false;

  constructor(core: CoreSampler, private sky: SkyLighting, canopy: CanopyField | null, overrides?: string | null) {
    this.base = bakeBase(core, canopy);
    for (const kv of overrides?.split(',') ?? []) {
      const [k, v] = kv.split(':');
      const uu = (mistU as Record<string, { value: unknown }>)[k];
      if (uu && typeof uu.value === 'number') uu.value = Number(v);
    }
    const ph = new THREE.DepthTexture(1, 1);
    ph.type = THREE.FloatType;
    ph.compareFunction = THREE.GreaterEqualCompare;
    this.shadowTex = texture(ph);
  }

  /** swap in the real shadow map once it exists */
  update(t: number) {
    mistU.time.value = t;
    if (this.ready) return;
    const map = (this.sky.sun.shadow as unknown as { map: THREE.RenderTarget | null }).map;
    if (map?.depthTexture) {
      this.shadowTex.value = map.depthTexture;
      this.ready = true;
    }
  }

  /**
   * (L.rgb, T) of the mist in front of each pixel. `depth` is the scene pass's depth texture node; `camera` the main
   * camera (its matrices are read as uniforms: inside a post pass the built-in camera nodes are the quad's).
   */
  node(depth: N, camera: THREE.PerspectiveCamera): N {
    const u = this.sky.u;
    const base = this.base;
    const vol = this.vol;
    const projInv = uniform(camera.projectionMatrixInverse);
    const camWorld = uniform(camera.matrixWorld);
    const camPos = uniform(camera.position);
    const smat = lightShadowMatrix(this.sky.sun);
    const shadowTex = this.shadowTex;
    /** (sun visibility, inside the shadow map 0..1 with a soft border) */
    const sunVis = (p: N): N => {
      const s = smat.mul(vec4(p, 1));
      const c = s.xyz.div(s.w);
      const suv = vec2(c.x, c.y.oneMinus());
      const e = min(min(suv.x, suv.x.oneMinus()), min(suv.y, suv.y.oneMinus()));
      const inside = smoothstep(0.0, 0.08, e);
      return vec2(mix(float(1), shadowTex.sample(suv.clamp(0.0, 1.0)).compare(c.z.sub(0.0004)), step(0.0, e)), inside);
    };
    /**
     * (mist extinction, air extinction, open sky) over the march segment around p, whose ends lie `ya` and `yb` above
     * the floor there: the thin sheet's height falloff is integrated over the segment (a midpoint sample missed a
     * layer a few tens of centimetres thick on the long far segments).
     */
    const baseAt = (p: N): N => texture(base, p.xz.sub(vec2(TEX.x0, TEX.z0)).div(vec2(TEX.w * TEX.step, TEX.h * TEX.step)));
    const media = (p: N, b: N, ya: N, yb: N): N => {
      const floorY = b.x.mul(12).sub(2);
      const h = p.y.sub(floorY);
      const q = p.sub(vec3(mistU.drift).mul(mistU.time));
      // scene fix (critic polish r1: "thick, opaque cotton wool", "round puffs at bank height", "a uniform milky band"):
      // morning steam fog lies right on the water as a thin layer (scale height 0.15-0.6 m), in banks and thin spots
      // ~9 m across, combed into streaks along the river by the drift, frayed into wisps
      const n1 = texture3D(vol, vec3(q.x.mul(0.11), q.y.mul(0.5), q.z.mul(0.13)).div(8)).x;
      const nS = texture3D(vol, vec3(q.x.mul(0.045), q.y.mul(0.9), q.z.mul(0.6)).div(8).add(0.61)).y;
      const n2 = texture3D(vol, vec3(q.x.mul(0.42), q.y.mul(1.6), q.z.mul(0.5)).div(8).add(0.37)).y;
      const bank = smoothstep(0.3, 0.66, n1);
      const streak = smoothstep(0.36, 0.72, nS.mul(0.8).add(n2.mul(0.2)));
      const wisp = smoothstep(0.3, 0.72, n2.mul(0.75).add(n1.mul(0.35)));
      const hs = n1.mul(0.45).add(0.15).mul(mistU.thick);
      const ha = max(ya, 0.0), hb = max(yb, 0.0);
      const ea = exp(ha.div(hs).negate()), eb = exp(hb.div(hs).negate());
      const dh = hb.sub(ha);
      const avg = mix(exp(max(h, 0.0).div(hs).negate()), ea.sub(eb).mul(hs).div(dh.add(1e-5)), step(0.02, abs(dh)));
      const sheet = avg.mul(bank.mul(0.95).add(0.08)).mul(streak.mul(0.88).add(0.12)).mul(wisp.mul(0.7).add(0.3));
      // steam lifting off the water in slow columns 0.5-1.5 m across that curl round as they rise, lean downwind and
      // thin out within a few metres (round 2's 3 m plumes reached up into the bank's trees: the haze over the forest)
      const lean = vec3(mistU.drift.x, 0, mistU.drift.z).mul(h.mul(2.0));
      const curlA = h.mul(0.9).add(mistU.time.mul(0.07));
      const curl = vec3(curlA.cos(), 0, curlA.sin()).mul(h.mul(0.22));
      const pq = p.sub(vec3(mistU.drift).mul(mistU.time.mul(1.4))).sub(lean).add(curl);
      const colN = texture3D(vol, vec3(pq.x.mul(0.36), pq.y.mul(0.08).sub(mistU.time.mul(0.006)), pq.z.mul(0.36)).div(4).add(0.21)).z;
      // (h held under 6 m: past ~10 m these edges would cross; the plume is gone by then)
      const hF = h.clamp(0.0, 6.0);
      const fray = smoothstep(hF.mul(0.08).add(0.3), hF.mul(0.05).add(0.62), n2);
      const plume = smoothstep(0.64, 0.8, colN).mul(exp(h.div(1.5).negate())).mul(smoothstep(0.0, 0.3, h)).mul(fray);
      const mist = b.y.mul(sheet.add(plume.mul(mistU.plume.mul(0.6)))).mul(step(-0.05, h)).mul(poolBoost(p)).mul(mistU.density);
      const air = b.w.mul(float(1).sub(smoothstep(12.0, TOP, h))).mul(step(-0.05, h)).mul(mistU.air);
      return vec3(mist, air, b.z);
    };
    return Fn(() => {
      // the texel's top-left full-resolution pixel, so the depth-aware filter (postFilters.ts) knows which depth it saw
      const uvD = screenUV.sub(vec2(0.25).div(screenSize));
      const d = depth.sample(uvD).x;
      const vp = getViewPosition(uvD, d, projInv);
      const wp = camWorld.mul(vec4(vp, 1)).xyz;
      const toP = wp.sub(camPos);
      const dist = toP.length();
      const dir = toP.div(dist.max(1e-4));
      // the march: from the camera (or where the ray comes down into the layer) to the scene, at most 140 m
      const t0 = max(float(0), camPos.y.sub(TOP).div(dir.y.negate().max(1e-3)).mul(step(TOP, camPos.y)));
      // rays climbing out of the layer stop where they leave it
      const tTop = mix(float(1e4), float(TOP).sub(camPos.y).div(dir.y.max(1e-3)), step(1e-3, dir.y));
      const t1 = min(min(dist, tTop), float(140));
      const len = max(t1.sub(t0), 0.0);
      const STEPS = 12;
      const jitter = interleavedGradientNoise(screenCoordinate.xy);
      const T = float(1).toVar();
      const L = vec3(0).toVar();
      const cosT = dot(dir, u.sunDir);
      const hg = (g: N) => float(1).sub(g.mul(g)).div(pow(float(1).add(g.mul(g)).sub(g.mul(cosT).mul(2)), 1.5)).mul(1 / (4 * Math.PI));
      const sc = vec3(u.sunColor);
      const sunCol = mix(vec3(dot(sc, vec3(0.2126, 0.7152, 0.0722))), sc, mistU.sunSat).mul(vec3(mistU.sunTint));
      const sun = sunCol.mul(u.sunIntensity).mul(mistU.sunGain);
      const sunM = sun.mul(hg(mistU.g).mul(0.85).add(0.15 / (4 * Math.PI)));
      // (forest polish: a smaller isotropic floor, so the forest air lights up in shafts toward the sun and no longer
      // veils the trunks looking away from it, C1's north bank)
      const sunA = sun.mul(hg(mistU.gAir).mul(0.9).add(0.015 / (4 * Math.PI)));
      const skyE = vec3(u.horizon).mul(0.55).add(vec3(u.zenith).mul(0.6)).mul(1 / Math.PI).mul(mistU.skyGain).mul(vec3(mistU.skyTint));
      Loop(STEPS, ({ i }: { i: N }) => {
        const a = float(i).add(jitter).div(STEPS);
        const a2 = float(i).add(1).add(jitter).div(STEPS);
        const ta = t0.add(len.mul(a.mul(a)));
        const tb = t0.add(len.mul(a2.mul(a2)));
        const p = camPos.add(dir.mul(ta.add(tb).mul(0.5)));
        const near = smoothstep(mistU.clearNear, mistU.clearFar, ta);
        // heights of the segment's ends over the floor under its middle
        const bm = baseAt(p);
        const fy = bm.x.mul(12).sub(2);
        const m = media(p, bm, camPos.y.add(dir.y.mul(ta)).sub(fy), camPos.y.add(dir.y.mul(tb)).sub(fy));
        const sv = sunVis(p);
        // outside the shadow map the air's sun visibility is unknown: no air there (the analytic haze carries on)
        const sigM = m.x.mul(near), sigA = m.y.mul(sv.y).mul(near);
        const sig = sigM.add(sigA).max(1e-6);
        const tr = exp(sig.mul(tb.sub(ta)).negate());
        const lit = sunM.mul(sv.x).add(skyE.mul(m.z.mul(0.8).add(0.2))).mul(sigM).add(sunA.mul(sv.x).mul(sigA)).div(sig);
        L.addAssign(T.mul(lit).mul(float(1).sub(tr)));
        T.mulAssign(tr);
      });
      return vec4(L, T);
    })();
  }
}
