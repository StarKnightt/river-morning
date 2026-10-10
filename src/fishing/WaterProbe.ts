import type { RiverWater } from '../water/RiverWater';
import type { RiverField } from '../water/RiverField';
import { river, newSample } from '../world/River';
import { MAX_EVENTS, eventA, ringHeight, surfaceU } from '../water/surface';

export interface WaterPoint {
  /** surface height (m), with splash rings when a ring time was given */
  y: number;
  flowX: number;
  flowZ: number;
  /** section speed of the river here (m/s): riffle ~1.5-2, pool 0.25 */
  speed: number;
  /** water depth (m); <= 0 on dry ground */
  depth: number;
  /** rendered ground / river bed height (m) */
  groundY: number;
  /** depth > 2 cm */
  wet: boolean;
}

/**
 * Allocation-free water lookups for the float, line points and fish: reads the baked river field directly
 * (RiverWater.query does the same through a closure per call), falls back to the river polyline outside it or
 * with ?water=0. Splashes go through the water's own event API.
 */
export class WaterProbe {
  private field: RiverField | null;
  private rs = newSample();

  constructor(readonly water: RiverWater | null, readonly groundY: (x: number, z: number) => number) {
    this.field = water?.data.field ?? null;
  }

  /** current water clock (s): the time splash rings are evaluated at */
  get time(): number {
    return surfaceU.time.value;
  }

  /** `ringT`: add splash rings evaluated at that time (NaN = smooth surface, cheaper) */
  sample(x: number, z: number, out: WaterPoint, ringT = NaN): WaterPoint {
    const f = this.field;
    let ok = false;
    if (f) {
      const fi = (x - f.x0) / f.texel, fj = (z - f.z0) / f.texel;
      if (fi >= 0 && fj >= 0 && fi <= f.w - 1 && fj <= f.h - 1) {
        const i = Math.min(f.w - 2, Math.floor(fi)), j = Math.min(f.h - 2, Math.floor(fj));
        const tx = fi - i, ty = fj - j;
        const k = (j * f.w + i) * 4, kw = f.w * 4, a = f.a, b = f.b;
        const w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
        out.flowX = a[k] * w00 + a[k + 4] * w10 + a[k + kw] * w01 + a[k + kw + 4] * w11;
        out.flowZ = a[k + 1] * w00 + a[k + 5] * w10 + a[k + kw + 1] * w01 + a[k + kw + 5] * w11;
        out.y = a[k + 3] * w00 + a[k + 7] * w10 + a[k + kw + 3] * w01 + a[k + kw + 7] * w11;
        out.groundY = b[k] * w00 + b[k + 4] * w10 + b[k + kw] * w01 + b[k + kw + 4] * w11;
        out.speed = b[k + 1] * w00 + b[k + 5] * w10 + b[k + kw + 1] * w01 + b[k + kw + 5] * w11;
        ok = true;
      }
    }
    if (!ok) {
      const s = river.sample(x, z, this.rs);
      const g = this.groundY(x, z);
      const rel = s.waterY > g ? Math.min(1.35, Math.max(0.06, Math.pow((s.waterY - g) / Math.max(0.6 * s.depth, 0.05), 0.67))) : 0;
      out.flowX = s.tx * s.speed * rel;
      out.flowZ = s.tz * s.speed * rel;
      out.y = s.waterY;
      out.groundY = g;
      out.speed = s.speed;
    }
    out.depth = out.y - out.groundY;
    out.wet = out.depth > 0.02;
    if (!out.wet) {
      out.flowX = 0;
      out.flowZ = 0;
    }
    if (ringT === ringT && this.water) out.y += ringHeight(x, z, ringT);
    return out;
  }

  /** full splash (two rings, foam, droplets). strength 0.3 = a lure landing, 1 = a fish jumping clear. */
  splash(x: number, z: number, t: number, strength: number) {
    this.water?.events.splash(x, z, t, strength);
  }

  /** the rings of a splash without droplets */
  splashRings(x: number, z: number, t: number, strength: number) {
    const ev = this.water?.events;
    if (!ev) return;
    const s = Math.max(0.02, strength);
    ev.ring(x, z, t, 0.04 * s, 3.5 + 1.5 * Math.min(s, 1), 0.13 * Math.sqrt(s));
    ev.ring(x, z, t + 0.35, 0.01 * s, 2.5 + 1.0 * Math.min(s, 1), 0.07 * Math.sqrt(s));
  }

  /** a single soft swell (a fish rising) */
  ring(x: number, z: number, t: number, amplitude: number, life = 2, foamRadius = 0) {
    this.water?.events.ring(x, z, t, amplitude, life, foamRadius);
  }

  /** the float landing: a crown of drops, the jet, crisp rings (strength ~1 at the end of a cast) */
  plop(x: number, z: number, t: number, strength: number) {
    this.water?.events.plop(x, z, t, strength);
  }

  /**
   * A float going in (strength ~1 at the end of a cast), built from the water's own events: a few big glassy beads
   * thrown up close round it rather than a spray of fine flecks, the surface broken under them (a small swell with a
   * foam collar), and a bright first ring that is already spreading a fifth of a second later, a finer one inside it.
   */
  floatPlop(x: number, z: number, t: number, strength: number) {
    const ev = this.water?.events;
    if (!ev) return;
    const s = Math.max(0.05, strength), k = Math.min(s, 2), r = Math.sqrt(s);
    ev.ring(x, z, t, 0.003 * s, 1.8, 0.05 * r);
    ev.train(x, z, t, 0.0042 * s, 5, { wavelength: 0.045, speed: 0.34, r0: 0.035 });
    ev.train(x, z, t + 0.12, 0.0022 * s, 4, { wavelength: 0.03, speed: 0.27, r0: 0.025 });
    const y = this.sample(x, z, this.pw).y, seed = Math.round(x * 131 + z * 71 + t * 7);
    // still in the air ~0.2 s on: the beads thrown up and just out, a low ring of smaller ones, the jet from the hole
    ev.droplets.crown(x, y, z, Math.round(8 * k), 1.0 * r, 0.0055, seed, 0.02, 0.12, 0.38);
    ev.droplets.crown(x, y, z, Math.round(6 * k), 0.7 * r, 0.004, seed + 2, 0.024, 0.3, 0.6);
    ev.droplets.burst(x, y, z, Math.max(2, Math.round(3 * k)), 1.05 * r, 0.0065, seed + 1, 0.1, 0.75, 0.07);
  }
  private pw: WaterPoint = { y: 0, flowX: 0, flowZ: 0, speed: 0, depth: 0, groundY: 0, wet: false };

  /** a crisp ripple train (a float dipping, a nibble, a wrist in the water, the line cutting the surface) */
  train(x: number, z: number, t: number, amplitude: number, life = 2.5, wavelength = 0.026, speed = 0.26, r0 = 0.02) {
    this.water?.events.train(x, z, t, amplitude, life, { wavelength, speed, r0 });
  }

  /** a fish turning under the surface: a smooth dome spreading as it flattens */
  boil(x: number, z: number, t: number, amplitude: number, radius: number, life = 2, spread = 0.12) {
    this.water?.events.boil(x, z, t, amplitude, radius, life, spread);
  }

  /** a tail stroke's vortex */
  swirl(x: number, z: number, t: number, amplitude: number, radius: number, life = 1.4, spin = 1) {
    this.water?.events.swirl(x, z, t, amplitude, radius, life, spin);
  }

  /** drops flung along (dx, dz) */
  spray(x: number, y: number, z: number, dx: number, dz: number, count: number, speed: number, size: number, seed: number) {
    this.water?.events.spray(x, y, z, dx, dz, count, speed, size, seed);
  }

  /** a drop falling off a held fish or a wet hand */
  drip(x: number, y: number, z: number, seed: number) {
    this.water?.events.droplets.burst(x, y, z, 1, 0.08, 0.0048, seed, 0.5, 0.3);
  }

  /** advance the drops in flight (photo poses step the water's events with their scripted run) */
  stepEvents(dt: number, t: number) {
    this.water?.events.update(dt, t);
  }

  /** forget every ring and drop (shot poses start from calm water) */
  clearRings() {
    if (this.water) this.water.events.clear();
    else for (let i = 0; i < MAX_EVENTS; i++) eventA[i].z = -1e5;
  }
}
