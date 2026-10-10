import type { WaterProbe, WaterPoint } from './WaterProbe';

/**
 * The fishing line as a verlet rope: point 0 is pinned to the rod tip, the last point to the float (or free).
 * Distance constraints only resist stretching, so a slack line sags in the air and lies on the water, where it
 * floats and drifts with the current. Typed arrays, no allocations per step.
 */
export class Rope {
  readonly n: number;
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly z: Float32Array;
  private px: Float32Array;
  private py: Float32Array;
  private pz: Float32Array;
  /** water surface (or -1e9 where dry), current and ground under every point, refreshed once per frame */
  private wy: Float32Array;
  private wfx: Float32Array;
  private wfz: Float32Array;
  private wg: Float32Array;
  private wd: Float32Array;
  /** total rest length (m): the line paid out */
  rest = 1;
  /** if > 0, the length the verlet uses instead of `rest` (a loaded line in a fight is drawn straight) */
  drawRest = 0;
  /** inverse mass of the free end (a float is heavier than a line point) */
  endInvMass = 0.25;
  /** velocity kept per step in the air (along the line, when airNormalDamp < 1) */
  airDamp = 0.996;
  /**
   * < 1: velocity kept per step across the line in the air. A thin line's drag across it is far stronger than along
   * it, so in a cast it trails the float in a smooth belly instead of flying on past it.
   */
  airNormalDamp = 1;
  /** a point that carries the float (the hold: it rides the line between the tip and the fish), and its inverse mass */
  beadIndex = -1;
  beadInvMass = 1;
  private wp: WaterPoint = { y: 0, flowX: 0, flowZ: 0, speed: 0, depth: 0, groundY: 0, wet: false };

  constructor(n = 40) {
    this.n = n;
    const f = () => new Float32Array(n);
    this.x = f(); this.y = f(); this.z = f();
    this.px = f(); this.py = f(); this.pz = f();
    this.wy = f(); this.wfx = f(); this.wfz = f(); this.wg = f(); this.wd = f();
    this.wy.fill(-1e9);
    this.wg.fill(-1e9);
  }

  /** straight line from a to b, at rest */
  reset(ax: number, ay: number, az: number, bx: number, by: number, bz: number) {
    const n = this.n;
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      this.x[i] = this.px[i] = ax + (bx - ax) * t;
      this.y[i] = this.py[i] = ay + (by - ay) * t;
      this.z[i] = this.pz[i] = az + (bz - az) * t;
    }
  }

  /**
   * Give the line a sideways twist (m/s across it, varying along it): falling slack onto the water it lies down in
   * soft S-curves instead of a ruled line.
   */
  kink(speed: number, seed: number) {
    const n = this.n, h = 1 / 120;
    for (let i = 1; i < n - 1; i++) {
      const dx = this.x[i + 1] - this.x[i - 1], dz = this.z[i + 1] - this.z[i - 1];
      const l = Math.hypot(dx, dz) || 1;
      const v = speed * (Math.sin(i * 0.52 + seed) * 0.7 + Math.sin(i * 0.21 + seed * 2.1) * 0.5) * h;
      this.px[i] += (dz / l) * v;
      this.pz[i] -= (dx / l) * v;
    }
  }

  /** keep this share of every point's velocity (the float has landed: the line stops paying out behind it) */
  damp(k: number) {
    for (let i = 1; i < this.n; i++) {
      this.px[i] = this.x[i] - (this.x[i] - this.px[i]) * k;
      this.py[i] = this.y[i] - (this.y[i] - this.py[i]) * k;
      this.pz[i] = this.z[i] - (this.z[i] - this.pz[i]) * k;
    }
  }

  /** water surface height under point i (-1e9 where dry, or a film too shallow to mirror anything), as sampled */
  waterY(i: number) {
    return this.wd[i] > 0.12 ? this.wy[i] : -1e9;
  }

  /** water surface height under point i wherever there is water (-1e9 on dry ground) */
  surfaceY(i: number) {
    return this.wy[i];
  }

  setStart(x: number, y: number, z: number) {
    this.x[0] = this.px[0] = x;
    this.y[0] = this.py[0] = y;
    this.z[0] = this.pz[0] = z;
  }

  setEnd(x: number, y: number, z: number) {
    const k = this.n - 1;
    this.x[k] = this.px[k] = x;
    this.y[k] = this.py[k] = y;
    this.z[k] = this.pz[k] = z;
  }

  endX() { return this.x[this.n - 1]; }
  endY() { return this.y[this.n - 1]; }
  endZ() { return this.z[this.n - 1]; }

  /** distance between the two ends */
  span(): number {
    const k = this.n - 1;
    return Math.hypot(this.x[k] - this.x[0], this.y[k] - this.y[0], this.z[k] - this.z[0]);
  }

  /** water and ground under each point (once per frame; the line moves little between frames) */
  sampleWater(probe: WaterProbe) {
    const wp = this.wp;
    for (let i = 1; i < this.n; i++) {
      probe.sample(this.x[i], this.z[i], wp);
      this.wy[i] = wp.wet ? wp.y : -1e9;
      this.wfx[i] = wp.flowX;
      this.wfz[i] = wp.flowZ;
      this.wg[i] = wp.groundY;
      this.wd[i] = wp.depth;
    }
  }

  /**
   * One verlet step of h seconds. The start is always pinned (setStart before each step); the end is pinned when
   * `endPinned` (setEnd before each step), otherwise it hangs with `endInvMass`.
   */
  step(h: number, endPinned: boolean, iterations = 12) {
    const n = this.n, x = this.x, y = this.y, z = this.z, px = this.px, py = this.py, pz = this.pz;
    const g = 9.81 * h * h;
    const last = endPinned ? n - 1 : n;
    for (let i = 1; i < last; i++) {
      const onWater = y[i] <= this.wy[i] + 0.004;
      let vx = x[i] - px[i], vy = y[i] - py[i], vz = z[i] - pz[i];
      if (onWater) {
        // floating line: relaxes toward the current (~0.1 s), vertical motion killed by the surface
        vx = vx * 0.92 + this.wfx[i] * h * 0.08;
        vz = vz * 0.92 + this.wfz[i] * h * 0.08;
        vy = vy > 0 ? vy * 0.5 : 0;
      } else if (this.airNormalDamp < 1) {
        let tx = x[Math.min(n - 1, i + 1)] - x[i - 1], ty = y[Math.min(n - 1, i + 1)] - y[i - 1], tz = z[Math.min(n - 1, i + 1)] - z[i - 1];
        const tl = Math.hypot(tx, ty, tz);
        const kt = this.airDamp, kn = this.airNormalDamp;
        if (tl > 1e-9) {
          tx /= tl; ty /= tl; tz /= tl;
          const vt = vx * tx + vy * ty + vz * tz;
          vx = tx * vt * kt + (vx - tx * vt) * kn;
          vy = ty * vt * kt + (vy - ty * vt) * kn;
          vz = tz * vt * kt + (vz - tz * vt) * kn;
        } else {
          vx *= kn; vy *= kn; vz *= kn;
        }
      } else {
        const k = this.airDamp;
        vx *= k; vy *= k; vz *= k;
      }
      px[i] = x[i]; py[i] = y[i]; pz[i] = z[i];
      x[i] += vx; y[i] += vy - g; z[i] += vz;
      if (y[i] < this.wy[i] + 0.002) y[i] = this.wy[i] + 0.002;
      const gy = this.wg[i] + 0.004;
      if (y[i] < gy) {
        // resting on gravel: friction
        y[i] = gy;
        px[i] += (x[i] - px[i]) * 0.5;
        pz[i] += (z[i] - pz[i]) * 0.5;
      }
    }
    const seg = (this.drawRest > 0 ? Math.min(this.drawRest, this.rest) : this.rest) / (n - 1);
    const endW = endPinned ? 0 : this.endInvMass;
    for (let it = 0; it < iterations; it++) {
      for (let s = 0; s < n - 1; s++) {
        const dx = x[s + 1] - x[s], dy = y[s + 1] - y[s], dz = z[s + 1] - z[s];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 <= seg * seg) continue;
        const d = Math.sqrt(d2);
        const w0 = s === 0 ? 0 : s === this.beadIndex ? this.beadInvMass : 1;
        const w1 = s + 1 === n - 1 ? endW : s + 1 === this.beadIndex ? this.beadInvMass : 1;
        const ws = w0 + w1;
        if (ws === 0) continue;
        const k = (d - seg) / (d * ws);
        if (w0) {
          x[s] += dx * k * w0; y[s] += dy * k * w0; z[s] += dz * k * w0;
        }
        if (w1) {
          x[s + 1] -= dx * k * w1; y[s + 1] -= dy * k * w1; z[s + 1] -= dz * k * w1;
        }
      }
    }
  }
}
