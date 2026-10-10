import { sampleField, type FieldSample, type RiverField } from '../water/RiverField';

/**
 * Where trout live in this river, read from the baked river field (CPU arrays, the same field the water and the
 * caustics use): holding lies in the current (riffle pockets behind rocks and seams beside the fast lane, the seams
 * of the tongue at the pool head, the tail-out), slack water to cruise in the pool, and deep refuges to bolt to.
 * Pure CPU, deterministic, also runs under tsx (scripts/fish-sim.ts).
 */
export interface WaterAt {
  waterY: number;
  bedY: number;
  depth: number;
  flowX: number;
  flowZ: number;
  speed: number;
}

export type StationKind = 'riffle' | 'head' | 'pool' | 'tail';

export interface Station {
  kind: StationKind;
  x: number;
  z: number;
  /** preferred height of the fish's centre (m) */
  y: number;
  depth: number;
  /** holding in the current (riffle, head, tail) or cruising slack water (pool) */
  holds: boolean;
  /** how far a cruiser wanders from it (m) */
  range: number;
  /** a holder that lies this far under the surface instead of down on the bed (m): in the lee of the fallen spruce,
   *  where the bed is 2-3 m down */
  suspend?: number;
}

export interface RockLike {
  x: number;
  z: number;
  /** footprint radius (m) */
  r: number;
  /** top of the rock (m) */
  topY: number;
}

/** Sections of the river by x (World layout: riffle x -34..-12, pool head, deep pool -4..18, tail-out). */
export const REACH = {
  riffle: [-35, -11] as [number, number],
  head: [-10, 1] as [number, number],
  pool: [0, 19] as [number, number],
  tail: [19, 31] as [number, number],
};

/** Small seeded PRNG (mulberry32): the school is the same on every load and in every capture. */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Habitat {
  private fs: FieldSample = { flowX: 0, flowZ: 0, foam: 0, waterY: 0, bedY: 0, speed: 0 };

  constructor(readonly field: RiverField, readonly rocks: RockLike[]) {}

  /** Water at (x, z). Returns false (and depth 0) outside the baked field or on dry ground. */
  at(x: number, z: number, out: WaterAt): boolean {
    if (!sampleField(this.field, x, z, this.fs)) {
      out.depth = 0;
      out.flowX = out.flowZ = out.speed = 0;
      return false;
    }
    out.waterY = this.fs.waterY;
    out.bedY = this.fs.bedY;
    out.depth = this.fs.waterY - this.fs.bedY;
    out.flowX = this.fs.flowX;
    out.flowZ = this.fs.flowZ;
    out.speed = this.fs.speed;
    return out.depth > 0;
  }

  depthAt(x: number, z: number): number {
    return sampleField(this.field, x, z, this.fs) ? this.fs.waterY - this.fs.bedY : 0;
  }

  /**
   * Current felt at height y: the depth-averaged field current shaped by a 1/7 power-law profile, so a trout
   * lying near the bed holds in water ~20 % slower than the surface (it is also why they hug the bottom).
   */
  currentAt(x: number, y: number, z: number, out: { x: number; z: number }): { x: number; z: number } {
    if (!sampleField(this.field, x, z, this.fs) || this.fs.waterY - this.fs.bedY <= 0.02) {
      out.x = out.z = 0;
      return out;
    }
    const d = this.fs.waterY - this.fs.bedY;
    const eta = Math.min(1, Math.max(0.02, (y - this.fs.bedY) / d));
    const k = (8 / 7) * eta ** (1 / 7);
    out.x = this.fs.flowX * k;
    out.z = this.fs.flowZ * k;
    return out;
  }

  /** Unit direction toward deeper water (central differences over 0.5 m), zero on a flat bed. */
  deeper(x: number, z: number, out: { x: number; z: number }): { x: number; z: number } {
    const e = 0.5;
    const gx = this.depthAt(x + e, z) - this.depthAt(x - e, z);
    const gz = this.depthAt(x, z + e) - this.depthAt(x, z - e);
    const l = Math.hypot(gx, gz);
    out.x = l > 1e-4 ? gx / l : 0;
    out.z = l > 1e-4 ? gz / l : 0;
    return out;
  }

  /** Distance to the nearest rock surface (m, negative inside its footprint) and that rock. */
  nearestRock(x: number, z: number): { d: number; rock: RockLike | null } {
    let best = Infinity, rock: RockLike | null = null;
    for (const r of this.rocks) {
      const d = Math.hypot(x - r.x, z - r.z) - r.r;
      if (d < best) { best = d; rock = r; }
    }
    return { d: best, rock };
  }

  /** Max current speed within `radius` (m): a lie beside a fast lane gets the food it carries. */
  private fastNear(x: number, z: number, radius: number): number {
    let m = 0;
    const w: WaterAt = { waterY: 0, bedY: 0, depth: 0, flowX: 0, flowZ: 0, speed: 0 };
    for (let a = 0; a < 8; a++) {
      const ang = (a / 8) * Math.PI * 2;
      if (this.at(x + Math.cos(ang) * radius, z + Math.sin(ang) * radius, w)) m = Math.max(m, Math.hypot(w.flowX, w.flowZ));
    }
    return m;
  }

  /**
   * Holding lies of one reach, best first: current a trout can sit in (`speed` m/s band) next to faster water,
   * deep enough, away from the rocks' footprints; plus the cushion just downstream of every rock in the reach.
   */
  lies(kind: StationKind, o: { minDepth: number; speed: [number, number]; fast: number; spacing: number; count: number; seed: number }): Station[] {
    const [x0, x1] = REACH[kind];
    const w: WaterAt = { waterY: 0, bedY: 0, depth: 0, flowX: 0, flowZ: 0, speed: 0 };
    const cand: Array<{ x: number; z: number; score: number }> = [];
    const r = rng(o.seed);
    for (let x = x0; x <= x1; x += 0.5) {
      for (let z = this.field.z0 + 0.5; z <= this.field.z1 - 0.5; z += 0.5) {
        const jx = x + (r() - 0.5) * 0.3, jz = z + (r() - 0.5) * 0.3;
        if (!this.at(jx, jz, w) || w.depth < o.minDepth) continue;
        const u = Math.hypot(w.flowX, w.flowZ);
        if (u < o.speed[0] || u > o.speed[1]) continue;
        const rk = this.nearestRock(jx, jz);
        if (rk.d < 0.15) continue;
        const fast = this.fastNear(jx, jz, 1.2);
        if (fast < o.fast) continue;
        // deeper is safer, mid-band current is best, the edge of fast water brings the food
        const mid = 1 - Math.abs((u - (o.speed[0] + o.speed[1]) / 2) / ((o.speed[1] - o.speed[0]) / 2));
        cand.push({ x: jx, z: jz, score: Math.min(w.depth, 1.5) * 0.6 + mid * 0.5 + Math.min(fast - o.fast, 1) * 0.4 + r() * 0.15 });
      }
    }
    // the cushion behind each rock of the reach (the classic lie)
    for (const rock of this.rocks) {
      if (rock.x < x0 || rock.x > x1) continue;
      if (!this.at(rock.x, rock.z, w)) continue;
      const u = Math.hypot(w.flowX, w.flowZ) || 1;
      const tx = w.flowX / u || 1, tz = w.flowZ / u;
      const x = rock.x + tx * (rock.r + 0.4), z = rock.z + tz * (rock.r + 0.4);
      if (this.at(x, z, w) && w.depth >= o.minDepth * 0.85) cand.push({ x, z, score: 2 + r() * 0.2 });
    }
    cand.sort((a, b) => b.score - a.score);
    const out: Station[] = [];
    for (const c of cand) {
      if (out.length >= o.count) break;
      if (out.some((s) => Math.hypot(s.x - c.x, s.z - c.z) < o.spacing)) continue;
      this.at(c.x, c.z, w);
      out.push({ kind, x: c.x, z: c.z, y: w.bedY + Math.min(0.14, w.depth * 0.3), depth: w.depth, holds: true, range: 0.4 });
    }
    return out;
  }

  /** Slack water to cruise in (pool): homes spread over the pool, `count` of them, deepest first. */
  slack(o: { minDepth: number; maxSpeed: number; spacing: number; count: number; seed: number; range: number }): Station[] {
    const [x0, x1] = REACH.pool;
    const w: WaterAt = { waterY: 0, bedY: 0, depth: 0, flowX: 0, flowZ: 0, speed: 0 };
    const r = rng(o.seed);
    const cand: Array<{ x: number; z: number; score: number }> = [];
    for (let x = x0; x <= x1; x += 0.75) {
      for (let z = this.field.z0 + 0.5; z <= this.field.z1 - 0.5; z += 0.75) {
        if (!this.at(x, z, w) || w.depth < o.minDepth || Math.hypot(w.flowX, w.flowZ) > o.maxSpeed) continue;
        cand.push({ x, z, score: Math.min(w.depth, 2.5) + r() * 1.2 });
      }
    }
    cand.sort((a, b) => b.score - a.score);
    const out: Station[] = [];
    for (const c of cand) {
      if (out.length >= o.count) break;
      if (out.some((s) => Math.hypot(s.x - c.x, s.z - c.z) < o.spacing)) continue;
      this.at(c.x, c.z, w);
      out.push({ kind: 'pool', x: c.x, z: c.z, y: w.waterY - w.depth * 0.5, depth: w.depth, holds: false, range: o.range });
    }
    return out;
  }

  /** A home at a given point (hand-placed margin trout; `holds`: a lie it holds low on), or null if it is not water of `minDepth`. */
  homeAt(kind: StationKind, x: number, z: number, minDepth: number, range: number, holds = kind !== 'pool'): Station | null {
    const w: WaterAt = { waterY: 0, bedY: 0, depth: 0, flowX: 0, flowZ: 0, speed: 0 };
    if (!this.at(x, z, w) || w.depth < minDepth) return null;
    return { kind, x, z, y: w.waterY - w.depth * 0.5, depth: w.depth, holds, range };
  }

  /** Deep cover to bolt to: the deepest water within `reach` m that lies away from the threat. */
  refuge(x: number, z: number, awayX: number, awayZ: number, reach: number, minDepth: number, out: { x: number; z: number }) {
    const w: WaterAt = { waterY: 0, bedY: 0, depth: 0, flowX: 0, flowZ: 0, speed: 0 };
    let best = -Infinity;
    out.x = x;
    out.z = z;
    for (let a = 0; a < 24; a++) {
      const ang = (a / 24) * Math.PI * 2;
      for (const d of [reach * 0.35, reach * 0.65, reach]) {
        const px = x + Math.cos(ang) * d, pz = z + Math.sin(ang) * d;
        if (!this.at(px, pz, w) || w.depth < minDepth * 0.6) continue;
        const away = (Math.cos(ang) * awayX + Math.sin(ang) * awayZ);
        const score = Math.min(w.depth, minDepth * 2) + away * 1.2 - d * 0.05;
        if (score > best) { best = score; out.x = px; out.z = pz; }
      }
    }
    return out;
  }
}
