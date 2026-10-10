import * as THREE from 'three/webgpu';
import { river, newSample, type RiverSample } from '../world/River';

/**
 * Water surface for the river: a ribbon of rings perpendicular to the flow. Every ring is flat across and
 * sits at the local water level, so the riffle steps down along the flow while the pool stays exactly on
 * its reflection plane (y = POOL_LEVEL). Each ring runs past the outermost wet point of the bank and tucks
 * under it; the terrain hides the overlap and the water material fades the contact line by water depth.
 * Only public river data is used (river.xs/zs/ss and river.sample), so edits to the control points do not
 * break the builder, and the bank search adapts to terrain edits.
 */

export interface RibbonSection {
  name: string;
  /** centreline x range; the centreline is monotonic in x */
  x0: number;
  x1: number;
  /** ring spacing along the flow (m) */
  step: number;
  /** quads across the river */
  across: number;
}

export interface RibbonOptions {
  sections: RibbonSection[];
  /** height of the rendered ground: the core grid inside the corridor, the coarse grid outside it */
  groundY: (x: number, z: number) => number;
  /** how far the ribbon runs under the bank past the last wet point (m) */
  tuck?: number;
  /** farthest the ribbon reaches past the channel edge (m) */
  maxReach?: number;
}

interface Ring {
  x: number;
  z: number;
  tx: number;
  tz: number;
  s: number;
  y: number;
  half: number;
  speed: number;
  depth: number;
  asym: number;
  north: number;
  south: number;
}

const S: RiverSample = newSample();
const A = new THREE.Vector2();
const B = new THREE.Vector2();

/** Arc length of the centreline point with the given x. */
export function sAtX(x: number): number {
  const xs = river.xs, n = river.count;
  if (x <= xs[0]) return 0;
  if (x >= xs[n - 1]) return river.length;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (xs[m] < x) lo = m; else hi = m;
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return river.ss[lo] + (river.ss[hi] - river.ss[lo]) * t;
}

/** Centreline point at arc length s (clamped to the polyline). */
export function centreAt(s: number, out: THREE.Vector2): THREE.Vector2 {
  const ss = river.ss, n = river.count;
  const sc = Math.min(Math.max(s, 0), river.length);
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (ss[m] < sc) lo = m; else hi = m;
  }
  const t = (sc - ss[lo]) / Math.max(1e-6, ss[hi] - ss[lo]);
  return out.set(river.xs[lo] + (river.xs[hi] - river.xs[lo]) * t, river.zs[lo] + (river.zs[hi] - river.zs[lo]) * t);
}

/**
 * Lateral distance (m) from the centreline to just past the outermost wet point on one side
 * (+1 south, -1 north). Searching from the outside in keeps side channels and water behind
 * emergent bars inside the ribbon; dry islands are covered and hidden by the terrain.
 */
function reach(r: Ring, side: number, groundY: RibbonOptions['groundY'], tuck: number, maxReach: number): number {
  const lx = -r.tz * side, lz = r.tx * side;
  const far = r.half + maxReach;
  for (let d = far; d > r.half * 0.5; d -= 0.2) {
    if (groundY(r.x + lx * d, r.z + lz * d) < r.y + 0.02) return Math.min(d + tuck, far);
  }
  return r.half * 0.5 + tuck;
}

/** One geometry per section; neighbouring sections share their joining ring (straight, level rings: no cracks). */
export function buildRiverRibbon(o: RibbonOptions): THREE.BufferGeometry[] {
  const tuck = o.tuck ?? 0.8, maxReach = o.maxReach ?? 7;
  return o.sections.map((sec) => {
    const s0 = sAtX(sec.x0), s1 = sAtX(sec.x1);
    const n = Math.max(2, Math.round((s1 - s0) / sec.step) + 1);
    const rings: Ring[] = [];
    for (let i = 0; i < n; i++) {
      const s = s0 + ((s1 - s0) * i) / (n - 1);
      // tangent from a +-1.5 m chord: smooth through the polyline's segments
      centreAt(s - 1.5, A);
      centreAt(s + 1.5, B);
      const tl = Math.hypot(B.x - A.x, B.y - A.y);
      const tx = (B.x - A.x) / tl, tz = (B.y - A.y) / tl;
      centreAt(s, A);
      river.sample(A.x, A.y, S);
      const r: Ring = {
        x: A.x, z: A.y, tx, tz, s, y: S.waterY, half: S.half, speed: S.speed, depth: S.depth, asym: S.asym,
        north: 0, south: 0,
      };
      r.south = reach(r, 1, o.groundY, tuck, maxReach);
      r.north = reach(r, -1, o.groundY, tuck, maxReach);
      rings.push(r);
    }
    // an edge that dips between two rings would cut into the water: take the local maximum over +-2 rings
    for (const side of ['north', 'south'] as const) {
      const v = rings.map((r) => r[side]);
      rings.forEach((r, i) => {
        for (let j = Math.max(0, i - 2); j <= Math.min(n - 1, i + 2); j++) r[side] = Math.max(r[side], v[j]);
      });
    }

    const row = sec.across + 1;
    const pos = new Float32Array(n * row * 3);
    const nor = new Float32Array(n * row * 3);
    const uvs = new Float32Array(n * row * 2);
    const flow = new Float32Array(n * row * 4);
    rings.forEach((r, i) => {
      for (let j = 0; j < row; j++) {
        const d = -r.north + ((r.north + r.south) * j) / sec.across;
        const k = i * row + j;
        // lateral unit vector (+d = south) is (-tz, tx), matching RiverSample.d
        pos[k * 3] = r.x - r.tz * d;
        pos[k * 3 + 1] = r.y;
        pos[k * 3 + 2] = r.z + r.tx * d;
        nor[k * 3 + 1] = 1;
        uvs[k * 2] = r.s;
        uvs[k * 2 + 1] = d;
        // flow beyond the baked field: fastest over the thalweg, slow toward the banks
        const u = d / r.half;
        const lat = Math.min(1, Math.abs(u - 0.4 * r.asym));
        const v = r.speed * Math.max(0.08, 1.15 * (1 - lat ** 2.5));
        flow[k * 4] = r.tx * v;
        flow[k * 4 + 1] = r.tz * v;
        flow[k * 4 + 2] = u;
        flow[k * 4 + 3] = r.depth;
      }
    });

    // rings advance downstream (+i), vertices run north to south (+j): (p, q, a) faces +y
    const index: number[] = [];
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < sec.across; j++) {
        const p = i * row + j, q = p + 1, a = p + row, b = a + 1;
        index.push(p, q, a, a, q, b);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    g.setAttribute('aFlow', new THREE.BufferAttribute(flow, 4));
    g.setIndex(n * row > 65535 ? new THREE.Uint32BufferAttribute(index, 1) : new THREE.Uint16BufferAttribute(index, 1));
    g.computeBoundingBox();
    g.computeBoundingSphere();
    g.name = `water-${sec.name}`;
    return g;
  });
}

/**
 * Sections. The pool is its own chunk so the mirror can test whether it is on screen.
 * Near spacing 0.4-0.5 m (the surface is flat; normals are per pixel), far sections 2.5 m.
 */
export const RIVER_SECTIONS: RibbonSection[] = [
  { name: 'far-up', x0: -480, x1: -128, step: 2.5, across: 6 },
  { name: 'riffle', x0: -128, x1: -8, step: 0.4, across: 28 },
  { name: 'pool', x0: -8, x1: 30, step: 0.4, across: 32 },
  { name: 'tail', x0: 30, x1: 116, step: 0.5, across: 24 },
  { name: 'far-down', x0: 116, x1: 480, step: 2.5, across: 6 },
];
