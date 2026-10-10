import { sampleTerrain, newPoint } from './Terrain';
import { fbm, smoothstep, clamp } from './noise';
import { LAYOUT } from './Layout';
import { river } from './River';
import { CAMERA_SPOTS } from '../debug/CameraSpots';

/**
 * Where the conifers stand. Pure CPU code (also run under tsx for top-down previews): one candidate per jittered
 * grid cell, kept by a density built from the terrain masks, the camp clearings, the camera sightlines and a
 * "sun window" that thins the crowns whose shadow would fall on the pool and the bar.
 */

export interface TreeVariant {
  id: string;
  kind: 'lodgepole' | 'ponderosa' | 'spruce' | 'fir';
  /** highest needle card (m) */
  height: number;
  /** typical silhouette radius (m) */
  crownR: number;
  /** live crown base above the ground (m) */
  crownBase: number;
  bark: 'pine' | 'fir';
  /** trunk radius at the ground (m, before scaling) */
  trunkR: number;
  /** linear needle tint (trees.json suggestedTintLinear, calibrated) */
  needleTint: [number, number, number];
  barkTint: [number, number, number];
  /** bark only (a dead snag with a broken top and branch stubs) */
  snag?: boolean;
  /** a young tree model (small at scale 1): its LODs switch by size as well as distance */
  young?: boolean;
}

export const TREE_VARIANTS: TreeVariant[] = [
  { id: 'lodgepole_a', kind: 'lodgepole', height: 23.8, crownR: 1.7, crownBase: 12.7, bark: 'pine', trunkR: 0.257, needleTint: [0.58, 0.76, 0.84], barkTint: [1.55, 1.45, 1.42] },
  { id: 'lodgepole_b', kind: 'lodgepole', height: 27.9, crownR: 1.55, crownBase: 17.2, bark: 'pine', trunkR: 0.26, needleTint: [0.58, 0.76, 0.84], barkTint: [1.55, 1.45, 1.42] },
  { id: 'lodgepole_c', kind: 'lodgepole', height: 16.8, crownR: 1.55, crownBase: 6.5, bark: 'pine', trunkR: 0.169, needleTint: [0.58, 0.76, 0.84], barkTint: [1.55, 1.45, 1.42] },
  { id: 'ponderosa_a', kind: 'ponderosa', height: 31.2, crownR: 3.5, crownBase: 14.8, bark: 'pine', trunkR: 0.609, needleTint: [0.66, 0.84, 0.86], barkTint: [1.75, 1.42, 1.22] },
  { id: 'ponderosa_b', kind: 'ponderosa', height: 20.9, crownR: 2.5, crownBase: 6.6, bark: 'pine', trunkR: 0.378, needleTint: [0.66, 0.84, 0.86], barkTint: [1.75, 1.42, 1.22] },
  { id: 'spruce_a', kind: 'spruce', height: 25.8, crownR: 2.2, crownBase: 1.0, bark: 'fir', trunkR: 0.365, needleTint: [0.6, 0.66, 1.25], barkTint: [1.35, 1.3, 1.25] },
  { id: 'spruce_b', kind: 'spruce', height: 30.7, crownR: 2.35, crownBase: 4.2, bark: 'fir', trunkR: 0.448, needleTint: [0.6, 0.66, 1.25], barkTint: [1.35, 1.3, 1.25] },
  { id: 'spruce_c', kind: 'spruce', height: 14.8, crownR: 1.75, crownBase: 0.0, bark: 'fir', trunkR: 0.221, needleTint: [0.6, 0.66, 1.25], barkTint: [1.35, 1.3, 1.25] },
  { id: 'fir_a', kind: 'fir', height: 18.6, crownR: 1.35, crownBase: 0.7, bark: 'fir', trunkR: 0.26, needleTint: [0.58, 0.66, 1.2], barkTint: [1.5, 1.5, 1.5] },
  { id: 'fir_b', kind: 'fir', height: 12.6, crownR: 1.15, crownBase: 0.2, bark: 'fir', trunkR: 0.18, needleTint: [0.58, 0.66, 1.2], barkTint: [1.5, 1.5, 1.5] },
  // forest polish: young trees as they grow wild (broad skirts, lopsided, gaps between the whorls, bent leaders, grey
  // dead lower branches) and broken-topped snags
  { id: 'spruce_y', kind: 'spruce', height: 9.5, crownR: 1.95, crownBase: 0.0, bark: 'fir', trunkR: 0.11, needleTint: [0.62, 0.69, 1.18], barkTint: [1.35, 1.3, 1.25], young: true },
  { id: 'fir_y', kind: 'fir', height: 7.6, crownR: 1.6, crownBase: 0.0, bark: 'fir', trunkR: 0.09, needleTint: [0.6, 0.7, 1.12], barkTint: [1.5, 1.5, 1.5], young: true },
  { id: 'spruce_t', kind: 'spruce', height: 12.3, crownR: 1.9, crownBase: 3.0, bark: 'fir', trunkR: 0.13, needleTint: [0.6, 0.66, 1.25], barkTint: [1.35, 1.3, 1.25], young: true },
  { id: 'snag_a', kind: 'lodgepole', height: 16.2, crownR: 1.0, crownBase: 0.0, bark: 'pine', trunkR: 0.2, needleTint: [1, 1, 1], barkTint: [1.55, 1.45, 1.42], snag: true },
  { id: 'snag_b', kind: 'spruce', height: 11.3, crownR: 1.0, crownBase: 0.0, bark: 'fir', trunkR: 0.22, needleTint: [1, 1, 1], barkTint: [1.35, 1.3, 1.25], snag: true },
];

const V = Object.fromEntries(TREE_VARIANTS.map((v, i) => [v.id, i])) as Record<string, number>;

export interface TreeInst {
  x: number;
  y: number;
  z: number;
  yaw: number;
  scale: number;
  /** variant index */
  v: number;
  /** 0..1: far trees are thinned by it (a tree is drawn while rank < keep(distance)) */
  rank: number;
  /** per-tree colour jitter, multiplied into the species tint */
  jitter: [number, number, number];
  /** height and crown radius after scaling (m) */
  h: number;
  r: number;
  /** crown base above the ground after scaling (m) */
  cb: number;
  /** crown width relative to the variant's (non-uniform scale across the trunk axis) */
  sx: number;
  /** lean (rad) toward the world xz direction `tiltDir` (rad) */
  tilt: number;
  tiltDir: number;
  /** a dead snag: bark and bare branches only, weathered grey */
  dead: boolean;
}

/** Placement area (the tree instances); beyond it the terrain's canopy tone carries the forest. */
export const FOREST_REGION = { x0: -200, x1: 210, z0: -175, z1: 165 };
const CELL = 3.7;

/** The low morning sun after the sky rotation (SkyLighting): toward the north-east, 20 degrees up. */
export const DEFAULT_SUN: [number, number, number] = [0.72, 0.345, -0.602];

/** Footpaths kept clear of trees and grass: campfire bench to the cabin and down to the dock. */
export const PATHS: Array<[number, number, number, number]> = [
  [LAYOUT.campfire.x, LAYOUT.campfire.z, LAYOUT.cabin.x - 1.5, LAYOUT.cabin.z - 4.0],
  [LAYOUT.campfire.x, LAYOUT.campfire.z, LAYOUT.dock.x, LAYOUT.dock.zBase + 0.6],
];

/** distance from (x, z) to the segment (ax, az)-(bx, bz) */
export function segDist(x: number, z: number, ax: number, az: number, bx: number, bz: number) {
  const dx = bx - ax, dz = bz - az;
  const t = clamp(((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz), 0, 1);
  return Math.hypot(x - ax - dx * t, z - az - dz * t);
}

/** a footpath that wanders a little, so it never reads as a ruler line */
export function pathDist(x: number, z: number) {
  let d = Infinity;
  for (const [ax, az, bx, bz] of PATHS) d = Math.min(d, segDist(x + fbm(z * 0.15, 3.3, 2) * 0.9, z, ax, az, bx, bz));
  return d;
}

/** Clearings: cabin terrace, campfire bench, dock base, spawn. */
const CLEARINGS: Array<[number, number, number]> = [
  [LAYOUT.cabin.x, LAYOUT.cabin.z, 10.5],
  [LAYOUT.campfire.x, LAYOUT.campfire.z, 6.5],
  [LAYOUT.dock.x, LAYOUT.dock.zBase, 4.5],
  [LAYOUT.spawn.x, LAYOUT.spawn.z, 4.0],
];

/** Sightlines of the fixed camera spots: [eye x, eye z, dir x, dir z, length, radius] */
function sightlines() {
  return CAMERA_SPOTS.map((s) => {
    const dx = s.target[0] - s.eye[0], dz = s.target[2] - s.eye[2];
    const l = Math.hypot(dx, dz);
    // only right in front of the lens: what the view looks at (the far bank in C4) keeps its trees
    const elevated = s.eye[1] > 5;
    return [s.eye[0], s.eye[2], dx / l, dz / l, elevated ? 34 : 11, elevated ? 7.5 : 3.0] as const;
  });
}

/**
 * Where the low sun should still reach (x0, z0, x1, z1 on the water plane, fraction of shading trees kept):
 * the C2 shallows and splash, the C4 glitter path, the bar and the south half of the pool, the riffle, the camp,
 * and the far half of the pool (mostly left in the north shore's shade, which is what the pool mirrors).
 */
const SUN_WINDOWS: Array<[number, number, number, number, number]> = [
  [0, 0, 7, 6.5, 0.12],
  [13, -5.5, 20, 1, 0.12],
  [-4, -2.5, 21, 9.5, 0.5],
  [-32, -4, -15, 8, 0.15],
  [-2, 9.5, 14, 26, 0.55],
  [-4, -9, 19, -2.5, 0.85],
];

/** Trees whose shadow can fall on the play area cast shadows (horizontal distance from the shadow focus). */
export const SHADOW_FOCUS: [number, number] = [2, 4];
export const castsShadow = (t: TreeInst) => t.h > (TREE_VARIANTS[t.v].young ? 4.5 : 3) && Math.hypot(t.x - SHADOW_FOCUS[0], t.z - SHADOW_FOCUS[1]) < 60 + 22 * ((t.rank * 7.31) % 1);

/** small deterministic hash of a cell (0..1) */
function hash2(i: number, j: number, k: number) {
  let h = (i * 374761393 + j * 668265263 + k * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/** pick from weighted ids with u in [0, 1) */
function pick(weights: Array<[string, number]>, u: number) {
  let sum = 0;
  for (const [, w] of weights) sum += w;
  let a = u * sum;
  for (const [id, w] of weights) {
    a -= w;
    if (a <= 0) return V[id];
  }
  return V[weights[weights.length - 1][0]];
}

export interface ForestStats {
  candidates: number;
  placed: number;
  sunThinned: number;
  fringe?: { near: number; lines: number; sun: number; placed: number };
  understory?: number;
  thickets?: number;
  thicketTrees?: number;
  infill?: number;
  low?: number;
  snags?: number;
}

/**
 * Young conifers as they grow wild (forest polish): broad, lopsided spruce and ragged fir with gaps between the whorls
 * and bent leaders, a few thicket-grown spruce with grey dead lower branches, and some of round 2's narrow spires.
 */
const YOUNG: Array<[string, number]> = [['spruce_y', 3], ['fir_y', 3], ['spruce_t', 1.2], ['fir_b', 0.6], ['spruce_c', 0.5]];

/** a 2D bucket grid of the trees placed so far, for spacing checks */
class Spacing {
  private cells = new Map<number, number[]>();
  constructor(private out: TreeInst[], private cell = 4) {}
  private key(i: number, j: number) {
    return (i + 4096) * 8192 + (j + 4096);
  }
  add(t: TreeInst) {
    const k = this.key(Math.floor(t.x / this.cell), Math.floor(t.z / this.cell));
    let a = this.cells.get(k);
    if (!a) this.cells.set(k, (a = []));
    a.push(this.out.length - 1);
  }
  /** true if a tree stands closer than `d` + `kr` x its crown radius */
  near(x: number, z: number, d: number, kr = 0) {
    const ci = Math.floor(x / this.cell), cj = Math.floor(z / this.cell);
    for (let a = -2; a <= 2; a++) for (let b = -2; b <= 2; b++) {
      for (const i of this.cells.get(this.key(ci + a, cj + b)) ?? []) {
        const t = this.out[i];
        if (Math.hypot(t.x - x, t.z - z) < d + kr * t.r) return true;
      }
    }
    return false;
  }
}

/**
 * Deterministic tree list. `groundY` is the rendered ground (fine corridor grid / 4 m hill grid).
 * The sun window: crowns whose shadow on the water plane would cover the pool, the near bar or the camp are
 * mostly removed (some stay for dappled shade), so the low sun still reaches the water through the forest.
 * Round 2: trees grow in clumps with gaps (a jittered cell grid alone read as a planted row), vary strongly in height
 * and width, some lean (toward the water on the undercut north bank) and a few are dead grey snags; young trees in
 * the open are spruce and fir with branches to the ground; an understory of young conifers fills the gaps of the
 * north-bank forest and the south forest edge; the north shore's young trees stand in clumps, not in single file.
 * Forest polish: young trees crowd together in thickets of very different heights; edges facing the river or the
 * meadow are walls of spruce and fir branched to the ground; mid-size spruce and fir close the north-bank canopy;
 * seedlings and squat shrubs on the floor near the play area; snags are broken-topped models with branch stubs.
 */
export function placeTrees(groundY: (x: number, z: number) => number, sun: [number, number, number] = DEFAULT_SUN,
  stats?: ForestStats): TreeInst[] {
  const tp = newPoint();
  const lines = sightlines();
  const sh = Math.hypot(sun[0], sun[2]);
  const sx = sun[0] / sh, sz = sun[2] / sh, cotE = sh / sun[1];
  const out: TreeInst[] = [];
  const spacing = new Spacing(out);
  const push = (t: TreeInst) => {
    out.push(t);
    spacing.add(t);
  };
  let candidates = 0, sunThinned = 0, snags = 0;

  /** clearings, footpaths and camera sightlines (with `pad` metres of extra room) */
  const blocked = (x: number, z: number, pad: number) => {
    for (const [cx, cz, r] of CLEARINGS) if (Math.hypot(x - cx, z - cz) < r + pad * 0.5) return true;
    if (pathDist(x, z) < 1.8 + pad * 0.4) return true;
    for (const [ex, ez, dx, dz, len, rad] of lines) {
      const t = clamp((x - ex) * dx + (z - ez) * dz, -4, len);
      if (Math.hypot(x - ex - dx * t, z - ez - dz * t) < rad + pad) return true;
    }
    return false;
  };
  /** fraction of such trees the sun windows keep (the crown's shadow strip on the water plane) */
  const sunKeep = (x: number, y: number, z: number, h: number, r: number, cb: number, windows: typeof SUN_WINDOWS) => {
    const a0 = (y + cb) * cotE, a1 = (y + h) * cotE;
    let keep = 1;
    for (const [wx0, wz0, wx1, wz1, k] of windows) {
      if (k >= keep) continue;
      for (let s = 0; s <= 8; s++) {
        const a = a0 + ((a1 - a0) * s) / 8;
        const px = x - sx * a, pz = z - sz * a;
        if (px > wx0 - r && px < wx1 + r && pz > wz0 - r && pz < wz1 + r) {
          keep = k;
          break;
        }
      }
    }
    return keep;
  };
  /** young trees answer to the two hero windows only while short; taller ones throw their shade much further */
  const youngWindows = (h: number) => (h < 6 ? SUN_WINDOWS.slice(0, 2) : SUN_WINDOWS);
  const slopeNy = (x: number, z: number) => {
    const e = 1.2;
    const gx = (groundY(x + e, z) - groundY(x - e, z)) / (2 * e), gz = (groundY(x, z + e) - groundY(x, z - e)) / (2 * e);
    return 1 / Math.hypot(gx, 1, gz);
  };
  /** ground a young tree may grow on: not in the water, on the gravel bar or on a bare steep face */
  const groundOk = (x: number, z: number, minEdge: number, minNy: number) => {
    const d = Math.abs(z - river.zAt(x));
    if (d < 48) {
      sampleTerrain(x, z, tp);
      if (tp.edge < minEdge || tp.gravel > 0.25) return false;
    }
    return slopeNy(x, z) >= minNy;
  };
  /**
   * One tree. `young`: a sapling or young tree, buried a little deeper so its lowest branches touch the ground.
   * `h0`..`h5` are per-tree hashes.
   */
  const make = (x: number, z: number, v: number, scale: number, young: boolean, hs: (k: number) => number,
    o: { lean?: number; leanDir?: number; dead?: boolean; narrow?: number } = {}): TreeInst => {
    const T = TREE_VARIANTS[v];
    const sxw = (0.8 + 0.42 * hs(13)) * (o.narrow ?? 1);
    const h = T.height * scale, r = T.crownR * scale * sxw, cb = T.crownBase * scale;
    const y = groundY(x, z) - 0.12 - (young ? 0.12 + 0.18 * Math.min(scale, 1) : 0);
    const jt = hs(7), jb = hs(8);
    const lean = o.lean ?? (hs(14) < 0.16 ? 0.025 + 0.08 * hs(15) : 0.012 * hs(15));
    return {
      x, y, z, yaw: hs(9) * Math.PI * 2, scale, v, rank: hs(10),
      jitter: [0.9 + 0.16 * jt, 0.92 + 0.13 * jb, 0.9 + 0.16 * (1 - jt)],
      h, r, cb, sx: sxw, tilt: lean, tiltDir: o.leanDir ?? hs(16) * Math.PI * 2, dead: o.dead ?? T.snag ?? false,
    };
  };

  const nx = Math.ceil((FOREST_REGION.x1 - FOREST_REGION.x0) / CELL);
  const nz = Math.ceil((FOREST_REGION.z1 - FOREST_REGION.z0) / CELL);
  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) {
      for (let sub = 0; sub < 2; sub++) {
        candidates++;
        const hs = (k: number) => hash2(i, j, k + sub * 40);
        // the whole cell: a 0.15..0.85 jitter kept the trees on an even lattice
        const x = FOREST_REGION.x0 + (i + hs(1)) * CELL;
        const z = FOREST_REGION.z0 + (j + hs(2)) * CELL;
        const u = hs(3);
        // far from the channel only the side and a rough distance matter (the full terrain sample is ~50 us)
        const dz = z - river.zAt(x);
        if (Math.abs(dz) > 48) {
          tp.edge = Math.abs(dz) - 5;
          tp.side = dz > 0 ? 1 : -1;
          tp.gravel = 0;
        } else {
          sampleTerrain(x, z, tp);
        }
        if (tp.edge < 1.5) continue;
        const north = tp.side < 0;
        const ny = slopeNy(x, z);
        const nA = fbm(x * 0.016 + 3.1, z * 0.016 - 7.7, 3);
        const nB = fbm(x * 0.05 - 1.3, z * 0.05 + 4.4, 2);
        // clumps ~10-15 m across with gaps between them
        const clump = smoothstep(-0.32, 0.32, fbm(x * 0.075 + 11.3, z * 0.075 - 5.1, 2));
        let p: number;
        let edgeBand = 0, forestF = 0;
        let zone: 'shore' | 'north' | 'south' | 'meadow';
        if (north) {
          // the shaded north bank: forest right down to the water, thinning only on bare cut faces
          p = smoothstep(1.5, 4.5, tp.edge) * 0.95;
          zone = tp.edge < 14 ? 'shore' : 'north';
          forestF = 1;
        } else {
          // the player's side: an open bar and meadow with lone trees, the forest from ~25 m back
          const fe = 25 + nA * 10;
          forestF = smoothstep(fe - 6, fe + 8, tp.edge);
          // the forest edge is not a wall of trunks: young trees and saplings step down from it into the meadow
          edgeBand = smoothstep(fe - 16, fe - 5, tp.edge) * (1 - smoothstep(fe + 3, fe + 9, tp.edge));
          const lone = smoothstep(3, 6, tp.edge) * 0.05 * smoothstep(0.1, 0.4, nB + 0.2);
          p = Math.max(forestF * 0.9, lone);
          zone = forestF > 0.5 ? 'south' : tp.edge < 10 ? 'shore' : 'meadow';
        }
        // never on the gravel bar or bare steep faces; patches of denser and thinner stand
        const ground = (1 - smoothstep(0.15, 0.5, tp.gravel)) * smoothstep(0.62, 0.84, ny);
        p *= ground * clamp(0.82 + nA * 1.3 + nB * 0.35, 0.12, 1.15);
        // clumps and gaps; the second candidate of a cell only grows where the clump is dense (crowns overlap); the
        // north bank's second candidates are denser (a closed roof of crowns from above)
        p *= sub === 0 ? 0.56 + 0.72 * clump : Math.max(0, clump - (north ? 0.45 : 0.58)) * (north ? 1.9 : 1.6) * forestF;
        // clearings, footpaths and the camera sightlines
        for (const [cx, cz, r] of CLEARINGS) p *= smoothstep(r, r + 3.5, Math.hypot(x - cx, z - cz));
        p *= smoothstep(1.8, 3.8, pathDist(x, z));
        for (const [ex, ez, dx, dz2, len, rad] of lines) {
          const t = clamp((x - ex) * dx + (z - ez) * dz2, -4, len);
          const d = Math.hypot(x - ex - dx * t, z - ez - dz2 * t);
          p *= smoothstep(rad, rad + 2.5, d);
        }
        // young trees crowd the shore and the south forest edge (on the north bank the fringe pass below adds
        // thickets of young spruce and fir right at the water)
        const ps = sub > 0 ? 0 : (north ? 0.3 * smoothstep(1.4, 2.5, tp.edge) * (1 - smoothstep(7, 11, tp.edge))
          : 0.035 * smoothstep(2.5, 4, tp.edge) * (1 - smoothstep(9, 14, tp.edge)) + 0.7 * edgeBand * (0.35 + 0.8 * clump))
          * ground * clamp(0.8 + nB, 0.2, 1.2);
        const w = hs(4);
        // the shore up-sun of the pool keeps its tall lodgepoles (the low sun passes under their crowns onto the
        // water) but behind a ragged edge of spruce and fir with branches to the ground
        const upSun = north && x > -2 && x < 40 && tp.edge < 30;
        // the first metres of the south forest face the meadow: spruce and fir clothed to the ground, not bare boles
        const southEdge = zone === 'south' && forestF < 0.8;
        const tryTree = (young: boolean): TreeInst | null => {
          let v: number;
          if (young) v = pick(YOUNG, w);
          else if (zone === 'shore' && upSun) v = tp.edge < 7
            ? pick([['spruce_c', 2], ['fir_a', 1.6], ['fir_b', 1.2], ['lodgepole_b', 1], ['spruce_a', 0.8]], w)
            : pick([['lodgepole_b', 3.4], ['lodgepole_a', 2.6], ['spruce_a', 1.2], ['spruce_c', 1], ['fir_a', 0.8]], w);
          else if (zone === 'shore') v = pick([['spruce_a', 3], ['spruce_c', 3], ['fir_a', 2.5], ['fir_b', 2], ['lodgepole_c', 1.2], ['spruce_b', 1]], w);
          else if (zone === 'meadow') v = pick([['ponderosa_b', 3], ['spruce_c', 2], ['fir_b', 1.5], ['ponderosa_a', 1]], w);
          // shaded north slopes are spruce-fir forest; the lodgepoles' high, thin crowns let the floor show from above
          else if (zone === 'north') v = pick([['spruce_a', 3], ['spruce_b', 2.5], ['fir_a', 2], ['lodgepole_a', 1.6], ['lodgepole_b', 1.2], ['spruce_c', 1], ['lodgepole_c', 0.6], ['ponderosa_a', 0.3]], w);
          else if (southEdge) v = pick([['spruce_a', 3], ['spruce_c', 2], ['fir_a', 2], ['spruce_b', 1], ['ponderosa_b', 0.8], ['lodgepole_a', 0.6]], w);
          else v = pick([['spruce_a', 2.5], ['spruce_b', 2], ['lodgepole_a', 2], ['lodgepole_b', 1.5], ['fir_a', 1.5], ['ponderosa_a', 1.2], ['ponderosa_b', 0.8], ['fir_b', 0.6]], w);
          let T = TREE_VARIANTS[v];
          // strong size variation (round 1: 0.82-1.12 read as one even-aged stand); young trees graded up toward
          // the forest they step down from
          let scale = young
            ? ((1.4 + 5.6 * Math.pow(hs(5), 1.2)) * (north ? 1 : 0.75 + 0.5 * smoothstep(0, 1, edgeBand))) / T.height
            : 0.58 + 0.66 * Math.pow(hs(5), 0.85);
          // a few dead snags in the forest: silver, cracked, broken-topped, with branch stubs
          const dead = !young && zone !== 'meadow' && (T.kind === 'lodgepole' || T.kind === 'spruce') && hs(18) < 0.045;
          if (dead) {
            v = T.kind === 'lodgepole' ? V.snag_a : V.snag_b;
            T = TREE_VARIANTS[v];
            scale = 0.6 + 0.55 * hs(5);
          }
          const narrow = young ? 1 : 1 - 0.14 * clump;
          const h = T.height * scale, r = T.crownR * scale, cb = T.crownBase * scale;
          const y = groundY(x, z) - 0.12;
          // saplings only answer to the two hero windows (their short shadows are welcome dapples elsewhere)
          if (hs(young ? 12 : 6) > sunKeep(x, y, z, h, r, cb, young ? youngWindows(h) : SUN_WINDOWS)) {
            sunThinned++;
            return null;
          }
          if (spacing.near(x, z, young ? 0.6 : 0.9, young ? 0.35 : 0.25)) return null;
          // the undercut north bank tips its shore trees toward the water
          const shoreLean = north && tp.edge < 9 && hs(17) < 0.35;
          if (dead) snags++;
          return make(x, z, v, scale, young, hs, {
            narrow,
            dead,
            ...(shoreLean ? { lean: 0.05 + 0.09 * hs(19), leanDir: Math.PI / 2 + (hs(20) - 0.5) * 1.2 } : {}),
          });
        };
        let tree: TreeInst | null = null;
        if (u < p) tree = tryTree(false);
        if (!tree && hs(11) < ps && !blocked(x, z, 1)) tree = tryTree(true);
        if (tree) push(tree);
      }
    }
  }

  // the north shore's young spruce and fir right at the water (what the pool mirrors): thickets of very different
  // sizes, crowns overlapping, gaps between them, spread over the first ~12 m of the bank
  const fr = { near: 0, lines: 0, sun: 0, placed: 0 };
  for (let x = -62; x <= 82; x += 1.4) {
    const zc = river.zAt(x);
    for (let z = zc - 26; z < zc - 1.5; z += 1.4) {
      const k = Math.round(x * 13.7 + z * 7.1);
      const hs = (n: number) => hash2(k, n, 31);
      const px = x + (hs(1) - 0.5) * 1.3, pz = z + (hs(2) - 0.5) * 1.3;
      sampleTerrain(px, pz, tp);
      if (tp.side > 0 || tp.edge < 1.4 || tp.edge > 14 || tp.gravel > 0.3) continue;
      const cl = smoothstep(-0.12, 0.22, fbm(px * 0.16 + 2.7, pz * 0.16 - 8.1, 2));
      const band = smoothstep(1.4, 2.4, tp.edge) * (1 - smoothstep(8, 13, tp.edge));
      const pf = 0.95 * band * cl * smoothstep(0.55, 0.8, slopeNy(px, pz));
      if (hs(3) >= pf) continue;
      const v = pick(YOUNG, hs(4));
      const T = TREE_VARIANTS[v];
      // mostly small, a few nearly grown
      let hT = 1.0 + 7.0 * Math.pow(hs(5), 1.5);
      if (spacing.near(px, pz, 0.3 + 0.07 * hT, 0.15)) { fr.near++; continue; }
      if (blocked(px, pz, 1)) { fr.lines++; continue; }
      const gy = groundY(px, pz);
      if (hs(6) > sunKeep(px, gy, pz, hT, T.crownR * hT / T.height, T.crownBase * hT / T.height, youngWindows(hT))) {
        // where a tall one would shade the water the sun must reach, a short one still clothes the bank (C4's far
        // shore: the trunks of the high-crowned lodgepoles the low sun passes under)
        hT = Math.min(2.2, Math.max(0.8, hT * 0.3));
        if (hs(21) > sunKeep(px, gy, pz, hT, T.crownR * hT / T.height, T.crownBase * hT / T.height, youngWindows(hT))) {
          sunThinned++;
          fr.sun++;
          continue;
        }
      }
      const scale = hT / T.height;
      fr.placed++;
      push(make(px, pz, v, scale, true, hs, hs(17) < 0.3 ? { lean: 0.04 + 0.08 * hs(18), leanDir: Math.PI / 2 + (hs(19) - 0.5) } : {}));
    }
  }

  // thickets: crowded clumps of young spruce and fir in the gaps of the north forest, stepping down from the south
  // forest edge into the meadow, now and then in the south forest, a few small groups out in the meadow (round 2's
  // young trees stood one by one on bare ground: "a Christmas-tree lot" from above)
  let thickets = 0, thicketTrees = 0;
  for (let x = -150; x <= 160; x += 6.5) {
    const zc0 = river.zAt(x);
    for (let z = zc0 - 82; z <= zc0 + 72; z += 6.5) {
      const k = Math.round(x * 3.7 + z * 17.3);
      const hs = (n: number) => hash2(k, n, 71);
      const cx = x + (hs(1) - 0.5) * 5.5, cz = z + (hs(2) - 0.5) * 5.5;
      const dz = cz - river.zAt(cx), north = dz < 0, d = Math.abs(dz);
      if (d < 7) continue;
      let pt: number, hMax: number;
      if (north) {
        pt = 0.26 * smoothstep(12, 18, d) * (1 - smoothstep(60, 82, d));
        hMax = 9;
      } else {
        const fe = 25 + fbm(cx * 0.016 + 3.1, cz * 0.016 - 7.7, 3) * 10;
        const edgeT = smoothstep(fe - 12, fe - 5, d) * (1 - smoothstep(fe + 2, fe + 8, d));
        pt = 0.6 * edgeT + 0.08 * smoothstep(fe + 2, fe + 8, d) * (1 - smoothstep(55, 72, d)) + 0.015 * smoothstep(9, 14, d);
        hMax = 5 + 6 * edgeT;
      }
      pt *= smoothstep(-0.3, 0.2, fbm(cx * 0.05 + 7.3, cz * 0.05 - 2.2, 2) + 0.1);
      if (hs(3) >= pt || blocked(cx, cz, 2.5)) continue;
      thickets++;
      const n = 4 + Math.floor(hs(4) * 12);
      const R = 1.5 + 2.8 * hs(5);
      for (let m = 0; m < n; m++) {
        const hm = (q: number) => hash2(k, m * 13 + q, 72);
        const a = m * 2.39996 + hm(1) * 0.9, rr = R * Math.sqrt((m + 0.5) / n) * (0.7 + 0.6 * hm(2));
        const px = cx + Math.cos(a) * rr, pz = cz + Math.sin(a) * rr;
        if (!groundOk(px, pz, 2.5, 0.7) || blocked(px, pz, 0.8)) continue;
        const v = pick(YOUNG, hm(3));
        const T = TREE_VARIANTS[v];
        // a few nearly grown in the middle, most knee to head high, small ones round the edge
        const hT = 0.8 + (hMax - 0.8) * Math.pow(hm(4), 1.5) * (1.15 - 0.55 * (rr / R));
        const scale = hT / T.height;
        if (spacing.near(px, pz, 0.26 + 0.07 * hT, 0.12)) continue;
        const h = T.height * scale, r = T.crownR * scale, cb = T.crownBase * scale;
        if (hm(5) > sunKeep(px, groundY(px, pz), pz, h, r, cb, youngWindows(h))) {
          sunThinned++;
          continue;
        }
        push(make(px, pz, v, scale, true, hm, hm(6) < 0.25 ? { lean: 0.03 + 0.08 * hm(7) } : {}));
        thicketTrees++;
      }
    }
  }

  // the north-bank canopy closes: mid-size spruce and fir, branched to the ground, fill the holes between the tall
  // crowns (from above a closed, lumpy roof with little floor showing), tested against the sun windows like the
  // tall trees
  let infill = 0;
  for (let x = -150; x <= 160; x += 4.0) {
    const zc0 = river.zAt(x);
    for (let z = zc0 - 92; z <= zc0 - 9; z += 4.0) {
      const k = Math.round(x * 7.9 + z * 3.1);
      const hs = (n: number) => hash2(k, n, 83);
      const px = x + (hs(1) - 0.5) * 3.6, pz = z + (hs(2) - 0.5) * 3.6;
      const d = river.zAt(px) - pz;
      if (d < 9 || hs(3) > 0.8 * (1 - smoothstep(72, 92, d))) continue;
      if (spacing.near(px, pz, 2.0, 0.4)) continue;
      if (!groundOk(px, pz, 4, 0.68) || blocked(px, pz, 1.5)) continue;
      const v = pick([['spruce_c', 2], ['fir_a', 2], ['spruce_t', 2], ['spruce_y', 1.4], ['spruce_a', 1]], hs(4));
      const T = TREE_VARIANTS[v];
      const scale = (6.5 + 8 * Math.pow(hs(5), 0.9)) / T.height;
      const h = T.height * scale, r = T.crownR * scale, cb = T.crownBase * scale;
      if (hs(6) > sunKeep(px, groundY(px, pz), pz, h, r, cb, SUN_WINDOWS)) {
        sunThinned++;
        continue;
      }
      push(make(px, pz, v, scale, true, hs));
      infill++;
    }
  }

  // understory: young conifers in clumps filling the gaps of the north-bank forest and the south forest edge, so you
  // cannot see far in between the trunks
  let under = 0;
  for (let x = -150; x <= 160; x += 2.6) {
    const zc = river.zAt(x);
    for (let z = zc - 95; z <= zc + 80; z += 2.6) {
      const k = Math.round(x * 5.3 + z * 11.9);
      const hs = (n: number) => hash2(k, n, 57);
      const px = x + (hs(1) - 0.5) * 2.4, pz = z + (hs(2) - 0.5) * 2.4;
      const dz = pz - river.zAt(px);
      const north = dz < 0;
      const d = Math.abs(dz);
      if (d < 9) continue;
      let pu: number;
      if (north) {
        pu = 0.24 * smoothstep(9, 16, d) * (1 - smoothstep(70, 95, d));
      } else {
        const fe = 25 + fbm(px * 0.016 + 3.1, pz * 0.016 - 7.7, 3) * 10;
        pu = 0.22 * smoothstep(fe - 4, fe + 4, d) * (1 - smoothstep(60, 80, d));
      }
      pu *= smoothstep(0.0, 0.4, fbm(px * 0.11 - 4.4, pz * 0.11 + 9.2, 2) + 0.1);
      if (hs(3) >= pu) continue;
      if (d < 48) {
        sampleTerrain(px, pz, tp);
        if (tp.gravel > 0.2 || tp.edge < 6) continue;
      }
      if (slopeNy(px, pz) < 0.7) continue;
      if (blocked(px, pz, 2)) continue;
      const v = pick(YOUNG, hs(4));
      const T = TREE_VARIANTS[v];
      const scale = (1.2 + 5.5 * Math.pow(hs(5), 1.3)) / T.height;
      if (spacing.near(px, pz, 0.5 + 0.7 * scale, 0.2)) continue;
      const h = T.height * scale, r = T.crownR * scale, cb = T.crownBase * scale;
      if (hs(6) > sunKeep(px, groundY(px, pz), pz, h, r, cb, youngWindows(h))) continue;
      push(make(px, pz, v, scale, true, hs));
      under++;
    }
  }

  // seedlings and squat shrubs on the forest floor around the play area (a young spruce at knee height, squashed and
  // wide, reads as a juniper or huckleberry clump); none cast shadows (see castsShadow)
  let low = 0;
  for (let x = -75; x <= 90; x += 1.7) {
    const zc = river.zAt(x);
    for (let z = zc - 48; z <= zc + 46; z += 1.7) {
      const k = Math.round(x * 9.1 + z * 4.3);
      const hs = (n: number) => hash2(k, n, 97);
      const px = x + (hs(1) - 0.5) * 1.6, pz = z + (hs(2) - 0.5) * 1.6;
      const dz = pz - river.zAt(px), north = dz < 0, d = Math.abs(dz);
      if (d < 4) continue;
      const fe = 25 + fbm(px * 0.016 + 3.1, pz * 0.016 - 7.7, 3) * 10;
      const inF = north ? smoothstep(3, 8, d) : smoothstep(fe - 8, fe + 2, d);
      const pl = 0.09 * inF * smoothstep(-0.15, 0.3, fbm(px * 0.21 + 5.1, pz * 0.21 - 3.3, 2));
      if (hs(3) >= pl) continue;
      if (!groundOk(px, pz, 2, 0.72) || blocked(px, pz, 0.6)) continue;
      if (spacing.near(px, pz, 0.45, 0.1)) continue;
      const shrub = hs(4) < 0.55;
      const v = pick(shrub ? [['spruce_y', 2], ['fir_y', 1], ['spruce_t', 0.4]] : YOUNG, hs(5));
      const T = TREE_VARIANTS[v];
      const hT = shrub ? 0.45 + 0.75 * hs(6) : 0.3 + 1.0 * Math.pow(hs(6), 1.4);
      push(make(px, pz, v, hT / T.height, true, hs, shrub ? { narrow: 1.5 + 0.6 * hs(7) } : {}));
      low++;
    }
  }
  if (stats) {
    stats.candidates = candidates;
    stats.placed = out.length;
    stats.sunThinned = sunThinned;
    stats.fringe = fr;
    stats.understory = under;
    stats.thickets = thickets;
    stats.thicketTrees = thicketTrees;
    stats.infill = infill;
    stats.low = low;
    stats.snags = snags;
  }
  return out;
}

/**
 * Canopy over a ground point, for the ground shaders and the grass: x = crown cover 0..1 (spruce and fir crowns
 * shade more than the open pine crowns), y = closeness to a trunk 0..1 (litter and roots at the foot).
 */
export class CanopyField {
  private cells = new Map<number, number[]>();
  constructor(private trees: TreeInst[], private cell = 6) {
    trees.forEach((t, i) => {
      // a snag's stubs shade nothing
      if (t.dead) return;
      const k = this.key(Math.floor(t.x / cell), Math.floor(t.z / cell));
      let a = this.cells.get(k);
      if (!a) this.cells.set(k, (a = []));
      a.push(i);
    });
  }
  private key(i: number, j: number) {
    return (i + 4096) * 8192 + (j + 4096);
  }
  sample(x: number, z: number, out: [number, number]) {
    let open = 1, trunk = 0;
    const ci = Math.floor(x / this.cell), cj = Math.floor(z / this.cell);
    for (let a = -1; a <= 1; a++) {
      for (let b = -1; b <= 1; b++) {
        const list = this.cells.get(this.key(ci + a, cj + b));
        if (!list) continue;
        for (const i of list) {
          const t = this.trees[i];
          const d = Math.hypot(x - t.x, z - t.z);
          const kind = TREE_VARIANTS[t.v].kind;
          const dense = kind === 'spruce' || kind === 'fir' ? 0.8 : 0.5;
          // the crown's footprint, a little wider than its silhouette radius where the lowest branches spread
          const c = dense * smoothstep(t.r * 1.35, t.r * 0.35, d) * Math.min(1, t.h / 8);
          open *= 1 - c;
          trunk = Math.max(trunk, smoothstep(2.2 * Math.sqrt(t.scale), 0.25, d));
        }
      }
    }
    out[0] = 1 - open;
    out[1] = trunk;
    return out;
  }
}

/** Fraction of far trees kept at distance d (m): full density to 120 m, a third by 360 m. */
export const keepAt = (d: number) => 1 - 0.66 * smoothstep(120, 360, d);
