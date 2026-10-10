import { fbm } from '../world/noise';
import { LAYOUT } from '../world/Layout';
import {
  CABIN, CABIN_GEO, CABIN_PROPS, CAMPFIRE, DOCK, FIRE_GEO, FIRE_PIECES, cabinX, cabinZ, fireX, fireZ, toCabinX, toCabinZ,
} from './layout';

/**
 * The ground of a camp someone lives at: packed dirt round the fire in a ragged yard reaching out toward the seats,
 * the firewood and the paths, darkened by ash and charcoal kicked out of the ring; paths worn from the door to the
 * fire, from the door down to the dock, from the fire to the dock and to the woodpile, wandering in gentle curves,
 * their beaten middles paler and smoother; trampled earth along the foot of the walls, widest before the porch; chips
 * round the woodpile and the chopping block. Round 3 (critic r2: "a neat oval of pale, cold blue-grey grit with a
 * crisp edge, like a gravel car park"; the paths "a narrow, dark, dead-straight strip ... like mulch", "from C6 I
 * can't trace any path"). Pure math (no three.js), sampled once per vertex of the corridor grid (World.ts bakes it
 * into `aCampG`, the terrain draws it with ground.ts `campGroundLayer`) and per grass tuft (layout.ts campGrassMask).
 * Imports layout.ts, which imports this module back: nothing here touches layout at module load.
 */
export interface CampWear {
  /**
   * 0..1 packed bare dirt (the turf worn away); 1..2 on the beaten middle of a path (1 + how polished). World.ts
   * bakes it as it is; campGrassMask and the terrain layer split it (min(worn, 1), worn - 1).
   */
  worn: number;
  /** 0..1 ash and charcoal kicked out of the fire over the dirt */
  ash: number;
  /** 0..1 wood chips and bark (where wood is split and carried) */
  chips: number;
}

const sstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

interface P2 { x: number; z: number }
interface Path { pts: P2[]; hw: number; x0: number; x1: number; z0: number; z1: number }
let anchors: { block: P2; stack: P2; wood: P2; paths: Path[]; lobes: number[] } | null = null;

/**
 * A path wandering in gentle curves from a to b: a cubic Bezier whose two handles lean to opposite sides by `bow`
 * and `bow2` (m; an S when their signs differ), as a polyline with its bounds
 */
function curve(a: P2, b: P2, bow: number, bow2: number, hw: number): Path {
  const dx = b.x - a.x, dz = b.z - a.z, l = Math.hypot(dx, dz);
  const px = -dz / l, pz = dx / l;
  const c1 = { x: a.x + dx / 3 + px * bow, z: a.z + dz / 3 + pz * bow };
  const c2 = { x: a.x + (2 * dx) / 3 + px * bow2, z: a.z + (2 * dz) / 3 + pz * bow2 };
  const pts: P2[] = [];
  for (let i = 0; i <= 20; i++) {
    const t = i / 20, u = 1 - t;
    const k0 = u * u * u, k1 = 3 * u * u * t, k2 = 3 * u * t * t, k3 = t * t * t;
    pts.push({ x: k0 * a.x + k1 * c1.x + k2 * c2.x + k3 * b.x, z: k0 * a.z + k1 * c1.z + k2 * c2.z + k3 * b.z });
  }
  const m = hw * 2.5;
  return {
    pts, hw, x0: Math.min(...pts.map((p) => p.x)) - m, x1: Math.max(...pts.map((p) => p.x)) + m,
    z0: Math.min(...pts.map((p) => p.z)) - m, z1: Math.max(...pts.map((p) => p.z)) + m,
  };
}

function polyDist(x: number, z: number, pts: P2[]) {
  let best = Infinity;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    const dx = b.x - a.x, dz = b.z - a.z;
    const t = Math.min(1, Math.max(0, ((x - a.x) * dx + (z - a.z) * dz) / (dx * dx + dz * dz)));
    best = Math.min(best, Math.hypot(x - a.x - dx * t, z - a.z - dz * t));
  }
  return best;
}

function getAnchors() {
  if (anchors) return anchors;
  const st = CABIN_GEO.step;
  const sx = (st.x0 + st.x1) / 2;
  // in front of the door step (the low stone when the porch stands high)
  const door = { x: cabinX(sx, st.z1 + 0.9), z: cabinZ(sx, st.z1 + 0.9) };
  const dock = { x: DOCK.x, z: DOCK.zShore + 0.7 };
  const fire = { x: CAMPFIRE.x, z: CAMPFIRE.z };
  const fw = FIRE_PIECES.firewood;
  const wood = { x: fireX(fw.x, fw.z), z: fireZ(fw.x, fw.z) };
  const ws = CABIN_PROPS.woodStack, cb = CABIN_PROPS.choppingBlock;
  const stack = { x: cabinX(ws.x, ws.z + 0.55), z: cabinZ(ws.x, ws.z + 0.55) };
  const block = { x: cabinX(cb.x, cb.z), z: cabinZ(cb.x, cb.z) };
  // part 3's two footpaths (ForestLayout PATHS, kept clear of trees, props and grass blades): the fire to beside the
  // porch, where the woodpile stands, and the fire to the dock. Copied here, not imported: ForestLayout pulls in
  // three.js and this module stays pure. The worn paths follow them within their cleared strips, bowed a little.
  const p3a = { x: LAYOUT.cabin.x - 1.5, z: LAYOUT.cabin.z - 4.0 };
  const p3b = { x: LAYOUT.dock.x, z: LAYOUT.dock.zBase + 0.6 };
  anchors = {
    block, stack, wood,
    paths: [
      // [a, b, the handles' lean to either side (m), half width of the worn strip (m)]; the ones along part 3's
      // cleared strips lean less
      curve(door, fire, -1.6, 0.5, 0.34),
      curve(door, dock, 1.7, -1.3, 0.34),
      curve(fire, p3b, 0.55, -0.35, 0.32),
      curve(fire, p3a, -0.5, 0.3, 0.3),
      curve(block, stack, 0.15, -0.1, 0.24),
      curve(door, block, -0.25, 0.15, 0.22),
      curve(fire, wood, 0.15, -0.1, 0.24),
    ],
    // the yard reaches out toward where people go: the seats, the firewood, the bucket, the paths out of it
    lobes: [FIRE_PIECES.seat1, FIRE_PIECES.seat2, fw, FIRE_PIECES.bucket].map((p) => Math.atan2(fireZ(p.x, p.z) - fire.z, fireX(p.x, p.z) - fire.x))
      .concat([door, dock, p3a].map((p) => Math.atan2(p.z - fire.z, p.x - fire.x))),
  };
  return anchors;
}

export function campGround(x: number, z: number, out: CampWear): CampWear {
  const A = getAnchors();
  // ragged edges at two scales
  const n1 = fbm(x * 0.9 + 3.1, z * 0.9 - 1.7, 2), n2 = fbm(x * 3.3 - 5.2, z * 3.3 + 2.4, 2);
  const rag = n1 * 0.35 + n2 * 0.15;
  // the fire yard: an irregular patch of packed dirt, lobed toward the seats, the wood and the paths, its edge
  // breaking up over half a metre or more into trampled grass
  const dx = x - CAMPFIRE.x, dz = z - CAMPFIRE.z;
  const df = Math.hypot(dx, dz), th = Math.atan2(dz, dx);
  let R = 1.4 + 0.26 * Math.sin(2 * th + 0.6) + 0.17 * Math.sin(3 * th + 2.2) + 0.1 * Math.sin(5 * th + 1.3);
  for (const l of A.lobes) R += 0.5 * Math.exp(-((Math.atan2(Math.sin(th - l), Math.cos(th - l)) / 0.32) ** 2));
  let worn = 1 - sstep(R - 0.8, R + 0.4, df + rag * 1.1);
  // ash and charcoal kicked out of the ring: darkest against the stones, out in uneven tongues
  const tongue = 0.25 * Math.sin(4 * th + 1.0) + 0.15 * Math.sin(7 * th + 2.6);
  const ash = (1 - sstep(FIRE_GEO.ringR - 0.05, 1.05 + tongue + 0.25 * n1, df + rag * 0.5)) * (0.65 + 0.35 * sstep(-0.2, 0.3, n2));
  // paths: a worn strip with soft, ragged edges and a paler beaten middle, turf in tufts here and there along it
  const patchy = 0.75 + 0.25 * sstep(-0.25, 0.2, fbm(x * 1.7 - 9.1, z * 1.7 + 4.4, 2));
  const wx = x + fbm(z * 0.3, 7.3, 2) * 0.6, wz = z + fbm(x * 0.3, -2.1, 2) * 0.6;
  let polish = 0;
  for (const P of A.paths) {
    if (wx < P.x0 || wx > P.x1 || wz < P.z0 || wz > P.z1) continue;
    const d = polyDist(wx, wz, P.pts) + rag * 0.3;
    worn = Math.max(worn, (1 - sstep(P.hw * 0.45, P.hw * 2.0, d)) * patchy);
    // (round 4 wip: a beaten middle ~0.3 m across fell between the 0.35 m vertices of the grid it is baked on and
    // never showed) ~0.6 m, fading out over the worn edge
    polish = Math.max(polish, (1 - sstep(P.hw * 0.25, P.hw * 1.3, d + rag * 0.1)) * (0.6 + 0.4 * patchy));
  }
  // trampled along the foot of the walls, a broad patch before the porch and the step
  const lx = toCabinX(x, z), lz = toCabinZ(x, z);
  const g = CABIN_GEO;
  const dWall = Math.max(Math.abs(lx) - g.walls.hx, Math.abs(lz) - g.walls.hz);
  worn = Math.max(worn, (1 - sstep(0.15, 0.95, dWall + rag * 0.5)) * 0.85);
  const dPorch = Math.max(g.porch.x0 - 0.4 - lx, lx - g.porch.x1 - 0.5, g.porch.z0 - lz, lz - g.step.z1 - 1.3);
  worn = Math.max(worn, 1 - sstep(0.0, 0.8, dPorch + rag * 0.6));
  // chips: round the chopping block and the woodpile, the firewood by the fire, a few on the paths between
  const dBlock = Math.hypot(x - A.block.x, z - A.block.z), dStack = Math.hypot(x - A.stack.x, z - A.stack.z);
  const dWood = Math.hypot(x - A.wood.x, z - A.wood.z);
  let chips = Math.max(1 - sstep(0.4, 1.6, dBlock + rag), (1 - sstep(0.4, 1.3, dStack + rag)) * 0.7, (1 - sstep(0.3, 1.0, dWood + rag)) * 0.8);
  worn = Math.max(worn, chips * 0.9);
  chips = Math.max(chips, Math.min(1, worn) * 0.15);
  out.worn = Math.min(1, Math.max(0, worn)) + (worn > 0.5 ? polish : 0);
  out.ash = Math.min(1, Math.max(0, ash));
  out.chips = Math.min(1, Math.max(0, chips));
  return out;
}

/** Quick reject for callers that sample everywhere: the camp's ground lies inside this box (world). */
export function nearCamp(x: number, z: number) {
  return x > Math.min(CAMPFIRE.x, CABIN.x - 6) - 4 && x < Math.max(DOCK.x, CABIN.x + 6) + 4 && z > DOCK.zShore - 2 && z < CABIN.z + 7;
}
