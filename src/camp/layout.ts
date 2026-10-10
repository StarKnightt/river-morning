import { LAYOUT } from '../world/Layout';
import { fbm } from '../world/noise';
import { campGround, nearCamp, type CampWear } from './groundMask';

/**
 * Part 4 placement: the cabin, the dock, the campfire set and the props around them, in world metres (Y up, the
 * river flowing +X, the camp on the south bank). Pure math (no three.js), so part 3's grass and prop placers and
 * CPU probes can import it. Cabin-local numbers are measured from cabin.glb (its door and porch face +Z).
 */

/** The GLB's door and porch face +Z; half a turn puts them toward the river (north, -Z). */
export const CABIN = { x: LAYOUT.cabin.x, z: LAYOUT.cabin.z, yaw: LAYOUT.cabin.rotY + Math.PI };
const CC = Math.cos(CABIN.yaw), CS = Math.sin(CABIN.yaw);

/** cabin-local (x, z) -> world, as three's rotation.y followed by the translation */
export const cabinX = (lx: number, lz: number) => CABIN.x + lx * CC + lz * CS;
export const cabinZ = (lx: number, lz: number) => CABIN.z - lx * CS + lz * CC;
/** world -> cabin-local */
export const toCabinX = (x: number, z: number) => (x - CABIN.x) * CC - (z - CABIN.z) * CS;
export const toCabinZ = (x: number, z: number) => (x - CABIN.x) * CS + (z - CABIN.z) * CC;

/** cabin.glb, local metres above its base (the bottom of the stone piers) */
export const CABIN_GEO = {
  /** outer log faces at x +-3.0 and z +-2.5 (round logs bulge ~0.15 m further) */
  walls: { hx: 3.15, hz: 2.65 },
  chimney: { x: 3.42, z: 0, hx: 0.47, hz: 0.84 },
  /** porch deck (planks) and its height */
  porch: { x0: -2.3, x1: 0.9, z0: 2.47, z1: 4.07, y: 0.503 },
  /**
   * the door step stone in front of the porch (raised onto a base stone when the porch stands high, else dropped
   * onto the ground): its back edge reaches under the porch boards, past the pier beside it
   */
  step: { x0: -1.18, x1: -0.19, z0: 4.05, z1: 4.66, top: 0.207, bottom: -0.044 },
  porchPosts: [[-2.106, 3.914], [0.725, 3.885]] as Array<[number, number]>,
  /** centre of the flue opening on the chimney crown */
  flue: [3.32, 5.473, 0] as [number, number, number],
  /** bottom of the sill logs (the lowest course) */
  sill: 0.32,
  /**
   * Cabin_PorchStones holds seven stones (three stacked pier pairs and the door step): the step is the only one
   * reaching past z 4.21, tested on a piece's cabin-local bounds
   */
  isStepStone: (_min: { z: number }, max: { z: number }) => max.z > 4.4,
  /** centre of the river-facing window glass (and its outward normal is +Z) */
  frontWindow: [1.4, 1.707, 2.396] as [number, number, number],
};

/** The dock runs unrotated from its shore end at the layout's dock base out along -Z (dock.glb: 6.5 m). */
export const DOCK = {
  x: LAYOUT.dock.x,
  zShore: LAYOUT.dock.zBase,
  length: 6.5,
  /** walkable half width of the planks */
  halfWidth: 0.62,
  /** bearer under the first post pair (local z) and how far its bottom sits below the deck top */
  firstBearerZ: -0.25,
  bearerDepth: 0.38,
  /** post pairs (local z), x = +-postX; the outer pair rises above the deck as mooring posts */
  postZ: [-0.25, -2.31, -4.35, -6.3],
  postX: 0.71,
  postR: 0.15,
  mooringTop: 0.55,
};
export const DOCK_TIP_Z = DOCK.zShore - DOCK.length;

/**
 * Deck height: the first bearer rests on the gravel bar (its bottom 2 cm into it), within the brief's 0.5-0.7 m
 * over the pool. With the bar at ~0.18 m that is ~0.54 m.
 */
export function deckHeight(groundY: (x: number, z: number) => number) {
  const z = DOCK.zShore + DOCK.firstBearerZ;
  const g = Math.max(groundY(DOCK.x - DOCK.postX, z), groundY(DOCK.x + DOCK.postX, z), groundY(DOCK.x, z));
  return Math.min(0.62, Math.max(0.5, g + DOCK.bearerDepth - 0.02));
}

export type SurfaceKind = 'wood' | 'stone';
/** A walkable top that is not terrain (the character lane's WalkSurface shape, plus its kind for footsteps). */
export interface CampSurface {
  readonly name: string;
  readonly kind: SurfaceKind;
  heightAt(x: number, z: number, margin?: number): number | null;
}

/** The heights the camp stands at on the rendered ground, shared by the placement (index.ts) and the walk surfaces. */
export function campHeights(groundY: (x: number, z: number) => number) {
  // the cabin: high enough that the back sill clears the slope by 12 cm, the front piers touching the ground
  const wall: number[] = [];
  for (const [lx, lz] of [[-3, -2.5], [-1, -2.5], [1, -2.5], [3, -2.5], [-3, 0], [3, 0], [-3, 2.5], [-1, 2.5], [1, 2.5], [3, 2.5]]) {
    wall.push(groundY(cabinX(lx, lz), cabinZ(lx, lz)));
  }
  const sorted = [...wall].sort((a, b) => a - b);
  const median = (sorted[4] + sorted[5]) / 2;
  const cabinY = Math.max(median, sorted[sorted.length - 1] - (CABIN_GEO.sill - 0.12));
  const porchY = cabinY + CABIN_GEO.porch.y;
  // Porch steps. The flat drops toward the river, so the porch boards stand ~0.7 m over the ground in front of
  // them: the authored step stone (a little higher) rests on a broader stone set into the ground, and a third one
  // lies on the ground in front, giving risers of ~0.2 m. On level ground the step stone simply sits down.
  const st = CABIN_GEO.step;
  const sx = (st.x0 + st.x1) / 2, sz = (st.z0 + st.z1) / 2;
  const stepGround = groundY(cabinX(sx, sz), cabinZ(sx, sz));
  let stepDrop = Math.min(0, stepGround - 0.03 - (cabinY + st.bottom));
  const lift = Math.max(0, porchY - 0.25 - (cabinY + st.top));
  const stair = cabinY + st.bottom + lift - stepGround > 0.05;
  if (stair) stepDrop = lift;
  const stepY = cabinY + st.top + stepDrop;
  // the low stone lies just in front of the base stone (12 % larger than the step, so 0.34 m in front of its centre)
  const lowAt = { lx: sx + 0.06, lz: sz + 0.62 };
  const lowGround = groundY(cabinX(lowAt.lx, lowAt.lz), cabinZ(lowAt.lx, lowAt.lz));
  const low = stair && stepY - lowGround > 0.3;
  const stepLowY = low ? lowGround + 0.01 + st.top : -Infinity;
  // the dock, and a flat stone (the door step's twin) to step up onto it where it starts on the bar
  const deckY = deckHeight(groundY);
  // (its back edge a few centimetres off the deck's end beam)
  const dockStepAt = { x: DOCK.x + 0.05, z: DOCK.zShore + 0.36 };
  const dockStepGround = groundY(dockStepAt.x, dockStepAt.z);
  const dockStepY = dockStepGround - 0.03 - st.bottom + st.top;
  return {
    cabinY, porchY, stepCentre: { lx: sx, lz: sz }, stepDrop, lift, stair, stepY, low, lowAt, lowGround, stepLowY,
    deckY, dockStepAt, dockStepGround, dockStepY,
  };
}
export type CampHeights = ReturnType<typeof campHeights>;

/**
 * The dock deck, the stone up onto it, the porch boards and the porch steps. Each step reaches back to the edge of
 * the top above it: walkers sample the floor exactly, and a hand's width of bare ground between two tops would be a
 * pit they cannot climb out of (risers over 0.32 m).
 */
export function campSurfaces(h: CampHeights): CampSurface[] {
  const P = CABIN_GEO.porch, st = CABIN_GEO.step;
  const shx = (st.x1 - st.x0) / 2, shz = (st.z1 - st.z0) / 2;
  const ds = h.dockStepAt, lowAt = h.lowAt;
  const out: CampSurface[] = [
    {
      name: 'dock-deck', kind: 'wood',
      heightAt: (x, z, margin = 0) => (Math.abs(x - DOCK.x) <= DOCK.halfWidth + margin && z >= DOCK_TIP_Z - margin && z <= DOCK.zShore + margin ? h.deckY : null),
    },
    {
      // the bar alone is ~0.39 m below the deck
      name: 'dock-step', kind: 'stone',
      heightAt: (x, z, margin = 0) => (Math.abs(x - ds.x) <= shx - 0.02 + margin && z >= DOCK.zShore - margin && z <= ds.z + shz + margin ? h.dockStepY : null),
    },
    {
      name: 'porch', kind: 'wood',
      heightAt: (x, z, margin = 0) => {
        const lx = toCabinX(x, z), lz = toCabinZ(x, z);
        return lx >= P.x0 - margin && lx <= P.x1 + margin && lz >= P.z0 - margin && lz <= P.z1 + margin ? h.porchY : null;
      },
    },
    {
      name: 'porch-step', kind: 'stone',
      heightAt: (x, z, margin = 0) => {
        const lx = toCabinX(x, z), lz = toCabinZ(x, z);
        return lx >= st.x0 - margin && lx <= st.x1 + margin && lz >= Math.min(st.z0, P.z1) - margin && lz <= st.z1 + margin ? h.stepY : null;
      },
    },
  ];
  if (h.low) {
    out.push({
      name: 'porch-step-low', kind: 'stone',
      heightAt: (x, z, margin = 0) => {
        const lx = toCabinX(x, z) - lowAt.lx, lz = toCabinZ(x, z) - lowAt.lz;
        return Math.abs(lx) <= shx + margin && lz >= st.z1 - lowAt.lz - margin && lz <= shz + margin ? h.stepLowY : null;
      },
    });
  }
  return out;
}

/** The campfire set (campfire.glb root = centre of the stone ring on the ground). */
export const CAMPFIRE = { x: LAYOUT.campfire.x, z: LAYOUT.campfire.z, yaw: 0 };
/** fire-local: where the flames start (the log crossing) and the coffee pot's spout tip */
export const FIRE_GEO = {
  flameBase: [0.0, 0.085, -0.01] as [number, number, number],
  /** the ring's outer reach (round 3: two logs rest up on the stones and stick out past them) */
  ringR: 0.7,
  spout: [0.045, 0.411, 0.142] as [number, number, number],
  potLid: [0.13, 0.475, 0.04] as [number, number, number],
};

/**
 * Pieces re-placed in fire-local metres (x, z, yaw deg): the set's own stump seat stays south of the fire (whoever
 * sits there looks over the fire to the river); a second stump to the west; the split firewood moved north-west,
 * off the paths to the dock (east-north-east) and the cabin (south-east); the water bucket beside the seat.
 */
export const FIRE_PIECES = {
  /** the set's own stump seat, where campfire.glb has it (its Stump_Seat pivot) */
  seat1: { x: -0.35, z: 1.15 },
  seat2: { x: -1.3, z: -0.2, yaw: 74 },
  /** (round 2: well clear of the ring; round 1's pile was pushed against the stones) */
  firewood: { x: -1.05, z: -1.25, yaw: -41 },
  bucket: { x: 0.42, z: 1.38, yaw: 30 },
};

/** Props around the cabin, cabin-local (x, z, yaw deg relative to the cabin; y from the ground or the porch). */
export const CABIN_PROPS = {
  /**
   * split wood stacked under the front eave between the porch and the window corner, long side to the wall with a
   * hand's width of air behind it (round 2: by the chimney gable it was hidden from every fixed spot)
   */
  woodStack: { x: 1.86, z: 2.98, yaw: 0 },
  /**
   * chopping block out in the yard before the woodpile, the axe left stuck in it, its handle across the view from
   * the porch spot (round 3: at x 2.25 it stood just outside P4c's frame and was lost in the grass in P4b)
   */
  choppingBlock: { x: 1.05, z: 4.75, yaw: -120 },
  /** last night's lantern on the porch boards beside the door */
  lantern: { x: -1.62, z: 2.76, yaw: 20 },
};

export interface CircleBlock { x: number; z: number; r: number }
export interface BoxBlock { x: number; z: number; hx: number; hz: number; rotY: number }

/**
 * 2D blockers for the player in world x/z (the character lane's Ground.addCircle / addBox format; boxes turn about
 * Y like three's rotation.y). Sizes measured from the GLBs. Not blocked: what is walked on (the dock deck, the porch,
 * the step stones: walk surfaces) and the dock posts under the deck.
 */
export function campBlockers(): { circles: CircleBlock[]; boxes: BoxBlock[] } {
  const g = CABIN_GEO;
  const deg = Math.PI / 180;
  const boxes: BoxBlock[] = [
    // the log walls (round logs bulge to +-3.15 / +-2.65; the corner log ends a few centimetres further)
    { x: CABIN.x, z: CABIN.z, hx: g.walls.hx, hz: g.walls.hz, rotY: CABIN.yaw },
    { x: cabinX(g.chimney.x, g.chimney.z), z: cabinZ(g.chimney.x, g.chimney.z), hx: g.chimney.hx, hz: g.chimney.hz, rotY: CABIN.yaw },
  ];
  // the woodpile under the front eave: 1.6 m long; its pieces reach 0.27 m back and up to 0.34 m out from the pivot
  // (the ones pushed proud of the face)
  const ws = CABIN_PROPS.woodStack;
  boxes.push({ x: cabinX(ws.x, ws.z + 0.035), z: cabinZ(ws.x, ws.z + 0.035), hx: 0.8, hz: 0.31, rotY: CABIN.yaw + ws.yaw * deg });
  const circles: CircleBlock[] = g.porchPosts.map(([px, pz]) => ({ x: cabinX(px, pz), z: cabinZ(px, pz), r: 0.13 }));
  // the lantern on the porch boards
  const ln = CABIN_PROPS.lantern;
  circles.push({ x: cabinX(ln.x, ln.z), z: cabinZ(ln.x, ln.z), r: 0.08 });
  // the chopping block (the axe handle stays inside its radius)
  const cb = CABIN_PROPS.choppingBlock;
  circles.push({ x: cabinX(cb.x, cb.z), z: cabinZ(cb.x, cb.z), r: 0.3 });
  // the fire ring (the grate, the pot and the logs are inside it), the two stump seats, the firewood, the bucket
  circles.push({ x: CAMPFIRE.x, z: CAMPFIRE.z, r: FIRE_GEO.ringR });
  // (round 3: the stumps lean up to 8 cm and are out of round: their tops reach ~0.41 m from the foot's centre)
  for (const p of [FIRE_PIECES.seat1, FIRE_PIECES.seat2]) circles.push({ x: fireX(p.x, p.z), z: fireZ(p.x, p.z), r: 0.37 });
  // (round 3: the firewood is a loose spill of pieces ~1.2 m across, not a tidy stack)
  const fw = FIRE_PIECES.firewood, bk = FIRE_PIECES.bucket;
  circles.push({ x: fireX(fw.x, fw.z), z: fireZ(fw.x, fw.z), r: 0.45 });
  circles.push({ x: fireX(bk.x, bk.z), z: fireZ(bk.x, bk.z), r: 0.18 });
  // the mooring posts at the tip stand at the deck's edges (dock.glb: x -0.703 / +0.723, radius 0.13 / 0.165)
  circles.push({ x: DOCK.x - 0.703, z: DOCK.zShore - 6.291, r: 0.13 }, { x: DOCK.x + 0.723, z: DOCK.zShore - 6.339, r: 0.165 });
  return { circles, boxes };
}

/** What the camp fills in on the character lane's Ground (part 5); structural, so no import of part 5. */
export interface CampGround {
  readonly surfaces: { readonly name: string; heightAt(x: number, z: number, margin?: number): number | null }[];
  addCircle(x: number, z: number, r: number): void;
  addBox(b: BoxBlock): void;
}

/**
 * The camp's walk surfaces take the place of the character lane's stand-in deck ('dock-deck', y 0.45) and its
 * blockers join the rocks and trunks (Camp.attachGround).
 */
export function addCampToGround(ground: CampGround, surfaces: readonly CampSurface[], blockers = campBlockers()) {
  const s = ground.surfaces;
  for (let i = s.length - 1; i >= 0; i--) if (s[i].name === 'dock-deck') s.splice(i, 1);
  s.push(...surfaces);
  for (const c of blockers.circles) ground.addCircle(c.x, c.z, c.r);
  for (const b of blockers.boxes) ground.addBox(b);
}

const FC = Math.cos(CAMPFIRE.yaw), FS = Math.sin(CAMPFIRE.yaw);
export const fireX = (lx: number, lz: number) => CAMPFIRE.x + lx * FC + lz * FS;
export const fireZ = (lx: number, lz: number) => CAMPFIRE.z - lx * FS + lz * FC;

const ramp = (d: number, w: number) => Math.min(1, Math.max(0, d / w));

const _cg: CampWear = { worn: 0, ash: 0, chips: 0 };

/**
 * 0 where grass must not grow (under the cabin and its porch, the door step, the fire ring, the seats and the
 * woodpile), easing to 1 over 0.4 m; thinned to a few tufts on the camp's worn ground (groundMask.ts: the paths,
 * the fire yard, the trampled foot of the walls). Hook in part 3's Grass.ts: `p *= campGrassMask(x, z)`.
 */
export function campGrassMask(x: number, z: number): number {
  const lx = toCabinX(x, z), lz = toCabinZ(x, z);
  const g = CABIN_GEO;
  let d = Math.max(Math.abs(lx) - g.walls.hx - 0.1, Math.abs(lz) - g.walls.hz - 0.1);
  d = Math.min(d, Math.max(g.porch.x0 - 0.15 - lx, lx - g.porch.x1 - 0.15, g.porch.z0 - lz, lz - g.porch.z1 - 0.2));
  // the step stones (a second one lies in front where the porch stands high)
  d = Math.min(d, Math.max(g.step.x0 - 0.1 - lx, lx - g.step.x1 - 0.15, g.step.z0 - lz, lz - g.step.z1 - 0.7));
  d = Math.min(d, Math.max(Math.abs(lx - g.chimney.x) - g.chimney.hx - 0.3, Math.abs(lz - g.chimney.z) - g.chimney.hz - 0.1));
  // the woodpile (1.6 x 0.45 m, long side along its yaw)
  const ws = CABIN_PROPS.woodStack, wy = (ws.yaw * Math.PI) / 180;
  const ox = lx - ws.x, oz = lz - ws.z;
  const ax = Math.abs(ox * Math.cos(wy) - oz * Math.sin(wy)), az = Math.abs(ox * Math.sin(wy) + oz * Math.cos(wy));
  d = Math.min(d, Math.max(ax - 0.85, az - 0.35));
  let m = ramp(d, 0.4);
  const dx = x - CAMPFIRE.x, dz = z - CAMPFIRE.z;
  m = Math.min(m, ramp(Math.hypot(dx, dz) - FIRE_GEO.ringR - 0.25, 0.5));
  // (scene fix, critic polish r1: paths "smooth, even-width ribbons with clean edges"): the grass gives way to the
  // worn ground raggedly, tufts left standing out on it and bites trodden into the turf
  if (nearCamp(x, z)) {
    const w = Math.min(1, campGround(x, z, _cg).worn);
    const rag = fbm(x * 3.7 + 1.9, z * 3.7 - 6.1, 2) * 0.45 + fbm(x * 9.1 - 3.3, z * 9.1 + 2.8, 1) * 0.25;
    m *= 1 - 0.92 * Math.min(1, Math.max(0, w + rag * (w > 0.05 && w < 0.95 ? 1 : 0.4)));
  }
  return m;
}
