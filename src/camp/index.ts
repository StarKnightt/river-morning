import * as THREE from 'three/webgpu';
import { GLTFLoader, type GLTF } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import type { Engine } from '../core/Engine';
import type { Assets } from '../core/Assets';
import { timings, type World } from '../world/World';
import { params, shotMode } from '../core/params';
import { Assembler, findNode, hash3, piecesOf, type VertexEdit } from './assemble';
import { CampMaterials } from './materials';
import { createFlames, createPuffs, createRibbon } from './effects';
import { campU, FIRE_GAIN } from './uniforms';
import { addPier, addStone, loadStoneScans, pick, STONE_TINTS, type StoneId, type StoneScans } from './stones';
import { logWalls, roofShakes, dockTimber, porchDeck, porchShelter, underDeck } from './carpentry';
import { buildChimney } from './chimney';
import { Acc, buildFirewoodHeap, buildWoodpile, scatterChips, woodAcc } from './firewood';
import { buildFire } from './fire';
import { buildInterior, type Interior } from './interior';
import { mergeableCaster, mergeShadowCasters } from '../gfx/shadowProxy';
import {
  CABIN, CABIN_GEO, CABIN_PROPS, CAMPFIRE, DOCK, DOCK_TIP_Z, FIRE_GEO, FIRE_PIECES, addCampToGround, cabinX, cabinZ, campBlockers,
  campHeights, campSurfaces, fireX, fireZ, toCabinX, toCabinZ, type BoxBlock, type CampGround, type CampSurface, type CircleBlock,
  type SurfaceKind,
} from './layout';

export { campBlockers, campGrassMask, campHeights, campSurfaces, CABIN, DOCK, CAMPFIRE } from './layout';
export type { BoxBlock, CampGround, CampSurface, CircleBlock, SurfaceKind } from './layout';
export { campU, FIRE_GAIN } from './uniforms';

/**
 * Part 4: the log cabin with chimney smoke, the campfire with its coffee pot, the short dock into the calm pool, and
 * the props someone living here leaves around (Blender lane GLBs, prep\blender; fieldstones from the project's rock
 * scans). Hooks for later parts:
 *  - walking (part 5): surfaceAt(x, z), `surfaces` (the character lane's WalkSurface shape), `blockers` (2D
 *    circles and boxes), attachGround(player.ground) for all of it;
 *  - casting (part 6): dockTip, castStand (the fishing lane's STANDS shape);
 *  - night (part 8): attachTimeOfDay(world.timeOfDay); without it setWindowGlow(0..1), fireStrength (0..1),
 *    flicker (per frame) and fireLightPosition.
 * URL switches: ?camp=0 (no camp, cost A/B), ?campfx=0 (no smoke or flames), ?campnight (window lit, fire at 1).
 */

const Q = new URLSearchParams(location.search);
export const campParams = { on: Q.get('camp') !== '0', fx: Q.get('campfx') !== '0', night: Q.has('campnight') };

export interface SurfaceHit { height: number; kind: SurfaceKind }

/**
 * What the camp uses of the time-of-day lane's TimeOfDayController (part 8). Structural, so this module builds with
 * or without part 8: `camp.attachTimeOfDay(world.timeOfDay)` once it lands.
 */
export interface CampTimeOfDay {
  registerCabin(root: THREE.Object3D, o?: { windowLight?: THREE.Vector3; windowCandela?: number }): number;
  registerCampfire(root: THREE.Object3D): number;
  readonly state: { readonly night: number };
  readonly nightLights: {
    readonly fire: THREE.PointLight;
    readonly flame: THREE.Object3D;
    /** its fire light 0.45 m over `worldPos` (and its glow sprite 0.55 m over it); `ringCenter` on the ground */
    placeFire(worldPos: THREE.Vector3, ringCenter?: THREE.Vector3): void;
    /** spheres (x, y, z, radius) that shadow the fire light */
    setFireOccluders(list: THREE.Vector4[]): void;
  } | null;
}

const FILES = ['cabin', 'dock', 'campfire', 'props'] as const;
type CampFiles = Record<(typeof FILES)[number], GLTF>;
let pending: Promise<CampFiles> | null = null;

/**
 * Start downloading the four GLBs (16.7 MB, meshopt + WebP). Optional: call it early in main() so they load while
 * the terrain builds; buildCamp() starts it otherwise. Not cached in Assets: the GLBs embed several copies of the
 * same maps, and only the copies the camp uses should stay alive.
 */
export function prefetchCamp(assets: Assets): Promise<CampFiles> | null {
  if (!campParams.on) return null;
  if (!pending) {
    const loader = new GLTFLoader(assets.manager);
    loader.setMeshoptDecoder(MeshoptDecoder);
    pending = Promise.all(FILES.map((f) => loader.loadAsync(`${assets.base}camp/${f}.glb`)))
      .then((g) => Object.fromEntries(FILES.map((f, i) => [f, g[i]])) as CampFiles);
  }
  return pending;
}

/** ?campstats: after the warm-up, log how many pipelines and shader programs exist (A/B with ?camp=0) */
function installStats(engine: Engine) {
  let f = 0;
  engine.addUpdater(() => {
    if (++f !== 45) return;
    type Pipes = { caches: Map<unknown, unknown>; programs: { vertex: Map<unknown, unknown>; fragment: Map<unknown, unknown> } };
    const p = (engine.renderer as unknown as { _pipelines?: Pipes })._pipelines;
    if (p) console.warn(`camp stats: ${p.caches.size} pipelines, ${p.programs.vertex.size} vertex + ${p.programs.fragment.size} fragment programs`);
  });
}

/**
 * ?campab: GPU cost of the camp in one session and one view: it is shown for 6 s and hidden for 6 s, over and over;
 * each phase's mean GPU frame time (the engine's rolling GPU timestamps) is logged after the first cycle.
 */
function installAB(engine: Engine, camp: Camp) {
  const T = 6000;
  let t0 = -1, phase = -1;
  const on: number[] = [], off: number[] = [];
  const last = new THREE.Vector3(1e9, 0, 0);
  engine.addUpdater(() => {
    const now = performance.now();
    const cam = engine.camera.position;
    if (cam.distanceTo(last) > 0.01) {
      // a new view: start over (shown first, so a screenshot taken right after the move includes the camp)
      last.copy(cam);
      t0 = now + 2000;
      phase = -1;
      on.length = off.length = 0;
      camp.group.visible = true;
    }
    if (now < t0) return;
    const p = Math.floor((now - t0) / T);
    if (p === phase) return;
    if (phase >= 2) (phase % 2 === 0 ? on : off).push(engine.frameStats().gpuAvg);
    phase = p;
    camp.group.visible = p % 2 === 0;
    if (on.length && off.length && on.length === off.length) {
      const m = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
      console.warn(`camp ab @${cam.x.toFixed(1)},${cam.y.toFixed(1)},${cam.z.toFixed(1)}: gpu with ${m(on).toFixed(2)} ms, without ${m(off).toFixed(2)} ms, `
        + `cost ${(m(on) - m(off)).toFixed(2)} ms over ${on.length} cycles [${on.map((v, i) => (v - off[i]).toFixed(2)).join(' ')}]`);
    }
  });
}

/** Load (or finish loading), place and add the camp. Call after buildWorld() and before the warm-up frames. */
export async function buildCamp(o: { engine: Engine; world: World; assets: Assets }): Promise<Camp | null> {
  if (Q.has('campstats')) installStats(o.engine);
  if (!campParams.on) return null;
  const t0 = performance.now();
  const [files, scans] = await Promise.all([prefetchCamp(o.assets)!, loadStoneScans(o.assets)]);
  const t1 = performance.now();
  const camp = new Camp(o.engine, o.world, files, scans);
  camp.stats.waitMs = Math.round(t1 - t0);
  camp.stats.buildMs = Math.round(performance.now() - t1);
  o.engine.scene.add(camp.group);
  o.engine.addUpdater((dt, t) => camp.update(dt, t));
  timings.campWait = camp.stats.waitMs;
  timings.campBuild = camp.stats.buildMs;
  timings.campTris = camp.stats.triangles;
  timings.campMeshes = camp.stats.meshes;
  if (params.debug || shotMode) (window as unknown as { __camp: Camp }).__camp = camp;
  if (Q.has('campab')) installAB(o.engine, camp);
  return camp;
}

/** Layered sines with incommensurate rates and a slow breath; the time-of-day lane's fireFlicker, same values. */
export function fireFlicker(t: number) {
  const s = Math.sin;
  return 0.86 + 0.07 * s(t * 7.3 + 0.4) + 0.05 * s(t * 11.9 + 1.7) + 0.035 * s(t * 19.1 + 2.9)
    + 0.025 * s(t * 31.7 + 0.8) + 0.04 * s(t * 1.13) * s(t * 2.71 + 1.3);
}

const UP = new THREE.Vector3(0, 1, 0);
const deg = (d: number) => (d * Math.PI) / 180;
const place = (x: number, y: number, z: number, yaw: number, extra?: THREE.Quaternion, scale = 1) => {
  const q = new THREE.Quaternion().setFromAxisAngle(UP, yaw);
  if (extra) q.multiply(extra);
  return new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), q, new THREE.Vector3(scale, scale, scale));
};
/** the source material of that name in a glTF scene */
function srcMat(root: THREE.Object3D, name: string): THREE.MeshStandardMaterial {
  let m: THREE.MeshStandardMaterial | null = null;
  root.traverse((o) => {
    const mm = (o as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined;
    if (!m && (o as THREE.Mesh).isMesh && mm?.name === name) m = mm;
  });
  if (!m) throw new Error(`camp: material ${name} not found`);
  return m;
}
/** cluster boxes whose x/z centres lie within `r` of each other (the stacked stones of one pier) */
function clusters(boxes: Array<{ min: THREE.Vector3; max: THREE.Vector3 }>, r = 0.2) {
  const out: Array<{ min: THREE.Vector3; max: THREE.Vector3 }> = [];
  for (const b of boxes) {
    const cx = (b.min.x + b.max.x) / 2, cz = (b.min.z + b.max.z) / 2;
    const c = out.find((o) => Math.abs((o.min.x + o.max.x) / 2 - cx) < r && Math.abs((o.min.z + o.max.z) / 2 - cz) < r);
    if (c) {
      c.min.min(b.min);
      c.max.max(b.max);
    } else out.push({ min: b.min.clone(), max: b.max.clone() });
  }
  return out;
}

/** cabin-local x of the door's middle (cabin.glb Cabin_Door: -1.172 .. -0.228): the porch's worn track */
const DOOR_X = -0.7;
const DROP = (): false => false;

/** where the grate and the pot sit relative to the set's own placement (fire-local m): all four legs inside the ring */
const GRATE_SHIFT = new THREE.Vector3(-0.12, 0, -0.05);
/** the coffee pot (campfire.glb): its axis in fire-local x/z before the shift, body radius by height above the ring's ground */
const POT = { x: 0.13, z: 0.04, y0: 0.23, top: 0.475, spoutDeg: 130 };
const potBody = (y: number) => (y < 0.36 ? 0.066 : 0.066 - 0.02 * Math.min(1, (y - 0.36) / 0.05));

/**
 * A stump (seat or chopping block) weathered out of round, leaning, its foot flaring into its roots and sitting
 * unevenly on the ground, its top sawn a little out of true. World space; `base` its foot, `height` its top.
 * `flareBy`: how far the foot spreads (the GLB's stump has ~16 vertices round its foot: round 3's 0.3 pulled one
 * of them out into a flat fin).
 */
function stumpEdit(cx: number, cz: number, base: number, seed: number, height: number, leanBy = 0.08, flareBy = 0.1): VertexEdit {
  return (p, _t, i) => {
    const dx = p[i * 3] - cx, dz = p[i * 3 + 2] - cz, y = p[i * 3 + 1] - base;
    const th = Math.atan2(dz, dx);
    const foot = Math.pow(1 - Math.min(1, Math.max(0, y / 0.17)), 2);
    const flare = 1 + flareBy * foot * (0.65 + 0.35 * Math.sin(5 * th + seed * 2));
    const f = (1 + 0.07 * Math.sin(3 * th + seed) + 0.04 * Math.sin(5 * th + 2 * seed) + 0.03 * Math.sin(11 * th + seed * 3)) * flare;
    const lean = leanBy * Math.min(1, Math.max(0, y / height));
    p[i * 3] = cx + dx * f + lean * Math.cos(seed);
    p[i * 3 + 2] = cz + dz * f + lean * Math.sin(seed);
    // one side of the foot sunk deeper into the dirt
    if (y < 0.08) p[i * 3 + 1] -= 0.035 * (0.5 + 0.5 * Math.cos(th - seed * 1.7)) * (1 - y / 0.08);
    if (y > height - 0.06) p[i * 3 + 1] += (dx * Math.cos(seed * 2) + dz * Math.sin(seed * 2)) * 0.06;
  };
}

export class Camp {
  readonly group = new THREE.Group();
  /** the cabin's meshes (contains the 'WindowGlass' material: TimeOfDayController.registerCabin works on it) */
  readonly cabinRoot = new THREE.Group();
  /** at the centre of the stone ring on the ground (TimeOfDayController.registerCampfire centres its light here) */
  readonly campfireRoot = new THREE.Object3D();
  readonly dockRoot = new THREE.Group();
  /** the door on its hinge axis; setDoorOpen() swings it inward */
  readonly door = new THREE.Group();
  /** the room inside (interior.ts; door.ts shows it while the door is open or someone is inside) */
  readonly interior: Interior;
  /** deck height and the standing spot at the dock tip (part 6) */
  readonly deckY: number;
  readonly dockTip = new THREE.Vector3();
  /** the fishing lane's STANDS entry for the dock tip (yaw 0 = facing north, across the pool) */
  readonly castStand = { x: 0, y: 0, z: 0, yawDeg: 0, pitchDeg: -11 };
  /** where part 8's campfire point light belongs: in the flames under the pot (see fireLightLift) */
  readonly fireLightPosition = new THREE.Vector3();
  /** just outside the river-facing window, a little under its middle (part 8's window light) */
  readonly windowLightPosition = new THREE.Vector3();
  /** spheres (world x, y, z, radius) that shadow part 8's fire light: ring stones, logs, stumps, firewood, bucket */
  readonly fireOccluders: THREE.Vector4[] = [];
  readonly cabinY: number;
  readonly porchY: number;
  /** top of the porch step stone, and of the lower one in front when the porch stands high (else -Infinity) */
  readonly stepY: number;
  readonly stepLowY: number;
  /** top of the flat stone at the dock's shore end */
  readonly dockStepY: number;
  readonly surfaces: CampSurface[];
  /** 2D player blockers in world x/z (the character lane's Ground.addCircle / addBox shapes) */
  readonly blockers: { circles: CircleBlock[]; boxes: BoxBlock[] } = campBlockers();
  readonly stats = { triangles: 0, meshes: 0, materials: 0, textures: 0, waitMs: 0, buildMs: 0, stones: 0, billets: 0 };
  /** the morning's small coffee fire; with part 8 attached fireStrength rises from it to 1 with the night */
  dayStrength = 0.45;
  /** false: part 8 (or anyone) sets fireStrength itself */
  autoStrength = true;
  /**
   * part 8's fire light at full strength and flicker, and how far it is raised above where that lane puts it
   * (0.45 m over the ring): 0.45 m over the ash, 14 cd burnt the ash, the logs and the ring's inner faces out to
   * white under AgX; 1.1 m up at 8 cd lit the pot from above (part 8 round 2, critic r1). Now 0.33 m up in the
   * flames beside the pot (it hangs over the centre from 0.23 m up), with NightLights' softer falloff and the
   * stones', logs' and stumps' shadows; 11 cd at part 8's night exposure 1.75 keeps the ash and the stones' inner faces
   * deep orange and the pool round the ring far brighter than the moonlit ground. Lower, at 0.18 m,
   * the ring's stones shadowed the ground for a metre all round
   */
  fireCandela = 10;
  fireLightLift = -0.12;
  private mats: CampMaterials;
  private tod: CampTimeOfDay | null = null;
  private _strength = 0.45;
  private _flicker = 1;
  private _glow = 0;
  private hit: SurfaceHit = { height: 0, kind: 'wood' };
  private doorBase = new THREE.Quaternion();

  constructor(engine: Engine, world: World, files: CampFiles, scans: StoneScans) {
    void engine;
    const groundY = world.groundY;
    // ?caus=0 switches the underwater light off on the bed and rocks; the dock posts follow
    this.mats = new CampMaterials(params.caus ? world.water?.data.field ?? null : null, world.sky);
    const asm = new Assembler(this.mats, groundY);
    for (const f of FILES) files[f].scene.updateMatrixWorld(true);
    const cabin = files.cabin.scene, fire = files.campfire.scene, props = files.props.scene;

    // ---- heights of the cabin, its porch steps and the dock (layout.ts campHeights)
    const h = campHeights(groundY);
    this.cabinY = h.cabinY;
    this.porchY = h.porchY;
    this.stepY = h.stepY;
    this.stepLowY = h.stepLowY;
    this.deckY = h.deckY;
    this.dockStepY = h.dockStepY;
    const cy = h.cabinY;
    const C = place(CABIN.x, cy, CABIN.z, CABIN.yaw);
    const st = CABIN_GEO.step;
    const sx = h.stepCentre.lx, sz = h.stepCentre.lz;
    const fg = groundY(CAMPFIRE.x, CAMPFIRE.z);
    campU.fireCenter.value.set(CAMPFIRE.x, fg, CAMPFIRE.z);
    // soot lines at world heights: the coffee pot's lower half, the chimney's top courses
    // (the pot blackened from the fire up over most of its body, crusted; the chimney's top courses with streaks)
    this.mats.calibrate('Enamel', { soot: [fg - 0.02 + POT.y0 + 0.19, fg - 0.02 + POT.y0 + 0.06, 0.96] });
    this.mats.calibrate('PierStone', { soot: [cy + 4.1, cy + 5.25, 0.95], sootStreak: 1.3 });

    // ---- cabin: log walls (every log its own), roof (every shake its own, a doubled ragged edge), windows; under
    // the porch roof less sky reaches the wall, deck and door, and the wood there stays browner and darker
    const shelter = porchShelter(cy);
    const sheltered = (p: Float32Array, t: Float32Array, i: number) => {
      const s = 1 - shelter(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
      t[i * 3] *= 1 - 0.3 * s; t[i * 3 + 1] *= 1 - 0.36 * s; t[i * 3 + 2] *= 1 - 0.44 * s;
    };
    const lowLogs: VertexEdit = (p, t, i) => {
      const y = p[i * 3 + 1] - cy;
      const k = Math.min(1, Math.max(0, (y - 0.32) / 0.5));
      t[i * 3] *= 0.8 + 0.2 * k; t[i * 3 + 1] *= 0.8 + 0.2 * k; t[i * 3 + 2] *= 0.74 + 0.26 * k;
      sheltered(p, t, i);
    };
    const frameShade: VertexEdit = (_p, t, i) => {
      t[i * 3] *= 0.55; t[i * 3 + 1] *= 0.55; t[i * 3 + 2] *= 0.55;
    };
    const logs = piecesOf(cabin, cabin, (pv, mat) => pv === 'Cabin_Walls' && mat === 'LogWood');
    const walls = logWalls(logs, cy), deck = porchDeck(cy, DOOR_X);
    asm.add(cabin, cabin, C, {
      group: 'cabin',
      filter: (pv) => pv === 'Cabin_Walls' || pv === 'Cabin_Porch' || pv === 'Cabin_Windows',
      edit: (pv, mat) => pv === 'Cabin_Walls' && (mat === 'LogWood' || mat === 'LogEnds' || mat === 'Chinking') ? lowLogs
        : pv === 'Cabin_Windows' && mat === 'Boards' ? frameShade : null,
      pieces: (pv, mat) => walls(pv, mat) ?? deck(pv, mat),
      occlude: shelter,
    });
    asm.add(cabin, cabin, C, { group: 'cabin', filter: (pv) => pv === 'Cabin_Roof', pieces: roofShakes(cy) });
    asm.add(cabin, cabin, C, { group: 'cabin', filter: (pv) => pv === 'Cabin_Roof', pieces: roofShakes(cy, true) });
    // ---- the chimney in fieldstone, sooted at the top
    buildChimney(asm, C, { stone: srcMat(cabin, 'PierStone'), mortar: srcMat(cabin, 'Chinking'), flue: srcMat(fire, 'SootedIron') });

    // ---- fieldstones: the piers under the walls and the porch beam, the porch steps
    let stones = 0;
    const pierBoxes = piecesOf(cabin, cabin, (pv) => pv === 'Cabin_Foundation');
    const porchBoxes = piecesOf(cabin, cabin, (pv) => pv === 'Cabin_PorchStones').filter((b) => !CABIN_GEO.isStepStone(b.min, b.max));
    // (the porch piers: flat stones packed tight, set back into the shade under the beam, round 2's read as cairns)
    const back: [number, number] = [cabinX(0, -1) - CABIN.x, cabinZ(0, -1) - CABIN.z];
    const piers = clusters(pierBoxes), porchPiers = clusters(porchBoxes);
    const deckShade = underDeck(cy);
    [...piers, ...porchPiers].forEach((b, k) => {
      const lx = (b.min.x + b.max.x) / 2, lz = (b.min.z + b.max.z) / 2;
      stones += addPier(asm, scans, groundY, {
        x: cabinX(lx, lz), z: cabinZ(lx, lz), top: cy + b.max.y, fx: b.max.x - b.min.x, fz: b.max.z - b.min.z, yaw: CABIN.yaw, seed: k,
        packed: k >= piers.length ? back : undefined, occlude: k >= piers.length ? deckShade : undefined,
      });
    });
    const stepAt = (lx: number, lz: number) => ({ x: cabinX(lx, lz), z: cabinZ(lx, lz) });
    const stepW = st.x1 - st.x0, stepD = st.z1 - st.z0;
    const sp = stepAt(sx, sz);
    const sg = groundY(sp.x, sp.z);
    // the steps: low, heavy boulders bedded deep in the dirt with mud trodden onto their tops (round 2: "a pale slab
    // perched on a dark rock"); round 4 wip: still a pale dome sitting on the dirt, so more of it goes under (its
    // widest girth below the ground line), darker; no broader than 1.3x the walkable top (boots beside it clip its
    // flank); scene fix (critic polish r1: "a rounded dome sitting on the dirt; you couldn't step on it"): raised and
    // cut flat well down its dome (`flatTop`), a worn slab at the walk height
    addStone(asm, scans, {
      id: 'namaqualand_boulder_05', x: sp.x + 0.02, z: sp.z, y: h.stepY, top: true, size: [stepW * 1.3, h.stepY - sg + 0.36, stepD * 1.32],
      yaw: CABIN.yaw + deg(4), tint: [0.56, 0.545, 0.52], roll: 0, mud: 1, flatTop: 0.16,
    }, 'cabin');
    stones++;
    if (h.low) {
      const lp = stepAt(h.lowAt.lx, h.lowAt.lz);
      addStone(asm, scans, {
        id: 'rock_moss_02_r07', x: lp.x, z: lp.z, y: h.stepLowY, top: true, size: [stepW * 1.32, h.stepLowY - h.lowGround + 0.2, stepD * 1.3],
        yaw: CABIN.yaw + deg(12), tint: [0.78, 0.76, 0.72], mud: 1, flatTop: 0.1,
      }, 'cabin');
      stones++;
    }

    // ---- the door keeps its hinge: baked relative to its pivot, the pivot placed in the world; an iron latch
    // and pull, and the planks round them worn dark by hands (campU.latch, the Boards material)
    const doorNode = findNode(cabin, 'Cabin_Door');
    const doorAsm = new Assembler(this.mats, () => -1e3);
    // (baked in its pivot's frame: the porch's shelter is looked up in the world)
    const doorW = new THREE.Matrix4().multiplyMatrices(C, doorNode.matrixWorld);
    const dv = new THREE.Vector3();
    const doorShelter = (x: number, y: number, z: number) => {
      dv.set(x, y, z).applyMatrix4(doorW);
      return shelter(dv.x, dv.y, dv.z);
    };
    doorAsm.add(cabin, doorNode, new THREE.Matrix4(), {
      group: 'door', filter: (pv) => pv === 'Cabin_Door', groundDirt: false, occlude: doorShelter,
      edit: (_pv, mat) => (mat === 'Boards' ? (p, t, i) => {
        const s = 1 - doorShelter(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
        t[i * 3] *= 1 - 0.3 * s; t[i * 3 + 1] *= 1 - 0.36 * s; t[i * 3 + 2] *= 1 - 0.44 * s;
      } : null),
    });
    // the latch on the planks' outer face near the free edge (door bounds in cabin-local metres: the hinge is at
    // the -x side), carried into the door pivot's frame
    const planks = piecesOf(cabin, cabin, (pv) => pv === 'Cabin_Door').filter((b) => b.max.y - b.min.y > 1.4 && b.max.x - b.min.x < 0.35);
    const dmin = new THREE.Vector3(Infinity, Infinity, Infinity), dmax = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    for (const b of planks) {
      dmin.min(b.min);
      dmax.max(b.max);
    }
    const toDoor = doorNode.matrixWorld.clone().invert();
    const iron = srcMat(fire, 'SootedIron');
    for (const [w, hh, d, ox, oy] of [[0.028, 0.17, 0.04, 0, -0.08], [0.22, 0.032, 0.02, -0.07, 0.05], [0.04, 0.06, 0.04, 0.04, 0.05]]) {
      const g = new THREE.BoxGeometry(w, hh, d).translate(dmax.x - 0.12 + ox, dmin.y + 1.0 + oy, dmax.z + d / 2).applyMatrix4(toDoor);
      doorAsm.addGeometry(g, iron, { group: 'door', groundDirt: false }, 'door-latch');
    }
    const latchAt = new THREE.Vector3(dmax.x - 0.12, dmin.y + 1.0, dmax.z).applyMatrix4(toDoor);
    new THREE.Matrix4().multiplyMatrices(C, doorNode.matrixWorld).decompose(this.door.position, this.door.quaternion, new THREE.Vector3());
    this.doorBase.copy(this.door.quaternion);
    this.door.name = 'camp-door';
    for (const m of doorAsm.build()) this.door.add(m);
    this.cabinRoot.add(this.door);
    this.door.updateMatrixWorld(true);
    const lw = latchAt.clone().applyMatrix4(this.door.matrixWorld);
    campU.latch.value.set(lw.x, lw.y, lw.z, 0.24);

    // ---- cabin props: the woodpile under the front eave, the chopping block in the yard, the lantern on the porch
    const ws = CABIN_PROPS.woodStack;
    const wsx = cabinX(ws.x, ws.z), wsz = cabinZ(ws.x, ws.z);
    const wsYaw = CABIN.yaw + deg(ws.yaw);
    let wsG = Infinity;
    for (const a of [-0.75, 0, 0.75]) wsG = Math.min(wsG, groundY(wsx + Math.cos(wsYaw) * a, wsz - Math.sin(wsYaw) * a));
    const wsM = place(wsx, wsG - 0.02, wsz, wsYaw);
    // the GLB's four stakes stay; its 101 identical billets give way to a stack built piece by piece (firewood.ts)
    asm.add(props, findNode(props, 'Prop_WoodStack'), wsM, {
      group: 'cabin', filter: (pv) => pv === 'Prop_WoodStack', pieces: (_pv, mat) => (mat === 'WeatheredWood' ? null : DROP),
    });
    const pile = woodAcc();
    this.stats.billets = buildWoodpile(pile, wsM, { x0: -0.7, x1: 0.7, height: 0.9, seed: 5 });
    pile.bark.addTo(asm, srcMat(props, 'Bark'), 'cabin', 'PileBark', 'woodpile-bark');
    pile.split.addTo(asm, srcMat(props, 'SplitWood'), 'cabin', 'PileSplit', 'woodpile-split');
    pile.ends.addTo(asm, srcMat(props, 'EndGrain'), 'cabin', 'PileEnds', 'woodpile-ends');
    const cb = CABIN_PROPS.choppingBlock;
    const cbx = cabinX(cb.x, cb.z), cbz = cabinZ(cb.x, cb.z);
    let cbG = Infinity;
    for (const [dx, dz] of [[0.2, 0], [-0.2, 0], [0, 0.2], [0, -0.2]]) cbG = Math.min(cbG, groundY(cbx + dx, cbz + dz));
    const cbM = place(cbx, cbG - 0.03, cbz, CABIN.yaw + deg(cb.yaw));
    const block = findNode(props, 'Prop_ChoppingBlock');
    const stumpRemap = (_pv: string, mat: string) => (mat === 'Bark' ? 'StumpBark' : mat === 'EndGrainBlock' || mat === 'EndGrain' ? 'StumpTop' : null);
    // (a chopping block sits square; the axe follows its slight lean)
    asm.add(props, block, cbM, { group: 'cabin', filter: (pv) => pv === 'Prop_ChoppingBlock', remap: stumpRemap, edit: () => stumpEdit(cbx, cbz, cbG - 0.03, 2.2, 0.47, 0.025, 0.07) });
    asm.add(props, block, cbM, {
      group: 'cabin', small: true, filter: (pv) => pv === 'Prop_Axe',
      edit: () => (p, _t, i) => {
        p[i * 3] += 0.025 * Math.cos(2.2);
        p[i * 3 + 2] += 0.025 * Math.sin(2.2);
      },
    });
    // chips and bark flakes where the wood is split, a few along the stack's face
    const chips = new Acc(), flakes = new Acc();
    scatterChips(chips, flakes, groundY, { x: cbx, z: cbz, r: 1.0, count: 48, seed: 31, fresh: 0.45 });
    for (let k = 0; k < 3; k++) {
      const lx = ws.x - 0.5 + k * 0.5;
      scatterChips(chips, flakes, groundY, { x: cabinX(lx, ws.z + 0.42), z: cabinZ(lx, ws.z + 0.42), r: 0.32, count: 7, seed: 41 + k });
    }
    chips.addTo(asm, srcMat(props, 'SplitWood'), 'cabin', 'PileSplit', 'chips', true, false);
    flakes.addTo(asm, srcMat(props, 'Bark'), 'cabin', 'PileBark', 'bark-flakes', true, false);
    // a hurricane lantern stands about shin-high (the model is 18 cm)
    const ln = CABIN_PROPS.lantern;
    asm.add(props, findNode(props, 'Prop_Lantern'), place(cabinX(ln.x, ln.z), this.porchY, cabinZ(ln.x, ln.z), CABIN.yaw + deg(ln.yaw), undefined, 2.1), {
      group: 'cabin', small: true, filter: (pv) => pv === 'Prop_Lantern', groundDirt: false,
    });

    // ---- dock: unrotated from its shore end; the first bearer rests on the bar
    const dockAt = { x: DOCK.x, y: this.deckY, z: DOCK.zShore };
    asm.add(files.dock.scene, files.dock.scene, place(dockAt.x, dockAt.y, dockAt.z, 0), { group: 'dock', groundDirt: false, pieces: dockTimber(dockAt) });
    // one big, pale, flat-topped rock to step up onto the deck where it starts on the bar, pressed into the earth
    const ds = h.dockStepAt;
    addStone(asm, scans, {
      id: 'namaqualand_boulder_05', x: ds.x, z: ds.z + 0.03, y: h.dockStepY, top: true, size: [1.18, h.dockStepY - h.dockStepGround + 0.16, 0.8],
      yaw: deg(7), tint: [1.42, 1.4, 1.34], flatTop: 0.06,
    }, 'dock');
    stones++;
    // gravel heaped round the shore posts where they are driven into the bar
    for (const px of [-DOCK.postX, DOCK.postX]) {
      const pz = DOCK.zShore + DOCK.postZ[0];
      for (let k = 0; k < 7; k++) {
        const a = k * 0.9 + (px > 0 ? 0.4 : 1.7), r = 0.14 + 0.1 * hash3(px, k, 1);
        const x = dockAt.x + px + Math.cos(a) * r, z = pz + Math.sin(a) * r;
        const s = 0.07 + 0.08 * hash3(px, k, 2);
        const id: StoneId = pick(['rock_09', 'stone_01', 'rock_07', 'rock_09'] as const, hash3(px, k, 3));
        addStone(asm, scans, {
          id, x, z, y: groundY(x, z) - s * 0.35, size: [s * 1.3, s * 0.7, s], yaw: hash3(px, k, 4) * 6.28,
          tint: pick(STONE_TINTS, hash3(px, k, 5)), tilt: 0.3 * hash3(px, k, 6), tiltDir: a,
        }, 'dock', true);
        stones++;
      }
    }

    // ---- campfire (fire.ts): a tight ring of sunk, sooted stones; five logs pushed in from it, burnt narrow and round
    // at their inner ends and crazed into blocks of char, two resting up on the stones; half-burnt ends and charcoal
    // in a heap of grey ash spilling out between the stones; the GLB's glowing embers lifted onto the heap
    const F = place(CAMPFIRE.x, fg - 0.02, CAMPFIRE.z, CAMPFIRE.yaw);
    const fb = buildFire(asm, scans, groundY, CAMPFIRE.x, CAMPFIRE.z, {
      bark: srcMat(fire, 'Bark'), end: srcMat(fire, 'EndGrain'), charred: srcMat(fire, 'Charred'), ash: srcMat(fire, 'Ash'),
    });
    stones += fb.stones;
    this.stats.stones = stones;
    this.fireOccluders.push(...fb.occluders);
    asm.add(fire, fire, F.clone().multiply(new THREE.Matrix4().makeTranslation(0, 0.045, 0)), {
      group: 'campfire', filter: (pv) => pv === 'Fire_Embers',
      // the embers' own glowing chunks stay; their flat charcoal disc gives way to the ash bed
      pieces: (_pv, mat) => (mat === 'Charred' ? DROP : null),
    });
    // the grate and the percolator: its spout pressed back short into the body, a glass knob on the lid
    const FG = F.clone().multiply(new THREE.Matrix4().makeTranslation(GRATE_SHIFT.x, GRATE_SHIFT.y, GRATE_SHIFT.z));
    const potX = fireX(POT.x + GRATE_SHIFT.x, POT.z + GRATE_SHIFT.z), potZ = fireZ(POT.x + GRATE_SHIFT.x, POT.z + GRATE_SHIFT.z);
    const potY0 = fg - 0.02;
    const sd = deg(POT.spoutDeg), sdx = Math.cos(sd), sdz = Math.sin(sd);
    const spoutEdit: VertexEdit = (p, _t, i) => {
      const dx = p[i * 3] - potX, dz = p[i * 3 + 2] - potZ, y = p[i * 3 + 1] - potY0;
      const r = Math.hypot(dx, dz);
      const rb = potBody(y);
      if (r < rb + 0.004 || (dx * sdx + dz * sdz) / Math.max(r, 1e-4) < 0.45) return;
      const r2 = rb + (r - rb) * 0.38;
      p[i * 3] = potX + (dx / r) * r2;
      p[i * 3 + 2] = potZ + (dz / r) * r2;
      p[i * 3 + 1] -= (r - r2) * 0.12;
    };
    asm.add(fire, fire, FG, { group: 'campfire', small: true, filter: (pv) => pv === 'CoffeePot' || pv === 'Fire_Grate', edit: (pv) => (pv === 'CoffeePot' ? spoutEdit : null) });
    {
      const knob = new THREE.SphereGeometry(0.017, 12, 6, 0, Math.PI * 2, 0, Math.PI / 2).translate(potX, potY0 + POT.top - 0.003, potZ);
      asm.addGeometry(knob, srcMat(props, 'Lantern_01_glass'), { group: 'campfire', small: true, groundDirt: false }, 'pot-knob');
    }
    const spoutTip = new THREE.Vector3(potX + sdx * (potBody(0.41) + 0.087 * 0.38), potY0 + 0.41 - 0.087 * 0.62 * 0.12, potZ + sdz * (potBody(0.41) + 0.087 * 0.38));
    // stump seats weathered grey, checked and out of round, leaning, flaring into their roots at the foot
    const stump = findNode(fire, 'Stump_Seat');
    const s0 = { x: stump.position.x, z: stump.position.z };
    const stumpTilt = stump.quaternion;
    for (const [sp2, yaw, seed] of [[s0, CAMPFIRE.yaw, 1.3], [FIRE_PIECES.seat2, CAMPFIRE.yaw + deg(FIRE_PIECES.seat2.yaw), 4.1]] as const) {
      const x = fireX(sp2.x, sp2.z), z = fireZ(sp2.x, sp2.z), g = groundY(x, z) - 0.02;
      asm.add(fire, stump, place(x, g, z, yaw, stumpTilt), {
        group: 'campfire', filter: (pv) => pv === 'Stump_Seat', remap: stumpRemap, edit: () => stumpEdit(x, z, g, seed, 0.46),
      });
      this.fireOccluders.push(new THREE.Vector4(x, g + 0.13, z, 0.23), new THREE.Vector4(x, g + 0.3, z, 0.2));
    }
    // the split firewood (round 2: "a tidy little stack of pale, near-identical pieces", a crate in C1): an untidy
    // spill of mixed pieces and sticks with chips and bark round it (firewood.ts)
    const fw = FIRE_PIECES.firewood;
    const fwx = fireX(fw.x, fw.z), fwz = fireZ(fw.x, fw.z);
    const heap = woodAcc();
    buildFirewoodHeap(heap, groundY, { x: fwx, z: fwz, yaw: CAMPFIRE.yaw + deg(fw.yaw), seed: 11, count: 15, blockers: this.blockers.circles });
    heap.bark.addTo(asm, srcMat(props, 'Bark'), 'campfire', 'PileBark', 'firewood-bark');
    heap.split.addTo(asm, srcMat(props, 'SplitWood'), 'campfire', 'PileSplit', 'firewood-split');
    heap.ends.addTo(asm, srcMat(props, 'EndGrain'), 'campfire', 'PileEnds', 'firewood-ends');
    {
      const a = CAMPFIRE.yaw + deg(fw.yaw), ca = Math.cos(a) * 0.36, sa = Math.sin(a) * 0.36;
      for (const k of [-1, 0, 1]) this.fireOccluders.push(new THREE.Vector4(fwx + ca * k, groundY(fwx + ca * k, fwz - sa * k) + 0.1, fwz - sa * k, 0.17));
    }
    const fChips = new Acc(), fFlakes = new Acc();
    scatterChips(fChips, fFlakes, groundY, { x: fwx, z: fwz, r: 0.8, count: 26, seed: 21, fresh: 0.25 });
    fChips.addTo(asm, srcMat(props, 'SplitWood'), 'campfire', 'PileSplit', 'chips', true, false);
    fFlakes.addTo(asm, srcMat(props, 'Bark'), 'campfire', 'PileBark', 'bark-flakes', true, false);
    const bk = FIRE_PIECES.bucket;
    asm.add(props, findNode(props, 'Prop_Bucket'), place(fireX(bk.x, bk.z), groundY(fireX(bk.x, bk.z), fireZ(bk.x, bk.z)) - 0.01, fireZ(bk.x, bk.z), CAMPFIRE.yaw + deg(bk.yaw)), {
      group: 'campfire', small: true, filter: (pv) => pv === 'Prop_Bucket',
    });
    {
      const bx = fireX(bk.x, bk.z), bz = fireZ(bk.x, bk.z);
      this.fireOccluders.push(new THREE.Vector4(bx, groundY(bx, bz) + 0.15, bz, 0.15));
    }

    // ---- merge
    for (const m of asm.build((g) => g === 'cabin')) this.cabinRoot.add(m);
    for (const m of asm.build((g) => g === 'dock')) this.dockRoot.add(m);
    const fireMeshes = asm.build();
    this.cabinRoot.name = 'camp-cabin';
    this.dockRoot.name = 'camp-dock';
    this.campfireRoot.name = 'camp-campfire-root';
    this.campfireRoot.position.set(CAMPFIRE.x, fg, CAMPFIRE.z);
    this.group.name = 'camp';
    this.group.add(this.cabinRoot, this.dockRoot, this.campfireRoot, ...fireMeshes);
    this.stats.triangles = Math.round(asm.triangles + doorAsm.triangles);
    // (polish round) the ~50 static casters share one shadow-only mesh per side; the hinged door keeps its own
    const casters: THREE.Mesh[] = [];
    this.group.updateMatrixWorld(true);
    this.group.traverse((o) => {
      const m = o as THREE.Mesh;
      let inDoor = false;
      for (let p: THREE.Object3D | null = m; p; p = p.parent) if (p === this.door) inDoor = true;
      if (m.isMesh && !inDoor && mergeableCaster(m)) casters.push(m);
    });
    for (const p of mergeShadowCasters(casters, 'camp', this.group)) this.group.add(p);
    // the room inside, after the casters merge (its meshes cast no sun shadows): hidden until the door opens
    this.interior = buildInterior(this.mats, C, cy, { cabin, fire, props });
    this.interior.root.visible = false;
    this.group.add(this.interior.root);
    for (const m of this.interior.backing) this.cabinRoot.add(m);

    // ---- hooks
    const fgeo = FIRE_GEO.flameBase;
    const flameBase = new THREE.Vector3(fireX(fgeo[0], fgeo[2]), fg - 0.02 + fgeo[1], fireZ(fgeo[0], fgeo[2]));
    this.fireLightPosition.copy(flameBase).add(new THREE.Vector3(0, 0.38 + this.fireLightLift, 0));
    const fwn = CABIN_GEO.frontWindow;
    this.windowLightPosition.set(cabinX(fwn[0], fwn[2] + 0.35), cy + fwn[1] - 0.25, cabinZ(fwn[0], fwn[2] + 0.35));
    campU.lamp.value.set(cabinX(fwn[0] - 0.35, fwn[2] - 1.3), cy + fwn[1] - 0.45, cabinZ(fwn[0] - 0.35, fwn[2] - 1.3));
    this.dockTip.set(DOCK.x, this.deckY, DOCK_TIP_Z + 0.4);
    Object.assign(this.castStand, { x: this.dockTip.x, y: this.deckY, z: this.dockTip.z });
    this.surfaces = campSurfaces(h);

    // ---- effects
    if (campParams.fx) {
      const sky = world.sky;
      const fl = CABIN_GEO.flue;
      const flue = new THREE.Vector3(cabinX(fl[0], fl[2]), cy + fl[1] + 0.06, cabinZ(fl[0], fl[2]));
      // the chimney on a calm morning (scene fix, critic polish r1: round 4's plume was "a dense white billow like steam
      // from a locomotive"): thin, see-through blue-grey smoke, a narrow ribbon out of the flue that rises a little,
      // bends over downwind and flattens into a drifting haze
      this.group.add(createPuffs({
        name: 'chimney-smoke', origin: flue, count: 130, life: 22, rise: 0.62, riseTau: 3.4, riseTail: 0.015, wind: 0.75, windTau: 3.2,
        size0: 0.2, grow: 0.22, opacity: 0.16, turb: 0.14, meander: 0.05, albedo: [0.68, 0.73, 0.82], albedoFar: [0.62, 0.68, 0.78], fadeIn: 0.25,
        flatten: 0.6, lumps: 0.45, thin: 0.72, dilute: 0.9,
      }, sky));
      // the campfire: a thin thread that wavers, kinks and twists, leans off downwind and frays into wisps (scene fix:
      // denser and longer, the critic could not find it)
      this.group.add(createRibbon({
        name: 'campfire-smoke', origin: new THREE.Vector3(CAMPFIRE.x, fg + 0.22, CAMPFIRE.z), strands: 2, segments: 40, life: 6.0, rise: 0.5,
        riseTau: 1.8, riseTail: 0.14, wind: 0.5, windTau: 1.3, width0: 0.06, grow: 0.11, opacity: 0.4, meander: 0.12, spread: 0.06, curl: 0.09,
        albedo: [0.66, 0.7, 0.78], fadeIn: 0.25, fireGlow: 0.7, kink: 0.12, breakup: 0.45, pattern: [0.8, 2.2],
      }, sky));
      this.group.add(createRibbon({
        name: 'coffee-steam', origin: spoutTip, strands: 1, segments: 16, life: 1.7, rise: 0.3, riseTau: 1.0, riseTail: 0.04, wind: 0.25, windTau: 0.7,
        width0: 0.03, grow: 0.13, opacity: 0.2, meander: 0.06, spread: 0.0, curl: 0.04, albedo: [0.93, 0.95, 0.97], fadeIn: 0.1,
      }, sky));
      this.group.add(createFlames(fb.flames));
    }

    this.stats.meshes = 0;
    this.group.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) this.stats.meshes++;
    });
    this.stats.materials = this.mats.materialCount;
    this.stats.textures = this.mats.textureCount;
    if (campParams.night) {
      this.setWindowGlow(1);
      this.fireStrength = 1;
    }
    this.checkClearance(world);
  }

  /** 0..1 how big the campfire burns: flames, embers and smoke (part 8 raises it at night) */
  get fireStrength() {
    return this._strength;
  }
  set fireStrength(v: number) {
    this._strength = Math.min(1, Math.max(0, v));
  }

  /** this frame's flicker (~0.65-1.07): part 8's fire light intensity = base x fireStrength x flicker */
  get flicker() {
    return this._flicker;
  }

  /** The cabin's lit window (warm lamp light behind old glass), 0 by day .. 1 at night. No recompile. */
  setWindowGlow(v: number) {
    this._glow = Math.min(1, Math.max(0, v));
    const m = this.mats.windowGlass;
    if (!m) return;
    m.emissive.setRGB(1.0, 0.58, 0.26);
    m.emissiveIntensity = 7.0 * this._glow;
  }

  get windowGlow() {
    return this._glow;
  }

  /**
   * Part 8 (the time-of-day lane). Call once right after buildCamp, before the first frame. Its controller then
   * lights the window at night (registerCabin) and centres its fire light on the ring (registerCampfire); the camp
   * keeps the only flame (the lane's flame sprite is hidden; its soft glow sprite stays). Every frame after the
   * lane's own update: fireStrength and the flame/ember gains follow its night, and its point light follows the
   * camp, intensity = fireCandela x night x fireStrength x flicker.
   */
  attachTimeOfDay(tod: CampTimeOfDay) {
    this.tod = tod;
    tod.registerCabin(this.cabinRoot, { windowLight: this.windowLightPosition, windowCandela: 1.7 });
    tod.registerCampfire(this.campfireRoot);
    const nl = tod.nightLights;
    if (!nl) return;
    nl.flame.visible = false;
    // off the pot's axis toward the yard (the pot hangs right over the fire): its belly catches the flames
    nl.placeFire(this.campfireRoot.position.clone().add(new THREE.Vector3(0.085, this.fireLightLift, 0.085)), this.campfireRoot.position);
    nl.setFireOccluders(this.fireOccluders);
  }

  /**
   * Part 5 (the character lane's Ground): the camp's walk surfaces take the stand-in deck's place ('dock-deck') and
   * its blockers join the rocks and trunks. PlayerController already pushes every step out of the blockers
   * (Ground.resolve) and checks the risers onto and off the surfaces (0.32 m up, 0.6 m down).
   */
  attachGround(ground: CampGround) {
    addCampToGround(ground, this.surfaces, this.blockers);
  }

  /** Swing the door inward about its hinge (radians, 0 = shut, ~1.7 = wide open). */
  setDoorOpen(angle: number) {
    this.door.quaternion.copy(this.doorBase).multiply(new THREE.Quaternion().setFromAxisAngle(UP, angle));
  }

  /**
   * Walkable top at (x, z) that is not terrain (the dock deck, the porch boards, the step stones), or null;
   * `margin` widens every top (foot placement over an edge). The returned object is reused by the next call.
   */
  surfaceAt(x: number, z: number, margin = 0): SurfaceHit | null {
    let best = -Infinity, kind: SurfaceKind = 'wood';
    for (const s of this.surfaces) {
      const y = s.heightAt(x, z, margin);
      if (y !== null && y > best) {
        best = y;
        kind = s.kind;
      }
    }
    if (best === -Infinity) return null;
    this.hit.height = best;
    this.hit.kind = kind;
    return this.hit;
  }

  update(_dt: number, t: number) {
    const f = fireFlicker(t);
    this._flicker = f;
    const tod = this.tod;
    if (tod) {
      const night = tod.state?.night ?? 0;
      campU.night.value = night;
      if (this.autoStrength) this.fireStrength = this.dayStrength + (1 - this.dayStrength) * night;
      campU.flameGain.value = FIRE_GAIN.flameDay + (FIRE_GAIN.flameNight - FIRE_GAIN.flameDay) * night;
      campU.emberGain.value = FIRE_GAIN.emberDay + (FIRE_GAIN.emberNight - FIRE_GAIN.emberDay) * night;
      const light = tod.nightLights?.fire;
      if (light) light.intensity = this.fireCandela * night * this._strength * f;
    }
    campU.time.value = t;
    campU.flicker.value = f;
    campU.strength.value = this._strength;
  }

  /** Part 3 places trees and bank props on its own; report anything standing inside a camp structure. */
  private checkClearance(world: World) {
    const hits: string[] = [];
    for (const t of world.trees) {
      const lx = toCabinX(t.x, t.z), lz = toCabinZ(t.x, t.z);
      if (Math.abs(lx) < 3.6 && lz > -3.0 && lz < 4.9) hits.push(`tree in cabin (${t.x.toFixed(1)}, ${t.z.toFixed(1)})`);
      if (Math.abs(t.x - DOCK.x) < 1.0 && t.z > DOCK_TIP_Z - 0.5 && t.z < DOCK.zShore + 0.8) hits.push(`tree on dock (${t.x.toFixed(1)}, ${t.z.toFixed(1)})`);
      if (Math.hypot(t.x - CAMPFIRE.x, t.z - CAMPFIRE.z) < 2.2) hits.push(`tree at campfire (${t.x.toFixed(1)}, ${t.z.toFixed(1)})`);
    }
    for (const r of world.rocks) {
      if (Math.abs(r.x - DOCK.x) < 1.2 && r.z > DOCK_TIP_Z - 0.5 && r.z < DOCK.zShore + 0.8) hits.push(`rock on dock (${r.x.toFixed(1)}, ${r.z.toFixed(1)})`);
    }
    if (hits.length) console.warn('camp: part 3 objects inside camp structures:', hits.join('; '));
  }
}
