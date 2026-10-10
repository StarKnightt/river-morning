import * as THREE from 'three/webgpu';
import type { Engine } from '../core/Engine';
import type { Assets } from '../core/Assets';
import { params } from '../core/params';
import { grassCap } from '../core/quality';
import { buildGrid, buildMountainRing, HeightField } from './Terrain';
import { createCoreTerrainMaterial, createFarTerrainMaterial } from './TerrainMaterial';
import { SkyLighting } from './SkyLighting';
import { TimeOfDayController } from './TimeOfDay';
import { prepareRiverWater, RiverWater } from '../water/RiverWater';
import { loadRocks, placeRocks, type PlacedRock } from '../water/Rocks';
import { causticTexture } from '../water/CausticTile';
import { CanopyField, placeTrees, type TreeInst } from './ForestLayout';
import { loadBankProps, placeBankProps, type CoreSampler } from './BankProps';
import { buildFallenSpruce, spruceObstacles } from './FallenSpruce';
import { Grass } from './Grass';
import { Mist } from './Mist';
import { Forest, windU } from './Forest';
import { buildFarForest } from './FarForest';
import { meadowAt } from './Meadow';
import { campGround, nearCamp } from '../camp/groundMask';
import { LAYER_FOLIAGE, LAYER_SHADOW } from '../core/layers';
import { mergeableCaster, mergeShadowCasters } from '../gfx/shadowProxy';
import { ShadowTrigger, shadowTriggerStats } from '../gfx/shadowTrigger';
import { FishSchool } from '../fish/FishSchool';

/** The fine river-corridor grid; everything outside is the coarser hill mesh. */
export const timings: Record<string, number> = {};
const tick = (k: string, t0: number) => (timings[k] = Math.round(performance.now() - t0));

export const CORE = { x0: -120, z0: -44, x1: 110, z1: 56, step: 0.35 };

export interface World {
  sky: SkyLighting;
  timeOfDay: TimeOfDayController;
  heightField: HeightField;
  /** height of the rendered ground: the fine corridor grid inside it, the 4 m hill grid outside */
  groundY: (x: number, z: number) => number;
  water: RiverWater | null;
  rocks: PlacedRock[];
  trees: TreeInst[];
  forest: Forest | null;
  canopy: CanopyField;
  grassField: Grass | null;
  fish: FishSchool | null;
}

export async function buildWorld(engine: Engine, assets: Assets, onStep: (label: string) => void): Promise<World> {
  const T0 = performance.now();
  const { scene } = engine;
  const sky = new SkyLighting(scene);
  // part 8: owns the sky plates, the sun/moon and the fades (its updater runs before the water's)
  const timeOfDay = new TimeOfDayController(engine, sky, assets);

  onStep('Sky');
  const skyLoad = timeOfDay.load(params.tod);

  onStep('Ground textures');
  // hero bed pebbles at 2k on their own binding; the other corridor layers share two 1k arrays
  const arrayIds = [
    'dry_river_pebbles', 'gravelly_sand', 'river_small_rocks', 'brown_mud_rocks_01',
    'forrest_ground_01', 'forrest_ground_03', 'lichen_rock',
  ] as const;
  const texLoads = Promise.all([
    assets.material('ganges_river_pebbles'),
    assets.material('rocky_terrain_02'),
    assets.material('aerial_rocks_02'),
    assets.material('snow_02'),
    assets.materialArray([...arrayIds], '_1k'),
  ]);
  // the caustic net bakes on the CPU while the textures download
  if (params.water) {
    const tc = performance.now();
    causticTexture();
    tick('caustics', tc);
  }
  const [bedPebbles, grass, rock, snow, arr] = await texLoads;
  const L = (id: (typeof arrayIds)[number]) => ({ arr, layer: arrayIds.indexOf(id) });

  tick('textures', T0);
  onStep('Terrain');
  let t1 = performance.now();
  const core = buildGrid({ ...CORE, skirt: 1.5 });
  tick('coreGrid', t1);
  t1 = performance.now();
  const mid = buildGrid({
    x0: -520, z0: -520, x1: 520, z1: 520, step: 4,
    hole: { x0: CORE.x0, z0: CORE.z0, x1: CORE.x1, z1: CORE.z1, drop: 3 },
  });
  tick('midGrid', t1);
  const heightField = new HeightField(core.heights, core.nx, core.nz, CORE.x0, CORE.z0, CORE.step);
  const midField = new HeightField(mid.heights, mid.nx, mid.nz, -520, -520, 4);
  const inCore = (x: number, z: number) => x > CORE.x0 && x < CORE.x1 && z > CORE.z0 && z < CORE.z1;
  const groundY = (x: number, z: number) => (inCore(x, z) ? heightField.get(x, z) : midField.get(x, z));

  // tree positions (CPU, deterministic) before the ground material: it darkens the floor under the crowns
  t1 = performance.now();
  const trees = params.trees ? placeTrees(groundY) : [];
  const canopy = new CanopyField(trees);
  {
    // per-vertex canopy and meadow on the corridor grid (no extra sampler in the ground shader): x = crown cover,
    // y = trunk foot, z = bare patch, w = dead straw
    const pos = core.geometry.getAttribute('position') as THREE.BufferAttribute;
    const arr = new Float32Array(pos.count * 4);
    const c: [number, number] = [0, 0];
    const md = { bare: 0, straw: 0 };
    for (let i = 0; i < pos.count; i++) {
      canopy.sample(pos.getX(i), pos.getZ(i), c);
      meadowAt(pos.getX(i), pos.getZ(i), md);
      arr[i * 4] = c[0];
      arr[i * 4 + 1] = c[1];
      arr[i * 4 + 2] = md.bare;
      arr[i * 4 + 3] = md.straw;
    }
    core.geometry.setAttribute('aForest', new THREE.BufferAttribute(arr, 4));
    // part 4: the camp's worn ground (src/camp/groundMask.ts): x = packed dirt, y = ash, z = chips and bark
    const cga = new Float32Array(pos.count * 4);
    if (new URLSearchParams(location.search).get('camp') !== '0') {
      const cg = { worn: 0, ash: 0, chips: 0 };
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i), z = pos.getZ(i);
        if (!nearCamp(x, z)) continue;
        campGround(x, z, cg);
        cga[i * 4] = cg.worn;
        cga[i * 4 + 1] = cg.ash;
        cga[i * 4 + 2] = cg.chips;
      }
    }
    core.geometry.setAttribute('aCampG', new THREE.BufferAttribute(cga, 4));
  }
  tick('treePlacement', t1);

  // water: ribbon mesh and the baked flow field (it knows where the rocks will stand) before the bed material
  onStep('Water');
  t1 = performance.now();
  const rocks = params.water && params.rocks ? placeRocks(groundY) : [];
  // (the fallen spruce is a bank prop: where it lies awash it is an obstacle to the water too)
  const obstacles = [...rocks.map((r) => r.obstacle), ...(params.props ? spruceObstacles() : [])];
  const waterData = params.water ? prepareRiverWater({ groundY, core: CORE, obstacles }) : null;
  tick('water', t1);

  const coreMesh = new THREE.Mesh(core.geometry, createCoreTerrainMaterial({
    bedPebbles,
    dryPebbles: L('dry_river_pebbles'),
    sand: L('gravelly_sand'),
    silt: L('river_small_rocks'),
    muddyGravel: L('brown_mud_rocks_01'),
    grass: L('forrest_ground_01'),
    needles: L('forrest_ground_03'),
    steep: {
      arr,
      bedrock: arrayIds.indexOf('lichen_rock'),
      topsoil: arrayIds.indexOf('brown_mud_rocks_01'),
      subsoil: arrayIds.indexOf('brown_mud_rocks_01'),
    },
  }, sky.u.sunDir, CORE, params.caus ? waterData?.field : null));
  coreMesh.name = 'terrain-core';
  coreMesh.receiveShadow = true;
  coreMesh.castShadow = true;
  scene.add(coreMesh);

  const farMat = createFarTerrainMaterial({ grass, rock, snow });
  const midMesh = new THREE.Mesh(mid.geometry, farMat);
  midMesh.name = 'terrain-mid';
  midMesh.receiveShadow = true;
  midMesh.castShadow = true;
  scene.add(midMesh);

  t1 = performance.now();
  const ring = buildMountainRing(480, 16000, 1024, 200, { half: 520, drop: 6 });
  tick('mountainRing', t1);
  const ringMesh = new THREE.Mesh(ring, farMat);
  ringMesh.name = 'terrain-mountains';
  scene.add(ringMesh);
  if (params.trees) {
    // the forest carpeting the valley walls beyond the tree instances: heights from the 4 m hill grid, and from
    // the mountain ring's own (log-polar) vertices beyond it
    t1 = performance.now();
    const rp = ring.getAttribute('position').array as Float32Array;
    const R0 = 480, R1 = 16000, SEG = 1024, RINGS = 200, kr = Math.log(R1 / R0);
    const ringY = (x: number, z: number) => {
      const r = Math.hypot(x, z);
      if (r < R0 || r >= R1) return null;
      let a = Math.atan2(z, x);
      if (a < 0) a += Math.PI * 2;
      const fi = (a / (Math.PI * 2)) * SEG, fj = (Math.log(r / R0) / kr) * RINGS;
      const i = Math.min(SEG - 1, Math.floor(fi)), j = Math.min(RINGS - 1, Math.floor(fj)), ti = fi - i, tj = fj - j;
      const y = (jj: number, ii: number) => rp[(jj * (SEG + 1) + ii) * 3 + 1];
      return (y(j, i) * (1 - ti) + y(j, i + 1) * ti) * (1 - tj) + (y(j + 1, i) * (1 - ti) + y(j + 1, i + 1) * ti) * tj;
    };
    const farY = (x: number, z: number) => (Math.abs(x) < 516 && Math.abs(z) < 516 ? midField.get(x, z) : ringY(x, z));
    scene.add(buildFarForest(farY, timings));
    tick('farForest', t1);
  }

  await skyLoad;
  tick('skyReady', T0);
  sky.installFog();
  // the sun's shadow camera draws its own layer set (default + shadow-only casters) instead of the camera's
  sky.sun.shadow.camera.layers.enable(LAYER_SHADOW);
  engine.camera.layers.enable(LAYER_FOLIAGE);
  // the sun and the shadow frustum are fixed; only the slow sway of the trees moves the map, so it is redrawn every
  // few frames (scheduled after the water's updater, below)
  sky.sun.shadow.autoUpdate = false;
  sky.sun.shadow.needsUpdate = true;

  let forest: Forest | null = null;
  if (trees.length) {
    onStep('Forest');
    t1 = performance.now();
    forest = await Forest.create(assets, trees);
    if (!params.treeShadow) for (const c of forest.group.children) if (c.layers.isEnabled(LAYER_SHADOW)) c.visible = false;
    scene.add(forest.group);
    tick('forest', t1);
    const f = forest;
    // before the water's updater: the mirror renders the repacked sets
    engine.addUpdater((_dt, t) => {
      windU.time.value = t;
      f.update(engine.camera);
    });
  }

  let water: RiverWater | null = null;
  if (waterData) {
    water = new RiverWater(engine, sky, waterData, { debugView: params.wdbg, mirror: params.refl, shadowWarp: params.shwarp });
    const w = water;
    engine.addUpdater((dt, t) => w.update(dt, t));
    engine.addPostFrame(() => w.postFrame());
    const mirror = w.mirror;
    if (forest && mirror) forest.mirrorRect = (cam, out) => mirror.screenRect(cam, out);
  }
  {
    // the map is redrawn every few frames, after the frame (Engine.addPostFrame) by a render of its own; never right
    // after a mirror refresh, so the two never stack into one gap between frames (the redraw waits a frame)
    const sh = sky.sun.shadow, trigger = new ShadowTrigger(sky.sun);
    scene.add(trigger.mesh);
    let f = 0, due = false;
    engine.addUpdater(() => {
      if (++f % params.shadowEvery === 0) due = true;
    });
    engine.addPostFrame(() => {
      if (!due || engine.frameKind & 1) return;
      sh.needsUpdate = true;
      trigger.render(engine.renderer, scene);
      shadowTriggerStats.redraws++;
      if (sh.needsUpdate) shadowTriggerStats.misses++;
      due = false;
      engine.frameKind |= 2;
    });
  }
  if (rocks.length) {
    onStep('Rocks');
    t1 = performance.now();
    scene.add(await loadRocks(assets, rocks, waterData?.field ?? null, params.caus));
    tick('rocks', t1);
  }
  let fish: FishSchool | null = null;
  if (water) {
    onStep('Fish');
    t1 = performance.now();
    fish = await FishSchool.create(engine, assets, water, rocks, groundY);
    tick('fish', t1);
  }
  const wat = core.geometry.getAttribute('aWater').array as Float32Array;
  const bnk = core.geometry.getAttribute('aBank').array as Float32Array;
  const sampler: CoreSampler = {
    at(x, z, o) {
      const i = Math.round((x - CORE.x0) / CORE.step), j = Math.round((z - CORE.z0) / CORE.step);
      if (i < 0 || j < 0 || i >= core.nx || j >= core.nz) return false;
      const k = j * core.nx + i;
      o.waterY = wat[k * 4]; o.edge = wat[k * 4 + 1]; o.apron = wat[k * 4 + 3];
      o.rock = bnk[k * 4]; o.belowTop = bnk[k * 4 + 2]; o.gravel = bnk[k * 4 + 3]; o.y = core.heights[k];
      return true;
    },
    groundY,
  };
  if (params.props) {
    onStep('Bank props');
    t1 = performance.now();
    const placed = placeBankProps(sampler, canopy, trees);
    timings.propCount = placed.length;
    timings.treeCount = trees.length;
    tick('propPlacement', t1);
    const [props, spruce] = await Promise.all([
      loadBankProps(assets, placed, waterData?.field ?? null),
      params.spruce ? buildFallenSpruce(assets, groundY, params.caus ? waterData?.field ?? null : null) : null,
    ]);
    scene.add(props);
    if (spruce) {
      scene.add(spruce);
      timings.spruceTris = spruce.geometry.index!.count / 3;
    }
    tick('props', t1);
  }
  {
    // (polish round) the riffle rocks, the bank boulders and logs and the fallen spruce share shadow-only meshes, one
    // per material side, instead of ~50 casters: the shadow redraw every third frame is limited by its draw calls
    t1 = performance.now();
    const casters: THREE.Mesh[] = [];
    scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh && /^(rock-|props-|fallen-spruce)/.test(m.name) && mergeableCaster(m)) casters.push(m);
    });
    const proxies = mergeShadowCasters(casters, 'statics');
    if (proxies.length) scene.add(...proxies);
    timings.shadowMerged = casters.length;
    timings.shadowProxyTris = proxies.reduce((s, p) => s + p.geometry.index!.count / 3, 0);
    tick('shadowProxy', t1);
  }
  if (params.mist) {
    const mist = new Mist(sampler, sky, trees.length ? canopy : null, params.mu);
    engine.mistNode = (depth) => mist.node(depth, engine.camera);
    // the volume replaces most of the thin 4 m height-fog layer over the water (it would count twice)
    sky.u.mistDensity.value = 0.0002;
    engine.addUpdater((_dt, t) => mist.update(t));
  }
  let grassField: Grass | null = null;
  if (params.grass) {
    const g = (grassField = new Grass(sampler, canopy, grassCap()));
    scene.add(g.group);
    engine.addUpdater(() => g.update(engine.camera));
  }
  if (params.mirrorTest) {
    // the cube on the north bank must appear mirrored and touching at the waterline in C4; the one under
    // the surface must not appear in the mirror (only through refraction)
    const red = new THREE.MeshStandardNodeMaterial({ color: 0xff2010, roughness: 0.6 });
    const a = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.5), red);
    a.position.set(10, groundY(10, -13) + 0.25, -13);
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.5), red);
    b.position.set(12, -1.0, -5);
    scene.add(a, b);
  }
  // last, after the trout: every lit material already in the scene gets a sun-only light list (the campfire light
  // reaches only the camp's ground and what is added later: part 4's camp)
  await timeOfDay.attachWorld({ water, groundY });
  return { sky, timeOfDay, heightField, groundY, water, rocks, trees, forest, canopy, grassField, fish };
}
