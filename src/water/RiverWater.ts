import * as THREE from 'three/webgpu';
import type { Engine } from '../core/Engine';
import { params } from '../core/params';
import type { SkyLighting } from '../world/SkyLighting';
import { river, newSample, POOL_LEVEL } from '../world/River';
import { buildRiverRibbon, RIVER_SECTIONS } from './RiverRibbon';
import { bakeRiverField, sampleField, type FieldSample, type Obstacle, type RiverField } from './RiverField';
import { bakeWaterDetail } from './DetailTile';
import { ringHeight, updateLiveEvents, updateSurfaceUniforms, surfaceU } from './surface';
import { setUnderwaterSun, uw } from './Underwater';
import { LAYER_NO_REFLECT, LAYER_WATER, PoolReflector } from './PoolReflector';
import { createWaterMaterial, waterU } from './WaterMaterial';
import { WaterEvents } from './WaterEvents';
import { bakeRiffle, type RiffleField } from './RiffleField';
import { riffleU } from './RiffleWater';

export interface WaterData {
  geometries: THREE.BufferGeometry[];
  field: RiverField;
  /** pool chunk bounds, for the mirror's visibility test */
  poolBox: THREE.Box3;
  /** fast-water features of the riffle and the pool head (null with ?riffle=0) */
  riffle: RiffleField | null;
}

/**
 * CPU half, run right after the terrain grids exist and before the terrain material is created (the bed
 * material needs the field for its caustics).
 */
export function prepareRiverWater(o: {
  groundY: (x: number, z: number) => number;
  core: { x0: number; z0: number; x1: number; z1: number };
  obstacles?: Obstacle[];
}): WaterData {
  const geometries = buildRiverRibbon({ sections: RIVER_SECTIONS, groundY: o.groundY });
  const near = new THREE.Box3();
  geometries.filter((g) => !g.name.startsWith('water-far')).forEach((g) => near.union(g.boundingBox!));
  const field = bakeRiverField({
    x0: Math.max(near.min.x, o.core.x0), x1: Math.min(near.max.x, o.core.x1),
    z0: Math.max(near.min.z, o.core.z0), z1: Math.min(near.max.z, o.core.z1),
    texel: 0.25, groundY: o.groundY, obstacles: o.obstacles,
  });
  const poolBox = geometries.find((g) => g.name === 'water-pool')!.boundingBox!.clone().expandByScalar(0.5);
  const riffle = params.riffle ? bakeRiffle({ field, groundY: o.groundY, obstacles: o.obstacles ?? [] }) : null;
  return { geometries, field, poolBox, riffle };
}

export interface WaterQuery {
  y: number;
  flowX: number;
  flowZ: number;
  speed: number;
  depth: number;
  foam: number;
}

/** GPU half: material, meshes, mirror, events. Needs scene.environment, i.e. after sky.load(). */
export class RiverWater {
  readonly mirror: PoolReflector | null;
  readonly events: WaterEvents;
  readonly material: THREE.MeshPhysicalNodeMaterial;
  readonly meshes: THREE.Mesh[] = [];
  private fs: FieldSample = { flowX: 0, flowZ: 0, foam: 0, waterY: 0, bedY: 0, speed: 0 };
  private rs = newSample();
  private q: WaterQuery = { y: 0, flowX: 0, flowZ: 0, speed: 0, depth: 0, foam: 0 };
  private size = new THREE.Vector2();

  constructor(private engine: Engine, sky: SkyLighting, readonly data: WaterData,
    o: { debugView?: string | null; mirror?: boolean; shadowWarp?: boolean } = {}) {
    const { scene, renderer, camera } = engine;
    setUnderwaterSun(sky.sunDir);
    if (o.shadowWarp === false) uw.warp.value = 0;
    const m = sky.meta;
    // sky fill for foam and the water body: mean of horizon and zenith at the scene's environment intensity
    waterU.skyAmbient.value.setRGB(
      (m.horizonColor[0] + m.zenithColor[0]) * 0.5 * scene.environmentIntensity,
      (m.horizonColor[1] + m.zenithColor[1]) * 0.5 * scene.environmentIntensity,
      (m.horizonColor[2] + m.zenithColor[2]) * 0.5 * scene.environmentIntensity,
    );

    for (const kv of params.wu?.split(',') ?? []) {
      const [k, v] = kv.split(':');
      const u = (waterU as Record<string, { value: unknown }>)[k] ?? (uw as Record<string, { value: unknown }>)[k]
        ?? (riffleU as Record<string, { value: unknown }>)[k];
      if (u && typeof u.value === 'number') u.value = Number(v);
    }
    this.mirror = o.mirror === false ? null : new PoolReflector(renderer, scene, sky.sun, POOL_LEVEL, data.poolBox);
    // the mirror refreshes every fourth frame (its projective lookup stays exact; the content is up to three frames
    // old) at 0.45 resolution. Part 3: with the forest in it one mirror render costs ~7 ms (C2), every other frame
    // was ~2.4 ms of the budget
    if (this.mirror) {
      this.mirror.every = params.mirrorEvery;
      this.mirror.resolutionScale = params.mirrorScale;
      this.mirror.setOutline(data.geometries);
    }
    this.material = createWaterMaterial({
      field: data.field, env: scene.environment!, detail: bakeWaterDetail(), mirror: this.mirror, sky, debugView: o.debugView,
      riffle: data.riffle,
    });
    for (const g of data.geometries) {
      const mesh = new THREE.Mesh(g, this.material);
      mesh.name = g.name;
      mesh.layers.set(LAYER_WATER);
      mesh.renderOrder = 1; // first of the transparents: splashes (3) and mist (part 3) test against its depth
      mesh.castShadow = false;
      mesh.visible = !params.noWaterMesh;
      mesh.receiveShadow = true;
      scene.add(mesh);
      this.meshes.push(mesh);
    }
    camera.layers.enable(LAYER_WATER);
    camera.layers.enable(LAYER_NO_REFLECT);

    this.events = new WaterEvents((x, z) => this.query(x, z));
    this.events.droplets.sunColor.value.copy(sky.u.sunColor.value).multiplyScalar(sky.sun.intensity * 0.08);
    this.events.droplets.skyColor.value.copy(waterU.skyAmbient.value);
    scene.add(this.events.droplets.sprite);
  }

  /** true while the loading screen is up: the mirror renders every frame even if the pool is off screen */
  warm = false;

  /** Engine updater: runs before pipeline.render(), so the mirror is ready when the scene pass draws the water. */
  update(dt: number, t: number) {
    const cam = this.engine.camera;
    this.engine.renderer.getDrawingBufferSize(this.size);
    uw.pixAngle.value = 2 / (cam.projectionMatrix.elements[5] * Math.max(1, this.size.y));
    updateSurfaceUniforms(t);
    this.events.update(dt, t);
    updateLiveEvents(t);
    // the mirror renders before the frame only behind the loading screen, the frame the pool comes into view (its
    // first image must be there) or after a jump of the camera; its regular refresh runs after the frame (postFrame)
    this.mirrorPre = false;
    const m = this.mirror;
    if (m && (this.warm || m.u.on.value === 0 || m.jumped(this.engine.camera))) {
      m.update(this.engine.camera, this.warm, this.warm);
      this.mirrorPre = m.renderedNow;
      if (m.renderedNow) this.engine.frameKind |= 1;
    }
  }

  private mirrorPre = false;

  /** Engine post-frame task: the pool mirror's regular refresh (every `mirror.every` frames) */
  postFrame() {
    const m = this.mirror;
    if (!m || this.warm || this.mirrorPre || m.u.on.value === 0) return;
    m.update(this.engine.camera);
    if (m.renderedNow) this.engine.frameKind |= 1;
  }

  /** Water at (x, z): surface height (with rings), current, depth. For the bobber, lure, fish and debris. */
  query(x: number, z: number): WaterQuery {
    const q = this.q;
    if (sampleField(this.data.field, x, z, this.fs)) {
      q.y = this.fs.waterY;
      q.flowX = this.fs.flowX;
      q.flowZ = this.fs.flowZ;
      q.speed = this.fs.speed;
      q.depth = this.fs.waterY - this.fs.bedY;
      q.foam = this.fs.foam;
    } else {
      const s = river.sample(x, z, this.rs);
      q.y = s.waterY;
      q.flowX = s.tx * s.speed;
      q.flowZ = s.tz * s.speed;
      q.speed = s.speed;
      q.depth = s.depth;
      q.foam = 0;
    }
    q.y += ringHeight(x, z, surfaceU.time.value);
    return q;
  }

  /** A splash at (x, z): expanding rings that drift with the current, a foam blob and a droplet burst. */
  splash(x: number, z: number, strength = 1, t = surfaceU.time.value) {
    this.events.splash(x, z, t, strength);
  }

  /**
   * Deterministic capture of a splash `age` seconds old (?splash=x,z,age,strength): rings by time, droplets stepped
   * (with the little rings of the drops that already fell back).
   */
  debugSplash(x: number, z: number, age: number, strength = 1) {
    const t = this.engine.time, t0 = t - age;
    surfaceU.time.value = t;
    this.events.splash(x, z, t0, strength);
    for (let s = 0; s < age; s += 1 / 120) {
      this.events.droplets.update(1 / 120, (px, pz) => this.query(px, pz).y, this.events.landed(t0 + s));
    }
    updateLiveEvents(t);
  }
}
