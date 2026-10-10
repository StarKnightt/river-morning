import * as THREE from 'three/webgpu';
import type { Engine } from '../core/Engine';
import type { Assets } from '../core/Assets';
import { instanceCapacity } from '../core/instancing';
import { LAYER_FOLIAGE, LAYER_MIRROR, LAYER_SHADOW } from '../core/layers';
import type { RiverWater } from '../water/RiverWater';
import { eventA, MAX_EVENTS, updateLiveEvents } from '../water/surface';
import { Habitat, type RockLike } from './FishHabitat';
import { createFishMaterial, fishU } from './FishMaterial';
import { prewarpU } from '../water/Prewarp';
import { LIMB_SLOT0, bodyA, bodyB, bodyU, setFishBodies } from '../water/BodyShadows';
import { uw } from '../water/Underwater';
import { FishSim, type BiteEvent, type Fish, type FightState, type LineInput, type LureInput, type ScriptedToken, type SplashRequest } from './FishSim';
import { SPECIES, SPECIES_IDS, type Species, type SpeciesId } from './species';

/**
 * The river's trout (part 7): two InstancedMeshes per species, the body and the fin cards, sharing every vertex
 * and per-instance attribute and the instance matrix (not one mesh with two geometry groups: the shadow pass draws
 * both groups with the same override material, three keeps one render object for both and rebuilds it between
 * them, which destroyed buffers the shadow map's command buffer still used, so the whole shadow map was dropped).
 * Swum in the vertex shader (FishMaterial.ts), driven by the CPU behaviour in FishSim.ts. Layers: foliage
 * (main view, kept out of the GTAO pre-pass: no AO halos round a trout over the bed), shadow (the sun's shadow
 * camera: they throw shadows on the bed) and mirror (a leap shows in the pool; under water the mirror's oblique
 * near plane clips them).
 *
 * URL switches: ?trout=0 (none), ?troutseed=N, ?troutsim=seconds of behaviour run before the first frame (24),
 * ?troutdbg=flat|normal|silver|nouw, ?troutlure=x,y,z (a bait hanging there, to watch inspections and takes),
 * ?troutjump=age[,id] (captures: a trout frozen `age` s into a leap, with its splash). Not `fish*`: part 6's
 * fishing owns ?fish, ?fishseed and window.__RM_FISH.
 * Console: window.__RM_TROUT (summary(), lure(x,y,z), splash(x,z,s), u: material uniforms, school).
 */
export interface FishInfo {
  id: number;
  species: SpeciesId;
  name: string;
  latin: string;
  lengthM: number;
  weightKg: number;
  state: string;
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  scale: number;
}

interface SpeciesMesh {
  sp: Species;
  body: THREE.InstancedMesh;
  fins: THREE.InstancedMesh;
  /** per instance: aSwim, aSide, aFwd, aPos (vec4 each) */
  inst: THREE.InstancedInterleavedBuffer;
  fish: Fish[];
}

declare global {
  interface Window {
    __RM_TROUT?: unknown;
  }
}

const q = new URLSearchParams(typeof location !== 'undefined' ? location.search : '');
/**
 * Stamina a hooked trout spends, times the lane's natural rate (FishSim fightDrain): with part 6's reel rules a typical
 * trout is played out in ~10-25 s instead of 40-68 s (tuned headless, see the staging notes). ?troutdrain=N overrides.
 */
export const FIGHT_DRAIN = Number(q.get('troutdrain') ?? 2.2);
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _e = new THREE.Euler(0, 0, 0, 'YXZ');
const _p = new THREE.Vector3(), _s = new THREE.Vector3(), _x = new THREE.Vector3(), _z = new THREE.Vector3();
const _box = new THREE.Box3();
/** submerged trout this frame: shadow capsule (BodyShadows.ts) and squared distance to the camera */
const _shadowed: Array<{ d: number; a: THREE.Vector4; b: THREE.Vector4 }> = [];
const _fwd = new THREE.Vector3(), _rel = new THREE.Vector3();
/** trout farther than this from the camera (m) throw no crisp shadow */
const SHADOW_RANGE = 10;
/** at most this many trout throw a crisp shadow */
const SHADOW_SLOTS = 8;

/** Float copy of a (quantized, meshopt) glTF primitive with its node transform baked in (fish.json loader notes). */
function bakeGeometry(mesh: THREE.Mesh): THREE.BufferGeometry {
  mesh.updateWorldMatrix(true, false);
  const src = mesh.geometry;
  const g = new THREE.BufferGeometry();
  for (const name of ['position', 'normal', 'uv'] as const) {
    const a = src.getAttribute(name) as THREE.BufferAttribute | undefined;
    if (!a) throw new Error(`trout mesh ${mesh.name} has no ${name}`);
    const out = new Float32Array(a.count * a.itemSize);
    for (let i = 0; i < a.count; i++) {
      out[i * a.itemSize] = a.getX(i);
      if (a.itemSize > 1) out[i * a.itemSize + 1] = a.getY(i);
      if (a.itemSize > 2) out[i * a.itemSize + 2] = a.getZ(i);
    }
    g.setAttribute(name, new THREE.BufferAttribute(out, a.itemSize));
  }
  g.setIndex(src.index ? Array.from(src.index.array as ArrayLike<number>) : null);
  g.applyMatrix4(mesh.matrixWorld);
  const map = (mesh.material as THREE.MeshStandardMaterial).map;
  const uvA = g.getAttribute('uv') as THREE.BufferAttribute;
  if (map && (map.offset.x !== 0 || map.offset.y !== 0 || map.repeat.x !== 1 || map.repeat.y !== 1 || map.rotation !== 0)) {
    map.updateMatrix();
    const v = new THREE.Vector3();
    for (let i = 0; i < uvA.count; i++) {
      v.set(uvA.getX(i), uvA.getY(i), 1).applyMatrix3(map.matrix);
      uvA.setXY(i, v.x, v.y);
    }
  }
  return g;
}

/**
 * Body and fins as two geometries over one set of vertex attributes, with `aFin`: 0 on the body and the median
 * fins, rising to 1 toward the tips of the paired (pectoral and pelvic) fins, which stick out of the median plane.
 */
function mergeTrout(body: THREE.BufferGeometry, fins: THREE.BufferGeometry): { body: THREE.BufferGeometry; fins: THREE.BufferGeometry } {
  const g = new THREE.BufferGeometry();
  const nb = body.getAttribute('position').count, nf = fins.getAttribute('position').count;
  for (const name of ['position', 'normal', 'uv']) {
    const a = body.getAttribute(name) as THREE.BufferAttribute, b = fins.getAttribute(name) as THREE.BufferAttribute;
    const out = new Float32Array((nb + nf) * a.itemSize);
    out.set(a.array as Float32Array, 0);
    out.set(b.array as Float32Array, nb * a.itemSize);
    g.setAttribute(name, new THREE.BufferAttribute(out, a.itemSize));
  }
  const fin = new Float32Array(nb + nf);
  // the body never reads aFin: its part carries the body's height in each slice along the length (-1 belly .. +1
  // back), so the species' paint (FishMaterial.ts) follows the real back and belly lines
  const bp = body.getAttribute('position');
  const NB = 48;
  let z0 = Infinity, z1 = -Infinity;
  for (let i = 0; i < nb; i++) {
    z0 = Math.min(z0, bp.getZ(i));
    z1 = Math.max(z1, bp.getZ(i));
  }
  const lo = new Float32Array(NB).fill(Infinity), hi = new Float32Array(NB).fill(-Infinity);
  const bin = (z: number) => Math.min(NB - 1, Math.max(0, Math.floor(((z - z0) / Math.max(1e-6, z1 - z0)) * NB)));
  for (let i = 0; i < nb; i++) {
    const b = bin(bp.getZ(i)), y = bp.getY(i);
    lo[b] = Math.min(lo[b], y);
    hi[b] = Math.max(hi[b], y);
  }
  for (let i = 0; i < nb; i++) {
    const b = bin(bp.getZ(i));
    fin[i] = (bp.getY(i) - (lo[b] + hi[b]) / 2) / Math.max(1e-4, (hi[b] - lo[b]) / 2);
  }
  const fp = fins.getAttribute('position');
  for (let i = 0; i < nf; i++) {
    const x = Math.abs(fp.getX(i));
    const t = Math.min(1, Math.max(0, (x - 0.004) / 0.016));
    fin[nb + i] = t * t * (3 - 2 * t);
  }
  g.setAttribute('aFin', new THREE.BufferAttribute(fin, 1));
  const ib = body.index!.array as ArrayLike<number>, iff = fins.index!.array as ArrayLike<number>;
  const gf = new THREE.BufferGeometry();
  for (const [k, a] of Object.entries(g.attributes)) gf.setAttribute(k, a);
  const U = nb + nf < 65536 ? Uint16Array : Uint32Array;
  g.setIndex(new THREE.BufferAttribute(U.from(ib), 1));
  gf.setIndex(new THREE.BufferAttribute(U.from(Array.from(iff, (i) => i + nb)), 1));
  g.computeBoundingSphere();
  gf.computeBoundingSphere();
  return { body: g, fins: gf };
}

export class FishSchool {
  readonly group = new THREE.Group();
  /** called when a trout closes its mouth on the lure: call hook(fishId) within `window` seconds */
  onBite: ((e: BiteEvent) => void) | null = null;
  /** a bite that was not hooked in time ('spat'), or a hooked trout that threw the hook on a slack line ('threw') */
  onLoose: ((fishId: number, why: 'spat' | 'threw') => void) | null = null;
  /** a trout goes for the lure (it bites a moment later) */
  onTake: ((fishId: number) => void) | null = null;
  private meshes: SpeciesMesh[] = [];
  private seenT0 = new Float64Array(MAX_EVENTS).fill(Number.NaN);
  private ownSplashes: Array<{ x: number; z: number }> = [];
  private playerFromGame = false;
  private lastDt = 0;

  private constructor(
    private engine: Engine,
    private water: RiverWater,
    readonly sim: FishSim,
    private groundY: ((x: number, z: number) => number) | null,
  ) {
    this.group.name = 'fish';
    sim.onBite = (e) => this.onBite?.(e);
    sim.onLoose = (id, why) => this.onLoose?.(id, why);
    sim.onTake = (id) => this.onTake?.(id);
  }

  /**
   * Load the four trout, place the school from the river field and add it to the scene; it updates itself from
   * an engine updater. Returns null with ?trout=0.
   */
  static async create(engine: Engine, assets: Assets, water: RiverWater, rocks: Array<{ x: number; z: number; obstacle: { r: number; topY: number } }>,
    groundY: ((x: number, z: number) => number) | null = null): Promise<FishSchool | null> {
    if (q.get('trout') === '0') return null;
    const field = water.data.field;
    const rockLike: RockLike[] = rocks.map((r) => ({ x: r.x, z: r.z, r: r.obstacle.r, topY: r.obstacle.topY }));
    const sim = new FishSim(new Habitat(field, rockLike), { seed: Number(q.get('troutseed') ?? 20261009), fightDrain: FIGHT_DRAIN });
    sim.presimulate(Number(q.get('troutsim') ?? 24));
    const school = new FishSchool(engine, water, sim, groundY);
    const debug = q.get('troutdbg');
    const underwater = debug !== 'nouw';

    const loaded = await Promise.all(SPECIES_IDS.map(async (id) => {
      const sp = SPECIES[id];
      const gltf = await assets.gltf(sp.file);
      gltf.scene.updateMatrixWorld(true);
      let body: THREE.Mesh | null = null, fins: THREE.Mesh | null = null;
      gltf.scene.traverse((o) => {
        const mm = o as THREE.Mesh;
        if (!mm.isMesh) return;
        const name = (mm.material as THREE.Material).name ?? '';
        if (name.endsWith('_Fins') || (mm.material as THREE.Material).alphaTest > 0) fins = mm;
        else body = mm;
      });
      if (!body || !fins) throw new Error(`${sp.file}: expected a body and a fin primitive`);
      const b = body as THREE.Mesh, f = fins as THREE.Mesh;
      const geo = mergeTrout(bakeGeometry(b), bakeGeometry(f));
      // a repainted skin replaces the painting where it read wrong (species.ts look.skin; same UV layout)
      const map = sp.look.skin ? await assets.texture(sp.look.skin, true, false) : (b.material as THREE.MeshStandardMaterial).map!;
      map.anisotropy = 8;
      return { sp, geo, map };
    }));

    for (const { sp, geo, map } of loaded) {
      const fish = sim.fish.filter((f) => f.sp.id === sp.id);
      if (!fish.length) continue;
      const cap = fish.length;
      // The four per-instance vec4s share one vertex buffer: with instanceCapacity() the instance matrices are vertex
      // attributes too (below 1025 three names their uniform buffer after a node id: one program per mesh instead of
      // one for every species), 6 of WebGPU's default 8 vertex buffers. Static usage although rewritten every frame:
      // three re-uploads a dynamic buffer in every pass that draws it, in full once the first pass has used the
      // update ranges; a static one uploads once per version.
      const inst = new THREE.InstancedInterleavedBuffer(new Float32Array(cap * 16), 16, 1);
      for (const g of [geo.body, geo.fins]) {
        g.setAttribute('aSwim', new THREE.InterleavedBufferAttribute(inst, 4, 0));
        g.setAttribute('aSide', new THREE.InterleavedBufferAttribute(inst, 4, 4));
        g.setAttribute('aFwd', new THREE.InterleavedBufferAttribute(inst, 4, 8));
        g.setAttribute('aPos', new THREE.InterleavedBufferAttribute(inst, 4, 12));
      }
      const mo = { map, noseZ: sp.noseZ, length: sp.modelLength, field, underwater, debug, look: sp.look };
      const sphere = new THREE.Sphere();
      const make = (g: THREE.BufferGeometry, fins: boolean) => {
        const mesh = new THREE.InstancedMesh(g, createFishMaterial({ ...mo, fins }), instanceCapacity(cap));
        mesh.name = `trout-${sp.id}-${fins ? 'fins' : 'body'}`;
        mesh.layers.set(LAYER_FOLIAGE);
        mesh.layers.enable(LAYER_SHADOW);
        mesh.layers.enable(LAYER_MIRROR);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.boundingSphere = sphere;
        school.group.add(mesh);
        return mesh;
      };
      const body = make(geo.body, false), fins = make(geo.fins, true);
      fins.instanceMatrix = body.instanceMatrix;
      school.meshes.push({ sp, body, fins, inst, fish });
    }
    engine.camera.layers.enable(LAYER_FOLIAGE);
    engine.scene.add(school.group);

    const jump = q.get('troutjump');
    if (jump) {
      const [age = 0.5, id] = jump.split(',').map(Number);
      school.pendingJump = { age, id: Number.isFinite(id) ? id : undefined };
    }
    const lure = q.get('troutlure');
    if (lure) {
      const [x, y, z] = lure.split(',').map(Number);
      sim.setLure({ x, y, z, kind: 'bobber' });
    }
    school.writeInstances();
    engine.addUpdater((dt, t) => school.update(dt, t));
    window.__RM_TROUT = {
      school,
      u: fishU,
      pw: prewarpU,
      bodies: { a: bodyA, b: bodyB, u: bodyU, uw },
      summary: () => sim.summary(),
      lure: (x: number, y: number, z: number, kind: LureInput['kind'] = 'bobber') => sim.setLure({ x, y, z, kind }),
      splash: (x: number, z: number, s = 1) => water.splash(x, z, s),
    };
    return school;
  }

  // ------------------------------------------------------------------------------------------------ API (part 6)

  /** The lure, bobber bait or fly in the water this frame (null when it is out of the water). */
  setLure(l: LureInput | null) {
    this.sim.setLure(l);
  }

  /** Send the nearest calm trout within `range` m to the lure (part 6's patience timer ran out): its id or -1. */
  attract(range: number, species?: SpeciesId): number {
    return this.sim.attract(range, species);
  }

  /** Photo-mode poses: step the school through a scripted catch, then put every trout back (FishSim.beginScripted). */
  beginScripted(seed: number): ScriptedToken {
    return this.sim.beginScripted(seed);
  }

  endScripted(t: ScriptedToken) {
    this.sim.endScripted(t);
    this.ownSplashes.length = 0;
    this.writeInstances();
  }

  /** Strike: returns true if the trout still had the lure and is now hooked. */
  hook(fishId: number): boolean {
    return this.sim.hook(fishId);
  }

  /** Every frame of the fight: rod tip and line tension (N). */
  setLine(fishId: number, line: LineInput) {
    this.sim.setLine(fishId, line);
  }

  /** Fight steering for a hooked trout: pull direction and force, stamina, phase, airborne, landable. */
  fight(fishId: number): FightState | null {
    return this.sim.fightState(fishId);
  }

  unhook(fishId: number) {
    this.sim.unhook(fishId);
  }

  /** Landed: the trout stays drawn, posed by setHeld() every frame (the angler's fish socket), until release()/keep(). */
  land(fishId: number): FishInfo | null {
    return this.sim.land(fishId) ? this.info(fishId) : null;
  }

  /**
   * `scale` multiplies the trout's own size (the brook wants ~1.3 in the angler's hold pose: species.holdScale);
   * `sag`: how far its body sags between the hands, as a fraction of its length (0 in the water).
   */
  setHeld(fishId: number, position: THREE.Vector3, quaternion: THREE.Quaternion, scale = 1, sag = 0) {
    this.sim.setHeld(fishId, position, quaternion, scale, sag);
  }

  /** The hooked trout comes up and thrashes at the surface for `seconds` (photo poses: the fight's moment). */
  thrash(fishId: number, seconds = 2) {
    this.sim.thrash(fishId, seconds);
  }

  release(fishId: number, at?: { x: number; z: number }) {
    this.sim.release(fishId, at);
  }

  keep(fishId: number) {
    this.sim.keep(fishId);
  }

  /** The player's (angler's) position; until the game calls this, a camera near the ground stands in. */
  setPlayer(p: THREE.Vector3 | null) {
    this.playerFromGame = true;
    this.sim.setPlayer(p, this.lastDt);
  }

  info(fishId: number): FishInfo | null {
    const f = this.sim.fish[fishId];
    if (!f) return null;
    return {
      id: f.id, species: f.sp.id, name: f.sp.name, latin: f.sp.latin, lengthM: f.length, weightKg: f.weight, state: f.state,
      position: f.pos.clone(), quaternion: new THREE.Quaternion().setFromEuler(_e.set(-f.pitch, f.yaw, f.roll, 'YXZ')), scale: f.scale,
    };
  }

  /** Species data (sockets, hold offset) for the angler's hold pose and the catch card. */
  species(id: SpeciesId): Species {
    return SPECIES[id];
  }

  // ------------------------------------------------------------------------------------------------ frame

  update(dt: number, t: number) {
    this.lastDt = dt;
    if (this.pendingJump) {
      this.stageJump(this.pendingJump.age, t, this.pendingJump.id);
      this.pendingJump = null;
    }
    if (dt > 0) {
      if (!this.playerFromGame) {
        const c = this.engine.camera.position;
        const ground = Math.max(this.groundY ? this.groundY(c.x, c.z) : 0, this.water.query(c.x, c.z).y);
        this.sim.setPlayer(c.y - ground < 2.6 ? c : null, dt);
      }
      this.watchSplashes(t);
      this.sim.update(dt);
      this.flushSplashes(t);
    }
    this.writeInstances();
  }

  /** New rings in the water's event buffer (lure plops, the player's splashes): the trout react to them. */
  private watchSplashes(t: number) {
    for (let i = 0; i < MAX_EVENTS; i++) {
      const e = eventA[i];
      if (e.z === this.seenT0[i] || e.z > t) continue;
      this.seenT0[i] = e.z;
      if (e.w < 0.003 || t - e.z > 0.5) continue;
      if (this.ownSplashes.some((s) => Math.abs(s.x - e.x) < 1e-3 && Math.abs(s.z - e.y) < 1e-3)) continue;
      this.sim.splash(e.x, e.y, e.w / 0.04);
    }
  }

  /** The trout's own surface events (rises, leaps, thrashing on the line) go to the water. */
  private flushSplashes(t: number) {
    const out = this.sim.outbox;
    if (!out.length) return;
    for (const s of out) {
      this.surfaceEvent(s, t, 0);
      if (s.kind === 'trail') continue;
      this.ownSplashes.push({ x: s.x, z: s.z });
      // a leap close by startles the neighbours a little
      this.sim.splash(s.x, s.z, s.strength * 0.5);
    }
    out.length = 0;
    if (this.ownSplashes.length > 32) this.ownSplashes.splice(0, this.ownSplashes.length - 32);
  }

  private seed = 1;
  private rnd = () => ((this.seed = (this.seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  /** a drop size: mostly fine, now and then a big one (log-normal round `m` metres, clamped) */
  private dropSize(m: number, lo: number, hi: number) {
    const g = Math.sqrt(-2 * Math.log(Math.max(1e-6, this.rnd()))) * Math.cos(6.2832 * this.rnd());
    return Math.min(hi, Math.max(lo, m * Math.exp(0.65 * g)));
  }

  /**
   * One surface event of a trout, `age` s old (0 live; shot-mode captures stage older ones: rings by time, drops
   * flown forward). A leap leaves a frothy broken patch with rings already spreading and a burst of drops of every
   * size; the tail of a fish in the air trails a ragged curtain of water; a slap of a thrashing fish throws white
   * water, a low fan of drops and clumps off the tail and fewer off the head, over a churned patch.
   */
  private surfaceEvent(s: SplashRequest, t: number, age: number) {
    const ev = this.water.events, t0 = t - age, L = s.size ?? 0.35;
    const surf = (x: number, z: number) => this.water.query(x, z).y;
    const drop = (x: number, y: number, z: number, vx: number, vy: number, vz: number, size: number) =>
      age > 0 ? ev.droplets.spawnAged(x, y, z, vx, vy, vz, size, age, surf) : ev.droplets.spawn(x, y, z, vx, vy, vz, size);
    this.seed = (Math.round(s.x * 977 + s.z * 613 + t0 * 131) >>> 0) || 1;
    const r = this.rnd;
    switch (s.kind) {
      case 'trail': {
        // water streaming off the tail: slower than the fish, flung out across the tail's sweep, every size (a curtain,
        // not a thread of beads)
        const n = 2 + (r() < s.strength ? 3 : 0);
        for (let k = 0; k < n; k++) {
          const f = 0.4 + 0.55 * r();
          drop(s.x + (r() - 0.5) * 0.14, s.y ?? surf(s.x, s.z), s.z + (r() - 0.5) * 0.14, (s.vx ?? 0) * f + (r() - 0.5) * 1.8,
            (s.vy ?? 0) * f + (r() - 0.5) * 0.9, (s.vz ?? 0) * f + (r() - 0.5) * 1.8, this.dropSize(0.0065, 0.0015, 0.04));
        }
        return;
      }
      case 'exit': {
        const st = s.strength;
        // the swell and the frothy broken patch where it left, a frothier core (both broken froth, not the thrash's dense
        // white: a dense core read as a cotton-wool ball), two ring trains already running out in a wide disc (from a
        // low angle stacked bright and dark ellipses that wrinkle the reflections; every live ring costs every water
        // pixel, so no third)
        ev.ring(s.x, s.z, t0, 0.08 * st, 7, 0.06 + 0.2 * L);
        ev.ring(s.x + (r() - 0.5) * 0.06, s.z + (r() - 0.5) * 0.06, t0 + 0.05, 0.016 * st, 5, 0.03 + 0.12 * L);
        ev.train(s.x, s.z, t0 + 0.02, 0.03 * st, 6, { wavelength: 0.08, speed: 0.65, r0: 0.08 });
        ev.train(s.x, s.z, t0 + 0.1, 0.026 * st, 6, { wavelength: 0.2, speed: 1.1, r0: 0.14 });
        const y = surf(s.x, s.z), hx = s.dx ?? 0, hz = s.dz ?? 0;
        for (let k = 0; k < 52; k++) {
          const a = r() * 6.2832, rr = r() * 0.08, up = 0.8 + 2.6 * r() * r(), f = 0.2 + 0.6 * r();
          drop(s.x + Math.cos(a) * rr, y + 0.01, s.z + Math.sin(a) * rr, hx * f + Math.cos(a) * 0.8 * r(), up, hz * f + Math.sin(a) * 0.8 * r(),
            this.dropSize(0.008, 0.0015, 0.045));
        }
        return;
      }
      case 'entry':
        this.water.splash(s.x, s.z, s.strength, t0);
        ev.ring(s.x, s.z, t0 + 0.04, 0.01 * s.strength, 4.5, 0.08 + 0.3 * L);
        return;
      case 'slap': {
        const st = s.strength, hx = s.dx ?? 1, hz = s.dz ?? 0, side = s.side ?? 1;
        const tx = s.x - hx * 0.45 * L, tz = s.z - hz * 0.45 * L;
        // white water over the body (a negative foam radius: dense froth) and a swell; the churned rim of the foam ring
        // reads as broken, dark water (the boil, the tail's swirl and the long trains come from the line's side,
        // FishingSim; the ring slots are full during a fight, so no more rings here)
        ev.ring(s.x - hx * 0.15 * L, s.z - hz * 0.15 * L, t0, 0.05 * st, 6, -(0.16 + 0.45 * L));
        const y = surf(s.x, s.z);
        // a burst of white water straight up off the whole body, about as wide as the fish is long
        for (let k = 0; k < 22; k++) {
          const u = (r() - 0.5) * L, a = r() * 6.2832, v = 1.4 + 2.0 * r(), sp = 0.45 * v * r();
          drop(s.x + hx * u + (r() - 0.5) * 0.06, y + 0.01, s.z + hz * u + (r() - 0.5) * 0.06, Math.cos(a) * sp, v, Math.sin(a) * sp,
            this.dropSize(0.028, 0.008, 0.075));
        }
        // the tail throws a low fan off to the side it slammed toward, and clumps of white water; the head flicks a few
        const sx = -hz * side, sz = hx * side;
        const fan = (cx: number, cz: number, ax: number, az: number, n: number, v0: number, v1: number, m: number, hi: number) => {
          for (let k = 0; k < n; k++) {
            const a = Math.atan2(az, ax) + (r() - 0.5) * 1.3, el = 0.25 + 0.75 * r(), v = v0 + (v1 - v0) * r();
            drop(cx + (r() - 0.5) * 0.08, y + 0.01, cz + (r() - 0.5) * 0.08, Math.cos(a) * Math.cos(el) * v, Math.sin(el) * v,
              Math.sin(a) * Math.cos(el) * v, this.dropSize(m, 0.002, hi));
          }
        };
        fan(tx, tz, sx - hx * 0.4, sz - hz * 0.4, Math.round(36 + 22 * st), 1.4, 3.8, 0.009, 0.042);
        fan(tx, tz, sx, sz, Math.round(8 + 6 * st), 0.6, 1.6, 0.034, 0.065);
        fan(s.x + hx * 0.4 * L, s.z + hz * 0.4 * L, -sx + hx * 0.5, -sz + hz * 0.5, Math.round(10 + 8 * st), 1.2, 2.8, 0.007, 0.03);
        return;
      }
      default:
        if (age > 0) this.water.debugSplash(s.x, s.z, age, s.strength);
        else this.water.splash(s.x, s.z, s.strength, t0);
    }
  }

  /**
   * Captures (?troutjump=age): one trout frozen `age` seconds into a leap, its splash rings and drops aged to match.
   * Staged on the first update: shot mode fixes the clock only after the world is built, and the rings are timed by
   * that clock
   */
  private pendingJump: { age: number; id?: number } | null = null;

  private stageJump(age: number, t: number, id?: number) {
    const f = this.sim.stageJump(age, id);
    if (!f) return;
    for (const s of this.sim.stagedSplashes) {
      this.surfaceEvent(s, t, s.age);
      if (s.kind !== 'trail') this.ownSplashes.push({ x: s.x, z: s.z });
    }
    updateLiveEvents(t);
    this.sim.outbox.length = 0;
  }

  private writeInstances() {
    let nb = 0;
    const cam = this.engine.camera.position;
    this.engine.camera.getWorldDirection(_fwd);
    for (const sm of this.meshes) {
      const { body: mesh, fins, inst } = sm;
      const ia = inst.array as Float32Array;
      let k = 0;
      _box.makeEmpty();
      for (const f of sm.fish) {
        if (!f.visible || f.state === 'gone') continue;
        let scale = f.scale;
        if (f.state === 'held') {
          _p.copy(f.heldPos);
          _q.copy(f.heldQuat);
          scale *= f.heldScale;
        } else {
          _p.copy(f.pos);
          _q.setFromEuler(_e.set(-f.pitch, f.yaw, f.roll, 'YXZ'));
        }
        _m.compose(_p, _q, _s.setScalar(scale));
        mesh.setMatrixAt(k, _m);
        _x.set(1, 0, 0).applyQuaternion(_q);
        _z.set(0, 0, 1).applyQuaternion(_q);
        // a submerged trout throws its crisp shadow on the bed (BodyShadows.ts): tail tip to snout
        if (f.prewarp) {
          const nose = sm.sp.noseZ * scale, tail = (sm.sp.noseZ - sm.sp.modelLength) * scale;
          const c = (_shadowed[nb] ??= { d: 0, a: new THREE.Vector4(), b: new THREE.Vector4() });
          c.b.set(_p.x + _z.x * nose, _p.y + _z.y * nose, _p.z + _z.z * nose, 0);
          c.a.set(_p.x + _z.x * tail, _p.y + _z.y * tail, _p.z + _z.z * tail, 0.095 * sm.sp.modelLength * scale);
          c.d = _rel.copy(_p).sub(cam).dot(_fwd) < -1 ? Infinity : _p.distanceToSquared(cam);
          nb++;
        }
        const o = k * 16;
        ia[o] = f.phase; ia[o + 1] = f.amp; ia[o + 2] = f.bend; ia[o + 3] = f.finPhase;
        ia[o + 4] = _x.x; ia[o + 5] = _x.y; ia[o + 6] = _x.z; ia[o + 7] = scale;
        // fin amplitude, plus 2 while the refraction pre-warp applies (off while the body breaks the surface; a held
        // trout only while the hands hold it under the water); the held trout's sag between the hands
        ia[o + 8] = _z.x; ia[o + 9] = _z.y; ia[o + 10] = _z.z; ia[o + 11] = f.finAmp + (f.prewarp ? 2 : 0);
        // (negative: the leap's arch in the air)
        ia[o + 12] = _p.x; ia[o + 13] = _p.y; ia[o + 14] = _p.z;
        ia[o + 15] = f.state === 'held' ? f.heldSag : f.state === 'jump' && !f.prewarp ? -0.95 * Math.min(0.7, Math.abs(f.bend)) : 0;
        _box.expandByPoint(_p);
        k++;
      }
      mesh.count = fins.count = k;
      if (k === 0) continue;
      // a sphere round the school of this species, padded for the fish's length, the swim and the refraction
      // pre-warp (submerged trout are drawn down toward the bed, up to the pool's depth away)
      _box.getBoundingSphere(mesh.boundingSphere!);
      mesh.boundingSphere!.radius += sm.sp.size[2] * 0.7 + 3.2;
      mesh.instanceMatrix.clearUpdateRanges();
      mesh.instanceMatrix.addUpdateRange(0, k * 16);
      mesh.instanceMatrix.needsUpdate = true;
      inst.clearUpdateRanges();
      inst.addUpdateRange(0, k * 16);
      inst.needsUpdate = true;
    }
    // the shadow slots go to the eight trout nearest the camera (of the sixteen slots: with the larger school they would
    // all fill near the bank, and every underwater pixel loops over them); past ~10 m (or behind the camera) a trout's
    // shadow is a few pixels at most
    for (let i = 1; i < nb; i++) {
      const c = _shadowed[i];
      let j = i - 1;
      for (; j >= 0 && _shadowed[j].d > c.d; j--) _shadowed[j + 1] = _shadowed[j];
      _shadowed[j + 1] = c;
    }
    let ns = 0;
    while (ns < nb && ns < Math.min(LIMB_SLOT0, SHADOW_SLOTS) && _shadowed[ns].d < SHADOW_RANGE * SHADOW_RANGE) ns++;
    nb = ns;
    for (let i = 0; i < nb; i++) {
      bodyA[i].copy(_shadowed[i].a);
      bodyB[i].copy(_shadowed[i].b);
    }
    setFishBodies(nb);
    // a trout held a few centimetres under the surface (the release): the water refracts it, and the hands under
    // it, by that depth instead of the bed's (Prewarp.ts shallow disc)
    const disc = prewarpU.disc.value;
    disc.z = 0;
    for (const f of this.sim.fish) {
      if (f.state !== 'held' || !f.prewarp) continue;
      const wy = this.water.query(f.heldPos.x, f.heldPos.z).y;
      disc.set(f.heldPos.x, f.heldPos.z, 0.2 + 0.55 * f.length, Math.min(0.22, Math.max(0.08, wy - f.heldPos.y + 0.075)));
    }
  }
}
