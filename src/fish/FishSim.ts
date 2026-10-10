import * as THREE from 'three/webgpu';
import { Habitat, REACH, rng, type Station, type StationKind, type WaterAt } from './FishHabitat';
import { SPECIES, SWIM, weightKg, type Species, type SpeciesId } from './species';

/**
 * Trout behaviour on the CPU: holding in the current facing upstream, cruising and hovering in the calm pool,
 * feeding darts and rises, bolting from the player and from splashes and drifting back later, inspecting and
 * taking a lure, and the fight on the line. No rendering here (FishSchool draws it), so it also runs headless
 * under tsx (scripts/fish-sim.ts). Metres, seconds, world axes (river flows +x, the player's bank is +z).
 */
export type FishState =
  | 'hold' | 'cruise' | 'hover' | 'feed' | 'rise' | 'jump' | 'startle' | 'flee' | 'hide' | 'return'
  | 'inspect' | 'take' | 'bite' | 'refuse' | 'hooked' | 'held' | 'gone';

export interface LureInput {
  x: number;
  y: number;
  z: number;
  /** 'bobber': bait hanging under a float, 'lure': retrieved, 'fly': on the surface */
  kind?: 'bobber' | 'lure' | 'fly';
  /** how fast the lure moves (m/s); a twitch attracts, a lure ripped through the water is ignored */
  speed?: number;
}

export interface BiteEvent {
  fishId: number;
  species: SpeciesId;
  name: string;
  lengthM: number;
  weightKg: number;
  /** where the mouth closed on the lure (m) */
  x: number;
  y: number;
  z: number;
  /** seconds the fish keeps the lure in its mouth: call hook(fishId) within this window */
  window: number;
  /** 0.3 a soft pluck at a bait, 1 a hard grab */
  strength: number;
}

/** 'thrash': up on the surface, shaking its head with the back and tail out, throwing water */
export type FightPhase = 'run' | 'shake' | 'jump' | 'sulk' | 'rest' | 'tired' | 'thrash';

export interface FightState {
  fishId: number;
  species: SpeciesId;
  lengthM: number;
  weightKg: number;
  /** hook (mouth) position (m) */
  mouth: THREE.Vector3;
  velocity: THREE.Vector3;
  /** unit direction the fish swims and pulls toward */
  pull: THREE.Vector3;
  /** thrust it puts into that direction now, and its best (N) */
  pullForce: number;
  maxForce: number;
  /** 0..1 */
  stamina: number;
  phase: FightPhase;
  airborne: boolean;
  /** played out and close to the rod tip: can be landed (land(fishId)) */
  landable: boolean;
  /** surface thrashes so far (counts up; the tackle plays a sound on each new one) */
  thrashes: number;
}

export interface LineInput {
  /** rod tip (m) */
  tip: { x: number; y: number; z: number };
  /** line tension pulling the hook toward the tip (N) */
  tension: number;
}

export interface SplashRequest {
  x: number;
  z: number;
  strength: number;
  /**
   * what made it (FishSchool turns each into its own water): a rise, a leap leaving the water ('exit') or falling
   * back ('entry'), a slap of the body thrashing on top ('slap'), the water streaming off the tail of a fish in the
   * air ('trail': y, vx..vz give the tail tip and its velocity)
   */
  kind?: 'rise' | 'exit' | 'entry' | 'slap' | 'trail';
  /** the fish's length (m) */
  size?: number;
  /** the fish's heading (unit, horizontal) and the side it rolled onto (+-1) */
  dx?: number;
  dz?: number;
  side?: number;
  y?: number;
  vx?: number;
  vy?: number;
  vz?: number;
}

interface Fight {
  phase: FightPhase;
  phaseT: number;
  phaseDur: number;
  dir: THREE.Vector3;
  effort: number;
  stamina: number;
  maxForce: number;
  tension: number;
  tip: THREE.Vector3;
  lineAge: number;
  slackT: number;
  thrashT: number;
  /** thrashing on top: the phase time of the next slap (a steady beat, so a photo can catch one) */
  slapAt: number;
  pullNow: number;
  state: FightState;
}

const TAU = Math.PI * 2;
const G = 9.81;
const wrapAngle = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
const smooth = (cur: number, target: number, rate: number, dt: number) => cur + (target - cur) * (1 - Math.exp(-rate * dt));
const smoothstep = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

export class Fish {
  readonly length: number;
  readonly scale: number;
  readonly weight: number;
  readonly pos = new THREE.Vector3();
  /** ground velocity (m/s) */
  readonly vel = new THREE.Vector3();
  yaw = 0;
  pitch = 0;
  roll = 0;
  yawRate = 0;
  state: FishState = 'hold';
  stateT = 0;
  stateDur = 0;
  /** time spent close to the lure in this inspection (the look only starts once it got there) */
  lookT = 0;
  readonly tgt = new THREE.Vector3();
  tgtSpeed = 0;
  /** swim animation (FishSchool feeds these to the vertex shader) */
  phase: number;
  amp = SWIM.hold.amp;
  freq = SWIM.hold.hz;
  bend = 0;
  finPhase: number;
  finAmp = 0.5;
  visible = true;
  /** the renderer's refraction pre-warp (FishMaterial.ts) needs the whole body under the surface */
  prewarp = true;
  /** seconds since it left the surface in a leap */
  airT = 0;
  spookedUntil = -1;
  lureCooldown = 0;
  alarm = 0;
  hunger: number;
  nextFeed: number;
  /** slow sway phases while holding or hovering */
  readonly sway: [number, number, number];
  readonly refugeP = new THREE.Vector3();
  fight: Fight | null = null;
  /** transform while held by the angler (land()) */
  readonly heldPos = new THREE.Vector3();
  readonly heldQuat = new THREE.Quaternion();
  heldScale = 1;
  /** sag of the body between the hands, a fraction of its length (FishMaterial) */
  heldSag = 0;
  respawnAt = -1;
  /** sent to the lure by attract(): it takes it after the inspection */
  willTake = false;
  /** one of the larger school's extra trout (populateMore): attract() sends it only when none of the sixteen is near */
  extra = false;
  /** set when jumping or rising: the surface event is sent once */
  private flags = 0;

  constructor(readonly id: number, readonly sp: Species, public home: Station, length: number, r: () => number) {
    this.length = length;
    this.scale = length / sp.modelLength;
    this.weight = weightKg(sp, length);
    this.phase = r() * TAU;
    this.finPhase = r() * TAU;
    this.hunger = 0.4 + r() * 0.6;
    this.nextFeed = 2 + r() * 8;
    this.sway = [r() * TAU, r() * TAU, r() * TAU];
  }

  get busy() {
    return this.state === 'hooked' || this.state === 'held' || this.state === 'gone' || this.state === 'bite' || this.state === 'take'
      || this.state === 'jump' || this.state === 'rise';
  }

  get calm() {
    return this.state === 'hold' || this.state === 'cruise' || this.state === 'hover' || this.state === 'feed' || this.state === 'return';
  }

  /** world-space mouth (hook) position */
  mouth(out: THREE.Vector3, q: THREE.Quaternion) {
    const m = this.sp.mouth;
    return out.set(m[0], m[1], m[2]).multiplyScalar(this.scale).applyQuaternion(q).add(this.pos);
  }

  setFlag(bit: number, on: boolean) {
    this.flags = on ? this.flags | bit : this.flags & ~bit;
  }

  hasFlag(bit: number) {
    return (this.flags & bit) !== 0;
  }
}

const F_SURFACED = 1;
const F_ENTERED = 2;

/** thrashing on top: the first slap and the beat (s); the fight's photo is taken 0.06 s after the fourth one */
const THRASH_BEAT = { first: 0.08, period: 0.38 };
/** a fish in the air sheds a curtain of water off its tail for this long after it leaves the surface (s) */
const TRAIL_TIME = 0.3;

export interface SimOptions {
  seed?: number;
  /** multiplier on the take probability after an inspection (tests: 10 = always) */
  takeChance?: number;
  /** multiplier on the stamina a hooked trout spends (the game's fights are shorter than a real one) */
  fightDrain?: number;
}

/** What beginScripted() saved: every trout's state and the school's clocks, restored by endScripted(). */
export interface ScriptedToken {
  r: () => number;
  sim: Record<string, unknown>;
  fish: Array<Record<string, unknown>>;
}

/** One school: the distribution of the river's trout and their minds. */
export class FishSim {
  readonly fish: Fish[] = [];
  readonly outbox: SplashRequest[] = [];
  /** the two trout on the shallow margin in front of C2 (the rainbow first: the captures' leap) */
  readonly marginIds: number[] = [];
  /** called when a fish closes its mouth on the lure */
  onBite: ((e: BiteEvent) => void) | null = null;
  /** called when a bite was not hooked in time (the fish spat the lure) or a hooked fish threw the hook */
  onLoose: ((fishId: number, why: 'spat' | 'threw') => void) | null = null;
  /** called when a fish goes for the lure (it closes its mouth on it a moment later) */
  onTake: ((fishId: number) => void) | null = null;
  time = 0;
  private r: () => number;
  private w: WaterAt = { waterY: 0, bedY: 0, depth: 0, flowX: 0, flowZ: 0, speed: 0 };
  private w2: WaterAt = { waterY: 0, bedY: 0, depth: 0, flowX: 0, flowZ: 0, speed: 0 };
  private cur = { x: 0, z: 0 };
  private dir2 = { x: 0, z: 0 };
  private lure: LureInput | null = null;
  private lurePrev = new THREE.Vector3();
  private player = new THREE.Vector3(1e6, 0, 1e6);
  private playerSpeed = 0;
  private playerOn = false;
  private nextRise: number;
  private nextJump: number;
  /** calm trout within 5 m of the lure this step */
  private nearLure = 0;
  private takeChance: number;
  private fightDrain: number;
  private q = new THREE.Quaternion();
  private e = new THREE.Euler(0, 0, 0, 'YXZ');
  private v3 = new THREE.Vector3();
  private v3b = new THREE.Vector3();
  private desired = new THREE.Vector3();

  constructor(readonly habitat: Habitat, o: SimOptions = {}) {
    this.r = rng(o.seed ?? 20261009);
    this.takeChance = o.takeChance ?? 1;
    this.fightDrain = o.fightDrain ?? 1;
    this.nextRise = 4 + this.r() * 8;
    this.nextJump = 40 + this.r() * 50;
    this.populate();
  }

  // ------------------------------------------------------------------------------------------------ population

  /**
   * About 16 trout: cutthroat and brook in the riffle pockets, rainbows, a cutthroat and a brown on the seams of
   * the tongue at the pool head, browns and rainbows cruising the pool at mixed depths (two of them on the shallow
   * margin in front of the C2 close-up spot), a big brown in the tail-out.
   */
  private populate() {
    const H = this.habitat;
    // the lies are searched for more fish than these sixteen; the extras join after them (populateMore), so every one
    // of the sixteen keeps its id, size and lie
    const riffleAll = H.lies('riffle', { minDepth: 0.28, speed: [0.25, 1.5], fast: 1.1, spacing: 1.6, count: 10, seed: 11 });
    const headAll = H.lies('head', { minDepth: 0.55, speed: [0.3, 1.0], fast: 0.85, spacing: 1.6, count: 6, seed: 12 });
    const riffle = riffleAll.slice(0, 5), head = headAll.slice(0, 4);
    const tail = H.lies('tail', { minDepth: 0.4, speed: [0.2, 0.9], fast: 0.55, spacing: 2, count: 1, seed: 13 });
    const pool = H.slack({ minDepth: 0.9, maxSpeed: 0.45, spacing: 3, count: 4, seed: 14, range: 3.5 });
    // the shallow margin in front of the point bar: trout cruise it in the morning (C2 looks across it)
    const margin = [H.homeAt('pool', 3.2, 3.0, 0.4, 0.9), H.homeAt('pool', 4.7, 2.8, 0.4, 0.9)].filter((s): s is Station => !!s);
    const plan: Array<[Station, SpeciesId]> = [];
    const riffleSp: SpeciesId[] = ['cutthroat', 'brook', 'cutthroat', 'brook', 'cutthroat'];
    riffle.forEach((s, i) => plan.push([s, riffleSp[i % riffleSp.length]]));
    const headSp: SpeciesId[] = ['rainbow', 'cutthroat', 'rainbow', 'brown'];
    head.forEach((s, i) => plan.push([s, headSp[i % headSp.length]]));
    const poolSp: SpeciesId[] = ['brown', 'rainbow', 'brown', 'rainbow'];
    const poolRainbows: number[] = [];
    pool.forEach((s, i) => {
      if (i % 2 === 1) poolRainbows.push(plan.length);
      plan.push([s, poolSp[i % poolSp.length]]);
    });
    const marginSp: SpeciesId[] = ['rainbow', 'brown'];
    margin.forEach((s, i) => {
      this.marginIds.push(plan.length);
      plan.push([s, marginSp[i]]);
    });
    tail.forEach((s) => plan.push([s, 'brown']));
    plan.forEach(([st, sid], id) => {
      const sp = SPECIES[sid];
      // lengths cluster around the typical size (sum of two uniforms), browns in the pool a little bigger
      const u = (this.r() + this.r()) / 2;
      let len = u < 0.5 ? sp.size[0] + (sp.size[1] - sp.size[0]) * (u * 2) : sp.size[1] + (sp.size[2] - sp.size[1]) * ((u - 0.5) * 2);
      if (st.kind === 'pool' || st.kind === 'tail') len = Math.min(sp.size[2], len * 1.06);
      const f = new Fish(id, sp, st, len, this.r);
      this.placeAtHome(f);
      this.fish.push(f);
    });
    // Lies placed by hand once the school is built (so every trout keeps its size and the random stream its course):
    // the two margin trout hold low over the gravel at the lip of the drop-off, along the current, their shadows on
    // the stones; beyond them the pool's two rainbows (which cruised 2.5 m down, where nothing of them showed) hold
    // near the bottom a metre down, where C2 picks them out as darker torpedoes over the stones
    const lies: Array<[number | undefined, number, number]> = [
      [this.marginIds[0], 3.1, 2.45], [this.marginIds[1], 4.8, 2.45], [poolRainbows[0], 2.6, 1.35], [poolRainbows[1], 5.4, 1.5],
    ];
    for (const [id, x, z] of lies) {
      const f = id !== undefined ? this.fish[id] : undefined;
      const st = H.homeAt('pool', x, z, 0.4, 0.9, true);
      if (!f || !st) continue;
      f.home = st;
      H.at(x, z, this.w);
      f.pos.set(x, this.holdY(f, this.w), z);
      f.vel.set(0, 0, 0);
      if (Math.hypot(this.w.flowX, this.w.flowZ) > 0.02) f.yaw = Math.atan2(-this.w.flowX, -this.w.flowZ);
      f.tgt.copy(f.pos);
      f.state = 'hold';
      f.stateT = f.stateDur = 0;
    }
    this.populateMore(riffleAll.slice(5), headAll.slice(4));
  }

  /**
   * The rest of the school (about 27 more): the sixteen above held mostly where a player never looks (the riffle far
   * upstream, the middle of the pool 2.5 m down). These hold in the clear water a player on the bank and the dock
   * looks into, every one in some current so they line up heads upstream: the margin over the gravel bar near spawn,
   * the lip of the drop-off, beside the dock, the lee of the fallen spruce (suspended under the surface: the bed there
   * is 2-3 m down), the soft water at the tail of the pool, and more riffle cushions beside the boulders and seams at
   * the pool head. Mostly young fish (a river holds far more of them than big ones); no brown outgrows the two 46 cm
   * browns, so a call for the biggest brown still brings one of them.
   */
  private populateMore(riffleX: Station[], headX: Station[]) {
    const H = this.habitat;
    const taken: Station[] = this.fish.map((f) => f.home);
    const plan: Array<[Station, SpeciesId, number]> = [];
    const add = (st: Station | null, sid: SpeciesId, big: number) => {
      if (!st || taken.some((s) => Math.hypot(s.x - st.x, s.z - st.z) < 1.1)) return;
      taken.push(st);
      plan.push([st, sid, big]);
    };
    const lie = (kind: StationKind, x: number, z: number, minDepth: number, suspend?: number) => {
      const st = H.homeAt(kind, x, z, minDepth, 0.6, true);
      if (st && suspend !== undefined) st.suspend = suspend;
      return st;
    };
    // [x, z, species, size 0 young .. 1 typical]: places read off the river field (depth 0.35-1.2 m, current 0.09-0.6 m/s)
    const bar: Array<[number, number, SpeciesId, number]> = [[-2, 3.3, 'brown', 0.25], [0, 3.3, 'rainbow', 0.3], [1.5, 3.2, 'brown', 0.4], [6.3, 3.2, 'rainbow', 0.2], [8, 3.2, 'brown', 0.3]];
    const lip: Array<[number, number, SpeciesId, number]> = [[-0.5, 2.2, 'rainbow', 0.6], [7, 2.3, 'brown', 0.7], [8.8, 2.2, 'rainbow', 0.5], [14.5, 2.4, 'brown', 0.8], [16.5, 2.3, 'rainbow', 0.6]];
    const dock: Array<[number, number, SpeciesId, number]> = [[10.7, 3, 'brown', 0.3], [12.3, 2.4, 'rainbow', 0.5], [10.8, 1.6, 'brown', 0.7]];
    for (const [x, z, sid, big] of bar) add(lie('pool', x, z, 0.3), sid, big);
    for (const [x, z, sid, big] of lip) add(lie('pool', x, z, 0.5), sid, big);
    for (const [x, z, sid, big] of dock) add(lie('pool', x, z, 0.4), sid, big);
    // the spruce's lee: one low on the shallower root end, three suspended half a metre down along the trunk
    add(lie('pool', 9.56, -9.13, 0.6), 'brown', 0.8);
    add(lie('pool', 10.55, -7.42, 0.6, 0.55), 'brown', 0.9);
    add(lie('pool', 11.54, -5.72, 0.6, 0.5), 'rainbow', 0.7);
    add(lie('pool', 12.28, -4.43, 0.6, 0.45), 'brown', 0.6);
    for (const [x, z, sid, big] of [[20, 2, 'rainbow', 0.6], [23, 2.2, 'brown', 0.5], [24.5, 1.2, 'rainbow', 0.5]] as Array<[number, number, SpeciesId, number]>) {
      add(lie('tail', x, z, 0.35), sid, big);
    }
    const headSp: SpeciesId[] = ['rainbow', 'cutthroat'];
    headX.forEach((s, i) => add(s, headSp[i % 2], 0.5));
    // the riffle: the two lies the first search left over, then softer water on the boulders' cushions and seams
    const rifSp: SpeciesId[] = ['brook', 'cutthroat'];
    let nr = 0;
    for (const s of [...riffleX, ...H.lies('riffle', { minDepth: 0.22, speed: [0.15, 1.6], fast: 0.8, spacing: 1.3, count: 16, seed: 21 })]) {
      if (nr >= 5) break;
      const before = plan.length;
      add(s, rifSp[nr % 2], 0.45);
      if (plan.length > before) nr++;
    }
    for (const [st, sid, big] of plan) {
      const sp = SPECIES[sid];
      const u = (this.r() + this.r()) / 2;
      const k = clamp(big * 0.8 + (u - 0.5) * 0.6, 0, 1.15);
      let len = sp.size[0] + (sp.size[1] - sp.size[0]) * k;
      if (sid === 'brown') len = Math.min(len, 0.44);
      const f = new Fish(this.fish.length, sp, st, len, this.r);
      f.extra = true;
      this.placeAtHome(f);
      this.fish.push(f);
    }
  }

  private placeAtHome(f: Fish) {
    const H = this.habitat, w = this.w, st = f.home;
    H.at(st.x, st.z, w);
    f.pos.set(st.x, st.holds ? this.holdY(f, w) : this.cruiseY(f, w, this.r()), st.z);
    f.vel.set(0, 0, 0);
    // face into the current, or a random way in slack water
    const u = Math.hypot(w.flowX, w.flowZ);
    f.yaw = u > 0.05 ? Math.atan2(-w.flowX, -w.flowZ) : this.r() * TAU;
    f.stateT = 0;
    f.stateDur = 0;
    f.tgt.copy(f.pos);
    if (st.holds) f.state = 'hold';
    else {
      f.state = 'cruise';
      this.pickCruiseTarget(f);
    }
  }

  /** a holding trout lies a little above the bed, where the current is slower (or suspended, if its lie says so) */
  private holdY(f: Fish, w: WaterAt) {
    if (f.home.suspend !== undefined) return clamp(w.waterY - f.home.suspend, w.bedY + 0.05, w.waterY - 0.08 * f.scale - 0.04);
    return clamp(w.bedY + 0.06 * f.scale + 0.05 + Math.min(0.1, w.depth * 0.12), w.bedY + 0.05, w.waterY - 0.08 * f.scale - 0.04);
  }

  /** a cruiser keeps to its species' band of the water column */
  private cruiseY(f: Fish, w: WaterAt, u: number) {
    const [a, b] = f.sp.depthBand;
    const frac = a + (b - a) * u;
    return clamp(w.waterY - w.depth * frac, w.bedY + 0.07 * f.scale + 0.05, w.waterY - 0.07 * f.scale - 0.08);
  }

  // ------------------------------------------------------------------------------------------------ inputs

  /** The player (angler) position, and how fast they move; null: nobody near the water. */
  setPlayer(p: { x: number; y: number; z: number } | null, dt = 0) {
    if (!p) {
      this.playerOn = false;
      return;
    }
    if (this.playerOn && dt > 0) this.playerSpeed = smooth(this.playerSpeed, Math.hypot(p.x - this.player.x, p.z - this.player.z) / dt, 4, dt);
    this.player.set(p.x, p.y, p.z);
    this.playerOn = true;
  }

  setLure(l: LureInput | null) {
    if (l && this.lure) this.lurePrev.set(this.lure.x, this.lure.y, this.lure.z);
    else if (l) this.lurePrev.set(l.x, l.y, l.z);
    this.lure = l ? { ...l } : null;
    if (!l) {
      for (const f of this.fish) {
        if (f.state === 'inspect' || f.state === 'take') this.setState(f, 'refuse', 1.5);
        if (f.state === 'bite') this.loose(f, 'spat');
      }
    }
  }

  /**
   * Something hit the water at (x, z). strength 1 is a fish leaping clear, 0.3 a lure landing, 0.1 a rise.
   * Hard splashes scatter every trout within a few metres; a plop only startles the nearest and draws the others.
   */
  splash(x: number, z: number, strength: number) {
    for (const f of this.fish) {
      if (f.state === 'hooked' || f.state === 'held' || f.state === 'gone') continue;
      const d = Math.hypot(f.pos.x - x, f.pos.z - z);
      if (strength >= 0.45) {
        if (d < 2 + 5 * strength * f.sp.caution) this.spook(f, x, z, 18 + 30 * Math.min(strength, 1.5) * this.r());
      } else if (strength >= 0.08) {
        if (d < (0.9 + 1.6 * strength) * f.sp.caution) this.startle(f, x, z);
        else if (d < 5 && f.calm) f.lureCooldown = Math.max(0, f.lureCooldown - 3); // curious
      }
    }
  }

  /** Strike: set the hook in the fish that holds the lure. Returns false if that fish no longer has it. */
  hook(fishId: number): boolean {
    const f = this.fish[fishId];
    if (!f || f.state !== 'bite') return false;
    const maxForce = (2 + 12 * f.weight) * f.sp.fightPower;
    const st: FightState = {
      fishId: f.id, species: f.sp.id, lengthM: f.length, weightKg: f.weight, mouth: new THREE.Vector3(), velocity: new THREE.Vector3(),
      pull: new THREE.Vector3(1, 0, 0), pullForce: 0, maxForce, stamina: 1, phase: 'run', airborne: false, landable: false, thrashes: 0,
    };
    f.fight = {
      phase: 'run', phaseT: 0, phaseDur: 0, dir: new THREE.Vector3(1, 0, 0), effort: 1, stamina: 1, maxForce, tension: 0,
      tip: new THREE.Vector3(this.player.x, this.player.y, this.player.z), lineAge: 99, slackT: 0, thrashT: 0, slapAt: 0, pullNow: 0, state: st,
    };
    this.setState(f, 'hooked', 0);
    this.nextRun(f, true);
    // the others saw the commotion
    for (const o of this.fish) if (o !== f && (o.state === 'inspect' || o.state === 'take')) this.setState(o, 'refuse', 2);
    return true;
  }

  /** Line input for a hooked fish, every frame of the fight. */
  setLine(fishId: number, line: LineInput) {
    const fi = this.fish[fishId]?.fight;
    if (!fi) return;
    fi.tip.set(line.tip.x, line.tip.y, line.tip.z);
    fi.tension = Math.max(0, line.tension);
    fi.lineAge = 0;
  }

  fightState(fishId: number): FightState | null {
    return this.fish[fishId]?.fight?.state ?? null;
  }

  /** The line broke or the angler gave up: the fish is free and bolts. */
  unhook(fishId: number) {
    const f = this.fish[fishId];
    if (!f || f.state !== 'hooked') return;
    f.fight = null;
    this.spook(f, this.player.x, this.player.z, 30 + 30 * this.r());
  }

  /** Landed: the angler holds it now (setHeld each frame) until release() or keep(). */
  land(fishId: number): Fish | null {
    const f = this.fish[fishId];
    if (!f || (f.state !== 'hooked' && f.state !== 'bite')) return null;
    f.fight = null;
    f.heldPos.copy(f.pos);
    this.q.setFromEuler(this.e.set(-f.pitch, f.yaw, f.roll, 'YXZ'));
    f.heldQuat.copy(this.q);
    f.heldScale = 1;
    this.setState(f, 'held', 0);
    return f;
  }

  setHeld(fishId: number, pos: { x: number; y: number; z: number }, quat: { x: number; y: number; z: number; w: number }, scale = 1,
    sag = 0) {
    const f = this.fish[fishId];
    if (!f || f.state !== 'held') return;
    f.heldPos.set(pos.x, pos.y, pos.z);
    f.heldQuat.set(quat.x, quat.y, quat.z, quat.w);
    f.heldScale = scale;
    f.heldSag = sag;
    // held under the water (the release): drawn where the water's refraction looks for it, like a swimming trout
    f.prewarp = this.habitat.at(pos.x, pos.z, this.w) && this.w.depth > 0.05 && pos.y < this.w.waterY - 0.02;
  }

  /** The hooked trout comes up and thrashes at the surface for `seconds` (a headshake on top, the back out). */
  thrash(fishId: number, seconds = 2) {
    const f = this.fish[fishId];
    const fi = f?.fight;
    if (!f || !fi || f.state !== 'hooked' || fi.phase === 'jump') return;
    fi.phase = 'thrash';
    fi.phaseT = 0;
    fi.phaseDur = seconds;
    fi.effort = 1;
    fi.thrashT = 0;
    fi.slapAt = THRASH_BEAT.first;
  }

  /** Back into the water (at its held position, or at `at`): it bolts for cover and settles back later. */
  release(fishId: number, at?: { x: number; z: number }) {
    const f = this.fish[fishId];
    if (!f || f.state !== 'held') return;
    const x = at?.x ?? f.heldPos.x, z = at?.z ?? f.heldPos.z;
    this.habitat.at(x, z, this.w);
    f.pos.set(x, this.w.depth > 0 ? Math.max(this.w.bedY + 0.06, this.w.waterY - 0.12) : f.heldPos.y, z);
    f.vel.set(0, 0, 0);
    this.e.setFromQuaternion(f.heldQuat, 'YXZ');
    f.yaw = this.e.y;
    f.pitch = 0;
    f.roll = 0;
    this.setState(f, 'hover', 0);
    this.spook(f, this.player.x, this.player.z, 40 + 30 * this.r());
  }

  /** Kept (creel): gone from the river; another trout of its kind takes its lie a few minutes later. */
  keep(fishId: number) {
    const f = this.fish[fishId];
    if (!f) return;
    f.fight = null;
    f.visible = false;
    f.respawnAt = this.time + 150 + 120 * this.r();
    this.setState(f, 'gone', 0);
  }

  /**
   * Send the nearest calm trout within `range` m (horizontally) of the lure to it: it swims over, inspects and takes it.
   * The fishing loop calls this when its patience timer runs out, so a bite always comes from a trout the player can
   * see arrive. Needs setLure() first. Returns the trout's id, or -1 when none is near enough or free. With `species`
   * the biggest of that species within range comes instead (tests and photo poses ask for a big brown).
   */
  attract(range: number, species?: SpeciesId): number {
    const L = this.lure;
    if (!L) return -1;
    if (this.fish.some((o) => o.willTake || o.state === 'take' || o.state === 'bite' || o.state === 'hooked')) return -1;
    let best: Fish | null = null, bestD = range;
    // the sixteen first (the bites keep their pace and their fish), the extra trout only when none of them is in range
    for (const extra of [false, true]) {
      for (const f of this.fish) {
        if (f.extra !== extra || !f.visible || !f.calm || f.spookedUntil > this.time || (species && f.sp.id !== species)) continue;
        const d = Math.hypot(f.pos.x - L.x, f.pos.z - L.z);
        if (d > range) continue;
        if (species ? !best || f.length > best.length : d < bestD) {
          best = f;
          bestD = d;
        }
      }
      if (best) break;
    }
    if (!best) return -1;
    this.setState(best, 'inspect', 1 + this.r() * 1.5);
    best.willTake = true;
    best.lureCooldown = 25 + this.r() * 30;
    return best.id;
  }

  /**
   * Photo-mode poses step the school through a scripted catch: save every trout and the clocks, draw from a fresh
   * seeded random stream, and endScripted() puts everything back, so the frozen school the other photo spots see is
   * the same whatever pose ran before.
   */
  beginScripted(seed: number): ScriptedToken {
    const copy = (o: object, skip: Set<string>) => {
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(o)) {
        if (skip.has(k) || typeof v === 'function') continue;
        if (v instanceof THREE.Vector3 || v instanceof THREE.Quaternion) out[k] = v.clone();
        else if (Array.isArray(v)) out[k] = [...v];
        else if (v === null || typeof v !== 'object') out[k] = v;
        else if (k === 'lure') out[k] = { ...(v as object) };
      }
      return out;
    };
    const token: ScriptedToken = {
      r: this.r,
      sim: copy(this, new Set(['fish', 'outbox', 'habitat', 'w', 'w2', 'cur', 'dir2', 'q', 'e', 'v3', 'v3b', 'desired'])),
      fish: this.fish.map((f) => ({ ...copy(f, new Set(['sp', 'home'])), home: f.home, fight: f.fight })),
    };
    this.r = rng(seed);
    return token;
  }

  endScripted(t: ScriptedToken) {
    const put = (o: object, saved: Record<string, unknown>) => {
      const target = o as Record<string, unknown>;
      for (const [k, v] of Object.entries(saved)) {
        const cur = target[k];
        if (cur instanceof THREE.Vector3 || cur instanceof THREE.Quaternion) (cur as THREE.Vector3).copy(v as THREE.Vector3);
        else if (Array.isArray(cur) && Array.isArray(v)) cur.splice(0, cur.length, ...v);
        else target[k] = v;
      }
    };
    put(this, t.sim);
    this.fish.forEach((f, i) => put(f, t.fish[i]));
    this.outbox.length = 0;
    this.r = t.r;
  }

  // ------------------------------------------------------------------------------------------------ update

  update(dt: number) {
    if (dt <= 0) return;
    // long frames are split so the steering stays stable
    const steps = Math.max(1, Math.ceil(dt / (1 / 30)));
    const h = dt / steps;
    for (let s = 0; s < steps; s++) this.step(h);
  }

  /** Run the school for `seconds` at a fixed step without emitting splashes (a natural first frame). */
  presimulate(seconds: number, step = 1 / 30) {
    for (let t = 0; t < seconds; t += step) this.step(step);
    this.outbox.length = 0;
  }

  private step(dt: number) {
    this.time += dt;
    const t = this.time;
    this.scheduleSurfaceLife(t);
    const L = this.lure;
    this.nearLure = 0;
    if (L) for (const f of this.fish) if (f.calm && Math.hypot(f.pos.x - L.x, f.pos.z - L.z) < 5) this.nearLure++;
    for (const f of this.fish) {
      f.stateT += dt;
      f.lureCooldown = Math.max(0, f.lureCooldown - dt);
      this.perceive(f, dt);
      this.think(f, dt);
      this.move(f, dt);
      this.animate(f, dt);
    }
    if (this.lure) this.lurePrev.set(this.lure.x, this.lure.y, this.lure.z);
  }

  /** a rise every 5-14 s, most of them within sight of the player; a fish leaping clear every minute or two */
  private scheduleSurfaceLife(t: number) {
    if (t >= this.nextRise) {
      this.nextRise = t + 5 + this.r() * 9;
      const c = this.fish.filter((f) => f.calm && f.spookedUntil < t && this.habitat.depthAt(f.pos.x, f.pos.z) < 1.7);
      const seen = this.playerOn ? c.filter((f) => Math.hypot(f.pos.x - this.player.x, f.pos.z - this.player.z) < 16) : [];
      const from = seen.length && this.r() < 0.75 ? seen : c;
      if (from.length) this.startRise(from[Math.floor(this.r() * from.length)]);
    }
    if (t >= this.nextJump) {
      this.nextJump = t + 55 + this.r() * 75;
      const c = this.fish.filter((f) => f.calm && f.spookedUntil < t && f.home.kind !== 'riffle' && this.habitat.depthAt(f.pos.x, f.pos.z) > 0.5);
      let total = 0;
      for (const f of c) total += f.sp.jumpiness;
      let pick = this.r() * total;
      for (const f of c) {
        pick -= f.sp.jumpiness;
        if (pick <= 0) { this.startJump(f, 2.6 + this.r() * 1.0); break; }
      }
    }
  }

  // ------------------------------------------------------------------------------------------------ perception

  private perceive(f: Fish, dt: number) {
    if (f.busy || f.state === 'flee' || f.state === 'hide') {
      f.alarm = 0;
      return;
    }
    // the player: close to the edge, tall above the water, or walking fast toward it
    if (this.playerOn) {
      const H = this.habitat;
      const d = Math.hypot(f.pos.x - this.player.x, f.pos.z - this.player.z);
      H.at(f.pos.x, f.pos.z, this.w);
      const tall = clamp((this.player.y - this.w.waterY) / 1.7, 0, 1.4);
      const near = smoothstep(4.2, 1.6, d) * (0.45 + 0.55 * tall);
      const fast = smoothstep(7.5, 3.0, d) * clamp((this.playerSpeed - 0.6) / 1.4, 0, 1);
      f.alarm += (near * 1.1 + fast * 2.2) * f.sp.caution * dt;
    }
    f.alarm = Math.max(0, f.alarm - 0.35 * dt);
    if (f.alarm > 1) {
      f.alarm = 0;
      this.spook(f, this.player.x, this.player.z, 25 + 35 * this.r());
      // a bolting trout spooks its neighbours
      for (const o of this.fish) {
        if (o !== f && o.calm && o.pos.distanceTo(f.pos) < 2.6) o.alarm += 0.7;
      }
      return;
    }
    // a lure in range
    const L = this.lure;
    if (L && f.calm && f.lureCooldown <= 0 && f.spookedUntil < this.time) {
      const dx = L.x - f.pos.x, dy = L.y - f.pos.y, dz = L.z - f.pos.z;
      const d = Math.hypot(dx, dy, dz);
      const moving = clamp((L.speed ?? 0) / 0.4, 0, 1);
      const range = (L.kind === 'lure' ? 5 : L.kind === 'fly' ? 4.2 : 3.4) + moving;
      if (d < range) {
        const front = (Math.sin(f.yaw) * dx + Math.cos(f.yaw) * dz) / Math.max(d, 1e-3);
        const tooFast = (L.speed ?? 0) > 1.4;
        const busyOthers = this.fish.filter((o) => o.state === 'inspect' || o.state === 'take' || o.state === 'bite' || o.state === 'hooked').length;
        const p = f.sp.boldness * f.hunger * (0.5 + moving) * (front > -0.2 || d < 1.4 ? 1 : 0.2) * (tooFast ? 0 : 1) * (1 - d / range) * 0.9;
        // however many trout lie round the lure, together they come to look no more often than two did: the larger
        // school keeps the pace of the bites
        const crowd = Math.max(1, this.nearLure / 2);
        if (busyOthers < 2 && this.r() < (p / crowd) * dt) {
          this.setState(f, 'inspect', 1.5 + this.r() * 3.5);
          f.lureCooldown = 25 + this.r() * 30;
        }
      }
    }
  }

  private spook(f: Fish, x: number, z: number, seconds: number) {
    // on the line the fight steers it
    if (f.fight || f.state === 'held' || f.state === 'gone') return;
    if (f.state === 'bite') {
      this.loose(f, 'spat');
      return;
    }
    f.spookedUntil = Math.max(f.spookedUntil, this.time + seconds);
    if ((f.state === 'jump' || f.state === 'hooked') && f.hasFlag(F_SURFACED) && !f.hasFlag(F_ENTERED)) {
      // in the air (a leap, or off the line mid-leap): it bolts once it is back in the water
      if (f.state === 'hooked') this.setState(f, 'jump', 3);
      return;
    }
    let ax = f.pos.x - x, az = f.pos.z - z;
    const l = Math.hypot(ax, az) || 1;
    ax /= l;
    az /= l;
    const deep = f.home.kind === 'riffle' ? 0.45 : 1.4;
    const r = this.habitat.refuge(f.pos.x, f.pos.z, ax, az, f.home.kind === 'riffle' ? 5 : 9, deep, this.dir2);
    this.habitat.at(r.x, r.z, this.w);
    f.refugeP.set(r.x, this.w.depth > 0 ? Math.max(this.w.bedY + 0.08 * f.scale + 0.06, this.w.waterY - this.w.depth * 0.8) : f.pos.y, r.z);
    // the first dash goes straight away from the threat, then it swims to cover
    f.tgt.set(f.pos.x + ax * 1.6, f.pos.y - 0.1, f.pos.z + az * 1.6);
    f.tgtSpeed = 1.7 + this.r() * 1.0;
    this.setState(f, 'flee', 0.45 + this.r() * 0.45);
  }

  private startle(f: Fish, x: number, z: number) {
    if (!f.calm && f.state !== 'inspect') return;
    let ax = f.pos.x - x, az = f.pos.z - z;
    const l = Math.hypot(ax, az) || 1;
    ax /= l;
    az /= l;
    f.tgt.set(f.pos.x + ax * 0.7, f.pos.y, f.pos.z + az * 0.7);
    f.tgtSpeed = 1.2;
    this.setState(f, 'startle', 0.4);
  }

  private startRise(f: Fish) {
    this.habitat.at(f.pos.x, f.pos.z, this.w);
    const up = Math.hypot(this.w.flowX, this.w.flowZ) > 0.05 ? 0.25 : 0;
    const u = Math.hypot(this.w.flowX, this.w.flowZ) || 1;
    f.tgt.set(f.pos.x - (this.w.flowX / u) * up, this.w.waterY, f.pos.z - (this.w.flowZ / u) * up);
    f.tgtSpeed = 0.55 + this.r() * 0.3;
    f.setFlag(F_SURFACED, false);
    this.setState(f, 'rise', 4);
  }

  /** Leap clear of the water: `v` is the vertical speed at the surface (2.6 m/s clears ~0.35 m). */
  private startJump(f: Fish, v: number) {
    this.habitat.at(f.pos.x, f.pos.z, this.w);
    f.tgt.set(Math.sin(f.yaw), v, Math.cos(f.yaw));
    f.setFlag(F_SURFACED, false);
    f.setFlag(F_ENTERED, false);
    this.setState(f, 'jump', 3);
  }

  private loose(f: Fish, why: 'spat' | 'threw') {
    f.fight = null;
    // out of 'bite' first: spook() hands a fish that holds the lure back to loose()
    if (f.state === 'bite') this.setState(f, 'refuse', 0);
    this.spook(f, this.player.x, this.player.z, why === 'spat' ? 10 + 15 * this.r() : 30 + 30 * this.r());
    this.onLoose?.(f.id, why);
  }

  private setState(f: Fish, s: FishState, dur: number) {
    if (s !== 'inspect') f.willTake = false;
    f.state = s;
    f.stateT = 0;
    f.stateDur = dur;
    f.lookT = 0;
  }

  // ------------------------------------------------------------------------------------------------ decisions

  private think(f: Fish, dt: number) {
    const H = this.habitat, w = this.w, t = this.time;
    switch (f.state) {
      case 'hold': {
        // a lie in the current: sway on station, dart out to intercept drifting food now and then
        H.at(f.home.x, f.home.z, w);
        const [a, b, c] = f.sway;
        const u = Math.hypot(w.flowX, w.flowZ) || 1;
        const tx = w.flowX / u, tz = w.flowZ / u;
        const lat = 0.1 * Math.sin(t * 0.43 + a) + 0.05 * Math.sin(t * 1.1 + b);
        const along = 0.07 * Math.sin(t * 0.31 + c);
        f.tgt.set(f.home.x - tz * lat + tx * along, this.holdY(f, w) + 0.03 * Math.sin(t * 0.7 + a), f.home.z + tx * lat + tz * along);
        f.tgtSpeed = 0.5;
        if (t > f.nextFeed) {
          f.nextFeed = t + 4 + this.r() * 10;
          // mostly up and out to meet the drift; now and then it drops back under a bit of food first
          const side = (this.r() - 0.5) * 1.1, upstream = this.r() < 0.25 ? -(0.35 + this.r() * 0.45) : 0.3 + this.r() * 0.5;
          f.tgt.set(f.home.x - tx * upstream - tz * side, Math.min(w.waterY - 0.1, f.pos.y + 0.08 + this.r() * 0.18), f.home.z - tz * upstream + tx * side);
          f.tgtSpeed = 0.7 + this.r() * 0.5;
          this.setState(f, 'feed', 0.6 + this.r() * 0.6);
        }
        break;
      }
      case 'feed':
        if (f.stateT > f.stateDur || f.pos.distanceTo(f.tgt) < 0.08) this.setState(f, f.home.holds ? 'hold' : 'hover', 2 + this.r() * 4);
        break;
      case 'cruise':
        if (f.pos.distanceTo(f.tgt) < 0.25 || f.stateT > 20) {
          // arrived (or lost): hover a while
          this.setState(f, 'hover', 2 + this.r() * 7);
          f.tgt.copy(f.pos);
        }
        break;
      case 'hover': {
        H.at(f.pos.x, f.pos.z, w);
        // tiny drift around the hover point, a slow turn now and then
        f.tgtSpeed = 0.08;
        if (f.stateT > f.stateDur) {
          if (f.home.holds) this.setState(f, 'hold', 0);
          else {
            this.setState(f, 'cruise', 0);
            this.pickCruiseTarget(f);
          }
        }
        break;
      }
      case 'startle':
        if (f.stateT > f.stateDur) this.setState(f, f.home.holds ? 'hold' : 'hover', 1.5 + this.r() * 2);
        break;
      case 'flee':
        if (f.stateT > f.stateDur) {
          // the dash is over: swim to cover
          f.tgt.copy(f.refugeP);
          f.tgtSpeed = 0.6 + this.r() * 0.3;
          this.setState(f, 'hide', 0);
        }
        break;
      case 'hide':
        if (f.pos.distanceTo(f.refugeP) < 0.4) f.tgtSpeed = 0.1;
        if (t > f.spookedUntil) {
          f.tgt.set(f.home.x, f.home.y, f.home.z);
          f.tgtSpeed = 0.3 + this.r() * 0.15;
          this.setState(f, 'return', 0);
        }
        break;
      case 'return':
        H.at(f.home.x, f.home.z, w);
        f.tgt.set(f.home.x, f.home.holds ? this.holdY(f, w) : f.home.y, f.home.z);
        if (Math.hypot(f.pos.x - f.home.x, f.pos.z - f.home.z) < (f.home.holds ? 0.3 : f.home.range * 0.6)) {
          if (f.home.holds) this.setState(f, 'hold', 0);
          else this.setState(f, 'hover', 1 + this.r() * 3);
        }
        break;
      case 'rise': {
        // tilts up until the snout breaks the surface, a sip, back down
        H.at(f.pos.x, f.pos.z, w);
        if (!f.hasFlag(F_SURFACED)) {
          // the snout stops just under the film (a sip): the ring says the rest
          f.tgt.y = w.waterY - 0.015 - f.sp.noseZ * f.scale * Math.sin(0.5);
          const m = this.mouthNow(f);
          if (m.y >= w.waterY - 0.03 || f.stateT > 3) {
            f.setFlag(F_SURFACED, true);
            this.outbox.push({ x: m.x, z: m.z, strength: 0.06 + 0.12 * clamp((f.length - 0.2) / 0.3, 0, 1), kind: 'rise', size: f.length });
            f.stateT = 0;
            f.stateDur = 0.5;
          }
        } else if (f.stateT > f.stateDur) {
          f.tgt.set(f.home.x, f.home.holds ? this.holdY(f, w) : f.home.y, f.home.z);
          this.setState(f, f.home.holds ? 'return' : 'hover', 2 + this.r() * 3);
        }
        break;
      }
      case 'jump':
        // movement handles the ballistic flight; leave once back under and slowed (bolting if spooked meanwhile)
        if ((f.hasFlag(F_ENTERED) && f.stateT > 0.6) || f.stateT > 4) {
          if (f.spookedUntil > t) this.spook(f, this.player.x, this.player.z, f.spookedUntil - t);
          else this.setState(f, f.home.holds ? 'return' : 'hover', 2);
        }
        break;
      case 'inspect': {
        const L = this.lure;
        if (!L) { this.setState(f, 'refuse', 1); break; }
        // hang behind and a little below the lure, now and then nose right up to it
        const lv = this.v3.set(L.x - this.lurePrev.x, L.y - this.lurePrev.y, L.z - this.lurePrev.z);
        const lsp = lv.length() / Math.max(dt, 1e-3);
        let bx = f.pos.x - L.x, bz = f.pos.z - L.z;
        if (lsp > 0.05) { bx = -lv.x; bz = -lv.z; }
        const bl = Math.hypot(bx, bz) || 1;
        const close = Math.sin(f.stateT * 2.3) > 0.75 ? 0.12 : 0.32 + 0.1 * Math.sin(f.stateT * 0.9);
        const ahead = f.length * 0.42;
        f.tgt.set(L.x + (bx / bl) * (close + ahead), L.y - 0.06, L.z + (bz / bl) * (close + ahead));
        H.at(f.tgt.x, f.tgt.z, w);
        if (w.depth > 0) f.tgt.y = clamp(f.tgt.y, w.bedY + 0.06 * f.scale + 0.04, w.waterY - 0.05 * f.scale - 0.03);
        // noticed from up to 5 m away: swim over, then look for `stateDur` s; give up if it never gets close
        const near = f.pos.distanceTo(f.tgt) < 0.3 + f.length * 0.5;
        if (near) f.lookT += dt;
        f.tgtSpeed = (near ? 0.35 : f.willTake ? 0.8 : 0.6) + lsp;
        if (!near && f.stateT > f.stateDur + (f.willTake ? 40 : 12)) this.setState(f, 'refuse', 1.5);
        else if (f.lookT > f.stateDur) {
          const small = 1.15 - 0.5 * clamp((f.length - f.sp.size[0]) / (f.sp.size[2] - f.sp.size[0]), 0, 1);
          const moving = clamp((L.speed ?? 0) / 0.5, 0, 1);
          const p = f.willTake ? 1 : clamp(f.sp.boldness * f.hunger * small * (0.55 + 0.45 * moving) * this.takeChance, 0, 1);
          const taker = this.fish.some((o) => o !== f && (o.state === 'take' || o.state === 'bite' || o.state === 'hooked'));
          if (!taker && this.r() < p) {
            f.tgtSpeed = 0.9 + this.r() * 0.6;
            this.setState(f, 'take', 1.2);
            this.onTake?.(f.id);
          } else this.setState(f, 'refuse', 1.5 + this.r());
        }
        break;
      }
      case 'take': {
        const L = this.lure;
        if (!L) { this.setState(f, 'refuse', 1); break; }
        const m = this.mouthNow(f);
        // aim the mouth (not the centre) at the lure
        f.tgt.set(f.pos.x + (L.x - m.x), f.pos.y + (L.y - m.y), f.pos.z + (L.z - m.z));
        if (Math.hypot(L.x - m.x, L.y - m.y, L.z - m.z) < 0.035 + 0.02 * f.scale) {
          const win = (f.sp.id === 'brown' ? 0.55 : 0.75) + this.r() * 0.6;
          this.setState(f, 'bite', win);
          f.hunger = Math.max(0.2, f.hunger - 0.3);
          this.onBite?.({
            fishId: f.id, species: f.sp.id, name: f.sp.name, lengthM: f.length, weightKg: f.weight,
            x: L.x, y: L.y, z: L.z, window: win, strength: L.kind === 'bobber' ? 0.35 + 0.4 * this.r() : 0.7 + 0.3 * this.r(),
          });
        } else if (f.stateT > f.stateDur) this.setState(f, 'refuse', 1.5);
        break;
      }
      case 'bite': {
        const L = this.lure;
        if (!L) { this.loose(f, 'spat'); break; }
        const m = this.mouthNow(f);
        f.tgt.set(f.pos.x + (L.x - m.x), f.pos.y + (L.y - m.y), f.pos.z + (L.z - m.z));
        if (f.stateT > f.stateDur) this.loose(f, 'spat');
        break;
      }
      case 'refuse':
        if (f.stateT < 0.05) {
          // turn away and drift off
          f.tgt.set(f.pos.x - Math.sin(f.yaw) * 0.8 + (this.r() - 0.5), f.pos.y - 0.05, f.pos.z - Math.cos(f.yaw) * 0.8 + (this.r() - 0.5));
          f.tgtSpeed = 0.35;
        }
        if (f.stateT > f.stateDur) {
          f.tgt.set(f.home.x, f.home.y, f.home.z);
          f.tgtSpeed = 0.3;
          this.setState(f, 'return', 0);
        }
        break;
      case 'hooked':
        this.fightThink(f, dt);
        break;
      case 'held':
        break;
      case 'gone':
        if (f.respawnAt > 0 && t > f.respawnAt) this.respawn(f);
        break;
    }
  }

  private pickCruiseTarget(f: Fish) {
    const H = this.habitat, w = this.w;
    // mostly near home, now and then a longer round of the pool
    const far = this.r() < 0.18;
    const R = far ? f.home.range * 2.2 : f.home.range;
    for (let k = 0; k < 12; k++) {
      const a = this.r() * TAU, d = R * Math.sqrt(this.r());
      const x = f.home.x + Math.cos(a) * d, z = f.home.z + Math.sin(a) * d;
      if (!H.at(x, z, w) || w.depth < 0.4 + 0.15 * f.scale) continue;
      if (H.nearestRock(x, z).d < 0.4) continue;
      f.tgt.set(x, this.cruiseY(f, w, this.r()), z);
      f.tgtSpeed = f.length * (0.6 + this.r() * 0.6) + 0.06;
      return;
    }
    f.tgt.set(f.home.x, f.home.y, f.home.z);
    f.tgtSpeed = 0.25;
  }

  private respawn(f: Fish) {
    // a new trout of the same kind, at its lie or the next best one away from the player
    f.visible = true;
    f.respawnAt = -1;
    f.spookedUntil = -1;
    f.hunger = 0.5 + this.r() * 0.5;
    this.placeAtHome(f);
  }

  private mouthNow(f: Fish) {
    this.q.setFromEuler(this.e.set(-f.pitch, f.yaw, f.roll, 'YXZ'));
    return f.mouth(this.v3b, this.q);
  }

  // ------------------------------------------------------------------------------------------------ the fight

  /** pick the next run: away from the angler, toward deep water and down the current, never into the shallows */
  private nextRun(f: Fish, first = false) {
    const fi = f.fight!;
    const H = this.habitat, w = this.w;
    const ax = f.pos.x - fi.tip.x, az = f.pos.z - fi.tip.z;
    const away = Math.atan2(ax, az);
    let best = -Infinity, bestA = away;
    for (let k = 0; k < 9; k++) {
      const a = away + (k - 4) * 0.3 + (this.r() - 0.5) * 0.2;
      const px = f.pos.x + Math.sin(a) * 2.5, pz = f.pos.z + Math.cos(a) * 2.5;
      if (!H.at(px, pz, w)) continue;
      const down = (Math.sin(a) * w.flowX + Math.cos(a) * w.flowZ) / Math.max(0.1, Math.hypot(w.flowX, w.flowZ));
      const s = Math.min(w.depth, 2) * 0.8 + Math.cos(a - away) * 0.7 + down * 0.3 + this.r() * 0.4;
      if (s > best) { best = s; bestA = a; }
    }
    fi.dir.set(Math.sin(bestA), 0, Math.cos(bestA));
    const tired = fi.stamina < 0.12;
    const roll = this.r();
    if (tired) {
      fi.phase = 'tired';
      fi.phaseDur = 99;
      fi.effort = 0.06;
    } else if (!first && roll < f.sp.jumpiness * 0.45 && fi.stamina > 0.3) {
      fi.phase = 'jump';
      fi.phaseDur = 2.5;
      fi.effort = 1;
      this.startJumpOnLine(f);
    } else if (!first && f.sp.sulks && roll < 0.45) {
      fi.phase = 'sulk';
      fi.phaseDur = 2 + this.r() * 3;
      fi.effort = 0.55;
      fi.dir.y = -0.6;
      fi.dir.normalize();
    } else if (!first && roll > 0.82) {
      fi.phase = 'shake';
      fi.phaseDur = 0.6 + this.r() * 0.6;
      fi.effort = 0.7;
    } else if (!first && roll > 0.7) {
      fi.phase = 'rest';
      fi.phaseDur = 1 + this.r() * 1.5;
      fi.effort = 0.15;
    } else if (!first && roll > 0.6 && fi.stamina > 0.25) {
      // up on top, rolling and shaking its head with the back out: the water flies
      fi.phase = 'thrash';
      fi.phaseDur = 1.2 + this.r() * 1.0;
      fi.effort = 0.95;
      fi.slapAt = THRASH_BEAT.first;
    } else {
      fi.phase = 'run';
      fi.phaseDur = (first ? 1.5 : 1.2) + this.r() * 3;
      fi.effort = 0.75 + this.r() * 0.25;
    }
    fi.phaseT = 0;
  }

  private startJumpOnLine(f: Fish) {
    f.setFlag(F_SURFACED, false);
    f.setFlag(F_ENTERED, false);
    f.tgt.set(f.fight!.dir.x, 2.4 + this.r() * 0.8, f.fight!.dir.z);
  }

  private fightThink(f: Fish, dt: number) {
    const fi = f.fight!;
    fi.phaseT += dt;
    fi.lineAge += dt;
    if (fi.lineAge > 0.5) fi.tension = 0;
    if (fi.phase !== 'tired' && fi.stamina < 0.12) this.nextRun(f);
    else if (fi.phaseT > fi.phaseDur && fi.phase !== 'tired') this.nextRun(f);
    // stamina: spent by the effort, faster against a tight line; a little back while it rests
    const tr = clamp(fi.tension / Math.max(fi.maxForce, 1), 0, 1.5);
    fi.stamina -= dt * (0.006 + 0.04 * fi.effort * (0.45 + 0.55 * tr)) * this.fightDrain;
    if (fi.effort < 0.2) fi.stamina += dt * 0.005;
    fi.stamina = clamp(fi.stamina, 0, 1);
    // slack line: a lively fish may throw the hook
    if (fi.tension < 0.3 && fi.stamina > 0.25) {
      fi.slackT += dt;
      const inAir = fi.phase === 'jump' && f.hasFlag(F_SURFACED) && !f.hasFlag(F_ENTERED);
      if (fi.slackT > 2.5 && !inAir && this.r() < 0.35 * dt) { this.loose(f, 'threw'); return; }
    } else fi.slackT = 0;
    const st = fi.state;
    st.stamina = fi.stamina;
    st.phase = fi.phase;
  }

  // ------------------------------------------------------------------------------------------------ movement

  private move(f: Fish, dt: number) {
    if (f.state === 'held' || f.state === 'gone') return;
    if (f.state === 'jump' || (f.state === 'hooked' && f.fight!.phase === 'jump')) {
      if (this.moveJump(f, dt)) {
        const st = f.fight?.state;
        if (st) {
          st.mouth.copy(this.mouthNow(f));
          st.velocity.copy(f.vel);
          st.airborne = f.hasFlag(F_SURFACED) && !f.hasFlag(F_ENTERED);
          st.pullForce = f.fight!.maxForce * 0.5;
          st.pull.copy(f.vel).normalize();
        }
        return;
      }
    }
    if (f.state === 'hooked') {
      this.moveHooked(f, dt);
      return;
    }
    const H = this.habitat, w = this.w;
    const d = this.desired;
    // desired ground velocity toward the target
    d.subVectors(f.tgt, f.pos);
    const dist = d.length();
    let speed = f.tgtSpeed;
    let accel = 1.5;
    switch (f.state) {
      case 'hold': speed = Math.min(0.5, dist * 1.6); accel = 2.5; break;
      case 'hover': speed = Math.min(0.08, dist * 0.6); accel = 0.6; break;
      case 'feed': speed = Math.min(f.tgtSpeed, dist * 3 + 0.1); accel = 4; break;
      case 'cruise': case 'return': speed = Math.min(f.tgtSpeed, dist * 0.9 + 0.05); accel = 1.2; break;
      case 'flee': case 'startle': accel = 9; break;
      case 'hide': speed = Math.min(f.tgtSpeed, dist * 0.8 + 0.05); accel = 1.5; break;
      case 'inspect': speed = Math.min(f.tgtSpeed, dist * 2.2); accel = 2.2; break;
      case 'take': speed = Math.min(f.tgtSpeed, dist * 6 + 0.2); accel = 6; break;
      case 'bite': speed = Math.min(1.5, dist * 8); accel = 8; break;
      case 'rise': speed = Math.min(f.tgtSpeed, dist * 2 + 0.05); accel = 2; break;
      case 'refuse': speed = Math.min(f.tgtSpeed, dist * 1.2 + 0.05); accel = 1.4; break;
    }
    if (dist > 1e-4) d.multiplyScalar(speed / dist);
    else d.set(0, 0, 0);
    const surfacing = f.state === 'rise' && !f.hasFlag(F_SURFACED);
    // separation from the others
    for (const o of this.fish) {
      if (o === f || !o.visible || o.state === 'held' || o.state === 'gone') continue;
      const sx = f.pos.x - o.pos.x, sy = f.pos.y - o.pos.y, sz = f.pos.z - o.pos.z;
      const dd = Math.hypot(sx, sy * 2, sz);
      const want = (f.length + o.length) * 0.55;
      if (dd < want && dd > 1e-4) {
        const push = ((want - dd) / want) * 0.35;
        d.x += (sx / dd) * push;
        d.y += (sy / dd) * push * 0.5;
        d.z += (sz / dd) * push;
      }
    }
    // look ahead: steer away from shallows and rocks
    const ahead = 0.4 + f.vel.length() * 0.6;
    const hx = f.pos.x + (f.vel.x || Math.sin(f.yaw) * 0.1) * ahead / Math.max(f.vel.length(), 0.1);
    const hz = f.pos.z + (f.vel.z || Math.cos(f.yaw) * 0.1) * ahead / Math.max(f.vel.length(), 0.1);
    const minD = this.minDepth(f);
    if (f.state !== 'bite' && f.state !== 'take' && (!H.at(hx, hz, w) || w.depth < minD)) {
      H.deeper(f.pos.x, f.pos.z, this.dir2);
      d.x += this.dir2.x * 0.6;
      d.z += this.dir2.z * 0.6;
    }
    const rk = H.nearestRock(f.pos.x, f.pos.z);
    if (rk.rock && rk.d < 0.25 && f.pos.y < rk.rock.topY + 0.1) {
      const rx = f.pos.x - rk.rock.x, rz = f.pos.z - rk.rock.z, rl = Math.hypot(rx, rz) || 1;
      d.x += (rx / rl) * (0.25 - rk.d) * 3;
      d.z += (rz / rl) * (0.25 - rk.d) * 3;
    }
    // accelerate toward it (ground frame)
    const dv = this.v3.subVectors(d, f.vel);
    const maxDv = accel * dt;
    const l = dv.length();
    if (l > maxDv) dv.multiplyScalar(maxDv / l);
    f.vel.add(dv);
    this.integrate(f, dt, surfacing);
  }

  private minDepth(f: Fish) {
    // a played-out fish can be drawn into a few inches of water, lying on its side
    return f.fight?.phase === 'tired' ? 0.05 * f.scale + 0.05 : 0.12 * f.scale + 0.08;
  }

  /** move, then keep the fish in the water: off the bed, under the surface, out of the shallows and the rocks */
  private integrate(f: Fish, dt: number, allowSurface = false) {
    const H = this.habitat, w = this.w;
    const px = f.pos.x, pz = f.pos.z;
    f.pos.addScaledVector(f.vel, dt);
    const minD = this.minDepth(f);
    if (!H.at(f.pos.x, f.pos.z, w) || w.depth < minD) {
      // the shallows: slide along the depth contour (drop the part of the move toward shallower water), lose the
      // speed into them and drift toward deeper water
      H.deeper(px, pz, this.dir2);
      const mx = f.pos.x - px, mz = f.pos.z - pz;
      const into = -(mx * this.dir2.x + mz * this.dir2.z);
      f.pos.x = px + mx + (into > 0 ? this.dir2.x * into : 0);
      f.pos.z = pz + mz + (into > 0 ? this.dir2.z * into : 0);
      if (!H.at(f.pos.x, f.pos.z, w) || w.depth < minD) {
        // a curving shore: step out a little as it slides
        f.pos.x += this.dir2.x * 0.03;
        f.pos.z += this.dir2.z * 0.03;
        if (!H.at(f.pos.x, f.pos.z, w) || w.depth < minD) {
          f.pos.x = px;
          f.pos.z = pz;
        }
      }
      const vin = -(f.vel.x * this.dir2.x + f.vel.z * this.dir2.z);
      if (vin > 0) {
        f.vel.x += this.dir2.x * vin;
        f.vel.z += this.dir2.z * vin;
      }
      f.vel.x += this.dir2.x * 0.3 * dt;
      f.vel.z += this.dir2.z * 0.3 * dt;
    }
    if (!H.at(f.pos.x, f.pos.z, w)) {
      // outside the field or on dry ground (only if it started there): hold it where it is, under the old level
      f.vel.set(0, Math.min(f.vel.y, 0), 0);
      return;
    }
    const rk = H.nearestRock(f.pos.x, f.pos.z);
    if (rk.rock && rk.d < 0.04 && f.pos.y < rk.rock.topY + 0.05) {
      const rx = f.pos.x - rk.rock.x, rz = f.pos.z - rk.rock.z, rl = Math.hypot(rx, rz) || 1;
      f.pos.x = rk.rock.x + (rx / rl) * (rk.rock.r + 0.04);
      f.pos.z = rk.rock.z + (rz / rl) * (rk.rock.r + 0.04);
    }
    // the bed under the nose and the tail too (sloping margins)
    const fx = Math.sin(f.yaw), fz = Math.cos(f.yaw);
    let bed = w.bedY;
    const nose = this.w2;
    if (H.at(f.pos.x + fx * f.length * 0.4, f.pos.z + fz * f.length * 0.4, nose)) bed = Math.max(bed, nose.bedY);
    if (H.at(f.pos.x - fx * f.length * 0.55, f.pos.z - fz * f.length * 0.55, nose)) bed = Math.max(bed, nose.bedY);
    H.at(f.pos.x, f.pos.z, w);
    // in water thinner than the fish (a played-out fish drawn into the shallows) the surface wins
    const hi = Math.max(w.waterY - (allowSurface ? 0.0 : 0.095 * f.scale + 0.02), bed + 0.03);
    const lo = Math.min(bed + 0.065 * f.scale + 0.035, hi);
    if (f.pos.y < lo) {
      f.pos.y = lo;
      if (f.vel.y < 0) f.vel.y = 0;
    }
    if (f.pos.y > hi) {
      f.pos.y = hi;
      if (f.vel.y > 0) f.vel.y = 0;
    }
  }

  /** ballistic leap; returns false once it is back under and the normal movement takes over */
  private moveJump(f: Fish, dt: number): boolean {
    const H = this.habitat, w = this.w;
    H.at(f.pos.x, f.pos.z, w);
    const surf = w.depth > 0 ? w.waterY : f.pos.y;
    if (!f.hasFlag(F_SURFACED)) {
      // launch: accelerate up toward the surface
      const want = this.v3.set(f.tgt.x * 0.9, f.tgt.y, f.tgt.z * 0.9);
      const dv = this.v3b.subVectors(want, f.vel);
      const l = dv.length(), maxDv = 14 * dt;
      if (l > maxDv) dv.multiplyScalar(maxDv / l);
      f.vel.add(dv);
      f.pos.addScaledVector(f.vel, dt);
      if (f.pos.y >= surf) {
        f.setFlag(F_SURFACED, true);
        f.vel.y = f.tgt.y;
        f.airT = 0;
        this.outbox.push({ x: f.pos.x, z: f.pos.z, strength: 0.55 + 0.35 * clamp(f.weight, 0, 1), kind: 'exit', size: f.length,
          dx: f.vel.x, dz: f.vel.z });
      }
      // never through the bed while it gathers speed
      if (f.pos.y < w.bedY + 0.05) f.pos.y = w.bedY + 0.05;
      return true;
    }
    if (!f.hasFlag(F_ENTERED)) {
      // in the air
      f.vel.y -= G * dt;
      f.vel.multiplyScalar(1 - 0.08 * dt);
      f.pos.addScaledVector(f.vel, dt);
      f.airT += dt;
      if (f.airT < TRAIL_TIME) {
        // water streams off the tail as it comes clear: a ragged curtain of drops falling back behind it
        const cp = Math.cos(f.pitch), tl = (f.sp.noseZ - f.sp.modelLength) * f.scale;
        const fx = Math.sin(f.yaw) * cp, fy = Math.sin(f.pitch), fz = Math.cos(f.yaw) * cp;
        this.outbox.push({ x: f.pos.x + fx * tl, y: f.pos.y + fy * tl, z: f.pos.z + fz * tl, strength: 1 - f.airT / TRAIL_TIME,
          kind: 'trail', size: f.length, vx: f.vel.x, vy: f.vel.y, vz: f.vel.z });
      }
      if (f.vel.y < 0 && f.pos.y < surf) {
        f.setFlag(F_ENTERED, true);
        this.outbox.push({ x: f.pos.x, z: f.pos.z, strength: 0.8 + 0.4 * clamp(f.weight, 0, 1), kind: 'entry', size: f.length });
      }
      return true;
    }
    // back in: water brakes the dive
    if (f.stateT < 4 && f.vel.y < -0.2) {
      f.vel.multiplyScalar(Math.exp(-5 * dt));
      this.integrate(f, dt);
      return true;
    }
    if (f.state === 'hooked' && f.fight!.phase === 'jump') f.fight!.phaseDur = 0;
    return false;
  }

  private moveHooked(f: Fish, dt: number) {
    const fi = f.fight!;
    const H = this.habitat, w = this.w;
    H.at(f.pos.x, f.pos.z, w);
    // run direction, shaken from side to side in a head shake
    let dx = fi.dir.x, dy = fi.dir.y, dz = fi.dir.z;
    const thrashing = fi.phase === 'thrash';
    if (fi.phase === 'shake' || thrashing) {
      const a = Math.sin(fi.phaseT * TAU * (thrashing ? 3 : 4)) * (thrashing ? 1.1 : 0.7);
      const c = Math.cos(a), s = Math.sin(a);
      [dx, dz] = [dx * c - dz * s, dx * s + dz * c];
    }
    // thrashing: straight up to the film and held there, rolling from side to side with the flank and tail out
    if (thrashing) dy = clamp((w.waterY - 0.004 - f.pos.y) * 4, -0.3, 1.2);
    // avoid the shallows ahead and the bank
    const minD = this.minDepth(f);
    if (!H.at(f.pos.x + dx * 0.8, f.pos.z + dz * 0.8, this.w2) || this.w2.depth < minD + 0.1) {
      H.deeper(f.pos.x, f.pos.z, this.dir2);
      dx = dx * 0.3 + this.dir2.x;
      dz = dz * 0.3 + this.dir2.z;
    }
    const dl = Math.hypot(dx, dy, dz) || 1;
    dx /= dl; dy /= dl; dz /= dl;
    // keep off the surface unless it is thrashing there; a tired fish comes up and rolls
    if (fi.phase === 'tired') dy = clamp((w.waterY - 0.1 - f.pos.y) * 2, -0.5, 0.5);
    const effort = fi.effort * (0.3 + 0.7 * Math.sqrt(fi.stamina));
    const thrust = fi.maxForce * effort;
    fi.pullNow = thrust;
    const mEff = f.weight * 1.3 + 0.05;
    const vTerm = 1.3 + 1.2 * Math.sqrt(f.weight);
    const kd = fi.maxForce / (vTerm * vTerm);
    // the line pulls the mouth toward the rod tip
    this.q.setFromEuler(this.e.set(-f.pitch, f.yaw, f.roll, 'YXZ'));
    const m = f.mouth(this.v3b, this.q);
    const lx = fi.tip.x - m.x, ly = fi.tip.y - m.y, lz = fi.tip.z - m.z, ll = Math.hypot(lx, ly, lz) || 1;
    H.currentAt(f.pos.x, f.pos.y, f.pos.z, this.cur);
    const rvx = f.vel.x - this.cur.x, rvy = f.vel.y, rvz = f.vel.z - this.cur.z;
    const rv = Math.hypot(rvx, rvy, rvz);
    const ax = (dx * thrust + (lx / ll) * fi.tension - kd * rv * rvx) / mEff;
    const ay = (dy * thrust + (ly / ll) * fi.tension * 0.6 - kd * rv * rvy) / mEff;
    const az = (dz * thrust + (lz / ll) * fi.tension - kd * rv * rvz) / mEff;
    f.vel.x += ax * dt;
    f.vel.y += ay * dt;
    f.vel.z += az * dt;
    const vmax = vTerm * 1.6;
    const vl = f.vel.length();
    if (vl > vmax) f.vel.multiplyScalar(vmax / vl);
    this.integrate(f, dt, true);
    // thrashing on the surface throws water (harder and faster in a thrash on top)
    H.at(f.pos.x, f.pos.z, w);
    fi.thrashT -= dt;
    // on top the body slams down on a steady beat (each half-roll one slap), otherwise now and then
    if (thrashing) while (fi.phaseT >= fi.slapAt + THRASH_BEAT.period) fi.slapAt += THRASH_BEAT.period;
    const due = thrashing ? fi.phaseT >= fi.slapAt : fi.thrashT <= 0;
    if (f.pos.y > w.waterY - 0.09 * f.scale - 0.04 && effort > 0.35 && due) {
      if (thrashing) fi.slapAt += THRASH_BEAT.period;
      fi.thrashT = thrashing ? THRASH_BEAT.period : 0.35 + this.r() * 0.35;
      const k = effort * clamp(f.weight * 1.5, 0.3, 1);
      const hx = Math.sin(f.yaw), hz = Math.cos(f.yaw);
      this.outbox.push({
        x: f.pos.x, z: f.pos.z, strength: thrashing ? 0.32 + 0.3 * k : 0.2 + 0.25 * k, kind: 'slap', size: f.length,
        dx: hx, dz: hz, side: f.roll >= 0 ? 1 : -1,
      });
      fi.state.thrashes++;
    }
    const st = fi.state;
    st.mouth.copy(this.mouthNow(f));
    st.velocity.copy(f.vel);
    st.pull.set(dx, dy, dz);
    st.pullForce = thrust;
    st.airborne = false;
    // within a rod's length of the angler (horizontally): net it or tail it by hand
    st.landable = fi.phase === 'tired' && Math.hypot(st.mouth.x - fi.tip.x, st.mouth.z - fi.tip.z) < 3.0;
  }

  // ------------------------------------------------------------------------------------------------ orientation and swim

  private animate(f: Fish, dt: number) {
    if (f.state === 'held') {
      // feeble flaps in the hand now and then
      const flap = Math.sin(this.time * 0.9 + f.sway[0]) > 0.6 ? 1 : 0;
      f.amp = smooth(f.amp, flap * 0.07, 6, dt);
      f.freq = 2.6;
      f.phase = (f.phase + TAU * f.freq * dt) % TAU;
      f.bend = smooth(f.bend, 0, 3, dt);
      f.finAmp = smooth(f.finAmp, 0.2, 3, dt);
      return;
    }
    if (f.state === 'gone') return;
    const H = this.habitat;
    H.currentAt(f.pos.x, f.pos.y, f.pos.z, this.cur);
    const inAir = (f.state === 'jump' || (f.state === 'hooked' && f.fight?.phase === 'jump')) && f.hasFlag(F_SURFACED) && !f.hasFlag(F_ENTERED);
    // heading follows the motion through the water (into the current while holding on station)
    let rvx = f.vel.x - (inAir ? 0 : this.cur.x), rvz = f.vel.z - (inAir ? 0 : this.cur.z);
    const rvy = f.vel.y;
    if (f.state === 'hooked' && !inAir) {
      // on a tight line the head is pulled round toward the angler
      const fi = f.fight!;
      const tx = fi.tip.x - f.pos.x, tz = fi.tip.z - f.pos.z, tl = Math.hypot(tx, tz) || 1;
      const k = clamp(fi.tension / Math.max(fi.maxForce, 1), 0, 1) * (fi.phase === 'tired' ? 1 : 0.5);
      rvx = rvx * (1 - k) + (tx / tl) * k * 0.5;
      rvz = rvz * (1 - k) + (tz / tl) * k * 0.5;
    }
    const rh = Math.hypot(rvx, rvz);
    let wantYaw = f.yaw;
    // on its lie it faces into the current while it shifts about the station (in the margin's gentle current its own
    // station-keeping turned it across the stream); farther out it turns to swim back
    const onStation = f.state === 'hold' && Math.hypot(f.tgt.x - f.pos.x, f.tgt.z - f.pos.z) < 0.4;
    if (onStation && Math.hypot(this.cur.x, this.cur.z) > 0.004) wantYaw = Math.atan2(-this.cur.x, -this.cur.z);
    else if (rh > 0.04) wantYaw = Math.atan2(rvx, rvz);
    else if (f.state === 'hover' || f.state === 'hide') wantYaw = f.yaw + Math.sin(this.time * 0.21 + f.sway[1]) * 0.25 * dt;
    if (f.state === 'bite' && this.lure) {
      // the mouth stays on the lure: face it
      wantYaw = Math.atan2(this.lure.x - f.pos.x, this.lure.z - f.pos.z);
    }
    const burst = f.state === 'flee' || f.state === 'startle' || f.state === 'take' || f.state === 'jump' || f.state === 'hooked';
    const maxTurn = burst ? 7 : f.state === 'feed' || f.state === 'inspect' ? 3.2 : f.state === 'hover' ? 0.9 : 1.8;
    const dy = clamp(wrapAngle(wantYaw - f.yaw), -maxTurn * dt, maxTurn * dt);
    f.yaw = wrapAngle(f.yaw + dy);
    f.yawRate = smooth(f.yawRate, dy / Math.max(dt, 1e-4), 10, dt);
    // pitch with the climb or dive; in the air it follows the arc
    const rs = Math.hypot(rh, rvy);
    let wantPitch = rs > 0.06 ? Math.atan2(rvy, Math.max(rh, 0.05)) : 0;
    const pmax = inAir || f.state === 'rise' || f.state === 'jump' ? 1.35 : 0.55;
    wantPitch = clamp(wantPitch, -pmax, pmax);
    if (f.state === 'rise' && !f.hasFlag(F_SURFACED)) wantPitch = Math.max(wantPitch, 0.45);
    f.pitch = smooth(f.pitch, wantPitch, inAir ? 12 : 4, dt);
    // a slight bank into turns; a hooked fish rolls in the shakes and lies on its side when played out
    let wantRoll = clamp(-f.yawRate * 0.06, -0.35, 0.35);
    let rollRate = 5;
    if (f.state === 'hooked') {
      const fi = f.fight!;
      if (fi.phase === 'tired') wantRoll = 1.25;
      else if (fi.phase === 'shake') wantRoll = Math.sin(fi.phaseT * TAU * 4) * 0.6;
      else if (fi.phase === 'thrash') {
        // slammed from one side onto the other, flank up at every slap (each half-roll one beat)
        wantRoll = Math.cos((Math.PI * (fi.phaseT - THRASH_BEAT.first)) / THRASH_BEAT.period) * 1.4;
        rollRate = 22;
      } else wantRoll += Math.sin(this.time * 2.1 + f.sway[2]) * 0.15;
    } else if (inAir) {
      // a leap turns it part way onto its side, the back toward one bank, the flank catching the sun
      wantRoll = (f.sway[2] > Math.PI ? 1 : -1) * (0.75 + 0.25 * Math.sin(f.stateT * 4 + f.sway[2]));
      rollRate = 9;
    } else if (f.state === 'feed' || f.state === 'take') wantRoll += Math.sin(f.stateT * 6) * 0.25; // the flank flashes as it takes
    f.roll = smooth(f.roll, wantRoll, rollRate, dt);
    // tail beat: relative speed sets the frequency, the state the amplitude
    const vRel = Math.hypot(rh, rvy);
    const L = f.length;
    let amp: number, hz: number, fin: number;
    switch (f.state) {
      case 'hover': case 'hide':
        amp = SWIM.hover.amp; hz = SWIM.hover.hz; fin = 1; break;
      case 'hold':
        amp = SWIM.hold.amp; hz = Math.max(SWIM.hold.hz, vRel / (0.65 * L)); fin = 0.55; break;
      case 'flee': case 'startle': case 'take':
        amp = SWIM.burst.amp; hz = Math.max(4, Math.min(SWIM.burst.hz, vRel / (0.45 * L))); fin = 0; break;
      case 'jump':
        // in the air the tail keeps kicking: a big slow S
        amp = inAir ? 0.21 : SWIM.burst.amp; hz = inAir ? 2.8 : SWIM.burst.hz; fin = 0; break;
      case 'hooked': {
        const fi = f.fight!;
        const e = fi.effort * (0.3 + 0.7 * Math.sqrt(fi.stamina));
        amp = fi.phase === 'tired' ? 0.04 : fi.phase === 'thrash' ? SWIM.line.amp * 1.25 : SWIM.line.amp * (0.6 + 0.4 * e);
        hz = fi.phase === 'tired' ? 1.4 : fi.phase === 'thrash' ? 5.5 : SWIM.line.hz * (0.6 + 0.6 * e);
        fin = 0.2;
        break;
      }
      case 'bite':
        amp = SWIM.hold.amp * 1.5; hz = 2.5; fin = 0.6; break;
      default:
        amp = SWIM.cruise.amp * clamp(vRel / (0.8 * L) + 0.4, 0.5, 1.2);
        hz = clamp(vRel / (0.65 * L), 1.2, 4);
        fin = clamp(1 - vRel / (0.8 * L), 0.1, 0.8);
    }
    f.amp = smooth(f.amp, amp, inAir ? 10 : 3, dt);
    f.freq = smooth(f.freq, Math.min(hz, 7), 4, dt);
    f.phase = (f.phase + TAU * f.freq * dt) % TAU;
    // the body curves into a turn (a C-shape about the centre of mass); even a resting trout keeps a slight curve,
    // its tail angled into the current, and a leaping one flexes into a C and back
    let wantBend = clamp(f.yawRate * 0.075, -0.45, 0.45);
    if (inAir) wantBend = 0.7 * Math.sin(f.airT * TAU * 1.4 + f.sway[1]);
    else if (f.state === 'hold' || f.state === 'hover' || f.state === 'hide') wantBend += 0.22 * Math.sin(this.time * 0.11 + f.sway[1] * 3);
    f.bend = smooth(f.bend, wantBend, inAir ? 14 : 8, dt);
    f.finAmp = smooth(f.finAmp, fin, 2, dt);
    f.finPhase = (f.finPhase + TAU * (1.6 + 1.4 * f.finAmp) * dt) % TAU;
    // leaping or thrashing at the surface the body breaks the film: no pre-warp then
    if (f.state === 'jump' || f.state === 'hooked') {
      H.at(f.pos.x, f.pos.z, this.w);
      f.prewarp = !inAir && this.w.waterY - f.pos.y > 0.13 * f.scale + 0.03;
    } else f.prewarp = true;
  }

  /** splashes of the last stageJump, with their age at the frozen moment */
  readonly stagedSplashes: Array<SplashRequest & { age: number }> = [];

  /**
   * Captures (shot mode freezes time): one trout `age` seconds into a leap across the view, by default the pool
   * trout nearest the shallow margin in front of the C2 spot. Its splashes land in `stagedSplashes`.
   */
  stageJump(age: number, id?: number): Fish | null {
    // by default the rainbow holding on the margin in front of C2 (round 1 took the pool trout nearest (3.4, 1.4),
    // the same fish while it cruised there)
    const f: Fish | undefined = id !== undefined ? this.fish[id] : this.fish[this.marginIds[0] ?? -1];
    if (!f) return null;
    f.yaw = -Math.PI / 2;
    // it rolls with the flank toward C2 and the low bank camera (both on its +z side) and up to the sky: rolled the
    // other way they saw its dark back edge-on, a stick
    if (f.sway[2] < Math.PI) f.sway[2] += Math.PI;
    // it rises from where it cruised in round 1 (a quarter metre down), so the leap keeps its height in the captures
    this.habitat.at(f.pos.x, f.pos.z, this.w);
    if (this.w.depth > 0.4) f.pos.y = Math.max(f.pos.y, this.w.waterY - 0.26);
    this.startJump(f, 2.9);
    this.outbox.length = 0;
    this.stagedSplashes.length = 0;
    const dt = 1 / 120;
    for (let t = 0; t < age; t += dt) {
      f.stateT += dt;
      this.move(f, dt);
      this.animate(f, dt);
      for (const s of this.outbox) this.stagedSplashes.push({ ...s, age: Math.max(0.02, age - t - dt) });
      this.outbox.length = 0;
    }
    // the frozen frame catches it hard bent (a real leap photo is a C or an S, never a gently curved plank)
    if (f.pos.y > this.w.waterY) f.bend = (f.bend < 0 ? -1 : 1) * Math.max(Math.abs(f.bend), 0.62);
    return f;
  }

  /** stations of one kind (debug) */
  stations(kind?: StationKind) {
    return this.fish.map((f) => f.home).filter((s) => !kind || s.kind === kind);
  }

  /** a short human-readable summary (debug overlay, tests) */
  summary() {
    return this.fish.map((f) => `${String(f.id).padStart(2)} ${f.sp.id.padEnd(9)} ${(f.length * 100).toFixed(0)}cm ${f.home.kind.padEnd(6)} ${f.state.padEnd(7)} ` +
      `(${f.pos.x.toFixed(1)}, ${f.pos.y.toFixed(2)}, ${f.pos.z.toFixed(1)}) depth ${this.habitat.depthAt(f.pos.x, f.pos.z).toFixed(2)}`);
  }
}

export { REACH };
