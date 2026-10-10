import * as THREE from 'three/webgpu';
import type { CaughtFish, CharacterHooks, FishingButtons, FishingState, FishSource, FishSourceBite, HeldFish, PromptKind, SpeciesId } from './types';
import { HABITATS, SPECIES, SPECIES_IDS, habitatAt, weightKg, type Habitat, type SpeciesDef } from './species';
import type { FishingEventBus, FishingEventType, SoundId } from './events';
import { Rope } from './Rope';
import type { WaterPoint, WaterProbe } from './WaterProbe';
import { TUNE } from './config';
import { Rng, approach, clamp, easeInOut, easeOutQuad, smoothstep } from './util';

/** Character hooks with every optional query filled in by the host (see index.ts). */
export type ResolvedHooks = CharacterHooks & Required<Pick<CharacterHooks, 'getAim' | 'getFeet' | 'getFishHold'>>;

const G = 9.81;
const H = 1 / 120;
const UP = new THREE.Vector3(0, 1, 0);
const ZERO = new THREE.Vector3();
const _v = new THREE.Vector3(), _w = new THREE.Vector3(), _fwd = new THREE.Vector3();
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion();

type FishMode = 'none' | 'run' | 'side' | 'hold' | 'tired' | 'jump' | 'thrash' | 'held' | 'free';

/** The hooked (then held, then released) fish. */
export class FishBody {
  mode: FishMode = 'none';
  species: SpeciesDef = SPECIES.brown;
  lengthCm = 45;
  weightKg = 1;
  /** root scale: caught length / model length */
  scale = 1;
  /** pull factor from size (heavier fish pull harder) */
  sizePull = 1;
  readonly pos = new THREE.Vector3();
  readonly vel = new THREE.Vector3();
  /** desired swim direction (unit, horizontal) */
  readonly dir = new THREE.Vector3(0, 0, 1);
  readonly quat = new THREE.Quaternion();
  modeTime = 0;
  side = 1;
  stamina = 1;
  pull = 0;
  speed = 0;
  depth = 0.5;
  swimHz = 2;
  swimAmp = 0.08;
  /** accumulated swim phase (rad), the swim shader's clock */
  swimPhase = 0;
  roll = 0;
  visible = false;
  inAir = false;
  /** false while the character parents the fish (getFishHold returned false) */
  placed = true;
  freeTime = 0;
  thrashTime = 0;
  thrashCooldown = 0;
  jumpIn = -1;
  flexTime = 0;
  /** sag between the hands (fraction of its length) while held */
  sag = 0;
  /** released by hand: rests a moment before swimming off (an escaped fish just goes) */
  recover = false;

  reset() {
    this.mode = 'none';
    this.visible = this.inAir = false;
    this.placed = true;
    this.modeTime = this.freeTime = this.thrashTime = this.flexTime = this.roll = this.swimPhase = this.sag = 0;
    this.thrashCooldown = 2;
    this.jumpIn = -1;
    this.side = 1;
    this.stamina = 1;
    this.recover = false;
    this.vel.set(0, 0, 0);
  }
}

/** cast speed for a flat-ground distance: one-time calibration of the drag integrator */
function flatRange(v: number, h0: number): number {
  const e = (TUNE.castElevation * Math.PI) / 180;
  let x = 0, y = h0, vx = Math.cos(e) * v, vy = Math.sin(e) * v;
  const dt = 1 / 120;
  for (let i = 0; i < 2400 && y > 0; i++) {
    const s = Math.hypot(vx, vy);
    vx -= vx * s * TUNE.airDrag * dt;
    vy -= (G + vy * s * TUNE.airDrag) * dt;
    x += vx * dt;
    y += vy * dt;
  }
  return x;
}
function speedFor(dist: number): number {
  let lo = 0.5, hi = 60;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (flatRange(mid, 2.0) < dist) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}
const V_MIN2 = speedFor(TUNE.castMin) ** 2;
const V_MAX2 = speedFor(TUNE.castMax) ** 2;

export const AIM_POINTS = 28;
/** the float at rest: its waterline this far above its middle, so only the painted cap and the quill show */
const REST_SINK = 0.013;
/** line (m) a float drifting off may draw from the reel after it lands before he holds it */
const FEED = 1.5;

export class FishingSim {
  state: FishingState = 'idle';
  stateTime = 0;
  /** rod tip, aim and feet: refreshed by the host every frame before update() */
  readonly tip = new THREE.Vector3();
  readonly aimOrigin = new THREE.Vector3();
  readonly aimDir = new THREE.Vector3(0, 0, -1);
  readonly feet = new THREE.Vector3();

  power = 0;
  castPower = 0;
  readonly rope = new Rope(40);
  readonly bobber = new THREE.Vector3();
  readonly bobberVel = new THREE.Vector3();
  /** visual tilt of the float (x, z lean) */
  readonly bobberTilt = new THREE.Vector2();
  bobberSink = 0;
  bobberSinkVel = 0;
  sinkTarget = 0;
  bobberVisible = true;
  /** the float is on water (false: it landed on the bank) */
  onWater = false;
  readonly lure = new THREE.Vector3();
  readonly lureVel = new THREE.Vector3();
  readonly hook = new THREE.Vector3();
  /** the rope's end is pinned (to the float, the lure or the fish's mouth) */
  endPinned = false;

  /** aim preview: points along the predicted flight (and whether each is over water), landing point, whether it lands in fishable water */
  readonly aimPts = new Float32Array(AIM_POINTS * 3);
  readonly aimWet = new Uint8Array(AIM_POINTS);
  aimCount = 0;
  readonly aimLanding = new THREE.Vector3();
  aimOnWater = false;

  habitat: Habitat = 'pool';
  biteTimer = Infinity;
  private nibbles = [0, 0, 0];
  private nibbleCount = 0;
  private hookWindow = 1;
  private twitchT = 0;
  private biteDips = 0;

  readonly fish = new FishBody();
  private held: HeldFish | null = null;
  tension = 0;
  strain = 0;
  slackTime = 0;
  reeling = false;
  dragging = false;
  private dragSoundT = 0;
  private reelRate = 0;
  private readonly landFrom = new THREE.Vector3();
  private readonly landFromQ = new THREE.Quaternion();
  private letGo = 0.6;
  private released = false;

  bend = 0;
  private bendVel = 0;
  private bendTarget = 0;
  readonly bendDir = new THREE.Vector3(0, 0, -1);
  /** reel handle turns accumulated this frame (+ winding in); the host applies and clears it */
  crankDelta = 0;

  caught = 0;
  biggest: CaughtFish | null = null;
  last: CaughtFish | null = null;
  lastRecord = false;
  prompt: PromptKind = 'cast';
  toast: string | null = null;
  toastTime = 0;

  /** shot poses: the hooked fish comes up thrashing at this many seconds into the fight (its back out, spray flying) */
  forceThrashAt = -1;
  /** forced bite plan for deterministic shot poses */
  forceBite: { delay: number; species: SpeciesId; lengthCm: number; nibbles: number[] } | null = null;
  /** shot poses: rings only (droplets would hang in the frozen frame) */
  ringsOnly = false;
  suppressSplash = false;
  /** no splashes or rings at all (dry runs of shot poses) */
  muteWater = false;
  /** the float's next plunge pushes out a ring (re-armed when it rises again) */
  private dipArmed = false;
  private lastDipT = -1;
  /** a float riding still water now and then sends out a faint ring (the bait settling, the line easing): the clock to the next */
  private settleT = 0;
  private settleN = 0;
  /** line drawn off by the drifting float since it landed (FEED at most) */
  private fed = 0;
  /** the water surface over the hooked or released fish (for its shape seen through the water) */
  get fishSurfaceY(): number {
    const wp = this.water.sample(this.fish.pos.x, this.fish.pos.z, this.wq);
    return wp.wet ? wp.y : -1e9;
  }

  /** release: where the fish is put back and the way it faces */
  readonly releaseAt = new THREE.Vector3();
  readonly releaseDir = new THREE.Vector3(-1, 0, 0);
  /** fight: where the line cut the surface last step, and the clocks of the disturbances round the fish */
  private readonly lineIn = new THREE.Vector3();
  private wakeT = 0;
  private sprayT = 0;
  private swirlT = 0;
  private scatterT = 0;
  private wristT = 0;
  private pillowT = 0;
  private dripT = 0;
  private swirlSpin = 1;
  /**
   * Part 7's trout (world.fish). With it the bites come from the school (a trout seen coming to the bait takes it)
   * and the hooked, landed and released trout is the school's; null: part 6's own bites and fish (the fallback).
   */
  school: FishSource | null = null;
  /** the school's trout that bit, is on the line or in the hands (-1: none) */
  schoolFish = -1;
  private attracted = false;
  /** seconds since a trout was sent to the bait: one that gave up chasing it (a drifting float) is replaced */
  private attractT = 0;
  private lureOut = true;
  private readonly lureIn = { x: 0, y: 0, z: 0, kind: 'bobber' as const, speed: 0 };
  private lastThrashes = 0;
  private wasAirborne = false;
  private thrashSoundT = 0;
  /** the school trout's heading last step (NaN: none yet) and the clock between its turn boils */
  private turnHeading = Number.NaN;
  private boilT = 0;
  /** where the school's trout is let go (the first water over 12 cm from the hands) */
  private readonly schoolReleaseAt = new THREE.Vector3();

  private acc = 0;
  private time = 0;
  private castP = false;
  private castR = false;
  private reelP = false;
  private cancelP = false;
  private castHeld = false;
  private reelHeld = false;
  private wp: WaterPoint = { y: 0, flowX: 0, flowZ: 0, speed: 0, depth: 0, groundY: 0, wet: false };
  private wq: WaterPoint = { y: 0, flowX: 0, flowZ: 0, speed: 0, depth: 0, groundY: 0, wet: false };

  constructor(
    readonly water: WaterProbe,
    readonly hooks: ResolvedHooks,
    readonly events: FishingEventBus,
    readonly rng: Rng,
    private fishRoot: (s: SpeciesId) => THREE.Object3D,
  ) {}

  /** back to a dangling float at the tip (new tackle) */
  reset() {
    this.enter('idle');
    this.acc = 0;
    this.reel(false, 0);
    this.released = false;
    this.toast = null;
    this.toastTime = 0;
    this.bobberVisible = true;
    this.castP = this.castR = this.reelP = this.cancelP = false;
    this.tension = this.strain = 0;
    this.bend = this.bendVel = 0;
    this.bendDir.set(0, 0, -1);
    this.fish.reset();
    if (this.school && this.schoolFish >= 0) this.school.unhook(this.schoolFish);
    this.schoolFish = -1;
    this.feedLure(false);
    this.rope.reset(this.tip.x, this.tip.y, this.tip.z, this.tip.x, this.tip.y - TUNE.hangLength, this.tip.z);
    this.rope.rest = TUNE.hangLength;
    this.hooks.idle();
  }

  /** the bait under the float, for the school (once per frame while it is in the water) */
  private feedLure(inWater: boolean) {
    const sc = this.school;
    if (!sc) return;
    if (inWater) {
      const l = this.lureIn;
      l.x = this.hook.x;
      l.y = this.hook.y;
      l.z = this.hook.z;
      l.speed = Math.hypot(this.bobberVel.x, this.bobberVel.z);
      sc.setLure(l);
      this.lureOut = false;
    } else if (!this.lureOut) {
      sc.setLure(null);
      this.lureOut = true;
    }
  }

  /** per frame: dt (s), t = water clock (s), buttons with their edges since the last frame */
  update(dt: number, t: number, b: FishingButtons) {
    this.castP ||= b.castPressed;
    this.castR ||= b.castReleased;
    this.reelP ||= b.reelPressed;
    this.cancelP ||= !!b.cancel;
    this.castHeld = b.cast;
    this.reelHeld = b.reel;
    this.time = t;
    if (dt > 0) {
      this.rope.sampleWater(this.water);
      this.acc += dt;
      let n = 0;
      while (this.acc >= H && n < 8) {
        this.step(H);
        this.acc -= H;
        n++;
      }
      if (n === 8) this.acc = 0;
      if (this.toastTime > 0 && (this.toastTime -= dt) <= 0) this.toast = null;
      this.feedLure((this.state === 'floating' || this.state === 'bite') && this.onWater);
    }
    this.frameOutputs();
  }

  // ------------------------------------------------------------------------------------------------------------

  private enter(s: FishingState) {
    this.state = s;
    this.stateTime = 0;
    this.rope.drawRest = 0;
    this.rope.airDamp = 0.996;
    this.rope.airNormalDamp = 1;
    this.rope.beadIndex = -1;
    const e = this.events.begin('state', null, this.bobber.x, this.bobber.y, this.bobber.z);
    e.state = s;
    this.events.send();
  }

  private emit(type: FishingEventType, sound: SoundId | null, p: THREE.Vector3, gain = 1) {
    const e = this.events.begin(type, sound, p.x, p.y, p.z, gain);
    e.state = this.state;
    e.tension = this.tension;
    e.species = this.fish.mode === 'none' ? null : this.fish.species.id;
    e.lengthCm = this.fish.lengthCm;
    e.weightKg = this.fish.weightKg;
    return e;
  }

  private say(text: string, seconds = 2.4) {
    this.toast = text;
    this.toastTime = seconds;
  }

  private splash(x: number, z: number, strength: number) {
    if (this.suppressSplash || this.muteWater) return;
    if (this.ringsOnly) this.water.splashRings(x, z, this.time, strength);
    else this.water.splash(x, z, this.time, strength);
  }

  private boil(x: number, z: number, amplitude: number, radius: number, life = 2.2) {
    if (!this.muteWater) this.water.boil(x, z, this.time, amplitude, radius, life);
  }

  private swirl(x: number, z: number, amplitude: number, radius: number, life = 1.3) {
    if (this.muteWater) return;
    this.swirlSpin = -this.swirlSpin;
    this.water.swirl(x, z, this.time, amplitude, radius, life, this.swirlSpin);
  }

  private seed() {
    return Math.floor(this.rng.next() * 1e9);
  }

  /** a crisp ripple train (no-op in the dry runs of the photo poses) */
  private train(x: number, z: number, amplitude: number, life = 2.5, wavelength = 0.026, speed = 0.26, r0 = 0.02) {
    if (this.muteWater) return;
    this.water.train(x, z, this.time, amplitude, life, wavelength, speed, r0);
  }

  private step(h: number) {
    this.stateTime += h;
    const castPressed = this.castP, castReleased = this.castR, reelPressed = this.reelP, cancel = this.cancelP;
    this.castP = this.castR = this.reelP = this.cancelP = false;
    const rope = this.rope;
    rope.setStart(this.tip.x, this.tip.y, this.tip.z);

    switch (this.state) {
      case 'idle':
        this.stepHanging(h);
        this.bendTarget = 0.035;
        if (castPressed) {
          this.power = 0;
          this.enter('aiming');
          this.hooks.playAim?.(0);
        }
        break;

      case 'aiming': {
        this.stepHanging(h, TUNE.castDrop);
        this.power = easeOutQuad(clamp(this.stateTime / TUNE.chargeTime, 0, 1));
        // the float swinging under the drawn-back tip bows it back
        this.bendTarget = 0.11;
        this.bendDir.subVectors(this.bobber, this.tip);
        if (reelPressed || cancel) {
          this.enter('idle');
          this.hooks.idle();
        } else if (castReleased || !this.castHeld) {
          if (this.power < 0.06) {
            this.enter('idle');
            this.hooks.idle();
          } else {
            this.castPower = this.power;
            const delay = Math.max(0, this.hooks.playCast(this.castPower));
            this.letGo = delay;
            this.enter('windup');
            this.emit('cast', this.castPower > 0.7 ? 'cast_whoosh_heavy' : 'cast_whoosh', this.tip, 0.5 + 0.5 * this.castPower).strength = this.castPower;
            this.events.send();
          }
        }
        break;
      }

      case 'windup':
        this.stepHanging(h, TUNE.castDrop);
        // the tip lags behind the forward stroke
        this.bendTarget = 0.28 * this.castPower;
        this.bendDir.copy(this.aimDir).negate();
        if (this.stateTime >= this.letGo) this.launch();
        break;

      case 'flight':
        this.stepFlight(h);
        break;

      case 'floating':
        this.stepFloating(h, castPressed);
        break;

      case 'bite':
        this.stepBite(h, castPressed);
        break;

      case 'fight':
        this.stepFight(h);
        break;

      case 'landing':
      case 'holding':
      case 'releasing':
        this.stepLanded(h);
        break;

      case 'snapped':
        rope.step(h, false);
        this.bendTarget = 0;
        if (this.stateTime > 2.0) {
          this.bobberVisible = true;
          rope.rest = TUNE.hangLength;
          rope.reset(this.tip.x, this.tip.y, this.tip.z, this.tip.x, this.tip.y - TUNE.hangLength, this.tip.z);
          this.enter('idle');
          this.hooks.idle();
        }
        break;
    }

    if (this.fish.mode === 'free') this.stepFreeFish(h);
    this.stepRodBend(h);
  }

  /** float hanging from the tip on a short line (idle, aiming, windup); drawn back to cast he lets out a longer drop */
  private stepHanging(h: number, drop = TUNE.hangLength) {
    const rope = this.rope;
    rope.rest += (drop - rope.rest) * approach(4, h);
    rope.endInvMass = 0.25;
    // the hand steadies a hanging float: a swing or two, not a pendulum for seconds; drawing the rod back he lifts it
    // slowly, so the float swings under the tip instead of being flung over it
    rope.airDamp = this.state === 'aiming' ? 0.955 : 0.992;
    this.endPinned = false;
    rope.step(h, false);
    this.bobber.set(rope.endX(), rope.endY(), rope.endZ());
    this.hook.set(this.bobber.x, this.bobber.y - 0.06, this.bobber.z);
    this.onWater = false;
    this.bobberSink = this.bobberSinkVel = 0;
    this.bobberTilt.set(0, 0);
  }

  private launchVelocity(power: number, out: THREE.Vector3) {
    const e = (TUNE.castElevation * Math.PI) / 180;
    const v = Math.sqrt(V_MIN2 + (V_MAX2 - V_MIN2) * power);
    out.copy(this.aimDir).multiplyScalar(Math.cos(e) * v);
    out.y = Math.sin(e) * v;
    return out;
  }

  private launch() {
    this.lure.copy(this.aimOrigin);
    this.launchVelocity(this.castPower, this.lureVel);
    // lay the line straight from the tip to the float: the dangling line would whip into a pigtail
    this.rope.reset(this.tip.x, this.tip.y, this.tip.z, this.lure.x, this.lure.y, this.lure.z);
    this.rope.rest = Math.max(0.3, this.lure.distanceTo(this.tip));
    this.endPinned = true;
    this.enter('flight');
    // the line peeling off behind the float: little drag along it, a lot across it (it trails, never overtakes)
    this.rope.airDamp = 0.985;
    this.rope.airNormalDamp = 0.9;
    if (this.castPower > 0.35) {
      this.emit('payout', 'reel_pull', this.tip, 0.4 + 0.6 * this.castPower).strength = this.castPower;
      this.events.send();
    }
  }

  private integrateLure(p: THREE.Vector3, v: THREE.Vector3, h: number) {
    const s = v.length();
    v.addScaledVector(v, -s * TUNE.airDrag * h);
    v.y -= G * h;
    p.addScaledVector(v, h);
  }

  private stepFlight(h: number) {
    const rope = this.rope;
    this.integrateLure(this.lure, this.lureVel, h);
    // the line peels off the reel behind the float: a long, shallow belly sagging between the tip and the float
    rope.rest = Math.max(rope.rest, this.lure.distanceTo(this.tip) * 1.03);
    rope.setEnd(this.lure.x, this.lure.y, this.lure.z);
    rope.step(h, true);
    this.bobber.copy(this.lure);
    this.hook.set(this.lure.x, this.lure.y - 0.05, this.lure.z);
    this.bendTarget = 0.05;
    const wp = this.water.sample(this.lure.x, this.lure.z, this.wp);
    const surface = wp.wet ? wp.y : wp.groundY;
    if (this.lure.y <= surface || this.stateTime > 6) {
      this.lure.y = surface;
      this.bobber.copy(this.lure);
      this.onWater = wp.wet && wp.depth > 0.12;
      if (this.onWater) {
        if (!this.suppressSplash && !this.muteWater) this.water.floatPlop(this.lure.x, this.lure.z, this.time, 0.75 + 0.5 * this.castPower);
        this.emit('splash', 'splash_small', this.lure, 0.6 + 0.4 * this.castPower).strength = 0.25 + 0.15 * this.castPower;
        this.events.send();
      } else {
        this.emit('land_ground', null, this.lure);
        this.events.send();
      }
      // it plops in rather than skids (the line behind it brakes it)
      this.bobberVel.set(this.lureVel.x * 0.03, 0, this.lureVel.z * 0.03);
      this.bobberSink = 0;
      this.bobberSinkVel = Math.max(-0.6, this.lureVel.y * 0.12) * -1;
      this.sinkTarget = REST_SINK;
      this.lastDipT = this.time;
      this.dipArmed = false;
      // slack: the line keeps falling after the float has landed and lies down on the water in loose curves (the
      // belly it had in the air, kinked by the line's own twist)
      rope.rest = this.bobber.distanceTo(this.tip) * 1.06 + 0.6;
      // the line behind it stops paying out with it (it ran on past the float and hooked back into it)
      rope.damp(0.2);
      rope.kink(0.45, this.castPower * 7.3 + 1.1);
      this.enterFloating(true);
    }
  }

  private enterFloating(newBite: boolean) {
    this.enter('floating');
    // the line falling the last bit onto the water and hanging from the tip settles through the air, not swinging
    this.rope.airNormalDamp = 0.94;
    this.settleT = 1.1;
    this.fed = 0;
    this.hooks.idle();
    if (newBite) this.planBite(1);
  }

  private planBite(scale: number) {
    if (!this.onWater) {
      this.biteTimer = Infinity;
      this.nibbleCount = 0;
      return;
    }
    this.attracted = false;
    const f = this.forceBite;
    if (f) {
      this.biteTimer = f.delay;
      this.nibbleCount = this.school ? 0 : Math.min(3, f.nibbles.length);
      for (let i = 0; i < this.nibbleCount; i++) this.nibbles[i] = f.delay - f.nibbles[i];
      return;
    }
    const wp = this.water.sample(this.bobber.x, this.bobber.z, this.wp);
    this.habitat = habitatAt(wp.speed, wp.depth);
    const [a, b] = HABITATS[this.habitat].bite;
    this.biteTimer = this.rng.range(a, b) * TUNE.biteScale * scale;
    // with the school the timer is patience: when it runs out a trout is sent to the bait; the twitch comes from it
    if (this.school) {
      this.nibbleCount = 0;
      return;
    }
    // a few nibbles before the take: timers counted down like biteTimer (fire when biteTimer passes them)
    this.nibbleCount = Math.floor(this.rng.range(0, 3.99));
    for (let i = 0; i < this.nibbleCount; i++) this.nibbles[i] = this.biteTimer * this.rng.range(0.12, 0.6);
  }

  /** the float riding the water, drifting with the current, held by the line */
  private stepBobber(h: number) {
    const wp = this.water.sample(this.bobber.x, this.bobber.z, this.wp, this.time);
    const b = this.bobber, v = this.bobberVel;
    if (this.onWater && wp.depth > 0.03) {
      const a = approach(1.6, h);
      v.x += (wp.flowX - v.x) * a;
      v.z += (wp.flowZ - v.z) * a;
      b.x += v.x * h;
      b.z += v.z * h;
      // buoyancy: a stiff, damped spring around the floating depth
      this.bobberSinkVel += (-(this.bobberSink - this.sinkTarget) * 160 - this.bobberSinkVel * 10) * h;
      this.bobberSink += this.bobberSinkVel * h;
      b.y = wp.y - this.bobberSink;
      // every plunge of the float pushes out another ring
      if (this.bobberSinkVel < -0.02) this.dipArmed = true;
      else if (this.dipArmed && this.bobberSinkVel > 0.1 && this.time - this.lastDipT > 0.14) {
        this.dipArmed = false;
        this.lastDipT = this.time;
        // a fish tugging it under pushes out bigger, longer rings than the float settling
        if (this.state === 'bite') this.train(b.x, b.z, clamp(this.bobberSinkVel * 0.009, 0.0012, 0.0045), 3.8, 0.05, 0.33, 0.026);
        else this.train(b.x, b.z, clamp(this.bobberSinkVel * 0.0055, 0.0006, 0.0028), 3.2, 0.034, 0.27, 0.022);
      }
    } else {
      v.multiplyScalar(1 - approach(8, h));
      b.x += v.x * h;
      b.z += v.z * h;
      b.y = Math.max(wp.groundY, wp.wet ? wp.y : -1e9) + 0.015;
      this.bobberSink = this.bobberSinkVel = 0;
    }
    // the line: never farther from the tip than its length (the float swings in toward the bank)
    const d = b.distanceTo(this.tip), L = this.rope.rest + 0.05;
    let tilt = 0;
    if (d > L) {
      _v.subVectors(this.tip, b);
      const dh = Math.hypot(_v.x, _v.z);
      if (dh > 1e-4) {
        const move = Math.min(dh, (d - L) * (d / dh));
        b.x += (_v.x / dh) * move;
        b.z += (_v.z / dh) * move;
        const out = (v.x * -_v.x + v.z * -_v.z) / dh;
        if (out > 0) {
          v.x += (_v.x / dh) * out;
          v.z += (_v.z / dh) * out;
        }
        tilt = clamp((d - L) * 4, 0, 0.5);
      }
    }
    _v.subVectors(this.tip, b);
    const dh = Math.max(1e-4, Math.hypot(_v.x, _v.z));
    this.bobberTilt.set((_v.x / dh) * tilt, (_v.z / dh) * tilt);
  }

  private stepFloating(h: number, castPressed: boolean) {
    const rope = this.rope;
    this.stepBobber(h);
    // the angler takes up slack as the float drifts in, and lets it draw a little line as it drifts off: the last few
    // metres keep lying on the water in loose curves
    const tidy = this.bobber.distanceTo(this.tip) * 1.05 + 0.55;
    if (rope.rest > tidy) rope.rest -= (rope.rest - tidy) * approach(0.8, h);
    else if (this.fed < FEED && !this.reelHeld) {
      const d = Math.min(FEED - this.fed, (tidy - rope.rest) * approach(0.8, h));
      rope.rest += d;
      this.fed += d;
      this.crankDelta -= d / 0.16;
    }
    if (this.onWater && !this.reelHeld && (this.settleT -= h) <= 0) {
      // now and then a faint ring off the float riding still water (timed by count, not the seeded stream the bites use)
      this.settleN++;
      this.settleT = 1.5 + 0.9 * (0.5 + 0.5 * Math.sin(this.settleN * 2.39996));
      this.train(this.bobber.x, this.bobber.z, 0.0011 + 0.0003 * Math.sin(this.settleN * 1.7), 3.2, 0.032, 0.25, 0.024);
    }
    this.reel(this.reelHeld, TUNE.retrieveSpeed);
    if (this.reelHeld) {
      rope.rest = Math.max(0.5, rope.rest - TUNE.retrieveSpeed * h);
      this.crankDelta += (TUNE.retrieveSpeed * h) / 0.16;
      if (rope.rest < 1.5 || this.bobber.distanceTo(this.tip) < 1.3) {
        this.reel(false, 0);
        this.rope.rest = this.bobber.distanceTo(this.tip);
        this.enter('idle');
        this.hooks.idle();
        return;
      }
    } else if (this.onWater) {
      const before = this.biteTimer;
      this.biteTimer -= h;
      for (let i = 0; i < this.nibbleCount; i++) {
        const n = this.nibbles[i];
        if (before > n && this.biteTimer <= n) this.nibble();
      }
      if (this.biteTimer <= 0) {
        if (!this.school) this.startBite();
        else if (!this.attracted) {
          // the nearest calm trout comes over, inspects the bait and takes it (schoolBite); none near: try again later
          // a forced bite (tests, photo poses) asks for the biggest trout of its species, from twice as far
          const fb = this.forceBite;
          if ((fb ? this.school.attract(TUNE.attractRange * 2, fb.species) : this.school.attract(TUNE.attractRange)) >= 0) {
            this.attracted = true;
            this.attractT = 0;
          } else this.biteTimer = 4 * TUNE.biteScale;
        } else if ((this.attractT += h) > 25) {
          // it lost interest (or the float drifted away from it): send another
          this.attracted = false;
          this.biteTimer = 3 * TUNE.biteScale;
        }
      }
    }
    if (castPressed && this.state === 'floating') {
      // struck with nothing on: the float jerks toward the angler, fish nearby are spooked
      this.hooks.playStrike?.();
      _v.subVectors(this.tip, this.bobber).setY(0).normalize();
      this.bobberVel.addScaledVector(_v, 0.9);
      rope.rest = Math.max(1.6, rope.rest - 0.3);
      if (this.onWater) this.biteTimer += 2.5 * TUNE.biteScale;
      this.ring(0.004, 1.5);
      this.emit('miss', null, this.bobber);
      this.events.send();
    }
    this.hook.set(this.bobber.x, this.bobber.y - (this.onWater ? Math.min(TUNE.leader, Math.max(0.05, this.wp.depth - 0.1)) : 0.02), this.bobber.z);
    rope.setEnd(this.bobber.x, this.bobber.y + 0.02, this.bobber.z);
    this.endPinned = true;
    rope.step(h, true);
    this.bendTarget = 0.035 + (this.bobber.distanceTo(this.tip) > rope.rest ? 0.08 : 0) + (this.reelHeld ? 0.06 : 0);
    this.bendDir.set(rope.x[8] - rope.x[0], rope.y[8] - rope.y[0], rope.z[8] - rope.z[0]);
  }

  private ring(amplitude: number, life: number, foam = 0) {
    if (this.muteWater) return;
    this.water.ring(this.bobber.x, this.bobber.z, this.time, amplitude, life, foam);
  }

  private nibble() {
    this.bobberSinkVel += 0.3;
    this.ring(0.0012, 1.4);
    this.train(this.bobber.x, this.bobber.z, 0.0018, 3.4, 0.042, 0.3, 0.025);
    this.emit('nibble', 'plop', this.bobber, 0.35).strength = 0.3;
    this.events.send();
  }

  private pickFish() {
    const f = this.fish;
    const fb = this.forceBite;
    let id: SpeciesId;
    let length: number;
    if (fb) {
      id = fb.species;
      length = fb.lengthCm;
    } else {
      const wp = this.water.sample(this.bobber.x, this.bobber.z, this.wp);
      this.habitat = habitatAt(wp.speed, wp.depth);
      const hab = HABITATS[this.habitat];
      const r = this.rng.next();
      let acc = 0;
      id = 'brown';
      for (let i = 0; i < SPECIES_IDS.length; i++) {
        acc += hab.weights[SPECIES_IDS[i]];
        if (r <= acc) {
          id = SPECIES_IDS[i];
          break;
        }
      }
      const s = SPECIES[id];
      const bonus = hab.sizeBonus[id] ?? 1;
      length = clamp(s.lengthCm.typical * bonus * Math.exp(this.rng.gauss() * 0.13), s.lengthCm.min, s.lengthCm.max);
    }
    const s = SPECIES[id];
    f.species = s;
    f.lengthCm = Math.round(length);
    f.weightKg = weightKg(s, f.lengthCm);
    f.scale = f.lengthCm / 100 / s.modelLength;
    f.sizePull = (f.lengthCm / s.lengthCm.typical) ** 1.5;
    const root = this.fishRoot(id);
    this.held = {
      root, species: id, name: s.name, lengthCm: f.lengthCm, weightKg: f.weightKg, scale: f.scale,
      holdOffset: s.holdOffset * f.scale, holdScale: s.holdScale,
    };
  }

  private startBite() {
    this.pickFish();
    this.hookWindow = this.fish.species.hookWindow;
    this.dipFloat();
  }

  private dipFloat() {
    // the fish mouths the bait: the float dips, bobs back half up, dips again (each plunge rings), then goes under
    this.sinkTarget = 0.024;
    this.bobberSinkVel += 0.42;
    this.twitchT = 0.2;
    this.biteDips = 0;
    this.ring(0.003, 2.2, 0.015);
    // the take's first crisp ring, ahead of the plunges' own
    this.train(this.bobber.x, this.bobber.z, 0.0038, 3.8, 0.055, 0.36, 0.03);
    this.enter('bite');
    this.emit('bite', 'bite_plip', this.bobber, 0.8).strength = 1;
    this.events.send();
  }

  /** a school trout closed its mouth on the bait (FishSchool.onBite) */
  schoolBite(e: FishSourceBite) {
    // aiming, reeling in or the float out of the water: it spits the bait on its own
    if (!this.school || this.state !== 'floating' || !this.onWater || this.reelHeld) return;
    const f = this.fish, s = SPECIES[e.species], sp = this.school.species(e.species);
    this.schoolFish = e.fishId;
    f.species = s;
    f.lengthCm = Math.round(e.lengthM * 100);
    f.weightKg = e.weightKg;
    f.scale = e.lengthM / sp.modelLength;
    f.sizePull = (f.lengthCm / s.lengthCm.typical) ** 1.5;
    this.held = {
      root: this.fishRoot(e.species), species: e.species, name: s.name, lengthCm: f.lengthCm, weightKg: f.weightKg, scale: f.scale,
      holdOffset: sp.holdOffset * f.scale, holdScale: sp.holdScale,
    };
    // strike inside the trout's own window (it spits the bait when that runs out)
    this.hookWindow = Math.max(0.3, e.window - 0.05);
    this.dipFloat();
  }

  /** a school trout goes for the bait: the float trembles as it noses in (part 6's nibble) */
  schoolTake(_fishId: number) {
    if (this.state === 'floating' && this.onWater) this.nibble();
  }

  /** the school's trout got off: 'spat' (not struck in time) or 'threw' (slack line in the fight) */
  schoolLoose(fishId: number, why: 'spat' | 'threw') {
    if (fishId !== this.schoolFish) return;
    if (why === 'threw' && this.state === 'fight') this.threw();
    else if (why === 'spat' && this.state === 'bite') this.escapeBite();
  }

  private stepBite(h: number, castPressed: boolean) {
    const rope = this.rope;
    // the fish tugs the float down and shakes it: two scripted dips, then the take pulls it under
    if ((this.twitchT -= h) <= 0) {
      this.biteDips++;
      if (this.biteDips === 1) {
        this.twitchT = 0.32;
        this.sinkTarget = 0.026;
        this.bobberSinkVel += 0.3;
      } else {
        this.twitchT = this.rng.range(0.12, 0.3);
        this.sinkTarget = this.biteDips > 3 ? this.rng.range(0.06, 0.095) : this.rng.range(0.035, 0.07);
      }
      this.bobberVel.x += this.rng.range(-0.25, 0.25);
      this.bobberVel.z += this.rng.range(-0.25, 0.25);
    }
    this.stepBobber(h);
    this.hook.set(this.bobber.x, this.bobber.y - Math.min(TUNE.leader, Math.max(0.05, this.wp.depth - 0.1)), this.bobber.z);
    rope.setEnd(this.bobber.x, this.bobber.y + 0.02, this.bobber.z);
    rope.step(h, true);
    this.bendTarget = 0.06;
    if (castPressed) {
      if (!this.school) this.startFight();
      else if (this.schoolFish >= 0 && this.school.hook(this.schoolFish)) this.startFight();
      else this.escapeBite();
    } else if (this.stateTime > this.hookWindow) {
      this.escapeBite();
    }
  }

  /** the bite window passed (or the trout already spat the bait): it got away, the float rides on */
  private escapeBite() {
    this.sinkTarget = REST_SINK;
    this.ring(0.0015, 1.5);
    this.say('It got away');
    this.emit('escape', null, this.bobber);
    this.events.send();
    this.fish.mode = 'none';
    this.schoolFish = -1;
    this.enter('floating');
    this.hooks.idle();
    this.planBite(TUNE.spookFactor);
  }

  private startFight() {
    const f = this.fish;
    const wp = this.water.sample(this.bobber.x, this.bobber.z, this.wp);
    f.mode = 'none';
    f.stamina = 1;
    f.visible = true;
    f.placed = true;
    f.inAir = false;
    f.thrashCooldown = 2;
    f.jumpIn = -1;
    f.pos.set(this.bobber.x, wp.y - Math.min(0.5, Math.max(0.12, wp.depth * 0.5)), this.bobber.z);
    _v.subVectors(this.bobber, this.feet).setY(0).normalize();
    f.dir.copy(_v);
    f.vel.copy(_v).multiplyScalar(1.0);
    if (this.school) {
      // the school draws and moves it; part 6 keeps only the tackle's view of it (mouth, pull, stamina)
      f.mode = 'run';
      f.visible = false;
      this.lastThrashes = 0;
      this.wasAirborne = false;
      this.thrashSoundT = 0;
      this.turnHeading = Number.NaN;
      this.boilT = 0;
    } else this.chooseMode(true);
    this.tension = 0.4;
    this.strain = 0;
    this.slackTime = 0;
    this.sinkTarget = 0;
    this.hooks.playStrike?.();
    this.enter('fight');
    this.hooks.playReel(false, this.tension);
    this.emit('hook', null, this.bobber);
    this.events.send();
  }

  /** pick the fish's next move: run away, swing left/right across the current, hold deep, or give up */
  private chooseMode(first = false) {
    const f = this.fish, s = f.species, S = f.stamina, rng = this.rng;
    const vigor = 0.45 + 0.55 * S;
    const away = _w.subVectors(f.pos, this.feet).setY(0);
    if (away.lengthSq() < 1e-6) away.set(0, 0, -1);
    away.normalize();
    let angle = 0;
    if (S < 0.16) {
      // spent: it hangs across the current near the surface; only reeling brings it in
      f.mode = 'tired';
      f.modeTime = rng.range(1.5, 3);
      f.pull = 0.12 * s.strength;
      f.speed = 0.3;
      f.depth = 0.25;
      if (rng.next() < 0.6) f.side = -f.side;
      angle = f.side * rng.range(1.3, 1.9);
    } else {
      const r = first ? 0 : rng.next();
      if (r < 0.12 + 0.3 * S) {
        f.mode = 'run';
        f.modeTime = rng.range(0.9, 2.2) * (0.7 + 0.3 * s.endurance);
        f.pull = rng.range(0.9, 1.25) * s.strength * f.sizePull * vigor;
        f.speed = 1.8 * s.speed * vigor;
        f.depth = rng.range(0.3, 0.6);
        angle = rng.range(-0.6, 0.6);
        if (!first && rng.next() < s.jump * (0.6 + 0.4 * S)) f.jumpIn = rng.range(0.25, 0.6);
      } else if (r < 0.78) {
        // swing left or right: mostly alternating, sometimes the same side again
        f.mode = 'side';
        if (rng.next() < 0.75) f.side = -f.side;
        f.modeTime = rng.range(1.2, 3.0);
        f.pull = rng.range(0.42, 0.65) * s.strength * f.sizePull * vigor;
        f.speed = 1.05 * s.speed * vigor;
        f.depth = rng.range(0.35, 0.7);
        angle = f.side * rng.range(1.15, 1.9);
      } else {
        f.mode = 'hold';
        f.modeTime = rng.range(1.0, 2.4);
        f.pull = 0.35 * s.strength * f.sizePull * vigor;
        f.speed = 0.35;
        f.depth = rng.range(0.8, 1.6);
        angle = rng.range(-0.8, 0.8);
      }
    }
    const c = Math.cos(angle), sn = Math.sin(angle);
    f.dir.set(away.x * c - away.z * sn, 0, away.x * sn + away.z * c);
    this.keepDeep(f.dir, f.pos, false);
    // turning hard under the surface, it heaves up a boil
    const wt = this.water.sample(f.pos.x, f.pos.z, this.wq);
    if (!first && wt.wet && wt.y - f.pos.y < 0.8) {
      const k = clamp(1 - (wt.y - f.pos.y) / 0.8, 0.25, 1) * Math.min(1.3, f.sizePull);
      this.boil(f.pos.x, f.pos.z, 0.012 * k, 0.16 + 0.1 * k);
      this.train(f.pos.x, f.pos.z, 0.0014 * k, 2.2, 0.03, 0.28, 0.2);
    }
    // thrash at the surface close in
    if (!first && f.mode !== 'tired' && f.thrashCooldown <= 0 && this.tension > 0.45
      && f.pos.distanceTo(this.feet) < 7 && rng.next() < 0.35) {
      f.mode = 'thrash';
      f.thrashTime = 0;
      f.modeTime = 0.75;
      f.thrashCooldown = 4;
    }
  }

  /** steer a swim direction away from water too shallow for the fish (unless it is tired and being led in) */
  private keepDeep(dir: THREE.Vector3, p: THREE.Vector3, tired: boolean) {
    if (tired) return;
    const wq = this.water.sample(p.x + dir.x * 1.5, p.z + dir.z * 1.5, this.wq);
    if (wq.depth > 0.35) return;
    let best = -1, bx = dir.x, bz = dir.z;
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      const dx = Math.cos(a), dz = Math.sin(a);
      const d = this.water.sample(p.x + dx * 1.8, p.z + dz * 1.8, this.wq).depth;
      if (d > best) {
        best = d;
        bx = dx;
        bz = dz;
      }
    }
    dir.set(bx, 0, bz);
  }

  private stepFight(h: number) {
    if (this.school) return this.stepFightSchool(h);
    const f = this.fish, rope = this.rope, s = f.species;
    f.modeTime -= h;
    f.thrashCooldown -= h;
    if (f.jumpIn > 0 && (f.jumpIn -= h) <= 0 && f.mode === 'run') this.startJump();
    if (f.modeTime <= 0 && f.mode !== 'jump') this.chooseMode();
    if (this.forceThrashAt > 0 && this.stateTime >= this.forceThrashAt && this.stateTime - h < this.forceThrashAt && f.mode !== 'jump') {
      f.mode = 'thrash';
      f.thrashTime = 0;
      f.modeTime = 0.9;
      f.thrashCooldown = 4;
    }

    const wp = this.water.sample(f.pos.x, f.pos.z, this.wp, this.time);
    if (f.mode === 'jump') {
      f.vel.y -= G * h;
      f.pos.addScaledVector(f.vel, h);
      if (f.vel.y < 0 && f.pos.y < wp.y) {
        f.inAir = false;
        f.vel.y *= 0.15;
        this.splash(f.pos.x, f.pos.z, 0.9 + 0.2 * f.sizePull);
        this.emit('jump', 'splash_big', f.pos, 0.7 + 0.3 * Math.min(1, f.sizePull)).strength = 0.9 + 0.2 * f.sizePull;
        this.events.send();
        f.mode = 'side';
        f.modeTime = 1.2;
      }
    } else {
      // swim toward the chosen direction, at the chosen depth, out of the shallows
      _v.copy(f.dir).multiplyScalar(f.speed);
      const a = approach(3, h);
      f.vel.x += (_v.x - f.vel.x) * a;
      f.vel.z += (_v.z - f.vel.z) * a;
      const nx = f.pos.x + f.vel.x * h, nz = f.pos.z + f.vel.z * h;
      const wn = this.water.sample(nx, nz, this.wq);
      if (wn.depth > 0.16 || wn.depth > wp.depth) {
        f.pos.x = nx;
        f.pos.z = nz;
      } else {
        f.vel.x *= 0.5;
        f.vel.z *= 0.5;
        f.modeTime = Math.min(f.modeTime, 0.1);
      }
      let depth = f.depth;
      if (f.mode === 'thrash') {
        const start = f.thrashTime === 0;
        f.thrashTime += h;
        depth = 0.03;
        const k = Math.floor(f.thrashTime / 0.28), k0 = Math.floor((f.thrashTime - h) / 0.28);
        if (start) {
          this.splash(f.pos.x, f.pos.z, 0.42);
          this.boil(f.pos.x, f.pos.z, 0.016, 0.26);
          this.emit('thrash', 'fish_thrash', f.pos, 0.6 + 0.4 * Math.min(1, f.sizePull));
          this.events.send();
        } else if (k !== k0 && k < 3) {
          // each slap of the tail: a splash, a swirl and a fan of spray off it
          this.splash(f.pos.x, f.pos.z, 0.36);
          _w.set(0, 0, -0.45 * f.lengthCm / 100).applyQuaternion(f.quat).add(f.pos);
          this.swirl(_w.x, _w.z, 0.014, 0.12);
          if (!this.muteWater) this.water.spray(_w.x, wp.y, _w.z, -f.vel.z, f.vel.x, 14, 1.9, 0.007, this.seed());
        }
      }
      const ty = clamp(wp.y - Math.min(depth, Math.max(0.05, wp.depth - 0.08)), wp.groundY + 0.06, wp.y - 0.02);
      f.vel.y = (ty - f.pos.y) * 3;
      f.pos.y += f.vel.y * h;
    }

    // the line holds the fish on a sphere around the tip; tension comes from how hard it swims away
    const reeling = this.reelHeld;
    const total = rope.rest + 0.6;
    _v.subVectors(f.pos, this.tip);
    const dist = Math.max(1e-3, _v.length());
    _v.multiplyScalar(1 / dist);
    const over = dist - total;
    if (over > 0) {
      f.pos.addScaledVector(_v, -over);
      const radial = f.vel.dot(_v);
      if (radial > 0) f.vel.addScaledVector(_v, -radial);
    }
    const swimAway = Math.max(0, f.dir.x * _v.x + f.dir.z * _v.z);
    const pullOut = f.mode === 'jump' ? f.pull * 0.6 : f.pull * (0.25 + 0.75 * swimAway);
    const tight = smoothstep(-0.5, 0, over);
    const target = tight * (pullOut + (reeling ? 0.26 + 0.14 * f.stamina : 0));
    this.tension += (target - this.tension) * approach(9, h);

    const dragLimit = reeling ? TUNE.dragReeling : TUNE.dragFree;
    let slip = 0;
    if (this.tension > dragLimit) {
      slip = (this.tension - dragLimit) * 4.5;
      rope.rest += slip * h;
      this.tension -= (this.tension - dragLimit) * approach(6, h);
      this.crankDelta -= (slip * h) / 0.16;
    }
    this.dragging = slip > 0.04;
    if (this.dragging && (this.dragSoundT -= h) <= 0) {
      this.dragSoundT = 0.8;
      this.emit('drag', 'reel_drag', this.tip, clamp(slip / 1.5, 0.3, 1));
      this.events.send();
    } else if (!this.dragging) {
      this.dragSoundT = 0;
    }
    let rate = 0;
    if (reeling && this.tension < dragLimit) {
      rate = TUNE.reelSpeed * (1 - 0.65 * Math.min(1, this.tension)) * (f.mode === 'tired' ? 1.15 : 1);
      rope.rest = Math.max(0.8, rope.rest - rate * h);
      this.crankDelta += (rate * h) / 0.16;
    }
    this.reel(reeling, rate);

    if (this.tension > TUNE.breakTension) this.strain += ((this.tension - TUNE.breakTension) * 2 + 0.6) * h;
    else this.strain = Math.max(0, this.strain - 0.5 * h);
    if (this.tension < 0.05 && f.stamina > 0.25) this.slackTime += h;
    else this.slackTime = 0;
    f.stamina = Math.max(0, f.stamina - (h * (0.022 + 0.075 * Math.min(this.tension, 1.3) + (f.mode === 'run' ? 0.03 : 0)))
      / (s.endurance * (0.7 + 0.3 * f.sizePull)));

    this.orientFish(h, _v, tight);
    f.swimPhase += 2 * Math.PI * f.swimHz * h;
    // the float rides above the fish, a little toward the angler, dragged under when the line is tight
    _w.subVectors(this.tip, f.pos).setY(0);
    const wl = Math.max(1e-3, _w.length());
    this.bobber.set(f.pos.x + (_w.x / wl) * 0.35, 0, f.pos.z + (_w.z / wl) * 0.35);
    const wb = this.water.sample(this.bobber.x, this.bobber.z, this.wq, this.time);
    this.bobber.y = (wb.wet ? wb.y : wb.groundY) - clamp(this.tension * 0.05, 0, 0.05);
    this.bobberTilt.set((_w.x / wl) * clamp(this.tension, 0, 0.7), (_w.z / wl) * clamp(this.tension, 0, 0.7));
    this.mouth(this.hook);
    this.fightWater(h, wp);
    rope.setEnd(this.bobber.x, this.bobber.y + 0.02, this.bobber.z);
    // the drawn line follows the load: straight once it pulls, a little sag when it goes slack
    const span = this.bobber.distanceTo(this.tip);
    rope.drawRest = span * (1 + 0.05 * (1 - smoothstep(0.05, 0.5, this.tension)));
    rope.step(h, true);
    this.bendTarget = 0.2 + 0.95 * smoothstep(0, 0.95, this.tension);
    this.bendDir.set(rope.x[8] - rope.x[0], rope.y[8] - rope.y[0], rope.z[8] - rope.z[0]);

    if (this.strain >= 1) return this.snap();
    if (this.slackTime > 3.5 && !this.school) return this.threw();
    const toFeet = Math.hypot(f.pos.x - this.feet.x, f.pos.z - this.feet.z);
    const beached = wp.depth < 0.12 && toFeet < TUNE.landRadius * 2.2;
    if (f.mode !== 'jump' && (toFeet < TUNE.landRadius || beached || rope.rest < 1.3)) this.startLanding();
  }

  /**
   * The water round a hooked fish: a V of rings where the tight line hisses through the surface as it is dragged
   * across, a fan of spray when it cuts fast, swirls behind the tail strokes when the fish is shallow, a scatter of
   * small rings over it. (Boils come from its turns, splashes from thrashing and jumps.)
   */
  private fightWater(h: number, wp: WaterPoint) {
    const f = this.fish, b = this.bobber;
    const vx = (b.x - this.lineIn.x) / h, vz = (b.z - this.lineIn.z) / h;
    const first = this.stateTime <= h * 1.5;
    this.lineIn.copy(b);
    if (this.muteWater || first || !wp.wet) return;
    const sp = Math.min(3, Math.hypot(vx, vz));
    const tight = clamp((this.tension - 0.15) / 0.6, 0, 1);
    if ((this.wakeT -= h) <= 0 && sp > 0.15 && tight > 0) {
      this.wakeT = 0.085;
      this.train(b.x, b.z, clamp(0.0008 + sp * tight * 0.0016, 0, 0.003), 0.9, 0.028, 0.24, 0.01);
    }
    if ((this.sprayT -= h) <= 0 && sp > 0.35 && this.tension > 0.4) {
      this.sprayT = this.rng.range(0.06, 0.12);
      this.water.spray(b.x, wp.y, b.z, vx, vz, 3 + Math.round(sp * 3), 0.7 + sp * 0.5, 0.0045, this.seed());
    }
    const depth = wp.y - f.pos.y;
    if ((this.swirlT -= h) <= 0 && depth < 0.65 && f.mode !== 'jump') {
      this.swirlT = this.rng.range(0.4, 0.85) / (0.6 + 0.6 * f.stamina);
      const L = f.lengthCm / 100, k = clamp(1 - depth / 0.65, 0.2, 1);
      _w.set(0, 0, -0.55 * L).applyQuaternion(f.quat).add(f.pos);
      this.swirl(_w.x, _w.z, (0.005 + 0.009 * Math.min(1.3, f.sizePull)) * k, 0.07 + 0.08 * L);
      this.train(_w.x, _w.z, 0.0011 * k + 0.0003, 1.8, 0.026, 0.26, 0.05);
    }
    if ((this.scatterT -= h) <= 0) {
      this.scatterT = this.rng.range(0.18, 0.45);
      const a = this.rng.range(0, Math.PI * 2), r = this.rng.range(0.1, 0.65);
      this.train(f.pos.x + Math.cos(a) * r, f.pos.z + Math.sin(a) * r, this.rng.range(0.0004, 0.0011), 1.5, 0.02, 0.22, 0.008);
    }
  }

  /**
   * One slap of a school trout thrashing on top (part 7 counts them and throws its own splash and spray): a dark boil
   * where the body rolls, a swirl off the tail, and wide trains of rings that break up the reflections.
   */
  private thrashWater(vel: THREE.Vector3) {
    if (this.muteWater) return;
    const f = this.fish, L = f.lengthCm / 100;
    const wp = this.water.sample(f.pos.x, f.pos.z, this.wq);
    if (!wp.wet) return;
    let hx = vel.x, hz = vel.z;
    const hl = Math.hypot(hx, hz);
    if (hl > 1e-3) {
      hx /= hl;
      hz /= hl;
    } else {
      hx = f.dir.x;
      hz = f.dir.z;
    }
    const k = Math.min(1.3, f.sizePull);
    // the body behind the mouth, the tail at its end
    const cx = f.pos.x - hx * 0.45 * L, cz = f.pos.z - hz * 0.45 * L;
    const tx = f.pos.x - hx * 0.95 * L, tz = f.pos.z - hz * 0.95 * L;
    this.boil(cx, cz, 0.012 + 0.008 * k, 0.2 + 0.35 * L, 2.4);
    this.swirl(tx, tz, 0.012 + 0.006 * k, 0.1 + 0.12 * L);
    // rings running out for metres: a long, fast train that sets the reflections wobbling
    this.train(cx, cz, 0.0026 + 0.0014 * k, 3.2, 0.05, 0.34, 0.12 + 0.2 * L);
    this.train(cx, cz, 0.003 + 0.0015 * k, 4.5, 0.15, 0.62, 0.15 + 0.2 * L);
    // the slap itself (white water, drops of every size and clumps off the tail, the head's flick) is the school's
    // own splash (part 7, FishSchool 'slap'), thrown with the same beat
  }

  /** a hard turn by the school trout close under the surface wells up as a dark boil */
  private turnBoil(h: number, vel: THREE.Vector3) {
    const f = this.fish;
    this.boilT -= h;
    if (Math.hypot(vel.x, vel.z) < 0.25) return;
    const hd = Math.atan2(vel.x, vel.z), prev = this.turnHeading;
    this.turnHeading = hd;
    if (Number.isNaN(prev) || this.boilT > 0) return;
    const rate = Math.abs(Math.atan2(Math.sin(hd - prev), Math.cos(hd - prev))) / h;
    const wp = this.water.sample(f.pos.x, f.pos.z, this.wq);
    const depth = wp.y - f.pos.y;
    if (!wp.wet || rate < 2.6 || depth > 0.7) return;
    this.boilT = 0.45;
    this.boil(f.pos.x, f.pos.z, (0.006 + 0.01 * Math.min(1.3, f.sizePull)) * (1 - depth / 0.7), 0.18 + 0.2 * (f.lengthCm / 100), 2);
  }

  /**
   * The fight with a school trout: it swims and pulls (part 7: runs, shakes, leaps, sulks, stamina); the tackle turns
   * its pull into tension (pull / TUNE.lineNewtons, the share along the line), slips the drag, strains and snaps the
   * line and reels as with part 6's own fish, and hands the line's pull back in newtons (plus a stiff spring for any
   * length beyond the line, so the trout stays on part 6's sphere round the tip).
   */
  private stepFightSchool(h: number) {
    const sc = this.school!;
    // the fight's photograph: the trout is sent up to thrash on top in time (it needs ~a second to get there)
    const thrashAt = this.forceThrashAt - 0.85;
    if (this.forceThrashAt > 0 && this.stateTime >= thrashAt && this.stateTime - h < thrashAt) sc.thrash?.(this.schoolFish, 2.2);
    const st = sc.fight(this.schoolFish);
    if (!st) return this.threw();
    const f = this.fish, rope = this.rope;
    f.pos.copy(st.mouth);
    f.vel.copy(st.velocity);
    f.stamina = st.stamina;
    f.mode = st.airborne ? 'jump' : 'run';
    _fwd.set(st.velocity.x, 0, st.velocity.z);
    if (_fwd.lengthSq() > 1e-4) this.setFishQuat(_fwd.normalize(), 0);
    const reeling = this.reelHeld;
    _v.subVectors(f.pos, this.tip);
    const dist = Math.max(1e-3, _v.length());
    _v.multiplyScalar(1 / dist);
    const over = dist - (rope.rest + 0.6);
    const away = Math.max(0, st.pull.x * _v.x + st.pull.y * _v.y + st.pull.z * _v.z);
    // newtons to tension: linear for a typical trout, saturating at part 6's hardest runs (~1.25) for the big browns
    // (the rod's bend soaks up the surges), so the drag keeps a line that is not being reeled below the break
    const force = st.airborne ? st.maxForce * 0.3 : st.pullForce * (0.25 + 0.75 * away);
    const pullOut = 1.25 * Math.tanh(force / (1.25 * TUNE.lineNewtons));
    const tight = smoothstep(-0.5, 0, over);
    const target = tight * (pullOut + (reeling ? 0.26 + 0.14 * st.stamina : 0));
    this.tension += (target - this.tension) * approach(9, h);

    const dragLimit = reeling ? TUNE.dragReeling : TUNE.dragFree;
    let slip = 0;
    if (this.tension > dragLimit) {
      slip = (this.tension - dragLimit) * 4.5;
      rope.rest += slip * h;
      this.tension -= (this.tension - dragLimit) * approach(6, h);
      this.crankDelta -= (slip * h) / 0.16;
    }
    this.dragging = slip > 0.04;
    if (this.dragging && (this.dragSoundT -= h) <= 0) {
      this.dragSoundT = 0.8;
      this.emit('drag', 'reel_drag', this.tip, clamp(slip / 1.5, 0.3, 1));
      this.events.send();
    } else if (!this.dragging) {
      this.dragSoundT = 0;
    }
    let rate = 0;
    if (reeling && this.tension < dragLimit) {
      rate = TUNE.reelSpeed * (1 - 0.65 * Math.min(1, this.tension)) * (st.phase === 'tired' ? 1.15 : 1);
      rope.rest = Math.max(0.8, rope.rest - rate * h);
      this.crankDelta += (rate * h) / 0.16;
    }
    this.reel(reeling, rate);
    if (this.tension > TUNE.breakTension) this.strain += ((this.tension - TUNE.breakTension) * 2 + 0.6) * h;
    else this.strain = Math.max(0, this.strain - 0.5 * h);
    sc.setLine(this.schoolFish, { tip: this.tip, tension: this.tension * TUNE.lineNewtons + TUNE.lineStiffness * Math.max(0, over) });

    // its leaps and surface thrashing make their own splashes in the water (part 7); here only their sounds
    if (this.wasAirborne && !st.airborne) {
      this.emit('jump', 'splash_big', f.pos, 0.7 + 0.3 * Math.min(1, f.sizePull)).strength = 0.9 + 0.2 * f.sizePull;
      this.events.send();
    }
    this.wasAirborne = st.airborne;
    this.thrashSoundT -= h;
    if (st.thrashes !== this.lastThrashes) {
      this.lastThrashes = st.thrashes;
      this.thrashWater(st.velocity);
      if (this.thrashSoundT <= 0) {
        this.thrashSoundT = 0.8;
        this.emit('thrash', 'fish_thrash', f.pos, 0.6 + 0.4 * Math.min(1, f.sizePull));
        this.events.send();
      }
    }
    // a hard turn close under the surface wells up as a dark boil
    this.turnBoil(h, st.velocity);

    // the float rides above the fish, a little toward the angler, dragged under when the line is tight
    _w.subVectors(this.tip, f.pos).setY(0);
    const wl = Math.max(1e-3, _w.length());
    this.bobber.set(f.pos.x + (_w.x / wl) * 0.35, 0, f.pos.z + (_w.z / wl) * 0.35);
    const wb = this.water.sample(this.bobber.x, this.bobber.z, this.wq, this.time);
    this.bobber.y = (wb.wet ? wb.y : wb.groundY) - clamp(this.tension * 0.05, 0, 0.05);
    this.bobberTilt.set((_w.x / wl) * clamp(this.tension, 0, 0.7), (_w.z / wl) * clamp(this.tension, 0, 0.7));
    this.hook.copy(f.pos);
    const wp = this.water.sample(f.pos.x, f.pos.z, this.wp);
    this.fightWater(h, wp);
    rope.setEnd(this.bobber.x, this.bobber.y + 0.02, this.bobber.z);
    const span = this.bobber.distanceTo(this.tip);
    rope.drawRest = span * (1 + 0.05 * (1 - smoothstep(0.05, 0.5, this.tension)));
    rope.step(h, true);
    this.bendTarget = 0.2 + 0.95 * smoothstep(0, 0.95, this.tension);
    this.bendDir.set(rope.x[8] - rope.x[0], rope.y[8] - rope.y[0], rope.z[8] - rope.z[0]);

    if (this.strain >= 1) return this.snap();
    const toFeet = Math.hypot(f.pos.x - this.feet.x, f.pos.z - this.feet.z);
    const beached = wp.depth < 0.12 && toFeet < TUNE.landRadius * 2.2;
    if (!st.airborne && (st.landable || beached || rope.rest < 1.3)) this.startLanding();
  }

  private startJump() {
    const f = this.fish;
    const wp = this.water.sample(f.pos.x, f.pos.z, this.wp);
    if (wp.depth < 0.4) return;
    f.mode = 'jump';
    f.inAir = true;
    f.pos.y = wp.y - 0.05;
    f.vel.set(f.dir.x * 1.4, this.rng.range(2.6, 3.4) / Math.sqrt(Math.max(0.6, f.sizePull)), f.dir.z * 1.4);
    f.roll = 0;
    this.splash(f.pos.x, f.pos.z, 0.5);
  }

  /** fish frame from its motion: nose along the swim, turned toward the line when it is led in */
  private orientFish(h: number, lineDir: THREE.Vector3, tight: number) {
    const f = this.fish;
    if (f.mode === 'jump') {
      _fwd.copy(f.vel).normalize();
      f.roll += h * 9;
      f.swimHz = 7;
      f.swimAmp = 0.14;
    } else {
      _fwd.set(f.vel.x, 0, f.vel.z);
      if (_fwd.lengthSq() < 0.01) _fwd.copy(f.dir);
      _fwd.normalize();
      // turned toward the line when it is led in (horizontal only: a fish under the rod tip must not stand on its tail)
      const toward = f.mode === 'tired' ? 0.4 * tight : 0.15 * tight;
      const lh = Math.hypot(lineDir.x, lineDir.z);
      if (lh > 1e-3) {
        _fwd.x += (-lineDir.x / lh - _fwd.x) * toward;
        _fwd.z += (-lineDir.z / lh - _fwd.z) * toward;
      }
      const hl = Math.hypot(_fwd.x, _fwd.z) || 1;
      const pitch = clamp(f.vel.y * 0.3, -0.3, 0.3);
      _fwd.set((_fwd.x / hl) * Math.cos(pitch), Math.sin(pitch), (_fwd.z / hl) * Math.cos(pitch));
      f.roll += (Math.sin(this.stateTime * 3.1) * 0.35 * tight - f.roll) * approach(4, h);
      const m = f.mode;
      f.swimHz = m === 'run' ? 5.5 : m === 'side' ? 3.5 : m === 'thrash' ? 8 : m === 'tired' ? 2.2 : 1.8;
      f.swimAmp = m === 'run' ? 0.12 : m === 'side' ? 0.14 : m === 'thrash' ? 0.16 : m === 'tired' ? 0.1 : 0.06;
    }
    this.setFishQuat(_fwd, f.roll);
  }

  private setFishQuat(fwd: THREE.Vector3, roll: number) {
    _m.lookAt(fwd, ZERO, Math.abs(fwd.y) > 0.98 ? _w.set(1, 0, 0) : UP);
    this.fish.quat.setFromRotationMatrix(_m);
    if (roll) this.fish.quat.multiply(_q.setFromAxisAngle(_w.set(0, 0, 1), roll));
  }

  /** the snout tip in world space (hook) */
  private mouth(out: THREE.Vector3) {
    const f = this.fish;
    return out.set(0, 0, f.species.noseZ * f.scale).applyQuaternion(f.quat).add(f.pos);
  }

  private reel(on: boolean, rate: number) {
    if (on && !this.reeling) {
      this.reeling = true;
      const e = this.emit('reel', 'reel_wind', this.tip, 0.8);
      e.loop = 'start';
      this.events.send();
    } else if (!on && this.reeling) {
      this.reeling = false;
      const e = this.emit('reel', 'reel_wind', this.tip, 0.8);
      e.loop = 'stop';
      this.events.send();
    }
    this.reelRate = on ? rate : 0;
  }

  private snap() {
    this.reel(false, 0);
    this.say('The line snapped');
    this.emit('snap', null, this.tip, 1);
    this.events.send();
    this.fish.mode = 'free';
    this.fish.freeTime = 0;
    this.fish.recover = false;
    // the school's trout bolts with the hook (part 7)
    if (this.school) {
      if (this.schoolFish >= 0) this.school.unhook(this.schoolFish);
      this.schoolFish = -1;
      this.fish.mode = 'none';
      this.fish.visible = false;
    }
    this.bobberVisible = false;
    this.rope.endInvMass = 1;
    this.endPinned = false;
    this.bendVel = -6;
    this.enter('snapped');
    this.hooks.idle();
  }

  private threw() {
    this.reel(false, 0);
    this.say('The fish threw the hook');
    this.emit('threw', null, this.fish.pos);
    this.events.send();
    this.fish.mode = 'free';
    this.fish.freeTime = 0;
    this.fish.recover = false;
    if (this.school) {
      if (this.schoolFish >= 0) this.school.unhook(this.schoolFish);
      this.schoolFish = -1;
      this.fish.mode = 'none';
      this.fish.visible = false;
    }
    this.onWater = this.water.sample(this.bobber.x, this.bobber.z, this.wp).wet;
    this.sinkTarget = REST_SINK;
    this.enter('floating');
    this.hooks.idle();
    this.planBite(TUNE.spookFactor);
  }

  private startLanding() {
    const f = this.fish;
    this.reel(false, 0);
    this.released = false;
    this.landFrom.copy(f.pos);
    this.landFromQ.copy(f.quat);
    const info = this.school && this.schoolFish >= 0 ? this.school.land(this.schoolFish) : null;
    if (info) {
      // lifted from where the school's trout is (its centre and attitude)
      f.pos.copy(info.position);
      f.quat.copy(info.quaternion);
      this.landFrom.copy(info.position);
      this.landFromQ.copy(info.quaternion);
    }
    f.mode = 'held';
    f.inAir = true;
    this.splash(f.pos.x, f.pos.z, 0.25);
    this.emit('landed', 'drip', f.pos, 0.7);
    this.events.send();
    this.enter('landing');
    const held = this.held!;
    this.hooks.playHold(held);
    const e = this.emit('hold_fish', null, f.pos);
    e.species = held.species;
    this.events.send();
  }

  private stepLanded(h: number) {
    const f = this.fish, rope = this.rope;
    f.placed = this.hooks.getFishHold(_v, _q);
    if (this.state === 'landing') {
      const k = easeInOut(clamp(this.stateTime / TUNE.landTime, 0, 1));
      if (f.placed) {
        f.pos.lerpVectors(this.landFrom, _v, k);
        f.pos.y += Math.sin(k * Math.PI) * 0.25;
        f.quat.slerpQuaternions(this.landFromQ, _q, k);
      }
      f.swimHz = 6;
      f.swimAmp = 0.12 * (1 - k) + 0.04;
      if (this.stateTime >= TUNE.landTime) this.startHolding();
    } else if (this.state === 'holding') {
      if (f.placed) {
        f.pos.copy(_v);
        f.quat.copy(_q);
      }
      // the fish flexes in the hands now and then; water runs off it and the hands
      if ((f.flexTime -= h) <= -this.rng.range(1.2, 2.4)) f.flexTime = 0.45;
      this.drips(h, 1);
      const flex = f.flexTime > 0;
      f.swimHz = flex ? 5 : 1.2;
      f.swimAmp = flex ? 0.1 : 0.015;
      if (this.stateTime >= TUNE.holdTime) {
        this.releaseSpot();
        this.letGo = Math.max(0, this.hooks.playRelease?.(this.held!) ?? 0.6);
        this.released = false;
        this.enter('releasing');
      }
    } else {
      if (!this.released) {
        if (f.placed) {
          f.pos.copy(_v);
          f.quat.copy(_q);
        }
        // held facing the current, its gills working; the wrists in the water push out rings, the tail kicks once
        // just before it is let go
        f.swimHz = 1.6;
        f.swimAmp = this.stateTime > this.letGo - 0.35 ? 0.11 : 0.03;
        this.drips(h, 0.4);
        this.wristRings(h);
        if (this.stateTime >= this.letGo - 0.3 && this.stateTime - h < this.letGo - 0.3) {
          _w.set(0, 0, -0.5 * f.lengthCm / 100).applyQuaternion(f.quat).add(f.pos);
          if (this.water.sample(_w.x, _w.z, this.wq).wet) {
            this.swirl(_w.x, _w.z, 0.007, 0.07);
            this.train(_w.x, _w.z, 0.0012, 2.2, 0.026, 0.26, 0.04);
          }
        }
        if (this.stateTime >= this.letGo) {
          this.released = true;
          f.mode = 'free';
          f.placed = true;
          f.recover = true;
          f.freeTime = 0;
          // it rests a moment in the water, then darts off diagonally away from the bank
          _w.subVectors(f.pos, this.feet).setY(0).normalize();
          const side = this.rng.sign() * 0.95;
          f.dir.set(_w.x * Math.cos(side) - _w.z * Math.sin(side), 0, _w.x * Math.sin(side) + _w.z * Math.cos(side));
          f.vel.set(0, -0.2, 0);
          if (this.water.sample(f.pos.x, f.pos.z, this.wq).wet) {
            this.boil(f.pos.x, f.pos.z, 0.006, 0.14, 1.8);
            this.train(f.pos.x, f.pos.z, 0.0012, 2.4, 0.03, 0.28, 0.12);
          }
          this.emit('release', null, f.pos);
          this.events.send();
        }
      } else if (this.stateTime >= this.letGo + 2.2) {
        // (he gets up off his knee first)
        this.enter('idle');
        this.hooks.idle();
        return;
      }
    }
    f.swimPhase += 2 * Math.PI * f.swimHz * h;
    // out of the water its weight sags the body between the hands; in the water it floats level again
    const sagTo = this.state === 'releasing' ? 0 : this.state === 'holding' ? 0.085 : 0.085 * clamp(this.stateTime / TUNE.landTime, 0, 1);
    f.sag += (sagTo - f.sag) * approach(5, h);
    if (this.school && this.schoolFish >= 0) {
      if (!this.released) {
        const k = this.state === 'landing' ? easeInOut(clamp(this.stateTime / TUNE.landTime, 0, 1)) : 1;
        this.school.setHeld(this.schoolFish, f.pos, f.quat, 1 + ((this.held?.holdScale ?? 1) - 1) * k, f.sag);
      } else {
        // let go: back into the school, in the water nearest the hands toward where it came from (let go from the
        // hands under the surface it only kicks off: the boil and the rings above are its splash)
        const at = this.findReleasePoint(this.schoolReleaseAt);
        this.school.release(this.schoolFish, { x: at.x, z: at.z });
        if (at.distanceToSquared(f.pos) > 1e-6) this.splash(at.x, at.z, 0.3);
        this.emit('splash', 'splash_small', at, 0.5).strength = 0.3;
        this.events.send();
        this.schoolFish = -1;
        f.mode = 'none';
        f.visible = false;
      }
    }
    // line to the fish's mouth while it is shown in the hands (hanging in a loose curve from the rod), unhooked for
    // the release: the float then dangles from the tip on slack line
    if (this.state !== 'releasing' && !this.released) {
      this.mouth(this.hook);
      rope.rest += (this.hook.distanceTo(this.tip) + 0.5 - rope.rest) * approach(6, h);
      rope.setEnd(this.hook.x, this.hook.y, this.hook.z);
      // the float is the heaviest thing on the line: it hangs in a soft droop between the tip and the mouth
      const k = Math.max(1, Math.round(rope.n * (1 - TUNE.leader / Math.max(rope.rest, 1))) - 1);
      rope.beadIndex = k;
      rope.beadInvMass = 0.08;
      rope.airNormalDamp = 0.95;
      rope.step(h, true);
      this.endPinned = true;
      this.bobber.set(rope.x[k], rope.y[k], rope.z[k]);
    } else {
      rope.beadIndex = -1;
      this.stepHanging(h);
    }
    this.bendTarget = this.state === 'landing' ? 0.18 : 0.02;
    this.bendDir.set(rope.x[8] - rope.x[0], rope.y[8] - rope.y[0], rope.z[8] - rope.z[0]);
  }

  /**
   * Where the fish goes back: in front of him (his facing), the nearest point within reach with water deep enough to
   * hold it under, across in front of him with its head upstream. The character puts its hands there (on the stones
   * if there is no water).
   */
  private releaseSpot() {
    // in front of his right knee (he kneels on the right one): out of the shadow of his head and hat
    const d = this.aimDir, f = this.feet, wq = this.wq;
    const sx = -d.z * 0.2, sz = d.x * 0.2;
    let r = 0.36;
    for (let k = 0.26; k <= 0.7; k += 0.04) {
      if (this.water.sample(f.x + d.x * k + sx, f.z + d.z * k + sz, wq).wet && wq.depth > 0.15) {
        r = k;
        break;
      }
    }
    const px = f.x + d.x * r + sx, pz = f.z + d.z * r + sz;
    const wp = this.water.sample(px, pz, wq);
    const inWater = wp.wet && wp.depth > 0.15;
    this.releaseAt.set(px, inWater ? wp.y - 0.055 : Math.max(wp.groundY, wp.wet ? wp.y : -1e9) + 0.05, pz);
    // held across in front of him (parallel to the bank as he faces the water), its head to the upstream side: a
    // trout pointed straight upstream lay along his line of sight whenever he faced down or up the river, end-on to
    // anyone in front of him and under his forearms
    const fl = Math.hypot(wp.flowX, wp.flowZ);
    const ux = fl > 0.02 ? -wp.flowX / fl : -1, uz = fl > 0.02 ? -wp.flowZ / fl : 0;
    const side = -d.z * ux + d.x * uz >= 0 ? 1 : -1;
    this.releaseDir.set(-d.z * side, 0, d.x * side);
    this.hooks.releaseInto?.(this.releaseAt, this.releaseDir, this.fish.lengthCm / 100);
  }

  /** water dripping off the held fish (and the hands under it), `rate` 1 = a fish just lifted out */
  private drips(h: number, rate: number) {
    const f = this.fish;
    if (this.muteWater || (this.dripT -= h) > 0) return;
    this.dripT = this.rng.range(0.04, 0.13) / rate;
    const L = f.lengthCm / 100;
    _w.set(this.rng.range(-0.01, 0.01), -0.045 * L / 0.47, this.rng.range(-0.48, 0.42) * L).applyQuaternion(f.quat).add(f.pos);
    this.water.drip(_w.x, _w.y, _w.z, this.seed());
  }

  /**
   * rings from the wrists where the hands hold the fish in the water (at its head and tail), and the small pillow of
   * water each forearm pushes up on its upstream side where it goes in (part 7)
   */
  private wristRings(h: number) {
    const f = this.fish;
    if ((this.pillowT -= h) <= 0 && !this.muteWater) {
      this.pillowT = 0.14;
      const L = f.lengthCm / 100;
      for (const u of [0.32, -0.36]) {
        // the forearm comes up out of the water a little toward him from the hand under the fish
        _w.set(0, 0, u * L).applyQuaternion(f.quat).add(f.pos);
        const tx = this.feet.x - _w.x, tz = this.feet.z - _w.z, tl = Math.hypot(tx, tz) || 1;
        _w.x += (tx / tl) * 0.07;
        _w.z += (tz / tl) * 0.07;
        const wp = this.water.sample(_w.x, _w.z, this.wq);
        if (!wp.wet || wp.y <= f.pos.y) continue;
        const fl = Math.hypot(wp.flowX, wp.flowZ);
        const ux = fl > 0.02 ? -wp.flowX / fl : 0, uz = fl > 0.02 ? -wp.flowZ / fl : 0;
        this.water.boil(_w.x + ux * 0.05, _w.z + uz * 0.05, this.time, 0.0035, 0.03, 0.5, 0.01);
      }
    }
    if ((this.wristT -= h) > 0) return;
    this.wristT = this.rng.range(0.3, 0.55);
    const L = f.lengthCm / 100;
    for (const u of [0.32, -0.36]) {
      _w.set(this.rng.range(-0.03, 0.03), 0, u * L).applyQuaternion(f.quat).add(f.pos);
      const wp = this.water.sample(_w.x, _w.z, this.wq);
      if (wp.wet && wp.y > f.pos.y) this.train(_w.x, _w.z, this.rng.range(0.0011, 0.0017), 2.6, 0.032, 0.26, 0.035);
    }
  }

  /** the first point from the hands toward where the trout was lifted from with water over 12 cm (its release spot) */
  private findReleasePoint(out: THREE.Vector3) {
    const f = this.fish;
    for (let i = 0; i <= 16; i++) {
      out.lerpVectors(f.pos, this.landFrom, i / 16);
      if (this.water.sample(out.x, out.z, this.wq).depth > 0.12) return out;
    }
    return out.copy(this.landFrom);
  }

  private startHolding() {
    const f = this.fish;
    this.caught++;
    const c: CaughtFish = { species: f.species.id, name: f.species.name, lengthCm: f.lengthCm, weightKg: f.weightKg };
    this.lastRecord = !this.biggest || c.lengthCm > this.biggest.lengthCm;
    if (this.lastRecord) this.biggest = c;
    this.last = c;
    f.flexTime = 0.4;
    this.enter('holding');
    const e = this.emit('catch', null, f.pos);
    e.record = this.lastRecord;
    this.events.send();
  }

  /** a released (or escaped) fish swims off and fades out */
  private stepFreeFish(h: number) {
    const f = this.fish;
    f.freeTime += h;
    const wp = this.water.sample(f.pos.x, f.pos.z, this.wp);
    if (f.pos.y > wp.y + 0.01 && f.inAir) {
      f.vel.y -= G * h;
      f.pos.addScaledVector(f.vel, h);
      if (f.pos.y <= Math.max(wp.y, wp.groundY + 0.04)) {
        f.inAir = false;
        f.vel.multiplyScalar(0.4);
        if (this.state === 'releasing' || this.state === 'idle') {
          this.splash(f.pos.x, f.pos.z, 0.3);
          this.emit('splash', 'splash_small', f.pos, 0.5).strength = 0.3;
          this.events.send();
        }
      }
    } else {
      f.inAir = false;
      // released fish recover for half a second before the dash; escaped ones go at once
      const rest = f.recover && f.freeTime < 0.5;
      const burst = !rest && f.freeTime < 1.3;
      if (burst && wp.y - f.pos.y < 0.45 && (this.swirlT -= h) <= 0) {
        this.swirlT = 0.22;
        _w.set(0, 0, -0.5 * f.lengthCm / 100).applyQuaternion(f.quat).add(f.pos);
        this.swirl(_w.x, _w.z, 0.008, 0.08, 1.1);
      }
      this.keepDeep(f.dir, f.pos, false);
      _v.copy(f.dir).multiplyScalar(rest ? 0.05 : burst ? 1.7 : 0.9);
      const a = approach(burst ? 4 : 2, h);
      f.vel.x += (_v.x - f.vel.x) * a;
      f.vel.z += (_v.z - f.vel.z) * a;
      f.pos.x += f.vel.x * h;
      f.pos.z += f.vel.z * h;
      const ty = clamp(wp.y - Math.min(0.45, Math.max(0.05, wp.depth - 0.08)), wp.groundY + 0.05, wp.y - 0.03);
      f.pos.y += (ty - f.pos.y) * approach(2.5, h);
      // never under the gravel (released on a dry bar it slides back to the water)
      if (f.pos.y < wp.groundY + 0.04) f.pos.y = wp.groundY + 0.04;
      f.swimHz = rest ? 1.4 : burst ? 6 : 2.2;
      f.swimAmp = rest ? 0.05 : burst ? 0.12 : 0.08;
      _fwd.set(f.vel.x, 0, f.vel.z);
      if (rest) _fwd.copy(f.dir);
      if (_fwd.lengthSq() > 1e-4) {
        _q.copy(f.quat);
        this.setFishQuat(_fwd.normalize(), 0);
        f.quat.slerpQuaternions(_q, f.quat, approach(rest ? 3 : 8, h));
      }
    }
    if (this.state !== 'landing' && this.state !== 'holding' && this.state !== 'releasing') f.swimPhase += 2 * Math.PI * f.swimHz * h;
    if (f.freeTime > 3.4 || Math.hypot(f.pos.x - this.feet.x, f.pos.z - this.feet.z) > 12) {
      f.mode = 'none';
      f.visible = false;
    }
  }

  private stepRodBend(h: number) {
    this.bendVel += ((this.bendTarget - this.bend) * 160 - this.bendVel * 13) * h;
    this.bend = clamp(this.bend + this.bendVel * h, -0.35, 1.15);
  }

  /** per-frame outputs: prompt, aim preview, reel loop update, character fight pose */
  private frameOutputs() {
    const s = this.state;
    this.prompt = s === 'idle' ? 'cast'
      : s === 'aiming' ? 'release'
        : s === 'floating' ? (this.onWater ? 'wait' : 'reelin')
          : s === 'bite' ? 'hook'
            : s === 'fight' ? (this.tension > 0.92 || this.strain > 0.25 ? 'ease' : 'reel')
              : null;
    if (s === 'aiming') {
      this.predictAim();
      this.hooks.playAim?.(this.power);
    } else this.aimCount = 0;
    if (s === 'fight') this.hooks.playReel(this.reelHeld, this.tension);
    if (this.reeling) {
      const e = this.emit('reel', 'reel_wind', this.tip, 0.8);
      e.loop = 'update';
      e.strength = clamp(this.reelRate / TUNE.reelSpeed, 0, 1);
      e.rate = 0.85 + 0.35 * e.strength;
      this.events.send();
    }
  }

  /**
   * The flight the current power would give (same integrator as the real cast), drawn on the water: the points are
   * the path's shadow on the surface under it, every ~0.5 m, ending at the landing point.
   */
  private predictAim() {
    const p = _v.copy(this.aimOrigin), v = this.launchVelocity(this.power, _w);
    const h = 1 / 60;
    let n = 0, lastX = p.x, lastZ = p.z;
    this.aimOnWater = false;
    for (let i = 0; i < 480; i++) {
      this.integrateLure(p, v, h);
      const wp = this.water.sample(p.x, p.z, this.wq);
      const surface = wp.wet ? wp.y : wp.groundY;
      const landed = p.y <= surface;
      if (landed) p.y = surface;
      if ((Math.hypot(p.x - lastX, p.z - lastZ) >= 0.55 || landed) && n < AIM_POINTS) {
        this.aimPts[n * 3] = p.x;
        this.aimPts[n * 3 + 1] = surface + 0.03;
        this.aimPts[n * 3 + 2] = p.z;
        this.aimWet[n] = wp.wet && wp.depth > 0.05 ? 1 : 0;
        lastX = p.x;
        lastZ = p.z;
        n++;
      }
      if (landed) {
        this.aimOnWater = wp.wet && wp.depth > 0.12;
        break;
      }
    }
    this.aimCount = n;
    this.aimLanding.copy(p);
  }

  private hud = {
    state: 'idle' as FishingState, prompt: null as PromptKind, toast: null as string | null, power: 0, tension: 0,
    caught: 0, biggest: null as CaughtFish | null, last: null as CaughtFish | null, lastRecord: false,
  };

  /** HUD snapshot (the same object every frame) */
  hudState() {
    const h = this.hud;
    h.state = this.state;
    h.prompt = this.prompt;
    h.toast = this.toast;
    h.power = this.power;
    h.tension = this.tension;
    h.caught = this.caught;
    h.biggest = this.biggest;
    h.last = this.last;
    h.lastRecord = this.lastRecord;
    return h;
  }

  /** for the HUD and tests */
  get hookedSpecies(): SpeciesId | null {
    return this.state === 'bite' || this.fish.mode !== 'none' ? this.fish.species.id : null;
  }

  get heldFish(): HeldFish | null {
    return this.held;
  }
}
