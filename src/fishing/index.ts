import * as THREE from 'three/webgpu';
import type { Engine } from '../core/Engine';
import type { Assets } from '../core/Assets';
import type { World } from '../world/World';
import { shotMode } from '../core/params';
import { LAYER_NO_REFLECT } from '../core/layers';
import { FISH_ASSETS, STANDS, TUNE, fishParams } from './config';
import { fishingEvents, type FishingEvent } from './events';
import { FishingInput } from './FishingInput';
import { FishingSim, type ResolvedHooks } from './FishingSim';
import { FishingView } from './FishingView';
import { FishModels } from './FishModels';
import { SegmentRodRig, buildProceduralRod } from './RodRig';
import { StandInAngler } from './StandIn';
import { WaterProbe } from './WaterProbe';
import { ANGLER_POWER, ANGLER_STAND, POSE_CAMERAS, POSE_NAMES, POSE_SEED, POSE_SIDE, POSE_STOPS, RELEASE_STAND, ScriptedPlayer, type PoseAnchor, type PoseCamera, type PoseName } from './poses';
import { Rng } from './util';
import { eventA, eventB, eventC, eventsU } from '../water/surface';
import { SPECIES_IDS } from './species';
import { FishingHud } from '../ui/FishingHud';
import type { PoseAngler } from './angler';
import type { CharacterHooks, FishingButtons, FishingState, RodRig, SpeciesId } from './types';

export type { CharacterHooks, RodRig, HeldFish, FishingState, SpeciesId } from './types';
export { fishingEvents } from './events';
export type { FishingEvent, FishingEventType, SoundId } from './events';
export { connectFishingAudio, type FishingAudioSink, type FishingAudioPayloads } from './audioBridge';
export { SegmentRodRig, buildProceduralRod } from './RodRig';
export { anglerHooks, type PoseAngler } from './angler';
export { STANDS } from './config';

/** The Blender rod as a RodRig (procedural stand-in if fishing_rod.glb is missing). Parent `rig.root` to hand_r. */
export async function loadRodRig(assets: Assets): Promise<SegmentRodRig> {
  try {
    const gltf = await assets.gltf(FISH_ASSETS.rod);
    const rig = SegmentRodRig.fromGltf(gltf.scene.clone(true));
    if (rig) return rig;
  } catch (e) {
    console.warn(`[fishing] ${FISH_ASSETS.rod} not found, using the procedural rod`, e);
  }
  return buildProceduralRod();
}

export interface FishingOptions {
  engine: Engine;
  world: World;
  assets: Assets;
  /** the character (part 5); omitted: the camera stand-in angler */
  character?: CharacterHooks;
  /** the character's rod (e.g. loadRodRig(assets) parented to hand_r); required with `character` */
  rod?: RodRig;
  /** with `character`: placing and stepping him for the photo-mode poses (`fish-*` spots) */
  angler?: PoseAngler;
  /**
   * where the character's rod is (part 5's RodGround): only 'hand' fishes; otherwise the buttons do nothing and no
   * prompt shows, and while it is 'laid' on the ground the line and float are hidden
   */
  rodPlace?: () => 'hand' | 'busy' | 'laid';
}

const NO_BUTTONS: FishingButtons = { cast: false, castPressed: false, castReleased: false, reel: false, reelPressed: false, reelReleased: false };

/** turn rates on the spot (rad/s): to the camera's heading while aiming; toward the fish once it runs wide */
const TURN_AIM = 3.0;
const TURN_FIGHT = 0.8;
const FIGHT_SLACK = 0.45;

export interface FishingSystem {
  readonly sim: FishingSim;
  readonly view: FishingView;
  readonly events: typeof fishingEvents;
  readonly standIn: StandInAngler | null;
  readonly hud: FishingHud | null;
  /** one frame of the loop (registered as an engine updater by installFishing) */
  frame(dt: number, t: number, buttons?: FishingButtons): void;
  /** stop reading the mouse (menus, cutscenes) */
  setInputEnabled(on: boolean): void;
  /** fresh tackle (line reeled in at the tip) on the next frame, once the rod's tip is where the angler now is */
  resetTackle(): void;
  dispose(): void;
}

interface FishDebug {
  state: string;
  stateTime: number;
  tension: number;
  strain: number;
  power: number;
  caught: number;
  biggest: string;
  species: string | null;
  lengthCm: number;
  fishDist: number;
  /** fish models currently shown (0 until one is hooked) */
  fishVisible: number;
  /** part 7's trout that bit, is hooked or held (-1: none, or no school) */
  schoolFish: number;
  bobber: [number, number, number];
  /** live water rings (x, z, age, amplitude, life, kind) */
  water: { t: number; live: number; rings: number[][]; drops: { n: number; top: number } | null; fish: number[]; release: number[]; feet: number[] };
  log: { t: number; type: string; sound: string | null; loop: string }[];
  pose: (name: string) => void;
  /** cost A/B: the line, float, aim track and fish shown or not (the rod stays) */
  visible: boolean;
  /** tests: the next bite is this fish after `delay` s on the water (no nibbles); one bite only */
  forceNext: (species: SpeciesId, lengthCm: number, delay?: number) => void;
}

declare global {
  interface Window {
    __RM_FISH?: FishDebug;
  }
}

const _d = new THREE.Vector3(), _q = new THREE.Quaternion();
const _feet = new THREE.Vector3(), _mid = new THREE.Vector3(), _eye = new THREE.Vector3(), _look = new THREE.Vector3();
const DEG = Math.PI / 180;
const _head = new THREE.Vector3();

/** fallbacks for character hooks that skip the optional queries; idle() becomes walk() once the line is in */
function resolveHooks(c: CharacterHooks, camera: THREE.PerspectiveCamera, rod: RodRig, groundY: (x: number, z: number) => number,
  state: () => FishingState): ResolvedHooks {
  return {
    playCast: (p) => c.playCast(p),
    playReel: (r, t) => c.playReel(r, t),
    playHold: (f) => c.playHold(f),
    idle: () => (c.walk && state() === 'idle' ? c.walk() : c.idle()),
    walk: c.walk ? () => c.walk!() : undefined,
    face: c.face ? (x, z, dt, rate, slack) => c.face!(x, z, dt, rate, slack) : undefined,
    playAim: c.playAim ? (p) => c.playAim!(p) : undefined,
    playStrike: c.playStrike ? () => c.playStrike!() : undefined,
    playRelease: c.playRelease ? (f) => c.playRelease!(f) : undefined,
    releaseInto: c.releaseInto ? (p, d, l) => c.releaseInto!(p, d, l) : undefined,
    attend: c.attend ? (a, aw, b, bw) => c.attend!(a, aw, b, bw) : undefined,
    getAim: c.getAim ? (o, d) => c.getAim!(o, d) : (o, d) => {
      rod.getTip(o);
      camera.getWorldDirection(d);
      d.y = 0;
      d.normalize();
    },
    getFeet: c.getFeet ? (o) => c.getFeet!(o) : (o) => {
      rod.root.getWorldPosition(o);
      o.y = groundY(o.x, o.z);
      return o;
    },
    getFishHold: c.getFishHold ? (p, q) => c.getFishHold!(p, q) : (p, q) => {
      p.set(0, -0.15, -0.75).applyMatrix4(camera.matrixWorld);
      q.copy(camera.quaternion).multiply(_q.setFromAxisAngle(_d.set(0, 1, 0), -Math.PI / 2));
      return true;
    },
  };
}

/**
 * Part 6: the fishing loop. One call from main.ts after the world is built (before the warm-up frames, so its
 * pipelines compile behind the loading screen). Registers its own engine updater.
 */
export async function installFishing(o: FishingOptions): Promise<FishingSystem> {
  const { engine, world, assets } = o;
  const { scene, camera, renderer } = engine;
  camera.layers.enable(LAYER_NO_REFLECT);

  // part 7's trout (world.fish) bite, fight and are held; part 6's own trout load only without them (?trout=0)
  const school = world.fish ?? null;
  const [rod, fishModels] = await Promise.all([
    o.rod ? Promise.resolve(o.rod) : loadRodRig(assets),
    school ? Promise.resolve(null) : FishModels.load(assets, FISH_ASSETS.fish),
  ]);
  const noRoot = new THREE.Object3D();
  const probe = new WaterProbe(world.water, world.groundY);
  const input = new FishingInput(renderer.domElement, fishParams.pointerLock && !o.character && !shotMode, !!o.character && !shotMode);
  let standIn: StandInAngler | null = null;
  let hooks: ResolvedHooks;
  let simRef: FishingSim | null = null;
  if (o.character) {
    hooks = resolveHooks(o.character, camera, rod, world.groundY, () => simRef?.state ?? 'idle');
  } else {
    standIn = new StandInAngler(camera, rod, probe, input);
    const st = STANDS[fishParams.stand] ?? STANDS.pool;
    standIn.placeAt(st.x, st.z, st.yawDeg, st.pitchDeg);
    standIn.active = !shotMode;
    scene.add(rod.root);
    hooks = standIn;
  }
  const rng = new Rng(fishParams.seed ?? ((performance.now() * 1000) >>> 0));
  const sim = new FishingSim(probe, hooks, fishingEvents, rng, (id) => fishModels?.get(id).root ?? noRoot);
  simRef = sim;
  if (school) {
    sim.school = school;
    school.onBite = (e) => sim.schoolBite(e);
    school.onLoose = (id, why) => sim.schoolLoose(id, why);
    school.onTake = (id) => sim.schoolTake(id);
  }
  const sky = world.sky;
  // the character's rod stays his (visibility, and his photo poses' bend while the fishing is hidden)
  const view = new FishingView(renderer, rod, fishModels, { dir: sky.sunDir, color: sky.sun.color, get intensity() { return sky.sun.intensity; } }, !o.character);
  scene.add(view.group);
  view.visible = !shotMode;
  const hudParent = document.getElementById('hud');
  const hud = hudParent && (!shotMode || fishParams.hud) ? new FishingHud(hudParent) : null;

  standIn?.update(0, false);
  rod.root.updateMatrixWorld(true);
  rod.getTip(sim.tip);
  sim.reset();

  // debug / test hooks
  const log: FishDebug['log'] = [];
  fishingEvents.on('*', (e: FishingEvent) => {
    if (e.loop === 'update') return;
    log.push({ t: Math.round(engine.time * 100) / 100, type: e.type, sound: e.sound, loop: e.loop });
    if (log.length > 200) log.shift();
  });

  let disposed = false;
  /** a photo pose's scripted run is stepping the loop (no turning to the camera while it aims) */
  let posing = false;
  let tackleReset = false;
  let lineShown = true;
  /**
   * Where he looks and where the rod in hand points: the spot he aims at while drawing back, the float in the air
   * (the rod's tip following it through the follow-through) and on the water, the line where it cuts in, the fish
   * he releases. Eased so the head never snaps.
   */
  const lookAt = new THREE.Vector3(), lookTo = new THREE.Vector3();
  let lookW = 0, rodW = 0;
  const attend = (fn: NonNullable<ResolvedHooks['attend']>) => {
    const s = sim.state;
    let want = 0, rodWant = 0;
    if ((s === 'aiming' || s === 'windup') && sim.aimCount > 1) {
      lookTo.copy(sim.aimLanding);
      want = 0.9;
    } else if (s === 'flight' || s === 'floating' || s === 'bite') {
      lookTo.copy(sim.bobber);
      want = s === 'flight' ? 0.8 : 0.85;
      rodWant = s === 'flight' ? 0.55 : 0;
    } else if (s === 'fight') {
      lookTo.copy(sim.bobber);
      want = 0.7;
    } else if (s === 'releasing' && sim.fish.visible) {
      lookTo.copy(sim.fish.pos);
      want = 0.9;
    }
    const k = posing ? 1 : 0.25;
    if (want > 0) lookAt.lerp(lookTo, lookW > 0.01 ? k : 1);
    lookW += (want - lookW) * k;
    rodW += (rodWant - rodW) * k;
    fn(lookW > 0.01 ? lookAt : null, lookW, rodW > 0.01 ? sim.bobber : null, rodW);
  };
  const frame = (dt: number, t: number, buttons: FishingButtons = input) => {
    if (disposed) return;
    const warm = !window.__RM?.ready;
    const rodAt = posing || !o.rodPlace ? 'hand' : o.rodPlace();
    standIn?.update(dt, !shotMode);
    hooks.getAim(sim.aimOrigin, sim.aimDir);
    hooks.getFeet(sim.feet);
    rod.getTip(sim.tip);
    if (standIn?.moved || tackleReset) {
      // carried to another place: fresh tackle there (a line or fish left behind would be dragged across the map)
      if (standIn) standIn.moved = false;
      tackleReset = false;
      sim.reset();
    }
    if (!shotMode && !posing && (rodAt !== 'laid') !== lineShown) {
      lineShown = rodAt !== 'laid';
      view.visible = lineShown;
    }
    if (standIn) standIn.lineTarget.copy(sim.state === 'fight' ? sim.fish.pos : sim.bobber);
    sim.update(dt, t, rodAt === 'hand' || buttons !== input ? buttons : NO_BUTTONS);
    if (hooks.face && dt > 0) {
      if (sim.state === 'aiming' && !posing) {
        // the cast goes where the player looks (the gameplay camera's heading)
        camera.getWorldDirection(_d);
        hooks.face(_d.x, _d.z, dt, TURN_AIM);
      } else if (sim.state === 'fight') {
        _d.subVectors(sim.fish.pos, sim.feet);
        hooks.face(_d.x, _d.z, dt, TURN_FIGHT, FIGHT_SLACK);
      }
    }
    if (hooks.attend) attend(hooks.attend);
    view.sync(sim, camera, t, warm);
    if (hud) {
      const hs = sim.hudState();
      // the character owns the pointer lock: no prompts until the mouse is captured (its own hint shows until then)
      if (!input.ready || rodAt !== 'hand') hs.prompt = null;
      hud.update(hs);
    }
    if (buttons === input) input.consume();
  };
  engine.addUpdater((dt, t) => frame(dt, t));

  const idleButtons: FishingButtons = { cast: false, castPressed: false, castReleased: false, reel: false, reelPressed: false, reelReleased: false };
  const angler = o.character ? o.angler ?? null : null;
  /** a pose moved the angler and the tackle: the next other spot puts both back */
  let posed = false;
  /** the school as it was before the shown pose's scripted run (put back when another spot is shown) */
  let poseToken: ReturnType<NonNullable<typeof school>['beginScripted']> | null = null;
  const restoreSchool = () => {
    if (school && poseToken) school.endScripted(poseToken);
    poseToken = null;
  };

  /** the pose camera in the angler's frame (see POSE_CAMERAS) */
  const frameAngler = (spec: PoseCamera) => {
    if (!angler) return;
    if ('tp' in spec) {
      angler.view(spec.tp.yaw * DEG, spec.tp.pitch * DEG, spec.tp.distance, spec.tp.fov);
      return;
    }
    hooks.getFeet(_feet);
    _mid.addVectors(sim.tip, sim.bobber).multiplyScalar(0.5);
    const yaw = spec.frame === 'line' ? Math.atan2(sim.bobber.x - _feet.x, sim.bobber.z - _feet.z) : angler.yaw;
    const rx = -Math.cos(yaw), rz = Math.sin(yaw), fx = Math.sin(yaw), fz = Math.cos(yaw);
    const anchor = (a: PoseAnchor | undefined) => (a === 'tip' ? sim.tip : a === 'float' ? sim.bobber : a === 'fish' ? sim.fish.pos
      : a === 'mid' ? _mid : _feet);
    const rel = (o: readonly number[], b: THREE.Vector3, out: THREE.Vector3) =>
      out.set(b.x + rx * o[0] + fx * o[2], b.y + o[1], b.z + rz * o[0] + fz * o[2]);
    camera.position.copy(rel(spec.eye, anchor(spec.eyeAt), _eye));
    camera.up.set(0, 1, 0);
    camera.lookAt(rel(spec.look, anchor(spec.at), _look));
    camera.fov = spec.fov;
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
  };

  /** shot poses: a deterministic scripted run ending on the frozen clock */
  const runPose = (raw: string) => {
    const side = raw.endsWith('-side');
    const name = (side ? raw.slice(0, -5) : raw) as PoseName;
    if (!POSE_NAMES.includes(name) || (!standIn && !angler)) return;
    const st = STANDS.pool;
    // the release is played out at the water's edge: knees in the shallows, the fish held in the deeper water
    const stand = name === 'release' || name === 'releasetop' ? RELEASE_STAND : ANGLER_STAND;
    const place = () => {
      if (angler) {
        const a = stand.yawDeg * DEG;
        angler.place(stand.x, stand.z, Math.atan2(-Math.sin(a), -Math.cos(a)));
      } else {
        standIn!.placeAt(st.x, st.z, st.yawDeg, st.pitchDeg);
        standIn!.resetPose();
      }
    };
    posed = true;
    restoreSchool();
    if (standIn) standIn.active = true;
    if (name === 'off') {
      if (standIn) standIn.placeAt(st.x, st.z, st.yawDeg, st.pitchDeg);
      else {
        place();
        sim.reset();
        frameAngler(POSE_CAMERAS.idle);
      }
      view.visible = false;
      frame(0, engine.time, idleButtons);
      return;
    }
    view.visible = true;
    const stop = POSE_STOPS[name];
    const simulate = (t0: number, dry: boolean): number => {
      rng.seed(POSE_SEED);
      fishingEvents.muted = true;
      posing = true;
      view.drawLine = false;
      // with part 7 the trout steps along (its own seeded stream) and is put back afterwards: the bite comes from the
      // school when the 4 s patience runs out; the dry run hears the same plop (its rings are cleared before the real run)
      const token = school ? school.beginScripted(POSE_SEED) : null;
      sim.forceBite = { delay: 4.0, species: 'brown', lengthCm: 47, nibbles: [1.6, 2.9] };
      // the fight's photograph: the trout comes up thrashing just before it
      sim.forceThrashAt = name === 'fight' || name === 'fightclose' ? 2.15 : -1;
      sim.muteWater = dry && !school;
      sim.caught = 0;
      // calm water, no drops in the air: the run makes its own rings and splashes, and its drops are stepped with it
      // (frozen at the last step like a photograph)
      probe.clearRings();
      place();
      rod.setBend(0, null);
      standIn?.update(0, false);
      rod.getTip(sim.tip);
      sim.reset();
      const player = new ScriptedPlayer(angler ? ANGLER_POWER : undefined);
      const dt = 1 / 60;
      let t = 0;
      for (let i = 0; i < 90 * 60; i++, t += dt) {
        if (stop(sim, t)) break;
        player.think(sim, t, name === 'aim');
        if (school) {
          school.setPlayer(_head.set(sim.feet.x, sim.feet.y + 1.65, sim.feet.z));
          school.update(dt, t0 + t);
        }
        angler?.step(dt);
        frame(dt, t0 + t, player);
        if (!dry) probe.stepEvents(dt, t0 + t);
      }
      fishingEvents.muted = false;
      sim.muteWater = false;
      posing = false;
      view.drawLine = true;
      if (token && dry) school!.endScripted(token);
      else poseToken = token;
      return t;
    };
    const T = engine.time;
    const duration = simulate(T, true);
    simulate(T - duration, false);
    sim.forceBite = null;
    sim.forceThrashAt = -1;
    view.focus = 0;
    if (angler) {
      frameAngler(side ? POSE_SIDE : POSE_CAMERAS[name]);
      // the angler and the fish in his hands are the subject: the line in front of them blurs
      if (name === 'hold' || name === 'release' || name === 'releasetop') view.focus = camera.position.distanceTo(sim.fish.pos);
    } else if (side && standIn) {
      // from the side: the angler's right, looking across the line
      standIn.active = false;
      const fx = -Math.sin(standIn.yaw), fz = -Math.cos(standIn.yaw);
      const mx = (sim.tip.x + sim.bobber.x) / 2, mz = (sim.tip.z + sim.bobber.z) / 2;
      camera.position.set(standIn.feet.x - fz * 7 - fx * 2.5, standIn.feet.y + 2.2, standIn.feet.z + fx * 7 - fz * 2.5);
      camera.lookAt(mx, (sim.tip.y + sim.bobber.y) / 2 - 0.3, mz);
      camera.updateMatrixWorld();
    }
    frame(0, T, idleButtons);
  };

  // photo mode: `fish-<pose>` spots; any other spot hands the camera back and hides the fishing
  if (window.__RM) {
    const prevSpot = window.__RM.spot;
    window.__RM.spot = (id: string) => {
      if (id.startsWith('fish-')) {
        // a capture spot like any other: its GPU stats start over (main.ts's spot hook does this for the rest)
        engine.resetGpuStats();
        return runPose(id.slice(5));
      }
      if (standIn) standIn.active = false;
      if (posed && angler) {
        // fresh tackle, the school and the angler as before the pose, the default lens (cam: spots keep the camera's fov)
        sim.reset();
        restoreSchool();
        probe.clearRings();
        view.focus = 0;
        angler.reset();
        camera.fov = 50;
        camera.updateProjectionMatrix();
      }
      posed = false;
      if (shotMode) view.visible = false;
      prevSpot(id);
    };
  }

  window.__RM_FISH = {
    get state() { return sim.state; },
    get stateTime() { return sim.stateTime; },
    get tension() { return sim.tension; },
    get strain() { return sim.strain; },
    get power() { return sim.power; },
    get caught() { return sim.caught; },
    get biggest() { return sim.biggest ? `${sim.biggest.name} ${sim.biggest.lengthCm} cm` : ''; },
    get species() { return sim.hookedSpecies; },
    get lengthCm() { return sim.fish.lengthCm; },
    get fishDist() { return Math.hypot(sim.fish.pos.x - sim.feet.x, sim.fish.pos.z - sim.feet.z); },
    get fishVisible() { return fishModels ? SPECIES_IDS.filter((id) => fishModels.get(id).root.visible).length : sim.schoolFish >= 0 ? 1 : 0; },
    get schoolFish() { return sim.schoolFish; },
    get bobber() { return [sim.bobber.x, sim.bobber.y, sim.bobber.z] as [number, number, number]; },
    get water() {
      const t = probe.time;
      const w = world.water;
      return { t, live: eventsU.live.value, drops: w ? w.events.droplets.stats((x, z) => w.query(x, z).y) : null,
        fish: [sim.fish.pos.x, sim.fish.pos.y, sim.fish.pos.z, sim.fish.visible ? 1 : 0], release: [sim.releaseAt.x, sim.releaseAt.y, sim.releaseAt.z],
        feet: [sim.feet.x, sim.feet.y, sim.feet.z], rings: eventA.map((a, i) => [a.x, a.y, Math.round((t - a.z) * 100) / 100, a.w, eventB[i].z, eventC[i].x])
        .filter((r) => (r[2] as number) >= 0 && (r[2] as number) < (r[4] as number)) };
    },
    log,
    pose: runPose,
    get visible() { return view.visible; },
    set visible(v: boolean) { view.visible = v; },
    forceNext: (species, lengthCm, delay = 2) => {
      sim.forceBite = { delay, species, lengthCm, nibbles: [] };
      const off = fishingEvents.on('bite', () => {
        sim.forceBite = null;
        off();
      });
    },
  };

  return {
    sim, view, events: fishingEvents, standIn, hud,
    frame,
    setInputEnabled: (on) => {
      input.enabled = on;
    },
    resetTackle: () => {
      tackleReset = true;
    },
    dispose: () => {
      // the engine has no removeUpdater: the registered frame() just stops doing anything
      disposed = true;
      input.dispose();
      scene.remove(view.group);
      if (standIn) scene.remove(rod.root);
      hud?.root.remove();
    },
  };
}

export { TUNE };
