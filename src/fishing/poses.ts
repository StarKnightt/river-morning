import type { FishingButtons, FishingState } from './types';
import type { FishingSim } from './FishingSim';
import { TUNE } from './config';

/**
 * Deterministic fishing moments for photo mode (`pnpm shots --extra fish --spots fish-aim,...`): one scripted run
 * of the loop (fixed seed, fixed fish) simulated at 60 Hz up to the moment, ending exactly on the frozen clock so
 * splash rings show at the right age. A `-side` suffix views the same moment from the side of the angler.
 */
export const POSE_NAMES = ['idle', 'aim', 'flight', 'splash', 'float', 'bite', 'fight', 'fightclose', 'near', 'hold', 'release', 'releasetop', 'off'] as const;
export type PoseName = (typeof POSE_NAMES)[number];

/** cast power of the scripted run: lands ~11 m out in the deep pool from the pool stand */
export const POSE_POWER = 0.42;
export const POSE_SEED = 20261009;

/**
 * Where the angler (part 5) stands for the poses: the gravel bar at the pool's edge, facing across the pool (yaw in
 * the stand-in's convention: 0 = north (-Z), + = turning west). The camera stand-in keeps STANDS.pool.
 */
export const ANGLER_STAND = { x: 4.0, z: 5.0, yawDeg: 20 };
/**
 * The release pose's stand, 1.85 m farther out on the same line: down on one knee in ~35 cm of water on the slope of
 * the shelf, the other knee up at the surface, so with his back at a slant (not folded over) his forearms still go
 * under it; the bed drops to ~50 cm a third of a metre in front of him, enough to hold a 47 cm trout under the
 * surface. The rod lies on the dry stones at the edge of the bar, off to his left (angler.ts LAY).
 */
export const RELEASE_STAND = { x: 3.375, z: 3.285, yawDeg: 20 };
/** his cast is longer than the stand-in's (rod tip higher and farther out at the release): this lands in the deep pool */
export const ANGLER_POWER = 0.24;

/** what a pose camera looks at: the angler's feet, the rod tip, the float, the hooked fish, or tip-to-float midway */
export type PoseAnchor = 'feet' | 'tip' | 'float' | 'fish' | 'mid';
/**
 * Pose cameras for the angler. `tp`: the gameplay camera (yaw offset from his facing and pitch in degrees, + looks
 * down). Otherwise eye and look offsets in his frame [right, up, forward] (m): the eye from `eyeAt` (default his
 * feet), the look point from `at`.
 */
export type PoseCamera =
  | { tp: { yaw: number; pitch: number; distance: number; fov: number } }
  | {
    eye: [number, number, number]; eyeAt?: PoseAnchor; look: [number, number, number]; at: PoseAnchor; fov: number;
    /** 'line': the frame's forward is from his feet toward the float/fish (he turns toward a running trout) */
    frame?: 'line';
  };

export const POSE_CAMERAS: Record<Exclude<PoseName, 'off'>, PoseCamera> = {
  idle: { tp: { yaw: 0, pitch: 12, distance: 2.75, fov: 55 } },
  // behind him and off to his right: the rod drawn back over his shoulder with the float hanging from its tip, the
  // dotted track running out across the water to the landing ring (straight behind him it stood in a column)
  aim: { eye: [2.2, 2.0, -3.6], look: [0.1, 2.0, 6.2], at: 'feet', fov: 58 },
  // behind his right shoulder, up: the rod forward, the line arcing out to the float in the air
  flight: { eye: [1.8, 2.1, -3.0], look: [0, -0.5, 0], at: 'mid', fov: 50 },
  // close over the water (~4 m from the float): the float landing, the line running back to the rod
  splash: { eye: [1.4, 1.1, -3.8], eyeAt: 'float', look: [0, 0.05, 0], at: 'float', fov: 40 },
  // from behind him and above his hat, down the line to the float riding the pool ~12 m out: the rod tip in the
  // frame, the line dropping from it to the water and lying on it the last stretch to the float
  float: { eye: [0.8, 2.3, -3.0], look: [-0.3, 1.7, 0], at: 'float', fov: 40 },
  // close over the water: the float dipping, the first ring
  bite: { eye: [0.9, 0.8, -2.8], eyeAt: 'float', look: [0, 0.02, 0], at: 'float', fov: 38 },
  // from his right side: leaning back against the pull, the rod in one deep arc up over him with its tip in the frame,
  // the line running taut from it down to the water where the trout is
  fight: { eye: [3.8, 1.25, -0.6], look: [0.4, 1.45, 2.6], at: 'feet', fov: 50, frame: 'line' },
  near: { eye: [2.8, 1.7, -1.0], look: [0, 0, 0], at: 'mid', fov: 45, frame: 'line' },
  // the fight's moment from ~2.5 m off the trout, across the line (a check view of the thrash, not a critic spot)
  fightclose: { eye: [1.8, 1.1, -1.6], eyeAt: 'fish', look: [0, 0, 0], at: 'fish', fov: 40, frame: 'line' },
  // front: the trout held up across his hands (part 5's P6 framing)
  hold: { eye: [-0.5, 1.55, 1.9], look: [0, 1.3, 0.25], at: 'feet', fov: 40 },
  // in front of him and to his left, low over the water: down on his right knee with the left one up, his back at a
  // slant and his head up over the trout held across in front of him (broadside to the lens), forearms in the water,
  // the rod lying on the stones at the edge of the bar beyond him, clear of him in the frame
  release: { eye: [-1.6, 0.85, 2.1], eyeAt: 'fish', look: [-0.2, 0.3, -0.4], at: 'fish', fov: 40 },
  // the release from over the trout's back (a check view, not a critic spot)
  releasetop: { eye: [0.6, 1.5, 0.9], eyeAt: 'fish', look: [0, 0, 0], at: 'fish', fov: 50 },
};
/** `-side` suffix: from his right, a little behind, looking out along the line */
export const POSE_SIDE: PoseCamera = { eye: [5.0, 1.8, -2.5], look: [0, 0.8, 4.5], at: 'feet', fov: 55, frame: 'line' };

export interface PoseStop {
  done: (sim: FishingSim, t: number) => boolean;
}

const at = (s: FishingState, time: number) => (sim: FishingSim) => sim.state === s && sim.stateTime >= time;

export const POSE_STOPS: Record<Exclude<PoseName, 'off'>, PoseStop['done']> = {
  idle: (_s, t) => t >= 0.45,
  aim: at('aiming', 0.3),
  // the float just over the top of its climb, out ahead of the rod, the line trailing up to it in one open curve
  // (later the line traced the float's way over the top: an upside-down V); the follow-through has settled by 0.4 s
  flight: at('flight', 0.65),
  // the crown still falling back, the jet up, the first crisp ring spreading round the float
  splash: at('floating', 0.21),
  float: at('floating', 3.0),
  // the second dip: the float half under, the rings of the first dip and of this one spreading
  bite: at('bite', 0.3),
  fight: at('fight', 2.6),
  fightclose: at('fight', 2.6),
  near: (sim) => sim.state === 'fight' && Math.hypot(sim.fish.pos.x - sim.feet.x, sim.fish.pos.z - sim.feet.z) < 4.2,
  hold: at('holding', 1.0),
  // on one knee in the shallows, the trout held facing upstream under the surface, a moment before it kicks away
  release: at('releasing', 2.7),
  releasetop: at('releasing', 2.7),
};

/** the scripted player: cast at 0.5 s, hook 0.3 s into the bite, reel with tension hysteresis */
export class ScriptedPlayer implements FishingButtons {
  cast = false;
  castPressed = false;
  castReleased = false;
  reel = false;
  reelPressed = false;
  reelReleased = false;
  private hold: number;

  constructor(power = POSE_POWER) {
    this.hold = TUNE.chargeTime * (1 - Math.sqrt(1 - power));
  }

  private set(k: 'cast' | 'reel', down: boolean) {
    if (k === 'cast') {
      if (down && !this.cast) this.castPressed = true;
      if (!down && this.cast) this.castReleased = true;
      this.cast = down;
    } else {
      if (down && !this.reel) this.reelPressed = true;
      if (!down && this.reel) this.reelReleased = true;
      this.reel = down;
    }
  }

  /** decide this frame's buttons from the simulation (call before sim.update) */
  think(sim: FishingSim, t: number, holdAim: boolean) {
    this.castPressed = this.castReleased = this.reelPressed = this.reelReleased = false;
    const s = sim.state;
    if (s === 'idle' && t >= 0.5 && sim.caught === 0 && !this.cast) this.set('cast', true);
    if (s === 'aiming' && sim.stateTime >= this.hold && !holdAim) this.set('cast', false);
    if (s === 'bite' && sim.stateTime >= 0.5) this.set('cast', true);
    if (s === 'fight') {
      if (this.cast) this.set('cast', false);
      if (!this.reel && sim.tension < 0.55) this.set('reel', true);
      if (this.reel && (sim.tension > 0.82 || sim.strain > 0.05)) this.set('reel', false);
    } else if (this.reel) this.set('reel', false);
    if (s === 'floating' && this.cast) this.set('cast', false);
  }
}
