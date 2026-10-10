import type * as THREE from 'three/webgpu';

/** One fishing loop: idle -> aiming -> windup -> flight -> floating -> bite -> fight -> landing -> holding -> releasing. */
export type FishingState =
  | 'idle' | 'aiming' | 'windup' | 'flight' | 'floating' | 'bite' | 'fight' | 'landing' | 'holding' | 'releasing' | 'snapped';

export type SpeciesId = 'brown' | 'rainbow' | 'brook' | 'cutthroat';

/**
 * A rod the fishing system can drive. `SegmentRodRig` implements it for the Blender rod (8 nested bend segments)
 * and for the procedural stand-in; the character lane can pass its own (e.g. the same rig parented to hand_r).
 */
export interface RodRig {
  readonly root: THREE.Object3D;
  /** world position of the tip-top ring, where the line leaves (world matrices must be current) */
  getTip(out: THREE.Vector3): THREE.Vector3;
  /** bend 0..1 (1 = hooped under a big fish), toward a world-space direction (the line); null keeps the last plane */
  setBend(amount: number, towardWorld?: THREE.Vector3 | null): void;
  /** spin the reel handle by a signed number of turns (+ = winding in) */
  crank(turns: number): void;
}

/** The fish handed to the character for the hold_fish moment. */
export interface HeldFish {
  /** fish frame: origin at the centre of mass, +Z toward the nose, +Y up, painted left flank toward +X */
  readonly root: THREE.Object3D;
  readonly species: SpeciesId;
  readonly name: string;
  readonly lengthCm: number;
  readonly weightKg: number;
  /** uniform scale of the root (caught length / model length) */
  readonly scale: number;
  /** metres along +Z that put the tail wrist in the right hand (fish.json hold_fish_offset_m, already scaled) */
  readonly holdOffset: number;
  /** extra scale suggested while held (the brook is short for the pose's 0.226 m between the palms) */
  readonly holdScale: number;
}

/**
 * What the fishing system asks of the character (part 5). Every call is a one-way notification; timings that the
 * gameplay depends on come back as return values. Without a character the camera stand-in implements it.
 * Clip names refer to character.json.
 */
export interface CharacterHooks {
  /** start the cast (clip `cast`); return seconds until the line leaves the rod (its `release` event) */
  playCast(power: number): number;
  /** fight pose (clip `reel_loop`, scaled by `reeling`); tension 0..1 for effort/lean */
  playReel(reeling: boolean, tension: number): void;
  /** landing: `land_lift` then `hold_fish`. Place the fish with getFishHold (or parent fish.root and return false there) */
  playHold(fish: HeldFish): void;
  /** back to the waiting pose (clip `wait`: rod in both hands, watching the float) */
  idle(): void;
  /** optional: aiming pose while the left button is held, power 0..1 */
  playAim?(power: number): void;
  /** optional: hook set (clip `strike`) */
  playStrike?(): void;
  /** optional: release (clip `release_fish` / `kneel_release`); return seconds until the hands let go of the fish */
  playRelease?(fish: HeldFish): number;
  /** optional: world transform of the held fish's frame (hand socket + holdOffset); return false if you parent it */
  getFishHold?(pos: THREE.Vector3, quat: THREE.Quaternion): boolean;
  /** optional: where the line leaves at the release moment and the horizontal cast direction */
  getAim?(origin: THREE.Vector3, dir: THREE.Vector3): void;
  /** optional: the angler's feet on the ground (landing distance, release spot) */
  getFeet?(out: THREE.Vector3): THREE.Vector3;
  /**
   * optional: the line is in (state `idle`): free to walk with the rod in hand. Called instead of idle() then;
   * idle() still comes while the float is out (after an escaped bite, a thrown hook, a snap).
   */
  walk?(): void;
  /**
   * optional: turn on the spot toward world direction (x, z), at most `rate` rad/s, only by the part of the angle
   * beyond `slack` rad (aiming: toward the camera's heading; the fight: toward the fish)
   */
  face?(x: number, z: number, dt: number, rate: number, slack?: number): void;
  /** optional: at the start of the release, where the fish is to be held in the water and the way it faces (upstream) */
  releaseInto?(target: THREE.Vector3, upstream: THREE.Vector3, lengthM: number): void;
  /**
   * optional, every frame: a world point to watch (the landing spot, the float, the line; null = the clip's own gaze)
   * and one to turn the rod in hand toward (the float in flight), each with a weight 0..1
   */
  attend?(look: THREE.Vector3 | null, lookW: number, rodAt: THREE.Vector3 | null, rodW: number): void;
}

/** Buttons as the simulation sees them: held flags plus press/release edges since the last frame. */
export interface FishingButtons {
  cast: boolean;
  castPressed: boolean;
  castReleased: boolean;
  reel: boolean;
  reelPressed: boolean;
  reelReleased: boolean;
  /** cancel an aim (window lost focus) */
  cancel?: boolean;
}

export interface CaughtFish {
  species: SpeciesId;
  name: string;
  lengthCm: number;
  weightKg: number;
}

export type PromptKind = 'cast' | 'release' | 'wait' | 'reelin' | 'hook' | 'reel' | 'ease' | null;

/**
 * Part 7's trout as the loop sees them (`world.fish`, a FishSchool, fits structurally): which trout bites and its
 * size, the hooked trout's motion, stamina and landing, drawing it in the hands and letting it go. The loop keeps
 * the tackle: tension, drag, strain and snap, reeling, the HUD, sounds and the catch card.
 */
export interface FishSource {
  /** the bait in the water this frame (null when it is out) */
  setLure(l: { x: number; y: number; z: number; kind?: 'bobber' | 'lure' | 'fly'; speed?: number } | null): void;
  /** send the nearest calm trout within `range` m to the bait (the biggest of `species`, if given); its id, or -1 */
  attract(range: number, species?: SpeciesId): number;
  hook(fishId: number): boolean;
  /** every frame of the fight: rod tip and the line's pull on the hook (N) */
  setLine(fishId: number, line: { tip: { x: number; y: number; z: number }; tension: number }): void;
  fight(fishId: number): FishSourceFight | null;
  unhook(fishId: number): void;
  land(fishId: number): { position: THREE.Vector3; quaternion: THREE.Quaternion; scale: number } | null;
  /** `sag`: how far its body sags between the hands, a fraction of its length */
  setHeld(fishId: number, position: THREE.Vector3, quaternion: THREE.Quaternion, scale?: number, sag?: number): void;
  /** the hooked trout comes up and thrashes on top for `seconds` (the photo poses' fight) */
  thrash?(fishId: number, seconds?: number): void;
  release(fishId: number, at?: { x: number; z: number }): void;
  species(id: SpeciesId): { holdOffset: number; holdScale: number; modelLength: number };
}

export interface FishSourceFight {
  readonly mouth: THREE.Vector3;
  readonly velocity: THREE.Vector3;
  /** unit direction it swims and pulls toward, and its thrust now (N) */
  readonly pull: THREE.Vector3;
  readonly pullForce: number;
  readonly maxForce: number;
  readonly stamina: number;
  readonly phase: string;
  readonly airborne: boolean;
  readonly landable: boolean;
  readonly thrashes: number;
}

export interface FishSourceBite {
  fishId: number;
  species: SpeciesId;
  lengthM: number;
  weightKg: number;
  /** seconds it keeps the bait: the strike must come within this */
  window: number;
  /** 0.35-0.75 a pluck at a bait */
  strength: number;
}
