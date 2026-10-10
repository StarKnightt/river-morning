import type { SlotName } from './Animator';

/**
 * Fixed character views for the part 5 gauntlet (`pnpm shots --spots P1,P2,...`), like C1-C6 for the scenery:
 * the angler is placed and posed at a frozen clip time, then the camera framed relative to him. Append `h` to an
 * id (P1h) for the same view with the angler and rod hidden: the GPU difference is their cost.
 */
export interface PlayerSpot {
  id: string;
  label: string;
  /** feet position; yaw in degrees (character +Z toward (sin, cos)); omitted yaw faces the river */
  at: { x: number; z: number; yaw?: number };
  pose: { clip: SlotName; t?: number; speed?: number; phase?: number; rodHeld?: boolean; bend?: number };
  /**
   * 'tp': the gameplay camera, yaw relative to the character's facing (deg), pitch (deg, + looks down);
   * 'rel': eye and look-at in the character's frame [right, up, forward] (m)
   */
  camera:
    | { mode: 'tp'; yaw?: number; pitch?: number; distance?: number }
    | { mode: 'rel'; eye: [number, number, number]; look: [number, number, number]; fov?: number };
}

export const PLAYER_SPOTS: PlayerSpot[] = [
  { id: 'P1', label: 'spawn, gameplay camera toward the river', at: { x: 0, z: 9 }, pose: { clip: 'loco', t: 0.8, rodHeld: true }, camera: { mode: 'tp', pitch: 12 } },
  {
    id: 'P2', label: 'three-quarter front close-up (skin, wool, felt, leather)', at: { x: -1.5, z: 6.6, yaw: 150 },
    pose: { clip: 'loco', t: 0.8, rodHeld: true }, camera: { mode: 'rel', eye: [1.25, 1.6, 2.0], look: [0, 1.3, 0], fov: 38 },
  },
  {
    id: 'P3', label: 'cast at the line release (2.47 s), side view', at: { x: 0.5, z: 5.8, yaw: 185 },
    pose: { clip: 'cast', t: 2.47 }, camera: { mode: 'rel', eye: [4.2, 1.7, 0.4], look: [0, 1.9, 0.9], fov: 50 },
  },
  {
    id: 'P4', label: 'reeling, rod bent toward the water', at: { x: 0.5, z: 5.8, yaw: 185 },
    pose: { clip: 'reel_loop', t: 1.0, bend: 0.55 }, camera: { mode: 'tp', yaw: 35, pitch: 8, distance: 2.6 },
  },
  {
    id: 'P5', label: 'walking on the dock deck, rod in hand', at: { x: 11.5, z: 4.4, yaw: 180 },
    pose: { clip: 'loco', speed: 0.85, phase: 0.3, rodHeld: true }, camera: { mode: 'tp', yaw: 40, pitch: 14 },
  },
  {
    id: 'P6', label: 'hold_fish, front', at: { x: 0.5, z: 6.4, yaw: 160 },
    pose: { clip: 'hold_fish', t: 0.5 }, camera: { mode: 'rel', eye: [-0.5, 1.55, 1.9], look: [0, 1.3, 0.25], fov: 40 },
  },
  {
    id: 'P7', label: 'wading at the depth limit (~0.35 m), slow walk', at: { x: 0, z: 3.25, yaw: 190 },
    pose: { clip: 'loco', speed: 0.85, phase: 0.6, rodHeld: true }, camera: { mode: 'tp', yaw: 70, pitch: 10, distance: 3.0 },
  },
  {
    id: 'P8', label: 'boots on the dock deck, low side view (check against the placed dock)', at: { x: 11.5, z: 5.2, yaw: 180 },
    pose: { clip: 'loco', t: 0.8, rodHeld: true }, camera: { mode: 'rel', eye: [2.2, 0.75, 0.6], look: [0, 0.35, 0.1], fov: 40 },
  },
  {
    id: 'P9', label: 'standing across the 28-degree bank at spawn, legs from the front (foot IK)', at: { x: 0, z: 9, yaw: -75 },
    pose: { clip: 'loco', t: 0.8, rodHeld: true }, camera: { mode: 'rel', eye: [0.5, 1.0, 2.6], look: [0, 0.55, 0], fov: 42 },
  },
];
