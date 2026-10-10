import * as THREE from 'three/webgpu';
import { uniform } from '../gfx/tsl';

/** Values shared by the camp's materials and effects; written once per frame by Camp.update. */
export const campU = {
  time: uniform(0),
  /** 0..1 how big the campfire burns (a small cooking fire in the morning) */
  strength: uniform(0.45),
  /** flicker factor of this frame (~0.65-1.07), shared by the flames, the embers and part 8's fire light */
  flicker: uniform(1),
  /** breeze at chimney height (world m/s): down the valley (+X) and a little south, like the mist and the trees */
  wind: uniform(new THREE.Vector3(0.72, 0, 0.3)),
  /** flame radiance at the core for strength 1 (pre-exposure; bloom starts above ~2.2) */
  flameGain: uniform(2.6),
  /** ember emissive gain for strength 1 */
  emberGain: uniform(2.4),
  /** centre of the fire ring on the ground (world): the burnt wood and the ash bed are graded from it */
  fireCenter: uniform(new THREE.Vector3(0, -1e3, 0)),
  /** the door latch (world xyz) and the radius of the hand-worn patch round it */
  latch: uniform(new THREE.Vector4(0, -1e3, 0, 0.2)),
  /**
   * Smoke colour for part 8 (per time of day): the tint of the sunlit side (multiplies the sun's own colour and
   * strength; warm white in the morning) and of the side in shade (multiplies the sky light; cool blue-grey)
   */
  smokeSun: uniform(new THREE.Vector3(1.0, 0.93, 0.83)),
  smokeShade: uniform(new THREE.Vector3(0.74, 0.82, 0.97)),
  /** part 8's night amount (0 by day): the coals spread, the cracks glow along the burnt wood, the smoke takes the flames */
  night: uniform(0),
  /** the oil lamp inside the cabin behind the river-facing window (world): the panes are hottest in line with it */
  lamp: uniform(new THREE.Vector3(0, -1e3, 0)),
};

/**
 * Flame and ember gains by day and at night. Without part 8 they stay at the day values; with it attached they
 * follow its night, so at night the flames outshine the stones the fire light falls on. Part 8's night renders at
 * exposure 1.5, which brightens them further: at 5 / 4 the flames and coals burnt out to white. (Part 8 round 2:
 * exposure 2; at 2.2 the overlapping tongues still summed to "a formless white-yellow glow under the grill": the
 * flames lower so each tongue keeps its shape, the coals higher so they glow red-orange.)
 */
export const FIRE_GAIN = { flameDay: 2.6, flameNight: 1.6, emberDay: 2.4, emberNight: 1.7 };
