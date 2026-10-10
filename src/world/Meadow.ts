import { fbm, smoothstep } from './noise';

/**
 * Patchiness of the open grass, shared by the ground shader (baked per vertex into the corridor grid) and the grass
 * blades, so both agree: `bare` = worn or thin patches where the soil and old litter show, `straw` = last year's dead
 * grass, straw-yellow, in drifts between the green (ref_01's right bank).
 */
export function meadowAt(x: number, z: number, out: { bare: number; straw: number }) {
  out.bare = smoothstep(0.3, 0.5, fbm(x * 0.075 + 7.7, z * 0.075 - 3.1, 3)) * smoothstep(-0.1, 0.25, fbm(x * 0.31 - 1.4, z * 0.31 + 6.2, 2));
  out.straw = smoothstep(0.02, 0.38, fbm(x * 0.045 - 5.5, z * 0.045 + 2.4, 3));
  return out;
}
