import type * as THREE from 'three/webgpu';
import { uniform, type N } from '../gfx/tsl';

/**
 * Where the water meets the gravel. The bed stones' own relief (the packed height of the bed-pebble set) decides
 * which stones break the surface, so the waterline runs round every stone instead of along the smooth ground mesh.
 * The core terrain material (its `under` mask) and the water material (its contact fade and meniscus) apply the same
 * rule at the same point (the opaque point behind the water pixel), so the two always agree.
 */
export const shoreU = {
  /** relief of the stones at the waterline (m per unit of packed height around 0.5) */
  relief: uniform(0.04),
  /** the contact sits this much deeper than the ground mesh's (m): only stone tops break out beyond the mesh line */
  shift: uniform(0.008),
};

/** the bed pebbles' packed colour + height texture and its tiling (m); set by the core terrain material */
export const shoreStones: { tex: THREE.Texture | null; size: number } = { tex: null, size: 2.16 };

/** height (m) of the effective stone surface above the ground mesh, from the packed height h (0..1) */
export const stoneLift = (h: N): N => h.sub(0.5).mul(shoreU.relief).add(shoreU.shift);
