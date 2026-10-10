import { LAYOUT } from '../world/Layout';
import { newPoint, sampleTerrain } from '../world/Terrain';
import type { Surface } from './events';

const tp = newPoint();

// the prepared cabin (prep/blender/blender-assets.json): 6.0 x 5.0 m log footprint (local x +-3.0, z +-2.5) with
// the porch deck on the door side out to z 4.65; the door side is turned 180 degrees to face the river
const CABIN_TURN = Math.PI;

/** campfire bench to the cabin and down to the dock: the rocky-trail footpaths */
const PATHS: Array<[number, number, number, number]> = [
  [LAYOUT.campfire.x, LAYOUT.campfire.z, LAYOUT.cabin.x - 1.5, LAYOUT.cabin.z - 4.0],
  [LAYOUT.campfire.x, LAYOUT.campfire.z, LAYOUT.dock.x, LAYOUT.dock.zBase + 0.6],
];

function segDist(x: number, z: number, ax: number, az: number, bx: number, bz: number) {
  const dx = bx - ax, dz = bz - az;
  const t = Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz)));
  return Math.hypot(x - ax - dx * t, z - az - dz * t);
}

/**
 * Footstep surface from the world layout: dock planks, the cabin's plank floor and porch, wading in the shallows,
 * the gravel bar and footpaths, grass elsewhere. One terrain sample (~50 us), once per step.
 */
export function defaultSurface(x: number, z: number): Surface {
  const d = LAYOUT.dock;
  if (Math.abs(x - d.x) < 0.95 && z > d.zTip - 0.4 && z < d.zBase + 0.3) return 'wood';
  const c = LAYOUT.cabin, a = c.rotY + CABIN_TURN, ca = Math.cos(a), sa = Math.sin(a);
  const dx = x - c.x, dz = z - c.z;
  const lx = ca * dx - sa * dz, lz = sa * dx + ca * dz;
  if (Math.abs(lx) < 3.1 && lz > -2.6 && lz < 4.7) return lz < 2.6 ? 'creak' : 'wood';
  sampleTerrain(x, z, tp);
  if (tp.waterY - tp.y > 0.04) return 'water';
  if (tp.gravel > 0.4) return 'gravel';
  for (const [ax, az, bx, bz] of PATHS) if (segDist(x, z, ax, az, bx, bz) < 0.8) return 'gravel';
  return 'grass';
}
