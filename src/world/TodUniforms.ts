import * as THREE from 'three/webgpu';
import { uniform } from '../gfx/tsl';

/**
 * Time-of-day values read by shaders outside the sky plate (part 8 round 2), written every frame of a fade by
 * TimeOfDayController.apply. A leaf module, so the terrain and the sky can read them without importing the controller.
 */
export const todSkyU = {
  /** 0..1: the pre-sunrise sky's bands away from the sun (a rose Belt of Venus over the blue-grey Earth's shadow),
   * the warm glow low toward it and the deep blue overhead */
  dawn: uniform(0),
  /** first light on the summits while the valley still lies in shade: colour x strength (0 = none) */
  alpen: uniform(new THREE.Color(0, 0, 0)),
  /** height (m) above which the summits catch it (it fades in over ~150 m) */
  alpenLine: uniform(1250),
  /** the moon's light (colour x intensity x night) that the ridges hold off the far faces turned away from it */
  ridge: uniform(new THREE.Color(0, 0, 0)),
  /** roughness of the far snowfields: backlit by a low moon, rough-0.6 snow seen edge-on glared as if sunlit */
  snowRough: uniform(0.6),
  /** toward the sun, unit (at dawn a few degrees over the horizon, behind the ridges) */
  sunDir: uniform(new THREE.Vector3(0.7, 0.05, -0.7).normalize()),
};
