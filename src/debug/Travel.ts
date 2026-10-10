import * as THREE from 'three/webgpu';
import { applyCameraSpot } from './CameraSpots';
import { heightAt } from '../world/Terrain';
import { LAYOUT } from '../world/Layout';

/** Extra destinations beyond the six fixed camera spots (eye height above ground, absolute look target). */
const EXTRA: Record<string, { eye: [number, number, number]; target: [number, number, number] }> = {
  '7': { eye: [-30, 1.7, 8.5], target: [-62, 3, 5] },
  '8': { eye: [30, 1.7, 6.5], target: [62, -0.5, 4] },
  '9': { eye: [9, 1.7, 16], target: [9, 0.4, -5] },
  '0': { eye: [LAYOUT.spawn.x, 1.7, LAYOUT.spawn.z], target: [LAYOUT.spawn.x - 6, 0.4, LAYOUT.spawn.z - 10] },
};

export interface TravelHooks {
  /** move the player instead of the camera (set once the character exists) */
  onTeleport?: (feet: THREE.Vector3, look: THREE.Vector3) => void;
  /** let camera controllers re-read the new pose */
  afterMove?: () => void;
}

/** Jumps between fixed places; no UI on purpose. */
export function installTravelKeys(camera: THREE.PerspectiveCamera, hooks: TravelHooks = {}) {
  const look = new THREE.Vector3();
  const feet = new THREE.Vector3();
  window.addEventListener('keydown', (e) => {
    if (e.repeat || e.ctrlKey || e.altKey || e.metaKey) return;
    const el = e.target as HTMLElement | null;
    if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
    const m = /^(?:Digit|Numpad)(\d)$/.exec(e.code);
    if (!m) return;
    const key = m[1];
    if (key >= '1' && key <= '6') {
      const spot = applyCameraSpot(camera, `C${key}`);
      if (!spot) return;
      look.fromArray(spot.target);
    } else {
      const dst = EXTRA[key];
      if (!dst) return;
      const [x, h, z] = dst.eye;
      camera.position.set(x, Math.max(heightAt(x, z), 0) + h, z);
      look.fromArray(dst.target);
      camera.lookAt(look);
    }
    if (hooks.onTeleport) {
      feet.set(camera.position.x, heightAt(camera.position.x, camera.position.z), camera.position.z);
      hooks.onTeleport(feet, look);
    }
    hooks.afterMove?.();
  });
}
