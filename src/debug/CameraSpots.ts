import * as THREE from 'three/webgpu';
import { heightAt } from '../world/Terrain';

/**
 * Fixed camera spots. Every build part is photographed from exactly these views so the
 * critic always compares like with like. Eye positions are given as height above ground.
 */
export interface CameraSpot {
  id: string;
  label: string;
  eye: [number, number, number]; // x, height above ground, z
  target: [number, number, number]; // absolute
  fov: number;
}

export const CAMERA_SPOTS: CameraSpot[] = [
  { id: 'C1', label: 'gravel bar looking upstream', eye: [8, 1.65, 8.6], target: [-26, 0.9, 2.0], fov: 50 },
  { id: 'C2', label: 'clear water close-up', eye: [4.6, 1.15, 6.4], target: [3.2, -0.7, 2.2], fov: 50 },
  { id: 'C3', label: 'riffle from the south bank', eye: [-19, 1.6, 9.6], target: [-25, 0.2, -1.0], fov: 50 },
  { id: 'C4', label: 'dock tip across the pool', eye: [11.5, 2.25, 2.0], target: [15, 1.6, -16], fov: 55 },
  { id: 'C5', label: 'water edge looking up to camp', eye: [4.0, 1.6, 5.4], target: [8.5, 3.6, 22], fov: 55 },
  { id: 'C6', label: 'overview', eye: [-16, 20, 32], target: [6, -1, 0], fov: 50 },
];

export function applyCameraSpot(camera: THREE.PerspectiveCamera, id: string): CameraSpot | null {
  const spot = CAMERA_SPOTS.find((s) => s.id === id);
  if (!spot) return null;
  const [x, h, z] = spot.eye;
  // ground under the eye, but never below the pool surface (dock / over-water spots)
  const ground = Math.max(heightAt(x, z), 0);
  camera.position.set(x, ground + h, z);
  camera.fov = spot.fov;
  camera.lookAt(new THREE.Vector3(...spot.target));
  camera.updateProjectionMatrix();
  return spot;
}
