// Scratch probe (not part of the game): water depth and C2 screen position of candidate splash points.
import * as THREE from 'three';
import { river, newSample } from '../src/world/River';
import { heightAt } from '../src/world/Terrain';

const S = newSample();
const cam = new THREE.PerspectiveCamera(50, 16 / 9, 0.05, 20000);
cam.position.set(4.6, Math.max(heightAt(4.6, 6.4), 0) + 1.15, 6.4);
cam.lookAt(3.2, -0.7, 2.2);
cam.updateMatrixWorld();
const pts: [number, number][] = [];
for (let z = 2.6; z <= 3.9; z += 0.15) for (let x = 2.8; x <= 4.6; x += 0.3) pts.push([x, z]);
for (const [x, z] of pts) {
  river.sample(x, z, S);
  const g = heightAt(x, z);
  const v = new THREE.Vector3(x, S.waterY, z).project(cam);
  const sx = (v.x * 0.5 + 0.5) * 1920, sy = (0.5 - v.y * 0.5) * 1080;
  if (S.waterY - g < 0.05) continue;
  console.log(`${x.toFixed(2)},${z.toFixed(2)} depth ${(S.waterY - g).toFixed(2)} screen ${sx.toFixed(0)},${sy.toFixed(0)}`);
}
