import * as THREE from 'three/webgpu';
import type { Ground } from './Ground';

export interface Framing {
  /** camera distance behind the pivot (m) */
  distance: number;
  /** sideways offset to the right of the character (m): over the right shoulder */
  shoulder: number;
  /** pivot height above the feet (m) */
  height: number;
  /** vertical field of view (degrees) */
  fov: number;
}

/** Close, RDR2-like follow: the angler on the left third, the river ahead readable. */
export const FRAMING_EXPLORE: Framing = { distance: 2.75, shoulder: 0.6, height: 1.55, fov: 55 };

const PITCH_MIN = THREE.MathUtils.degToRad(-35);
const PITCH_MAX = THREE.MathUtils.degToRad(60);
const CLEARANCE = 0.25;
const WATER_CLEARANCE = 0.12;
/** closest the eye comes to a wall (the ground's box blockers: cabin walls, chimney, woodpile) */
const WALL_CLEARANCE = 0.22;
/** closest the eye comes to the pivot (his head) when nothing is in the way */
const MIN_EYE = 0.6;

/**
 * Over-the-shoulder orbit: the mouse sets yaw/pitch, the pivot follows the feet through critically damped
 * springs (softer vertically, so steps and the walk's bob do not shake the view), and the boom shortens where
 * the terrain or the water surface would come between the pivot and the eye.
 */
export class ThirdPersonCamera {
  /** view azimuth, same convention as the character's yaw: looking along (sin yaw, 0, cos yaw) */
  yaw = 0;
  /** + looks down */
  pitch = THREE.MathUtils.degToRad(12);
  sensitivity = 0.0022;
  framing: Framing = { ...FRAMING_EXPLORE };
  private target: Framing = { ...FRAMING_EXPLORE };
  private readonly pivot = new THREE.Vector3();
  private readonly pivotVel = new THREE.Vector3();
  private boom = FRAMING_EXPLORE.distance;
  private shoulderNow = FRAMING_EXPLORE.shoulder;
  private snapNext = true;
  /** more solids the eye keeps out of, in 3D (the cabin's ceiling, lintel and furniture) */
  solid: ((x: number, y: number, z: number) => boolean) | null = null;

  constructor(private camera: THREE.PerspectiveCamera, private ground: Ground) {}

  onMouse(dx: number, dy: number) {
    this.yaw -= dx * this.sensitivity;
    this.pitch = THREE.MathUtils.clamp(this.pitch + dy * this.sensitivity, PITCH_MIN, PITCH_MAX);
  }

  /** Ease to another framing (fishing views in part 6). */
  setFraming(f: Partial<Framing>) {
    Object.assign(this.target, f);
  }

  /** Jump without smoothing on the next update (spawn, teleports, captures). */
  snap() {
    this.snapNext = true;
  }

  update(dt: number, feet: THREE.Vector3) {
    const fr = this.framing, tg = this.target;
    const kf = this.snapNext || dt <= 0 ? 1 : 1 - Math.exp(-dt * 4);
    fr.distance += (tg.distance - fr.distance) * kf;
    fr.shoulder += (tg.shoulder - fr.shoulder) * kf;
    fr.height += (tg.height - fr.height) * kf;
    fr.fov += (tg.fov - fr.fov) * kf;

    const want = _v.set(feet.x, feet.y + fr.height, feet.z);
    if (this.snapNext || dt <= 0) {
      this.pivot.copy(want);
      this.pivotVel.set(0, 0, 0);
    } else {
      springTo(this.pivot, this.pivotVel, want, dt, 0.09, 0.2);
    }

    const cy = Math.cos(this.pitch), sy = Math.sin(this.pitch);
    const view = _view.set(Math.sin(this.yaw) * cy, -sy, Math.cos(this.yaw) * cy);
    const right = _right.set(-Math.cos(this.yaw), 0, Math.sin(this.yaw));
    // shoulder point: the boom starts here, so the view stays parallel to the character's line of sight; beside a
    // wall the offset shrinks at once and eases back out
    const g = this.ground, px = this.pivot.x, pz = this.pivot.z;
    let sh = fr.shoulder;
    for (let k = 0; k < 4 && sh > 0.02 && g.inBox(px + right.x * sh, pz + right.z * sh, WALL_CLEARANCE); k++) sh *= 0.5;
    if (g.inBox(px + right.x * sh, pz + right.z * sh, WALL_CLEARANCE)) sh = 0;
    if (this.snapNext || dt <= 0 || sh < this.shoulderNow) this.shoulderNow = sh;
    else this.shoulderNow += (sh - this.shoulderNow) * (1 - Math.exp(-dt * 3));
    const shoulder = _s.copy(this.pivot).addScaledVector(right, this.shoulderNow);
    const sGround = this.floorAt(shoulder.x, shoulder.z, CLEARANCE);
    if (shoulder.y < sGround) shoulder.y = sGround;

    // boom: the free length before the eye would dip under the terrain, the water or a deck, or come within
    // WALL_CLEARANCE of a wall (that edge found to ~1 cm)
    let free = fr.distance;
    const steps = 12;
    const at = (t: number) => _p.set(shoulder.x - view.x * t, shoulder.y - view.y * t, shoulder.z - view.z * t);
    for (let i = 1; i <= steps; i++) {
      const t = (fr.distance * i) / steps;
      const p = at(t);
      if (g.inBox(p.x, p.z, WALL_CLEARANCE) || this.solid?.(p.x, p.y, p.z)) {
        let lo = (fr.distance * (i - 1)) / steps, hi = t;
        for (let k = 0; k < 5; k++) {
          const mid = (lo + hi) / 2, q = at(mid);
          if (g.inBox(q.x, q.z, WALL_CLEARANCE) || this.solid?.(q.x, q.y, q.z)) hi = mid;
          else lo = mid;
        }
        free = lo;
        break;
      }
      if (p.y < this.floorAt(p.x, p.z, CLEARANCE)) {
        free = Math.max(0.35, (fr.distance * (i - 1)) / steps);
        break;
      }
    }
    // pull in at once, let out gently; but where the shoulder offset has collapsed (a wall at his right) the eye
    // backs off at once to MIN_EYE from the pivot, or it would sit inside his head while the boom eases out
    if (this.snapNext || dt <= 0 || free < this.boom) this.boom = free;
    else this.boom += (free - this.boom) * (1 - Math.exp(-dt * 3));
    const need = Math.sqrt(Math.max(0, MIN_EYE * MIN_EYE - this.shoulderNow * this.shoulderNow));
    if (this.boom < need) this.boom = Math.min(free, need);
    this.snapNext = false;

    const cam = this.camera;
    cam.position.copy(shoulder).addScaledVector(view, -this.boom);
    const floor = this.floorAt(cam.position.x, cam.position.z, CLEARANCE * 0.6);
    if (cam.position.y < floor) cam.position.y = floor;
    _target.copy(cam.position).add(view);
    cam.up.set(0, 1, 0);
    cam.lookAt(_target);
    if (Math.abs(cam.fov - fr.fov) > 1e-3) {
      cam.fov = fr.fov;
      cam.updateProjectionMatrix();
    }
  }

  /** Lowest the eye may go at (x, z): above the terrain, the water surface and any deck. */
  private floorAt(x: number, z: number, clearance: number) {
    const t = this.ground.terrainY(x, z);
    const deck = this.ground.surfaceAt(x, z, t);
    return Math.max(t + clearance, this.ground.waterY(x, z) + WATER_CLEARANCE, deck ? deck.y + clearance : -Infinity);
  }
}

/** Critically damped spring per axis (smooth time tXZ horizontally, tY vertically). */
function springTo(p: THREE.Vector3, v: THREE.Vector3, target: THREE.Vector3, dt: number, tXZ: number, tY: number) {
  const step = (x: number, vx: number, to: number, t: number): [number, number] => {
    const w = 2 / t, e = w * dt, k = 1 / (1 + e + 0.48 * e * e + 0.235 * e * e * e);
    const c = x - to, tmp = (vx + w * c) * dt;
    return [to + (c + tmp) * k, (vx - w * tmp) * k];
  };
  [p.x, v.x] = step(p.x, v.x, target.x, tXZ);
  [p.y, v.y] = step(p.y, v.y, target.y, tY);
  [p.z, v.z] = step(p.z, v.z, target.z, tXZ);
}

const _v = new THREE.Vector3();
const _view = new THREE.Vector3();
const _right = new THREE.Vector3();
const _s = new THREE.Vector3();
const _p = new THREE.Vector3();
const _target = new THREE.Vector3();
