import * as THREE from 'three/webgpu';
import type { Ground, GroundPoint } from './Ground';

export const MOVE = {
  /** a bank stroll (walk_slow's native speed is 0.805 m/s) and Shift's brisk walk (walk's 1.498 m/s) */
  walkSpeed: 0.85,
  runSpeed: 1.5,
  accel: 2.6,
  decel: 4.5,
  /** max turn rate (rad/s) and how fast the facing chases the wanted direction (1/s) */
  turnRate: 5.5,
  turnSmooth: 9,
  maxSlopeUp: THREE.MathUtils.degToRad(40),
  maxSlopeDown: THREE.MathUtils.degToRad(55),
  maxStepUp: 0.32,
  maxStepDown: 0.6,
  /** deepest water the angler wades into (m); walk surfaces (the dock) are exempt */
  maxWade: 0.4,
  radius: 0.28,
  /** how far ahead of his centre the boots reach at a deck or step edge (a riser too high stops the boots there) */
  toeReach: 0.24,
};

const wrapPi = (a: number) => a - Math.round(a / (2 * Math.PI)) * 2 * Math.PI;
const SLIDE = [0.6, -0.6, 1.2, -1.2];
/** the boots' edge probes around the direction of a step (radians, ~35 degrees each side) */
const TOE_FAN = [0, 0.6, -0.6];

/**
 * Third-person walking: WASD relative to the camera, Shift walks faster, smooth acceleration and turning; the
 * angler moves where he faces (no strafing), follows the floor, slows uphill and in water, and slides along
 * edges he cannot cross (slopes over 40 degrees, water deeper than 0.4 m, steps over 0.32 m, the play area).
 */
export class PlayerController {
  /** feet position on the floor */
  readonly position = new THREE.Vector3();
  /** facing: the character's +Z points to (sin yaw, 0, cos yaw) */
  yaw = 0;
  /** forward speed being commanded (m/s) */
  speed = 0;
  /** speed actually achieved over the ground (m/s), what the walk cycle follows */
  groundSpeed = 0;
  /** fishing and cut-scenes: no walking (turning with the camera still works for the camera only) */
  movementLocked = false;
  inputEnabled = true;
  /** walk to this point on his own (the rod pick-up); cleared on arrival, or by the walking keys (seekCancelled) */
  seek: { x: number; z: number } | null = null;
  seekCancelled = false;
  /** play area (the fine terrain grid, minus a margin) */
  bounds = { x0: -105, x1: 95, z0: -36, z1: 50 };
  readonly floor: GroundPoint = { y: 0, waterY: 0, depth: 0, surface: null };
  readonly keys = new Set<string>();
  private readonly probe: GroundPoint = { y: 0, waterY: 0, depth: 0, surface: null };
  private readonly toeProbe: GroundPoint = { y: 0, waterY: 0, depth: 0, surface: null };
  private readonly next = { x: 0, z: 0 };
  private visualY = 0;

  constructor(private ground: Ground) {}

  teleport(x: number, z: number, yaw: number) {
    this.ground.sample(x, z, this.floor);
    this.position.set(x, this.floor.y, z);
    this.visualY = this.floor.y;
    this.yaw = yaw;
    this.speed = this.groundSpeed = 0;
  }

  /** Smoothed floor height for the character's root (steps onto the dock ease over ~0.1 s). */
  get rootY() {
    return this.visualY;
  }

  get forward(): THREE.Vector3 {
    return _fwd.set(Math.sin(this.yaw), 0, Math.cos(this.yaw));
  }

  bindKeys(target: Window = window) {
    target.addEventListener('keydown', (e) => {
      if (!e.repeat) this.keys.add(e.code);
    });
    target.addEventListener('keyup', (e) => this.keys.delete(e.code));
    target.addEventListener('blur', () => this.keys.clear());
  }

  update(dt: number, viewYaw: number) {
    if (dt <= 0) return;
    const k = this.keys;
    let f = 0, r = 0;
    if (this.inputEnabled && !this.movementLocked) {
      if (k.has('KeyW') || k.has('ArrowUp')) f += 1;
      if (k.has('KeyS') || k.has('ArrowDown')) f -= 1;
      if (k.has('KeyD') || k.has('ArrowRight')) r += 1;
      if (k.has('KeyA') || k.has('ArrowLeft')) r -= 1;
    }
    // view forward (sin, cos) and right (-cos, sin) in xz
    let dx = Math.sin(viewYaw) * f - Math.cos(viewYaw) * r;
    let dz = Math.cos(viewYaw) * f + Math.sin(viewYaw) * r;
    let ease = 1;
    const sk = this.seek;
    if (sk && !this.movementLocked) {
      if (f !== 0 || r !== 0) {
        this.seek = null;
        this.seekCancelled = true;
      } else {
        const sx = sk.x - this.position.x, sz = sk.z - this.position.z, d = Math.hypot(sx, sz);
        if (d < 0.12) this.seek = null;
        else {
          dx = sx / d;
          dz = sz / d;
          ease = THREE.MathUtils.clamp(d / 0.6, 0.25, 1);
        }
      }
    }
    const mag = Math.hypot(dx, dz);
    let target = 0;
    if (mag > 1e-3) {
      const run = k.has('ShiftLeft') || k.has('ShiftRight');
      target = (run ? MOVE.runSpeed : MOVE.walkSpeed) * ease;
      const dy = wrapPi(Math.atan2(dx, dz) - this.yaw);
      const turn = THREE.MathUtils.clamp(dy * (1 - Math.exp(-MOVE.turnSmooth * dt)), -MOVE.turnRate * dt, MOVE.turnRate * dt);
      this.yaw = wrapPi(this.yaw + turn);
      // sharp reversals turn nearly on the spot instead of walking a wide arc
      target *= THREE.MathUtils.clamp(Math.cos(dy), 0.15, 1);
    }
    const rate = target > this.speed ? MOVE.accel : MOVE.decel;
    this.speed += THREE.MathUtils.clamp(target - this.speed, -rate * dt, rate * dt);

    const fx = Math.sin(this.yaw), fz = Math.cos(this.yaw);
    const p = this.position;
    // slower uphill and in deeper water
    let scale = 1;
    if (!this.floor.surface) {
      const ahead = (this.ground.terrainY(p.x + fx * 0.3, p.z + fz * 0.3) - this.ground.terrainY(p.x, p.z)) / 0.3;
      if (ahead > 0) scale *= 1 - 0.55 * Math.sin(Math.atan(ahead));
      if (this.floor.depth > 0.05) scale *= 1 - 0.45 * THREE.MathUtils.clamp(this.floor.depth / MOVE.maxWade, 0, 1);
    }
    const step = this.speed * scale * dt;
    let moved = false;
    if (step > 1e-5) {
      for (let i = -1; i < SLIDE.length && !moved; i++) {
        const a = i < 0 ? 0 : SLIDE[i];
        const len = step * Math.cos(a);
        const ca = Math.cos(a), sa = Math.sin(a);
        const mx = fx * ca + fz * sa, mz = -fx * sa + fz * ca;
        this.next.x = p.x + mx * len;
        this.next.z = p.z + mz * len;
        this.ground.resolve(this.next, MOVE.radius);
        if (this.walkable(this.next.x, this.next.z, mx, mz)) moved = true;
      }
      if (!moved) this.speed *= Math.exp(-dt * 8);
    }
    const ox = p.x, oz = p.z;
    if (moved) {
      p.x = this.next.x;
      p.z = this.next.z;
    }
    this.ground.sample(p.x, p.z, this.floor);
    p.y = this.floor.y;
    // steps ease, slopes follow exactly (the grid is smooth already)
    const gap = p.y - this.visualY;
    this.visualY = Math.abs(gap) > 0.04 ? this.visualY + gap * (1 - Math.exp(-dt * 16)) : p.y;
    const achieved = Math.hypot(p.x - ox, p.z - oz) / dt;
    this.groundSpeed += (achieved - this.groundSpeed) * (1 - Math.exp(-dt * 14));
  }

  /** Can the feet go to (x, z), moving along (mx, mz)? */
  private walkable(x: number, z: number, mx: number, mz: number): boolean {
    const b = this.bounds;
    if (x < b.x0 || x > b.x1 || z < b.z0 || z > b.z1) return false;
    const g = this.ground.sample(x, z, this.probe);
    const from = this.floor.y;
    // the boots lead: at a deck, porch or step edge he cannot climb (or drop off) they stop at the edge, instead of
    // his centre stopping there with half his body inside the deck's side or out over the drop. A fan of three
    // probes, so a slide along the edge cannot inch him closer to it.
    for (const a of TOE_FAN) {
      const ca = Math.cos(a), sa = Math.sin(a);
      const tx = mx * ca + mz * sa, tz = -mx * sa + mz * ca;
      const toe = this.ground.sample(x + tx * MOVE.toeReach, z + tz * MOVE.toeReach, this.toeProbe);
      if ((toe.surface || this.floor.surface) && (toe.y - from > MOVE.maxStepUp || from - toe.y > MOVE.maxStepDown)) return false;
    }
    if (g.surface || this.floor.surface) {
      // onto, along or off a deck: a step, not a slope
      if (g.y - from > MOVE.maxStepUp || from - g.y > MOVE.maxStepDown) return false;
      return g.surface !== null || g.depth <= MOVE.maxWade;
    }
    if (g.depth > MOVE.maxWade) return false;
    const T = this.ground.terrainY;
    const slope = (T(x + mx * 0.25, z + mz * 0.25) - T(x - mx * 0.1, z - mz * 0.1)) / 0.35;
    return slope <= Math.tan(MOVE.maxSlopeUp) && slope >= -Math.tan(MOVE.maxSlopeDown);
  }
}

const _fwd = new THREE.Vector3();
