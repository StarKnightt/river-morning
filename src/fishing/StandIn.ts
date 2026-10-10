import * as THREE from 'three/webgpu';
import type { HeldFish, RodRig } from './types';
import type { ResolvedHooks } from './FishingSim';
import type { WaterProbe, WaterPoint } from './WaterProbe';
import type { FishingInput } from './FishingInput';
import { TUNE } from './config';
import { clamp, easeInOut } from './util';

const DEG = Math.PI / 180;
const _q1 = new THREE.Quaternion(), _q2 = new THREE.Quaternion(), _e = new THREE.Euler();
const _v = new THREE.Vector3(), _w = new THREE.Vector3();
const AX_X = new THREE.Vector3(1, 0, 0), AX_Y = new THREE.Vector3(0, 1, 0);
/**
 * Rod in the right third of the view, pointing a little to the right: the float lands at the centre of the view
 * and must never sit behind the blank.
 */
const ROD_YAW = -4 * DEG;
const GRIP_X = 0.34;
/** the rod's local +Y is the blank; rolled 180 deg so the reel and guides (+Z) hang below */
const ROLL_FLIP = new THREE.Quaternion().setFromAxisAngle(AX_Y, Math.PI);

type Action = 'idle' | 'aim' | 'cast' | 'strike' | 'reel' | 'hold' | 'release';

/** rod pose in the body frame (feet + yaw): blank pitch above horizontal, yaw toward the screen centre, grip */
interface Pose { phi: number; yaw: number; gx: number; gy: number; gz: number }

/** critically damped spring per pose channel */
class Spring {
  v = 0;
  constructor(public x: number) {}
  step(target: number, omega: number, dt: number) {
    this.v += ((target - this.x) * omega * omega - this.v * 2 * omega) * dt;
    this.x += this.v * dt;
  }
}

/**
 * Character stand-in until part 5 lands: a first-person angler on the bank. It owns the camera (mouse look under
 * pointer lock, Q/E or arrow keys to turn, WASD to walk the bank), holds the rod in the lower right of the view,
 * animates cast, strike, reel, hold and release poses procedurally, and presents the caught fish to the camera.
 * It implements every CharacterHooks query, so the loop is fully playable without a character.
 */
export class StandInAngler implements ResolvedHooks {
  readonly feet = new THREE.Vector3();
  yaw = 0;
  pitch = 0;
  /** drives the camera (false in photo mode until a fishing pose is requested) */
  active = true;
  /** where the line goes (the float or the fish), for the fight pose */
  readonly lineTarget = new THREE.Vector3();
  /** set when the angler was carried to where other code put the camera; the owner clears it */
  moved = false;
  /** where this rig last put the camera */
  private readonly camSet = new THREE.Vector3(Number.NaN, 0, 0);
  private action: Action = 'idle';
  private actionT = 0;
  private aimPower = 0;
  private tension = 0;
  private held: HeldFish | null = null;
  private releaseDelay = 0.65;
  private readonly releasePoint = new THREE.Vector3();
  private readonly holdPos = new THREE.Vector3();
  private readonly holdQuat = new THREE.Quaternion();
  private readonly awayQuat = new THREE.Quaternion();
  private phi = new Spring(24 * DEG);
  private yawOff = new Spring(ROD_YAW);
  private gx = new Spring(GRIP_X);
  private gy = new Spring(1.06);
  private gz = new Spring(-0.3);
  private wp: WaterPoint = { y: 0, flowX: 0, flowZ: 0, speed: 0, depth: 0, groundY: 0, wet: false };

  constructor(
    readonly camera: THREE.PerspectiveCamera,
    readonly rod: RodRig,
    private water: WaterProbe,
    private input: FishingInput | null,
  ) {}

  /** back to the idle rod pose at once (deterministic shot poses) */
  resetPose() {
    this.setAction('idle');
    this.held = null;
    this.phi = new Spring(24 * DEG);
    this.yawOff = new Spring(ROD_YAW);
    this.gx = new Spring(GRIP_X);
    this.gy = new Spring(1.06);
    this.gz = new Spring(-0.3);
  }

  placeAt(x: number, z: number, yawDeg: number, pitchDeg: number) {
    this.feet.set(x, this.water.groundY(x, z), z);
    this.yaw = yawDeg * DEG;
    this.pitch = pitchDeg * DEG;
    this.camSet.x = Number.NaN;
  }

  /** jump to a place looking at a point */
  teleport(feet: THREE.Vector3, look: THREE.Vector3) {
    this.camSet.x = Number.NaN;
    this.feet.set(feet.x, this.water.groundY(feet.x, feet.z), feet.z);
    const dx = look.x - feet.x, dz = look.z - feet.z;
    this.yaw = Math.atan2(-dx, -dz);
    this.pitch = clamp(Math.atan2(look.y - (this.feet.y + TUNE.eyeHeight), Math.hypot(dx, dz)), -60 * DEG, 35 * DEG);
  }

  private adoptCamera() {
    const cam = this.camera;
    cam.getWorldDirection(_v);
    this.feet.set(cam.position.x, this.water.groundY(cam.position.x, cam.position.z), cam.position.z);
    this.yaw = Math.atan2(-_v.x, -_v.z);
    this.pitch = clamp(Math.asin(clamp(_v.y, -1, 1)), -60 * DEG, 35 * DEG);
    this.moved = true;
  }

  /** body forward on the ground (unit) */
  forward(out: THREE.Vector3) {
    return out.set(-Math.sin(this.yaw), 0, -Math.cos(this.yaw));
  }

  /** camera look/walk (live only) and the procedural rod pose; dt = 0 still places camera and rod */
  update(dt: number, live: boolean) {
    const inp = this.input;
    // something else moved the camera since the last frame (a debug camera, a jump to another place): stand there
    if (this.active && live && this.camera.position.distanceToSquared(this.camSet) > 1e-4) this.adoptCamera();
    if (this.active && live && inp && dt > 0) {
      const sens = 0.0022;
      this.yaw -= inp.lookX * sens;
      this.pitch -= inp.lookY * sens;
      const k = inp.keys;
      const turn = (k.KeyQ || k.ArrowLeft ? 1 : 0) - (k.KeyE || k.ArrowRight ? 1 : 0);
      this.yaw += turn * 1.5 * dt;
      this.pitch += ((k.ArrowUp ? 1 : 0) - (k.ArrowDown ? 1 : 0)) * 0.9 * dt;
      const fwd = (k.KeyW ? 1 : 0) - (k.KeyS ? 1 : 0), side = (k.KeyD ? 1 : 0) - (k.KeyA ? 1 : 0);
      if ((fwd || side) && (this.action === 'idle' || this.action === 'reel')) {
        const sp = 1.4 * dt / Math.hypot(fwd, side);
        const s = Math.sin(this.yaw), c = Math.cos(this.yaw);
        const nx = this.feet.x + (-s * fwd + c * side) * sp, nz = this.feet.z + (-c * fwd - s * side) * sp;
        // wade the margins, never into water deeper than a boot
        if (this.water.sample(nx, nz, this.wp).depth < 0.3) {
          this.feet.x = nx;
          this.feet.z = nz;
        }
      }
      this.pitch = clamp(this.pitch, -60 * DEG, 35 * DEG);
    }
    // the release happens at your feet: ease the view down to watch the fish swim off (unless you are looking around)
    if (this.active && this.action === 'release' && dt > 0 && !(inp && (inp.lookX || inp.lookY))) {
      this.pitch += (-24 * DEG - this.pitch) * (1 - Math.exp(-3 * dt));
    }
    this.feet.y = this.water.groundY(this.feet.x, this.feet.z);
    if (this.active) {
      const cam = this.camera;
      cam.position.set(this.feet.x, this.feet.y + TUNE.eyeHeight, this.feet.z);
      cam.rotation.set(this.pitch, this.yaw, 0, 'YXZ');
      cam.updateMatrixWorld();
      this.camSet.copy(cam.position);
    }
    this.actionT += dt;
    this.stepPose(dt);
  }

  private target(out: Pose): Pose {
    const a = this.action, t = this.actionT;
    out.phi = 24 * DEG; out.yaw = ROD_YAW; out.gx = GRIP_X; out.gy = 1.06; out.gz = -0.3;
    if (a === 'aim') {
      const p = this.aimPower;
      out.phi = (24 + 78 * p) * DEG; out.yaw = ROD_YAW - 4 * p * DEG; out.gy = 1.06 + 0.22 * p; out.gz = -0.3 + 0.18 * p;
    } else if (a === 'cast') {
      out.phi = (t < 0.3 ? 30 : 24) * DEG;
    } else if (a === 'strike' && t < 0.9) {
      // a missed strike settles back into the waiting pose; a hooked fish switches to 'reel' after 0.35 s
      out.phi = (t < 0.3 ? 64 : 34) * DEG; out.gy = 1.14; out.gz = -0.34;
    } else if (a === 'reel') {
      // follow the fish left and right, lean into the pull
      this.forward(_v);
      _w.subVectors(this.lineTarget, this.feet).setY(0);
      const ang = Math.atan2(_v.x * _w.z - _v.z * _w.x, _v.x * _w.x + _v.z * _w.z);
      out.phi = (28 - 6 * this.tension) * DEG;
      out.yaw = clamp(ROD_YAW - ang * 0.55, -30 * DEG, 35 * DEG);
      out.gx = GRIP_X - 0.03; out.gy = 1.12; out.gz = -0.36 - 0.06 * this.tension;
    } else if (a === 'hold' || a === 'release') {
      // rod lowered to the right, out of the way of the fish held up to the camera
      out.phi = -12 * DEG; out.yaw = -42 * DEG; out.gx = 0.34; out.gy = 0.95; out.gz = -0.12;
    }
    return out;
  }

  private pose: Pose = { phi: 0, yaw: 0, gx: 0, gy: 0, gz: 0 };

  private stepPose(dt: number) {
    const p = this.target(this.pose);
    const fast = this.action === 'cast' && this.actionT < 0.3 ? 46 : this.action === 'strike' && this.actionT < 0.3 ? 30 : 11;
    if (dt > 0) {
      this.phi.step(p.phi, fast, dt);
      this.yawOff.step(p.yaw, 11, dt);
      this.gx.step(p.gx, 9, dt);
      this.gy.step(p.gy, 9, dt);
      this.gz.step(p.gz, 9, dt);
    }
    const root = this.rod.root;
    this.bodyToWorld(_v.set(this.gx.x, this.gy.x, this.gz.x), root.position);
    _q1.setFromAxisAngle(AX_Y, this.yaw + this.yawOff.x);
    _q2.setFromAxisAngle(AX_X, -(Math.PI / 2 - this.phi.x));
    root.quaternion.copy(_q1).multiply(_q2).multiply(ROLL_FLIP);
    root.updateMatrixWorld(true);
  }

  private bodyToWorld(local: THREE.Vector3, out: THREE.Vector3) {
    const s = Math.sin(this.yaw), c = Math.cos(this.yaw);
    return out.set(this.feet.x + local.x * c + local.z * s, this.feet.y + local.y, this.feet.z - local.x * s + local.z * c);
  }

  // --- CharacterHooks ------------------------------------------------------------------------------------------

  playCast(_power: number): number {
    this.setAction('cast');
    return 0.14;
  }

  playAim(power: number) {
    if (this.action !== 'aim') this.setAction('aim');
    this.aimPower = power;
  }

  playStrike() {
    this.setAction('strike');
  }

  playReel(_reeling: boolean, tension: number) {
    if (this.action !== 'reel' && !(this.action === 'strike' && this.actionT < 0.35)) this.setAction('reel');
    this.tension = tension;
  }

  playHold(fish: HeldFish) {
    this.held = fish;
    this.setAction('hold');
  }

  playRelease(fish: HeldFish): number {
    this.held = fish;
    this.setAction('release');
    // the nearest fishable water ahead
    this.forward(_v);
    let found = false;
    for (let d = 0.8; d <= 4; d += 0.2) {
      const x = this.feet.x + _v.x * d, z = this.feet.z + _v.z * d;
      const wp = this.water.sample(x, z, this.wp);
      if (wp.depth > 0.15) {
        this.releasePoint.set(x, wp.y - 0.04, z);
        found = true;
        break;
      }
    }
    if (!found) this.releasePoint.set(this.feet.x + _v.x * 2, this.feet.y + 0.05, this.feet.z + _v.z * 2);
    this.computeHold(this.holdPos, this.holdQuat);
    // lowered into the water facing diagonally out (it swims off showing its flank, not its tail)
    _e.set(0, Math.atan2(_v.x, _v.z) + 0.9, 0);
    this.awayQuat.setFromEuler(_e);
    this.releaseDelay = 0.65;
    return this.releaseDelay;
  }

  idle() {
    if (this.action !== 'cast' || this.actionT > 0.3) this.setAction('idle');
    this.held = null;
  }

  getAim(origin: THREE.Vector3, dir: THREE.Vector3) {
    this.forward(dir);
    // the tip at the end of the forward stroke (phi 30 deg, the idle yaw offset)
    const phi = 30 * DEG, yo = ROD_YAW, L = 2.53;
    _w.set(GRIP_X - Math.sin(yo) * Math.cos(phi) * L, 1.06 + Math.sin(phi) * L, -0.3 - Math.cos(yo) * Math.cos(phi) * L);
    this.bodyToWorld(_w, origin);
  }

  getFeet(out: THREE.Vector3) {
    return out.copy(this.feet);
  }

  getFishHold(pos: THREE.Vector3, quat: THREE.Quaternion): boolean {
    if (this.action === 'release' && this.held) {
      const k = easeInOut(clamp(this.actionT / this.releaseDelay, 0, 1));
      pos.lerpVectors(this.holdPos, this.releasePoint, k);
      quat.slerpQuaternions(this.holdQuat, this.awayQuat, k);
      return true;
    }
    this.computeHold(pos, quat);
    return true;
  }

  /** the fish held up in front of the camera, painted left flank toward the viewer, nose to the left */
  private computeHold(pos: THREE.Vector3, quat: THREE.Quaternion) {
    const len = this.held ? this.held.lengthCm / 100 : 0.4;
    const t = this.actionT;
    pos.set(0.03, -0.16 + Math.sin(t * 1.7) * 0.006, -(0.42 + len * 0.75));
    this.camera.updateMatrixWorld();
    pos.applyMatrix4(this.camera.matrixWorld);
    _e.set(6 * DEG + Math.sin(t * 1.1) * 0.02, -90 * DEG + 16 * DEG, 0, 'YXZ');
    _q1.setFromEuler(_e);
    quat.copy(this.camera.quaternion).multiply(_q1);
  }

  private setAction(a: Action) {
    this.action = a;
    this.actionT = 0;
  }
}
