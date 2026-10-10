import * as THREE from 'three/webgpu';
import type { Character } from './Character';

const _A = new THREE.Vector3(), _B = new THREE.Vector3(), _C = new THREE.Vector3(), _T = new THREE.Vector3();
const _B2 = new THREE.Vector3(), _dir = new THREE.Vector3(), _pd = new THREE.Vector3(), _u = new THREE.Vector3(), _w = new THREE.Vector3();
const _q = new THREE.Quaternion(), _qw = new THREE.Quaternion(), _qp = new THREE.Quaternion(), _qe = new THREE.Quaternion();
const _ident = new THREE.Quaternion();
const _up = new THREE.Vector3(0, 1, 0);

const worldPos = (o: THREE.Object3D, out: THREE.Vector3) => out.setFromMatrixPosition(o.matrixWorld);

/** Rotate a bone by a world-space rotation about its own pivot; children follow. Rig bones carry no scale. */
export function rotateWorld(bone: THREE.Object3D, q: THREE.Quaternion) {
  _qw.setFromRotationMatrix(bone.matrixWorld).premultiply(q);
  _qp.setFromRotationMatrix(bone.parent!.matrixWorld).invert();
  bone.quaternion.copy(_qp.multiply(_qw));
  bone.updateMatrixWorld(true);
}

/** Give a bone a world rotation (its parent's matrices must be current). */
export function setWorldRotation(bone: THREE.Object3D, q: THREE.Quaternion) {
  _qp.setFromRotationMatrix(bone.parent!.matrixWorld).invert();
  bone.quaternion.copy(_qp.multiply(q));
  bone.updateMatrixWorld(true);
}

/**
 * Analytic two-bone IK in world space: moves `end`'s pivot to `target` (lerped by weight from where it is), the
 * middle joint bending toward `pole`. The end bone keeps its world rotation. Bone matrices must be current.
 */
export function solveTwoBone(upper: THREE.Object3D, mid: THREE.Object3D, end: THREE.Object3D, target: THREE.Vector3, pole: THREE.Vector3, weight = 1) {
  if (weight <= 1e-4) return;
  worldPos(upper, _A);
  worldPos(mid, _B);
  worldPos(end, _C);
  _qe.setFromRotationMatrix(end.matrixWorld);
  _T.copy(_C).lerp(target, Math.min(1, weight));
  const lab = _A.distanceTo(_B), lbc = _B.distanceTo(_C);
  _dir.subVectors(_T, _A);
  const dist = THREE.MathUtils.clamp(_dir.length(), Math.abs(lab - lbc) + 1e-4, lab + lbc - 1e-4);
  _dir.normalize();
  _pd.subVectors(pole, _A);
  _pd.addScaledVector(_dir, -_pd.dot(_dir));
  if (_pd.lengthSq() < 1e-8) {
    _pd.subVectors(_B, _A);
    _pd.addScaledVector(_dir, -_pd.dot(_dir));
  }
  _pd.normalize();
  const cosA = THREE.MathUtils.clamp((lab * lab + dist * dist - lbc * lbc) / (2 * lab * dist), -1, 1);
  const sinA = Math.sqrt(1 - cosA * cosA);
  _B2.copy(_A).addScaledVector(_dir, cosA * lab).addScaledVector(_pd, sinA * lab);
  _q.setFromUnitVectors(_u.subVectors(_B, _A).normalize(), _w.subVectors(_B2, _A).normalize());
  rotateWorld(upper, _q);
  worldPos(mid, _B);
  worldPos(end, _C);
  _q.setFromUnitVectors(_u.subVectors(_C, _B).normalize(), _w.subVectors(_T, _B).normalize());
  rotateWorld(mid, _q);
  setWorldRotation(end, _qe);
}

const ANKLE_REST = 0.104;
const MAX_FOOT_TILT = THREE.MathUtils.degToRad(25);

/**
 * Feet on uneven ground: each foot moves by the ground height under its ankle relative to the character's root,
 * the hips drop for the lower foot (rise for a dip under the root), legs re-solve with the knees forward, and
 * planted feet tilt to the slope (up to 25 degrees).
 */
export class FootIK {
  enabled = true;
  private offL = 0;
  private offR = 0;
  private hips = 0;
  private readonly sides: { thigh: THREE.Bone; calf: THREE.Bone; foot: THREE.Bone }[];
  private readonly pelvis: THREE.Bone;
  private readonly anim = [new THREE.Vector3(), new THREE.Vector3()];
  private readonly target = new THREE.Vector3();
  private readonly pole = new THREE.Vector3();
  private readonly ground: (x: number, z: number) => number;

  constructor(c: Character, ground: (x: number, z: number) => number) {
    this.ground = ground;
    this.sides = (['l', 'r'] as const).map((s) => ({ thigh: c.bone(`thigh_${s}`), calf: c.bone(`calf_${s}`), foot: c.bone(`foot_${s}`) }));
    this.pelvis = c.bone('pelvis');
  }

  /**
   * After the clips are applied and the root placed; bone matrices current. `forward`: the character's facing.
   * `base`: where each ankle should be before the ground offset (the clip's ankles, moved by the stance); omitted,
   * the ankles stay where the clip put them.
   */
  update(dt: number, rootY: number, forward: THREE.Vector3, base?: readonly THREE.Vector3[], kneeOut?: readonly number[]) {
    if (!this.enabled) return;
    const k = dt > 0 ? 1 - Math.exp(-dt * 18) : 1;
    const offs = [0, 0];
    this.sides.forEach((side, i) => {
      const a = base ? this.anim[i].copy(base[i]) : worldPos(side.foot, this.anim[i]);
      offs[i] = THREE.MathUtils.clamp(this.ground(a.x, a.z) - rootY, -0.3, 0.35);
    });
    this.offL += (offs[0] - this.offL) * k;
    this.offR += (offs[1] - this.offR) * k;
    const hipsTarget = THREE.MathUtils.clamp(Math.min(this.offL, this.offR), -0.3, 0.12);
    this.hips += (hipsTarget - this.hips) * (dt > 0 ? 1 - Math.exp(-dt * 10) : 1);
    if (!base && Math.abs(this.offL) < 0.004 && Math.abs(this.offR) < 0.004 && Math.abs(this.hips) < 0.004) return;

    // hips: a world-space vertical move expressed in the pelvis parent's frame
    _qp.setFromRotationMatrix(this.pelvis.parent!.matrixWorld).invert();
    this.pelvis.position.add(_u.set(0, this.hips, 0).applyQuaternion(_qp));
    this.pelvis.updateMatrixWorld(true);

    const offsSmoothed = [this.offL, this.offR];
    this.sides.forEach((side, i) => {
      const a = this.anim[i];
      const lift = a.y - rootY - ANKLE_REST;
      this.target.set(a.x, a.y + offsSmoothed[i], a.z);
      worldPos(side.calf, this.pole).addScaledVector(forward, 0.4);
      // knee turned out over the toes (the uphill leg on a slope): along his right for the right leg, left for the left
      const out = kneeOut?.[i] ?? 0;
      if (out !== 0) this.pole.add(_w.set(-forward.z, 0, forward.x).multiplyScalar(i === 0 ? -out : out));
      solveTwoBone(side.thigh, side.calf, side.foot, this.target, this.pole, 1);
      // planted foot follows the slope under it
      const contact = 1 - THREE.MathUtils.smoothstep(lift, 0.02, 0.1);
      if (contact > 0.01) {
        const e = 0.15;
        const gx = (this.ground(a.x + e, a.z) - this.ground(a.x - e, a.z)) / (2 * e);
        const gz = (this.ground(a.x, a.z + e) - this.ground(a.x, a.z - e)) / (2 * e);
        _u.set(-gx, 1, -gz).normalize();
        _q.setFromUnitVectors(_up, _u);
        const ang = 2 * Math.acos(Math.min(1, Math.abs(_q.w)));
        if (ang > MAX_FOOT_TILT) _q.slerpQuaternions(_ident, _q, MAX_FOOT_TILT / ang);
        _q.slerpQuaternions(_ident, _q, contact);
        rotateWorld(side.foot, _q);
      }
    });
  }

  /** Snap the smoothing (after a teleport). */
  reset() {
    this.offL = this.offR = this.hips = 0;
  }
}
