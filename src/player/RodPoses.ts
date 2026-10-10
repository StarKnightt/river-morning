import * as THREE from 'three/webgpu';
import type { Character } from './Character';
import type { SlotName } from './Animator';

/** Rod-space height (m above the reel seat) of the hand's palm on the cork grip (cork 0.065-0.296): the reel hangs
 * right under the fist and ~9 cm of cork runs on above it under the thumb. */
export const GRIP_HAND = 0.155;

/** [t s, right, up, forward (m, body frame from the feet), elevation above the horizon (deg, past 90 = behind),
 * azimuth to the right (deg), optionally the swing plane's lean toward the right shoulder (deg, else the path's)] */
type Key = [number, number, number, number, number, number] | [number, number, number, number, number, number, number];
interface Path {
  keys: Key[];
  /** the swing plane leans this far toward the right shoulder (deg) */
  tilt: number;
}

/**
 * The rod goes where an angler holds it, and the hands are put on it afterwards (PoseLayer): the cork in the right
 * fist out in front of the belly while he waits (far enough out that the reel and the lower cork hang clear of the
 * coat), the reel hanging under the rod, the cast a single-handed overhead stroke (lift, pause behind the right
 * shoulder, a short fast forward stroke that stops at the release, 2.47 s, with the casting arm out at shoulder
 * height and forearm and rod in one line, then the follow-through).
 */
const WAIT: Path = { tilt: 8, keys: [[0, 0.15, 1.1, 0.37, 50, 10]] };
const LOCO: Path = { tilt: 8, keys: [[0, 0.15, 1.08, 0.35, 46, 8]] };
export const CAST: Path = {
  tilt: 15,
  keys: [
    [0.0, 0.15, 1.1, 0.37, 50, 10, 15],
    [0.3, 0.16, 1.06, 0.42, 20, 10, 15],
    [0.8, 0.18, 1.2, 0.38, 38, 12, 16],
    [1.2, 0.21, 1.44, 0.24, 80, 14, 22],
    // drawn back: the rod past upright to about one o'clock over his right shoulder, the casting hand up by his ear
    [1.5, 0.2, 1.6, 0.08, 112, 13, 30],
    [1.9, 0.19, 1.64, 0.04, 121, 12, 32],
    [2.22, 0.19, 1.64, 0.05, 121, 11, 32],
    [2.36, 0.21, 1.58, 0.3, 82, 10, 22],
    [2.47, 0.2, 1.5, 0.5, 32, 8, 15],
    // the follow-through, held while the float flies: the arm out toward the target, the tip where the float went
    [2.65, 0.18, 1.46, 0.57, 21, 6, 12],
    [3.0, 0.17, 1.43, 0.6, 17, 5, 12],
    [3.3, 0.16, 1.39, 0.57, 19, 6, 12],
    [3.5, 0.15, 1.27, 0.47, 30, 8, 15],
  ],
};
const STRIKE: Path = {
  tilt: 12,
  keys: [[0.0, 0.15, 1.1, 0.37, 22, 6], [0.25, 0.15, 1.18, 0.36, 42, 6], [0.47, 0.15, 1.3, 0.32, 68, 6], [0.62, 0.15, 1.38, 0.28, 86, 6], [0.93, 0.15, 1.42, 0.24, 95, 6]],
};
/**
 * reel_loop: playing a big fish the rod arm goes up high (the fist on the cork at eye height, the butt clear of the
 * body), the left hand up at the crank of the reel under it (PoseLayer). [right, up, forward] of the butt cap.
 */
export const REEL = { butt: [0.12, 1.38, 0.32] as const, elev: 70, az: 10, tilt: 6 };
/** butt cap below the reel seat (rod space) */
const BUTT_Y = -0.075;
/**
 * Both hands on a fish: the rod tucked under the right arm, the top of the cork clamped in the armpit, grip and reel
 * under the elbow at his side, the rod rising out to his right and a little behind (its line hangs from the tip in a
 * loose curve down to the fish's mouth, off to that side). [right, up, forward] of the clamped point, TUCK_Y up the rod.
 */
const TUCK = { at: [0.215, 1.2, 0.0] as const, elev: 38, az: 122, tilt: 0 };
const TUCK_Y = 0.2;
/**
 * Crouched to release a fish the rod is laid on the stones beside him: reel seat by his right heel, the rod running
 * back up the bar, resting on the highest stones under it. [right, forward] of the reel seat, azimuth (deg).
 */
const LAID = { at: [0.5, -0.55] as const, az: 150 };
/** rod axis height over the ground where it rests: on the reel's rim at the seat, on the blank elsewhere */
const LAID_CLEAR = (s: number) => (s < 0.12 ? 0.036 : 0.006);

type Socket = { kind: 'grip'; path: Path } | { kind: 'reel' } | { kind: 'stow' } | { kind: 'tuck' } | { kind: 'laid' };
const BY_CLIP: Partial<Record<SlotName, Socket>> = {
  wait: { kind: 'grip', path: WAIT },
  cast: { kind: 'grip', path: CAST },
  strike: { kind: 'grip', path: STRIKE },
  reel_loop: { kind: 'reel' },
  land_lift: { kind: 'tuck' },
  hold_fish: { kind: 'tuck' },
  kneel_release: { kind: 'laid' },
  kneel: { kind: 'laid' },
};
const IN_HAND = new Set(['grip', 'reel']);

/** Catmull-Rom through the keys (uniform in key index), so the swing has no corners at the keys. */
export function samplePath(keys: readonly (readonly number[])[], t: number, out: number[] = []): number[] {
  const n = keys[0].length;
  if (keys.length === 1 || t <= keys[0][0]) {
    for (let i = 1; i < n; i++) out[i - 1] = keys[0][i];
    return out;
  }
  if (t >= keys[keys.length - 1][0]) {
    for (let i = 1; i < n; i++) out[i - 1] = keys[keys.length - 1][i];
    return out;
  }
  let i = 0;
  while (t > keys[i + 1][0]) i++;
  const p0 = keys[Math.max(0, i - 1)], p1 = keys[i], p2 = keys[i + 1], p3 = keys[Math.min(keys.length - 1, i + 2)];
  const u = (t - p1[0]) / (p2[0] - p1[0]);
  for (let c = 1; c < n; c++) {
    const a = p0[c], b = p1[c], c2 = p2[c], d = p3[c];
    out[c - 1] = 0.5 * (2 * b + (-a + c2) * u + (2 * a - 5 * b + 4 * c2 - d) * u * u + (-a + 3 * b - 3 * c2 + d) * u * u * u);
  }
  return out;
}

/** Body frame at the feet: right, up, forward for a facing yaw. */
export function bodyFrame(yaw: number, R: THREE.Vector3, F: THREE.Vector3) {
  R.set(-Math.cos(yaw), 0, Math.sin(yaw));
  F.set(Math.sin(yaw), 0, Math.cos(yaw));
}

/** Rod axis (+Y toward the tip) and reel side (+Z, hanging below the rod in its swing plane). */
function rodAxes(elev: number, az: number, tilt: number, R: THREE.Vector3, F: THREE.Vector3, y: THREE.Vector3, z: THREE.Vector3) {
  const er = THREE.MathUtils.degToRad(elev), ar = THREE.MathUtils.degToRad(az), tr = THREE.MathUtils.degToRad(tilt);
  const pf = _pf.copy(F).multiplyScalar(Math.cos(ar)).addScaledVector(R, Math.sin(ar));
  const pu = _pu.set(0, Math.cos(tr), 0).addScaledVector(R, Math.sin(tr));
  y.copy(pf).multiplyScalar(Math.cos(er)).addScaledVector(pu, Math.sin(er)).normalize();
  z.copy(pf).multiplyScalar(Math.sin(er)).addScaledVector(pu, -Math.cos(er));
  z.addScaledVector(y, -z.dot(y)).normalize();
}

/** Rod world transforms for each clip, blended by the animator's slot weights. */
export class RodPoser {
  private readonly stowLocal = new THREE.Matrix4();
  private readonly spine: THREE.Bone;
  /** spine_03's rest position in the feet frame [right, up, forward] */
  private readonly chestRest: THREE.Vector3;
  /** floor height (the player's), for the rod laid on the stones */
  ground: ((x: number, z: number) => number) | null = null;
  /**
   * The rod in hand turned toward a world point (weight 0..1), about the fist: after the cast it follows the float
   * through its flight. Set by the fishing loop each frame.
   */
  readonly aim = { at: new THREE.Vector3(), w: 0, /** the fist moved by this (world) while aiming (the cabin's indoor carry) */ shift: new THREE.Vector3() };
  /** where the laid rod went down (it stays put while he kneels) */
  private readonly laid = { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), set: false };
  /** the rod put down on the ground (RodGround): while `on`, the 'laid' socket (his crouch) is this pose */
  readonly onGround = { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), on: false };
  /** 0..1: the walking stance's rod lies at `ground` instead of in his hand or on his back */
  away = 0;

  constructor(c: Character) {
    this.spine = c.bone('spine_03');
    // slung across the back from the right hip to over the left shoulder, reel facing out, riding on spine_03
    const d = new THREE.Vector3(0.3, 0.93, -0.2).normalize();
    const u = new THREE.Vector3(0, 0, -1);
    u.addScaledVector(d, -u.dot(d)).normalize();
    const x = new THREE.Vector3().crossVectors(d, u);
    const rest = new THREE.Matrix4().makeBasis(x, d, u).setPosition(-0.16, 0.95, -0.21);
    const sp = c.restWorld.get('spine_03')!;
    this.stowLocal.copy(sp).invert().multiply(rest);
    const p = new THREE.Vector3().setFromMatrixPosition(sp);
    this.chestRest = new THREE.Vector3(-p.x, p.y, p.z);
  }

  /** World point of a body-frame offset, carried by the chest's sway (so the rod rides with breathing and the walk). */
  bodyPoint(origin: THREE.Vector3, yaw: number, right: number, up: number, fwd: number, out: THREE.Vector3): THREE.Vector3 {
    bodyFrame(yaw, _R, _F);
    out.copy(origin).addScaledVector(_R, right).addScaledVector(_F, fwd);
    out.y += up;
    // chest displacement from its rest place in the feet frame
    _c.setFromMatrixPosition(this.spine.matrixWorld);
    _d.copy(origin).addScaledVector(_R, this.chestRest.x).addScaledVector(_F, this.chestRest.z);
    _d.y += this.chestRest.y;
    return out.addScaledVector(_c.sub(_d), 0.85);
  }

  private socket(s: Socket, t: number, origin: THREE.Vector3, yaw: number, pos: THREE.Vector3, quat: THREE.Quaternion) {
    if (s.kind === 'stow') {
      _m.multiplyMatrices(this.spine.matrixWorld, this.stowLocal);
      _m.decompose(pos, quat, _s);
      return;
    }
    bodyFrame(yaw, _R, _F);
    if (s.kind === 'laid') {
      if (this.onGround.on) {
        pos.copy(this.onGround.pos);
        quat.copy(this.onGround.quat);
        return;
      }
      if (!this.laid.set) this.layDown(origin);
      pos.copy(this.laid.pos);
      quat.copy(this.laid.quat);
      return;
    }
    if (s.kind === 'tuck') {
      rodAxes(TUCK.elev, TUCK.az, TUCK.tilt, _R, _F, _y, _z);
      _x.crossVectors(_y, _z);
      quat.setFromRotationMatrix(_m.makeBasis(_x, _y, _z));
      this.bodyPoint(origin, yaw, TUCK.at[0], TUCK.at[1], TUCK.at[2], pos).addScaledVector(_y, -TUCK_Y);
      return;
    }
    if (s.kind === 'reel') {
      rodAxes(REEL.elev, REEL.az, REEL.tilt, _R, _F, _y, _z);
      _x.crossVectors(_y, _z);
      quat.setFromRotationMatrix(_m.makeBasis(_x, _y, _z));
      this.bodyPoint(origin, yaw, REEL.butt[0], REEL.butt[1], REEL.butt[2], pos).addScaledVector(_y, -BUTT_Y);
      return;
    }
    const k = samplePath(s.path.keys, t, _k);
    rodAxes(k[3], k[4], s.path.keys[0].length > 6 ? k[5] : s.path.tilt, _R, _F, _y, _z);
    _x.crossVectors(_y, _z);
    quat.setFromRotationMatrix(_m.makeBasis(_x, _y, _z));
    this.bodyPoint(origin, yaw, k[0], k[1], k[2], pos).addScaledVector(_y, -GRIP_HAND);
  }

  /**
   * The rod laid on the bar beside him (body frame at `origin`, the feet), resting on the highest stones under it:
   * the upper hull of the floor along its length, the rod on the hull edge under its balance point near the reel. The
   * reel lies on its side, facing out.
   */
  private layDown(origin: THREE.Vector3) {
    const fx = _F.x, fz = _F.z, rx = _R.x, rz = _R.z;
    const ar = THREE.MathUtils.degToRad(LAID.az);
    const dx = fx * Math.cos(ar) + rx * Math.sin(ar), dz = fz * Math.cos(ar) + rz * Math.sin(ar);
    const bx = origin.x + rx * LAID.at[0] + fx * LAID.at[1], bz = origin.z + rz * LAID.at[0] + fz * LAID.at[1];
    const g = this.ground ?? (() => origin.y);
    const pts: [number, number][] = [];
    for (let s = -0.075; s <= 2.5; s += 0.125) {
      let h = -1e9;
      // the highest stone across the rod's width
      for (const o of [-0.012, 0, 0.012]) h = Math.max(h, g(bx + dx * s - dz * o, bz + dz * s + dx * o));
      pts.push([s, h + LAID_CLEAR(s)]);
    }
    // upper convex hull (monotone chain)
    const hull: [number, number][] = [];
    for (const p of pts) {
      while (hull.length >= 2) {
        const a = hull[hull.length - 2], b = hull[hull.length - 1];
        if ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0) hull.pop();
        else break;
      }
      hull.push(p);
    }
    const cm = 0.45;
    let k = 0;
    while (k < hull.length - 2 && hull[k + 1][0] < cm) k++;
    const a = hull[k], b = hull[Math.min(hull.length - 1, k + 1)];
    const slope = b[0] > a[0] ? (b[1] - a[1]) / (b[0] - a[0]) : 0;
    const y0 = a[1] - slope * a[0];
    _y.set(dx, slope, dz).normalize();
    // the reel on its side, facing away from him
    _z.set(-dz, 0, dx);
    if (_z.x * rx + _z.z * rz < 0) _z.negate();
    _z.addScaledVector(_y, -_z.dot(_y)).normalize();
    _x.crossVectors(_y, _z);
    this.laid.quat.setFromRotationMatrix(_m.makeBasis(_x, _y, _z));
    this.laid.pos.set(bx, y0, bz);
    this.laid.set = true;
  }

  /** Weighted blend over the animator's slots (positions averaged, rotations sign-aligned and normalised). */
  blend(slots: readonly { name: SlotName; w: number; t: number }[], rodLayer: number, origin: THREE.Vector3, yaw: number, outPos: THREE.Vector3, outQuat: THREE.Quaternion) {
    if (!slots.some((sl) => (sl.name === 'kneel' || sl.name === 'kneel_release') && sl.w > 0)) this.laid.set = false;
    outPos.set(0, 0, 0);
    _acc.set(0, 0, 0, 0);
    let total = 0;
    const add = (s: Socket, t: number, w: number) => {
      if (w <= 1e-5) return;
      this.socket(s, t, origin, yaw, _p, _qq);
      outPos.addScaledVector(_p, w);
      const sgn = total > 0 && _acc.x * _qq.x + _acc.y * _qq.y + _acc.z * _qq.z + _acc.w * _qq.w < 0 ? -1 : 1;
      _acc.x += _qq.x * w * sgn;
      _acc.y += _qq.y * w * sgn;
      _acc.z += _qq.z * w * sgn;
      _acc.w += _qq.w * w * sgn;
      total += w;
    };
    for (const sl of slots) {
      if (sl.name === 'loco') {
        const carried = 1 - this.away;
        add({ kind: 'grip', path: LOCO }, 0, sl.w * rodLayer * carried);
        add({ kind: 'stow' }, 0, sl.w * (1 - rodLayer) * carried);
        add({ kind: 'laid' }, 0, sl.w * this.away);
      } else {
        add(BY_CLIP[sl.name] ?? { kind: 'stow' }, sl.t, sl.w);
      }
    }
    if (total <= 0) return;
    outPos.multiplyScalar(1 / total);
    outQuat.set(_acc.x, _acc.y, _acc.z, _acc.w).normalize();
    // turned toward the aim point about the fist on the cork
    const w = this.aim.w * RodPoser.heldWeight(slots, rodLayer);
    if (w > 1e-3) {
      _y.set(0, 1, 0).applyQuaternion(outQuat);
      _p.set(0, GRIP_HAND, 0).applyQuaternion(outQuat).add(outPos).addScaledVector(this.aim.shift, Math.min(1, w));
      _c.copy(this.aim.at).sub(_p).normalize();
      _qq.setFromUnitVectors(_y, _c);
      _qq.slerpQuaternions(_ident, _qq, Math.min(1, w));
      outQuat.premultiply(_qq);
      outPos.copy(_p).addScaledVector(_y.set(0, 1, 0).applyQuaternion(outQuat), -GRIP_HAND);
    }
  }

  /** Weight of the reeling hold in the current blend (drives the crank hand IK). */
  static reelWeight(slots: readonly { name: SlotName; w: number }[]) {
    return slots.find((s) => s.name === 'reel_loop')?.w ?? 0;
  }

  /** Weight of the rod held in the hands (grip or reel sockets). */
  static heldWeight(slots: readonly { name: SlotName; w: number }[], rodLayer: number) {
    let w = 0;
    for (const s of slots) {
      if (s.name === 'loco') w += s.w * rodLayer;
      else if (IN_HAND.has(BY_CLIP[s.name]?.kind ?? '')) w += s.w;
    }
    return w;
  }

  /** Weight of the slung rod (strap across the coat). */
  static stowWeight(slots: readonly { name: SlotName; w: number }[], rodLayer: number) {
    let w = 0;
    for (const s of slots) {
      if (s.name === 'loco') w += s.w * (1 - rodLayer);
      else if (!BY_CLIP[s.name]) w += s.w;
    }
    return w;
  }
}

const _m = new THREE.Matrix4();
const _s = new THREE.Vector3();
const _qq = new THREE.Quaternion();
const _x = new THREE.Vector3();
const _y = new THREE.Vector3();
const _z = new THREE.Vector3();
const _p = new THREE.Vector3();
const _R = new THREE.Vector3();
const _F = new THREE.Vector3();
const _c = new THREE.Vector3();
const _d = new THREE.Vector3();
const _pf = new THREE.Vector3();
const _pu = new THREE.Vector3();
const _k: number[] = [];
const _acc = new THREE.Vector4();
const _ident = new THREE.Quaternion();
