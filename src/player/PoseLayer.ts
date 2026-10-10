import * as THREE from 'three/webgpu';
import type { Character } from './Character';
import type { SlotName } from './Animator';
import { rotateWorld, setWorldRotation, solveTwoBone } from './IK';
import { GRIP_HAND, RodPoser, bodyFrame, samplePath } from './RodPoses';
import type { RodRig } from './RodRig';
import { SOCKETS } from './CharacterSpec';

type V3 = [number, number, number];
type FingerPose = 'grip' | 'relaxed' | 'fish' | 'reel' | 'open';

/**
 * The procedural layer over the mocap:
 *  - an easy stance: weight on the right leg, the left foot set ahead, hips dropped and rolled, shoulders low and
 *    relaxed, counter-tilted, the head up watching the water (fades out as he walks); across a slope the uphill
 *    foot comes in under him and that knee bends;
 *  - the rod is placed first (RodPoser) and the hands go onto it: the right fist round the cork above the reel with
 *    the thumb along the top of the cork, the left hand hanging by his side while he waits, down by the hip holding
 *    the line through the cast, on the crank while he reels; at the release he stands tall, weight on the front
 *    foot; playing a fish he leans back against it, knees bent, weight on the back foot, elbows up;
 *  - holding a fish: one hand round the wrist of the tail, the other under the belly, at different heights;
 *  - finger poses from the GLB's four finger clips, the head turned to where the line goes or down at the fish.
 */
export class PoseLayer {
  private readonly hand: Record<'l' | 'r', { bone: THREE.Bone; palm: THREE.Vector3; handle: THREE.Vector3; normal: THREE.Vector3; finger: THREE.Vector3 }>;
  private readonly fingers = new Map<FingerPose, Map<string, THREE.Quaternion>>();
  private readonly fingerBones: Record<'l' | 'r', THREE.Bone[]>;
  /** clip ankles before the stance, then the stance's ankle targets (for FootIK) */
  readonly feet = [new THREE.Vector3(), new THREE.Vector3()];
  /** the hold_fish palms after the pose (fish socket): l round the tail, r under the head */
  readonly palms = { l: new THREE.Vector3(), r: new THREE.Vector3(), w: 0 };
  /** length of the fish in his hands (m, with its hold scale): sets the span between the hands */
  fishLength = 0.46;
  /**
   * Releasing a fish (kneel_release): where it is held in the water and the way it faces (upstream), its length; after
   * the pose `pos` follows the hands under it and `w` says how far they hold it.
   */
  readonly release = { target: new THREE.Vector3(), dir: new THREE.Vector3(-1, 0, 0), length: 0.45, pos: new THREE.Vector3(), w: 0 };
  /** a world point to watch (the landing spot, the float, the line's entry), weight 0..1; set each frame by the fishing loop */
  readonly attention = { at: new THREE.Vector3(), w: 0 };
  /** floor height under a foot (set by the player): the stance reads the slope across his feet */
  ground: ((x: number, z: number) => number) | null = null;
  /** how far FootIK turns each knee out (m of pole offset): the uphill knee on a slope */
  readonly kneeOut = [0, 0];

  constructor(private c: Character, private poser: RodPoser) {
    const side = (s: 'l' | 'r') => {
      const bone = c.bone(`hand_${s}`);
      const rest = c.restWorld;
      const q = new THREE.Quaternion().setFromRotationMatrix(rest.get(`hand_${s}`)!);
      const qi = q.clone().invert();
      const at = (n: string) => new THREE.Vector3().setFromMatrixPosition(rest.get(n)!);
      const wrist = at(`hand_${s}`), index = at(`index_01_${s}`), pinky = at(`pinky_01_${s}`), middle = at(`middle_01_${s}`);
      const finger = middle.clone().sub(wrist).normalize();
      const thumbSide = index.clone().sub(pinky).normalize();
      // the palm faces the side the fingers curl to: cross order mirrors between the hands
      const normal = s === 'r' ? new THREE.Vector3().crossVectors(thumbSide, finger) : new THREE.Vector3().crossVectors(finger, thumbSide);
      normal.normalize();
      const sock = s === 'r' ? SOCKETS.gripR : SOCKETS.gripL;
      return {
        bone,
        palm: new THREE.Vector3(...sock.palm),
        // the cork runs through the fist from the little finger to the index knuckle
        handle: thumbSide.clone().applyQuaternion(qi).normalize(),
        normal: normal.applyQuaternion(qi).normalize(),
        finger: finger.applyQuaternion(qi).normalize(),
      };
    };
    this.hand = { l: side('l'), r: side('r') };
    const names = ['thumb', 'index', 'middle', 'ring', 'pinky'].flatMap((f) => ['01', '02', '03'].map((n) => `${f}_${n}`));
    this.fingerBones = { l: names.map((n) => c.bone(`${n}_l`)), r: names.map((n) => c.bone(`${n}_r`)) };
    // open: the bind pose's straight fingers (captured before any clip has played)
    const open = new Map<string, THREE.Quaternion>();
    for (const s of ['l', 'r'] as const) for (const b of this.fingerBones[s]) open.set(b.name, b.quaternion.clone());
    this.fingers.set('open', open);
    const src: Record<Exclude<FingerPose, 'open'>, string> = { grip: 'fingers_grip_rod', relaxed: 'fingers_relaxed', fish: 'fingers_fish_hold', reel: 'fingers_reel' };
    for (const [k, clip] of Object.entries(src) as [FingerPose, string][]) {
      const map = new Map<string, THREE.Quaternion>();
      const cl = c.clips.get(clip);
      if (cl) for (const t of cl.tracks) {
        if (!t.name.endsWith('.quaternion')) continue;
        const v = t.values;
        map.set(t.name.slice(0, t.name.lastIndexOf('.')), new THREE.Quaternion(v[0], v[1], v[2], v[3]).normalize());
      }
      this.fingers.set(k, map);
    }
  }

  /**
   * Before foot IK (bone matrices current): stance and spine. Records the clip's ankles and leaves the stance's
   * ankle targets in `feet`. `moving` 0..1 fades the stance out while walking.
   */
  body(slots: readonly { name: SlotName; w: number; t: number }[], moving: number, yaw: number) {
    const c = this.c;
    bodyFrame(yaw, _R, _F);
    _U.set(0, 1, 0);
    this.feet[0].setFromMatrixPosition(c.bone('foot_l').matrixWorld);
    this.feet[1].setFromMatrixPosition(c.bone('foot_r').matrixWorld);
    let stance = 0, lean = 0, twist = 0, drop = 0, tall = 0, back = 0;
    for (const s of slots) {
      const p = slotBody(s.name, s.t);
      const w = s.w * (s.name === 'loco' ? 1 - moving : 1);
      stance += p.stance * w;
      lean += p.lean * w;
      twist += p.twist * w;
      drop += p.drop * w;
      tall += p.tall * w;
      back += p.back * w;
    }
    if (stance < 1e-3 && Math.abs(lean) < 1e-3 && Math.abs(twist) < 1e-3) return;
    if (stance < 1e-3) {
      // no stance (kneeling): only the spine's lean
      const spine = [c.bone('spine_01'), c.bone('spine_02'), c.bone('spine_03')];
      [0.3, 0.35, 0.35].forEach((sh, i) => rotateWorld(spine[i], _q.setFromAxisAngle(_R, THREE.MathUtils.degToRad(-lean * sh))));
      return;
    }
    const pelvis = c.bone('pelvis');
    // the feet the stance sets (the left ahead and a little out, the right back under the weight); across a slope
    // the uphill foot comes in under him and that knee bends
    this.feet[0].addScaledVector(_F, 0.12 * stance).addScaledVector(_R, -0.035 * stance);
    this.feet[1].addScaledVector(_F, -0.025 * stance);
    let slopeRoll = 0;
    this.kneeOut[0] = this.kneeOut[1] = 0;
    if (this.ground) {
      const hl = this.ground(this.feet[0].x, this.feet[0].z), hr = this.ground(this.feet[1].x, this.feet[1].z);
      const k = Math.min(1, Math.abs(hl - hr) / 0.1) * stance;
      if (k > 1e-3) {
        const ui = hl > hr ? 0 : 1, up = this.feet[ui], dn = this.feet[1 - ui];
        // the uphill foot comes in under its hip and back from the stance's step forward (no lunge)
        _v.setFromMatrixPosition(pelvis.matrixWorld).sub(up);
        _v.y = 0;
        const d = _v.length();
        if (d > 1e-4) up.addScaledVector(_v, Math.min(0.15 * k, d * 0.85) / d);
        up.addScaledVector(_F, -0.05 * k);
        // the downhill foot a little in as well: a narrower stance across the slope
        _v.setFromMatrixPosition(pelvis.matrixWorld).sub(dn);
        _v.y = 0;
        const dd = _v.length();
        if (dd > 1e-4) dn.addScaledVector(_v, Math.min(0.05 * k, dd * 0.4) / dd);
        // that knee bends forward and out over the toes, so the bend reads from the front as well; the uphill hip
        // rides higher (the pelvis rolls with the slope)
        this.kneeOut[ui] = 0.22 * k;
        slopeRoll = (ui === 0 ? 6 : -6) * k;
      }
    }
    // hips: toward the weight (right) foot, down, the free (left) hip dropped, turned a little; at the cast's release
    // forward over the front foot and up (standing tall); braced against a fish back over the right foot
    _qp.setFromRotationMatrix(pelvis.parent!.matrixWorld).invert();
    _v.copy(_R).multiplyScalar(0.03 * stance - 0.045 * tall + 0.015 * back);
    _v.addScaledVector(_F, 0.07 * tall - 0.045 * back);
    _v.y -= (0.024 + drop) * stance - 0.02 * tall;
    pelvis.position.add(_v.applyQuaternion(_qp));
    pelvis.updateMatrixWorld(true);
    rotateWorld(pelvis, _q.setFromAxisAngle(_F, THREE.MathUtils.degToRad(-4.5 * stance + slopeRoll))
      .multiply(_q2.setFromAxisAngle(_U, THREE.MathUtils.degToRad(-5 * stance)))
      .multiply(_q3.setFromAxisAngle(_R, THREE.MathUtils.degToRad(3 * tall))));
    // the upper body counters the hips (shoulders tilt the other way), leans and twists through the spine; at the
    // release the mocap's rounded back straightens
    const spine = [c.bone('spine_01'), c.bone('spine_02'), c.bone('spine_03')];
    const share = [0.3, 0.35, 0.35];
    spine.forEach((b, i) => {
      const roll = THREE.MathUtils.degToRad(i === 0 ? 5 * stance : 0);
      const pitch = THREE.MathUtils.degToRad((lean + 3 * stance - 6 * tall) * share[i]);
      const yawS = THREE.MathUtils.degToRad((twist + 6 * stance) * share[i]);
      _q.setFromAxisAngle(_F, roll).multiply(_q2.setFromAxisAngle(_R, -pitch)).premultiply(_q3.setFromAxisAngle(_U, -yawS));
      rotateWorld(b, _q);
    });
    // shoulders low and easy: dropped a little, barely rounded (round 2's forward roll rounded his back)
    for (const s of ['l', 'r'] as const) {
      const sg = s === 'l' ? -1 : 1;
      rotateWorld(c.bone(`clavicle_${s}`), _q.setFromAxisAngle(_U, THREE.MathUtils.degToRad(2 * stance * sg))
        .multiply(_q2.setFromAxisAngle(_F, THREE.MathUtils.degToRad(-3.5 * stance * sg))));
    }
    // the head tipped a little toward the weight side, and pulled back over the shoulders at the release
    rotateWorld(c.bone('neck_01'), _q.setFromAxisAngle(_F, THREE.MathUtils.degToRad(4 * stance))
      .multiply(_q2.setFromAxisAngle(_R, THREE.MathUtils.degToRad(4 * tall))));
  }

  /**
   * After the rod is placed: hands onto the rod or the fish, fingers, head. `phase` is the walk cycle (0..1),
   * `origin` the feet.
   */
  arms(slots: readonly { name: SlotName; w: number; t: number }[], rodLayer: number, moving: number, phase: number, yaw: number, origin: THREE.Vector3, rod: RodRig) {
    bodyFrame(yaw, _R, _F);
    _U.set(0, 1, 0);
    const held = RodPoser.heldWeight(slots, rodLayer);
    const reelW = RodPoser.reelWeight(slots);
    // right fist on the cork, the rod's axis through it, palm toward his left
    if (held > 1e-3) {
      _rodY.setFromMatrixColumn(rod.mesh.matrixWorld, 1).normalize();
      _target.set(0, GRIP_HAND, 0).applyMatrix4(rod.mesh.matrixWorld);
      _n.copy(_R).negate().addScaledVector(_U, -0.25);
      this.orient('r', _rodY, _n, true, _qh);
      this.reach('r', _target, _qh, held, 'grip', reelW);
    }
    // left hand: per slot target, blended
    let wl = 0;
    _lp.set(0, 0, 0);
    _lq.set(0, 0, 0, 0);
    let crank = 0;
    let fishW = 0;
    for (const s of slots) {
      const w = s.w;
      if (w < 1e-4) continue;
      let mode = leftMode(s.name, rodLayer);
      if (s.name === 'loco') mode = rodLayer > 0.5 ? 'hang' : 'none';
      if (mode === 'none') continue;
      if (mode === 'crank') {
        crank += w;
        continue;
      }
      if (mode === 'fish') {
        fishW += w;
        continue;
      }
      const p = mode === 'line' ? samplePath(CAST_LINE, s.name === 'cast' ? s.t : 0, _k) : HANG;
      const swing = mode === 'hang' ? Math.sin(phase * Math.PI * 2) * 0.1 * moving : 0;
      this.poser.bodyPoint(origin, yaw, p[0], p[1], p[2] + swing, _v);
      // palm toward the thigh, fingers down and a little forward
      _f.copy(_U).negate().addScaledVector(_F, mode === 'line' ? 0.45 : 0.2).normalize();
      _n.copy(_R).addScaledVector(_F, -0.2).normalize();
      this.orient('l', _f, _n, false, _qh);
      _lp.addScaledVector(_v, w);
      addQuat(_lq, _qh, w);
      wl += w;
    }
    if (wl > 1e-3) {
      _lp.multiplyScalar(1 / wl);
      _qh.set(_lq.x, _lq.y, _lq.z, _lq.w).normalize();
      this.reach('l', _lp, _qh, Math.min(1, wl), 'l-free', 0);
    }
    if (crank > 1e-3) {
      // left fingers on the crank knob, palm facing the reel, fingers along the rod
      rod.knobWorld(_target);
      _rodX.setFromMatrixColumn(rod.mesh.matrixWorld, 0).normalize();
      _rodY.setFromMatrixColumn(rod.mesh.matrixWorld, 1).normalize();
      _n.copy(_rodX).negate();
      this.orient('l', _rodY, _n, false, _qh);
      // the knob sits between thumb and index, 3 cm past the palm centre
      _target.addScaledVector(_v.copy(this.hand.l.finger).applyQuaternion(_qh), -0.03);
      this.reach('l', _target, _qh, crank, 'crank', reelW);
    }
    this.palms.w = fishW;
    if (fishW > 1e-3) {
      // shown off level in front of his chest, side-on: the left fist round the wrist of the tail, the right hand
      // cradling it under the throat behind the gills, so the head sticks out past the fingers (the rod is tucked
      // under that arm). The span between the hands follows the fish (tail wrist at 0.77 of its length from the
      // snout, the hand at 0.29); the head hand is lower: the fish lies on its palm, the fist closes round the wrist
      const span = 0.48 * this.fishLength;
      this.poser.bodyPoint(origin, yaw, -0.02 - span / 2, 1.357, 0.38, this.palms.l);
      this.poser.bodyPoint(origin, yaw, -0.02 + span / 2, 1.375 - 0.012 - 0.1 * this.fishLength, 0.4, this.palms.r);
      // head hand: palm up under the belly, fingers forward round the near flank, the wrist taking the weight
      _f.copy(_F).addScaledVector(_R, -0.25).addScaledVector(_U, -0.05).normalize();
      _n.copy(_U).addScaledVector(_F, -0.15).normalize();
      this.orient('r', _f, _n, false, _qh);
      this.reach('r', this.palms.r, _qh, fishW, 'tuck', 0);
      // tail hand: a fist round the tail, the fish's axis through it toward the head, palm down onto the tail
      _f.copy(this.palms.r).sub(this.palms.l).normalize();
      _n.copy(_U).negate().addScaledVector(_F, -0.35).normalize();
      this.orient('l', _f, _n, true, _qh);
      this.reach('l', this.palms.l, _qh, fishW, 'fish', 0);
    }
    // releasing: both hands under the fish in the water, one under the head, one cupping the tail
    let relW = 0;
    for (const sl of slots) {
      if (sl.name === 'kneel') relW += sl.w * sl.w;
      else if (sl.name === 'kneel_release') relW += sl.w * smoothstepK(0.55, 1.2, sl.t) * (1 - smoothstepK(3.5, 4.1, sl.t));
    }
    this.release.w = relW;
    if (relW > 1e-3) {
      const r = this.release, L = r.length;
      const headSide: 'l' | 'r' = r.dir.dot(_R) > 0 ? 'r' : 'l';
      const tailSide: 'l' | 'r' = headSide === 'r' ? 'l' : 'r';
      _hp.copy(r.target).addScaledVector(r.dir, 0.2 * L);
      _hp.y -= 0.045 * L / 0.45;
      _tp.copy(r.target).addScaledVector(r.dir, -0.34 * L);
      _tp.y -= 0.025 * L / 0.45;
      // palms up, fingers pointing out over the water under the fish
      _n.copy(_U).addScaledVector(_F, -0.2).normalize();
      for (const [side, at] of [[headSide, _hp], [tailSide, _tp]] as const) {
        _f.copy(_F).addScaledVector(r.dir, side === headSide ? 0.25 : -0.25).normalize();
        this.orient(side, _f, _n, false, _qh);
        this.reach(side, at, _qh, relW, 'release', 0);
      }
      // the fish stays down in the water where it is held (it follows the hands a little where they fall short)
      this.c.palm(headSide, _a).sub(_hp);
      this.c.palm(tailSide, _b).sub(_tp);
      r.pos.copy(r.target).addScaledVector(_a.add(_b), 0.08 * Math.min(1, relW));
    }
    // fingers
    const fr = new Map<FingerPose, number>(), fl = new Map<FingerPose, number>();
    const add = (m: Map<FingerPose, number>, k: FingerPose, w: number) => m.set(k, (m.get(k) ?? 0) + w);
    add(fr, 'grip', held);
    // the head hand cups (fingers curled up a little), the tail hand closes; both cup under a fish being released
    add(fr, 'relaxed', fishW * 0.72 + relW * 0.6 + Math.max(0, 1 - held - fishW - relW) * 0.5);
    add(fr, 'open', fishW * 0.28 + relW * 0.4);
    add(fl, 'relaxed', wl * 0.7 + relW * 0.6 + Math.max(0, 1 - wl - fishW - crank - relW) * 0.4);
    add(fl, 'open', relW * 0.4);
    add(fl, 'grip', wl * 0.3 + crank * 0.45 + fishW);
    add(fl, 'reel', crank * 0.55);
    this.applyFingers('r', fr);
    this.applyFingers('l', fl);
    // the thumb lies along the top of the cork, pointing up the rod
    if (held > 1e-3) this.thumbOnCork(rod, held);
    // head: to where the line goes, or down at the fish
    let lw = 0;
    _lp.set(0, 0, 0);
    for (const s of slots) {
      const lk = slotLook(s.name);
      if (!lk || s.w < 1e-4) continue;
      const w = s.w * lk[3];
      if (s.name === 'land_lift') _v.copy(this.palms.l).lerp(this.palms.r, 0.5).addScaledVector(_F, 0.25);
      else if (s.name === 'kneel_release' || s.name === 'kneel') _v.copy(this.release.pos);
      else {
        _v.copy(origin).addScaledVector(_R, lk[0]).addScaledVector(_F, lk[2]);
        _v.y += lk[1];
      }
      _lp.addScaledVector(_v, w);
      lw += w;
    }
    if (lw > 1e-3) _lp.multiplyScalar(1 / lw);
    // eyes on what the fishing is about: the spot he casts to, the float, the line
    const att = this.attention;
    if (att.w > 1e-3) {
      if (lw > 1e-3) _lp.lerp(att.at, att.w);
      else _lp.copy(att.at);
      lw = Math.max(lw, att.w * 0.9);
    }
    if (lw > 1e-3) this.look(_lp, Math.min(1, lw));
  }

  /** Where the fish goes back into the water and the way it faces (the current), its length (m). */
  setRelease(target: THREE.Vector3, upstream: THREE.Vector3, length: number) {
    this.release.target.copy(target);
    this.release.dir.set(upstream.x, 0, upstream.z).normalize();
    this.release.length = length;
    this.release.pos.copy(target);
  }

  /** Hand world rotation that points hand-local `axis` (handle or finger) along `dir` with the palm toward `palmDir`. */
  private orient(s: 'l' | 'r', dir: THREE.Vector3, palmDir: THREE.Vector3, handle: boolean, out: THREE.Quaternion) {
    const h = this.hand[s];
    const a = handle ? h.handle : h.finger;
    // local frame (a, n', a x n') -> world frame (dir, palm', dir x palm')
    _la.copy(a);
    _ln.copy(h.normal).addScaledVector(_la, -h.normal.dot(_la)).normalize();
    _lb.crossVectors(_la, _ln);
    _wa.copy(dir).normalize();
    _wn.copy(palmDir).addScaledVector(_wa, -palmDir.dot(_wa)).normalize();
    _wb.crossVectors(_wa, _wn);
    _m1.makeBasis(_la, _ln, _lb);
    _m2.makeBasis(_wa, _wn, _wb);
    return out.setFromRotationMatrix(_m2.multiply(_m1.transpose()));
  }

  /**
   * Two-bone IK so the palm lands on `palmTarget` with the hand at `q`; elbows hang down and out, and lift up and
   * out (`lift` 0..1) when he braces against a fish.
   */
  private reach(s: 'l' | 'r', palmTarget: THREE.Vector3, q: THREE.Quaternion, w: number, kind: 'grip' | 'l-free' | 'crank' | 'fish' | 'tuck' | 'release', lift: number) {
    const c = this.c, h = this.hand[s];
    const ua = c.bone(`upperarm_${s}`), la = c.bone(`lowerarm_${s}`), hand = h.bone;
    _qc.setFromRotationMatrix(hand.matrixWorld).slerp(q, Math.min(1, w));
    setWorldRotation(hand, _qc);
    _wr.copy(palmTarget).sub(_v2.copy(h.palm).applyQuaternion(_qc));
    const out = s === 'l' ? -1 : 1;
    _pole.setFromMatrixPosition(ua.matrixWorld).lerp(_wr, 0.5);
    // (the arm with the rod tucked under it keeps its elbow in at his side; reaching into the water the elbows bend
    // out and down)
    const outBy = kind === 'tuck' ? 0.06 : kind === 'release' ? 0.3 : kind === 'fish' ? 0.24 : 0.3;
    _pole.addScaledVector(_R, out * (outBy + 0.25 * lift)).addScaledVector(_U, -0.35 + 0.42 * lift)
      .addScaledVector(_F, kind === 'grip' ? 0.02 : kind === 'tuck' ? -0.12 : -0.06);
    solveTwoBone(ua, la, hand, _wr, _pole, Math.min(1, w));
  }

  /** Aim the right thumb along the top of the cork, toward the tip, over the fist's grip pose. */
  private thumbOnCork(rod: RodRig, w: number) {
    const c = this.c;
    const t1 = c.bone('thumb_01_r'), t2 = c.bone('thumb_02_r'), t3 = c.bone('thumb_03_r');
    _rodY.setFromMatrixColumn(rod.mesh.matrixWorld, 1).normalize();
    _rodZ.setFromMatrixColumn(rod.mesh.matrixWorld, 2).normalize();
    // the cork's top surface (away from the reel), 5 and 8 cm up from the palm
    _target.set(0, GRIP_HAND + 0.05, 0).applyMatrix4(rod.mesh.matrixWorld).addScaledVector(_rodZ, -0.0135);
    this.aim(t1, t2, _target, w);
    _target.set(0, GRIP_HAND + 0.085, 0).applyMatrix4(rod.mesh.matrixWorld).addScaledVector(_rodZ, -0.012);
    this.aim(t2, t3, _target, w);
  }

  /** Turn `bone` (pivot) so its child's pivot lies on the line toward `target` (weight w). */
  private aim(bone: THREE.Bone, child: THREE.Bone, target: THREE.Vector3, w: number) {
    _a.setFromMatrixPosition(bone.matrixWorld);
    _b.setFromMatrixPosition(child.matrixWorld).sub(_a).normalize();
    _d.copy(target).sub(_a).normalize();
    _q.setFromUnitVectors(_b, _d);
    _q.slerpQuaternions(_ident, _q, Math.min(1, w));
    rotateWorld(bone, _q);
  }

  private applyFingers(s: 'l' | 'r', weights: Map<FingerPose, number>) {
    let total = 0;
    for (const v of weights.values()) total += Math.max(0, v);
    if (total < 1e-3) return;
    for (const b of this.fingerBones[s]) {
      _acc.set(0, 0, 0, 0);
      for (const [k, w] of weights) {
        if (w <= 0) continue;
        const q = this.fingers.get(k)?.get(b.name);
        if (q) addQuat(_acc, q, w);
      }
      if (_acc.lengthSq() < 1e-8) continue;
      _q.set(_acc.x, _acc.y, _acc.z, _acc.w).normalize();
      b.quaternion.slerp(_q, Math.min(1, total));
    }
    this.fingerBones[s][0].parent?.updateMatrixWorld(true);
  }

  /** Turn the neck and head toward a world point (up to ~55 degrees). */
  private look(target: THREE.Vector3, w: number) {
    const c = this.c;
    const head = c.bone('head'), neck = c.bone('neck_01');
    for (const [b, share] of [[neck, 0.4], [head, 1.0]] as const) {
      _v.setFromMatrixPosition(head.matrixWorld);
      _f.setFromMatrixColumn(head.matrixWorld, 2).normalize();
      _d.copy(target).sub(_v).normalize();
      _q.setFromUnitVectors(_f, _d);
      const ang = 2 * Math.acos(Math.min(1, Math.abs(_q.w)));
      const lim = THREE.MathUtils.degToRad(55);
      if (ang > lim) _q.slerpQuaternions(_ident, _q, lim / ang);
      _q.slerpQuaternions(_ident, _q, w * share);
      rotateWorld(b, _q);
    }
  }
}

/**
 * per clip: stance amount, forward lean (deg), shoulder twist (deg, + turns the right shoulder back), hip drop (m),
 * standing tall over the front foot (0..1, the cast's release), weight back on the right foot (0..1, a fish pulling)
 */
function slotBody(name: SlotName, t: number): { stance: number; lean: number; twist: number; drop: number; tall: number; back: number } {
  switch (name) {
    // standing easy, knees soft but not crouched
    case 'loco': case 'wait': return { stance: 1, lean: 0, twist: 0, drop: -0.012, tall: 0, back: 0 };
    case 'cast': {
      const k = samplePath(CAST_BODY, t, _kb);
      // drawn back the weight goes onto the back foot; through the release and the follow-through onto the front one
      const back = smoothstepK(1.0, 1.5, t) * (1 - smoothstepK(2.25, 2.45, t)) * 0.8;
      const tall = Math.exp(-(((t - 2.47) / 0.5) ** 2)) + smoothstepK(2.47, 2.7, t) * (1 - smoothstepK(3.2, 3.6, t)) * 0.6;
      return { stance: 1, lean: k[0], twist: k[1], drop: 0.005, tall: Math.min(1, tall), back };
    }
    case 'strike': return { stance: 1, lean: -4, twist: 4, drop: 0.01, tall: 0, back: 0.3 };
    // braced against the pull: leaning back from the hips, knees bent, the weight on the back foot
    case 'reel_loop': return { stance: 1, lean: -15, twist: 8, drop: 0.055, tall: 0, back: 1 };
    // showing the fish: standing up straight, the shoulders a little back
    case 'land_lift': case 'hold_fish': return { stance: 1, lean: -3, twist: 0, drop: 0.012, tall: 0, back: 0 };
    // kneeling at the edge: the mocap's hunched back straightened a little (the hands still reach the water)
    // crouched low at the edge, leaning out over the water to hold the fish in it
    case 'kneel': return { stance: 0, lean: 16, twist: 0, drop: 0, tall: 0, back: 0 };
    case 'kneel_release': return { stance: 0, lean: 11 * smoothstepK(0.5, 1.1, t) * (1 - smoothstepK(3.6, 4.4, t)), twist: 0, drop: 0, tall: 0, back: 0 };
    default: return { stance: 0, lean: 0, twist: 0, drop: 0, tall: 0, back: 0 };
  }
}

function leftMode(name: SlotName, rodLayer: number): 'hang' | 'line' | 'crank' | 'fish' | 'none' {
  // (kneel_release: both hands go to the fish in arms())
  switch (name) {
    case 'loco': return rodLayer > 0.5 ? 'hang' : 'none';
    case 'wait': return 'hang';
    case 'cast': case 'strike': return 'line';
    case 'reel_loop': return 'crank';
    case 'land_lift': case 'hold_fish': return 'fish';
    default: return 'none';
  }
}

/** head look target per clip in the feet frame [right, up, forward] and its weight */
function slotLook(name: SlotName): [number, number, number, number] | null {
  switch (name) {
    // waiting for a rise: eyes on the water a few metres out
    case 'loco': case 'wait': return [0.6, -0.25, 7, 0.85];
    case 'cast': return [0.9, -0.1, 13, 0.9];
    case 'strike': return [0.6, -0.2, 10, 0.6];
    case 'reel_loop': return [1.2, -0.6, 8, 0.75];
    // showing the fish he looks up from it, out toward whoever is watching
    case 'hold_fish': return [-0.3, 1.45, 2.2, 0.75];
    case 'land_lift': return [0, 0, 0, 0.6];
    case 'kneel_release': case 'kneel': return [0, 0, 0, 0.85];
    default: return null;
  }
}

/** left hand hanging easy by his side [right, up, forward], the elbow a little bent */
const HANG: V3 = [-0.22, 0.99, 0.12];
/** the line hand through the cast: low by the hip, a short haul down on the forward stroke */
const CAST_LINE: [number, number, number, number][] = [
  [0.0, -0.2, 0.93, 0.09], [0.8, -0.18, 1.0, 0.2], [1.5, -0.16, 0.95, 0.14], [2.2, -0.17, 0.96, 0.12],
  [2.4, -0.16, 0.9, 0.1], [2.6, -0.2, 0.96, 0.17], [3.5, -0.2, 0.93, 0.09],
];
/** cast body: forward lean (deg) and shoulder twist (deg): opening into the backcast, driving through the forward stroke */
const CAST_BODY: [number, number, number][] = [
  [0.0, 0, 0], [0.8, -1, 4], [1.5, -5, 12], [2.2, -5, 14], [2.47, 2, -8], [2.85, 3, -10], [3.5, 0, 0],
];

function smoothstepK(a: number, b: number, x: number) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

function addQuat(acc: THREE.Vector4, q: THREE.Quaternion, w: number) {
  const sg = acc.x * q.x + acc.y * q.y + acc.z * q.z + acc.w * q.w < 0 ? -1 : 1;
  acc.x += q.x * w * sg;
  acc.y += q.y * w * sg;
  acc.z += q.z * w * sg;
  acc.w += q.w * w * sg;
}

const _R = new THREE.Vector3(), _F = new THREE.Vector3(), _U = new THREE.Vector3(0, 1, 0);
const _v = new THREE.Vector3(), _v2 = new THREE.Vector3(), _d = new THREE.Vector3(), _f = new THREE.Vector3(), _n = new THREE.Vector3();
const _a = new THREE.Vector3(), _b = new THREE.Vector3();
const _target = new THREE.Vector3(), _rodY = new THREE.Vector3(), _rodX = new THREE.Vector3(), _rodZ = new THREE.Vector3(), _wr = new THREE.Vector3(), _pole = new THREE.Vector3();
const _lp = new THREE.Vector3(), _lq = new THREE.Vector4(), _acc = new THREE.Vector4();
const _hp = new THREE.Vector3(), _tp = new THREE.Vector3();
const _la = new THREE.Vector3(), _ln = new THREE.Vector3(), _lb = new THREE.Vector3(), _wa = new THREE.Vector3(), _wn = new THREE.Vector3(), _wb = new THREE.Vector3();
const _m1 = new THREE.Matrix4(), _m2 = new THREE.Matrix4();
const _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion(), _q3 = new THREE.Quaternion(), _qp = new THREE.Quaternion(), _qh = new THREE.Quaternion(), _qc = new THREE.Quaternion();
const _ident = new THREE.Quaternion();
const _k: number[] = [], _kb: number[] = [];
