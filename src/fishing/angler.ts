import * as THREE from 'three/webgpu';
import { shotMode } from '../core/params';
import type { Player } from '../player/Player';
import { CAST_EVENTS } from '../player/CharacterSpec';
import type { CharacterHooks, RodRig } from './types';
import { smoothstep as smooth } from './util';

/**
 * The `cast` clip holds the rod out front until ~0.8 s, lifts it by 1.2 s, holds it back over the shoulder until
 * ~2.2 s and strokes forward to the release at 2.47 s. While the button is held the clip is scrubbed from AIM_FROM
 * toward AIM_TO with the power: the rod goes back past upright at once and on to about one o'clock as the bar fills;
 * the release plays on from there, so the line leaves ~0.3-0.6 s after the button comes up instead of 2.5 s.
 */
export const AIM_FROM = 1.55;
export const AIM_TO = 2.15;
const aimTime = (power: number) => AIM_FROM + (AIM_TO - AIM_FROM) * Math.sqrt(Math.min(1, Math.max(0, power) / 0.75));

const wrapPi = (a: number) => a - Math.round(a / (2 * Math.PI)) * 2 * Math.PI;

/**
 * Releasing a fish (kneel_release) the clip folds him flat over his knees, head below the shoulders. A layer of
 * `wait`'s upright spine, neck and head blended over it (this much of it against the clip's 1) puts his back at a
 * slant with the head up over the fish; the pose layer's lean and the arms' reach to the fish in the water act on top.
 */
const TORSO_W = 1.1;
const TORSO_BONES = new Set(['spine_01', 'spine_02', 'spine_03', 'neck_01', 'head']);
/**
 * Where he put the rod down before wading in, when the pose layer's spot by his heel is under water: on the nearest
 * stones standing clear of the water, `along` m down the edge on his left from straight in toward them (not hidden
 * behind him from in front), the rod lying along the edge (turned inland where the edge bends into the water), the
 * reel on its side.
 */
const LAY = { along: 1.0, reach: 4, clear: 0.01 };
/** rod axis height over the ground where it rests: on the reel's rim at the seat, on the blank elsewhere */
const LAY_CLEAR = (s: number) => (s < 0.12 ? 0.036 : 0.006);

/** What photo-mode poses need of the angler besides the hooks. */
export interface PoseAngler {
  /** stand at (x, z) facing yaw (character convention: +Z toward (sin yaw, cos yaw)), rod in both hands */
  place(x: number, z: number, yaw: number): void;
  /** advance his animation and pose by dt (photo mode passes dt = 0 to the engine updaters) */
  step(dt: number): void;
  /** the gameplay camera behind him: yaw offset from his facing and pitch (rad, + looks down), boom length, fov */
  view(yawOffset: number, pitch: number, distance: number, fov: number): void;
  /** facing (character yaw) */
  readonly yaw: number;
  /** back to the spawn stance (leaving the poses for another photo spot) */
  reset(): void;
}

/**
 * Part 5's angler as the fishing loop's character: `installFishing({ engine, world, assets, ...anglerHooks(player) })`.
 * Wraps `Player.fishingHooks()` (unchanged) with what the loop and the walking controller need between them:
 *  - walk(): the line is in, so he goes back to walking with the rod in hand (the fishing clips lock his feet
 *    through Player.lockWhileFishing; `wait` after every idle() would have kept him rooted for good);
 *  - playAim(): he stops and draws the rod back while the button is held (the cast clip, scrubbed by the power);
 *  - playCast(): the cast clip plays on from there to the release;
 *  - playStrike() / playReel(): the hook set plays out before the reel takes over (the fight asks for the reel every
 *    frame, which cut the strike off at once), and a strike at nothing goes back to waiting, not to reeling;
 *  - face(): turns him on the spot (the loop turns him to the camera's heading while aiming, toward the fish in
 *    the fight);
 *  - the release: the upright torso layer over the kneel, and the rod on dry stones (see TORSO_W and LAY).
 */
export function anglerHooks(player: Player): { character: CharacterHooks; rod: RodRig; angler: PoseAngler } {
  const base = player.fishingHooks();
  // fishingHooks() hands the crank to the loop; photo mode keeps part 5's clip-timed crank (P4) outside the fish poses
  const clipCrank = shotMode;
  player.autoCrank = clipCrank;
  const a = player.anim, ctl = player.controller, ch = player.character;

  const wait = ch.clip('wait');
  const torsoClip = new THREE.AnimationClip('kneel_release#torso', wait.duration,
    wait.tracks.filter((t) => t.name.endsWith('.quaternion') && TORSO_BONES.has(t.name.slice(0, t.name.lastIndexOf('.')))));
  const torso = ch.mixer.clipAction(torsoClip);
  torso.timeScale = 0;
  torso.setEffectiveWeight(0);
  torso.play();
  /** the torso layer follows the kneel's weight and clip time (set before the mixer runs) */
  const syncTorso = () => {
    let w = 0;
    for (const s of a.blend) if (s.name === 'kneel_release') w += s.w * smooth(0.35, 1.0, s.t) * (1 - smooth(3.4, 4.1, s.t));
    torso.setEffectiveWeight(w * TORSO_W);
  };

  const laid = { pos: new THREE.Vector3(), quat: new THREE.Quaternion(), on: false, set: false };
  /** the rod laid on the nearest dry stones (LAY), or nowhere new if its own spot is dry */
  const layDown = () => {
    laid.set = true;
    laid.on = false;
    const g = player.ground;
    const dry = (x: number, z: number) => g.feetY(x, z) > g.waterY(x, z) + LAY.clear;
    const rodPos = player.rod.mesh.position;
    if (dry(rodPos.x, rodPos.z)) return;
    const px = ctl.position.x, pz = ctl.position.z;
    // the nearest dry stones round him: the way in to the bank
    let near = Infinity, nx = 0, nz = 0;
    for (let k = 0; k < 32; k++) {
      const a = (k / 32) * Math.PI * 2, cx = Math.cos(a), cz = Math.sin(a);
      for (let d = 0.4; d < near && d <= LAY.reach; d += 0.05) {
        if (dry(px + cx * d, pz + cz * d)) {
          near = d;
          nx = cx;
          nz = cz;
          break;
        }
      }
    }
    if (near > LAY.reach) return;
    // down the edge on his left
    let ex = -nz, ez = nx;
    if (ex * Math.cos(ctl.yaw) - ez * Math.sin(ctl.yaw) < 0) {
      ex = -ex;
      ez = -ez;
    }
    let bx = px + nx * (near + 0.12) + ex * LAY.along, bz = pz + nz * (near + 0.12) + ez * LAY.along;
    for (let i = 0; i < 30 && !dry(bx, bz); i++) {
      bx += nx * 0.05;
      bz += nz * 0.05;
    }
    // along the edge, turned inland as far as it takes to keep the blank off the water
    let dx = ex, dz = ez;
    for (let turn = 0; turn <= 6; turn++) {
      const a = turn * 0.12, ca = Math.cos(a), sa = Math.sin(a);
      dx = ex * ca + nx * sa;
      dz = ez * ca + nz * sa;
      let ok = true;
      for (let s = 0.3; s <= 2.5 && ok; s += 0.2) ok = dry(bx + dx * s, bz + dz * s);
      if (ok) break;
    }
    // resting on the highest stones under it: the upper hull of the floor along its length, on the hull edge under its
    // balance point near the reel
    const pts: [number, number][] = [];
    for (let s = -0.075; s <= 2.5; s += 0.125) {
      let h = -1e9;
      for (const o of [-0.012, 0, 0.012]) h = Math.max(h, g.feetY(bx + dx * s - dz * o, bz + dz * s + dx * o));
      pts.push([s, h + LAY_CLEAR(s)]);
    }
    const hull: [number, number][] = [];
    for (const p of pts) {
      while (hull.length >= 2) {
        const p0 = hull[hull.length - 2], p1 = hull[hull.length - 1];
        if ((p1[0] - p0[0]) * (p[1] - p0[1]) - (p1[1] - p0[1]) * (p[0] - p0[0]) >= 0) hull.pop();
        else break;
      }
      hull.push(p);
    }
    let k = 0;
    while (k < hull.length - 2 && hull[k + 1][0] < 0.45) k++;
    const h0 = hull[k], h1 = hull[Math.min(hull.length - 1, k + 1)];
    const slope = h1[0] > h0[0] ? (h1[1] - h0[1]) / (h1[0] - h0[0]) : 0;
    _y.set(dx, slope, dz).normalize();
    // the reel on its side, facing away from the water
    _z.set(-dz, 0, dx);
    if (_z.x * nx + _z.z * nz < 0) _z.negate();
    _z.addScaledVector(_y, -_z.dot(_y)).normalize();
    _x.crossVectors(_y, _z);
    laid.quat.setFromRotationMatrix(_m.makeBasis(_x, _y, _z));
    laid.pos.set(bx, h0[1] - slope * h0[0], bz);
    laid.on = true;
  };
  /**
   * After his pose: the rod from the pose layer's spot onto the dry stones, as far as the kneel has taken over. Lying
   * on the ground it is straight (its own sag bowed the blank down through the stones, out of sight).
   */
  const layRod = () => {
    const rod = player.rod;
    let w = 0;
    for (const s of a.blend) if (s.name === 'kneel_release' || s.name === 'kneel') w += s.w;
    if (w < 1e-3) {
      laid.set = false;
      rod.support = 0;
      return;
    }
    if (!laid.set && w > 0.5) layDown();
    const k = smooth(0.5, 1, w);
    rod.support = k;
    if (laid.on) {
      _p.copy(rod.mesh.position).lerp(laid.pos, k);
      _q.copy(rod.mesh.quaternion).slerp(laid.quat, k);
      rod.setPose(_p, _q);
    }
    rod.update(0);
    // the bone matrices go up once a frame, and his self-shadow pass has already sent the pose layer's
    rod.mesh.skeleton.update();
  };

  const character: CharacterHooks = {
    ...base,
    playCast: (power: number) => {
      const rate = 0.9 + 0.25 * Math.min(1, Math.max(0, power));
      const from = a.state === 'cast' ? a.time.t : aimTime(power);
      a.play('cast', { next: 'wait', timeScale: rate, fade: 0.12, startAt: from });
      return Math.max(0.05, (CAST_EVENTS.release - from) / rate);
    },
    playAim: (power: number) => {
      a.play('cast', { next: 'wait', timeScale: 0, fade: 0.4, startAt: aimTime(power) });
    },
    playStrike: () => a.play('strike', { next: 'wait', fade: 0.12 }),
    playReel: (reeling: boolean, tension: number) => {
      if (a.state !== 'strike') base.playReel(reeling, tension);
    },
    walk: () => {
      // (up out of the crouch at the water's edge slowly)
      if (a.state !== 'loco') a.stop(a.state === 'kneel' || a.state === 'kneel_release' ? 0.7 : 0.35);
    },
    face: (x: number, z: number, dt: number, rate: number, slack = 0) => {
      if (x * x + z * z < 1e-8 || dt <= 0) return;
      const d = wrapPi(Math.atan2(x, z) - ctl.yaw);
      const over = Math.sign(d) * Math.max(0, Math.abs(d) - slack);
      const step = rate * dt;
      ctl.yaw = wrapPi(ctl.yaw + Math.max(-step, Math.min(step, over)));
    },
    // the loop asks for the aim first thing every frame, after his update: the release's layer and rod go in here
    getAim: (origin: THREE.Vector3, dir: THREE.Vector3) => {
      syncTorso();
      layRod();
      base.getAim(origin, dir);
    },
  };
  const angler: PoseAngler = {
    place: (x, z, yaw) => {
      player.autoCrank = false;
      ctl.teleport(x, z, yaw);
      a.setPose('wait', 0, { rodHeld: true });
      torso.setEffectiveWeight(0);
      laid.set = false;
      player.rod.support = 0;
      player.rod.resetDynamics();
      player.footIK.reset();
      player.setHidden(false);
      player.update(0);
    },
    step: (dt) => {
      syncTorso();
      player.update(dt, false);
    },
    view: (yawOffset, pitch, distance, fov) => {
      const cam = player.camera;
      cam.yaw = ctl.yaw + yawOffset;
      cam.pitch = pitch;
      cam.setFraming({ distance, fov });
      cam.snap();
      cam.update(0, ctl.position.clone().setY(ctl.rootY));
    },
    get yaw() {
      return ctl.yaw;
    },
    reset: () => {
      torso.setEffectiveWeight(0);
      laid.set = false;
      player.rod.support = 0;
      player.resetToSpawn();
      player.autoCrank = clipCrank;
    },
  };
  return { character, rod: player.rod, angler };
}

const _x = new THREE.Vector3(), _y = new THREE.Vector3(), _z = new THREE.Vector3(), _p = new THREE.Vector3();
const _m = new THREE.Matrix4(), _q = new THREE.Quaternion();
