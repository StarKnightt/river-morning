import * as THREE from 'three/webgpu';
import '../ui/rod-prompt.css';
import type { Animator } from './Animator';
import type { Ground, GroundPoint } from './Ground';
import type { PlayerController } from './PlayerController';
import type { PoseLayer } from './PoseLayer';
import { GRIP_HAND, type RodPoser } from './RodPoses';
import type { RodRig } from './RodRig';

/** Where the rod is, as the fishing loop sees it: in his hands, being put down or picked up, or lying on the ground. */
export type RodPlace = 'hand' | 'busy' | 'laid';

/** What the rod layer needs of the fishing loop (wired in main.ts). */
export interface RodFishing {
  /** the loop's state: the rod can go down from 'idle' (line in) or 'floating' (reeled in first) */
  state(): string;
  /** reel the line in: fresh tackle at the tip on the next frame */
  reelIn(): void;
}

type Phase = 'hand' | 'reeling' | 'laying' | 'standing' | 'laid' | 'seeking' | 'facing' | 'lifting';

/** pick-up range: his feet to the nearest point of the rod (m), and the widest angle off his facing */
const PICK_RANGE = 1.5;
const FACING_COS = Math.cos(THREE.MathUtils.degToRad(70));
/** reel seat to tip-top (m) */
const ROD_LEN = 2.53;
/** the crouch fades in this long; the rod is on the ground (or in his hand) once it is fully in */
const CROUCH_FADE = 0.45;
const SWITCH_AT = 0.55;
const RISE_FADE = 0.6;
/** walking to the rod before he crouches for it gives up after this long (s) */
const SEEK_TIME = 2.5;
const FACE_RATE = 6;
const TOAST_S = 1.6;
/** the rod lying reel side down: the reel's rim props the butt up, the blank lies on the ground toward the tip */
const clearance = (s: number) => (s < -0.034 ? 0.013 : s <= 0.034 ? 0.081 : s <= 0.3 ? 0.0125 : 0.0068 * (1 - (s - 0.3) / 2.23) + 0.0015);
/** the longest gap under the rod (m) and its steepest rise per metre: more reads as floating or sliding off */
const MAX_GAP = 0.1;
const MAX_SLOPE = 0.35;
/** places tried for the reel seat: [right, forward] of his feet (m) and the rod's heading from his facing (deg, + right) */
const SPOTS: readonly (readonly [number, number, number])[] = [
  [0.38, 0.05, 172], [0.38, 0.05, 150], [-0.38, 0.05, -172], [-0.38, 0.05, -150], [0.38, 0.05, -168],
  [-0.38, 0.05, 168], [0.4, 0.25, 120], [-0.4, 0.25, -120], [0.38, 0.1, 25], [-0.38, 0.1, -25], [0.38, 0.1, 0], [-0.38, 0.1, 0],
];

/**
 * The rod put down and picked up again (E). In his hands he can fish; E (line in, nothing on) reels in, crouches and
 * lays the rod on the ground beside him, reel side down, on the terrain or the dock deck, never in the water, a rock or
 * a wall; then he walks and stands hands-free and the fishing buttons do nothing. Within ~1.5 m of the rod and facing
 * it a fading prompt offers it back: E walks him beside it, crouches and stands up with the rod in hand. The rod stays
 * where it lies until then (jumps and time of day leave it there).
 *
 * The crouch is the `kneel` clip; the rod poser's 'laid' socket follows `poser.onGround` while this layer owns it, and
 * `poser.away` puts the walking rod on the ground instead of in the hand or on the back.
 */
export class RodGround {
  phase: Phase = 'hand';
  fishing: RodFishing | null = null;
  /**
   * Something else in reach takes E first (the cabin door): while `prompt()` returns text it shows instead of the
   * rod's, and E goes to `press()` (with the rod in hand or lying down, standing).
   */
  override: { prompt(): string | null; press(): void } | null = null;
  /** the rod went down or came up (handling sound), at the reel seat */
  onHandle: ((kind: 'down' | 'up', at: THREE.Vector3) => void) | null = null;
  private t = 0;
  private pressed = false;
  private readonly seekTo = new THREE.Vector3();
  private faceYaw = 0;
  private readonly prompt: HTMLElement | null;
  private toastLeft = 0;
  private shown = '';
  private otherPrompt: string | null = null;
  private readonly g: GroundPoint = { y: 0, waterY: 0, depth: 0, surface: null };

  constructor(
    private readonly anim: Animator,
    private readonly ctl: PlayerController,
    private readonly ground: Ground,
    private readonly poser: RodPoser,
    private readonly pose: PoseLayer,
    private readonly rod: RodRig,
    hud: HTMLElement | null,
  ) {
    window.addEventListener('keydown', (e) => {
      if (e.code !== 'KeyE' || e.repeat || e.ctrlKey || e.altKey || e.metaKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
      this.pressed = true;
    });
    this.prompt = hud ? document.createElement('div') : null;
    if (this.prompt && hud) {
      this.prompt.className = 'rod-prompt';
      hud.appendChild(this.prompt);
    }
  }

  /** what the fishing loop sees */
  get place(): RodPlace {
    if (this.phase === 'hand') return 'hand';
    // (the line comes back once the rod is up in his hand: fresh tackle at the tip, not dragged up from the ground)
    return this.poser.away > 0.5 || this.phase === 'lifting' ? 'laid' : 'busy';
  }

  /** reel seat and tip direction of the rod on the ground (world), for tests */
  get laidAt(): { x: number; y: number; z: number; yaw: number } | null {
    if (!this.poser.onGround.on) return null;
    const p = this.poser.onGround.pos;
    _y.set(0, 1, 0).applyQuaternion(this.poser.onGround.quat);
    return { x: p.x, y: p.y, z: p.z, yaw: Math.atan2(_y.x, _y.z) };
  }

  /** the pick-up prompt would show (in range and facing the rod) */
  get canPickUp(): boolean {
    return this.phase === 'laid' && this.inReach();
  }

  /** back in his hands at once (spawn, photo spots) */
  reset() {
    this.phase = 'hand';
    this.poser.onGround.on = false;
    this.poser.away = 0;
    this.rod.resting = false;
    this.ctl.seek = null;
    this.pressed = false;
    this.anim.rodHeld = true;
  }

  /** a jump moved him: a rod already down stays where it lies; one still in his hands stays there */
  onTeleport() {
    this.ctl.seek = null;
    if (this.phase === 'reeling' || (this.phase === 'laying' && this.poser.away < 0.5)) {
      this.reset();
      this.anim.stop(0.2);
    } else if (this.phase === 'seeking' || this.phase === 'facing') {
      this.phase = 'laid';
    } else if (this.phase === 'lifting' && this.poser.away > 0.5) {
      this.phase = 'laid';
      this.anim.stop(0.2);
    }
  }

  update(dt: number) {
    const free = (this.phase === 'hand' || this.phase === 'laid') && this.anim.state === 'loco';
    const other = free ? this.override?.prompt() ?? null : null;
    if (other !== null && this.pressed) this.override!.press();
    const pressed = this.pressed && other === null;
    this.pressed = false;
    this.otherPrompt = other;
    this.t += dt;
    const a = this.anim, ctl = this.ctl;
    switch (this.phase) {
      case 'hand':
        if (pressed) this.tryPutDown();
        break;
      case 'reeling': {
        // the loop reels in on its next frame (that also hands him back to walking); then he crouches
        const s = this.fishing?.state() ?? 'idle';
        if (s === 'idle' && a.state === 'loco') this.crouch('laying');
        else if (this.t > 0.6) this.phase = 'hand';
        break;
      }
      case 'laying':
        if (this.poser.away < 0.5 && this.t >= SWITCH_AT) {
          // crouched: the rod is on the ground, his hands free
          this.poser.away = 1;
          a.rodHeld = false;
          a.rodLayer = 0;
          this.rod.resting = true;
          this.onHandle?.('down', this.poser.onGround.pos);
        }
        if (this.t >= SWITCH_AT + 0.12) {
          a.stop(RISE_FADE);
          this.phase = 'standing';
          this.t = 0;
        }
        break;
      case 'standing':
        if (this.t >= RISE_FADE) this.phase = 'laid';
        break;
      case 'laid':
        if (pressed && this.inReach()) {
          this.walkToRod();
        }
        break;
      case 'seeking':
        // the player's own keys take over (the controller drops the seek); he arrives; or it gives up where he is
        if (pressed || ctl.seekCancelled) {
          ctl.seek = null;
          ctl.seekCancelled = false;
          this.phase = 'laid';
        } else if (!ctl.seek || this.t > SEEK_TIME) {
          ctl.seek = null;
          this.phase = 'facing';
          this.t = 0;
        }
        break;
      case 'facing': {
        const d = wrapPi(this.faceYaw - ctl.yaw);
        const step = FACE_RATE * dt;
        ctl.yaw = wrapPi(ctl.yaw + Math.max(-step, Math.min(step, d)));
        if (Math.abs(d) < 0.05 || this.t > 0.5) this.crouch('lifting');
        break;
      }
      case 'lifting':
        if (this.poser.away > 0.5 && this.t >= SWITCH_AT) {
          // crouched by it: the rod in his hand as he stands up
          this.poser.away = 0;
          a.rodHeld = true;
          a.rodLayer = 1;
          this.rod.resting = false;
          this.rod.resetDynamics();
          a.stop(RISE_FADE);
          this.onHandle?.('up', this.poser.onGround.pos);
        }
        if (this.t >= SWITCH_AT + RISE_FADE) {
          this.poser.onGround.on = false;
          this.phase = 'hand';
          this.fishing?.reelIn();
        }
        break;
    }
    this.updatePrompt(dt);
  }

  private crouch(next: 'laying' | 'lifting') {
    // the crouch's hands (the release's reach) go to the cork where it lies, and his eyes with them
    const gp = this.poser.onGround;
    _y.set(0, 1, 0).applyQuaternion(gp.quat);
    _p3.copy(gp.pos).addScaledVector(_y, GRIP_HAND + 0.03);
    this.pose.setRelease(_p3, _y, 0.3);
    this.anim.play('kneel', { fade: CROUCH_FADE });
    this.phase = next;
    this.t = 0;
  }

  private tryPutDown() {
    const s = this.fishing?.state() ?? 'idle';
    const a = this.anim.state;
    if (!(s === 'idle' && a === 'loco') && !(s === 'floating' && a === 'wait')) return;
    if (!this.findSpot()) {
      this.toast("Can't put the rod down here");
      return;
    }
    this.poser.onGround.on = true;
    this.t = 0;
    if (s === 'floating') {
      this.fishing?.reelIn();
      this.phase = 'reeling';
    } else {
      this.crouch('laying');
    }
  }

  /** The rod's pose on the floor beside him (poser.onGround), from the first place in SPOTS that takes it. */
  private findSpot(): boolean {
    const ctl = this.ctl, yaw = ctl.yaw;
    const rx = -Math.cos(yaw), rz = Math.sin(yaw), fx = Math.sin(yaw), fz = Math.cos(yaw);
    for (const [r, f, az] of SPOTS) {
      const h = yaw - THREE.MathUtils.degToRad(az);
      if (this.fit(ctl.position.x + rx * r + fx * f, ctl.position.z + rz * r + fz * f, Math.sin(h), Math.cos(h))) return true;
    }
    return false;
  }

  /**
   * Can the rod lie with its reel seat over (bx, bz) heading (dx, dz)? It rests on the highest points of the floor under
   * it (the upper hull under its balance point, so nothing pokes through), dry or on a deck, clear of rocks, trunks and
   * walls, without a gap under it or a slope it would slide down.
   */
  private fit(bx: number, bz: number, dx: number, dz: number): boolean {
    const g = this.g, pts = _pts;
    pts.length = 0;
    for (let s = -0.075; s <= ROD_LEN + 1e-3; s += 0.1) {
      let hmax = -1e9;
      for (const o of [-0.03, 0, 0.03]) {
        const x = bx + dx * s - dz * o, z = bz + dz * s + dx * o;
        this.ground.sample(x, z, g);
        if (!g.surface && g.depth > -0.03) return false;
        _p.x = x;
        _p.z = z;
        this.ground.resolve(_p, 0.07);
        if (Math.abs(_p.x - x) + Math.abs(_p.z - z) > 1e-6) return false;
        hmax = Math.max(hmax, g.y);
      }
      pts.push([s, hmax + clearance(s), hmax]);
    }
    const hull = _hull;
    hull.length = 0;
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
    const p0 = hull[k], p1 = hull[Math.min(hull.length - 1, k + 1)];
    const slope = p1[0] > p0[0] ? (p1[1] - p0[1]) / (p1[0] - p0[0]) : 0;
    if (Math.abs(slope) > MAX_SLOPE) return false;
    const y0 = p0[1] - slope * p0[0];
    for (const p of pts) if (y0 + slope * p[0] - p[1] > MAX_GAP) return false;
    // +Y along the rod, +Z (the reel) down
    _y.set(dx, slope, dz).normalize();
    _z.set(0, -1, 0).addScaledVector(_y, _y.y).normalize();
    _x.crossVectors(_y, _z);
    this.poser.onGround.quat.setFromRotationMatrix(_m.makeBasis(_x, _y, _z));
    this.poser.onGround.pos.set(bx, y0, bz);
    return true;
  }

  /** Feet to the nearest point of the rod within PICK_RANGE, and that point no more than 70 degrees off his facing. */
  private inReach(): boolean {
    const ctl = this.ctl, gp = this.poser.onGround;
    _y.set(0, 1, 0).applyQuaternion(gp.quat);
    const hx = _y.x, hz = _y.z, hl = Math.hypot(hx, hz) || 1;
    const px = ctl.position.x - gp.pos.x, pz = ctl.position.z - gp.pos.z;
    const s = THREE.MathUtils.clamp((px * hx + pz * hz) / hl, 0, ROD_LEN * hl) / hl;
    const cx = gp.pos.x + hx * s - ctl.position.x, cz = gp.pos.z + hz * s - ctl.position.z;
    const d = Math.hypot(cx, cz);
    if (d > PICK_RANGE) return false;
    return d < 0.35 || (cx * Math.sin(ctl.yaw) + cz * Math.cos(ctl.yaw)) / d > FACING_COS;
  }

  /** Walk beside the grip on his side of the rod, facing it with the grip a little to his right, then crouch. */
  private walkToRod() {
    const gp = this.poser.onGround, ctl = this.ctl;
    _y.set(0, 1, 0).applyQuaternion(gp.quat);
    _p3.copy(gp.pos).addScaledVector(_y, GRIP_HAND);
    const hl = Math.hypot(_y.x, _y.z) || 1;
    let nx = -_y.z / hl, nz = _y.x / hl;
    if ((ctl.position.x - _p3.x) * nx + (ctl.position.z - _p3.z) * nz < 0) {
      nx = -nx;
      nz = -nz;
    }
    this.seekTo.set(_p3.x + nx * 0.42, 0, _p3.z + nz * 0.42);
    this.faceYaw = Math.atan2(_p3.x - this.seekTo.x, _p3.z - this.seekTo.z) + 0.45;
    const far = Math.hypot(this.seekTo.x - ctl.position.x, this.seekTo.z - ctl.position.z) > 0.12;
    ctl.seek = far ? this.seekTo : null;
    this.phase = far ? 'seeking' : 'facing';
    this.t = 0;
  }

  private toast(text: string) {
    this.toastLeft = TOAST_S;
    this.setPrompt(text, 'rod-prompt on toast');
  }

  private updatePrompt(dt: number) {
    if (!this.prompt) return;
    if (this.toastLeft > 0) {
      this.toastLeft -= dt;
      if (this.toastLeft > 0) return;
    }
    if (this.otherPrompt !== null) this.setPrompt(this.otherPrompt, 'rod-prompt on');
    else if (this.canPickUp) this.setPrompt('<kbd>E</kbd> Pick up rod', 'rod-prompt on');
    else this.setPrompt(null, 'rod-prompt');
  }

  private setPrompt(html: string | null, cls: string) {
    const el = this.prompt!;
    if (html !== null && html !== this.shown) {
      this.shown = html;
      el.innerHTML = html;
    }
    if (el.className !== cls) el.className = cls;
  }
}

const wrapPi = (a: number) => a - Math.round(a / (2 * Math.PI)) * 2 * Math.PI;
const _pts: [number, number, number][] = [];
const _hull: [number, number, number][] = [];
const _p = { x: 0, z: 0 };
const _p3 = new THREE.Vector3();
const _x = new THREE.Vector3();
const _y = new THREE.Vector3();
const _z = new THREE.Vector3();
const _m = new THREE.Matrix4();
