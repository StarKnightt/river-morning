import * as THREE from 'three/webgpu';
import type { Character } from './Character';
import { CAST_EVENTS, LOCO, LOOPING, type ClipName } from './CharacterSpec';

export type SlotName = 'loco' | ClipName;

export interface PlayOptions {
  /** cross-fade time in seconds (default 0.3) */
  fade?: number;
  /** what follows a one-shot clip when it ends (default 'loco'); null holds its last frame */
  next?: SlotName | null;
  /** playback rate (default 1) */
  timeScale?: number;
  /** clip time to start from (s) */
  startAt?: number;
  /** clip events (cast: raise, backcast_top, release, settle) as their time is crossed */
  onEvent?: (name: string) => void;
  /** a one-shot clip reached its end */
  onDone?: () => void;
}

const CLIP_EVENTS: Partial<Record<ClipName, Record<string, number>>> = { cast: CAST_EVENTS };

interface Slot {
  name: SlotName;
  action: THREE.AnimationAction | null;
  w: number;
  t: number;
  duration: number;
  once: boolean;
  rate: number;
  next: SlotName | null;
  done: boolean;
  onEvent?: (name: string) => void;
  onDone?: () => void;
}

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/**
 * Weight-managed blend of the angler's clips. All action times are driven here (the mixer only evaluates), so
 * captures with frozen time are exact and events fire on the frame their time is crossed.
 *
 *  - `loco`: idle / walk_slow / walk blended by speed with the walk phases locked together; with the rod in hand
 *    the upper body (spine_02 up, arms, fingers) comes from `wait` instead of the walk's swinging arms.
 *  - one active slot at a time (`loco` or any clip); switching cross-fades every other slot out.
 */
export class Animator {
  /** ground speed set by the controller each frame (m/s) */
  speed = 0;
  /** rod held in both hands while walking (otherwise the arms swing and the rod is slung on the back) */
  rodHeld = true;
  /** 0..1, follows rodHeld */
  rodLayer = 1;
  /** any clip event, after the slot's own onEvent */
  onEvent: ((clip: SlotName, name: string) => void) | null = null;
  /** a walking foot plants (for footstep sounds and wading rings) */
  onFootfall: ((side: 'l' | 'r') => void) | null = null;

  private readonly slots: Slot[] = [];
  private active: Slot;
  private fadeRate = 1 / 0.3;
  private phase = 0;
  private idleT = 0;
  private moving = 0;
  private walkW = 0;
  private readonly loco: Record<'idleB' | 'idleU' | 'slowB' | 'slowU' | 'walkB' | 'walkU' | 'waitU', THREE.AnimationAction>;
  private readonly actions = new Map<ClipName, THREE.AnimationAction>();

  constructor(readonly character: Character) {
    const mk = (clip: THREE.AnimationClip) => {
      const a = character.mixer.clipAction(clip);
      a.setLoop(THREE.LoopRepeat, Infinity);
      a.timeScale = 0;
      a.setEffectiveWeight(0);
      a.play();
      return a;
    };
    this.loco = {
      idleB: mk(character.clip('idle#base')), idleU: mk(character.clip('idle#upper')),
      slowB: mk(character.clip('walk_slow#base')), slowU: mk(character.clip('walk_slow#upper')),
      walkB: mk(character.clip('walk#base')), walkU: mk(character.clip('walk#upper')),
      waitU: mk(character.clip('wait#upper')),
    };
    this.mkAction = (name) => {
      let a = this.actions.get(name);
      if (!a) {
        a = mk(character.clip(name));
        this.actions.set(name, a);
      }
      return a;
    };
    this.active = this.newSlot('loco', {});
    this.active.w = 1;
    this.slots.push(this.active);
  }

  private readonly mkAction: (name: ClipName) => THREE.AnimationAction;

  private newSlot(name: SlotName, o: PlayOptions): Slot {
    const action = name === 'loco' ? null : this.mkAction(name);
    const duration = action ? action.getClip().duration : 0;
    return {
      name, action, w: 0, t: o.startAt ?? 0, duration, rate: o.timeScale ?? 1,
      once: name !== 'loco' && !LOOPING.has(name), next: o.next === undefined ? 'loco' : o.next, done: false,
      onEvent: o.onEvent, onDone: o.onDone,
    };
  }

  /** Name of the slot being faded in (or held). */
  get state(): SlotName {
    return this.active.name;
  }

  /** 0 standing .. 1 walking (smoothed) */
  get movingAmount() {
    return this.moving;
  }

  /** walk cycle phase 0..1 (walk_slow's footfalls) */
  get walkPhase() {
    return this.phase;
  }

  /** Time in the active clip (s), and its length. */
  get time() {
    return { t: this.active.t, duration: this.active.duration };
  }

  /** Cross-fade to a clip (or back to locomotion with 'loco'). Replaying the active clip restarts it. */
  play(name: SlotName, o: PlayOptions = {}) {
    const fade = Math.max(o.fade ?? 0.3, 1e-3);
    this.fadeRate = 1 / fade;
    let slot = this.slots.find((s) => s.name === name);
    if (slot?.once) {
      // one action per clip: a one-shot that is still fading out restarts from the top
      slot.t = o.startAt ?? 0;
      slot.done = false;
    }
    if (!slot) {
      slot = this.newSlot(name, o);
      // `wait` continues the idle's breathing when it takes over from the rod-in-hand stance
      if (name === 'wait' && o.startAt === undefined) slot.t = this.idleT % slot.duration;
      this.slots.push(slot);
    } else {
      slot.rate = o.timeScale ?? 1;
      slot.next = o.next === undefined ? 'loco' : o.next;
      slot.onEvent = o.onEvent;
      slot.onDone = o.onDone;
      if (o.startAt !== undefined) slot.t = o.startAt;
      slot.done = false;
    }
    this.active = slot;
  }

  /** Back to locomotion. */
  stop(fade = 0.35) {
    this.play('loco', { fade });
  }

  /**
   * Snap to a pose with no fades (frozen captures): one slot at full weight at clip time t; for 'loco', the
   * speed and walk phase (0..1) give the stride.
   */
  setPose(name: SlotName, t: number, o: { speed?: number; phase?: number; rodHeld?: boolean } = {}) {
    for (const s of this.slots) s.action?.setEffectiveWeight(0);
    this.slots.length = 0;
    const slot = this.newSlot(name, { startAt: t, next: null });
    slot.w = 1;
    this.slots.push(slot);
    this.active = slot;
    this.rodHeld = o.rodHeld ?? this.rodHeld;
    this.rodLayer = this.rodHeld ? 1 : 0;
    this.speed = o.speed ?? 0;
    this.phase = o.phase ?? 0;
    this.idleT = t;
    this.moving = smooth(0.08, 0.45, this.speed);
    this.update(0);
  }

  /** Slots with their current weights (sum 1), for blending things that follow the clips (rod socket). */
  get blend(): readonly { name: SlotName; w: number; t: number }[] {
    return this.slots;
  }

  update(dt: number) {
    // slot weights: the active one fades in, the rest out, renormalised so they always sum to one
    let sum = 0;
    for (const s of this.slots) {
      const target = s === this.active ? 1 : 0;
      s.w += Math.sign(target - s.w) * Math.min(Math.abs(target - s.w), this.fadeRate * dt);
      sum += s.w;
    }
    if (sum <= 1e-6) {
      this.active.w = 1;
      sum = 1;
    }
    for (let i = this.slots.length - 1; i >= 0; i--) {
      const s = this.slots[i];
      s.w /= sum;
      if (s !== this.active && s.w <= 1e-4) {
        s.action?.setEffectiveWeight(0);
        this.slots.splice(i, 1);
      }
    }
    this.rodLayer += Math.sign((this.rodHeld ? 1 : 0) - this.rodLayer) * Math.min(Math.abs((this.rodHeld ? 1 : 0) - this.rodLayer), dt / 0.35);

    this.updateLoco(dt);
    for (const s of this.slots) if (s.action) this.advance(s, dt);
    this.character.restorePose();
    this.character.mixer.update(dt);
    this.character.savePose();
  }

  private updateLoco(dt: number) {
    const v = this.speed;
    // idle <-> walking by speed (the controller's acceleration makes this a ~0.2 s blend), walk_slow <-> walk above
    // the slow walk's native speed; the stride rate follows the ground speed so the feet do not slide
    const movingTarget = smooth(0.08, 0.45, v);
    this.moving += (movingTarget - this.moving) * (1 - Math.exp(-dt * 12));
    if (dt === 0) this.moving = movingTarget;
    const w = Math.min(1, Math.max(0, (v - LOCO.walkSlow.speed) / (LOCO.walk.speed - LOCO.walkSlow.speed)));
    this.walkW += (w - this.walkW) * (1 - Math.exp(-dt * 8));
    if (dt === 0) this.walkW = w;
    const cycle = THREE.MathUtils.lerp(LOCO.walkSlow.cycle, LOCO.walk.cycle, this.walkW);
    const native = THREE.MathUtils.lerp(LOCO.walkSlow.speed, LOCO.walk.speed, this.walkW);
    const rate = THREE.MathUtils.clamp(v / native, 0.55, 1.35);
    const prev = this.phase;
    this.phase = (this.phase + (dt * rate) / cycle) % 1;
    if (this.onFootfall && this.moving > 0.5 && dt > 0) {
      const crossed = (p: number) => (prev <= p && this.phase > p) || (prev > this.phase && (p > prev || p <= this.phase));
      if (crossed(LOCO.footfallL)) this.onFootfall('l');
      if (crossed(LOCO.footfallR)) this.onFootfall('r');
    }
    this.idleT += dt;

    const L = this.loco;
    const W = this.slots.find((s) => s.name === 'loco')?.w ?? 0;
    const aIdle = 1 - this.moving, aSlow = this.moving * (1 - this.walkW), aWalk = this.moving * this.walkW;
    const r = this.rodLayer;
    L.idleB.setEffectiveWeight(W * aIdle);
    L.idleU.setEffectiveWeight(W * aIdle * (1 - r));
    L.slowB.setEffectiveWeight(W * aSlow);
    L.slowU.setEffectiveWeight(W * aSlow * (1 - r));
    L.walkB.setEffectiveWeight(W * aWalk);
    L.walkU.setEffectiveWeight(W * aWalk * (1 - r));
    L.waitU.setEffectiveWeight(W * r);
    const idleDur = L.idleB.getClip().duration;
    L.idleB.time = L.idleU.time = this.idleT % idleDur;
    L.waitU.time = this.idleT % L.waitU.getClip().duration;
    L.slowB.time = L.slowU.time = this.phase * LOCO.walkSlow.cycle * 0.99999;
    L.walkB.time = L.walkU.time = ((this.phase + LOCO.walkPhaseOffset) % 1) * LOCO.walk.cycle * 0.99999;
  }

  private advance(s: Slot, dt: number) {
    const prev = s.t;
    s.t += dt * s.rate;
    if (s.once) {
      if (s.t >= s.duration) {
        s.t = s.duration;
        if (!s.done) {
          s.done = true;
          this.fireEvents(s, prev, s.t);
          s.onDone?.();
          if (s === this.active && s.next) this.play(s.next, { fade: 0.3 });
          s.action!.time = s.duration * 0.99999;
          s.action!.setEffectiveWeight(s.w);
          return;
        }
      }
    } else if (s.duration > 0) {
      s.t %= s.duration;
    }
    if (s.t !== prev) this.fireEvents(s, prev, s.t);
    s.action!.time = Math.min(s.t, s.duration * 0.99999);
    s.action!.setEffectiveWeight(s.w);
  }

  private fireEvents(s: Slot, from: number, to: number) {
    const ev = s.name === 'loco' ? undefined : CLIP_EVENTS[s.name];
    if (!ev || to <= from) return;
    for (const [name, at] of Object.entries(ev)) {
      if (from < at && to >= at) {
        s.onEvent?.(name);
        this.onEvent?.(s.name, name);
      }
    }
  }
}
