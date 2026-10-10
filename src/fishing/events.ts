import type { FishingState, SpeciesId } from './types';

/**
 * Tiny synchronous event bus for the fishing loop. The audio lane subscribes with `fishingEvents.onSound(fn)`:
 * `e.sound` is an audio.json group (pick a random variant of that group, apply its group gain, then `e.gain`).
 * Gameplay listeners use `fishingEvents.on(type, fn)`.
 *
 * The event object is reused for every emit (no allocation per event): read it inside the callback, copy
 * what you keep.
 */

/** audio.json `group` names (E:\shots\river-morning\prep\audio\audio.json) */
export type SoundId =
  | 'cast_whoosh' | 'cast_whoosh_heavy' | 'splash_small' | 'splash_big' | 'plop' | 'bite_plip'
  | 'reel_wind' | 'reel_drag' | 'reel_pull' | 'fish_thrash' | 'drip';

export type FishingEventType =
  | 'state'      // every state change (e.state)
  | 'cast'       // the rod whips forward: cast_whoosh (cast_whoosh_heavy above power 0.7)
  | 'payout'     // line pulled off the reel by a long cast: reel_pull
  | 'splash'     // the float lands on the water: splash_small; the released fish: splash_small
  | 'land_ground'// the float lands on gravel or grass (no sound file yet)
  | 'nibble'     // small twitch before a bite: plop (quiet)
  | 'bite'       // the float dips under: bite_plip. The hook window starts now.
  | 'hook'       // hook set (strike)
  | 'miss'       // struck with nothing on
  | 'escape'     // the bite window passed: the fish let go
  | 'reel'       // reel loop: loop = start | update (every frame, rate + tension) | stop -> reel_wind
  | 'drag'       // the drag slips, the fish takes line: reel_drag (re-emitted every ~0.8 s while it slips)
  | 'jump'       // the fish falls back after a jump: splash_big
  | 'thrash'     // the fish thrashes at the surface: fish_thrash
  | 'snap'       // the line broke (no sound file yet)
  | 'threw'      // the fish threw the hook on a slack line
  | 'landed'     // the fish leaves the water: drip
  | 'hold_fish'  // pose hook: the hold_fish moment starts (species, length, weight)
  | 'release'    // the hands let go, the fish swims off
  | 'catch';     // stats updated (e.record = biggest so far)

export type LoopPhase = 'once' | 'start' | 'update' | 'stop';

export interface FishingEvent {
  type: FishingEventType;
  sound: SoundId | null;
  loop: LoopPhase;
  /** world position of the sound */
  x: number;
  y: number;
  z: number;
  /** 0..1 relative loudness on top of the audio.json group gain */
  gain: number;
  /**
   * physical size: cast power 0..1, splash strength in RiverWater.splash units, bite 1 / nibble 0.3, crank speed
   * 0..1 on reel updates; other events copy `gain`
   */
  strength: number;
  /** playback rate (reel speed) */
  rate: number;
  /** line tension 0..1+ (reel, drag) */
  tension: number;
  state: FishingState;
  species: SpeciesId | null;
  lengthCm: number;
  weightKg: number;
  record: boolean;
}

type Listener = (e: FishingEvent) => void;

export class FishingEventBus {
  private byType = new Map<string, Listener[]>();
  private sound: Listener[] = [];
  private all: Listener[] = [];
  /** drop every event (deterministic pre-rolls of the shot poses must not play sounds) */
  muted = false;
  /** the one event object every emit fills in */
  readonly e: FishingEvent = {
    type: 'state', sound: null, loop: 'once', x: 0, y: 0, z: 0, gain: 1, strength: 1, rate: 1, tension: 0, state: 'idle',
    species: null, lengthCm: 0, weightKg: 0, record: false,
  };

  /** subscribe to one type, or '*' for everything; returns an unsubscribe function */
  on(type: FishingEventType | '*', fn: Listener): () => void {
    const list = type === '*' ? this.all : (this.byType.get(type) ?? (this.byType.set(type, []), this.byType.get(type)!));
    list.push(fn);
    return () => {
      const i = list.indexOf(fn);
      if (i >= 0) list.splice(i, 1);
    };
  }

  /** subscribe to events that carry a sound (audio lane) */
  onSound(fn: Listener): () => void {
    this.sound.push(fn);
    return () => {
      const i = this.sound.indexOf(fn);
      if (i >= 0) this.sound.splice(i, 1);
    };
  }

  /** reset the shared event, fill it with `begin(...)` + field writes, then `send()` */
  begin(type: FishingEventType, sound: SoundId | null, x: number, y: number, z: number, gain = 1): FishingEvent {
    const e = this.e;
    e.type = type;
    e.sound = sound;
    e.loop = 'once';
    e.x = x;
    e.y = y;
    e.z = z;
    e.gain = gain;
    e.strength = gain;
    e.rate = 1;
    e.tension = 0;
    e.record = false;
    return e;
  }

  send() {
    if (this.muted) return;
    const e = this.e;
    const list = this.byType.get(e.type);
    if (list) for (let i = 0; i < list.length; i++) list[i](e);
    if (e.sound) for (let i = 0; i < this.sound.length; i++) this.sound[i](e);
    for (let i = 0; i < this.all.length; i++) this.all[i](e);
  }
}

/** The shared bus: `import { fishingEvents } from '../fishing/events'` from the audio lane. */
export const fishingEvents = new FishingEventBus();
