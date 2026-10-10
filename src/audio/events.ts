import type { TimeOfDay } from '../core/params';
import type { GroupId } from './catalog';

/** Ground under a footstep: 'creak' is the cabin's old plank floor, 'water' is wading in the shallows. */
export type Surface = 'gravel' | 'wood' | 'grass' | 'creak' | 'water';

/**
 * Gameplay events the sound listens to. Positions are world metres. A missing y means the water surface for water
 * events and the ground otherwise. Payloads are read during emit() and never kept, so an emitter can reuse one
 * object per event type.
 */
export interface GameEventMap {
  /** rod swing; power 0..1 (a heavier whoosh from 0.7) */
  cast: { x: number; y: number; z: number; power?: number };
  /** something hits the water; strength as RiverWater.splash: 1 a fish jumping clear, 0.3 a lure landing, 0.1 a rise */
  splash: { x: number; y?: number; z: number; strength?: number };
  /** a small drop or a fish sipping a fly; size 1 = a normal plop */
  plop: { x: number; y?: number; z: number; size?: number };
  /** the bobber dips: strength 0..1 (a nibble ~0.3, a take 1) */
  bite: { x: number; y?: number; z: number; strength?: number };
  /** the crank starts turning; the reel's position if known (otherwise the reel plays non-positional) */
  reelStart: { x?: number; y?: number; z?: number };
  reelStop: Record<string, never>;
  /** crank speed and line tension, both 0..1; emit whenever they change (every frame is fine) */
  reel: { speed: number; tension: number };
  /** a fish takes line against the drag: one ratchet zip (a slow clicker run below 0.3), strength 0..1 */
  drag: { x?: number; y?: number; z?: number; strength?: number };
  /** a sustained run against the drag, 0..1 (0 ends it): zips repeat faster and higher with more pull */
  dragRun: { amount: number };
  /** a hooked fish fighting at the surface */
  thrash: { x: number; y?: number; z: number; strength?: number };
  /** a foot lands; surface from the surface query when omitted; speed in m/s (walk 1.5, slow walk 0.8) */
  footstep: { x: number; y: number; z: number; surface?: Surface; speed?: number };
  /** a ?tod preset or an hour of the day (0..24, wraps) */
  timeOfDay: { tod?: TimeOfDay; hour?: number };
  /** the cabin door swings */
  door: { x: number; y: number; z: number };
  /** any catalog group directly (escape hatch for sounds without their own event) */
  sound: { id: GroupId; x?: number; y?: number; z?: number; gainDb?: number; rate?: number };
}

type Handler<T> = (e: T) => void;

/** Tiny typed event bus. emit() does not allocate; on()/off() copy the handler list. */
export class EventBus<M extends object> {
  private handlers = new Map<keyof M, Handler<never>[]>();

  /** subscribe; returns the unsubscribe function */
  on<K extends keyof M>(type: K, fn: Handler<M[K]>): () => void {
    const list = this.handlers.get(type) ?? [];
    this.handlers.set(type, [...list, fn as Handler<never>]);
    return () => this.off(type, fn);
  }

  off<K extends keyof M>(type: K, fn: Handler<M[K]>) {
    const list = this.handlers.get(type);
    if (list) this.handlers.set(type, list.filter((h) => h !== fn));
  }

  emit<K extends keyof M>(type: K, e: M[K]) {
    const list = this.handlers.get(type) as Handler<M[K]>[] | undefined;
    if (!list) return;
    for (let i = 0; i < list.length; i++) list[i](e);
  }
}

/** The game's event bus. Gameplay code emits here; the sound (and anything else) subscribes. */
export const gameEvents = new EventBus<GameEventMap>();
