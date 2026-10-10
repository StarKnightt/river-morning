import { fishingEvents, type FishingEvent, type FishingEventBus } from './events';

/** What the bridge sends, by event: the fishing subset of the sound's `GameEventMap` (part 9 patch, src/audio/events.ts). */
export interface FishingAudioPayloads {
  cast: { x: number; y: number; z: number; power?: number };
  splash: { x: number; y?: number; z: number; strength?: number };
  bite: { x: number; y?: number; z: number; strength?: number };
  reelStart: { x?: number; y?: number; z?: number };
  reelStop: Record<string, never>;
  reel: { speed: number; tension: number };
  drag: { x?: number; y?: number; z?: number; strength?: number };
  thrash: { x: number; y?: number; z: number; strength?: number };
  /** a catalog group by name: 'reel_pull' (line pulled off by a cast) and 'drip' (the fish lifted out) */
  sound: { id: string; x?: number; y?: number; z?: number; gainDb?: number };
}

/** The sound's event bus as the bridge uses it. The part 9 `gameEvents` fits as it is: `connectFishingAudio(gameEvents)`. */
export interface FishingAudioSink {
  emit<K extends keyof FishingAudioPayloads>(type: K, e: FishingAudioPayloads[K]): void;
}

/**
 * Forwards the fishing loop's sounds to the game's sound bus. The reel (loop, drag zips, line pulled off by a cast)
 * plays non-positional because it sits in the angler's hands; everything else plays where it happens.
 * Returns the unsubscribe function.
 */
export function connectFishingAudio(sink: FishingAudioSink, bus: FishingEventBus = fishingEvents): () => void {
  // payloads are read during emit() and never kept, so one object per kind is reused
  const cast = { x: 0, y: 0, z: 0, power: 0 };
  const hit = { x: 0, y: 0, z: 0, strength: 0 };
  const reel = { speed: 0, tension: 0 };
  const zip = { strength: 0 };
  const inHands = {};
  const none: Record<string, never> = {};
  const pull = { id: 'reel_pull', gainDb: 0 };
  const drip = { id: 'drip', x: 0, y: 0, z: 0, gainDb: 0 };
  const at = (e: FishingEvent, strength: number) => {
    hit.x = e.x;
    hit.y = e.y;
    hit.z = e.z;
    hit.strength = strength;
    return hit;
  };
  const db = (gain: number) => 20 * Math.log10(Math.max(gain, 1e-3));

  return bus.onSound((e) => {
    switch (e.type) {
      case 'cast':
        cast.x = e.x;
        cast.y = e.y;
        cast.z = e.z;
        cast.power = e.strength;
        sink.emit('cast', cast);
        break;
      case 'payout':
        pull.gainDb = db(e.gain);
        sink.emit('sound', pull);
        break;
      case 'splash':
      case 'jump':
        sink.emit('splash', at(e, e.strength));
        break;
      case 'nibble':
      case 'bite':
        sink.emit('bite', at(e, e.strength));
        break;
      case 'reel':
        if (e.loop === 'start') sink.emit('reelStart', inHands);
        else if (e.loop === 'stop') sink.emit('reelStop', none);
        else {
          reel.speed = e.strength;
          reel.tension = Math.min(1, Math.max(0, e.tension));
          sink.emit('reel', reel);
        }
        break;
      case 'drag':
        zip.strength = e.strength;
        sink.emit('drag', zip);
        break;
      case 'thrash':
        sink.emit('thrash', at(e, e.strength));
        break;
      case 'landed':
        drip.x = e.x;
        drip.y = e.y;
        drip.z = e.z;
        drip.gainDb = db(e.gain);
        sink.emit('sound', drip);
        break;
    }
  });
}
