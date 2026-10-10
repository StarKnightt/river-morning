import type { Engine } from '../core/Engine';
import { params, shotMode, type TimeOfDay } from '../core/params';
import { LAYOUT } from '../world/Layout';
import { river } from '../world/River';
import { heightAt } from '../world/Terrain';
import { AudioEngine, type SurfaceQuery, type Vec3Like } from './AudioEngine';
import { gameEvents, type EventBus, type GameEventMap } from './events';
import { defaultSurface } from './surface';

export { gameEvents, EventBus, type GameEventMap, type Surface } from './events';
export type { SurfaceQuery, Vec3Like } from './AudioEngine';

/** What UI and gameplay code call. In photo mode (?shot) and with ?audio=0 every method is a no-op. */
export interface GameAudio {
  readonly enabled: boolean;
  readonly events: EventBus<GameEventMap>;
  /** 0..1 (perceptual: the gain is volume squared), remembered in localStorage */
  readonly volume: number;
  readonly muted: boolean;
  setVolume(v: number): void;
  setMuted(m: boolean): void;
  toggleMute(): boolean;
  setBusVolume(bus: 'ambience' | 'sfx', v: number): void;
  setTimeOfDay(tod: TimeOfDay | number): void;
  /** player position for the ambience mix (the camera until set) */
  setFocus(fn: (() => Vec3Like) | null): void;
  setSurfaceQuery(fn: SurfaceQuery | null): void;
  /** crank speed and line tension, 0..1 each (same as the 'reel' event) */
  setReel(speed: number, tension: number): void;
  /** a fish running against the drag, 0..1, 0 stops */
  setDrag(amount: number): void;
  /** campfire crackle level (0 = out, 1 = burning) and the coffee pot */
  setFire(intensity: number, coffee?: boolean): void;
  dispose(): void;
}

export interface InstallAudioOptions {
  /** rendered ground height (World.groundY); sources given without y sit on it */
  groundY?: (x: number, z: number) => number;
  /** water surface height for splashes and bites given without y */
  waterY?: (x: number, z: number) => number;
  /** footstep surface; default: dock, cabin, shallows, gravel bar and paths from the world layout */
  surface?: SurfaceQuery;
  focus?: () => Vec3Like;
  bus?: EventBus<GameEventMap>;
  tod?: TimeOfDay | number;
}

declare global {
  interface Window {
    /** sound state and controls for the console and automated checks (absent in photo mode) */
    __RMAudio?: {
      state: () => ReturnType<AudioEngine['debugState']>;
      setVolume: (v: number) => void;
      setMuted: (m: boolean) => void;
      emit: <K extends keyof GameEventMap>(type: K, e: GameEventMap[K]) => void;
      engine: AudioEngine;
    };
  }
}

const SETTINGS_KEY = 'rm.audio';
/** main.ts shows the game at frame 30; the files start downloading just after */
const PREFETCH_FRAME = 32;
const GESTURES = ['pointerdown', 'pointerup', 'keydown', 'touchend', 'click'] as const;
const NOT_ACTIVATING = new Set(['Escape', 'Shift', 'Control', 'Alt', 'Meta', 'CapsLock', 'Tab']);

function loadSettings(): { volume: number; muted: boolean } {
  try {
    const s = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}');
    return { volume: typeof s.volume === 'number' ? s.volume : 0.8, muted: s.muted === true };
  } catch {
    return { volume: 0.8, muted: false };
  }
}

function saveSettings(volume: number, muted: boolean) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({ volume, muted }));
  } catch {
    // storage blocked: the setting lasts for this session
  }
}

function nullAudio(bus: EventBus<GameEventMap>): GameAudio {
  const noop = () => {};
  return {
    enabled: false, events: bus, volume: 0, muted: true,
    setVolume: noop, setMuted: noop, toggleMute: () => true, setBusVolume: noop, setTimeOfDay: noop, setFocus: noop,
    setSurfaceQuery: noop, setReel: noop, setDrag: noop, setFire: noop, dispose: noop,
  };
}

let current: GameAudio | null = null;

/** The installed sound (a no-op stand-in before installAudio() and in photo mode), e.g. for a settings menu. */
export function getAudio(): GameAudio {
  return (current ??= nullAudio(gameEvents));
}

/**
 * Sound for the game. Call once after engine.start(). Nothing happens in photo mode (?shot) or with ?audio=0: no
 * AudioContext, no listeners, no downloads. Otherwise the compressed files download after the first frames, and the
 * first user gesture creates the AudioContext, decodes (one file at a time) and fades the ambience in.
 * URL switches: ?audio=0 (off), ?vol=0..1 (start volume, not saved), ?audiodebug (state panel).
 */
export function installAudio(engine: Engine, opts: InstallAudioOptions = {}): GameAudio {
  if (current?.enabled) return current;
  const q = new URLSearchParams(location.search);
  const bus = opts.bus ?? gameEvents;
  const webAudio = typeof AudioContext !== 'undefined' || 'webkitAudioContext' in window;
  if (shotMode || q.get('audio') === '0' || !webAudio) return (current = nullAudio(bus));

  const saved = loadSettings();
  const vol = Number(q.get('vol') ?? NaN);
  const eng = new AudioEngine({
    volume: Number.isFinite(vol) ? vol : saved.volume,
    muted: saved.muted,
    tod: opts.tod ?? params.tod,
    surface: opts.surface ?? defaultSurface,
    groundY: opts.groundY ?? heightAt,
    waterY: opts.waterY ?? ((x) => river.waterYAt(x)),
    campfire: LAYOUT.campfire,
  });
  eng.setListener(engine.camera);
  if (opts.focus) eng.focus = opts.focus;

  const onGesture = (e: Event) => {
    if (!eng.wantsGesture) return;
    if (e.type === 'keydown' && NOT_ACTIVATING.has((e as KeyboardEvent).key)) return;
    eng.unlock();
  };
  const onVisibility = () => eng.setHidden(document.hidden);
  for (const t of GESTURES) window.addEventListener(t, onGesture, { capture: true, passive: true });
  document.addEventListener('visibilitychange', onVisibility);

  let frames = 0, disposed = false;
  engine.addUpdater(() => {
    if (disposed) return;
    if (++frames === PREFETCH_FRAME) eng.prefetch();
    eng.update();
  });

  // an audio bug must never break the gameplay code that emitted the event
  let reported = false;
  const sub = <K extends keyof GameEventMap>(type: K, fn: (e: GameEventMap[K]) => void) =>
    bus.on(type, (e) => {
      try {
        fn(e);
      } catch (err) {
        if (!reported) console.warn('[audio]', err);
        reported = true;
      }
    });
  const offs = [
    sub('cast', (e) => eng.cast(e.x, e.y, e.z, e.power)),
    sub('splash', (e) => eng.splash(e.x, e.y, e.z, e.strength)),
    sub('plop', (e) => eng.plop(e.x, e.y, e.z, e.size)),
    sub('bite', (e) => eng.bite(e.x, e.y, e.z, e.strength)),
    sub('reelStart', (e) => eng.reelStart(e.x, e.y, e.z)),
    sub('reelStop', () => eng.reelStop()),
    sub('reel', (e) => eng.setReel(e.speed, e.tension)),
    sub('drag', (e) => eng.drag(e.strength, e.x, e.y, e.z)),
    sub('dragRun', (e) => eng.setDrag(e.amount)),
    sub('thrash', (e) => eng.thrash(e.x, e.y, e.z, e.strength)),
    sub('footstep', (e) => eng.footstep(e.x, e.y, e.z, e.surface, e.speed)),
    sub('timeOfDay', (e) => eng.setTimeOfDay(e.hour ?? e.tod ?? 'morning')),
    sub('door', (e) => eng.door(e.x, e.y, e.z)),
    sub('sound', (e) => eng.play(e.id, e.x, e.y, e.z, e.gainDb, e.rate)),
  ];

  let panelTimer = 0;
  if (q.has('audiodebug')) {
    const el = document.createElement('div');
    el.className = 'debug-panel';
    el.style.top = '96px';
    (document.getElementById('hud') ?? document.body).appendChild(el);
    panelTimer = window.setInterval(() => (el.textContent = panelText(eng)), 250);
  }

  const audio: GameAudio = {
    enabled: true,
    events: bus,
    get volume() {
      return eng.volume;
    },
    get muted() {
      return eng.muted;
    },
    setVolume(v) {
      eng.setVolume(v);
      saveSettings(eng.volume, eng.muted);
    },
    setMuted(m) {
      eng.setMuted(m);
      saveSettings(eng.volume, eng.muted);
    },
    toggleMute() {
      audio.setMuted(!eng.muted);
      return eng.muted;
    },
    setBusVolume: (b, v) => eng.setBusVolume(b, v),
    setTimeOfDay: (t) => eng.setTimeOfDay(t),
    setFocus: (fn) => (eng.focus = fn),
    setSurfaceQuery: (fn) => (eng.surface = fn),
    setReel: (s, t) => eng.setReel(s, t),
    setDrag: (a) => eng.setDrag(a),
    setFire: (i, c) => eng.setFire(i, c),
    dispose() {
      disposed = true;
      for (const off of offs) off();
      for (const t of GESTURES) window.removeEventListener(t, onGesture, { capture: true });
      document.removeEventListener('visibilitychange', onVisibility);
      clearInterval(panelTimer);
      eng.close();
      delete window.__RMAudio;
      current = null;
    },
  };
  window.__RMAudio = {
    state: () => eng.debugState(),
    setVolume: (v) => audio.setVolume(v),
    setMuted: (m) => audio.setMuted(m),
    emit: (type, e) => bus.emit(type, e),
    engine: eng,
  };
  return (current = audio);
}

function panelText(eng: AudioEngine) {
  const s = eng.debugState();
  const beds = Object.entries(s.beds).map(([k, v]) => `${k} ${v}`);
  return `audio ${s.context} ${s.sampleRate} Hz  vol ${s.volume.toFixed(2)}${s.muted ? ' MUTED' : ''}  hour ${s.hour}\n` +
    `decoded ${s.decoded} (${s.decodedMB} MB)  voices ${s.voices}  reel ${s.reel ? 'on' : 'off'}  limiter ${s.limiterDb} dB\n` +
    `${beds.slice(0, 4).join('  ')}\n${beds.slice(4).join('  ')}  riffle lp ${(s.riffleCutoff / 1000).toFixed(1)}k\n` +
    `listener ${s.listener.join(' ')}  last ${s.lastEvent}`;
}
