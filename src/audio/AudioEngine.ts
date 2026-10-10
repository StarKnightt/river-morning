import type { TimeOfDay } from '../core/params';
import { Ambience, bedsNeeded, type AmbienceHost } from './Ambience';
import { AUDIO_DIR, GROUP_IDS, GROUPS, LOOP_IDS, LOOPS, type GroupId, type LoopId } from './catalog';
import type { Surface } from './events';
import { dayMix, hourOf } from './timeOfDay';
import { clamp, dbToGain, gainToDb, jitter, lerp } from './util';

export type SurfaceQuery = (x: number, z: number) => Surface;

/** Anything with a world matrix (the THREE camera): the listener follows it. */
export interface ListenerSource {
  matrixWorld: { elements: ArrayLike<number> };
  updateMatrixWorld?: () => void;
}

export interface Vec3Like {
  x: number;
  y: number;
  z: number;
}

/** Ogg Vorbis everywhere it decodes; otherwise an `.m4a` (AAC) twin of each file, if one was deployed. */
function pickExtension(): 'ogg' | 'm4a' {
  try {
    return document.createElement('audio').canPlayType('audio/ogg; codecs="vorbis"') ? 'ogg' : 'm4a';
  } catch {
    return 'ogg';
  }
}

function reshape(ctx: BaseAudioContext, b: AudioBuffer, store?: 'mono' | 'left'): AudioBuffer {
  if (!store || b.numberOfChannels < 2) return b;
  const out = ctx.createBuffer(1, b.length, b.sampleRate);
  const o = out.getChannelData(0), l = b.getChannelData(0);
  if (store === 'left') o.set(l);
  else {
    const r = b.getChannelData(1);
    for (let i = 0; i < o.length; i++) o[i] = (l[i] + r[i]) * 0.5;
  }
  return out;
}

/**
 * Fetches the compressed files (cheap, any time) and decodes them one at a time (the only CPU-heavy step). Decoded
 * buffers are shared, so an OfflineAudioContext harness can reuse them across renders.
 */
export class SoundBank {
  readonly buffers = new Map<string, AudioBuffer>();
  decodedBytes = 0;
  failures = 0;
  private fetches = new Map<string, Promise<ArrayBuffer | null>>();
  private decodes = new Map<string, Promise<AudioBuffer | null>>();
  private chain: Promise<unknown> = Promise.resolve();
  private warned = new Set<string>();

  constructor(readonly base = import.meta.env.BASE_URL + AUDIO_DIR, readonly ext: 'ogg' | 'm4a' = pickExtension()) {}

  fetch(file: string, ext = this.ext): Promise<ArrayBuffer | null> {
    const key = `${file}.${ext}`;
    let p = this.fetches.get(key);
    if (!p) {
      p = fetch(this.base + key, { priority: 'low' } as RequestInit)
        .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`HTTP ${r.status}`))))
        .catch((e) => {
          this.warn(key, e);
          return null;
        });
      this.fetches.set(key, p);
    }
    return p;
  }

  decode(ctx: BaseAudioContext, file: string, store?: 'mono' | 'left'): Promise<AudioBuffer | null> {
    let p = this.decodes.get(file);
    if (!p) {
      p = this.chain.then(async () => {
        let b = await this.decodeAs(ctx, file, this.ext);
        if (!b && this.ext === 'ogg') b = await this.decodeAs(ctx, file, 'm4a');
        if (!b) {
          this.failures++;
          return null;
        }
        b = reshape(ctx, b, store);
        this.buffers.set(file, b);
        this.decodedBytes += b.length * b.numberOfChannels * 4;
        return b;
      });
      this.chain = p.catch(() => null);
      this.decodes.set(file, p);
    }
    return p;
  }

  /** resolves once everything queued so far has been decoded (or failed) */
  idle(): Promise<unknown> {
    return this.chain;
  }

  private async decodeAs(ctx: BaseAudioContext, file: string, ext: 'ogg' | 'm4a') {
    const data = await this.fetch(file, ext);
    this.fetches.delete(`${file}.${ext}`);
    if (!data) return null;
    try {
      return await ctx.decodeAudioData(data);
    } catch (e) {
      this.warn(`${file}.${ext}`, e);
      return null;
    }
  }

  private warn(key: string, e: unknown) {
    if (this.warned.has(key)) return;
    this.warned.add(key);
    console.warn(`[audio] ${key}: ${e instanceof Error ? e.message : String(e)}`);
  }
}

function setPos(p: PannerNode, x: number, y: number, z: number) {
  if (p.positionX) {
    p.positionX.value = x;
    p.positionY.value = y;
    p.positionZ.value = z;
  } else p.setPosition(x, y, z);
}

/** A reusable one-shot channel: per play only the buffer source is new. */
class Voice {
  src: AudioBufferSourceNode | null = null;
  group: GroupId | null = null;
  startedAt = 0;
  endsAt = 0;

  constructor(readonly gain: GainNode, readonly panner: PannerNode | null) {}

  readonly ended = (e: Event) => {
    (e.target as AudioBufferSourceNode).disconnect();
    if (e.target === this.src) {
      this.src = null;
      this.group = null;
    }
  };
}

/**
 * Settings of the master limiter: a safety net for stacked transients (campfire pops under a splash and a reel
 * zip). Normal levels stay under its threshold.
 */
const LIMITER = { threshold: -2.5, knee: 2.5, ratio: 16, attack: 0.002, release: 0.18 };

function applyLimiter(c: DynamicsCompressorNode) {
  c.threshold.value = LIMITER.threshold;
  c.knee.value = LIMITER.knee;
  c.ratio.value = LIMITER.ratio;
  c.attack.value = LIMITER.attack;
  c.release.value = LIMITER.release;
}

/**
 * The Web Audio spec gives DynamicsCompressorNode an automatic make-up gain on every signal; measure it once
 * (a quiet constant through the same settings) so the master can divide it out and stay at unity below threshold.
 * A new compressor starts out releasing from a deep gain reduction, so the reading waits until that has settled.
 */
async function limiterMakeup(): Promise<number> {
  const oc = new OfflineAudioContext(1, 96000, 48000);
  const src = oc.createConstantSource();
  src.offset.value = 0.001;
  const c = oc.createDynamicsCompressor();
  applyLimiter(c);
  src.connect(c).connect(oc.destination);
  src.start();
  const v = (await oc.startRendering()).getChannelData(0)[95000];
  return v > 0 ? v / 0.001 : 1;
}

export interface AudioEngineOptions {
  volume: number;
  muted: boolean;
  tod: TimeOfDay | number;
  bank?: SoundBank;
  surface?: SurfaceQuery | null;
  /** ground height, for sources given without y (the campfire) */
  groundY?: (x: number, z: number) => number;
  /** water surface height, for water events given without y */
  waterY?: (x: number, z: number) => number;
  campfire: { x: number; z: number };
}

const POSITIONAL_VOICES = 14;
const FLAT_VOICES = 6;

/**
 * Web Audio engine: master bus (volume, mute, limiter), ambience bus and sfx bus, a voice pool for positional
 * one-shots, the reel loop and drag, footsteps by surface, and the ambience beds (Ambience). Nothing is created
 * before attach(): unlock() does that from a user gesture with a real AudioContext, a test harness with an
 * OfflineAudioContext. update() runs every frame and does not allocate.
 */
export class AudioEngine implements AmbienceHost {
  ctx: BaseAudioContext | null = null;
  ambience: Ambience | null = null;
  readonly bank: SoundBank;
  volume: number;
  muted: boolean;
  surface: SurfaceQuery | null;
  /** player position for the ambience mix; the listener (camera) when null */
  focus: (() => Vec3Like) | null = null;
  limiter: DynamicsCompressorNode | null = null;
  /** resolves once the limiter's make-up gain is divided out (offline renders wait for it) */
  makeupReady: Promise<void> = Promise.resolve();
  /** the measured make-up gain of the limiter (dB) */
  makeupDb = 0;
  lastEvent = '';
  private rt: AudioContext | null = null;
  private master!: GainNode;
  private makeup!: GainNode;
  private amb!: GainNode;
  private sfx!: GainNode;
  private busVolume = { ambience: 1, sfx: 1 };
  private pos: Voice[] = [];
  private flat: Voice[] = [];
  private lastPick: Record<string, number> = {};
  private hour: number;
  private listener: ListenerSource | null = null;
  private lp = new Float64Array(9).fill(1e9);
  private lx = 0;
  private ly = 0;
  private lz = 0;
  private rx = 1;
  private rz = 0;
  private nextMix = 0;
  private prefetching = false;
  private groundY: (x: number, z: number) => number;
  private waterY: (x: number, z: number) => number;
  private campfire: { x: number; z: number };
  private lastStep = -1;
  private suspendTimer = 0;
  // reel
  private reelSrc: AudioBufferSourceNode | null = null;
  private reelOut: GainNode | null = null;
  private reel2d!: GainNode;
  private reel3d!: GainNode;
  private reelPanner!: PannerNode;
  private reelOn = false;
  private reelPositional = false;
  private reelPos = { x: 0, y: 0, z: 0 };
  private reelSpeed = 0;
  private reelTension = 0;
  private reelDirty = false;
  private reelG = -1;
  private reelR = -1;
  private dragAmount = 0;
  private nextDrag = 0;

  constructor(o: AudioEngineOptions) {
    this.volume = clamp(o.volume, 0, 1);
    this.muted = o.muted;
    this.hour = hourOf(o.tod);
    this.bank = o.bank ?? new SoundBank();
    this.surface = o.surface ?? null;
    this.groundY = o.groundY ?? (() => 0);
    this.waterY = o.waterY ?? (() => 0);
    this.campfire = o.campfire;
    for (const id of GROUP_IDS) this.lastPick[id] = -1;
  }

  /** the context exists and is running (an offline context counts as running) */
  get live() {
    return this.ctx !== null && (this.rt === null || this.rt.state === 'running');
  }

  /** a user gesture would start or resume sound */
  get wantsGesture() {
    return !this.muted && (this.rt === null || this.rt.state !== 'running');
  }

  setListener(src: ListenerSource | null) {
    this.listener = src;
  }

  /**
   * From a user-gesture handler: creates the AudioContext (first time) or resumes it. Without transient user
   * activation it does nothing, so the browser never logs an autoplay warning.
   */
  unlock() {
    if (this.muted || document.hidden) return;
    const act = (navigator as Navigator & { userActivation?: { isActive: boolean } }).userActivation;
    if (act && !act.isActive) return;
    if (!this.rt) {
      const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AC) return;
      this.rt = new AC({ latencyHint: 'interactive' });
      this.attach(this.rt);
    }
    if (this.rt.state !== 'running') void this.rt.resume();
  }

  /** Build the graph on a context and start decoding. A real context comes from unlock(). */
  attach(ctx: BaseAudioContext) {
    this.ctx = ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 0;
    this.limiter = ctx.createDynamicsCompressor();
    applyLimiter(this.limiter);
    this.makeup = ctx.createGain();
    this.master.connect(this.limiter).connect(this.makeup).connect(ctx.destination);
    this.makeupReady = limiterMakeup().then((m) => {
      this.makeup.gain.value = 1 / m;
      this.makeupDb = gainToDb(m);
    }, () => {});
    this.amb = ctx.createGain();
    this.amb.connect(this.master);
    this.sfx = ctx.createGain();
    this.sfx.connect(this.master);
    this.amb.gain.value = this.busVolume.ambience ** 2;
    this.sfx.gain.value = this.busVolume.sfx ** 2;

    for (let i = 0; i < POSITIONAL_VOICES; i++) {
      const g = ctx.createGain(), p = ctx.createPanner();
      p.panningModel = 'equalpower';
      p.distanceModel = 'inverse';
      p.maxDistance = 10000;
      g.connect(p).connect(this.sfx);
      this.pos.push(new Voice(g, p));
    }
    for (let i = 0; i < FLAT_VOICES; i++) {
      const g = ctx.createGain();
      g.connect(this.sfx);
      this.flat.push(new Voice(g, null));
    }
    this.reel2d = ctx.createGain();
    this.reel2d.connect(this.sfx);
    this.reelPanner = ctx.createPanner();
    this.reelPanner.panningModel = 'equalpower';
    this.reelPanner.distanceModel = 'inverse';
    this.reelPanner.refDistance = 3;
    this.reelPanner.connect(this.sfx);
    this.reel3d = ctx.createGain();
    this.reel3d.connect(this.reelPanner);
    this.reel2d.gain.value = this.reel3d.gain.value = 0;

    const fx = this.campfire.x, fz = this.campfire.z;
    this.ambience = new Ambience(this, ctx, this.amb, { x: fx, y: this.groundY(fx, fz) + 0.25, z: fz }, this.hour);
    // a gentle start: the master rises over ~1.5 s
    this.applyMaster(0.5);
    this.loadAll();
  }

  /** Fetch every file (compressed, low priority, one after another) without decoding. Call after the first frames. */
  prefetch() {
    if (this.prefetching) return;
    this.prefetching = true;
    const order = this.fileOrder();
    void (async () => {
      for (const f of order) await this.bank.fetch(f);
    })();
  }

  loadLoop(id: LoopId) {
    return this.bank.decode(this.ctx!, LOOPS[id].file, LOOPS[id].store);
  }

  private fileOrder() {
    const out: string[] = [LOOPS.riverCalm.file];
    const add = (f: string) => {
      if (!out.includes(f)) out.push(f);
    };
    for (const id of GROUP_IDS) if (GROUPS[id].priority === 0) GROUPS[id].files.forEach(add);
    for (const id of bedsNeeded(dayMix(this.hour, { birds: 0, crickets: 0, owl: 0, wind: 0 }))) add(LOOPS[id].file);
    add(LOOPS.reel.file);
    for (const id of GROUP_IDS) if (GROUPS[id].priority === 1) GROUPS[id].files.forEach(add);
    for (const id of LOOP_IDS) add(LOOPS[id].file);
    return out;
  }

  private loadAll() {
    const ctx = this.ctx!;
    // the calm river first: it is what the first click brings in
    this.ambience!.request('riverCalm');
    for (const id of GROUP_IDS) if (GROUPS[id].priority === 0) for (const f of GROUPS[id].files) void this.bank.decode(ctx, f);
    this.ambience!.requestNeeded();
    void this.loadLoop('reel');
    for (const id of GROUP_IDS) if (GROUPS[id].priority === 1) for (const f of GROUPS[id].files) void this.bank.decode(ctx, f);
  }

  /** Every frame. Cheap when there is no running context. */
  update() {
    const ctx = this.ctx;
    if (!ctx || !this.live) return;
    const now = ctx.currentTime;
    this.updateListener();
    if (this.reelOn) this.updateReel(now);
    if (this.dragAmount > 0.05 && now >= this.nextDrag) this.dragTick(now);
    if (now >= this.nextMix - 0.005 && this.ambience) {
      this.nextMix = Math.max(this.nextMix + 0.1, now + 0.05);
      const f = this.focus?.();
      this.ambience.update(now, f ? f.x : this.lx, f ? f.y : this.ly, f ? f.z : this.lz, this.rx, this.rz);
    }
  }

  private updateListener() {
    const cam = this.listener;
    if (!cam) return;
    cam.updateMatrixWorld?.();
    const e = cam.matrixWorld.elements, lp = this.lp;
    const px = e[12], py = e[13], pz = e[14];
    let fx = -e[8], fy = -e[9], fz = -e[10];
    const fl = Math.hypot(fx, fy, fz) || 1;
    fx /= fl;
    fy /= fl;
    fz /= fl;
    const ux = e[4], uy = e[5], uz = e[6];
    this.lx = px;
    this.ly = py;
    this.lz = pz;
    const rl = Math.hypot(e[0], e[2]) || 1;
    this.rx = e[0] / rl;
    this.rz = e[2] / rl;
    const moved = Math.abs(px - lp[0]) + Math.abs(py - lp[1]) + Math.abs(pz - lp[2]) > 1e-3;
    const turned = Math.abs(fx - lp[3]) + Math.abs(fy - lp[4]) + Math.abs(fz - lp[5]) + Math.abs(ux - lp[6]) +
      Math.abs(uy - lp[7]) + Math.abs(uz - lp[8]) > 1e-4;
    if (!moved && !turned) return;
    lp[0] = px; lp[1] = py; lp[2] = pz; lp[3] = fx; lp[4] = fy; lp[5] = fz; lp[6] = ux; lp[7] = uy; lp[8] = uz;
    const L = this.ctx!.listener;
    if (L.positionX) {
      if (moved) {
        L.positionX.value = px;
        L.positionY.value = py;
        L.positionZ.value = pz;
      }
      if (turned) {
        L.forwardX.value = fx;
        L.forwardY.value = fy;
        L.forwardZ.value = fz;
        L.upX.value = ux;
        L.upY.value = uy;
        L.upZ.value = uz;
      }
    } else {
      L.setPosition(px, py, pz);
      L.setOrientation(fx, fy, fz, ux, uy, uz);
    }
  }

  // ----- one-shots -----

  /** Play a catalog group: positional when x and z are given (y defaults to the ground); variant picks a file. */
  play(id: GroupId, x?: number, y?: number, z?: number, gainDb = 0, rate = 1, variant = -1): boolean {
    const ctx = this.ctx;
    if (!ctx || !this.live) return false;
    const g = GROUPS[id];
    const buf = (variant >= 0 && this.bank.buffers.get(g.files[variant % g.files.length])) || this.pick(id);
    if (!buf) return false;
    const now = ctx.currentTime;
    const positional = x !== undefined && z !== undefined;
    const v = this.voiceFor(id, positional, now);
    if (!v) return false;
    const gp = v.gain.gain;
    let t0 = now;
    gp.cancelScheduledValues(now);
    if (v.src && v.endsAt > now) {
      // stealing a voice that still sounds: a 12 ms fade instead of a click
      gp.setValueAtTime(gp.value, now);
      gp.linearRampToValueAtTime(0, now + 0.012);
      v.src.stop(now + 0.013);
      t0 = now + 0.013;
    } else if (v.src) {
      v.src.disconnect();
    }
    v.src = null;
    const r = rate * (1 + jitter() * g.rateJitter);
    gp.setValueAtTime(dbToGain(g.gainDb + gainDb + jitter() * g.gainJitter), t0);
    if (v.panner && positional) {
      v.panner.refDistance = g.ref;
      v.panner.rolloffFactor = g.rolloff;
      setPos(v.panner, x, y ?? this.groundY(x, z), z);
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.playbackRate.value = r;
    src.connect(v.gain);
    src.onended = v.ended;
    src.start(t0);
    v.src = src;
    v.group = id;
    v.startedAt = t0;
    v.endsAt = t0 + buf.duration / r;
    this.lastEvent = id;
    return true;
  }

  /** a decoded variant, never the same one twice in a row */
  private pick(id: GroupId): AudioBuffer | null {
    const files = GROUPS[id].files, n = files.length;
    let i = Math.floor(Math.random() * n);
    if (n > 1 && i === this.lastPick[id]) i = (i + 1 + Math.floor(Math.random() * (n - 1))) % n;
    for (let k = 0; k < n; k++) {
      const j = (i + k) % n, b = this.bank.buffers.get(files[j]);
      if (b) {
        this.lastPick[id] = j;
        return b;
      }
    }
    return null;
  }

  private voiceFor(id: GroupId, positional: boolean, now: number): Voice | null {
    const pool = positional ? this.pos : this.flat;
    let free: Voice | null = null, oldest: Voice | null = null, oldestSame: Voice | null = null, same = 0;
    for (let i = 0; i < pool.length; i++) {
      const v = pool[i];
      if (!v.src || v.endsAt <= now) {
        if (!free) free = v;
        continue;
      }
      if (v.group === id) {
        same++;
        if (!oldestSame || v.startedAt < oldestSame.startedAt) oldestSame = v;
      }
      if (!oldest || v.startedAt < oldest.startedAt) oldest = v;
    }
    if (same >= GROUPS[id].maxVoices) return oldestSame;
    return free ?? oldest;
  }

  /** rod swing, power 0..1 */
  cast(x: number, y: number, z: number, power = 0.5) {
    const p = clamp(power, 0, 1);
    this.play(p > 0.7 ? 'cast_heavy' : 'cast', x, y, z, -4 + 4 * p, 0.95 + 0.1 * p);
  }

  /** strength as RiverWater.splash: 1 a fish jumping clear, 0.3 a lure landing, 0.1 a fish rising */
  splash(x: number, y: number | undefined, z: number, strength = 0.3) {
    const yy = y ?? this.waterY(x, z);
    if (strength >= 0.6) {
      const s = Math.min(strength, 2);
      this.play('splash_big', x, yy, z, 4 * Math.log2(s), 1.04 - 0.06 * Math.min(s, 1.5));
    } else if (strength >= 0.17) {
      this.play('splash_small', x, yy, z, 8 * Math.log10(strength / 0.3), 1 + (0.3 - strength) * 0.4);
    } else {
      this.plop(x, yy, z, strength / 0.1);
    }
  }

  plop(x: number, y: number | undefined, z: number, size = 1) {
    const s = clamp(size, 0.2, 3);
    this.play('plop', x, y ?? this.waterY(x, z), z, clamp(6 * Math.log2(s), -9, 3), clamp(1.1 - 0.15 * Math.log2(s), 0.8, 1.5));
  }

  /** the bobber dips: strength 0..1; a take gets the clear plip, a nibble the soft one */
  bite(x: number, y: number | undefined, z: number, strength = 1) {
    this.play('bite', x, y ?? this.waterY(x, z), z, 6 * Math.log2(clamp(strength, 0.15, 1.5)), 1, strength >= 0.5 ? 0 : 1);
  }

  thrash(x: number, y: number | undefined, z: number, strength = 1) {
    this.play('thrash', x, y ?? this.waterY(x, z), z, 6 * Math.log2(clamp(strength, 0.3, 1.5)));
  }

  drip(x: number, y: number, z: number) {
    this.play('drip', x, y, z);
  }

  door(x: number, y: number, z: number) {
    this.play('door', x, y, z);
  }

  /** a foot lands; speed in m/s scales the level (slow walk -5 dB, run +3 dB) */
  footstep(x: number, y: number, z: number, surface?: Surface, speed = 1.5) {
    const ctx = this.ctx;
    if (!ctx || !this.live) return;
    const now = ctx.currentTime;
    if (now - this.lastStep < 0.12) return;
    this.lastStep = now;
    const s = surface ?? this.surface?.(x, z) ?? 'gravel';
    const db = clamp(6 * Math.log2(Math.max(0.2, speed) / 1.5), -7, 3);
    switch (s) {
      case 'wood': this.play('step_wood', x, y, z, db); break;
      case 'creak': this.play(Math.random() < 0.5 ? 'step_creak' : 'step_wood', x, y, z, db); break;
      case 'grass': this.play('step_grass', x, y, z, db); break;
      // no wading recordings: a soft, high splash stands in
      case 'water': this.play('splash_small', x, y, z, db - 13, 1.4); break;
      default: this.play('step_gravel', x, y, z, db);
    }
  }

  // ----- reel -----

  reelStart(x?: number, y?: number, z?: number) {
    this.reelOn = true;
    this.reelDirty = true;
    this.reelPositional = x !== undefined && z !== undefined;
    if (x !== undefined && z !== undefined) this.setReelPosition(x, y ?? this.groundY(x, z) + 1.1, z);
    this.ensureReel();
  }

  setReelPosition(x: number, y: number, z: number) {
    this.reelPos.x = x;
    this.reelPos.y = y;
    this.reelPos.z = z;
    if (this.reelPanner) setPos(this.reelPanner, x, y, z);
  }

  /** crank speed and line tension, 0..1 each: the loop's rate and level follow on the next frame */
  setReel(speed: number, tension: number) {
    const s = clamp(speed, 0, 1), k = clamp(tension, 0, 1);
    if (s === this.reelSpeed && k === this.reelTension) return;
    this.reelSpeed = s;
    this.reelTension = k;
    this.reelDirty = true;
  }

  reelStop() {
    this.reelOn = false;
    const ctx = this.ctx, src = this.reelSrc, out = this.reelOut;
    if (!ctx || !src || !out) return;
    const now = ctx.currentTime;
    out.gain.cancelScheduledValues(now);
    out.gain.setTargetAtTime(0, now, 0.04);
    src.stop(now + 0.3);
    this.reelSrc = null;
  }

  private ensureReel() {
    const ctx = this.ctx;
    if (!ctx || !this.live || !this.reelOn || this.reelSrc) return;
    const buf = this.bank.buffers.get(LOOPS.reel.file);
    if (!buf) return;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true;
    const out = this.reelPositional ? this.reel3d : this.reel2d;
    src.connect(out);
    src.onended = () => src.disconnect();
    src.start(ctx.currentTime, Math.random() * buf.duration);
    this.reelSrc = src;
    this.reelOut = out;
    this.reelG = this.reelR = -1;
    this.reelDirty = true;
  }

  private updateReel(now: number) {
    if (!this.reelSrc) {
      this.ensureReel();
      if (!this.reelSrc) return;
    }
    if (!this.reelDirty) return;
    this.reelDirty = false;
    const s = this.reelSpeed, k = this.reelTension;
    // a strained crank turns slower and sounds a little louder
    const g = s < 0.03 ? 0 : dbToGain(LOOPS.reel.gainDb - 14 * (1 - Math.pow(s, 0.6)) + 2.5 * k);
    const r = 0.8 + 0.45 * s - 0.08 * k;
    if (Math.abs(g - this.reelG) > 0.02 * Math.max(g, this.reelG) + 1e-5) {
      this.reelOut!.gain.setTargetAtTime(g, now, 0.05);
      this.reelG = g;
    }
    if (Math.abs(r - this.reelR) > 0.004) {
      this.reelSrc!.playbackRate.setTargetAtTime(r, now, 0.08);
      this.reelR = r;
    }
  }

  /** one drag event: a ratchet zip, or a slow clicker run below 0.3 */
  drag(strength = 0.6, x?: number, y?: number, z?: number) {
    const a = clamp(strength, 0, 1);
    const px = x ?? (this.reelPositional ? this.reelPos.x : undefined);
    const py = y ?? (this.reelPositional ? this.reelPos.y : undefined);
    const pz = z ?? (this.reelPositional ? this.reelPos.z : undefined);
    if (a < 0.3) this.play('reel_pull', px, py, pz, -4 + 8 * a, 0.9 + 0.4 * a);
    else this.play('reel_drag', px, py, pz, lerp(-5, 0, (a - 0.3) / 0.7), 0.92 + 0.22 * a);
  }

  /** a fish running against the drag, 0..1 (0 stops): repeats zips, faster and higher with more pull */
  setDrag(amount: number) {
    this.dragAmount = clamp(amount, 0, 1);
  }

  private dragTick(now: number) {
    const a = this.dragAmount;
    this.drag(a);
    this.nextDrag = now + (a < 0.3 ? 2.1 : lerp(0.78, 0.3, (a - 0.3) / 0.7)) * (0.85 + 0.3 * Math.random());
  }

  // ----- settings -----

  setTimeOfDay(v: TimeOfDay | number) {
    this.hour = hourOf(v);
    this.ambience?.setHour(this.hour);
  }

  setFire(intensity: number, coffee?: boolean) {
    if (!this.ambience) return;
    this.ambience.fire = clamp(intensity, 0, 2);
    if (coffee !== undefined) this.ambience.coffee = coffee;
  }

  setVolume(v: number) {
    this.volume = clamp(v, 0, 1);
    this.applyMaster();
  }

  setBusVolume(bus: 'ambience' | 'sfx', v: number) {
    this.busVolume[bus] = clamp(v, 0, 1);
    const ctx = this.ctx;
    if (ctx) (bus === 'ambience' ? this.amb : this.sfx).gain.setTargetAtTime(this.busVolume[bus] ** 2, ctx.currentTime, 0.05);
  }

  setMuted(m: boolean) {
    this.muted = m;
    clearTimeout(this.suspendTimer);
    if (m) {
      // the context sleeps once the fade is done: no audio thread work while muted
      this.suspendTimer = window.setTimeout(() => {
        if (this.muted && this.rt?.state === 'running') void this.rt.suspend();
      }, 400);
    } else if (!this.rt) this.unlock();
    else if (!document.hidden) void this.rt.resume();
    this.applyMaster();
  }

  /** document visibility: a hidden tab sleeps */
  setHidden(hidden: boolean) {
    if (!this.rt) return;
    if (hidden) void this.rt.suspend();
    else if (!this.muted) void this.rt.resume();
  }

  private applyMaster(tc = 0.06) {
    const ctx = this.ctx;
    if (!ctx) return;
    this.master.gain.setTargetAtTime(this.muted ? 0 : this.volume * this.volume, ctx.currentTime, tc);
  }

  close() {
    this.reelStop();
    void this.rt?.close();
  }

  // ----- debug -----

  /** for the debug panel and automated checks (allocates; not for per-frame use) */
  debugState() {
    const a = this.ambience;
    const r1 = (v: number) => Math.round(v * 10) / 10;
    // per bed: its level in dB while it plays, else where it is in loading
    const beds: Record<string, string> = {};
    const levels: Record<string, number> = {};
    if (a) {
      for (const [k, b] of Object.entries(a.beds)) {
        const db = r1(gainToDb(a.level[k as keyof typeof a.level]));
        levels[k] = b.playing ? db : -120;
        beds[k] = b.playing ? `${db}` : b.buffer ? 'idle' : b.requested ? 'loading' : 'off';
      }
    }
    let voices = 0;
    const now = this.ctx?.currentTime ?? 0;
    for (const v of this.pos) if (v.src && v.endsAt > now) voices++;
    for (const v of this.flat) if (v.src && v.endsAt > now) voices++;
    return {
      context: this.rt ? this.rt.state : this.ctx ? 'offline' : 'none',
      sampleRate: this.ctx?.sampleRate ?? 0,
      volume: this.volume,
      muted: this.muted,
      hour: this.hour,
      decoded: this.bank.buffers.size,
      decodedMB: r1(this.bank.decodedBytes / 1e6),
      failures: this.bank.failures,
      beds,
      levels,
      riffleCutoff: Math.round(a?.level.cutoff ?? 0),
      day: a ? { ...a.day } : null,
      voices,
      reel: this.reelSrc !== null,
      limiterDb: r1(this.limiter?.reduction ?? 0),
      listener: [this.lx, this.ly, this.lz].map(r1),
      lastEvent: this.lastEvent,
    };
  }
}
