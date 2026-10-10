import type { Engine } from './Engine';
import { shotMode } from './params';
import { Q } from './quality';

const q = new URLSearchParams(location.search);

/** play mode only: photo mode (?shot) stays native, ?scale= pins the scale, ?gov=0 turns it off */
export const governorWanted = !shotMode && !q.has('scale') && q.get('gov') !== '0';

/**
 * render scale steps as fractions of the native pixel count (the pixel ratio gets their square roots). The ceiling
 * (step 1) is the quality preset's drawing buffer (its DPR and megapixel caps, Engine.basePixelRatio); the floor is
 * the preset's minScale squared (Low 0.36, Medium 0.49, High 0.56, Ultra 0.64 of the pixels).
 */
const STEPS = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.42, 0.36];
/** High and Ultra never go below a 1920x1080 pixel count: a 1080p screen renders native there (it was blurry once) */
const FLOOR_PIXELS = 1920 * 1080;
const keepFullHD = () => Q.preset === 'high' || Q.preset === 'ultra';
/** one judgement per CHUNK seconds of frames */
const CHUNK = 0.5;
/** GPU ms per frame: above HI for DOWN_CHUNKS in a row steps down; the next step's predicted cost under LO for UP_CHUNKS steps up */
const HI = 15.6, LO = 13.2, DOWN_CHUNKS = 3, UP_CHUNKS = 8;
/** without GPU timestamps: the wall frame interval (ms), only when the main thread is not the bottleneck */
const WALL_HI = 17.8;
/** after any change: the passes resize, so frames are not judged for a while */
const SETTLE = 2;

/**
 * Dynamic resolution: holds 60 fps on big screens (4K, 5K, ultrawide at 200 %) by lowering the render scale in
 * small steps when the GPU stays over budget, and raising it again only with clear headroom. Judged on GPU time
 * (timestamp queries), so a CPU-bound frame never blurs the picture for nothing; a step up that has to come back
 * down within 8 s blocks further step-ups for 30 s, then 60, 120 (no oscillation).
 */
export class ResolutionGovernor {
  level = 0;
  /** linear factor on the pixel ratio */
  get scale() {
    return Math.sqrt(STEPS[this.level]);
  }
  private floorLevel = 0;
  private wall: number[] = [];
  private cpu: number[] = [];
  private gpu: number[] = [];
  private chunkT = 0;
  private over = 0;
  private under = 0;
  private settle = 3;
  private lastUpAt = -1e9;
  private upBlockedUntil = 0;
  private blockFor = 30;
  private clock = 0;
  readonly history: Array<{ t: number; from: number; to: number; cost: number; why: string }> = [];
  lastCost = 0;
  lastSource: 'gpu' | 'wall' | '' = '';

  constructor(private engine: Engine) {
    document.addEventListener('visibilitychange', () => this.reset(1));
    window.addEventListener('resize', () => this.reset(1));
  }

  private reset(settle: number) {
    this.wall.length = this.cpu.length = this.gpu.length = 0;
    this.chunkT = 0;
    this.over = this.under = 0;
    this.settle = Math.max(this.settle, settle);
  }

  /** the lowest level allowed for the current native drawing buffer */
  private updateFloor() {
    const native = this.engine.nativePixels();
    const presetFloor = Math.max(STEPS[STEPS.length - 1], Q.k.minScale * Q.k.minScale);
    const minFrac = Math.min(1, Math.max(presetFloor, keepFullHD() ? FLOOR_PIXELS / Math.max(1, native) : 0));
    let f = 0;
    while (f + 1 < STEPS.length && STEPS[f + 1] >= minFrac - 1e-6) f++;
    this.floorLevel = f;
  }

  gpuFrame(ms: number) {
    if (ms > 0 && ms < 200) this.gpu.push(ms);
  }

  /** once per frame with the wall interval since the last frame and the main-thread time of the last frame */
  frame(intervalMs: number, cpuMs: number) {
    if (!window.__RM?.ready) return;
    const dt = intervalMs / 1000;
    this.clock += Math.min(dt, 0.25);
    if (intervalMs > 250) {
      // a hitch or a tab coming back: start over
      this.reset(0.5);
      return;
    }
    if (this.settle > 0) {
      this.settle -= dt;
      this.gpu.length = 0;
      return;
    }
    this.wall.push(intervalMs);
    this.cpu.push(cpuMs);
    this.chunkT += dt;
    if (this.chunkT < CHUNK) return;
    this.judge();
  }

  private judge() {
    const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / Math.max(1, a.length);
    const med = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1] ?? 0;
    this.updateFloor();
    if (this.level > this.floorLevel) this.set(this.floorLevel, 0, 'floor');
    let over: boolean, roomUp: boolean, cost: number;
    if (this.gpu.length >= 3) {
      cost = mean(this.gpu);
      this.lastSource = 'gpu';
      over = cost > HI;
      const next = this.level > 0 ? STEPS[this.level - 1] / STEPS[this.level] : Infinity;
      roomUp = cost * next < LO;
    } else {
      cost = med(this.wall);
      this.lastSource = 'wall';
      const cpuBound = med(this.cpu) > cost * 0.75;
      over = cost > WALL_HI && !cpuBound;
      roomUp = false;
    }
    this.lastCost = cost;
    this.wall.length = this.cpu.length = this.gpu.length = 0;
    this.chunkT = 0;
    this.over = over ? this.over + 1 : 0;
    this.under = roomUp ? this.under + 1 : 0;
    if (this.over >= DOWN_CHUNKS && this.level < STEPS.length - 1 && this.level < this.floorLevel) {
      if (this.clock - this.lastUpAt < 8) {
        this.upBlockedUntil = this.clock + this.blockFor;
        this.blockFor = Math.min(120, this.blockFor * 2);
      }
      this.set(this.level + 1, cost, 'over');
    } else if (this.under >= UP_CHUNKS && this.level > 0 && this.clock >= this.upBlockedUntil) {
      this.lastUpAt = this.clock;
      this.set(this.level - 1, cost, 'headroom');
    }
  }

  private set(level: number, cost: number, why: string) {
    if (level === this.level) return;
    this.history.push({ t: Math.round(this.clock * 10) / 10, from: STEPS[this.level], to: STEPS[level], cost: Math.round(cost * 100) / 100, why });
    if (this.history.length > 50) this.history.shift();
    this.level = level;
    this.engine.setRenderScale(this.scale);
    this.reset(SETTLE);
  }

  state() {
    const [w, h] = this.engine.bufferSize();
    return { scale: this.scale, pixels: STEPS[this.level], level: this.level, floorLevel: this.floorLevel, buffer: `${w}x${h}`,
      cost: Math.round(this.lastCost * 100) / 100, source: this.lastSource, history: this.history };
  }
}

/** installs the governor in play mode; null in photo mode, with ?scale= or ?gov=0 */
export function installGovernor(engine: Engine) {
  if (!governorWanted) return null;
  const gov = new ResolutionGovernor(engine);
  engine.onFrame = (iv, cpu) => gov.frame(iv, cpu);
  engine.onGpuFrame = (ms) => gov.gpuFrame(ms);
  return gov;
}
