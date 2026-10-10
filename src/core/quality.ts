/**
 * Graphics quality: four presets (Low, Medium, High, Ultra) and Auto, the default.
 *
 * Evaluated before any system reads `params` (Engine.ts imports it right after params.ts): the chosen preset writes
 * params' defaults, URL switches still win. High is the look and budget the game was tuned to (RTX 4060 at 1080p).
 *
 * Auto: a first guess from the GPU's name (the WebGL renderer string, read synchronously so module-level knobs such
 * as the grass range can use it), the CPU and memory, then a short measurement once the game shows (GPU timestamps,
 * else frame times): the highest preset predicted to hold 60 fps at the player's real resolution is remembered per
 * GPU in localStorage. A gentle dynamic render scale within the preset's range is the safety net during play.
 * Photo mode (?shot=) and automated test browsers use High unless ?quality= says otherwise, so captures stay fixed.
 *
 * Also here: the adapter probe (only supported limits are requested; no WebGPU means the WebGL 2 backend on Low, or a
 * clear message), the loading-screen GPU line, and recovery from a lost device (reload one preset lower).
 */
import { params, shotMode } from './params';
import type * as THREE from 'three/webgpu';
// (the loading line's and the menu's styles, ui/quality.css, are imported by main.ts and SettingsMenu.ts: this module
// also loads in the headless sims, through World.ts, where Node cannot import CSS)

export type Preset = 'low' | 'medium' | 'high' | 'ultra';
export type Mode = Preset | 'auto';
export const PRESETS: readonly Preset[] = ['low', 'medium', 'high', 'ultra'];
export const PRESET_LABEL: Record<Preset, string> = { low: 'Low', medium: 'Medium', high: 'High', ultra: 'Ultra' };

export interface Knobs {
  /** device pixel ratio cap */
  maxDpr: number;
  /** drawing-buffer cap in megapixels at render scale 1 (1440p is 3.7, 4K 8.3) */
  maxMP: number;
  /** lowest render scale Auto's dynamic resolution may use */
  minScale: number;
  shadowMap: number;
  /** sun shadow redraw every N frames (at up to ~90 fps; faster frame rates stretch it, see adaptRefresh) */
  shadowEvery: number;
  /** GTAO on/off, samples and resolution */
  ao: boolean;
  aoSamples: number;
  aoScale: number;
  /** mist pass resolution */
  mistScale: number;
  /** pool mirror on/off, resolution (fraction of the frame's) and refresh interval (frames) */
  mirror: boolean;
  mirrorScale: number;
  mirrorEvery: number;
  /** underwater light on the bed: caustics, beam absorption, refracted shadows */
  caustics: boolean;
  bloom: boolean;
  /** grass draw distance (x 31 m blades, x 36 m cards) */
  grassRange: number;
  /** forest LOD distances (x 40 / 85 / 360 m) */
  treeRange: number;
  /** crisp trout shadows on the bed (the nearest first) */
  fishShadows: number;
}

export const TABLE: Record<Preset, Knobs> = {
  low: {
    maxDpr: 1, maxMP: 0.92, minScale: 0.6, shadowMap: 1024, shadowEvery: 6, ao: false, aoSamples: 4, aoScale: 0.5,
    mistScale: 0.35, mirror: false, mirrorScale: 0.3, mirrorEvery: 8, caustics: false, bloom: true, grassRange: 0.6, treeRange: 0.65, fishShadows: 6,
  },
  medium: {
    maxDpr: 1.5, maxMP: 2.1, minScale: 0.7, shadowMap: 2048, shadowEvery: 4, ao: false, aoSamples: 6, aoScale: 0.5,
    mistScale: 0.5, mirror: true, mirrorScale: 0.35, mirrorEvery: 6, caustics: true, bloom: true, grassRange: 0.8, treeRange: 0.85, fishShadows: 10,
  },
  // today's look (part 8 round 2's mirror at 0.4); the sun's shadow map, fixed in place, only follows the trees' slow
  // sway, so it is redrawn every 6th frame (final pass: C3, W2 under 14 ms GPU p95; every 4th put them over)
  high: {
    maxDpr: 2, maxMP: 3.7, minScale: 0.75, shadowMap: 2048, shadowEvery: 6, ao: true, aoSamples: 8, aoScale: 0.5,
    mistScale: 0.5, mirror: true, mirrorScale: 0.4, mirrorEvery: 4, caustics: true, bloom: true, grassRange: 1, treeRange: 1, fishShadows: 16,
  },
  ultra: {
    maxDpr: 2, maxMP: 8.3, minScale: 0.8, shadowMap: 4096, shadowEvery: 2, ao: true, aoSamples: 8, aoScale: 1,
    mistScale: 1, mirror: true, mirrorScale: 0.6, mirrorEvery: 2, caustics: true, bloom: true, grassRange: 1.25, treeRange: 1.3, fishShadows: 16,
  },
};

/** GPU cost of a preset relative to High at the same drawing-buffer size (RTX 4060 at 1080p) */
const COST: Record<Preset, number> = { low: 0.55, medium: 0.8, high: 1, ultra: 1.55 };
/**
 * share of a frame's GPU time that does not scale with the pixel count (shadow map, geometry): the 4060 at High
 * measures 11.2 ms at 2.07 MP and 16.6 ms at 3.7 MP, which this predicts within 2 %
 */
const FIXED = 0.35;
const pixelCost = (mp: number, mpRef: number) => FIXED + (1 - FIXED) * (mp / mpRef);
/**
 * GPU ms per frame (median) for 60 fps once the mirror and shadow frames are added: the RTX 4060 at High and 1080p
 * measures 9.4-12.4 ms GPU with a wall p95 of 13-15.5 ms. Staying needs 14, rising a clear margin (12), and a lower
 * preset is picked under 13.
 */
const BUDGET_MS = 13;
const STAY_MS = 14;
const RISE_MS = 12;

const url = new URLSearchParams(location.search);
const LS_MODE = 'rm.quality';
const LS_AUTO = 'rm.quality.auto';

const store = {
  get(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, v: string) {
    try {
      localStorage.setItem(key, v);
    } catch {
      // private mode or storage off: Auto simply measures again next time
    }
  },
};

interface AutoEntry { preset: Preset; gpuMs?: number; lost?: number; at: number }
const readAuto = (): Record<string, AutoEntry> => {
  try {
    return JSON.parse(store.get(LS_AUTO) ?? '{}') as Record<string, AutoEntry>;
  } catch {
    return {};
  }
};
const writeAuto = (key: string, e: Partial<AutoEntry>) => {
  const all = readAuto();
  all[key] = { ...(all[key] ?? { preset: 'high', at: 0 }), ...e, at: Date.now() };
  store.set(LS_AUTO, JSON.stringify(all));
};

const isPreset = (s: string | null): s is Preset => s !== null && (PRESETS as readonly string[]).includes(s);
const step = (p: Preset, d: number): Preset => PRESETS[Math.max(0, Math.min(PRESETS.length - 1, PRESETS.indexOf(p) + d))];

/** the GPU as WebGL names it (ANGLE on Windows: "ANGLE (NVIDIA, NVIDIA GeForce RTX 4060 (0x00002882) Direct3D11 ...") */
function webglRenderer(): string {
  try {
    const c = document.createElement('canvas');
    const gl = (c.getContext('webgl2') ?? c.getContext('webgl')) as WebGLRenderingContext | null;
    if (!gl) return '';
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const r = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return r;
  } catch {
    return '';
  }
}

function cleanName(r: string): string {
  const m = /^ANGLE \((?:[^,]+), (.+?)(?: \(0x[0-9a-f]+\))?(?: Direct3D| OpenGL| Vulkan| Metal|, )/i.exec(r);
  return (m ? m[1] : r).replace(/\s+/g, ' ').trim();
}

type Kind = 'discrete' | 'integrated' | 'software' | 'mobile' | 'unknown';

function classify(name: string, vendor = ''): { kind: Kind; guess: Preset } {
  const n = `${vendor} ${name}`.toLowerCase();
  const mobile = /android|iphone|ipad/i.test(navigator.userAgent);
  if (/swiftshader|llvmpipe|softpipe|basic render|microsoft basic|software/.test(n)) return { kind: 'software', guess: 'low' };
  if (mobile || /adreno|mali|powervr|apple a\d/.test(n)) return { kind: 'mobile', guess: 'low' };
  if (/apple/.test(n)) return { kind: 'integrated', guess: /m\d (max|ultra)/.test(n) ? 'high' : 'medium' };
  if (/nvidia|geforce|quadro|rtx|gtx/.test(n)) {
    if (/\bmx ?\d/.test(n)) return { kind: 'discrete', guess: 'low' };
    const rtx = /rtx\s*a?(\d{4})/.exec(n);
    if (rtx) {
      const tier = Number(rtx[1]) % 100, gen = Math.floor(Number(rtx[1]) / 100);
      if (tier >= 80 && gen >= 30) return { kind: 'discrete', guess: 'ultra' };
      return { kind: 'discrete', guess: tier >= 60 || gen >= 40 ? 'high' : 'medium' };
    }
    if (/gtx\s*(10[5-9]0|16[5-6]0|9\d0)/.test(n)) return { kind: 'discrete', guess: /1050|1650|9\d0/.test(n) ? 'low' : 'medium' };
    return { kind: 'discrete', guess: 'high' };
  }
  if (/amd|radeon|ati/.test(n)) {
    const rx = /rx\s*(\d{3,4})/.exec(n);
    if (rx) {
      const m = Number(rx[1]);
      if (m >= 7900 || (m >= 6800 && m < 7000)) return { kind: 'discrete', guess: 'ultra' };
      if (m >= 6600 || (m >= 5700 && m < 6000)) return { kind: 'discrete', guess: 'high' };
      return { kind: 'discrete', guess: m >= 5000 ? 'medium' : 'low' };
    }
    // APUs: "AMD Radeon(TM) Graphics", "Radeon 780M", "Vega 8"
    if (/radeon\(tm\) graphics|radeon graphics|vega|\d{3}m\b/.test(n)) return { kind: 'integrated', guess: 'low' };
    return { kind: 'unknown', guess: 'high' };
  }
  if (/intel/.test(n)) {
    if (/arc.*a\d{3}|arc\(tm\) a|b5\d0|b7\d0/.test(n)) return { kind: 'discrete', guess: /a3\d0/.test(n) ? 'medium' : 'high' };
    return { kind: 'integrated', guess: 'low' };
  }
  return { kind: 'unknown', guess: 'high' };
}

function decide(kind: Kind, guess: Preset): Preset {
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 8;
  const cores = navigator.hardwareConcurrency || 8;
  let p = guess;
  if ((mem <= 4 || cores <= 4) && PRESETS.indexOf(p) > 1) p = 'medium';
  if (kind === 'software') p = 'low';
  return p;
}

// ---- the choice, made synchronously before any system reads params ----

const webdriver = navigator.webdriver === true;
const urlQ = url.get('quality');
const urlBackend = url.get('backend');
const renderer0 = webglRenderer();
const name0 = cleanName(renderer0);
const class0 = classify(name0);

export const Q = {
  /** what the player asked for (Auto by default) */
  mode: 'auto' as Mode,
  /** the preset in use */
  preset: 'high' as Preset,
  k: TABLE.high,
  /** fixed for captures and tests: no calibration, no dynamic resolution */
  fixed: false,
  /** where the preset came from: url, shot, test, choice, stored, guess, webgl, lost */
  source: '',
  gpuName: name0 || 'unknown GPU',
  gpuKey: (name0 || 'unknown').toLowerCase(),
  kind: class0.kind,
  backend: (urlBackend === 'webgl' ? 'webgl' : 'webgpu') as 'webgpu' | 'webgl',
  /** what Auto measured and picked on this GPU (shown in the menu) */
  autoPick: null as Preset | null,
  autoMs: 0,
  /** dynamic render scale (1 = the preset's full resolution) */
  scale: 1,
  limits: {} as Record<string, number>,
  timestamps: false,
};

function choose() {
  const stored = readAuto()[Q.gpuKey];
  if (stored && isPreset(stored.preset)) Q.autoPick = stored.preset;
  const userMode = store.get(LS_MODE);
  let p: Preset;
  if (isPreset(urlQ)) [p, Q.mode, Q.fixed, Q.source] = [urlQ, urlQ, true, 'url'];
  else if (urlQ === 'auto') [p, Q.mode, Q.source] = [Q.autoPick ?? decide(Q.kind, class0.guess), 'auto', Q.autoPick ? 'stored' : 'guess'];
  else if (shotMode) [p, Q.mode, Q.fixed, Q.source] = ['high', 'high', true, 'shot'];
  else if (webdriver) [p, Q.mode, Q.fixed, Q.source] = ['high', 'high', true, 'test'];
  else if (isPreset(userMode)) [p, Q.mode, Q.source] = [userMode, userMode, 'choice'];
  else [p, Q.mode, Q.source] = [Q.autoPick ?? decide(Q.kind, class0.guess), 'auto', Q.autoPick ? 'stored' : 'guess'];
  if (Q.backend === 'webgl' && !isPreset(urlQ)) [p, Q.source] = ['low', 'webgl'];
  apply(p);
}

/** write the preset into params' defaults (URL switches keep their value) */
function apply(p: Preset) {
  Q.preset = p;
  Q.k = TABLE[p];
  const k = Q.k;
  const set = <K extends keyof typeof params>(key: K, switchName: string, v: (typeof params)[K]) => {
    if (!url.has(switchName)) params[key] = v;
  };
  set('shadowMap', 'shadowmap', k.shadowMap);
  set('shadowEvery', 'shadowevery', k.shadowEvery);
  set('aoSamples', 'aosamples', k.aoSamples);
  set('mirrorScale', 'mirrorscale', k.mirrorScale);
  set('mirrorEvery', 'mirrorevery', k.mirrorEvery);
  set('refl', 'refl', k.mirror);
  set('caus', 'caus', k.caustics);
  Q.timestamps = Q.mode === 'auto' && !Q.fixed;
}

choose();

/** the renderer's pixel ratio: the preset's DPR and megapixel caps, then ?scale= and the dynamic scale */
export function pixelRatio(): number {
  const dpr = Math.min(window.devicePixelRatio || 1, Q.k.maxDpr);
  const css = Math.max(1, window.innerWidth * window.innerHeight);
  const cap = Math.sqrt((Q.k.maxMP * 1e6) / css);
  return Math.min(dpr, cap) * params.scale * Q.scale;
}

/** grass instance capacity for the preset's draw distance (World.ts) */
export const grassCap = () => Q.k.grassRange * Q.k.grassRange;

// ---- adapter probe, loading line, WebGL fallback, device loss ----

let gpuLine: HTMLElement | null = null;
function loadingNote(text: string, hint?: string) {
  const inner = document.querySelector('#loading .loading-inner');
  if (!inner) return;
  if (!gpuLine) {
    gpuLine = document.createElement('div');
    gpuLine.className = 'loading-gpu';
    inner.appendChild(gpuLine);
  }
  gpuLine.textContent = text;
  if (hint) {
    const h = document.createElement('div');
    h.className = 'loading-hint';
    h.textContent = hint;
    inner.appendChild(h);
  }
}

/** shown instead of the game when neither WebGPU nor (acceptable) WebGL 2 is there */
export function fatal(title: string, body: string): never {
  const inner = document.querySelector('#loading .loading-inner');
  if (inner) {
    const box = document.createElement('div');
    box.className = 'loading-fatal';
    const t = document.createElement('div');
    t.className = 'loading-fatal-title';
    t.textContent = title;
    const b = document.createElement('div');
    b.textContent = body;
    box.append(t, b);
    inner.appendChild(box);
    inner.querySelector('.loading-bar')?.remove();
  }
  throw new Error(title);
}

const NO_WEBGPU =
  'River Morning needs WebGPU. It runs in Chrome or Edge (version 113 or newer) on Windows, macOS and ChromeOS, ' +
  'Chrome on recent Android phones, Safari 26 on macOS and iOS, and Firefox 141 or newer on Windows.';

/**
 * Before the renderer exists: one adapter request like three's own (same feature level), limits capped by what it
 * supports, the WebGL 2 backend on Low when WebGPU is missing, and the GPU named on the loading screen.
 */
export async function probeGpu(): Promise<void> {
  let adapter: GPUAdapter | null = null;
  if (Q.backend === 'webgpu') {
    try {
      adapter = (await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance', featureLevel: 'compatibility' } as GPURequestAdapterOptions)) ?? null;
    } catch {
      adapter = null;
    }
    if (!adapter) {
      if (!('WebGL2RenderingContext' in window) || !renderer0) fatal('WebGPU is not available', NO_WEBGPU);
      Q.backend = 'webgl';
      if (!isPreset(urlQ)) apply('low');
      Q.source = 'webgl';
    }
  }
  if (adapter) {
    const lim = adapter.limits;
    Q.limits = {
      maxSampledTexturesPerShaderStage: Math.min(32, lim.maxSampledTexturesPerShaderStage),
      maxSamplersPerShaderStage: Math.min(32, lim.maxSamplersPerShaderStage),
    };
    const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
    const vendor = info?.vendor ?? '';
    const fallback = (info as (GPUAdapterInfo & { isFallbackAdapter?: boolean }) | undefined)?.isFallbackAdapter
      ?? (adapter as GPUAdapter & { isFallbackAdapter?: boolean }).isFallbackAdapter ?? false;
    // WebGL may run on a different GPU than WebGPU (two-GPU laptops): then trust WebGPU's vendor and re-decide
    if (vendor && name0 && !name0.toLowerCase().includes(vendor.toLowerCase().slice(0, 3))) {
      Q.gpuName = `${vendor} ${info?.architecture ?? ''}`.trim();
      Q.gpuKey = `webgpu:${Q.gpuName}`.toLowerCase();
      const c = classify(Q.gpuName, vendor);
      Q.kind = c.kind;
      if (!Q.fixed && Q.mode === 'auto') {
        const stored = readAuto()[Q.gpuKey];
        Q.autoPick = stored && isPreset(stored.preset) ? stored.preset : null;
        apply(Q.autoPick ?? decide(c.kind, c.guess));
      }
    } else if (!name0 && vendor) {
      Q.gpuName = `${vendor} ${info?.architecture ?? ''}`.trim();
    }
    if (fallback) {
      Q.kind = 'software';
      if (!Q.fixed) apply('low');
    }
  }
  window.__RM.adapter = Q.gpuName;
  const label = `${Q.gpuName}  ·  ${Q.backend === 'webgpu' ? 'WebGPU' : 'WebGL 2'}`;
  const hint = Q.kind === 'software'
    ? 'Running without graphics acceleration: turn on hardware acceleration in your browser settings'
    : Q.kind === 'integrated' && !/apple/i.test(Q.gpuName)
      ? 'Running on integrated graphics: for full quality, set your browser to use the high-performance GPU'
      : Q.backend === 'webgl' ? 'WebGPU is not available here, so the game runs in a simpler mode' : undefined;
  loadingNote(label, hint);
}

let unloading = false;
let lostHandled = false;
if (typeof addEventListener === 'function') {
  addEventListener('pagehide', () => (unloading = true));
  addEventListener('beforeunload', () => (unloading = true));
}

function onDeviceLost(reason: string, message: string) {
  if (unloading || lostHandled) return;
  lostHandled = true;
  // nothing more can be drawn with this device: stop the loop instead of failing every frame until the reload
  liveRenderer?.setAnimationLoop(null);
  const lower = step(Q.preset, -1);
  const entry = readAuto()[Q.gpuKey];
  const losses = (entry?.lost ?? 0) + 1;
  console.error(`device lost (${reason}): ${message}`);
  if (window.__RM) window.__RM.error = `device lost (${reason}): ${message}`;
  const give = Q.preset === 'low' && losses >= 2;
  const el = document.createElement('div');
  el.className = 'device-lost';
  el.textContent = give
    ? 'The graphics card stopped responding. Try closing other apps or restarting the browser, then reload.'
    : `The graphics card stopped responding. Restarting at ${PRESET_LABEL[lower]} quality…`;
  document.body.appendChild(el);
  if (shotMode || give) return;
  writeAuto(Q.gpuKey, { preset: lower, lost: losses });
  if (Q.mode !== 'auto') store.set(LS_MODE, lower);
  const next = new URL(location.href);
  if (isPreset(urlQ)) next.searchParams.set('quality', lower);
  setTimeout(() => location.replace(next.toString()), 2500);
}

/** after renderer.init(): the backend that really runs, and the device-loss hooks */
export function afterInit(renderer: THREE.WebGPURenderer) {
  liveRenderer = renderer;
  const backend = renderer.backend as unknown as { isWebGPUBackend?: boolean; device?: GPUDevice };
  if (!backend.isWebGPUBackend && Q.backend === 'webgpu') {
    // three fell back to WebGL 2 by itself (the device request failed)
    Q.backend = 'webgl';
    if (!isPreset(urlQ)) apply('low');
    renderer.setPixelRatio(pixelRatio());
    loadingNote(`${Q.gpuName}  ·  WebGL 2`, 'WebGPU could not start here, so the game runs in a simpler mode');
  }
  // three reports losses other than 'destroyed' (and WebGL context loss) here
  (renderer as unknown as { onDeviceLost: (i: { reason?: string; message?: string }) => void }).onDeviceLost = (i) =>
    onDeviceLost(i.reason ?? 'unknown', i.message ?? '');
  // 'destroyed' too: the ?debug hook below destroys the device to test this path
  backend.device?.lost.then((i) => onDeviceLost(i.reason, i.message));
  if (params.debug && backend.device) {
    (window.__RM as unknown as { loseDevice: () => void }).loseDevice = () => backend.device!.destroy();
  }
}

// ---- in play: warm-up at low resolution, calibration, dynamic resolution, refresh pacing ----

interface EngineLike {
  renderer: THREE.WebGPURenderer;
  addUpdater(fn: (dt: number, t: number) => void): void;
  frameLog(): { wall: number[]; gpu: number[] };
  hold: boolean;
}
interface WorldLike {
  water?: { mirror?: { every: number; resolutionScale: number } | null } | null;
  forest?: unknown;
}

const median = (a: number[]) => {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
};
const pct = (a: number[], p: number) => {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(s.length * p))];
};

/** megapixels of a preset at render scale 1 in this window */
function presetMP(p: Preset) {
  const k = TABLE[p];
  const dpr = Math.min(window.devicePixelRatio || 1, k.maxDpr);
  const css = window.innerWidth * window.innerHeight;
  return Math.min(css * dpr * dpr, k.maxMP * 1e6) / 1e6;
}

export const qualityLog: string[] = [];

/** called once the world and the post chain exist, before the loop starts */
export async function startQuality(engine: EngineLike, world: WorldLike) {
  const r = (liveRenderer = engine.renderer);
  (window.__RM as unknown as { quality: () => unknown }).quality = qualityState;
  // the forest's LOD distances (read every repack)
  if (Q.k.treeRange !== 1) {
    const { LOD_DIST } = await import('../world/Forest');
    for (let i = 0; i < LOD_DIST.length; i++) LOD_DIST[i] = Math.round(LOD_DIST[i] * Q.k.treeRange);
  }
  // weak GPUs: the warm-up frames draw everything unculled; at a tiny resolution they stay far from a driver timeout
  const tinyWarm = (Q.preset === 'low' || Q.preset === 'medium') && !url.has('scale');
  if (tinyWarm) r.setPixelRatio(Math.min(pixelRatio(), 480 / Math.max(1, window.innerHeight)));

  const mirror = world.water?.mirror ?? null;
  const adaptShadow = !url.has('shadowevery') && !shotMode;
  const adaptMirror = !url.has('mirrorevery') && !shotMode;
  let readyAt = 0, lastCheck = 0, over = 0, under = 0, cooldown = 0;
  let calibrated = Q.mode !== 'auto' || Q.fixed;
  let menuMounted = false;

  engine.addUpdater(() => {
    const now = performance.now();
    if (!window.__RM.ready) return;
    if (!readyAt) {
      readyAt = now;
      if (tinyWarm) (engine as unknown as { resize(): void }).resize();
      if (!shotMode && !menuMounted) {
        menuMounted = true;
        void import('../ui/SettingsMenu').then((m) => m.mountSettingsMenu(engine));
      }
      return;
    }
    if (now - lastCheck < 1000) return;
    lastCheck = now;
    const log = engine.frameLog();
    const wall = log.wall.slice(-60), gpu = log.gpu.slice(-60);
    const fps = 1000 / Math.max(1, median(wall));

    // frame-count schedules tuned at up to ~100 fps: stretched at higher rates (240 Hz), never denser than the preset's
    if (adaptShadow) params.shadowEvery = Math.max(Q.k.shadowEvery, Math.round((Q.k.shadowEvery * fps) / 100));
    if (adaptMirror && mirror) mirror.every = Math.max(Q.k.mirrorEvery, Math.round((Q.k.mirrorEvery * fps) / 100));

    if (!calibrated) {
      if (now - readyAt < 4500) return;
      calibrated = true;
      calibrate(engine, log, mirror);
      return;
    }
    // (one controller only: ?dynres=0, or the robust lane's ResolutionGovernor if it is merged too)
    if (Q.mode !== 'auto' || Q.fixed || url.has('scale') || url.get('dynres') === '0' || 'governor' in window.__RM || now < cooldown) return;
    // the safety net: slow, with hysteresis; 5 % steps are not visible as a pop
    const w50 = median(wall), w90 = pct(wall, 0.9), g50 = gpu.length >= 20 ? median(gpu.slice(-30)) : 0;
    // aim: a 60 fps median; with GPU timings, rise only with clear GPU headroom (a 60 Hz vsync hides it in wall time)
    // a slow main thread with an idle GPU (weak CPU) is not helped by fewer pixels: wall time counts only then
    const cpuBound = g50 > 0 && g50 < 12;
    const isOver = (!cpuBound && (w50 > 17 || w90 > 19.5)) || g50 > 14.5;
    const isUnder = w90 < 18 && (g50 > 0 ? g50 < 11 : w90 < 17.6);
    over = isOver ? over + 1 : 0;
    under = isUnder ? under + 1 : 0;
    let next = Q.scale;
    if (over >= 2) next = Math.max(Q.k.minScale, Q.scale - 0.05);
    else if (under >= 6) next = Math.min(1, Q.scale + 0.05);
    if (next !== Q.scale) {
      Q.scale = Math.round(next * 100) / 100;
      (engine as unknown as { resize(): void }).resize();
      over = under = 0;
      cooldown = now + 2500;
      qualityLog.push(`${Math.round(now)} scale ${Q.scale} (wall p90 ${w90.toFixed(1)}, gpu ${g50.toFixed(1)})`);
    }
  });
}

/**
 * Auto, a few seconds into play: predict each preset's GPU time from this one's (relative costs x pixel counts) and
 * remember the highest that holds 60 fps. Going up waits for the next load (the menu says so) while this session
 * gets the extra resolution; going down a lot, before the player has clicked in, reloads at once.
 */
function calibrate(engine: EngineLike, log: { wall: number[]; gpu: number[] }, mirror: { every: number; resolutionScale: number } | null) {
  const wall = log.wall.slice(-150), gpu = log.gpu.slice(-150);
  // the frames were drawn at the resolution governor's current share of the preset's pixels
  const gov = (window.__RM as unknown as { governor?: () => { pixels?: number } }).governor?.();
  const mpNow = presetMP(Q.preset) * (gov?.pixels ?? 1);
  const haveGpu = gpu.length >= 40;
  const ms = haveGpu ? median(gpu) : median(wall);
  const w90 = pct(wall, 0.9);
  let pick: Preset = 'low';
  const cur = PRESETS.indexOf(Q.preset);
  for (const p of PRESETS) {
    const i = PRESETS.indexOf(p);
    const est = ms * (COST[p] / COST[Q.preset]) * pixelCost(presetMP(p), mpNow);
    // without GPU timings a vsync-capped frame hides the headroom: never go above the current preset
    if (!haveGpu && i > cur) break;
    // hysteresis: rising needs a clear margin, staying a little less (estimates are not exact)
    const limit = haveGpu ? (i > cur ? RISE_MS : i === cur ? STAY_MS : BUDGET_MS) : i === cur ? 18 : 16.9;
    if (est <= limit) pick = p;
  }
  // the main thread can be the limit too (a slow CPU): keep a preset whose wall time already misses 60 fps from rising
  if (w90 > 20 && PRESETS.indexOf(pick) > PRESETS.indexOf(Q.preset)) pick = Q.preset;
  Q.autoPick = pick;
  Q.autoMs = ms;
  writeAuto(Q.gpuKey, { preset: pick, gpuMs: Math.round(ms * 10) / 10 });
  qualityLog.push(`calibrate ${Q.preset}: ${haveGpu ? 'gpu' : 'wall'} ${ms.toFixed(1)} ms at ${mpNow.toFixed(2)} MP (wall p90 ${w90.toFixed(1)}) -> ${pick}`);
  const di = PRESETS.indexOf(pick) - PRESETS.indexOf(Q.preset);
  if (di < 0 && ms > STAY_MS * 1.35 && !document.pointerLockElement && !isPreset(urlQ)) {
    const el = document.createElement('div');
    el.className = 'device-lost';
    el.textContent = `Adjusting graphics for this device (${PRESET_LABEL[pick]})…`;
    document.body.appendChild(el);
    engine.hold = true;
    setTimeout(() => location.reload(), 600);
    return;
  }
  if (di > 0) {
    // this session: the higher preset's resolution and refresh rates (structure follows on the next load)
    const k = TABLE[pick];
    Q.k = { ...Q.k, maxDpr: k.maxDpr, maxMP: k.maxMP, shadowEvery: k.shadowEvery, mirrorEvery: k.mirrorEvery };
    if (mirror) mirror.resolutionScale = Math.max(mirror.resolutionScale, k.mirrorScale);
    (engine as unknown as { resize(): void }).resize();
  }
}

let liveRenderer: THREE.WebGPURenderer | null = null;

export function qualityState() {
  const c = liveRenderer?.domElement;
  return {
    mode: Q.mode, preset: Q.preset, source: Q.source, fixed: Q.fixed, autoPick: Q.autoPick, autoMs: Math.round(Q.autoMs * 10) / 10,
    scale: Q.scale, pixelRatio: Math.round(pixelRatio() * 1000) / 1000, buffer: c ? `${c.width}x${c.height}` : '',
    gpu: Q.gpuName, kind: Q.kind, backend: Q.backend, shadowEvery: params.shadowEvery, log: qualityLog,
  };
}

/** the menu's choice: stored, applied on reload */
export function chooseMode(m: Mode) {
  store.set(LS_MODE, m);
  const next = new URL(location.href);
  next.searchParams.delete('quality');
  location.replace(next.toString());
}
