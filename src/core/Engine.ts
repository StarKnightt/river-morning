import * as THREE from 'three/webgpu';
import { DEFAULT_MAX_DIM, fitPixelRatio, guardRenderer } from './deviceGuard';
import { governorWanted } from './ResolutionGovernor';
import {
  builtinAOContext, float, mrt, normalView, pass, renderOutput, screenUV, uniform, vec3, vec4,
  dot, mix, smoothstep, length, rtt, type N,
} from '../gfx/tsl';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { smaa } from 'three/addons/tsl/display/SMAANode.js';
import { gaussianBlur } from 'three/addons/tsl/display/GaussianBlurNode.js';
import { params } from './params';
import { Q, afterInit, pixelRatio, probeGpu } from './quality';
import { mistFilter, sharpen } from '../gfx/postFilters';

export type Updater = (dt: number, t: number) => void;

/** Display-space grade applied after tone mapping (film-like, slightly warm highlights). */
export const grade = {
  saturation: uniform(1.0),
  contrast: uniform(1.13),
  lift: uniform(new THREE.Vector3(0.0, 0.003, 0.008)),
  gain: uniform(new THREE.Vector3(1.03, 1.0, 0.96)),
  vignette: uniform(0.22),
  exposure: uniform(1.0),
  /** 0 by day; at night (part 8) bright warm light (fire, the lit window, firelit ground) keeps its colour under the
   * low `saturation`, as the eye sees lamplight in colour while the moonlit scene around it reads grey-blue */
  keepWarm: uniform(0),
  /** saturation of the sky's own pixels (part 8: a deep blue night sky over a nearly monochrome moonlit valley) */
  skySat: uniform(1.0),
};

export class Engine {
  readonly renderer: THREE.WebGPURenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.05, 20000);
  /** the GTAO pre-pass camera: a copy of `camera` that sees layer 0 only */
  readonly preCam = new THREE.PerspectiveCamera(50, 16 / 9, 0.05, 20000);
  pipeline!: THREE.RenderPipeline;
  readonly timer = new THREE.Timer();
  private updaters: Updater[] = [];
  /** optional volumetric mist: (scene depth texture node) -> (in-scattered light rgb, transmittance a), half res */
  mistNode: ((depth: N) => N) | null = null;
  /** frozen time for deterministic captures */
  fixedTime: number | null = null;
  time = 0;
  frame = 0;
  private frameTimes: number[] = [];
  private gpuTimes: number[] = [];
  private resolving = false;
  /**
   * What the frame being built does beyond the plain passes, set by updaters: bit 1 the pool mirror rendered,
   * bit 2 the sun's shadow map is redrawn. Frame stats are split by it (the heavy frames decide the p95).
   */
  frameKind = 0;
  private kinds: number[] = [];
  private cpuTimes: number[] = [];
  private gpuKinds: number[] = [];
  private lastKind = 0;
  private lastCpu = 0;
  /**
   * Work that runs right after a frame, in a task of its own: the periodic pool-mirror refresh and sun shadow redraw.
   * The display shows a frame once the GPU has finished it, rounded up to the next refresh (5 ms at 200 Hz), so a
   * frame that also carried that work slipped by a whole refresh; after the frame it fills the GPU's wait instead.
   */
  private postFns: Array<() => void> = [];
  private readonly postChannel = new MessageChannel();

  /** Ask for the higher per-stage texture limit the terrain shaders need, capped by the adapter. */
  static async create(container: HTMLElement) {
    // quality.ts: one adapter probe (limits it supports, WebGL 2 on Low without WebGPU)
    await probeGpu();
    const e = new Engine(container, Q.limits);
    await e.renderer.init();
    // robust's GPU error log first, then quality.ts's device-loss recovery (reload one preset lower) replaces its overlay
    guardRenderer(e.renderer);
    afterInit(e.renderer);
    const device = (e.renderer.backend as unknown as { device?: GPUDevice }).device;
    if (device) e.maxDim = device.limits.maxTextureDimension2D;
    e.resize();
    return e;
  }

  /** dynamic resolution (ResolutionGovernor): a factor on the pixel ratio, 1 = native */
  renderScale = 1;
  /** the device's 2D texture limit: the drawing buffer must fit it on both sides */
  maxDim = DEFAULT_MAX_DIM;
  /** per frame: wall interval since the last frame and the last frame's main-thread time (ms) */
  onFrame: ((intervalMs: number, cpuMs: number) => void) | null = null;
  /** per GPU-timed frame (timestamp queries on): its GPU ms */
  onGpuFrame: ((ms: number) => void) | null = null;

  /** native resolution on scaled displays (Windows 120 % gives 1.2), capped at 2, times ?scale */
  private basePixelRatio() {
    return pixelRatio();
  }

  /** the drawing buffer at render scale 1 (pixels), after the device and MAX_PIXELS clamps */
  nativePixels() {
    const w = window.innerWidth, h = window.innerHeight, pr = fitPixelRatio(w, h, this.basePixelRatio(), this.maxDim);
    return Math.floor(w * pr) * Math.floor(h * pr);
  }

  bufferSize(): [number, number] {
    const c = this.renderer.domElement;
    return [c.width, c.height];
  }

  setRenderScale(s: number) {
    if (s === this.renderScale) return;
    this.renderScale = s;
    this.resize();
  }

  private constructor(container: HTMLElement, requiredLimits: Record<string, number>) {
    this.renderer = new THREE.WebGPURenderer({
      antialias: false,
      powerPreference: 'high-performance',
      reversedDepthBuffer: true,
      requiredLimits,
      forceWebGL: Q.backend === 'webgl',
      // GPU timestamps measure our own passes even when another app shares the GPU
      // (the resolution governor judges GPU time, not the vsync-bound wall clock; and Auto's calibration)
      trackTimestamp: params.shot !== null || params.debug || governorWanted || Q.timestamps,
    } as ConstructorParameters<typeof THREE.WebGPURenderer>[0]);
    // the pixel ratio: see basePixelRatio and resize (clamped to the device's texture limit and MAX_PIXELS)
    this.renderer.setPixelRatio(fitPixelRatio(window.innerWidth, window.innerHeight, this.basePixelRatio(), this.maxDim));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.toneMapping = params.lin ? THREE.NoToneMapping : THREE.AgXToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    container.appendChild(this.renderer.domElement);
    window.addEventListener('resize', () => this.resize());
    this.watchPixelRatio();
    this.resize();
  }

  /** moving the window to a screen with other scaling, or browser zoom, changes devicePixelRatio */
  private watchPixelRatio() {
    const mq = matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
    mq.addEventListener('change', () => {
      this.resize();
      this.watchPixelRatio();
    }, { once: true });
  }

  get backendName(): string {
    const b = (this.renderer as unknown as { backend: { isWebGPUBackend?: boolean } }).backend;
    return b.isWebGPUBackend ? 'WebGPU' : 'WebGL2';
  }

  setupPost() {
    const { scene, camera, renderer } = this;
    const pipeline = new THREE.RenderPipeline(renderer);
    pipeline.outputColorTransform = false;

    if (params.noPost) {
      const sp = pass(scene, camera);
      pipeline.outputNode = renderOutput(sp);
      this.pipeline = pipeline;
      return;
    }

    // normal/depth pre-pass feeds GTAO; AO only attenuates indirect light. It draws every mesh with one cheap
    // material (geometric normals): GTAO wants the large-scale shape, stone-scale occlusion comes from the
    // texture AO, and re-running the layered terrain shader here cost ~3 ms.
    // layer 0 only: the override material would draw foliage cards as solid quads (AO halos round every crown),
    // and grass and mist are too fine or too soft for half-resolution AO. Its own camera (synced every frame), not
    // setLayers: the pre-pass renders nested inside the scene pass (the AO texture is only reached through the
    // scene pass's context), and with the same camera it rebuilt the scene pass's render list shorter mid-loop.
    this.preCam.layers.set(0);
    const prePass = pass(params.noAO ? new THREE.Scene() : scene, this.preCam);
    prePass.name = 'prepass';
    prePass.transparent = false;
    // at the AO's own resolution (half on High): GTAO reads it at that size anyway, and the full-resolution copy cost
    // 0.1-0.5 ms on the water views for no visible change (final pass A/B: under 0.11 % of pixels moved by > 8/255)
    prePass.setResolutionScale(params.preScale ?? Q.k.aoScale);
    const preMat = new THREE.MeshBasicNodeMaterial();
    preMat.name = 'prepass';
    preMat.colorNode = vec3(0);
    prePass.overrideMaterial = preMat;
    prePass.setMRT(mrt({ output: normalView }));
    const preNormal = prePass.getTextureNode();
    const preDepth = prePass.getTextureNode('depth');
    const aoPass = ao(preDepth, preNormal, camera);
    aoPass.resolutionScale = Q.k.aoScale;
    aoPass.radius.value = 0.6;
    aoPass.distanceExponent.value = 1.4;
    aoPass.thickness.value = 1.2;
    aoPass.samples.value = params.aoSamples;
    const aoTex = aoPass.getTextureNode();

    const scenePass = pass(scene, camera);
    scenePass.name = 'scene';
    if (Q.k.ao) scenePass.contextNode = builtinAOContext(aoTex.sample(screenUV).r);
    let hdr = scenePass.getTextureNode('output');
    if (this.mistNode) {
      // the mist is soft: half resolution is enough (it cost 4.5-7 ms marched at full resolution in the scene pass)
      // filtered without crossing depth or mist edges (postFilters.ts); ?mistfilter=old: the plain Gaussian (A/B)
      const mistRT = rtt(this.mistNode(scenePass.getTextureNode('depth')), null, null, { resolutionScale: Q.k.mistScale });
      const mist = (params.mistFilterOld ? gaussianBlur(mistRT, 1, 1.6) : mistFilter(mistRT, scenePass.getTextureNode('depth'))) as N;
      hdr = params.mistDbg ? vec4(mist.rgb.mul(6), 1).add(vec4(float(1).sub(mist.a).mul(0.5), 0, 0, 0)) : hdr.mul(mist.a).add(mist.rgb);
    }

    // the HDRI sun disc is ~1e5; clamp so the glare stays a camera bloom, not a white-out
    const bloomPass = bloom(hdr.min(vec3(params.bloomMax ?? 48)), 0.14, 0.32, 2.2);
    const combined = (Q.k.bloom ? hdr.add(bloomPass) : hdr).mul(grade.exposure);

    // tone map + sRGB, then grade in display space
    const display = renderOutput(combined, THREE.AgXToneMapping, THREE.SRGBColorSpace);
    const lum = dot(display.rgb, vec3(0.2126, 0.7152, 0.0722));
    // (part 8 round 2: judged on the scene colour, not the tone-mapped one, and from a dimmer level: AgX greys bright
    // orange toward cream, so the firelit ash and the ground a few metres out failed the test and went beige)
    const lumC = dot(combined.rgb, vec3(0.2126, 0.7152, 0.0722));
    const warmC = combined.r.sub(combined.b).div(combined.r.add(combined.g).add(combined.b).max(1e-4));
    const keep = grade.keepWarm.mul(smoothstep(0.012, 0.07, lumC)).mul(smoothstep(0.06, 0.3, warmC));
    // reversed depth: the sky is the cleared far plane
    const isSky = scenePass.getTextureNode('depth').sample(screenUV).x.lessThan(1e-7).select(float(1), float(0));
    let c = mix(vec3(lum), display.rgb, mix(mix(grade.saturation, grade.skySat, isSky), float(1), keep));
    c = c.sub(0.5).mul(grade.contrast).add(0.5);
    c = c.mul(grade.gain).add(grade.lift.mul(float(1).sub(c)));
    const v = length(screenUV.sub(0.5).mul(vec3(1.0, 0.75, 0).xy));
    c = c.mul(mix(float(1), float(1).sub(grade.vignette), smoothstep(0.35, 0.85, v)));
    const graded = vec4(c.clamp(0, 1), 1);

    const aa = smaa(graded);
    pipeline.outputNode = params.sharpen > 0 ? sharpen(aa.getTextureNode(), params.sharpen) : aa;
    this.pipeline = pipeline;
  }

  private syncPreCam() {
    const c = this.camera, p = this.preCam;
    c.updateMatrixWorld();
    p.position.copy(c.position);
    p.quaternion.copy(c.quaternion);
    p.fov = c.fov;
    p.aspect = c.aspect;
    p.near = c.near;
    p.far = c.far;
    p.zoom = c.zoom;
    p.updateProjectionMatrix();
    p.updateMatrixWorld();
  }

  addUpdater(fn: Updater) {
    this.updaters.push(fn);
  }

  /** run `fn` after every frame (see postFns); it may set frameKind bits for the work it did */
  addPostFrame(fn: () => void) {
    this.postFns.push(fn);
  }

  private runPost() {
    this.frameKind = this.lastKind;
    for (const fn of this.postFns) fn();
    this.lastKind = this.frameKind;
    this.resolveGpu(this.lastKind);
  }

  /**
   * The GPU time of every frame, the work after it included. three resolves the timestamp queries asynchronously and
   * returns only the last frame's total, but keeps every pass of the frames since the previous resolve; those are
   * summed here per frame id. (One sample per resolve came every 2-3 frames and locked onto the 3- and 4-frame
   * cadences of the shadow and mirror refreshes, and 240 of them reached back over earlier capture spots.)
   */
  private resolveGpu(kind: number) {
    const backend = this.renderer.backend as unknown as {
      trackTimestamp: boolean;
      timestampQueryPool?: Record<string, { frames: number[]; timestamps: Map<string, number> } | null>;
    };
    if (!backend.trackTimestamp) return;
    this.gpuFrameKind.set(this.renderer.info.frame, kind);
    if (this.resolving) return;
    this.resolving = true;
    this.renderer.resolveTimestampsAsync('render').then(() => {
      const pool = backend.timestampQueryPool?.render;
      if (pool) this.takeGpuFrames(pool.frames, pool.timestamps);
      this.resolving = false;
    }, () => (this.resolving = false));
  }

  private gpuFrameKind = new Map<number, number>();
  private lastGpuFrame = -1;
  /** ?gpupasses=1: GPU ms per pass, by what it draws (see hookPassLabels), since the last reset */
  private passLabels = new Map<string, string>();
  private passMs = new Map<string, { ms: number; n: number }>();
  private passFrames = 0;

  private takeGpuFrames(frames: number[], timestamps: Map<string, number>) {
    const fresh = new Set(frames.filter((f) => f > this.lastGpuFrame));
    if (!fresh.size) return;
    const total = new Map<number, number>();
    for (const [uid, ms] of timestamps) {
      const cut = uid.lastIndexOf(':f'), f = Number(uid.slice(cut + 2));
      if (!fresh.has(f)) continue;
      total.set(f, (total.get(f) ?? 0) + ms);
      if (this.passLabels.size) {
        const label = this.passLabels.get(uid.slice(0, cut)) ?? 'other';
        const e = this.passMs.get(label) ?? { ms: 0, n: 0 };
        e.ms += ms;
        e.n++;
        this.passMs.set(label, e);
      }
    }
    for (const f of frames) {
      if (!fresh.has(f) || !total.has(f)) continue;
      this.gpuTimes.push(total.get(f)!);
      this.onGpuFrame?.(total.get(f)!);
      this.gpuKinds.push(this.gpuFrameKind.get(f) ?? 0);
      this.lastGpuFrame = Math.max(this.lastGpuFrame, f);
      this.passFrames++;
    }
    while (this.gpuTimes.length > 240) {
      this.gpuTimes.shift();
      this.gpuKinds.shift();
    }
    for (const f of this.gpuFrameKind.keys()) if (f <= this.lastGpuFrame) this.gpuFrameKind.delete(f);
  }

  /** a capture spot changed: GPU stats (and the pass split) start over, without the frame under way */
  resetGpuStats() {
    this.gpuTimes.length = 0;
    this.gpuKinds.length = 0;
    this.passMs.clear();
    this.passFrames = 0;
    this.lastGpuFrame = Math.max(this.lastGpuFrame, this.renderer.info.frame);
  }

  /** ?gpupasses=1: name each render's timestamp query by its scene (or post quad), camera and target */
  private hookPassLabels() {
    type Begin = (uid: string, scene: THREE.Object3D, camera: THREE.Camera, rt: THREE.RenderTarget | null) => void;
    const insp = this.renderer.inspector as unknown as { beginRender: Begin };
    const base = insp.beginRender.bind(insp);
    insp.beginRender = (uid, scene, camera, rt) => {
      base(uid, scene, camera, rt);
      const key = uid.slice(0, uid.lastIndexOf(':f'));
      if (this.passLabels.has(key)) return;
      const q = scene as unknown as { isQuadMesh?: boolean; material?: THREE.Material };
      const what = q.isQuadMesh ? `quad ${q.material?.name || q.material?.type}` : scene.name || scene.type;
      const cam = camera === this.camera ? 'main' : camera === this.preCam ? 'pre' : camera.name || camera.type;
      this.passLabels.set(key, `${what} | ${cam} | ${rt ? `${rt.width}x${rt.height}` : 'canvas'}`);
    };
  }

  /** ?gpupasses=1: GPU ms per frame and renders per frame of each pass, since the last reset */
  gpuPasses() {
    const n = Math.max(1, this.passFrames);
    return {
      frames: this.passFrames,
      passes: Object.fromEntries([...this.passMs.entries()].sort((a, b) => b[1].ms - a[1].ms)
        .map(([k, e]) => [k, { ms: Math.round((e.ms / n) * 1000) / 1000, perFrame: Math.round((e.n / n) * 100) / 100 }])),
    };
  }

  resize() {
    const w = window.innerWidth, h = window.innerHeight;
    // a minimised or collapsing window: a 0-sized canvas would make every pass's texture invalid
    if (w < 2 || h < 2) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(fitPixelRatio(w, h, this.basePixelRatio() * this.renderScale, this.maxDim));
    this.renderer.setSize(w, h);
  }

  /** wall-clock and GPU frame times (ms) over the last frames */
  frameStats() {
    const stat = (f: number[]) => {
      if (!f.length) return { avg: 0, p95: 0, med: 0 };
      const s = [...f].sort((a, b) => a - b);
      return { avg: f.reduce((a, b) => a + b, 0) / f.length, p95: s[Math.floor(s.length * 0.95)], med: s[Math.floor(s.length / 2)] };
    };
    const w = stat(this.frameTimes), g = stat(this.gpuTimes);
    // per frame kind (0 plain, 1 mirror, 2 shadow, 3 both): the wall interval that follows the frame, its main-thread
    // time and the GPU samples taken of it
    const byKind: Record<string, { n: number; wallMed: number; wallP95: number; cpuMed: number; cpuP95: number; gpuN: number; gpuAvg: number; gpuP95: number }> = {};
    for (const k of [0, 1, 2, 3]) {
      const wk = this.frameTimes.filter((_, i) => this.kinds[i] === k);
      if (!wk.length) continue;
      const ck = this.cpuTimes.filter((_, i) => this.kinds[i] === k), gk = this.gpuTimes.filter((_, i) => this.gpuKinds[i] === k);
      const sw = stat(wk), sc = stat(ck), sg = stat(gk);
      byKind[k] = { n: wk.length, wallMed: sw.med, wallP95: sw.p95, cpuMed: sc.med, cpuP95: sc.p95, gpuN: gk.length, gpuAvg: sg.avg, gpuP95: sg.p95 };
    }
    return { avg: w.avg, p95: w.p95, med: w.med, gpuAvg: g.avg, gpuP95: g.p95, byKind };
  }

  /** the last 240 frames as recorded: wall interval after each frame (ms), its kind and its main-thread time; GPU ms per frame and its kind */
  frameLog() {
    const r = (a: number[]) => a.map((v) => Math.round(v * 10) / 10);
    return { wall: r(this.frameTimes), kind: [...this.kinds], cpu: r(this.cpuTimes), gpu: r(this.gpuTimes), gpuKind: [...this.gpuKinds] };
  }

  /** while true the loop neither updates nor renders (the warm-up waits for its pipelines); an updater may set it */
  hold = false;

  start() {
    if (params.gpuPasses) this.hookPassLabels();
    this.postChannel.port1.onmessage = () => this.runPost();
    let last = performance.now();
    this.renderer.setAnimationLoop(() => {
      const now = performance.now();
      if (this.hold) {
        last = now;
        return;
      }
      this.frameTimes.push(now - last);
      this.onFrame?.(now - last, this.lastCpu);
      this.kinds.push(this.lastKind);
      this.cpuTimes.push(this.lastCpu);
      if (this.frameTimes.length > 240) {
        this.frameTimes.shift();
        this.kinds.shift();
        this.cpuTimes.shift();
      }
      last = now;
      this.frameKind = 0;
      this.timer.update();
      const dt = Math.min(this.timer.getDelta(), 1 / 20);
      this.time = this.fixedTime ?? this.time + dt;
      for (const u of this.updaters) u(this.fixedTime !== null ? 0 : dt, this.time);
      if (this.hold) return;
      this.syncPreCam();
      this.pipeline.render();
      this.frame++;
      this.lastKind = this.frameKind;
      this.lastCpu = performance.now() - now;
      if (this.postFns.length) this.postChannel.port2.postMessage(0);
      else this.resolveGpu(this.lastKind);
    });
  }
}
