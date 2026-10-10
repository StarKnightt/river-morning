import type * as THREE from 'three/webgpu';

export interface PipeLog { label: string; obj: string; ms: number; vs: number; fs: number }

interface PipelinesLike {
  updateForRender: (ro: unknown) => void;
  getForRender: (ro: unknown, promises: Promise<void>[] | null) => unknown;
}
interface NodesLike { getForRender: (ro: { material?: { name?: string; type?: string } }) => unknown }

/**
 * Parallel pipeline warm-up behind the loading screen. three creates render pipelines synchronously in its normal
 * render path, and the GPU process compiles synchronous pipelines one after another (frame 4 of round 1 stalled
 * ~15 s); `compileAsync` awaits them one at a time and builds them for the canvas target, not the pass targets.
 * Between `begin()` and `end()` the normal path takes three's own async branch (createRenderPipelineAsync), so the
 * warm-up frames queue every variant the real passes use (scene-pass target, pre-pass override, shadow pass, mirror,
 * post quads) at once and Dawn compiles them in parallel; an object whose pipeline is not ready is skipped
 * (`Pipelines.isReady`), so nothing is drawn half-built. One-off renders made meanwhile (the environment's PMREM)
 * drew nothing and must be redone afterwards.
 *
 * `log`: every async pipeline's latency and WGSL sizes, plus the CPU time spent building node graphs per material;
 * 'serial' compiles the pipelines one at a time, so each latency is that pipeline's own compile time (diagnosis
 * only: the load gets as slow as round 1's).
 */
export class PipelineWarmup {
  private pending: Promise<void>[] = [];
  private pipelines: PipelinesLike;
  private orig: PipelinesLike['updateForRender'] | null = null;
  private unpatch: Array<() => void> = [];
  readonly log: PipeLog[] = [];
  /** CPU ms building node graphs, by material name */
  readonly build: Record<string, number> = {};
  /** the largest fragment shader's WGSL (log modes only) */
  biggest = '';
  private curObj = '';

  constructor(renderer: THREE.WebGPURenderer, private mode: 'off' | 'log' | 'serial' = 'off') {
    const r = renderer as unknown as {
      _pipelines: PipelinesLike; _nodes: NodesLike;
      backend: { device?: GPUDevice; createRenderPipeline: (ro: { object?: { name?: string } }, p: unknown) => void };
    };
    this.pipelines = r._pipelines;
    if (mode === 'off') return;
    if (r.backend.device) this.unpatch.push(this.instrument(r.backend.device));
    const be = r.backend, crp = be.createRenderPipeline;
    be.createRenderPipeline = (ro, p) => {
      this.curObj = ro.object?.name ?? '';
      crp.call(be, ro, p);
    };
    this.unpatch.push(() => (be.createRenderPipeline = crp));
    const nodes = r._nodes, get = nodes.getForRender;
    nodes.getForRender = (ro) => {
      const t0 = performance.now();
      const s = get.call(nodes, ro);
      const k = ro.material?.name || ro.material?.type || '?';
      this.build[k] = (this.build[k] ?? 0) + performance.now() - t0;
      return s;
    };
    this.unpatch.push(() => (nodes.getForRender = get));
  }

  get count() {
    return this.pending.length;
  }

  begin() {
    const p = this.pipelines;
    this.orig = p.updateForRender;
    p.updateForRender = (ro) => {
      p.getForRender(ro, this.pending);
    };
  }

  /** back to synchronous creation; resolves when every queued pipeline is compiled */
  async end(onProgress?: (done: number, total: number) => void): Promise<void> {
    if (this.orig) this.pipelines.updateForRender = this.orig;
    this.orig = null;
    let done = 0;
    const total = this.pending.length;
    await Promise.all(this.pending.map((q) => q.then(() => onProgress?.(++done, total))));
    for (const u of this.unpatch) u();
    this.log.sort((a, b) => b.ms - a.ms);
    for (const k of Object.keys(this.build)) this.build[k] = Math.round(this.build[k]);
  }

  private instrument(device: GPUDevice) {
    const sizes = new WeakMap<GPUShaderModule, number>();
    const createModule = device.createShaderModule;
    const createAsync = device.createRenderPipelineAsync;
    let chain: Promise<unknown> = Promise.resolve();
    const self = this;
    device.createShaderModule = function (d: GPUShaderModuleDescriptor) {
      const m = createModule.call(this, d);
      sizes.set(m, d.code.length);
      if (d.code.length > self.biggest.length) self.biggest = d.code;
      return m;
    };
    device.createRenderPipelineAsync = (d: GPURenderPipelineDescriptor) => {
      // three resets its shared descriptor right after this call; only `multisample` is mutated in place
      const desc = { ...d, multisample: { ...d.multisample } } as GPURenderPipelineDescriptor;
      const entry: PipeLog = {
        label: d.label ?? '', obj: this.curObj, ms: 0, vs: sizes.get(d.vertex.module) ?? 0, fs: d.fragment ? sizes.get(d.fragment.module) ?? 0 : 0,
      };
      const run = () => {
        const t0 = performance.now();
        return createAsync.call(device, desc).then((pl) => {
          entry.ms = Math.round(performance.now() - t0);
          this.log.push(entry);
          return pl;
        });
      };
      if (this.mode !== 'serial') return run();
      const p = chain.then(run);
      chain = p.catch(() => undefined);
      return p;
    };
    return () => {
      device.createShaderModule = createModule;
      device.createRenderPipelineAsync = createAsync;
    };
  }
}
