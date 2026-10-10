/**
 * Headless check of the resolution governor (no browser, no GPU): `pnpm exec tsx scripts/govsim.ts`.
 * A fake engine whose GPU frame cost is fixed + per-pixel (with noise), fed at the frame rate that cost allows on a
 * 60 Hz screen. Exits non-zero if a scenario changes the scale when it must not, oscillates, or misses its target.
 */
export {};
const g = globalThis as unknown as Record<string, unknown>;
const listeners: Record<string, Array<() => void>> = {};
g.location = { search: '' };
g.window = Object.assign(g, {
  __RM: { ready: true }, innerWidth: 1920, innerHeight: 1080, devicePixelRatio: 2,
  addEventListener: (k: string, f: () => void) => (listeners[k] ??= []).push(f),
});
g.document = { addEventListener: (k: string, f: () => void) => (listeners[k] ??= []).push(f) };
const { ResolutionGovernor } = await import('../src/core/ResolutionGovernor');
const { fitPixelRatio } = await import('../src/core/deviceGuard');

type Scenario = { name: string; css: [number, number]; dpr: number; secs: number;
  gpu: (px: number, t: number) => number; cpu?: number; expect: (hist: number[], final: number) => string | null };

let seed = 7;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

function run(s: Scenario) {
  const w = s.css[0], h = s.css[1];
  Object.assign(g, { innerWidth: w, innerHeight: h, devicePixelRatio: s.dpr });
  let scale = 1;
  const engine = {
    setRenderScale: (v: number) => (scale = v),
    nativePixels: () => {
      const pr = fitPixelRatio(w, h, Math.min(s.dpr, 2));
      return Math.floor(w * pr) * Math.floor(h * pr);
    },
    bufferSize: () => [0, 0] as [number, number],
  };
  const gov = new ResolutionGovernor(engine as never);
  const native = engine.nativePixels();
  const scales: number[] = [];
  for (let t = 0; t < s.secs;) {
    const px = native * scale * scale;
    const cost = s.gpu(px, t) * (0.9 + 0.2 * rnd());
    const cpu = s.cpu ?? 4;
    const frame = Math.max(cost, cpu);
    const iv = Math.ceil(frame / 16.667 - 1e-6) * 16.667;
    gov.gpuFrame(cost);
    gov.frame(iv, cpu);
    t += iv / 1000;
    if (!scales.length || scales[scales.length - 1] !== scale) scales.push(scale);
  }
  const pixFrac = scales.map((v) => Math.round(v * v * 100) / 100);
  const bad = s.expect(pixFrac, pixFrac[pixFrac.length - 1]);
  console.log(`${bad ? 'FAIL' : 'ok  '} ${s.name}: pixel fractions ${pixFrac.join(' -> ')} | changes ${gov.history.length}${bad ? ' | ' + bad : ''}`);
  return !bad;
}

const MP = 1e6;
const results = [
  // the user's 4060 at 1080p: even a heavy 20 ms frame never lowers a 1080p screen
  run({ name: '1080p, 20 ms GPU (floor holds native)', css: [1920, 1080], dpr: 1, secs: 60, gpu: () => 20,
    expect: (_, f) => (f === 1 ? null : 'scaled a 1080p screen') }),
  // 4060-like at 4K: ~31 ms native, 23 ms at 0.6: goes to the floor and stays
  run({ name: '4K, 4060-like (11 + 20/MP*8.3)', css: [1920, 1080], dpr: 2, secs: 40, gpu: (px) => 11.3 + 20.1 * (px / (8.29 * MP)),
    expect: (h, f) => (f === 0.6 && h.length === 5 ? null : 'expected 1 -> 0.6 in four steps') }),
  // 4080-like at 4K (~2x): 16 ms native: one step down, then stable (no oscillation at the edge)
  run({ name: '4K, 4080-like (half the 4060)', css: [1920, 1080], dpr: 2, secs: 120, gpu: (px) => (11.3 + 20.1 * (px / (8.29 * MP))) / 2,
    expect: (h, f) => (f >= 0.8 && h.length <= 3 ? null : 'oscillated or dropped too far') }),
  // 4090-like at 4K (~3x): ~10.5 ms: never changes
  run({ name: '4K, 4090-like (a third)', css: [1920, 1080], dpr: 2, secs: 60, gpu: (px) => (11.3 + 20.1 * (px / (8.29 * MP))) / 3,
    expect: (h) => (h.length === 1 ? null : 'changed scale with headroom') }),
  // a heavy spot for 6 s (smoke over the camp at 4K on a 4080), then back: down, then up again after the headroom
  run({ name: '4K 4080, 6 s heavy spike', css: [1920, 1080], dpr: 2, secs: 60,
    gpu: (px, t) => ((11.3 + 20.1 * (px / (8.29 * MP))) / 2.6) * (t > 10 && t < 16 ? 1.6 : 1),
    expect: (h, f) => (f === 1 && h.length >= 3 && h.length <= 7 ? null : 'did not recover to native') }),
  // CPU-bound (4090 at 1440p, 18 ms main thread, 7 ms GPU): never lowers the resolution
  run({ name: 'CPU-bound 1440p (cpu 18 ms, gpu 7 ms)', css: [2560, 1440], dpr: 1, secs: 40, gpu: () => 7, cpu: 18,
    expect: (h) => (h.length === 1 ? null : 'lowered the scale on a CPU-bound frame') }),
  // cost right at the threshold with noise: bounded changes (the step-up block)
  run({ name: '4K at the threshold, noisy', css: [1920, 1080], dpr: 2, secs: 300, gpu: (px) => 6 + 9.8 * (px / (8.29 * MP)),
    expect: (h) => (h.length <= 4 ? null : `${h.length - 1} changes in 5 min`) }),
];
const failed = results.filter((r) => !r).length;
console.log(failed ? `${failed} scenario(s) failed` : 'all governor scenarios passed');
process.exit(failed ? 1 : 0);
