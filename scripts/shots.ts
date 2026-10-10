/**
 * Fixed-spot screenshot runner for the critic gauntlet.
 *
 *   pnpm shots --part p1 --round r1 [--spots C1,C3] [--tod morning] [--url http://127.0.0.1:5173/]
 *
 * Opens the installed Chrome on the second (portrait) monitor, renders a true 1920x1080
 * viewport, captures every camera spot and writes a JSON report next to the PNGs.
 * Screenshots never go inside the project: they land in E:\shots\river-morning.
 */
import { chromium } from 'playwright-core';
import { createServer, type ViteDevServer } from 'vite';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const args = process.argv.slice(2);
const arg = (name: string, def: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const part = arg('part', 'p1');
const round = arg('round', 'r1');
const tod = arg('tod', 'morning');
const url = arg('url', 'http://127.0.0.1:5173/');
// extra URL parameters; '~' also separates them (cmd.exe splits a command line at '&')
const extra = arg('extra', '').replaceAll('~', '&');
const perfSeconds = Number(arg('perf', '4'));
const outDir = join('E:\\shots\\river-morning', part, round);

// primary monitor (the second one is off since 10 Oct)
const WINDOW_POS = '1930,30';
const WINDOW_SIZE = '1060,900';

/** Dev servers only run while needed: reuse one that is already up, otherwise start one just for this run. */
async function ensureServer(): Promise<ViteDevServer | null> {
  try {
    const r = await fetch(url);
    if (r.ok) return null;
  } catch {
    // not running
  }
  const server = await createServer({
    root: join(import.meta.dirname, '..'),
    logLevel: 'warn',
    server: { host: '127.0.0.1', port: 5173, strictPort: true },
  });
  await server.listen();
  return server;
}

let ownServer: ViteDevServer | null = null;

async function main() {
  await mkdir(outDir, { recursive: true });
  ownServer = await ensureServer();
  const browser = await chromium.launch({
    channel: 'chrome',
    headless: false,
    args: [
      `--window-position=${WINDOW_POS}`,
      `--window-size=${WINDOW_SIZE}`,
      '--force_high_performance_gpu',
      '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding',
      '--disable-background-timer-throttling',
      '--disable-features=CalculateNativeWinOcclusion',
      '--mute-audio',
    ],
  });
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  // A granted pointer lock in this headed window captures the real mouse of whoever is at the PC.
  await context.addInitScript(() => {
    Element.prototype.requestPointerLock = function () {
      return Promise.resolve();
    } as typeof Element.prototype.requestPointerLock;
  });
  const page = await context.newPage();
  const consoleLines: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error' || m.type() === 'warning') consoleLines.push(`[${m.type()}] ${m.text()}`);
  });
  page.on('pageerror', (e) => consoleLines.push(`[pageerror] ${e.message}`));

  const t0 = Date.now();
  const spotsArg = arg('spots', '');
  await page.goto(`${url}?shot=C1&tod=${tod}${extra ? '&' + extra : ''}`, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__RM?.ready || window.__RM?.error, null, { timeout: 120_000, polling: 250 });
  const wallLoadMs = Date.now() - t0;
  const meta = await page.evaluate(() => ({
    error: window.__RM.error, loadMs: window.__RM.loadMs, timings: window.__RM.timings, backend: window.__RM.backend,
    adapter: window.__RM.adapter, spots: window.__RM.spots, pipelog: window.__RM.pipelog, nodeBuild: window.__RM.nodeBuild,
  }));
  if (meta.error) {
    console.error('page error:', meta.error);
    await page.screenshot({ path: join(outDir, 'error.png') });
    await writeFile(join(outDir, 'report.json'), JSON.stringify({ meta, consoleLines }, null, 2));
    await browser.close();
    await ownServer?.close();
    process.exit(1);
  }
  // the loading overlay fades for 0.8 s once ready; a main thread busy compiling can hold it over the first shot
  await page.waitForFunction(() => {
    const el = document.getElementById('loading');
    return !el || Number(getComputedStyle(el).opacity) < 0.01;
  }, null, { timeout: 20_000, polling: 100 }).catch(() => console.log('warning: loading overlay still visible'));
  const spots = spotsArg ? spotsArg.split(spotsArg.includes('cam:') ? ';' : ',') : meta.spots;
  const results: Record<string, unknown>[] = [];
  for (const id of spots) {
    await page.evaluate((s) => window.__RM.spot(s), id);
    // let shadows, AO and any temporal filters settle
    await page.waitForTimeout(900);
    const file = join(outDir, `${id.startsWith("cam:") ? "cam" + spots.indexOf(id) : id}.png`);
    await page.screenshot({ path: file, type: 'png' });
    await page.waitForTimeout(perfSeconds * 1000);
    const stats = await page.evaluate(() => ({ frame: window.__RM.frameStats(), info: window.__RM.info() }));
    // --raw: every frame's wall interval, kind and main-thread time (frame pacing analysis)
    const raw = args.includes('--raw') ? await page.evaluate(() => (window.__RM as unknown as { frameLog?: () => unknown }).frameLog?.()) : undefined;
    // --gpupasses (with --extra gpupasses=1): GPU ms per frame of each pass at this spot
    const gpuPasses = args.includes('--gpupasses')
      ? await page.evaluate(() => (window.__RM as unknown as { gpuPasses?: () => unknown }).gpuPasses?.()) : undefined;
    // --pretris: the triangles the GTAO pre-pass draws from this spot, by object
    const preTris = args.includes('--pretris')
      ? await page.evaluate(() => (window.__RM as unknown as { preTris?: () => unknown }).preTris?.()) : undefined;
    results.push({ id, file, ...stats, raw, gpuPasses, preTris });
    const f = stats.frame;
    console.log(`${id}: wall median ${f.med.toFixed(1)} ms (${(1000 / f.med).toFixed(0)} fps), avg ${f.avg.toFixed(1)}, p95 ${f.p95.toFixed(1)} | gpu avg ${f.gpuAvg.toFixed(2)} ms p95 ${f.gpuP95.toFixed(2)} ms | draws ${stats.info.drawCalls} passes ${stats.info.passes} tris ${stats.info.triangles}`);
    if (f.byKind) {
      const names = ['plain', 'mirror', 'shadow', 'both'];
      console.log('    ' + Object.entries(f.byKind).map(([k, s]) => `${names[Number(k)]} n${s.n} wall ${s.wallMed.toFixed(1)}/${s.wallP95.toFixed(1)}`
        + ` cpu ${s.cpuMed.toFixed(1)}/${s.cpuP95.toFixed(1)} gpu(${s.gpuN}) ${s.gpuAvg.toFixed(2)}/${s.gpuP95.toFixed(2)}`).join(' | '));
    }
  }
  // --passes: what the shadow pass and the pool mirror draw, at the last spot (window.__RM.passes in main.ts)
  const passes = args.includes('--passes')
    ? await page.evaluate(() => (window.__RM as unknown as { passes?: () => unknown }).passes?.() ?? null) : undefined;
  if (passes) console.log('passes:', JSON.stringify(passes));
  const report = { part, round, tod, url, wallLoadMs, ...meta, results, passes, consoleLines, at: new Date().toISOString() };
  await writeFile(join(outDir, 'report.json'), JSON.stringify(report, null, 2));
  console.log(`load ${meta.loadMs.toFixed(0)} ms (wall ${wallLoadMs} ms) backend ${meta.backend} adapter ${meta.adapter}`);
  console.log('timings', JSON.stringify(meta.timings));
  if (meta.pipelog?.length) {
    console.log('slowest pipelines:\n' + meta.pipelog.slice(0, 15).map((p) => `  ${p.ms} ms  vs ${p.vs} fs ${p.fs}  ${p.label}  @${p.obj}`).join('\n'));
    const by: Record<string, { n: number; ms: number; fs: number }> = {};
    for (const p of meta.pipelog) {
      const k = p.label.replace(/^renderPipeline_/, '').replace(/_\d+$/, '');
      const e = (by[k] ??= { n: 0, ms: 0, fs: 0 });
      e.n++;
      e.ms += p.ms;
      e.fs = Math.max(e.fs, p.fs);
    }
    console.log('pipelines by material (count, summed ms, largest fs):\n' + Object.entries(by).sort((a, b) => b[1].ms - a[1].ms)
      .map(([k, e]) => `  ${k}: ${e.n}, ${e.ms} ms, fs ${e.fs}`).join('\n'));
  }
  if (meta.nodeBuild) console.log('node build ms:', JSON.stringify(Object.fromEntries(Object.entries(meta.nodeBuild).sort((a, b) => b[1] - a[1]))));
  const big = await page.evaluate(() => window.__RM.biggestShader ?? '');
  if (big) await writeFile(join(outDir, 'biggest-shader.wgsl'), big);
  if (consoleLines.length) console.log('console:\n' + consoleLines.slice(0, 20).join('\n'));
  await browser.close();
  await ownServer?.close();
}

main().catch(async (e) => {
  console.error(e);
  await ownServer?.close();
  process.exit(1);
});
