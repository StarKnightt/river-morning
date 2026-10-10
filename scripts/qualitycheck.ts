/**
 * Graphics quality check (src/core/quality.ts): play mode, muted, second monitor, pointer lock stubbed.
 *
 *   pnpm tsx scripts/qualitycheck.ts --out <dir> [--q auto|low|medium|high|ultra] [--gpu high|low] [--dsf 2]
 *     [--uncapped] [--throttle 4] [--backend webgl] [--lose] [--spots C1,C2] [--perf 4] [--settle 7]
 *
 * --gpu low launches Chrome with --force_low_power_gpu (the integrated GPU) instead of --force_high_performance_gpu.
 * --dsf 2 renders the 1920x1080 viewport at device scale factor 2 (a 3840x2160 drawing buffer, as on a 4K screen).
 * --uncapped adds --disable-gpu-vsync --disable-frame-rate-limit; --throttle N slows the CPU N x (CDP).
 * --lose destroys the WebGPU device (?debug hook) once ready and checks the reload one preset lower.
 * Writes report.json and a JPEG per spot into --out (relative to E:\shots\river-morning\prep\code\quality).
 */
import { chromium } from 'playwright-core';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const args = process.argv.slice(2);
const arg = (name: string, def: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const has = (name: string) => args.includes(`--${name}`);
const out = join('E:\\shots\\river-morning\\prep\\code\\quality', arg('out', 'run'));
const url = arg('url', 'http://127.0.0.1:5191/');
const q = arg('q', 'auto');
const spots = arg('spots', 'C1,C2,C3,C4,C5,C6').split(',').filter(Boolean);
const perf = Number(arg('perf', '4'));
const settle = Number(arg('settle', '7'));
const dsf = Number(arg('dsf', '1'));

async function main() {
  await mkdir(out, { recursive: true });
  const flags = [
    '--window-position=1930,30', '--window-size=1060,900',
    arg('gpu', 'high') === 'low' ? '--force_low_power_gpu' : '--force_high_performance_gpu',
    '--mute-audio', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding',
    '--disable-background-timer-throttling', '--disable-features=CalculateNativeWinOcclusion',
    ...(has('uncapped') ? ['--disable-gpu-vsync', '--disable-frame-rate-limit'] : []),
  ];
  // --channel msedge: Windows pins chrome.exe to the high-performance GPU on this PC (Graphics settings), which overrides
  // --force_low_power_gpu; Edge (same Chromium WebGPU) has no such entry and honours it
  const browser = await chromium.launch({ channel: arg('channel', 'chrome'), headless: false, args: flags });
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: dsf });
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
  if (has('throttle')) {
    const cdp = await context.newCDPSession(page);
    await cdp.send('Emulation.setCPUThrottlingRate', { rate: Number(arg('throttle', '4')) });
  }
  // --shot: photo mode (the camera spots C1..C6 move the view; time frozen), with the quality from --q
  const extra = [has('shot') ? 'shot=C1' : '', `quality=${q}`, 'vol=0', has('lose') ? 'debug' : '', arg('backend', '') ? `backend=${arg('backend', '')}` : '', arg('extra', '').replaceAll('~', '&')]
    .filter(Boolean).join('&');
  const t0 = Date.now();
  await page.goto(`${url}?${extra}`, { waitUntil: 'domcontentloaded' });
  const waitReady = async (label: string) => {
    await page.waitForFunction(() => window.__RM?.ready || window.__RM?.error || document.querySelector('.loading-fatal'), null, { timeout: 240_000, polling: 250 });
    const m = await page.evaluate(() => ({
      ready: window.__RM.ready, error: window.__RM.error, loadMs: window.__RM.loadMs, backend: window.__RM.backend, adapter: window.__RM.adapter,
      pipelines: window.__RM.timings?.pipelines, compile: window.__RM.timings?.compile,
      fatal: document.querySelector('.loading-fatal')?.textContent ?? null,
      gpuLine: document.querySelector('.loading-gpu')?.textContent ?? null, hint: document.querySelector('.loading-hint')?.textContent ?? null,
      quality: (window.__RM as unknown as { quality?: () => unknown }).quality?.() ?? null,
    }));
    console.log(`${label}: wall ${Date.now() - t0} ms`, JSON.stringify(m));
    return m;
  };
  const meta = await waitReady('ready');
  const report: Record<string, unknown> = { flags, extra, dsf, meta, at: new Date().toISOString() };
  if (meta.fatal || meta.error) {
    await page.screenshot({ path: join(out, 'state.jpg'), type: 'jpeg', quality: 80 });
  }
  if (has('lose') && meta.ready) {
    await page.waitForTimeout(1500);
    const before = await page.evaluate(() => (window.__RM as unknown as { quality: () => { preset: string } }).quality().preset);
    await page.evaluate(() => (window.__RM as unknown as { loseDevice: () => void }).loseDevice());
    await page.waitForTimeout(800);
    await page.screenshot({ path: join(out, 'lost.jpg'), type: 'jpeg', quality: 80 });
    const msg = await page.evaluate(() => document.querySelector('.device-lost')?.textContent ?? null);
    console.log('lost message:', msg);
    await page.waitForEvent('load', { timeout: 20_000 }).catch(() => console.log('no reload seen'));
    await page.waitForTimeout(500);
    const after = await waitReady('after reload');
    report.lost = { before, msg, after };
  }
  if (!meta.fatal && !meta.error) {
    // calibration runs 4.5 s after ready; the dynamic scale needs a few more seconds; Auto may reload once (going down)
    let loads = 0;
    page.on('load', () => loads++);
    await page.waitForTimeout(settle * 1000);
    if (loads) {
      const again = await waitReady(`reloaded by Auto (${loads})`);
      report.reload = again;
      await page.waitForTimeout(settle * 1000);
    }
    const results: Record<string, unknown>[] = [];
    for (const id of spots) {
      await page.evaluate((s) => window.__RM.spot(s), id);
      await page.waitForTimeout(900);
      await page.screenshot({ path: join(out, `${id}.jpg`), type: 'jpeg', quality: 80 });
      await page.waitForTimeout(perf * 1000);
      const s = await page.evaluate(() => ({
        frame: window.__RM.frameStats(), quality: (window.__RM as unknown as { quality: () => unknown }).quality(),
      }));
      const f = s.frame;
      const qs = s.quality as { preset: string; scale: number; buffer: string; autoPick: string | null };
      console.log(`${id}: wall med ${f.med.toFixed(1)} p95 ${f.p95.toFixed(1)} (${(1000 / f.med).toFixed(0)} fps) | gpu avg ${f.gpuAvg.toFixed(2)} p95 ${f.gpuP95.toFixed(2)} | ${qs.preset} scale ${qs.scale} ${qs.buffer} pick ${qs.autoPick}`);
      results.push({ id, ...s });
    }
    report.results = results;
    report.final = await page.evaluate(() => (window.__RM as unknown as { quality: () => unknown }).quality());
  }
  report.consoleLines = consoleLines;
  await writeFile(join(out, 'report.json'), JSON.stringify(report, null, 2));
  if (consoleLines.length) console.log('console:\n' + consoleLines.slice(0, 12).join('\n'));
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
