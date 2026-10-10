/**
 * Big-screen and robustness check: frame rate, GPU time, the resolution governor, render-target sizes, GPU errors,
 * fast resizes and a background tab, at any viewport / scale factor (see chromeFlags.ts).
 *
 *   RM_VIEWPORT=1920x1080 RM_DSF=2 pnpm tsx scripts/robustcheck.ts --out robust/2160p [--url http://127.0.0.1:5190/]
 *     [--seconds 15] [--extra gov=0~scale=1] [--resize] [--background 30]
 *
 * Play mode with ?debug (GPU timestamps on). Never renders above 5120x2880 (chromeFlags refuses).
 */
import { chromium } from 'playwright-core';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { conditionLabel, extraChromeArgs, viewportOpts } from './chromeFlags';

const args = process.argv.slice(2);
const arg = (name: string, def: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const url = arg('url', 'http://127.0.0.1:5190/');
const outDir = join('E:\\shots\\river-morning\\prep\\code', arg('out', 'robust/check'));
const seconds = Math.min(60, Number(arg('seconds', '15')));
const extra = arg('extra', '').replaceAll('~', '&');
const bgSeconds = Math.min(60, Number(arg('background', '0')));

async function main() {
  await mkdir(outDir, { recursive: true });
  const opts = viewportOpts();
  const browser = await chromium.launch({
    channel: 'chrome', headless: false,
    args: ['--window-position=1930,30', '--window-size=1060,900', '--force_high_performance_gpu', '--mute-audio',
      '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-background-timer-throttling',
      '--disable-features=CalculateNativeWinOcclusion', ...extraChromeArgs()],
  });
  const report: Record<string, unknown> = { at: new Date().toISOString(), condition: conditionLabel(), url, extra };
  const consoleLines: string[] = [];
  try {
    const context = await browser.newContext(opts);
    await context.addInitScript({ content: 'window.__name = (f) => f;' });
    await context.addInitScript(() => {
      Element.prototype.requestPointerLock = function () {
        return Promise.resolve();
      } as typeof Element.prototype.requestPointerLock;
    });
    const page = await context.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error' || m.type() === 'warning') consoleLines.push(`[${m.type()}] ${m.text()}`);
    });
    page.on('pageerror', (e) => consoleLines.push(`[pageerror] ${e.message}`));
    const t0 = Date.now();
    await page.goto(`${url}?debug&vol=0&gpupasses=1${extra ? '&' + extra : ''}`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__RM?.ready || window.__RM?.error, null, { timeout: 120_000, polling: 250 });
    report.loadWallMs = Date.now() - t0;
    report.error = await page.evaluate(() => window.__RM.error);
    if (report.error) throw new Error(String(report.error));
    report.gpu = await page.evaluate(() => (window as unknown as { __RM_GPU?: unknown }).__RM_GPU ?? null);
    const sample = () => page.evaluate(() => {
      const s = window.__RM.frameStats();
      const c = document.querySelector('canvas')!;
      return {
        fpsMed: Math.round(1000 / Math.max(0.1, s.med)), wallMed: Math.round(s.med * 100) / 100, wallP95: Math.round(s.p95 * 100) / 100,
        gpuAvg: Math.round(s.gpuAvg * 100) / 100, gpuP95: Math.round(s.gpuP95 * 100) / 100,
        canvas: `${c.width}x${c.height}`, css: `${innerWidth}x${innerHeight}`, dpr: devicePixelRatio,
        gov: (window.__RM as unknown as { governor: () => { scale: number; pixels?: number; cost?: number; source?: string } }).governor(),
      };
    });
    const timeline: unknown[] = [];
    await page.waitForTimeout(2500);
    for (let i = 0; i < seconds; i++) {
      await page.waitForTimeout(1000);
      const s = await sample();
      timeline.push(s);
      console.log(`${i + 1}s ${s.fpsMed} fps wall ${s.wallMed}/${s.wallP95} gpu ${s.gpuAvg}/${s.gpuP95} canvas ${s.canvas} scale ${s.gov.scale.toFixed(3)} (${s.gov.pixels ?? '-'}) cost ${s.gov.cost ?? '-'} ${s.gov.source ?? ''}`);
    }
    report.timeline = timeline;
    report.passes = await page.evaluate(() => (window.__RM as unknown as { gpuPasses: () => unknown }).gpuPasses());
    await page.screenshot({ path: join(outDir, 'view.png') });

    if (args.includes('--resize')) {
      // fast resizes (window drags, fullscreen toggles, snapping): every size must end in a matching canvas, no errors
      const sizes = [[1280, 720], [2560, 1080], [800, 1200], [1920, 1080], [3440, 1440], [640, 360], [1600, 900], [opts.viewport.width, opts.viewport.height]];
      for (let k = 0; k < 2; k++) for (const [w, h] of sizes) {
        await page.setViewportSize({ width: w, height: h });
        await page.waitForTimeout(60);
      }
      await page.waitForTimeout(1500);
      report.afterResize = await sample();
      console.log('after resizes', JSON.stringify(report.afterResize));
    }
    if (bgSeconds > 0) {
      // another tab in front: this one is hidden (no frames), then comes back
      const before = await page.evaluate(() => ({ t: performance.now(), frame: window.__RM.info().passes }));
      const other = await context.newPage();
      await other.goto('about:blank');
      await other.bringToFront();
      await page.waitForTimeout(500);
      const hidden = await page.evaluate(() => document.visibilityState);
      await other.waitForTimeout(bgSeconds * 1000);
      await page.bringToFront();
      await other.close();
      await page.waitForTimeout(2500);
      report.background = { hidden, seconds: bgSeconds, before, after: await sample(), error: await page.evaluate(() => window.__RM.error) };
      console.log('after background', JSON.stringify(report.background));
    }
    if (args.includes('--gpuclean')) {
      // a real device loss: chrome://gpuclean ends this test browser's own GPU process (not the user's Chrome)
      const other = await context.newPage();
      await other.goto('chrome://gpuclean').catch((e) => console.log('gpuclean:', String(e).split('\n')[0]));
      await other.close().catch(() => {});
      await page.waitForTimeout(4000);
      report.deviceLost = await page.evaluate(() => ({
        overlay: [...document.querySelectorAll('[role=alertdialog] h2')].map((h) => h.textContent),
        lost: (window as unknown as { __RM_GPU?: { errors: { lost: string | null } } }).__RM_GPU?.errors.lost ?? null,
      }));
      await page.screenshot({ path: join(outDir, 'device-lost.png') });
      console.log('device lost', JSON.stringify(report.deviceLost));
    }
    report.gpuErrors = await page.evaluate(() => ((window as unknown as { __RM_GPU?: { errors: unknown } }).__RM_GPU?.errors ?? null));
    report.governor = await page.evaluate(() => (window.__RM as unknown as { governor: () => unknown }).governor());
  } catch (e) {
    report.failure = String((e as Error)?.stack ?? e);
    console.error(e);
  } finally {
    report.consoleLines = consoleLines;
    await writeFile(join(outDir, 'report.json'), JSON.stringify(report, null, 2));
    console.log(`console lines: ${consoleLines.length}`, consoleLines.slice(0, 12).join('\n'));
    console.log('gpu errors', JSON.stringify(report.gpuErrors));
    await browser.close();
  }
}

main();
