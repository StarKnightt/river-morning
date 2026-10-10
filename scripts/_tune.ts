// Scratch tuning harness (not part of the game): one page load, then a list of steps, each optionally switching the
// time of day and the camera spot and evaluating a snippet in the page, then a screenshot.
//   pnpm exec tsx scripts/_tune.ts <steps.json> <outDir> [extra url params with ~]
// steps.json: [{ "name": "a", "tod": "golden", "spot": "C6", "js": "__TOD.presets.golden.light.minElevation = 12" }]
import { chromium } from 'playwright-core';
import { createServer } from 'vite';
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

interface Step { name: string; tod?: string; spot?: string; js?: string; wait?: number }

const [stepsFile, outDir, extraRaw = ''] = process.argv.slice(2);
const extra = extraRaw.replaceAll('~', '&');
const url = 'http://127.0.0.1:5173/';

async function main() {
  const steps = JSON.parse(await readFile(stepsFile, 'utf8')) as Step[];
  await mkdir(outDir, { recursive: true });
  let server = null;
  try {
    await fetch(url);
  } catch {
    server = await createServer({ root: join(import.meta.dirname, '..'), logLevel: 'warn', server: { host: '127.0.0.1', port: 5173, strictPort: true } });
    await server.listen();
  }
  const browser = await chromium.launch({
    channel: 'chrome', headless: false,
    args: ['--window-position=1930,30', '--window-size=1060,900', '--force_high_performance_gpu', '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding', '--disable-background-timer-throttling', '--disable-features=CalculateNativeWinOcclusion', '--mute-audio'],
  });
  try {
    const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
    await context.addInitScript(() => {
      Element.prototype.requestPointerLock = function () { return Promise.resolve(); } as typeof Element.prototype.requestPointerLock;
    });
    const page = await context.newPage();
    page.on('pageerror', (e) => console.log('[pageerror]', e.message));
    page.on('console', (m) => { if (m.type() === 'error') console.log('[console]', m.text()); });
    const first = steps[0];
    await page.goto(`${url}?shot=${first.spot ?? 'C1'}&tod=${first.tod ?? 'morning'}${extra ? '&' + extra : ''}`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => (window as any).__RM?.ready || (window as any).__RM?.error, null, { timeout: 120_000, polling: 250 });
    await page.waitForFunction(() => { const el = document.getElementById('loading'); return !el || Number(getComputedStyle(el).opacity) < 0.01; }, null, { timeout: 20_000 }).catch(() => {});

    for (const s of steps) {
      const r = await page.evaluate(async (st) => {
        const w = window as any;
        let out: unknown = null;
        if (st.js) out = await (0, eval)(`(async () => { ${st.js} })()`);
        if (st.tod) await w.__TOD.set(st.tod, true);
        if (st.spot) w.__RM.spot(st.spot);
        return out;
      }, s);

      await page.waitForTimeout(s.wait ?? 900);
      await page.screenshot({ path: join(outDir, `${s.name}.png`) });
      console.log(s.name, r === null || r === undefined ? '' : JSON.stringify(r));
    }
  } finally {
    await browser.close();
    await server?.close();
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
