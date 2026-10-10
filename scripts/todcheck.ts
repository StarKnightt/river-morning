/**
 * Part 8 play check: the time of day in normal play (no ?debug, no ?shot).
 *
 *   pnpm tsx scripts/todcheck.ts [--out p8\r1-todkeys] [--url http://127.0.0.1:5173/]
 *
 * A clean screen: no time-of-day buttons, the fishing tally (top left) hidden until it changes or Tab is held; T
 * steps morning -> golden hour -> night -> dawn -> morning with the ~2 s fade, a small label with the new time's name
 * that fades after ~1.5 s, the URL's ?tod= and the sound's hour following; three quick presses queue one step, never
 * more; T is ignored with Ctrl, Alt or Meta held, on key repeat and while typing in a text field; photo mode (?shot)
 * has neither the label nor the key. Frame intervals
 * (requestAnimationFrame) before and during every fade. Muted browser, ?vol=0, pointer lock faked. Writes
 * todcheck.json and a mid-fade screenshot under E:\shots\river-morning\<out>; starts and stops its own dev server
 * like scripts/shots.ts.
 */
import { chromium, type Page } from 'playwright-core';
import { extraChromeArgs, viewportOpts } from './chromeFlags';
import { createServer, type ViteDevServer } from 'vite';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { TOD_HOUR } from '../src/audio/timeOfDay';
import type { TimeOfDay } from '../src/core/params';

const args = process.argv.slice(2);
const arg = (name: string, def: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const url = arg('url', 'http://127.0.0.1:5173/');
const outDir = join('E:\\shots\\river-morning', arg('out', 'p8\\r1-todkeys'));

async function ensureServer(): Promise<ViteDevServer | null> {
  try {
    if ((await fetch(url)).ok) return null;
  } catch {
    // not running
  }
  const server = await createServer({ root: join(import.meta.dirname, '..'), logLevel: 'warn', server: { host: '127.0.0.1', port: Number(new URL(url).port) || 5173, strictPort: true } });
  await server.listen();
  return server;
}

const T0 = Date.now();
const checks: { name: string; ok: boolean; s: number; detail?: unknown }[] = [];
const check = (name: string, ok: boolean, detail?: unknown) => {
  const s = Math.round((Date.now() - T0) / 100) / 10;
  checks.push({ name, ok, s, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'} ${s}s ${name}${detail !== undefined ? ' ' + JSON.stringify(detail) : ''}`);
};
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const POINTER_LOCK_STUB = `(() => {
  let locked = null;
  const changed = () => queueMicrotask(() => document.dispatchEvent(new Event('pointerlockchange')));
  Element.prototype.requestPointerLock = function () { locked = this; changed(); return Promise.resolve(); };
  Object.defineProperty(Document.prototype, 'pointerLockElement', { configurable: true, get: () => locked });
  Document.prototype.exitPointerLock = function () { locked = null; changed(); };
  window.__pointerLockStub = true;
})();`;

type TodGet = { current: TimeOfDay; target: TimeOfDay | null; k: number };
const tod = (page: Page) => page.evaluate(() => (window as any).__TOD.get() as TodGet);
/** the time named by the T label while it shows (null once it has faded) */
const label = (page: Page) => page.evaluate(() => (document.querySelector('.tod-label.is-on') as HTMLElement | null)?.dataset.tod ?? null);
const tallyOn = (page: Page) => page.evaluate(() => {
  const el = document.querySelector('.fh-tally');
  return !!el && el.classList.contains('fh-on');
});
const urlTod = (page: Page) => page.evaluate(() => new URL(location.href).searchParams.get('tod'));
const hour = (page: Page) => page.evaluate(() => (window as any).__RMAudio?.state?.().hour as number | undefined);

/** rAF intervals (ms) from now until stopFrames */
const startFrames = (page: Page) => page.evaluate(() => {
  const w = window as any;
  const gen = (w.__tcGen = (w.__tcGen ?? 0) + 1);
  w.__tcFrames = [];
  w.__tcOn = true;
  let last = performance.now();
  const f = (t: number) => {
    if (!w.__tcOn || w.__tcGen !== gen) return;
    w.__tcFrames.push(t - last);
    last = t;
    requestAnimationFrame(f);
  };
  requestAnimationFrame(f);
});
const stopFrames = async (page: Page) => {
  const iv = await page.evaluate(() => {
    const w = window as any;
    w.__tcOn = false;
    return (w.__tcFrames as number[]).slice(1);
  });
  const s = [...iv].sort((a, b) => a - b);
  const q = (p: number) => Math.round(s[Math.min(s.length - 1, Math.floor(p * s.length))] * 10) / 10;
  return { frames: iv.length, median: q(0.5), p95: q(0.95), max: Math.round((s[s.length - 1] ?? 0) * 10) / 10 };
};

async function waitSettled(page: Page, want: TimeOfDay, timeoutMs: number) {
  const end = Date.now() + timeoutMs;
  while (Date.now() < end) {
    const g = await tod(page);
    if (!g.target && g.current === want) return true;
    await sleep(30);
  }
  return false;
}

async function main() {
  await mkdir(outDir, { recursive: true });
  const server = await ensureServer();
  const browser = await chromium.launch({
    channel: 'chrome', headless: false,
    args: ['--window-position=1930,30', '--window-size=1060,900', '--force_high_performance_gpu', '--mute-audio',
      '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-background-timer-throttling',
      '--disable-features=CalculateNativeWinOcclusion', ...extraChromeArgs()],
  });
  const report: Record<string, unknown> = { at: new Date().toISOString(), url };
  const consoleLines: string[] = [];
  try {
    const context = await browser.newContext(viewportOpts());
    await context.addInitScript({ content: 'window.__name = (f) => f;' });
    await context.addInitScript({ content: POINTER_LOCK_STUB });
    const page = await context.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error' || m.type() === 'warning') consoleLines.push(`[${m.type()}] ${m.text()}`);
    });
    page.on('pageerror', (e) => consoleLines.push(`[pageerror] ${e.message}`));

    const t0 = Date.now();
    await page.goto(`${url}?vol=0`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__RM?.ready || window.__RM?.error, null, { timeout: 120_000, polling: 250 });
    report.loadMs = Date.now() - t0;
    const err = await page.evaluate(() => window.__RM.error);
    if (err) throw new Error(err);
    await page.waitForFunction(() => {
      const el = document.getElementById('loading');
      return !el || Number(getComputedStyle(el).opacity) < 0.01;
    }, null, { timeout: 20_000, polling: 100 }).catch(() => undefined);

    // the other three skies decode in a worker after the first frame
    const tReady = Date.now();
    const allReady = await page.waitForFunction(() => ['dawn', 'morning', 'golden', 'night'].every((t) => (window as any).__TOD.ready(t)), null, { timeout: 60_000, polling: 250 })
      .then(() => true, () => false);
    check('all four skies ready', allReady, { secondsAfterLoad: Math.round((Date.now() - tReady) / 100) / 10 });

    const layout = await page.evaluate(() => {
      const b = document.querySelector('.fh-tally')?.getBoundingClientRect();
      const op = (sel: string) => {
        const el = document.querySelector(sel);
        return el ? Number(getComputedStyle(el).opacity) : null;
      };
      return {
        buttons: document.querySelectorAll('.tod-bar, .tod-btn, #hud button').length,
        tally: b ? { x: Math.round(b.x), y: Math.round(b.y) } : null, tallyOpacity: op('.fh-tally'), labelOpacity: op('.tod-label'),
      };
    });
    check('clean screen: no time-of-day buttons, the tally (top left) and the T label hidden', layout.buttons === 0 && !!layout.tally && layout.tally.x < 960
      && layout.tallyOpacity === 0 && layout.labelOpacity === 0, layout);
    check('starts at morning (sound hour)', (await tod(page)).current === 'morning', { hour: await hour(page) });
    // Tab shows the tally while it is held
    await page.keyboard.down('Tab');
    await sleep(150);
    const tabOn = await tallyOn(page);
    await page.keyboard.up('Tab');
    await sleep(150);
    check('Tab shows the tally while held', tabOn && !(await tallyOn(page)), { tabOn });

    // the first key press starts the sound (part 9) and its decoding: let that settle so the fades are measured alone
    // (J is bound to nothing; Shift, Ctrl, Alt, Tab and Escape do not start the sound)
    await page.keyboard.press('KeyJ');
    let decoded = -1, stableSince = Date.now();
    const audioEnd = Date.now() + 20_000;
    while (Date.now() < audioEnd) {
      const d = await page.evaluate(() => (window as any).__RMAudio?.state?.().decoded as number | undefined);
      if (d !== decoded) {
        decoded = d ?? -1;
        stableSince = Date.now();
      } else if (Date.now() - stableSince > 2000) break;
      await sleep(250);
    }
    report.audio = await page.evaluate(() => {
      const s = (window as any).__RMAudio?.state?.();
      return s ? { context: s.context, decoded: s.decoded, decodedMB: s.decodedMB, volume: s.volume, hour: s.hour } : null;
    });

    // baseline frame pacing before any fade
    await startFrames(page);
    await sleep(2000);
    report.steady = await stopFrames(page);

    const fades: Record<string, unknown>[] = [];
    for (const want of ['golden', 'night', 'dawn', 'morning'] as TimeOfDay[]) {
      const from = (await tod(page)).current;
      await startFrames(page);
      const tPress = Date.now();
      await page.keyboard.press('KeyT');
      await sleep(120);
      const shown = await label(page);
      const g = await tod(page);
      const settled = await waitSettled(page, want, 8000);
      const fadeS = Math.round((Date.now() - tPress) / 100) / 10;
      const frames = await stopFrames(page);
      await sleep(300);
      const u = await urlTod(page), h = await hour(page);
      const faded = (await label(page)) === null;
      const ok = settled && shown === want && faded && g.target === want && u === want && h === TOD_HOUR[want];
      check(`T: ${from} -> ${want} (label at once then faded, fade, URL, sound hour)`, ok, { shown, faded, target: g.target, fadeS, url: u, hour: h, frames });
      fades.push({ from, to: want, fadeS, ...frames });
      await page.screenshot({ path: join(outDir, `tod_${want}.png`) });
      await sleep(600);
    }
    report.fades = fades;

    // three quick presses from morning: golden, then one queued step to night, never a third
    await page.keyboard.press('KeyT');
    await sleep(150);
    await page.keyboard.press('KeyT');
    await sleep(150);
    await page.keyboard.press('KeyT');
    const toNight = await waitSettled(page, 'night', 9000);
    await sleep(2500);
    const after = await tod(page);
    check('three quick presses queue one step (morning -> golden -> night)', toNight && after.current === 'night' && !after.target, after);

    // ignored: modifiers, key repeat, typing in a text field
    const before = await tod(page);
    for (const mod of ['ctrlKey', 'altKey', 'metaKey', 'repeat'] as const) {
      await page.evaluate((m) => window.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyT', key: 't', bubbles: true, [m]: true })), mod);
      await sleep(250);
      const g = await tod(page);
      check(`T ignored with ${mod}`, !g.target && g.current === before.current, g);
    }
    for (const kind of ['input', 'textarea', 'contenteditable'] as const) {
      await page.evaluate((k) => {
        const el = document.createElement(k === 'contenteditable' ? 'div' : k);
        if (k === 'contenteditable') el.contentEditable = 'true';
        el.id = 'tc-field';
        Object.assign(el.style, { position: 'fixed', left: '40%', top: '40%', width: '200px', height: '30px', zIndex: '99', background: '#fff' });
        document.body.appendChild(el);
        el.focus();
      }, kind);
      await page.keyboard.press('KeyT');
      await sleep(250);
      const g = await tod(page);
      await page.evaluate(() => document.getElementById('tc-field')?.remove());
      check(`T ignored while typing in a ${kind}`, !g.target && g.current === before.current, g);
    }

    // a golden -> night fade half-way (not measured: a screenshot stalls the page)
    for (const want of ['dawn', 'morning', 'golden'] as TimeOfDay[]) {
      await page.keyboard.press('KeyT');
      await waitSettled(page, want, 8000);
      await sleep(300);
    }
    await page.keyboard.press('KeyT');
    await sleep(1000);
    await page.screenshot({ path: join(outDir, 'fade_golden_to_night_mid.png') });
    check('golden -> night fade reaches night after the mid-fade screenshot', await waitSettled(page, 'night', 8000));

    // photo mode: no buttons, no label, no key
    await page.goto(`${url}?shot=C1&tod=morning`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__RM?.ready || window.__RM?.error, null, { timeout: 120_000, polling: 250 });
    await sleep(1500);
    const noBar = await page.evaluate(() => !document.querySelector('.tod-bar, .tod-btn, .tod-label'));
    await page.keyboard.press('KeyT');
    await sleep(2500);
    const g = await tod(page);
    check('photo mode: no buttons or label and T does nothing', noBar && g.current === 'morning' && !g.target, { noBar, ...g });
  } catch (e) {
    report.error = String((e as Error)?.stack ?? e);
    console.error(e);
  } finally {
    report.checks = checks;
    report.passed = checks.filter((c) => c.ok).length;
    report.failed = checks.filter((c) => !c.ok).map((c) => c.name);
    report.consoleLines = consoleLines;
    await writeFile(join(outDir, 'todcheck.json'), JSON.stringify(report, null, 2), { encoding: 'utf-8' });
    console.log(`${report.passed} passed, ${(report.failed as string[]).length} failed`);
    await browser.close();
    await server?.close();
  }
}

main();
