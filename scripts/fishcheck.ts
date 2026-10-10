/**
 * Parts 6 + 9 play check: the angler fishes with real mouse and key input, and the sound engine plays every fishing
 * sound (at volume 0, in a muted browser: nothing reaches the speakers).
 *
 *   pnpm tsx scripts/fishcheck.ts [--out stage\a-play] [--url http://127.0.0.1:5173/]
 *
 * Play mode with ?debug (window.__player), ?fishfast (bites ~5x sooner), ?fishseed and ?vol=0. Pointer lock is faked
 * before the page loads: requestPointerLock only sets what document.pointerLockElement returns, so the fishing input
 * (which counts the buttons only while the pointer is locked) works and the real mouse is never captured. Clicks,
 * the LMB/RMB holds and W are real Playwright input events. Rounds: a catch (aim, cast, float, bite, hook, a fight
 * reeled with the tension rule, landing, hold, release), an ignored bite that gets away, greedy reeling that snaps
 * the line on a big brown. Checks the walk hand-off (he walks again once the line is in), the HUD tally and that
 * every fishing event sounded (the engine's play() calls by group). Writes fishcheck.json and screenshots under
 * E:\shots\river-morning\<out>; starts and stops its own dev server like scripts/shots.ts.
 */
import { chromium, type Page } from 'playwright-core';
import { extraChromeArgs, viewportOpts } from './chromeFlags';
import { createServer, type ViteDevServer } from 'vite';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const args = process.argv.slice(2);
const arg = (name: string, def: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const url = arg('url', 'http://127.0.0.1:5173/');
const outDir = join('E:\\shots\\river-morning', arg('out', 'p6\\play'));
const extra = arg('extra', '').replaceAll('~', '&');
/** the pool stand at the water's edge (src/fishing/poses.ts ANGLER_STAND), facing across the pool */
const STAND = { x: 4.0, z: 5.0, yawDeg: 20 };

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
    // tsx (esbuild keepNames) wraps named functions in __name(); the callbacks sent to the page need it first
    await context.addInitScript({ content: 'window.__name = (f) => f;' });
    // a real pointer lock in this headed window would capture the mouse of whoever is at the PC: fake one (a plain
    // string, so no build step can rewrite it; play() refuses to click until it is in place)
    await context.addInitScript({ content: POINTER_LOCK_STUB });
    const page = await context.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error' || m.type() === 'warning') consoleLines.push(`[${m.type()}] ${m.text()}`);
    });
    page.on('pageerror', (e) => consoleLines.push(`[pageerror] ${e.message}`));
    const t0 = Date.now();
    await page.goto(`${url}?debug&fishfast&fishseed=7&vol=0${extra ? '&' + extra : ''}`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__RM?.ready || window.__RM?.error, null, { timeout: 120_000, polling: 250 });
    report.loadMs = Date.now() - t0;
    const err = await page.evaluate(() => window.__RM.error);
    if (err) throw new Error(err);
    await page.waitForTimeout(1200);
    report.rounds = await play(page);
  } catch (e) {
    report.error = String((e as Error)?.stack ?? e);
    console.error(e);
  } finally {
    report.checks = checks;
    report.passed = checks.filter((c) => c.ok).length;
    report.failed = checks.filter((c) => !c.ok).map((c) => c.name);
    report.consoleLines = consoleLines;
    await writeFile(join(outDir, 'fishcheck.json'), JSON.stringify(report, null, 2), { encoding: 'utf-8' });
    console.log(`${report.passed} passed, ${(report.failed as string[]).length} failed`);
    await browser.close();
    await server?.close();
  }
}

const fish = (page: Page) => page.evaluate(() => {
  const f = (window as any).__RM_FISH;
  return { state: f.state as string, t: f.stateTime as number, tension: f.tension as number, strain: f.strain as number, power: f.power as number,
    caught: f.caught as number, biggest: f.biggest as string, species: f.species as string | null, lengthCm: f.lengthCm as number,
    bobber: f.bobber as number[], fishVisible: f.fishVisible as number };
});
const playerPos = (page: Page) => page.evaluate(() => {
  const c = (window as any).__player.controller;
  return { x: c.position.x as number, z: c.position.z as number, anim: (window as any).__player.anim.state as string };
});
const hud = (page: Page) => page.evaluate(() => ({
  count: document.querySelector('.fh-count')?.textContent ?? null,
  best: document.querySelectorAll('.fh-value')[1]?.textContent ?? null,
  prompt: document.querySelector('.fh-prompt')?.textContent ?? '',
  promptOn: !!document.querySelector('.fh-prompt.fh-on'),
  card: document.querySelector('.fh-card.fh-on .fh-species')?.textContent ?? null,
}));
async function waitState(page: Page, states: string[], timeoutMs: number): Promise<string | null> {
  const end = Date.now() + timeoutMs;
  while (Date.now() < end) {
    const s = await page.evaluate(() => (window as any).__RM_FISH.state as string);
    if (states.includes(s)) return s;
    await sleep(40);
  }
  return null;
}
/**
 * The game is frozen (window.__hold) while the capture runs: with vsync and the frame-rate limit off (RM_UNCAPPED,
 * 144-240 Hz screens) a screenshot can take over a second, and the cast meter, a bite's window or a fight ran on
 * meanwhile (the cast topped out at power 1, the bite got away before the strike).
 */
const shot = async (page: Page, name: string) => {
  await page.evaluate(() => (window as any).__hold?.(true));
  try {
    await page.screenshot({ path: join(outDir, `${name}.png`) });
  } finally {
    await page.evaluate(() => (window as any).__hold?.(false));
  }
};

/** stand at the pool's edge facing across it, the camera behind him */
async function toStand(page: Page) {
  await page.evaluate((s) => {
    const P = (window as any).__player;
    const a = (s.yawDeg * Math.PI) / 180, fx = -Math.sin(a), fz = -Math.cos(a);
    P.teleportTo({ x: s.x, y: 0, z: s.z }, { x: s.x + fx * 10, y: 0, z: s.z + fz * 10 });
    P.camera.yaw = Math.atan2(fx, fz);
    P.camera.pitch = (10 * Math.PI) / 180;
  }, STAND);
  await page.waitForTimeout(400);
}

/** hold LMB until the power reaches `power`, then let go; returns the cast power */
async function cast(page: Page, power: number, shotName?: string): Promise<number> {
  await page.mouse.down({ button: 'left' });
  // the power of the last cast stays readable until the new aim starts
  await waitState(page, ['aiming'], 1500);
  const end = Date.now() + 3000;
  let p = 0;
  let shotTaken = !shotName;
  while (Date.now() < end) {
    p = await page.evaluate(() => (window as any).__RM_FISH.power as number);
    if (p >= power - 0.04) break;
    if (!shotTaken && p > 0.2) {
      await shot(page, shotName!);
      shotTaken = true;
    }
  }
  await page.mouse.up({ button: 'left' });
  return p;
}

/** reel with the tension rule (on below 0.55, off above 0.82 or once the line strains) until the fight ends */
/**
 * Plays a fight with the mouse. `greedy` holds the reel throughout; with `leadMs` it first stays off the reel until the
 * drag has zipped once (at most `leadMs`): with the reel held the drag only slips above 1.2 while the line already
 * strains above 1.0, so a greedy fight can snap without the drag ever running.
 */
async function fight(page: Page, greedy: boolean, shotAt?: { name: string; after: number }, leadMs = 0) {
  const t0 = Date.now();
  let reeling = false, maxT = 0, maxStrain = 0, shotTaken = !shotAt, lead = greedy && leadMs > 0, leadEnd = 0;
  for (;;) {
    const f = await fish(page);
    if (f.state !== 'fight') {
      if (reeling) await page.mouse.up({ button: 'right' });
      return { end: f.state, seconds: (Date.now() - t0) / 1000, maxTension: +maxT.toFixed(2), maxStrain: +maxStrain.toFixed(2), species: f.species, lengthCm: f.lengthCm, ...(leadMs ? { leadSeconds: (leadEnd || Date.now()) / 1000 - t0 / 1000 } : {}) };
    }
    maxT = Math.max(maxT, f.tension);
    maxStrain = Math.max(maxStrain, f.strain);
    if (!shotTaken && Date.now() - t0 > shotAt!.after) {
      await shot(page, shotAt!.name);
      shotTaken = true;
    }
    if (lead && (Date.now() - t0 > leadMs || (await page.evaluate(() => (window as any).__snd?.played?.reel_drag ?? 0)) > 0)) {
      lead = false;
      leadEnd = Date.now();
    }
    const want: boolean = lead ? false : greedy || (reeling ? !(f.tension > 0.82 || f.strain > 0.05) : f.tension < 0.55);
    if (want !== reeling) {
      reeling = want;
      if (want) await page.mouse.down({ button: 'right' });
      else await page.mouse.up({ button: 'right' });
    }
    if (Date.now() - t0 > 120_000) throw new Error('fight did not end in 120 s');
    await sleep(40);
  }
}

async function play(page: Page) {
  const rounds: Record<string, unknown> = {};
  // the sound engine's calls by catalog group (every one-shot goes through play(); the reel loop has its own)
  await page.evaluate(() => {
    const A = (window as any).__RMAudio;
    if (!A) throw new Error('no window.__RMAudio (sound not installed)');
    const eng = A.engine, rec = { played: {} as Record<string, number>, failed: {} as Record<string, number>, reelStart: 0, reelStop: 0, log: [] as string[] };
    (window as any).__snd = rec;
    const play = eng.play.bind(eng);
    eng.play = (id: string, ...rest: unknown[]) => {
      const ok = play(id, ...rest);
      (ok ? rec.played : rec.failed)[id] = ((ok ? rec.played : rec.failed)[id] ?? 0) + 1;
      if (rec.log.length < 400) rec.log.push(`${((window as any).__RM_FISH?.state ?? '')}:${id}${ok ? '' : '(x)'}`);
      return ok;
    };
    const rs = eng.reelStart.bind(eng), rp = eng.reelStop.bind(eng);
    eng.reelStart = (...a: unknown[]) => { rec.reelStart++; return rs(...a); };
    eng.reelStop = () => { rec.reelStop++; return rp(); };
  });
  const stub = await page.evaluate(() => (window as any).__pointerLockStub === true && !String(Element.prototype.requestPointerLock).includes('[native code]'));
  check('pointer lock stub installed before any click', stub);
  if (!stub) throw new Error('pointer lock stub missing: refusing to click (a real lock would capture the mouse)');
  const canvas = await page.$('canvas');
  const box = (await canvas!.boundingBox())!;
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);

  // the click that captures the mouse (faked) never casts; it is also the gesture that starts the sound
  const before = await fish(page);
  check('fishing idle at start, line in', before.state === 'idle', before.state);
  await page.mouse.click(cx, cy);
  await page.waitForTimeout(300);
  check('the locking click did not cast', (await fish(page)).state === 'idle');
  const lock = await page.evaluate(() => document.pointerLockElement?.tagName ?? null);
  check('pointer lock faked (real mouse never captured)', lock === 'CANVAS' && stub, lock);

  // walking works while the line is in (the fishing loop's idle hands back to locomotion)
  const p0 = await playerPos(page);
  await page.keyboard.down('KeyW');
  await page.waitForTimeout(1300);
  await page.keyboard.up('KeyW');
  await page.waitForTimeout(300);
  const p1 = await playerPos(page);
  const walked = Math.hypot(p1.x - p0.x, p1.z - p0.z);
  check('he walks with the line in (W)', walked > 0.5, { walked: +walked.toFixed(2), anim: p0.anim });

  // sound: running, every file decoded
  const decoded = await page.waitForFunction(() => (window as any).__RMAudio.state().decoded >= 60, null, { timeout: 20_000, polling: 250 })
    .then(() => true, () => false);
  const a0 = await page.evaluate(() => (window as any).__RMAudio.state());
  check('sound running after the first gesture, files decoded', a0.context === 'running' && decoded, { context: a0.context, decoded: a0.decoded, volume: a0.volume, MB: a0.decodedMB });
  check('sound at volume 0 (nothing audible)', a0.volume === 0, a0.volume);

  // ---- round 1: a catch
  await toStand(page);
  // from the water's edge the pool takes power ~0.3-0.5 (more lands on the far bank)
  const pw = await cast(page, 0.4, 'a01-aim');
  check('cast power in range', pw > 0.3 && pw < 0.62, +pw.toFixed(2));
  const fl = await waitState(page, ['flight'], 4000);
  await page.waitForTimeout(250);
  await shot(page, 'a02-flight');
  check('the float flies', fl === 'flight');
  const st1 = await waitState(page, ['floating'], 6000);
  await page.waitForTimeout(600);
  const f1 = await fish(page);
  await shot(page, 'a03-float');
  const h1 = await hud(page);
  check('float lands on the pool', st1 === 'floating' && /wait for a bite/i.test(h1.prompt), { bobber: f1.bobber.map((v) => +v.toFixed(2)), prompt: h1.prompt });
  const b1 = await waitState(page, ['bite'], 40_000);
  check('a bite comes', b1 === 'bite');
  // with part 7 (window.__RM_TROUT) the bite is one of the river's trout, which came to the bait
  const trout = await page.evaluate(() => !!(window as any).__RM_TROUT);
  if (trout) {
    const sf = await page.evaluate(() => (window as any).__RM_FISH.schoolFish as number);
    check('the bite is a school trout (part 7)', sf >= 0, sf);
  }
  await page.waitForTimeout(120);
  await shot(page, 'a04-bite');
  await page.mouse.click(cx, cy);
  const hooked = await waitState(page, ['fight'], 1500);
  check('LMB hooks the fish', hooked === 'fight', (await fish(page)).species);
  await page.waitForTimeout(150);
  const strikeAnim = (await playerPos(page)).anim;
  check('the hook set plays (strike), then the reel', strikeAnim === 'strike', strikeAnim);
  const r1 = await fight(page, false, { name: 'a05-fight', after: 1800 });
  rounds.catch = r1;
  check('skilled reeling lands it', r1.end === 'landing' || r1.end === 'holding', r1);
  const hold = await waitState(page, ['holding'], 3000);
  await page.waitForTimeout(700);
  await shot(page, 'a06-hold');
  const h2 = await hud(page);
  check('held up with the catch card', hold === 'holding' && !!h2.card, h2);
  await waitState(page, ['releasing'], 6000);
  await page.waitForTimeout(500);
  await shot(page, 'a07-release');
  const idle1 = await waitState(page, ['idle'], 8000);
  const f2 = await fish(page);
  const h3 = await hud(page);
  check('released, line in again', idle1 === 'idle');
  if (trout) {
    const held = await page.evaluate(() => ((window as any).__RM_TROUT.summary() as string[]).filter((l) => / held /.test(l)));
    check('the released trout is back in the river (none held)', held.length === 0, held);
  }
  check('HUD counts the catch and the biggest fish', f2.caught === 1 && h3.count === '1' && !!h3.best && h3.best !== '\u2014', { caught: f2.caught, hud: h3 });
  await page.waitForTimeout(800);
  const q0 = await playerPos(page);
  await page.keyboard.down('KeyW');
  await page.waitForTimeout(900);
  await page.keyboard.up('KeyW');
  const q1 = await playerPos(page);
  check('he walks again after the release', Math.hypot(q1.x - q0.x, q1.z - q0.z) > 0.3, { anim: q0.anim });

  // ---- round 2: an ignored bite gets away
  await toStand(page);
  await cast(page, 0.4);
  await waitState(page, ['floating'], 8000);
  const b2 = await waitState(page, ['bite'], 40_000);
  check('second bite comes', b2 === 'bite');
  const esc = await waitState(page, ['floating'], 3000);
  await page.waitForTimeout(150);
  const h4 = await hud(page);
  await shot(page, 'a08-escaped');
  const log2 = await page.evaluate(() => (window as any).__RM_FISH.log.slice(-6).map((e: { type: string }) => e.type));
  check('an ignored bite gets away', esc === 'floating' && /got away/i.test(h4.prompt) && log2.includes('escape'), { prompt: h4.prompt, log: log2 });
  // a strike at nothing: the float jerks, he goes back to waiting (not to reeling)
  await page.mouse.click(cx, cy);
  await page.waitForTimeout(1600);
  const miss = await playerPos(page);
  const log3 = await page.evaluate(() => (window as any).__RM_FISH.log.slice(-3).map((e: { type: string }) => e.type));
  check('a strike at nothing goes back to waiting', log3.includes('miss') && miss.anim === 'wait', { anim: miss.anim, log: log3 });
  // reel in
  await page.mouse.down({ button: 'right' });
  const in2 = await waitState(page, ['idle'], 15_000);
  await page.mouse.up({ button: 'right' });
  check('holding RMB reels the float in', in2 === 'idle');

  // ---- round 3: greedy reeling snaps the line on a big brown
  await toStand(page);
  await page.evaluate(() => (window as any).__RM_FISH.forceNext('brown', 58, 1.5));
  await cast(page, 0.4);
  await waitState(page, ['floating'], 8000);
  // with part 7 the biggest brown nearby swims over to the bait (up to ~24 m), but with 43 trout another one can take
  // it on its own first, and that bite clears the forced one (a 31 cm rainbow cannot snap the line): let such a bite
  // go, ask for the brown again and wait for it (at most three tries)
  let b3: string | null = null;
  const skipped: string[] = [];
  for (let tries = 0; tries < 3; tries++) {
    b3 = await waitState(page, ['bite'], 45_000);
    if (b3 !== 'bite') break;
    const bf = await fish(page);
    if (!bf.species || (bf.species === 'brown' && bf.lengthCm >= 40)) break;
    skipped.push(`${bf.species} ${bf.lengthCm}`);
    await waitState(page, ['floating'], 4000);
    await page.evaluate(() => (window as any).__RM_FISH.forceNext('brown', 58, 1.5));
    b3 = null;
  }
  if (skipped.length) rounds.greedySkipped = skipped;
  await page.waitForTimeout(100);
  await page.mouse.click(cx, cy);
  const hooked3 = await waitState(page, ['fight'], 1500);
  check('big brown hooked', b3 === 'bite' && hooked3 === 'fight', (await fish(page)).lengthCm);
  const r3 = await fight(page, true, undefined, 4000);
  rounds.greedy = r3;
  await page.waitForTimeout(200);
  const h5 = await hud(page);
  await shot(page, 'a09-snapped');
  check('greedy reeling snaps the line', r3.end === 'snapped' && /snapped/i.test(h5.prompt), { ...r3, prompt: h5.prompt });
  await waitState(page, ['idle'], 4000);

  // ---- sound: every fishing event played its group
  const snd = await page.evaluate(() => (window as any).__snd);
  rounds.sound = snd;
  const n = (id: string) => snd.played[id] ?? 0;
  check('cast whoosh on every cast', n('cast') + n('cast_heavy') >= 3, { cast: n('cast'), heavy: n('cast_heavy') });
  check('float landing splashes', n('splash_small') >= 3, n('splash_small'));
  check('bite plips', n('bite') >= 3, n('bite'));
  check('reel loop started and stopped', snd.reelStart >= 2 && snd.reelStop >= 2, { start: snd.reelStart, stop: snd.reelStop });
  check('drip when the fish is lifted out', n('drip') >= 1, n('drip'));
  check('drag zips while the fish takes line', n('reel_drag') >= 1, n('reel_drag'));
  check('line pulled off the reel by a longer cast', n('reel_pull') >= 1, n('reel_pull'));
  check('footsteps while walking', n('step_gravel') + n('step_grass') + n('step_wood') + n('step_creak') >= 2,
    Object.fromEntries(Object.entries(snd.played).filter(([k]) => k.startsWith('step_'))));
  check('no sound failed to play', Object.keys(snd.failed).length === 0, snd.failed);
  const a1 = await page.evaluate(() => (window as any).__RMAudio.state());
  rounds.audioState = a1;
  rounds.frame = await page.evaluate(() => ({ frame: window.__RM.frameStats(), info: window.__RM.info() }));
  return rounds;
}

main();
