/**
 * Play check for the rod put down and picked up again (E) and the clean screen around it.
 *
 *   pnpm tsx scripts/rodcheck.ts [--out prep\code\rod\checks\rodcheck] [--url http://127.0.0.1:5173/]
 *
 * Play mode with ?debug (window.__player), ?fishfast, ?fishseed=7 and ?vol=0 in a muted browser; pointer lock faked
 * before the page loads (the real mouse is never captured). At the pool's edge: the line hangs short at the tip; E in
 * the shallows says it cannot go down there; E on the bar lays it on the ground beside him (dry, on the floor, not in a
 * rock); the fishing buttons do nothing and the line is hidden; he walks 5 m away (no prompt), a jump moves him
 * without it, he walks back and the prompt shows; E picks it up; then a cast, a bite, a fight and a landed fish. Writes
 * rodcheck.json and screenshots under E:\shots\river-morning\<out>; starts and stops its own dev server like
 * scripts/shots.ts.
 */
import { chromium, type Page } from 'playwright-core';
import { createServer, type ViteDevServer } from 'vite';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const args = process.argv.slice(2);
const arg = (name: string, def: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const url = arg('url', 'http://127.0.0.1:5173/');
const outDir = join('E:\\shots\\river-morning', arg('out', 'prep\\code\\rod\\checks\\rodcheck'));
/** the pool stand at the water's edge (as fishcheck.ts), facing across the pool */
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

/** the angler, his rod layer and the fishing loop as the checks read them */
const state = (page: Page) => page.evaluate(() => {
  const P = (window as any).__player, F = (window as any).__RM_FISH, R = P.rodGround;
  const V = P.controller.position.constructor;
  const tip = P.rod.tipWorld(new V()), hand = new V().setFromMatrixPosition(P.character.bone('hand_r').matrixWorld);
  const rodPos = P.rod.mesh.position;
  const g = { y: 0, waterY: 0, depth: 0, surface: null };
  const laid = R.laidAt;
  let floor = null as null | { y: number; depth: number; deck: boolean };
  if (laid) {
    P.ground.sample(laid.x, laid.z, g);
    floor = { y: g.y, depth: g.depth, deck: !!g.surface };
  }
  return {
    phase: R.phase as string, place: R.place as string, canPickUp: R.canPickUp as boolean, laid: laid as null | { x: number; y: number; z: number; yaw: number },
    floor, x: P.controller.position.x as number, z: P.controller.position.z as number, anim: P.anim.state as string, rodLayer: P.anim.rodLayer as number,
    away: P.poser.away as number, resting: P.rod.resting as boolean, rod: [rodPos.x, rodPos.y, rodPos.z] as number[], handToRod: hand.distanceTo(rodPos) as number,
    fish: F.state as string, caught: F.caught as number, lineVisible: F.visible as boolean, hang: Math.hypot(F.bobber[0] - tip.x, F.bobber[1] - tip.y, F.bobber[2] - tip.z),
    prompt: document.querySelector('.rod-prompt.on')?.textContent ?? '', fishPrompt: document.querySelector('.fh-prompt.fh-on')?.textContent ?? '',
    tallyOn: !!document.querySelector('.fh-tally.fh-on'),
  };
});
type S = Awaited<ReturnType<typeof state>>;
async function waitFor(page: Page, ok: (s: S) => boolean, timeoutMs: number): Promise<S> {
  const end = Date.now() + timeoutMs;
  let s = await state(page);
  while (!ok(s) && Date.now() < end) {
    await sleep(50);
    s = await state(page);
  }
  return s;
}
const shot = (page: Page, name: string) => page.screenshot({ path: join(outDir, `${name}.png`) });
const r2 = (v: number) => Math.round(v * 100) / 100;

/** point the camera (and so the walking keys) toward a world point, then hold W until `until` or the timeout */
async function walkToward(page: Page, x: number, z: number, until: (s: S) => boolean, timeoutMs: number) {
  await page.evaluate(([tx, tz]) => {
    const P = (window as any).__player, c = P.controller.position;
    P.camera.yaw = Math.atan2(tx - c.x, tz - c.z);
  }, [x, z]);
  await page.keyboard.down('KeyW');
  const s = await waitFor(page, until, timeoutMs);
  await page.keyboard.up('KeyW');
  await sleep(500);
  return s;
}

async function toStand(page: Page) {
  await page.evaluate((s) => {
    const P = (window as any).__player;
    const a = (s.yawDeg * Math.PI) / 180, fx = -Math.sin(a), fz = -Math.cos(a);
    P.teleportTo({ x: s.x, y: 0, z: s.z }, { x: s.x + fx * 10, y: 0, z: s.z + fz * 10 });
    P.camera.yaw = Math.atan2(fx, fz);
    P.camera.pitch = (10 * Math.PI) / 180;
  }, STAND);
  await sleep(500);
}

async function play(page: Page) {
  const stub = await page.evaluate(() => (window as any).__pointerLockStub === true && !String(Element.prototype.requestPointerLock).includes('[native code]'));
  check('pointer lock stub installed before any click', stub);
  if (!stub) throw new Error('pointer lock stub missing: refusing to click (a real lock would capture the mouse)');
  const box = (await (await page.$('canvas'))!.boundingBox())!;
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.click(cx, cy);
  await sleep(400);

  // the line reeled up short at the tip while he walks with the rod
  await toStand(page);
  await page.keyboard.down('KeyW');
  await sleep(700);
  await page.keyboard.up('KeyW');
  await sleep(1200);
  const s0 = await state(page);
  check('rod in hand, line in, the float hanging short at the tip', s0.phase === 'hand' && s0.fish === 'idle' && s0.hang < 0.3 && s0.lineVisible, { hang: r2(s0.hang), phase: s0.phase });
  await shot(page, 'r00-hand');

  // in the shallows there is nowhere to lay it
  await toStand(page);
  const wade = await walkToward(page, STAND.x - Math.sin(0.35) * 10, STAND.z - Math.cos(0.35) * 10, () => false, 3500);
  const depth = await page.evaluate(() => (window as any).__player.controller.floor.depth as number);
  await page.keyboard.press('KeyE');
  await sleep(250);
  const sw = await state(page);
  check("E in the water: \"Can't put the rod down here\", rod kept", sw.phase === 'hand' && /can.t put the rod down here/i.test(sw.prompt), { depth: r2(depth), prompt: sw.prompt, x: r2(wade.x), z: r2(wade.z) });
  await shot(page, 'r01-cant');
  await sleep(1800);

  // on the bar: E lays it down beside him
  await toStand(page);
  await page.evaluate(() => {
    const P = (window as any).__player;
    P.camera.yaw += Math.PI;
  });
  await page.keyboard.down('KeyW');
  await sleep(1500);
  await page.keyboard.up('KeyW');
  await sleep(700);
  await page.keyboard.press('KeyE');
  const mid = await waitFor(page, (s) => s.anim === 'kneel', 1500);
  check('E: he crouches to lay it down', mid.anim === 'kneel' && mid.phase !== 'hand', { anim: mid.anim, phase: mid.phase });
  await sleep(250);
  await shot(page, 'r02-laying');
  const s1 = await waitFor(page, (s) => s.phase === 'laid', 4000);
  const gap = s1.laid && s1.floor ? s1.laid.y - s1.floor.y : NaN;
  check('the rod lies on the ground beside him (reel seat over the floor, dry or on the deck)', s1.phase === 'laid' && !!s1.laid && gap > 0.0 && gap < 0.12
    && !!s1.floor && (s1.floor.deck || s1.floor.depth < 0) && s1.resting, { laid: s1.laid && [r2(s1.laid.x), r2(s1.laid.y), r2(s1.laid.z)], gapAboveFloor: r2(gap), floor: s1.floor });
  check('hands free: no rod layer, no line, standing again', s1.rodLayer === 0 && s1.away === 1 && !s1.lineVisible && s1.anim === 'loco', { rodLayer: s1.rodLayer, line: s1.lineVisible, anim: s1.anim });
  await shot(page, 'r03-laid');

  // the fishing buttons do nothing without the rod
  await page.mouse.down({ button: 'left' });
  await sleep(700);
  await page.mouse.up({ button: 'left' });
  await sleep(300);
  const sb = await state(page);
  check('LMB without the rod: no aim, no cast, no prompt', sb.fish === 'idle' && sb.phase === 'laid' && sb.fishPrompt === '', { fish: sb.fish, fishPrompt: sb.fishPrompt });

  // 5 m away and back
  const laid = s1.laid!;
  // along the bar, upstream of the stand (open gravel; the bank rises behind him and the dock lies downstream)
  const away = await walkToward(page, laid.x - 0.94 * 20, laid.z + 0.342 * 20, (s) => Math.hypot(s.x - laid.x, s.z - laid.z) > 5.2, 12_000);
  const dAway = Math.hypot(away.x - laid.x, away.z - laid.z);
  check('he walks 5 m away hands-free; the rod stays; no prompt out there', dAway > 5 && !!away.laid && Math.hypot(away.laid.x - laid.x, away.laid.z - laid.z) < 1e-3
    && away.prompt === '' && away.rodLayer === 0, { dAway: r2(dAway), prompt: away.prompt });
  await shot(page, 'r04-away');

  // a jump moves him; the rod stays where it lies
  await page.keyboard.press('Digit1');
  await sleep(700);
  const tr = await state(page);
  const moved = Math.hypot(tr.x - away.x, tr.z - away.z);
  check('jump: he moves without the rod, it stays', moved > 3 && tr.phase === 'laid' && !!tr.laid && Math.hypot(tr.laid.x - laid.x, tr.laid.z - laid.z) < 1e-3, { moved: r2(moved) });
  await page.evaluate(([x, z, lx, lz]) => {
    (window as any).__player.teleportTo({ x, y: 0, z }, { x: lx, y: 0, z: lz });
  }, [away.x, away.z, laid.x, laid.z]);
  await sleep(600);

  const back = await walkToward(page, laid.x, laid.z, (s) => s.canPickUp, 9000);
  await sleep(500);
  const sp = await state(page);
  check('walking back: "E Pick up rod" shows within reach, facing it', back.canPickUp && /pick up rod/i.test(sp.prompt), { prompt: sp.prompt, d: r2(Math.hypot(back.x - laid.x, back.z - laid.z)) });
  await shot(page, 'r05-prompt');
  await page.keyboard.press('KeyE');
  const crouch = await waitFor(page, (s) => s.anim === 'kneel', 4000);
  check('E: he goes to it and crouches', crouch.anim === 'kneel', { phase: crouch.phase });
  const s2 = await waitFor(page, (s) => s.phase === 'hand', 5000);
  const tHand = Date.now();
  await sleep(800);
  // the fresh float swings under the rising tip for a moment (fishing polish rope): wait for it to settle, at most 3 s
  const s3 = await waitFor(page, (s) => s.hang < 0.3, 3000);
  check('picked up: rod in hand, line back, ready to fish', s2.phase === 'hand' && s3.rodLayer === 1 && s3.away === 0 && !s3.resting && s3.lineVisible
    && s3.handToRod < 0.35 && s3.fish === 'idle' && s3.hang < 0.3,
    { handToRod: r2(s3.handToRod), hang: r2(s3.hang), rodLayer: s3.rodLayer, settledS: r2((Date.now() - tHand) / 1000) });
  await shot(page, 'r06-picked');

  // and it fishes: cast, bite, fight, land
  await toStand(page);
  const caught0 = (await state(page)).caught;
  await page.mouse.down({ button: 'left' });
  await waitFor(page, (s) => s.fish === 'aiming', 1500);
  const pEnd = Date.now() + 3000;
  while (Date.now() < pEnd && (await page.evaluate(() => (window as any).__RM_FISH.power as number)) < 0.36) await sleep(20);
  await page.mouse.up({ button: 'left' });
  const fl = await waitFor(page, (s) => s.fish === 'floating', 8000);
  check('cast after the pick-up: the float lands', fl.fish === 'floating', fl.fish);
  const bite = await waitFor(page, (s) => s.fish === 'bite', 40_000);
  check('a bite comes', bite.fish === 'bite');
  await page.mouse.click(cx, cy);
  const hooked = await waitFor(page, (s) => s.fish === 'fight', 1500);
  check('LMB hooks it', hooked.fish === 'fight');
  let reeling = false;
  const fEnd = Date.now() + 120_000;
  for (;;) {
    const f = await page.evaluate(() => {
      const F = (window as any).__RM_FISH;
      return { state: F.state as string, tension: F.tension as number, strain: F.strain as number };
    });
    if (f.state !== 'fight' || Date.now() > fEnd) break;
    const want: boolean = reeling ? !(f.tension > 0.82 || f.strain > 0.05) : f.tension < 0.55;
    if (want !== reeling) {
      reeling = want;
      if (want) await page.mouse.down({ button: 'right' });
      else await page.mouse.up({ button: 'right' });
    }
    await sleep(40);
  }
  if (reeling) await page.mouse.up({ button: 'right' });
  const held = await waitFor(page, (s) => s.fish === 'holding', 4000);
  check('the fish is landed and held', held.fish === 'holding' && held.caught === caught0 + 1, { fish: held.fish, caught: held.caught });
  await sleep(500);
  await shot(page, 'r07-hold');
  const tally = await state(page);
  check('the tally shows after the catch', tally.tallyOn);
  const done = await waitFor(page, (s) => s.fish === 'idle', 10_000);
  check('released, line in, rod in hand', done.fish === 'idle' && done.phase === 'hand');
  await sleep(4500);
  const later = await state(page);
  check('the tally fades a few seconds after the change', !later.tallyOn);
}

async function main() {
  await mkdir(outDir, { recursive: true });
  const server = await ensureServer();
  const browser = await chromium.launch({
    channel: 'chrome', headless: false,
    args: ['--window-position=1930,30', '--window-size=1060,900', '--force_high_performance_gpu', '--mute-audio',
      '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-background-timer-throttling',
      '--disable-features=CalculateNativeWinOcclusion'],
  });
  const report: Record<string, unknown> = { at: new Date().toISOString(), url };
  const consoleLines: string[] = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
    await context.addInitScript({ content: 'window.__name = (f) => f;' });
    await context.addInitScript({ content: POINTER_LOCK_STUB });
    const page = await context.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error' || m.type() === 'warning') consoleLines.push(`[${m.type()}] ${m.text()}`);
    });
    page.on('pageerror', (e) => consoleLines.push(`[pageerror] ${e.message}`));
    const t0 = Date.now();
    await page.goto(`${url}?debug&fishfast&fishseed=7&vol=0`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__RM?.ready || window.__RM?.error, null, { timeout: 120_000, polling: 250 });
    report.loadMs = Date.now() - t0;
    const err = await page.evaluate(() => window.__RM.error);
    if (err) throw new Error(err);
    await sleep(1200);
    await play(page);
  } catch (e) {
    report.error = String((e as Error)?.stack ?? e);
    console.error(e);
  } finally {
    report.checks = checks;
    report.passed = checks.filter((c) => c.ok).length;
    report.failed = checks.filter((c) => !c.ok).map((c) => c.name);
    report.consoleLines = consoleLines;
    await writeFile(join(outDir, 'rodcheck.json'), JSON.stringify(report, null, 2), { encoding: 'utf-8' });
    console.log(`${report.passed} passed, ${(report.failed as string[]).length} failed`);
    await browser.close();
    await server?.close();
  }
}

main();
