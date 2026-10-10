/**
 * Play check for the cabin's door and the room inside (E opens the door, he walks in, the camera follows).
 *
 *   pnpm tsx scripts/cabincheck.ts [--out prep\code\cabin\checks\cabincheck] [--url http://127.0.0.1:5187/]
 *
 * Play mode with ?debug (window.__player, __cabin), ?fishseed=7 and ?vol=0 in a muted browser on the second monitor;
 * pointer lock faked before the page loads (the real mouse is never captured). Hands free (the rod laid in the yard):
 * the closed door blocks him; E opens it; he walks in, turns round and walks out; E closes it. Then the same with the
 * rod in hand (LMB does nothing indoors), and the rod laid on the cabin floor and picked up again. Every frame of the
 * walks the camera's eye is tested against the walls, the roof, the lintel and the furniture. Writes cabincheck.json
 * and screenshots under E:\shots\river-morning\<out>; starts and stops its own dev server like scripts/shots.ts.
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
const url = arg('url', 'http://127.0.0.1:5187/');
const outDir = join('E:\\shots\\river-morning', arg('out', 'prep\\code\\cabin\\checks\\cabincheck'));

async function ensureServer(): Promise<ViteDevServer | null> {
  try {
    if ((await fetch(url)).ok) return null;
  } catch {
    // not running
  }
  const server = await createServer({ root: join(import.meta.dirname, '..'), logLevel: 'warn', server: { host: '127.0.0.1', port: Number(new URL(url).port) || 5187, strictPort: true } });
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
  // tsx keeps function names with a __name() helper inside the page.evaluate callbacks
  window.__name = (f) => f;
})();`;

/** the cabin frame (src/camp/layout.ts: LAYOUT.cabin x 9, z 23.5, rotY -0.12, turned half a turn) */
const CY = -0.12 + Math.PI, CC = Math.cos(CY), CS = Math.sin(CY);
const W = (lx: number, lz: number) => ({ x: 9 + lx * CC + lz * CS, z: 23.5 - lx * CS + lz * CC });
const r2 = (v: number) => Math.round(v * 100) / 100;

/**
 * What the checks read; installs (once) a per-frame probe of the eye: the worst intrusion into the logs, roof,
 * lintel or furniture since the last reset (m), and of the rod's tip into the ceiling and walls.
 */
const state = (page: Page) => page.evaluate(() => {
  const w = window as any, P = w.__player, C = w.__cabin, R = P.rodGround, F = w.__RM_FISH;
  const camp = w.__camp;
  const CY = -0.12 + Math.PI, cc = Math.cos(CY), cs = Math.sin(CY);
  const toL = (x: number, z: number) => [(x - 9) * cc - (z - 23.5) * cs, (x - 9) * cs + (z - 23.5) * cc];
  if (!w.__eyeProbe) {
    w.__eyeProbe = { worst: 0, where: null as unknown, frames: 0, tipWorst: 0 };
    const cam = P.camera.camera ?? null;
    void cam;
    const tick = () => {
      const e = w.__eyeProbe;
      const c = (w.__RM_CAMERA ?? null) as any;
      void c;
      const p = (P as any).engine?.camera?.position ?? (P.camera as any).camera?.position;
      if (p) {
        const [lx, lz] = toL(p.x, p.z), ly = p.y - camp.cabinY;
        const ax = Math.abs(lx), az = Math.abs(lz);
        let d = 0;
        // inside the log walls' slab (between the inner and the outer bulge), not in the doorway's opening
        const inDoorway = lx > -1.19 && lx < -0.21 && ly < 2.622 && lz > 2.0;
        if (ly > 0.3 && ly < 3.2) {
          const slab = Math.max(Math.min(ax - 2.68, 3.15 - ax) * (az < 2.65 ? 1 : 0), Math.min(az - 2.18, 2.65 - az) * (ax < 3.15 ? 1 : 0));
          if (slab > 0 && !inDoorway) d = Math.max(d, slab);
        }
        if (ax < 2.68 && az < 2.18) {
          d = Math.max(d, ly - (4.57 - 0.629 * az), ly - 3.62);
          for (const b of camp.interior.blocks) {
            const dx = b.hx - Math.abs(lx - b.x), dz = b.hz - Math.abs(lz - b.z), dy = Math.min(ly - b.y0, b.y1 - ly);
            if (dx > 0 && dz > 0 && dy > 0) d = Math.max(d, Math.min(dx, dz, dy));
          }
          d = Math.max(d, 0.55 + 0.05 - ly);
        }
        if (d > e.worst) {
          e.worst = d;
          e.where = [Math.round(lx * 100) / 100, Math.round(ly * 100) / 100, Math.round(lz * 100) / 100];
        }
        e.frames++;
        // the rod's tip inside the room: how far into the roof or the walls
        if (C.inside && R.phase === 'hand') {
          const V = P.controller.position.constructor;
          const t = P.rod.tipWorld(new V());
          const [tx, tz] = toL(t.x, t.z), ty = t.y - camp.cabinY;
          // into the roof over the room, or into a log wall's slab (not out through the open doorway)
          const atx = Math.abs(tx), atz = Math.abs(tz);
          const doorway = tx > -1.19 && tx < -0.21 && ty < 2.622;
          let into = 0;
          if (atx < 2.68 && atz < 2.18) into = Math.max(0, ty - (4.57 - 0.629 * atz));
          else if (!doorway && atx < 3.15 && atz < 2.65 && ty > 0.5 && ty < 3.1) into = Math.min(atx > 2.68 ? atx - 2.68 : 9, atz > 2.18 ? atz - 2.18 : 9);
          e.tipWorst = Math.max(e.tipWorst, into);
        }
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  const [lx, lz] = toL(P.controller.position.x, P.controller.position.z);
  return {
    lx, lz, inside: C.inside as boolean, open: C.open as number, blocked: C.blocked as boolean, phase: R.phase as string, anim: P.anim.state as string,
    prompt: document.querySelector('.rod-prompt.on')?.textContent ?? '', fishPrompt: document.querySelector('.fh-prompt.fh-on')?.textContent ?? '',
    roomShown: camp.interior.root.visible as boolean, fish: F?.state as string, eye: w.__eyeProbe.worst as number, eyeAt: w.__eyeProbe.where, tip: w.__eyeProbe.tipWorst as number,
    laid: R.laidAt as null | { x: number; y: number; z: number }, framing: P.camera.framing.distance as number,
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
const resetEye = (page: Page) => page.evaluate(() => {
  const e = (window as any).__eyeProbe;
  if (e) Object.assign(e, { worst: 0, where: null, tipWorst: 0 });
});

/** steer (camera yaw toward a cabin-local point) and hold W until `until` or the timeout */
async function walkTo(page: Page, lx: number, lz: number, until: (s: S) => boolean, timeoutMs: number) {
  const end = Date.now() + timeoutMs;
  await page.keyboard.down('KeyW');
  let s = await state(page);
  while (!until(s) && Date.now() < end) {
    const t = W(lx, lz);
    await page.evaluate(([tx, tz]) => {
      const P = (window as any).__player, c = P.controller.position;
      P.camera.yaw = Math.atan2(tx - c.x, tz - c.z);
    }, [t.x, t.z]);
    await sleep(60);
    s = await state(page);
  }
  await page.keyboard.up('KeyW');
  await sleep(400);
  return state(page);
}
async function place(page: Page, lx: number, lz: number, faceLx: number, faceLz: number) {
  const a = W(lx, lz), b = W(faceLx, faceLz);
  await page.evaluate(([x, z, fx, fz]) => {
    const P = (window as any).__player;
    P.teleportTo({ x, y: 0, z }, { x: fx, y: 0, z: fz });
    P.camera.yaw = Math.atan2(fx - x, fz - z);
    P.camera.pitch = (12 * Math.PI) / 180;
  }, [a.x, a.z, b.x, b.z]);
  await sleep(700);
}
async function press(page: Page, key: string, ms = 250) {
  await page.keyboard.press(key);
  await sleep(ms);
}

/** in from the porch, a turn, back out; E opens and closes; `tag` names the screenshots */
async function roundTrip(page: Page, tag: string) {
  await place(page, -0.7, 3.55, -0.7, 0);
  const s0 = await waitFor(page, (s) => /open door/i.test(s.prompt), 2000);
  check(`${tag}: facing the shut door from the porch, "E Open door" shows`, /open door/i.test(s0.prompt) && s0.open === 0 && !s0.roomShown, { prompt: s0.prompt, open: s0.open, room: s0.roomShown });
  // the shut door blocks him
  const sb = await walkTo(page, -0.7, 0, () => false, 1800);
  check(`${tag}: the shut door blocks him`, sb.lz > 2.45 && !sb.inside, { lz: r2(sb.lz) });
  await shot(page, `${tag}-0-shut`);
  await press(page, 'KeyE', 100);
  const mid = await state(page);
  const so = await waitFor(page, (s) => s.open >= 1, 2000);
  check(`${tag}: E swings it open (smoothly, inward)`, so.open >= 1 && mid.open > 0 && mid.open < 1 && /close door/i.test(so.prompt) && so.roomShown, { mid: r2(mid.open), prompt: so.prompt });
  await shot(page, `${tag}-1-open`);
  await resetEye(page);
  const si = await walkTo(page, -0.4, 0.2, (s) => s.inside && s.lz < 0.6, 9000);
  check(`${tag}: he walks in`, si.inside && si.lz < 0.7, { lx: r2(si.lx), lz: r2(si.lz), framing: r2(si.framing) });
  await sleep(1200);
  await shot(page, `${tag}-2-inside`);
  // turn round on the spot and look back at the door
  await page.evaluate(() => {
    (window as any).__player.camera.yaw += Math.PI;
  });
  await page.keyboard.down('KeyW');
  await sleep(350);
  await page.keyboard.up('KeyW');
  await sleep(900);
  await shot(page, `${tag}-3-turned`);
  // look up at the roof and down at the floor, swing the view round
  for (const [dy, dp] of [[0.8, -0.5], [0.8, 0.9], [1.6, 0.3], [-2.5, -0.2]] as const) {
    await page.evaluate(([y, p]) => {
      const P = (window as any).__player;
      P.camera.yaw += y;
      P.camera.pitch = p;
    }, [dy, dp]);
    await sleep(500);
  }
  const so2 = await walkTo(page, -0.7, 3.7, (s) => s.lz > 3.2, 9000);
  check(`${tag}: he turns and walks out through the doorway`, so2.lz > 3.0 && !so2.inside, { lz: r2(so2.lz) });
  const eye = await state(page);
  check(`${tag}: the camera never went into the logs, roof, lintel or furniture`, eye.eye < 0.03, { worst: r2(eye.eye), at: eye.eyeAt });
  await place(page, -0.7, 3.5, -0.7, 0);
  await press(page, 'KeyE', 100);
  const sc = await waitFor(page, (s) => s.open <= 0, 2000);
  await sleep(300);
  const sc2 = await state(page);
  check(`${tag}: E closes it; the room is hidden again`, sc.open === 0 && !sc2.roomShown && /open door/i.test(sc2.prompt), { open: sc.open, room: sc2.roomShown });
  return eye;
}

async function play(page: Page) {
  const stub = await page.evaluate(() => (window as any).__pointerLockStub === true && !String(Element.prototype.requestPointerLock).includes('[native code]'));
  check('pointer lock stub installed before any click', stub);
  if (!stub) throw new Error('pointer lock stub missing: refusing to click (a real lock would capture the mouse)');
  const box = (await (await page.$('canvas'))!.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await sleep(400);

  // hands free: the rod laid in the yard, well away from the door
  await place(page, 2.4, 6.2, 2.4, 9);
  await press(page, 'KeyE', 200);
  const sl = await waitFor(page, (s) => s.phase === 'laid', 4000);
  check('the rod laid in the yard first (E away from the door is still the rod\'s)', sl.phase === 'laid', { phase: sl.phase });
  await roundTrip(page, 'a-handsfree');

  // the door in its arc: he stands in the doorway, E to close: it waits for him
  await place(page, -0.7, 3.5, -0.7, 0);
  await press(page, 'KeyE');
  await waitFor(page, (s) => s.open >= 1, 2000);
  await place(page, -0.65, 1.95, -0.65, 0.5);
  await page.evaluate(() => {
    const P = (window as any).__player;
    P.camera.yaw += Math.PI;
  });
  await sleep(300);
  await place(page, -0.95, 1.9, -0.7, 3.5);
  const sa = await state(page);
  await press(page, 'KeyE', 900);
  const sw = await state(page);
  check('standing in the door\'s arc: E to close, it waits (does not swing through him)', sw.open > 0.95 && (sw.blocked || sw.open >= 1), { prompt: sa.prompt, open: r2(sw.open), blocked: sw.blocked });
  await place(page, -0.7, 3.6, -0.7, 0);
  await sleep(1200);
  const sw2 = await state(page);
  check('out of its way, it closes', sw2.open === 0, { open: r2(sw2.open) });

  // pick the rod up again
  const laid = sl.laid!;
  await page.evaluate(([x, z]) => {
    const P = (window as any).__player;
    P.teleportTo({ x: x + 0.9, y: 0, z: z + 0.3 }, { x, y: 0, z });
  }, [laid.x, laid.z]);
  await sleep(600);
  await press(page, 'KeyE', 200);
  const sh = await waitFor(page, (s) => s.phase === 'hand', 6000);
  check('rod picked up in the yard', sh.phase === 'hand', { phase: sh.phase });

  // with the rod
  const eyeRod = await roundTrip(page, 'b-rod');
  // (info: the 2.5 m rod cannot always clear a 2.6 m room; how far its tip went into the roof or a wall's logs)
  check('carrying the rod indoors: deepest tip intrusion into the roof or a log wall (info, m)', true, { tipInto: r2(eyeRod.tip) });

  // inside: LMB does nothing; the rod goes down on the floor and comes up again
  await place(page, -0.7, 3.5, -0.7, 0);
  await press(page, 'KeyE');
  await waitFor(page, (s) => s.open >= 1, 2000);
  const sIn = await walkTo(page, 0.2, -0.3, (s) => s.inside && s.lz < 0.2, 9000);
  await sleep(600);
  await page.mouse.down({ button: 'left' });
  await sleep(900);
  const sAim = await state(page);
  await page.mouse.up({ button: 'left' });
  await sleep(600);
  const sCast = await state(page);
  check('indoors LMB does nothing: no aim, no cast, no cast hint', sAim.fish === 'idle' && sCast.fish === 'idle' && sCast.fishPrompt === '', { aim: sAim.fish, after: sCast.fish, hint: sCast.fishPrompt, inside: sIn.inside });
  await shot(page, 'c-0-rod-inside');
  // face along the room (the long way) and lay it down
  await place(page, -0.2, -0.45, 2.0, -0.45);
  await press(page, 'KeyE', 200);
  const sd = await waitFor(page, (s) => s.phase === 'laid', 4000);
  const floorOk = !!sd.laid && Math.abs(sd.laid.y - (await page.evaluate(() => (window as any).__camp.cabinY + 0.55))) < 0.12;
  check('E inside (away from the door) lays the rod on the cabin floor', sd.phase === 'laid' && floorOk, { phase: sd.phase, laid: sd.laid && [r2(sd.laid.x), r2(sd.laid.y), r2(sd.laid.z)], prompt: sd.prompt });
  await shot(page, 'c-1-rod-on-floor');
  await page.evaluate(() => {
    const P = (window as any).__player;
    P.camera.yaw += 2.4;
  });
  await page.keyboard.down('KeyW');
  await sleep(900);
  await page.keyboard.up('KeyW');
  await sleep(400);
  const back = await state(page);
  if (back.laid) {
    const t = back.laid;
    await page.evaluate(([x, z]) => {
      const P = (window as any).__player, c = P.controller.position;
      P.controller.yaw = Math.atan2(x - c.x, z - c.z);
      P.camera.yaw = P.controller.yaw;
    }, [t.x, t.z]);
  }
  await sleep(300);
  const sp = await waitFor(page, (s) => /pick up rod/i.test(s.prompt), 2000);
  await press(page, 'KeyE', 200);
  const su = await waitFor(page, (s) => s.phase === 'hand', 6000);
  check('picked up again from the cabin floor', su.phase === 'hand' && /pick up rod/i.test(sp.prompt), { phase: su.phase, prompt: sp.prompt });
  await shot(page, 'c-2-rod-picked-up');
  // a few seconds' frame times standing in the room
  await page.evaluate(() => (window as any).__RM.frameStats && (window as any).__player && 0);
  await sleep(3000);
  const fs = await page.evaluate(() => (window as any).__RM.frameStats());
  check('frame times in the room (play mode, info)', true, { med: r2(fs.med), p95: r2(fs.p95), gpuAvg: r2(fs.gpuAvg), gpuP95: r2(fs.gpuP95) });
}

async function main() {
  await mkdir(outDir, { recursive: true });
  const server = await ensureServer();
  const browser = await chromium.launch({
    channel: 'chrome', headless: false,
    args: ['--window-position=1930,30', '--window-size=1060,900', '--mute-audio', '--enable-unsafe-webgpu', '--force_high_performance_gpu'],
  });
  const consoleLines: string[] = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    page.on('console', (m) => {
      if (m.type() === 'error' || m.type() === 'warning') consoleLines.push(`[${m.type()}] ${m.text()}`);
    });
    page.on('pageerror', (e) => consoleLines.push(`[pageerror] ${e.message}`));
    await page.addInitScript(POINTER_LOCK_STUB);
    await page.goto(`${url}?debug&vol=0&fishseed=7&tod=morning`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => (window as any).__RM?.ready === true || (window as any).__RM?.error, null, { timeout: 120_000 });
    const err = await page.evaluate(() => (window as any).__RM.error);
    if (err) throw new Error(err);
    await sleep(1500);
    await play(page);
  } catch (e) {
    check('run finished without an exception', false, String((e as Error)?.stack ?? e));
  } finally {
    await browser.close();
    await server?.close();
  }
  const passed = checks.filter((c) => c.ok).length;
  console.log(`${passed}/${checks.length} passed`);
  for (const l of consoleLines) console.log(l);
  await writeFile(join(outDir, 'cabincheck.json'), JSON.stringify({ passed, total: checks.length, checks, consoleLines }, null, 2));
}
main();
