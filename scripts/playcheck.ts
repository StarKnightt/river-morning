/**
 * Part 5 play check: walks the angler through the game in play mode and records what happens.
 *
 *   pnpm tsx scripts/playcheck.ts [--out p5\r1-play] [--url http://127.0.0.1:5173/]
 *
 * Play mode with ?debug (window.__player, window.__camp). No pointer lock: requestPointerLock is stubbed before the
 * page loads and mouse look goes through the third-person camera's yaw. Real key presses (W, Shift+W, S) check the
 * input path; the longer walks use a page-side autopilot that steers the camera and holds W every frame, so a lost
 * window focus cannot drop the keys. Every frame it records the floor under him (terrain, water, walk surfaces), the
 * deepest water he stands in, and the camera's clearance to the terrain, the water, the wall boxes and his head.
 * Also audits every camp mesh against the walk blockers. Writes playcheck.json and screenshots under
 * E:\shots\river-morning\<out>; starts and stops its own dev server like scripts/shots.ts.
 */
import { chromium, type Page } from 'playwright-core';
import { extraChromeArgs, viewportOpts } from './chromeFlags';
import { createServer, type ViteDevServer } from 'vite';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import {
  CABIN, CABIN_GEO, CABIN_PROPS, CAMPFIRE, DOCK, DOCK_TIP_Z, FIRE_GEO, FIRE_PIECES, cabinX, cabinZ, fireX, fireZ, toCabinX, toCabinZ,
} from '../src/camp/layout';
import { LAYOUT } from '../src/world/Layout';

const args = process.argv.slice(2);
const arg = (name: string, def: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const url = arg('url', 'http://127.0.0.1:5173/');
const outDir = join('E:\\shots\\river-morning', arg('out', 'p5\\r1-play'));
const R = 0.28;

async function ensureServer(): Promise<ViteDevServer | null> {
  try {
    if ((await fetch(url)).ok) return null;
  } catch {
    // not running
  }
  const server = await createServer({ root: join(import.meta.dirname, '..'), logLevel: 'warn', server: { host: '127.0.0.1', port: 5173, strictPort: true } });
  await server.listen();
  return server;
}

interface State { x: number; y: number; z: number; yaw: number; speed: number; depth: number; surface: string; arrived: boolean }
const cab = (lx: number, lz: number): [number, number] => [cabinX(lx, lz), cabinZ(lx, lz)];
const r2 = (v: number) => Math.round(v * 100) / 100;

async function main() {
  await mkdir(outDir, { recursive: true });
  const server = await ensureServer();
  const browser = await chromium.launch({
    channel: 'chrome', headless: false,
    // muted: the first key press starts the game's sound (part 9); nothing may reach the speakers in a test
    args: ['--window-position=1930,30', '--window-size=1060,900', '--force_high_performance_gpu', '--mute-audio', '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding', '--disable-background-timer-throttling', '--disable-features=CalculateNativeWinOcclusion', ...extraChromeArgs()],
  });
  const report: Record<string, unknown> = { at: new Date().toISOString() };
  const consoleLines: string[] = [];
  try {
    const context = await browser.newContext(viewportOpts());
    // a granted pointer lock in this headed window would capture the real mouse of whoever is at the PC
    await context.addInitScript(() => {
      Element.prototype.requestPointerLock = function () {
        return Promise.resolve();
      } as typeof Element.prototype.requestPointerLock;
    });
    // tsx (esbuild keepNames) wraps named functions in __name(); the callbacks sent to the page need it too
    await context.addInitScript({ content: 'window.__name = (f) => f;' });
    const page = await context.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error' || m.type() === 'warning') consoleLines.push(`[${m.type()}] ${m.text()}`);
    });
    page.on('pageerror', (e) => consoleLines.push(`[pageerror] ${e.message}`));
    const t0 = Date.now();
    await page.goto(`${url}?debug&vol=0`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__RM?.ready || window.__RM?.error, null, { timeout: 120_000, polling: 250 });
    report.loadMs = Date.now() - t0;
    const err = await page.evaluate(() => window.__RM.error);
    if (err) throw new Error(err);
    await page.waitForTimeout(1500);
    await installProbe(page);
    report.audit = await audit(page);
    report.tests = await runTests(page);
  } catch (e) {
    report.error = String((e as Error)?.stack ?? e);
    console.error(e);
  } finally {
    report.consoleLines = consoleLines;
    await writeFile(join(outDir, 'playcheck.json'), JSON.stringify(report, null, 2), { encoding: 'utf-8' });
    await browser.close();
    await server?.close();
  }
}

/** The page-side autopilot and per-frame recorder (window.__pc). */
async function installProbe(page: Page) {
  await page.evaluate(() => {
    const w = window as any;
    const P = w.__player;
    if (!P) throw new Error('no window.__player (needs ?debug and the angler)');
    const st: any = {
      target: null, tol: 0.25, run: false, holdYaw: null, arrived: false,
      surfaces: [] as string[], last: '', maxDepth: 0, maxY: -1e9,
      cam: { frames: 0, inside: 0, near: 0, minTerrain: 99, minWater: 99, minHead: 99 },
      rod: { frames: 0, inBox: 0, maxInBox: 0, underGround: 0, minAboveGround: 99, maxReach: 0 },
      reset() {
        this.surfaces = []; this.last = ''; this.maxDepth = 0; this.maxY = -1e9;
        this.cam = { frames: 0, inside: 0, near: 0, minTerrain: 99, minWater: 99, minHead: 99 };
        this.rod = { frames: 0, inBox: 0, maxInBox: 0, underGround: 0, minAboveGround: 99, maxReach: 0 };
      },
    };
    w.__pc = st;
    const head = { x: 0, y: 0, z: 0 };
    const bones: Map<string, any> = P.character.bones;
    const headBone = bones.get('head') ?? bones.get('neck_01') ?? [...bones.values()].find((b) => /head/i.test(b.name)) ?? null;
    st.headBone = headBone?.name ?? null;
    const body = () => {
      const c = P.controller;
      if (st.target) {
        const dx = st.target[0] - c.position.x, dz = st.target[1] - c.position.z;
        if (Math.hypot(dx, dz) < st.tol) {
          st.target = null;
          st.arrived = true;
          c.keys.clear();
        } else {
          P.camera.yaw = Math.atan2(dx, dz);
          c.keys.add('KeyW');
          if (st.run) c.keys.add('ShiftLeft');
          else c.keys.delete('ShiftLeft');
        }
      } else if (st.holdYaw !== null) {
        P.camera.yaw = st.holdYaw;
        c.keys.add('KeyW');
      }
      const f = c.floor;
      const name = f.surface ? f.surface.name : f.depth > 0.03 ? 'water' : 'terrain';
      if (name !== st.last) {
        st.surfaces.push(`${name} @ ${c.position.x.toFixed(2)}, ${c.position.y.toFixed(2)}, ${c.position.z.toFixed(2)}`);
        st.last = name;
      }
      if (!f.surface) st.maxDepth = Math.max(st.maxDepth, f.depth);
      st.maxY = Math.max(st.maxY, c.position.y);
      const cam = P.camera.camera.position, G = P.ground;
      if (headBone) {
        const e = headBone.matrixWorld.elements;
        head.x = e[12]; head.y = e[13]; head.z = e[14];
        st.cam.minHead = Math.min(st.cam.minHead, Math.hypot(cam.x - head.x, cam.y - head.y, cam.z - head.z));
      }
      st.cam.frames++;
      if (G.inBox(cam.x, cam.z, 0.01)) st.cam.inside++;
      if (G.inBox(cam.x, cam.z, 0.15)) st.cam.near++;
      st.cam.minTerrain = Math.min(st.cam.minTerrain, cam.y - G.terrainY(cam.x, cam.z));
      st.cam.minWater = Math.min(st.cam.minWater, cam.y - G.waterY(cam.x, cam.z));
      // the rod in his hands (held ~10 cm further out since the character polish): how far its tip reaches ahead
      // of him, how deep any of it goes into a box blocker's footprint (walls, chimney, firebox: boxes count as
      // full height) and how close it comes to the ground
      const rod = P.rod, r = st.rod;
      rodPts.length = 0;
      for (const b of [rod.handle, ...rod.segments]) {
        const e = b.matrixWorld.elements;
        rodPts.push([e[12], e[13], e[14]]);
      }
      rod.getTip(tip);
      rodPts.push([tip.x, tip.y, tip.z]);
      let inside = 0, above = 99;
      for (const [x, y, z] of rodPts) {
        for (const b of G.boxes) {
          const cs = Math.cos(b.rotY), sn = Math.sin(b.rotY);
          const lx = (x - b.x) * cs - (z - b.z) * sn, lz = (x - b.x) * sn + (z - b.z) * cs;
          inside = Math.max(inside, Math.min(b.hx - Math.abs(lx), b.hz - Math.abs(lz)));
        }
        above = Math.min(above, y - G.terrainY(x, z));
      }
      r.frames++;
      if (inside > 0) r.inBox++;
      r.maxInBox = Math.max(r.maxInBox, inside);
      if (above < 0) r.underGround++;
      r.minAboveGround = Math.min(r.minAboveGround, above);
      r.maxReach = Math.max(r.maxReach, (tip.x - c.position.x) * Math.sin(c.yaw) + (tip.z - c.position.z) * Math.cos(c.yaw));
    };
    const rodPts: number[][] = [];
    const tip = new P.controller.position.constructor();
    const loop = () => {
      try {
        body();
      } catch (e) {
        st.error = String(e);
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  });
}

const state = (page: Page): Promise<State> => page.evaluate(() => {
  const w = window as any;
  const c = w.__player.controller;
  return {
    x: c.position.x, y: c.position.y, z: c.position.z, yaw: c.yaw, speed: c.groundSpeed, depth: c.floor.depth,
    surface: c.floor.surface ? c.floor.surface.name : 'terrain', arrived: w.__pc.arrived,
  };
});
const fmt = (s: State) => ({ x: r2(s.x), y: r2(s.y), z: r2(s.z), yawDeg: r2((s.yaw * 180) / Math.PI), depth: r2(s.depth), surface: s.surface });
const segment = (page: Page) => page.evaluate(() => {
  const st = (window as any).__pc;
  const c = st.cam;
  return {
    probeError: st.error ?? null, surfaces: [...st.surfaces], maxDepth: Math.round(st.maxDepth * 1000) / 1000,
    camera: { frames: c.frames, insideWallBox: c.inside, within15cmOfWall: c.near, minAboveTerrain: +c.minTerrain.toFixed(3), minAboveWater: +c.minWater.toFixed(3), minToHead: +c.minHead.toFixed(3) },
    rod: { frames: st.rod.frames, framesInBoxBlocker: st.rod.inBox, maxIntoBoxBlocker: +st.rod.maxInBox.toFixed(3), framesUnderGround: st.rod.underGround, minAboveGround: +st.rod.minAboveGround.toFixed(3), maxTipReachAhead: +st.rod.maxReach.toFixed(3) },
  };
});
const resetSeg = (page: Page) => page.evaluate(() => (window as any).__pc.reset());
const stop = (page: Page) => page.evaluate(() => {
  const w = window as any;
  w.__pc.target = null;
  w.__pc.holdYaw = null;
  w.__player.controller.keys.clear();
});
const teleport = (page: Page, x: number, z: number, lookX: number, lookZ: number) =>
  page.evaluate(([x, z, lx, lz]) => (window as any).__player.teleportTo({ x, y: 0, z }, { x: lx, y: 0, z: lz }), [x, z, lookX, lookZ]);
const shot = (page: Page, name: string) => page.screenshot({ path: join(outDir, `${name}.png`), type: 'png' });

/** Steer to (x, z): arrived within tol, stuck (under 3 cm progress in 2.5 s) or timed out. */
async function walkTo(page: Page, x: number, z: number, o: { tol?: number; run?: boolean; timeout?: number } = {}) {
  await page.evaluate(([x, z, tol, run]) => {
    const st = (window as any).__pc;
    Object.assign(st, { target: [x, z], tol, run: !!run, arrived: false, holdYaw: null });
  }, [x, z, o.tol ?? 0.25, o.run ? 1 : 0]);
  const t0 = Date.now();
  let ref = await state(page), refT = Date.now();
  for (;;) {
    await page.waitForTimeout(200);
    const s = await state(page);
    if (s.arrived) return { result: 'arrived', at: fmt(s), seconds: (Date.now() - t0) / 1000 };
    if (Math.hypot(s.x - ref.x, s.z - ref.z) > 0.03) {
      ref = s;
      refT = Date.now();
    } else if (Date.now() - refT > 2500) {
      await stop(page);
      return { result: 'stopped', at: fmt(s), seconds: (Date.now() - t0) / 1000 };
    }
    if (Date.now() - t0 > (o.timeout ?? 45_000)) {
      await stop(page);
      return { result: 'timeout', at: fmt(s), seconds: (Date.now() - t0) / 1000 };
    }
  }
}

/** Hold W along a fixed view yaw until he stops (or the timeout). */
async function holdForward(page: Page, yaw: number, timeout = 25_000) {
  await page.evaluate((y) => Object.assign((window as any).__pc, { holdYaw: y, target: null }), yaw);
  const t0 = Date.now();
  let ref = await state(page), refT = Date.now();
  for (;;) {
    await page.waitForTimeout(200);
    const s = await state(page);
    if (Math.hypot(s.x - ref.x, s.z - ref.z) > 0.02) {
      ref = s;
      refT = Date.now();
    } else if (Date.now() - refT > 2000) break;
    if (Date.now() - t0 > timeout) break;
  }
  await stop(page);
  return fmt(await state(page));
}

async function runTests(page: Page) {
  const T: Record<string, unknown> = {};
  const P = (fn: string) => page.evaluate(fn);
  const yawToward = (fx: number, fz: number, tx: number, tz: number) => Math.atan2(tx - fx, tz - fz);

  // 0. spawn
  await P('window.__player.resetToSpawn()');
  await page.waitForTimeout(800);
  const spawn = await state(page);
  const spawnInfo = await page.evaluate(() => {
    const w = window as any;
    const P = w.__player, cam = P.camera.camera.position;
    return { spawnYawDeg: (P.spawn.yaw * 180) / Math.PI, camera: [cam.x, cam.y, cam.z].map((v: number) => Math.round(v * 100) / 100), blockers: P.ground.blockerCount, surfaces: P.ground.surfaces.map((s: any) => s.name) };
  });
  T.spawn = { ...fmt(spawn), ...spawnInfo, layoutSpawn: LAYOUT.spawn };
  await shot(page, '00_spawn');

  // 1. real key presses along the bank (view east): W 2 s, Shift+W 1.5 s, S 1.5 s (he turns round and walks back)
  await page.evaluate(() => {
    (window as any).__player.camera.yaw = Math.PI / 2;
  });
  await page.waitForTimeout(300);
  const k0 = await state(page);
  await page.keyboard.down('KeyW');
  await page.waitForTimeout(2000);
  const kW = await state(page);
  await page.keyboard.down('Shift');
  await page.waitForTimeout(1500);
  const kRun = await state(page);
  await page.keyboard.up('Shift');
  await page.keyboard.up('KeyW');
  await page.waitForTimeout(600);
  const kStop = await state(page);
  await shot(page, '01_walked_east');
  await page.keyboard.down('KeyS');
  await page.waitForTimeout(1500);
  const kS = await state(page);
  await page.keyboard.up('KeyS');
  await page.waitForTimeout(600);
  T.keys = {
    start: fmt(k0),
    afterW2s: { ...fmt(kW), dx: r2(kW.x - k0.x), dz: r2(kW.z - k0.z), speed: r2(kW.speed) },
    afterShiftW1500ms: { ...fmt(kRun), speed: r2(kRun.speed) },
    released: { ...fmt(kStop), speed: r2(kStop.speed) },
    afterS1500ms: { ...fmt(kS), dx: r2(kS.x - kStop.x), facingDeg: r2((kS.yaw * 180) / Math.PI) },
  };

  // 2. wading: from spawn straight at the river until he stops
  await P('window.__player.resetToSpawn()');
  await resetSeg(page);
  const sy = spawn.yaw;
  T.wade = { stoppedAt: await holdForward(page, sy), ...(await segment(page)) };
  await shot(page, '02_wade_limit');
  // and with Shift along the limit (he must not go deeper)
  await page.evaluate(() => (window as any).__player.controller.keys.add('ShiftLeft'));
  await resetSeg(page);
  T.wadeAlong = { stoppedAt: await holdForward(page, sy + 0.9, 6000), ...(await segment(page)) };
  await stop(page);

  // 3. bank -> dock step -> deck -> tip, then try to walk off the tip and off the side
  await P('window.__player.resetToSpawn()');
  await resetSeg(page);
  const dock: Record<string, unknown> = {};
  dock.toBar = await walkTo(page, DOCK.x + 0.05, DOCK.zShore + 1.3, { run: true });
  dock.toStep = await walkTo(page, DOCK.x + 0.05, DOCK.zShore + 0.25, { tol: 0.2 });
  dock.toMid = await walkTo(page, DOCK.x, DOCK.zShore - 3.2, { tol: 0.2 });
  await shot(page, '03a_deck_mid');
  dock.toTip = await walkTo(page, DOCK.x, DOCK_TIP_Z - 0.5, { tol: 0.1 });
  dock.offTipNorth = await holdForward(page, Math.PI, 3000);
  await shot(page, '03b_deck_tip');
  dock.offSideWest = await holdForward(page, -Math.PI / 2, 3000);
  Object.assign(dock, await segment(page), { deckTipZ: DOCK_TIP_Z, camp: await P('({ deckY: window.__camp.deckY, dockStepY: window.__camp.dockStepY })') });
  T.dock = dock;

  // 4. back along the deck, off the step, up the bank to the yard and up the porch steps
  await resetSeg(page);
  const porch: Record<string, unknown> = {};
  porch.backToBar = await walkTo(page, DOCK.x + 0.05, DOCK.zShore + 1.0, { run: true });
  porch.upBank = await walkTo(page, 11.2, 12.5, { run: true });
  const yard = cab(-0.6, 7.0);
  porch.toYard = await walkTo(page, yard[0], yard[1], { run: true });
  const st = CABIN_GEO.step;
  const sx = (st.x0 + st.x1) / 2, sz = (st.z0 + st.z1) / 2;
  const low = cab(sx + 0.06, sz + 0.62), step = cab(sx, sz), deck = cab(-0.7, 3.3);
  porch.toLowStep = await walkTo(page, low[0], low[1], { tol: 0.2 });
  porch.toStep = await walkTo(page, step[0], step[1], { tol: 0.2 });
  porch.toPorch = await walkTo(page, deck[0], deck[1], { tol: 0.25 });
  await shot(page, '04_porch');
  // along the porch toward the far (west) end and into the wall behind
  const pe = cab(-1.9, 3.3);
  porch.alongPorch = await walkTo(page, pe[0], pe[1], { tol: 0.25 });
  Object.assign(porch, await segment(page), { camp: await P('({ porchY: window.__camp.porchY, stepY: window.__camp.stepY, stepLowY: window.__camp.stepLowY })') });
  T.porch = porch;

  // 5. blockers: into a side wall, the back wall, the chimney and the fire ring
  const blocks: Record<string, unknown> = {};
  const into = async (name: string, from: [number, number], to: [number, number]) => {
    await teleport(page, from[0], from[1], to[0], to[1]);
    await page.waitForTimeout(400);
    await resetSeg(page);
    const r = await walkTo(page, to[0], to[1], { tol: 0.05, timeout: 15_000 });
    const s = await state(page);
    blocks[name] = { ...r, local: { lx: r2(toCabinX(s.x, s.z)), lz: r2(toCabinZ(s.x, s.z)) }, ...(await segment(page)) };
    return s;
  };
  const sideFrom = cab(-4.9, 0.4), sideTo = cab(0, 0.4);
  const side = await into('sideWall', sideFrom, sideTo);
  blocks.sideWallGap = r2(-toCabinX(side.x, side.z) - CABIN_GEO.walls.hx - R);
  await shot(page, '05a_side_wall');
  // camera with its back to that wall: the boom points into it, then the shoulder offset points into it
  const yc = CABIN.yaw;
  const camTest = async (name: string, yaw: number) => {
    await resetSeg(page);
    await page.evaluate((y) => {
      (window as any).__player.camera.yaw = y;
    }, yaw);
    await page.waitForTimeout(1200);
    const c = await page.evaluate(() => {
      const p = (window as any).__player.camera.camera.position;
      return { x: p.x, y: p.y, z: p.z };
    });
    blocks[name] = { cameraLocal: { lx: r2(toCabinX(c.x, c.z)), lz: r2(toCabinZ(c.x, c.z)), y: r2(c.y) }, ...(await segment(page)) };
    await shot(page, name);
  };
  await camTest('05b_camera_boom_into_wall', Math.atan2(-Math.cos(yc), Math.sin(yc)));
  await camTest('05c_camera_shoulder_into_wall', yc + Math.PI);
  const back = await into('backWall', cab(0.3, -4.7), cab(0.3, 0));
  blocks.backWallGap = r2(-toCabinZ(back.x, back.z) - CABIN_GEO.walls.hz - R);
  const chim = await into('chimney', cab(5.3, 0.2), cab(0, 0.2));
  blocks.chimneyGapToBlocker = r2(toCabinX(chim.x, chim.z) - (CABIN_GEO.chimney.x + CABIN_GEO.chimney.hx) - R);
  blocks.chimneyGapToFirebox = r2(toCabinX(chim.x, chim.z) - 3.98 - 0.07 - R);
  await shot(page, '05d_chimney');
  const fire = await into('fireRing', [CAMPFIRE.x + 2.3, CAMPFIRE.z + 0.05], [CAMPFIRE.x, CAMPFIRE.z]);
  blocks.fireRingDistance = r2(Math.hypot(fire.x - CAMPFIRE.x, fire.z - CAMPFIRE.z));
  blocks.fireRingExpected = r2(FIRE_GEO.ringR + R);
  await shot(page, '05e_fire_ring');
  // edges too high to climb: the dock's side from the shallows by its shore end, the porch front beside the steps
  const dz = DOCK.zShore - 0.6;
  const dockSide = await into('dockSide', [DOCK.x - 1.6, dz], [DOCK.x, dz]);
  blocks.dockSideBootsGap = r2(DOCK.x - DOCK.halfWidth - dockSide.x - 0.24);
  await shot(page, '05f_dock_side');
  const pf = await into('porchFront', cab(0.45, 6.2), cab(0.45, 3.0));
  blocks.porchFrontBootsGap = r2(toCabinZ(pf.x, pf.z) - CABIN_GEO.porch.z1 - 0.24);
  await shot(page, '05g_porch_front');
  T.blockers = blocks;

  // 6. jumps to the spawn and the dock through real key presses: the angler moves, not just the camera
  await teleport(page, CAMPFIRE.x + 2.3, CAMPFIRE.z, CAMPFIRE.x, CAMPFIRE.z);
  await page.waitForTimeout(300);
  await page.keyboard.press('Digit0');
  await page.waitForTimeout(800);
  const k0b = await state(page);
  await page.keyboard.press('Digit4');
  await page.waitForTimeout(800);
  const k4 = await state(page);
  await shot(page, '06_jump_dock');
  T.jump = { spawn: fmt(k0b), dock: fmt(k4), yawTowardSpawnView: r2((yawToward(LAYOUT.spawn.x, LAYOUT.spawn.z, LAYOUT.spawn.x - 6, LAYOUT.spawn.z - 10) * 180) / Math.PI) };

  // 7. (polish round) the camp paths and the meadow: the forest polish adds 1,347 trees (each a walk blocker in
  // Player.ts) and the camp polish moved and grew the camp's blockers
  T.paths = await walkPaths(page);
  T.meadow = await walkMeadow(page);
  return T;
}

type P2 = { x: number; z: number };

/** groundMask.ts's path curve (a cubic Bezier bowed to either side), 21 points */
function curve(a: P2, b: P2, bow: number, bow2: number): P2[] {
  const dx = b.x - a.x, dz = b.z - a.z, l = Math.hypot(dx, dz), px = -dz / l, pz = dx / l;
  const c1 = { x: a.x + dx / 3 + px * bow, z: a.z + dz / 3 + pz * bow };
  const c2 = { x: a.x + (2 * dx) / 3 + px * bow2, z: a.z + (2 * dz) / 3 + pz * bow2 };
  const pts: P2[] = [];
  for (let i = 0; i <= 20; i++) {
    const t = i / 20, u = 1 - t, k0 = u * u * u, k1 = 3 * u * u * t, k2 = 3 * u * t * t, k3 = t * t * t;
    pts.push({ x: k0 * a.x + k1 * c1.x + k2 * c2.x + k3 * b.x, z: k0 * a.z + k1 * c1.z + k2 * c2.z + k3 * b.z });
  }
  return pts;
}

/** the seven worn camp paths, rebuilt from groundMask.ts's anchors (the module keeps them private) */
function campPaths(): { name: string; pts: P2[] }[] {
  const st = CABIN_GEO.step, sx = (st.x0 + st.x1) / 2;
  const door = { x: cabinX(sx, st.z1 + 0.9), z: cabinZ(sx, st.z1 + 0.9) };
  const dock = { x: DOCK.x, z: DOCK.zShore + 0.7 };
  const fire = { x: CAMPFIRE.x, z: CAMPFIRE.z };
  const fw = FIRE_PIECES.firewood, wood = { x: fireX(fw.x, fw.z), z: fireZ(fw.x, fw.z) };
  const ws = CABIN_PROPS.woodStack, cb = CABIN_PROPS.choppingBlock;
  const stack = { x: cabinX(ws.x, ws.z + 0.55), z: cabinZ(ws.x, ws.z + 0.55) };
  const block = { x: cabinX(cb.x, cb.z), z: cabinZ(cb.x, cb.z) };
  const p3a = { x: LAYOUT.cabin.x - 1.5, z: LAYOUT.cabin.z - 4.0 };
  const p3b = { x: LAYOUT.dock.x, z: LAYOUT.dock.zBase + 0.6 };
  const def: [string, P2, P2, number, number][] = [
    ['door-fire', door, fire, -1.6, 0.5], ['door-dock', door, dock, 1.7, -1.3], ['fire-dock', fire, p3b, 0.55, -0.35],
    ['fire-woodpile', fire, p3a, -0.5, 0.3], ['block-woodpile', block, stack, 0.15, -0.1], ['door-block', door, block, -0.25, 0.15],
    ['fire-firewood', fire, wood, 0.15, -0.1],
  ];
  return def.map(([name, a, b, bow, bow2]) => ({ name, pts: curve(a, b, bow, bow2) }));
}

/** points every `step` metres along a polyline */
function resample(pts: P2[], step: number): [number, number][] {
  const out: [number, number][] = [[pts[0].x, pts[0].z]];
  let carry = 0;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i], l = Math.hypot(b.x - a.x, b.z - a.z);
    let d = step - carry;
    while (d <= l) {
      out.push([a.x + ((b.x - a.x) * d) / l, a.z + ((b.z - a.z) * d) / l]);
      d += step;
    }
    carry = l - (d - step);
  }
  return out;
}

/** which of the points a walker of radius R fits on (the controller's own blocker test, Ground.resolve) */
const fitsAt = (page: Page, pts: [number, number][]) => page.evaluate(([pts, R]) => {
  const G = (window as any).__player.ground, q = { x: 0, z: 0 };
  return (pts as [number, number][]).map(([x, z]) => {
    q.x = x; q.z = z;
    G.resolve(q, R as number);
    return Math.hypot(q.x - x, q.z - z) < 1e-3;
  });
}, [pts, R] as const);

/** Walk each camp path through its free points; the points the walker does not fit on away from its ends. */
async function walkPaths(page: Page) {
  const out: Record<string, unknown> = {};
  for (const p of campPaths()) {
    const s = resample(p.pts, 0.25), fits = await fitsAt(page, s);
    const a = p.pts[0], b = p.pts[p.pts.length - 1];
    // the ends sit at the fire ring, the woodpile, the chopping block or the firewood spill, whose blockers are meant
    const atEnd = (x: number, z: number) => Math.min(Math.hypot(x - a.x, z - a.z), Math.hypot(x - b.x, z - b.z)) < 1.4;
    const blockedMid = s.filter(([x, z], i) => !fits[i] && !atEnd(x, z)).map(([x, z]) => [r2(x), r2(z)]);
    const free = s.map((_, i) => (fits[i] ? i : -1)).filter((i) => i >= 0);
    if (free.length < 2) {
      out[p.name] = { length: r2((s.length - 1) * 0.25), blockedMid, walk: 'no free points' };
      continue;
    }
    const first = free[0], last = free[free.length - 1];
    const way = free.filter((i) => i > first && (i - first) % 6 === 0).concat(last);
    await teleport(page, s[first][0], s[first][1], s[way[0]][0], s[way[0]][1]);
    await page.waitForTimeout(400);
    const legs: unknown[] = [];
    let ok = true;
    for (const i of way) {
      const r = await walkTo(page, s[i][0], s[i][1], { tol: 0.35, timeout: 15_000 });
      if (r.result !== 'arrived') {
        ok = false;
        legs.push({ to: [r2(s[i][0]), r2(s[i][1])], ...r });
      }
    }
    out[p.name] = { length: r2((s.length - 1) * 0.25), walked: `${r2((last - first) * 0.25)} m`, ok, blockedMid, failedLegs: legs };
  }
  return out;
}

/** The meadow: the ground reachable from spawn on a 0.25 m grid, then a walk round it with a sidestep when he stops. */
async function walkMeadow(page: Page) {
  const grid = await page.evaluate(([R, sx, sz]) => {
    const G = (window as any).__player.ground, q = { x: 0, z: 0 };
    const x0 = -45, x1 = 40, z0 = -6, z1 = 42, s = 0.25;
    const nx = Math.round((x1 - x0) / s) + 1, nz = Math.round((z1 - z0) / s) + 1;
    const ok = new Uint8Array(nx * nz);
    for (let j = 0; j < nz; j++) {
      for (let i = 0; i < nx; i++) {
        const x = x0 + i * s, z = z0 + j * s;
        q.x = x; q.z = z;
        G.resolve(q, R);
        if (Math.hypot(q.x - x, q.z - z) > 1e-3) continue;
        const t = G.terrainY(x, z);
        if (G.waterY(x, z) - t > 0.4 && !G.surfaceAt(x, z, t)) continue;
        ok[j * nx + i] = 1;
      }
    }
    // flood fill from spawn (4-neighbours): what he can walk to
    const seen = new Uint8Array(nx * nz), stack: number[] = [];
    const start = Math.round((sz - z0) / s) * nx + Math.round((sx - x0) / s);
    if (ok[start]) {
      seen[start] = 1;
      stack.push(start);
    }
    while (stack.length) {
      const k = stack.pop()!, i = k % nx, j = (k - i) / nx;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ii = i + di, jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= nx || jj >= nz) continue;
        const kk = jj * nx + ii;
        if (ok[kk] && !seen[kk]) {
          seen[kk] = 1;
          stack.push(kk);
        }
      }
    }
    // walkable ground he cannot reach, as connected pockets (size and centre)
    const lab = new Uint8Array(nx * nz), pockets: { m2: number; x: number; z: number }[] = [];
    let walkable = 0, reached = 0;
    for (let k = 0; k < nx * nz; k++) {
      walkable += ok[k];
      reached += seen[k];
      if (!ok[k] || seen[k] || lab[k]) continue;
      let n = 0, cx = 0, cz = 0;
      const st = [k];
      lab[k] = 1;
      while (st.length) {
        const c = st.pop()!, i = c % nx, j = (c - i) / nx;
        n++; cx += x0 + i * s; cz += z0 + j * s;
        for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const ii = i + di, jj = j + dj, kk = jj * nx + ii;
          if (ii >= 0 && jj >= 0 && ii < nx && jj < nz && ok[kk] && !seen[kk] && !lab[kk]) {
            lab[kk] = 1;
            st.push(kk);
          }
        }
      }
      pockets.push({ m2: n * s * s, x: Math.round((cx / n) * 10) / 10, z: Math.round((cz / n) * 10) / 10 });
    }
    pockets.sort((a, b) => b.m2 - a.m2);
    // tour targets: two arcs south of the spawn (the river lies north), each snapped to the nearest reachable cell
    const targets: [number, number][] = [];
    for (const [rad, dir] of [[10, 1], [18, -1]] as const) {
      for (let a = -90; a <= 90; a += 30) {
        const ang = ((dir > 0 ? a : -a) * Math.PI) / 180;
        const tx = sx + Math.sin(ang) * rad, tz = sz + Math.cos(ang) * rad;
        let best = -1, bd = Infinity;
        const ci = Math.round((tx - x0) / s), cj = Math.round((tz - z0) / s);
        for (let dj = -12; dj <= 12; dj++) {
          for (let di = -12; di <= 12; di++) {
            const ii = ci + di, jj = cj + dj, kk = jj * nx + ii;
            if (ii < 0 || jj < 0 || ii >= nx || jj >= nz || !seen[kk]) continue;
            const d = di * di + dj * dj;
            if (d < bd) {
              bd = d;
              best = kk;
            }
          }
        }
        if (best >= 0) targets.push([x0 + (best % nx) * s, z0 + Math.floor(best / nx) * s]);
      }
    }
    return { walkableM2: walkable * s * s, reachedM2: reached * s * s, pockets: pockets.slice(0, 12), targets };
  }, [R, LAYOUT.spawn.x, LAYOUT.spawn.z] as const);

  await P0(page);
  const legs: unknown[] = [];
  let stops = 0, failed = 0;
  for (const [tx, tz] of [...grid.targets, [LAYOUT.spawn.x, LAYOUT.spawn.z] as [number, number]]) {
    let r = await walkTo(page, tx, tz, { tol: 0.4, timeout: 25_000 });
    const tries: string[] = [];
    // a trunk in the straight line stops the autopilot: sidestep 2 m left, then right, and go on
    for (const side of [1, -1]) {
      if (r.result === 'arrived') break;
      stops++;
      const s = await state(page), dx = tx - s.x, dz = tz - s.z, l = Math.hypot(dx, dz) || 1;
      await walkTo(page, s.x - (dz / l) * 2 * side, s.z + (dx / l) * 2 * side, { tol: 0.4, timeout: 8000 });
      r = await walkTo(page, tx, tz, { tol: 0.4, timeout: 25_000 });
      tries.push(side > 0 ? 'left' : 'right');
    }
    if (r.result !== 'arrived') failed++;
    legs.push({ to: [r2(tx), r2(tz)], result: r.result, seconds: r.seconds, detours: tries });
  }
  await shot(page, '07_meadow');
  return { ...grid, targets: undefined, legs, stops, failed };
}

async function P0(page: Page) {
  await page.evaluate(() => (window as any).__player.resetToSpawn());
  await page.waitForTimeout(600);
}

/** Camp geometry at body height (0.12-1.8 m over the floor) that lies outside every camp blocker. */
async function audit(page: Page) {
  return page.evaluate(() => {
    const w = window as any;
    const camp = w.__camp, G = w.__player.ground;
    if (!camp) return { error: 'no window.__camp' };
    const circles: Array<{ x: number; z: number; r: number }> = camp.blockers.circles;
    // the player's boxes: the camp's three plus anything main.ts adds (the round-2 firebox)
    const boxes: Array<{ x: number; z: number; hx: number; hz: number; rotY: number }> = G.boxes;
    const local = (b: { x: number; z: number; rotY: number }, x: number, z: number) => {
      const cs = Math.cos(b.rotY), sn = Math.sin(b.rotY);
      return [(x - b.x) * cs - (z - b.z) * sn, (x - b.x) * sn + (z - b.z) * cs];
    };
    const outside = (x: number, z: number) => {
      let best = Infinity;
      for (const c of circles) best = Math.min(best, Math.hypot(x - c.x, z - c.z) - c.r);
      for (const b of boxes) {
        const [lx, lz] = local(b, x, z);
        best = Math.min(best, Math.hypot(Math.max(0, Math.abs(lx) - b.hx), Math.max(0, Math.abs(lz) - b.hz)));
      }
      return best;
    };
    type Cell = { n: number; d: number; x: number; z: number; h0: number; h1: number; edge: boolean; names: Set<string> };
    const cells = new Map<string, Cell>();
    const walls = boxes[0];
    const eaves = { maxOut: 0, minH: 99, n: 0 };
    camp.group.updateMatrixWorld(true);
    let verts = 0;
    camp.group.traverse((o: any) => {
      if (!o.isMesh || !o.geometry?.attributes?.position) return;
      // the shadow-only copies (gfx/shadowProxy.ts) repeat the meshes already counted
      if (/-shadow-\d$/.test(o.name ?? '')) return;
      const mat = Array.isArray(o.material) ? o.material[0] : o.material;
      if (mat?.transparent) return;
      const pos = o.geometry.attributes.position;
      const inst = o.isInstancedMesh ? o.count : 1;
      for (let k = 0; k < inst; k++) {
        let e = o.matrixWorld.elements;
        if (o.isInstancedMesh) {
          const M = o.matrixWorld.clone(), I = o.matrixWorld.clone();
          o.getMatrixAt(k, I);
          e = M.multiply(I).elements;
        }
        for (let i = 0; i < pos.count; i++) {
          const x0 = pos.getX(i), y0 = pos.getY(i), z0 = pos.getZ(i);
          const x = e[0] * x0 + e[4] * y0 + e[8] * z0 + e[12];
          const y = e[1] * x0 + e[5] * y0 + e[9] * z0 + e[13];
          const z = e[2] * x0 + e[6] * y0 + e[10] * z0 + e[14];
          verts++;
          const t = G.terrainY(x, z);
          const s = G.surfaceAt(x, z, t);
          const h = y - (s ? s.y : t);
          if (walls && /cabin/.test(o.name ?? '') && h > 1.8 && h < 3.2) {
            const [lx, lz] = local(walls, x, z);
            const out = Math.max(Math.abs(lx) - walls.hx, Math.abs(lz) - walls.hz);
            if (out > 0.05) {
              eaves.n++;
              eaves.maxOut = Math.max(eaves.maxOut, out);
              eaves.minH = Math.min(eaves.minH, h);
            }
          }
          if (h < 0.12 || h > 1.8) continue;
          const d = outside(x, z);
          if (d < 0.05) continue;
          const key = `${Math.round(x * 2)},${Math.round(z * 2)}`;
          let c = cells.get(key);
          if (!c) cells.set(key, (c = { n: 0, d: 0, x, z, h0: h, h1: h, edge: !s && G.surfaceAt(x, z, t, 0.4) !== null, names: new Set() }));
          c.n++;
          if (d > c.d) Object.assign(c, { d, x, z });
          c.h0 = Math.min(c.h0, h);
          c.h1 = Math.max(c.h1, h);
          c.names.add(o.name || o.parent?.name || '?');
        }
      }
    });
    const list = [...cells.values()].sort((a, b) => b.d - a.d);
    const row = (c: Cell) => ({ x: +c.x.toFixed(2), z: +c.z.toFixed(2), outside: +c.d.toFixed(2), n: c.n, h: `${c.h0.toFixed(2)}-${c.h1.toFixed(2)}`, names: [...c.names].join('|') });
    return {
      verts, cells: list.length, blockers: { circles: circles.length, boxes: boxes.length },
      cabinEavesOutsideWalls: { vertices: eaves.n, maxBeyondWallBox: +eaves.maxOut.toFixed(2), lowestOverFloor: +eaves.minH.toFixed(2) },
      // geometry at body height away from any walk surface: what a walker could push into
      obstacles: list.filter((c) => !c.edge).map(row),
      // beside the deck, the porch and the step stones (the step-height rule and the boots' edge test handle these)
      surfaceEdges: list.filter((c) => c.edge).slice(0, 25).map(row),
    };
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
