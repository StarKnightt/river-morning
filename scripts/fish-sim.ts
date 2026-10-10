/**
 * Headless check of the trout behaviour (no browser, no GPU): `pnpm exec tsx scripts/fish-sim.ts`.
 * Bakes the same river field the game does (from the analytic terrain instead of the render grid), then runs the
 * school through holding and cruising, a splash, the player walking up to the edge, a lure, a bite, a fight, a
 * landing and a release. Prints what happened and exits non-zero if a check fails.
 */
import type { WaterAt } from '../src/fish/FishHabitat';
import type { BiteEvent, Fish } from '../src/fish/FishSim';

// the game modules read URL switches at import time (core/params.ts): imported after this, not hoisted above it
(globalThis as unknown as { location: { search: string } }).location ??= { search: '' };
const { heightAt } = await import('../src/world/Terrain');
const { placeRocks } = await import('../src/water/Rocks');
const { Habitat } = await import('../src/fish/FishHabitat');
const { FishSim } = await import('../src/fish/FishSim');
const { CORE } = await import('../src/world/World');
const { prepareRiverWater } = await import('../src/water/RiverWater');

const t0 = Date.now();
const rocks = placeRocks(heightAt);
const water = prepareRiverWater({ groundY: heightAt, core: CORE, obstacles: rocks.map((r) => r.obstacle) });
const habitat = new Habitat(water.field, rocks.map((r) => ({ x: r.x, z: r.z, r: r.obstacle.r, topY: r.obstacle.topY })));
const sim = new FishSim(habitat, { takeChance: 10 });
console.log(`field + school in ${Date.now() - t0} ms, ${sim.fish.length} trout`);

let failures = 0;
const check = (ok: boolean, what: string) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
  if (!ok) failures++;
};
const w: WaterAt = { waterY: 0, bedY: 0, depth: 0, flowX: 0, flowZ: 0, speed: 0 };
const inWater = () => sim.fish.filter((f) => f.visible && f.state !== 'held').every((f) => {
  habitat.at(f.pos.x, f.pos.z, w);
  return w.depth > 0.08 && f.pos.y > w.bedY && (f.pos.y < w.waterY + 0.02 || f.state === 'jump' || f.state === 'hooked');
});

sim.presimulate(30);
console.log(sim.summary().join('\n'));
check(sim.fish.length >= 30 && sim.fish.length <= 45, `30-45 trout (${sim.fish.length})`);
const kinds = (k: string) => sim.fish.filter((f) => f.home.kind === k).map((f) => f.sp.id);
console.log('riffle', kinds('riffle').join(','), '| head', kinds('head').join(','), '| pool', kinds('pool').join(','), '| tail', kinds('tail').join(','));
check(kinds('riffle').every((s) => s === 'cutthroat' || s === 'brook'), 'cutthroat and brook in the riffle');
check(kinds('pool').every((s) => s === 'brown' || s === 'rainbow'), 'brown and rainbow in the pool');
check(inWater(), 'every trout in the water after 30 s');

// holders face upstream
const holders = sim.fish.filter((f) => f.state === 'hold');
let facing = 0;
for (const f of holders) {
  habitat.at(f.pos.x, f.pos.z, w);
  const up = Math.atan2(-w.flowX, -w.flowZ);
  if (Math.abs(Math.atan2(Math.sin(f.yaw - up), Math.cos(f.yaw - up))) < 0.6) facing++;
}
check(holders.length > 0 && facing >= holders.length * 0.8, `holders face upstream (${facing}/${holders.length})`);

// surface life over three minutes
let rises = 0, jumps = 0;
for (let s = 0; s < 180 * 30; s++) {
  sim.update(1 / 30);
  for (const e of sim.outbox) {
    // (the curtain of drops off a leaping trout's tail is not a splash of its own)
    if (e.kind === 'trail') continue;
    if (e.strength < 0.3) rises++;
    else jumps++;
  }
  sim.outbox.length = 0;
}
console.log(`3 min: ${rises} rises, ${jumps} jump splashes (exit + entry)`);
check(rises >= 6 && rises <= 40, 'rises every 6-20 s');
check(inWater(), 'every trout in the water after 3 minutes');

// a hard splash in the pool scatters the trout around it, and they drift back
const pool = sim.fish.filter((f) => f.home.kind === 'pool');
const p0 = pool[0];
sim.splash(p0.pos.x, p0.pos.z, 1.0);
const fled = sim.fish.filter((f) => f.state === 'flee').length;
check(fled >= 1, `splash: ${fled} trout bolt`);
for (let s = 0; s < 4 * 30; s++) sim.update(1 / 30);
check(p0.pos.distanceTo(p0.refugeP) < 9, 'they head for cover');
for (let s = 0; s < 120 * 30; s++) sim.update(1 / 30);
const back = sim.fish.filter((f) => f.state !== 'flee' && f.state !== 'hide').length;
check(back >= sim.fish.length - 2, `two minutes later ${back}/${sim.fish.length} are back to normal`);
sim.outbox.length = 0;

// the player walks up to the edge by the riffle
const rf = sim.fish.find((f) => f.home.kind === 'riffle')!;
habitat.at(rf.pos.x, rf.pos.z, w);
for (let s = 0; s < 3 * 30; s++) {
  const k = s / 90;
  sim.setPlayer({ x: rf.pos.x + 1.2, y: w.waterY + 1.7, z: rf.pos.z + 6 - 4.4 * k }, 1 / 30);
  sim.update(1 / 30);
}
check(rf.state === 'flee' || rf.state === 'hide', `walking up to the riffle edge spooks the trout there (${rf.state})`);
sim.setPlayer(null);
for (let s = 0; s < 90 * 30; s++) sim.update(1 / 30);

// a lure in front of a pool trout: inspect, take, bite
let inspected = false;
const lureFor = (target: Fish | undefined, seconds = 40): BiteEvent | null => {
  if (!target) return null;
  let got: BiteEvent | null = null;
  sim.onBite = (e) => (got = e);
  const lure = { x: target.pos.x + Math.sin(target.yaw) * 1.2, y: target.pos.y + 0.1, z: target.pos.z + Math.cos(target.yaw) * 1.2, kind: 'lure' as const, speed: 0.25 };
  for (let s = 0; s < seconds * 30 && !got; s++) {
    lure.x += 0.004; // a slow retrieve
    sim.setLure(lure);
    sim.update(1 / 30);
    if (sim.fish.some((f) => f.state === 'inspect')) inspected = true;
  }
  return got;
};
const calmPoolTrout = () => sim.fish.find((f) => f.home.kind === 'pool' && f.calm);
const bite = lureFor(calmPoolTrout());
check(inspected, 'a trout comes to inspect the lure');
check(!!bite, bite ? `bite: ${bite.name} ${(bite.lengthM * 100).toFixed(0)} cm ${bite.weightKg.toFixed(2)} kg, window ${bite.window.toFixed(2)} s` : 'no bite in 40 s');
if (bite) {
  const id = bite.fishId;
  check(sim.hook(id), 'hook set');
  // a simple angler: rod tip on the bank, line tension rises with distance, reeling 0.4 m/s
  const tip = { x: 4.6, y: 2.0, z: 6.6 };
  let lineLen = sim.fish[id].pos.distanceTo({ x: tip.x, y: tip.y, z: tip.z } as never) + 0.5;
  let landable = false, maxPull = 0, tMax = 0;
  for (let s = 0; s < 240 * 30; s++) {
    const st = sim.fightState(id);
    if (!st) break;
    const d = st.mouth.distanceTo({ x: tip.x, y: tip.y, z: tip.z } as never);
    const stretch = d - lineLen;
    const tension = Math.max(0, Math.min(st.maxForce * 0.8, stretch * 25 + 1.5));
    if (tension < st.maxForce * 0.6) lineLen = Math.max(1.2, lineLen - 0.4 / 30); // reel when it is not running hard
    else lineLen += 0.3 / 30; // drag gives line
    // once it tires, the angler walks along the bank toward it
    if (st.phase === 'tired') tip.x += Math.sign(st.mouth.x - tip.x) * Math.min(0.8 / 30, Math.abs(st.mouth.x - tip.x));
    sim.setLine(id, { tip, tension });
    sim.update(1 / 30);
    maxPull = Math.max(maxPull, st.pullForce);
    if (s % 300 === 0) {
      console.log(`  t ${(s / 30).toFixed(0).padStart(3)} s ${st.phase.padEnd(5)} stamina ${st.stamina.toFixed(2)} pull ${st.pullForce.toFixed(1)} N ` +
        `tension ${tension.toFixed(1)} N line ${lineLen.toFixed(1)} m mouth (${st.mouth.x.toFixed(1)}, ${st.mouth.y.toFixed(2)}, ${st.mouth.z.toFixed(1)})`);
    }
    if (st.landable) { landable = true; tMax = s / 30; break; }
  }
  check(landable, `played out and landable after ${tMax.toFixed(0)} s (max pull ${maxPull.toFixed(1)} N)`);
  const f = sim.land(id);
  check(!!f && f.state === 'held', 'landed and held');
  sim.setHeld(id, { x: 4.6, y: 1.4, z: 5.8 }, { x: 0, y: 0, z: 0, w: 1 });
  sim.release(id, { x: 4.0, z: 3.0 });
  for (let s = 0; s < 3 * 30; s++) sim.update(1 / 30);
  check(sim.fish[id].state === 'flee' || sim.fish[id].state === 'hide', 'released: it bolts for cover');
}
const bolted = (id: number) => ['flee', 'hide', 'jump'].includes(sim.fish[id].state);
const lost: Array<[number, string]> = [];
sim.onLoose = (id, why) => lost.push([id, why]);
// a bite not struck in time: the trout spits the lure and bolts
const b2 = lureFor(calmPoolTrout());
check(!!b2, 'another bite');
if (b2) {
  for (let s = 0; s < 2 * 30; s++) sim.update(1 / 30);
  check(lost.some(([id, why]) => id === b2.fishId && why === 'spat') && bolted(b2.fishId), `not struck in time: spat and bolted (${sim.fish[b2.fishId].state})`);
}
// the line breaks mid-fight: free, it bolts
const b3 = lureFor(calmPoolTrout());
if (b3 && sim.hook(b3.fishId)) {
  for (let s = 0; s < 3 * 30; s++) {
    sim.setLine(b3.fishId, { tip: { x: 4.6, y: 2.0, z: 6.6 }, tension: 4 });
    sim.update(1 / 30);
  }
  sim.unhook(b3.fishId);
  for (let s = 0; s < 2 * 30; s++) sim.update(1 / 30);
  check(bolted(b3.fishId), `line broken: it bolts (${sim.fish[b3.fishId].state})`);
} else check(false, 'a third bite to break the line on');
// a slack line: a lively trout throws the hook (2.5 s of slack, then ~0.35 per second)
const b4 = lureFor(calmPoolTrout());
if (b4 && sim.hook(b4.fishId)) {
  for (let s = 0; s < 20 * 30 && !lost.some(([id, why]) => id === b4.fishId && why === 'threw'); s++) sim.update(1 / 30);
  for (let s = 0; s < 2 * 30; s++) sim.update(1 / 30);
  check(lost.some(([id, why]) => id === b4.fishId && why === 'threw') && bolted(b4.fishId), `slack line: threw the hook and bolted (${sim.fish[b4.fishId].state})`);
} else check(false, 'a fourth bite for the slack line');
sim.setLure(null);
for (let s = 0; s < 60 * 30; s++) sim.update(1 / 30);
check(inWater(), 'every trout in the water at the end');
const c0 = performance.now();
for (let s = 0; s < 60 * 60; s++) sim.update(1 / 60);
console.log(`behaviour: ${((performance.now() - c0) / 3600).toFixed(3)} ms of CPU per 60 fps frame for ${sim.fish.length} trout`);
console.log(sim.summary().join('\n'));
console.log(failures ? `${failures} check(s) failed` : 'all checks passed');
process.exit(failures ? 1 : 0);
