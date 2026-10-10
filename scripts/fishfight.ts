/**
 * Parts 6 + 7 headless fight check (no browser, no GPU): `pnpm exec tsx scripts/fishfight.ts [--line 12] [--drain 2.6] [--runs 12]`.
 * Part 6's FishingSim with part 7's FishSim as its trout (the same coupling as the game: the school's bite starts the
 * bite, its trout pulls on the line, the tackle's tension goes back to it in newtons), on the river field the game
 * bakes from the analytic terrain, from the pool's edge (the photo poses' stand). Scripted players: `good` reels
 * while the tension is under 0.55 and lets go above 0.82 or once the line strains; `greedy` holds the reel down;
 * `ignore` never strikes. Prints each run (bite wait, trout, fight length, outcome) and the summary; exits non-zero if
 * a good player lands under 3/4 or with a median fight outside 8-25 s, greedy reeling never snaps the line, an ignored
 * bite stays on, or a bite takes over a minute.
 */
import type { FishingButtons, FishSource, HeldFish } from '../src/fishing/types';

const args = process.argv.slice(2);
const arg = (n: string, d: string) => {
  const i = args.indexOf(`--${n}`);
  return i >= 0 ? args[i + 1] : d;
};
// the game modules read URL switches at import time (core/params.ts, fishing/config.ts): set before importing them
(globalThis as unknown as { location: { search: string } }).location ??= { search: `?fishline=${arg('line', '9')}` };
const THREE = await import('three/webgpu');
const { heightAt } = await import('../src/world/Terrain');
const { placeRocks } = await import('../src/water/Rocks');
const { CORE } = await import('../src/world/World');
const { prepareRiverWater } = await import('../src/water/RiverWater');
const { Habitat } = await import('../src/fish/FishHabitat');
const { FishSim } = await import('../src/fish/FishSim');
const { SPECIES: TROUT } = await import('../src/fish/species');
const { FishingSim } = await import('../src/fishing/FishingSim');
const { WaterProbe } = await import('../src/fishing/WaterProbe');
const { FishingEventBus } = await import('../src/fishing/events');
const { Rng } = await import('../src/fishing/util');
const { TUNE } = await import('../src/fishing/config');
const { ANGLER_STAND } = await import('../src/fishing/poses');

const drain = Number(arg('drain', '2.2'));
const runs = Number(arg('runs', '12'));
const power = Number(arg('power', '0.3'));
const t0 = Date.now();
const rocks = placeRocks(heightAt);
const water = prepareRiverWater({ groundY: heightAt, core: CORE, obstacles: rocks.map((r) => r.obstacle) });
const habitat = new Habitat(water.field, rocks.map((r) => ({ x: r.x, z: r.z, r: r.obstacle.r, topY: r.obstacle.topY })));
console.log(`field in ${Date.now() - t0} ms; line ${TUNE.lineNewtons} N per unit of tension, fight drain x${drain}, cast power ${power}`);

type Policy = 'good' | 'greedy' | 'ignore';
interface Result { policy: Policy; run: number; outcome: string; biteWait: number; species: string; cm: number; fight: number; maxT: number; landedAt: string }

function run(policy: Policy, n: number): Result {
  const fsim = new FishSim(habitat, { seed: 20261009, fightDrain: drain });
  fsim.presimulate(24 + n * 9);
  const q = new THREE.Quaternion(), e = new THREE.Euler(0, 0, 0, 'YXZ');
  const src: FishSource = {
    setLure: (l) => fsim.setLure(l),
    attract: (r, sp) => fsim.attract(r, sp),
    hook: (id) => fsim.hook(id),
    setLine: (id, l) => fsim.setLine(id, l),
    fight: (id) => fsim.fightState(id),
    unhook: (id) => fsim.unhook(id),
    land: (id) => {
      const f = fsim.land(id);
      return f ? { position: f.pos.clone(), quaternion: q.setFromEuler(e.set(-f.pitch, f.yaw, f.roll, 'YXZ')).clone(), scale: f.scale } : null;
    },
    setHeld: (id, p, qq, s) => fsim.setHeld(id, p, qq, s),
    release: (id, at) => fsim.release(id, at),
    species: (id) => TROUT[id],
  };
  // casts fan out across the pool (-30, 0, +30 degrees) so different trout come to the bait; --at x,z,yaw moves the stand
  const at = arg('at', '').split(',').map(Number);
  const st = at.length === 3 ? { x: at[0], z: at[1], yawDeg: at[2] } : ANGLER_STAND, a = ((st.yawDeg + [0, -30, 30][n % 3]) * Math.PI) / 180;
  const feet = new THREE.Vector3(st.x, heightAt(st.x, st.z), st.z);
  const fwd = new THREE.Vector3(-Math.sin(a), 0, -Math.cos(a));
  const tip = feet.clone().addScaledVector(fwd, 2.3).add(new THREE.Vector3(0, 2.05, 0));
  const holdPos = feet.clone().addScaledVector(fwd, 0.5).add(new THREE.Vector3(0, 1.3, 0));
  const hooks = {
    playCast: () => 0.4, playReel: () => {}, playHold: (_f: HeldFish) => {}, idle: () => {}, playStrike: () => {},
    playRelease: () => 1.1,
    getAim: (o: InstanceType<typeof THREE.Vector3>, d: InstanceType<typeof THREE.Vector3>) => { o.copy(tip); d.copy(fwd); },
    getFeet: (o: InstanceType<typeof THREE.Vector3>) => o.copy(feet),
    getFishHold: (p: InstanceType<typeof THREE.Vector3>, qq: InstanceType<typeof THREE.Quaternion>) => { p.copy(holdPos); qq.identity(); return true; },
  };
  // the water's event API as far as the trout notice it: splashes and the float's plop startle or draw them, the rest
  // (rings, boils, swirls, ripple trains, spray, drips) is only drawn
  const droplets = new Proxy({}, { get: () => () => {} });
  const heard = { splash: (x: number, z: number, _t: number, s: number) => fsim.splash(x, z, s), plop: (x: number, z: number) => fsim.splash(x, z, 0.3), droplets };
  const events = new Proxy(heard as Record<string, unknown>, { get: (o, k: string) => o[k] ?? (() => {}) });
  const stub = { data: { field: water.field }, events };
  const probe = new WaterProbe(stub as unknown as ConstructorParameters<typeof WaterProbe>[0], heightAt);
  // the float lands through floatPlop (rings, trains and drops, fishing polish): heard as the plop was
  probe.floatPlop = (x: number, z: number) => fsim.splash(x, z, 0.3);
  const bus = new FishingEventBus();
  const sim = new FishingSim(probe, hooks, bus, new Rng(1000 + n), () => new THREE.Object3D());
  sim.school = src;
  fsim.onBite = (ev) => sim.schoolBite(ev);
  fsim.onLoose = (id, why) => sim.schoolLoose(id, why);
  fsim.onTake = (id) => sim.schoolTake(id);
  sim.tip.copy(tip);
  sim.reset();
  const b: FishingButtons = { cast: false, castPressed: false, castReleased: false, reel: false, reelPressed: false, reelReleased: false };
  const press = (k: 'cast' | 'reel', down: boolean) => {
    if (k === 'cast') { if (down && !b.cast) b.castPressed = true; if (!down && b.cast) b.castReleased = true; b.cast = down; }
    else { if (down && !b.reel) b.reelPressed = true; if (!down && b.reel) b.reelReleased = true; b.reel = down; }
  };
  const holdFor = TUNE.chargeTime * (1 - Math.sqrt(1 - power));
  // --fps N: the frame rate the game would run at (60 default; 240 checks the high-refresh screens)
  const dt = 1 / Number(arg('fps', '60'));
  let t = 0, landedT = -1, biteT = -1, fightT = -1, maxT = 0, outcome = 'timeout', landedAt = '', escaped = 0;
  bus.on('escape', () => escaped++);
  const head = new THREE.Vector3();
  for (; t < 180; t += dt) {
    fsim.setPlayer(head.copy(feet).setY(feet.y + 1.65), dt);
    fsim.update(dt);
    hooks.getAim(sim.aimOrigin, sim.aimDir);
    sim.feet.copy(feet);
    const s = sim.state;
    if (s === 'idle' && t > 0.5 && !b.cast && landedT < 0) press('cast', true);
    if (s === 'aiming' && sim.stateTime >= holdFor) press('cast', false);
    if (s === 'floating' && landedT < 0) {
      landedT = t;
      landedAt = `${sim.onWater ? 'water' : 'land'} ${Math.hypot(sim.bobber.x - feet.x, sim.bobber.z - feet.z).toFixed(1)} m`;
      if (!sim.onWater) { outcome = 'on land'; break; }
    }
    if (s === 'bite' && biteT < 0) biteT = t;
    if (s === 'bite' && policy !== 'ignore' && t - biteT > 0.3) press('cast', true);
    if (s === 'fight') {
      if (b.cast) press('cast', false);
      if (fightT < 0) fightT = t;
      maxT = Math.max(maxT, sim.tension);
      if (policy === 'greedy') press('reel', true);
      else if (!b.reel && sim.tension < 0.55) press('reel', true);
      else if (b.reel && (sim.tension > 0.82 || sim.strain > 0.05)) press('reel', false);
    } else if (b.reel) press('reel', false);
    sim.update(dt, t, b);
    b.castPressed = b.castReleased = b.reelPressed = b.reelReleased = false;
    if (sim.state === 'landing') { outcome = 'landed'; break; }
    if (sim.state === 'snapped') { outcome = 'snapped'; break; }
    if (fightT >= 0 && sim.state === 'floating') { outcome = 'threw'; break; }
    if (policy === 'ignore' && escaped) { outcome = 'escaped'; break; }
  }
  const f = sim.fish;
  return {
    policy, run: n, outcome, biteWait: biteT >= 0 ? +(biteT - landedT).toFixed(1) : -1, species: biteT >= 0 ? f.species.id : '-', cm: biteT >= 0 ? f.lengthCm : 0,
    fight: fightT >= 0 ? +(t - fightT).toFixed(1) : 0, maxT: +maxT.toFixed(2), landedAt,
  };
}

const all: Result[] = [];
for (const policy of ['good', 'greedy', 'ignore'] as Policy[]) {
  for (let n = 0; n < (policy === 'good' ? runs : Math.ceil(runs / 2)); n++) {
    const r = run(policy, n);
    all.push(r);
    console.log(`${policy.padEnd(6)} #${String(n).padStart(2)} ${r.outcome.padEnd(8)} float ${r.landedAt.padEnd(12)} bite after ${String(r.biteWait).padStart(5)} s  ${r.species.padEnd(9)} ${String(r.cm).padStart(2)} cm  fight ${String(r.fight).padStart(5)} s  max tension ${r.maxT}`);
  }
}
// casts that landed on the bank say nothing about the fight
const inWater = all.filter((r) => r.outcome !== 'on land');
const good = inWater.filter((r) => r.policy === 'good'), landed = good.filter((r) => r.outcome === 'landed');
const fights = landed.map((r) => r.fight).sort((x, y) => x - y);
const med = fights.length ? fights[Math.floor(fights.length / 2)] : 0;
const greedy = inWater.filter((r) => r.policy === 'greedy'), ignore = inWater.filter((r) => r.policy === 'ignore');
const waits = good.filter((r) => r.biteWait >= 0).map((r) => r.biteWait).sort((x, y) => x - y);
console.log(`good: landed ${landed.length}/${good.length}, fights ${fights[0] ?? '-'}-${fights[fights.length - 1] ?? '-'} s (median ${med} s); bite waits ${waits[0] ?? '-'}-${waits[waits.length - 1] ?? '-'} s`);
console.log(`greedy: snapped ${greedy.filter((r) => r.outcome === 'snapped').length}/${greedy.length}; ignore: escaped ${ignore.filter((r) => r.outcome === 'escaped').length}/${ignore.length}`);
// typical trout (36-46 cm) take ~14-20 s; a stand whose nearest trout are small (33-34 cm) lands them in ~9-11 s
const ok = landed.length >= good.length * 0.75 && med >= 8 && med <= 25 && greedy.some((r) => r.outcome === 'snapped')
  && ignore.every((r) => r.outcome === 'escaped') && waits.length > 0 && waits[waits.length - 1] <= 60;
console.log(ok ? 'all checks passed' : 'CHECKS FAILED');
process.exit(ok ? 0 : 1);
