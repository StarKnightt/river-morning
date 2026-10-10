import './style.css';
import './ui/quality.css';
import * as THREE from 'three/webgpu';
import { Engine } from './core/Engine';
import { startQuality } from './core/quality';
import { Assets } from './core/Assets';
import { params, shotMode } from './core/params';
import { buildWorld, timings } from './world/World';
import { applyCameraSpot, CAMERA_SPOTS } from './debug/CameraSpots';
import { buildCamp, prefetchCamp, CABIN } from './camp';
import { CHIMNEY_BOXES } from './camp/chimney';
import { CABIN_GEO, cabinX, cabinZ } from './camp/layout';
import { PipelineWarmup, type PipeLog } from './core/Warmup';
import { shadowTriggerStats } from './gfx/shadowTrigger';
import { gameEvents, installAudio } from './audio';
import { mountTimeOfDayKey } from './ui/TimeOfDayKey';
import { installGovernor } from './core/ResolutionGovernor';

declare global {
  interface Window {
    __RM: {
      ready: boolean;
      error: string | null;
      loadMs: number;
      timings: Record<string, number>;
      backend: string;
      adapter: string;
      frameStats: () => {
        avg: number; p95: number; med: number; gpuAvg: number; gpuP95: number;
        /** per frame kind (0 plain, 1 pool mirror, 2 shadow map, 3 both) */
        byKind?: Record<string, { n: number; wallMed: number; wallP95: number; cpuMed: number; cpuP95: number; gpuN: number; gpuAvg: number; gpuP95: number }>;
      };
      info: () => Record<string, number>;
      spot: (id: string) => void;
      spots: string[];
      pipelog?: PipeLog[];
      nodeBuild?: Record<string, number>;
      biggestShader?: string;
    };
  }
}

const loadStart = performance.now();
window.__RM = {
  ready: false, error: null, loadMs: 0, timings: {}, backend: '', adapter: '',
  frameStats: () => ({ avg: 0, p95: 0, med: 0, gpuAvg: 0, gpuP95: 0 }), info: () => ({}), spot: () => {}, spots: CAMERA_SPOTS.map((s) => s.id),
};

const loadingEl = document.getElementById('loading')!;
const fillEl = loadingEl.querySelector('.loading-fill') as HTMLElement;
const textEl = loadingEl.querySelector('.loading-text') as HTMLElement;
const setProgress = (p: number, label?: string) => {
  fillEl.style.width = `${Math.round(p * 100)}%`;
  if (label) textEl.textContent = label;
};

async function main() {
  const engine = await Engine.create(document.getElementById('app')!);
  window.__RM.backend = engine.backendName;
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    const info = adapter?.info;
    window.__RM.adapter = info ? `${info.vendor} ${info.architecture} ${info.description}`.trim() : 'unknown';
  } catch {
    window.__RM.adapter = 'unknown';
  }

  const assets = new Assets();
  assets.anisotropy = 16;
  assets.manager.onProgress = (_url, loaded, total) => setProgress(0.1 + 0.75 * (loaded / Math.max(total, 1)));
  prefetchCamp(assets);

  const world = await buildWorld(engine, assets, (label) => setProgress(Number(fillEl.style.width.replace('%', '')) / 100 || 0.05, label));
  // part 8: the T key steps the time of day, with a short fading label (not in photo mode; no buttons)
  if (!shotMode) mountTimeOfDayKey(world.timeOfDay, document.getElementById('hud')!);
  const camp = await buildCamp({ engine, world, assets });
  // part 4 + 8: the lit window, and part 8's fire light on the ring following the camp's flicker (the camp keeps the
  // only flame: part 8's flame sprite is hidden)
  camp?.attachTimeOfDay(world.timeOfDay);
  // part 8's soft glow sprite read as an orange ball hanging over the camp's fire at night (C5, C6): the camp's flames,
  // embers, the smoke's fire glow and the fire light carry it
  if (camp && world.timeOfDay.nightLights) world.timeOfDay.nightLights.glow.visible = false;
  const abHide = new URLSearchParams(location.search).get('abhide');
  if (abHide) (await import('./debug/ABToggle')).installABToggle(engine, abHide.split(','));

  const camera = engine.camera;
  if (params.shot) applyCameraSpot(camera, params.shot);
  else applyCameraSpot(camera, 'C1');
  if (params.cam && params.cam.length === 6) {
    const [x, y, z, tx, ty, tz] = params.cam;
    camera.position.set(x, y, z);
    camera.lookAt(tx, ty, tz);
  }

  let fly: { syncFromCamera(): void } | null = null;
  let player: import('./player/Player').Player | null = null;
  const orbit = new URLSearchParams(location.search).has('orbit');
  const playerControl = !shotMode && !params.fly && !orbit;
  if (!shotMode) {
    const { installTravelKeys } = await import('./debug/Travel');
    // a jump moves the angler (the camera snaps behind him); ?fly and ?orbit keep moving only the camera
    installTravelKeys(camera, {
      afterMove: () => fly?.syncFromCamera(),
      onTeleport: playerControl ? (feet, look) => player?.teleportTo(feet, look) : undefined,
    });
  }
  if (params.fly && !shotMode) {
    const { FlyCamera } = await import('./debug/FlyCamera');
    const flyCam = new FlyCamera(camera, engine.renderer.domElement);
    fly = flyCam;
    engine.addUpdater((dt) => flyCam.update(dt));
  } else if (orbit && !shotMode) {
    const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
    const controls = new OrbitControls(camera, engine.renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.update();
    engine.addUpdater(() => controls.update());
  }

  // part 5: the angler; third-person controls only in play mode (photo mode, ?fly and ?orbit keep their camera).
  // Installed before the warm-up frames so his pipelines compile behind the loading screen.
  const { installPlayer } = await import('./player/Player');
  player = await installPlayer(engine, assets, world, { control: playerControl, frozenTime: shotMode ? params.t : null, debug: params.debug });
  // part 4's dock deck, porch and step stones replace the stand-in deck; the cabin, chimney, fire ring, seats,
  // woodpile, chopping block and mooring posts block his walk
  if (player && camp) {
    camp.attachGround(player.ground);
    // the fieldstone firebox (part 4 round 2, cabin-local CHIMNEY_BOXES[0]) reaches 9 cm past the camp's chimney
    // blocker: one more box over it, 6 cm wider for the stones set proud of its faces
    const fb = CHIMNEY_BOXES[0], lx = (fb.x0 + fb.x1) / 2, lz = (fb.z0 + fb.z1) / 2;
    player.ground.addBox({ x: cabinX(lx, lz), z: cabinZ(lx, lz), hx: (fb.x1 - fb.x0) / 2 + 0.06, hz: (fb.z1 - fb.z0) / 2 + 0.06, rotY: CABIN.yaw });
    // the crossed log ends at the four corners stand 0.15-0.24 m proud of the wall box (round 2 lengthened them)
    const w = CABIN_GEO.walls;
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const cx = sx * (w.hx - 0.03), cz = sz * (w.hz - 0.05);
        player.ground.addBox({ x: cabinX(cx, cz), z: cabinZ(cx, cz), hx: 0.36, hz: 0.36, rotY: CABIN.yaw });
      }
    }
  }
  // the cabin's door (E) and the room inside (src/camp/door.ts): play mode only
  let cabin: import('./camp/door').CabinPlay | null = null;
  if (player && camp && playerControl) {
    const { installCabinPlay } = await import('./camp/door');
    const tod = world.timeOfDay;
    cabin = installCabinPlay({
      engine, camp, player, night: () => tod.state?.night ?? 0, lamp: tod.nightLights?.windowLight ?? null,
      sound: (x, y, z, speed) => gameEvents.emit('footstep', { x, y, z, speed }),
    });
    if (params.debug) Object.assign(window, { __cabin: cabin });
  }
  if (camp && !cabin && new URLSearchParams(location.search).has('cabinopen')) {
    const tod = world.timeOfDay;
    (await import('./camp/door')).installCabinOpenView({ engine, camp, night: () => tod.state?.night ?? 0, lamp: tod.nightLights?.windowLight ?? null });
  }

  if (player && world.fish) {
    // part 7: the trout see the angler (his head over the water), not the camera
    const p = player, school = world.fish, head = new THREE.Vector3();
    engine.addUpdater(() => school.setPlayer(head.set(p.controller.position.x, p.controller.rootY + 1.65, p.controller.position.z)));
  }

  engine.setupPost();
  await startQuality(engine, world);
  setProgress(0.9, 'Preparing shaders');
  timings.sinceStart = Math.round(performance.now() - loadStart);
  if (shotMode) engine.fixedTime = params.t;
  engine.time = engine.fixedTime ?? 0;

  const water = world.water;
  if (water && params.splash) {
    // ?splash=x,z[,age s[,strength]]: a splash frozen at the given age, for captures. At 0.62 s the swell is
    // ~0.26 m out, the jet has fallen back in the middle and its last drops are still in the air.
    const [x, z, age = 0.62, strength = 1] = params.splash.split(',').map(Number);
    water.debugSplash(x, z, age, strength);
  }
  if (water && params.debug) {
    // F: splash where the camera looks at the water (S is taken by the fly camera)
    const dir = new THREE.Vector3();
    window.addEventListener('keydown', (e) => {
      if (e.key !== 'f' && e.key !== 'F') return;
      camera.getWorldDirection(dir);
      for (let d = 0.5; d < 80; d += 0.25) {
        const x = camera.position.x + dir.x * d, y = camera.position.y + dir.y * d, z = camera.position.z + dir.z * d;
        if (y < water.query(x, z).y) {
          water.splash(x, z, 1);
          gameEvents.emit('splash', { x, z, strength: 1 });
          break;
        }
      }
    });
  }

  window.__RM.timings = timings;
  window.__RM.frameStats = () => engine.frameStats();
  (window.__RM as unknown as { frameLog: () => unknown }).frameLog = () => engine.frameLog();
  (window.__RM as unknown as { gpuPasses: () => unknown }).gpuPasses = () => engine.gpuPasses();
  window.__RM.info = () => {
    const r = engine.renderer.info;
    const f = world.forest?.stats;
    // render.calls counts renderer.render() calls since the start (it grew ~7k per spot in round 1's reports);
    // drawCalls and frameCalls are the last frame's draws and passes
    return {
      drawCalls: r.render.drawCalls, passes: r.render.frameCalls, renderCallsTotal: r.render.calls,
      triangles: r.render.triangles, geometries: r.memory.geometries, textures: r.memory.textures,
      ...(f ? { trees: f.visible, lod0: f.lod[0], lod1: f.lod[1], lod2: f.lod[2], casters: f.casters, mirrorTrees: f.mirror } : {}),
      ...(world.grassField ? { tufts: world.grassField.stats.tufts } : {}),
      shadowRedraws: shadowTriggerStats.redraws, shadowTriggerMisses: shadowTriggerStats.misses,
    };
  };
  // what the sun's shadow pass and the pool mirror draw (objects that pass the layer, visibility and caster tests;
  // frustum culling not applied), grouped by name with digits folded: draws per pass decide the heavy frames' cost
  (window.__RM as unknown as { passes: () => unknown }).passes = () => {
    const sunCam = world.sky.sun.shadow.camera, mirrorCam = world.water?.mirror?.cam ?? null;
    const shadow: Record<string, number> = {}, mirror: Record<string, number> = {};
    let nShadow = 0, nMirror = 0, emptyShadow = 0, emptyMirror = 0;
    const seen = (o: THREE.Object3D): boolean => o.visible && (!o.parent || seen(o.parent));
    engine.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh || !seen(m)) return;
      const n = (m.material && Array.isArray(m.material) ? Math.max(1, m.geometry.groups.length) : 1);
      const empty = (m as THREE.InstancedMesh).isInstancedMesh && (m as THREE.InstancedMesh).count === 0;
      const key = (m.name || m.parent?.name || m.type).replace(/\d+/g, '#');
      if (m.castShadow && m.layers.test(sunCam.layers)) {
        shadow[key] = (shadow[key] ?? 0) + n;
        nShadow += n;
        if (empty) emptyShadow++;
      }
      if (mirrorCam && m.layers.test(mirrorCam.layers)) {
        mirror[key] = (mirror[key] ?? 0) + n;
        nMirror += n;
        if (empty) emptyMirror++;
      }
    });
    const sort = (r: Record<string, number>) => Object.fromEntries(Object.entries(r).sort((a, b) => b[1] - a[1]));
    return { nShadow, emptyShadow, nMirror, emptyMirror, shadow: sort(shadow), mirror: sort(mirror) };
  };
  // the triangles the GTAO pre-pass draws from the current view (layer 0, frustum-culled), largest objects first
  (window.__RM as unknown as { preTris: () => unknown }).preTris = () => {
    const cam = engine.preCam, fr = new THREE.Frustum(), sph = new THREE.Sphere();
    fr.setFromProjectionMatrix(new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse), cam.coordinateSystem, cam.reversedDepth);
    const by: Record<string, number> = {};
    let total = 0;
    const seen = (o: THREE.Object3D): boolean => o.visible && (!o.parent || seen(o.parent));
    engine.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh || !m.layers.test(cam.layers) || !seen(m)) return;
      const im = m as THREE.InstancedMesh, g = m.geometry;
      if (m.frustumCulled && !im.isInstancedMesh) {
        if (!g.boundingSphere) g.computeBoundingSphere();
        if (!fr.intersectsSphere(sph.copy(g.boundingSphere!).applyMatrix4(m.matrixWorld))) return;
      }
      const tris = ((g.index ? g.index.count : g.getAttribute('position').count) / 3) * (im.isInstancedMesh ? im.count : 1);
      const key = (m.name || m.parent?.name || m.type).replace(/\d+/g, '#');
      by[key] = (by[key] ?? 0) + tris;
      total += tris;
    });
    return { total, by: Object.fromEntries(Object.entries(by).sort((a, b) => b[1] - a[1]).slice(0, 25)) };
  };
  window.__RM.spot = (id: string) => {
    engine.resetGpuStats();
    if (player?.spot(id)) return;
    if (id.startsWith('cam:')) {
      const [x, y, z, tx, ty, tz] = id.slice(4).split(',').map(Number);
      camera.position.set(x, y, z);
      camera.lookAt(tx, ty, tz);
      return;
    }
    applyCameraSpot(camera, id);
  };

  // part 6: fishing (src/fishing), on by default (?fish=0: off). The angler casts and plays the fish (part 5's
  // fishingHooks, src/fishing/angler.ts); without him (?player=0) the camera stand-in does. After his updater (the
  // rod tip is current when the line reads it), after the spot hook (the fish-* photo poses wrap it) and before the
  // warm-up (line, float, aim track and fish compile behind the loading screen). Part 9 hears it through the bridge.
  if (new URLSearchParams(location.search).get('fish') !== '0') {
    const { anglerHooks, connectFishingAudio, installFishing, STANDS } = await import('./fishing');
    if (camp) STANDS.dock = camp.castStand;
    const p = player, rodGround = player?.rodGround ?? null;
    // (no casting inside the cabin or from its doorway: the fishing buttons and prompts rest, the rod stays in hand)
    const indoors = () => !!cabin?.indoors;
    const fishing = await installFishing({
      engine, world, assets, ...(p ? anglerHooks(p) : {}), ...(rodGround ? { rodPlace: () => (rodGround.place === 'hand' && indoors() ? 'busy' : rodGround.place) } : {}),
    });
    connectFishingAudio(gameEvents);
    // the rod put down and picked up (E) reels the line in; a jump gives fresh tackle where he lands
    if (rodGround) rodGround.fishing = { state: () => fishing.sim.state, reelIn: () => fishing.resetTackle() };
    if (p) p.onTeleport = () => fishing.resetTackle();
  }

  // warm-up behind the loading screen: draw everything (no frustum culling, mirror forced on, every tree set
  // filled), so every pipeline variant compiles before the first real frame instead of hitching when the view
  // turns. The first frames only queue the pipelines (compiled in parallel, see Warmup.ts), then the loop holds
  // until they are ready; a few synchronous frames catch any variant that only appears once the others draw.
  const unculled: THREE.Object3D[] = [];
  engine.scene.traverse((o) => {
    if (o.frustumCulled) {
      o.frustumCulled = false;
      unculled.push(o);
    }
  });
  if (water) water.warm = true;
  const warm = new PipelineWarmup(engine.renderer, params.pipelog);
  warm.begin();
  const ASYNC = 3, SYNC = 4, READY = ASYNC + SYNC + 6;
  let tc = 0;

  engine.start();
  const governor = installGovernor(engine);
  (window.__RM as unknown as { governor: () => unknown }).governor = () => governor?.state() ?? { scale: engine.renderScale, pinned: true };
  // part 9: sound (src/audio). Nothing at all in photo mode or with ?audio=0; otherwise silent until the first click
  // or key press (browser autoplay rules), files downloading just after the loading screen.
  const audio = installAudio(engine, { groundY: world.groundY });
  if (audio.enabled && player) {
    // the angler's footfalls (the wading rings already listen to them) and the ambience mix around him
    const p = player, ring = p.anim.onFootfall, foot = new THREE.Vector3();
    p.anim.onFootfall = (side) => {
      ring?.(side);
      foot.setFromMatrixPosition(p.character.bone(`foot_${side}`).matrixWorld);
      gameEvents.emit('footstep', { x: foot.x, y: foot.y, z: foot.z, speed: p.controller.groundSpeed });
    };
    audio.setFocus(() => p.controller.position);
    // the rod laid down or picked up: the softest footstep on what it lies on (gravel, grass or the deck's wood)
    if (p.rodGround) p.rodGround.onHandle = (_kind, at) => gameEvents.emit('footstep', { x: at.x, y: at.y, z: at.z, speed: 0.2 });
  }
  // part 8 -> part 9: the beds crossfade with the light (birds by day, crickets and the owl at night)
  if (audio.enabled) world.timeOfDay.onChange((tod, phase) => {
    if (phase !== 'loading') gameEvents.emit('timeOfDay', { tod });
  });
  if (audio.enabled && camp) {
    // the crackle follows part 4's fire: the small coffee fire by day (part 8 builds it up at night)
    let fire = -1;
    engine.addUpdater(() => {
      if (Math.abs(camp.fireStrength - fire) < 0.02) return;
      fire = camp.fireStrength;
      audio.setFire(fire, true);
    });
  }
  let frames = 0;
  engine.addUpdater(() => {
    frames++;
    if (frames <= READY) timings[`f${frames}`] = Math.round(performance.now() - loadStart);
    if (frames <= ASYNC + SYNC) world.forest?.warm();
    if (frames === ASYNC + 1) {
      engine.hold = true;
      tc = performance.now();
      timings.pipelines = warm.count;
      void warm.end((done, total) => setProgress(0.9 + 0.1 * (done / total), `Preparing shaders ${done}/${total}`)).then(() => {
        timings.compile = Math.round(performance.now() - tc);
        window.__RM.pipelog = warm.log;
        window.__RM.nodeBuild = warm.build;
        window.__RM.biggestShader = warm.biggest;
        // the environment's PMREM was rendered while its pipelines were still compiling (it drew nothing)
        if (engine.scene.environment) engine.scene.environment.needsPMREMUpdate = true;
        engine.hold = false;
      });
    }
    if (frames === ASYNC + SYNC + 1) {
      for (const o of unculled) o.frustumCulled = true;
      if (water) water.warm = false;
      world.forest?.endWarmup();
    }
    if (frames === READY) {
      window.__RM.loadMs = performance.now() - loadStart;
      window.__RM.ready = true;
      setProgress(1, 'Ready');
      loadingEl.classList.add('done');
    }
  });

  if (params.debug) {
    // the check scripts freeze the game around their screenshots: a capture can take seconds under an uncapped
    // compositor, and the cast meter, a bite's window and a fight must not run on meanwhile
    Object.assign(window, { __hold: (on: boolean) => { engine.hold = on; } });
    const panel = document.createElement('div');
    panel.className = 'debug-panel';
    document.getElementById('hud')!.appendChild(panel);
    setInterval(() => {
      const s = engine.frameStats();
      const i = window.__RM.info();
      const p = camera.position;
      panel.textContent = `${engine.backendName} | ${window.__RM.adapter}\n` +
        `${(1000 / s.avg).toFixed(0)} fps  avg ${s.avg.toFixed(1)} ms  p95 ${s.p95.toFixed(1)} ms\n` +
        `calls ${i.calls}  tris ${(i.triangles / 1e6).toFixed(2)}M  tex ${i.textures}\n` +
        `render scale ${engine.renderScale.toFixed(2)} (${Math.round(engine.renderScale ** 2 * 100)} % px) ${engine.bufferSize().join('x')}${governor ? '' : ' pinned'}\n` +
        `cam ${p.x.toFixed(1)} ${p.y.toFixed(1)} ${p.z.toFixed(1)}`;
    }, 500);
  }
  void world;
  void THREE;
}

main().catch((e) => {
  console.error(e);
  window.__RM.error = String(e?.stack ?? e);
  textEl.textContent = 'Error: ' + String(e?.message ?? e);
});
