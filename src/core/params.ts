/** URL switches. Shot mode is used by scripts/shots.ts to capture fixed camera spots. */
const q = new URLSearchParams(location.search);

export type TimeOfDay = 'dawn' | 'morning' | 'golden' | 'night';

export const params = {
  /** fixed camera spot id (C1..C6); enables photo mode: no HUD, frozen time */
  shot: q.get('shot'),
  tod: (q.get('tod') as TimeOfDay | null) ?? 'morning',
  debug: q.has('debug'),
  /** free-fly debug camera */
  fly: q.has('fly'),
  /** disable post-processing (validation baseline) */
  noPost: q.has('nopost'),
  /** with ?nopost: no tone mapping, so debug views can be read back as linear values */
  lin: q.has('lin'),
  /** bloom input clamp (A/B) */
  bloomMax: q.has('bloommax') ? Number(q.get('bloommax')) : null,
  /** water uniforms for A/B: ?wu=glint:0,mirrorCap:0.5 */
  wu: q.get('wu'),
  /** render scale override */
  scale: q.has('scale') ? Number(q.get('scale')) : 1,
  /** arbitrary camera: x,y,z,tx,ty,tz (absolute) */
  cam: q.get('cam')?.split(',').map(Number) ?? null,
  /** debug view for materials: e.g. layer:3 */
  dbg: q.get('dbg'),
  /** frozen simulation time in seconds for deterministic captures */
  t: q.has('t') ? Number(q.get('t')) : 12.0,
  /** ?water=0: no water (baseline for costs and part-1 work) */
  water: q.get('water') !== '0',
  /** ?nowatermesh: the water surface hidden, everything under it (field, caustics, refracted shadows) kept */
  noWaterMesh: q.has('nowatermesh'),
  /** ?refl=0: pool mirror off (sky reflection only) */
  refl: q.get('refl') !== '0',
  /** ?caus=0: no underwater light hook on the bed and rocks (caustics, beam absorption, refracted shadows) */
  caus: q.get('caus') !== '0',
  /** ?shwarp=0: underwater shadow lookups along the air direction */
  shwarp: q.get('shwarp') !== '0',
  /** ?rocks=0: no riffle rocks */
  rocks: q.get('rocks') !== '0',
  /** ?riffle=0: generic fast water (no riffle/pool-head bake: standing waves, rock wakes, tongues, plunge) */
  riffle: q.get('riffle') !== '0',
  /** ?pool=0: the pool mirror, waterline and caustic net as before the pool mirror + waterline patch */
  pool: q.get('pool') !== '0',
  /** ?trees=0: no forest (A/B and costs) */
  trees: q.get('trees') !== '0',
  /** ?props=0: no bank boulders, stones, logs, ferns */
  props: q.get('props') !== '0',
  /** ?spruce=0: no fallen spruce in the pool (cost A/B; its flood sticks and its scum line stay) */
  spruce: q.get('spruce') !== '0',
  /** ?grass=0: no grass blades */
  grass: q.get('grass') !== '0',
  /** ?mist=0: no mist */
  mist: q.get('mist') !== '0',
  /** ?mistdbg: the mist alone (in-scatter x6, red = opacity / 2) */
  mistDbg: q.has('mistdbg'),
  /** ?mu=density:0.03,sunGain:1: mist uniforms for A/B */
  mu: q.get('mu'),
  /** ?mistfilter=old: the half-resolution mist blurred and upsampled without regard to depth (A/B) */
  mistFilterOld: q.get('mistfilter') === 'old',
  /** ?sharpen=0..1: contrast-adaptive sharpening after SMAA (off by default) */
  sharpen: Number(q.get('sharpen') ?? 0),
  /** ?valleyramp=M: metres over which the valley aerosol fades in beyond its clear zone (0: the old hard start) */
  valleyRamp: q.has('valleyramp') ? Number(q.get('valleyramp')) : null,
  /** the valley-wall tree cards with per-tree variation and backlit edges (round 2, default since the final pass);
   *  ?farforest=old: round 1's cards (A/B) */
  farForestOld: q.get('farforest') === 'old',
  /** ?prescale=: resolution of the GTAO normal/depth pre-pass (A/B; default: the AO's own resolution) */
  preScale: q.has('prescale') ? Number(q.get('prescale')) : null,
  /** ?treeshadow=0: trees cast no shadows (cost A/B) */
  treeShadow: q.get('treeshadow') !== '0',
  /** ?mirrorevery=N: pool mirror refresh interval (frames) */
  mirrorEvery: Math.max(1, Number(q.get('mirrorevery') ?? 4)),
  /** ?aosamples=N: GTAO samples */
  aoSamples: Number(q.get('aosamples') ?? 8),
  /** ?shadowevery=N: redraw the (fixed) sun shadow map every N frames (part 8 round 2: 3 -> 4, the slow sway does not
   * show at 15 Hz and C3's redraw frames set its GPU p95 on the 14 ms line; final pass: 4 -> 6, C3 and W2 were over) */
  shadowEvery: Math.max(1, Number(q.get('shadowevery') ?? 6)),
  /** ?shadowmap=N: the sun's shadow map size (the PCF radius follows, same softening in metres) */
  shadowMap: Number(q.get('shadowmap') ?? 2048),
  /** ?mirrorscale=S: the pool mirror's resolution, a fraction of the frame's (part 8 round 2: 0.45 -> 0.4 for C2's
   * GPU budget after the water fix; its blur is measured in mirror texels, so the copy keeps its sharpness) */
  mirrorScale: Number(q.get('mirrorscale') ?? 0.4),
  /** ?noao=1: the GTAO pre-pass draws nothing (A/B switch; its GPU time stays ~2.5 ms: the frame's first pass also
   * carries GPU work queued before it, see DECISIONS.md "Polish round") */
  noAO: q.get('noao') === '1',
  /** ?gpupasses=1: GPU time split by pass in the frame stats (window.__RM.gpuPasses; capture flag --gpupasses) */
  gpuPasses: q.get('gpupasses') === '1',
  /** water debug view: normal|flow|foam|mask|depth|path|T|fresnel|mirror|refrok|refl|bg|rough */
  wdbg: q.get('wdbg'),
  /** splash for captures: x,z[,age s[,strength]] */
  splash: q.get('splash'),
  /** red cubes above and below the pool to check the mirror */
  mirrorTest: q.has('mirrortest'),
  /**
   * ?precompile=1: renderer.compileAsync before the first frames. Off by default: it builds canvas-target
   * pipelines the pass-based frames never use (measured -3 s load without it).
   */
  precompile: q.get('precompile') === '1',
  /** ?pipelog: log every warm-up pipeline's compile latency (report.json); ?pipelog=serial compiles them one at a time */
  pipelog: (q.has('pipelog') ? (q.get('pipelog') === 'serial' ? 'serial' : 'log') : 'off') as 'off' | 'log' | 'serial',
};

export const shotMode = params.shot !== null;
