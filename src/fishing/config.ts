/**
 * Fishing switches and tuning. URL switches are parsed here so the only hook in main.ts is one import.
 *   ?fish=0          no fishing at all (the loop is on by default; main.ts skips the import)
 *   ?fishfast        bites come ~5x sooner (testing)
 *   ?fishhud         show the HUD even in ?shot photo mode (HUD captures)
 *   ?fishseed=N      deterministic fish (species, sizes, fight) for tests
 *   ?fishstand=pool|riffle|dock stand-in fishing spot (gravel bar by the pool | south bank of the riffle | the dock tip)
 *   ?fishlock=0      no pointer lock for the stand-in camera (keys only; used by the playtest)
 */
const q = new URLSearchParams(typeof location === 'undefined' ? '' : location.search);

export const fishParams = {
  enabled: q.get('fish') !== '0',
  fast: q.has('fishfast'),
  hud: q.has('fishhud'),
  seed: q.has('fishseed') ? Number(q.get('fishseed')) : null,
  stand: q.get('fishstand') ?? 'pool',
  pointerLock: q.get('fishlock') !== '0',
};

/** Under public/assets/. Copied from the prep folders (see INTEGRATION.md); procedural stand-ins load if missing. */
export const FISH_ASSETS = {
  rod: 'models/props/fishing_rod.glb',
  fish: 'models/fish/trout_all.glb',
};

/**
 * Stand-in fishing spots (feet x, z; yaw in degrees, 0 = facing north (-Z), + = turning west (left)). main.ts adds
 * `dock` from part 4 (`camp.castStand`, the dock tip).
 */
export const STANDS: Record<string, { x: number; z: number; yawDeg: number; pitchDeg: number }> = {
  // south gravel bar: water 2 m ahead, the 2.2-2.4 m deep pool 8-12 m out, the far bank ~16 m
  pool: { x: 4.0, z: 6.5, yawDeg: 20, pitchDeg: -11 },
  // riffle: 0.2-0.6 m deep water at ~2 m/s from 2 to 8 m out (rainbows and cutthroat); cast softly
  riffle: { x: -16.0, z: 6.5, yawDeg: 40, pitchDeg: -14 },
};

export const TUNE = {
  /** eye height of the stand-in camera above the ground */
  eyeHeight: 1.65,
  /** cast distance range from the rod tip (m, flat ground) at power 0 and 1; the pool is ~14 m wide */
  castMin: 2.0,
  castMax: 16,
  /** seconds of holding the button from power 0 to 1 */
  chargeTime: 1.1,
  /** launch elevation (deg) */
  castElevation: 30,
  /** air drag on the float rig (1/m, quadratic) */
  airDrag: 0.018,
  /** line hanging from the tip when idle (m): reeled up short while he walks, so it never streaks across the view */
  hangLength: 0.12,
  /** and drawn back to cast: the float swings under the tip on a longer drop */
  castDrop: 1.6,
  /** float to hook (m) */
  leader: 0.7,
  /** reel speed with no load (m/s) */
  reelSpeed: 1.45,
  /** retrieve speed for the float without a fish (m/s) */
  retrieveSpeed: 1.8,
  /** drag slips above this tension (not reeling / while reeling) */
  dragFree: 0.72,
  dragReeling: 1.2,
  /** the line strains above this tension and snaps once the strain builds up */
  breakTension: 1.0,
  /** landed once the fish is this close to the feet (m) */
  landRadius: 2.5,
  /** seconds: lift into the hands, hold, lower */
  landTime: 0.8,
  holdTime: 2.8,
  /** spooked fish come back slower after a missed bite */
  spookFactor: 1.35,
  /** bite delays scale (?fishfast: 0.2) */
  biteScale: q.has('fishfast') ? 0.2 : 1,
  // with part 7's trout (world.fish)
  /**
   * newtons per unit of tension: a trout's pull (N) / lineNewtons is its share of the tension (1 = the line strains),
   * saturating toward 1.25 for the strongest (FishingSim.stepFightSchool); tuned with scripts/fishfight.ts
   */
  lineNewtons: Number(q.get('fishline') ?? 9),
  /** extra pull per metre the hooked trout is beyond the line's length (N/m): the line holds it like part 6's sphere */
  lineStiffness: 40,
  /** when the patience timer runs out, the nearest calm trout within this many metres comes to the bait */
  attractRange: 12,
};
