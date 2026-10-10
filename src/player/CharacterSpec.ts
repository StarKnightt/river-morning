/**
 * Facts about the prepared angler (E:\shots\river-morning\prep\character\character.json): 1.82 m in boots,
 * origin on the ground between the soles, faces +Z, 53-joint MPFB game_engine rig, in-place clips at 30 fps.
 */
export const ANGLER_URL = 'character/angler.glb';
/** same path as the part 6 patch uses, so the GLB is copied once */
export const ROD_URL = 'models/props/fishing_rod.glb';

export const LOCO = {
  walkSlow: { speed: 0.805, cycle: 1.367 },
  walk: { speed: 1.498, cycle: 1.067 },
  /**
   * walk's phase relative to walk_slow, so their footfalls line up when blended (walk_slow strikes R 0.122 /
   * L 0.695 of its cycle, walk R 0.859 / L 0.344; the slow walk is uneven, this splits the difference).
   */
  walkPhaseOffset: 0.693,
  /** footfall phases of the blended cycle (walk_slow's), for wading rings */
  footfallL: 0.695,
  footfallR: 0.122,
};

/** Clip-local event times (s). `release` is when the line leaves the rod tip. */
export const CAST_EVENTS = { raise: 0.3, backcast_top: 2.45, release: 2.47, settle: 2.85 };
export const REEL_TURNS_PER_LOOP = 7;

/** Bones a "rod in hand" upper-body layer takes from `wait` while the legs walk. */
export const UPPER_BODY = new Set<string>([
  'spine_02', 'spine_03', 'clavicle_l', 'clavicle_r', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 'lowerarm_r', 'hand_l', 'hand_r',
  ...['thumb', 'index', 'middle', 'ring', 'pinky'].flatMap((f) => ['01', '02', '03'].flatMap((n) => [`${f}_${n}_l`, `${f}_${n}_r`])),
]);

/** Grip points in hand-bone space (palm centre, where a 2.5 cm handle runs through the fist). */
export const SOCKETS = {
  gripR: { bone: 'hand_r', palm: [-0.0188, 0.0851, 0.0106] as const },
  gripL: { bone: 'hand_l', palm: [0.0188, 0.0851, 0.0106] as const },
  /** hold_fish: fish centre and tail->head axis in hand_l space (a 30-40 cm trout) */
  fish: { bone: 'hand_l', centre: [0.0854, 0.1156, -0.0016] as const, headAxis: [-0.6911, -0.507, 0.5151] as const },
};

/** sit_* clips sit on air: the seat top goes at this height under the thighs */
export const SEAT_HEIGHT = 0.447;

/** Every clip in the GLB (24): 20 body clips and 4 finger-only poses. */
export type ClipName =
  | 'idle' | 'idle_look' | 'walk_slow' | 'walk' | 'wait' | 'cast' | 'strike' | 'reel_loop' | 'land_lift' | 'hold_fish'
  | 'kneel' | 'kneel_release' | 'release_fish' | 'sit_enter' | 'sit_idle' | 'sit_exit' | 'drink' | 'lantern_idle'
  | 'pickup' | 'chop';

export const LOOPING: ReadonlySet<ClipName> = new Set<ClipName>(['idle', 'walk_slow', 'walk', 'wait', 'reel_loop', 'hold_fish', 'kneel', 'sit_idle', 'lantern_idle', 'chop']);
