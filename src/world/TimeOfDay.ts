import * as THREE from 'three/webgpu';
import { lights } from 'three/tsl';
import { uniform } from '../gfx/tsl';
import { grade, type Engine } from '../core/Engine';
import type { Assets, SkyMeta } from '../core/Assets';
import { shotMode, type TimeOfDay } from '../core/params';
import { SKY_ROTATION_Y, type SkyLighting } from './SkyLighting';
import { SkyLibrary, type SkyAsset } from './SkyLibrary';
import { SkyBlend } from './SkyBlend';
import { nightU } from './NightSky';
import { fireU, NightLights } from './NightLights';
import { mistU } from './Mist';
import { todSkyU } from './TodUniforms';
import { forestU } from './Forest';
import { propU } from './BankProps';
import { campU } from '../camp/uniforms';
import { setUnderwaterSun } from '../water/Underwater';
import { waterU } from '../water/WaterMaterial';
import type { RiverWater } from '../water/RiverWater';

export const TIMES_OF_DAY: TimeOfDay[] = ['dawn', 'morning', 'golden', 'night'];
export const isTimeOfDay = (s: unknown): s is TimeOfDay => TIMES_OF_DAY.includes(s as TimeOfDay);

type V3 = [number, number, number];

/**
 * One time of day. Radiance is "pre-exposed": every preset is authored for the same exposure (1), so bloom,
 * the mirror cap and the water's tuning keep working at night; the night sky plate is scaled down instead.
 */
export interface TodPreset {
  /** linear gain (rgb, so also a tint) of the sky plate: visible sky, image-based light and measured sky colours */
  skyGain: V3;
  /** deepening of the visible sky toward the zenith */
  zenithTint: V3;
  /** haze band along the horizon */
  band: number;
  /** sun-side glow of the sky and of its reflection, relative to the morning */
  glow: number;
  light: {
    /** compass bearing / elevation (deg) overriding the plate's sun (the night's moon) */
    bearing?: number;
    elevation?: number;
    /** lift a sun lower than this (deg): grazing light acnes and the dawn sun is below the plate's horizon */
    minElevation?: number;
    /** absolute colour, or null for the plate's measured sun colour times colorMul */
    color: V3 | null;
    colorMul?: V3;
    /** irradiance; null = the plate's measured disc times the sky gain */
    intensity: number | null;
    shadowRadius: number;
    shadowBias: number;
    shadowNormalBias: number;
  };
  /** scene.environmentIntensity (sky fill) */
  envIntensity: number;
  /** tint of the image-based light only (the visible sky keeps skyGain): the shade under the open sky */
  envTint: V3;
  /** 0..1: the pre-sunrise bands painted over the plate (todSkyU.dawn) */
  dawnSky: number;
  /** first light on the summits while the valley is in shade (todSkyU: colour x amount above `line` m) */
  alpen: { color: V3; amount: number; line: number };
  fog: { hazeDensity: number; hazeHeight: number; valleyDensity: number; valleyHeight: number; mistDensity: number; mistHeight: number; fogBrightness: number };
  water: { mirrorCap: number; valleyGain: number; calmSpread: number };
  /** `skySat`: saturation of the sky's pixels (the land's is `saturation`) */
  grade: { saturation: number; skySat: number; contrast: number; lift: V3; gain: V3; vignette: number; exposure: number };
  /** 0..1: stars, moon, campfire, lit window */
  night: number;
  /** density of part 3's volumetric valley mist relative to the morning (mistU.density x this; also todU.mist) */
  mist: number;
  /** warmth of the haze's forward lobe toward the sun or moon (SkyLighting u.hazeWarm) */
  hazeWarm: V3;
  /** colour of the volumetric mist (Mist.ts mistU.sunSat / sunTint / skyTint): its sunlit and its sky-lit share */
  mistColor: { sunSat: number; sunTint: V3; skyTint: V3 };
  /** the mist sheet's thickness (mistU.thick, x its 0.15-0.6 m scale height) and its sky-lit share (mistU.skyGain) */
  mistShape: { thick: number; skyGain: number };
  /** the camp's smoke (campU.smokeSun / smokeShade): tint of its sunlit side and of its side in shade */
  smoke: { sun: V3; shade: V3 };
  /** needles lit through from behind (forestU.translucency) */
  translucency: number;
  /** least roughness of the needles and of the far snow (forestU.needleRough, todSkyU.snowRough) */
  rough: { needles: number; snow: number };
  /** saturation of the bank boulders' moss (propU.mossSat): greens go grey-olive in low light */
  mossSat: number;
}

/** Part 1's morning, unchanged (SkyLighting.load, Engine.grade); the others were tuned against it. */
export const TOD_PRESETS: Record<TimeOfDay, TodPreset> = {
  dawn: {
    // the plate is pre-sunrise and ~3x brighter than the morning plate (Poly Haven normalises each exposure).
    // (round 2, critic r1: "an overcast day through a dirty window", the gravel as bright as the sky, the forest milky:
    // before sunrise the sky is the brightest thing, the valley floor dim, cool and drained, the forest dark)
    skyGain: [0.4, 0.41, 0.45],
    zenithTint: [0.46, 0.62, 1.06],
    band: 0.1,
    glow: 0.5,
    light: {
      // the sun has not cleared the ridges: a faint rose light, no sun shafts in the mist and no caustics (round 1's
      // 3.0 drew the morning's shafts over the fallen log, C4); the summits catch it through `alpen`
      minElevation: 3, color: [1.0, 0.6, 0.52], intensity: 0.3,
      shadowRadius: 6, shadowBias: -0.0008, shadowNormalBias: 0.12,
    },
    envIntensity: 0.75,
    envTint: [0.9, 0.96, 1.12],
    dawnSky: 1,
    // a warm peach cap on the highest summits above the blue-shaded valley (ref_02)
    // (through the dawn's thinner haze: at 1/2600 the valley aerosol veiled the peaks one flat grey, critic r1)
    alpen: { color: [1.0, 0.5, 0.34], amount: 40, line: 1150 },
    // (post-critic: the 4 m mist layer hung through the forest at trunk height, C1)
    fog: { hazeDensity: 1 / 14000, hazeHeight: 1400, valleyDensity: 1 / 4200, valleyHeight: 110, mistDensity: 0.0008, mistHeight: 3.2, fogBrightness: 0.92 },
    water: { mirrorCap: 0.14, valleyGain: 0.45, calmSpread: 0.022 },
    grade: { saturation: 0.7, skySat: 1.0, contrast: 1.08, lift: [0.0, 0.003, 0.009], gain: [0.96, 0.99, 1.05], vignette: 0.26, exposure: 1.3 },
    night: 0,
    // (round 2, critic r1: "a flat grey veil of one density hiding the water", "from above no mist at all": a thin
    // sheet a little deeper than the morning's, lit bright by the sky, so it breaks into wisps over the dark water and
    // shows from above as a pale ribbon along the river; at 2.4x as deep it stood as a grey wall in C4)
    mist: 1.0,
    mistShape: { thick: 0.9, skyGain: 1.6 },
    hazeWarm: [1.0, 0.86, 0.78],
    mistColor: { sunSat: 0.4, sunTint: [1.0, 0.92, 0.92], skyTint: [0.9, 0.96, 1.1] },
    // pale grey-blue in the valley's shade (round 1: "a dense rope of even pink-mauve")
    smoke: { sun: [0.96, 0.9, 0.9], shade: [0.7, 0.78, 0.98] },
    translucency: 0.45,
    rough: { needles: 0.8, snow: 0.75 },
    mossSat: 0.45,
  },
  morning: {
    skyGain: [1, 1, 1],
    zenithTint: [0.58, 0.74, 1.04],
    band: 0.3,
    glow: 1,
    light: {
      color: null, colorMul: [1.0, 0.95, 0.86], intensity: null,
      shadowRadius: 3.5, shadowBias: -0.0004, shadowNormalBias: 0.04,
    },
    envIntensity: 0.85,
    envTint: [1, 1, 1],
    dawnSky: 0,
    alpen: { color: [1, 1, 1], amount: 0, line: 1250 },
    fog: { hazeDensity: 1 / 9500, hazeHeight: 1400, valleyDensity: 1 / 4200, valleyHeight: 140, mistDensity: 0.0005, mistHeight: 4.0, fogBrightness: 0.92 },
    water: { mirrorCap: 0.3, valleyGain: 1, calmSpread: 0.022 },
    grade: { saturation: 1.0, skySat: 1.0, contrast: 1.13, lift: [0.0, 0.003, 0.008], gain: [1.03, 1.0, 0.96], vignette: 0.22, exposure: 1.0 },
    night: 0,
    mist: 1,
    mistShape: { thick: 1, skyGain: 1 },
    // as tuned by the forest and camp polish (SkyLighting, Mist.ts, camp/uniforms.ts defaults)
    hazeWarm: [1.0, 0.8, 0.56],
    mistColor: { sunSat: 0.45, sunTint: [1.0, 0.97, 0.9], skyTint: [1.0, 1.0, 1.0] },
    smoke: { sun: [1.0, 0.93, 0.83], shade: [0.74, 0.82, 0.97] },
    // (round 2, critic r1: in C4 "the sunlit tips of the young spruces burn out into round white cotton balls": a little
    // less light through the needles and a little less glare off them, so backlit spruce keeps thin yellow-green edges)
    translucency: 0.72,
    rough: { needles: 0.76, snow: 0.6 },
    mossSat: 1,
  },
  golden: {
    skyGain: [0.55, 0.55, 0.55],
    // pale blue overhead even into the low sun (round 1: one even cream from the top of the frame to the notch)
    zenithTint: [0.44, 0.62, 1.06],
    band: 0.25,
    glow: 0.8,
    light: {
      // (round 2, critic r1: "golden hour from above has no long shadows": at 7 deg the forest's shadow covered the whole
      // meadow; at 11 deg the forest edge rakes long stripes across it and the cabin throws its own)
      minElevation: 11, color: [1.0, 0.62, 0.32], intensity: 8.0,
      shadowRadius: 3.0, shadowBias: -0.0006, shadowNormalBias: 0.09,
    },
    // the shade under the open sky cool blue-grey, warm light only where the sun reaches (round 1: olive-brown shade, C4)
    envIntensity: 0.55,
    envTint: [0.84, 0.95, 1.2],
    dawnSky: 0,
    alpen: { color: [1, 1, 1], amount: 0, line: 1250 },
    fog: { hazeDensity: 1 / 9000, hazeHeight: 1400, valleyDensity: 1 / 4400, valleyHeight: 160, mistDensity: 0.0006, mistHeight: 4.0, fogBrightness: 0.95 },
    water: { mirrorCap: 0.3, valleyGain: 0.9, calmSpread: 0.022 },
    grade: { saturation: 1.04, skySat: 1.04, contrast: 1.12, lift: [0.002, 0.002, 0.004], gain: [1.04, 1.0, 0.94], vignette: 0.24, exposure: 1.0 },
    night: 0,
    mist: 0.6,
    mistShape: { thick: 1, skyGain: 1 },
    // the low sun gilds the thin mist and the smoke's sunlit side gold-cream; their shade grey-blue, never brown
    hazeWarm: [1.0, 0.76, 0.48],
    mistColor: { sunSat: 0.55, sunTint: [1.0, 0.93, 0.8], skyTint: [0.86, 0.94, 1.12] },
    smoke: { sun: [1.0, 0.86, 0.7], shade: [0.74, 0.8, 0.96] },
    translucency: 0.9,
    rough: { needles: 0.72, snow: 0.6 },
    mossSat: 0.6,
  },
  night: {
    // the plate is exposure-normalised (horizon ~2): scaled to a deep blue, brightest toward the horizon (round 1: "a flat
    // smoky grey")
    skyGain: [0.009, 0.02, 0.058],
    zenithTint: [0.6, 0.74, 1.0],
    band: 0.1,
    glow: 0.03,
    light: {
      // the moon, low in the west over the upstream V: the only window where both the moon and its reflection clear
      // the valley walls and part 3's forest from the pool (the sky over the pool opens at ~19-20 deg there, ~41 deg
      // over the camp in the south); a gibbous moon setting in the west in the evening.
      // (round 2, critic r1: "the moonlight has no direction", "day for night": the moon is the one hard light, as the sun
      // by day, over a sky fill at about the morning's share, so what faces the moon goes silver, the rest sinks toward
      // black and the shadows are as crisp as the morning's)
      bearing: 268, elevation: 21.5, color: [0.74, 0.84, 1.0], intensity: 0.42,
      shadowRadius: 3.5, shadowBias: -0.0004, shadowNormalBias: 0.04,
    },
    // (round 1 lifted the fill to 6.5x the visible sky: it lit every face as brightly as the moon did; at 1.1 the view
    // across the pool into the bank's shade, C4, went black. 3.5 gives the shade about the morning's ratio to the direct
    // light; the black lift keeps the bank, the log and the rocks there just readable)
    envIntensity: 2.2,
    envTint: [0.9, 0.96, 1.1],
    dawnSky: 0,
    alpen: { color: [1, 1, 1], amount: 0, line: 1250 },
    // (post-critic: thinner, so the ranges toward the moon stand dark against the sky instead of a pale moonlit haze)
    fog: { hazeDensity: 1 / 22000, hazeHeight: 1400, valleyDensity: 1 / 14000, valleyHeight: 140, mistDensity: 0.0012, mistHeight: 3.5, fogBrightness: 0.9 },
    // the moon (~30) and stars must reach the pool mirror (the day cap of 0.3 is for the HDRI sun); a night breeze
    // roughens the pool a little, so the moon breaks into a column of glints toward the eye (ref_03)
    // (post-critic: the moon's disc mirrored whole read as "a coin dropped on glass": capped, the glints carry it)
    water: { mirrorCap: 4, valleyGain: 0.03, calmSpread: 0.1 },
    // nearly monochrome land under a deep blue sky, soft (round 1's twice as strong blue black-lift greyed the whole
    // frame); a darker exposure than round 1's, so the campfire and the lit window far outshine the moonlight
    grade: { saturation: 0.34, skySat: 0.95, contrast: 1.06, lift: [0.009, 0.014, 0.028], gain: [0.9, 0.98, 1.1], vignette: 0.32, exposure: 1.75 },
    night: 1,
    mist: 0.9,
    mistShape: { thick: 1, skyGain: 1 },
    // moonlight: no warm lobe round the moon, a cool grey-blue mist and smoke (the fire still warms the smoke's foot)
    hazeWarm: [0.82, 0.9, 1.0],
    mistColor: { sunSat: 0.25, sunTint: [0.85, 0.9, 1.0], skyTint: [0.82, 0.9, 1.06] },
    smoke: { sun: [0.78, 0.86, 1.0], shade: [0.56, 0.64, 0.82] },
    // moonlight never pushes needles near white (round 1: the backlit spruces in N1 and C1 frost-flocked)
    translucency: 0.08,
    // (and the moon's forward glare off needles and snowfields seen edge-on: rough as a matte)
    rough: { needles: 0.96, snow: 0.95 },
    mossSat: 0.3,
  },
};

/** Everything a preset drives, resolved against its sky plate, so two of them can be blended. */
export interface TodState {
  sunDir: THREE.Vector3;
  lightColor: THREE.Color;
  lightIntensity: number;
  glow: number;
  horizon: THREE.Color;
  zenith: THREE.Color;
  zenithTint: THREE.Color;
  band: number;
  envIntensity: number;
  envTint: THREE.Color;
  dawnSky: number;
  /** colour x amount */
  alpen: THREE.Color;
  alpenLine: number;
  fog: TodPreset['fog'];
  shadowRadius: number;
  shadowBias: number;
  shadowNormalBias: number;
  mirrorCap: number;
  calmSpread: number;
  valley: THREE.Color;
  grade: { saturation: number; skySat: number; contrast: number; lift: THREE.Vector3; gain: THREE.Vector3; vignette: number; exposure: number };
  night: number;
  mist: number;
  mistThick: number;
  mistSkyGain: number;
  milkyRef: number;
  hazeWarm: THREE.Color;
  mistSunSat: number;
  mistSunTint: THREE.Color;
  mistSkyTint: THREE.Color;
  smokeSun: THREE.Color;
  smokeShade: THREE.Color;
  translucency: number;
  needleRough: number;
  snowRough: number;
  mossSat: number;
}

/** Uniforms other parts may read (part 3 mist, part 9 sound by `state.night`). */
export const todU = {
  /** valley mist density relative to the morning (thickest at dawn) */
  mist: uniform(1),
  /** 0..1 night amount */
  night: uniform(0),
};

const WATER_VALLEY = new THREE.Color(0.03, 0.038, 0.036);
/** part 3's tuned morning mist density */
const MIST_BASE = mistU.density.value;
const lum = (c: THREE.Color) => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
const smooth = (x: number) => x * x * (3 - 2 * x);

function dirFromBearing(bearingDeg: number, elevDeg: number, out = new THREE.Vector3()) {
  const b = THREE.MathUtils.degToRad(bearingDeg), e = THREE.MathUtils.degToRad(elevDeg);
  return out.set(Math.sin(b) * Math.cos(e), Math.sin(e), -Math.cos(b) * Math.cos(e));
}

/** Where the night preset puts the moon (also its light). */
export const TOD_MOON_DIR = dirFromBearing(TOD_PRESETS.night.light.bearing!, TOD_PRESETS.night.light.elevation!);

function bearingOf(d: THREE.Vector3) {
  return { bearing: (THREE.MathUtils.radToDeg(Math.atan2(d.x, -d.z)) + 360) % 360, elevation: THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(d.y, -1, 1))) };
}

function resolve(p: TodPreset, sky: SkyAsset): TodState {
  const m = sky.meta;
  const gain = new THREE.Color(...p.skyGain);
  const rotDeg = m.rotationYDeg ?? THREE.MathUtils.radToDeg(SKY_ROTATION_Y);
  // background/environment sample with the inverse rotation, so the visible sun sits at R * d
  const sunDir = new THREE.Vector3().fromArray(m.sunDir).applyEuler(new THREE.Euler(0, THREE.MathUtils.degToRad(rotDeg), 0)).normalize();
  const L = p.light;
  if (L.bearing !== undefined || L.elevation !== undefined) {
    const cur = bearingOf(sunDir);
    dirFromBearing(L.bearing ?? cur.bearing, L.elevation ?? cur.elevation, sunDir);
  }
  if (L.minElevation !== undefined) {
    const cur = bearingOf(sunDir);
    if (cur.elevation < L.minElevation) dirFromBearing(cur.bearing, L.minElevation, sunDir);
  }
  const lightColor = L.color ? new THREE.Color(...L.color) : new THREE.Color().fromArray(m.sunColor).multiply(new THREE.Color(...(L.colorMul ?? [1, 1, 1])));
  const lightIntensity = L.intensity ?? Math.max(...m.sunIrradiance) * lum(gain);
  const horizon = new THREE.Color().fromArray(m.horizonColor).multiply(gain);
  const zenith = new THREE.Color().fromArray(m.zenithColor).multiply(gain);
  return {
    sunDir, lightColor, lightIntensity, glow: p.glow, horizon, zenith,
    zenithTint: new THREE.Color(...p.zenithTint), band: p.band, envIntensity: p.envIntensity, envTint: new THREE.Color(...p.envTint),
    dawnSky: p.dawnSky, alpen: new THREE.Color(...p.alpen.color).multiplyScalar(p.alpen.amount), alpenLine: p.alpen.line, fog: { ...p.fog },
    shadowRadius: L.shadowRadius, shadowBias: L.shadowBias, shadowNormalBias: L.shadowNormalBias,
    mirrorCap: p.water.mirrorCap, calmSpread: p.water.calmSpread, valley: WATER_VALLEY.clone().multiplyScalar(p.water.valleyGain),
    grade: {
      saturation: p.grade.saturation, skySat: p.grade.skySat, contrast: p.grade.contrast, lift: new THREE.Vector3(...p.grade.lift),
      gain: new THREE.Vector3(...p.grade.gain), vignette: p.grade.vignette, exposure: p.grade.exposure,
    },
    night: p.night, mist: p.mist, mistThick: p.mistShape.thick, mistSkyGain: p.mistShape.skyGain,
    // the night plate's dark sky is ~0.47 (its median) before the gain
    milkyRef: 0.47 * lum(gain),
    hazeWarm: new THREE.Color(...p.hazeWarm), mistSunSat: p.mistColor.sunSat, mistSunTint: new THREE.Color(...p.mistColor.sunTint),
    mistSkyTint: new THREE.Color(...p.mistColor.skyTint), smokeSun: new THREE.Color(...p.smoke.sun), smokeShade: new THREE.Color(...p.smoke.shade),
    translucency: p.translucency, needleRough: p.rough.needles, snowRough: p.rough.snow, mossSat: p.mossSat,
  };
}

function mixState(a: TodState, b: TodState, k: number): TodState {
  const n = (x: number, y: number) => x + (y - x) * k;
  const c = (x: THREE.Color, y: THREE.Color) => x.clone().lerp(y, k);
  const fog = {} as TodPreset['fog'];
  for (const key of Object.keys(a.fog) as (keyof TodPreset['fog'])[]) fog[key] = n(a.fog[key], b.fog[key]);
  // the light swings across the sky and dims on the way (so the swing does not read as a searchlight)
  const sunDir = a.sunDir.clone().lerp(b.sunDir, k).normalize();
  if (sunDir.lengthSq() < 1e-6) sunDir.copy(k < 0.5 ? a.sunDir : b.sunDir);
  const dip = 1 - 0.55 * Math.sin(Math.PI * k);
  return {
    sunDir, lightColor: c(a.lightColor, b.lightColor), lightIntensity: n(a.lightIntensity, b.lightIntensity) * dip,
    glow: n(a.glow, b.glow), horizon: c(a.horizon, b.horizon), zenith: c(a.zenith, b.zenith),
    zenithTint: c(a.zenithTint, b.zenithTint), band: n(a.band, b.band), envIntensity: n(a.envIntensity, b.envIntensity),
    envTint: c(a.envTint, b.envTint), dawnSky: n(a.dawnSky, b.dawnSky), alpen: c(a.alpen, b.alpen), alpenLine: n(a.alpenLine, b.alpenLine), fog,
    shadowRadius: n(a.shadowRadius, b.shadowRadius), shadowBias: n(a.shadowBias, b.shadowBias), shadowNormalBias: n(a.shadowNormalBias, b.shadowNormalBias),
    mirrorCap: Math.exp(n(Math.log(a.mirrorCap), Math.log(b.mirrorCap))), calmSpread: n(a.calmSpread, b.calmSpread), valley: c(a.valley, b.valley),
    grade: {
      saturation: n(a.grade.saturation, b.grade.saturation), skySat: n(a.grade.skySat, b.grade.skySat), contrast: n(a.grade.contrast, b.grade.contrast),
      lift: a.grade.lift.clone().lerp(b.grade.lift, k), gain: a.grade.gain.clone().lerp(b.grade.gain, k),
      vignette: n(a.grade.vignette, b.grade.vignette), exposure: n(a.grade.exposure, b.grade.exposure),
    },
    night: n(a.night, b.night), mist: n(a.mist, b.mist), mistThick: n(a.mistThick, b.mistThick), mistSkyGain: n(a.mistSkyGain, b.mistSkyGain),
    milkyRef: n(a.milkyRef, b.milkyRef),
    hazeWarm: c(a.hazeWarm, b.hazeWarm), mistSunSat: n(a.mistSunSat, b.mistSunSat), mistSunTint: c(a.mistSunTint, b.mistSunTint),
    mistSkyTint: c(a.mistSkyTint, b.mistSkyTint), smokeSun: c(a.smokeSun, b.smokeSun), smokeShade: c(a.smokeShade, b.smokeShade),
    translucency: n(a.translucency, b.translucency), needleRough: n(a.needleRough, b.needleRough), snowRough: n(a.snowRough, b.snowRough),
    mossSat: n(a.mossSat, b.mossSat),
  };
}

export type TodPhase = 'loading' | 'start' | 'end';

/** ?todbudget=off: no night lights, stars or moon (GPU cost A/B only) */
const Q = new URLSearchParams(location.search);
const BUDGET_OFF = Q.get('todbudget') === 'off';
/** debug A/B: ?todenv=static lights with the plate's own probe through three's lazy PMREM (no blend, no fades) */
const ENV_STATIC = Q.get('todenv') === 'static';
/** debug A/B: ?todfire=all lets the campfire light reach every lit material (see restrictFireLight) */
const FIRE_ALL = Q.get('todfire') === 'all';

/**
 * Time of day: dawn, morning, golden hour and night, with ~2 s cross-fades of sky, image-based light, sun/moon,
 * fog, grade and night lights. Replaces SkyLighting.load() (it owns scene.background/environment and the sun's
 * values) and keeps SkyLighting's uniforms as the shared interface, so the fog, water, terrain and forest follow.
 * The night replaces the sun with the moon in the same shadow-casting light.
 */
export class TimeOfDayController {
  readonly library: SkyLibrary;
  blend!: SkyBlend;
  nightLights: NightLights | null = null;
  current: TimeOfDay = 'morning';
  /** the time of day being faded in, or null when settled */
  target: TimeOfDay | null = null;
  /** blended state of this frame (read only) */
  state!: TodState;
  duration = 2.0;
  /** re-render the blended light probe every n frames during a fade (its PMREM is rebuilt each time) */
  envEvery = 3;
  /** redraw the sun's 4096² shadow map every n frames while the light moves (part 3 redraws it every 3rd anyway) */
  shadowEvery = 2;
  private from!: TodState;
  private to: TodState | null = null;
  private queued: TimeOfDay | null = null;
  private k = 0;
  private last = 0;
  private frame = 0;
  private water: RiverWater | null = null;
  private lastSunDir = new THREE.Vector3();
  private listeners: ((tod: TimeOfDay, phase: TodPhase) => void)[] = [];
  private prefetchAt = shotMode ? -1 : 200;
  /**
   * The presets' thin 4 m water-fog density is relative to the morning's: with part 3's volumetric mist World.ts
   * lowers that layer (0.0002 instead of 0.0005), and the morning must stay as tuned (set in attachWorld)
   */
  private mistFogScale = 1;

  constructor(private engine: Engine, readonly sky: SkyLighting, assets: Assets) {
    this.library = new SkyLibrary(engine.renderer, assets);
    // before the water's updater: the mirror renders this frame's sky
    engine.addUpdater((_dt, t) => this.update(t));
    (window as unknown as { __TOD: unknown }).__TOD = {
      set: (tod: TimeOfDay, instant = false) => this.set(tod, { instant }),
      get: () => ({ current: this.current, target: this.target, k: this.k }),
      ready: (tod: TimeOfDay) => this.library.isReady(tod),
      presets: TOD_PRESETS,
      /** the shared uniforms the presets drive (tuning from the console or a capture script) */
      u: { grade, nightU, mistU, todSkyU, forestU, propU, campU, waterU, fireU, todU, sky: sky.u },
      /** debug camera for captures: absolute eye, target and vertical field of view */
      view: (x: number, y: number, z: number, tx: number, ty: number, tz: number, fov?: number) => {
        const cam = engine.camera;
        cam.position.set(x, y, z);
        cam.lookAt(tx, ty, tz);
        if (fov) cam.fov = fov;
        cam.updateProjectionMatrix();
      },
    };
  }

  onChange(fn: (tod: TimeOfDay, phase: TodPhase) => void) {
    this.listeners.push(fn);
  }

  private emit(tod: TimeOfDay, phase: TodPhase) {
    for (const f of this.listeners) f(tod, phase);
  }

  /** Initial sky (instead of SkyLighting.load): sets scene background/environment and the sun before the water is built. */
  async load(tod: TimeOfDay) {
    const t0 = isTimeOfDay(tod) ? tod : 'morning';
    const asset = await this.library.get(t0);
    const p = TOD_PRESETS[t0];
    this.blend = new SkyBlend({ asset, rotY: asset.meta.rotationYDeg ?? THREE.MathUtils.radToDeg(SKY_ROTATION_Y), gain: new THREE.Color(...p.skyGain) });
    const scene = this.engine.scene;
    scene.background = null;
    scene.backgroundNode = this.blend.backgroundNode(this.sky, !BUDGET_OFF);
    scene.backgroundIntensity = 1.0;
    if (!ENV_STATIC) this.blend.renderEnv(this.engine.renderer);
    scene.environment = ENV_STATIC ? asset.env : this.blend.envTexture;
    scene.environmentRotation.set(0, ENV_STATIC ? THREE.MathUtils.degToRad(asset.meta.rotationYDeg ?? 76) : 0, 0);
    scene.backgroundRotation.set(0, 0, 0);
    this.current = t0;
    this.from = resolve(p, asset);
    this.state = this.from;
    this.apply(this.state);
    this.setMeta(asset, this.state);
  }

  /**
   * At the end of buildWorld (everything built, before the first frame): the night lights, the water's
   * sky-dependent uniforms and the campfire light's reach.
   */
  async attachWorld(o: { water: RiverWater | null; groundY: (x: number, z: number) => number }) {
    this.water = o.water;
    this.mistFogScale = this.sky.u.mistDensity.value / TOD_PRESETS.morning.fog.mistDensity;
    if (!BUDGET_OFF) {
      this.nightLights = new NightLights(o.groundY);
      this.engine.scene.add(this.nightLights.group);
    }
    // the water's lighting folds the sun's sparkles and body light into its colour: lit by the campfire it would
    // tint the moon's glitter path orange. It reflects the fire through the pool mirror instead.
    if (o.water) o.water.material.lightsNode = lights([this.sky.sun]);
    if (this.nightLights && !FIRE_ALL) this.restrictFireLight();
    this.apply(this.state);
  }

  /**
   * A point light costs every lit fragment of every material that lists it, day and night (intensity 0 still runs):
   * measured ~1-1.7 ms with the forest's overdrawn needle cards and the grass. So the fire only reaches the corridor
   * ground (the camp's ground) and whatever is added after this call (part 4's cabin, campfire set and props);
   * every other lit material already in the scene (trees, grass, bank props, rocks, far terrain) gets the sun alone.
   */
  private restrictFireLight() {
    const sunOnly = lights([this.sky.sun]);
    const seen = new Set<THREE.Material>();
    let n = 0;
    this.engine.scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh || mesh.name === 'terrain-core') return;
      for (const m of (Array.isArray(mesh.material) ? mesh.material : [mesh.material]) as (THREE.Material & { lights?: boolean; lightsNode?: unknown })[]) {
        if (!m || seen.has(m)) continue;
        seen.add(m);
        if (m.lights === false || m.lightsNode) continue;
        m.lightsNode = sunOnly;
        n++;
      }
    });
    console.info('time of day: campfire light kept off', n, 'materials');
  }

  /**
   * Part 4 hooks (call before the first frame, e.g. right after adding the GLB to the scene). `windowLight`: also a
   * warm point light there (spill on the porch); it costs like the campfire light on every lit material, day and
   * night, so it is off unless asked for.
   */
  registerCabin(root: THREE.Object3D, o: { windowLight?: THREE.Vector3; windowCandela?: number } = {}) {
    if (!this.nightLights) return 0;
    if (o.windowLight) {
      this.nightLights.addWindowLight(o.windowLight, o.windowCandela);
      // the corridor ground (most of every frame) keeps the sun and the fire only: the window's spill is for the cabin,
      // its porch and the woodpile (part 4's meshes, added after attachWorld, see restrictFireLight)
      const core = this.engine.scene.getObjectByName('terrain-core') as THREE.Mesh | undefined;
      const m = core?.material as (THREE.Material & { lightsNode?: unknown }) | undefined;
      if (m && !FIRE_ALL) m.lightsNode = lights([this.sky.sun, this.nightLights.fire]);
    }
    return this.nightLights.registerWindows(root);
  }

  registerCampfire(root: THREE.Object3D) {
    return this.nightLights ? this.nightLights.registerCampfire(root) : 0;
  }

  isReady(tod: TimeOfDay) {
    return this.library.isReady(tod);
  }

  /** Load the other skies in the background (about 55 MB of VRAM each once resident). */
  prefetchOthers() {
    return this.library.prefetch(TIMES_OF_DAY.filter((t) => t !== this.current));
  }

  /** Fade to a time of day (queued if a fade is running). `instant` for captures. */
  async set(tod: TimeOfDay, o: { instant?: boolean; duration?: number } = {}) {
    if (!isTimeOfDay(tod)) return;
    if (this.target) {
      this.queued = tod === this.target ? null : tod;
      return;
    }
    if (tod === this.current && !o.instant) return;
    if (!this.library.isReady(tod)) this.emit(tod, 'loading');
    let asset: SkyAsset;
    try {
      asset = await this.library.get(tod);
    } catch (e) {
      console.error('time of day: sky failed to load', tod, e);
      this.emit(this.current, 'end');
      return;
    }
    if (this.target) {
      // another fade started while this sky loaded
      this.queued = tod;
      return;
    }
    const p = TOD_PRESETS[tod];
    const slot = { asset, rotY: asset.meta.rotationYDeg ?? THREE.MathUtils.radToDeg(SKY_ROTATION_Y), gain: new THREE.Color(...p.skyGain) };
    const to = resolve(p, asset);
    if (o.instant) {
      this.blend.setSlot('A', slot);
      this.blend.setSlot('B', slot);
      this.blend.u.blend.value = 0;
      this.current = tod;
      this.from = to;
      this.state = to;
      this.apply(to);
      this.setMeta(asset, to);
      this.blend.renderEnv(this.engine.renderer);
      this.emit(tod, 'end');
      return;
    }
    this.duration = o.duration ?? this.duration;
    this.blend.setSlot('B', slot);
    this.from = this.state;
    this.to = to;
    this.target = tod;
    this.k = 0;
    this.last = performance.now();
    this.emit(tod, 'start');
  }

  private update(t: number) {
    this.frame++;
    if (this.frame === this.prefetchAt) void this.prefetchOthers();
    nightU.time.value = t;
    if (this.target && this.to) {
      // wall-clock progress: shot mode freezes the engine clock (dt = 0), a fade still has to finish
      const now = performance.now();
      this.k = Math.min(1, this.k + (now - this.last) / 1000 / Math.max(0.05, this.duration));
      this.last = now;
      const k = smooth(this.k);
      this.blend.u.blend.value = k;
      this.state = mixState(this.from, this.to, k);
      this.apply(this.state);
      if (this.k >= 1) {
        const tod = this.target;
        this.blend.settle();
        this.current = tod;
        this.from = this.to;
        this.state = this.to;
        this.apply(this.state);
        this.setMeta(this.blend.slotA.asset, this.state);
        this.blend.renderEnv(this.engine.renderer);
        this.target = null;
        this.to = null;
        this.emit(tod, 'end');
        const q = this.queued;
        this.queued = null;
        if (q && q !== this.current) void this.set(q);
      } else if (this.frame % this.envEvery === 0) {
        this.blend.renderEnv(this.engine.renderer);
      }
    }
    this.nightLights?.update(t, this.state?.night ?? 0);
  }

  /** Push a state into SkyLighting's uniforms and light, the sky blend, the grade, the water and the night sky. */
  private apply(s: TodState) {
    const sky = this.sky, sun = sky.sun, scene = this.engine.scene;
    sky.sunDir.copy(s.sunDir);
    sky.u.sunDir.value.copy(s.sunDir);
    // the sky glow (background and water reflection) is sunColor x shape; the fog uses sunColor x intensity:
    // scale the colour by the glow and divide the intensity, so the in-scattered light stays physical
    const g = Math.max(s.glow, 1e-3);
    sky.u.sunColor.value.copy(s.lightColor).multiplyScalar(g);
    sky.u.sunIntensity.value = s.lightIntensity / g;
    sky.u.horizon.value.copy(s.horizon);
    sky.u.zenith.value.copy(s.zenith);
    const f = s.fog;
    sky.u.hazeDensity.value = f.hazeDensity;
    sky.u.hazeHeight.value = f.hazeHeight;
    sky.u.valleyDensity.value = f.valleyDensity;
    sky.u.valleyHeight.value = f.valleyHeight;
    sky.u.mistDensity.value = f.mistDensity * this.mistFogScale;
    sky.u.mistHeight.value = f.mistHeight;
    sky.u.fogBrightness.value = f.fogBrightness;
    sun.color.copy(s.lightColor);
    sun.intensity = s.lightIntensity;
    const focus = sun.target.position;
    sun.position.copy(focus).addScaledVector(s.sunDir, 200);
    // part 3 redraws the (otherwise fixed) sun's shadow map only every few frames: redraw it when the light moves,
    // every `shadowEvery` frames during a fade and always on the last step
    if (this.lastSunDir.distanceToSquared(s.sunDir) > 1e-10 && (!this.target || this.frame % this.shadowEvery === 0 || this.k >= 1)) {
      sun.shadow.needsUpdate = true;
      this.lastSunDir.copy(s.sunDir);
    }
    // the presets' PCF radii were tuned on a 4096² map; the polish round's smaller map keeps the same softening in metres
    sun.shadow.radius = s.shadowRadius * (sun.shadow.mapSize.x / 4096);
    sun.shadow.bias = s.shadowBias;
    sun.shadow.normalBias = s.shadowNormalBias;
    scene.environmentIntensity = s.envIntensity;
    if (this.blend) {
      this.blend.u.zenithTint.value.copy(s.zenithTint);
      this.blend.u.band.value = s.band;
      this.blend.u.milkyRef.value = Math.max(s.milkyRef, 1e-4);
      this.blend.u.envTint.value.copy(s.envTint);
    }
    todSkyU.dawn.value = s.dawnSky;
    todSkyU.alpen.value.copy(s.alpen);
    todSkyU.alpenLine.value = s.alpenLine;
    todSkyU.sunDir.value.copy(s.sunDir);
    todSkyU.ridge.value.copy(s.lightColor).multiplyScalar(s.lightIntensity * s.night);
    grade.skySat.value = s.grade.skySat;
    grade.saturation.value = s.grade.saturation;
    grade.contrast.value = s.grade.contrast;
    grade.lift.value.copy(s.grade.lift);
    grade.gain.value.copy(s.grade.gain);
    grade.vignette.value = s.grade.vignette;
    grade.exposure.value = s.grade.exposure;
    grade.keepWarm.value = s.night;
    nightU.amount.value = s.night;
    nightU.moonDir.value.copy(TOD_MOON_DIR);
    todU.mist.value = s.mist;
    mistU.density.value = MIST_BASE * s.mist;
    mistU.thick.value = s.mistThick;
    mistU.skyGain.value = s.mistSkyGain;
    forestU.translucency.value = s.translucency;
    // (post-critic) moonlit spruce nearly black; broken water and the waterline faint at night; the moon's glitter a
    // soft broken column; the dawn mist thin near the eye and densest toward the far bank, without steam puffs
    forestU.needleGain.value = 1 - 0.5 * s.night;
    waterU.haze.value = 0.04 + 0.46 * s.night;
    waterU.foamLight.value = 1 - 0.7 * s.night;
    waterU.edgeLight.value = 1 - 0.85 * s.night;
    mistU.clearNear.value = 1.2 + 1.8 * s.dawnSky;
    mistU.clearFar.value = 4.5 + 10.5 * s.dawnSky;
    mistU.plume.value = 1 - 0.85 * s.dawnSky;
    forestU.needleRough.value = s.needleRough;
    todSkyU.snowRough.value = s.snowRough;
    propU.mossSat.value = s.mossSat;
    waterU.calmSpread.value = s.calmSpread;
    sky.u.hazeWarm.value.copy(s.hazeWarm);
    mistU.sunSat.value = s.mistSunSat;
    mistU.sunTint.value.copy(s.mistSunTint);
    mistU.skyTint.value.copy(s.mistSkyTint);
    campU.smokeSun.value.set(s.smokeSun.r, s.smokeSun.g, s.smokeSun.b);
    campU.smokeShade.value.set(s.smokeShade.r, s.smokeShade.g, s.smokeShade.b);
    todU.night.value = s.night;
    setUnderwaterSun(s.sunDir);
    waterU.mirrorCap.value = s.mirrorCap;
    waterU.valley.value.copy(s.valley);
    // sky fill for foam and the water body: mean of horizon and zenith at the scene's environment intensity
    waterU.skyAmbient.value.copy(s.horizon).add(s.zenith).multiplyScalar(0.5 * s.envIntensity);
    const w = this.water;
    if (w) {
      w.events.droplets.sunColor.value.copy(s.lightColor).multiplyScalar(s.lightIntensity * 0.08);
      w.events.droplets.skyColor.value.copy(waterU.skyAmbient.value);
    }
  }

  /** sky.meta as SkyLighting.load left it (RiverWater reads it once at construction) */
  private setMeta(asset: SkyAsset, s: TodState) {
    const m: SkyMeta = {
      ...asset.meta,
      sunDir: asset.meta.sunDir,
      horizonColor: [s.horizon.r, s.horizon.g, s.horizon.b],
      zenithColor: [s.zenith.r, s.zenith.g, s.zenith.b],
    };
    this.sky.meta = m;
  }
}
