import * as THREE from 'three/webgpu';
import {
  abs, atan, attribute, cameraPosition, dot, exp, float, hash, length, max, min, mix, normalize, normalMap, normalWorld,
  normalWorldGeometry, positionWorld, pow, reflect, smoothstep, step, texture, uniform, uv, vec2, vec3, type N,
} from '../gfx/tsl';
import { materialEmissive } from 'three/tsl';
import { installUnderwater } from '../water/Underwater';
import type { RiverField } from '../water/RiverField';
import type { SkyLighting } from '../world/SkyLighting';
import { POOL_LEVEL } from '../world/River';
import { tnoise } from '../gfx/noiseTex';
import { campU } from './uniforms';
import { indoorSky } from './indoor';

/**
 * Weathered timber and stone for the camp. The Blender exports set metallic 1.0 next to Poly Haven ARM maps whose
 * metal channel is ~0 with speckles up to 0.16 (and the fire-pit map is all-metal behind a 0 factor): every wood
 * and stone surface here gets metalness 0; only the axe head, the bucket hoops and the lantern keep their metal
 * masks. Albedo is calibrated per material to a target mean linear luminance next to the scene's own references
 * (bank logs 0.09, bed and bank stones 0.16-0.17, bleached driftwood ~0.3; DECISIONS 'Albedo calibration'), the
 * scans partly desaturated toward grey, roughness spread over a weathered range (the rough_wood map alone averages
 * 0.51, which read as varnish). Per-material values are uniforms, so most materials share one program.
 * Vertex attribute aCamp = (tint multiplier rgb, ground-contact occlusion), baked at load.
 */

const LUM = vec3(0.2126, 0.7152, 0.0722);
const lum3 = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** mean linear luminance of each base colour map (measured from the GLBs' WebP maps, sRGB decoded) */
const TEX_LUM: Record<string, number> = {
  rough_wood_Diffuse_1k: 0.1319,
  wood_planks_grey_Diffuse_2k: 0.0619,
  wood_planks_grey_Diffuse_1k: 0.061,
  endgrain_weathered_512_Diffuse: 0.1065,
  endgrain_weathered_Diffuse: 0.1066,
  rock_wall_08_Diffuse_2k: 0.0753,
  rock_01_Diffuse_1k: 0.1551,
  clay_plaster_Diffuse_1k: 0.1127,
  graniteware_enamel_Diffuse: 0.0395,
  campfire_ash_Diffuse: 0.1704,
  charred_wood_Diffuse: 0.0045,
  bark_brown_02_Diffuse_1k: 0.1071,
  endgrain_fresh_512_Diffuse: 0.1144,
  stone_fire_pit_Diffuse_1k: 0.1104,
  wooden_axe_diff_1k: 0.1286,
  wooden_bucket_01_diff_1k: 0.0671,
  endgrain_block_512_Diffuse: 0.1088,
  Lantern_01_brass_diff_1k: 0.0733,
};
/** mean of each ARM map's occlusion (red), to keep the target luminance when part of it goes into the albedo */
const AO_MEAN: Record<string, number> = {
  rough_wood_arm_1k: 0.78, wood_planks_grey_arm_2k: 0.97, wood_planks_grey_arm_1k: 0.97, endgrain_weathered_512_arm: 0.98,
  endgrain_weathered_arm: 0.98, rock_wall_08_arm_2k: 0.85, rock_01_arm_1k: 0.77, clay_plaster_arm_1k: 0.97,
  bark_brown_02_arm_1k: 0.59, endgrain_fresh_512_arm: 0.98, endgrain_block_512_arm: 0.98,
};

type Kind = 'surface' | 'dock' | 'embers' | 'fire';
interface Calib {
  /** target mean linear albedo luminance */
  lum: number;
  /** how much of the scan's colour is kept (0 grey .. 1 as scanned) */
  sat: number;
  /** range the map's roughness (green) is spread over */
  rough: [number, number];
  /** colour bias (luminance-normalised at build) */
  bias?: [number, number, number];
  /** large-scale weathering: brightness +- this fraction in ~2 m patches, silver vs browner */
  vary?: number;
  /** rain streaks running down vertical faces (darker bands, fraction) */
  streak?: number;
  /** contrast of the scan's luminance around its mean (1 = as scanned): grain, checks and knots read in full sun */
  contrast?: number;
  /** share of the map's occlusion multiplied into the albedo (dirt in the grain; direct sun ignores aoNode) */
  cavity?: number;
  /** strength of the map's occlusion (red) */
  ao?: number;
  normal?: number;
  kind?: Kind;
  /** keep the map's metal mask (blue) */
  metal?: boolean;
  /** mean luminance of the base map when it is not in TEX_LUM (the project's rock scans) */
  texLum?: number;
  /** rust-brown streaks along the grain of horizontal logs (fraction of the surface) */
  grain?: number;
  /** damp, mud and moss near the ground (from the baked ground-contact term) */
  foot?: number;
  /** lichen spots and moss patches (old shakes) */
  lichen?: number;
  /** chinking fallen out in patches (dark gaps) and mud-brown repairs */
  patchy?: number;
  /** soot: world heights where it starts and is full (start > full: blackened downward), and its strength */
  soot?: [number, number, number];
  /** soot running down below its line in streaks (m) */
  sootStreak?: number;
  /** the metal mask turned to rust (old bucket hoops) */
  rust?: number;
  /** polished and darkened by hands round the door latch (campU.latch) */
  worn?: number;
  /** fire program: 1 = the ash bed under the logs (procedural ash and charcoal), 0 = burnt wood */
  bed?: number;
  /** fire program: burnt fully within [0] m of the ring's centre, not at all beyond [1] */
  burn?: [number, number];
  /** fire program: log-space uvs (charred from the char front along the log, blocks in rows round it) */
  logs?: number;
  /** deep dark checks along the grain (the GLB's logs and planks and the generated wood run their grain along uv v) */
  checks?: number;
  /** bold rust-brown stains along the grain (uv) */
  stain?: number;
  /** end grain (disc-mapped uv, pith at 0.5): radial checks opening toward the rim */
  rcheck?: number;
  /** end grain: weathered growth rings standing proud */
  rings?: number;
  /** end grain: this many wide checks round the disc (a few dropped), black inside, widening toward the rim */
  rcount?: number;
  /** grime and pale ash smudges (stump tops by the fire) */
  ashy?: number;
  /** brown-black crust and drips (the coffee pot) */
  crust?: number;
  /** graniteware: white flecks in the enamel */
  speckle?: number;
}

const CAL: Record<string, Calib> = {
  // cabin: grey-brown weathered logs with rust streaks, darker ragged end grain, dark weathered shakes with lichen,
  // chinking sunk back in shadow with gaps, mixed fieldstone
  // (round 3: silver logs with bold rust-brown stains and deep dark checks; the ends darker than the logs, cracked)
  LogWood: { lum: 0.125, sat: 0.26, rough: [0.55, 1.0], bias: [0.98, 0.98, 0.99], vary: 0.3, streak: 0.12, contrast: 1.7, cavity: 0.6, normal: 1.25, grain: 0.18, stain: 0.8, checks: 0.9, foot: 1 },
  // (scene fix, critic polish r1: "a stack of identical discs": weathered grey, wide radial checks)
  LogEnds: { lum: 0.045, sat: 0.1, rough: [0.6, 1.0], bias: [1.0, 0.97, 0.94], vary: 0.3, contrast: 1.3, cavity: 0.85, normal: 1.5, rcheck: 1.5, rcount: 6, foot: 1 },
  RoofShakes: { lum: 0.056, sat: 0.3, rough: [0.7, 1.0], bias: [0.97, 0.98, 0.97], vary: 0.32, contrast: 1.7, cavity: 0.7, normal: 1.5, lichen: 1 },
  Boards: { lum: 0.12, sat: 0.4, rough: [0.45, 1.0], vary: 0.16, streak: 0.14, contrast: 1.3, cavity: 0.4, worn: 1, foot: 0.6, checks: 0.6 },
  Chinking: { lum: 0.07, sat: 0.25, rough: [0.7, 1.0], bias: [1.0, 0.95, 0.86], vary: 0.2, ao: 1, patchy: 1, foot: 1 },
  Mortar: { lum: 0.045, sat: 0.2, rough: [0.8, 1.0], bias: [1.0, 0.97, 0.92], vary: 0.2, patchy: 0.4 },
  // the chimney's fieldstone (round 3: darker, browner, more mixed; lichen spots, moss at the foot, soot at the top)
  PierStone: { lum: 0.092, sat: 0.42, rough: [0.6, 1.0], bias: [1.0, 0.97, 0.94], vary: 0.2, contrast: 1.45, cavity: 0.6, foot: 1, lichen: 0.45 },
  ChimneyStone: { lum: 0.13, sat: 0.45, rough: [0.55, 1.0], bias: [0.98, 0.97, 0.98], vary: 0.1, contrast: 1.35, cavity: 0.5, normal: 1.3, foot: 1 },
  // the project's rock scans as fieldstones (piers, steps, the fire ring); per-stone colour comes in the vertex tint
  rock_moss_01_r03: { lum: 0.13, sat: 0.35, rough: [0.65, 1.0], texLum: lum3(0.0874, 0.0593, 0.031), vary: 0.1, contrast: 1.3, foot: 1 },
  rock_moss_02_r07: { lum: 0.13, sat: 0.35, rough: [0.65, 1.0], texLum: 0.075, vary: 0.1, contrast: 1.3, foot: 1 },
  rock_moss_02_r11: { lum: 0.13, sat: 0.35, rough: [0.65, 1.0], texLum: lum3(0.1044, 0.0903, 0.0582), vary: 0.1, contrast: 1.3, foot: 1 },
  namaqualand_boulder_05: { lum: 0.14, sat: 0.4, rough: [0.6, 1.0], texLum: 0.12, vary: 0.1, contrast: 1.25, foot: 1 },
  rock_07: { lum: 0.14, sat: 0.45, rough: [0.55, 1.0], texLum: lum3(0.187, 0.1211, 0.0731), contrast: 1.2, foot: 0.6 },
  stone_01: { lum: 0.14, sat: 0.45, rough: [0.55, 1.0], texLum: lum3(0.2245, 0.1186, 0.0717), contrast: 1.2, foot: 0.6 },
  rock_09: { lum: 0.13, sat: 0.45, rough: [0.55, 1.0], texLum: lum3(0.0851, 0.0506, 0.0242), contrast: 1.2, foot: 0.6 },
  // dock: sun-bleached silver planks over a darker frame; wet and slimy toward and below the water (from the pool level)
  DockPlanks: { lum: 0.24, sat: 0.22, rough: [0.55, 1.0], bias: [0.98, 0.99, 1.0], vary: 0.18, normal: 1.3, contrast: 1.55, cavity: 0.6, kind: 'dock', checks: 0.8 },
  DockTimber: { lum: 0.15, sat: 0.35, rough: [0.55, 1.0], bias: [1.0, 0.98, 0.95], vary: 0.16, contrast: 1.45, cavity: 0.5, kind: 'dock', checks: 0.7 },
  EndGrainWeathered: { lum: 0.08, sat: 0.35, rough: [0.45, 1.0], contrast: 1.2, kind: 'dock', rcheck: 0.9 },
  // campfire set and props
  FirePitStone: { lum: 0.13, sat: 0.4, rough: [0.45, 1.0], vary: 0.08 },
  Bark: { lum: 0.085, sat: 0.75, rough: [0.0, 1.0], ao: 0.8, cavity: 0.4, foot: 0.5 },
  EndGrain: { lum: 0.1, sat: 0.3, rough: [0.45, 1.0], contrast: 1.5, bias: [1.0, 0.97, 0.93], cavity: 0.4 },
  SplitWood: { lum: 0.11, sat: 0.32, rough: [0.6, 1.0], bias: [1.0, 0.96, 0.9], contrast: 1.4, cavity: 0.5, vary: 0.2 },
  WeatheredWood: { lum: 0.11, sat: 0.45, rough: [0.55, 1.0], contrast: 1.3, cavity: 0.4 },
  EndGrainBlock: { lum: 0.1, sat: 0.3, rough: [0.4, 1.0], contrast: 1.4, bias: [1.0, 0.97, 0.94] },
  // stump seats: bark weathered grey and loose; tops rough and grey with wide radial cracks, rings standing proud,
  // grime and ash (round 2's read as concrete discs)
  StumpBark: { lum: 0.09, sat: 0.22, rough: [0.4, 1.0], ao: 0.9, cavity: 0.6, contrast: 1.55, bias: [1.0, 0.98, 0.95], vary: 0.25, foot: 1 },
  // (round 4 wip: a clean blue-grey disc with even rings: warmer and darker against the sky light it faces, the
  // scan's rings flattened, wide checks, proud rings at uneven spacing)
  StumpTop: { lum: 0.057, sat: 0.22, rough: [0.72, 1.0], contrast: 1.05, bias: [1.0, 0.9, 0.77], cavity: 0.7, rcheck: 1.2, rings: 1.4, ashy: 1, rcount: 7 },
  Ash: { lum: 0.22, sat: 0.2, rough: [0.0, 1.0] },
  Charred: { lum: 0.025, sat: 0.4, rough: [0.0, 1.0] },
  Embers: { lum: 0.022, sat: 0.4, rough: [0.0, 1.0], kind: 'embers' },
  // graniteware: matte charcoal-blue with its white fleck, crusted and blackened from the fire up
  Enamel: { lum: 0.04, sat: 0.5, rough: [0.62, 0.92], contrast: 1.5, crust: 1, speckle: 1 },
  wooden_axe: { lum: 0.12, sat: 0.9, rough: [0.0, 1.0], metal: true },
  wooden_bucket_01: { lum: 0.07, sat: 0.18, rough: [0.45, 1.0], metal: true, rust: 1, contrast: 1.4, bias: [1.0, 0.98, 0.95], vary: 0.2 },
  Lantern_01_brass: { lum: 0.0733, sat: 1.0, rough: [0.0, 1.0], metal: true },
  // split firewood (firewood.ts): grey-brown bark, split faces and end grain whose weathering is in the vertex tint
  PileBark: { lum: 0.068, sat: 0.4, rough: [0.4, 1.0], ao: 0.8, cavity: 0.5, contrast: 1.35, bias: [1.0, 0.97, 0.93], vary: 0.15, foot: 0.6 },
  PileSplit: { lum: 0.105, sat: 0.3, rough: [0.6, 1.0], bias: [1.0, 0.96, 0.9], contrast: 1.4, cavity: 0.5, checks: 0.4 },
  PileEnds: { lum: 0.092, sat: 0.38, rough: [0.5, 1.0], contrast: 1.12, bias: [1.0, 0.97, 0.93], cavity: 0.55, rcheck: 0.75 },
  // the fire program: burning logs charred into blocks with ash in the cracks (fire.ts: log-space uvs), their sawn
  // outer ends, the ash bed, charcoal lumps
  CharLog: { lum: 0.074, sat: 0.72, rough: [0.45, 1.0], kind: 'fire', logs: 1, burn: [0.2, 0.5] },
  CharEnd: { lum: 0.045, sat: 0.35, rough: [0.6, 1.0], kind: 'fire', logs: 1, bias: [1.0, 0.94, 0.86] },
  CharLump: { lum: 0.03, sat: 0.3, rough: [0.4, 1.0], kind: 'fire', burn: [5.0, 6.0] },
  AshBed: { lum: 0.3, sat: 0.15, rough: [0.85, 1.0], kind: 'fire', bed: 1, burn: [0.0, 1.0], normal: 1.1 },
};

// the fire ring's stones: each scan greyed and darkened (round 3 wip: their own buff and rust read as olive loaves),
// no moss at their feet (round 4 wip: the sunlit ones still olive), only the damp and ash there
// (scene fix, critic polish r1: "even dark lumps": paler granite grey, so the soot on the fire side shows)
for (const id of ['rock_09', 'rock_07', 'stone_01', 'namaqualand_boulder_05']) CAL[`Ring_${id}`] = { ...CAL[id], lum: 0.13, sat: 0.08, contrast: 1.35, foot: 0.3 };

/** untextured materials: base colour, roughness, metalness, fake reflection of the treeline and sky */
const PLAIN: Record<string, [[number, number, number], number, number, number]> = {
  Iron: [[0.1, 0.08, 0.07], 0.7, 0.6, 0],
  SootedIron: [[0.05, 0.045, 0.042], 0.78, 0.45, 0],
  // an unlit room behind old glass: what shows is the reflection (part 8 raises its emissive at night)
  WindowGlass: [[0.006, 0.0065, 0.007], 0.3, 0.0, 1.35],
  LanternGlass: [[0.03, 0.028, 0.024], 0.05, 0.0, 1],
};

/** source material -> the material it is drawn with (same maps; tint differences go into the vertex tint) */
export const MAT_ALIAS: Record<string, string> = { DockTimberWet: 'DockTimber', RoofShakesDark: 'RoofShakes', Lantern_01_glass: 'LanternGlass' };
/** vertex tint of an aliased material relative to the one it is drawn with (its baseColorFactor ratio) */
export const ALIAS_TINT: Record<string, [number, number, number]> = { RoofShakesDark: [0.72, 0.74, 0.7] };

/** More camp materials on the same programs (the cabin's room, interior.ts): call before their first use. */
export function addCampMaterials(o: { cal?: Record<string, Calib>; plain?: typeof PLAIN }) {
  Object.assign(CAL, o.cal);
  Object.assign(PLAIN, o.plain);
}
export type CampCalib = Calib;

export class CampMaterials {
  private cache = new Map<string, THREE.MeshStandardNodeMaterial>();
  private textures = new Map<string, THREE.Texture>();
  /** the lit-window material (part 8: emissive / emissiveIntensity, or Camp.setWindowGlow) */
  windowGlass: THREE.MeshStandardNodeMaterial | null = null;
  embers: THREE.MeshStandardNodeMaterial | null = null;

  constructor(private field: RiverField | null, private sky: SkyLighting) {}

  /**
   * One texture object per image across the four GLBs (they embed the same Poly Haven maps several times). Maps
   * without a name (the rock scans, already shared through Assets) are their own key.
   */
  private canon(t: THREE.Texture | null): THREE.Texture {
    if (!t) throw new Error('camp: missing map');
    const key = t.name ? `${t.name}|${t.colorSpace}` : t.uuid;
    let c = this.textures.get(key);
    if (!c) {
      c = t;
      c.anisotropy = 8;
      this.textures.set(key, c);
    }
    return c;
  }

  get textureCount() {
    return this.textures.size;
  }

  private over = new Map<string, Partial<Calib>>();

  /** Override calibration values of a material before its first use (soot lines at world heights known at build). */
  calibrate(name: string, c: Partial<Calib>) {
    this.over.set(name, { ...this.over.get(name), ...c });
  }

  /** the material a source glTF material is drawn with (`as`: a camp material of its own on the same maps) */
  get(src: THREE.MeshStandardMaterial, as: string | null = null): THREE.MeshStandardNodeMaterial {
    const name = as ?? MAT_ALIAS[src.name] ?? src.name;
    let m = this.cache.get(name);
    if (m) return m;
    const cal = CAL[name] ? { ...CAL[name], ...this.over.get(name) } : null;
    if (PLAIN[name]) m = this.plain(name);
    else if (cal?.kind === 'fire') m = this.fire(src, name, cal);
    else if (cal) m = this.surface(src, name, cal);
    else {
      console.warn(`camp: no calibration for material ${src.name}`);
      m = this.surface(src, name, { lum: 0.12, sat: 0.7, rough: [0.4, 1.0] });
    }
    this.cache.set(name, m);
    return m;
  }

  get materialCount() {
    return this.cache.size;
  }

  private plain(name: string) {
    const [c, r, met, refl] = PLAIN[name];
    const m = new THREE.MeshStandardNodeMaterial();
    m.name = name;
    const v = attribute('aCamp', 'vec4');
    m.colorNode = uniform(new THREE.Vector3(...c)).mul(v.xyz);
    m.roughnessNode = uniform(r);
    m.metalnessNode = uniform(met);
    m.aoNode = v.w.mul(indoorSky(positionWorld, normalWorldGeometry));
    // Glass: the environment map is open sky all round, so a pane facing the meadow mirrored a flat pale sky. Its
    // reflection instead: the forest's ragged skyline (a dark wall of spruce below it, the sky above), rippled a
    // little by old glass, weighted by Fresnel, plus a sharp glint of the low sun; the material's own emissive (part
    // 8's lit window) is added on top unchanged.
    const u = this.sky.u;
    const V = normalize(positionWorld.sub(cameraPosition));
    const Nw = normalWorld;
    const ripple = vec3(tnoise(positionWorld.xy.mul(7.0), 2.2).sub(0.5), 0, tnoise(positionWorld.zy.mul(7.0), 3.3).sub(0.5)).mul(0.035);
    const R = normalize(reflect(V, Nw).add(ripple));
    const az = atan(R.x, R.z);
    // (round 3: "flat grey card"): a hard, bright patch of sky with the jagged tips of the spruce line across it
    const tw = az.mul(11.0).add(tnoise(vec2(az.mul(2.0), 2.5), 6.3).mul(1.4));
    const teeth = pow(float(1).sub(abs(tw.fract().sub(0.5)).mul(2.0)), 3.0).mul(mix(0.03, 0.11, tnoise(vec2(az.mul(7.0), 3.5), 7.1)));
    const skyline = float(0.12).add(tnoise(vec2(az.mul(4.0), 0.5), 4.1).sub(0.5).mul(0.16)).add(teeth);
    const skyC = mix(vec3(u.horizon), vec3(u.zenith), smoothstep(0.0, 0.7, R.y)).mul(1.3);
    const trees = vec3(0.012, 0.018, 0.013).mul(mix(0.6, 1.4, tnoise(vec2(az.mul(40.0), R.y.mul(30.0)), 7.7)));
    const env = mix(trees, skyC, smoothstep(skyline.sub(0.012), skyline.add(0.012), R.y));
    const cosV = max(dot(V.negate(), Nw), 0);
    const fres = float(0.05).add(pow(float(1).sub(cosV), 5).mul(0.95));
    const glint = pow(max(dot(R, u.sunDir), 0), 900).mul(u.sunIntensity).mul(vec3(u.sunColor)).mul(0.6);
    let lit: N = materialEmissive;
    if (name === 'WindowGlass') {
      // (part 8 round 2, critic r1: "a small flat yellow card"): the lamp inside, hottest where the view ray passes
      // nearest it, so the bright patch moves with the eye as through a real window; the lamplit room round it amber,
      // deeper at the panes' edges (ref_05)
      const toLamp = vec3(campU.lamp).sub(positionWorld);
      const miss = length(toLamp.sub(V.mul(dot(toLamp, V).max(0))));
      const spot = exp(miss.mul(miss).div(-0.11));
      const room = exp(miss.mul(miss).div(-0.9));
      lit = vec3(materialEmissive).mul(mix(vec3(0.5, 0.26, 0.14), vec3(1.0, 0.86, 0.66), spot)).mul(spot.mul(2.6).add(room.mul(0.9)).add(0.35));
    }
    m.emissiveNode = env.mul(fres).add(glint).mul(uniform(refl)).add(lit);
    if (name === 'WindowGlass') this.windowGlass = m;
    return m;
  }

  private surface(src: THREE.MeshStandardMaterial, name: string, cal: Calib) {
    const m = new THREE.MeshStandardNodeMaterial();
    m.name = name;
    if (src.side === THREE.DoubleSide) m.side = THREE.DoubleSide;
    const map = this.canon(src.map);
    const armMap = this.canon(src.roughnessMap ?? src.metalnessMap ?? src.aoMap);
    const t = uv();
    const base = texture(map, t).rgb;
    const arm = texture(armMap, t);
    const v = attribute('aCamp', 'vec4');
    const f = src.color, b = cal.bias ?? [1, 1, 1];
    const c = [f.r * b[0], f.g * b[1], f.b * b[2]] as const;
    const texLum = cal.texLum ?? TEX_LUM[map.name];
    if (texLum === undefined) console.warn(`camp: no luminance for map ${map.name}`);
    const cavity = src.aoMap ? cal.cavity ?? 0 : 0;
    const k = cal.lum / ((texLum ?? 0.12) * Math.max(1e-4, lum3(c[0], c[1], c[2])) * (1 - cavity * (1 - (AO_MEAN[armMap.name] ?? 0.9))));
    const vary = uniform(cal.vary ?? 0);
    // noise from the shared baked tile (src/gfx/noiseTex.ts): procedural MaterialX noise here cost seconds of compile
    const p = positionWorld;
    // weathering patches a couple of metres across: silver-grey where the sun and rain bleach, browner in between
    // (a planar mapping that shifts with height, so walls of any facing vary)
    const n = smoothstep(0.35, 0.65, tnoise(vec2(p.x.add(p.y.mul(0.37)), p.z.sub(p.y.mul(0.61))).mul(0.5), 1));
    const shade = mix(float(1).sub(vary), float(1).add(vary), n);
    const hue = mix(vec3(1), mix(vec3(0.95, 0.99, 1.05), vec3(1.07, 0.99, 0.88), n), vary.mul(2.5).min(1));
    // rain streaks: narrow along the wall, long down it, darkening where they run
    const sn = smoothstep(0.5, 0.68, tnoise(vec2(p.x.add(p.z).mul(3.2), p.y.mul(0.3)), 2));
    const streaks = float(1).add(uniform(cal.streak ?? 0).mul(float(0.3).sub(sn)));
    // the scan's luminance contrast around its mean, and part of its occlusion as dirt in the grain
    const l = dot(base, LUM);
    // (a uniform, not a constant: per-material constants give every material its own shader program)
    const grainC = pow(l.max(1e-4).div(uniform(texLum ?? 0.12)), uniform((cal.contrast ?? 1) - 1)).min(4);
    const dirt = mix(float(1), arm.x, uniform(cavity));
    let col: N = mix(vec3(l), base, uniform(cal.sat)).mul(grainC).mul(dirt)
      .mul(uniform(new THREE.Vector3(c[0] * k, c[1] * k, c[2] * k))).mul(v.xyz).mul(shade).mul(hue).mul(streaks);
    let rough: N = mix(uniform(cal.rough[0]), uniform(cal.rough[1]), arm.y);
    // rust-brown streaks running along the grain of the horizontal logs (round 1: the same grey on every log)
    const gs = tnoise(vec2(p.x.add(p.z).mul(0.7), p.y.mul(11.0)), 6.1);
    const rustS = smoothstep(0.56, 0.72, gs).mul(uniform(cal.grain ?? 0));
    col = mix(col, col.mul(vec3(1.3, 0.9, 0.62)), rustS);
    // bold rust-brown stains and deep dark checks along the grain (uv v runs along the grain on logs and planks)
    const stainA = smoothstep(0.5, 0.66, tnoise(vec2(t.x.mul(1.7), t.y.mul(0.3)), 16.3).add(tnoise(t.mul(vec2(6.0, 1.4)), 17.4).sub(0.5).mul(0.3)))
      .mul(uniform(cal.stain ?? 0));
    col = mix(col, col.mul(vec3(1.42, 0.8, 0.5)).mul(0.82), stainA);
    const chkN = tnoise(vec2(t.x.mul(7.0), t.y.mul(0.42)), 14.2);
    const chkA = smoothstep(0.042, 0.016, abs(chkN.sub(0.5))).mul(smoothstep(0.48, 0.62, tnoise(vec2(t.x.mul(2.3), t.y.mul(0.2)), 15.1)))
      .mul(uniform(cal.checks ?? 0));
    col = mix(col, col.mul(0.1), chkA);
    rough = mix(rough, float(1), chkA);
    // end grain (pith at the uv centre): radial checks opening toward the rim, weathered rings standing proud
    const eg = t.sub(0.5);
    const egR = length(eg);
    const egDir = eg.div(egR.max(1e-4));
    const rcU = uniform(cal.rcheck ?? 0), ringU = uniform(cal.rings ?? 0);
    const rcN = tnoise(egDir.mul(0.62).add(vec2(3.1, 7.7)), 18.2);
    const rcW = mix(float(0.003), float(0.026), smoothstep(0.04, 0.46, egR));
    const rcFine = smoothstep(rcW, rcW.mul(0.3), abs(rcN.sub(0.5))).mul(smoothstep(0.06, 0.2, egR))
      .mul(smoothstep(0.4, 0.58, tnoise(egDir.mul(0.5).add(vec2(11.3, 5.2)), 19.1))).mul(rcU).clamp(0, 1);
    // a few wide checks (stump tops, corner log ends): spaced round the disc and wandering, a few missing, each its
    // own width, opening toward the rim
    const rcK = uniform(cal.rcount ?? 0);
    const cpos = atan(eg.y, eg.x).mul(rcK.div(Math.PI * 2)).add(tnoise(egDir.mul(0.9).add(egR.mul(3.0)), 21.1).sub(0.5).mul(0.5));
    const cid = cpos.add(0.5).floor(), cf = cpos.fract();
    const cdist = min(cf, cf.oneMinus()).mul(egR).mul(float(Math.PI * 2).div(rcK.max(1)));
    const cwid = mix(float(0.0015), float(0.017), smoothstep(0.05, 0.47, egR)).mul(hash(cid.add(3.7)).mul(0.8).add(0.4));
    const rcWide = smoothstep(cwid, cwid.mul(0.3), cdist).mul(smoothstep(0.07, 0.22, egR))
      .mul(hash(cid.add(9.1)).lessThan(0.78).select(1, 0)).mul(rcK.min(1));
    const rcA = max(rcFine, rcWide);
    // rings at uneven spacing (round 4 wip: evenly spaced, they read as a target)
    const ringPh = egR.mul(28.0).add(tnoise(vec2(egR.mul(5.0), 0.5), 20.9).mul(8.0)).add(tnoise(egDir.mul(1.6).add(egR.mul(4.0)), 20.4).mul(3.6));
    const ringS = ringPh.mul(Math.PI * 2).sin();
    col = col.mul(float(1).add(ringS.mul(0.04).mul(ringU)));
    col = mix(col, vec3(0.009, 0.008, 0.007), rcA);
    rough = mix(rough, float(1), rcA);
    // grime darkening toward the rim and in patches, pale ash smudged over it (stump tops by the fire)
    const ashyU = uniform(cal.ashy ?? 0);
    col = col.mul(float(1).sub(smoothstep(0.25, 0.46, egR).mul(0.35).add(smoothstep(0.55, 0.75, tnoise(p.xz.mul(7.0), 21.6)).mul(0.3)).mul(ashyU)));
    // (round 4 wip: big pale blotches read as blue paint) smaller, greyer smudges
    const ashA = smoothstep(0.58, 0.74, tnoise(p.xz.mul(17.0).add(p.y), 22.7).add(tnoise(p.xz.mul(40.0), 23.8).sub(0.5).mul(0.35))).mul(ashyU);
    col = mix(col, vec3(0.15, 0.146, 0.14), ashA.mul(0.6));
    // brown-black crust and drips over the soot (the coffee pot)
    const crustU = uniform(cal.crust ?? 0);
    const crustA = smoothstep(0.52, 0.68, tnoise(vec2(p.x.add(p.z).mul(70.0), p.y.mul(18.0)), 24.9).add(tnoise(p.xz.mul(140.0), 25.1).sub(0.5).mul(0.3)))
      .mul(crustU);
    col = mix(col, vec3(0.022, 0.016, 0.011), crustA.mul(0.8));
    rough = mix(rough, float(0.95), crustA);
    // graniteware's white flecks, dulled where the soot and the crust cover them
    const speck = smoothstep(0.72, 0.76, tnoise(vec2(p.x.add(p.z).mul(260.0), p.y.mul(240.0)), 27.9)).mul(uniform(cal.speckle ?? 0)).mul(float(1).sub(crustA));
    col = mix(col, vec3(0.3, 0.31, 0.33), speck.mul(0.55));
    // damp, mud and moss at the foot (ground contact baked as aCamp.w: 0.5 on the ground .. 1 at 0.45 m)
    const near = smoothstep(0.9, 0.55, v.w).mul(uniform(cal.foot ?? 0));
    const mossN = smoothstep(0.42, 0.68, tnoise(p.xz.mul(2.7).add(p.y.mul(1.9)), 7.3));
    col = mix(col, col.mul(vec3(0.6, 0.58, 0.54)), near.mul(0.8));
    col = mix(col, mix(vec3(0.03, 0.045, 0.012), vec3(0.06, 0.08, 0.02), tnoise(p.xz.mul(9.0), 2.9)), near.mul(mossN).mul(0.55));
    rough = mix(rough, float(0.95), near.mul(mossN));
    // lichen spots and moss patches on old shakes: moss in broad damp drifts, pale lichen rosettes scattered
    const lich = uniform(cal.lichen ?? 0);
    const mossP = smoothstep(0.55, 0.78, tnoise(p.xz.mul(0.9).add(p.y.mul(0.5)), 8.8)).mul(smoothstep(0.35, 0.6, tnoise(p.xz.mul(5.0), 3.9)));
    col = mix(col, mix(vec3(0.022, 0.032, 0.01), vec3(0.05, 0.065, 0.018), tnoise(p.xz.mul(13.0), 5.5)), mossP.mul(lich).mul(0.8));
    const spots = smoothstep(0.7, 0.76, tnoise(p.xz.mul(11.0).add(p.y.mul(4.0)), 9.4));
    col = mix(col, vec3(0.16, 0.17, 0.14), spots.mul(lich).mul(0.55));
    // chinking: fallen out in patches (the dark gap behind), daubed again with browner mud in others
    const pat = uniform(cal.patchy ?? 0);
    const gap = smoothstep(0.6, 0.66, tnoise(vec2(p.x.add(p.z).mul(1.7), p.y.mul(6.0)), 10.1));
    const mud = smoothstep(0.45, 0.6, tnoise(vec2(p.x.add(p.z).mul(0.8), p.y.mul(3.0)), 11.3));
    col = mix(col, col.mul(vec3(1.05, 0.82, 0.6)), mud.mul(pat).mul(0.7));
    col = mix(col, vec3(0.008, 0.007, 0.006), gap.mul(pat));
    // soot: blackened below (pot) or above (chimney crown) a ragged line
    const sootU = uniform(new THREE.Vector3(...(cal.soot ?? [0, 1, 0])));
    const sootRun = smoothstep(0.5, 0.72, tnoise(vec2(p.x.add(p.z).mul(5.0), p.y.mul(0.35)), 26.6)).mul(uniform(cal.sootStreak ?? 0));
    const sootT = smoothstep(sootU.x.sub(sootRun), sootU.y, p.y.add(tnoise(p.xz.mul(31.0).add(p.y.mul(17.0)), 12.7).sub(0.5).mul(0.05))).mul(sootU.z);
    col = mix(col, vec3(0.011, 0.01, 0.009), sootT);
    rough = mix(rough, float(0.92), sootT);
    // the metal mask turned to rust
    const metal = arm.z.mul(uniform(cal.metal ? 1 : 0));
    const rustA = metal.mul(uniform(cal.rust ?? 0)).clamp(0, 1);
    // dark, mottled old iron (round 2's even rust read as a copper bail)
    const rustM = smoothstep(0.38, 0.66, tnoise(p.xz.mul(40.0).add(p.y.mul(23.0)), 13.1).add(tnoise(p.xz.mul(130.0).add(p.y.mul(90.0)), 13.7).sub(0.5).mul(0.4)));
    const rustC = mix(vec3(0.022, 0.015, 0.011), vec3(0.085, 0.04, 0.017), rustM);
    col = mix(col, rustC, rustA);
    rough = mix(rough, float(0.85), rustA);
    // polished and darkened by hands round the door latch
    const worn = smoothstep(campU.latch.w, campU.latch.w.mul(0.2), length(p.sub(campU.latch.xyz))).mul(uniform(cal.worn ?? 0));
    col = mix(col, col.mul(vec3(0.42, 0.38, 0.33)), worn);
    rough = mix(rough, float(0.4), worn);
    if (cal.kind === 'dock') ({ col, rough } = dockWet(col, rough));
    m.colorNode = col;
    m.roughnessNode = rough;
    m.metalnessNode = metal.mul(float(1).sub(rustA.mul(0.85)));
    m.aoNode = mix(float(1), arm.x, uniform(src.aoMap ? cal.ao ?? 1 : 0)).mul(v.w).mul(indoorSky(p, normalWorldGeometry));
    const ns = cal.normal ?? 1;
    const nTex = texture(this.canon(src.normalMap), t);
    // the proud rings tilt the normal in and out across each ring (end grain only)
    const nRing = vec3(nTex.xy.add(egDir.mul(ringPh.mul(Math.PI * 2).cos().mul(0.05).mul(ringU))), nTex.z);
    m.normalNode = normalMap(nRing, uniform(new THREE.Vector2(src.normalScale.x * ns, src.normalScale.y * ns)));
    if (cal.kind === 'embers') {
      // glowing cracks that breathe: a slow drifting noise over the logs, the fire's flicker and strength
      const breath = smoothstep(0.3, 0.7, tnoise(p.xz.mul(6.0).add(vec2(campU.time.mul(0.11), campU.time.mul(-0.06))), 3));
      m.emissiveNode = texture(this.canon(src.emissiveMap), t).rgb.mul(campU.emberGain).mul(mix(0.35, 1.0, campU.strength))
        .mul(campU.flicker).mul(breath.mul(1.2).add(0.4));
      this.embers = m;
    }
    // caustics, beam absorption and refracted shadows on the posts under the water (identity model matrix)
    if (cal.kind === 'dock' && this.field) installUnderwater(m, { field: this.field, surfaceY: float(POOL_LEVEL) });
    return m;
  }

  /**
   * Burnt wood, charcoal and the ash bed (one program). Round 1's logs were plain bark darkened by a vertex tint:
   * "the fire has never burned anything". Here, toward the ring's centre, the wood turns to charcoal broken into a
   * checkerboard of blocks (a warped 3D grid, only the planes that cut the surface count) with pale ash packed into
   * the cracks and a white ash crust on the tops; the deepest cracks glow near the coals. The bed: pale powdery ash
   * in drifts with black charcoal grit and lumps and a few live coals near the middle.
   */
  private fire(src: THREE.MeshStandardMaterial, name: string, cal: Calib) {
    const m = new THREE.MeshStandardNodeMaterial();
    m.name = name;
    const map = this.canon(src.map);
    const t = uv();
    const base = texture(map, t).rgb;
    const v = attribute('aCamp', 'vec4');
    const texLum = cal.texLum ?? TEX_LUM[map.name] ?? 0.12;
    const f = src.color, b = cal.bias ?? [1, 1, 1];
    const c = [f.r * b[0], f.g * b[1], f.b * b[2]] as const;
    const k = cal.lum / (texLum * Math.max(1e-4, lum3(c[0], c[1], c[2])));
    const l = dot(base, LUM);
    const wood = mix(vec3(l), base, uniform(cal.sat)).mul(uniform(new THREE.Vector3(c[0] * k, c[1] * k, c[2] * k))).mul(v.xyz);
    const p = positionWorld;
    const fc = campU.fireCenter;
    const d = length(p.xz.sub(fc.xz));
    const br = uniform(new THREE.Vector2(...(cal.burn ?? [0.2, 0.5])));
    const nb = tnoise(p.xz.mul(7.0).add(p.y.mul(3.0)), 12.3);
    const logs = uniform(cal.logs ?? 0);
    // log mode (fire.ts): uv in metres is (distance past the char front, distance round the log); world mode (the
    // bed, the lumps): burnt by distance from the ring's centre
    const um = t.mul(0.5);
    const nbL = tnoise(vec2(um.y.mul(8.0), um.x.mul(6.0)), 12.9);
    const burn = mix(float(1).sub(smoothstep(br.x, br.y, d.add(nb.sub(0.5).mul(0.14)))), float(1).sub(smoothstep(-0.035, 0.02, um.x.add(nbL.sub(0.5).mul(0.06)))), logs);
    // charcoal blocks. World mode: the zero lines of two noise octaves (ridged) close into an irregular network
    // ~3-5 cm across, projected three ways. Log mode (round 3: "smooth, dull grey tubes"): bold blocks in rows round
    // the log, each row its own block length and stagger, the joints wobbling, white ash packed into every crack
    const nG = abs(normalWorldGeometry);
    const wt = nG.mul(nG).mul(nG.mul(nG));
    const wsum = wt.x.add(wt.y).add(wt.z).max(1e-4);
    const ridges = (q2: N, sd: number) => {
      const r1 = abs(tnoise(q2, sd).sub(0.5)), r2 = abs(tnoise(q2.mul(2.3).add(5.1), sd + 1.7).sub(0.5));
      return max(smoothstep(0.035, 0.008, r1), smoothstep(0.03, 0.006, r2).mul(0.75));
    };
    const crackW = ridges(p.yz.mul(19.0), 13.1).mul(wt.x).add(ridges(p.zx.mul(19.0), 14.2).mul(wt.y)).add(ridges(p.xy.mul(19.0), 15.3).mul(wt.z)).div(wsum);
    // log mode (scene fix, critic polish r1: "regular stripes like rope or bamboo"): an alligator net of blocks in rows
    // ~3 cm round the log whose joints wander, every joint along the log jittered on its own so each block has its own
    // length and no crack rings the log; the cracks across the grain wider, white with ash in patches, black elsewhere;
    // those along the grain black and broken; a few blocks crusted grey-white, more toward the burning tip
    const HR = 0.031;
    const around = um.y.add(tnoise(um.mul(vec2(13.0, 4.0)), 26.1).sub(0.5).mul(0.024));
    const rowF = around.div(HR);
    const row = rowF.floor();
    const L0 = 0.024;
    const along = um.x.add(hash(row.mul(1.37)).mul(0.05)).add(tnoise(um.mul(vec2(6.0, 21.0)), 27.3).sub(0.5).mul(0.016));
    const cf = along.div(L0);
    const c0 = cf.floor();
    const fa = cf.sub(c0);
    // the joint in cell c sits at c + j(c), j in 0.15..0.85
    const jAt = (c: N) => hash(row.mul(3.17).add(c.mul(1.931)).add(0.27)).mul(0.7).add(0.15);
    const jm = jAt(c0.sub(1)).sub(1), j0 = jAt(c0), jp = jAt(c0.add(1)).add(1);
    const ex = min(min(abs(fa.sub(jm)), abs(fa.sub(j0))), abs(fa.sub(jp))).mul(L0);
    const blk = c0.add(step(j0, fa));
    const fy = rowF.fract();
    const ey = min(fy, fy.oneMinus()).mul(HR);
    const bid = row.mul(13.1).add(blk.mul(7.7));
    // cracks 1.5-4 mm wide, a pixel or two at the close-up
    const cw = mix(float(0.0016), float(0.0042), tnoise(um.mul(9.0), 28.5)).mul(hash(bid.add(0.31)).mul(0.6).add(0.7));
    const crT = float(1).sub(smoothstep(cw.mul(0.3), cw.mul(1.2), ex));
    const crG = float(1).sub(smoothstep(cw.mul(0.18), cw.mul(0.55), ey)).mul(step(hash(bid.add(0.6)), 0.88));
    const crackL = max(crT, crG);
    const crack = mix(crackW, crackL, logs);
    // each block a shade of its own (a finer noise in world mode, the block's id in log mode)
    const cr = mix(tnoise(p.xz.mul(31.0).add(p.y.mul(17.0)), 16.9), hash(bid), logs);
    const charC = mix(vec3(0.011, 0.0105, 0.01), vec3(0.04, 0.038, 0.036), cr).mul(mix(float(1), cr.mul(0.5).add(0.6), logs));
    // white ash in the cross cracks where it has gathered, black in the rest
    const fillT = smoothstep(0.46, 0.62, tnoise(um.mul(vec2(14.0, 9.0)), 16.6));
    // (part 8 round 2, critic r1: at night "the logs look like clean grey wood": firelit, the ash in their cracks and on
    // their crusts read as pale wood; at night the burning logs are black blocks with glowing cracks)
    const ashy = float(1).sub(campU.night.mul(logs).mul(0.8));
    const fill = mix(smoothstep(0.3, 0.6, tnoise(p.xz.mul(19.0).add(p.y.mul(11.0)), 16.4)).mul(0.8).add(0.2),
      fillT.mul(crT.div(crT.add(crG).max(1e-3))).mul(ashy), logs);
    let charCol: N = mix(charC, mix(vec3(0.42, 0.41, 0.395), vec3(0.4, 0.394, 0.382), logs), crack.mul(fill));
    charCol = mix(charCol, vec3(0.006, 0.0058, 0.0055), crack.mul(fill.oneMinus()).mul(logs));
    const coat = step(hash(bid.add(0.77)), float(1).sub(smoothstep(-0.12, -0.02, um.x)).mul(0.3).add(0.1)).mul(smoothstep(-0.35, 0.2, normalWorldGeometry.y)).mul(logs);
    charCol = mix(charCol, vec3(0.36, 0.355, 0.345).mul(mix(0.75, 1.1, hash(bid.add(0.13)))), coat.mul(crack.oneMinus()).mul(0.85).mul(ashy));
    // a white ash crust on the upper faces, in patches (on the logs only a little: their black blocks must show)
    const nUp = normalWorldGeometry.y;
    const crust = smoothstep(0.55, 0.85, nUp.add(tnoise(p.xz.mul(11.0), 17.5).sub(0.5).mul(0.9))).mul(smoothstep(0.35, 0.65, tnoise(p.xz.mul(4.0), 18.6)));
    charCol = mix(charCol, vec3(0.42, 0.41, 0.4).mul(mix(0.8, 1.1, cr)), crust.mul(mix(0.7, 0.45, logs)).mul(mix(float(1), crack.oneMinus(), logs)).mul(ashy));
    // log mode: each block a low dome bevelled down into its cracks and tilted its own way, so the blocks catch the
    // light one by one; it takes the bark's relief's place where the log has burnt (tangent space: +x along the log,
    // +y round it; GLTFLoader flips the y scale for derivative tangents, undone here)
    const blockN = vec2(hash(bid.add(0.51)).sub(0.5), hash(bid.add(0.93)).sub(0.5).mul(uniform(Math.sign(src.normalScale.y) || 1))).mul(0.5);
    // scorched in a ragged band where the fire has not reached yet (log mode: a few centimetres past the front)
    // (scene fix wip: the scorch reaches 11 cm past the char front, was 7.5)
    const sootB = mix(float(1).sub(smoothstep(br.y.sub(0.05), br.y.add(0.12), d.add(nb.sub(0.5).mul(0.2)))), float(1).sub(smoothstep(0.0, 0.11, um.x.add(nbL.sub(0.5).mul(0.05)))), logs);
    // (at night, firelit from a hand away, even 0.04 char read grey: black logs with glowing cracks)
    let col: N = mix(wood.mul(float(1).sub(sootB.mul(0.82))), charCol.mul(float(1).sub(campU.night.mul(logs).mul(0.6))), burn);
    // the ash bed: grey powder in drifts, darker where it is still sooty, black grit and charcoal chunks all through
    // it (round 3: "a flat, clean, near-white disc like a paper plate"), live coals near the middle
    const bed = uniform(cal.bed ?? 0);
    // pale grey-white powder (scene fix, critic polish r1: round 4's 0.19 read as "a flat dark slate-grey disc like a
    // rubber mat"), whitest on the crust of the heap's tops, greyer in the hollows and where charcoal dust is mixed in
    const ashTex = l.div(uniform(texLum)).clamp(0.5, 1.6);
    let ash: N = vec3(0.31, 0.306, 0.3).mul(mix(0.68, 1.15, tnoise(p.xz.mul(4.0), 19.7))).mul(mix(0.85, 1.1, tnoise(p.xz.mul(19.0), 20.8)))
      .mul(mix(1.0, ashTex, 0.45));
    const crestA = smoothstep(0.9, 0.99, nUp).mul(smoothstep(0.4, 0.7, tnoise(p.xz.mul(7.0), 23.9)));
    ash = mix(ash, vec3(0.44, 0.436, 0.43), crestA.mul(0.6));
    const lump = smoothstep(0.7, 0.74, tnoise(p.xz.mul(16.0), 21.9));
    const grit = smoothstep(0.64, 0.69, tnoise(p.xz.mul(90.0), 22.1)).mul(smoothstep(0.35, 0.6, tnoise(p.xz.mul(6.0), 22.6)));
    const scorch = float(1).sub(smoothstep(0.12, 0.5, d));
    // grey drifts where charcoal dust is mixed in, black grit and bits of charcoal through it, greyer by the coals
    ash = mix(ash, ash.mul(vec3(0.5, 0.49, 0.48)), smoothstep(0.5, 0.76, tnoise(p.xz.mul(3.3), 22.9)).mul(0.65));
    ash = mix(ash, vec3(0.025, 0.023, 0.021), grit.mul(0.7));
    ash = mix(ash, ash.mul(vec3(0.6, 0.59, 0.58)), scorch.mul(smoothstep(0.4, 0.66, tnoise(p.xz.mul(2.6), 23.3))).mul(0.55));
    // out at the rim the ash thins over the dirt, greyer and browner (the ground's own ash and charcoal take over)
    ash = mix(ash, ash.mul(vec3(0.48, 0.43, 0.38)), smoothstep(0.44, 0.62, d).mul(0.75));
    // (part 8 round 2, critic r1: at night "the ash bed is lit cream like white sand": under the fire light AgX takes the
    // pale powder to cream; at night it is the grey of ash dusted with soot round a burning fire)
    ash = ash.mul(mix(float(1), float(0.4), campU.night));
    col = mix(col, ash, bed);
    // charcoal blocks each with their own sheen
    let rough: N = mix(mix(uniform(cal.rough[0]), uniform(cal.rough[1]), 0.7), mix(mix(float(0.5), mix(float(0.42), float(0.72), cr), logs), float(0.95), crack), burn);
    rough = mix(rough, float(0.97), bed);
    // at night the char's block sheen caught the moon in white bands round every log (they read as clean grey wood)
    rough = mix(rough, float(0.96), campU.night.mul(logs).mul(burn));
    // live coals: the deepest cracks of the burning logs near the middle, and a few lumps in the bed
    // (part 8 round 2: at night the bed of live coals spreads under the logs, and the cracks glow along each log's burnt
    // end toward its tip, critic r1: "no red-orange coals")
    const hot = float(1).sub(smoothstep(0.06, mix(float(0.24), float(0.34), campU.night), d));
    const tipGlow = float(1).sub(smoothstep(-0.2, -0.02, um.x)).mul(campU.night).mul(logs);
    const breath = smoothstep(0.3, 0.7, tnoise(p.xz.mul(9.0).add(vec2(campU.time.mul(0.13), campU.time.mul(-0.07))), 24.4));
    const coal = mix(crack.mul(burn).mul(max(hot, tipGlow.mul(0.9))), lump.mul(smoothstep(0.6, 0.66, tnoise(p.xz.mul(31.0), 25.5))).mul(hot), bed);
    m.emissiveNode = mix(vec3(1.0, 0.3, 0.06), vec3(1.0, 0.24, 0.04), campU.night).mul(coal).mul(campU.emberGain).mul(mix(0.3, 1.0, campU.strength))
      .mul(campU.flicker).mul(breath.mul(1.4).add(0.3)).mul(0.5);
    m.colorNode = col;
    m.roughnessNode = rough;
    m.metalnessNode = float(0);
    m.aoNode = v.w;
    const nTex = texture(this.canon(src.normalMap), t);
    const nUse = mix(nTex.xyz, vec3(blockN.add(0.5), 1.0), burn.mul(logs));
    const ns = cal.normal ?? 1;
    m.normalNode = normalMap(nUse, uniform(new THREE.Vector2(src.normalScale.x * ns, src.normalScale.y * ns)));
    return m;
  }
}

/**
 * Timber at the waterline: a soaked dark band a hand's width above the calm pool with a ragged top, darkest right
 * at the surface, glossy; below it green-brown slime that thickens with depth. The pool is flat (y = POOL_LEVEL),
 * so no field lookup.
 */
function dockWet(col: N, rough: N): { col: N; rough: N } {
  const p = positionWorld;
  const h = p.y.sub(POOL_LEVEL);
  const band = float(0.17).add(tnoise(vec2(p.x.sub(p.z).mul(5.0), p.y.mul(2.0)), 4).sub(0.5).mul(0.12))
    .add(tnoise(vec2(p.x.add(p.z).mul(16.0), p.y.mul(6.5)), 5).sub(0.5).mul(0.04));
  const wet = smoothstep(band.add(0.015), band.sub(0.015), h);
  const soak = smoothstep(0.1, -0.01, h);
  const under = smoothstep(0.0, -0.08, h);
  const lw = dot(col, LUM);
  let c: N = mix(col, mix(vec3(lw), col, 1.35).mul(mix(float(0.42), float(0.3), soak)), wet);
  // submerged: soaked wood under green-brown slime, streaky
  const slime = mix(vec3(0.035, 0.04, 0.016), vec3(0.06, 0.07, 0.025), tnoise(vec2(p.x.add(p.z).mul(9.0), p.y.mul(3.0)), 6.6));
  c = mix(c, mix(col.mul(vec3(0.5, 0.55, 0.38)), slime, smoothstep(-0.02, -0.25, h).mul(0.85)), under);
  const far = smoothstep(4.0, 20.0, positionWorld.sub(cameraPosition).length());
  let r: N = mix(rough, mix(float(0.28), float(0.5), far), wet);
  r = mix(r, float(0.7), under);
  return { col: c, rough: r };
}
