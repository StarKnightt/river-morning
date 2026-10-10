import * as THREE from 'three/webgpu';
import {
  abs, attribute, cameraPosition, cameraViewMatrix, cos, cross, dFdx, dFdy, dot, float, floor, fract, fwidth, length, max,
  mix, normalize, positionGeometry, positionLocal, positionView, positionWorld, pow, sign, sin, smoothstep, step,
  uniform, uv, vec3, vec4, type N,
} from '../gfx/tsl';
import { LAYER_NO_REFLECT } from '../core/layers';
import { fbm, smoothstep as sstep } from './noise';
import { LAYOUT } from './Layout';
import { pathDist, type CanopyField } from './ForestLayout';
import type { CoreSampler, CorePoint } from './BankProps';
import { windU } from './Forest';
import { tnoise } from '../gfx/noiseTex';
import { meadowAt } from './Meadow';
import { campGrassMask } from '../camp/layout';
import { CAMERA_SPOTS } from '../debug/CameraSpots';
import { Q } from '../core/quality';

const EYES = CAMERA_SPOTS.map((s) => [s.eye[0], s.eye[2]] as const);

/**
 * Grass blades near the camera, instanced over the banks in 6 m chunks, generated lazily from the bank masks (no grass
 * on the gravel bar, in the water, on the footpaths, on bare cut faces or under dense crowns; shoots up through the
 * gravel at the bar's top edge). Scene fix (critic polish r1: "a flat, smooth grey-green ground with identical
 * fan-shaped tufts spaced evenly over it, like a polka-dot carpet"): a dense, continuous sward of overlapping blade
 * patches in three shapes (soft and lodged, mixed, upright) whose height follows clumps and hollows at three scales,
 * tussocks gathered into big clumps of several tufts with dead straw lying at their foot, tall dry grass, seed heads,
 * and the odd drift of flowers (yarrow, arnica, lupine, paintbrush). By the water: sedge clumps and horsetail
 * colonies on the damp margins, and roots hanging out of the cut-bank faces under the turf lip. Flattened dead weed
 * in the flood wrack line on the bar. Repacked when the camera moves, nearest chunks first: patches and tufts thin
 * out and shrink into the ground toward 31 m, where the ground's own sward colouring carries on.
 */

const CHUNK = 6;
const RANGE = Math.round(31 * Q.k.grassRange);
/**
 * Grass lane (user, 10 Oct: "look at these grasses, do they look like grass?"): from a few metres out the sward is
 * carried by short cards, each a vertical strip turned toward the camera with ~36 blades cut out of it per pixel over a
 * dark root band, so the meadow reads as one continuous sward to ~36 m instead of separate tufts on a flat ground
 */
const CARD_RANGE = Math.round(36 * Q.k.grassRange);
const CARD_STEP = 0.42;
const CARD_BLADES = 14;

/** geometry kinds (one instanced mesh each, one program for all) */
const K_PATCH_A = 0, K_PATCH_B = 1, K_PATCH_C = 2, K_TUSSOCK = 3, K_TUSSOCK_B = 4, K_TALL = 5, K_SEED = 6, K_WRACK = 7,
  K_FLOWER = 8, K_SEDGE = 9, K_HORSETAIL = 10, K_ROOT = 11;
const KINDS = 12;
const isPatch = (k: number) => k <= K_PATCH_C;

interface Tuft {
  x: number; y: number; z: number; yaw: number; h: number; w: number; kind: number; dry: number; damp: number; rank: number;
  /** lip tufts hang out over a cut bank, wrack weed lies flat, roots droop out of a face: tilt (rad) toward `tiltDir` */
  tilt: number; tiltDir: number;
  /** infill only drawn near the camera (the sward round the angler's feet) */
  near?: boolean;
}

function hash(a: number, b: number, c: number) {
  let h = (Math.imul(a | 0, 374761393) + Math.imul(b | 0, 668265263) + Math.imul(c | 0, 1274126177)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

function finish(pos: number[], nor: number[], uvs: number[], bl: number[], idx: number[]) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  const nv = new THREE.Float32BufferAttribute(nor, 3);
  for (let i = 0; i < nv.count; i++) {
    const x = nv.getX(i), y = nv.getY(i), z = nv.getZ(i), l = Math.hypot(x, y, z);
    nv.setXYZ(i, x / l, y / l, z / l);
  }
  g.setAttribute('normal', nv);
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  g.setAttribute('aBlade', new THREE.Float32BufferAttribute(bl, 4));
  g.setIndex(idx);
  g.computeBoundingSphere();
  return g;
}

/** blade types (aBlade.w) and heads (aBlade.z) the material tells apart */
const T_TUFT = 0, T_PATCH = 1, T_HORSETAIL = 2, T_ROOT = 3, T_SEDGE = 4;
const H_NONE = 0, H_SEED = 1, H_FLOWER = 2;

interface BladeSpec {
  n: number;
  /** radius of the disc the blades are rooted in (m) */
  spread: number;
  h0: number; h1: number; hPow?: number;
  /** lean range; `outward`: the blades lean away from the tuft centre (tussocks), else every which way */
  lean: [number, number]; outward: boolean;
  /** blade width range (m) */
  w0: number; w1: number;
  seg: number;
  head?: number; type?: number;
  /** sideways kinks (m at the tip): roots wander as they hang */
  kink?: number;
  seed: number;
}

/**
 * A set of `n` blades rooted in a small disc, each arching (forward lean grows with t², the tip droops a little);
 * uv.y runs root (0) to tip (1), aBlade = (random per blade, lean, head, type). Seed heads: a thin stem widening into
 * a narrow head over its top fifth; flower heads: two crossed stems, each ending in a small round head.
 */
function bladeGeometry(o: BladeSpec): THREE.BufferGeometry {
  const pos: number[] = [], nor: number[] = [], uvs: number[] = [], bl: number[] = [];
  const idx: number[] = [];
  let r = o.seed;
  const rnd = () => ((r = (Math.imul(r, 1664525) + 1013904223) >>> 0) / 4294967296);
  const head = o.head ?? H_NONE, type = o.type ?? T_TUFT, SEG = o.seg;
  for (let b = 0; b < o.n; b++) {
    const a = rnd() * Math.PI * 2, rad = Math.sqrt(rnd()) * o.spread;
    const rx = Math.cos(a) * rad, rz = Math.sin(a) * rad;
    const h = o.h0 + (o.h1 - o.h0) * Math.pow(rnd(), o.hPow ?? 0.75);
    const facing = rnd() * Math.PI * 2;
    const la = o.outward ? a + (rnd() - 0.5) * 0.8 : rnd() * Math.PI * 2;
    const lr = o.lean[0] + (o.lean[1] - o.lean[0]) * rnd();
    const lk = o.outward ? lr * (0.4 + rad / Math.max(o.spread, 1e-3)) * (0.6 + 0.8 * rnd()) : lr;
    const w = o.w0 + (o.w1 - o.w0) * rnd();
    const br = rnd();
    const kp = rnd() * 6.28, kd = rnd() * 6.28;
    const ribbons = head === H_FLOWER ? [facing, facing + Math.PI / 2] : [facing];
    for (const f of ribbons) {
      const fx = Math.cos(f), fz = Math.sin(f);
      const base = pos.length / 3;
      for (let s = 0; s <= SEG; s++) {
        const t = s / SEG;
        const off = lk * t * t * h;
        const kk = (o.kink ?? 0) * t * Math.sin(t * 9 + kp);
        const cx = rx + Math.cos(la) * off + Math.cos(kd) * kk, cz = rz + Math.sin(la) * off + Math.sin(kd) * kk;
        // (grass lane: soft tips that bow over; at 0.18 the blades stood up stiff as bristles)
        const cy = h * t * (1 - 0.26 * Math.min(lk, 1.3) * t * t);
        let half: number;
        if (head === H_SEED) half = t > 0.8 ? 0.0055 * Math.sin(((t - 0.8) / 0.2) * Math.PI * 0.85 + 0.25) : 0.0016;
        // (a round head closing at the top; wip: sampled by 5 segments it was a flared strip with a flat cut top)
        else if (head === H_FLOWER) half = t > 0.8 ? 0.0017 + 0.0135 * Math.sin(((t - 0.8) / 0.2) * Math.PI) : 0.0017;
        else if (type === T_HORSETAIL) half = (w / 2) * (1 - 0.35 * t);
        else half = (w / 2) * (1 - 0.94 * Math.pow(t, 1.35));
        for (const side of [-1, 1]) {
          pos.push(cx + fx * half * side, cy, cz + fz * half * side);
          // a normal halfway between the blade's face and straight up: blades lit like a volume, never black edge-on
          nor.push(-fz * 0.6 + Math.cos(la) * 0.2, 0.75, fx * 0.6 + Math.sin(la) * 0.2);
          uvs.push(side < 0 ? 0 : 1, t);
          bl.push(br, lk, head, type);
        }
      }
      for (let s = 0; s < SEG; s++) {
        const q = base + s * 2;
        idx.push(q, q + 1, q + 2, q + 1, q + 3, q + 2);
      }
    }
  }
  return finish(pos, nor, uvs, bl, idx);
}

/** the twelve kinds and the most instances each draws */
const SPECS: Array<{ spec: BladeSpec; cap: number }> = [
  // sward patches: mixed (round 2's patch), soft and lodged, upright and taller
  // (grass lane: more blades, more of them bowed over)
  { spec: { n: 36, spread: 0.26, h0: 0.09, h1: 0.3, hPow: 0.8, lean: [0.2, 0.95], outward: false, w0: 0.003, w1: 0.007, seg: 3, type: T_PATCH, seed: 7 }, cap: 30000 },
  { spec: { n: 46, spread: 0.23, h0: 0.06, h1: 0.24, hPow: 0.9, lean: [0.6, 1.3], outward: false, w0: 0.003, w1: 0.006, seg: 3, type: T_PATCH, seed: 71 }, cap: 22000 },
  { spec: { n: 30, spread: 0.28, h0: 0.12, h1: 0.4, hPow: 0.7, lean: [0.12, 0.6], outward: false, w0: 0.0035, w1: 0.0075, seg: 3, type: T_PATCH, seed: 113 }, cap: 22000 },
  // tussocks: a tight one and a big ragged one
  { spec: { n: 30, spread: 0.09, h0: 0.2, h1: 0.5, lean: [0.55, 0.55], outward: true, w0: 0.004, w1: 0.009, seg: 4, seed: 23 }, cap: 9000 },
  { spec: { n: 46, spread: 0.15, h0: 0.22, h1: 0.62, lean: [0.45, 0.85], outward: true, w0: 0.0035, w1: 0.008, seg: 4, seed: 131 }, cap: 6000 },
  // tall wispy dry grass
  { spec: { n: 12, spread: 0.06, h0: 0.32, h1: 0.68, lean: [0.7, 0.7], outward: true, w0: 0.004, w1: 0.009, seg: 4, seed: 37 }, cap: 6000 },
  // seed heads above the sward
  { spec: { n: 8, spread: 0.05, h0: 0.42, h1: 0.85, hPow: 1, lean: [0.08, 0.26], outward: true, w0: 0.003, w1: 0.003, seg: 5, head: H_SEED, seed: 53 }, cap: 8000 },
  // flood wrack and the dead straw at the foot of the tussocks: dead weed laid flat
  { spec: { n: 14, spread: 0.12, h0: 0.2, h1: 0.45, lean: [0.9, 0.9], outward: true, w0: 0.004, w1: 0.009, seg: 4, seed: 61 }, cap: 8000 },
  // flowers: a few stems with small round heads
  { spec: { n: 5, spread: 0.07, h0: 0.2, h1: 0.46, hPow: 1, lean: [0.05, 0.22], outward: true, w0: 0.003, w1: 0.003, seg: 15, head: H_FLOWER, seed: 89 }, cap: 3000 },
  // sedge: stiff, keeled, wider blue-green blades arching out of a dense clump
  { spec: { n: 36, spread: 0.11, h0: 0.32, h1: 0.82, hPow: 0.6, lean: [0.38, 0.62], outward: true, w0: 0.005, w1: 0.0095, seg: 4, type: T_SEDGE, seed: 97 }, cap: 6000 },
  // horsetail: straight jointed stems in a colony
  { spec: { n: 7, spread: 0.14, h0: 0.16, h1: 0.46, hPow: 1, lean: [0.02, 0.1], outward: false, w0: 0.0045, w1: 0.006, seg: 4, type: T_HORSETAIL, seed: 101 }, cap: 6000 },
  // roots hanging out of a cut face, wandering: a few thick ones among thin (wip: all 2-4 mm, invisible past a few
  // metres, so the faces showed none)
  { spec: { n: 8, spread: 0.14, h0: 0.2, h1: 0.62, lean: [0.1, 0.4], outward: false, w0: 0.003, w1: 0.02, seg: 5, type: T_ROOT, kink: 0.05, seed: 109 }, cap: 4000 },
];

export const grassU = {
  /** linear albedo: deep live green, sunny yellow-green, golden straw, old grey straw, the dark sward base */
  green: uniform(new THREE.Color(0.042, 0.072, 0.02)),
  yellow: uniform(new THREE.Color(0.085, 0.098, 0.028)),
  straw: uniform(new THREE.Color(0.165, 0.132, 0.062)),
  dead: uniform(new THREE.Color(0.125, 0.112, 0.078)),
  root: uniform(new THREE.Color(0.016, 0.02, 0.009)),
  gain: uniform(1.0),
  /** backlit glow (translucency) of live blades and of the seed heads */
  translucency: uniform(0.26),
  seedGlow: uniform(0.55),
};

function grassMaterial() {
  const m = new THREE.MeshSSSNodeMaterial();
  m.name = 'grass';
  m.side = THREE.DoubleSide;
  const inst = attribute('aGrass', 'vec4'); // x dryness, y dampness, z tint, w rank
  const blade = attribute('aBlade', 'vec4'); // x random, y lean, z head, w blade type
  const t = uv().y;
  const band = (v: N, lo: number, hi: number) => step(lo, v).mul(float(1).sub(step(hi, v)));
  const isSeed = band(blade.z, 0.5, 1.5), isFlower = step(1.5, blade.z);
  const isPatchB = band(blade.w, 0.5, 1.5), isHorse = band(blade.w, 1.5, 2.5), isRoot = band(blade.w, 2.5, 3.5), isSedge = step(3.5, blade.w);
  // wind: rooted bend, gust fronts travelling with the breeze, a faster flutter at the tips; horsetail and roots stiff
  const p = positionLocal;
  const along = dot(p.xz, windU.dir);
  const gust = sin(along.mul(0.45).sub(windU.time.mul(1.1))).mul(0.5).add(0.5).pow(1.6);
  const chop = sin(along.mul(1.7).sub(windU.time.mul(2.3)).add(blade.x.mul(6.28))).mul(0.5).add(0.5);
  const stiff = float(1).sub(isHorse.mul(0.7)).sub(isRoot.mul(0.85));
  const bend = gust.mul(0.75).add(chop.mul(0.25)).mul(windU.strength).mul(0.55).add(0.03).mul(stiff);
  const k = pow(t, 1.5).mul(positionGeometry.y.mul(1.0).add(0.05));
  const flutter = sin(windU.time.mul(7.0).add(blade.x.mul(17.0)).add(along.mul(2.0))).mul(0.012).mul(t).mul(stiff);
  const side = vec3(windU.dir.y.negate(), 0, windU.dir.x);
  m.positionNode = p.add(vec3(windU.dir.x, 0, windU.dir.y).mul(bend.mul(k))).add(side.mul(flutter)).sub(vec3(0, bend.mul(k).mul(0.25), 0));
  // colour: mottled live green and yellow-green with dead straw through it, golden tips, a dark base in the sward;
  // drier patches carry more straw, damp ground by the water greener and darker
  const dry = inst.x;
  const seedHead = isSeed.mul(smoothstep(0.72, 0.86, t));
  const flowerHead = isFlower.mul(smoothstep(0.78, 0.83, t));
  // dead blades: in a patch the share follows the dryness, in a tuft a fixed few plus the dry tufts' share; sedge
  // clumps hold fewer, horsetail, roots and flowers none
  const deadTuft = smoothstep(0.7, 0.78, blade.x).mul(dry.mul(0.6).add(0.4)).mul(float(1).sub(isSeed)).mul(float(1).sub(isSedge.mul(0.55)));
  const deadPatch = float(1).sub(smoothstep(dry.mul(0.6).add(0.08), dry.mul(0.6).add(0.12), blade.x));
  // (dryness > 1.2: flood wrack and the straw under the tussocks, all dead weed)
  const special = isHorse.add(isRoot).add(isFlower).clamp(0, 1);
  const deadBlade = mix(deadTuft, deadPatch, isPatchB).max(smoothstep(1.2, 1.4, dry)).mul(float(1).sub(special)).toVar();
  const nW = tnoise(positionWorld.xz.mul(0.7), 3.1);
  const nB = tnoise(positionWorld.xz.mul(2.9), 5.7);
  // live blades in blotches: deep green, sunny yellow-green and a cooler blue-green where damp
  const nC = tnoise(positionWorld.xz.mul(0.23), 8.3);
  let live: N = mix(vec3(grassU.green), vec3(grassU.yellow), smoothstep(0.25, 0.85, nW.mul(0.6).add(blade.x.mul(0.25)).add(dry.mul(0.3))));
  live = mix(live, vec3(grassU.green).mul(vec3(0.85, 1.0, 1.25)), smoothstep(0.55, 0.8, nC).mul(inst.y.mul(0.6).add(0.25)));
  // sedge: a darker, bluer green
  live = mix(live, vec3(grassU.green).mul(vec3(0.78, 0.92, 1.22)).mul(mix(0.8, 1.1, blade.x)), isSedge.mul(0.85));
  const tipDry = smoothstep(0.6, 1.0, t).mul(dry.mul(0.6).add(0.3)).mul(float(1).sub(special)).max(isSeed.mul(smoothstep(0.3, 0.85, t))).clamp(0, 1);
  let col: N = mix(live, vec3(grassU.straw), tipDry.mul(mix(0.5, 1.0, nB)));
  col = mix(col, mix(vec3(grassU.straw), vec3(grassU.dead), blade.x.mul(0.8).add(nB.mul(0.2))), deadBlade);
  // seed heads pale, the low sun catches them as a bright fuzz over the sward
  col = mix(col, vec3(grassU.straw).mul(vec3(1.45, 1.38, 1.3)), seedHead.mul(0.8));
  col = mix(col, col.mul(vec3(0.75, 0.95, 0.72)), inst.y.mul(float(1).sub(deadBlade)));
  // (grass lane: wider blade-to-blade brightness, decorrelated from the dead-blade pick, so the sward has grain)
  col = col.mul(mix(0.72, 1.25, fract(blade.x.mul(7.13)))).mul(inst.z);
  // the lower third of a blade: shaded olive with old straw through it, black only at the root (scene fix wip: a
  // near-black lower 40 % stood every near blade up as a dark bristle against the paler ground between them)
  const baseC = mix(col, vec3(grassU.dead), 0.4).mul(0.5);
  col = mix(mix(vec3(grassU.root), baseC, smoothstep(0.0, 0.12, t)), col, smoothstep(0.08, 0.36, t).max(special)).toVar();
  // horsetail: fresh green jointed stems, a darker sheath at every joint (wip: dark stems with black joints read as
  // black wires over the pale bar)
  const joint = smoothstep(0.8, 0.92, fract(t.mul(blade.x.mul(3.0).add(5.0)).add(blade.x)));
  const horse = vec3(0.062, 0.105, 0.03).mul(mix(0.8, 1.15, nB)).mul(mix(float(1), float(0.5), joint));
  col = mix(col, horse, isHorse);
  // roots: weathered grey-brown where they hang out in the air, darker at the face
  const root = mix(vec3(0.04, 0.032, 0.022), vec3(0.11, 0.094, 0.072), t.mul(0.6).add(blade.x.mul(0.4)));
  col = mix(col, root, isRoot);
  // flower heads: yarrow white, arnica yellow, lupine purple, paintbrush red, picked per instance
  const hue = fract(inst.w.mul(5.37));
  const fc = mix(mix(mix(vec3(0.5, 0.49, 0.44), vec3(0.52, 0.34, 0.03), step(0.4, hue)), vec3(0.11, 0.065, 0.27), step(0.64, hue)), vec3(0.26, 0.05, 0.03), step(0.86, hue));
  col = mix(col, fc, flowerHead);
  // (0.89 before the face normals: blades seen from their sunlit side now catch more of the low sun)
  m.colorNode = vec4(col.mul(grassU.gain).mul(0.76), 1);
  // a weak, rough reflector: at 0.25 the sky sheen made every blade read as plastic, and the sun's grazing-angle
  // sheen on the upward-bent normals lit backlit clumps white (C4); was 0.07 everywhere. Grass lane: a faint waxy
  // sheen of the sky on the upper live blades, so in shade the sward keeps some form instead of a flat dark mat
  m.specularIntensityNode = mix(float(0.05), float(0.11), smoothstep(0.35, 0.95, t)).mul(float(1).sub(deadBlade.mul(0.6)));
  m.roughnessNode = mix(mix(mix(float(0.88), float(0.74), smoothstep(0.4, 1.0, t)), float(0.95), deadBlade), float(0.95), isRoot);
  m.metalnessNode = float(0);
  // the sward shades its own lower half from the sky; the tops see the whole open dome, so tree shadows on the
  // meadow stay cool and the blades readable in them (round 1: near-black stencilled shadows); roots hang in the
  // shade under the turf lip
  // (horsetail stands alone on the open bar, no sward round it)
  // (grass lane: the tips see a little more of the sky, so shaded sward keeps its form: 1.7 -> 1.9)
  m.aoNode = mix(mix(float(0.5), float(1.9), smoothstep(0.0, 0.7, t)), float(0.6), isRoot).max(isHorse.mul(0.75));
  // the blade's own face turned toward the viewer, bent well up: a blade seen from its sunlit side is lit, one seen
  // against the sun shows its shaded face and only the translucent glow (round 2's normals leaned up whichever side
  // was seen, so backlit clumps were lit like a lawn from above and read as white fibre-optic strands in C4)
  const fN = normalize(cross(dFdx(positionView), dFdy(positionView)));
  const toEye = normalize(positionView.negate());
  const face = fN.mul(sign(dot(fN, toEye)).add(0.001));
  m.normalNode = normalize(face.mul(0.38).add(cameraViewMatrix.mul(vec4(0, 0.62, 0, 0)).xyz));
  // backlit live blades are translucent yellow-green, only toward the sun and only at the tips; dead straw lets
  // little through; seed heads and petals glow a little more; roots none
  const glow = mix(grassU.translucency, grassU.seedGlow, seedHead.max(flowerHead)).mul(float(1).sub(deadBlade.mul(0.75)))
    .mul(smoothstep(0.25, 0.9, t)).mul(float(1).sub(isRoot));
  m.thicknessColorNode = max(col, vec3(0)).mul(vec3(0.85, 1.12, 0.3)).mul(glow);
  m.thicknessDistortionNode = float(0.2);
  m.thicknessPowerNode = float(8.0);
  m.thicknessScaleNode = float(1.6);
  m.thicknessAttenuationNode = float(0.5);
  return m;
}

/** a unit sward card: x -0.5..0.5 across, y 0..1 up, two rows so the wind bends it */
function cardGeometry(): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  const pos: number[] = [], uvs: number[] = [], nor: number[] = [], idx: number[] = [];
  for (let r = 0; r <= 2; r++) {
    for (const u of [0, 1]) {
      pos.push(u - 0.5, r / 2, 0);
      nor.push(0, 0, 1);
      uvs.push(u, r / 2);
    }
  }
  idx.push(0, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4);
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  g.setIndex(idx);
  g.computeBoundingSphere();
  return g;
}

/**
 * The sward cards' program: turned about the vertical toward the camera and bent by the same wind as the blades;
 * per pixel two staggered rows of tapering, leaning blades of mixed heights over a ragged root band, coloured and lit
 * like the blades (live greens, dead straw, dry tips, the dark self-shadowed base), the rest discarded.
 */
function cardMaterial() {
  const m = new THREE.MeshSSSNodeMaterial();
  m.name = 'grass-card';
  m.side = THREE.DoubleSide;
  m.alphaTest = 0.5;
  const c = attribute('aCard', 'vec4'); // root centre x, y, z; width (m)
  const s = attribute('aCard2', 'vec4'); // height (m), dryness, dampness, seed
  const pg = positionGeometry;
  const toCam = cameraPosition.xz.sub(c.xz);
  const f0 = toCam.div(max(length(toCam), 1e-3));
  // turned up to ~30 degrees off the camera by hash, so neighbouring cards are not all parallel to the view (their
  // level tops read as rows across the meadow)
  const ya = fract(s.w.mul(7.77)).sub(0.5).mul(1.1);
  const cy = cos(ya), sy = sin(ya);
  const f = { x: f0.x.mul(cy).sub(f0.y.mul(sy)), y: f0.x.mul(sy).add(f0.y.mul(cy)) };
  const lx = pg.x.mul(c.w);
  const along = dot(c.xz, windU.dir);
  const gust = sin(along.mul(0.45).sub(windU.time.mul(1.1))).mul(0.5).add(0.5).pow(1.6);
  const chop = sin(along.mul(1.7).sub(windU.time.mul(2.3)).add(s.w.mul(6.28))).mul(0.5).add(0.5);
  const bend = gust.mul(0.75).add(chop.mul(0.25)).mul(windU.strength).mul(0.55).add(0.03);
  const k = pow(pg.y, 1.5).mul(s.x.add(0.05));
  m.positionNode = vec3(c.x.add(f.y.negate().mul(lx)), c.y.add(pg.y.mul(s.x)), c.z.add(f.x.mul(lx)))
    .add(vec3(windU.dir.x, 0, windU.dir.y).mul(bend.mul(k))).sub(vec3(0, bend.mul(k).mul(0.25), 0));

  const cu = uv();
  const U = cu.x.mul(CARD_BLADES), V = cu.y;
  // a blade never narrower than about a pixel: far cards merge into a dense ragged-topped carpet
  const fw = fwidth(U);
  const seed = s.w.mul(173.0);
  // low at the card's ends, so the cards overlap into one sward instead of a row of strips
  const ends = smoothstep(0.0, 0.25, cu.x).mul(smoothstep(0.0, 0.25, float(1).sub(cu.x)));
  let cov: N = float(0), rr: N = float(0), tt: N = float(0);
  const rows: Array<[number, number, number]> = [[0, 0, 1], [1, 0.5, 0.7]];
  for (const [row, off, hs] of rows) {
    const cell = floor(U.add(off));
    for (const kk of [-1, 0, 1]) {
      const i = cell.add(kk);
      const n = i.add(seed).add(row * 41.0);
      const r1 = fract(sin(n.mul(12.9898)).mul(43758.5453));
      const r2 = fract(sin(n.mul(78.233)).mul(43758.5453));
      const r3 = fract(sin(n.mul(37.719)).mul(43758.5453));
      // (wip: tips at much the same height drew each card's top as a level dash across the meadow)
      const H = r2.mul(r2).mul(0.75).add(0.25).mul(hs).mul(ends.mul(0.4).add(0.6));
      const tb = V.div(H);
      // leaning and bowed: the tip up to a blade spacing to either side
      const cx = i.sub(off).add(r1.mul(0.6).add(0.2)).add(r3.sub(0.5).mul(2.0).mul(tb).mul(tb));
      const half = max(float(0.16).mul(float(1).sub(tb).max(0.0).pow(0.8)), fw.mul(0.55));
      const ins = step(abs(U.sub(cx)), half).mul(step(tb, 1.0));
      const nw = ins.mul(float(1).sub(cov));
      rr = rr.add(r1.mul(nw));
      tt = tt.add(tb.mul(nw));
      cov = max(cov, ins);
    }
  }
  // the dense base of the sward, ragged along its top
  const band = float(0.1).add(sin(U.mul(1.3).add(seed)).mul(0.04)).add(sin(U.mul(3.7).add(seed.mul(1.7))).mul(0.025)).mul(ends);
  const inBand = step(V, band).mul(float(1).sub(cov));
  rr = rr.add(fract(seed.mul(0.37).add(floor(U).mul(0.61))).mul(inBand));
  cov = max(cov, inBand);

  const dry = s.y, damp = s.z;
  const nW = tnoise(positionWorld.xz.mul(0.7), 3.1);
  const nB = tnoise(positionWorld.xz.mul(2.9), 5.7);
  let live: N = mix(vec3(grassU.green), vec3(grassU.yellow), smoothstep(0.25, 0.85, nW.mul(0.6).add(rr.mul(0.25)).add(dry.mul(0.3))));
  live = mix(live, live.mul(vec3(0.75, 0.95, 0.72)), damp);
  const dead = float(1).sub(step(dry.mul(0.5).add(0.08), fract(rr.mul(7.31))));
  const tipDry = smoothstep(0.55, 1.0, tt).mul(dry.mul(0.5).add(0.25));
  let col: N = mix(live, vec3(grassU.straw), tipDry.mul(mix(0.5, 1.0, nB)));
  col = mix(col, mix(vec3(grassU.straw), vec3(grassU.dead), rr.mul(0.8).add(nB.mul(0.2))), dead);
  col = col.mul(mix(0.62, 1.25, fract(rr.mul(3.7))));
  // the sward shades its own base, olive with old straw in it (wip: a near-black base on every card drew dark rows
  // across the meadow at grazing angles, like a ploughed field)
  col = mix(col, vec3(grassU.dead).mul(0.6), float(1).sub(smoothstep(0.0, 0.22, V)).mul(0.35)).mul(mix(0.62, 1.0, smoothstep(0.0, 0.5, V)));
  m.colorNode = vec4(col.mul(grassU.gain).mul(0.76), 1);
  m.opacityNode = cov;
  // the blades' faint sheen and sky fill (see grassMaterial)
  m.specularIntensityNode = mix(float(0.05), float(0.1), smoothstep(0.4, 1.0, V)).mul(float(1).sub(dead.mul(0.6)));
  m.roughnessNode = mix(mix(float(0.88), float(0.76), smoothstep(0.4, 1.0, V)), float(0.95), dead);
  m.metalnessNode = float(0);
  m.aoNode = mix(float(0.9), float(1.8), smoothstep(0.0, 0.65, V));
  const toEye = normalize(positionView.negate());
  m.normalNode = normalize(toEye.mul(0.38).add(cameraViewMatrix.mul(vec4(0, 0.62, 0, 0)).xyz));
  const glow = grassU.translucency.mul(float(1).sub(dead.mul(0.75))).mul(smoothstep(0.25, 0.9, V));
  m.thicknessColorNode = max(col, vec3(0)).mul(vec3(0.85, 1.12, 0.3)).mul(glow);
  m.thicknessDistortionNode = float(0.2);
  m.thicknessPowerNode = float(8.0);
  m.thicknessScaleNode = float(1.6);
  m.thicknessAttenuationNode = float(0.5);
  return m;
}

interface Card { x: number; y: number; z: number; w: number; h: number; dry: number; damp: number; seed: number }

const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion(), _p = new THREE.Vector3(), _s = new THREE.Vector3(), _ax = new THREE.Vector3();
const _vp = new THREE.Matrix4(), _fr = new THREE.Frustum(), _box = new THREE.Box3();
const UP = new THREE.Vector3(0, 1, 0);

export class Grass {
  readonly group = new THREE.Group();
  private meshes: THREE.InstancedMesh[] = [];
  private attrs: THREE.InstancedBufferAttribute[] = [];
  private chunks = new Map<number, Tuft[]>();
  private lastPos = new THREE.Vector3(1e9, 0, 0);
  private lastVP = new THREE.Matrix4();
  private cp: CorePoint = { waterY: 0, edge: 0, apron: 0, rock: 0, belowTop: 3, gravel: 0, y: 0 };
  private cv: [number, number] = [0, 0];
  private md = { bare: 0, straw: 0 };
  private order: Array<{ i: number; j: number; d: number }> = [];
  private cards = new Map<number, Card[]>();
  private cardMesh: THREE.InstancedMesh;
  private cardA: THREE.InstancedBufferAttribute;
  private cardB: THREE.InstancedBufferAttribute;
  /** card chunks still to generate (a few per frame, so walking never stalls on a ring of new chunks) */
  private pending = false;
  stats = { tufts: 0, chunks: 0, cards: 0 };

  constructor(private core: CoreSampler, private canopy: CanopyField, capScale = 1) {
    this.group.name = 'grass';
    {
      const cap = Math.round(14000 * capScale);
      const g = cardGeometry();
      this.cardA = new THREE.InstancedBufferAttribute(new Float32Array(cap * 4), 4);
      this.cardB = new THREE.InstancedBufferAttribute(new Float32Array(cap * 4), 4);
      g.setAttribute('aCard', this.cardA);
      g.setAttribute('aCard2', this.cardB);
      // (the instance matrices stay identity: the program places and turns each card from aCard)
      const im = (this.cardMesh = new THREE.InstancedMesh(g, cardMaterial(), cap));
      im.count = 0;
      im.frustumCulled = false;
      im.castShadow = false;
      im.receiveShadow = true;
      im.layers.set(LAYER_NO_REFLECT);
      im.name = 'grass-cards';
    }
    const mat = grassMaterial();
    for (const { spec, cap: c } of SPECS) {
      const g = bladeGeometry(spec);
      const cap = Math.max(1025, Math.round(c * capScale));
      // default usage (see Forest.ts makePart): update() uploads just the packed range when the camera has moved
      const a = new THREE.InstancedBufferAttribute(new Float32Array(cap * 4), 4);
      g.setAttribute('aGrass', a);
      const im = new THREE.InstancedMesh(g, mat, cap);
      im.count = 0;
      im.frustumCulled = false;
      im.castShadow = false;
      im.receiveShadow = true;
      im.layers.set(LAYER_NO_REFLECT);
      im.name = `grass-${this.meshes.length}`;
      this.meshes.push(im);
      this.attrs.push(a);
      this.group.add(im);
    }
    // after the blades: they cover much of the cards near the camera, and the discarding program draws last
    this.group.add(this.cardMesh);
  }

  /** the sward cards of one chunk: one every ~0.42 m wherever the sward grows, as tall as its patches */
  private cardChunk(ci: number, cj: number): Card[] {
    const k = this.key(ci, cj);
    let list = this.cards.get(k);
    if (list) return list;
    list = [];
    const cp = this.cp;
    const o = { bar: 0, lip: 0, gx: 0, gz: 0, ny: 1 };
    let n = 0;
    for (let a = 0; a < CHUNK / CARD_STEP; a++) {
      for (let b = 0; b < CHUNK / CARD_STEP; b++, n++) {
        const x = ci * CHUNK + (a + hash(k, n, 121)) * CARD_STEP, z = cj * CHUNK + (b + hash(k, n, 122)) * CARD_STEP;
        const p = this.ground(x, z, o);
        // (not on the gravel or hanging over a cut bank's lip: a strip of sward standing there reads as a cut-out)
        if (p <= 0.05 || o.bar > 0.25 || o.lip || hash(k, n, 123) >= p * 1.15) continue;
        const c1 = fbm(x * 2.6 + 1.3, z * 2.6 - 0.7, 1), c2 = fbm(x * 0.95 + 4.2, z * 0.95, 2), c3 = fbm(x * 0.3, z * 0.3 - 2.6, 2);
        const damp = sstep(9, 2, cp.edge) * 0.8;
        const dry = Math.max(0, Math.min(1, Math.max(0.3 + c3 * 0.6 - damp * 0.4 + (hash(k, n, 124) - 0.5) * 0.3, this.md.straw * 0.95 - damp * 0.3)));
        // the patches' height field (clumps and hollows at three scales), a little under their tallest blades
        const tall = (0.5 + 0.95 * sstep(-0.35, 0.5, c2) + 0.45 * sstep(-0.2, 0.5, c3) + damp * 0.35) * (0.62 + 0.62 * sstep(-0.35, 0.4, c1));
        // a card turns toward the camera, so by a path or the yard it is narrower and does not reach across them
        const narrow = Math.min(1, 0.35 + p, Math.max(0.3, (pathDist(x, z) - 0.4) / 0.6));
        list.push({
          x, y: cp.y - 0.02, z, w: (0.42 + 0.2 * hash(k, n, 125)) * narrow,
          h: 0.26 * tall * (0.5 + 1.0 * hash(k, n, 126)) * (0.55 + 0.45 * Math.min(1, p * 1.5)), dry, damp, seed: hash(k, n, 127),
        });
      }
    }
    this.cards.set(k, list);
    return list;
  }

  private key(i: number, j: number) {
    return (i + 2048) * 4096 + (j + 2048);
  }

  /**
   * Where grass grows at (x, z) (0..1, from the masks shared by the patches and the tufts), with the point's samples
   * left in `cp`; also returns the bar fraction, the lip flag and the ground's slope through `o`.
   */
  private ground(x: number, z: number, o: { bar: number; lip: number; gx: number; gz: number; ny: number }): number {
    const cp = this.cp, cv = this.cv;
    if (!this.core.at(x, z, cp)) return 0;
    if (cp.y < cp.waterY + 0.03 || cp.edge < 0.25) return 0;
    // ground for grass: none on the bar, shoots up through its top edge
    const bar = sstep(0.12, 0.55, cp.gravel);
    // grass and weeds creep a little way into the gravel at the bar's top, in broken patches
    const creep = sstep(-0.1, 0.35, fbm(x * 0.9 + 3.3, z * 0.9 - 1.1, 2));
    let p = (1 - bar) + bar * 0.55 * creep * (1 - sstep(0.6, 0.97, cp.gravel)) * sstep(0.06, 0.35, cp.y - cp.waterY);
    meadowAt(x, z, this.md);
    // (scene fix: worn patches thin the sward rather than bare it; the ground shows only on the paths)
    p *= 1 - 0.45 * this.md.bare;
    // not on bare steep faces (the turf lip at the top of a cut bank keeps its grass)
    const e = 0.35;
    const gx = (this.core.groundY(x + e, z) - this.core.groundY(x - e, z)) / (2 * e);
    const gz = (this.core.groundY(x, z + e) - this.core.groundY(x, z - e)) / (2 * e);
    const ny = 1 / Math.hypot(gx, 1, gz);
    const lip = cp.belowTop < 0.22 && ny < 0.9 ? 1 : 0;
    p *= Math.max(sstep(0.72, 0.86, ny), lip * 0.9);
    // footpaths trampled bare (their edges ragged), the camp worn, little grass under dense crowns
    p *= sstep(0.45, 1.0, pathDist(x, z) + 0.25 * fbm(x * 2.3 + 1.7, z * 2.3 - 4.1, 1));
    p *= sstep(2.2, 3.4, Math.hypot(x - LAYOUT.campfire.x, z - LAYOUT.campfire.z));
    p *= 0.35 + 0.65 * sstep(5.0, 7.0, Math.hypot(x - LAYOUT.cabin.x, z - LAYOUT.cabin.z));
    // none under the cabin, its porch and steps, the woodpile or in the fire ring (part 4)
    p *= campGrassMask(x, z);
    this.canopy.sample(x, z, cv);
    // and none in the litter ring at the foot of a trunk
    p *= 1 - sstep(0.25, 0.75, cv[0] + cv[1] * 0.75);
    o.bar = bar;
    o.lip = lip;
    o.gx = gx;
    o.gz = gz;
    o.ny = ny;
    return p;
  }

  /** deterministic patches and tufts of one chunk */
  private chunk(ci: number, cj: number): Tuft[] {
    const k = this.key(ci, cj);
    let list = this.chunks.get(k);
    if (list) return list;
    list = [];
    const cp = this.cp;
    const o = { bar: 0, lip: 0, gx: 0, gz: 0, ny: 1 };
    const push = (t: Omit<Tuft, 'tilt' | 'tiltDir'> & Partial<Pick<Tuft, 'tilt' | 'tiltDir'>>) =>
      list!.push({ tilt: 0, tiltDir: 0, ...t });
    // the sward: a patch every ~0.3 m wherever grass grows (round 2: 0.42 m, the ground showed between them), its
    // shape and height following clumps and hollows at three scales and the damp
    {
      const step = 0.3;
      let n = 0;
      for (let a = 0; a < CHUNK / step; a++) {
        for (let b = 0; b < CHUNK / step; b++, n++) {
          const x = ci * CHUNK + (a + hash(k, n, 21)) * step, z = cj * CHUNK + (b + hash(k, n, 22)) * step;
          const p = this.ground(x, z, o);
          if (p <= 0 || hash(k, n, 23) >= p * (o.bar > 0.3 ? 0.6 : 1)) continue;
          const c1 = fbm(x * 2.6 + 1.3, z * 2.6 - 0.7, 1), c2 = fbm(x * 0.95 + 4.2, z * 0.95, 2), c3 = fbm(x * 0.3, z * 0.3 - 2.6, 2);
          const damp = sstep(9, 2, cp.edge) * 0.8;
          const dry = Math.max(0, Math.min(1, Math.max(0.3 + c3 * 0.6 - damp * 0.4 + (hash(k, n, 24) - 0.5) * 0.3, this.md.straw * 0.95 - damp * 0.3)));
          // uneven: ankle high on dry knolls and by paths, knee high in the damp hollows and tussocky patches, with
          // small hollows and humps a few tens of centimetres across
          const tall = (0.5 + 0.95 * sstep(-0.35, 0.5, c2) + 0.45 * sstep(-0.2, 0.5, c3) + damp * 0.35) * (0.62 + 0.62 * sstep(-0.35, 0.4, c1));
          const kr = hash(k, n, 29);
          // soft lodged grass in the damp and tall patches, upright on the dry knolls, mixed elsewhere; now and then a
          // taller upright clump standing out of the sward (grass lane)
          const clumpUp = o.bar < 0.3 && hash(k, n, 39) < 0.07;
          const kind = clumpUp ? K_PATCH_C : kr < 0.2 + 0.35 * sstep(0.0, 0.6, c2 + damp * 0.5) ? K_PATCH_B : kr > 0.78 - 0.3 * sstep(0.3, 0.8, dry) ? K_PATCH_C : K_PATCH_A;
          push({
            x, y: cp.y - 0.01, z, yaw: hash(k, n, 25) * Math.PI * 2,
            h: tall * (0.5 + 1.0 * hash(k, n, 26)) * (clumpUp ? 1.55 : 1) * (o.bar > 0.5 ? 0.55 : 1),
            w: 0.85 + 0.35 * hash(k, n, 27), kind, dry, damp, rank: hash(k, n, 28),
          });
          // infill between the patches, drawn only within ~8 m: the dense sward round the angler's feet (grass lane:
          // the near grass read thin and uniform)
          if (o.bar < 0.3) {
            const ix = x + (hash(k, n, 141) - 0.5) * step, iz = z + (hash(k, n, 142) - 0.5) * step;
            push({
              x: ix, y: cp.y - 0.01, z: iz, yaw: hash(k, n, 143) * Math.PI * 2, h: tall * (0.4 + 0.9 * hash(k, n, 144)),
              w: 0.8 + 0.3 * hash(k, n, 145), kind: hash(k, n, 146) < 0.45 ? K_PATCH_B : K_PATCH_A,
              dry: Math.min(1, dry + (hash(k, n, 147) - 0.4) * 0.3), damp, rank: hash(k, n, 148), near: true,
            });
          }
          // a few seed stems through the sward itself, more where it is dry
          if (o.bar < 0.3 && hash(k, n, 149) < 0.025 + 0.04 * dry) {
            push({
              x: x + (hash(k, n, 150) - 0.5) * 0.2, y: cp.y - 0.01, z: z + (hash(k, n, 151) - 0.5) * 0.2, yaw: hash(k, n, 152) * Math.PI * 2,
              h: (0.55 + 0.5 * hash(k, n, 153)) * Math.min(1.3, tall), w: 0.9, kind: K_SEED, dry: Math.max(0.5, dry), damp, rank: hash(k, n, 154),
            });
          }
          // dead straw lying at the base of the sward where it is dry (last year's grass, flattened)
          if (o.bar < 0.3 && hash(k, n, 30) < 0.07 + 0.16 * sstep(0.35, 0.8, dry)) {
            push({
              x: x + (hash(k, n, 31) - 0.5) * 0.2, y: cp.y - 0.005, z: z + (hash(k, n, 32) - 0.5) * 0.2, yaw: hash(k, n, 33) * Math.PI * 2,
              h: 0.45 + 0.4 * hash(k, n, 34), w: 0.8 + 0.5 * hash(k, n, 35), kind: K_WRACK, dry: 1.5, damp, rank: hash(k, n, 36),
              tilt: 1.2 + 0.25 * hash(k, n, 37), tiltDir: hash(k, n, 38) * Math.PI * 2,
            });
          }
        }
      }
    }
    // tussocks gathered into clumps, tall dry grass, seed heads and flowers above the sward; tufts hanging over the
    // lip of a cut bank
    {
      const step = 0.6;
      let n = 0;
      for (let a = 0; a < CHUNK / step; a++) {
        for (let b = 0; b < CHUNK / step; b++, n++) {
          const x = ci * CHUNK + (a + hash(k, n, 1)) * step, z = cj * CHUNK + (b + hash(k, n, 2)) * step;
          let p = this.ground(x, z, o);
          if (p <= 0) continue;
          const y0 = cp.y, lip = o.lip, out = Math.atan2(o.gz, o.gx);
          // tussocks at three scales
          const c1 = fbm(x * 3.1, z * 3.1 + 1.7, 1), c2 = fbm(x * 0.95 + 4.2, z * 0.95, 2), c3 = fbm(x * 0.3, z * 0.3 - 2.6, 2);
          const clump = sstep(-0.1, 0.45, fbm(x * 0.55 - 3.7, z * 0.55 + 8.1, 2));
          p *= Math.max(0, 0.1 + c1 * 0.5 + c2 * 0.8 + c3 * 0.5) * (0.35 + 1.1 * clump);
          if (lip) p = Math.max(p, 0.8);
          const bar = o.bar;
          const damp = sstep(9, 2, cp.edge) * 0.8;
          const dryness = Math.max(0, Math.min(1, Math.max(0.25 + c3 * 0.6 - damp * 0.4 + (hash(k, n, 4) - 0.5) * 0.3, this.md.straw * 0.85 - damp * 0.3)));
          // flowers in drifts over the open, sunny meadow, away from the water (not on the trodden edges of the
          // paths and the yard)
          const drift = sstep(-0.05, 0.4, fbm(x * 0.11 + 2.9, z * 0.11 - 6.3, 2)) * sstep(4, 9, cp.edge) * (1 - bar);
          if (drift > 0 && hash(k, n, 12) < 0.16 * drift * Math.min(1, p * 3 + 0.3) && pathDist(x, z) > 1.4 && campGrassMask(x, z) > 0.95) {
            push({
              x: x + (hash(k, n, 13) - 0.5) * 0.3, y: y0 - 0.01, z: z + (hash(k, n, 14) - 0.5) * 0.3, yaw: hash(k, n, 15) * Math.PI * 2,
              h: 0.7 + 0.6 * hash(k, n, 16), w: 0.8 + 0.4 * hash(k, n, 17), kind: K_FLOWER, dry: 0.2, damp, rank: hash(k, n, 18),
            });
          }
          if (hash(k, n, 3) >= p * 0.85) continue;
          const kindR = hash(k, n, 5);
          // seed heads in drifts over the sunny meadow, tall dry grass on the dry patches, tussocks elsewhere
          const seeds = bar < 0.3 && hash(k, n, 11) < 0.12 + 0.3 * sstep(0.2, 0.6, dryness) * sstep(5, 12, cp.edge);
          const big = !lip && clump > 0.55 && hash(k, n, 19) < 0.45;
          const kind = seeds ? K_SEED : big ? K_TUSSOCK_B : c2 > 0.1 && kindR < 0.7 ? K_TUSSOCK : dryness > 0.42 && kindR < 0.5 ? K_TALL : K_TUSSOCK;
          const tall = (0.65 + 0.6 * Math.max(0, c2 + 0.4) + damp * 0.3) * (0.7 + 0.7 * clump);
          // (lip tufts lean downhill, out over the edge: the gradient points uphill; before the scene fix they leaned
          // back up the bank)
          push({
            x, y: y0 - 0.015, z, yaw: hash(k, n, 6) * Math.PI * 2, h: tall * (0.65 + 0.7 * hash(k, n, 7)) * (bar > 0.5 ? 0.7 : 1) * (1 + lip * 0.3),
            w: (0.8 + 0.45 * hash(k, n, 8)) * (big ? 1.15 : 1), kind: lip ? K_TUSSOCK : kind, dry: dryness, damp, rank: hash(k, n, 9),
            tilt: lip * (0.55 + 0.35 * hash(k, n, 10)), tiltDir: out,
          });
          // a clump is several tussocks grown together, with dead straw lying round its foot
          if (kind === K_TUSSOCK || kind === K_TUSSOCK_B) {
            const sat = Math.floor(clump * 3.2 * hash(k, n, 40));
            for (let s = 0; s < sat; s++) {
              const aa = hash(k, n, 41 + s) * Math.PI * 2, rr = 0.12 + 0.2 * hash(k, n, 45 + s);
              const sx = x + Math.cos(aa) * rr, sz = z + Math.sin(aa) * rr;
              if (this.ground(sx, sz, o) <= 0.2) continue;
              push({
                x: sx, y: cp.y - 0.015, z: sz, yaw: hash(k, n, 49 + s) * Math.PI * 2, h: tall * (0.55 + 0.6 * hash(k, n, 53 + s)),
                w: 0.75 + 0.4 * hash(k, n, 57 + s), kind: hash(k, n, 61 + s) < 0.35 ? K_TUSSOCK_B : K_TUSSOCK, dry: Math.min(1, dryness + 0.1), damp,
                rank: hash(k, n, 65 + s),
              });
            }
            if (hash(k, n, 70) < 0.6) {
              push({
                x: x + (hash(k, n, 71) - 0.5) * 0.25, y: y0 - 0.005, z: z + (hash(k, n, 72) - 0.5) * 0.25, yaw: hash(k, n, 73) * Math.PI * 2,
                h: 0.6 + 0.5 * hash(k, n, 74), w: 1.0 + 0.6 * hash(k, n, 75), kind: K_WRACK, dry: 1.5, damp, rank: hash(k, n, 76),
                tilt: 1.15 + 0.3 * hash(k, n, 77), tiltDir: hash(k, n, 78) * Math.PI * 2,
              });
            }
          }
        }
      }
    }
    // by the water: sedge clumps on the damp margins and banks, horsetail colonies on damp sand and gravel near the
    // waterline, roots hanging out of the cut faces under the turf lip
    {
      const step = 0.5;
      let n = 0;
      const sq: CorePoint = { waterY: 0, edge: 0, apron: 0, rock: 0, belowTop: 3, gravel: 0, y: 0 };
      for (let a = 0; a < CHUNK / step; a++) {
        for (let b = 0; b < CHUNK / step; b++, n++) {
          const x = ci * CHUNK + (a + hash(k, n, 81)) * step, z = cj * CHUNK + (b + hash(k, n, 82)) * step;
          if (!this.core.at(x, z, sq)) continue;
          const above = sq.y - sq.waterY;
          if (above < 0.02 || sq.edge < 0.1 || sq.edge > 7) continue;
          if (pathDist(x, z) < 1.0 || campGrassMask(x, z) < 0.5 || Math.hypot(x - LAYOUT.campfire.x, z - LAYOUT.campfire.z) < 4) continue;
          // (not in front of the fixed camera spots' lenses, nor in C2's clear shallows round its splash)
          if (EYES.some(([ex, ez]) => Math.hypot(x - ex, z - ez) < 2.4) || Math.hypot(x - 3.7, z - 3.35) < 2.2) continue;
          if (this.canopy.sample(x, z, this.cv)[1] > 0.5) continue;
          const e = 0.3;
          const gx = (this.core.groundY(x + e, z) - this.core.groundY(x - e, z)) / (2 * e);
          const gz = (this.core.groundY(x, z + e) - this.core.groundY(x, z - e)) / (2 * e);
          const ny = 1 / Math.hypot(gx, 1, gz);
          // roots: out of earth faces 5-40 cm under the lip, drooping out and down over the face
          if (ny < 0.8 && sq.rock < 0.8 && sq.belowTop > 0.05 && sq.belowTop < 0.42) {
            const pr = 0.55 * sstep(-0.25, 0.25, fbm(x * 0.7 + 5.3, z * 0.7 - 2.2, 2));
            if (hash(k, n, 83) < pr) {
              const face = Math.acos(Math.min(1, ny));
              const gl = Math.max(Math.hypot(gx, gz), 1e-3);
              push({
                x: x - (gx / gl) * 0.03, y: sq.y + 0.02, z: z - (gz / gl) * 0.03, yaw: hash(k, n, 84) * Math.PI * 2,
                h: 0.7 + 0.8 * hash(k, n, 85), w: 0.8 + 0.5 * hash(k, n, 86), kind: K_ROOT, dry: 0, damp: 0, rank: hash(k, n, 87),
                tilt: Math.PI - (Math.PI / 2 - face + 0.25 + 0.3 * hash(k, n, 88)), tiltDir: Math.atan2(gz, gx),
              });
            }
            continue;
          }
          if (ny < 0.82 || sq.rock > 0.6) continue;
          // horsetail: colonies on damp sand and gravel just above the water (wip: thinner, closer to the water and
          // clear of the dock, where a colony read as black wires over the pale bar)
          const colony = sstep(0.1, 0.4, fbm(x * 0.21 - 7.7, z * 0.21 + 3.1, 2));
          const wetGround = sstep(0.5, 0.12, above) * sstep(0.1, 0.6, sq.edge) * (1 - sstep(3.0, 4.5, sq.edge));
          const byDock = Math.abs(x - LAYOUT.dock.x) < 1.7 && z > LAYOUT.dock.zTip - 0.5 && z < LAYOUT.dock.zBase + 1.5;
          if (!byDock && hash(k, n, 89) < 0.55 * colony * wetGround) {
            for (let s = 0; s < 3; s++) {
              const hx = x + (hash(k, n, 90 + s) - 0.5) * 0.45, hz = z + (hash(k, n, 93 + s) - 0.5) * 0.45;
              push({
                x: hx, y: this.core.groundY(hx, hz) - 0.01, z: hz, yaw: hash(k, n, 96 + s) * Math.PI * 2, h: 0.7 + 0.6 * hash(k, n, 99 + s),
                w: 0.85 + 0.4 * hash(k, n, 102 + s), kind: K_HORSETAIL, dry: 0, damp: 1, rank: hash(k, n, 105 + s),
              });
            }
            continue;
          }
          // sedge: clumps along the damp margins and up the lower bank, a few out on the bar's top among the stones
          const sedgeN = sstep(-0.05, 0.35, fbm(x * 0.33 + 4.4, z * 0.33 - 9.2, 2));
          const marg = sstep(0.05, 0.25, above) * (1 - sstep(1.2, 2.2, above)) * (1 - sstep(4.5, 7, sq.edge));
          const onBar = sq.gravel > 0.5 ? 0.25 : 1;
          if (hash(k, n, 108) < 0.5 * sedgeN * marg * onBar) {
            push({
              x, y: sq.y - 0.02, z, yaw: hash(k, n, 109) * Math.PI * 2, h: 0.75 + 0.6 * hash(k, n, 110), w: 0.85 + 0.5 * hash(k, n, 111),
              kind: K_SEDGE, dry: 0.15 + 0.35 * hash(k, n, 112), damp: 0.9, rank: hash(k, n, 113),
            });
          }
        }
      }
    }
    // the flood wrack along the bar's top: mats of dead weed laid flat where the last high water left them (with the
    // sticks and bark in BankProps)
    {
      const step = 0.5;
      let n = 0;
      for (let a = 0; a < CHUNK / step; a++) {
        for (let b = 0; b < CHUNK / step; b++, n++) {
          const x = ci * CHUNK + (a + hash(k, n, 41)) * step, z = cj * CHUNK + (b + hash(k, n, 42)) * step;
          if (!this.core.at(x, z, cp) || cp.edge < 0) continue;
          const lw = wrackLine(cp);
          if (lw <= 0 || hash(k, n, 43) >= lw * 0.5 * sstep(-0.2, 0.3, fbm(x * 0.35 + 1.1, z * 0.35, 2))) continue;
          if (pathDist(x, z) < 0.8 || campGrassMask(x, z) < 0.5) continue;
          push({
            x, y: cp.y - 0.005, z, yaw: hash(k, n, 44) * Math.PI * 2, h: 0.6 + 0.6 * hash(k, n, 45), w: 1.0 + 0.6 * hash(k, n, 46),
            kind: K_WRACK, dry: 1.5, damp: 0, rank: hash(k, n, 47), tilt: 1.15 + 0.3 * hash(k, n, 48), tiltDir: hash(k, n, 49) * Math.PI * 2,
          });
        }
      }
    }
    // by rank, so update() stops at the first tuft a far chunk thins out
    list.sort((a2, b2) => a2.rank - b2.rank);
    this.chunks.set(k, list);
    return list;
  }

  update(camera: THREE.PerspectiveCamera, force = false) {
    camera.updateMatrixWorld();
    _vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const cp = camera.position;
    if (!force && !this.pending && cp.distanceToSquared(this.lastPos) < 1e-6 && _vp.equals(this.lastVP)) return;
    this.lastPos.copy(cp);
    this.lastVP.copy(_vp);
    _fr.setFromProjectionMatrix(_vp, camera.coordinateSystem, camera.reversedDepth);
    const counts = new Array(KINDS).fill(0);
    const i0 = Math.floor((cp.x - CARD_RANGE) / CHUNK), i1 = Math.floor((cp.x + CARD_RANGE) / CHUNK);
    const j0 = Math.floor((cp.z - CARD_RANGE) / CHUNK), j1 = Math.floor((cp.z + CARD_RANGE) / CHUNK);
    // nearest chunks first: the blades are opaque, so near ones drawn first spare the far ones' shading (early z)
    const order = this.order;
    order.length = 0;
    for (let i = i0; i <= i1; i++) {
      for (let j = j0; j <= j1; j++) {
        const cx = (i + 0.5) * CHUNK, cz = (j + 0.5) * CHUNK;
        const d = Math.hypot(cx - cp.x, cz - cp.z);
        if (d > CARD_RANGE + CHUNK) continue;
        const gy = this.core.groundY(cx, cz);
        _box.min.set(i * CHUNK, gy - 4, j * CHUNK);
        _box.max.set((i + 1) * CHUNK, gy + 4, (j + 1) * CHUNK);
        if (!_fr.intersectsBox(_box)) continue;
        order.push({ i, j, d });
      }
    }
    order.sort((a, b) => a.d - b.d);
    for (const { i, j, d: dc } of order) {
      if (dc > RANGE + CHUNK) continue;
      // the most any tuft of this chunk keeps (the tufts' rule at the chunk's nearest point)
      const kMax = 1 - 0.75 * sstep(9, RANGE, dc - CHUNK * 0.71);
      for (const t of this.chunk(i, j)) {
        if (t.rank > kMax) break;
        const d = Math.hypot(t.x - cp.x, t.y - cp.y, t.z - cp.z);
        if (d > RANGE) continue;
        // thin out with distance; the survivors grow wider so the far sward keeps its body (the patches thin
        // sooner: up close they carry the sward, further out the ground's sward colouring does)
        // seen from high above (C6) the patches thin further: blades from the top are lines over the ground, whose
        // own sward colouring reads better there
        const elev = sstep(4, 16, cp.y - t.y);
        const patch = isPatch(t.kind);
        const keep0 = patch ? (1 - 0.8 * sstep(4, RANGE - 6, d)) * (1 - 0.6 * elev) : 1 - 0.75 * sstep(9, RANGE, d);
        // (the patches are twice round 2's density for the near sward; from 5-11 m on, half of them go, back to about
        // round 2's count: their thin blades cost the most there, seen edge-on in layers, and the ground carries it)
        // (grass lane: past ~10 m the sward cards carry the sward, so most of the far patches go: they cost the most
        // and showed as separate dots on the ground)
        const keep = patch ? keep0 * (1 - 0.5 * sstep(5, 11, d)) * (1 - 0.75 * sstep(9, 20, d)) : keep0;
        if (t.rank > keep) continue;
        if (t.near && t.rank > 1 - sstep(4, 8, d)) continue;
        const fade = 1 - sstep(RANGE - 8, RANGE, d);
        // a blade is 3-9 mm wide: past ~5 m it is under a pixel, so distant blades widen to stay ~1 px
        const widen = Math.max(1, 0.25 + d * 0.15) * (patch ? Math.sqrt(1 / Math.max(keep0, 0.3)) : 1);
        const c = counts[t.kind];
        const im = this.meshes[t.kind];
        if (c >= im.instanceMatrix.count) continue;
        _q.setFromAxisAngle(UP, t.yaw);
        if (t.tilt > 0) {
          _ax.set(-Math.sin(t.tiltDir), 0, Math.cos(t.tiltDir));
          _q.premultiply(_q2.setFromAxisAngle(_ax, t.tilt));
        }
        _m.compose(_p.set(t.x, t.y, t.z), _q, _s.set(t.w * widen, t.h * fade, t.w * widen));
        (im.instanceMatrix.array as Float32Array).set(_m.elements, c * 16);
        this.attrs[t.kind].setXYZW(c, t.dry, t.damp, 0.9 + 0.2 * t.rank, t.rank);
        counts[t.kind] = c + 1;
      }
    }
    // the sward cards, nearest chunks first; new chunks a few per frame after the first (loading) update
    let nc = 0, budget = this.stats.chunks ? 8 : 1e9;
    this.pending = false;
    const ca = this.cardA.array as Float32Array, cb = this.cardB.array as Float32Array, cap = this.cardMesh.instanceMatrix.count;
    for (const { i, j } of order) {
      if (!this.cards.has(this.key(i, j))) {
        if (budget-- <= 0) {
          this.pending = true;
          continue;
        }
      }
      for (const t of this.cardChunk(i, j)) {
        if (nc >= cap) break;
        const d = Math.hypot(t.x - cp.x, t.y - cp.y, t.z - cp.z);
        if (d > CARD_RANGE) continue;
        // a low root band near the camera (the blades stand in it, and a card turning with an orbiting camera shows
        // least there), full height from ~7 m, sunk into the ground toward the far edge, where the ground's own sward
        // colouring carries on; seen steeply from above a card is a strip lying across the view (wip: short dark
        // dashes under the camera, and C6), so it sinks from ~22 to ~42 degrees down
        const tanE = (cp.y - t.y) / Math.max(Math.hypot(t.x - cp.x, t.z - cp.z), 0.1);
        const hs = (0.22 + 0.78 * sstep(1.5, 7.0, d)) * (1 - sstep(CARD_RANGE - 8, CARD_RANGE, d)) * (1 - sstep(0.4, 0.9, tanE));
        if (hs < 0.04) continue;
        ca[nc * 4] = t.x;
        ca[nc * 4 + 1] = t.y;
        ca[nc * 4 + 2] = t.z;
        ca[nc * 4 + 3] = t.w;
        cb[nc * 4] = t.h * hs;
        cb[nc * 4 + 1] = t.dry;
        cb[nc * 4 + 2] = t.damp;
        cb[nc * 4 + 3] = t.seed;
        nc++;
      }
    }
    this.stats.cards = nc;
    if (!nc) {
      // one empty card, so the program is still drawn (and compiled during the warm-up) where no grass is in view
      ca.fill(0, 0, 4);
      cb.fill(0, 0, 4);
      nc = 1;
    }
    this.cardMesh.count = nc;
    for (const a of [this.cardA, this.cardB]) {
      a.clearUpdateRanges();
      a.addUpdateRange(0, nc * 4);
      a.needsUpdate = true;
    }
    this.meshes.forEach((im, k) => {
      im.count = counts[k];
      if (!counts[k]) return;
      im.instanceMatrix.clearUpdateRanges();
      im.instanceMatrix.addUpdateRange(0, counts[k] * 16);
      im.instanceMatrix.needsUpdate = true;
      const a = this.attrs[k];
      a.clearUpdateRanges();
      a.addUpdateRange(0, counts[k] * 4);
      a.needsUpdate = true;
    });
    this.stats.tufts = counts.reduce((s, c) => s + c, 0);
    this.stats.chunks = order.length;
  }
}

/**
 * The flood wrack line on the bars (0..1): the band where the last high water left its sticks, bark and dead weed,
 * at the top of the gravel ~0.4-0.65 m above today's river. Shared with BankProps.
 */
export function wrackLine(cp: CorePoint): number {
  const above = cp.y - cp.waterY;
  const fringe = sstep(0.0, 0.3, cp.gravel) * (1 - sstep(0.6, 0.95, cp.gravel));
  return fringe * sstep(0.32, 0.45, above) * (1 - sstep(0.62, 0.78, above));
}
