import * as THREE from 'three/webgpu';
import {
  abs, attribute, cameraPosition, cross, dFdx, dFdy, dot, float, max, mix, normalize, normalMap, normalWorldGeometry,
  positionView, positionWorld, sign, smoothstep, texture, uniform, uv, vec2, vec3, vec4, type N,
} from '../gfx/tsl';
import { tnoise, tnoiseS } from '../gfx/noiseTex';
import { setOpaqueAlbedo } from '../gfx/opaque';
import type { Assets } from '../core/Assets';
import { LAYER_FOLIAGE, LAYER_NO_REFLECT, LAYER_SHADOW } from '../core/layers';
import { bakeGeometry, ROCK_MODELS } from '../water/Rocks';
import { fieldNodes, type RiverField } from '../water/RiverField';
import { river } from './River';
import { fbm, smoothstep as sstep } from './noise';
import { LAYOUT } from './Layout';
import { pathDist, type CanopyField, type TreeInst } from './ForestLayout';
import { CAMERA_SPOTS } from '../debug/CameraSpots';
import { wrackLine } from './Grass';
import { instanceCapacity } from '../core/instancing';
import { SPRUCE, spruceAxis, spruceDist, spruceRadius } from './FallenSpruce';

/**
 * Bank dressing (Poly Haven CC0 scans from the prep lane): crowds of rocks from pebbles to car-sized boulders packed
 * together and half buried at the foot of the cut banks and in the forest, toe rocks the waterline wanders around,
 * cobbles bedded into the bank faces, stones and cobbles where the bar's gravel runs out into the grass and a
 * flood-debris line of bleached sticks along the bar's top, chunky driftwood, a spruce fallen from the north bank
 * into the pool, deadfall, stumps and broken branches on the forest floor, ferns in clumps under the trees. All
 * instanced (one mesh per model and LOD), placed deterministically from the corridor grid's bank attributes.
 */

/** per-vertex data of the corridor grid (aWater, aBank) by position */
export interface CoreSampler {
  /** fills `o` and returns true inside the grid */
  at(x: number, z: number, o: CorePoint): boolean;
  groundY(x: number, z: number): number;
}
export interface CorePoint { waterY: number; edge: number; apron: number; rock: number; belowTop: number; gravel: number; y: number }

type Kind = 'boulder' | 'stone' | 'log' | 'fern';
interface PropModel { file: string; size: [number, number, number]; lum: number; ao: boolean; kind: Kind }

const lum3 = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
const fromRiffle = (id: string, kind: Kind = 'boulder'): PropModel => ({
  file: `rocks/${id}.glb`, size: ROCK_MODELS[id].size, lum: ROCK_MODELS[id].lum, ao: ROCK_MODELS[id].ao, kind,
});
const MODELS: Record<string, PropModel> = {
  rock_moss_01_r01: { file: 'rocks/rock_moss_01_r01.glb', size: [2.238, 1.473, 3.364], lum: lum3(0.123, 0.0755, 0.031), ao: false, kind: 'boulder' },
  rock_moss_01_r02: { file: 'rocks/rock_moss_01_r02.glb', size: [2.655, 1.262, 3.264], lum: lum3(0.103, 0.0668, 0.0316), ao: false, kind: 'boulder' },
  rock_moss_01_r03: { file: 'rocks/rock_moss_01_r03.glb', size: [2.109, 1.1, 2.08], lum: lum3(0.0874, 0.0593, 0.031), ao: false, kind: 'boulder' },
  rock_moss_01_r04: { file: 'rocks/rock_moss_01_r04.glb', size: [2.128, 1.768, 1.964], lum: lum3(0.1248, 0.0804, 0.034), ao: false, kind: 'boulder' },
  rock_moss_01_r05: { file: 'rocks/rock_moss_01_r05.glb', size: [1.817, 1.205, 2.999], lum: lum3(0.1204, 0.0795, 0.0351), ao: false, kind: 'boulder' },
  rock_moss_01_r06: { file: 'rocks/rock_moss_01_r06.glb', size: [2.126, 1.262, 2.757], lum: lum3(0.146, 0.0961, 0.0418), ao: false, kind: 'boulder' },
  rock_moss_02_r11: { file: 'rocks/rock_moss_02_r11.glb', size: [1.954, 1.31, 1.529], lum: lum3(0.1044, 0.0903, 0.0582), ao: false, kind: 'boulder' },
  rock_moss_02_r07: fromRiffle('rock_moss_02_r07'),
  rock_moss_02_r08: fromRiffle('rock_moss_02_r08'),
  rock_moss_02_r09: fromRiffle('rock_moss_02_r09'),
  rock_moss_02_r10: fromRiffle('rock_moss_02_r10'),
  rock_moss_02_r12: fromRiffle('rock_moss_02_r12'),
  rock_moss_02_r13: fromRiffle('rock_moss_02_r13'),
  boulder_01: fromRiffle('boulder_01'),
  rock_07: { file: 'rocks/rock_07.glb', size: [0.168, 0.143, 0.319], lum: lum3(0.187, 0.1211, 0.0731), ao: true, kind: 'stone' },
  rock_09: { file: 'rocks/rock_09.glb', size: [0.074, 0.033, 0.144], lum: lum3(0.0851, 0.0506, 0.0242), ao: true, kind: 'stone' },
  stone_01: { file: 'rocks/stone_01.glb', size: [0.146, 0.071, 0.088], lum: lum3(0.2245, 0.1186, 0.0717), ao: true, kind: 'stone' },
  dead_tree_trunk: { file: 'logs/dead_tree_trunk.glb', size: [3.05, 0.287, 0.277], lum: lum3(0.0446, 0.0324, 0.0159), ao: true, kind: 'log' },
  dead_tree_trunk_02: { file: 'logs/dead_tree_trunk_02.glb', size: [4.048, 1.054, 1.053], lum: lum3(0.1433, 0.091, 0.0467), ao: true, kind: 'log' },
  tree_stump_01: { file: 'logs/tree_stump_01.glb', size: [1.426, 0.572, 1.587], lum: lum3(0.1192, 0.0823, 0.0435), ao: true, kind: 'log' },
  fern_02_a: { file: 'plants/fern_02_a.glb', size: [0.552, 0.287, 0.613], lum: 0.1, ao: false, kind: 'fern' },
  fern_02_b: { file: 'plants/fern_02_b.glb', size: [0.99, 0.428, 0.894], lum: 0.1, ao: false, kind: 'fern' },
  fern_02_c: { file: 'plants/fern_02_c.glb', size: [0.874, 0.349, 0.765], lum: 0.1, ao: false, kind: 'fern' },
  fern_02_d: { file: 'plants/fern_02_d.glb', size: [0.573, 0.213, 0.593], lum: 0.1, ao: false, kind: 'fern' },
};
/** blocky boulders for the big rock of a crowd */
const BIG = ['rock_moss_01_r01', 'rock_moss_01_r02', 'rock_moss_01_r04', 'rock_moss_01_r05', 'rock_moss_01_r06', 'rock_moss_02_r13', 'rock_moss_02_r10', 'boulder_01'];
const MED = ['rock_moss_01_r03', 'rock_moss_02_r11', 'rock_moss_02_r07', 'rock_moss_02_r12', 'rock_moss_02_r08', 'rock_moss_02_r09', 'rock_moss_01_r04', 'boulder_01'];
const STONES = ['rock_07', 'rock_07', 'stone_01', 'rock_09', 'rock_07'];
const FERNS = ['fern_02_a', 'fern_02_b', 'fern_02_c', 'fern_02_d'];

export interface PropInst {
  id: string;
  /** 0 = full scan, 1 = its LOD1 (small rocks, deadfall) */
  lod: 0 | 1;
  x: number; y: number; z: number;
  yaw: number;
  /** tilt: about a horizontal axis perpendicular to tiltDir (rad) */
  tiltDir: number; tilt: number;
  /** scale per axis */
  s: [number, number, number];
  /** x moss on top faces 0..1, y brightness jitter, z bleach (driftwood) 0..1, w seed */
  p: [number, number, number, number];
  /** x open sky above (1 - crown cover) */
  q: number;
  /**
   * forest polish: 'floor' = main view and shadows only (forest-floor deadfall, sod, wrack: the pool mirror never
   * shows them), 'small' = main view only (stones and small rocks: under a shadow texel, specks in the mirror)
   */
  role?: 'floor' | 'small';
}

function hash(a: number, b: number, c: number) {
  let h = (Math.imul(a | 0, 374761393) + Math.imul(b | 0, 668265263) + Math.imul(c | 0, 1274126177)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/**
 * Hand-placed wood: id, x, z, yaw deg, length scale, thickness scale, sink m, moss, bleach, pitch deg (tip down).
 * Driftwood is the chunky trunk, bleached and half sunk in the gravel at the top of the bars.
 */
const LOGS: Array<[string, number, number, number, number, number, number, number, number, number]> = [
  ['dead_tree_trunk_02', -9.5, 8.4, 12, 1.15, 0.62, 0.17, 0.0, 0.92, 0],
  ['dead_tree_trunk_02', 7.2, 7.9, -24, 0.78, 0.55, 0.13, 0.0, 0.9, 2],
  ['dead_tree_trunk_02', 22.5, 7.6, -28, 1.0, 0.6, 0.15, 0.0, 0.88, 0],
  ['dead_tree_trunk_02', -27.0, 9.2, 74, 0.9, 0.7, 0.2, 0.1, 0.6, 0],
  ['dead_tree_trunk_02', -14.0, -8.8, 160, 1.4, 0.75, 0.15, 0.4, 0.15, 0],
];
export function placeBankProps(core: CoreSampler, canopy: CanopyField, trees: TreeInst[]): PropInst[] {
  const out: PropInst[] = [];
  const cp: CorePoint = { waterY: 0, edge: 0, apron: 0, rock: 0, belowTop: 3, gravel: 0, y: 0 };
  const cv: [number, number] = [0, 0];
  const eyes = CAMERA_SPOTS.map((s) => [s.eye[0], s.eye[2]] as const);
  const blocked = (x: number, z: number, r: number) => {
    for (const [ex, ez] of eyes) if (Math.hypot(x - ex, z - ez) < 2.2 + r) return true;
    if (Math.abs(x - LAYOUT.dock.x) < 1.4 + r && z > LAYOUT.dock.zTip - 0.5 && z < LAYOUT.dock.zBase + 1) return true;
    if (Math.hypot(x - LAYOUT.campfire.x, z - LAYOUT.campfire.z) < 3.8 + r) return true;
    if (Math.hypot(x - LAYOUT.cabin.x, z - LAYOUT.cabin.z) < 7.5 + r) return true;
    // the C2 splash and its clear shallows
    if (Math.hypot(x - 3.7, z - 3.35) < 1.6 + r) return true;
    // the fallen spruce and its root plate (FallenSpruce.ts)
    if (spruceDist(x, z) < 0.55 + r || Math.hypot(x - SPRUCE.base[0], z - SPRUCE.base[2]) < 1.7 + r) return true;
    return false;
  };
  const slopeAt = (x: number, z: number) => {
    const e = 0.4;
    const gx = (core.groundY(x + e, z) - core.groundY(x - e, z)) / (2 * e), gz = (core.groundY(x, z + e) - core.groundY(x, z - e)) / (2 * e);
    return { ny: 1 / Math.hypot(gx, 1, gz), gx, gz };
  };
  const openAt = (x: number, z: number) => 1 - canopy.sample(x, z, cv)[0];

  /**
   * A rock of `height` m standing at (x, z), `sink` of its height in the ground, stretched per axis; small ones are
   * the LOD1 scan. Returns false when it may not stand there.
   */
  const rock = (id: string, x: number, z: number, height: number, sink: number, moss: number, k: number, salt: number): boolean => {
    const M = MODELS[id];
    const sc = height / M.size[1];
    const s: [number, number, number] = [sc * (0.78 + 0.5 * hash(k, salt, 1)), sc * (0.8 + 0.35 * hash(k, salt, 2)), sc * (0.78 + 0.5 * hash(k, salt, 3))];
    const r = 0.5 * Math.max(M.size[0] * s[0], M.size[2] * s[2]);
    if (blocked(x, z, r * 0.8)) return false;
    if (!core.at(x, z, cp)) return false;
    const { ny, gx, gz } = slopeAt(x, z);
    const small = height < 0.45;
    out.push({
      // (forest polish: the scan's LOD1 up to 1.25 m; the crowds roughly doubled the rock count)
      id, lod: M.kind === 'boulder' && height < 1.25 ? 1 : 0,
      x, y: core.groundY(x, z) - sink * M.size[1] * s[1], z, yaw: hash(k, salt, 4) * Math.PI * 2,
      tiltDir: small ? hash(k, salt, 5) * Math.PI * 2 : Math.atan2(gz, gx),
      tilt: small ? hash(k, salt, 6) * 0.45 : Math.acos(Math.min(1, ny)) * 0.6 + hash(k, salt, 6) * 0.12,
      s, p: [moss, 0.82 + 0.3 * hash(k, salt, 7), 0, hash(k, salt, 8)], q: openAt(x, z), role: height < 0.3 ? 'small' : undefined,
    });
    return true;
  };

  // ---- crowds of rocks: one big block, a few medium ones packed against it, small stones in the gaps; most at the
  // foot of the shaded north bank's cut faces, some on the south bank's toe and on the north forest floor
  const centres: Array<[number, number]> = [];
  let k = 0;
  for (let x = -80; x <= 94; x += 1.1) {
    const zc = river.zAt(x);
    for (let z = zc - 40; z <= zc + 22; z += 1.1, k++) {
      const px = x + (hash(k, 1, 3) - 0.5) * 1.0, pz = z + (hash(k, 2, 3) - 0.5) * 1.0;
      if (!core.at(px, pz, cp)) continue;
      if (cp.edge < -0.5 || cp.edge > 40) continue;
      const north = pz < river.zAt(px);
      const { ny } = slopeAt(px, pz);
      const steep = 1 - ny;
      const face = sstep(0.12, 0.3, steep) * sstep(0.1, 0.35, cp.belowTop) * (1 - sstep(2.6, 3, cp.belowTop));
      const toe = Math.max(sstep(0.15, 0.5, cp.apron), sstep(0.06, 0.2, steep) * (1 - sstep(0.4, 0.9, cp.gravel)) * sstep(1.6, 0.2, Math.abs(cp.edge - 0.8)));
      const floor = north ? sstep(9, 14, cp.edge) * 0.12 : 0;
      const crowdN = sstep(-0.25, 0.3, fbm(px * 0.07 + 4.1, pz * 0.07, 2));
      const p = (Math.max(face, toe) * (north ? 0.1 : 0.03) + floor * 0.05) * (0.25 + 1.2 * crowdN);
      if (hash(k, 3, 3) >= p) continue;
      if (centres.some(([cx, cz]) => Math.hypot(cx - px, cz - pz) < (north ? 2.6 : 4.2))) continue;
      centres.push([px, pz]);
      const mossy = north ? 0.75 + 0.25 * hash(k, 4, 3) : 0.45 + 0.35 * hash(k, 4, 3);
      // car-sized blocks only at the foot of the north bank; elsewhere the biggest is knee to waist high
      const bigH = north && toe > face ? 0.9 + 1.3 * Math.pow(hash(k, 5, 3), 1.4) : 0.45 + 0.6 * hash(k, 5, 3);
      if (!rock(BIG[Math.floor(hash(k, 6, 3) * BIG.length)], px, pz, bigH, 0.48 + 0.14 * hash(k, 7, 3), mossy, k, 11)) continue;
      const R0 = bigH * 0.9;
      const nMed = 1 + Math.floor(hash(k, 8, 3) * 4), nSmall = 3 + Math.floor(hash(k, 9, 3) * 9);
      for (let m = 0; m < nMed; m++) {
        const a = hash(k, 20 + m, 3) * Math.PI * 2, h = bigH * (0.25 + 0.4 * hash(k, 30 + m, 3));
        const d = (R0 + h) * (0.6 + 0.4 * hash(k, 40 + m, 3));
        rock(MED[Math.floor(hash(k, 50 + m, 3) * MED.length)], px + Math.cos(a) * d, pz + Math.sin(a) * d, h, 0.4 + 0.2 * hash(k, 60 + m, 3), mossy * 0.85, k, 100 + m);
      }
      for (let m = 0; m < nSmall; m++) {
        const a = hash(k, 70 + m, 3) * Math.PI * 2, d = R0 * (0.7 + 1.6 * Math.sqrt(hash(k, 80 + m, 3)));
        const h = 0.05 + 0.3 * Math.pow(hash(k, 90 + m, 3), 1.6);
        const id = h > 0.22 ? MED[Math.floor(hash(k, 110 + m, 3) * MED.length)] : STONES[Math.floor(hash(k, 110 + m, 3) * STONES.length)];
        rock(id, px + Math.cos(a) * d, pz + Math.sin(a) * d, h, 0.25 + 0.2 * hash(k, 120 + m, 3), mossy * 0.4, k, 200 + m);
      }
    }
  }

  // ---- toe rocks: the north bank's waterline wanders in and out around rocks standing half in the water
  k = 0;
  for (let x = -70; x <= 90; x += 0.9, k++) {
    const zc = river.zAt(x);
    let zw = NaN;
    for (let z = zc; z > zc - 14; z -= 0.2) {
      if (core.at(x, z, cp) && cp.y > cp.waterY) {
        zw = z;
        break;
      }
    }
    if (Number.isNaN(zw)) continue;
    core.at(x, zw - 0.6, cp);
    if (cp.belowTop < 0.35) continue;
    // cobbles half in the water every metre or so, so nowhere does the waterline run on unbroken (C4, round 1)
    for (let c = 0; c < 2; c++) {
      if (hash(k, 40 + c, 13) >= 0.55) continue;
      const cs = 0.09 + 0.2 * Math.pow(hash(k, 50 + c, 13), 1.4);
      rock(STONES[Math.floor(hash(k, 60 + c, 13) * STONES.length)], x + (hash(k, 70 + c, 13) - 0.5) * 0.9,
        zw + (hash(k, 80 + c, 13) - 0.45) * 0.7, cs, 0.35, 0.2, k, 320 + c);
    }
    const pt = 0.18 + 0.36 * sstep(-0.2, 0.35, fbm(x * 0.21 + 1.7, 3.3, 2));
    if (hash(k, 1, 13) >= pt) continue;
    const pz = zw + (hash(k, 2, 13) - 0.4) * 0.9;
    const h = 0.28 + 0.75 * Math.pow(hash(k, 3, 13), 1.3);
    rock(MED[Math.floor(hash(k, 4, 13) * MED.length)], x + (hash(k, 5, 13) - 0.5) * 0.6, pz, h, 0.4, 0.55, k, 300);
  }

  // ---- forest polish: armour at the foot of the north bank, along the pool and the riffle: packed crowds of mixed
  // sizes on the waterline, half buried, the big blocks sunk into the bank and the water (critic r2: "boulders are
  // scarce at the bank foot", C1, C3, C6)
  k = 0;
  for (let x = -75; x <= 92; x += 1.6, k++) {
    const zc = river.zAt(x);
    let zw = NaN;
    for (let z = zc; z > zc - 14; z -= 0.2) {
      if (core.at(x, z, cp) && cp.y > cp.waterY) {
        zw = z;
        break;
      }
    }
    if (Number.isNaN(zw)) continue;
    const pa = 0.62 * sstep(-0.3, 0.2, fbm(x * 0.09 + 3.7, 1.9, 2) + 0.12);
    if (hash(k, 1, 29) >= pa) continue;
    // across the bank: negative into the water, positive up the bank (north, -z)
    const at = (along: number, across: number) => [x + along, zw - across] as const;
    const bigH = 0.5 + 1.15 * Math.pow(hash(k, 2, 29), 1.5);
    {
      const [bx, bz] = at((hash(k, 3, 29) - 0.5) * 0.8, (hash(k, 4, 29) - 0.4) * 1.1);
      rock(BIG[Math.floor(hash(k, 5, 29) * BIG.length)], bx, bz, bigH, 0.5 + 0.12 * hash(k, 6, 29), 0.72, k, 400);
    }
    const nMed = 2 + Math.floor(hash(k, 7, 29) * 4), nSmall = 4 + Math.floor(hash(k, 8, 29) * 8);
    for (let m = 0; m < nMed; m++) {
      const [mx, mz] = at((hash(k, 10 + m, 29) - 0.5) * 3.0, (hash(k, 20 + m, 29) - 0.35) * 1.8);
      const h = 0.2 + 0.45 * Math.pow(hash(k, 30 + m, 29), 1.2);
      rock(MED[Math.floor(hash(k, 40 + m, 29) * MED.length)], mx, mz, h, 0.42 + 0.18 * hash(k, 50 + m, 29), 0.6, k, 410 + m);
    }
    for (let m = 0; m < nSmall; m++) {
      const [sxp, szp] = at((hash(k, 60 + m, 29) - 0.5) * 3.4, (hash(k, 70 + m, 29) - 0.4) * 2.0);
      const h = 0.06 + 0.2 * Math.pow(hash(k, 80 + m, 29), 1.5);
      const id = h > 0.18 ? MED[Math.floor(hash(k, 90 + m, 29) * MED.length)] : STONES[Math.floor(hash(k, 90 + m, 29) * STONES.length)];
      rock(id, sxp, szp, h, 0.35 + 0.2 * hash(k, 100 + m, 29), 0.3, k, 430 + m);
    }
  }

  // ---- slumped sod at the foot of the earth cut banks: chunks of turf and soil that broke off the lip, tipped over
  // as they fell (critic r2: "slumped chunks at its foot")
  k = 0;
  for (let x = -75; x <= 90; x += 0.9) {
    const zc = river.zAt(x);
    for (let z = zc - 20; z <= zc + 16; z += 0.9, k++) {
      const px = x + (hash(k, 1, 37) - 0.5) * 0.8, pz = z + (hash(k, 2, 37) - 0.5) * 0.8;
      if (!core.at(px, pz, cp)) continue;
      if (cp.rock > 0.4 || cp.y < cp.waterY - 0.05 || cp.belowTop < 0.4) continue;
      const foot = sstep(0.15, 0.5, cp.apron) * (1 - sstep(0.85, 1.0, cp.apron));
      const ps = foot * 0.2 * sstep(-0.2, 0.3, fbm(px * 0.17 + 9.1, pz * 0.17 - 4.4, 2));
      if (hash(k, 3, 37) >= ps) continue;
      const h = 0.22 + 0.4 * hash(k, 4, 37);
      const id = MED[Math.floor(hash(k, 5, 37) * MED.length)];
      const M = MODELS[id];
      const sc = h / M.size[1];
      if (blocked(px, pz, 0.5 * sc * Math.max(M.size[0], M.size[2]))) continue;
      out.push({
        id, lod: 1, x: px, y: core.groundY(px, pz) - h * 0.3, z: pz, yaw: hash(k, 6, 37) * Math.PI * 2,
        tiltDir: hash(k, 7, 37) * Math.PI * 2, tilt: 0.2 + 0.5 * hash(k, 8, 37),
        s: [sc * (0.9 + 0.4 * hash(k, 9, 37)), sc, sc * (0.9 + 0.4 * hash(k, 10, 37))],
        // bleach < 0 marks soil: brown earth sides, a cap of turf on top
        p: [0.95, 0.85 + 0.3 * hash(k, 11, 37), -1, hash(k, 12, 37)], q: openAt(px, pz), role: 'floor',
      });
    }
  }

  // ---- the sod lip of the earth cut banks (scene fix, critic polish r1: "a smooth grey wall with a flat face and a
  // straight top"): flat slabs of turf and the soil under it overhanging the face along its top edge, some tipped
  // out and sagging, some slumped part way down the face; roots hang out under them (Grass.ts)
  k = 0;
  {
    const cq2: CorePoint = { waterY: 0, edge: 0, apron: 0, rock: 0, belowTop: 3, gravel: 0, y: 0 };
    for (let x = -75; x <= 90; x += 0.55) {
      const zc = river.zAt(x);
      for (let z = zc - 20; z <= zc + 16; z += 0.55, k++) {
        const px = x + (hash(k, 1, 47) - 0.5) * 0.5, pz = z + (hash(k, 2, 47) - 0.5) * 0.5;
        if (!core.at(px, pz, cp)) continue;
        if (cp.belowTop > 0.12 || cp.rock > 0.75 || cp.y < cp.waterY + 0.3 || cp.edge > 9) continue;
        const { gx, gz } = slopeAt(px, pz);
        const gl = Math.hypot(gx, gz);
        if (gl < 0.15) continue;
        const ox = -gx / gl, oz = -gz / gl;
        // a real face below the lip: 0.45 m out (downhill) the ground is well down the face, and steep
        if (!core.at(px + ox * 0.45, pz + oz * 0.45, cq2) || cq2.belowTop < 0.3 || cq2.rock > 0.8) continue;
        const { ny: nyF } = slopeAt(px + ox * 0.45, pz + oz * 0.45);
        if (nyF > 0.82) continue;
        const pl = 0.55 * sstep(-0.25, 0.25, fbm(px * 0.23 + 2.7, pz * 0.23 - 8.8, 2));
        if (hash(k, 3, 47) >= pl) continue;
        const h = 0.2 + 0.18 * hash(k, 4, 47);
        const id = MED[Math.floor(hash(k, 5, 47) * MED.length)];
        const M = MODELS[id];
        const sc = h / M.size[1];
        const slump = hash(k, 6, 47) < 0.28;
        const outD = slump ? 0.35 + 0.4 * hash(k, 7, 47) : 0.08 + 0.12 * hash(k, 7, 47);
        const sx = px + ox * outD, sz = pz + oz * outD;
        if (blocked(sx, sz, 0.5)) continue;
        const flatY = 0.38 + 0.12 * hash(k, 8, 47);
        // the slab's top flush with the turf at the lip (slumped ones lower, on the face)
        const top = slump ? core.groundY(sx, sz) + 0.05 : cp.y + 0.02;
        out.push({
          id, lod: 1, x: sx, y: top - M.size[1] * sc * flatY * 0.85, z: sz, yaw: Math.atan2(oz, ox) + (hash(k, 9, 47) - 0.5) * 0.6,
          tiltDir: Math.atan2(gz, gx), tilt: slump ? 0.45 + 0.4 * hash(k, 10, 47) : 0.12 + 0.22 * hash(k, 10, 47),
          s: [sc * (1.3 + 0.5 * hash(k, 11, 47)), sc * flatY, sc * (1.1 + 0.4 * hash(k, 12, 47))],
          // bleach < 0: soil sides, a cap of turf on top
          p: [1.0, 0.85 + 0.3 * hash(k, 13, 47), -1, hash(k, 14, 47)], q: openAt(sx, sz), role: 'floor',
        });
      }
    }
  }

  // ---- cobbles bedded into the cut-bank faces, sunk most of their size into the bank
  k = 0;
  for (let x = -75; x <= 90; x += 0.7) {
    const zc = river.zAt(x);
    for (let z = zc - 22; z <= zc + 18; z += 0.7, k++) {
      const px = x + (hash(k, 1, 17) - 0.5) * 0.6, pz = z + (hash(k, 2, 17) - 0.5) * 0.6;
      if (!core.at(px, pz, cp)) continue;
      if (cp.y < cp.waterY - 0.05 || cp.belowTop < 0.15 || cp.belowTop > 2.6 || cp.rock > 0.8) continue;
      const { ny, gx, gz } = slopeAt(px, pz);
      if (ny > 0.8) continue;
      const pc = 0.42 * sstep(-0.1, 0.35, fbm(px * 0.3 - 2.2, pz * 0.3 + 5.5, 2));
      if (hash(k, 3, 17) >= pc) continue;
      const size = 0.08 + 0.38 * Math.pow(hash(k, 4, 17), 1.4);
      const id = size > 0.22 ? MED[Math.floor(hash(k, 5, 17) * MED.length)] : STONES[Math.floor(hash(k, 5, 17) * STONES.length)];
      const M = MODELS[id];
      const sc = size / Math.max(M.size[0], M.size[1], M.size[2]);
      if (blocked(px, pz, size)) continue;
      // pushed into the face along its normal
      const nl = Math.hypot(gx, 1, gz), nxw = -gx / nl, nzw = -gz / nl;
      // (scene fix: sunk 30 % of their size, so they stand out of the face; was 45 %)
      const into = size * 0.3;
      out.push({
        id, lod: 1, x: px - nxw * into, y: core.groundY(px, pz) - (1 / nl) * into * 0.5, z: pz - nzw * into,
        yaw: hash(k, 6, 17) * Math.PI * 2, tiltDir: hash(k, 7, 17) * Math.PI * 2, tilt: 0.3 + hash(k, 8, 17) * 0.9,
        s: [sc, sc, sc], p: [0.15 * hash(k, 9, 17), 0.8 + 0.3 * hash(k, 10, 17), 0, hash(k, 11, 17)], q: openAt(px, pz),
      });
    }
  }

  // ---- stones and cobbles where the bar's gravel runs out into the grass (scattered a few metres into it and half
  // buried in the turf), a few cobbles and boulders out on the bar itself, and the flood wrack along the bar's top:
  // grey sticks and broken branches in clumps where the last high water left them (with the dead weed in Grass.ts)
  k = 0;
  const towardWater = (x: number, z: number, d: number) => z + (z < river.zAt(x) ? d : -d);
  const cq: CorePoint = { waterY: 0, edge: 0, apron: 0, rock: 0, belowTop: 3, gravel: 0, y: 0 };
  for (let x = -70; x <= 85; x += 0.55) {
    const zc = river.zAt(x);
    for (let z = zc - 16; z <= zc + 18; z += 0.55, k++) {
      const px = x + (hash(k, 1, 5) - 0.5) * 0.5, pz = z + (hash(k, 2, 5) - 0.5) * 0.5;
      if (!core.at(px, pz, cp)) continue;
      if (cp.y < cp.waterY + 0.015 || cp.edge < 0) continue;
      const fringe = sstep(0.0, 0.25, cp.gravel) * (1 - sstep(0.55, 0.95, cp.gravel));
      const grass = sstep(0.0, 0.08, cp.gravel) * (1 - sstep(0.08, 0.25, cp.gravel));
      // turf within ~3 m of the bar's top: cobbles the floods left there, now half buried in the grass
      let turf = 0;
      if (cp.gravel < 0.05) {
        for (const d of [1.2, 2.6]) {
          if (core.at(px, towardWater(px, pz, d), cq) && cq.gravel > 0.3) {
            turf = d < 2 ? 1 : 0.6;
            break;
          }
        }
      }
      const onBar = sstep(0.6, 0.9, cp.gravel) * 0.018;
      const n = sstep(-0.2, 0.3, fbm(px * 0.21, pz * 0.21 + 3.3, 2));
      const p = (fringe * 0.38 + grass * 0.16 + turf * 0.1) * (0.35 + 0.9 * n) + onBar + cp.apron * 0.08;
      const hh = hash(k, 3, 5);
      if (hh < p) {
        if (blocked(px, pz, 0.1) && hash(k, 4, 5) < 0.8) continue;
        const big = hash(k, 5, 5) < (onBar > 0 ? 0.3 : 0.1);
        const size = big ? 0.35 + 0.35 * hash(k, 6, 5) : 0.06 * Math.pow(6, hash(k, 6, 5)); // 6-36 cm, a few 35-70 cm
        const id = size > 0.3 ? MED[Math.floor(hash(k, 7, 5) * MED.length)] : STONES[Math.floor(hash(k, 7, 5) * STONES.length)];
        const M = MODELS[id];
        const sc = size / Math.max(M.size[0], M.size[2]);
        // in the turf they sit deeper, grown in
        const sink = turf > 0 || grass > 0.5 ? 0.45 + 0.2 * hash(k, 8, 5) : 0.25 + 0.2 * hash(k, 8, 5);
        out.push({
          id, lod: 1, x: px, y: core.groundY(px, pz) - M.size[1] * sc * sink, z: pz,
          yaw: hash(k, 9, 5) * Math.PI * 2, tiltDir: hash(k, 10, 5) * Math.PI * 2, tilt: hash(k, 11, 5) * 0.3,
          s: [sc, sc * (0.8 + 0.3 * hash(k, 12, 5)), sc], p: [(0.12 + 0.3 * turf) * hash(k, 13, 5), 0.85 + 0.3 * hash(k, 14, 5), 0, hash(k, 15, 5)], q: 1,
          role: size < 0.3 ? 'small' : undefined,
        });
        continue;
      }
      // flood wrack: grey sticks and broken branches in clumps along the line of the last high water
      const line = wrackLine(cp);
      if (hash(k, 16, 5) < line * 0.16 * (0.2 + 1.2 * n) && !blocked(px, pz, 0.6)) {
        const along = -Math.atan2((river.zAt(px + 1) - river.zAt(px - 1)) / 2, 1);
        const nS = 1 + Math.floor(hash(k, 22, 5) * 3);
        for (let s2 = 0; s2 < nS; s2++) {
          const hs2 = (q: number) => hash(k, 30 + s2 * 7 + q, 5);
          const len = 0.12 + 0.4 * hs2(1), th = 0.12 + 0.3 * hs2(2);
          const ox = px + (hs2(3) - 0.5) * 0.7, oz = pz + (hs2(4) - 0.5) * 0.5;
          out.push({
            id: 'dead_tree_trunk', lod: 1, x: ox, y: core.groundY(ox, oz) - 0.015, z: oz,
            yaw: along + (hs2(5) - 0.5) * 2.2, tiltDir: 0, tilt: 0,
            s: [len, th, th], p: [0, 0.82 + 0.25 * hs2(6), 0.7 + 0.3 * hs2(7), hs2(8)], q: 1, role: 'floor',
          });
        }
      }
    }
  }
  // ---- hand-placed driftwood and the trunk fallen into the pool
  LOGS.forEach(([id, x, z, yaw, len, th, sink, moss, bleach, pitch], i) => {
    out.push({
      id, lod: 0, x, y: core.groundY(x, z) - sink, z, yaw: (yaw * Math.PI) / 180, tiltDir: (-yaw * Math.PI) / 180, tilt: (-pitch * Math.PI) / 180,
      s: [len, th, th], p: [moss, 0.9 + 0.2 * hash(i, 1, 7), bleach, hash(i, 2, 7)], q: 1,
    });
  });
  // flood sticks and broken branches the current has pinned against the fallen spruce (FallenSpruce.ts) on its
  // upstream side where it lies awash, half sunk; two more lodged across its back
  {
    const along = -Math.atan2(SPRUCE.dir[1], SPRUCE.dir[0]);
    const upX = -SPRUCE.dir[1], upZ = SPRUCE.dir[0];
    for (let i = 0; i < 9; i++) {
      const across = i >= 7;
      const sA = across ? 4.4 + (i - 7) * 2.1 : 3.4 + i * 0.62 + (hash(i, 3, 29) - 0.5) * 0.3;
      const [ax, ay, az] = spruceAxis(sA);
      const rT = spruceRadius(sA);
      const side = across ? 0 : rT + 0.08 + 0.22 * hash(i, 4, 29);
      const len = across ? 0.45 + 0.25 * hash(i, 5, 29) : 0.3 + 0.45 * hash(i, 5, 29);
      const th = 0.32 + 0.38 * hash(i, 6, 29);
      out.push({
        id: 'dead_tree_trunk', lod: 1, x: ax + upX * side, z: az + upZ * side,
        y: across ? ay + rT - 0.06 : -0.287 * th * (0.45 + 0.2 * hash(i, 7, 29)),
        // the current holds them along its side (turned across the view, one wet stick glinted white in C4)
        yaw: along + (across ? (i === 7 ? 1.25 : -1.4) : (hash(i, 8, 29) - 0.5) * 0.6),
        tiltDir: hash(i, 9, 29) * Math.PI * 2, tilt: hash(i, 10, 29) * 0.12,
        // waterlogged and dark; the two across its back dry, barely bleached (paler, they read as birch sticks in C4)
        s: [len, th, th], p: [0, 0.68 + 0.14 * hash(i, 11, 29), across ? 0.12 : 0.08 * hash(i, 12, 29), hash(i, 13, 29)], q: 1,
      });
    }
  }

  // ---- deadfall on the forest floor: rotting logs lying at all angles, some propped up on another log or a stump
  // at one end, crossed pairs, root plates torn up at the foot of the bigger trunks, stumps and broken branches
  // among the trunks (forest polish: denser, and no longer all lying flat)
  k = 0;
  const logDir = (yaw: number) => [Math.cos(yaw), -Math.sin(yaw)] as const;
  /** a log of model `id`, centre (x, z), length scale `len`, thickness `th`; `pitch` (rad) raises its +x end */
  const log = (id: string, x: number, z: number, yaw: number, len: number, th: number, pitch: number, moss: number, seed: number, open: number) => {
    const M = MODELS[id];
    const L = M.size[0] * len;
    const [dx, dz] = logDir(yaw);
    // the low (-x) end rests on the ground, the rest rises at `pitch`
    const lx = x - dx * L * 0.5 * Math.cos(pitch), lz = z - dz * L * 0.5 * Math.cos(pitch);
    const y = core.groundY(lx, lz) + L * 0.5 * Math.sin(pitch) + M.size[1] * th * (pitch > 0.05 ? 0.15 : -0.3);
    out.push({
      id, lod: 1, x, y, z, yaw, tiltDir: -yaw, tilt: pitch,
      s: [len, th, th], p: [moss, 0.78 + 0.3 * hash(seed, 1, 41), 0, hash(seed, 2, 41)], q: open, role: 'floor',
    });
    return { L, lx, lz, dx, dz };
  };
  for (let x = -88; x <= 98; x += 2.2) {
    const zc = river.zAt(x);
    for (let z = zc - 48; z <= zc + 44; z += 2.2, k++) {
      const px = x + (hash(k, 1, 23) - 0.5) * 2.0, pz = z + (hash(k, 2, 23) - 0.5) * 2.0;
      if (!core.at(px, pz, cp)) continue;
      if (cp.edge < 4 || cp.gravel > 0.1) continue;
      const north = pz < river.zAt(px);
      canopy.sample(px, pz, cv);
      const inForest = north ? sstep(4, 8, cp.edge) : sstep(0.2, 0.5, cv[0]);
      if (cv[1] > 0.6) continue;
      const { ny } = slopeAt(px, pz);
      if (ny < 0.72) continue;
      const pd = inForest * (north ? 0.26 : 0.14) * (0.3 + sstep(-0.2, 0.3, fbm(px * 0.09 + 6.6, pz * 0.09 - 1.1, 2)));
      const r = hash(k, 3, 23);
      if (r >= pd || pathDist(px, pz) < 1.6 || blocked(px, pz, 1.5)) continue;
      const kind = hash(k, 4, 23);
      const moss = 0.55 + 0.45 * hash(k, 5, 23);
      const open = 1 - cv[0];
      const yaw = hash(k, 10, 23) * Math.PI * 2;
      if (kind < 0.62) {
        const chunky = hash(k, 6, 23) < 0.45;
        const id = chunky ? 'dead_tree_trunk_02' : 'dead_tree_trunk';
        const len = chunky ? 0.7 + 0.9 * hash(k, 7, 23) : 1.0 + 1.4 * hash(k, 7, 23);
        const th = chunky ? 0.45 + 0.4 * hash(k, 8, 23) : 1.1 + 1.0 * hash(k, 8, 23);
        // a third lean on something at one end, the rest lie flat, some sunk in the litter
        const pitch = kind < 0.2 ? 0.14 + 0.38 * hash(k, 9, 23) : hash(k, 9, 23) * 0.06;
        const lg = log(id, px, pz, yaw, len, th, pitch, moss, k, open);
        if (pitch > 0.1 && hash(k, 11, 23) < 0.6) {
          // what holds it up: a shorter log across it near its high end
          const t = 0.35 + 0.15 * hash(k, 12, 23);
          const hx = px + lg.dx * lg.L * t, hz = pz + lg.dz * lg.L * t;
          log('dead_tree_trunk_02', hx, hz, yaw + Math.PI / 2 + (hash(k, 13, 23) - 0.5) * 0.8, 0.35 + 0.3 * hash(k, 14, 23),
            0.35 + 0.25 * hash(k, 15, 23), 0, moss, k + 7001, open);
        } else if (pitch < 0.1 && hash(k, 16, 23) < 0.45) {
          // a second log lying across the first
          const t = (hash(k, 17, 23) - 0.5) * 0.6;
          log(hash(k, 18, 23) < 0.5 ? 'dead_tree_trunk' : 'dead_tree_trunk_02', px + lg.dx * lg.L * t, pz + lg.dz * lg.L * t,
            yaw + 0.9 + 1.3 * hash(k, 19, 23), chunky ? 0.6 + 0.5 * hash(k, 20, 23) : 0.8 + 0.8 * hash(k, 20, 23),
            chunky ? 0.4 + 0.3 * hash(k, 21, 23) : 0.9 + 0.7 * hash(k, 21, 23), 0.06 + 0.1 * hash(k, 22, 23), moss * 0.9, k + 9001, open);
        }
        if (chunky && pitch < 0.1 && th > 0.6 && hash(k, 24, 23) < 0.45) {
          // the root plate torn up with it: a stump turned on its side at the log's low end
          const S = MODELS.tree_stump_01;
          const sc = (MODELS.dead_tree_trunk_02.size[1] * th * 2.4) / Math.max(S.size[0], S.size[2]);
          out.push({
            id: 'tree_stump_01', lod: 1, x: lg.lx - lg.dx * 0.15, y: core.groundY(lg.lx, lg.lz) + 0.42 * sc * S.size[2] - 0.1, z: lg.lz - lg.dz * 0.15,
            yaw, tiltDir: -yaw, tilt: -Math.PI / 2 + (hash(k, 25, 23) - 0.5) * 0.4,
            s: [sc, sc * (0.7 + 0.4 * hash(k, 26, 23)), sc], p: [moss * 0.7, 0.8 + 0.2 * hash(k, 27, 23), 0, hash(k, 28, 23)], q: open,
            role: 'floor',
          });
        }
      } else if (kind < 0.8) {
        const sc = 0.55 + 0.5 * hash(k, 7, 23);
        out.push({
          id: 'tree_stump_01', lod: 1, x: px, y: core.groundY(px, pz) - 0.12, z: pz, yaw,
          tiltDir: 0, tilt: 0, s: [sc, sc * (0.8 + 0.6 * hash(k, 8, 23)), sc], p: [moss * 0.8, 0.85 + 0.25 * hash(k, 13, 23), 0, hash(k, 14, 23)], q: open,
          role: 'floor',
        });
      } else {
        // broken branches and a short stick or three
        for (let b = 0; b < 3; b++) {
          const ox = px + (hash(k, 30 + b, 23) - 0.5) * 2.2, oz = pz + (hash(k, 40 + b, 23) - 0.5) * 2.2;
          const len = 0.3 + 0.45 * hash(k, 50 + b, 23), th = 0.3 + 0.35 * hash(k, 60 + b, 23);
          out.push({
            id: 'dead_tree_trunk', lod: 1, x: ox, y: core.groundY(ox, oz) - 0.03, z: oz, yaw: hash(k, 70 + b, 23) * Math.PI * 2,
            tiltDir: hash(k, 80 + b, 23) * Math.PI * 2, tilt: 0.05 + 0.1 * hash(k, 85 + b, 23), s: [len, th, th], p: [moss * 0.6, 0.85, 0, hash(k, 90 + b, 23)], q: open,
            role: 'floor',
          });
        }
      }
    }
  }

  // ---- ferns in clumps with gaps, in the shade under the trees and along the north bank's foot
  k = 0;
  for (let x = -72; x <= 90; x += 0.95) {
    const zc = river.zAt(x);
    for (let z = zc - 40; z <= zc + 44; z += 0.95, k++) {
      const px = x + (hash(k, 1, 9) - 0.5) * 0.9, pz = z + (hash(k, 2, 9) - 0.5) * 0.9;
      if (!core.at(px, pz, cp)) continue;
      if (cp.edge < 0.8 || cp.gravel > 0.3) continue;
      const north = pz < river.zAt(px);
      canopy.sample(px, pz, cv);
      const { ny } = slopeAt(px, pz);
      const shade = Math.max(sstep(0.15, 0.6, cv[0]), north ? sstep(1, 3, cp.edge) * (1 - sstep(10, 16, cp.edge)) * 0.6 : 0);
      // clumps ~2-4 m across with bare gaps (round 1's even band along the bank foot read as a planted border)
      const clump = sstep(0.0, 0.36, fbm(px * 0.32 - 2.2, pz * 0.32 + 7.7, 2));
      const p = shade * 0.85 * sstep(0.55, 0.85, ny) * clump * (1 - sstep(0.75, 1, cv[1]));
      if (hash(k, 3, 9) >= p || pathDist(px, pz) < 1.3 || blocked(px, pz, 0.4)) continue;
      const id = FERNS[Math.floor(hash(k, 4, 9) * FERNS.length)];
      // (scene fix, critic polish r1: up to 3.4x, the fronds stood 2-3 m wide and read as banana or taro leaves; a sword
      // or lady fern is a clump about a metre across)
      const sc = 0.6 + 0.8 * Math.pow(hash(k, 8, 9), 1.3) * (0.7 + 0.5 * clump);
      out.push({
        id, lod: 0, x: px, y: core.groundY(px, pz) - 0.03, z: pz, yaw: hash(k, 5, 9) * Math.PI * 2,
        tiltDir: hash(k, 6, 9) * Math.PI * 2, tilt: hash(k, 7, 9) * 0.15, s: [sc, sc * (0.8 + 0.4 * hash(k, 11, 9)), sc],
        p: [0, 0.8 + 0.35 * hash(k, 9, 9), 0, hash(k, 10, 9)], q: 1 - cv[0],
      });
      // a second, smaller fern beside it: ferns grow in groups of crowns
      if (hash(k, 12, 9) < 0.55 * clump) {
        const a = hash(k, 13, 9) * Math.PI * 2, d = 0.35 + 0.35 * hash(k, 14, 9);
        const fx = px + Math.cos(a) * d, fz = pz + Math.sin(a) * d, s2 = sc * (0.55 + 0.3 * hash(k, 15, 9));
        if (!blocked(fx, fz, 0.3) && pathDist(fx, fz) > 1.3) {
          out.push({
            id: FERNS[Math.floor(hash(k, 16, 9) * FERNS.length)], lod: 0, x: fx, y: core.groundY(fx, fz) - 0.03, z: fz,
            yaw: hash(k, 17, 9) * Math.PI * 2, tiltDir: a, tilt: 0.05 + 0.12 * hash(k, 18, 9), s: [s2, s2 * (0.8 + 0.4 * hash(k, 19, 9)), s2],
            p: [0, 0.8 + 0.35 * hash(k, 20, 9), 0, hash(k, 21, 9)], q: 1 - cv[0],
          });
        }
      }
    }
  }
  void trees;
  return out;
}

const LUM = vec3(0.2126, 0.7152, 0.0722);

/** shared prop uniforms (calibration) */
export const propU = {
  /** albedo of bare granite boulders (dark grey, ref_02) and of the paler river stones */
  graniteLum: uniform(0.072),
  stoneLum: uniform(0.12),
  mossGain: uniform(1.0),
  /** saturation of the moss (part 8: greens go grey-olive in low light and lose their colour by moonlight) */
  mossSat: uniform(1.0),
};

/**
 * Rocks and logs: the warm scans pulled toward dark grey granite (boulders) or paler river stones, cracks and hollows
 * darker, thick yellow-green moss cushions on the tops with bare rock lower down (ref_03), weathered grey-brown wood
 * or bleached silver-grey driftwood, little sky under the crowns, a dark wet band where they stand in the water.
 * Per-model values are uniforms so all share one program.
 */
function solidMaterial(src: THREE.MeshStandardMaterial, M: PropModel, field: RiverField | null) {
  const m = new THREE.MeshStandardNodeMaterial();
  m.name = 'props-solid';
  const t = uv();
  const base = texture(src.map!, t).rgb;
  const orm = texture(src.roughnessMap ?? src.metalnessMap!, t);
  const inst = attribute('aProp', 'vec4');
  const open = attribute('aOpen', 'float');
  const l = dot(base, LUM).max(1e-4);
  const isLog = uniform(M.kind === 'log' ? 1 : 0);
  const rel = l.div(uniform(M.lum));
  // moss in the scan: olive, its green as strong as its red (the bare rock and wood are brown, green < red)
  const mossScan = smoothstep(0.86, 1.0, base.y.div(base.x.max(1e-3))).mul(smoothstep(0.02, 0.05, l));
  // granite: grey with a cool cast, cracks and hollows (dark in the scan) deeper still
  const crack = smoothstep(0.25, 0.85, rel);
  const lumT = mix(propU.stoneLum, propU.graniteLum, uniform(M.kind === 'boulder' ? 1 : 0));
  const rockCol = mix(vec3(l), base, 0.18).mul(vec3(0.94, 0.98, 1.04)).mul(lumT.div(uniform(M.lum))).mul(mix(0.45, 1.0, crack));
  const woodCol = mix(vec3(l), base, 0.55).mul(vec3(1.0, 0.96, 0.9)).mul(uniform(0.09 / M.lum));
  let col: N = mix(rockCol, woodCol, isLog).mul(inst.y);
  // driftwood bleached silver-grey by sun and floods (inst.z > 0); inst.z < 0 marks a chunk of slumped sod
  const bleachF = inst.z.max(0.0);
  const earthF = inst.z.negate().clamp(0.0, 1.0);
  col = mix(col, vec3(l).mul(uniform(0.3 / M.lum)).mul(vec3(0.95, 0.94, 0.92)), bleachF.mul(smoothstep(0.0, 0.12, l)));
  col = mix(col, vec3(0.056, 0.042, 0.029).mul(mix(0.6, 1.35, crack)).mul(inst.y), earthF);
  // (oblique 2D projections of the world position stand in for 3D noise on surfaces of any orientation)
  const nUp = normalWorldGeometry.y;
  const pw = vec2(positionWorld.x.add(positionWorld.z.mul(0.6)), positionWorld.y.add(positionWorld.z.mul(0.55)));
  const n1 = tnoise(pw.mul(1.9), 1.7);
  const n2 = tnoise(pw.mul(7.3), 4.9);
  // scene fix (critic polish r1: "smooth rounded blobs with a uniform green felt cap"; ref_03): moss in thick lumpy
  // cushions a hand or two across, gathered into patches, thickest on the top faces and in the hollows, with grey
  // rock and pale lichen showing between them; yellow-green on the cushions' crowns, dark olive down between them
  const n3 = tnoise(pw.mul(11.0), 6.1);
  const n4 = tnoise(pw.mul(26.0), 7.7);
  const cover = nUp.mul(0.7).add(n1.sub(0.5).mul(1.15)).add(n2.sub(0.5).mul(0.4)).add(inst.x.mul(0.45)).sub(0.78);
  const cushion = smoothstep(0.46, 0.52, n3.add(cover.mul(0.55))).mul(smoothstep(0.0, 0.15, inst.x));
  const crown = smoothstep(0.45, 0.85, n3).mul(0.6).add(n4.mul(0.4));
  const moss = max(mossScan.mul(0.55).mul(smoothstep(0.25, 0.5, n3.add(cover.mul(0.4)))), cushion).mul(propU.mossGain).clamp(0, 1).toVar();
  const mossHue = mix(vec3(0.02, 0.03, 0.008), vec3(0.15, 0.19, 0.03), crown.mul(0.8).add(smoothstep(0.4, 1.2, rel).mul(0.2)))
    // a few cushions browner (dried on the sunny side)
    .mul(mix(vec3(1), vec3(1.35, 1.0, 0.8), smoothstep(0.62, 0.8, tnoise(pw.mul(3.3), 8.1)).mul(0.6)));
  const mossCol = mix(vec3(dot(mossHue, LUM)).mul(vec3(1.0, 1.02, 0.94)), mossHue, propU.mossSat);
  // a sod chunk's cap is the turf it broke from: olive grass with straw in it
  const turfCol = mix(vec3(0.045, 0.062, 0.022), vec3(0.12, 0.1, 0.05), n1.mul(0.7));
  // bare rock between the cushions: pale grey-green lichen rosettes and dark fracture lines (rocks only)
  const rockF = float(1).sub(isLog).mul(float(1).sub(earthF));
  const lich = smoothstep(0.6, 0.67, tnoise(pw.mul(15.0), 9.9)).mul(float(1).sub(moss)).mul(smoothstep(-0.4, 0.5, nUp)).mul(rockF);
  col = mix(col, vec3(dot(col, LUM)).mul(1.6).add(vec3(0.05, 0.056, 0.045)), lich.mul(0.65));
  const fr = abs(tnoise(vec2(pw.x.mul(2.3), pw.y.mul(1.4)), 10.7).sub(0.5));
  const fracture = float(1).sub(smoothstep(0.006, 0.02, fr)).mul(smoothstep(0.35, 0.6, tnoise(pw.mul(0.9), 11.3))).mul(rockF).mul(float(1).sub(moss));
  col = col.mul(float(1).sub(fracture.mul(0.7)));
  col = mix(col, mix(mossCol, turfCol, earthF), moss);
  // under the crowns a rock sees little sky and gets no direct sun beyond the shadow map: keep it dark
  col = col.mul(mix(0.7, 1.0, open));
  let rough: N = mix(mix(orm.y, 0.9, lich), 0.96, moss);
  let ao: N = mix(float(1), orm.x, uniform(M.ao ? 1 : 0)).mul(mix(0.35, 1.0, open)).mul(mix(0.55, 1.0, crack));
  if (field) {
    // standing in the water: a soaked dark band a hand's width above the surface, glossy
    const F = fieldNodes(field);
    const inside = F.inside(positionWorld.xz);
    const waterY = mix(float(-50), F.A(positionWorld.xz).w, inside);
    const band = float(0.1).add(tnoiseS(pw.mul(4.0), 2.3).mul(0.035));
    const wet = smoothstep(waterY.add(band).add(0.012), waterY.add(band).sub(0.012), positionWorld.y);
    col = mix(col, mix(vec3(dot(col, LUM)), col, 1.4).mul(0.42), wet);
    const far = smoothstep(6.0, 25.0, positionWorld.sub(cameraPosition).length());
    rough = mix(rough, mix(0.3, 0.6, far), wet);
    ao = ao.mul(float(1).sub(wet.mul(0.4)));
  }
  m.colorNode = col;
  m.roughnessNode = rough;
  m.metalnessNode = float(0);
  m.aoNode = ao;
  // the scan's relief, and the moss cushions standing proud of it (bump from their height through screen-space
  // derivatives; the normal is in view space)
  const nMap = normalMap(texture(src.normalMap!, t), uniform(new THREE.Vector2(src.normalScale.x, src.normalScale.y)));
  const H = moss.mul(crown.mul(0.75).add(0.25)).mul(0.045).add(fracture.mul(-0.005));
  const dpx = dFdx(positionView), dpy = dFdy(positionView);
  const r1 = cross(dpy, nMap), r2 = cross(nMap, dpx);
  const det = dot(dpx, r1);
  const grad = r1.mul(dFdx(H)).add(r2.mul(dFdy(H))).mul(sign(det));
  m.normalNode = normalize(abs(det).mul(nMap).sub(grad));
  setOpaqueAlbedo(m, m.colorNode);
  return m;
}

function fernMaterial(src: THREE.MeshStandardMaterial) {
  const m = new THREE.MeshStandardNodeMaterial();
  m.name = 'fern';
  m.side = THREE.DoubleSide;
  m.alphaTest = 0.5;
  const t = uv();
  const c = texture(src.map!, t);
  const inst = attribute('aProp', 'vec4');
  const open = attribute('aOpen', 'float');
  // a shade fern: deep green, a little yellower toward the frond tips
  m.colorNode = vec4(c.rgb.mul(vec3(0.62, 0.72, 0.5)).mul(inst.y).mul(mix(0.75, 1.0, open)), c.a);
  // matte: a fern frond has no gloss (round 3's 0.8 caught the sky and read as big glossy leaves)
  m.roughnessNode = float(0.97);
  m.metalnessNode = float(0);
  m.aoNode = mix(float(0.45), float(1.0), open);
  return m;
}

/**
 * Cut a rounded boulder scan into a fractured block (scene fix, critic polish r1: "smooth rounded blobs"; ref_02):
 * a few planes slice its top and flanks flat, so it has flat faces meeting in edges; the scan's own surface carries
 * on over the rest. In the scan's box-normalised frame, so a plane cuts the same share of a long or a tall rock.
 */
function chisel(g: THREE.BufferGeometry, id: string) {
  g.computeBoundingBox();
  const bb = g.boundingBox!, c = bb.getCenter(new THREE.Vector3()), hs = bb.getSize(new THREE.Vector3()).multiplyScalar(0.5);
  let r = 0;
  for (let i = 0; i < id.length; i++) r = (Math.imul(r, 31) + id.charCodeAt(i)) | 0;
  const rnd = () => ((r = (Math.imul(r, 1664525) + 1013904223) >>> 0) / 4294967296);
  const planes: Array<[number, number, number, number]> = [];
  const nSide = 3 + Math.floor(rnd() * 3), a0 = rnd() * Math.PI * 2;
  // a tilted, nearly flat top, and joint faces round the flanks, each tipped a little
  {
    const az = rnd() * Math.PI * 2, el = Math.PI / 2 - (0.12 + 0.38 * rnd());
    planes.push([Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az), 0.55 + 0.25 * rnd()]);
  }
  for (let k = 0; k < nSide; k++) {
    const az = a0 + (k / nSide) * Math.PI * 2 + (rnd() - 0.5) * 0.8, el = (rnd() - 0.35) * 0.7;
    planes.push([Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az), 0.62 + 0.25 * rnd()]);
  }
  const pos = g.getAttribute('position') as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    let ux = (pos.getX(i) - c.x) / hs.x, uy = (pos.getY(i) - c.y) / hs.y, uz = (pos.getZ(i) - c.z) / hs.z;
    for (const [nx, ny, nz, d] of planes) {
      const e = ux * nx + uy * ny + uz * nz - d;
      if (e > 0) {
        ux -= e * nx;
        uy -= e * ny;
        uz -= e * nz;
      }
    }
    pos.setXYZ(i, c.x + ux * hs.x, c.y + uy * hs.y, c.z + uz * hs.z);
  }
  pos.needsUpdate = true;
  g.computeVertexNormals();
  g.computeBoundingSphere();
}

const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion(), _p = new THREE.Vector3(),
  _s = new THREE.Vector3(), _ax = new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);

/** Load the models used by `placed` and build one instanced mesh per model and LOD. */
export async function loadBankProps(assets: Assets, placed: PropInst[], field: RiverField | null): Promise<THREE.Group> {
  const group = new THREE.Group();
  group.name = 'bank-props';
  const byId = new Map<string, PropInst[]>();
  for (const p of placed) {
    // single-LOD scans (the small stones, ferns) always use their one mesh
    const key = `${p.id}|${MODELS[p.id].kind === 'boulder' || MODELS[p.id].kind === 'log' ? p.lod : 0}|${p.role ?? ''}`;
    let a = byId.get(key);
    if (!a) byId.set(key, (a = []));
    a.push(p);
  }
  // one source mesh and material per model and LOD, shared by its role groups; three r186 still builds a node graph
  // for every InstancedMesh (its uuid is part of the cache key), so each extra group costs one build in the warm-up
  const shared = new Map<string, Promise<{ src: THREE.Mesh; mat: THREE.Material }>>();
  const source = (id: string, lod: number) => {
    const k = `${id}|${lod}`;
    let p = shared.get(k);
    if (!p) {
      p = (async () => {
        const M = MODELS[id];
        const gltf = await assets.gltf(`models/${M.file}`);
        let mesh: THREE.Mesh | null = null;
        const want = `LOD${lod}`;
        gltf.scene.traverse((o) => {
          const mm = o as THREE.Mesh;
          if (!mm.isMesh) return;
          const hit = mm.name.includes(want) || !!mm.parent?.name.includes(want);
          if (!mesh || (hit && !(mesh.name.includes(want) || mesh.parent?.name.includes(want)))) mesh = mm;
        });
        if (!mesh) throw new Error(`no mesh in ${id}`);
        const src = mesh as THREE.Mesh;
        const mat = M.kind === 'fern' ? fernMaterial(src.material as THREE.MeshStandardMaterial)
          : solidMaterial(src.material as THREE.MeshStandardMaterial, M, field);
        return { src, mat };
      })();
      shared.set(k, p);
    }
    return p;
  };
  await Promise.all([...byId].map(async ([key, list]) => {
    const [id, lodS, role] = key.split('|');
    const lod = Number(lodS);
    const M = MODELS[id];
    const { src, mat } = await source(id, lod);
    const g = bakeGeometry(src);
    if (M.kind === 'boulder') chisel(g, id);
    const cap = instanceCapacity(list.length);
    const attr = new THREE.InstancedBufferAttribute(new Float32Array(cap * 4), 4);
    const openA = new THREE.InstancedBufferAttribute(new Float32Array(cap), 1);
    g.setAttribute('aProp', attr);
    g.setAttribute('aOpen', openA);
    const im = new THREE.InstancedMesh(g, mat, cap);
    im.count = list.length;
    list.forEach((p, i) => {
      _q.setFromAxisAngle(UP, p.yaw);
      if (p.tilt !== 0) {
        _ax.set(Math.cos(p.tiltDir + Math.PI / 2), 0, Math.sin(p.tiltDir + Math.PI / 2));
        _q2.setFromAxisAngle(_ax, p.tilt);
        _q.premultiply(_q2);
      }
      _m.compose(_p.set(p.x, p.y, p.z), _q, _s.set(p.s[0], p.s[1], p.s[2]));
      im.setMatrixAt(i, _m);
      attr.setXYZW(i, p.p[0], p.p[1], p.p[2], p.p[3]);
      openA.setX(i, p.q);
    });
    im.instanceMatrix.needsUpdate = true;
    im.computeBoundingSphere();
    im.name = `props-${id}-lod${lod}${role ? '-' + role : ''}`;
    im.castShadow = M.kind !== 'fern' && M.kind !== 'stone' && role !== 'small';
    im.receiveShadow = true;
    if (M.kind === 'fern') im.layers.set(LAYER_FOLIAGE);
    // pebbles, cobbles and small rocks: main view only (their shadows are under a texel and the mirror shows them as
    // specks); forest-floor wood and sod: main view and the sun's shadow map, not the pool mirror
    else if (M.kind === 'stone' || role === 'small') im.layers.set(LAYER_NO_REFLECT);
    else if (role === 'floor') {
      im.layers.set(LAYER_NO_REFLECT);
      im.layers.enable(LAYER_SHADOW);
    }
    group.add(im);
  }));
  return group;
}
