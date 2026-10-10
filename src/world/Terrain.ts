import * as THREE from 'three/webgpu';
import { river, valley, newSample } from './River';
import { fbm, ridged, smoothstep, clamp } from './noise';
import { LAYOUT, type FlatZone } from './Layout';

/**
 * One global height function drives every terrain mesh, the gameplay height queries and
 * the material masks, so the river bed, banks, hills and far mountains always agree.
 */

export interface TerrainPoint {
  y: number;
  /** local water surface height (from the river) */
  waterY: number;
  /** distance from the channel edge, negative inside the channel (m) */
  edge: number;
  /** flow speed at the nearest centreline point (m/s) */
  speed: number;
  /** +1 south (player) bank, -1 north bank */
  side: number;
  /** 0..1 soil and stones slumped off a cut bank onto the bar */
  apron: number;
  /** 0..1 the bank is bedrock rather than earth */
  rock: number;
  /** depth below the top edge of the nearest cut bank (m, 0..3); the root mat and topsoil are the first ~0.3 m */
  belowTop: number;
  /** 0..1 loose river gravel (bed and bar) rather than soil and grass */
  gravel: number;
}

export const newPoint = (): TerrainPoint => ({ y: 0, waterY: 0, edge: 0, speed: 0, side: 1, apron: 0, rock: 0, belowTop: 3, gravel: 0 });

const S = newSample();

const flatCache = new Map<FlatZone, number>();
const FT = newPoint();
function flatTarget(f: FlatZone): number {
  let v = flatCache.get(f);
  if (v === undefined) {
    v = f.y ?? sampleTerrain(f.x, f.z, FT, true).y + (f.dy ?? 0);
    flatCache.set(f, v);
  }
  return v;
}

/** smooth minimum whose rounding grows with the heights, so it stays crisp near the valley floor */
const softMin = (a: number, b: number) => {
  const k = 0.5 + 0.15 * Math.max(0, Math.min(a, b));
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
};

/** A few big massifs of different heights instead of one even row of peaks: x, z, radius, extra height. */
const MASSIFS: Array<[number, number, number, number]> = [
  [1310, 5270, 2700, 2300], // south, at the head of the side valley behind the camp (C5)
  [-4600, 3600, 2400, 1400], // south-west
  [-7600, -700, 3200, 1500], // head of the main valley (C1)
  [900, -6300, 3400, 2200], // north, behind the north wall
  [-3200, -6900, 2600, 1600], // north-west
  [7200, 2600, 3000, 1500], // down-valley
];

/** Side valley opening south-south-east behind the camp: axis origin and direction. */
const SV = { x: 14, z: 30, ax: 0.24, az: 0.97 };

export function sampleTerrain(x: number, z: number, out: TerrainPoint, skipFlats = false): TerrainPoint {
  const r = Math.hypot(x, z);
  if (r > 700) {
    // far field: the valley profile has saturated, skip the polyline search
    S.d = z - river.zAt(x);
    S.half = 4; S.depth = 0.7; S.waterY = river.waterYAt(x); S.speed = 1; S.asym = 0; S.curv = 0;
    S.tx = 1; S.tz = 0; S.s = x + 760;
  } else {
    river.sample(x, z, S);
  }
  const sgn = S.d >= 0 ? 1 : -1;
  const south = sgn > 0;
  const t = S.d / S.half;
  // the north bank's margin wanders in and out by a few decimetres every metre or two (scoured pockets, slumped
  // sod, stones), so its waterline never runs ruler-straight (round 1, C4). The south bars keep their shape.
  const margin = Math.abs(S.d) - S.half;
  const nearM = south ? 0 : 1 - smoothstep(1.5, 4.0, Math.abs(margin));
  const wander = nearM > 0 ? (fbm(x * 0.5 + 3.1, z * 0.5 - 1.9, 2) * 0.65 + fbm(x * 1.1 - 4.4, z * 1.1 + 2.2, 2) * 0.35) * 1.8 * nearM : 0;
  const edge = margin - wander;
  // inner (+1, depositional) or outer (-1, erosional) bank of the local bend
  const I = clamp(sgn * S.curv * 40, -1, 1);
  const inner = Math.max(I, 0), outer = Math.max(-I, 0);
  const so = south ? 0 : 37.0; // decorrelates the two banks
  // bedrock where the river cuts into the valley side, in a few stretches of the outer banks
  const rockN = fbm(S.s * 0.035 + 5.5 + so, 9.0, 3);
  const rock = smoothstep(0.0, 0.22, rockN + outer * 0.45 - 0.3);
  let y: number;
  let apron = 0, gravel = 1, belowTop = 3;

  if (edge < 0) {
    // channel bed. The thalweg sits toward the outer bank; the bed rises steeply (not sheer) to an outer
    // bank and shoals gently toward an inner one.
    const c = S.asym;
    const u = t < c ? (t - c) / (1 + c) : (t - c) / (1 - c);
    const Iu = clamp((u < 0 ? -1 : 1) * S.curv * 40, -1, 1);
    const p = Iu < 0 ? 1.7 - 0.15 * -Iu : 1.75 - 0.1 * Iu;
    const prof = 1 - Math.pow(Math.min(1, Math.abs(u)), p);
    // scour pockets along the deep side, shallower crossings between them
    const scour = 1 + 0.2 * fbm(S.s * 0.05 + 4.4, 0.7, 2);
    y = S.waterY - 0.08 - (S.depth * scour - 0.08) * prof;
    // riffles step down in low cobble ledges instead of one smooth ramp
    const fastF = smoothstep(1.3, 1.9, S.speed);
    if (fastF > 0) {
      const ph = S.s / 6.5 + S.d * 0.07 + fbm(S.s * 0.1, S.d * 0.2, 2) * 0.35;
      const f = ph - Math.floor(ph);
      const saw = f * (1 - smoothstep(0.8, 1.0, f)) + smoothstep(0.62, 0.8, f) * (1 - smoothstep(0.8, 0.92, f)) * 0.25;
      y += (saw - 0.45) * 0.17 * fastF * Math.min(1, prof * 3);
    }
    // bed relief: gravel ribs elongated with the flow, lobes, scours and small lumps
    y += fbm(S.s * 0.09, S.d * 0.25, 4) * 0.16 * (0.4 + prof);
    y += smoothstep(0.05, 0.3, fbm(S.s * 0.06 + 2.0, S.d * 0.3 + so, 3)) * 0.1 * (1 - prof);
    y += fbm(x * 0.6 + 3.1, z * 0.6, 2) * 0.04;
    belowTop = 3;
  } else {
    const e = edge;
    // along-bank variation at three scales: the bank changes character within metres
    const nA = fbm(S.s * 0.045 + 3.0 + so, 0.5, 3);
    const nB = fbm(S.s * 0.13 + 7.0 + so, 1.5, 3);
    const nC = fbm(S.s * 0.37 + 1.3 + so, 4.5, 2);
    // bar: broad on the inner side of a bend, a thin strip on the outer side
    // (bank position and steepness change over metres, not decimetres, or the bank top juts out in sharp
    // promontories that read as isolated blocks)
    const barW = Math.max(0.5, 1.0 + 8.0 * smoothstep(-0.4, 0.8, I) + nA * 2.6 + nB * 0.45);
    // rain gullies: V-shaped notches in a few stretches, where the bank top dips and spills a fan onto the bar
    const gully = smoothstep(0.8, 0.97, 1 - Math.abs(fbm(S.s * 0.05 + 2.2 + so, 0.3, 2)))
      * smoothstep(0.05, 0.25, fbm(S.s * 0.021 + 6.0 + so, 2.0, 2));
    // terrace level behind the bank varies in 2D (never along the river alone, which would groove the
    // land behind it); the cut height follows from how high the bar has built up below it
    let tl = 0.95 + 0.45 * fbm(x * 0.05 + 3.3, z * 0.05, 3) + 0.25 * fbm(x * 0.13, z * 0.13 + 1.1, 2);
    const cutting = smoothstep(0.2, 0.8, outer);
    // how the bank face stands: sheer cuts belong to tall outer banks and bedrock; low banks and inner bends
    // are slumped, grassy slopes. Slump blocks and fresh scarps alternate every several metres (a warped wave).
    const wave = Math.sin(S.s * 0.45 + nA * 5.0 + so);
    const slump = clamp(0.45 + 0.55 * wave + 0.6 * nB + 0.45 * inner - 0.1 * outer - rock * 0.6 + gully * 0.6
      + 0.6 * (1 - cutting) * smoothstep(1.5, 0.7, tl), 0, 1);
    const cutW = 0.4 + slump * slump * 4.2;
    const top = barW + cutW;
    // outer bends cut into higher ground (and bedrock stands proud): taller banks whose top rises and falls
    // along the bank and eases back a few metres in; inner bends grade down into the point bar
    const lip = 1 - smoothstep(top + 3, top + 25, e);
    tl += (cutting * (1.3 + 0.9 * nB) + rock * 0.6 - inner * 0.35) * lip;
    // gullies notch the terrace edge and die out a few metres back
    tl -= gully * 0.55 * Math.max(tl - 0.3, 0) * (1 - smoothstep(top, top + 3.5, e));
    if (south) tl += smoothstep(4, 26, e) * 2.8; // rising to the cabin plateau
    const terraceY = S.waterY - 0.08 + Math.max(tl, 0.25);
    // bar: low by the water, a little steeper toward the back, never reaching the terrace
    const rise = (q: number) => q * 0.045 + Math.min(q, 2) * 0.03 + smoothstep(0.5, 1.0, q / barW) * barW * 0.02;
    const barTop = Math.min(rise(barW), Math.max(tl, 0.25) - 0.22);
    const eb = Math.min(e, barW);
    let yb = S.waterY - 0.08 + rise(eb) * (barTop / Math.max(rise(barW), 1e-3));
    const lobe = smoothstep(0.05, 0.28, fbm(S.s * 0.07 + so, S.d * 0.16, 3));
    const chute = smoothstep(0.86, 0.97, 1 - Math.abs(fbm(S.s * 0.022 + 9.1 + so, S.d * 0.2, 2)));
    const onBar = 1 - smoothstep(barW - 1.0, barW, e);
    yb += (lobe * 0.13 * smoothstep(0, 1.2, e + 0.3) - chute * 0.1 * smoothstep(1.0, 3.0, e)) * onBar;
    // apron of slumped soil and stones at the foot, wider below gullies
    const apronW = 0.9 + 1.6 * gully + 0.8 * slump;
    const cutH = terraceY - (S.waterY - 0.08 + barTop);
    const apronH = (0.1 + 0.25 * gully + 0.08 * Math.max(nC, 0)) * clamp(cutH * 1.3, 0, 1);
    yb += apronH * smoothstep(barW - apronW, barW, e);
    const footY = S.waterY - 0.08 + barTop + apronH;
    let face = smoothstep(barW, top, e);
    // bedrock breaks into a couple of ledges; earth faces bulge and slump unevenly with stones sticking out
    if (rock > 0) {
      // smooth staircase (flat treads, steep risers); a floor() here cracked the face into vertical panels
      const lq = face * 2.6 + fbm(x * 0.3, z * 0.3 + 2.0, 2) * 0.8;
      const ledges = (lq - Math.sin(lq * 2 * Math.PI) / (2 * Math.PI)) / 2.6;
      face += (clamp(ledges, 0, 1) - face) * rock * 0.8;
    }
    const lumps = (fbm(x * 0.8 + 1.7, z * 0.8, 2) * 0.6 + fbm(x * 2.1, z * 2.1 - 3.3, 2) * 0.4) * 4 * face * (1 - face);
    y = e < barW ? yb : footY + (terraceY - footY) * face + lumps * Math.min(0.35, (terraceY - footY) * 0.3);
    apron = smoothstep(barW - apronW, barW, e) * (1 - smoothstep(barW + cutW * 0.3, barW + cutW * 0.6 + 0.2, e));
    // gravel ends in a ragged line at the foot of the bank
    gravel = 1 - smoothstep(barW - 0.9 + nC * 0.6, barW + 0.15, e);
    const cutTop = terraceY;
    // valley walls follow the smooth valley axis, not every bend of the channel
    const ev = sgn * (z - valley.zAt(x));
    let wall: number;
    if (south) {
      wall = smoothstep(30, 150, ev) * 62 + smoothstep(110, 640, ev) * 380;
    } else {
      wall = smoothstep(11, 95, ev) * 34 + smoothstep(60, 480, ev) * 290;
    }
    // ravines and spurs on the valley walls: ridged noise stretched downslope (roughly north-south)
    const wallF = smoothstep(50, 220, ev);
    if (wallF > 0) {
      wall += wallF * (ridged(x * 0.011 + 2.0, z * 0.0035 - 1.0, 5) - 0.45) * 55;
      wall += wallF * (ridged(x * 0.0035 + 4.1, z * 0.0035, 6) - 0.35) * 60;
    }
    if (south) {
      // the side valley is a V cut into the south wall: its floor climbs gently, its sides at ~30 deg
      const along = (x - SV.x) * SV.ax + (z - SV.z) * SV.az;
      if (along > -60) {
        const a = Math.max(along, 0);
        // the side valley winds a little and its walls are uneven
        const wind = fbm(a * 0.0035 + 1.7, 0.4, 2) * 70 * smoothstep(60, 300, a);
        const lat = Math.abs((x - SV.x) * SV.az - (z - SV.z) * SV.ax + wind);
        const sideSlope = 0.62 + 0.18 * fbm(x * 0.006, z * 0.006 + 3.0, 2);
        const sv = a * 0.07 + Math.max(0, lat - (16 + a * 0.1)) * sideSlope + fbm(x * 0.02, z * 0.02, 3) * 6 * smoothstep(0, 80, a);
        wall += (softMin(wall, Math.max(sv, 0)) - wall) * smoothstep(0.2, 2.0, wall);
      }
    }
    y += wall;
    const near = 1 - smoothstep(0, 6, e);
    // lumpy terraces: mid-frequency relief so nothing reads as a smooth dome
    y += fbm(x * 0.13 + 1.3, z * 0.13 - 4.0, 3) * (0.05 * near + 0.3 * (1 - near) * (1 - smoothstep(40, 120, e)));
    const amp = 0.04 * near + (0.15 + smoothstep(6, 40, e) * 1.4 + smoothstep(40, 300, e) * 16) * (1 - near);
    y += fbm(x * 0.045, z * 0.045, 5) * amp;
    y += fbm(x * 0.35 + 7.7, z * 0.35, 3) * (0.03 + 0.07 * (1 - near));
    belowTop = clamp(cutTop - y, 0, 3);
    // high on the walls every steep face is rock
    belowTop = Math.max(belowTop, 3 * smoothstep(20, 40, ev));
  }

  // flattened building spots: level toward the natural height at their centre, so they never
  // become mounds or pits, then keep a little lumpiness so they do not read as a sports pitch
  if (!skipFlats) {
    for (const f of LAYOUT.flats) {
      const dist = Math.hypot(x - f.x, z - f.z);
      if (dist < f.r + f.blend) {
        const w = (1 - smoothstep(f.r, f.r + f.blend, dist)) * (f.strength ?? 0.85);
        const target = flatTarget(f) + fbm(x * 0.4, z * 0.4, 2) * 0.1;
        y += (target - y) * w;
      }
    }
  }

  // distant mountain ranges: lower background ranges plus a few big massifs
  if (r > 600) {
    const north = clamp(-z / r, 0, 1), west = clamp(-x / r, 0, 1), southF = clamp(z / r, 0, 1);
    const ramp = smoothstep(600, 3200, r) * (1 - smoothstep(9000, 12000, r));
    const amp = 380 + 520 * north + 300 * west + 120 * southF;
    // domain warp breaks the ridged noise into ranges and cirques instead of a regular lattice
    const wx = x + fbm(x * 0.00021, z * 0.00021, 3) * 1400;
    const wz = z + fbm(x * 0.00021 + 5.2, z * 0.00021 - 3.1, 3) * 1400;
    // fewer octaves than before: the finest ones only aliased into spiky "crumpled paper" on the ring mesh
    const m = ridged(wx * 0.00042 + 11.3, wz * 0.00042 - 2.7, 6);
    let boost = 0;
    for (const [mx, mz, mr, mh] of MASSIFS) {
      const q = Math.hypot(x - mx, z - mz) / mr;
      if (q < 1) {
        const b = 1 - q * q;
        boost += mh * b * b;
      }
    }
    y += ramp * (m * amp + boost * (0.4 + 0.8 * m) + smoothstep(600, 2500, r) * 120);
  }

  out.y = y;
  out.waterY = S.waterY;
  out.edge = edge;
  out.speed = S.speed;
  out.side = sgn;
  out.apron = apron;
  out.rock = rock;
  out.belowTop = belowTop;
  out.gravel = gravel;
  return out;
}

const TP = newPoint();
export function heightAt(x: number, z: number) {
  return sampleTerrain(x, z, TP).y;
}

export interface GridSpec {
  x0: number; z0: number; x1: number; z1: number; step: number;
  /** lower vertices that fall strictly inside this rectangle (hidden under a finer mesh) */
  hole?: { x0: number; z0: number; x1: number; z1: number; drop: number };
  /** add a vertical skirt around the border to hide cracks against coarser meshes */
  skirt?: number;
}

/** separable box blur of a grid (radius in cells), clamped at the borders */
function boxBlur(src: Float32Array, nx: number, nz: number, rad: number): Float32Array {
  const tmp = new Float32Array(src.length), dst = new Float32Array(src.length);
  const n = 2 * rad + 1;
  for (let j = 0; j < nz; j++) {
    const row = j * nx;
    let acc = 0;
    for (let i = -rad; i <= rad; i++) acc += src[row + clamp(i, 0, nx - 1)];
    for (let i = 0; i < nx; i++) {
      tmp[row + i] = acc / n;
      acc += src[row + Math.min(nx - 1, i + rad + 1)] - src[row + Math.max(0, i - rad)];
    }
  }
  for (let i = 0; i < nx; i++) {
    let acc = 0;
    for (let j = -rad; j <= rad; j++) acc += tmp[clamp(j, 0, nz - 1) * nx + i];
    for (let j = 0; j < nz; j++) {
      dst[j * nx + i] = acc / n;
      acc += tmp[Math.min(nz - 1, j + rad + 1) * nx + i] - tmp[Math.max(0, j - rad) * nx + i];
    }
  }
  return dst;
}

/**
 * Regular grid terrain mesh. Attributes:
 *  position, normal (from grid finite differences), aWater = (waterY, edge, speed, apron),
 *  aBank = (rock, hollow, belowTop, gravel); hollow > 0 in local dips (m below the surrounding ground)
 */
export function buildGrid(spec: GridSpec): { geometry: THREE.BufferGeometry; heights: Float32Array; nx: number; nz: number } {
  const nx = Math.round((spec.x1 - spec.x0) / spec.step) + 1;
  const nz = Math.round((spec.z1 - spec.z0) / spec.step) + 1;
  const heights = new Float32Array(nx * nz);
  const water = new Float32Array(nx * nz * 4);
  const bank = new Float32Array(nx * nz * 4);
  const tp = newPoint();
  for (let j = 0; j < nz; j++) {
    const z = spec.z0 + j * spec.step;
    for (let i = 0; i < nx; i++) {
      const x = spec.x0 + i * spec.step;
      sampleTerrain(x, z, tp);
      const k = j * nx + i;
      let y = tp.y;
      const h = spec.hole;
      if (h && x > h.x0 && x < h.x1 && z > h.z0 && z < h.z1) y -= h.drop;
      heights[k] = y;
      water[k * 4] = tp.waterY;
      water[k * 4 + 1] = tp.edge;
      water[k * 4 + 2] = tp.speed;
      water[k * 4 + 3] = tp.apron;
      bank[k * 4] = tp.rock;
      bank[k * 4 + 2] = tp.belowTop;
      bank[k * 4 + 3] = tp.gravel;
    }
  }
  // hollows: how far a point sits below the mean of its surroundings, at ~1.5 m and ~5 m scales
  const r1 = Math.max(1, Math.round(0.8 / spec.step)), r2 = Math.max(2, Math.round(2.6 / spec.step));
  const b1 = boxBlur(heights, nx, nz, r1), b2 = boxBlur(heights, nx, nz, r2);
  for (let k = 0; k < nx * nz; k++) bank[k * 4 + 1] = clamp((b1[k] - heights[k]) * 0.6 + (b2[k] - heights[k]) * 0.4, -1, 1);

  const skirt = spec.skirt ?? 0;
  const borderCount = skirt > 0 ? 2 * (nx + nz) - 4 : 0;
  const vcount = nx * nz + borderCount;
  const pos = new Float32Array(vcount * 3);
  const nor = new Float32Array(vcount * 3);
  const wat = new Float32Array(vcount * 4);
  const bnk = new Float32Array(vcount * 4);
  const s2 = spec.step * 2;
  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      pos[k * 3] = spec.x0 + i * spec.step;
      pos[k * 3 + 1] = heights[k];
      pos[k * 3 + 2] = spec.z0 + j * spec.step;
      const hl = heights[j * nx + Math.max(0, i - 1)], hr = heights[j * nx + Math.min(nx - 1, i + 1)];
      const hd = heights[Math.max(0, j - 1) * nx + i], hu = heights[Math.min(nz - 1, j + 1) * nx + i];
      const ddx = (hr - hl) / s2, ddz = (hu - hd) / s2;
      const l = Math.hypot(ddx, 1, ddz);
      nor[k * 3] = -ddx / l;
      nor[k * 3 + 1] = 1 / l;
      nor[k * 3 + 2] = -ddz / l;
      wat.set(water.subarray(k * 4, k * 4 + 4), k * 4);
      bnk.set(bank.subarray(k * 4, k * 4 + 4), k * 4);
    }
  }

  const index: number[] = [];
  for (let j = 0; j < nz - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
      // alternate the diagonal for a less directional look
      if ((i + j) & 1) index.push(a, c, b, b, c, d);
      else index.push(a, c, d, a, d, b);
    }
  }

  if (skirt > 0) {
    // border loop in order, then a dropped copy
    const loop: number[] = [];
    for (let i = 0; i < nx; i++) loop.push(i);
    for (let j = 1; j < nz; j++) loop.push(j * nx + nx - 1);
    for (let i = nx - 2; i >= 0; i--) loop.push((nz - 1) * nx + i);
    for (let j = nz - 2; j > 0; j--) loop.push(j * nx);
    const base = nx * nz;
    loop.forEach((src, n) => {
      const k = base + n;
      pos[k * 3] = pos[src * 3];
      pos[k * 3 + 1] = pos[src * 3 + 1] - skirt;
      pos[k * 3 + 2] = pos[src * 3 + 2];
      nor.set(nor.subarray(src * 3, src * 3 + 3), k * 3);
      wat.set(wat.subarray(src * 4, src * 4 + 4), k * 4);
      bnk.set(bnk.subarray(src * 4, src * 4 + 4), k * 4);
    });
    for (let n = 0; n < loop.length; n++) {
      const a = loop[n], b = loop[(n + 1) % loop.length];
      const a2 = base + n, b2 = base + ((n + 1) % loop.length);
      index.push(a, a2, b, b, a2, b2);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  geometry.setAttribute('aWater', new THREE.BufferAttribute(wat, 4));
  geometry.setAttribute('aBank', new THREE.BufferAttribute(bnk, 4));
  geometry.setIndex(vcount > 65535 ? new THREE.Uint32BufferAttribute(index, 1) : new THREE.Uint16BufferAttribute(index, 1));
  geometry.computeBoundingSphere();
  geometry.computeBoundingBox();
  return { geometry, heights, nx, nz };
}

/** Polar ring for the distant mountains: exponential radial spacing. */
export function buildMountainRing(r0: number, r1: number, segments: number, rings: number, hole: { half: number; drop: number }) {
  const pos: number[] = [], nor: number[] = [], wat: number[] = [];
  const tp = newPoint();
  const k = Math.log(r1 / r0);
  for (let j = 0; j <= rings; j++) {
    const r = r0 * Math.exp((k * j) / rings);
    for (let i = 0; i <= segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      const x = Math.cos(a) * r, z = Math.sin(a) * r;
      sampleTerrain(x, z, tp);
      let y = tp.y;
      if (Math.abs(x) < hole.half && Math.abs(z) < hole.half) y -= hole.drop;
      pos.push(x, y, z);
      wat.push(tp.waterY, tp.edge, tp.speed, tp.apron);
    }
  }
  // normals from the grid itself: cross of the angular and radial neighbour differences
  const row0 = segments + 1;
  const vtx = (j: number, i: number) => {
    const k = (j * row0 + i) * 3;
    return new THREE.Vector3(pos[k], pos[k + 1], pos[k + 2]);
  };
  const va = new THREE.Vector3(), vr = new THREE.Vector3(), nn = new THREE.Vector3();
  for (let j = 0; j <= rings; j++) {
    for (let i = 0; i <= segments; i++) {
      const ia = (i + 1) % segments, ib = (i - 1 + segments) % segments;
      va.subVectors(vtx(j, ia), vtx(j, ib));
      vr.subVectors(vtx(Math.min(rings, j + 1), i), vtx(Math.max(0, j - 1), i));
      nn.crossVectors(vr, va).normalize();
      if (nn.y < 0) nn.negate();
      nor.push(nn.x, nn.y, nn.z);
    }
  }
  // counter-clockwise seen from above (b is a step round, c a step out: (b - a) x (c - a) points up). (Part 8 round 2:
  // the old order (a, c, b) faced every triangle down, so the culling kept the far side of each mountain and dropped
  // the slopes facing the eye: the ranges were drawn inside out, lit as if they faced the light behind them, critic
  // r1: "the snow peaks glow on the faces toward the camera while the moon hangs above and behind them")
  const index: number[] = [];
  const row = segments + 1;
  for (let j = 0; j < rings; j++) {
    for (let i = 0; i < segments; i++) {
      const a = j * row + i, b = a + 1, c = a + row, d = c + 1;
      index.push(a, b, c, b, d, c);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  geometry.setAttribute('aWater', new THREE.Float32BufferAttribute(wat, 4));
  geometry.setIndex(index);
  geometry.computeBoundingSphere();
  return geometry;
}

/** Bilinear height lookup into the fine grid (gameplay: walking, placing props). */
export class HeightField {
  constructor(
    private heights: Float32Array, private nx: number, private nz: number,
    private x0: number, private z0: number, private step: number,
  ) {}

  get(x: number, z: number): number {
    const fx = (x - this.x0) / this.step, fz = (z - this.z0) / this.step;
    if (fx < 0 || fz < 0 || fx >= this.nx - 1 || fz >= this.nz - 1) return heightAt(x, z);
    const i = Math.floor(fx), j = Math.floor(fz);
    const tx = fx - i, tz = fz - j;
    const h = this.heights, n = this.nx;
    const a = h[j * n + i], b = h[j * n + i + 1], c = h[(j + 1) * n + i], d = h[(j + 1) * n + i + 1];
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * tz;
  }
}
