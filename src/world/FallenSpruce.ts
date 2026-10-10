import * as THREE from 'three/webgpu';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import {
  atan, attribute, dot, exp, float, max, mix, normalMap, normalWorldGeometry, positionWorld, smoothstep, texture, uniform, uv,
  vec2, vec3, type N,
} from '../gfx/tsl';
import { tnoise } from '../gfx/noiseTex';
import { setOpaqueAlbedo } from '../gfx/opaque';
import type { Assets } from '../core/Assets';
import { installUnderwater } from '../water/Underwater';
import { bakeGeometry } from '../water/Rocks';
import type { Obstacle, RiverField } from '../water/RiverField';
import { parseGlb } from './glb';
import { POOL_LEVEL } from './River';

/**
 * The spruce fallen from the north bank into the pool (part 3 leftover, rebuilt in part 4). The forest's own spruce
 * (spruce_a LOD0 bark: the trunk and its branch skeleton, no needles) scaled to a ~1.1 m butt and a 0.7 m bole,
 * its top snapped off 9.8 m up, lying from a root plate torn out of a notch in the cut bank down into the
 * water at 9 degrees: dry above, awash for ~4 m, its broken end ~0.5 m under. Most branches are broken to stubs or
 * snapped short; the ones that went into the bank or the bed end there. The root plate stands up behind the butt:
 * the stump scan (tree_stump_01) turned onto its side as the root collar, a lumpy disc of soil and old forest floor,
 * and roots (copies of the spruce's own branch tubes) sticking out of its rim. One merged mesh with world positions
 * (the underwater hook reads positionLocal as world space) and one material; it is on the default layer, so the
 * main view, the pool mirror, the sun's shadow map and the AO pre-pass each draw it once.
 */
export const SPRUCE = {
  /** trunk base: the middle of the root plate, at the foot of the notch the fall tore out of the bank */
  base: [7.3, 0.85, -11.25] as const,
  /** fall direction in x/z (unit) and how steeply the trunk goes down (deg) */
  dir: [0.5, 0.866] as const,
  pitchDeg: 9,
  /** roll of the tree about its own axis (rad): which whorls end up on top */
  roll: 0.6,
  scale: 1.3,
  /** where the top snapped off, metres along the trunk */
  length: 9.8,
  /** root plate radius across and up (m): a disc of earth and roots well over a man's height */
  plate: [1.75, 1.55] as const,
};

const UP = new THREE.Vector3(0, 1, 0);
const PITCH = (SPRUCE.pitchDeg * Math.PI) / 180;
const AX = new THREE.Vector3(SPRUCE.dir[0] * Math.cos(PITCH), -Math.sin(PITCH), SPRUCE.dir[1] * Math.cos(PITCH)).normalize();
/** the tree's frame: model +Y along the fallen trunk, turned by the roll */
const Q_TREE = new THREE.Quaternion().setFromUnitVectors(UP, AX).multiply(new THREE.Quaternion().setFromAxisAngle(UP, SPRUCE.roll));
/** two directions across the trunk (its cylindrical frame) */
const SIDE = new THREE.Vector3(1, 0, 0).applyQuaternion(Q_TREE);
const UP2 = new THREE.Vector3().crossVectors(AX, SIDE).normalize();
/** model radius of spruce_a's trunk (m) at model height y: measured from the GLB, 0.25 at 1 m to 0.135 at 12 m */
const trunkR = (y: number) => Math.max(0.03, 0.262 - 0.0105 * y);
/** extra girth over the standing tree (the model's own butt flare, +50 %, is kept as it is) */
const GIRTH = 1.12;
/** world point on the trunk axis s metres from the base */
export function spruceAxis(s: number): [number, number, number] {
  const b = SPRUCE.base;
  return [b[0] + AX.x * s, b[1] + AX.y * s, b[2] + AX.z * s];
}
/** trunk radius (m, as drawn) s metres from the base */
export const spruceRadius = (s: number) => trunkR(s / SPRUCE.scale) * GIRTH * SPRUCE.scale;

/** 2D distance from (x, z) to the trunk's footprint (the axis from the root plate to the break) */
export function spruceDist(x: number, z: number): number {
  const b = SPRUCE.base;
  const dx = x - b[0], dz = z - b[2];
  const hl = Math.hypot(AX.x, AX.z), ux = AX.x / hl, uz = AX.z / hl;
  const t = Math.min(SPRUCE.length * hl, Math.max(0, dx * ux + dz * uz));
  return Math.hypot(dx - ux * t, dz - uz * t);
}

/**
 * The water's field round the trunk where it lies awash (baked before any model loads): a scum line on the
 * upstream side, the bed lifted to the trunk so the water there reads as shallow. A chain of small cylinders
 * whose radius is the trunk's half-width at the waterline.
 */
export function spruceObstacles(): Obstacle[] {
  const out: Obstacle[] = [];
  for (let s = 0; s <= SPRUCE.length; s += 0.35) {
    const [x, y, z] = spruceAxis(s);
    const r = spruceRadius(s);
    const dy = y - POOL_LEVEL;
    if (Math.abs(dy) >= r * 0.95) continue;
    out.push({ x, z, r: Math.sqrt(r * r - dy * dy), topY: y + r });
  }
  return out;
}

function hash(a: number, b: number) {
  let h = (Math.imul(a | 0, 374761393) + Math.imul(b | 0, 668265263)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

interface Piece { pos: number[]; nrm: number[]; uv: number[]; spr: number[]; idx: number[] }
interface Branch { c: number; x: number; y: number; z: number; r: number; tx: number; ty: number; tz: number; reach: number; tris: number[] }

/** the bark mesh split into its pieces (ez-tree writes the trunk and every branch as separate tubes) */
function analyse(g: THREE.BufferGeometry) {
  const P = g.getAttribute('position');
  const idx = g.index!.array;
  const n = P.count;
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (a: number) => {
    while (parent[a] !== a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  };
  for (let t = 0; t < idx.length; t += 3) {
    const a = find(idx[t]);
    parent[find(idx[t + 1])] = a;
    parent[find(idx[t + 2])] = a;
  }
  const comp = new Int32Array(n);
  for (let i = 0; i < n; i++) comp[i] = find(i);
  // the trunk is the piece reaching highest
  const top = new Map<number, number>();
  for (let i = 0; i < n; i++) top.set(comp[i], Math.max(top.get(comp[i]) ?? -1e9, P.getY(i)));
  let trunk = -1, best = -1e9;
  for (const [c, y] of top) if (y > best) [best, trunk] = [y, c];
  // each branch: its base (the vertex nearest the axis), its reach, its general direction and its triangles
  const br = new Map<number, Branch>();
  for (let i = 0; i < n; i++) {
    const c = comp[i];
    if (c === trunk) continue;
    const x = P.getX(i), y = P.getY(i), z = P.getZ(i), r = Math.hypot(x, z);
    const b = br.get(c);
    if (!b) br.set(c, { c, x, y, z, r, tx: 0, ty: 0, tz: 0, reach: 0, tris: [] });
    else if (r < b.r) Object.assign(b, { x, y, z, r });
  }
  for (let i = 0; i < n; i++) {
    const b = br.get(comp[i]);
    if (!b) continue;
    const dx = P.getX(i) - b.x, dy = P.getY(i) - b.y, dz = P.getZ(i) - b.z, d = Math.hypot(dx, dy, dz);
    if (d > b.reach) Object.assign(b, { reach: d, tx: dx / d, ty: dy / d, tz: dz / d });
  }
  for (let t = 0; t < idx.length; t += 3) br.get(comp[idx[t]])?.tris.push(t);
  return { comp, trunk, branches: [...br.values()].sort((a, b) => a.y - b.y) };
}

/**
 * A crooked tube along `pts` (world), `sides` round, radius per point, its last ring pinched into a ragged tip;
 * ring frames are carried along the line so it does not twist.
 */
function tube(out: Piece, pts: THREE.Vector3[], radii: number[], sides: number, spr: [number, number, number, number], tipJag = 0) {
  const o0 = out.pos.length / 3;
  const t = new THREE.Vector3(), n1 = new THREE.Vector3(), n2 = new THREE.Vector3(), d = new THREE.Vector3(), tmp = new THREE.Vector3();
  let len = 0;
  for (let i = 0; i < pts.length; i++) {
    if (i > 0) len += pts[i].distanceTo(pts[i - 1]);
    t.subVectors(pts[Math.min(pts.length - 1, i + 1)], pts[Math.max(0, i - 1)]).normalize();
    if (i === 0) n1.crossVectors(t, Math.abs(t.y) < 0.9 ? UP : tmp.set(1, 0, 0)).normalize();
    else n1.addScaledVector(t, -n1.dot(t)).normalize();
    n2.crossVectors(t, n1).normalize();
    for (let k = 0; k < sides; k++) {
      const a = (k / sides) * Math.PI * 2;
      d.copy(n1).multiplyScalar(Math.cos(a)).addScaledVector(n2, Math.sin(a));
      const last = i === pts.length - 1;
      const jag = last ? tipJag * (0.4 + 0.6 * hash(o0 + k, 91)) : 0;
      out.pos.push(pts[i].x + d.x * radii[i] + t.x * jag, pts[i].y + d.y * radii[i] + t.y * jag, pts[i].z + d.z * radii[i] + t.z * jag);
      out.nrm.push(d.x, d.y, d.z);
      out.uv.push((k / sides) * 0.6, len * 1.2);
      out.spr.push(...spr);
    }
  }
  for (let i = 0; i < pts.length - 1; i++) {
    for (let k = 0; k < sides; k++) {
      const a = o0 + i * sides + k, b = o0 + i * sides + ((k + 1) % sides);
      out.idx.push(a, b + sides, b, a, a + sides, b + sides);
    }
  }
  // the broken end closed with a fan
  const c = out.pos.length / 3, l = pts[pts.length - 1];
  t.subVectors(l, pts[pts.length - 2]).normalize();
  out.pos.push(l.x + t.x * tipJag * 0.3, l.y + t.y * tipJag * 0.3, l.z + t.z * tipJag * 0.3);
  out.nrm.push(t.x, t.y, t.z);
  out.uv.push(0.3, len * 1.2);
  out.spr.push(...spr);
  const r0 = o0 + (pts.length - 1) * sides;
  for (let k = 0; k < sides; k++) out.idx.push(r0 + k, r0 + ((k + 1) % sides), c);
}

/** a kinked line from `p0` along `dir` for `L` metres in `n` segments, each bending by up to `bend` rad */
function crooked(p0: THREE.Vector3, dir: THREE.Vector3, L: number, n: number, bend: number, seed: number, droop = 0) {
  const pts = [p0.clone()];
  const d = dir.clone().normalize(), q = new THREE.Quaternion(), ax = new THREE.Vector3();
  for (let i = 1; i <= n; i++) {
    ax.set(hash(seed, i * 3) - 0.5, hash(seed, i * 3 + 1) - 0.5, hash(seed, i * 3 + 2) - 0.5).normalize();
    d.applyQuaternion(q.setFromAxisAngle(ax, bend * (0.5 + hash(seed, i * 7))));
    d.y -= droop;
    d.normalize();
    pts.push(pts[i - 1].clone().addScaledVector(d, L / n));
  }
  return pts;
}

/**
 * Whorls of snapped branch stubs along the trunk (critic r2: "thin pale straws of about the same length stick out
 * evenly all along and all round the trunk, like a bottle brush"): every 0.4-0.6 m a ring of three to five short,
 * thick stubs (3-13 cm, as thick as a spruce's branch bases, swollen where they leave the trunk), snapped ragged,
 * dark with old bark (aSpr.y 0.35), a few gone altogether. Stubs that would go into the bank end there.
 */
function addStubs(groundY: (x: number, z: number) => number, out: Piece) {
  const e = new THREE.Vector3(), dir = new THREE.Vector3(), start = new THREE.Vector3();
  let s = 0.75, w = 0;
  while (s < SPRUCE.length - 0.3) {
    const [cx, cy, cz] = spruceAxis(s);
    const r = spruceRadius(s);
    const along = s / SPRUCE.length;
    const n = 3 + Math.floor(hash(w, 1) * 3);
    const th0 = hash(w, 2) * 6.28;
    for (let k = 0; k < n; k++) {
      const seed = w * 17 + k;
      if (hash(seed, 13) < 0.12) continue;
      const th = th0 + (k / n) * Math.PI * 2 + (hash(seed, 3) - 0.5) * 0.6;
      e.copy(SIDE).multiplyScalar(Math.cos(th)).addScaledVector(UP2, Math.sin(th));
      start.set(cx, cy, cz).addScaledVector(e, r * 0.86).addScaledVector(AX, (hash(seed, 4) - 0.5) * 0.05);
      // dead spruce branches leave the trunk nearly square, a little toward the old top
      const tilt = (8 + 22 * hash(seed, 5)) * (Math.PI / 180);
      dir.copy(e).multiplyScalar(Math.cos(tilt)).addScaledVector(AX, Math.sin(tilt)).normalize();
      const L = 0.035 + 0.1 * Math.pow(hash(seed, 7), 1.4);
      const rb = (0.026 + 0.026 * hash(seed, 9)) * (1.15 - 0.5 * along);
      const tip = start.clone().addScaledVector(dir, L + r * 0.14);
      if (tip.y < groundY(tip.x, tip.z) - 0.03) continue;
      const pts = [start.clone().addScaledVector(dir, -r * 0.1), start.clone().addScaledVector(dir, L * 0.4 + r * 0.04), tip];
      tube(out, pts, [rb * 1.4, rb * 1.04, rb * 0.92], 6, [0, 0.35, along, hash(seed, 10)], rb * 0.9);
    }
    s += 0.4 + 0.22 * hash(w, 12);
    w++;
  }
}

/** a crooked, tapering dead twig off `at`, with a side twig now and then (grey, aSpr.y 1) */
function twig(out: Piece, at: THREE.Vector3, dir: THREE.Vector3, L: number, r0: number, seed: number, along: number) {
  const pts = crooked(at, dir, L, 4, 0.6, seed, 0.07);
  tube(out, pts, pts.map((_, i) => r0 * (1 - 0.65 * (i / (pts.length - 1)))), 3, [0, 1, along, hash(seed, 1)], r0 * 0.4);
  if (hash(seed, 2) < 0.35) {
    const d2 = new THREE.Vector3(hash(seed, 3) - 0.5, hash(seed, 4) - 0.3, hash(seed, 5) - 0.5).normalize().add(dir).normalize();
    const p2 = crooked(pts[2], d2, L * 0.45, 3, 0.7, seed * 7 + 1, 0.08);
    tube(out, p2, p2.map((_, i) => r0 * 0.6 * (1 - 0.6 * (i / (p2.length - 1)))), 3, [0, 1, along, hash(seed, 6)], r0 * 0.3);
  }
}

/**
 * The few branches the fall left whole (critic r2: "the one long broken branch is a ruler-straight, smooth, evenly
 * coloured stick, like a dowel"): five long dead branches, grey, crooked at every knot, tapering, carrying fine kinked
 * twigs, and a handful of shorter broken ones; ends that would go into the bank or the bed stop there.
 */
function addBranches(groundY: (x: number, z: number) => number, out: Piece) {
  const LONG: Array<[number, number, number]> = [[1.9, 1.15, 1.7], [3.2, 2.5, 1.35], [4.4, 0.45, 2.0], [5.7, 1.7, 1.15], [6.9, 2.9, 0.95]];
  const list = LONG.map(([s, th, L], i) => ({ s, th, L, long: true, seed: 500 + i }));
  for (let k = 0; k < 9; k++) {
    list.push({ s: 1.2 + 7.6 * hash(k, 61), th: 0.2 + 2.7 * hash(k, 62), L: 0.25 + 0.3 * hash(k, 63), long: false, seed: 600 + k });
  }
  const e = new THREE.Vector3(), dir = new THREE.Vector3(), start = new THREE.Vector3();
  for (const B of list) {
    const [cx, cy, cz] = spruceAxis(B.s);
    const r = spruceRadius(B.s);
    const along = B.s / SPRUCE.length;
    e.copy(SIDE).multiplyScalar(Math.cos(B.th)).addScaledVector(UP2, Math.sin(B.th));
    start.set(cx, cy, cz).addScaledVector(e, r * 0.85);
    const tilt = (25 + 20 * hash(B.seed, 1)) * (Math.PI / 180);
    dir.copy(e).multiplyScalar(Math.cos(tilt)).addScaledVector(AX, Math.sin(tilt)).normalize();
    const n = B.long ? 10 : 5;
    let pts = crooked(start, dir, B.L, n, B.long ? 0.24 : 0.3, B.seed, 0.035);
    const cut = pts.findIndex((p) => p.y < groundY(p.x, p.z) - 0.02);
    if (cut >= 0) pts = pts.slice(0, Math.max(2, cut));
    if (pts.length < 2) continue;
    const rb = (B.long ? 0.04 + 0.012 * hash(B.seed, 2) : 0.025 + 0.012 * hash(B.seed, 2)) * (1.1 - 0.35 * along);
    tube(out, pts, pts.map((_, i) => rb * (1 - 0.8 * Math.pow(i / (n), 0.85))), 6, [0, 1, along, hash(B.seed, 3)], rb * 0.5);
    const nt = B.long ? pts.length - 2 : 1;
    for (let j = 0; j < nt; j++) {
      const sd = B.seed * 13 + j;
      if (hash(sd, 4) > 0.75) continue;
      const at = pts[Math.min(pts.length - 1, 2 + j)];
      const td = new THREE.Vector3(hash(sd, 5) - 0.5, hash(sd, 6) - 0.35, hash(sd, 7) - 0.5).normalize().addScaledVector(dir, 0.8).normalize();
      const tl = (0.08 + 0.24 * hash(sd, 8)) * (1 - 0.4 * (j / Math.max(1, nt)));
      const tip = at.clone().addScaledVector(td, tl);
      if (tip.y < groundY(tip.x, tip.z)) continue;
      twig(out, at, td, tl, 0.008 * (1 - 0.3 * (j / Math.max(1, nt))), sd, along);
    }
  }
}

/**
 * The root plate's caught stones and torn rootlets: stones of 10-35 cm pressed into the soil of its face and rim,
 * thin roots hanging from its lower rim, soil clods along the bottom edge.
 */
function addPlateDetail(groundY: (x: number, z: number) => number, out: Piece) {
  const side = new THREE.Vector3(1, 0, 0).applyQuaternion(Q_TREE), up2 = new THREE.Vector3().crossVectors(AX, side).normalize();
  const base = new THREE.Vector3(...SPRUCE.base);
  const [rx, ry] = SPRUCE.plate;
  for (let k = 0; k < 14; k++) {
    const th = hash(k, 81) * 6.28, rr = 0.45 + 0.5 * Math.sqrt(hash(k, 82));
    const at = base.clone().addScaledVector(side, Math.cos(th) * rx * rr).addScaledVector(up2, Math.sin(th) * ry * rr).addScaledVector(AX, -0.2 - 0.25 * hash(k, 83));
    const s = 0.06 + 0.12 * hash(k, 84);
    let g: THREE.BufferGeometry = new THREE.IcosahedronGeometry(1, 1);
    g.deleteAttribute('normal');
    g.deleteAttribute('uv');
    g = mergeVertices(g);
    const p = g.getAttribute('position') as THREE.BufferAttribute;
    for (let i = 0; i < p.count; i++) {
      const f = 0.8 + 0.4 * hash(Math.round(p.getX(i) * 50) + k, Math.round(p.getY(i) * 50) + Math.round(p.getZ(i) * 50) * 7);
      p.setXYZ(i, p.getX(i) * f * s * 1.3, p.getY(i) * f * s * 0.8, p.getZ(i) * f * s);
    }
    g.rotateX(hash(k, 85) * 3).rotateY(hash(k, 86) * 3).translate(at.x, at.y, at.z);
    g.computeVertexNormals();
    const P = g.getAttribute('position'), Nn = g.getAttribute('normal');
    const o0 = out.pos.length / 3;
    for (let i = 0; i < P.count; i++) {
      out.pos.push(P.getX(i), P.getY(i), P.getZ(i));
      out.nrm.push(Nn.getX(i), Nn.getY(i), Nn.getZ(i));
      out.uv.push(P.getX(i) * 2, P.getY(i) * 2);
      out.spr.push(2, 0, 0.3, hash(k, 87));
    }
    const idx = g.index!.array;
    for (let t = 0; t < idx.length; t++) out.idx.push(o0 + idx[t]);
  }
  // rootlets hanging from the lower rim
  const dn = new THREE.Vector3(0, -1, 0);
  for (let k = 0; k < 26; k++) {
    const th = Math.PI * (1.15 + 0.7 * hash(k, 88));
    const at = base.clone().addScaledVector(side, Math.cos(th) * rx * (0.8 + 0.25 * hash(k, 89))).addScaledVector(up2, Math.sin(th) * ry * (0.8 + 0.25 * hash(k, 89)))
      .addScaledVector(AX, -0.15 - 0.4 * hash(k, 90));
    if (at.y < groundY(at.x, at.z) + 0.05) continue;
    const pts = crooked(at, dn.clone().addScaledVector(AX, 0.3), 0.15 + 0.4 * hash(k, 91), 4, 0.5, 300 + k, 0.1);
    tube(out, pts, pts.map((_, i) => (0.012 + 0.012 * hash(k, 92)) * (1 - 0.6 * (i / (pts.length - 1)))), 3, [1, 1, 0.5, hash(k, 93)], 0.006);
  }
}

/**
 * The spruce's bark mesh laid down: the trunk thickened, its top broken off in a splintered cone. Its ez-tree branch
 * tubes are dropped (round 3: they read as straight dowels and even straws); addStubs and addBranches grow what
 * the fall left instead.
 */
function layTrunk(g: THREE.BufferGeometry, A: ReturnType<typeof analyse>, out: Piece) {
  const P = g.getAttribute('position'), Nr = g.getAttribute('normal'), U = g.getAttribute('uv');
  const idx = g.index!.array;
  const n = P.count;
  const H = SPRUCE.length / SPRUCE.scale;
  const M = new THREE.Matrix4().compose(new THREE.Vector3(...SPRUCE.base), Q_TREE, new THREE.Vector3(SPRUCE.scale, SPRUCE.scale, SPRUCE.scale));
  const NM = new THREE.Matrix3().getNormalMatrix(M);
  const v = new THREE.Vector3(), nv = new THREE.Vector3();
  const remap = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    const c = A.comp[i];
    let x = P.getX(i), y = P.getY(i), z = P.getZ(i);
    const isTrunk = c === A.trunk;
    if (isTrunk) {
      if (y > H + 1.6) continue;
      const az = Math.atan2(z, x);
      let k = GIRTH;
      if (y > H) {
        // splintered break: what is above collapses onto a jagged cone
        const jag = 0.12 + 0.14 * Math.sin(az * 3 + 1.3) + 0.09 * Math.sin(az * 7 + 0.4) + 0.06 * Math.sin(az * 13);
        k *= 0.25 + 0.5 * (0.5 + 0.5 * Math.sin(az * 5 + 2.1));
        y = H + Math.max(0.02, jag);
      }
      v.set(x * k, y, z * k);
    } else continue;
    v.applyMatrix4(M);
    nv.set(Nr.getX(i), Nr.getY(i), Nr.getZ(i)).applyMatrix3(NM).normalize();
    remap[i] = out.pos.length / 3;
    out.pos.push(v.x, v.y, v.z);
    out.nrm.push(nv.x, nv.y, nv.z);
    out.uv.push(U.getX(i), U.getY(i));
    // x: 0 bark, 1 roots and soil; y: 1 on thin pieces (branches, roots); z: 0 at the butt .. 1 at the break; w: per piece
    out.spr.push(0, isTrunk ? 0 : 1, Math.min(1, Math.max(0, P.getY(i) / H)), hash(c, 7));
  }
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t], b = idx[t + 1], c = idx[t + 2];
    if (remap[a] < 0 || remap[b] < 0 || remap[c] < 0) continue;
    out.idx.push(remap[a], remap[b], remap[c]);
  }
}

/** the stump scan turned onto its side as the root collar: its top inside the butt, its root spread behind it */
function addCollar(g: THREE.BufferGeometry, roll: number, scale: [number, number, number], out: Piece) {
  const q = new THREE.Quaternion().setFromUnitVectors(UP, AX).multiply(new THREE.Quaternion().setFromAxisAngle(UP, roll));
  const at = new THREE.Vector3(...SPRUCE.base).addScaledVector(AX, -(0.573 * scale[1] - 0.3));
  const geo = g.clone().applyMatrix4(new THREE.Matrix4().compose(at, q, new THREE.Vector3(...scale)));
  const P = geo.getAttribute('position'), Nr = geo.getAttribute('normal'), U = geo.getAttribute('uv');
  const o0 = out.pos.length / 3;
  for (let i = 0; i < P.count; i++) {
    out.pos.push(P.getX(i), P.getY(i), P.getZ(i));
    out.nrm.push(Nr.getX(i), Nr.getY(i), Nr.getZ(i));
    out.uv.push(U.getX(i), U.getY(i));
    out.spr.push(1, 0, 0.35, 0.3);
  }
  const idx = geo.index!.array;
  for (let t = 0; t < idx.length; t++) out.idx.push(o0 + idx[t]);
}

/**
 * The torn-out root plate: a lumpy disc of soil standing up behind the butt, ragged at the rim, its face toward
 * the river the old forest floor (litter and moss), its back the bare soil the bank let go of.
 */
function addSoil(out: Piece) {
  const g = new THREE.SphereGeometry(1, 30, 14);
  const P = g.getAttribute('position'), U = g.getAttribute('uv');
  const [rx, ry] = SPRUCE.plate;
  const s = Math.sin;
  const front: boolean[] = [], rims: number[] = [];
  for (let i = 0; i < P.count; i++) {
    const x = P.getX(i), y = P.getY(i), z = P.getZ(i);
    front.push(y > 0);
    rims.push(Math.hypot(x, z));
    const az = Math.atan2(z, x), rim = Math.hypot(x, z);
    // torn: big lobes and small rips round the rim
    const ragged = 1 + 0.16 * s(az * 3 + 0.7) + 0.1 * s(az * 7 + 2.2) + 0.06 * s(az * 13 + 1.1) + 0.05 * s(az * 23 + 0.4) * rim;
    // the forest floor peeled up with it toward the river, a thick wedge of soil behind
    const thick = y > 0 ? 0.32 + 0.12 * s(az * 5 + 0.3) : 0.55 + 0.2 * s(az * 4 + 1.9);
    const lump = 1 + 0.16 * s(az * 9 + y * 5) * rim + 0.06 * s(az * 17 + y * 11);
    P.setXYZ(i, x * rx * ragged * lump, y * thick, z * ry * ragged * lump);
  }
  g.computeVertexNormals();
  const at = new THREE.Vector3(...SPRUCE.base).addScaledVector(AX, -0.42);
  g.applyMatrix4(new THREE.Matrix4().compose(at, Q_TREE, new THREE.Vector3(1, 1, 1)));
  const Nr = g.getAttribute('normal');
  const o0 = out.pos.length / 3;
  for (let i = 0; i < P.count; i++) {
    out.pos.push(P.getX(i), P.getY(i), P.getZ(i));
    out.nrm.push(Nr.getX(i), Nr.getY(i), Nr.getZ(i));
    out.uv.push(U.getX(i) * 6, U.getY(i) * 2.5);
    // (w: 0.55 + 0.45 x how far out toward the torn rim, where the mineral soil shows)
    out.spr.push(1, 0, front[i] ? 0 : 1, 0.55 + 0.45 * rims[i]);
  }
  const idx = g.index!.array;
  for (let t = 0; t < idx.length; t++) out.idx.push(o0 + idx[t]);
}

/**
 * Roots torn out with the plate: thick buttress roots spreading from the butt across its face, half sunk in it, and
 * roots out of its rim, crooked and tapering, bent back toward the bank and down (they grew down and out); those
 * that would go into the bank end there. Grown here (round 3: copies of the spruce's straight branch tubes read as
 * "a dark smudge of sticks").
 */
function addRoots(groundY: (x: number, z: number) => number, out: Piece) {
  const COUNT = 32;
  const base = new THREE.Vector3(...SPRUCE.base);
  const [rx, ry] = SPRUCE.plate;
  const e = new THREE.Vector3(), dir = new THREE.Vector3();
  for (let k = 0; k < COUNT; k++) {
    const th = (k / COUNT) * Math.PI * 2 + (hash(k, 32) - 0.5) * 0.4;
    e.copy(SIDE).multiplyScalar(Math.cos(th) * rx).addScaledVector(UP2, Math.sin(th) * ry);
    const rimR = e.length();
    e.normalize();
    const face = hash(k, 33) < 0.38;
    const start = base.clone().addScaledVector(e, face ? 0.3 + 0.2 * hash(k, 34) : rimR * (0.78 + 0.14 * hash(k, 34)))
      .addScaledVector(AX, face ? -0.1 : -0.3 - 0.2 * hash(k, 35));
    if (start.y < groundY(start.x, start.z) + 0.03) continue;
    const tilt = face ? 0.12 : 0.35 + 0.45 * hash(k, 36);
    dir.copy(e).multiplyScalar(Math.cos(tilt)).addScaledVector(AX, -Math.sin(tilt)).normalize();
    const L = face ? 0.4 + 0.5 * hash(k, 38) : 0.45 + 0.85 * hash(k, 38);
    const n = face ? 5 : 7;
    let pts = crooked(start, dir, L, n, face ? 0.18 : 0.32, 700 + k, face ? 0 : 0.06);
    const cut = pts.findIndex((p) => p.y < groundY(p.x, p.z) - 0.05);
    if (cut >= 0) pts = pts.slice(0, Math.max(2, cut));
    if (pts.length < 2) continue;
    const r0 = face ? 0.055 + 0.05 * hash(k, 37) : 0.03 + 0.03 * hash(k, 37);
    tube(out, pts, pts.map((_, i) => r0 * (1 - 0.75 * Math.pow(i / n, 0.8))), 5, [1, 1, 0.5, hash(k, 39)], r0 * 0.5);
  }
}

const LUM = vec3(0.2126, 0.7152, 0.0722);

/** calibration (linear albedo targets: the forest's spruce bark ~0.075, bleached driftwood ~0.3) */
export const spruceU = {
  barkGain: uniform(1.25),
  /** peeled, weathered wood and the dead branches: grey (2.6 read as bleached white rods in the low sun) */
  greyGain: uniform(1.7),
  rootGain: uniform(1.0),
  /** extra loss with depth under the surface (1/m, per channel): the submerged trunk sinks into the green */
  murk: uniform(new THREE.Vector3(3.2, 1.5, 2.2)),
};

function spruceMaterial(
  bark: { c: THREE.Texture; n: THREE.Texture; m: THREE.Texture },
  root: { c: THREE.Texture; n: THREE.Texture; m: THREE.Texture; flipY: boolean },
  field: RiverField | null,
) {
  const m = new THREE.MeshStandardNodeMaterial();
  m.name = 'fallen-spruce';
  const t = uv();
  const s = attribute('aSpr', 'vec4');
  const isStone = smoothstep(1.5, 2.0, s.x);
  const isRoot = s.x.clamp(0, 1).mul(isStone.oneMinus()), thin = s.y, along = s.z;
  const p = positionWorld;
  const pw = vec2(p.x.add(p.z.mul(0.6)), p.y.add(p.z.mul(0.55)));
  const n1 = tnoise(pw.mul(0.9), 3.1);
  const n2 = tnoise(pw.mul(4.3), 5.7);
  const nUp = normalWorldGeometry.y;
  // bark: scaly grey-brown (the map's scales at more contrast), peeled to silver wood over the sun-bleached top
  // and on every dead branch, the wood split in long dark checks along the trunk
  const bc = texture(bark.c, t).rgb;
  const lb = dot(bc, LUM);
  const barkCol = mix(vec3(lb), bc, 1.1).mul(lb.div(0.075).max(0.05).pow(0.45)).mul(spruceU.barkGain).mul(vec3(1.0, 0.95, 0.9));
  // (the bark map's fissures at full contrast made the grey wood blotchy, like birch)
  const grey = vec3(mix(lb, float(0.057), 0.45)).mul(spruceU.greyGain).mul(vec3(1.0, 0.97, 0.93)).add(0.006);
  // dead branches and twigs (aSpr.y 1) grey; the snapped stubs (0.35) dark with old bark
  const branchy = smoothstep(0.6, 0.95, thin);
  const stubby = smoothstep(0.15, 0.4, thin).mul(branchy.oneMinus());
  // (scene fix, critic polish r1: "a smooth dark pipe": more of the bark peeled off the dry top in long patches)
  const peel = max(smoothstep(0.42, 0.6, n1.add(n2.mul(0.2))).mul(smoothstep(-0.2, 0.5, nUp)), branchy.mul(0.85));
  // checks: cylindrical coordinates about the trunk's axis; long thin lines across the angle, slow along the trunk
  const rel = p.sub(vec3(...SPRUCE.base));
  const sAx = dot(rel, vec3(AX.x, AX.y, AX.z));
  const th = atan(dot(rel, vec3(UP2.x, UP2.y, UP2.z)), dot(rel, vec3(SIDE.x, SIDE.y, SIDE.z)));
  const chk = smoothstep(0.66, 0.71, tnoise(vec2(th.mul(3.2), sAx.mul(0.4)), 8.1)).max(smoothstep(0.69, 0.73, tnoise(vec2(th.mul(5.1), sAx.mul(0.7)), 9.3)));
  let col: N = mix(barkCol, grey.mul(float(1).sub(chk.mul(0.75).mul(thin.oneMinus()))), peel);
  col = mix(col, barkCol.mul(vec3(0.5, 0.46, 0.43)), stubby.mul(0.85));
  // the root plate: roots and soil in browns and clay greys; its face toward the river litter and moss
  const rc = texture(root.c, t).rgb;
  let rootCol: N = mix(vec3(dot(rc, LUM)), rc, 0.55).mul(spruceU.rootGain).mul(vec3(1.0, 0.88, 0.74)).mul(mix(0.65, 1.35, n2));
  rootCol = mix(rootCol, vec3(dot(rc, LUM)).mul(spruceU.rootGain.mul(1.3)).mul(vec3(0.95, 0.93, 0.88)), smoothstep(0.55, 0.75, n1).mul(thin.oneMinus()).mul(0.6));
  // (round 3: the plate's face in patches of old litter and moss, not one dark smudge; its torn rim lighter mineral soil)
  const litter = smoothstep(0.15, 0.05, along).mul(thin.oneMinus()).mul(smoothstep(0.4, 0.62, n1));
  rootCol = mix(rootCol, mix(vec3(0.045, 0.036, 0.022), vec3(0.06, 0.075, 0.02), smoothstep(0.45, 0.7, n1)), litter.mul(0.6));
  const rimA = smoothstep(0.78, 0.97, s.w).mul(thin.oneMinus());
  rootCol = mix(rootCol, vec3(0.15, 0.118, 0.082).mul(mix(0.8, 1.15, n2)), rimA.mul(0.75));
  // roots out in the air for years weather grey-brown
  rootCol = mix(rootCol, vec3(dot(rootCol, LUM)).mul(1.35).mul(vec3(1.0, 0.96, 0.9)), thin.mul(0.55));
  col = mix(col, rootCol, isRoot);
  // stones caught in the plate: grey granite, dirty
  col = mix(col, vec3(0.11, 0.108, 0.1).mul(mix(0.65, 1.3, n2)).mul(mix(0.8, 1.2, s.w)), isStone);
  // moss in patches on the upper faces near the bank (shade, damp), none out on the water
  // (scene fix: moss in patches along the whole dry top, thickest near the bank)
  const mossA = smoothstep(0.5, 0.85, nUp.add(n1.sub(0.5).mul(0.9))).mul(smoothstep(0.45, 0.65, n2)).mul(float(1).sub(smoothstep(0.05, 0.75, along)).mul(0.7).add(0.3))
    .mul(thin.oneMinus()).mul(isRoot.oneMinus()).mul(smoothstep(0.25, 0.6, p.y.sub(POOL_LEVEL))).mul(0.85);
  col = mix(col, mix(vec3(0.03, 0.042, 0.011), vec3(0.075, 0.1, 0.022), n2), mossA);
  const ba = texture(bark.m, t), ra = texture(root.m, t);
  const arm = mix(ba, ra, isRoot);
  let rough: N = mix(float(0.72), float(1.0), arm.y);
  // waterline: soaked dark a hand's width and more above the surface, higher on the underside (it wicks up),
  // glossy; under it an olive film, and the deeper it lies the more it turns green and sinks into the pool
  const h = p.y.sub(POOL_LEVEL);
  const band = float(0.16).add(n2.sub(0.5).mul(0.1)).add(smoothstep(0.3, -0.7, nUp).mul(0.25));
  const wet = smoothstep(band.add(0.03), band.sub(0.03), h);
  const lc = dot(col, LUM);
  col = mix(col, mix(vec3(lc), col, 1.25).mul(0.3), wet);
  // (thin branches and roots stay duller: glossy, a wet branch dipping toward the low sun glinted white)
  rough = mix(rough, mix(mix(float(0.16), float(0.3), n2), float(0.55), thin), wet);
  const under = smoothstep(0.0, -0.06, h);
  col = mix(col, col.mul(vec3(0.5, 0.68, 0.38)), under);
  const depth = h.negate().max(0);
  col = col.mul(exp(spruceU.murk.mul(depth).negate()));
  // in-scattered green: within half a metre the trunk is mostly the pool's own colour
  col = mix(col, vec3(0.008, 0.022, 0.018), float(1).sub(exp(depth.mul(-3.2))).mul(0.9));
  // no bright sky edge on what lies under the surface
  rough = mix(rough, float(0.85), under);
  m.colorNode = col;
  m.roughnessNode = rough;
  m.metalnessNode = float(0);
  m.aoNode = mix(float(1), arm.x, 0.9).mul(mix(float(1), float(0.7), wet.mul(0.5)));
  // (GLTFLoader gives the stump's map normalScale.y -1 for derivative tangents; the tree GLBs' maps need none)
  const rn = texture(root.n, t);
  const rootN = root.flipY ? vec3(rn.x, rn.y.oneMinus(), rn.z) : rn.xyz;
  m.normalNode = normalMap(mix(texture(bark.n, t).xyz, rootN, isRoot), uniform(new THREE.Vector2(2.1, 2.1)));
  setOpaqueAlbedo(m, m.colorNode);
  // caustics, beam absorption, refracted shadows and sky light fading with depth on what lies under the water
  if (field) installUnderwater(m, { field, surfaceY: float(POOL_LEVEL) });
  return m;
}

/** Build the fallen spruce (one mesh) on the rendered ground. */
export async function buildFallenSpruce(assets: Assets, groundY: (x: number, z: number) => number, field: RiverField | null): Promise<THREE.Mesh> {
  const T = (f: string, srgb: boolean) => assets.texture(`trees/textures/${f}.webp`, srgb, true);
  const [bytes, stumpGltf, bc, bn, bm] = await Promise.all([
    fetch(`${assets.base}trees/spruce_a_lod0.glb`).then((r) => {
      if (!r.ok) throw new Error(`${r.status} trees/spruce_a_lod0.glb`);
      return r.arrayBuffer();
    }),
    assets.gltf('models/logs/tree_stump_01.glb'),
    T('bark_fir_c', true), T('bark_fir_n', false), T('bark_fir_m', false),
  ]);
  const barkGeo = parseGlb(bytes).find((q) => q.material.startsWith('bark'))!.geometry;
  const lods: THREE.Mesh[] = [];
  stumpGltf.scene.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) lods.push(o as THREE.Mesh);
  });
  const isLod0 = (mm: THREE.Mesh) => mm.name.includes('LOD0') || !!mm.parent?.name.includes('LOD0');
  const s0 = lods.find(isLod0) ?? lods[0];
  const sm = s0.material as THREE.MeshStandardMaterial;
  const piece: Piece = { pos: [], nrm: [], uv: [], spr: [], idx: [] };
  const A = analyse(barkGeo);
  layTrunk(barkGeo, A, piece);
  addCollar(bakeGeometry(s0), 0.4, [1.5, 1.2, 1.45], piece);
  addSoil(piece);
  addRoots(groundY, piece);
  addStubs(groundY, piece);
  addBranches(groundY, piece);
  addPlateDetail(groundY, piece);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(piece.pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(piece.nrm, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(piece.uv, 2));
  g.setAttribute('aSpr', new THREE.Float32BufferAttribute(piece.spr, 4));
  g.setIndex(piece.idx);
  g.computeBoundingSphere();
  const mat = spruceMaterial(
    { c: bc, n: bn, m: bm },
    { c: sm.map!, n: sm.normalMap!, m: (sm.roughnessMap ?? sm.metalnessMap ?? sm.aoMap)!, flipY: sm.normalScale.y < 0 },
    field,
  );
  const mesh = new THREE.Mesh(g, mat);
  mesh.name = 'fallen-spruce';
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}
