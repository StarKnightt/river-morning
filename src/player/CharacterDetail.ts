import * as THREE from 'three/webgpu';
import { LAYER_FOLIAGE } from '../core/layers';

/**
 * Load-time analysis of the angler GLB (no new assets): rest-pose positions by CPU skinning, UV islands, and
 * per-vertex attributes the materials read (garment regions and coat parts, seam and hem distances, crease
 * coordinates at the elbows and knees, wear and dust masks, cavity, face and hand zones, the beard's depth and the
 * shadow it throws onto the neck and collar). Bind-space edits shape the clothes: natural sloping shoulders and
 * slack sleeves, a lumpy wool outline, an uneven coat hem with sagging pockets, a back vent and rolled lapels,
 * trousers that crumple over the boots, a neckerchief with volume, and a hat brim with thickness and a wave.
 * Facial hair the GLB lacks (moustache, beard fringe, loose hair) is built as strand cards on the head bone.
 *
 * Landmarks below are in the character's rest space (metres, +Z forward, origin between the soles), measured on
 * the delivered mesh (prep/character/glb/angler_nomeshopt.glb).
 */
export const LANDMARKS = {
  noseTip: [0, 1.668, 0.175],
  eyeL: [0.0334, 1.7093, 0.1432],
  eyeR: [-0.0334, 1.7093, 0.1432],
  mouth: [0, 1.632, 0.16],
  /** brim ellipse in head-bone space (rest head at (0, 1.669, 0.044), axes = character axes) */
  brim: { y: 0.078, cz: 0.016, ax: 0.182, az: 0.2 },
} as const;

/** Coat parts (aCloth.x), from each UV island's size and place. */
export const COAT_PART = { sleeve: 1, back: 2, front: 3, lapel: 4, collar: 5, flap: 6 } as const;

/**
 * Attach float attributes as one interleaved buffer: WebGPU allows 8 vertex buffers per pipeline and the skinned
 * mesh already uses 5 (position, normal, uv, skin index and weight).
 */
function packAttributes(g: THREE.BufferGeometry, count: number, attrs: [string, Float32Array, number][]) {
  const stride = attrs.reduce((s, a) => s + a[2], 0);
  const data = new Float32Array(count * stride);
  let off = 0;
  for (const [, arr, size] of attrs) {
    for (let i = 0; i < count; i++) for (let k = 0; k < size; k++) data[i * stride + off + k] = arr[i * size + k];
    off += size;
  }
  const ib = new THREE.InterleavedBuffer(data, stride);
  off = 0;
  for (const [name, , size] of attrs) {
    g.setAttribute(name, new THREE.InterleavedBufferAttribute(ib, size, off));
    off += size;
  }
}

/** Rest-pose (bind) vertex positions and normals in character space. */
export interface Rest {
  pos: Float32Array;
  nrm: Float32Array;
  uv: Float32Array;
  index: Uint32Array;
  /** dominant bone name per vertex */
  bone: string[];
  count: number;
}

export function restOf(mesh: THREE.SkinnedMesh): Rest {
  const g = mesh.geometry;
  const pos = g.getAttribute('position'), nrm = g.getAttribute('normal'), uvA = g.getAttribute('uv');
  const si = g.getAttribute('skinIndex'), sw = g.getAttribute('skinWeight');
  const sk = mesh.skeleton;
  const mats = sk.bones.map((b, i) => new THREE.Matrix4().multiplyMatrices(b.matrixWorld, sk.boneInverses[i]).multiply(mesh.bindMatrix));
  const n = pos.count;
  const P = new Float32Array(n * 3), Nn = new Float32Array(n * 3), U = new Float32Array(n * 2);
  const bone: string[] = new Array(n);
  const e = new Float32Array(16);
  for (let i = 0; i < n; i++) {
    e.fill(0);
    let best = -1, bw = -1;
    for (let k = 0; k < 4; k++) {
      const w = sw.getComponent(i, k);
      if (w <= 0) continue;
      const bi = si.getComponent(i, k);
      const m = mats[bi].elements;
      for (let q = 0; q < 16; q++) e[q] += m[q] * w;
      if (w > bw) { bw = w; best = bi; }
    }
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    P[i * 3] = e[0] * x + e[4] * y + e[8] * z + e[12];
    P[i * 3 + 1] = e[1] * x + e[5] * y + e[9] * z + e[13];
    P[i * 3 + 2] = e[2] * x + e[6] * y + e[10] * z + e[14];
    const nx = nrm.getX(i), ny = nrm.getY(i), nz = nrm.getZ(i);
    let ox = e[0] * nx + e[4] * ny + e[8] * nz, oy = e[1] * nx + e[5] * ny + e[9] * nz, oz = e[2] * nx + e[6] * ny + e[10] * nz;
    const l = Math.hypot(ox, oy, oz) || 1;
    ox /= l; oy /= l; oz /= l;
    Nn[i * 3] = ox; Nn[i * 3 + 1] = oy; Nn[i * 3 + 2] = oz;
    U[i * 2] = uvA.getX(i);
    U[i * 2 + 1] = uvA.getY(i);
    bone[i] = best >= 0 ? sk.bones[best].name : '';
  }
  const idx = g.index ? Uint32Array.from(g.index.array as ArrayLike<number>) : Uint32Array.from({ length: n }, (_, i) => i);
  return { pos: P, nrm: Nn, uv: U, index: idx, bone, count: n };
}

/** Connected components of the index buffer (UV islands: a seam splits vertices). */
function islands(r: Rest): Int32Array {
  const par = new Int32Array(r.count);
  for (let i = 0; i < r.count; i++) par[i] = i;
  const find = (x: number): number => {
    while (par[x] !== x) {
      par[x] = par[par[x]];
      x = par[x];
    }
    return x;
  };
  const I = r.index;
  for (let t = 0; t < I.length; t += 3) {
    const a = find(I[t]), b = find(I[t + 1]), c = find(I[t + 2]);
    par[b] = a;
    par[find(c)] = a;
  }
  const out = new Int32Array(r.count);
  for (let i = 0; i < r.count; i++) out[i] = find(i);
  return out;
}

/** Vertex neighbours from the triangles, plus vertices welded across UV seams (same rest position). */
function neighbours(r: Rest, weld = true): number[][] {
  const nb: Set<number>[] = Array.from({ length: r.count }, () => new Set<number>());
  const I = r.index;
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t], b = I[t + 1], c = I[t + 2];
    nb[a].add(b); nb[a].add(c); nb[b].add(a); nb[b].add(c); nb[c].add(a); nb[c].add(b);
  }
  if (weld) {
    const twins = weldTwins(r, 0.0015);
    for (let i = 0; i < r.count; i++) for (const j of twins[i]) for (const k of nb[j]) if (k !== i) nb[i].add(k);
  }
  return nb.map((s) => [...s]);
}

/** Other vertices within `eps` of each vertex (seam duplicates). */
function weldTwins(r: Rest, eps: number): number[][] {
  // hashed grid cells (a collision only costs a distance test)
  const key = (x: number, y: number, z: number) => ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) | 0;
  const cell = new Map<number, number[]>();
  const P = r.pos;
  const ci = new Int32Array(r.count * 3);
  for (let i = 0; i < r.count; i++) {
    const x = Math.floor(P[i * 3] / eps), y = Math.floor(P[i * 3 + 1] / eps), z = Math.floor(P[i * 3 + 2] / eps);
    ci[i * 3] = x; ci[i * 3 + 1] = y; ci[i * 3 + 2] = z;
    const k = key(x, y, z);
    let a = cell.get(k);
    if (!a) cell.set(k, (a = []));
    a.push(i);
  }
  const out: number[][] = Array.from({ length: r.count }, () => []);
  const e2 = eps * eps;
  for (let i = 0; i < r.count; i++) {
    const cx = ci[i * 3], cy = ci[i * 3 + 1], cz = ci[i * 3 + 2];
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const a = cell.get(key(cx + dx, cy + dy, cz + dz));
      if (!a) continue;
      for (const j of a) {
        if (j === i) continue;
        const ddx = P[j * 3] - P[i * 3], ddy = P[j * 3 + 1] - P[i * 3 + 1], ddz = P[j * 3 + 2] - P[i * 3 + 2];
        if (ddx * ddx + ddy * ddy + ddz * ddz < e2 && !out[i].includes(j)) out[i].push(j);
      }
    }
  }
  return out;
}

/** Nearest-point queries on a point set (rest metres), hashed in cubic cells (size them ~half the search cap). */
class PointGrid {
  private readonly cells = new Map<number, number[]>();
  constructor(private readonly P: Float32Array, private readonly cell: number) {
    for (let i = 0; i < P.length / 3; i++) {
      const k = this.key(Math.floor(P[i * 3] / cell), Math.floor(P[i * 3 + 1] / cell), Math.floor(P[i * 3 + 2] / cell));
      let a = this.cells.get(k);
      if (!a) this.cells.set(k, (a = []));
      a.push(i);
    }
  }
  private key(x: number, y: number, z: number) {
    return ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) | 0;
  }
  /** distance to the nearest point, capped at `cap` (searches cells within the cap) */
  nearest(x: number, y: number, z: number, cap: number): number {
    const c = this.cell, P = this.P, rr = Math.ceil(cap / c);
    const cx = Math.floor(x / c), cy = Math.floor(y / c), cz = Math.floor(z / c);
    let best = cap * cap;
    for (let dx = -rr; dx <= rr; dx++) for (let dy = -rr; dy <= rr; dy++) for (let dz = -rr; dz <= rr; dz++) {
      const a = this.cells.get(this.key(cx + dx, cy + dy, cz + dz));
      if (!a) continue;
      for (const j of a) {
        const ex = P[j * 3] - x, ey = P[j * 3 + 1] - y, ez = P[j * 3 + 2] - z;
        const d2 = ex * ex + ey * ey + ez * ez;
        if (d2 < best) best = d2;
      }
    }
    return Math.sqrt(best);
  }
}

/**
 * Cavity from the rest mesh: how far the neighbours rise above each vertex's tangent plane (concave > 0), two
 * smoothing passes. Folds, eye sockets, the gaps between fingers and the creases of the boots read darker.
 */
function cavity(r: Rest, nb: number[][], scale: number): Float32Array {
  const P = r.pos, N = r.nrm;
  let c = new Float32Array(r.count);
  for (let i = 0; i < r.count; i++) {
    let s = 0, w = 0;
    for (const j of nb[i]) {
      const dx = P[j * 3] - P[i * 3], dy = P[j * 3 + 1] - P[i * 3 + 1], dz = P[j * 3 + 2] - P[i * 3 + 2];
      const l2 = dx * dx + dy * dy + dz * dz;
      if (l2 < 1e-10) continue;
      s += (dx * N[i * 3] + dy * N[i * 3 + 1] + dz * N[i * 3 + 2]) / l2;
      w++;
    }
    c[i] = w ? s / w : 0;
  }
  for (let pass = 0; pass < 2; pass++) {
    const o = new Float32Array(r.count);
    for (let i = 0; i < r.count; i++) {
      let s = c[i] * 2, w = 2;
      for (const j of nb[i]) { s += c[j]; w++; }
      o[i] = s / w;
    }
    c = o;
  }
  for (let i = 0; i < r.count; i++) c[i] = Math.max(-1, Math.min(1, c[i] * scale));
  return c;
}

/** Multi-source shortest path along mesh edges (rest metres) from the `src` vertices. */
function distanceFrom(r: Rest, nb: number[][], src: number[], cap: number): Float32Array {
  const P = r.pos;
  const d = new Float32Array(r.count).fill(cap);
  const heap: [number, number][] = [];
  const push = (v: number, k: number) => {
    heap.push([k, v]);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heap[p][0] <= heap[i][0]) break;
      [heap[p], heap[i]] = [heap[i], heap[p]];
      i = p;
    }
  };
  const pop = (): [number, number] => {
    const top = heap[0], last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1, rr = l + 1;
        let m = i;
        if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
        if (rr < heap.length && heap[rr][0] < heap[m][0]) m = rr;
        if (m === i) break;
        [heap[m], heap[i]] = [heap[i], heap[m]];
        i = m;
      }
    }
    return top;
  };
  for (const s of src) {
    d[s] = 0;
    push(s, 0);
  }
  while (heap.length) {
    const [k, v] = pop();
    if (k > d[v] || k >= cap) continue;
    for (const j of nb[v]) {
      const nk = k + Math.hypot(P[j * 3] - P[v * 3], P[j * 3 + 1] - P[v * 3 + 1], P[j * 3 + 2] - P[v * 3 + 2]);
      if (nk < d[j]) {
        d[j] = nk;
        push(j, nk);
      }
    }
  }
  return d;
}

/** Vertices on an island's open boundary (edges used by one triangle). */
function boundary(r: Rest): Uint8Array {
  const I = r.index, n = r.count;
  const count = new Map<number, number>();
  const key = (a: number, b: number) => (a < b ? a * n + b : b * n + a);
  for (let t = 0; t < I.length; t += 3) {
    for (const [a, b] of [[I[t], I[t + 1]], [I[t + 1], I[t + 2]], [I[t + 2], I[t]]]) {
      const k = key(a, b);
      count.set(k, (count.get(k) ?? 0) + 1);
    }
  }
  const out = new Uint8Array(n);
  for (const [k, c] of count) {
    if (c !== 1) continue;
    out[Math.floor(k / n)] = 1;
    out[k % n] = 1;
  }
  return out;
}

/** Luminance of a texture at each vertex's UV (glTF v down), read from a small canvas copy. */
function sampleLuma(tex: THREE.Texture | null, r: Rest): Float32Array | null {
  const img = tex?.image as CanvasImageSource | undefined;
  if (!img || typeof OffscreenCanvas === 'undefined') return null;
  const S = 128;
  const cv = new OffscreenCanvas(S, S);
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;
  try {
    ctx.drawImage(img, 0, 0, S, S);
  } catch {
    return null;
  }
  const px = ctx.getImageData(0, 0, S, S).data;
  const out = new Float32Array(r.count);
  for (let i = 0; i < r.count; i++) {
    let u = r.uv[i * 2], v = r.uv[i * 2 + 1];
    if (tex!.flipY) v = 1 - v;
    const x = Math.min(S - 1, Math.max(0, Math.floor(u * S))), y = Math.min(S - 1, Math.max(0, Math.floor(v * S)));
    const k = (y * S + x) * 4;
    out[i] = (0.2126 * px[k] + 0.7152 * px[k + 1] + 0.0722 * px[k + 2]) / 255;
  }
  return out;
}

type V3 = [number, number, number];
const sub = (a: ArrayLike<number>, b: ArrayLike<number>): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len3 = (a: ArrayLike<number>) => Math.hypot(a[0], a[1], a[2]);
const norm3 = (a: V3): V3 => {
  const l = len3(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
const cross3 = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const sstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Smooth value noise in [-1, 1] (trilinear over hashed lattice values, smoothstep-faded). */
function vnoise3(x: number, y: number, z: number): number {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const fx = x - xi, fy = y - yi, fz = z - zi;
  const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy), w = fz * fz * (3 - 2 * fz);
  const h = (i: number, j: number, k: number) => {
    let s = (Math.imul(i, 374761393) + Math.imul(j, 668265263) + Math.imul(k, 2147483647)) | 0;
    s = Math.imul(s ^ (s >>> 13), 1274126177);
    return (((s ^ (s >>> 16)) >>> 0) / 4294967296) * 2 - 1;
  };
  const l = (a: number, b: number, t: number) => a + (b - a) * t;
  return l(
    l(l(h(xi, yi, zi), h(xi + 1, yi, zi), u), l(h(xi, yi + 1, zi), h(xi + 1, yi + 1, zi), u), v),
    l(l(h(xi, yi, zi + 1), h(xi + 1, yi, zi + 1), u), l(h(xi, yi + 1, zi + 1), h(xi + 1, yi + 1, zi + 1), u), v),
    w,
  );
}

export interface Joints {
  /** rest world position of each bone's pivot */
  at(name: string): V3;
}

/**
 * Crease coordinate on a limb chain A (upper joint) - B (bending joint) - C: signed distance along the chain from
 * B (negative toward A), how far round the limb the vertex sits toward the bend's inside (+1) or outside (-1).
 */
function limbCoord(p: V3, A: V3, B: V3, C: V3, inside: V3): { t: number; side: number; radial: number } {
  const u = norm3(sub(B, A)), f = norm3(sub(C, B));
  const pa = sub(p, B);
  const tu = dot3(pa, u), tf = dot3(pa, f);
  let t: number, axis: V3;
  if (tu <= 0) {
    // above the joint: along the upper segment
    t = tu;
    axis = u;
  } else {
    t = Math.max(tf, 0);
    axis = f;
  }
  const along = t < 0 ? tu : t;
  const q: V3 = [pa[0] - axis[0] * along, pa[1] - axis[1] * along, pa[2] - axis[2] * along];
  const radial = len3(q);
  const rd = norm3(q);
  // inside direction perpendicular to this segment
  const ins = norm3([inside[0] - axis[0] * dot3(inside, axis), inside[1] - axis[1] * dot3(inside, axis), inside[2] - axis[2] * dot3(inside, axis)]);
  return { t, side: dot3(rd, ins), radial };
}

/** Where a vertex sits round a limb segment A -> B: unit radial direction, its "top" share (+1 up) and angle. */
function aroundLimb(p: V3, A: V3, B: V3): { top: number; ang: number; t: number } {
  const ax = norm3(sub(B, A));
  const pa = sub(p, A);
  const t = dot3(pa, ax);
  const rq = norm3([pa[0] - ax[0] * t, pa[1] - ax[1] * t, pa[2] - ax[2] * t]);
  const up = norm3([-ax[0] * ax[1], 1 - ax[1] * ax[1], -ax[2] * ax[1]]);
  const side = cross3(ax, up);
  return { top: dot3(rq, up), ang: Math.atan2(dot3(rq, side), dot3(rq, up)), t };
}

const isArm = (b: string) => /^(upperarm|lowerarm|hand)_/.test(b);
const isLeg = (b: string) => /^(thigh|calf|foot|ball)_/.test(b);
const sideOf = (b: string) => (b.endsWith('_l') ? 'l' : 'r');

/**
 * Lowering the arms from the bind A-pose swings a shoulder cap weighted to the upper arm outward and up (linear
 * blend skinning): round 2's domes that rose above his chin. The coat's cap above the shoulder joint follows the
 * clavicle instead, so the shoulder line stays put and falls away from the collar while the sleeve hangs from it.
 * (At rest every bone's skin matrix is the same, so rest positions do not depend on the weights.)
 */
function shoulderWeights(mesh: THREE.SkinnedMesh, r: Rest, coat: (i: number) => boolean, j: Joints) {
  const g = mesh.geometry;
  const si = g.getAttribute('skinIndex') as THREE.BufferAttribute, sw = g.getAttribute('skinWeight') as THREE.BufferAttribute;
  const bones = mesh.skeleton.bones;
  const at = (n: string) => bones.findIndex((b) => b.name === n);
  for (const s of ['l', 'r'] as const) {
    const ua = at(`upperarm_${s}`), cl = at(`clavicle_${s}`);
    if (ua < 0 || cl < 0) continue;
    const P = j.at(`upperarm_${s}`), px = Math.abs(P[0]);
    for (let i = 0; i < r.count; i++) {
      const x = r.pos[i * 3], y = r.pos[i * 3 + 1];
      if (!coat(i) || (s === 'l') !== x > 0) continue;
      // fully over the top toward the collar (it rose above his chin), fading out over the outer cap (left with the
      // clavicle it overhangs the lowered arm: the pose corrective in the material pulls that part in)
      const f = 0.85 * sstep(P[1] - 0.01, P[1] + 0.04, y) * (1 - sstep(-0.01, 0.06, Math.abs(x) - px));
      if (f < 0.01) continue;
      let kU = -1, kC = -1, kMin = 0;
      for (let k = 0; k < 4; k++) {
        const b = si.getComponent(i, k), w = sw.getComponent(i, k);
        if (b === ua && w > 0) kU = k;
        else if (b === cl) kC = k;
        if (w < sw.getComponent(i, kMin)) kMin = k;
      }
      if (kU < 0) continue;
      let wU = sw.getComponent(i, kU);
      if (kC < 0) {
        // no clavicle influence yet: take over the smallest one (its little weight joins the upper arm's)
        if (kMin === kU) continue;
        wU += sw.getComponent(i, kMin);
        si.setComponent(i, kMin, cl);
        sw.setComponent(i, kMin, 0);
        kC = kMin;
      }
      const move = wU * f;
      sw.setComponent(i, kU, wU - move);
      sw.setComponent(i, kC, sw.getComponent(i, kC) + move);
    }
  }
  si.needsUpdate = true;
  sw.needsUpdate = true;
}

/** Beard vertices of the gear mesh (atlas .5 <= u < .75, v < .5) as a flat position array. */
export function beardPoints(gear: Rest): Float32Array {
  const out: number[] = [];
  for (let i = 0; i < gear.count; i++) {
    const u = gear.uv[i * 2], v = gear.uv[i * 2 + 1];
    if (u >= 0.5 && u < 0.75 && v < 0.5) out.push(gear.pos[i * 3], gear.pos[i * 3 + 1], gear.pos[i * 3 + 2]);
  }
  return new Float32Array(out);
}

/**
 * Clothes: one mesh holding coat, trousers and shirt (Suit3). Adds aRest (vec3), aRegion (coat, trousers, shirt,
 * button), aEdge (seam distance, hem distance, cavity, sleeve), aLimb (axial distance, inside, joint code, weight),
 * aWear (dust, abrasion, grime, fade), aCloth (coat part, lapel roll-line distance, beard shadow, shoulder cap); shapes the
 * cloth in bind space.
 */
export function annotateClothes(mesh: THREE.SkinnedMesh, j: Joints, map: THREE.Texture | null, beard: Float32Array | null) {
  const r = restOf(mesh);
  const isl = islands(r);
  const nb = neighbours(r, false);
  const luma = sampleLuma(map, r);
  // per island: size, bones, height range, brightness
  const info = new Map<number, { n: number; min: V3; max: V3; leg: number; arm: number; luma: number }>();
  for (let i = 0; i < r.count; i++) {
    const k = isl[i];
    let s = info.get(k);
    if (!s) info.set(k, (s = { n: 0, min: [9, 9, 9], max: [-9, -9, -9], leg: 0, arm: 0, luma: 0 }));
    s.n++;
    for (let a = 0; a < 3; a++) {
      s.min[a] = Math.min(s.min[a], r.pos[i * 3 + a]);
      s.max[a] = Math.max(s.max[a], r.pos[i * 3 + a]);
    }
    if (isLeg(r.bone[i])) s.leg++;
    if (isArm(r.bone[i])) s.arm++;
    s.luma += luma ? luma[i] : 0;
  }
  const kind = new Map<number, 'coat' | 'trousers' | 'shirt' | 'button'>();
  const part = new Map<number, number>();
  for (const [k, s] of info) {
    const diag = len3(sub(s.max, s.min));
    const l = s.luma / s.n;
    if (s.n <= 24 && diag < 0.04) kind.set(k, 'button');
    else if (luma && l > 0.33) kind.set(k, 'shirt');
    else if (s.leg / s.n > 0.55 && s.max[1] < 1.12) kind.set(k, 'trousers');
    else kind.set(k, 'coat');
    if (kind.get(k) !== 'coat') continue;
    // Suit3's coat islands: two sleeves, two back panels meeting in the centre seam, two fronts, two lapels, the
    // collar halves and the pocket flaps
    let cp: number = COAT_PART.front;
    if (s.arm / s.n > 0.5) cp = COAT_PART.sleeve;
    else if (s.min[1] > 1.52) cp = COAT_PART.collar;
    else if (s.n < 60) cp = COAT_PART.flap;
    else if (s.min[1] > 1.15 && s.max[2] > 0.15 && s.n < 140) cp = COAT_PART.lapel;
    else if (s.max[2] < 0.06) cp = COAT_PART.back;
    part.set(k, cp);
  }

  shoulderWeights(mesh, r, (i) => kind.get(isl[i]) === 'coat', j);

  // seams (an island edge that meets another island) and hems (open edges)
  const bnd = boundary(r);
  const twins = weldTwins(r, 0.004);
  const seamSrc: number[] = [], hemSrc: number[] = [], vSrc: number[] = [];
  const shirtPts: number[] = [];
  for (let i = 0; i < r.count; i++) if (kind.get(isl[i]) === 'shirt') shirtPts.push(r.pos[i * 3], r.pos[i * 3 + 1], r.pos[i * 3 + 2]);
  const shirtGrid = new PointGrid(new Float32Array(shirtPts), 0.02);
  for (let i = 0; i < r.count; i++) {
    if (!bnd[i]) continue;
    const kk = kind.get(isl[i]);
    if (kk === 'button') continue;
    const meets = twins[i].some((t) => isl[t] !== isl[i] && bnd[t] && kind.get(isl[t]) === kk);
    (meets ? seamSrc : hemSrc).push(i);
    // the lapel's edge along the shirt V: its roll line runs just inside it
    if (part.get(isl[i]) === COAT_PART.lapel && shirtGrid.nearest(r.pos[i * 3], r.pos[i * 3 + 1], r.pos[i * 3 + 2], 0.03) < 0.022) vSrc.push(i);
  }
  const seamD = distanceFrom(r, nb, seamSrc, 0.2);
  const hemD = distanceFrom(r, nb, hemSrc, 0.3);
  const vD = distanceFrom(r, nb, vSrc, 0.06);
  const cav = cavity(r, neighbours(r, true), 0.012);
  const beardGrid = beard && beard.length ? new PointGrid(beard, 0.04) : null;

  const SH = { l: j.at('upperarm_l'), r: j.at('upperarm_r') };
  const EL = { l: j.at('lowerarm_l'), r: j.at('lowerarm_r') };
  const WR = { l: j.at('hand_l'), r: j.at('hand_r') };
  const HI = { l: j.at('thigh_l'), r: j.at('thigh_r') };
  const KN = { l: j.at('calf_l'), r: j.at('calf_r') };
  const AN = { l: j.at('foot_l'), r: j.at('foot_r') };

  const aRest = new Float32Array(r.count * 3);
  const aRegion = new Float32Array(r.count * 4);
  const aEdge = new Float32Array(r.count * 4);
  const aLimb = new Float32Array(r.count * 4);
  const aWear = new Float32Array(r.count * 4);
  const aCloth = new Float32Array(r.count * 4);
  const disp = new Float32Array(r.count * 3);
  for (let i = 0; i < r.count; i++) {
    const p: V3 = [r.pos[i * 3], r.pos[i * 3 + 1], r.pos[i * 3 + 2]];
    const nrm: V3 = [r.nrm[i * 3], r.nrm[i * 3 + 1], r.nrm[i * 3 + 2]];
    aRest.set(p, i * 3);
    const kk = kind.get(isl[i])!;
    const reg = kk === 'coat' ? 0 : kk === 'trousers' ? 1 : kk === 'shirt' ? 2 : 3;
    aRegion[i * 4 + reg] = 1;
    const b = r.bone[i];
    const cp = part.get(isl[i]) ?? 0;
    const sleeve = kk === 'coat' && isArm(b) ? 1 : 0;
    aEdge[i * 4] = seamD[i];
    aEdge[i * 4 + 1] = hemD[i];
    aEdge[i * 4 + 2] = cav[i];
    aEdge[i * 4 + 3] = sleeve;
    const ax = Math.abs(p[0]), sx = p[0] >= 0 ? 1 : -1;
    // beard shadow onto the collar, kerchief and coat collar under the chin
    let bOcc = 0;
    if (beardGrid && p[1] > 1.42) bOcc = 1 - sstep(0.008, 0.07, beardGrid.nearest(p[0], p[1], p[2], 0.08));
    // the outer shoulder cap that lowering the arm swings out of line (pulled in by the material's pose corrective)
    const lat = ax - Math.abs(SH.l[0]);
    const cap = kk === 'coat' ? Math.exp(-(((lat - 0.035) / 0.032) ** 2)) * sstep(SH.l[1] - 0.045, SH.l[1], p[1]) * (1 - sstep(SH.l[1] + 0.055, SH.l[1] + 0.085, p[1])) : 0;
    aCloth.set([cp, cp === COAT_PART.lapel ? vD[i] : 1, bOcc, cap], i * 4);

    let dust = 0, abr = 0, grime = 0, fade = 0;
    if ((kk === 'coat' || kk === 'shirt') && isArm(b)) {
      const s = sideOf(b) as 'l' | 'r';
      const lc = limbCoord(p, SH[s], EL[s], WR[s], [0, 0, 1]);
      const w = Math.exp(-((lc.t / 0.1) ** 2));
      aLimb.set([lc.t, lc.side, s === 'l' ? 1 : 2, w], i * 4);
      // elbows rub through at the back, cuffs at the edge
      abr = Math.max(abr, Math.exp(-((lc.t / 0.05) ** 2)) * sstep(0.1, -0.7, lc.side) * 0.85);
      if (kk === 'coat') {
        abr = Math.max(abr, (1 - sstep(0.0, 0.035, hemD[i])) * sstep(0.12, 0.25, lc.t) * 0.95);
        dust = 0.14 + 0.2 * sstep(0.05, 0.25, lc.t) * sstep(0.3, -0.5, lc.side);
        grime = (1 - sstep(0.0, 0.06, hemD[i])) * sstep(0.12, 0.25, lc.t) * 0.75;
      } else {
        grime = 0.6;
      }
    } else if (kk === 'trousers' && isLeg(b)) {
      const s = sideOf(b) as 'l' | 'r';
      const lc = limbCoord(p, HI[s], KN[s], AN[s], [0, 0, -1]);
      const w = Math.exp(-((lc.t / 0.13) ** 2));
      aLimb.set([lc.t, lc.side, s === 'l' ? 3 : 4, w], i * 4);
      const front = sstep(-0.2, 0.6, nrm[2]);
      const knee = Math.exp(-((lc.t / 0.09) ** 2)) * sstep(0.0, -0.8, lc.side);
      dust = 0.18 + 0.55 * sstep(0.6, 0.12, p[1]) + 0.6 * knee * front + 0.18 * sstep(0.9, 0.6, p[1]) * front;
      abr = 0.55 * knee + (1 - sstep(0, 0.03, hemD[i])) * sstep(0.25, 0.12, p[1]) * 0.8;
      // seat and back of the knees: shiny, grimy wear; grime along the front pocket openings
      grime = sstep(0.0, 0.4, -nrm[2]) * Math.exp(-(((p[1] - 0.93) / 0.07) ** 2)) * 0.6;
      const pocket = Math.exp(-(((ax - 0.125 - (1.05 - p[1]) * 0.35) / 0.012) ** 2)) * sstep(0.88, 0.95, p[1]) * sstep(1.08, 1.0, p[1]) * front;
      grime = Math.max(grime, pocket * 0.9);
    } else if (kk === 'coat') {
      // coat body: dust low and on the front, worn hem, pocket and lapel edges, grimy collar
      dust = 0.1 + 0.26 * sstep(1.05, 0.86, p[1]) + 0.1 * sstep(0, 0.8, nrm[2]);
      abr = (1 - sstep(0.0, 0.008, hemD[i])) * (p[1] < 0.92 ? 0.6 : 0.3);
      grime = Math.max(sstep(1.48, 1.56, p[1]) * 0.7, cp === COAT_PART.flap ? (1 - sstep(0.0, 0.012, hemD[i])) * 0.8 : 0);
    } else if (kk === 'shirt') {
      grime = sstep(1.5, 1.6, p[1]) * 0.75;
    }
    // sun-faded tops: shoulders, the tops of the sleeves and the thighs
    fade = sstep(0.15, 0.75, nrm[1]);
    aWear.set([Math.min(1, dust), Math.min(1, abr), Math.min(1, grime), fade], i * 4);

    // ---- shape (rest space): a displacement per vertex, applied below with welded twins averaged
    let dn = 0;
    let dx = 0, dy = 0, dz = 0;
    if (kk === 'coat') {
      if (cp === COAT_PART.sleeve) {
        // slack sleeves: no padding; in the cross-section the top rests on the arm, the underside sags, and a few
        // soft lengthwise drape folds run down it; the cloth bunches above the cuff
        const s = sideOf(b) as 'l' | 'r';
        const upper = dot3(sub(p, SH[s]), norm3(sub(EL[s], SH[s]))) < len3(sub(EL[s], SH[s]));
        const al = upper ? aroundLimb(p, SH[s], EL[s]) : aroundLimb(p, EL[s], WR[s]);
        // Suit3's sleeves are cut for padding (7 cm round the upper arm): taken in, except at the cuff
        dn -= (upper ? 0.009 : 0.005) * sstep(0.03, 0.12, hemD[i]);
        dn += -0.0045 * Math.max(0, al.top) + 0.004 * Math.max(0, -al.top);
        dn += 0.003 * Math.sin(al.ang * 3 + 1.7 * vnoise3(p[0] * 9, p[1] * 9, p[2] * 9)) * sstep(0.02, 0.08, al.t);
        dn += 0.003 * (1 - sstep(0.0, 0.07, hemD[i])) * (upper ? 0 : 1);
      } else if (cp === COAT_PART.collar) {
        // the collar lies on the shirt collar instead of standing off the neck
        dn += 0.0015;
      } else if (cp === COAT_PART.lapel) {
        // the lapel lies on the chest and rolls over at the roll line just inside its edge along the V
        dn += 0.009 + 0.0045 * Math.exp(-(((vD[i] - 0.009) / 0.009) ** 2)) + 0.002 * (1 - sstep(0.0, 0.006, vD[i]));
      } else {
        // the body stands off the shirt (bulk), less over the shoulder blades and at the neck
        dn += 0.012 - 0.008 * sstep(1.4, 1.52, p[1]) * sstep(0.08, 0.14, ax);
        if (cp !== COAT_PART.flap) dn += 0.012 * sstep(1.25, 1.05, p[1]) * sstep(0.84, 0.92, p[1]); // the skirt hangs looser over the hips
        // hip pockets bulge (flaps at y 1.03-1.09, |x| 0.07-0.17) and the flaps sag
        for (const k of [-1, 1]) {
          const ex = p[0] * k - 0.12, ey = p[1] - 1.0;
          if (nrm[2] > 0.1) dn += 0.016 * Math.exp(-(ex * ex) / (0.05 * 0.05) - (ey * ey) / (0.05 * 0.05));
        }
        if (cp === COAT_PART.flap && p[1] < 1.1) {
          const k = sstep(1.086, 1.027, p[1]);
          dy -= 0.005 * k;
          dn += 0.004 * k;
        }
        // a lumpy outline: felted wool never holds a smooth shell
        dn += 0.0032 * vnoise3(p[0] * 7 + 3.1, p[1] * 7, p[2] * 7 - 1.7) + 0.0014 * vnoise3(p[0] * 19, p[1] * 19 + 5.3, p[2] * 19);
        // an uneven hem: dips and lifts along it, the front corners droop, the back hangs over the seat
        if (p[1] < 1.02) {
          const hw = 1 - sstep(0.0, 0.11, hemD[i]);
          const ang = Math.atan2(p[0], p[2]);
          let h = 0.011 * vnoise3(ang * 1.9 + 7.3, 0.5, 0.5) + 0.005 * vnoise3(ang * 5.1, 2.5, 1.5);
          h -= 0.008 * Math.exp(-(((p[0] + 0.015) / 0.05) ** 2)) * (p[2] > 0.05 ? 1 : 0);
          h -= 0.005 * sstep(0.0, -0.6, nrm[2]);
          dy += h * hw;
          dn += 0.004 * hw * sstep(0.0, -0.6, nrm[2]);
        }
        // the vent's left flap overlaps the right one (the bottom 22 cm of the centre-back seam)
        const vent = cp === COAT_PART.back ? (1 - sstep(1.04, 1.1, p[1])) * Math.exp(-((p[0] / 0.03) ** 2)) : 0;
        if (vent > 0) dn += p[0] > 0.0005 ? 0.0035 * vent : -0.001 * vent;
      }
      if (cp === COAT_PART.sleeve) {
        // the sleeve keeps a lumpier, slacker outline too
        dn += 0.0022 * vnoise3(p[0] * 8 - 2.2, p[1] * 8, p[2] * 8 + 4.1);
      }
      // natural sloping shoulders: the shoulder line falls from the collar to the arm, the padded cap is gone
      const g = sstep(0.095, 0.17, ax) * (1 - 0.55 * sstep(0.24, 0.31, ax));
      const tw = sstep(-0.2, 0.6, nrm[1]);
      dy -= 0.022 * g * sstep(1.4, 1.49, p[1]) * (0.35 + 0.65 * tw);
      // and narrower: Suit3's padded cap stands ~8 cm outside the shoulder joint (a natural shoulder ~5 cm); the
      // nearer the cap sits to the joint, the less a lowered arm swings it out
      dx -= sx * 0.035 * sstep(0.12, 0.25, ax) * sstep(1.37, 1.45, p[1]);
    } else if (kk === 'trousers') {
      const lc = aLimb[i * 4 + 3] > 0 ? aLimb[i * 4] : 0;
      // wool trousers hang straight and slack: baggy at the knee, no flare at the bottom
      dn += 0.008 + 0.002 * sstep(0.5, 0.3, p[1]);
      dn += 0.008 * Math.exp(-((lc / 0.08) ** 2)) * sstep(-0.2, -0.8, aLimb[i * 4 + 1]);
      // the suit's legs widen toward the hem (stovepipes once they stack up): taken in at the bottom
      dn -= 0.006 * sstep(0.3, 0.14, p[1]);
      // where the legs break over the boots the cloth crumples into a few uneven folds
      const s = sideOf(b) as 'l' | 'r';
      const angL = Math.atan2(p[0] - KN[s][0], p[2] - KN[s][2]);
      dn += 0.0055 * sstep(0.32, 0.17, p[1]) * Math.sin(p[1] * (Math.PI * 2 / 0.06) + 2.2 * vnoise3(p[0] * 11, p[1] * 5, p[2] * 11) + angL);
      dn += 0.002 * vnoise3(p[0] * 9 + 1.3, p[1] * 6, p[2] * 9);
      // the legs' last 15 cm stack up over the boot shafts (the hem rises from 0.11 to ~0.18 m)
      if (p[1] < 0.26) dy += 0.45 * (0.26 - p[1]);
      // the hem rests on the boot's instep at the front and hangs to the heel at the back
      const hw = 1 - sstep(0.0, 0.05, hemD[i]);
      dy += hw * (0.009 * sstep(0.0, 0.8, nrm[2]) - 0.004 * sstep(0.0, -0.8, nrm[2]));
    } else if (kk === 'shirt' && p[1] > 1.488 && !isArm(b)) {
      // the neckerchief round the collar has volume: soft folds round the neck and a loose knot at the throat
      const ang = Math.atan2(p[0], p[2]);
      let d = 0.0035 + 0.0022 * Math.sin(ang * 5 + 1.3 + 2 * vnoise3(p[0] * 30, p[1] * 30, p[2] * 30));
      d += 0.011 * Math.exp(-((p[0] / 0.024) ** 2) - (((p[1] - 1.515) / 0.017) ** 2)) * (p[2] > 0.03 ? 1 : 0);
      dn += d * sstep(1.488, 1.505, p[1]) * (1 - sstep(1.585, 1.6, p[1]));
    }
    disp[i * 3] = nrm[0] * dn + dx;
    disp[i * 3 + 1] = nrm[1] * dn + dy;
    disp[i * 3 + 2] = nrm[2] * dn + dz;
  }
  // seams between pieces are welded through the average
  displace(mesh, r, disp, weldTwins(r, 0.002));
  packAttributes(mesh.geometry, r.count, [['aRest', aRest, 3], ['aRegion', aRegion, 4], ['aEdge', aEdge, 4], ['aLimb', aLimb, 4], ['aWear', aWear, 4], ['aCloth', aCloth, 4]]);
}

/**
 * Move vertices by rest-space offsets in the geometry's own (possibly quantized) space. Welded twins move by their
 * average so seams stay closed.
 */
function displace(mesh: THREE.SkinnedMesh, r: Rest, disp: Float32Array, twins: number[][]) {
  const o = new Float32Array(disp);
  for (let i = 0; i < r.count; i++) {
    if (!twins[i].length) continue;
    for (let k = 0; k < 3; k++) {
      let s = disp[i * 3 + k];
      for (const t of twins[i]) s += disp[t * 3 + k];
      o[i * 3 + k] = s / (twins[i].length + 1);
    }
  }
  moveRest(mesh, (i, out) => out.set(o[i * 3], o[i * 3 + 1], o[i * 3 + 2]), (i) => o[i * 3] !== 0 || o[i * 3 + 1] !== 0 || o[i * 3 + 2] !== 0);
}

/**
 * Skin: face and hands. aRest, aCav (cavity, the beard's shadow on the neck), aZone (face, hand, red:
 * nose/cheekbones/ears, stubble), aZone2 (eye socket, lips, crow's feet + forehead lines weight, knuckles).
 */
export function annotateSkin(mesh: THREE.SkinnedMesh, j: Joints, r: Rest, beard: Float32Array | null) {
  const cav = cavity(r, neighbours(r, true), 0.01);
  const aRest = new Float32Array(r.count * 3);
  const aZone = new Float32Array(r.count * 4);
  const aZone2 = new Float32Array(r.count * 4);
  const aCav = new Float32Array(r.count * 2);
  const knuckles: V3[] = [];
  for (const s of ['l', 'r']) for (const f of ['index', 'middle', 'ring', 'pinky']) for (const n of ['01', '02', '03']) knuckles.push(j.at(`${f}_${n}_${s}`));
  const beardGrid = beard && beard.length ? new PointGrid(beard, 0.025) : null;
  const L = LANDMARKS;
  const g2 = (p: V3, c: readonly number[], rx: number, ry: number, rz: number) =>
    Math.exp(-(((p[0] - c[0]) / rx) ** 2) - (((p[1] - c[1]) / ry) ** 2) - (((p[2] - c[2]) / rz) ** 2));
  for (let i = 0; i < r.count; i++) {
    const p: V3 = [r.pos[i * 3], r.pos[i * 3 + 1], r.pos[i * 3 + 2]];
    const nz = r.nrm[i * 3 + 2];
    aRest.set(p, i * 3);
    aCav[i * 2] = cav[i];
    const b = r.bone[i];
    const hand = /^(hand|thumb|index|middle|ring|pinky)_/.test(b) || (/^lowerarm_/.test(b) && Math.abs(p[0]) > 0.44) ? 1 : 0;
    const face = p[1] > 1.55 && !hand ? 1 : 0;
    // under the jaw the beard keeps the sky and the sun off the neck
    if (beardGrid && face && p[1] < 1.615) aCav[i * 2 + 1] = (1 - sstep(0.004, 0.04, beardGrid.nearest(p[0], p[1], p[2], 0.05))) * sstep(1.615, 1.585, p[1]);
    let red = 0, stub = 0, sock = 0, lips = 0, lines = 0, knk = 0;
    if (face) {
      red = Math.max(
        g2(p, L.noseTip, 0.018, 0.022, 0.03) * 0.9,
        g2(p, [0.05, 1.688, 0.125], 0.024, 0.016, 0.03) * 0.75,
        g2(p, [-0.05, 1.688, 0.125], 0.024, 0.016, 0.03) * 0.75,
        sstep(0.068, 0.085, Math.abs(p[0])) * sstep(1.64, 1.68, p[1]) * 0.8, // ears
      );
      // stubble where the beard thins onto the cheeks (up to the cheekbones) and over the upper lip
      const jaw = sstep(1.715, 1.67, p[1]) * sstep(0.03, 0.055, Math.abs(p[0])) * sstep(0.04, 0.09, p[2]);
      const lip = g2(p, [0, 1.648, 0.165], 0.03, 0.009, 0.03);
      stub = Math.min(1, jaw * 0.9 + lip * 0.9 + sstep(1.65, 1.6, p[1]) * sstep(0.08, 0.13, p[2]));
      sock = Math.max(g2(p, L.eyeL, 0.022, 0.014, 0.03), g2(p, L.eyeR, 0.022, 0.014, 0.03));
      lips = g2(p, L.mouth, 0.024, 0.008, 0.03) * sstep(0.12, 0.155, p[2]);
      // crow's feet beside the outer eye corners, lines across the forehead
      const crow = Math.max(g2(p, [0.066, 1.708, 0.115], 0.012, 0.012, 0.03), g2(p, [-0.066, 1.708, 0.115], 0.012, 0.012, 0.03));
      const brow = sstep(1.735, 1.75, p[1]) * sstep(1.8, 1.775, p[1]) * sstep(0.05, 0.12, p[2]) * sstep(0.06, 0.03, Math.abs(p[0]));
      lines = Math.min(1, crow + brow);
    }
    if (hand) {
      let best = 1;
      for (const k of knuckles) best = Math.min(best, len3(sub(p, k)));
      knk = Math.exp(-((best / 0.011) ** 2));
    }
    aZone.set([face, hand, red, stub], i * 4);
    aZone2.set([sock, lips, lines, knk * (0.4 + 0.6 * sstep(-0.2, 0.6, nz))], i * 4);
  }
  packAttributes(mesh.geometry, r.count, [['aRest', aRest, 3], ['aCav', aCav, 2], ['aZone', aZone, 4], ['aZone2', aZone2, 4]]);
}

/**
 * Gear atlas: boots, hat, hair, beard, brows/lashes/eyes. aRest, aGear: boots and hat (cavity, toe crease weight,
 * dust, sweat line); hair and beard (cavity, depth over the skin, -, -). The brim gains a felt edge with thickness,
 * a slow wave and a slight curl at the back.
 */
export function annotateGear(mesh: THREE.SkinnedMesh, r: Rest, skin: Rest | null) {
  const cav = cavity(r, neighbours(r, true), 0.01);
  const aRest = new Float32Array(r.count * 3);
  const aGear = new Float32Array(r.count * 4);
  const B = LANDMARKS.brim;
  const dy = new Float32Array(r.count);
  /** beard cards pulled in toward the face where it thins (up the cheeks): not the same depth everywhere */
  const bx = new Float32Array(r.count), bz = new Float32Array(r.count);
  const under: number[] = [];
  let skinGrid: PointGrid | null = null;
  if (skin) {
    const head: number[] = [];
    for (let i = 0; i < skin.count; i++) if (skin.pos[i * 3 + 1] > 1.5) head.push(skin.pos[i * 3], skin.pos[i * 3 + 1], skin.pos[i * 3 + 2]);
    skinGrid = new PointGrid(new Float32Array(head), 0.02);
  }
  for (let i = 0; i < r.count; i++) {
    const p: V3 = [r.pos[i * 3], r.pos[i * 3 + 1], r.pos[i * 3 + 2]];
    const u = r.uv[i * 2], v = r.uv[i * 2 + 1];
    aRest.set(p, i * 3);
    const boots = u < 0.5 && v >= 0.5;
    const hat = u >= 0.5 && v >= 0.5;
    const hairOrBeard = v < 0.5 && u < 0.75;
    let toe = 0, dust = 0, sweat = 0;
    if (boots) {
      // creases across the toe box just behind the ball of the foot, dust toward the toes and the welt
      toe = Math.exp(-(((p[2] - 0.115) / 0.03) ** 2)) * sstep(0.05, 0.1, p[1]) * sstep(0.13, 0.08, p[1]);
      dust = 0.35 + 0.45 * sstep(0.08, 0.0, p[1]) + 0.3 * sstep(0.08, 0.2, p[2]);
    } else if (hat) {
      const ry = r.nrm[i * 3 + 1];
      dust = 0.25 + 0.55 * sstep(0.3, 0.9, ry);
      // sweat and grease soak through the felt just above the band
      sweat = Math.exp(-(((p[1] - 1.8) / 0.012) ** 2)) * sstep(0.1, 0.13, Math.hypot(p[0], p[2] - 0.06) + 0.03);
      // brim: a slow wave over its outer part, the back curling up a little, and a felt edge with thickness
      const rr = Math.hypot((p[0] + 0.003) / B.ax, (p[2] - 0.06) / B.az);
      if (rr > 0.62 && p[1] < 1.77) {
        const ang = Math.atan2(p[0], p[2] - 0.06);
        const outer = sstep(0.62, 1.0, rr);
        dy[i] = 0.006 * Math.sin(ang * 3 + 0.7) * outer + 0.007 * sstep(0.2, -0.9, Math.cos(ang)) * outer * outer;
        if (ry < -0.5 && rr > 0.85) {
          dy[i] -= 0.0065 * sstep(0.85, 1.0, rr);
          under.push(i);
        }
      }
    } else if (hairOrBeard && skinGrid) {
      sweat = 0;
      dust = 0;
      // how far the strand card stands off the skin (the beard is darkest and densest close to it)
      toe = skinGrid.nearest(p[0], p[1], p[2], 0.04);
      if (u >= 0.5) {
        const thin = Math.min(1, sstep(1.645, 1.69, p[1]) * (0.35 + 0.65 * sstep(0.07, 0.11, p[2])) + sstep(1.68, 1.714, p[1]) * 0.6);
        const pull = Math.max(0, toe - 0.0015) * 0.75 * thin;
        const hx = -p[0], hz = 0.035 - p[2], hl = Math.hypot(hx, hz) || 1;
        bx[i] = (hx / hl) * pull;
        bz[i] = (hz / hl) * pull;
        toe -= pull;
      }
    }
    aGear.set([cav[i], toe, Math.min(1, dust), sweat], i * 4);
  }
  // the edge band's lower row coincides with the underside's outer ring: it drops with it
  const twins = weldTwins(r, 0.0015);
  const isUnder = new Uint8Array(r.count);
  for (const i of under) isUnder[i] = 1;
  for (const i of under) for (const t of twins[i]) if (!isUnder[t]) dy[t] = dy[i];
  moveRest(mesh, (i, out) => out.set(bx[i], dy[i], bz[i]), (i) => dy[i] !== 0 || bx[i] !== 0 || bz[i] !== 0);
  // the strand fringe roots on the pulled-in cards
  for (let i = 0; i < r.count; i++) {
    r.pos[i * 3] += bx[i];
    r.pos[i * 3 + 2] += bz[i];
  }
  packAttributes(mesh.geometry, r.count, [['aRest', aRest, 3], ['aGear', aGear, 4]]);
}

/** Offset chosen vertices by a rest-space vector, written back through each vertex's rest transform. */
function moveRest(mesh: THREE.SkinnedMesh, offset: (i: number, out: THREE.Vector3) => void, pick: (i: number) => boolean) {
  const g = mesh.geometry;
  const pos = g.getAttribute('position') as THREE.BufferAttribute;
  const si = g.getAttribute('skinIndex'), sw = g.getAttribute('skinWeight');
  const sk = mesh.skeleton;
  const mats = sk.bones.map((b, i) => new THREE.Matrix4().multiplyMatrices(b.matrixWorld, sk.boneInverses[i]).multiply(mesh.bindMatrix));
  const m = new THREE.Matrix4(), inv = new THREE.Matrix3(), v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    if (!pick(i)) continue;
    const e = m.elements;
    e.fill(0);
    for (let k = 0; k < 4; k++) {
      const w = sw.getComponent(i, k);
      if (w <= 0) continue;
      const me = mats[si.getComponent(i, k)].elements;
      for (let q = 0; q < 16; q++) e[q] += me[q] * w;
    }
    inv.setFromMatrix4(m).invert();
    offset(i, v);
    v.applyMatrix3(inv);
    pos.setXYZ(i, pos.getX(i) + v.x, pos.getY(i) + v.y, pos.getZ(i) + v.z);
  }
  pos.needsUpdate = true;
}

/** Deterministic hash in [0, 1). */
function rnd(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Strand cards in head-bone space: a heavy moustache over the upper lip drooping past the mouth corners, a soft
 * fringe of single hairs round the beard's outline and up the cheeks where it thins, and short loose hairs at the
 * nape. Each card is a tapered, gently curved ribbon; the material cuts 4 hairs out of it (uv.x across, uv.y root
 * -> tip) and gives each card its own tone (aTone).
 */
export function buildStrands(skin: Rest, gear: Rest, headRest: THREE.Matrix4): THREE.BufferGeometry {
  const toHead = headRest.clone().invert();
  const R = rnd(7);
  const pos: number[] = [], uv: number[] = [], tone: number[] = [], nrm: number[] = [];
  const idx: number[] = [];
  const v = new THREE.Vector3(), d = new THREE.Vector3(), side = new THREE.Vector3(), n = new THREE.Vector3();
  /** one curved card: root p (rest space), direction dir, surface normal nn, length, width, droop (m) */
  const card = (p: THREE.Vector3, dir: THREE.Vector3, nn: THREE.Vector3, length: number, width: number, droop: number, t: number) => {
    const seg = 3;
    const base = pos.length / 3;
    side.crossVectors(dir, nn).normalize();
    for (let s = 0; s <= seg; s++) {
      const f = s / seg;
      const w = width * (1 - 0.75 * f);
      // the card bends down (droop) and slightly off the skin as it grows
      v.copy(p).addScaledVector(dir, length * f).addScaledVector(nn, length * 0.12 * f * (1 - f) + 0.0015).add(d.set(0, -droop * f * f, 0));
      for (const k of [-1, 1]) {
        const q = v.clone().addScaledVector(side, (k * w) / 2).applyMatrix4(toHead);
        pos.push(q.x, q.y, q.z);
        uv.push(k < 0 ? 0 : 1, f);
        tone.push(t);
        n.copy(nn).transformDirection(toHead);
        nrm.push(n.x, n.y, n.z);
      }
    }
    for (let s = 0; s < seg; s++) {
      const a = base + s * 2;
      idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
  };
  const P = new THREE.Vector3(), N = new THREE.Vector3(), D = new THREE.Vector3();
  // moustache roots on the upper lip skin between the nose base and the lip
  const lip: number[] = [];
  for (let i = 0; i < skin.count; i++) {
    const x = skin.pos[i * 3], y = skin.pos[i * 3 + 1], z = skin.pos[i * 3 + 2];
    if (Math.abs(x) < 0.032 && y > 1.638 && y < 1.66 && z > 0.135 && skin.nrm[i * 3 + 2] > 0.35) lip.push(i);
  }
  for (let c = 0; c < 360 && lip.length; c++) {
    const i = lip[Math.floor(R() * lip.length)];
    P.set(skin.pos[i * 3], skin.pos[i * 3 + 1], skin.pos[i * 3 + 2]);
    N.set(skin.nrm[i * 3], skin.nrm[i * 3 + 1], skin.nrm[i * 3 + 2]);
    // hairs grow down and out from the philtrum, longer toward the corners where they droop past the mouth
    const sx = Math.sign(P.x || (R() - 0.5));
    const ax = Math.abs(P.x) / 0.032;
    D.set(sx * (0.25 + 0.55 * ax), -1, 0.25).normalize();
    card(P, D, N, 0.015 + 0.015 * ax + R() * 0.006, 0.006 + R() * 0.0025, 0.004 + 0.008 * ax, 0.25 + R() * 0.45);
  }
  // fringe: single hairs out of the beard's outline (beard cards' outer rows) and over the cheeks
  const beardV: number[] = [];
  for (let i = 0; i < gear.count; i++) {
    const u = gear.uv[i * 2], vv = gear.uv[i * 2 + 1];
    if (u >= 0.5 && u < 0.75 && vv < 0.5 && gear.pos[i * 3 + 1] > 1.57) beardV.push(i);
  }
  for (let c = 0; c < 240 && beardV.length; c++) {
    const i = beardV[Math.floor(R() * beardV.length)];
    P.set(gear.pos[i * 3], gear.pos[i * 3 + 1], gear.pos[i * 3 + 2]);
    N.set(gear.nrm[i * 3], gear.nrm[i * 3 + 1], gear.nrm[i * 3 + 2]);
    if (N.lengthSq() < 0.5) continue;
    // grow mostly downward along the jaw, splaying off the surface a little; short and sparse up the cheeks
    const high = sstep(1.63, 1.69, P.y);
    D.set((R() - 0.5) * 0.6, -1, (R() - 0.3) * 0.5).addScaledVector(N, 0.35 + 0.3 * high).normalize();
    card(P, D, N, (0.012 + R() * 0.014) * (1 - 0.5 * high), 0.003 + R() * 0.0015, 0.003, 0.2 + R() * 0.7);
  }
  // loose hairs at the nape, out from under the hat
  for (let c = 0; c < 60; c++) {
    let best = -1;
    for (let tries = 0; tries < 30 && best < 0; tries++) {
      const i = Math.floor(R() * gear.count);
      const u = gear.uv[i * 2], vv = gear.uv[i * 2 + 1];
      const y = gear.pos[i * 3 + 1], z = gear.pos[i * 3 + 2], x = gear.pos[i * 3];
      if (u < 0.5 && vv < 0.5 && y < 1.71 && z < -0.01 && Math.abs(x) < 0.065) best = i;
    }
    if (best < 0) continue;
    const i = best;
    P.set(gear.pos[i * 3], gear.pos[i * 3 + 1], gear.pos[i * 3 + 2]);
    N.set(gear.nrm[i * 3], gear.nrm[i * 3 + 1], gear.nrm[i * 3 + 2]);
    D.set((R() - 0.5) * 0.5, -1, -0.15).addScaledVector(N, 0.15).normalize();
    card(P, D, N, 0.009 + R() * 0.008, 0.004 + R() * 0.002, 0.002, R());
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setAttribute('aTone', new THREE.Float32BufferAttribute(tone, 1));
  geo.setIndex(idx);
  geo.computeBoundingSphere();
  return geo;
}

/** The strand mesh rides on the head bone; main view only (not in the AO pre-pass, the mirror or the shadow map). */
export function strandMesh(geo: THREE.BufferGeometry, mat: THREE.Material): THREE.Mesh {
  const m = new THREE.Mesh(geo, mat);
  m.name = 'angler-strands';
  m.layers.set(LAYER_FOLIAGE);
  m.castShadow = false;
  m.receiveShadow = true;
  m.frustumCulled = false;
  return m;
}
