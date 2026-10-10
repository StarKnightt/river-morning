import * as THREE from 'three/webgpu';
import { hash3, type Assembler, type GroundY } from './assemble';

/**
 * Split firewood built piece by piece (critic r2: the woodpile "a wall of identical pale-tan wedge ends, all the same
 * size and colour and all flush, repeating like wallpaper"; the firewood by the ring "a tidy little stack of pale,
 * near-identical pieces", a crate in C1). Rounds, halves, thirds and quarters split from logs of very different
 * girth, each its own length, roll, lean and weathering: end grain from fresh cream to rain-greyed and dark inside the
 * bark's dark rim, split faces rough along the grain. Geometry is generated in world space and handed to the
 * assembler as one merged geometry per material (bark, split faces, ends), drawn with the camp's shared surface
 * program; per-vertex tints carry the weathering.
 */

export type Shape = 'round' | 'half' | 'third' | 'quarter';

/** a cross-section outline point (pith at the origin): on the bark or on a split face */
interface OP { x: number; y: number; bark: boolean }

/** vertex accumulator with a tint per vertex (multiplied in by the assembler) */
export class Acc {
  pos: number[] = [];
  nrm: number[] = [];
  uv: number[] = [];
  tint: number[] = [];
  idx: number[] = [];

  get count() {
    return this.pos.length / 3;
  }

  v(p: THREE.Vector3, n: THREE.Vector3, u: number, w: number, t: readonly [number, number, number]) {
    this.pos.push(p.x, p.y, p.z);
    this.nrm.push(n.x, n.y, n.z);
    this.uv.push(u, w);
    this.tint.push(t[0], t[1], t[2]);
    return this.count - 1;
  }

  /** bake into the assembler's `group` with `src`'s maps, drawn as the camp material `as` */
  addTo(asm: Assembler, src: THREE.MeshStandardMaterial, group: string, as: string, label: string, small = false, groundDirt = true) {
    if (!this.idx.length) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nrm, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    g.setIndex(this.idx);
    const tint = this.tint;
    asm.addGeometry(g, src, {
      group, small, groundDirt, remap: () => as,
      edit: () => (_p, t, i) => {
        t[i * 3] *= tint[i * 3]; t[i * 3 + 1] *= tint[i * 3 + 1]; t[i * 3 + 2] *= tint[i * 3 + 2];
      },
    }, label);
  }
}

export interface WoodAcc { bark: Acc; split: Acc; ends: Acc }
export const woodAcc = (): WoodAcc => ({ bark: new Acc(), split: new Acc(), ends: new Acc() });

export interface BilletOpts {
  shape: Shape;
  /** radius of the log it was split from (m) */
  R: number;
  len: number;
  seed: number;
  /** cut faces: 0 fresh cream .. 0.8 rain-grey .. 1 dark and old */
  age: number;
}

const lerp3 = (a: readonly number[], b: readonly number[], t: number): [number, number, number] =>
  [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
// (round 4 wip: the grey ends still read tan under the eave: a cooler grey takes the end grain's warmth out)
const AGE: Array<readonly [number, number, number]> = [[1.58, 1.3, 0.94], [1.08, 1.0, 0.92], [0.64, 0.76, 0.97], [0.37, 0.4, 0.47]];
/** tint of a cut face by its age (piecewise between fresh, seasoned, grey and dark) */
export function ageTint(age: number): [number, number, number] {
  const a = Math.min(0.999, Math.max(0, age)) * 3;
  const i = Math.floor(a);
  return lerp3(AGE[i], AGE[i + 1], a - i);
}

/** the cross-section outline, counter-clockwise, pith at the origin */
export function outline(shape: Shape, R: number, seed: number): OP[] {
  const h = (k: number) => hash3(seed, k, 7.7);
  const wob = (th: number) => R * (1 + 0.07 * (h(1) - 0.5) + 0.05 * Math.sin(3 * th + 6.28 * h(2)) + 0.03 * Math.sin(5 * th + 6.28 * h(3)));
  const out: OP[] = [];
  if (shape === 'round') {
    const n = 12;
    for (let i = 0; i < n; i++) {
      const th = (i / n) * Math.PI * 2;
      const r = wob(th);
      out.push({ x: Math.cos(th) * r, y: Math.sin(th) * r, bark: true });
    }
    return out;
  }
  const span = (shape === 'half' ? Math.PI : shape === 'third' ? (2 * Math.PI) / 3 : Math.PI / 2) * (0.88 + 0.22 * h(4));
  const na = shape === 'half' ? 8 : shape === 'third' ? 6 : 5;
  for (let i = 0; i <= na; i++) {
    const th = (span * i) / na;
    const r = wob(th);
    out.push({ x: Math.cos(th) * r, y: Math.sin(th) * r, bark: true });
  }
  // split faces: rough along the grain, the brittle pith broken off the corner
  const a = out[na], b = out[0];
  const jag = (k: number) => (h(10 + k) - 0.5) * 0.09 * R;
  const along = (p: OP, q: OP, t: number, k: number) => {
    const dx = q.x - p.x, dy = q.y - p.y, l = Math.hypot(dx, dy) || 1;
    out.push({ x: p.x + dx * t + (dy / l) * jag(k), y: p.y + dy * t - (dx / l) * jag(k), bark: false });
  };
  if (shape === 'half') {
    for (let k = 1; k <= 3; k++) along(a, b, k / 4, k);
    return out;
  }
  const pith = { x: 0, y: 0, bark: false };
  along(a, pith, 0.5, 1);
  const c = 0.06 + 0.05 * h(20);
  out.push({ x: a.x * c, y: a.y * c, bark: false }, { x: b.x * c, y: b.y * c, bark: false });
  along(pith, b, 0.5, 2);
  return out;
}

const _p = new THREE.Vector3(), _n = new THREE.Vector3(), _m3 = new THREE.Matrix3();

/**
 * One billet: its outline swept along local z (-len/2 .. len/2), the sawn ends tilted a few degrees each, placed by
 * `M` (local -> world, rotation and translation only). Bark faces get smooth radial normals, split faces flat ones;
 * the end caps map the end-grain disc (pith at the texture's centre) with the bark rim darkened.
 */
export function billet(acc: WoodAcc, o: BilletOpts, M: THREE.Matrix4) {
  const h = (k: number) => hash3(o.seed, k, 3.3);
  const P = outline(o.shape, o.R, o.seed);
  const n = P.length, L = o.len;
  _m3.getNormalMatrix(M);
  const tilt = [[(h(1) - 0.5) * 0.24, (h(2) - 0.5) * 0.24], [(h(3) - 0.5) * 0.24, (h(4) - 0.5) * 0.24]];
  const zEnd = (x: number, y: number, e: number) => (e ? L / 2 : -L / 2) + tilt[e][0] * x + tilt[e][1] * y;
  const kb = 0.6 + 0.6 * h(5);
  const barkT: [number, number, number] = h(6) < 0.5 ? [kb, kb * 0.97, kb * 0.93] : [kb * 0.95, kb * 0.96, kb * 0.98];
  const faceT = ageTint(o.age);
  const splitT = lerp3(faceT, [1, 1, 1], 0.25);
  const ou = h(7) * 5, ov = h(8) * 5;
  const put = (a: Acc, x: number, y: number, z: number, nx: number, ny: number, nz: number, u: number, w: number, t: [number, number, number]) => {
    _p.set(x, y, z).applyMatrix4(M);
    _n.set(nx, ny, nz).applyMatrix3(_m3).normalize();
    return a.v(_p, _n, u, w, t);
  };
  // sides
  let s = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const p = P[i], q = P[j];
    const len = Math.hypot(q.x - p.x, q.y - p.y);
    const bark = p.bark && q.bark;
    const a = bark ? acc.bark : acc.split;
    const tile = bark ? 0.7 : 0.45;
    const fn = [q.y - p.y, p.x - q.x];
    const fl = Math.hypot(fn[0], fn[1]) || 1;
    const nrm = (pt: OP) => {
      if (!bark) return [fn[0] / fl, fn[1] / fl];
      const r = Math.hypot(pt.x, pt.y) || 1;
      return [pt.x / r, pt.y / r];
    };
    const np = nrm(p), nq = nrm(q);
    const t = bark ? barkT : splitT;
    const v0 = put(a, p.x, p.y, zEnd(p.x, p.y, 0), np[0], np[1], 0, (s + ou) / tile, (zEnd(p.x, p.y, 0) + ov) / tile, t);
    const v1 = put(a, q.x, q.y, zEnd(q.x, q.y, 0), nq[0], nq[1], 0, (s + len + ou) / tile, (zEnd(q.x, q.y, 0) + ov) / tile, t);
    const v2 = put(a, q.x, q.y, zEnd(q.x, q.y, 1), nq[0], nq[1], 0, (s + len + ou) / tile, (zEnd(q.x, q.y, 1) + ov) / tile, t);
    const v3 = put(a, p.x, p.y, zEnd(p.x, p.y, 1), np[0], np[1], 0, (s + ou) / tile, (zEnd(p.x, p.y, 1) + ov) / tile, t);
    a.idx.push(v0, v1, v2, v0, v2, v3);
    s += len;
  }
  // ends: a fan from the centroid; the disc's radius maps to 0.44 of the texture (its dark surround is the bark rim)
  let cx = 0, cy = 0;
  for (const p of P) {
    cx += p.x;
    cy += p.y;
  }
  cx /= n;
  cy /= n;
  const rot = h(9) * 6.283, cr = Math.cos(rot), sr = Math.sin(rot), flip = h(10) < 0.5 ? -1 : 1;
  const k = 0.44 / o.R;
  const tu = (x: number, y: number) => 0.5 + (x * cr - y * sr) * k * flip;
  const tv = (x: number, y: number) => 0.5 + (x * sr + y * cr) * k;
  for (let e = 0; e < 2; e++) {
    const sgn = e ? 1 : -1;
    const nz = [-tilt[e][0] * sgn, -tilt[e][1] * sgn, sgn];
    const nl = Math.hypot(nz[0], nz[1], nz[2]);
    // the other end of a piece weathers on its own (one end out in the rain, one in the stack)
    const et = e ? faceT : lerp3(faceT, ageTint(Math.min(1, o.age + 0.25)), 0.6);
    const c = put(acc.ends, cx, cy, zEnd(cx, cy, e), nz[0] / nl, nz[1] / nl, nz[2] / nl, tu(cx, cy), tv(cx, cy), et);
    const rim = P.map((p) => {
      const t: [number, number, number] = p.bark ? [et[0] * 0.5, et[1] * 0.46, et[2] * 0.42] : [et[0] * 0.92, et[1] * 0.92, et[2] * 0.92];
      return put(acc.ends, p.x, p.y, zEnd(p.x, p.y, e), nz[0] / nl, nz[1] / nl, nz[2] / nl, tu(p.x, p.y), tv(p.x, p.y), t);
    });
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      if (e) acc.ends.idx.push(c, rim[i], rim[j]);
      else acc.ends.idx.push(c, rim[j], rim[i]);
    }
  }
}

/** bounds of a rolled outline, and its lowest / highest point at a horizontal offset (for stacking) */
function profile(P: OP[], roll: number) {
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const Q = P.map((p) => ({ x: p.x * cr - p.y * sr, y: p.x * sr + p.y * cr }));
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const q of Q) {
    x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y);
  }
  /** lowest and highest y of the outline over the vertical line at x */
  const span = (x: number): [number, number] => {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < Q.length; i++) {
      const a = Q[i], b = Q[(i + 1) % Q.length];
      if ((x < a.x && x < b.x) || (x > a.x && x > b.x)) continue;
      const t = Math.abs(b.x - a.x) < 1e-6 ? 0.5 : (x - a.x) / (b.x - a.x);
      const y = a.y + (b.y - a.y) * t;
      lo = Math.min(lo, y);
      hi = Math.max(hi, y);
    }
    return lo === Infinity ? [y0, y0] : [lo, hi];
  };
  return { x0, x1, y0, y1, span };
}

/** a piece split from a log: the shape and log girth for a 0..1 random (mostly quarters and halves) */
function pickPiece(r: number, r2: number, big = 1): { shape: Shape; R: number } {
  if (r < 0.36) return { shape: 'quarter', R: (0.07 + 0.12 * Math.pow(r2, 1.2)) * big };
  if (r < 0.6) return { shape: 'half', R: (0.05 + 0.08 * r2) * big };
  if (r < 0.74) return { shape: 'third', R: (0.07 + 0.08 * r2) * big };
  if (r < 0.9) return { shape: 'round', R: (0.03 + 0.045 * r2) * big };
  return { shape: 'round', R: (0.07 + 0.035 * r2) * big };
}

/** how weathered a stacked piece's cut faces are: fresh cream, grey, dark or seasoned between (round 2: all one tan) */
function pickAge(r: number, r2: number, top: number) {
  // (scene fix, critic polish r1: the stack's ends "a neat wall of identical round ends": fewer fresh ones, more grey)
  if (r < 0.14) return top > 0.75 ? 0.35 + 0.15 * r2 : 0.15 * r2;
  if (r < 0.7) return 0.62 + 0.26 * r2;
  if (r < 0.86) return 0.88 + 0.12 * r2;
  return 0.3 + 0.2 * r2;
}

const _q = new THREE.Quaternion(), _e = new THREE.Euler(), _s = new THREE.Vector3(1, 1, 1);

/**
 * The woodpile against the wall, in the frame `M` of the props GLB's Prop_WoodStack (x along the 1.5 m stack, z its
 * depth with +z the face, y up from its foot): two bark rails on the ground, then course after course of pieces
 * laid left to right, each resting on whatever lies under it (pieces nest into the gaps of the course below), its
 * end pushed out or sunk back up to ~7 cm, turned and tilted a little; the top course stops piece by piece at a
 * ragged height. Returns the number of pieces.
 */
export function buildWoodpile(acc: WoodAcc, M: THREE.Matrix4, o: { x0: number; x1: number; height: number; seed: number }) {
  const RES = 0.01;
  const nx = Math.ceil((o.x1 - o.x0) / RES);
  const h = (k: number, j = 0) => hash3(o.seed + k * 0.37, j, 5.5);
  // rails: two weathered poles on the ground under the stack
  for (const rz of [-0.12, 0.11]) {
    const m = new THREE.Matrix4().compose(new THREE.Vector3(0, 0.035, rz), _q.setFromEuler(_e.set(0, Math.PI / 2 + (rz > 0 ? 0.03 : -0.02), 0)), _s);
    billet(acc, { shape: 'round', R: 0.04, len: o.x1 - o.x0 + 0.16, seed: o.seed + rz * 10, age: 0.95 }, M.clone().multiply(m));
  }
  const floor = new Float32Array(nx).fill(0.07);
  let pieces = 0;
  for (let pass = 0, id = 0; pass < 60; pass++) {
    let x = o.x0 + h(pass, 1) * 0.02;
    let placed = 0;
    while (x < o.x1 - 0.03) {
      id++;
      const { shape, R } = pickPiece(h(id, 2), h(id, 3));
      const P = outline(shape, R, o.seed * 31 + id);
      // halves mostly lie flat side down or up, quarters and thirds any way round
      const roll = shape === 'half' ? (h(id, 4) < 0.6 ? Math.PI : 0) + (h(id, 5) - 0.5) * 0.5 : h(id, 4) * 6.283;
      const pr = profile(P, roll);
      const w = pr.x1 - pr.x0;
      if (x + w > o.x1 + 0.015) {
        if (w < 0.05) break;
        x += 0.01;
        continue;
      }
      // rest on the highest point under the piece, sinking a few millimetres into the gaps
      let y = -Infinity;
      const c0 = Math.max(0, Math.floor((x - o.x0) / RES)), c1 = Math.min(nx - 1, Math.floor((x + w - o.x0) / RES));
      for (let c = c0; c <= c1; c++) y = Math.max(y, floor[c] - pr.span(pr.x0 + (c - c0 + 0.5) * RES)[0]);
      y -= 0.002 + 0.004 * h(id, 6);
      const top = o.height + (h(id, 7) - 0.5) * 0.16 + 0.05 * Math.sin(x * 9 + o.seed);
      if (y + pr.y1 > top) {
        // the top course: leave a gap and try the next spot
        x += w * (0.4 + 0.5 * h(id, 8));
        continue;
      }
      for (let c = c0; c <= c1; c++) floor[c] = Math.max(floor[c], y + pr.span(pr.x0 + (c - c0 + 0.5) * RES)[1]);
      const len = 0.33 + 0.13 * h(id, 9);
      // pushed out or sunk back; now and then one well proud of the face
      const dz = (h(id, 10) - 0.5) * 0.1 + (h(id, 11) < 0.12 ? 0.06 : 0);
      const yNorm = Math.min(1, y / o.height);
      const age = pickAge(h(id, 12), h(id, 13), yNorm);
      const m = new THREE.Matrix4().compose(
        new THREE.Vector3(x - pr.x0, y, dz),
        _q.setFromEuler(_e.set((h(id, 14) - 0.5) * 0.06, (h(id, 15) - 0.5) * 0.08, roll, 'YXZ')),
        _s,
      );
      billet(acc, { shape, R, len, seed: o.seed * 31 + id, age }, M.clone().multiply(m));
      pieces++;
      placed++;
      x += w + 0.003 + 0.024 * h(id, 16) ** 1.5;
    }
    if (!placed && pass > 3) break;
  }
  return pieces;
}

/**
 * Firewood dropped by the fire: an untidy spill of mixed pieces and a few thin sticks lying every which way, the
 * later ones resting on (or leaning off) the earlier ones (a coarse height grid), a couple fallen off to the side.
 * World coordinates; `groundY` for the ground. Returns the pieces' footprint radius.
 */
export function buildFirewoodHeap(acc: WoodAcc, groundY: GroundY,
  o: { x: number; z: number; yaw: number; seed: number; count: number; blockers?: Array<{ x: number; z: number; r: number }> }) {
  const CELL = 0.03, N = 56, half = (N * CELL) / 2;
  const hm = new Float32Array(N * N);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) hm[j * N + i] = groundY(o.x - half + (i + 0.5) * CELL, o.z - half + (j + 0.5) * CELL);
  const at = (x: number, z: number) => {
    const i = Math.floor((x - o.x + half) / CELL), j = Math.floor((z - o.z + half) / CELL);
    return i < 0 || j < 0 || i >= N || j >= N ? -1 : j * N + i;
  };
  const h = (k: number, j: number) => hash3(o.seed + k * 0.71, j, 9.1);
  let reach = 0;
  for (let k = 0; k < o.count; k++) {
    const stick = k >= o.count - 3;
    const pp = stick ? { shape: 'round' as Shape, R: 0.018 + 0.016 * h(k, 1) } : pickPiece(h(k, 1), h(k, 2), 0.85);
    const len = stick ? 0.5 + 0.35 * h(k, 3) : 0.3 + 0.15 * h(k, 3);
    // most in a loose spill, a few rolled or thrown aside; a piece that would pile up too high slides off outward
    const aside = h(k, 4) < 0.22;
    let rr = aside ? 0.42 + 0.3 * h(k, 5) : 0.44 * Math.sqrt(h(k, 5));
    const ra = h(k, 6) * 6.283;
    const yaw = o.yaw + (h(k, 7) - 0.5) * 2.6;
    const ax = Math.cos(yaw), az = -Math.sin(yaw);
    const P = outline(pp.shape, pp.R, o.seed * 17 + k);
    const roll = h(k, 8) * 6.283;
    const pr = profile(P, roll);
    let px = 0, pz = 0, y0 = 0, y1 = 0;
    for (let tries = 0; tries < 5; tries++) {
      px = o.x + Math.cos(ra) * rr * 1.25;
      pz = o.z + Math.sin(ra) * rr * 0.85;
      // the resting height under each half of the piece, and how far it tilts between them
      const rest = (t0: number, t1: number) => {
        let y = -Infinity;
        for (let s = t0; s <= t1; s += 0.05) {
          for (const off of [pr.x0 * 0.7, 0, pr.x1 * 0.7]) {
            const c = at(px + ax * s * len - az * off, pz + az * s * len + ax * off);
            if (c >= 0) y = Math.max(y, hm[c]);
          }
        }
        return y === -Infinity ? groundY(px, pz) : y;
      };
      y0 = rest(-0.5, -0.05) - pr.y0;
      y1 = rest(0.05, 0.5) - pr.y0;
      // (where its underside would rest, above the ground)
      if (Math.max(y0, y1) + pr.y0 - groundY(px, pz) < 0.14) break;
      rr += 0.16;
    }
    const pitch = Math.max(-0.45, Math.min(0.45, Math.atan2(y1 - y0, len * 0.55)));
    const yc = Math.max(y0, y1) - Math.abs(y1 - y0) * 0.45 - 0.004;
    const m = new THREE.Matrix4().compose(
      new THREE.Vector3(px, yc, pz),
      _q.setFromEuler(_e.set(0, yaw + Math.PI / 2, 0, 'YXZ')).multiply(new THREE.Quaternion().setFromEuler(_e.set(-pitch, 0, roll, 'XYZ'))),
      _s,
    );
    const age = stick ? 0.7 : Math.min(1, 0.15 + 0.75 * h(k, 9));
    billet(acc, { shape: pp.shape, R: pp.R, len, seed: o.seed * 17 + k, age }, m);
    // what it now covers
    for (let s = -0.5; s <= 0.5; s += 0.04) {
      const yy = yc + Math.sin(pitch) * s * len + pr.y1;
      for (const off of [pr.x0 * 0.8, 0, pr.x1 * 0.8]) {
        const c = at(px + ax * s * len - az * off, pz + az * s * len + ax * off);
        if (c >= 0) hm[c] = Math.max(hm[c], yy);
      }
    }
    reach = Math.max(reach, Math.hypot(px - o.x, pz - o.z) + len / 2);
    // every piece standing shin-high (over 0.12 m) blocks a walker: two circles along it (the spill reaches well past
    // the layout's circle; sticks and thin pieces lie lower and are stepped over)
    const top = yc + Math.abs(Math.sin(pitch)) * len * 0.5 + pr.y1 - groundY(px, pz);
    if (o.blockers && top > 0.12) {
      for (const s of [-0.28, 0.28]) o.blockers.push({ x: px + ax * s * len, z: pz + az * s * len, r: len * 0.24 + pp.R * 0.6 });
    }
  }
  return reach;
}

/**
 * Chips and bark flakes lying where wood is split: thin irregular slabs (pale split wood, a few fresh) and dark bark
 * curls, scattered round (x, z) out to `r`, thickest near the middle. Small: main view and shadows only.
 */
export function scatterChips(chips: Acc, bark: Acc, groundY: GroundY, o: { x: number; z: number; r: number; count: number; seed: number; fresh?: number }) {
  const h = (k: number, j: number) => hash3(o.seed + k * 0.53, j, 4.4);
  for (let k = 0; k < o.count; k++) {
    const isBark = h(k, 1) < 0.3;
    const rr = o.r * Math.pow(h(k, 2), 0.7), ra = h(k, 3) * 6.283;
    const x = o.x + Math.cos(ra) * rr, z = o.z + Math.sin(ra) * rr;
    const L = 0.025 + 0.06 * h(k, 4) * (isBark ? 1.2 : 1), W = L * (0.3 + 0.35 * h(k, 5)), T = isBark ? 0.006 : 0.004 + 0.008 * h(k, 6);
    const yaw = h(k, 7) * 6.283, tilt = (h(k, 8) - 0.5) * 0.5;
    const m = new THREE.Matrix4().compose(new THREE.Vector3(x, groundY(x, z) + T * 0.3, z), _q.setFromEuler(_e.set(tilt, yaw, (h(k, 9) - 0.5) * 0.3, 'YXZ')), _s);
    const nm = new THREE.Matrix3().getNormalMatrix(m);
    const age = (o.fresh ?? 0.3) > h(k, 10) ? 0.05 : 0.5 + 0.45 * h(k, 11);
    const t: [number, number, number] = isBark ? [0.55, 0.5, 0.46] : ageTint(age);
    const a = isBark ? bark : chips;
    // an irregular slab: the top face's corners pulled in and out, the bottom face smaller
    const c = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([u, v], i) => [u * L * (0.5 + 0.25 * h(k, 20 + i)), v * W * (0.5 + 0.25 * h(k, 30 + i))]);
    const faces: Array<[number[], number[], number[], number[], number[]]> = [];
    const P3 = (i: number, top: boolean) => [c[i][0] * (top ? 1 : 0.85), top ? T / 2 : -T / 2, c[i][1] * (top ? 1 : 0.85)];
    faces.push([P3(0, true), P3(3, true), P3(2, true), P3(1, true), [0, 1, 0]]);
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      const dx = c[j][0] - c[i][0], dz = c[j][1] - c[i][1], l = Math.hypot(dx, dz) || 1;
      faces.push([P3(i, false), P3(i, true), P3(j, true), P3(j, false), [dz / l, 0.3, -dx / l]]);
    }
    for (const [a0, a1, a2, a3, nn] of faces) {
      _n.set(nn[0], nn[1], nn[2]).applyMatrix3(nm).normalize();
      const ids = [a0, a1, a2, a3].map((q, i) => {
        _p.set(q[0], q[1], q[2]).applyMatrix4(m);
        return a.v(_p, _n, (q[0] + h(k, 12)) * 4 + (i & 1) * 0.05, (q[2] + h(k, 13)) * 4, t);
      });
      a.idx.push(ids[0], ids[1], ids[2], ids[0], ids[2], ids[3]);
    }
  }
}
