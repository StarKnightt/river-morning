import type * as THREE from 'three/webgpu';
import { hash3, type PieceInfo, type VertexEdit } from './assemble';
import { cabinX, cabinZ, toCabinX, toCabinZ } from './layout';

/**
 * Hand-built timber, edited piece by piece (the GLB's logs, shakes and planks are separate connected pieces; the
 * edits move their vertices in the GLB's own frame). Critic r1: the walls "look like a printed board" (every log
 * the same thickness, dead straight, the same streaks repeating, the corner ends a column of identical discs), the
 * roof "a flat, pale, finely speckled sheet like sandpaper" with "ruler-straight edges", the dock planks all the
 * same width with ends cut flush in a line.
 */

/** drop every piece */
const DROP = (): false => false;

const sstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** cabin-local <-> world for vertex edits (the cabin stands at height `cy`) */
export function cabinFrame(cy: number) {
  const L = { x: 0, y: 0, z: 0 };
  return {
    toLocal(p: Float32Array, i: number) {
      const x = p[i * 3], z = p[i * 3 + 2];
      L.x = toCabinX(x, z);
      L.y = p[i * 3 + 1] - cy;
      L.z = toCabinZ(x, z);
      return L;
    },
    toWorld(p: Float32Array, i: number, x: number, y: number, z: number) {
      p[i * 3] = cabinX(x, z);
      p[i * 3 + 1] = y + cy;
      p[i * 3 + 2] = cabinZ(x, z);
    },
  };
}

interface Log {
  axis: 'x' | 'z';
  yc: number; oc: number; a0: number; a1: number;
  corner0: boolean; corner1: boolean;
  s0: number; taper: number; sag: number; bow: number; end0: number; end1: number; jag: number; ph: number;
  tint: [number, number, number]; du: number; dv: number;
  /** axe-cut corner ends: the angle of the ridge between the two facets, and how steep they are */
  f0: number; f1: number; k0: number; k1: number;
}

type Bounds = { min: THREE.Vector3; max: THREE.Vector3 };

/**
 * The log walls: every log its own thickness (+-12 %), taper, sag or hump and a slight bow, so some touch and some
 * leave wide gaps; its texture shifted so no two logs show the same streaks; greyer, browner or more silver. At the
 * corners the ends stick out by different lengths, their outline ragged; their end discs (darker, a ring pattern
 * of their own) follow the same deformation. The chinking is pushed back into the joints (in shadow).
 */
export function logWalls(logs: Bounds[], cy: number) {
  const F = cabinFrame(cy);
  const list: Log[] = logs.map((b) => {
    const axis: 'x' | 'z' = b.max.x - b.min.x > b.max.z - b.min.z ? 'x' : 'z';
    const h = (k: number) => hash3(b.min.x * 3 + k, b.min.y * 7 + b.max.y, b.min.z * 5 - k);
    const a0 = axis === 'x' ? b.min.x : b.min.z, a1 = axis === 'x' ? b.max.x : b.max.z;
    // corner ends reach past the crossing wall (x logs past +-2.9, z logs past +-2.4)
    const lim = axis === 'x' ? 2.9 : 2.4;
    const corner0 = a0 < -lim, corner1 = a1 > lim;
    const brown = h(9) < 0.3, silver = !brown && h(9) < 0.5;
    const k = 0.84 + 0.3 * h(10);
    const tint: [number, number, number] = brown ? [k * 1.06, k * 0.98, k * 0.88] : silver ? [k * 0.96, k * 0.99, k * 1.04] : [k, k, k];
    return {
      axis, yc: (b.min.y + b.max.y) / 2, oc: axis === 'x' ? (b.min.z + b.max.z) / 2 : (b.min.x + b.max.x) / 2, a0, a1, corner0, corner1,
      s0: 0.88 + 0.24 * h(1), taper: (h(2) - 0.5) * 0.24, sag: (h(3) - 0.3) * 0.06, bow: (h(4) - 0.5) * 0.05,
      end0: corner0 ? (h(5) - 0.3) * 0.28 : 0, end1: corner1 ? (h(6) - 0.3) * 0.28 : 0, jag: 0.05 + 0.1 * h(7), ph: h(8) * 6.28,
      tint, du: h(11) * 3.7, dv: h(12) * 2.9,
      f0: h(13) * Math.PI, f1: h(14) * Math.PI, k0: 0.25 + 0.35 * h(15), k1: 0.25 + 0.35 * h(16),
    };
  });
  const find = (axis: 'x' | 'z', yc: number, oc: number, a: number) => {
    let best: Log | null = null, bd = 0.09;
    for (const L of list) {
      if (L.axis !== axis || a < L.a0 - 0.03 || a > L.a1 + 0.03) continue;
      const d = Math.abs(L.yc - yc) + Math.abs(L.oc - oc);
      if (d < bd) [best, bd] = [L, d];
    }
    return best;
  };
  /** move one vertex (cabin-local) as the log deforms; returns false outside the log's span */
  const deform = (L: Log, x: number, y: number, z: number, out: { x: number; y: number; z: number }) => {
    const a = L.axis === 'x' ? x : z, o = L.axis === 'x' ? z : x;
    const t = Math.min(1, Math.max(0, (a - L.a0) / (L.a1 - L.a0)));
    const bend = 4 * t * (1 - t);
    const ay = L.yc - L.sag * bend, ao = L.oc + L.bow * bend;
    const dy = y - L.yc, dO = o - L.oc;
    let s = L.s0 * (1 + L.taper * (t - 0.5));
    // ragged ends: the outline of the last few centimetres wanders round the log
    const endW = Math.max(0, 1 - (a - L.a0) / 0.07, 1 - (L.a1 - a) / 0.07);
    if (endW > 0) {
      const th = Math.atan2(dy, dO);
      s *= 1 + Math.min(1, endW) * L.jag * (0.6 * Math.sin(3 * th + L.ph) + 0.4 * Math.sin(7 * th + 2 * L.ph));
    }
    let a2 = a;
    if (L.corner0) a2 -= L.end0 * sstep(L.a0 + 0.4, L.a0, a);
    if (L.corner1) a2 += L.end1 * sstep(L.a1 - 0.4, L.a1, a);
    // corner ends chopped with an axe from two sides (round 2's read as freshly sawn rounds): the end face breaks into
    // two facets meeting in a ridge; the log's last ring of vertices and the end disc's rim move together
    const R0 = 0.14 * L.s0;
    if (L.corner0 && a < L.a0 + 0.035) {
      const q = dy * Math.cos(L.f0) + dO * Math.sin(L.f0);
      a2 -= L.k0 * (0.45 * R0 - Math.abs(q)) * (1 - (a - L.a0) / 0.035);
    }
    if (L.corner1 && a > L.a1 - 0.035) {
      const q = dy * Math.cos(L.f1) + dO * Math.sin(L.f1);
      a2 += L.k1 * (0.45 * R0 - Math.abs(q)) * (1 - (L.a1 - a) / 0.035);
    }
    const y2 = ay + dy * s, o2 = ao + dO * s;
    out.x = L.axis === 'x' ? a2 : o2;
    out.z = L.axis === 'x' ? o2 : a2;
    out.y = y2;
  };
  const tmp = { x: 0, y: 0, z: 0 };
  return (pivot: string, material: string) => {
    if (pivot !== 'Cabin_Walls') return null;
    if (material === 'LogWood') {
      return (piece: PieceInfo): VertexEdit | null => {
        const b = piece;
        const axis: 'x' | 'z' = b.max.x - b.min.x > b.max.z - b.min.z ? 'x' : 'z';
        const L = find(axis, (b.min.y + b.max.y) / 2, axis === 'x' ? (b.min.z + b.max.z) / 2 : (b.min.x + b.max.x) / 2, axis === 'x' ? (b.min.x + b.max.x) / 2 : (b.min.z + b.max.z) / 2);
        if (!L) return null;
        return (p, t, i, _h, uv) => {
          const l = F.toLocal(p, i);
          const a = L.axis === 'x' ? l.x : l.z;
          deform(L, l.x, l.y, l.z, tmp);
          F.toWorld(p, i, tmp.x, tmp.y, tmp.z);
          // the corner ends weather darkest and greyest: out in the rain on three sides (round 4 wip: the sunlit ends
          // still read pale tan against the shaded wall)
          const endW = Math.max(L.corner0 ? sstep(L.a0 + 0.6, L.a0, a) : 0, L.corner1 ? sstep(L.a1 - 0.6, L.a1, a) : 0);
          const endK = 1 - 0.56 * endW;
          t[i * 3] *= L.tint[0] * endK * (1 - 0.05 * endW); t[i * 3 + 1] *= L.tint[1] * endK; t[i * 3 + 2] *= L.tint[2] * endK * (1 + 0.05 * endW);
          uv[i * 2] += L.du;
          uv[i * 2 + 1] += L.dv;
        };
      };
    }
    if (material === 'LogEnds') {
      return (piece: PieceInfo): VertexEdit | null => {
        const b = piece;
        // a disc: flat across the log's axis
        const axis: 'x' | 'z' = b.max.x - b.min.x < b.max.z - b.min.z ? 'x' : 'z';
        const a = axis === 'x' ? (b.min.x + b.max.x) / 2 : (b.min.z + b.max.z) / 2;
        const L = find(axis, (b.min.y + b.max.y) / 2, axis === 'x' ? (b.min.z + b.max.z) / 2 : (b.min.x + b.max.x) / 2, a);
        if (!L) return null;
        const rot = piece.seed * 6.28, cr = Math.cos(rot), sr = Math.sin(rot);
        const ou = (hash3(a, L.yc, 3) - 0.5) * 0.2, ov = (hash3(a, L.yc, 4) - 0.5) * 0.2;
        // the most weathered wood on the cabin: grey-brown to near black, never paler than the logs
        const k = 0.5 + 0.42 * hash3(a, L.yc, 5);
        return (p, t, i, _h, uv) => {
          const l = F.toLocal(p, i);
          deform(L, l.x, l.y, l.z, tmp);
          F.toWorld(p, i, tmp.x, tmp.y, tmp.z);
          t[i * 3] *= L.tint[0] * k; t[i * 3 + 1] *= L.tint[1] * k; t[i * 3 + 2] *= L.tint[2] * k;
          const u = uv[i * 2] - 0.5, v = uv[i * 2 + 1] - 0.5;
          uv[i * 2] = 0.5 + u * cr - v * sr + ou;
          uv[i * 2 + 1] = 0.5 + u * sr + v * cr + ov;
        };
      };
    }
    if (material === 'Chinking') {
      // pushed 2.5 cm back into the joint; a few strips sag or bulge
      return (piece: PieceInfo): VertexEdit | null => {
        const k = 0.8 + 0.35 * piece.seed;
        return (p, t, i) => {
          const l = F.toLocal(p, i);
          let { x, z } = l;
          if (Math.abs(l.z) > 2.3 && Math.abs(l.x) < 2.95) z -= Math.sign(l.z) * 0.025;
          else if (Math.abs(l.x) > 2.8) x -= Math.sign(l.x) * 0.025;
          F.toWorld(p, i, x, l.y, z);
          t[i * 3] *= k; t[i * 3 + 1] *= k; t[i * 3 + 2] *= k;
        };
      };
    }
    return null;
  };
}

/** the roof's eaves and gable edges (cabin-local): main roof and porch roof */
const ROOF = {
  main: { x0: -3.45, x1: 3.32, zBack: -2.98, zFront: 2.65, slope: 1.856 / 2.95 },
  porch: { x0: -2.55, x1: 1.15, zFront: 4.24, slope: 0.384 / 1.75 },
};

/**
 * The shakes: each lifts at its butt (curled, so every course throws a shadow line), a few are twisted, split dark
 * or missing; colours from silver-grey to near black with browner ones between (moss and lichen come in the
 * material). Along the eaves and gables the shakes overhang by uneven lengths, and `edge` adds a second, doubled
 * course under the edge shakes: a thick, ragged edge instead of a ruler line. Fascia and barge boards are dropped.
 */
export function roofShakes(cy: number, edge = false) {
  const F = cabinFrame(cy);
  return (pivot: string, material: string) => {
    if (pivot !== 'Cabin_Roof') return null;
    if (material === 'Boards') {
      // the thin fascia boards along the eaves and the barge boards up the gables
      if (edge) return DROP;
      return (b: PieceInfo) => {
        const sx = b.max.x - b.min.x, sz = b.max.z - b.min.z;
        return (sz < 0.035 && sx > 3) || sx < 0.035 ? false : null;
      };
    }
    if (material !== 'RoofShakes' && material !== 'RoofShakesDark') return edge ? DROP : null;
    return (b: PieceInfo): VertexEdit | false | null => {
      const sx = b.max.x - b.min.x, sz = b.max.z - b.min.z;
      // the dark underlay sheet and anything else larger than a shake stays as it is
      if (sx > 0.6 || sz > 0.7) return edge ? false : null;
      const h = (k: number) => hash3(b.min.x * 11 + k, b.min.y * 13, b.min.z * 7 - k);
      const zc = (b.min.z + b.max.z) / 2, xc = (b.min.x + b.max.x) / 2;
      const porch = b.min.z > 2.6 || (b.max.z > 2.75 && b.max.y < 3.12 && xc > ROOF.porch.x0 - 0.1 && xc < ROOF.porch.x1 + 0.1);
      const front = zc > 0;
      const R = porch ? ROOF.porch : ROOF.main;
      const eaveZ = porch ? ROOF.porch.zFront : front ? ROOF.main.zFront : ROOF.main.zBack;
      const atEave = front ? b.max.z > eaveZ - 0.07 : b.min.z < eaveZ + 0.07;
      const atGable0 = b.min.x < R.x0 + 0.06, atGable1 = b.max.x > R.x1 - 0.06;
      if (edge && !atEave && !atGable0 && !atGable1) return false;
      if (!edge && !atEave && h(1) < 0.035) return false;
      const curl = 0.008 + 0.03 * h(2) * h(2) + (h(3) < 0.12 ? 0.02 : 0);
      const twist = (h(4) - 0.5) * 0.024;
      const ext = atEave ? 0.03 + 0.12 * h(5) : 0;
      const g0 = atGable0 ? 0.03 + 0.1 * h(6) : 0, g1 = atGable1 ? 0.03 + 0.1 * h(7) : 0;
      const r = h(8);
      const k = 0.62 + 0.6 * h(9);
      // whole courses run lighter or darker (rows that read from above, C6), and shakes from silver to near black
      // within them (round 2: "one flat, even, dark grey sheet like roofing felt")
      const cf = 0.72 + 0.56 * hash3(Math.round(b.min.y * 9), porch ? 7 : front ? 3 : 5, 2.2);
      const tint: [number, number, number] = (r < 0.12 ? [0.4, 0.4, 0.41] : r < 0.3 ? [k * 1.1, k * 0.96, k * 0.82] : r < 0.46 ? [1.45, 1.47, 1.5] : [k, k, k * 0.98])
        .map((c) => c * cf) as [number, number, number];
      const ou = h(10) * 4.1, ov = h(11) * 3.3;
      // the doubled course sits a finger under the edge and a little further out (a thick, ragged eave)
      const dDown = edge ? 0.03 : 0, dOut = edge ? 0.015 + 0.035 * h(12) : 0;
      return (p, t, i, _h, uv) => {
        const l = F.toLocal(p, i);
        let { x, y, z } = l;
        const tb = front ? (z - b.min.z) / sz : (b.max.z - z) / sz;
        y += curl * tb * tb + twist * ((x - xc) / (sx / 2)) * tb;
        const out = (ext + dOut) * sstep(0.4, 1.0, tb);
        if (out > 0) {
          z += front ? out : -out;
          y -= out * R.slope;
        }
        if (g0 > 0 || (edge && atGable0)) x -= (g0 + (atGable0 ? dOut : 0)) * sstep(b.min.x + sx * 0.6, b.min.x, x);
        if (g1 > 0 || (edge && atGable1)) x += (g1 + (atGable1 ? dOut : 0)) * sstep(b.max.x - sx * 0.6, b.max.x, x);
        y -= dDown;
        F.toWorld(p, i, x, y, z);
        t[i * 3] *= tint[0]; t[i * 3 + 1] *= tint[1]; t[i * 3 + 2] *= tint[2];
        uv[i * 2] += ou;
        uv[i * 2 + 1] += ov;
      };
    };
  };
}

/**
 * Sky light under the porch roof (round 2: "the porch roof casts no shade: the door, the wall under the roof and the
 * deck are as bright as open wall"): the roof and its beam hide all but a band of sky low over the yard, so the
 * wall darkens toward the top and the corners; 1 in the open. Multiplied into the baked contact occlusion
 * (indirect light only; the sun's own shadow comes from the shadow map).
 */
export function porchShelter(cy: number) {
  const P = ROOF.porch;
  // the porch roof's front edge with its beam (cabin-local height)
  const yEdge = 2.5;
  return (x: number, y: number, z: number) => {
    const lx = toCabinX(x, z), lz = toCabinZ(x, z), ly = y - cy;
    if (lx < P.x0 - 0.3 || lx > P.x1 + 0.3 || lz < 2.3 || lz > P.zFront + 0.1 || ly > 3.05 || ly < 0.15) return 1;
    const front = Math.max(0.05, P.zFront - lz);
    const side = Math.max(0.05, Math.min(lx - P.x0, P.x1 - lx));
    const up = Math.max(0.02, yEdge - ly);
    const open = Math.max(Math.atan2(up, front), 0.6 * Math.atan2(up, side + 0.3)) / (Math.PI / 2);
    const inside = sstep(P.x0 - 0.3, P.x0 + 0.2, lx) * sstep(P.x1 + 0.3, P.x1 - 0.2, lx) * sstep(P.zFront + 0.1, P.zFront - 0.3, lz);
    return 1 - inside * 0.7 * (1 - Math.pow(open, 0.8));
  };
}

/** sky light under the porch deck: the piers there stand in its shadow, a little more open toward the front edge */
export function underDeck(cy: number) {
  const P = ROOF.porch;
  return (x: number, y: number, z: number) => {
    const lx = toCabinX(x, z), lz = toCabinZ(x, z), ly = y - cy;
    if (lx < P.x0 - 0.15 || lx > P.x1 + 0.15 || lz < 2.4 || lz > 4.35 || ly > 0.55) return 1;
    return 0.36 + 0.5 * sstep(3.8, 4.35, lz);
  };
}

/**
 * The porch deck (round 2: "new boards stained grey: every board the same width and tone, dead flat and spotless,
 * with no worn track to the door ... on a sharp-edged sawn front beam"): each board narrower by its own amount, so
 * dark gaps open between them, cupped, twisted, some lifted at the front end; darker where rain reaches the front
 * edge and along the wall; worn pale along the way from the step to the door. The long beams hewn uneven and sagging.
 */
export function porchDeck(cy: number, doorX: number) {
  const F = cabinFrame(cy);
  return (pivot: string, material: string) => {
    if (pivot !== 'Cabin_Porch') return null;
    if (material === 'Boards') {
      return (b: PieceInfo): VertexEdit | null => {
        const sx = b.max.x - b.min.x, sz = b.max.z - b.min.z;
        if (sz < 1.0) return null;
        const h = (k: number) => hash3(b.min.x * 13 + k, b.min.z * 3, 4.7);
        const xc = (b.min.x + b.max.x) / 2, zc = (b.min.z + b.max.z) / 2;
        const sw = 0.78 + 0.18 * h(1);
        const cup = 0.002 + 0.005 * h(2);
        const lift = h(3) < 0.25 ? 0.008 + 0.016 * h(4) : 0;
        const twist = (h(5) - 0.5) * 0.01;
        const k = 0.76 + 0.36 * h(6);
        const brown = h(7) < 0.3;
        const tread = Math.max(0, 1 - Math.abs(xc - doorX) / 0.7);
        const ou = h(8) * 3.1, ov = h(9) * 2.3;
        return (p, t, i, _h, uv) => {
          const l = F.toLocal(p, i);
          let { x, y } = l;
          const u = (x - xc) / (sx / 2), w = (l.z - zc) / (sz / 2);
          x = xc + (x - xc) * sw;
          y += cup * (u * u - 0.33) + lift * sstep(0.3, 1.0, w) + twist * u * w;
          F.toWorld(p, i, x, y, l.z);
          // (round 4 wip: the worn track barely showed in P4c)
          const damp = (sstep(3.5, 4.07, l.z) * 0.32 + sstep(2.85, 2.47, l.z) * 0.18) * (1 - 0.5 * tread);
          const pale = tread * (1 - sstep(3.85, 4.1, l.z) * 0.4) * 0.8;
          const m = k * (1 - damp) * (1 + pale);
          t[i * 3] *= m * (brown ? 1.05 : 1); t[i * 3 + 1] *= m; t[i * 3 + 2] *= m * (brown ? 0.88 : 1) * (1 - pale * 0.15);
          uv[i * 2] += ou;
          uv[i * 2 + 1] += ov;
        };
      };
    }
    if (material === 'LogWood') {
      return (b: PieceInfo): VertexEdit | null => {
        const sx = b.max.x - b.min.x;
        if (sx < 2.5) return null;
        const x0 = b.min.x, x1 = b.max.x;
        return (p, _t, i) => {
          const l = F.toLocal(p, i);
          const tt = Math.min(1, Math.max(0, (l.x - x0) / (x1 - x0)));
          const j = hash3(Math.round(l.x * 30), Math.round(l.y * 30), Math.round(l.z * 30)) - 0.5;
          F.toWorld(p, i, l.x, l.y - 0.025 * 4 * tt * (1 - tt) + j * 0.014, l.z + j * 0.012);
        };
      };
    }
    return null;
  };
}

/**
 * The dock (its GLB frame: x across, z along toward -Z, deck top at y 0; placed unrotated at `origin`): planks
 * of different widths and lengths with staggered ends, cupped and a few lifted at one end, bleached to different
 * silvers; the two mooring posts at the tip of different heights, leaning, their tops weathered round and ragged.
 */
export function dockTimber(origin: { x: number; y: number; z: number }) {
  return (pivot: string, material: string) => {
    if (pivot !== 'Dock') return null;
    if (material === 'DockPlanks') {
      return (b: PieceInfo): VertexEdit | null => {
        const h = (k: number) => hash3(b.min.x * 5 + k, b.min.z * 9, b.max.z * 3 - k);
        const xc = (b.min.x + b.max.x) / 2, zc = (b.min.z + b.max.z) / 2;
        const hl = (b.max.x - b.min.x) / 2, hw = (b.max.z - b.min.z) / 2;
        if (hl < 0.3) return null;
        // (round 3: "planks of nearly equal width with straight, lined-up edges": very different widths now, a few
        // with a chunk broken off a corner, more of them standing proud)
        const sw = 0.62 + 0.38 * h(1);
        const off = (1 - sw) * hw * (h(12) - 0.5) * 1.6;
        const d0 = (h(2) - 0.4) * 0.17, d1 = (h(3) - 0.4) * 0.17;
        const cup = 0.002 + 0.008 * h(4);
        const lift = h(5) < 0.3 ? 0.014 + 0.02 * h(6) : 0;
        const liftEnd = h(7) < 0.5 ? -1 : 1;
        const k = 0.82 + 0.38 * h(8);
        const brown = h(9) < 0.25;
        const tint: [number, number, number] = brown ? [k * 1.05, k * 0.99, k * 0.9] : [k, k * 1.0, k * 1.01];
        const ou = h(10) * 3.1, ov = h(11) * 2.7;
        const chunk = h(13) < 0.22 ? (h(14) < 0.5 ? -1 : 1) : 0, chunkSide = h(15) < 0.5 ? -1 : 1, chunkL = 0.06 + 0.12 * h(16);
        return (p, t, i, _h, uv) => {
          let x = p[i * 3] - origin.x, y = p[i * 3 + 1] - origin.y, z = p[i * 3 + 2] - origin.z;
          const u = (x - xc) / hl, w = (z - zc) / hw;
          z = zc + off + (z - zc) * sw;
          x += (u < 0 ? -d0 : d1) * sstep(0.55, 1.0, Math.abs(u));
          // a chunk split off one corner: the end pulled back on one side
          if (chunk && u * chunk > 0.6 && w * chunkSide > 0.1) x -= chunk * chunkL * sstep(0.1, 0.9, w * chunkSide) * sstep(0.6, 1.0, u * chunk);
          y += cup * (w * w - 0.33) + lift * Math.max(0, u * liftEnd);
          p[i * 3] = x + origin.x;
          p[i * 3 + 1] = y + origin.y;
          p[i * 3 + 2] = z + origin.z;
          t[i * 3] *= tint[0]; t[i * 3 + 1] *= tint[1]; t[i * 3 + 2] *= tint[2];
          uv[i * 2] += ou;
          uv[i * 2 + 1] += ov;
        };
      };
    }
    if (material === 'DockTimber' || material === 'DockTimberWet' || material === 'EndGrainWeathered') {
      return (b: PieceInfo): VertexEdit | null => {
        const zc = (b.min.z + b.max.z) / 2, xc = (b.min.x + b.max.x) / 2;
        const sx = b.max.x - b.min.x, sz = b.max.z - b.min.z, sy = b.max.y - b.min.y;
        // every post its own girth, out of round and rough (round 2: "posts like turned poles, all one diameter")
        const post = sx < 0.4 && sz < 0.4 && sy > 0.45;
        // (keyed to the post's place rounded to 5 cm: its dry and soaked pieces get the same girth)
        const hp = (k: number) => hash3((Math.round(xc * 20) / 20) * 7 + k, (Math.round(zc * 20) / 20) * 3, 1.9);
        const gk = post ? 0.8 + 0.42 * hp(1) : 1;
        const gph = hp(2) * 6.28;
        const rough = (x: number, y: number, z: number) => {
          if (!post) return [x, z];
          const dx = x - xc, dz = z - zc, th = Math.atan2(dz, dx);
          const f = gk * (1 + 0.06 * Math.sin(3 * th + gph) + 0.035 * Math.sin(7 * th + y * 5) + 0.02 * Math.sin(13 * th + y * 11));
          return [xc + dx * f, zc + dz * f];
        };
        // the mooring posts: the pieces at the tip pair that rise well above the deck (the deck top is y 0)
        if (zc > -6.0 || Math.abs(xc) < 0.45 || sx > 0.4) {
          if (!post) return null;
          return (p, _t, i) => {
            const [x, z] = rough(p[i * 3] - origin.x, p[i * 3 + 1] - origin.y, p[i * 3 + 2] - origin.z);
            p[i * 3] = x + origin.x;
            p[i * 3 + 2] = z + origin.z;
          };
        }
        const left = xc < 0;
        const dh = left ? 0.16 : -0.07;
        const lean = left ? [-0.045, 0.02] : [0.03, -0.05];
        const top = 0.55;
        return (p, _t, i) => {
          let x = p[i * 3] - origin.x, y = p[i * 3 + 1] - origin.y, z = p[i * 3 + 2] - origin.z;
          if (y > 0) {
            const f = y / top;
            // weathered round and split at the top
            if (y > top - 0.07) {
              const s = 1 - 0.22 * sstep(top - 0.07, top, y);
              x = xc + (x - xc) * s;
              z = zc + (z - zc) * s;
              y += 0.02 * Math.sin(Math.atan2(z - zc, x - xc) * 3 + (left ? 0.5 : 2.0)) * sstep(top - 0.03, top, y);
            }
            y += dh * f;
            x += lean[0] * f * f;
            z += lean[1] * f * f;
          }
          p[i * 3] = x + origin.x;
          p[i * 3 + 1] = y + origin.y;
          p[i * 3 + 2] = z + origin.z;
        };
      };
    }
    return null;
  };
}
