import * as THREE from 'three/webgpu';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { hash3, type Assembler } from './assemble';
import { STONE_TINTS, pick } from './stones';

/**
 * The chimney rebuilt in fieldstone (critic r1: "a straight pale-grey column, the same width top to bottom, with an
 * even all-over stone pattern; it reads as poured concrete, and its top is clean"). Courses of irregular, rounded
 * stones of mixed sizes and colours (grey, buff, rust) laid against a dark mortar core set back behind their faces
 * (deep joints), a broad firebox base, shoulders stepping in on all three open sides to a narrower stack, a crown of
 * flat stones round a black flue; the top courses are sooted (the material's soot line). Cabin-local metres
 * (the chimney stands on the +X gable, the wall face at x 3.0), baked into world space with the cabin's matrix.
 */

interface Box { x0: number; x1: number; z0: number; z1: number; y0: number; y1: number }

/** firebox, two shoulder steps, the stack (cabin-local) */
export const CHIMNEY_BOXES: Box[] = [
  { x0: 3.0, x1: 3.98, z0: -0.86, z1: 0.86, y0: -0.14, y1: 1.85 },
  { x0: 3.0, x1: 3.84, z0: -0.72, z1: 0.72, y0: 1.85, y1: 2.12 },
  { x0: 3.0, x1: 3.72, z0: -0.58, z1: 0.58, y0: 2.12, y1: 2.4 },
  { x0: 3.0, x1: 3.64, z0: -0.44, z1: 0.44, y0: 2.4, y1: 5.36 },
];
/** the stack's inner face (toward the roof) shows only above the roof's gable edge */
const INNER_FROM = 4.25;

/** One rounded, lumpy stone: L along x, H up, D deep (z), centred; uvs planar on its face at ~0.7 m per tile. */
function stoneGeo(L: number, H: number, D: number, seed: number): THREE.BufferGeometry {
  let g: THREE.BufferGeometry = new THREE.BoxGeometry(1, 1, 1, 3, 2, 2);
  g.deleteAttribute('normal');
  g.deleteAttribute('uv');
  g = mergeVertices(g);
  const p = g.getAttribute('position') as THREE.BufferAttribute;
  const r = 0.07 + 0.13 * hash3(seed, 1, 2);
  const a = 0.5 - r;
  const v = new THREE.Vector3(), c = new THREE.Vector3(), d = new THREE.Vector3();
  const uv = new Float32Array(p.count * 2);
  const ou = hash3(seed, 3, 4) * 5, ov = hash3(seed, 5, 6) * 5;
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    // a rounded box: the point clamped into the inner box, pushed out by the corner radius
    c.set(Math.max(-a, Math.min(a, v.x)), Math.max(-a, Math.min(a, v.y)), Math.max(-a, Math.min(a, v.z)));
    d.subVectors(v, c);
    const len = d.length() || 1;
    v.copy(c).addScaledVector(d, r / len);
    // lumps, the same for every vertex at one corner
    const k = 1 + 0.22 * (hash3(v.x * 9 + seed, v.y * 9, v.z * 9) - 0.5);
    v.multiplyScalar(k);
    // the face flatter than the back and sides (the mason laid the flat side out); the outline skewed a little
    if (v.z > 0) v.z *= 0.6;
    v.x += v.y * (hash3(seed, 13, 14) - 0.5) * 0.3;
    v.set(v.x * L, v.y * H, v.z * D);
    p.setXYZ(i, v.x, v.y, v.z);
    uv[i * 2] = (v.x + ou) / 0.7;
    uv[i * 2 + 1] = (v.y + ov) / 0.7;
  }
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.computeVertexNormals();
  return g;
}

/** a box with uvs scaled to its faces (~1 m per tile), for the mortar core and the flue */
function boxGeo(w: number, h: number, d: number): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(w, h, d);
  const uv = g.getAttribute('uv') as THREE.BufferAttribute;
  const dims = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]];
  for (let f = 0; f < 6; f++) for (let k = 0; k < 4; k++) {
    const i = f * 4 + k;
    uv.setXY(i, uv.getX(i) * dims[f][0], uv.getY(i) * dims[f][1]);
  }
  return g;
}

/**
 * Lay the chimney into the assembler. `C` places cabin-local metres in the world; `stone` is the fieldstone
 * material's source (rock_01 maps), `mortar` the chinking's (clay plaster), `flue` the sooted iron's.
 */
export function buildChimney(asm: Assembler, C: THREE.Matrix4, src: { stone: THREE.MeshStandardMaterial; mortar: THREE.MeshStandardMaterial; flue: THREE.MeshStandardMaterial }) {
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler();
  let n = 0;
  const place = (g: THREE.BufferGeometry, x: number, y: number, z: number, yaw: number, tilt: number) => {
    e.set(tilt, yaw, 0, 'YXZ');
    q.setFromEuler(e);
    m.compose(new THREE.Vector3(x, y, z), q, new THREE.Vector3(1, 1, 1)).premultiply(C);
    g.applyMatrix4(m);
    return g;
  };
  const addStone = (L: number, H: number, D: number, x: number, y: number, z: number, yaw: number, seed: number) => {
    const g = place(stoneGeo(L, H, D, seed), x, y, z, yaw, (hash3(seed, 7, 8) - 0.5) * 0.2);
    // (round 3: "pale, evenly coursed buff blocks"): a mixed lot, grey, brown, rust and dark
    const tint = pick(STONE_TINTS, hash3(seed, 9, 10));
    const k = 0.6 + 0.7 * hash3(seed, 11, 12);
    asm.addGeometry(g, src.stone, {
      group: 'cabin',
      edit: () => (_p, t, i) => {
        t[i * 3] *= tint[0] * k; t[i * 3 + 1] *= tint[1] * k; t[i * 3 + 2] *= tint[2] * k;
      },
    }, 'chimney-stone');
    n++;
  };
  /**
   * Stones along one face: `u0..u1` along it, at the face plane `w` (outward normal `nrm` in cabin-local x/z),
   * courses from y0 to y1; bigger stones low down.
   */
  const face = (axis: 'x' | 'z', w: number, sgn: number, u0: number, u1: number, y0: number, y1: number, seed: number) => {
    let y = y0, course = 0;
    while (y < y1 - 0.05) {
      const low = y < 1.2 ? 1 : 0;
      const ch = Math.min(y1 - y, 0.14 + 0.24 * hash3(seed, course, 1) ** 1.5 + 0.1 * low);
      // alternate the starting joint so the joints do not line up course to course
      let u = u0 - (course % 2 ? 0.12 + 0.1 * hash3(seed, course, 2) : 0.02);
      let s = 0;
      while (u < u1 - 0.04) {
        const L = Math.min(u1 + 0.03 - u, 0.12 + 0.44 * hash3(seed, course * 31 + s, 3) ** 1.3 + 0.14 * low);
        const Lc = Math.max(L, 0.08);
        const H = ch - 0.015 - 0.07 * hash3(seed, course * 31 + s, 4) ** 2;
        const D = 0.16 + 0.08 * hash3(seed, course * 31 + s, 5);
        const out = 0.002 + 0.045 * hash3(seed, course * 31 + s, 6) ** 1.5;
        const uc = Math.max(u0 + 0.02, Math.min(u1 - 0.02, u + Lc / 2));
        const yc = y + ch / 2 + (hash3(seed, course * 31 + s, 7) - 0.5) * 0.045;
        const id = seed * 1000 + course * 31 + s;
        if (axis === 'z') addStone(Lc, H, D, uc, yc, w + sgn * (out - D * 0.32), sgn > 0 ? 0 : Math.PI, id);
        else addStone(Lc, H, D, w + sgn * (out - D * 0.32), yc, uc, sgn > 0 ? Math.PI / 2 : -Math.PI / 2, id);
        u += Lc + 0.02 + 0.015 * hash3(seed, course * 31 + s, 8);
        s++;
      }
      y += ch;
      course++;
    }
  };
  CHIMNEY_BOXES.forEach((b, k) => {
    face('z', b.z1, 1, b.x0 + 0.02, b.x1, b.y0, b.y1, 11 + k);
    face('z', b.z0, -1, b.x0 + 0.02, b.x1, b.y0, b.y1, 21 + k);
    face('x', b.x1, 1, b.z0, b.z1, b.y0, b.y1, 31 + k);
    if (b.y1 > INNER_FROM) face('x', b.x0, -1, b.z0, b.z1, Math.max(b.y0, INNER_FROM), b.y1, 41 + k);
    // the mortar core, set back ~3 cm behind the stone faces (the joints read deep and dark)
    const core = place(boxGeo(b.x1 - b.x0 - 0.06, b.y1 - b.y0, b.z1 - b.z0 - 0.06), (b.x0 + b.x1) / 2 - 0.0, (b.y0 + b.y1) / 2, (b.z0 + b.z1) / 2, 0, 0);
    asm.addGeometry(core, src.mortar, { group: 'cabin' }, 'chimney-mortar');
    // flat cap stones on the shoulder steps, sloping off the rain
    if (k < CHIMNEY_BOXES.length - 1) {
      const nb = CHIMNEY_BOXES[k + 1];
      const caps: Array<[number, number, number, number]> = [
        [(nb.x1 + b.x1) / 2, 0, b.x1 - nb.x1 + 0.08, b.z1 - b.z0],
        [(b.x0 + nb.x1) / 2, (nb.z1 + b.z1) / 2, nb.x1 - b.x0, b.z1 - nb.z1 + 0.08],
        [(b.x0 + nb.x1) / 2, (nb.z0 + b.z0) / 2, nb.x1 - b.x0, nb.z0 - b.z0 + 0.08],
      ];
      caps.forEach(([cx, cz, sx, sz], j) => {
        const pieces = Math.max(1, Math.round(Math.max(sx, sz) / 0.42));
        for (let i = 0; i < pieces; i++) {
          const along = sz > sx;
          const L = (along ? sz : sx) / pieces;
          const ux = along ? cx : cx - sx / 2 + L * (i + 0.5), uz = along ? cz - sz / 2 + L * (i + 0.5) : cz;
          const g = place(stoneGeo(along ? sx + 0.04 : L - 0.02, 0.07, along ? L - 0.02 : sz + 0.04, 700 + k * 50 + j * 10 + i),
            ux, b.y1 + 0.02, uz, 0, j === 0 ? 0.12 : 0);
          const tint = pick(STONE_TINTS, hash3(k, j, i));
          asm.addGeometry(g, src.stone, { group: 'cabin', edit: () => (_p, t, ii) => { t[ii * 3] *= tint[0]; t[ii * 3 + 1] *= tint[1]; t[ii * 3 + 2] *= tint[2]; } }, 'chimney-stone');
        }
      });
    }
  });
  // the crown: four flat stones round the flue, overhanging a little
  const top = CHIMNEY_BOXES[CHIMNEY_BOXES.length - 1];
  const cx = (top.x0 + top.x1) / 2, yTop = top.y1;
  const crown: Array<[number, number, number, number]> = [
    [cx, top.z1 - 0.11, top.x1 - top.x0 + 0.1, 0.3], [cx, top.z0 + 0.11, top.x1 - top.x0 + 0.1, 0.3],
    [top.x0 + 0.08, 0, 0.22, top.z1 - top.z0 - 0.3], [top.x1 - 0.08, 0, 0.22, top.z1 - top.z0 - 0.3],
  ];
  crown.forEach(([x, z, sx, sz], i) => {
    const g = place(stoneGeo(sx, 0.1, sz, 900 + i), x, yTop + 0.05, z, 0, 0);
    const k = 0.8 + 0.2 * hash3(i, 3, 3);
    asm.addGeometry(g, src.stone, { group: 'cabin', edit: () => (_p, t, ii) => { t[ii * 3] *= k; t[ii * 3 + 1] *= k; t[ii * 3 + 2] *= k; } }, 'chimney-stone');
  });
  const flue = place(boxGeo(0.26, 0.14, 0.3), cx, yTop + 0.02, 0, 0, 0);
  asm.addGeometry(flue, src.flue, { group: 'cabin', groundDirt: false }, 'chimney-flue');
  return n;
}
