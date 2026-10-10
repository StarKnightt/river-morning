import * as THREE from 'three/webgpu';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { Assembler, findNode, hash3, type AddOptions } from './assemble';
import { addCampMaterials, type CampMaterials } from './materials';
import { buildWoodpile, woodAcc } from './firewood';
import { indoorU, ROOM } from './indoor';

/**
 * The room inside the cabin (the user's "cant i open the house?"): an 1890s fishing cabin's one room. A bunk in the
 * back corner with a straw tick, a cream wool blanket with two red bars and a pillow; a cast-iron box stove on a
 * fieldstone hearth against the chimney gable, its pipe up and back through the logs into the stone chimney, split
 * wood stacked beside it; a table under the river window with a bench, the oil lantern and a tin cup and plate on
 * it; a shelf with tins, two jars and the wicker creel; pegs by the door corner with a coat and a hat; a rag rug.
 * Timber, stone, iron and glass are the camp's own maps and programs (no new shaders); the GLB's plank floor and
 * the logs' inner faces are the room. Cabin-local metres, baked into world space with the cabin's matrix.
 */

addCampMaterials({
  cal: {
    // timber out of the weather: brown, darkened and smoothed by hands, never silvered like the walls outside
    RoomWood: { lum: 0.07, sat: 0.6, rough: [0.45, 0.9], bias: [1.08, 0.94, 0.78], vary: 0.12, contrast: 1.35, cavity: 0.5, checks: 0.25 },
    // wool, canvas and felt on the plaster maps (a fine felted grain); their colours are in the vertex tint
    Wool: { lum: 0.05, sat: 0.25, rough: [0.88, 1.0], contrast: 0.8, normal: 0.5 },
    HearthStone: { lum: 0.075, sat: 0.35, rough: [0.6, 1.0], bias: [1.0, 0.97, 0.94], vary: 0.15, contrast: 1.4, cavity: 0.6 },
    Wicker: { lum: 0.11, sat: 0.65, rough: [0.55, 1.0], bias: [1.12, 0.96, 0.7], contrast: 1.4, cavity: 0.5 },
  },
  plain: {
    Tin: [[0.28, 0.27, 0.25], 0.45, 0.85, 0.45],
    // the room's lantern: its own glass, so its glow leaves the porch lantern and the pot's knob alone
    LanternGlassLit: [[0.03, 0.028, 0.024], 0.05, 0.0, 1],
  },
});

const F = ROOM.floor;
type V3 = [number, number, number];

/** A walk blocker (cabin-local box) and how high it stands (the camera's boom keeps clear of it). */
export interface RoomBlock { x: number; z: number; hx: number; hz: number; y0: number; y1: number }

export interface Interior {
  root: THREE.Group;
  /** walk blockers and camera solids, cabin-local */
  blocks: RoomBlock[];
  /** the lantern's flame (world) and its glass */
  lantern: THREE.Vector3;
  lanternGlass: THREE.MeshStandardNodeMaterial;
  /**
   * mud chinking behind the logs, facing into the room only (the GLB's chinking faces out: from inside the gaps between
   * the round logs showed the yard); always drawn, a sun shadow caster, invisible from outside
   */
  backing: THREE.Mesh[];
  triangles: number;
}

/** box with uvs in metres (v along axis `along`: the grain) */
function box(sx: number, sy: number, sz: number, along = 0, k = 1): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(sx, sy, sz);
  planarUV(g, along, k);
  return g;
}
function planarUV(g: THREE.BufferGeometry, along: number, k: number, seed = 0) {
  const p = g.getAttribute('position'), n = g.getAttribute('normal'), uv = g.getAttribute('uv');
  const ou = hash3(seed, 1, 2) * 3, ov = hash3(seed, 3, 4) * 3;
  for (let i = 0; i < p.count; i++) {
    const a = [Math.abs(n.getX(i)), Math.abs(n.getY(i)), Math.abs(n.getZ(i))];
    const ax = a[0] > a[1] && a[0] > a[2] ? 0 : a[1] > a[2] ? 1 : 2;
    let [u, v] = [0, 1, 2].filter((c) => c !== ax);
    if (u === along) [u, v] = [v, u];
    const P = [p.getX(i), p.getY(i), p.getZ(i)];
    uv.setXY(i, P[u] * k + ou, P[v] * k + ov);
  }
}
/** rounded, slightly lumpy box (mattress, pillow, coat, stones), centred */
function soft(sx: number, sy: number, sz: number, r: number, seed: number, lump = 0.04): THREE.BufferGeometry {
  let g: THREE.BufferGeometry = new THREE.BoxGeometry(1, 1, 1, 4, 2, 3);
  g.deleteAttribute('normal');
  g.deleteAttribute('uv');
  g = mergeVertices(g);
  const p = g.getAttribute('position') as THREE.BufferAttribute;
  const v = new THREE.Vector3(), c = new THREE.Vector3(), d = new THREE.Vector3();
  const h = [sx / 2 - r, sy / 2 - r, sz / 2 - r].map((x) => Math.max(0.001, x));
  for (let i = 0; i < p.count; i++) {
    v.set(p.getX(i) * sx, p.getY(i) * sy, p.getZ(i) * sz);
    c.set(Math.max(-h[0], Math.min(h[0], v.x)), Math.max(-h[1], Math.min(h[1], v.y)), Math.max(-h[2], Math.min(h[2], v.z)));
    d.subVectors(v, c);
    v.copy(c).addScaledVector(d, r / (d.length() || 1));
    v.multiplyScalar(1 + lump * (hash3(v.x * 9 + seed, v.y * 9, v.z * 9) - 0.5));
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(p.count * 2), 2));
  planarUV(g, 0, 1.5, seed);
  return g;
}
function cyl(r: number, h: number, seg = 16, r2 = r): THREE.BufferGeometry {
  const g = new THREE.CylinderGeometry(r2, r, h, seg);
  const uv = g.getAttribute('uv');
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * 2 * Math.PI * r, uv.getY(i) * h);
  return g;
}

/**
 * Build the room into its own merged meshes (one per material, the camp's programs). `C` places cabin-local metres,
 * `cy` the cabin's base height.
 */
export function buildInterior(mats: CampMaterials, C: THREE.Matrix4, cy: number,
  src: { cabin: THREE.Object3D; fire: THREE.Object3D; props: THREE.Object3D }): Interior {
  const yaw = Math.atan2(C.elements[8], C.elements[0]);
  indoorU.frame.value.set(Math.cos(yaw), Math.sin(yaw), C.elements[12], C.elements[14]);
  indoorU.baseY.value = cy;
  const floorY = cy + F;
  // contact shade against the plank floor, not the terrain under the cabin
  const asm = new Assembler(mats, () => floorY);
  const m = (name: string, root: THREE.Object3D) => {
    let found: THREE.MeshStandardMaterial | null = null;
    root.traverse((o) => {
      const mm = (o as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined;
      if (!found && (o as THREE.Mesh).isMesh && mm?.name === name) found = mm;
    });
    if (!found) throw new Error(`interior: material ${name} not found`);
    return found as THREE.MeshStandardMaterial;
  };
  const S = {
    wood: m('LogWood', src.cabin), wool: m('Chinking', src.cabin), stone: m('PierStone', src.cabin),
    iron: m('SootedIron', src.fire), wicker: m('SplitWood', src.props), glass: m('Lantern_01_glass', src.props),
    bark: m('Bark', src.props), split: m('SplitWood', src.props), ends: m('EndGrain', src.props),
  };
  const blocks: RoomBlock[] = [];
  const M = new THREE.Matrix4(), Q = new THREE.Quaternion(), E = new THREE.Euler(), ONE = new THREE.Vector3(1, 1, 1);
  /** add a cabin-local geometry: position, yaw/pitch/roll, material (source + camp name), tint */
  const put = (g: THREE.BufferGeometry, at: V3, s: THREE.MeshStandardMaterial, as: string, tint: V3 = [1, 1, 1], rot: V3 = [0, 0, 0], small = false) => {
    E.set(rot[0], rot[1], rot[2], 'YXZ');
    M.compose(new THREE.Vector3(...at), Q.setFromEuler(E), ONE).premultiply(C);
    g.applyMatrix4(M);
    const o: AddOptions = {
      group: 'interior', small, groundDirt: false, remap: () => as,
      edit: () => (_p, t, i) => {
        t[i * 3] *= tint[0]; t[i * 3 + 1] *= tint[1]; t[i * 3 + 2] *= tint[2];
      },
    };
    asm.addGeometry(g, s, o, `interior-${as}`);
  };
  const wood = (sx: number, sy: number, sz: number, at: V3, along = 0, tone = 1, rot: V3 = [0, 0, 0]) => {
    const k = tone * (0.86 + 0.28 * hash3(at[0], at[1], at[2]));
    put(box(sx, sy, sz, along, 1.1), at, S.wood, 'RoomWood', [k, k * 0.98, k * 0.95], rot);
  };
  const block = (x0: number, x1: number, z0: number, z1: number, y1: number, y0 = F) =>
    blocks.push({ x: (x0 + x1) / 2, z: (z0 + z1) / 2, hx: (x1 - x0) / 2, hz: (z1 - z0) / 2, y0, y1 });

  // ---- the bunk in the back corner, its head against the west gable: pole posts, pegged rails, a straw tick
  {
    const x0 = -2.64, x1 = -0.8, z0 = -2.14, z1 = -1.3;
    for (const [x, z, h] of [[x0 + 0.05, z0 + 0.05, 0.98], [x0 + 0.05, z1 - 0.05, 0.98], [x1 - 0.05, z0 + 0.05, 0.72], [x1 - 0.05, z1 - 0.05, 0.72]]) {
      wood(0.09, h, 0.09, [x, F + h / 2, z], 1, 0.9);
    }
    wood(x1 - x0 - 0.1, 0.15, 0.05, [(x0 + x1) / 2, F + 0.36, z1 - 0.05], 0);
    wood(x1 - x0 - 0.1, 0.15, 0.05, [(x0 + x1) / 2, F + 0.36, z0 + 0.05], 0);
    wood(0.05, 0.15, z1 - z0 - 0.1, [x1 - 0.05, F + 0.36, (z0 + z1) / 2], 2);
    wood(0.05, 0.3, z1 - z0 - 0.1, [x0 + 0.05, F + 0.62, (z0 + z1) / 2], 2, 0.95);
    for (let k = 0; k < 7; k++) wood(0.1, 0.02, z1 - z0 - 0.14, [x0 + 0.2 + k * 0.25, F + 0.42, (z0 + z1) / 2], 2, 0.8);
    // the straw tick, sagging a little in the middle; the pillow at the head
    const tick = soft(x1 - x0 - 0.14, 0.13, z1 - z0 - 0.12, 0.05, 3, 0.05);
    put(tick, [(x0 + x1) / 2, F + 0.5, (z0 + z1) / 2], S.wool, 'Wool', [3.9, 3.4, 2.6]);
    put(soft(0.36, 0.1, z1 - z0 - 0.26, 0.045, 7, 0.08), [x0 + 0.3, F + 0.6, (z0 + z1) / 2], S.wool, 'Wool', [4.4, 4.1, 3.5], [0, 0.08, 0.05]);
    // the blanket: cream wool with two red bars toward its foot (a trade point blanket), drawn up to the pillow and hanging over the front rail
    const bx0 = x0 + 0.5, bx1 = x1 - 0.06, top = F + 0.575;
    const bands: Array<[number, number, V3]> = [
      [bx0, bx1 - 0.42, [3.0, 2.72, 2.2]], [bx1 - 0.42, bx1 - 0.32, [3.3, 0.62, 0.38]], [bx1 - 0.32, bx1 - 0.24, [3.0, 2.72, 2.2]],
      [bx1 - 0.24, bx1 - 0.14, [3.3, 0.62, 0.38]], [bx1 - 0.14, bx1, [3.0, 2.72, 2.2]],
    ];
    for (const [a, b, t] of bands) {
      put(soft(b - a, 0.03, z1 - z0 - 0.02, 0.012, a * 7, 0.08), [(a + b) / 2, top, (z0 + z1) / 2 + 0.01], S.wool, 'Wool', t);
      put(soft(b - a, 0.26, 0.025, 0.01, a * 9, 0.1), [(a + b) / 2, top - 0.13, z1 + 0.012], S.wool, 'Wool', t, [0.06, 0, 0]);
    }
    block(x0, x1, z0, z1 + 0.04, F + 0.98);
  }

  // ---- the stove on its hearth against the chimney gable (the stone chimney stands outside at x 3.0..3.98)
  {
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const sx = 0.32 + 0.06 * hash3(i, j, 1), sz = 0.36 + 0.06 * hash3(i, j, 2);
        put(soft(sx, 0.05, sz, 0.02, i * 3 + j, 0.06), [1.82 + i * 0.36 + (hash3(i, j, 3) - 0.5) * 0.03, F + 0.022, -0.42 + j * 0.42], S.stone, 'HearthStone',
          [0.8 + 0.4 * hash3(i, j, 4), 0.78 + 0.36 * hash3(i, j, 4), 0.74 + 0.3 * hash3(i, j, 4)], [0, (hash3(i, j, 5) - 0.5) * 0.2, 0]);
      }
    }
    // a guard of flat stones on the logs behind it
    for (let r = 0; r < 3; r++) {
      for (let j = 0; j < 3; j++) {
        put(soft(0.07, 0.36, 0.38, 0.025, 20 + r * 3 + j, 0.06), [2.62, F + 0.24 + r * 0.38, -0.4 + j * 0.4 + (r % 2) * 0.05], S.stone, 'HearthStone',
          [0.75 + 0.4 * hash3(r, j, 6), 0.72 + 0.36 * hash3(r, j, 6), 0.7 + 0.3 * hash3(r, j, 6)]);
      }
    }
    const I = (g: THREE.BufferGeometry, at: V3, rot: V3 = [0, 0, 0], k = 1) => put(g, at, S.iron, 'SootedIron', [k, k, k], rot);
    for (const [x, z] of [[1.96, -0.28], [1.96, 0.28], [2.38, -0.28], [2.38, 0.28]]) I(box(0.05, 0.16, 0.05), [x, F + 0.12, z]);
    I(box(0.5, 0.5, 0.64), [2.17, F + 0.45, 0]);
    I(box(0.55, 0.03, 0.7), [2.17, F + 0.715, 0], [0, 0, 0], 1.2);
    I(box(0.02, 0.24, 0.3), [1.91, F + 0.46, 0.02], [0, 0, 0], 1.5);
    I(box(0.03, 0.03, 0.09), [1.89, F + 0.5, 0.12], [0, 0, 0], 2);
    I(box(0.05, 0.03, 0.6), [1.9, F + 0.25, 0]);
    I(cyl(0.065, 1.02), [2.2, F + 1.24, 0]);
    I(new THREE.SphereGeometry(0.07, 14, 8), [2.2, F + 1.76, 0]);
    I(cyl(0.065, 0.72), [2.56, F + 1.76, 0], [0, 0, Math.PI / 2]);
    I(cyl(0.082, 0.03), [2.65, F + 1.76, 0], [0, 0, Math.PI / 2], 1.3);
    block(1.88, 2.46, -0.36, 0.36, F + 0.75);
    blocks.push({ x: 2.2, z: 0, hx: 0.08, hz: 0.08, y0: F, y1: F + 1.84 }, { x: 2.45, z: 0, hx: 0.25, hz: 0.08, y0: F + 1.66, y1: F + 1.86 });
    // split wood stacked between the stove and the front corner, along the wall
    const pile = woodAcc();
    const wm = new THREE.Matrix4().compose(new THREE.Vector3(2.42, F, 0.95), new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2), ONE).premultiply(C);
    buildWoodpile(pile, wm, { x0: -0.3, x1: 0.3, height: 0.42, seed: 17 });
    pile.bark.addTo(asm, S.bark, 'interior', 'PileBark', 'interior-pile-bark', false, false);
    pile.split.addTo(asm, S.split, 'interior', 'PileSplit', 'interior-pile-split', false, false);
    pile.ends.addTo(asm, S.ends, 'interior', 'PileEnds', 'interior-pile-ends', false, false);
    block(2.2, 2.66, 0.6, 1.3, F + 0.5);
  }

  // ---- the table under the river window, the bench before it; the lantern, a tin cup and plate
  const tx0 = 0.8, tx1 = 1.92, tz0 = 1.44, tz1 = 2.1, tTop = F + 0.76;
  {
    wood(tx1 - tx0, 0.045, tz1 - tz0, [(tx0 + tx1) / 2, tTop - 0.0225, (tz0 + tz1) / 2], 0, 1.05);
    for (const [x, z] of [[tx0 + 0.07, tz0 + 0.07], [tx0 + 0.07, tz1 - 0.07], [tx1 - 0.07, tz0 + 0.07], [tx1 - 0.07, tz1 - 0.07]]) wood(0.07, 0.715, 0.07, [x, F + 0.3575, z], 1, 0.9);
    wood(tx1 - tx0 - 0.2, 0.09, 0.03, [(tx0 + tx1) / 2, tTop - 0.09, tz0 + 0.07], 0, 0.85);
    wood(0.03, 0.09, tz1 - tz0 - 0.2, [tx0 + 0.07, tTop - 0.09, (tz0 + tz1) / 2], 2, 0.85);
    wood(0.03, 0.09, tz1 - tz0 - 0.2, [tx1 - 0.07, tTop - 0.09, (tz0 + tz1) / 2], 2, 0.85);
    block(tx0, tx1, tz0, tz1, tTop);
    const bz0 = 0.98, bz1 = 1.24;
    wood(1.0, 0.05, bz1 - bz0, [1.36, F + 0.445, (bz0 + bz1) / 2], 0, 1.0);
    for (const x of [0.95, 1.77]) wood(0.05, 0.42, 0.22, [x, F + 0.21, (bz0 + bz1) / 2], 1, 0.9);
    block(0.86, 1.86, bz0, bz1, F + 0.47);
    const T = (g: THREE.BufferGeometry, at: V3, tint: V3, rot: V3 = [0, 0, 0]) => put(g, at, S.iron, 'Tin', tint, rot, true);
    T(cyl(0.04, 0.085, 14, 0.043), [1.42, tTop + 0.0425, 1.63], [1, 1, 1]);
    T(cyl(0.11, 0.014, 20), [1.62, tTop + 0.007, 1.86], [0.9, 0.9, 0.88]);
  }
  // the hurricane lantern from the porch set (props.glb), a little larger, its glass its own material
  const lanternBase = new THREE.Vector3(1.08, tTop, 1.8);
  {
    const ln = findNode(src.props, 'Prop_Lantern');
    const s = 1.9;
    const lm = new THREE.Matrix4().compose(lanternBase, new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.6), new THREE.Vector3(s, s, s)).premultiply(C);
    asm.add(src.props, ln, lm, {
      group: 'interior', small: true, groundDirt: false, filter: (pv) => pv === 'Prop_Lantern',
      remap: (_pv, mat) => (mat === 'Lantern_01_glass' ? 'LanternGlassLit' : null),
    });
  }

  // ---- the shelf on the back wall: tins, two jars and the creel
  {
    const sx0 = -0.6, sx1 = 0.8, sz0 = -2.17, sz1 = -1.93, sy = F + 1.42;
    wood(sx1 - sx0, 0.03, sz1 - sz0, [(sx0 + sx1) / 2, sy - 0.015, (sz0 + sz1) / 2], 0, 1.0);
    for (const x of [sx0 + 0.15, sx1 - 0.15]) wood(0.04, 0.2, 0.18, [x, sy - 0.13, sz0 + 0.09], 1, 0.85);
    const T = (g: THREE.BufferGeometry, at: V3, tint: V3, mat = 'Tin', s = S.iron) => put(g, at, s, mat, tint, [0, 0, 0], true);
    const tins: Array<[number, number, number, V3]> = [[-0.48, 0.05, 0.13, [2.1, 0.62, 0.42]], [-0.36, 0.045, 0.1, [1.6, 1.25, 0.5]], [-0.25, 0.04, 0.12, [1, 1, 1]], [-0.15, 0.035, 0.07, [1.2, 1.1, 1.0]]];
    for (const [x, r, h, t] of tins) T(cyl(r, h, 16), [x, sy + h / 2, -2.06], t);
    for (const [x, h] of [[0.0, 0.16], [0.12, 0.13]]) {
      T(cyl(0.048, h, 16), [x, sy + h / 2, -2.05], [1, 1, 1], 'LanternGlass', S.glass);
      T(cyl(0.05, 0.018, 16), [x, sy + h + 0.009, -2.05], [0.8, 0.8, 0.78]);
    }
    // the wicker creel, lid shut, its strap hanging down in front
    put(soft(0.36, 0.2, 0.2, 0.04, 31, 0.03), [0.5, sy + 0.1, -2.05], S.wicker, 'Wicker', [1, 1, 1], [0, 0.04, 0]);
    put(soft(0.37, 0.03, 0.21, 0.012, 32, 0.03), [0.5, sy + 0.21, -2.05], S.wicker, 'Wicker', [0.85, 0.85, 0.85], [0, 0.04, 0]);
    put(box(0.035, 0.28, 0.008, 1, 2), [0.42, sy + 0.08, -1.94], S.wood, 'RoomWood', [0.45, 0.32, 0.22]);
    block(sx0, sx1, sz0, sz1, sy + 0.24, sy - 0.25);
  }

  // ---- pegs on the west wall by the door corner: the coat, the hat
  {
    const wx = -ROOM.ix + 0.01;
    for (const z of [1.1, 1.7]) put(cyl(0.022, 0.13, 10), [wx + 0.06, F + 1.66, z], S.wood, 'RoomWood', [0.9, 0.85, 0.8], [0, 0, -(Math.PI / 2 - 0.25)], true);
    // a sack coat hung by its collar, flaring to the hem
    const coat = soft(0.15, 0.92, 0.5, 0.05, 41, 0.05);
    const p = coat.getAttribute('position') as THREE.BufferAttribute;
    for (let i = 0; i < p.count; i++) {
      const y = p.getY(i), t = (y + 0.46) / 0.92;
      p.setZ(i, p.getZ(i) * (1.08 - 0.3 * t * t));
      p.setX(i, p.getX(i) * (1.0 - 0.35 * t * t) + 0.03 * (1 - t));
    }
    coat.computeVertexNormals();
    put(coat, [wx + 0.1, F + 1.2, 1.1], S.wool, 'Wool', [1.05, 0.8, 0.58]);
    // the hat on the other peg, its crown out from the wall
    put(cyl(0.085, 0.1, 18, 0.075), [wx + 0.13, F + 1.62, 1.7], S.wool, 'Wool', [0.62, 0.58, 0.54], [0, 0, -(Math.PI / 2 - 0.2)]);
    put(cyl(0.16, 0.012, 24), [wx + 0.075, F + 1.635, 1.7], S.wool, 'Wool', [0.58, 0.54, 0.5], [0, 0, -(Math.PI / 2 - 0.2)]);
    block(-ROOM.ix, -ROOM.ix + 0.24, 0.8, 1.38, F + 1.7, F + 0.7);
    block(-ROOM.ix, -ROOM.ix + 0.24, 1.52, 1.88, F + 1.8, F + 1.45);
  }

  // ---- a rag rug in the middle of the floor
  put(soft(1.3, 0.012, 0.85, 0.005, 51, 0.0), [0.15, F + 0.006, -0.35], S.wool, 'Wool', [1.9, 1.1, 0.8], [0, 0.05, 0]);

  const root = new THREE.Group();
  root.name = 'camp-interior';
  for (const mesh of asm.build()) {
    // its shadows only fall inside the closed room: no sun-map casters (the sun reaches in through the door and
    // windows onto the floor)
    mesh.castShadow = false;
    root.add(mesh);
  }
  const lanternGlass = mats.get(S.glass, 'LanternGlassLit');
  // ---- the chinking behind the logs, a few centimetres outside their centre planes, round the door and the windows
  const back = new Assembler(mats, () => -1e3);
  const Y0 = 0.5, YF = 3.12, YS = 3.0;
  const wall = (u0: number, u1: number, y0: number, y1: number, side: 'front' | 'back' | 'west' | 'east') => {
    const g = new THREE.PlaneGeometry(u1 - u0, y1 - y0);
    const uv = g.getAttribute('uv');
    for (let i = 0; i < uv.count; i++) uv.setXY(i, u0 + uv.getX(i) * (u1 - u0), y0 + uv.getY(i) * (y1 - y0));
    const uc = (u0 + u1) / 2, yc = (y0 + y1) / 2;
    const at: Record<typeof side, [V3, number]> = {
      front: [[uc, yc, ROOM.cz + 0.03], Math.PI], back: [[uc, yc, -ROOM.cz - 0.03], 0],
      west: [[-ROOM.cx - 0.03, yc, uc], Math.PI / 2], east: [[ROOM.cx + 0.03, yc, uc], -Math.PI / 2],
    };
    const [p, ry] = at[side];
    M.compose(new THREE.Vector3(...p), Q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), ry), ONE).premultiply(C);
    g.applyMatrix4(M);
    back.addGeometry(g, S.wool, { group: 'backing', groundDirt: false, remap: () => 'Chinking' }, 'interior-backing');
  };
  const D = ROOM.door;
  wall(-ROOM.cx, D.x0 - 0.02, Y0, YF, 'front');
  wall(D.x1 + 0.02, 1.0, Y0, YF, 'front');
  wall(1.8, ROOM.cx, Y0, YF, 'front');
  wall(D.x0 - 0.02, D.x1 + 0.02, D.top + 0.02, YF, 'front');
  wall(1.0, 1.8, Y0, 1.39, 'front');
  wall(1.0, 1.8, 2.07, YF, 'front');
  wall(-ROOM.cx, ROOM.cx, Y0, YF, 'back');
  wall(-ROOM.cz, ROOM.cz, Y0, YS, 'east');
  wall(-ROOM.cz, -0.38, Y0, YS, 'west');
  wall(0.39, ROOM.cz, Y0, YS, 'west');
  wall(-0.38, 0.39, Y0, 1.54, 'west');
  wall(-0.38, 0.39, 2.21, YS, 'west');
  // the ceiling: boards a finger under the roof sheathing, both slopes (inward faces only; with the chinking they close
  // every gap the sun map could find in the roof or the walls)
  const S0 = 4.555, SL = 0.629, ex = ROOM.cx + 0.035, ez = ROOM.cz + 0.035;
  for (const sg of [1, -1]) {
    const pts = [[-ex, S0, 0], [ex, S0, 0], [ex, S0 - SL * ez, sg * ez], [-ex, S0 - SL * ez, sg * ez]].map((q) => new THREE.Vector3(q[0], q[1], q[2]).applyMatrix4(C));
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const len = Math.hypot(ez, SL * ez);
    g.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 2 * ex, 0, 2 * ex, len, 0, len], 2));
    g.setIndex(sg > 0 ? [0, 2, 1, 0, 3, 2] : [0, 1, 2, 0, 2, 3]);
    g.computeVertexNormals();
    if (g.getAttribute('normal').getY(0) > 0) {
      g.setIndex(sg > 0 ? [0, 1, 2, 0, 2, 3] : [0, 2, 1, 0, 3, 2]);
      g.computeVertexNormals();
    }
    back.addGeometry(g, m('Boards', src.cabin), { group: 'backing', groundDirt: false, edit: () => (_p, t, i) => { t[i * 3] *= 0.8; t[i * 3 + 1] *= 0.76; t[i * 3 + 2] *= 0.72; } }, 'interior-ceiling');
  }
  const backing = back.build();
  const lw = new THREE.Vector3(lanternBase.x, lanternBase.y + 0.17, lanternBase.z).applyMatrix4(C);
  return { root, blocks, lantern: lw, lanternGlass, backing, triangles: Math.round(asm.triangles + back.triangles) };
}
