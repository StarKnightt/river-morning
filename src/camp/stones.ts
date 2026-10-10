import * as THREE from 'three/webgpu';
import type { Assets } from '../core/Assets';
import { bakeGeometry } from '../water/Rocks';
import { hash3, type Assembler, type GroundY } from './assemble';

/**
 * Fieldstones from the project's own rock scans (part 3's bank props: Poly Haven CC0), for the cabin's piers, the
 * porch and dock steps, the fire ring and the gravel round the dock's shore posts. Round 1 used the GLB's cut
 * blocks: "cast concrete or cinder block". Each stone is a scan (900-1,200 triangles at LOD1, 400-800 for the small
 * river stones) normalised to a unit box, scaled to the size it stands in for, turned, tilted, sunk into the
 * ground and coloured on its own (grey, buff, rust, dark, pale); drawn with the camp's shared surface program.
 */

export const STONE_IDS = ['rock_moss_01_r03', 'rock_moss_02_r07', 'rock_moss_02_r11', 'namaqualand_boulder_05', 'rock_07', 'stone_01', 'rock_09'] as const;
export type StoneId = (typeof STONE_IDS)[number];
export interface StoneScan { geo: THREE.BufferGeometry; src: THREE.MeshStandardMaterial; size: THREE.Vector3 }
export type StoneScans = Record<StoneId, StoneScan>;

/** Load the scans (already cached by the bank props) at LOD1 where they have one, centred in a unit box. */
export async function loadStoneScans(assets: Assets): Promise<StoneScans> {
  const out = {} as StoneScans;
  await Promise.all(STONE_IDS.map(async (id) => {
    const gltf = await assets.gltf(`models/rocks/${id}.glb`);
    const meshes: THREE.Mesh[] = [];
    gltf.scene.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) meshes.push(o as THREE.Mesh);
    });
    const lod1 = meshes.find((m) => /LOD1/.test(m.name) || /LOD1/.test(m.parent?.name ?? ''));
    const mesh = lod1 ?? meshes[0];
    const geo = bakeGeometry(mesh);
    geo.computeBoundingBox();
    const bb = geo.boundingBox!;
    const c = bb.getCenter(new THREE.Vector3()), s = bb.getSize(new THREE.Vector3());
    geo.translate(-c.x, -c.y, -c.z);
    geo.scale(1 / s.x, 1 / s.y, 1 / s.z);
    out[id] = { geo, src: mesh.material as THREE.MeshStandardMaterial, size: s.clone() };
  }));
  return out;
}

/** stone colours (multiplied into the calibrated scan): grey, buff, rust, dark, pale */
export const STONE_TINTS: Array<[number, number, number]> = [
  [0.97, 0.99, 1.02], [1.16, 1.03, 0.84], [1.22, 0.92, 0.72], [0.72, 0.73, 0.75], [1.22, 1.21, 1.17], [1.05, 1.0, 0.94],
];

export interface StonePlacement {
  id: StoneId;
  x: number;
  z: number;
  /** world height of its bottom, or of its top with `top` */
  y: number;
  top?: boolean;
  /** metres: x along its yaw, y up, z across */
  size: [number, number, number];
  yaw: number;
  /** tilt (rad) about a horizontal axis perpendicular to `tiltDir` (rad, world xz) */
  tilt?: number;
  tiltDir?: number;
  tint: [number, number, number];
  /** soot on the faces toward a point (the fire ring's inner faces) and on the top */
  soot?: { x: number; z: number; k: number };
  /** a heat crack: the plane's normal through the stone's centre */
  crack?: [number, number, number];
  /** the scan's own roll first (rad about its long axis): which face ends up on top */
  roll?: number;
  /**
   * keep the scan's own proportions: size[0] is then the length of its longest horizontal side (laid along the
   * yaw), size[1] a squash of its height (1 = as scanned); a box fitted to an elongated scan made it a block
   */
  aspect?: boolean;
  /** mud trodden onto its top (0..1) */
  mud?: number;
  /** draw with this camp material (the scan's maps, its own calibration) instead of the scan's own */
  as?: string;
  /** sky light reaching it (world x, y, z), multiplied into the contact occlusion */
  occlude?: (x: number, y: number, z: number) => number;
  /**
   * with `top`: a flat-topped step (scene fix, critic polish r1: "a rounded dome sitting on the dirt; you couldn't
   * step on it"): the scan's top reaches this far (m) over `y`, its bottom as far deeper, and everything above `y` is
   * pressed nearly flat, so its top is a worn slab at the walk height
   */
  flatTop?: number;
}

const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion(), _ax = new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0), X = new THREE.Vector3(1, 0, 0);

/** Bake one fieldstone into the assembler's `group` (world space, the scan's own material). */
export function addStone(asm: Assembler, scans: StoneScans, s: StonePlacement, group: string, small = false) {
  const scan = scans[s.id];
  const geo = scan.geo.clone();
  let size = s.size, yaw = s.yaw;
  if (s.aspect) {
    const n = scan.size, k = s.size[0] / Math.max(n.x, n.z);
    size = [n.x * k, n.y * k * s.size[1], n.z * k];
    if (n.z > n.x) yaw += Math.PI / 2;
  }
  const lift = s.top ? s.flatTop ?? 0 : 0;
  // (a flat top grows the scan `lift` taller at both ends: its widest girth stays where it was, under the ground line)
  if (lift > 0) size = [size[0], size[1] + 2 * lift, size[2]];
  const cy = s.top ? s.y + lift - size[1] / 2 : s.y + size[1] / 2;
  _q.setFromAxisAngle(UP, yaw);
  if (s.roll) _q.multiply(_q2.setFromAxisAngle(X, s.roll));
  if (s.tilt) {
    const td = s.tiltDir ?? 0;
    _ax.set(Math.cos(td + Math.PI / 2), 0, Math.sin(td + Math.PI / 2));
    _q.premultiply(_q2.setFromAxisAngle(_ax, s.tilt));
  }
  _m.compose(new THREE.Vector3(s.x, cy, s.z), _q, new THREE.Vector3(...size));
  geo.applyMatrix4(_m);
  if (lift > 0) {
    const pa = geo.getAttribute('position') as THREE.BufferAttribute;
    for (let i = 0; i < pa.count; i++) {
      const y = pa.getY(i);
      if (y > s.y) pa.setY(i, s.y + (y - s.y) * 0.06);
    }
    geo.computeVertexNormals();
  }
  const halfX = Math.max(size[0], size[2]) / 2, halfY = size[1] / 2;
  const topY = lift > 0 ? s.y + lift * 0.06 : cy + halfY;
  const ds = s.soot ? Math.hypot(s.x - s.soot.x, s.z - s.soot.z) : 0;
  asm.addGeometry(geo, scan.src, {
    group, small, occlude: s.occlude, remap: s.as ? () => s.as! : undefined,
    edit: () => (p, t, i) => {
      const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
      let r = s.tint[0], g = s.tint[1], b = s.tint[2];
      if (s.soot) {
        // black on the faces toward the fire and over the top, unevenly
        const dv = Math.hypot(x - s.soot.x, z - s.soot.z);
        const inner = Math.min(1, Math.max(0, (ds - dv) / (0.6 * halfX) + 0.15));
        const top = Math.min(1, Math.max(0, (y - cy) / halfY));
        const k = Math.min(1, Math.max(inner, top * inner * 0.5 + top * 0.5)) * s.soot.k * (0.65 + 0.35 * hash3(x * 7, y * 7, z * 7));
        const f = 1 - 0.8 * k;
        r *= f; g *= f; b *= f;
      }
      if (s.crack) {
        const d = (x - s.x) * s.crack[0] + (y - cy) * s.crack[1] + (z - s.z) * s.crack[2];
        const f = 1 - 0.8 * Math.exp(-((d / 0.007) ** 2));
        r *= f; g *= f; b *= f;
      }
      if (s.mud) {
        // dried mud and grit on the top where boots land, in patches
        const top = Math.min(1, Math.max(0, (y - (topY - 0.11)) / 0.07)) * (0.55 + 0.45 * hash3(x * 9, z * 9, 3)) * s.mud;
        r *= 1 - 0.38 * top; g *= 1 - 0.5 * top; b *= 1 - 0.62 * top;
      }
      t[i * 3] *= r; t[i * 3 + 1] *= g; t[i * 3 + 2] *= b;
    },
  }, `stone-${s.id}`);
}

/** pick from a list by a 0..1 random */
export const pick = <T>(list: readonly T[], r: number) => list[Math.min(list.length - 1, Math.floor(r * list.length))];

/**
 * A pier of one or two fieldstones under a log wall or the porch beam: `top` (world) where the timber rests, its
 * footprint `fx` x `fz` (m) at (x, z), half sunk into the ground. The bottom stone is broad and flat, the top one
 * smaller; both turned and coloured on their own. `packed` (a unit world direction toward the wall): flat stones
 * packed tight in a short stack, set back under the beam into its shadow, the lowest one mostly in the dirt.
 * Returns the number of stones.
 */
export function addPier(asm: Assembler, scans: StoneScans, groundY: GroundY,
  o: { x: number; z: number; top: number; fx: number; fz: number; yaw: number; seed: number; packed?: [number, number]; occlude?: (x: number, y: number, z: number) => number }) {
  const g = groundY(o.x, o.z);
  const H = o.top - g;
  const r = (k: number) => hash3(o.x * 3.1 + k, o.z * 1.7 - k, o.seed * 13 + k);
  const big: StoneId[] = ['rock_moss_01_r03', 'namaqualand_boulder_05', 'rock_moss_02_r07', 'rock_moss_02_r11'];
  const sink = 0.13 + 0.05 * r(1);
  if (o.packed) {
    const flat: StoneId[] = ['namaqualand_boulder_05', 'rock_moss_02_r07', 'namaqualand_boulder_05', 'rock_moss_01_r03'];
    // grey and buff, one weathering: stones gathered from the same bank
    const tints = [STONE_TINTS[0], STONE_TINTS[1], STONE_TINTS[5], STONE_TINTS[3]].map((t) => t.map((c) => c * 0.8) as [number, number, number]);
    // (round 4 wip: three rounded stones still read as a cairn): thinner slabs, one more, each offset and turned a
    // little more like a dry-stacked pier, the lowest broad and mostly in the dirt
    const n = H > 0.4 ? 4 : 3;
    const bury = 0.18 + 0.04 * r(20);
    const total = H + bury;
    const bx = o.x + o.packed[0] * 0.17, bz = o.z + o.packed[1] * 0.17;
    let y = g - bury;
    for (let k = 0; k < n; k++) {
      const th = total * (k === 0 ? 0.4 : 0.6 / (n - 1)) + 0.03;
      const spread = k === 0 ? 1.38 : 1.06 + 0.2 * r(110 + k);
      const last = k === n - 1;
      addStone(asm, scans, {
        id: pick(flat, r(21 + k)), x: bx + (r(30 + k) - 0.5) * 0.07, z: bz + (r(40 + k) - 0.5) * 0.05, y: last ? o.top : y, top: last,
        size: [o.fx * spread * (0.95 + 0.15 * r(50 + k)), th, o.fz * spread * (0.95 + 0.15 * r(60 + k))],
        yaw: o.yaw + (r(70 + k) - 0.5) * 0.7, tint: pick(tints, r(80 + k)), tilt: 0.04 * r(90 + k), tiltDir: r(100 + k) * 6.28,
        occlude: o.occlude,
      }, 'cabin');
      y += th - 0.03;
    }
    return n;
  }
  if (H < 0.3) {
    addStone(asm, scans, {
      id: pick(big, r(2)), x: o.x, z: o.z, y: o.top, top: true, size: [o.fx * (1.25 + 0.2 * r(3)), H + sink, o.fz * (1.2 + 0.2 * r(4))],
      yaw: o.yaw + (r(5) - 0.5) * 0.9, tint: pick(STONE_TINTS, r(6)), tilt: 0.06 * r(7), tiltDir: r(8) * 6.28,
    }, 'cabin');
    return 1;
  }
  const hb = 0.55 * H;
  addStone(asm, scans, {
    id: pick(big, r(2)), x: o.x + (r(9) - 0.5) * 0.08, z: o.z + (r(10) - 0.5) * 0.08, y: g - sink,
    size: [o.fx * (1.4 + 0.25 * r(3)), hb + sink, o.fz * (1.35 + 0.25 * r(4))], yaw: o.yaw + (r(5) - 0.5) * 1.2,
    tint: pick(STONE_TINTS, r(6)), tilt: 0.05 * r(7), tiltDir: r(8) * 6.28,
  }, 'cabin');
  const tops: StoneId[] = ['rock_moss_02_r07', 'namaqualand_boulder_05', 'rock_07', 'rock_moss_02_r11', 'stone_01'];
  addStone(asm, scans, {
    id: pick(tops, r(11)), x: o.x, z: o.z, y: o.top, top: true, size: [o.fx * (1.0 + 0.2 * r(12)), 0.5 * H + 0.02, o.fz * (0.95 + 0.2 * r(13))],
    yaw: o.yaw + (r(14) - 0.5) * 1.4, tint: pick(STONE_TINTS, r(15)), tilt: 0.05 * r(16), tiltDir: r(17) * 6.28,
  }, 'cabin');
  return 2;
}
