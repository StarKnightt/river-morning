import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { LAYER_NO_REFLECT, LAYER_SHADOW } from '../core/layers';
import { ALIAS_TINT, type CampMaterials } from './materials';

export type GroundY = (x: number, z: number) => number;

/**
 * Edit of one baked vertex in world space: `p` holds x, y, z at i*3 (writable), `tint` rgb at i*3 (multiplied in).
 * `h` is the height above the rendered ground before the edit; `uv` (writable) at i*2.
 */
export type VertexEdit = (p: Float32Array, tint: Float32Array, i: number, h: number, uv: Float32Array) => void;

/**
 * One connected piece of a baked mesh (triangles joined through shared corners). Bounds in the space of the `ref`
 * node passed to `add` (the GLB's own frame: cabin-local for the cabin), in world space for `addGeometry`.
 */
export interface PieceInfo {
  min: THREE.Vector3;
  max: THREE.Vector3;
  /** a stable 0..1 random per piece (from its bounds) */
  seed: number;
}

export interface AddOptions {
  /** merged group the pieces join ('cabin', 'campfire', 'dock'); one mesh per group and material */
  group: string;
  /** main view and sun shadows only: not mirrored in the pool, not in the AO pre-pass (tiny props) */
  small?: boolean;
  /** only meshes under pivot nodes whose name passes */
  filter?: (pivot: string) => boolean;
  /** per-vertex edit (draping footings onto the ground, soot, splash dirt) for a pivot and source material */
  edit?: (pivot: string, material: string) => VertexEdit | null;
  /**
   * per connected piece in world space, after `edit`: a vertex edit for that piece, `false` to drop it, null to
   * leave it (a log of a wall, a shake of the roof, a plank of the dock)
   */
  pieces?: (pivot: string, material: string) => ((piece: PieceInfo) => VertexEdit | false | null) | null;
  /** draw with another camp material than the source's own (same maps, its own calibration) */
  remap?: (pivot: string, material: string) => string | null;
  /** dirty the bottom few centimetres where it meets the ground (not for the dock: its posts stand in water) */
  groundDirt?: boolean;
  /** sky light reaching a vertex (world x, y, z after the edits), 0..1, multiplied into the contact occlusion */
  occlude?: (x: number, y: number, z: number) => number;
  /**
   * keep only the connected pieces (triangles joined through shared corners) whose bounds pass, in the space of
   * `ref`: one stone of a mesh of several, whatever its triangles' extents
   */
  piece?: (min: THREE.Vector3, max: THREE.Vector3) => boolean;
}

interface Bake { key: string; group: string; small: boolean; src: THREE.MeshStandardMaterial; name: string | null; geo: THREE.BufferGeometry }

/** The pivot node of a glTF mesh: the Blender lane put every mesh under '<Name>' as '<Name>_mesh'. */
export function pivotOf(o: THREE.Object3D): THREE.Object3D {
  let p: THREE.Object3D = o;
  while (p.parent && ((p as THREE.Mesh).isMesh || /_mesh(_\d+)?$/.test(p.name))) p = p.parent;
  return p;
}

export function findNode(root: THREE.Object3D, name: string): THREE.Object3D {
  const n = root.getObjectByName(name);
  if (!n) throw new Error(`camp: node ${name} not found`);
  return n;
}

const _rel = new THREE.Matrix4(), _inv = new THREE.Matrix4();

/**
 * Float copy of a (quantized) glTF primitive with `m` applied; position, normal, uv and a 32-bit index. A
 * KHR_texture_transform left by uv quantization is baked into the uvs (the node materials sample raw uv).
 */
function bake(mesh: THREE.Mesh, m: THREE.Matrix4): THREE.BufferGeometry {
  const src = mesh.geometry;
  const g = new THREE.BufferGeometry();
  for (const name of ['position', 'normal', 'uv'] as const) {
    const a = src.getAttribute(name) as THREE.BufferAttribute | undefined;
    if (!a) throw new Error(`camp: ${mesh.name} has no ${name}`);
    const n = a.itemSize, out = new Float32Array(a.count * n);
    for (let i = 0; i < a.count; i++) {
      out[i * n] = a.getX(i);
      out[i * n + 1] = a.getY(i);
      if (n > 2) out[i * n + 2] = a.getZ(i);
    }
    g.setAttribute(name, new THREE.BufferAttribute(out, n));
  }
  const idx = src.index;
  if (idx) g.setIndex(new THREE.BufferAttribute(Uint32Array.from(idx.array as ArrayLike<number>), 1));
  else g.setIndex(new THREE.BufferAttribute(Uint32Array.from({ length: src.getAttribute('position').count }, (_, i) => i), 1));
  g.applyMatrix4(m);
  const map = (mesh.material as THREE.MeshStandardMaterial).map;
  if (map && (map.offset.x !== 0 || map.offset.y !== 0 || map.repeat.x !== 1 || map.repeat.y !== 1 || map.rotation !== 0)) {
    map.updateMatrix();
    const uv = g.getAttribute('uv') as THREE.BufferAttribute;
    const v = new THREE.Vector3();
    for (let i = 0; i < uv.count; i++) {
      v.set(uv.getX(i), uv.getY(i), 1).applyMatrix3(map.matrix);
      uv.setXY(i, v.x, v.y);
    }
  }
  return g;
}

/** Connected pieces: vertices at the same position (to 0.1 mm: split normals and uvs) count as one corner. */
function components(g: THREE.BufferGeometry) {
  const p = g.getAttribute('position') as THREE.BufferAttribute;
  const corner = new Int32Array(p.count);
  const ids = new Map<string, number>();
  for (let i = 0; i < p.count; i++) {
    const key = `${Math.round(p.getX(i) * 1e4)},${Math.round(p.getY(i) * 1e4)},${Math.round(p.getZ(i) * 1e4)}`;
    let c = ids.get(key);
    if (c === undefined) ids.set(key, (c = ids.size));
    corner[i] = c;
  }
  const parent = Int32Array.from({ length: ids.size }, (_, i) => i);
  const find = (a: number) => {
    while (parent[a] !== a) a = parent[a] = parent[parent[a]];
    return a;
  };
  const src = g.index!.array as Uint32Array;
  for (let t = 0; t < src.length; t += 3) {
    const a = find(corner[src[t]]);
    parent[find(corner[src[t + 1]])] = a;
    parent[find(corner[src[t + 2]])] = a;
  }
  // dense piece ids in order of first appearance
  const dense = new Map<number, number>();
  const piece = new Int32Array(p.count);
  for (let i = 0; i < p.count; i++) {
    const r = find(corner[i]);
    let d = dense.get(r);
    if (d === undefined) dense.set(r, (d = dense.size));
    piece[i] = d;
  }
  const n = dense.size;
  const lo = new Float32Array(n * 3).fill(Infinity), hi = new Float32Array(n * 3).fill(-Infinity);
  for (let i = 0; i < p.count; i++) {
    const k = piece[i] * 3;
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    if (x < lo[k]) lo[k] = x;
    if (y < lo[k + 1]) lo[k + 1] = y;
    if (z < lo[k + 2]) lo[k + 2] = z;
    if (x > hi[k]) hi[k] = x;
    if (y > hi[k + 1]) hi[k + 1] = y;
    if (z > hi[k + 2]) hi[k + 2] = z;
  }
  return { piece, n, lo, hi };
}

/** Bounds of the connected pieces of the meshes under `root` whose pivot and material pass, in the space of `ref`. */
export function piecesOf(root: THREE.Object3D, ref: THREE.Object3D, filter: (pivot: string, material: string) => boolean) {
  const out: Array<{ min: THREE.Vector3; max: THREE.Vector3 }> = [];
  const inv = new THREE.Matrix4().copy(ref.matrixWorld).invert(), rel = new THREE.Matrix4();
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !filter(pivotOf(mesh).name, (mesh.material as THREE.Material).name)) return;
    const geo = bake(mesh, rel.multiplyMatrices(inv, mesh.matrixWorld));
    const c = components(geo);
    for (let k = 0; k < c.n; k++) {
      out.push({ min: new THREE.Vector3(c.lo[k * 3], c.lo[k * 3 + 1], c.lo[k * 3 + 2]), max: new THREE.Vector3(c.hi[k * 3], c.hi[k * 3 + 1], c.hi[k * 3 + 2]) });
    }
  });
  return out;
}

/** keep only the triangles whose first corner's piece passes */
function dropPieces(g: THREE.BufferGeometry, piece: Int32Array, keep: (p: number) => boolean) {
  const src = g.index!.array as Uint32Array;
  const out: number[] = [];
  for (let t = 0; t < src.length; t += 3) if (keep(piece[src[t]])) out.push(src[t], src[t + 1], src[t + 2]);
  g.setIndex(new THREE.BufferAttribute(Uint32Array.from(out), 1));
  return out.length > 0;
}

const sstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** a stable 0..1 hash of three numbers (cm resolution) */
export function hash3(x: number, y: number, z: number) {
  let h = (Math.imul(Math.round(x * 100) | 0, 374761393) + Math.imul(Math.round(y * 100) | 0, 668265263)
    + Math.imul(Math.round(z * 100) | 0, 1274126177)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/**
 * Collects baked pieces and merges them: one mesh per group x material x layer set. Every mesh ends up with an
 * identity model matrix (the underwater hook and the shadow warp read positionLocal as world space).
 */
export class Assembler {
  private bakes: Bake[] = [];
  triangles = 0;

  constructor(private mats: CampMaterials, private groundY: GroundY) {}

  /**
   * Bake the meshes under `root` so that the node `ref` lands on `world`: final = world x inverse(ref) x mesh.
   * `root` and `ref` must have current world matrices (a glTF scene at the origin).
   */
  add(root: THREE.Object3D, ref: THREE.Object3D, world: THREE.Matrix4, o: AddOptions) {
    _inv.copy(ref.matrixWorld).invert();
    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      const pivot = pivotOf(mesh).name;
      if (o.filter && !o.filter(pivot)) return;
      const src = mesh.material as THREE.MeshStandardMaterial;
      _rel.multiplyMatrices(_inv, mesh.matrixWorld);
      const geo = bake(mesh, _rel);
      if (o.piece && !this.keepRef(geo, o.piece)) return;
      const c = o.pieces ? components(geo) : null;
      geo.applyMatrix4(world);
      this.addBaked(geo, src, o, pivot, c);
    });
  }

  /** Add a geometry already in world space (position, normal, uv; indexed), drawn with `src`'s camp material. */
  addGeometry(geo: THREE.BufferGeometry, src: THREE.MeshStandardMaterial, o: AddOptions, pivot = 'generated') {
    if (!geo.index) geo.setIndex(new THREE.BufferAttribute(Uint32Array.from({ length: geo.getAttribute('position').count }, (_, i) => i), 1));
    else if (!(geo.index.array instanceof Uint32Array)) geo.setIndex(new THREE.BufferAttribute(Uint32Array.from(geo.index.array as ArrayLike<number>), 1));
    this.addBaked(geo, src, o, pivot, o.pieces ? components(geo) : null);
  }

  private keepRef(geo: THREE.BufferGeometry, test: (min: THREE.Vector3, max: THREE.Vector3) => boolean) {
    const c = components(geo);
    const pass: boolean[] = [];
    for (let k = 0; k < c.n; k++) {
      pass.push(test(new THREE.Vector3(c.lo[k * 3], c.lo[k * 3 + 1], c.lo[k * 3 + 2]), new THREE.Vector3(c.hi[k * 3], c.hi[k * 3 + 1], c.hi[k * 3 + 2])));
    }
    return dropPieces(geo, c.piece, (p) => pass[p]);
  }

  private addBaked(geo: THREE.BufferGeometry, src: THREE.MeshStandardMaterial, o: AddOptions, pivot: string, c: ReturnType<typeof components> | null) {
    const factory = o.pieces?.(pivot, src.name) ?? null;
    let perPiece: Array<VertexEdit | null> | null = null;
    let piece: Int32Array | null = null;
    if (factory && c) {
      piece = c.piece;
      perPiece = [];
      const drop: boolean[] = [];
      for (let k = 0; k < c.n; k++) {
        const min = new THREE.Vector3(c.lo[k * 3], c.lo[k * 3 + 1], c.lo[k * 3 + 2]);
        const max = new THREE.Vector3(c.hi[k * 3], c.hi[k * 3 + 1], c.hi[k * 3 + 2]);
        const seed = hash3(min.x + max.x, min.y + max.y * 0.37, min.z + max.z);
        const r = factory({ min, max, seed });
        drop.push(r === false);
        perPiece.push(r || null);
      }
      if (drop.some((d) => d) && !dropPieces(geo, piece, (p) => !drop[p])) return;
    }
    this.decorate(geo, src.name, o.edit?.(pivot, src.name) ?? null, perPiece, piece, o.groundDirt !== false, o.occlude ?? null);
    const small = !!o.small;
    const name = o.remap?.(pivot, src.name) ?? null;
    const mat = this.mats.get(src, name);
    this.bakes.push({ key: `${o.group}|${mat.name}|${small ? 's' : 'm'}`, group: o.group, small, src, name, geo });
  }

  /** aCamp = (tint rgb, ground-contact occlusion) from the height above the rendered ground */
  private decorate(geo: THREE.BufferGeometry, srcName: string, edit: VertexEdit | null, perPiece: Array<VertexEdit | null> | null,
    piece: Int32Array | null, dirt: boolean, occlude: ((x: number, y: number, z: number) => number) | null) {
    const pos = geo.getAttribute('position') as THREE.BufferAttribute;
    const p = pos.array as Float32Array;
    const uvA = geo.getAttribute('uv') as THREE.BufferAttribute;
    const uv = uvA.array as Float32Array;
    const n = pos.count;
    const tint = new Float32Array(n * 3).fill(1);
    const at = ALIAS_TINT[srcName];
    if (at) for (let i = 0; i < n; i++) tint.set(at, i * 3);
    const out = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      let h = p[i * 3 + 1] - this.groundY(p[i * 3], p[i * 3 + 2]);
      if (edit) {
        edit(p, tint, i, h, uv);
        h = p[i * 3 + 1] - this.groundY(p[i * 3], p[i * 3 + 2]);
      }
      const pe = perPiece && piece ? perPiece[piece[i]] : null;
      if (pe) {
        pe(p, tint, i, h, uv);
        h = p[i * 3 + 1] - this.groundY(p[i * 3], p[i * 3 + 2]);
      }
      let r = tint[i * 3], g = tint[i * 3 + 1], b = tint[i * 3 + 2];
      if (dirt) {
        // soil splashed onto the bottom few centimetres
        const d = sstep(0.0, 0.22, h);
        r *= 0.78 + 0.22 * d; g *= 0.76 + 0.24 * d; b *= 0.7 + 0.3 * d;
      }
      out[i * 4] = r;
      out[i * 4 + 1] = g;
      out[i * 4 + 2] = b;
      // less sky and bounce light where it meets the ground (GTAO adds the larger-scale shade)
      out[i * 4 + 3] = (0.5 + 0.5 * sstep(0.0, 0.45, h)) * (occlude ? occlude(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]) : 1);
    }
    pos.needsUpdate = true;
    uvA.needsUpdate = true;
    geo.setAttribute('aCamp', new THREE.BufferAttribute(out, 4));
  }

  /** Merge into meshes; `filter` picks groups. The pieces are consumed. */
  build(filter: (group: string) => boolean = () => true): THREE.Mesh[] {
    const byKey = new Map<string, Bake[]>();
    const rest: Bake[] = [];
    for (const b of this.bakes) {
      if (!filter(b.group)) {
        rest.push(b);
        continue;
      }
      let a = byKey.get(b.key);
      if (!a) byKey.set(b.key, (a = []));
      a.push(b);
    }
    this.bakes = rest;
    const meshes: THREE.Mesh[] = [];
    for (const [key, list] of byKey) {
      for (const b of list) for (const k of Object.keys(b.geo.attributes)) if (!['position', 'normal', 'uv', 'aCamp'].includes(k)) b.geo.deleteAttribute(k);
      const geo = list.length === 1 ? list[0].geo : mergeGeometries(list.map((b) => b.geo), false);
      if (!geo) throw new Error(`camp: merge failed for ${key}`);
      geo.computeBoundingBox();
      geo.computeBoundingSphere();
      const mesh = new THREE.Mesh(geo, this.mats.get(list[0].src, list[0].name));
      mesh.name = `camp-${key.replaceAll('|', '-')}`;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      if (list[0].small) {
        mesh.layers.set(LAYER_NO_REFLECT);
        mesh.layers.enable(LAYER_SHADOW);
      }
      this.triangles += (geo.index ? geo.index.count : geo.getAttribute('position').count) / 3;
      meshes.push(mesh);
    }
    return meshes;
  }
}
