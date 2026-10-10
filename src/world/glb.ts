import * as THREE from 'three/webgpu';

/**
 * Minimal reader for the uncompressed GLBs the tree prep lane wrote (one mesh, float/unorm8/u16 accessors,
 * external textures). It returns plain geometries per primitive and never touches the textures: the 30 tree
 * files share six WebP maps, which GLTFLoader would decode once per file.
 */
interface GltfAccessor { bufferView: number; byteOffset?: number; componentType: number; count: number; type: string; normalized?: boolean }
interface GltfJson {
  meshes: Array<{ primitives: Array<{ attributes: Record<string, number>; indices: number; material: number }> }>;
  materials: Array<{ name: string }>;
  accessors: GltfAccessor[];
  bufferViews: Array<{ byteOffset?: number; byteLength: number; byteStride?: number }>;
  nodes: Array<{ name: string; extras?: Record<string, unknown> }>;
}

const SIZE: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

export interface GlbPrimitive {
  material: string;
  geometry: THREE.BufferGeometry;
}

/** Attribute names: POSITION, NORMAL, TEXCOORD_0 (v flipped to three's image convention), COLOR_0 -> color, _WIND -> wind. */
export function parseGlb(buf: ArrayBuffer): GlbPrimitive[] {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a GLB');
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jsonLen))) as GltfJson;
  const binStart = 20 + jsonLen + 8;
  const read = (idx: number, flipV = false): THREE.BufferAttribute => {
    const a = json.accessors[idx];
    const bv = json.bufferViews[a.bufferView];
    const n = SIZE[a.type];
    const off = binStart + (bv.byteOffset ?? 0) + (a.byteOffset ?? 0);
    if (bv.byteStride && bv.byteStride !== n * (a.componentType === 5126 ? 4 : a.componentType === 5123 ? 2 : 1)) {
      throw new Error('interleaved GLB accessors are not supported');
    }
    if (a.componentType === 5126) {
      const arr = new Float32Array(buf.slice(off, off + a.count * n * 4));
      if (flipV) for (let i = 1; i < arr.length; i += 2) arr[i] = 1 - arr[i];
      return new THREE.BufferAttribute(arr, n);
    }
    if (a.componentType === 5121) return new THREE.BufferAttribute(new Uint8Array(buf.slice(off, off + a.count * n)), n, !!a.normalized);
    if (a.componentType === 5123) return new THREE.BufferAttribute(new Uint16Array(buf.slice(off, off + a.count * n * 2)), n, !!a.normalized);
    if (a.componentType === 5125) return new THREE.BufferAttribute(new Uint32Array(buf.slice(off, off + a.count * n * 4)), n);
    throw new Error(`component type ${a.componentType}`);
  };
  const names: Record<string, string> = { POSITION: 'position', NORMAL: 'normal', TEXCOORD_0: 'uv', COLOR_0: 'color', _WIND: 'wind' };
  return json.meshes[0].primitives.map((p) => {
    const g = new THREE.BufferGeometry();
    for (const [k, idx] of Object.entries(p.attributes)) {
      const name = names[k];
      if (name) g.setAttribute(name, read(idx, k === 'TEXCOORD_0'));
    }
    g.setIndex(read(p.indices));
    g.computeBoundingSphere();
    return { material: json.materials[p.material]?.name ?? '', geometry: g };
  });
}
