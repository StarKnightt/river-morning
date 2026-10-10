import * as THREE from 'three/webgpu';
import { clamp, float, positionLocal, sin, uniform, vec3, type N } from '../gfx/tsl';
import type { Assets } from '../core/Assets';
import { SPECIES, SPECIES_IDS, type SpeciesDef } from './species';
import type { SpeciesId } from './types';

const SWIM_WAVELENGTH = 0.95;

/**
 * Travelling body wave of swim_vertex.glsl (prep fish set) as a TSL position node: bends the body sideways in the
 * model frame (+Z nose, origin = centre of mass) and keeps the centre of mass on its path. `phase` is the
 * accumulated 2*pi*f*t, so frequency changes stay smooth; `amp` is the tail amplitude as a fraction of the length.
 */
function swimPosition(def: SpeciesDef, phase: N, amp: N, sag: N): N {
  const len = float(def.modelLength), k = (2 * Math.PI) / SWIM_WAVELENGTH;
  const lateral = (s: N) => {
    const env = s.mul(s.mul(1.3).sub(0.45)).add(0.15);
    return amp.mul(len).mul(env).mul(sin(s.mul(k).sub(phase)));
  };
  const s = clamp(float(def.noseZ).sub(positionLocal.z).div(len), 0, 1);
  const sCom = float(def.noseZ / def.modelLength);
  // held by the head and the wrist of the tail, the body sags a little between the hands
  const u = s.sub(0.55).div(0.38);
  const drop = sag.mul(len).mul(float(1).sub(u.mul(u)).max(0));
  return positionLocal.add(vec3(lateral(s).sub(lateral(sCom)), drop.negate(), 0));
}

/** One species, ready to show: `root` is the fish frame (scale it by caught length / model length). */
export class FishModel {
  readonly root = new THREE.Group();
  constructor(
    readonly def: SpeciesDef,
    readonly meshes: THREE.Mesh[],
    readonly fromGltf: boolean,
    /** swim clock (rad) and tail amplitude, shared by the species' materials */
    readonly phase: { value: number },
    readonly amp: { value: number },
    /** sag between the hands (fraction of the length) while it is held */
    readonly sag: { value: number },
  ) {
    this.root.name = `fish-${def.id}`;
    for (const m of meshes) {
      m.castShadow = false;
      m.receiveShadow = true;
      this.root.add(m);
    }
    this.root.visible = false;
  }

  setShadow(cast: boolean) {
    for (const m of this.meshes) m.castShadow = cast;
  }
}

/** Dequantize a (meshopt/quantized) primitive into float attributes with its node transform baked in. */
function bakeGeometry(src: THREE.BufferGeometry, m: THREE.Matrix4): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  const copy = (name: string, size: number) => {
    const a = src.getAttribute(name);
    if (!a) return;
    const out = new Float32Array(a.count * size);
    for (let i = 0; i < a.count; i++) {
      out[i * size] = a.getX(i);
      if (size > 1) out[i * size + 1] = a.getY(i);
      if (size > 2) out[i * size + 2] = a.getZ(i);
    }
    g.setAttribute(name, new THREE.BufferAttribute(out, size));
  };
  copy('position', 3);
  copy('normal', 3);
  copy('uv', 2);
  if (src.index) g.setIndex(Array.from(src.index.array as ArrayLike<number>));
  g.applyMatrix4(m);
  if (g.getAttribute('normal')) {
    const n = g.getAttribute('normal') as THREE.BufferAttribute;
    for (let i = 0; i < n.count; i++) {
      const x = n.getX(i), y = n.getY(i), z = n.getZ(i), l = Math.hypot(x, y, z) || 1;
      n.setXYZ(i, x / l, y / l, z / l);
    }
  } else g.computeVertexNormals();
  g.computeBoundingSphere();
  return g;
}

function swimMaterial(src: THREE.MeshStandardMaterial, model: { phase: N; amp: N; sag: N }, def: SpeciesDef) {
  const m = new THREE.MeshStandardNodeMaterial();
  m.name = src.name;
  m.color.copy(src.color);
  m.map = src.map;
  m.normalMap = src.normalMap;
  m.roughness = src.roughness;
  m.metalness = src.metalness;
  m.roughnessMap = src.roughnessMap;
  m.alphaTest = src.alphaTest;
  m.transparent = false;
  m.side = src.side;
  m.positionNode = swimPosition(def, model.phase, model.amp, model.sag);
  return m;
}

/** spindle-shaped body with a dark back and pale belly, plus a tail fin; same frame and length as the GLB fish */
function proceduralFish(def: SpeciesDef, phase: N, amp: N, sag: N): THREE.Mesh[] {
  const L = def.modelLength, nose = def.noseZ, tail = nose - L;
  const pts: THREE.Vector2[] = [];
  for (let i = 0; i <= 16; i++) {
    const s = i / 16; // 0 = nose, 1 = tail
    const r = Math.sin(Math.PI * Math.pow(s, 0.75)) * (s < 0.92 ? 1 : 0.5) * L * 0.11;
    pts.push(new THREE.Vector2(Math.max(r, 0.002), nose - s * L));
  }
  const body = new THREE.LatheGeometry(pts, 12);
  // lathe revolves around Y: its profile y is our z
  body.rotateX(Math.PI / 2);
  body.scale(0.4, 1, 1);
  const pos = body.getAttribute('position');
  const col = new Float32Array(pos.count * 3);
  const [r, g, b] = def.color;
  for (let i = 0; i < pos.count; i++) {
    const up = THREE.MathUtils.clamp(pos.getY(i) / (L * 0.11) * 0.5 + 0.5, 0, 1);
    const k = 0.55 + 0.9 * (1 - up);
    col[i * 3] = Math.min(1, r * (up < 0.35 ? 2.4 : k));
    col[i * 3 + 1] = Math.min(1, g * (up < 0.35 ? 2.4 : k));
    col[i * 3 + 2] = Math.min(1, b * (up < 0.35 ? 2.2 : k));
  }
  body.setAttribute('color', new THREE.BufferAttribute(col, 3));
  body.computeVertexNormals();
  const fin = new THREE.BufferGeometry();
  fin.setAttribute('position', new THREE.Float32BufferAttribute([
    0, 0, tail + L * 0.1, 0, L * 0.09, tail - L * 0.02, 0, -L * 0.09, tail - L * 0.02,
  ], 3));
  fin.computeVertexNormals();
  const mb = new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.35 });
  mb.positionNode = swimPosition(def, phase, amp, sag);
  const mf = new THREE.MeshStandardNodeMaterial({ color: new THREE.Color(...def.color), roughness: 0.5, side: THREE.DoubleSide });
  mf.positionNode = swimPosition(def, phase, amp, sag);
  return [new THREE.Mesh(body, mb), new THREE.Mesh(fin, mf)];
}

/**
 * The four trout for the hooked / held / released fish. Loads trout_all.glb from the prep fish set (one node per
 * species, body + fin primitives); falls back to procedural stand-ins if the file is not in public/assets yet.
 * Fish are opaque and drawn before the water, so the water refracts and absorbs them like the bed.
 */
export class FishModels {
  private constructor(readonly models: Record<SpeciesId, FishModel>) {}

  static async load(assets: Assets, path: string): Promise<FishModels> {
    let gltf: { scene: THREE.Object3D } | null = null;
    try {
      gltf = await assets.gltf(path);
    } catch (e) {
      console.warn(`[fishing] ${path} not found, using procedural fish`, e);
    }
    const models = {} as Record<SpeciesId, FishModel>;
    for (const id of SPECIES_IDS) {
      const def = SPECIES[id];
      const node = gltf?.scene.getObjectByName(def.node) ?? null;
      const proto = { phase: uniform(0), amp: uniform(0.08), sag: uniform(0) };
      let meshes: THREE.Mesh[] = [];
      if (node && node.parent) {
        node.parent.updateMatrixWorld(true);
        const parentInv = new THREE.Matrix4().copy(node.parent.matrixWorld).invert();
        node.traverse((o) => {
          const mesh = o as THREE.Mesh;
          if (!mesh.isMesh) return;
          const rel = new THREE.Matrix4().multiplyMatrices(parentInv, mesh.matrixWorld);
          const mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
          meshes.push(new THREE.Mesh(bakeGeometry(mesh.geometry, rel), swimMaterial(mat as THREE.MeshStandardMaterial, proto, def)));
        });
      }
      const fromGltf = meshes.length > 0;
      if (!fromGltf) meshes = proceduralFish(def, proto.phase, proto.amp, proto.sag);
      models[id] = new FishModel(def, meshes, fromGltf, proto.phase, proto.amp, proto.sag);
    }
    return new FishModels(models);
  }

  get(id: SpeciesId): FishModel {
    return this.models[id];
  }
}
