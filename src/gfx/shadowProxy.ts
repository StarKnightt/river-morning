import * as THREE from 'three/webgpu';
import { LAYER_SHADOW } from '../core/layers';

/**
 * Static casters merged into shadow-only meshes. The sun's shadow pass draws every caster with one shared depth
 * material and takes only the side, vertex displacement and alpha cut-outs from the object's own material, so
 * static opaque meshes without displacement give the same shadow merged into one draw per material side. It cuts the
 * shadow pass from 165 objects to 66 (the main thread's share of a redraw) and drops 8 shadow-only pipeline variants.
 * The originals stop casting; they still receive shadows and draw in every other pass.
 */
export function mergeShadowCasters(meshes: THREE.Mesh[], name: string, parent?: THREE.Object3D): THREE.Mesh[] {
  parent?.updateWorldMatrix(true, false);
  const toParent = parent ? parent.matrixWorld.clone().invert() : new THREE.Matrix4();
  // the faces the shadow pass draws (PCF): shadowSide if set, else the other side of a one-sided material
  const flip: Record<number, THREE.Side> = { [THREE.FrontSide]: THREE.BackSide, [THREE.BackSide]: THREE.FrontSide, [THREE.DoubleSide]: THREE.DoubleSide };
  const bySide = new Map<THREE.Side, THREE.Mesh[]>();
  for (const m of meshes) {
    const mat = m.material as THREE.Material;
    const side = mat.shadowSide ?? flip[mat.side];
    let list = bySide.get(side);
    if (!list) bySide.set(side, (list = []));
    list.push(m);
  }
  const out: THREE.Mesh[] = [];
  const v = new THREE.Vector3(), inst = new THREE.Matrix4(), M = new THREE.Matrix4();
  for (const [side, list] of bySide) {
    let nv = 0, ni = 0;
    for (const m of list) {
      const g = m.geometry, n = (m as THREE.InstancedMesh).isInstancedMesh ? (m as THREE.InstancedMesh).count : 1;
      nv += g.getAttribute('position').count * n;
      ni += (g.index ? g.index.count : g.getAttribute('position').count) * n;
    }
    const pos = new Float32Array(nv * 3), idx = new Uint32Array(ni);
    let ov = 0, oi = 0;
    for (const m of list) {
      m.updateWorldMatrix(true, false);
      const g = m.geometry, p = g.getAttribute('position'), index = g.index;
      const im = (m as THREE.InstancedMesh).isInstancedMesh ? (m as THREE.InstancedMesh) : null;
      for (let k = 0; k < (im ? im.count : 1); k++) {
        M.multiplyMatrices(toParent, m.matrixWorld);
        if (im) M.multiply((im.getMatrixAt(k, inst), inst));
        for (let i = 0; i < p.count; i++) {
          v.fromBufferAttribute(p, i).applyMatrix4(M);
          const o = (ov + i) * 3;
          pos[o] = v.x;
          pos[o + 1] = v.y;
          pos[o + 2] = v.z;
        }
        if (index) for (let i = 0; i < index.count; i++) idx[oi + i] = ov + index.getX(i);
        else for (let i = 0; i < p.count; i++) idx[oi + i] = ov + i;
        // a mirrored instance would turn its faces inside out
        if (M.determinant() < 0) for (let i = oi; i < oi + (index ? index.count : p.count); i += 3) [idx[i + 1], idx[i + 2]] = [idx[i + 2], idx[i + 1]];
        oi += index ? index.count : p.count;
        ov += p.count;
      }
      m.castShadow = false;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeBoundingSphere();
    const mat = new THREE.MeshBasicNodeMaterial();
    mat.shadowSide = side;
    const mesh = new THREE.Mesh(geo, mat);
    mesh.name = `${name}-shadow-${side}`;
    mesh.layers.set(LAYER_SHADOW);
    mesh.castShadow = true;
    mesh.receiveShadow = false;
    out.push(mesh);
  }
  return out;
}

/** Whether a mesh can join a merged caster: static, opaque in the shadow pass, no displacement or cut-out. */
export function mergeableCaster(m: THREE.Mesh): boolean {
  const mat = m.material as THREE.Material & { positionNode?: unknown; castShadowNode?: unknown; maskNode?: unknown; maskShadowNode?: unknown; alphaMap?: unknown };
  return m.castShadow && !Array.isArray(m.material) && !(m as THREE.SkinnedMesh).isSkinnedMesh && !m.morphTargetInfluences
    && !mat.positionNode && !mat.castShadowNode && !mat.maskNode && !mat.maskShadowNode && !mat.alphaMap && !(mat.alphaTest > 0);
}
