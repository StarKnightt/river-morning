import * as THREE from 'three/webgpu';
import {
  cameraPosition, dot, float, mix, normalMap, normalWorldGeometry, positionLocal, positionWorld, smoothstep, texture, uniform, uv, vec2, vec3, type N,
} from '../gfx/tsl';
import { tnoiseS } from '../gfx/noiseTex';
import { setOpaqueAlbedo } from '../gfx/opaque';
import type { Assets } from '../core/Assets';
import { river, newSample } from '../world/River';
import type { Obstacle, RiverField } from './RiverField';
import { fieldNodes } from './RiverField';
import { installUnderwater } from './Underwater';

/**
 * Boulders in the riffle and at the pool head (Poly Haven CC0 scans, prepared by the prep lane: LOD0 only,
 * quantized, meshopt-compressed). Each placement bakes its transform into its own geometry copy, so every rock
 * mesh has an identity model matrix: the underwater shadow warp can then use positionLocal as world space.
 * The warm Karoo/forest scans are tinted toward the cool grey granite of the bed.
 */
interface RockModel {
  /** LOD0 bounds (m) from models.json: x, y (height), z */
  size: [number, number, number];
  /** mean linear albedo luminance of the base colour map (models.json) */
  lum: number;
  /** ARM texture carries AO in red */
  ao: boolean;
}

const LUM = vec3(0.2126, 0.7152, 0.0722);
const lum3 = (c: [number, number, number]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

export const ROCK_MODELS: Record<string, RockModel> = {
  boulder_01: { size: [1.269, 1.004, 1.825], lum: lum3([0.171, 0.113, 0.072]), ao: true },
  namaqualand_boulder_05: { size: [1.364, 0.536, 0.746], lum: lum3([0.38, 0.267, 0.154]), ao: true },
  rock_moss_02_r07: { size: [1.692, 0.745, 1.215], lum: lum3([0.137, 0.116, 0.066]), ao: false },
  rock_moss_02_r08: { size: [1.206, 0.529, 0.984], lum: lum3([0.112, 0.099, 0.067]), ao: false },
  rock_moss_02_r09: { size: [1.629, 0.61, 1.011], lum: lum3([0.142, 0.124, 0.078]), ao: false },
  rock_moss_02_r10: { size: [2.203, 0.768, 1.552], lum: lum3([0.142, 0.13, 0.095]), ao: false },
  rock_moss_02_r12: { size: [1.58, 0.736, 1.443], lum: lum3([0.091, 0.079, 0.045]), ao: false },
  rock_moss_02_r13: { size: [2.091, 1.256, 1.138], lum: lum3([0.093, 0.081, 0.046]), ao: false },
};

/**
 * Placements: centreline x, lateral position as a fraction of the local half width (+ = south bank),
 * model, scale, yaw (deg) and where the top should sit relative to the water surface (m, + = above).
 * The rock is lowered further if needed so its base is always at least 5 cm into the bed.
 */
const PLACEMENTS: Array<[number, number, string, number, number, number]> = [
  // riffle (C3 frames x -27..-19, C1 sees it from downstream)
  [-20.8, -0.22, 'rock_moss_02_r08', 0.8, 20, 0.07],
  [-22.9, 0.42, 'namaqualand_boulder_05', 0.7, -35, 0.04],
  [-24.9, -0.5, 'rock_moss_02_r09', 0.65, 70, 0.1],
  [-26.6, 0.12, 'boulder_01', 0.42, 110, 0.16],
  [-29.2, 0.48, 'rock_moss_02_r12', 0.55, 10, 0.05],
  [-18.4, 0.55, 'rock_moss_02_r07', 0.5, -60, -0.05],
  [-31.6, -0.3, 'rock_moss_02_r10', 0.45, 40, 0.06],
  [-16.0, -0.45, 'rock_moss_02_r13', 0.42, 15, 0.12],
  [-27.8, -0.15, 'rock_moss_02_r08', 0.5, 140, -0.07],
  // pool head: on the shallower margins where the riffle spills in
  [-5.6, 0.68, 'boulder_01', 0.55, 30, 0.18],
  [-2.8, -0.82, 'rock_moss_02_r13', 0.8, -40, 0.18],
];

export interface PlacedRock {
  id: string;
  x: number;
  z: number;
  /** base height (m) */
  y: number;
  scale: number;
  yaw: number;
  obstacle: Obstacle;
  /** debug: water depth at the rock, top above the water (m) */
  depth: number;
  above: number;
}

/** World placements and the obstacles the water field needs (before any model is loaded). */
export function placeRocks(groundY: (x: number, z: number) => number): PlacedRock[] {
  const S = newSample();
  return PLACEMENTS.map(([cx, u, id, scale0, yawDeg, top]) => {
    const M = ROCK_MODELS[id];
    const cz = river.zAt(cx);
    river.sample(cx, cz, S);
    const x = cx - S.tz * u * S.half, z = cz + S.tx * u * S.half;
    river.sample(x, z, S);
    // lowest ground under the footprint
    const bedUnder = (rad: number) => {
      let b = groundY(x, z);
      for (let a = 0; a < 6; a++) {
        const ang = (a / 6) * Math.PI * 2;
        b = Math.min(b, groundY(x + Math.cos(ang) * rad * 0.7, z + Math.sin(ang) * rad * 0.7));
      }
      return b;
    };
    // a rock too short to reach its target above the water from 4 cm inside the bed grows (up to 2x)
    let scale = scale0;
    for (let it = 0; it < 3; it++) {
      const rad = 0.42 * (M.size[0] + M.size[2]) * 0.5 * scale;
      const need = S.waterY + top - (bedUnder(rad) - 0.04);
      if (M.size[1] * scale >= need) break;
      scale = Math.min(scale0 * 2, need / M.size[1]);
    }
    const sx = M.size[0] * scale, sy = M.size[1] * scale, sz = M.size[2] * scale;
    const rad = 0.42 * (sx + sz) * 0.5;
    const bed = bedUnder(rad);
    const y = Math.min(bed - 0.04, S.waterY + top - sy);
    const topY = y + sy;
    return {
      id, x, z, y, scale, yaw: (yawDeg * Math.PI) / 180,
      obstacle: { x, z, r: rad, topY },
      depth: S.waterY - bed, above: topY - S.waterY,
    };
  });
}

/** Float copy of a (quantized) glTF mesh with its node transform baked in. */
export function bakeGeometry(mesh: THREE.Mesh): THREE.BufferGeometry {
  mesh.updateWorldMatrix(true, false);
  const src = mesh.geometry;
  const g = new THREE.BufferGeometry();
  for (const name of ['position', 'normal', 'uv'] as const) {
    const a = src.getAttribute(name) as THREE.BufferAttribute | undefined;
    if (!a) continue;
    const out = new Float32Array(a.count * a.itemSize);
    for (let i = 0; i < a.count; i++) {
      out[i * a.itemSize] = a.getX(i);
      if (a.itemSize > 1) out[i * a.itemSize + 1] = a.getY(i);
      if (a.itemSize > 2) out[i * a.itemSize + 2] = a.getZ(i);
    }
    g.setAttribute(name, new THREE.BufferAttribute(out, a.itemSize));
  }
  if (src.index) g.setIndex(src.index.clone());
  g.applyMatrix4(mesh.matrixWorld);
  // KHR_texture_transform from UV quantization: bake it into the uvs (the node material samples raw uv)
  const map = (mesh.material as THREE.MeshStandardMaterial).map;
  const uvA = g.getAttribute('uv') as THREE.BufferAttribute | undefined;
  if (map && uvA && (map.offset.x !== 0 || map.offset.y !== 0 || map.repeat.x !== 1 || map.repeat.y !== 1 || map.rotation !== 0)) {
    map.updateMatrix();
    const v = new THREE.Vector3();
    for (let i = 0; i < uvA.count; i++) {
      v.set(uvA.getX(i), uvA.getY(i), 1).applyMatrix3(map.matrix);
      uvA.setXY(i, v.x, v.y);
    }
  }
  return g;
}

/**
 * Per-model values are uniforms, not constants, so all rock materials generate the same shader code and share
 * one compiled program per pass (eight programs with the caustic hook were ~20 extra pipeline compiles at load).
 */
function rockMaterial(src: THREE.MeshStandardMaterial, model: RockModel, field: RiverField | null, underwater: boolean) {
  const m = new THREE.MeshStandardNodeMaterial();
  m.name = 'rocks';
  const t = uv();
  const base = texture(src.map!, t).rgb;
  const orm = texture(src.roughnessMap ?? src.metalnessMap!, t);
  // cool grey granite: mostly grey, a quarter of the scan's own colour, scaled to the bed stones' brightness
  const l = dot(base, LUM);
  let col = mix(vec3(l), base, 0.25).mul(vec3(0.94, 0.98, 1.05)).mul(uniform(0.17 / model.lum));
  let rough = orm.y;
  let wet: N = float(0);
  if (field) {
    const F = fieldNodes(field);
    const waterY = F.A(positionWorld.xz).w;
    // wet below the waterline and in a ragged splash band a hand's width above it (higher on the upstream face
    // where the water piles up): darker, more saturated, glossy, with a distinct top edge
    const pw = vec2(positionWorld.x.add(positionWorld.z.mul(0.7)), positionWorld.y.add(positionWorld.z.mul(0.5)));
    // (round 2: 12 cm soaked the whole of the small riffle stones, which stand only 10-20 cm out of the water)
    const band = float(0.06).add(tnoiseS(pw.mul(4.0), 2.3).mul(0.03)).add(tnoiseS(pw.mul(13.0), 6.1).mul(0.012));
    wet = smoothstep(waterY.add(band).add(0.012), waterY.add(band).sub(0.012), positionWorld.y);
    // soaked darkest right at the waterline, a little lighter toward the top of the band
    const soak = smoothstep(waterY.add(0.06), waterY.sub(0.01), positionWorld.y);
    // (round 1's x0.27-0.4 made the small riffle stones coal-black, darker than anything else in C3)
    col = mix(col, mix(vec3(dot(col, LUM)), col, 1.3).mul(mix(float(0.74), float(0.6), soak)), wet);
    // dry tops above the splash band are paler, sun-dried grey
    col = col.mul(float(1).add(smoothstep(0.4, 0.9, normalWorldGeometry.y).mul(float(1).sub(wet)).mul(0.5)));
    const far = smoothstep(6.0, 25.0, positionWorld.sub(cameraPosition).length());
    rough = mix(rough, mix(0.26, 0.55, far), wet);
  }
  m.colorNode = col;
  m.roughnessNode = rough;
  m.metalnessNode = float(0);
  // the moss-set maps carry no AO in red. The wet band's low reflections meet the banks, not the open sky of the
  // environment map (it read as a pale, dry lump).
  m.aoNode = mix(float(1), orm.x, uniform(model.ao ? 1 : 0)).mul(float(1).sub(wet.mul(0.3)));
  m.normalNode = normalMap(texture(src.normalMap!, t), uniform(new THREE.Vector2(src.normalScale.x, src.normalScale.y)));
  if (field && underwater) {
    const F = fieldNodes(field);
    installUnderwater(m, { field, surfaceY: F.A(positionWorld.xz).w, castSurfaceY: F.A(positionLocal.xz).w });
  }
  setOpaqueAlbedo(m, m.colorNode);
  return m;
}

/** Load the models and add one mesh per placement. `underwater`: caustics and refracted shadows on the rocks. */
export async function loadRocks(assets: Assets, placed: PlacedRock[], field: RiverField | null, underwater = true): Promise<THREE.Group> {
  const group = new THREE.Group();
  group.name = 'rocks';
  const ids = [...new Set(placed.map((p) => p.id))];
  const loaded = await Promise.all(ids.map(async (id) => {
    const gltf = await assets.gltf(`models/rocks/${id}.glb`);
    let mesh: THREE.Mesh | null = null;
    gltf.scene.traverse((o) => {
      const mm = o as THREE.Mesh;
      if (!mm.isMesh) return;
      const lod0 = mm.name.includes('LOD0') || !!mm.parent?.name.includes('LOD0');
      if (!mesh || (lod0 && !mesh.name.includes('LOD0'))) mesh = mm;
    });
    if (!mesh) throw new Error(`no mesh in ${id}`);
    const m = mesh as THREE.Mesh;
    return { id, geometry: bakeGeometry(m), material: rockMaterial(m.material as THREE.MeshStandardMaterial, ROCK_MODELS[id], field, underwater) };
  }));
  const byId = new Map(loaded.map((l) => [l.id, l]));
  const mat = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3();
  for (const p of placed) {
    const l = byId.get(p.id)!;
    const g = l.geometry.clone();
    q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), p.yaw);
    mat.compose(new THREE.Vector3(p.x, p.y, p.z), q, sc.setScalar(p.scale));
    g.applyMatrix4(mat);
    g.computeBoundingBox();
    g.computeBoundingSphere();
    const mesh = new THREE.Mesh(g, l.material);
    mesh.name = `rock-${p.id}`;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
  }
  return group;
}
