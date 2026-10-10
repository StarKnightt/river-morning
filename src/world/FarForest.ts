import * as THREE from 'three/webgpu';
import {
  attribute, cameraPosition, cameraViewMatrix, dFdx, dFdy, float, hash, instanceIndex, length, log2, max, mix, normalize,
  positionLocal, smoothstep, step, texture, uniform, uv, varying, vec2, vec3, vec4, type N,
} from '../gfx/tsl';
import { LAYER_FOLIAGE } from '../core/layers';
import { params } from '../core/params';
import { forestU } from './Forest';
import { fbm, smoothstep as sstep } from './noise';
import { FOREST_REGION } from './ForestLayout';
import { river } from './River';

/**
 * The forest on the valley walls beyond the tree instances (round 1: bare grey slopes behind the bank trees, C1).
 * Camera-facing cards of conifer silhouettes, one quad each, carpeting the hills below the treeline up to the
 * skyline; the scene's aerial perspective makes them smaller, bluer and paler with distance, and rock outcrops and
 * steep faces stay bare. The silhouettes are drawn on the CPU at load (four kinds: spruce, subalpine fir spire,
 * lodgepole, broad old spruce) with lit outer branch tiers and dark gaps; shading comes from a cone normal across
 * the card, the lower crown shaded as in a closed stand. Main view only (not mirrored, no shadows, no AO pre-pass).
 */

const CELL_W = 128, CELL_H = 512, KINDS = 4;

interface Kind { base: number; rmax: number; pow: number; tiers: number; ragged: number }
const KIND_DEFS: Kind[] = [
  { base: 0.03, rmax: 0.84, pow: 1.0, tiers: 15, ragged: 0.2 }, // spruce
  { base: 0.02, rmax: 0.58, pow: 1.2, tiers: 13, ragged: 0.16 }, // subalpine fir spire
  { base: 0.22, rmax: 0.66, pow: 0.72, tiers: 9, ragged: 0.3 }, // lodgepole: crown held high, rounded top
  { base: 0.0, rmax: 0.96, pow: 0.95, tiers: 18, ragged: 0.22 }, // broad old spruce
];

const srgb = (c: number) => Math.round(255 * Math.min(1, c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055));

function vnoise(x: number, y: number, seed: number) {
  const h = (i: number, j: number) => {
    let n = (Math.imul(i, 374761393) + Math.imul(j, 668265263) + Math.imul(seed, 1274126177)) | 0;
    n = Math.imul(n ^ (n >>> 13), 1274126177);
    return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
  };
  const i = Math.floor(x), j = Math.floor(y), fx = x - i, fy = y - j;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const a = h(i, j) + (h(i + 1, j) - h(i, j)) * sx, b = h(i, j + 1) + (h(i + 1, j + 1) - h(i, j + 1)) * sx;
  return a + (b - a) * sy;
}

/** RGBA8 sRGB atlas: alpha = coverage, colour = needle albedo (lit outer tiers, dark core and gaps) */
function silhouetteAtlas(): THREE.DataTexture {
  const W = CELL_W * KINDS, H = CELL_H;
  const data = new Uint8Array(W * H * 4);
  const SS = 3;
  for (let k = 0; k < KINDS; k++) {
    const K = KIND_DEFS[k];
    for (let py = 0; py < H; py++) {
      for (let px = 0; px < CELL_W; px++) {
        let cov = 0, shade = 0;
        for (let sy = 0; sy < SS; sy++) for (let sxx = 0; sxx < SS; sxx++) {
          const u = ((px + (sxx + 0.5) / SS) / CELL_W) * 2 - 1;
          const t = (py + (sy + 0.5) / SS) / H;
          let rad: number;
          let tier = 0;
          if (t < K.base) {
            rad = 0.035;
          } else {
            const s = (t - K.base) / (1 - K.base);
            if (s > 0.995) continue;
            const ts = s * K.tiers + vnoise(s * 3, k, 5) * 0.6;
            tier = ts - Math.floor(ts);
            // each branch tier is widest at its drooping tips (the tier's bottom), narrower above
            const tierShape = 0.6 + 0.4 * Math.pow(1 - tier, 0.75);
            rad = K.rmax * Math.pow(1 - s, K.pow) * tierShape;
            // ragged edges; a little asymmetry from one side to the other
            rad *= 1 + K.ragged * (vnoise(t * 70, u > 0 ? 3 : 7, 11 + k) - 0.5) * 2 + 0.08 * Math.sign(u) * (vnoise(t * 9, 1, k) - 0.5);
            rad = Math.max(rad, 0.03 * (1 - s));
          }
          const e = Math.abs(u) / Math.max(rad, 1e-3);
          if (e > 1) continue;
          // lacy outer edge: holes where the noise is low, mostly in the outer third
          const lace = vnoise(u * 22 + k * 7, t * 160, 23);
          if (e > 0.62 && lace < (e - 0.62) * 1.1) continue;
          cov++;
          const lit = Math.pow(e, 0.55) * (0.78 + 0.22 * (1 - tier));
          const hole = vnoise(u * 9 + k * 3, t * 60, 31) < 0.32 && e < 0.7 ? 0.45 : 1;
          shade += (0.18 + 0.82 * lit) * hole;
        }
        const n = SS * SS;
        const a = cov / n;
        const v = cov ? shade / cov : 0;
        const o = (py * W + k * CELL_W + px) * 4;
        // linear albedo: dark needles, a touch of blue-green in the shade, yellower where the tiers catch light
        data[o] = srgb(0.012 + 0.04 * v * v);
        data[o + 1] = srgb(0.02 + 0.062 * v * Math.sqrt(v));
        data[o + 2] = srgb(0.014 + 0.026 * v);
        data[o + 3] = Math.round(a * 255);
      }
    }
  }
  // fill the colour of empty texels from their covered neighbours, so mips do not bleed dark fringes
  for (let pass = 0; pass < 4; pass++) {
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4;
      if (data[o + 3] > 0) continue;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const q = (yy * W + xx) * 4;
        if (data[q + 3] > 0) {
          data[o] = data[q]; data[o + 1] = data[q + 1]; data[o + 2] = data[q + 2];
          break;
        }
      }
    }
  }
  const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  return tex;
}

export const farForestU = { gain: uniform(1.0) };

function cardMaterial(atlas: THREE.Texture) {
  const r2 = !params.farForestOld;
  const m = r2 ? new THREE.MeshSSSNodeMaterial() : new THREE.MeshStandardNodeMaterial();
  m.name = 'far-forest';
  m.side = THREE.DoubleSide;
  m.alphaTest = 0.45;
  const base = attribute('aBase', 'vec3');
  const dims = attribute('aDims', 'vec4'); // width, height, kind, tint
  // per tree (distance round 2: four silhouettes at one width and tint read as a comb of identical cards): mirrored or
  // not, 0.8-1.2x as wide, a little bluer or yellower
  const rnd = varying(vec3(hash(instanceIndex), hash(instanceIndex.add(7919)), hash(instanceIndex.add(15881))));
  const width = r2 ? dims.x.mul(rnd.y.mul(0.4).add(0.8)) : dims.x;
  const toCam = cameraPosition.xz.sub(base.xz);
  const side = normalize(vec2(toCam.y, toCam.x.negate()));
  const q = positionLocal;
  m.positionNode = vec3(base.x.add(side.x.mul(q.x).mul(width)), base.y.add(q.y.mul(dims.y)), base.z.add(side.y.mul(q.x).mul(width)));
  const t = uv();
  const st = vec2(dims.z.add(r2 ? mix(t.x, t.x.oneMinus(), step(0.5, rnd.x)) : t.x).div(KINDS), t.y);
  const c = texture(atlas, st);
  // coverage-preserving alpha: thin crowns would otherwise dissolve in the mips
  const px = st.mul(vec2(CELL_W * KINDS, CELL_H));
  const mip = max(log2(max(length(dFdx(px)), length(dFdy(px)))), 0.0);
  const alpha = c.a.mul(mip.mul(0.22).add(1));
  const albedo = c.rgb.mul(dims.w).mul(farForestU.gain).mul(r2 ? mix(vec3(0.9, 1.0, 1.1), vec3(1.1, 1.03, 0.82), rnd.z) : vec3(1));
  m.colorNode = vec4(albedo, alpha);
  if (r2) {
    // backlit, a distant crown glows at its thin edges as the near needles do (Forest.ts), at half their strength:
    // toward the low sun the stands had no light side at all and read as one flat dark comb
    const sss = m as THREE.MeshSSSNodeMaterial;
    sss.thicknessColorNode = albedo.mul(vec3(1.05, 1.2, 0.38)).mul(float(1).sub(smoothstep(0.5, 1.0, c.a)).mul(0.6).add(0.4))
      .mul(forestU.translucency).mul(0.5);
    sss.thicknessDistortionNode = float(0.6);
    sss.thicknessPowerNode = float(5.0);
    sss.thicknessScaleNode = float(2.4);
    sss.thicknessAttenuationNode = float(0.5);
  }
  // a cone across the card: the side toward the sun lit, the other in shade
  const facing = normalize(vec3(toCam.x, 0, toCam.y));
  const across = t.x.sub(0.5).mul(2);
  const nW = normalize(vec3(side.x, 0, side.y).mul(across.mul(1.1)).add(vec3(0, 0.45, 0)).add(facing.mul(0.75)));
  m.normalNode = normalize(cameraViewMatrix.mul(vec4(nW, 0)).xyz);
  m.roughnessNode = float(0.9);
  m.metalnessNode = float(0);
  // a closed stand: the lower crowns stand in their neighbours' shade and see little sky
  const low = mix(float(0.3), float(1.0), smoothstep(0.0, 0.85, t.y));
  (m as unknown as { receivedShadowNode: (s: N) => N }).receivedShadowNode = (s) => s.mul(low);
  m.aoNode = mix(float(0.45), float(1.0), t.y);
  return m;
}

/**
 * `heightAt`: rendered ground height (the 4 m hill grid, the mountain ring beyond); null outside the terrain.
 * Cards stand on a jittered 8.5 m grid from the edge of the tree instances out to ~2.9 km, thinning beyond 1.2 km.
 */
export function buildFarForest(heightAt: (x: number, z: number) => number | null, stats?: Record<string, number>): THREE.Mesh {
  const R = 2900, S = 8.5;
  const n = Math.ceil((2 * R) / S);
  const pos: number[] = [], dim: number[] = [];
  const hsh = (i: number, j: number, k: number) => {
    let h = (Math.imul(i, 374761393) + Math.imul(j, 668265263) + Math.imul(k, 1274126177)) | 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  };
  const M = 3; // stay a few metres outside the instanced forest
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = -R + (i + hsh(i, j, 1)) * S, z = -R + (j + hsh(i, j, 2)) * S;
      if (x > FOREST_REGION.x0 + M && x < FOREST_REGION.x1 - M && z > FOREST_REGION.z0 + M && z < FOREST_REGION.z1 - M) continue;
      const d = Math.hypot(x, z);
      if (d > R) continue;
      const keep = 1 - 0.55 * sstep(1200, R, d);
      if (hsh(i, j, 3) > keep) continue;
      const y = heightAt(x, z);
      if (y === null) continue;
      // treeline ~1300-1500 m (forest polish: up to the ridges of the valley walls); thinner and smaller near it
      const tl = 1380 + 150 * fbm(x * 0.0011 + 1.3, z * 0.0011 - 0.4, 2);
      if (y > tl) continue;
      // not in the river channel up and down the valley
      if (Math.abs(z - river.zAt(x)) < 14) continue;
      const e = 6;
      const gx = ((heightAt(x + e, z) ?? y) - (heightAt(x - e, z) ?? y)) / (2 * e);
      const gz = ((heightAt(x, z + e) ?? y) - (heightAt(x, z - e) ?? y)) / (2 * e);
      const ny = 1 / Math.hypot(gx, 1, gz);
      // bare rock on the steepest faces and in the odd outcrop
      if (ny < 0.6) continue;
      const rock = fbm(x * 0.012 - 7.1, z * 0.012 + 2.2, 2);
      if (rock > 0.34 + (ny - 0.6) * 0.6) continue;
      // meadows and avalanche gaps break the carpet now and then
      const gap = fbm(x * 0.006 + 4.4, z * 0.006 - 3.3, 2);
      if (gap > 0.36 && hsh(i, j, 4) < 0.85) continue;
      const nearTop = sstep(tl - 180, tl, y);
      const kind = Math.min(KINDS - 1, Math.floor(hsh(i, j, 5) * KINDS));
      // (scene fix, critic polish r1: "a row of same-height spires like a picket fence"): stands of different ages a
      // few hundred metres across, and within them every tree its own height; darker in the gullies, lighter on the
      // spurs, where the low sun reaches the crowns
      const age = 0.62 + 0.75 * sstep(-0.35, 0.4, fbm(x * 0.0045 + 9.1, z * 0.0045 - 2.7, 2));
      const lap = ((heightAt(x + 25, z) ?? y) + (heightAt(x - 25, z) ?? y) + (heightAt(x, z + 25) ?? y) + (heightAt(x, z - 25) ?? y)) / 4 - y;
      const gully = sstep(0.5, 6, lap), spur = sstep(0.5, 6, -lap);
      // farther cards a little bigger, so the carpet keeps its coverage where it is thinned
      const hgt = (8 + 26 * Math.pow(hsh(i, j, 6), 1.1)) * age * (1 - 0.45 * nearTop) * (1 + 0.35 * sstep(1200, R, d));
      pos.push(x, y - 0.6, z);
      const tint = (0.7 + 0.55 * hsh(i, j, 7)) * (1 - 0.32 * gully) * (1 + 0.16 * spur);
      dim.push(hgt * (0.21 + 0.09 * hsh(i, j, 8)) * (kind === 3 ? 1.12 : 1), hgt, kind, tint);
    }
  }
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-0.5, 0, 0, 0.5, 0, 0, -0.5, 1, 0, 0.5, 1, 0]), 3));
  g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), 2));
  g.setIndex([0, 1, 2, 2, 1, 3]);
  g.setAttribute('aBase', new THREE.InstancedBufferAttribute(new Float32Array(pos), 3));
  g.setAttribute('aDims', new THREE.InstancedBufferAttribute(new Float32Array(dim), 4));
  g.instanceCount = pos.length / 3;
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 400, 0), R + 600);
  const mesh = new THREE.Mesh(g, cardMaterial(silhouetteAtlas()));
  mesh.name = 'far-forest';
  mesh.layers.set(LAYER_FOLIAGE);
  mesh.castShadow = false;
  mesh.receiveShadow = true;
  if (stats) stats.farTrees = g.instanceCount;
  return mesh;
}
