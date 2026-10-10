import * as THREE from 'three/webgpu';
import { texture, vec2, type N } from './tsl';

/**
 * Baked gradient noise for the big ground and prop shaders. The core terrain shader inlined 24 MaterialX Perlin calls
 * (integer hashing at 8 lattice corners each): 82 KB of WGSL and 4.5 s to compile per pass. One texture fetch per call
 * instead, from a tileable 256² RGBA8 tile holding four independent 2D slices of Ken Perlin's improved noise (the
 * construction mx_noise_float uses), 8 texels per lattice cell, periodic over 32 cells. tnoise(p) has one feature per
 * unit of p like mx_noise_float(vec3(p, z)) * 0.5 + 0.5, and mips average it toward 0.5 far away instead of aliasing.
 */
const SIZE = 256;
const PERIOD = 32;

function bake(): THREE.DataTexture {
  const perm = new Uint8Array(512);
  const p = Array.from({ length: 256 }, (_, i) => i);
  let r = 1013904223;
  for (let i = 255; i > 0; i--) {
    r = (Math.imul(r, 1664525) + 1013904223) >>> 0;
    const j = r % (i + 1);
    [p[i], p[j]] = [p[j], p[i]];
  }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
  const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  const grad = (h: number, x: number, y: number, z: number) => {
    h &= 15;
    const u = h < 8 ? x : y, v = h < 4 ? y : h === 12 || h === 14 ? x : z;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
  };
  const noise = (x: number, y: number, z: number) => {
    const X = Math.floor(x), Y = Math.floor(y), Z = Math.floor(z);
    x -= X; y -= Y; z -= Z;
    const x0 = ((X % PERIOD) + PERIOD) % PERIOD, y0 = ((Y % PERIOD) + PERIOD) % PERIOD;
    const x1 = (x0 + 1) % PERIOD, y1 = (y0 + 1) % PERIOD, z0 = Z & 255, z1 = (Z + 1) & 255;
    const h = (a: number, b: number, c: number) => perm[perm[perm[a] + b] + c];
    const u = fade(x), v = fade(y), w = fade(z);
    return lerp(
      lerp(lerp(grad(h(x0, y0, z0), x, y, z), grad(h(x1, y0, z0), x - 1, y, z), u),
        lerp(grad(h(x0, y1, z0), x, y - 1, z), grad(h(x1, y1, z0), x - 1, y - 1, z), u), v),
      lerp(lerp(grad(h(x0, y0, z1), x, y, z - 1), grad(h(x1, y0, z1), x - 1, y, z - 1), u),
        lerp(grad(h(x0, y1, z1), x, y - 1, z - 1), grad(h(x1, y1, z1), x - 1, y - 1, z - 1), u), v),
      w);
  };
  // fractional z: on an integer plane the z gradients contribute nothing and the slice is weaker
  const slices = [0.31, 1.62, 2.47, 3.83];
  const data = new Uint8Array(SIZE * SIZE * 4);
  const k = PERIOD / SIZE;
  for (let j = 0; j < SIZE; j++) {
    for (let i = 0; i < SIZE; i++) {
      for (let c = 0; c < 4; c++) {
        const n = 0.982 * noise((i + 0.5) * k, (j + 0.5) * k, slices[c]);
        data[(j * SIZE + i) * 4 + c] = Math.round(Math.min(1, Math.max(0, n * 0.5 + 0.5)) * 255);
      }
    }
  }
  const t = new THREE.DataTexture(data, SIZE, SIZE, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.anisotropy = 4;
  t.colorSpace = THREE.NoColorSpace;
  t.needsUpdate = true;
  return t;
}

let tex: THREE.DataTexture | null = null;
export const noiseTexture = () => (tex ??= bake());

/** all four channels at p (independent noises at the same place, one fetch) */
export function tnoise4(p: N, off: [number, number] = [0, 0]): N {
  return texture(noiseTexture(), p.div(PERIOD).add(vec2(off[0], off[1])));
}

/** 0..1 noise with one feature per unit of p (tiles every 32 units); `seed` picks the channel and an offset */
export function tnoise(p: N, seed: number): N {
  const ch = Math.abs(Math.floor(seed * 7.31)) % 4;
  const s = tnoise4(p, [(seed * 0.1373) % 1, (seed * 0.3119) % 1]);
  return [s.x, s.y, s.z, s.w][ch];
}

/** -1..1, like mx_noise_float */
export const tnoiseS = (p: N, seed: number): N => tnoise(p, seed).mul(2).sub(1);
