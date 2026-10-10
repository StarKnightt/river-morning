import * as THREE from 'three/webgpu';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import { GLTFLoader, type GLTF } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

export interface SkyMeta {
  sunDir: [number, number, number];
  sunElevationDeg: number;
  sunColor: [number, number, number];
  sunIrradiance: [number, number, number];
  horizonColor: [number, number, number];
  zenithColor: [number, number, number];
}

/**
 * Reads ImageBitmap pixels back as raw RGBA through a throwaway WebGL2 context. A 2D canvas
 * would premultiply alpha and destroy colour wherever the packed height/AO alpha is small.
 */
class RawDecoder {
  private gl: WebGL2RenderingContext;
  private tex: WebGLTexture;
  private fb: WebGLFramebuffer;

  constructor() {
    const gl = new OffscreenCanvas(1, 1).getContext('webgl2', { premultipliedAlpha: false });
    if (!gl) throw new Error('WebGL2 unavailable for texture decoding');
    this.gl = gl;
    this.tex = gl.createTexture()!;
    this.fb = gl.createFramebuffer()!;
  }

  read(bmp: ImageBitmap, out: Uint8Array, offset: number) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
    gl.readPixels(0, 0, bmp.width, bmp.height, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(out.buffer, offset, bmp.width * bmp.height * 4));
  }
}

let decoder: RawDecoder | null = null;
const rawDecoder = () => (decoder ??= new RawDecoder());

/**
 * Central loader with progress reporting. Packed textures carry data in alpha, so images
 * are decoded as ImageBitmaps without premultiplication or colour-space conversion.
 */
export class Assets {
  readonly manager = new THREE.LoadingManager();
  private bitmapLoader = new THREE.ImageBitmapLoader(this.manager);
  private hdrLoader = new HDRLoader(this.manager);
  private gltfLoader = new GLTFLoader(this.manager);
  private cache = new Map<string, Promise<unknown>>();
  anisotropy = 8;
  base = import.meta.env.BASE_URL + 'assets/';

  constructor() {
    this.bitmapLoader.setOptions({ imageOrientation: 'flipY', premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
    // prepared models are meshopt-compressed and quantized
    this.gltfLoader.setMeshoptDecoder(MeshoptDecoder);
  }

  private once<T>(key: string, make: () => Promise<T>): Promise<T> {
    let p = this.cache.get(key) as Promise<T> | undefined;
    if (!p) {
      p = make();
      this.cache.set(key, p);
    }
    return p;
  }

  texture(path: string, srgb: boolean, repeat = true): Promise<THREE.Texture> {
    return this.once(`tex:${path}:${srgb}`, async () => {
      const bmp = await this.bitmapLoader.loadAsync(this.base + path);
      const t = new THREE.Texture(bmp as unknown as HTMLImageElement);
      t.flipY = false;
      t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
      t.wrapS = t.wrapT = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
      t.anisotropy = this.anisotropy;
      t.generateMipmaps = true;
      t.minFilter = THREE.LinearMipmapLinearFilter;
      t.needsUpdate = true;
      return t;
    });
  }

  /** packed material: `<id>_ch.webp` (albedo + height) and `<id>_nr.webp` (normal xy, rough, ao) */
  async material(id: string): Promise<{ ch: THREE.Texture; nr: THREE.Texture }> {
    const [ch, nr] = await Promise.all([this.texture(`tex/${id}_ch.webp`, true), this.texture(`tex/${id}_nr.webp`, false)]);
    return { ch, nr };
  }

  /**
   * Several packed materials as two DataArrayTextures (colour+height, normal/rough/ao).
   * WebGPU caps samplers at 16 per stage, so terrain layers share two array bindings.
   * Layers are `tex/<id>_ch<suffix>.webp`; all must have the same size.
   */
  async materialArray(ids: string[], suffix: string): Promise<{ ch: THREE.DataArrayTexture; nr: THREE.DataArrayTexture }> {
    return this.once(`arr:${ids.join(',')}:${suffix}`, async () => {
      const load = (p: string) => this.bitmapLoader.loadAsync(this.base + p) as Promise<ImageBitmap>;
      const chB = await Promise.all(ids.map((id) => load(`tex/${id}_ch${suffix}.webp`)));
      const nrB = await Promise.all(ids.map((id) => load(`tex/${id}_nr${suffix}.webp`)));
      const make = (bitmaps: ImageBitmap[], srgb: boolean) => {
        const w = bitmaps[0].width, h = bitmaps[0].height;
        const data = new Uint8Array(w * h * 4 * bitmaps.length);
        bitmaps.forEach((b, i) => {
          if (b.width !== w || b.height !== h) throw new Error(`array layer size mismatch: ${ids[i]}`);
          rawDecoder().read(b, data, i * w * h * 4);
          b.close();
        });
        const t = new THREE.DataArrayTexture(data, w, h, bitmaps.length);
        t.format = THREE.RGBAFormat;
        t.type = THREE.UnsignedByteType;
        t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.generateMipmaps = true;
        t.minFilter = THREE.LinearMipmapLinearFilter;
        t.magFilter = THREE.LinearFilter;
        t.anisotropy = this.anisotropy;
        t.needsUpdate = true;
        return t;
      };
      return { ch: make(chB, true), nr: make(nrB, false) };
    });
  }

  hdri(path: string): Promise<THREE.DataTexture> {
    return this.once(`hdr:${path}`, async () => {
      const t = (await this.hdrLoader.loadAsync(this.base + path)) as THREE.DataTexture;
      t.mapping = THREE.EquirectangularReflectionMapping;
      t.colorSpace = THREE.LinearSRGBColorSpace;
      t.minFilter = THREE.LinearFilter;
      t.magFilter = THREE.LinearFilter;
      t.generateMipmaps = false;
      return t;
    });
  }

  json<T>(path: string): Promise<T> {
    return this.once(`json:${path}`, async () => {
      const r = await fetch(this.base + path);
      if (!r.ok) throw new Error(`${r.status} ${path}`);
      return (await r.json()) as T;
    });
  }

  gltf(path: string): Promise<GLTF> {
    return this.once(`gltf:${path}`, () => this.gltfLoader.loadAsync(this.base + path));
  }
}
