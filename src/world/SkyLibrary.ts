import * as THREE from 'three/webgpu';
import type { Assets, SkyMeta } from '../core/Assets';
import type { TimeOfDay } from '../core/params';
import { fetchSky, type DecodedHDR } from '../core/rgbe';

/** What tools/prep_hdri.py writes next to each sky (sky/<tod>.json); the morning file of part 1 lacks the extras. */
export interface SkyMetaTod extends SkyMeta {
  source?: string;
  /** sky rotation about +Y that puts the sun on its compass bearing (background and IBL sample with the inverse) */
  rotationYDeg?: number;
  sunBearingDeg?: number;
  /** false for a pre-sunrise or night sky: no disc was measured */
  sunVisible?: boolean;
  sunPhiDeg?: number;
  skyIrradianceH?: number;
  env?: string;
  /** visible sky cropped to the upper hemisphere; vMin = equirect v of its bottom edge */
  bg?: { file: string; width: number; height: number; vMin: number; elevationMinDeg: number };
}

export interface SkyAsset {
  tod: TimeOfDay;
  meta: SkyMetaTod;
  /** visible sky (equirect, possibly cropped: see vMin) */
  bg: THREE.DataTexture;
  /** 2k image-based light, sun clamped out (full sphere) */
  env: THREE.DataTexture;
  vMin: number;
}

/** The skies of part 1 (morning), used when a json has no prepared background or environment file. */
const FALLBACK: Record<TimeOfDay, { bg: string; env: string }> = {
  dawn: { bg: 'hdri/qwantani_dawn_puresky_4k.hdr', env: 'sky/dawn_env.rgbe.gz' },
  morning: { bg: 'hdri/qwantani_morning_puresky_4k.hdr', env: 'sky/morning_env.rgbe.gz' },
  golden: { bg: 'hdri/qwantani_sunset_puresky_4k.hdr', env: 'sky/golden_env.rgbe.gz' },
  night: { bg: 'hdri/qwantani_night_puresky_4k.hdr', env: 'sky/night_env.rgbe.gz' },
};

interface Pending {
  resolve: (d: DecodedHDR) => void;
  reject: (e: Error) => void;
}

/**
 * Loads the four skies on demand: the json on the main thread, the sky files decoded to half floats in a
 * worker, uploaded to the GPU at once (renderer.initTexture) and the CPU copy dropped (three only uploads
 * again when a texture's version changes, which these never do). Each sky is ~55 MB of VRAM (bg 4096x1152
 * + env 2048x1024, RGBA16F). Without workers it decodes on the main thread (a hitch of a few hundred ms).
 */
export class SkyLibrary {
  private cache = new Map<TimeOfDay, Promise<SkyAsset>>();
  private done = new Set<TimeOfDay>();
  private worker: Worker | null = null;
  private pending = new Map<number, Pending>();
  private nextId = 1;

  constructor(private renderer: THREE.WebGPURenderer, private assets: Assets) {
    try {
      this.worker = new Worker(new URL('../core/skyDecode.worker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = (e: MessageEvent<{ id: number; ok: boolean; width?: number; height?: number; data?: Uint16Array; error?: string }>) => {
        const d = e.data;
        const p = this.pending.get(d.id);
        if (!p) return;
        this.pending.delete(d.id);
        if (d.ok && d.data) p.resolve({ width: d.width!, height: d.height!, data: d.data });
        else p.reject(new Error(d.error ?? 'sky decode failed'));
      };
      this.worker.onerror = () => {
        // a broken worker (e.g. blocked by a CSP): decode the rest on the main thread
        for (const p of this.pending.values()) p.reject(new Error('sky worker failed'));
        this.pending.clear();
        this.worker = null;
      };
    } catch {
      this.worker = null;
    }
  }

  isReady(tod: TimeOfDay) {
    return this.done.has(tod);
  }

  get(tod: TimeOfDay): Promise<SkyAsset> {
    let p = this.cache.get(tod);
    if (!p) {
      p = this.load(tod).then((s) => {
        this.done.add(tod);
        return s;
      });
      p.catch(() => this.cache.delete(tod));
      this.cache.set(tod, p);
    }
    return p;
  }

  /** Load the given skies one after another in the background (after the first frame, between interactions). */
  prefetch(list: TimeOfDay[], gapMs = 400) {
    let chain: Promise<unknown> = Promise.resolve();
    for (const tod of list) {
      chain = chain.then(() => this.get(tod).catch(() => null)).then(() => new Promise((r) => setTimeout(r, gapMs)));
    }
    return chain;
  }

  private async load(tod: TimeOfDay): Promise<SkyAsset> {
    const meta = await this.assets.json<SkyMetaTod>(`sky/${tod}.json`);
    const bgPath = meta.bg?.file ?? FALLBACK[tod].bg;
    const envPath = meta.env ?? FALLBACK[tod].env;
    const [bgD, envD] = await Promise.all([this.decode(bgPath), this.decode(envPath)]);
    const bg = this.texture(bgD, `${tod}-bg`);
    const env = this.texture(envD, `${tod}-env`);
    return { tod, meta, bg, env, vMin: meta.bg?.vMin ?? 0 };
  }

  private decode(path: string): Promise<DecodedHDR> {
    const url = new URL(this.assets.base + path, location.href).href;
    const w = this.worker;
    if (w) {
      const id = this.nextId++;
      return new Promise<DecodedHDR>((resolve, reject) => {
        this.pending.set(id, { resolve, reject });
        w.postMessage({ id, url });
      }).catch(() => this.decodeHere(url));
    }
    return this.decodeHere(url);
  }

  private decodeHere(url: string): Promise<DecodedHDR> {
    return fetchSky(url);
  }

  /** the same texture HDRLoader + Assets.hdri() make, uploaded now */
  private texture(d: DecodedHDR, name: string) {
    const t = new THREE.DataTexture(d.data, d.width, d.height, THREE.RGBAFormat, THREE.HalfFloatType);
    t.name = name;
    t.mapping = THREE.EquirectangularReflectionMapping;
    t.colorSpace = THREE.LinearSRGBColorSpace;
    t.minFilter = THREE.LinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.generateMipmaps = false;
    t.flipY = true;
    t.needsUpdate = true;
    this.renderer.initTexture(t);
    // the GPU copy is all three needs from now on (it uploads again only on a version change)
    (t as unknown as { image: { width: number; height: number; data: Uint16Array | null } }).image = { width: d.width, height: d.height, data: null };
    return t;
  }

  dispose() {
    this.worker?.terminate();
    this.worker = null;
  }
}
