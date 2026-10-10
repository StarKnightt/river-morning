import * as THREE from 'three/webgpu';
import {
  abs, cos, dot, equirectUV, float, Fn, If, length, luminance, max, mix, normalize, pow, positionWorldDirection, screenUV,
  sin, smoothstep, texture, uniform, vec2, vec3, vec4, type N,
} from '../gfx/tsl';
import type { SkyLighting } from './SkyLighting';
import type { SkyAsset } from './SkyLibrary';
import { moonSky, nightU, starField } from './NightSky';
import { todSkyU } from './TodUniforms';

/** A world direction in a sky plate's frame: t = R_y(-theta) w, rot = (cos theta, sin theta) (the inverse rotation). */
const toSky = (w: N, rot: N) => vec3(w.x.mul(rot.x).sub(w.z.mul(rot.y)), w.y, w.x.mul(rot.y).add(w.z.mul(rot.x)));

/** equirect uv of a (possibly cropped) plate whose bottom edge sits at v = vMin */
const plateUV = (d: N, vMin: N) => {
  const e = equirectUV(d);
  return vec2(e.x, e.y.sub(vMin).div(float(1).sub(vMin)));
};

export interface SkySlot {
  asset: SkyAsset;
  rotY: number;
  gain: THREE.Color;
}

/**
 * Two sky slots cross-faded by `blend` (0 = A, 1 = B), each with its own plate, rotation and gain:
 *  - the visible background (custom background node: plate + zenith deepening + sun-side glow + horizon band,
 *    then stars and the moon at night). The pool mirror renders it from the mirrored camera.
 *  - the image-based light: both 2k probes blended into one RGBA16F equirect target in world orientation
 *    (environmentRotation 0), prefiltered here into one persistent PMREM target whose texture is
 *    scene.environment for the whole game. The water material captured scene.environment at construction, so
 *    that texture object never changes, only its content: once per settled change, every `envEvery` frames
 *    while a fade runs. Prefiltering here (in an updater, outside any pass) rather than through three's lazy
 *    needsPMREMUpdate keeps the rebuild out of the scene pass, where it would run nested with the pass's context.
 */
export class SkyBlend {
  readonly u = {
    blend: uniform(0),
    rotA: uniform(new THREE.Vector2(1, 0)),
    rotB: uniform(new THREE.Vector2(1, 0)),
    gainA: uniform(new THREE.Color(1, 1, 1)),
    gainB: uniform(new THREE.Color(1, 1, 1)),
    vMinA: uniform(0),
    vMinB: uniform(0),
    /** sky deepening toward the zenith (the part 1 morning: (0.58, 0.74, 1.04)) */
    zenithTint: uniform(new THREE.Color(0.58, 0.74, 1.04)),
    /** strength of the haze band along the horizon (morning 0.3) */
    band: uniform(0.3),
    /** luminance of the night plate's dark sky after its gain: the Milky Way is what stands above it */
    milkyRef: uniform(0.01),
    /** tint of the image-based light only (applied when the probe is rendered) */
    envTint: uniform(new THREE.Color(1, 1, 1)),
  };
  readonly envTarget: THREE.RenderTarget;
  private pmrem: THREE.PMREMGenerator | null = null;
  private cube: THREE.RenderTarget | null = null;
  private bgA: N;
  private bgB: N;
  private envA: N;
  private envB: N;
  private quad: THREE.QuadMesh;
  slotA: SkySlot;
  slotB: SkySlot;

  constructor(first: SkySlot) {
    this.slotA = first;
    this.slotB = first;
    // exact ray direction (the interpolated normals of the 32x32 background sphere are off by up to ~0.3 deg,
    // enough to make the moon and stars drift when they cross a facet)
    const view = positionWorldDirection;
    this.bgA = texture(first.asset.bg, plateUV(toSky(view, this.u.rotA), this.u.vMinA));
    this.bgB = texture(first.asset.bg, plateUV(toSky(view, this.u.rotB), this.u.vMinB));

    // equirect texel -> world direction. screenUV.y runs down the target and the PMREM samples it at
    // equirectUV(d) the same way, so v = screenUV.y (1 = zenith) keeps the two consistent on both backends
    const s = screenUV;
    const phi = s.x.sub(0.5).mul(2 * Math.PI), el = s.y.sub(0.5).mul(Math.PI);
    const d = vec3(cos(phi).mul(cos(el)), sin(el), sin(phi).mul(cos(el)));
    this.envA = texture(first.asset.env, equirectUV(toSky(d, this.u.rotA)));
    this.envB = texture(first.asset.env, equirectUV(toSky(d, this.u.rotB)));
    const mat = new THREE.MeshBasicNodeMaterial();
    mat.name = 'SkyBlend.env';
    mat.fog = false;
    mat.lights = false;
    mat.colorNode = vec4(mix(this.envA.rgb.mul(this.u.gainA), this.envB.rgb.mul(this.u.gainB), this.u.blend).mul(vec3(this.u.envTint)), 1);
    this.quad = new THREE.QuadMesh(mat);
    this.envTarget = new THREE.RenderTarget(2048, 1024, { type: THREE.HalfFloatType, depthBuffer: false, generateMipmaps: false });
    const t = this.envTarget.texture;
    t.name = 'tod-env';
    t.mapping = THREE.EquirectangularReflectionMapping;
    t.minFilter = THREE.LinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.wrapS = THREE.RepeatWrapping;
    this.setSlot('A', first);
    this.setSlot('B', first);
  }

  setSlot(which: 'A' | 'B', s: SkySlot) {
    const th = (s.rotY * Math.PI) / 180;
    if (which === 'A') {
      this.slotA = s;
      this.bgA.value = s.asset.bg;
      this.envA.value = s.asset.env;
      this.u.rotA.value.set(Math.cos(th), Math.sin(th));
      this.u.gainA.value.copy(s.gain);
      this.u.vMinA.value = s.asset.vMin;
    } else {
      this.slotB = s;
      this.bgB.value = s.asset.bg;
      this.envB.value = s.asset.env;
      this.u.rotB.value.set(Math.cos(th), Math.sin(th));
      this.u.gainB.value.copy(s.gain);
      this.u.vMinB.value = s.asset.vMin;
    }
    for (const tex of [s.asset.bg, s.asset.env]) tex.wrapS = THREE.RepeatWrapping;
  }

  /** B becomes A (end of a transition) */
  settle() {
    this.setSlot('A', this.slotB);
    this.u.blend.value = 0;
  }

  /**
   * The visible sky, as SkyLighting.load built it for part 1 (zenith deepening, sun-side glow strongest low down,
   * a faint haze band at the horizon), from the two slots, plus stars and the moon.
   */
  backgroundNode(sky: SkyLighting, night = true): N {
    const u = this.u;
    // an Fn so the night branch has a shader stack to live on
    return Fn(() => {
      const view = positionWorldDirection;
      const plate = mix(this.bgA.rgb.mul(u.gainA), this.bgB.rgb.mul(u.gainB), u.blend);
      const up = smoothstep(0.04, 0.75, view.y);
      let c: N = plate.mul(mix(vec3(1.0), vec3(u.zenithTint), up));
      const cosSun = max(dot(view, sky.u.sunDir), 0.0);
      const low = smoothstep(0.8, 0.0, view.y);
      c = c.add(vec3(sky.u.sunColor).mul(pow(cosSun, 4.0).mul(0.22).add(pow(cosSun, 20.0).mul(0.45)).mul(low)));
      const band = smoothstep(0.16, 0.0, abs(view.y.sub(0.015)));
      c = mix(c, vec3(sky.u.horizon).mul(vec3(1.02, 1.0, 0.97)), band.mul(u.band));
      // (round 2, critic r1: the dawn plate read as one flat grey-lavender from the top of the frame to the peaks) before
      // sunrise: away from the sun a rose band (the Belt of Venus) over a blue-grey one (the Earth's shadow), toward it a
      // warm glow low down; the deep blue overhead is the zenith tint
      const dawn = todSkyU.dawn;
      const sH = normalize(vec2(todSkyU.sunDir.x, todSkyU.sunDir.z));
      const vH = vec2(view.x, view.z);
      const cosAz = dot(vH.div(length(vH).max(1e-4)), sH);
      const away = smoothstep(-0.25, 0.85, cosAz.negate());
      const el = view.y;
      const H = luminance(vec3(sky.u.horizon));
      const earthShadow = smoothstep(0.0, 0.025, el).mul(float(1).sub(smoothstep(0.055, 0.11, el)));
      const belt = smoothstep(0.07, 0.13, el).mul(float(1).sub(smoothstep(0.22, 0.4, el)));
      c = mix(c, vec3(0.42, 0.5, 0.68).mul(H.mul(0.62)), earthShadow.mul(away).mul(dawn).mul(0.7));
      c = mix(c, vec3(1.0, 0.56, 0.62).mul(H.mul(0.85)), belt.mul(away).mul(dawn).mul(0.85));
      const toward = smoothstep(0.2, 0.95, cosAz).mul(float(1).sub(smoothstep(0.0, 0.32, el)));
      c = c.add(vec3(1.0, 0.66, 0.4).mul(H.mul(0.7)).mul(toward).mul(dawn));
      if (!night) return c;
      // stars and moon only while it is (becoming) night: a uniform branch, so the day pays nothing for them (the
      // background is drawn first, under every pixel of the frame and of the mirror)
      const out = c.toVar();
      If(nightU.amount.greaterThan(0.0005), () => {
        const milky = smoothstep(1.12, 1.9, luminance(plate).div(u.milkyRef.max(1e-6)));
        const moon = moonSky(view);
        out.addAssign(starField(view, milky).mul(moon.starFade).add(moon.color));
      });
      return out;
    })();
  }

  /** scene.environment: the prefiltered probe (exists after the first renderEnv) */
  get envTexture(): THREE.Texture {
    return this.cube!.texture;
  }

  /** Re-render the blended probe and prefilter it (cube size 512, as three made it from the 2k plate before). */
  renderEnv(renderer: THREE.WebGPURenderer) {
    const prevRT = renderer.getRenderTarget(), prevMRT = renderer.getMRT();
    renderer.setMRT(null);
    renderer.setRenderTarget(this.envTarget);
    this.quad.render(renderer);
    renderer.setRenderTarget(prevRT);
    this.pmrem ??= new THREE.PMREMGenerator(renderer);
    this.cube = this.pmrem.fromEquirectangular(this.envTarget.texture, this.cube);
    renderer.setRenderTarget(prevRT);
    renderer.setMRT(prevMRT);
  }

  dispose() {
    this.envTarget.dispose();
    this.cube?.dispose();
    this.pmrem?.dispose();
  }
}
