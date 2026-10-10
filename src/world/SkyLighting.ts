import * as THREE from 'three/webgpu';
import {
  abs, backgroundRotation, cameraPosition, dot, equirectUV, exp, float, fog, max, min, mix, normalWorldGeometry,
  positionWorld, pow, select, smoothstep, step, texture, uniform, vec3, type N,
} from '../gfx/tsl';
import { params } from '../core/params';
import { POOL_LEVEL } from './River';

/**
 * The HDRI sun sits ESE; rotating the sky 76 deg about Y puts the low morning sun in the north-east
 * (a summer sunrise). That side-lights the main views, keeps the far (north) bank in cool shade and
 * gives the pool a dark reflection to see into.
 */
export const SKY_ROTATION_Y = (76 * Math.PI) / 180;
import type { Assets, SkyMeta } from '../core/Assets';
import type { TimeOfDay } from '../core/params';

const SKY_FILES: Record<TimeOfDay, { bg: string; env: string; meta: string }> = {
  dawn: { bg: 'hdri/qwantani_dawn_puresky_4k.hdr', env: 'sky/dawn_env.hdr', meta: 'sky/dawn.json' },
  morning: { bg: 'hdri/qwantani_morning_puresky_4k.hdr', env: 'sky/morning_env.hdr', meta: 'sky/morning.json' },
  golden: { bg: 'hdri/qwantani_sunset_puresky_4k.hdr', env: 'sky/golden_env.hdr', meta: 'sky/golden.json' },
  night: { bg: 'hdri/qwantani_night_puresky_4k.hdr', env: 'sky/night_env.hdr', meta: 'sky/night.json' },
};

/** Play-area centre used to aim the sun's shadow camera. */
const FOCUS = new THREE.Vector3(2, 0, 4);

export class SkyLighting {
  readonly sun = new THREE.DirectionalLight(0xffffff, 1);
  readonly sunDir = new THREE.Vector3(0.7, 0.35, 0.5).normalize();
  meta!: SkyMeta;

  /** shared with water, mist and particle shaders */
  readonly u = {
    sunDir: uniform(new THREE.Vector3(0.7, 0.35, 0.5).normalize()),
    sunColor: uniform(new THREE.Color(1, 0.85, 0.65)),
    horizon: uniform(new THREE.Color(0.8, 0.85, 0.9)),
    zenith: uniform(new THREE.Color(0.2, 0.3, 0.5)),
    /** sun irradiance (the directional light's intensity), for in-scattering */
    sunIntensity: uniform(9.3),
    /** clear air: long-range blue aerial perspective */
    hazeDensity: uniform(1 / 9500),
    hazeHeight: uniform(1400),
    /** morning aerosol pooled in the valley: forward-scatters the low sun */
    valleyDensity: uniform(1 / 4200),
    valleyHeight: uniform(140),
    // thin: at 0.0012 its sunward glow over 10-20 m of water was as bright as the water and flattened its reflections
    mistDensity: uniform(0.0005),
    mistHeight: uniform(4.0),
    mistBase: uniform(0.0),
    fogBrightness: uniform(0.92),
    /** metres in front of the camera without valley aerosol (the mist pass marches the near air) */
    valleyClear: uniform(110),
    /** metres beyond that over which the aerosol fades in (smoothstep) */
    valleyRamp: uniform(params.valleyRamp ?? 500),
    /** extra warmth of the sun's forward lobe in the haze */
    hazeWarm: uniform(new THREE.Color(1.0, 0.8, 0.56)),
  };

  constructor(private scene: THREE.Scene) {
    const s = this.sun;
    s.castShadow = true;
    // (polish round) 2048², was 4096²: the redraw every third frame is fill-bound on the needle cards (4.8 ms at C3,
    // 4.1 of it the trees; 2.7 ms at 3072², 1.65 ms at 2048²). The PCF radius is in texels, so it is scaled to keep
    // the same 8.9 cm softening on the ground, which already blurs away anything finer than a texel (5 cm); the fine
    // cross-hatch the larger maps left in shaded dirt and planks is mostly gone
    const ms = params.shadowMap;
    s.shadow.mapSize.set(ms, ms);
    const cam = s.shadow.camera as THREE.OrthographicCamera;
    cam.left = -52; cam.right = 52; cam.top = 52; cam.bottom = -52;
    cam.near = 1; cam.far = 400;
    s.shadow.bias = -0.0004;
    s.shadow.normalBias = 0.04;
    s.shadow.radius = 3.5 * (ms / 4096);
    s.target.position.copy(FOCUS);
    scene.add(s, s.target);
  }

  async load(assets: Assets, tod: TimeOfDay) {
    const f = SKY_FILES[tod];
    const [bg, env, meta] = await Promise.all([assets.hdri(f.bg), assets.hdri(f.env), assets.json<SkyMeta>(f.meta)]);
    this.meta = meta;
    this.scene.background = bg;
    this.scene.environment = env;
    this.scene.backgroundRotation.set(0, SKY_ROTATION_Y, 0);
    this.scene.environmentRotation.set(0, SKY_ROTATION_Y, 0);
    // a clear mountain morning: deep blue overhead fading paler to the horizon, a warm glow on the sun side
    // (strongest low down) and a faint haze band along the horizon
    const view = normalWorldGeometry;
    const dir = backgroundRotation.mul(view);
    const up = smoothstep(0.04, 0.75, view.y);
    let sky: N = texture(bg, equirectUV(dir)).rgb.mul(mix(vec3(1.0), vec3(0.58, 0.74, 1.04), up));
    const cosSun = max(dot(view, this.u.sunDir), 0.0);
    const low = smoothstep(0.8, 0.0, view.y);
    sky = sky.add(vec3(this.u.sunColor).mul(pow(cosSun, 4.0).mul(0.22).add(pow(cosSun, 20.0).mul(0.45)).mul(low)));
    const band = smoothstep(0.16, 0.0, abs(view.y.sub(0.015)));
    sky = mix(sky, vec3(this.u.horizon).mul(vec3(1.02, 1.0, 0.97)), band.mul(0.3));
    this.scene.backgroundNode = sky;
    this.scene.backgroundIntensity = 1.0;
    // the sun carries the contrast: sky fill at 85% keeps open shade cool, clean and detailed without going flat
    this.scene.environmentIntensity = 0.85;
    // background/environment sample with the inverse rotation, so the visible sun sits at R * d
    this.sunDir.fromArray(meta.sunDir).applyEuler(new THREE.Euler(0, SKY_ROTATION_Y, 0)).normalize();
    this.u.sunDir.value.copy(this.sunDir);
    // a touch more gold than the plate's disc measurement: the first hours after sunrise
    const c = new THREE.Color().fromArray(meta.sunColor).multiply(new THREE.Color(1.0, 0.95, 0.86));
    this.u.sunColor.value.copy(c);
    this.u.horizon.value.fromArray(meta.horizonColor);
    this.u.zenith.value.fromArray(meta.zenithColor);
    this.sun.color.copy(c);
    // irradiance of the disc measured from the HDRI, so sun and sky stay in proportion
    this.sun.intensity = Math.max(...meta.sunIrradiance);
    this.u.sunIntensity.value = this.sun.intensity;
    this.sun.position.copy(FOCUS).addScaledVector(this.sunDir, 200);
    this.sun.target.position.copy(FOCUS);
  }

  /**
   * Aerial perspective in three exponential height layers integrated along the view ray:
   *  - clear air (scale height 1.4 km): blue airlight, so distant ranges turn paler and bluer;
   *  - morning aerosol pooled in the valley (140 m): Henyey-Greenstein forward lobe of the sun, so views
   *    toward the low sun glow warm and backlit shade is lifted, while away from the sun it is faint and sky-blue;
   *  - a thin mist over the water (4 m), lit the same way with a broader lobe.
   */
  installFog() {
    const u = this.u;
    const toFrag = positionWorld.sub(cameraPosition);
    const dist0 = toFrag.length();
    const dir = toFrag.div(max(dist0, 1e-4));
    // the pool mirror renders from a virtual camera under the water. Its rays only cross air from the water surface
    // on (the water material adds the eye-to-water part): integrated from the camera's own height, the height layers
    // exploded under the surface (the 4 m mist layer at ~190x its density 21 m down, so from above, C6, the mirror
    // held a flat beige veil instead of the forest)
    const inMirror = params.pool ? step(cameraPosition.y, float(POOL_LEVEL - 0.01)) : float(0);
    const camY = mix(cameraPosition.y, float(POOL_LEVEL), inMirror);
    const dist = dist0.mul(mix(float(1), positionWorld.y.sub(POOL_LEVEL).div(positionWorld.y.sub(cameraPosition.y).max(1e-3)).clamp(0, 1), inMirror));
    const dy = positionWorld.y.sub(camY);

    const heightIntegral = (density: N, scaleH: N, base: N, len: N = dist) => {
      const a = exp(camY.sub(base).negate().div(scaleH));
      const b = exp(positionWorld.y.sub(base).negate().div(scaleH));
      const ratio = select(abs(dy).greaterThan(0.05), a.sub(b).mul(scaleH).div(dy), a);
      return density.mul(len).mul(max(ratio, 0.0));
    };
    const tauAir = heightIntegral(u.hazeDensity, u.hazeHeight, float(0));
    // the valley aerosol only counts beyond the near stand: up close it filled every gap between the trunks with an
    // even sunlit glow (round 1: "smoke in a room"); in there the mist pass marches it through the shadow map
    // instead, so only the air the sun actually reaches lights up (shafts)
    // ...and fades in over the next `valleyRamp` m (the integral of a smoothstep, then full density): at full density
    // from 110 m its sunward glow veiled the valley walls 300-800 m out in one flat beige, forest and gullies gone,
    // while the ranges kilometres off hardly change (-250 m of aerosol in ~3 km)
    const sVal = min(dist0.sub(u.valleyClear), dist).max(0.0);
    const xVal = sVal.div(u.valleyRamp.max(1.0)).min(1.0);
    const lenVal = u.valleyRamp.mul(xVal.mul(xVal).mul(xVal).mul(xVal.mul(-0.5).add(1.0))).add(sVal.sub(u.valleyRamp).max(0.0));
    const tauVal = heightIntegral(u.valleyDensity, u.valleyHeight, float(0), lenVal);
    const tauMist = heightIntegral(u.mistDensity, u.mistHeight, u.mistBase);
    const tau = tauAir.add(tauVal).add(tauMist);
    const factor = float(1).sub(exp(tau.negate()));

    const cosSun = dot(dir, u.sunDir);
    const hg = (g: number) => float((1 - g * g) / (4 * Math.PI)).div(pow(float(1 + g * g).sub(cosSun.mul(2 * g)), 1.5));
    const sunE = vec3(u.sunColor).mul(u.sunIntensity);
    // clear air: sky-blue airlight (bluer than the bright horizon band), a soft Rayleigh lift toward and away from the sun
    const skyTint = mix(vec3(u.horizon).mul(vec3(0.6, 0.76, 1.04)), vec3(u.zenith).mul(2.2), max(dir.y, 0.0).mul(1.4).min(1.0));
    const airCol = skyTint.mul(u.fogBrightness).add(sunE.mul(vec3(0.25, 0.5, 1.0)).mul(cosSun.mul(cosSun).add(1.0).mul(0.004)));
    // aerosol and mist: the sun's forward lobe plus sky light from the blue dome. Toward a low sun the light is
    // peach-gold (it crossed far more air: ref_04) and the dome's blue fades out under the glow
    const toSun = max(cosSun, 0.0);
    const dome = vec3(u.zenith).mul(1.7).add(vec3(u.horizon).mul(0.18)).mul(float(1).sub(toSun.mul(toSun).mul(0.55)));
    const mieCol = sunE.mul(vec3(u.hazeWarm)).mul(hg(0.72).mul(0.85)).add(dome);
    const mistCol = sunE.mul(hg(0.55).mul(0.85)).add(vec3(u.horizon).mul(0.7));
    const color = airCol.mul(tauAir).add(mieCol.mul(tauVal)).add(mistCol.mul(tauMist)).div(max(tau, 1e-6));
    this.fogColor = color;
    this.fogFactor = factor;
    this.scene.fogNode = fog(color, factor);
  }

  /** in-scatter colour and opacity of the air between the camera and the shaded point (set by installFog) */
  fogColor: N = null;
  fogFactor: N = null;

  /**
   * Radiance scattered toward the viewer by the valley's morning aerosol along a direction (the forward lobe of
   * the sun plus the blue dome), as in installFog. Water reflections of the backlit valley use it.
   */
  airlight(dir: N): N {
    const u = this.u;
    const g = 0.72;
    const cosSun = dot(dir, u.sunDir);
    const hg = float((1 - g * g) / (4 * Math.PI)).div(pow(float(1 + g * g).sub(cosSun.mul(2 * g)), 1.5));
    const toSun = max(cosSun, 0.0);
    return vec3(u.sunColor).mul(vec3(u.hazeWarm)).mul(u.sunIntensity).mul(hg.mul(0.85))
      .add(vec3(u.zenith).mul(1.7).add(vec3(u.horizon).mul(0.18)).mul(float(1).sub(toSun.mul(toSun).mul(0.55))));
  }
}
