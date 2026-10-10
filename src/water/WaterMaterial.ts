import * as THREE from 'three/webgpu';
import {
  abs, attribute, BRDF_Lambert, cameraFar, cameraNear, cameraPosition, cameraProjectionMatrix, cameraViewMatrix, clamp, diffuseColor,
  dFdx, dFdy, dot, exp, float, floor, fract, fwidth, getScreenPosition, getScreenPositionFromClip, hash, inverseSqrt, length, log2,
  luminance, max, min, mix, normalize, normalView, perspectiveDepthToViewZ, pmremTexture, positionView, positionWorld, pow,
  reflect, refract, screenUV, smoothstep, step, texture, uniform, uv, vec2, vec3, vec4, viewportDepthTexture,
  viewportOpaqueMipTexture, type N,
} from '../gfx/tsl';
import type { SkyLighting } from '../world/SkyLighting';
import { mistU } from '../world/Mist';
import { advect, chopFromFlow, ringField, surfaceU } from './surface';
import { refractDepth } from './Prewarp';
import { fieldNodes, type RiverField } from './RiverField';
import { uw } from './Underwater';
import type { PoolReflector } from './PoolReflector';
import { DETAIL_SLOPE } from './DetailTile';
import type { RiffleField } from './RiffleField';
import { riffleTerms } from './RiffleWater';
import { params } from '../core/params';
import { shoreStones, stoneLift } from './Shore';

/**
 * River water, one material for the whole ribbon.
 *
 * Lit part (three's physical lighting, sun only, shadowed): the foam and bubbles, the sun sparkles (stochastic
 * facets aligned with the sun's half vector) and the sunlight scattered by the water body toward the eye (a
 * forward lobe, so water seen toward the low sun glows), all sun-coloured and gone in shade. A faint GGX lobe.
 * The scene IBL is switched off for this material.
 * Emissive part: the refracted bed through screen space (projected refracted ray, foreground rejection, per-channel
 * Beer-Lambert along the real path), the pool mirror or the sky and haze-lit valley, Fresnel-weighted, the sky
 * light scattered by the water body, faded to the background at the contact line with a thin meniscus.
 */
export const waterU = {
  /**
   * scattering colour of the water body (single-scattering albedo, relative): a dark emerald-teal. Shallow water
   * shows almost none of it (1 - T is small), deep water takes it on as the bed fades out.
   */
  body: uniform(new THREE.Color(0.0039, 0.017, 0.0125)),
  detail: uniform(1),
  /** 1 = physical refraction offset, 0 = none (debug) */
  refraction: uniform(1),
  /** farthest reflected distance (m) the mirror's ripple offset uses; anything beyond wobbles like the sky */
  reflDist: uniform(400),
  /** what low reflections see in this valley: the dark banks and forest (haze is added along the direction) */
  valley: uniform(new THREE.Color(0.03, 0.038, 0.036)),
  /** sky light on foam and bubbles (the IBL is off here); set from the sky meta */
  skyAmbient: uniform(new THREE.Color(0.38, 0.41, 0.43)),
  foam: uniform(1),
  /** albedo of a sun-aligned facet in a sparkle cell (lit by the sun, times Fresnel) */
  glint: uniform(1500),
  /**
   * the glitter the pixels cannot resolve: a soft lobe around the sun's half vector, radiance ~ haze x E / pi x
   * Fresnel at its peak. The sun on rippled water is a scatter of very bright facets with dark water between
   * them; drawn at its true average it reads as a smooth white glare, so the sparkles carry the energy.
   */
  haze: uniform(0.04),
  wind: uniform(new THREE.Vector2(0.9, 0.2)),
  /**
   * luminance cap of the mirrored sky: the HDRI sun disc and its bright halo would mirror as a smooth white sheet;
   * the sun on the water comes from the sparkles instead
   */
  mirrorCap: uniform(0.3),
  /** facet spread of calm water's sparkles (part 8: a night breeze breaks the moon into a column of glints) */
  calmSpread: uniform(0.022),
  /** light on foam and bubbles, and on the bright waterline (part 8: at night broken water is a faint grey) */
  foamLight: uniform(1),
  edgeLight: uniform(1),
};

/**
 * Direct light: Lambert only (foam, bubbles, the body's scattered sunlight, the sparkles and the glitter haze all
 * go through the colour). three's physical model always adds a GGX lobe with f90 = 1, whatever specularIntensity
 * says: at the grazing angle of a 20-degree sun its peak was ~6, a white sheet over the whole glitter path.
 */
class WaterLighting extends THREE.LightingModel {
  direct({ lightDirection, lightColor, reflectedLight }: { lightDirection: N; lightColor: N; reflectedLight: N }) {
    const dotNL = normalView.dot(lightDirection).clamp();
    reflectedLight.directDiffuse.addAssign(dotNL.mul(lightColor).mul(BRDF_Lambert({ diffuseColor: diffuseColor.rgb })));
  }
}

/** Sky and mirror reflections are added in emissiveNode; the scene IBL would add the sky a second time. */
class RiverWaterMaterial extends THREE.MeshPhysicalNodeMaterial {
  setupEnvironment(): null {
    return null;
  }

  setupLightingModel() {
    return new WaterLighting() as unknown as ReturnType<THREE.MeshPhysicalNodeMaterial['setupLightingModel']>;
  }
}

/** The HDRI sun disc (~1e5) overflows half floats: clamp, then cap luminance like the env clamp (24). */
export const tame = (c: N) => {
  const f = c.min(vec3(60000));
  return f.mul(min(float(1), float(24).div(luminance(f).max(1e-4))));
};

/** Soft highlight shoulder on luminance, hue kept: unchanged below `knee`, approaching `top` above it. */
const shoulder = (c: N, knee: N, top: N) => {
  const l = luminance(c).max(1e-4);
  const over = l.sub(knee).max(0);
  return c.mul(min(l, knee).add(over.div(over.div(top.sub(knee).max(1e-3)).add(1))).div(l));
};

/** Henyey-Greenstein phase function (1/sr). */
const hg = (cosT: N, g: number) => float((1 - g * g) / (4 * Math.PI)).div(pow(float(1 + g * g).sub(cosT.mul(2 * g)), 1.5));

/**
 * Mist.ts's sheet on the water as a reflected ray crosses it, averaged over its banks, streaks and wisps: the extinction
 * at the surface per unit of `mistU.density` where the base map says 1, and the scale height (m). It must follow
 * Mist.ts: { share: 0.45, height: 0.9 } for the 0.3-1.5 m sheet at density 0.072 (polish round); { share: 0.2,
 * height: 0.43 } for the scene fix's 0.15-0.6 m sheet at density 0.2: the mean of the bank x streak x wisp product
 * over Mist.ts's noise volume (bank and wisp both follow n1, so it is 0.205, not the 0.18 of the factors' means) and
 * its sheet-weighted scale height, whose product (0.087) is the sheet's mean column depth per unit density.
 */
const MIST_SHEET = { share: 0.2, height: 0.43 };

export function createWaterMaterial(o: {
  field: RiverField;
  /** scene.environment (2k equirect, sun clamped out); its PMREM is shared with the scene */
  env: THREE.Texture;
  detail: THREE.Texture;
  mirror: PoolReflector | null;
  sky: SkyLighting;
  debugView?: string | null;
  /** fast-water features of the riffle and the pool head (RiffleField.ts); null: generic fast water */
  riffle?: RiffleField | null;
}) {
  const m = new RiverWaterMaterial();
  m.name = 'water';
  // the pool mirror + waterline patch (?pool=0: the water as before it)
  const PP = params.pool;
  m.transparent = true; // drawn after the opaques (the viewport copies hold the bed), and kept out of the GTAO pre-pass
  m.depthWrite = true; // splashes and mist behind the surface stay hidden
  m.side = THREE.FrontSide;
  m.ior = 1.333;
  m.metalness = 0;

  const F = fieldNodes(o.field);
  const P = positionWorld;
  const xz = P.xz;
  const D = o.detail;
  const aFlow = attribute('aFlow', 'vec4');
  const inside = F.inside(xz).toVar();
  const fa = F.A(xz).toVar(), fb = F.B(xz).toVar();
  const flow = mix(aFlow.xy, fa.xy, inside).toVar();
  // the ripples ride the ribbon's smooth current. The field's own current shears round every rock (potential flow,
  // ~10 /s at the flanks) and across the cobbles; the flow map stretched the ripple tile several times over there
  // and swirled it round each rock like a slow-shutter frame (W1, W2). The rocks' own water is baked (RiffleField).
  const flowAdv = aFlow.xy.toVar();
  const speedSec = mix(length(aFlow.xy), fb.y, inside).toVar();
  const depthBaked = mix(aFlow.w, fa.w.sub(fb.x), inside).toVar();
  // the deep pool, where no light comes back up from the bed
  const deepK = smoothstep(0.7, 1.8, depthBaked).toVar();
  const bump = fb.z.mul(inside);
  const chop = chopFromFlow(flow).toVar();
  const fp = length(fwidth(xz)).max(1e-4).toVar();
  const dist = length(cameraPosition.sub(P)).toVar();
  const trail = fa.z.mul(inside).toVar();
  // broken water downstream of rocks, in the pool-head seam and over shallow humps in fast water (in slow water a
  // hump does nothing to the surface)
  let turb: N = smoothstep(0.08, 0.5, trail).add(smoothstep(0.05, 0.2, bump).mul(smoothstep(0.9, 1.5, speedSec)).mul(0.5)).min(1).toVar();
  // very shallow water at the edges is glassy: the surface tension and the bed damp the ripples
  const glassy = smoothstep(0.02, 0.18, depthBaked).toVar();
  // fast, shallow riffle water
  const fastW = smoothstep(1.05, 1.8, speedSec).mul(inside).toVar();
  // splash rings: slope xy, foam z, the churned patch where the splash went in w (rough like broken water)
  const ring = ringField(xz).toVar();

  // ---- surface slopes. Three octaves of an irregular baked ripple tile (DetailTile.ts), each advected with the
  // two-phase flow map and rotated so the octaves never line up; standing lumps over the riffle (they belong to
  // the bed, so they do not move); glassy ripples stretched along the current where the pool tails out; the
  // water heaped against rocks; splash rings. Decoded slopes have an RMS of 1, so each gain is an RMS slope.
  const dec = (t: N) => t.xy.mul(2).sub(1).mul(DETAIL_SLOPE);
  // the per-place phase offset (the two flow phases never reset in sync everywhere) is held to a stretch of ~0.1: the
  // advected position is xz - flow T (p - 0.5), so the offset's gradient times the advection length T |flow| stretches
  // every advected pattern along the current. From the mip's 0.46 m cells (gradient ~0.5 /m, up to 1.4 /m) that was a
  // median stretch of 1.1 at 1.8 m/s, up to 3: the riffle's ripples, bed refraction and froth smeared and folded over
  // in creased patches like wet paint pushed with a thumb (the slow-shutter look of W1, W2, C3)
  const phaseOff = texture(D, xz.div(7.3)).level(5).z.mul(1.6).mul(min(float(1), float(0.1).div(length(flowAdv).max(0.05))));
  const pa = advect(xz, flowAdv, phaseOff, 0), pb = advect(xz, flowAdv, phaseOff, 0.5);
  const norm = inverseSqrt(pa.w.mul(pa.w).add(pb.w.mul(pb.w))).toVar();
  // the riffle and the pool head (RiffleField.ts): standing waves, water heaped round the rocks, glassy tongues,
  // white water, the plunge into the pool. They replace the generic crest contours, coarse rock relief and trails.
  const rif = riffleTerms(o.riffle ?? null, { sd: uv(), fp, pa, pb, norm, detail: D, flow, fastW, depth: depthBaked, bump });
  const rcovN = rif.cov.oneMinus().toVar();
  turb = max(turb.mul(rif.calm), rif.turb).toVar();
  const rough2 = max(turb, ring.w).toVar();
  const rotate = (v: N, ang: number) => {
    const c = Math.cos(ang), s = Math.sin(ang);
    return vec2(v.x.mul(c).sub(v.y.mul(s)), v.x.mul(s).add(v.y.mul(c)));
  };
  // sample the tile in a rotated frame, return the slope in world axes. The mip bias drops an octave's short
  // components: calm water keeps only long, smooth undulations (its fine ripples twinkled like a noisy high-ISO
  // frame), broken water keeps them all
  const broken = smoothstep(0.06, 0.35, max(chop, rough2)).mul(rif.calm).toVar();
  const calmBias = broken.oneMinus().mul(4.5);
  const octave = (scale: number, ang: number, off: number, bias: N) => {
    const ta = texture(D, rotate(pa.pos, ang).div(scale).add(off)).bias(bias);
    const tb = texture(D, rotate(pb.pos, ang).div(scale).add(off + 0.5)).bias(bias);
    return rotate(dec(ta).mul(pa.w).add(dec(tb).mul(pb.w)).mul(norm), -ang);
  };
  const s1 = octave(2.3, 0.0, 0.0, calmBias), s2 = octave(0.83, 0.65, 0.17, calmBias), s3 = octave(0.27, 2.1, 0.41, float(0));
  // riffle crests: standing waves fixed to the bed in the fast thread of the current, short-crested and stretched
  // across the riffle's course. Every H = 0.5 contour of the tile (height H ~ 0.5 +- 0.125) becomes a thin crest,
  // h = exp(-u²) with u = (H - 0.5) / 0.05: a narrow band with a steep flank on each side (one catches the sky,
  // the other a dark line) and flat, glassy water between the bands; the fine moving ripples ride on top (a2, a3)
  const fastL = mix(float(0.45), float(1), smoothstep(0.6, 1.6, length(flow))).mul(fastW).toVar();
  const RA = -0.24;
  const cxz = rotate(xz, -RA);
  const crT = texture(D, vec2(cxz.x.div(2.2), cxz.y.div(3.6)).add(0.37)).toVar();
  const cu = crT.z.sub(0.5).div(0.05);
  // the crest shape is a nonlinear function of the filtered tile: once a pixel spans a crest (far away, grazing)
  // it only aliases into steep random facets that mirror the sky, so it fades out with the footprint
  const crestG = exp(cu.mul(cu).negate()).mul(float(1).sub(smoothstep(0.035, 0.14, fp))).mul(rcovN);
  const crest = crestG.mul(fastL).toVar();
  const crD = dec(crT);
  const sCrest = rotate(vec2(crD.x, crD.y.mul(0.6)), RA).mul(cu.mul(crestG).mul(-2)).mul(fastL.mul(0.2));
  // breeze patches drifting over the calm pool (cat's paws): finer ripples in patches
  const gustN = texture(D, xz.div(23).sub(waterU.wind.mul(surfaceU.time.mul(0.004)))).level(4).z;
  const gust = smoothstep(0.6, 0.85, gustN).toVar();
  const calmA = mix(float(1), float(0.3), glassy.oneMinus());
  // the still, deep pool keeps its long undulations at 40 %: at full strength they scrambled the far bank's reflection
  // into vertical smears and wobbled the near reflections into blobs (C4). A breeze patch only adds capillary ripples
  // (the fine octave): its long-wave share bent the trunks' reflections into molten-chrome blobs. A film of a few
  // centimetres at the margins carries fine capillary ripples instead of lying glassy, so it reads as water (C2): they
  // wobble the stones a little and catch the open sky
  const calmR = float(1).sub(rough2.min(1));
  const still = PP ? smoothstep(0.5, 0.3, speedSec).mul(smoothstep(0.12, 0.4, depthBaked)).mul(calmR) : float(0);
  const calmK = mix(float(1), float(0.4), still).toVar();
  const film = (PP ? smoothstep(0.006, 0.025, depthBaked).mul(smoothstep(0.22, 0.06, depthBaked)).mul(smoothstep(0.7, 0.3, speedSec)).mul(calmR)
    : float(0)).toVar();
  // the calm pool is nearly glassy: a faint, slow, smooth wobble (RMS slope ~0.008), finer ripples only in the
  // breeze patches
  // fast water carries its relief in short steep wavelets (the finest octave), not in long smooth swells: a long slope
  // shifts the bed and the reflections by one smooth field over metres, which smeared both along curves (W1, W2, C3)
  const a1 = mix(0.0045, 0.07, chop).mul(calmK).mul(mix(1, 0.45, fastW)).add(turb.mul(mix(0.04, 0.015, fastW)))
    .add(gust.mul(PP ? 0.0008 : 0.004)).toVar();
  const a2 = mix(0.0015, 0.06, chop).mul(calmK).add(rough2.mul(0.06)).add(gust.mul(PP ? 0.0015 : 0.004)).add(crest.mul(0.05)).toVar();
  // (the film's ripples are not damped by calmA, which the tile sum below applies)
  const a3 = broken.mul(0.045).add(fastW.mul(0.08)).add(rough2.mul(0.06)).add(gust.mul(PP ? 0.012 : 0.005)).add(crest.mul(0.05))
    .add(film.mul(0.05).div(calmA)).toVar();
  // over the riffle the crests carry the relief: less of the even chop, so the backs between them stay glassy
  const tileK = calmA.mul(mix(mix(float(1), float(0.6), fastW), rif.tiles, rif.cov)).mul(waterU.detail).toVar();
  const sTiles = s1.mul(a1).add(s2.mul(a2)).add(s3.mul(a3)).mul(tileK).toVar();
  // riffle lumps: standing humps fixed to the bed (two scales), stronger behind rocks and over humps
  const lumpT = texture(D, rotate(xz, 0.64).div(3.4).add(0.29)).toVar();
  const lumpT2 = texture(D, rotate(xz, -1.05).div(1.3).add(0.61));
  // (steep enough that the backs tilted away from the eye mirror the dark bank and hide the bed, the faces tilted
  // toward it catch the sky: at half this the riffle read as a flat window onto the cobbles)
  // (halved with the short wavelets above: at full strength these metre-scale humps swirled the reflections)
  const sLump = rotate(dec(lumpT), -0.64).mul(0.05).add(rotate(dec(lumpT2), 1.05).mul(0.04)).mul(fastW.mul(turb.mul(0.7).add(0.8)))
    .mul(rif.lumps);
  // tail-out: the current speeds up out of the pool (section speed rising downstream) and stretches the ripples
  // into long glassy streaks; world x is the river's general course
  const fdir = flow.div(length(flow).max(0.05)).toVar();
  const accel = F.B(xz.add(fdir.mul(1.5))).y.sub(fb.y).mul(inside);
  const tail = smoothstep(0.02, 0.1, accel).mul(smoothstep(0.35, 0.6, speedSec)).mul(fastW.oneMinus()).toVar();
  const tA = texture(D, vec2(pa.pos.x.div(3.1), pa.pos.y.div(0.55)).add(0.13));
  const tB = texture(D, vec2(pb.pos.x.div(3.1), pb.pos.y.div(0.55)).add(0.63));
  const tv = dec(tA).mul(pa.w).add(dec(tB).mul(pb.w)).mul(norm);
  const sTail = vec2(tv.x.mul(0.18), tv.y).mul(tail.mul(0.035));
  // the water heaped against rocks, the trough and V ridges behind them (baked relief, finite differences)
  const e = 0.2;
  const relief = vec2(F.B(xz.add(vec2(e, 0))).w.sub(fb.w), F.B(xz.add(vec2(0, e))).w.sub(fb.w)).div(e).mul(inside);
  // (the tile's thin crest bands at half strength: the riffle bake's rows of standing crests carry that now, and at full
  // strength the bands combed the reflections into fine parallel streaks)
  const slope = sTiles.add(sLump).add(sCrest.mul(0.5)).add(sTail).add(relief.mul(rcovN)).add(rif.slope).add(ring.xy).toVar();
  const nW = normalize(vec3(slope.x.negate(), 1, slope.y.negate())).toVar();
  const nV = normalize(cameraViewMatrix.mul(vec4(nW, 0)).xyz).toVar();
  m.normalNode = nV;
  // the bed is refracted by the short ripples only: the long standing waves shift it by a smooth field, which stretched
  // the riffle bed along the view direction like a motion-blurred photo. The finest octave and the thin crest bands fold
  // the bed image several times within a few pixels (moire-like combed streaks, W2): the bed breaks into pieces at
  // the scale of the wavelets instead. A splash's swell bends the bed at half its slope: at full slope its radial shift
  // fanned the stones under it out into a pinwheel and the rings read as a distortion in the bed rather than as rings
  // on the surface bending the reflected bank (C2)
  const sRef = s1.mul(a1).add(s2.mul(a2)).add(s3.mul(a3.mul(0.4))).mul(tileK).add(ring.xy.mul(0.5)).add(rif.refr);
  const nRefV = normalize(cameraViewMatrix.mul(vec4(normalize(vec3(sRef.x.negate(), 1, sRef.y.negate())), 0)).xyz);

  // ---- refraction: one float depth copy (the scene pass uses depth32float under reversed depth), sampled twice
  const depthCopy = new THREE.DepthTexture(1, 1);
  depthCopy.type = THREE.FloatType;
  const depthTex = viewportDepthTexture(screenUV, null, depthCopy);
  const eyeAt = (uv: N) => perspectiveDepthToViewZ(depthTex.sample(uv), cameraNear, cameraFar).negate();
  const pm = cameraProjectionMatrix;
  // view-space point on the camera ray through uv at eye depth e
  const viewAt = (uv: N, e2: N) => vec3(
    uv.x.mul(2).sub(1).add(pm.element(2).x).mul(e2).div(pm.element(0).x),
    uv.y.oneMinus().mul(2).sub(1).add(pm.element(2).y).mul(e2).div(pm.element(1).y),
    e2.negate(),
  );
  const fragEye = positionView.z.negate();
  const e0 = eyeAt(screenUV).max(fragEye).toVar();
  const P0 = positionView.mul(e0.div(fragEye)); // the opaque point straight behind this pixel
  const upV = cameraViewMatrix.mul(vec4(0, 1, 0, 0)).xyz.toVar();
  const h0 = dot(positionView.sub(P0), upV).max(0).toVar(); // its depth under the surface
  // first-order screen-space refraction: march the refracted ray down to the bed depth and project it. The
  // flat-water part (the bed looks raised and slightly magnified) is smooth; the ripple part wobbles the bed in
  // cells and is capped (more in broken water, where the bed only shows in glimpses), so it never smears. The
  // depth is the smooth baked bed depth, not the depth behind the pixel (that jumps at the outline of anything
  // under the water and would show it twice).
  // (Prewarp.ts: inside its shallow disc, round a trout held just under the surface, the depth of that trout)
  const hOff = mix(h0, refractDepth(xz, depthBaked.clamp(0, 4)), inside).toVar();
  const I = positionView.normalize();
  const refrAt = (n: N) => {
    const r = refract(I, n, 1 / 1.333);
    return getScreenPosition(positionView.add(r.mul(hOff.div(dot(r, upV).negate().max(0.05)))), pm);
  };
  const uvF0 = refrAt(upV).toVar();
  const dUv = refrAt(nRefV).sub(uvF0).mul(waterU.refraction);
  const cap = mix(0.006, 0.022, max(chop, turb));
  // extra blur of the refracted bed: broken water shows it only in glimpses, and at grazing angles even the faint
  // ripples of the calm pool scramble it (the far shelf in C4 read as a crisp pebble field under the reflection)
  // A film of a few centimetres barely displaces the bed whatever the angle: the margins stay crisp (scaled by
  // depth, the grazing blur frosted every shallow edge seen from a low eye, C1 and C2)
  const sinV = cameraPosition.sub(P).normalize().y.toVar();
  const graze = float(1).sub(smoothstep(0.08, 0.4, sinV)).mul(smoothstep(0.06, 0.35, hOff)).toVar();
  // Only what the pixels cannot resolve blurs the bed: the ripples finer than the footprint spread the refracted rays
  // over a small cone. The bed's shift per unit slope (lever) is ~0.25 h looking straight down and ~1.8 h at a grazing
  // view. Resolved ripples wobble the bed in sharp broken pieces instead (a fixed 2-3 mip blur for fast water, from the
  // bank, melted the riffle stones into blobs under a pin-sharp bank: the slow-shutter smear of W1, W2 and C3).
  const unres = a3.mul(a3).mul(smoothstep(0.004, 0.03, fp)).add(a2.mul(a2).mul(smoothstep(0.015, 0.1, fp)))
    .add(a1.mul(a1).mul(smoothstep(0.05, 0.3, fp))).sqrt().mul(calmA);
  const lever = hOff.mul(sinV.oneMinus().mul(sinV.oneMinus()).mul(2.2).add(0.25));
  const slowW = float(1).sub(smoothstep(0.45, 0.9, speedSec));
  // bubbles in broken water veil the bed softly; calm water at a grazing angle keeps its scatter blur (the far shelf
  // under C4's reflection read as a crisp pebble field)
  const extraBlur = log2(lever.mul(unres).div(fp).add(1)).mul(1.4).add(turb.mul(0.35)).add(graze.mul(slowW).mul(2.2)).toVar();
  // and never pushes a sample past the frame (it read the clamped border row there)
  const scrEdge = min(min(screenUV.x, screenUV.x.oneMinus()), min(screenUV.y, screenUV.y.oneMinus()));
  // (a soft cap: the hard one gave every saturated offset the same length, so the bed streaked out radially wherever the
  // offset's direction turned)
  const capS = cap.mul(1.25);
  const ripple = dUv.mul(capS.div(capS.mul(capS).add(dot(dUv, dUv)).sqrt())).mul(smoothstep(0.0, 0.03, scrEdge)).toVar();
  // one sample with the flat refraction, one without it. Looking into deep water at a grazing angle the
  // refracted bed point falls off screen; any clamp or fit of the uv repeats the edge row as streaks, so the
  // colours cross-fade to the unshifted sample instead as the refracted target nears the screen border.
  const edge = min(min(uvF0.x, uvF0.x.oneMinus()), min(uvF0.y, uvF0.y.oneMinus()));
  const wRefr = smoothstep(0.0, 0.06, edge).mul(waterU.refraction).toVar();
  const lookup = (uvIn: N) => {
    const uv = uvIn.clamp(0.0005, 0.9995).toVar();
    const ev = eyeAt(uv);
    const Pt = viewAt(uv, ev);
    // reject samples in front of the water (legs, posts, rod) and above its plane (a rock or the bank behind)
    const okS = step(fragEye.add(0.02), ev).mul(step(dot(Pt.sub(positionView), upV), -0.01));
    const path = length(mix(P0, Pt, okS).sub(positionView)).min(60).toVar(); // metres of water the light crossed
    // forward scattering softens the bed with the length of water crossed: the shallows stay crisp, a bed 2 m
    // down is a soft mottle (its sharp pebbles read as an even grain over the whole pool)
    const col = tame(viewportOpaqueMipTexture(mix(screenUV, uv, okS)).level(clamp(path.mul(0.6).sub(0.7).max(0).add(extraBlur), 0, 5)).rgb);
    return { col, path, ok: okS };
  };
  const A = lookup(mix(screenUV, uvF0, waterU.refraction).add(ripple));
  const B = lookup(screenUV.add(ripple));
  const ok = mix(B.ok, A.ok, wRefr);
  const path = mix(B.path, A.path, wRefr).toVar();
  const T = exp(uw.sigmaT.mul(path).negate()).toVar();
  const bgF = mix(B.col, A.col, wRefr);
  // the bed behind was already fogged over its whole distance: undo the air in front of the surface, which
  // this material's own fog adds back, so the transmitted light is not fogged twice
  const fogF = o.sky.fogFactor.min(0.95).toVar();
  const unfog = (c: N) => max(c.sub(o.sky.fogColor.mul(fogF)).div(float(1).sub(fogF)), vec3(0));
  const bg = unfog(bgF).toVar();
  // at the contact line h0 -> 0, so the samples converge on screenUV and bg is the unrefracted bank behind the water.
  // About a centimetre: on a flat bar (slope ~0.01) a 4 cm fade spread the waterline over metres, and the dry
  // stones only picked up a veil as they went under
  // The stones' own relief decides where the surface ends (Shore.ts), at the opaque point behind the pixel as in the
  // terrain's `under` mask: the waterline wanders round every stone that breaks the surface instead of running along
  // the smooth ground mesh
  const P0w = cameraPosition.add(P.sub(cameraPosition).mul(e0.div(fragEye)));
  const lift = PP && shoreStones.tex
    ? stoneLift(texture(shoreStones.tex, P0w.xz.div(shoreStones.size)).a).mul(smoothstep(0.09, 0.05, h0)) : float(0);
  const hEff = h0.sub(lift).toVar();
  const shore = (PP ? smoothstep(0.0, 0.006, hEff) : smoothstep(0.0, 0.012, h0)).toVar();

  // ---- Fresnel and reflection
  const V = cameraPosition.sub(P).normalize().toVar();
  const Fr = float(0.0204).add(float(0.9796).mul(pow(float(1).sub(dot(nW, V).clamp(0, 1)), 5))).toVar();
  const R0 = reflect(V.negate(), nW);
  const R = normalize(vec3(R0.x, abs(R0.y).add(0.002), R0.z)).toVar(); // wave backs that face down still see the sky
  // pmremTexture applies the scene's environment rotation itself: pass the world direction
  const skyR = pmremTexture(o.env, R, chop.mul(0.12).add(smoothstep(15, 90, dist).mul(0.1)).add(0.03)).rgb;
  // the same zenith deepening and sun-side glow as the visible sky (SkyLighting's background node)
  const up = smoothstep(0.04, 0.75, R.y);
  const cosSun = max(dot(R, o.sky.u.sunDir), 0.0);
  const glow = vec3(o.sky.u.sunColor).mul(pow(cosSun, 4.0).mul(0.22).add(pow(cosSun, 20.0).mul(0.45)).mul(smoothstep(0.8, 0.0, R.y)));
  const sky = skyR.mul(mix(vec3(1.0), vec3(0.58, 0.74, 1.04), up)).add(glow);
  // low reflections see the dark banks and valley walls (up to ~20 degrees from most of the river) through a
  // little morning haze: warm toward the sun, cool away from it
  const valleyR = vec3(waterU.valley).add(o.sky.airlight(R).mul(0.06));
  // the skyline the reflection meets: the valley walls rise ~30 degrees across the valley (north and south), the
  // valley opens to ~10 degrees along its axis (x). A fixed 6-25 degree ramp let wave faces over the riffle mirror
  // open sky where the dark hill stands, a pale sheet under it
  const across = smoothstep(0.3, 0.9, abs(R.z).div(length(R.xz).max(1e-3)));
  const skyline = mix(float(0.17), float(0.5), across);
  const envRefl = mix(valleyR, sky, smoothstep(skyline.sub(0.08), skyline.add(0.05), R.y));
  // screen-space reflection of what the frame already shows (valley walls, banks, rocks) for the water the pool
  // mirror does not cover: the reflected ray is marched in view space against the opaque depth copy. A low
  // reflection off the riffle then picks up the dark hill it faces and only wave faces tilted up toward the sky
  // catch light (the even sky/valley ramp made fast water a pale sheet under a dark hill). Where the ray leaves
  // the frame or finds nothing (the sky), the sky/valley model takes over.
  const Rv = cameraViewMatrix.mul(vec4(R, 0)).xyz.toVar();
  let hitUv: N = vec2(0.5, 0.5);
  let hitW: N = float(0);
  // a ray that skims a surface crosses it between two samples and is already far behind it at the next one: the
  // step after a crossing (the previous sample in front) accepts up to its own length, and the hit moves back to
  // where the two depth gaps say the crossing was. With the fixed tolerance alone every ray grazing the valley
  // floor ~230 m away missed, a straight pale band of open sky along the near water in C1
  let prevUv: N = screenUV;
  let prevGap: N = fragEye.sub(e0);
  let prevD = 0;
  for (const d of [1.5, 3, 5.5, 9, 15, 25, 42, 70, 120, 220, 450, 1000, 2500, 5500]) {
    const q = positionView.add(Rv.mul(d));
    const uvq = getScreenPosition(q, pm);
    const eq = q.z.negate();
    const es = eyeAt(uvq.clamp(0.001, 0.999));
    const onScreen = step(0, min(min(uvq.x, uvq.x.oneMinus()), min(uvq.y, uvq.y.oneMinus()))).mul(step(0.05, eq));
    const gap = eq.sub(es).toVar();
    const crossed = step(prevGap, 0);
    const tol = mix(float(d * 0.45 + 1), float(Math.max(d * 0.45, (d - prevD) * 1.1) + 1), crossed);
    const hit = step(0, gap).mul(step(gap, tol)).mul(onScreen).mul(hitW.oneMinus()).toVar();
    const tc = prevGap.negate().div(gap.sub(prevGap).max(1e-4)).clamp(0, 1);
    hitUv = mix(hitUv, mix(uvq, mix(prevUv, uvq, tc), crossed), hit).toVar();
    hitW = max(hitW, hit).toVar();
    prevUv = uvq;
    prevGap = gap;
    prevD = d;
  }
  const hitEdge = min(min(hitUv.x, hitUv.x.oneMinus()), min(hitUv.y, hitUv.y.oneMinus()));
  const wSSR = hitW.mul(smoothstep(0.0, 0.06, hitEdge)).toVar();
  // the sample was fogged over the eye-to-object path, about the water-to-object path for anything far away; this
  // material adds the eye-to-water part
  // (sharp: the resolved wavelets break the reflected bank into pieces; at mip ~2 it was a smear over broken water)
  const ssrCol = tame(viewportOpaqueMipTexture(hitUv).level(broken.mul(0.6).add(0.25)).rgb);
  let reflection: N = mix(envRefl, ssrCol, wSSR);
  // how far the reflected ray runs to what it mirrors (the pool mirror knows; elsewhere the open valley)
  let reflDm: N = waterU.reflDist;
  let wMirror: N = float(0);
  let calm: N = float(0);
  // share of the sky the water can see (1 in the open; the mirror says where a bank and its trees stand over it)
  let skyVis: N = float(1);
  if (o.mirror) {
    const mir = o.mirror.u;
    // ripples deflect the reflected ray by an angle a. The deflected ray meets the mirrored scene at distance Dm
    // (from the mirror's own depth at the flat lookup); seen from the eye, d away, that content shifts by
    // a Dm / (d + Dm), i.e. a lookup offset of d Dm / (d + Dm) metres at the water: the bank right behind the
    // far shore barely wobbles, the sky wobbles by the full angle
    const Rflat = reflect(V.negate(), vec3(0, 1, 0));
    const uv0 = getScreenPositionFromClip(mir.mat.mul(vec4(P, 1))).toVar();
    // (polish round) the mirror renders only the part of its image the pool can show: lookups stay inside it
    const zMir = o.mirror.depthTex.sample(uv0.clamp(mir.rect.xy, mir.rect.zw));
    const q = mir.inv.mul(vec4(uv0.x.mul(2).sub(1), uv0.y.oneMinus().mul(2).sub(1), zMir, 1)).toVar();
    const Dm = length(q.xyz.div(q.w.max(1e-6)).sub(P)).min(waterU.reflDist);
    const clip = mir.mat.mul(vec4(P.add(R0.sub(Rflat).mul(dist.mul(Dm).div(dist.add(Dm)))), 1));
    const ndc = clip.xy.div(clip.w.max(1e-4));
    // the wobbled lookup stays on the mirror image: toward the frame it fades to the flat one (past the image
    // edge the water fell back to the open sky, a ragged white fringe stuck to the frame)
    const border = min(min(uv0.x, uv0.x.oneMinus()), min(uv0.y, uv0.y.oneMinus()));
    const ruvW = mix(uv0, vec2(ndc.x.mul(0.5).add(0.5), ndc.y.mul(-0.5).add(0.5)), smoothstep(0.0, 0.05, border)).toVar();
    const ruv = ruvW.clamp(mir.rect.xy.max(0.001), mir.rect.zw.min(0.999)).toVar();
    const inView = step(0, border);
    // the mirror is exact on the flat, calm pool at y = POOL_LEVEL; ripples only wobble it
    calm = smoothstep(0.06, 0.02, abs(P.y.sub(mir.planeY))).mul(smoothstep(0.95, 0.5, speedSec));
    wMirror = mir.on.mul(calm).mul(inView).toVar();
    // the ripples the pixels cannot resolve (and the calm pool's own micro-roughness) spread the reflected rays over a
    // small cone: the copy is sharp where it meets what it mirrors (Dm -> 0) and softer for what stands farther behind,
    // softest in the breeze patches. The blur radius in the mirror image's own texels picks its mip level (water fix 2:
    // it was measured in screen pixels, a level too blurred at 0.45 of the frame's resolution: the copy of the bank
    // melted into dark smudges and the backlit crowns into loose bright blobs, C4, P4d)
    const spread = float(0.0016).add(gust.mul(0.008)).add(film.mul(0.004)).add(rough2.mul(0.02));
    const mirTexel = uw.pixAngle.div(o.mirror.resolutionScale);
    const mirLod = log2(spread.mul(Dm).div(dist.add(Dm)).div(mirTexel).add(1)).min(3);
    const mirrorCol = (PP ? o.mirror.colorTex.sample(ruv).level(mirLod) : o.mirror.colorTex.sample(ruv)).rgb.min(vec3(60000));
    if (PP) {
      // The mirror renders its own fog over the water-to-object part of the path only (SkyLighting.installFog) and
      // this material adds the eye-to-water part, so nothing is taken out here. Bright things keep their gradation
      // and colour: a hard luminance cap flattened the sunlit haze between the trunks and the sunlit crowns into flat
      // tan plateaus that the ripples cut into molten-chrome blobs (C4). Only the sun and its halo stay near the open
      // sky's brightness (the sparkles carry the sun); the rest of the sky and the scene get a soft shoulder.
      const isSky = step(o.mirror.depthTex.sample(ruv), 1e-9); // reversed depth: the cleared far plane
      const nearSun = smoothstep(0.85, 0.985, dot(R, o.sky.u.sunDir));
      const knee = mix(float(1.2), mix(float(1.0), waterU.mirrorCap.mul(0.8), nearSun), isSky);
      const top = mix(float(3.0), mix(float(1.8), waterU.mirrorCap.mul(1.6), nearSun), isSky);
      // a ripple can tilt the reflected ray off the mirror image (the margins' capillary ripples catch the open sky
      // above the frame): the sky/valley reflection takes over there instead of the clamped edge row. Only past the
      // image's edge: a fade inside it turned the frame's border rows into a pale strip (C4, C2 right edge)
      const onImg = min(min(ruvW.x, ruvW.x.oneMinus()), min(ruvW.y, ruvW.y.oneMinus()));
      wMirror = wMirror.mul(smoothstep(-0.012, -0.002, onImg)).toVar();
      // a bank and its trees standing over the water hide much of its sky: where the flat reflection meets them within
      // a few tens of metres, the sky-lit body glow is dimmed, so their mirror image is not washed out by an even
      // emerald veil (C4 under the far shore, C6 along it)
      // (water fix 2: to 0.2 over the deep pool. At 0.5 the deep pool under the dark bank still glowed emerald as if lit
      // from inside and outshone the bank's mirror image, C4, P4d)
      skyVis = mix(float(1), mix(float(0.5), float(0.2), deepK), step(1e-9, zMir).mul(float(1).sub(smoothstep(8.0, 45.0, Dm)))
        .mul(wMirror)).toVar();
      reflection = mix(reflection, shoulder(mirrorCol, knee, top), wMirror);
    } else {
      // the mirror image was fogged over the whole eye-water-bank path by its own render; this material fogs its
      // output again over the eye-water part, so that part is taken out here. Its sun and bright halo are capped
      // near the open sky's brightness.
      // (part 3 tried lifting the cap off everything but the mirrored sky: from above, C6's pool then mirrored the
      // bright backlit haze on the far wall and went milky)
      const capped = mirrorCol.mul(min(float(1), waterU.mirrorCap.div(luminance(mirrorCol).max(1e-4))));
      reflection = mix(reflection, unfog(capped), wMirror);
    }
    reflDm = mix(reflDm, Dm, wMirror);
  }
  if (params.mist && !params.noPost) {
    // The mist sheet lying on the water is crossed again by the reflected ray (the mist pass marches each view ray
    // only as far as the water surface): its in-scatter veils the mirrored bank as it veils the bank itself. Into the
    // low sun the pool goes silver with it and the bank's shadows on the water fade under the glare (W4). A mean sheet
    // (MIST_SHEET) and a mean sun visibility along the ray: the pixel's own shadow would print the tree shadows on the
    // water again. The sheet's mean: the base map's amount (0.9 over water deeper than ~0.7 m, a tenth over the shallow
    // margins, thinner over fast water), the share left by the banks, gaps and wisps of its noise, the pool's boost; and
    // clear in front of the lens as the mist pass keeps it (at the full density everywhere C2's and W5's teal shallows
    // went a flat grey).
    const sinR = R.y.max(0.03);
    const poolB = float(1).sub(smoothstep(7.0, 18.0, length(vec2(xz.x.sub(8).mul(0.7), xz.y.add(3))))).mul(0.8).add(1);
    const sig0 = mistU.density.mul(MIST_SHEET.share).mul(smoothstep(0.12, 0.72, depthBaked).mul(0.88).add(0.12))
      .mul(mix(float(1), float(0.55), fastW)).mul(poolB).mul(smoothstep(mistU.clearNear, mistU.clearFar, dist));
    const sheetH = mistU.thick.mul(MIST_SHEET.height);
    const tauM = sig0.mul(sheetH).div(sinR).mul(float(1).sub(exp(reflDm.mul(sinR).div(sheetH).negate())));
    const Tm = exp(tauM.negate());
    const gM = mistU.g, cosM = dot(R, o.sky.u.sunDir);
    const hgM = float(1).sub(gM.mul(gM)).div(pow(float(1).add(gM.mul(gM)).sub(gM.mul(cosM).mul(2)), 1.5)).mul(1 / (4 * Math.PI));
    const sc = vec3(o.sky.u.sunColor);
    // (the forward lobe only: the mist pass's isotropic floor stands for light scattered more than once, which a sheet
    // this thin hardly holds; with it, looking away from the sun, it greyed the teal water by the bar, W5, C2)
    const sunM = mix(vec3(luminance(sc)), sc, mistU.sunSat).mul(vec3(mistU.sunTint)).mul(o.sky.u.sunIntensity).mul(mistU.sunGain)
      .mul(hgM.mul(0.85));
    const skyM = vec3(o.sky.u.horizon).mul(0.55).add(vec3(o.sky.u.zenith).mul(0.6)).mul(1 / Math.PI).mul(mistU.skyGain)
      .mul(vec3(mistU.skyTint));
    // The glare toward the sun is what silvers the pool: its in-scatter only within ~30-60 deg of the sun, the sky-lit
    // part at a twenty-fifth. Across the light (C2 looks ~68 deg from the sun, W5 away from it) the mean sheet laid an
    // even grey sheen over the teal shallows and the channel, which the mist pass's own veil already carries.
    const fwd = smoothstep(0.45, 0.85, cosM);
    reflection = reflection.mul(Tm).add(sunM.mul(fwd.mul(0.9)).add(skyM.mul(0.04)).mul(float(1).sub(Tm)));
  }

  // ---- foam: causal masks turned into patches by multi-scale noise (the tile's height, stretched along the
  // current into streaks and advected with it), coverage growing with the mask, textured with fine bubbles
  // (the Worley lace at ~2 cm cells; at 9 cm its cell borders read as a white net)
  // streaks along the current; in slow water the jet's bubbles string out into long thin lines (at 6:1 they read
  // as a field of short white dashes down the C6 pool)
  // (water fix 2: 3:1 in slow water, bubble rafts and lacy strings. At 11:1 the pool-head foam lines drew long straight
  // hairlines over the calm pool like scratches on glass, C2, W4, W5, C4)
  const strK = mix(float(0.3), float(0.36), smoothstep(0.4, 1.1, speedSec));
  const fUV = (p: N, s: number, off: number) => vec2(p.x.mul(strK), p.y).div(s).add(off);
  const noiseC = (s: number, off: number) => texture(D, fUV(pa.pos, s, off)).z.sub(0.5).mul(pa.w)
    .add(texture(D, fUV(pb.pos, s, off + 0.5)).z.sub(0.5).mul(pb.w)).mul(norm);
  const lace = noiseC(1.15, 0.1).mul(1.3).add(noiseC(0.37, 0.7).mul(0.9)).add(0.5).toVar();
  const bubbles = texture(D, fUV(pa.pos, 0.2, 0.33)).w.mul(pa.w).add(texture(D, fUV(pb.pos, 0.2, 0.83)).w.mul(pb.w));
  // drifting foam on slow water is a lacy net of bubble strings with dark holes (Worley cell borders, ~7 cm cells)
  const slowF = float(1).sub(smoothstep(0.3, 0.8, speedSec)).toVar();
  const lacy = texture(D, fUV(pa.pos, 0.6, 0.47)).w.mul(pa.w).add(texture(D, fUV(pb.pos, 0.6, 0.97)).w.mul(pb.w));
  // white water only where the current breaks: at and behind rocks (the baked trails), where the riffle pours into
  // the pool, over humps just under fast water; a little at fast margins. Nothing scattered over open water.
  const shoreFoam = smoothstep(0.04, 0.0, h0).mul(smoothstep(1.3, 1.9, speedSec)).mul(0.3);
  // humps make white water only where the current pours over them within a few centimetres of the surface (as
  // a mask over every hump of the cobble bed it scattered white confetti over the whole riffle)
  const bumpFoam = smoothstep(0.1, 0.22, bump).mul(smoothstep(0.09, 0.03, depthBaked)).mul(smoothstep(1.4, 2.0, speedSec)).mul(0.3);
  // natural sources are max-combined and stay below solid white (lace keeps showing); splash foam can go solid
  const natural = max(max(trail.mul(rcovN), shoreFoam), max(bumpFoam, rif.foam)).min(mix(0.85, 0.97, rif.heavy));
  const mask = clamp(max(natural, ring.z).mul(waterU.foam), 0, 1).toVar();
  // splash foam is a round froth of small bubbles, not streaks: isotropic noise
  const froth = texture(D, pa.pos.div(0.3).add(0.21)).z;
  const drift = mix(lace, lacy.mul(0.9).add(lace.mul(0.3)), slowF.mul(rif.iso.oneMinus()));
  const pattern = mix(mix(drift, rif.pattern, rif.iso), froth, smoothstep(0.0, 0.05, ring.z.sub(natural)));
  // the noise peaks alone clear the threshold: without a source they scattered white flecks over the calm pool.
  // Soft-edged, and frothy rather than a flat white: the bubble net shows through everywhere.
  // (part 8 round 2, critic r1: from above the pool head's white water read as one smooth, round bright patch in the
  // same place at every hour, "a reflection that ignores where the sun or moon is"): past ~10 m, where its bubble lace is
  // finer than a pixel, the heavy mass and its plume break into rafts and streaks along the current over dark water
  const farFoam = smoothstep(9.0, 30.0, dist).mul(max(rif.heavy, smoothstep(0.3, 0.7, rif.aer)));
  // (the lace's own streaked octaves: no extra fetches on every water pixel, C2 and C3 sit on the GPU budget)
  const rafts = smoothstep(0.42, 0.66, lace.mul(0.75).add(0.1));
  const cover = smoothstep(float(1).sub(mask), float(1.4).sub(mask), pattern.mul(0.8).add(mask.mul(0.3)))
    .mul(smoothstep(0.02, 0.1, mask)).mul(mix(float(1), rafts.mul(0.85).add(0.15), farFoam));
  // froth, not paint: the bubble cells darker in their middles and coarser clumps shading the patch (a flat
  // white fill read as splats of paint at the rocks)
  const clump = noiseC(0.6, 0.27).add(0.5).clamp(0, 1);
  const foam = cover.mul(mix(mix(0.2, 1.0, bubbles), 1, rif.solid)).mul(mix(0.5, 1.0, clump)).mul(shore).toVar();
  // White water has height: heaped crests lit bright on the sunward side and grey on the other, darker down in the gaps,
  // highest where it piles onto the rocks and in the plunge (a flat white fill read as paper cut-outs laid on the
  // water, W3, C3). Its slope comes from the screen derivatives of a heap height, faded once a lump is a few pixels.
  const heapK = rif.solid.mul(1.5).add(0.5);
  const heap = cover.mul(clump.mul(0.7).add(0.3)).mul(mix(0.75, 1.0, bubbles)).toVar();
  const dPx0 = dFdx(xz), dPy0 = dFdy(xz), hx0 = dFdx(heap.mul(heapK)), hy0 = dFdy(heap.mul(heapK));
  const det0 = dPx0.x.mul(dPy0.y).sub(dPx0.y.mul(dPy0.x));
  const gHeap = vec2(dPy0.y.mul(hx0).sub(dPx0.y.mul(hy0)), dPx0.x.mul(hy0).sub(dPy0.x.mul(hx0)))
    .div(det0.abs().max(1e-9).mul(det0.greaterThan(0).select(1, -1))).mul(0.045)
    .mul(float(1).sub(smoothstep(0.012, 0.05, fp))).toVar();
  const gHeapC = gHeap.mul(min(float(1), float(1.4).div(length(gHeap).max(1e-6))));
  const nF = normalize(vec3(slope.x.mul(0.5).add(gHeapC.x).negate(), 1, slope.y.mul(0.5).add(gHeapC.y).negate())).toVar();
  // sunlit fraction relative to what the lighting model applies with the water's own normal; sky light by facing
  // and by how deep in the heap a point sits
  const heapSun = max(dot(nF, o.sky.u.sunDir), 0).div(max(dot(nW, o.sky.u.sunDir), 0.05)).min(2.5);
  const heapOcc = mix(mix(0.55, 0.38, rif.heavy), 1.0, smoothstep(0.15, 0.85, heap)).mul(nF.y.mul(0.3).add(0.7));
  // bubbles in the water column of broken water: the bed shows only in glimpses
  const aer = smoothstep(0.15, 0.7, mask).mul(0.7).add(fastW.mul(0.04)).add(fastL.mul(0.08)).add(crest.mul(0.15)).max(rif.aer.mul(0.85))
    .min(0.9).mul(shore).mul(mix(float(1), rafts.mul(0.6).add(0.4), farFoam)).toVar();

  // ---- sun sparkles: each small cell (about 1.6 px, at least 1.5 cm) holds one facet tilted at random around
  // the local surface; where it lines up with the sun's half vector it flashes. Fed through the lit colour:
  // sun-coloured, shadowed, and gone under foam.
  const Hs = normalize(o.sky.u.sunDir.add(V));
  const cell = max(float(0.015), fp.mul(2.0));
  const gp = mix(pb.pos, pa.pos, step(0.5, pa.w)).div(cell);
  const id = floor(gp), fr = fract(gp);
  const tw = floor(surfaceU.time.mul(9));
  const seed = id.x.mul(12.9898).add(id.y.mul(78.233)).add(tw.mul(37.719));
  const r1 = hash(seed), r2 = hash(seed.add(17.17)), r3 = hash(seed.add(41.3));
  // facet spread: calm water is a near mirror, its sparkles stay in a tight sun path (spread evenly over the pool
  // they read as a high-ISO twinkle); broken water and the riffle crests have steep faces that catch even a side sun
  const sig = mix(waterU.calmSpread, float(0.2), broken).add(rough2.mul(0.25)).add(gust.mul(0.02)).add(crest.mul(0.15));
  const facet = normalize(vec3(slope.x.add(r1.sub(0.5).mul(sig.mul(2))).negate(), 1, slope.y.add(r2.sub(0.5).mul(sig.mul(2))).negate()));
  const align = float(1).sub(dot(facet, Hs));
  const dotShape = smoothstep(0.42, 0.1, length(fr.sub(vec2(r3.mul(0.4).add(0.3), r1.mul(0.4).add(0.3)))));
  // a facet flashes only within ~1.5 degrees of the half vector (the sun disc is 0.5 degrees): rare, very bright
  // points. Their true brightness (~1e4) would bloom a dense path into one smooth glare, so each is held to a
  // white-hot ~12 (glint ~ 12 / (Fr N.L E / pi)): countless separate sparkles.
  const sparkle = smoothstep(0.00036, 0.0, align).mul(dotShape).mul(min(Fr.mul(waterU.glint), 12 * Math.PI / 3.2))
    .mul(float(1).sub(foam)).mul(shore);
  // the resolved ripples catch the sun along their own shape: short curved squiggles that follow each ripple, densest in
  // the glitter path toward the sun (only where the water is rippled enough for them to break up; the glassy pool keeps
  // its points)
  const sigRes = length(vec3(a1, a2, a3)).mul(calmA);
  const squiggle = float(1).sub(smoothstep(0.00006, 0.0005, float(1).sub(dot(nW, Hs)))).mul(smoothstep(0.012, 0.045, sigRes))
    .mul(min(Fr.mul(waterU.glint), 12 * Math.PI / 3.2)).mul(0.35).mul(float(1).sub(foam)).mul(shore);
  // the soft lobe the sparkles sit in, as wide as the facet spread, broken up by the resolved ripples
  const hazeLobe = exp(float(1).sub(dot(nW, Hs)).negate().div(sig.mul(sig).mul(0.6).add(0.0015)));

  // ---- the water body: light scattered toward the eye inside the water column (1 - T of it, through the
  // surface). Sun part: a forward-peaked lobe between the refracted sunbeam and the refracted view ray, lit and
  // shadowed through the colour (the lighting gives E N.L / pi; N.L is divided out, 4 pi turns the phase
  // function into a factor of 1 for isotropic scattering); sky part: emissive.
  const rView = refract(V.negate(), nW, 1 / 1.333);
  const cosScat = dot(uw.sunRefr, rView.negate());
  const phase = hg(cosScat, 0.65).mul(0.75).add(0.25 / (4 * Math.PI));
  const column = vec3(1, 1, 1).sub(T).mul(float(1).sub(Fr)).mul(shore).toVar();
  const nDotL = max(dot(nW, o.sky.u.sunDir), 0.05);
  const bodySun = vec3(waterU.body).mul(phase.mul(uw.sunTF).mul(Math.PI * 4)).mul(column).div(nDotL);
  // the sky lights the water column from the whole dome (more than its share through the flat surface, and the light
  // stones under the shallows send it back up through the column): the shallows deepen into a cool teal-green. A deep
  // pool's column has no bed to light it from below and takes the dome as the sun's light, ~0.35-0.5 of isotropic
  // where the eye sees it (water fix 2: at 2.2 x the dome, the deep pool in the bank's shade glowed a saturated emerald
  // as if lit from inside and outshone the bank's mirror image, C4, P4d, C6)
  const bodySky = vec3(waterU.body).mul(waterU.skyAmbient).mul(mix(1.9, 0.55, deepK)).mul(column).mul(skyVis);
  // bubbles scatter white light back, more of it toward the sun
  const bubbleCol = vec3(0.72, 0.77, 0.8).mul(rif.tint);

  // ---- compose. Transmitted bed (absorbed, partly hidden by bubbles) + reflection by Fresnel + body light;
  // foam and bubbles are lit (sun with shadow, plus sky fill), so are the sparkles and the body's sun part.
  // radiance leaving the water into air drops by 1 / n² (0.56); the bed's light trapped by total internal
  // reflection lights the bed again (~x1.15). Without it the bed outshone the mirrored bank at grazing angles.
  // at grazing angles the faint ripples of calm water scatter the light coming up from the bed over a wide cone
  // (the refracted ray is ~4x as sensitive to the slope there): only part of it reaches the eye, and the mirror
  // dominates the strip under the far shore as in a photograph
  // the deep, slack pool's bed is dark (fines and biofilm, little light reaching it): the pool reads as a dark mirror and
  // the water's own colour shows where one looks down into shallow water over light stones
  const deepDark = mix(float(1), float(0.5), smoothstep(1.0, 2.6, depthBaked).mul(slowW));
  const trans = bg.mul(T).mul(float(1).sub(aer.mul(0.65))).mul(0.65).mul(mix(float(1), float(0.45), graze.mul(calm))).mul(deepDark);
  const surface = trans.mul(float(1).sub(Fr)).add(reflection.mul(Fr)).add(bodySky);
  // thin bright meniscus where the surface meets a stone or the bank: ~2-3 px wide whatever the bed slope, but a
  // few millimetres in the world, so it fades out once a pixel is wider than that
  // (part 4: capped at a few pixel footprints: where something stands over the water, h0 jumps by metres across its
  // silhouette and the uncapped fwidth drew a dashed white line along every dock post, brace and stringer)
  const dh = fwidth(h0).max(1e-4).min(fp.mul(3.0));
  // (part 3: it faded out once a pixel spanned 4 mm, i.e. a few metres away: now it holds to ~12 mm, ~10 m)
  // (part 3 round 2: brighter and a little wider, the critic could not find the bright edge line in C2)
  // (round 2: only where the water deepens past a few centimetres within ~0.8 m. Around puddles and films on the
  // flat bar the line traced their outlines like white wires snaking over the stones, C1 and C5)
  const depthAt = (dx: number, dz: number) => {
    const q = xz.add(vec2(dx, dz));
    return F.A(q).w.sub(F.B(q).x);
  };
  const deepNear = max(max(depthAt(0.8, 0), depthAt(-0.8, 0)), max(depthAt(0, 0.8), depthAt(0, -0.8)));
  // and it glints in pieces, not as one continuous ruled line along a bank (C4)
  const glints = mix(float(0.25), float(1), smoothstep(0.38, 0.62, texture(D, xz.mul(0.4).add(0.17)).z));
  const realEdge = mix(float(1), smoothstep(0.035, 0.09, deepNear), inside).mul(glints);
  // The meniscus is a few millimetres tall: it fills a pixel only within a few metres and is a sliver of one at the
  // far bank (at full strength there it traced the C4 waterline as a pale ruled line). On the bars it shows at the
  // real waterline (the deepening test only has to keep it off puddles), broken where the stones break the surface,
  // and it reflects the sky the water can see.
  const dhE = fwidth(hEff).max(1e-4).min(fp.mul(3.0));
  const reach = min(float(1), float(0.0035).div(dist.mul(uw.pixAngle))).pow(2);
  // It climbs every stone it touches: on the side facing the eye it tilts toward it and mirrors the sky above, on the
  // far side it tilts away and mirrors the stone. Crescents, not closed outlines (round each stone in C2 it read as
  // wire). The world gradient of the contact depth (from its screen derivatives) points away from the stone.
  const dPx = dFdx(xz), dPy = dFdy(xz), dHx = dFdx(hEff), dHy = dFdy(hEff);
  const det = dPx.x.mul(dPy.y).sub(dPx.y.mul(dPy.x));
  const gH = vec2(dPy.y.mul(dHx).sub(dPx.y.mul(dHy)), dPx.x.mul(dHy).sub(dPy.x.mul(dHx))).mul(det.greaterThan(0).select(1, -1));
  const facing = smoothstep(-0.35, 0.5, dot(gH.div(length(gH).max(1e-12)), normalize(cameraPosition.xz.sub(xz))));
  const meniscus = PP
    ? smoothstep(dhE.mul(3.0), dhE.mul(0.5), hEff).mul(step(0.0005, hEff)).mul(reach).mul(skyVis.mul(0.85)).mul(facing)
      .mul(mix(float(1), smoothstep(0.012, 0.035, deepNear), inside).mul(glints))
    : smoothstep(dh.mul(3.2), dh.mul(0.6), h0).mul(step(0.0005, h0)).mul(0.8).mul(min(float(1), float(0.012).div(fp)))
      .mul(realEdge);
  // the plume: the bubbles carried under the surface keep the water a milky pale aqua for metres below the plunge
  // (only where they are dense: the thin tail of the plume a dozen metres down the pool milked over C2's shallows)
  const plume = smoothstep(0.2, 0.6, rif.aer).mul(float(1).sub(smoothstep(0.05, 0.4, foam)));
  const foamCol = vec3(0.8, 0.84, 0.86).mul(foam).mul(heapOcc).mul(rif.solid.mul(0.25).add(1)).mul(waterU.foamLight);
  const bubbleL = bubbleCol.mul(aer.mul(mix(0.45, 0.85, plume))).mul(float(1).sub(foam)).mul(waterU.foamLight);
  const haze = hazeLobe.mul(Fr).mul(waterU.haze).div(nDotL).mul(shore);
  m.colorNode = foamCol.mul(heapSun).add(bubbleL).add(vec3(sparkle.add(squiggle))).add(bodySun.add(haze).mul(float(1).sub(foam)));
  m.metalnessNode = float(0);
  m.emissiveNode = mix(bg, surface, shore).mul(float(1).sub(foam)).add(foamCol.add(bubbleL).mul(waterU.skyAmbient))
    .add(vec3(waterU.skyAmbient).mul(meniscus).mul(waterU.edgeLight));

  if (o.debugView) {
    const views: Record<string, N> = {
      normal: nW.mul(0.5).add(0.5),
      flow: vec3(flow.mul(0.3).add(0.5), chop),
      foam: vec3(foam),
      mask: vec3(mask),
      depth: vec3(h0.mul(0.4)),
      path: vec3(path.mul(0.1)),
      T,
      fresnel: vec3(Fr),
      mirror: vec3(wMirror, calm, o.mirror ? o.mirror.u.on : float(0)),
      refrok: vec3(ok.oneMinus(), ok, 0),
      refl: reflection,
      reflF: reflection.mul(Fr),
      // x8, linear with ?nopost&lin: transmitted bed, reflection, body light (sun part at flat N.L)
      split: vec3(luminance(trans.mul(float(1).sub(Fr))), luminance(reflection.mul(Fr)),
        luminance(bodySky.add(bodySun.mul(nDotL).mul(o.sky.u.sunIntensity).div(Math.PI)))).mul(8),
      bg,
      sparkle: vec3(sparkle.mul(0.05)),
      tail: vec3(tail, fastW, turb),
      ssr: vec3(wSSR, hitW, crest),
      aer: vec3(aer),
      // riffle bake (RiffleField.ts): foam, glassy tongues, bubbles; slope magnitude x3, coverage, foam source
      riffle: vec3(foam, rif.glass, aer),
      rslope: vec3(length(rif.slope).mul(3), rif.cov, rif.foam),
      // pool mirror + waterline patch: effective depth at the stone-scale contact (x20 + 0.5), meniscus, sky visibility
      shore: vec3(hEff.mul(20).add(0.5).clamp(0, 1), meniscus, skyVis),
      dbgRef: vec3(sRef.mul(2.5).add(0.5), 0),
      dbgRif: vec3(rif.slope.mul(2.5).add(0.5), 0),
      dbgTile: vec3(sTiles.mul(2.5).add(0.5), 0),
    };
    m.lights = false;
    m.fog = false;
    m.colorNode = vec3(0);
    m.emissiveNode = views[o.debugView] ?? vec3(1, 0, 1);
  }
  return m;
}
