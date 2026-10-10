import * as THREE from 'three/webgpu';
import {
  abs, attribute, cameraPosition, cameraViewMatrix, clamp, cos, cross, diffuseColor, dot, exp, faceDirection, float, floor,
  hash, interleavedGradientNoise, max, min, mix, normalize, normalViewGeometry, normalWorld, output, positionGeometry,
  reflect, screenCoordinate, sin, smoothstep, sqrt, step, texture, uniform, uv, varying, vec2, vec3, vec4, type N,
} from '../gfx/tsl';
import { fieldNodes, type RiverField } from '../water/RiverField';
import { refractionPrewarp as prewarpPoint } from '../water/Prewarp';
import { pixelFootprint, shadowWarp, underwaterSun, uw } from '../water/Underwater';
import { NO_PAINT, SWIM, type SpeciesLook } from './species';

/**
 * Trout skin as a node material, shared by the four species (per-species values are uniforms, so every species
 * runs the same compiled program; only the texture binding differs).
 *
 * Swim (swim_vertex.glsl, ported): a travelling body wave bends each vertex sideways, amplitude growing toward
 * the tail (subcarangiform), the offset at the centre of mass subtracted so the fish stays on its path, plus a
 * C-shaped bend into turns and sculling paired fins. InstancedMesh applies the instance matrix *before*
 * positionNode, so the vertex is rebuilt in world space from per-instance attributes written by FishSchool
 * (aPos = position; aSide = lateral axis, scale; aFwd = nose axis, fin amplitude + pre-warp switch; aSwim = phase,
 * tail amplitude (fraction of the length), turn bend, fin phase); the instance matrix still turns the normals.
 * Normals bend with the slope of the wave (n += slope n.x along the nose axis).
 *
 * Light: wet skin in the air: a slime film over the scales (a clearcoat with sharp highlights along the back and
 * flank), the silvered flanks a broad mirror that flashes in the sun, the back darker than the belly
 * (countershading); thin fins glow amber where the sun shines through them. Under the water skin and water have almost the same
 * index, so the skin's sheen vanishes; what is left is the guanine silvering of the flanks and belly, a broad
 * mirror that flashes when it catches Snell's window or the refracted sun. The direct sun arrives along the
 * refracted beam (steeper, spread over a wider cross-section), absorbed and caustic-patterned like on the bed
 * (Underwater.ts); sky light fades with depth per channel; reflections outside Snell's window see dark water;
 * the belly only gets what the bed sends back. Shadows are cast and received along the refracted beam.
 */
export const fishU = {
  /**
   * multipliers over each species' calibration (species.ts look: the illustrations are as bright as paper; a wet
   * trout is darker and deeper in colour), for A/B in the console
   */
  albedo: uniform(1),
  albedoWater: uniform(1),
  saturation: uniform(1),
  /** countershading strength multiplier */
  back: uniform(1),
  /**
   * the wet film's clearcoat in the air (0 = the old low sheen). Three's clearcoat reflects 4 % head-on (ior 1.5); a
   * film of water and slime ~2.5 %: at 0.9 it laid an even veil of the surroundings over the flank (AgX then bleached
   * the gold to cream)
   */
  coat: uniform(0.6),
  coatRough: uniform(0.08),
  /** sunlight through the thin fins (diffuse transmission, fraction of the fin's albedo) */
  finGlow: uniform(0.75),
  /** extra light returned from the belly side (bed and water column), fraction of the sky light */
  upwell: uniform(0.22),
  /**
   * 1: draw submerged trout where the water's refraction expects them (see refractionPrewarp); 0 if the water ever
   * refracts by the depth behind each pixel instead of the baked bed depth
   */
  prewarp: uniform(1),
  /**
   * shadow casters drawn this much wider and taller than the trout (1 = true size). Only trout in the air (a leap,
   * the thrash, the hands) cast into the sun's shadow map now: under the water BodyShadows.ts draws their shadows
   * crisp (the map's 3.5-texel filter of 2.5 cm texels blurred a 5 cm trout's shadow into a smudge)
   */
  shadowFat: uniform(1.0),
  /**
   * body shape over the models (even, slim tubes with pointed heads, read as twigs and sausages): added depth at the
   * blunt head and the shoulder, width at the head and shoulder, the full belly, the wrist pinched, the tail fin
   * flared; the median fins drawn this much smaller (they hung under the held trout like card); paired fins spread
   */
  head: uniform(0.2),
  shoulder: uniform(0.2),
  headW: uniform(0.32),
  shoulderW: uniform(0.26),
  belly: uniform(0.16),
  wrist: uniform(0.24),
  tail: uniform(0.26),
  medianFin: uniform(0.78),
  finSpread: uniform(1.0),
  /** share of the species' painted gradient over the illustration's colours (spots kept from the illustration) */
  paint: uniform(1),
  /** pinpoint glints of the wet scales in the air */
  sparkle: uniform(6),
  /** cap on the colour raised for the pre-warped path's extra absorption (8 lit the deep lower edge as a gold rim) */
  boostCap: uniform(1.6),
  /** under the water: colour kept (the rest goes to its luminance) and how much the fins' through-light stays */
  waterSat: uniform(0.72),
  waterFinGlow: uniform(0.3),
};

/**
 * The water refracts by the baked bed depth (see water/Prewarp.ts): each submerged trout vertex is drawn where
 * the water will look for it. Off per instance while the body breaks the surface (leaps, thrashing on the line);
 * on for a trout held under the water at the release.
 */
export function refractionPrewarp(field: RiverField, X: N, on: N) {
  return prewarpPoint(field, X, fishU.prewarp.mul(on), cameraPosition);
}

interface FishUnder {
  under: N;
  diffuse: N;
  specular: N;
  sunView: N;
  spread: N;
}

type DirectInput = Parameters<THREE.LightingModel['direct']>[0];
type Builder = Parameters<THREE.LightingModel['direct']>[1];

class FishLighting extends THREE.PhysicalLightingModel {
  constructor(private u: FishUnder | null, clearcoat: boolean, private finN: N | null) {
    super(clearcoat);
  }

  direct(input: DirectInput, builder: Builder) {
    const u = this.u;
    let lightDirection = input.lightDirection as N, lightColor = input.lightColor as N;
    if (u) {
      // under water the sun comes down the refracted beam: steeper, and its power spread over a wider cross-section
      // (the same flux per horizontal area, so a flat bed and a fish's back agree; a flank gets ~3x less than in air)
      lightDirection = normalize(mix(lightDirection, u.sunView, u.under));
      lightColor = lightColor.mul(mix(float(1), u.spread, u.under));
    }
    super.direct({ ...input, lightDirection, lightColor }, builder);
    if (this.finN) {
      // a fin is a thin membrane: light from behind comes through it, tinted by the membrane (amber, the rays darker);
      // under the water most of it is gone (the lit-through fins drew a glowing rim round every trout seen from above)
      const through = max(dot(this.finN, lightDirection).negate(), 0);
      const glow = u ? fishU.finGlow.mul(mix(float(1), fishU.waterFinGlow, u.under)) : fishU.finGlow;
      (input.reflectedLight.directDiffuse as N).addAssign(lightColor.mul(diffuseColor.rgb).mul(through.mul(glow)));
    }
  }

  indirect(builder: Builder) {
    super.indirect(builder);
    const { reflectedLight } = (builder as unknown as { context: { reflectedLight: { indirectDiffuse: N; indirectSpecular: N } } }).context;
    // the membrane takes in the sky through its far face too (a tail thrown up against the light, in the body's
    // shadow, was an unlit black flag)
    if (this.finN) {
      const k = this.u ? mix(float(1), fishU.waterFinGlow, this.u.under) : float(1);
      reflectedLight.indirectDiffuse.mulAssign(float(1).add(fishU.finGlow.mul(1.2).mul(k)));
    }
    if (!this.u) return;
    reflectedLight.indirectDiffuse.mulAssign(this.u.diffuse);
    reflectedLight.indirectSpecular.mulAssign(this.u.specular);
  }
}

class FishNodeMaterial extends THREE.MeshPhysicalNodeMaterial {
  fishUnder: FishUnder | null = null;
  finNormal: N | null = null;

  setupLightingModel() {
    return new FishLighting(this.fishUnder, this.useClearcoat, this.finNormal) as unknown as ReturnType<THREE.MeshPhysicalNodeMaterial['setupLightingModel']>;
  }
}

export interface FishMaterialOptions {
  map: THREE.Texture;
  noseZ: number;
  length: number;
  field: RiverField | null;
  fins: boolean;
  look: SpeciesLook;
  /** caustics, refracted sun, absorbed sky and refracted shadows (off: ?troutdbg=nouw) */
  underwater: boolean;
  /** debug view: 'flat' (unlit albedo), 'normal' (bent normal) */
  debug?: string | null;
}

const TAU = Math.PI * 2;
const LUM = vec3(0.2126, 0.7152, 0.0722);
const F0_SKIN = 0.0278; // ior 1.4 in air

/**
 * The swim, as nodes: the world position of the bent vertex and the slope of the wave (for the normals). The
 * position is built from the per-instance attributes and the raw geometry only (not from positionLocal, which
 * positionNode overwrites), so the same node is safe in the vertex position, the varyings and the shadow pass.
 */
export function swimNodes(o: { noseZ: N; length: N; fins: boolean }) {
  const aSwim = attribute('aSwim', 'vec4');
  const aSide = attribute('aSide', 'vec4');
  const aFwd = attribute('aFwd', 'vec4');
  const aPos = attribute('aPos', 'vec4');
  const K = TAU / SWIM.wavelength;
  const g = positionGeometry;
  const env = (s: N) => s.mul(s.mul(1.3).sub(0.45)).add(0.15);
  const s = clamp(o.noseZ.sub(g.z).div(o.length), 0, 1);
  const sc = o.noseZ.div(o.length);
  const ph = s.mul(K).sub(aSwim.x);
  const A = aSwim.y.mul(o.length);
  const sn = sin(ph), cs = cos(ph);
  const ds = s.sub(sc);
  const bendL = aSwim.z.mul(o.length);
  // lateral offset (model metres): the body wave minus its value at the centre of mass, plus the turn bend
  const lat = A.mul(env(s)).mul(sn).sub(A.mul(env(sc)).mul(sin(sc.mul(K).sub(aSwim.x)))).add(bendL.mul(ds).mul(ds));
  // d(lat)/ds
  const slope = A.mul(s.mul(2.6).sub(0.45).mul(sn).add(env(s).mul(K).mul(cs))).add(bendL.mul(ds).mul(2));
  const side = aSide.xyz, scale = aSide.w, fwd = aFwd.xyz;
  const up = cross(fwd, side);
  // aFwd.w packs the fin amplitude (0..1) and the instance's refraction pre-warp switch (+2)
  const warp = step(1.5, aFwd.w);
  const finAmp = aFwd.w.sub(warp.mul(2));
  // held in the air (aPos.w = sag, a fraction of the length): the hands hold it under the throat behind the gills
  // (s 0.29) and round the wrist of the tail (s 0.79); the body sags between them, the head and the tail beyond the
  // hands droop, and the soft belly is pressed in under the fingers and bulges out round them
  const sagL = max(aPos.w, 0).mul(o.length);
  // in the air (aPos.w < 0, FishSchool): the leap's arch, head and tail down round the middle, so the C reads even
  // when the trout is rolled flank-on to the camera and its sideways bend points at the lens
  const arch = min(aPos.w, 0).mul(o.length);
  const u = s.sub(0.54).div(0.25);
  const between = max(float(1).sub(u.mul(u)), 0);
  const tailEnd = max(s.sub(0.81), 0).div(0.19);
  const headEnd = max(float(0.29).sub(s), 0).div(0.29);
  const droop = between.mul(between).add(tailEnd.mul(tailEnd).mul(0.8)).add(headEnd.mul(headEnd).mul(0.6));
  const grip = (c: number, w: number) => exp(s.sub(c).div(w).pow(2).negate());
  const belly = smoothstep(0.0, o.length.mul(0.07), g.y.negate());
  const press = grip(0.29, 0.03).add(grip(0.79, 0.022));
  const bulge = grip(0.225, 0.03).add(grip(0.355, 0.03)).add(grip(0.745, 0.022).mul(0.6)).add(grip(0.835, 0.022).mul(0.6));
  const dent = press.mul(0.4).sub(bulge.mul(0.22)).mul(belly);
  const vSag = sagL.mul(dent.sub(droop)).add(arch.mul(s.sub(sc)).mul(s.sub(sc)));
  // the shape over the model (fishU head..finSpread): a deeper, blunt head and shoulder over a full belly, a wider
  // head and shoulders, the wrist pinched and the tail fin flared; on the fin cards the median fins shrink toward the
  // body (their roots sink into it) and the paired fins spread out to the sides
  const bump = (c: number, w: number) => exp(s.sub(c).div(w).pow(2).negate());
  const below = smoothstep(0.0, 0.012, g.y.negate());
  const deep = float(1).add(bump(0.17, 0.075).mul(fishU.head)).add(bump(0.36, 0.17).mul(fishU.shoulder))
    .add(bump(0.47, 0.15).mul(fishU.belly).mul(below)).sub(bump(0.83, 0.045).mul(fishU.wrist))
    .add(smoothstep(0.87, 1.0, s).mul(fishU.tail));
  const wide = float(1).add(bump(0.14, 0.1).mul(fishU.headW)).add(bump(0.34, 0.18).mul(fishU.shoulderW))
    .sub(bump(0.83, 0.05).mul(fishU.wrist.mul(0.5)));
  let gx: N = g.x.mul(wide), gy: N = g.y.mul(deep);
  if (o.fins) {
    const median = float(1).sub(smoothstep(0.002, 0.006, abs(g.x)));
    const mid = smoothstep(0.08, 0.16, s).mul(float(1).sub(smoothstep(0.8, 0.88, s)));
    gy = gy.mul(mix(float(1), fishU.medianFin, median.mul(mid)));
    gx = gx.mul(float(1).add(attribute('aFin', 'float').mul(fishU.finSpread)));
  }
  // the instance transform (as the instance matrix applies it), then the bend along the lateral axis
  let p: N = aPos.xyz.add(side.mul(gx.add(lat)).add(up.mul(gy.add(vSag))).add(fwd.mul(g.z)).mul(scale));
  if (o.fins) {
    // paired fins scull: aFin is 0 on the median fins and at a paired fin's root, 1 at its tip
    const lr = g.x.sign();
    const fl = attribute('aFin', 'float').mul(finAmp).mul(sin(aSwim.w.add(lr.mul(0.6))));
    p = p.add(up.mul(fl.mul(0.009).mul(scale))).add(side.mul(lr.mul(fl).mul(0.004).mul(scale)));
  }
  return { position: p, slope, side, fwd, warp, centre: aPos.xyz, scale, up, s };
}

export function createFishMaterial(o: FishMaterialOptions): THREE.MeshPhysicalNodeMaterial {
  const m = new FishNodeMaterial();
  m.name = o.fins ? 'trout-fins' : 'trout-body';
  const uNose = uniform(o.noseZ), uLen = uniform(o.length);
  const swim = swimNodes({ noseZ: uNose, length: uLen, fins: o.fins });
  const field = o.field;
  const warped = field && o.underwater ? refractionPrewarp(field, swim.position, swim.warp) : null;
  m.positionNode = warped ? warped.position : swim.position;
  // where the trout really is (lighting, shadows); positionWorld is the drawn (pre-warped) point
  const P = varying(swim.position, 'vFishReal');

  // normals bend with the wave: n += (d lat / dz) n.x along the nose axis
  const kN = varying(swim.slope.div(uLen), 'vFishBend');
  const sideV = varying(cameraViewMatrix.mul(vec4(swim.side, 0)).xyz, 'vFishSide');
  const fwdV = varying(cameraViewMatrix.mul(vec4(swim.fwd, 0)).xyz, 'vFishFwd');
  const nG = normalViewGeometry;
  const bent = normalize(nG.add(fwdV.mul(kN.mul(dot(nG, sideV)))));
  m.normalNode = o.fins ? bent.mul(faceDirection) : bent;

  // colour from the illustration, calibrated toward a wild fish (per-species uniforms: one program for all four)
  const look = o.look;
  const uAlb = uniform(look.albedo).mul(fishU.albedo), uAlbW = uniform(look.albedoWater).mul(fishU.albedoWater);
  const uSat = uniform(look.saturation).mul(fishU.saturation), uGamma = uniform(look.gamma);
  const uBack = uniform(new THREE.Vector3(...look.back)), uBackK = uniform(look.backK).mul(fishU.back);
  const uSilver = uniform(look.silver);
  const tex = texture(o.map, uv());
  const lum = dot(tex.rgb, LUM);
  const hi = max(tex.r, max(tex.g, tex.b)), lo = min(tex.r, min(tex.g, tex.b));
  const sat = hi.sub(lo).div(hi.max(1e-3));
  // the pale, unsaturated parts of the paintings are the silvered flanks and the belly (falling edge written as
  // 1 - smoothstep: reversed edges are undefined in WGSL and returned 0 here, so no trout ever silvered)
  const silver = smoothstep(0.3, 0.62, lum).mul(float(1).sub(smoothstep(0.12, 0.38, sat)));
  // wet skin: deeper colour; the fins (thin membranes) a little duller
  const wetT = max(mix(vec3(lum), tex.rgb, o.fins ? uSat.mul(0.85) : uSat), vec3(0, 0, 0)).pow(vec3(uGamma));
  const nM = attribute('normal', 'vec3');
  const sV = varying(swim.s, 'vFishS');
  // the species' own colouring from the back to the belly (species.ts look.paint; the illustrations read as biscuit
  // with even freckles, the rainbow as a goldfish): the illustration keeps only its markings, the dark spots and
  // pepper and (red-dominant) the brown's red spots in their pale halos
  const pt = look.paint ?? NO_PAINT;
  const v3 = (c: readonly number[]) => uniform(new THREE.Vector3(c[0], c[1], c[2]));
  // the head keeps most of the illustration (its eye, jaw line and gill cover); so does the tail fin (part of the body
  // mesh: painted it read as a cream card), tinted darker below
  const headW = float(1).sub(smoothstep(0.1, 0.26, sV));
  const tailW = smoothstep(0.84, 0.89, sV);
  const uPK = uniform(pt.k).mul(fishU.paint).mul(float(1).sub(headW.mul(0.7))).mul(float(1).sub(tailW)), uBandK = uniform(pt.bandK), uRedK = uniform(pt.redK);
  // height in the body's slice (FishSchool mergeTrout: aFin on the body), -1 belly .. +1 back
  const aH = o.fins ? float(0) : attribute('aFin', 'float');
  let grad: N = mix(v3(pt.flank), v3(pt.head), headW.mul(0.8));
  grad = mix(grad, v3(pt.back), max(smoothstep(0.0, 0.75, aH), smoothstep(0.35, 0.9, nM.y)));
  grad = mix(grad, v3(pt.belly), smoothstep(0.35, 0.85, aH.negate()));
  const band = float(1).sub(smoothstep(0.08, 0.3, abs(aH.sub(0.02)))).mul(smoothstep(0.1, 0.2, sV)).mul(float(1).sub(smoothstep(0.8, 0.9, sV)));
  grad = mix(grad, v3(pt.band), band.mul(uBandK));
  const redness = float(1).sub(smoothstep(0.15, 0.35, tex.g.div(tex.r.max(1e-3)))).mul(smoothstep(0.05, 0.12, tex.r)).mul(uRedK);
  const halo = smoothstep(0.42, 0.62, lum).mul(float(1).sub(smoothstep(0.18, 0.4, sat))).mul(smoothstep(-0.5, -0.1, aH))
    .mul(float(1).sub(smoothstep(0.2, 0.5, aH))).mul(uRedK);
  grad = mix(grad, v3(pt.belly), halo.mul(0.7));
  grad = mix(grad, vec3(0.012, 0.01, 0.008), float(1).sub(smoothstep(0.03, 0.1, lum)));
  grad = mix(grad, v3(pt.red), redness);
  // fins: amber-olive membranes; the tail darker and browner
  const finTint = mix(v3(pt.fin), v3(pt.fin).mul(vec3(0.5, 0.46, 0.36)), tailW);
  const finK = uniform(pt.k).mul(fishU.paint);
  const finT = wetT.mul(mix(vec3(1, 1, 1), finTint, finK));
  const wet = o.fins ? finT : mix(wetT, grad, uPK).mul(mix(vec3(1, 1, 1), finTint, tailW.mul(finK)));
  // countershading: the back seen from above is darker and more olive than the painting's dorsal edge (the side
  // projection squeezes the whole back into that edge); the model normal's height picks it out
  const backK = o.fins ? float(0) : smoothstep(0.15, 0.85, nM.y).mul(uBackK);
  // in the air the silvered flank is a broad mirror (its diffuse colour gives way to the reflection)
  const silverA = o.fins ? float(0) : silver.mul(uSilver);
  const tone = (o.fins ? wet.mul(0.62) : mix(wet, wet.mul(vec3(uBack)), backK)).mul(float(1).sub(silverA.mul(0.35)));
  const albedo = tone.mul(uAlb);
  m.colorNode = o.fins ? vec4(albedo, tex.a) : albedo;
  if (o.fins) {
    // the membrane's soft edge (the repainted skins ramp their fin alpha over a few texels): dithered, not cut
    m.alphaTest = 0.5;
    m.alphaTestNode = interleavedGradientNoise(screenCoordinate.xy).mul(0.9).add(0.05);
    m.side = THREE.DoubleSide;
    m.finNormal = m.normalNode;
  }
  m.metalnessNode = float(0);
  m.ior = 1.4;
  // the slime film over the scales in the air: a clearcoat with sharp highlights along the back and flank
  m.clearcoatNormalNode = m.normalNode;
  m.clearcoatRoughnessNode = fishU.coatRough;
  // the film is heaviest along the curve of the back: a long soft streak of sky there, less low on the flank
  // (the thin fins barely: their sheen of the bright sky at grazing angles veiled the dark tail as a beige card)
  const coatAir = fishU.coat.mul(o.fins ? float(0.12) : smoothstep(-0.2, 0.8, nM.y).mul(0.8).add(0.6));
  // pinpoint glints: here and there a scale catches the light (a random per scale cell of the skin's UV)
  const cell = floor(uv().mul(vec2(260, 130)));
  const spark = o.fins ? float(0) : smoothstep(0.95, 0.99, hash(cell.x.add(cell.y.mul(1031))));
  // under the film the skin itself reflects little (film and skin have nearly the same index): its broad sheen was
  // the suede look; the guanine silvering keeps its mirror
  const airRough = o.fins ? float(0.62) : mix(float(0.34), float(0.24), silverA).mul(float(1).sub(spark.mul(0.55)));
  const airSpec = o.fins ? float(0.1) : mix(float(0.35), float(0.75), silverA).mul(float(1).add(spark.mul(fishU.sparkle)));
  const airSpecColor = vec3(mix(float(1), float(9), silverA));

  if (field && o.underwater && warped) {
    const F = fieldNodes(field);
    const surfY = F.A(P.xz).w;
    const h = surfY.sub(P.y);
    const under = smoothstep(0.0, 0.025, h);
    const depth = h.max(0);
    // sky light fading with depth (red first); from below only what the bed and the water column send back
    const skyT = exp(uw.kd.mul(depth).negate());
    const N = normalWorld;
    const belly = mix(fishU.upwell, float(1), smoothstep(-0.5, 0.55, N.y));
    const diffuse = mix(vec3(1, 1, 1), skyT.mul(belly), under);
    // reflections, seen along the refracted view ray: inside Snell's window the sky (absorbed on the way up),
    // outside it the underside of the surface mirroring the dark water
    const V = cameraPosition.sub(P).normalize();
    const vy = sqrt(max(float(1).sub(float(1).sub(V.y.mul(V.y)).div(1.777)), 0));
    const Vw = normalize(vec3(V.x.div(1.333), vy, V.z.div(1.333)));
    const R = reflect(Vw.negate(), N);
    const win = smoothstep(0.6, 0.72, R.y);
    const Tup = exp(uw.sigmaT.mul(depth.div(R.y.max(0.25))).negate());
    const specular = mix(vec3(1, 1, 1), Tup.mul(win), under);
    m.fishUnder = {
      under,
      diffuse,
      specular,
      sunView: normalize(cameraViewMatrix.mul(vec4(uw.sunRefr.negate(), 0)).xyz),
      spread: uw.sunAir.y.div(uw.sunRefr.y.negate().max(0.08)),
    };
    // wet skin in the air; under water only the silvering reflects (broad, guanine platelets are not a mirror)
    // (the thin fins, lit through from every side under the water, keep their pale edges there)
    // the water dulls the colours, more with depth (a metre down a trout is a grey-green ghost)
    const satW = mix(float(1), fishU.waterSat, under).mul(float(1).sub(smoothstep(0.3, 1.2, depth).mul(0.45)));
    const albedoW = mix(vec3(dot(tone, LUM)), tone, satW).mul(mix(uAlb, uAlbW, under));
    m.colorNode = o.fins ? vec4(albedoW, tex.a) : albedoW;
    m.roughnessNode = mix(airRough, mix(float(0.5), float(0.36), silver), under);
    const f0w = mix(float(0.004), float(0.11), silver), f90w = mix(float(0.06), float(0.38), silver);
    m.specularIntensityNode = mix(airSpec, f90w, under);
    m.specularColorNode = mix(airSpecColor, vec3(f0w.div(f90w.mul(F0_SKIN))), under);
    // the film is gone under the water (skin and water have almost the same index)
    m.clearcoatNode = coatAir.mul(float(1).sub(under));
    // sun: caustics, Fresnel loss and absorption along the refracted beam, shade under foam; shadow lookups and
    // casters moved along the beam (the swum position, so the shadow swims too)
    const sun = underwaterSun(field);
    const foot = pixelFootprint();
    (m as unknown as { receivedShadowNode: (s: N) => N }).receivedShadowNode = (s) => s.mul(sun(P, surfY, foot));
    // looked up 10 cm toward the sun: a trout is never shadowed by its own (fattened) caster, only by the banks,
    // rocks and other trout (its far side is dark anyway: N.L)
    m.receivedShadowPositionNode = shadowWarp(P.add(uw.sunRefr.negate().mul(0.1)), surfY);
    // (optionally fattened across the body about the centre line, so the filtered shadow keeps a readable core);
    // a submerged trout (pre-warp on) is folded to one point far below: BodyShadows.ts draws its shadow crisp
    const rel = swim.position.sub(swim.centre);
    const along = swim.fwd.mul(dot(rel, swim.fwd));
    const caster = swim.centre.add(along).add(rel.sub(along).mul(fishU.shadowFat));
    m.castShadowPositionNode = mix(shadowWarp(caster, F.A(caster.xz).w), vec3(0, -1000, 0), swim.warp);
    // the water will absorb along the pre-warped path: raise the colour by what that adds (capped)
    const extra = varying(warped.extra, 'vFishExtra');
    const boost = exp(uw.sigmaT.mul(extra)).min(vec3(fishU.boostCap));
    m.outputNode = vec4(output.rgb.mul(boost), output.a);
  } else {
    m.roughnessNode = airRough;
    m.specularIntensityNode = airSpec;
    m.specularColorNode = airSpecColor;
    m.clearcoatNode = coatAir;
  }

  if (o.debug === 'flat') {
    m.lights = false;
    m.colorNode = o.fins ? vec4(tex.rgb, tex.a) : tex.rgb;
  } else if (o.debug === 'normal') {
    m.lights = false;
    const nW = normalize(cameraViewMatrix.transpose().mul(vec4(bent, 0)).xyz);
    m.colorNode = o.fins ? vec4(nW.mul(0.5).add(0.5), tex.a) : nW.mul(0.5).add(0.5);
  } else if (o.debug === 'silver') {
    m.lights = false;
    m.colorNode = o.fins ? vec4(vec3(silver), tex.a) : vec3(silver);
  }
  return m;
}
