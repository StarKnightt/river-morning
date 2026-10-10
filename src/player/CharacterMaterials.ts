import * as THREE from 'three/webgpu';
import {
  Fn, float, vec2, vec3, vec4, mix, smoothstep, saturate, dot, normalize, pow, abs, max, min, step, attribute, uniform,
  texture, uv, positionWorld, positionView, normalView, positionViewDirection, cameraViewMatrix,
  mx_noise_float, sin, fract, length, select, luminance, dFdx, dFdy, normalMap, diffuseContribution, BRDF_Lambert, cross,
  sign, atan, exp, faceDirection, normalViewGeometry, normalWorldGeometry, positionLocal, normalLocal, modelWorldMatrix,
  floor, cameraPosition, modelWorldMatrixInverse, uniformArray, int, type N,
} from '../gfx/tsl';
import { fieldNodes, type RiverField } from '../water/RiverField';
import { refractionPrewarp } from '../water/Prewarp';
import { LANDMARKS, COAT_PART } from './CharacterDetail';
import { charShadow, charShadowU, sunMapLookup } from './CharShadow';

/**
 * The river's baked field, set before the angler loads (installPlayer): where his hands, arms, knees or boots are
 * under the water they are drawn where the water's refraction looks for them (water/Prewarp.ts), so a hand holding
 * a trout under the surface shows bent at the surface instead of the stones under it. The sun's shadow pass gets
 * the unwarped position.
 */
export const anglerWater: { field: RiverField | null } = { field: null };

/** `soak`: how much darker the material is wet (cloth soaks dark, skin barely) */
function readsUnderWater<M extends THREE.MeshPhysicalNodeMaterial>(m: M, soak: number): M {
  const field = anglerWater.field;
  if (!field) return m;
  const local: N = m.positionNode ?? positionLocal;
  const world = modelWorldMatrix.mul(vec4(local, 1)).xyz;
  const warped = refractionPrewarp(field, world, float(1), cameraPosition).position;
  m.castShadowPositionNode = local;
  m.positionNode = modelWorldMatrixInverse.mul(vec4(warped, 1)).xyz;
  // the meniscus: where a wrist, a sleeve or a boot goes into the water a hairline film climbs it and catches the sky;
  // the cloth above is soaked dark for a few centimetres
  const above = positionWorld.y.sub(fieldNodes(field).A(positionWorld.xz).w);
  const rim = smoothstep(-0.0015, 0.0008, above).mul(float(1).sub(smoothstep(0.0015, 0.005, above)));
  const glow = vec3(anglerU.sky).mul(rim.mul(0.3));
  m.emissiveNode = m.emissiveNode ? (m.emissiveNode as N).add(glow) : glow;
  if (m.colorNode) m.colorNode = (m.colorNode as N).mul(float(1).sub(float(1).sub(smoothstep(0.015, 0.07, above)).mul(soak)));
  return m;
}

/** ?anglerdbg=self: red where he shades a sun-facing part of himself; =flap: the coat's pocket flaps in red */
const DBG = new URLSearchParams(location.search).get('anglerdbg');

/**
 * The angler's materials (the GLB ships plain PBR): every garment its own matte cloth and tone with its
 * construction (seams, a back vent, rolled lapels, sagging pockets, a waistcoat with its buttons and watch chain, a
 * faded neckerchief), creases and folds where his joints bend, wear where hands and ground touch; weathered,
 * leathery skin; facial hair in the hair's colours that thins into stubble; a pale split-cane rod with a cork grip,
 * varnished silk wraps and a tip that fades to a hair. One lighting model adds what the scene's lights cannot: his
 * own sharp self-shadow (CharShadow), the thin warm halo of fibres and hair against a low sun, light bounced up from
 * the sunlit bar, and a little wrapped diffuse for cloth and skin.
 */
export const anglerU = {
  /** elbow L, elbow R, knee L, knee R bend (0 straight .. 1 fully bent) */
  bend: uniform(new THREE.Vector4()),
  /** world -> head bone (brim shadow) */
  headInv: uniform(new THREE.Matrix4()),
  /** sun direction in head-bone space */
  sunHead: uniform(new THREE.Vector3(0, 1, 0)),
  /** sun direction in world space */
  sunWorld: uniform(new THREE.Vector3(0.72, 0.35, -0.6).normalize()),
  /** irradiance from the sunlit ground below (W/m^2-ish in scene units, onto a down-facing surface) */
  bounce: uniform(new THREE.Color(0.55, 0.47, 0.36)),
  /** rod slung on the back: its strap shows across the coat */
  strap: uniform(0),
  /** extra open-sky irradiance on up-facing surfaces (the HDRI's dome under the canopy reads too dark on cloth) */
  sky: uniform(new THREE.Color(0.12, 0.15, 0.2)),
  /** how far each upper arm reaches forward (left, right; 0 hanging .. 1 forward): drag lines across the back */
  reach: uniform(new THREE.Vector2()),
  /** main camera position and its pixel footprint per metre of distance (the rod tip fades below a pixel) */
  camMain: uniform(new THREE.Vector3()),
  pix: uniform(0.0009),
  /** how far each upper arm hangs below the bind A-pose (left, right; 0 .. 1): the shoulder cap's pose corrective */
  armDown: uniform(new THREE.Vector2()),
  /** hands and cuffs wet from handling a fish (0 dry .. 1 dripping) */
  wet: uniform(0),
  /** the wrists in the rest pose (the sleeve cuffs round them get wet) */
  wristL: uniform(new THREE.Vector3(0.55, 1.0, 0)),
  wristR: uniform(new THREE.Vector3(-0.55, 1.0, 0)),
};

interface LightOpts {
  /** strength of the backlit halo and its tint (linear rgb node) */
  halo: N;
  haloTint: N;
  /** wrapped diffuse past the terminator */
  wrap: number;
  /** 0..1 node: how much the analytic hat brim shades this surface (only without the self-shadow map) */
  brim: N | null;
  /** ground bounce scale */
  bounce: number;
  /** extra soft occlusion of the sun and the fill (the beard over the neck and collar), 1 = none */
  occ: N | null;
  /**
   * The rod's varnished cane: a streak of sun along it where its axis (view space) puts the reflection on the eye's
   * cone, on the sunny side of its width (`thin` 1: the whole width, below a pixel or two); `mask` where it applies
   */
  fiber?: { axis: N; mask: N; thin: N; k: number; sigma: number } | null;
}

/** each rod segment's axis (world, +Y toward the tip), set by the rod rig every update: the varnish streak follows the bend */
export const rodAxisU = uniformArray(Array.from({ length: 8 }, () => new THREE.Vector3(0, 1, 0)), 'vec3');

/** Analytic shadow of the brim (an ellipse in head space) along the sun direction: 1 lit, ~0.1 shaded. */
const brimShadow = Fn(() => {
  const B = LANDMARKS.brim;
  const p = anglerU.headInv.mul(vec4(positionWorld, 1)).xyz;
  const L = vec3(anglerU.sunHead);
  const ly = max(L.y, 0.02);
  const s = float(B.y).sub(p.y).div(ly);
  const h = p.xz.add(L.xz.mul(s));
  const e = length(vec2(h.x.div(B.ax), h.y.sub(B.cz).div(B.az)));
  const inside = fall(1.06, 0.9, e).mul(step(0.0, s)).mul(smoothstep(0.0, 0.05, L.y));
  return float(1).sub(inside.mul(0.9));
});

/**
 * The brim also hides the sky overhead: under it the eyes and forehead get a fraction of the open shade's light
 * (1 open .. ~0.45 right under the brim), so in shade too the eyes sit in shadow.
 */
const brimSkyAO = Fn(() => {
  const B = LANDMARKS.brim;
  const p = anglerU.headInv.mul(vec4(positionWorld, 1)).xyz;
  const e = length(vec2(p.x.div(B.ax), p.z.sub(B.cz).div(B.az)));
  const dy = float(B.y).sub(p.y);
  const k = saturate(float(1).sub(e)).mul(0.19).div(dy.max(0).add(0.012));
  const occ = k.div(k.add(1)).mul(step(0.0, dy)).mul(step(e, 1.0));
  return float(1).sub(occ.mul(0.62));
});

class AnglerLighting extends THREE.PhysicalLightingModel {
  constructor(private o: LightOpts, clearcoat: boolean, sheen: boolean) {
    super(clearcoat, sheen);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  direct(data: any, builder: any) {
    const { lightDirection, reflectedLight, lightNode } = data;
    let lightColor: N = data.lightColor;
    const sun = lightNode?.light?.isDirectionalLight === true;
    if (sun) {
      // his own body, hat and rod: the sharp self-shadow map (the scene's sun map no longer sees him, see CharShadow)
      const self = charShadow(positionWorld, normalWorldGeometry);
      lightColor = lightColor.mul(self);
      // (only where the surface faces the sun: there the red is a shadow he casts on himself)
      if (DBG === 'self') reflectedLight.directDiffuse.addAssign(vec3(float(1).sub(self).mul(charShadowU.on).mul(step(0.15, dot(normalView, lightDirection))), 0, 0).mul(3.0));
      if (this.o.brim) lightColor = lightColor.mul(mix(float(1), brimShadow(), this.o.brim.mul(float(1).sub(charShadowU.on))));
      if (this.o.occ) lightColor = lightColor.mul(this.o.occ);
    }
    if (this.o.wrap > 0) {
      const nl = dot(normalView, lightDirection);
      const extra = saturate(nl.add(this.o.wrap).div(1 + this.o.wrap)).sub(saturate(nl)).max(0);
      reflectedLight.directDiffuse.addAssign(lightColor.mul(extra.mul(0.75)).mul(BRDF_Lambert({ diffuseColor: diffuseContribution })));
    }
    super.direct({ ...data, lightColor }, builder);
    if (sun) {
      // fibres and hair standing off the outline scatter the low sun forward: a thin warm line where the surface
      // turns away from the eye and the light comes from behind it
      const ndv = abs(dot(normalView, positionViewDirection));
      const rim = pow(float(1).sub(ndv), 6.0).mul(2.2).add(pow(float(1).sub(ndv), 2.0).mul(0.25));
      const back = pow(saturate(dot(positionViewDirection.negate(), lightDirection)), 1.3);
      const reach = saturate(dot(normalView, lightDirection).add(0.6).mul(1.1));
      reflectedLight.directDiffuse.addAssign(lightColor.mul(this.o.haloTint).mul(rim.mul(back).mul(reach).mul(this.o.halo)));
      const f = this.o.fiber;
      if (f) {
        const T = f.axis, V = positionViewDirection, L = lightDirection;
        const u = dot(L, T).add(dot(V, T));
        const lobe = exp(u.mul(u).mul(-1 / (2 * f.sigma * f.sigma)));
        const h = normalize(L.add(V));
        const hp = h.sub(T.mul(dot(h, T)));
        const across = smoothstep(0.55, 0.95, dot(normalView, hp.div(max(length(hp), 1e-4))));
        reflectedLight.directSpecular.addAssign(lightColor.mul(lobe.mul(mix(across, float(1), f.thin)).mul(f.mask).mul(f.k)));
      }
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  indirectDiffuse(builder: any) {
    super.indirectDiffuse(builder);
    if (this.o.bounce > 0) {
      // the sunlit gravel and grass light the underside: shadow sides stay brown and keep their folds
      const downV = normalize(cameraViewMatrix.mul(vec4(0, -1, 0, 0)).xyz);
      const down = saturate(dot(normalView, downV).mul(0.5).add(0.5));
      let fill: N = vec3(anglerU.bounce).mul(down).add(vec3(anglerU.sky).mul(float(1).sub(down)));
      if (this.o.occ) fill = fill.mul(this.o.occ);
      builder.context.reflectedLight.indirectDiffuse.addAssign(
        fill.mul(this.o.bounce).mul(BRDF_Lambert({ diffuseColor: diffuseContribution })),
      );
    }
  }
}

class AnglerMaterial extends THREE.MeshPhysicalNodeMaterial {
  lightOpts: LightOpts = { halo: float(0), haloTint: vec3(1), wrap: 0, brim: null, bounce: 1, occ: null };
  setupLightingModel() {
    return new AnglerLighting(this.lightOpts, this.useClearcoat, this.useSheen);
  }
}

/** Perturb a view-space normal by a scalar height field (metres) through screen derivatives (Mikkelsen 2010). */
function bump(nrm: N, height: N): N {
  const dpdx = dFdx(positionView), dpdy = dFdy(positionView);
  const dhdx = dFdx(height), dhdy = dFdy(height);
  const r1 = cross(dpdy, nrm), r2 = cross(nrm, dpdx);
  const det = dot(dpdx, r1);
  const grad = sign(det).mul(r1.mul(dhdx).add(r2.mul(dhdy)));
  return normalize(abs(det).mul(nrm).sub(grad));
}

const srgb = (r: number, g: number, b: number) => new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
const col = (c: THREE.Color) => vec3(c.r, c.g, c.b);
const desat = (c: N, k: number | N) => mix(c, vec3(luminance(c)), k);
/** falling edge, 1 below b .. 0 above a (a > b): WGSL leaves smoothstep with reversed edges undefined */
const fall = (a: number | N, b: number | N, x: N) => float(1).sub(smoothstep(b, a, x));
/** fibrous noise in [0, 1] on rest-space positions (sticks to the cloth, never swims) */
const n01 = (p: N, f: number | N) => mx_noise_float(p.mul(f)).mul(0.5).add(0.5);
/** gaussian bump round c with radii r (rest space) */
const g3 = (p: N, c: readonly [number, number, number], r: readonly [number, number, number]) => {
  const d = p.sub(vec3(c[0], c[1], c[2])).div(vec3(r[0], r[1], r[2]));
  return exp(dot(d, d).negate());
};
/** 1 where `k` (a float attribute holding a small integer code) equals `v` */
const isCode = (k: N, v: number) => step(v - 0.5, k).mul(step(k, v + 0.5));

// ---------------------------------------------------------------- clothes

function clothes(src: THREE.MeshStandardMaterial): THREE.MeshPhysicalNodeMaterial {
  const m = new AnglerMaterial();
  m.name = src.name;
  m.side = THREE.DoubleSide;
  m.metalness = 0;
  const rest = attribute('aRest', 'vec3');
  const region = attribute('aRegion', 'vec4');
  const edge = attribute('aEdge', 'vec4');
  const limb = attribute('aLimb', 'vec4');
  const wear = attribute('aWear', 'vec4');
  const cloth = attribute('aCloth', 'vec4');
  const coat = region.x, trousers = region.y, shirt = region.z, button = region.w;
  const cpart = cloth.x;
  const backP = isCode(cpart, COAT_PART.back), lapelP = isCode(cpart, COAT_PART.lapel), flapP = isCode(cpart, COAT_PART.flap);
  const bOcc = cloth.z;
  const t = uv();
  // the suit texture's folds and seams, without its fine herringbone weave (read as corduroy): a blurred mip
  const tex = texture(src.map!, t).bias(2.5).rgb;
  // the texture's own shading (folds baked by the suit's author) as relative detail round each cloth's tone
  // (its wool averages 0.023 in luminance, its shirt 0.31)
  const detail = pow(luminance(tex).div(mix(float(0.023), float(0.31), shirt)).clamp(0.55, 1.5), 0.7);

  // rest-space position of the painted parts (front view): x his left, y up
  const px = rest.x, py = rest.y, pz = rest.z;
  const ax = abs(px);
  // waistcoat over the shirt front (the coat's lapels cover the V below ~1.42 m), cut in a V below the neckerchief;
  // its row of horn buttons and a watch chain show in the coat's V
  const vTop = float(1.457).add(ax.mul(0.9));
  const waistcoat = shirt.mul(fall(vTop.add(0.003), vTop.sub(0.003), py)).mul(step(0.0, pz));
  const btnD = min(min(length(vec2(px.add(0.004), py.sub(1.315))), length(vec2(px.add(0.004), py.sub(1.36)))),
    min(length(vec2(px.add(0.004), py.sub(1.405))), length(vec2(px.add(0.004), py.sub(1.45)))));
  const wcBtn = waistcoat.mul(fall(0.0058, 0.0042, btnD));
  const btnRim = waistcoat.mul(fall(0.0068, 0.0058, btnD)).mul(float(1).sub(fall(0.0058, 0.0042, btnD)));
  // the chain: from the second button's hole, sagging across to the pocket under his left lapel; links catch light
  const cx = saturate(px.add(0.004).div(0.06));
  const chainY = float(1.405).sub(cx.mul(0.022)).sub(sin(cx.mul(Math.PI)).mul(0.011));
  const chainD = abs(py.sub(chainY));
  const chain = waistcoat.mul(fall(0.0014, 0.0007, chainD)).mul(step(-0.004, px)).mul(step(px, 0.058));
  const links = sin(px.mul(Math.PI * 2 / 0.0032)).mul(0.5).add(0.5);
  // pull wrinkles across the belly between the buttons
  const pull = sin(py.sub(ax.mul(ax).mul(4.0)).mul(Math.PI * 2 / 0.045).add(1.2)).mul(fall(0.06, 0.012, abs(px.add(0.004))));

  // the neckerchief: a big soft cloth round the collar, knotted loosely at the throat, its two ends spilling down
  // the shirt front over the waistcoat; it fills the throat down to the waistcoat (no clean white V)
  const band = shirt.mul(smoothstep(vTop.sub(0.004), vTop.add(0.002), py)).mul(fall(1.587, 1.577, py));
  const knot = shirt.mul(exp(px.div(0.026).pow(2.0).add(py.sub(1.512).div(0.022).pow(2.0)).negate())).mul(step(0.03, pz));
  // below the knot the folded cloth drapes in a broad, crumpled point over the top of the waistcoat; two short
  // ends stick out of the knot
  const drapeW = float(0.05).mul(saturate(py.sub(1.418).div(0.08)));
  const drape = shirt.mul(step(0.0, pz)).mul(step(py, 1.5)).mul(fall(drapeW, drapeW.sub(0.004), abs(px.sub(0.003).add(py.sub(1.46).mul(0.08)))));
  const tail = (x0: number, slope: number, tip: number, w0: number) => {
    const drop = float(1.505).sub(py);
    const xc = float(x0).add(drop.mul(slope));
    const hw = mix(float(w0), float(0.004), saturate(drop.div(1.505 - tip)));
    return shirt.mul(step(tip, py)).mul(step(py, 1.507)).mul(fall(hw, hw.sub(0.0025), abs(px.sub(xc)))).mul(step(0.0, pz));
  };
  const tails = max(tail(-0.012, -0.55, 1.468, 0.011), tail(0.014, 0.45, 1.474, 0.01));
  const kerchief = saturate(max(max(band, knot.mul(1.6)), max(tails, drape)));
  // folds: round the neck converging on the knot, a few soft ones down the ends
  const kAng = atan(px, pz);
  const kNoise = mx_noise_float(rest.mul(vec3(55.0, 25.0, 55.0)));
  const kFold = mix(sin(kAng.mul(7.0).add(py.sub(1.54).mul(70.0)).add(kNoise.mul(1.6))),
    sin(px.mul(Math.PI * 2 / 0.018).add(py.mul(40.0)).add(kNoise.mul(2.2))).mul(0.7), max(tails, drape.mul(float(1).sub(band))));
  // the slung rod's leather strap: over the outer left shoulder to the right hip, front and back of the coat body,
  // passing under the breast pocket (over its flap the strap read as a box-like buckle)
  const sd = abs(px.mul(0.866).add(py.sub(1.18).mul(-0.5)));
  const along = px.mul(0.5).add(py.sub(1.18).mul(0.866));
  const onStrap = coat.mul(float(1).sub(edge.w)).mul(smoothstep(0.95, 1.0, py)).mul(anglerU.strap);
  const strap = onStrap.mul(fall(0.027, 0.023, sd));
  const strapEdge = strap.mul(smoothstep(0.019, 0.024, sd));
  // the cloth bulges either side of the strap and drags in diagonal pulls below it
  const strapBulge = onStrap.mul(exp(sd.sub(0.03).div(0.006).pow(2.0).negate()));
  const strapDrag = onStrap.mul(float(1).sub(fall(0.027, 0.023, sd))).mul(exp(sd.sub(0.025).div(0.03).negate()))
    .mul(sin(along.mul(Math.PI * 2 / 0.05).add(sd.mul(40.0))));

  // cloth tones (front light: dark brown wool coat, darker grey-brown trousers, a greyed working shirt); the
  // coat's colour drifts in big soft patches, not an all-over print
  const patch = n01(rest, 2.6).mul(0.6).add(n01(rest, 6.5).mul(0.4));
  const nap = mx_noise_float(rest.mul(140.0));
  const coatC = mix(col(srgb(0.235, 0.195, 0.145)), col(srgb(0.285, 0.238, 0.178)), patch);
  const trouC = mix(col(srgb(0.19, 0.178, 0.163)), col(srgb(0.232, 0.218, 0.198)), patch);
  const shirtC = mix(col(srgb(0.44, 0.425, 0.39)), col(srgb(0.5, 0.48, 0.44)), patch);
  const wcC = mix(col(srgb(0.145, 0.126, 0.108)), col(srgb(0.18, 0.157, 0.134)), nap.mul(0.5).add(0.5));
  // faded madder-red cotton gone a dull rust-brown: soft, the folds darker, bleached where the sun gets it
  const kerBase = mix(col(srgb(0.19, 0.1, 0.08)), col(srgb(0.29, 0.155, 0.122)), kFold.mul(0.5).add(0.5));
  const kerC = mix(desat(kerBase, 0.25), col(srgb(0.36, 0.25, 0.21)), saturate(patch.sub(0.45)).mul(0.6));
  const btnC = col(srgb(0.1, 0.085, 0.07));
  let albedo: N = coatC.mul(coat).add(trouC.mul(trousers)).add(shirtC.mul(shirt)).add(btnC.mul(button));
  albedo = albedo.mul(detail);
  albedo = mix(albedo, wcC.mul(detail.mul(0.5).add(0.5)).mul(float(1).add(pull.mul(0.08))), waistcoat);
  albedo = mix(albedo, btnC.mul(float(1.2).sub(btnD.mul(60.0))), wcBtn);
  albedo = mix(albedo, btnC.mul(0.6), btnRim.mul(0.8));
  albedo = mix(albedo, col(srgb(0.7, 0.53, 0.27)).mul(links.mul(0.5).add(0.6)), chain);
  albedo = mix(albedo, kerC, kerchief);
  albedo = mix(albedo, col(srgb(0.11, 0.07, 0.045)).mul(nap.mul(0.1).add(0.95)), strap);
  albedo = mix(albedo, col(srgb(0.2, 0.14, 0.09)), strapEdge.mul(0.6));

  // creases: rings round the elbows and knees, deeper the more the joint bends and on its inside; trousers stack
  // over the boots
  const code = limb.z;
  const bendNow = select(code.lessThan(1.5), anglerU.bend.x, select(code.lessThan(2.5), anglerU.bend.y,
    select(code.lessThan(3.5), anglerU.bend.z, anglerU.bend.w)));
  const inside = saturate(limb.y.mul(0.6).add(0.5));
  const creaseAmt = limb.w.mul(step(0.5, code)).mul(float(0.25).add(bendNow.mul(1.3))).mul(inside.mul(0.85).add(0.15));
  const wob = mx_noise_float(rest.mul(17.0)).mul(3.2);
  const ring = sin(limb.x.mul(Math.PI * 2 / 0.042).add(wob).add(limb.y.mul(2.1)));
  // not every ring runs all the way round: folds break up, merge and fork
  const ridge = pow(abs(ring), 0.55).mul(sign(ring)).mul(smoothstep(0.25, 0.65, n01(rest, 13.0)));
  const stackW = trousers.mul(fall(0.29, 0.18, py));
  const stack = stackW.mul(sin(py.mul(Math.PI * 2 / 0.026).add(wob.mul(1.4))));
  // drag lines fanning from each armpit across the back toward the opposite hip, stronger as the arms reach forward
  const dragN = mx_noise_float(rest.mul(9.0)).mul(1.5);
  const drag = (s: number, reach: N) => {
    const dx = px.sub(s * 0.16).mul(s), dy = py.sub(1.36);
    const dist = length(vec2(dx, dy));
    const a = atan(dy, dx);
    const fan = smoothstep(-3.0, -2.6, a).mul(fall(-1.6, -1.9, a));
    return sin(a.mul(23.0).add(dragN)).mul(fan)
      .mul(exp(dist.div(0.11).negate())).mul(smoothstep(0.015, 0.05, dist)).mul(reach.mul(0.65).add(0.35));
  };
  const backDrag = backP.mul(drag(1, anglerU.reach.x).add(drag(-1, anglerU.reach.y)));
  // the centre-back seam (a straight line down the back, from the collar to the vent) and the vent below it: the
  // slit where the left flap overlaps the right, the overlap's shadow on the right flap
  const cb = abs(px);
  const ventM = backP.mul(fall(1.1, 1.04, py));
  const cbSeam = backP.mul(float(1).sub(ventM)).mul(fall(0.0016, 0.0005, cb)).mul(step(py, 1.56));
  const cbWelt = backP.mul(float(1).sub(ventM)).mul(exp(cb.sub(0.004).div(0.0022).pow(2.0).negate())).mul(step(py, 1.56));
  const slit = ventM.mul(fall(0.0013, 0.0004, abs(px.sub(0.0006))));
  const ventShade = ventM.mul(step(px, 0.0)).mul(fall(0.009, 0.0, cb));
  const ventEdge = ventM.mul(step(0.0, px)).mul(fall(0.004, 0.0012, px));
  // lapels roll over at the roll line just inside their edge along the V
  const roll = lapelP.mul(exp(cloth.y.sub(0.009).div(0.005).pow(2.0).negate()));
  const lapelEdge = lapelP.mul(fall(0.0025, 0.0006, cloth.y));
  // felted wool: soft, uneven nap; the linen rumples in long soft folds
  const rumple = shirt.mul(mx_noise_float(rest.mul(vec3(30.0, 12.0, 30.0)))).mul(float(1).sub(kerchief));
  // the other seams: sewn a little darker over a soft raised welt (their distance field is per vertex, so no
  // hairline groove: it zig-zagged along the triangles)
  const sdist = edge.x;
  const groove = fall(0.009, 0.0015, sdist).mul(float(1).sub(button)).mul(float(1).sub(backP));
  const welt = exp(sdist.sub(0.006).div(0.005).pow(2.0).negate()).mul(float(1).sub(button)).mul(float(1).sub(shirt));
  const height = ridge.mul(creaseAmt).mul(0.005).add(stack.mul(0.003))
    .add(nap.mul(0.00035).mul(float(1).sub(shirt)))
    .add(cbWelt.mul(0.0008)).sub(cbSeam.mul(0.0007))
    .add(rumple.mul(0.0025))
    .add(backDrag.mul(0.0016))
    .add(welt.mul(0.0006)).sub(groove.mul(0.0002))
    .add(roll.mul(0.0018)).add(ventEdge.mul(0.0008))
    .add(pull.mul(waistcoat).mul(0.0006))
    .add(kFold.mul(kerchief).mul(0.0014))
    .add(strapBulge.mul(0.0012)).sub(strap.mul(0.0005)).add(strapDrag.mul(0.0009))
    .add(wcBtn.mul(0.0012));
  const valley = saturate(ridge.negate()).mul(creaseAmt).add(saturate(stack.negate()).mul(0.6).mul(stackW))
    .add(saturate(backDrag.negate()).mul(0.5));
  albedo = albedo.mul(float(1).sub(valley.mul(0.36)));

  // seams sewn darker, hems and edges rubbed paler; the vent's slit and the overlap's shade
  const hemLine = fall(0.008, 0.002, edge.y).mul(float(1).sub(button));
  albedo = albedo.mul(float(1).sub(groove.mul(0.22))).mul(float(1).add(welt.mul(0.04)));
  albedo = albedo.mul(float(1).sub(cbSeam.mul(0.55))).mul(float(1).sub(slit.mul(0.85))).mul(float(1).sub(ventShade.mul(0.32)));
  albedo = mix(albedo, desat(albedo, 0.4).mul(1.2), max(lapelEdge.mul(0.3), roll.mul(0.08)));
  // wear gathered where hands and ground touch: dust on the trouser knees and shins and the coat's skirt, rubbed
  // cuffs, elbows, hems and pocket edges (frayed), grime at the collar and pocket mouths, sun-faded shoulders
  const breakup = n01(rest, 26.0).mul(0.6).add(n01(rest, 7.0).mul(0.4));
  const dust = smoothstep(0.4, 0.85, wear.x.mul(breakup.mul(1.4).add(0.05)));
  const fray = smoothstep(0.2, 0.6, nap).mul(hemLine);
  const abr = saturate(wear.y.mul(breakup.mul(1.4))).max(hemLine.mul(0.55)).max(fray).mul(float(1).sub(button));
  albedo = mix(albedo, desat(albedo, 0.45).mul(1.4).add(0.008), abr.mul(0.5));
  albedo = mix(albedo, col(srgb(0.56, 0.53, 0.47)), dust.mul(mix(float(0.55), float(0.2), shirt)).mul(float(1).sub(waistcoat)).mul(float(1).sub(kerchief)));
  albedo = albedo.mul(float(1).sub(wear.z.mul(breakup.mul(0.6).add(0.4)).mul(0.42)));
  albedo = mix(albedo, desat(albedo, 0.35).mul(1.1), wear.w.mul(0.3).mul(coat.add(trousers)));
  // cavity: folds the mesh carries, under the collar, inside the pockets; the lapel edges shade the waistcoat
  const vEdge = float(0.03).add(py.sub(1.36).max(0).mul(0.5)).sub(ax);
  const vShade = shirt.mul(fall(0.012, 0.0, vEdge)).mul(step(py, 1.49)).mul(step(1.25, py));
  albedo = albedo.mul(float(1).sub(saturate(edge.z).mul(0.45))).mul(float(1).sub(vShade.mul(0.35)));
  if (DBG === 'flap') albedo = mix(albedo, vec3(0.8, 0, 0), flapP);
  // the cuffs soaked from holding a fish in the river: darker, a little sheen
  const cuffWet = max(fall(0.13, 0.07, length(rest.sub(vec3(anglerU.wristL)))), fall(0.13, 0.07, length(rest.sub(vec3(anglerU.wristR)))))
    .mul(anglerU.wet).mul(float(1).sub(trousers)).toVar();
  albedo = albedo.mul(float(1).sub(cuffWet.mul(0.4)));
  m.colorNode = albedo;

  // matte wool and linen: broad, weak specular, no sheen on the trousers; worn spots (elbows, cuffs, seat) shine a
  // little; buttons, the strap and the chain glossy
  // (the strap's leather is worn matte: glossy, it flashed white on the pocket flap under it like a buckle)
  const glossy = max(max(button, wcBtn), max(chain, strap.mul(0.3)));
  const worn = saturate(wear.y.mul(breakup.mul(1.5)).sub(0.15)).mul(float(1).sub(shirt));
  const rough = mix(mix(float(0.95), float(0.98), trousers), float(0.84), shirt).sub(worn.mul(0.3)).sub(wear.z.mul(0.1));
  m.roughnessNode = mix(mix(rough, float(0.55), cuffWet), float(0.32), glossy);
  m.specularIntensityNode = mix(mix(float(0.2), float(0.1), trousers).add(worn.mul(0.3)), float(0.85), glossy);
  m.metalnessNode = chain;
  const base = select(faceDirection.greaterThan(0), normalMap(texture(src.normalMap!, t).bias(2.0), vec2(0.5)), normalize(normalViewGeometry));
  m.normalNode = bump(base, height);
  m.aoNode = mix(texture(src.aoMap!, t).r, float(1), 0.6).mul(float(1).sub(bOcc.mul(0.5))).mul(float(1).sub(vShade.mul(0.4)));

  // folds that break the outline: cloth bunches in the crook of a bent elbow and behind a bent knee
  const bunch = exp(limb.x.div(0.05).pow(2.0).negate()).mul(step(0.5, code)).mul(bendNow).mul(saturate(limb.y.mul(1.2).add(0.2)))
    .mul(mx_noise_float(rest.mul(30.0)).mul(0.4).add(0.8)).mul(select(code.lessThan(2.5), float(0.008), float(0.006)));
  // pose corrective for the shoulder cap: as the arm hangs below the bind A-pose, linear blend skinning swings the
  // outer cap out past the sleeve; pull it back in
  const armDown = select(px.greaterThan(0), anglerU.armDown.x, anglerU.armDown.y);
  const capPull = cloth.w.mul(armDown).mul(0.016);
  m.positionNode = positionLocal.add(normalLocal.mul(bunch.add(strap.mul(0.002)).sub(capPull)));

  m.lightOpts = {
    halo: float(1.0).mul(float(1).sub(shirt.mul(0.4))),
    haloTint: mix(albedo.mul(5.0).add(0.12), vec3(1.0, 0.9, 0.78), 0.45),
    wrap: 0.15,
    brim: null,
    bounce: 1.3,
    occ: float(1).sub(bOcc.mul(0.45)),
  };
  m.receivedShadowPositionNode = sunMapLookup();
  return m;
}

// ---------------------------------------------------------------- skin

function skin(src: THREE.MeshStandardMaterial): THREE.MeshPhysicalNodeMaterial {
  const m = new AnglerMaterial();
  m.name = src.name;
  m.metalness = 0;
  const rest = attribute('aRest', 'vec3');
  const zone = attribute('aZone', 'vec4');
  const zone2 = attribute('aZone2', 'vec4');
  const cav2 = attribute('aCav', 'vec2');
  const cav = cav2.x, bOcc = cav2.y;
  const face = zone.x, hand = zone.y, red = zone.z, stub = zone.w;
  const sock = zone2.x, lips = zone2.y, lines = zone2.z, knk = zone2.w;
  const base = texture(src.map!, uv()).rgb;
  const mx = vec3(abs(rest.x), rest.y, rest.z);
  // weathered outdoor skin: sun-browned and leathery, darker than his shirt, the hands darker and dirtier still. The
  // texture's pink cast is dropped: only its relative light and dark (its face averages 0.31 in luminance) is kept
  const detail = pow(luminance(base).div(0.31).clamp(0.55, 1.45), 0.8);
  const blotch = n01(rest, 45.0);
  let c: N = mix(col(srgb(0.49, 0.35, 0.262)), col(srgb(0.4, 0.29, 0.215)), hand).mul(detail).mul(blotch.mul(0.16).add(0.92));
  // weather shows ruddy brown on the nose, cheekbones and ears, and on the knuckles
  c = c.add(vec3(0.034, 0.006, -0.002).mul(red.mul(blotch.mul(0.6).add(0.5))));
  c = c.add(vec3(0.016, 0.004, -0.002).mul(knk));
  c = mix(c, c.mul(vec3(0.8, 0.66, 0.62)), lips.mul(0.6));
  // stubble where the beard thins onto the cheeks and over the lip: dark roots, denser toward the beard's edge
  const grain = n01(rest, 170.0);
  const stubble = stub.mul(grain.mul(0.4).add(0.45));
  c = mix(c, col(srgb(0.14, 0.11, 0.085)), stubble);
  // structure the smooth mesh lacks: hollows under the cheekbones and at the temples, deep sockets
  const hollow = max(g3(mx, [0.048, 1.652, 0.11], [0.017, 0.014, 0.04]), g3(mx, [0.068, 1.732, 0.085], [0.012, 0.016, 0.03]).mul(0.6)).mul(face);
  // creases: crow's feet fanning from the outer eye corners, furrows across the forehead, lines under the eyes,
  // the folds from the nose wings past the mouth corners, two lines between the brows
  const ang = atan(rest.y.sub(1.708), abs(rest.x).sub(0.06));
  const crow = sin(ang.mul(7.0).add(mx_noise_float(rest.mul(70.0)).mul(1.2))).mul(smoothstep(0.056, 0.064, abs(rest.x))).mul(step(rest.y, 1.716));
  const brow = sin(rest.y.mul(Math.PI * 2 / 0.0085).add(mx_noise_float(rest.mul(vec3(40.0, 10.0, 40.0))).mul(1.6)));
  const lineH = select(abs(rest.x).greaterThan(0.045), crow, brow);
  const underEye = exp(rest.y.sub(1.6985).sub(abs(rest.x).sub(0.035).pow(2.0).mul(2.0)).div(0.0016).pow(2.0).negate())
    .mul(smoothstep(0.016, 0.024, abs(rest.x))).mul(fall(0.058, 0.05, abs(rest.x))).mul(step(0.115, rest.z)).mul(face);
  // nasolabial fold: distance to the segment from the nose wing to beside the mouth corner (front view)
  const A = vec2(0.021, 1.657), Bv = vec2(0.034, 1.626);
  const pq = vec2(abs(rest.x), rest.y);
  const ab = Bv.sub(A);
  const tt = saturate(dot(pq.sub(A), ab).div(dot(ab, ab)));
  const naso = exp(length(pq.sub(A.add(ab.mul(tt)))).div(0.0034).pow(2.0).negate()).mul(face).mul(step(0.12, rest.z));
  const glab = exp(abs(rest.x).sub(0.006).div(0.0012).pow(2.0).negate()).mul(smoothstep(1.712, 1.718, rest.y))
    .mul(fall(1.738, 1.73, rest.y)).mul(step(0.13, rest.z)).mul(face);
  const furrow = pow(saturate(lineH.negate()), 2.0).mul(lines).add(naso.mul(0.9)).add(underEye.mul(0.7)).add(glab.mul(0.6));
  c = c.mul(float(1).sub(furrow.mul(0.55))).mul(float(1).sub(hollow.mul(0.26)));
  // sunken eye sockets and the creases the mesh carries (nostrils, lips, between the fingers); dirt in the hands'
  // creases and round the nails, ground into the knuckles
  const ao = float(1).sub(saturate(cav).mul(mix(float(0.6), float(0.85), hand))).mul(float(1).sub(sock.mul(0.42)));
  const dirt = hand.mul(smoothstep(0.45, 0.85, n01(rest, 90.0))).mul(0.3);
  c = c.mul(ao).mul(float(1).sub(dirt));
  // wet hands: darker skin with a wet shine
  const wetH = hand.mul(anglerU.wet).toVar();
  c = c.mul(float(1).sub(wetH.mul(0.22)));
  m.colorNode = c;
  m.aoNode = mix(float(1), brimSkyAO(), face).mul(float(1).sub(bOcc.mul(0.55)));
  // rough and matte: a little oil on the nose and forehead only
  m.roughnessNode = mix(mix(float(0.66), float(0.56), red.mul(face)).add(blotch.mul(0.08)).add(hand.mul(0.1)).add(stubble.mul(0.1)), float(0.24), wetH);
  m.specularIntensityNode = mix(float(0.4).mul(ao).mul(float(1).sub(dirt)), float(0.85), wetH);
  // leathery: fine wrinkles and pores over the whole face, the knuckles' creases
  const fine = mx_noise_float(rest.mul(vec3(110.0, 330.0, 110.0))).mul(0.00014);
  const h = furrow.mul(-0.0012).sub(hollow.mul(0.0035)).sub(sock.mul(0.0014)).add(fine).add(knk.mul(0.0006));
  m.normalNode = bump(normalView, h);
  m.lightOpts = {
    halo: float(0.15),
    haloTint: vec3(0.8, 0.5, 0.35),
    wrap: 0.2,
    brim: face,
    bounce: 1,
    occ: float(1).sub(bOcc.mul(0.6)),
  };
  m.receivedShadowPositionNode = sunMapLookup();
  return m;
}

// ---------------------------------------------------------------- gear atlas: boots, hat, hair, beard, eyes

/** Gear atlas regions (glTF uv, v down) */
function gearRegions() {
  const t = uv();
  const right = step(0.5, t.x), low = step(0.5, t.y);
  return {
    t,
    boots: float(1).sub(right).mul(low),
    hat: right.mul(low),
    hair: float(1).sub(right).mul(float(1).sub(low)),
    beard: right.mul(float(1).sub(low)).mul(float(1).sub(step(0.75, t.x))),
    eyes: step(0.75, t.x).mul(float(1).sub(low)),
  };
}

/**
 * The beard cards (one fur-blob texture on 115 cards) as hair: dense and dark close to the skin, breaking into
 * single strands at its outline, toward its outer surface and high on the cheeks, where it thins into stubble.
 */
function beardShape(map: THREE.Texture) {
  const r = gearRegions();
  const rest = attribute('aRest', 'vec3');
  const gA = attribute('aGear', 'vec4');
  const tx = texture(map, r.t);
  const s = r.t.x.sub(0.5).mul(4.0), w = r.t.y.mul(2.0);
  // locks ~4 mm apart (finer lines alias into a speckle, the "scouring pad"), wavering down the card
  const lineCoord = s.mul(10.0).add(mx_noise_float(vec3(s.mul(3.0), w.mul(2.5), rest.y.mul(40.0))).mul(1.1));
  const strandL = fall(0.42, 0.14, abs(fract(lineCoord).sub(0.5)));
  const outer = smoothstep(0.005, 0.017, gA.y);
  // high on the cheeks (front of the face above the jaw) and up the sideburns
  const thin = saturate(smoothstep(1.645, 1.69, rest.y).mul(smoothstep(0.07, 0.11, rest.z).mul(0.65).add(0.35))
    .add(smoothstep(1.68, 1.714, rest.y).mul(0.6)));
  // solid in its core (lock lines across 115 overlapping cards read as a scouring pad), frayed at each card's soft
  // edge and up the cheeks
  const sparse = saturate(max(thin, fall(0.7, 0.25, tx.a)));
  const alpha = tx.a.mul(mix(float(1.2), strandL.mul(1.3), sparse)).mul(float(1).sub(thin.mul(0.5)));
  return { r, tx, lineCoord, strandL, outer, thin, alpha };
}

function gearAlphaOf(b: ReturnType<typeof beardShape>): N {
  const strand = smoothstep(0.03, 0.4, luminance(b.tx.rgb));
  return b.tx.a.mul(mix(float(1), strand.mul(0.5).add(0.62), b.r.hair)).mul(float(1).sub(b.r.beard)).add(b.alpha.mul(b.r.beard));
}

/** Cut-out of the gear atlas (hair and beard cards; the rest opaque), shared with the self-shadow proxy. */
export function gearAlpha(map: THREE.Texture): N {
  return gearAlphaOf(beardShape(map));
}

function gear(src: THREE.MeshStandardMaterial): THREE.MeshPhysicalNodeMaterial {
  const m = new AnglerMaterial();
  m.name = src.name;
  m.side = THREE.DoubleSide;
  m.metalness = 0;
  const rest = attribute('aRest', 'vec3');
  const gA = attribute('aGear', 'vec4');
  const bs = beardShape(src.map!);
  const { boots, hat, hair, beard, eyes, t } = bs.r;
  const tx = bs.tx;
  const lum = luminance(tx.rgb);
  const breakup = n01(rest, 30.0).mul(0.6).add(n01(rest, 9.0).mul(0.4));

  // boots: dark oiled leather gone matte, grey with dust toward the welt and the toes, creased across the toes
  const bootC = mix(col(srgb(0.11, 0.08, 0.06)), col(srgb(0.17, 0.12, 0.085)), n01(rest, 40.0)).mul(pow(lum.div(0.09).clamp(0.5, 1.5), 0.6));
  const toeRing = sin(rest.z.mul(Math.PI * 2 / 0.009).add(mx_noise_float(rest.mul(80.0)).mul(2.0)));
  const toe = gA.y.mul(boots);
  const scuff = smoothstep(0.62, 0.8, n01(rest, 55.0)).mul(boots);
  let bootA: N = mix(bootC, col(srgb(0.36, 0.33, 0.29)), saturate(gA.z.mul(breakup.mul(1.2))).mul(0.55));
  bootA = mix(bootA, desat(bootA, 0.6).mul(1.9), scuff.mul(0.5));
  bootA = bootA.mul(float(1).sub(saturate(toeRing.negate()).mul(toe).mul(0.4)));

  // felt hat: dark brown-grey, dusty on top, a sweat line soaked through above the band; the brim's outer edge
  // darker and greasy from handling (it read as a pale cardboard edge from behind)
  const B = LANDMARKS.brim;
  const rr = length(vec2(rest.x.add(0.003).div(B.ax), rest.z.sub(0.06).div(B.az)));
  const brimEdge = hat.mul(smoothstep(0.9, 0.99, rr)).mul(step(rest.y, 1.78));
  const hatC = mix(col(srgb(0.14, 0.12, 0.105)), col(srgb(0.19, 0.165, 0.14)), breakup).mul(pow(lum.div(0.05).clamp(0.4, 1.6), 0.5));
  let hatA: N = mix(hatC, col(srgb(0.42, 0.39, 0.34)), saturate(gA.z.mul(breakup.mul(1.1)).sub(0.12)).mul(0.35).mul(float(1).sub(brimEdge.mul(0.75))));
  hatA = hatA.mul(float(1).sub(gA.w.mul(smoothstep(0.3, 0.7, n01(rest, 45.0)).mul(0.5).add(0.25)))).mul(float(1).sub(brimEdge.mul(0.3)));

  // hair and beard: one family of dark brown with a little grey, darker toward the skin; strands from the texture
  const strand = smoothstep(0.03, 0.4, lum);
  const hairBase = mix(col(srgb(0.17, 0.128, 0.096)), col(srgb(0.3, 0.228, 0.168)), n01(rest, 25.0));
  const grey = smoothstep(0.55, 0.85, n01(rest, 140.0)).mul(0.55);
  const nearSkin = fall(0.012, 0.002, gA.y);
  const hairA = mix(hairBase.mul(mix(float(0.75), float(1.3), strand)), col(srgb(0.5, 0.48, 0.45)), grey.mul(strand)).mul(float(1).sub(nearSkin.mul(0.3)));
  // the beard: streaks along the hairs (not a speckle), dark roots, lighter and greyer single hairs at its outline
  const streak = mx_noise_float(vec3(bs.lineCoord.mul(0.7), t.y.mul(6.0), rest.y.mul(30.0))).mul(0.5).add(0.5);
  const greyHair = step(0.94, fract(floor(bs.lineCoord).mul(0.6180339)));
  let beardA: N = desat(hairBase, 0.15).mul(mix(float(0.32), float(0.78), bs.outer.mul(0.6).add(streak.mul(0.4))));
  beardA = beardA.mul(mix(float(0.85), float(1.15), strand.mul(bs.strandL)));
  beardA = mix(beardA, col(srgb(0.36, 0.34, 0.32)), greyHair.mul(0.35).mul(bs.outer.mul(0.6).add(0.4)));

  // eyes: the whites a little dimmed and yellowed (no doll's stare) but open and catching light; brows and lashes
  // the hair's darker brown
  const eyeA = mix(tx.rgb.mul(vec3(0.5, 0.46, 0.4)), col(srgb(0.14, 0.105, 0.08)), fall(0.25, 0.05, lum).mul(0.6));

  let albedo: N = bootA.mul(boots).add(hatA.mul(hat)).add(hairA.mul(hair)).add(beardA.mul(beard)).add(eyeA.mul(eyes));
  albedo = albedo.mul(float(1).sub(saturate(gA.x).mul(0.5).mul(float(1).sub(beard))));
  m.colorNode = albedo;
  m.aoNode = mix(float(1), brimSkyAO(), eyes.add(hair).add(beard).min(1)).mul(float(1).sub(beard.mul(nearSkin).mul(0.35)));
  m.opacityNode = gearAlphaOf(bs);
  m.alphaTest = 0.3;
  m.roughnessNode = boots.mul(mix(float(0.62), float(0.85), saturate(gA.z.add(scuff)))).add(hat.mul(mix(float(0.95), float(0.8), brimEdge)))
    .add(hair.mul(0.6)).add(beard.mul(0.76)).add(eyes.mul(0.12));
  m.specularIntensityNode = boots.mul(0.6).add(hat.mul(0.25)).add(hair.mul(0.45)).add(beard.mul(mix(float(0.12), float(0.28), bs.outer))).add(eyes.mul(1.0));
  // beard cards are smooth curved strips: a strand-scale height along the hairs breaks their cylinder shading
  const height = saturate(toeRing.negate()).mul(toe).mul(-0.0012).add(mx_noise_float(rest.mul(160.0)).mul(0.00025).mul(hat.add(boots)))
    .add(bs.strandL.mul(0.0009).add(streak.mul(0.0004)).mul(beard));
  m.normalNode = bump(normalMap(texture(src.normalMap!, t), vec2(0.6)), height);
  const fuzzy = hat.add(hair).add(beard).min(1);
  m.lightOpts = {
    halo: mix(float(0.35), float(1.6), fuzzy).mul(float(1).sub(brimEdge.mul(0.6))).mul(float(1).sub(beard.mul(0.4))),
    haloTint: mix(albedo.mul(4.0).add(0.15), vec3(1.0, 0.86, 0.66), 0.5),
    wrap: 0.2,
    // the brim shades the eyes, the hair under it and the beard
    brim: eyes.add(hair).add(beard).min(1),
    bounce: 1,
    occ: null,
  };
  m.receivedShadowPositionNode = sunMapLookup();
  return m;
}

// ---------------------------------------------------------------- strands (moustache, beard fringe, loose hairs)

export function strandMaterial(): THREE.MeshPhysicalNodeMaterial {
  const m = new AnglerMaterial();
  m.name = 'Angler_Strands';
  m.side = THREE.DoubleSide;
  m.metalness = 0;
  const t = uv();
  const tone = attribute('aTone', 'float');
  // 4 hairs per card, each tapering to its tip, with uneven lengths
  const k = t.x.mul(4.0).add(tone.mul(7.0));
  const id = k.floor();
  const across = abs(fract(k).sub(0.5));
  const lenH = fract(sin(id.mul(12.9898).add(tone.mul(78.233))).mul(43758.5453)).mul(0.45).add(0.55);
  const width = float(0.24).mul(float(1).sub(t.y.div(lenH)).max(0));
  m.opacityNode = float(1).sub(smoothstep(width.sub(0.08), width, across)).mul(step(t.y, lenH));
  m.alphaTest = 0.5;
  // the hair's own dark brown, dark at the root and lighter at the tips that catch the light
  const base = mix(col(srgb(0.11, 0.085, 0.066)), col(srgb(0.27, 0.205, 0.152)), tone);
  m.colorNode = mix(base, col(srgb(0.42, 0.4, 0.37)), step(0.93, tone).mul(0.6)).mul(float(0.62).add(t.y.mul(0.45)));
  m.aoNode = brimSkyAO();
  m.roughness = 0.72;
  m.specularIntensity = 0.3;
  m.lightOpts = { halo: float(1.8), haloTint: vec3(1.0, 0.85, 0.65), wrap: 0.3, brim: float(1), bounce: 1, occ: null };
  return m;
}

// ---------------------------------------------------------------- rod

/** aRod parts beyond the GLB's segments: 8 reel, 9 crank, 10 the guides added in RodRig */
export const ROD_GUIDE_PART = 10;

/**
 * Split cane: pale honey-straw with darker flecks and node marks, a thin varnish highlight; the cork grip pale and
 * pitted, darker where the hand holds it; the silk wraps a deep see-through claret under varnish, near the cane's own
 * tone; aged nickel bands and wire guides; the brass click reel's rims polished where they catch the light. The thin
 * upper rod fades with its pixel coverage, so the tip tapers to a hair and vanishes against the background.
 * aRod = (rod-space height, part: 0-7 segments, 8 reel, 9 crank, 10 guides; radius off the rod axis; reel rim).
 */
function rod(src: THREE.MeshStandardMaterial): THREE.MeshPhysicalNodeMaterial {
  const m = new AnglerMaterial();
  m.name = src.name;
  const aRod = attribute('aRod', 'vec4');
  const y = aRod.x, part = aRod.y, radius = aRod.z, rim = aRod.w;
  const t = uv();
  const tx = texture(src.map!, t).rgb;
  const mx = max(max(tx.r, tx.g), tx.b), mn = min(min(tx.r, tx.g), tx.b);
  const sat = mx.sub(mn).div(max(mx, 1e-3));
  const reel = step(7.5, part).mul(step(part, 9.5));
  const guide = step(9.5, part);
  // (linear colour) red silk has almost no green; the orange cane and the cork keep ~40 % of their red in green
  const wrap = step(tx.g, tx.r.mul(0.12)).mul(step(0.03, tx.r)).mul(float(1).sub(reel)).mul(float(1).sub(guide));
  // nickel bands and ferrules: bright and grey
  const metalBand = fall(0.5, 0.35, sat).mul(step(0.15, mx)).mul(float(1).sub(reel)).mul(float(1).sub(guide));
  const seat0 = step(part, 0.5);
  const cork = seat0.mul(step(0.062, y)).mul(step(y, 0.296));
  const walnut = seat0.mul(step(-0.045, y)).mul(step(y, 0.03)).mul(float(1).sub(metalBand));
  const cane = float(1).sub(reel).sub(guide).sub(wrap).sub(metalBand).sub(cork).sub(walnut).clamp(0, 1).mul(step(-0.04, y).max(step(0.5, part)));
  // straw cane: long streaks of darker fibre, staggered node marks every ~18 cm
  const streak = mx_noise_float(vec2(y.mul(9.0), t.x.mul(40.0)).xyy).mul(0.5).add(0.5);
  const nodeMark = exp(fract(y.div(0.18).add(part.mul(0.37))).sub(0.5).abs().mul(-60.0));
  // pale enough that the low gold sun warms it to honey, not orange (round 1's cane printed orange in this light)
  let caneC: N = mix(col(srgb(0.66, 0.565, 0.4)), col(srgb(0.8, 0.7, 0.52)), streak);
  caneC = mix(caneC, col(srgb(0.44, 0.35, 0.23)), nodeMark.mul(0.55));
  const pits = smoothstep(0.62, 0.78, mx_noise_float(vec3(y.mul(220.0), t.x.mul(160.0), part)).mul(0.5).add(0.5));
  const held = exp(y.sub(0.15).div(0.05).pow(2.0).negate()).mul(0.3);
  const corkC = mix(col(srgb(0.72, 0.6, 0.44)), col(srgb(0.45, 0.34, 0.22)), pits.mul(0.8)).mul(float(1).sub(held));
  // varnished silk: deep claret-brown, the cane's tone showing through it; the thread lines stay
  const thread = luminance(tx).div(max(luminance(col(srgb(0.42, 0.05, 0.04))), 1e-3)).clamp(0.7, 1.3);
  const wrapC = mix(caneC, col(srgb(0.3, 0.085, 0.05)), 0.72).mul(thread);
  // aged nickel (no white flashes between the wraps); brass plates dull, their rims polished
  const nickelC = col(srgb(0.55, 0.53, 0.49));
  const reelC = mix(mix(tx.mul(vec3(0.72, 0.64, 0.5)), tx, metalBand), col(srgb(0.95, 0.78, 0.46)), rim.mul(reel));
  let albedo: N = caneC.mul(cane).add(corkC.mul(cork)).add(wrapC.mul(wrap)).add(tx.mul(walnut)).add(reelC.mul(reel))
    .add(nickelC.mul(metalBand)).add(col(srgb(0.62, 0.6, 0.55)).mul(guide));
  albedo = albedo.add(tx.mul(float(1).sub(cane.add(cork).add(wrap).add(walnut).add(reel).add(metalBand).add(guide)).clamp(0, 1)));
  m.colorNode = albedo;
  const metal = max(max(metalBand, guide), reel.mul(smoothstep(0.12, 0.3, mx)));
  m.metalnessNode = metal;
  m.roughnessNode = cane.mul(0.32).add(cork.mul(0.85)).add(walnut.mul(0.4)).add(wrap.mul(0.3)).add(metalBand.mul(0.42)).add(guide.mul(0.28))
    .add(reel.mul(mix(mix(float(0.5), float(0.45), metal), float(0.22), rim)));
  m.specularIntensityNode = mix(float(0.5), float(1.0), max(cane, wrap));
  // varnish over the cane and the wraps: a clear coat gives the one thin highlight line along the blank
  m.clearcoatNode = cane.mul(0.6).add(wrap.mul(0.8));
  m.clearcoatRoughness = 0.18;
  m.normalNode = normalMap(texture(src.normalMap!, t), vec2(0.5));
  // the upper blank tapers on to a hair (the GLB's top stays ~3 mm thick): drawn at its true radius down to half a
  // pixel, then a pixel wide fading by its coverage, so the tip thins out and vanishes against the background
  const fades = step(0.5, part).mul(step(part, 7.5)).mul(step(0.85, y));
  const trueR = radius.mul(float(1).sub(smoothstep(1.4, 2.53, y).mul(0.45)));
  const wp = modelWorldMatrix.mul(vec4(positionLocal, 1)).xyz;
  const pixW = length(wp.sub(vec3(anglerU.camMain))).mul(anglerU.pix);
  const widen = max(pixW.mul(0.5), trueR).sub(radius).mul(fades);
  // the guides' 1.4 mm wire likewise: a pixel wide and faint at a few metres, there to see close to the lens
  const wireR = 0.0007;
  const widenWire = max(pixW.mul(0.5).sub(wireR), 0).mul(guide);
  m.positionNode = positionLocal.add(normalLocal.mul(widen.add(widenWire)));
  const pixF = length(positionWorld.sub(vec3(anglerU.camMain))).mul(anglerU.pix);
  const coverage = saturate(trueR.mul(2).div(max(pixF, 1e-4)));
  const wireCover = saturate(float(wireR * 2).div(max(pixF, 1e-4)));
  m.opacityNode = mix(mix(float(1), coverage.max(0.03), fades), wireCover.pow(0.8).max(0.08), guide);
  m.transparent = true;
  m.depthWrite = true;
  // each segment is rigid: its axis from the rig, in view space, for the varnish's streak of sun along the cane
  const axis = normalize(cameraViewMatrix.mul(vec4(rodAxisU.element(int(min(part, 7).add(0.5))), 0)).xyz);
  m.lightOpts = {
    halo: float(0.7), haloTint: vec3(1.0, 0.9, 0.7), wrap: 0, brim: null, bounce: 0.6, occ: null,
    fiber: { axis, mask: cane.add(wrap).clamp(0, 1), thin: fades, k: 0.3, sigma: 0.1 },
  };
  m.receivedShadowPositionNode = sunMapLookup();
  return m;
}

/** Replace the GLB materials (matched by name) with the node materials above. */
export function upgradeMaterial(src: THREE.Material): THREE.Material {
  const s = src as THREE.MeshStandardMaterial;
  switch (src.name) {
    case 'Angler_Skin': return readsUnderWater(skin(s), 0.08);
    case 'Angler_Clothes': return readsUnderWater(clothes(s), 0.35);
    case 'Angler_Gear': return readsUnderWater(gear(s), 0.2);
    case 'FishingRod': return rod(s);
    default: return src;
  }
}
