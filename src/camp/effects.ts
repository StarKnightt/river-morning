import * as THREE from 'three/webgpu';
import {
  abs, attribute, cameraFar, cameraNear, cameraPosition, clamp, cos, cross, dot, exp, float, fract, hash, length, max, mix,
  normalize, perspectiveDepthToViewZ, positionGeometry, positionView, positionWorld, pow, screenUV, sin, smoothstep, step,
  uniform, varying, vec2, vec3, viewportDepthTexture, type N,
} from '../gfx/tsl';
import { LAYER_NO_REFLECT } from '../core/layers';
import type { SkyLighting } from '../world/SkyLighting';
import { tnoise } from '../gfx/noiseTex';
import { campU } from './uniforms';

/** shared baked noise (one feature per unit) stretched from its ~0.3-0.7 range to ~0-1 */
const wide = (n: N) => clamp(n.sub(0.5).mul(1.9).add(0.5), 0, 1);

/**
 * Chimney smoke, the campfire's smoke thread, coffee steam and the small flames, all computed on the GPU from the
 * engine time (no CPU simulation, nothing allocated per frame, deterministic in photo mode). Soft particles: each
 * fades out where it meets the opaque scene, read from one float depth copy shared by all effects (the scene pass
 * uses depth32float under reversed depth; three's shared 24-bit copy fails silently, see DECISIONS). It is a copy of
 * the depth buffer, never a scene pass rendered from inside a material. Main view only (LAYER_NO_REFLECT): not
 * mirrored, not in the AO pre-pass, no shadows.
 */

let depthNode: N = null;
/** eye distance of the opaque scene behind this pixel (one float depth copy for all camp effects) */
function sceneEye(): N {
  if (!depthNode) {
    const depthCopy = new THREE.DepthTexture(1, 1);
    depthCopy.type = THREE.FloatType;
    depthNode = viewportDepthTexture(screenUV, null, depthCopy);
  }
  return perspectiveDepthToViewZ(depthNode.sample(screenUV), cameraNear, cameraFar).negate();
}

/** the sky's light on smoke: the low sun through a forward-scattering phase, and the dome */
function smokeLight(sky: SkyLighting) {
  const u = sky.u;
  const vd = normalize(positionWorld.sub(cameraPosition));
  const cosT = dot(vd, u.sunDir);
  const g = 0.6;
  const hg = float((1 - g * g) / (4 * Math.PI)).div(pow(float(1 + g * g).sub(cosT.mul(2 * g)), 1.5));
  const phase = hg.mul(0.85).add(0.3 / (4 * Math.PI));
  const sunE = vec3(u.sunColor).mul(u.sunIntensity);
  const skyAmb = vec3(u.horizon).mul(0.35).add(vec3(u.zenith).mul(0.65)).mul(0.7);
  return { phase, sunE, skyAmb };
}

export interface RibbonOptions {
  name: string;
  origin: THREE.Vector3;
  /** strands in the plume (overlapping ribbons, each wandering on its own) and segments along each */
  strands: number;
  segments: number;
  /** seconds of travel the plume shows (its length in time) */
  life: number;
  /** initial rise speed (m/s), its e-folding time (s) as the gas cools, and the slow rise after (m/s) */
  rise: number;
  riseTau: number;
  riseTail: number;
  /** fraction of the breeze (campU.wind) the plume feels, and how long the gas takes to pick it up (s) */
  wind: number;
  windTau: number;
  /** width at the mouth (m) and how fast it spreads (m/s) */
  width0: number;
  grow: number;
  /** opacity at the mouth */
  opacity: number;
  /** sideways wander of the whole plume (m/s of age) and how far the strands drift apart (m/s) */
  meander: number;
  spread: number;
  /** small-scale curl (m): steam and a fire's thread twist as they rise */
  curl: number;
  albedo: [number, number, number];
  /** seconds for the gas to reach full density after leaving the mouth */
  fadeIn: number;
  /** warm underlight from the fire below at night (0 for chimney and steam) */
  fireGlow?: number;
  /** quick sideways kinks (m at 1 s of age): a small fire's thread wavers and jerks as it rises */
  kink?: number;
  /** share of its life over which the thread breaks into wisps (0.55 = the old default) */
  breakup?: number;
  /** density pattern: features across the ribbon and per second of gas travel (1.1 / 1.25 = the old default) */
  pattern?: [number, number];
}

/**
 * A continuous plume: camera-facing ribbons whose centreline is the path of the gas that left the mouth `age`
 * seconds ago (rising fast while hot, then slowly; taking up the breeze; wandering as the source sways), so the
 * plume is one body, densest at the mouth, loosening, widening and fading as it bends downwind. The density
 * pattern moves with the gas (noise keyed to the time the gas left the mouth). Light: the low sun with a
 * forward-scattering phase and a brighter sunward edge, sky light from the dome (bluish grey where the sun does not
 * reach), self-shadowed where dense. Every parameter is a uniform, so all ribbons share one program. Since round 3
 * it draws the thin threads (the campfire's smoke, the coffee steam); the chimney's plume is made of puffs.
 */
export function createRibbon(o: RibbonOptions, sky: SkyLighting): THREE.Mesh {
  const S = o.strands, K = o.segments;
  const nv = S * (K + 1) * 2;
  const pos = new Float32Array(nv * 3), aT = new Float32Array(nv), aSide = new Float32Array(nv), aStrand = new Float32Array(nv);
  const idx = new Uint32Array(S * K * 6);
  let v = 0, q = 0;
  for (let s = 0; s < S; s++) {
    for (let j = 0; j <= K; j++) {
      for (const side of [-1, 1]) {
        pos.set([o.origin.x, o.origin.y, o.origin.z], v * 3);
        aT[v] = j / K;
        aSide[v] = side;
        aStrand[v] = s;
        v++;
      }
      if (j < K) {
        const a = (s * (K + 1) + j) * 2;
        idx.set([a, a + 1, a + 3, a, a + 3, a + 2], q);
        q += 6;
      }
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('aT', new THREE.BufferAttribute(aT, 1));
  geo.setAttribute('aSide', new THREE.BufferAttribute(aSide, 1));
  geo.setAttribute('aStrand', new THREE.BufferAttribute(aStrand, 1));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  const reach = Math.hypot(0.8 * o.wind * o.life, o.rise * o.riseTau + o.riseTail * o.life) + o.width0 + o.grow * o.life + (o.kink ?? 0) * 2;
  const centre = o.origin.clone().add(new THREE.Vector3(0.3 * o.wind * o.life, 0.5 * (o.rise * o.riseTau + o.riseTail * o.life), 0.15 * o.wind * o.life));
  geo.boundingSphere = new THREE.Sphere(centre, reach);
  geo.boundingBox = new THREE.Box3().setFromCenterAndSize(centre, new THREE.Vector3(2 * reach, 2 * reach, 2 * reach));

  const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide });
  m.name = `camp-${o.name}`;
  const pat = o.pattern ?? [1.1, 1.25];
  const P = {
    origin: uniform(o.origin.clone()), life: uniform(o.life), rise: uniform(o.rise), riseTau: uniform(o.riseTau),
    riseTail: uniform(o.riseTail), wind: uniform(o.wind), windTau: uniform(o.windTau), width0: uniform(o.width0), grow: uniform(o.grow),
    opacity: uniform(o.opacity), meander: uniform(o.meander), spread: uniform(o.spread), curl: uniform(o.curl), fadeIn: uniform(o.fadeIn),
    albedo: uniform(new THREE.Vector3(...o.albedo)), glow: uniform(o.fireGlow ?? 0), kink: uniform(o.kink ?? 0),
    breakup: uniform(o.breakup ?? 0.55), pattern: uniform(new THREE.Vector2(pat[0], pat[1])),
  };
  const t01 = attribute('aT', 'float');
  const side = attribute('aSide', 'float');
  const strand = attribute('aStrand', 'float');
  const r1 = hash(strand.mul(1.37).add(0.11)), r2 = hash(strand.mul(2.71).add(0.37)), r3 = hash(strand.mul(5.13).add(0.71));
  const W = campU.wind.mul(P.wind);
  const wl = max(length(vec2(W.x, W.z)), 1e-3);
  const perp = vec3(W.z.negate(), 0, W.x).div(wl);
  const time = campU.time;
  // more segments near the mouth, where the plume bends and is densest
  const ageOf = (u: N) => pow(u, 1.5).mul(P.life);
  /** where the gas that left the mouth `a` seconds ago is now */
  const C = (a: N) => {
    const te = time.sub(a);
    const rise = P.rise.mul(P.riseTau).mul(float(1).sub(exp(a.div(P.riseTau).negate()))).add(a.mul(P.riseTail));
    const drift = W.mul(a.sub(P.windTau.mul(float(1).sub(exp(a.div(P.windTau).negate())))));
    const wander = sin(te.mul(0.31).add(1.3)).add(sin(te.mul(0.77).add(4.1)).mul(0.5)).add(sin(te.mul(1.9).add(r2.mul(6.0))).mul(0.2));
    const lateral = perp.mul(wander.mul(P.meander).mul(a)).add(vec3(0, sin(te.mul(0.53).add(0.7)).mul(P.meander.mul(0.3)).mul(a), 0));
    // kinks: quick jerks of the source (draughts round the fire) carried up with the gas
    const jerk = sin(te.mul(2.3).add(r2.mul(6.0))).mul(0.6).add(sin(te.mul(4.7).add(1.3)).mul(0.4)).add(sin(te.mul(7.9).add(r3.mul(5.0))).mul(0.25));
    const kinkV = perp.mul(jerk).add(vec3(W.x, 0, W.z).div(wl).mul(sin(te.mul(3.3).add(0.4)).mul(0.5))).mul(P.kink.mul(pow(a.add(0.01), 0.7)));
    // the strands drift apart as the plume loosens, each swaying on its own
    const sOff = perp.mul(r1.sub(0.5).add(sin(te.mul(0.9).add(r3.mul(6.28))).mul(0.25))).add(vec3(0, r2.sub(0.5).mul(0.5), 0))
      .mul(P.spread.mul(a));
    // curl: a small helix that grows with age
    const ph = te.mul(2.3).add(r1.mul(6.28)).add(a.mul(1.7));
    const curl = perp.mul(cos(ph)).add(vec3(W.x, 0, W.z).div(wl).mul(sin(ph))).mul(P.curl.mul(pow(a.add(0.02), 0.5)));
    return P.origin.add(vec3(0, rise, 0)).add(drift).add(lateral).add(kinkV).add(sOff).add(curl);
  };
  const age = ageOf(t01);
  const c0 = C(age);
  const c1 = C(age.add(P.life.mul(0.015)));
  const T = normalize(c1.sub(c0).add(vec3(0, 1e-4, 0)));
  const viewDir = normalize(c0.sub(cameraPosition));
  const across = normalize(cross(T, viewDir).add(vec3(1e-5, 0, 0)));
  const width = P.width0.add(age.mul(P.grow)).mul(r3.mul(0.35).add(0.83));
  m.positionNode = c0.add(across.mul(side.mul(width).mul(0.5)));
  const a01 = age.div(P.life);
  const dilute = pow(P.width0.div(width), 0.75);
  const vAlpha = varying(P.opacity.mul(smoothstep(0, P.fadeIn, age)).mul(float(1).sub(smoothstep(0.45, 1.0, a01))).mul(dilute.mul(0.8).add(0.2)));
  const vSide = varying(side);
  const vTe = varying(time.sub(age));
  const vAge = varying(a01);
  const vWidth = varying(width);
  const vRise = varying(c0.y.sub(P.origin.y));
  const vSun = varying(dot(across, sky.u.sunDir));
  const vSeed = varying(r1.mul(17.0));

  // the pattern moves with the gas; its features keep their share of the width as the plume spreads; the edge
  // itself wanders, so the ribbon never shows a clean outline
  const nuv = vec2(vSide.mul(P.pattern.x).add(vSeed), vTe.mul(P.pattern.y));
  const nz = wide(tnoise(nuv, 6)).mul(0.6).add(wide(tnoise(nuv.mul(vec2(2.4, 2.9)).add(7.3), 7)).mul(0.4));
  const x = vSide.add(tnoise(vec2(vSeed.add(3.1), vTe.mul(2.3)), 5).sub(0.5).mul(0.7));
  const e = float(1).sub(x.mul(x)).max(0);
  const prof = e.mul(e);
  // dense and smooth at the mouth, broken into wisps further up
  const loosen = mix(float(0.12), float(0.66), smoothstep(0.0, P.breakup, vAge));
  const dens = clamp(nz.sub(loosen).mul(2.2).add(prof.mul(0.35)), 0, 1).mul(prof);
  const L = smokeLight(sky);
  // the sunward edge of the plume is lit through less smoke than the far edge
  const sunEdge = mix(float(0.55), float(1.3), clamp(vSun.mul(x).mul(0.5).add(0.5), 0, 1));
  const selfShadow = mix(float(1), float(0.45), dens.mul(0.55).add(float(1).sub(vAge).mul(0.3)).clamp(0, 1)).mul(sunEdge);
  const night = smoothstep(0.6, 1.0, campU.strength);
  // the flames light the smoke from below for its first hand-span or so, then it greys (part 8 round 2, critic r1: "grey
  // right down to the grill")
  const glow = vec3(1.0, 0.42, 0.14).mul(campU.flameGain.mul(P.glow).mul(night).mul(campU.flicker).mul(exp(vRise.mul(-2.6))));
  const col = P.albedo.mul(L.sunE.mul(L.phase).mul(selfShadow).mul(campU.smokeSun).add(L.skyAmb.mul(campU.smokeShade).mul(0.8))).add(glow);
  const fragEye = positionView.z.negate();
  const soft = smoothstep(0.0, vWidth.mul(0.5), sceneEye().sub(fragEye));
  const near = smoothstep(0.3, 1.5, fragEye);
  m.colorNode = col;
  m.opacityNode = dens.mul(vAlpha).mul(soft).mul(near);
  const mesh = new THREE.Mesh(geo, m);
  mesh.name = `camp-${o.name}`;
  mesh.layers.set(LAYER_NO_REFLECT);
  mesh.renderOrder = 2;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return mesh;
}

export interface PuffOptions {
  name: string;
  origin: THREE.Vector3;
  /** puffs alive at once (they are reborn at the mouth when they die, so the plume never thins out) */
  count: number;
  /** seconds a puff lives (the plume's length in time) */
  life: number;
  /** as RibbonOptions: rise, its decay, the slow rise after, the share of the breeze and how fast it is taken up */
  rise: number;
  riseTau: number;
  riseTail: number;
  wind: number;
  windTau: number;
  /** diameter at the mouth (m) and how fast a puff swells (m/s) */
  size0: number;
  grow: number;
  /** peak opacity of one puff */
  opacity: number;
  /** how far puffs stray from the plume's path as it breaks up (m at 1 s of age, growing as age^0.75) */
  turb: number;
  /** sideways wander of the whole plume (m/s of age) */
  meander: number;
  /** albedo at the mouth and out in the haze */
  albedo: [number, number, number];
  albedoFar: [number, number, number];
  /** seconds for a puff to reach full density after leaving the mouth */
  fadeIn: number;
  /** how far old puffs squash flat as the plume spreads into a haze layer (0 round .. 1 flat) */
  flatten?: number;
  /** lumpiness of a puff's edge (0.95 = billowy round 3 puffs; thin smoke is softer) */
  lumps?: number;
  /** thin smoke is lit through: 0 = dense puffs that shade their own far side .. 1 = no self-shadow */
  thin?: number;
  /** how fast a puff's opacity falls as it swells (exponent of size0 / size; 0.8 = round 3) */
  dilute?: number;
}

/**
 * A plume of soft, lumpy, self-lit puffs (round 2's chimney ribbon read as "a grey rope or a hank of combed hair,
 * an even-width band with fine parallel streaks, bending over in one smooth hook, faintest at the mouth"): each puff
 * leaves the mouth small, dense and white, rides the gas path (rise, breeze, wander), swells, turns, strays further
 * from the path as it ages, and fades into a widening, ragged haze. Many overlap at the mouth, so it is thickest and
 * whitest there. Each puff is lit as a lumpy ball: the side toward the low sun warm white (campU.smokeSun), the far
 * side the sky's cool blue-grey (campU.smokeShade). Camera-facing quads, one program, no CPU work per frame.
 */
export function createPuffs(o: PuffOptions, sky: SkyLighting): THREE.Mesh {
  const n = o.count;
  const pos = new Float32Array(n * 12), corner = new Float32Array(n * 8), id = new Float32Array(n * 4);
  const idx = new Uint32Array(n * 6);
  const cs = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 4; k++) {
      const v = i * 4 + k;
      pos.set([o.origin.x, o.origin.y, o.origin.z], v * 3);
      corner.set(cs[k], v * 2);
      id[v] = i;
    }
    idx.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3], i * 6);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('aCorner', new THREE.BufferAttribute(corner, 2));
  geo.setAttribute('aId', new THREE.BufferAttribute(id, 1));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  const reach = Math.hypot(0.8 * o.wind * o.life, o.rise * o.riseTau + o.riseTail * o.life) + o.size0 + o.grow * o.life + o.turb * Math.pow(o.life, 0.75);
  const centre = o.origin.clone().add(new THREE.Vector3(0.3 * o.wind * o.life, 0.5 * (o.rise * o.riseTau + o.riseTail * o.life), 0.15 * o.wind * o.life));
  geo.boundingSphere = new THREE.Sphere(centre, reach);
  geo.boundingBox = new THREE.Box3().setFromCenterAndSize(centre, new THREE.Vector3(2 * reach, 2 * reach, 2 * reach));

  const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
  m.name = `camp-${o.name}`;
  const P = {
    origin: uniform(o.origin.clone()), life: uniform(o.life), rise: uniform(o.rise), riseTau: uniform(o.riseTau),
    riseTail: uniform(o.riseTail), wind: uniform(o.wind), windTau: uniform(o.windTau), size0: uniform(o.size0), grow: uniform(o.grow),
    opacity: uniform(o.opacity), turb: uniform(o.turb), meander: uniform(o.meander), fadeIn: uniform(o.fadeIn),
    albedo: uniform(new THREE.Vector3(...o.albedo)), albedoFar: uniform(new THREE.Vector3(...o.albedoFar)), count: uniform(n),
    flatten: uniform(o.flatten ?? 0), lumps: uniform(o.lumps ?? 0.95), thin: uniform(o.thin ?? 0), dilute: uniform(o.dilute ?? 0.8),
  };
  const pid = attribute('aId', 'float');
  const cn = attribute('aCorner', 'vec2');
  const time = campU.time;
  // evenly spread over their lives; each rebirth draws new randoms
  const cyc = time.div(P.life).add(pid.mul(0.6180339));
  const age = cyc.fract().mul(P.life);
  const gen = cyc.floor();
  const sd = pid.mul(7.13).add(gen.mul(1.731));
  const r1 = hash(sd), r2 = hash(sd.add(0.37)), r3 = hash(sd.add(0.71)), r4 = hash(sd.add(1.13));
  const te = time.sub(age);
  const W = campU.wind.mul(P.wind);
  const wl = max(length(vec2(W.x, W.z)), 1e-3);
  const perp = vec3(W.z.negate(), 0, W.x).div(wl);
  const down = vec3(W.x, 0, W.z).div(wl);
  const rise = P.rise.mul(P.riseTau).mul(float(1).sub(exp(age.div(P.riseTau).negate()))).add(age.mul(P.riseTail));
  const drift = W.mul(age.sub(P.windTau.mul(float(1).sub(exp(age.div(P.windTau).negate())))));
  const wander = sin(te.mul(0.31).add(1.3)).add(sin(te.mul(0.77).add(4.1)).mul(0.5)).add(sin(te.mul(1.9).add(0.6)).mul(0.25));
  const lateral = perp.mul(wander.mul(P.meander).mul(age)).add(vec3(0, sin(te.mul(0.53).add(0.7)).mul(P.meander.mul(0.4)).mul(age), 0));
  // breaking up: puffs stray from the path, rolling over as the plume churns
  const stray = P.turb.mul(pow(age.add(0.05), 0.75));
  const roll = vec3(0, sin(te.mul(1.1).add(r4.mul(6.28))).mul(0.5), 0).add(down.mul(cos(te.mul(1.1).add(r4.mul(6.28))).mul(0.5)));
  const off = perp.mul(r1.sub(0.5).mul(2.0)).add(vec3(0, r2.sub(0.5).mul(1.4), 0)).add(down.mul(r3.sub(0.5).mul(1.2))).add(roll).mul(stray);
  const centreW = P.origin.add(vec3(0, rise, 0)).add(drift).add(lateral).add(off);
  const size = P.size0.add(age.mul(P.grow)).mul(r4.mul(0.55).add(0.72));
  const viewDir = normalize(centreW.sub(cameraPosition));
  const right = normalize(cross(viewDir, vec3(0, 1, 0)));
  const up = cross(right, viewDir);
  const rot = r1.mul(6.283).add(time.mul(r2.sub(0.5).mul(0.35)));
  const cr = cos(rot), sr = sin(rot);
  const qx = cn.x.mul(cr).sub(cn.y.mul(sr)), qy = cn.x.mul(sr).add(cn.y.mul(cr));
  const a01 = age.div(P.life);
  // spreading out into a flat layer as the plume levels off (scene fix: a calm morning's smoke flattens into a haze)
  const flat = float(1).sub(P.flatten.mul(smoothstep(0.12, 0.6, a01)));
  m.positionNode = centreW.add(right.mul(qx.mul(size).mul(0.5))).add(up.mul(qy.mul(size).mul(0.5).mul(flat)));
  // the gas thins as it spreads (round 4 wip: the overlapping old puffs piled up into a dense white cloud)
  const dilute = pow(P.size0.div(size), P.dilute);
  const vAlpha = varying(P.opacity.mul(smoothstep(0.0, P.fadeIn, age)).mul(float(1).sub(smoothstep(0.3, 1.0, a01))).mul(dilute.mul(0.84).add(0.16)));
  const vC = varying(cn);
  const vAge = varying(a01);
  const vSeed = varying(r3.mul(13.0).add(r1.mul(5.0)));
  const vSize = varying(size);
  // the sun's direction in the puff's own frame (turned with it), for its lumpy-ball lighting
  const su = dot(right, sky.u.sunDir), sv = dot(up, sky.u.sunDir);
  const vSunXY = varying(vec2(su.mul(cr).add(sv.mul(sr)), sv.mul(cr).sub(su.mul(sr))));
  const vSunZ = varying(dot(viewDir.negate(), sky.u.sunDir));

  const rr = length(vC);
  const q = vC.mul(1.15).add(vec2(vSeed, vSeed.mul(0.7)));
  const n1 = wide(tnoise(q, 6)), n2 = wide(tnoise(q.mul(2.6).add(4.1), 7));
  const lump = n1.mul(0.62).add(n2.mul(0.38));
  // a lumpy ball: billows round its edge, denser in the middle
  const shape = float(1).sub(smoothstep(0.2, 1.0, rr.add(lump.sub(0.5).mul(P.lumps))));
  const dens = shape.mul(shape.mul(0.6).add(0.4)).mul(vAlpha);
  const nzz = float(1).sub(rr.mul(rr)).max(0).sqrt();
  const sunDot = vC.x.mul(vSunXY.x).add(vC.y.mul(vSunXY.y)).add(nzz.mul(vSunZ));
  const lit = smoothstep(-0.4, 0.75, sunDot.add(lump.sub(0.5).mul(0.6)));
  const L = smokeLight(sky);
  const alb = mix(P.albedo, P.albedoFar, smoothstep(0.1, 0.7, vAge));
  // young, dense puffs shade their own far side; the old haze is lit through
  const shadeK = mix(mix(float(0.12), float(0.45), smoothstep(0.15, 0.8, vAge)), float(0.85), P.thin);
  const col = alb.mul(L.sunE.mul(L.phase).mul(campU.smokeSun).mul(mix(shadeK, float(1.25), lit))
    .add(L.skyAmb.mul(campU.smokeShade).mul(mix(float(0.95), float(0.75), lit))));
  const fragEye = positionView.z.negate();
  const soft = smoothstep(0.0, vSize.mul(0.35), sceneEye().sub(fragEye));
  const near = smoothstep(0.3, 1.5, fragEye);
  m.colorNode = col;
  m.opacityNode = dens.mul(soft).mul(near);
  const mesh = new THREE.Mesh(geo, m);
  mesh.name = `camp-${o.name}`;
  mesh.layers.set(LAYER_NO_REFLECT);
  mesh.renderOrder = 2;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return mesh;
}

/**
 * Flames standing up from the burning wood (fire.ts places them in the gaps between the logs, where the coals are
 * hottest, of every size): camera-facing cards (turned about the world up axis), each holding two or three tongues of
 * their own height, lean and sway, ragged at their edges, holed and torn by the turbulence rising through them, with
 * licks breaking off above the tips (scene fix, critic polish r1: round 4's "row of identical saturated orange
 * teardrops, like a gas burner"). Added to the scene (see-through like a real flame, so by day they read pale against
 * the sunlit ash and wood), pale yellow-white low in the core, orange toward the edges and tips, a faint blue foot.
 * Part 8 raises strength and gain at night.
 */
export function createFlames(points: Array<{ p: THREE.Vector3; size: number }>): THREE.Mesh {
  const n = points.length;
  const pos = new Float32Array(n * 12), corner = new Float32Array(n * 8), id = new Float32Array(n * 4), sz = new Float32Array(n * 4);
  const idx = new Uint16Array(n * 6);
  const cs = [[-1, 0], [1, 0], [1, 1], [-1, 1]];
  const box = new THREE.Box3();
  points.forEach((f, i) => {
    for (let k = 0; k < 4; k++) {
      const v = i * 4 + k;
      pos.set([f.p.x, f.p.y, f.p.z], v * 3);
      corner.set(cs[k], v * 2);
      id[v] = i;
      sz[v] = f.size;
    }
    idx.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3], i * 6);
    box.expandByPoint(f.p);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('aCorner', new THREE.BufferAttribute(corner, 2));
  geo.setAttribute('aId', new THREE.BufferAttribute(id, 1));
  geo.setAttribute('aSize', new THREE.BufferAttribute(sz, 1));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  box.expandByVector(new THREE.Vector3(0.35, 0.3, 0.35));
  box.max.y += 0.45;
  geo.boundingBox = box;
  geo.boundingSphere = box.getBoundingSphere(new THREE.Sphere());
  const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  m.name = 'camp-flames';
  m.fog = false;
  const fid = attribute('aId', 'float');
  const c = attribute('aCorner', 'vec2');
  const fsz = attribute('aSize', 'float');
  const r1 = hash(fid.mul(3.17).add(0.21)), r2 = hash(fid.mul(4.31).add(0.57)), r3 = hash(fid.mul(6.07).add(0.13)), r4 = hash(fid.mul(8.93).add(0.77));
  const t = campU.time;
  // (identity model matrix: the geometry's own position is the flame's foot in the world)
  const b = positionGeometry;
  const toCam = cameraPosition.sub(b);
  const hz = normalize(vec3(toCam.x, 0, toCam.z));
  const right = vec3(hz.z, 0, hz.x.negate());
  const s = mix(float(0.7), float(1.25), campU.strength).mul(campU.night.mul(0.4).add(1));
  // each flame flares and dies back on its own rhythm
  const life = wide(tnoise(vec2(t.mul(1.25).add(r4.mul(31.0)), r1.mul(13.0)), 8));
  const h = float(0.15).add(r2.mul(0.13)).mul(fsz).mul(s).mul(life.mul(0.55).add(0.65)).mul(campU.flicker.mul(0.25).add(0.78));
  const w = float(0.085).add(r3.mul(0.05)).mul(fsz).mul(s).mul(life.mul(0.2).add(0.9));
  const lean = campU.wind.mul(c.y.mul(c.y).mul(h).mul(0.18));
  m.positionNode = b.add(right.mul(c.x.mul(w))).add(vec3(0, c.y.mul(h), 0)).add(lean);
  const vC = varying(c);
  const vSeed = varying(r1.add(r3.mul(3.7)));
  const vLife = varying(life);
  const x = vC.x, y = vC.y;
  // turbulence rising through the flame
  const n1 = wide(tnoise(vec2(x.mul(1.2).add(vSeed.mul(5.0)), y.mul(2.1).sub(t.mul(3.5))), 8));
  const n2 = wide(tnoise(vec2(x.mul(3.0).add(vSeed.mul(7.0)), y.mul(4.8).sub(t.mul(7.2))), 9));
  const turb = n1.mul(0.6).add(n2.mul(0.4)).sub(0.5);
  // two or three tongues per card (the side ones on most cards), each its own height, offset and sway
  let body: N = float(0);
  let core: N = float(0);
  for (const [k, ox, h0] of [[0, -0.45, 0.6], [1, 0.0, 1.0], [2, 0.42, 0.72]] as const) {
    const rk = fract(vSeed.mul(2.3 + k * 1.37).add(k * 0.31));
    const present = k === 1 ? float(1) : step(0.3, rk);
    const hk = float(h0).mul(rk.mul(0.5).add(0.62)).min(1);
    const yk = y.div(hk);
    const ph = vSeed.mul(3.1 + k * 1.7).add(k * 2.1);
    const sway = sin(t.mul(5.3 + k * 1.1).add(ph)).mul(0.17).add(sin(t.mul(9.1 + k * 1.3).add(ph.mul(1.7))).mul(0.08))
      .mul(yk.mul(yk)).add(turb.mul(0.5).mul(yk));
    const hw = float(0.36).mul(clamp(yk.add(0.06).div(0.28), 0, 1).sqrt()).mul(pow(float(1).sub(yk).max(0), 0.9)).add(0.012);
    const dk = abs(x.sub(ox).sub(sway)).div(hw);
    const bk = float(1).sub(smoothstep(0.4, 1.0, dk.add(turb.mul(0.7)))).mul(float(1).sub(smoothstep(0.55, 1.0, yk.add(turb.mul(0.4)))));
    body = max(body, bk.mul(present));
    core = max(core, clamp(float(1).sub(dk), 0, 1).mul(float(1).sub(yk.mul(0.75)).max(0)).mul(present));
  }
  // holes and torn edges where the turbulence thins the gas, more toward the tips; licks breaking off above them
  const holes = smoothstep(0.32, 0.5, wide(tnoise(vec2(x.mul(2.6).add(vSeed.mul(3.0)), y.mul(3.6).sub(t.mul(6.4))), 11)));
  body = body.mul(mix(float(1), holes, smoothstep(0.2, 0.75, y)));
  const lick = smoothstep(0.62, 0.78, wide(tnoise(vec2(x.mul(2.2).add(vSeed.mul(9.0)), y.mul(2.6).sub(t.mul(7.8))), 10)))
    .mul(smoothstep(0.45, 0.7, y)).mul(float(1).sub(smoothstep(0.85, 1.0, y))).mul(float(1).sub(smoothstep(0.35, 0.75, abs(x))));
  body = max(body, lick.mul(0.7));
  // temperature: hottest low in the cores
  const T = core.mul(body.mul(0.5).add(0.5));
  // pale by day: yellow-white core, soft orange edges, a little deeper orange at the tips
  const col = mix(
    mix(vec3(0.95, 0.3, 0.05), vec3(1.0, 0.5, 0.14), smoothstep(0.05, 0.3, T)),
    mix(vec3(1.0, 0.66, 0.28), vec3(1.0, 0.82, 0.52), smoothstep(0.6, 0.9, T)),
    smoothstep(0.3, 0.6, T),
  );
  const I = campU.flameGain.mul(float(0.4).add(pow(T, 1.3).mul(1.45))).mul(campU.flicker).mul(mix(float(0.6), float(1.0), campU.strength)).mul(vLife.mul(0.35).add(0.75));
  // a faint blue foot where the gas first burns
  const foot = float(1).sub(smoothstep(0.02, 0.14, y)).mul(core.min(1)).mul(0.5);
  const soft = smoothstep(0.0, 0.035, sceneEye().sub(positionView.z.negate()));
  // at night deeper orange: the overlapping tongues summed to a cream-white glow (part 8 round 2, critic r1)
  const nightCol = mix(vec3(1), vec3(1.0, 0.72, 0.42), campU.night);
  m.colorNode = col.mul(nightCol).mul(I).add(vec3(0.12, 0.2, 0.55).mul(foot).mul(campU.flameGain).mul(0.35));
  m.opacityNode = body.mul(mix(float(0.36), float(0.7), smoothstep(0.1, 0.6, T))).add(foot.mul(0.12)).clamp(0, 1).mul(soft);
  const mesh = new THREE.Mesh(geo, m);
  mesh.name = 'camp-flames';
  mesh.layers.set(LAYER_NO_REFLECT);
  mesh.renderOrder = 3;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return mesh;
}
