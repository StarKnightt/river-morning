import * as THREE from 'three/webgpu';
import {
  Fn, If, abs, cameraPosition, dot, exp, float, length, log2, mix, normalWorldGeometry,
  positionLocal, positionWorld, smoothstep, texture, uniform, vec2, vec3, type N,
} from '../gfx/tsl';
import { ringHessian, surfaceU } from './surface';
import { fieldNodes, type RiverField } from './RiverField';
import { CAUSTIC_RANGE, CAUSTIC_TILE, causticTexture } from './CausticTile';
import { bodyShadow } from './BodyShadows';
import { params } from '../core/params';

/**
 * Light under the water for every lit receiver (bed, rocks; later fish and dock posts):
 *  - caustics: a baked net (CausticTile.ts) projected along the refracted sunbeam from the surface entry point,
 *    stretched along the sun's azimuth by the beam's slant, drifting with the current and slowly warped so it
 *    morphs; blurred with depth by the sun disc (mip level) and fading out by ~3 m. Fast water uses a coarser
 *    copy. Splash rings add their own analytic caustic ring (1/|det J| of the swell);
 *  - Fresnel loss and per-channel absorption along the refracted sunbeam;
 *  - shade under foam (foam blocks ~80 % of the light);
 *  - the crisp shadows of the trout and of forearms held in the water (BodyShadows.ts);
 *  - shadow-map lookups (and casters) moved along the refracted beam, so the bank's shadow lands where
 *    the bent sunlight puts it (at a 20° sun the air direction misplaces it by ~1.8 m per metre of depth);
 *  - sky light fading with depth, per channel (red first).
 * The sun part goes through material.receivedShadowNode, so it only touches the sun's direct term and composes
 * with the regular shadow map.
 */
export const uw = {
  /** toward the sun (or the moon at night) in air */
  sunAir: uniform(new THREE.Vector3(0, 1, 0)),
  /** the same light after refraction through flat water (pointing down) */
  sunRefr: uniform(new THREE.Vector3(0, -1, 0)),
  /** d(r.xz / -r.y) / d(slope) at slope 0, as (Qxx, Qxz, Qzx, Qzz) */
  sunQ: uniform(new THREE.Vector4(-0.25, 0, 0, -0.25)),
  /** flat-water Fresnel transmission of the beam */
  sunTF: uniform(0.86),
  /** horizontal direction of the sun (unit xz) and the stretch of the caustic net along it (1 / cos of the beam) */
  sunAz: uniform(new THREE.Vector2(1, 0)),
  sunStretch: uniform(1.4),
  /**
   * extinction σa + σs per metre, clear mountain river water: red goes first (water itself), green least, a
   * little grey loss from fine silt. Over the ~0.5-1 m light path of the shallows the stones keep their colours;
   * over the ~7 m down-and-up path of the 2.5-3 m pool the bed sinks into a dark emerald instead of the olive
   * khaki a grey extinction gave a sunlit silt bed seen from above.
   */
  sigmaT: uniform(new THREE.Vector3(0.5, 0.25, 0.29)),
  /** diffuse attenuation of sky light with depth, per channel (1/m) */
  kd: uniform(new THREE.Vector3(0.4, 0.13, 0.17)),
  /**
   * 0 = flat-water lighting only (debug and A/B). (water fix 2: 0.85 for the hand-sized net. At 1.8 the darkest
   * 70 % of the bed took no sun at all and the light came only in sparse lines up to 12x: a scatter of white glints on
   * stone tops rather than a web, C2, W3)
   */
  causticGain: uniform(0.85),
  /** 0 = shadow lookups along the air direction (debug and A/B) */
  warp: uniform(1),
  /**
   * angle of one screen pixel (rad), set every frame from the camera and the drawing buffer: screenSize does
   * not resolve to the viewport inside the lighting (shadow hook) context
   */
  pixAngle: uniform(8.6e-4),
};

const ETA = 1 / 1.333;

/** Recompute the refracted beam after the sky (time of day) changes. L points toward the light. */
export function setUnderwaterSun(L: THREE.Vector3) {
  const refr = (sx: number, sz: number) => {
    const n = new THREE.Vector3(-sx, 1, -sz).normalize();
    const I = L.clone().normalize().negate();
    const ndi = n.dot(I);
    const k = 1 - ETA * ETA * (1 - ndi * ndi);
    // GLSL refract(I, N, eta) = eta I - (eta (N·I) + sqrt(k)) N
    return I.multiplyScalar(ETA).sub(n.multiplyScalar(ETA * ndi + Math.sqrt(Math.max(k, 0))));
  };
  const q = (sx: number, sz: number) => {
    const r = refr(sx, sz);
    return [r.x / -r.y, r.z / -r.y];
  };
  const e = 1e-3;
  const ax = q(-e, 0), bx = q(e, 0), az = q(0, -e), bz = q(0, e);
  uw.sunQ.value.set((bx[0] - ax[0]) / (2 * e), (bz[0] - az[0]) / (2 * e), (bx[1] - ax[1]) / (2 * e), (bz[1] - az[1]) / (2 * e));
  const r0 = refr(0, 0);
  uw.sunRefr.value.copy(r0);
  uw.sunAir.value.copy(L).normalize();
  const c = Math.max(0, uw.sunAir.value.y);
  uw.sunTF.value = 1 - (0.0204 + 0.9796 * (1 - c) ** 5);
  const h = Math.hypot(L.x, L.z);
  uw.sunAz.value.set(h > 1e-4 ? L.x / h : 1, h > 1e-4 ? L.z / h : 0);
  uw.sunStretch.value = 1 / Math.max(0.3, -r0.y);
}

/**
 * Pixel footprint on the receiver (m) without screen-space derivatives, so it can be used inside branches:
 * distance x pixel angle, widened at grazing angles by the foreshortening to the power 0.8 (close to the long
 * axis: with the geometric mean of the two axes a net seen nearly edge-on aliased into rows of parallel dashes
 * along the C2 waterline, where the bank drops away from the eye). w: cosine of the view angle on the receiver.
 */
export function pixelFootprint(): N {
  const toCam = cameraPosition.sub(positionWorld);
  const d = toCam.length();
  const pix = uw.pixAngle;
  const cosI = abs(dot(normalWorldGeometry, toCam.div(d.max(1e-4)))).max(0.05);
  return vec2(d.mul(pix).div(cosI.pow(0.8)), cosI);
}

const TEXEL = CAUSTIC_TILE / 512;
const FAST_SCALE = 1.25;
/** scale of the fine net of the shallows, relative to the tile (cells ~7-13 cm, a hand across) */
const FINE = 0.5;

/**
 * Direct light at an underwater point relative to the light above the water (vec3). 1 above the surface.
 */
export function underwaterSun(field: RiverField) {
  const F = fieldNodes(field);
  const tex = causticTexture();
  return Fn(([P, surfaceY, foot]: [N, N, N]) => {
    const footprint = foot.x;
    const K = vec3(1, 1, 1).toVar();
    const h = surfaceY.sub(P.y).toVar();
    If(h.greaterThan(0.002), () => {
      const r = uw.sunRefr;
      const t = h.div(r.y.negate().max(0.08));
      const S = P.xz.sub(r.xz.mul(t));
      const f = F.A(S);
      const fast = smoothstep(0.6, 1.5, length(f.xy));
      // sun-aligned frame: the beam's slant stretches the net along the sun's azimuth
      const az = uw.sunAz;
      const q = vec2(dot(S, az).div(uw.sunStretch), dot(S, vec2(az.y.negate(), az.x)));
      // a slow warp makes the cells morph rather than slide; the drift follows the river's general course (+x)
      // as one uniform motion, so it never shears where the current changes
      const time = surfaceU.time;
      const wv = texture(tex, q.div(2.3).add(vec2(time.mul(0.021), time.mul(-0.013)))).level(2);
      const wq = vec2(wv.y, wv.z).sub(0.5);
      // the sun disc (0.53°) blurs the net with depth; the pixel footprint removes what the screen can't resolve
      // (0.7 of it: the lines stay crisp a little longer). A film of a few centimetres cannot focus the ripples into
      // the fine net: soft dapples there (the fine net drew parallel dashed streaks along the waterline)
      const blur = t.mul(0.0093).max(footprint.mul(0.7)).max(smoothstep(0.16, 0.02, h).mul(0.03));
      const cA = texture(tex, q.sub(vec2(time.mul(0.11), 0)).div(CAUSTIC_TILE).add(wq.mul(0.07)))
        .level(log2(blur.div(TEXEL)).max(0)).x;
      const cB = texture(tex, q.sub(vec2(time.mul(0.9), 0)).div(CAUSTIC_TILE * FAST_SCALE).add(wq.mul(0.1)).add(0.37))
        .level(log2(blur.div(TEXEL * FAST_SCALE)).max(0)).x;
      let net: N = mix(cA, cB, fast).mul(CAUSTIC_RANGE);
      // no room to focus in a film of water; scattering washes the net out with depth; a bed seen nearly edge-on
      // compresses the net below what the pixels can draw
      let contrast: N = smoothstep(3.2, 0.35, h).mul(smoothstep(0.02, 0.16, h)).mul(smoothstep(0.06, 0.2, foot.y))
        .mul(uw.causticGain);
      if (params.pool) {
        // The short capillary ripples focus first: over the shallows the net is finer (a rotated copy of the tile at
        // half its size, drifting with the breeze) and already shows in a film of a centimetre or two, soft while it is
        // still out of focus. Deeper the longer ripples' coarser net takes over. (The film's soft blur of the coarse net
        // alone left the sunlit margins evenly lit, C1, C2.) (water fix 2: the fine net over the first ~0.3 m, from
        // 2 mm deep, at full contrast from ~3 cm: ramped from 8 mm to 10 cm the C2 shelf showed none of it and read as
        // dry gravel up to the soft edge of the net)
        const qF = vec2(q.x.mul(0.83).sub(q.y.mul(0.56)), q.x.mul(0.56).add(q.y.mul(0.83)));
        const blurF = t.mul(0.0093).max(footprint.mul(0.7)).max(float(1).sub(smoothstep(0.012, 0.1, h)).mul(0.005));
        const cF = texture(tex, qF.sub(vec2(time.mul(0.045), time.mul(0.03))).div(CAUSTIC_TILE * FINE).add(wq.mul(0.12)).add(0.53))
          .level(log2(blurF.div(TEXEL * FINE)).max(0)).x;
        net = mix(net, cF.mul(CAUSTIC_RANGE), float(1).sub(smoothstep(0.15, 0.45, h)).mul(fast.oneMinus()));
        contrast = float(1).sub(smoothstep(0.35, 3.2, h)).mul(smoothstep(0.002, 0.03, h)).mul(smoothstep(0.06, 0.2, foot.y))
          .mul(uw.causticGain);
      }
      // a splash swell focuses a ring of light under its crest (zero rings alive: H = 0, factor 1)
      const H = ringHessian(S);
      const Q = uw.sunQ;
      const j00 = h.mul(Q.x.mul(H.x).add(Q.y.mul(H.y))).add(1);
      const j01 = h.mul(Q.x.mul(H.y).add(Q.y.mul(H.z)));
      const j10 = h.mul(Q.z.mul(H.x).add(Q.w.mul(H.y)));
      const j11 = h.mul(Q.z.mul(H.y).add(Q.w.mul(H.z))).add(1);
      const inv = float(1).div(abs(j00.mul(j11).sub(j01.mul(j10))).max(0.2));
      const ringF = inv.div(inv.sub(1).max(0).mul(0.45).add(1));
      // (the gain above 1 pushes the darkest cells below zero, which took light from the sky term too)
      const focus = (params.pool ? mix(float(1), net, contrast).max(0.03) : mix(float(1), net, contrast)).mul(mix(float(1), ringF, uw.causticGain));
      const lit = focus.mul(uw.sunTF).mul(float(1).sub(f.z.mul(0.8))).mul(bodyShadow(P, r.negate()));
      const T = exp(uw.sigmaT.mul(t).negate());
      K.assign(mix(vec3(1, 1, 1), T.mul(lit), smoothstep(0.0, 0.03, h)));
    });
    return K;
  });
}

/**
 * Position used for shadow lookups and shadow casting under water: X + (-L - r) t, t = h / |r.y|.
 * Every point on one refracted beam maps onto the air ray through the beam's entry point, deeper points
 * farther from the light, so casters and receivers stay consistent. Identity above the water.
 */
export const shadowWarp = (P: N, surfaceY: N) => {
  const t = surfaceY.sub(P.y).max(0).mul(uw.warp).div(uw.sunRefr.y.negate().max(0.08));
  return P.add(uw.sunAir.negate().sub(uw.sunRefr).mul(t));
};

type ShadowHook = (shadow: N) => N;

/**
 * Hook a lit node material into the water. Call it last: it wraps whatever receivedShadowNode the material
 * already has (the terrain's stone self-shadowing) instead of replacing it, and scales its colorNode.
 * `surfaceY` is the water level above the shaded point: the terrain passes its aWater.x attribute; rocks pass
 * fieldNodes(field).A(positionWorld.xz).w. Meshes must have an identity model transform (positionLocal = world).
 * Sky light fades with depth per channel: the albedo carries that attenuation and the sun term divides it out
 * again, since the sunbeam has its own (absorption along the refracted beam, caustics).
 */
export function installUnderwater(m: THREE.MeshStandardNodeMaterial, o: { field: RiverField; surfaceY: N; castSurfaceY?: N }) {
  const sun = underwaterSun(o.field);
  const footprint = pixelFootprint();
  const mat = m as unknown as { receivedShadowNode: ShadowHook | null; colorNode: N | null };
  const depth = o.surfaceY.sub(positionWorld.y).max(0);
  const skyT = exp(uw.kd.mul(depth).negate()).toVar();
  const prev = mat.receivedShadowNode;
  mat.receivedShadowNode = (shadow) => (prev ? prev(shadow) : shadow).mul(sun(positionWorld, o.surfaceY, footprint)).div(skyT);
  m.receivedShadowPositionNode = shadowWarp(positionWorld, o.surfaceY);
  m.castShadowPositionNode = shadowWarp(positionLocal, o.castSurfaceY ?? o.surfaceY);
  if (mat.colorNode) mat.colorNode = mat.colorNode.mul(skyT);
}

/** Debug: the caustic factor itself as an unlit colour (?dbg=caustic), 1 shown as mid grey. */
export function underwaterDebugColor(field: RiverField, surfaceY: N): N {
  return underwaterSun(field)(positionWorld, surfaceY, pixelFootprint()).div(2);
}
