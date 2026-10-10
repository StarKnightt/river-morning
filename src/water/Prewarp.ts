import * as THREE from 'three/webgpu';
import { float, length, mix, smoothstep, sqrt, step, uniform, vec3, type N } from '../gfx/tsl';
import { fieldNodes, type RiverField } from './RiverField';

/** height over the baked bed a pre-warped point is drawn at: this fraction of the depth, at least `liftMin` m */
export const prewarpU = {
  liftK: uniform(0.2),
  liftMin: uniform(0.04),
  liftMax: uniform(0.15),
  /**
   * A shallow disc (x, z, radius, depth; radius 0 = none): inside it the water refracts as if the bed were only
   * `depth` down (WaterMaterial.ts reads the same uniform). Set while a trout is held a few centimetres under the
   * surface: refracted by the bed's 35 cm, the trout and the hands under it wobbled with every ripple like the
   * stones and melted into them; by their own depth they keep a clear outline. The edge is soft (radius x 1.6).
   */
  disc: uniform(new THREE.Vector4(0, 0, 0, 0)),
};

/** depth the water assumes at surface point xz: the baked bed depth, or the shallow disc's inside it */
export function refractDepth(xz: N, bedDepth: N): N {
  const d = prewarpU.disc;
  const r = d.z.max(1e-4);
  const inside = float(1).sub(smoothstep(r, r.mul(1.6), length(xz.sub(d.xy)))).mul(step(1e-4, d.z));
  return mix(bedDepth, bedDepth.min(d.w), inside);
}

/**
 * Where a submerged point has to be drawn for the water to show it in its place.
 *
 * The water (WaterMaterial.ts) refracts the opaque image by the smooth baked *bed* depth, not by the depth behind
 * each pixel (that would show anything under the water twice at its outline), and it rejects samples that lie in
 * front of the water surface. Anything between the surface and the bed (a trout, a hand holding it) would be shown
 * displaced by the bed's offset, then rejected, so the stones it covers show in its place. So in the opaque pass
 * each submerged vertex X is moved to where the water will look for it: the surface entry point S of the real
 * refracted ray to X is solved (Newton on sin i = n sin t in the vertical plane through the eye), that ray is
 * extended to the baked bed depth at S (Q = S + (X - S) hBed / h, exactly the point the water samples for that
 * pixel), and the vertex is drawn on the eye ray through Q, a little in front of the bed. The water then shows it
 * where flat-water refraction puts it, at its real size, bent at the surface. The water also absorbs along
 * |X* - S| instead of |X - S|: `extra` is that difference (raise the colour by exp(sigma extra) to compensate).
 *
 * Off above the water (a smooth 8 mm onset under the surface), for a camera under it (the mirror's), where the
 * baked water at X is under 2 cm deep (dry ground below the river's level) and where `gate` is 0. Only for the main
 * view: materials that cast shadows must give the shadow pass the unwarped position (castShadowPositionNode).
 * If the water ever refracts by the depth behind each pixel instead, gate this off.
 */
export function refractionPrewarp(field: RiverField, X: N, gate: N, cameraPos: N) {
  const F = fieldNodes(field);
  const C = cameraPos;
  const n = 1.333;
  const AX = F.A(X.xz);
  const ys = AX.w;
  const h = ys.sub(X.y);
  const H = C.y.sub(ys);
  const hh = h.max(1e-3), HH = H.max(1e-3);
  const D = X.xz.sub(C.xz);
  const d = length(D).max(1e-4);
  const u = D.div(d);
  // horizontal distance a from the eye to the entry point: paraxial guess, then three Newton steps
  let a: N = d.mul(HH).mul(n).div(hh.add(HH.mul(n)));
  for (let k = 0; k < 3; k++) {
    const ra = sqrt(a.mul(a).add(HH.mul(HH)));
    const b = d.sub(a);
    const rb = sqrt(b.mul(b).add(hh.mul(hh)));
    const f = a.div(ra).sub(b.div(rb).mul(n));
    const fp = HH.mul(HH).div(ra.mul(ra).mul(ra)).add(hh.mul(hh).mul(n).div(rb.mul(rb).mul(rb)));
    a = a.sub(f.div(fp)).clamp(0, d);
  }
  const S = vec3(C.x.add(u.x.mul(a)), ys, C.z.add(u.y.mul(a)));
  const hBed = refractDepth(S.xz, F.A(S.xz).w.sub(F.B(S.xz).x)).max(0.05);
  const Q = S.add(X.sub(S).mul(hBed.div(hh)));
  const CQ = Q.sub(C);
  const L = length(CQ).max(1e-3);
  // a little over the bed (vertically: seen at a grazing angle a few cm along the ray is under a cm of height, and
  // the real terrain on a sloping margin stands that much above the 25 cm baked bed, hiding what is drawn there),
  // close enough that the depth step at the outline stays under a third of the depth (no meniscus line round it)
  // (every point on one refracted ray maps to the same Q: a little more lift for the shallower ones keeps their
  // order, so the hand under a trout stays under it and a trout's near flank in front of its far fin)
  const lift = hBed.mul(prewarpU.liftK).max(prewarpU.liftMin).min(prewarpU.liftMax).add(hBed.sub(h).max(0).mul(0.04));
  const drop = C.y.sub(Q.y).max(1e-3);
  const t = float(1).sub(lift.div(drop)).min(L.sub(0.03).div(L)).max(0);
  const Xs = C.add(CQ.mul(t));
  const wet = step(0.02, ys.sub(F.B(X.xz).x));
  const w = smoothstep(0.0, 0.008, h).mul(step(0.05, H)).mul(wet).mul(gate);
  const extra = length(Xs.sub(S)).sub(length(X.sub(S))).max(0).mul(w);
  return { position: mix(X, Xs, w), extra, weight: w };
}
