import type * as THREE from 'three/webgpu';
import {
  cos, float, Fn, If, length, max, min, mix, sin, smoothstep, texture, uniform, vec2, vec3, vec4, type N,
} from '../gfx/tsl';
import { riffleNodes, type RiffleField } from './RiffleField';
import type { Phase } from './surface';

/** Gains for A/B (?wu=rWaves:0 etc. reach them through RiverWater's uniform parser). */
export const riffleU = {
  /** standing waves and the water heaped round the rocks (baked slopes) */
  rWaves: uniform(1),
  /** white water: froth, breaking crests, streaks and seams */
  rFoam: uniform(1),
  /** bubbles in the water column */
  rAer: uniform(1),
  /** how much the glassy tongues damp the fine chop */
  rGlass: uniform(1),
  /** the small white ripples over the cobbles of the shallow fast water */
  rRip: uniform(1),
};

/** spacing (m) and height (m) of the small standing ripples over the cobbles */
const RIP_LAMBDA = 0.45;
const RIP_AMP = 0.008;

export interface RiffleTerms {
  /** 1 where these terms replace the generic fast-water ones (crest contours, coarse rock relief, foam trails) */
  cov: N;
  /** surface slope (world xz) to add to the water's normal */
  slope: N;
  /** the part of it that shifts the refracted bed */
  refr: N;
  /** 0..1 smooth, fast water: the fine chop is damped there */
  glass: N;
  /** 1 - how much the glassy tongues calm the generic turbulence and the fine chop */
  calm: N;
  /** gain of the material's standing lumps (the baked standing waves carry the relief here) */
  lumps: N;
  /** gain of the moving chop where the bake covers (the material damps it to 0.6 in fast water elsewhere) */
  tiles: N;
  /** natural foam source 0..0.85, to 0.97 in the plunge's core (max-combined with the material's own) */
  foam: N;
  /** 0..1 the plunge's heavy white water */
  heavy: N;
  /** 0..1 how much of that foam is round, bubbly froth rather than streaks */
  iso: N;
  /** froth pattern for it (two-phase, rides the current) */
  pattern: N;
  /** 0..0.9 how opaque the clumps of white water at the rocks and the plunge stay (the bubble cells thin them out) */
  solid: N;
  /** turbulence 0..1 (chop, sparkle spread and bed blur of broken water) */
  turb: N;
  /** bubbles in the water column 0..1 */
  aer: N;
  /** colour of the light the bubbles scatter back: white near the surface, pale turquoise from deeper down */
  tint: N;
}

/**
 * Shading terms of the riffle bake for the water material. `sd` is the ribbon's uv (arc length, lateral distance),
 * `pa`/`pb`/`norm` the material's two-phase flow map, `fp` its pixel footprint (m), `flow`, `fastW`, `depth` and
 * `bump` its current, fast-water weight, baked depth and bed humps. Without a bake (?riffle=0) every term is
 * neutral and the material is the generic one.
 */
export function riffleTerms(r: RiffleField | null, c: {
  sd: N; fp: N; pa: Phase; pb: Phase; norm: N; detail: THREE.Texture; flow: N; fastW: N; depth: N; bump: N;
}): RiffleTerms {
  if (!r) {
    const z = float(0), one = float(1);
    return {
      cov: z, slope: vec2(0, 0), refr: vec2(0, 0), glass: z, calm: one, lumps: one, tiles: one, foam: z, heavy: z, iso: z,
      pattern: float(0.5), solid: z, turb: z, aer: z, tint: vec3(1, 1, 1),
    };
  }
  const F = riffleNodes(r);
  const ta = F.A(c.sd).toVar(), tb = F.B(c.sd).toVar();
  const cov = tb.w.toVar();
  // the baked relief is 0.1 m texels (crests 0.75-1.9 m apart, ~0.2 m wide): resolved up close, then it fades to a
  // third (a pixel spanning a crest only aliases its flanks into random steep facets)
  const res = float(1).sub(smoothstep(0.04, 0.3, c.fp).mul(0.7)).toVar();
  const glass = ta.w.mul(riffleU.rGlass).toVar();
  // noise riding the current (the tile's height, both flow phases): P = the two phases' positions, W = their
  // weights and the blend's normalisation
  const D = c.detail;
  const advAt = (P: N, W: N, scale: number, off: number) => texture(D, P.xy.div(scale).add(off)).z.sub(0.5).mul(W.x)
    .add(texture(D, P.zw.div(scale).add(off + 0.5)).z.sub(0.5).mul(W.y)).mul(W.z);
  const phases = vec4(c.pa.pos, c.pb.pos), weights = vec4(c.pa.w, c.pb.w, c.norm, c.fp);

  // small white ripples over the cobbles: shallow fast water over a cobble bed is a field of short standing
  // wavelets, rows across the current with a crest dash behind every stone near the surface, breaking white in
  // flickers. Where from the field (fast, shallow, more of them over the bed's humps, none on the glassy tongues);
  // fixed to the bed in the river frame; the white on them comes and goes with the water.
  const rip = c.fastW.mul(smoothstep(0.45, 0.18, c.depth)).mul(glass.oneMinus()).mul(cov)
    .mul(smoothstep(0.0, 0.1, c.bump).mul(0.6).add(0.4)).mul(riffleU.rRip).toVar();

  // The costly part (16 of the 20 texture reads, the ripples' trigonometry) only runs where the bake has white
  // water, crests or ripples: the calm pool and the margins skip it. B = baked froth, streaks, crests, ripple
  // weight. Out: ripple slope along the current, foam source, froth pattern, broken water.
  const whiteWater = Fn(([sdIn, pIn, wIn, bIn]: [N, N, N, N]) => {
    const sd = sdIn.toVar(), P = pIn.toVar(), W = wIn.toVar(), B = bIn.toVar();
    const out = vec4(0, 0, 0.5, 0).toVar();
    If(max(max(B.x, B.y), max(B.z, B.w)).greaterThan(0.002), () => {
      const adv = (scale: number, off: number) => advAt(P, W, scale, off);
      const fp = W.w, rip = B.w;
      const jit = texture(D, sd.mul(vec2(1 / 2.7, 1 / 1.6)).add(0.31)).z
        .add(texture(D, sd.mul(vec2(1 / 1.15, 1 / 0.85)).add(0.67)).z.mul(0.5));
      // which stretches of a row carry a dash: two turned octaves (one axis-aligned tile ~1 m across repeated visibly)
      const r1 = vec2(sd.x.mul(0.88).sub(sd.y.mul(0.48)), sd.x.mul(0.48).add(sd.y.mul(0.88)));
      const r2 = vec2(sd.x.mul(0.62).add(sd.y.mul(0.78)), sd.y.mul(0.62).sub(sd.x.mul(0.78)));
      const segN = texture(D, r1.mul(vec2(1 / 1.9, 1 / 1.1)).add(0.13)).z
        .add(texture(D, r2.mul(vec2(1 / 0.83, 1 / 0.57)).add(0.71)).z.mul(0.6));
      const seg = smoothstep(0.64, 0.98, segN).toVar();
      const ang = sd.x.div(RIP_LAMBDA).add(jit.mul(3)).mul(2 * Math.PI).toVar();
      const cw = cos(ang).mul(0.5).add(0.5).toVar();
      const cw2 = cw.mul(cw), cw4 = cw2.mul(cw2);
      // their crests are a pixel or two wide from ~10 m on: past that only their mean whitening is left
      const rfade = float(1).sub(smoothstep(0.012, 0.045, fp)).toVar();
      // slope along the current: d/ds of A cw^4 (the dashes' slow phase wobble left out)
      const ripS = sin(ang).mul(cw2.mul(cw)).mul(-4 * Math.PI * RIP_AMP / RIP_LAMBDA).mul(rip).mul(seg.mul(0.6).add(0.4))
        .mul(float(1).sub(smoothstep(0.02, 0.08, fp)));
      const flick = smoothstep(-0.05, 0.22, adv(0.33, 0.71));
      // white tufts on the crest tops (a crest ^8 drew thin white scratches)
      const ripWhite = cw4.mul(cw).mul(seg).mul(flick).mul(rfade).mul(0.65).add(rfade.oneMinus().mul(0.08)).mul(rip);

      // white curls on the breaking crests: the crest line stays put, the curls on it come and go with the water
      const curl = smoothstep(-0.15, 0.25, adv(0.42, 0.19));
      const caps = B.z.mul(curl).mul(0.85);
      // white water comes in clumps that ride the current and break up (a smooth mask drew white discs round the
      // rocks), except in the heavy water of the plunge: one white body (patchy, it read as a pale rippled sheet).
      // The rocks' froth tops out at 0.75 in the bake, the plunge at 0.95
      const patch = smoothstep(-0.14, 0.16, adv(0.62, 0.13)).toVar();
      const heavy = smoothstep(0.78, 0.92, B.x);
      const froth = B.x.mul(mix(mix(0.4, 1.2, patch), 1.15, heavy)).add(B.x.mul(B.x).mul(0.4)).min(1).toVar();
      const streak = B.y.mul(smoothstep(0.35, 0.03, fp)).mul(mix(0.55, 1.1, patch)).toVar();
      // the froth's mask stays under the material's saturation, so its pattern keeps holes (at 0.85 most of the
      // pattern cleared the coverage ramp: a flat pale sheet behind every rock). The plunge's core is one churning white
      // body (water fix 2: held at 0.85 it read as a pale frosted oval, W3)
      const foam = max(max(froth.mul(mix(0.68, 0.95, heavy)), caps), max(streak.mul(0.8), ripWhite))
        .mul(riffleU.rFoam).min(mix(0.85, 0.97, heavy));
      // froth at the rocks and the plunge is round and bubbly: clumps of ~0.4 m, finer blobs, and cells of ~8 cm
      // whose dark middles read as the holes between bubbles (the material's own bubble cells are 2 cm, too fine past
      // ~2 m). Strong (sd ~0.5) and centred on 0.5 (cell net mean 0.73), so the coverage ramp cuts it into white
      // clumps and dark holes instead of a soft grey wash
      const cells = texture(D, P.xy.div(0.75).add(0.27)).w.mul(W.x).add(texture(D, P.zw.div(0.75).add(0.77)).w.mul(W.y));
      // (water fix 2) the clumps carry the froth's gaps, the bubble cells only nick it, least in the plunge's core: at
      // full strength their dark middles speckled the white water into a pale frosted sheet from the bank (W3, W1)
      const pattern = adv(0.45, 0.61).mul(2.6).add(adv(0.14, 0.83).mul(mix(1.5, 0.9, heavy)))
        .add(cells.sub(0.73).mul(mix(1.3, 0.6, heavy))).add(mix(0.5, 0.8, heavy));
      out.assign(vec4(ripS, foam, pattern, max(max(froth, caps.mul(0.8)), streak.mul(0.5))));
    });
    return out;
  });
  const ww = whiteWater(c.sd, phases, weights, vec4(tb.x, tb.z, ta.z, rip)).toVar();

  const fdir = c.flow.div(length(c.flow).max(0.05));
  const ripSlope = fdir.mul(ww.x).toVar();
  const slope = ta.xy.mul(riffleU.rWaves).mul(res).add(ripSlope).toVar();
  const heavy = smoothstep(0.78, 0.92, tb.x);
  // (the plunge body too: its seam streaks had switched it to the material's streaky lace, fine combed lines)
  const iso = max(smoothstep(0.15, 0.6, tb.x).mul(float(1).sub(smoothstep(0.3, 0.7, tb.z))), heavy);
  // broken water: the white water, and the shallow fast water over the cobbles; the glassy tongues stay smooth
  const calm = float(1).sub(glass.mul(0.8)).toVar();
  const turb = max(ww.w, rip.mul(0.7)).mul(calm).toVar();
  // bubbles under the surface well up in patches (boils) that drift down the pool
  const boil = smoothstep(-0.18, 0.22, advAt(phases, weights, 1.4, 0.47));
  const aer = tb.y.mul(mix(0.5, 1.0, boil)).mul(riffleU.rAer).toVar();
  // bubbles deeper in the plume (aerated, but no longer white on top) scatter light that crossed more water
  const deep = aer.mul(float(1).sub(smoothstep(0.1, 0.6, ww.w)));
  // (above 1: the plume glows a pale turquoise; at the material's bubble brightness it did not show)
  const tint = mix(vec3(1, 1, 1), vec3(0.56, 1.08, 1.02), min(deep.mul(1.3), 0.9));
  // (the small rows of ripples at a third: a periodic shift of the bed drew moire stripes in it)
  const refr = ta.xy.mul(riffleU.rWaves).mul(res).mul(0.4).add(ripSlope.mul(0.3));
  const solid = smoothstep(0.4, 0.85, tb.x).mul(mix(0.4, 0.9, heavy));
  const lumps = mix(float(1), float(0.7), cov).mul(float(1).sub(glass.mul(0.9)));
  // fast water over cobbles is rough in every direction (rows across the current hardly show from the bank, a view
  // across the river): full chop, more over the shallow broken water, calmer under heavy froth. The pillows, tongues
  // and V-slicks are smooth (water fix 2: at half chop the pillow against a rock read as the same rippled sheet)
  const tiles = rip.mul(0.3).add(1).mul(float(1).sub(glass.mul(0.85))).mul(float(1).sub(smoothstep(0.6, 0.95, tb.x).mul(0.4)));
  return { cov, slope, refr, glass, calm, lumps, tiles, foam: ww.y, heavy, iso, pattern: ww.z, solid, turb, aer, tint };
}
