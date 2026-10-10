import { attribute, clamp, float, max, mix, smoothstep, vec2, vec3, type N } from '../gfx/tsl';
import { tnoise } from '../gfx/noiseTex';

export { campGround, nearCamp, type CampWear } from './groundMask';

/** A layer sample of the core terrain shader (structural; TerrainMaterial's LayerSample). */
export interface GroundSample { albedo: N; height: N; pert: N; rough: N; ao: N }

/**
 * The camp's worn ground (groundMask.ts, baked per vertex as `aCampG` = worn (+ the polish of a path's beaten middle
 * above 1), ash, chips) as one more layer of the core terrain's height-aware blend (TerrainMaterial), from taps it
 * already makes. Round 3 (critic r2: "laid gravel, not trodden earth ... a neat oval of pale, cold blue-grey grit
 * with a crisp edge"): warm tan to dark brown packed dirt, scuffed and dusty, the soil scan's stones kept small and
 * faint; darkest near the fire where ash and charcoal are kicked out over it; chips and bark by the wood; needles
 * and twigs everywhere; the beaten middle of a path paler and smoother; and at the edge flattened, yellowed grass
 * thinning out over the dirt instead of a line. `w` is its weight on the open ground, `keep` what is left for the
 * grass and the forest floor.
 */
export function campGroundLayer(L: { mud: GroundSample; silt: GroundSample; sand: GroundSample }, xz: N) {
  const a = attribute('aCampG', 'vec4');
  const worn = a.x.min(1), polish = a.x.sub(1).clamp(0, 1), ash = a.y, chips = a.z;
  const n1 = tnoise(xz.mul(1.9), 31.1), n2 = tnoise(xz.mul(6.3), 32.2), n3 = tnoise(xz.mul(17.0), 37.7), nL = tnoise(xz.mul(0.55), 36.6);
  const lumOf = (c: N) => c.x.mul(0.2126).add(c.y.mul(0.7152)).add(c.z.mul(0.0722));
  // packed earth: dusty tan where it dried, darker brown where it is trodden hard and damp; the soil scan's
  // pebbles and grit kept, small and faint
  const scan = clamp(lumOf(L.mud.albedo).div(0.12), 0.6, 1.5);
  let alb: N = mix(vec3(0.07, 0.049, 0.032), vec3(0.168, 0.122, 0.079), smoothstep(0.3, 0.72, nL.mul(0.65).add(n1.mul(0.35))));
  alb = alb.mul(mix(float(1), scan, 0.4));
  // (scene fix, critic polish r1: "smooth plaster with brush-stroke scratches"): gravelly, the small-rocks scan's
  // pebbles showing through the packed dirt in patches, trodden in
  const grav = smoothstep(0.38, 0.62, n2.mul(0.6).add(nL.mul(0.4))).mul(0.55).add(0.12).toVar();
  alb = mix(alb, mix(vec3(lumOf(L.silt.albedo)), L.silt.albedo, 0.45).mul(0.8), grav.mul(float(1).sub(polish.mul(0.6))));
  // scuffs: drag marks darker, dust paler
  alb = alb.mul(mix(0.86, 1.12, n3));
  // the beaten middle of a path: paler, smoother, dry dust over it (round 4 wip: from P4b still a dark strip in the
  // grass)
  const dust = vec3(0.2, 0.163, 0.118).mul(mix(0.86, 1.1, n3)).mul(mix(0.92, 1.06, n1));
  alb = mix(alb, mix(alb.mul(vec3(1.45, 1.36, 1.22)), dust, 0.6), polish.mul(0.9));
  // ash and charcoal kicked out of the ring: dark sooty dirt, grey ash in flecks and drifts, black grit
  const sooty = smoothstep(0.12, 0.8, ash.add(n1.sub(0.5).mul(0.5)).add(n2.sub(0.5).mul(0.3)));
  alb = mix(alb, vec3(0.034, 0.03, 0.027).mul(mix(0.8, 1.2, n3)), sooty.mul(0.85));
  const fleck = smoothstep(0.66, 0.74, tnoise(xz.mul(26.0), 39.2)).mul(smoothstep(0.08, 0.55, ash));
  alb = mix(alb, vec3(0.17, 0.166, 0.16).mul(mix(0.75, 1.1, n2)), fleck.mul(0.4));
  const grit = smoothstep(0.66, 0.71, tnoise(xz.mul(37.0), 33.3)).mul(ash.mul(0.9).add(chips.mul(0.25)));
  alb = mix(alb, vec3(0.015, 0.014, 0.013), grit);
  // chips (pale, angular) and bark flakes (dark), oriented every which way
  const cuv = vec2(xz.x.add(xz.y.mul(0.6)), xz.y.sub(xz.x.mul(0.4)));
  const chip = smoothstep(0.66, 0.7, tnoise(cuv.mul(13.0), 34.4)).mul(chips);
  const bark = smoothstep(0.67, 0.71, tnoise(cuv.mul(vec2(9.0, 19.0)), 35.5)).mul(chips);
  alb = mix(alb, vec3(0.27, 0.21, 0.14).mul(mix(0.8, 1.15, n2)), chip.mul(0.9));
  alb = mix(alb, vec3(0.045, 0.031, 0.019), bark.mul(0.8));
  // needles and twigs blown in: thin dark strokes in two directions, swept off the beaten paths
  const nd = max(smoothstep(0.72, 0.745, tnoise(vec2(xz.x.mul(48.0), xz.y.mul(5.0)), 38.1)), smoothstep(0.72, 0.745, tnoise(vec2(xz.x.add(xz.y).mul(4.0), xz.x.sub(xz.y).mul(40.0)), 38.9)));
  alb = mix(alb, vec3(0.05, 0.035, 0.021), nd.mul(0.3).mul(smoothstep(0.45, 0.7, n2)).mul(float(1).sub(polish.mul(0.7))));
  // the edge: toward its outer part the camp layer is trampled grass and litter (flattened, yellowed, green left in
  // the tufts) close to the terrain's own grass in tone, so where the height-aware blend hands over to the grass
  // it does so between two similar colours (round 3 wip: a hard "coastline" against the dirt)
  const e1 = worn.add(n1.sub(0.5).mul(0.35)).add(n2.sub(0.5).mul(0.22)).add(n3.sub(0.5).mul(0.15));
  const outer = float(1).sub(smoothstep(0.32, 0.8, e1));
  const blades = smoothstep(0.42, 0.62, tnoise(vec2(xz.x.mul(3.0).add(xz.y.mul(9.0)), xz.y.mul(3.0).sub(xz.x.mul(9.0))), 40.4));
  const litter = mix(vec3(0.062, 0.07, 0.04), vec3(0.122, 0.104, 0.056), blades.mul(0.6).add(n2.mul(0.4))).mul(mix(0.8, 1.12, n3));
  alb = mix(alb, litter, outer.mul(float(1).sub(sooty.mul(0.8))));
  // the trodden edge of a path or the yard lies a little lower than the turf: a shadowed rim along it
  const rim = smoothstep(0.25, 0.5, e1).mul(float(1).sub(smoothstep(0.55, 0.85, e1)));
  alb = alb.mul(float(1).sub(rim.mul(0.32)));
  const w = smoothstep(0.03, 0.4, worn.add(n1.sub(0.5).mul(0.35)).add(n2.sub(0.5).mul(0.25)).add(n3.sub(0.5).mul(0.2)));
  const layer: GroundSample = {
    albedo: alb,
    // packed flat in the middle, as uneven as the grass at the outer part (so neither wins on its texture alone)
    height: mix(L.mud.height.mul(0.25).add(0.12), float(0.45), outer),
    pert: mix(L.mud.pert, L.silt.pert, grav).mul(mix(mix(0.55, 0.2, polish), float(0.7), outer)),
    rough: mix(L.mud.rough, 0.95, 0.6),
    ao: mix(float(1), L.mud.ao, 0.45),
  };
  return { layer, w, keep: w.oneMinus() };
}
