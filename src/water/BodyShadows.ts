import * as THREE from 'three/webgpu';
import { Fn, If, Loop, abs, clamp, dot, float, int, length, mix, smoothstep, uniform, uniformArray, vec3, type N } from '../gfx/tsl';
import { Q } from '../core/quality';

/**
 * Sharp shadows of small bodies under the water (the trout, a forearm held in the river), cast on the bed, the rocks
 * and other trout along the refracted sunbeam. The sun's shadow map (2.5 cm texels, a 3.5-texel filter) blurs a
 * 5 cm wide trout's shadow away, and a fattened caster only made it a soft smudge under the fish; in sunny shallows
 * a trout's shadow is a crisp dark copy of it on the gravel, set apart from the body, the first thing you see.
 *
 * Each body is a tapered capsule from `a` to `b`: the receiver's ray toward the sun is tested against the segment
 * (closest approach), and the profile along it (fish: a tail fan, the narrow wrist, the thickest point a third back
 * from the snout, the blunt head; limb: even) sets the radius. The penumbra widens with the distance along the beam
 * (the sun disc and forward scattering). A body never shadows itself: receivers closer than ~2.5 radii along the
 * beam are skipped (the far flank is dark from N.L anyway).
 *
 * Slots 0..15: the trout (FishSchool, submerged ones only: in the air the shadow map has them); 16..23: limbs.
 */
export const MAX_BODIES = 24;
export const LIMB_SLOT0 = 16;
/** xyz: the tail end (fish) or the elbow (limb); w: radius at the thickest point (m), 0 = empty */
export const bodyA = Array.from({ length: MAX_BODIES }, () => new THREE.Vector4(0, -1e4, 0, 0));
/** xyz: the snout (fish) or the far end (limb); w: profile (0 fish, 1 even) */
export const bodyB = Array.from({ length: MAX_BODIES }, () => new THREE.Vector4(0, -1e4, 0, 0));
export const bodyU = {
  a: uniformArray(bodyA, 'vec4'),
  b: uniformArray(bodyB, 'vec4'),
  /** slots 0..count-1 are tested */
  count: uniform(0, 'int'),
  /** share of the direct sun a body blocks (forward scattering fills the core a little) */
  dark: uniform(0.94),
};

let fishCount = 0;
let limbCount = 0;
const refresh = () => {
  bodyU.count.value = limbCount > 0 ? LIMB_SLOT0 + limbCount : fishCount;
};

/** FishSchool, every frame: the submerged trout as (tail tip, snout, radius); slots past `n` are cleared. */
export function setFishBodies(n: number) {
  // nearest first: the graphics preset keeps the closest Q.k.fishShadows (all 16 from High up)
  const m = Math.min(n, LIMB_SLOT0, Q.k.fishShadows);
  for (let i = m; i < Math.max(n, fishCount); i++) bodyA[i].w = 0;
  fishCount = m;
  refresh();
}

/** The angler's forearms and hands under the water (release): up to 8 capsules from `list`, or none. */
export function setLimbBodies(list: ReadonlyArray<{ a: THREE.Vector3; b: THREE.Vector3; r: number }>) {
  const n = Math.min(list.length, MAX_BODIES - LIMB_SLOT0);
  for (let k = 0; k < MAX_BODIES - LIMB_SLOT0; k++) {
    const i = LIMB_SLOT0 + k;
    if (k < n) {
      const l = list[k];
      bodyA[i].set(l.a.x, l.a.y, l.a.z, l.r);
      bodyB[i].set(l.b.x, l.b.y, l.b.z, 1);
    } else bodyA[i].w = 0;
  }
  limbCount = n;
  refresh();
}

/**
 * Direct-sun factor at P (world) from every body (1 = unshadowed). `toSun`: unit direction toward the sun along the
 * refracted beam (-uw.sunRefr).
 */
export const bodyShadow = Fn(([P, toSun]: [N, N]) => {
  const lit = float(1).toVar();
  Loop({ start: int(0), end: bodyU.count, type: 'int' }, ({ i }: { i: N }) => {
    const A4 = bodyU.a.element(i), B4 = bodyU.b.element(i);
    const R = A4.w;
    const A = A4.xyz, d = B4.xyz.sub(A);
    // cheap reject first: where this receiver's ray toward the sun crosses the body's mid-height, it must pass within
    // half a length (plus the radius, the body's slope and a margin) of the body; almost every pixel is far from all
    const mid = A.add(d.mul(0.5));
    const tm = mid.y.sub(P.y).div(toSun.y.max(0.2));
    const near = length(P.xz.add(toSun.xz.mul(tm)).sub(mid.xz)).lessThan(length(d).mul(0.5).add(R.mul(3)).add(abs(d.y)).add(0.05));
    If(R.greaterThan(0).and(tm.greaterThan(R.mul(-2))).and(near), () => {
      const cc = dot(d, d).max(1e-6);
      // closest approach of the ray P + toSun t (t >= 0) to the segment A + d u
      const w0 = P.sub(A);
      const bb = dot(toSun, d), dd = dot(toSun, w0), ee = dot(d, w0);
      const den = cc.sub(bb.mul(bb)).max(1e-6);
      const u0 = clamp(ee.sub(bb.mul(dd)).div(den), 0, 1);
      const t = dot(toSun, A.add(d.mul(u0)).sub(P)).max(0);
      const Q = P.add(toSun.mul(t));
      const u = clamp(dot(Q.sub(A), d).div(cc), 0, 1);
      // a trout is half again as deep as it is wide (its slanting shadow is that much wider); limbs are round
      const off = Q.sub(A.add(d.mul(u)));
      const dist = length(vec3(off.x, off.y.mul(mix(float(0.65), float(1), B4.w)), off.z));
      // fish profile from the tail tip (u 0) to the snout (1): a fan at the tail, the narrow wrist, thickest a third
      // back from the snout, a blunt head; limbs are even
      const fan = mix(float(0.5), float(0.24), smoothstep(0.0, 0.17, u));
      const bodyP = mix(float(0.24), float(1), smoothstep(0.17, 0.64, u)).mul(float(1).sub(smoothstep(0.66, 1.0, u).mul(0.5)));
      const prof = mix(mix(fan, bodyP, step017(u)), float(1), B4.w);
      const rad = R.mul(prof);
      const soft = t.mul(0.014).add(0.003);
      const self = smoothstep(R.mul(1.6), R.mul(2.6), t);
      const cover = float(1).sub(smoothstep(rad.sub(soft), rad.add(soft), dist)).mul(self);
      lit.mulAssign(float(1).sub(cover.mul(bodyU.dark)));
    });
  });
  return lit;
});

/** 0 on the tail fan, 1 on the body (a smooth switch at the wrist) */
const step017 = (u: N) => smoothstep(0.15, 0.19, u);
