/** Small deterministic helpers shared by the fishing simulation (no allocations). */

/** mulberry32: fast seeded PRNG, deterministic across browsers (shot poses, tests). */
export class Rng {
  private s: number;
  constructor(seed = 1) {
    this.s = seed >>> 0;
  }
  seed(seed: number) {
    this.s = seed >>> 0;
  }
  next(): number {
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  range(a: number, b: number): number {
    return a + (b - a) * this.next();
  }
  /** roughly normal (sum of three uniforms), mean 0, sd ~1 */
  gauss(): number {
    return (this.next() + this.next() + this.next() - 1.5) * 2;
  }
  sign(): number {
    return this.next() < 0.5 ? -1 : 1;
  }
}

export const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const smooth01 = (t: number) => {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
};
export const smoothstep = (a: number, b: number, v: number) => smooth01((v - a) / (b - a));
/** frame-rate independent exponential approach factor for a rate (1/s) */
export const approach = (rate: number, dt: number) => 1 - Math.exp(-rate * dt);
export const easeOutQuad = (t: number) => 1 - (1 - t) * (1 - t);
export const easeInOut = (t: number) => (t < 0.5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t));
