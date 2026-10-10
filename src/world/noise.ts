/** Deterministic 2D gradient noise used for terrain shaping on the CPU. */

const PERM = new Uint8Array(512);
const GRAD: Array<[number, number]> = [];
{
  let s = 1337;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const p = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  for (let i = 0; i < 512; i++) PERM[i] = p[i & 255];
  for (let i = 0; i < 256; i++) {
    const a = rnd() * Math.PI * 2;
    GRAD.push([Math.cos(a), Math.sin(a)]);
  }
}

const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);

export function noise2(x: number, y: number): number {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const X = xi & 255, Y = yi & 255;
  const g = (ix: number, iy: number, dx: number, dy: number) => {
    const gr = GRAD[PERM[ix + PERM[iy]]];
    return gr[0] * dx + gr[1] * dy;
  };
  const n00 = g(X, Y, xf, yf);
  const n10 = g(X + 1, Y, xf - 1, yf);
  const n01 = g(X, Y + 1, xf, yf - 1);
  const n11 = g(X + 1, Y + 1, xf - 1, yf - 1);
  const u = fade(xf), v = fade(yf);
  return (n00 + u * (n10 - n00) + v * (n01 - n00 + u * (n11 - n10 - n01 + n00))) * 1.4;
}

export function fbm(x: number, y: number, octaves = 5, lacunarity = 2.03, gain = 0.5): number {
  let sum = 0, amp = 1, norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amp * noise2(x, y);
    norm += amp;
    amp *= gain;
    // rotate each octave to avoid axis-aligned artefacts
    const nx = x * 0.8 - y * 0.6, ny = x * 0.6 + y * 0.8;
    x = nx * lacunarity + 17.3;
    y = ny * lacunarity - 9.1;
  }
  return sum / norm;
}

/** Ridged multifractal: sharp crests for mountain ranges. Returns roughly 0..1. */
export function ridged(x: number, y: number, octaves = 7): number {
  let sum = 0, amp = 0.5, weight = 1, norm = 0;
  for (let i = 0; i < octaves; i++) {
    let n = 1 - Math.abs(noise2(x, y));
    n *= n;
    n *= weight;
    weight = Math.min(1, Math.max(0, n * 2));
    sum += n * amp;
    norm += amp;
    amp *= 0.5;
    const nx = x * 0.8 - y * 0.6, ny = x * 0.6 + y * 0.8;
    x = nx * 2.1 + 3.7;
    y = ny * 2.1 - 1.3;
  }
  return sum / norm;
}

export const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
