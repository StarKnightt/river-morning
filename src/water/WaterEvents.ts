import * as THREE from 'three/webgpu';
import {
  atan, cameraPosition, cameraViewMatrix, float, instancedDynamicBufferAttribute, length, max as maxN, smoothstep, step, uniform, uv, vec2, vec3, vec4,
} from '../gfx/tsl';
import { MAX_EVENTS, RING_KIND, eventA, eventB, eventC } from './surface';
import { LAYER_WATER } from './PoolReflector';
import { uw } from './Underwater';

/** What the event API needs from the water at a point (RiverWater.query). */
export type WaterAt = (x: number, z: number) => { y: number; flowX: number; flowZ: number };

/** A ripple train's shape: wavelength (m), speed of its front (m/s), radius the front starts from (m). */
export interface TrainShape {
  wavelength?: number;
  speed?: number;
  r0?: number;
}

/**
 * Splashes and rings. A splash layers a sharp first ring, a softer echo, a short-lived foam blob and a burst
 * of droplets falling back with their own tiny rings. strength 1 is a fish jumping clear (4 cm swell);
 * 0.3 is a lure landing, 0.1 a fish rising to a fly. Smaller disturbances use the crisp ripple trains (a float
 * landing, a bite, hands in the water, the line cutting the surface), boils and swirls (a fish turning under it).
 */
export class WaterEvents {
  readonly droplets: Droplets;

  constructor(private at: WaterAt, maxDroplets = 256) {
    this.droplets = new Droplets(maxDroplets);
  }

  /**
   * A ring at (x, z) starting at time t (engine time). It drifts with the current at its centre. Takes the lowest
   * free slot (the shaders loop over the slots up to the highest live one); with none free it replaces the weakest
   * live ring, or is dropped if it would be the weakest itself.
   */
  ring(x: number, z: number, t: number, amplitude: number, life = 3.5, foamRadius = 0,
    kind: number = RING_KIND.swell, size = 0, speed = 0, r0 = 0) {
    let i = -1;
    for (let k = 0; k < MAX_EVENTS; k++) {
      if (t - eventA[k].z >= eventB[k].z || eventA[k].z > t + 10) { i = k; break; }
    }
    if (i < 0) {
      const weight = (a: number, kd: number) => a * (kd === RING_KIND.train ? 6 : 1);
      let best = Infinity;
      for (let k = 0; k < MAX_EVENTS; k++) {
        const left = Math.max(0, 1 - (t - eventA[k].z) / eventB[k].z);
        const w = weight(eventA[k].w, eventC[k].x) * left;
        if (w < best) { best = w; i = k; }
      }
      if (weight(amplitude, kind) < best) return;
    }
    const w = this.at(x, z);
    eventA[i].set(x, z, t, amplitude);
    eventB[i].set(w.flowX, w.flowZ, life, foamRadius);
    eventC[i].set(kind, size, speed, r0);
  }

  /** A crisp train of capillary ripples (the rings a float, a nibble or a wrist pushes out). amplitude in m. */
  train(x: number, z: number, t: number, amplitude: number, life = 3, s: TrainShape = {}) {
    this.ring(x, z, t, amplitude, life, 0, RING_KIND.train, s.wavelength ?? 0.03, s.speed ?? 0.28, s.r0 ?? 0.02);
  }

  /** A smooth dome of water welling up over a fish turning below, radius (m) spreading at `spread` m/s. */
  boil(x: number, z: number, t: number, amplitude: number, radius: number, life = 2, spread = 0.12) {
    this.ring(x, z, t, amplitude, life, 0, RING_KIND.boil, radius, spread);
  }

  /** A vortex dimple left by a tail stroke, spin +-1 (its spiral arms of broken water). */
  swirl(x: number, z: number, t: number, amplitude: number, radius: number, life = 1.4, spin = 1) {
    this.ring(x, z, t, amplitude, life, 0, RING_KIND.swirl, radius, 0, spin);
  }

  splash(x: number, z: number, t: number, strength = 1) {
    const s = Math.max(0.02, strength);
    // the main swell, then the jet falling back in the middle a third of a second later
    this.ring(x, z, t, 0.04 * s, 3.5 + 1.5 * Math.min(s, 1), 0.13 * Math.sqrt(s));
    this.ring(x, z, t + 0.35, 0.01 * s, 2.5 + 1.0 * Math.min(s, 1), 0.07 * Math.sqrt(s));
    const y = this.at(x, z).y, seed = Math.round(x * 131 + z * 71 + t * 7);
    const count = Math.round(50 * Math.min(s, 2));
    if (count > 0) this.droplets.burst(x, y, z, count, 2.4 * Math.sqrt(s), 0.012 + 0.006 * Math.min(s, 1), seed, 0.6);
    // the jet: a few drops thrown nearly straight up, still falling back about 0.6 s after the hit
    const jet = Math.round(6 * Math.min(s, 2));
    if (jet > 0) this.droplets.burst(x, y, z, jet, 3.3 * Math.sqrt(s), 0.014, seed + 1, 0.3, 0.85);
  }

  /**
   * A small object dropping onto still water (a float at the end of a cast: strength ~1): a crown of fine drops
   * thrown up and out round it, the jet rising from the closing hole a moment later, a crisp ring train with a
   * second one inside it, and the soft swell of the water pushed aside under them.
   */
  plop(x: number, z: number, t: number, strength = 1) {
    const s = Math.max(0.05, strength);
    this.ring(x, z, t, 0.0035 * s, 2.4, 0.03 * Math.sqrt(s));
    this.train(x, z, t, 0.003 * s, 5, { wavelength: 0.04, speed: 0.3, r0: 0.03 });
    this.train(x, z, t + 0.17, 0.0017 * s, 4, { wavelength: 0.028, speed: 0.25, r0: 0.02 });
    const y = this.at(x, z).y, seed = Math.round(x * 131 + z * 71 + t * 7);
    const k = Math.min(s, 2);
    // the crown: a low ring of fine drops and a taller, sparser one; the jet a moment later
    this.droplets.crown(x, y, z, Math.round(28 * k), 1.2 * Math.sqrt(s), 0.0045, seed, 0.024, 0.45, 0.85);
    this.droplets.crown(x, y, z, Math.round(22 * k), 1.75 * Math.sqrt(s), 0.0062, seed + 2, 0.02, 0.15, 0.4);
    this.droplets.burst(x, y, z, Math.max(1, Math.round(7 * k)), 1.8 * Math.sqrt(s), 0.006, seed + 1, 0.12, 0.7, 0.07);
  }

  /** Drops flung along (dx, dz) from a point (the line hissing through the surface, a tail slapping it). */
  spray(x: number, y: number, z: number, dx: number, dz: number, count: number, speed: number, size: number, seed: number) {
    this.droplets.fan(x, y, z, Math.atan2(dz, dx), count, speed, size, seed);
  }

  private dropRings = 0;

  /** Droplet ring: the biggest drops leave a faint ring where they fall back, every third fine one a tiny train. */
  landed = (t: number) => (x: number, z: number, size: number) => {
    if (size > 0.022) this.ring(x, z, t, 0.0015, 1.6, 0.03);
    else if (size > 0.003 && size < 0.009 && this.dropRings++ % 3 === 0) {
      this.train(x, z, t, 0.00035 + size * 0.05, 1.3, { wavelength: 0.016, speed: 0.2, r0: 0.004 });
    }
  };

  update(dt: number, t: number) {
    this.droplets.update(dt, (x, z) => this.at(x, z).y, this.landed(t));
  }

  /** calm water: every ring and drop gone (photo poses start from here) */
  clear() {
    for (let i = 0; i < MAX_EVENTS; i++) eventA[i].z = -1e5;
    this.droplets.clear();
  }
}

/**
 * Up to `max` droplets as one instanced sprite (SpriteNodeMaterial + an InstancedBufferAttribute updated on
 * the CPU each frame). CPU simulation keeps it identical on WebGPU and the WebGL2 fallback, and it is
 * deterministic in shot mode (dt = 0, seeded bursts). A drop smaller than ~1.5 pixels is drawn that wide and
 * fainter, so fine spray neither shimmers nor vanishes. Each drop is drawn stretched along its motion on the
 * screen (the streak a fast drop leaves in a photograph) and catches the sun its own way: most show the sky, a few
 * flash with the sun.
 */
export class Droplets {
  readonly sprite: THREE.Sprite;
  readonly sunColor = uniform(new THREE.Color(1, 0.82, 0.59));
  readonly skyColor = uniform(new THREE.Color(0.38, 0.41, 0.43));
  /** streak length = speed across the view x this (s, an exposure; at 1/110 s most spray still read as round beads) */
  readonly shutter = uniform(1 / 70);
  /** GPU: position and the size it is drawn at (0 while waiting or free) */
  private data: Float32Array;
  /** GPU: velocity and how brightly it catches the sun (0..1) */
  private gvel: Float32Array;
  private vel: Float32Array;
  /** real size (m); 0 = free */
  private size: Float32Array;
  /** seconds before it appears (the jet rises after the crown) */
  private wait: Float32Array;
  private attr: THREE.InstancedBufferAttribute;
  private vattr: THREE.InstancedBufferAttribute;
  private glintSeed = 1;

  constructor(readonly max: number) {
    this.data = new Float32Array(max * 4);
    this.gvel = new Float32Array(max * 4);
    this.vel = new Float32Array(max * 3);
    this.size = new Float32Array(max);
    this.wait = new Float32Array(max);
    this.attr = new THREE.InstancedBufferAttribute(this.data, 4);
    this.attr.setUsage(THREE.DynamicDrawUsage);
    this.vattr = new THREE.InstancedBufferAttribute(this.gvel, 4);
    this.vattr.setUsage(THREE.DynamicDrawUsage);
    const m = new THREE.SpriteNodeMaterial({ transparent: true, depthWrite: false });
    // pass the attribute object: a raw vec4 typed array would not be marked instanced
    const p = instancedDynamicBufferAttribute(this.attr);
    const v = instancedDynamicBufferAttribute(this.vattr);
    m.positionNode = p.xyz;
    const minS = length(p.xyz.sub(cameraPosition)).mul(uw.pixAngle).mul(1.5);
    const s = maxN(p.w, minS).mul(step(1e-6, p.w));
    // the streak: the motion across the view during the exposure, laid along the sprite's x
    const vv = cameraViewMatrix.mul(vec4(v.xyz, 0)).xy;
    const streak = length(vv).mul(this.shutter);
    m.rotationNode = atan(vv.y, vv.x.add(1e-5));
    m.scaleNode = vec2(s.add(streak.mul(step(1e-6, p.w))), s);
    const cover = p.w.div(s.max(1e-6)).pow(2).min(1);
    const d = length(uv().sub(0.5)).mul(2);
    // a droplet is a tiny lens: the sky round it and a point of sun in its middle (bright in a few, dull in most);
    // bloom does the rest. A streak spreads the same light over its length.
    const glint = v.w.mul(v.w).mul(v.w).mul(1.7).add(0.35);
    const thin = s.div(s.add(streak)).max(0.25);
    const col = vec3(this.skyColor).mul(1.1).add(vec3(this.sunColor).mul(0.9).mul(glint));
    const one = float(1);
    m.colorNode = vec4(col.mul(one.sub(smoothstep(0.0, 0.6, d)).mul(1.4).add(0.75)), one.sub(smoothstep(0.5, 1, d)).mul(0.62).mul(cover).mul(thin.sqrt()));
    this.sprite = new THREE.Sprite(m);
    this.sprite.count = max;
    this.sprite.frustumCulled = false;
    this.sprite.renderOrder = 3;
    this.sprite.layers.set(LAYER_WATER);
  }

  private rng(seed: number) {
    let s = seed >>> 0;
    return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  }

  private add(i: number, x: number, y: number, z: number, vx: number, vy: number, vz: number, size: number, delay: number) {
    this.data[i * 4] = x;
    this.data[i * 4 + 1] = y;
    this.data[i * 4 + 2] = z;
    this.data[i * 4 + 3] = delay > 0 ? 0 : size;
    this.vel[i * 3] = vx;
    this.vel[i * 3 + 1] = vy;
    this.vel[i * 3 + 2] = vz;
    this.gvel[i * 4] = vx;
    this.gvel[i * 4 + 1] = vy;
    this.gvel[i * 4 + 2] = vz;
    // how squarely this drop catches the sun (a hash of the slot and a running seed: stable in shot mode)
    const h = Math.sin((i + 1) * 12.9898 + (this.glintSeed = (this.glintSeed * 1.37 + 0.71) % 97) * 78.233) * 43758.5453;
    this.gvel[i * 4 + 3] = h - Math.floor(h);
    this.size[i] = size;
    this.wait[i] = delay;
  }

  /** One drop (the first free slot), launched now; false when every slot is taken. */
  spawn(x: number, y: number, z: number, vx: number, vy: number, vz: number, size: number): boolean {
    for (let i = 0; i < this.max; i++) {
      if (this.size[i] > 0) continue;
      this.add(i, x, y, z, vx, vy, vz, size, 0);
      this.attr.needsUpdate = true;
      this.vattr.needsUpdate = true;
      return true;
    }
    return false;
  }

  /**
   * One drop launched `age` s ago (shot-mode captures of a moment): flown forward with the same gravity and drag;
   * dropped if it has already fallen back into the water.
   */
  spawnAged(x: number, y: number, z: number, vx: number, vy: number, vz: number, size: number, age: number,
    surfaceY: (x: number, z: number) => number): boolean {
    const h = 1 / 120;
    for (let t = 0; t < age; t += h) {
      vy -= 9.81 * h;
      const drag = 1 - 0.6 * h;
      vx *= drag;
      vy *= drag;
      vz *= drag;
      x += vx * h;
      y += vy * h;
      z += vz * h;
      if (vy < 0 && y < surfaceY(x, z)) return false;
    }
    return this.spawn(x, y, z, vx, vy, vz, size);
  }

  /** `spread`: largest angle from vertical (rad); launch speeds range over [vmin, 1] x speed; `delay` (s) */
  burst(x: number, y: number, z: number, count: number, speed: number, size: number, seed: number, spread = 0.6, vmin = 0.45, delay = 0) {
    const rnd = this.rng(seed);
    for (let n = 0, i = 0; n < count && i < this.max; i++) {
      if (this.size[i] > 0) continue;
      const th = rnd() * spread, ph = rnd() * Math.PI * 2, v = speed * (vmin + (1 - vmin) * rnd());
      // drops of every size: mostly small, a few big ones
      this.add(i, x + (rnd() - 0.5) * 0.05, y + 0.01, z + (rnd() - 0.5) * 0.05,
        Math.sin(th) * Math.cos(ph) * v, Math.cos(th) * v, Math.sin(th) * Math.sin(ph) * v, size * (0.35 + 1.6 * rnd() ** 2.2), delay);
      n++;
    }
    this.attr.needsUpdate = true;
    this.vattr.needsUpdate = true;
  }

  /** A crown: drops leaving a ring of radius r0, thrown outward between thMin and thMax from vertical. */
  crown(x: number, y: number, z: number, count: number, speed: number, size: number, seed: number, r0: number, thMin: number, thMax: number) {
    const rnd = this.rng(seed);
    for (let n = 0, i = 0; n < count && i < this.max; i++) {
      if (this.size[i] > 0) continue;
      const ph = ((n + rnd() * 0.8) / count) * Math.PI * 2, th = thMin + (thMax - thMin) * rnd(), v = speed * (0.55 + 0.45 * rnd());
      const cx = Math.cos(ph), cz = Math.sin(ph);
      this.add(i, x + cx * r0, y + 0.006, z + cz * r0, Math.sin(th) * cx * v, Math.cos(th) * v, Math.sin(th) * cz * v,
        size * (0.5 + rnd()), 0);
      n++;
    }
    this.attr.needsUpdate = true;
    this.vattr.needsUpdate = true;
  }

  /** A fan of drops flung low toward azimuth `az` (rad from +x toward +z), +-0.5 rad wide. */
  fan(x: number, y: number, z: number, az: number, count: number, speed: number, size: number, seed: number) {
    const rnd = this.rng(seed);
    for (let n = 0, i = 0; n < count && i < this.max; i++) {
      if (this.size[i] > 0) continue;
      const a = az + (rnd() - 0.5), el = 0.35 + rnd() * 0.6, v = speed * (0.5 + 0.5 * rnd());
      this.add(i, x, y + 0.005, z, Math.cos(el) * Math.cos(a) * v, Math.sin(el) * v, Math.cos(el) * Math.sin(a) * v,
        size * (0.35 + 1.6 * rnd() ** 2.2), 0);
      n++;
    }
    this.attr.needsUpdate = true;
    this.vattr.needsUpdate = true;
  }

  update(dt: number, surfaceY: (x: number, z: number) => number, landed: (x: number, z: number, size: number) => void) {
    if (dt <= 0) return;
    const d = this.data, v = this.vel;
    let live = false;
    for (let i = 0; i < this.max; i++) {
      if (this.size[i] <= 0) continue;
      live = true;
      if (this.wait[i] > 0) {
        if ((this.wait[i] -= dt) > 0) continue;
        d[i * 4 + 3] = this.size[i];
      }
      const k = i * 4, kv = i * 3;
      v[kv + 1] -= 9.81 * dt;
      const drag = 1 - 0.6 * dt;
      v[kv] *= drag;
      v[kv + 1] *= drag;
      v[kv + 2] *= drag;
      d[k] += v[kv] * dt;
      d[k + 1] += v[kv + 1] * dt;
      d[k + 2] += v[kv + 2] * dt;
      this.gvel[k] = v[kv];
      this.gvel[k + 1] = v[kv + 1];
      this.gvel[k + 2] = v[kv + 2];
      if (v[kv + 1] < 0 && d[k + 1] < surfaceY(d[k], d[k + 2])) {
        landed(d[k], d[k + 2], this.size[i]);
        d[k + 3] = 0;
        this.size[i] = 0;
      }
    }
    if (live) {
      this.attr.needsUpdate = true;
      this.vattr.needsUpdate = true;
    }
  }

  /** drops in the air (debug): count, and the highest one above the water */
  stats(surfaceY: (x: number, z: number) => number) {
    let n = 0, top = 0;
    for (let i = 0; i < this.max; i++) {
      if (this.size[i] <= 0 || this.data[i * 4 + 3] <= 0) continue;
      n++;
      top = Math.max(top, this.data[i * 4 + 1] - surfaceY(this.data[i * 4], this.data[i * 4 + 2]));
    }
    return { n, top: Math.round(top * 1000) / 1000 };
  }

  clear() {
    this.size.fill(0);
    for (let i = 0; i < this.max; i++) this.data[i * 4 + 3] = 0;
    this.attr.needsUpdate = true;
    this.vattr.needsUpdate = true;
  }
}
