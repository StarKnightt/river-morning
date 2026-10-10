import { river, newSample } from '../world/River';
import { LOOPS, type GroupId, type LoopId } from './catalog';
import { dayMix, type DayMix } from './timeOfDay';
import { clamp, dbToGain, smoothstep } from './util';

/** What the ambience needs from the engine. */
export interface AmbienceHost {
  loadLoop(id: LoopId): Promise<AudioBuffer | null>;
  play(id: GroupId, x?: number, y?: number, z?: number, gainDb?: number, rate?: number): boolean;
}

export type BedId = Exclude<LoopId, 'reel'>;

/** Load order: the river first, then the layers that carry the time of day, then the camp. */
const BED_ORDER: BedId[] = ['riverCalm', 'riverRiffle', 'wind', 'birds', 'crickets', 'campfire', 'coffee'];

/** Beds a day mix needs decoded (birds and crickets only when they are heard). */
export const bedsNeeded = (day: DayMix): BedId[] =>
  BED_ORDER.filter((id) => (id === 'birds' ? day.birds > 0 : id === 'crickets' ? day.crickets > 0 : true));

const G = {
  riverCalm: dbToGain(LOOPS.riverCalm.gainDb),
  riverRiffle: dbToGain(LOOPS.riverRiffle.gainDb),
  birds: dbToGain(LOOPS.birds.gainDb),
  wind: dbToGain(LOOPS.wind.gainDb),
  crickets: dbToGain(LOOPS.crickets.gainDb),
  campfire: dbToGain(LOOPS.campfire.gainDb),
  coffee: dbToGain(LOOPS.coffee.gainDb),
};

interface RiverMix {
  /** summed intensities (normalised by the references below) */
  calm: number;
  riffle: number;
  /** -1..1: where each bed comes from, projected on the camera's right vector */
  calmPan: number;
  rifflePan: number;
}

/**
 * The river as point emitters every 2 m along the walkable corridor. Each station is split between the calm bed and
 * the riffle bed by its flow speed, so walking along the bank crossfades the two by where the fast water really is.
 * The riffle sounds from its whole surface (centreline emitters, distance measured to the water) and falls as 1/r²
 * per emitter (~3 dB per doubling close by, 6 dB once the riffle looks short). The calm gurgle comes from the two
 * water edges, louder where the water moves (pool head and tail) than over the deep slack pool, and falls as 1/r³:
 * it only carries a few metres from the bank.
 */
class RiverEmitters {
  private readonly n: number;
  private readonly x: Float32Array;
  private readonly z: Float32Array;
  private readonly y: Float32Array;
  private readonly half: Float32Array;
  /** unit normal to the flow (toward the south bank) */
  private readonly nx: Float32Array;
  private readonly nz: Float32Array;
  private readonly wc: Float32Array;
  private readonly wr: Float32Array;
  private refCalm = 1;
  private refRiffle = 1;

  constructor() {
    const X0 = -130, X1 = 120, STEP = 2;
    const n = (this.n = Math.floor((X1 - X0) / STEP) + 1);
    this.x = new Float32Array(n);
    this.z = new Float32Array(n);
    this.y = new Float32Array(n);
    this.half = new Float32Array(n);
    this.nx = new Float32Array(n);
    this.nz = new Float32Array(n);
    this.wc = new Float32Array(n);
    this.wr = new Float32Array(n);
    const s = newSample();
    for (let i = 0; i < n; i++) {
      const x = X0 + i * STEP, z = river.zAt(x);
      river.sample(x, z, s);
      this.x[i] = x;
      this.z[i] = z;
      this.y[i] = s.waterY;
      this.half[i] = s.half;
      this.nx[i] = -s.tz;
      this.nz[i] = s.tx;
      this.wr[i] = smoothstep(1.0, 1.9, s.speed);
      this.wc[i] = (1 - this.wr[i]) * (0.45 + 0.55 * smoothstep(0.2, 0.9, s.speed)) * 0.5;
    }
    // 0 dB spots: on the bar by the pool (camera spot C2) and on the bank above the riffle (C3)
    const m: RiverMix = { calm: 0, riffle: 0, calmPan: 0, rifflePan: 0 };
    this.measure(4.6, river.waterYAt(4.6) + 1.4, 6.4, 1, 0, m);
    const rc = m.calm;
    this.measure(-19, river.waterYAt(-19) + 1.8, 9.6, 1, 0, m);
    this.refCalm = rc;
    this.refRiffle = m.riffle;
  }

  measure(fx: number, fy: number, fz: number, rx: number, rz: number, out: RiverMix) {
    let ic = 0, ir = 0, cx = 0, cz = 0, qx = 0, qz = 0;
    for (let i = 0; i < this.n; i++) {
      const dx = this.x[i] - fx, dz = this.z[i] - fz, dy2 = (this.y[i] - fy) ** 2 + 1;
      const rh = Math.sqrt(dx * dx + dz * dz) + 1e-3;
      const r = Math.max(0, rh - 0.7 * this.half[i]);
      const d2 = r * r + dy2;
      const b = this.wr[i] / d2;
      ir += b;
      qx += (b * dx) / rh;
      qz += (b * dz) / rh;
      const w = this.wc[i];
      if (w <= 0) continue;
      const ex = this.nx[i] * this.half[i], ez = this.nz[i] * this.half[i];
      for (let side = -1; side <= 1; side += 2) {
        const sx = dx + side * ex, sz = dz + side * ez;
        const e2 = sx * sx + sz * sz + dy2;
        const a = w / (e2 * Math.sqrt(e2)), sl = Math.sqrt(sx * sx + sz * sz) + 1e-3;
        ic += a;
        cx += (a * sx) / sl;
        cz += (a * sz) / sl;
      }
    }
    out.calm = ic / this.refCalm;
    out.riffle = ir / this.refRiffle;
    out.calmPan = ic > 0 ? (cx * rx + cz * rz) / ic : 0;
    out.rifflePan = ir > 0 ? (qx * rx + qz * rz) / ir : 0;
  }
}

/** Slow gusts in [0, 1] from three incommensurate swells (37 s, 17.3 s, 6.1 s), so the pattern never lines up. */
function gust(t: number) {
  const w = 0.55 * Math.sin(t * 0.1698 + 1.3) + 0.3 * Math.sin(t * 0.3632 + 4.1) + 0.15 * Math.sin(t * 1.03 + 2.2);
  return smoothstep(-0.25, 0.85, w);
}

/** One looping bed: a buffer source started at a random point (two for the rebuilt stereo crickets) into a gain. */
class Bed {
  buffer: AudioBuffer | null = null;
  requested = false;
  private src: AudioBufferSourceNode | null = null;
  private src2: AudioBufferSourceNode | null = null;
  private level = 0;
  private fresh = false;
  private quietSince = -1;

  constructor(private ctx: BaseAudioContext, readonly gain: GainNode, private input: AudioNode, private split = false) {}

  get playing() {
    return this.src !== null;
  }

  /** linear target gain; the source starts when the bed becomes audible and stops after 8 s of silence */
  set(target: number, now: number, tc: number) {
    if (!this.buffer) return;
    if (target > 1e-4) {
      this.quietSince = -1;
      if (!this.src) this.start(now);
    } else if (this.src) {
      if (this.quietSince < 0) this.quietSince = now;
      else if (now - this.quietSince > 8) this.stop(now);
    }
    if (Math.abs(target - this.level) > 0.02 * Math.max(target, this.level) + 1e-5) {
      this.gain.gain.setTargetAtTime(target, now, this.fresh ? Math.max(tc, 0.8) : tc);
      this.level = target;
      this.fresh = false;
    }
  }

  private start(now: number) {
    const b = this.buffer!;
    const offset = Math.random() * b.duration;
    this.gain.gain.cancelScheduledValues(now);
    this.gain.gain.setValueAtTime(0, now);
    this.level = 0;
    this.fresh = true;
    this.src = this.source(b, now, offset, 0);
    // the crickets' right channel is the same loop half a loop later (how the prep built the file)
    if (this.split) this.src2 = this.source(b, now, (offset + b.duration / 2) % b.duration, 1);
  }

  private source(b: AudioBuffer, now: number, offset: number, channel: number) {
    const s = this.ctx.createBufferSource();
    s.buffer = b;
    s.loop = true;
    if (this.split) s.connect(this.input, 0, channel);
    else s.connect(this.input);
    s.start(now, offset);
    return s;
  }

  private stop(now: number) {
    for (const s of [this.src, this.src2]) {
      if (!s) continue;
      s.stop(now);
      s.disconnect();
    }
    this.src = this.src2 = null;
    this.quietSince = -1;
  }
}

function pointPanner(ctx: BaseAudioContext, x: number, y: number, z: number, ref: number, rolloff: number, out: AudioNode) {
  const p = ctx.createPanner();
  p.panningModel = 'equalpower';
  p.distanceModel = 'inverse';
  p.refDistance = ref;
  p.rolloffFactor = rolloff;
  p.maxDistance = 10000;
  if (p.positionX) {
    p.positionX.value = x;
    p.positionY.value = y;
    p.positionZ.value = z;
  } else p.setPosition(x, y, z);
  p.connect(out);
  return p;
}

const ramp = (p: AudioParam, v: number, last: number, eps: number, now: number, tc: number) => {
  if (Math.abs(v - last) <= eps) return last;
  p.setTargetAtTime(v, now, tc);
  return v;
};

/**
 * The ambience beds, mixed by the listener's position and the time of day: calm river and riffle (crossfaded along
 * the river, panned toward the water), wind in the pines with slow gusts, morning birds and night crickets with
 * occasional spot calls (birds by day, an owl at night), and the campfire crackle and coffee pot as positional
 * sources at the fire.
 */
export class Ambience {
  readonly beds: Record<BedId, Bed>;
  /** current linear gains and the riffle's low-pass, for the debug panel */
  readonly level: Record<BedId, number> & { cutoff: number } =
    { riverCalm: 0, riverRiffle: 0, wind: 0, birds: 0, crickets: 0, campfire: 0, coffee: 0, cutoff: 0 };
  readonly day: DayMix = { birds: 0, crickets: 0, owl: 0, wind: 0 };
  fire = 1;
  coffee = true;
  private target: DayMix = { birds: 0, crickets: 0, owl: 0, wind: 0 };
  private em = new RiverEmitters();
  private m: RiverMix = { calm: 0, riffle: 0, calmPan: 0, rifflePan: 0 };
  private calmPan: StereoPannerNode;
  private rifflePan: StereoPannerNode;
  private riffleLp: BiquadFilterNode;
  private windLp: BiquadFilterNode;
  private last = -1;
  private nextBird = 0;
  private nextOwl = 0;
  private lastCalmPan = 0;
  private lastRifflePan = 0;
  private lastCut = 18000;
  private lastWindCut = 4000;

  constructor(private host: AmbienceHost, ctx: BaseAudioContext, out: AudioNode, fire: { x: number; y: number; z: number },
    hour: number) {
    const gain = (to: AudioNode) => {
      const g = ctx.createGain();
      g.gain.value = 0;
      g.connect(to);
      return g;
    };
    const lowpass = (f: number, to: AudioNode) => {
      const b = ctx.createBiquadFilter();
      b.type = 'lowpass';
      b.frequency.value = f;
      b.Q.value = 0.5;
      b.connect(to);
      return b;
    };
    const calmGain = gain(out);
    this.calmPan = ctx.createStereoPanner();
    this.calmPan.connect(calmGain);
    const riffleGain = gain(out);
    this.rifflePan = ctx.createStereoPanner();
    this.rifflePan.connect(riffleGain);
    this.riffleLp = lowpass(this.lastCut, this.rifflePan);
    const windGain = gain(out);
    this.windLp = lowpass(this.lastWindCut, windGain);
    const birdsGain = gain(out);
    const cricketsGain = gain(out);
    const merger = ctx.createChannelMerger(2);
    merger.connect(cricketsGain);
    const fireGain = gain(pointPanner(ctx, fire.x, fire.y, fire.z, 1.5, 1.0, out));
    const coffeeGain = gain(pointPanner(ctx, fire.x + 0.35, fire.y + 0.1, fire.z + 0.2, 0.6, 1.2, out));
    this.beds = {
      riverCalm: new Bed(ctx, calmGain, this.calmPan),
      riverRiffle: new Bed(ctx, riffleGain, this.riffleLp),
      wind: new Bed(ctx, windGain, this.windLp),
      birds: new Bed(ctx, birdsGain, birdsGain),
      crickets: new Bed(ctx, cricketsGain, merger, true),
      campfire: new Bed(ctx, fireGain, fireGain),
      coffee: new Bed(ctx, coffeeGain, coffeeGain),
    };
    dayMix(hour, this.target);
    Object.assign(this.day, this.target);
  }

  setHour(hour: number) {
    dayMix(hour, this.target);
    this.requestNeeded();
  }

  /** queue the beds the current time of day needs (each decodes once) */
  requestNeeded() {
    for (const id of bedsNeeded(this.target)) this.request(id);
  }

  request(id: BedId) {
    const bed = this.beds[id];
    if (bed.requested) return;
    bed.requested = true;
    void this.host.loadLoop(id).then((b) => (bed.buffer = b));
  }

  /** 10 Hz: focus position (the player, or the camera) and the camera's right vector in XZ */
  update(now: number, fx: number, fy: number, fz: number, rx: number, rz: number) {
    const dt = this.last < 0 ? 1 : clamp(now - this.last, 0, 1);
    this.last = now;
    // a time-of-day change crossfades over a few seconds
    const k = 1 - Math.exp(-dt / 1.5), d = this.day, t = this.target;
    d.birds += (t.birds - d.birds) * k;
    d.crickets += (t.crickets - d.crickets) * k;
    d.owl += (t.owl - d.owl) * k;
    d.wind += (t.wind - d.wind) * k;

    const m = this.m, L = this.level, B = this.beds, tc = 0.3;
    this.em.measure(fx, fy, fz, rx, rz, m);
    const riffle = clamp(Math.sqrt(m.riffle), 0, 1.5);
    L.riverCalm = G.riverCalm * clamp(Math.sqrt(m.calm), 0, 1.5);
    L.riverRiffle = G.riverRiffle * riffle;
    B.riverCalm.set(L.riverCalm, now, tc);
    B.riverRiffle.set(L.riverRiffle, now, tc);
    this.lastCalmPan = ramp(this.calmPan.pan, clamp(m.calmPan * 0.55, -0.55, 0.55), this.lastCalmPan, 0.02, now, tc);
    this.lastRifflePan = ramp(this.rifflePan.pan, clamp(m.rifflePan * 0.55, -0.55, 0.55), this.lastRifflePan, 0.02, now, tc);
    // a far riffle loses its hiss first
    L.cutoff = 2500 + 15500 * smoothstep(0.03, 0.5, riffle);
    this.lastCut = ramp(this.riffleLp.frequency, L.cutoff, this.lastCut, this.lastCut * 0.04, now, tc);

    const g = gust(now);
    L.wind = G.wind * d.wind * dbToGain(-4 + 6 * g);
    B.wind.set(L.wind, now, 0.6);
    this.lastWindCut = ramp(this.windLp.frequency, 1600 * Math.pow(5.5, g), this.lastWindCut, this.lastWindCut * 0.04, now, 0.6);

    L.birds = G.birds * d.birds;
    L.crickets = G.crickets * d.crickets;
    L.campfire = G.campfire * this.fire;
    L.coffee = this.coffee && this.fire > 0 ? G.coffee : 0;
    B.birds.set(L.birds, now, tc);
    B.crickets.set(L.crickets, now, tc);
    B.campfire.set(L.campfire, now, tc);
    B.coffee.set(L.coffee, now, tc);

    // spot calls: a bird in the trees every ~10-25 s by day (rarer at golden hour), an owl now and then at night
    if (d.birds > 0.05 && now >= this.nextBird) {
      if (this.nextBird > 0) this.spot('bird_call', fx, fy, fz, 14, 38, 5, 12, 0, 2 * Math.PI, 20 * Math.log10(Math.max(d.birds, 0.3)));
      this.nextBird = now + Math.min(150, (9 + 16 * Math.random()) / d.birds);
    }
    if (d.owl > 0.5 && now >= this.nextOwl) {
      // from the dense forest on the north bank
      if (this.nextOwl > 0) this.spot('owl', fx, fy, fz, 35, 70, 10, 16, 1.1 * Math.PI, 1.9 * Math.PI, 0);
      this.nextOwl = now + 25 + 45 * Math.random();
    }
  }

  private spot(id: GroupId, fx: number, fy: number, fz: number, r0: number, r1: number, h0: number, h1: number,
    a0: number, a1: number, gainDb: number) {
    const a = a0 + (a1 - a0) * Math.random(), r = r0 + (r1 - r0) * Math.random();
    this.host.play(id, fx + Math.cos(a) * r, fy + h0 + (h1 - h0) * Math.random(), fz + Math.sin(a) * r, gainDb);
  }
}
