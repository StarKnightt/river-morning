import * as THREE from 'three/webgpu';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { hash3, type Assembler, type GroundY } from './assemble';
import { Acc } from './firewood';
import { addStone, pick, STONE_TINTS, type StoneId, type StoneScans } from './stones';

/**
 * The fire itself (critic r2: "the logs read as smooth, dull grey tubes, the same tone from the middle of the fire
 * out to their clean, pale, sawn outer ends"; "the ash bed a flat, clean, near-white disc with a crisp round edge,
 * like a paper plate"; "the ring stones smooth loaves set apart with wide gaps, sitting on top of the ground").
 * Five logs pushed in from the ring like a star, their inner ends burnt narrow and round and their outer ends still
 * in bark, two of them resting up on the ring stones; half-burnt ends and charcoal lying in a heap of grey ash that
 * banks up against the wood and spills out between the stones; a tight wall of stones sunk half into the dirt.
 * The logs carry log-space uvs (u: metres from the char front / 0.5, negative inside the fire; v: metres round the
 * log / 0.5): the fire program cracks the charred part into blocks in that frame (materials.ts `fire`).
 */

export interface LogSpec {
  /** burning inner tip and the outer end (world) */
  tip: THREE.Vector3;
  end: THREE.Vector3;
  /** radius at the outer end (m) */
  R: number;
  /** charred length from the tip (m) */
  burn: number;
  seed: number;
}

const _a = new THREE.Vector3(), _u = new THREE.Vector3(), _v = new THREE.Vector3(), _p = new THREE.Vector3(), _n = new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);

/**
 * One burning log as a tube from its tip to its sawn outer end: narrowed to under half its girth and rounded where
 * it burns, the char roughened, the outer end a slightly tilted sawn disc (`ends`, end-grain mapping).
 */
export function burningLog(tube: Acc, ends: Acc, o: LogSpec) {
  const h = (k: number) => hash3(o.seed, k, 6.1);
  const L = o.tip.distanceTo(o.end);
  _a.subVectors(o.end, o.tip).normalize();
  _u.crossVectors(_a, Math.abs(_a.y) < 0.95 ? UP : new THREE.Vector3(1, 0, 0)).normalize();
  _v.crossVectors(_a, _u).normalize();
  const A = _a.clone(), U = _u.clone(), V = _v.clone();
  const SIDES = 12;
  const rings: number[] = [];
  for (let s = 0.004; s < L - 0.02; s += Math.min(0.06, 0.008 + s * 0.28)) rings.push(s);
  rings.push(L);
  const ph = [h(1) * 6.28, h(2) * 6.28];
  const rad = (s: number, th: number, k: number) => {
    const t = Math.min(1, s / o.burn);
    // burnt narrow toward the tip, rounded off over its last few centimetres
    let r = o.R * (s < o.burn ? 0.42 + 0.58 * Math.pow(t * t * (3 - 2 * t), 0.75) : 1);
    r *= Math.sqrt(Math.min(1, s / 0.032));
    r *= 1 + 0.05 * Math.sin(3 * th + ph[0]) + 0.03 * Math.sin(5 * th + ph[1]) + 0.025 * Math.sin(s * 23 + th);
    // char is rough and lumpy, bark a little less so
    r *= 1 + (s < o.burn ? 0.07 : 0.02) * (hash3(o.seed, k, Math.round(th * 10)) - 0.5);
    return r;
  };
  const tint: [number, number, number] = [1, 1, 1];
  const base = tube.count;
  // the tip: one vertex, pushed a little past the first ring
  _p.copy(o.tip).addScaledVector(A, -0.004);
  tube.v(_p, _n.copy(A).negate(), (0 - o.burn) / 0.5, 0, tint);
  for (let k = 0; k < rings.length; k++) {
    const s = rings[k];
    const s2 = rings[Math.min(rings.length - 1, k + 1)], s1 = rings[Math.max(0, k - 1)];
    for (let j = 0; j <= SIDES; j++) {
      const th = (j / SIDES) * Math.PI * 2;
      const c = Math.cos(th), sn = Math.sin(th);
      const r = rad(s, th % 6.2831, k);
      const dr = (rad(s2, th % 6.2831, k + 1) - rad(s1, th % 6.2831, Math.max(0, k - 1))) / Math.max(1e-3, s2 - s1);
      _p.copy(o.tip).addScaledVector(A, s).addScaledVector(U, c * r).addScaledVector(V, sn * r);
      _n.copy(U).multiplyScalar(c).addScaledVector(V, sn).addScaledVector(A, -dr).normalize();
      tube.v(_p, _n, (s - o.burn) / 0.5, (th * o.R) / 0.5, tint);
    }
  }
  const ring = (k: number, j: number) => base + 1 + k * (SIDES + 1) + j;
  for (let j = 0; j < SIDES; j++) tube.idx.push(base, ring(0, j + 1), ring(0, j));
  for (let k = 0; k < rings.length - 1; k++) {
    for (let j = 0; j < SIDES; j++) {
      const a = ring(k, j), b = ring(k, j + 1), c = ring(k + 1, j), d = ring(k + 1, j + 1);
      tube.idx.push(a, b, d, a, d, c);
    }
  }
  // the sawn outer end, tilted a little; the end-grain disc centred on the pith, its rim dark
  const tx = (h(3) - 0.5) * 0.3, ty = (h(4) - 0.5) * 0.3;
  _n.copy(A).addScaledVector(U, -tx).addScaledVector(V, -ty).normalize();
  const rot = h(5) * 6.28;
  const kk = 0.44 / o.R;
  const disc = (x: number, y: number) => [0.5 + (x * Math.cos(rot) - y * Math.sin(rot)) * kk, 0.5 + (x * Math.sin(rot) + y * Math.cos(rot)) * kk];
  const e0 = ends.count;
  _p.copy(o.end).addScaledVector(A, 0.002);
  const d0 = disc(0, 0);
  ends.v(_p, _n, d0[0], d0[1], [1, 1, 1]);
  for (let j = 0; j < SIDES; j++) {
    const th = (j / SIDES) * Math.PI * 2;
    const r = rad(L, th, rings.length - 1);
    const x = Math.cos(th) * r, y = Math.sin(th) * r;
    _p.copy(o.end).addScaledVector(U, x).addScaledVector(V, y).addScaledVector(A, tx * x + ty * y);
    const d = disc(x, y);
    ends.v(_p, _n, d[0], d[1], [0.55, 0.5, 0.46]);
  }
  for (let j = 0; j < SIDES; j++) ends.idx.push(e0, e0 + 1 + j, e0 + 1 + ((j + 1) % SIDES));
}

export interface FireBuild {
  /** points along the burning wood where flames stand (world), with a size factor each */
  flames: Array<{ p: THREE.Vector3; size: number }>;
  stones: number;
  /** spheres (x, y, z, radius; world) standing for the ring stones and the logs in the fire light's shadow test */
  occluders: THREE.Vector4[];
}

/**
 * Lay the fire into the assembler's 'campfire' group, centred on (cx, cz) on the rendered ground.
 * `src`: the campfire GLB's Bark, EndGrain, Charred and Ash materials (their maps).
 */
export function buildFire(asm: Assembler, scans: StoneScans, groundY: GroundY, cx: number, cz: number,
  src: { bark: THREE.MeshStandardMaterial; end: THREE.MeshStandardMaterial; charred: THREE.MeshStandardMaterial; ash: THREE.MeshStandardMaterial }): FireBuild {
  const g0 = groundY(cx, cz);
  const RING = 0.57;
  // ---- the ring: a tight wall of stones, broken rock more than river cobbles, sunk half into the dirt
  const IDS: StoneId[] = ['rock_09', 'namaqualand_boulder_05', 'rock_07', 'rock_09', 'stone_01', 'namaqualand_boulder_05', 'rock_09', 'rock_07'];
  // greys, a dark one, a pale one: fire-blackened and ash-dusted, not the buff and rust of the bank stones
  const RING_TINTS = [STONE_TINTS[0], STONE_TINTS[3], STONE_TINTS[4], STONE_TINTS[0], STONE_TINTS[5]];
  let stones = 0;
  const occluders: THREE.Vector4[] = [];
  const ringTop: Array<{ a: number; top: number }> = [];
  {
    let th = 0.3, k = 0;
    while (th < 0.3 + Math.PI * 2 - 0.1 && k < 28) {
      const r = (j: number) => hash3(k, j, 43);
      // (scene fix: 12-32 cm, round 4's 16-29 cm read as even lumps)
      const L = 0.12 + 0.2 * r(1) ** 1.4;
      const a = th + L / (2 * RING);
      const id = IDS[Math.floor(r(2) * IDS.length) % IDS.length];
      const sc = scans[id].size;
      const squash = 0.62 + 0.62 * r(3);
      const H = sc.y * (L / Math.max(sc.x, sc.z)) * squash;
      const rr = RING + (r(4) - 0.5) * 0.05;
      const x = cx + Math.cos(a) * rr, z = cz + Math.sin(a) * rr;
      const sink = H * (0.42 + 0.14 * r(5));
      addStone(asm, scans, {
        id, x, z, y: groundY(x, z) - sink, size: [L, squash, 0], aspect: true,
        yaw: -a + Math.PI / 2 + (r(6) - 0.5) * 0.5,
        tilt: 0.05 + 0.12 * r(7), tiltDir: a + (r(8) < 0.6 ? Math.PI : 0), tint: pick(RING_TINTS, r(9)),
        soot: { x: cx, z: cz, k: 0.8 + 0.2 * r(10) },
        crack: r(11) < 0.25 ? [Math.cos(a + 1.2), 0.3, Math.sin(a + 1.2)] : undefined, roll: r(12) < 0.5 ? Math.PI : 0,
        as: `Ring_${id}`,
      }, 'campfire');
      ringTop.push({ a, top: groundY(x, z) - sink + H });
      // a sphere sunk into the ground so its cap stands as high and as wide as the stone does
      const vis = Math.max(0.025, H - sink), rs = Math.max(Math.min(L * 0.5, 0.16), vis);
      occluders.push(new THREE.Vector4(x, groundY(x, z) + vis - rs, z, rs));
      // the next one butts up against it (a few overlap, a few leave a crack)
      th += (L / RING) * (0.86 + 0.12 * r(13));
      k++;
      stones++;
    }
  }
  const topAt = (a: number) => {
    let best = ringTop[0], bd = Infinity;
    for (const t of ringTop) {
      const d = Math.abs(Math.atan2(Math.sin(t.a - a), Math.cos(t.a - a)));
      if (d < bd) [best, bd] = [t, d];
    }
    return best.top;
  };

  // ---- five logs pushed in from the ring: inner ends in the coals, two resting up on the stones
  const tube = new Acc(), endA = new Acc(), stubT = new Acc(), stubE = new Acc();
  const logs: LogSpec[] = [];
  const flames: FireBuild['flames'] = [];
  for (let i = 0; i < 5; i++) {
    const h = (k: number) => hash3(i, k, 77);
    const a = 0.5 + (i / 5) * Math.PI * 2 + (h(1) - 0.5) * 0.55;
    const tipR = 0.05 + 0.06 * h(2);
    const onRing = i === 1 || i === 3;
    const endR = onRing ? RING + 0.06 + 0.06 * h(3) : 0.4 + 0.08 * h(3);
    const R = 0.04 + 0.022 * h(4);
    const ta = a + (h(5) - 0.5) * 0.3;
    const tip = new THREE.Vector3(cx + Math.cos(ta) * tipR, g0 + 0.055 + 0.03 * h(6), cz + Math.sin(ta) * tipR);
    const ex = cx + Math.cos(a) * endR, ez = cz + Math.sin(a) * endR;
    const ey = onRing ? topAt(a) + R * 0.8 : groundY(ex, ez) + R * 0.85 + 0.012;
    const end = new THREE.Vector3(ex, ey, ez);
    // (scene fix wip: charred most of their length, as in the reference; at 17-27 cm the outer bark read as rope)
    const burn = Math.min(0.25 + 0.12 * h(7), tip.distanceTo(end) - 0.08);
    const spec = { tip, end, R, burn, seed: 300 + i };
    logs.push(spec);
    // the outer two thirds of each log (the inner ends lie in the flames)
    for (const f of [0.4, 0.7, 0.97]) {
      const q = tip.clone().lerp(end, f);
      occluders.push(new THREE.Vector4(q.x, q.y, q.z, R * 1.1));
    }
    burningLog(tube, endA, spec);
    // a lick along the top of the burning end of some logs (scene fix: round 4's two flames per log, one at its tip,
    // stood in a row like a gas burner)
    const dir = end.clone().sub(tip).normalize();
    if (h(10) < 0.55) flames.push({ p: tip.clone().addScaledVector(dir, burn * (0.25 + 0.4 * h(11))).add(new THREE.Vector3(0, R * 0.6, 0)), size: 0.3 + 0.45 * h(8) ** 2 });
  }
  // flames come out of the gaps between the logs, where the coals are hottest and the gas gathers: one or two big
  // ones near the middle, smaller ones of every size further out, a few tiny licks over the coals
  {
    const ang = logs.map((L) => Math.atan2(L.end.z - cz, L.end.x - cx)).sort((a, b) => a - b);
    ang.forEach((a0, i) => {
      let a1 = ang[(i + 1) % ang.length];
      if (a1 <= a0) a1 += Math.PI * 2;
      const n = 1 + Math.floor(hash3(i, 1, 58) * 2.6);
      for (let j = 0; j < n; j++) {
        const hh = (k: number) => hash3(i * 7 + j, k, 59);
        const a = a0 + (a1 - a0) * (0.3 + 0.4 * hh(1));
        const r = 0.08 + 0.15 * hh(2);
        const big = Math.max(0, 1 - r / 0.26) * (j === 0 ? 1 : 0.6);
        flames.push({ p: new THREE.Vector3(cx + Math.cos(a) * r, g0 + 0.06 + 0.035 * big, cz + Math.sin(a) * r), size: 0.28 + 1.25 * big * (0.45 + 0.55 * hh(3)) + 0.25 * hh(4) ** 3 });
      }
    });
  }
  // half-burnt ends and sticks lying in the ash
  for (let i = 0; i < 5; i++) {
    const h = (k: number) => hash3(i, k, 91);
    const a = h(1) * 6.28, r = 0.16 + 0.24 * h(2);
    const L = 0.07 + 0.09 * h(3), R = 0.014 + 0.016 * h(4);
    const x = cx + Math.cos(a) * r, z = cz + Math.sin(a) * r;
    const yaw = h(5) * 6.28;
    const dx = Math.cos(yaw) * L / 2, dz = Math.sin(yaw) * L / 2;
    const y = groundY(x, z) + 0.025 + R * 0.6;
    burningLog(stubT, stubE, { tip: new THREE.Vector3(x - dx, y, z - dz), end: new THREE.Vector3(x + dx, y + 0.008, z + dz), R, burn: L * (0.7 + 0.5 * h(6)), seed: 400 + i });
  }
  tube.addTo(asm, src.bark, 'campfire', 'CharLog', 'fire-logs', false, false);
  endA.addTo(asm, src.end, 'campfire', 'CharEnd', 'fire-log-ends', false, false);
  stubT.addTo(asm, src.bark, 'campfire', 'CharLog', 'fire-stubs', true, false);
  stubE.addTo(asm, src.end, 'campfire', 'CharEnd', 'fire-stub-ends', true, false);

  // ---- the ash bed: grey powder heaped in the middle and against the wood, banked at the stones, spilling out
  // between them in a few tongues; its rim dips under the dirt (the ground's own ash takes over outside)
  {
    // (scene fix, critic polish r1: "a flat dark slate-grey disc like a rubber mat": a higher, lumpier heap)
    const RINGS = 16, SEG = 72;
    const lob = [hash3(1, 2, 3) * 6.28, hash3(2, 3, 4) * 6.28, hash3(3, 4, 5) * 6.28];
    const outR = (a: number) => {
      let r = 0.5 + 0.045 * Math.sin(3 * a + 1.1) + 0.035 * Math.sin(5 * a + 2.3) + 0.02 * Math.sin(9 * a + 0.4);
      for (const l of lob) r += 0.2 * Math.max(0, Math.cos(a - l)) ** 18;
      return r;
    };
    const logBank = (x: number, z: number) => {
      let b = 0;
      for (const L of logs) {
        const ax = L.end.x - L.tip.x, az = L.end.z - L.tip.z, len = Math.hypot(ax, az);
        const t = Math.max(0, Math.min(1, ((x - L.tip.x) * ax + (z - L.tip.z) * az) / (len * len)));
        const d = Math.hypot(x - L.tip.x - ax * t, z - L.tip.z - az * t);
        const w = L.R + 0.05;
        if (d < w) b = Math.max(b, 0.05 * (1 - d / w) ** 1.3 * (t * len < L.burn + 0.05 ? 1 : 0.4));
      }
      return b;
    };
    const pos: number[] = [], uvs: number[] = [], idx: number[] = [];
    for (let i = 0; i <= RINGS; i++) {
      for (let j = 0; j < (i === 0 ? 1 : SEG); j++) {
        const a = (j / SEG) * Math.PI * 2;
        const R = outR(a);
        const r = (i / RINGS) * R;
        const x = cx + Math.cos(a) * r, z = cz + Math.sin(a) * r;
        // a heap in the middle, off-centre, mounded up against the wood; drifts and lumps over it
        const hr = Math.hypot(x - cx - 0.04, z - cz + 0.03);
        const heap = 0.07 * Math.max(0, 1 - (hr / 0.4) ** 2) ** 1.2;
        const bank = 0.034 * Math.exp(-(((r - 0.47) / 0.055) ** 2)) * (r < RING ? 1 : 0.3);
        const bump = 0.016 * Math.sin(x * 29 + z * 9) * Math.sin(z * 23 - x * 7) + 0.01 * Math.sin(x * 61 - z * 37) * Math.sin(z * 53 + x * 19)
          + 0.008 * (hash3(Math.round(x * 40), Math.round(z * 40), 5) - 0.5);
        // beyond the ring only a thin skin of ash over the dirt
        const thin = r > RING - 0.02 ? 0.35 : 1;
        const sink = i === RINGS ? -0.025 : i === RINGS - 1 ? 0.0 : 0.006;
        pos.push(x, groundY(x, z) + sink + (heap + bank + logBank(x, z) + bump) * thin, z);
        uvs.push(x * 1.8, z * 1.8);
      }
    }
    const rs = (i: number) => (i === 0 ? 0 : 1 + (i - 1) * SEG);
    for (let j = 0; j < SEG; j++) idx.push(0, rs(1) + ((j + 1) % SEG), rs(1) + j);
    for (let i = 1; i < RINGS; i++) {
      for (let j = 0; j < SEG; j++) {
        const a0 = rs(i) + j, a1 = rs(i) + ((j + 1) % SEG), b0 = rs(i + 1) + j, b1 = rs(i + 1) + ((j + 1) % SEG);
        idx.push(a0, a1, b1, a0, b1, b0);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    asm.addGeometry(g, src.ash, { group: 'campfire', groundDirt: false, remap: () => 'AshBed' }, 'ash-bed');
  }

  // ---- charcoal: chunks all through the ash, thickest round the logs' burnt ends, a few kicked out of the ring
  for (let k = 0; k < 96; k++) {
    const out = k >= 84;
    const r = out ? RING + 0.05 + 0.5 * hash3(k, 1, 7) : 0.46 * Math.pow(hash3(k, 1, 7), 0.7);
    const a = hash3(k, 2, 7) * 6.283;
    const x = cx + Math.cos(a) * r, z = cz + Math.sin(a) * r;
    const s = (out ? 0.01 : 0.012) + 0.034 * Math.pow(hash3(k, 3, 7), 1.5);
    let geo: THREE.BufferGeometry = new THREE.IcosahedronGeometry(1, 0);
    geo.deleteAttribute('normal');
    geo = mergeVertices(geo);
    const pa = geo.getAttribute('position') as THREE.BufferAttribute;
    for (let i = 0; i < pa.count; i++) {
      const f = 0.65 + 0.7 * hash3(pa.getX(i) * 7 + k, pa.getY(i) * 7, pa.getZ(i) * 7);
      pa.setXYZ(i, pa.getX(i) * f, pa.getY(i) * f, pa.getZ(i) * f);
    }
    geo = geo.toNonIndexed();
    const lift = out ? 0.0 : 0.06 * Math.max(0, 1 - (r / 0.4) ** 2) ** 1.2;
    geo.applyMatrix4(new THREE.Matrix4().compose(
      new THREE.Vector3(x, groundY(x, z) + lift + s * 0.2, z),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(hash3(k, 4, 7) * 6.28, hash3(k, 5, 7) * 6.28, 0)),
      new THREE.Vector3(s * (1.1 + 0.9 * hash3(k, 6, 7)), s * 0.65, s)));
    geo.computeVertexNormals();
    asm.addGeometry(geo, src.charred, { group: 'campfire', small: true, groundDirt: false, remap: () => 'CharLump' }, 'charcoal');
  }
  // small licks of flame in the coals between the logs
  for (let k = 0; k < 5; k++) {
    const a = hash3(k, 9, 9) * 6.28, r = 0.06 + 0.12 * hash3(k, 10, 9);
    flames.push({ p: new THREE.Vector3(cx + Math.cos(a) * r, g0 + 0.065, cz + Math.sin(a) * r), size: 0.16 + 0.2 * hash3(k, 11, 9) });
  }
  return { flames, stones, occluders };
}
