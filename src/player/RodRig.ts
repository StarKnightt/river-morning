import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import type { Assets } from '../core/Assets';
import { ROD_GUIDE_PART, rodAxisU, upgradeMaterial } from './CharacterMaterials';
import { ROD_URL } from './CharacterSpec';

/** The Blender rod's build constants (prep/blender/scripts/build_rod.py): guide heights, grip end, tip. */
const GUIDES = [0.7, 1.0, 1.3, 1.6, 1.9, 2.15, 2.36];
const GRIP_TOP = 0.3;
const TIP_Y = 2.525;
/** hexagon circumradius of the cane blank at rod height y (13.6 mm across at the grip, 3 mm at the tip) */
const blankR = (y: number) => {
  const t = Math.min(1, Math.max(0, (y - GRIP_TOP) / (TIP_Y - GRIP_TOP)));
  return 0.0068 * (1 - t) ** 0.92 + 0.0015 * t;
};
/** reel spool centre in rod space (its axis runs along X) and plate radius */
const REEL_C = new THREE.Vector3(0, 0, 0.047);
const REEL_R = 0.034;

/** A wire tube along a polyline (closed for rings): 5 sides, normals from the tube's axis. */
function wireTube(pts: THREE.Vector3[], radius: number, closed: boolean, pos: number[], nrm: number[], idx: number[]) {
  const n = pts.length, sides = 5, base = pos.length / 3;
  const t = new THREE.Vector3(), a = new THREE.Vector3(), b = new THREE.Vector3(), d = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    const p0 = pts[closed ? (i - 1 + n) % n : Math.max(0, i - 1)], p1 = pts[closed ? (i + 1) % n : Math.min(n - 1, i + 1)];
    t.subVectors(p1, p0).normalize();
    a.set(0, 0, 1);
    if (Math.abs(t.dot(a)) > 0.9) a.set(1, 0, 0);
    a.addScaledVector(t, -a.dot(t)).normalize();
    b.crossVectors(t, a);
    for (let k = 0; k < sides; k++) {
      const ang = (k / sides) * Math.PI * 2;
      d.copy(a).multiplyScalar(Math.cos(ang)).addScaledVector(b, Math.sin(ang));
      pos.push(pts[i].x + d.x * radius, pts[i].y + d.y * radius, pts[i].z + d.z * radius);
      nrm.push(d.x, d.y, d.z);
    }
  }
  const segs = closed ? n : n - 1;
  for (let i = 0; i < segs; i++) {
    const i2 = (i + 1) % n;
    for (let k = 0; k < sides; k++) {
      const k2 = (k + 1) % sides;
      const a0 = base + i * sides + k, a1 = base + i * sides + k2, b0 = base + i2 * sides + k, b1 = base + i2 * sides + k2;
      idx.push(a0, b0, a1, a1, b0, b1);
    }
  }
}

/**
 * Standing ring guides on two-footed wire frames, stood off the blank on the reel side (+Z): the GLB's own rings
 * (1.3 mm wire facing along the rod) never showed, so these replace them, one per guide wrap, on that segment.
 */
function guideGeometry(k: number): THREE.BufferGeometry {
  const y = GUIDES[k], r = blankR(y);
  const ring = Math.max(0.0034, 0.0072 - 0.0006 * (k + 1));
  const wire = 0.0007;
  const cz = r + 0.0045 + ring;
  const pos: number[] = [], nrm: number[] = [], idx: number[] = [];
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i < 12; i++) {
    const a = (i / 12) * Math.PI * 2;
    pts.push(new THREE.Vector3(Math.cos(a) * ring, y, cz + Math.sin(a) * ring));
  }
  wireTube(pts, wire, true, pos, nrm, idx);
  for (const s of [-1, 1]) {
    // a foot flat under the wrap, then the leg up to the bottom of the ring
    wireTube([
      new THREE.Vector3(0, y + s * 0.015, r * 0.96), new THREE.Vector3(0, y + s * 0.009, r * 1.02),
      new THREE.Vector3(0, y + s * 0.004, r + 0.0025), new THREE.Vector3(0, y + s * 0.0012, cz - ring),
    ], wire, false, pos, nrm, idx);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array((pos.length / 3) * 2).fill(0.875), 2));
  g.setIndex(idx);
  return g;
}

/** Cork grip swell, metres up the rod from the reel seat: where a fist holds it. */
export const GRIP_Y = 0.13;
/** Total bend at setBend(1): a deep fighting arc, the tip pointing ~100 degrees off the butt. */
const MAX_BEND = 1.75;
/** the most a loaded rod turns its tip toward the line (rad) */
const MAX_ALIGN = 2.1;
/** where the line leaves the tip-top ring, in Rod_Seg7 space (rod space (0, 2.53, 0.003)) */
const TIP_LOCAL = new THREE.Vector3(0, 0.17, 0.003);
/** crank knob at the end of the handle arm, in Rod_ReelHandle space */
const KNOB_LOCAL = new THREE.Vector3(0.012, -0.026, 0);
/** rod-space height of each bend joint (Rod_Seg1..7): bend shares grow toward the tip */
const JOINT_Y = [0.7, 1.0, 1.3, 1.6, 1.9, 2.15, 2.36];
const SHARE = (() => {
  const w = JOINT_Y.map((y) => Math.pow(y / 2.53, 1.5));
  const s = w.reduce((a, b) => a + b, 0);
  return w.map((v) => v / s);
})();
/** passive spring: ~2.2 Hz split cane, lightly damped; gain maps the felt acceleration to bend (rad per m/s^2) */
const SPRING_K = (2 * Math.PI * 2.2) ** 2;
const SPRING_C = 2 * 0.12 * Math.sqrt(SPRING_K);
const INERTIA_GAIN = 3.2;
const MAX_PASSIVE = 1.0;
const GRAVITY = new THREE.Vector3(0, -9.81, 0);

function floatAttributes(src: THREE.BufferGeometry): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  for (const name of ['position', 'normal', 'uv']) {
    const a = src.getAttribute(name) as THREE.BufferAttribute | undefined;
    if (!a) continue;
    const out = new Float32Array(a.count * a.itemSize);
    for (let i = 0; i < a.count; i++) {
      out[i * a.itemSize] = a.getX(i);
      if (a.itemSize > 1) out[i * a.itemSize + 1] = a.getY(i);
      if (a.itemSize > 2) out[i * a.itemSize + 2] = a.getZ(i);
    }
    g.setAttribute(name, new THREE.BufferAttribute(out, a.itemSize));
  }
  g.setIndex(src.index ? Array.from(src.index.array) : null);
  return g;
}

/**
 * The Blender fishing rod (2.60 m split cane, click reel) as one skinned mesh: Rod_Seg0 (butt, reel seat, cork)
 * then 7 bend joints to Rod_Seg7 (tip), Rod_Reel > Rod_ReelHandle (crank). Rod space: +Y toward the tip from the
 * reel seat, reel and guides on +Z, crank on +X.
 *
 * Part 6 drives it: setPose() each frame (the player does this from the hand sockets), setBend() for the load of
 * the line, crank() / spinCrank() while reeling, tipWorld() (getTip()) for the line's first point. root, getTip,
 * setBend and crank are the part 6 lane's RodRig interface.
 */
export class RodRig {
  readonly mesh: THREE.SkinnedMesh;
  readonly segments: THREE.Bone[];
  readonly reel: THREE.Bone;
  readonly handle: THREE.Bone;
  private crankRad = 0;
  private bendAmount = 0;
  private readonly bendToward = new THREE.Vector3();
  private hasToward = false;
  // passive bend vector in rod-local XZ (radians of total bend) and its rate
  private bx = 0;
  private bz = 0;
  private vx = 0;
  private vz = 0;
  private readonly prevMid = new THREE.Vector3();
  private readonly prevVel = new THREE.Vector3();
  private readonly acc = new THREE.Vector3();
  private history = 0;
  /** 0..1: how much the rod lies on the ground along its length (laid down): its own sag and the line's load go */
  support = 0;
  /** lying on the ground (RodGround): straight, no sway, sag or line load */
  resting = false;

  private constructor(mesh: THREE.SkinnedMesh, segments: THREE.Bone[], reel: THREE.Bone, handle: THREE.Bone) {
    this.mesh = mesh;
    this.segments = segments;
    this.reel = reel;
    this.handle = handle;
  }

  static async load(assets: Assets): Promise<RodRig> {
    const gltf = await assets.gltf(ROD_URL);
    const scene = gltf.scene;
    scene.updateMatrixWorld(true);
    const node = (name: string) => {
      const o = scene.getObjectByName(name);
      if (!o) throw new Error(`rod: no node ${name}`);
      return o;
    };
    const names = [...Array.from({ length: 8 }, (_, i) => `Rod_Seg${i}`), 'Rod_Reel', 'Rod_ReelHandle'];
    const bones = names.map((n) => {
      const src = node(n);
      const b = new THREE.Bone();
      b.name = n;
      b.position.copy(src.position);
      b.quaternion.copy(src.quaternion);
      b.scale.copy(src.scale);
      return b;
    });
    for (let i = 1; i < 8; i++) bones[i - 1].add(bones[i]);
    bones[0].add(bones[8]);
    bones[8].add(bones[9]);

    let material: THREE.Material | null = null;
    const parts: THREE.BufferGeometry[] = [];
    const skin = (g: THREE.BufferGeometry, bi: number, part: number) => {
      const count = g.getAttribute('position').count;
      const idx = new Uint16Array(count * 4), wt = new Float32Array(count * 4);
      for (let i = 0; i < count; i++) {
        idx[i * 4] = bi;
        wt[i * 4] = 1;
      }
      g.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(idx, 4));
      g.setAttribute('skinWeight', new THREE.Float32BufferAttribute(wt, 4));
      // rod-space height, part, distance off the rod axis and the reel plates' rims, for the material
      const pa = g.getAttribute('position') as THREE.BufferAttribute;
      const na = g.getAttribute('normal') as THREE.BufferAttribute;
      const aRod = new Float32Array(count * 4);
      for (let i = 0; i < count; i++) {
        const x = pa.getX(i), y = pa.getY(i), z = pa.getZ(i);
        const r = Math.hypot(x, z);
        aRod[i * 4] = y;
        aRod[i * 4 + 1] = part;
        aRod[i * 4 + 2] = r;
        if (part === 8) aRod[i * 4 + 3] = Math.hypot(y - REEL_C.y, z - REEL_C.z) > REEL_R * 0.84 && Math.abs(x) > 0.007 ? 1 : 0;
        if (part >= 1 && part <= 7 && y < TIP_Y - 0.03) {
          if (r > blankR(y) * 1.5 + 0.0015) {
            // the GLB's own guide ring and legs: folded onto the axis (inside the blank)
            pa.setXYZ(i, 0, y, 0);
          } else if (y > 0.85 && r > 1e-5) {
            // smooth radial normals on the thin upper blank, so its sub-pixel widening opens no cracks
            na.setXYZ(i, x / r, 0, z / r);
          }
        }
      }
      g.setAttribute('aRod', new THREE.BufferAttribute(aRod, 4));
      parts.push(g);
    };
    names.forEach((n, bi) => {
      const mesh = node(`${n}_mesh`) as THREE.Mesh;
      material ??= upgradeMaterial(mesh.material as THREE.Material);
      const g = floatAttributes(mesh.geometry);
      g.applyMatrix4(mesh.matrixWorld);
      skin(g, bi, bi);
      // guide k sits at the base of segment k + 1
      if (bi >= 1 && bi <= 7) skin(guideGeometry(bi - 1), bi, ROD_GUIDE_PART);
    });
    const geometry = mergeGeometries(parts);
    if (!geometry) throw new Error('rod: merge failed');
    geometry.computeBoundingSphere();
    const mesh = new THREE.SkinnedMesh(geometry, material!);
    mesh.name = 'player-rod';
    mesh.add(bones[0]);
    mesh.updateMatrixWorld(true);
    mesh.bind(new THREE.Skeleton(bones));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    return new RodRig(mesh, bones.slice(0, 8), bones[8], bones[9]);
  }

  /** Rod origin (reel seat) and orientation (+Y tip, +Z reel side) in world space. */
  setPose(position: THREE.Vector3, quaternion: THREE.Quaternion) {
    this.mesh.position.copy(position);
    this.mesh.quaternion.copy(quaternion);
  }

  /**
   * Load from the line, 0..1 (1 = a deep fighting arc). `toward` (world direction, e.g. tip -> bobber) picks the
   * bend plane; null keeps the last plane; omitted, the rod bends toward its reel and guide side (down: the reel
   * hangs under the rod since round 2).
   */
  setBend(amount: number, toward?: THREE.Vector3 | null) {
    this.bendAmount = THREE.MathUtils.clamp(amount, 0, 1);
    if (toward === null) return;
    this.hasToward = !!toward;
    if (toward) this.bendToward.copy(toward);
  }

  /** The rod's root object (part 6's RodRig interface). */
  get root(): THREE.Object3D {
    return this.mesh;
  }

  /** part 6's RodRig interface: tip-top ring in world space. */
  getTip(out: THREE.Vector3): THREE.Vector3 {
    return this.tipWorld(out);
  }

  /** part 6's RodRig interface: spin the reel handle by signed turns (+ = winding in). */
  crank(turns: number) {
    this.crankRad += turns * Math.PI * 2;
  }

  get bend() {
    return this.bendAmount;
  }

  /** Crank angle about the spool axis (radians, + = retrieving). */
  setCrank(angle: number) {
    this.crankRad = angle;
  }

  spinCrank(delta: number) {
    this.crankRad += delta;
  }

  get crankAngle() {
    return this.crankRad;
  }

  /** Forget the motion history (after a teleport or a pose snap), so the spring does not see a huge jolt. */
  resetDynamics() {
    this.history = 0;
    this.vx = this.vz = 0;
    this.bx = this.bz = 0;
  }

  /** After setPose: passive sway and sag, the external bend and the crank onto the bones. */
  update(dt: number) {
    const m = this.mesh;
    m.updateMatrixWorld();
    if (this.resting) {
      this.resetDynamics();
      for (let i = 1; i < 8; i++) this.segments[i].quaternion.identity();
      this.handle.quaternion.setFromAxisAngle(_xAxis, this.crankRad);
      m.updateMatrixWorld(true);
      return;
    }
    // felt acceleration at mid-rod: gravity minus the support's acceleration, across the rod in rod space
    const mid = _v.set(0, 1.2, 0).applyMatrix4(m.matrixWorld);
    if (dt > 0 && this.history >= 2) {
      const vel = _v2.copy(mid).sub(this.prevMid).divideScalar(dt);
      const a = _v3.copy(vel).sub(this.prevVel).divideScalar(dt);
      if (a.length() > 80) a.setLength(80);
      this.acc.lerp(a, 1 - Math.exp(-dt * 30));
      this.prevVel.copy(vel);
    } else if (dt > 0) {
      this.prevVel.copy(_v2.copy(mid).sub(this.prevMid).divideScalar(dt));
      this.acc.set(0, 0, 0);
    }
    if (dt > 0) this.history++;
    this.prevMid.copy(mid);
    _q.copy(m.quaternion).invert();
    const f = _v2.copy(GRAVITY).sub(this.acc).applyQuaternion(_q);
    const free = 1 - this.support;
    const tx = (INERTIA_GAIN / SPRING_K) * f.x * free, tz = (INERTIA_GAIN / SPRING_K) * f.z * free;
    if (dt > 0) {
      const n = Math.ceil(dt * 120), h = dt / n;
      for (let i = 0; i < n; i++) {
        this.vx += (SPRING_K * (tx - this.bx) - SPRING_C * this.vx) * h;
        this.vz += (SPRING_K * (tz - this.bz) - SPRING_C * this.vz) * h;
        this.bx += this.vx * h;
        this.bz += this.vz * h;
      }
      const l = Math.hypot(this.bx, this.bz);
      if (l > MAX_PASSIVE) {
        this.bx *= MAX_PASSIVE / l;
        this.bz *= MAX_PASSIVE / l;
      }
    } else {
      this.bx = tx;
      this.bz = tz;
    }
    // external load from the line; a loaded rod's tip is dragged round until it points along the line
    let ex = 0, ez = 1;
    let bendTheta = this.bendAmount * MAX_BEND * free;
    if (this.hasToward) {
      const d = _v3.copy(this.bendToward).applyQuaternion(_q);
      const l = Math.hypot(d.x, d.z);
      if (l > 1e-4) {
        ex = d.x / l;
        ez = d.z / l;
        const k = THREE.MathUtils.smoothstep(this.bendAmount, 0.3, 0.75) * free;
        bendTheta += (Math.min(Math.atan2(l, d.y), MAX_ALIGN) - bendTheta) * k;
      }
    }
    const Bx = this.bx + ex * bendTheta;
    const Bz = this.bz + ez * bendTheta;
    const theta = Math.hypot(Bx, Bz);
    // rotating +Y toward (bx, 0, bz) is a rotation about (bz, 0, -bx); every joint shares the same axis, so the
    // segments stay in one plane
    if (theta > 1e-6) _axis.set(Bz / theta, 0, -Bx / theta);
    for (let i = 1; i < 8; i++) {
      if (theta > 1e-6) this.segments[i].quaternion.setFromAxisAngle(_axis, theta * SHARE[i - 1]);
      else this.segments[i].quaternion.identity();
    }
    this.handle.quaternion.setFromAxisAngle(_xAxis, this.crankRad);
    m.updateMatrixWorld(true);
    // the material's varnish streak runs along each segment's axis
    const axes = rodAxisU.array as THREE.Vector3[];
    for (let i = 0; i < 8; i++) axes[i].setFromMatrixColumn(this.segments[i].matrixWorld, 1).normalize();
  }

  /** Where the line leaves the tip-top ring (world). */
  tipWorld(out: THREE.Vector3): THREE.Vector3 {
    return out.copy(TIP_LOCAL).applyMatrix4(this.segments[7].matrixWorld);
  }

  /** Direction the tip points (world, unit). */
  tipDirection(out: THREE.Vector3): THREE.Vector3 {
    return out.setFromMatrixColumn(this.segments[7].matrixWorld, 1).normalize();
  }

  /** Crank knob (world), for the reeling hand. */
  knobWorld(out: THREE.Vector3): THREE.Vector3 {
    return out.copy(KNOB_LOCAL).applyMatrix4(this.handle.matrixWorld);
  }
}

const _v = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _axis = new THREE.Vector3(1, 0, 0);
const _xAxis = new THREE.Vector3(1, 0, 0);
