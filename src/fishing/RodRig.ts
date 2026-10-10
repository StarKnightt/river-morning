import * as THREE from 'three/webgpu';
import type { RodRig } from './types';

const _inv = new THREE.Quaternion(), _t = new THREE.Vector3(), _a = new THREE.Vector3();

/** share of the total bend carried by each joint, butt (Seg1) to tip (Seg7) */
const BEND_WEIGHTS = [0.04, 0.07, 0.1, 0.14, 0.18, 0.22, 0.25];
/** total tip deflection at bend 1 (rad): hooped under a big fish */
const MAX_BEND = 1.5;

/**
 * Bends a rod built as a chain of nested segment nodes (Blender rod: Rod_Seg0 > Rod_Seg1 > ... > Rod_Seg7, +Y along
 * the rod, rest rotations identity). Every joint turns about the same axis, perpendicular to the rod and to the
 * line, with angles growing toward the tip, so the curve stays in the plane of the pull. The reel handle
 * (Rod_ReelHandle) spins about its local X.
 */
export class SegmentRodRig implements RodRig {
  private bendAxis = new THREE.Vector3(1, 0, 0);
  private crankAngle = 0;

  constructor(
    readonly root: THREE.Object3D,
    private segs: THREE.Object3D[],
    private handle: THREE.Object3D | null,
    /** tip-top ring in the last segment's frame */
    private tipLocal: THREE.Vector3,
  ) {}

  /** wrap the Blender rod (fishing_rod.glb scene); the line leaves the tip-top at (0, 2.53, 0.003) in rod space */
  static fromGltf(scene: THREE.Object3D): SegmentRodRig | null {
    const segs: THREE.Object3D[] = [];
    for (let i = 0; i < 8; i++) {
      const s = scene.getObjectByName(`Rod_Seg${i}`);
      if (!s) return null;
      segs.push(s);
    }
    scene.updateMatrixWorld(true);
    const tipRod = new THREE.Vector3(0, 2.53, 0.003);
    // rod space = Rod_Seg0's frame; bring the tip into the last segment's frame
    const seg0Inv = new THREE.Matrix4().copy(segs[0].matrixWorld).invert();
    const lastInRod = new THREE.Matrix4().multiplyMatrices(seg0Inv, segs[7].matrixWorld);
    const tipLocal = tipRod.applyMatrix4(lastInRod.invert());
    scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) {
        m.castShadow = true;
        m.receiveShadow = true;
      }
    });
    return new SegmentRodRig(scene, segs, scene.getObjectByName('Rod_ReelHandle') ?? null, tipLocal);
  }

  getTip(out: THREE.Vector3): THREE.Vector3 {
    return this.segs[7].localToWorld(out.copy(this.tipLocal));
  }

  setBend(amount: number, towardWorld?: THREE.Vector3 | null) {
    if (towardWorld && towardWorld.lengthSq() > 1e-8) {
      // the line direction in rod space (Rod_Seg0 frame), then the axis Y x dir: rotating +Y about it leans it toward dir
      this.segs[0].getWorldQuaternion(_inv).invert();
      _t.copy(towardWorld).applyQuaternion(_inv);
      _a.set(_t.z, 0, -_t.x);
      if (_a.lengthSq() > 1e-6) this.bendAxis.copy(_a.normalize());
    }
    const total = amount * MAX_BEND;
    for (let i = 1; i < 8; i++) this.segs[i].quaternion.setFromAxisAngle(this.bendAxis, total * BEND_WEIGHTS[i - 1]);
  }

  crank(turns: number) {
    if (!this.handle || turns === 0) return;
    this.crankAngle = (this.crankAngle + turns * Math.PI * 2) % (Math.PI * 2);
    this.handle.quaternion.setFromAxisAngle(_t.set(1, 0, 0), this.crankAngle);
  }
}

/**
 * Procedural stand-in with the Blender rod's layout (used when fishing_rod.glb is not in public/assets yet):
 * split-cane blank tapering 7 -> 1.5 mm in 8 nested segments (joints at 0.70, 1.00, ... 2.36 m, tip 2.53 m),
 * cork grip, brass click reel with a crank. ~400 triangles, one material per part.
 */
export function buildProceduralRod(): SegmentRodRig {
  const cane = new THREE.MeshStandardNodeMaterial({ color: new THREE.Color(0.42, 0.26, 0.1), roughness: 0.45 });
  const cork = new THREE.MeshStandardNodeMaterial({ color: new THREE.Color(0.55, 0.42, 0.26), roughness: 0.8 });
  const brass = new THREE.MeshStandardNodeMaterial({ color: new THREE.Color(0.62, 0.45, 0.18), roughness: 0.35, metalness: 1 });
  const wrap = new THREE.MeshStandardNodeMaterial({ color: new THREE.Color(0.25, 0.04, 0.03), roughness: 0.6 });
  const joints = [0, 0.7, 1.0, 1.3, 1.6, 1.9, 2.15, 2.36, 2.53];
  const radius = (y: number) => 0.0068 - (0.0068 - 0.0015) * (y / 2.53);
  const segs: THREE.Object3D[] = [];
  let parent: THREE.Object3D | null = null;
  for (let i = 0; i < 8; i++) {
    const seg = new THREE.Object3D();
    seg.name = `Rod_Seg${i}`;
    seg.position.y = i === 0 ? 0 : joints[i] - joints[i - 1];
    const y0 = i === 0 ? -0.075 : joints[i], y1 = joints[i + 1];
    const g = new THREE.CylinderGeometry(radius(y1), radius(y0), y1 - y0, 6, 1, false);
    g.translate(0, (y1 + y0) / 2 - joints[i], 0);
    const blank = new THREE.Mesh(g, cane);
    seg.add(blank);
    if (i > 0) {
      // a silk wrap over every joint, like the guides on the Blender rod
      const w = new THREE.Mesh(new THREE.CylinderGeometry(radius(joints[i]) * 1.5, radius(joints[i]) * 1.5, 0.012, 6), wrap);
      seg.add(w);
    }
    if (parent) parent.add(seg);
    parent = seg;
    segs.push(seg);
  }
  const grip = new THREE.Mesh(new THREE.CylinderGeometry(0.0125, 0.0115, 0.24, 10), cork);
  grip.position.y = 0.17;
  segs[0].add(grip);
  const reel = new THREE.Object3D();
  reel.name = 'Rod_Reel';
  reel.position.set(0, 0, 0.032);
  const drum = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 0.022, 16), brass);
  drum.rotation.z = Math.PI / 2;
  reel.add(drum);
  const handle = new THREE.Object3D();
  handle.name = 'Rod_ReelHandle';
  handle.position.set(0.016, 0, 0);
  const arm = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.026, 0.005), brass);
  arm.position.set(0.004, -0.012, 0);
  const knob = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.012, 6), cork);
  knob.rotation.z = Math.PI / 2;
  knob.position.set(0.01, -0.024, 0);
  handle.add(arm, knob);
  reel.add(handle);
  segs[0].add(reel);
  segs[0].traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.isMesh) {
      m.castShadow = true;
      m.receiveShadow = true;
    }
  });
  return new SegmentRodRig(segs[0], segs, handle, new THREE.Vector3(0, joints[8] - joints[7], 0.003));
}

