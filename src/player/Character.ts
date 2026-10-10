import * as THREE from 'three/webgpu';
import type { Assets } from '../core/Assets';
import { timings } from '../world/World';
import { gearAlpha, strandMaterial, upgradeMaterial } from './CharacterMaterials';
import { annotateClothes, annotateGear, annotateSkin, beardPoints, buildStrands, restOf, strandMesh } from './CharacterDetail';
import type { N } from '../gfx/tsl';
import { ANGLER_URL, SOCKETS, UPPER_BODY, type ClipName } from './CharacterSpec';

const trackBone = (t: THREE.KeyframeTrack) => t.name.slice(0, t.name.lastIndexOf('.'));
const trackProp = (t: THREE.KeyframeTrack) => t.name.slice(t.name.lastIndexOf('.') + 1);

function isConstant(t: THREE.KeyframeTrack, rest: ArrayLike<number>, eps: number) {
  const v = t.values, n = t.getValueSize();
  for (let i = 0; i < v.length; i++) if (Math.abs(v[i] - rest[i % n]) > eps) return false;
  return true;
}

/**
 * Every exported clip keys translation, rotation and scale on all 53 joints. Scale never changes and only the
 * pelvis really translates (in-place clips), so the rest are dropped: 54 tracks per clip instead of 159.
 */
function cleanClip(clip: THREE.AnimationClip, bones: Map<string, THREE.Bone>): THREE.AnimationClip {
  const keep = clip.tracks.filter((t) => {
    const bone = bones.get(trackBone(t));
    const prop = trackProp(t);
    if (!bone) return false;
    if (prop === 'scale') return !isConstant(t, [1, 1, 1], 1e-4);
    if (prop === 'position') return bone.name === 'pelvis' || !isConstant(t, bone.position.toArray(), 1e-4);
    return true;
  });
  return new THREE.AnimationClip(clip.name, clip.duration, keep);
}

function subClip(clip: THREE.AnimationClip, name: string, upper: boolean): THREE.AnimationClip {
  return new THREE.AnimationClip(name, clip.duration, clip.tracks.filter((t) => UPPER_BODY.has(trackBone(t)) === upper));
}

/**
 * The angler: one skinned mesh (skin, clothes, gear atlas), 53 bones, 24 in-place clips. `object` is what the
 * controller moves and turns (origin between the soles, +Z forward).
 */
export class Character {
  readonly object = new THREE.Group();
  readonly mixer: THREE.AnimationMixer;
  readonly meshes: THREE.SkinnedMesh[] = [];
  readonly bones = new Map<string, THREE.Bone>();
  readonly clips = new Map<string, THREE.AnimationClip>();
  /** rest-pose (bind) world matrix of each bone with the character at the origin */
  readonly restWorld = new Map<string, THREE.Matrix4>();
  /** moustache, beard fringe and loose hairs, riding on the head bone */
  strands: THREE.Mesh | null = null;
  /** the gear atlas mesh (hat, boots, hair, beard, eyes) and its cut-out, for the self-shadow proxy */
  private gearMesh: THREE.SkinnedMesh | null = null;
  private gearCut: N | null = null;
  private readonly poseBones: THREE.Bone[] = [];
  private poseSnap = new Float32Array(0);

  private constructor(gltfScene: THREE.Object3D, animations: THREE.AnimationClip[]) {
    this.object.name = 'player-angler';
    this.object.add(gltfScene);
    gltfScene.traverse((o) => {
      const b = o as THREE.Bone;
      if (b.isBone) this.bones.set(b.name, b);
      const m = o as THREE.SkinnedMesh;
      if (m.isSkinnedMesh) {
        // an overhead cast leaves the bind-pose bounds; it is always near the camera anyway
        m.frustumCulled = false;
        m.castShadow = true;
        m.receiveShadow = true;
        this.meshes.push(m);
      }
    });
    this.object.updateMatrixWorld(true);
    for (const [name, b] of this.bones) this.restWorld.set(name, b.matrixWorld.clone());
    // rest-pose analysis (regions, creases, wear, bulk, brim) before the materials that read it
    const joints = {
      at: (n: string): [number, number, number] => {
        const p = new THREE.Vector3().setFromMatrixPosition(this.restWorld.get(n)!);
        return [p.x, p.y, p.z];
      },
    };
    const t0 = performance.now();
    const byName = (n: string) => this.meshes.find((m) => (m.material as THREE.Material).name === n) ?? null;
    const skinM = byName('Angler_Skin'), gearM = byName('Angler_Gear'), clothM = byName('Angler_Clothes');
    // rest data first: the beard shades the neck and collar, its cards read their depth over the skin
    const skinRest = skinM ? restOf(skinM) : null, gearRest = gearM ? restOf(gearM) : null;
    const beard = gearRest ? beardPoints(gearRest) : null;
    if (clothM) annotateClothes(clothM, joints, (clothM.material as THREE.MeshStandardMaterial).map, beard);
    if (skinM && skinRest) annotateSkin(skinM, joints, skinRest, beard);
    if (gearM && gearRest) annotateGear(gearM, gearRest, skinRest);
    if (gearM) {
      const map = (gearM.material as THREE.MeshStandardMaterial).map;
      this.gearMesh = gearM;
      this.gearCut = map ? gearAlpha(map) : null;
    }
    for (const m of this.meshes) m.material = upgradeMaterial(m.material as THREE.Material);
    if (skinRest && gearRest) {
      this.strands = strandMesh(buildStrands(skinRest, gearRest, this.restWorld.get('head')!), strandMaterial());
      this.bone('head').add(this.strands);
    }
    timings.anglerPrep = Math.round(performance.now() - t0);
    this.poseBones.push(...this.bones.values());
    this.poseSnap = new Float32Array(this.poseBones.length * 7);
    this.savePose();
    for (const raw of animations) {
      const c = cleanClip(raw, this.bones);
      this.clips.set(c.name, c);
    }
    // base (legs, pelvis, spine_01, neck, head) and upper-body halves for layering the rod arms over the walk
    for (const name of ['idle', 'walk_slow', 'walk', 'wait'] as const) {
      const c = this.clips.get(name)!;
      this.clips.set(`${name}#base`, subClip(c, `${name}#base`, false));
      this.clips.set(`${name}#upper`, subClip(c, `${name}#upper`, true));
    }
    this.mixer = new THREE.AnimationMixer(gltfScene);
  }

  static async load(assets: Assets): Promise<Character> {
    const gltf = await assets.gltf(ANGLER_URL);
    return new Character(gltf.scene, gltf.animations);
  }

  /**
   * The mixer only writes a channel whose value changed since its last write, so anything that edits bones after
   * it (foot IK, the crank hand, part 6's look-at) would stay and stack on a held pose. The animator restores the
   * last clean pose before each mixer update and saves the new one after it.
   */
  savePose() {
    const s = this.poseSnap;
    this.poseBones.forEach((b, i) => {
      b.position.toArray(s, i * 7);
      b.quaternion.toArray(s, i * 7 + 3);
    });
  }

  restorePose() {
    const s = this.poseSnap;
    this.poseBones.forEach((b, i) => {
      b.position.fromArray(s, i * 7);
      b.quaternion.fromArray(s, i * 7 + 3);
    });
  }

  bone(name: string): THREE.Bone {
    const b = this.bones.get(name);
    if (!b) throw new Error(`angler: no bone ${name}`);
    return b;
  }

  clip(name: ClipName | `${'idle' | 'walk_slow' | 'walk' | 'wait'}#${'base' | 'upper'}`): THREE.AnimationClip {
    const c = this.clips.get(name);
    if (!c) throw new Error(`angler: no clip ${name}`);
    return c;
  }

  /** World position of a hand's grip point (needs current bone matrices). */
  palm(side: 'l' | 'r', out: THREE.Vector3): THREE.Vector3 {
    const s = side === 'r' ? SOCKETS.gripR : SOCKETS.gripL;
    return out.set(s.palm[0], s.palm[1], s.palm[2]).applyMatrix4(this.bone(s.bone).matrixWorld);
  }

  /** hold_fish socket: fish centre and orientation (+Z = toward the fish's head) in world space. */
  fishSocket(outPos: THREE.Vector3, outQuat: THREE.Quaternion) {
    const s = SOCKETS.fish;
    const hand = this.bone(s.bone);
    outPos.set(s.centre[0], s.centre[1], s.centre[2]).applyMatrix4(hand.matrixWorld);
    hand.getWorldQuaternion(outQuat);
    _v.set(s.headAxis[0], s.headAxis[1], s.headAxis[2]).applyQuaternion(outQuat).normalize();
    // fish nose +Z along the head axis, back (+Y) as close to world up as it gets
    _m.lookAt(_v, _zero, _up);
    outQuat.setFromRotationMatrix(_m);
  }

  setVisible(v: boolean) {
    this.object.visible = v;
  }

  /** Cut-out node for one of his meshes in the self-shadow map (the gear atlas' hair and beard cards), else null. */
  shadowCut(m: THREE.SkinnedMesh): N | null {
    return m === this.gearMesh ? this.gearCut : null;
  }
}

const _v = new THREE.Vector3();
const _m = new THREE.Matrix4();
const _zero = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
