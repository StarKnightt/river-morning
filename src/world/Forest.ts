import * as THREE from 'three/webgpu';
import {
  attribute, dFdx, dFdy, dot, float, length, log2, max, mix, normalMap, normalViewGeometry, positionLocal, positionWorld,
  sin, smoothstep, texture, uniform, uv, vec2, vec3, vec4, type N,
} from '../gfx/tsl';
import type { Assets } from '../core/Assets';
import { LAYER_FOLIAGE, LAYER_MIRROR, LAYER_SHADOW } from '../core/layers';
import { parseGlb } from './glb';
import { castsShadow, keepAt, SHADOW_FOCUS, TREE_VARIANTS, type TreeInst } from './ForestLayout';
import { POOL_LEVEL } from './River';
import { instanceCapacity } from '../core/instancing';
import { rootFlareGeometry } from './RootFlare';
import { tnoise } from '../gfx/noiseTex';

/**
 * The conifers: 15 variants x 3 LODs (ez-tree skeletons, Poly Haven twig/bark maps): the tree prep lane's 10, the
 * forest polish's young thicket trees and bark-only snags.
 * One InstancedMesh per variant x LOD x part (bark, needles), repacked on the CPU whenever the camera moves:
 * per-tree frustum test (the main view, or its mirror image for the pool reflection), LOD by distance with
 * hysteresis, far trees thinned by rank, front to back. Shadows come from a static set of LOD2 casters on a
 * shadow-only layer (the near stand only); the visible trees cast none themselves.
 */

/**
 * LOD0 up to 40 m, LOD1 to 85 m, LOD2 to 360 m (thinned beyond 150 m), nothing beyond: the terrain's canopy tone.
 * (LOD1 to 120 m as planned cost ~0.5 ms more for no visible gain past 85 m at 1080p.)
 */
export const LOD_DIST = [40, 85, 360];

export const windU = {
  time: uniform(0),
  /** world XZ direction the wind blows toward (normalised) */
  dir: uniform(new THREE.Vector2(0.8, 0.6).normalize()),
  /** a quiet morning */
  strength: uniform(0.25),
};

/**
 * Wind sway in world space (the forest meshes sit at the origin, so positionLocal is already instanced):
 * the trunk bends with height², branches bob with their sway weight, needle cards flutter at their tips.
 * The gust phase travels with the position, so neighbouring crowns move a little out of step.
 */
function windPosition(): N {
  const w = attribute('wind', 'vec4');
  const p = positionLocal;
  const t = windU.time;
  const ph0 = dot(p.xz, vec2(0.071, 0.053));
  const gust = sin(t.mul(0.37).add(ph0)).mul(0.55).add(sin(t.mul(0.93).add(ph0.mul(1.7))).mul(0.2)).add(0.45);
  const trunk = w.w.mul(w.w).mul(windU.strength).mul(0.45).mul(gust);
  const ph = w.z.mul(6.2832).add(ph0);
  const branch = w.x.mul(windU.strength).mul(0.22).mul(sin(t.mul(1.25).add(ph)));
  const flutter = w.y.mul(windU.strength).mul(0.06).mul(sin(t.mul(4.3).add(ph.mul(3.1)).add(p.y.mul(1.7))));
  const d = vec3(windU.dir.x, 0, windU.dir.y);
  return p.add(d.mul(trunk.add(branch.mul(0.6)).add(flutter.mul(0.6)))).add(vec3(0, branch.mul(-0.5).add(flutter), 0));
}

export const forestU = {
  /** brightness of the needles (calibration) */
  needleGain: uniform(1.0),
  barkGain: uniform(1.0),
  /** translucency of sunlit needles seen against the sun */
  translucency: uniform(0.9),
  /** least roughness of the needle cards (part 8: rougher at night, where a backlit moon frosted their edges white) */
  needleRough: uniform(0.72),
};

/**
 * Beyond the shadow map (and the casters, which are only the near stand) a crown gets no shadow from its
 * neighbours: its lower part is shaded as in a closed stand instead.
 */
function farCrownShade(): N {
  const f = smoothstep(48.0, 62.0, length(positionWorld.xz.sub(vec2(SHADOW_FOCUS[0], SHADOW_FOCUS[1]))));
  const hf = attribute('wind', 'vec4').w;
  return mix(float(1), mix(float(0.45), float(1.0), hf.mul(hf)), f);
}

/** coverage-preserving alpha: the mip-averaged alpha of thin needles falls under the cutoff with distance */
function needleAlpha(c: N): N {
  const st = uv().mul(2048);
  const mip = max(log2(max(length(dFdx(st)), length(dFdy(st)))), 0.0);
  return c.a.mul(mip.mul(0.25).add(1));
}

/** one wind node shared by the depth and colour passes, so both compute bit-identical positions */
let windNode: N = null;
const sharedWind = () => (windNode ??= windPosition());

/**
 * Depth-only twin of the needles, drawn first (renderOrder -1): alpha-tested cards overlap several layers deep,
 * and with this the lit needle shader runs about once per pixel instead of once per layer.
 */
function needleDepthMaterial(tex: { c: THREE.Texture }) {
  const m = new THREE.MeshBasicNodeMaterial();
  m.name = 'needles-depth';
  m.side = THREE.DoubleSide;
  m.alphaTest = 0.4;
  m.colorWrite = false;
  m.fog = false;
  m.colorNode = vec4(0, 0, 0, needleAlpha(texture(tex.c, uv())));
  m.positionNode = sharedWind();
  return m;
}

/** `twinned`: a depth twin already wrote the nearest needle, so this one tests against it and writes nothing */
function needleMaterial(tex: { c: THREE.Texture; m: THREE.Texture }, twinned: boolean) {
  const m = new THREE.MeshSSSNodeMaterial();
  m.name = twinned ? 'needles' : 'needles-solo';
  m.side = THREE.DoubleSide;
  // the shadow pass copies alphaTest and reads colorNode.a, so shadows get the needle holes too
  m.alphaTest = 0.4;
  m.depthWrite = !twinned;
  // a needle card is a weak, rough reflector: at full dielectric strength the sky sheen at grazing angles frosted
  // every crown edge pale grey
  m.specularIntensityNode = float(0.2);
  const t = uv();
  const c = texture(tex.c, t);
  const orm = texture(tex.m, t);
  const vis = attribute('color', 'vec4').x;
  const tint = attribute('aTint', 'vec4');
  const alpha = needleAlpha(c);
  // the crown's inside is much darker than its sunlit shell: a low sun carves conifers into light and dark
  // (scene fix: 0.32 -> 0.4, the stand's depth reads in the shade instead of going near-black)
  const albedo = c.rgb.mul(tint.rgb).mul(mix(0.4, 1.0, vis.mul(vis))).mul(forestU.needleGain).toVar();
  m.colorNode = vec4(albedo, alpha);
  m.roughnessNode = max(orm.y, forestU.needleRough);
  m.metalnessNode = float(0);
  m.aoNode = mix(float(0.12), float(1), vis).mul(mix(float(1), orm.x, 0.5));
  // the stored crown-hull normals, not negated on back faces (DoubleSide would flip normalView there)
  m.normalNode = normalViewGeometry;
  m.positionNode = sharedWind();
  (m as unknown as { receivedShadowNode: (s: N) => N }).receivedShadowNode = (s) => s.mul(farCrownShade());
  // needles with the sun behind them glow yellow-green, only in the crown's thin outer shell and most at the
  // silhouette (the distortion bends the lobe toward the edge); at 4x scale the whole backlit side read white
  const shell = smoothstep(0.55, 0.95, vis);
  m.thicknessColorNode = albedo.mul(vec3(1.05, 1.2, 0.38)).mul(orm.w).mul(shell).mul(forestU.translucency);
  m.thicknessDistortionNode = float(0.6);
  m.thicknessPowerNode = float(5.0);
  m.thicknessScaleNode = float(2.4);
  m.thicknessAttenuationNode = float(0.5);
  return m;
}

function barkMaterial(tex: { c: THREE.Texture; n: THREE.Texture; m: THREE.Texture }) {
  const m = new THREE.MeshStandardNodeMaterial();
  m.name = 'bark';
  const t = uv();
  const vc = attribute('color', 'vec4');
  const vis = vc.x;
  const tint = attribute('aTint', 'vec4');
  const arm = texture(tex.m, t);
  let col: N = texture(tex.c, t).rgb.mul(tint.rgb).mul(mix(0.72, 1.0, vis)).mul(forestU.barkGain);
  // dead snags (tint.w = 1) and dead branch wood (COLOR_0.a = 0 in the forest polish models): bark peeled, the wood
  // weathered silver-grey, split by long dark checks along the grain
  const dead = max(tint.w, float(1).sub(vc.w)).toVar();
  const l = dot(col, vec3(0.2126, 0.7152, 0.0722));
  const checks = smoothstep(0.62, 0.8, tnoise(vec2(t.x.mul(9.0), t.y.mul(0.35)), 4.2)).mul(tint.w);
  col = mix(col, vec3(l).mul(vec3(1.62, 1.6, 1.56)).add(0.016).mul(float(1).sub(checks.mul(0.7))), dead);
  m.colorNode = col;
  m.normalNode = normalMap(texture(tex.n, t));
  m.roughnessNode = mix(arm.y, 0.9, dead);
  m.metalnessNode = float(0);
  // (scene fix: trunks keep some sky light in the stand, so they read in the shade)
  m.aoNode = arm.x.mul(mix(float(0.3), float(1), vis));
  m.positionNode = sharedWind();
  (m as unknown as { receivedShadowNode: (s: N) => N }).receivedShadowNode = (s) => s.mul(farCrownShade());
  return m;
}

interface Part {
  mesh: THREE.InstancedMesh;
  tint: THREE.InstancedBufferAttribute;
  /** needles skip dead snags, so bark and needles of a set keep their own counts */
  needles: boolean;
  /** depth-only twin sharing geometry and instance buffer (needles in view) */
  twin?: THREE.InstancedMesh;
}

/** one instanced mesh per variant x LOD x part; its own geometry wrapper carries the per-instance tint */
function makePart(src: THREE.BufferGeometry, mat: THREE.Material, cap: number, layer: number, caster: boolean,
  twinMat?: THREE.Material, needles = !!twinMat): Part {
  const g = new THREE.BufferGeometry();
  for (const k of Object.keys(src.attributes)) g.setAttribute(k, src.getAttribute(k));
  g.setIndex(src.index);
  const n = instanceCapacity(cap);
  // default usage, not DynamicDrawUsage: three r186 re-uploads a dynamic attribute whole (all `n` instances) in every
  // render call even when unchanged; flush() uploads just the packed range when the camera has moved
  const tint = new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4);
  g.setAttribute('aTint', tint);
  const mesh = new THREE.InstancedMesh(g, mat, n);
  mesh.count = 0;
  mesh.frustumCulled = false;
  mesh.layers.set(layer);
  mesh.castShadow = caster;
  mesh.receiveShadow = !caster;
  let twin: THREE.InstancedMesh | undefined;
  if (twinMat) {
    twin = new THREE.InstancedMesh(g, twinMat, n);
    twin.instanceMatrix = mesh.instanceMatrix;
    twin.count = 0;
    twin.frustumCulled = false;
    twin.layers.set(layer);
    twin.castShadow = twin.receiveShadow = false;
    twin.renderOrder = -1;
  }
  return { mesh, tint, twin, needles };
}

const _vp = new THREE.Matrix4(), _fr = new THREE.Frustum(), _sph = new THREE.Sphere();
const _frM = new THREE.Frustum(), _pm = new THREE.Matrix4(), _mr = new THREE.Vector4();
const _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion(), _s = new THREE.Vector3(), _p = new THREE.Vector3(), _m = new THREE.Matrix4();
const _ax = new THREE.Vector3();
/** casters drawn at LOD1 (closer to the shadow focus; small trees always at LOD2) */
const nearCaster = (t: TreeInst) => t.h > 6 && Math.hypot(t.x - SHADOW_FOCUS[0], t.z - SHADOW_FOCUS[1]) < 38;
/**
 * Small trees switch LODs as if they were further away (a 2 m sapling at LOD0 drew 15-18k triangles for a few
 * hundred pixels): LOD distance x max(1, LOD_REF_H / height); tiny ones are dropped beyond CULL_PER_M x height.
 */
const LOD_REF_H = 10;
const CULL_PER_M = 26;
const lodDist = (t: TreeInst, d: number) => d * Math.max(1, LOD_REF_H / t.h);
const UP = new THREE.Vector3(0, 1, 0);

/** `needles` is null for the bark-only snag models */
interface TreeSet { bark: Part; needles: Part | null }

const POOL_CENTRE: [number, number] = [7, -3];

/** the pool mirror's own cheap copies: LOD1 within this distance, LOD2 beyond (it renders at half resolution) */
const MIRROR_LOD1 = 28;

export class Forest {
  readonly group = new THREE.Group();
  /** main view: [variant][lod] */
  private sets: TreeSet[][] = [];
  /** mirror: [variant][0 = LOD1, 1 = LOD2] */
  private msets: TreeSet[][] = [];
  private mats: Float32Array;
  private needleTint: Float32Array;
  private barkTint: Float32Array;
  private lod: Int8Array;
  private visIdx: Int32Array;
  private visD: Float32Array;
  private mirIdx: Int32Array;
  private mirD: Float32Array;
  private lastVP = new THREE.Matrix4();
  private lastPos = new THREE.Vector3(1e9, 0, 0);
  stats = { visible: 0, lod: [0, 0, 0], casters: 0, mirror: 0 };
  /** the pool's screen rectangle from a camera (NDC; false: not on screen), set once the pool mirror exists */
  mirrorRect: ((camera: THREE.PerspectiveCamera, out: THREE.Vector4) => boolean) | null = null;
  /** empty mirror sets are hidden once the warm-up has compiled their pipelines (see endWarmup) */
  private mirrorHideEmpty = false;

  /** after the warm-up: empty mirror sets skip three's per-object work in the mirror pass like the main view's */
  endWarmup() {
    this.mirrorHideEmpty = true;
    this.lastPos.set(1e9, 0, 0);
  }

  private constructor(readonly trees: TreeInst[]) {
    this.group.name = 'forest';
    const n = trees.length;
    this.mats = new Float32Array(n * 16);
    this.needleTint = new Float32Array(n * 4);
    this.barkTint = new Float32Array(n * 4);
    this.lod = new Int8Array(n).fill(-1);
    this.visIdx = new Int32Array(n);
    this.visD = new Float32Array(n);
    this.mirIdx = new Int32Array(n);
    this.mirD = new Float32Array(n);
    trees.forEach((t, i) => {
      _q.setFromAxisAngle(UP, t.yaw);
      if (t.tilt > 0) {
        _ax.set(-Math.sin(t.tiltDir), 0, Math.cos(t.tiltDir));
        _q.premultiply(_q2.setFromAxisAngle(_ax, t.tilt));
      }
      _m.compose(_p.set(t.x, t.y, t.z), _q, _s.set(t.scale * t.sx, t.scale, t.scale * t.sx));
      this.mats.set(_m.elements, i * 16);
      const V = TREE_VARIANTS[t.v];
      for (let c = 0; c < 3; c++) {
        this.needleTint[i * 4 + c] = V.needleTint[c] * t.jitter[c];
        this.barkTint[i * 4 + c] = V.barkTint[c] * (0.94 + 0.12 * t.jitter[c] - 0.06);
      }
      this.needleTint[i * 4 + 3] = this.barkTint[i * 4 + 3] = t.dead ? 1 : 0;
    });
  }

  static async create(assets: Assets, trees: TreeInst[]): Promise<Forest> {
    const f = new Forest(trees);
    const T = (p: string, srgb: boolean, repeat = true) => assets.texture(`trees/textures/${p}.webp`, srgb, repeat);
    const [nc, nm, pc, pn, pm, fc, fn, fm] = await Promise.all([
      T('needles_c', true, false), T('needles_m', false, false),
      T('bark_pine_c', true), T('bark_pine_n', false), T('bark_pine_m', false),
      T('bark_fir_c', true), T('bark_fir_n', false), T('bark_fir_m', false),
    ]);
    const needles = needleMaterial({ c: nc, m: nm }, true);
    const needlesSolo = needleMaterial({ c: nc, m: nm }, false);
    const needlesDepth = needleDepthMaterial({ c: nc });
    const bark = { pine: barkMaterial({ c: pc, n: pn, m: pm }), fir: barkMaterial({ c: fc, n: fn, m: fm }) };
    const geos = await Promise.all(TREE_VARIANTS.map((v) => Promise.all([0, 1, 2].map(async (l) => {
      const r = await fetch(`${assets.base}trees/${v.id}_lod${l}.glb`);
      if (!r.ok) throw new Error(`${r.status} trees/${v.id}_lod${l}.glb`);
      const prims = parseGlb(await r.arrayBuffer());
      return { bark: prims.find((p) => p.material.startsWith('bark'))!.geometry, needles: prims.find((p) => p.material === 'needles')?.geometry ?? null };
    }))));
    const count = new Array(TREE_VARIANTS.length).fill(0);
    for (const t of trees) count[t.v]++;
    TREE_VARIANTS.forEach((v, vi) => {
      // main view: detailed LODs, needles behind a depth twin
      f.sets.push([0, 1, 2].map((l) => {
        const ng = geos[vi][l].needles;
        const set: TreeSet = {
          bark: makePart(geos[vi][l].bark, bark[v.bark], count[vi], LAYER_FOLIAGE, false),
          needles: ng ? makePart(ng, needles, count[vi], LAYER_FOLIAGE, false, needlesDepth, true) : null,
        };
        set.bark.mesh.name = `tree-${v.id}-lod${l}-bark`;
        f.group.add(set.bark.mesh);
        if (set.needles) {
          set.needles.mesh.name = `tree-${v.id}-lod${l}-needles`;
          set.needles.twin!.name = `tree-${v.id}-lod${l}-needles-depth`;
          f.group.add(set.needles.mesh, set.needles.twin!);
        }
        return set;
      }));
      // mirror: LOD1 near, LOD2 beyond, ordinary depth writes (a quarter of the pixels, no twin needed)
      f.msets.push([1, 2].map((l) => {
        const ng = geos[vi][l].needles;
        const set: TreeSet = {
          bark: makePart(geos[vi][l].bark, bark[v.bark], count[vi], LAYER_MIRROR, false),
          needles: ng ? makePart(ng, needlesSolo, count[vi], LAYER_MIRROR, false, undefined, true) : null,
        };
        set.bark.mesh.name = `tree-${v.id}-mirror${l}-bark`;
        f.group.add(set.bark.mesh);
        if (set.needles) {
          set.needles.mesh.name = `tree-${v.id}-mirror${l}-needles`;
          f.group.add(set.needles.mesh);
        }
        return set;
      }));
      // static shadow casters, seen only by the sun's shadow camera: LOD1 within ~38 m of the shadow focus (lacy
      // shadows with sun flecks on the bar and meadow), LOD2 for the rest of the near stand
      const casters = trees.map((t, i) => [t, i] as const).filter(([t]) => t.v === vi && castsShadow(t));
      for (const [l, near] of [[1, true], [2, false]] as const) {
        const list = casters.filter(([t]) => nearCaster(t) === near);
        if (!list.length) continue;
        const ng = geos[vi][l].needles;
        const set: TreeSet = {
          bark: makePart(geos[vi][l].bark, bark[v.bark], list.length, LAYER_SHADOW, true),
          needles: ng ? makePart(ng, needlesSolo, list.length, LAYER_SHADOW, true, undefined, true) : null,
        };
        for (const [, i] of list) f.put(set, i);
        Forest.flush(set);
        set.bark.mesh.name = `tree-${v.id}-shadow${l}-bark`;
        f.stats.casters += list.length;
        f.group.add(set.bark.mesh);
        if (set.needles) {
          set.needles.mesh.name = `tree-${v.id}-shadow${l}-needles`;
          f.group.add(set.needles.mesh);
        }
      }
    });
    f.addFlares(bark, trees);
    return f;
  }

  /** root flares at the foot of the trees near the play area (main view only) */
  private addFlares(bark: Record<'pine' | 'fir', THREE.Material>, trees: TreeInst[]) {
    for (const kind of ['pine', 'fir'] as const) {
      const list = trees.map((t, i) => [t, i] as const).filter(([t]) => TREE_VARIANTS[t.v].bark === kind
        && TREE_VARIANTS[t.v].trunkR * t.scale > 0.065 && Math.hypot(t.x - SHADOW_FOCUS[0], t.z - SHADOW_FOCUS[1]) < 90);
      if (!list.length) continue;
      const geo = rootFlareGeometry(kind === 'pine' ? 7 : 19);
      const part = makePart(geo, bark[kind], list.length, LAYER_FOLIAGE, false, undefined, false);
      const arr = part.mesh.instanceMatrix.array as Float32Array, tint = part.tint.array as Float32Array;
      list.forEach(([t, i], k) => {
        const rr = TREE_VARIANTS[t.v].trunkR * t.scale;
        _q.setFromAxisAngle(UP, t.yaw * 1.7 + 0.5);
        // the flare follows the ground at the trunk, a little sunk (the tree itself stands 12 cm deeper)
        _m.compose(_p.set(t.x, t.y + 0.09, t.z), _q, _s.set(rr, rr * 1.05, rr));
        arr.set(_m.elements, k * 16);
        tint.set(this.barkTint.subarray(i * 4, i * 4 + 4), k * 4);
      });
      part.mesh.count = list.length;
      part.mesh.frustumCulled = true;
      part.mesh.computeBoundingSphere();
      part.mesh.name = `tree-flares-${kind}`;
      this.group.add(part.mesh);
    }
  }

  /** append tree i to a set (a dead snag only to its bark) */
  private put(s: TreeSet, i: number) {
    for (const p of [s.bark, s.needles]) {
      if (!p || (p.needles && this.trees[i].dead)) continue;
      const k = p.mesh.count;
      (p.mesh.instanceMatrix.array as Float32Array).set(this.mats.subarray(i * 16, i * 16 + 16), k * 16);
      (p.tint.array as Float32Array).set((p.needles ? this.needleTint : this.barkTint).subarray(i * 4, i * 4 + 4), k * 4);
      p.mesh.count = k + 1;
      if (p.twin) p.twin.count = k + 1;
    }
  }

  private static setCount(s: TreeSet, n: number) {
    s.bark.mesh.count = n;
    if (!s.needles) return;
    s.needles.mesh.count = n;
    if (s.needles.twin) s.needles.twin.count = n;
  }

  /**
   * Hide an empty main-view set: three still does its per-object CPU work for an empty InstancedMesh in every pass
   * that sees it (it only skips the draw), and most of the 15 x 3 sets are empty in any one view. The mirror's sets
   * stay visible: the pool mirror renders before warm() refills them, and must still compile their pipelines.
   */
  private static showIfFilled(s: TreeSet) {
    for (const p of [s.bark, s.needles]) {
      if (!p) continue;
      p.mesh.visible = p.mesh.count > 0;
      if (p.twin) p.twin.visible = p.mesh.visible;
    }
  }

  private static flush(s: TreeSet) {
    for (const p of [s.bark, s.needles]) {
      if (!p || p.mesh.count === 0) continue;
      p.mesh.instanceMatrix.clearUpdateRanges();
      p.mesh.instanceMatrix.addUpdateRange(0, p.mesh.count * 16);
      p.mesh.instanceMatrix.needsUpdate = true;
      p.tint.clearUpdateRanges();
      p.tint.addUpdateRange(0, p.mesh.count * 4);
      p.tint.needsUpdate = true;
    }
  }

  /** pack the listed trees front to back into their sets */
  private pack(idx: Int32Array, dist: Float32Array, n: number, setOf: (i: number, d: number) => TreeSet, all: TreeSet[][],
    hideEmpty: boolean) {
    const order = new Uint32Array(n);
    for (let k = 0; k < n; k++) order[k] = k;
    order.sort((a, b) => dist[a] - dist[b]);
    for (const lods of all) for (const s of lods) Forest.setCount(s, 0);
    for (const k0 of order) {
      const i = idx[k0];
      this.put(setOf(i, dist[k0]), i);
    }
    for (const lods of all) {
      for (const s of lods) {
        Forest.flush(s);
        if (hideEmpty) Forest.showIfFilled(s);
      }
    }
  }

  /**
   * Repack when the camera moved (a cheap no-op otherwise): the main view gets the trees in its frustum, the
   * mirror the trees whose mirror image is in it (the pool reflects crowns above the frame).
   */
  update(camera: THREE.PerspectiveCamera, force = false) {
    camera.updateMatrixWorld();
    _vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const cp = camera.position;
    if (!force && cp.distanceToSquared(this.lastPos) < 1e-6 && _vp.equals(this.lastVP)) return;
    this.lastVP.copy(_vp);
    this.lastPos.copy(cp);
    _fr.setFromProjectionMatrix(_vp, camera.coordinateSystem, camera.reversedDepth);
    const trees = this.trees;
    let n = 0, nm = 0;
    const far = LOD_DIST[2];
    let mirrorOn = cp.y > POOL_LEVEL + 0.05;
    // a copy is drawn only if its mirror image falls where the pool can be seen: the view's frustum narrowed to the
    // pool's screen rectangle (x and y rows of the clip matrix only)
    let mirrorFr = _fr;
    if (mirrorOn && this.mirrorRect) {
      if (!this.mirrorRect(camera, _mr)) mirrorOn = false;
      else if ((_mr.z - _mr.x) * (_mr.w - _mr.y) < 3.6) {
        const cx = (_mr.x + _mr.z) / 2, sx = (_mr.z - _mr.x) / 2, cy = (_mr.y + _mr.w) / 2, sy = (_mr.w - _mr.y) / 2;
        const e = _pm.copy(camera.projectionMatrix).elements;
        for (let j = 0; j < 16; j += 4) {
          e[j] = (e[j] - cx * e[j + 3]) / sx;
          e[j + 1] = (e[j + 1] - cy * e[j + 3]) / sy;
        }
        _frM.setFromProjectionMatrix(_pm.multiply(camera.matrixWorldInverse), camera.coordinateSystem, camera.reversedDepth);
        mirrorFr = _frM;
      }
    }
    for (let i = 0; i < trees.length; i++) {
      const t = trees[i];
      const dx = t.x - cp.x, dz = t.z - cp.z, dy = t.y + t.h * 0.4 - cp.y;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > far || t.rank >= keepAt(d) || d > CULL_PER_M * t.h) {
        this.lod[i] = -1;
        continue;
      }
      const dl = lodDist(t, d);
      _sph.center.set(t.x, t.y + t.h * 0.5, t.z);
      _sph.radius = t.h * 0.55 + t.r;
      if (_fr.intersectsSphere(_sph)) {
        let l = dl < LOD_DIST[0] ? 0 : dl < LOD_DIST[1] ? 1 : 2;
        const prev = this.lod[i];
        if (prev >= 0 && prev !== l) {
          const edge = LOD_DIST[Math.min(l, prev)];
          if (Math.abs(dl - edge) < edge * 0.06) l = prev;
        }
        this.lod[i] = l;
        this.visIdx[n] = i;
        this.visD[n] = d;
        n++;
      } else {
        this.lod[i] = -1;
      }
      // the pool only mirrors what stands within ~130 m of it (beyond, a tree's reflection is a speck at a grazing
      // angle); C2 was drawing ~1300 copies into a mirror of which a small patch of water is seen
      if (mirrorOn && t.h > (TREE_VARIANTS[t.v].young ? 6 : 3) && Math.hypot(t.x - POOL_CENTRE[0], t.z - POOL_CENTRE[1]) < 130) {
        _sph.center.y = 2 * POOL_LEVEL - _sph.center.y;
        if (mirrorFr.intersectsSphere(_sph)) {
          this.mirIdx[nm] = i;
          this.mirD[nm] = dl;
          nm++;
        }
      }
    }
    this.stats.lod = [0, 0, 0];
    this.pack(this.visIdx, this.visD, n, (i) => {
      this.stats.lod[this.lod[i]]++;
      return this.sets[trees[i].v][this.lod[i]];
    }, this.sets, true);
    this.pack(this.mirIdx, this.mirD, nm, (i, d) => this.msets[trees[i].v][d < MIRROR_LOD1 ? 0 : 1], this.msets, this.mirrorHideEmpty);
    this.stats.visible = n;
    this.stats.mirror = nm;
  }

  /** pipeline warm-up behind the loading screen: every set draws at least its first tree */
  warm() {
    [this.sets, this.msets].forEach((all) => all.forEach((lods, vi) => {
      const i = this.trees.findIndex((t) => t.v === vi && (!t.dead || !!TREE_VARIANTS[vi].snag));
      if (i < 0) return;
      for (const s of lods) {
        if (s.bark.mesh.count > 0 && (!s.needles || s.needles.mesh.count > 0)) continue;
        Forest.setCount(s, 0);
        this.put(s, i);
        Forest.flush(s);
        Forest.showIfFilled(s);
      }
    }));
    this.lastPos.set(1e9, 0, 0);
  }
}
