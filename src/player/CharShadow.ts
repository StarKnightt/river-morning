import * as THREE from 'three/webgpu';
import { abs, dot, float, mix, step, texture, uniform, vec2, vec3, vec4, positionWorld, type N } from '../gfx/tsl';

const q = new URLSearchParams(location.search);
/** ?charshadow=0: no self-shadow map (the sun map's soft self-shadow and the analytic brim shadow instead) */
export const charShadowEnabled = q.get('charshadow') !== '0';

/** map resolution and the half-size of its square (m): 2.6 m across in 1024 texels = 2.5 mm a texel on him */
const SIZE = 1024;
const HALF = 1.3;
/** the light camera sits this far from his centre toward the sun; depths are stored relative to that distance */
const DIST = 3;

/**
 * The angler's own sun shadow, sharp: the scene's sun map has 2.5 cm texels and a wide PCF, so the brim, nose,
 * arms and rod threw no crisp shadow on him while the stones beside him had hard ones. Each frame his skinned
 * meshes and the rod are drawn from the sun's direction into a small map fitted round him (depth along the sun ray
 * in metres, relative to the light camera's distance), and his materials test against it. Trees and terrain still
 * shade him through the scene's sun map, looked up 0.4 m toward the sun so that map's blurry copy of his own body
 * drops out (moving the lookup along the light ray leaves every other caster where it was).
 */
export const charShadowU = {
  camPos: uniform(new THREE.Vector3(0, 10, 0)),
  /** unit vector toward the sun */
  toSun: uniform(new THREE.Vector3(0, 1, 0)),
  right: uniform(new THREE.Vector3(1, 0, 0)),
  up: uniform(new THREE.Vector3(0, 0, 1)),
  /** 1 while the map is live (0 before the first render, with ?charshadow=0 or while he is hidden) */
  on: uniform(0),
};

export const charShadowTarget = new THREE.RenderTarget(SIZE, SIZE, {
  type: THREE.HalfFloatType,
  format: THREE.RedFormat,
  depthBuffer: true,
  generateMipmaps: false,
  minFilter: THREE.NearestFilter,
  magFilter: THREE.NearestFilter,
});
charShadowTarget.texture.name = 'angler-self-shadow';

/** Distance along the sun ray from the light camera, minus DIST (what the map stores; 0 at his centre). */
export const sunDepth = (p: N): N => dot(vec3(charShadowU.camPos).sub(p), vec3(charShadowU.toSun)).sub(DIST);

/**
 * Sun visibility through the angler himself at world point `p` (normal `n`): 1 lit, 0 in his own shadow; a 3x3
 * PCF over 2.5 mm texels gives a ~7 mm penumbra, as sharp as the sun's on the stones. Outside the map: lit.
 */
export function charShadow(p: N, n: N): N {
  const L = vec3(charShadowU.toSun);
  const nl = dot(n, L);
  // receiver offset along the normal and toward the sun, more at grazing angles (acne on the silhouette)
  const pr = p.add(n.mul(0.003)).add(L.mul(float(0.003).add(float(0.006).mul(float(1).sub(abs(nl))))));
  const rel = pr.sub(vec3(charShadowU.camPos));
  const nx = dot(rel, vec3(charShadowU.right)).div(HALF), ny = dot(rel, vec3(charShadowU.up)).div(HALF);
  const base = vec2(nx.mul(0.5).add(0.5), float(0.5).sub(ny.mul(0.5)));
  const d = sunDepth(pr);
  const t = 1 / SIZE;
  let lit: N = float(0);
  for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
    const s = texture(charShadowTarget.texture, base.add(vec2(i * t, j * t))).r;
    lit = lit.add(step(d, s));
  }
  const inside = step(abs(nx), 0.98).mul(step(abs(ny), 0.98));
  return mix(float(1), lit.div(9), inside.mul(charShadowU.on));
}

/** World position the scene's sun map is read at for him (see charShadowU). */
export function sunMapLookup(): N {
  return positionWorld.add(vec3(charShadowU.toSun).mul(float(0.4).mul(charShadowU.on)));
}

const FAR = new THREE.Color(100, 0, 0);

function depthMaterial(name: string, side: THREE.Side, opacity: N | null, alphaTest: number): THREE.MeshBasicNodeMaterial {
  const m = new THREE.MeshBasicNodeMaterial();
  m.name = name;
  m.side = side;
  m.fog = false;
  m.colorNode = vec4(sunDepth(positionWorld), 0, 0, 1);
  if (opacity) {
    m.opacityNode = opacity;
    m.alphaTest = alphaTest;
  }
  return m;
}

/** Proxies of the angler's skinned meshes and the rod (same geometry and skeleton), drawn into the self-shadow map. */
export class CharShadow {
  private readonly scene = new THREE.Scene();
  private readonly cam = new THREE.OrthographicCamera(-HALF, HALF, HALF, -HALF, DIST - 2.2, DIST + 2.2);
  private readonly pairs: { src: THREE.SkinnedMesh; dst: THREE.SkinnedMesh }[] = [];
  private readonly clear = new THREE.Color();

  /**
   * `alpha` gives each source mesh's cut-out (the gear atlas' hair and beard cards), or null for solid ones.
   */
  constructor(meshes: readonly THREE.SkinnedMesh[], alpha: (m: THREE.SkinnedMesh) => N | null) {
    this.scene.name = 'angler-self-shadow';
    this.scene.matrixWorldAutoUpdate = false;
    const solid = depthMaterial('AnglerSelfShadow', THREE.DoubleSide, null, 0);
    for (const src of meshes) {
      const a = alpha(src);
      const mat = a ? depthMaterial('AnglerSelfShadowCut', THREE.DoubleSide, a, 0.3) : solid;
      const dst = new THREE.SkinnedMesh(src.geometry, mat);
      dst.name = `${src.name}-selfshadow`;
      dst.bind(src.skeleton, src.bindMatrix);
      dst.matrixAutoUpdate = false;
      dst.matrixWorldAutoUpdate = false;
      dst.frustumCulled = false;
      this.scene.add(dst);
      this.pairs.push({ src, dst });
    }
  }

  /** Fit the light camera round `centre` along `toSun` and draw the proxies (bones and matrices must be current). */
  render(renderer: THREE.WebGPURenderer, centre: THREE.Vector3, toSun: THREE.Vector3, visible: boolean) {
    const u = charShadowU;
    if (!charShadowEnabled || !visible) {
      u.on.value = 0;
      return;
    }
    const cam = this.cam;
    cam.position.copy(centre).addScaledVector(toSun, DIST);
    cam.up.set(0, 1, 0);
    if (Math.abs(toSun.y) > 0.99) cam.up.set(0, 0, 1);
    cam.lookAt(centre);
    cam.updateMatrixWorld();
    u.camPos.value.copy(cam.position);
    u.toSun.value.copy(toSun);
    u.right.value.setFromMatrixColumn(cam.matrixWorld, 0).normalize();
    u.up.value.setFromMatrixColumn(cam.matrixWorld, 1).normalize();
    for (const p of this.pairs) {
      // attached bind mode: the source recomputes its bindMatrixInverse from its world matrix every frame
      p.dst.matrixWorld.copy(p.src.matrixWorld);
      p.dst.bindMatrix.copy(p.src.bindMatrix);
      p.dst.bindMatrixInverse.copy(p.src.bindMatrixInverse);
    }
    const prevTarget = renderer.getRenderTarget(), prevMRT = renderer.getMRT();
    const prevAuto = renderer.autoClear;
    renderer.getClearColor(this.clear);
    const prevAlpha = renderer.getClearAlpha();
    renderer.setMRT(null);
    renderer.setRenderTarget(charShadowTarget);
    // empty map = far (100 m): a half float holds it exactly
    renderer.setClearColor(FAR, 1);
    renderer.autoClear = true;
    renderer.render(this.scene, cam);
    renderer.setRenderTarget(prevTarget);
    renderer.setMRT(prevMRT);
    renderer.setClearColor(this.clear, prevAlpha);
    renderer.autoClear = prevAuto;
    u.on.value = 1;
  }
}
