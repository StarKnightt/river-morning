import * as THREE from 'three/webgpu';
import { texture, uniform, type N } from '../gfx/tsl';
import { LAYER_MIRROR } from '../core/layers';
import { params } from '../core/params';

export { LAYER_WATER, LAYER_NO_REFLECT } from '../core/layers';

const _pos = new THREE.Vector3(), _fwd = new THREE.Vector3(), _up = new THREE.Vector3(), _quat = new THREE.Quaternion();
const _plane = new THREE.Plane(), _clip = new THREE.Vector4(), _q = new THREE.Vector4(), _pt = new THREE.Vector3();
const _vp = new THREE.Matrix4(), _size = new THREE.Vector2(), _frustum = new THREE.Frustum(), _rect = new THREE.Vector4();
const UP = new THREE.Vector3(0, 1, 0);

/** convex hull of (x, z) pairs, counter-clockwise (monotone chain) */
function convexHull(xz: ArrayLike<number>): Float32Array {
  const n = xz.length / 2, idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => xz[a * 2] - xz[b * 2] || xz[a * 2 + 1] - xz[b * 2 + 1]);
  const cross = (o: number, a: number, b: number) =>
    (xz[a * 2] - xz[o * 2]) * (xz[b * 2 + 1] - xz[o * 2 + 1]) - (xz[a * 2 + 1] - xz[o * 2 + 1]) * (xz[b * 2] - xz[o * 2]);
  const lower: number[] = [], upper: number[] = [];
  for (const i of idx) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], i) <= 0) lower.pop();
    lower.push(i);
  }
  for (let k = idx.length - 1; k >= 0; k--) {
    const i = idx[k];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], i) <= 0) upper.pop();
    upper.push(i);
  }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  const out = new Float32Array(hull.length * 2);
  hull.forEach((i, k) => {
    out[k * 2] = xz[i * 2];
    out[k * 2 + 1] = xz[i * 2 + 1];
  });
  return out;
}

/**
 * Replace the near plane of a perspective projection with `clip` (view space, kept side positive) and put
 * the far plane through the far frustum corner (Lengyel's oblique clipping) for all three depth conventions:
 *  - reversed depth (WebGPU, or WebGL2 with EXT_clip_control): depth 1 at the near plane, 0 at the far plane,
 *    so row 3 becomes row 4 minus the scaled plane and the corner is the one that maps to depth 0;
 *  - WebGPU [0, 1]: row 3 = C / (C·q)   (what three's ReflectorNode does);
 *  - WebGL [-1, 1]: row 3 = 2C / (C·q) + (0, 0, 1, 0).
 * three's reflector() only has the last two; with reversedDepthBuffer its kept side flips and the mirror
 * shows only what lies between the virtual camera and the water plane.
 */
export function obliqueClip(P: THREE.Matrix4, clip: THREE.Vector4, reversed: boolean, webgpu: boolean) {
  const e = P.elements;
  if (reversed) {
    _q.set((Math.sign(clip.x) + e[8]) / e[0], (Math.sign(clip.y) + e[9]) / e[5], -1, e[10] / e[14]);
    const c = clip.multiplyScalar(1 / clip.dot(_q));
    e[2] = -c.x;
    e[6] = -c.y;
    e[10] = -1 - c.z;
    e[14] = -c.w;
  } else {
    _q.set((Math.sign(clip.x) + e[8]) / e[0], (Math.sign(clip.y) + e[9]) / e[5], -1, (1 + e[10]) / e[14]);
    const c = clip.multiplyScalar((webgpu ? 1 : 2) / clip.dot(_q));
    e[2] = c.x;
    e[6] = c.y;
    e[10] = webgpu ? c.z : c.z + 1;
    e[14] = c.w;
  }
}

/**
 * Planar mirror for the calm pool (y = POOL_LEVEL). Renders the scene (layer 0 only) from the mirrored camera
 * into a reduced-resolution half-float target. The water samples it projectively with the matrix it was rendered
 * with, so a copy a few frames old stays correct while the camera moves. Skipped when the pool chunk is off screen.
 * The sun's shadow map is reused, not redrawn.
 *
 * (polish round) Two targets: the water reads the front one while the next copy is drawn into the back one, in up
 * to three column bands from one frozen pose, one band per gap between frames (Engine.addPostFrame); the last band
 * swaps them. Only the part of the image the pool can show is drawn.
 */
export class PoolReflector {
  private readonly rts = [0, 1].map(() => new THREE.RenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: true }));
  private front = 0;
  /** the front copy, for the water's lookups (texture nodes made from these follow the swaps) */
  readonly colorTex: N;
  readonly depthTex: N;
  readonly cam = new THREE.PerspectiveCamera();
  readonly u = {
    mat: uniform(new THREE.Matrix4()), inv: uniform(new THREE.Matrix4()), on: uniform(0), planeY: uniform(0),
    /** where the water may look up the mirror (u0, v0, u1, v1): the rendered region, inset where it ends inside the image */
    rect: uniform(new THREE.Vector4(0, 0, 1, 1)),
  };
  resolutionScale = 0.5;
  /** start a new copy every n-th update (1 = every frame); a copy in bands takes that many updates to finish */
  every = 1;
  enabled = true;
  /** true when the last update() rendered (a band of) the mirror */
  renderedNow = false;
  private sinceStart = 1e9;
  /** the pool's convex outline on the water plane (x, z pairs); null: the whole frame is rendered */
  private hull: Float32Array | null = null;
  /** the copy being drawn into the back target: frozen pose, full clip matrix, pixel region, bands */
  private cycle: {
    pos: THREE.Vector3; quat: THREE.Quaternion; proj: THREE.Matrix4; mat: THREE.Matrix4; rect: THREE.Vector4;
    px0: number; py0: number; px1: number; py1: number; bands: number; next: number; from: THREE.Vector3;
  } | null = null;

  constructor(
    private renderer: THREE.WebGPURenderer,
    private scene: THREE.Scene,
    private sun: THREE.DirectionalLight,
    readonly planeY: number,
    private bounds: THREE.Box3,
  ) {
    this.u.planeY.value = planeY;
    this.cam.name = 'mirror';
    this.cam.layers.set(0);
    this.cam.layers.enable(LAYER_MIRROR);
    for (const rt of this.rts) {
      // (pool mirror + waterline patch) a mip chain: the water blurs the copy by the ripples it cannot resolve
      rt.texture.generateMipmaps = params.pool;
      rt.texture.minFilter = params.pool ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
      rt.texture.magFilter = THREE.LinearFilter;
      // float depth like PassNode under reversed depth
      rt.depthTexture = new THREE.DepthTexture(1, 1);
      rt.depthTexture.type = THREE.FloatType;
    }
    this.colorTex = texture(this.rts[0].texture);
    this.depthTex = texture(this.rts[0].depthTexture!);
  }

  /** the target the water reads */
  get rt(): THREE.RenderTarget {
    return this.rts[this.front];
  }

  /**
   * The water meshes: the outline of all water within 10 cm of the pool's level (the water uses the mirror where it
   * lies within 6 cm of it and runs slowly, also in the riffle's and the tail's edges) limits the mirror to the part
   * of the frame that water can cover.
   */
  setOutline(geometries: THREE.BufferGeometry[]) {
    const xz: number[] = [];
    for (const g of geometries) {
      const pos = g.getAttribute('position');
      for (let i = 0; i < pos.count; i++) {
        if (Math.abs(pos.getY(i) - this.planeY) < 0.1) xz.push(pos.getX(i), pos.getZ(i));
      }
    }
    this.hull = xz.length >= 6 ? convexHull(xz) : null;
  }

  /**
   * The screen rectangle (NDC x0, y0, x1, y1) the pool's outline covers from `camera`, clipped by the near plane,
   * with a margin for the ripples' deflection and the blur mips, clamped to the frame. False when none of it is
   * in front of the camera. Without an outline the whole frame.
   */
  screenRect(camera: THREE.PerspectiveCamera, out: THREE.Vector4): boolean {
    const h = this.hull;
    if (!h) {
      out.set(-1, -1, 1, 1);
      return true;
    }
    const v = camera.matrixWorldInverse.elements, p = camera.projectionMatrix.elements, y = this.planeY;
    const zc = -camera.near * 1.01, n = h.length / 2;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    const emit = (vx: number, vy: number, vz: number) => {
      const w = p[3] * vx + p[7] * vy + p[11] * vz + p[15];
      const nx = (p[0] * vx + p[4] * vy + p[8] * vz + p[12]) / w, ny = (p[1] * vx + p[5] * vy + p[9] * vz + p[13]) / w;
      x0 = Math.min(x0, nx); x1 = Math.max(x1, nx); y0 = Math.min(y0, ny); y1 = Math.max(y1, ny);
    };
    // the outline in view space, clipped against the near plane (Sutherland-Hodgman, one plane)
    let ax = 0, ay = 0, az = 0;
    for (let i = 0; i <= n; i++) {
      const k = (i % n) * 2, x = h[k], z = h[k + 1];
      const bx = v[0] * x + v[4] * y + v[8] * z + v[12], by = v[1] * x + v[5] * y + v[9] * z + v[13], bz = v[2] * x + v[6] * y + v[10] * z + v[14];
      if (i > 0) {
        const aIn = az < zc, bIn = bz < zc;
        if (aIn) emit(ax, ay, az);
        if (aIn !== bIn) {
          const t = (zc - az) / (bz - az);
          emit(ax + (bx - ax) * t, ay + (by - ay) * t, zc);
        }
      }
      ax = bx; ay = by; az = bz;
    }
    if (x0 > x1) return false;
    const m = 0.14;
    out.set(Math.max(-1, x0 - m), Math.max(-1, y0 - m), Math.min(1, x1 + m), Math.min(1, y1 + m));
    return out.x < out.z && out.y < out.w;
  }

  /** `warm`: render even when the pool is off screen (pipeline warm-up behind the loading screen) */
  update(camera: THREE.PerspectiveCamera, force = false, warm = false) {
    this.renderedNow = false;
    camera.updateMatrixWorld();
    const camPos = camera.getWorldPosition(_pos);
    _vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    _frustum.setFromProjectionMatrix(_vp, camera.coordinateSystem, camera.reversedDepth);
    const onScreen = this.screenRect(camera, _rect);
    const visible = this.enabled && camPos.y > this.planeY + 0.05 && (warm || (onScreen && _frustum.intersectsBox(this.bounds)));
    if (!visible) {
      this.u.on.value = 0;
      this.cycle = null;
      return;
    }
    this.sinceStart++;
    // the first copy (or the warm-up's, or the one after a jump) is drawn whole and shown at once; then a copy in
    // bands is under way or due
    const now = force || warm || this.u.on.value === 0 || this.jumped(camera);
    if (!now && this.cycle) {
      this.drawBand();
      return;
    }
    if (!now && this.sinceStart < this.every) return;
    this.sinceStart = 0;
    this.startCycle(camera, onScreen && !warm, now);
    this.drawBand();
  }

  /**
   * The camera moved more than 3 m since the shown copy started (a jump, a capture spot): that copy no longer
   * covers the view, so the next one is drawn whole before the frame instead of over the next 3-6 frames.
   */
  jumped(camera: THREE.PerspectiveCamera): boolean {
    return this.u.on.value === 1 && camera.getWorldPosition(_pt).distanceToSquared(this.shownFrom) > 9;
  }

  private shownFrom = new THREE.Vector3();

  private startCycle(camera: THREE.PerspectiveCamera, restrict: boolean, whole: boolean) {
    const r = this.renderer;
    const camPos = camera.getWorldPosition(_pos);
    // mirrored pose built with lookAt: a proper rotation, so triangle winding and culling match the main view
    const vc = this.cam;
    vc.fov = camera.fov;
    vc.aspect = camera.aspect;
    vc.near = camera.near;
    vc.far = camera.far;
    vc.zoom = camera.zoom;
    vc.coordinateSystem = r.coordinateSystem;
    // set before the first render: otherwise the renderer flips the flag itself and rebuilds (un-clips) this projection
    (vc as unknown as { _reversedDepth: boolean })._reversedDepth = r.reversedDepthBuffer;
    vc.updateProjectionMatrix();
    camera.getWorldDirection(_fwd);
    _up.set(0, 1, 0).applyQuaternion(camera.getWorldQuaternion(_quat));
    const y2 = 2 * this.planeY;
    vc.position.set(camPos.x, y2 - camPos.y, camPos.z);
    vc.up.set(_up.x, -_up.y, _up.z);
    vc.lookAt(camPos.x + _fwd.x, y2 - (camPos.y + _fwd.y), camPos.z + _fwd.z);
    vc.updateMatrixWorld();

    // oblique near plane 2 cm under the surface: nothing below the water reaches the mirror, the waterline does
    _plane.setFromNormalAndCoplanarPoint(UP, _pt.set(0, this.planeY - 0.02, 0)).applyMatrix4(vc.matrixWorldInverse);
    _clip.set(_plane.normal.x, _plane.normal.y, _plane.normal.z, _plane.constant);
    obliqueClip(vc.projectionMatrix, _clip, r.reversedDepthBuffer, r.coordinateSystem === THREE.WebGPUCoordinateSystem);

    r.getDrawingBufferSize(_size);
    const w = Math.max(1, Math.round(_size.x * this.resolutionScale)), h = Math.max(1, Math.round(_size.y * this.resolutionScale));
    // a resize empties both copies: the next one is drawn whole
    let resized = false;
    for (const rt of this.rts) {
      if (rt.width === w && rt.height === h) continue;
      rt.setSize(w, h);
      resized = true;
    }

    // only the part of the image the pool can show is rendered (at C1 or the campfire the pool is a strip of the
    // frame, yet the whole mirror cost ~5 ms). The mirror camera sees the water plane mirrored left to right.
    let px0 = 0, py0 = 0, px1 = w, py1 = h;
    if (restrict) {
      const x0 = -_rect.z, x1 = -_rect.x, y0 = _rect.y, y1 = _rect.w;
      if ((x1 - x0) * (y1 - y0) < 3.6) {
        px0 = Math.max(0, Math.floor((x0 + 1) * 0.5 * w));
        px1 = Math.min(w, Math.ceil((x1 + 1) * 0.5 * w));
        py0 = Math.max(0, Math.floor((1 - y1) * 0.5 * h));
        py1 = Math.min(h, Math.ceil((1 - y0) * 0.5 * h));
      }
    }
    // the water's lookups stay inside the rendered part, 16 px in where it ends inside the image (the blur mips
    // would mix in the cleared surround)
    const iu = 16 / w, iv = 16 / h;
    const rect = new THREE.Vector4(px0 > 0 ? px0 / w + iu : 0, py0 > 0 ? py0 / h + iv : 0, px1 < w ? px1 / w - iu : 1, py1 < h ? py1 / h - iv : 1);
    // the GPU time grows with the pixels drawn: a pool filling the frame (C2, ~6.5 ms) is drawn in three bands, one per
    // gap between frames; a strip of water in one
    const area = ((px1 - px0) * (py1 - py0)) / (w * h);
    const bands = whole || resized ? 1 : Math.min(Math.max(1, this.every - 1), area > 0.6 ? 3 : area > 0.3 ? 2 : 1);
    this.cycle = {
      pos: vc.position.clone(), quat: vc.quaternion.clone(), proj: vc.projectionMatrix.clone(),
      mat: new THREE.Matrix4().multiplyMatrices(vc.projectionMatrix, vc.matrixWorldInverse), rect,
      px0, py0, px1, py1, bands, next: 0, from: camPos.clone(),
    };
  }

  /** draw the next band of the copy under way into the back target; the last one makes it the front */
  private drawBand() {
    const c = this.cycle!, r = this.renderer, vc = this.cam, back = this.rts[1 - this.front];
    const w = back.width, h = back.height;
    // the pose is the copy's own (frozen when it started), whatever the camera did since
    vc.position.copy(c.pos);
    vc.quaternion.copy(c.quat);
    vc.updateMatrixWorld();
    // the bands are columns: what the mirror shows changes mostly from top to bottom (the far bank's crowns in a few
    // rows, sky in the rest), so equal rows cost 1.9 / 3.9 / 2.4 ms at C2 and the dearest one set the frame time
    const bx0 = c.px0 + Math.floor(((c.px1 - c.px0) * c.next) / c.bands);
    const bx1 = c.px0 + Math.floor(((c.px1 - c.px0) * (c.next + 1)) / c.bands);
    // a sub-frustum of exactly this pixel rectangle: only the clip matrix's x and y rows change, so the depth it writes
    // (which the water turns back into distance with u.inv) is the full frustum's, and culling drops what lies outside
    vc.projectionMatrix.copy(c.proj);
    if (bx0 > 0 || c.py0 > 0 || bx1 < w || c.py1 < h) {
      const X0 = (bx0 / w) * 2 - 1, X1 = (bx1 / w) * 2 - 1, Y1 = 1 - (c.py0 / h) * 2, Y0 = 1 - (c.py1 / h) * 2;
      const cx = (X0 + X1) / 2, sx = (X1 - X0) / 2, cy = (Y0 + Y1) / 2, sy = (Y1 - Y0) / 2;
      const e = vc.projectionMatrix.elements;
      for (let j = 0; j < 16; j += 4) {
        e[j] = (e[j] - cx * e[j + 3]) / sx;
        e[j + 1] = (e[j + 1] - cy * e[j + 3]) / sy;
      }
    }
    vc.projectionMatrixInverse.copy(vc.projectionMatrix).invert();
    // WebGPU viewport: y from the top, as the water's lookup coordinates
    back.viewport.set(bx0, c.py0, bx1 - bx0, c.py1 - c.py0);

    // ShadowNode renders once per frame *per camera*: without this the sun map is redrawn for the mirror
    const shadow = this.sun.shadow, auto = shadow.autoUpdate, autoClear = r.autoClear;
    shadow.autoUpdate = false;
    // the first band clears the whole back target (nobody reads it); the others keep what is drawn
    r.autoClear = c.next === 0;
    const prevRT = r.getRenderTarget(), prevMRT = r.getMRT();
    r.setMRT(null);
    r.setRenderTarget(back);
    r.render(this.scene, vc);
    r.setRenderTarget(prevRT);
    r.setMRT(prevMRT);
    r.autoClear = autoClear;
    shadow.autoUpdate = auto;
    this.renderedNow = true;
    if (++c.next < c.bands) return;
    // the copy is complete: it becomes the front, with the matrix and region it was drawn with
    this.front = 1 - this.front;
    this.colorTex.value = back.texture;
    this.depthTex.value = back.depthTexture;
    this.u.mat.value.copy(c.mat);
    this.u.inv.value.copy(c.mat).invert();
    this.u.rect.value.copy(c.rect);
    this.u.on.value = 1;
    this.shownFrom.copy(c.from);
    this.cycle = null;
  }

  dispose() {
    for (const rt of this.rts) rt.dispose();
  }
}
