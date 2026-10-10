import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import {
  BRDF_Lambert, attribute, cameraPosition, cameraViewMatrix, diffuseContribution, dot, exp, float, instancedDynamicBufferAttribute,
  length, max, mix, normalWorld, normalize, positionViewDirection, positionWorld, saturate, smoothstep, sqrt, uniform, uv, vec2,
  vec3, vec4,
} from '../gfx/tsl';
import { LAYER_NO_REFLECT } from '../core/layers';
import { charShadow } from '../player/CharShadow';
import type { FishingSim } from './FishingSim';
import { AIM_POINTS } from './FishingSim';
import type { FishModels } from './FishModels';
import type { RodRig, SpeciesId } from './types';
import { SPECIES_IDS } from './species';
import { clamp, smoothstep as smoothstepJs } from './util';

/** source points (rod tip, rope, leader) and the smoothed points drawn between them */
const MAX_SRC = 64;
const MAX_PTS = 720;
/** each rope segment is cut into steps about this many pixels long on screen, at most MAX_SUB of them */
const STEP_PX = 3;
const MAX_SUB = 24;
const LEADER_PTS = 5;
/** line radius (m): a 0.7 mm silk line, drawn at least a pixel wide with its coverage as alpha below that */
const LINE_RADIUS = 0.00035;
/** the faintest a thread of line ever gets before it is lost (near, and the farther floor) */
const FLOOR_NEAR = 0.035;
const FLOOR_FAR = 0.012;
/** foreground blur of the line passing close to the lens: blur circle (px) per dioptre in front of the focus */
const COC_PX = 13;
/** a meniscus dent where floating line touches the water (m across), and the spacing of the dents along it */
const DENT = 0.004;
const DENT_GAP = 0.085;
/** the loose curves of slack lying on the water: their size across the line (m) and how far they take to fade in (m) */
const WANDER = { amp: 0.07, ramp: 0.7 };
/**
 * The fibre's light (FiberLighting): the widths of the reflection (R) and transmission (TT) lobes across the cone
 * (in sin(theta_i) + sin(theta_o)), the TT lobe's width round the direction toward the light (in 1 + cos(phi)), their
 * strengths per unit of sun, and the warm tint of light that has passed through the silk.
 */
const FIB = { sigmaR: 0.07, sigmaT: 0.2, phiT: 0.06, kR: 0.45, kT: 2.0, tint: [1.0, 0.8, 0.52] as const };
const _cam = new THREE.Vector3(), _t = new THREE.Vector3(), _toCam = new THREE.Vector3();
const _n = new THREE.Vector3(), _n2 = new THREE.Vector3(), _tip = new THREE.Vector3(), _p = new THREE.Vector3();
const _w = new THREE.Vector3();

/** One camera-facing strip, rebuilt on the CPU each frame (2 vertices per point). */
class Strip {
  readonly mesh: THREE.Mesh;
  readonly x = new Float32Array(MAX_PTS);
  readonly y = new Float32Array(MAX_PTS);
  readonly z = new Float32Array(MAX_PTS);
  /** half width (m), opacity, softness of the edges (0 sharp .. ~1 a soft blur), meniscus glint */
  readonly half = new Float32Array(MAX_PTS);
  readonly alpha = new Float32Array(MAX_PTS);
  readonly soft = new Float32Array(MAX_PTS);
  readonly bead = new Float32Array(MAX_PTS);
  /** normal (for the sky's light) and the fibre's direction, world space */
  readonly nx = new Float32Array(MAX_PTS);
  readonly ny = new Float32Array(MAX_PTS);
  readonly nz = new Float32Array(MAX_PTS);
  readonly tx = new Float32Array(MAX_PTS);
  readonly ty = new Float32Array(MAX_PTS);
  readonly tz = new Float32Array(MAX_PTS);
  count = 0;
  private pos: THREE.BufferAttribute;
  private nrm: THREE.BufferAttribute;
  private cov: THREE.BufferAttribute;
  private bed: THREE.BufferAttribute;
  private sid: THREE.BufferAttribute;
  private fib: THREE.BufferAttribute;

  /** `mirrored`: the line's reflection on the water (shaded as seen from the eye's mirror image) */
  constructor(material: THREE.Material, name: string, private readonly mirrored: boolean) {
    const g = new THREE.BufferGeometry();
    const dyn = (size: number) => {
      const a = new THREE.BufferAttribute(new Float32Array(MAX_PTS * 2 * size), size);
      a.setUsage(THREE.DynamicDrawUsage);
      return a;
    };
    this.pos = dyn(3);
    this.nrm = dyn(3);
    this.cov = dyn(1);
    this.bed = dyn(1);
    this.sid = dyn(1);
    this.fib = dyn(4);
    g.setAttribute('position', this.pos);
    g.setAttribute('normal', this.nrm);
    g.setAttribute('aCov', this.cov);
    g.setAttribute('aBead', this.bed);
    g.setAttribute('aSide', this.sid);
    g.setAttribute('aFib', this.fib);
    const idx: number[] = [];
    for (let i = 0; i < MAX_PTS - 1; i++) {
      const a = i * 2;
      idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    g.setIndex(idx);
    this.mesh = new THREE.Mesh(g, material);
    this.mesh.name = name;
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 2;
    this.mesh.receiveShadow = true;
    this.mesh.layers.set(LAYER_NO_REFLECT);
  }

  /** the strip's vertices for this camera position */
  upload(cam: THREE.Vector3) {
    const n = this.count;
    const P = this.pos.array as Float32Array, Nn = this.nrm.array as Float32Array, F = this.fib.array as Float32Array;
    const C = this.cov.array as Float32Array, B = this.bed.array as Float32Array, S = this.sid.array as Float32Array;
    const mir = this.mirrored ? 1 : 0;
    const X = this.x, Y = this.y, Z = this.z;
    for (let i = 0; i < n; i++) {
      const a = i > 0 ? i - 1 : 0, b = i < n - 1 ? i + 1 : n - 1;
      // across the strip: the line's direction crossed with the way to the eye
      let tx = X[b] - X[a], ty = Y[b] - Y[a], tz = Z[b] - Z[a];
      let tl = tx * tx + ty * ty + tz * tz;
      if (tl < 1e-14) {
        tx = 0; ty = 1; tz = 0; tl = 1;
      }
      tl = 1 / Math.sqrt(tl);
      tx *= tl; ty *= tl; tz *= tl;
      let cx = cam.x - X[i], cy = cam.y - Y[i], cz = cam.z - Z[i];
      const cl = Math.sqrt(cx * cx + cy * cy + cz * cz);
      if (cl > 0) {
        cx /= cl; cy /= cl; cz /= cl;
      }
      let qx = ty * cz - tz * cy, qy = tz * cx - tx * cz, qz = tx * cy - ty * cx;
      const ql = qx * qx + qy * qy + qz * qz, half = this.half[i];
      if (ql < 1e-10) {
        qx = half; qy = 0; qz = 0;
      } else {
        const s = half / Math.sqrt(ql);
        qx *= s; qy *= s; qz *= s;
      }
      const k = i * 6;
      P[k] = X[i] - qx; P[k + 1] = Y[i] - qy; P[k + 2] = Z[i] - qz;
      P[k + 3] = X[i] + qx; P[k + 4] = Y[i] + qy; P[k + 5] = Z[i] + qz;
      Nn[k] = Nn[k + 3] = this.nx[i]; Nn[k + 1] = Nn[k + 4] = this.ny[i]; Nn[k + 2] = Nn[k + 5] = this.nz[i];
      const f = i * 8;
      F[f] = F[f + 4] = this.tx[i]; F[f + 1] = F[f + 5] = this.ty[i]; F[f + 2] = F[f + 6] = this.tz[i]; F[f + 3] = F[f + 7] = mir;
      C[i * 2] = C[i * 2 + 1] = clamp(this.alpha[i], 0, 1);
      B[i * 2] = B[i * 2 + 1] = this.bead[i];
      S[i * 2] = -this.soft[i];
      S[i * 2 + 1] = this.soft[i];
    }
    this.pos.needsUpdate = this.nrm.needsUpdate = this.cov.needsUpdate = this.bed.needsUpdate = this.sid.needsUpdate = this.fib.needsUpdate = true;
    this.mesh.geometry.setDrawRange(0, Math.max(0, n - 1) * 6);
    this.mesh.visible = n > 1;
  }
}

/**
 * Light on a thin fibre from the shadowed sun (sky and bounce come from the standard model): diffuse round its axis
 * (lit side toward the light), the reflection off its surface (R) and the light that passes through it (TT). Both
 * leave on the cone round the axis that mirrors the light's angle to it, so a curving line lights in glints where it
 * turns onto that cone and stays a faint thread elsewhere; TT only toward the light: with the sun behind it the line
 * glows end to end. Close by the angler his own sharp shadow falls on it too. The reflection strip (aFib.w = 1) shades
 * the line as the water mirrors it, seen from the eye's mirror image.
 */
class FiberLighting extends THREE.PhysicalLightingModel {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  direct({ lightDirection, lightColor, reflectedLight, lightNode }: any) {
    const f = attribute('aFib', 'vec4');
    let E = lightColor;
    if (lightNode?.light?.isDirectionalLight === true) E = E.mul(charShadow(positionWorld, normalWorld));
    const up = normalize(cameraViewMatrix.mul(vec4(0, 1, 0, 0)).xyz);
    const v0 = positionViewDirection;
    const V = normalize(v0.sub(up.mul(dot(v0, up).mul(f.w.mul(2)))));
    const T = normalize(cameraViewMatrix.mul(vec4(f.xyz, 0)).xyz);
    const L = lightDirection;
    const sI = dot(L, T), sO = dot(V, T);
    const cI = sqrt(max(float(1).sub(sI.mul(sI)), 0));
    const lp = L.sub(T.mul(sI)), vp = V.sub(T.mul(sO));
    const cosPhi = dot(lp, vp).div(max(length(lp).mul(length(vp)), 1e-4));
    const u = sI.add(sO);
    const u2 = u.mul(u);
    const lobeR = exp(u2.mul(-1 / (2 * FIB.sigmaR * FIB.sigmaR))).mul(sqrt(saturate(cosPhi.mul(0.5).add(0.5))));
    const lobeT = exp(u2.mul(-1 / (2 * FIB.sigmaT * FIB.sigmaT))).mul(exp(cosPhi.add(1).mul(-1 / FIB.phiT)));
    reflectedLight.directDiffuse.addAssign(E.mul(BRDF_Lambert({ diffuseColor: diffuseContribution })).mul(cI.mul(cosPhi.mul(0.35).add(0.45))));
    reflectedLight.directSpecular.addAssign(E.mul(vec3(...FIB.tint).mul(lobeT.mul(FIB.kT)).add(lobeR.mul(FIB.kR))));
  }
}

class LineMaterial extends THREE.MeshStandardNodeMaterial {
  setupLightingModel() {
    return new FiberLighting();
  }
}

/** the float's paint: the sun through the angler's own sharp shadow as well (it hangs by him in the hold) */
class FloatLighting extends THREE.PhysicalLightingModel {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  direct(data: any, builder: any) {
    const sun = data.lightNode?.light?.isDirectionalLight === true;
    super.direct(sun ? { ...data, lightColor: data.lightColor.mul(charShadow(positionWorld, normalWorld)) } : data, builder);
  }
}

class FloatMaterial extends THREE.MeshStandardNodeMaterial {
  setupLightingModel() {
    return new FloatLighting();
  }
}

/**
 * The line: the verlet rope smoothed into a spline and drawn as a camera-facing ribbon, plus its reflection on still
 * water. Real line is barely there: its true coverage of the pixel (never quite lost), lit as a fibre, so it shows
 * where it catches the light and stays a faint thread elsewhere. Close to the lens it spreads into a faint blur
 * holding the same light; lying on the water it is fainter still and broken, with the little bright dents of its
 * meniscus; under the surface it is gone within centimetres; just over still water it meets its own reflection.
 */
class Line {
  readonly material: LineMaterial;
  readonly strip: Strip;
  readonly mirror: Strip;
  /** what the meniscus dents mirror: mostly sky, a fleck of sun */
  readonly beadColor = uniform(new THREE.Color(1, 1, 1));
  readonly sx = new Float32Array(MAX_SRC);
  readonly sy = new Float32Array(MAX_SRC);
  readonly sz = new Float32Array(MAX_SRC);
  /** water surface under each source point deep enough to mirror (-1e9: none) */
  readonly sw = new Float32Array(MAX_SRC);
  /** water surface under each source point where there is any water (-1e9: dry) */
  readonly sf = new Float32Array(MAX_SRC);
  srcCount = 0;
  /** distance (m) the picture is focused at; the line in front of it blurs */
  focus = 6;
  /** where along their pattern the curls of slack on the water start (a new one each cast) */
  phase = 0;
  private wy = new Float32Array(MAX_PTS);
  private arc = new Float64Array(MAX_PTS);
  private wf = new Float32Array(MAX_PTS);
  private sub = new Int32Array(MAX_SRC);
  private knot = new Float64Array(MAX_SRC);

  constructor() {
    const m = new LineMaterial({ transparent: true, depthWrite: false });
    // undyed silk, dressed and wet: a pale buff thread with a soft sheen
    m.colorNode = vec3(0.4, 0.37, 0.3);
    m.roughnessNode = float(0.45);
    m.metalnessNode = float(0);
    const side = attribute('aSide', 'float');
    m.opacityNode = attribute('aCov', 'float').mul(float(1).sub(side.mul(side)));
    m.emissiveNode = vec3(this.beadColor).mul(attribute('aBead', 'float'));
    this.material = m;
    this.strip = new Strip(m, 'fishing-line', false);
    this.mirror = new Strip(m, 'fishing-line-reflection', true);
  }

  /** smooth the source points (centripetal Catmull-Rom), then shade and upload both strips */
  build(camera: THREE.Camera, pixAngle: number) {
    const S = this.strip, n0 = this.srcCount;
    camera.getWorldPosition(_cam);
    const sx = this.sx, sy = this.sy, sz = this.sz;
    // ---- how finely to cut each rope segment: steps of ~STEP_PX on screen (meniscus dents and glints need them)
    let total = 1;
    const knot = this.knot;
    for (let i = 0; i < n0 - 1; i++) {
      const ex = sx[i + 1] - sx[i], ey = sy[i + 1] - sy[i], ez = sz[i + 1] - sz[i];
      const len = Math.sqrt(ex * ex + ey * ey + ez * ez);
      // centripetal parametrisation: the knot step is the square root of the chord
      knot[i] = Math.sqrt(Math.max(1e-6, len));
      const mx = (sx[i] + sx[i + 1]) / 2 - _cam.x, my = (sy[i] + sy[i + 1]) / 2 - _cam.y, mz = (sz[i] + sz[i + 1]) / 2 - _cam.z;
      const dc = Math.max(0.05, Math.sqrt(mx * mx + my * my + mz * mz));
      this.sub[i] = clamp(Math.ceil(len / (pixAngle * dc * STEP_PX)), 1, MAX_SUB);
      total += this.sub[i];
    }
    if (total > MAX_PTS) {
      const k = (MAX_PTS - 1) / (total - 1);
      for (let i = 0; i < n0 - 1; i++) this.sub[i] = Math.max(1, Math.floor(this.sub[i] * k));
    }
    // ---- spline through the source points
    let n = 0;
    // a repeated end point's knot step (the chord to itself, floored) and the nudge that keeps its knots apart
    const end = Math.sqrt(1e-6) + 1e-3;
    for (let i = 0; i < n0 - 1; i++) {
      const i0 = Math.max(0, i - 1), i3 = Math.min(n0 - 1, i + 2);
      const t1 = i0 === i ? end : knot[i - 1], t2 = t1 + knot[i], t3 = t2 + (i3 === i + 1 ? end : knot[i + 1]);
      const r1 = 1 / t1, r21 = 1 / (t2 - t1), r32 = 1 / (t3 - t2), r2 = 1 / t2, r31 = 1 / (t3 - t1);
      const sub = this.sub[i];
      const steps = i === n0 - 2 ? sub + 1 : sub;
      for (let s = 0; s < steps && n < MAX_PTS; s++) {
        const u = t1 + ((t2 - t1) * s) / sub;
        for (let c = 0; c < 3; c++) {
          const A = c === 0 ? sx : c === 1 ? sy : sz;
          const p0 = A[i0], p1 = A[i], p2 = A[i + 1], p3 = A[i3];
          const a1 = ((t1 - u) * p0 + u * p1) * r1;
          const a2 = ((t2 - u) * p1 + (u - t1) * p2) * r21;
          const a3 = ((t3 - u) * p2 + (u - t2) * p3) * r32;
          const b1 = ((t2 - u) * a1 + u * a2) * r2;
          const b2 = ((t3 - u) * a2 + (u - t1) * a3) * r31;
          (c === 0 ? S.x : c === 1 ? S.y : S.z)[n] = ((t2 - u) * b1 + (u - t1) * b2) * r21;
        }
        const f = s / sub;
        let wa = this.sw[i], wb = this.sw[i + 1];
        this.wy[n] = wa > -1e8 && wb > -1e8 ? wa + (wb - wa) * f : -1e9;
        wa = this.sf[i];
        wb = this.sf[i + 1];
        this.wf[n] = wa > -1e8 && wb > -1e8 ? wa + (wb - wa) * f : wa > -1e8 && f < 0.5 ? wa : wb > -1e8 && f >= 0.5 ? wb : -1e9;
        n++;
      }
    }
    S.count = n;
    this.wander(S, n);
    // ---- shade
    let s = 0;
    const M = this.mirror;
    let m = 0;
    for (let i = 0; i < n; i++) {
      let step = 0;
      if (i > 0) {
        const ex = S.x[i] - S.x[i - 1], ey = S.y[i] - S.y[i - 1], ez = S.z[i] - S.z[i - 1];
        step = Math.sqrt(ex * ex + ey * ey + ez * ez);
      }
      s += step;
      const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
      _t.set(S.x[b] - S.x[a], S.y[b] - S.y[a], S.z[b] - S.z[a]);
      if (_t.lengthSq() < 1e-14) _t.set(0, 1, 0);
      _t.normalize();
      _toCam.set(_cam.x - S.x[i], _cam.y - S.y[i], _cam.z - S.z[i]);
      const dist = Math.max(0.05, _toCam.length());
      _toCam.multiplyScalar(1 / dist);
      this.shade(S, i, dist, pixAngle, s);
      const wf = this.wf[i];
      const hf = wf > -1e8 ? S.y[i] - wf : 1e9;
      if (hf < -0.004) {
        // under the surface (the leader down to the hook): lost within a few centimetres of water
        S.alpha[i] *= Math.exp((hf + 0.004) / 0.012);
      } else if (hf < 0.012) {
        // floating: fainter, and broken where it sinks into the surface film; the meniscus dents along it catch
        // the sky in a row of tiny beads
        const k = 1 - smoothstepJs(0.004, 0.012, hf);
        const broken = smoothstepJs(0.35, 0.7, 0.5 + 0.32 * Math.sin(s * 2.3 + 0.6) + 0.22 * Math.sin(s * 6.1 + 2.0));
        S.alpha[i] *= 1 - k * (0.93 - 0.35 * broken);
        const dent = this.dentAt(s, Math.max(step, 1e-4));
        if (dent > 0) {
          const px = pixAngle * dist;
          const cover = Math.min(1, DENT / px) * Math.min(1, (DENT * 1.5) / Math.max(step, DENT * 1.5));
          S.bead[i] = dent * k;
          S.alpha[i] = Math.max(S.alpha[i], dent * k * cover);
          // a dimple is wider than the line: a dot close by, not a dash along it
          S.half[i] = Math.max(S.half[i], DENT * 0.5 * k);
        }
      }
      // the reflection: over still water, where the line hangs within a few metres of it
      const h = this.wy[i] > -1e8 ? S.y[i] - this.wy[i] : 1e9;
      if (h > 0.002 && h < 3 && _cam.y > this.wy[i] + 0.05) {
        const H = _cam.y - this.wy[i], kk = H / (H + h);
        const qx = _cam.x + (S.x[i] - _cam.x) * kk, qz = _cam.z + (S.z[i] - _cam.z) * kk;
        const ix = S.x[i] - _cam.x, iy = 2 * this.wy[i] - S.y[i] - _cam.y, iz = S.z[i] - _cam.z;
        const dm = Math.sqrt(ix * ix + iy * iy + iz * iz);
        const ox = qx - _cam.x, oz = qz - _cam.z;
        const dq = Math.sqrt(ox * ox + H * H + oz * oz);
        const cosT = H / Math.max(1e-3, dq);
        const fr = 0.02 + 0.98 * Math.pow(1 - cosT, 5);
        M.x[m] = qx; M.y[m] = this.wy[i] + 0.003; M.z[m] = qz;
        // as wide on screen as the line's image down there (the strip lies nearer, on the surface)
        const covT = (LINE_RADIUS * 2) / (pixAngle * dm);
        let wPx = Math.max(1, covT);
        let alpha = Math.max(Math.min(1, covT), FLOOR_NEAR * 0.6);
        const coc = COC_PX * Math.max(0, 1 / dm - 1 / this.focus);
        if (coc > wPx) {
          alpha *= wPx / coc;
          wPx = coc;
        }
        M.half[m] = 0.5 * wPx * pixAngle * dq;
        M.alpha[m] = alpha * fr * smoothstepJs(0.002, 0.008, h) * (1 - smoothstepJs(1.2, 3, h));
        M.soft[m] = S.soft[i];
        M.bead[m] = 0;
        M.tx[m] = S.tx[i]; M.ty[m] = S.ty[i]; M.tz[m] = S.tz[i];
        M.nx[m] = S.nx[i]; M.ny[m] = -S.ny[i]; M.nz[m] = S.nz[i];
        m++;
      } else if (m > 0 && M.alpha[m - 1] > 0) {
        // a gap: one invisible point closes the strip without a streak back to the line
        M.x[m] = M.x[m - 1]; M.y[m] = M.y[m - 1]; M.z[m] = M.z[m - 1];
        M.half[m] = M.half[m - 1];
        M.alpha[m] = 0;
        M.soft[m] = 0;
        M.bead[m] = 0;
        M.tx[m] = M.tx[m - 1]; M.ty[m] = M.ty[m - 1]; M.tz[m] = M.tz[m - 1];
        M.nx[m] = M.nx[m - 1]; M.ny[m] = M.ny[m - 1]; M.nz[m] = M.nz[m - 1];
        m++;
      }
    }
    M.count = m;
    S.upload(_cam);
    M.upload(_cam);
  }

  /**
   * Slack lying on the water keeps the curls it had on the spool: loose curves in the water plane along each stretch
   * of floating line, fading in from where it touches down and out to where it goes under at the float, so it still
   * meets both. (The rope's links only resist stretching: its slack on the water bunches up along it, ruler-straight.)
   */
  private wander(S: Strip, n: number) {
    const arc = this.arc, X = S.x, Y = S.y, Z = S.z;
    let s = 0;
    for (let i = 0; i < n; i++) {
      if (i > 0) {
        const ex = X[i] - X[i - 1], ey = Y[i] - Y[i - 1], ez = Z[i] - Z[i - 1];
        s += Math.sqrt(ex * ex + ey * ey + ez * ez);
      }
      arc[i] = s;
    }
    const floating = (i: number) => {
      const wf = this.wf[i], hf = Y[i] - wf;
      return wf > -1e8 && hf >= -0.004 && hf < 0.012;
    };
    for (let a = 0; a < n;) {
      if (!floating(a)) {
        a++;
        continue;
      }
      let b = a;
      while (b + 1 < n && floating(b + 1)) b++;
      const len = arc[b] - arc[a];
      let dx = X[b] - X[a], dz = Z[b] - Z[a];
      const dl = Math.sqrt(dx * dx + dz * dz);
      if (len > 0.3 && dl > 1e-3) {
        dx /= dl;
        dz /= dl;
        const ramp = Math.min(WANDER.ramp, len / 3), ph = this.phase;
        for (let i = a; i <= b; i++) {
          // along the line from the float, so the curls stay put on it as the float drifts
          const v = arc[b] - arc[i];
          const w = smoothstepJs(0, ramp, arc[i] - arc[a]) * smoothstepJs(0, ramp, v);
          const o = WANDER.amp * w * (0.75 * Math.sin(v * 2.6 + ph) + 0.25 * Math.sin(v * 6.1 + 2.1 + ph * 1.7));
          X[i] -= dz * o;
          Z[i] += dx * o;
        }
      }
      a = b + 1;
    }
  }

  /**
   * How much of a meniscus dent the stretch of floating line round arc length s holds (0 none .. 1): dents ~DENT_GAP
   * apart, unevenly spaced and bright, a third of them missing, a few millimetres each.
   */
  private dentAt(s: number, step: number) {
    const k = Math.round(s / DENT_GAP);
    let best = 0;
    for (let j = k - 1; j <= k + 1; j++) {
      const h = Math.sin(j * 127.1 + 31.7) * 43758.5453;
      const r = h - Math.floor(h);
      if (r < 0.35) continue;
      const h2 = Math.sin(j * 311.7 + 74.7) * 43758.5453;
      const r2 = h2 - Math.floor(h2);
      const c = (j + (r2 - 0.5) * 0.9) * DENT_GAP;
      const dd = Math.abs(s - c);
      if (dd < step * 0.5 + DENT * 0.5) best = Math.max(best, 0.25 + 0.75 * ((r - 0.35) / 0.65) ** 2);
    }
    return best;
  }

  private shade(S: Strip, i: number, dist: number, pixAngle: number, s: number) {
    const px = pixAngle * dist;
    const covT = (LINE_RADIUS * 2) / px;
    let wPx = Math.max(1, covT);
    let alpha = Math.max(Math.min(1, covT), FLOOR_NEAR + (FLOOR_FAR - FLOOR_NEAR) * smoothstepJs(6, 30, dist));
    // in front of the focus the line blurs into a soft, wide, faint band: the same light spread over the blur circle
    const coc = COC_PX * Math.max(0, 1 / dist - 1 / this.focus);
    if (coc > wPx) {
      alpha *= wPx / coc;
      wPx = coc;
    }
    S.half[i] = 0.5 * wPx * px;
    S.alpha[i] = alpha;
    S.soft[i] = 0.92 * smoothstepJs(1.5, 5, wPx);
    S.bead[i] = 0;
    // the line's memory: a slow, uneven wobble of its direction (across the view and in depth), so a glint lights a
    // stretch of it and breaks off
    const tw1 = Math.sin(s * 13.7 + 1.7) * 0.022 + Math.sin(s * 5.3 + 0.4) * 0.018 + Math.sin(s * 31.0 + 2.2) * 0.01;
    const tw2 = Math.sin(s * 9.1 + 0.3) * 0.02 + Math.sin(s * 3.7 + 2.9) * 0.016;
    _w.copy(_t);
    _n.crossVectors(_t, _toCam);
    if (_n.lengthSq() > 1e-10) {
      _n.normalize();
      _n2.crossVectors(_n, _t);
      _w.addScaledVector(_n, tw1).addScaledVector(_n2, tw2).normalize();
    }
    S.tx[i] = _w.x; S.ty[i] = _w.y; S.tz[i] = _w.z;
    // the normal the sky lights it by: across the fibre, between up and the eye
    _n.set(_toCam.x, _toCam.y + 1, _toCam.z);
    _n.addScaledVector(_w, -_n.dot(_w));
    if (_n.lengthSq() < 1e-8) _n.copy(_toCam).addScaledVector(_w, -_toCam.dot(_w));
    _n.normalize();
    S.nx[i] = _n.x; S.ny[i] = _n.y; S.nz[i] = _n.z;
  }
}

/** a red-and-white cork float with a quill, one mesh: origin at its middle, the waterline set by the sim's sink */
function buildBobber(): THREE.Group {
  const r = BOB_R;
  const pts: THREE.Vector2[] = [];
  for (let i = 0; i <= 12; i++) {
    const a = (i / 12) * Math.PI;
    const y = -Math.cos(a) * r * 1.15;
    // a little pear-shaped: fuller on top
    const rad = Math.sin(a) * r * (y > 0 ? 1.0 : 0.82);
    pts.push(new THREE.Vector2(Math.max(rad, 0.0015), y));
  }
  const body = new THREE.LatheGeometry(pts, 16);
  const quill = new THREE.CylinderGeometry(0.0021, 0.003, 0.05, 7, 4);
  quill.translate(0, r * 1.15 + 0.022, 0);
  const paint = (g: THREE.BufferGeometry, isQuill: boolean) => {
    const p = g.getAttribute('position');
    const col = new Float32Array(p.count * 3), rough = new Float32Array(p.count);
    for (let i = 0; i < p.count; i++) {
      const y = p.getY(i);
      let c: [number, number, number], ro: number;
      if (isQuill) {
        // a varnished quill with its sight tip painted red
        const tip = y > r * 1.15 + 0.034;
        c = tip ? [0.72, 0.05, 0.03] : [0.5, 0.42, 0.25];
        ro = tip ? 0.24 : 0.35;
      } else {
        const band = Math.abs(y - 0.002) < 0.0035;
        const red = y > 0.002;
        // old enamel on cork: a deep vermilion cap, the lower half an ivory gone a little yellow
        c = band ? [0.05, 0.04, 0.03] : red ? [0.72, 0.045, 0.028] : [0.6, 0.57, 0.5];
        // wet gloss paint
        ro = red ? 0.22 : 0.36;
      }
      col.set(c, i * 3);
      rough[i] = ro;
    }
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aRough', new THREE.BufferAttribute(rough, 1));
    g.deleteAttribute('uv');
  };
  paint(body, false);
  paint(quill, true);
  const merged = mergeGeometries([body.toNonIndexed(), quill.toNonIndexed()])!;
  const mat = new FloatMaterial({ vertexColors: true });
  mat.roughnessNode = attribute('aRough', 'float');
  const g = new THREE.Group();
  g.name = 'fishing-bobber';
  const a = new THREE.Mesh(merged, mat);
  a.receiveShadow = true;
  g.add(a);
  return g;
}
const BOB_R = 0.022;
/** the float's radius at height y above its middle (lathe profile) */
const bobberRadius = (y: number) => {
  const c = clamp(-y / (BOB_R * 1.15), -1, 1);
  const rad = Math.sqrt(1 - c * c) * BOB_R;
  return y > 0 ? rad : rad * 0.82;
};

/**
 * What the float does to the water round it: a thin bright collar of meniscus where it breaks the surface, and its
 * reflection (the painted cap and the quill mirrored in the still water toward the eye). Both lie on the surface:
 * one small strip mesh rebuilt each frame for the camera.
 */
class FloatWater {
  readonly mesh: THREE.Mesh;
  private pos: THREE.BufferAttribute;
  private col: THREE.BufferAttribute;
  private static readonly RING = 20;
  private static readonly REFL = [0, 0.25, 0.55, 0.9, 1.0];
  private readonly verts = FloatWater.RING * 2 + 10 + 8;

  constructor() {
    const g = new THREE.BufferGeometry();
    this.pos = new THREE.BufferAttribute(new Float32Array(this.verts * 3), 3);
    this.col = new THREE.BufferAttribute(new Float32Array(this.verts * 4), 4);
    this.pos.setUsage(THREE.DynamicDrawUsage);
    this.col.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', this.pos);
    g.setAttribute('aCol', this.col);
    const idx: number[] = [];
    const R = FloatWater.RING;
    for (let i = 0; i < R; i++) {
      const a = i * 2, b = ((i + 1) % R) * 2;
      idx.push(a, b, a + 1, a + 1, b, b + 1);
    }
    const base = R * 2;
    for (let i = 0; i < FloatWater.REFL.length - 1; i++) {
      const a = base + i * 2;
      idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
    g.setIndex(idx);
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e5);
    const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
    const c = attribute('aCol', 'vec4');
    m.colorNode = c.rgb;
    m.opacityNode = c.a;
    this.mesh = new THREE.Mesh(g, m);
    this.mesh.name = 'fishing-float-water';
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 2;
    this.mesh.layers.set(LAYER_NO_REFLECT);
  }

  /**
   * Float at (x, surfaceY, z) sunk `sink` m (its middle below the surface), drawn `scale` x; `lit`: its cap's lit
   * colour, `sky`: what the meniscus mirrors.
   */
  update(x: number, surfaceY: number, z: number, sink: number, scale: number, cam: THREE.Vector3, lit: THREE.Color, sky: THREE.Color) {
    const P = this.pos.array as Float32Array, C = this.col.array as Float32Array;
    const R = FloatWater.RING;
    const y = surfaceY + 0.004;
    const rw = Math.max(0.003, bobberRadius(sink / scale) * scale);
    for (let i = 0; i < R; i++) {
      const a = (i / R) * Math.PI * 2, ca = Math.cos(a), sa = Math.sin(a);
      // the meniscus climbs the float: bright where it faces the eye's reflection of the sky
      const face = 0.6 + 0.4 * Math.max(0, -(ca * (cam.x - x) + sa * (cam.z - z)) / (Math.hypot(cam.x - x, cam.z - z) || 1));
      P.set([x + ca * rw * 0.96, y, z + sa * rw * 0.96], i * 6);
      P.set([x + ca * (rw + 0.0065 * scale), y, z + sa * (rw + 0.0065 * scale)], i * 6 + 3);
      C.set([sky.r * 1.3, sky.g * 1.3, sky.b * 1.3, 0.55 * face], i * 8);
      C.set([sky.r, sky.g, sky.b, 0], i * 8 + 4);
    }
    // the reflection, from the waterline toward the eye: each height h above the water shows at D h / (H + h)
    const H = cam.y - surfaceY, D = Math.hypot(cam.x - x, cam.z - z);
    const fx = (cam.x - x) / (D || 1), fz = (cam.z - z) / (D || 1);
    const capTop = BOB_R * 1.15 * scale - sink, quillTop = capTop + 0.05 * scale;
    const cosT = H / Math.max(1e-3, Math.hypot(D, H));
    const fr = H > 0.02 ? Math.min(0.6, (0.02 + 0.98 * Math.pow(1 - cosT, 5)) * 1.3) : 0;
    const base = R * 2;
    FloatWater.REFL.forEach((u, k) => {
      const onCap = u < 0.9;
      const h = onCap ? Math.max(0, capTop) * (u / 0.9) : quillTop;
      const along = H > 0.02 ? (D * h) / (H + h) : 0;
      const w = onCap ? bobberRadius(sink / scale + h / scale) * scale : 0.0022 * scale;
      const px = x + fx * along, pz = z + fz * along;
      P.set([px - fz * w, y + 0.0005, pz + fx * w], (base + k * 2) * 3);
      P.set([px + fz * w, y + 0.0005, pz - fx * w], (base + k * 2 + 1) * 3);
      const a = fr * (k === FloatWater.REFL.length - 1 ? 0.25 : 0.85);
      const cr = onCap ? lit.r : lit.r * 0.75, cg = onCap ? lit.g : lit.g * 4, cb = onCap ? lit.b : lit.b * 4;
      C.set([cr, cg, cb, a], (base + k * 2) * 4);
      C.set([cr, cg, cb, a], (base + k * 2 + 1) * 4);
    });
    this.pos.needsUpdate = this.col.needsUpdate = true;
  }
}

/**
 * A fish just under the surface seen through clear water: a dark, soft-edged trout shape lying on the surface over it.
 * The water refracts what lies under it as if it were on the bed, so a fish held a few centimetres down would show the
 * stones in its place; this is what the eye reads instead (darker the nearer the surface, gone below ~0.7 m).
 */
class FishShade {
  readonly mesh: THREE.Mesh;
  readonly strength = uniform(0);

  constructor() {
    const g = new THREE.PlaneGeometry(1, 1);
    g.rotateX(-Math.PI / 2);
    const m = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
    // nose at uv.y = 1 (local -z), tail at 0: a spindle, fullest a third back from the nose, the tail wrist narrow
    const t = uv();
    const s = float(1).sub(t.y);
    const half = s.pow(0.62).mul(Math.PI).sin().mul(0.5).mul(float(1).sub(smoothstep(0.86, 1.0, s)).mul(0.75).add(0.25)).max(0.04);
    const across = t.x.sub(0.5).abs().mul(2);
    const body = float(1).sub(smoothstep(half.mul(0.55), half, across));
    m.colorNode = vec3(0.012, 0.016, 0.014);
    m.opacityNode = body.mul(this.strength).mul(smoothstep(0.0, 0.06, t.y)).mul(float(1).sub(smoothstep(0.97, 1.0, t.y)));
    this.mesh = new THREE.Mesh(g, m);
    this.mesh.name = 'fishing-fish-shade';
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 2;
    this.mesh.layers.set(LAYER_NO_REFLECT);
    this.mesh.visible = false;
  }

  /** fish centre, nose direction (+z of its frame), length (m), the water surface over it */
  update(pos: THREE.Vector3, quat: THREE.Quaternion, length: number, surfaceY: number) {
    const depth = surfaceY - pos.y;
    const k = depth > 0.015 ? 1 - smoothstepJs(0.12, 0.7, depth) : 0;
    this.mesh.visible = k > 0.01;
    if (!this.mesh.visible) return;
    _w.set(0, 0, 1).applyQuaternion(quat);
    this.mesh.position.set(pos.x, surfaceY + 0.012, pos.z);
    this.mesh.rotation.set(0, Math.atan2(-_w.x, -_w.z), 0);
    this.mesh.scale.set(length * 0.25, 1, length * 1.02);
    this.strength.value = 0.62 * k;
  }
}

/** dotted flight arc and a landing ring on the water while aiming, veiled by the mist with distance like the far bank */
class AimPreview {
  readonly dots: THREE.Sprite;
  readonly ring: THREE.Mesh;
  private data = new Float32Array(AIM_POINTS * 4);
  private attr: THREE.InstancedBufferAttribute;
  readonly ringColor = uniform(new THREE.Color(1, 0.94, 0.8));
  readonly ringAlpha = uniform(0.6);
  /** the mist's colour the marks fade into */
  readonly veil = uniform(new THREE.Color(0.55, 0.57, 0.56));

  constructor() {
    this.attr = new THREE.InstancedBufferAttribute(this.data, 4);
    this.attr.setUsage(THREE.DynamicDrawUsage);
    const m = new THREE.SpriteNodeMaterial({ transparent: true, depthWrite: false });
    const p = instancedDynamicBufferAttribute(this.attr);
    m.positionNode = p.xyz;
    m.scaleNode = vec2(p.w, p.w);
    const d = length(uv().sub(0.5)).mul(2);
    const k = exp(length(p.xyz.sub(cameraPosition)).mul(-0.13));
    m.colorNode = vec4(mix(vec3(this.veil), vec3(0.95, 0.9, 0.78).mul(0.75), k), float(1).sub(smoothstep(0.3, 1, d)).mul(k.mul(0.32).add(0.06)));
    this.dots = new THREE.Sprite(m);
    this.dots.count = AIM_POINTS;
    this.dots.frustumCulled = false;
    this.dots.renderOrder = 4;
    this.dots.layers.set(LAYER_NO_REFLECT);
    // a soft-edged ring: a disc whose alpha is a band round 0.26 m
    const rg = new THREE.CircleGeometry(0.36, 48);
    rg.rotateX(-Math.PI / 2);
    const rm = new THREE.MeshBasicNodeMaterial({ transparent: true, depthWrite: false });
    const r = length(uv().sub(0.5)).mul(0.72);
    const band = exp(r.sub(0.265).div(0.03).pow(2).negate());
    const kr = exp(float(0).sub(this.ringFadeU));
    rm.colorNode = vec4(mix(vec3(this.veil), vec3(this.ringColor).mul(0.85), kr), band.mul(this.ringAlpha).mul(kr.mul(0.8).add(0.2)));
    this.ring = new THREE.Mesh(rg, rm);
    this.ring.renderOrder = 4;
    this.ring.frustumCulled = false;
    this.ring.layers.set(LAYER_NO_REFLECT);
  }

  /** the ring's mist veil: its distance (m) x 0.13, set on the CPU (one ring) */
  private readonly ringFadeU = uniform(0);

  update(sim: FishingSim, camera: THREE.Camera, t: number, force: boolean) {
    const n = force && sim.aimCount === 0 ? 0 : sim.aimCount;
    const on = n > 1 || force;
    this.dots.visible = this.ring.visible = on;
    if (!on) return;
    camera.getWorldPosition(_cam);
    const d = this.data;
    // the track starts where it leaves the bank: dots on the stones at his feet stood in a column under the tip
    let first = 0;
    while (first < n - 2 && !sim.aimWet[first]) first++;
    for (let i = 0; i < AIM_POINTS; i++) {
      if (i < n - 1 && i >= first) {
        // the track on the water: dots grow with distance, softening as the mist takes them, and fade in over the
        // first metre or two
        const x = sim.aimPts[i * 3], y = sim.aimPts[i * 3 + 1], z = sim.aimPts[i * 3 + 2];
        const dist = Math.hypot(x - _cam.x, y - _cam.y, z - _cam.z);
        d[i * 4] = x; d[i * 4 + 1] = y; d[i * 4 + 2] = z;
        d[i * 4 + 3] = (0.012 + dist * 0.0042) * Math.min(1, 0.35 + (i - first) * 0.25);
      } else if (force && i >= n - 1) {
        // warm-up: something in front of the camera so the pipeline compiles
        camera.getWorldDirection(_p);
        d[i * 4] = _cam.x + _p.x * (3 + i * 0.2); d[i * 4 + 1] = _cam.y + _p.y * 3; d[i * 4 + 2] = _cam.z + _p.z * (3 + i * 0.2);
        d[i * 4 + 3] = 0.03;
      } else {
        d[i * 4 + 3] = 0;
      }
    }
    this.attr.needsUpdate = true;
    if (n > 1) this.ring.position.set(sim.aimLanding.x, sim.aimLanding.y + 0.02, sim.aimLanding.z);
    else this.ring.position.set(d[0], d[1], d[2]);
    const dist = this.ring.position.distanceTo(_cam);
    this.ring.scale.setScalar(1 + Math.hypot(this.ring.position.x - _cam.x, this.ring.position.z - _cam.z) * 0.02);
    this.ringFadeU.value = dist * 0.13;
    if (sim.aimOnWater || force) this.ringColor.value.setRGB(1, 0.94, 0.8);
    else this.ringColor.value.setRGB(0.75, 0.42, 0.3);
    this.ringAlpha.value = 0.42 + 0.12 * Math.sin(t * 5);
  }
}

/** Every mesh of the fishing loop except the rod (owned by its rig) and the HUD. */
export class FishingView {
  readonly group = new THREE.Group();
  readonly line = new Line();
  readonly bobber = buildBobber();
  readonly floatWater = new FloatWater();
  readonly fishShade = new FishShade();
  readonly aim = new AimPreview();
  visible = true;
  /** photo poses set the distance the picture is focused at (the line in front of it blurs); 0 = the float's */
  focus = 0;
  /** false: the line's strips are left as they are (a photo pose's scripted run, whose last frame alone is seen) */
  drawLine = true;
  private size = new THREE.Vector2();
  /** undefined = unknown (after the warm-up frames): the next frame sets every fish's visibility */
  private shownSpecies: SpeciesId | null | undefined = undefined;
  private readonly lit = new THREE.Color();
  private readonly sky = new THREE.Color();

  constructor(
    private renderer: THREE.WebGPURenderer,
    private rod: RodRig,
    /** part 6's own trout (null: part 7's school draws the hooked and held trout) */
    readonly fish: FishModels | null,
    /** live: part 8 swaps the sun for the moon and fades its colour and strength */
    private sun: { dir: THREE.Vector3; color: THREE.Color; readonly intensity: number },
    /** false: the rod belongs to the character (its visibility and, while the fishing is hidden, its bend) */
    private ownsRod = true,
  ) {
    this.group.name = 'fishing';
    this.group.add(this.line.strip.mesh, this.line.mirror.mesh, this.bobber, this.floatWater.mesh, this.fishShade.mesh, this.aim.dots, this.aim.ring);
    if (fish) for (const id of SPECIES_IDS) this.group.add(fish.get(id).root);
    this.followSun();
  }

  /** the float's cap as the sun and sky light it (for its reflection), the sky the meniscus mirrors, the line's dents */
  private followSun() {
    const sun = this.sun;
    const e = sun.intensity * Math.max(0.1, sun.dir.y) * 0.32;
    this.lit.setRGB(0.72, 0.045, 0.028).multiply(sun.color).multiplyScalar(e).addScalar(0.006);
    this.sky.copy(sun.color).multiplyScalar(sun.intensity * 0.05);
    // a dent in the meniscus is a tiny curved mirror: mostly the bright sky, and some sun (the sky's share follows
    // the light, so the dents dim with part 8's dusk and moon)
    this.line.beadColor.value.copy(sun.color).multiplyScalar(sun.intensity * 0.1).addScalar(sun.intensity * 0.0376);
  }

  /**
   * Apply the simulation to the meshes for this camera. `warm`: the loading screen is up, show everything so every
   * pipeline compiles before the first real frame.
   */
  sync(sim: FishingSim, camera: THREE.PerspectiveCamera, t: number, warm: boolean) {
    const vis = this.visible || warm;
    this.group.visible = vis;
    if (this.ownsRod) this.rod.root.visible = vis;
    else if (!vis) {
      sim.crankDelta = 0;
      return;
    }
    // the rod: bend toward the line, crank, then the tip where the line starts this frame
    this.rod.setBend(sim.bend, sim.bendDir);
    if (sim.crankDelta) this.rod.crank(sim.crankDelta);
    sim.crankDelta = 0;
    this.rod.root.updateMatrixWorld(true);
    this.rod.getTip(_tip);
    if (!vis) return;
    camera.getWorldPosition(_cam);

    // the float rides the water, leans toward the pull; scaled up with distance so it stays readable (a speck at
    // 15 m, as real floats are), its waterline where the sim puts it
    const scale = clamp(_cam.distanceTo(sim.bobber) / 6, 1, 2.6);
    const riding = sim.onWater && (sim.state === 'floating' || sim.state === 'bite');
    const surface = sim.bobber.y + (riding ? sim.bobberSink : 0);

    // line: rope (tip -> float), then the leader (float -> hook) where it differs
    const rope = sim.rope, L = this.line;
    L.sx[0] = _tip.x; L.sy[0] = _tip.y; L.sz[0] = _tip.z;
    L.sw[0] = L.sf[0] = -1e9;
    let n = 1;
    for (let i = 1; i < rope.n && n < MAX_SRC; i++, n++) {
      L.sx[n] = rope.x[i]; L.sy[n] = rope.y[i]; L.sz[n] = rope.z[i];
      L.sw[n] = rope.waterY(i);
      L.sf[n] = rope.surfaceY(i);
    }
    const leader = sim.state === 'floating' || sim.state === 'bite' || sim.state === 'fight';
    if (leader) {
      const ex = rope.x[rope.n - 1], ey = rope.y[rope.n - 1], ez = rope.z[rope.n - 1];
      // under the float the leader goes down into the water: the surface there hides it
      const below = sim.onWater || sim.state === 'fight' ? (sim.state === 'fight' ? sim.fishSurfaceY : surface) : -1e9;
      for (let k = 1; k <= LEADER_PTS && n < MAX_SRC; k++, n++) {
        const f = k / LEADER_PTS;
        L.sx[n] = ex + (sim.hook.x - ex) * f;
        L.sy[n] = ey + (sim.hook.y - ey) * f;
        L.sz[n] = ez + (sim.hook.z - ez) * f;
        L.sw[n] = -1e9;
        L.sf[n] = below;
      }
    }
    L.srcCount = n;
    L.focus = this.focus > 0 ? this.focus : clamp(_cam.distanceTo(sim.bobber), 2, 40);
    L.phase = sim.castPower * 37.1;
    this.followSun();
    this.renderer.getDrawingBufferSize(this.size);
    const pixAngle = 2 * Math.tan((camera.fov * Math.PI) / 360) / Math.max(1, this.size.y);
    if (this.drawLine) L.build(camera, pixAngle);

    const b = this.bobber;
    b.visible = sim.bobberVisible || warm;
    b.position.copy(sim.bobber);
    if (riding) b.position.y = surface - sim.bobberSink * scale;
    b.rotation.set(sim.bobberTilt.y * 0.9, 0, -sim.bobberTilt.x * 0.9);
    b.scale.setScalar(scale);
    const floating = b.visible && riding;
    this.floatWater.mesh.visible = floating || warm;
    if (floating || warm) this.floatWater.update(sim.bobber.x, surface, sim.bobber.z, sim.bobberSink * scale, scale, _cam, this.lit, this.sky);

    this.aim.update(sim, camera, t, warm);

    // fish: the hooked species only (all four during warm-up); part 7's school draws its own trout under the water
    if (!this.fish) {
      this.fishShade.mesh.visible = false;
      return;
    }
    const f = sim.fish;
    const show: SpeciesId | null = f.visible && f.mode !== 'none' ? f.species.id : null;
    if (show !== this.shownSpecies || warm) {
      for (const id of SPECIES_IDS) this.fish.get(id).root.visible = warm || id === show;
      this.shownSpecies = warm ? undefined : show;
    }
    if (warm) {
      camera.getWorldDirection(_p);
      for (let i = 0; i < SPECIES_IDS.length; i++) {
        const r = this.fish.get(SPECIES_IDS[i]).root;
        r.position.set(_cam.x + _p.x * 2 + (i - 1.5) * 0.4, _cam.y + _p.y * 2 - 0.2, _cam.z + _p.z * 2);
      }
    } else if (show) {
      const m = this.fish.get(show);
      if (f.placed) {
        m.root.position.copy(f.pos);
        m.root.quaternion.copy(f.quat);
        m.root.scale.setScalar(f.scale);
      }
      m.phase.value = f.swimPhase;
      m.amp.value = f.swimAmp;
      m.sag.value = f.sag;
      m.setShadow(f.inAir || sim.state === 'landing' || sim.state === 'holding' || sim.state === 'releasing');
    }
    // the fish just under the surface, as the eye sees it through the water
    const under = show && !warm && f.placed && (sim.state === 'releasing' || ((sim.state === 'fight' || f.mode === 'free') && !f.inAir));
    if (under) this.fishShade.update(f.pos, f.quat, f.lengthCm / 100, sim.fishSurfaceY);
    else this.fishShade.mesh.visible = warm;
  }
}
