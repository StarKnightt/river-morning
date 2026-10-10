/**
 * Loosely typed TSL entry point for shader graphs. @types/three types node graphs very
 * strictly (vec3 vs color unions etc.), which is noise for shader code; game logic stays typed.
 */
import * as TSLTyped from 'three/tsl';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const TSL = TSLTyped as any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type N = any;

export const {
  abs, attribute, builtinAOContext, cameraPosition, cameraViewMatrix, clamp, color, cos, dFdx, dFdy, distance, dot,
  exp, float, floor, fog, fract, Fn, fwidth, hash, If, instanceIndex, int, length, log, Loop, max, min, mix, mod, mrt,
  mx_fractal_noise_float, mx_noise_float, mx_noise_vec3, mx_worley_noise_float, normalize, normalView, normalWorld,
  output, pass, positionLocal, positionView, positionWorld, pow, reflect, refract, renderOutput, saturate, screenUV,
  select, sign, sin, smoothstep, sqrt, step, texture, time, uniform, uv, vec2, vec3, vec4, viewportDepthTexture,
  viewportLinearDepth, viewportSharedTexture, linearDepth, reflector, transformNormalToView, modelWorldMatrix,
  cameraProjectionMatrix, cameraNear, cameraFar, perspectiveDepthToViewZ, getViewPosition, screenSize, varying,
  vertexIndex, uniformArray, cross, atan, asin, acos, PI, TWO_PI, oneMinus, positionGeometry, normalLocal, objectPosition,
  cameraProjectionMatrixInverse, viewportUV, emissive, depth, backgroundRotation, equirectUV, normalWorldGeometry,
  positionWorldDirection, viewportOpaqueMipTexture, pmremTexture, getScreenPosition, getScreenPositionFromClip,
  instancedDynamicBufferAttribute, inverseSqrt, luminance, normalMap, log2, exp2, BRDF_Lambert, diffuseColor,
  normalViewGeometry, positionViewDirection, faceDirection, viewportSafeUV, interleavedGradientNoise, screenCoordinate,
  lightShadowMatrix, rtt, texture3D, diffuseContribution, modelWorldMatrixInverse,
} = TSL;
