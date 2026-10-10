import type * as THREE from 'three/webgpu';
import type { N } from './tsl';

/**
 * Albedo for an opaque node material, kept out of its shadow pass. three multiplies the shadow-pass alpha by
 * `colorNode.a`, which compiled the whole colour graph into each shadow shader (the core terrain's: 36 KB, 1.5 s).
 * The node is the material's `colorNode` only while the material builds its own shader.
 */
export function setOpaqueAlbedo(m: THREE.Material, albedo: N) {
  const mm = m as unknown as { colorNode: N; setupDiffuseColor: (builder: unknown) => void };
  const setup = mm.setupDiffuseColor;
  mm.colorNode = null;
  mm.setupDiffuseColor = function (this: typeof mm, builder: unknown) {
    this.colorNode = albedo;
    try {
      setup.call(this, builder);
    } finally {
      this.colorNode = null;
    }
  };
}
