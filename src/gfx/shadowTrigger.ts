import * as THREE from 'three/webgpu';
import { LAYER_TRIGGER } from '../core/layers';

/**
 * Redraws a light's shadow map outside the frame's own passes. three draws a shadow map from inside the first render
 * of the scene that shades a receiver with that light while `shadow.needsUpdate` is set (one shadow node per light,
 * shared by every render of the scene). This draws one tiny receiver, which only its own camera and the light see,
 * into a 4 x 4 target: the shadow pass is the whole cost. If it ever drew nothing, needsUpdate would stay set and the
 * next frame's scene pass would redraw the map as before.
 */
/** redraws asked of the trigger, and how many left `needsUpdate` set (the scene pass then redrew the map) */
export const shadowTriggerStats = { redraws: 0, misses: 0 };

export class ShadowTrigger {
  readonly mesh: THREE.Mesh;
  private readonly cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  private readonly rt = new THREE.RenderTarget(4, 4);

  constructor(light: THREE.Light) {
    light.layers.enable(LAYER_TRIGGER);
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshStandardNodeMaterial({ roughness: 1 }));
    this.mesh.name = 'shadow-trigger';
    this.mesh.rotation.x = -Math.PI / 2;
    this.mesh.position.set(0, -500, 0);
    this.mesh.layers.set(LAYER_TRIGGER);
    this.mesh.receiveShadow = true;
    this.mesh.castShadow = false;
    this.mesh.frustumCulled = false;
    this.cam.position.set(0, -499, 0);
    this.cam.lookAt(0, -500, 0);
    this.cam.layers.set(LAYER_TRIGGER);
    this.cam.updateMatrixWorld();
  }

  render(renderer: THREE.WebGPURenderer, scene: THREE.Scene) {
    const prevRT = renderer.getRenderTarget(), prevMRT = renderer.getMRT();
    renderer.setMRT(null);
    renderer.setRenderTarget(this.rt);
    renderer.render(scene, this.cam);
    renderer.setRenderTarget(prevRT);
    renderer.setMRT(prevMRT);
  }
}
