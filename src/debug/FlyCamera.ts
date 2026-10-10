import * as THREE from 'three/webgpu';

/**
 * Debug free-fly camera (?fly): click the canvas to lock the mouse, WASD to move, E/Space up, Q/C down,
 * Shift to go faster, Esc to release. The real third-person controls arrive with the character (part 5).
 */
export class FlyCamera {
  private keys = new Set<string>();
  private yaw: number;
  private pitch: number;
  private readonly fwd = new THREE.Vector3();
  private readonly right = new THREE.Vector3();
  private readonly euler = new THREE.Euler(0, 0, 0, 'YXZ');

  constructor(private camera: THREE.PerspectiveCamera, dom: HTMLElement) {
    this.euler.setFromQuaternion(camera.quaternion, 'YXZ');
    this.yaw = this.euler.y;
    this.pitch = this.euler.x;
    dom.addEventListener('click', () => {
      if (document.pointerLockElement !== dom) dom.requestPointerLock();
    });
    document.addEventListener('mousemove', (ev) => {
      if (document.pointerLockElement !== dom) return;
      this.yaw -= ev.movementX * 0.0022;
      this.pitch = THREE.MathUtils.clamp(this.pitch - ev.movementY * 0.0022, -1.55, 1.55);
    });
    window.addEventListener('keydown', (ev) => this.keys.add(ev.code));
    window.addEventListener('keyup', (ev) => this.keys.delete(ev.code));
    window.addEventListener('blur', () => this.keys.clear());
  }

  /** re-read yaw/pitch after something else moved the camera (a jump) */
  syncFromCamera() {
    this.euler.setFromQuaternion(this.camera.quaternion, 'YXZ');
    this.yaw = this.euler.y;
    this.pitch = this.euler.x;
  }

  update(dt: number) {
    const k = this.keys;
    this.euler.set(this.pitch, this.yaw, 0, 'YXZ');
    this.camera.quaternion.setFromEuler(this.euler);
    const speed = (k.has('ShiftLeft') || k.has('ShiftRight') ? 14 : 3.5) * dt;
    this.fwd.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
    this.right.set(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const p = this.camera.position;
    if (k.has('KeyW')) p.addScaledVector(this.fwd, speed);
    if (k.has('KeyS')) p.addScaledVector(this.fwd, -speed);
    if (k.has('KeyD')) p.addScaledVector(this.right, speed);
    if (k.has('KeyA')) p.addScaledVector(this.right, -speed);
    if (k.has('KeyE') || k.has('Space')) p.y += speed;
    if (k.has('KeyQ') || k.has('KeyC')) p.y -= speed;
  }
}
