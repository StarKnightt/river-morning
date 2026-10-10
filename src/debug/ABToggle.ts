import type * as THREE from 'three/webgpu';
import type { Engine } from '../core/Engine';

/**
 * ?abhide=name1,name2: GPU cost of named scene objects in one session and one view (the camp's ?campab for any
 * object). They are shown for 6 s and hidden for 6 s, over and over; after a warm cycle each phase's mean GPU frame
 * time (the engine's rolling GPU timestamps) is logged with console.warn, so shot reports keep it.
 */
export function installABToggle(engine: Engine, names: string[]) {
  const objs: THREE.Object3D[] = [];
  engine.scene.traverse((o) => {
    if (names.includes(o.name)) objs.push(o);
  });
  if (!objs.length) return;
  const T = 6000;
  let t0 = -1, phase = -1;
  const on: number[] = [], off: number[] = [];
  let lx = 1e9, ly = 0, lz = 0;
  const show = (v: boolean) => objs.forEach((o) => (o.visible = v));
  engine.addUpdater(() => {
    const now = performance.now();
    const p = engine.camera.position;
    if (Math.abs(p.x - lx) + Math.abs(p.y - ly) + Math.abs(p.z - lz) > 0.01) {
      // a new view: start over, shown first (a screenshot taken right after the move includes everything)
      [lx, ly, lz] = [p.x, p.y, p.z];
      t0 = now + 2000;
      phase = -1;
      on.length = off.length = 0;
      show(true);
    }
    if (now < t0) return;
    const k = Math.floor((now - t0) / T);
    if (k === phase) return;
    if (phase >= 2) (phase % 2 === 0 ? on : off).push(engine.frameStats().gpuAvg);
    phase = k;
    show(k % 2 === 0);
    if (on.length && on.length === off.length) {
      const m = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
      console.warn(`ab ${names.join('+')} @${p.x.toFixed(1)},${p.y.toFixed(1)},${p.z.toFixed(1)}: gpu with ${m(on).toFixed(2)} ms, `
        + `without ${m(off).toFixed(2)} ms, cost ${(m(on) - m(off)).toFixed(2)} ms over ${on.length} cycles [${on.map((v, i) => (v - off[i]).toFixed(2)).join(' ')}]`);
    }
  });
}
