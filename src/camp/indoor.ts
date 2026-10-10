import * as THREE from 'three/webgpu';
import { abs, dot, exp, float, Fn, If, length, max, min, mix, normalWorld, positionWorld, step, uniform, vec3, type N } from '../gfx/tsl';

/**
 * The room inside the log cabin, cabin-local metres (cabin.glb: door and porch toward +Z, base at the bottom of the
 * piers). The walls are round logs on centre planes x +-2.85 and z +-2.35 that bulge 0.17 m to either side; the
 * GLB's plank floor (one quad) lies at 0.55 between the centre planes; the roof sheathing's underside falls from
 * the ridge at 4.57 by 0.629 per metre out from z 0.
 */
export const ROOM = {
  /** the log walls' centre planes (the floor quad reaches them) */
  cx: 2.85, cz: 2.35,
  /** the logs' inner bulge: the clear room */
  ix: 2.68, iz: 2.18,
  /** outer bulge (the old solid wall blocker) */
  ox: 3.15, oz: 2.65,
  floor: 0.55,
  /** the threshold board in the doorway, its top */
  sill: 0.544,
  /** the doorway between its jambs, the lintel's underside; the door leaf's mid-plane (planks inside, ledges outside) */
  door: { x0: -1.19, x1: -0.21, top: 2.622, mid: 2.437, x: -0.7 },
  frontWindow: { x: 1.4, y: 1.73, z: 2.35 },
  sideWindow: { x: -2.85, y: 1.88, z: 0 },
  /** underside of the roof sheathing at |z| */
  roof: (az: number) => 4.57 - 0.629 * az,
  /** the purlins' undersides (3.62) and the ridge pole's (4.37) */
  purlin: 3.62,
};

/** World to cabin-local for the shader: (cos yaw, sin yaw, cabin x, cabin z), the cabin's base height; the door 0..1. */
export const indoorU = {
  frame: uniform(new THREE.Vector4(1, 0, 0, -1e4)),
  baseY: uniform(-1e4),
  door: uniform(0),
};

/**
 * Sky light that reaches a surface inside the cabin (multiplies the camp materials' ambient occlusion, so it dims the
 * environment and sky light only): exactly 1 outside the room, so nothing seen from outside changes. Inside the
 * log walls' centre planes (above the side walls the gables' inner boards; in the doorway the door leaf's inner
 * half) and under the roof sheathing: a little bounce everywhere, more near the two windows and, while it is open,
 * the door, most on the faces turned toward them.
 */
export function indoorSky(p: N, n: N): N {
  const f = indoorU.frame;
  const dx = p.x.sub(f.z), dz = p.z.sub(f.w);
  const lx = dx.mul(f.x).sub(dz.mul(f.y));
  const lz = dx.mul(f.y).add(dz.mul(f.x));
  const ly = p.y.sub(indoorU.baseY);
  const ax = abs(lx), az = abs(lz);
  const D = ROOM.door;
  // (3.5 cm outside the centre planes: the chinking behind the logs; the outer chinking and the glass stay outside)
  const xLim = mix(float(ROOM.cx + 0.035), float(2.935), step(3.01, ly));
  const inDoor = step(D.x0 - 0.05, lx).mul(step(lx, D.x1 + 0.05)).mul(step(ly, D.top + 0.02));
  const zLim = mix(float(ROOM.cz + 0.035), float(D.mid), inDoor);
  // (from just under the plank floor: the crawl space under it, seen past the piers, stays as it was)
  const inside = step(ax, xLim).mul(step(az, zLim)).mul(step(0.53, ly)).mul(step(ly, float(4.6).sub(az.mul(0.629))));
  const L = vec3(lx, ly, lz);
  const nl = vec3(n.x.mul(f.x).sub(n.z.mul(f.y)), n.y, n.x.mul(f.y).add(n.z.mul(f.x)));
  const from = (cx: number, cy: number, cz: number, r2: number, k: number | N) => {
    const d = vec3(cx, cy, cz).sub(L);
    const len = max(length(d), 1e-3);
    const facing = dot(nl, d.div(len)).mul(0.6).add(0.4).clamp(0.05, 1);
    return exp(dot(d, d).mul(-1 / r2)).mul(facing).mul(k);
  };
  const fw = ROOM.frontWindow, sw = ROOM.sideWindow;
  const sky = min(float(0.15)
    .add(from(fw.x, fw.y, fw.z - 0.1, 1.8, 0.5))
    .add(from(sw.x + 0.1, sw.y, sw.z, 1.6, 0.45))
    .add(from(D.x, 1.45, ROOM.iz, 3.0, indoorU.door.mul(1.0))), 1);
  return mix(float(1), sky, inside);
}

/** the room's lamp (part 8's window light while it hangs in the room): on, and where it is (world) */
export const lampU = { on: uniform(0), pos: uniform(new THREE.Vector3(0, -1e3, 0)) };

/**
 * The lamp's shadow (the light's shadowNode, no shadow map): 1 inside the room (and on the door's inner half); outside
 * only where the line to the lamp leaves the room through the open doorway or a window. The logs, the roof and the
 * shut door hold the rest, so the porch and the yard are lit only through the openings.
 */
export function lampShadow(): N {
  return Fn(() => {
    const vis = float(1).toVar();
    If(lampU.on.greaterThan(0.5), () => {
      const f = indoorU.frame, D = ROOM.door;
      const local = (q: N) => {
        const dx = q.x.sub(f.z), dz = q.z.sub(f.w);
        return vec3(dx.mul(f.x).sub(dz.mul(f.y)), q.y.sub(indoorU.baseY), dx.mul(f.y).add(dz.mul(f.x)));
      };
      const p = local(positionWorld.add(normalWorld.mul(0.02))), l = local(lampU.pos);
      const ax = abs(p.x), az = abs(p.z);
      const inDoor = step(D.x0 - 0.05, p.x).mul(step(p.x, D.x1 + 0.05)).mul(step(p.y, D.top + 0.02));
      const zLim = mix(float(ROOM.cz + 0.035), float(D.mid), inDoor);
      const inside = step(ax, ROOM.cx + 0.035).mul(step(az, zLim)).mul(step(0.53, p.y)).mul(step(p.y, float(4.6).sub(az.mul(0.629))));
      // where the line from the lamp leaves the room: through the front wall (the door, the river window) or the west
      // gable (its window); the first plane it crosses inside the other walls' span is its exit
      const d = p.sub(l);
      const tz = float(ROOM.cz).sub(l.z).div(d.z.max(1e-4)), tx = float(-ROOM.cx).sub(l.x).div(d.x.min(-1e-4));
      const ez = l.add(d.mul(tz)), ex = l.add(d.mul(tx));
      const front = step(1e-4, d.z).mul(step(tz, 1)).mul(step(abs(ez.x), ROOM.cx));
      const door = step(D.x0, ez.x).mul(step(ez.x, D.x1)).mul(step(ez.y, D.top)).mul(indoorU.door);
      const win = step(1.0, ez.x).mul(step(ez.x, 1.8)).mul(step(1.41, ez.y)).mul(step(ez.y, 2.05));
      const west = step(d.x, -1e-4).mul(step(tx, 1)).mul(step(abs(ex.z), ROOM.cz));
      const sw = step(-0.37, ex.z).mul(step(ex.z, 0.37)).mul(step(1.55, ex.y)).mul(step(ex.y, 2.2));
      vis.assign(max(inside, min(front.mul(max(door, win)).add(west.mul(sw)), 1)));
    });
    return vis;
  })();
}

/** cabin-local helpers for the play code (yaw as CABIN.yaw) */
export function cabinFrameOf(x: number, z: number, yaw: number) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return {
    toX: (wx: number, wz: number) => (wx - x) * c - (wz - z) * s,
    toZ: (wx: number, wz: number) => (wx - x) * s + (wz - z) * c,
    fromX: (lx: number, lz: number) => x + lx * c + lz * s,
    fromZ: (lx: number, lz: number) => z - lx * s + lz * c,
  };
}
