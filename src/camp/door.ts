import * as THREE from 'three/webgpu';
import type { Camp } from './index';
import type { Engine } from '../core/Engine';
import type { Player } from '../player/Player';
import type { Box } from '../player/Ground';
import { FRAMING_EXPLORE, type Framing } from '../player/ThirdPersonCamera';
import { MOVE } from '../player/PlayerController';
import { CABIN } from './layout';
import { cabinFrameOf, indoorU, lampShadow, lampU, ROOM } from './indoor';
import type { N } from '../gfx/tsl';

/** wide open (inward, about the hinge), and how long a swing takes */
const OPEN = THREE.MathUtils.degToRad(100);
const SWING_S = 0.7;
/** the prompt: feet within this of the doorway's middle, facing it within 60 degrees */
const REACH = 1.5;
const FACE_COS = 0.5;
/** indoors the camera comes in to a close shoulder view that fits the room */
export const FRAMING_INDOOR: Framing = { distance: 1.55, shoulder: 0.4, height: 1.5, fov: 60 };
/** how close the eye comes to the ceiling, the lintel and the furniture */
const CAM_CLEAR = 0.2;
const ease = (u: number) => u * u * (3 - 2 * u);
const sstep = (a: number, b: number, x: number) => ease(Math.min(1, Math.max(0, (x - a) / (b - a))));

export interface CabinPlay {
  /** his feet are inside the room (or in the doorway's inner half) */
  readonly inside: boolean;
  /** inside, or in the open doorway: no casting here */
  readonly indoors: boolean;
  /** 0 shut .. 1 wide open */
  readonly open: number;
  /** the door is waiting for him to step out of its way */
  readonly blocked: boolean;
  /** what the door does next: 1 opening, -1 closing, 0 still */
  readonly swing: number;
  toggle(): void;
}

/**
 * Play mode: the cabin's door opens and closes with E, the doorway and the room become walkable, the camera follows
 * him in, the lantern lights the room. Call once after `camp.attachGround(player.ground)`, before the warm-up.
 *  - E (through RodGround's prompt: the door takes E while its prompt shows, the rod's put-down / pick-up otherwise);
 *  - the solid wall blocker becomes four log walls with a doorway, the door leaf a blocker that swings with it (it
 *    never swings through him: it waits until he steps out of its way);
 *  - the plank floor and the threshold are walk surfaces; the furniture blocks the walk; the camera's boom keeps clear
 *    of the ceiling, the lintel and the furniture in 3D;
 *  - the room is drawn only while the door is open or he is inside;
 *  - part 8's window light (the lamp behind the river window at night) moves onto the room's lantern while the room
 *    shows: dim by day, warm at night; the lantern's glass glows.
 */
export function installCabinPlay(o: {
  engine: Engine; camp: Camp; player: Player; night: () => number; lamp: THREE.PointLight | null;
  sound?: (x: number, y: number, z: number, speed: number) => void;
}): CabinPlay {
  const { camp, player } = o;
  const ground = player.ground, ctl = player.controller;
  const F = cabinFrameOf(CABIN.x, CABIN.z, CABIN.yaw);
  const cy = camp.cabinY, D = ROOM.door;
  const interior = camp.interior;

  // ---- walls with a doorway instead of one solid box; the furniture
  const seg = (x0: number, x1: number, z0: number, z1: number, cam = true): Box => {
    const lx = (x0 + x1) / 2, lz = (z0 + z1) / 2;
    return { x: F.fromX(lx, lz), z: F.fromZ(lx, lz), hx: (x1 - x0) / 2, hz: (z1 - z0) / 2, rotY: CABIN.yaw, cam };
  };
  const solid = camp.blockers.boxes[0];
  const at = ground.boxes.indexOf(solid);
  if (at >= 0) ground.boxes.splice(at, 1);
  else console.warn('cabin: the wall blocker is not in the ground');
  ground.boxes.push(
    seg(-ROOM.ox, ROOM.ox, -ROOM.oz, -ROOM.iz), seg(-ROOM.ox, -ROOM.ix, -ROOM.oz, ROOM.oz), seg(ROOM.ix, ROOM.ox, -ROOM.oz, ROOM.oz),
    seg(-ROOM.ox, D.x0, ROOM.iz, ROOM.oz), seg(D.x1, ROOM.ox, ROOM.iz, ROOM.oz),
  );
  for (const b of interior.blocks) ground.boxes.push(seg(b.x - b.hx, b.x + b.hx, b.z - b.hz, b.z + b.hz, false));
  const floorY = cy + ROOM.floor, sillY = cy + ROOM.sill;
  ground.surfaces.push(
    { name: 'cabin-floor', heightAt: (x, z, m = 0) => (Math.abs(F.toX(x, z)) <= ROOM.ix + m && Math.abs(F.toZ(x, z)) <= ROOM.iz + m ? floorY : null) },
    {
      name: 'cabin-sill',
      heightAt: (x, z, m = 0) => {
        const lx = F.toX(x, z), lz = F.toZ(x, z);
        return lx >= D.x0 - m && lx <= D.x1 + m && lz >= ROOM.iz - m && lz <= 2.56 + m ? sillY : null;
      },
    },
  );

  // ---- the leaf: its bounds in the door's own frame, a blocker that follows its swing
  const door = camp.door;
  const base = door.quaternion.clone();
  const lb = new THREE.Box3();
  door.traverse((m) => {
    const g = (m as THREE.Mesh).geometry;
    if (!(m as THREE.Mesh).isMesh || !g) return;
    if (!g.boundingBox) g.computeBoundingBox();
    lb.union(g.boundingBox!);
  });
  const leaf: Box = { x: 0, z: 0, hx: (lb.max.x - lb.min.x) / 2, hz: Math.max(0.035, (lb.max.z - lb.min.z) / 2), rotY: 0 };
  ground.boxes.push(leaf);
  const lc = new THREE.Vector3((lb.min.x + lb.max.x) / 2, 0, (lb.min.z + lb.max.z) / 2);
  const q = new THREE.Quaternion(), qy = new THREE.Quaternion(), v = new THREE.Vector3(), UP = new THREE.Vector3(0, 1, 0);
  const leafAt = (a: number, out: Box) => {
    q.copy(base).multiply(qy.setFromAxisAngle(UP, a));
    v.copy(lc).applyQuaternion(q).add(door.position);
    out.x = v.x;
    out.z = v.z;
    out.rotY = 2 * Math.atan2(q.y, q.w);
    return out;
  };
  const probe: Box = { ...leaf };
  /** how far his body's centre is from the leaf at angle a */
  const gap = (a: number) => {
    leafAt(a, probe);
    const cs = Math.cos(probe.rotY), sn = Math.sin(probe.rotY);
    const px = ctl.position.x - probe.x, pz = ctl.position.z - probe.z;
    const lx = px * cs - pz * sn, lz = px * sn + pz * cs;
    return Math.hypot(Math.max(0, Math.abs(lx) - probe.hx), Math.max(0, Math.abs(lz) - probe.hz));
  };
  /** would the leaf, going from angle a0 to a1, come into him? (moving away from him it may always go) */
  const hits = (a0: number, a1: number) => {
    const g1 = gap(a1);
    return g1 < MOVE.radius + 0.05 && g1 < gap(a0) - 1e-5;
  };
  leafAt(0, leaf);

  // ---- the swing, E and the prompt
  let u = 0, dir = 0, blocked = false;
  const mid = new THREE.Vector3(F.fromX(D.x, D.mid), 0, F.fromZ(D.x, D.mid));
  const latch = () => v.set(F.fromX(D.x1 - 0.1, D.mid), cy + 1.55, F.fromZ(D.x1 - 0.1, D.mid));
  const api = {
    inside: false, indoors: false, open: 0, blocked: false, swing: 0,
    toggle() {
      dir = dir === 0 ? (u > 0.5 ? -1 : 1) : -dir;
      if (dir > 0 && u <= 0) {
        const p = latch();
        o.sound?.(p.x, p.y, p.z, 0.3);
      }
    },
  };
  const inReach = () => {
    const dx = mid.x - ctl.position.x, dz = mid.z - ctl.position.z, d = Math.hypot(dx, dz);
    if (d > REACH) return false;
    return d < 0.45 || (dx * Math.sin(ctl.yaw) + dz * Math.cos(ctl.yaw)) / d > FACE_COS;
  };
  if (player.rodGround) {
    player.rodGround.override = {
      prompt: () => (inReach() ? (dir > 0 || (dir === 0 && u > 0.5) ? '<kbd>E</kbd> Close door' : '<kbd>E</kbd> Open door') : null),
      press: () => api.toggle(),
    };
  }

  // ---- the camera: the ceiling (under the purlins), the lintel and the furniture
  player.camera.solid = (x, y, z) => {
    const lx = F.toX(x, z), lz = F.toZ(x, z), ly = y - cy;
    if (lz > ROOM.iz - CAM_CLEAR && lz < ROOM.oz + CAM_CLEAR && lx > D.x0 - CAM_CLEAR && lx < D.x1 + CAM_CLEAR && ly > D.top - CAM_CLEAR) return true;
    if (Math.abs(lx) > ROOM.ix || Math.abs(lz) > ROOM.iz) return false;
    if (ly > Math.min(ROOM.roof(Math.abs(lz)), ROOM.purlin) - CAM_CLEAR) return true;
    for (const b of interior.blocks) {
      if (Math.abs(lx - b.x) < b.hx + CAM_CLEAR * 0.6 && Math.abs(lz - b.z) < b.hz + CAM_CLEAR * 0.6 && ly > b.y0 - 0.1 && ly < b.y1 + CAM_CLEAR * 0.6) return true;
    }
    return false;
  };

  // ---- the rod indoors. His hand drops to his right side. In and round the doorway the rod lies low along the
  // doorway's axis, pointing into the room (ahead of him going in, trailing at his side going out), fixed to the doorway
  // and not to his facing, so he can turn in the doorway; from the porch it aims through the doorway's middle. Deeper in
  // the room it stands upright beside him, leaning toward the ridge (the room is highest under it) and away from the
  // gables. Blended by where his hand is, so it eases in and out at the threshold.
  let aimOwned = false;
  const n = new THREE.Vector3(), nd = new THREE.Vector3();
  player.beforePose = () => {
    const rg = player.rodGround, aim = player.poser.aim;
    const held = !!rg && rg.phase === 'hand';
    const fx = Math.sin(ctl.yaw), fz = Math.cos(ctl.yaw), rx = -Math.cos(ctl.yaw), rz = Math.sin(ctl.yaw);
    // the hand at his right side (world), and in the cabin's frame
    const hx = ctl.position.x + rx * 0.28, hz = ctl.position.z + rz * 0.28, hy = ctl.rootY + 0.88;
    const ax = F.toX(hx, hz), az = F.toZ(hx, hz);
    const wDoor = (1 - sstep(0.5, 0.95, Math.abs(ax - D.x))) * sstep(0.8, 1.45, az) * (1 - sstep(4.5, 5.2, az));
    const w = !held ? 0 : api.inside ? 1 : wDoor * ease(Math.min(1, u * 1.5));
    if (w <= 1e-3) {
      if (aimOwned) {
        aim.w = 0;
        aim.shift.set(0, 0, 0);
      }
      aimOwned = false;
      return;
    }
    const c = Math.cos(CABIN.yaw), s = Math.sin(CABIN.yaw);
    // low along the doorway's axis, toward a point 2.6 m inside on it (cabin-local)
    const tz = D.mid - 2.6;
    const ddx = D.x - ax, ddz = tz - az, dh = Math.hypot(ddx, ddz) || 1;
    nd.set(ddx / dh, -0.06, ddz / dh).normalize();
    // upright, leaning toward the ridge and away from the gables
    const lzl = -Math.sign(az) * 0.62 * Math.min(1, Math.abs(az) / 1.9);
    const lxl = -Math.sign(ax) * 0.5 * sstep(1.7, 2.45, Math.abs(ax));
    n.set(lxl, 1, lzl).normalize();
    const k = api.inside ? wDoor : 1;
    n.multiplyScalar(1 - k).addScaledVector(nd, k).normalize();
    // back to world
    const wx = n.x * c + n.z * s, wz = -n.x * s + n.z * c;
    aim.at.set(hx, hy, hz).addScaledVector(v.set(wx, n.y, wz), 4);
    aim.shift.set(rx * 0.13 - fx * 0.35, -0.2, rz * 0.13 - fz * 0.35);
    aim.w = w;
    aimOwned = true;
  };

  // ---- per frame, after the player (his position is this frame's)
  const L = o.lamp;
  const home = L ? { pos: L.position.clone(), dist: L.distance, color: L.color.clone() } : null;
  if (L) maskLamp(L);
  const lampAt = interior.lantern.clone().add(new THREE.Vector3(0, 0.14, 0));
  const glass = interior.lanternGlass;
  glass.emissive = glass.emissive ?? new THREE.Color();
  let indoorFrame = false, lampIn = false;
  const update = (dt: number, t: number) => {
    // the swing (it holds while the leaf would touch him)
    blocked = false;
    if (dir !== 0) {
      const nu = Math.min(1, Math.max(0, u + (dir * dt) / SWING_S));
      if (hits(OPEN * ease(u), OPEN * ease(nu))) blocked = true;
      else u = nu;
      if ((dir > 0 && u >= 1) || (dir < 0 && u <= 0)) {
        if (dir < 0) {
          const p = latch();
          o.sound?.(p.x, p.y, p.z, 0.55);
        }
        dir = 0;
      }
    }
    const a = OPEN * ease(u);
    camp.setDoorOpen(a);
    leafAt(a, leaf);
    indoorU.door.value = ease(u);
    const lx = F.toX(ctl.position.x, ctl.position.z), lz = F.toZ(ctl.position.x, ctl.position.z);
    const inside = Math.abs(lx) < ROOM.cx && lz > -ROOM.cz && lz < D.mid;
    const doorway = Math.abs(lx - D.x) < 0.8 && lz >= D.mid && lz < 3.3 && u > 0.3;
    Object.assign(api, { inside, indoors: inside || doorway, open: ease(u), blocked, swing: dir });
    // the room shows while the door is open or he is inside (and through the warm-up, so it compiles behind the
    // loading screen)
    interior.root.visible = !window.__RM?.ready || u > 0.001 || inside;
    // the camera comes in to the close indoor view in the room and the open doorway
    const want = inside || doorway;
    if (want !== indoorFrame) {
      player.camera.setFraming(want ? FRAMING_INDOOR : FRAMING_EXPLORE);
      indoorFrame = want;
    }
    // the lamp: part 8's window light onto the lantern while the room shows
    const night = o.night();
    const flick = 0.95 + 0.03 * Math.sin(t * 9.1) + 0.02 * Math.sin(t * 23.7 + 1.3);
    const lit = interior.root.visible && !!window.__RM?.ready;
    if (L && home) {
      if (lit) {
        lampOn(L, lampAt, night, flick);
        lampIn = true;
      } else if (lampIn) {
        L.position.copy(home.pos);
        L.distance = home.dist;
        L.color.copy(home.color);
        lampU.on.value = 0;
        lampIn = false;
      }
    }
    lampGlass(glass, night, flick);
  };
  o.engine.addUpdater(update);
  return api;
}

/**
 * ?cabinopen (photo mode and checks): the door stands open, the room shows and the lantern burns, as when he has
 * opened it. No walking, prompt or camera hooks.
 */
export function installCabinOpenView(o: { engine: Engine; camp: Camp; night: () => number; lamp: THREE.PointLight | null }) {
  const interior = o.camp.interior, L = o.lamp;
  const lampAt = interior.lantern.clone().add(new THREE.Vector3(0, 0.14, 0));
  o.camp.setDoorOpen(OPEN);
  indoorU.door.value = 1;
  interior.root.visible = true;
  const glass = interior.lanternGlass;
  if (L) maskLamp(L);
  o.engine.addUpdater((_dt, t) => {
    const night = o.night();
    const flick = 0.95 + 0.03 * Math.sin(t * 9.1) + 0.02 * Math.sin(t * 23.7 + 1.3);
    if (L) lampOn(L, lampAt, night, flick);
    lampGlass(glass, night, flick);
  });
}

/** the lamp's light stays in the room (lampShadow: through the open door and the windows only); set before the warm-up */
function maskLamp(L: THREE.PointLight) {
  L.castShadow = true;
  (L.shadow as unknown as { shadowNode: N }).shadowNode = lampShadow();
}

/** an oil lamp's pool: deep warm orange, bright on the table under it, falling off into the corners */
function lampOn(L: THREE.PointLight, at: THREE.Vector3, night: number, flick: number) {
  L.position.copy(at);
  L.distance = 6.2;
  L.decay = 2;
  L.color.setRGB(1.0, 0.36, 0.085);
  L.intensity = (1.0 + 7.3 * night) * flick;
  lampU.on.value = 1;
  lampU.pos.value.copy(at);
}

function lampGlass(glass: THREE.MeshStandardMaterial | THREE.MeshStandardNodeMaterial, night: number, flick: number) {
  glass.emissive.setRGB(1.0, 0.46, 0.13);
  glass.emissiveIntensity = (1.6 + 2.8 * night) * flick;
}
