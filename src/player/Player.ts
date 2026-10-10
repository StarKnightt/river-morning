import * as THREE from 'three/webgpu';
import type { Engine } from '../core/Engine';
import type { Assets } from '../core/Assets';
import type { World } from '../world/World';
import { LAYOUT } from '../world/Layout';
import { river, newSample } from '../world/River';
import { Character } from './Character';
import { Animator } from './Animator';
import { RodRig } from './RodRig';
import { RodPoser } from './RodPoses';
import { FootIK } from './IK';
import { PoseLayer } from './PoseLayer';
import { Ground, type GroundPoint } from './Ground';
import { PlayerController } from './PlayerController';
import { FRAMING_EXPLORE, ThirdPersonCamera } from './ThirdPersonCamera';
import { PLAYER_SPOTS, type PlayerSpot } from './PlayerSpots';
import { CAST_EVENTS, REEL_TURNS_PER_LOOP } from './CharacterSpec';
import { anglerU, anglerWater } from './CharacterMaterials';
import { CharShadow } from './CharShadow';
import { RodGround } from './RodGround';
import { setLimbBodies } from '../water/BodyShadows';
import type { SlotName } from './Animator';

const q = new URLSearchParams(location.search);
/** ?player=0: no angler (scenery baselines); ?ik=0: no foot planting */
export const playerParams = {
  enabled: q.get('player') !== '0',
  footIK: q.get('ik') !== '0',
};

export interface PlayerOptions {
  /** third-person controls and camera (off in photo mode, under ?fly and ?orbit) */
  control: boolean;
  /** photo mode: the pose is snapped at this clip time and nothing moves */
  frozenTime: number | null;
  /** letter keys preview the fishing clips (see Player.bindDebugKeys) */
  debug: boolean;
}

/** Facing from (x, z) straight across the flow toward the channel. */
export function riverFacingYaw(x: number, z: number): number {
  const s = river.sample(x, z, newSample());
  // (-tz, tx) points to the south bank (d > 0)
  const sg = s.d >= 0 ? -1 : 1;
  return Math.atan2(sg * -s.tz, sg * s.tx);
}

/**
 * The third-person angler: character + animation blend + rod + foot IK, walking controller and the
 * over-the-shoulder camera. Part 6 drives fishing through `fishingHooks()` and `rod` (its RodRig interface), or
 * directly through `anim` (cast / reel_loop / hold_fish ...), `rod` (setBend, crank, tipWorld),
 * `controller.movementLocked` and `camera.setFraming`.
 */
export class Player {
  readonly anim: Animator;
  readonly poser: RodPoser;
  readonly pose: PoseLayer;
  readonly controller: PlayerController;
  readonly camera: ThirdPersonCamera;
  readonly ground: Ground;
  readonly footIK: FootIK;
  /** fishing clips stop the walking input (turn off if part 6 manages movementLocked itself) */
  lockWhileFishing = true;
  /** crank follows reel_loop's clip time (7 turns a loop); part 6 sets false to spin it itself */
  autoCrank = true;
  readonly spawn: { x: number; z: number; yaw: number };
  /** his sharp self-shadow (CharShadow), drawn after the pose each frame */
  readonly selfShadow: CharShadow;
  /** E puts the rod down and picks it up again (play mode only) */
  readonly rodGround: RodGround | null = null;
  /** after the walk and before the pose, each play frame (the cabin: the rod's carry indoors) */
  beforePose: ((dt: number) => void) | null = null;
  /** a jump moved him (main.ts gives the fishing loop fresh tackle at the new place) */
  onTeleport: (() => void) | null = null;
  private spotActive: PlayerSpot | null = null;
  private lastDt = 0;
  private hidden = false;
  private hint: HTMLElement | null = null;

  constructor(
    private engine: Engine,
    private world: World,
    readonly character: Character,
    readonly rod: RodRig,
    private opts: PlayerOptions,
  ) {
    this.anim = new Animator(character);
    this.poser = new RodPoser(character);
    this.pose = new PoseLayer(character, this.poser);
    this.ground = new Ground(world.groundY, world.water ? (x, z) => world.water!.query(x, z).y : null);
    for (const r of world.rocks) if (r.above > 0) this.ground.addCircle(r.x, r.z, r.obstacle.r * 0.9);
    // part 3's trees (World.trees once the forest exists): trunks, ~0.45 m across on a 25 m conifer
    const trees = (world as { trees?: readonly { x: number; z: number; h: number }[] }).trees;
    if (trees) for (const t of trees) this.ground.addCircle(t.x, t.z, 0.05 + 0.009 * t.h);
    this.controller = new PlayerController(this.ground);
    this.camera = new ThirdPersonCamera(engine.camera, this.ground);
    this.footIK = new FootIK(character, this.ground.feetY);
    this.footIK.enabled = playerParams.footIK;
    this.pose.ground = this.ground.feetY;
    this.poser.ground = this.ground.feetY;
    this.selfShadow = new CharShadow([...character.meshes, rod.mesh], (m) => character.shadowCut(m));
    anglerU.wristL.value.setFromMatrixPosition(character.restWorld.get('hand_l')!);
    anglerU.wristR.value.setFromMatrixPosition(character.restWorld.get('hand_r')!);
    this.spawn = { x: LAYOUT.spawn.x, z: LAYOUT.spawn.z, yaw: riverFacingYaw(LAYOUT.spawn.x, LAYOUT.spawn.z) };
    this.anim.onFootfall = (side) => this.wadeRing(side);
    if (opts.control) this.rodGround = new RodGround(this.anim, this.controller, this.ground, this.poser, this.pose, rod, document.getElementById('hud'));
    this.resetToSpawn();
    if (opts.control) this.bindInput(engine.renderer.domElement);
    if (opts.debug) this.bindDebugKeys();
  }

  /** Engine updater. `selfShadow` false leaves the self-shadow map to the next frame's update (photo poses step
   * him thousands of times within one frame). */
  update(dt: number, selfShadow = true) {
    this.lastDt = dt;
    const ctl = this.controller;
    const live = this.opts.control && !this.spotActive;
    if (live) {
      this.rodGround?.update(dt);
      ctl.movementLocked = this.lockWhileFishing && this.anim.state !== 'loco';
      ctl.update(dt, this.camera.yaw);
      this.anim.speed = ctl.groundSpeed;
      this.beforePose?.(dt);
    }
    this.anim.update(dt);
    this.applyPose(dt);
    this.limbShadows();
    if (live) this.camera.update(dt, _feet.set(ctl.position.x, ctl.rootY, ctl.position.z));
    if (!selfShadow) return;
    // his self-shadow from the final pose of this frame, before the scene renders; from beyond 35 m he is a few
    // dozen pixels tall and the sun map's own soft self-shadow does (the materials fall back to it)
    _centre.set(ctl.position.x, ctl.rootY + 0.95, ctl.position.z);
    const near = this.engine.camera.position.distanceToSquared(_centre) < 35 * 35;
    this.selfShadow.render(this.engine.renderer, _centre, this.world.sky.sunDir, !this.hidden && near);
  }

  /** Root, stance, foot IK, rod, hands and head onto the animated skeleton. */
  private applyPose(dt: number) {
    const c = this.character, ctl = this.controller;
    // boots settle into turf and gravel, not onto decks and steps
    const sink = ctl.floor.surface ? 0 : SINK;
    c.object.position.set(ctl.position.x, ctl.rootY - sink, ctl.position.z);
    c.object.rotation.set(0, ctl.yaw, 0);
    c.object.updateMatrixWorld(true);
    const blend = this.anim.blend;
    const a = this.anim;
    this.pose.body(blend, a.movingAmount, ctl.yaw);
    this.footIK.update(dt, ctl.rootY - sink, ctl.forward, this.pose.feet, this.pose.kneeOut);

    const reelW = RodPoser.reelWeight(blend);
    if (this.autoCrank && reelW > 0) {
      const s = blend.find((b) => b.name === 'reel_loop')!;
      const dur = this.anim.character.clip('reel_loop').duration;
      this.rod.setCrank((s.t / dur) * REEL_TURNS_PER_LOOP * Math.PI * 2);
    }
    _feet.set(ctl.position.x, ctl.rootY - sink, ctl.position.z);
    this.poser.blend(blend, a.rodLayer, _feet, ctl.yaw, _p, _q);
    this.rod.setPose(_p, _q);
    this.rod.update(dt);
    this.pose.arms(blend, a.rodLayer, a.movingAmount, a.walkPhase, ctl.yaw, _feet, this.rod);
    this.updateLook(blend);
  }

  /**
   * Fish in the hands (hold_fish): level, back up, from the tail fist (left) toward the head hand (right): the wrist of
   * the tail in the fist, the centre of mass 0.385 of its length ahead of it, the belly on the head hand's fingers.
   * Releasing it (kneel_release): held level in the water facing upstream where the hands are under it.
   */
  fishHold(pos: THREE.Vector3, quat: THREE.Quaternion): boolean {
    const r = this.pose.release;
    if (r.w > 0.05) {
      pos.copy(r.pos);
      _m4.lookAt(r.dir, _zero, _up);
      quat.setFromRotationMatrix(_m4);
      return true;
    }
    const p = this.pose.palms;
    if (p.w < 0.05) {
      this.character.fishSocket(pos, quat);
      return false;
    }
    _v.copy(p.r).sub(p.l).setY(0).normalize();
    pos.copy(p.l).addScaledVector(_v, 0.385 * this.pose.fishLength);
    // the fist's hole (closed for the cork) sits under its palm point: the 4.5 cm deep wrist is centred in it, the
    // fingers wrapping it down to the belly line
    pos.y = p.l.y + 0.008;
    _m4.lookAt(_v, _zero, _up);
    quat.setFromRotationMatrix(_m4);
    return true;
  }

  /** Material inputs that follow the pose: joint bends for the creases, the head frame for the brim's shadow. */
  private updateLook(blend: readonly { name: SlotName; w: number }[]) {
    const c = this.character, u = anglerU;
    const bend = (a: string, b: string, e: string, lo: number, hi: number) => {
      _a.setFromMatrixPosition(c.bone(a).matrixWorld);
      _b.setFromMatrixPosition(c.bone(b).matrixWorld);
      _e.setFromMatrixPosition(c.bone(e).matrixWorld);
      const d1 = _b.sub(_a).normalize(), d2 = _e.sub(_target.setFromMatrixPosition(c.bone(b).matrixWorld)).normalize();
      return THREE.MathUtils.smoothstep(Math.acos(THREE.MathUtils.clamp(d1.dot(d2), -1, 1)), lo, hi);
    };
    u.bend.value.set(
      bend('upperarm_l', 'lowerarm_l', 'hand_l', 0.3, 2.2), bend('upperarm_r', 'lowerarm_r', 'hand_r', 0.3, 2.2),
      bend('thigh_l', 'calf_l', 'foot_l', 0.08, 1.5), bend('thigh_r', 'calf_r', 'foot_r', 0.08, 1.5),
    );
    const head = c.bone('head');
    u.headInv.value.copy(head.matrixWorld).invert();
    const sky = this.world.sky;
    const sun = sky.u.sunDir.value as THREE.Vector3;
    u.sunWorld.value.copy(sun);
    u.sunHead.value.copy(sun).transformDirection(u.headInv.value);
    // light thrown up by the sunlit bar (gravel and grass ~0.26 albedo, most of it in sun round him), onto a
    // down-facing surface; and the open sky's share from above
    const e = sky.sun.intensity * Math.max(sun.y, 0) * 0.26 * 0.8;
    u.bounce.value.copy(sky.sun.color).multiplyScalar(e);
    u.sky.value.setRGB(sky.u.zenith.value.r * 1.6 + 0.06, sky.u.zenith.value.g * 1.6 + 0.07, sky.u.zenith.value.b * 1.6 + 0.08);
    u.strap.value = RodPoser.stowWeight(blend, this.anim.rodLayer) * (1 - this.poser.away);
    // arms reaching forward pull drag lines across the back of the coat
    const yaw = this.controller.yaw;
    _fwd.set(Math.sin(yaw), 0, Math.cos(yaw));
    const reach = (s: 'l' | 'r') => {
      _a.setFromMatrixPosition(c.bone(`upperarm_${s}`).matrixWorld);
      _b.setFromMatrixPosition(c.bone(`lowerarm_${s}`).matrixWorld).sub(_a).normalize();
      return THREE.MathUtils.smoothstep(_b.dot(_fwd), 0.05, 0.75);
    };
    u.reach.value.set(reach('l'), reach('r'));
    // how far each upper arm hangs below the bind A-pose (41.5 degrees out from straight down, in the body's
    // frontal plane): drives the shoulder cap's pose corrective
    _rgt.set(-Math.cos(yaw), 0, Math.sin(yaw));
    const down = (s: 'l' | 'r') => {
      _a.setFromMatrixPosition(c.bone(`upperarm_${s}`).matrixWorld);
      _b.setFromMatrixPosition(c.bone(`lowerarm_${s}`).matrixWorld).sub(_a).normalize();
      const out = Math.max(0, _b.dot(_rgt) * (s === 'r' ? 1 : -1));
      const abd = THREE.MathUtils.radToDeg(Math.atan2(out, Math.max(1e-3, -_b.y)));
      return THREE.MathUtils.clamp((41.5 - abd) / 30, 0, 1);
    };
    u.armDown.value.set(down('l'), down('r'));
    // a fish in his hands (or held in the river) wets them and the cuffs; they dry over a minute or so
    const handling = blend.some((b) => b.w > 0.3 && (b.name === 'land_lift' || b.name === 'hold_fish' || b.name === 'kneel_release' || b.name === 'kneel'));
    u.wet.value = handling ? 1 : Math.max(0, u.wet.value - this.lastDt / 60);
    // the main camera and its pixel footprint per metre (the rod's tip fades below a pixel)
    const cam = this.engine.camera;
    u.camMain.value.setFromMatrixPosition(cam.matrixWorld);
    const hPx = Math.max(1, this.engine.renderer.domElement.height);
    u.pix.value = (2 * Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2)) / (hPx * (cam.zoom || 1));
  }

  /**
   * Forearms and hands held in the river (the release): their crisp shadows on the bed, breaking the net of light
   * on the stones round them (water/BodyShadows.ts), for the parts under the surface.
   */
  private limbShadows() {
    const water = this.world.water;
    if (!water || this.hidden || this.pose.release.w < 0.3) {
      if (this.limbs.length) {
        this.limbs.length = 0;
        setLimbBodies(this.limbs);
      }
      return;
    }
    this.limbs.length = 0;
    const c = this.character;
    let k = 0;
    const seg = (a: THREE.Vector3, b: THREE.Vector3, r: number) => {
      const ya = water.query(a.x, a.z).y, yb = water.query(b.x, b.z).y;
      if (a.y >= ya && b.y >= yb) return;
      // only the part under the surface
      if (a.y > ya) a.lerp(b, (a.y - ya) / Math.max(1e-4, a.y - b.y));
      else if (b.y > yb) b.lerp(a, (b.y - yb) / Math.max(1e-4, b.y - a.y));
      const e = _limb[k++] ??= { a: new THREE.Vector3(), b: new THREE.Vector3(), r: 0 };
      e.a.copy(a);
      e.b.copy(b);
      e.r = r;
      this.limbs.push(e);
    };
    for (const s of ['l', 'r'] as const) {
      _a.setFromMatrixPosition(c.bone(`lowerarm_${s}`).matrixWorld);
      _b.setFromMatrixPosition(c.bone(`hand_${s}`).matrixWorld);
      _e.copy(_b);
      seg(_a, _b, 0.034);
      _target.setFromMatrixPosition(c.bone(`middle_02_${s}`).matrixWorld);
      seg(_e, _target, 0.038);
    }
    setLimbBodies(this.limbs);
  }

  private limbs: Array<{ a: THREE.Vector3; b: THREE.Vector3; r: number }> = [];

  /** A faint ring where a walking foot plants in water. */
  private wadeRing(side: 'l' | 'r') {
    const water = this.world.water;
    if (!water || this.controller.floor.surface || this.hidden) return;
    _knob.setFromMatrixPosition(this.character.bone(`foot_${side}`).matrixWorld);
    const depth = this.ground.waterY(_knob.x, _knob.z) - this.ground.terrainY(_knob.x, _knob.z);
    if (depth < 0.03) return;
    const s = Math.min(depth / 0.3, 1);
    water.events.ring(_knob.x, _knob.z, this.engine.time, 0.002 + 0.004 * s, 1.6 + s, 0.02 + 0.03 * s);
  }

  /** Back to the spawn stance: facing the river, rod in both hands. */
  resetToSpawn() {
    const s = this.spawn;
    anglerU.wet.value = 0;
    this.rodGround?.reset();
    this.controller.teleport(s.x, s.z, s.yaw);
    this.anim.setPose('loco', this.opts.frozenTime ?? 0, { rodHeld: true });
    this.rod.setBend(0);
    this.rod.resetDynamics();
    this.footIK.reset();
    this.setHidden(false);
    this.spotActive = null;
    this.camera.yaw = s.yaw;
    this.camera.pitch = THREE.MathUtils.degToRad(12);
    this.camera.setFraming(FRAMING_EXPLORE);
    this.camera.snap();
    this.applyPose(0);
  }

  /**
   * The part 6 lane's CharacterHooks (src/fishing/types.ts), structurally typed so this file does not depend on
   * it: `installFishing({ engine, world, assets, character: player.fishingHooks(), rod: player.rod })`. The fishing
   * code then drives the crank itself (rod.crank), so the clip-timed crank turns off.
   */
  fishingHooks() {
    this.autoCrank = false;
    const a = this.anim, ctl = this.controller;
    return {
      /** cast clip, a little quicker with power; seconds until the line leaves the tip */
      playCast: (power: number) => {
        const rate = 0.9 + 0.25 * THREE.MathUtils.clamp(power, 0, 1);
        a.play('cast', { next: 'wait', timeScale: rate, fade: 0.2 });
        return CAST_EVENTS.release / rate;
      },
      /** reel_loop; when not winding the arms barely move (the crank hand stays on the knob by IK) */
      playReel: (reeling: boolean, _tension: number) => a.play('reel_loop', { timeScale: reeling ? 1 : 0.15, fade: 0.25 }),
      playHold: (fish: { holdOffset: number; lengthCm?: number; holdScale?: number }) => {
        this.pose.fishLength = ((fish.lengthCm ?? 46) / 100) * (fish.holdScale ?? 1);
        a.play('land_lift', { next: 'hold_fish', fade: 0.25 });
      },
      idle: () => {
        a.play('wait', { fade: 0.35 });
      },
      playStrike: () => a.play('strike', { next: 'reel_loop', fade: 0.12 }),
      /** kneel_release: down on one knee in the shallows, the fish held in the water facing upstream, let go ~2.9 s
       * in, then up again */
      playRelease: (_fish: unknown) => {
        a.play('kneel_release', { next: 'wait', fade: 0.35 });
        return 2.9;
      },
      /** where to hold the fish in the water (facing upstream) when he releases it */
      releaseInto: (target: THREE.Vector3, upstream: THREE.Vector3, lengthM: number) => this.pose.setRelease(target, upstream, lengthM),
      /** eyes on a world point (weight 0..1; null looks where the clip does), the rod in hand turned toward another */
      attend: (look: THREE.Vector3 | null, lookW: number, rodAt: THREE.Vector3 | null, rodW: number) => {
        this.pose.attention.w = look ? lookW : 0;
        if (look) this.pose.attention.at.copy(look);
        this.poser.aim.w = rodAt ? rodW : 0;
        if (rodAt) this.poser.aim.at.copy(rodAt);
      },
      getFishHold: (pos: THREE.Vector3, quat: THREE.Quaternion) => {
        // (placed by its length between the hands: the species' hold offset was for the old fixed palms)
        this.fishHold(pos, quat);
        return true;
      },
      getAim: (origin: THREE.Vector3, dir: THREE.Vector3) => {
        this.rod.tipWorld(origin);
        dir.set(Math.sin(ctl.yaw), 0, Math.cos(ctl.yaw));
      },
      getFeet: (out: THREE.Vector3) => out.set(ctl.position.x, ctl.rootY, ctl.position.z),
    };
  }

  setHidden(h: boolean) {
    this.hidden = h;
    this.character.setVisible(!h);
    this.rod.mesh.visible = !h;
  }

  /**
   * window.__RM.spot hook: P1..P9 (append 'h' to hide the angler) pose him and frame the camera; any other id puts
   * him back at spawn and returns false so the scenery spot moves the camera.
   */
  spot(id: string): boolean {
    const hide = /^P\d+h$/.test(id);
    const s = PLAYER_SPOTS.find((p) => p.id === (hide ? id.slice(0, -1) : id));
    if (!s) {
      if (this.spotActive || this.hidden) this.resetToSpawn();
      return false;
    }
    const yaw = s.at.yaw !== undefined ? THREE.MathUtils.degToRad(s.at.yaw) : riverFacingYaw(s.at.x, s.at.z);
    this.rodGround?.reset();
    this.controller.teleport(s.at.x, s.at.z, yaw);
    this.anim.setPose(s.pose.clip, s.pose.t ?? 0, { speed: s.pose.speed, phase: s.pose.phase, rodHeld: s.pose.rodHeld ?? true });
    this.rod.resetDynamics();
    if (s.pose.bend) this.rod.setBend(s.pose.bend, _v.set(Math.sin(yaw), -0.8, Math.cos(yaw)));
    else this.rod.setBend(0);
    this.footIK.reset();
    this.setHidden(hide);
    this.applyPose(0);
    const cam = this.engine.camera;
    if (s.camera.mode === 'tp') {
      this.camera.yaw = yaw + THREE.MathUtils.degToRad(s.camera.yaw ?? 0);
      this.camera.pitch = THREE.MathUtils.degToRad(s.camera.pitch ?? 12);
      this.camera.setFraming({ ...FRAMING_EXPLORE, distance: s.camera.distance ?? FRAMING_EXPLORE.distance });
      this.camera.snap();
      this.camera.update(0, _feet.set(this.controller.position.x, this.controller.rootY, this.controller.position.z));
    } else {
      const rx = -Math.cos(yaw), rz = Math.sin(yaw), fx = Math.sin(yaw), fz = Math.cos(yaw);
      const at = (o: [number, number, number], out: THREE.Vector3) =>
        out.set(this.controller.position.x + rx * o[0] + fx * o[2], this.controller.rootY + o[1], this.controller.position.z + rz * o[0] + fz * o[2]);
      cam.position.copy(at(s.camera.eye, _v));
      cam.up.set(0, 1, 0);
      cam.lookAt(at(s.camera.look, _target));
      cam.fov = s.camera.fov ?? 50;
      cam.updateProjectionMatrix();
    }
    this.spotActive = s;
    return true;
  }

  private bindInput(dom: HTMLElement) {
    dom.addEventListener('click', () => {
      if (document.pointerLockElement === dom) return;
      const r = dom.requestPointerLock() as unknown as Promise<void> | undefined;
      r?.catch?.(() => {});
    });
    document.addEventListener('mousemove', (e) => {
      if (document.pointerLockElement === dom && !this.spotActive) this.camera.onMouse(e.movementX, e.movementY);
    });
    this.controller.bindKeys(window);
    const hud = document.getElementById('hud');
    if (!hud) return;
    const el = document.createElement('div');
    el.textContent = 'Click to look around  ·  WASD to walk  ·  Shift to walk faster  ·  E puts the rod down or picks it up  ·  Esc pauses';
    Object.assign(el.style, {
      position: 'absolute', left: '0', right: '0', bottom: '26px', textAlign: 'center', fontSize: '13px',
      letterSpacing: '0.08em', color: 'rgba(239, 230, 210, 0.78)', textShadow: '0 1px 3px rgba(0, 0, 0, 0.7)',
      transition: 'opacity 0.6s ease', pointerEvents: 'none',
    });
    hud.appendChild(el);
    this.hint = el;
    document.addEventListener('pointerlockchange', () => {
      if (this.hint) this.hint.style.opacity = document.pointerLockElement === dom ? '0' : '1';
    });
  }

  /**
   * Jump hook (debug/Travel.ts onTeleport): feet on the floor nearest the target that he can stand on,
   * facing `look`, camera behind him.
   */
  teleportTo(feet: THREE.Vector3, look: THREE.Vector3) {
    const p = this.nearestStandable(feet.x, feet.z);
    const yaw = Math.atan2(look.x - p.x, look.z - p.z);
    this.controller.teleport(p.x, p.z, yaw);
    this.rodGround?.onTeleport();
    this.rod.resetDynamics();
    this.footIK.reset();
    this.spotActive = null;
    this.setHidden(false);
    this.camera.yaw = yaw;
    this.camera.snap();
    this.onTeleport?.();
  }

  /** (x, z) itself if he can stand there, else the closest point on rings out to 15 m that he can. */
  private nearestStandable(x: number, z: number): { x: number; z: number } {
    const g: GroundPoint = { y: 0, waterY: 0, depth: 0, surface: null };
    const ok = (px: number, pz: number) => {
      this.ground.sample(px, pz, g);
      return g.surface !== null || g.depth <= 0.35;
    };
    if (ok(x, z)) return { x, z };
    for (let r = 0.5; r <= 15; r += 0.5) {
      const n = Math.max(8, Math.round(r * 6));
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2;
        if (ok(x + Math.cos(a) * r, z + Math.sin(a) * r)) return { x: x + Math.cos(a) * r, z: z + Math.sin(a) * r };
      }
    }
    return { x: this.spawn.x, z: this.spawn.z };
  }

  /**
   * ?debug letter keys (R/Space/mouse are part 6's fishing input): G wait, C cast (then
   * wait), V strike (then reel), X reel_loop, H land_lift then hold_fish, Z release_fish, N back to walking,
   * U rod in hand / slung, B bend the rod, I foot IK on/off. T is part 8's time-of-day key.
   */
  private bindDebugKeys() {
    window.addEventListener('keydown', (e) => {
      if (e.repeat || e.ctrlKey || e.altKey || e.metaKey) return;
      const a = this.anim;
      switch (e.code) {
        case 'KeyG': a.play('wait'); break;
        case 'KeyC': a.play('cast', { next: 'wait', onEvent: (n) => console.info(`[player] cast ${n}`) }); break;
        case 'KeyV': a.play('strike', { next: 'reel_loop', fade: 0.15 }); break;
        case 'KeyX': a.play('reel_loop'); break;
        case 'KeyH': a.play('land_lift', { next: 'hold_fish', fade: 0.25 }); break;
        case 'KeyZ': a.play('release_fish', { next: 'wait', fade: 0.3 }); break;
        case 'KeyN': a.stop(); break;
        case 'KeyU': a.rodHeld = !a.rodHeld; break;
        case 'KeyB': this.rod.setBend(this.rod.bend > 0 ? 0 : 0.6); break;
        case 'KeyI': this.footIK.enabled = !this.footIK.enabled; break;
        default: return;
      }
    });
    Object.assign(window, { __player: this });
  }
}

/**
 * Load the angler and rod, add them to the scene and register the per-frame update. Null with ?player=0, or if the
 * assets fail to load (logged; the scenery keeps working with the spot camera).
 */
export async function installPlayer(engine: Engine, assets: Assets, world: World, opts: PlayerOptions): Promise<Player | null> {
  if (!playerParams.enabled) return null;
  // (before the materials are built: his submerged parts are drawn through the water's refraction)
  anglerWater.field = world.water?.data.field ?? null;
  let character: Character, rod: RodRig;
  try {
    [character, rod] = await Promise.all([Character.load(assets), RodRig.load(assets)]);
  } catch (e) {
    console.error('[player] angler assets failed to load, continuing without the player', e);
    return null;
  }
  engine.scene.add(character.object, rod.mesh);
  const player = new Player(engine, world, character, rod, opts);
  engine.addUpdater((dt) => player.update(dt));
  return player;
}

const _feet = new THREE.Vector3();
const _p = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _v = new THREE.Vector3();
const _knob = new THREE.Vector3();
const _target = new THREE.Vector3();
const _m4 = new THREE.Matrix4();
const _zero = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
/** how far the boots settle into turf and gravel (m) */
const SINK = 0.012;
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _e = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _rgt = new THREE.Vector3();
const _centre = new THREE.Vector3();
const _limb: Array<{ a: THREE.Vector3; b: THREE.Vector3; r: number }> = [];
