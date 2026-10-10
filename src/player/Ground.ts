import { river, newSample } from '../world/River';
import { LAYOUT } from '../world/Layout';

/** A walkable top that is not terrain (dock deck, porch, bench top): its height where it covers (x, z), else null. */
export interface WalkSurface {
  readonly name: string;
  heightAt(x: number, z: number, margin?: number): number | null;
}

/** Axis-aligned rectangle at a fixed height. */
export function rectSurface(name: string, x0: number, x1: number, z0: number, z1: number, y: number): WalkSurface {
  return {
    name,
    heightAt: (x, z, margin = 0) => (x >= x0 - margin && x <= x1 + margin && z >= z0 - margin && z <= z1 + margin ? y : null),
  };
}

/**
 * Stand-in for the part 4 dock until its collider exists: the deck from the layout's dock line (x 11.5 +- 0.6,
 * z 7.6 -> 1.6). dock.glb has its deck top at local y 0 and sits ~0.45 m above the pool at its outer end
 * (blender-assets.json), so the deck is at y 0.45 (0.26 m above the bar at the shore end). Swap this for the real
 * deck when the dock is placed: `player.ground.surfaces` is a plain array.
 */
export const DOCK_DECK: WalkSurface = rectSurface(
  'dock-deck', LAYOUT.dock.x - 0.6, LAYOUT.dock.x + 0.6, Math.min(LAYOUT.dock.zTip, LAYOUT.dock.zBase), Math.max(LAYOUT.dock.zTip, LAYOUT.dock.zBase), 0.45,
);

export interface Circle { x: number; z: number; r: number }
const CELL = 4;
const cellKey = (i: number, j: number) => (i + 4096) * 8192 + (j + 4096);
/** Oriented box: centre, half extents along its own axes, rotation about Y (radians). */
export interface Box {
  x: number; z: number; hx: number; hz: number; rotY: number;
  /** false: blocks the walk only (furniture: the camera tests it in 3D through its `solid` hook) */
  cam?: boolean;
}

export interface GroundPoint {
  /** floor height the feet stand on */
  y: number;
  /** water surface above the terrain here (y of the water, NaN where unknown) */
  waterY: number;
  /** water depth over the terrain (m, <= 0 on dry land) */
  depth: number;
  /** walk surface providing the floor, or null for terrain */
  surface: WalkSurface | null;
}

/**
 * What the player can stand on: the rendered terrain, the river (wading depth), walk surfaces and 2D blockers
 * (rocks now; tree trunks, the cabin and the campfire ring when parts 3/4 register them).
 */
export class Ground {
  readonly surfaces: WalkSurface[] = [DOCK_DECK];
  readonly boxes: Box[] = [];
  private readonly rs = newSample();
  /** circles filed in 4 m cells (padded by a walker's radius), so thousands of trunks cost one cell's list */
  private readonly cells = new Map<number, Circle[]>();
  private circleCount = 0;

  constructor(
    /** rendered ground height (World.groundY) */
    readonly terrainY: (x: number, z: number) => number,
    /** water surface height, null to use the river's analytic level (no water mesh) */
    private waterAt: ((x: number, z: number) => number) | null,
  ) {}

  waterY(x: number, z: number): number {
    if (this.waterAt) return this.waterAt(x, z);
    return river.sample(x, z, this.rs).waterY;
  }

  /** Highest walk surface at (x, z) that is not buried in the terrain. */
  surfaceAt(x: number, z: number, terrain: number, margin = 0): { y: number; s: WalkSurface } | null {
    let best: { y: number; s: WalkSurface } | null = null;
    for (const s of this.surfaces) {
      const y = s.heightAt(x, z, margin);
      if (y !== null && y > terrain - 0.05 && (!best || y > best.y)) best = { y, s };
    }
    return best;
  }

  sample(x: number, z: number, out: GroundPoint): GroundPoint {
    const t = this.terrainY(x, z);
    const s = this.surfaceAt(x, z, t);
    const w = this.waterY(x, z);
    out.waterY = w;
    out.depth = w - t;
    out.y = s ? s.y : t;
    out.surface = s ? s.s : null;
    return out;
  }

  /** Floor for foot placement: surfaces count a little past their edge so a heel over the edge does not drop. */
  feetY = (x: number, z: number): number => {
    const t = this.terrainY(x, z);
    const s = this.surfaceAt(x, z, t, 0.25);
    return s ? Math.max(s.y, t) : t;
  };

  /** A round blocker (rock, tree trunk, post). Radius up to 2 m. */
  addCircle(x: number, z: number, r: number) {
    const c = { x, z, r: Math.min(r, 2) };
    // filed under every cell a walker of radius <= 0.5 m could touch it from
    const pad = c.r + 0.5;
    const i0 = Math.floor((x - pad) / CELL), i1 = Math.floor((x + pad) / CELL);
    const j0 = Math.floor((z - pad) / CELL), j1 = Math.floor((z + pad) / CELL);
    for (let i = i0; i <= i1; i++) {
      for (let j = j0; j <= j1; j++) {
        const k = cellKey(i, j);
        const list = this.cells.get(k);
        if (list) list.push(c);
        else this.cells.set(k, [c]);
      }
    }
    this.circleCount++;
  }

  /** A box blocker (cabin walls, woodpile): centre, half extents, rotation about Y. */
  addBox(b: Box) {
    this.boxes.push(b);
  }

  get blockerCount() {
    return this.circleCount + this.boxes.length;
  }

  /** Is (x, z) inside a box blocker grown by `pad`? Boxes count as full-height walls (the camera's boom). */
  inBox(x: number, z: number, pad: number): boolean {
    for (const b of this.boxes) {
      if (b.cam === false) continue;
      const cs = Math.cos(b.rotY), sn = Math.sin(b.rotY);
      const lx = (x - b.x) * cs - (z - b.z) * sn;
      const lz = (x - b.x) * sn + (z - b.z) * cs;
      const dx = Math.max(0, Math.abs(lx) - b.hx), dz = Math.max(0, Math.abs(lz) - b.hz);
      if (dx * dx + dz * dz < pad * pad) return true;
    }
    return false;
  }

  /** Push a point of radius r out of the blockers (in place). */
  resolve(p: { x: number; z: number }, r: number) {
    const list = this.cells.get(cellKey(Math.floor(p.x / CELL), Math.floor(p.z / CELL)));
    if (list) {
      for (const c of list) {
        const dx = p.x - c.x, dz = p.z - c.z, d = Math.hypot(dx, dz), min = c.r + r;
        if (d < min && d > 1e-6) {
          p.x = c.x + (dx / d) * min;
          p.z = c.z + (dz / d) * min;
        }
      }
    }
    for (const b of this.boxes) {
      const cs = Math.cos(b.rotY), sn = Math.sin(b.rotY);
      // into box space (rotation about Y by rotY maps local x to (cos, -sin))
      const lx = (p.x - b.x) * cs - (p.z - b.z) * sn;
      const lz = (p.x - b.x) * sn + (p.z - b.z) * cs;
      const cx = Math.max(-b.hx, Math.min(b.hx, lx)), cz = Math.max(-b.hz, Math.min(b.hz, lz));
      const dx = lx - cx, dz = lz - cz, d = Math.hypot(dx, dz);
      if (d >= r) continue;
      let ox: number, oz: number;
      if (d < 1e-6) {
        // centre inside: leave through the nearest face
        const ex = b.hx - Math.abs(lx), ez = b.hz - Math.abs(lz);
        ox = ex < ez ? (Math.sign(lx) || 1) * (b.hx + r) : lx;
        oz = ex < ez ? lz : (Math.sign(lz) || 1) * (b.hz + r);
      } else {
        ox = cx + (dx / d) * r;
        oz = cz + (dz / d) * r;
      }
      p.x = b.x + ox * cs + oz * sn;
      p.z = b.z - ox * sn + oz * cs;
    }
  }
}
