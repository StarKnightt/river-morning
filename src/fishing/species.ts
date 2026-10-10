import type { SpeciesId } from './types';

/**
 * The four trout of the prep fish set (E:\shots\river-morning\prep\fish\fish.json): model frame and sockets from
 * the GLB, gameplay values chosen here. Strength 1.0 is a fish whose runs snap the line if you keep reeling.
 */
export interface SpeciesDef {
  id: SpeciesId;
  name: string;
  latin: string;
  /** node name in trout_all.glb */
  node: string;
  /** model length (m) and snout z in the fish frame (origin = centre of mass) */
  modelLength: number;
  noseZ: number;
  /** fish.json hold_fish_offset_m (tail wrist into the right hand) and the suggested scale while held */
  holdOffset: number;
  holdScale: number;
  /** Fulton condition factor of the model: weight (g) = K * L(cm)^3 / 100 */
  fultonK: number;
  lengthCm: { min: number; typical: number; max: number };
  /** pull of a run at typical size, normalised to the line's break tension */
  strength: number;
  /** stamina scale (bigger = longer fight) */
  endurance: number;
  /** chance that a run ends in a jump */
  jump: number;
  /** seconds the bobber stays down before the fish lets go */
  hookWindow: number;
  /** swim speed scale */
  speed: number;
  /** fallback body colour (procedural stand-in), linear RGB */
  color: [number, number, number];
}

export const SPECIES: Record<SpeciesId, SpeciesDef> = {
  brown: {
    id: 'brown', name: 'Brown trout', latin: 'Salmo trutta', node: 'Trout_Brown',
    modelLength: 0.45, noseZ: 0.1776, holdOffset: 0.059, holdScale: 1, fultonK: 1.007,
    lengthCm: { min: 28, typical: 45, max: 62 }, strength: 1.0, endurance: 1.25, jump: 0.08, hookWindow: 1.1, speed: 1.0,
    color: [0.32, 0.24, 0.12],
  },
  rainbow: {
    id: 'rainbow', name: 'Rainbow trout', latin: 'Oncorhynchus mykiss', node: 'Trout_Rainbow',
    modelLength: 0.4, noseZ: 0.1542, holdOffset: 0.038, holdScale: 1, fultonK: 0.967,
    lengthCm: { min: 25, typical: 40, max: 52 }, strength: 0.95, endurance: 1.0, jump: 0.35, hookWindow: 0.9, speed: 1.15,
    color: [0.36, 0.38, 0.33],
  },
  brook: {
    id: 'brook', name: 'Brook trout', latin: 'Salvelinus fontinalis', node: 'Trout_Brook',
    modelLength: 0.3, noseZ: 0.1172, holdOffset: 0.002, holdScale: 1.3, fultonK: 0.913,
    lengthCm: { min: 18, typical: 30, max: 36 }, strength: 0.6, endurance: 0.75, jump: 0.1, hookWindow: 0.75, speed: 1.05,
    color: [0.2, 0.22, 0.14],
  },
  cutthroat: {
    id: 'cutthroat', name: 'Cutthroat trout', latin: 'Oncorhynchus clarkii', node: 'Trout_Cutthroat',
    modelLength: 0.38, noseZ: 0.1413, holdOffset: 0.042, holdScale: 1, fultonK: 0.897,
    lengthCm: { min: 24, typical: 38, max: 48 }, strength: 0.8, endurance: 0.95, jump: 0.15, hookWindow: 1.0, speed: 1.0,
    color: [0.34, 0.3, 0.18],
  },
};

export const SPECIES_IDS: SpeciesId[] = ['brown', 'rainbow', 'brook', 'cutthroat'];

export type Habitat = 'riffle' | 'pool' | 'run' | 'shallow';

/**
 * Who bites where: rainbows and cutthroat hold in fast riffle water, big browns in the deep slow pool,
 * brook trout along the shallow margins. Bite delays (s) per habitat: active riffle fish bite sooner.
 */
export const HABITATS: Record<Habitat, { weights: Record<SpeciesId, number>; bite: [number, number]; sizeBonus: Partial<Record<SpeciesId, number>> }> = {
  riffle: { weights: { rainbow: 0.45, cutthroat: 0.35, brook: 0.12, brown: 0.08 }, bite: [4, 10], sizeBonus: {} },
  pool: { weights: { brown: 0.42, rainbow: 0.2, cutthroat: 0.2, brook: 0.18 }, bite: [7, 16], sizeBonus: { brown: 1.08 } },
  run: { weights: { rainbow: 0.3, cutthroat: 0.3, brown: 0.25, brook: 0.15 }, bite: [5, 12], sizeBonus: {} },
  shallow: { weights: { brook: 0.4, cutthroat: 0.25, rainbow: 0.25, brown: 0.1 }, bite: [9, 20], sizeBonus: {} },
};

/** Habitat from the water under the float: section speed (m/s) and depth (m). */
export function habitatAt(sectionSpeed: number, depth: number): Habitat {
  if (depth < 0.45) return 'shallow';
  if (sectionSpeed >= 1.0) return 'riffle';
  if (sectionSpeed < 0.45 && depth > 1.0) return 'pool';
  return 'run';
}

export const weightKg = (s: SpeciesDef, lengthCm: number) => (s.fultonK * lengthCm ** 3) / 100 / 1000;
