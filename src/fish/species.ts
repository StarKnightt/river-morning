/**
 * The four trout from the fish prep lane (E:\shots\river-morning\prep\fish\fish.json). glTF frame of every model:
 * +Y up, nose along +Z, the painted left flank on +X, origin at the body's centre of mass. Lengths and sockets are
 * in model metres; a fish of another size is the same mesh scaled uniformly.
 */
export type SpeciesId = 'brown' | 'rainbow' | 'brook' | 'cutthroat';

export interface Species {
  id: SpeciesId;
  name: string;
  latin: string;
  /** under public/assets/ */
  file: string;
  /** length of the model, snout to tail tip (m) */
  modelLength: number;
  /** model z of the snout (fish.json nose_z_m) */
  noseZ: number;
  /** model-space sockets (m): mouth = snout tip (hook and line), belly and tail-wrist grips for the hold pose */
  mouth: [number, number, number];
  bellyGrip: [number, number, number];
  tailGrip: [number, number, number];
  /** fish.json hold_fish_offset_m: move the fish this far along its +Z in the angler's `hold_fish` socket */
  holdOffset: number;
  /** extra scale while held (the brook is too short for the pose's 0.226 m between the palms) */
  holdScale: number;
  /** Fulton condition factor (fish.json): weight (g) = K x length(cm)^3 / 100 */
  fultonK: number;
  /** wild lengths in this river (m): smallest, typical, largest */
  size: [number, number, number];
  /** 0..1: how readily it inspects and takes a lure */
  boldness: number;
  /** multiplier on how easily the player and splashes spook it */
  caution: number;
  /** 0..1: chance of leaping (rises to jumps, and on the line) */
  jumpiness: number;
  /** preferred band of the water column, as a fraction of the depth from the surface (0) to the bed (1) */
  depthBand: [number, number];
  /** multiplier on the pull when hooked */
  fightPower: number;
  /** brown trout sulk on the bottom when hooked; rainbows run and jump */
  sulks: boolean;
  look: SpeciesLook;
}

/** How the species' skin is drawn (FishMaterial.ts): per-species uniforms over one shared program. */
export interface SpeciesLook {
  /** albedo of the texture in the air and in the water (the paintings are as bright as paper) */
  albedo: number;
  albedoWater: number;
  saturation: number;
  /** > 1 deepens the painting's mid-tones */
  gamma: number;
  /**
   * countershading: seen from above the back is darker and more olive than the painting's dorsal edge (the side
   * projection squeezes the whole back into that edge); `back` multiplies the colour there, `backK` how much
   */
  back: [number, number, number];
  backK: number;
  /** guanine silvering of the flanks in the air: a rainbow's flank flashes almost white in the sun */
  silver: number;
  /** a skin repainted from the painting (public/assets/), used instead of the GLB's texture */
  skin?: string;
  /** the species' colouring painted over the illustration in the shader (FishMaterial.ts); its markings are kept */
  paint?: SpeciesPaint;
}

/**
 * Colours (linear, in the illustration's units: the look's albedo still scales them) from the back over the flank to
 * the belly, the head, a lateral band (the rainbow's pink), the red spots and a tint for the fins. `k`: share over the
 * illustration's own colours; `bandK`, `redK`: how much band and red spots.
 */
export interface SpeciesPaint {
  back: [number, number, number];
  head: [number, number, number];
  flank: [number, number, number];
  belly: [number, number, number];
  band: [number, number, number];
  red: [number, number, number];
  fin: [number, number, number];
  k: number;
  bandK: number;
  redK: number;
}

export const NO_PAINT: SpeciesPaint = {
  back: [0, 0, 0], head: [0, 0, 0], flank: [0, 0, 0], belly: [0, 0, 0], band: [0, 0, 0], red: [0, 0, 0], fin: [1, 1, 1],
  k: 0, bandK: 0, redK: 0,
};

export const SPECIES: Record<SpeciesId, Species> = {
  brown: {
    id: 'brown', name: 'brown trout', latin: 'Salmo trutta', file: 'fish/trout_brown.glb',
    modelLength: 0.45, noseZ: 0.1776,
    mouth: [0, 0.0015, 0.1776], bellyGrip: [0, -0.0442, 0.0066], tailGrip: [0, -0.0003, -0.172],
    holdOffset: 0.059, holdScale: 1, fultonK: 1.007,
    size: [0.28, 0.4, 0.56], boldness: 0.35, caution: 1.3, jumpiness: 0.12, depthBand: [0.55, 0.95], fightPower: 1.1, sulks: true,
    // the USFWS painting's brown is peach-pink, dusted with small even freckles: repainted with an olive-gold head and
    // back, buttery gold flanks, a cream belly, big black and haloed red spots. Held in his shadow at 0.74 the gold
    // went cream under AgX; darker and more saturated it stays gold
    // Painted over in the shader (round 2's repaint still read as leopard print on biscuit): deep buttery gold flanks,
    // an olive head, a dark olive-brown back, a clean cream belly; its black and haloed red spots kept
    look: {
      albedo: 0.52, albedoWater: 0.95, saturation: 1.25, gamma: 1.08, back: [0.7, 0.68, 0.55], backK: 0.35, silver: 0.15, skin: 'fish/trout_brown_skin.webp',
      paint: {
        back: [0.085, 0.065, 0.018], head: [0.26, 0.2, 0.035], flank: [0.5, 0.25, 0.022], belly: [0.7, 0.6, 0.34],
        band: [0, 0, 0], red: [0.42, 0.035, 0.01], fin: [0.8, 0.58, 0.28], k: 0.9, bandK: 0, redK: 1,
      },
    },
  },
  rainbow: {
    id: 'rainbow', name: 'rainbow trout', latin: 'Oncorhynchus mykiss', file: 'fish/trout_rainbow.glb',
    modelLength: 0.4, noseZ: 0.1542,
    mouth: [0, -0.0024, 0.1542], bellyGrip: [0, -0.0412, 0.0022], tailGrip: [0, -0.0009, -0.1507],
    holdOffset: 0.038, holdScale: 1, fultonK: 0.967,
    size: [0.26, 0.36, 0.48], boldness: 0.55, caution: 1.0, jumpiness: 0.55, depthBand: [0.25, 0.75], fightPower: 1.15, sulks: false,
    // greenish-olive back, heavily speckled, silvery flanks with the pink band
    // painted over: an olive-grey back (greyer and greener than a brown's from above), silver-white flanks and belly,
    // a vivid pink-magenta band from cheek to tail; the illustration's black pepper kept
    look: {
      albedo: 0.54, albedoWater: 0.76, saturation: 1.0, gamma: 1.12, back: [0.62, 0.7, 0.6], backK: 0.35, silver: 1.0,
      paint: {
        back: [0.085, 0.11, 0.07], head: [0.15, 0.16, 0.1], flank: [0.55, 0.57, 0.55], belly: [0.72, 0.72, 0.7],
        band: [0.62, 0.12, 0.24], red: [0, 0, 0], fin: [0.62, 0.62, 0.56], k: 0.85, bandK: 0.85, redK: 0,
      },
    },
  },
  brook: {
    id: 'brook', name: 'brook trout', latin: 'Salvelinus fontinalis', file: 'fish/trout_brook.glb',
    modelLength: 0.3, noseZ: 0.1172,
    mouth: [0, -0.0013, 0.1172], bellyGrip: [0, -0.0293, 0.0044], tailGrip: [0, 0.0032, -0.1151],
    holdOffset: 0.002, holdScale: 1.3, fultonK: 0.913,
    size: [0.18, 0.25, 0.33], boldness: 0.75, caution: 0.85, jumpiness: 0.15, depthBand: [0.45, 0.95], fightPower: 0.9, sulks: false,
    look: { albedo: 0.54, albedoWater: 0.66, saturation: 1.0, gamma: 1.12, back: [0.62, 0.66, 0.56], backK: 0.45, silver: 0.25 },
  },
  cutthroat: {
    id: 'cutthroat', name: 'cutthroat trout', latin: 'Oncorhynchus clarkii', file: 'fish/trout_cutthroat.glb',
    modelLength: 0.38, noseZ: 0.1413,
    mouth: [0, -0.0016, 0.1413], bellyGrip: [0, -0.0347, -0.0031], tailGrip: [0, 0.0013, -0.1548],
    holdOffset: 0.042, holdScale: 1, fultonK: 0.897,
    size: [0.25, 0.34, 0.45], boldness: 0.65, caution: 0.9, jumpiness: 0.25, depthBand: [0.35, 0.9], fightPower: 1.0, sulks: false,
    look: { albedo: 0.54, albedoWater: 0.66, saturation: 1.0, gamma: 1.12, back: [0.6, 0.64, 0.5], backK: 0.5, silver: 0.55 },
  },
};

export const SPECIES_IDS: SpeciesId[] = ['brown', 'rainbow', 'brook', 'cutthroat'];

/** Swim presets (fish.json -> swim.presets): tail-tip amplitude as a fraction of the length, beats per second. */
export const SWIM = {
  wavelength: 0.95,
  hover: { amp: 0.03, hz: 1.2 },
  hold: { amp: 0.05, hz: 1.6 },
  cruise: { amp: 0.08, hz: 2.0 },
  burst: { amp: 0.12, hz: 6.0 },
  line: { amp: 0.14, hz: 3.5 },
};

/** Weight in kg from the length (m) with the species' condition factor. */
export const weightKg = (sp: Species, length: number) => (sp.fultonK * (length * 100) ** 3) / 100 / 1000;
