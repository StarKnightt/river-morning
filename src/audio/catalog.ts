/**
 * The prepared sound set (`E:\shots\river-morning\prep\audio\audio.json`): Ogg Vorbis, 48 kHz, served from
 * `public/assets/audio/`. Loudness and peaks are copied from audio.json: loops are integrated LUFS, one-shot groups
 * are max momentary LUFS (every variant of a group is normalised to the same level) with the group's highest true
 * peak. Mix gains are dB relative to the file, at the layer's reference spot or the group's reference distance.
 */

export const AUDIO_DIR = 'assets/audio/';

export type LoopId = 'riverCalm' | 'riverRiffle' | 'birds' | 'wind' | 'crickets' | 'campfire' | 'coffee' | 'reel';

export interface LoopDef {
  file: string;
  lufs: number;
  peak: number;
  gainDb: number;
  /**
   * How the decoded buffer is kept: 'mono' averages the channels (point sources behind a PannerNode), 'left' keeps
   * channel 0 only (the crickets' right channel is the left one half a loop later; playback rebuilds it). Both halve
   * the decoded memory.
   */
  store?: 'mono' | 'left';
}

export const LOOPS: Record<LoopId, LoopDef> = {
  riverCalm: { file: 'amb_river_calm', lufs: -23.3, peak: -7.4, gainDb: -2 },
  riverRiffle: { file: 'amb_river_riffle', lufs: -22.9, peak: -10.2, gainDb: -1 },
  birds: { file: 'amb_forest_birds_morning', lufs: -23.2, peak: -8.2, gainDb: -8 },
  wind: { file: 'amb_wind_pines', lufs: -23.3, peak: -8.3, gainDb: -9 },
  crickets: { file: 'amb_night_crickets', lufs: -23.4, peak: -8.1, gainDb: -7, store: 'left' },
  // peak-bound file (impulsive crackle, -32.9 LUFS at -1.9 dBTP): +6 dB over the river beds, as the audio prep says
  campfire: { file: 'amb_campfire', lufs: -32.9, peak: -1.9, gainDb: 4, store: 'mono' },
  coffee: { file: 'amb_coffee_simmer', lufs: -23.2, peak: -4.0, gainDb: -6 },
  reel: { file: 'sfx_reel_wind_loop', lufs: -20.0, peak: -1.5, gainDb: -7 },
};

export type GroupId =
  | 'step_gravel' | 'step_wood' | 'step_grass' | 'step_creak'
  | 'splash_small' | 'splash_big' | 'thrash' | 'plop' | 'drip' | 'bite'
  | 'cast' | 'cast_heavy' | 'reel_drag' | 'reel_fast' | 'reel_pull'
  | 'owl' | 'bird_call' | 'door';

export interface GroupDef {
  files: string[];
  lufs: number;
  peak: number;
  gainDb: number;
  /** PannerNode reference distance (m) and roll-off factor of the inverse distance model */
  ref: number;
  rolloff: number;
  /** random variation per play: playback rate +- fraction, gain +- dB */
  rateJitter: number;
  gainJitter: number;
  /** simultaneous voices of this group; a new play steals the oldest */
  maxVoices: number;
  /** 0: decoded right after the first gesture (gameplay feedback), 1: after the ambience beds */
  priority: 0 | 1;
}

const files = (stem: string, n: number) => Array.from({ length: n }, (_, i) => `${stem}_${String(i + 1).padStart(2, '0')}`);

export const GROUPS: Record<GroupId, GroupDef> = {
  step_gravel: { files: files('sfx_step_gravel', 6), lufs: -23.3, peak: -1.3, gainDb: -3, ref: 4, rolloff: 1, rateJitter: 0.05, gainJitter: 1.5, maxVoices: 3, priority: 0 },
  step_wood: { files: files('sfx_step_wood', 6), lufs: -21.0, peak: -0.9, gainDb: -3, ref: 4, rolloff: 1, rateJitter: 0.04, gainJitter: 1.5, maxVoices: 3, priority: 0 },
  step_grass: { files: files('sfx_step_grass', 5), lufs: -26.2, peak: -1.1, gainDb: -3, ref: 4, rolloff: 1, rateJitter: 0.05, gainJitter: 1.5, maxVoices: 3, priority: 0 },
  step_creak: { files: files('sfx_step_wood_creak', 3), lufs: -20.5, peak: -0.7, gainDb: -5, ref: 4, rolloff: 1, rateJitter: 0.03, gainJitter: 1.5, maxVoices: 2, priority: 1 },
  splash_small: { files: files('sfx_splash_small', 2), lufs: -28.2, peak: -1.9, gainDb: 3, ref: 4, rolloff: 0.5, rateJitter: 0.06, gainJitter: 1.5, maxVoices: 3, priority: 0 },
  splash_big: { files: files('sfx_splash_big', 2), lufs: -23.8, peak: -1.1, gainDb: -1, ref: 5, rolloff: 0.6, rateJitter: 0.05, gainJitter: 1.5, maxVoices: 2, priority: 0 },
  thrash: { files: files('sfx_fish_thrash', 3), lufs: -16.2, peak: -1.7, gainDb: -6, ref: 4, rolloff: 0.7, rateJitter: 0.05, gainJitter: 1.5, maxVoices: 2, priority: 0 },
  plop: { files: files('sfx_plop', 4), lufs: -19.5, peak: -1.0, gainDb: -4, ref: 3, rolloff: 0.6, rateJitter: 0.1, gainJitter: 2, maxVoices: 3, priority: 0 },
  drip: { files: files('sfx_drip', 2), lufs: -25.1, peak: -1.5, gainDb: -6, ref: 2, rolloff: 1, rateJitter: 0.1, gainJitter: 2, maxVoices: 2, priority: 1 },
  bite: { files: files('sfx_bite_plip', 2), lufs: -27.8, peak: -1.5, gainDb: 3.5, ref: 4, rolloff: 0.3, rateJitter: 0.04, gainJitter: 1, maxVoices: 2, priority: 0 },
  cast: { files: files('sfx_cast_whoosh', 4), lufs: -19.2, peak: -0.9, gainDb: -5, ref: 3, rolloff: 1, rateJitter: 0.07, gainJitter: 1.5, maxVoices: 2, priority: 0 },
  cast_heavy: { files: files('sfx_cast_whoosh_heavy', 2), lufs: -16.2, peak: -1.1, gainDb: -6, ref: 3, rolloff: 1, rateJitter: 0.05, gainJitter: 1.5, maxVoices: 2, priority: 0 },
  reel_drag: { files: files('sfx_reel_drag', 4), lufs: -16.4, peak: -6.7, gainDb: -10, ref: 3, rolloff: 1, rateJitter: 0.04, gainJitter: 1, maxVoices: 3, priority: 0 },
  reel_fast: { files: ['sfx_reel_wind_fast_01'], lufs: -18.4, peak: -1.5, gainDb: -8, ref: 3, rolloff: 1, rateJitter: 0.03, gainJitter: 1, maxVoices: 1, priority: 1 },
  reel_pull: { files: ['sfx_reel_pull_01'], lufs: -19.8, peak: -1.8, gainDb: -7, ref: 3, rolloff: 1, rateJitter: 0.03, gainJitter: 1, maxVoices: 1, priority: 0 },
  owl: { files: files('sfx_owl_hoot', 2), lufs: -15.9, peak: -7.5, gainDb: -6, ref: 20, rolloff: 0.7, rateJitter: 0.03, gainJitter: 2, maxVoices: 1, priority: 1 },
  bird_call: {
    files: ['sfx_bird_canyon_wren_01', 'sfx_bird_canyon_wren_02', 'sfx_bird_meadowlark_01', 'sfx_bird_meadowlark_02'],
    lufs: -16.2, peak: -8.9, gainDb: -9, ref: 12, rolloff: 0.8, rateJitter: 0.03, gainJitter: 2.5, maxVoices: 2, priority: 1,
  },
  door: { files: files('sfx_door_creak', 2), lufs: -16.3, peak: -3.8, gainDb: -8, ref: 3, rolloff: 1, rateJitter: 0.03, gainJitter: 1, maxVoices: 1, priority: 1 },
};

export const GROUP_IDS = Object.keys(GROUPS) as GroupId[];
export const LOOP_IDS = Object.keys(LOOPS) as LoopId[];
