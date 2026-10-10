/**
 * Hand-placed landmarks of the fishing spot. Everything here is in metres, Y up,
 * river flowing west -> east (+X), player bank to the south (+Z).
 */
export interface FlatZone {
  x: number;
  z: number;
  /** full-strength radius */
  r: number;
  /** blend width beyond r */
  blend: number;
  /** absolute target height; when omitted the natural height at the centre (+ dy) is used */
  y?: number;
  dy?: number;
  /** 0..1 how flat it gets (1 = dead level) */
  strength?: number;
}

export const LAYOUT = {
  cabin: { x: 9, z: 23.5, rotY: -0.12 },
  campfire: { x: 1.5, z: 13.2 },
  /** the dock model is 6.5 m long from its shore end (part 4: src/camp/layout.ts DOCK, DOCK_TIP_Z) */
  dock: { x: 11.5, zBase: 7.6, zTip: 1.1 },
  spawn: { x: 0, z: 9 },
  flats: [
    // cabin terrace
    { x: 9, z: 23.5, r: 5.0, blend: 6, strength: 0.8 },
    // campfire bench
    { x: 1.5, z: 13.2, r: 3.0, blend: 3 },
  ] as FlatZone[],
};
