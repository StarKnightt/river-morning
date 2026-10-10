/**
 * Render layers. The main camera sees 0-3; the pool mirror sees 0 and 5; the sun's shadow camera sees 0 and 4;
 * the GTAO normal pre-pass sees 0 only.
 */
export const LAYER_DEFAULT = 0;
/** Water surface and splashes: never in the mirror. */
export const LAYER_WATER = 1;
/** Grass blades, mist, small debris: main view only (not mirrored, not in the AO pre-pass). */
export const LAYER_NO_REFLECT = 2;
/**
 * Foliage in the main view (detailed tree LODs and their depth twins, ferns): kept out of the AO pre-pass (its
 * override material ignores alpha cut-outs) and out of the mirror, which draws its own cheap copies.
 */
export const LAYER_FOLIAGE = 3;
/** Shadow-only casters (cheap tree LODs): only the sun's shadow camera renders them. */
export const LAYER_SHADOW = 4;
/** Mirror-only copies (cheap tree LODs for the pool reflection). */
export const LAYER_MIRROR = 5;
/** The shadow trigger (gfx/shadowTrigger.ts) and the sun: a render that only redraws the sun's shadow map. */
export const LAYER_TRIGGER = 6;
