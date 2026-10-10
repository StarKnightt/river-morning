import type { FishingButtons } from './types';

/**
 * Mouse and keyboard for the fishing loop. Left button (or Space) casts and hooks, right button (or R) reels.
 * Held flags plus press/release edges, cleared by consume() once per frame. With `pointerLock` (stand-in camera)
 * the first click also captures the mouse for looking around (Esc releases it). With `requireLock` (the character
 * owns the pointer lock) the buttons only count while the pointer is locked, so the click that locks never casts.
 */
export class FishingInput implements FishingButtons {
  cast = false;
  castPressed = false;
  castReleased = false;
  reel = false;
  reelPressed = false;
  reelReleased = false;
  /** window lost focus: cancel an aim instead of casting */
  cancel = false;
  /** mouse movement under pointer lock since the last frame (px) */
  lookX = 0;
  lookY = 0;
  /** held keys by KeyboardEvent.code */
  readonly keys: Record<string, boolean> = {};
  enabled = true;

  constructor(private el: HTMLElement, private pointerLock: boolean, readonly requireLock = false) {
    el.addEventListener('mousedown', this.onDown);
    window.addEventListener('mouseup', this.onUp);
    el.addEventListener('contextmenu', this.onMenu);
    window.addEventListener('mousemove', this.onMove);
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', this.onBlur);
    document.addEventListener('pointerlockchange', this.onLockChange);
  }

  /** the buttons are live (false while `requireLock` waits for the pointer lock) */
  get ready(): boolean {
    return !this.requireLock || document.pointerLockElement !== null;
  }

  private press(which: 'cast' | 'reel', down: boolean) {
    if (which === 'cast') {
      if (down && !this.cast) this.castPressed = true;
      if (!down && this.cast) this.castReleased = true;
      this.cast = down;
    } else {
      if (down && !this.reel) this.reelPressed = true;
      if (!down && this.reel) this.reelReleased = true;
      this.reel = down;
    }
  }

  private onDown = (e: MouseEvent) => {
    if (!this.enabled || !this.ready) return;
    if (e.button === 0) this.press('cast', true);
    else if (e.button === 2) this.press('reel', true);
    if (this.pointerLock && document.pointerLockElement !== this.el) {
      try {
        const r = this.el.requestPointerLock() as unknown as Promise<void> | undefined;
        r?.catch?.(() => {});
      } catch {
        // pointer lock refused (no gesture, iframe): keys still turn the view
      }
    }
  };

  private onUp = (e: MouseEvent) => {
    if (e.button === 0) this.press('cast', false);
    else if (e.button === 2) this.press('reel', false);
  };

  private onMenu = (e: Event) => e.preventDefault();

  private onMove = (e: MouseEvent) => {
    if (document.pointerLockElement === this.el) {
      this.lookX += e.movementX;
      this.lookY += e.movementY;
    }
  };

  private onKeyDown = (e: KeyboardEvent) => {
    if (!this.enabled || e.repeat) return;
    if (this.ready && e.code === 'Space') {
      this.press('cast', true);
      e.preventDefault();
    } else if (this.ready && e.code === 'KeyR') {
      this.press('reel', true);
    }
    this.keys[e.code] = true;
  };

  private onKeyUp = (e: KeyboardEvent) => {
    if (e.code === 'Space') this.press('cast', false);
    else if (e.code === 'KeyR') this.press('reel', false);
    this.keys[e.code] = false;
  };

  private onBlur = () => {
    if (this.cast) this.cancel = true;
    this.cast = false;
    this.reel = false;
    for (const k in this.keys) this.keys[k] = false;
  };

  /** Esc freed the mouse while the character owns the lock: let go of everything, like losing focus */
  private onLockChange = () => {
    if (this.requireLock && document.pointerLockElement === null) this.onBlur();
  };

  /** clear edges and look deltas (once per frame, after the simulation read them) */
  consume() {
    this.castPressed = this.castReleased = this.reelPressed = this.reelReleased = this.cancel = false;
    this.lookX = this.lookY = 0;
  }

  dispose() {
    this.el.removeEventListener('mousedown', this.onDown);
    window.removeEventListener('mouseup', this.onUp);
    this.el.removeEventListener('contextmenu', this.onMenu);
    window.removeEventListener('mousemove', this.onMove);
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    window.removeEventListener('blur', this.onBlur);
    document.removeEventListener('pointerlockchange', this.onLockChange);
  }
}
