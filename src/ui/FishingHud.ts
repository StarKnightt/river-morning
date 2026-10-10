import './fishing-hud.css';
import { shotMode } from '../core/params';
import type { CaughtFish, FishingState, PromptKind } from '../fishing/types';

export interface HudInput {
  state: FishingState;
  prompt: PromptKind;
  toast: string | null;
  power: number;
  tension: number;
  caught: number;
  biggest: CaughtFish | null;
  last: CaughtFish | null;
  lastRecord: boolean;
}

/** the tally shows this long after it changes (ms), and while Tab is held */
const TALLY_MS = 4000;
/** the cast hint fades this long after it appears (ms); it comes back after the next catch, reel-in or pick-up */
const CAST_HINT_MS = 4500;

const PROMPTS: Record<Exclude<PromptKind, null>, string> = {
  cast: 'Hold <kbd>LMB</kbd> to cast <span style="opacity:.6">&middot;</span> <kbd>E</kbd> put the rod down',
  release: 'Release to cast',
  wait: 'Wait for a bite <span style="opacity:.6">&middot;</span> hold <kbd>RMB</kbd> to reel in',
  reelin: 'Hold <kbd>RMB</kbd> to reel in',
  hook: '<kbd>LMB</kbd> to hook!',
  reel: 'Hold <kbd>RMB</kbd> to reel',
  ease: 'Ease off, the line will snap',
};

const el = (cls: string, parent: HTMLElement, tag = 'div') => {
  const e = document.createElement(tag);
  e.className = cls;
  parent.appendChild(e);
  return e;
};

/**
 * Minimal fishing overlay: a faint tally in the top left corner (fish caught, biggest fish) that shows for a few
 * seconds after it changes and while Tab is held, one prompt line at the bottom (the cast hint fades after a few
 * seconds), a thin power bar while aiming and a catch card during the hold. DOM writes only when something changed.
 */
export class FishingHud {
  readonly root: HTMLElement;
  private tally: HTMLElement;
  private tabHeld = false;
  private tallyUntil = 0;
  private castSince = -1;
  private count: HTMLElement;
  private best: HTMLElement;
  private prompt: HTMLElement;
  private power: HTMLElement;
  private powerFill: HTMLElement;
  private card: HTMLElement;
  private cardSpecies: HTMLElement;
  private cardMeta: HTMLElement;
  private cardRecord: HTMLElement;
  private bestRef: CaughtFish | null = null;
  private shown = { count: -1, best: '', prompt: '', promptCls: '', powerOn: false, power: -1, card: false, cardKey: '', tint: -1, tally: false };

  constructor(parent: HTMLElement) {
    this.root = el('fh-root', parent);
    const tally = el('fh-tally', this.root);
    this.tally = tally;
    window.addEventListener('keydown', (e) => {
      if (e.code !== 'Tab' || e.ctrlKey || e.altKey || e.metaKey) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      e.preventDefault();
      this.tabHeld = true;
    });
    window.addEventListener('keyup', (e) => {
      if (e.code === 'Tab') this.tabHeld = false;
    });
    window.addEventListener('blur', () => {
      this.tabHeld = false;
    });
    const r1 = el('fh-row', tally);
    el('fh-label', r1, 'span').textContent = 'Fish caught';
    this.count = el('fh-value fh-count', r1, 'span');
    el('fh-rule', tally);
    const r2 = el('fh-row', tally);
    el('fh-label', r2, 'span').textContent = 'Biggest';
    this.best = el('fh-value', r2, 'span');
    this.prompt = el('fh-prompt', this.root);
    this.power = el('fh-power', this.root);
    this.powerFill = el('', this.power, 'i');
    this.card = el('fh-card', this.root);
    this.cardSpecies = el('fh-species', this.card);
    this.cardMeta = el('fh-meta', this.card);
    this.cardRecord = el('fh-record', this.card);
  }

  setVisible(v: boolean) {
    this.root.classList.toggle('fh-hidden', !v);
  }

  update(s: HudInput) {
    const sh = this.shown;
    const now = performance.now();
    if (s.caught !== sh.count) {
      if (sh.count >= 0) this.tallyUntil = now + TALLY_MS;
      sh.count = s.caught;
      this.count.textContent = String(s.caught);
    }
    if (s.biggest !== this.bestRef || sh.best === '') {
      if (sh.best !== '') this.tallyUntil = now + TALLY_MS;
      this.bestRef = s.biggest;
      sh.best = s.biggest ? `${s.biggest.name}, ${s.biggest.lengthCm} cm` : '\u2014';
      this.best.textContent = sh.best;
    }
    // (HUD captures in photo mode, ?fishhud, keep it up)
    const tallyOn = shotMode || this.tabHeld || now < this.tallyUntil;
    if (tallyOn !== sh.tally) {
      sh.tally = tallyOn;
      this.tally.classList.toggle('fh-on', tallyOn);
    }
    // the cast hint is for when he has just got the line in: it fades after a few seconds
    if (s.prompt !== 'cast') this.castSince = -1;
    else if (this.castSince < 0) this.castSince = now;
    const castStale = !shotMode && s.prompt === 'cast' && now - this.castSince > CAST_HINT_MS;

    // one line: a toast (what just happened) wins over the action prompt; hidden while the catch card shows
    const cardOn = (s.state === 'holding' || s.state === 'releasing') && !!s.last;
    let html = '', cls = 'fh-prompt';
    if (!cardOn) {
      if (s.toast) {
        html = s.toast;
        cls += ' fh-on fh-toast';
      } else if (s.prompt && !castStale) {
        html = PROMPTS[s.prompt];
        cls += ' fh-on' + (s.prompt === 'hook' || s.prompt === 'ease' ? ' fh-urgent' : '');
      }
    }
    if (html && html !== sh.prompt) {
      sh.prompt = html;
      this.prompt.innerHTML = html;
    }
    if (cls !== sh.promptCls) {
      sh.promptCls = cls;
      this.prompt.className = cls;
    }
    // reel prompt warms toward red as the line tightens
    const tint = s.prompt === 'reel' || s.prompt === 'ease' ? Math.min(10, Math.round(Math.max(0, s.tension - 0.4) * 16)) : 0;
    if (tint !== sh.tint) {
      sh.tint = tint;
      const k = tint / 10;
      this.prompt.style.color = tint ? `rgb(${239 + 16 * k | 0}, ${230 - 110 * k | 0}, ${210 - 140 * k | 0})` : '';
    }

    const powerOn = s.state === 'aiming';
    if (powerOn !== sh.powerOn) {
      sh.powerOn = powerOn;
      this.power.classList.toggle('fh-on', powerOn);
    }
    const p = Math.round(s.power * 100);
    if (powerOn && p !== sh.power) {
      sh.power = p;
      this.powerFill.style.width = `${p}%`;
    }

    if (cardOn) {
      const c = s.last!;
      const key = `${c.name}|${c.lengthCm}|${s.lastRecord}`;
      if (key !== sh.cardKey) {
        sh.cardKey = key;
        this.cardSpecies.textContent = c.name;
        this.cardMeta.textContent = `${c.lengthCm} cm \u00b7 ${c.weightKg.toFixed(2)} kg`;
        this.cardRecord.textContent = s.lastRecord && s.caught > 1 ? 'Biggest yet' : '';
      }
    }
    if (cardOn !== sh.card) {
      sh.card = cardOn;
      this.card.classList.toggle('fh-on', cardOn);
    }
  }
}
