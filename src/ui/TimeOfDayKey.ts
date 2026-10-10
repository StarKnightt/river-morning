import './timeofday.css';
import { shotMode, type TimeOfDay } from '../core/params';
import type { TimeOfDayController } from '../world/TimeOfDay';

const LABELS: [TimeOfDay, string][] = [
  ['dawn', 'Dawn'],
  ['morning', 'Morning'],
  ['golden', 'Golden hour'],
  ['night', 'Night'],
];
/** how long the name stays up after T (s); it then fades out (timeofday.css) */
const LABEL_S = 1.5;

/**
 * T steps the time of day: Dawn -> Morning -> Golden hour -> Night -> Dawn, each through the controller's ~2 s fade
 * (a press during a fade queues the step after the target). Ignored with Ctrl/Alt/Meta, on key repeat, in text
 * fields and in ?shot photo mode (captures use ?tod=). No buttons: the new time's name shows small for ~1.5 s and
 * fades. The URL's ?tod= follows the time once a fade ends.
 */
export function mountTimeOfDayKey(tod: TimeOfDayController, hud: HTMLElement) {
  const label = document.createElement('div');
  label.className = 'tod-label';
  label.setAttribute('aria-live', 'polite');
  hud.appendChild(label);
  let loading: TimeOfDay | null = null;
  let hideTimer = 0;

  const active = () => loading ?? tod.target ?? tod.current;

  const show = (id: TimeOfDay) => {
    label.textContent = LABELS.find(([t]) => t === id)?.[1] ?? '';
    label.dataset.tod = id;
    label.classList.add('is-on');
    clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => label.classList.remove('is-on'), LABEL_S * 1000);
  };

  if (!shotMode) {
    window.addEventListener('keydown', (e) => {
      if (e.code !== 'KeyT' || e.repeat || e.ctrlKey || e.altKey || e.metaKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
      const i = LABELS.findIndex(([id]) => id === active());
      const next = LABELS[(i + 1) % LABELS.length][0];
      void tod.set(next);
      show(next);
    });
  }

  tod.onChange((t, phase) => {
    if (phase === 'loading') loading = t;
    else if (loading === t || phase === 'end') loading = null;
    if (phase === 'end') {
      const url = new URL(location.href);
      url.searchParams.set('tod', tod.current);
      history.replaceState(history.state, '', url);
    }
  });
  return label;
}
