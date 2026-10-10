import type { TimeOfDay } from '../core/params';

/** Hour of the day each ?tod preset stands for. */
export const TOD_HOUR: Record<TimeOfDay, number> = { dawn: 6.0, morning: 8.5, golden: 18.6, night: 23.0 };

/** Linear gains of the time-dependent layers. */
export interface DayMix {
  birds: number;
  crickets: number;
  owl: number;
  wind: number;
}

// (hour, value) pairs, piecewise linear. Birds: a dawn chorus, quieter by midday, few at golden hour, none at night.
// Crickets linger into dawn and start again at dusk. Valley wind is calm at night and dawn, strongest in the afternoon.
const BIRDS = [0, 0, 4.5, 0, 5.5, 0.7, 6.5, 1, 9, 1, 12, 0.6, 16, 0.5, 18, 0.4, 19.5, 0.2, 20.5, 0, 24, 0];
const CRICKETS = [0, 1, 4.5, 1, 6, 0.35, 7.5, 0, 17.5, 0, 19, 0.15, 20.5, 0.8, 21.5, 1, 24, 1];
const OWL = [0, 1, 4.5, 1, 5.5, 0, 20.5, 0, 21.5, 1, 24, 1];
const WIND = [0, 0.55, 6, 0.6, 9, 0.8, 14, 1, 18.5, 0.95, 21, 0.6, 24, 0.55];

function curve(c: number[], h: number) {
  for (let i = 2; i < c.length; i += 2) {
    if (h <= c[i]) return c[i - 1] + ((c[i + 1] - c[i - 1]) * (h - c[i - 2])) / (c[i] - c[i - 2]);
  }
  return c[c.length - 1];
}

export function dayMix(hour: number, out: DayMix): DayMix {
  const h = ((hour % 24) + 24) % 24;
  out.birds = curve(BIRDS, h);
  out.crickets = curve(CRICKETS, h);
  out.owl = curve(OWL, h);
  out.wind = curve(WIND, h);
  return out;
}

export const hourOf = (v: TimeOfDay | number) => (typeof v === 'number' ? v : (TOD_HOUR[v] ?? TOD_HOUR.morning));
