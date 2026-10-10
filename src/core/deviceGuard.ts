import type * as THREE from 'three/webgpu';

/**
 * Keeps the screen from ever going dead: a clear message when WebGPU is missing, a calm overlay with a Reload
 * button when the GPU device is lost (driver reset, GPU hang, sleep), and every uncaptured GPU error logged.
 * Inline styles only, so it shows even if the page's CSS has not loaded.
 */

/** the largest drawing buffer we ever ask for: 5120x2880 (5K), whatever the screen, zoom or `?scale` */
export const MAX_PIXELS = 5120 * 2880;
/** the WebGPU default for maxTextureDimension2D; the device is never asked for more */
export const DEFAULT_MAX_DIM = 8192;

export const gpuErrors = { count: 0, first: [] as string[], lost: null as string | null };

function overlay(title: string, body: string, buttons: Array<{ label: string; onClick: () => void }>) {
  const el = document.createElement('div');
  el.setAttribute('role', 'alertdialog');
  el.style.cssText = 'position:fixed;inset:0;z-index:100000;display:flex;align-items:center;justify-content:center;'
    + 'background:rgba(12,16,18,0.88);color:#e8e4da;font:15px/1.5 Georgia,serif;';
  const card = document.createElement('div');
  card.style.cssText = 'max-width:520px;padding:28px 32px;border:1px solid rgba(232,228,218,0.25);border-radius:6px;'
    + 'background:rgba(20,26,28,0.95);box-shadow:0 8px 40px rgba(0,0,0,0.5);';
  const h = document.createElement('h2');
  h.textContent = title;
  h.style.cssText = 'margin:0 0 12px;font-weight:normal;font-size:22px;';
  const p = document.createElement('div');
  p.innerHTML = body;
  card.append(h, p);
  const row = document.createElement('div');
  row.style.cssText = 'margin-top:20px;display:flex;gap:12px;';
  for (const b of buttons) {
    const btn = document.createElement('button');
    btn.textContent = b.label;
    btn.style.cssText = 'padding:8px 18px;font:inherit;color:#14191b;background:#e8e4da;border:0;border-radius:4px;cursor:pointer;';
    btn.addEventListener('click', b.onClick);
    row.appendChild(btn);
  }
  card.appendChild(row);
  el.appendChild(card);
  document.body.appendChild(el);
  return el;
}

const HOW_TO = 'River Morning needs <b>WebGPU</b>. It runs in current <b>Chrome</b> or <b>Edge</b> (version 113 or newer) '
  + 'on Windows, macOS and ChromeOS, and in recent <b>Firefox</b> and <b>Safari</b> releases.<br><br>'
  + 'If your browser is current: check that hardware acceleration is on (Chrome: Settings &rsaquo; System &rsaquo; '
  + '"Use graphics acceleration when available"), update your graphics driver, and look at <code>chrome://gpu</code> '
  + 'for "WebGPU: Hardware accelerated". On Linux, enable <code>chrome://flags/#enable-unsafe-webgpu</code>.';

/**
 * Resolves with the adapter when WebGPU is usable. Without it, shows the explanation and resolves with null only if
 * the player chooses to try the WebGL 2 fallback anyway.
 */
export async function requireWebGPU(): Promise<GPUAdapter | null> {
  let adapter: GPUAdapter | null = null;
  let reason = '';
  if (!('gpu' in navigator) || !navigator.gpu) reason = 'This browser does not offer WebGPU.';
  else {
    try {
      adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) reason = 'WebGPU is present but no graphics adapter is available (it may be turned off or blocklisted).';
    } catch (e) {
      reason = `WebGPU could not start: ${String((e as Error)?.message ?? e)}`;
    }
  }
  if (adapter) return adapter;
  console.error(`[river-morning] ${reason}`);
  return new Promise((resolve) => {
    const el = overlay('WebGPU is not available', `<p style="margin:0 0 10px">${reason}</p>${HOW_TO}`, [
      { label: 'Reload', onClick: () => location.reload() },
      { label: 'Try anyway', onClick: () => { el.remove(); resolve(null); } },
    ]);
  });
}

/** Device lost and uncaptured GPU errors (three calls renderer.onDeviceLost / onError from the device's handlers). */
export function guardRenderer(renderer: THREE.WebGPURenderer) {
  const r = renderer as unknown as {
    onDeviceLost: (info: { message: string; reason: string | null }) => void;
    onError: (e: { type: string; message: string }) => void;
    setAnimationLoop: (fn: null) => void;
  };
  const baseLost = r.onDeviceLost.bind(renderer), baseError = r.onError?.bind(renderer);
  r.onDeviceLost = (info) => {
    baseLost(info);
    gpuErrors.lost = `${info.reason ?? 'unknown'}: ${info.message}`;
    r.setAnimationLoop(null);
    overlay('The graphics device was reset',
      'The GPU stopped responding to the game (a driver update or reset, the PC waking from sleep, or the GPU '
      + 'running out of memory). Reloading the page brings the river back.'
      + `<p style="margin:10px 0 0;opacity:0.6;font-size:12px">${info.reason ?? ''} ${info.message ?? ''}</p>`,
      [{ label: 'Reload', onClick: () => location.reload() }]);
  };
  r.onError = (e) => {
    gpuErrors.count++;
    if (gpuErrors.first.length < 20) {
      gpuErrors.first.push(`${e.type}: ${e.message}`);
      console.error(`[webgpu] uncaptured ${e.type}: ${e.message}`);
    } else if (gpuErrors.first.length === 20) {
      gpuErrors.first.push('...');
      console.error('[webgpu] more uncaptured errors follow; only the first 20 are logged');
    }
    if (!baseError) return;
    if (gpuErrors.count <= 20) baseError(e);
  };
}

/** a short console note of what the adapter offers against what the game asks for (limits it raises) */
export function logAdapter(adapter: GPUAdapter | null, asked: Record<string, number>) {
  if (!adapter) return;
  const l = adapter.limits as unknown as Record<string, number>;
  const short = Object.entries(asked).filter(([k, v]) => (l[k] ?? 0) < v);
  if (short.length) console.warn('[river-morning] adapter below requested limits:', Object.fromEntries(short.map(([k]) => [k, l[k]])));
  (window as unknown as { __RM_GPU?: unknown }).__RM_GPU = {
    asked, maxTextureDimension2D: l.maxTextureDimension2D, maxSampledTexturesPerShaderStage: l.maxSampledTexturesPerShaderStage,
    maxSamplersPerShaderStage: l.maxSamplersPerShaderStage, errors: gpuErrors,
  };
}

/**
 * The pixel ratio for a w x h CSS-pixel canvas: the wanted ratio, lowered until the drawing buffer fits the
 * device's 2D texture limit on both sides and MAX_PIXELS in total (5K or 8K screens, ultrawides at 200 %, or
 * browser zoom-out can otherwise ask for a canvas WebGPU refuses).
 */
export function fitPixelRatio(w: number, h: number, want: number, maxDim = DEFAULT_MAX_DIM) {
  let pr = Math.max(0.1, want);
  const cw = Math.max(1, w), ch = Math.max(1, h);
  pr = Math.min(pr, (maxDim - 1) / cw, (maxDim - 1) / ch);
  const px = cw * ch * pr * pr;
  if (px > MAX_PIXELS) pr *= Math.sqrt(MAX_PIXELS / px);
  return pr;
}
