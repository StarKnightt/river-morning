/**
 * The pause panel: Esc opens it, Esc (or a click on the scene) closes it. Graphics: Auto / Low / Medium / High / Ultra,
 * the GPU in use and what Auto picked. A new choice reloads the game (presets change shaders and buffers).
 *
 * Pointer lock: while the mouse is captured Chrome keeps Esc for itself (it releases the lock and the page never sees
 * the key), so the panel opens when the lock is lost and is not taken back by us; with the mouse free, Esc toggles it.
 * Clicking the scene takes the lock again (Player.ts), which closes the panel. Never mounted in photo mode.
 */
import { PRESETS, PRESET_LABEL, Q, chooseMode, qualityState, type Mode } from '../core/quality';
import './quality.css';

export function mountSettingsMenu(engine: { hold: boolean }) {
  if (document.querySelector('.settings-menu')) return;
  const panel = document.createElement('div');
  panel.className = 'settings-menu';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', 'Paused');
  panel.hidden = true;

  const title = document.createElement('div');
  title.className = 'settings-title';
  title.textContent = 'Paused';
  const label = document.createElement('div');
  label.className = 'settings-label';
  label.textContent = 'Graphics';
  const row = document.createElement('div');
  row.className = 'settings-row';
  const buttons = new Map<Mode, HTMLButtonElement>();
  for (const m of ['auto', ...PRESETS] as Mode[]) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'settings-opt';
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      if (m === Q.mode) return;
      note.textContent = 'Reloading…';
      chooseMode(m);
    });
    buttons.set(m, b);
    row.appendChild(b);
  }
  const gpu = document.createElement('div');
  gpu.className = 'settings-info';
  const auto = document.createElement('div');
  auto.className = 'settings-info';
  const note = document.createElement('div');
  note.className = 'settings-note';
  const resume = document.createElement('div');
  resume.className = 'settings-note';
  resume.textContent = 'Esc or click the scene to resume';
  const credits = document.createElement('a');
  credits.className = 'settings-credits';
  credits.href = `${import.meta.env.BASE_URL}CREDITS.md`;
  credits.target = '_blank';
  credits.rel = 'noopener';
  credits.textContent = 'Credits';
  panel.append(title, label, row, auto, gpu, note, resume, credits);
  // on <body>, above the HUD: the HUD itself stays free of controls (the clean screen holds no buttons)
  document.body.appendChild(panel);

  const refresh = () => {
    const s = qualityState();
    for (const [m, b] of buttons) {
      b.textContent = m === 'auto' ? 'Auto' : PRESET_LABEL[m];
      b.setAttribute('aria-pressed', String(m === Q.mode));
    }
    const picked = Q.autoPick ?? (Q.mode === 'auto' ? Q.preset : null);
    auto.textContent = Q.mode === 'auto'
      ? `Auto is using ${PRESET_LABEL[Q.preset]}` + (picked && picked !== Q.preset ? `, ${PRESET_LABEL[picked]} from the next start` : '')
      : `Fixed at ${PRESET_LABEL[Q.preset]}` + (picked ? `  ·  Auto would pick ${PRESET_LABEL[picked]}` : '');
    gpu.textContent = `${s.gpu}  ·  ${s.backend === 'webgpu' ? 'WebGPU' : 'WebGL 2'}  ·  ${s.buffer}${s.scale < 1 ? ` (${Math.round(s.scale * 100)} %)` : ''}`;
    note.textContent = 'Changing the graphics setting restarts the game.';
  };

  let open = false, lastUnlock = 0, wasLocked = false;
  const show = (on: boolean) => {
    if (on === open) return;
    open = on;
    if (on) refresh();
    panel.hidden = !on;
    engine.hold = on;
  };
  window.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || document.pointerLockElement) return;
    // the key that released the lock (if the browser passes it on at all) already opened the panel
    if (performance.now() - lastUnlock < 300) return;
    show(!open);
  });
  document.addEventListener('pointerlockchange', () => {
    if (document.pointerLockElement) {
      wasLocked = true;
      show(false);
      return;
    }
    lastUnlock = performance.now();
    if (wasLocked) show(true);
    wasLocked = false;
  });
}
