/**
 * Opt-in test conditions for the check scripts (environment variables, so every script takes them the same way):
 *
 *   RM_UNCAPPED=1      no vsync and no frame-rate limit (Chrome --disable-gpu-vsync --disable-frame-rate-limit):
 *                      with a small viewport this 4060 reaches 200+ fps, the 144-240 Hz screens' case
 *   RM_VIEWPORT=WxH    the page's CSS viewport (default 1920x1080), e.g. 960x540 uncapped, 3440x1440 ultrawide
 *   RM_DSF=N           device scale factor (default 1): 1920x1080 at 2 is a 3840x2160 drawing buffer
 */
export const uncapped = process.env.RM_UNCAPPED === '1';

export function extraChromeArgs(): string[] {
  return uncapped ? ['--disable-gpu-vsync', '--disable-frame-rate-limit'] : [];
}

export function viewportOpts(def = { width: 1920, height: 1080 }) {
  const m = /^(\d+)x(\d+)$/.exec(process.env.RM_VIEWPORT ?? '');
  const viewport = m ? { width: Number(m[1]), height: Number(m[2]) } : def;
  const deviceScaleFactor = Number(process.env.RM_DSF ?? 1) || 1;
  if (viewport.width * deviceScaleFactor > 5120 || viewport.height * deviceScaleFactor > 2880) {
    throw new Error(`refusing a drawing buffer over 5120x2880 (${viewport.width}x${viewport.height} at ${deviceScaleFactor})`);
  }
  return { viewport, deviceScaleFactor };
}

export function conditionLabel() {
  const { viewport: v, deviceScaleFactor: d } = viewportOpts();
  return `${v.width}x${v.height}@${d}${uncapped ? ' uncapped' : ''}`;
}
