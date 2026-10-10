/**
 * Sky decode worker: fetches a Radiance .hdr and turns it into half-float RGBA off the main thread (a 4k sky
 * took ~200-400 ms of main-thread time through HDRLoader, a visible hitch when a time of day is chosen).
 */
import { decodeRGBE } from './rgbe';

interface Req {
  id: number;
  url: string;
}

self.onmessage = async (e: MessageEvent<Req>) => {
  const { id, url } = e.data;
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    const out = decodeRGBE(await r.arrayBuffer());
    self.postMessage({ id, ok: true, width: out.width, height: out.height, data: out.data }, { transfer: [out.data.buffer] });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err as Error)?.message ?? err) });
  }
};
