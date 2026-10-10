/**
 * Packs the prepared skies for shipping: every `.hdr` a `public/assets/sky/<tod>.json` points at (bg.file, env)
 * becomes `<name>.rgbe.gz` (src/core/rgbe.ts: the same RGBE bytes as four delta-coded planes, gzipped), the json
 * is repointed, and the result is checked byte for byte against the .hdr. Run it after tools/prep_hdri.py; the
 * .hdr files then stay out of public/ (they are only the prep tool's output).
 *
 *   pnpm tsx scripts/pack-sky.ts [--from <dir with the .hdr files>]
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync, gunzipSync } from 'node:zlib';
import { packRGBE, readRGBE } from '../src/core/rgbe';

const args = process.argv.slice(2);
const skyDir = join(import.meta.dirname, '..', 'public', 'assets', 'sky');
const from = args.includes('--from') ? args[args.indexOf('--from') + 1] : skyDir;
const toArrayBuffer = (b: Uint8Array) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;

for (const tod of ['dawn', 'morning', 'golden', 'night']) {
  const jsonPath = join(skyDir, `${tod}.json`);
  let text = readFileSync(jsonPath, 'utf8');
  const meta = JSON.parse(text);
  const repoint = (rel: string) => {
    if (!rel.endsWith('.hdr')) return rel;
    const name = rel.replace(/^sky\//, '').replace(/\.hdr$/, '');
    const src = join(from, `${name}.hdr`);
    if (!existsSync(src)) throw new Error(`missing ${src}`);
    const hdr = readFileSync(src);
    const raw = readRGBE(toArrayBuffer(hdr));
    const gz = gzipSync(packRGBE(raw), { level: 9 });
    const back = readRGBE(toArrayBuffer(gunzipSync(gz)));
    if (back.width !== raw.width || back.height !== raw.height || Buffer.compare(Buffer.from(back.rgbe), Buffer.from(raw.rgbe)) !== 0) {
      throw new Error(`${name}: the packed sky does not decode to the same RGBE bytes`);
    }
    writeFileSync(join(skyDir, `${name}.rgbe.gz`), gz);
    const gzHdr = gzipSync(hdr, { level: 6 }).length;
    console.log(`${name}: ${raw.width}x${raw.height}, .hdr ${(hdr.length / 1e6).toFixed(2)} MB (gzipped ${(gzHdr / 1e6).toFixed(2)} MB)`
      + ` -> .rgbe.gz ${(gz.length / 1e6).toFixed(2)} MB, same RGBE bytes`);
    // only the two paths change in the json, as text, so the prep tool's numbers stay exactly as written
    text = text.replace(`"${rel}"`, `"sky/${name}.rgbe.gz"`);
    return `sky/${name}.rgbe.gz`;
  };
  if (meta.bg?.file) repoint(meta.bg.file);
  if (meta.env) repoint(meta.env);
  writeFileSync(jsonPath, text);
}
