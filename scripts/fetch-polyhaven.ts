/**
 * Downloads CC0 assets from Poly Haven into raw/polyhaven (git-ignored and outside public/, so never served) and
 * records each one in assets-manifest.json (used to build CREDITS.md). tools/pack_textures.py and
 * tools/prep_hdri.py turn the raw files into what the game loads.
 *
 * Usage:
 *   pnpm assets:polyhaven                # fetch everything in ASSETS below
 *   pnpm assets:polyhaven river_small_rocks qwantani_morning_puresky
 */
import { mkdir, writeFile, readFile, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';

type Kind = 'hdri' | 'texture' | 'model';
interface AssetReq {
  id: string;
  kind: Kind;
  res: '1k' | '2k' | '4k';
  /** texture maps to keep (Poly Haven map keys) */
  maps?: string[];
  /** hdri file format */
  format?: 'hdr' | 'exr';
  note: string;
}

const UA = 'river-morning-asset-fetch/0.1 (local dev build)';
const ROOT = join(import.meta.dirname, '..');
const OUT = join(ROOT, 'raw', 'polyhaven');
const MANIFEST = join(ROOT, 'assets-manifest.json');

const TEX_MAPS = ['Diffuse', 'nor_gl', 'arm', 'Displacement'];

export const ASSETS: AssetReq[] = [
  // --- skies (part 1: morning; other times of day are added in part 8)
  { id: 'qwantani_morning_puresky', kind: 'hdri', res: '4k', format: 'hdr', note: 'morning sky + image based light' },
  // --- river bed and banks
  { id: 'river_small_rocks', kind: 'texture', res: '2k', maps: TEX_MAPS, note: 'river bed pebbles' },
  { id: 'ganges_river_pebbles', kind: 'texture', res: '2k', maps: TEX_MAPS, note: 'pebble bar / shallow bed' },
  { id: 'gravelly_sand', kind: 'texture', res: '2k', maps: TEX_MAPS, note: 'sand in the deep pool' },
  { id: 'dry_river_pebbles', kind: 'texture', res: '2k', maps: TEX_MAPS, note: 'dry pebble bank above the waterline' },
  { id: 'forest_ground_04', kind: 'texture', res: '2k', maps: TEX_MAPS, note: 'forest floor up the bank' },
  { id: 'brown_mud_leaves_01', kind: 'texture', res: '2k', maps: TEX_MAPS, note: 'muddy needles near the cabin path' },
  { id: 'rocky_terrain_02', kind: 'texture', res: '2k', maps: TEX_MAPS, note: 'steep rocky slopes' },
  { id: 'snow_02', kind: 'texture', res: '1k', maps: TEX_MAPS, note: 'snow on distant peaks' },
  { id: 'aerial_rocks_02', kind: 'texture', res: '1k', maps: TEX_MAPS, note: 'distant mountain rock' },
  // part 1 round 0 self-review: the bank needs pine-forest ground, not sandy soil
  { id: 'forrest_ground_01', kind: 'texture', res: '2k', maps: TEX_MAPS, note: 'grassy bank with twigs and litter' },
  { id: 'forrest_ground_03', kind: 'texture', res: '2k', maps: TEX_MAPS, note: 'pine needle forest floor' },
  { id: 'lichen_rock', kind: 'texture', res: '2k', maps: TEX_MAPS, note: 'steep north bank rock faces' },
  { id: 'mossy_rock', kind: 'texture', res: '2k', maps: TEX_MAPS, note: 'mossy rock faces and boulders' },
  { id: 'rocky_trail', kind: 'texture', res: '1k', maps: TEX_MAPS, note: 'path from the bank to the cabin' },
  { id: 'brown_mud_rocks_01', kind: 'texture', res: '1k', maps: TEX_MAPS, note: 'muddy gravel between bar and grass' },
  { id: 'forest_leaves_02', kind: 'texture', res: '1k', maps: TEX_MAPS, note: 'mossy litter in shade' },
];

async function getJson(url: string): Promise<any> {
  const r = await fetch(url, { headers: { 'User-Agent': UA } });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

async function exists(p: string) {
  try { await stat(p); return true; } catch { return false; }
}

async function download(url: string, dest: string, expectedSize?: number) {
  if (await exists(dest)) {
    const s = await stat(dest);
    if (!expectedSize || s.size === expectedSize) return false;
  }
  await mkdir(dirname(dest), { recursive: true });
  const r = await fetch(url, { headers: { 'User-Agent': UA } });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  const buf = Buffer.from(await r.arrayBuffer());
  await writeFile(dest, buf);
  return true;
}

interface ManifestEntry {
  id: string; kind: Kind; name: string; authors: string[]; license: 'CC0';
  source: string; files: string[]; note: string;
}

async function fetchAsset(a: AssetReq): Promise<ManifestEntry> {
  const info = await getJson(`https://api.polyhaven.com/info/${a.id}`);
  const files = await getJson(`https://api.polyhaven.com/files/${a.id}`);
  const saved: string[] = [];
  if (a.kind === 'hdri') {
    const f = files.hdri[a.res][a.format ?? 'hdr'];
    const rel = `hdri/${a.id}_${a.res}.${a.format ?? 'hdr'}`;
    await download(f.url, join(OUT, rel), f.size);
    saved.push(rel);
  } else if (a.kind === 'texture') {
    for (const map of a.maps ?? TEX_MAPS) {
      const entry = files[map]?.[a.res];
      if (!entry) { console.warn(`  ${a.id}: no ${map} @${a.res}`); continue; }
      const f = entry.jpg ?? entry.png;
      const ext = entry.jpg ? 'jpg' : 'png';
      const rel = `textures/${a.id}/${a.id}_${map.toLowerCase()}_${a.res}.${ext}`;
      await download(f.url, join(OUT, rel), f.size);
      saved.push(rel);
    }
  } else {
    const g = files.gltf[a.res].gltf;
    const base = `models/${a.id}`;
    const rel = `${base}/${a.id}_${a.res}.gltf`;
    await download(g.url, join(OUT, rel), g.size);
    saved.push(rel);
    for (const [incPath, inc] of Object.entries<any>(g.include ?? {})) {
      const relInc = `${base}/${incPath}`;
      await download(inc.url, join(OUT, relInc), inc.size);
      saved.push(relInc);
    }
  }
  const authors = Object.keys(info.authors ?? {});
  console.log(`  ok ${a.id} (${saved.length} files) by ${authors.join(', ')}`);
  return {
    id: a.id, kind: a.kind, name: info.name, authors, license: 'CC0',
    source: `https://polyhaven.com/a/${a.id}`, files: saved, note: a.note,
  };
}

async function main() {
  const only = process.argv.slice(2);
  const list = only.length ? ASSETS.filter((a) => only.includes(a.id)) : ASSETS;
  let manifest: Record<string, ManifestEntry> = {};
  if (await exists(MANIFEST)) manifest = JSON.parse(await readFile(MANIFEST, 'utf8'));
  for (const a of list) {
    try {
      manifest[a.id] = await fetchAsset(a);
    } catch (e) {
      console.error(`  FAILED ${a.id}:`, (e as Error).message);
    }
  }
  await writeFile(MANIFEST, JSON.stringify(manifest, null, 2));
  console.log(`manifest: ${Object.keys(manifest).length} assets`);
}

main();
