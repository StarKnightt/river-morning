"""Pack Poly Haven texture sets into two RGBA WebP files per material.

  <id>_ch<suffix>.webp  RGB = albedo (sRGB), A = height (linear)
  <id>_nr<suffix>.webp  RG = normal XY (OpenGL, linear), B = roughness, A = ambient occlusion

The game reconstructs normal Z in the shader. Two samples per layer instead of four.

usage:
  python tools/pack_textures.py                                 everything the game loads (src/world/World.ts)
  python tools/pack_textures.py <id>[:size] ...                 native or given size, no suffix
  python tools/pack_textures.py --array 1024 <id> <id> ...      writes <id>_ch_1k.webp for array layers

Only the files the game loads go to public/assets/tex (everything there ships with the site); any other
material or size is written to raw/packed/ instead, so it cannot end up in the build unused.
"""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'raw' / 'polyhaven' / 'textures'  # pnpm assets:polyhaven downloads here
OUT = ROOT / 'public' / 'assets' / 'tex'
SPARE = ROOT / 'raw' / 'packed'
# what World.ts loads: four materials at their own size, seven as 1k array layers
SINGLE = ['ganges_river_pebbles', 'rocky_terrain_02', 'aerial_rocks_02', 'snow_02']
ARRAY_1K = ['dry_river_pebbles', 'gravelly_sand', 'river_small_rocks', 'brown_mud_rocks_01',
            'forrest_ground_01', 'forrest_ground_03', 'lichen_rock']


def find(folder: Path, key: str):
    hits = sorted(folder.glob(f'*_{key}_*'))
    if not hits:
        raise FileNotFoundError(f'{folder.name}: no {key}')
    return hits[0]


def pack(tex_id: str, size: int | None, suffix: str = ''):
    folder = SRC / tex_id
    diff = Image.open(find(folder, 'diffuse')).convert('RGB')
    disp = Image.open(find(folder, 'displacement')).convert('L')
    nor = Image.open(find(folder, 'nor_gl')).convert('RGB')
    arm = Image.open(find(folder, 'arm')).convert('RGB')
    size = size or diff.width
    rs = lambda im: im if im.width == size else im.resize((size, size), Image.LANCZOS)
    diff, disp, nor, arm = rs(diff), rs(disp), rs(nor), rs(arm)
    ch = Image.merge('RGBA', (*diff.split(), disp))
    nx, ny, _ = nor.split()
    ao, rough, _ = arm.split()
    nr = Image.merge('RGBA', (nx, ny, rough, ao))
    shipped = (suffix == '' and tex_id in SINGLE) or (suffix == '_1k' and tex_id in ARRAY_1K)
    out = OUT if shipped else SPARE
    out.mkdir(parents=True, exist_ok=True)
    ch.save(out / f'{tex_id}_ch{suffix}.webp', 'WEBP', quality=90, alpha_quality=100, method=4)
    nr.save(out / f'{tex_id}_nr{suffix}.webp', 'WEBP', quality=92, alpha_quality=100, method=4)
    kb = lambda p: (out / p).stat().st_size // 1024
    print(f'{tex_id} @{size}: ch {kb(f"{tex_id}_ch{suffix}.webp")} KB, nr {kb(f"{tex_id}_nr{suffix}.webp")} KB'
          + ('' if shipped else f'  (not loaded by the game: {out})'), flush=True)


if __name__ == '__main__':
    argv = sys.argv[1:]
    if not argv:
        for tid in SINGLE:
            pack(tid, None)
        for tid in ARRAY_1K:
            pack(tid, 1024, '_1k')
    elif argv[0] == '--array':
        size = int(argv[1])
        for tid in argv[2:]:
            pack(tid, size, f'_{size // 1024}k')
    else:
        for arg in argv:
            tid, _, sz = arg.partition(':')
            pack(tid, int(sz) if sz else None)
