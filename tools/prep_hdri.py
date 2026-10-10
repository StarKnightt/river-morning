"""Prepare a Poly Haven equirect .hdr for the game (one sky per time of day).

Outputs (next to public/assets/sky/ by default, or in --out):
  <name>_env.hdr   2k copy with the sun disc clamped out (image based light without the sun,
                   so the shadow-casting directional light is the only sun term)
  <name>_bg.hdr    (with --bg) the visible sky at full resolution, cropped to the upper hemisphere plus a margin
                   below the horizon (the terrain hides everything lower): ~45 % less memory and download
  <name>.json      sun direction (three.js equirect convention), sun colour and irradiance, sky colours, and the
                   sky rotation about Y that puts the sun on the requested compass bearing

usage:
  python tools/prep_hdri.py raw/polyhaven/hdri/qwantani_morning_puresky_4k.hdr morning
  python tools/prep_hdri.py <src.hdr> <name> [--out DIR] [--bg] [--bearing DEG | --rot DEG] [--night]

  --bearing DEG  compass bearing of the sun (or the pre-sunrise glow) in the game world, clockwise from north
                 (north = -Z, east = +X): stores rotationYDeg = sunPhi - bearing + 90
  --rot DEG      store this rotation directly (night: there is no sun to aim)
  --night        no sun: the brightest point is a star, nothing is measured as a light
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

try:  # OpenCV reads and writes RLE .hdr natively (fast); the numpy reader below is the fallback
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

LUM = np.array([0.2126, 0.7152, 0.0722])
BG_ROWS_FRACTION = 1152 / 2048  # rows kept from the top of the equirect: down to -11.25 deg elevation


def read_hdr(path):
    if cv2 is not None:
        bgr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr[..., ::-1].astype(np.float64)
    data = Path(path).read_bytes()
    head_end = data.index(b'\n\n') + 2
    nl = data.index(b'\n', head_end)
    m = re.match(r'-Y (\d+) \+X (\d+)', data[head_end:nl].decode())
    h, w = int(m.group(1)), int(m.group(2))
    buf = np.frombuffer(data, dtype=np.uint8, offset=nl + 1)
    img = np.zeros((h, w, 4), dtype=np.uint8)
    pos = 0
    for y in range(h):
        pos += 4
        for c in range(4):
            x = 0
            while x < w:
                n = int(buf[pos]); pos += 1
                if n > 128:
                    n -= 128
                    img[y, x:x + n, c] = buf[pos]; pos += 1
                else:
                    img[y, x:x + n, c] = buf[pos:pos + n]; pos += n
                x += n
    e = img[..., 3].astype(np.int32)
    scale = np.where(e == 0, 0.0, np.ldexp(1.0, e - 136))
    return img[..., :3].astype(np.float64) * scale[..., None]


def write_hdr(path, rgb):
    if cv2 is not None:
        # RLE-compressed RGBE (three's HDRLoader and the game's sky decoder read both RLE and flat scanlines)
        if cv2.imwrite(str(path), np.ascontiguousarray(rgb[..., ::-1]).astype(np.float32)):
            return
    h, w, _ = rgb.shape
    mx = rgb.max(axis=2)
    mant, ex = np.frexp(mx)
    scale = np.where(mx > 1e-32, mant * 256.0 / np.maximum(mx, 1e-32), 0.0)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    out[..., 3] = np.where(mx > 1e-32, ex + 128, 0).astype(np.uint8)
    header = f'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n'.encode()
    Path(path).write_bytes(header + out.tobytes())  # flat (non-RLE) scanlines are valid RGBE


def direction(u, v):
    el, phi = (v - 0.5) * math.pi, (u - 0.5) * 2 * math.pi
    return [math.cos(phi) * math.cos(el), math.sin(el), math.sin(phi) * math.cos(el)], el, phi


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def main():
    args = sys.argv[1:]
    flags = {}
    pos = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ('--out', '--bearing', '--rot'):
            flags[a] = args[i + 1]
            i += 2
        elif a.startswith('--'):
            flags[a] = True
            i += 1
        else:
            pos.append(a)
            i += 1
    src, name = pos[0], pos[1]
    out_dir = Path(flags['--out']) if '--out' in flags else Path(src).resolve().parents[1] / 'sky'
    out_dir.mkdir(parents=True, exist_ok=True)
    night = '--night' in flags

    rgb = read_hdr(src)
    h, w, _ = rgb.shape
    lum = rgb @ LUM

    # sun = brightest blob (above 2 % of the peak and at least 50); none in a pre-sunrise sky or at night
    r, c = np.unravel_index(np.argmax(lum), lum.shape)
    thr = max(lum[r, c] * 0.02, 50.0)
    ys, xs = np.nonzero(lum > thr)
    sun_visible = len(ys) > 0 and not night
    lat = (0.5 - (ys + 0.5) / h) * math.pi
    domega = (2 * math.pi / w) * (math.pi / h) * np.cos(lat)
    if sun_visible:
        wts = lum[ys, xs]
        cy, cx = (ys * wts).sum() / wts.sum(), (xs * wts).sum() / wts.sum()
        sun_dir, el, phi = direction((cx + 0.5) / w, 1.0 - (cy + 0.5) / h)
        sun_irr = (rgb[ys, xs] * domega[:, None]).sum(0)
        sun_col = (sun_irr / sun_irr.max()).tolist()
    else:
        # no disc: aim at the brightest part of the horizon band (the glow over the hidden sun), on the horizon
        band = lum[int(h * 0.40):int(h * 0.5)].mean(0)
        k = max(1, w // 32)
        sm = np.convolve(np.concatenate([band[-k:], band, band[:k]]), np.ones(2 * k + 1) / (2 * k + 1), 'same')[k:-k]
        bc = int(np.argmax(sm))
        sun_dir, el, phi = direction((bc + 0.5) / w, 0.5)
        sun_irr = np.zeros(3)
        glow = rgb[int(h * 0.44):int(h * 0.5), max(0, bc - k):bc + k].reshape(-1, 3).mean(0)
        sun_col = (glow / max(glow.max(), 1e-9)).tolist()

    # clamp the sun out of the light probe, downsample to 2k
    clamp = 24.0
    kc = np.minimum(1.0, clamp / np.maximum(lum, 1e-6))
    env = rgb * kc[..., None]
    env2 = env.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    write_hdr(out_dir / f'{name}_env.hdr', env2.astype(np.float32))

    horizon = rgb[int(h * 0.47):int(h * 0.5)].reshape(-1, 3).mean(0)
    zenith = rgb[: int(h * 0.05)].reshape(-1, 3).mean(0)
    # cosine-weighted sky irradiance on a horizontal surface, from the clamped probe (the sun excluded)
    lat_all = (0.5 - (np.arange(h) + 0.5) / h) * math.pi
    wgt = (np.sin(np.maximum(lat_all, 0)) * np.cos(lat_all) * (2 * math.pi / w) * (math.pi / h))[:, None]
    sky_irr_h = float((((env @ LUM)) * wgt).sum())

    meta = {
        'source': Path(src).name,
        'sunDir': [round(x, 5) for x in sun_dir],
        'sunElevationDeg': round(math.degrees(el), 2),
        'sunPhiDeg': round(math.degrees(phi), 3),
        'sunVisible': bool(sun_visible),
        'sunColor': [round(float(x), 4) for x in sun_col],
        'sunIrradiance': [round(float(x), 3) for x in sun_irr],
        'horizonColor': [round(float(x), 4) for x in horizon],
        'zenithColor': [round(float(x), 4) for x in zenith],
        'skyIrradianceH': round(sky_irr_h, 4),
        'envClamp': clamp,
        'env': f'sky/{name}_env.hdr',
    }
    if night:
        meta['brightestPoint'] = {'lum': round(float(lum[r, c]), 2), 'row': int(r), 'col': int(c)}

    if '--bg' in flags:
        rows = int(round(h * BG_ROWS_FRACTION))
        write_hdr(out_dir / f'{name}_bg.hdr', rgb[:rows].astype(np.float32))
        meta['bg'] = {
            'file': f'sky/{name}_bg.hdr', 'width': w, 'height': rows,
            # v (three.js equirect convention, 1 = zenith) of the bottom edge of the cropped image
            'vMin': round(1.0 - rows / h, 6), 'elevationMinDeg': round(90.0 - rows * 180.0 / h, 3),
        }

    if '--bearing' in flags or '--rot' in flags:
        if '--rot' in flags:
            rot = float(flags['--rot'])
        else:
            rot = wrap180(math.degrees(phi) - float(flags['--bearing']) + 90.0)
        th = math.radians(rot)
        sx, sy, sz = sun_dir
        wx, wz = sx * math.cos(th) + sz * math.sin(th), -sx * math.sin(th) + sz * math.cos(th)
        meta['rotationYDeg'] = round(rot, 3)
        meta['sunBearingDeg'] = round(math.degrees(math.atan2(wx, -wz)) % 360.0, 2)
        meta['sunDirWorld'] = [round(wx, 5), round(sy, 5), round(wz, 5)]

    (out_dir / f'{name}.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
