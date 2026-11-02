"""Tileable procedural textures, encoded as DXT1 DDS with mipmaps.

Generating our own textures sidesteps imagery licensing entirely. The noise
is shaped in the frequency domain, so it is periodic and tiles seamlessly.
CC0 photo textures can replace these later through the same `Texture` interface.

DDS encoding uses Pillow's DXT1 block encoder (Pillow >= 11.2) one mip level
at a time, then writes a single legacy DDS header holding the whole chain.
That is the layout the game's loader expects for DXT1.
"""

from __future__ import annotations

import io
import struct

import numpy as np
from PIL import Image

DDS_HEADER_FLAGS = 0x000A1007  # CAPS | HEIGHT | WIDTH | PIXELFORMAT | MIPMAPCOUNT | LINEARSIZE
DDS_CAPS = 0x00401008  # TEXTURE | MIPMAP | COMPLEX


def tileable_noise(size: int, beta: float, seed: int) -> np.ndarray:
    """Periodic 1/f^beta noise scaled to [0, 1]."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(f, f)
    radius = np.hypot(fx, fy)
    radius[0, 0] = 1.0
    spectrum = np.fft.fft2(rng.normal(size=(size, size))) / radius**beta
    spectrum[0, 0] = 0
    out = np.real(np.fft.ifft2(spectrum))
    out -= out.min()
    return out / max(out.max(), 1e-12)


def _mix(base: np.ndarray, variation: np.ndarray, amount: np.ndarray) -> np.ndarray:
    rgb = base[None, None, :] * (1 + (variation[..., None] - 0.5) * amount[None, None, :])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def asphalt(size: int = 512, seed: int = 1) -> Image.Image:
    coarse = tileable_noise(size, 1.2, seed)
    grain = tileable_noise(size, 0.2, seed + 1)
    v = 0.55 * coarse + 0.45 * grain
    return Image.fromarray(_mix(np.array([92.0, 92.0, 96.0]), v, np.array([0.45, 0.45, 0.42])))


def grass(size: int = 512, seed: int = 2) -> Image.Image:
    patches = tileable_noise(size, 1.6, seed)
    blades = tileable_noise(size, 0.4, seed + 1)
    v = 0.6 * patches + 0.4 * blades
    dry = tileable_noise(size, 2.0, seed + 2)[..., None]
    green = _mix(np.array([88.0, 118.0, 58.0]), v, np.array([0.5, 0.45, 0.5])).astype(float)
    straw = _mix(np.array([150.0, 140.0, 88.0]), v, np.array([0.35, 0.35, 0.4])).astype(float)
    t = np.clip((dry - 0.55) * 3, 0, 1)
    return Image.fromarray((green * (1 - t) + straw * t).astype(np.uint8))


def ground(size: int = 512, seed: int = 3) -> Image.Image:
    soil = tileable_noise(size, 1.4, seed)
    cover = tileable_noise(size, 1.8, seed + 1)[..., None]
    v = 0.5 * soil + 0.5 * tileable_noise(size, 0.5, seed + 2)
    dirt = _mix(np.array([120.0, 104.0, 78.0]), v, np.array([0.45, 0.45, 0.45])).astype(float)
    veg = _mix(np.array([84.0, 104.0, 60.0]), v, np.array([0.45, 0.4, 0.45])).astype(float)
    t = np.clip((cover - 0.4) * 2.5, 0, 1)
    return Image.fromarray((dirt * (1 - t) + veg * t).astype(np.uint8))


def _blocks(im: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="DDS", pixel_format=fmt)
    data = buf.getvalue()
    header = 148 if data[84:88] == b"DX10" else 128
    return data[header:]


def to_dds(im: Image.Image, fmt: str = "DXT1") -> bytes:
    """DXT1 (opaque) or DXT5 (with alpha) DDS with a mip chain down to 4 px.

    For DXT5 each level is downsized with alpha premultiplied and then
    un-premultiplied, so transparent pixels never bleed dark into a sprite's edges.
    """
    if fmt not in ("DXT1", "DXT5"):
        raise ValueError(f"unsupported DDS format {fmt}")
    w, h = im.size
    if w % 4 or h % 4:
        raise ValueError("DXT textures need sides divisible by 4")
    mode = "RGB" if fmt == "DXT1" else "RGBA"
    levels = []
    cur = im.convert(mode)
    while True:
        levels.append(_blocks(cur, fmt))
        if min(cur.size) <= 4:
            break
        size = (max(4, cur.size[0] // 2), max(4, cur.size[1] // 2))
        if mode == "RGBA":
            cur = cur.convert("RGBa").resize(size, Image.Resampling.LANCZOS).convert("RGBA")
        else:
            cur = cur.resize(size, Image.Resampling.LANCZOS)
    block = 8 if fmt == "DXT1" else 16
    linear = ((w + 3) // 4) * ((h + 3) // 4) * block
    header = b"DDS " + struct.pack("<7I", 124, DDS_HEADER_FLAGS, h, w, linear, 0, len(levels))
    header += b"\0" * 44
    header += struct.pack("<2I", 32, 0x4) + fmt.encode() + b"\0" * 20
    header += struct.pack("<5I", DDS_CAPS, 0, 0, 0, 0)
    assert len(header) == 128
    return header + b"".join(levels)


def to_dds_dxt1(im: Image.Image) -> bytes:
    return to_dds(im, "DXT1")


ROAD_TILE_M = 12.0  # one texture repeat along the road: a 3 m dash and a 9 m gap (Australian rural centre line)
LINE_W_M = 0.10
EDGE_INSET_M = 0.15
DASH_M = 3.0


def road_marked(width_m: float = 6.4, size: int = 512, seed: int = 1) -> Image.Image:
    """Asphalt with white edge lines and a dashed centre line.

    The road mesh stretches u = 0..1 across the carriageway and repeats the
    texture every ROAD_TILE_M along it, so markings sit on the real edges
    whatever the road's width (line widths scale a little with it).
    """
    img = np.asarray(asphalt(size, seed)).astype(float)
    u = (np.arange(size) + 0.5) / size
    v = (np.arange(size) + 0.5) / size
    paint = np.zeros((size, size), dtype=bool)
    half_line = LINE_W_M / width_m / 2
    for centre in (EDGE_INSET_M / width_m, 1 - EDGE_INSET_M / width_m):
        paint[:, np.abs(u - centre) <= half_line] = True
    dash_rows = v < DASH_M / ROAD_TILE_M
    paint[np.ix_(dash_rows, np.abs(u - 0.5) <= half_line)] = True
    wear = tileable_noise(size, 0.8, seed + 9)
    white = np.array([225.0, 225.0, 218.0]) * (0.88 + 0.12 * wear)[..., None]
    img[paint] = white[paint]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def steel(size: int = 256, seed: int = 21) -> Image.Image:
    """Weathered galvanised steel for guard rails."""
    v = 0.6 * tileable_noise(size, 1.2, seed) + 0.4 * tileable_noise(size, 0.3, seed + 1)
    return Image.fromarray(_mix(np.array([168.0, 172.0, 174.0]), v, np.array([0.25, 0.25, 0.25])))


LIBRARY = {"asphalt": asphalt, "road_marked": road_marked, "grass": grass, "ground": ground, "steel": steel}
