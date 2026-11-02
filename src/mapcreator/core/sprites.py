"""Procedural scenery textures: eucalypt-style tree cards and simple building materials.

Drawn in code so there's no licensing question. They're aimed at distance
viewing, where a driver sees scenery at speed. CC0 photo textures can replace
them later without touching the geometry.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from mapcreator.core.textures import tileable_noise

TREE_CELL = 512
TREE_GRID = 2  # 2 x 2 variants in one atlas


def _tree(rng: np.random.Generator, size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = size / 2 + rng.normal(0, size * 0.02)
    # Eucalypts: tall, pale, often leaning trunks with an open, clumpy crown.
    trunk_top = size * rng.uniform(0.32, 0.45)
    base_w = size * rng.uniform(0.035, 0.05)
    lean = rng.normal(0, size * 0.03)
    grey = rng.uniform(140, 190)
    bark = np.array([grey, grey * rng.uniform(0.93, 0.98), grey * rng.uniform(0.82, 0.9)])
    steps = 40
    for i in range(steps):
        t0, t1 = i / steps, (i + 1) / steps
        y0 = size - t0 * (size - trunk_top)
        y1 = size - t1 * (size - trunk_top)
        w = base_w * (1 - 0.6 * t0)
        x = cx + lean * t0
        shade = bark * rng.uniform(0.85, 1.05)
        d.rectangle([x - w / 2, y1, x + w / 2, y0], fill=(*shade.astype(int), 255))
    # A few limbs from the upper trunk into the crown.
    for _ in range(rng.integers(3, 6)):
        t = rng.uniform(0.55, 0.95)
        x0 = cx + lean * t
        y0 = size - t * (size - trunk_top)
        ang = rng.uniform(-2.4, -0.7) if rng.random() < 0.5 else rng.uniform(-2.4, -0.7) * -1 - np.pi
        length = size * rng.uniform(0.12, 0.25)
        x1, y1 = x0 + np.cos(ang) * length, y0 + np.sin(ang) * length
        d.line([x0, y0, x1, y1], fill=(*(bark * 0.9).astype(int), 255), width=max(3, int(base_w * 0.35)))
    # Crown: several separate clumps of small leaf blobs, with sky showing between them.
    crown_cx, crown_cy = cx + lean, trunk_top
    base_leaf = np.array([rng.uniform(70, 95), rng.uniform(92, 115), rng.uniform(72, 92)])
    for _ in range(int(rng.integers(4, 8))):
        kx = crown_cx + rng.normal(0, size * 0.14)
        ky = crown_cy - size * rng.uniform(0.0, 0.22)
        krx, kry = size * rng.uniform(0.07, 0.14), size * rng.uniform(0.05, 0.09)
        for _ in range(int(rng.integers(40, 70))):
            a = rng.uniform(0, 2 * np.pi)
            r = np.sqrt(rng.uniform(0, 1))
            x, y = kx + np.cos(a) * r * krx, ky + np.sin(a) * r * kry
            s = size * rng.uniform(0.01, 0.026)
            tint = base_leaf * rng.uniform(0.72, 1.18) + (14 if y < ky else 0)
            d.ellipse([x - s, y - s * 0.7, x + s, y + s * 0.7], fill=(*np.clip(tint, 0, 255).astype(int), 255))
    return img


def tree_atlas(seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    size = TREE_CELL * TREE_GRID
    atlas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for i in range(TREE_GRID * TREE_GRID):
        atlas.paste(_tree(rng, TREE_CELL), ((i % TREE_GRID) * TREE_CELL, (i // TREE_GRID) * TREE_CELL))
    # Fill transparent pixels' colour with nearby foliage colour so mipmaps don't fringe dark.
    rgb = atlas.convert("RGB").filter(ImageFilter.MaxFilter(9))
    out = Image.composite(atlas.convert("RGB"), rgb, atlas.getchannel("A"))
    out.putalpha(atlas.getchannel("A"))
    return out


def tree_uv_cell(variant: int) -> tuple[float, float, float, float]:
    """(u0, v0, u1, v1) of a variant's cell; v = 0 at the top of the image."""
    c, r = variant % TREE_GRID, variant // TREE_GRID
    inset = 1.0 / (TREE_CELL * TREE_GRID)
    return c / TREE_GRID + inset, r / TREE_GRID + inset, (c + 1) / TREE_GRID - inset, (r + 1) / TREE_GRID - inset


def weatherboard(size: int = 256, seed: int = 11) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.array([228.0, 222.0, 205.0]) * rng.uniform(0.95, 1.0)
    y = np.arange(size)[:, None]
    boards = 0.88 + 0.12 * ((y % 32) / 32.0)  # 8 boards per tile, each shaded top to bottom
    grain = tileable_noise(size, 1.0, seed)[..., None] * 0.08 + 0.96
    img = base * boards[..., None] * grain
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGB")


ROOF_COLOURS = {
    "zincalume": (176, 180, 182),
    "red": (128, 52, 44),
    "green": (66, 92, 70),
    "charcoal": (64, 66, 70),
}


def corrugated(colour: tuple[int, int, int], size: int = 256, seed: int = 13) -> Image.Image:
    x = np.arange(size)[None, :]
    ribs = 0.85 + 0.15 * (0.5 + 0.5 * np.cos(2 * np.pi * x / 16.0))
    weather = tileable_noise(size, 1.5, seed)[..., None] * 0.1 + 0.95
    img = np.array(colour, dtype=float) * ribs[..., None] * weather
    return Image.fromarray(np.clip(np.broadcast_to(img, (size, size, 3)), 0, 255).astype(np.uint8))
