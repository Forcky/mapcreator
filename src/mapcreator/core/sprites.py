"""Procedural scenery textures: bark and leaves for the 3D trees, and simple building materials.

Drawn in code so there's no licensing question. CC0 photo textures can replace
them later without touching the geometry.

3D trees (Custom Shaders Patch Tree FX, see `treefx`):
- One leaf atlas and one bark atlas that every tree model shares, with a cell per species, so the models'
  textures are byte for byte the same (CSP can then share them; see `treefx`).
- Leaf sprays are drawn at twice the size and scaled down, so edges are smooth. Leaves are always greener
  than red and bark never is, because the Tree FX shader tells leaves from bark by that difference.
- Bark columns: pale gum, fibrous stringybark, pine plates, and grey rainforest and tea-tree bark. Each
  column tiles along a branch (v); around it (u) a column is used once.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from mapcreator.core.textures import tileable_noise

SUPERSAMPLE = 2

def _leaf(d: ImageDraw.ImageDraw, x: float, y: float, angle: float, length: float, width: float, colour,
          curve: float = 0.15) -> None:
    """A lanceolate (eucalypt) leaf from its stalk at (x, y), pointing along `angle` (image coordinates)."""
    t = np.linspace(0.0, 1.0, 9)
    half = width / 2 * np.sin(np.pi * t) ** 0.7 * (1 - 0.35 * t)
    bend = curve * length * np.sin(np.pi * t / 2) ** 2
    ux, uy = np.cos(angle), np.sin(angle)
    px, py = -uy, ux
    cx, cy = x + ux * t * length + px * bend, y + uy * t * length + py * bend
    left = list(zip(cx + px * half, cy + py * half))
    right = list(zip(cx - px * half, cy - py * half))[::-1]
    d.polygon(left + right, fill=colour)


def _downsample(img: Image.Image, size) -> Image.Image:
    return img.convert("RGBa").resize(size, Image.Resampling.LANCZOS).convert("RGBA")


def _bleed(atlas: Image.Image) -> Image.Image:
    """Fill transparent pixels' colour with nearby foliage colour, so mipmaps don't fringe dark."""
    rgb = atlas.convert("RGB").filter(ImageFilter.MaxFilter(9))
    out = Image.composite(atlas.convert("RGB"), rgb, atlas.getchannel("A"))
    out.putalpha(atlas.getchannel("A"))
    return out


# -- 3D tree textures ------------------------------------------------------------

LEAF_CELL = 256
LEAF_GRID = 3  # 3 x 3 sprays: three eucalypt, then wattle, tea-tree, myrtle, pine, and a second myrtle and pine
WATTLE_CELL, TEA_TREE_CELL, MYRTLE_CELLS, PINE_CELLS = 3, 4, (5, 7), (6, 8)
LEAF_KINDS = ("eucalypt", "eucalypt", "eucalypt", "wattle", "tea_tree", "myrtle", "pine", "myrtle", "pine")
MIN_LEAF_GREEN = 14  # leaves stay at least this much greener than red (the Tree FX leaf mask)
LEAF_BASE = {  # mean leaf colour per kind (RGB); eucalypts vary per spray
    "wattle": (58.0, 84.0, 52.0),
    "tea_tree": (92.0, 110.0, 84.0),  # fine grey-green
    "myrtle": (40.0, 70.0, 36.0),  # dark, glossy
    "pine": (44.0, 72.0, 46.0),
}


def _needles(d: ImageDraw.ImageDraw, x: float, y: float, along: float, size: float, colour, rng) -> None:
    """A tuft of pine needles from a point on a twig, sweeping forward along it."""
    for _ in range(int(rng.integers(5, 9))):
        a = along + rng.normal(0, 0.55)
        length = size * rng.uniform(0.07, 0.11)
        d.line([(x, y), (x + np.cos(a) * length, y + np.sin(a) * length)], fill=colour,
               width=max(2, int(0.9 * SUPERSAMPLE)))


def _spray(rng: np.random.Generator, kind: str) -> Image.Image:
    """A hanging twig of leaves filling one leaf-card cell, twig entering at the top."""
    ss = SUPERSAMPLE
    size = LEAF_CELL * ss
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    twig = (112, 84, 70, 255)
    wattle = kind == "wattle"
    base = np.array(LEAF_BASE[kind]) if kind in LEAF_BASE else np.array([rng.uniform(84, 96), rng.uniform(106, 118), 88.0])

    def colour(shade):
        c = base * shade * rng.uniform(0.95, 1.05, 3)
        c[1] = max(c[1], c[0] + MIN_LEAF_GREEN)
        return tuple(int(v) for v in np.clip(c, 0, 255)) + (255,)

    twigs = []
    count = {"eucalypt": (4, 6), "wattle": (6, 9), "tea_tree": (8, 12), "myrtle": (7, 10), "pine": (5, 8)}[kind]
    for _ in range(int(rng.integers(*count))):
        x0, y0 = size * rng.uniform(0.38, 0.62), size * rng.uniform(0.0, 0.08)
        ang = np.pi / 2 + rng.uniform(-0.95, 0.95)  # from straight down, fanning out
        length = size * rng.uniform(0.55, 0.85)
        bend = rng.normal(0, 0.25)
        t_ = np.linspace(0, 1, 7)
        a = ang + bend * t_
        steps = np.column_stack([np.cos(a), np.sin(a)]) * length / 6
        pts = np.vstack([[x0, y0], [x0, y0] + np.cumsum(steps[:-1], axis=0)])
        pts[:, 0] = np.clip(pts[:, 0], size * 0.06, size * 0.94)
        pts[:, 1] = np.clip(pts[:, 1], 0, size * 0.94)
        twigs.append(pts)
    for pts in twigs:
        d.line([tuple(q) for q in pts], fill=twig, width=max(2, int(1.4 * ss)), joint="curve")
    for pts in twigs:
        seg = np.diff(pts, axis=0)
        lengths = np.hypot(*seg.T)
        total = lengths.sum()
        step = {"eucalypt": 0.022, "wattle": 0.02, "tea_tree": 0.012, "myrtle": 0.016, "pine": 0.03}[kind] * size
        side = 1
        for s in np.arange(step * 0.7, total, step * rng.uniform(0.8, 1.2)):
            i = min(int(np.searchsorted(np.cumsum(lengths), s)), len(seg) - 1)
            f = (s - (np.cumsum(lengths)[i] - lengths[i])) / max(lengths[i], 1e-9)
            x, y = pts[i] + seg[i] * f
            along = np.arctan2(seg[i][1], seg[i][0])
            side = -side
            if kind == "pine":
                _needles(d, x, y, along, size, colour(rng.uniform(0.8, 1.15)), rng)
            elif kind == "tea_tree":  # tiny narrow leaves, close-set all round the twig
                for k in range(2):
                    ang = along + (1 if k else -1) * rng.uniform(0.5, 1.2)
                    _leaf(d, x, y, ang, size * rng.uniform(0.022, 0.034), size * 0.008, colour(rng.uniform(0.8, 1.15)),
                          curve=0.0)
            elif kind == "myrtle":  # small, nearly round, dense
                ang = along + side * rng.uniform(0.6, 1.3)
                _leaf(d, x, y, ang, size * rng.uniform(0.035, 0.05), size * rng.uniform(0.026, 0.034),
                      colour(rng.uniform(0.8, 1.25)), curve=0.0)
            elif wattle:
                for k in range(2):  # feathery: short leaflets both sides
                    ang = along + (1 if k else -1) * rng.uniform(1.0, 1.5)
                    _leaf(d, x, y, ang, size * rng.uniform(0.04, 0.06), size * 0.013, colour(rng.uniform(0.75, 1.1)),
                          curve=0.05)
            else:
                for _ in range(2 if rng.random() < 0.45 else 1):
                    ang = along + side * rng.uniform(0.25, 0.9)
                    ang += (np.pi / 2 - ang) * rng.uniform(0.2, 0.6)  # leaves hang
                    _leaf(d, x, y, ang, size * rng.uniform(0.13, 0.19), size * rng.uniform(0.03, 0.042),
                          colour(rng.uniform(0.72, 1.18)), curve=rng.uniform(-0.25, 0.25))
        if kind == "eucalypt":  # a tuft at the tip
            x, y = pts[-1]
            for _ in range(int(rng.integers(4, 8))):
                _leaf(d, x, y, np.pi / 2 + rng.normal(0, 0.6), size * rng.uniform(0.1, 0.16), size * 0.032,
                      colour(rng.uniform(0.8, 1.2)), curve=rng.uniform(-0.25, 0.25))
    return _downsample(img, (LEAF_CELL, LEAF_CELL))


def leaf_sprays(seed: int = 21) -> Image.Image:
    """The shared leaf atlas. Cells are drawn in order from one generator, so adding cells at the end leaves
    the earlier ones unchanged."""
    rng = np.random.default_rng(seed)
    atlas = Image.new("RGBA", (LEAF_CELL * LEAF_GRID, LEAF_CELL * LEAF_GRID), (0, 0, 0, 0))
    for i, kind in enumerate(LEAF_KINDS):
        atlas.paste(_spray(rng, kind), ((i % LEAF_GRID) * LEAF_CELL, (i // LEAF_GRID) * LEAF_CELL))
    return _bleed(atlas)


def leaf_uv_cell(cell: int) -> tuple[float, float, float, float]:
    c, r = cell % LEAF_GRID, cell // LEAF_GRID
    inset = 1.0 / (LEAF_CELL * LEAF_GRID)
    return c / LEAF_GRID + inset, r / LEAF_GRID + inset, (c + 1) / LEAF_GRID - inset, (r + 1) / LEAF_GRID - inset


def _bark_height(size: tuple[int, int], seed: int) -> np.ndarray:
    w, h = size
    n = tileable_noise(h, 1.5, seed)
    streaks = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((max(w // 6, 4), h)).resize(
        (w, h), Image.Resampling.BILINEAR), dtype=float) / 255.0
    return streaks


def bark(size: tuple[int, int] = (256, 512), seed: int = 23) -> Image.Image:
    """Smooth, pale gum bark with darker ribbons. Red never drops below green: bark isn't leaf."""
    hgt = _bark_height(size, seed)
    pale = np.array([204.0, 198.0, 186.0])
    ribbon = np.array([158.0, 150.0, 132.0])
    t = np.clip((hgt - 0.55) * 4, 0, 1)[..., None]
    fine = tileable_noise(size[1], 0.6, seed + 1)
    fine = np.asarray(Image.fromarray((fine * 255).astype(np.uint8)).resize(size), dtype=float)[..., None] / 255.0
    rgb = (pale * (1 - t) + ribbon * t) * (0.9 + 0.15 * fine)
    rgb[..., 1] = np.minimum(rgb[..., 1], rgb[..., 0])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def bark_normal(size: tuple[int, int] = (256, 512), seed: int = 23, strength: float = 6.0) -> Image.Image:
    return _normal_from(_bark_height(size, seed), strength)


def _normal_from(hgt: np.ndarray, strength: float) -> Image.Image:
    gx = (np.roll(hgt, -1, axis=1) - np.roll(hgt, 1, axis=1)) * strength
    gy = (np.roll(hgt, -1, axis=0) - np.roll(hgt, 1, axis=0)) * strength
    n = np.stack([-gx, -gy, np.ones_like(hgt)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")


BARK_SIZE = (256, 512)  # one column of the bark atlas
SMOOTH_BARK, ROUGH_BARK, PINE_BARK, GREY_BARK = range(4)
BARK_COLUMNS = 4
# Per column after the first: (ridge colour, furrow colour, columns of streaks across the width, rows down the
# height, normal-map strength). Few columns and many rows make vertical fibres; few of both, blocky plates.
_BARKS = {
    ROUGH_BARK: ((122.0, 108.0, 94.0), (70.0, 60.0, 52.0), 40, None, 10.0),  # stringybark and peppermint
    PINE_BARK: ((134.0, 94.0, 72.0), (60.0, 44.0, 36.0), 10, 20, 12.0),  # radiata pine's plates
    GREY_BARK: ((142.0, 134.0, 122.0), (104.0, 98.0, 90.0), 14, None, 4.0),  # myrtle, tea-tree, paperbark
}


def _streaks(size: tuple[int, int], seed: int, across: int, down: int | None) -> np.ndarray:
    w, h = size
    n = tileable_noise(h, 1.5, seed)
    small = Image.fromarray((n * 255).astype(np.uint8)).resize((across, down or h), Image.Resampling.BILINEAR)
    return np.asarray(small.resize((w, h), Image.Resampling.BICUBIC), dtype=float) / 255.0


def _bark_column(column: int, seed: int) -> tuple[Image.Image, Image.Image]:
    if column == SMOOTH_BARK:
        return bark(BARK_SIZE, seed), bark_normal(BARK_SIZE, seed)
    ridge, furrow, across, down, strength = _BARKS[column]
    hgt = _streaks(BARK_SIZE, seed + 7 * column, across, down)
    t = np.clip((hgt - 0.35) * 2.2, 0, 1)[..., None]
    fine = tileable_noise(BARK_SIZE[1], 0.6, seed + 7 * column + 1)
    fine = np.asarray(Image.fromarray((fine * 255).astype(np.uint8)).resize(BARK_SIZE), dtype=float)[..., None] / 255.0
    rgb = (np.array(furrow) * (1 - t) + np.array(ridge) * t) * (0.88 + 0.2 * fine)
    rgb[..., 1] = np.minimum(rgb[..., 1], rgb[..., 0])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB"), _normal_from(hgt, strength)


def bark_atlas(seed: int = 23) -> tuple[Image.Image, Image.Image]:
    """The shared bark atlas and its normal map: BARK_COLUMNS columns side by side, the first today's gum."""
    w, h = BARK_SIZE
    colour, normal = Image.new("RGB", (w * BARK_COLUMNS, h)), Image.new("RGB", (w * BARK_COLUMNS, h))
    for c in range(BARK_COLUMNS):
        col, nm = _bark_column(c, seed)
        colour.paste(col, (c * w, 0))
        normal.paste(nm, (c * w, 0))
    return colour, normal


# Power lines: galvanised steel towers, timber poles and dark wires (`core.power`'s materials).
POWER_COLOURS = {
    "power_metal": (152, 156, 158),
    "power_timber": (104, 86, 66),
    "power_cable": (58, 60, 62),
}


def plain(colour: tuple[int, int, int], size: int = 64, seed: int = 31, noise: float = 6.0) -> Image.Image:
    """A flat colour with a little noise, so a surface doesn't read as untextured."""
    rng = np.random.default_rng(seed)
    px = np.asarray(colour, dtype=float)[None, None, :] + rng.normal(0.0, noise, (size, size, 1))
    return Image.fromarray(np.clip(px, 0, 255).astype(np.uint8), "RGB")


def flat_normal(size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), (128, 128, 255))


# -- buildings --------------------------------------------------------------------

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
