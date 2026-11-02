"""Procedural scenery textures: tree cards, bark and leaves for the 3D trees, and simple building materials.

Drawn in code so there's no licensing question. They're aimed at distance
viewing, where a driver sees scenery at speed. CC0 photo textures can replace
them later without touching the geometry.

Tree cards (the KN5's own trees, which every player sees):
- An atlas of eight variants, 384 x 768 each: five eucalypt shapes, a wattle, a she-oak and a dead stag.
- Each is drawn at twice the size and scaled down, so leaf and branch edges are smooth.
- Clumps behind the branches are darker than those in front, and every clump is lit from the upper left,
  which gives the flat card some depth.

3D trees (Custom Shaders Patch Tree FX, see `treefx`):
- Leaf sprays, pale gum bark and its normal map. Leaves are always greener than red and bark never is,
  because the Tree FX shader tells leaves from bark by that difference.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from mapcreator.core.textures import tileable_noise

SUPERSAMPLE = 2

# -- tree cards ----------------------------------------------------------------

TREE_CELL_W, TREE_CELL_H = 384, 768
TREE_COLS, TREE_ROWS = 4, 2
TREE_VARIANTS = TREE_COLS * TREE_ROWS
CARD_ASPECT = TREE_CELL_W / TREE_CELL_H  # card width / height that shows a variant undistorted

CARD_KINDS = ("euc_tall", "euc_spread", "euc_young", "euc_twin", "euc_dense", "wattle", "sheoak", "stag")
# trunk: where the first limbs leave the trunk (share of height); limbs: count range; angle: limb angle from
# vertical (degrees); clumps: foliage density; leaf: base leaf colour; bark: bark grey level.
_CARD = {
    "euc_tall": dict(trunk=0.50, limbs=(4, 6), angle=(14, 34), clumps=1.0, leaf=(86, 108, 88), bark=178),
    "euc_spread": dict(trunk=0.34, limbs=(4, 6), angle=(30, 56), clumps=1.1, leaf=(92, 112, 84), bark=168),
    "euc_young": dict(trunk=0.24, limbs=(6, 9), angle=(14, 30), clumps=0.9, leaf=(98, 120, 96), bark=186),
    "euc_twin": dict(trunk=0.30, limbs=(4, 5), angle=(20, 40), clumps=1.0, leaf=(84, 104, 82), bark=172, twin=True),
    "euc_dense": dict(trunk=0.32, limbs=(6, 8), angle=(22, 45), clumps=1.5, leaf=(74, 96, 74), bark=150),
    "wattle": dict(trunk=0.16, limbs=(6, 8), angle=(35, 65), clumps=1.9, leaf=(58, 84, 52), bark=92, small=True),
    "sheoak": dict(trunk=0.22, limbs=(5, 7), angle=(25, 50), clumps=1.3, leaf=(80, 94, 76), bark=110, needles=True),
    "stag": dict(trunk=0.30, limbs=(4, 6), angle=(20, 50), clumps=0.1, leaf=(90, 104, 84), bark=168),
}
# Which variants suit short trees (under SHORT_M) and tall ones, and how often each is picked.
SHORT_M = 11.0
_SHORT_WEIGHTS = np.array([0, 0, 0.35, 0, 0.05, 0.35, 0.25, 0])
_TALL_WEIGHTS = np.array([0.24, 0.2, 0.08, 0.12, 0.2, 0, 0.12, 0.04])
_LIGHT = np.array([-0.6, -0.8])  # from the upper left, in image coordinates (y down)


def card_variants(height: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """An atlas variant for each tree: wattles and young trees for short ones, big eucalypts for tall ones."""
    height = np.asarray(height, dtype=float)
    out = rng.choice(TREE_VARIANTS, size=len(height), p=_TALL_WEIGHTS)
    short = height < SHORT_M
    out[short] = rng.choice(TREE_VARIANTS, size=int(short.sum()), p=_SHORT_WEIGHTS)
    return out


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


def _taper(d: ImageDraw.ImageDraw, a, b, w0: float, w1: float, colour) -> None:
    (x0, y0), (x1, y1) = a, b
    length = max(np.hypot(x1 - x0, y1 - y0), 1e-6)
    px, py = -(y1 - y0) / length, (x1 - x0) / length
    d.polygon([(x0 + px * w0 / 2, y0 + py * w0 / 2), (x1 + px * w1 / 2, y1 + py * w1 / 2),
               (x1 - px * w1 / 2, y1 - py * w1 / 2), (x0 - px * w0 / 2, y0 - py * w0 / 2)], fill=colour)


def _skeleton(p: dict, rng: np.random.Generator):
    """Branches and leaf clumps in tree units: base at (0, 0), x to the right, y up, the top near y = 1."""
    segs: list[tuple] = []  # (x0, y0, x1, y1, w0, w1)
    clumps: list[tuple] = []  # (x, y, size)
    w_trunk = 0.022 if p.get("small") else 0.028

    def limb(x, y, ang, length, w, depth):
        n = 4
        for i in range(n):
            ang = ang * 0.9 + rng.normal(0, 0.07)  # limbs curve upward a little
            x1, y1 = x + np.sin(ang) * length / n, y + np.cos(ang) * length / n
            w1 = w * (1 - 0.13 * (i + 1))
            segs.append((x, y, x1, y1, w, w1))
            if depth < 2 and i in (1, 2, 3) and rng.random() < 0.75:
                side = 1 if rng.random() < 0.5 else -1
                limb(x1, y1, ang + side * rng.uniform(0.3, 0.7), length * rng.uniform(0.35, 0.55), w1 * 0.6, depth + 1)
            if i >= (1 if p.get("small") else 2):
                clumps.append((x1, y1, rng.uniform(0.045, 0.075) * (1.15 if depth == 0 else 1.0)
                               * (1.35 if p.get("small") else 1.0)))
            x, y, w = x1, y1, w1

    lean = rng.normal(0, 0.03)
    stems = [(0.0, lean)]
    if p.get("twin"):
        fork = rng.uniform(0.08, 0.16)
        segs.append((0.0, 0.0, lean * fork, fork, w_trunk * 1.2, w_trunk))
        stems = [(lean * fork, lean - 0.07), (lean * fork, lean + 0.07)]
    for x_base, top_x in stems:
        y0 = segs[-1][3] if p.get("twin") else 0.0
        leader_top = rng.uniform(0.78, 0.86)
        w = w_trunk * (0.8 if p.get("twin") else 1.0)
        n = 8
        xs = x_base + (top_x - x_base) * np.linspace(0, 1, n + 1) + np.r_[0, rng.normal(0, 0.006, n)]
        ys = y0 + (leader_top - y0) * np.linspace(0, 1, n + 1)
        ws = w * (1 - 0.7 * np.linspace(0, 1, n + 1))
        for i in range(n):
            segs.append((xs[i], ys[i], xs[i + 1], ys[i + 1], ws[i], ws[i + 1]))
        clumps.append((xs[-1], ys[-1], 0.07))
        k = int(rng.integers(*p["limbs"]) / len(stems)) + 1
        starts = np.sort(rng.uniform(p["trunk"], leader_top - 0.05, k))
        side = 1 if rng.random() < 0.5 else -1
        for y_start in starts:
            side = -side
            i = int(np.clip(np.searchsorted(ys, y_start), 1, n))
            ang = side * np.radians(rng.uniform(*p["angle"]))
            length = (1.0 - y_start) * rng.uniform(0.55, 0.85)
            limb(xs[i], ys[i], ang, length, ws[i] * 0.6, 0)
    return segs, clumps


def _fit(segs, clumps, half_w: float = 0.245, top: float = 0.985):
    """Squeeze the tree so every branch and clump stays inside the card."""
    xs = [abs(s[0]) for s in segs] + [abs(s[2]) for s in segs] + [abs(c[0]) + c[2] for c in clumps]
    ys = [s[3] for s in segs] + [c[1] + c[2] * 0.7 for c in clumps]
    kx = min(1.0, half_w / max(max(xs), 1e-6))
    ky = min(1.0, top / max(max(ys), 1e-6))
    segs = [(x0 * kx, y0 * ky, x1 * kx, y1 * ky, w0, w1) for x0, y0, x1, y1, w0, w1 in segs]
    clumps = [(x * kx, y * ky, r * min(1.0, kx * 1.2)) for x, y, r in clumps]
    return segs, clumps


def _card_tree(kind: str, rng: np.random.Generator) -> Image.Image:
    p = _CARD[kind]
    ss = SUPERSAMPLE
    w, h = TREE_CELL_W * ss, TREE_CELL_H * ss
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    unit = h * 0.985  # pixels per tree unit

    def px(x, y):
        return w / 2 + x * unit, h - y * unit

    segs, clumps = _fit(*_skeleton(p, rng))
    back = rng.random(len(clumps)) < 0.4
    leaf = np.array(p["leaf"], dtype=float)
    grey = p["bark"]
    bark = np.array([grey, grey * 0.96, grey * 0.88])

    def draw_clumps(which, layer_shade):
        for (cx, cy, r), is_back in zip(clumps, back):
            if is_back != which:
                continue
            n = int(np.clip(70 * p["clumps"] * (r / 0.06) ** 2, 8, 220))
            ccx, ccy = px(cx, cy)
            rx, ry = r * unit, r * unit * rng.uniform(0.6, 0.8)
            a = rng.uniform(0, 2 * np.pi, n)
            rr = np.sqrt(rng.uniform(0, 1, n))
            lx, ly = ccx + np.cos(a) * rr * rx, ccy + np.sin(a) * rr * ry
            lit = np.clip(np.cos(a) * rr * _LIGHT[0] + np.sin(a) * rr * _LIGHT[1], -1, 1)
            height_boost = 0.08 * (cy - 0.5)
            for i in range(n):
                shade = layer_shade * (0.8 + 0.26 * lit[i] + height_boost) * rng.uniform(0.9, 1.1)
                col = tuple(int(c) for c in np.clip(leaf * shade * rng.uniform(0.95, 1.05, 3), 0, 255)) + (255,)
                if p.get("needles"):
                    ang = np.pi / 2 + rng.normal(0, 0.25)
                    length = rng.uniform(10, 16) * ss
                    d.line([lx[i], ly[i], lx[i] + np.cos(ang) * length, ly[i] + np.sin(ang) * length],
                           fill=col, width=max(1, int(1.2 * ss)))
                elif p.get("small"):
                    _leaf(d, lx[i], ly[i], rng.uniform(0, 2 * np.pi), rng.uniform(4, 7) * ss, rng.uniform(2.5, 3.5) * ss,
                          col, curve=0.0)
                else:
                    _leaf(d, lx[i], ly[i], np.pi / 2 + rng.normal(0, 0.7), rng.uniform(7, 12) * ss,
                          rng.uniform(2.0, 3.2) * ss, col)

    draw_clumps(True, 0.72)
    dark = tuple(int(c) for c in bark * 0.74) + (255,)
    for x0, y0, x1, y1, w0, w1 in segs:  # dark bodies first, then the lit left sides, so joints don't seam
        _taper(d, px(x0, y0), px(x1, y1), w0 * unit * 1.02, w1 * unit * 1.02, dark)
    for x0, y0, x1, y1, w0, w1 in segs:
        a, b = px(x0, y0), px(x1, y1)
        off = np.array([-0.18 * w0 * unit, 0.0])
        _taper(d, tuple(np.array(a) + off), tuple(np.array(b) + off * (w1 / max(w0, 1e-9))),
               w0 * unit * 0.6, w1 * unit * 0.6, tuple(int(c) for c in np.clip(bark * 1.08, 0, 255)) + (255,))
    draw_clumps(False, 1.0)
    return _downsample(img, (TREE_CELL_W, TREE_CELL_H))


def _downsample(img: Image.Image, size) -> Image.Image:
    return img.convert("RGBa").resize(size, Image.Resampling.LANCZOS).convert("RGBA")


def _bleed(atlas: Image.Image) -> Image.Image:
    """Fill transparent pixels' colour with nearby foliage colour, so mipmaps don't fringe dark."""
    rgb = atlas.convert("RGB").filter(ImageFilter.MaxFilter(9))
    out = Image.composite(atlas.convert("RGB"), rgb, atlas.getchannel("A"))
    out.putalpha(atlas.getchannel("A"))
    return out


def tree_atlas(seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    atlas = Image.new("RGBA", (TREE_CELL_W * TREE_COLS, TREE_CELL_H * TREE_ROWS), (0, 0, 0, 0))
    for i, kind in enumerate(CARD_KINDS):
        atlas.paste(_card_tree(kind, rng), ((i % TREE_COLS) * TREE_CELL_W, (i // TREE_COLS) * TREE_CELL_H))
    return _bleed(atlas)


def tree_uv_cell(variant: int) -> tuple[float, float, float, float]:
    """(u0, v0, u1, v1) of a variant's cell; v = 0 at the top of the image."""
    c, r = variant % TREE_COLS, variant // TREE_COLS
    iu, iv = 1.0 / (TREE_CELL_W * TREE_COLS), 1.0 / (TREE_CELL_H * TREE_ROWS)
    return c / TREE_COLS + iu, r / TREE_ROWS + iv, (c + 1) / TREE_COLS - iu, (r + 1) / TREE_ROWS - iv


# -- 3D tree textures ------------------------------------------------------------

LEAF_CELL = 256
LEAF_GRID = 2  # 2 x 2 sprays: three eucalypt, one wattle
WATTLE_CELL = 3
MIN_LEAF_GREEN = 14  # leaves stay at least this much greener than red (the Tree FX leaf mask)


def _spray(rng: np.random.Generator, wattle: bool) -> Image.Image:
    """A hanging twig of leaves filling one leaf-card cell, twig entering at the top."""
    ss = SUPERSAMPLE
    size = LEAF_CELL * ss
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    twig = (112, 84, 70, 255)
    base = np.array([58.0, 84.0, 52.0]) if wattle else np.array([rng.uniform(84, 96), rng.uniform(106, 118), 88.0])

    def colour(shade):
        c = base * shade * rng.uniform(0.95, 1.05, 3)
        c[1] = max(c[1], c[0] + MIN_LEAF_GREEN)
        return tuple(int(v) for v in np.clip(c, 0, 255)) + (255,)

    twigs = []
    for _ in range(int(rng.integers(6, 9) if wattle else rng.integers(4, 6))):
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
        step = (0.02 if wattle else 0.022) * size
        side = 1
        for s in np.arange(step * 0.7, total, step * rng.uniform(0.8, 1.2)):
            i = min(int(np.searchsorted(np.cumsum(lengths), s)), len(seg) - 1)
            f = (s - (np.cumsum(lengths)[i] - lengths[i])) / max(lengths[i], 1e-9)
            x, y = pts[i] + seg[i] * f
            along = np.arctan2(seg[i][1], seg[i][0])
            side = -side
            if wattle:
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
        if not wattle:  # a tuft at the tip
            x, y = pts[-1]
            for _ in range(int(rng.integers(4, 8))):
                _leaf(d, x, y, np.pi / 2 + rng.normal(0, 0.6), size * rng.uniform(0.1, 0.16), size * 0.032,
                      colour(rng.uniform(0.8, 1.2)), curve=rng.uniform(-0.25, 0.25))
    return _downsample(img, (LEAF_CELL, LEAF_CELL))


def leaf_sprays(seed: int = 21) -> Image.Image:
    rng = np.random.default_rng(seed)
    atlas = Image.new("RGBA", (LEAF_CELL * LEAF_GRID, LEAF_CELL * LEAF_GRID), (0, 0, 0, 0))
    for i in range(LEAF_GRID * LEAF_GRID):
        atlas.paste(_spray(rng, wattle=i == WATTLE_CELL), ((i % LEAF_GRID) * LEAF_CELL, (i // LEAF_GRID) * LEAF_CELL))
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
    hgt = _bark_height(size, seed)
    gx = (np.roll(hgt, -1, axis=1) - np.roll(hgt, 1, axis=1)) * strength
    gy = (np.roll(hgt, -1, axis=0) - np.roll(hgt, 1, axis=0)) * strength
    n = np.stack([-gx, -gy, np.ones_like(hgt)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")


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
