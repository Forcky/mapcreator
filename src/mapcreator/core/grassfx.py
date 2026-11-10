"""The Grass FX atlas: the blades, tussocks, buttongrass, rushes, sedges and bracken that Custom Shaders Patch
draws on our ground (`core.grass` says which grow where).

CSP's Grass FX takes "a single texture atlas for the whole race track", split into a grid. Its first row is the
basic grass that spawns everywhere; the other rows hold pieces, which a track groups and gives each configuration
a chance of (cup.acstuff.club/docs/csp/tracks/grass-fx, "Custom textures"). Ours is 8 columns by 4 rows of
256 px cells, as CSP's own atlases are 256 px a cell (`grass_fx/highlands.dds` 8 x 3, `generic.dds` 16 x 1, in
acc-extension-config's `textures/`):

- **Row 0:** eight clumps of fine blades, the base grass.
- **Pieces:** a piece is a cell or a block of cells; `PIECES` gives each its place, as CSP's `PIECE_n` counts it
  (columns from 1, rows from 0).
- Every piece stands on the bottom edge of its block, as CSP's do: their cover thickens towards the bottom.

Blade colour comes from the ground. CSP's base blades are a green-only shading mask (red and blue zero) over
alpha, and a piece with any red or blue keeps its own colour instead ("Customly colorized pieces (if color has
red or blue component disabling surface color sync)", CSP 0.3.0 changelog). So most of ours are green only
(`SYNCED`), shaded darker towards the root like an ambient occlusion term. Only the bracken, Themeda's rust seed
heads and buttongrass's flower stalks and buttons are coloured; on CSP 0.2.11 those "fade to green with
distance" (fixed in 0.3.0).

Our own generator, after the idea of CSP's MIT-licensed `ac-grass-gen` (tapered blades with alpha, filled green
only): no code from it is used. Drawn at twice the size and scaled down with alpha premultiplied, so edges are
smooth; deterministic for a seed. Each mip level's transparent pixels take the colour of their cell's pieces
(green only in a synced cell), so a blade's edge never blends into black or into a stray colour.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from mapcreator.core import textures

CELL_PX = 256
COLUMNS, ROWS = 8, 4
SIZE = (COLUMNS * CELL_PX, ROWS * CELL_PX)  # 2048 x 1024
FILE = "grass_fx/mapcreator.dds"  # next to ext_config.ini; a name of our own, not one of CSP's
SUPERSAMPLE = 2
SEED = 7
# Green of a synced blade at its root and at its tip, before each blade's own shade. CSP's own base blades run
# from about 205 at the root to 230 at the tip (median 215; decoded from highlands.dds), and their transparent
# pixels are 223. Ours shade a little more at the root, the denser pieces more again.
ROOT_G, TIP_G = 165.0, 236.0


# How a piece is coloured: green only, so CSP colours it from the ground (SYNCED); synced leaves with coloured
# seed heads or stalks (MIXED); or its own colours throughout (OWN).
SYNCED, MIXED, OWN = "synced", "mixed", "own"
# In a MIXED piece, a pixel whose red and blue are both under this share of its green is leaf, and is made green
# only: so the antialiased edge where a brown stalk crosses a leaf isn't a bright green that keeps its own colour.
LEAF_SHARE = 0.35


@dataclass(frozen=True)
class Piece:
    name: str
    col: int  # CSP's column, counted from 1
    row: int  # CSP's row, counted from 0
    w: int = 1  # cells across
    h: int = 1  # cells down
    colours: str = SYNCED

    def pixels(self) -> tuple[int, int, int, int]:
        """(left, top, width, height) in the atlas."""
        return (self.col - 1) * CELL_PX, self.row * CELL_PX, self.w * CELL_PX, self.h * CELL_PX

    def key(self) -> str:
        """Its place as CSP's `PIECE_n` value: column, row, and its size when bigger than a cell."""
        return f"{self.col}, {self.row}" + (f", {self.w}, {self.h}" if (self.w, self.h) != (1, 1) else "")


# -- drawing ------------------------------------------------------------------------------------------------------

def _green(g: float) -> tuple[int, int, int, int]:
    return 0, int(np.clip(g, 0, 255)), 0, 255


def _rgb(c) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in np.clip(c, 0, 255)) + (255,)


def _stem(base, length: float, lean: float, curl: float, steps: int = 12) -> np.ndarray:
    """Points up a stem from `base` (x, y in image pixels, y down): `lean` is its angle from vertical at the root
    (radians, + to the right) and `curl` how much more it turns by the tip, so a leaf leaning out arches over."""
    t = (np.arange(steps) + 0.5) / steps
    ang = lean + curl * t ** 1.6
    step = np.column_stack([np.sin(ang), -np.cos(ang)]) * length / steps
    return np.vstack([base, base + np.cumsum(step, axis=0)])


def _fit(pts: np.ndarray, box: tuple[float, float], margin: float) -> float:
    """The share of a stem's length that keeps it inside the image (`box` is width, height)."""
    w, h = box
    base = pts[0]
    scale = 1.0
    for p in pts[1:]:
        d = p - base
        for lo, hi, v, b in ((margin, w - margin, d[0], base[0]), (margin, h, d[1], base[1])):
            if b + v < lo:
                scale = min(scale, (lo - b) / v)
            elif b + v > hi:
                scale = min(scale, (hi - b) / v)
    return max(scale, 0.05)


def _fitted(base, length: float, lean: float, curl: float, box: tuple[float, float], margin: float,
            steps: int = 12) -> np.ndarray:
    """`_stem`, shortened if need be to stay `margin` inside the image."""
    pts = _stem(base, length, lean, curl, steps)
    return _stem(base, length * _fit(pts, box, margin), lean, curl, steps)


def _blade(d: ImageDraw.ImageDraw, pts: np.ndarray, width: float, colour: Callable[[float], tuple], *,
           taper: float = 0.9, round_tip: bool = False) -> None:
    """A tapered strip along `pts`, `width` across at the root, drawn as one quad a segment so its colour can
    change along it: `colour(t)` with t from 0 at the root to 1 at the tip."""
    seg = np.diff(pts, axis=0)
    tangent = np.vstack([seg[:1], (seg[1:] + seg[:-1]) / 2, seg[-1:]])
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    t = np.linspace(0.0, 1.0, len(pts))
    half = width / 2 * ((1 - t ** 4) if round_tip else (1 - t) ** taper)
    half = np.maximum(half, 0.35 * SUPERSAMPLE * (t < 1))
    left, right = pts + normal * half[:, None], pts - normal * half[:, None]
    for i in range(len(pts) - 1):
        d.polygon([tuple(left[i]), tuple(left[i + 1]), tuple(right[i + 1]), tuple(right[i])],
                  fill=colour((t[i] + t[i + 1]) / 2))


def _shade(depth: float, rng: np.random.Generator, root: float = ROOT_G, tip: float = TIP_G) -> Callable:
    """A synced blade's green from root to tip: darker at the root, and darker the further back it stands."""
    k = (0.88 + 0.12 * depth) * rng.uniform(0.95, 1.04)
    return lambda t: _green((root + (tip - root) * t ** 0.65) * k)


def _leaves(d, rng, box, n: int, *, spread: float, length: tuple[float, float], lean: float,
            curl: tuple[float, float], width: tuple[float, float], fan: float = 0.0, dome: float = 0.0,
            root: float = ROOT_G, taper: float = 0.9, round_tip: bool = False) -> None:
    """`n` synced leaves from a base around the middle of the bottom edge (`spread`: its spread as a share of the
    width), back ones first. `fan` leans a leaf with where it springs from; `dome` shortens the outer ones, for a
    rounded hummock; `curl` arches each away from upright by a random amount in its range."""
    w, h = box
    order = sorted(rng.uniform(0, 1, n))  # depth: back (dark) first
    for depth in order:
        x = w / 2 + np.clip(rng.normal(0, spread * w), -0.42 * w, 0.42 * w)
        off = (x - w / 2) / (w / 2)
        a = rng.normal(0, lean) + fan * off
        size = h * rng.uniform(*length) * (1 - dome * off ** 2)
        c = np.sign(a or 1.0) * rng.uniform(*curl)
        pts = _fitted(np.array([x, h - 1.0]), size, a, c, box, 2.0 * SUPERSAMPLE)
        _blade(d, pts, rng.uniform(*width) * SUPERSAMPLE, _shade(depth, rng, root=root), taper=taper,
               round_tip=round_tip)


def _canvas(piece: Piece) -> tuple[Image.Image, ImageDraw.ImageDraw, tuple[float, float]]:
    w, h = piece.w * CELL_PX * SUPERSAMPLE, piece.h * CELL_PX * SUPERSAMPLE
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im), (float(w), float(h))


def base_clump(piece: Piece, rng: np.random.Generator) -> Image.Image:
    """12-20 fine blades with varied lean and curl: the grass CSP spawns everywhere."""
    im, d, box = _canvas(piece)
    _leaves(d, rng, box, int(rng.integers(12, 21)), spread=0.1, length=(0.5, 0.97), lean=0.22, curl=(0.0, 0.6),
            width=(6.5, 10.0))
    return im


def poa_tussock(piece: Piece, rng: np.random.Generator) -> Image.Image:
    """Poa: a fountain of fine leaves arching out from a narrow base."""
    im, d, box = _canvas(piece)
    _leaves(d, rng, box, int(rng.integers(70, 121)), spread=0.035, length=(0.45, 1.0), lean=0.55,
            curl=(0.4, 1.5), width=(2.2, 3.4), root=145.0)
    return im


def themeda_tuft(piece: Piece, rng: np.random.Generator) -> Image.Image:
    """Themeda (kangaroo grass): a leafy tuft, and thin culms with rust seed heads (coloured)."""
    im, d, box = _canvas(piece)
    w, h = box
    _leaves(d, rng, box, int(rng.integers(18, 29)), spread=0.06, length=(0.3, 0.6), lean=0.35, curl=(0.2, 0.8),
            width=(3.0, 4.5))
    for _ in range(int(rng.integers(6, 11))):
        x = w / 2 + rng.normal(0, 0.05 * w)
        a = rng.normal(0, 0.18)
        pts = _fitted(np.array([x, h - 1.0]), h * rng.uniform(0.62, 0.95), a, rng.normal(0, 0.15), box,
                      12 * SUPERSAMPLE)
        _blade(d, pts, 1.6 * SUPERSAMPLE, _shade(1.0, rng), taper=0.3)
        tip = pts[-1]
        for k in range(int(rng.integers(3, 6))):  # drooping spikelets in a loose cluster at the top
            r = rng.uniform(2.5, 4.0) * SUPERSAMPLE
            cx, cy = tip[0] + rng.normal(0, 3.0 * SUPERSAMPLE), tip[1] + k * 4.0 * SUPERSAMPLE
            rust = np.array([152.0, 74.0, 38.0]) * rng.uniform(0.85, 1.15)
            d.ellipse([cx - r * 0.6, cy - r, cx + r * 0.6, cy + r * 1.4], fill=_rgb(rust))
    return im


def rank_grass(piece: Piece, rng: np.random.Generator) -> Image.Image:
    """Rank grass gone to seed: long blades and taller culms with open seed heads, all synced."""
    im, d, box = _canvas(piece)
    w, h = box
    _leaves(d, rng, box, int(rng.integers(20, 31)), spread=0.08, length=(0.4, 0.75), lean=0.3, curl=(0.1, 0.7),
            width=(4.0, 6.0))
    for _ in range(int(rng.integers(5, 9))):
        x = w / 2 + rng.normal(0, 0.08 * w)
        a, c = rng.normal(0, 0.15), rng.normal(0, 0.2)
        pts = _fitted(np.array([x, h - 1.0]), h * rng.uniform(0.8, 0.98), a, c, box, 10 * SUPERSAMPLE)
        _blade(d, pts, 1.8 * SUPERSAMPLE, _shade(1.0, rng), taper=0.3)
        for k in range(int(rng.integers(6, 13))):  # the panicle: short branches off the top of the culm
            i = int(len(pts) * rng.uniform(0.75, 0.98))
            side = 1 if k % 2 else -1
            br = _stem(pts[i], h * rng.uniform(0.05, 0.11), a + side * rng.uniform(0.4, 0.9), side * 0.3, steps=4)
            _blade(d, br, 1.2 * SUPERSAMPLE, lambda t: _green(TIP_G * 0.95), taper=0.2)
    return im


def _pinna(d, start: np.ndarray, angle: float, length: float, colour: Callable[[float], tuple],
           rng: np.random.Generator) -> None:
    """One pinna of a bracken frond: a fine midrib with small oblong pinnules either side, angled forward and
    shrinking towards its tip, with gaps between them so the frond reads as lace."""
    pts = _stem(start, length, angle, rng.normal(0, 0.25), steps=6)
    _blade(d, pts, 0.9 * SUPERSAMPLE, colour, taper=0.3)
    seg = np.diff(pts, axis=0)
    total = float(np.linalg.norm(seg, axis=1).sum())
    lobe = max(total * 0.16, 2.6 * SUPERSAMPLE)
    n = max(int(total / (0.6 * lobe)), 2)
    for k in range(n):
        f = (k + 0.5) / n
        i = min(int(f * len(seg)), len(seg) - 1)
        heading = np.arctan2(seg[i][1], seg[i][0]) + np.pi / 2
        size = lobe * (1 - 0.7 * f)
        for side in (-1, 1):
            tip = _stem(pts[i] + seg[i] * (f * len(seg) - i), size, heading + side * 1.1, 0.0, steps=3)
            _blade(d, tip, size * 0.65, lambda t, f=f: colour(f), taper=0.6)


def bracken(piece: Piece, rng: np.random.Generator, browned: float = 0.0) -> Image.Image:
    """Bracken (Pteridium esculentum): fronds on stalks that rise and arch over, each with a triangular blade of
    pinnae, longest at its foot. Coloured, so it keeps its own green whatever the ground; `browned` turns up to
    that share of each pinna, from its tip, rust."""
    im, d, box = _canvas(piece)
    w, h = box
    green, rust = np.array([66.0, 104.0, 38.0]), np.array([150.0, 90.0, 46.0])
    for _ in range(int(rng.integers(4, 7))):
        x = w / 2 + rng.normal(0, 0.14 * w)
        a = rng.normal(0, 0.35)
        c = np.sign(a or 1.0) * rng.uniform(0.9, 1.4)
        pts = _fitted(np.array([x, h - 1.0]), h * rng.uniform(1.0, 1.35), a, c, box, 14 * SUPERSAMPLE, steps=20)
        shade = rng.uniform(0.85, 1.1)
        own = rng.uniform(0.6, 1.0) * browned  # how far this frond has browned

        def colour(t, shade=shade, own=own):
            mix = float(np.clip((t - (1 - own)) / own, 0, 1)) if own > 0 else 0.0
            return _rgb((green * (1 - mix) + rust * mix) * shade)

        _blade(d, pts, 2.6 * SUPERSAMPLE, lambda t, s=shade: _rgb(np.array([118.0, 112.0, 60.0]) * s), taper=0.5)
        seg = np.diff(pts, axis=0)
        reach = float(np.linalg.norm(seg, axis=1).sum())
        first = 7  # the stalk is bare for about a third of its length
        for i in range(first, len(pts) - 1):
            f = (i - first) / (len(pts) - 1 - first)
            heading = np.arctan2(seg[i][1], seg[i][0]) + np.pi / 2  # the rachis's heading, as `_stem` measures it
            length = reach * 0.32 * (1 - f) ** 0.9 + 4 * SUPERSAMPLE
            for side in (-1, 1):
                start = pts[i] + seg[i] * (0.5 if side > 0 else 0.0)  # pinnae alternate a little along the rachis
                _pinna(d, start, heading + side * rng.uniform(0.9, 1.2), length * rng.uniform(0.85, 1.1), colour,
                       rng)
    return im


def bracken_browned(piece: Piece, rng: np.random.Generator) -> Image.Image:
    return bracken(piece, rng, browned=0.6)


def buttongrass(piece: Piece, rng: np.random.Generator) -> Image.Image:
    """A buttongrass (Gymnoschoenus) hummock: a dense dome of fine, wiry, upright leaves."""
    im, d, box = _canvas(piece)
    _leaves(d, rng, box, int(rng.integers(160, 251)), spread=0.16, length=(0.55, 0.92), lean=0.12, curl=(0.0, 0.25),
            width=(1.8, 2.8), fan=0.9, dome=0.55, root=135.0, taper=0.6)
    return im


def buttongrass_flowering(piece: Piece, rng: np.random.Generator) -> Image.Image:
    """A buttongrass hummock (synced) with two to four tall flower stalks, each with its round brown button
    (coloured)."""
    im, d, box = _canvas(piece)
    w, h = box
    _leaves(d, rng, (w, h), int(rng.integers(120, 181)), spread=0.16, length=(0.26, 0.45), lean=0.12,
            curl=(0.0, 0.25), width=(1.8, 2.8), fan=0.9, dome=0.55, root=135.0, taper=0.6)
    for _ in range(int(rng.integers(2, 5))):
        x = w / 2 + rng.normal(0, 0.12 * w)
        a, c = rng.normal(0, 0.1), rng.normal(0, 0.06)
        pts = _fitted(np.array([x, h - 1.0]), h * rng.uniform(0.75, 0.97), a, c, box, 10 * SUPERSAMPLE)
        stalk = np.array([128.0, 100.0, 62.0]) * rng.uniform(0.9, 1.1)
        _blade(d, pts, 2.2 * SUPERSAMPLE, lambda t, s=stalk: _rgb(s * (0.8 + 0.2 * t)), taper=0.2)
        r = rng.uniform(5.5, 7.5) * SUPERSAMPLE
        cx, cy = pts[-1]
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_rgb(np.array([72.0, 54.0, 36.0]) * rng.uniform(0.9, 1.1)))
        for _ in range(6):  # a few paler spikelets on the button
            q = np.array([cx, cy]) + rng.normal(0, r * 0.4, 2)
            s = r * 0.25
            d.ellipse([q[0] - s, q[1] - s, q[0] + s, q[1] + s], fill=_rgb((118, 92, 60)))
    return im


def rushes(piece: Piece, rng: np.random.Generator) -> Image.Image:
    """A tall rush clump: straight, round, upright stems."""
    im, d, box = _canvas(piece)
    _leaves(d, rng, box, int(rng.integers(30, 51)), spread=0.07, length=(0.55, 0.97), lean=0.1, curl=(0.0, 0.08),
            width=(3.0, 4.5), fan=0.4, root=150.0, round_tip=True)
    return im


def sedge(piece: Piece, rng: np.random.Generator) -> Image.Image:
    """A sedge clump: flat leaves arching out, and a few stems with small spikelets near the top."""
    im, d, box = _canvas(piece)
    w, h = box
    _leaves(d, rng, box, int(rng.integers(40, 71)), spread=0.06, length=(0.4, 0.9), lean=0.5, curl=(0.3, 0.9),
            width=(4.0, 6.0), root=150.0)
    for _ in range(int(rng.integers(5, 9))):
        x = w / 2 + rng.normal(0, 0.05 * w)
        a = rng.normal(0, 0.2)
        pts = _fitted(np.array([x, h - 1.0]), h * rng.uniform(0.7, 0.95), a, rng.normal(0, 0.1), box,
                      10 * SUPERSAMPLE)
        _blade(d, pts, 2.0 * SUPERSAMPLE, _shade(1.0, rng), taper=0.3)
        for k in range(int(rng.integers(2, 4))):
            p = pts[-1 - 2 * k]
            r = rng.uniform(3.0, 4.5) * SUPERSAMPLE
            d.ellipse([p[0] - r * 0.6, p[1] - r * 1.4, p[0] + r * 0.6, p[1] + r * 1.4], fill=_green(TIP_G * 0.55))
    return im


# The atlas's layout. The groups and configurations that use these pieces are in `core.grass`.
BASE = tuple(Piece(f"base_{i}", i + 1, 0) for i in range(COLUMNS))
POA, POA_2 = Piece("poa", 1, 1, 2, 1), Piece("poa_2", 3, 1, 2, 1)
THEMEDA, RANK = Piece("themeda", 5, 1, colours=MIXED), Piece("rank", 6, 1)
BRACKEN, BRACKEN_BROWN = Piece("bracken", 7, 1, 2, 1, colours=OWN), Piece("bracken_brown", 7, 2, 2, 1, colours=OWN)
BUTTONGRASS, BUTTONGRASS_2 = Piece("buttongrass", 1, 2, 2, 1), Piece("buttongrass_2", 3, 2, 2, 1)
BUTTONGRASS_FLOWERING = Piece("buttongrass_flowering", 5, 2, 1, 2, colours=MIXED)
RUSHES = Piece("rushes", 6, 2, 1, 2)
SEDGE = Piece("sedge", 1, 3, 2, 1)
# Cells 3-4 and 7-8 of row 3 are spare: room for a dry variant, heath, or spinifex for the mainland.
DRAW = {**{p.name: base_clump for p in BASE}, POA.name: poa_tussock, POA_2.name: poa_tussock,
        THEMEDA.name: themeda_tuft, RANK.name: rank_grass, BRACKEN.name: bracken, BRACKEN_BROWN.name: bracken_browned,
        BUTTONGRASS.name: buttongrass, BUTTONGRASS_2.name: buttongrass,
        BUTTONGRASS_FLOWERING.name: buttongrass_flowering, RUSHES.name: rushes, SEDGE.name: sedge}
PIECES = BASE + (POA, POA_2, THEMEDA, RANK, BRACKEN, BUTTONGRASS, BUTTONGRASS_2, BUTTONGRASS_FLOWERING, RUSHES,
                 BRACKEN_BROWN, SEDGE)
def _downsample(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.convert("RGBa").resize(size, Image.Resampling.LANCZOS).convert("RGBA")


def atlas(seed: int = SEED) -> Image.Image:
    """The atlas, RGBA, SIZE. Pieces are drawn in PIECES order from one generator, so a piece added at the end
    leaves the others as they were."""
    rng = np.random.default_rng(seed)
    out = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    for p in PIECES:
        x, y, w, h = p.pixels()
        out.paste(_downsample(DRAW[p.name](p, rng), (w, h)), (x, y))
    return fill(out)


def _cell_owner() -> np.ndarray:
    """(ROWS, COLUMNS): the index in PIECES of the piece covering each cell, or -1 for a spare cell."""
    owner = np.full((ROWS, COLUMNS), -1)
    for i, p in enumerate(PIECES):
        owner[p.row:p.row + p.h, p.col - 1:p.col - 1 + p.w] = i
    return owner


def fill(level: Image.Image) -> Image.Image:
    """`level` (the atlas or one of its mips) with each transparent pixel set to the mean colour of its piece's
    opaque pixels, green only unless the piece has its own colours, so filtering at a blade's edge stays on its
    colour. A synced piece is made green only throughout, and a mixed one wherever a pixel is leaf
    (`LEAF_SHARE`). Spare cells get a plain synced green."""
    a = np.asarray(level.convert("RGBA")).copy()
    cw, ch = max(a.shape[1] // COLUMNS, 1), max(a.shape[0] // ROWS, 1)
    owner = _cell_owner()
    for i in [*range(len(PIECES)), -1]:
        cells = np.argwhere(owner == i)
        if not len(cells):
            continue
        mask = np.zeros(a.shape[:2], dtype=bool)
        for r, c in cells:
            mask[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = True
        solid = mask & (a[..., 3] >= 128)
        colour = a[solid][:, :3].mean(axis=0) if solid.any() else np.array([0.0, 200.0, 0.0])
        kind = PIECES[i].colours if i >= 0 else SYNCED
        if kind != OWN:
            colour = np.array([0.0, colour[1], 0.0])
        clear = mask & (a[..., 3] == 0)
        a[clear, :3] = np.round(colour).astype(np.uint8)
        if kind == MIXED:
            g = a[..., 1].astype(float)
            mask &= (a[..., 0] < LEAF_SHARE * g) & (a[..., 2] < LEAF_SHARE * g)
        if kind != OWN:
            a[mask, 0] = 0  # DXT keeps an exact 0, so a synced pixel stays green only after compression too
            a[mask, 2] = 0
    return Image.fromarray(a, "RGBA")


def atlas_dds(seed: int = SEED) -> bytes:
    """The atlas as DXT5 with mips, each level filled (`fill`)."""
    return textures.to_dds(atlas(seed), "DXT5", fill=fill)
