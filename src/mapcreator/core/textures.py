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
from functools import partial

import numpy as np
from PIL import Image

DDS_HEADER_FLAGS = 0x000A1007  # CAPS | HEIGHT | WIDTH | PIXELFORMAT | MIPMAPCOUNT | LINEARSIZE
DDS_CAPS = 0x00401008  # TEXTURE | MIPMAP | COMPLEX


def tileable_noise(size: int | tuple[int, int], beta: float, seed: int) -> np.ndarray:
    """Periodic 1/f^beta noise scaled to [0, 1]: square, or (rows, columns). It wraps along both axes."""
    rows, cols = (int(size), int(size)) if np.ndim(size) == 0 else (int(size[0]), int(size[1]))
    rng = np.random.default_rng(seed)
    fx, fy = np.meshgrid(np.fft.fftfreq(cols), np.fft.fftfreq(rows))
    radius = np.hypot(fx, fy)
    radius[0, 0] = 1.0
    spectrum = np.fft.fft2(rng.normal(size=(rows, cols))) / radius**beta
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


def _grass_rgb(size: int | tuple[int, int], seed: int) -> np.ndarray:
    patches = tileable_noise(size, 1.6, seed)
    blades = tileable_noise(size, 0.4, seed + 1)
    v = 0.6 * patches + 0.4 * blades
    dry = tileable_noise(size, 2.0, seed + 2)[..., None]
    green = _mix(np.array([88.0, 118.0, 58.0]), v, np.array([0.5, 0.45, 0.5])).astype(float)
    straw = _mix(np.array([150.0, 140.0, 88.0]), v, np.array([0.35, 0.35, 0.4])).astype(float)
    t = np.clip((dry - 0.55) * 3, 0, 1)
    return green * (1 - t) + straw * t


def grass(size: int = 512, seed: int = 2) -> Image.Image:
    return Image.fromarray(_grass_rgb(size, seed).astype(np.uint8))


def _ground_layers(size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The terrain texture's bare soil, its vegetation, and how much vegetation covers each pixel."""
    soil = tileable_noise(size, 1.4, seed)
    cover = tileable_noise(size, 1.8, seed + 1)[..., None]
    v = 0.5 * soil + 0.5 * tileable_noise(size, 0.5, seed + 2)
    dirt = _mix(np.array([120.0, 104.0, 78.0]), v, np.array([0.45, 0.45, 0.45])).astype(float)
    veg = _mix(np.array([84.0, 104.0, 60.0]), v, np.array([0.45, 0.4, 0.45])).astype(float)
    return dirt, veg, np.clip((cover - 0.4) * 2.5, 0, 1)


def ground(size: int = 512, seed: int = 3) -> Image.Image:
    dirt, veg, t = _ground_layers(size, seed)
    return Image.fromarray((dirt * (1 - t) + veg * t).astype(np.uint8))


# The ground of each grass class (`core.grass`): a base colour, then patches over it, each as (colour, share of the
# texture, noise exponent: higher makes bigger, smoother patches). CSP's Grass FX colours its blades from the ground
# under them and grows them only where that colour passes the class's mask, so the patches decide where grass
# grows as well as how the class looks beyond the grass. A class's texture repeats every 16 m with warped
# coordinates (`mesh.warped_uv`), and its patches are small beside that (exponents near 1): with one big patch a
# repeat, the repeat itself shows as rows across open ground. Pasture and towns use today's `ground`. Starting
# points, to tune in game with the Grass FX debug app (`docs/known-issues.md`).
GROUND_COVERS = {
    # Saltmarsh, cushion moor and other short herbland: green and fine, hardly any dry patches.
    "ground_short": ((82, 124, 54), (((104, 136, 66), 0.3, 0.9), ((146, 140, 92), 0.05, 1.2))),
    # Native grassland: straw, olive and blue-green tussocks, with a little bare soil.
    "ground_tussock": ((158, 146, 96), (((124, 124, 76), 0.35, 1.0), ((110, 122, 96), 0.2, 1.1),
                                         ((122, 108, 82), 0.06, 0.9))),
    # Buttongrass moorland: olive-tan, with small dark peat pools that stay bare (below the configuration's
    # luminance).
    "ground_moor": ((136, 124, 80), (((112, 114, 66), 0.3, 1.0), ((48, 40, 32), 0.14, 0.9),
                                      ((164, 148, 100), 0.12, 0.8))),
    # Sedgeland and wetland: dark green, with water-dark patches.
    "ground_wet": ((60, 82, 46), (((86, 96, 54), 0.25, 1.0), ((34, 42, 36), 0.16, 1.0))),
    # Bracken fernland: green-brown, with russet dead fronds.
    "ground_bracken": ((92, 100, 54), (((124, 90, 56), 0.3, 1.1), ((64, 86, 42), 0.15, 0.9))),
    # Regrowth, weeds and easements: rank grass between pasture and bracken, with scrub.
    "ground_rough": ((88, 108, 58), (((142, 132, 88), 0.15, 1.1), ((118, 90, 58), 0.12, 1.0),
                                      ((62, 80, 46), 0.15, 1.1))),
    # Grassy woodland: mostly leaf litter, with straw-green grassy patches.
    "ground_woodland": ((116, 98, 74), (((110, 114, 68), 0.35, 1.1), ((76, 64, 50), 0.1, 0.9),
                                         ((136, 126, 110), 0.06, 0.6))),
    # Forest, plantation and scrub floor: grey-brown litter and bark flakes.
    "ground_forest": ((96, 82, 64), (((66, 56, 46), 0.25, 1.0), ((134, 120, 100), 0.06, 0.6))),
    # Forest seen from afar, where it has few or no trees: a dark canopy green.
    "ground_canopy": ((46, 62, 38), (((34, 46, 30), 0.3, 1.1), ((60, 78, 46), 0.15, 1.0))),
    # Sand, mud, rock and pans.
    "ground_bare": ((150, 142, 124), (((124, 122, 116), 0.3, 1.1), ((98, 92, 82), 0.1, 0.9))),
}
PATCH_EDGE = 0.12  # how soft a patch's edge is, as a share of the pixels ranked across it


def ground_cover(name: str, size: int = 512, seed: int | None = None) -> Image.Image:
    """The ground texture `name` from GROUND_COVERS: tileable, a base colour with each patch layer over it."""
    base, patches = GROUND_COVERS[name]
    seed = 50 + 20 * list(GROUND_COVERS).index(name) if seed is None else seed
    grain = 0.55 * tileable_noise(size, 1.5, seed) + 0.45 * tileable_noise(size, 0.4, seed + 1)
    rgb = _mix(np.array(base, dtype=float), grain, np.full(3, 0.4)).astype(float)
    for k, (colour, share, beta) in enumerate(patches):
        rank = _uniform(tileable_noise(size, beta, seed + 2 + k))
        t = np.clip((rank - (1 - share)) / PATCH_EDGE + 0.5, 0.0, 1.0)[..., None]  # about `share` of the pixels
        rgb = rgb * (1 - t) + _mix(np.array(colour, dtype=float), grain, np.full(3, 0.35)) * t
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _blocks(im: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="DDS", pixel_format=fmt)
    data = buf.getvalue()
    header = 148 if data[84:88] == b"DX10" else 128
    return data[header:]


def to_dds(im: Image.Image, fmt: str = "DXT1", fill=None) -> bytes:
    """DXT1 (opaque) or DXT5 (with alpha) DDS with a mip chain down to 4 px.

    For DXT5 each level is downsized with alpha premultiplied and then
    un-premultiplied, so transparent pixels never bleed dark into a sprite's edges.
    `fill`, if given, takes each level's image and returns it with its transparent pixels' colour set, before
    it's encoded: the samples beside a sprite's edge then blend into its own colour, not black.
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
        if fill is not None:
            cur = fill(cur)
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
# The carriageway each sealed texture is drawn for: u = 0..1 stretches it to the road's real width.
ROAD_MARKED_WIDTH_M = 6.4
ROAD_PLAIN_WIDTH_M = 4.0


def _sealed(width_m: float, size: int, seed: int, centre_line: bool) -> Image.Image:
    img = np.asarray(asphalt(size, seed)).astype(float)
    u = (np.arange(size) + 0.5) / size
    v = (np.arange(size) + 0.5) / size
    paint = np.zeros((size, size), dtype=bool)
    half_line = LINE_W_M / width_m / 2
    for centre in (EDGE_INSET_M / width_m, 1 - EDGE_INSET_M / width_m):
        paint[:, np.abs(u - centre) <= half_line] = True
    if centre_line:
        dash_rows = v < DASH_M / ROAD_TILE_M
        paint[np.ix_(dash_rows, np.abs(u - 0.5) <= half_line)] = True
    wear = tileable_noise(size, 0.8, seed + 9)
    white = np.array([225.0, 225.0, 218.0]) * (0.88 + 0.12 * wear)[..., None]
    img[paint] = white[paint]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def road_marked(width_m: float = ROAD_MARKED_WIDTH_M, size: int = 512, seed: int = 1) -> Image.Image:
    """Asphalt with white edge lines and a dashed centre line.

    The road mesh stretches u = 0..1 across the carriageway and repeats the
    texture every ROAD_TILE_M along it, so markings sit on the real edges
    whatever the road's width (line widths scale a little with it).
    """
    return _sealed(width_m, size, seed, centre_line=True)


def road_plain(width_m: float = ROAD_PLAIN_WIDTH_M, size: int = 512, seed: int = 1) -> Image.Image:
    """Asphalt with edge lines and no centre line, for single-lane seals (`lanes=1` or `lane_markings=no`).
    Australian narrow seals generally carry no centre line; most single-lane ones in Tasmania have no edge lines
    either, but v1 keeps them."""
    return _sealed(width_m, size, seed, centre_line=False)


def edge_line_u(texture: str) -> float:
    """How far in from u = 0 a sealed texture's edge line reaches (and out from u = 1 on the other side): its
    outer edge, as a share of the carriageway. 0 for a texture with no edge lines."""
    width = {"road_marked": ROAD_MARKED_WIDTH_M, "road_plain": ROAD_PLAIN_WIDTH_M}.get(texture)
    return 0.0 if width is None else (EDGE_INSET_M + LINE_W_M / 2) / width


GRAVEL_TILE_M = 24.0  # one gravel repeat along the road: twice ROAD_TILE_M, so the period shows less
GRAVEL_SIZE = (2048, 512)  # (rows along the road, columns across it): 1.2 cm per pixel along, 0.9-1.2 cm across
GRAVEL_WIDTH_M = {1: 4.5, 2: 6.0}  # the carriageway each layout is drawn for; u = 0..1 stretches it to the road's
# The palette comes from the CC0 photo texture "gravel_ground_01" on Poly Haven (diffuse, 1k; Nov 2026): the mean of
# its greyer and browner halves (split by saturation after a 12 px blur), and of its lightest 5% and darkest 4% of
# pixels for stones. A grey-brown, as Tasmanian dolerite gravel is (to confirm in game). No pixels are copied.
GRAVEL_GREY = np.array([149.0, 135.0, 111.0])
GRAVEL_BROWN = np.array([144.0, 127.0, 98.0])
STONE_LIGHT = np.array([190.0, 177.0, 156.0])
STONE_DARK = np.array([98.0, 80.0, 50.0])
WHEEL_PATH_FROM_LANE_CENTRE_M = 0.8
WHEEL_PATH_HALF_M = 0.25  # each wheel path about 0.5 m wide
GRASS_STRIP_HALF_M = 0.15  # a single-lane road's centre strip, about 0.3 m wide


def _band(x: np.ndarray, centre: float, half: float, soft: float) -> np.ndarray:
    """A soft-edged stripe along the road: 1 up to `half - soft` from `centre`, easing to 0 at `half + soft`."""
    t = np.clip((half + soft - np.abs(x - centre)) / (2 * soft), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _uniform(a: np.ndarray) -> np.ndarray:
    """Each value's rank, scaled to [0, 1): thresholds on it pick an exact share of the pixels."""
    flat = a.ravel()
    out = np.empty(flat.size)
    out[flat.argsort(kind="stable")] = np.arange(flat.size) / flat.size
    return out.reshape(a.shape)


def gravel_road(lanes: int = 2, size: tuple[int, int] = GRAVEL_SIZE, seed: int = 41) -> Image.Image:
    """A gravel road seen from above: u = 0..1 across the carriageway (as `road_marked`), repeating every
    GRAVEL_TILE_M along it. Every layer is periodic along the road, so the texture tiles without a seam.

    - **Base:** a grey and a brown mixed by low-frequency noise, with brightness patches and grain.
    - **Stones:** the lightest 5.5% and darkest 3.5% of mid-frequency noise, 1-3 px (1-3 cm) each.
    - **Wheel paths:** a pair per lane, 0.8 m either side of its centre and about 0.5 m wide: compacted fines,
      6% darker, with 30% of the stones.
    - **Loose ridges:** between the paths of a pair, between the lanes, and within 0.4 m of each edge: a touch
      lighter, with 180% of the stones.
    - **Potholes:** a few darker patches in the running surface.
    - **One lane only:** a patchy grass strip about 0.3 m wide down the middle and green fringes at the edges,
      coloured as the `grass` texture. CSP's Grass FX grows blades on green, so with the material in
      `GRASS_MATERIALS` they grow there and nowhere else on the road.
    """
    rows, cols = size
    width = GRAVEL_WIDTH_M[lanes]
    x = ((np.arange(cols) + 0.5) / cols - 0.5) * width  # metres from the centreline, + = right in the image
    lane_centres = [0.0] if lanes == 1 else [-width / 4, width / 4]
    path = np.zeros(cols)
    ridge = np.zeros(cols)
    for c in lane_centres:
        for side in (-1.0, 1.0):
            path = np.maximum(path, _band(x, c + side * WHEEL_PATH_FROM_LANE_CENTRE_M, WHEEL_PATH_HALF_M, 0.12))
        ridge = np.maximum(ridge, _band(x, c, 0.25, 0.15))
    if lanes == 2:
        ridge = np.maximum(ridge, _band(x, 0.0, 0.2, 0.15))
    edge = np.maximum(_band(x, -width / 2, 0.4, 0.15), _band(x, width / 2, 0.4, 0.15))
    ridge = np.maximum(ridge, edge)

    hue = tileable_noise(size, 1.6, seed)[..., None]
    bright = 0.6 * tileable_noise(size, 1.6, seed + 1) + 0.4 * tileable_noise(size, 0.3, seed + 2)
    rgb = GRAVEL_GREY * (1 - hue) + GRAVEL_BROWN * hue
    rgb = rgb * (1 + (bright[..., None] - 0.5) * 0.35)
    rgb = rgb * (1 - 0.06 * path + 0.04 * ridge)[None, :, None]

    density = (1.0 - 0.7 * path + 0.8 * ridge)[None, :]
    rank = _uniform(tileable_noise(size, 0.6, seed + 3))
    grain = tileable_noise(size, 0.2, seed + 4)[..., None]
    light = rank > 1 - 0.055 * density
    dark = rank < 0.035 * density
    rgb = np.where(light[..., None], 0.75 * STONE_LIGHT * (0.92 + 0.16 * grain) + 0.25 * rgb, rgb)
    rgb = np.where(dark[..., None], 0.75 * STONE_DARK * (0.9 + 0.2 * grain) + 0.25 * rgb, rgb)

    pot = tileable_noise(size, 2.0, seed + 5)
    holes = np.clip((pot - np.quantile(pot, 0.993)) / 0.02, 0, 1) * (1 - edge)[None, :]
    rgb = rgb * (1 - 0.18 * holes[..., None])

    if lanes == 1:
        # Ragged edges (a few centimetres either way) and gaps every metre or two: grass, not a painted line.
        ragged = (tileable_noise(size, 1.3, seed + 6) - 0.5) * 0.14
        patchy = np.clip((tileable_noise(size, 1.1, seed + 8) - 0.32) * 3, 0, 1)
        xr = x[None, :] + ragged
        strip = np.maximum(_band(xr, 0.0, GRASS_STRIP_HALF_M, 0.06),
                           np.maximum(_band(xr, -width / 2, 0.12, 0.06), _band(xr, width / 2, 0.12, 0.06)))
        cover = (0.9 * strip * patchy)[..., None]
        rgb = rgb * (1 - cover) + _grass_rgb(size, seed + 7) * cover
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def gravel_road_narrow() -> Image.Image:
    """`gravel_road` for a single-lane road: one pair of wheel paths, and the grass strip."""
    return gravel_road(lanes=1)


def mean_colour(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0)


def hex_colour(rgb, scale: float = 1.0) -> str:
    r, g, b = (int(round(min(max(c * scale, 0.0), 255.0))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


# Dark shade of each pair, as a share of the texture's mean. CSP's generic pairs run from 0.41 (soil) to 0.7 (grass).
DIRT_SHADE = 0.55


def tyre_dirt_colours() -> dict[str, str]:
    """`[TYRES_FX]` colour pairs for CSP from our own textures, so dust and dirty tyres match the ground they came
    from: dirt from the gravel road, grass from the grass, soil from the terrain's bare earth (its vegetation
    left out). A is the mean colour, B a darker shade of it, as in CSP's generic pairs
    (`config/tracks/generic/generic.ini`)."""
    soil = _ground_layers(512, 3)[0]
    means = (("DIRT", mean_colour(gravel_road())), ("GRASS", mean_colour(grass())),
             ("SOIL", soil.reshape(-1, 3).mean(axis=0)))
    out = {}
    for key, mean in means:
        out[f"{key}_COLOR_A"] = hex_colour(mean)
        out[f"{key}_COLOR_B"] = hex_colour(mean, DIRT_SHADE)
    return out


def steel(size: int = 256, seed: int = 21) -> Image.Image:
    """Weathered galvanised steel for guard rails."""
    v = 0.6 * tileable_noise(size, 1.2, seed) + 0.4 * tileable_noise(size, 0.3, seed + 1)
    return Image.fromarray(_mix(np.array([168.0, 172.0, 174.0]), v, np.array([0.25, 0.25, 0.25])))


def water(size: int = 512, seed: int = 31) -> Image.Image:
    """Deep blue-green with soft ripples: what vanilla AC shows (CSP draws its own water shader instead)."""
    ripple = tileable_noise(size, beta=2.4, seed=seed)
    base = np.array([38, 74, 84], dtype=float)
    light = np.array([70, 112, 120], dtype=float)
    rgb = base + (light - base) * np.clip(ripple * 0.5 + 0.5, 0, 1)[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def kerb(size: int = 256, seed: int = 61) -> Image.Image:
    """Weathered concrete for a roundabout island's kerb (`core.roundabouts`): a mid grey, a little warm, with fine
    grain and darker stains. Not near white, which would clip in the headlights at night."""
    grain = 0.55 * tileable_noise(size, 0.4, seed) + 0.45 * tileable_noise(size, 1.6, seed + 1)
    return Image.fromarray(_mix(np.array([150.0, 148.0, 142.0]), grain, np.array([0.18, 0.18, 0.18])))


LIBRARY = {"asphalt": asphalt, "road_marked": road_marked, "road_plain": road_plain, "gravel_road": gravel_road,
           "gravel_road_narrow": gravel_road_narrow, "grass": grass, "ground": ground, "steel": steel, "water": water,
           "kerb": kerb,
           **{name: partial(ground_cover, name) for name in GROUND_COVERS}}
