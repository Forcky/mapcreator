"""Closure props: what closes a side road off the stage, Targa and Motorsport Australia style. Standalone builders.

`closure` builds one closure's props, their colliders and a slot for a vehicle in a local frame, and `place` puts
them on a track. Nothing calls them yet: placement comes with the side-road stubs.

**Levels**, by the maintainer's answers to the intersections research of 10 Nov 2026 (its Q6 and Q8-Q12), after
Motorsport Australia's 2026 Special Stage Rally standing regulations, art. 11.10 "Road Closure Standards" (MA levels
A, B and C):
- **major** (a trunk to tertiary side road): course tape at the mouth; barrier boards on trestles across the full
  width with a ROAD CLOSED sign in the middle, and a slot for a vehicle behind them; 25 m further in, a second tape
  and another ROAD CLOSED sign.
- **minor** (unclassified, residential, living street, `road`): the tape; one obstacle across the road, barrier
  boards or big square straw bales; the second tape and sign.
- **limited** (permit, forestry, destination access): the tape and a sign.
- No arrows, chevrons or DETOUR signs (Q12).

**Layout along the stub** (`layout`), in metres from the mainline's edge line (Q8): the tape at 4 m, the obstacle's
line at 12 m, the second tape 25 m beyond the obstacle (37 m) with the sign 0.6 m behind it. MA asks for "a minimum
of 25 m" between the obstacle and the tape and sign. A stub too short for all of that (under 45 m, Q8) carries the
sign on the obstacle; shorter still (under 15 m) it gets the tape and a sign only. On an **overshoot** stub, the
side road straight ahead where the route turns, no obstacle or sign stands in the first 25 m (Q9): the obstacle
moves to 27 m and the second tape to 52 m, and a stub under 30 m gets the tape alone.

**Collision** (Q9: no invisible walls in open air). Straw bales are solid; barrier boards, their trestles and the
signs are passable, as the tape is (the maintainer's decision of 10 Nov 2026, after the first in-game test: at 60 km/h
boards on trestles stopped a car like concrete, where real ones fold over). So a major closure's boards don't stop a
car, and a minor closure built with bales does. Each bale carries its own box collider, exactly its visible extent,
and the row of them spans exactly the given width with BALE_GAP_M between bales, each flush with its own face. The
vehicle (its own module) is solid too. Which props are solid is one setting, `SOLID` (or `closure`'s `solid`): a
board's or sign's collider would run from its trestles' front feet to their back feet.

**The local frame:** the origin is on the stub's centreline at the mainline's edge line; +z runs into the stub, away
from the mainline; +x is to the right looking into the stub; +y is up; metres. It's left-handed, so it maps onto
AC's axes by a mirror: `place` transforms positions and normals and rewinds the triangles. Triangles wind
counter-clockwise seen from the side their normal points to, as the KN5's do; in the frame's own numbers that makes
the cross product of the edges point against the normal.

**Textures:** all drawn here, into one atlas (`atlas`, 1024 x 512): the tape's red diagonal stripes on white
(MA's Tape A, 75 mm, Q10) with an A4 warning tag, the boards' yellow and black diagonal stripes (mirrored left and
right, on both faces), the sign's black ROAD CLOSED on white (AS 1742.3's T2-4, 1800 x 300 mm, double-sided, Q11)
in Pillow's built-in Aileron font (CC0) as the furniture's speed signs are, and the straw, a bale to a face. The tape
and stripes are bands the atlas's full width, so their texture coordinates run on past 1 and wrap.

**Night** (`NIGHT_DIFFUSE`, through `package.ext_config_ini`): the boards' and the sign's faces and the tape are on
`RETRO_MATERIAL`, which CSP brightens at night as it does the furniture's CAM and sign faces, but much less (0.9
against their 2.0). Everything else, on `MATERIAL`, comes down (0.35), as the police car's paint does: CSP's night
exposure is set for the dark road, so a light surface square to the headlights at 10 m clips to white. Their black is
pure black, and **neither material has any specular** (`KN5_PROPERTIES`): AC adds the specular term on top of the
texture, not through it, so with the headlights beside the eye it lit the black legend and stripes across a face
square to them, and lowering ksDiffuse at night only made that floor a larger share of the white.

Sizes the sources don't give are marked as guesses. The research behind them cites AS 1742.3-2019 (through
Queensland's MUTCD Part 3 and QGTTM Part 3, as AS 1742.3 itself is paywalled), the NSW roads authority's QA
specification G10 (yellow trestles) and MA's rules.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mapcreator.core.furniture import Atlas
from mapcreator.core.scene import MAX_VERTICES, Z_SIGN, LocalFrame, Material, Mesh
from mapcreator.core.textures import tileable_noise

LEVELS = ("major", "minor", "limited")
OBSTACLES = ("boards", "bales")  # a minor closure's obstacle; a major one always has boards

# -- layout along the stub (Q8, Q9), metres from the mainline's edge line --------------------------------------
STUB_M = 100.0
TAPE_M = 4.0
OBSTACLE_M = 12.0
WARNING_BEYOND_M = 25.0  # MA SSR 11.10(a): the tape and sign "a minimum of 25m" from the obstacle
OVERSHOOT_CLEAR_M = 25.0  # an overshoot stub: no obstacle or sign nearer than this (Q9)
OVERSHOOT_OBSTACLE_M = 27.0  # so its obstacle's line stands here, its front feet 2.4 m past the clear zone
FULL_LAYOUT_SPARE_M = 8.0  # stub needed beyond the second tape: 37 + 8 = Q8's 45 m
OBSTACLE_SPARE_M = 3.0  # stub needed beyond an obstacle's line (or a lone sign's)
SIGN_BEHIND_TAPE_M = 0.6  # the second sign stands just behind the second tape
MIN_WIDTH_M = 2.0

# -- course tape (MA SSR 11.10(d), Tape A) ----------------------------------------------------------------------
TAPE_W_M = 0.075
TAPE_TIE_M = 1.0  # tied this high on its stakes (guess: MA gives no number, "visible to the crew")
TAPE_SAG = 0.012  # sag as a share of the span, up to TAPE_SAG_MAX_M: so it hangs 0.85-1.0 m up (guess 0.8-1.0)
TAPE_SAG_MAX_M = 0.15
TAPE_SEGMENTS = 8
TAPE_PERIOD_M = 0.15  # one red and one white diagonal stripe along the tape (guess: "square or diagonal")
TAPE_TILE_M = 2.4  # the tape's band in the atlas: 16 periods
STAKE_OUT_M = 0.3  # each stake stands this far beyond the width it closes
STAKE_W = 0.035  # a star picket, drawn as a square post with a yellow safety cap
STAKE_TOP_M = 1.05
STAKE_SINK_M = 0.3
TAG = (0.21, 0.297)  # MA SSR 11.10(e): an A4 warning tag on the tape
TAG_FROM_CENTRE_M = 0.4

# -- barrier boards on trestles (QGTTM Part 3 App. A.2-A.3; NSW G10 4.1) -----------------------------------------
BOARD_MAX_M = 2.5  # "2.5 m long" (secondary source); boards are cut equal to fill a span
BOARD_H = 0.2  # "150 mm to 200 mm" (unverified)
BOARD_T = 0.025
BOARD_CENTRE_M = 1.0  # "about 1 m above the pavement" (unverified)
STRIPE_W_M = 0.1  # each yellow or black stripe, measured along the board, at 45 degrees (guess)
STRIPE_TILE_M = 2.0  # the stripes' band in the atlas: 10 pairs
# Which way the stripes fall: down towards the middle, a V. AS 1742.3's own rule wasn't seen (it's paywalled); the US
# MUTCD (2009, section 6F.68) asks for it on a barricade "where no turns are intended", as at a side road closed off
# the stage, and for stripes falling away from the middle where traffic may turn either way. False gives that.
STRIPES_FALL_TO_MIDDLE = True
MIN_BOARD_M = 0.8  # narrower than this either side of a sign, and the sign stands raised behind the boards instead
TWO_TRESTLES_FROM_M = 1.6  # a shorter board stands on one trestle in its middle ("1 or 2 'A frame' stands")
# Each A-frame: on each side of the board an open frame of two slim yellow legs, splayed a little and braced by a
# crossbar, meeting the other side's under the board (guess). In game the first version's solid 0.45 m panels read as
# blocks. The legs stay at most about 1.4 m apart along a row, so a car (about 1.6 m wide) always meets one.
LEG_R = 0.022  # a leg's tube: a smooth three-sided prism this far from its axis to its edges, about 40 mm across
LEG_SPREAD = (0.48, 0.34)  # between a frame's two legs, at the feet and under the board
CROSSBAR_AT = 0.3  # up the legs, as a share of their length: about 0.3 m up, at bumper height
TRESTLE_FOOT_M = 0.3  # each foot's outside this far in front of or behind the board
TRESTLE_INSET_M = 0.35  # a trestle's middle from its board's end, so its feet don't stick out past it (G10)
LEG_SINK_M = 0.03

# -- ROAD CLOSED sign (AS 1742.3-2019 Table 4.5, T2-4) -------------------------------------------------------------
SIGN = (1.8, 0.3)
SIGN_T = 0.025
SIGN_CENTRE_M = 1.0  # on trestles like a board's, level with the boards (guess: bottom 0.3-1.0 m)
SIGN_RAISED_M = 1.25  # behind bales (0.9 m) or narrow boards, so it shows over them
SIGN_BEHIND_M = 0.2  # gap between the obstacle's back and a raised sign's front feet

# -- straw bales (MA SSR 11.10: "hay/straw bales" are a barricade) -------------------------------------------------
BALE = (2.4, 0.9, 1.2)  # a big square bale's longest length, its height and depth (guess: a common 8 x 3 x 4 ft)
BALE_SINK_M = 0.05
# In game the first row read as one straw block: now the bales stand apart, each a little out of line, and each face
# is drawn as one bale with darker, ragged edges.
BALE_GAP_M = 0.06  # between neighbouring bales, far narrower than any car
BALE_STAGGER_M = 0.05  # each bale up to this far in front of or behind the line
BALE_HEIGHT_VARY_M = 0.03
BALE_LENGTH_VARY = 0.08  # each bale's length varies by up to this share, the row still filling the width
BALE_TEX_M = 2.0  # one bale's face in the atlas: half a straw band, stretched to the bale's length
TWINES_M = (0.12, 0.36, 0.6, 0.84, 1.08)  # across the bale's depth, over its top and down its ends

# -- the vehicle (another module builds it; Q1, Q7) ----------------------------------------------------------------
VEHICLE = (4.9, 1.9)  # the footprint kept for it, length and width
VEHICLE_ANGLE_DEG = 30.0  # parked behind the boards, nose out towards the mainline, turned in towards the middle
VEHICLE_GAP_M = 0.5  # between the boards' back feet and the nearest corner

COLLIDER_BELOW_M = 1.0  # colliders reach this far below the lowest ground under them
# Which props carry colliders (`closure`'s `solid`): by kind, and by the element they stand in. In game (10 Nov 2026)
# a car met them as what it saw at about 20 km/h, but at 60 they were a rigid wall: boards on trestles that would
# fold over stopped it like concrete, and two of four cars glanced along the face and spun off its end. So, by the
# maintainer's decision: the bales alone. Boards, trestles and signs are passable, like the tape.
SOLID_KINDS = ("boards", "bales", "signs")
SOLID_ELEMENTS = ("obstacle", "warning", "sign")
SOLID = ("bales",)

MATERIAL = "closure"
RETRO_MATERIAL = "closure_retro"
TEXTURE = "closure"
# The KN5 values by day for both materials (`track_stage.MATERIAL_PROPS["closure"]`): the walls', but no specular.
# AC's ksPerPixel adds the specular term on top of the texture rather than multiplying it in (as Content Manager's
# re-implementation of Kunos's shaders has it: `txColor * (ambient + diffuse) + specular`), so a pure-black texel still
# gets ksSpecular x (N.H)^ksSpecularEXP x the light. With the headlights beside the eye a face square to them has
# N.H close to 1 across its width (0.99 to 0.93 over the middle 4 m of a sign 13 m off, at the walls' exponent 10).
# The walls' 0.05 then put a floor under the legend and the black stripes of 7-10% of the sign's white at its night
# ksDiffuse of 0.9 (2.6% on the furniture's boards, whose 2.0 dwarfs it): lowering ksDiffuse at night shrank the white
# and left the floor. On Brinktop Road at night (11 Nov 2026), with the beam's hot spot on the boards at 12-15 m and
# CSP's auto-exposure up for the dark around them, that floor reached white: the sign 91% near white with no dark
# letter pixels, though its texture keeps pure-black stroke cores down to mip 3 (the mips sampled out to about 25 m),
# and the stripes' yellow paled in the middle of the beam (white light added to it). Retroreflective sheeting and straw
# have no mirror-like highlight to speak of, and the trestles' tubes are too thin to show one.
KN5_PROPERTIES = {"ksAmbient": 0.4, "ksDiffuse": 0.6, "ksSpecular": 0.0, "ksSpecularEXP": 10.0}
# ksDiffuse at night, by material; by day both keep the KN5's (the track stage's walls' 0.6). In game at 22:00 (CSP,
# 10 Nov 2026), at 8-12 m and square to the beam:
# - with the furniture boards' 2.0 on the retroreflective faces, the sign burnt out (93% near white, its legend
#   unreadable) and the stripes washed pale; at 1.3 the sign was still 79% near white, its letters pale grey;
# - with no adjustment (0.6) the straw bales glowed near white (83%, then 70% darker), and even a plain grey block
#   (106 by day) came up to 141 at night. CSP's night exposure is set for the dark road (30 in the beam), so any light
#   surface facing the headlights at 10 m clips. So the plain material comes down at night, as the police car's body
#   does (0.35), and the retroreflective faces get less than the police car's band (1.3): theirs are big, flat
#   faces square to the beam. From 25 m and more the faces still glinted like the reflectors at 1.3; to retest at 0.9.
NIGHT_DIFFUSE = {MATERIAL: 0.35, RETRO_MATERIAL: 0.9}

# The most each element and each level may take, as (vertices, triangles) of its visual meshes, at any width up to
# BUDGET_WIDTH_M and any stub length; and of its colliders. The research's estimates were about 250 vertices for a
# minor closure and 700 for a major one's props (its vehicle apart). At 8 m (a 6 m road and 1 m verges) a major
# closure takes 536 vertices, a minor one 588 with boards or 324 with bales, and a limited one 168. The trestles'
# tubes have a triangle for each corner, so triangles run closer to vertices than on boxes.
BUDGET_WIDTH_M = 14.0
BUDGET = {"tape": (80, 56), "obstacle_boards": (620, 560), "obstacle_bales": (220, 150), "warning": (170, 140),
          "sign": (96, 86), "major": (860, 750), "minor": (770, 670), "limited": (170, 140)}
COLLIDER_BUDGET = {"major": (0, 0), "minor": (140, 70), "limited": (0, 0)}  # with SOLID: the bales alone

ATLAS_SIZE = (1024, 512)
# Full-width bands: name -> (first row, rows, metres one repeat spans). The tape and stripes repeat along u; each
# straw band holds two bales' faces side by side (the second weathered), BALE_TEX_M each.
BANDS = {"tape": (0, 32, TAPE_TILE_M), "stripes": (32, 64, STRIPE_TILE_M), "straw_side": (96, 128, 2 * BALE_TEX_M),
         "straw_top": (224, 128, 2 * BALE_TEX_M)}
# Cells, as pixel boxes (x0, y0, x1, y1).
CELLS = {"road_closed": (0, 352, 768, 480), "tag_front": (768, 352, 858, 480), "tag_back": (858, 352, 948, 480),
         "picket": (948, 352, 980, 480), "trestle": (0, 480, 32, 512), "white": (32, 480, 64, 512),
         "black": (64, 480, 96, 512)}
LEGEND = "ROAD CLOSED"

WHITE = (244, 244, 240)
# The sign's white, a light grey: at 246 the white round the legend clipped in the beam and bloomed into the letters,
# which then read pale grey (in game, 10 Nov 2026). The police car's white paint came down to 224 for the same reason.
SIGN_WHITE = (200, 200, 196)
# Pure black, for the legend and the stripes: ksDiffuse multiplies the texture, so no night boost can lift it. At 18
# (7%) the boost and the near beam lifted the legend into the top of CSP's tone curve with the white around it.
BLACK = (0, 0, 0)
TAPE_RED = (204, 22, 32)
RETRO_YELLOW = (250, 210, 0)  # as the furniture's CAMs
TRESTLE_YELLOW = (234, 186, 22)
# The straw is about 0.72 of the first version's (mean luminance 118 against 164): in game its bales glowed near white
# in the headlights at 8-12 m (83%), though not retroreflective. They're big flat faces square to the beam where it's
# strongest, and the straw was nearly as light as the sign's white (luminance 208 by day against the sign's 194).
STRAW_LIGHT = (182, 165, 111)
STRAW_DARK = (115, 92, 49)
STRAW_WEATHERED = (0.93, 0.9, 0.86)  # the second bale's tint: a little older and greyer
TWINE = (222, 116, 36)
PICKET = (46, 56, 48)
PICKET_CAP = (236, 196, 30)


# -- the atlas ---------------------------------------------------------------------------------------------------

def _diagonal(rows: int, tile_m: float, height_m: float, stripe_m: float, ss: int = 4) -> np.ndarray:
    """Share of each pixel on the second colour of 45-degree stripes `stripe_m` wide (measured along u), in a band
    `rows` high that spans `tile_m` along u and `height_m` across. Rising to the right in the image: mirror it
    by running u the other way."""
    w = ATLAS_SIZE[0]
    xm = (np.arange(w * ss) + 0.5) / (ss * w) * tile_m
    ym = (np.arange(rows * ss) + 0.5) / (ss * rows) * height_m
    k = (np.floor((xm[None, :] + ym[:, None]) / stripe_m).astype(np.int64) % 2).astype(float)
    return k.reshape(rows, ss, w, ss).mean(axis=(1, 3))


def _blend(a, b, t: np.ndarray) -> np.ndarray:
    return np.asarray(a, float) * (1 - t[..., None]) + np.asarray(b, float) * t[..., None]


def _bale_face(cols: int, rows: int, height_m: float, seed: int, twines: bool, tint) -> np.ndarray:
    """One bale's face, BALE_TEX_M long and `height_m` across: stalks lying mostly across it (in each flake's plane),
    darker seams between flakes, on the top the twines running along it, and darker, ragged edges all round."""
    ss = 2
    noise = tileable_noise((rows, cols), 1.1, seed)
    base = _blend(STRAW_DARK, STRAW_LIGHT, 0.25 + 0.6 * noise)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    img = img.resize((cols * ss, rows * ss), Image.Resampling.BICUBIC)
    d = ImageDraw.Draw(img)
    ppm = cols * ss / BALE_TEX_M
    rng = np.random.default_rng(seed)
    seams = np.cumsum(np.random.default_rng(seed // 2).uniform(0.12, 0.26, 20))  # the same on a bale's side and top
    seam = tuple(int(c * 1.08) for c in STRAW_DARK) if twines else tuple(int(c * 0.85) for c in STRAW_DARK)
    for x in seams[seams < BALE_TEX_M - 0.1] * ppm:  # a flake's edge: a darker, slightly ragged line
        xs = x + rng.normal(0, 1.2, rows + 1)
        d.line(list(zip(xs, np.linspace(0, rows * ss, rows + 1))), fill=seam, width=3 if twines else 4)
    for _ in range(1100 if twines else 1300):
        x, y = rng.uniform(0, cols * ss), rng.uniform(0, rows * ss)
        angle = np.pi / 2 + rng.normal(0, 0.45)
        length = rng.uniform(0.06, 0.2) * ppm
        colour = tuple(int(c) for c in _blend(STRAW_DARK, STRAW_LIGHT, np.array(rng.uniform())))
        dx, dy = np.cos(angle) * length / 2, np.sin(angle) * length / 2
        d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=colour, width=2)
    if twines:
        for depth in TWINES_M:
            y = (1 - depth / height_m) * rows * ss
            d.line(list(zip(np.linspace(0, cols * ss, 129), y + rng.normal(0, 0.6, 129))), fill=TWINE, width=2)
    rgb = np.asarray(img.resize((cols, rows), Image.Resampling.LANCZOS), dtype=float)
    # Darker towards every edge, over about 8 cm, raggedly: the rounded, looser straw at a bale's edges.
    u = (np.arange(cols) + 0.5) / cols * BALE_TEX_M
    v = (np.arange(rows) + 0.5) / rows * height_m
    edge = np.minimum(np.minimum(u, BALE_TEX_M - u)[None, :], np.minimum(v, height_m - v)[:, None])
    edge = edge + 0.025 * (tileable_noise((rows, cols), 1.4, seed + 1) - 0.5)
    shade = 1 - 0.45 * np.clip(1 - edge / 0.08, 0, 1) ** 1.5
    return rgb * shade[..., None] * np.asarray(tint, float)


def _straw(rows: int, height_m: float, seed: int, twines: bool) -> Image.Image:
    """A straw band: two bales' faces side by side, the second weathered."""
    half = ATLAS_SIZE[0] // 2
    faces = [_bale_face(half, rows, height_m, seed + 10 * k, twines, tint)
             for k, tint in enumerate(((1.0, 1.0, 1.0), STRAW_WEATHERED))]
    return Image.fromarray(np.clip(np.hstack(faces), 0, 255).astype(np.uint8))


def _fit_font(d: ImageDraw.ImageDraw, text: str, width: float, height: float, stroke: int = 0):
    for size in range(160, 6, -1):
        font = ImageFont.load_default(size=size)
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font, stroke_width=stroke)
        if x1 - x0 <= width and y1 - y0 <= height:
            return font
    return ImageFont.load_default(size=6)


def _sign_face(d: ImageDraw.ImageDraw, box) -> None:
    """T2-4: black legend and border on white (SIGN_WHITE)."""
    x0, y0, x1, y1 = box
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=SIGN_WHITE)
    d.rectangle([x0 + 6, y0 + 6, x1 - 7, y1 - 7], outline=BLACK, width=5)
    font = _fit_font(d, LEGEND, 0.84 * (x1 - x0), 0.5 * (y1 - y0), stroke=2)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), LEGEND, fill=BLACK, font=font, anchor="mm", stroke_width=2,
           stroke_fill=BLACK)


def _tag(d: ImageDraw.ImageDraw, box, lines) -> None:
    """An A4 tag in a plastic sleeve: (text, colour, share of the height) lines, top to bottom."""
    x0, y0, x1, y1 = box
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(236, 239, 242), outline=(170, 176, 182), width=2)
    y = y0 + 8
    for text, colour, share in lines:
        h = share * (y1 - y0)
        d.text(((x0 + x1) / 2, y + h / 2), text, fill=colour, font=_fit_font(d, text, 0.82 * (x1 - x0), h), anchor="mm")
        y += h + 3


@lru_cache(maxsize=1)
def atlas() -> Atlas:
    """Every face the closures need, in one texture (shared by every closure: don't draw on it)."""
    w, h = ATLAS_SIZE
    img = Image.new("RGB", (w, h), (128, 128, 128))
    noise = tileable_noise((h, w), 0.8, 51)
    grain = 1 + 0.03 * (noise - 0.5)

    row, rows, tile = BANDS["tape"]
    tape = _blend(WHITE, TAPE_RED, _diagonal(rows, tile, TAPE_W_M, TAPE_PERIOD_M / 2)) * grain[row:row + rows, :, None]
    img.paste(Image.fromarray(np.clip(tape, 0, 255).astype(np.uint8)), (0, row))
    row, rows, tile = BANDS["stripes"]
    stripes = _blend(RETRO_YELLOW, BLACK, _diagonal(rows, tile, BOARD_H, STRIPE_W_M)) * grain[row:row + rows, :, None]
    img.paste(Image.fromarray(np.clip(stripes, 0, 255).astype(np.uint8)), (0, row))
    row, rows, _ = BANDS["straw_side"]
    img.paste(_straw(rows, BALE[1], 61, twines=False), (0, row))
    row, rows, _ = BANDS["straw_top"]
    img.paste(_straw(rows, BALE[2], 62, twines=True), (0, row))

    d = ImageDraw.Draw(img)
    _sign_face(d, CELLS["road_closed"])
    _tag(d, CELLS["tag_front"], [("DANGER", TAPE_RED, 0.16), ("HIGH SPEED", BLACK, 0.11), ("CAR RALLY", BLACK, 0.11),
                                  ("IN PROGRESS", BLACK, 0.11)])
    _tag(d, CELLS["tag_back"], [("ROAD CLOSED", BLACK, 0.12), ("DATE ______", BLACK, 0.1),
                                 ("TIME ______", BLACK, 0.1), ("PHONE _____", BLACK, 0.1)])
    x0, y0, x1, y1 = CELLS["picket"]
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=PICKET)
    d.rectangle([x0 + 13, y0, x0 + 18, y1 - 1], fill=tuple(c + 26 for c in PICKET))  # the star's ridge
    d.rectangle([x0, y0, x1 - 1, y0 + 5], fill=PICKET_CAP)  # the top 4% of the cell: a 6 cm cap
    for name, colour in (("trestle", TRESTLE_YELLOW), ("white", SIGN_WHITE), ("black", BLACK)):
        x0, y0, x1, y1 = CELLS[name]
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=colour)

    cells = {name: (x0 / w, y0 / h, x1 / w, y1 / h) for name, (x0, y0, x1, y1) in CELLS.items()}
    cells |= {name: (0.0, row / h, 1.0, (row + rows) / h) for name, (row, rows, _) in BANDS.items()}
    return Atlas(img, cells)


def materials() -> dict[str, Material]:
    """The scene materials: both draw on the one atlas, saved under the texture name `TEXTURE`."""
    return {name: Material(name, TEXTURE) for name in (MATERIAL, RETRO_MATERIAL)}


# -- texture coordinates (corners bottom left, bottom right, top right, top left, seen from the face's front) -----

def _cell_uv(name: str) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = CELLS[name]
    w, h = ATLAS_SIZE
    u0, v0, u1, v1 = (x0 + 1) / w, (y0 + 1) / h, (x1 - 1) / w, (y1 - 1) / h  # a pixel in, against mip bleed
    return [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]


def _solid_uv(name: str) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = CELLS[name]
    c = ((x0 + x1) / 2 / ATLAS_SIZE[0], (y0 + y1) / 2 / ATLAS_SIZE[1])
    return [c] * 4


def _band_v(name: str) -> tuple[float, float]:
    row, rows, _ = BANDS[name]
    return (row + 0.5) / ATLAS_SIZE[1], (row + rows - 0.5) / ATLAS_SIZE[1]  # top, bottom


def _band_uv(name: str, u_left: float, u_right: float) -> list[tuple[float, float]]:
    """The band's full height up the face; u from `u_left` at its left end to `u_right` at its right (as seen from
    its front). u runs on past 1 and wraps."""
    top, bottom = _band_v(name)
    return [(u_left, bottom), (u_right, bottom), (u_right, top), (u_left, top)]


# -- geometry ----------------------------------------------------------------------------------------------------

class _Builder:
    """Quads and strips in the local frame, gathered per material."""

    def __init__(self):
        self.parts: dict[str, tuple[list, list, list, list]] = {}  # material -> positions, normals, uvs, triangles

    def add(self, material: str, pos, nrm, uv, tri) -> None:
        p, n, u, t = self.parts.setdefault(material, ([], [], [], []))
        base = len(p)
        p.extend(np.asarray(pos, float))
        n.extend(np.asarray(nrm, float))
        u.extend(np.asarray(uv, float))
        t.extend(np.asarray(tri, np.int64) + base)

    def quad(self, corners, normal, uvs, material: str) -> None:
        """Four corners in order round the face (for its texture: bottom left first, seen from its front)."""
        self.add(material, corners, [normal] * 4, uvs, [(0, 1, 2), (0, 2, 3)])

    def box(self, x0, x1, y0, y1, z0, z1, faces: dict) -> None:
        """An axis-aligned box's faces: name -> (uvs, material), for `front` (facing -z, the mainline), `back`,
        `left` (-x), `right` and `top`."""
        corners = {
            "front": ([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], (0, 0, -1)),
            "back": ([(x1, y0, z1), (x0, y0, z1), (x0, y1, z1), (x1, y1, z1)], (0, 0, 1)),
            "left": ([(x0, y0, z1), (x0, y0, z0), (x0, y1, z0), (x0, y1, z1)], (-1, 0, 0)),
            "right": ([(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)], (1, 0, 0)),
            "top": ([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], (0, 1, 0)),
        }
        for face, (uvs, material) in faces.items():
            c, n = corners[face]
            self.quad(c, n, uvs, material)

    def tube(self, a, b, uvs, material: str, radius: float = LEG_R) -> None:
        """A slim three-sided prism from a to b with shared corners and normals out from its axis, so it shades as a
        round tube. Open ends: they're smaller than a pixel from the car."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = (b - a) / np.linalg.norm(b - a)
        ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])  # a leg's, or a crossbar's
        p = ref - d * (d @ ref)  # square to the tube
        p /= np.linalg.norm(p)
        q = np.cross(d, p)
        # One corner across the board, two either side of it: symmetric front to back, so a front foot and a back
        # foot reach the same distance from the board.
        out = np.array([np.cos(t) * p + np.sin(t) * q for t in np.radians((0.0, 120.0, 240.0))])
        pos = np.vstack([a + radius * out, b + radius * out])
        k = np.arange(3)
        k1 = (k + 1) % 3
        tri = np.concatenate([np.column_stack([k, k1, 3 + k1]), np.column_stack([k, 3 + k1, 3 + k])])
        self.add(material, pos, np.vstack([out, out]), [uvs[0]] * 6, tri)

    def trestle(self, x: float, z: float, feet: tuple[float, float], top: float) -> None:
        """An A-frame at x under a board or sign whose bottom is at `top`: on each side an open frame of two legs
        from the ground (`feet`: front and back) to the board's underside, and a crossbar."""
        uvs = _solid_uv("trestle")
        wf, wt = LEG_SPREAD[0] / 2, LEG_SPREAD[1] / 2
        foot_z = TRESTLE_FOOT_M - LEG_R * math.sin(math.radians(60.0))  # so a foot's outside is at TRESTLE_FOOT_M
        for side, g in zip((-1.0, 1.0), feet):
            legs = []
            for sx in (-1.0, 1.0):
                foot = np.array([x + sx * wf, g - LEG_SINK_M, z + side * foot_z])
                head = np.array([x + sx * wt, top, z + side * LEG_R])
                self.tube(foot, head, uvs, MATERIAL)
                legs.append(foot + CROSSBAR_AT * (head - foot))
            self.tube(legs[0], legs[1], uvs, MATERIAL)

    def meshes(self, prefix: str) -> list[Mesh]:
        out = []
        for material in (MATERIAL, RETRO_MATERIAL):
            if material in self.parts:
                pos, nrm, uv, tri = self.parts[material]
                out.append(_mesh(f"{prefix}_{len(out)}", material, np.array(pos), np.array(nrm), np.array(uv),
                                 np.array(tri, dtype=np.int64)))
        return out


def _mesh(name: str, material: str, pos, nrm, uv, tri, renderable: bool = True) -> Mesh:
    """A local-frame mesh, every triangle wound counter-clockwise seen from its normal's side (the frame is
    left-handed, so the edges' cross product points against the normal)."""
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    flip = (np.cross(p1 - p0, p2 - p0) * nrm[tri].mean(axis=1)).sum(axis=1) > 0
    tri = np.where(flip[:, None], tri[:, [0, 2, 1]], tri)
    return Mesh(name, material, pos.astype(np.float32), nrm.astype(np.float32), uv.astype(np.float32), tri,
                renderable=renderable)


@dataclass(frozen=True)
class Box:
    """A collider's extent in the local frame."""
    x0: float
    x1: float
    z0: float
    z1: float
    y0: float
    y1: float


def _collider(name: str, boxes: list[Box]) -> Mesh:
    """Invisible boxes, four sides and a top each, wound outwards: one physics mesh, solid by its name (WALL)."""
    b = _Builder()
    faces = {f: (_solid_uv("black"), MATERIAL) for f in ("front", "back", "left", "right", "top")}
    for k in boxes:
        b.box(k.x0, k.x1, k.y0, k.y1, k.z0, k.z1, faces)
    pos, nrm, uv, tri = b.parts[MATERIAL]
    return _mesh(name, MATERIAL, np.array(pos), np.array(nrm), np.array(uv), np.array(tri, dtype=np.int64),
                 renderable=False)


Ground = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _flat(x, z):
    return np.zeros(np.broadcast(np.asarray(x), np.asarray(z)).shape)


def _g(ground: Ground, x, z) -> np.ndarray:
    return np.asarray(ground(np.asarray(x, float), np.asarray(z, float)), dtype=float)


# -- the props ---------------------------------------------------------------------------------------------------

def _tape(b: _Builder, z: float, width: float, ground: Ground, tag_x: float) -> None:
    """Tape across `width` between two stakes just beyond it, tied at TAPE_TIE_M and sagging, with an A4 tag.
    Both faces; visual only."""
    xl, xr = -(width / 2 + STAKE_OUT_M), width / 2 + STAKE_OUT_M
    gl, gr = _g(ground, [xl, xr], [z, z])
    for x, g in ((xl, gl), (xr, gr)):
        uvs = _cell_uv("picket")
        h = STAKE_W / 2
        b.box(x - h, x + h, g - STAKE_SINK_M, g + STAKE_TOP_M, z - h, z + h,
              {f: (uvs, MATERIAL) for f in ("front", "back", "left", "right")})
    span = xr - xl
    sag = min(TAPE_SAG * span, TAPE_SAG_MAX_M)
    t = np.linspace(0.0, 1.0, TAPE_SEGMENTS + 1)
    x = xl + t * span
    top = gl + TAPE_TIE_M + (gr - gl) * t - 4 * sag * t * (1 - t)
    zt = z - STAKE_W / 2 - 0.002  # on the mainline's side of the stakes
    _, _, tile = BANDS["tape"]
    v_top, v_bottom = _band_v("tape")
    i = np.arange(TAPE_SEGMENTS)
    n = TAPE_SEGMENTS + 1
    grid = np.concatenate([np.column_stack([i, i + 1, n + i + 1]), np.column_stack([i, n + i + 1, n + i])])
    for facing in (-1.0, 1.0):  # the face towards the mainline, then the one away from it
        pos = np.concatenate([np.column_stack([x, top - TAPE_W_M, np.full(n, zt)]),
                              np.column_stack([x, top, np.full(n, zt)])])
        u = facing * -x / tile  # increasing to the right as seen from each side
        uv = np.concatenate([np.column_stack([u, np.full(n, v_bottom)]), np.column_stack([u, np.full(n, v_top)])])
        b.add(RETRO_MATERIAL, pos, [(0.0, 0.0, facing)] * len(pos), uv, grid)
    # The tag hangs from the tape's lower edge, on the mainline's side.
    hang = float(np.interp(tag_x, x, top)) - TAPE_W_M
    tw, th = TAG
    zg = zt - 0.003
    for facing, cell in ((-1.0, "tag_front"), (1.0, "tag_back")):
        xa, xb = (tag_x - tw / 2, tag_x + tw / 2) if facing < 0 else (tag_x + tw / 2, tag_x - tw / 2)
        b.quad([(xa, hang - th, zg), (xb, hang - th, zg), (xb, hang, zg), (xa, hang, zg)], (0, 0, facing),
               _cell_uv(cell), MATERIAL)


def _trestle_feet(x0: float, x1: float, z: float, ground: Ground) -> tuple[list[float], np.ndarray]:
    """Where the A-frames under a board or sign from x0 to x1 stand (one or two), and the ground at their feet:
    (front, back) for each."""
    xs = [(x0 + x1) / 2] if x1 - x0 < TWO_TRESTLES_FROM_M else [x0 + TRESTLE_INSET_M, x1 - TRESTLE_INSET_M]
    g = _g(ground, np.repeat(xs, 2), np.tile([z - TRESTLE_FOOT_M, z + TRESTLE_FOOT_M], len(xs)))
    return xs, g.reshape(len(xs), 2)


def _panel(b: _Builder, x0: float, x1: float, z: float, size: tuple[float, float], centre_y: float, ground: Ground,
           faces: tuple[list, list], top_cell: str, ends: tuple[bool, bool]) -> Box:
    """A board or sign (`size`: height, thickness) on its trestles: `faces` are the front's and back's texture
    coordinates (retroreflective). `ends` say whether its left and right ends show (a row's outer ends). The legs end
    under the panel. Returns its extent, for its collider."""
    height, thick = size
    xs, feet = _trestle_feet(x0, x1, z, ground)
    corners = _g(ground, [x0, x1, x0, x1], [z - TRESTLE_FOOT_M] * 2 + [z + TRESTLE_FOOT_M] * 2)
    cy = float(feet.mean()) + centre_y
    y0, y1 = cy - height / 2, cy + height / 2
    for x, (gf, gb) in zip(xs, feet):
        b.trestle(x, z, (gf, gb), y0)
    t = thick / 2
    sides = {"front": (faces[0], RETRO_MATERIAL), "back": (faces[1], RETRO_MATERIAL),
             "top": (_solid_uv(top_cell), MATERIAL)}
    if ends[0]:
        sides["left"] = (_solid_uv(top_cell), MATERIAL)
    if ends[1]:
        sides["right"] = (_solid_uv(top_cell), MATERIAL)
    b.box(x0, x1, y0, y1, z - t, z + t, sides)
    lowest = min(float(feet.min()), float(corners.min()))
    return Box(x0, x1, z - TRESTLE_FOOT_M, z + TRESTLE_FOOT_M, lowest - COLLIDER_BELOW_M, y1)


def _board(b: _Builder, x0: float, x1: float, z: float, ground: Ground, ends: tuple[bool, bool]) -> Box:
    """A barrier board: stripes on both faces, falling towards the middle of the road (or away from it, with
    STRIPES_FALL_TO_MIDDLE False). A stripe's u is a function of x alone, so the pattern runs on from board to board
    and mirrors at x = 0; seen from the other side it falls the same way."""
    left = (x0 + x1) / 2 < 0
    k = (-1.0 if left == STRIPES_FALL_TO_MIDDLE else 1.0) / STRIPE_TILE_M
    front = _band_uv("stripes", k * x0, k * x1)
    back = _band_uv("stripes", k * x1, k * x0)
    return _panel(b, x0, x1, z, (BOARD_H, BOARD_T), BOARD_CENTRE_M, ground, (front, back), "trestle", ends)


def _sign(b: _Builder, x: float, z: float, ground: Ground, centre_y: float = SIGN_CENTRE_M) -> Box:
    """The ROAD CLOSED sign on its trestles, the legend on both faces, each reading left to right from its side."""
    w = SIGN[0]
    face = _cell_uv("road_closed")
    return _panel(b, x - w / 2, x + w / 2, z, (SIGN[1], SIGN_T), centre_y, ground, (face, face), "white", (True, True))


def _bale(b: _Builder, x0: float, x1: float, z: float, height: float, ground: Ground, variant: int,
          mirror: bool) -> Box:
    """A big square bale lying across the road, its long sides to the mainline and away, its middle `z` in. Each face
    shows one bale (`variant` 0 or 1, mirrored or not), stretched to its length. The twines run along it, over its
    top and down its ends; the seams between flakes line up on its side and top."""
    depth = BALE[2]
    z0, z1 = z - depth / 2, z + depth / 2
    g = _g(ground, [x0, x1, x0, x1], [z0, z0, z1, z1])
    y0, y1 = float(g.min()) - BALE_SINK_M, float(g.mean()) + height
    half = 0.5
    pad = 1.0 / ATLAS_SIZE[0]
    ua, ub = variant * half + pad, (variant + 1) * half - pad
    if mirror:
        ua, ub = ub, ua
    faces = {"front": (_band_uv("straw_side", ua, ub), MATERIAL), "back": (_band_uv("straw_side", ua, ub), MATERIAL),
             "top": (_band_uv("straw_top", ua, ub), MATERIAL)}
    b.box(x0, x1, y0, y1, z0, z1, faces)
    # The ends: the top band turned on its side, so its twines run down them; from the middle of the bale's face,
    # clear of its darker ends.
    top, bottom = _band_v("straw_top")
    lo = variant * half + 0.3 * half
    hi = lo + (y1 - y0) / BALE_TEX_M * half
    b.quad([(x0, y0, z1), (x0, y0, z0), (x0, y1, z0), (x0, y1, z1)], (-1, 0, 0),
           [(lo, top), (lo, bottom), (hi, bottom), (hi, top)], MATERIAL)
    b.quad([(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)], (1, 0, 0),
           [(lo, bottom), (lo, top), (hi, top), (hi, bottom)], MATERIAL)
    return Box(x0, x1, z0, z1, y0 - COLLIDER_BELOW_M + BALE_SINK_M, y1)


def _split(x0: float, x1: float, most: float) -> list[tuple[float, float]]:
    """Equal pieces no longer than `most` from x0 to x1."""
    n = max(math.ceil((x1 - x0) / most - 1e-9), 1)
    edges = np.linspace(x0, x1, n + 1)
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


def _board_row(b: _Builder, z: float, width: float, ground: Ground,
               sign: bool) -> tuple[list[tuple[str, Box]], bool]:
    """Boards across `width`, mirrored either side of the middle, with the sign in the middle when it fits.
    Returns each prop's kind and extent, and whether the sign still needs a place (too narrow to sit between
    boards)."""
    half = width / 2
    in_row = sign and half - SIGN[0] / 2 >= MIN_BOARD_M
    inner = SIGN[0] / 2 if in_row else 0.0
    pieces = [(-b1, -b0) for b0, b1 in _split(inner, half, BOARD_MAX_M)][::-1] + _split(inner, half, BOARD_MAX_M)
    props = [("boards", _board(b, x0, x1, z, ground, (i == 0, i == len(pieces) - 1)))
             for i, (x0, x1) in enumerate(pieces)]
    if in_row:
        props.append(("signs", _sign(b, 0.0, z, ground)))
    return props, sign and not in_row


def _bale_row(b: _Builder, z: float, width: float, ground: Ground, seed: int) -> list[tuple[str, Box]]:
    """Bales across `width`, BALE_GAP_M apart, the first and last flush with its edges; each a little longer or
    shorter, taller or lower, nearer or further than the line, and showing one of the two bales either way round."""
    n = max(math.ceil((width + BALE_GAP_M) / (BALE[0] + BALE_GAP_M) - 1e-9), 1)
    rng = np.random.default_rng(seed)
    lengths = 1 + rng.uniform(-BALE_LENGTH_VARY, BALE_LENGTH_VARY, n)
    lengths *= (width - (n - 1) * BALE_GAP_M) / lengths.sum()
    starts = -width / 2 + np.concatenate([[0.0], np.cumsum(lengths[:-1] + BALE_GAP_M)])
    out = []
    for i, (x0, length) in enumerate(zip(starts, lengths)):
        x1 = width / 2 if i == n - 1 else x0 + length
        zc = z + rng.uniform(-BALE_STAGGER_M, BALE_STAGGER_M)
        height = BALE[1] + rng.uniform(-BALE_HEIGHT_VARY_M, BALE_HEIGHT_VARY_M)
        out.append(("bales", _bale(b, float(x0), float(x1), zc, height, ground, int(rng.integers(2)),
                                   bool(rng.integers(2)))))
    return out


# -- the kit -----------------------------------------------------------------------------------------------------

@dataclass
class Element:
    """One line of props across the stub, `z` metres in. Visual meshes one per material, colliders as one
    invisible physics mesh (none for the tape, or where none of its props is solid)."""
    name: str  # "tape", "obstacle", "warning" (the second tape and its sign), "sign" (a lone sign)
    z: float
    meshes: list[Mesh]
    colliders: list[Mesh]
    boxes: list[Box] = field(default_factory=list)  # the colliders' extents
    props: list[tuple[str, Box]] = field(default_factory=list)  # every board, bale and sign: its kind and extent


@dataclass(frozen=True)
class Slot:
    """Room kept for a vehicle: its footprint's middle on the ground and the way its nose points, in the local
    frame. The vehicle's own module builds it and its collider."""
    name: str
    z: float
    position: tuple[float, float, float]
    forward: tuple[float, float, float]
    size: tuple[float, float] = VEHICLE  # length, width


@dataclass
class Closure:
    level: str
    overshoot: bool
    length: float
    obstacle: str | None  # "boards" or "bales", None without one
    elements: list[Element]
    slots: list[Slot]

    @property
    def atlas(self) -> Atlas:
        return atlas()

    @property
    def materials(self) -> dict[str, Material]:
        return materials()

    def element(self, name: str) -> Element | None:
        return next((e for e in self.elements if e.name == name), None)

    def meshes(self) -> list[Mesh]:
        return [m for e in self.elements for m in e.meshes]

    def colliders(self) -> list[Mesh]:
        return [m for e in self.elements for m in e.colliders]

    def counts(self) -> dict:
        """Vertices and triangles, visual and collider, per element and in all."""
        def tally(ms):
            return {"vertices": sum(len(m.positions) for m in ms), "triangles": sum(len(m.triangles) for m in ms)}
        per = {e.name: {**tally(e.meshes), "collider_vertices": tally(e.colliders)["vertices"],
                        "collider_triangles": tally(e.colliders)["triangles"]} for e in self.elements}
        all_ = tally(self.meshes())
        col = tally(self.colliders())
        return {**all_, "collider_vertices": col["vertices"], "collider_triangles": col["triangles"], "elements": per}


def layout(level: str, *, length: float = STUB_M, overshoot: bool = False) -> dict[str, float]:
    """Where each element's line crosses the stub, metres in from the mainline's edge line.

    - Every level: `tape` at TAPE_M (4 m).
    - major and minor, on a stub of at least 45 m (60 m overshooting): `obstacle` at OBSTACLE_M (12 m;
      OVERSHOOT_OBSTACLE_M, 27 m, overshooting) and `warning`, the second tape, 25 m beyond it (37 m; 52 m). Shorter,
      down to 15 m (30 m): the obstacle alone, carrying the sign. Shorter still: a lone `sign`, as far in as fits.
    - limited: a lone `sign` at the obstacle's place, or as far in as fits.
    - A lone sign never stands nearer than 6 m, and on an overshoot stub no prop nearer than 26 m: past
      OVERSHOOT_CLEAR_M with room for its feet. Where it can't, the tape is all there is.
    """
    _check(level)
    ob = OVERSHOOT_OBSTACLE_M if overshoot else OBSTACLE_M
    warn = ob + WARNING_BEYOND_M
    nearest = OVERSHOOT_CLEAR_M + 1.0 if overshoot else TAPE_M + 2.0
    furthest = length - OBSTACLE_SPARE_M
    out: dict[str, float] = {}
    if length < TAPE_M + 1.0:
        return out
    out["tape"] = TAPE_M
    if level != "limited" and length >= warn + FULL_LAYOUT_SPARE_M:
        out["obstacle"], out["warning"] = ob, warn
    elif level != "limited" and furthest >= ob:
        out["obstacle"] = ob
    elif min(ob, furthest) >= nearest:
        out["sign"] = min(ob, furthest)
    return out


def _check(level: str) -> None:
    if level not in LEVELS:
        raise ValueError(f"closure level {level!r} is not one of {LEVELS}")


def _solid_rule(solid: Collection[str] | None) -> Callable[[str, str], bool]:
    """Whether a prop of a kind in an element is solid. `solid` names kinds (SOLID_KINDS) and elements
    (SOLID_ELEMENTS); a prop is solid when its kind and its element are both named, and naming none of either
    means all of them. So ("bales",) is the bales alone, ("obstacle",) everything in the obstacle's line,
    ("obstacle", "boards", "bales") its boards and bales but not its sign, and () nothing."""
    names = set(SOLID if solid is None else solid)
    unknown = names - set(SOLID_KINDS) - set(SOLID_ELEMENTS)
    if unknown:
        raise ValueError(f"solid names {sorted(unknown)} aren't among {SOLID_KINDS + SOLID_ELEMENTS}")
    if not names:
        return lambda kind, element: False
    kinds = (names & set(SOLID_KINDS)) or set(SOLID_KINDS)
    elements = (names & set(SOLID_ELEMENTS)) or set(SOLID_ELEMENTS)
    return lambda kind, element: kind in kinds and element in elements


def _element(name: str, z: float, b: _Builder, props: list[tuple[str, Box]], solid) -> Element:
    tag = name.upper()
    boxes = [box for kind, box in props if solid(kind, name)]
    return Element(name, z, b.meshes(f"CLOSURE_{tag}"), [_collider(f"1WALL_CLOSURE_{tag}", boxes)] if boxes else [],
                   boxes, props)


def closure(level: str, width: float | Callable[[float], float], *, overshoot: bool = False, length: float = STUB_M,
            obstacle: str = "boards", ground: Ground | None = None, seed: int = 0,
            solid: Collection[str] | None = None) -> Closure:
    """One closure's props, colliders and vehicle slot, in the local frame (module docstring).

    - `level`: "major", "minor" or "limited" (Q6).
    - `width`: the width to close, centred on the stub's centreline, in metres: a number, or a function of the
      distance in. The boards or bales span exactly it at the obstacle, and each tape spans it with its stakes
      STAKE_OUT_M beyond. Normally the carriageway and its verges.
    - `overshoot`: the stub is the road straight ahead where the route turns, so no obstacle or sign in its first
      25 m (Q9).
    - `length`: how much of the stub the props may use: 100 m (Q8), less where the side road ends or meets the route.
    - `obstacle`: a minor closure's, "boards" or "bales". A major one always has boards; a limited one none.
    - `ground(x, z)`: the stub's surface height relative to its centreline at the same distance in (its crown and
      its verges' fall), so feet stand on it. Flat if left out.
    - `seed`: varies the tag's place and the bales, so neighbouring closures differ.
    - `solid`: which props carry colliders, by kind ("boards", "bales", "signs") and element ("obstacle", "warning",
      "sign"): a prop is solid when both are named, naming none of either meaning all (`_solid_rule`). None takes
      SOLID: the bales alone. The tape never is.

    Layout (`layout`): tape at 4 m; obstacle at 12 m; second tape at 37 m with the sign 0.6 m behind it. A major
    closure's obstacle has a ROAD CLOSED sign in the middle of its boards and a vehicle slot behind them; a minor
    one's carries the sign only where the stub is too short for the second tape. Each element's `props` lists every
    board, bale and sign with its extent, solid or not; `boxes` are the solid ones', its colliders.
    """
    _check(level)
    if obstacle not in OBSTACLES:
        raise ValueError(f"obstacle {obstacle!r} is not one of {OBSTACLES}")
    is_solid = _solid_rule(solid)
    width_at = width if callable(width) else (lambda z, w=float(width): w)
    ground = ground or _flat
    where = layout(level, length=length, overshoot=overshoot)
    rng = np.random.default_rng(seed)
    elements: list[Element] = []
    slots: list[Slot] = []
    kind = None

    def wide(z):
        w = float(width_at(z))
        if w < MIN_WIDTH_M:
            raise ValueError(f"width {w:.2f} m at {z:.1f} m is under {MIN_WIDTH_M} m")
        return w

    def tag_x():
        return float(rng.choice([-1.0, 1.0]) * TAG_FROM_CENTRE_M * rng.uniform(0.5, 1.5))

    if "tape" in where:
        z = where["tape"]
        b = _Builder()
        _tape(b, z, wide(z), ground, tag_x())
        elements.append(_element("tape", z, b, [], is_solid))
    if "obstacle" in where:
        z = where["obstacle"]
        w = wide(z)
        kind = "boards" if level == "major" else obstacle
        sign = level == "major" or "warning" not in where
        b = _Builder()
        if kind == "boards":
            props, raised = _board_row(b, z, w, ground, sign)
        else:
            props, raised = _bale_row(b, z, w, ground, seed), sign
        if raised:  # behind the obstacle, high enough to show over it
            back = max(box.z1 for _, box in props)
            props.append(("signs", _sign(b, 0.0, back + SIGN_BEHIND_M + TRESTLE_FOOT_M, ground, SIGN_RAISED_M)))
        elements.append(_element("obstacle", z, b, props, is_solid))
        if level == "major":
            slot = _vehicle_slot(max(box.z1 for _, box in props), w, ground, length)
            if slot is not None:
                slots.append(slot)
    if "warning" in where:
        z = where["warning"]
        b = _Builder()
        _tape(b, z, wide(z), ground, tag_x())
        props = [("signs", _sign(b, 0.0, z + SIGN_BEHIND_TAPE_M, ground))]
        elements.append(_element("warning", z, b, props, is_solid))
    if "sign" in where:
        z = where["sign"]
        wide(z)
        b = _Builder()
        elements.append(_element("sign", z, b, [("signs", _sign(b, 0.0, z, ground))], is_solid))
    return Closure(level, overshoot, length, kind, elements, slots)


def _vehicle_slot(back: float, width: float, ground: Ground, length: float) -> Slot | None:
    """Behind the boards (`back`: the furthest-back feet, a raised sign's included), to the right of the middle, nose
    out and turned in towards the middle."""
    a = math.radians(VEHICLE_ANGLE_DEG)
    half_l, half_w = VEHICLE[0] / 2, VEHICLE[1] / 2
    reach_z = half_l * math.cos(a) + half_w * math.sin(a)
    reach_x = half_l * math.sin(a) + half_w * math.cos(a)
    zc = back + VEHICLE_GAP_M + reach_z
    if zc + reach_z + OBSTACLE_SPARE_M > length:
        return None
    xc = max(0.0, min(width / 4, width / 2 - reach_x))
    y = float(_g(ground, xc, zc))
    return Slot("vehicle", zc, (xc, y, zc), (-math.sin(a), 0.0, -math.cos(a)))


# -- placing a closure on a track --------------------------------------------------------------------------------

Pose = tuple[float, float, float, float]  # easting, northing, height of the stub's centreline, and its heading


@dataclass
class PlacedSlot:
    name: str
    position: np.ndarray  # AC axes, on the ground
    forward: np.ndarray  # AC axes, unit: where the vehicle's nose points
    e: float
    n: float
    h: float
    heading: float  # radians, east = 0, counter-clockwise: the vehicle's nose
    size: tuple[float, float]


@dataclass
class Placed:
    meshes: list[Mesh]
    colliders: list[Mesh]
    slots: list[PlacedSlot]


def straight(e: float, n: float, h: float, heading: float, grade: float = 0.0) -> Callable[[float], Pose]:
    """The pose of a straight stub from its mouth: `heading` (radians, east = 0, counter-clockwise) into the stub,
    rising `grade` (a fraction) as it goes."""
    c, s = math.cos(heading), math.sin(heading)
    return lambda z: (e + c * z, n + s * z, h + grade * z, heading)


def place(kit: Closure, pose: Pose | Callable[[float], Pose], frame: LocalFrame, index: int = 0) -> Placed:
    """The closure's meshes, colliders and slots in AC axes, ready to append to a scene.

    `pose` is the stub's centreline at the mouth (easting, northing, height, heading into the stub: radians, east =
    0, counter-clockwise), for a straight, level stub; or a function giving it at any distance in, so on a stub that
    bends or climbs each element stands across the stub at its own distance, at the centreline's height there.
    Names carry `index`: `CLOSURE_###_<ELEMENT>_k` and `1WALL_CLOSURE_###_<ELEMENT>`; `merge` packs many closures'
    meshes into fewer.
    """
    at = pose if callable(pose) else straight(*pose)
    meshes, colliders = [], []
    for el in kit.elements:
        p = at(el.z)
        for m in el.meshes:
            meshes.append(_to_ac(m, el.z, p, frame, m.name.replace("CLOSURE_", f"CLOSURE_{index:03d}_", 1)))
        for m in el.colliders:
            colliders.append(_to_ac(m, el.z, p, frame, m.name.replace("CLOSURE_", f"CLOSURE_{index:03d}_", 1)))
    slots = []
    for s in kit.slots:
        e, n, h, hd = at(s.z)
        r, f = np.array([math.sin(hd), -math.cos(hd)]), np.array([math.cos(hd), math.sin(hd)])
        x, y, z = s.position
        en = np.array([e, n]) + r * x + f * (z - s.z)
        fx, _, fz = s.forward
        fen = r * fx + f * fz
        pos = frame.to_ac(en[0], en[1], h + y)
        fwd = np.array([fen[0], 0.0, Z_SIGN * fen[1]])
        slots.append(PlacedSlot(s.name, pos, fwd / np.linalg.norm(fwd), float(en[0]), float(en[1]), float(h + y),
                                float(math.atan2(fen[1], fen[0])), s.size))
    return Placed(meshes, colliders, slots)


def _to_ac(m: Mesh, z_line: float, pose: Pose, frame: LocalFrame, name: str) -> Mesh:
    e, n, h, hd = pose
    r = np.array([math.sin(hd), -math.cos(hd)])  # +x: right of the heading
    f = np.array([math.cos(hd), math.sin(hd)])  # +z
    x, y, z = (m.positions[:, i].astype(float) for i in range(3))
    zz = z - z_line
    pos = frame.to_ac(e + x * r[0] + zz * f[0], n + x * r[1] + zz * f[1], h + y)
    nx, ny, nz = (m.normals[:, i].astype(float) for i in range(3))
    nrm = np.column_stack([nx * r[0] + nz * f[0], ny, Z_SIGN * (nx * r[1] + nz * f[1])])
    tri = m.triangles
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    outward = (np.cross(p1 - p0, p2 - p0) * nrm[tri].mean(axis=1)).sum(axis=1) >= 0
    tri = np.where(outward[:, None], tri, tri[:, [0, 2, 1]])
    return Mesh(name, m.material, pos.astype(np.float32), nrm.astype(np.float32), m.uvs.copy(), tri, m.renderable)


def merge(meshes: list[Mesh], prefix: str = "CLOSURE") -> list[Mesh]:
    """Pack placed meshes into as few as the vertex limit allows, one run per material and visibility, so a track's
    closures cost a few draw calls rather than several each. Visual meshes are named `<prefix>_k`, colliders
    `1WALL_<prefix>_k`."""
    groups: dict[tuple[str, bool], list[Mesh]] = {}
    for m in meshes:
        groups.setdefault((m.material, m.renderable), []).append(m)
    out: list[Mesh] = []
    for (material, renderable), ms in groups.items():
        batch: list[Mesh] = []
        count = 0
        for m in ms + [None]:
            if m is None or count + len(m.positions) > MAX_VERTICES:
                if batch:
                    name = f"{prefix if renderable else '1WALL_' + prefix}_{len(out)}"
                    base = np.cumsum([0] + [len(b.positions) for b in batch[:-1]])
                    out.append(Mesh(name, material, np.concatenate([b.positions for b in batch]),
                                    np.concatenate([b.normals for b in batch]), np.concatenate([b.uvs for b in batch]),
                                    np.concatenate([b.triangles + k for b, k in zip(batch, base)]), renderable))
                batch, count = [], 0
            if m is not None:
                batch.append(m)
                count += len(m.positions)
    return out
