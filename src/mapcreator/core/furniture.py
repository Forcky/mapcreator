"""Roadside furniture: guide posts, curve alignment markers and speed limit signs. Visual only.

Placed by rule, as Australian road authorities place them (`AS 1742.2`, through the public guides that
cite it), the way the road markings are:
- **Guide posts** on both sides: about 1 m tall, a red reflector facing the traffic they're on the left
  of and a white one facing the traffic they're on the right of. Spacing by curve radius, closer on the
  outside of a curve (`GUIDE_POST_SPACING`, Main Roads WA's "Design of Guide Posts", citing AS 1742.2-2009
  clause 4.2.4.4). None where a guard rail runs, and none on unsealed road.
- **Curve alignment markers** (CAMs, black chevrons on yellow) on the outside of a curve whose advisory
  speed is at least 15 km/h below the speed limit, at least three per curve so three are in view, boards
  back to back for both directions. Their spacing here is an approximation: the AS 1742.2 table isn't
  openly published. None on unsealed road: a bend partly on gravel counts only by its sealed part.
- **Speed limit signs** where OpenStreetMap's `maxspeed` changes along the route, and once near each
  layout's start, on the left for each direction of travel.

The textures are drawn here into one atlas: the numerals use Pillow's built-in Aileron font (CC0).

The faces that are retroreflective on the real thing have materials of their own on the same atlas, with the same
values by day: `REFLECTOR_MATERIAL` for the guide post reflectors and `RETRO_MATERIAL` for CAM boards and speed sign
faces. CSP brightens them at night (`NIGHT_DIFFUSE`, through `package.ext_config_ini`), so they catch the headlights the
way real reflectors do; posts, poles and the backs of speed signs stay on `MATERIAL`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import uniform_filter1d

from mapcreator.core.scene import MAX_VERTICES, Z_SIGN, LocalFrame, Mesh

# (radius up to, metres; spacing on the outside of the curve; on the inside). Straights: 150 m.
GUIDE_POST_SPACING = ((100, 6, 12), (200, 10, 20), (300, 15, 30), (400, 20, 40), (600, 40, 40), (800, 50, 50),
                      (1200, 60, 60), (2000, 90, 90), (np.inf, 150, 150))
GUIDE_POST_OFFSET_M = 0.8  # beyond the road's edge, on the verge
GUIDE_POST_HEIGHT_M = 1.0
GUIDE_POST_W, GUIDE_POST_D = 0.10, 0.05
VERGE_FALL = 0.06  # the verge falls this much per metre from the road's edge (as `roadside` assumes)
CURVATURE_SMOOTH_M = 20.0

CAM_SPEED_DROP_KMH = 15.0  # a curve gets CAMs when its advisory speed is at least this far below the limit
CAM_SIDE_FRICTION = 0.35  # side friction for the advisory speed, v = sqrt(127 R (e + f)) in km/h
CAM_MIN = 3
CAM_MAX = 6  # per bend: spread evenly over the bend and its approach when the spacing would give more
CAM_MAX_ADVISORY_KMH = 60.0  # and only bends whose advisory speed is at most this get them
CAM_SPACING_FACTOR = 2.0  # CAMs at twice the outside guide post spacing for the curve's radius, 12-40 m apart
CAM_OFFSET_M = 2.0
CAM_BOARD = (0.45, 0.6)  # width, height
CAM_BOTTOM_M = 1.2
CAM_MIN_CURVE_M = 15.0  # shorter tight spots (a kink in the centreline) get none
DEFAULT_LIMIT_KMH = 80.0  # where OSM gives no limit: the speed CAMs are judged against. Tasmania's rural default is
# 100 km/h, but winding back roads are mostly posted lower; at 100 a 20 km coast road got 723 CAMs (4 Nov).

SIGN_BOARD = (0.45, 0.6)
SIGN_BOTTOM_M = 1.5
SIGN_OFFSET_M = 2.0
SIGN_FROM_START_M = 70.0  # the repeater near each layout's start: past the start gate (50 m)
POST_R = 0.04
# A guide post's reflector, on each of its two faces (a separate quad for the reflective material): its width, its
# height and how far its top is below the post's top. The black band behind it runs 0.1-0.41 m below the top.
REFLECTOR_W, REFLECTOR_H, REFLECTOR_TOP_M = 0.04, 0.18, 0.16
REFLECTOR_GAP_M = 0.002  # in front of the face, so it never fights it for depth

MATERIAL = "furniture"
REFLECTOR_MATERIAL = "furniture_reflector"
RETRO_MATERIAL = "furniture_retro"
# ksDiffuse at night (by day both keep the KN5's value, 0.6), tuned in game at 22:00 with CSP 0.2.11 (5 Nov 2026):
# - a guide post reflector is small and one solid colour: at 12 it stands out in the beam at 25-80 m with a slight
#   halo, as a real one does; at 24 it blooms like a lamp; at 6 or less it reads as paint;
# - a CAM or sign face is a large board with a legend, and in the beam it already reaches the top of CSP's tone curve
#   unboosted (a CAM at 45 m renders (217, 217, 164) at 0.6): more only bleaches it, the yellow to white (nearly
#   white from 2.5) and the sign's ring and numerals away (from about 6; at 12 a sign 90 m away reads as a light).
#   2 keeps them coloured and readable and lifts them where the beam is weaker, further off.
NIGHT_DIFFUSE = {REFLECTOR_MATERIAL: 12.0, RETRO_MATERIAL: 2.0}
CELL = 128
COLS = 8


@dataclass
class Atlas:
    image: Image.Image
    cells: dict  # name -> (u0, v0, u1, v1)


def atlas(limits: list[int]) -> Atlas:
    """Every face the furniture needs, in one texture: guide post faces, CAM chevrons, speed signs."""
    rows = 1 + (len(limits) + COLS - 1) // COLS
    img = Image.new("RGB", (COLS * CELL, rows * CELL), (240, 240, 236))
    d = ImageDraw.Draw(img)
    cells = {}

    def cell(name, col, row, w=1.0, h=1.0):
        x0, y0 = col * CELL, row * CELL
        x1, y1 = x0 + int(CELL * w), y0 + int(CELL * h)
        cells[name] = (x0 / img.width, y0 / img.height, x1 / img.width, y1 / img.height)
        return x0, y0, x1, y1

    x0, y0, x1, y1 = cell("post", 0, 0)
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(244, 244, 240))
    d.rectangle([x0, y0 + 10, x1 - 1, y0 + 40], fill=(20, 20, 20))  # the black band near the top
    for name, col, colour in (("reflector_red", 1, (200, 24, 24)), ("reflector_white", 6, (250, 250, 250))):
        x0, y0, x1, y1 = cell(name, col, 0)
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=colour)
    x0, y0, x1, y1 = cell("white", 2, 0)
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(244, 244, 240))
    x0, y0, x1, y1 = cell("metal", 3, 0)
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(150, 152, 150))
    for col, (name, sign) in enumerate((("cam_left", -1), ("cam_right", 1)), start=4):
        x0, y0, x1, y1 = cell(name, col, 0, 0.75, 1.0)
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(250, 210, 0))
        w, h, cx = x1 - x0, y1 - y0, (x0 + x1) / 2
        tip, back = cx + sign * 0.3 * w, cx - sign * 0.25 * w
        thick = 0.22 * w
        d.polygon([(back, y0 + 0.1 * h), (tip, y0 + 0.5 * h), (back, y0 + 0.9 * h), (back - sign * thick, y0 + 0.9 * h),
                   (tip - sign * thick, y0 + 0.5 * h), (back - sign * thick, y0 + 0.1 * h)], fill=(15, 15, 15))
    font = ImageFont.load_default(size=int(CELL * 0.36))
    for i, limit in enumerate(limits):
        x0, y0, x1, y1 = cell(f"speed_{limit}", i % COLS, 1 + i // COLS, 0.75, 1.0)
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(250, 250, 250), outline=(20, 20, 20), width=3)
        w = x1 - x0
        cx, cy, r = (x0 + x1) / 2, y0 + (y1 - y0) / 2, w * 0.44
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(210, 20, 30), width=int(w * 0.09))
        d.text((cx, cy), str(limit), fill=(15, 15, 15), font=font, anchor="mm")
    return Atlas(img, cells)


def speed_limits(run: np.ndarray, runs: list[dict]) -> np.ndarray:
    """The posted limit (km/h) at each station from OSM's `maxspeed` on its way, NaN where it's missing."""
    per_run = []
    for r in runs:
        parts = str((r.get("tags") or {}).get("maxspeed") or "").split()
        # km/h is OSM's default unit; anything else (mph, "AU:rural", "signals") isn't a number for our signs.
        if len(parts) in (1, 2) and parts[0].isdigit() and parts[1:] in ([], ["km/h"]):
            per_run.append(float(parts[0]))
        else:
            per_run.append(np.nan)
    return np.asarray(per_run, dtype=float)[np.asarray(run, dtype=int)] if per_run else np.full(len(run), np.nan)


def _spacing(radius: float, outside: bool) -> float:
    for limit, out, inner in GUIDE_POST_SPACING:
        if radius < limit:
            return float(out if outside else inner)
    return 150.0


def furniture(s, xy, z, heading, curvature, half_l, half_r, slope_l, slope_r, superelevation, limits,
              frame: LocalFrame, rail_l=None, rail_r=None, unsealed=None) -> tuple[list[Mesh], dict, Atlas]:
    """Guide posts, CAMs and speed signs along the road: meshes (in AC axes), a summary, and the texture atlas
    their material uses.

    `limits` is the posted speed at each station (NaN where unknown), `rail_l`/`rail_r` where a guard rail runs
    on that side (no guide posts there), `unsealed` where the road is gravel (no guide posts or CAMs there; speed
    signs as mapped). Left is left of the direction the stations run (the forward layout).
    """
    n = len(s)
    step = float(s[1] - s[0]) if n > 1 else 1.0
    k = uniform_filter1d(np.asarray(curvature, dtype=float), max(int(CURVATURE_SMOOTH_M / step), 1), mode="nearest")
    radius = 1.0 / np.maximum(np.abs(k), 1e-6)
    tangent = np.column_stack([np.cos(heading), np.sin(heading)])
    left = np.column_stack([-np.sin(heading), np.cos(heading)])
    rails = {"L": np.zeros(n, bool) if rail_l is None else np.asarray(rail_l, bool),
             "R": np.zeros(n, bool) if rail_r is None else np.asarray(rail_r, bool)}
    gravel = np.zeros(n, bool) if unsealed is None else np.asarray(unsealed, bool)
    half = {"L": np.asarray(half_l, float), "R": np.asarray(half_r, float)}
    slope = {"L": np.asarray(slope_l, float), "R": np.asarray(slope_r, float)}
    sign = {"L": 1.0, "R": -1.0}
    known = sorted({int(v) for v in np.asarray(limits)[np.isfinite(limits)]})
    tex = atlas(known)
    builder = _Builder(tex)

    def point(i, side, beyond):
        """Region (e, n) and verge height `beyond` metres past the edge on `side`, at station i."""
        off = half[side][i] + beyond
        e, nn = xy[i] + left[i] * (sign[side] * off)
        return e, nn, z[i] - slope[side][i] * half[side][i] - beyond * VERGE_FALL

    # CAMs first: the outside of a qualifying curve has them instead of guide posts.
    limit = np.where(np.isfinite(limits), limits, DEFAULT_LIMIT_KMH)
    advisory = np.sqrt(127.0 * radius * (np.abs(superelevation) + CAM_SIDE_FRICTION))
    tight = (advisory <= limit - CAM_SPEED_DROP_KMH) & (advisory <= CAM_MAX_ADVISORY_KMH)
    cam_zone = {"L": np.zeros(n, bool), "R": np.zeros(n, bool)}
    cams = curves = 0
    # None on gravel: a bend partly on it counts only by its sealed part, which must qualify on its own (long
    # enough), and a marker whose place falls on the gravel isn't put up.
    tight &= ~gravel
    # Each bend on its own: an S-bend's two halves have their outsides on opposite sides.
    bends = [(i0, i1, True) for i0, i1 in _runs(tight & (k > 0))] + [(i0, i1, False) for i0, i1 in _runs(tight & (k < 0))]
    for i0, i1, turn_left in sorted(bends):
        if s[i1] - s[i0] < CAM_MIN_CURVE_M:
            continue
        side = "R" if turn_left else "L"  # turning left, the outside is on the right
        r_min = float(radius[i0:i1 + 1].min())
        gap = min(max(CAM_SPACING_FACTOR * _spacing(r_min, True), 12.0), 40.0)
        lead = gap  # one marker before and after the curve itself, so three are in view on the way in
        stations = np.arange(s[i0] - lead, s[i1] + lead + 0.1, gap)
        if len(stations) < CAM_MIN:
            stations = np.linspace(s[i0] - lead, s[i1] + lead, CAM_MIN)
        elif len(stations) > CAM_MAX:
            stations = np.linspace(s[i0] - lead, s[i1] + lead, CAM_MAX)
        idx = np.clip(np.searchsorted(s, stations), 0, n - 1)
        cam_zone[side][max(idx[0] - 1, 0):idx[-1] + 2] = True
        for i in np.unique(idx[~gravel[idx]]):
            e, nn, h = point(i, side, CAM_OFFSET_M)
            builder.sign(e, nn, h, tangent[i], CAM_BOARD, CAM_BOTTOM_M,
                         front="cam_left" if turn_left else "cam_right",  # seen by forward traffic
                         back="cam_right" if turn_left else "cam_left")  # the same curve turns the other way
            cams += 1
        curves += 1

    posts = {"L": 0, "R": 0}
    for side in ("L", "R"):
        # The spacing is checked at every station, so the posts close up as soon as a curve begins.
        last = s[0] + 5.0 - np.inf
        for i in range(n):
            outside = (k[i] > 0) == (side == "R")
            if s[i] < s[0] + 5.0 or s[i] - last < _spacing(float(radius[i]), outside):
                continue
            last = s[i]
            if rails[side][i] or cam_zone[side][i] or gravel[i]:
                continue
            e, nn, h = point(i, side, GUIDE_POST_OFFSET_M)
            # The forward traffic has a left-side post on its left (red) and a right-side one on its right (white).
            builder.guide_post(e, nn, h, tangent[i], front="reflector_red" if side == "L" else "reflector_white",
                               back="reflector_white" if side == "L" else "reflector_red")
            posts[side] += 1

    signs = 0
    known_at = np.isfinite(limits)
    changes = [i for i in range(1, n) if known_at[i] and known_at[i - 1] and limits[i] != limits[i - 1]]
    placements = [(i, "L", float(limits[i]), 1.0) for i in changes]  # forward traffic, entering the new zone
    placements += [(i - 1, "R", float(limits[i - 1]), -1.0) for i in changes]  # reverse traffic, the other way
    first = int(np.searchsorted(s, s[0] + SIGN_FROM_START_M))
    last = int(np.searchsorted(s, s[-1] - SIGN_FROM_START_M))
    if 0 < first < n and known_at[first]:
        placements.append((first, "L", float(limits[first]), 1.0))
    if 0 < last < n and known_at[last]:
        placements.append((last, "R", float(limits[last]), -1.0))
    for i, side, value, facing in placements:
        e, nn, h = point(i, side, SIGN_OFFSET_M)
        builder.sign(e, nn, h, tangent[i] * facing, SIGN_BOARD, SIGN_BOTTOM_M, front=f"speed_{int(value)}",
                     back="metal", back_reflects=False)
        signs += 1
    meshes = builder.meshes(frame)
    return meshes, {"guide_posts": posts, "cam_curves": curves, "cams": cams, "speed_signs": signs,
                    "speed_limits": known}, tex


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2] - 1))


class _Builder:
    """Boxes and boards in region coordinates (e, n, h), packed into meshes under the vertex limit, one set of
    meshes per material."""

    def __init__(self, tex: Atlas):
        self.tex = tex
        # positions, normals, uvs, triangles
        self.parts = {m: ([], [], [], []) for m in (MATERIAL, REFLECTOR_MATERIAL, RETRO_MATERIAL)}

    def _quad(self, corners, normal, cell, material=MATERIAL):
        u0, v0, u1, v1 = self.tex.cells[cell]
        pos, nrm, uv, tri = self.parts[material]
        i = len(pos)
        pos.extend(corners)
        nrm.extend([normal] * 4)
        uv.extend([(u0, v1), (u1, v1), (u1, v0), (u0, v0)])  # bottom left, bottom right, top right, top left
        tri.extend([(i, i + 1, i + 2), (i, i + 2, i + 3)])

    def _box(self, e, n, h0, h1, along, w, d, faces, materials=(MATERIAL, MATERIAL)):
        """A box standing on (e, n) from h0 to h1, `w` across the road and `d` along it. `faces` are atlas cells:
        (the side seen by traffic travelling along `along`, the side seen by traffic travelling against it, the
        other two sides); `materials` are the first two faces' materials (the other two sides are `MATERIAL`)."""
        a = np.array([along[0], along[1], 0.0])
        x = np.array([-along[1], along[0], 0.0])  # across
        c = np.array([e, n, 0.0])
        lo, hi = np.array([0, 0, h0]), np.array([0, 0, h1])
        hw, hd = x * w / 2, a * d / 2
        seen_along, seen_against, side_cell = faces
        # Traffic travelling along `along` sees the face that points back at it (-along), and vice versa.
        self._quad([c - hd + hw + lo, c - hd - hw + lo, c - hd - hw + hi, c - hd + hw + hi], -a, seen_along,
                   materials[0])
        self._quad([c + hd - hw + lo, c + hd + hw + lo, c + hd + hw + hi, c + hd - hw + hi], a, seen_against,
                   materials[1])
        self._quad([c - hd - hw + lo, c + hd - hw + lo, c + hd - hw + hi, c - hd - hw + hi], -x, side_cell)
        self._quad([c + hd + hw + lo, c - hd + hw + lo, c - hd + hw + hi, c + hd + hw + hi], x, side_cell)

    def guide_post(self, e, n, h, along, front, back):
        """A post with a reflector on each face: `front` (an atlas cell) seen by the traffic coming along `along`,
        `back` by the other way."""
        self._box(e, n, h - 0.3, h + GUIDE_POST_HEIGHT_M, along, GUIDE_POST_W, GUIDE_POST_D, ("post", "post", "white"))
        a = np.array([along[0], along[1], 0.0])
        x = np.array([-along[1], along[0], 0.0]) * REFLECTOR_W / 2
        top = h + GUIDE_POST_HEIGHT_M - REFLECTOR_TOP_M
        lo, hi = np.array([0, 0, top - REFLECTOR_H]), np.array([0, 0, top])
        for facing, cell in ((-1.0, front), (1.0, back)):
            c = np.array([e, n, 0.0]) + a * facing * (GUIDE_POST_D / 2 + REFLECTOR_GAP_M)
            sx = x * -facing  # bottom left first, seen from the side the face points to (as _box's faces)
            self._quad([c + sx + lo, c - sx + lo, c - sx + hi, c + sx + hi], a * facing, cell, REFLECTOR_MATERIAL)

    def sign(self, e, n, h, along, board, bottom, front, back, back_reflects=True):
        """A board on a post, its face towards the traffic coming along `along`, and the other face behind it.
        The face is retroreflective, and so is the back when it's a board too (a CAM's)."""
        w, hgt = board
        self._box(e, n, h - 0.4, h + bottom + hgt * 0.5, along, POST_R * 2, POST_R * 2, ("metal", "metal", "metal"))
        self._box(e - along[0] * 0.05, n - along[1] * 0.05, h + bottom, h + bottom + hgt, along, w, 0.02,
                  (front, back, "metal"), (RETRO_MATERIAL, RETRO_MATERIAL if back_reflects else MATERIAL))

    def meshes(self, frame: LocalFrame) -> list[Mesh]:
        out = []
        for material, (pos, nrm, uv, tri) in self.parts.items():
            if not pos:
                continue
            pos = np.array(pos, dtype=float)
            nrm = np.array(nrm, dtype=float)
            uv = np.array(uv, dtype=float)
            tri = np.array(tri, dtype=np.int64)
            ac = frame.to_ac(pos[:, 0], pos[:, 1], pos[:, 2])
            ac_n = np.column_stack([nrm[:, 0], nrm[:, 2], Z_SIGN * nrm[:, 1]])
            p0, p1, p2 = ac[tri[:, 0]], ac[tri[:, 1]], ac[tri[:, 2]]
            outward = (np.cross(p1 - p0, p2 - p0) * ac_n[tri].mean(axis=1)).sum(axis=1) >= 0
            tri = np.where(outward[:, None], tri, tri[:, [0, 2, 1]])
            per = (MAX_VERTICES // 4) * 4  # quads never straddle two meshes
            for start in range(0, len(ac), per):
                sel = slice(start, start + per)
                t = tri[(tri[:, 0] >= start) & (tri[:, 0] < start + per)] - start
                out.append(Mesh(f"FURNITURE_{len(out)}", material, ac[sel].astype(np.float32),
                                ac_n[sel].astype(np.float32), uv[sel].astype(np.float32), t))
        return out
