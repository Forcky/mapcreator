"""Side-road stubs: the first 100 m of each real side road (`core.junctions`), drawn as road and joined to the
mainline by a corner flare either side.

**The plan** (`plan`, run by the section stage; `03_section/stubs.json`, its format in docs/ac-format.md). For each
junction:
- **the line:** the side road as junctions.json has it, its start moved onto the smoothed centreline and then smoothed
  as the route is (`centreline.smooth`). Its rows are every metre from where it leaves the mainline's edge (d = 0)
  to its end, so `d` is "metres in from the mainline's edge line";
- **width, surface and look** by the mainline's rules (`core.section`): the tagged or class-default width as mapped,
  times the job's width option; sealed or unsealed by its `surface` tag, or, untagged, as the mainline is at the
  junction; one lane or two by the same rules. Its crown is the mainline's for that surface, with no camber;
- **heights:** beside the mainline the surface carries on the mainline's own cross-fall, and further out it follows
  the side road's own profile (the DEM along it, smoothed over DEM_SMOOTH_M) with its crown. The two blend over
  BLEND_M from the mainline's edge, or longer where the side road sits well above or below it, so it climbs or falls
  no more than MAX_BLEND_GRADE away from the road;
- **two corner flares**, a curve either side from the mainline's edge to the side road's edge, tangent to both: a
  corner of FLARE_RADIUS_M, which makes the flare on the acute side of a skewed junction longer (its tangent length
  is the radius over tan(angle / 2)). The tangent length is capped at FLARE_MAX_M, the radius kept at
  FLARE_MIN_RADIUS_M or more. The road between the flares, the mainline's edge and the stub's first full row is the
  throat;
- **the mouth:** the mainline's rows between the two flares' tangent points.

A stub that would come within a verge of another part of the route is cut short before it; one that would overlap
a stub already kept is cut short, or with smaller flares left out, keeping major roads, then longer ones, first. A
stub whose mouth would leave a timing gate nowhere clear to stand (`layout.gate_distances`) is left out. Each one left
out is listed under `skipped`, with the reason.

**The meshes** (`build`, run by the mesh stage) come from the plan alone, so the mesh and scenery stages, which run
side by side, see the same stubs:
- the stub's own ribbon (`mesh.build_ribbon`) from its first full row on: road, verges and skirts like the
  mainline's, its rows a metre apart for FINE_ROWS_M and COARSE_STEP_M beyond (the cost is in docs/known-issues.md);
- the throat, triangulated over the mainline's edge vertices, the flares and the ribbon's first row, sharing their
  vertices exactly, with the stub's own cross-section inside it;
- a mown verge along each flare, from the mainline's verge to the stub's;
- the mainline's verge and skirt cut across the mouth, its skirt held short of the stub's where they'd meet (and the
  stub's of the mainline's: each stops where the ground is nearer the other's centreline), and its edge lines broken
  across the mouth by texture coordinates (`mesh.ribbon_meshes(edge_break=)`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import uniform_filter1d
from scipy.spatial import Delaunay, cKDTree

from mapcreator.core import centreline as cl
from mapcreator.core import junctions as jn
from mapcreator.core import mesh
from mapcreator.core import section as sec
from mapcreator.core import textures
from mapcreator.core.layout import RoadModel, gate_distances
from mapcreator.core.scene import LocalFrame, Mesh

FORMAT_VERSION = 1

FLARE_RADIUS_M = cl.CORNER_RADIUS_M  # 8 m: "about the path a car takes through a town junction"
FLARE_MAX_M = 25.0  # the longest a flare's tangent runs along the road, on the acute side of a skewed junction
FLARE_MIN_RADIUS_M = 3.5  # and its tightest corner: over the 2.5 m a skirt always reaches past its verge
FLARE_MAX_TANGENT_M = 40.0  # a corner needing more than this is too skewed to flare: no stub
FLARE_SAMPLE_M = 1.0  # flare points no further apart than this, as the ribbons' rows are
START_EASE_M = 20.0  # the side road's start moves onto the centreline, easing back to its own line over this
BACK_M = 40.0  # the side road's first direction carried back across the mainline, to find where its edges leave it
CORNER_WITHIN_M = 40.0  # an edge must leave the mainline's edge within this of where its centreline does
BLEND_M = 15.0
MAX_BLEND_GRADE = 0.12
DEM_SMOOTH_M = 15.0
MIN_RIBBON_M = 10.0  # a stub with less than this of its own ribbon past the throat is left out
CLEAR_M = 0.5  # between a stub's verge and another part of the route's verge, or another stub's
SHRINK = 0.5  # a stub overlapping another tries again with its flares' radii this much smaller (to the minimum)
STEINER_CLEAR_M = 0.6  # the stub's own cross-section points inside the throat stay this far from its edge
FINE_ROWS_M = 20.0  # the stub's ribbon has a row every metre up to this far in (the closures stand inside it)
COARSE_STEP_M = 2.0  # and a row every this many metres beyond
MAIN_REACH_M = 300.0  # along the mainline either side of a junction, the stretch its stub is measured against
# The throat is plain surface, no lines: its texture coordinates stay inside a band of each road texture with no
# markings (sealed) or no wheel path (gravel), mirrored back and forth across it every `band` metres.
THROAT_U = {"road": (0.07, 0.43), "road_plain": (0.09, 0.43), "gravel": (0.19, 0.31), "gravel_narrow": (0.13, 0.23)}
TEXTURE_WIDTH_M = {"road": textures.ROAD_MARKED_WIDTH_M, "road_plain": textures.ROAD_PLAIN_WIDTH_M,
                   "gravel": textures.GRAVEL_WIDTH_M[2], "gravel_narrow": textures.GRAVEL_WIDTH_M[1]}
WINDOW_M = 400.0  # stubs' meshes are packed by material per this much of the mainline, as trunk colliders are


# -- the plan ---------------------------------------------------------------------------------------------------------

class _Mainline:
    """The smoothed centreline and cross-section, and projections onto it near one junction."""

    def __init__(self, c, section):
        self.s = np.asarray(c["s"], float)
        self.xy = np.asarray(c["xy"], float)
        self.heading = np.asarray(c["heading"], float)
        self.z = np.asarray(c["z"], float)
        self.half = {"L": np.asarray(section["half_left"], float), "R": np.asarray(section["half_right"], float)}
        self.slope = {"L": np.asarray(section["slope_left"], float), "R": np.asarray(section["slope_right"], float)}
        self.style = np.asarray(section["style"]) if "style" in section else np.zeros(len(self.s), np.int8)
        self.nrm = cl.normals(self.heading)
        self.tree = cKDTree(self.xy)

    def at(self, s: float) -> np.ndarray:
        return np.array([np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])])

    def to_xy(self, s: np.ndarray, o: np.ndarray) -> np.ndarray:
        """Points at stations `s`, `o` to the left of the centreline (the inverse of `_Near.project`)."""
        h = np.interp(s, self.s, self.heading)
        base = np.column_stack([np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])])
        return base + np.column_stack([-np.sin(h), np.cos(h)]) * np.asarray(o, float)[:, None]

    def row(self, s: float) -> float:
        return (s - self.s[0]) / (self.s[1] - self.s[0])

    def edge(self, r: int, side: str) -> np.ndarray:
        sign = 1.0 if side == "L" else -1.0
        return self.xy[r] + self.nrm[r] * sign * self.half[side][r]

    def edge_h(self, r: int, side: str) -> float:
        return float(self.z[r] - self.slope[side][r] * self.half[side][r])

    def near(self, at: float, reach: float) -> "_Near":
        return _Near(self, np.flatnonzero(np.abs(self.s - at) <= reach))


class _Near:
    """Projections onto the mainline's stretch near a junction (not another part of the route passing by)."""

    def __init__(self, m: _Mainline, idx: np.ndarray):
        self.m, self.idx = m, idx
        self.tree = cKDTree(m.xy[idx])

    def project(self, p) -> tuple[np.ndarray, np.ndarray]:
        """Each point's station and offset (+ to the left) on the mainline."""
        p = np.atleast_2d(np.asarray(p, float))
        _, k = self.tree.query(p)
        i = self.idx[k]
        h = self.m.heading[i]
        d = p - self.m.xy[i]
        return self.m.s[i] + d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h), d[:, 0] * -np.sin(h) + d[:, 1] * np.cos(h)

    def beyond(self, p, side: str) -> np.ndarray:
        """How far each point lies beyond the mainline's edge on `side` (negative: on the road, or across it)."""
        sp, o = self.project(p)
        sign = 1.0 if side == "L" else -1.0
        return sign * o - np.interp(sp, self.m.s, self.m.half[side])

    def surface(self, p, side: str) -> np.ndarray:
        """The mainline's surface carried on beyond its edge on `side`: its outer half's cross-fall."""
        sp, o = self.project(p)
        sign = 1.0 if side == "L" else -1.0
        return np.interp(sp, self.m.s, self.m.z) - np.interp(sp, self.m.s, self.m.slope[side]) * (sign * o)


def _crossing(x: np.ndarray, g: np.ndarray, start: int) -> float | None:
    """The first x from index `start` on where g rises through 0, interpolated."""
    for k in range(max(start, 1), len(g)):
        if g[k] >= 0 > g[k - 1]:
            return float(x[k - 1] + (x[k] - x[k - 1]) * (-g[k - 1]) / (g[k] - g[k - 1]))
    return None


def _interp_rows(dd: np.ndarray, at: np.ndarray, *arrays) -> list[np.ndarray]:
    out = []
    for a in arrays:
        a = np.asarray(a, float)
        out.append(np.column_stack([np.interp(dd, at, a[:, k]) for k in range(a.shape[1])]) if a.ndim == 2
                   else np.interp(dd, at, a))
    return out


def _fillet(t1: np.ndarray, corner: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
    """The circular arc rounding the corner between two lines, from `t1` on the first (the road's edge, in its own
    frame) to the second (the line from `corner` through `end`, the stub's edge), with tangents the same length
    either side of the corner; then straight on along the second line to `end`. Points at most FLARE_SAMPLE_M apart,
    and the arc's radius."""
    tangent = float(np.hypot(*(t1 - corner)))
    back = (t1 - corner) / max(tangent, 1e-9)
    out = (end - corner) / max(float(np.hypot(*(end - corner))), 1e-9)
    alpha = float(np.arccos(np.clip(back @ out, -1.0, 1.0)))
    radius = tangent * math.tan(alpha / 2)
    t2 = corner + out * tangent
    inward = np.array([-back[1], back[0]])
    if inward @ out < 0:
        inward = -inward
    centre = t1 + inward * radius
    a1 = math.atan2(*(t1 - centre)[::-1])
    sweep = (math.atan2(*(t2 - centre)[::-1]) - a1 + math.pi) % (2 * math.pi) - math.pi
    n = max(int(math.ceil(radius * abs(sweep) / FLARE_SAMPLE_M)), 1)
    a = a1 + sweep * np.linspace(0.0, 1.0, n + 1)
    arc = centre + radius * np.column_stack([np.cos(a), np.sin(a)])
    rest = float(np.hypot(*(end - t2)))
    if rest > 1e-3:
        k = max(int(math.ceil(rest / FLARE_SAMPLE_M)), 1)
        arc = np.vstack([arc, t2 + (end - t2) * (np.arange(1, k + 1) / k)[:, None]])
    return arc, radius


def _flare_shape_ok(curve: np.ndarray, corner_deg: float) -> bool:
    """Whether a flare drawn in the road's frame came out as a corner in plan: a simple curve, no point far from the
    last, turning about as much as the corner does (180 degrees less its angle). Far out on the inside of a tight bend
    the road's frame folds over itself, and a long flare drawn there can loop round instead."""
    line = shapely.LineString(curve)
    if not line.is_simple:
        return False
    step = np.hypot(*np.diff(curve, axis=0).T)
    if step.max() > 3 * FLARE_SAMPLE_M:
        return False
    d = np.diff(curve, axis=0)[step > 1e-6]
    heading = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((np.diff(heading) + np.pi) % (2 * np.pi) - np.pi).sum()
    return bool(turn <= math.radians(180.0 - corner_deg + 35.0))


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@dataclass
class _Candidate:
    record: dict
    priority: tuple
    throat: shapely.Geometry  # the throat's road, beyond the mainline's edge
    rows_xy: np.ndarray  # the stub's rows from the ribbon's first on, and their headings
    rows_n: np.ndarray
    half: float
    first: int  # the ribbon's first row, of the plan's rows
    body: shapely.Geometry | None = None  # its ribbon's road and verges, once kept


def _candidate(j: dict, m: _Mainline, dem_sample, width_factor: float, scale: dict[str, float] | None = None,
               ) -> tuple[_Candidate | None, str | None]:
    """A junction's stub, or why there's none. `scale` shrinks each flare's radius ("upstream", "downstream")."""
    scale = scale or {"upstream": 1.0, "downstream": 1.0}
    side = jn.SIDE_CODE[j["side"]]
    at = jn._station(m.xy, m.s, np.asarray(j["xy"], float), float(j["at_m"]))
    line = np.asarray(j["line"], float)
    if len(line) < 2 or cl.cumulative(line)[-1] < MIN_RIBBON_M:
        return None, "short"
    ease = np.clip(1.0 - cl.cumulative(line) / START_EASE_M, 0.0, 1.0)[:, None]
    line = line + (m.at(at) - line[0]) * ease
    try:
        sm = cl.smooth(line)
    except ValueError:
        return None, "short"
    near = m.near(at, sm.s[-1] + MAIN_REACH_M)

    # The side road's first direction carried back across the mainline, so its edges can be seen to leave the road.
    u0 = np.array([math.cos(sm.heading[0]), math.sin(sm.heading[0])])
    back = np.arange(BACK_M, 0.0, -1.0)
    pe = np.vstack([sm.xy[0] - back[:, None] * u0, sm.xy])
    he = np.concatenate([np.full(len(back), sm.heading[0]), sm.heading])
    de = np.concatenate([-back, sm.s])
    d0 = _crossing(de, near.beyond(pe, side), len(back))
    if d0 is None:
        return None, "stays_on_road"

    # Width, surface and look, by the mainline's rules.
    tags = {**(j.get("tags") or {}), "highway": j.get("highway")}
    surface = sec.surface_class(tags)
    if surface is None:  # untagged: as the mainline is where it meets it
        surface = sec.UNSEALED if sec.is_unsealed(m.style[int(round(m.row(at)))]) else sec.SEALED
    unsealed = surface == sec.UNSEALED
    mapped, source = sec.width_from_tags(tags, unsealed=unsealed)
    width = mapped * width_factor
    w = width / 2
    style = sec.style_code(unsealed, sec.single_lane(tags, mapped, unsealed))
    crown = sec.CROWN_UNSEALED if unsealed else sec.CROWN

    # Where each edge leaves the mainline's edge: the corners.
    ne = cl.normals(he)
    corners = []
    for es in (1.0, -1.0):  # the stub's left edge, then its right, looking out along it
        g = near.beyond(pe + es * w * ne, side)
        within = np.flatnonzero(de <= d0 + CORNER_WITHIN_M)
        k_min = int(within[np.argmin(g[within])])
        dc = _crossing(de[:within[-1] + 1], g[:within[-1] + 1], k_min + 1)
        if dc is None:
            return None, "corner"
        pc = np.array([np.interp(dc, de, pe[:, 0]), np.interp(dc, de, pe[:, 1])]) + es * w * \
            cl.normals(np.array([np.interp(dc, de, he)]))[0]
        sc = float(near.project(pc)[0][0])
        corners.append({"es": es, "dc": dc, "pc": pc, "sc": sc, "heading": float(np.interp(dc, de, he))})
    up, down = sorted(corners, key=lambda k: k["sc"])

    # The plan's rows: every metre from the mainline's edge (d = 0) to the end.
    dd = d0 + np.arange(0.0, sm.s[-1] - d0 + 1e-9, 1.0)
    rows_xy, rows_h, rows_k = _interp_rows(dd, sm.s, sm.xy, sm.heading, sm.curvature)
    rows_n = cl.normals(rows_h)
    n_rows = len(dd)

    def edge_pt(k, es):
        return rows_xy[k] + es * w * rows_n[k]

    sign_side = 1.0 if side == "L" else -1.0
    flares, first = {}, 0
    for name, cn, toward in (("upstream", up, -1.0), ("downstream", down, 1.0)):  # along the road, away from the stub
        hm = float(np.interp(cn["sc"], m.s, m.heading))
        mdir = toward * np.array([math.cos(hm), math.sin(hm)])
        u = np.array([math.cos(cn["heading"]), math.sin(cn["heading"])])
        alpha = float(np.arccos(np.clip(u @ mdir, -1.0, 1.0)))
        half_tan = math.tan(alpha / 2)
        if half_tan < 1e-3:
            return None, "skew"
        radius = max(min(FLARE_RADIUS_M, FLARE_MAX_M * half_tan) * scale[name], FLARE_MIN_RADIUS_M)
        tangent = radius / half_tan
        if tangent > FLARE_MAX_TANGENT_M:
            return None, "skew"
        r_t1 = m.row(cn["sc"] + toward * tangent)
        r_t1 = int(math.floor(r_t1)) if toward < 0 else int(math.ceil(r_t1))
        if r_t1 < 2 or r_t1 > len(m.s) - 3:
            return None, "road_end"
        tangent = abs(float(m.s[r_t1]) - cn["sc"])  # the tangent point on a row of the road's
        # The stub's row the flare ends on: past the arc's end, as far from the corner as the road's tangent point,
        # measured in the road's frame, where the arc is drawn (a bending side road's edge can be shorter than its
        # centreline).
        corner_so = np.array([cn["sc"], np.interp(cn["sc"], m.s, m.half[side])])
        k_t2 = max(int(math.ceil(cn["dc"] + tangent - d0 + 1e-6)), 1)
        while k_t2 < n_rows - 1:
            s_e, o_e = near.project(edge_pt(k_t2, cn["es"]))
            if np.hypot(s_e[0] - corner_so[0], sign_side * o_e[0] - corner_so[1]) >= tangent + 0.25:
                break
            k_t2 += 1
        flares[name] = {"es": cn["es"], "corner": cn["sc"], "row": r_t1, "k": k_t2, "radius": radius,
                        "tangent": tangent, "angle": math.degrees(alpha)}
        first = max(first, k_t2)
    if n_rows - 1 - first < MIN_RIBBON_M:
        return None, "short"

    # Each flare is drawn in the mainline's own frame (station, offset beyond the centreline), where its edge is a
    # straight line, so the arc leaves it tangentially and never dips inside it, wherever the road bends.
    for f in flares.values():
        end = edge_pt(f["k"], f["es"])
        s_end, o_end = near.project(end)
        corner = np.array([f["corner"], np.interp(f["corner"], m.s, m.half[side])])
        so, f["radius"] = _fillet(np.array([m.s[f["row"]], m.half[side][f["row"]]]), corner,
                                  np.array([s_end[0], sign_side * o_end[0]]))
        curve = m.to_xy(so[:, 0], sign_side * so[:, 1])
        curve[0], curve[-1] = m.edge(f["row"], side), end
        f["points"] = np.vstack([curve, [edge_pt(k, f["es"]) for k in range(f["k"] + 1, first + 1)]]) \
            if f["k"] < first else curve
        f["shape_ok"] = _flare_shape_ok(curve, f["angle"])
    a, b = flares["upstream"]["row"], flares["downstream"]["row"]
    if b - a < 2:
        return None, "corner"
    across = [rows_xy[first] + o * rows_n[first] for o in (w, w / 2, 0.0, -w / 2, -w)]  # the stub's left to right
    if flares["downstream"]["es"] < 0:
        across = across[::-1]
    ring = ([m.edge(r, side) for r in range(a, b + 1)] + list(flares["downstream"]["points"][1:])
            + across[1:-1] + list(flares["upstream"]["points"][::-1][:-1]))
    throat = shapely.Polygon(ring)
    lo, hi = max(a - 20, 0), min(b + 20, len(m.s) - 1)
    road = shapely.Polygon([m.edge(r, "L") for r in range(lo, hi + 1)] + [m.edge(r, "R") for r in range(hi, lo - 1, -1)])
    inside = [rows_xy[k] + t * w * rows_n[k] for k in range(1, first) for t in (0.0, 0.5, -0.5)] if first > 1 else []
    if (not throat.is_valid or throat.area < 1.0 or throat.intersection(road.buffer(0)).area > 0.05
            or not all(f["shape_ok"] for f in flares.values())
            or not shapely.union(throat, road).buffer(0.01).contains(shapely.MultiPoint(inside or [rows_xy[0]]))):
        # A side road bending hard inside a long flare can leave the flare's straight run across its carriageway:
        # smaller flares follow it more closely. Its points on the mainline's own road are road already.
        if min(scale.values()) > SHRINK ** 2 - 1e-9:
            return _candidate(j, m, dem_sample, width_factor, {k: v * SHRINK for k, v in scale.items()})
        return None, "flare"

    # Heights: the mainline's cross-fall carried on, blending into the side road's own profile and crown.
    own = dem_sample(rows_xy[:, 0], rows_xy[:, 1])
    if not np.isfinite(own).any():
        own = np.full(n_rows, m.edge_h(int(round(m.row(at))), side))
    else:
        ok = np.isfinite(own)
        own = np.interp(np.arange(n_rows), np.flatnonzero(ok), own[ok])
    own = uniform_filter1d(own, max(int(round(DEM_SMOOTH_M)), 1), mode="nearest")
    rise = float(own[0] - near.surface(rows_xy[:1], side)[0])
    blend = float(np.clip(max(BLEND_M, abs(rise) / MAX_BLEND_GRADE), BLEND_M, max(BLEND_M, 0.8 * (n_rows - 1))))
    row_tree = cKDTree(rows_xy)

    def height(p):
        p = np.atleast_2d(np.asarray(p, float))
        _, k = row_tree.query(p)
        t = np.column_stack([np.cos(rows_h[k]), np.sin(rows_h[k])])
        rel = p - rows_xy[k]
        d = np.clip(k + (rel * t).sum(axis=1), 0.0, n_rows - 1)
        lateral = (rel * rows_n[k]).sum(axis=1)
        stub_h = np.interp(d, np.arange(n_rows), own) - crown * np.abs(lateral)
        main_h = near.surface(p, side)
        wgt = _smoothstep(near.beyond(p, side) / blend)
        return main_h + wgt * (stub_h - main_h)

    z = height(rows_xy)
    z_left = height(rows_xy + w * rows_n)
    z_right = height(rows_xy - w * rows_n)
    for f in flares.values():
        h = height(f["points"])
        h[0] = m.edge_h(f["row"], side)
        h[-1] = z_left[first] if f["es"] > 0 else z_right[first]
        f["heights"] = h

    carriageway = shapely.Polygon(np.vstack([rows_xy[first:] + w * rows_n[first:],
                                             (rows_xy[first:] - w * rows_n[first:])[::-1]])).buffer(0)
    outline = shapely.unary_union([throat.buffer(0), carriageway])
    if outline.geom_type != "Polygon":
        outline = max(outline.geoms, key=lambda g: g.area)
    angle = math.degrees(math.atan2(math.sin(rows_h[0] - np.interp(at, m.s, m.heading)),
                                    math.cos(rows_h[0] - np.interp(at, m.s, m.heading))))

    def r2(a):
        return np.round(np.asarray(a, float), 2).tolist()

    def r3(a):
        return np.round(np.asarray(a, float), 3).tolist()

    record = {
        "junction": j["id"], "name": j.get("name"), "highway": j.get("highway"), "closure": j.get("closure"),
        "side": j["side"], "station_m": round(at, 2), "angle_deg": round(angle, 1),
        "surface": surface, "style": sec.STYLES[style], "width_m": round(width, 3), "width_mapped_m": round(mapped, 2),
        "width_source": source, "crown": crown, "length_m": float(n_rows - 1), "ends": j.get("ends"),
        "ribbon_from_m": float(first), "blend_m": round(blend, 1),
        "mouth": {"from_m": round(float(m.s[a]), 2), "to_m": round(float(m.s[b]), 2), "rows": [a, b]},
        "flares": {name: {"stub_side": "left" if f["es"] > 0 else "right", "radius_m": round(f["radius"], 2),
                          "tangent_m": round(f["tangent"], 2), "angle_deg": round(f["angle"], 1),
                          "mainline_row": f["row"], "stub_row": f["k"],
                          "points": [[*p, h] for p, h in zip(r2(f["points"]), r3(f["heights"]))]}
                   for name, f in flares.items()},
        "rows": {"d": list(range(n_rows)), "xy": r2(rows_xy), "z": r3(z), "z_left": r3(z_left),
                 "z_right": r3(z_right), "heading": np.round(rows_h, 5).tolist(),
                 "curvature": np.round(rows_k, 5).tolist(), "grade": np.round(np.gradient(z), 4).tolist(),
                 "half_m": [round(w, 3)] * n_rows},
        "outline": r2(np.asarray(outline.exterior.coords)[:-1]),
    }
    priority = (0 if j.get("closure") == "major" else 1, -(n_rows - 1), j["id"])
    return _Candidate(record, priority, throat.buffer(0), rows_xy[first:], rows_n[first:], w, first), None


def _body(cand: _Candidate) -> np.ndarray:
    """A stub's ribbon from its first row, its road and verges, as quads from row to row."""
    off = cand.half + mesh.VERGE_M
    left, right = cand.rows_xy + off * cand.rows_n, cand.rows_xy - off * cand.rows_n
    return shapely.polygons(np.stack([left[:-1], left[1:], right[1:], right[:-1]], axis=1))


def _clip(cand: _Candidate, rows: int) -> _Candidate | None:
    """The stub cut to its first `rows` ribbon rows, or None if too little is left."""
    if rows - 1 < MIN_RIBBON_M:
        return None
    rec = cand.record
    keep = cand.first + rows  # plan rows kept
    rec["rows"] = {k: v[:keep] for k, v in rec["rows"].items()}
    rec["length_m"] = float(keep - 1)
    rec["ends"] = "clipped"
    xy = np.asarray(rec["rows"]["xy"], float)[cand.first:]
    n = cand.rows_n[:rows]
    carriageway = shapely.Polygon(np.vstack([xy + cand.half * n, (xy - cand.half * n)[::-1]])).buffer(0)
    outline = shapely.unary_union([cand.throat, carriageway])
    if outline.geom_type != "Polygon":
        outline = max(outline.geoms, key=lambda g: g.area)
    rec["outline"] = np.round(np.asarray(outline.exterior.coords)[:-1], 2).tolist()
    cand.rows_xy, cand.rows_n = cand.rows_xy[:rows], n
    return cand


def _fit(cand: _Candidate, m: _Mainline, kept: list[_Candidate]) -> tuple[_Candidate | None, str | None]:
    """The stub cut short where it would come within a verge of another part of the route, or overlap a stub already
    kept; None, with the reason, if too little is left, or its mouth shares the road's rows with a kept stub's on
    the same side, or its throat or its flares' verges overlap a kept stub."""
    at = cand.record["station_m"]
    dist, idx = m.tree.query(cand.rows_xy)
    # Another part: much further along the road than in a straight line, as junctions.json judges it.
    other = np.abs(m.s[idx] - at) > jn.OTHER_PART_RATIO * dist + jn.OTHER_PART_SLACK_M
    need = np.maximum(m.half["L"][idx], m.half["R"][idx]) + cand.half + 2 * mesh.VERGE_M + CLEAR_M
    close = np.flatnonzero(other & (dist < need))
    if len(close):
        cand = _clip(cand, int(close[0]) - 2)
        if cand is None:
            return None, "near_route"
    if not kept:
        return cand, None
    a, b = cand.record["mouth"]["rows"]
    for k in kept:
        ka, kb = k.record["mouth"]["rows"]
        if k.record["side"] == cand.record["side"] and a < kb and ka < b:
            return None, "overlaps"
    bodies = shapely.unary_union([k.body for k in kept])
    if cand.throat.intersection(shapely.union(bodies, shapely.unary_union([k.throat for k in kept]))).area > 0.01 \
            or cand.throat.buffer(mesh.VERGE_M).intersects(bodies):
        return None, "overlaps"
    hit = np.flatnonzero(shapely.intersects(_body(cand), shapely.union(
        bodies, shapely.unary_union([k.throat.buffer(mesh.VERGE_M) for k in kept]))))
    if len(hit):
        cand = _clip(cand, int(hit[0]))
        if cand is None:
            return None, "overlaps"
    return cand, None


def plan(junctions: list[dict], c, section, dem_sample, *, width_factor: float = 1.0,
         layouts: tuple[bool, ...] = (False,)) -> dict:
    """The stubs for the junctions junctions.json lists: the content of stubs.json (without `crs`). `c` and
    `section` are the centreline's and cross-section's arrays (`centreline.npz`, `section.npz`), `dem_sample` the
    region's elevation, `layouts` whether each layout is reversed (for the timing gates)."""
    m = _Mainline(c, section)
    cands, skipped = [], []
    for j in junctions:
        cand, why = _candidate(j, m, dem_sample, width_factor)
        if cand is None:
            skipped.append({"junction": j["id"], "name": j.get("name"), "reason": why})
        else:
            cands.append((cand, j))
    kept: list[_Candidate] = []
    for cand, j in sorted(cands, key=lambda cj: cj[0].priority):
        fitted, reason = _fit(cand, m, kept)
        if reason == "overlaps":  # with smaller flares, it may fit beside the other
            again, _ = _candidate(j, m, dem_sample, width_factor, {"upstream": SHRINK, "downstream": SHRINK})
            if again is not None:
                fitted, reason = _fit(again, m, kept)
        if fitted is None:
            skipped.append({"junction": j["id"], "name": j.get("name"), "reason": reason})
            continue
        fitted.body = shapely.unary_union(_body(fitted)).buffer(0)
        kept.append(fitted)
    kept = _clear_gates(kept, junctions, c, section, layouts, skipped)
    kept.sort(key=lambda k: (k.record["station_m"], k.record["junction"]))
    return {"version": FORMAT_VERSION,
            "rule": {"flare_radius_m": FLARE_RADIUS_M, "flare_max_m": FLARE_MAX_M,
                     "flare_min_radius_m": FLARE_MIN_RADIUS_M, "blend_m": BLEND_M, "max_blend_grade": MAX_BLEND_GRADE,
                     "width_factor": width_factor},
            "stubs": [k.record for k in kept],
            "skipped": sorted(skipped, key=lambda s: s["junction"])}


def _clear_gates(kept: list[_Candidate], junctions: list[dict], c, section, layouts, skipped) -> list[_Candidate]:
    """The stubs less any whose mouth would leave a timing gate nowhere clear to stand: a gate end on a stub would
    fail validation ("gate spans the road"). The track stage moves gates out of mouths (`layout.gate_distances`)."""
    s = np.asarray(c["s"], float)
    while kept:
        by_id = {k.record["junction"]: k for k in kept}
        mouths = jn.mouths(junctions, s, c["xy"], c["heading"], section["half_left"], section["half_right"],
                           stubs={i: k.record for i, k in by_id.items()})
        drop = None
        for reverse in layouts:
            road = RoadModel.from_arrays(s, c["xy"], c["z"], c["heading"], c["curvature"], c["grade"],
                                         section["half_left"], section["half_right"], section["slope_left"],
                                         section["slope_right"], reverse)
            total = float(s[-1] - s[0])
            spans = [((total - (mo.s1 - s[0]), total - (mo.s0 - s[0])) if reverse else (mo.s0 - s[0], mo.s1 - s[0]),
                      mo) for mo in mouths]
            for d in gate_distances(road, sorted(sp for sp, _ in spans)):
                blocking = [by_id[mo.junction["id"]] for (lo, hi), mo in spans
                            if lo <= d <= hi and mo.junction["id"] in by_id]
                if blocking:
                    drop = max(blocking, key=lambda k: k.priority)
                    break
            if drop is not None:
                break
        if drop is None:
            return kept
        kept = [k for k in kept if k is not drop]
        skipped.append({"junction": drop.record["junction"], "name": drop.record["name"], "reason": "timing_gate"})
    return kept


# -- reading the plan -------------------------------------------------------------------------------------------------

def by_junction(stubs: list[dict] | None) -> dict[int, dict]:
    """Stubs (stubs.json's `stubs`), by their junction's id."""
    return {s["junction"]: s for s in stubs or []}


def pose(stub: dict):
    """The stub's centreline as a function of the distance in from the mainline's edge line (d, metres), giving
    (easting, northing, height, heading): heading in radians, east = 0, counter-clockwise, pointing into the stub.
    Past the end it carries on in a straight line at the last row's grade. This is the form `core.closures` places
    its props with."""
    r = stub["rows"]
    d, xy, z, hd = (np.asarray(r[k], float) for k in ("d", "xy", "z", "heading"))

    def at(x: float) -> tuple[float, float, float, float]:
        if x <= d[-1]:
            return (float(np.interp(x, d, xy[:, 0])), float(np.interp(x, d, xy[:, 1])), float(np.interp(x, d, z)),
                    float(np.interp(x, d, hd)))
        extra = x - d[-1]
        grade = float(r["grade"][-1])
        return (float(xy[-1, 0] + extra * math.cos(hd[-1])), float(xy[-1, 1] + extra * math.sin(hd[-1])),
                float(z[-1] + grade * extra), float(hd[-1]))

    return at


def surface_height(stub: dict, d: float, lateral: float) -> float:
    """The stub's road surface `lateral` metres to the left (+) or right (-) of its centreline, `d` metres in: straight
    from the centreline's height to each edge's, as the mesh is. Beyond an edge the surface's fall carries on."""
    r = stub["rows"]
    ds = np.asarray(r["d"], float)
    z = float(np.interp(d, ds, r["z"]))
    edge = float(np.interp(d, ds, r["z_left" if lateral >= 0 else "z_right"]))
    half = float(np.interp(d, ds, r["half_m"]))
    return z + (edge - z) * abs(lateral) / half


def ground_height(stub: dict, d: float, lateral: float, dem_sample=None) -> float:
    """The ground's height at a point beside or on the stub: its surface over the carriageway; its verge, falling
    `mesh.VERGE_FALL` over `mesh.VERGE_M` from the edge, beside it; beyond that the DEM (`dem_sample(x, y)`) where
    given, or the verge's outer height."""
    half = float(np.interp(d, stub["rows"]["d"], stub["rows"]["half_m"]))
    if abs(lateral) <= half:
        return surface_height(stub, d, lateral)
    edge = surface_height(stub, d, math.copysign(half, lateral))
    if abs(lateral) <= half + mesh.VERGE_M or dem_sample is None:
        return edge - mesh.VERGE_M * mesh.VERGE_FALL * min(abs(lateral) - half, mesh.VERGE_M) / mesh.VERGE_M
    e, n, _, hd = pose(stub)(d)
    x, y = e - math.sin(hd) * lateral, n + math.cos(hd) * lateral
    h = float(np.asarray(dem_sample(np.array([x]), np.array([y])))[0])
    return h if np.isfinite(h) else edge - mesh.VERGE_M * mesh.VERGE_FALL


# -- the meshes -------------------------------------------------------------------------------------------------------

def side_caps(xy: np.ndarray, nrm: np.ndarray, others: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """How far from each centreline point its skirt may reach on each side so it stays nearer its own centreline than
    any of `others` (another road's centreline points): for a point q at distance D, off to the side by a, at most
    D^2 / 2a. Points further than `radius` don't count. Inf where nothing is near."""
    n = len(xy)
    cap_l, cap_r = np.full(n, np.inf), np.full(n, np.inf)
    others = np.asarray(others, float).reshape(-1, 2)
    if not len(others):
        return cap_l, cap_r
    tree = cKDTree(others)
    for i, near in enumerate(tree.query_ball_point(np.asarray(xy, float), r=radius)):
        if not near:
            continue
        dv = others[near] - xy[i]
        d2 = (dv * dv).sum(axis=1)
        a = dv @ nrm[i]
        if (a > 1e-6).any():
            cap_l[i] = float((d2[a > 1e-6] / (2 * a[a > 1e-6])).min())
        if (a < -1e-6).any():
            cap_r[i] = float((d2[a < -1e-6] / (-2 * a[a < -1e-6])).min())
    return cap_l, cap_r


def _ribbon_rows(stub: dict) -> np.ndarray:
    """The plan rows the stub's own ribbon is built on: from its first, every metre to FINE_ROWS_M, then every
    COARSE_STEP_M, and its last."""
    first, last = int(stub["ribbon_from_m"]), len(stub["rows"]["d"]) - 1
    fine = np.arange(first, min(max(int(FINE_ROWS_M), first), last) + 1)
    coarse = np.arange(fine[-1] + COARSE_STEP_M, last, COARSE_STEP_M).astype(int)
    return np.unique(np.concatenate([fine, coarse, [last]]))


def _centre(stub: dict, rows: np.ndarray | None = None) -> np.ndarray:
    xy = np.asarray(stub["rows"]["xy"], float)
    return xy[_ribbon_rows(stub) if rows is None else rows]


REACH_M = 2 * (mesh.VERGE_M + mesh.SKIRT_STEPS_M[-1] + 4.0)  # skirts further apart than this never meet


def mainline_caps(stubs: list[dict], xy: np.ndarray, heading: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mainline's skirt caps (`mesh.build_ribbon(skirt_cap_l=, skirt_cap_r=)`): held short of every stub's ribbon."""
    pts = np.vstack([_centre(s) for s in stubs]) if stubs else np.zeros((0, 2))
    return side_caps(np.asarray(xy, float), cl.normals(np.asarray(heading, float)), pts, REACH_M)


@dataclass
class Built:
    """The stubs' meshes and what the mainline and terrain need to make room for them."""

    meshes: list[Mesh]  # roads (1ROAD_J / 1GRAVEL_J) and verges (1GRASS_J), packed by material per WINDOW_M
    cut: dict[str, np.ndarray]  # the mainline's verge and skirt quads taken out, per side (`mesh.ribbon_meshes`)
    edge_break: dict[str, np.ndarray]  # the mainline's rows whose edge line breaks, per side
    footprint: mesh.Footprint  # for the terrain (`mesh.terrain_meshes(extra=)`)
    stats: dict


def _triangulate(boundary: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Triangles over a simple polygon's boundary points (in order) and points inside it, indexing
    vstack([boundary, inner]); every boundary edge is a triangle's edge, and the triangles cover the polygon exactly.
    A Delaunay triangulation of them all, kept inside the polygon; if that loses an edge, the polygon's own
    constrained triangulation, without the inner points."""
    poly = shapely.Polygon(boundary)
    pts = np.vstack([boundary, inner]) if len(inner) else boundary
    nb = len(boundary)

    def ok(tri):
        if not len(tri):
            return False
        p = pts[tri]
        area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                            - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
        if abs(area.sum() - poly.area) > 1e-6 * max(poly.area, 1.0) + 1e-4:
            return False
        edges = {frozenset(e) for t in tri.tolist() for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
        return all(frozenset((k, (k + 1) % nb)) in edges for k in range(nb))

    tri = Delaunay(pts).simplices
    cen = pts[tri].mean(axis=1)
    tri = tri[shapely.contains_xy(poly, cen[:, 0], cen[:, 1])]
    if ok(tri):
        return tri
    index = {(float(x), float(y)): i for i, (x, y) in enumerate(boundary.tolist())}
    out = []
    for g in shapely.get_parts(shapely.constrained_delaunay_triangles(poly)):
        corners = np.asarray(g.exterior.coords)[:3]
        out.append([index[(float(x), float(y))] for x, y in corners.tolist()])
    tri = np.array(out, dtype=np.int64)
    if not ok(tri):
        raise ValueError("throat could not be triangulated")
    return tri


def _dedupe(points: np.ndarray, heights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.concatenate([[True], np.hypot(*np.diff(points, axis=0).T) > 1e-6])
    return points[keep], heights[keep]


def _tri_wave(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs(np.mod(x, 2.0) - 1.0)


def build(stubs: list[dict], rb: mesh.Ribbon, s: np.ndarray, frame: LocalFrame, dem_sample, *, skirt: bool,
          main_xy: np.ndarray) -> Built:
    """The stubs' meshes, the mainline's cut and edge break, and the terrain's footprint, from the plan's stubs.

    `rb` is the mainline's ribbon (built with `mainline_caps`), `s` its stations, `main_xy` its centreline points.
    With `skirt` False (ground classes) the stubs' verge meshes are the 2 m verge alone, and their skirts go with the
    terrain, as the mainline's do."""
    rows = len(s)
    cut = {"L": np.zeros(rows - 1, bool), "R": np.zeros(rows - 1, bool)}
    edge_break = {"L": np.zeros(rows, bool), "R": np.zeros(rows, bool)}
    roads: dict[tuple[str, int], list] = {}
    grass: dict[int, list] = {}
    ribbons, patches, edges = [], [], []
    centres = [_centre(st) for st in stubs]
    vertices = 0
    for k, st in enumerate(stubs):
        side = jn.SIDE_CODE[st["side"]]
        a, b = st["mouth"]["rows"]
        cut[side][a:b] = True
        edge_break[side][a + 1:b] = True
        r = st["rows"]
        ribbon_rows = _ribbon_rows(st)
        xy = np.asarray(r["xy"], float)
        z, zl, zr = (np.asarray(r[key], float) for key in ("z", "z_left", "z_right"))
        heading, curv = np.asarray(r["heading"], float), np.asarray(r["curvature"], float)
        half = np.asarray(r["half_m"], float)
        sel = ribbon_rows
        others = np.vstack([main_xy] + [c for i, c in enumerate(centres) if i != k])
        cap_l, cap_r = side_caps(xy[sel], cl.normals(heading[sel]), others, REACH_M)
        sb = mesh.build_ribbon(xy[sel], z[sel], heading[sel], curv[sel], half[sel], half[sel],
                               (z[sel] - zl[sel]) / half[sel], (z[sel] - zr[sel]) / half[sel], dem_sample,
                               skirt_cap_l=cap_l, skirt_cap_r=cap_r)
        ribbons.append(sb)
        material = st["style"]
        tile = textures.GRAVEL_TILE_M if st["surface"] == sec.UNSEALED else textures.ROAD_TILE_M
        window = int(st["station_m"] // WINDOW_M)

        # The stub's own road and verges.
        rc = sb.road_cols
        e, n, h = sb.e[:, rc], sb.n[:, rc], sb.h[:, rc]
        nr, nc = e.shape
        uv = np.column_stack([np.tile(np.linspace(0, 1, nc), nr), np.repeat(ribbon_rows / tile, nc)])
        roads.setdefault((material, window), []).append(
            (np.column_stack([e.ravel(), n.ravel(), h.ravel()]), uv, mesh._grid_triangles(nr, nc)))
        left, right = (sb.left_cols, sb.right_cols) if skirt else mesh._verge_strips(sb)
        for cols in (left, right):
            e, n, h = sb.e[:, cols], sb.n[:, cols], sb.h[:, cols]
            nr, nc = e.shape
            grass.setdefault(window, []).append((np.column_stack([e.ravel(), n.ravel(), h.ravel()]), None,
                                                 mesh._grid_triangles(nr, nc)))

        # The throat: the mainline's edge vertices, the flares, the ribbon's first row, and the stub's own
        # cross-section inside it.
        edge_col = rb.road_cols.start if side == "L" else rb.road_cols.stop - 1
        verge_col = edge_col - 1 if side == "L" else edge_col + 1
        main_edge = np.column_stack([rb.e[a:b + 1, edge_col], rb.n[a:b + 1, edge_col]])
        main_edge_h = rb.h[a:b + 1, edge_col]

        def stub_col(es, verge=False):
            col = (rc.start if es > 0 else rc.stop - 1)
            return col + (0 if not verge else (-1 if es > 0 else 1))

        flares = {}
        for name in ("upstream", "downstream"):
            f = st["flares"][name]
            es = 1.0 if f["stub_side"] == "left" else -1.0
            pts = np.asarray(f["points"], float)
            fxy, fh = pts[:, :2].copy(), pts[:, 2].copy()
            row = a if name == "upstream" else b
            fxy[0], fh[0] = (rb.e[row, edge_col], rb.n[row, edge_col]), rb.h[row, edge_col]
            col = stub_col(es)
            fxy[-1], fh[-1] = (sb.e[0, col], sb.n[0, col]), sb.h[0, col]
            fxy, fh = _dedupe(fxy, fh)
            flares[name] = (es, row, fxy, fh)
        across = [(sb.e[0, cc], sb.n[0, cc], sb.h[0, cc]) for cc in range(rc.start, rc.stop)]  # stub's left to right
        if flares["downstream"][0] < 0:
            across = across[::-1]
        across = np.array(across)
        down, up = flares["downstream"], flares["upstream"]
        ring = np.vstack([main_edge, down[2][1:], across[1:-1, :2], up[2][::-1][:-1]])
        ring_h = np.concatenate([main_edge_h, down[3][1:], across[1:-1, 2], up[3][::-1][:-1]])
        first = int(st["ribbon_from_m"])
        inner, inner_h = [], []
        poly = shapely.Polygon(ring)
        shapely.prepare(poly)
        boundary = poly.exterior
        for kk in range(1, first):
            nrm = cl.normals(np.array([heading[kk]]))[0]
            for t in (1.0, 0.5, 0.0, -0.5, -1.0):
                p = xy[kk] + nrm * t * half[kk]
                if poly.contains(shapely.Point(p)) and boundary.distance(shapely.Point(p)) >= STEINER_CLEAR_M:
                    inner.append(p)
                    edge = zl[kk] if t > 0 else zr[kk]
                    inner_h.append(z[kk] + (edge - z[kk]) * abs(t))
        inner = np.array(inner).reshape(-1, 2)
        tri = _triangulate(ring, inner)
        pts = np.vstack([ring, inner])
        hts = np.concatenate([ring_h, np.asarray(inner_h, float)])
        # Plain surface: (d, lateral) from the ribbon's first row, the lateral mirrored inside an unmarked band.
        u0, u1 = THROAT_U[material]
        band = (u1 - u0) * TEXTURE_WIDTH_M[material]
        o = np.array([sb.e[0, rc.start + 2], sb.n[0, rc.start + 2]])
        hd = float(heading[first])
        along = (pts - o) @ np.array([math.cos(hd), math.sin(hd)]) + first
        lateral = (pts - o) @ np.array([-math.sin(hd), math.cos(hd)])
        uv = np.column_stack([u0 + (u1 - u0) * _tri_wave(lateral / band), along / tile])
        roads.setdefault((material, window), []).append((np.column_stack([pts, hts]), uv, tri))
        patches.append(ring)

        # A mown verge along each flare, from the mainline's verge to the stub's.
        for es, row, fxy, fh in (up, down):
            outer_main = np.array([rb.e[row, verge_col], rb.n[row, verge_col]])
            col = stub_col(es, verge=True)
            outer_stub = np.array([sb.e[0, col], sb.n[0, col]])
            w0 = float(np.hypot(*(outer_main - fxy[0])))
            w1 = float(np.hypot(*(outer_stub - fxy[-1])))
            tang = np.gradient(fxy, axis=0)
            tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
            perp = np.column_stack([-tang[:, 1], tang[:, 0]])
            if perp[0] @ (outer_main - fxy[0]) < 0:
                perp = -perp
            frac = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(fxy, axis=0).T))])
            frac /= max(frac[-1], 1e-9)
            outer = fxy + perp * (w0 + (w1 - w0) * frac)[:, None]
            outer_h = fh - mesh.VERGE_M * mesh.VERGE_FALL
            outer[0], outer_h[0] = outer_main, rb.h[row, verge_col]
            outer[-1], outer_h[-1] = outer_stub, sb.h[0, col]
            strip_xy = np.stack([fxy, outer], axis=1)  # (points, 2 columns, 2)
            strip_h = np.stack([fh, outer_h], axis=1)
            grass.setdefault(window, []).append((np.column_stack([strip_xy.reshape(-1, 2), strip_h.ravel()]), None,
                                                 mesh._grid_triangles(len(fxy), 2)))
            patches.append(np.vstack([fxy, outer[::-1]]))
            edges.append((outer, outer_h))

    meshes: list[Mesh] = []
    letter = {"road": "", "road_plain": "N", "gravel": "", "gravel_narrow": "N"}
    for (material, window), pieces in sorted(roads.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        key = "1GRAVEL" if sec.is_unsealed(sec.STYLES.index(material)) else "1ROAD"
        meshes += mesh.soup_meshes(f"{key}_J{window:03d}{letter[material]}", material, pieces, frame)
    for window, pieces in sorted(grass.items()):
        # As the mainline's verges: AC x and z over 6 m, so they meet without a seam.
        pieces = [(p, frame.to_ac(p[:, 0], p[:, 1], p[:, 2])[:, [0, 2]] / 6.0, t) for p, _, t in pieces]
        meshes += mesh.soup_meshes(f"1GRASS_J{window:03d}", "grass", pieces, frame)
    for msh in meshes:
        vertices += len(msh.positions)
    stats = {"stubs": len(stubs), "meshes": len(meshes), "vertices": vertices,
             "triangles": int(sum(len(msh.triangles) for msh in meshes)),
             "length_m": round(sum(st["length_m"] - st["ribbon_from_m"] for st in stubs), 1)}
    return Built(meshes, cut, edge_break, mesh.Footprint(cut, ribbons, patches, edges), stats)
