"""Roundabouts the route drives: the rest of the ring drawn as road, a grass island inside it, and every exit the
route doesn't take closed off with a stub, as a side road is.

The maintainer's answers to the intersections research (10 Nov 2026) this follows:
- **Q14:** draw the whole ring and a grass island, with no collider on the island; close the other exits; keep the
  route's centreline exactly as it is; and say at preview time when a ring is small enough that the route may fail
  validation's 5 m bend check (`preview_notes`).
- **Q15:** a reverse layout drives the ring against its one-way. Nothing on the ring says which way to go: no arrows
  or chevrons anywhere (Q12).

**What the route stage records** (`core.junctions`): each ring the route drives, its nodes in the order it's driven,
where the route joins and leaves it, and its exits. A way leaving the ring where the route is on it is a side road
like any other, so `core.stubs` closes it. The rest are this module's.

**The plan** (`plan`, run by the section stage; `03_section/roundabouts.json`, its format in docs/ac-format.md):
- **The ring's line:** the mapped ring from where the route leaves it, round to where it joins it, carried on
  EXTEND_M back along the route's own arc at each end, and smoothed as the route is (`centreline.smooth`). At each
  end its first EASE_M are moved so its inner edge starts just inside the road's edge (FIT_INSIDE_M): where the
  route leaves the ring the two part along the inner edge, not with a step (the route's line is up to 3 m off the
  mapped ring on small rings). Its rows are a metre apart.
- **Width, surface and look** by the road's rules (`core.section`): each of the ring's ways' tagged or class-default
  width as mapped, times the job's width option, eased along the ring; sealed or unsealed by its `surface`. Crown,
  no camber.
- **Heights:** the DEM along the ring (smoothed over DEM_SMOOTH_M) with its crown, blended into the road's own
  cross-fall near the road (`stubs.BLEND_M`, longer where the ring sits well above or below the road), so where they
  meet the ring is the road's surface carried on.
- **The two ends** (`_end`) are joined to the road as a side road's stub is (`core.stubs`): where each of the ring's
  edges leaves the road's edge, a flare tangent to both (8 m, the route's corner radius; capped as a stub's are),
  and between them the throat, triangulated over the road's own edge vertices. The road's edge rows between the
  flares are the mouth. The flare on the island's side is short: there the ring's inner edge peels off the road's.
- **The island:** inside the ring's inner edge and the road's island-side edge, between the two ends. Its edge is
  every vertex of the road's edge, the island-side flares and the ring's inner edge, shared exactly. It rises
  ISLAND_KERB_H over ISLAND_KERB_W at its edge (a mountable kerb, concrete), then domes gently (DOME_GRADE, at most
  DOME_MAX_M) under mown grass: the verge's `grass` material, Grass FX class A where the grass pack is on. No
  collider: a car that cuts across it drives on grass. Trees stay off it unless its inner radius is at least
  TREES_INNER_RADIUS_M, and then they keep the road's clearance from its edge; guide posts and rails never stand
  on it.
- **The road beside it:** its verge and skirt on the island's side come out from one end's mouth to the other's
  (the island, throats and flares cover that ground), and its edge line there breaks, as across a side road's mouth.
  The ring has an edge line on its outer edge only: none on the island's side, and none across the route's own
  entry and exit, where the road is the route's.
- **Closed exits:** each exit off the rest of the ring is a stub leaving the ring (`stubs.plan` with the ring as the
  road), closed as a side road is (`core.stub_closures`), its level by its class. An exit that meets the route again
  within its first 100 m is the other carriageway of a road the route arrives or leaves by: it isn't drawn. Two
  exits that become one road within 100 m (the two one-way arms of a road with a splitter island) are one stub, down
  the middle, at that road's width.
- A ring whose ends can't be joined, whose island isn't one piece, or whose mouth would leave a timing gate nowhere
  to stand is left out whole, with the reason (`skipped`); the route's arc is then built as before. Each exit's stub
  is kept or left out on its own, as a side road's is.

**The meshes** (`build`, run by the mesh stage), from the plan alone: the ring's own ribbon (`1ROAD_RB<k>_###`,
verge `1GRASS_RB<k>_V###L/R` on its outer side), the throats (`1ROAD_RB<k>_J`), the outer flares' verges, the
island (`1GRASS_RB<k>_I`) and its kerb (`1ROAD_RB<k>_K`, material `kerb`), and the exits' stubs (`stubs.build` on the
ring's ribbon). The terrain leaves room for them all, sharing their outlines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

from mapcreator.core import centreline as cl
from mapcreator.core import junctions as jn
from mapcreator.core import mesh
from mapcreator.core import section as sec
from mapcreator.core import stubs
from mapcreator.core import textures
from mapcreator.core.scene import LocalFrame, Mesh

FORMAT_VERSION = 1

EXTEND_M = 25.0  # the ring's line carries on this far back along the route's arc at each end, at most
EXTEND_SHARE = 0.4  # and never more than this share of the route's arc
FIT_INSIDE_M = 0.3  # at each end the ring's inner edge starts this far inside the road's edge
EASE_M = 15.0  # and eases back onto the mapped ring over this, past where the route leaves it
DEM_SMOOTH_M = stubs.DEM_SMOOTH_M
WIDTH_EASE_M = 30.0
MIN_RIBBON_M = 10.0  # a ring with less than this of its own ribbon between its ends isn't drawn
CORNER_WITHIN_M = 40.0  # each of the ring's edges must leave the road's edge within this of its centreline
# The flares where the ring meets the road are a splitter island's nose, tighter than a side road's corner (8 m): so
# the throats stay short and leave the ring's own ribbon room for its exits. Not under stubs.FLARE_MIN_RADIUS_M, or
# a flare's verge would fold over itself.
END_FLARE_RADIUS_M = 4.0
# An exit that doesn't fit on the ring with a side road's 8 m flares tries again with these shares of them.
EXIT_FLARE_SCALES = (1.0, 0.5)
ISLAND_KERB_H = 0.15  # the island's kerb: 150 mm high (a mountable kerb)
ISLAND_KERB_W = 0.3  # and this wide
DOME_GRADE = 0.03  # inside the kerb the grass rises this much a metre in from the kerb
DOME_MAX_M = 0.5  # to at most this above the kerb's top
ISLAND_LATTICE_M = 3.0  # the island's grass has a point every this many metres inside its kerb
ISLAND_POINT_CLEAR_M = 1.0  # kept this far from the kerb, so no sliver triangles
TREES_INNER_RADIUS_M = 15.0  # an island at least this big (its inner radius) may have trees, clear of its edge
SMALL_RING_M = 10.3  # the preview warns of a ring this small or smaller: the largest that failed in the survey
EXIT_ID_BASE = 100000  # the ring's exits are side roads too; their ids are this plus 100 times the ring's, plus k
KERB_MATERIAL = "kerb"
KERB_TEXTURE = "kerb"
KERB_TILE_M = 2.0


# -- the ring's line and rows -----------------------------------------------------------------------------------------

def _loop_sample(loop: np.ndarray, start_m: float, end_m: float) -> np.ndarray:
    """The closed ring `loop` (its nodes, in driving order, not repeated) from `start_m` to `end_m` along it, measured
    from its first node (either may be negative or past its perimeter: it goes round), with every node between."""
    closed = np.vstack([loop, loop[:1]])
    s = cl.cumulative(closed)
    per = float(s[-1])
    laps = int(math.floor(start_m / per)) - 1, int(math.ceil(end_m / per)) + 1
    ss = np.concatenate([s[:-1] + k * per for k in range(laps[0], laps[1] + 1)])
    pts = np.vstack([loop] * (laps[1] - laps[0] + 1))
    inside = (ss > start_m) & (ss < end_m)

    def at(m):
        m = m - per * math.floor(m / per)
        return np.array([np.interp(m, s, closed[:, 0]), np.interp(m, s, closed[:, 1])])

    return np.vstack([at(start_m)[None], pts[inside], at(end_m)[None]])


def _ring_widths(ring: dict, width_factor: float) -> tuple[np.ndarray, list[dict]]:
    """Each segment of the ring's loop (from node k to k+1, in driving order): its way's width as mapped (with the
    job's width option); and each way's surface and look."""
    ways = ring.get("ways") or [{"tags": ring.get("tags") or {}}]
    looks = []
    for w in ways:
        tags = {**(w.get("tags") or {}), "highway": (w.get("tags") or {}).get("highway", ring.get("highway"))}
        surface = sec.surface_class(tags) or sec.SEALED
        unsealed = surface == sec.UNSEALED
        width, source = sec.width_from_tags(tags, unsealed=unsealed)
        looks.append({"surface": surface, "width_m": width * width_factor, "mapped_m": width, "source": source,
                      "style": sec.style_code(unsealed, sec.single_lane(tags, width, unsealed))})
    n = len(ring["ring"]) - (1 if ring.get("closed") else 0)
    seg_way = ring.get("segment_way") or [0] * n
    return np.array([looks[k]["width_m"] for k in seg_way], float), looks


@dataclass
class _Rows:
    """The ring's rows, a metre apart along its smoothed line, from one end's overlap with the road to the other's."""
    s: np.ndarray
    xy: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    half: np.ndarray
    nrm: np.ndarray = field(init=False)

    def __post_init__(self):
        self.nrm = cl.normals(self.heading)

    def edge(self, k, es):
        """Its edge on side `es` (+1 its left, -1 its right, looking along it) at row(s) k."""
        k = np.asarray(k)
        return self.xy[k] + (es * self.half[k])[..., None] * self.nrm[k]

    def reversed(self) -> "_Rows":
        return _Rows(self.s[-1] - self.s[::-1], self.xy[::-1].copy(), self.heading[::-1] + math.pi,
                     -self.curvature[::-1], self.half[::-1].copy())


# -- one end: where the ring leaves the road ------------------------------------------------------------------------

def _end(rows: _Rows, m: stubs._Mainline, near: stubs._Near, side: str, island_es: float,
         scale: float = 1.0) -> tuple[dict | None, str | None]:
    """Where the ring (its rows from deep in the road outwards) leaves the road on `side`: as a side road's stub does
    (`stubs._candidate`), a flare either side between the road's edge and the ring's, and the throat between them.
    `island_es` is the ring's island-side edge (+1 its left, -1 its right). Returns the end, or why there's none."""
    de = rows.s
    d0 = stubs._crossing(de, near.beyond(rows.xy, side), 1)
    if d0 is None:
        return None, "stays_on_road"
    corners = []
    for es in (1.0, -1.0):
        edge = rows.edge(np.arange(len(de)), es)
        g = near.beyond(edge, side)
        # Only this end's half of the ring: on a small ring the far end comes back onto the road within reach.
        within = np.flatnonzero(de <= min(d0 + CORNER_WITHIN_M, max(de[-1] / 2, d0 + 1.0)))
        k_min = int(within[np.argmin(g[within])])
        dc = stubs._crossing(de[:within[-1] + 1], g[:within[-1] + 1], k_min + 1)
        if dc is None:
            return None, "corner"
        pc = np.array([np.interp(dc, de, edge[:, 0]), np.interp(dc, de, edge[:, 1])])
        sc = float(near.project(pc)[0][0])
        corners.append({"es": es, "dc": dc, "sc": sc, "heading": float(np.interp(dc, de, rows.heading))})
    up, down = sorted(corners, key=lambda k: k["sc"])
    n_rows = len(de)
    sign_side = 1.0 if side == "L" else -1.0
    flares, first = {}, 0
    for name, cn, toward in (("upstream", up, -1.0), ("downstream", down, 1.0)):
        hm = float(np.interp(cn["sc"], m.s, m.heading))
        mdir = toward * np.array([math.cos(hm), math.sin(hm)])
        u = np.array([math.cos(cn["heading"]), math.sin(cn["heading"])])
        alpha = float(np.arccos(np.clip(u @ mdir, -1.0, 1.0)))
        half_tan = math.tan(alpha / 2)
        if half_tan < 1e-3:
            return None, "skew"
        radius = max(min(END_FLARE_RADIUS_M, stubs.FLARE_MAX_M * half_tan) * scale, stubs.FLARE_MIN_RADIUS_M)
        tangent = radius / half_tan
        if tangent > stubs.FLARE_MAX_TANGENT_M:
            return None, "skew"
        r_t1 = m.row(cn["sc"] + toward * tangent)
        r_t1 = int(math.floor(r_t1)) if toward < 0 else int(math.ceil(r_t1))
        if r_t1 < 2 or r_t1 > len(m.s) - 3:
            return None, "road_end"
        tangent = abs(float(m.s[r_t1]) - cn["sc"])
        corner_so = np.array([cn["sc"], np.interp(cn["sc"], m.s, m.half[side])])
        k_t2 = max(int(np.searchsorted(de, cn["dc"] + tangent + 1e-6)), 1)
        while k_t2 < n_rows - 1:
            s_e, o_e = near.project(rows.edge(k_t2, cn["es"])[None])
            if np.hypot(s_e[0] - corner_so[0], sign_side * o_e[0] - corner_so[1]) >= tangent + 0.25:
                break
            k_t2 += 1
        flares[name] = {"es": cn["es"], "corner": cn["sc"], "row": r_t1, "k": k_t2, "radius": radius,
                        "tangent": tangent, "angle": math.degrees(alpha), "island": cn["es"] == island_es}
        first = max(first, k_t2)
    if n_rows - 1 - first < 2:
        return None, "short"
    for f in flares.values():
        end = rows.edge(f["k"], f["es"])
        s_end, o_end = near.project(end[None])
        corner = np.array([f["corner"], np.interp(f["corner"], m.s, m.half[side])])
        so, f["radius"] = stubs._fillet(np.array([m.s[f["row"]], m.half[side][f["row"]]]), corner,
                                        np.array([s_end[0], sign_side * o_end[0]]))
        curve = m.to_xy(so[:, 0], sign_side * so[:, 1])
        curve[0], curve[-1] = m.edge(f["row"], side), end
        f["points"] = (np.vstack([curve, rows.edge(np.arange(f["k"] + 1, first + 1), f["es"])])
                       if f["k"] < first else curve)
        f["shape_ok"] = stubs._flare_shape_ok(curve, f["angle"])
    a, b = flares["upstream"]["row"], flares["downstream"]["row"]
    if b - a < 2:
        return None, "corner"
    across = [rows.xy[first] + o * rows.half[first] * rows.nrm[first] for o in (1.0, 0.5, 0.0, -0.5, -1.0)]
    if flares["downstream"]["es"] < 0:
        across = across[::-1]
    ring = ([m.edge(r, side) for r in range(a, b + 1)] + list(flares["downstream"]["points"][1:])
            + across[1:-1] + list(flares["upstream"]["points"][::-1][:-1]))
    throat = shapely.Polygon(ring)
    lo, hi = max(a - 20, 0), min(b + 20, len(m.s) - 1)
    road = shapely.Polygon([m.edge(r, "L") for r in range(lo, hi + 1)]
                           + [m.edge(r, "R") for r in range(hi, lo - 1, -1)]).buffer(0)
    k0 = int(np.searchsorted(de, d0))
    inside = [rows.xy[k] + t * rows.half[k] * rows.nrm[k] for k in range(max(k0, 1), first) for t in (0.0, 0.5, -0.5)]
    if (not throat.is_valid or throat.area < 1.0 or throat.intersection(road).area > 0.05
            or not all(f["shape_ok"] for f in flares.values())
            or not shapely.union(throat, road).buffer(0.01).contains(shapely.MultiPoint(inside or [rows.xy[first]]))):
        if scale > stubs.SHRINK ** 2 - 1e-9:
            return _end(rows, m, near, side, island_es, scale * stubs.SHRINK)
        return None, "flare"
    p0 = np.array([np.interp(d0, de, rows.xy[:, 0]), np.interp(d0, de, rows.xy[:, 1])])
    return {"d0": float(d0), "first": int(first), "mouth": (int(a), int(b)), "flares": flares, "throat": throat,
            "inside_from": max(k0, 1), "station_m": float(near.project(p0[None])[0][0])}, None


# -- the plan ---------------------------------------------------------------------------------------------------------

def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _fit_line(line: np.ndarray, near: stubs._Near, side: str, island_es: float, half: np.ndarray,
              hold: float) -> np.ndarray:
    """The ring's extended line moved sideways at each end so its island-side edge starts FIT_INSIDE_M inside the
    road's edge: by as much as that takes over its first and last `hold` metres (the extension along the route's arc),
    easing back onto the mapped ring over the next EASE_M."""
    s = cl.cumulative(line)
    d = np.gradient(line, axis=0)
    nl = cl.normals(np.arctan2(d[:, 1], d[:, 0]))
    sign_side = 1.0 if side == "L" else -1.0
    offset = np.zeros(len(line))
    for k, dist in ((0, s), (len(line) - 1, s[-1] - s)):
        edge = line[k] + island_es * half[k] * nl[k]
        sp, o = near.project(edge[None])
        target = sign_side * (float(np.interp(sp[0], near.m.s, near.m.half[side])) - FIT_INSIDE_M)
        # The road's left and the ring's left agree where the ring runs along the road at its ends.
        offset += (target - float(o[0])) * (1.0 - _smoothstep((dist - hold) / EASE_M))
    return line + nl * offset[:, None]


def _rows_for(ring: dict, near: stubs._Near, side: str, island_es: float,
              width_factor: float) -> tuple[_Rows, list[dict], np.ndarray] | tuple[None, str, None]:
    """The ring's rows from the route's arc, round the rest of the ring, and back onto the route's arc; each way's look,
    and the style at each row."""
    pts = np.asarray(ring["ring"], float)
    loop = pts[:-1] if ring.get("closed") else pts
    entry, leave = ring["route_ring_index"]
    seg = np.hypot(*np.diff(np.vstack([loop, loop[:1]]), axis=0).T)
    at = np.concatenate([[0.0], np.cumsum(seg)])
    per = float(at[-1])
    arc = (float(at[leave]) - float(at[entry])) % per  # the route's arc, joining to leaving
    if arc < 1.0 or per - arc < 2 * MIN_RIBBON_M:
        return None, "touches", None
    ext = min(EXTEND_M, EXTEND_SHARE * arc)
    start = float(at[leave]) - ext
    line = _loop_sample(loop, start, start + (per - arc) + 2 * ext)
    seg_w, looks = _ring_widths(ring, width_factor)
    seg_way = ring.get("segment_way") or [0] * len(seg_w)
    s_line = cl.cumulative(line)
    # Each point's segment of the loop (the midpoint between two points decides, so a node takes the segment after it).
    mid = (start + np.concatenate([(s_line[:-1] + s_line[1:]) / 2, s_line[-1:]])) % per
    k_seg = np.clip(np.searchsorted(at, mid, side="right") - 1, 0, len(seg_w) - 1)
    line = _fit_line(line, near, side, island_es, seg_w[k_seg] / 2, ext)
    try:
        sm = cl.smooth(line)
    except ValueError:
        return None, "short", None
    pick = np.clip(np.searchsorted(s_line, sm.s, side="right") - 1, 0, len(k_seg) - 1)
    width = sec.eased(sm.s, seg_w[k_seg][pick], WIDTH_EASE_M)
    style = np.array([looks[seg_way[k]]["style"] for k in k_seg[pick]], np.int8)
    return _Rows(sm.s, sm.xy, sm.heading, sm.curvature, width / 2), looks, style


def _record_rows(rows: _Rows, k0: int, k1: int, z, zl, zr, style) -> dict:
    sl = slice(k0, k1 + 1)
    return {"d": np.round(rows.s[sl] - rows.s[k0], 3).tolist(), "xy": np.round(rows.xy[sl], 2).tolist(),
            "z": np.round(z[sl], 3).tolist(), "z_left": np.round(zl[sl], 3).tolist(),
            "z_right": np.round(zr[sl], 3).tolist(), "heading": np.round(rows.heading[sl], 5).tolist(),
            "curvature": np.round(rows.curvature[sl], 5).tolist(), "grade": np.round(np.gradient(z[sl]), 4).tolist(),
            "half_m": np.round(rows.half[sl], 3).tolist(), "style": [sec.STYLES[int(v)] for v in style[sl]]}


def _flare_record(f: dict, heights: np.ndarray) -> dict:
    return {"stub_side": "left" if f["es"] > 0 else "right", "island": bool(f["island"]),
            "radius_m": round(f["radius"], 2), "tangent_m": round(f["tangent"], 2), "angle_deg": round(f["angle"], 1),
            "mainline_row": f["row"], "points": [[round(float(p[0]), 2), round(float(p[1]), 2), round(float(h), 3)]
                                                 for p, h in zip(f["points"], heights)]}


def plan(rings: list[dict], c, section, dem_sample, *, width_factor: float = 1.0,
         layouts: tuple[bool, ...] = (False,), mouths_avoid=None) -> dict:
    """The roundabouts to draw, for the rings junctions.json lists: the content of roundabouts.json (without `crs`).
    `c` and `section` are the centreline's and cross-section's arrays, `dem_sample` the region's elevation, `layouts`
    whether each layout is reversed (for the timing gates)."""
    m = stubs._Mainline(c, section)
    out, skipped = [], []
    for ring in rings:
        rec, why = _plan_ring(ring, m, c, section, dem_sample, width_factor, layouts)
        if rec is None:
            skipped.append({"roundabout": ring["id"], "name": ring.get("name"), "reason": why})
        else:
            out.append(rec)
    return {"version": FORMAT_VERSION,
            "rule": {"extend_m": EXTEND_M, "fit_inside_m": FIT_INSIDE_M, "ease_m": EASE_M,
                     "island_kerb_h": ISLAND_KERB_H, "island_kerb_w": ISLAND_KERB_W, "dome_grade": DOME_GRADE,
                     "dome_max_m": DOME_MAX_M, "trees_inner_radius_m": TREES_INNER_RADIUS_M,
                     "width_factor": width_factor},
            "roundabouts": out, "skipped": skipped}


def _plan_ring(ring: dict, m: stubs._Mainline, c, section, dem_sample, width_factor: float,
               layouts: tuple[bool, ...]) -> tuple[dict | None, str | None]:
    if not ring.get("on_route") or not ring.get("closed") or ring.get("route_ring_index") is None:
        return None, "not_driven"
    if not ring.get("oneway"):
        return None, "not_one_way"
    centre = np.asarray(ring["centre"], float)
    # The island is on the right of a ring driven clockwise, which is the route's right there.
    island_es = -1.0 if ring.get("clockwise", True) else 1.0
    side = "R" if island_es < 0 else "L"
    at = jn._station(m.xy, m.s, centre, float(ring.get("route_from_m") or 0.0))
    reach = 2 * float(ring["radius_m"]) + 2 * EXTEND_M + stubs.MAIN_REACH_M
    near = m.near(at, reach)
    rows, looks, style = _rows_for(ring, near, side, island_es, width_factor)
    if rows is None:
        return None, looks
    # Each end, from deep in the road outwards: the leaving end forwards, the joining end backwards.
    end1, why = _end(rows, m, near, side, island_es)
    if end1 is None:
        return None, f"leaving_{why}"
    back = rows.reversed()
    end2, why = _end(back, m, near, side, -island_es)
    if end2 is None:
        return None, f"joining_{why}"
    n = len(rows.s)
    k0, k1 = end1["first"], n - 1 - end2["first"]
    if rows.s[k1] - rows.s[k0] < MIN_RIBBON_M:
        return None, "short"
    # The mouths, in the road's stations: the joining end's is the earlier.
    a2, b2 = end2["mouth"]
    a1, b1 = end1["mouth"]
    if not b2 < a1:
        return None, "ends_overlap"

    # Heights: the ring's own profile and crown, blended into the road's cross-fall near the road.
    unsealed = sec.is_unsealed(style)
    crown = np.where(unsealed, sec.CROWN_UNSEALED, sec.CROWN)
    own = dem_sample(rows.xy[:, 0], rows.xy[:, 1])
    ok = np.isfinite(own)
    if not ok.any():
        return None, "no_dem"
    own = np.interp(np.arange(n), np.flatnonzero(ok), own[ok])
    own = uniform_filter1d(own, max(int(round(DEM_SMOOTH_M)), 1), mode="nearest")
    rise = max(abs(float(own[k] - near.surface(rows.xy[k:k + 1], side)[0])) for k in (k0, k1))
    blend = float(np.clip(max(stubs.BLEND_M, rise / stubs.MAX_BLEND_GRADE), stubs.BLEND_M,
                          max(stubs.BLEND_M, 0.4 * (rows.s[k1] - rows.s[k0]))))
    row_tree = cKDTree(rows.xy)

    def height(p):
        p = np.atleast_2d(np.asarray(p, float))
        _, k = row_tree.query(p)
        t = np.column_stack([np.cos(rows.heading[k]), np.sin(rows.heading[k])])
        rel = p - rows.xy[k]
        d = np.clip(k + (rel * t).sum(axis=1), 0.0, n - 1)
        lateral = (rel * rows.nrm[k]).sum(axis=1)
        ring_h = np.interp(d, np.arange(n), own) - np.interp(d, np.arange(n), crown) * np.abs(lateral)
        main_h = near.surface(p, side)
        wgt = _smoothstep(near.beyond(p, side) / blend)
        return main_h + wgt * (ring_h - main_h)

    idx = np.arange(n)
    z, zl, zr = height(rows.xy), height(rows.edge(idx, 1.0)), height(rows.edge(idx, -1.0))
    # Each flare ends on the ring's edge at that end's first full row: its left or right looking along that end
    # (the joining end looks back along the ring, so its left is the ring's right).
    for e, k, left, right in ((end1, k0, zl, zr), (end2, k1, zr, zl)):
        for f in e["flares"].values():
            h = height(f["points"])
            h[0] = m.edge_h(f["row"], side)
            h[-1] = left[k] if f["es"] > 0 else right[k]
            f["heights"] = h

    # The island: the road's island-side edge between the two ends, the island-side flares, the ring's inner edge.
    f1 = next(f for f in end1["flares"].values() if f["island"])
    f2 = next(f for f in end2["flares"].values() if f["island"])
    r_from, r_to = f2["row"], f1["row"]
    if r_to - r_from < 2:
        return None, "island"
    road_edge = [m.edge(r, side) for r in range(r_from, r_to + 1)]
    inner = rows.edge(np.arange(k0, k1 + 1), island_es)
    boundary = np.vstack([road_edge, f1["points"][1:-1], inner, f2["points"][::-1][1:-1]])
    island = shapely.Polygon(boundary)
    if not island.is_valid or island.area < 4.0:
        return None, "island"
    paved = shapely.unary_union([end1["throat"].buffer(0), end2["throat"].buffer(0),
                                 shapely.Polygon(np.vstack([rows.edge(np.arange(k0, k1 + 1), 1.0),
                                                            rows.edge(np.arange(k1, k0 - 1, -1), -1.0)])).buffer(0)])
    if paved.geom_type != "Polygon":
        paved = paved.buffer(stubs.OUTLINE_CLOSE_M, join_style="mitre").buffer(-stubs.OUTLINE_CLOSE_M,
                                                                                join_style="mitre")
    if paved.geom_type != "Polygon":
        return None, "outline"
    inner_r = float(np.hypot(*(inner - centre).T).mean())

    def r2(a):
        return np.round(np.asarray(a, float), 2).tolist()

    rec = {
        "id": ring["id"], "name": ring.get("name"), "label": label(ring), "ref": (ring.get("tags") or {}).get("ref"),
        "highway": ring.get("highway"), "way_ids": ring.get("way_ids"),
        "radius_m": ring["radius_m"], "clockwise": bool(ring.get("clockwise", True)),
        "island_side": "left" if side == "L" else "right", "centre": r2(centre),
        "surface": looks[0]["surface"], "width_m": [round(float(rows.half[k0:k1 + 1].min() * 2), 2),
                                                    round(float(rows.half[k0:k1 + 1].max() * 2), 2)],
        "blend_m": round(blend, 1), "length_m": round(float(rows.s[k1] - rows.s[k0]), 1),
        "rows": _record_rows(rows, k0, k1, z, zl, zr, style),
        "mouth": {"from_m": round(float(m.s[a2]), 2), "to_m": round(float(m.s[b1]), 2), "rows": [int(a2), int(b1)]},
        "ends": [_end_record(end1, rows, z, zl, zr, "leaving", side, m),
                 _end_record(end2, back, z[::-1], zr[::-1], zl[::-1], "joining", side, m)],
        "island": {"outline": r2(boundary), "area_m2": round(float(island.area), 1), "inner_radius_m": round(inner_r, 1),
                   "trees": inner_r >= TREES_INNER_RADIUS_M, "mainline_rows": [int(r_from), int(r_to)],
                   "kerb_h": ISLAND_KERB_H, "kerb_w": ISLAND_KERB_W},
        "outline": r2(np.asarray(paved.exterior.coords)[:-1]),
    }
    # Its exits off the rest of the ring, closed as side roads are.
    ring_c, ring_sec = ribbon_arrays(rec)
    exits, exit_skipped = _exit_junctions(ring, rec, ring_c)
    rec["exits"] = []
    rec["exits_skipped"] = exit_skipped
    if exits:
        # They keep off the road the route drives, its verge and a little more, as a stub keeps off another part of
        # the route (`stubs.CLEAR_M`).
        ks = np.arange(int(near.idx.min()), int(near.idx.max()) + 1)
        left = np.array([m.edge(k, "L") for k in ks])
        right = np.array([m.edge(k, "R") for k in ks])
        road = shapely.union_all(shapely.make_valid(shapely.polygons(
            np.stack([left[:-1], left[1:], right[1:], right[:-1]], axis=1))))
        keep_off = road.buffer(mesh.VERGE_M + stubs.CLEAR_M)
        planned = None
        for scale in EXIT_FLARE_SCALES:
            attempt = stubs.plan(exits, ring_c, ring_sec, dem_sample, width_factor=width_factor, layouts=(),
                                 flare_scale=scale, avoid=keep_off)
            if planned is None or len(attempt["stubs"]) > len(planned["stubs"]):
                planned = attempt
        rec["exits"] = planned["stubs"]
        for st in rec["exits"]:
            st["parent"] = f"ring:{ring['id']}"
        rec["exits_skipped"] += [{**sk, "reason": "near_route" if sk["reason"] == "roundabout" else sk["reason"]}
                                 for sk in planned["skipped"]]
        rec["exit_junctions"] = [j for j in exits if j["id"] in {st["junction"] for st in rec["exits"]}]
    else:
        rec["exit_junctions"] = []
    return rec, None


def label(ring: dict) -> str:
    """How a report names a ring: its name, else its ref, else the roads that leave it."""
    if ring.get("name"):
        return str(ring["name"])
    ref = (ring.get("tags") or {}).get("ref") or ring.get("ref")
    if ref:
        return f"the {ref} roundabout"
    roads = sorted({e["name"] for e in ring.get("exits") or [] if e.get("name")})
    return f"the roundabout at {' and '.join(roads)}" if roads else "an unnamed roundabout"


def _end_record(e: dict, rows: _Rows, z, zl, zr, which: str, side: str, m: stubs._Mainline) -> dict:
    """An end as the mesh stage and the checks read it, shaped as a side road's stub is (`core.stubs`): its rows from
    just inside the road to the ring's first full row (`ribbon_from_m`, a row index here), looking along that end, so
    the validator checks its seams and cover as it checks a stub's (`validate.stub_check`)."""
    a, b = e["mouth"]
    first = e["first"]
    k_lo = max(e["inside_from"] - 1, 0)
    sl = slice(k_lo, first + 1)
    rows_rec = {"d": np.round(rows.s[sl] - rows.s[k_lo], 3).tolist(), "xy": np.round(rows.xy[sl], 2).tolist(),
                "z": np.round(z[sl], 3).tolist(), "z_left": np.round(zl[sl], 3).tolist(),
                "z_right": np.round(zr[sl], 3).tolist(), "heading": np.round(rows.heading[sl], 5).tolist(),
                "half_m": np.round(rows.half[sl], 3).tolist()}
    outline = e["throat"].buffer(0)
    return {"end": which, "side": "left" if side == "L" else "right", "station_m": round(e["station_m"], 2),
            "ribbon_from_m": int(first - k_lo),
            "mouth": {"from_m": round(float(m.s[a]), 2), "to_m": round(float(m.s[b]), 2), "rows": [int(a), int(b)]},
            "flares": {name: _flare_record(f, f["heights"]) for name, f in e["flares"].items()},
            "rows": rows_rec,
            "outline": np.round(np.asarray(outline.exterior.coords)[:-1], 2).tolist()}


def ribbon_arrays(rec: dict) -> tuple[dict, dict]:
    """The ring's own ribbon as a road's centreline and cross-section arrays (`centreline.npz`, `section.npz`), for
    the exits' stubs (`stubs.plan`, `stubs.build`) and their closures: rows a metre apart, crown, no camber."""
    r = rec["rows"]
    xy = np.asarray(r["xy"], float)
    z, zl, zr = (np.asarray(r[k], float) for k in ("z", "z_left", "z_right"))
    half = np.asarray(r["half_m"], float)
    d = np.asarray(r["d"], float)
    style = np.array([sec.STYLES.index(v) for v in r["style"]], np.int8)
    c = {"s": d, "xy": xy, "heading": np.asarray(r["heading"], float), "z": z,
         "curvature": np.asarray(r["curvature"], float), "grade": np.asarray(r["grade"], float)}
    section = {"half_left": half, "half_right": half, "slope_left": (z - zl) / half, "slope_right": (z - zr) / half,
               "style": style}
    return c, section


def _exit_junctions(ring: dict, rec: dict, ring_c: dict) -> tuple[list[dict], list[dict]]:
    """The exits off the rest of the ring as junctions.json's junctions, leaving the ring's ribbon: each a side road
    the route doesn't use. One that meets the route again is the other carriageway of a road the route uses, and
    isn't drawn; two that become one road within their first 100 m are one, down the middle, at that road's width."""
    xy = np.asarray(ring_c["xy"], float)
    s = np.asarray(ring_c["s"], float)
    hd = np.asarray(ring_c["heading"], float)
    out, skipped, pending = [], [], []
    for k, e in enumerate(ring.get("exits") or []):
        if e.get("on_route"):
            continue  # leaves where the route is on the ring: a side road of the route's (core.junctions)
        jid = EXIT_ID_BASE + 100 * int(ring["id"]) + k
        if e.get("ends") == "meets_route":
            skipped.append({"junction": jid, "name": e.get("name"), "reason": "route_carriageway"})
            continue
        pending.append((jid, e))
    used = set()
    for i, (jid, e) in enumerate(pending):
        if jid in used:
            continue
        mate = None
        for jid2, e2 in pending[i + 1:]:
            if jid2 not in used and e2["far_way"]["way_id"] == e["far_way"]["way_id"] and \
                    e2["far_way"]["way_id"] not in (e["way_id"], e2["way_id"]):
                mate = (jid2, e2)
                break
        line = np.asarray(e["line"], float)
        record = dict(e)
        if mate is not None:
            used.add(mate[0])
            other = np.asarray(mate[1]["line"], float)
            n_pts = max(int(min(cl.cumulative(line)[-1], cl.cumulative(other)[-1])), 2)
            a, _, _ = cl.resample(line, cl.cumulative(line)[-1] / n_pts)
            b, _, _ = cl.resample(other, cl.cumulative(other)[-1] / n_pts)
            k_min = min(len(a), len(b))
            line = (a[:k_min] + b[:k_min]) / 2
            far = e["far_way"]
            record.update({"line": np.round(line, 2).tolist(), "tags": far["tags"], "highway": far["highway"],
                           "width_m": far["width_m"], "surface_class": far["surface_class"],
                           "length_m": round(float(cl.cumulative(line)[-1]), 1), "paired_with": mate[1]["way_id"]})
            skipped.append({"junction": mate[0], "name": mate[1].get("name"), "reason": "paired",
                            "with": jid})
        p0 = line[0]
        kk = int(np.argmin(np.hypot(*(xy - p0).T)))
        u = np.array([math.cos(hd[kk]), math.sin(hd[kk])])
        direction = jn._along(line, jn.ANGLE_AT_M) - p0
        angle = jn._wrap(jn._heading(direction) - jn._heading(u))
        out.append({**record, "id": jid, "at_m": round(float(s[kk]), 1), "xy": np.round(p0, 2).tolist(),
                    "side": "left" if angle > 0 else "right", "angle_deg": round(angle, 1),
                    "off_mainline_deg": round(min(abs(angle), 180 - abs(angle)), 1), "route_turn_deg": 0.0,
                    "closure": jn.closure_level({**(record.get("tags") or {}), "highway": record.get("highway")}),
                    "roundabout": ring["id"]})
    return out, skipped


# -- reading the plan -------------------------------------------------------------------------------------------------

def exit_stubs(rings: list[dict]) -> list[dict]:
    """Every ring's exits' stubs (stubs.json's shape, each with `parent`: the ring it leaves)."""
    return [st for r in rings for st in r.get("exits") or []]


def exit_junctions(rings: list[dict]) -> list[dict]:
    """Every ring's exits as junctions.json's junctions (`core.junctions`), for the closures and the checks."""
    return [j for r in rings for j in r.get("exit_junctions") or []]


def parents(rings: list[dict]) -> dict[str, tuple[dict, dict]]:
    """Each ring's ribbon as a road's arrays (`ribbon_arrays`), by the `parent` its exits' stubs carry."""
    return {f"ring:{r['id']}": ribbon_arrays(r) for r in rings}


def centres(rings: list[dict]) -> np.ndarray:
    """Every ring's and its exits' centreline points: other roads' skirts stop short of them (`stubs.side_caps`)."""
    pts = [np.asarray(r["rows"]["xy"], float) for r in rings]
    pts += [stubs._centre(st) for st in exit_stubs(rings)]
    return np.vstack(pts) if pts else np.zeros((0, 2))


def paved(rings: list[dict]) -> shapely.Geometry:
    """The rings' paved outlines (their ribbons and the throats where they meet the road), as one geometry."""
    return shapely.unary_union([shapely.Polygon(r["outline"]).buffer(0) for r in rings]) if rings else shapely.Polygon()


def avoid(rings: list[dict]) -> shapely.Geometry | None:
    """The ground a side road's stub keeps off (`stubs.plan(avoid=)`): each ring's paved outline, its island and its
    exits' stubs. None with no rings."""
    if not rings:
        return None
    parts = [shapely.Polygon(r["outline"]).buffer(0) for r in rings]
    parts += [shapely.Polygon(r["island"]["outline"]).buffer(0) for r in rings]
    parts += [shapely.Polygon(st["outline"]).buffer(mesh.VERGE_M) for st in exit_stubs(rings)]
    return shapely.unary_union(parts)


def zone(rings: list[dict], beyond_m: float) -> shapely.Geometry | None:
    """The ground to keep clear of trees, buildings or power-line supports: `beyond_m` beyond the rings' paved outlines
    and their exits' stubs; and each island, all of it, or on a big one (`island.trees`) its edge out to `beyond_m` in
    from the kerb."""
    if not rings:
        return None
    parts = [shapely.Polygon(r["outline"]).buffer(beyond_m) for r in rings]
    parts += [shapely.Polygon(st["outline"]).buffer(beyond_m) for st in exit_stubs(rings)]
    for r in rings:
        island = shapely.Polygon(r["island"]["outline"]).buffer(0)
        parts.append(island.difference(island.buffer(-beyond_m)) if r["island"]["trees"] else island)
    return shapely.unary_union(parts)


def masks(rings: list[dict], s: np.ndarray, margin_m: float = jn.MOUTH_MARGIN_M) -> dict[str, np.ndarray]:
    """Which centreline stations a ring takes on the road's island side, from one end's mouth to the other's, and
    `margin_m` either side: no guard rail, guide post, curve marker or sign stands there."""
    s = np.asarray(s, float)
    out = {"L": np.zeros(len(s), bool), "R": np.zeros(len(s), bool)}
    for r in rings:
        side = "L" if r["island_side"] == "left" else "R"
        out[side] |= (s >= r["mouth"]["from_m"] - margin_m) & (s <= r["mouth"]["to_m"] + margin_m)
    return out


def cam_mask(rings: list[dict], s: np.ndarray) -> np.ndarray:
    """The stations of the route's arc of each ring, from one end's mouth to the other's: no curve alignment marker
    stands there on either side (`furniture.furniture(no_cams=)`). Nothing on a ring points the way to go (Q15)."""
    s = np.asarray(s, float)
    out = np.zeros(len(s), bool)
    for r in rings:
        out |= (s >= r["mouth"]["from_m"]) & (s <= r["mouth"]["to_m"])
    return out


def layout_intervals(rings: list[dict], s: np.ndarray, reverse: bool,
                     margin_m: float = jn.MOUTH_MARGIN_M) -> list[tuple[float, float]]:
    """The rings' stretches of the road as (from, to) metres from a layout's start: no timing gate stands in one."""
    s0, total = float(s[0]), float(s[-1] - s[0])
    spans = [(r["mouth"]["from_m"] - margin_m - s0, r["mouth"]["to_m"] + margin_m - s0) for r in rings]
    if reverse:
        spans = [(total - b, total - a) for a, b in spans]
    return sorted(spans)


# -- the meshes -------------------------------------------------------------------------------------------------------

@dataclass
class Built:
    """The rings' meshes, and what the road and the terrain need to make room for them."""
    meshes: list[Mesh]
    cut: dict[str, np.ndarray]  # the road's verge and skirt quads taken out, per side
    edge_break: dict[str, np.ndarray]  # the road's rows whose edge line breaks, per side
    ribbons: list[mesh.Ribbon]
    ribbon_cuts: list
    open_last: list[bool]
    patches: list[np.ndarray]
    edges: list[tuple[np.ndarray, np.ndarray]]
    interior: set
    stats: dict


def _strip(rr: mesh.Ribbon, island: str) -> mesh.Ribbon:
    """A ring's ribbon less its verge and skirt on the island's side: the island meets its road's edge there."""
    rc = rr.road_cols
    if island == "R":
        keep = slice(0, rc.stop)
        return mesh.Ribbon(rr.e[:, keep], rr.n[:, keep], rr.h[:, keep], rc, rr.left_cols, slice(rc.stop - 1, rc.stop))
    off = rc.start
    keep = slice(off, rr.e.shape[1])
    return mesh.Ribbon(rr.e[:, keep], rr.n[:, keep], rr.h[:, keep], slice(0, 5), slice(0, 1),
                       slice(rr.right_cols.start - off, rr.right_cols.stop - off))


def _idw(points: np.ndarray, heights: np.ndarray, at: np.ndarray, power: float = 2.0) -> np.ndarray:
    """Heights at `at` interpolated from scattered `points` by inverse distance (the island's base, from its edge)."""
    if not len(at):
        return np.zeros(0)
    d = np.hypot(at[:, None, 0] - points[None, :, 0], at[:, None, 1] - points[None, :, 1])
    w = 1.0 / np.maximum(d, 1e-3) ** power
    return (w * heights[None, :]).sum(axis=1) / w.sum(axis=1)


def _cdt(outer: np.ndarray, holes: list[np.ndarray]) -> np.ndarray:
    """Triangles over a polygon with holes, using only its own vertices (outer, then each hole's, in order)."""
    pts = np.vstack([outer] + holes)
    index = {(float(x), float(y)): i for i, (x, y) in enumerate(pts.tolist())}
    poly = shapely.Polygon(outer, holes)
    out = []
    for g in shapely.get_parts(shapely.constrained_delaunay_triangles(poly)):
        corners = np.asarray(g.exterior.coords)[:3]
        tri = [index.get((float(x), float(y))) for x, y in corners.tolist()]
        if None not in tri:
            out.append(tri)
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def _densify(points: np.ndarray, step: float = 1.0) -> np.ndarray:
    """A closed ring's points (not repeated), no two further apart than `step`."""
    closed = np.vstack([points, points[:1]])
    out = []
    for a, b in zip(closed[:-1], closed[1:]):
        k = max(int(math.ceil(float(np.hypot(*(b - a))) / step)), 1)
        out.append(a + (b - a) * (np.arange(k) / k)[:, None])
    return np.vstack(out)


def _island(boundary: np.ndarray, heights: np.ndarray, frame: LocalFrame):
    """The island's kerb (a band ISLAND_KERB_W wide, rising ISLAND_KERB_H from the road's edge) and its grass inside,
    domed gently: (kerb piece, grass pieces), each (positions, uv, triangles) in region axes, the grass sharing the
    kerb's top edge exactly. (None, None) if there's no room inside the kerb."""
    poly = shapely.Polygon(boundary)
    inner = poly.buffer(-ISLAND_KERB_W, join_style="mitre", mitre_limit=2.0)
    parts = [p for p in sorted(shapely.get_parts(inner), key=lambda g: -g.area) if p.area > 1.0] \
        if not inner.is_empty else []
    if not parts:
        return None, None
    rings = [_densify(np.asarray(p.exterior.coords)[:-1], 1.0) for p in parts]
    tops = [_idw(boundary, heights, r) + ISLAND_KERB_H for r in rings]
    tri = _cdt(boundary, rings)
    if not len(tri):
        return None, None
    kerb_xy = np.vstack([boundary] + rings)
    kerb_h = np.concatenate([heights] + tops)
    kerb_uv = frame.to_ac(kerb_xy[:, 0], kerb_xy[:, 1], kerb_h)[:, [0, 2]] / KERB_TILE_M
    kerb = (np.column_stack([kerb_xy, kerb_h]), kerb_uv, tri)
    grass = []
    for p, ring, top in zip(parts, rings, tops):
        minx, miny, maxx, maxy = p.bounds
        gx, gy = np.meshgrid(np.arange(minx, maxx, ISLAND_LATTICE_M), np.arange(miny, maxy, ISLAND_LATTICE_M))
        lattice = np.column_stack([gx.ravel(), gy.ravel()])
        shapely.prepare(p)
        lattice = lattice[shapely.contains_xy(p, lattice[:, 0], lattice[:, 1])]
        dist = cKDTree(ring).query(lattice)[0] if len(lattice) else np.zeros(0)
        lattice, dist = lattice[dist >= ISLAND_POINT_CLEAR_M], dist[dist >= ISLAND_POINT_CLEAR_M]
        h_lat = _idw(boundary, heights, lattice) + ISLAND_KERB_H + np.minimum(DOME_GRADE * dist, DOME_MAX_M)
        t = stubs._triangulate(ring, lattice.reshape(-1, 2))
        xy = np.vstack([ring, lattice]) if len(lattice) else ring
        h = np.concatenate([top, h_lat])
        grass.append((np.column_stack([xy, h]), frame.to_ac(xy[:, 0], xy[:, 1], h)[:, [0, 2]] / 6.0, t))
    return kerb, grass


def _throat(end: dict, rb: mesh.Ribbon, side: str, rr: mesh.Ribbon, row: int, cols: list[int], material: str):
    """An end's throat, as a side road's (`stubs.build`): the road's edge vertices across its mouth, its flares, the
    ring's ribbon's row where the ribbon starts (`row`; `cols`, its road's columns from the end's left to its right,
    looking along the end), and the end's own cross-section inside. Returns (piece, boundary ring, its two flares as
    (es, road row, points, heights, island side)), or None."""
    a, b = end["mouth"]["rows"]
    edge_col = rb.road_cols.start if side == "L" else rb.road_cols.stop - 1
    main_edge = np.column_stack([rb.e[a:b + 1, edge_col], rb.n[a:b + 1, edge_col]])
    main_edge_h = rb.h[a:b + 1, edge_col]
    flares = {}
    for name in ("upstream", "downstream"):
        f = end["flares"][name]
        es = 1.0 if f["stub_side"] == "left" else -1.0
        pts = np.asarray(f["points"], float)
        fxy, fh = pts[:, :2].copy(), pts[:, 2].copy()
        mrow = a if name == "upstream" else b
        fxy[0], fh[0] = (rb.e[mrow, edge_col], rb.n[mrow, edge_col]), rb.h[mrow, edge_col]
        cc = cols[0] if es > 0 else cols[-1]
        fxy[-1], fh[-1] = (rr.e[row, cc], rr.n[row, cc]), rr.h[row, cc]
        fxy, fh = stubs._dedupe(fxy, fh)
        flares[name] = (es, mrow, fxy, fh, bool(f["island"]))
    across = np.array([(rr.e[row, cc], rr.n[row, cc], rr.h[row, cc]) for cc in cols])
    if flares["downstream"][0] < 0:
        across = across[::-1]
    down, up = flares["downstream"], flares["upstream"]
    ring = np.vstack([main_edge, down[2][1:], across[1:-1, :2], up[2][::-1][:-1]])
    ring_h = np.concatenate([main_edge_h, down[3][1:], across[1:-1, 2], up[3][::-1][:-1]])
    poly = shapely.Polygon(ring)
    if not poly.is_valid:
        return None
    shapely.prepare(poly)
    boundary = poly.exterior
    r = end["rows"]
    xy = np.asarray(r["xy"], float)
    z, zl, zr = (np.asarray(r[k], float) for k in ("z", "z_left", "z_right"))
    hd, half = np.asarray(r["heading"], float), np.asarray(r["half_m"], float)
    k_first = int(end["ribbon_from_m"])
    inner, inner_h = [], []
    for kk in range(1, k_first):
        nrm = cl.normals(np.array([hd[kk]]))[0]
        for t in (1.0, 0.5, 0.0, -0.5, -1.0):
            p = xy[kk] + nrm * t * half[kk]
            if poly.contains(shapely.Point(p)) and boundary.distance(shapely.Point(p)) >= stubs.STEINER_CLEAR_M:
                inner.append(p)
                inner_h.append(z[kk] + ((zl[kk] if t > 0 else zr[kk]) - z[kk]) * abs(t))
    inner = np.array(inner).reshape(-1, 2)
    tri = stubs._triangulate(ring, inner)
    pts = np.vstack([ring, inner])
    hts = np.concatenate([ring_h, np.asarray(inner_h, float)])
    u0, u1 = stubs.THROAT_U[material]
    band = (u1 - u0) * stubs.TEXTURE_WIDTH_M[material]
    tile = textures.GRAVEL_TILE_M if sec.is_unsealed(sec.STYLES.index(material)) else textures.ROAD_TILE_M
    o, h0 = xy[k_first], float(hd[k_first])
    along = (pts - o) @ np.array([math.cos(h0), math.sin(h0)])
    lateral = (pts - o) @ np.array([-math.sin(h0), math.cos(h0)])
    uv = np.column_stack([u0 + (u1 - u0) * stubs._tri_wave(lateral / band), along / tile])
    return (np.column_stack([pts, hts]), uv, tri), ring, (up, down)


def _flare_verge(rb: mesh.Ribbon, side: str, rr: mesh.Ribbon, row: int, verge_col: int, flare):
    """A mown verge along an outer flare, from the road's verge vertex at its tangent row to the ring's (`verge_col`)
    at its row, as a side road's (`stubs.build`). Returns (piece, patch ring, outer edge points, their heights)."""
    _, mrow, fxy, fh, _ = flare
    edge_col = rb.road_cols.start if side == "L" else rb.road_cols.stop - 1
    main_verge = edge_col - 1 if side == "L" else edge_col + 1
    outer_main = np.array([rb.e[mrow, main_verge], rb.n[mrow, main_verge]])
    outer_ring = np.array([rr.e[row, verge_col], rr.n[row, verge_col]])
    w0 = float(np.hypot(*(outer_main - fxy[0])))
    w1 = float(np.hypot(*(outer_ring - fxy[-1])))
    tang = np.gradient(fxy, axis=0)
    tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
    perp = np.column_stack([-tang[:, 1], tang[:, 0]])
    if perp[0] @ (outer_main - fxy[0]) < 0:
        perp = -perp
    frac = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(fxy, axis=0).T))])
    frac /= max(frac[-1], 1e-9)
    outer = fxy + perp * (w0 + (w1 - w0) * frac)[:, None]
    outer_h = fh - mesh.VERGE_M * mesh.VERGE_FALL
    outer[0], outer_h[0] = outer_main, rb.h[mrow, main_verge]
    outer[-1], outer_h[-1] = outer_ring, rr.h[row, verge_col]
    strip_xy = np.stack([fxy, outer], axis=1)
    strip_h = np.stack([fh, outer_h], axis=1)
    piece = (np.column_stack([strip_xy.reshape(-1, 2), strip_h.ravel()]), None, mesh._grid_triangles(len(fxy), 2))
    return piece, np.vstack([fxy, outer[::-1]]), outer, outer_h


def build(rings: list[dict], rb: mesh.Ribbon, s: np.ndarray, frame: LocalFrame, dem_sample, *, skirt: bool,
          others: np.ndarray) -> Built:
    """The rings' meshes from the plan (`plan`): each ring's ribbon, throats, outer flares' verges, island and kerb, and
    its exits' stubs; the road's cut and edge break; and the pieces the terrain leaves room for. `rb` is the road's
    ribbon, `s` its stations, `others` every other centreline point the rings' skirts stop short of (the road's and
    its stubs'). With `skirt` False (ground classes) the ring's verge mesh is the 2 m verge alone, and its skirt goes
    with the terrain, as the road's does."""
    rows = len(s)
    cut = {"L": np.zeros(rows - 1, bool), "R": np.zeros(rows - 1, bool)}
    edge_break = {"L": np.zeros(rows, bool), "R": np.zeros(rows, bool)}
    meshes: list[Mesh] = []
    ribbons, ribbon_cuts, open_last, patches, edges = [], [], [], [], []
    interior: set = set()
    others = np.asarray(others, float).reshape(-1, 2)
    for r in rings:
        side = "L" if r["island_side"] == "left" else "R"
        outer = "R" if side == "L" else "L"
        a, b = r["mouth"]["rows"]
        cut[side][a:b] = True
        edge_break[side][a + 1:b] = True
        c_ring, sec_ring = ribbon_arrays(r)
        xy, hd = c_ring["xy"], c_ring["heading"]
        rest = centres([q for q in rings if q is not r])
        ex = r.get("exits") or []
        exit_pts = np.vstack([stubs._centre(st) for st in ex]) if ex else np.zeros((0, 2))
        cap_l, cap_r = stubs.side_caps(xy, cl.normals(hd), np.vstack([others, rest, exit_pts]), stubs.REACH_M)
        half = sec_ring["half_left"]
        full = mesh.build_ribbon(xy, c_ring["z"], hd, c_ring["curvature"], half, half, sec_ring["slope_left"],
                                 sec_ring["slope_right"], dem_sample, skirt_cap_l=cap_l, skirt_cap_r=cap_r)
        rr = _strip(full, side)
        n_rows = rr.e.shape[0]
        # Its exits, leaving the ring as side roads leave the road.
        ex_built = (stubs.build(ex, rr, c_ring["s"], frame, dem_sample, skirt=skirt,
                                main_xy=np.vstack([xy, others, rest]), name_tag=f"R{r['id']}_") if ex else None)
        ring_cut = ex_built.cut if ex_built is not None else {"L": np.zeros(n_rows - 1, bool),
                                                              "R": np.zeros(n_rows - 1, bool)}
        eb = {"L": np.zeros(n_rows, bool), "R": np.zeros(n_rows, bool)}
        if ex_built is not None:
            eb[outer] |= ex_built.edge_break[outer]
        eb[side][:] = True  # no edge line on the island's side
        style = sec_ring["style"]
        meshes += mesh.ribbon_meshes(rr, c_ring["s"], frame, style=style, skirt=skirt, cut=ring_cut, edge_break=eb,
                                     name_prefix=f"RB{r['id']}_")
        ribbons.append(rr)
        ribbon_cuts.append(ring_cut)
        open_last.append(True)
        island_col = rr.road_cols.stop - 1 if side == "R" else rr.road_cols.start
        outer_verge = rr.road_cols.start - 1 if side == "R" else rr.road_cols.stop
        interior |= {(float(e), float(n)) for e, n in zip(rr.e[:, island_col], rr.n[:, island_col])}
        if ex_built is not None:
            meshes += ex_built.meshes
            ribbons += ex_built.footprint.ribbons
            ribbon_cuts += [None] * len(ex_built.footprint.ribbons)
            open_last += [False] * len(ex_built.footprint.ribbons)
            patches += ex_built.footprint.patches
            edges += ex_built.footprint.edges
        # The two throats, and the verges along the outer flares.
        roads, grass, island_flares = [], [], []
        cols = list(range(rr.road_cols.start, rr.road_cols.stop))
        for end, row, looking in ((r["ends"][0], 0, cols), (r["ends"][1], n_rows - 1, cols[::-1])):
            material = sec.STYLES[int(style[row])]
            got = _throat(end, rb, side, rr, row, looking, material)
            if got is None:
                raise ValueError(f"roundabout {r['id']}: its {end['end']} end's throat isn't a simple polygon")
            piece, ring_pts, flares = got
            roads.append((material, piece))
            patches.append(ring_pts)
            for fl in flares:
                if fl[4]:
                    island_flares.append(fl)
                    continue
                vpiece, vpatch, vout, vout_h = _flare_verge(rb, side, rr, row, outer_verge, fl)
                grass.append(vpiece)
                patches.append(vpatch)
                edges.append((vout, vout_h))
        for material in sorted({mat for mat, _ in roads}):
            key = "1GRAVEL" if sec.is_unsealed(sec.STYLES.index(material)) else "1ROAD"
            letter = "N" if material in ("road_plain", "gravel_narrow") else ""
            meshes += mesh.soup_meshes(f"{key}_RB{r['id']}_T{letter}", material,
                                       [p for mat, p in roads if mat == material], frame)
        grass = [(p, frame.to_ac(p[:, 0], p[:, 1], p[:, 2])[:, [0, 2]] / 6.0, t) for p, _, t in grass]
        # The island: the road's island-side edge, the leaving end's island flare, the ring's inner edge, and the
        # joining end's island flare, every vertex shared with the meshes it meets.
        r_from, r_to = r["island"]["mainline_rows"]
        edge_col = rb.road_cols.start if side == "L" else rb.road_cols.stop - 1
        fl_leave = next(f for f in island_flares if f[1] == r_to)
        fl_join = next(f for f in island_flares if f[1] == r_from)
        bxy = np.vstack([np.column_stack([rb.e[r_from:r_to + 1, edge_col], rb.n[r_from:r_to + 1, edge_col]]),
                         fl_leave[2][1:-1], np.column_stack([rr.e[:, island_col], rr.n[:, island_col]]),
                         fl_join[2][::-1][1:-1]])
        bh = np.concatenate([rb.h[r_from:r_to + 1, edge_col], fl_leave[3][1:-1], rr.h[:, island_col],
                             fl_join[3][::-1][1:-1]])
        bxy, bh = stubs._dedupe(bxy, bh)
        if np.hypot(*(bxy[-1] - bxy[0])) < 1e-6:
            bxy, bh = bxy[:-1], bh[:-1]
        kerb, island_grass = _island(bxy, bh, frame)
        if kerb is None:
            raise ValueError(f"roundabout {r['id']}: no room for its island inside the kerb")
        meshes += mesh.soup_meshes(f"1ROAD_RB{r['id']}_K", KERB_MATERIAL, [kerb], frame)
        meshes += mesh.soup_meshes(f"1GRASS_RB{r['id']}_I", "grass", island_grass + grass, frame)
        patches.append(bxy)
    stats = {"rings": len(rings), "exits": len(exit_stubs(rings)), "meshes": len(meshes),
             "vertices": int(sum(len(m.positions) for m in meshes)),
             "triangles": int(sum(len(m.triangles) for m in meshes))}
    return Built(meshes, cut, edge_break, ribbons, ribbon_cuts, open_last, patches, edges, interior, stats)


def footprint(stub_built, ring_built: Built, cut: dict) -> mesh.Footprint:
    """The terrain's footprint with the side roads' stubs (`stubs.Built`, or None) and the rings together. `cut` is the
    road's verge and skirt quads taken out for both."""
    sb = stub_built.footprint if stub_built is not None else mesh.Footprint(cut, [], [], [])
    return mesh.Footprint(cut, sb.ribbons + ring_built.ribbons, sb.patches + ring_built.patches,
                          sb.edges + ring_built.edges,
                          ribbon_cuts=[None] * len(sb.ribbons) + ring_built.ribbon_cuts,
                          open_last=[False] * len(sb.ribbons) + ring_built.open_last,
                          interior=ring_built.interior)


# -- checking them as built -------------------------------------------------------------------------------------------

def checks(rings: list[dict], c, section, frame: LocalFrame) -> list:
    """Each ring as the validator's stub checks see it (`validate.stub_check`, `validate.stub_failures`), keyed
    ("ring", id, part): each end shaped as a stub on the road; the ring's ribbon, drawn as road along its centreline
    with no hole round it; the island, with no hole in it; and each exit, a stub on the ring (keyed by its junction)."""
    from mapcreator.core import validate

    def ac(p):
        p = np.asarray(p, float).reshape(-1, 2)
        return frame.to_ac(p[:, 0], p[:, 1], np.zeros(len(p)))[:, [0, 2]]

    out = []
    no_seams = np.zeros((0, 2, 2))
    for r in rings:
        for end in r["ends"]:
            out.append((("ring", r["id"], end["end"]),
                        validate.stub_check(end, c["s"], c["xy"], c["heading"], section["half_left"],
                                            section["half_right"], frame)))
        rows = np.asarray(r["rows"]["xy"], float)
        out.append((("ring", r["id"], "ring"),
                    validate.StubCheck(shapely.Polygon(ac(r["outline"])).buffer(0), ac(rows[1:-1:2]), no_seams)))
        out.append((("ring", r["id"], "island"),
                    validate.StubCheck(shapely.Polygon(ac(r["island"]["outline"])).buffer(0), np.zeros((0, 2)),
                                       no_seams)))
        c_ring, sec_ring = ribbon_arrays(r)
        for st in r.get("exits") or []:
            out.append((st["junction"], validate.stub_check(st, c_ring["s"], c_ring["xy"], c_ring["heading"],
                                                            sec_ring["half_left"], sec_ring["half_right"], frame)))
    return out


def settle(state: tuple[list[dict], list[dict]], build, check, rebuilds: int = 3, log=None):
    """`stubs.settle` with roundabouts: build with (stubs, rings), check, and build again without any stub, ring exit
    or ring the check finds faulty (a fault in a ring's end, ribbon or island leaves the whole ring out; its exits go
    with it; an exit's fault, only that exit), until nothing is. After `rebuilds` rounds all of them go, so neither
    a side road nor a roundabout ever fails a build. Returns the stubs and rings kept, the last result, and what was
    left out: stubs.json's `skipped` records for the stubs, and roundabouts.json's for the rings and exits."""
    stubs_now, rings_now = state
    dropped_stubs: list[dict] = []
    dropped_rings: list[dict] = []
    for attempt in range(rebuilds + 2):
        result = build((stubs_now, rings_now))
        found = check((stubs_now, rings_now), result) if (stubs_now or rings_now) else {}
        if not found:
            return stubs_now, rings_now, result, dropped_stubs, dropped_rings
        bad_rings = {key[1] for key in found if isinstance(key, tuple)}
        for st in stubs_now:
            if st["junction"] in found:
                dropped_stubs.append({"junction": st["junction"], "name": st.get("name"), "reason": "build_check",
                                      "faults": found[st["junction"]]})
                if log is not None:
                    log.warning("side-road stub left out: %s (junction %s) at %.0f m: %s",
                                st.get("name") or "an unnamed road", st["junction"], st["station_m"],
                                ", ".join(found[st["junction"]]))
        kept_rings = []
        for r in rings_now:
            if r["id"] in bad_rings or attempt >= rebuilds:
                why = sorted({f"{key[2]}: {f}" for key, faults in found.items() if isinstance(key, tuple)
                              and key[1] == r["id"] for f in faults})
                dropped_rings.append({"roundabout": r["id"], "name": r.get("name"), "reason": "build_check",
                                      "faults": why})
                if log is not None:
                    log.warning("roundabout left as the route's arc alone: %s (%s)", r.get("label") or "unnamed",
                                "; ".join(why) or "gave up")
                continue
            ex_bad = [st for st in r.get("exits") or [] if st["junction"] in found]
            for st in ex_bad:
                dropped_rings.append({"roundabout": r["id"], "junction": st["junction"], "name": st.get("name"),
                                      "reason": "build_check", "faults": found[st["junction"]]})
                if log is not None:
                    log.warning("roundabout exit left out: %s (junction %s): %s", st.get("name") or "an unnamed road",
                                st["junction"], ", ".join(found[st["junction"]]))
            if ex_bad:
                r = {**r, "exits": [st for st in r["exits"] if st["junction"] not in found]}
            kept_rings.append(r)
        if attempt >= rebuilds:
            dropped_stubs += [{"junction": st["junction"], "name": st.get("name"), "reason": "build_check",
                               "faults": []} for st in stubs_now if st["junction"] not in found]
            stubs_now = []
        else:
            stubs_now = [st for st in stubs_now if st["junction"] not in found]
        rings_now = kept_rings
    return stubs_now, rings_now, build((stubs_now, rings_now)), dropped_stubs, dropped_rings


# -- the preview's warning --------------------------------------------------------------------------------------------

NEAR_RING_M = 40.0  # the route's tightest bend within this of a roundabout counts as the roundabout's


def tightest_near(xy: np.ndarray, from_m: float, to_m: float) -> float | None:
    """The route's tightest radius between `from_m` and `to_m` (and NEAR_RING_M either side) as validation's route
    check judges it: the line smoothed as the profile stage smooths it (`centreline.smooth`), its curvature every
    metre smoothed as `ai_line.radius_of` smooths it. None if the line is too short to smooth."""
    from mapcreator.core import ai_line

    try:
        line = cl.smooth(np.asarray(xy, float))
    except ValueError:
        return None
    raw = cl.cumulative(np.asarray(xy, float))
    # The smoothed line's stations drift a little from the routed line's: map the stretch across by projection.
    lo, hi = from_m - NEAR_RING_M, to_m + NEAR_RING_M
    pts = np.column_stack([np.interp([max(lo, 0.0), min(hi, raw[-1])], raw, xy[:, 0]),
                           np.interp([max(lo, 0.0), min(hi, raw[-1])], raw, xy[:, 1])])
    k = cKDTree(line.xy).query(pts)[1]
    r = ai_line.radius_of(line.curvature)
    a, b = int(min(k)), int(max(k))
    return float(r[a:b + 1].min()) if b > a else None


FIT_TOLERANCE = 0.05  # a fitted circle's points stay within this share of its radius of it, or it's no circle


def fit_radius(points: np.ndarray) -> float | None:
    """A circle's radius through points on an arc (least squares), or None: fewer than three, or no circle fits them
    (a straight line)."""
    p = np.asarray(points, float)
    if len(p) < 3:
        return None
    centre = p.mean(axis=0)
    q = p - centre
    a = np.column_stack([2 * q, np.ones(len(q))])
    if np.linalg.matrix_rank(a, tol=1e-6 * max(float(np.abs(q).max()), 1e-9)) < 3:
        return None
    rhs = (q * q).sum(axis=1)
    (cx, cy, c0), *_ = np.linalg.lstsq(a, rhs, rcond=None)
    r2 = c0 + cx * cx + cy * cy
    if r2 <= 0 or not np.isfinite(r2) or math.sqrt(r2) > 1000.0:
        return None
    r = math.sqrt(r2)
    if np.abs(np.hypot(q[:, 0] - cx, q[:, 1] - cy) - r).max() > FIT_TOLERANCE * r:
        return None
    return float(r)


def _distance(m: float) -> str:
    return f"{m:.0f} m" if m < 1000 else f"{m / 1000:.1f} km"


def preview_notes(xy: np.ndarray, rings: list[dict]) -> list[dict]:
    """What to tell the person drawing a route about the roundabouts it drives (Q14), before they build: each ring
    (`from_m`, `to_m` along `xy`, the route's line in metres, and `radius_m`, None where unknown) whose radius is
    SMALL_RING_M or less, or whose stretch of the smoothed route bends tighter than validation's 5 m. Each note has
    the ring's place along the route, its radius, the tightest bend found, and a plain message."""
    from mapcreator.core import validate

    out = []
    for ring in rings:
        radius = ring.get("radius_m")
        tight = tightest_near(xy, ring["from_m"], ring["to_m"])
        small = radius is not None and radius <= SMALL_RING_M
        fails = tight is not None and tight < validate.MIN_RADIUS_M
        if not (small or fails):
            continue
        along = (ring["from_m"] + ring["to_m"]) / 2
        size = f" ({radius:.0f} m radius)" if radius is not None else ""
        # The preview's line is the router's, not always the build's (GraphHopper simplifies it), so neither message
        # promises what the build will find.
        if fails:
            text = (f"The route goes round a small roundabout{size} {_distance(along)} along, and its path through it "
                    f"looks tighter than the {validate.MIN_RADIUS_M:.0f} m radius a build accepts (about "
                    f"{tight:.1f} m). If the build stops there, you can generate it anyway, tight bend and all.")
        else:
            text = (f"The route goes round a small roundabout{size} {_distance(along)} along. A path through a ring "
                    f"this small can bend tighter than the {validate.MIN_RADIUS_M:.0f} m radius a build accepts: if "
                    "the build stops there, you can generate it anyway, tight bend and all.")
        out.append({"along_m": round(along, 1), "radius_m": round(radius, 1) if radius is not None else None,
                    "tightest_m": round(tight, 1) if tight is not None else None, "message": text})
    return out
