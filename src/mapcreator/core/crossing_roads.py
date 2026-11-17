"""Roads crossing the route without meeting it, drawn: the road under the route's own bridge, and a road's bridge over
the route with its approaches (`core.crossings` lists them in `01_route/crossings.json`).

The maintainer's answers to the flyovers research (17 Nov 2026):

- **A road under the route's bridge** (Q6, "yes"): its surface, verges and skirt from the DEM for REACH_M either side of
  the route's deck, as its own ribbon with the terrain cut round it, like a side road's stub (`core.stubs`) but with
  no flares, no closure and no police car: it's out of reach, and the route's parapets keep cars off it. It's drawn
  only where the route's deck stands MIN_GAP_M or more above the ground the DEM shows under it (the rails' rule,
  `roadside.BRIDGE_GAP_M`); with less, the elevation data predates the bridge and there's no room under it.
- **A road over the route on its own bridge** (Q8, "yes"): the deck, visual only, in the generic style chosen for the
  route's own bridges (Q3: a concrete beam deck with a concrete barrier and one steel rail), with piers and
  abutments that have colliders where the eye sees them. The deck's top runs straight between the road's heights at
  the bridge way's own ends (as the route's own bridges do, `profile.fill_runs`), and it's drawn only where its
  underside clears the route by CLEARANCE_M or more, the lowest clearance Australian practice allows on local roads
  (5.4 m on main roads). Otherwise the whole structure is left out, with the reason: where the DEM shows no gap
  under the bridge, the elevation data predates it.
- **The road over the route beyond its bridge** (Q9, "yes"): its surface on the terrain for REACH_M past each end of
  the bridge, built as a road under the route's bridge is, with no closure: it's up on its embankment, out of reach.

**The plan** (`plan`, run by the section stage, `03_section/crossing_roads.json`; its format is in docs/ac-format.md)
holds each road to draw as rows a ROW_M apart along its smoothed line, with heights, width and look, and each deck
with its abutments and piers, or why it's left out. The mesh and scenery stages, which run side by side, both read
it: the mesh stage draws from it, and the scenery stage keeps trees, buildings and power-line supports off the roads
and from under the decks.

**The meshes** (`build` and `decks`, run by the mesh stage). A road's ribbon (`mesh.build_ribbon`) gets road, verge
and skirt like a stub's, and joins the terrain's footprint (`mesh.Footprint`). It keeps off the ground the route, its
side roads' stubs and its roundabouts already take, and off the other crossing roads drawn before it (major roads
first): where even its road and verge would reach them, its rows are left out, so a road under the route's bridge
stops short of today's embankment either side of it (the route's own open deck, with the terrain under it, is stage
1's work; where the route's ground leaves room under the deck, the road carries on under it). A deck is its own
visual meshes, `BRIDGE_X<k>_*`, and its colliders, `1WALL_PIER_X<k>` and `1WALL_ABUT_X<k>`. The ground under a deck
is held below its underside (`GroundUnderDecks`), as `water.lowered` holds it under water.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

from mapcreator.core import centreline as cl
from mapcreator.core import dem_age
from mapcreator.core import junctions as jn
from mapcreator.core import mesh
from mapcreator.core import profile as pf
from mapcreator.core import section as sec
from mapcreator.core import textures
from mapcreator.core.scene import MAX_VERTICES, LocalFrame, Mesh

FORMAT_VERSION = 1

REACH_M = 40.0  # each crossing road is drawn this far past the route's deck (under) or past its bridge's ends (over)
DECK_EDGE_M = 1.5  # the route's deck reaches this far past its road's edge: the kerb, footway and parapet (stage 1)
ROW_M = 2.0  # a crossing road's rows, and a deck's, this far apart
DEM_SMOOTH_M = 15.0  # a crossing road's heights: the DEM along it, smoothed over this, as a stub's own profile
FILL_PAD_M = 3.0  # a deck's top runs straight between the road's heights this far past its ends (`profile.fill_runs`)
MIN_GAP_M = 1.5  # a road under the route's bridge needs the deck this far above the DEM's ground under it
CLEARANCE_M = 4.6  # a deck over the route needs its underside this far above the route's road and verges
DECK_DEPTH_M = 1.3  # a beam deck's depth, its road's surface to its underside (a guess at a typical overpass)
KERB_M = 1.0  # a deck's concrete footway either side of its road, as the route's own bridges have (Q5)
PARAPET_W = 0.35  # the concrete barrier's width
PARAPET_H = 0.8  # its height above the footway
RAIL_TOP = 1.1  # the steel rail's top above the footway
RAIL_SIZE = 0.1  # its square section
RAIL_POST_M = 2.5  # its posts this far apart
POST_SIZE = 0.08
ABUT_GAP_M = 2.0  # an abutment's face stands where the ground under the deck rises within this of its underside...
ABUT_CLEAR_M = 1.5  # ...and at least this far beyond the route's verge
ABUT_RAMP_M = 2.0  # behind the face the ground may rise into the deck's end over this
UNDER_CLEAR_M = 0.5  # the ground under a deck's open span is held this far below its underside
TOP_CLEAR_M = 0.3  # and under its ends this far below its top
CLAMP_BEYOND_M = 2.0  # the ground is held down this far beyond the deck's edges too
PIER_SPAN_M = 30.0  # a span longer than this between abutments gets piers
PIER_CLEAR_M = 1.5  # a pier stands at least this far from the route's road edge...
PIER_GRID_CLEAR_M = 3.0  # ...and this far within GRID_ZONE_M of either end of the route, clear of the grid's spawns
GRID_ZONE_M = 100.0
PIER_R = 0.45  # a pier column's radius
PIER_SIDES = 8
HEADSTOCK = (1.0, 0.8)  # the beam across a pier's columns under the deck: along the deck, deep
WALL_EMBED_M = 0.3  # walls and colliders reach this far below the ground
PIER_EMBED_M = 0.5
PIER_COLLIDER = (-1.0, 3.0)  # a pier's collider, from its ground, as a tree trunk's (`scenery.TRUNK_COLLIDER`)
MIN_ROWS = 6  # a piece of crossing road with fewer rows (10 m) isn't drawn
CLEAR_M = 0.5  # between a crossing road's ground and the route's, a stub's, or another crossing road's, as a stub's
WINDOW_M = 400.0  # crossing roads' meshes are packed by material per this much of the route, as the stubs' are
MATCH_M = 50.0  # a crossing is found on the smoothed centreline within this of where its route distance falls
CLASS_RANK = {"motorway": 0, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4, "unclassified": 5,
              "residential": 6, "living_street": 7, "road": 8}
CONCRETE = "bridge_concrete"  # the deck's concrete: the roundabouts' weathered-concrete texture (`textures.kerb`)
SOFFIT = "bridge_soffit"  # its underside: the same texture, darker, since nothing lights it from below
CONCRETE_TEXTURE = "kerb"
RAIL_MATERIAL = "rail"  # the guard rails' galvanised steel (`roadside`)
PIER_MESH = "1WALL_PIER"  # the piers' and abutments' invisible colliders: solid, as any physics mesh named WALL
ABUTMENT_MESH = "1WALL_ABUT"
DECK_MESH = "BRIDGE_X"  # a deck's visual meshes
KN5_PROPERTIES = {CONCRETE: {"ksAmbient": 0.38, "ksDiffuse": 0.5, "ksSpecular": 0.03, "ksSpecularEXP": 12.0},
                  SOFFIT: {"ksAmbient": 0.22, "ksDiffuse": 0.3, "ksSpecular": 0.01, "ksSpecularEXP": 8.0}}
# Why a crossing isn't drawn, as the build's warnings say it. The elevation data's own reasons are the profile stage's
# verdict (`core.dem_age`), worded as it words them, with what was left out.
DEM_REASONS = {"dem_predates": False, "dem_coarse": True}  # reason: whether it's the coarse one (`dem_age.warning`)
REASONS = {
    "not_a_bridge": "it isn't mapped as a bridge",
    "long_bridge": "its bridge runs on further than the road data is followed",
    "short": "there's too little of it to draw",
    "no_elevation": "there's no elevation data along it",
}
LEFT_OUT = {"over": "its deck is left out", "under": "the road under it is left out"}


# -- the plan ---------------------------------------------------------------------------------------------------------

def _rank(highway: str | None) -> float:
    h = str(highway or "")
    return CLASS_RANK.get(h.removesuffix("_link"), len(CLASS_RANK)) + (0.5 if h.endswith("_link") else 0.0)


class _Route:
    """The route's smoothed centreline and cross-section, and projections onto its stretch near a crossing."""

    def __init__(self, c, section):
        self.s = np.asarray(c["s"], float)
        self.xy = np.asarray(c["xy"], float)
        self.heading = np.asarray(c["heading"], float)
        self.z = np.asarray(c["z"], float)
        self.z_raw = np.asarray(c["z_raw"], float) if "z_raw" in c else self.z
        self.half = {"L": np.asarray(section["half_left"], float), "R": np.asarray(section["half_right"], float)}
        self.slope = {"L": np.asarray(section["slope_left"], float), "R": np.asarray(section["slope_right"], float)}

    def station(self, crossing: dict, scale: float) -> float:
        near = np.flatnonzero(np.abs(self.s - crossing["at_m"] * scale) <= MATCH_M)
        p = np.asarray(crossing["xy"], float)
        if len(near) < 2:
            return jn._station(self.xy, self.s, p, crossing["at_m"] * scale)
        return jn._station(self.xy[near], self.s[near], p, crossing["at_m"] * scale)

    def project(self, at: float, p: np.ndarray, reach: float = 300.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each point's station, offset (+ to the left) and whether it lies alongside the route's stretch within
        `reach` of station `at` (not past its ends)."""
        idx = np.flatnonzero(np.abs(self.s - at) <= reach)
        p = np.atleast_2d(np.asarray(p, float))
        _, k = cKDTree(self.xy[idx]).query(p)
        i = idx[k]
        h = self.heading[i]
        d = p - self.xy[i]
        st = self.s[i] + d[:, 0] * np.cos(h) + d[:, 1] * np.sin(h)
        off = -d[:, 0] * np.sin(h) + d[:, 1] * np.cos(h)
        inside = (st >= self.s[0]) & (st <= self.s[-1])
        return st, off, inside

    def half_at(self, st: np.ndarray, off: np.ndarray) -> np.ndarray:
        return np.where(off >= 0, np.interp(st, self.s, self.half["L"]), np.interp(st, self.s, self.half["R"]))

    def surface(self, st: np.ndarray, off: np.ndarray) -> np.ndarray:
        """The route's road surface, carried on over its verge as the verge falls, at (station, offset)."""
        half = self.half_at(st, off)
        slope = np.where(off >= 0, np.interp(st, self.s, self.slope["L"]), np.interp(st, self.s, self.slope["R"]))
        a = np.abs(off)
        return (np.interp(st, self.s, self.z) - slope * np.minimum(a, half)
                - mesh.VERGE_FALL * np.clip(a - half, 0.0, None))


def _road_line(crossing: dict) -> dict | None:
    """The crossing road's line through the ways that carry it on (`crossings.road_through`), or, from a route stage
    before those were followed, the crossing way's own: None without either."""
    road = crossing.get("road")
    if road:
        return road
    if not crossing.get("line"):
        return None
    line = np.asarray(crossing["line"], float)
    return {"line": crossing["line"], "at_m": crossing.get("line_at_m", 0.0),
            "ways": [{"way_id": crossing.get("way_id"), "from_m": 0.0, "to_m": float(cl.cumulative(line)[-1]),
                      "bridge": bool(crossing.get("bridge")), "tunnel": bool(crossing.get("tunnel")),
                      "level": crossing.get("level"), "highway": crossing.get("highway")}], "ends": ["length", "length"]}


@dataclass
class _Line:
    """A crossing road's smoothed line, a station every metre, with its own crossing's station on it."""

    s: np.ndarray
    xy: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    at: float
    raw: np.ndarray
    raw_at: float

    def to_s(self, raw_m: float) -> float:
        """Where a point `raw_m` along the mapped line falls on the smoothed one."""
        rs = cl.cumulative(self.raw)
        p = np.array([np.interp(raw_m, rs, self.raw[:, 0]), np.interp(raw_m, rs, self.raw[:, 1])])
        return jn._station(self.xy, self.s, p, raw_m - self.raw_at + self.at)

    def rows(self, a: float, b: float) -> np.ndarray:
        """Stations from a to b, ROW_M apart, both ends exact."""
        n = max(int(math.ceil((b - a) / ROW_M - 1e-6)), 1)
        return np.linspace(a, b, n + 1)

    def at_s(self, ss: np.ndarray):
        ss = np.asarray(ss, float)
        xy = np.column_stack([np.interp(ss, self.s, self.xy[:, 0]), np.interp(ss, self.s, self.xy[:, 1])])
        return xy, np.interp(ss, self.s, self.heading), np.interp(ss, self.s, self.curvature)


def _smoothed(road: dict | None) -> _Line | None:
    if road is None:
        return None
    raw = np.asarray(road["line"], float)
    if len(raw) < 2:
        return None
    keep = np.concatenate([[True], np.hypot(*np.diff(raw, axis=0).T) > 1e-6])
    raw = raw[keep]
    try:
        sm = cl.smooth(raw)
    except ValueError:
        return None
    rs = cl.cumulative(raw)
    at_raw = float(road["at_m"])
    p = np.array([np.interp(at_raw, rs, raw[:, 0]), np.interp(at_raw, rs, raw[:, 1])])
    at = jn._station(sm.xy, sm.s, p, at_raw)
    return _Line(sm.s, sm.xy, sm.heading, sm.curvature, at, raw, at_raw)


def _heights(line: _Line, dem_sample, mask: np.ndarray | None = None, humps_at: float | None = None):
    """The road's surface along `line`'s stations: the DEM smoothed over DEM_SMOOTH_M, with the stations in `mask` (a
    bridge) run straight between the road's heights FILL_PAD_M past their ends, and with `humps_at` (where it
    passes under the route) a hump the DEM shows there bridged, as on the route (`profile.humps_under`). Returns the
    heights and the raw DEM, or None with no elevation at all."""
    raw = dem_sample(line.xy[:, 0], line.xy[:, 1])
    ok = np.isfinite(raw)
    if ok.sum() < 2:
        return None
    raw = np.interp(line.s, line.s[ok], raw[ok])
    fill = np.zeros(len(line.s), bool) if mask is None else mask.copy()
    if humps_at is not None:
        fill |= pf.humps_under(line.s, raw, [humps_at])
    try:
        z, _ = pf.fill_runs(line.s, raw, fill, pad_m=FILL_PAD_M if fill.any() else 0.0)
    except pf.NoElevation:
        return None
    return uniform_filter1d(z, max(int(round(DEM_SMOOTH_M)), 1), mode="nearest"), raw


def _look(crossing: dict, width_factor: float) -> dict:
    """Width, surface and look by the route's rules (`core.section`), from the crossing way's tags."""
    tags = {**(crossing.get("tags") or {}), "highway": crossing.get("highway")}
    surface = sec.surface_class(tags) or sec.SEALED
    unsealed = surface == sec.UNSEALED
    mapped, source = sec.width_from_tags(tags, unsealed=unsealed)
    style = sec.style_code(unsealed, sec.single_lane(tags, mapped, unsealed))
    return {"surface": surface, "style": sec.STYLES[style], "width_m": round(mapped * width_factor, 3),
            "width_mapped_m": round(mapped, 2), "width_source": source,
            "crown": sec.CROWN_UNSEALED if unsealed else sec.CROWN}


def _r2(a):
    return np.round(np.asarray(a, float), 2).tolist()


def _r3(a):
    return np.round(np.asarray(a, float), 3).tolist()


def _rows_record(line: _Line, ss: np.ndarray, z_s: np.ndarray) -> dict:
    xy, h, k = line.at_s(ss)
    return {"s": _r2(ss), "xy": _r2(xy), "z": _r3(np.interp(ss, line.s, z_s)), "heading": np.round(h, 5).tolist(),
            "curvature": np.round(k, 5).tolist()}


def _paved(rows: dict, half: float) -> shapely.Geometry:
    xy = np.asarray(rows["xy"], float)
    if len(xy) < 2:
        return shapely.Polygon()
    return shapely.LineString(xy).buffer(half, cap_style="flat")


def _skip(crossing: dict, at: float, reason: str, **extra) -> dict:
    return {"crossing": crossing.get("id"), "way_id": crossing.get("way_id"), "name": crossing.get("name"),
            "highway": crossing.get("highway"), "relation": crossing["relation"], "station_m": round(at, 1),
            "label": dem_age.label(crossing, at), "reason": reason,
            **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in extra.items()}}


def _verdict(crossing: dict, verdicts: dict | None) -> tuple[str | None, dict]:
    """The profile stage's verdict on the bridge at this crossing (`core.dem_age`): the reason to leave it out
    (`dem_predates` or `dem_coarse`) and what to record with it; None where the elevation data holds the bridge; and
    "measure" where there's no verdict (a profile stage from before them, or a bridge it couldn't judge), so it's
    measured here instead."""
    bridge = (verdicts or {}).get(crossing.get("id"))
    if bridge is None or bridge.get("verdict") not in ("holds", "predates", "coarse"):
        return "measure", {}
    if bridge["verdict"] == "holds":
        return None, {}
    b = dem_age.Bridge(**bridge)
    return ("dem_coarse" if b.verdict == "coarse" else "dem_predates"), {"dem_age": True, "dem_detail": b.detail(),
                                                                          "label": b.label}


def _under(crossing: dict, route: _Route, at: float, dem_sample, width_factor: float, rid: int,
           verdicts: dict | None = None):
    """A road under the route's bridge, or why it isn't drawn."""
    look = _look(crossing, width_factor)
    w = look["width_m"] / 2
    angle = max(math.radians(max(crossing.get("angle_deg") or 90.0, 10.0)), 1e-3)
    across = w / math.sin(angle) + 1.0
    near = np.abs(route.s - at) <= across
    gap = float(np.max(route.z[near] - route.z_raw[near])) if near.any() else 0.0
    reason, why = _verdict(crossing, verdicts)
    if reason == "measure":  # no verdict: the deck must stand clear of the DEM's ground under it
        reason = "dem_predates" if gap < MIN_GAP_M else None
    if reason is not None:
        return None, _skip(crossing, at, reason, gap_m=gap, **why)
    line = _smoothed(_road_line(crossing))
    if line is None:
        return None, _skip(crossing, at, "short")
    st, off, inside = route.project(at, line.xy)
    band = inside & (np.abs(off) <= route.half_at(st, off) + DECK_EDGE_M) & (np.abs(st - at) <= 50.0)
    k0 = int(np.argmin(np.abs(line.s - line.at)))
    if not band.any():
        i0 = i1 = k0
    else:
        if not band[k0]:
            k0 = int(np.flatnonzero(band)[np.argmin(np.abs(np.flatnonzero(band) - k0))])
        i0, i1 = k0, k0
        while i0 > 0 and band[i0 - 1]:
            i0 -= 1
        while i1 < len(band) - 1 and band[i1 + 1]:
            i1 += 1
    a, b = max(float(line.s[i0]) - REACH_M, 0.0), min(float(line.s[i1]) + REACH_M, float(line.s[-1]))
    hz = _heights(line, dem_sample, humps_at=line.at)
    if hz is None:
        return None, _skip(crossing, at, "no_elevation")
    z_s, _ = hz
    ss = line.rows(a, b)
    if len(ss) < MIN_ROWS:
        return None, _skip(crossing, at, "short")
    rows = _rows_record(line, ss, z_s)
    record = {"id": rid, "kind": "under", "crossings": [crossing.get("id")], "deck": None, "station_m": round(at, 2),
              "label": dem_age.label(crossing, at),
              "way_id": crossing.get("way_id"), "way_ids": [w_["way_id"] for w_ in _road_line(crossing)["ways"]],
              "name": crossing.get("name"), "highway": crossing.get("highway"), "rank": _rank(crossing.get("highway")),
              **look, "gap_m": round(gap, 2), "length_m": round(float(b - a), 1), "rows": rows,
              "outline": _r2(np.asarray(_paved(rows, w).exterior.coords)[:-1])}
    return record, None


def _bridge_span(road: dict, crossing: dict) -> tuple[float, float] | None:
    """The contiguous run of bridge ways along the road's line that holds the crossing, as (from, to) metres along
    the mapped line."""
    ways = road["ways"]
    at = float(road["at_m"])
    hold = [k for k, w in enumerate(ways) if w["from_m"] - 1e-6 <= at <= w["to_m"] + 1e-6 and w["bridge"]]
    if not hold:
        return None
    a = b = hold[0]
    while a > 0 and ways[a - 1]["bridge"] and abs(ways[a - 1]["to_m"] - ways[a]["from_m"]) < 0.05:
        a -= 1
    while b < len(ways) - 1 and ways[b + 1]["bridge"] and abs(ways[b + 1]["from_m"] - ways[b]["to_m"]) < 0.05:
        b += 1
    return float(ways[a]["from_m"]), float(ways[b]["to_m"])


def _deck_points(xy: np.ndarray, heading: np.ndarray, half: float, step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Points across each of a deck's rows, `half` either side of its centreline, `step` apart: (rows, across, 2), and
    the offsets."""
    offs = np.linspace(-half, half, max(int(round(2 * half / step)), 1) + 1)
    nrm = cl.normals(heading)
    return xy[:, None, :] + offs[None, :, None] * nrm[:, None, :], offs


def _over(group: list[dict], route: _Route, ats: list[float], dem_sample, width_factor: float, did: int, rid: int,
          verdicts: dict | None = None):
    """A road's bridge over the route, and its approaches, or why it isn't drawn."""
    crossing = group[0]
    at = ats[0]
    ids = [c.get("id") for c in group]
    if not crossing.get("bridge"):
        return None, [], _skip(crossing, at, "not_a_bridge", crossings=ids)
    flagged = [_verdict(c, verdicts) for c in group]
    reason, why = next(((r, w) for r, w in flagged if r not in (None, "measure")), (None, {}))
    if reason is not None:  # the elevation data doesn't hold it (`core.dem_age`): not measured again
        return None, [], _skip(crossing, at, reason, crossings=ids, **why)
    road = _road_line(crossing)
    span = _bridge_span(road, crossing) if road is not None else None
    line = _smoothed(road)
    if span is None or line is None:
        return None, [], _skip(crossing, at, "short")
    raw_len = float(cl.cumulative(np.asarray(road["line"], float))[-1])
    if span[0] <= 0.5 or span[1] >= raw_len - 0.5:
        return None, [], _skip(crossing, at, "long_bridge")
    b0, b1 = sorted((line.to_s(span[0]), line.to_s(span[1])))
    if b1 - b0 < 2 * ROW_M:
        return None, [], _skip(crossing, at, "short")
    on_bridge = (line.s >= b0) & (line.s <= b1)
    hz = _heights(line, dem_sample, mask=on_bridge)
    if hz is None:
        return None, [], _skip(crossing, at, "no_elevation")
    z_s, raw = hz
    dem_gap = float(np.max((z_s - raw)[on_bridge])) if on_bridge.any() else 0.0
    look = _look(crossing, width_factor)
    w = look["width_m"] / 2
    half = w + KERB_M + PARAPET_W
    ss = line.rows(b0, b1)
    xy, hd, _ = line.at_s(ss)
    top = np.interp(ss, line.s, z_s)
    under = top - DECK_DEPTH_M
    # The clearance: the deck's underside over the route's road and verges, sampled every half metre.
    pts, _ = _deck_points(xy, hd, half)
    clearance = np.inf
    for a in ats:
        st, off, inside = route.project(a, pts.reshape(-1, 2))
        on = inside & (np.abs(off) <= route.half_at(st, off) + mesh.VERGE_M) & (np.abs(st - a) <= 150.0)
        if on.any():
            gaps = np.repeat(under, pts.shape[1])[on] - route.surface(st[on], off[on])
            clearance = min(clearance, float(gaps.min()))
    if not np.isfinite(clearance):
        return None, [], _skip(crossing, at, "short")
    if clearance < CLEARANCE_M:
        # With no verdict on it, a DEM that shows no gap under the bridge predates it.
        measured = all(r == "measure" for r, _ in flagged)
        reason = "dem_predates" if measured and dem_gap < MIN_GAP_M else "low_clearance"
        return None, [], _skip(crossing, at, reason, crossings=ids, clearance_m=clearance, dem_gap_m=dem_gap)
    # The abutments: from the route outwards, where the ground under the deck rises to near its underside, clear of
    # the route's verge.
    ground = np.interp(ss, line.s, raw)
    st, off, inside = route.project(at, pts.reshape(-1, 2))
    st, off, inside = (v.reshape(pts.shape[:2]) for v in (st, off, inside))
    beyond = np.abs(off) - route.half_at(st, off) - mesh.VERGE_M
    clear_row = (~inside | (beyond >= ABUT_CLEAR_M)).all(axis=1) & (
        (off > 0).all(axis=1) | (off < 0).all(axis=1) | ~inside.any(axis=1))
    k_at = int(np.argmin(np.hypot(*(xy - np.asarray(crossing["xy"], float)).T)))
    faces = []
    for direction, end in ((-1, 0), (1, len(ss) - 1)):
        face = end
        k = k_at
        while 0 <= k + direction < len(ss) and k != end:
            k += direction
            if clear_row[k] and ground[k] >= under[k] - ABUT_GAP_M:
                face = k
                break
        faces.append(face)
    fa, fb = faces
    if fb - fa < 2:
        fa, fb = 0, len(ss) - 1
    # Piers: over PIER_SPAN_M between the faces, evenly, each moved along to clear the route's road.
    piers = []
    span_m = float(ss[fb] - ss[fa])
    if span_m > PIER_SPAN_M:
        n = int(math.ceil(span_m / PIER_SPAN_M))
        limit = np.where((st < route.s[0] + GRID_ZONE_M) | (st > route.s[-1] - GRID_ZONE_M), PIER_GRID_CLEAR_M,
                         PIER_CLEAR_M)
        edge_beyond = np.abs(off) - route.half_at(st, off)
        ok_row = (~inside | (edge_beyond >= limit + PIER_R)).all(axis=1) & (
            (off > 0).all(axis=1) | (off < 0).all(axis=1) | ~inside.any(axis=1))
        reach = max(int(round(span_m / n / 2 / ROW_M)), 1)
        for i in range(1, n):
            want = fa + (fb - fa) * i / n
            cands = [k for k in range(int(want) - reach, int(math.ceil(want)) + reach + 1)
                     if fa + 2 <= k <= fb - 2 and ok_row[k]]
            if not cands:
                continue
            k = min(cands, key=lambda kk: abs(kk - want))
            if any(abs(k - p["row"]) < 2 for p in piers):
                continue
            nrm = cl.normals(np.array([hd[k]]))[0]
            cols = 2 if 2 * half < 10.0 else 3
            offs = np.linspace(-(half - 1.2), half - 1.2, cols)
            piers.append({"row": k, "xy": _r2(xy[k]), "columns": _r2(xy[k] + offs[:, None] * nrm[None, :])})
    deck_rows = {"s": _r2(ss), "xy": _r2(xy), "z": _r3(top), "heading": np.round(hd, 5).tolist()}
    outline = _r2(np.asarray(shapely.LineString(xy).buffer(half, cap_style="flat").exterior.coords)[:-1])
    deck = {"id": did, "crossings": ids, "station_m": round(at, 2), "label": dem_age.label(crossing, at),
            "way_id": crossing.get("way_id"),
            "name": crossing.get("name"), "highway": crossing.get("highway"), **look,
            "deck_half_m": round(half, 3), "depth_m": DECK_DEPTH_M, "clearance_m": round(clearance, 2),
            "dem_gap_m": round(dem_gap, 2), "length_m": round(float(b1 - b0), 1), "rows": deck_rows,
            "abutments": [{"end": "start", "face_row": fa, "end_row": 0},
                          {"end": "end", "face_row": fb, "end_row": len(ss) - 1}],
            "piers": piers, "approaches": [], "outline": outline}
    approaches = []
    for a, b in ((max(b0 - REACH_M, 0.0), b0), (b1, min(b1 + REACH_M, float(line.s[-1])))):
        if b - a < (MIN_ROWS - 1) * ROW_M - 1e-6:
            continue
        rss = line.rows(a, b)
        rows = _rows_record(line, rss, z_s)
        approaches.append({
            "id": rid + len(approaches), "kind": "approach", "crossings": [c.get("id") for c in group], "deck": did,
            "station_m": round(at, 2), "label": dem_age.label(crossing, at), "way_id": crossing.get("way_id"),
            "way_ids": [w_["way_id"] for w_ in road["ways"]], "name": crossing.get("name"),
            "highway": crossing.get("highway"), "rank": _rank(crossing.get("highway")), **look,
            "length_m": round(float(b - a), 1), "rows": rows,
            "outline": _r2(np.asarray(_paved(rows, w).exterior.coords)[:-1])})
    deck["approaches"] = [r["id"] for r in approaches]
    return deck, approaches, None


def plan(crossings: list[dict], c, section, dem_sample, *, width_factor: float = 1.0,
         route_length_m: float | None = None, dem_ages: dict | None = None) -> dict:
    """The roads crossing the route to draw, the decks over it, and those left out: the content of
    crossing_roads.json. `crossings` are crossings.json's; `c` and `section` the smoothed centreline's and
    cross-section's arrays; `route_length_m` the routed polyline's length, to find each crossing on the smoothed
    centreline (its stations drift from the routed line's). `dem_ages`: the profile stage's verdicts on the bridges
    (`core.dem_age`, `02_profile/dem_age.json`). A bridge the elevation data doesn't hold is left out on its word;
    without a verdict the plan measures the DEM itself."""
    route = _Route(c, section)
    verdicts = {i: b for b in (dem_ages or {}).get("bridges") or [] for i in b.get("crossings") or []}
    scale = float(route.s[-1]) / route_length_m if route_length_m else 1.0
    roads, decks, skipped = [], [], []
    groups: dict[int, list[tuple[dict, float]]] = {}
    for cr in crossings:
        at = route.station(cr, scale)
        if cr.get("relation") == "under":
            road, skip = _under(cr, route, at, dem_sample, width_factor, len(roads), verdicts)
            (roads.append(road) if road is not None else skipped.append(skip))
        elif cr.get("relation") == "over" and not (cr.get("route") or {}).get("tunnel"):
            groups.setdefault(cr.get("way_id"), []).append((cr, at))
    for group in groups.values():
        deck, approaches, skip = _over([g[0] for g in group], route, [g[1] for g in group], dem_sample, width_factor,
                                       len(decks), len(roads), verdicts)
        if deck is None:
            skipped.append(skip)
            continue
        decks.append(deck)
        roads += approaches
    return {"version": FORMAT_VERSION,
            "rule": {"reach_m": REACH_M, "deck_edge_m": DECK_EDGE_M, "row_m": ROW_M, "min_gap_m": MIN_GAP_M,
                     "clearance_m": CLEARANCE_M, "deck_depth_m": DECK_DEPTH_M, "kerb_m": KERB_M,
                     "parapet_m": [PARAPET_W, PARAPET_H], "rail_top_m": RAIL_TOP, "pier_span_m": PIER_SPAN_M,
                     "pier_clear_m": PIER_CLEAR_M, "abutment_gap_m": ABUT_GAP_M, "width_factor": width_factor},
            "roads": roads, "decks": decks, "skipped": sorted(skipped, key=lambda k: (k["station_m"], k["crossing"]))}


def summary(doc: dict) -> dict:
    """What a plan, or the meshes as built, holds, for the stage's summary."""
    out = {"roads": {kind: sum(r["kind"] == kind for r in doc["roads"]) for kind in ("under", "approach")},
           "decks": len(doc["decks"])}
    if doc.get("skipped"):
        out["skipped"] = {}
        for k in doc["skipped"]:
            out["skipped"][k["reason"]] = out["skipped"].get(k["reason"], 0) + 1
    if doc.get("left_out"):
        out["left_out"] = len(doc["left_out"])
    return out


def describe(item: dict) -> str:
    """How the build's warnings name a crossing road, a deck or one left out (`dem_age.label`)."""
    if item.get("label"):
        return item["label"]
    name = item.get("name") or "an unnamed road"
    return f"{name} ({item.get('highway')}) at {item['station_m'] / 1000:.2f} km"


def dem_covered(doc: dict | None) -> set:
    """The crossings whose bridge was left out on the profile stage's word (`core.dem_age`). `warnings` says so for
    them, with what was left out, so the profile stage's own line for them isn't repeated."""
    return {i for k in (doc or {}).get("skipped") or [] if k.get("dem_age")
            for i in (k.get("crossings") or [k.get("crossing")])}


def warnings(doc: dict | None, built: dict | None = None) -> list[str]:
    """The build's warnings about crossing roads: those drawn, and each one left out with the reason."""
    out = []
    if not doc:
        return out
    shown = built or doc
    under = [r for r in shown["roads"] if r["kind"] == "under"]
    if under:
        out.append(f"{len(under)} road{'s' if len(under) != 1 else ''} drawn under the route's bridges, out of reach: "
                   + "; ".join(describe(r) for r in under))
    if shown["decks"]:
        out.append(f"{len(shown['decks'])} bridge{'s' if len(shown['decks']) != 1 else ''} drawn over the route, "
                   "visual only, with colliders on their piers and abutments: "
                   + "; ".join(f"{describe(d)}, {d['clearance_m']:.1f} m clear" for d in shown["decks"]))
    # Bridges the elevation data doesn't hold, worded as `dem_age` words them; those that read the same (a highway's two
    # decks over the route) share a line.
    same: dict[tuple, list[dict]] = {}
    for k in doc.get("skipped") or []:
        if k["reason"] in DEM_REASONS:
            same.setdefault((describe(k), k["relation"], k["reason"]), []).append(k)
            continue
        then = LEFT_OUT[k["relation"]]
        if k["reason"] == "low_clearance":
            out.append(f"{describe(k)}: its underside would clear the route by only {k['clearance_m']:.1f} m, under "
                       f"the {CLEARANCE_M:g} m needed, so {then}.")
        else:
            out.append(f"{describe(k)}: {REASONS[k['reason']]}, so {then}.")
    for (name, relation, reason), group in same.items():
        k = group[0]
        ways = len({g.get("way_id") for g in group})  # one way crossing twice (a loop) is one bridge
        then = LEFT_OUT[relation]
        if ways > 1:
            name += f" ({ways} {'decks' if relation == 'over' else 'roads'})"
            then = "their decks are left out" if relation == "over" else "the roads under it are left out"
        if k.get("dem_detail"):
            detail = k["dem_detail"]
        elif relation == "under":
            detail = f"the route's deck stands {k.get('gap_m', 0.0):.1f} m above the ground under it"
        else:
            detail = f"its underside would clear the route by {k.get('clearance_m', 0.0):.1f} m"
        out.append(dem_age.warning(name, coarse=DEM_REASONS[reason], then=then, detail=detail))
    for k in (built or {}).get("left_out") or []:
        what = "it isn't drawn" if k["kind"] == "under" else "the road beyond one end of its bridge isn't drawn"
        out.append(f"{describe(k)}: {k['why']}, so {what}.")
    return out


def zone(doc: dict | None, beyond_m: float) -> shapely.Geometry | None:
    """The ground to keep clear of trees, buildings and power-line supports: every crossing road's carriageway widened
    by `beyond_m` either side, and every deck's plan, whole. None without any."""
    if not doc:
        return None
    parts = [shapely.LineString(np.asarray(r["rows"]["xy"], float)).buffer(r["width_m"] / 2 + beyond_m,
                                                                          cap_style="flat")
             for r in doc["roads"] if len(r["rows"]["xy"]) > 1]
    parts += [shapely.Polygon(d["outline"]).buffer(beyond_m) for d in doc["decks"]]
    return shapely.unary_union(parts) if parts else None


def centres(doc: dict | None) -> np.ndarray | None:
    """Every crossing road's centreline points, which the route's skirt is held short of (`stubs.mainline_caps`), as
    it is of a stub's: None without any."""
    if not doc or not doc["roads"]:
        return None
    return np.vstack([np.asarray(r["rows"]["xy"], float) for r in doc["roads"]])


# -- the ground under the decks ---------------------------------------------------------------------------------------

class GroundUnderDecks:
    """A DEM whose `sample` is held below each deck: under its open span UNDER_CLEAR_M below its underside, and under
    its ends (between an abutment's face and the bridge's end, rising over ABUT_RAMP_M) TOP_CLEAR_M below its top, out
    to CLAMP_BEYOND_M beyond its edges. Everything else is the DEM's (`water.lowered` does the same under water)."""

    def __init__(self, dem, decks: list[dict]):
        self._dem = dem
        self._decks = []
        for d in decks:
            xy = np.asarray(d["rows"]["xy"], float)
            ss = np.asarray(d["rows"]["s"], float)
            top = np.asarray(d["rows"]["z"], float)
            fa, fb = (int(a["face_row"]) for a in d["abutments"])
            low = top - float(d["depth_m"]) - UNDER_CLEAR_M
            high = top - TOP_CLEAR_M
            t = ss - ss[0]
            ta, tb = t[fa], t[fb]
            ramp = np.clip(np.maximum(ta - t, t - tb) / ABUT_RAMP_M, 0.0, 1.0)
            limit = low + (high - low) * ramp
            poly = shapely.LineString(xy).buffer(float(d["deck_half_m"]) + CLAMP_BEYOND_M, cap_style="flat")
            shapely.prepare(poly)
            self._decks.append((poly, xy, np.asarray(d["rows"]["heading"], float), t, limit))

    def __getattr__(self, name):
        return getattr(self._dem, name)

    def limit(self, x, y) -> np.ndarray:
        """The height the ground is held under at each point: inf away from every deck."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        flat = np.column_stack([x.ravel(), y.ravel()])
        out = np.full(len(flat), np.inf)
        for poly, xy, hd, t, limit in self._decks:
            inside = shapely.contains_xy(poly, flat[:, 0], flat[:, 1])
            if not inside.any():
                continue
            p = flat[inside]
            _, k = cKDTree(xy).query(p)
            u = np.column_stack([np.cos(hd[k]), np.sin(hd[k])])
            along = t[k] + ((p - xy[k]) * u).sum(axis=1)
            out[inside] = np.minimum(out[inside], np.interp(along, t, limit))
        return out.reshape(x.shape)

    def sample(self, x, y):
        h = self._dem.sample(x, y)
        return np.minimum(h, self.limit(x, y))


# -- the meshes -------------------------------------------------------------------------------------------------------

@dataclass
class Built:
    """The crossing roads' meshes, and what the terrain needs to make room for them."""

    meshes: list[Mesh]  # roads (1ROAD_X / 1GRAVEL_X) and verges (1GRASS_X), packed by material per WINDOW_M
    ribbons: list[mesh.Ribbon]  # for the terrain's footprint
    edges: list[tuple[np.ndarray, np.ndarray]]  # their first rows' road vertices, which the footprint needs too
    pieces: list[dict]  # what was drawn: road id, the plan's rows (first and last) and its ribbon's outline
    left_out: list[dict]  # roads with no piece drawn, and why
    stats: dict = field(default_factory=dict)


def _rays_to(obstacles, xy: np.ndarray, nrm: np.ndarray, start: np.ndarray, length: float) -> np.ndarray:
    """Distance from each centreline point, along its normal, to where it first meets `obstacles`, looking from
    `start` metres out to `length`: inf where it meets nothing."""
    a = xy + nrm * start[:, None]
    b = xy + nrm * (start + length)[:, None]
    rays = shapely.linestrings(np.stack([a, b], axis=1))
    hit = shapely.intersection(rays, obstacles)
    d = shapely.distance(shapely.points(xy), hit)
    return np.where(np.isfinite(d) & ~shapely.is_empty(hit), d, np.inf)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def build(doc: dict, taken: shapely.Geometry, frame: LocalFrame, ground_sample, *, skirt: bool,
          others_xy: np.ndarray | None = None) -> Built:
    """The crossing roads' ribbons and meshes, from the plan (`doc`), keeping off `taken` (the ground the route, its
    stubs and roundabouts take, `mesh.terrain_region`) and off each other, major roads first. `others_xy`: other
    roads' centreline points their skirts stop short of, as a stub's do (`stubs.side_caps`). With `skirt` False
    (ground classes) the verge meshes are the 2 m verge alone, and the skirts go with the terrain."""
    from mapcreator.core import stubs

    roads = sorted(doc["roads"], key=lambda r: (r["rank"], -r["length_m"], r["id"]))
    centres = {r["id"]: np.asarray(r["rows"]["xy"], float) for r in roads}
    taken = taken if taken is not None and not taken.is_empty else shapely.Polygon()
    shapely.prepare(taken)
    obstacles = taken
    full = mesh.VERGE_M + mesh.SKIRT_STEPS_M[-1]
    pieces, left_out, ribbons, edges = [], [], [], []
    grass: dict[int, list] = {}
    road_parts: dict[tuple[str, int], list] = {}
    for r in roads:
        rows = r["rows"]
        xy = np.asarray(rows["xy"], float)
        z = np.asarray(rows["z"], float)
        hd = np.asarray(rows["heading"], float)
        k = np.asarray(rows["curvature"], float)
        ss = np.asarray(rows["s"], float)
        half = float(r["width_m"]) / 2
        nrm = cl.normals(hd)
        reach = half + mesh.VERGE_M + CLEAR_M
        segs = shapely.linestrings(np.stack([xy - nrm * reach, xy + nrm * reach], axis=1))
        shapely.prepare(obstacles)
        sections = shapely.buffer(segs, ROW_M / 2 + CLEAR_M, cap_style="flat")
        blocked = shapely.intersects(sections, obstacles)
        others = [centres[o] for o in centres if o != r["id"]]
        if others_xy is not None and len(others_xy):
            others.append(np.asarray(others_xy, float))
        others_pts = np.vstack(others) if others else np.zeros((0, 2))
        by_route = shapely.intersects(sections, taken)
        why = ("it runs too close beside another road drawn there" if (blocked & ~by_route).any()
               else "there's no room for it beside the route's own ground")
        drawn = False
        for a, b in _runs(~blocked):
            if b - a < MIN_ROWS:
                continue
            sel = slice(a, b)
            cap_l, cap_r = stubs.side_caps(xy[sel], nrm[sel], others_pts, stubs.REACH_M)
            start = np.full(b - a, half + mesh.VERGE_M)
            ray_l = _rays_to(obstacles, xy[sel], nrm[sel], start, full)
            ray_r = _rays_to(obstacles, xy[sel], -nrm[sel], start, full)
            cap_l = np.minimum(cap_l, ray_l - CLEAR_M)
            cap_r = np.minimum(cap_r, ray_r - CLEAR_M)
            crown = np.full(b - a, float(r["crown"]))
            hw = np.full(b - a, half)
            rb = mesh.build_ribbon(xy[sel], z[sel], hd[sel], k[sel], hw, hw, crown, crown, ground_sample,
                                   skirt_cap_l=cap_l, skirt_cap_r=cap_r)
            outline, _ = mesh._outline(rb)
            poly = shapely.Polygon(outline).buffer(0)
            if not poly.is_valid or poly.is_empty or poly.intersection(obstacles).area > 0.05:
                why = "it would overlap the ground the route or another road takes"
                continue
            drawn = True
            obstacles = shapely.union(obstacles, poly)
            ribbons.append(rb)
            rc = rb.road_cols
            edges.append((np.column_stack([rb.e[0, rc], rb.n[0, rc]]), rb.h[0, rc].copy()))
            pieces.append({**_piece(r, a, b), "outline": _r2(outline)})
            window = int(float(r["station_m"]) // WINDOW_M)
            tile = textures.GRAVEL_TILE_M if r["surface"] == sec.UNSEALED else textures.ROAD_TILE_M
            e, n, h = rb.e[:, rc], rb.n[:, rc], rb.h[:, rc]
            nr, nc = e.shape
            uv = np.column_stack([np.tile(np.linspace(0, 1, nc), nr), np.repeat(ss[sel] / tile, nc)])
            road_parts.setdefault((r["style"], window), []).append(
                (np.column_stack([e.ravel(), n.ravel(), h.ravel()]), uv, mesh._grid_triangles(nr, nc)))
            left, right = (rb.left_cols, rb.right_cols) if skirt else mesh._verge_strips(rb)
            for cols in (left, right):
                e, n, h = rb.e[:, cols], rb.n[:, cols], rb.h[:, cols]
                nr, nc = e.shape
                grass.setdefault(window, []).append((np.column_stack([e.ravel(), n.ravel(), h.ravel()]), None,
                                                     mesh._grid_triangles(nr, nc)))
        if not drawn:
            left_out.append({"road": r["id"], "kind": r["kind"], "name": r.get("name"), "highway": r.get("highway"),
                             "station_m": r["station_m"], "label": r.get("label"), "why": why})
    meshes: list[Mesh] = []
    letter = {"road": "", "road_plain": "N", "gravel": "", "gravel_narrow": "N"}
    for (material, window), parts in sorted(road_parts.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        key = "1GRAVEL" if sec.is_unsealed(sec.STYLES.index(material)) else "1ROAD"
        meshes += mesh.soup_meshes(f"{key}_X{window:03d}{letter[material]}", material, parts, frame)
    for window, parts in sorted(grass.items()):
        parts = [(p, frame.to_ac(p[:, 0], p[:, 1], p[:, 2])[:, [0, 2]] / 6.0, t) for p, _, t in parts]
        meshes += mesh.soup_meshes(f"1GRASS_X{window:03d}", "grass", parts, frame)
    stats = {"pieces": len(pieces), "left_out": len(left_out), "meshes": len(meshes),
             "vertices": int(sum(len(m.positions) for m in meshes)),
             "length_m": round(sum(p["length_m"] for p in pieces), 1)}
    return Built(meshes, ribbons, edges, pieces, left_out, stats)


def _piece(r: dict, a: int, b: int) -> dict:
    ss = r["rows"]["s"]
    return {"road": r["id"], "kind": r["kind"], "name": r.get("name"), "highway": r.get("highway"),
            "station_m": r["station_m"], "rows": [a, b - 1], "length_m": round(float(ss[b - 1] - ss[a]), 1)}


def footprint(extra: mesh.Footprint | None, built: Built, rows: int) -> mesh.Footprint | None:
    """The terrain's footprint with the crossing roads' ribbons added (`mesh.Footprint`): their outlines bound the
    terrain, their first rows' road included (a stub's meets its throat there; these meet the terrain), and their
    skirts join it with ground classes. `extra` is the stubs' and roundabouts' (None without them); `rows` the road's
    rows."""
    if not built.ribbons:
        return extra
    if extra is None:
        extra = mesh.Footprint({"L": np.zeros(rows - 1, bool), "R": np.zeros(rows - 1, bool)}, [], [], [])
    k = len(built.ribbons)
    return mesh.Footprint(
        extra.cut, list(extra.ribbons) + built.ribbons, list(extra.patches), list(extra.edges) + built.edges,
        ribbon_cuts=list(extra.ribbon_cuts) + [None] * k if extra.ribbon_cuts is not None else None,
        open_last=list(extra.open_last) + [False] * k if extra.open_last is not None else None,
        interior=extra.interior, spans=extra.spans)


def as_built(doc: dict, built: Built) -> dict:
    """The plan as the mesh stage drew it: each road's pieces, and those left out (`04_mesh/crossing_roads.json`)."""
    drawn = {}
    for p in built.pieces:
        drawn.setdefault(p["road"], []).append({"rows": p["rows"], "outline": p["outline"]})
    roads = [{**r, "pieces": drawn[r["id"]]} for r in doc["roads"] if r["id"] in drawn]
    return {**doc, "roads": roads, "left_out": built.left_out}


# -- the decks' meshes ------------------------------------------------------------------------------------------------

class _Soup:
    """Pieces of one material in AC axes, each with its own normals (walls face sideways, the underside down)."""

    def __init__(self):
        self.pos, self.nrm, self.uv, self.tri, self.count = [], [], [], [], 0

    def grid(self, pos: np.ndarray, outward: np.ndarray, uv: np.ndarray) -> None:
        """A grid of quads (rows, cols, 3) in AC axes, each wound to face `outward` (rows, cols, 3)."""
        r, c = pos.shape[:2]
        if r < 2 or c < 2:
            return
        p = pos.reshape(-1, 3)
        o = outward.reshape(-1, 3)
        o = o / np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-12)
        tri = mesh._grid_triangles(r, c)
        g = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
        want = o[tri].mean(axis=1)
        flip = (g * want).sum(axis=1) < 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        keep = np.linalg.norm(g, axis=1) > 1e-9
        self.pos.append(p)
        self.nrm.append(o)
        self.uv.append(uv.reshape(-1, 2))
        self.tri.append(tri[keep] + self.count)
        self.count += len(p)

    def meshes(self, name: str, material: str, renderable: bool = True) -> list[Mesh]:
        if not self.tri:
            return []
        pos, nrm, uv, tri = (np.concatenate(a) for a in (self.pos, self.nrm, self.uv, self.tri))
        if len(pos) > MAX_VERTICES:
            raise ValueError(f"{name}: {len(pos)} vertices exceeds {MAX_VERTICES}")
        return [Mesh(name, material, pos.astype(np.float32), nrm.astype(np.float32), uv.astype(np.float32),
                     tri.astype(np.int64), renderable=renderable)]


def _ac_dir(e: np.ndarray, n: np.ndarray, h: np.ndarray) -> np.ndarray:
    """A direction in region axes (east, north, up) in AC axes (x east, y up, z south)."""
    return np.stack([np.asarray(e, float), np.asarray(h, float), -np.asarray(n, float)], axis=-1)


def _wall(soup: _Soup, frame: LocalFrame, xy: np.ndarray, lo: np.ndarray, hi: np.ndarray, outward: np.ndarray,
          tile: float = 2.0) -> None:
    """A vertical wall along `xy` (n, 2) from `lo` to `hi`, facing `outward` (n, 2) in plan."""
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    pos = np.stack([frame.to_ac(xy[:, 0], xy[:, 1], lo), frame.to_ac(xy[:, 0], xy[:, 1], hi)], axis=1)
    out = _ac_dir(outward[:, 0], outward[:, 1], np.zeros(len(xy)))
    uv = np.stack([np.column_stack([along / tile, lo / tile]), np.column_stack([along / tile, hi / tile])], axis=1)
    soup.grid(pos, np.stack([out, out], axis=1), uv)


def _flat(soup: _Soup, frame: LocalFrame, a: np.ndarray, b: np.ndarray, ha: np.ndarray, hb: np.ndarray,
          up: float, tile: float = 2.0) -> None:
    """A strip between two lines `a` and `b` (n, 2) at heights ha and hb, facing up (`up` 1) or down (-1)."""
    pos = np.stack([frame.to_ac(a[:, 0], a[:, 1], ha), frame.to_ac(b[:, 0], b[:, 1], hb)], axis=1)
    out = np.zeros_like(pos)
    out[..., 1] = up
    soup.grid(pos, out, pos[..., [0, 2]] / tile)  # AC x and z: small numbers, unlike eastings


def _box(soup: _Soup, frame: LocalFrame, centre: np.ndarray, u: np.ndarray, size: tuple[float, float],
         lo: float, hi: float) -> None:
    """A box standing from `lo` to `hi`, `size` (along u, across) in plan, with its four sides and its top."""
    v = np.array([-u[1], u[0]])
    a, b = size[0] / 2, size[1] / 2
    corners = np.array([centre - u * a - v * b, centre + u * a - v * b, centre + u * a + v * b, centre - u * a + v * b])
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        d = q - p
        out = np.array([d[1], -d[0]])  # the corners run anticlockwise: the outside is to the right
        _wall(soup, frame, np.array([p, q]), np.array([lo, lo]), np.array([hi, hi]), np.array([out, out]), 1.0)
    _flat(soup, frame, corners[[0, 1]], corners[[3, 2]], np.array([hi, hi]), np.array([hi, hi]), 1.0, 1.0)


def _prism(soup: _Soup, frame: LocalFrame, centre: np.ndarray, radius: float, lo: float, hi: float) -> None:
    theta = np.arange(PIER_SIDES + 1) * 2 * np.pi / PIER_SIDES
    ring = np.column_stack([np.cos(theta), np.sin(theta)])
    xy = centre + radius * ring
    pos = np.stack([frame.to_ac(xy[:, 0], xy[:, 1], np.full(len(xy), lo)),
                    frame.to_ac(xy[:, 0], xy[:, 1], np.full(len(xy), hi))], axis=1)
    out = _ac_dir(ring[:, 0], ring[:, 1], np.zeros(len(ring)))
    uv = np.stack([np.column_stack([theta / np.pi, np.full(len(xy), lo / 2)]),
                   np.column_stack([theta / np.pi, np.full(len(xy), hi / 2)])], axis=1)
    soup.grid(pos, np.stack([out, out], axis=1), uv)


def deck_meshes(d: dict, frame: LocalFrame, ground_sample) -> tuple[list[Mesh], dict]:
    """One deck's meshes: its road (the road's material), its concrete (footways, barriers, fascias, ends,
    abutments, piers), its underside, its steel rail (all visual, `BRIDGE_X<k>_*`), and the invisible colliders on
    its piers and abutments (`1WALL_PIER_X<k>`, `1WALL_ABUT_X<k>`). `ground_sample` gives the ground the walls and
    piers stand on (the terrain's, held under the deck: `GroundUnderDecks`)."""
    k = d["id"]
    xy = np.asarray(d["rows"]["xy"], float)
    ss = np.asarray(d["rows"]["s"], float)
    top = np.asarray(d["rows"]["z"], float)
    hd = np.asarray(d["rows"]["heading"], float)
    nrm = cl.normals(hd)
    tang = np.column_stack([np.cos(hd), np.sin(hd)])
    w = float(d["width_m"]) / 2
    half = float(d["deck_half_m"])
    inner = w + KERB_M
    crown = float(d["crown"])
    edge = top - crown * w
    under = top - float(d["depth_m"])
    road, concrete, soffit, rail = _Soup(), _Soup(), _Soup(), _Soup()
    piers_c, abut_c = _Soup(), _Soup()

    def at(o):
        return xy + nrm * o

    # The road: five points across, with its crown, as a ribbon's.
    offs = np.array([w, w / 2, 0.0, -w / 2, -w])
    pts = xy[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    hts = top[:, None] - crown * np.abs(offs)[None, :]
    pos = frame.to_ac(pts[..., 0], pts[..., 1], hts)
    up = np.zeros_like(pos)
    up[..., 1] = 1.0
    tile = textures.GRAVEL_TILE_M if d["surface"] == sec.UNSEALED else textures.ROAD_TILE_M
    uv = np.stack(np.broadcast_arrays(np.linspace(0, 1, 5)[None, :], (ss / tile)[:, None]), axis=-1)
    road.grid(pos, up, uv)
    for sign in (1.0, -1.0):
        out = nrm * sign
        # The footway, flush with the road's edge; the barrier's inner face, its top and its outer face, which runs
        # on down as the deck's fascia to its underside.
        _flat(concrete, frame, at(sign * w), at(sign * inner), edge, edge, 1.0)
        _wall(concrete, frame, at(sign * inner), edge, edge + PARAPET_H, -out)
        _flat(concrete, frame, at(sign * inner), at(sign * half), edge + PARAPET_H, edge + PARAPET_H, 1.0)
        _wall(concrete, frame, at(sign * half), under, edge + PARAPET_H, out)
        # The steel rail on posts, centred on the barrier's top.
        mid = sign * (inner + half) / 2
        r_lo, r_hi = edge + RAIL_TOP - RAIL_SIZE, edge + RAIL_TOP
        for o, face in ((mid + RAIL_SIZE / 2, 1.0), (mid - RAIL_SIZE / 2, -1.0)):
            _wall(rail, frame, at(o), r_lo, r_hi, nrm * face)
        _flat(rail, frame, at(mid - RAIL_SIZE / 2), at(mid + RAIL_SIZE / 2), r_hi, r_hi, 1.0)
        _flat(rail, frame, at(mid - RAIL_SIZE / 2), at(mid + RAIL_SIZE / 2), r_lo, r_lo, -1.0)
        for s_post in np.arange(ss[0] + RAIL_POST_M / 2, ss[-1], RAIL_POST_M):
            p = np.array([np.interp(s_post, ss, at(mid)[:, 0]), np.interp(s_post, ss, at(mid)[:, 1])])
            u = np.array([np.interp(s_post, ss, tang[:, 0]), np.interp(s_post, ss, tang[:, 1])])
            base = float(np.interp(s_post, ss, edge)) + PARAPET_H
            _box(rail, frame, p, u / max(np.hypot(*u), 1e-9), (POST_SIZE, POST_SIZE), base, base + RAIL_TOP - PARAPET_H)
    # The underside, flat across, and the deck's two ends, closed from its underside to its top.
    _flat(soffit, frame, at(half), at(-half), under, under, -1.0)
    for row, face in ((0, -1.0), (len(ss) - 1, 1.0)):
        across = np.array([half, inner, w, w / 2, 0.0, -w / 2, -w, -inner, -half])
        p = xy[row] + np.outer(across, nrm[row])
        hi = np.array([edge[row]] * 3 + [top[row] - crown * w / 2, top[row], top[row] - crown * w / 2]
                      + [edge[row]] * 3)
        lo = np.full(len(across), under[row])
        _wall(concrete, frame, p, lo, hi, np.tile(tang[row] * face, (len(across), 1)))
        for sign in (1.0, -1.0):
            q = xy[row] + np.outer(np.array([sign * half, sign * inner]), nrm[row])
            _wall(concrete, frame, q, np.full(2, edge[row]), np.full(2, edge[row] + PARAPET_H),
                  np.tile(tang[row] * face, (2, 1)))
    # The abutments: a wall across the deck at each face, from its underside down into the ground, facing the
    # route; and wing walls along its edges from the face to the bridge's end.
    for ab in d["abutments"]:
        fr, er = int(ab["face_row"]), int(ab["end_row"])
        face = 1.0 if er < fr else -1.0  # towards the route: away from the bridge's end
        if fr == er and fr in (0, len(ss) - 1):
            face = 1.0 if fr == 0 else -1.0
        across = np.linspace(half, -half, max(int(math.ceil(2 * half)), 2) + 1)
        p = xy[fr] + np.outer(across, nrm[fr])
        g = ground_sample(p[:, 0], p[:, 1])
        g = np.where(np.isfinite(g), g, under[fr])
        lo = np.minimum(g, under[fr]) - WALL_EMBED_M
        hi = np.full(len(across), under[fr])
        # At the bridge's very end the deck's own end face closes it, over the road beyond.
        if fr != er and (hi - lo > WALL_EMBED_M + 0.05).any():
            outward = np.tile(tang[fr] * face, (len(across), 1))
            _wall(concrete, frame, p, lo, hi, outward)
            _wall(abut_c, frame, p, lo, hi, outward)
        lo_r, hi_r = sorted((fr, er))
        if hi_r > lo_r:
            sl = slice(lo_r, hi_r + 1)
            for sign in (1.0, -1.0):
                q = at(sign * half)[sl]
                g = ground_sample(q[:, 0], q[:, 1])
                g = np.where(np.isfinite(g), g, under[sl])
                lo = np.minimum(g, under[sl]) - WALL_EMBED_M
                if (under[sl] - lo > WALL_EMBED_M + 0.05).any():
                    _wall(concrete, frame, q, lo, under[sl], nrm[sl] * sign)
                    _wall(abut_c, frame, q, lo, under[sl], nrm[sl] * sign)
    # The piers: round columns from the ground to a headstock under the deck.
    for pier in d["piers"]:
        r = int(pier["row"])
        cols = np.asarray(pier["columns"], float)
        g = ground_sample(cols[:, 0], cols[:, 1])
        g = np.where(np.isfinite(g), g, under[r] - 6.0)
        beam_lo = under[r] - HEADSTOCK[1]
        for c_xy, gh in zip(cols, g):
            _prism(concrete, frame, c_xy, PIER_R, float(gh) - PIER_EMBED_M, beam_lo)
            _prism(piers_c, frame, c_xy, PIER_R, float(gh) + PIER_COLLIDER[0], float(gh) + PIER_COLLIDER[1])
        _box(concrete, frame, xy[r], tang[r], (HEADSTOCK[0], 2 * half - 0.4), beam_lo, under[r])
    out = (road.meshes(f"{DECK_MESH}{k}_ROAD", d["style"]) + concrete.meshes(f"{DECK_MESH}{k}_CONCRETE", CONCRETE)
           + soffit.meshes(f"{DECK_MESH}{k}_SOFFIT", SOFFIT) + rail.meshes(f"{DECK_MESH}{k}_RAIL", RAIL_MATERIAL)
           + piers_c.meshes(f"{PIER_MESH}_X{k}", RAIL_MATERIAL, renderable=False)
           + abut_c.meshes(f"{ABUTMENT_MESH}_X{k}", RAIL_MATERIAL, renderable=False))
    stats = {"meshes": len(out), "vertices": int(sum(len(m.positions) for m in out)), "piers": len(d["piers"])}
    return out, stats


def decks(doc: dict | None, frame: LocalFrame, ground_sample) -> tuple[list[Mesh], dict]:
    """Every deck's meshes (`deck_meshes`), and their numbers."""
    if not doc or not doc["decks"]:
        return [], {"decks": 0}
    meshes, verts, piers = [], 0, 0
    for d in doc["decks"]:
        ms, st = deck_meshes(d, frame, ground_sample)
        meshes += ms
        verts += st["vertices"]
        piers += st["piers"]
    return meshes, {"decks": len(doc["decks"]), "meshes": len(meshes), "vertices": verts, "piers": piers}
