"""Railways and paths that cross the route without meeting it (stage 2c of the flyovers work): footbridges and rail
bridges over the route, drawn as visual decks with solid piers and abutments, and paths and railways under the
route's own bridges, drawn on the ground.

The maintainer's answers this follows (flyovers research, 17 Nov 2026): Q7 (footbridges and rail bridges over the
route, and paths or railways under its bridges), Q8 (a structure over the route is visual only, with colliders on its
piers and abutments, and drawn only where its underside clears the route by 4.6 m or more; elsewhere it's left out
and the build's warnings say why), Q3 (one generic style: a concrete beam deck with a concrete barrier and one steel
rail) and Q6 (what's under the route's bridge is drawn for 40 m either side, out of reach, with no closures).

**What crosses.** The route stage lists every railway and path whose line crosses the route's at a point that isn't
a node they share (`find`, the same rule as `core.crossings` for roads: which is on top comes from `layer`, `bridge`
and `tunnel`). A footpath or level crossing that meets the road at grade shares a node with it and isn't listed. For
each, it records the whole bridge the crossing way is part of (`structure`: the way joined to the bridge ways that
continue it), the line on from the bridge's ends (the approaches, 40 m), and the roads the bridge passes over
(`roads_crossed`), so piers stay off them. The format of `01_route/crossing_lines.json` is in `docs/ac-format.md`.

**Over the route** (`build`, the mesh stage), for a way tagged as a bridge:
- **Height.** The deck's top runs straight between the DEM at the bridge's own two ends, never the DEM under it,
  which can show the ground from before the bridge (research Part 4e). Its underside is the deck's depth below.
- **Clearance.** Its underside must clear the road under it, as built (the route and any side road's stub), by
  CLEARANCE_M (4.6 m, the minimum for local roads in Australian practice; research Part 4c) everywhere. Otherwise the
  structure is left out with its reason: usually the DEM doesn't show the deck's height, because the bridge's
  approaches are ramps or stairs, not earth, or the DEM predates the bridge.
- **Abutments** stand where the built ground (verge, skirt and terrain) comes up to the underside, walking out from
  the route, or at the bridge's ends: a concrete wall across the deck, from the ground to the underside, with a
  collider (`1WALL_ABUT_...`). One that would stand within ROAD_CLEAR_M of a road's verge leaves the structure out.
- **Piers**, where a span between supports would be longer than the kind's longest span: in the gaps between the
  roads the bridge passes over, each at least ROAD_CLEAR_M beyond a road's verge, as evenly as the gaps allow; each
  one or two columns under a headstock, with a collider (`1WALL_PIER_...`).
- **The deck**: concrete, with the generic barrier and rail (a rail bridge, which also carries ballast and two rails;
  `railway=abandoned` has lost its rails) or a steel railing (a footbridge, narrower). Visual only: AC's AI ignores
  it, and it can never stop a car.
- **Approaches**: the railway or path on from each end of the bridge, on the built ground, for APPROACH_M.

**Under the route's bridge**, for a crossing way passing under a route way tagged as a bridge: the path or railway
on the built ground for UNDER_REACH_M either side of the route's verge, less the route's own ribbon (its road, verge
and skirt: while the route's bridge is an embankment, the line goes in under it). Left out where the route's deck
stands less than UNDER_GAP_M above the line's ground just beyond its ribbon (UNDER_PROBE_M; not the DEM under the
deck, which often bridges a narrow underpass). Visual only, and out of reach behind the bridge.

Ground strips (approaches, and lines under the route) stand on the built ground's highest triangle, a little above
it, so the terrain doesn't show through: a path as a thin slab, a railway as its ballast bed (and rails).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely

from mapcreator.core import crossings as crossings_core
from mapcreator.core import decks, section
from mapcreator.core.centreline import cumulative
from mapcreator.core.scene import Mesh

FORMAT_VERSION = 1

CLEARANCE_M = 4.6  # Q8: a structure over the route is drawn only where its underside clears the road by this
ROAD_CLEAR_M = 1.5  # piers and abutments stand at least this far beyond a road's verge (research Part 4c)
VERGE_M = 2.0  # a road's verge, beyond its carriageway (`mesh.VERGE_M`)
APPROACH_M = 40.0  # the railway or path drawn on from each end of a bridge over the route
UNDER_REACH_M = 40.0  # Q6: what's under the route's bridge is drawn this far either side of the route's verge
UNDER_GAP_M = {"path": 2.4, "railway": 4.0}  # the route's deck must stand this far above the ground beside it
UNDER_PROBE_M = (8.0, 18.0)  # that ground: along the line from this far to this far beyond the route's verge
MAX_DECK_M = 400.0  # a bridge longer than this is cut to this, centred on the route
STEP_M = 2.0  # deck and strip stations along the line
CHECK_STEP_M = 0.5  # clearance sampled this often along the deck, near the route
END_PROBE_M = (0.0, 2.0)  # the deck's height at each end: the DEM at the end and this far on beyond it
MIN_ANGLE_DEG = 20.0  # a crossing flatter than this is measured as if at this angle
ABUT_REACH_M = 0.3  # an abutment stands where the ground comes within this of the underside
ABUT_LENGTH_M = 1.0  # along the deck
ABUT_SPILL_M = 0.6  # wider than the deck by this, across it
COLLIDER_H = 3.0  # colliders reach this high above the ground (a car can't get higher)
STRIP_LIFT_M = 0.05  # a path's slab stands this far above the ground's highest point across it
STRIP_DEPTH_M = 0.6  # and reaches this far below, into the ground

LINE_KINDS = ("railway", "path")
TRACKED = ("rail", "light_rail", "narrow_gauge", "disused")  # railways that still have their rails
STANDARD_GAUGE_M = 1.435


@dataclass(frozen=True)
class Spec:
    """A kind of structure over the route: its deck's width and depth, its longest span, and its piers."""

    kind: str  # "footbridge" or "railbridge"
    depth_m: float
    max_span_m: float
    columns: int  # columns per pier
    column_m: float  # each column's side
    headstock: tuple[float, float]  # (along the deck, depth under it)
    ramp_grade: float  # beyond a level deck set by a signed clearance, it falls at this grade to its ends' ground


# A footbridge's ramps at 1:14, the steepest an accessible ramp may be (AS 1428.1); a railway at 1 in 40 (estimate).
FOOTBRIDGE = Spec("footbridge", depth_m=0.9, max_span_m=35.0, columns=1, column_m=0.6, headstock=(0.7, 0.6),
                  ramp_grade=1 / 14)
RAILBRIDGE = Spec("railbridge", depth_m=1.4, max_span_m=30.0, columns=2, column_m=0.9, headstock=(1.0, 1.0),
                  ramp_grade=1 / 40)
LEVEL_MARGIN_M = 2.0  # a deck set by a signed clearance is level this far beyond the road under it
RAIL_DECK_M = 5.2  # a single track's deck between its barriers' outer faces (estimate)
# A path's walking width by class where it has no `width` tag (estimate).
PATH_WIDTH_M = {"footway": 2.0, "path": 2.0, "steps": 1.5, "cycleway": 3.0, "bridleway": 3.0, "pedestrian": 3.0}
PATH_WIDTH_RANGE = (1.2, 8.0)
RAILING_SPACE_M = 0.15  # each side of a footbridge's walkway, for its railing
SEALED_PATH = frozenset({"asphalt", "paved", "chipseal", "sealed"})
CONCRETE_PATH = frozenset({"concrete", "paving_stones", "sett", "concrete:plates", "concrete:lanes"})
UNSEALED_PATH = frozenset({"unpaved", "gravel", "fine_gravel", "compacted", "dirt", "earth", "ground", "grass", "mud",
                           "sand", "woodchips", "pebblestone", "rock"})

# Materials, all visual, and their textures (`core.textures`): the concrete is the roundabouts' kerb texture, as the
# crossing roads' decks have it (stage 2b); the ballast its own; the steel is the guard rails' `rail`.
CONCRETE = "bridge_concrete"
BALLAST = "ballast"
STEEL = "rail"
TEXTURES = {CONCRETE: "kerb", BALLAST: "ballast", "path_asphalt": "asphalt", "path_ground": "ground"}


# -- what crosses: the route stage ----------------------------------------------------------------------------

def kind_of(tags: dict) -> str:
    """"path" for a way with a path's `highway` (a rail trail too), else "railway"."""
    return "path" if tags.get("highway") else "railway"


@dataclass
class LineWay:
    """A railway or path way, for `structure`: its OSM node ids and points in the region's CRS, and its tags."""

    nodes: list[int]
    xy: np.ndarray
    tags: dict


def find(net, route_xy: np.ndarray, route_seg_way: np.ndarray, *, to_lonlat=None) -> dict:
    """The railways and paths crossing the route: `core.crossings.find` over `net`, a network (`graph.RoadGraph`)
    of the route's own ways and the railways and paths, so a node they share is a junction there as it is for roads.
    Each record gains `kind` ("railway" or "path"), `railway`, and every tag the way has (the index keeps few)."""
    found = crossings_core.find(net, route_xy, route_seg_way, to_lonlat=to_lonlat)
    by_id = {w["id"]: w.get("tags") or {} for w in net.ways}
    for c in found["crossings"] + found["unresolved"]:
        tags = by_id.get(c["way_id"], {})
        c["kind"], c["railway"], c["tags"] = kind_of(tags), tags.get("railway"), dict(tags)
    return found


def _end_dir(xy: np.ndarray, at_start: bool) -> np.ndarray:
    """The unit direction leaving a line at one end, away from it."""
    d = xy[0] - xy[1] if at_start else xy[-1] - xy[-2]
    return d / max(float(np.hypot(*d)), 1e-9)


def _next_way(ways: dict[int, LineWay], ends: dict[int, list[int]], node: int, came_from: int, heading: np.ndarray,
              kind: str, bridge: bool | None) -> tuple[int, bool] | None:
    """The way that carries a line on through `node`: one that starts or ends there, of the same kind (and, if
    `bridge` is given, a bridge or not as asked), turning least. Returns (way id, whether it runs backwards)."""
    best, best_turn = None, 60.0
    for wid in ends.get(node, []):
        if wid == came_from:
            continue
        w = ways[wid]
        if kind_of(w.tags) != kind or len(w.xy) < 2:
            continue
        if bridge is not None and crossings_core.is_bridge(w.tags) != bridge:
            continue
        backwards = w.nodes[-1] == node
        xy = w.xy[::-1] if backwards else w.xy
        d = xy[1] - xy[0]
        cos = float(heading @ d) / max(float(np.hypot(*d)), 1e-9)
        turn = math.degrees(math.acos(min(max(cos, -1.0), 1.0)))
        if turn < best_turn:
            best, best_turn = (wid, backwards), turn
    return best


def _ends_index(ways: dict[int, LineWay]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for wid, w in ways.items():
        if len(w.nodes) >= 2:
            for n in {w.nodes[0], w.nodes[-1]}:
                out.setdefault(n, []).append(wid)
    return out


def _walk(ways, ends, wid: int, at_start: bool, reach_m: float, bridge: bool | None
          ) -> tuple[list[np.ndarray], int, int, bool]:
    """The line on from one end of a way for up to `reach_m`, through the ways that carry it on (`_next_way`).
    Returns its points in order away from the way (the way's end point first), how many ways it took, and the last
    way reached with whether the line's far end is that way's first node (the way itself, at `at_start`, if none)."""
    w = ways[wid]
    node = w.nodes[0] if at_start else w.nodes[-1]
    heading = _end_dir(w.xy, at_start)
    kind = kind_of(w.tags)
    pts, got, came, used, far_first = [w.xy[0] if at_start else w.xy[-1]], 0.0, wid, 0, at_start
    seen = {wid}
    while got < reach_m:
        nxt = _next_way(ways, ends, node, came, heading, kind, bridge)
        if nxt is None or nxt[0] in seen:
            break
        came, backwards = nxt
        seen.add(came)
        used += 1
        nw = ways[came]
        xy = nw.xy[::-1] if backwards else nw.xy
        for q in xy[1:]:
            pts.append(q)
            got += float(np.hypot(*(pts[-1] - pts[-2])))
            if got >= reach_m:
                break
        far_first = backwards  # walking a way backwards ends at its first node
        node = nw.nodes[0] if backwards else nw.nodes[-1]
        heading = _end_dir(xy, False)
    return pts, used, came, far_first


def _cut(line: np.ndarray, from_m: float, to_m: float) -> np.ndarray:
    """The part of a polyline between two distances along it."""
    s = cumulative(line)
    lo, hi = max(from_m, 0.0), min(to_m, float(s[-1]))
    inner = line[(s > lo) & (s < hi)]
    ends = [np.array([np.interp(m, s, line[:, 0]), np.interp(m, s, line[:, 1])]) for m in (lo, hi)]
    return np.vstack([ends[0][None], inner, ends[1][None]])


def structure(ways: dict[int, LineWay], way_id: int, at) -> dict:
    """The bridge a crossing way is part of, and what's on from it.

    - `deck`: the way joined end to end with the bridge ways of its kind that carry it on (`_next_way`), at most
      MAX_DECK_M, centred on the crossing point `at` if longer; None for a way that isn't a bridge.
    - `deck_at_m`: where the crossing is along it; `ways`: how many ways it took.
    - `approaches`: the line on from each end of the deck for APPROACH_M, [from its start, from its end], each
      starting at the deck's end; none at an end the deck was cut short of.
    """
    w = ways[way_id]
    if not crossings_core.is_bridge(w.tags):
        return {"deck": None}
    ends = _ends_index(ways)
    back = _walk(ways, ends, way_id, True, MAX_DECK_M, True)
    ahead = _walk(ways, ends, way_id, False, MAX_DECK_M, True)
    deck = np.vstack([np.array(back[0][::-1]), w.xy[1:-1], np.array(ahead[0])])
    deck = deck[np.r_[True, np.hypot(*np.diff(deck, axis=0).T) > 1e-6]]
    length = float(cumulative(deck)[-1])
    deck_at = float(shapely.LineString(deck).project(shapely.Point(np.asarray(at, float))))
    lo, hi = 0.0, length
    if length > MAX_DECK_M:
        lo = min(max(deck_at - MAX_DECK_M / 2, 0.0), length - MAX_DECK_M)
        hi = lo + MAX_DECK_M
        deck, deck_at = _cut(deck, lo, hi), deck_at - lo
    approaches: list[list] = [[], []]
    for i, (_, _, last, far_first) in enumerate((back, ahead)):
        if (i == 0 and lo > 0) or (i == 1 and hi < length):
            continue
        line = _walk(ways, ends, last, far_first, APPROACH_M, None)[0]
        if len(line) > 1:
            approaches[i] = np.round(_cut(np.array(line), 0.0, APPROACH_M), 2).tolist()
    return {"deck": np.round(deck, 2).tolist(), "deck_at_m": round(deck_at, 1), "ways": 1 + back[1] + ahead[1],
            "approaches": approaches}


def roads_crossed(line: np.ndarray, net, *, route_ways=()) -> list[dict]:
    """Where a line crosses the ways of a road network (`graph.RoadGraph`), along it: `line_at_m`, `way_id`,
    `angle_deg`, `route` (one of the route's own ways), and the road's `highway`, `lanes`, `width`, `oneway` and
    `narrow` tags (what `section.width_from_tags` needs), for keeping piers and abutments off them, and `maxheight`
    (a signed clearance under the same bridge)."""
    line = np.asarray(line, float)
    if len(line) < 2 or not len(net.seg_a):
        return []
    a, b = np.asarray(net.xy)[net.seg_a], np.asarray(net.xy)[net.seg_b]
    segs = shapely.linestrings(np.stack([a, b], axis=1))
    tree = shapely.STRtree(segs)
    s = cumulative(line)
    out, seen = [], set()
    for j in range(len(line) - 1):
        piece = shapely.LineString(line[j:j + 2])
        for k in tree.query(piece, predicate="intersects"):
            hit = shapely.intersection(piece, segs[k])
            if shapely.get_type_id(hit) != 0:
                continue
            p = np.array(shapely.get_coordinates(hit)[0])
            wi = int(net.seg_way[k])
            at = float(s[j] + np.hypot(*(p - line[j])))
            key = (wi, round(at, 1))
            if key in seen:
                continue
            seen.add(key)
            u, v = line[j + 1] - line[j], b[k] - a[k]
            cos = abs(float(u @ v)) / max(float(np.hypot(*u) * np.hypot(*v)), 1e-12)
            tags = net.ways[wi].get("tags") or {}
            out.append({"line_at_m": round(at, 2), "way_id": net.ways[wi]["id"],
                        "angle_deg": round(math.degrees(math.acos(min(cos, 1.0))), 1),
                        "route": net.ways[wi]["id"] in route_ways,
                        "tags": {t: tags[t] for t in ("highway", "lanes", "width", "oneway", "narrow", "maxheight")
                                 if t in tags}})
    return sorted(out, key=lambda r: r["line_at_m"])


# -- what's drawn: the mesh stage -----------------------------------------------------------------------------

@dataclass
class Built:
    """What `build` made: the meshes, a record per railway or path crossing (drawn or left out, and why), and stats."""

    meshes: list[Mesh] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    materials: set[str] = field(default_factory=set)

    @property
    def stats(self) -> dict:
        drawn = [r for r in self.records if r["drawn"]]
        out = {"crossings": len(self.records), "drawn": len(drawn), "left_out": len(self.records) - len(drawn)}
        for what in ("footbridge", "railbridge", "path_under", "railway_under"):
            n = sum(r["structure"] == what for r in drawn)
            if n:
                out[what + "s"] = n
        out["piers"] = sum(r.get("piers", 0) for r in drawn)
        return out


def gauge_m(tags: dict) -> float:
    """The track gauge in metres from `gauge` (millimetres; the first of several), else standard gauge."""
    try:
        g = float(str(tags.get("gauge") or "").split(";")[0]) / 1000.0
    except ValueError:
        return STANDARD_GAUGE_M
    return g if 0.5 <= g <= 2.0 else STANDARD_GAUGE_M


def path_width_m(tags: dict) -> float:
    """A path's walking width: its `width` tag (within PATH_WIDTH_RANGE), else its class's."""
    w = section._parse_metres(tags.get("width"))
    if w is not None and PATH_WIDTH_RANGE[0] <= w <= PATH_WIDTH_RANGE[1]:
        return w
    return PATH_WIDTH_M.get(tags.get("highway", ""), 2.0)


def path_material(tags: dict) -> str:
    """A path's surface: sealed (asphalt), concrete, or unsealed (bare ground), from `surface`, else its class:
    footways and steps concrete, cycleways asphalt, other paths bare ground."""
    v = str(tags.get("surface") or "").strip().lower()
    if v in SEALED_PATH:
        return "path_asphalt"
    if v in CONCRETE_PATH or v.startswith("concrete"):
        return CONCRETE
    if v in UNSEALED_PATH:
        return "path_ground"
    hw = tags.get("highway", "")
    return {"footway": CONCRETE, "steps": CONCRETE, "pedestrian": CONCRETE, "cycleway": "path_asphalt"}.get(
        hw, "path_ground")


class _Route:
    """The route's centreline for distances across it: the nearest station to a point, and how far off it is."""

    def __init__(self, c, sec):
        self.xy = np.asarray(c["xy"], float)
        self.s = np.asarray(c["s"], float)
        self.z = np.asarray(c["z"], float)
        self.fine = np.asarray(c["fine"], bool) if "fine" in c else np.ones(len(self.s), bool)  # LiDAR ground
        self.half = np.maximum(np.asarray(sec["half_left"], float), np.asarray(sec["half_right"], float))
        self.line = shapely.LineString(self.xy)

    def station(self, p) -> int:
        return int(np.argmin(np.hypot(*(self.xy - np.asarray(p, float)).T)))


def _sin(angle_deg: float) -> float:
    return math.sin(math.radians(max(float(angle_deg), MIN_ANGLE_DEG)))


def _keep_out(roads: list[dict], route: _Route, deck_line: np.ndarray, reach: float) -> list[tuple[float, float]]:
    """The stretches along a deck where nothing may stand: each road it passes over, its carriageway, verge and
    ROAD_CLEAR_M, plus `reach` (half a support's size), measured along the deck at the crossing's angle."""
    out = []
    for r in roads:
        if r.get("route"):
            k = route.station(_point_at(deck_line, r["line_at_m"]))
            half = float(route.half[k])
        else:
            half = section.width_from_tags(r.get("tags") or {})[0] / 2
        ext = (half + VERGE_M + ROAD_CLEAR_M + reach) / _sin(r["angle_deg"])
        out.append((r["line_at_m"] - ext, r["line_at_m"] + ext))
    return out


def _point_at(line: np.ndarray, at: float) -> np.ndarray:
    s = cumulative(line)
    return np.array([np.interp(at, s, line[:, 0]), np.interp(at, s, line[:, 1])])


def _inside(at: float, spans: list[tuple[float, float]]) -> bool:
    return any(a <= at <= b for a, b in spans)


def _end_height(dem_sample, deck_line: np.ndarray, at_start: bool) -> float:
    """The deck's height at one end: the DEM at the end and END_PROBE_M on beyond it, averaged (NaN-safe)."""
    end = deck_line[0] if at_start else deck_line[-1]
    out = _end_dir(deck_line, at_start)
    pts = np.array([end + out * d for d in END_PROBE_M])
    h = np.asarray(dem_sample(pts[:, 0], pts[:, 1]), float)
    return float(np.nanmean(h)) if np.isfinite(h).any() else float("nan")


def _piers(a: float, b: float, max_span: float, spans: list[tuple[float, float]]) -> list[float]:
    """Pier stations between supports at `a` and `b`, so no span is longer than `max_span` where the gaps between
    the keep-out stretches allow: as few as that takes, spread as evenly as the free places allow. A span that can't
    be split (no room for a pier) is left long."""
    if b - a <= max_span:
        return []
    free = [float(x) for x in np.arange(math.ceil(a) + 1.0, b - 1.0 + 1e-9, 1.0) if not _inside(float(x), spans)]
    greedy, at = [], a
    while b - at > max_span:
        reach = [x for x in free if at < x <= at + max_span]
        beyond = [x for x in free if x > at]
        if not reach and not beyond:
            break
        at = reach[-1] if reach else beyond[0]  # past a keep-out stretch: the next place a pier may stand
        greedy.append(at)
    if not greedy:
        return []
    longest = float(np.diff([a, *greedy, b]).max())
    even = [min(free, key=lambda x: abs(x - e)) for e in np.linspace(a, b, len(greedy) + 2)[1:-1]]
    if len(set(even)) == len(even) and float(np.diff([a, *even, b]).max()) <= max(longest, max_span):
        return even
    return greedy


def _gaps(xs: np.ndarray, contact: np.ndarray, min_m: float = 3.0) -> list[tuple[float, float]]:
    """The stretches along a deck where the ground stays clear of it, at least `min_m` long, each from where the
    ground last meets the deck (or the deck's end) to where it meets it again (or the deck's other end)."""
    out = []
    for i0, i1 in _true_runs(~contact):
        a = float(xs[i0 - 1]) if i0 > 0 else float(xs[0])
        b = float(xs[i1]) if i1 < len(xs) else float(xs[-1])
        if b - a >= min_m:
            out.append((a, b))
    return out


def _left_out(rec: dict, why: str, code: str = "", **extra) -> dict:
    """A structure left out: `why` in words, and `code` "dem" where the DEM can't place its deck (the build's
    warning then takes `core.dem_age`'s shared wording)."""
    return {**rec, "drawn": False, "reason": why, **({"reason_code": code} if code else {}), **extra}


def _base(c: dict, structure: str) -> dict:
    tags = c.get("tags") or {}
    return {"crossing": c["id"], "way_id": c["way_id"], "structure": structure, "relation": c["relation"],
            "kind": c.get("kind", kind_of(tags)), "railway": c.get("railway"), "highway": c.get("highway"),
            "name": c.get("name"), "at_m": c["at_m"]}


def build(crossings: list[dict], c, sec, frame, dem_sample, ground, road, footprint: shapely.Geometry | None
          ) -> Built:
    """The structures for the railways and paths crossing the route (`crossing_lines.json`'s `crossings`).

    `c` and `sec` are the centreline and section arrays; `dem_sample(e, n)` the DEM; `ground(e, n)` the built
    ground's height (verge, skirt and terrain: `stub_closures.TopSurface`), and `road(e, n)` the built road's (the
    route's and its stubs'), NaN where there's none; `footprint` the route's ribbon in plan (its road, verge and skirt,
    and its stubs'), which ground strips stay out of."""
    route = _Route(c, sec)
    out = Built()
    done: set = set()
    for cr in crossings:
        deck = (cr.get("structure") or {}).get("deck")
        # A way crossing the route twice, or two ways of one bridge each crossing it (split over the road): one.
        key = (cr["relation"], tuple(map(tuple, deck)) if deck else cr["way_id"])
        if key in done:
            continue
        done.add(key)
        if cr["relation"] == "over":
            rec = _over(cr, route, frame, dem_sample, ground, road, footprint, out)
        else:
            rec = _under(cr, route, frame, dem_sample, ground, footprint, out)
        out.records.append(rec)
    return out


def _over(cr: dict, route: _Route, frame, dem_sample, ground, road, footprint, out: Built) -> dict:
    tags = cr.get("tags") or {}
    kind = cr.get("kind", kind_of(tags))
    spec = RAILBRIDGE if kind == "railway" else FOOTBRIDGE
    rec = _base(cr, spec.kind)
    if cr["route"].get("tunnel"):
        return _left_out(rec, "the route is in a tunnel there")
    st = cr.get("structure") or {}
    if not cr.get("bridge") or not st.get("deck"):
        return _left_out(rec, "not mapped as a bridge")
    deck_line = np.asarray(st["deck"], float)
    deck_len = float(cumulative(deck_line)[-1])
    width = RAIL_DECK_M if kind == "railway" else path_width_m(tags) + 2 * RAILING_SPACE_M
    za, zb = _end_height(dem_sample, deck_line, True), _end_height(dem_sample, deck_line, False)
    level: dict = {}  # with a signed clearance: the level deck's height and the stretch it's level over

    def top(at):
        at = np.asarray(at, float)
        straight = za + (zb - za) * np.clip(at / max(deck_len, 1e-9), 0.0, 1.0)
        if not level:
            return straight
        lo, hi = level["over"]
        ramp = level["top"] - spec.ramp_grade * (np.maximum(lo - at, 0.0) + np.maximum(at - hi, 0.0))
        return np.fmax(ramp, straight)

    def underside(at):
        return top(at) - spec.depth_m

    # The road under the deck as built (the route's and its stubs'), sampled densely along the deck and across it.
    xs = np.arange(0.0, deck_len + 1e-9, CHECK_STEP_M)
    pts = _points_at(deck_line, xs)
    _, left = decks.frames(pts)
    under_at, under_h = [], []
    for off in (-width / 2, 0.0, width / 2):
        q = pts + left * off
        h = np.asarray(road(q[:, 0], q[:, 1]), float)
        ok = np.isfinite(h)
        under_at.append(xs[ok])
        under_h.append(h[ok])
    under_at, under_h = np.concatenate(under_at), np.concatenate(under_h)
    if not len(under_h):
        return _left_out(rec, "it doesn't pass over the built road")
    signed, signed_on = _signed(cr)
    rec |= {"deck_m": round(deck_len, 1), "ways": st.get("ways", 1)}
    if signed is not None:
        rec |= {"maxheight_m": signed, "maxheight_on": signed_on}
    clearance = round(float((underside(under_at) - under_h).min()), 2) if np.isfinite([za, zb]).all() else None
    rec |= {"clearance_m": clearance, "height_from": "dem_ends"}
    if (clearance is None or clearance < CLEARANCE_M) and signed is not None and signed >= CLEARANCE_M:
        # The bare-earth DEM doesn't show the deck (ramps or stairs at its ends, or data older than the bridge), but
        # the road under it is signed for its clearance: level over the road at that clearance, and falling at a ramp's
        # grade beyond it, to the straight line between its ends' ground.
        level |= {"top": float(under_h.max()) + signed + spec.depth_m,
                  "over": (float(under_at.min()) - LEVEL_MARGIN_M, float(under_at.max()) + LEVEL_MARGIN_M)}
        rec |= {"clearance_m": round(float((underside(under_at) - under_h).min()), 2), "height_from": "maxheight",
                "dem_clearance_m": clearance}
        clearance = rec["clearance_m"]
    if clearance is None:
        return _left_out(rec, "no elevation data at the bridge's ends")
    rec["fine"] = bool(route.fine[route.station(cr["xy"])])
    if clearance < CLEARANCE_M:
        if signed is not None:
            return _left_out(rec, f"its underside would clear the road by {clearance:.1f} m (the road is signed for "
                                  f"{signed:g} m there), under {CLEARANCE_M} m")
        return _left_out(rec, f"its underside would clear the road by {clearance:.1f} m, under {CLEARANCE_M} m: the "
                              "elevation data doesn't show the deck's height (its approaches are ramps or stairs, or "
                              "the data is older than the bridge), and the road isn't signed for a clearance", "dem")
    # Supports: under each stretch where the ground stays clear of the deck, an abutment at each end (where the
    # ground meets the deck, or at the deck's end) and piers between, all off the roads the bridge passes over.
    roads = _on_deck(cr.get("roads_crossed") or [], cr, float(st["deck_at_m"]))
    xs1 = np.arange(0.0, deck_len + 1e-9, 1.0)
    g1 = np.asarray(ground(*_points_at(deck_line, xs1).T), float)
    contact = np.isfinite(g1) & (g1 >= underside(xs1) - ABUT_REACH_M)
    abut_out = _keep_out(roads, route, deck_line, ABUT_LENGTH_M / 2)
    pier_out = _keep_out(roads, route, deck_line, spec.headstock[0] / 2)
    supports, piers = [], []
    for a, b in _gaps(xs1, contact):
        supports += [a, b]
        piers += _piers(a, b, spec.max_span_m, pier_out)
    if not supports:
        return _left_out(rec, "the ground comes up to the deck all along it")
    if any(_inside(x, abut_out) for x in supports):
        return _left_out(rec, "the ground comes up to the deck beside a road, with no room there for an abutment")

    prefix, short = ("RAILBRIDGE", "RB") if kind == "railway" else ("FOOTBRIDGE", "FB")
    name = f"{prefix}_{cr['id']}"
    line = decks.stations(deck_line, STEP_M)
    line_top = top(cumulative(line))
    concrete = [decks.deck(line, line_top, width, spec.depth_m)]
    steel: list[decks.Piece] = []
    stone: list[decks.Piece] = []
    if kind == "railway":
        c_parts, s_parts = decks.barrier(line, line_top, width)
        concrete += c_parts
        steel += s_parts
        stone.append(decks.ballast(line, line_top))
        if (cr.get("railway") or tags.get("railway")) in TRACKED:
            steel += decks.running_rails(line, line_top + decks.BALLAST[2], gauge_m(tags))
    else:
        steel += decks.railing(line, line_top, width)
    abut_walls, pier_walls = [], []
    for x in sorted(set(supports)):
        p, h = _point_at(deck_line, x), _heading_at(deck_line, x)
        u = float(underside(x))
        g = _ground_under(ground, p, h, ABUT_LENGTH_M, width + ABUT_SPILL_M)
        g = g if np.isfinite(g) else u - ABUT_REACH_M
        concrete.append(decks.box(p, h, ABUT_LENGTH_M, width + ABUT_SPILL_M, min(g, u - ABUT_REACH_M) - 1.0, u))
        if u - g > 0.3:
            abut_walls.append(decks.collider_box(p, h, ABUT_LENGTH_M, width + ABUT_SPILL_M, g - 0.3,
                                                 min(u, g + COLLIDER_H)))
    n_piers = 0
    hs_along, hs_depth = spec.headstock
    for x in piers:
        p, h = _point_at(deck_line, x), _heading_at(deck_line, x)
        u = float(underside(x))
        g = _ground_under(ground, p, h, hs_along, width)
        if not np.isfinite(g) or u - hs_depth - g < 0.5:
            continue  # the ground is (nearly) up to the deck here: no pier needed
        concrete.append(decks.box(p, h, hs_along, width + 0.4, u - hs_depth, u))
        side = np.array([-h[1], h[0]])
        offsets = [0.0] if spec.columns == 1 else [-(width / 2 - spec.column_m), width / 2 - spec.column_m]
        for o in offsets:
            q = p + side * o
            concrete.append(decks.box(q, h, spec.column_m, spec.column_m, g - 0.5, u - hs_depth))
            pier_walls.append(decks.collider_box(q, h, spec.column_m, spec.column_m, g - 0.3,
                                                 min(g + COLLIDER_H, u - hs_depth)))
        n_piers += 1

    approach_m = 0.0
    strips: dict[str, list[decks.Piece]] = {}
    for line_xy in st.get("approaches") or []:
        if len(line_xy) >= 2:
            approach_m += _strip(kind, tags, np.asarray(line_xy, float), ground, footprint, strips)

    out.meshes += decks.meshes(f"{name}_DECK", CONCRETE, concrete, frame)
    out.materials.add(CONCRETE)
    if steel:
        out.meshes += decks.meshes(f"{name}_STEEL", STEEL, steel, frame)
        out.materials.add(STEEL)
    if stone:
        out.meshes += decks.meshes(f"{name}_BALLAST", BALLAST, stone, frame)
        out.materials.add(BALLAST)
    for material, pieces in strips.items():
        out.meshes += decks.meshes(f"{name}_ON_{material.upper()}", material, pieces, frame)
        out.materials.add(material)
    if abut_walls:
        out.meshes += decks.meshes(f"1WALL_ABUT_{short}{cr['id']}", STEEL, abut_walls, frame, renderable=False)
    if pier_walls:
        out.meshes += decks.meshes(f"1WALL_PIER_{short}{cr['id']}", STEEL, pier_walls, frame, renderable=False)
    return {**rec, "drawn": True, "reason": None, "width_m": round(width, 2),
            "top_m": [round(float(top(0.0)), 2), round(float(top(deck_len)), 2)],
            "supports_m": [round(x, 1) for x in sorted(set(supports))], "piers": n_piers,
            "approach_m": round(approach_m, 1)}


def _signed(cr: dict) -> tuple[float | None, str | None]:
    """The clearance the road under a bridge is signed for (`maxheight`): the route's own, else the lowest of the
    other roads under the same bridge (the other carriageway's, usually), and which it was."""
    own = section._parse_metres(cr["route"].get("maxheight"))
    if own is not None:
        return own, "route"
    others = [m for r in cr.get("roads_crossed") or [] if not r.get("route")
              if (m := section._parse_metres((r.get("tags") or {}).get("maxheight"))) is not None]
    return (min(others), "another road under the bridge") if others else (None, None)


def _points_at(line: np.ndarray, at: np.ndarray) -> np.ndarray:
    s = cumulative(line)
    return np.column_stack([np.interp(at, s, line[:, 0]), np.interp(at, s, line[:, 1])])


def _on_deck(roads: list[dict], cr: dict, deck_at: float) -> list[dict]:
    """The roads a bridge passes over, as distances along its deck (the route stage measured them along it,
    `roads_crossed`); the route at `deck_at` if they don't include it."""
    out = [dict(r) for r in roads]
    if not any(r.get("route") for r in out):
        out.append({"line_at_m": deck_at, "route": True, "angle_deg": cr.get("angle_deg", 90.0), "tags": {}})
    return out


def _heading_at(line: np.ndarray, at: float) -> np.ndarray:
    s = cumulative(line)
    j = int(np.clip(np.searchsorted(s, at) - 1, 0, len(line) - 2))
    d = line[j + 1] - line[j]
    return d / max(float(np.hypot(*d)), 1e-9)


def _ground_under(ground, p: np.ndarray, h: np.ndarray, length: float, width: float) -> float:
    """The lowest built ground under a footprint (its corners and centre), NaN where there's none."""
    side = np.array([-h[1], h[0]])
    pts = np.array([p + h * a * length / 2 + side * b * width / 2 for a in (-1, 0, 1) for b in (-1, 0, 1)])
    g = np.asarray(ground(pts[:, 0], pts[:, 1]), float)
    return float(np.nanmin(g)) if np.isfinite(g).any() else float("nan")


def _strip(kind: str, tags: dict, line: np.ndarray, ground, footprint, strips: dict) -> float:
    """A railway or path on the built ground along `line`, less the route's footprint: its pieces added to `strips`
    by material. Returns the length drawn."""
    geom = shapely.LineString(line)
    if footprint is not None and not footprint.is_empty:
        geom = geom.difference(footprint.buffer(0.5))
    parts = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "LineString" and g.length >= 3.0]
    width = decks.BALLAST[1] if kind == "railway" else path_width_m(tags)
    got = 0.0
    for part in parts:
        pts = decks.stations(np.asarray(part.coords), 1.0)
        if len(pts) < 2:
            continue
        _, left = decks.frames(pts)
        h = np.column_stack([np.asarray(ground(*(pts + left * o).T), float) for o in (-width / 2, 0.0, width / 2)])
        ok = np.isfinite(h).all(axis=1)
        for i0, i1 in _true_runs(ok):
            if i1 - i0 < 3:
                continue
            seg, hh = pts[i0:i1], h[i0:i1]
            base = hh.max(axis=1)
            if kind == "railway":
                strips.setdefault(BALLAST, []).append(decks.ballast(seg, base + 0.02, below_m=STRIP_DEPTH_M))
                if tags.get("railway") in TRACKED:
                    strips.setdefault(STEEL, []).extend(
                        decks.running_rails(seg, base + 0.02 + decks.BALLAST[2], gauge_m(tags)))
            else:
                w = width / 2
                strips.setdefault(path_material(tags), []).append(
                    decks.sweep(seg, base + STRIP_LIFT_M, ((-w, -STRIP_DEPTH_M), (w, -STRIP_DEPTH_M), (w, 0.0),
                                                           (-w, 0.0))))
            got += float(cumulative(seg)[-1])
    return got


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _under(cr: dict, route: _Route, frame, dem_sample, ground, footprint, out: Built) -> dict:
    tags = cr.get("tags") or {}
    kind = cr.get("kind", kind_of(tags))
    rec = _base(cr, "railway_under" if kind == "railway" else "path_under")
    if not cr["route"].get("bridge"):
        return _left_out(rec, "the route isn't on a bridge there (an underpass under an embankment)")
    k = route.station(cr["xy"])
    line = np.asarray(cr["line"], float)
    sin = _sin(cr.get("angle_deg", 90.0))
    # The line's ground beside the route, where its cutting or valley shows: a DEM often bridges the ground under a
    # narrow deck. Each side's median over UNDER_PROBE_M[1] m from just beyond the route's ribbon; the lower side.
    near, far = ((float(route.half[k]) + VERGE_M + d) / sin for d in UNDER_PROBE_M)
    sides = []
    for sign in (-1.0, 1.0):
        at = cr["line_at_m"] + sign * np.linspace(near, far, 11)
        q = _points_at(line, at)
        h = np.asarray(dem_sample(q[:, 0], q[:, 1]), float)
        if np.isfinite(h).any():
            sides.append(float(np.nanmedian(h)))
    gap = round(float(route.z[k]) - min(sides), 2) if sides else None
    rec["gap_m"] = gap
    if gap is not None and gap < UNDER_GAP_M[kind]:
        return _left_out(rec, f"the route's bridge stands only {gap:.1f} m above the ground beside it")
    reach = (float(route.half[k]) + VERGE_M) / sin + UNDER_REACH_M
    part = _cut(line, cr["line_at_m"] - reach, cr["line_at_m"] + reach)
    strips: dict[str, list[decks.Piece]] = {}
    got = _strip(kind, tags, part, ground, footprint, strips)
    if not strips:
        return _left_out(rec, "no ground beside the route to draw it on")
    name = f"{'RAILWAY' if kind == 'railway' else 'PATH'}_UNDER_{cr['id']}"
    for material, pieces in strips.items():
        out.meshes += decks.meshes(f"{name}_{material.upper()}", material, pieces, frame)
        out.materials.add(material)
    return {**rec, "drawn": True, "reason": None, "strip_m": round(got, 1)}
