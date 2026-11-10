"""Side-road junctions and roundabouts on the route, and keeping roadside objects out of the side roads' mouths.

**Which junctions count** (`find`, run by the route stage). A side road leaving the route at one of the route's
nodes is a junction when:
- its way is a class the router may drive (`DRIVABLE`: motorway to living_street, `road` and the links). The road
  data holds nothing else, so driveways, service roads and tracks never are;
- it isn't private: of `motor_vehicle`, `vehicle` and `access`, the most specific one tagged isn't `private` or `no`;
- it leaves at least `MIN_ANGLE_DEG` off the mainline, measured `ANGLE_AT_M` out along it, against both the road
  ahead and the road behind;
- it's clear of the rest of the route: followed outwards, it stops where it meets another part of the route, and one
  that does so within `MIN_LEG_M` isn't a junction;
- it has no `bridge`, `tunnel` or `layer` other than 0 within its first `GRADE_FREE_M`: a road passing over or
  under isn't a junction.

Each side road is followed out to `LEG_M`, or to where it ends or meets the route: along its own way, then, at the
way's end, onto the straightest way that turns less than `CONTINUE_DEG`. It carries a closure level for the closures
to come: limited by its access tags (`LIMITED_ACCESS`), else major or minor by its class (`closure_level`).

Roundabouts (`junction=roundabout` or `circular`) the route drives or touches are recorded with their ring and their
exits, and left as they are: their exits aren't junctions here, and nothing is cleared at them.

**Keeping the mouths clear** (`mouths`, `masks`, `zone`; the mesh, scenery and track stages). A junction's mouth is
the stretch of the mainline where the side road's carriageway crosses the band from the road's edge to
`MOUTH_BAND_M` beyond it (where guard rails, guide posts and signs stand), plus `MOUTH_MARGIN_M` either side. No guard
rail, guide post, curve marker or sign stands in it, and no timing gate. Trees, buildings and power-line supports are
kept off the side road for its first `LEG_M` as they are kept off the mainline: the side road's own half width plus
the mainline's clearance for each.

The format of `01_route/junctions.json` is in `docs/ac-format.md` ("Junctions"). Coordinates are in the region's
projected CRS, as the route's are.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.spatial import cKDTree

from mapcreator.core import section as sec
from mapcreator.core.centreline import cumulative

FORMAT_VERSION = 1

LEG_M = 100.0  # how far each side road is followed: the stubs to come are this long
MIN_ANGLE_DEG = 15.0
ANGLE_AT_M = 15.0  # the side road's direction is from the junction to its point this far out
MIN_LEG_M = 15.0  # a side road that meets the route again within this isn't a junction
GRADE_FREE_M = 30.0  # no bridge, tunnel or layer within this of the junction
CONTINUE_DEG = 60.0  # at a way's end, the side road carries on along a way turning less than this
# A side road whose point comes within this of another part of the route has met it. "Another part": the road
# distance to the nearest route point is much longer than the straight distance (as the mesh's skirts judge it).
ROUTE_CLEAR_M = 10.0
OTHER_PART_RATIO = 1.4
OTHER_PART_SLACK_M = 5.0
# A way leaving a route node whose first metres lie this close along the route is the route itself: the segment a
# waypoint was spliced into, which runs on past the route's start or end.
ON_ROUTE_M = 0.25
ON_ROUTE_PROBE_M = 2.0

MOUTH_BAND_M = 3.0  # beyond the mainline's edge: guide posts stand at 0.8 m, guard rails at 1 m, signs at 2 m
MOUTH_BAND_STEP_M = 0.5
MOUTH_MARGIN_M = 5.0  # along the mainline, either side of where the side road's carriageway crosses that band

MAJOR = frozenset({"motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link",
                   "primary_link", "secondary_link", "tertiary_link"})
MINOR = frozenset({"unclassified", "residential", "living_street", "road"})
DRIVABLE = MAJOR | MINOR  # the router's classes (`providers.osm.overpass.DRIVABLE`)
PRIVATE = frozenset({"private", "no"})
LIMITED_ACCESS = frozenset({"permit", "forestry", "agricultural", "destination", "delivery", "customers", "military"})
ACCESS_KEYS = ("motor_vehicle", "vehicle", "access")  # most specific first
RING = frozenset({"roundabout", "circular"})
# The side road's tags kept in junctions.json, for the stubs and closures to come.
KEEP_TAGS = ("highway", "name", "ref", "surface", "lanes", "width", "narrow", "lane_markings", "oneway", "maxspeed",
             "access", "motor_vehicle", "vehicle", "junction")
SIDE_CODE = {"left": "L", "right": "R"}


def access_value(tags: dict) -> str | None:
    """The most specific of `motor_vehicle`, `vehicle` and `access` that's tagged, or None."""
    for key in ACCESS_KEYS:
        value = str(tags.get(key) or "").strip().lower()
        if value:
            return value
    return None


def is_private(tags: dict) -> bool:
    return access_value(tags) in PRIVATE


def closure_level(tags: dict) -> str:
    """How the side road would be closed: "limited" (permit, forestry, destination and the like), else "major"
    (trunk to tertiary and their links) or "minor" (the rest)."""
    if access_value(tags) in LIMITED_ACCESS:
        return "limited"
    return "major" if tags.get("highway") in MAJOR else "minor"


def grade_separated(tags: dict) -> bool:
    """A bridge, a tunnel or a layer other than 0: not at the level of the road it meets."""
    for key in ("bridge", "tunnel"):
        if str(tags.get(key) or "no").strip().lower() not in ("no", ""):
            return True
    return str(tags.get("layer") or "0").strip() not in ("0", "")


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _heading(v) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _along(line: np.ndarray, metres: float) -> np.ndarray:
    """The point `metres` along a polyline (its end if shorter)."""
    s = cumulative(line)
    m = float(np.clip(metres, 0.0, s[-1]))
    return np.array([np.interp(m, s, line[:, 0]), np.interp(m, s, line[:, 1])])


def _cut(line: np.ndarray, metres: float) -> np.ndarray:
    """A polyline's first `metres`."""
    s = cumulative(line)
    if metres >= s[-1]:
        return line
    k = int(np.searchsorted(s, metres, side="right"))
    return np.vstack([line[:k], _along(line, metres)[None]])


def _segment_distance(p: np.ndarray, line: np.ndarray) -> float:
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0.0, 1.0)
    return float(np.hypot(*(a + ab * t[:, None] - p).T).min())


@dataclass
class _Leg:
    line: np.ndarray  # from the junction node outwards
    ways: list[int]  # way indices, in order
    entered_m: list[float]  # where along the line each way starts
    nodes: list[int]  # network nodes passed, the junction's first
    node_m: list[float]  # and how far along the line each is
    ends: str  # "length", "dead_end", "branches", "joins" or "meets_route"


class _Network:
    """The road network's adjacency: each node's neighbours, with the way each segment belongs to."""

    def __init__(self, net):
        self.xy = np.asarray(net.xy, dtype=float)
        self.osm_node = np.asarray(net.osm_node)
        self.ways = net.ways
        self.adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for a, b, w in zip(np.asarray(net.seg_a).tolist(), np.asarray(net.seg_b).tolist(),
                           np.asarray(net.seg_way).tolist()):
            self.adj[a].append((b, w))
            self.adj[b].append((a, w))

    def tags(self, way: int) -> dict:
        return self.ways[way].get("tags") or {}

    def is_ring(self, way: int) -> bool:
        return self.tags(way).get("junction") in RING

    def walk(self, start: int, first: int, way: int, stop: set[int], limit: float) -> _Leg:
        """Follow a side road from node `start` through its neighbour `first` (on `way`) for `limit` metres, or to a
        dead end, a branch, or a node in `stop` (the route's)."""
        line = [self.xy[start]]
        ways, entered, nodes, node_m = [way], [0.0], [start], [0.0]
        seen = {way}
        length, prev, cur = 0.0, start, first
        ends = "length"
        while True:
            d = float(np.hypot(*(self.xy[cur] - line[-1])))
            if length + d >= limit:
                f = (limit - length) / max(d, 1e-9)
                line.append(line[-1] + (self.xy[cur] - line[-1]) * f)
                break
            line.append(self.xy[cur])
            length += d
            nodes.append(cur)
            node_m.append(length)
            if cur in stop:
                ends = "meets_route"
                break
            same = [n for n, w in self.adj[cur] if w == way and n != prev]
            if same:
                prev, cur = cur, min(same, key=lambda n: self._turn(line, n))
                continue
            options = [(n, w) for n, w in self.adj[cur] if w not in seen and n != prev]
            if not options:
                ends = "dead_end" if len(self.adj[cur]) <= 1 else "joins"
                break
            turn, n, w = min((self._turn(line, n), n, w) for n, w in options)
            if turn > CONTINUE_DEG:
                ends = "branches"
                break
            seen.add(w)
            ways.append(w)
            entered.append(length)
            way, prev, cur = w, cur, n
        return _Leg(np.array(line), ways, entered, nodes, node_m, ends)

    def _turn(self, line: list, n: int) -> float:
        if len(line) < 2:
            return 0.0
        return abs(_wrap(_heading(self.xy[n] - line[-1]) - _heading(line[-1] - line[-2])))


class _Route:
    """The routed polyline: its stations, the network node at each point (-1 where a waypoint was spliced in), and a
    1 m resampling for the distance tests."""

    def __init__(self, xy: np.ndarray, net: _Network):
        self.xy = np.asarray(xy, dtype=float)
        self.s = cumulative(self.xy)
        index: dict[tuple[float, float], list[int]] = defaultdict(list)
        for i, p in enumerate(net.xy.tolist()):
            index[tuple(p)].append(i)
        found = [index.get(tuple(p), []) for p in self.xy.tolist()]
        node = []
        for i, options in enumerate(found):
            # Two nodes at one place (an OSM mapping slip) are told apart by which one the route's next or last
            # point is joined to.
            near = {n for j in (i - 1, i + 1) if 0 <= j < len(found) for n in found[j]}
            joined = [n for n in options if any(m in near for m, _ in net.adj[n])]
            node.append((joined or options or [-1])[0])
        self.node = np.array(node, dtype=np.int64)
        self.nodes = {int(n) for n in self.node if n >= 0}
        real = [(i, int(n)) for i, n in enumerate(self.node) if n >= 0]
        self.pairs = {frozenset((a, b)) for (i0, a), (i1, b) in zip(real, real[1:]) if i1 - i0 <= 2}
        n = max(int(np.ceil(self.s[-1])), 1)
        self.dense_s = np.linspace(0.0, self.s[-1], n + 1)
        self.dense = np.column_stack([np.interp(self.dense_s, self.s, self.xy[:, 0]),
                                      np.interp(self.dense_s, self.s, self.xy[:, 1])])
        self.tree = cKDTree(self.dense)

    def at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.s[-1]))
        return np.array([np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])])

    def piece(self, s0: float, s1: float) -> np.ndarray:
        """The route between two stations, its own vertices included."""
        s0, s1 = max(s0, 0.0), min(s1, float(self.s[-1]))
        inner = self.xy[(self.s > s0) & (self.s < s1)]
        return np.vstack([self.at(s0)[None], inner, self.at(s1)[None]])

    def along_route(self, node_xy: np.ndarray, toward: np.ndarray, at_m: float) -> bool:
        """Whether the segment from the route node at station `at_m` toward `toward` starts along the route."""
        d = toward - node_xy
        length = float(np.hypot(*d))
        if length < 1e-6:
            return True
        probe = node_xy + d * min(ON_ROUTE_PROBE_M, length / 2) / length
        piece = self.piece(at_m - 2 * ON_ROUTE_PROBE_M, at_m + 2 * ON_ROUTE_PROBE_M)
        return len(piece) >= 2 and _segment_distance(probe, piece) < ON_ROUTE_M

    def other_part_m(self, line: np.ndarray, node_xy: np.ndarray, at_m: float) -> float | None:
        """How far along `line` (leaving the route at station `at_m`) it first comes within ROUTE_CLEAR_M of another
        part of the route, or None."""
        s_line = cumulative(line)
        n = max(int(np.ceil(s_line[-1])), 1)
        ss = np.linspace(0.0, s_line[-1], n + 1)
        pts = np.column_stack([np.interp(ss, s_line, line[:, 0]), np.interp(ss, s_line, line[:, 1])])
        for m, near in zip(ss, self.tree.query_ball_point(pts, ROUTE_CLEAR_M)):
            if not near:
                continue
            near = np.asarray(near)
            straight = np.hypot(*(self.dense[near] - node_xy).T)
            if (np.abs(self.dense_s[near] - at_m) > OTHER_PART_RATIO * straight + OTHER_PART_SLACK_M).any():
                return float(m)
        return None


def _leg_record(net: _Network, leg: _Leg, way: int) -> dict:
    tags = net.tags(way)
    surface = sec.surface_class(tags)
    width, source = sec.width_from_tags(tags, unsealed=surface == sec.UNSEALED)
    return {
        "way_id": net.ways[way]["id"],
        "way_ids": [net.ways[w]["id"] for w in leg.ways],
        "highway": tags.get("highway"),
        "name": tags.get("name"),
        "closure": closure_level(tags),
        "access": access_value(tags),
        "tags": {k: tags[k] for k in KEEP_TAGS if k in tags},
        "surface_class": surface,
        "width_m": round(float(width), 2),
        "width_source": source,
        "length_m": round(float(cumulative(leg.line)[-1]), 1),
        "ends": leg.ends,
        "line": np.round(leg.line, 2).tolist(),
    }


def _grade_separated_near(net: _Network, leg: _Leg) -> bool:
    return any(grade_separated(net.tags(w)) for w, m in zip(leg.ways, leg.entered_m) if m < GRADE_FREE_M)


def find(net, route_xy: np.ndarray, *, leg_m: float = LEG_M, to_lonlat=None) -> dict:
    """The junctions and roundabouts along a routed polyline: the content of junctions.json.

    `net` is the road network the route was found on: node coordinates `xy` and ids `osm_node`, segments
    `seg_a`/`seg_b` with their way's index `seg_way`, and `ways` ({"id", "tags"}), as
    `providers.osm.graph.RoadGraph` holds them. `route_xy` is the route's polyline: its points are network nodes,
    except where a waypoint was spliced in. `to_lonlat`, a transformer, adds each junction's longitude and latitude.
    Side roads that leave the route but don't count are listed under "skipped", with the reason.
    """
    nw = _Network(net)
    rt = _Route(route_xy, nw)
    rings = _rings(nw, rt)
    ring_node = {n for ring in rings for n in ring["nodes"]}
    found, skipped = [], []
    for i in range(1, len(rt.xy) - 1):  # a side road at the route's very end is closed by its end wall
        node = int(rt.node[i])
        if node < 0 or node in ring_node:
            continue  # a spliced waypoint; or a roundabout's node, recorded with the roundabout
        at = float(rt.s[i])
        here = rt.xy[i]
        u_in = here - rt.at(at - ANGLE_AT_M)
        u_out = rt.at(at + ANGLE_AT_M) - here
        u_in /= max(float(np.hypot(*u_in)), 1e-9)
        u_out /= max(float(np.hypot(*u_out)), 1e-9)
        tangent = u_in + u_out if float(np.hypot(*(u_in + u_out))) > 1e-6 else u_out
        for nb, way in nw.adj[node]:
            if frozenset((node, nb)) in rt.pairs or rt.along_route(here, nw.xy[nb], at):
                continue
            tags = nw.tags(way)
            leg = nw.walk(node, nb, way, rt.nodes - {node}, leg_m)
            clip = rt.other_part_m(leg.line, here, at)
            if clip is not None and clip < cumulative(leg.line)[-1]:
                leg.line, leg.ends = _cut(leg.line, clip), "meets_route"
            length = float(cumulative(leg.line)[-1])
            direction = _along(leg.line, ANGLE_AT_M) - here
            angle = _wrap(_heading(direction) - _heading(tangent))
            off = min(abs(_wrap(_heading(direction) - _heading(u_out))),
                      abs(_wrap(_heading(direction) - _heading(-u_in))))
            if tags.get("highway") not in DRIVABLE:
                reason = "class"
            elif is_private(tags):
                reason = "private"
            elif _grade_separated_near(nw, leg):
                reason = "grade_separated"
            elif any(n in ring_node and m < GRADE_FREE_M for n, m in zip(leg.nodes, leg.node_m)):
                reason = "roundabout"  # an entry or exit of a roundabout the route drives
            elif off < MIN_ANGLE_DEG:
                reason = "angle"
            elif leg.ends == "meets_route" and length < MIN_LEG_M:
                reason = "meets_route"
            else:
                reason = None
            record = {"node": int(nw.osm_node[node]), "at_m": round(at, 1), "xy": np.round(here, 2).tolist(),
                      "side": "left" if angle > 0 else "right", "angle_deg": round(angle, 1),
                      "off_mainline_deg": round(off, 1)}
            if reason is not None:
                skipped.append({**record, "way_id": nw.ways[way]["id"], "highway": tags.get("highway"),
                                "name": tags.get("name"), "reason": reason})
                continue
            turn = abs(_wrap(_heading(u_out) - _heading(u_in)))
            found.append({**record, "route_turn_deg": round(turn, 1), **_leg_record(nw, leg, way)})
    found.sort(key=lambda j: (j["at_m"], j["side"], j["angle_deg"]))
    junctions = [{"id": k, **j} for k, j in enumerate(found)]
    if to_lonlat is not None:
        for j in junctions:
            lon, lat = to_lonlat.transform(*j["xy"])
            j["lonlat"] = [round(float(lon), 7), round(float(lat), 7)]
    return {"version": FORMAT_VERSION,
            "rule": {"leg_m": leg_m, "min_angle_deg": MIN_ANGLE_DEG, "angle_at_m": ANGLE_AT_M, "min_leg_m": MIN_LEG_M,
                     "grade_free_m": GRADE_FREE_M, "classes": sorted(DRIVABLE), "private": sorted(PRIVATE),
                     "limited_access": sorted(LIMITED_ACCESS)},
            "junctions": junctions,
            "roundabouts": [_ring_record(nw, rt, ring, k, leg_m) for k, ring in enumerate(rings)],
            "skipped": skipped}


def _rings(nw: _Network, rt: _Route) -> list[dict]:
    """The roundabouts the route touches: each ring's ways, nodes and segments, its ways grouped where they share
    nodes."""
    rings, done = [], set()
    for n0 in sorted(rt.nodes):
        for _, w0 in nw.adj[n0]:
            if not nw.is_ring(w0) or w0 in done:
                continue
            ways, nodes, segs, todo = set(), set(), set(), [n0]
            while todo:
                n = todo.pop()
                if n in nodes:
                    continue
                nodes.add(n)
                for m, w in nw.adj[n]:
                    if nw.is_ring(w):
                        ways.add(w)
                        segs.add((min(n, m), max(n, m)))
                        todo.append(m)
            done |= ways
            rings.append({"ways": sorted(ways), "nodes": nodes, "segments": sorted(segs)})
    return rings


def _ring_order(ring: dict) -> tuple[list[int], bool]:
    """The ring's nodes in order round it (the first repeated at the end if it closes), and whether it closes."""
    nbrs: dict[int, list[int]] = defaultdict(list)
    for a, b in ring["segments"]:
        nbrs[a].append(b)
        nbrs[b].append(a)
    ends = sorted(n for n, m in nbrs.items() if len(m) == 1)
    first = ends[0] if ends else min(nbrs)
    order, prev, cur = [first], None, first
    while True:
        nxt = [m for m in nbrs[cur] if m != prev and (m not in order or (m == first and len(order) > 2))]
        if not nxt:
            return order, False
        prev, cur = cur, nxt[0]
        if cur == first:
            return order + [first], True
        order.append(cur)


def _ring_record(nw: _Network, rt: _Route, ring: dict, k: int, leg_m: float) -> dict:
    order, closed = _ring_order(ring)
    pts = nw.xy[order]
    loop = pts[:-1] if closed else pts
    centre = loop.mean(axis=0)
    on = [i for i, n in enumerate(rt.node) if n in ring["nodes"]]
    tags = nw.tags(ring["ways"][0])
    exits = []
    for n in (order[:-1] if closed else order):
        for m, w in nw.adj[n]:
            if nw.is_ring(w) or frozenset((n, m)) in rt.pairs:
                continue
            t = nw.tags(w)
            if t.get("highway") not in DRIVABLE or is_private(t):
                continue
            leg = nw.walk(n, m, w, (rt.nodes | ring["nodes"]) - {n}, leg_m)
            if _grade_separated_near(nw, leg):
                continue
            exits.append({"node": int(nw.osm_node[n]), "xy": np.round(nw.xy[n], 2).tolist(),
                          "on_route": n in rt.nodes, **_leg_record(nw, leg, w)})
    return {
        "id": k,
        "way_ids": [nw.ways[w]["id"] for w in ring["ways"]],
        "highway": tags.get("highway"),
        "name": tags.get("name"),
        "tags": {key: tags[key] for key in KEEP_TAGS if key in tags},
        "ring": np.round(pts, 2).tolist(),
        "closed": closed,
        "centre": np.round(centre, 2).tolist(),
        "radius_m": round(float(np.hypot(*(loop - centre).T).mean()), 1),
        "on_route": bool(on),
        "route_from_m": round(float(rt.s[on[0]]), 1) if on else None,
        "route_to_m": round(float(rt.s[on[-1]]), 1) if on else None,
        "exits": exits,
    }


# -- keeping the mouths clear --------------------------------------------------------------------------------------

@dataclass
class Mouth:
    """A junction's mouth on the mainline, and its side road, for the stages that keep them clear."""

    junction: dict  # its record in junctions.json
    side: str  # "L" or "R", of the centreline's direction (the forward layout)
    station_m: float  # the junction on the centreline
    s0: float  # the mouth's stretch along the centreline, margin included
    s1: float
    line: np.ndarray  # the side road from the junction outwards
    half_m: float  # its half width

    def corridor(self, beyond_m: float = 0.0) -> shapely.Geometry:
        """The side road's carriageway out to its end, widened by `beyond_m` either side."""
        return shapely.LineString(self.line).buffer(self.half_m + beyond_m, cap_style="flat")


def _station(xy: np.ndarray, s: np.ndarray, p: np.ndarray, near_m: float) -> float:
    """Where point `p` (a junction's node) falls on the centreline: its nearest point, or, where another part of the
    road passes within a metre of as near, the nearest point of the part nearest station `near_m`. The smoothed
    centreline's stations drift from the routed line's (67 m by 53 km on Lyell Highway), so `near_m` only breaks a
    tie."""
    a, b = xy[:-1], xy[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0.0, 1.0)
    d = np.hypot(*(a + ab * t[:, None] - p).T)
    at = s[:-1] + t * np.diff(s)
    near = np.flatnonzero(d <= d.min() + 1.0)
    parts = np.split(near, np.flatnonzero(np.diff(near) > 1) + 1)
    best = [int(part[np.argmin(d[part])]) for part in parts]
    return float(at[min(best, key=lambda k: abs(at[k] - near_m))])


def mouths(junctions: list[dict], s: np.ndarray, xy: np.ndarray, heading: np.ndarray, half_l: np.ndarray,
           half_r: np.ndarray, *, band_m: float = MOUTH_BAND_M, margin_m: float = MOUTH_MARGIN_M) -> list[Mouth]:
    """Each junction's mouth on the smoothed centreline (stations `s`, points `xy`, headings, half widths): where its
    side road's carriageway crosses the band from the edge to `band_m` beyond it on its side, plus `margin_m` along
    the road either side. The side road's line and width come from junctions.json."""
    s, xy = np.asarray(s, float), np.asarray(xy, float)
    nrm = np.column_stack([-np.sin(heading), np.cos(heading)])
    offsets = np.arange(0.0, band_m + 1e-9, MOUTH_BAND_STEP_M)
    out = []
    for j in junctions:
        line = np.asarray(j["line"], dtype=float)
        half = float(j["width_m"]) / 2
        side = SIDE_CODE[j["side"]]
        sign = 1.0 if side == "L" else -1.0
        edge = np.asarray(half_l if side == "L" else half_r, float)
        at = _station(xy, s, np.asarray(j["xy"], float), float(j["at_m"]))
        reach = float(j["length_m"]) + band_m + 20.0
        idx = np.flatnonzero(np.abs(s - at) <= reach)
        carriageway = shapely.LineString(line).buffer(half, cap_style="flat")
        shapely.prepare(carriageway)
        pts = xy[idx, None, :] + nrm[idx, None, :] * (sign * (edge[idx, None] + offsets[None, :]))[..., None]
        inside = shapely.contains_xy(carriageway, pts[..., 0].ravel(), pts[..., 1].ravel()).reshape(pts.shape[:2])
        hit = idx[inside.any(axis=1)]
        lo, hi = (float(s[hit.min()]), float(s[hit.max()])) if len(hit) else (at - half, at + half)
        out.append(Mouth(j, side, at, max(lo - margin_m, float(s[0])), min(hi + margin_m, float(s[-1])), line, half))
    return out


def masks(ms: list[Mouth], s: np.ndarray) -> dict[str, np.ndarray]:
    """Which centreline stations fall in a mouth, on each side ("L", "R")."""
    s = np.asarray(s, float)
    out = {"L": np.zeros(len(s), bool), "R": np.zeros(len(s), bool)}
    for m in ms:
        out[m.side] |= (s >= m.s0) & (s <= m.s1)
    return out


def side_band(m: Mouth, s: np.ndarray, xy: np.ndarray, width_m: float | None = None) -> shapely.Geometry:
    """The ground on the mouth's side of the centreline near its junction, as far out as its side road reaches."""
    reach = float(cumulative(m.line)[-1]) + 60.0
    idx = np.flatnonzero(np.abs(np.asarray(s) - m.station_m) <= reach)
    if len(idx) < 2:
        idx = np.arange(len(s))
    w = width_m if width_m is not None else reach
    return shapely.LineString(np.asarray(xy)[idx]).buffer(w if m.side == "L" else -w, single_sided=True)


def zone(ms: list[Mouth], s: np.ndarray, xy: np.ndarray, beyond_m: float) -> shapely.Geometry | None:
    """The side roads' carriageways widened by `beyond_m` either side, each kept to its own side of the mainline:
    the ground to keep clear of trees, buildings or power-line supports. None with no mouths."""
    if not ms:
        return None
    parts = [m.corridor(beyond_m).intersection(side_band(m, s, xy)) for m in ms]
    return shapely.unary_union(parts)


def layout_intervals(ms: list[Mouth], s: np.ndarray, reverse: bool) -> list[tuple[float, float]]:
    """The mouths as (from, to) metres from a layout's start, in driving order, on either side."""
    s0, total = float(s[0]), float(s[-1] - s[0])
    spans = [(m.s0 - s0, m.s1 - s0) for m in ms]
    if reverse:
        spans = [(total - b, total - a) for a, b in spans]
    return sorted(spans)
