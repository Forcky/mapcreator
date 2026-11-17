"""Roads that cross the route without meeting it: over it on a bridge, or under the route's own bridge.

OpenStreetMap draws a bridge over a road without joining the two, so a crossing road shares no node with the route
and is never a junction (`core.junctions`). `find`, run by the route stage, lists every way of the road data whose
line crosses the route's at a point that isn't a node they share, with which of the two is on top.

**Which is on top.** OSM's `layer` says only which of two crossing ways is above the other, never a height (the OSM
wiki, `Key:layer`). Each way's level is its `layer`; without one, 1 on a bridge, -1 in a tunnel, else 0 (the common
reading of an untagged bridge or tunnel). The crossing way is "over" the route when its level is higher, "under" it
when lower. Two ways crossing at the same level are a mapping slip (a missing junction node, or a missing `layer`):
they're listed apart, under "unresolved", and nothing builds on them.

The road data holds the router's classes only (`providers.osm.overpass.DRIVABLE`), so footbridges, railways and
waterways aren't listed. Coordinates are in the region's projected CRS, as the route's are. The format of
`01_route/crossings.json` is in `docs/ac-format.md` ("Crossings").

Today the profile stage reads it: where a road crosses over the route, the bare-earth DEM's embankment beside the
underpass must not lift the route (`core.profile.humps_under`). The structures to come (decks, piers, the roads
under the route's bridges) build on the same list.
"""

from __future__ import annotations

import math

import numpy as np
import shapely

from mapcreator.core.centreline import cumulative

FORMAT_VERSION = 1

CLIP_M = 150.0  # each crossing way's line is kept this far either side of the crossing, along the way
SAME_POINT_M = 0.05  # a crossing this close to a node the two ways share is that node: a junction, not a crossing
# The crossing road followed past its way's ends (`road_through`): into the way that turns least, if under this.
ROAD_TURN_MAX_DEG = 60.0
# The crossing way's tags kept in crossings.json.
KEEP_TAGS = ("highway", "name", "ref", "bridge", "tunnel", "layer", "lanes", "width", "oneway", "surface", "maxspeed",
             "bridge:structure", "bridge:name", "maxheight")


def _no(value) -> bool:
    return str(value or "no").strip().lower() in ("no", "")


def is_bridge(tags: dict) -> bool:
    """`bridge` tagged with anything but `no` (`low_water_crossing`, a ford, isn't a bridge)."""
    value = str(tags.get("bridge") or "").strip().lower()
    return not _no(value) and value != "low_water_crossing"


def is_tunnel(tags: dict) -> bool:
    return not _no(tags.get("tunnel"))


def level(tags: dict) -> float:
    """A way's level among the ways it crosses: its `layer`, else 1 on a bridge, -1 in a tunnel, else 0."""
    default = 1.0 if is_bridge(tags) else -1.0 if is_tunnel(tags) else 0.0
    raw = str(tags.get("layer") or "").split(";")[0].strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def _way_lines(net) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Each way's polyline in node order, as (node indices, points). The road data's segments are stored way by way,
    in order (`providers.osm.graph.build`)."""
    seg_way = np.asarray(net.seg_way)
    out = {}
    if not len(seg_way):
        return out
    starts = np.flatnonzero(np.r_[True, seg_way[1:] != seg_way[:-1]])
    ends = np.r_[starts[1:], len(seg_way)]
    a, b = np.asarray(net.seg_a), np.asarray(net.seg_b)
    for s0, s1 in zip(starts.tolist(), ends.tolist()):
        nodes = np.r_[a[s0:s1], b[s1 - 1]]
        out.setdefault(int(seg_way[s0]), (nodes, np.asarray(net.xy, float)[nodes]))
    return out


def _clip(line: np.ndarray, at: float, reach: float) -> tuple[np.ndarray, float]:
    """The part of a polyline within `reach` of the point `at` metres along it, and where that point now lies."""
    s = cumulative(line)
    lo, hi = max(at - reach, 0.0), min(at + reach, float(s[-1]))
    inner = line[(s > lo) & (s < hi)]
    ends = [np.array([np.interp(m, s, line[:, 0]), np.interp(m, s, line[:, 1])]) for m in (lo, hi)]
    return np.vstack([ends[0][None], inner, ends[1][None]]), at - lo


def _angle(route_xy: np.ndarray, k: int, line: np.ndarray, j: int) -> float:
    """The angle (0-90 degrees) between route segment k and segment j of the crossing way's line."""
    u = route_xy[k + 1] - route_xy[k]
    v = line[j + 1] - line[j]
    cos = abs(float(u @ v)) / max(float(np.hypot(*u) * np.hypot(*v)), 1e-12)
    return round(math.degrees(math.acos(min(cos, 1.0))), 1)


def find(net, route_xy: np.ndarray, route_seg_way: np.ndarray, *, to_lonlat=None) -> dict:
    """The crossings of a routed polyline: the content of crossings.json.

    `net` is the road network the route was found on (`providers.osm.graph.RoadGraph`: node coordinates `xy`,
    segments `seg_a`/`seg_b` with their way's index `seg_way`, and `ways`). `route_xy` is the route's polyline and
    `route_seg_way` the way index of each of its segments. `to_lonlat`, a transformer, adds each crossing's longitude
    and latitude.
    """
    route_xy = np.asarray(route_xy, dtype=float)
    route_seg_way = np.asarray(route_seg_way)
    route_s = cumulative(route_xy)
    keep = np.flatnonzero(np.hypot(*np.diff(route_xy, axis=0).T) > 1e-9)
    route_segments = shapely.linestrings(np.stack([route_xy[keep], route_xy[keep + 1]], axis=1))
    tree = shapely.STRtree(route_segments)
    xy = np.asarray(net.xy, dtype=float)
    route_ways = {int(w) for w in route_seg_way.tolist()}
    # The route's own nodes: its points are network nodes (except where a waypoint was spliced in) on its own ways. A
    # crossing way's node at the same place, not shared, is still a crossing.
    on_route_way = np.zeros(len(xy), dtype=bool)
    mine = np.isin(np.asarray(net.seg_way), list(route_ways))
    on_route_way[np.asarray(net.seg_a)[mine]] = True
    on_route_way[np.asarray(net.seg_b)[mine]] = True
    at_point: dict[tuple[float, float], list[int]] = {}
    for i, p in enumerate(xy.tolist()):
        at_point.setdefault((p[0], p[1]), []).append(i)
    route_nodes = {n for p in route_xy.tolist() for n in at_point.get((p[0], p[1]), []) if on_route_way[n]}
    lines = _way_lines(net)
    at_node = None  # each node's ways, for following a crossing road past its way's ends (`road_through`)
    found, unresolved, seen = [], [], set()
    for wi, (nodes, line) in lines.items():
        if wi in route_ways:
            continue  # the route's own ways (a route crossing itself on a loop isn't handled here)
        segments = shapely.linestrings(np.stack([line[:-1], line[1:]], axis=1))
        pairs = tree.query(segments, predicate="intersects")
        for j, k_local in zip(*pairs):
            k = int(keep[k_local])
            hit = shapely.intersection(segments[j], route_segments[k_local])
            if shapely.get_type_id(hit) != 0:  # not a single point: touching along a length (a spliced waypoint)
                continue
            p = np.array(shapely.get_coordinates(hit)[0])
            ends = [n for n in (nodes[j], nodes[j + 1]) if int(n) in route_nodes]
            if any(np.hypot(*(xy[n] - p)) < SAME_POINT_M for n in ends):
                continue  # a node the two share: a junction
            key = (wi, round(float(p[0]), 1), round(float(p[1]), 1))
            if key in seen:
                continue
            seen.add(key)
            at = float(route_s[k] + np.hypot(*(p - route_xy[k])))
            along = float(cumulative(line)[j] + np.hypot(*(p - line[j])))
            tags = net.ways[wi].get("tags") or {}
            mine = net.ways[int(route_seg_way[k])].get("tags") or {}
            theirs, ours = level(tags), level(mine)
            clipped, along_clipped = _clip(line, along, CLIP_M)
            record = {
                "at_m": round(at, 1), "xy": np.round(p, 2).tolist(),
                "relation": "over" if theirs > ours else "under" if theirs < ours else None,
                "angle_deg": _angle(route_xy, k, line, j),
                "way_id": net.ways[wi]["id"], "highway": tags.get("highway"), "name": tags.get("name"),
                "level": theirs, "bridge": is_bridge(tags), "tunnel": is_tunnel(tags),
                "tags": {t: tags[t] for t in KEEP_TAGS if t in tags},
                "way_length_m": round(float(cumulative(line)[-1]), 1),
                "line": np.round(clipped, 2).tolist(), "line_at_m": round(along_clipped, 1),
                "route": {"way_id": net.ways[int(route_seg_way[k])]["id"], "level": ours,
                          "bridge": is_bridge(mine), "tunnel": is_tunnel(mine)},
            }
            if to_lonlat is not None:
                lon, lat = to_lonlat.transform(*record["xy"])
                record["lonlat"] = [round(float(lon), 7), round(float(lat), 7)]
            if record["relation"]:
                if at_node is None:
                    at_node = _node_ways(lines)
                record["road"] = road_through(net, lines, at_node, wi, j, p, route_ways)
            (found if record["relation"] else unresolved).append(record)
    found.sort(key=lambda c: (c["at_m"], c["way_id"]))
    unresolved.sort(key=lambda c: (c["at_m"], c["way_id"]))
    for c in unresolved:
        del c["relation"]
    return {"version": FORMAT_VERSION,
            "rule": {"levels": "layer, else 1 on a bridge and -1 in a tunnel, else 0", "clip_m": CLIP_M},
            "crossings": [{"id": n, **c} for n, c in enumerate(found)],
            "unresolved": unresolved}


def _node_ways(lines: dict) -> dict[int, list[tuple[int, int]]]:
    """Each node's (way index, position in the way's node list), for every way through it."""
    out: dict[int, list[tuple[int, int]]] = {}
    for wi, (nodes, _) in lines.items():
        for k, n in enumerate(nodes.tolist()):
            out.setdefault(n, []).append((wi, k))
    return out


def _step(nodes: np.ndarray, k: int, d: int) -> int | None:
    """The next position from k in direction d along a way's nodes, round a closed way, or None at its end."""
    closed = len(nodes) > 2 and nodes[0] == nodes[-1]
    nxt = k + d
    if closed:
        last = len(nodes) - 1  # the closing node repeats the first
        if nxt > last:
            nxt = 1
        elif nxt < 0:
            nxt = last - 1
    return nxt if 0 <= nxt < len(nodes) else None


def _follow(lines: dict, at_node: dict, route_ways: set, way: int, k: int, d: int, reach: float,
            first: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, float, float]], str]:
    """The road's points from `first` (on way `way` between its node k-d and node k), through node k and on in
    direction d, past the way's end into the way that turns least (under ROAD_TURN_MAX_DEG, and not one of the
    route's own), until `reach` metres. Returns the points, each way's (index, from, to) metres from `first`, and why
    it stopped: `length`, `dead_end` (no way carries on) or `route` (only the route's own ways do)."""
    pts, ways, dist, used = [], [], 0.0, {(way, d)}
    prev = np.asarray(first, float)
    while True:
        nodes, line = lines[way]
        start, guard = dist, 0
        while k is not None and guard <= len(nodes):
            p = line[k]
            dist += float(np.hypot(*(p - prev)))
            pts.append(p)
            prev = p
            guard += 1
            if dist >= reach:
                ways.append((way, start, dist))
                return pts, ways, "length"
            nxt = _step(nodes, k, d)
            if nxt is None:
                break
            k = nxt
        ways.append((way, start, dist))
        node = int(nodes[k])
        back = pts[-2] if len(pts) > 1 else np.asarray(first, float)
        heading = pts[-1] - back
        best, blocked = None, False
        for w2, p2 in at_node.get(node, []):
            if w2 == way:
                continue
            n2, l2 = lines[w2]
            for d2 in (1, -1):
                q = _step(n2, p2, d2)
                if q is None or (w2, d2) in used:
                    continue
                out = l2[q] - l2[p2]
                cos = float(heading @ out) / max(float(np.hypot(*heading) * np.hypot(*out)), 1e-12)
                turn = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
                if turn >= ROAD_TURN_MAX_DEG:
                    continue
                if w2 in route_ways:
                    blocked = True
                    continue
                if best is None or turn < best[0]:
                    best = (turn, w2, q, d2)
        if best is None:
            return pts, ways, "route" if blocked else "dead_end"
        _, way, k, d = best
        used.add((way, d))


def road_through(net, lines: dict, at_node: dict, wi: int, j: int, p: np.ndarray, route_ways: set,
                 reach: float = CLIP_M) -> dict:
    """The crossing road's line within `reach` either side of its crossing at `p` (on segment j of way `wi`): along
    its own way and on past either end into the way that carries straight on (`_follow`), so a road drawn some way
    past the crossing doesn't stop where its way does (a bridge is usually its own way). The content of a crossing's
    `road` in crossings.json: `line`, `at_m` (where the crossing is along it), `ways` (each way along it: `way_id`,
    `from_m`, `to_m`, `bridge`, `tunnel`, `level`, `highway`) and `ends` (why it stopped, before and after)."""
    p = np.asarray(p, float)
    fwd, fwd_ways, fwd_end = _follow(lines, at_node, route_ways, wi, j + 1, 1, reach, p)
    bwd, bwd_ways, bwd_end = _follow(lines, at_node, route_ways, wi, j, -1, reach, p)
    line = np.vstack([np.asarray(bwd[::-1]).reshape(-1, 2), p[None], np.asarray(fwd).reshape(-1, 2)])
    s = cumulative(line)
    at = float(s[len(bwd)])
    clipped, at_clipped = _clip(line, at, reach)
    lo = at - at_clipped
    spans: list[tuple[int, float, float]] = []
    for w, a, b in reversed(bwd_ways):
        spans.append((w, at - b, at - a))
    for w, a, b in fwd_ways:
        spans.append((w, at + a, at + b))
    out_ways = []
    length = float(cumulative(clipped)[-1])
    for w, a, b in spans:
        a, b = max(a - lo, 0.0), min(b - lo, length)
        if b - a <= 1e-6:
            continue
        if out_ways and out_ways[-1]["way"] == w and abs(out_ways[-1]["to_m"] - a) < 1e-6:
            out_ways[-1]["to_m"] = b  # the crossing's own way, either side of the crossing
            continue
        out_ways.append({"way": w, "from_m": a, "to_m": b})
    for o in out_ways:
        w = o.pop("way")
        tags = net.ways[w].get("tags") or {}
        o.update({"way_id": net.ways[w]["id"], "from_m": round(o["from_m"], 2), "to_m": round(o["to_m"], 2),
                  "bridge": is_bridge(tags), "tunnel": is_tunnel(tags), "level": level(tags),
                  "highway": tags.get("highway")})
    out_ways = [{k: o[k] for k in ("way_id", "from_m", "to_m", "bridge", "tunnel", "level", "highway")}
                for o in out_ways]
    return {"line": np.round(clipped, 2).tolist(), "at_m": round(at_clipped, 2), "ways": out_ways,
            "ends": [bwd_end, fwd_end]}


def over(crossings: list[dict]) -> list[dict]:
    """The crossings where the other road passes over the route, the route not in a tunnel there (a tunnel's own
    profile is already a straight grade, portal to portal)."""
    return [c for c in crossings if c["relation"] == "over" and not c["route"]["tunnel"]]
