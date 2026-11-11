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
            (found if record["relation"] else unresolved).append(record)
    found.sort(key=lambda c: (c["at_m"], c["way_id"]))
    unresolved.sort(key=lambda c: (c["at_m"], c["way_id"]))
    for c in unresolved:
        del c["relation"]
    return {"version": FORMAT_VERSION,
            "rule": {"levels": "layer, else 1 on a bridge and -1 in a tunnel, else 0", "clip_m": CLIP_M},
            "crossings": [{"id": n, **c} for n, c in enumerate(found)],
            "unresolved": unresolved}


def over(crossings: list[dict]) -> list[dict]:
    """The crossings where the other road passes over the route, the route not in a tunnel there (a tunnel's own
    profile is already a straight grade, portal to portal)."""
    return [c for c in crossings if c["relation"] == "over" and not c["route"]["tunnel"]]
