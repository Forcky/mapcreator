"""A directed road graph from OSM ways, and routing between waypoints on it.

Each waypoint is projected onto its nearest road segment and spliced into the
graph as a virtual node. That way the route starts and ends exactly where it
was asked to, not at the nearest OSM node, which can be 100+ m away on a
straight rural road.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from mapcreator.job import MAX_SNAP_M

# A snapped waypoint this close to a node is on it.
ON_NODE_M = 1.0

# Mild preference for bigger roads, so a route between sparse waypoints follows
# the main road rather than a parallel residential street of equal length.
CLASS_FACTOR = {
    "motorway": 1.0, "trunk": 1.0, "primary": 1.0, "secondary": 1.0,
    "tertiary": 1.05, "unclassified": 1.1, "road": 1.2,
    "residential": 1.3, "living_street": 2.0,
}
LINK_FACTOR = 1.1


class RouteError(ValueError):
    pass


@dataclass
class RoadGraph:
    xy: np.ndarray  # (n, 2) node coordinates in the region's projected CRS
    osm_node: np.ndarray  # (n,) OSM node ids
    seg_a: np.ndarray  # (m,) node index
    seg_b: np.ndarray  # (m,) node index
    seg_way: np.ndarray  # (m,) index into `ways`
    seg_fwd: np.ndarray  # (m,) bool: a -> b allowed
    seg_rev: np.ndarray  # (m,) bool: b -> a allowed
    ways: list[dict] = field(default_factory=list)  # {"id": osm id, "tags": {...}}


@dataclass
class Route:
    xy: np.ndarray  # (k, 2) polyline
    seg_way: np.ndarray  # (k-1,) way index for each polyline segment
    ways: list[dict]
    retraced_m: float  # length of road driven more than once


def _direction(tags: dict) -> tuple[bool, bool]:
    oneway = tags.get("oneway", "")
    if oneway in ("yes", "true", "1"):
        return True, False
    if oneway == "-1":
        return False, True
    if oneway == "no":
        return True, True
    if tags.get("junction") in ("roundabout", "circular") or tags.get("highway") == "motorway":
        return True, False
    return True, True


def _factor(tags: dict) -> float:
    hw = tags.get("highway", "")
    if hw.endswith("_link"):
        return LINK_FACTOR
    return CLASS_FACTOR.get(hw, 1.5)


def build(osm: dict, to_xy: Transformer) -> RoadGraph:
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in osm["elements"] if e["type"] == "node"}
    ways = [e for e in osm["elements"] if e["type"] == "way"]
    if not ways:
        raise RouteError("no drivable roads in the area")

    ids = np.fromiter(nodes.keys(), dtype=np.int64)
    lonlat = np.array(list(nodes.values()), dtype=float)
    x, y = to_xy.transform(lonlat[:, 0], lonlat[:, 1])
    index = {nid: i for i, nid in enumerate(ids.tolist())}

    a, b, w, fwd, rev = [], [], [], [], []
    way_list = []
    for way in ways:
        refs = [index[r] for r in way["nodes"] if r in index]
        if len(refs) < 2:
            continue
        tags = way.get("tags", {})
        f, r = _direction(tags)
        wi = len(way_list)
        way_list.append({"id": way["id"], "tags": tags})
        a.extend(refs[:-1])
        b.extend(refs[1:])
        w.extend([wi] * (len(refs) - 1))
        fwd.extend([f] * (len(refs) - 1))
        rev.extend([r] * (len(refs) - 1))

    return RoadGraph(
        xy=np.column_stack([x, y]),
        osm_node=ids,
        seg_a=np.asarray(a, dtype=np.int64),
        seg_b=np.asarray(b, dtype=np.int64),
        seg_way=np.asarray(w, dtype=np.int64),
        seg_fwd=np.asarray(fwd, dtype=bool),
        seg_rev=np.asarray(rev, dtype=bool),
        ways=way_list,
    )


def _nearest_segment(g: RoadGraph, p: np.ndarray) -> tuple[int, float, np.ndarray, float]:
    """Index of the nearest segment, the parameter t along it, the projected point, and the distance."""
    pa = g.xy[g.seg_a]
    pb = g.xy[g.seg_b]
    d = pb - pa
    len2 = np.maximum((d * d).sum(axis=1), 1e-12)
    t = np.clip(((p - pa) * d).sum(axis=1) / len2, 0.0, 1.0)
    proj = pa + d * t[:, None]
    dist = np.hypot(*(proj - p).T)
    i = int(np.argmin(dist))
    return i, float(t[i]), proj[i], float(dist[i])


def route(g: RoadGraph, waypoints_xy: np.ndarray, *, max_snap_m: float = MAX_SNAP_M) -> Route:
    """Shortest class-weighted route through the waypoints in order, respecting one-way rules."""
    n = len(g.xy)
    seg_len = np.hypot(*(g.xy[g.seg_b] - g.xy[g.seg_a]).T)
    factor = np.array([_factor(g.ways[wi]["tags"]) for wi in g.seg_way])
    cost = seg_len * factor

    src, dst, wt, way_of = [], [], [], []

    def add(u: int, v: int, c: float, wi: int) -> None:
        src.append(u)
        dst.append(v)
        wt.append(max(c, 1e-6))
        way_of.append(wi)

    for k in range(len(g.seg_a)):
        u, v, wi = int(g.seg_a[k]), int(g.seg_b[k]), int(g.seg_way[k])
        if g.seg_fwd[k]:
            add(u, v, cost[k], wi)
        if g.seg_rev[k]:
            add(v, u, cost[k], wi)

    # Splice each waypoint in as a virtual node on its nearest segment.
    xy = [g.xy]
    virtual = []
    on_segment: dict[int, list[tuple[float, int]]] = {}
    for j, p in enumerate(np.asarray(waypoints_xy, dtype=float)):
        k, t, proj, dist = _nearest_segment(g, p)
        if dist > max_snap_m:
            raise RouteError(f"waypoint {j + 1} is {dist:.0f} m from the nearest drivable road")
        vi = n + j
        virtual.append(vi)
        on_segment.setdefault(k, []).append((t, vi))
        xy.append(proj[None, :])
        u, v, wi = int(g.seg_a[k]), int(g.seg_b[k]), int(g.seg_way[k])
        cu, cv = cost[k] * t, cost[k] * (1 - t)
        if g.seg_fwd[k]:
            add(u, vi, cu, wi)
            add(vi, v, cv, wi)
        if g.seg_rev[k]:
            add(v, vi, cv, wi)
            add(vi, u, cu, wi)
        # A waypoint that lands on a node is that node, whichever of the segments meeting there was nearest
        # first: join it both ways, or a one-way slip road listed first at a junction would force the route
        # down the slip (or find none at all).
        for end, along in ((u, seg_len[k] * t), (v, seg_len[k] * (1 - t))):
            if along < ON_NODE_M:
                add(end, vi, 0.0, wi)
                add(vi, end, 0.0, wi)
    # Waypoints sharing a segment connect directly, or the route would overshoot and come back.
    for k, members in on_segment.items():
        members.sort()
        wi = int(g.seg_way[k])
        for (t0, v0), (t1, v1) in zip(members, members[1:]):
            c = cost[k] * (t1 - t0)
            if g.seg_fwd[k]:
                add(v0, v1, c, wi)
            if g.seg_rev[k]:
                add(v1, v0, c, wi)

    all_xy = np.vstack(xy)
    size = len(all_xy)
    # Keep the cheapest edge per (u, v) pair and remember which way it belongs to.
    best: dict[tuple[int, int], tuple[float, int]] = {}
    for u, v, c, wi in zip(src, dst, wt, way_of):
        if (u, v) not in best or c < best[(u, v)][0]:
            best[(u, v)] = (c, wi)
    keys = list(best.keys())
    mat = coo_matrix(
        ([best[k][0] for k in keys], ([k[0] for k in keys], [k[1] for k in keys])), shape=(size, size)
    ).tocsr()

    path: list[int] = []
    for leg, (s, t) in enumerate(zip(virtual, virtual[1:])):
        dist, pred = dijkstra(mat, directed=True, indices=s, return_predecessors=True)
        if not np.isfinite(dist[t]):
            raise RouteError(f"no drivable connection from waypoint {leg + 1} to {leg + 2}")
        leg_nodes = [t]
        while leg_nodes[-1] != s:
            leg_nodes.append(int(pred[leg_nodes[-1]]))
        leg_nodes.reverse()
        path.extend(leg_nodes if not path else leg_nodes[1:])

    seg_way = np.array([best[(u, v)][1] for u, v in zip(path, path[1:])], dtype=np.int64)
    # Drop zero-length hops (a waypoint exactly on a node joins via a 0 m virtual edge).
    pts = all_xy[path]
    nonzero = np.hypot(*np.diff(pts, axis=0).T) > 1e-6
    line = np.vstack([pts[:1], pts[1:][nonzero]])
    seg_way = seg_way[nonzero]

    # Road driven more than once (Street2Sim refuses these; we report it).
    seen: dict[tuple[int, int], int] = {}
    retraced = 0.0
    for u, v in zip(path, path[1:]):
        key = (min(u, v), max(u, v))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            retraced += float(np.hypot(*(all_xy[v] - all_xy[u])))

    return Route(xy=line, seg_way=seg_way, ways=g.ways, retraced_m=retraced)
