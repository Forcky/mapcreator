"""Stage 1: snap the job's waypoints to OSM and write the routed polyline with its tags.

Roads come from the local OSM index when `osm_index` is set (production), else from Overpass
(development); see `providers/osm/roads.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import numpy as np
from scipy.spatial import cKDTree

from mapcreator.config import Settings
from mapcreator.core import centreline
from mapcreator.core import crossings as crossings_core
from mapcreator.core import junctions as junctions_core
from mapcreator.core import rail_and_paths
from mapcreator.core import ramps as ramps_core
from mapcreator.core.centreline import cumulative
from mapcreator.job import Job, Junctions
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.osm import graph as osm_graph
from mapcreator.providers.osm import roads as osm_roads
from mapcreator.regions import job_region

STAGE = "01_route"
# How the roads were selected from OpenStreetMap (`osm_roads.selection`), for the track's ATTRIBUTION.txt.
OSM_SELECTION = "osm_selection.json"
# The side-road junctions and roundabouts along the route (`core.junctions`; the format is in docs/ac-format.md).
JUNCTIONS = "junctions.json"
# The roads crossing over or under the route without meeting it (`core.crossings`; the format is in docs/ac-format.md).
CROSSINGS = "crossings.json"
# The railways and paths crossing over or under the route (`core.rail_and_paths`; the format is in docs/ac-format.md).
CROSSING_LINES = "crossing_lines.json"
# The ramps leaving or joining the route at a slip angle (`core.ramps`; the format is in docs/ac-format.md).
RAMPS = "ramps.json"


class RouteTooLong(osm_graph.RouteError):
    """A user-facing error: the route between the waypoints is longer than the job allows."""


class RouteTurnsBack(osm_graph.RouteError):
    """A user-facing error: the route goes back over road it has already driven, usually because a point landed on
    a side road or past a junction. A track can't use the same road twice, and its AI line can't turn on the spot."""


RETRACE_LIMIT_M = 1.0  # every route built before this check measured 0 m


def turn_back(xy: np.ndarray) -> tuple[int, float] | None:
    """Where the route first goes back the way it came: the index of the point, and how much road it drives twice
    (metres). Either a segment runs over road already driven (`centreline.retraced`), or a single node turns harder
    than `centreline.REVERSAL_DEG`, as where two points on one stretch of road send it out and back. No route built
    before this check had either: its sharpest single-node turn was a junction in Richmond, 108 degrees."""
    xy = np.asarray(xy, dtype=float)
    retraced_m, first = centreline.retraced(xy)
    found = [first] if first is not None and retraced_m > RETRACE_LIMIT_M else []
    seg = np.diff(xy, axis=0)
    keep = np.flatnonzero(np.hypot(*seg.T) > 1e-6)
    if len(keep) >= 2:
        h = np.arctan2(seg[keep, 1], seg[keep, 0])
        turn = np.degrees(np.abs(np.angle(np.exp(1j * np.diff(h)))))
        found += [int(i) for i in keep[1:][turn > centreline.REVERSAL_DEG]]  # the node the turn is at
    return (min(found), retraced_m) if found else None


def u_turn(xy: np.ndarray, rings: Sequence[tuple[float, float]] = (),
           names: Sequence[str | None] | None = None) -> int | None:
    """Where the route first makes a U-turn out of corners close together: more than `centreline.REVERSAL_DEG`
    within `centreline.REVERSAL_SPAN_M`. A point on the far side of a divided road does it, sending the route through
    a gap in the median and back down the other carriageway. Each corner is an ordinary turn, and no road is driven
    twice, so `turn_back` doesn't see it (12 Nov 2026: a 10 m gap, rounded to a 2.3 m bend that folded the road over
    itself). The index is the node where the U-turn starts.

    `rings`: the stretches on a roundabout (from, to, in metres along `xy`), which are left to validation. The path
    round a small ring can turn as hard, and a build can still go round it on request.

    `names`: the name of the road each segment of `xy` is on, where known. A U-turn then has to leave the road and
    come back onto it: the same road before the first corner and after the last, and another between, as through the
    median onto the other carriageway, or into a side street and straight back out. A hairpin on one road, and a turn
    off one road into a tight loop on another (a court's turning circle), are left to validation. (Tasmania, Nov
    2026: 38 roads have a hairpin drawn this tight, and 26 of them pass validation or can be built on request; no
    path through any of 391 roundabouts was refused.)"""
    xy = np.asarray(xy, dtype=float)
    seg = np.diff(xy, axis=0)
    keep = np.flatnonzero(np.hypot(*seg.T) > 1e-6)
    if len(keep) < 2:
        return None
    h = np.arctan2(seg[keep, 1], seg[keep, 0])
    turn = np.degrees(np.angle(np.exp(1j * np.diff(h))))  # signed, at the nodes keep[1:]
    node = keep[1:]
    at = cumulative(xy)[node]
    on_ring = np.zeros(len(node), dtype=bool)
    for a, b in rings:
        on_ring |= (at >= a - 1.0) & (at <= b + 1.0)
    for i in range(len(turn)):
        end = int(np.searchsorted(at, at[i] + centreline.REVERSAL_SPAN_M, side="right"))
        if on_ring[i:end].any():
            continue
        for j in i + np.flatnonzero(np.abs(np.cumsum(turn[i:end])) > centreline.REVERSAL_DEG):
            if names is None or _comes_back(names, int(keep[i]), int(keep[j + 1])):
                return int(node[i])
    return None


def _comes_back(names: Sequence[str | None], before: int, after: int) -> bool:
    """Whether the route leaves the road segment `before` is on and comes back onto it by segment `after`."""
    road = names[before] or ""
    return (names[after] or "") == road and any((names[k] or "") != road for k in range(before + 1, after))


def ring_spans(route: osm_graph.Route) -> list[tuple[float, float]]:
    """The stretches of a routed line on a roundabout's ways, in metres along it, for `u_turn`."""
    return _spans(cumulative(route.xy), [route.ways[w]["tags"].get("junction") in junctions_core.RING
                                         for w in route.seg_way])


def tunnel_spans(route: osm_graph.Route) -> list[tuple[float, float]]:
    """The stretches of a routed line through a tunnel (any `tunnel` tag but `no`), in metres along it, for the
    preview's note: a tunnel isn't covered yet, but built as an open cutting."""
    return _spans(cumulative(route.xy), [route.ways[w]["tags"].get("tunnel") not in (None, "no")
                                         for w in route.seg_way])


def _spans(s: np.ndarray, flags: Sequence[bool]) -> list[tuple[float, float]]:
    """The runs of flagged segments, as (from, to) in metres along a line whose stations are `s`."""
    out, k = [], 0
    while k < len(flags):
        if flags[k]:
            e = k
            while e + 1 < len(flags) and flags[e + 1]:
                e += 1
            out.append((float(s[k]), float(s[e + 1])))
            k = e + 1
        else:
            k += 1
    return out


def segment_names(route: osm_graph.Route) -> list[str | None]:
    """The name of the road each segment of a routed line is on, for `u_turn`."""
    return [route.ways[w]["tags"].get("name") for w in route.seg_way]


# Another part of the route this close (centreline to centreline) is a road over the route's own: two 3.2 m half
# widths overlap below 6.4 m. "Another part" is the junction finder's rule (much further along than straight across),
# so a hairpin's legs, whose along-distance is short, aren't another part.
ALONGSIDE_M = 5.0
ALONGSIDE_STEP_M = 1.0
# And only a real run back: at least this much further along than the part it's beside, for at least this long. A
# sharp V where a link meets a road at a narrow angle is close for a few metres, a short way along: validation's
# tight bend (which can be built on request) judges that. Going round even a small roundabout and back is longer.
ALONGSIDE_GAP_M = 30.0
ALONGSIDE_RUN_M = 10.0


def alongside(xy: np.ndarray) -> tuple[int, float] | None:
    """Where the route first comes back alongside itself: another part of it within `ALONGSIDE_M`, as when it goes
    round a roundabout and back up the other carriageway of a divided road beside the way it came (Nov 2026: a
    test route down a one-way split, round the roundabout and back up the other side, 2 m from itself, failed
    validation as an internal error). No road is driven twice and every turn is round the ring, so `turn_back` and
    `u_turn` don't see it. Returns the index of the point (the later of the two parts) and the distance along the
    route of the earlier part, or None."""
    xy = np.asarray(xy, dtype=float)
    s = cumulative(xy)
    if len(xy) < 2 or s[-1] <= 0:
        return None
    dense_s = np.arange(0.0, s[-1] + 1e-9, ALONGSIDE_STEP_M)
    dense = np.column_stack([np.interp(dense_s, s, xy[:, 0]), np.interp(dense_s, s, xy[:, 1])])
    pairs = cKDTree(dense).query_pairs(ALONGSIDE_M, output_type="ndarray")
    if not len(pairs):
        return None
    d = np.hypot(*(dense[pairs[:, 0]] - dense[pairs[:, 1]]).T)
    gap = np.abs(dense_s[pairs[:, 1]] - dense_s[pairs[:, 0]])
    far = pairs[(gap > junctions_core.OTHER_PART_RATIO * d + junctions_core.OTHER_PART_SLACK_M)
                & (gap >= ALONGSIDE_GAP_M)]
    later = np.unique(far.max(axis=1))
    if len(later) * ALONGSIDE_STEP_M < ALONGSIDE_RUN_M:
        return None
    k = int(later[0])
    earlier = int(far[far.max(axis=1) == k].min())
    return int(np.searchsorted(s, dense_s[k], side="left").clip(0, len(xy) - 1)), float(dense_s[earlier])


def alongside_message(along_m: float, earlier_m: float) -> str:
    """What to tell the person about `alongside`, in the build's words and the preview's."""
    return (f"The route comes back alongside itself {_distance(along_m)} along, beside where it was "
            f"{_distance(earlier_m)} along, usually by going round a roundabout and back up the other side of a "
            "divided road. A track can't run back beside itself: move the point near there, or finish the track "
            "before it turns.")


def u_turn_message(along_m: float) -> str:
    """What to tell the person about a `u_turn`, in the build's words and the preview's."""
    return (f"The route makes a U-turn {_distance(along_m)} along, usually because a point is on the far side of a "
            "divided road. A track can't go back the way it came: move the point near there onto the same side of "
            "the road as the route, or finish the track before it turns.")


def turns_back_message(retraced_m: float, along_m: float) -> str:
    """What to tell the person, in the build's words and the preview's."""
    twice = f" and drives {_distance(retraced_m)} of road twice" if retraced_m > RETRACE_LIMIT_M else ""
    return (f"The route turns back on itself {_distance(along_m)} along{twice}. A track can't go back the way it "
            "came: move the point near there onto the road you mean, or finish the track before it turns.")


def _distance(m: float) -> str:
    return f"{m:.0f} m" if m < 1000 else f"{m / 1000:.1f} km"


KEEP_TAGS = ("highway", "ref", "name", "lanes", "width", "surface", "oneway", "maxspeed", "bridge", "tunnel", "layer",
             "narrow", "lane_markings")


def _bbox(job: Job, margin_deg: float) -> tuple[float, float, float, float]:
    """West, south, east, north around the waypoints, in degrees."""
    lons = [w.lon for w in job.waypoints]
    lats = [w.lat for w in job.waypoints]
    return min(lons) - margin_deg, min(lats) - margin_deg, max(lons) + margin_deg, max(lats) + margin_deg


def runs(route: osm_graph.Route) -> list[dict]:
    """Consecutive stretches of the route on the same OSM way, with distances and kept tags."""
    s = cumulative(route.xy)
    out: list[dict] = []
    start = 0
    for i in range(1, len(route.seg_way) + 1):
        if i == len(route.seg_way) or route.seg_way[i] != route.seg_way[start]:
            way = route.ways[int(route.seg_way[start])]
            out.append({
                "from_m": round(float(s[start]), 2),
                "to_m": round(float(s[i]), 2),
                "way_id": way["id"],
                "tags": {k: way["tags"][k] for k in KEEP_TAGS if k in way["tags"]},
            })
            start = i
    return out


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    region = job_region(job.waypoints)

    data = osm_roads.for_bbox(_bbox(job, 0.02), settings)
    to_xy = region.to_xy()
    g = osm_graph.build(data, to_xy)
    wx, wy = to_xy.transform([w.lon for w in job.waypoints], [w.lat for w in job.waypoints])
    route = osm_graph.route(g, np.column_stack([wx, wy]))

    length = float(cumulative(route.xy)[-1])
    if length > job.max_length_km * 1000:
        raise RouteTooLong(f"The route is {length / 1000:.1f} km; the limit is {job.max_length_km:g} km.")
    back = turn_back(route.xy)
    if back is not None:
        raise RouteTurnsBack(turns_back_message(back[1], float(cumulative(route.xy)[back[0]])))
    u = u_turn(route.xy, ring_spans(route), segment_names(route))
    if u is not None:
        raise RouteTurnsBack(u_turn_message(float(cumulative(route.xy)[u])))
    beside = alongside(route.xy)
    if beside is not None:
        raise RouteTurnsBack(alongside_message(float(cumulative(route.xy)[beside[0]]), beside[1]))

    sharp = sharp_turns(route.xy)
    lon, lat = region.to_lonlat().transform(route.xy[:, 0], route.xy[:, 1])
    route_runs = runs(route)
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[round(a, 7), round(b, 7)] for a, b in zip(lon, lat)]},
            "properties": {"track_id": job.track_id, "region": region.key, "length_m": round(length, 1),
                           "retraced_m": round(route.retraced_m, 1), "runs": route_runs},
        }],
    }
    work.write_json(STAGE, "route.geojson", geojson)
    np.savez_compressed(
        work.stage(STAGE) / "route.npz",
        xy=route.xy,
        seg_run=_seg_run_index(route, route_runs),
        epsg=region.epsg,
    )
    work.write_json(STAGE, "runs.json", route_runs)
    work.write_json(STAGE, OSM_SELECTION, osm_roads.selection(settings))
    summary = {"length_m": round(length, 1), "retraced_m": round(route.retraced_m, 1), "ways": len(route_runs),
               "region": region.key, "sharp_turns": sharp}
    summary |= _write_crossings(g, route, region, work)
    summary |= _write_crossing_lines(data, g, route, region, settings, work)
    if job.junctions == Junctions.CLOSED:
        found = junctions_core.find(g, route.xy, to_lonlat=region.to_lonlat())
        work.write_json(STAGE, JUNCTIONS, {"version": found.pop("version"), "crs": f"EPSG:{region.epsg}", **found})
        summary |= {"junctions": len(found["junctions"]), "roundabouts": len(found["roundabouts"])}
        found_ramps = ramps_core.find(g, route.xy, to_lonlat=region.to_lonlat())
        work.write_json(STAGE, RAMPS, {"version": found_ramps.pop("version"), "crs": f"EPSG:{region.epsg}",
                                       **found_ramps})
        summary |= {"ramps": len(found_ramps["ramps"])}
    else:
        (work.stage(STAGE) / JUNCTIONS).unlink(missing_ok=True)  # built without them, as before they were found
        (work.stage(STAGE) / RAMPS).unlink(missing_ok=True)
    return summary


def _write_crossings(g: osm_graph.RoadGraph, route: osm_graph.Route, region, work: WorkDir) -> dict:
    """Write crossings.json (`core.crossings`) whatever the job's junctions option: the profile is built on it."""
    found = crossings_core.find(g, route.xy, route.seg_way, to_lonlat=region.to_lonlat())
    _signed_heights(found["crossings"], g, route)
    work.write_json(STAGE, CROSSINGS, {"version": found.pop("version"), "crs": f"EPSG:{region.epsg}", **found})
    return {"crossings": {"over": sum(c["relation"] == "over" for c in found["crossings"]),
                          "under": sum(c["relation"] == "under" for c in found["crossings"]),
                          "unresolved": len(found["unresolved"])}}


def _signed_heights(found: list[dict], g: osm_graph.RoadGraph, route: osm_graph.Route) -> None:
    """Add to each crossing the clearances signed under it (`crossings.signed_under`): the route's own `maxheight`
    there (`route.maxheight`), and for another road's bridge over it, those of the other roads under that bridge
    (`maxheight_beside`: the other carriageway's, usually)."""
    tags = {w["id"]: w.get("tags") or {} for w in g.ways}
    route_ids = {route.ways[int(w)]["id"] for w in route.seg_way}
    for c in found:
        mh = tags.get(c["route"]["way_id"], {}).get("maxheight")
        if mh is not None:
            c["route"]["maxheight"] = mh
        if c["relation"] == "over" and c.get("bridge"):
            beside = [r["tags"]["maxheight"]
                      for r in rail_and_paths.roads_crossed(np.asarray(c["line"], float), g, route_ways=route_ids)
                      if not r["route"] and r["way_id"] != c["way_id"] and "maxheight" in r["tags"]]
            if beside:
                c["maxheight_beside"] = beside


LINES_MARGIN_DEG = 0.003  # railways and paths are fetched in the route's box with this much more round it


def _write_crossing_lines(data: dict, g: osm_graph.RoadGraph, route: osm_graph.Route, region, settings: Settings,
                          work: WorkDir) -> dict:
    """Write crossing_lines.json (`core.rail_and_paths`): the railways and paths crossing over or under the route, each
    over it with the whole bridge it's part of, its approaches and the roads that bridge passes over. Without them
    (an OSM index built before they were kept, or Overpass not answering) the file lists none and says why, and the
    build goes on."""
    head = {"version": rail_and_paths.FORMAT_VERSION, "crs": f"EPSG:{region.epsg}"}
    lon, lat = region.to_lonlat().transform(route.xy[:, 0], route.xy[:, 1])
    m = LINES_MARGIN_DEG
    bbox = (float(min(lon)) - m, float(min(lat)) - m, float(max(lon)) + m, float(max(lat)) + m)
    try:
        lines = osm_roads.railways_and_paths_for_bbox(bbox, settings)
    except osm_roads.INDEX_ERRORS:
        why = "the OSM index was built before railways and paths were kept: rebuild it"
    except httpx.HTTPError:
        why = "the Overpass server didn't answer"
    else:
        why = None
    if why is not None:
        work.write_json(STAGE, CROSSING_LINES, {**head, "unavailable": why, "crossings": [], "unresolved": []})
        return {"crossing_lines": {"unavailable": why}}
    if not settings.osm_index:  # from Overpass: ATTRIBUTION.txt names these filters too (`osm_roads.selection_note`)
        work.write_json(STAGE, OSM_SELECTION, {**work.read_json(STAGE, OSM_SELECTION),
                                               "overpass_lines_filter": osm_roads.LINES_SELECTION})
    to_xy = region.to_xy()
    route_ids = {route.ways[int(w)]["id"] for w in route.seg_way}
    # The route's own ways with the railways and paths, as one network, so a node they share is a junction there.
    road_ways = [e for e in data["elements"] if e["type"] == "way" and e["id"] in route_ids]
    used = {n for w in road_ways for n in w["nodes"]}
    road_nodes = [e for e in data["elements"] if e["type"] == "node" and e["id"] in used]
    line_ways = [e for e in lines["elements"] if e["type"] == "way"]
    line_nodes = [e for e in lines["elements"] if e["type"] == "node"]
    net = osm_graph.build({"elements": road_nodes + line_nodes + road_ways + line_ways}, to_xy)
    index_of = {w["id"]: i for i, w in enumerate(net.ways)}
    seg_way = np.array([index_of[route.ways[int(w)]["id"]] for w in route.seg_way], dtype=np.int64)
    found = rail_and_paths.find(net, route.xy, seg_way, to_lonlat=region.to_lonlat())
    if line_nodes:
        ids = [e["id"] for e in line_nodes]
        x, y = to_xy.transform([e["lon"] for e in line_nodes], [e["lat"] for e in line_nodes])
        where = dict(zip(ids, zip(x, y)))
    else:
        where = {}
    ways = {w["id"]: rail_and_paths.LineWay(w["nodes"], np.array([where[n] for n in w["nodes"]], dtype=float),
                                            w.get("tags") or {})
            for w in line_ways if len(w["nodes"]) >= 2 and all(n in where for n in w["nodes"])}
    route_tags = {w["id"]: w.get("tags") or {} for w in road_ways}
    for c in found["crossings"]:
        if "maxheight" in route_tags.get(c["route"]["way_id"], {}):
            c["route"]["maxheight"] = route_tags[c["route"]["way_id"]]["maxheight"]  # a signed clearance, for the notes
        st = (rail_and_paths.structure(ways, c["way_id"], c["xy"])
              if c["relation"] == "over" and c["way_id"] in ways else {"deck": None})
        c["structure"] = st
        if st.get("deck"):
            c["roads_crossed"] = rail_and_paths.roads_crossed(np.array(st["deck"]), g, route_ways=route_ids)
    work.write_json(STAGE, CROSSING_LINES, {**head, "version": found.pop("version"), **found})
    return {"crossing_lines": {"over": sum(c["relation"] == "over" for c in found["crossings"]),
                               "under": sum(c["relation"] == "under" for c in found["crossings"]),
                               "unresolved": len(found["unresolved"])}}


def crossing_lines(work: WorkDir) -> list[dict]:
    """The railways and paths crossing over or under the route, as crossing_lines.json lists them: none for a route
    stage run before they were found, or without them."""
    path = work.root / STAGE / CROSSING_LINES
    return work.read_json(STAGE, CROSSING_LINES)["crossings"] if path.exists() else []


def crossings(work: WorkDir) -> list[dict]:
    """The roads crossing over or under the route, as crossings.json lists them: none for a route stage run before
    crossings were found."""
    path = work.root / STAGE / CROSSINGS
    return work.read_json(STAGE, CROSSINGS)["crossings"] if path.exists() else []


def junctions(work: WorkDir, job: Job) -> list[dict]:
    """The side-road junctions the route stage found, as junctions.json lists them: none for a job built without
    them, or a route stage run before they were found."""
    path = work.root / STAGE / JUNCTIONS
    if job.junctions != Junctions.CLOSED or not path.exists():
        return []
    return work.read_json(STAGE, JUNCTIONS)["junctions"]


def ramps(work: WorkDir, job: Job) -> list[dict]:
    """The slip ramps the route stage found (`core.ramps`), as ramps.json lists them: none for a job built without
    junctions, or a route stage run before ramps were found."""
    path = work.root / STAGE / RAMPS
    if job.junctions != Junctions.CLOSED or not path.exists():
        return []
    return work.read_json(STAGE, RAMPS)["ramps"]


def roundabouts(work: WorkDir, job: Job) -> list[dict]:
    """The roundabouts the route drives, as junctions.json lists them (`core.junctions`): none for a job built without
    junctions, or a route stage run before they were recorded in the order they're driven."""
    path = work.root / STAGE / JUNCTIONS
    if job.junctions != Junctions.CLOSED or not path.exists():
        return []
    return [r for r in work.read_json(STAGE, JUNCTIONS).get("roundabouts", [])
            if r.get("on_route") and "route_ring_index" in r]


def osm_selection(work: WorkDir, settings: Settings) -> dict[str, str]:
    """How this route's roads were selected from OpenStreetMap, as `run` recorded it. A work folder routed before
    that was recorded gets the answer for the current settings."""
    if (work.root / STAGE / OSM_SELECTION).exists():
        return work.read_json(STAGE, OSM_SELECTION)
    return osm_roads.selection(settings)


SHARP_TURN_DEG = 100.0


def sharp_turns(xy: np.ndarray, limit_deg: float = SHARP_TURN_DEG) -> list[dict]:
    """Places where the route turns harder than `limit_deg` at a single node, for the stage's summary.

    A real hairpin is drawn in OSM as many nodes, each turning a little. One node turning 100°+ is a turn
    into a side street at an acute junction (Richmond's Bridge Street into Wellington Street turns 108°),
    which the profile stage rounds to a drivable corner, or, past 150°, usually the route doubling back because a
    waypoint landed on the wrong road. `RouteTurnsBack` refuses that where it drives road twice; validation
    reports any other with its place.
    """
    seg = np.diff(xy, axis=0)
    keep = np.hypot(*seg.T) > 1e-6
    seg = seg[keep]
    if len(seg) < 2:
        return []
    h = np.arctan2(seg[:, 1], seg[:, 0])
    turn = np.degrees(np.abs(np.angle(np.exp(1j * np.diff(h)))))
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*seg.T))])
    return [{"at_m": round(float(cum[i + 1]), 1), "turn_deg": round(float(turn[i]), 1)}
            for i in np.flatnonzero(turn > limit_deg)]


def _seg_run_index(route: osm_graph.Route, route_runs: list[dict]) -> np.ndarray:
    """For each polyline segment, the index of the run it belongs to."""
    out = np.zeros(len(route.seg_way), dtype=np.int64)
    r = 0
    for i in range(1, len(route.seg_way)):
        if route.seg_way[i] != route.seg_way[i - 1]:
            r += 1
        out[i] = r
    assert r == len(route_runs) - 1
    return out
