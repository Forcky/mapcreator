"""Stage 1: snap the job's waypoints to OSM and write the routed polyline with its tags.

Roads come from the local OSM index when `osm_index` is set (production), else from Overpass
(development); see `providers/osm/roads.py`.
"""

from __future__ import annotations

import numpy as np

from mapcreator.config import Settings
from mapcreator.core import centreline
from mapcreator.core import crossings as crossings_core
from mapcreator.core import junctions as junctions_core
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
    if job.junctions == Junctions.CLOSED:
        found = junctions_core.find(g, route.xy, to_lonlat=region.to_lonlat())
        work.write_json(STAGE, JUNCTIONS, {"version": found.pop("version"), "crs": f"EPSG:{region.epsg}", **found})
        summary |= {"junctions": len(found["junctions"]), "roundabouts": len(found["roundabouts"])}
    else:
        (work.stage(STAGE) / JUNCTIONS).unlink(missing_ok=True)  # built without them, as before they were found
    return summary


def _write_crossings(g: osm_graph.RoadGraph, route: osm_graph.Route, region, work: WorkDir) -> dict:
    """Write crossings.json (`core.crossings`) whatever the job's junctions option: the profile is built on it."""
    found = crossings_core.find(g, route.xy, route.seg_way, to_lonlat=region.to_lonlat())
    work.write_json(STAGE, CROSSINGS, {"version": found.pop("version"), "crs": f"EPSG:{region.epsg}", **found})
    return {"crossings": {"over": sum(c["relation"] == "over" for c in found["crossings"]),
                          "under": sum(c["relation"] == "under" for c in found["crossings"]),
                          "unresolved": len(found["unresolved"])}}


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
