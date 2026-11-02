"""Stage 1: snap the job's waypoints to OSM and write the routed polyline with its tags."""

from __future__ import annotations

import numpy as np

from mapcreator.config import Settings
from mapcreator.core.centreline import cumulative
from mapcreator.job import Job
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.osm import graph as osm_graph
from mapcreator.providers.osm import overpass
from mapcreator.regions import same_region

STAGE = "01_route"
KEEP_TAGS = ("highway", "ref", "name", "lanes", "width", "surface", "oneway", "maxspeed", "bridge", "tunnel", "layer")


def _bbox(job: Job, margin_deg: float) -> tuple[float, float, float, float]:
    lons = [w.lon for w in job.waypoints]
    lats = [w.lat for w in job.waypoints]
    return min(lats) - margin_deg, min(lons) - margin_deg, max(lats) + margin_deg, max(lons) + margin_deg


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
    region = same_region([(w.lon, w.lat) for w in job.waypoints])

    data = overpass.fetch(overpass.query_for_bbox(*_bbox(job, 0.02)), settings)
    to_xy = region.to_xy()
    g = osm_graph.build(data, to_xy)
    wx, wy = to_xy.transform([w.lon for w in job.waypoints], [w.lat for w in job.waypoints])
    route = osm_graph.route(g, np.column_stack([wx, wy]))

    length = float(cumulative(route.xy)[-1])
    if length > job.max_length_km * 1000:
        raise ValueError(f"route is {length / 1000:.1f} km; the limit is {job.max_length_km:g} km")

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
    return {"length_m": round(length, 1), "retraced_m": round(route.retraced_m, 1), "ways": len(route_runs),
            "region": region.key, "sharp_turns": sharp}


SHARP_TURN_DEG = 100.0


def sharp_turns(xy: np.ndarray, limit_deg: float = SHARP_TURN_DEG) -> list[dict]:
    """Places where the route turns harder than `limit_deg` at a single node.

    A real hairpin is drawn in OSM as many nodes, each turning a little. One
    node turning 100°+ is nearly always the route doubling back at a junction,
    usually because a waypoint landed on the wrong road. That can't be smoothed
    into a drivable bend, so it's reported for the person picking the route.
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
