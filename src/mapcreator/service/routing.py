"""Route previews for the map page: where would a car go between these points?

The browser wants a quick answer while someone drags markers about, so this
is separate from the build's own routing (`pipeline/route_stage.py`), which
runs later on the worker and is the one that counts. Two implementations:

- `DevRouter` reuses the build's OSM graph, reading roads from the local OSM
  index when `osm_index` is set, else from an Overpass server. Fine for trying
  the service on one machine; too slow for a public map, and with Overpass,
  too heavy on a shared public server.
- `GraphHopperRouter` asks a self-hosted GraphHopper with a `car` profile.

Region checks happen before either router is asked anything, so a point in
an unsupported state costs nothing. Every exception here carries a message
that is safe to show the person using the map.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypedDict

import httpx
import numpy as np

from mapcreator import config
from mapcreator.core.centreline import cumulative
from mapcreator.job import Waypoint
from mapcreator.providers.osm import graph as osm_graph
from mapcreator.providers.osm import overpass
from mapcreator.providers.osm import roads as osm_roads
from mapcreator.regions import REGIONS, OutOfArea, Region, job_region, region_for

__all__ = ["DevRouter", "GraphHopperRouter", "NoRoute", "OutOfArea", "RoutePreview", "Router", "RouterUnavailable",
           "RoutingError", "check_region", "supported_regions"]


class RoutingError(Exception):
    """A routing problem whose message can be shown to the user as is."""


class NoRoute(RoutingError):
    pass


class RouterUnavailable(RoutingError):
    pass


class RoutePreview(TypedDict):
    coordinates: list[list[float]]  # [[lon, lat], ...]
    length_m: float


class Router(Protocol):
    def preview(self, waypoints: Sequence[Waypoint]) -> RoutePreview: ...


def supported_regions() -> str:
    return ", ".join(r.name for r in REGIONS)


def check_region(waypoints: Sequence[Waypoint]) -> Region:
    """The one region every point is in; `OutOfArea` names the first point that isn't."""
    if not waypoints:
        raise OutOfArea("No points given.")
    for i, w in enumerate(waypoints, start=1):
        try:
            region_for(w.lon, w.lat)
        except OutOfArea:
            raise OutOfArea(f"Point {i} is outside the supported area. Supported regions: "
                            f"{supported_regions()}.") from None
    try:
        return job_region(waypoints)  # the same choice the pipeline makes
    except OutOfArea:
        raise OutOfArea("All points must be in the same region.") from None


def _preview(lon: Sequence[float], lat: Sequence[float], length_m: float) -> RoutePreview:
    return {"coordinates": [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)],
            "length_m": round(float(length_m), 1)}


class DevRouter:
    """The build's own OSM routing, over the local OSM index or Overpass. For local development only."""

    # Every road in the points' bounding box is read and built into a graph, so keep the box modest.
    MAX_SPAN_DEG = 0.75
    MARGIN_DEG = 0.02

    def __init__(self, settings: config.Settings | None = None, *,
                 fetch: Callable[[str, config.Settings], dict] = overpass.fetch):
        self._settings = settings
        self._fetch = fetch  # used only when roads come from Overpass

    def preview(self, waypoints: Sequence[Waypoint]) -> RoutePreview:
        region = check_region(waypoints)
        lons = [w.lon for w in waypoints]
        lats = [w.lat for w in waypoints]
        if max(lons) - min(lons) > self.MAX_SPAN_DEG or max(lats) - min(lats) > self.MAX_SPAN_DEG:
            raise NoRoute("The points are too far apart for a preview.")
        m = self.MARGIN_DEG
        bbox = (min(lons) - m, min(lats) - m, max(lons) + m, max(lats) + m)
        settings = self._settings or config.load()
        try:
            data = osm_roads.for_bbox(bbox, settings, fetch=self._fetch)
        except httpx.HTTPError as e:
            raise RouterUnavailable("The map data service didn't answer. Try again in a minute.") from e
        except osm_roads.INDEX_ERRORS as e:
            raise RouterUnavailable("The map data isn't available right now. Try again later.") from e
        to_xy = region.to_xy()
        wx, wy = to_xy.transform(lons, lats)
        try:
            route = osm_graph.route(osm_graph.build(data, to_xy), np.column_stack([wx, wy]))
        except osm_graph.RouteError as e:
            raise NoRoute(_sentence(str(e))) from e
        lon, lat = region.to_lonlat().transform(route.xy[:, 0], route.xy[:, 1])
        return _preview(lon, lat, cumulative(route.xy)[-1])


# GraphHopper names the reason for a 400 in hints[].details (Java exception class names).
_GH_REASONS = {
    "PointOutOfBoundsException": (OutOfArea, "{which} is outside the area the router covers."),
    "PointNotFoundException": (NoRoute, "{which} isn't near a road a car can use."),
    "ConnectionNotFoundException": (NoRoute, "No drivable route connects these points."),
}


class GraphHopperRouter:
    """A self-hosted GraphHopper (https://github.com/graphhopper/graphhopper) with a `car` profile."""

    def __init__(self, base_url: str, *, client: httpx.Client | None = None, timeout_s: float = 10.0,
                 profile: str = "car"):
        self._url = base_url.rstrip("/") + "/route"
        self._client = client or httpx.Client(timeout=timeout_s, follow_redirects=False)
        self._profile = profile

    def preview(self, waypoints: Sequence[Waypoint]) -> RoutePreview:
        check_region(waypoints)
        body = {
            "points": [[w.lon, w.lat] for w in waypoints],  # GraphHopper's POST body takes [lon, lat]
            "profile": self._profile,
            "points_encoded": False,
            "instructions": False,
            "calc_points": True,
        }
        try:
            r = self._client.post(self._url, json=body)
        except httpx.HTTPError as e:
            raise RouterUnavailable("The route preview service didn't answer. Try again in a minute.") from e
        if r.status_code == 200:
            return self._parse(r)
        if 400 <= r.status_code < 500:
            raise self._error(r)
        raise RouterUnavailable("The route preview service is having trouble. Try again in a minute.")

    @staticmethod
    def _parse(r: httpx.Response) -> RoutePreview:
        try:
            path = r.json()["paths"][0]
            coords = path["points"]["coordinates"]
            length = float(path["distance"])
            lon = [float(c[0]) for c in coords]
            lat = [float(c[1]) for c in coords]  # a third value, if present, is elevation
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise RouterUnavailable("The route preview service sent an answer we couldn't read.") from e
        if len(lon) < 2:
            raise NoRoute("No drivable route connects these points.")
        return _preview(lon, lat, length)

    @staticmethod
    def _error(r: httpx.Response) -> RoutingError:
        try:
            hints = r.json().get("hints") or []
        except ValueError:
            hints = []
        for hint in hints:
            details = str(hint.get("details", "")) if isinstance(hint, dict) else ""
            for name, (kind, text) in _GH_REASONS.items():
                if details.endswith(name):
                    index = hint.get("point_index")
                    which = f"Point {index + 1}" if isinstance(index, int) else "One of the points"
                    return kind(text.format(which=which))
        return NoRoute("The router couldn't find a route between these points.")


def _sentence(text: str) -> str:
    text = text.strip()
    return (text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")) if text else text
