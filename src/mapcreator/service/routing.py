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
an unsupported state costs nothing. A preview that drives road twice says
where it turns back (`turns_back`), in the words the build would refuse it
with, so the page can stop it before anyone submits. A preview through a small
roundabout says so (`roundabouts`, `core.roundabouts.preview_notes`): the
route's path round a ring of 10.3 m radius or less may bend tighter than the
5 m a build accepts. GraphHopper names the stretches on a roundabout as a path
detail; the ring's radius is fitted to the route's points round it. Every
exception here carries a message that is safe to show the person using the map.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NotRequired, Protocol, TypedDict

import httpx
import numpy as np

from mapcreator import config
from mapcreator.core import junctions as junctions_core
from mapcreator.core import roundabouts
from mapcreator.core.centreline import cumulative
from mapcreator.job import Waypoint
from mapcreator.pipeline import route_stage
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


class TurnBack(TypedDict):
    lon: float  # where the route first goes back over its own road
    lat: float
    along_m: float
    retraced_m: float  # road driven twice (0 for a reversal on the spot)
    message: str  # the build's own words (`route_stage.RouteTurnsBack`)


class RoundaboutNote(TypedDict):
    lon: float  # the roundabout, midway round the route's arc of it
    lat: float
    along_m: float
    radius_m: float | None  # the ring's radius, where known
    tightest_m: float | None  # the route's tightest bend round it, as validation would judge it
    message: str


class RoutePreview(TypedDict):
    coordinates: list[list[float]]  # [[lon, lat], ...]
    length_m: float
    turns_back: NotRequired[TurnBack]  # only when the route drives road twice, which no build accepts
    roundabouts: NotRequired[list[RoundaboutNote]]  # only for a small roundabout, or one the build may find too tight


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


def _preview(lon: Sequence[float], lat: Sequence[float], length_m: float, xy: np.ndarray,
             rings: Sequence[dict] = (), names: Sequence[str | None] | None = None) -> RoutePreview:
    """The preview for the page; `xy` is the same line in the region's metres, to measure any road driven twice.
    `rings`: the roundabouts it drives, as `roundabouts.preview_notes` takes them (`from_m`, `to_m` along `xy`,
    `radius_m`). `names`: the road each segment of `xy` is on, where the router says (`route_stage.u_turn`)."""
    out: RoutePreview = {"coordinates": [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)],
                         "length_m": round(float(length_m), 1)}
    if rings:
        s = cumulative(xy)
        notes = []
        for note in roundabouts.preview_notes(xy, list(rings)):
            notes.append({"lon": round(float(np.interp(note["along_m"], s, lon)), 6),
                          "lat": round(float(np.interp(note["along_m"], s, lat)), 6), **note})
        if notes:
            out["roundabouts"] = notes
    found = route_stage.turn_back(xy)
    if found is not None:
        back, retraced_m = found
        along_m = float(cumulative(xy)[back])
        out["turns_back"] = {"lon": round(float(lon[back]), 6), "lat": round(float(lat[back]), 6),
                             "along_m": round(along_m, 1), "retraced_m": round(retraced_m, 1),
                             "message": route_stage.turns_back_message(retraced_m, along_m)}
        return out
    # A U-turn through a divided road's median: refused the same way, though no road is driven twice.
    u = route_stage.u_turn(xy, [(r["from_m"], r["to_m"]) for r in rings], names)
    if u is not None:
        along_m = float(cumulative(xy)[u])
        out["turns_back"] = {"lon": round(float(lon[u]), 6), "lat": round(float(lat[u]), 6),
                             "along_m": round(along_m, 1), "retraced_m": 0.0,
                             "message": route_stage.u_turn_message(along_m)}
    return out


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
            graph = osm_graph.build(data, to_xy)
            route = osm_graph.route(graph, np.column_stack([wx, wy]))
        except osm_graph.RouteError as e:
            raise NoRoute(_sentence(str(e))) from e
        lon, lat = region.to_lonlat().transform(route.xy[:, 0], route.xy[:, 1])
        # The roundabouts it drives, as the build's route stage records them (`core.junctions`).
        rings = [{"from_m": r["route_from_m"], "to_m": r["route_to_m"], "radius_m": r["radius_m"]}
                 for r in junctions_core.find(graph, route.xy)["roundabouts"] if r["on_route"]]
        return _preview(lon, lat, cumulative(route.xy)[-1], route.xy, rings, route_stage.segment_names(route))


# GraphHopper names the reason for a 400 in hints[].details (Java exception class names).
_GH_REASONS = {
    "PointOutOfBoundsException": (OutOfArea, "{which} is outside the area the router covers."),
    "PointNotFoundException": (NoRoute, "{which} isn't near a road a car can use."),
    "ConnectionNotFoundException": (NoRoute, "No drivable route connects these points."),
}


class GraphHopperRouter:
    """A self-hosted GraphHopper (https://github.com/graphhopper/graphhopper) with a `car` profile."""

    # The path detail naming the stretches on a roundabout. GraphHopper 11 built from deploy/graphhopper/config.yml
    # answers with it (checked over a Tasmania extract). A router that says it has no such detail is asked again
    # without it, and then not again: its previews have no roundabout notes.
    ROUNDABOUT_DETAIL = "roundabout"
    # The road each stretch is on, for a U-turn's check that the route left the road and came back onto it (checked
    # on the same GraphHopper 11; GraphHopper keeps street names in every graph).
    STREET_DETAIL = "street_name"

    def __init__(self, base_url: str, *, client: httpx.Client | None = None, timeout_s: float = 10.0,
                 profile: str = "car"):
        self._url = base_url.rstrip("/") + "/route"
        self._client = client or httpx.Client(timeout=timeout_s, follow_redirects=False)
        self._profile = profile
        self._details = True

    def preview(self, waypoints: Sequence[Waypoint]) -> RoutePreview:
        region = check_region(waypoints)
        body = {
            "points": [[w.lon, w.lat] for w in waypoints],  # GraphHopper's POST body takes [lon, lat]
            "profile": self._profile,
            "points_encoded": False,
            "instructions": False,
            "calc_points": True,
        }
        try:
            r = self._client.post(self._url, json={**body, "details": [self.ROUNDABOUT_DETAIL, self.STREET_DETAIL]}
                                  if self._details else body)
            if self._details and r.status_code == 400 and self._lacks_detail(r):
                self._details = False
                r = self._client.post(self._url, json=body)
        except httpx.HTTPError as e:
            raise RouterUnavailable("The route preview service didn't answer. Try again in a minute.") from e
        if r.status_code == 200:
            return self._parse(r, region)
        if 400 <= r.status_code < 500:
            raise self._error(r)
        raise RouterUnavailable("The route preview service is having trouble. Try again in a minute.")

    @staticmethod
    def _lacks_detail(r: httpx.Response) -> bool:
        """Whether a 400 says the graph has no such path detail (GraphHopper: "Cannot find the path details")."""
        try:
            return "path details" in str(r.json().get("message", "")).lower()
        except ValueError:
            return False

    @staticmethod
    def _parse(r: httpx.Response, region: Region) -> RoutePreview:
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
        x, y = region.to_xy().transform(lon, lat)
        xy = np.column_stack([x, y])
        return _preview(lon, lat, length, xy, GraphHopperRouter._rings(path, xy),
                        GraphHopperRouter._names(path, len(xy) - 1))

    @staticmethod
    def _names(path: dict, segments: int) -> list[str | None] | None:
        """The road each segment of the path is on, from its `street_name` detail ([from point, to point, name]), or
        None without the detail (a U-turn is then judged on its shape alone)."""
        details = path.get("details")
        spans = details.get(GraphHopperRouter.STREET_DETAIL) if isinstance(details, dict) else None
        if not isinstance(spans, list):
            return None
        names: list[str | None] = [None] * segments
        try:
            for a, b, name in spans:
                for k in range(max(int(a), 0), min(int(b), segments)):
                    names[k] = str(name) if name is not None else None
        except (TypeError, ValueError):
            return None
        return names

    @staticmethod
    def _rings(path: dict, xy: np.ndarray) -> list[dict]:
        """The roundabouts the path drives, from its `roundabout` detail ([from point, to point, on a roundabout]):
        where along it, and the ring's radius fitted to the path's points round it (None with too few points)."""
        try:
            spans = [(int(a), int(b)) for a, b, on in path.get("details", {}).get(GraphHopperRouter.ROUNDABOUT_DETAIL, [])
                     if on is True]
        except (TypeError, ValueError):
            return []
        s = cumulative(xy)
        out = []
        for a, b in spans:
            if not 0 <= a < b < len(xy):
                continue
            out.append({"from_m": float(s[a]), "to_m": float(s[b]),
                        "radius_m": roundabouts.fit_radius(xy[a:b + 1])})
        return out

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
