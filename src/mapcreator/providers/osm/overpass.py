"""Fetch drivable OSM ways, and the railways and paths that cross them, from an Overpass server, cached on disk.

For development only. The public Overpass servers are a shared resource with
fair-use limits, so every response is cached and a repeated build never asks
twice. Production reads a local Geofabrik extract instead.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx

from mapcreator.config import Settings
from mapcreator.providers import cache as cache_io

# Classes a car may be routed along. `service` and `track` are left out so a route
# between sparse waypoints doesn't wander through car parks or forestry tracks.
DRIVABLE = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|road|"
    "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"
)
# The ways a query keeps, before its bounding box. ATTRIBUTION.txt quotes it when the roads came from Overpass.
ROADS_FILTER = f'way["highway"~"^({DRIVABLE})$"]'
# What can cross a road without being one: railways (in use, disused or abandoned) and paths. Never routed on. They
# are kept so a footbridge or rail bridge over the route, or a path or line under its bridge, can be found and drawn.
RAILWAYS = "rail|light_rail|narrow_gauge|disused|abandoned"
PATHS = "footway|cycleway|path|pedestrian|steps|bridleway"
RAILWAYS_FILTER = f'way["railway"~"^({RAILWAYS})$"]'
PATHS_FILTER = f'way["highway"~"^({PATHS})$"]'


def _box(south: float, west: float, north: float, east: float) -> str:
    return f"({south:.6f},{west:.6f},{north:.6f},{east:.6f})"


def query_for_bbox(south: float, west: float, north: float, east: float) -> str:
    return (
        "[out:json][timeout:180];\n"
        f"{ROADS_FILTER}{_box(south, west, north, east)};\n"
        "(._;>;);\nout body;\n"
    )


def railways_and_paths_query_for_bbox(south: float, west: float, north: float, east: float) -> str:
    """The railways and paths in a box, with their nodes: the ways the local index keeps of them (`index.py`)."""
    box = _box(south, west, north, east)
    return (
        "[out:json][timeout:180];\n"
        f"({RAILWAYS_FILTER}{box};{PATHS_FILTER}{box};);\n"
        "(._;>;);\nout body;\n"
    )


def fetch(query: str, settings: Settings, *, refresh: bool = False) -> dict:
    cache = settings.cache_dir / "osm" / (hashlib.sha256(query.encode()).hexdigest()[:24] + ".json")
    if not refresh:
        cached = cache_io.read_json(cache)
        if cached is not None:
            return cached
    data = _post(query, settings)
    # Overpass reports a timeout or memory limit as a normal answer with a "remark" and partial (or no)
    # elements. Use it, but don't keep it, or every later build in this box would see the partial roads.
    if not data.get("remark"):
        cache_io.write_json(cache, data)
    return data


def _post(query: str, settings: Settings) -> dict:
    headers = {"User-Agent": settings.user_agent}
    for attempt in range(4):
        try:
            r = httpx.post(settings.overpass_url, data={"data": query}, timeout=240, headers=headers)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def load_cached(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
