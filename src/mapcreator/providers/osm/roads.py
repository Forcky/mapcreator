"""Drivable OSM roads in a bounding box: from the local index when one is configured, else Overpass.

Production sets `osm_index` to a file built by `index.py` from a Geofabrik
extract. Without one, development falls back to an Overpass server, whose
answers are cached on disk so a repeated build never asks twice. Index
answers aren't cached: a query takes milliseconds, and a cache would go
stale whenever the index is rebuilt from a newer extract.

Both sources return the same Overpass-style JSON, which `graph.build` reads.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from mapcreator.config import Settings
from mapcreator.providers.osm import index, overpass

# Errors that mean the configured index can't be read, as opposed to a problem with the road.
INDEX_ERRORS = (index.IndexUnavailable, sqlite3.Error)


def for_bbox(bbox: tuple[float, float, float, float], settings: Settings, *,
             fetch: Callable[[str, Settings], dict] | None = None) -> dict:
    """Overpass-style JSON for the drivable roads crossing `bbox` (west, south, east, north, WGS84).

    `fetch` replaces `overpass.fetch` when the roads come from Overpass (for tests).
    """
    if settings.osm_index:
        return index.query(settings.osm_index, bbox)
    west, south, east, north = bbox
    return (fetch or overpass.fetch)(overpass.query_for_bbox(south, west, north, east), settings)


def selection(settings: Settings) -> dict[str, str]:
    """How `for_bbox` picks the roads: the index's extract and its OSM data timestamp, or the Overpass filter.
    The route stage records it, and ATTRIBUTION.txt says it (`selection_note`)."""
    if settings.osm_index:
        meta = index.metadata(settings.osm_index)
        return {"index": meta.get("source", ""), "osm_timestamp": meta.get("osm_timestamp", "")}
    return {"overpass_filter": overpass.ROADS_FILTER}


def selection_note(sel: dict[str, str], *, build: str, source_url: str = "") -> str:
    """Where a track's recipients find how its OpenStreetMap data was selected, which ODbL 4.6 asks for when a
    track is made from a Derivative Database such as the index. `build` is `buildinfo.version()`, and
    `source_url` is where the code is published (`Settings.source_url`), left out when empty."""
    at = f" at {source_url}" if source_url else ""
    if "index" in sel:
        extract = sel["index"] or "an OpenStreetMap extract"
        of = f", with data as of {sel['osm_timestamp']}" if sel.get("osm_timestamp") else ""
        return (f"The OpenStreetMap data was selected with mapcreator's index builder "
                f"(src/mapcreator/providers/osm/index.py{at}, build {build}) from {extract}{of}.")
    return (f"The OpenStreetMap data was selected from an Overpass API server with the filter "
            f"{sel['overpass_filter']} around the route (src/mapcreator/providers/osm/overpass.py{at}, "
            f"build {build}).")
