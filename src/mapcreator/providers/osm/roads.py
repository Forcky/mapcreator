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
