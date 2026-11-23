"""Drivable OSM roads in a bounding box (and the railways and paths that cross them, the service roads and tracks
that meet them, and the tagged nodes on them): from the local index when one is configured, else Overpass.

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


def railways_and_paths_for_bbox(bbox: tuple[float, float, float, float], settings: Settings, *,
                                fetch: Callable[[str, Settings], dict] | None = None) -> dict:
    """Overpass-style JSON for the railways and paths crossing `bbox` (west, south, east, north, WGS84), from the
    same source as `for_bbox`. They are never routed on: they're what may cross the route on a bridge or under one.

    Overpass's answer is narrowed to what the index keeps (`index.as_indexed`): the same ways, tags and nodes, so a
    build finds the same crossings from either source. An index built before railways and paths were kept raises
    `index.IndexUnavailable` (one of `INDEX_ERRORS`), and the caller goes on without them.
    """
    if settings.osm_index:
        return index.railways_and_paths(settings.osm_index, bbox)
    west, south, east, north = bbox
    data = (fetch or overpass.fetch)(overpass.railways_and_paths_query_for_bbox(south, west, north, east), settings)
    return index.as_indexed(data, ("railway", "path"))


def service_ways_for_bbox(bbox: tuple[float, float, float, float], settings: Settings, *,
                          fetch: Callable[[str, Settings], dict] | None = None) -> dict:
    """Overpass-style JSON for the service roads and tracks crossing `bbox` (west, south, east, north, WGS84), from
    the same source as `for_bbox`: driveways among them. They are never routed on, given a stub or taken for a
    crossing; they're only where roadside objects keep out of a mouth (`core.junctions.find_driveways`).

    Overpass's answer is narrowed to what the index keeps (`index.as_indexed`), as for the railways and paths. An index
    built before they were kept raises `index.IndexUnavailable` (one of `INDEX_ERRORS`), and the caller goes on
    without them.
    """
    if settings.osm_index:
        return index.service_ways(settings.osm_index, bbox)
    west, south, east, north = bbox
    data = (fetch or overpass.fetch)(overpass.service_query_for_bbox(south, west, north, east), settings)
    return index.as_indexed(data, ("service",))


def road_nodes_for_bbox(bbox: tuple[float, float, float, float], settings: Settings, *,
                        fetch: Callable[[str, Settings], dict] | None = None) -> dict:
    """Overpass-style JSON for the tagged nodes on the drivable roads crossing `bbox` (west, south, east, north,
    WGS84), from the same source as `for_bbox`: traffic signals, crossings, stop and give-way signs and mapped signs
    (`index.is_road_node`), with the tags the index keeps of them (`index.road_node_tags`).

    From Overpass they come out of the roads query itself (its `out body` carries every node's tags), so the same
    query is asked, and answered from the cache when `for_bbox` has asked it, then narrowed to what the index keeps
    (`index.as_indexed_nodes`). An index built before they were kept raises `index.IndexUnavailable` (one of
    `INDEX_ERRORS`), and the caller goes on without them.
    """
    if settings.osm_index:
        return index.road_nodes(settings.osm_index, bbox)
    west, south, east, north = bbox
    return index.as_indexed_nodes((fetch or overpass.fetch)(overpass.query_for_bbox(south, west, north, east),
                                                            settings))


LINES_SELECTION = f"{overpass.RAILWAYS_FILTER} and {overpass.PATHS_FILTER}"
SERVICE_SELECTION = overpass.SERVICE_FILTER


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
    lines = (f" Railways and paths were selected with the filters {sel['overpass_lines_filter']}, narrowed to what "
             "the index builder keeps (src/mapcreator/providers/osm/index.py)." if sel.get("overpass_lines_filter")
             else "")
    service = (f" Service roads and tracks, whose mouths are kept clear, were selected with the filter "
               f"{sel['overpass_service_filter']}, narrowed the same way." if sel.get("overpass_service_filter")
               else "")
    return (f"The OpenStreetMap data was selected from an Overpass API server with the filter "
            f"{sel['overpass_filter']} around the route (src/mapcreator/providers/osm/overpass.py{at}, "
            f"build {build}).{lines}{service}")
