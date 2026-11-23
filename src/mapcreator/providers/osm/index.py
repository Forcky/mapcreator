"""A local OSM index for production: drivable roads, railways, paths, service roads and tracks, building footprints
and power lines in SQLite, and the tagged nodes on the roads (traffic signals, crossings, stop and give-way signs).

Public Overpass servers are for light development use only, so build workers
read OpenStreetMap from a Geofabrik extract instead. Reading a 900 MB
`.osm.pbf` for every job would take minutes, so the extract is indexed once:
one pass keeps the drivable highway ways (the classes `overpass.DRIVABLE`
asks for), railways and paths (`overpass.RAILWAYS` and `overpass.PATHS`, which
cross roads on bridges and under them), service roads and tracks
(`overpass.SERVICE`: driveways among them, whose mouths are kept clear; never
routed on), closed `building=*` ways and overhead
power lines (`power=line` and `power=minor_line`), with their node coordinates
resolved, in a SQLite file with an R*Tree per kind on the ways' lon/lat boxes.
The nodes of the drivable roads that carry a tag a build places something by
(`ROAD_NODE_HIGHWAY`, `crossing=*`, `traffic_sign=*`) are kept in a table of their
own with `ROAD_NODE_TAGS` of their tags; a road way's node ids say where each is.
A query then takes milliseconds and returns the same JSON shape as Overpass, so
`graph.build` works unchanged.

Reading `.osm.pbf` needs pyosmium (the `osm` extra). Without it, `.osm` XML
(also `.osm.gz` and `.osm.bz2`) is read with the standard library. That keeps
every node location in a Python dict, so it suits fixtures and small extracts,
not a national one. Only building the index needs pyosmium: querying needs
SQLite's R*Tree module, which CPython's own builds and the common Linux
packages include.

The index is written to a temporary file and moved into place when complete,
so a reader never sees half an index, and it is opened read-only and
immutable, so it can sit on a read-only mount in the worker's container.
Rebuild it (from a newer extract) rather than updating it in place.

    python -m mapcreator.providers.osm.index build australia-latest.osm.pbf data/osm/australia.sqlite
    python -m mapcreator.providers.osm.index query data/osm/australia.sqlite 147.30 -42.82 147.34 -42.79

The index is a derivative of OSM data (ODbL). It stays on the build machines;
what ships with a track is the route's `centreline.geojson` and attribution.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import os
import sqlite3
import struct
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import numpy as np
import shapely
from pyproj import Transformer

from mapcreator.providers.base import LONLAT_CRS
from mapcreator.providers.osm.overpass import DRIVABLE, PATHS, RAILWAYS, SERVICE

# Bump when the table layout changes; queries refuse an index built for another layout. An older layout still
# answers what it holds, so a worker keeps building tracks until its index is rebuilt: layout 1 has no power lines,
# layouts 1 and 2 have no railways or paths, and layouts 1 to 3 have no service roads or tracks.
# Layout 4 gained the tagged road nodes (`road_nodes`) before any layout 4 index was built for production, so a
# layout 4 file built before them (it has no `road_nodes` table) still reads, without them, as layout 3 reads without
# the service ways.
SCHEMA_VERSION = "4"
READABLE_SCHEMAS = ("1", "2", "3", "4")
POWER_LINES = frozenset({"line", "minor_line"})
RAILWAY_CLASSES = frozenset(RAILWAYS.split("|"))
PATH_CLASSES = frozenset(PATHS.split("|"))
SERVICE_CLASSES = frozenset(SERVICE.split("|"))
# What a service road or track keeps of its tags (layout 4): what it is, who may use it and how wide and what surface
# it is (for its mouth's width), its name, and whether it's level with the road it meets.
SERVICE_TAGS = ("highway", "service", "access", "motor_vehicle", "vehicle", "name", "surface", "width", "area",
                "bridge", "tunnel", "layer")
# What a railway or path keeps of its tags: what it is, its name, whether it's on a bridge, in a tunnel or on an
# embankment, and what decides how it looks. Its node ids are kept too: a path or level crossing that meets a road at
# grade shares a node with it, and a bridge doesn't.
LINE_TAGS = ("highway", "railway", "name", "ref", "bridge", "bridge:structure", "bridge:name", "layer", "tunnel",
             "covered", "indoor", "location", "embankment", "cutting", "width", "surface", "service", "usage",
             "gauge", "electrified", "footway", "area", "maxheight")
# The kinds whose ways keep their node ids.
_WITH_NODES = ("road", "railway", "path", "service")
# A power line's vertices that are mapped supports, stored per vertex in its tags as one letter each:
# T for power=tower (or portal), P for power=pole, - for anything else.
SUPPORTS_TAG = "mapcreator:supports"
_SUPPORT_CODE = {"tower": "T", "portal": "T", "pole": "P"}
ROAD_CLASSES = frozenset(DRIVABLE.split("|"))
# The nodes of drivable roads the index keeps, with their tags (layout 4): those tagged `highway` with one of these,
# or carrying any `crossing` or `traffic_sign` tag. Traffic signals and the crossings at them (and their
# `traffic_signals:direction`), stop and give-way signs, and mapped signs, for the signals and signs a track draws.
ROAD_NODE_HIGHWAY = frozenset({"traffic_signals", "stop", "give_way", "crossing"})
ROAD_NODE_KEYS = ("crossing", "traffic_sign")
# What a kept node keeps of its tags: these keys, and any key starting with one of the prefixes.
ROAD_NODE_TAGS = ("highway", "crossing", "crossing_ref", "traffic_signals", "traffic_sign", "button_operated",
                  "direction", "stop", "give_way")
ROAD_NODE_PREFIXES = ("crossing:", "traffic_signals:", "traffic_sign:")
# pyosmium's node location store. flex_mem keeps every node location in memory, a few GB for a
# national extract; `sparse_file_array,<file>` keeps them in a file instead, for small machines.
DEFAULT_LOCATIONS = "flex_mem"
XML_SUFFIXES = (".osm", ".osm.gz", ".osm.bz2", ".xml")
_BATCH = 20_000
_REPORT_EVERY = 500_000

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE ways (
    id INTEGER PRIMARY KEY,   -- OSM way id
    kind TEXT NOT NULL,       -- 'road', 'railway', 'path', 'service', 'building' or 'power'
    tags_json TEXT NOT NULL,
    coords BLOB NOT NULL,     -- little-endian float64, (n, 2) lon/lat in way order
    node_ids BLOB             -- little-endian int64, (n,) OSM node ids; roads, railways, paths and service ways only
);
-- One R*Tree per kind, so a road query in a city never wades through its buildings or footpaths.
CREATE VIRTUAL TABLE road_rtree USING rtree(id, minx, maxx, miny, maxy);
CREATE VIRTUAL TABLE building_rtree USING rtree(id, minx, maxx, miny, maxy);
CREATE VIRTUAL TABLE power_rtree USING rtree(id, minx, maxx, miny, maxy);
CREATE VIRTUAL TABLE railway_rtree USING rtree(id, minx, maxx, miny, maxy);
CREATE VIRTUAL TABLE path_rtree USING rtree(id, minx, maxx, miny, maxy);
CREATE VIRTUAL TABLE service_rtree USING rtree(id, minx, maxx, miny, maxy);
-- The tagged nodes on the roads (`is_road_node`), found through the road ways' node ids, so with no R*Tree of their own.
CREATE TABLE road_nodes (
    id INTEGER PRIMARY KEY,   -- OSM node id
    lon REAL NOT NULL,
    lat REAL NOT NULL,
    tags_json TEXT NOT NULL   -- `road_node_tags` of its tags
);
"""
_RTREE = {"road": "road_rtree", "building": "building_rtree", "power": "power_rtree", "railway": "railway_rtree",
          "path": "path_rtree", "service": "service_rtree"}
_COUNT = {"road": "roads", "building": "buildings", "power": "power_lines", "railway": "railways", "path": "paths",
          "service": "service_ways"}


class IndexUnavailable(RuntimeError):
    """The index file is missing, isn't an index, or was built for another table layout."""


@dataclass(slots=True)
class _Way:
    """A way of a kind the index keeps, as either reader hands it over, with nodes of unknown location dropped."""

    id: int
    kind: str  # "road", "railway", "path", "service", "building" or "power"
    tags: dict[str, str]
    coords: np.ndarray  # (n, 2) float64 lon/lat in way order
    refs: list[int] | None  # OSM node ids in line with `coords`, for `_WITH_NODES` kinds; None for the rest
    complete: bool  # every node's location was known


def is_road_node(tags) -> bool:
    """Whether a node with these tags is one the index keeps, if it's on a drivable road."""
    return tags.get("highway") in ROAD_NODE_HIGHWAY or any(k in tags for k in ROAD_NODE_KEYS)


def road_node_tags(tags) -> dict[str, str]:
    """What the index keeps of a road node's tags (`ROAD_NODE_TAGS`, and keys starting with `ROAD_NODE_PREFIXES`)."""
    return {k: v for k, v in tags.items() if k in ROAD_NODE_TAGS or k.startswith(ROAD_NODE_PREFIXES)}


def _kind(highway: str | None, building: str | None, closed: bool, power: str | None = None,
          railway: str | None = None) -> str | None:
    """Which kind of way this is, from the tags that decide it, or None to skip it unread.

    A way tagged as two kinds is the first of road, building, power line, path, railway and service way. So the kinds
    added later (paths and railways, layout 3; service roads and tracks, layout 4) take no way from the earlier ones:
    a former station drawn as a closed way with `building` and `railway=disused` stays a building. A rail trail (a
    cycleway that also carries `railway=abandoned`) is a path, and keeps both tags."""
    if highway in ROAD_CLASSES:
        return "road"
    if building not in (None, "no") and closed:
        return "building"
    if power in POWER_LINES:
        return "power"
    if highway in PATH_CLASSES:
        return "path"
    if railway in RAILWAY_CLASSES:
        return "railway"
    if highway in SERVICE_CLASSES:
        return "service"
    return None


# ----- building -----

def build_index(source: str | os.PathLike, out: str | os.PathLike, *, reader: str = "auto",
                locations: str = DEFAULT_LOCATIONS,
                progress: Callable[[dict[str, int]], None] | None = None) -> dict[str, int]:
    """Index the drivable roads, railways, paths, service roads and tracks, building footprints and power lines of an
    OSM file into a new SQLite file.

    `reader` is "osmium" (pyosmium: .pbf or XML), "xml" (standard library: XML only) or "auto"
    (pyosmium when installed, else XML). `locations` picks pyosmium's node location store.
    `progress`, if given, is called with running counts every few hundred thousand ways.
    Returns the counts: roads, buildings, power lines, railways, paths and service ways stored, other tagged ways
    skipped, and the tagged nodes kept on the roads (`road_nodes`).
    """
    src, dst = Path(source), Path(out)
    if not src.is_file():
        raise FileNotFoundError(f"no OSM file at {src}")
    ways, header = _open_reader(src, reader, locations)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    tmp.unlink(missing_ok=True)
    counts = {"roads": 0, "buildings": 0, "power_lines": 0, "railways": 0, "paths": 0, "service_ways": 0,
              "skipped": 0}
    stored = [name for name in counts if name != "skipped"]
    tagged = header.pop("road_nodes")  # node id -> (lon, lat, tags), every node `is_road_node` keeps on a road
    on_roads: set[int] = set()
    next_report = _REPORT_EVERY
    try:
        with closing(sqlite3.connect(tmp)) as con:
            # A crash leaves only the .partial file, which the next build deletes, so no journal. A
            # 256 MB page cache holds a national extract's R*Tree nodes while they're being filled.
            con.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF; PRAGMA cache_size = -262144;"
                              + _SCHEMA)
            batch: dict[str, list] = {kind: [] for kind in _RTREE}
            for way in ways:
                row = _row(way) if way is not None else None
                if row is None:
                    counts["skipped"] += 1
                    continue
                kind = row[1]
                if kind == "road":
                    on_roads.update(tagged.keys() & way.refs)
                batch[kind].append(row)
                counts[_COUNT[kind]] += 1
                if len(batch[kind]) >= _BATCH:
                    _insert(con, kind, batch[kind])
                    batch[kind] = []
                if progress and sum(counts[name] for name in stored) >= next_report:
                    progress(dict(counts))
                    next_report += _REPORT_EVERY
            for kind, rows in batch.items():
                _insert(con, kind, rows)
            counts["road_nodes"] = len(on_roads)
            con.executemany("INSERT INTO road_nodes (id, lon, lat, tags_json) VALUES (?, ?, ?, ?)",
                            ((i, tagged[i][0], tagged[i][1],
                              json.dumps(road_node_tags(tagged[i][2]), ensure_ascii=False, separators=(",", ":")))
                             for i in sorted(on_roads)))
            meta = {
                "schema": SCHEMA_VERSION,
                "source": src.name,  # the file name only: build machines' paths stay out of the index
                "osm_timestamp": header.get("osm_timestamp", ""),
                "built": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "reader": header["reader"],
                "road_classes": DRIVABLE,
                "railway_classes": RAILWAYS,
                "path_classes": PATHS,
                "service_classes": SERVICE,
                "road_node_highway": "|".join(sorted(ROAD_NODE_HIGHWAY)),
                "road_node_keys": "|".join(ROAD_NODE_KEYS),
                **{k: str(v) for k, v in counts.items()},
            }
            con.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta.items())
            con.commit()
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return counts


def _row(way: _Way) -> tuple | None:
    """The `ways` row for a way of a kept kind, plus its coordinates, or None if too little is known."""
    if way.kind in _WITH_NODES:
        if len(way.coords) < 2:
            return None  # at the edge of an extract, a way can lose all but one node
        tags, node_ids = _kept_tags(way.kind, way.tags), np.asarray(way.refs, dtype="<i8").tobytes()
    elif way.kind == "power":
        if len(way.coords) < 2:
            return None
        tags, node_ids = _kept_tags(way.kind, way.tags), None
    else:
        # A footprint with a node missing would be a different shape, so drop it whole.
        if not way.complete or len(way.coords) < 4:
            return None
        tags, node_ids = _kept_tags(way.kind, way.tags), None
    coords = way.coords.astype("<f8", copy=False)
    return (way.id, way.kind, json.dumps(tags, ensure_ascii=False, separators=(",", ":")), coords.tobytes(),
            node_ids, coords)


def _kept_tags(kind: str, tags: dict[str, str]) -> dict[str, str]:
    """The tags the index keeps for a way of this kind: all of a road's, `LINE_TAGS` of a railway's or path's and
    `SERVICE_TAGS` of a service way's."""
    if kind in ("railway", "path"):
        return {k: v for k, v in tags.items() if k in LINE_TAGS}
    if kind == "service":
        return {k: v for k, v in tags.items() if k in SERVICE_TAGS}
    if kind == "building":
        return _building_tags(tags)
    if kind == "power":
        return _power_tags(tags)
    return tags


def _building_tags(tags: dict[str, str]) -> dict[str, str]:
    """Only a building's shape and height: addresses and names would bloat a national index for nothing."""
    return {k: v for k, v in tags.items()
            if k == "building" or k.startswith(("building:", "roof:")) or k in ("height", "min_height")}


def _power_tags(tags: dict[str, str]) -> dict[str, str]:
    """What a power line's model needs: its kind, voltage, circuits and whether it's overhead."""
    return {k: v for k, v in tags.items()
            if k in ("power", "voltage", "circuits", "cables", "wires", "location", "line", "frequency", SUPPORTS_TAG)}


def _insert(con: sqlite3.Connection, kind: str, rows: list[tuple]) -> None:
    if not rows:
        return
    con.executemany("INSERT OR REPLACE INTO ways (id, kind, tags_json, coords, node_ids) VALUES (?, ?, ?, ?, ?)",
                    [r[:5] for r in rows])
    # Each way's box, for the whole batch at once: numpy's per-call overhead outweighs a 5-node way.
    coords = [r[5] for r in rows]
    starts = np.cumsum([0] + [len(c) for c in coords[:-1]])
    flat = np.concatenate(coords)
    lo, hi = np.minimum.reduceat(flat, starts, axis=0), np.maximum.reduceat(flat, starts, axis=0)
    con.executemany(f"INSERT OR REPLACE INTO {_RTREE[kind]} (id, minx, maxx, miny, maxy) VALUES (?, ?, ?, ?, ?)",
                    zip([r[0] for r in rows], lo[:, 0].tolist(), hi[:, 0].tolist(), lo[:, 1].tolist(),
                        hi[:, 1].tolist()))


def _open_reader(src: Path, reader: str, locations: str) -> tuple[Iterator[_Way | None], dict[str, str]]:
    if reader not in ("auto", "osmium", "xml"):
        raise ValueError(f"reader must be auto, osmium or xml, not {reader!r}")
    if reader == "auto":
        try:
            import osmium  # noqa: F401
            reader = "osmium"
        except ModuleNotFoundError as e:
            if e.name != "osmium":
                raise  # osmium is there, but something it needs isn't
            reader = "xml"
        except ImportError as e:
            # Installed but unloadable (e.g. a missing system library): say so, don't fall back to XML.
            raise RuntimeError(f"pyosmium is installed but failed to load: {e}") from e
    if reader == "xml":
        if not src.name.lower().endswith(XML_SUFFIXES):
            raise RuntimeError(f"{src.name}: reading anything but .osm XML needs pyosmium "
                               "(pip install -e .[osm])")
        tagged: dict[int, tuple[float, float, dict[str, str]]] = {}
        return _read_xml(src, tagged), {"reader": "xml", "road_nodes": tagged}
    return _read_osmium(src, locations)


def _read_osmium(src: Path, locations: str) -> tuple[Iterator[_Way | None], dict[str, str]]:
    """Ways through pyosmium. Yields None for a highway, railway, building or power way that isn't kept."""
    try:
        import osmium
        import osmium.geom
        import osmium.io
        import osmium.version
    except ModuleNotFoundError as e:
        if e.name != "osmium":
            raise
        raise RuntimeError("pyosmium isn't installed: pip install -e .[osm]") from None
    except ImportError as e:
        raise RuntimeError(f"pyosmium is installed but failed to load: {e}") from e

    # Geofabrik records the time of the data in the header; a closed reader leaves no threads behind.
    with osmium.io.Reader(str(src), osmium.osm.NOTHING) as r:
        timestamp = r.header().get("osmosis_replication_timestamp", "")

    # Every node goes through the location store before the filters, so ways get their
    # coordinates; only ways tagged highway, railway, building or power ever reach Python.
    fp = (osmium.FileProcessor(str(src))
          .with_locations(locations)
          .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
          .with_filter(osmium.filter.KeyFilter("highway", "railway", "building", "power")))
    # The nodes a road may carry a signal, crossing or sign on (`is_road_node`): a pass over the nodes with those keys,
    # before the ways, which keep those on a drivable road. Their locations are on libosmium's 1e-7 degree grid.
    tagged = {n.id: (n.location.lon, n.location.lat, {t.k: t.v for t in n.tags})
              for n in osmium.FileProcessor(str(src), osmium.osm.NODE)
              .with_filter(osmium.filter.KeyFilter("highway", *ROAD_NODE_KEYS))
              if n.location.valid() and is_road_node(n.tags)}
    header = {"reader": f"pyosmium {osmium.version.pyosmium_release}", "osm_timestamp": timestamp,
              "road_nodes": tagged}
    # Which nodes are towers and poles, for the power lines' supports: a pass over tagged nodes only (about 650,000
    # for Australia), before the ways.
    supports = {n.id: _SUPPORT_CODE[n.tags.get("power")]
                for n in osmium.FileProcessor(str(src), osmium.osm.NODE)
                .with_filter(osmium.filter.TagFilter(*(("power", v) for v in _SUPPORT_CODE)))}
    wkb = osmium.geom.WKBFactory()
    every_node = osmium.geom.use_nodes.ALL  # keep repeated locations, so coordinates line up with node ids

    def ways() -> Iterator[_Way | None]:
        # Per-node access from Python costs microseconds a node, and a national extract has tens
        # of millions of them, so coordinates come from one WKB call per way. Only roads, railways
        # and paths keep node ids (power lines read them for their supports). A way with a node of
        # unknown location (the edge of an extract) can't be made into WKB, so it takes the slow
        # path, which drops that node.
        for w in fp:
            tags = w.tags
            building = tags.get("building")
            kind = _kind(tags.get("highway"), building, w.is_closed(), tags.get("power"), tags.get("railway"))
            if kind is None:
                yield None
                continue
            try:
                coords = _wkb_coords(wkb.create_linestring(w, every_node))
                refs = [n.ref for n in w.nodes] if kind in (*_WITH_NODES, "power") else None
                complete = True
            except Exception:  # pyosmium raises InvalidLocationError, or a geometry error for one node
                refs, points = [], []
                for n in w.nodes:
                    loc = n.location
                    if loc.valid():
                        refs.append(n.ref)
                        points.append((loc.lon, loc.lat))
                complete = len(refs) == len(w.nodes)
                coords = np.array(points, dtype="<f8").reshape(-1, 2)
                refs = refs if kind in (*_WITH_NODES, "power") else None
            # Reading tags costs microseconds each, and most footprints are tagged building=yes alone.
            tag_dict = {"building": building} if kind == "building" and len(tags) == 1 else dict(tags)
            if kind == "power":
                tag_dict[SUPPORTS_TAG] = "".join(supports.get(r, "-") for r in refs)
                refs = None
            yield _Way(w.id, kind, tag_dict, coords, refs, complete)

    return ways(), header


def _wkb_coords(wkb_hex: str) -> np.ndarray:
    """(n, 2) coordinates of a WKB LineString, as pyosmium's WKBFactory writes it (hex, no SRID)."""
    b = bytes.fromhex(wkb_hex)
    order = "<" if b[0] == 1 else ">"
    geom_type, n = struct.unpack(order + "II", b[1:9])
    if geom_type != 2:
        raise ValueError(f"expected a WKB LineString, not geometry type {geom_type}")
    return np.frombuffer(b, dtype=order + "f8", count=2 * n, offset=9).reshape(n, 2)


def _open_xml(src: Path) -> BinaryIO:
    name = src.name.lower()
    if name.endswith(".gz"):
        return gzip.open(src, "rb")
    if name.endswith(".bz2"):
        return bz2.open(src, "rb")
    return src.open("rb")


def _fixed(value: str) -> float:
    """A coordinate on OSM's 1e-7 degree grid, as libosmium stores it, so both readers agree exactly."""
    return round(float(value) * 10_000_000) / 10_000_000


def _read_xml(src: Path, tagged: dict | None = None) -> Iterator[_Way | None]:
    """Ways from an OSM XML file, in one pass (nodes come before ways in a well-formed file).

    Yields None for a highway, railway, building or power way that isn't kept, like the pyosmium reader. `tagged`, if
    given, is filled with the nodes `is_road_node` keeps (id -> lon, lat, tags) as they're read, before the ways.
    """
    nodes: dict[int, tuple[float, float]] = {}
    supports: dict[int, str] = {}
    with _open_xml(src) as f:
        events = ET.iterparse(f, events=("start", "end"))
        _, root = next(events)
        for event, el in events:
            if event != "end" or el.tag not in ("node", "way", "relation"):
                continue
            if el.tag == "node":
                lon, lat = el.get("lon"), el.get("lat")
                if lon is not None and lat is not None:
                    nodes[int(el.get("id"))] = (_fixed(lon), _fixed(lat))
                node_tags = {t.get("k"): t.get("v") for t in el.iter("tag")}
                if node_tags.get("power") in _SUPPORT_CODE:
                    supports[int(el.get("id"))] = _SUPPORT_CODE[node_tags["power"]]
                if tagged is not None and lon is not None and lat is not None and is_road_node(node_tags):
                    tagged[int(el.get("id"))] = (*nodes[int(el.get("id"))], node_tags)
            elif el.tag == "way":
                tags = {t.get("k"): t.get("v") for t in el.iter("tag")}
                if "highway" in tags or "railway" in tags or "building" in tags or "power" in tags:
                    all_refs = [int(nd.get("ref")) for nd in el.iter("nd")]
                    closed = len(all_refs) > 1 and all_refs[0] == all_refs[-1]
                    kind = _kind(tags.get("highway"), tags.get("building"), closed, tags.get("power"),
                                 tags.get("railway"))
                    if kind is None:
                        yield None
                    else:
                        refs = [r for r in all_refs if r in nodes]
                        coords = np.array([nodes[r] for r in refs], dtype="<f8").reshape(-1, 2)
                        if kind == "power":
                            tags[SUPPORTS_TAG] = "".join(supports.get(r, "-") for r in refs)
                        yield _Way(int(el.get("id")), kind, tags, coords, refs if kind in _WITH_NODES else None,
                                   len(refs) == len(all_refs))
            root.clear()  # keep memory flat: each top-level element is finished with


# ----- querying -----

def _connect(index_path: str | os.PathLike) -> sqlite3.Connection:
    path = Path(index_path)
    if not path.is_file():
        raise IndexUnavailable(f"no OSM index at {path}; build one with "
                               "`python -m mapcreator.providers.osm.index build <extract> <index>`")
    # Read-only and immutable: no locks, no journal, works on a read-only mount.
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        row = con.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone()
    except sqlite3.DatabaseError as e:
        con.close()
        raise IndexUnavailable(f"{path.name} is not an OSM index ({e})") from e
    if row is None or row[0] not in READABLE_SCHEMAS:
        con.close()
        raise IndexUnavailable(f"{path.name} was built for another index layout; rebuild it")
    return con


def _check_bbox(bbox) -> tuple[float, float, float, float]:
    west, south, east, north = (float(v) for v in bbox)
    if not (np.isfinite([west, south, east, north]).all() and west < east and south < north):
        raise ValueError(f"bounding box must be west < east and south < north, not {bbox!r}")
    return west, south, east, north


def _candidates(con: sqlite3.Connection, kind: str, bbox: tuple[float, float, float, float]) -> list[tuple]:
    """Rows whose box overlaps `bbox` (west, south, east, north), in way id order for repeatable builds."""
    west, south, east, north = bbox
    return con.execute(
        f"SELECT w.id, w.tags_json, w.coords, w.node_ids FROM {_RTREE[kind]} r JOIN ways w ON w.id = r.id "
        "WHERE r.minx <= ? AND r.maxx >= ? AND r.miny <= ? AND r.maxy >= ? ORDER BY w.id",
        (east, west, north, south)).fetchall()


def _schema(con: sqlite3.Connection) -> str:
    return con.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone()[0]


def query(index_path: str | os.PathLike, bbox: tuple[float, float, float, float]) -> dict:
    """The drivable roads crossing `bbox` as Overpass JSON: `{"elements": [nodes..., ways...]}`.

    `bbox` is west, south, east, north in WGS84 degrees (the order of `Region.bbox` and shapely's
    `bounds`). Like Overpass's `way(bbox); (._;>;);`, a way is included whole, with every node,
    if any part of it crosses the box. Nodes carry no tags; the graph doesn't use them.
    """
    area = _check_bbox(bbox)
    with closing(_connect(index_path)) as con:
        rows = _candidates(con, "road", area)
    return _overpass_json(rows, area)


def railways_and_paths(index_path: str | os.PathLike, bbox: tuple[float, float, float, float]) -> dict:
    """The railways and paths crossing `bbox` as Overpass JSON, like `query` gives the roads: each way whole, with
    its node ids (a path or railway meeting a road at grade shares a node with it; a bridge doesn't) and `LINE_TAGS`
    of its tags. Never routed on: `graph.build` must not be given them with the roads.

    An index built before railways and paths were kept (layouts 1 and 2) raises `IndexUnavailable`, and the caller
    goes on without them until the index is rebuilt.
    """
    area = _check_bbox(bbox)
    with closing(_connect(index_path)) as con:
        if _schema(con) in ("1", "2"):
            raise IndexUnavailable(f"{Path(index_path).name} was built before railways and paths were indexed; "
                                   "rebuild it")
        rows = sorted(_candidates(con, "railway", area) + _candidates(con, "path", area), key=lambda r: r[0])
    return _overpass_json(rows, area)


def service_ways(index_path: str | os.PathLike, bbox: tuple[float, float, float, float]) -> dict:
    """The service roads and tracks crossing `bbox` (`overpass.SERVICE`: driveways among them) as Overpass JSON, like
    `query` gives the roads: each way whole, with its node ids (a driveway meets the road it leaves at a shared node)
    and `SERVICE_TAGS` of its tags. Only to keep their mouths clear: never routed on, never given a stub, never a
    crossing. `graph.build` must not be given them with the roads the route is found on.

    An index built before they were kept (layouts 1 to 3) raises `IndexUnavailable`, and the caller goes on without
    them until the index is rebuilt.
    """
    area = _check_bbox(bbox)
    with closing(_connect(index_path)) as con:
        if _schema(con) in ("1", "2", "3"):
            raise IndexUnavailable(f"{Path(index_path).name} was built before driveways, service roads and tracks "
                                   "were indexed; rebuild it")
        rows = _candidates(con, "service", area)
    return _overpass_json(rows, area)


def has_road_nodes(con: sqlite3.Connection) -> bool:
    """Whether an open index holds the tagged road nodes: layout 4, built since they were kept."""
    return _schema(con) not in ("1", "2", "3") and con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'road_nodes'").fetchone() is not None


_IN_BATCH = 900  # node ids per query, under SQLite's older limit of 999 parameters


def road_nodes(index_path: str | os.PathLike, bbox: tuple[float, float, float, float]) -> dict:
    """The tagged nodes on the drivable roads crossing `bbox` (`is_road_node`: traffic signals, crossings, stop and
    give-way signs, mapped signs) as Overpass JSON, `{"elements": [nodes...]}` in node id order, each with
    `road_node_tags` of its tags. The roads are those `query` gives, each whole, so a node may lie outside the box, as
    Overpass's `(._;>;)` gives it. Which way a node is on, and where, is in those ways' node ids.

    An index without them, built before layout 4 or a layout 4 one built before they were kept (it has no
    `road_nodes` table), raises `IndexUnavailable`, and the caller goes on without them until the index is rebuilt.
    """
    area = _check_bbox(bbox)
    with closing(_connect(index_path)) as con:
        if not has_road_nodes(con):
            raise IndexUnavailable(f"{Path(index_path).name} was built before the tagged road nodes (traffic signals, "
                                   "crossings, stop and give-way signs) were indexed; rebuild it")
        roads = _overpass_json(_candidates(con, "road", area), area)
        ids = sorted({n for e in roads["elements"] if e["type"] == "way" for n in e["nodes"]})
        found = []
        for k in range(0, len(ids), _IN_BATCH):
            part = ids[k:k + _IN_BATCH]
            found += con.execute(f"SELECT id, lon, lat, tags_json FROM road_nodes WHERE id IN "
                                 f"({','.join('?' * len(part))})", part).fetchall()
    return {"version": 0.6, "generator": "mapcreator osm index",
            "elements": [{"type": "node", "id": i, "lat": lat, "lon": lon, "tags": json.loads(t)}
                         for i, lon, lat, t in sorted(found)]}


def as_indexed_nodes(data: dict) -> dict:
    """An Overpass answer for the roads narrowed to the tagged road nodes the index keeps (`road_nodes`): the nodes of
    its road ways that `is_road_node` keeps, in node id order, with `road_node_tags` of their tags. So the development
    fallback, whose roads answer carries every node's tags, gives what the index gives."""
    elements = data.get("elements", [])
    on_roads = set()
    for e in elements:
        if e.get("type") == "way":
            tags, refs = e.get("tags") or {}, e.get("nodes") or []
            closed = len(refs) > 1 and refs[0] == refs[-1]
            if _kind(tags.get("highway"), tags.get("building"), closed, tags.get("power"),
                     tags.get("railway")) == "road":
                on_roads.update(refs)
    nodes = [{"type": "node", "id": e["id"], "lat": e["lat"], "lon": e["lon"], "tags": road_node_tags(e["tags"])}
             for e in elements if e.get("type") == "node" and e["id"] in on_roads and "lon" in e
             and is_road_node(e.get("tags") or {})]
    out = {"version": data.get("version", 0.6), "generator": data.get("generator", ""),
           "elements": sorted(nodes, key=lambda n: n["id"])}
    if data.get("remark"):
        out["remark"] = data["remark"]
    return out


def as_indexed(data: dict, kinds: tuple[str, ...]) -> dict:
    """An Overpass answer narrowed to what the index keeps of the given kinds ("road", "railway", "path", "service"),
    so the
    development fallback answers as the index does: those kinds' ways in way id order, with the tags the index keeps
    and the nodes they use (untagged). Overpass selects by one tag at a time, so a way the index files under another
    kind is left out: a residential street that also carries `railway=abandoned` is a road, not a railway."""
    elements = data.get("elements", [])
    located = {e["id"]: (e["lon"], e["lat"]) for e in elements if e.get("type") == "node" and "lon" in e}
    used: dict[int, tuple[float, float]] = {}
    ways = []
    for e in sorted((e for e in elements if e.get("type") == "way"), key=lambda e: e["id"]):
        tags, refs = e.get("tags") or {}, e.get("nodes") or []
        closed = len(refs) > 1 and refs[0] == refs[-1]
        kind = _kind(tags.get("highway"), tags.get("building"), closed, tags.get("power"), tags.get("railway"))
        if kind not in kinds or kind not in _WITH_NODES:
            continue
        refs = [r for r in refs if r in located]
        if len(refs) < 2:
            continue
        used.update((r, located[r]) for r in refs)
        ways.append({"type": "way", "id": e["id"], "nodes": refs, "tags": _kept_tags(kind, tags)})
    nodes = [{"type": "node", "id": i, "lat": lat, "lon": lon} for i, (lon, lat) in sorted(used.items())]
    out = {"version": data.get("version", 0.6), "generator": data.get("generator", ""), "elements": nodes + ways}
    if data.get("remark"):
        out["remark"] = data["remark"]
    return out


def _overpass_json(rows: list[tuple], area: tuple[float, float, float, float]) -> dict:
    """Overpass JSON for the candidate rows (id, tags_json, coords, node_ids) that cross `area`."""
    if not rows:
        return {"version": 0.6, "generator": "mapcreator osm index", "elements": []}
    coords = [np.frombuffer(r[2], dtype="<f8").reshape(-1, 2) for r in rows]
    # The R*Tree matches on boxes; a long curving road can box the area without entering it.
    which = np.repeat(np.arange(len(rows)), [len(c) for c in coords])
    crosses = shapely.intersects(shapely.linestrings(np.concatenate(coords), indices=which), shapely.box(*area))

    nodes: dict[int, tuple[float, float]] = {}
    ways = []
    for (way_id, tags_json, _, ids_blob), xy, keep in zip(rows, coords, crosses):
        if not keep:
            continue
        ids = np.frombuffer(ids_blob, dtype="<i8").tolist()
        nodes.update(zip(ids, xy.tolist()))
        ways.append({"type": "way", "id": way_id, "nodes": ids, "tags": json.loads(tags_json)})
    elements = [{"type": "node", "id": i, "lat": lat, "lon": lon} for i, (lon, lat) in sorted(nodes.items())]
    return {"version": 0.6, "generator": "mapcreator osm index", "elements": elements + ways}


def buildings(index_path: str | os.PathLike, bbox: tuple[float, float, float, float],
              epsg: int) -> list[shapely.Polygon]:
    """Building footprints intersecting `bbox`, as polygons in EPSG:`epsg`.

    `bbox` is minx, miny, maxx, maxy in the same metric CRS, the contract theLIST's building
    provider has, because the scenery stage works in the region's CRS. Self-touching rings are
    repaired with a zero buffer and split into their parts. Buildings mapped as multipolygon
    relations (courtyards, complexes) aren't in the index.
    """
    minx, miny, maxx, maxy = _check_bbox(bbox)
    lonlat = Transformer.from_crs(epsg, LONLAT_CRS, always_xy=True).transform_bounds(minx, miny, maxx, maxy,
                                                                                 densify_pts=21)
    with closing(_connect(index_path)) as con:
        rows = _candidates(con, "building", _check_bbox(lonlat))
    if not rows:
        return []
    rings = [np.frombuffer(r[2], dtype="<f8").reshape(-1, 2) for r in rows]
    x, y = Transformer.from_crs(LONLAT_CRS, epsg, always_xy=True).transform(*np.concatenate(rings).T)
    xy = np.column_stack([x, y])
    area = shapely.box(minx, miny, maxx, maxy)
    out: list[shapely.Polygon] = []
    for ring in np.split(xy, np.cumsum([len(r) for r in rings])[:-1]):
        poly = shapely.Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        for part in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly]):
            if not part.is_empty and part.intersects(area):
                out.append(part)
    return out


def power_lines(index_path: str | os.PathLike, bbox: tuple[float, float, float, float], epsg: int) -> list:
    """Overhead power lines crossing `bbox` (minx, miny, maxx, maxy in EPSG:`epsg`, like `buildings`), as
    `core.power.Line`s in that CRS. Underground and underwater lines and busbars are left out. Every vertex is a
    support. Vertices mapped as power=tower or power=pole carry that design; where a mapper drew a line
    sparsely, `power.densify` adds evenly spaced supports between vertices that aren't both mapped supports."""
    from mapcreator.core import power

    minx, miny, maxx, maxy = _check_bbox(bbox)
    lonlat = Transformer.from_crs(epsg, LONLAT_CRS, always_xy=True).transform_bounds(minx, miny, maxx, maxy,
                                                                                 densify_pts=21)
    with closing(_connect(index_path)) as con:
        if _schema(con) == "1":
            raise IndexUnavailable(f"{Path(index_path).name} was built before power lines were indexed; rebuild it")
        rows = _candidates(con, "power", _check_bbox(lonlat))
    to_xy = Transformer.from_crs(LONLAT_CRS, epsg, always_xy=True)
    out = []
    for way_id, tags_json, coords_blob, _ in rows:
        tags = json.loads(tags_json)
        if tags.get("location") in ("underground", "underwater", "indoor") or tags.get("line") in ("busbar", "bay"):
            continue
        lon, lat = np.frombuffer(coords_blob, dtype="<f8").reshape(-1, 2).T
        x, y = to_xy.transform(lon, lat)
        kind = power.TRANSMISSION if tags.get("power") == "line" else power.DISTRIBUTION
        try:
            circuits = int(tags.get("circuits") or 0) or max(int(tags.get("cables") or 3) // 3, 1)
        except ValueError:
            circuits = 1
        codes = tags.get(SUPPORTS_TAG, "")
        designs = ([{"T": "tower", "P": "pole"}.get(ch, "") for ch in codes] if len(codes) == len(x)
                   else [""] * len(x))
        out.append(power.densify(power.Line(np.column_stack([x, y]), kind, power.parse_kv(tags.get("voltage")),
                                            circuits, f"OSM way {way_id}", designs=designs)))
    return out


def metadata(index_path: str | os.PathLike) -> dict[str, str]:
    """What the index was built from and when: source file name, OSM data timestamp, counts."""
    with closing(_connect(index_path)) as con:
        return dict(con.execute("SELECT key, value FROM meta").fetchall())


# ----- command line -----

def _mga_epsg(lon: float) -> int:
    """GDA94 / MGA zone for a longitude, for counting buildings in a lon/lat box."""
    return 28300 + int((lon + 180) // 6) + 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mapcreator.providers.osm.index",
                                 description="Build or inspect the local OSM index of roads, railways, paths, "
                                             "service roads and tracks, buildings and power lines.")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="index an OSM extract (.osm.pbf needs pyosmium; .osm XML works without)")
    b.add_argument("extract", type=Path)
    b.add_argument("out", type=Path)
    b.add_argument("--reader", choices=("auto", "osmium", "xml"), default="auto")
    b.add_argument("--locations", default=DEFAULT_LOCATIONS,
                   help="pyosmium node location store: flex_mem (default, in memory) or "
                        "sparse_file_array,<file> to keep node locations on disk")
    q = sub.add_parser("query", help="count the roads, railways, paths, service ways and buildings in a box, "
                                     "for debugging")
    q.add_argument("index", type=Path)
    for name in ("west", "south", "east", "north"):
        q.add_argument(name, type=float)
    q.add_argument("--epsg", type=int, help="metric CRS for the buildings query (default: the box's MGA zone)")
    q.add_argument("--json", action="store_true", help="print the roads as Overpass JSON")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    if args.command == "build":
        def report(c: dict[str, int]) -> None:
            print(f"  {c['roads']:,} roads, {c['railways']:,} railways, {c['paths']:,} paths, "
                  f"{c['service_ways']:,} service ways, {c['buildings']:,} buildings, {c['power_lines']:,} power lines "
                  f"({time.perf_counter() - t0:.0f} s)", file=sys.stderr, flush=True)

        counts = build_index(args.extract, args.out, reader=args.reader, locations=args.locations, progress=report)
        size_mb = args.out.stat().st_size / 1e6
        print(f"indexed {counts['roads']:,} roads, {counts['railways']:,} railways, {counts['paths']:,} paths, "
              f"{counts['service_ways']:,} service roads and tracks, "
              f"{counts['buildings']:,} buildings and {counts['power_lines']:,} power lines, "
              f"and {counts['road_nodes']:,} tagged nodes on the roads "
              f"({counts['skipped']:,} other ways skipped) from {args.extract.name} "
              f"in {time.perf_counter() - t0:.1f} s: {args.out.name}, {size_mb:,.1f} MB")
        return 0

    bbox = (args.west, args.south, args.east, args.north)
    meta = metadata(args.index)
    roads = query(args.index, bbox)
    t_roads = time.perf_counter() - t0
    t1 = time.perf_counter()
    try:
        lines = railways_and_paths(args.index, bbox)
    except IndexUnavailable:
        lines = None  # an index from before railways and paths were kept
    t_lines = time.perf_counter() - t1
    t1 = time.perf_counter()
    try:
        service = service_ways(args.index, bbox)
    except IndexUnavailable:
        service = None  # an index from before service roads and tracks were kept
    t_service = time.perf_counter() - t1
    t1 = time.perf_counter()
    try:
        tagged = road_nodes(args.index, bbox)
    except IndexUnavailable:
        tagged = None  # an index from before the tagged road nodes were kept
    t_nodes = time.perf_counter() - t1
    epsg = args.epsg or _mga_epsg((args.west + args.east) / 2)
    projected = Transformer.from_crs(LONLAT_CRS, epsg, always_xy=True).transform_bounds(*bbox, densify_pts=21)
    t1 = time.perf_counter()
    footprints = buildings(args.index, projected, epsg)
    t_buildings = time.perf_counter() - t1
    if args.json:
        print(json.dumps(roads))
        return 0
    n_nodes = sum(e["type"] == "node" for e in roads["elements"])
    print(f"index: {meta.get('source', '?')}, OSM data as of {meta.get('osm_timestamp') or 'unknown'}, "
          f"built {meta.get('built', '?')}")
    print(f"roads: {len(roads['elements']) - n_nodes:,} ways, {n_nodes:,} nodes ({t_roads:.3f} s)")
    if lines is None:
        print("railways and paths: not in this index (built before layout 3); rebuild it")
    else:
        ways = [e for e in lines["elements"] if e["type"] == "way"]
        n_rail = sum(w["tags"].get("highway") not in PATH_CLASSES for w in ways)
        print(f"railways and paths: {n_rail:,} railways, {len(ways) - n_rail:,} paths ({t_lines:.3f} s)")
    if service is None:
        print("service roads and tracks: not in this index (built before layout 4); rebuild it")
    else:
        ways = [e for e in service["elements"] if e["type"] == "way"]
        n_drive = sum(w["tags"].get("service") == "driveway" for w in ways)
        n_track = sum(w["tags"].get("highway") == "track" for w in ways)
        print(f"service roads and tracks: {n_drive:,} driveways, {len(ways) - n_drive - n_track:,} other service "
              f"roads, {n_track:,} tracks ({t_service:.3f} s)")
    if tagged is None:
        print("tagged road nodes: not in this index (built before they were kept, in layout 4); rebuild it")
    else:
        found = [e["tags"] for e in tagged["elements"]]
        n_signals = sum(t.get("highway") == "traffic_signals" for t in found)
        n_crossings = sum(t.get("highway") == "crossing" or "crossing" in t for t in found)
        n_signs = sum(t.get("highway") in ("stop", "give_way") for t in found)
        print(f"tagged road nodes: {n_signals:,} traffic signals, {n_crossings:,} crossings, {n_signs:,} stop and "
              f"give-way signs, {len(found):,} in all ({t_nodes:.3f} s)")
    print(f"buildings: {len(footprints):,} in EPSG:{epsg} ({t_buildings:.3f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
