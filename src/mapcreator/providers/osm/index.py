"""A local OSM index for production: drivable roads and building footprints in SQLite.

Public Overpass servers are for light development use only, so build workers
read OpenStreetMap from a Geofabrik extract instead. Reading a 900 MB
`.osm.pbf` for every job would take minutes, so the extract is indexed once:
one pass keeps the drivable highway ways (the classes `overpass.DRIVABLE`
asks for) and closed `building=*` ways, with their node coordinates resolved,
in a SQLite file with an R*Tree per kind on the ways' lon/lat boxes. A query
then takes milliseconds and returns the same JSON shape as Overpass, so
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
from mapcreator.providers.osm.overpass import DRIVABLE

# Bump when the table layout changes; queries refuse an index built for another layout.
SCHEMA_VERSION = "1"
ROAD_CLASSES = frozenset(DRIVABLE.split("|"))
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
    kind TEXT NOT NULL,       -- 'road' or 'building'
    tags_json TEXT NOT NULL,
    coords BLOB NOT NULL,     -- little-endian float64, (n, 2) lon/lat in way order
    node_ids BLOB             -- little-endian int64, (n,) OSM node ids; roads only
);
-- One R*Tree per kind, so a road query in a city never wades through its buildings.
CREATE VIRTUAL TABLE road_rtree USING rtree(id, minx, maxx, miny, maxy);
CREATE VIRTUAL TABLE building_rtree USING rtree(id, minx, maxx, miny, maxy);
"""
_RTREE = {"road": "road_rtree", "building": "building_rtree"}


class IndexUnavailable(RuntimeError):
    """The index file is missing, isn't an index, or was built for another table layout."""


@dataclass(slots=True)
class _Way:
    """A road or building as either reader hands it over, with nodes of unknown location dropped."""

    id: int
    kind: str  # "road" or "building"
    tags: dict[str, str]
    coords: np.ndarray  # (n, 2) float64 lon/lat in way order
    refs: list[int] | None  # OSM node ids in line with `coords`, for roads; None for buildings
    complete: bool  # every node's location was known


def _kind(highway: str | None, building: str | None, closed: bool) -> str | None:
    """Which kind of way this is, from the tags that decide it, or None to skip it unread."""
    if highway in ROAD_CLASSES:
        return "road"
    if building not in (None, "no") and closed:
        return "building"
    return None


# ----- building -----

def build_index(source: str | os.PathLike, out: str | os.PathLike, *, reader: str = "auto",
                locations: str = DEFAULT_LOCATIONS,
                progress: Callable[[dict[str, int]], None] | None = None) -> dict[str, int]:
    """Index the drivable roads and building footprints of an OSM file into a new SQLite file.

    `reader` is "osmium" (pyosmium: .pbf or XML), "xml" (standard library: XML only) or "auto"
    (pyosmium when installed, else XML). `locations` picks pyosmium's node location store.
    `progress`, if given, is called with running counts every few hundred thousand ways.
    Returns the counts: roads and buildings stored, and highway or building ways skipped.
    """
    src, dst = Path(source), Path(out)
    if not src.is_file():
        raise FileNotFoundError(f"no OSM file at {src}")
    ways, header = _open_reader(src, reader, locations)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    tmp.unlink(missing_ok=True)
    counts = {"roads": 0, "buildings": 0, "skipped": 0}
    next_report = _REPORT_EVERY
    try:
        with closing(sqlite3.connect(tmp)) as con:
            # A crash leaves only the .partial file, which the next build deletes, so no journal. A
            # 256 MB page cache holds a national extract's R*Tree nodes while they're being filled.
            con.executescript("PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF; PRAGMA cache_size = -262144;"
                              + _SCHEMA)
            batch: dict[str, list] = {"road": [], "building": []}
            for way in ways:
                row = _row(way) if way is not None else None
                if row is None:
                    counts["skipped"] += 1
                    continue
                kind = row[1]
                batch[kind].append(row)
                counts["roads" if kind == "road" else "buildings"] += 1
                if len(batch[kind]) >= _BATCH:
                    _insert(con, kind, batch[kind])
                    batch[kind] = []
                if progress and counts["roads"] + counts["buildings"] >= next_report:
                    progress(dict(counts))
                    next_report += _REPORT_EVERY
            for kind, rows in batch.items():
                _insert(con, kind, rows)
            meta = {
                "schema": SCHEMA_VERSION,
                "source": src.name,  # the file name only: build machines' paths stay out of the index
                "osm_timestamp": header.get("osm_timestamp", ""),
                "built": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "reader": header["reader"],
                "road_classes": DRIVABLE,
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
    """The `ways` row for a road or building, plus its coordinates, or None if too little of it is known."""
    if way.kind == "road":
        if len(way.coords) < 2:
            return None  # at the edge of an extract, a way can lose all but one node
        tags, node_ids = way.tags, np.asarray(way.refs, dtype="<i8").tobytes()
    else:
        # A footprint with a node missing would be a different shape, so drop it whole.
        if not way.complete or len(way.coords) < 4:
            return None
        tags, node_ids = _building_tags(way.tags), None
    coords = way.coords.astype("<f8", copy=False)
    return (way.id, way.kind, json.dumps(tags, ensure_ascii=False, separators=(",", ":")), coords.tobytes(),
            node_ids, coords)


def _building_tags(tags: dict[str, str]) -> dict[str, str]:
    """Only a building's shape and height: addresses and names would bloat a national index for nothing."""
    return {k: v for k, v in tags.items()
            if k == "building" or k.startswith(("building:", "roof:")) or k in ("height", "min_height")}


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
        return _read_xml(src), {"reader": "xml"}
    return _read_osmium(src, locations)


def _read_osmium(src: Path, locations: str) -> tuple[Iterator[_Way | None], dict[str, str]]:
    """Ways through pyosmium. Yields None for a highway or building way that isn't kept."""
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
    # coordinates; only ways tagged highway or building ever reach Python.
    fp = (osmium.FileProcessor(str(src))
          .with_locations(locations)
          .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
          .with_filter(osmium.filter.KeyFilter("highway", "building")))
    header = {"reader": f"pyosmium {osmium.version.pyosmium_release}", "osm_timestamp": timestamp}
    wkb = osmium.geom.WKBFactory()
    every_node = osmium.geom.use_nodes.ALL  # keep repeated locations, so coordinates line up with node ids

    def ways() -> Iterator[_Way | None]:
        # Per-node access from Python costs microseconds a node, and a national extract has tens
        # of millions of them, so coordinates come from one WKB call per way. Only roads need
        # node ids. A way with a node of unknown location (the edge of an extract) can't be made
        # into WKB, so it takes the slow path, which drops that node.
        for w in fp:
            tags = w.tags
            building = tags.get("building")
            kind = _kind(tags.get("highway"), building, w.is_closed())
            if kind is None:
                yield None
                continue
            try:
                coords = _wkb_coords(wkb.create_linestring(w, every_node))
                refs = [n.ref for n in w.nodes] if kind == "road" else None
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
                refs = refs if kind == "road" else None
            # Reading tags costs microseconds each, and most footprints are tagged building=yes alone.
            tag_dict = {"building": building} if kind == "building" and len(tags) == 1 else dict(tags)
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


def _read_xml(src: Path) -> Iterator[_Way | None]:
    """Ways from an OSM XML file, in one pass (nodes come before ways in a well-formed file).

    Yields None for a highway or building way that isn't kept, like the pyosmium reader.
    """
    nodes: dict[int, tuple[float, float]] = {}
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
            elif el.tag == "way":
                tags = {t.get("k"): t.get("v") for t in el.iter("tag")}
                if "highway" in tags or "building" in tags:
                    all_refs = [int(nd.get("ref")) for nd in el.iter("nd")]
                    closed = len(all_refs) > 1 and all_refs[0] == all_refs[-1]
                    kind = _kind(tags.get("highway"), tags.get("building"), closed)
                    if kind is None:
                        yield None
                    else:
                        refs = [r for r in all_refs if r in nodes]
                        coords = np.array([nodes[r] for r in refs], dtype="<f8").reshape(-1, 2)
                        yield _Way(int(el.get("id")), kind, tags, coords, refs if kind == "road" else None,
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
    if row is None or row[0] != SCHEMA_VERSION:
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


def query(index_path: str | os.PathLike, bbox: tuple[float, float, float, float]) -> dict:
    """The drivable roads crossing `bbox` as Overpass JSON: `{"elements": [nodes..., ways...]}`.

    `bbox` is west, south, east, north in WGS84 degrees (the order of `Region.bbox` and shapely's
    `bounds`). Like Overpass's `way(bbox); (._;>;);`, a way is included whole, with every node,
    if any part of it crosses the box. Nodes carry no tags; the graph doesn't use them.
    """
    area = _check_bbox(bbox)
    with closing(_connect(index_path)) as con:
        rows = _candidates(con, "road", area)
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
                                 description="Build or inspect the local OSM index of roads and buildings.")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="index an OSM extract (.osm.pbf needs pyosmium; .osm XML works without)")
    b.add_argument("extract", type=Path)
    b.add_argument("out", type=Path)
    b.add_argument("--reader", choices=("auto", "osmium", "xml"), default="auto")
    b.add_argument("--locations", default=DEFAULT_LOCATIONS,
                   help="pyosmium node location store: flex_mem (default, in memory) or "
                        "sparse_file_array,<file> to keep node locations on disk")
    q = sub.add_parser("query", help="count the roads and buildings in a box, for debugging")
    q.add_argument("index", type=Path)
    for name in ("west", "south", "east", "north"):
        q.add_argument(name, type=float)
    q.add_argument("--epsg", type=int, help="metric CRS for the buildings query (default: the box's MGA zone)")
    q.add_argument("--json", action="store_true", help="print the roads as Overpass JSON")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    if args.command == "build":
        def report(c: dict[str, int]) -> None:
            print(f"  {c['roads']:,} roads, {c['buildings']:,} buildings ({time.perf_counter() - t0:.0f} s)",
                  file=sys.stderr, flush=True)

        counts = build_index(args.extract, args.out, reader=args.reader, locations=args.locations, progress=report)
        size_mb = args.out.stat().st_size / 1e6
        print(f"indexed {counts['roads']:,} roads and {counts['buildings']:,} buildings "
              f"({counts['skipped']:,} other ways skipped) from {args.extract.name} "
              f"in {time.perf_counter() - t0:.1f} s: {args.out.name}, {size_mb:,.1f} MB")
        return 0

    bbox = (args.west, args.south, args.east, args.north)
    meta = metadata(args.index)
    roads = query(args.index, bbox)
    t_roads = time.perf_counter() - t0
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
    print(f"buildings: {len(footprints):,} in EPSG:{epsg} ({t_buildings:.3f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
