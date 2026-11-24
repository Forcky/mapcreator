"""Councils' own kerb lines (the kerbs and footpaths research's phase 2 data, pulled forward): the City of Hobart's
Road Formation and Glenorchy City Council's kerbs, read from a local mirror.

**A kerb is its own feature** (the research's Q6). These lines only say where a kerb stands beside the road; the road's
centreline, profile, tags and routing stay OpenStreetMap's alone (`core.kerbs.council_faces`).

**Where they come from.** Neither is fetched by a build. An operator job (`scripts/mirror_kerb_lines.py`, outside the
worker) downloads both on a schedule into one SQLite file, which the worker reads by box (`kerb_lines` in the
settings, `MAPCREATOR_KERB_LINES`), as it reads the OSM index: the worker needs no new egress. ArcGIS Online's host
serves every one of its customers, so allowing it to the worker would open it to far more than Hobart's layer.
- **City of Hobart Road Formation** (ArcGIS Online, CC BY 4.0, the item's `licenseInfo`): layer 0 of
  `services1.arcgis.com/NHqdsnvwfSTg42I8/arcgis/rest/services/Road_Formation/FeatureServer`, lines typed `roatkb`
  (Top of Kerb) and `roaeof` (Edge of Formation, an unkerbed edge), in EPSG:28355.
- **Glenorchy City Council Kerbs** (data.gov.au `glenorchy-city-council-kerbs`, CC BY 3.0 AU), the GeoPackage in
  `maps.gcc.tas.gov.au/OpenData/gpkg/kerbs.zip`, EPSG:28355, kerb lines with no type (0.5 m accuracy for 92%).
- Clarence publishes nothing: kerbs there come from the rules and the cadastre (`core.kerbs.plan`).

**Without the mirror** (not configured, or not reachable) kerbs come from the rules, as in phase 1, and a road inside
either council's area gets a warning. For development only, `MAPCREATOR_KERB_LINES=live` reads both sources straight
from their hosts instead (`live`), cached; never set it on a worker.

The mirror (`FORMAT_VERSION`): table `sources` (name, url, licence, fetched, modified, features), table `lines` (id,
source, kind: `kerb` or `edge`, its line as WKB in EPSG:28355) and an R*Tree `lines_rtree` on their boxes. A refresh
writes a new file and moves it into place, so no build reads a half-written one.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import shapely

from mapcreator.providers.base import Attribution

FORMAT_VERSION = 1
EPSG = 28355
KERB, EDGE = "kerb", "edge"

HOBART = "hobart"
GLENORCHY = "glenorchy"
ATTRIBUTIONS = {
    HOBART: Attribution(
        dataset='"Road Formation" (Top of Kerb and Edge of Formation) from the City of Hobart',
        holder="© City of Hobart",
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
    ),
    GLENORCHY: Attribution(
        dataset='"Glenorchy City Council Kerbs"',
        holder="© Glenorchy City Council",
        licence="CC BY 3.0 AU",
        licence_url="https://creativecommons.org/licenses/by/3.0/au/",
    ),
}
HOBART_LAYER = ("https://services1.arcgis.com/NHqdsnvwfSTg42I8/arcgis/rest/services/Road_Formation/FeatureServer/0")
HOBART_TYPES = {"roatkb": KERB, "roaeof": EDGE}
HOBART_PAGE = 1000  # the layer's maxRecordCount
# The mirror keeps each line simplified to this (metres): Hobart's lines carry a vertex every centimetre or so, 84 MB
# as they come; a return's arc is still drawn to within 2 cm.
MIRROR_SIMPLIFY_M = 0.02
GLENORCHY_ZIP = "https://maps.gcc.tas.gov.au/OpenData/gpkg/kerbs.zip"
# Each council's lines' extent (MGA55 min x, min y, max x, max y; Hobart's layer extent and Glenorchy's GeoPackage
# contents, 24 Nov 2026), and 200 m more: a road inside one has kerb lines to expect, so going without them gets a
# warning, and a live read (`live`) asks only the councils whose area it reaches.
AREAS = {HOBART: (517800.0, 5245600.0, 529700.0, 5256500.0), GLENORCHY: (511200.0, 5254300.0, 526200.0, 5267100.0)}


@dataclass
class KerbLines:
    """The councils' lines in a box: (source, kind, line) each, in EPSG:28355, and the sources they came from."""
    lines: list = field(default_factory=list)
    sources: set = field(default_factory=set)

    def kind(self, kind: str) -> list:
        return [ln for _, k, ln in self.lines if k == kind]

    def attributions(self) -> list[Attribution]:
        return [ATTRIBUTIONS[s] for s in sorted(self.sources) if s in ATTRIBUTIONS]


def areas_for(bbox: tuple[float, float, float, float]) -> list[str]:
    """The councils whose area the box reaches into."""
    x0, y0, x1, y1 = bbox
    return [k for k, (a, b, c, d) in AREAS.items() if x0 <= c and x1 >= a and y0 <= d and y1 >= b]


# -- reading the mirror ------------------------------------------------------------------------------------------------

def read(path: str | Path, bbox: tuple[float, float, float, float]) -> KerbLines:
    """The mirror's lines whose boxes meet `bbox` (MGA55). The file is opened read-only and immutable."""
    uri = f"file:{Path(path).as_posix()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        version = con.execute("SELECT value FROM meta WHERE key = 'format_version'").fetchone()
        if version is None or int(version[0]) != FORMAT_VERSION:
            raise ValueError(f"kerb lines mirror {path}: format {version} isn't {FORMAT_VERSION}; build it again")
        rows = con.execute(
            "SELECT l.source, l.kind, l.wkb FROM lines_rtree r JOIN lines l ON l.id = r.id "
            "WHERE r.maxx >= ? AND r.minx <= ? AND r.maxy >= ? AND r.miny <= ?",
            (bbox[0], bbox[2], bbox[1], bbox[3])).fetchall()
    finally:
        con.close()
    out = KerbLines()
    for source, kind, wkb in rows:
        out.lines.append((source, kind, shapely.from_wkb(wkb)))
        out.sources.add(source)
    return out


# -- fetching (the mirror job, and development) ----------------------------------------------------------------------

def _get(url: str, params: dict | None = None, timeout: float = 120.0, user_agent: str = "mapcreator mirror"):
    import httpx

    for attempt in range(4):
        try:
            r = httpx.get(url, params=params, timeout=timeout, headers={"User-Agent": user_agent},
                          follow_redirects=True)
            r.raise_for_status()
            return r
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def hobart_lines(bbox: tuple[float, float, float, float] | None = None, user_agent: str = "mapcreator mirror"):
    """Hobart's Top of Kerb and Edge of Formation lines, (kind, line) each, all of them or those meeting `bbox`
    (MGA55): its FeatureServer's query, paged by HOBART_PAGE, as Esri JSON in EPSG:28355."""
    where = "Type_Description IN ('roatkb','roaeof')"
    out, offset = [], 0
    while True:
        params = {"where": where, "outFields": "Type_Description", "returnGeometry": "true", "outSR": EPSG,
                  "f": "json", "resultOffset": offset, "resultRecordCount": HOBART_PAGE, "orderByFields": "OBJECTID"}
        if bbox is not None:
            params |= {"geometry": json.dumps({"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3],
                                               "spatialReference": {"wkid": EPSG}}),
                       "geometryType": "esriGeometryEnvelope", "inSR": EPSG, "spatialRel": "esriSpatialRelIntersects"}
        page = _get(f"{HOBART_LAYER}/query", params, user_agent=user_agent).json()
        if "error" in page or "features" not in page:
            raise RuntimeError(f"Hobart's Road Formation query failed: {str(page.get('error'))[:200]}")
        for f in page["features"]:
            kind = HOBART_TYPES.get((f.get("attributes") or {}).get("Type_Description"))
            for path in (f.get("geometry") or {}).get("paths") or []:
                if kind and len(path) >= 2:
                    out.append((kind, shapely.LineString([p[:2] for p in path])))
        if not page.get("exceededTransferLimit") or not page["features"]:
            return out
        offset += len(page["features"])


def glenorchy_lines(data: bytes | None = None, user_agent: str = "mapcreator mirror"):
    """Glenorchy's kerb lines, (KERB, line) each, from its zipped GeoPackage (downloaded unless `data` is given)."""
    data = data if data is not None else _get(GLENORCHY_ZIP, user_agent=user_agent).content
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".gpkg"))
        gpkg = z.read(name)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "kerbs.gpkg"
        p.write_bytes(gpkg)
        con = sqlite3.connect(p)
        try:
            table, column = con.execute(
                "SELECT table_name, column_name FROM gpkg_geometry_columns LIMIT 1").fetchone()
            blobs = [r[0] for r in con.execute(f'SELECT "{column}" FROM "{table}"')]
        finally:
            con.close()
    out = []
    for blob in blobs:
        if blob is None:
            continue
        g = shapely.from_wkb(_gpkg_wkb(bytes(blob)))
        for part in shapely.get_parts(g):
            if part.geom_type == "LineString" and len(part.coords) >= 2:
                out.append((KERB, shapely.force_2d(part)))
    return out


def _gpkg_wkb(blob: bytes) -> bytes:
    """A GeoPackage geometry blob's WKB: past its 8-byte header and the envelope its flags say it carries."""
    flags = blob[3]
    envelope = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    return blob[8 + envelope:]


def write_mirror(path: str | Path, sources: dict[str, list], meta: dict[str, dict]) -> None:
    """Write the mirror at `path` (a new file, moved into place when complete): `sources` is each source's
    (kind, line) list, `meta` its url, licence and dates."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE sources (name TEXT PRIMARY KEY, url TEXT, licence TEXT, fetched TEXT, modified TEXT,
                                  features INTEGER);
            CREATE TABLE lines (id INTEGER PRIMARY KEY, source TEXT, kind TEXT, wkb BLOB);
            CREATE VIRTUAL TABLE lines_rtree USING rtree(id, minx, maxx, miny, maxy);
        """)
        con.execute("INSERT INTO meta VALUES ('format_version', ?)", (str(FORMAT_VERSION),))
        con.execute("INSERT INTO meta VALUES ('crs', ?)", (f"EPSG:{EPSG}",))
        k = 0
        for name, lines in sources.items():
            m = meta.get(name, {})
            con.execute("INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
                        (name, m.get("url"), m.get("licence"), m.get("fetched"), m.get("modified"), len(lines)))
            for kind, line in lines:
                line = shapely.simplify(line, MIRROR_SIMPLIFY_M, preserve_topology=False)
                if line.is_empty or line.length <= 0:
                    continue
                k += 1
                x0, y0, x1, y1 = line.bounds
                con.execute("INSERT INTO lines VALUES (?, ?, ?, ?)", (k, name, kind, shapely.to_wkb(line)))
                con.execute("INSERT INTO lines_rtree VALUES (?, ?, ?, ?, ?)", (k, x0, x1, y0, y1))
        con.commit()
    finally:
        con.close()
    tmp.replace(path)


def live(bbox: tuple[float, float, float, float], cache_dir: Path, user_agent: str) -> KerbLines:
    """Development only: both councils' lines in a box, straight from their hosts, cached per box (Hobart's) and whole
    (Glenorchy's zip). A worker reads the mirror."""
    out = KerbLines()
    areas = areas_for(bbox)
    folder = Path(cache_dir) / "kerb_lines"
    folder.mkdir(parents=True, exist_ok=True)
    if HOBART in areas:
        key = hashlib.sha256(json.dumps([round(v) for v in bbox]).encode()).hexdigest()[:16]
        f = folder / f"hobart_{key}.wkb.json"
        if f.exists():
            got = [(k, shapely.from_wkb(bytes.fromhex(h))) for k, h in json.loads(f.read_text())]
        else:
            got = hobart_lines(bbox, user_agent)
            f.write_text(json.dumps([(k, shapely.to_wkb(ln, hex=True)) for k, ln in got]))
        out.lines += [(HOBART, k, ln) for k, ln in got]
        if got:
            out.sources.add(HOBART)
    if GLENORCHY in areas:
        z = folder / "glenorchy_kerbs.zip"
        if not z.exists():
            z.write_bytes(_get(GLENORCHY_ZIP, user_agent=user_agent).content)
        box = shapely.box(*bbox)
        got = [(k, ln) for k, ln in glenorchy_lines(z.read_bytes()) if ln.intersects(box)]
        out.lines += [(GLENORCHY, k, ln) for k, ln in got]
        if got:
            out.sources.add(GLENORCHY)
    return out
