"""TASVEG 5.0 vegetation communities (NRE Tas, on theLIST; CC BY 3.0 AU) around a road, cached.

Only class labels reach a track: each tree's model family comes from the community it stands in
(`core.species`). No geometry or imagery from TASVEG ships.

Queried like the building footprints, but with the corridor itself rather than its bounding box: a long
diagonal road's box would take in a whole district. A long road goes in pieces, so each request stays
small. Polygons come back generalised to 2 m by the server (TASVEG's own boundaries are good to
2.5-17.5 m) and are deduplicated by OBJECTID across pieces. Release 5.0 is frozen ("No further updates
for release version 5.0"), so a cached answer stays right until 6.0.

The licence is in the metadata statement inside the open-data download (`TASVEG_5_0_Metadata_CCby.pdf`):
CC BY 3.0 AU, credited as "TASVEG 5.0 from www.theLIST.tas.gov.au ©State of Tasmania". The REST layer's
own copyright text is empty.
"""

from __future__ import annotations

import hashlib
import json
import time

import httpx
import numpy as np
import shapely
from shapely.ops import substring

from mapcreator.config import Settings
from mapcreator.core.species import Community
from mapcreator.providers import cache as cache_io
from mapcreator.providers.base import Attribution

ATTRIBUTION = Attribution(
    dataset="TASVEG 5.0 from www.theLIST.tas.gov.au",
    holder="© State of Tasmania",
    licence="CC BY 3.0 AU",
    licence_url="https://creativecommons.org/licenses/by/3.0/au/",
)
URL = "https://services.thelist.tas.gov.au/arcgis/rest/services/Public/NaturalEnvironmentV2/MapServer/9/query"
EPSG = 28355
PAGE = 1000
FIELDS = "OBJECTID,VEGCODE,NOTABLE_TR,NOTABLE_TD,WEED_TYPE"
PIECE_M = 15_000.0  # centreline length per query
QUERY_SIMPLIFY_M = 10.0  # the query polygon's own tolerance; the corridor is buffered a little more to make up
SERVER_OFFSET_M = 2.0  # maxAllowableOffset: the server generalises polygons to this


class ServiceError(RuntimeError):
    """theLIST answered, but not with features."""


def _fetch(geometry: dict, settings: Settings) -> list[dict]:
    features, offset = [], 0
    while True:
        data = {
            "geometry": json.dumps(geometry, separators=(",", ":")), "geometryType": "esriGeometryPolygon",
            "inSR": EPSG, "spatialRel": "esriSpatialRelIntersects", "outFields": FIELDS, "returnGeometry": "true",
            "outSR": EPSG, "maxAllowableOffset": SERVER_OFFSET_M, "geometryPrecision": 1, "f": "json",
            "resultOffset": offset, "resultRecordCount": PAGE,
        }
        page = None
        for attempt in range(4):
            try:
                # POST: a corridor polygon is too long for a URL.
                r = httpx.post(URL, data=data, timeout=120, headers={"User-Agent": settings.user_agent})
                r.raise_for_status()
                page = r.json()
                # ArcGIS reports a failed query as HTTP 200 with an "error" object instead of features.
                if "error" in page or "features" not in page:
                    raise ServiceError(f"theLIST TASVEG query failed: {str(page.get('error'))[:200]}")
                break
            except (httpx.HTTPError, ServiceError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        got = page.get("features", [])
        features.extend(got)
        if not page.get("exceededTransferLimit") or not got:
            return features
        offset += len(got)


def _pieces(centre_xy: np.ndarray, reach_m: float) -> list[shapely.Polygon]:
    """The corridor, cut along the road into pieces of about PIECE_M, as simple query polygons."""
    line = shapely.LineString(centre_xy)
    n = max(int(np.ceil(line.length / PIECE_M)), 1)
    out = []
    for k in range(n):
        part = substring(line, k * line.length / n, (k + 1) * line.length / n)
        area = part.buffer(reach_m + QUERY_SIMPLIFY_M).simplify(QUERY_SIMPLIFY_M)
        out.extend(area.geoms if area.geom_type == "MultiPolygon" else [area])
    return out


def _esri_json(poly: shapely.Polygon) -> dict:
    # Esri rings: the exterior clockwise, holes counter-clockwise.
    poly = shapely.orient_polygons(poly, exterior_cw=True)
    rings = [np.round(np.asarray(poly.exterior.coords), 1).tolist()]
    rings += [np.round(np.asarray(r.coords), 1).tolist() for r in poly.interiors]
    return {"rings": rings, "spatialReference": {"wkid": EPSG}}


def esri_polygon(rings: list) -> shapely.Geometry:
    """An Esri JSON polygon (any number of parts and holes) as a valid shapely geometry. Clockwise rings are
    exteriors and counter-clockwise rings are holes, each in the exterior that holds it."""
    shells, holes = [], []
    for ring in rings:
        if len(ring) < 4:
            continue
        lr = shapely.LinearRing(ring)
        (holes if lr.is_ccw else shells).append(lr)
    if not shells:
        return shapely.Polygon()
    polys = [shapely.Polygon(s) for s in shells]
    owned: list[list] = [[] for _ in polys]
    for h in holes:
        p = shapely.Point(h.coords[0])
        for i, shell in enumerate(polys):
            if shell.contains(p):
                owned[i].append(h)
                break
    parts = [shapely.Polygon(s, owned[i]).buffer(0) for i, s in enumerate(shells)]
    return shapely.union_all([p for p in parts if not p.is_empty])


def from_features(features: list[dict]) -> list[Community]:
    out, seen = [], set()
    for f in features:
        a = f.get("attributes") or {}
        oid = a.get("OBJECTID")
        if oid in seen:
            continue
        seen.add(oid)
        geom = esri_polygon((f.get("geometry") or {}).get("rings") or [])
        if geom.is_empty:
            continue
        out.append(Community(geom, (a.get("VEGCODE") or "").strip().upper(), (a.get("NOTABLE_TR") or "").strip().upper(),
                             (a.get("NOTABLE_TD") or "").strip(), (a.get("WEED_TYPE") or "").strip().upper()))
    return out


def communities(centre_xy: np.ndarray, reach_m: float, settings: Settings) -> list[Community]:
    """TASVEG communities (MGA55) within `reach_m` of the centreline, from cache when available."""
    features = []
    for piece in _pieces(np.asarray(centre_xy, dtype=float), reach_m):
        geometry = _esri_json(piece)
        key = hashlib.sha256(json.dumps(geometry["rings"]).encode()).hexdigest()[:16]
        cache = settings.cache_dir / "tasveg" / f"{key}.json"
        got = cache_io.read_json(cache)
        if got is None:
            got = _fetch(geometry, settings)
            cache_io.write_json(cache, got)  # only complete answers are cached
        features.extend(got)
    return from_features(features)
