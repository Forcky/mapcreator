"""LIST 2D Building Polygons (theLIST, CC BY 3.0 AU), queried by bounding box and cached.

The REST layer returns at most 2,000 features per request, so results are
paged. Footprints only: the MEAN_HGT field is empty around C324, so heights
are left to the scenery stage.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx
import shapely

from mapcreator.config import Settings
from mapcreator.providers import cache as cache_io
from mapcreator.providers.base import Attribution

ATTRIBUTION = Attribution(
    dataset='"LIST 2D Building Polygons" from theLIST',
    holder="© State of Tasmania",
    licence="CC BY 3.0 AU",
    licence_url="https://creativecommons.org/licenses/by/3.0/au/",
)
URL = "https://services.thelist.tas.gov.au/arcgis/rest/services/Public/TopographyAndRelief/MapServer/35/query"
EPSG = 28355
PAGE = 1000


def _fetch_page(bbox, offset: int, settings: Settings) -> dict:
    params = {
        "geometry": json.dumps({"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3],
                                "spatialReference": {"wkid": EPSG}}),
        "geometryType": "esriGeometryEnvelope", "inSR": EPSG, "spatialRel": "esriSpatialRelIntersects",
        "outFields": "BUILD_TY", "returnGeometry": "true", "outSR": EPSG, "f": "json",
        "resultOffset": offset, "resultRecordCount": PAGE,
    }
    for attempt in range(4):
        try:
            r = httpx.get(URL, params=params, timeout=90, headers={"User-Agent": settings.user_agent})
            r.raise_for_status()
            page = r.json()
            # ArcGIS reports a failed query as HTTP 200 with an "error" object instead of features.
            if "error" in page or "features" not in page:
                raise ServiceError(f"theLIST buildings query failed: {str(page.get('error'))[:200]}")
            return page
        except (httpx.HTTPError, ServiceError, ValueError):
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


class ServiceError(RuntimeError):
    """theLIST answered, but not with features."""


def footprints(bbox: tuple[float, float, float, float], settings: Settings) -> list[shapely.Polygon]:
    """Building polygons (MGA55) intersecting the box, from cache when available."""
    key = hashlib.sha256(json.dumps([round(v) for v in bbox]).encode()).hexdigest()[:16]
    cache = settings.cache_dir / "tas_buildings" / f"{key}.json"
    features = cache_io.read_json(cache)
    if features is None:
        features, offset = [], 0
        while True:
            page = _fetch_page(bbox, offset, settings)
            got = page.get("features", [])
            features.extend(got)
            if not page.get("exceededTransferLimit") or not got:
                break
            offset += len(got)
        cache_io.write_json(cache, features)  # only a complete answer is cached
    polys = []
    for f in features:
        rings = (f.get("geometry") or {}).get("rings") or []
        if not rings:
            continue
        poly = shapely.Polygon(rings[0], rings[1:]).buffer(0)
        if poly.is_empty:
            continue
        polys.extend(poly.geoms if poly.geom_type == "MultiPolygon" else [poly])
    return polys
