"""Where Tasmania's statewide 2 m DEM is LiDAR: theLIST's LiDAR Coverage Index - All Projects (CC BY 3.0 AU).

The statewide DEM (released August 2021) is about 65% LiDAR. The rest is interpolated from photogrammetric
contours, spot heights and the coastline at about 10 m, then resampled onto the same 2 m grid, so the grid
alone doesn't say which is which. The index has each LiDAR project's footprint and completion date; a
project finished after the DEM was made isn't in it. On the Lyell Highway (Nov 2026) the index put LiDAR
in the DEM at 57.8-63.0 km only (Forestry Tasmania 2014), and that is exactly where a hillshade shows the
road's cuttings. The 2021 and 2023-24 projects there came too late for the DEM.

Used only to decide how to treat the DEM; nothing from the index ships in a track.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import httpx
import shapely

from mapcreator.providers import cache as cache_io

URL = "https://services.thelist.tas.gov.au/arcgis/rest/services/Public/Indexes/MapServer/97/query"
EPSG = 28355
PAGE = 1000


class ServiceError(RuntimeError):
    """theLIST answered, but not with features."""


def _fetch(bbox, offset: int, user_agent: str) -> dict:
    params = {
        "geometry": json.dumps({"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3],
                                "spatialReference": {"wkid": EPSG}}),
        "geometryType": "esriGeometryEnvelope", "inSR": EPSG, "spatialRel": "esriSpatialRelIntersects",
        "outFields": "PROJ_NAME,COMPLETED,STATUS", "returnGeometry": "true", "outSR": EPSG,
        "maxAllowableOffset": 5, "f": "json", "resultOffset": offset, "resultRecordCount": PAGE,
    }
    for attempt in range(4):
        try:
            r = httpx.get(URL, params=params, timeout=90, headers={"User-Agent": user_agent})
            r.raise_for_status()
            page = r.json()
            if "error" in page or "features" not in page:  # ArcGIS reports a failed query as HTTP 200
                raise ServiceError(f"theLIST LiDAR index query failed: {str(page.get('error'))[:200]}")
            return page
        except (httpx.HTTPError, ServiceError, ValueError):
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def projects(bbox: tuple[float, float, float, float], cache_dir: Path, user_agent: str) -> list[dict]:
    """LiDAR project footprints (MGA55, Esri JSON) intersecting the box, from cache when available."""
    key = hashlib.sha256(json.dumps([round(v, -1) for v in bbox]).encode()).hexdigest()[:16]
    path = cache_dir / "tas_lidar_index" / f"{key}.json"
    features = cache_io.read_json(path)
    if features is None:
        features, offset = [], 0
        while True:
            page = _fetch(bbox, offset, user_agent)
            got = page.get("features", [])
            features.extend(got)
            if not page.get("exceededTransferLimit") or not got:
                break
            offset += len(got)
        cache_io.write_json(path, features)  # only a complete answer is cached
    return features


def coverage(features: list[dict], before: dt.date) -> shapely.Geometry:
    """The area of the projects completed before `before`: the LiDAR a DEM made then can hold."""
    cutoff = dt.datetime.combine(before, dt.time(), dt.UTC).timestamp() * 1000
    polys = []
    for f in features:
        done = (f.get("attributes") or {}).get("COMPLETED")
        rings = (f.get("geometry") or {}).get("rings") or []
        if done is None or done >= cutoff or not rings:
            continue
        # Esri rings: outer rings clockwise, holes counter-clockwise.
        outer = [shapely.Polygon(r) for r in rings if len(r) >= 4 and not shapely.LinearRing(r).is_ccw]
        holes = [shapely.Polygon(r) for r in rings if len(r) >= 4 and shapely.LinearRing(r).is_ccw]
        if outer:
            shape = shapely.union_all(outer).buffer(0)
            if holes:
                shape = shape.difference(shapely.union_all(holes))
            polys.append(shape)
    return shapely.union_all(polys) if polys else shapely.GeometryCollection()
