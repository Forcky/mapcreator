"""TasNetworks' overhead transmission lines and towers (theLIST open data, CC BY 3.0 AU), by bounding box.

Two theLIST layers, queried like the building footprints and cached:
- Tower Locations: a point at every transmission tower.
- Transmission Lines: two kinds of polyline, told apart by the data itself. Route lines run
  tower to tower, every vertex on a tower; circuit lines (named "... kV Transmission
  Circuit") are drawn beside them, one per circuit. A route line gives the spans; the
  circuits that follow it give its voltage and how many circuits its towers carry.

The open data download of the same datasets states the licence (`readme.txt`: "Data is
supplied under Creative Commons Attribution 3.0 Australia"). TasNetworks' distribution
poles and low-voltage lines are on theLIST too, but restricted to emergency services, so
they are never read here.
"""

from __future__ import annotations

import hashlib
import json
import time

import httpx
import numpy as np
import shapely
from scipy.spatial import cKDTree

from mapcreator.config import Settings
from mapcreator.core import power
from mapcreator.providers import cache as cache_io
from mapcreator.providers.base import Attribution

ATTRIBUTION = Attribution(
    dataset='"TasNetworks Overhead Transmission Line" and "TasNetworks Tower Locations" from theLIST',
    holder="© State of Tasmania",
    licence="CC BY 3.0 AU",
    licence_url="https://creativecommons.org/licenses/by/3.0/au/",
)
SERVICE = "https://services.thelist.tas.gov.au/arcgis/rest/services/Public/Infrastructure/MapServer"
TOWERS, LINES = 59, 61
EPSG = 28355
PAGE = 1000
ON_TOWER_M = 1.5  # a route line's vertex is on a tower when this close to one
FOLLOWS_M = 60.0  # a circuit follows a route line when its spans' midpoints are this close to it


class ServiceError(RuntimeError):
    """theLIST answered, but not with features."""


def _fetch(layer: int, bbox, fields: str, settings: Settings) -> list[dict]:
    features, offset = [], 0
    while True:
        params = {
            "geometry": json.dumps({"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3],
                                    "spatialReference": {"wkid": EPSG}}),
            "geometryType": "esriGeometryEnvelope", "inSR": EPSG, "spatialRel": "esriSpatialRelIntersects",
            "outFields": fields, "returnGeometry": "true", "outSR": EPSG, "f": "json",
            "resultOffset": offset, "resultRecordCount": PAGE,
        }
        page = None
        for attempt in range(4):
            try:
                r = httpx.get(f"{SERVICE}/{layer}/query", params=params, timeout=90,
                              headers={"User-Agent": settings.user_agent})
                r.raise_for_status()
                page = r.json()
                # ArcGIS reports a failed query as HTTP 200 with an "error" object instead of features.
                if "error" in page or "features" not in page:
                    raise ServiceError(f"theLIST layer {layer} query failed: {str(page.get('error'))[:200]}")
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


def lines(bbox: tuple[float, float, float, float], settings: Settings) -> list[power.Line]:
    """Transmission lines (MGA55) with a tower within `bbox`, cut to their towers in the box."""
    key = hashlib.sha256(json.dumps([round(v) for v in bbox]).encode()).hexdigest()[:16]
    cache = settings.cache_dir / "tas_power" / f"{key}.json"
    data = cache_io.read_json(cache)
    if data is None:
        data = {"towers": _fetch(TOWERS, bbox, "LOCATION_ID", settings),
                "lines": _fetch(LINES, bbox, "LINE_NAME", settings)}
        cache_io.write_json(cache, data)  # only complete answers are cached
    return from_features(data["towers"], data["lines"], bbox)


def from_features(towers: list[dict], features: list[dict], bbox) -> list[power.Line]:
    """Route lines, cut to their runs of towers inside `bbox`, with voltage and circuits from the circuit lines."""
    pts = np.array([[f["geometry"]["x"], f["geometry"]["y"]] for f in towers if f.get("geometry")], dtype=float)
    if len(pts) == 0:
        return []
    tree = cKDTree(pts)
    box = shapely.box(*bbox)
    routes, circuits = [], []
    for f in features:
        name = (f.get("attributes") or {}).get("LINE_NAME") or ""
        for path in (f.get("geometry") or {}).get("paths") or []:
            xy = np.asarray(path, dtype=float).reshape(-1, 2)
            if len(xy) < 2:
                continue
            d, _ = tree.query(xy)
            on = d <= ON_TOWER_M
            inside = shapely.contains_xy(box, xy[:, 0], xy[:, 1])
            if on[inside].sum() >= 2 and on[inside].mean() >= 0.5:
                routes.append((name, xy, on & inside))
            elif "kv" in name.lower():
                circuits.append((name, shapely.LineString(xy)))
    out = []
    for name, xy, keep in routes:
        for lo, hi in _runs(keep):
            run = xy[lo:hi]
            if len(run) < 2:
                continue
            mids = shapely.points((run[:-1] + run[1:]) / 2)
            following = [(c_name, float(np.median(shapely.distance(mids, c)))) for c_name, c in circuits]
            following = [c_name for c_name, dist in following if dist <= FOLLOWS_M]
            kv = max((power.parse_kv(c) or 0.0 for c in following), default=0.0) or None
            out.append(power.Line(run, power.TRANSMISSION, kv, max(len(following), 1), f"TasNetworks: {name}"))
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[lo, hi) index ranges where mask is True."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))
