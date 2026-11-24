"""LIST Cadastral Parcels (theLIST, CC BY 3.0 AU), queried by bounding box and cached: where the road reserve ends.

Roads in the cadastre are the gaps between parcels, or parcels typed `Casement`. So the parcels a road's footpath
backs onto are every parcel that isn't a casement or water (`CAD_TYPE1`), and the first such boundary out from the
centreline is the reserve's edge: the footpath's back edge (`core.kerbs`). The kerbs and footpaths research measured
it beyond 95-99% of Hobart's kerbs, a median 2.2-2.8 m behind the kerb face.

The layer is `Public/OpenDataWFS/MapServer/14` on theLIST's ArcGIS server, the same host as the building footprints
(`buildings.py`). It answers at most 2,000 features per request, so results are paged; only a complete answer is
cached. Only the parcels' shapes reach a track, as the edge of a footpath: no owner, title or valuation field is
asked for.
"""

from __future__ import annotations

import hashlib
import json
import time

import httpx
import shapely

from mapcreator.config import Settings
from mapcreator.providers import cache as cache_io
from mapcreator.providers.base import Attribution

ATTRIBUTION = Attribution(
    dataset='"LIST Cadastral Parcels" from theLIST',
    holder="© State of Tasmania",
    licence="CC BY 3.0 AU",
    licence_url="https://creativecommons.org/licenses/by/3.0/au/",
)
URL = "https://services.thelist.tas.gov.au/arcgis/rest/services/Public/OpenDataWFS/MapServer/14/query"
EPSG = 28355
PAGE = 1000
# Parcel types that are the road itself (a casement over a road reserve) or water, not land a footpath backs onto.
NOT_LOTS = frozenset({"Casement", "Water"})


class ServiceError(RuntimeError):
    """theLIST answered, but not with features."""


def _fetch_page(bbox, offset: int, settings: Settings) -> dict:
    params = {
        "geometry": json.dumps({"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3],
                                "spatialReference": {"wkid": EPSG}}),
        "geometryType": "esriGeometryEnvelope", "inSR": EPSG, "spatialRel": "esriSpatialRelIntersects",
        "outFields": "CAD_TYPE1", "returnGeometry": "true", "outSR": EPSG, "f": "json",
        "resultOffset": offset, "resultRecordCount": PAGE,
    }
    for attempt in range(4):
        try:
            r = httpx.get(URL, params=params, timeout=90, headers={"User-Agent": settings.user_agent})
            r.raise_for_status()
            page = r.json()
            # ArcGIS reports a failed query as HTTP 200 with an "error" object instead of features.
            if "error" in page or "features" not in page:
                raise ServiceError(f"theLIST cadastre query failed: {str(page.get('error'))[:200]}")
            return page
        except (httpx.HTTPError, ServiceError, ValueError):
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _features(bbox, settings: Settings) -> list[dict]:
    key = hashlib.sha256(json.dumps([round(v) for v in bbox]).encode()).hexdigest()[:16]
    cache = settings.cache_dir / "tas_parcels" / f"{key}.json"
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
    return features


def lots(bbox: tuple[float, float, float, float], settings: Settings) -> list[shapely.Polygon]:
    """The parcels (MGA55) intersecting the box that aren't road casements or water: the land beside a road reserve."""
    out = []
    for f in _features(bbox, settings):
        if (f.get("attributes") or {}).get("CAD_TYPE1") in NOT_LOTS:
            continue
        poly = esri_polygon((f.get("geometry") or {}).get("rings") or [])
        if poly is not None:
            out.extend(p for p in shapely.get_parts(poly) if p.geom_type == "Polygon" and not p.is_empty)
    return out


def esri_polygon(rings: list) -> shapely.Geometry | None:
    """An Esri JSON polygon's rings as a (multi)polygon: clockwise rings are outer rings and anticlockwise ones holes
    (the Esri convention), each hole in the outer ring holding it. A parcel in parts keeps every part."""
    outers, holes = [], []
    for ring in rings:
        if len(ring) < 4:
            continue
        lr = shapely.LinearRing([p[:2] for p in ring])
        (holes if shapely.is_ccw(lr) else outers).append(lr)
    if not outers:
        return None
    polys = []
    for outer in outers:
        shell = shapely.Polygon(outer)
        mine = [h for h in holes if shell.contains(shapely.Polygon(h).representative_point())]
        polys.append(shapely.make_valid(shapely.Polygon(outer, mine)))
    return shapely.unary_union(polys)
