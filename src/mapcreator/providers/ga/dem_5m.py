"""Geoscience Australia's 5 m LiDAR DEM (CC BY 4.0): fine elevation for the national fallback where it exists.

"Digital Elevation Model (DEM) of Australia derived from LiDAR 5 Metre Grid" covers about 245,000 km²: the
populated coastal zone, the Murray-Darling floodplains, and towns large and small (236 surveys, 2001-2015).
It's read through GA's WCS service in small tiles (GDA94 lon/lat), cached on disk like the other rasters.
Where there's no LiDAR the service answers 0.0 rather than no-data, so an exact 0.0 counts as no data.

`NationalElevation` is what the national regions use. It loads both this and DEM-S (~30 m) along the
corridor, then decides once per road: if the 5 m DEM covers (nearly) all of the centreline, the build uses
it, with DEM-S filling any small holes, and gets the fine-DEM treatment (DEM camber, guard rails). Otherwise
the whole build uses DEM-S, so a road never steps between two models where the LiDAR coverage ends.
The decision is kept in the cache, keyed by the centreline, so every stage of a build agrees even if the
service fails halfway through.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import LineString

from mapcreator.providers import cache
from mapcreator.providers.base import Attribution

log = logging.getLogger(__name__)

ATTRIBUTION = Attribution(
    dataset="Digital Elevation Model (DEM) of Australia derived from LiDAR 5 Metre Grid",
    holder="© Commonwealth of Australia (Geoscience Australia)",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
    note="https://doi.org/10.26186/89644",
)
WCS_URL = "https://services.ga.gov.au/gis/services/DEM_LiDAR_5m_2025/MapServer/WCSServer"
CRS = 4283  # GDA94 lon/lat, the service's own
RES_DEG = 0.00005  # about 5 m
TILE_DEG = 0.02  # 400 x 400 cells a request
# Share of the centreline the 5 m DEM must cover for a build to use it.
MIN_COVERAGE = 0.98

Fetcher = Callable[[tuple[float, float, float, float], int], np.ndarray]


def wcs_fetch(bbox: tuple[float, float, float, float], size: int, *, user_agent: str = "mapcreator") -> np.ndarray:
    """One tile from GA's WCS as float32 rows north to south, with 0.0 (no LiDAR) as NaN."""
    import httpx
    import rasterio

    params = {"service": "WCS", "version": "1.0.0", "request": "GetCoverage", "coverage": "1",
              "crs": f"EPSG:{CRS}", "bbox": ",".join(f"{v:.6f}" for v in bbox),
              "width": str(size), "height": str(size), "format": "GeoTIFF"}
    r = httpx.get(WCS_URL, params=params, timeout=60, headers={"User-Agent": user_agent})
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/tiff"):
        raise RuntimeError(f"GA WCS answered {r.headers.get('content-type')!r}, not a GeoTIFF")
    with rasterio.MemoryFile(io.BytesIO(r.content)) as mem, mem.open() as ds:
        a = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
    a[a == 0.0] = np.nan
    a[a < -1000] = np.nan
    return a


class Ga5mDem:
    """GA's 5 m DEM in TILE_DEG tiles, sampled in a region's projected CRS."""

    resolution_m = 5.0
    tile_px = int(round(TILE_DEG / RES_DEG))

    def __init__(self, cache_dir: Path, region_epsg: int, *, fetch: Fetcher | None = None):
        self.cache = cache_dir / "ga_dem_5m"
        self.to_ll = Transformer.from_crs(region_epsg, CRS, always_xy=True)
        self.fetch = fetch or wcs_fetch
        self.tiles: dict[tuple[int, int], np.ndarray] = {}

    def _tile(self, key: tuple[int, int]) -> np.ndarray:
        path = self.cache / f"{key[0]}_{key[1]}.npy"
        if path.exists():
            try:
                return np.load(path)
            except (OSError, ValueError, EOFError):
                pass
        w, s = key[0] * TILE_DEG, key[1] * TILE_DEG
        arr = self.fetch((w, s, w + TILE_DEG, s + TILE_DEG), self.tile_px).astype(np.float32)
        if arr.shape != (self.tile_px, self.tile_px):
            raise RuntimeError(f"GA WCS tile {key} came back {arr.shape}, not {self.tile_px} square")
        cache.save_npy(path, arr)
        return arr

    def load_corridor(self, line_xy, buffer_m: float) -> int:
        poly = LineString(line_xy).buffer(buffer_m, quad_segs=4)
        lon, lat = self.to_ll.transform(*np.asarray(poly.exterior.coords).T)
        footprint = shapely.Polygon(np.column_stack([lon, lat]))
        x0, y0, x1, y1 = footprint.bounds
        shapely.prepare(footprint)
        for i in range(math.floor(x0 / TILE_DEG), math.floor(x1 / TILE_DEG) + 1):
            for j in range(math.floor(y0 / TILE_DEG), math.floor(y1 / TILE_DEG) + 1):
                if (i, j) in self.tiles:
                    continue
                cell = shapely.box(i * TILE_DEG, j * TILE_DEG, (i + 1) * TILE_DEG, (j + 1) * TILE_DEG)
                if footprint.intersects(cell):
                    self.tiles[(i, j)] = self._tile((i, j))
        return len(self.tiles)

    def sample(self, x, y) -> np.ndarray:
        """Bilinear heights, any array shape; NaN outside loaded tiles or where there's no LiDAR."""
        shape = np.shape(x)
        lon, lat = self.to_ll.transform(np.ravel(np.asarray(x, dtype=float)), np.ravel(np.asarray(y, dtype=float)))
        lon, lat = np.atleast_1d(lon), np.atleast_1d(lat)
        # Continuous cell coordinates over the whole tile grid (cell centres at half steps).
        col = lon / RES_DEG - 0.5
        row = -lat / RES_DEG - 0.5  # rows run north to south
        c0, r0 = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
        fc, fr = col - c0, row - r0
        out = np.zeros(lon.shape)
        for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc), (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
            out += w * self._cells(r0 + dr, c0 + dc)
        return out.reshape(shape) if shape else out

    def _cells(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        px = self.tile_px
        ti = np.floor_divide(cols, px)
        # Tile j spans lat [j*T, (j+1)*T]; its first row is the northern edge, at global row -(j+1)*px.
        tj = -np.floor_divide(rows, px) - 1
        out = np.full(rows.shape, np.nan)
        for key in set(zip(ti.tolist(), tj.tolist())):
            tile = self.tiles.get(key)
            if tile is None:
                continue
            sel = (ti == key[0]) & (tj == key[1])
            r = rows[sel] + (key[1] + 1) * px
            c = cols[sel] - key[0] * px
            out[sel] = tile[r, c]
        return out


class NationalElevation:
    """The 5 m LiDAR DEM where it covers the whole road, else DEM-S (see the module docstring)."""

    def __init__(self, cache_dir: Path, region_epsg: int, *, fine: Ga5mDem | None = None, coarse=None):
        from mapcreator.providers.ga.dem_s import ATTRIBUTION as DEM_S
        from mapcreator.providers.ga.dem_s import GaDemS

        self.fine = fine or Ga5mDem(cache_dir, region_epsg)
        self.coarse = coarse if coarse is not None else GaDemS(cache_dir, region_epsg)
        self.decisions = cache_dir / "ga_dem_5m" / "decisions"
        self.use_fine = False
        self.coverage = 0.0
        self.resolution_m = getattr(self.coarse, "resolution_m", 30.0)
        self.attributions = (DEM_S,)

    def load_corridor(self, line_xy, buffer_m: float) -> int:
        line_xy = np.asarray(line_xy, dtype=float)
        tiles = self.coarse.load_corridor(line_xy, buffer_m)
        key = hashlib.sha256(np.round(line_xy, 2).tobytes()).hexdigest()[:24]
        decided = cache.read_json(self.decisions / f"{key}.json")
        try:
            if decided is None or decided["use_fine"]:
                tiles += self.fine.load_corridor(line_xy, buffer_m)
        except Exception as e:  # noqa: BLE001 - the service is a bonus; DEM-S always works
            if decided is not None and decided["use_fine"]:
                raise  # an earlier stage built on the 5 m DEM: mixing models now would be worse than failing
            log.warning("GA 5 m DEM unavailable (%s: %s); using DEM-S", type(e).__name__, e)
            decided = {"use_fine": False, "coverage": 0.0}
        if decided is None:
            heights = self.fine.sample(line_xy[:, 0], line_xy[:, 1])
            coverage = float(np.isfinite(heights).mean()) if len(heights) else 0.0
            decided = {"use_fine": coverage >= MIN_COVERAGE, "coverage": round(coverage, 4)}
            cache.write_json(self.decisions / f"{key}.json", decided)
        self.use_fine, self.coverage = bool(decided["use_fine"]), float(decided["coverage"])
        if self.use_fine:
            self.resolution_m = self.fine.resolution_m
            self.attributions = (ATTRIBUTION, self.attributions[-1])  # DEM-S still fills any holes
        return tiles

    def sample(self, x, y) -> np.ndarray:
        coarse = self.coarse.sample(x, y)
        if not self.use_fine:
            return coarse
        fine = self.fine.sample(x, y)
        return np.where(np.isfinite(fine), fine, coarse)
