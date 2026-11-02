"""Geoscience Australia's 5 m LiDAR DEM (CC BY 4.0), and how the national fallback picks its elevation.

"Digital Elevation Model (DEM) of Australia derived from LiDAR 5 Metre Grid" covers about 245,000 km²: the
populated coastal zone, the Murray-Darling floodplains, and towns large and small (236 surveys, 2001-2015).
It's read through GA's WCS service in small tiles (GDA94 lon/lat), cached on disk like the other rasters.
Where there's no LiDAR the service answers 0.0 rather than no-data, so an exact 0.0 counts as no data.

`NationalElevation` is what the national regions use. It has a ranked list of fine sources (finest first:
state LiDAR where a state publishes it, then this 5 m DEM) and DEM-S (~30 m) behind them all. It decides
once per road: the first fine source that covers (nearly) all of the centreline is used, with DEM-S
filling any small holes, and the build gets the fine-DEM treatment (DEM camber, guard rails). If none
covers the road, the whole build uses DEM-S, so a road never steps between two models where coverage ends.
The decision is kept in the cache, keyed by the centreline, so every stage of a build agrees even if a
service fails halfway through.
"""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

import numpy as np

from mapcreator.providers import cache
from mapcreator.providers.base import Attribution
from mapcreator.providers.tiled import Fetcher, TiledDem

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
# Share of the centreline a fine DEM must cover for a build to use it.
MIN_COVERAGE = 0.98


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


class Ga5mDem(TiledDem):
    """GA's 5 m DEM in 0.02 degree tiles, sampled in a region's projected CRS."""

    def __init__(self, cache_dir: Path, region_epsg: int, *, fetch: Fetcher | None = None):
        super().__init__(cache_dir, region_epsg, name="ga_dem_5m", crs=CRS, cell=RES_DEG, tile=TILE_DEG,
                         fetch=fetch or wcs_fetch, resolution_m=5.0, attribution=ATTRIBUTION)


def fine_sources(cache_dir: Path, region_epsg: int) -> list[TiledDem]:
    """The DEMs to try ahead of DEM-S for a road outside Tasmania, finest first. Each one only answers
    roads inside its own extent.

    Two more are written, tested against the live services, and left out until their licences are settled:
    - `providers.qld.dem_1m.QldDem1m`: Queensland's 1 m LiDAR. The service says CC BY-SA, the surveys CC BY
      4.0; share-alike would carry into every track.
    - `providers.nsw.dem_5m.NswDem5m`: NSW's statewide 5 m (and the ACT), which fills GA's holes in Canberra,
      the Blue Mountains and the Putty Road. CC BY 3.0 AU, but with a term about hosting material "on any
      external cloud platforms" unchanged. It would go after GA's 5 m, and QLD's 1 m first of all.
    """
    from mapcreator.providers.vic.dem_10m import VicDem10m

    return [Ga5mDem(cache_dir, region_epsg), VicDem10m(cache_dir, region_epsg)]


class NationalElevation:
    """The finest DEM that covers the whole road, else DEM-S (see the module docstring)."""

    def __init__(self, cache_dir: Path, region_epsg: int, *, fines: list | None = None, fine=None, coarse=None):
        from mapcreator.providers.ga.dem_s import ATTRIBUTION as DEM_S
        from mapcreator.providers.ga.dem_s import GaDemS

        if fines is None:
            fines = [fine] if fine is not None else fine_sources(cache_dir, region_epsg)
        self.fines = fines
        self.coarse = coarse if coarse is not None else GaDemS(cache_dir, region_epsg)
        self.decisions = cache_dir / "ga_dem_5m" / "decisions"
        from pyproj import Transformer

        from mapcreator.providers.base import LONLAT_CRS

        self._to_lonlat = Transformer.from_crs(region_epsg, LONLAT_CRS, always_xy=True)
        self.chosen = None
        self.coverage = 0.0  # the chosen source's, or the best any source reached
        self.resolution_m = getattr(self.coarse, "resolution_m", 30.0)
        self.attributions = (DEM_S,)

    @property
    def use_fine(self) -> bool:
        return self.chosen is not None

    def _by_name(self, name: str | None):
        return next((f for f in self.fines if f.name == name), None)

    def load_corridor(self, line_xy, buffer_m: float) -> int:
        line_xy = np.asarray(line_xy, dtype=float)
        tiles = self.coarse.load_corridor(line_xy, buffer_m)
        # Keyed by the road and by the list of sources, so adding a source makes every road decide again
        # (a decision made before Victoria's DEM existed would otherwise keep the Dandenongs on DEM-S).
        sources = ",".join(f.name for f in self.fines).encode()
        key = hashlib.sha256(np.round(line_xy, 2).tobytes() + b"|" + sources).hexdigest()[:24]
        decided = cache.read_json(self.decisions / f"{key}.json")
        if decided is not None:
            chosen = self._by_name(decided.get("source"))
            if chosen is not None:
                # An earlier stage built on this source: failing beats mixing models within one build.
                tiles += chosen.load_corridor(line_xy, buffer_m)
            coverage = float(decided.get("coverage", 0.0))
        else:
            chosen, coverage, tried = None, 0.0, {}
            lon, lat = self._to_lonlat.transform(line_xy[:, 0], line_xy[:, 1])
            for fine in self.fines:  # finest first: the first that covers the road wins
                if not fine.covers(float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))):
                    continue
                try:
                    tiles += fine.load_corridor(line_xy, buffer_m)
                except Exception as e:  # noqa: BLE001 - a fine source is a bonus; DEM-S always works
                    log.warning("%s unavailable (%s: %s); trying the next source", fine.name, type(e).__name__, e)
                    tried[fine.name] = None
                    continue
                heights = fine.sample(line_xy[:, 0], line_xy[:, 1])
                share = float(np.isfinite(heights).mean()) if len(heights) else 0.0
                tried[fine.name] = round(share, 4)
                coverage = max(coverage, share)
                if share >= MIN_COVERAGE:
                    chosen, coverage = fine, share
                    break
            decided = {"source": chosen.name if chosen else None, "coverage": round(coverage, 4), "tried": tried}
            cache.write_json(self.decisions / f"{key}.json", decided)

        self.chosen, self.coverage = chosen, coverage
        if chosen is not None:
            self.resolution_m = chosen.resolution_m
            self.attributions = (chosen.attribution, self.attributions[-1])  # DEM-S still fills any holes
        return tiles

    def sample(self, x, y) -> np.ndarray:
        coarse = self.coarse.sample(x, y)
        if self.chosen is None:
            return coarse
        fine = self.chosen.sample(x, y)
        return np.where(np.isfinite(fine), fine, coarse)
