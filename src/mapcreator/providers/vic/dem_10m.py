"""Victoria's 10 m DEM (Vicmap Elevation DEM 10m, CC BY 4.0), for Victorian roads that GA's 5 m LiDAR misses.

Victoria's 1 m LiDAR DEM is licensed for a fee, but the 10 m statewide DEM is open, and since a 2016 update
it includes LiDAR where Victoria had it. Stated accuracy is 5 m vertical or better. That's coarser than a
road's camber, so a build on it gets the coarse-DEM treatment, but with a far shorter profile filter than
DEM-S's 30 m allows.

It's served by Vicmap as a Service as ArcGIS elevation tiles: LERC-compressed float32 in Web Mercator, read
here at zoom 15 (about 3.8 m on the ground in Victoria, so finer than the data). Each tile is 257 x 257
samples on the tile's edges, the last row and column repeating the next tile's first, so 256 x 256 of them
are kept and treated as corner samples. Where there's no data (outside Victoria, the sea) the tiles hold 0.0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mapcreator.providers.base import Attribution
from mapcreator.providers.tiled import Fetcher, TiledDem

ATTRIBUTION = Attribution(
    dataset="Vicmap Elevation DEM 10m",
    holder="Copyright (c) The State of Victoria, Department of Energy, Environment and Climate Action",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
)
TILE_URL = ("https://tiles-ap1.arcgis.com/P744lA0wf4LlBZ84/arcgis/rest/services/Vicmap_10m_DEM/ImageServer"
            "/tile/{z}/{y}/{x}")
ZOOM = 15
HALF_WORLD = 20037508.342787  # Web Mercator's half width, metres
TILE_M = 2 * HALF_WORLD / 2 ** ZOOM
TILE_PX = 256
# The service's extent, from its own metadata (Web Mercator, converted to lon/lat).
EXTENT_LONLAT = (140.15, -39.20, 150.23, -33.84)


def decode(content: bytes) -> np.ndarray:
    """A 257 x 257 LERC tile as 256 x 256 float32, rows north to south, with 0.0 (no data) as NaN."""
    from rasterio.io import MemoryFile

    with MemoryFile(content) as mem, mem.open(driver="MRF", DATATYPE="Float32") as ds:
        a = ds.read(1)[:TILE_PX, :TILE_PX].astype(np.float32)
    a[a == 0.0] = np.nan
    a[a < -1000] = np.nan
    return a


def tile_fetch(bbox: tuple[float, float, float, float], size: int, *, user_agent: str = "mapcreator") -> np.ndarray:
    """The zoom-15 tile whose south-west corner is the box's (TiledDem asks on the same grid)."""
    import httpx

    i, j = round(bbox[0] / TILE_M), round(bbox[1] / TILE_M)
    x, y = i + 2 ** (ZOOM - 1), 2 ** (ZOOM - 1) - (j + 1)
    r = httpx.get(TILE_URL.format(z=ZOOM, y=y, x=x), timeout=60, headers={"User-Agent": user_agent})
    if r.status_code == 404:
        return np.full((TILE_PX, TILE_PX), np.nan, dtype=np.float32)
    r.raise_for_status()
    return decode(r.content)


class VicDem10m(TiledDem):
    def __init__(self, cache_dir: Path, region_epsg: int, *, fetch: Fetcher | None = None):
        super().__init__(cache_dir, region_epsg, name="vic_dem_10m", crs=3857, cell=TILE_M / TILE_PX, tile=TILE_M,
                         fetch=fetch or tile_fetch, resolution_m=10.0, attribution=ATTRIBUTION, corner=True,
                         extent_lonlat=EXTENT_LONLAT)
