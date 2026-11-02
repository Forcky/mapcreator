"""Geoscience Australia 1 second SRTM DEM-S (~30 m, CC BY 4.0): the Australia-wide elevation fallback.

DEM-S is the smoothed version of GA's SRTM-derived DEM with vegetation
offsets removed. It's one 38 GB cloud-optimised GeoTIFF (WGS84, 512 px
tiles) on Digital Earth Australia's public bucket, so it's read tile by tile
like the Tasmanian DEM.

At ~30 m it can't see a 6 m road's camber, a creek bridge or a cutting batter.
Tracks built on it get a gentler profile filter, no DEM camber and no
DEM-placed guard rails. Where a state's LiDAR DEM is available it should
replace this.
"""

from __future__ import annotations

from pathlib import Path

from mapcreator.providers.base import Attribution
from mapcreator.providers.raster import BlockRaster

ATTRIBUTION = Attribution(
    dataset="1 second SRTM Digital Elevation Model (DEM-S)",
    holder="© Commonwealth of Australia (Geoscience Australia)",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
)
URL = ("/vsicurl/https://dea-public-data.s3.ap-southeast-2.amazonaws.com/projects/elevation/"
       "ga_srtm_dem1sv1_0/dems1sv1_0.tif")


class GaDemS(BlockRaster):
    resolution_m = 30.0

    def __init__(self, cache_dir: Path, region_epsg: int, *, source: str = URL):
        super().__init__(source, cache_dir, region_epsg=region_epsg, cache_name="ga_dem_s")
