"""Tasmania 2 m DEM (theLIST, CC BY 3.0 AU).

The statewide DEM is a single ~95 GB cloud-optimised GeoTIFF (GDA94 / MGA55,
heights in AHD) with 512 x 512 pixel internal tiles: about 1 km² and ~1 MB
each. `BlockRaster` reads only the tiles a corridor touches. Point the source
at a local mirror for production builds (`tas_dem_source` in the config).
"""

from __future__ import annotations

from pathlib import Path

from mapcreator.providers.base import TAS_DEM
from mapcreator.providers.raster import BlockRaster

ATTRIBUTION = TAS_DEM
EPSG = 28355
BLOCK = 512


class TasDem(BlockRaster):
    def __init__(self, source: str, cache_dir: Path, *, workers: int = 8):
        super().__init__(source, cache_dir, region_epsg=EPSG, block=BLOCK, workers=workers, cache_name="tas_dem")
