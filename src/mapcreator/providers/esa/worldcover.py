"""ESA WorldCover 10 m land cover (2021, v200), CC BY 4.0. Used for tree placement anywhere.

Global 3 x 3 degree cloud-optimised GeoTIFFs on public cloud storage, in WGS84
with 1024 px tiles. A corridor needs a handful of tiles, read on demand and
cached like the DEM. Class 10 is tree cover, 20 shrubland, 30 grassland,
40 cropland, 50 built-up, 60 bare, 80 water.

It gives presence, not individual trees, so trees are scattered at a
plausible density inside tree-covered cells. Where an individual-crown canopy
model exists (Tasmania's CHM) that is used instead.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString

from mapcreator.providers.base import LONLAT_CRS, Attribution
from mapcreator.providers.raster import BlockRaster

ATTRIBUTION = Attribution(
    dataset="ESA WorldCover 10 m 2021 v200 (tree cover)",
    holder="© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by "
           "ESA WorldCover consortium",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
)
URL = ("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
       "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
TREE, SHRUB, GRASS, CROP, BUILT, WATER = 10, 20, 30, 40, 50, 80


def tile_name(lon: float, lat: float) -> str:
    """Tiles are named by their south-west corner on a 3 degree grid, e.g. S45E147."""
    la = math.floor(lat / 3) * 3
    lo = math.floor(lon / 3) * 3
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


class WorldCover:
    def __init__(self, cache_dir: Path, region_epsg: int, *, url_template: str = URL):
        self.cache_dir = cache_dir
        self.region_epsg = region_epsg
        self.url_template = url_template
        self.to_lonlat = Transformer.from_crs(region_epsg, LONLAT_CRS, always_xy=True)
        self.tiles: dict[str, BlockRaster] = {}

    def _tile(self, name: str) -> BlockRaster:
        if name not in self.tiles:
            self.tiles[name] = BlockRaster(self.url_template.format(tile=name), self.cache_dir,
                                           region_epsg=self.region_epsg, categorical=True, cache_name="worldcover")
        return self.tiles[name]

    def load_corridor(self, line_xy: np.ndarray, buffer_m: float) -> None:
        poly = LineString(line_xy).buffer(buffer_m)
        minx, miny, maxx, maxy = poly.bounds
        lons, lats = self.to_lonlat.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
        names = {tile_name(lo, la) for lo in (min(lons), max(lons)) for la in (min(lats), max(lats))}
        for name in names:
            self._tile(name).load_corridor(line_xy, buffer_m)

    def classes(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = np.asarray(x, float), np.asarray(y, float)
        lon, lat = self.to_lonlat.transform(x, y)
        names = np.array([tile_name(a, b) for a, b in zip(np.atleast_1d(lon), np.atleast_1d(lat))])
        out = np.zeros(x.shape, dtype=np.int32)
        for name in np.unique(names):
            m = names == name
            out[m] = self._tile(name).sample_nearest(x[m], y[m])
        return out
