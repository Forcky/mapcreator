"""NVIS 7.0 Major Vegetation Subgroups (DCCEEW, CC BY 4.0): what kind of vegetation each 100 m cell holds.

The National Vegetation Information System's extant subgroups raster covers Australia at 100 m (1 ha) cells, one
dominant subgroup per cell, in GDA2020 Australian Albers (EPSG:9473). It's published as a zipped file geodatabase,
which can't be range-read, so mirror it once as a COG and point `nvis_source` at that file. A GDAL with the
OpenFileGDB driver does it: `gdal_translate -of COG -co COMPRESS=DEFLATE -co BLOCKSIZE=512
-co OVERVIEW_RESAMPLING=NEAREST "OpenFileGDB:<gdb>:NVIS7_0_AUST_EXT_MVS_ALB" nvis7_0_mvs_9473.tif`. Only the
subgroup numbers are read here; `core.species` turns them into tree families.

100 m cells are coarse against a 250 m tree corridor, and the publisher warns that its non-vegetation and
non-native classes "should not be used for analyses", so NVIS only types trees, never removes them.
"""

from __future__ import annotations

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

from mapcreator.providers.base import Attribution

ATTRIBUTION = Attribution(
    dataset='"Australia - Present Major Vegetation Groups and Subgroups - NVIS Version 7.0"',
    holder="© Commonwealth of Australia (Department of Climate Change, Energy, the Environment and Water) 2024",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
)
EPSG = 9473
NO_DATA = 99  # "Unknown/No data"


class Nvis:
    """The subgroup at points in a region's CRS, read from a local COG for one corridor."""

    def __init__(self, source: str, region_epsg: int):
        self.source = source
        self._to_nvis = Transformer.from_crs(region_epsg, EPSG, always_xy=True)
        self._cells = None

    def load_corridor(self, centre_xy: np.ndarray, reach_m: float) -> None:
        """Read the cells under the corridor's bounding box, plus one cell all round."""
        xy = np.asarray(centre_xy, dtype=float)
        x0, y0 = xy.min(axis=0) - reach_m
        x1, y1 = xy.max(axis=0) + reach_m
        w, s, e, n = self._to_nvis.transform_bounds(x0, y0, x1, y1, densify_pts=21)
        with rasterio.open(self.source) as src:
            res = abs(src.res[0])
            window = from_bounds(w - res, s - res, e + res, n + res, src.transform).round_offsets().round_lengths()
            self._cells = src.read(1, window=window, boundless=True, fill_value=NO_DATA)
            self._transform = src.window_transform(window)

    @property
    def cells(self) -> np.ndarray:
        """The loaded cells (subgroup numbers), rows north to south."""
        if self._cells is None:
            raise RuntimeError("load_corridor first")
        return self._cells

    @property
    def cell_m(self) -> float:
        return abs(self._transform.a)

    def sample(self, grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """`grid` (shaped like `cells`) at each point; NO_DATA outside the loaded cells."""
        ex, ny = self._to_nvis.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        col, row = ~self._transform * (ex, ny)
        col, row = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
        h, w = grid.shape
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        out = np.full(len(col), NO_DATA, dtype=np.int64)
        out[inside] = grid[row[inside], col[inside]]
        return out

    def subgroups(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The subgroup number (NVIS MVS 1-99) at each point; NO_DATA outside the loaded cells."""
        return self.sample(self.cells, x, y)
