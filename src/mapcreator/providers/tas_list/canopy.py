"""Tasmania Tree Canopy Height model, 2 m (theLIST, CC BY 3.0 AU): individual crowns and heights.

theLIST publishes it as a ~10 GB GeoTIFF inside a 26 GB zip, and the TIFF is
deflate-compressed within the zip, so it can't be range-read remotely like
the DEM. Mirror it once (unzip `CHM_2m/TAS_CHM_2m.tif`) and point
`tas_chm_source` at the file. Without a mirror, tree placement falls back to
ESA WorldCover.

Heights include buildings and other objects above ground, so callers mask
out built-up areas and the road corridor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from mapcreator.providers.base import Attribution
from mapcreator.providers.raster import BlockRaster

ATTRIBUTION = Attribution(
    dataset='"Tree Canopy Height (CHM 2 m)" from theLIST',
    holder="© State of Tasmania",
    licence="CC BY 3.0 AU",
    licence_url="https://creativecommons.org/licenses/by/3.0/au/",
)
EPSG = 28355
MIN_TREE_M = 4.0
CROWN_WINDOW_PX = 4  # 8 m: one crown peak per window


class TasCanopy(BlockRaster):
    def __init__(self, source: str, cache_dir: Path):
        super().__init__(source, cache_dir, region_epsg=EPSG, cache_name="tas_chm")

    def crowns(self) -> tuple[np.ndarray, np.ndarray]:
        """Crown centres (x, y in MGA55) and heights from every loaded tile: local maxima above 4 m."""
        xs, ys, hs = [], [], []
        b = self.block
        for (bx, by), arr in self.blocks.items():
            a = np.nan_to_num(arr, nan=0.0)
            peaks = (a == maximum_filter(a, size=CROWN_WINDOW_PX)) & (a > MIN_TREE_M)
            r, c = np.nonzero(peaks)
            xs.append(self.x0 + (bx * b + c + 0.5) * self.res_x)
            ys.append(self.y0 + (by * b + r + 0.5) * self.res_y)
            hs.append(a[r, c])
        if not xs:
            return np.zeros((0, 2)), np.zeros(0)
        return np.column_stack([np.concatenate(xs), np.concatenate(ys)]), np.concatenate(hs).astype(float)
