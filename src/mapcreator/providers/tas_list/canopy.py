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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

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
        """Crown centres (x, y in MGA55) and heights from every loaded tile: local maxima above 4 m.

        Two things raw maxima get wrong, both measured on test roads (mapcreator#9):
        - Each tile is filtered with a border borrowed from its loaded neighbours. Filtered alone, a tile
          shows a false peak on its edge wherever a crown's top is in the next tile (about 2% of crowns).
        - The model is in whole metres, so a crown's top is often a plateau of equal pixels, every one a
          local maximum: about half the raw maxima were such repeats. Maxima that touch become one crown.
        """
        xs, ys, hs = [], [], []
        b = self.block
        for (bx, by) in self.blocks:
            a = self._padded(bx, by, CROWN_WINDOW_PX)
            peaks = (a == maximum_filter(a, size=CROWN_WINDOW_PX)) & (a > MIN_TREE_M)
            p = CROWN_WINDOW_PX
            r, c = np.nonzero(peaks[p:p + b, p:p + b])  # this tile's own pixels only
            xs.append(self.x0 + (bx * b + c + 0.5) * self.res_x)
            ys.append(self.y0 + (by * b + r + 0.5) * self.res_y)
            hs.append(a[r + p, c + p])
        if not xs:
            return np.zeros((0, 2)), np.zeros(0)
        xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        return merge_touching(xy, np.concatenate(hs).astype(float), abs(self.res_x))

    def _padded(self, bx: int, by: int, pad: int) -> np.ndarray:
        """Tile (bx, by) with `pad` pixels on every side from its loaded neighbours (0 where there's none)."""
        b = self.block
        out = np.zeros((b + 2 * pad, b + 2 * pad), dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                arr = self.blocks.get((bx + dx, by + dy))
                if arr is None:
                    continue
                # The neighbour's pixels in the padded tile's frame, clipped to the frame.
                r0, c0 = dy * b + pad, dx * b + pad
                rs, cs = max(r0, 0), max(c0, 0)
                re, ce = min(r0 + b, b + 2 * pad), min(c0 + b, b + 2 * pad)
                if rs < re and cs < ce:
                    out[rs:re, cs:ce] = np.nan_to_num(arr[rs - r0:re - r0, cs - c0:ce - c0], nan=0.0)
        return out


def merge_touching(xy: np.ndarray, height: np.ndarray, pixel_m: float) -> tuple[np.ndarray, np.ndarray]:
    """One crown per group of maxima on neighbouring pixels (diagonals too): at their centroid, the group's height.

    Touching maxima always have the same height (each is the highest in the other's window), so this only
    removes repeats of one crown top, never a separate tree.
    """
    if len(xy) < 2:
        return xy, height
    pairs = cKDTree(xy).query_pairs(r=pixel_m * 2 ** 0.5 * 1.01, output_type="ndarray")
    if not len(pairs):
        return xy, height
    n = len(xy)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    k, label = connected_components(graph, directed=False)
    count = np.bincount(label, minlength=k)
    centre = np.column_stack([np.bincount(label, xy[:, 0], k), np.bincount(label, xy[:, 1], k)]) / count[:, None]
    top = np.full(k, -np.inf)
    np.maximum.at(top, label, height)
    return centre, top
