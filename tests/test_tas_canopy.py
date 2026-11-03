"""Crowns from the canopy model: one per real crown top, the same across tile edges (#9)."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from mapcreator.providers.tas_list.canopy import TasCanopy, merge_touching

X0, Y0, RES, TILE = 500000.0, 5300000.0, 2.0, 16


def _canopy(tmp_path, heights: np.ndarray) -> TasCanopy:
    """A whole-metre canopy model in 16 px tiles, with every tile loaded."""
    h, w = heights.shape
    path = tmp_path / "chm.tif"
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="int16", crs="EPSG:28355",
                       transform=from_origin(X0, Y0, RES, RES), nodata=-32767, tiled=True, blockxsize=TILE,
                       blockysize=TILE) as ds:
        ds.write(heights.astype(np.int16), 1)
    chm = TasCanopy(str(path), tmp_path / "cache")
    chm.load_corridor(np.array([[X0, Y0], [X0 + w * RES, Y0 - h * RES]]), 200)
    assert len(chm.blocks) == (h // TILE) * (w // TILE)
    return chm


def _pixel(xy: np.ndarray) -> np.ndarray:
    """Crown positions back to (column, row)."""
    return np.column_stack([(xy[:, 0] - X0) / RES - 0.5, (Y0 - xy[:, 1]) / RES - 0.5])


def test_a_flat_crown_top_is_one_crown_at_its_centre(tmp_path):
    a = np.zeros((48, 48))
    a[19:23, 19:23] = 9
    a[20:22, 20:22] = 12  # a 2 x 2 plateau: four equal local maxima
    xy, h = _canopy(tmp_path, a).crowns()
    assert len(xy) == 1 and h[0] == 12
    assert _pixel(xy)[0] == pytest.approx([20.5, 20.5])


def test_a_crown_over_a_tile_edge_leaves_no_false_peak(tmp_path):
    a = np.zeros((48, 48))
    # The top (11 m) is at column 17, in the second tile; the crown slopes down across the edge at 15/16.
    # Filtered alone, the first tile would see its column 15 (9 m) as a peak.
    a[30, 12:20] = [6, 7, 8, 9, 10, 11, 10, 9]
    xy, h = _canopy(tmp_path, a).crowns()
    assert len(xy) == 1 and h[0] == 11
    assert _pixel(xy)[0] == pytest.approx([17, 30])


def test_separate_trees_stay_separate_and_low_ones_are_not_trees(tmp_path):
    a = np.zeros((48, 48))
    a[10, 10] = 15
    a[10, 16] = 18  # 12 m away: another tree
    a[40, 40] = 6
    a[40, 30] = 4   # not above MIN_TREE_M
    _, h = _canopy(tmp_path, a).crowns()
    assert sorted(h) == [6, 15, 18]


def test_merge_touching_joins_diagonal_neighbours_only():
    xy = np.array([[0, 0], [2, 2], [4, 4], [20, 0]], float)  # a diagonal run of three pixels, and one apart
    c, top = merge_touching(xy, np.array([10, 10, 10, 7], float), 2.0)
    by_height = np.argsort(top)
    assert len(c) == 2
    assert c[by_height[1]] == pytest.approx([2, 2]) and top[by_height[1]] == 10
    assert c[by_height[0]] == pytest.approx([20, 0]) and top[by_height[0]] == 7
