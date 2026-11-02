"""The tile cache must sample exactly like reading the whole raster, across tile edges."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from mapcreator.providers.tas_list.dem import BLOCK, TasDem

X0, Y0, RES = 500000.0, 5300000.0, 2.0


@pytest.fixture()
def plane_dem(tmp_path):
    """A 1300 x 1100 px plane z = 0.01 x' + 0.02 y' + 5, with a nodata hole."""
    h, w = 1100, 1300
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    x = X0 + (cols + 0.5) * RES
    y = Y0 - (rows + 0.5) * RES
    z = (0.01 * (x - X0) + 0.02 * (Y0 - y) + 5).astype(np.float32)
    z[50:60, 50:60] = -9999
    path = tmp_path / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32", crs="EPSG:28355",
                       transform=from_origin(X0, Y0, RES, RES), nodata=-9999, tiled=True, blockxsize=512,
                       blockysize=512) as ds:
        ds.write(z, 1)
    return path


def test_sampling_matches_the_plane_across_tile_edges(plane_dem, tmp_path):
    dem = TasDem(str(plane_dem), tmp_path / "cache", workers=2)
    # A line crossing the tile boundary at column 512 and row 512.
    line = np.array([[X0 + 900, Y0 - 900], [X0 + 1200, Y0 - 1200]])
    dem.load_corridor(line, 50)
    xs = np.linspace(X0 + 900, X0 + 1200, 301)
    ys = np.linspace(Y0 - 900, Y0 - 1200, 301)
    expected = 0.01 * (xs - X0) + 0.02 * (Y0 - ys) + 5
    assert dem.sample(xs, ys) == pytest.approx(expected, abs=1e-4)
    assert BLOCK * RES < 1200  # the test really crosses a boundary


def test_only_tiles_near_the_corridor_are_loaded_and_cached(plane_dem, tmp_path):
    dem = TasDem(str(plane_dem), tmp_path / "cache", workers=2)
    dem.load_corridor(np.array([[X0 + 100, Y0 - 100], [X0 + 200, Y0 - 120]]), 20)
    assert set(dem.blocks) == {(0, 0)}
    assert len(list(dem.cache.glob("*.npy"))) == 1
    # Outside loaded tiles is NaN, not a silent zero.
    assert np.isnan(dem.sample([X0 + 2400], [Y0 - 2000]))[0]


def test_nodata_reads_as_nan(plane_dem, tmp_path):
    dem = TasDem(str(plane_dem), tmp_path / "cache", workers=1)
    dem.load_corridor(LineString([(X0, Y0 - 1), (X0 + 300, Y0 - 1)]).coords, 200)
    assert np.isnan(dem.sample([X0 + 110], [Y0 - 110]))[0]
