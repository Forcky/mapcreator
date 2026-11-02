"""GA's 5 m LiDAR DEM over WCS, and the national fallback's choice between it and DEM-S."""

import numpy as np
import pytest
from pyproj import Transformer

from mapcreator.providers.ga import dem_5m
from mapcreator.providers.ga.dem_5m import Ga5mDem, NationalElevation

EPSG = 28355
TO_LL = Transformer.from_crs(EPSG, dem_5m.CRS, always_xy=True)
TO_XY = Transformer.from_crs(dem_5m.CRS, EPSG, always_xy=True)


def plane(lon, lat):
    return 100.0 + 1000.0 * (lon - 147.0) + 500.0 * (lat + 42.0)


def fake_service(no_lidar_east_of: float | None = None, calls: list | None = None):
    """Tiles of a plane (exact under bilinear sampling), with NaN where there's 'no LiDAR'."""

    def fetch(bbox, size):
        if calls is not None:
            calls.append(bbox)
        w, s, e, n = bbox
        res = (e - w) / size
        lon = w + (np.arange(size) + 0.5) * res
        lat = n - (np.arange(size) + 0.5) * res  # rows north to south
        lo, la = np.meshgrid(lon, lat)
        a = plane(lo, la).astype(np.float32)
        if no_lidar_east_of is not None:
            a[lo > no_lidar_east_of] = np.nan
        return a

    return fetch


def road(lon0, lat0, lon1, lat1, n=400):
    x0, y0 = TO_XY.transform(lon0, lat0)
    x1, y1 = TO_XY.transform(lon1, lat1)
    return np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])


def test_tiles_are_stitched_and_sampled_exactly(tmp_path):
    calls = []
    dem = Ga5mDem(tmp_path, EPSG, fetch=fake_service(calls=calls))
    xy = road(147.015, -42.815, 147.065, -42.785)  # crosses several 0.02 degree tiles both ways
    loaded = dem.load_corridor(xy, 300.0)
    assert loaded == len(calls) > 4
    lon, lat = TO_LL.transform(xy[:, 0], xy[:, 1])
    assert dem.sample(xy[:, 0], xy[:, 1]) == pytest.approx(plane(lon, lat), abs=1e-3)
    # Any array shape, as the track stage samples a grid for its preview image.
    grid_x, grid_y = np.meshgrid(xy[::40, 0], xy[::40, 1])
    glon, glat = TO_LL.transform(grid_x, grid_y)
    got = dem.sample(grid_x, grid_y)
    assert got.shape == grid_x.shape
    assert got[np.isfinite(got)] == pytest.approx(plane(glon, glat)[np.isfinite(got)], abs=1e-3)
    # A second build reads the cached tiles without asking the service.
    again = Ga5mDem(tmp_path, EPSG, fetch=lambda *a: pytest.fail("fetched a cached tile"))
    again.load_corridor(xy, 300.0)
    assert again.sample(xy[:, 0], xy[:, 1]) == pytest.approx(plane(lon, lat), abs=1e-3)


class FakeCoarse:
    resolution_m = 30.0

    def load_corridor(self, line_xy, buffer_m):
        return 1

    def sample(self, x, y):
        return np.full(np.shape(x), 50.0)


def national(tmp_path, fetch):
    return NationalElevation(tmp_path, EPSG, fine=Ga5mDem(tmp_path, EPSG, fetch=fetch), coarse=FakeCoarse())


def test_a_road_fully_under_lidar_uses_the_5m_dem(tmp_path):
    dem = national(tmp_path, fake_service())
    xy = road(147.015, -42.815, 147.035, -42.805)
    dem.load_corridor(xy, 300.0)
    assert dem.use_fine and dem.resolution_m == 5.0 and dem.coverage == 1.0
    assert dem_5m.ATTRIBUTION in dem.attributions
    lon, lat = TO_LL.transform(xy[:, 0], xy[:, 1])
    assert dem.sample(xy[:, 0], xy[:, 1]) == pytest.approx(plane(lon, lat), abs=1e-3)


def test_a_road_leaving_the_lidar_uses_dem_s_throughout(tmp_path):
    dem = national(tmp_path, fake_service(no_lidar_east_of=147.025))
    xy = road(147.015, -42.815, 147.035, -42.805)
    dem.load_corridor(xy, 300.0)
    assert not dem.use_fine and dem.resolution_m == 30.0 and 0.3 < dem.coverage < 0.7
    assert dem_5m.ATTRIBUTION not in dem.attributions
    assert (dem.sample(xy[:, 0], xy[:, 1]) == 50.0).all()  # no step where the LiDAR ends


def test_every_stage_of_a_build_makes_the_same_choice(tmp_path):
    xy = road(147.015, -42.815, 147.035, -42.805)
    national(tmp_path, fake_service()).load_corridor(xy, 300.0)  # the profile stage decides
    # A later stage, with the service now down and a wider corridor needing new tiles: it must not
    # quietly switch to DEM-S for the rest of the build.
    def down(*a):
        raise OSError("service unavailable")

    with pytest.raises(OSError):
        national(tmp_path, down).load_corridor(xy, 2000.0)
    later = national(tmp_path, down)
    later.load_corridor(xy, 300.0)  # the same corridor is all cached: fine
    assert later.use_fine


def test_the_service_being_down_means_dem_s_not_a_failed_build(tmp_path):
    def down(*a):
        raise OSError("service unavailable")

    dem = national(tmp_path, down)
    xy = road(147.015, -42.815, 147.035, -42.805)
    dem.load_corridor(xy, 300.0)
    assert not dem.use_fine and (dem.sample(xy[:, 0], xy[:, 1]) == 50.0).all()
