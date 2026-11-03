import numpy as np
import pytest

from mapcreator.core import centreline as cl
from mapcreator.core import profile as pf


def _arc(radius: float, sweep_deg: float, node_spacing: float) -> np.ndarray:
    n = max(int(np.radians(sweep_deg) * radius / node_spacing), 2)
    t = np.linspace(0, np.radians(sweep_deg), n + 1)
    return np.column_stack([radius * np.sin(t), radius * (1 - np.cos(t))])


def test_arc_keeps_its_radius_and_is_sampled_every_metre():
    line = cl.smooth(_arc(60.0, 120.0, 12.0))
    mid = slice(len(line.s) // 4, 3 * len(line.s) // 4)
    assert np.median(line.curvature[mid]) == pytest.approx(1 / 60.0, rel=0.08)
    assert np.allclose(np.diff(line.s), 1.0)
    assert line.max_deviation_m < 1.5


def test_a_junction_corner_is_rounded_to_a_drivable_turn():
    # Richmond, Nov 2026: Bridge Street into Wellington Street, 108 degrees at one node, nodes 5-7 m either side.
    a = np.array([[-60.0, 0.0], [-5.3, 0.0], [0.0, 0.0]])
    turn = np.radians(108.0)
    out = np.array([np.cos(turn), np.sin(turn)])
    line = cl.smooth(np.vstack([a, out * 7.0, out * 15.7, out * 60.0]))
    radius = 1 / np.abs(line.curvature).max()
    assert radius == pytest.approx(cl.CORNER_RADIUS_M, rel=0.25)
    # It passes r (1 - cos(turn / 2)) inside the two roads, about 3.3 m here, and no more.
    assert line.max_deviation_m == pytest.approx(cl.CORNER_RADIUS_M * (1 - np.cos(turn / 2)), abs=0.5)


def test_fillet_keeps_the_source_segments_and_leaves_reversals_alone():
    corner = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [100.0, 50.0]])
    pts, src = cl.fillet(corner)
    assert len(src) == len(pts) - 1 and set(src) == {0, 1, 2}
    assert (np.diff(src) >= 0).all()
    assert np.allclose(pts[0], corner[0]) and np.allclose(pts[-1], corner[-1])
    # Both corners are 90 degrees with 50 m between them: each gets the full 8 m radius, 8 m either side.
    assert np.allclose(pts[1], [42.0, 0.0]) and len(pts) > 20
    back = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 2.0]])  # 177 degrees: going back the way it came
    pts, src = cl.fillet(back)
    assert np.allclose(pts, back) and list(src) == [0, 1]


def test_gentle_bends_are_not_filleted():
    line = _arc(60.0, 120.0, 12.0)
    pts, _ = cl.fillet(line)
    assert np.allclose(pts, line)


def _grid_dem(cell: float, rough: float, seed: int = 0):
    """A hillside rising 30% to the north, on a grid of `cell` metres, bilinear between cells, with
    `rough` metres of noise per cell: LiDAR-like at 2 m with noise, coarse at 25 m without."""
    rng = np.random.default_rng(seed)
    n = int(1200 / cell) + 2
    xs = np.arange(n) * cell - 100.0
    zz = 0.3 * xs[:, None] + 0.0 * xs[None, :] + rng.normal(0, rough, (n, n))
    from scipy.interpolate import RegularGridInterpolator

    f = RegularGridInterpolator((xs, xs), zz, method="linear", bounds_error=False, fill_value=np.nan)
    return lambda x, y: f(np.column_stack([np.ravel(y), np.ravel(x)]))


def test_lidar_ground_is_fine_and_a_resampled_coarse_model_is_not():
    s = np.arange(0.0, 1000.0)
    xy = np.column_stack([s, np.full(s.size, 500.0)])
    heading = np.zeros(s.size)
    assert pf.fine_stations(s, xy, heading, _grid_dem(2.0, 0.08)).all()
    assert not pf.fine_stations(s, xy, heading, _grid_dem(25.0, 0.0)).any()


def test_a_coarse_stretch_is_capped_and_pinned_while_fine_stretches_keep_their_heights():
    s = np.arange(0.0, 3000.0)
    road = 100 + 0.04 * s  # a steady 4% climb
    bump = np.where((s > 1450) & (s < 1580), 34.0 * np.sin(np.pi * (s - 1450) / 130), 0.0)  # a hillside read
    z_raw = road + bump
    fine = (s < 1200) | (s > 1800)
    prof = pf.mixed_profile(s, z_raw, np.zeros(s.size, bool), fine)
    assert np.abs(prof.grade).max() <= pf.COARSE_MAX_GRADE + 0.01
    far = (s < 1000) | (s > 2000)
    assert np.abs(prof.z[far] - pf.smooth_profile(s, z_raw, np.zeros(s.size, bool)).z[far]).max() < 0.05
    every_fine = pf.mixed_profile(s, road, np.zeros(s.size, bool), np.ones(s.size, bool))
    assert np.allclose(every_fine.z, pf.smooth_profile(s, road, np.zeros(s.size, bool)).z)


def test_straight_line_has_no_curvature():
    line = cl.smooth(np.array([[0, 0], [300, 0], [600, 0.0]]))
    assert np.abs(line.curvature).max() < 1e-4


def test_heading_left_normal():
    n = cl.normals(np.array([0.0, np.pi / 2]))
    assert n[0] == pytest.approx([0, 1])
    assert n[1] == pytest.approx([-1, 0])


def test_profile_smoothing_removes_dem_speckle_but_keeps_hills():
    s = np.arange(0, 2000.0)
    hill = 30 * np.sin(2 * np.pi * s / 1000)
    rng = np.random.default_rng(0)
    p = pf.smooth_profile(s, hill + rng.normal(0, 0.05, s.size), np.zeros(s.size, bool))
    assert np.sqrt(np.mean((p.z - hill) ** 2)) < 0.03
    assert np.gradient(p.z, s).max() == pytest.approx(30 * 2 * np.pi / 1000, rel=0.05)


def test_narrow_dip_under_a_bridge_is_filled_but_a_valley_is_not():
    s = np.arange(0, 600.0)
    z = np.full(s.size, 20.0)
    z[300:306] -= 0.9  # 6 m creek bridge on a bare-earth DEM
    z[100:180] -= 3 * np.sin(np.linspace(0, np.pi, 80))  # 80 m wide real dip
    mask = pf.narrow_dips(s, z)
    assert mask[300:306].all()
    assert not mask[100:180].any()
    p = pf.smooth_profile(s, z, np.zeros(s.size, bool))
    assert abs(p.z[303] - 20.0) < 0.05


def test_nan_runs_are_bridged_linearly():
    s = np.arange(0, 100.0)
    z = s * 0.05
    z[40:50] = np.nan
    filled, mask = pf.fill_runs(s, z, np.zeros(s.size, bool), pad_m=0)
    assert np.allclose(filled, s * 0.05)
    assert mask[40:50].all() and mask.sum() == 10


def test_cross_slope_sign_and_clamp():
    s = np.arange(0, 500.0)
    cs = pf.cross_slope(s, np.full(s.size, 10.3), np.full(s.size, 10.0), np.full(s.size, 3.0))
    assert cs.mean() == pytest.approx(0.05, abs=1e-3)  # left edge 0.3 m higher over 6 m
    steep = pf.cross_slope(s, np.full(s.size, 12.0), np.full(s.size, 10.0), np.full(s.size, 3.0))
    assert steep.max() == pytest.approx(0.08)
