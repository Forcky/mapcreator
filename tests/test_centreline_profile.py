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
