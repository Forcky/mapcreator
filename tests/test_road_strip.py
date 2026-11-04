"""A mapped line that runs up a cutting beside the road: the profile takes the road's own strip.
And the validator's warning for crests and sags too tight for the AI's planned speed."""

import numpy as np
import pytest

from mapcreator.core import ai_line, profile, validate

S = np.arange(0.0, 400.0)
XY = np.column_stack([S, np.zeros_like(S)])  # the mapped line: east along y = 0
HEADING = np.zeros_like(S)


def batter(strip_at: float | None, steep=(150.0, 250.0), road_width: float = 8.0):
    """Ground climbing 7% along x. Across, between x = 150 and 250 the mapped line sits on a 40% side slope, with
    the real road a flat strip `road_width` across, centred `strip_at` metres to its left (None: no strip, just
    the slope)."""

    def sample(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        road = 0.07 * x
        on = (x >= steep[0]) & (x <= steep[1])
        if strip_at is None:
            return np.where(on, road + 0.4 * y, road)
        # Flat across the road; rising 40% away from it beyond, towards the line.
        across = np.abs(y - strip_at)
        return np.where(on, road + 0.4 * np.maximum(across - road_width / 2, 0.0), road)

    return sample


def test_the_road_strip_beside_a_line_on_a_cutting_gives_the_heights():
    sample = batter(10.0)
    z_raw = sample(XY[:, 0], XY[:, 1])
    got = profile.road_strip_heights(S, XY, HEADING, sample, z_raw, np.ones(len(S), bool))
    on = (S > 160) & (S < 240)
    assert got.off_road[on].all() and not got.off_road[S < 100].any()
    assert np.allclose(got.z[on], 0.07 * S[on], atol=0.05)  # the strip, not the batter 2.4 m above it
    assert (z_raw[on] - 0.07 * S[on]).min() > 2.0  # the line's own reading is on the batter, 2.4 m up
    assert np.nanmedian(got.strip_offset[on]) > 4  # to the left
    assert not got.coarse.any()


def test_a_narrow_road_s_strip_is_found_when_its_width_is_known():
    # A single-lane road, 4.5 m across, 8 m to the left of a line on a cutting. A 7-sample strip (6 m between
    # its outer samples) can't lie flat on it, so without the width the stretch is profiled as coarse; with it,
    # the strip is 5 samples (4 m) and fits.
    sample = batter(8.0, road_width=4.5)
    z_raw = sample(XY[:, 0], XY[:, 1])
    fine = np.ones(len(S), bool)
    on = (S > 160) & (S < 240)
    blind = profile.road_strip_heights(S, XY, HEADING, sample, z_raw, fine)
    assert blind.coarse[on].all()
    got = profile.road_strip_heights(S, XY, HEADING, sample, z_raw, fine, width=np.full(len(S), 4.5))
    assert got.off_road[on].all() and not got.coarse.any()
    assert np.allclose(got.z[on], 0.07 * S[on], atol=0.05)  # the road, not the batter
    assert (z_raw[on] - 0.07 * S[on]).min() > 2.0
    assert np.nanmedian(got.strip_offset[on]) == pytest.approx(8.0, abs=1.0)


@pytest.mark.parametrize("width,half", [(3.0, 2), (4.0, 2), (4.5, 2), (5.9, 2), (6.0, 3), (6.4, 3), (8.0, 4)])
def test_the_strip_follows_the_road_s_width(width, half):
    assert int(profile.strip_half(width)) == half  # 6.4 m: +-3 m, as before widths were known


def test_a_line_on_a_narrow_road_on_a_hillside_isn_t_taken_for_off_road():
    # The line is on a 4.5 m road cut into a 40% hillside: within +-3 m of the line, the old window reached
    # over the batter at both edges and read it as steep.
    sample = batter(0.0, road_width=4.5)
    z_raw = sample(XY[:, 0], XY[:, 1])
    got = profile.road_strip_heights(S, XY, HEADING, sample, z_raw, np.ones(len(S), bool), width=np.full(len(S), 4.5))
    assert not got.off_road.any() and np.array_equal(got.z, z_raw)


def test_with_no_strip_the_stretch_is_profiled_as_coarse():
    sample = batter(None)
    z_raw = sample(XY[:, 0], XY[:, 1])
    got = profile.road_strip_heights(S, XY, HEADING, sample, z_raw, np.ones(len(S), bool))
    on = (S > 160) & (S < 240)
    assert got.coarse[on].all() and np.array_equal(got.z, z_raw)


def test_a_line_on_its_road_and_coarse_dem_stretches_are_left_alone():
    flat = batter(0.0)  # the strip is under the line: it's on the road
    z_raw = flat(XY[:, 0], XY[:, 1])
    got = profile.road_strip_heights(S, XY, HEADING, flat, z_raw, np.ones(len(S), bool))
    assert not got.off_road.any() and np.array_equal(got.z, z_raw)
    steep = batter(10.0)
    z_raw = steep(XY[:, 0], XY[:, 1])
    got = profile.road_strip_heights(S, XY, HEADING, steep, z_raw, np.zeros(len(S), bool))
    assert not got.off_road.any()  # not on LiDAR: the coarse handling already applies


def line_with(heights, speed):
    """A straight AI line along x with these heights, every metre, at a constant planned speed (m/s)."""
    n = len(heights)
    extra = np.zeros((n, 18))
    extra[:, 0] = speed
    return ai_line.AiLine(np.column_stack([np.arange(n, dtype=float), heights, np.zeros(n)]),
                          np.arange(n, dtype=float), extra)


def test_a_sag_too_tight_for_the_ai_s_speed_is_a_warning_not_a_failure():
    x = np.arange(600.0)
    sag = np.where(np.abs(x - 300) < 40, (x - 300) ** 2 / (2 * 60.0), 40 ** 2 / (2 * 60.0) + 40 / 60.0 * (np.abs(x - 300) - 40))
    rep = validate.Report()
    validate.vertical_checks(rep, "t", line_with(sag, 30.0))  # 60 m at 108 km/h: about 1.5 g
    assert rep.ok and len(rep.warnings) == 1 and "sag" in rep.warnings[0]
    gentle = validate.Report()
    validate.vertical_checks(gentle, "t", line_with(sag / 10, 30.0))  # 600 m
    assert not gentle.warnings
