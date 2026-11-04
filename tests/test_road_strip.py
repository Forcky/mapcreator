"""A mapped line that runs up a cutting beside the road: the profile takes the road's own strip.
And the validator's warning for crests and sags too tight for the AI's planned speed."""

import numpy as np

from mapcreator.core import ai_line, profile, validate

S = np.arange(0.0, 400.0)
XY = np.column_stack([S, np.zeros_like(S)])  # the mapped line: east along y = 0
HEADING = np.zeros_like(S)


def batter(strip_at: float | None, steep=(150.0, 250.0)):
    """Ground climbing 7% along x. Across, between x = 150 and 250 the mapped line sits on a 40% side slope, with
    the real road a flat 8 m strip centred `strip_at` metres to its left (None: no strip, just the slope)."""

    def sample(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        road = 0.07 * x
        on = (x >= steep[0]) & (x <= steep[1])
        if strip_at is None:
            return np.where(on, road + 0.4 * y, road)
        # Flat within 4 m of the strip's centre; rising 40% away from it beyond, towards the line.
        across = np.abs(y - strip_at)
        return np.where(on, road + 0.4 * np.maximum(across - 4.0, 0.0), road)

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
