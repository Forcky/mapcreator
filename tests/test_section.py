import numpy as np
import pytest

from mapcreator.core import section as sec


@pytest.mark.parametrize("tags,expected,source", [
    ({"highway": "secondary", "width": "7.5"}, 7.5, "width"),
    ({"highway": "secondary", "width": "7.5 m"}, 7.5, "width"),
    ({"highway": "secondary", "width": "75"}, 6.4, "class"),  # implausible tag ignored
    ({"highway": "secondary", "lanes": "2"}, 6.4, "lanes"),
    ({"highway": "secondary"}, 6.4, "class"),
    ({"highway": "residential"}, 5.8, "class"),
    ({"highway": "primary_link", "oneway": "yes"}, 4.0, "class"),
    ({"highway": "secondary", "lanes": "1", "oneway": "yes"}, 4.0, "lanes"),
])
def test_width_from_tags(tags, expected, source):
    w, src = sec.width_from_tags(tags)
    assert w == pytest.approx(expected)
    assert src == source


def test_crown_on_straights_and_plane_in_bends():
    left, right = sec.half_slopes(np.array([0.0, 0.0125, 0.025, 0.05, -0.05, 0.2]))
    assert (left[0], right[0]) == pytest.approx((sec.CROWN, sec.CROWN))
    # Halfway to the crown, the high (left) side is flat and the low side keeps the crown.
    assert (left[1], right[1]) == pytest.approx((0.0, sec.CROWN))
    assert (left[2], right[2]) == pytest.approx((-sec.CROWN, sec.CROWN))  # first full plane
    assert (left[3], right[3]) == pytest.approx((-0.05, 0.05))
    assert (left[4], right[4]) == pytest.approx((0.05, -0.05))
    assert (left[5], right[5]) == pytest.approx((-sec.MAX_SUPERELEVATION, sec.MAX_SUPERELEVATION))


def test_width_changes_are_eased_not_stepped():
    s = np.arange(0, 400.0)
    w = np.where(s < 200, 6.0, 7.0)
    e = sec.eased(s, w)
    assert np.abs(np.diff(e)).max() < 0.05
    assert e[0] == pytest.approx(6.0) and e[-1] == pytest.approx(7.0)


@pytest.mark.parametrize("width,limit,kept", [
    (3.0, 0.04, 0.005), (4.0, 0.04, 0.005), (5.0, 0.04, 0.005), (5.7, 0.055, 0.0025), (6.4, 0.07, 0.0),
    (7.0, 0.07, 0.0),
])
def test_the_camber_limit_and_the_crown_kept_follow_the_width(width, limit, kept):
    assert float(sec.camber_limit(width)) == pytest.approx(limit)
    assert float(sec.crown_kept(width)) == pytest.approx(kept)


def test_the_camber_limit_changes_smoothly_with_an_eased_width():
    s = np.arange(0, 400.0)
    width = sec.eased(s, np.where(s < 200, 4.0, 6.4))  # a single-lane seal widening to two lanes
    limit = sec.camber_limit(width)
    assert limit[0] == pytest.approx(0.04) and limit[-1] == pytest.approx(0.07)
    assert np.abs(np.diff(limit)).max() < 0.002


@pytest.mark.parametrize("crown", [sec.CROWN, sec.CROWN_UNSEALED])
def test_a_narrow_road_keeps_a_crown_at_any_camber(crown):
    width = np.full(201, 4.0)
    limit, keep = sec.camber_limit(width), sec.crown_kept(width)
    e = np.clip(np.linspace(-0.12, 0.12, 201), -limit, limit)  # the DEM reads up to 12%; the limit holds it at 4%
    left, right = sec.half_slopes(e, crown, keep=keep)
    assert np.abs(e).max() == pytest.approx(0.04)
    assert ((left + right) / 2 >= keep - 1e-12).all()  # never one tilted plane: a break of at least 1% at the centre
    # Seal or gravel, at the limit it's the same section: the low side falls 4.5%, the high side rises 3.5%.
    assert (left[-1], right[-1]) == pytest.approx((-0.035, 0.045))
    # The camber, measured edge to edge, is the e asked for.
    assert ((right - left) / 2) == pytest.approx(e)
    # On a straight the crown is as before.
    flat_l, flat_r = sec.half_slopes(np.zeros(1), crown, keep=sec.crown_kept(4.0))
    assert (flat_l[0], flat_r[0]) == pytest.approx((crown, crown))


def test_a_wide_road_still_becomes_one_plane_in_a_banked_bend():
    left, right = sec.half_slopes(np.array([0.06]), sec.CROWN, keep=sec.crown_kept(6.4))
    assert (left[0], right[0]) == pytest.approx((-0.06, 0.06))
    assert float(sec.camber_limit(6.4)) == pytest.approx(sec.MAX_SUPERELEVATION)
