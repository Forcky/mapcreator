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
