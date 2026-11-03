"""A bend too tight to drive is a route failure (the person who drew the road can fix or accept it), not a
build failure; with `allow_tight_bends` it's a warning, down to a floor."""

import numpy as np
import pytest

from mapcreator.core import ai_line, validate
from mapcreator.pipeline.validate_stage import TightBend, ValidationFailed


def line(min_radius: float) -> ai_line.AiLine:
    n = 200
    extra = np.zeros((n, ai_line.EXTRA_FIELDS))
    radius = np.full(n, 300.0)
    radius[120] = min_radius
    extra[:, 4] = radius
    extra[:, 0] = np.minimum(np.sqrt(ai_line.MU * ai_line.G * radius), ai_line.V_MAX)
    pos = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)])
    return ai_line.AiLine(pos=pos, length=np.arange(n, dtype=float), extra=extra)


def test_a_gentle_road_passes():
    rep = validate.Report()
    validate.bend_checks(rep, "main", line(40.0))
    assert rep.ok and not rep.bends and not rep.warnings


def test_a_tight_bend_is_a_route_failure_and_says_where():
    rep = validate.Report()
    validate.bend_checks(rep, "main", line(3.2))
    assert rep.failed == [] and len(rep.route_failed) == 1 and not rep.ok
    assert rep.bends == [validate.TightBend("main", 3.2, 120.0, 0.0)]


def test_asked_for_it_is_a_warning():
    rep = validate.Report()
    validate.bend_checks(rep, "main", line(3.2), allow_tight_bends=True)
    assert rep.ok and rep.route_failed == [] and len(rep.warnings) == 1


def test_below_the_floor_not_even_on_request_and_the_slow_ai_is_put_down_to_the_bend():
    rep = validate.Report()
    validate.bend_checks(rep, "main", line(2.0), allow_tight_bends=True)  # 4.5 m/s at the bend
    assert not rep.ok and rep.failed == [] and len(rep.route_failed) == 1


def test_the_floor_keeps_the_ai_speed_check_passing():
    assert np.sqrt(ai_line.MU * ai_line.G * validate.FLOOR_RADIUS_M) > validate.MIN_AI_SPEED


def test_tight_bend_messages_are_for_people():
    can = TightBend(3.2, 442.0, 147.4389, -42.7337, True)
    assert isinstance(can, ValidationFailed)
    assert "3.2 m radius, about 0.4 km from the start" in str(can) and "generate it anyway" in str(can)
    assert can.detail() == {"radius_m": 3.2, "along_m": 442.0, "lon": 147.4389, "lat": -42.7337, "overridable": True}
    cannot = TightBend(1.1, 2500.0, 147.0, -42.0, False)
    assert "no car can drive" in str(cannot) and "anyway" not in str(cannot)
