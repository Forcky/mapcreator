"""Grid spots for AI opponents: pairs behind the start where the road is wide enough, single file where not."""

import json

import numpy as np
import pytest

from mapcreator.core import layout, package
from mapcreator.core.layout import RoadModel, grid, logic_dummies
from mapcreator.core.scene import LocalFrame


def straight(half: float, length: float = 1000.0, crown: float = 0.03) -> RoadModel:
    s = np.arange(0.0, length)
    flat = np.zeros(s.size)
    return RoadModel.from_arrays(s, np.column_stack([s, flat]), np.full(s.size, 100.0), flat, flat, flat,
                                 np.full(s.size, half), np.full(s.size, half), np.full(s.size, crown),
                                 np.full(s.size, crown), False)


def test_a_wide_road_gets_eight_spots_in_pairs_behind_spot_zero():
    spots = grid(straight(3.2))
    front = layout.GATE_FROM_END_M - layout.SPAWN_BEFORE_GATE_M
    assert len(spots) == layout.GRID_MAX
    assert spots[0] == (front, 0.0)  # where M3 checked the spawn
    assert spots[1:3] == [(front - 8, 2.0), (front - 8, -2.0)]
    assert all(d >= layout.GRID_CLEAR_OF_START_M for d, _ in spots)


def test_a_narrow_road_gets_single_file_and_fewer_spots():
    spots = grid(straight(2.5))
    assert all(off == 0.0 for _, off in spots)
    assert [d for d, _ in spots] == [42.0, 34.0, 26.0, 18.0, 10.0]  # until the road's start is too close


def test_each_spot_gets_a_pit_and_a_start_at_the_crowned_road_height():
    road = straight(3.2, crown=0.03)
    dummies = {d.name: d for d in logic_dummies(road, LocalFrame(0.0, 0.0, 0.0))}
    n = len(grid(road))
    assert sorted(k for k in dummies if k.startswith("AC_PIT_")) == sorted(f"AC_PIT_{k}" for k in range(n))
    for k in range(n):
        assert dummies[f"AC_PIT_{k}"].position == dummies[f"AC_START_{k}"].position
    centre, side = dummies["AC_START_0"].position, dummies["AC_START_1"].position
    assert centre[1] == pytest.approx(100.0 + layout.SPAWN_HEIGHT_M)
    assert side[1] == pytest.approx(100.0 - 0.03 * 2.0 + layout.SPAWN_HEIGHT_M)  # 2 m down the crown
    assert side[2] == pytest.approx(-2.0)  # left of an eastbound road is north, which is -z in AC


def test_pitboxes_matches_the_spots():
    text = package.ui_track_json(name="x", description="y", tags=[], lat=-42.0, lon=147.0, country="Australia",
                                 city="Tasmania", length_m=1000.0, width_m=(6.4, 6.4), run="point-to-point",
                                 version="t", year=2026, pitboxes=8)
    assert json.loads(text)["pitboxes"] == "8"
