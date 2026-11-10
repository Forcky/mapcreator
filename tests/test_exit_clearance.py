"""Exit clearance: where the racing line apexes early on a descent, it keeps room from the outside edge for the slide
AC's AI makes out of the bend (Lyell Highway reverse at 27.25 km, in game on 10 Nov), or the bend gets a hint where
the road is too narrow for it; the validator warns where a line doesn't."""

import numpy as np
import pytest

from mapcreator.core import ai_line, layout, package, racing_line, validate
from mapcreator.core import section as sec
from mapcreator.core.layout import RoadModel
from mapcreator.core.scene import LocalFrame

FRAME = LocalFrame(500_000.0, 5_200_000.0, 10.0)
NEED = ai_line.EXIT_CLEARANCE_M


def profile(*, offsets, radius=70.0, grade=-0.045, half=3.5, n=400):
    """Arrays every metre as fast_lane.ai holds them, for a left-hander: a straight, a bend tightening to `radius`
    over 100-140 m and held there to 240 m, opening to a straight by 300 m; the line's offset from the road's middle
    (+ left, the inside) interpolated from `offsets` as (metres, offset)."""
    s = np.arange(float(n))
    r = np.interp(s, [0, 100, 140, 240, 300, n], [1000, 300, radius, radius, 1000, 1000])
    a = np.interp(s, *zip(*offsets))
    return s, r, np.ones(n), np.full(n, grade), half - a, half + a


# Like 27.25 km: on the inside (+1.2 m) as the bend tightens, across to the outside half 10 m into its tight part,
# and 1.5 m out (2.0 m from the edge) through the rest of it.
EARLY = [(0, 0.0), (120, 1.2), (150, 0.0), (180, -1.5), (300, -1.0), (400, 0.0)]
# Like 18.5 km: on the inside all through the tight part, out to the edge only once the bend has opened.
LATE = [(0, 0.0), (120, 1.2), (240, 1.0), (330, -1.2), (400, 0.0)]


def test_an_early_apex_on_a_descent_is_found():
    found = ai_line.exit_stretches(*profile(offsets=EARLY))
    assert len(found) == 1
    x = found[0]
    # The tight part (within 10% of 70 m), and on until the bend opens past 1.5 x its radius, 2-3 m after it.
    assert 135 <= x.start <= 145 and 240 <= x.end <= 250 and x.left
    assert x.room == pytest.approx(2.0, abs=0.01) and x.room < NEED and x.radius_m == pytest.approx(70.0, abs=0.5)


def test_no_stretch_for_a_late_apex_uphill_slowly_or_for_a_moment():
    assert ai_line.exit_stretches(*profile(offsets=LATE)) == []  # the line stays inside through the tight part
    assert ai_line.exit_stretches(*profile(offsets=EARLY, grade=0.045)) == []  # uphill: the AI sheds its overspeed
    # At or below MARGIN_SLOW_KMH the AI keeps close to its line: a 30 m bend's corner speed is 58 km/h.
    assert ai_line.exit_stretches(*profile(offsets=EARLY, radius=30.0)) == []
    # Across to the outside only for the last 30 m of the tight part: under EXIT_HOLD_M.
    brief = [(0, 0.0), (120, 1.2), (200, 0.5), (215, -1.5), (300, -1.0), (400, 0.0)]
    assert ai_line.exit_stretches(*profile(offsets=brief)) == []


def test_the_clearance_holds_the_line_past_the_centreline():
    s = np.linspace(0.0, 100.0, 51)
    half = np.full(len(s), 3.5)
    flat = np.zeros(len(s))
    lo, hi = racing_line.bounds(s, half, half, flat, -100.0, 1e9, 2.0, 1.2)
    assert np.allclose(lo, -2.3) and np.allclose(hi, 2.3)
    right = np.where((s >= 40) & (s <= 60), NEED, 0.0)
    lo2, hi2 = racing_line.bounds(s, half, half, flat, -100.0, 1e9, 2.0, 1.2, (flat, right))
    assert np.allclose(hi2, hi)  # the left edge's margin as before
    assert lo2[s == 50][0] == pytest.approx(NEED - 3.5)  # 0.35 m left of the centreline
    assert np.all(lo2[(s < 25) | (s > 75)] == lo[(s < 25) | (s > 75)])
    # Too narrow for it and the left margin both: the left margin holds, and the line stops at it.
    narrow = np.full(len(s), 2.0)
    lo3, hi3 = racing_line.bounds(s, narrow, narrow, flat, -100.0, 1e9, 2.0, 1.2, (flat, right))
    mid = (s >= 45) & (s <= 55)
    assert np.allclose(lo3[mid], hi3[mid]) and np.allclose(hi3[mid], 0.8)


def _road(parts, grade, half=3.5):
    """A road with bends of (metres, radius) pieces, eased over about 30 m, on a steady grade."""
    k = np.concatenate([np.full(int(m), 0.0 if r == 0 else 1.0 / r) for m, r in parts])
    k = np.convolve(k, np.ones(31) / 31, mode="same")
    s = np.arange(len(k), dtype=float)
    heading = np.concatenate([[0.0], np.cumsum((k[1:] + k[:-1]) / 2)])
    mid = (heading[1:] + heading[:-1]) / 2
    xy = np.vstack([[0.0, 0.0], np.column_stack([np.cumsum(np.cos(mid)), np.cumsum(np.sin(mid))])])
    left, right = sec.half_slopes(np.zeros(len(s)))
    return RoadModel.from_arrays(s, xy + [500_000.0, 5_200_000.0], 300.0 + grade * s, heading, k,
                                 np.full(len(s), grade), np.full(len(s), half), np.full(len(s), half), left, right,
                                 False)


# A 70 m left-hander after a 110 m entry, opening through 120 m: the line comes off the inside 50 m into the tight
# part and runs out over the rest. With EXIT_HOLD_M lowered to 10 m that counts as an early apex.
LEFT_HANDER = [(600, 0), (40, 110), (100, 70), (200, 120), (500, 0)]


def _on_file(line):
    s = np.append(np.arange(0.0, line.length[-1], 1.0), line.length[-1])

    def at(k):
        return np.interp(s, line.length, line.extra[:, k])

    return s, at(4), np.sign(at(8)), at(17), at(5), at(6)


def test_the_built_line_keeps_the_room_and_nothing_else_moves(monkeypatch):
    road = _road(LEFT_HANDER, -0.045)
    monkeypatch.setattr(ai_line, "EXIT_HOLD_M", float("inf"))
    free = ai_line.build(road, FRAME, end_short_m=layout.AI_END_SHORT_M)
    assert free.exits == []
    monkeypatch.setattr(ai_line, "EXIT_HOLD_M", 10.0)
    found = ai_line.exit_stretches(*_on_file(free))
    assert len(found) == 1 and found[0].room < NEED - 0.5
    line = ai_line.build(road, FRAME, end_short_m=layout.AI_END_SHORT_M)
    assert len(line.exits) == 1
    a, b, r, short = line.exits[0]
    assert not short and abs(a - found[0].start) < 5 and abs(b - found[0].end) < 5 and r == pytest.approx(70, abs=2)
    s, *_, side_right = _on_file(line)
    held = (s >= a) & (s <= b)
    assert side_right[held].min() >= NEED - ai_line.EXIT_SHORT_TOL_M  # the room from the outside (right) edge
    assert ai_line.exit_hints(line) == []
    # Up to 100 m before the stretch's ease the line is where it was.
    before = free.station < np.interp(a, line.length, line.station) - ai_line.EXIT_EASE_M - 100
    assert np.allclose(free.offset[before], np.interp(free.station[before], line.station, line.offset), atol=0.02)
    # The margin for the line's planned speed still holds from both edges, as the validator checks it.
    margin = racing_line.edge_margin(line.extra[:, 0]) - 0.2
    assert np.all(line.extra[:, 5] >= margin) and np.all(line.extra[:, 6] >= margin)
    # Uphill, the same bend has no early apex to keep room for: the line is the same as with the rule off.
    up = _road(LEFT_HANDER, 0.045)
    with_rule = ai_line.build(up, FRAME, end_short_m=layout.AI_END_SHORT_M)
    monkeypatch.setattr(ai_line, "EXIT_HOLD_M", float("inf"))
    without = ai_line.build(up, FRAME, end_short_m=layout.AI_END_SHORT_M)
    assert with_rule.exits == [] and np.array_equal(with_rule.pos, without.pos)
    assert np.array_equal(with_rule.extra, without.extra)


def test_a_road_too_narrow_for_the_room_gets_a_hint(monkeypatch):
    # 5.6 m wide: the line comes off the inside later, just before the tight part ends, so the hold is lowered to
    # 1 m to count it. The room would put the line 1.05 m inside, past the 0.8 m its inside margin allows.
    monkeypatch.setattr(ai_line, "EXIT_HOLD_M", 1.0)
    road = _road(LEFT_HANDER, -0.045, half=2.8)
    line = ai_line.build(road, FRAME, end_short_m=layout.AI_END_SHORT_M)
    assert len(line.exits) == 1 and line.exits[0][3]  # short of the room
    hints = ai_line.exit_hints(line)
    assert len(hints) == 1
    assert hints[0].pace == pytest.approx(0.86) and hints[0].pace * ai_line.EXIT_OVERSPEED < 1.0
    assert (hints[0].start, hints[0].end) == pytest.approx(line.exits[0][:2])
    length = float(line.station[-1] + layout.AI_END_SHORT_M)
    written = ai_line.pace_hints(length, layout.finish_distance(length), line=line, bends=hints)
    assert [q for _, _, q in written][:5] == [0.95, 0.91, 0.86, 0.91, 0.95]


def test_the_validator_warns_where_a_line_apexes_early_without_the_room(tmp_path):
    s, r, turn, grade, side_left, side_right = profile(offsets=EARLY)
    extra = np.zeros((len(s), ai_line.EXTRA_FIELDS))
    extra[:, 4], extra[:, 5], extra[:, 6], extra[:, 8], extra[:, 13], extra[:, 17] = r, side_left, side_right, 1, 1, grade
    height = 100.0 + np.cumsum(grade)
    line = ai_line.AiLine(pos=np.column_stack([s, height, np.zeros(len(s))]), length=s, extra=extra)
    ini = tmp_path / "ai_hints.ini"
    ini.write_text(package.ai_hints_ini([]), encoding="ascii", newline="")
    rep = validate.Report()
    validate.exit_clearance_checks(rep, "reverse", line, ini)
    assert rep.ok and len(rep.warnings) == 1
    assert rep.warnings[0].startswith("reverse: the line apexes early in the 70 m bend at ")
    assert rep.warnings[0].endswith(f"leaves 2.0 m from the outside edge, where the AI's slide out of it needs {NEED:g} m; "
                                    "no hint is written there")
    # A line that keeps the room: no warning.
    late = profile(offsets=LATE)
    extra[:, 5], extra[:, 6] = late[4], late[5]
    rep = validate.Report()
    validate.exit_clearance_checks(rep, "reverse", line, ini)
    assert rep.warnings == [] and rep.passed == [f"reverse: no early apex short of {NEED:g} m from the outside edge"]
