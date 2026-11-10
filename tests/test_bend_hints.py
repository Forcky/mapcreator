"""Pace hints at risky bends (#63): a bend AC's AI would take faster than the road allows, because it starts over a
crest, runs down a steep grade or has no banking, gets an `ai_hints.ini` section, and the validator warns about it."""

import math
import re

import numpy as np
import pytest

from mapcreator.core import ai_line, layout, package, validate
from mapcreator.core import section as sec
from mapcreator.core.layout import RoadModel
from mapcreator.core.scene import LocalFrame

STEP = 5.0  # metres between points, as fast_lane.ai has them
# Like Lyell Highway forward at 0.7 km (#63): a 120 m left-hander that starts over a crest (a 600 m vertical radius)
# and runs down a 10% grade.
CREST = {"crest_r": 600.0, "descent": 0.1, "crest_from": 590.0}


def bend_line(*, radius=120.0, bend_from=600.0, bend_len=160.0, crest_r=None, crest_from=570.0, descent=0.0,
              hold=100.0, banking=0.0, total=1400.0, grades=None, ease=0.0) -> tuple[ai_line.AiLine, np.ndarray]:
    """A stand-in AI line as the file holds it, points STEP apart: straight, a left-hand bend of `radius` from
    `bend_from` for `bend_len` metres, straight again, its curvature eased in and out over `ease` metres either
    side (at once, with none). From `crest_from` the grade falls from level to `descent` over a crest of vertical
    radius `crest_r` (or at once, with none), holds it `hold` metres, then eases back to level over a 1,000 m sag;
    or `grades` gives the grade as (metres, grades) to interpolate. And the road's rise to the left beside each
    point, for `banking` into the bend."""
    s = np.arange(0.0, total + STEP / 2, STEP)
    bend = (s >= bend_from) & (s <= bend_from + bend_len)
    over = descent * crest_r if crest_r else STEP
    sag = descent * 1000.0
    grade = -np.interp(s, [crest_from, crest_from + over, crest_from + over + hold, crest_from + over + hold + sag],
                       [0.0, descent, descent, 0.0]) if grades is None else np.interp(s, *grades)
    height = 100.0 + np.concatenate([[0.0], np.cumsum((grade[1:] + grade[:-1]) / 2 * np.diff(s))])
    extra = np.zeros((len(s), ai_line.EXTRA_FIELDS))
    extra[:, 0] = 30.0
    extra[:, 4] = np.where(bend, radius, 1000.0)
    if ease:
        k = np.interp(s, [bend_from - ease, bend_from, bend_from + bend_len, bend_from + bend_len + ease],
                      [1e-3, 1 / radius, 1 / radius, 1e-3])
        extra[:, 4] = np.minimum(1 / k, 1000.0)
    extra[:, 5] = extra[:, 6] = 3.5
    extra[:, 8] = 1.0  # turning left (and + on the straights, as a file's sign of a tiny curvature can be)
    extra[:, 13] = 1.0
    extra[:, 17] = grade
    pos = np.column_stack([s, height + ai_line.LIFT_M, np.zeros(len(s))])
    line = ai_line.AiLine(pos=pos, length=s, extra=extra, station=s + ai_line.END_INSET_M, offset=np.zeros(len(s)))
    return line, np.where(bend, -banking, 0.0)  # banked into a left-hander: the right edge is higher


def at(paces: dict, where: float, key: str = "pace") -> float:
    return float(np.interp(where, paces["s"], paces[key]))


# --- The rule ---------------------------------------------------------------------------------------------------

def test_a_level_bend_gets_no_hint():
    line, _ = bend_line()
    paces = ai_line.bend_paces(line)
    assert np.all(paces["pace"] == 1.0)
    assert ai_line.bend_hints(line) == []
    # The AI reaches its corner speed there, sqrt(0.9 g R): the grip it counts on is all there is, and no more.
    assert at(paces, 680, "speed") == pytest.approx(math.sqrt(ai_line.BEND_GRIP * ai_line.G * 120.0), rel=0.01)


def test_the_pace_is_the_grip_left_after_the_crest_and_the_descent():
    # Over the top of a 1,000 m crest in the bend (10% up to 10% down over 200 m): the tyres carry
    # 1 - v^2 / (R_v g) of the car, at the AI's speed.
    line, _ = bend_line(grades=([580.0, 780.0], [0.1, -0.1]))
    assert at(ai_line.bend_paces(line), 680) == pytest.approx(math.sqrt(1 - ai_line.BEND_GRIP * 120.0 / 1000.0),
                                                              abs=0.005)
    # A steady 6% descent round it: g times the grade goes on the brakes, out of the 0.9 g the AI counts on.
    line, _ = bend_line(descent=0.06, crest_from=400.0, hold=600.0)
    assert at(ai_line.bend_paces(line), 680) == pytest.approx(math.sqrt(1 - 0.06 / ai_line.BEND_GRIP), abs=0.005)
    # Neither is enough for a hint on its own: the shortfall is too small (BEND_TRIGGER).
    assert ai_line.bend_hints(line) == []


def test_a_bend_over_a_crest_and_down_a_steep_grade_gets_a_hint():
    line, _ = bend_line(**CREST)
    hints = ai_line.bend_hints(line)
    assert len(hints) == 1
    hint = hints[0]
    assert 0.8 < hint.pace < ai_line.BEND_TRIGGER
    assert hint.pace == math.floor(hint.pace * 100 + 1e-9) / 100  # two decimals, rounded down
    assert 600 <= hint.start <= hint.at < hint.end <= 760  # where the bend is over the crest and descent
    assert 550 < hint.crest_radius_m < 700 and 0.03 < hint.descent <= 0.1 and hint.banking == 0.0
    assert hint.road_kmh < hint.ai_kmh
    # The pace that brings the AI's corner speed down to what the grip left allows (and lower by as much as the car
    # overshoots the AI's plan, here not at all: the crest is past the braking zone).
    corner = math.sqrt(ai_line.BEND_GRIP * ai_line.G * hint.radius_m) * 3.6
    assert hint.car_kmh == pytest.approx(hint.ai_kmh)
    assert hint.pace == pytest.approx(hint.road_kmh / corner, abs=0.011)


def test_banking_into_the_bend_takes_the_hint_away():
    line, banking = bend_line(**CREST, banking=0.07)
    level, banked = ai_line.bend_paces(line), ai_line.bend_paces(line, banking)
    assert np.all(banked["pace"] >= level["pace"])
    assert at(banked, 650) > at(level, 650) + 0.03
    assert ai_line.bend_hints(line, banking) == []
    # Banked away from the bend, the AI is shorter still.
    worse = ai_line.bend_hints(line, -banking)
    assert worse and worse[0].pace < ai_line.bend_hints(line)[0].pace and worse[0].banking == pytest.approx(-0.07)


def test_no_hint_in_a_wide_bend_or_for_a_moment():
    # Wider than BEND_MAX_RADIUS_M: hints did nothing there in game, and AC's AI is nearer its top speed.
    line, _ = bend_line(radius=400.0, **CREST)
    assert ai_line.bend_hints(line) == []
    # A kink over the crest that's gone in under BEND_MIN_S.
    line, _ = bend_line(bend_len=20.0, bend_from=620.0, **CREST)
    paces = ai_line.bend_paces(line)
    assert paces["pace"].min() < ai_line.BEND_TRIGGER
    assert ai_line.bend_hints(line) == []


def test_the_hint_is_never_below_the_advisory_speed():
    # A jump over a 150 m crest into a 120 m bend: the grip left is nearly nothing, but a sane driver still
    # takes the bend at its advisory speed, sqrt(127 R (e + 0.35)) km/h.
    line, _ = bend_line(crest_r=150.0, descent=0.2, crest_from=600.0)
    hint = ai_line.bend_hints(line)[0]
    advisory = math.sqrt(127.0 * hint.radius_m * ai_line.BEND_ADVISORY_FRICTION)
    corner = math.sqrt(ai_line.BEND_GRIP * ai_line.G * hint.radius_m) * 3.6
    assert hint.pace == pytest.approx(math.floor(advisory / corner * 100) / 100)
    assert hint.pace == pytest.approx(0.62, abs=0.01)


def test_the_road_banking_leaves_the_crown_out():
    s = np.arange(0.0, 500.0)
    flat = np.zeros(len(s))
    for e in (0.0, 0.01, 0.05, -0.07):
        left, right = sec.half_slopes(np.full(len(s), e))
        road = RoadModel.from_arrays(s, np.column_stack([s, flat]), flat, flat, flat, flat, np.full(len(s), 3.5),
                                     np.full(len(s), 3.5), left, right, False)
        line = ai_line.AiLine(pos=np.zeros((3, 3)), length=np.zeros(3), extra=np.zeros((3, 18)),
                              station=np.array([100.0, 200.0, 300.0]), offset=np.array([-1.0, 0.0, 1.5]))
        assert ai_line.road_banking(road, line) == pytest.approx(np.full(3, e))  # + is the left edge higher
        back = RoadModel.from_arrays(s, np.column_stack([s, flat]), flat, flat, flat, flat, np.full(len(s), 3.5),
                                     np.full(len(s), 3.5), left, right, True)
        assert ai_line.road_banking(back, line) == pytest.approx(np.full(3, -e))  # driven the other way


# --- Braking for a bend over a crest (Lyell Highway reverse at 14.0 km, in game on 10 Nov) ------------------------

# A straight, then over a 250 m crest from 500 m and down 15%, into a 45 m left-hander from 560 m eased in over the
# 50 m before it: the braking zone runs over the crest, where the tyres carry only part of the car.
OVER_THE_TOP = {"radius": 45.0, "bend_from": 560.0, "bend_len": 60.0, "ease": 50.0, "crest_r": 250.0,
                "crest_from": 500.0, "descent": 0.15, "hold": 400.0}


def test_the_car_keeps_to_the_plan_unless_a_crest_cuts_its_braking():
    d = np.arange(0.0, 300.0)
    curvature = np.where(d >= 200.0, 1 / 50.0, 1e-3)  # a straight, then a 50 m bend to brake for from 40 m/s
    none = np.zeros(len(d))
    # Level, downhill or uphill, the plan brakes at BRAKE plus g times the grade, less what cornering takes, and the
    # car at least as hard: with no crest it keeps to the plan exactly.
    for grade in (none, np.full(len(d), -0.1), np.full(len(d), 0.08)):
        planned = ai_line.speed_profile(d, curvature, grade, v_max=40.0, mu=ai_line.BEND_GRIP)
        assert np.array_equal(ai_line.arrival_speeds(d, planned, grade, none), planned)
    # Over a 300 m crest from 130 m to 170 m, in the braking zone, the tyres carry 1 - v^2 / (R_v g) of the car,
    # about half at 38 m/s: it brakes at that share of the rate and comes off the crest faster than planned.
    planned = ai_line.speed_profile(d, curvature, none, v_max=40.0, mu=ai_line.BEND_GRIP)
    assert planned[131] < planned[130] < 40.0  # braking there
    crest = np.where((d >= 130.0) & (d < 170.0), 1 / 300.0, 0.0)
    car = ai_line.arrival_speeds(d, planned, none, crest)
    assert np.all(car >= planned) and np.array_equal(car[:131], planned[:131])
    load = 1 - planned[130] ** 2 / (300.0 * ai_line.G)
    assert 0.45 < load < 0.55
    assert car[131] == pytest.approx(math.sqrt(planned[130] ** 2 - 2 * ai_line.BRAKE * load), abs=1e-9)
    assert car[170] > planned[170] + 3.0
    # Past the crest it brakes at the full rate again, and it's back on the plan in the bend.
    assert np.array_equal(car[-50:], planned[-50:])


def _with_hints(line, bends):
    """The AI's plan with the bends' hints applied (a hint scales the corner speed sqrt(BEND_GRIP g R), as a radius
    R pace^2 would), the car's speed braking for it, and the speed the grip allows, every metre."""
    p = ai_line.bend_paces(line)
    s = p["s"]
    pace = np.ones(len(s))
    for b in bends:
        for a, e, q in b.sections(s[-1]):
            pace[(s >= a) & (s < e)] = np.minimum(pace[(s >= a) & (s < e)], q)
    grade = np.interp(s, line.length, line.extra[:, 17])
    planned = ai_line.speed_profile(s, 1 / (p["radius"] * pace ** 2), grade, v_max=ai_line.BEND_TOP_KMH / 3.6,
                                    mu=ai_line.BEND_GRIP)
    car = ai_line.arrival_speeds(s, planned, grade, p["crest"])
    allowed = np.sqrt(ai_line.G * (ai_line.BEND_GRIP - p["descent"])
                      / (1 / p["radius"] + ai_line.BEND_GRIP * p["crest"]))
    return s, car, allowed


def test_braking_over_a_crest_into_a_steep_tight_bend_gets_a_hint_over_the_bend():
    line, _ = bend_line(**OVER_THE_TOP)
    paces = ai_line.bend_paces(line)
    # The AI plans to reach its corner speed, but braking over the crest the car comes into the bend faster.
    assert at(paces, 540, "car") > at(paces, 540, "speed") * 1.08
    hints = ai_line.bend_hints(line)
    assert len(hints) == 1
    hint = hints[0]
    # Over the bend, from where the car first comes in faster than the grip allows, not on the straight before.
    assert 510 <= hint.start <= 530 and 550 <= hint.end <= 600 and hint.start <= hint.at < hint.end
    assert hint.car_kmh > hint.ai_kmh + 5 and hint.road_kmh < hint.car_kmh
    assert hint.radius_m < ai_line.BEND_MAX_RADIUS_M and hint.descent > 0.1 and hint.crest_radius_m < 700
    # Low enough for the braking it has: with the hint, the car comes into the bend no faster than the grip allows,
    # from where the hint eases in to where it ends; with none, it's over by a sixth.
    s, car, allowed = _with_hints(line, hints)
    held = (s >= hint.start - 2 * ai_line.BEND_RAMP_M) & (s <= hint.end) & (paces["radius"] < ai_line.BEND_MAX_RADIUS_M)
    assert np.all(car[held] <= allowed[held])
    s, car, allowed = _with_hints(line, [])
    assert np.max(car[held] / allowed[held]) > 1.15
    # The pace: what the grip allows over the AI's corner speed, lower by as much as the car overshoots its plan.
    corner = math.sqrt(ai_line.BEND_GRIP * ai_line.G * hint.radius_m) * 3.6
    assert hint.pace == pytest.approx(hint.road_kmh / corner * hint.ai_kmh / hint.car_kmh, abs=0.011)
    assert 0.62 <= hint.pace < 0.8
    # The same bend and descent with the crest well before the braking zone: the AI arrives as it planned, short of
    # the grip only by the grade (a pace of 0.91), and gets no hint, as before.
    line, _ = bend_line(**{**OVER_THE_TOP, "crest_from": 300.0})
    assert ai_line.bend_hints(line) == []
    assert at(ai_line.bend_paces(line), 590) == pytest.approx(math.sqrt(1 - 0.15 / ai_line.BEND_GRIP), abs=0.005)


def _profiled_road(curvature, height, *, reverse: bool) -> RoadModel:
    """A road 7 m wide with no banking, sampled every metre in its forward direction, turning as `curvature` (1/m,
    + left) says and as high as `height`, as the layout driving it `reverse` or not sees it."""
    k, z = np.asarray(curvature, dtype=float), np.asarray(height, dtype=float)
    s = np.arange(len(k), dtype=float)
    heading = np.concatenate([[0.0], np.cumsum((k[1:] + k[:-1]) / 2)])
    mid = (heading[1:] + heading[:-1]) / 2
    xy = np.vstack([[0.0, 0.0], np.column_stack([np.cumsum(np.cos(mid)), np.cumsum(np.sin(mid))])])
    left, right = sec.half_slopes(np.zeros(len(s)))
    return RoadModel.from_arrays(s, xy + [500_000.0, 5_200_000.0], z, heading, k, np.gradient(z, s),
                                 np.full(len(s), 3.5), np.full(len(s), 3.5), left, right, reverse)


def test_in_reverse_the_hint_lands_on_the_bend(tmp_path):
    # Driven in reverse, metres r from the reverse start: over a 250 m crest from 500 m, down 15% from 537.5 m to
    # 800 m, into a 45 m left-hander at 560-620 m eased in from 510 m and out to 670 m. Forward it's the same road
    # from the other end: up 15% through a right-hander, then over the crest.
    r = np.arange(0.0, 1401.0)
    turn = np.interp(r, [510.0, 560.0, 620.0, 670.0], [0.0, 1 / 45.0, 1 / 45.0, 0.0])
    grade = -np.interp(r, [500.0, 537.5, 800.0, 900.0], [0.0, 0.15, 0.15, 0.0])
    height = 200.0 + np.concatenate([[0.0], np.cumsum((grade[1:] + grade[:-1]) / 2)])
    placed = {}
    for reverse in (False, True):
        road = _profiled_road(-turn[::-1], height[::-1], reverse=reverse)
        line = ai_line.build(road, LocalFrame(500_000.0, 5_200_000.0, 10.0), end_short_m=layout.AI_END_SHORT_M)
        path = tmp_path / ("reverse" if reverse else "forward") / "fast_lane.ai"
        ai_line.write(path, line)
        bends = ai_line.bend_hints(ai_line.read(path), ai_line.road_banking(road, line))  # as the track stage does
        finish = layout.finish_distance(road.length)
        hints = ai_line.pace_hints(road.length, finish, line=line, bends=bends)
        assert hints[-3:] == ai_line.runoff_hints(road.length, finish, line=line)
        total = line.length[-1]
        # Back from the spline fractions AC reads to metres along this layout's road.
        placed[reverse] = [(*np.interp([a * total, b * total], line.length, line.station), q)
                           for a, b, q in hints[:-3]]
    assert placed[False] == []  # uphill into the bend: nothing
    steps = placed[True]
    assert len(steps) == 5
    a, b, p = steps[2]
    assert p < 0.8 and 505 <= a <= 530 and 550 <= b <= 600  # the full pace on the bend's way in, in reverse metres
    assert steps[0][0] >= 450 and steps[-1][1] <= 650  # its ramps just before and into the bend
    assert all(x[1] == pytest.approx(y[0], abs=0.01) for x, y in zip(steps, steps[1:]))


# --- What the track writes ----------------------------------------------------------------------------------------

def test_a_hint_eases_in_and_out_in_two_steps():
    hint = ai_line.BendHint(start=100.0, end=150.0, pace=0.85, at=120.0, radius_m=120.0, crest_radius_m=550.0,
                            descent=0.05, banking=0.0, ai_kmh=115.0, road_kmh=100.0)
    assert hint.sections() == [(50.0, 75.0, 0.95), (75.0, 100.0, 0.9), (100.0, 150.0, 0.85), (150.0, 175.0, 0.9),
                               (175.0, 200.0, 0.95)]
    # Cut to the line at both ends.
    near = ai_line.BendHint(start=30.0, end=60.0, pace=0.85, at=40.0, radius_m=120.0, crest_radius_m=math.inf,
                            descent=0.1, banking=0.0, ai_kmh=100.0, road_kmh=85.0)
    assert near.sections(total=90.0) == [(0.0, 5.0, 0.95), (5.0, 30.0, 0.9), (30.0, 60.0, 0.85), (60.0, 85.0, 0.9),
                                         (85.0, 90.0, 0.95)]
    # A step that rounds to 1 is no hint at all.
    slight = ai_line.BendHint(start=100.0, end=150.0, pace=0.99, at=120.0, radius_m=120.0, crest_radius_m=math.inf,
                              descent=0.1, banking=0.0, ai_kmh=100.0, road_kmh=99.0)
    assert slight.sections() == [(75.0, 100.0, 0.99), (100.0, 150.0, 0.99), (150.0, 175.0, 0.99)]


def _sections(text: str) -> list[dict[str, str]]:
    return [dict(kv.split("=", 1) for kv in block.split("\r\n")[1:] if "=" in kv)
            for block in text.split("\r\n\r\n") if block.startswith("[HINT_")]


def test_bend_hints_go_in_with_the_run_off_and_gravel_without_stacking():
    line, _ = bend_line(**CREST, total=2400.0)
    length = float(line.station[-1] + layout.AI_END_SHORT_M)
    finish = layout.finish_distance(length)
    bends = ai_line.bend_hints(line)
    assert len(bends) == 1
    total = line.length[-1]
    p = bends[0].pace
    # Sealed: the bend's five steps, then the run-off ramp exactly as without them.
    hints = ai_line.pace_hints(length, finish, line=line, bends=bends)
    runoff = ai_line.runoff_hints(length, finish, line=line)
    assert hints[-3:] == runoff and hints[:-3] == [(a / total, b / total, q) for a, b, q in bends[0].sections()]
    values = [s["VALUE"] for s in _sections(package.ai_hints_ini(hints))]
    assert values == [f"{x:g}" for x in (round((2 + p) / 3, 2), round((1 + 2 * p) / 3, 2), p,
                                         round((1 + 2 * p) / 3, 2), round((2 + p) / 3, 2), 0.5, 0.2, 0.0)]
    # Without a bend, the same as before bends had hints.
    assert ai_line.pace_hints(length, finish, line=line, bends=[]) == runoff
    # Gravel over the bend: the lower pace holds in each stretch, never one on top of the other.
    assert bends[0].start == 600.0 and bends[0].end > 645.0 and layout.AI_GRAVEL_PACE < p
    gravel = ai_line.pace_hints(length, finish, line=line, bends=bends, unsealed=[(400.0, 640.0)])
    assert all(a[1] <= b[0] + 1e-12 and a[0] < a[1] for a, b in zip(gravel, gravel[1:]))
    frac = ai_line.spline_fraction(length, line=line)

    def pace_at(d):  # d along the line, which here is the road less END_INSET_M
        return [q for a, b, q in gravel if a <= frac(d + ai_line.END_INSET_M) < b] or [1.0]

    assert pace_at(450.0) == [layout.AI_GRAVEL_PACE]  # gravel alone
    assert pace_at(560.0) == pace_at(620.0) == [layout.AI_GRAVEL_PACE]  # both: the lower holds, once
    assert pace_at(645.0) == [p]  # past the gravel, the bend's own
    assert pace_at(bends[0].end + 10) == [round((1 + 2 * p) / 3, 2)]  # and its ramp out
    assert pace_at(1500.0) == [1.0]
    assert gravel[-3:] == runoff


def test_a_bend_hint_stops_at_the_finish():
    line, _ = bend_line(**CREST, total=1000.0)
    length = float(line.station[-1] + layout.AI_END_SHORT_M)
    finish = 700.0  # inside the bend's hint: from here the run-off ramp sets the pace
    hints = ai_line.pace_hints(length, finish, line=line, bends=ai_line.bend_hints(line))
    cut = ai_line.spline_fraction(length, line=line)(finish)
    runoff = ai_line.runoff_hints(length, finish, line=line)
    assert hints[-len(runoff):] == runoff
    assert all(b <= cut + 1e-12 for _, b, _ in hints[:-len(runoff)]) and hints[-len(runoff) - 1][1] == cut


# --- The validator ------------------------------------------------------------------------------------------------

class LevelRoad:
    """A stand-in for the road's triangles: level everywhere, so no banking."""

    def height(self, x, z):
        return 0.0


def _layout(tmp_path, line, hints):
    base = tmp_path / "forward"
    ai_line.write(base / "ai" / "fast_lane.ai", line)
    (base / "data").mkdir(parents=True)
    (base / "data" / "ai_hints.ini").write_text(package.ai_hints_ini(hints), encoding="ascii", newline="")
    return ai_line.read(base / "ai" / "fast_lane.ai"), base / "data" / "ai_hints.ini"


def test_the_validator_warns_where_a_bend_is_risky_and_says_the_hint_is_written(tmp_path):
    line, _ = bend_line(**CREST, total=2400.0)
    length = float(line.station[-1] + layout.AI_END_SHORT_M)
    finish = layout.finish_distance(length)
    written, ini = _layout(tmp_path, line, ai_line.pace_hints(length, finish, line=line,
                                                              bends=ai_line.bend_hints(line)))
    bend = ai_line.bend_hints(written)[0]
    rep = validate.Report()
    validate.bend_hint_checks(rep, "forward", written, LevelRoad(), ini)
    assert rep.ok and len(rep.warnings) == 1
    warning = rep.warnings[0]
    assert warning.startswith(f"forward: the 120 m bend at {bend.start:.0f}-{bend.end:.0f} m along the AI line (over a ")
    assert "% downhill, 0.0% banking): AC's AI may take it at" in warning
    assert warning.endswith(f"a hint of {bend.pace:g} is written there")
    # A track written without it: the same bend, and no hint there.
    (tmp_path / "forward" / "data" / "ai_hints.ini").write_text(
        package.ai_hints_ini(ai_line.runoff_hints(length, finish, line=line)), encoding="ascii", newline="")
    rep = validate.Report()
    validate.bend_hint_checks(rep, "forward", written, LevelRoad(), ini)
    assert rep.ok and len(rep.warnings) == 1 and rep.warnings[0].endswith(f"no hint of {bend.pace:g} is written there")


def test_the_validator_says_when_a_crest_cuts_the_braking(tmp_path):
    line, _ = bend_line(**OVER_THE_TOP, total=2400.0)
    length = float(line.station[-1] + layout.AI_END_SHORT_M)
    finish = layout.finish_distance(length)
    written, ini = _layout(tmp_path, line, ai_line.pace_hints(length, finish, line=line,
                                                              bends=ai_line.bend_hints(line)))
    bend = ai_line.bend_hints(written)[0]
    rep = validate.Report()
    validate.bend_hint_checks(rep, "reverse", written, LevelRoad(), ini)
    assert rep.ok and len(rep.warnings) == 1
    warning = rep.warnings[0]
    # Named by its tightest radius over the hint, not the radius on the way in where the car is furthest over.
    tight = re.match(rf"reverse: the (\d+) m bend at {bend.start:.0f}-{bend.end:.0f} m along the AI line \(over a ",
                     warning)
    assert tight and 45 <= int(tight.group(1)) <= 50 < bend.radius_m
    assert (f"braking over the crest before it, the car may reach the {bend.radius_m:.0f} m radius on its way in at "
            f"{bend.car_kmh:.0f} km/h, where AC's AI plans {bend.ai_kmh:.0f} km/h and the grip left allows "
            f"{bend.road_kmh:.0f} km/h ({bend.pace:g} of the ") in warning
    assert warning.endswith(f"a hint of {bend.pace:g} is written there")


def test_the_validator_passes_a_level_bend(tmp_path):
    line, _ = bend_line()
    written, ini = _layout(tmp_path, line, [])
    rep = validate.Report()
    validate.bend_hint_checks(rep, "main", written, LevelRoad(), ini)
    assert rep.ok and rep.warnings == [] and rep.passed == ["main: no bend the AI would take faster than the road allows"]
