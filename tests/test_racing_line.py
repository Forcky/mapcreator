"""The racing line (core/racing_line.py) and the AI line built on it (core/ai_line.py): out-in-out through bends,
on the centreline where the cars stand and stop, on the road surface, and written in AC's format."""

import struct
import time

import numpy as np
import pytest

from mapcreator.core import ai_line, racing_line, validate
from mapcreator.core import section as sec
from mapcreator.core.layout import AI_END_SHORT_M, RoadModel, finish_distance
from mapcreator.core.scene import LocalFrame

FRAME = LocalFrame(500_000.0, 5_200_000.0, 10.0)
HALF = 4.0  # an 8 m road: the line may stand 2.8 m either side of the centreline
ROOM = HALF - racing_line.EDGE_MARGIN_M


def road_from(curvature, *, half=HALF, superelevation=0.0, reverse=False) -> RoadModel:
    """A flat road 100 m up, sampled every metre, turning as `curvature` (1/m, + left) says."""
    k = np.asarray(curvature, dtype=float)
    s = np.arange(len(k), dtype=float)
    heading = np.concatenate([[0.0], np.cumsum((k[1:] + k[:-1]) / 2)])
    mid = (heading[1:] + heading[:-1]) / 2
    xy = np.vstack([[0.0, 0.0], np.column_stack([np.cumsum(np.cos(mid)), np.cumsum(np.sin(mid))])])
    xy += [500_000.0, 5_200_000.0]
    n = len(k)
    sl, sr = sec.half_slopes(np.full(n, superelevation))
    return RoadModel.from_arrays(s, xy, np.full(n, 100.0), heading, k, np.zeros(n), np.full(n, half),
                                 np.full(n, half), sl, sr, reverse)


def bends(*parts) -> np.ndarray:
    """Curvature every metre from (metres, radius) pieces: radius 0 is straight, + turns left, - right."""
    return np.concatenate([np.full(int(m), 0.0 if r == 0 else 1.0 / r) for m, r in parts])


def at(line, d, values):
    return float(np.interp(d, line.station, values))


def test_a_constant_radius_bend_is_taken_out_in_out():
    # 500 m straight, a 90-degree left-hander of 50 m radius (79 m long), 500 m straight.
    road = road_from(bends((500, 0), (np.pi / 2 * 50, 50), (500, 0)))
    line = ai_line.build(road, FRAME)
    start, end = 500.0, 500.0 + np.pi / 2 * 50
    assert at(line, start - 15, line.offset) < -0.6 * ROOM  # outside (right) on the way in
    assert at(line, (start + end) / 2, line.offset) > 0.8 * ROOM  # inside (left) at the apex
    assert at(line, end + 15, line.offset) < -0.6 * ROOM  # outside again on the way out
    assert np.abs(line.offset).max() <= ROOM + 0.05
    assert line.centre_radius.min() == pytest.approx(50, rel=0.05)
    # Wider than the road's own radius, even at the apex: without the curvature limit, the least summed squared
    # curvature hugs the inside there, at the road's radius less the 2.8 m it moved in (48 m).
    assert line.extra[:, 4].min() > 1.05 * 50


def test_a_straight_stays_on_the_centre():
    line = ai_line.build(road_from(bends((2000, 0))), FRAME)
    assert np.abs(line.offset).max() < 1e-9
    # Between two bends 1.2 km apart the line drifts out only near them.
    line = ai_line.build(road_from(bends((300, 0), (60, 40), (1200, 0), (60, -40), (300, 0))), FRAME)
    middle = (np.abs(line.station - 960) < 300)
    assert np.abs(line.offset[middle]).max() < 0.15


def test_the_line_keeps_to_the_centreline_over_the_grid_and_the_run_off():
    # Bends right where the grid stands and over the run-off: the line would cut them if it could.
    road = road_from(bends((20, 0), (40, 30), (600, 0), (40, -30), (400, 0), (60, 25), (240, 0)))
    line = ai_line.build(road, FRAME, end_short_m=AI_END_SHORT_M)
    finish = finish_distance(road.length)
    held = (line.station <= racing_line.HOLD_START_M) | (line.station >= finish - racing_line.HOLD_BEFORE_FINISH_M)
    assert held[:5].all() and held[-5:].all() and (~held).sum() > 50
    assert np.all(line.offset[held] == 0.0)
    centre = FRAME.to_ac(np.interp(line.station, road.d, road.e), np.interp(line.station, road.d, road.n),
                         np.interp(line.station, road.d, road.z) + ai_line.LIFT_M)
    assert np.abs(line.pos[held] - centre[held]).max() < 0.01  # the spline through the centreline's metres
    # Out of the held stretches, over BLEND_M, the line may move off only as far as the blend allows.
    blend = (line.station > racing_line.HOLD_START_M) & (line.station < racing_line.HOLD_START_M + racing_line.BLEND_M)
    x = (line.station[blend] - racing_line.HOLD_START_M) / racing_line.BLEND_M
    assert np.all(np.abs(line.offset[blend]) <= ROOM * x * x * (3 - 2 * x) + 0.05)
    # The run-off hints are fractions of this line's own length, which is shorter than the road's.
    hints = ai_line.runoff_hints(road.length, finish, line=line)
    assert hints[0][0] == pytest.approx(at(line, finish, line.length) / line.length[-1], abs=1e-9)
    assert hints[-1][1] == 1.0 and [p for *_, p in hints] == [0.5, 0.2, 0.0]


def test_the_ai_line_stops_where_it_did():
    road = road_from(bends((300, 0), (60, 40), (640, 0)))
    line = ai_line.build(road, FRAME, end_short_m=AI_END_SHORT_M)
    assert line.station[0] == pytest.approx(ai_line.END_INSET_M)
    assert line.station[-1] == pytest.approx(road.length - AI_END_SHORT_M)


@pytest.mark.parametrize("superelevation", [0.0, 0.05])
def test_heights_follow_the_crown_and_the_banking(superelevation):
    road = road_from(bends((300, 0), (100, 45), (200, 0), (100, -45), (300, 0)), superelevation=superelevation)
    line = ai_line.build(road, FRAME)
    n = line.offset
    assert np.abs(n).max() > 2.0  # the line does move across the road
    sl, sr = (float(v[0]) for v in sec.half_slopes(np.array([superelevation])))
    # The road falls from its crown to each edge by `slope` per metre, as the mesh builds it.
    surface = 100.0 - np.where(n >= 0, sl * n, -sr * n)
    assert line.pos[:, 1] == pytest.approx(surface + ai_line.LIFT_M - FRAME.h0, abs=1e-6)


def test_a_chicane_is_straightened():
    # Left then right, 40 m radius, 50 m each: the centreline swings 2.5 m; the line barely needs to.
    road = road_from(bends((400, 0), (50, 40), (50, -40), (400, 0)))
    line = ai_line.build(road, FRAME)
    assert line.centre_radius.min() == pytest.approx(40, rel=0.05)
    assert line.extra[:, 4].min() > 1.2 * 40
    assert at(line, 425, line.offset) > 0.5 * ROOM  # inside of the left-hander
    assert at(line, 475, line.offset) < -0.5 * ROOM  # inside of the right-hander
    assert np.abs(line.offset).max() <= ROOM + 0.05


def test_a_hairpin_keeps_the_line_on_the_road():
    # A 180-degree hairpin of 9 m radius in an 8 m road: the inside of it is only 5 m from the bend's centre.
    road = road_from(bends((300, 0), (np.pi * 9, 9), (300, 0)))
    line = ai_line.build(road, FRAME)
    assert np.all(np.diff(line.length) > 0)
    assert np.abs(line.offset).max() <= ROOM + 0.1  # a spline between stations 2 m apart can stand a little out
    assert line.extra[:, 4].min() > 9.0
    side = np.minimum(line.extra[:, 5], line.extra[:, 6])
    assert side.min() >= validate.AI_EDGE_MIN_M


def test_the_line_is_never_tighter_than_the_road_nearby():
    rng = np.random.default_rng(7)
    parts = [(200, 0)]
    for _ in range(25):
        parts += [(rng.uniform(20, 120), rng.choice([-1, 1]) * rng.uniform(15, 200)), (rng.uniform(0, 150), 0)]
    for reverse in (False, True):
        road = road_from(bends(*parts, (300, 0)), reverse=reverse)
        line = ai_line.build(road, FRAME, end_short_m=AI_END_SHORT_M)
        rep = validate.Report()
        validate.racing_line_checks(rep, "t", line, _flat_road_surface(), validate.road_middle(line)[1])
        assert [p for p in rep.passed if "nowhere tighter than the road nearby" in p], rep.failed


def test_a_long_road_solves_quickly():
    # 64 km of bends, as long as the longest route a job may ask for.
    rng = np.random.default_rng(64)
    parts = []
    while sum(m for m, _ in parts) < 64_000:
        parts += [(rng.uniform(30, 400), 0), (rng.uniform(15, 150), rng.choice([-1, 1]) * rng.uniform(12, 400))]
    road = road_from(bends(*parts), half=3.2)
    t = time.perf_counter()
    off = racing_line.offsets(road.d, road.e, road.n, road.heading, road.curvature, road.half_left, road.half_right,
                              hold_until=racing_line.HOLD_START_M, hold_from=road.length - 300)
    solve = time.perf_counter() - t
    assert np.all(off.n >= off.lo - 1e-9) and np.all(off.n <= off.hi + 1e-9)
    # Within the curvature limit everywhere: no tighter than the road nearby.
    h = np.interp(off.s, road.d, np.unwrap(road.heading))
    p = np.column_stack([np.interp(off.s, road.d, road.e - road.e[0]), np.interp(off.s, road.d, road.n - road.n[0])])
    k = racing_line._menger(p + np.column_stack([-np.sin(h), np.cos(h)]) * off.n[:, None])
    assert np.all(np.abs(k) <= off.limit * 1.01)
    t = time.perf_counter()
    line = ai_line.build(road, FRAME, end_short_m=AI_END_SHORT_M)
    whole = time.perf_counter() - t
    print(f"64 km: solve {solve:.1f} s, whole AI line {whole:.1f} s, {off.steps} steps")
    # About 3 s on a laptop, but 22-32 s on a build server's slower cores (4 Nov). This only catches a pathological
    # slowdown; a real build's budget is the worker's 1 h limit.
    assert whole < 120.0
    assert line.extra[:, 4].min() >= 0.95 * line.centre_radius.min()


def test_fast_lane_ai_round_trips_with_points_5_m_apart(tmp_path):
    road = road_from(bends((300, 0), (60, 30), (100, 0), (80, -60), (460, 0)))
    line = ai_line.build(road, FRAME, end_short_m=AI_END_SHORT_M)
    gaps = np.diff(line.length)
    assert ai_line.AI_POINT_SPACING_M == 5.0
    assert np.all((gaps > 4.5) & (gaps <= 5.05)) and np.median(gaps) == pytest.approx(5.0, abs=0.1)
    assert line.extra[:, 12][:-1] == pytest.approx(gaps)
    # Both edge distances measured from the line, summing to the road's width.
    assert line.extra[:, 5] + line.extra[:, 6] == pytest.approx(2 * HALF)
    assert line.extra[:, 5] == pytest.approx(HALF - line.offset)
    path = tmp_path / "ai" / "fast_lane.ai"
    ai_line.write(path, line)
    n = len(line.pos)
    assert path.stat().st_size == 16 + 20 * n + 4 + 72 * n + 4
    assert struct.unpack_from("<4i", path.read_bytes()) == (7, n, 0, 0)
    back = ai_line.read(path)
    assert back.pos == pytest.approx(line.pos, abs=1e-3)
    assert back.length == pytest.approx(line.length, abs=1e-3)
    assert back.extra == pytest.approx(line.extra, rel=1e-5, abs=1e-5)
    assert np.all(np.diff(back.length) > 0)


def _ac_xz(road: RoadModel, d: np.ndarray) -> np.ndarray:
    """AC x/z of the centreline at distances `d`."""
    return FRAME.to_ac(np.interp(d, road.d, road.e), np.interp(d, road.d, road.n), np.interp(d, road.d, road.z))[:, [0, 2]]


def _main_road_radius(road: RoadModel, end_short_m: float) -> tuple[np.ndarray, np.ndarray]:
    """The radius main's centreline AI line carried (extra field 4): every metre from END_INSET_M, |curvature|
    smoothed over 2 points, capped at 1,000 m. Written out here so the route check can't drift from it."""
    from scipy.ndimage import gaussian_filter1d

    d = np.arange(ai_line.END_INSET_M, road.length - end_short_m, 1.0)
    k = gaussian_filter1d(np.abs(np.interp(d, road.d, road.curvature)), 2)
    return d, np.minimum(1.0 / np.maximum(k, 1e-6), 1000.0)


@pytest.mark.parametrize("reverse", [False, True])
def test_the_route_check_reads_the_centreline_every_metre_as_main_did(reverse):
    # A 6 m hairpin, 180 degrees, on a 6.4 m road: the racing line's radius there is far wider, and the road's
    # middle found from its points 5 m apart reads tight. The route check reads the centreline itself.
    road = road_from(bends((300, 0), (np.pi * 6, 6), (300, 0)), half=3.2, reverse=reverse)
    d, r = ai_line.road_radius(road, end_short_m=AI_END_SHORT_M)
    d_main, r_main = _main_road_radius(road, AI_END_SHORT_M)
    assert np.array_equal(d, d_main) and np.array_equal(r, r_main)
    assert r.min() == pytest.approx(6.0, abs=0.05)
    line = ai_line.build(road, FRAME, end_short_m=AI_END_SHORT_M)
    assert line.extra[:, 4].min() > 7.0  # the racing line's own radius
    xz = _ac_xz(road, d)
    rep = validate.Report()
    validate.bend_checks(rep, "t", line, radius=r, at=xz)
    assert f"t: tightest bend radius {r.min():.1f} m (needs >= 5 m)" in rep.passed and rep.ok
    # A tight bend found this way is reported where the road bends, from the centreline's own points.
    rep = validate.Report()
    validate.bend_checks(rep, "t", line, radius=r * 0.6, at=xz)  # 3.6 m: tighter than any real road
    i = int(np.argmin(r))
    assert rep.route_failed and rep.bends == [validate.TightBend("t", float(r[i] * 0.6), xz[i, 0], xz[i, 1])]


def test_a_road_at_5_m_passes_the_route_check_as_on_main():
    # A bend of 5.05 m radius: main's centreline line read it as 5.0 m and passed. The road's middle found from
    # the racing line's points 5 m apart would read it tighter; the route check doesn't use that any more.
    road = road_from(bends((300, 0), (np.pi * 5.05, 5.05), (300, 0)), half=3.2)
    d, r = ai_line.road_radius(road, end_short_m=AI_END_SHORT_M)
    assert np.array_equal(r, _main_road_radius(road, AI_END_SHORT_M)[1]) and r.min() >= validate.MIN_RADIUS_M
    line = ai_line.build(road, FRAME, end_short_m=AI_END_SHORT_M)
    xz = _ac_xz(road, d)
    rep = validate.Report()
    validate.bend_checks(rep, "t", line, radius=r, at=xz)
    assert rep.ok and not rep.bends
    assert validate.road_middle(line)[1].min() < validate.MIN_RADIUS_M  # what reading the AI file would have said


def _flat_road_surface(half=HALF):
    """A validate.RoadSurface wide enough for anything (the radius check doesn't look at it)."""
    pos = np.array([[-1e5, 0, -1e5], [1e5, 0, -1e5], [-1e5, 0, 1e5], [1e5, 0, 1e5]], dtype=float)
    return validate.RoadSurface([{"class": 2, "name": "1ROAD_000", "vertices": {"pos": pos},
                                  "indices": np.array([0, 2, 1, 1, 2, 3])}])


def _straight_line(offset: np.ndarray, half: float = HALF) -> ai_line.AiLine:
    """An AI line 1 m apart along a straight road (east, AC +x), `offset` metres left of its middle."""
    n = len(offset)
    x = np.arange(n, dtype=float)
    pos = np.column_stack([x, np.zeros(n), -offset])  # left of east is north, AC -z
    extra = np.zeros((n, ai_line.EXTRA_FIELDS))
    extra[:, 0] = 20.0
    extra[:, 4] = 1000.0
    extra[:, 5], extra[:, 6] = half - offset, half + offset
    extra[:, 13] = 1.0
    return ai_line.AiLine(pos=pos, length=x, extra=extra)


def test_the_validator_warns_of_a_line_that_weaves_and_fails_one_too_near_an_edge():
    s = np.arange(400.0)
    good = _straight_line(np.zeros(len(s)))
    rep = validate.Report()
    validate.racing_line_checks(rep, "t", good, _flat_road_surface(), validate.road_middle(good)[1])
    assert rep.failed == [] and len(rep.passed) == 3
    # Weaving 1 m either way every 60 m on a straight road: a 91 m radius where the road has none.
    weave = _straight_line(np.sin(2 * np.pi * s / 60))
    middle, radius = validate.road_middle(weave)
    assert np.abs(middle[:, 1]).max() < 1e-6 and radius.min() > 500  # the road's middle, found from the edges
    rep = validate.Report()
    validate.racing_line_checks(rep, "t", weave, _flat_road_surface(), radius)
    # Tighter than the road is a warning, never a failure (#44): the build goes on.
    assert rep.failed == [] and rep.route_failed == [] and rep.ok
    assert [w for w in rep.warnings if "bends tighter than the road" in w]
    assert not [p for p in rep.passed if "nowhere tighter than the road nearby" in p]
    # 0.8 m from the left edge.
    near = _straight_line(np.full(len(s), HALF - 0.8))
    rep = validate.Report()
    validate.racing_line_checks(rep, "t", near, _flat_road_surface(), validate.road_middle(near)[1])
    assert [f for f in rep.failed if "inside both road edges" in f]
    # A road 6 m wide in the mesh: 2.5 m left of the middle leaves 0.5 m, so the triangles don't reach 1 m past.
    narrow = np.array([[-10, 0, -3], [500, 0, -3], [-10, 0, 3], [500, 0, 3]], dtype=float)
    road = validate.RoadSurface([{"class": 2, "name": "1ROAD_000", "vertices": {"pos": narrow},
                                  "indices": np.array([0, 2, 1, 1, 2, 3])}])
    rep = validate.Report()
    off_mesh = _straight_line(np.full(len(s), 2.5), half=4.0)  # the file claims 1.5 m to spare; the mesh disagrees
    validate.racing_line_checks(rep, "t", off_mesh, road, validate.road_middle(off_mesh)[1])
    assert [f for f in rep.failed if "triangles reach" in f]
