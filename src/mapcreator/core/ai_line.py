"""The AI driving line, `ai/fast_lane.ai` (version 7): a racing line across the road's full width.

Layout (little-endian), as read by Content Manager's AcTools:

    int32 version=7, int32 pointCount N, int32 lapTime=0, int32 sampleCount=0
    N x { f32 pos[3], f32 cumulativeLength, int32 id }
    int32 extraCount=N
    N x 18 f32 { speed, gas, brake, obsoleteLatG, radius, sideLeft, sideRight, camber,
                 direction, normal[3], length, forward[3], tag, grade }
    int32 hasGrid=0  (the game builds and caches the grid itself)

The line takes the bends out-in-out over the whole road, as on a closed road (`racing_line`), and keeps to
the centreline over the grid and from just before the finish to its end. Its points are about 5 m apart along
the line itself (`AI_POINT_SPACING_M`), each 0.4 m above the road surface where it stands, crown and banking
included. The game casts a ray down from each point to find the surface, and points on the surface crashed
it in an open-source generator's testing. Speeds are m/s from a grip-limited profile on the line's own
curvature, with forward (acceleration) and backward (braking) passes, so the AI brakes before bends instead
of in them, and braking shares the grip with cornering, so it slows before a bend tightens; the profile is
worked out every metre and read at each point. The racing line's margin from the edges is set from the same
profile without the V_MAX cap (`racing_line.edge_margin`), since AC's AI doesn't keep to our speeds, so the
line is solved again once they're known. sideLeft and sideRight are the distances from the line to each road edge: Content Manager draws the road's outline from them
(AcTools.Render's AiLaneObject).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from mapcreator.core import racing_line
from mapcreator.core.furniture import CAM_SIDE_FRICTION
from mapcreator.core.layout import AI_END_SHORT_M, AI_GRAVEL_PACE, AI_RUNOFF_RAMP, RoadModel, finish_distance
from mapcreator.core.scene import LocalFrame

LIFT_M = 0.4
END_INSET_M = 1.5
MU = 1.05
G = 9.81
V_MAX = 36.0  # m/s, about 130 km/h
BRAKE = 8.0  # m/s^2: the straight-line braking the speed profiles plan with on a level road
EXTRA_FIELDS = 18
# Between the points written to fast_lane.ai. AC measures its spline from the points in float32, and the
# rounding inflates the length it measures by about 1/spacing^2: on Lyell Highway (63 km, reaching 21.7 km
# from the local origin) 1 m points made it 32.8% long, 2 m 8.8%, 5 m 1.3% and 10 m 0.3% (in game, 4 Nov).
# Every spline fraction (the run-off hints, AC's progress) moved with it. 10 m points cut a chord 0.8 m inside
# a 15 m hairpin; at 5 m the AI drove Lyell's 15.8 m and 13.9 m hairpins within 1.0 m of the line.
AI_POINT_SPACING_M = 5.0
PROFILE_M = 1.0  # speeds, gas, brake, radius and grade are worked out this finely, then read at each point
FINE_M = 0.25  # the line is traced this finely to measure its length and curvature
SMOOTH_M = 2.0  # curvature and grade are smoothed over this many metres


@dataclass
class AiLine:
    pos: np.ndarray  # (n, 3) AC axes
    length: np.ndarray  # cumulative metres
    extra: np.ndarray  # (n, 18)
    # Not in the file: where each point stands on the road, for the track stage and the tests.
    station: np.ndarray | None = None  # metres along the centreline
    offset: np.ndarray | None = None  # metres left of the centreline (+) or right (-)
    centre_radius: np.ndarray | None = None  # the centreline's radius there, as extra field 4 is the line's


def radius_of(curvature: np.ndarray, step: float = PROFILE_M) -> np.ndarray:
    """The radius AC is given (extra field 4) from curvature sampled `step` metres apart: |curvature| smoothed over
    SMOOTH_M, capped at 1,000 m."""
    return np.minimum(1.0 / np.maximum(gaussian_filter1d(np.abs(curvature), SMOOTH_M / step), 1e-6), 1000.0)


def road_radius(road: RoadModel, *, end_short_m: float = END_INSET_M) -> tuple[np.ndarray, np.ndarray]:
    """The centreline's radius every PROFILE_M over the AI line's stretch (END_INSET_M from the layout's start to
    `end_short_m` before its end), as distances along the road and radii. This is the road's bend, which the route
    check judges (`validate.bend_checks`), not the racing line's; it's the radius main's centreline AI line carried
    at the same stations."""
    d = np.arange(END_INSET_M, road.length - end_short_m, PROFILE_M)
    return d, radius_of(np.interp(d, road.d, road.curvature))


def speed_profile(d: np.ndarray, curvature: np.ndarray, grade: np.ndarray, v_max: float = V_MAX, *,
                  mu: float = MU) -> np.ndarray:
    """Speeds (m/s) at distances `d`, PROFILE_M apart, along a line of the given curvature and grade, never over
    `v_max`, for a car with `mu` g of grip.

    Grip-limited in the bends, then a forward pass for acceleration and a backward pass for braking. Braking
    shares the tyres' grip with cornering (a friction ellipse: mu G turning, BRAKE braking): turning at
    lateral acceleration a, a car can brake at only sqrt(1 - (a / (mu G))^2) of its straight-line rate. So
    where a bend tightens, the line slows before it, while there's still grip to spare, instead of arriving at
    the grip limit for the radius it's on and needing to brake for the tighter one ahead with none left (#50).
    Acceleration is limited by traction alone, as before: AC's AI sets its own throttle."""
    radius = radius_of(curvature)
    grip = mu * G
    # Plain lists: these loops index one element at a time, which numpy arrays make several times slower.
    v = np.minimum(np.sqrt(grip * radius), v_max).tolist()
    ds, r, gr = np.diff(d).tolist(), radius.tolist(), np.asarray(grade, dtype=float).tolist()
    for i in range(1, len(v)):  # traction-limited acceleration, harder uphill
        a = max(4.5 - G * gr[i], 0.8)
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a * ds[i - 1]))
    for i in range(len(v) - 2, -1, -1):  # braking, easier uphill, less in a bend
        lateral = min(v[i + 1] ** 2 / r[i], grip)
        b = max(BRAKE + G * gr[i], 3.0) * math.sqrt(1.0 - (lateral / grip) ** 2)
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * b * ds[i]))
    return np.array(v)


def spline_fraction(length: float, *, end_short_m: float = AI_END_SHORT_M, line: AiLine | None = None):
    """Distance along a layout of `length` metres to a fraction of its AI spline (0 at the start of the line, 1 at
    its end): the space every hint is written in. Given the `line`, the fractions are of its own length: a racing
    line is shorter than the road, so a place on the road falls at a later fraction of it."""
    first, last = END_INSET_M, length - end_short_m
    span = max(last - first, 1.0)

    def frac(d: float) -> float:
        if line is None or line.station is None:
            return (d - first) / span
        return float(np.interp(d, line.station, line.length) / max(line.length[-1], 1.0))

    return frac


def gravel_hints(unsealed, length: float, finish: float, *, end_short_m: float = AI_END_SHORT_M,
                 pace: float = AI_GRAVEL_PACE, line: AiLine | None = None) -> list[tuple[float, float, float]]:
    """The AI's pace over unsealed road, as one (start, end, pace) section per run, in the run-off hints'
    fractions (`spline_fraction`). `unsealed` is the runs as (from, to) metres along the layout, apart from each
    other. Each is cut to the AI line's start and to the finish: from the finish on, the run-off ramp sets the
    pace (`runoff_hints`), so the two never overlap, and a run that's all after the finish gets none. A run under
    `section.MIN_SURFACE_RUN_M` was already merged into its neighbours (`section.settle`); only the cut at the
    finish can leave one shorter."""
    frac = spline_fraction(length, end_short_m=end_short_m, line=line)
    out = []
    for a, b in sorted(unsealed):
        a, b = max(a, END_INSET_M), min(b, finish)
        if b > a:
            out.append((frac(a), frac(b), pace))
    return out


def pace_hints(length: float, finish: float, *, unsealed=(), bends=(), end_short_m: float = AI_END_SHORT_M,
               line: AiLine | None = None) -> list[tuple[float, float, float]]:
    """Every hint for a layout, in order along its spline: AI_GRAVEL_PACE over its unsealed runs up to the finish
    (`gravel_hints`) and the risky bends' hints (`bend_hints`, which need the `line`), then the ramp over the
    run-off (`runoff_hints`). Where a bend's hint and the gravel's overlap, the lower pace holds there: the two are
    never applied on top of each other. Up to the finish only, as the gravel's. With no unsealed road and no risky
    bend, the run-off ramp alone, exactly as before gravel had hints."""
    before = gravel_hints(unsealed, length, finish, end_short_m=end_short_m, line=line)
    if bends and line is not None:
        total = max(float(line.length[-1]), 1.0)
        cut = spline_fraction(length, end_short_m=end_short_m, line=line)(finish)
        steps = [(a / total, min(b / total, cut), p) for bend in bends for a, b, p in bend.sections(total)]
        before = _lowest(before + [(a, b, p) for a, b, p in steps if b > a])
    return before + runoff_hints(length, finish, end_short_m=end_short_m, line=line)


def _lowest(sections) -> list[tuple[float, float, float]]:
    """(start, end, pace) sections that may overlap, as sections that don't: the lowest pace wherever two overlap,
    and neighbours of the same pace joined. In order along the line."""
    cuts = sorted({x for a, b, _ in sections for x in (a, b)})
    out: list[tuple[float, float, float]] = []
    for a, b in zip(cuts, cuts[1:]):
        paces = [p for s0, s1, p in sections if s0 <= a and s1 >= b]
        if not paces:
            continue
        p = min(paces)
        if out and out[-1][1] == a and out[-1][2] == p:
            out[-1] = (out[-1][0], b, p)
        else:
            out.append((a, b, p))
    return out


def runoff_hints(length: float, finish: float, *, end_short_m: float = AI_END_SHORT_M,
                 ramp=AI_RUNOFF_RAMP, line: AiLine | None = None) -> list[tuple[float, float, float]]:
    """The AI's pace after the finish, as (start, end, pace) sections in fractions of its spline (0 at the start
    of the line, 1 at its end) for a layout of `length` metres with the finish at `finish`. Contiguous, from the
    finish to the end of the line, the pace falling section by section. A run-off too short for a section skips
    it.

    Given the `line`, the fractions are of its own length (`spline_fraction`)."""
    frac = spline_fraction(length, end_short_m=end_short_m, line=line)
    last = length - end_short_m
    out, start = [], finish
    for before_end, pace in ramp:
        end = max(start, last - before_end)
        if end > start:
            out.append((frac(start), frac(end), pace))
            start = end
    if out:
        out[-1] = (out[-1][0], 1.0, out[-1][2])
    return out


# Risky bends (#63). AC's AI picks its own corner speed from the line's radius and the car's grip, whatever our
# speed field says (#52): about 0.9 g in an Abarth 500, so 110-120 km/h in Lyell Highway's 109-135 m left-hander at
# 0.7 km (in game, 9 Nov). It doesn't allow for a crest, where the car goes light, or for a descent, where it must
# brake as it turns: that bend starts just over a crest and runs down an 8.5% grade with no banking, and on lap 1
# it put most of an 8-car field off the road. A hint is what the AI obeys, and in a bend it works (docs/ac-format.md).
# Nor does it allow for a crest in its braking zone: braking for a bend at the rate it counts on, with the tyres
# carrying only part of the car over the crest, it arrives faster than it planned (Lyell Highway reverse at 14.0 km,
# a 44-66 m bend 40 m past a 220-280 m crest and down 15%: all 8 cars off at 86-99 km/h, in game on 10 Nov).
BEND_GRIP = 0.9  # g: what AC's AI corners at on a level, unbanked bend, so the grip it counts on everywhere
BEND_TOP_KMH = racing_line.MARGIN_TOP_KMH  # the fastest the AI goes on a straight (#52)
# Hints act in bends: on the run-off the same hints slowed the AI to 12-15 km/h in bends of 19-210 m but did
# nothing on a straight of 287-1,000 m (in game, 4 Nov). So wider bends get none.
BEND_MAX_RADIUS_M = 250.0
# A bend gets a hint where the AI should drive slower than this share of its own corner speed for at least
# BEND_MIN_S. At a pace of 0.9 the grip left is 0.81 of what the AI counts on, 1.7 m/s^2 short at 0.9 g, and in a
# second that slides a car 0.84 m wide (a t^2 / 2): inside the 1.3 m the line's margin already keeps for the AI's
# own drift at speed (racing_line.EDGE_MARGIN_FAST_M less EDGE_MARGIN_M). A smaller or shorter shortfall moves it less.
BEND_TRIGGER = 0.9
BEND_MIN_S = 1.0
BEND_RAMP_M = 25.0  # a hint eases in and out in two steps this long either side, a third of the way each
BEND_EDGE_M = 20.0  # metres at each end of the line left out: the crest's derivatives aren't reliable there
# Never slower than the bend's advisory speed, the speed a curve warning sign gives an ordinary driver:
# v = sqrt(127 R (e + f)) km/h, with the side friction f the curve markers use. With no banking that's
# sqrt(0.35 / BEND_GRIP), 0.62 of the AI's own corner speed.
BEND_ADVISORY_FRICTION = CAM_SIDE_FRICTION


@dataclass(frozen=True)
class BendHint:
    """A bend where AC's AI would corner faster than the road allows, and the hint written for it. Distances are
    metres along the AI line itself (`AiLine.length`), where it's furthest over the grip left (`at`) and what it
    is short of there."""

    start: float  # the full pace holds from here...
    end: float  # ...to here, easing in before and out after (`sections`)
    pace: float
    at: float
    radius_m: float
    crest_radius_m: float  # inf where there's no crest
    descent: float  # the grade down, as a fraction (0 uphill)
    banking: float  # the road's banking into the bend, as a fraction; negative is banked away from it
    ai_kmh: float  # the AI's likely speed (`bend_paces`)
    road_kmh: float  # the speed the grip left allows at the car's speed
    car_kmh: float | None = None  # the car's speed there, over ai_kmh where a crest cut its braking (`arrival_speeds`)

    def sections(self, total: float | None = None) -> list[tuple[float, float, float]]:
        """(start, end, pace) in metres along the line: two steps in, the full pace, two steps out, within the
        line's `total` length."""
        p, r = self.pace, BEND_RAMP_M
        ease = [round(p + (1 - p) * k / 3, 2) for k in (2, 1)]
        out = [(self.start - 2 * r, self.start - r, ease[0]), (self.start - r, self.start, ease[1]),
               (self.start, self.end, p), (self.end, self.end + r, ease[1]), (self.end + r, self.end + 2 * r, ease[0])]
        hi = float("inf") if total is None else total
        return [(max(a, 0.0), min(b, hi), q) for a, b, q in out if min(b, hi) > max(a, 0.0) and q < 1]


def vertical_curvature(s: np.ndarray, height: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The line's vertical curvature (1/m; + is a sag, - a crest) every metre of `grid`, from its heights at
    distances `s`: smoothed over 10 m before and after, as the validator's crest check reads it."""
    h = uniform_filter1d(np.interp(grid, s, height), 10, mode="nearest")
    return uniform_filter1d(np.gradient(np.gradient(h, grid), grid), 10, mode="nearest")


def arrival_speeds(d: np.ndarray, planned: np.ndarray, grade: np.ndarray, crest: np.ndarray) -> np.ndarray:
    """The car's speed (m/s) at distances `d` where AC's AI plans the speeds `planned`, braking with the grip it
    actually has. Where the plan slows, the car brakes at the straight-line rate the plan counts on (BRAKE), times
    the share of the car the tyres carry over a crest, 1 - v^2 crest / G at the car's speed (`crest` in 1/m), plus
    g times the grade (uphill helps, downhill hinders), until it's back on the plan. With no crest that's as hard as
    the plan's own braking or harder, so it keeps to the plan exactly; over a crest in a braking zone it arrives
    faster than planned. It doesn't share that braking with cornering, as the plan does, so in a bend's entry it
    can catch the plan up: the overshoot is the least the crest costs."""
    v = np.asarray(planned, dtype=float).tolist()
    out = list(v)
    ds, gr, k = np.diff(d).tolist(), np.asarray(grade, dtype=float).tolist(), np.asarray(crest, dtype=float).tolist()
    for i in range(len(v) - 1):
        if out[i] > v[i + 1]:
            load = max(1.0 - out[i] ** 2 * k[i] / G, 0.0)
            b = max(BRAKE * load + G * gr[i], 0.0)
            out[i + 1] = max(math.sqrt(max(out[i] ** 2 - 2 * b * ds[i], 0.0)), v[i + 1])
    return np.array(out)


def bend_paces(line: AiLine, banking: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The pace a hint should hold the AI to, every PROFILE_M along the line (1 where it needs none), with what it
    comes from, as arrays by name. Read from the line as written to fast_lane.ai, so the validator works it out
    from the file exactly as the track stage did: its own radius, turn, grade and heights. `banking` is the road's
    rise to the left (+) or right beside each point, per metre across it, crown aside (`road_banking`); none is level.

    The AI's likely speed is a speed profile at BEND_GRIP up to BEND_TOP_KMH, with its braking and acceleration
    (`speed_profile`), so a bend it can't reach at its corner speed isn't counted. The car follows it, except that
    over a crest in a braking zone it brakes with less grip than the AI counts on and arrives faster
    (`arrival_speeds`). At the car's speed v the grip left is the AI's own BEND_GRIP g less what goes elsewhere: a
    crest takes load off the tyres, 1 - v^2 / (R_v g) of it (R_v the crest's vertical radius), and a descent takes
    g times its grade to hold the speed on the brakes; banking into the bend adds g times its fall. Against what the
    bend asks, v^2 / R, where the grip left falls short the car should be slower by the square root of the
    shortfall. A hint scales the speed the AI works out from the line's curvature, sqrt(BEND_GRIP g R), so the pace
    is the speed the grip allows over that corner speed; and where the car arrives faster than the AI planned, lower
    by as much again, since the AI's slower target leaves the car overshooting it in the same proportion. Where the
    car keeps to the plan, that's the AI's own speed and the pace exactly as it was. The unloading is reckoned at
    the car's speed before the hint slows it, which leaves a little in hand. Only in bends of up to
    BEND_MAX_RADIUS_M, and never below the curve's advisory speed (BEND_ADVISORY_FRICTION)."""
    s = np.asarray(line.length, dtype=float)
    grid = np.append(np.arange(0.0, s[-1], PROFILE_M), s[-1]) if len(s) > 1 else s.copy()
    radius = np.maximum(np.interp(grid, s, line.extra[:, 4]), 1.0)
    turn = np.sign(np.interp(grid, s, line.extra[:, 8]))
    grade = np.interp(grid, s, line.extra[:, 17])
    crest = np.maximum(-vertical_curvature(s, np.asarray(line.pos, dtype=float)[:, 1], grid), 0.0)
    descent = np.maximum(-grade, 0.0)
    into = np.zeros(len(grid)) if banking is None else -turn * np.interp(grid, s, banking)
    speed = speed_profile(grid, turn / radius, grade, v_max=BEND_TOP_KMH / 3.6, mu=BEND_GRIP)
    car = arrival_speeds(grid, speed, grade, crest)
    left = G * (BEND_GRIP * (1.0 - car ** 2 * crest / G) - descent + into)
    need = car ** 2 / radius
    road = car * np.sqrt(np.clip(left / need, 0.0, 1.0))
    corner = np.sqrt(BEND_GRIP * G * radius)
    floor = np.sqrt((np.maximum(into, 0.0) + BEND_ADVISORY_FRICTION) / BEND_GRIP)
    pace = np.where((radius <= BEND_MAX_RADIUS_M) & (left < need), np.maximum(road / corner * (speed / car), floor),
                    1.0)
    pace = np.minimum(pace, 1.0)
    edge = (grid < BEND_EDGE_M) | (grid > grid[-1] - BEND_EDGE_M)
    pace[edge] = 1.0
    return {"s": grid, "pace": pace, "speed": speed, "car": car, "road": road, "radius": radius, "crest": crest,
            "descent": descent, "banking": into}


def bend_hints(line: AiLine, banking: np.ndarray | None = None) -> list[BendHint]:
    """The bends where AC's AI would corner faster than the road allows (`bend_paces`), each with its hint: the
    pace held from where it first falls under BEND_TRIGGER to where it's back over it, if that lasts BEND_MIN_S at
    the car's speed. Where the car overshoots the AI's plan, that starts where it first comes into the bend faster
    than the grip allows, so the hint covers the bend from its entry. The pace is the lowest in that stretch,
    rounded down to 0.01. Stretches that close on each other stay apart here; `pace_hints` writes the lower pace
    where their ramps overlap."""
    b = bend_paces(line, banking)
    s, pace = b["s"], b["pace"]
    low = np.r_[False, pace < BEND_TRIGGER, False].astype(np.int8)
    out = []
    for i0, i1 in zip(np.flatnonzero(np.diff(low) == 1), np.flatnonzero(np.diff(low) == -1)):
        if np.sum(PROFILE_M / np.maximum(b["car"][i0:i1], 1.0)) < BEND_MIN_S:
            continue
        i = i0 + int(np.argmin(pace[i0:i1]))
        crest = float(b["crest"][i])
        out.append(BendHint(
            start=float(s[i0]), end=float(s[i1 - 1]), pace=math.floor(float(pace[i]) * 100 + 1e-9) / 100,
            at=float(s[i]), radius_m=float(b["radius"][i]), crest_radius_m=1.0 / crest if crest > 0 else math.inf,
            descent=float(b["descent"][i]), banking=float(b["banking"][i]), ai_kmh=float(b["speed"][i] * 3.6),
            road_kmh=float(b["road"][i] * 3.6), car_kmh=float(b["car"][i] * 3.6)))
    return out


def road_banking(road: RoadModel, line: AiLine) -> np.ndarray:
    """The road's banking beside each point of the line, as its rise to the left per metre (- to the right): the
    superelevation, half the difference of its two falls, so the crown doesn't count. The crown falls into a bend
    on its inside half and away from it on its outside half, and the AI and a car beside it use both."""
    return (np.interp(line.station, road.d, road.slope_right) - np.interp(line.station, road.d, road.slope_left)) / 2


class _Traced:
    """A line `off` metres from the road's centreline (or on it, with `off` None), traced every FINE_M from
    `first` to `last` metres along the road, with its planned speed, curvature and grade every PROFILE_M."""

    def __init__(self, road: RoadModel, centre: CubicSpline, off: racing_line.Offsets | None, first: float,
                 last: float):
        self.road, self.centre, self.off = road, centre, off
        self.lateral = None
        if off is not None and len(off.s) > 1 and np.any(off.hi > off.lo):
            self.lateral = CubicSpline(off.s, off.n)
        # Trace the line finely for its length and heading.
        self.d_fine = np.append(np.arange(first, last, FINE_M), last)
        r_fine, self.a_fine = self.at(self.d_fine)
        self.arc_fine = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(r_fine, axis=0).T))])
        tangent = np.gradient(r_fine, self.arc_fine, axis=0)
        self.heading_fine = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        self.total = float(self.arc_fine[-1])
        # Speed, radius and grade every PROFILE_M along the line.
        self.arc_p = np.append(np.arange(0.0, self.total, PROFILE_M), self.total)
        self.d_p = np.interp(self.arc_p, self.arc_fine, self.d_fine)
        self.curv_p = np.interp(self.arc_p, self.arc_fine, np.gradient(self.heading_fine, self.arc_fine))
        a_p = np.interp(self.arc_p, self.arc_fine, self.a_fine)
        n = len(self.arc_p)
        grade_p = np.gradient(self.surface(self.d_p, a_p), self.arc_p) if n > 1 else np.zeros(n)
        self.grade_p = gaussian_filter1d(grade_p, SMOOTH_M / PROFILE_M)
        self.v_p = speed_profile(self.arc_p, self.curv_p, self.grade_p)

    def margin_speeds(self) -> np.ndarray:
        """The speeds the cars may reach, every PROFILE_M, for the edge margins (#52): the same profile, but up to
        MARGIN_TOP_KMH instead of V_MAX. AC's AI works out its own speeds and doesn't stop at V_MAX (in game it ran
        up to 220 km/h against our 130); above MARGIN_TOP_KMH the margin no longer grows."""
        return speed_profile(self.arc_p, self.curv_p, self.grade_p, v_max=racing_line.MARGIN_TOP_KMH / 3.6)

    def at(self, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Points on the line at stations `d` (east/north about the road's first point), and their offsets from
        the centreline."""
        p = self.centre(d)
        t = self.centre(d, 1)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        a = np.zeros(len(d))
        if self.lateral is not None:
            # Exactly on the centreline wherever it's held there, even between a station with room and the next
            # one without; elsewhere the spline through the stations.
            off = self.off
            free = (d > off.hold_until) & (d < off.hold_from) & (np.interp(d, off.s, off.hi - off.lo) > 0)
            a = np.where(free, self.lateral(d), 0.0)
        return p + np.column_stack([-t[:, 1], t[:, 0]]) * a[:, None], a

    def surface(self, d: np.ndarray, a: np.ndarray) -> np.ndarray:
        """The road's height under the line: it falls from the crown to each edge, by `slope_*` per metre."""
        road = self.road
        return (np.interp(d, road.d, road.z) - np.where(a >= 0, np.interp(d, road.d, road.slope_left) * a,
                                                         -np.interp(d, road.d, road.slope_right) * a))


# The racing line keeps a margin from the edges for its speed (racing_line.edge_margin), and the speed depends
# on the line. So the margins are first set from the centreline's speeds, the line solved, its own speeds
# worked out and the margins set again from them, and the line solved again from where it was: up to MARGIN_ROUNDS
# times, until no margin needs to grow by more than MARGIN_TOL_M. The last round only widens margins. On Grasstree
# Hill, Brinktop Road and Lower Swamp Road that takes both rounds, and the line written then comes within 2 cm of
# the margin its own speed asks for wherever that margin holds it.
MARGIN_ROUNDS = 2
MARGIN_TOL_M = 0.05


def _racing(road: RoadModel, centre: CubicSpline, first: float, last: float) -> _Traced:
    """The racing line, traced, with each edge margin suited to the line's own planned speed."""
    def solve(margin, start=None):
        return racing_line.offsets(road.d, road.e, road.n, road.heading, road.curvature, road.half_left,
                                   road.half_right, hold_until=racing_line.HOLD_START_M,
                                   hold_from=finish_distance(road.length) - racing_line.HOLD_BEFORE_FINISH_M,
                                   margin=margin, start=start)

    def margins(traced: _Traced) -> np.ndarray:
        return np.interp(road.d, traced.d_p, racing_line.edge_margin(traced.margin_speeds()))

    # The racing line is a little faster than the centreline, so this is a first guess, a little narrow.
    margin = margins(_Traced(road, centre, None, first, last))
    traced = _Traced(road, centre, solve(margin), first, last)
    for i in range(MARGIN_ROUNDS):
        if traced.lateral is None:  # held on the centreline everywhere: nothing to set
            break
        want = margins(traced)
        if np.all(want <= margin + MARGIN_TOL_M):
            break
        margin = want if i < MARGIN_ROUNDS - 1 else np.maximum(margin, want)
        traced = _Traced(road, centre, solve(margin, traced.off), first, last)
    return traced


def build(road: RoadModel, frame: LocalFrame, step: float = AI_POINT_SPACING_M, *,
          end_short_m: float = END_INSET_M, racing: bool = True) -> AiLine:
    """The AI line, from END_INSET_M after the layout's start to `end_short_m` before its end, with points about
    `step` metres apart: the racing line, or with `racing=False` the centreline."""
    first, last = END_INSET_M, road.length - end_short_m
    e0, n0 = float(road.e[0]), float(road.n[0])
    centre = CubicSpline(road.d, np.column_stack([road.e - e0, road.n - n0]))
    line = _racing(road, centre, first, last) if racing else _Traced(road, centre, None, first, last)
    arc_fine, d_fine, heading_fine, total = line.arc_fine, line.d_fine, line.heading_fine, line.total

    # Gas, brake and radius every PROFILE_M along the line, from its speeds...
    arc_p, curv_p, grade_p, v_p = line.arc_p, line.curv_p, line.grade_p, line.v_p
    dv = np.gradient(v_p, arc_p) if len(arc_p) > 1 else np.zeros(len(arc_p))  # per metre
    gas_p = np.where(dv >= -0.01, np.clip(0.4 + 6 * dv, 0, 1), 0.0)
    brake_p = np.where(dv < -0.01, np.clip(-4 * dv, 0, 1), 0.0)
    radius_p = radius_of(curv_p)

    # ...and read at the points written, evenly spaced from the line's first station to its last.
    count = max(int(np.ceil(total / step - 1e-9)), 1)
    arc = np.linspace(0.0, total, count + 1)
    d = np.interp(arc, arc_fine, d_fine)
    r, a = line.at(d)
    heading = np.interp(arc, arc_fine, heading_fine)
    grade = np.interp(arc, arc_p, grade_p)
    pos = frame.to_ac(e0 + r[:, 0], n0 + r[:, 1], line.surface(d, a) + LIFT_M)

    fwd = np.column_stack([np.cos(heading), grade, -np.sin(heading)])
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    side = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    normal = np.cross(fwd, side)
    normal *= np.sign(normal[:, 1:2])

    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    length_to_next = np.append(seg, seg[-1] if len(seg) else step)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])

    extra = np.zeros((len(d), EXTRA_FIELDS))
    extra[:, 0] = np.interp(arc, arc_p, v_p)
    extra[:, 1] = np.interp(arc, arc_p, gas_p)
    extra[:, 2] = np.interp(arc, arc_p, brake_p)
    extra[:, 4] = np.interp(arc, arc_p, radius_p)
    extra[:, 5] = np.interp(d, road.d, road.half_left) - a  # from the line to the left edge
    extra[:, 6] = np.interp(d, road.d, road.half_right) + a  # and to the right edge
    extra[:, 8] = np.sign(np.interp(arc, arc_p, curv_p))
    extra[:, 9:12] = normal
    extra[:, 12] = length_to_next
    extra[:, 13:16] = fwd
    extra[:, 17] = grade
    road_step = float(np.median(np.diff(road.d))) if len(road.d) > 1 else 1.0
    return AiLine(pos=pos, length=cumulative, extra=extra, station=d, offset=a,
                  centre_radius=tightest_near(d, road.d, radius_of(road.curvature, road_step)))


def tightest_near(points: np.ndarray, grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """At each of `points` (ascending), the smallest of `values` (sampled at ascending `grid`) within half the
    spacing either side of it: 5 m apart, the points would otherwise step over the tightest metre of a bend."""
    out = np.interp(points, grid, values)
    if len(points) < 2:
        return out
    half = np.diff(points) / 2
    sel = (grid >= points[0] - half[0]) & (grid <= points[-1] + half[-1])
    owner = np.searchsorted(points[:-1] + half, grid[sel])
    np.minimum.at(out, owner, values[sel])
    return out


def write(path: Path, line: AiLine) -> None:
    n = len(line.pos)
    out = [struct.pack("<4i", 7, n, 0, 0)]
    pts = np.zeros(n, dtype=[("pos", "<f4", 3), ("len", "<f4"), ("id", "<i4")])
    pts["pos"], pts["len"], pts["id"] = line.pos, line.length, np.arange(n)
    out.append(pts.tobytes())
    out.append(struct.pack("<i", n))
    out.append(line.extra.astype("<f4").tobytes())
    out.append(struct.pack("<i", 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(out))


def read(path: Path) -> AiLine:
    data = path.read_bytes()
    version, n, _lap, _samples = struct.unpack_from("<4i", data, 0)
    if version != 7:
        raise ValueError(f"fast_lane.ai version {version}, expected 7")
    o = 16
    pts = np.frombuffer(data, dtype=[("pos", "<f4", 3), ("len", "<f4"), ("id", "<i4")], count=n, offset=o)
    o += 20 * n
    (extra_n,) = struct.unpack_from("<i", data, o)
    o += 4
    extra = np.frombuffer(data, dtype="<f4", count=extra_n * EXTRA_FIELDS, offset=o).reshape(extra_n, EXTRA_FIELDS)
    o += 72 * extra_n
    (has_grid,) = struct.unpack_from("<i", data, o)
    o += 4
    if has_grid != 0 or o != len(data):
        raise ValueError("unexpected grid or trailing bytes in fast_lane.ai")
    return AiLine(pos=pts["pos"].astype(float), length=pts["len"].astype(float), extra=extra.astype(float))
