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
of in them; the profile is worked out every metre and read at each point. sideLeft and sideRight are the
distances from the line to each road edge: Content Manager draws the road's outline from them
(AcTools.Render's AiLaneObject).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from mapcreator.core import racing_line
from mapcreator.core.layout import AI_END_SHORT_M, AI_RUNOFF_RAMP, RoadModel, finish_distance
from mapcreator.core.scene import LocalFrame

LIFT_M = 0.4
END_INSET_M = 1.5
MU = 1.05
G = 9.81
V_MAX = 36.0  # m/s, about 130 km/h
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


def speed_profile(d: np.ndarray, curvature: np.ndarray, grade: np.ndarray) -> np.ndarray:
    """Speeds (m/s) at distances `d`, PROFILE_M apart, along a line of the given curvature and grade."""
    v = np.minimum(np.sqrt(MU * G * radius_of(curvature)), V_MAX)
    ds = np.diff(d)
    for i in range(1, len(v)):  # traction-limited acceleration, harder uphill
        a = max(4.5 - G * grade[i], 0.8)
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a * ds[i - 1]))
    for i in range(len(v) - 2, -1, -1):  # braking, easier uphill
        b = max(8.0 + G * grade[i], 3.0)
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * b * ds[i]))
    return v


def runoff_hints(length: float, finish: float, *, end_short_m: float = AI_END_SHORT_M,
                 ramp=AI_RUNOFF_RAMP, line: AiLine | None = None) -> list[tuple[float, float, float]]:
    """The AI's pace after the finish, as (start, end, pace) sections in fractions of its spline (0 at the start
    of the line, 1 at its end) for a layout of `length` metres with the finish at `finish`. Contiguous, from the
    finish to the end of the line, the pace falling section by section. A run-off too short for a section skips
    it.

    Given the `line`, the fractions are of its own length: a racing line is shorter than the road, so a place on
    the road falls at a later fraction of it."""
    first, last = END_INSET_M, length - end_short_m
    span = max(last - first, 1.0)

    def frac(d: float) -> float:
        if line is None or line.station is None:
            return (d - first) / span
        return float(np.interp(d, line.station, line.length) / max(line.length[-1], 1.0))

    out, start = [], finish
    for before_end, pace in ramp:
        end = max(start, last - before_end)
        if end > start:
            out.append((frac(start), frac(end), pace))
            start = end
    if out:
        out[-1] = (out[-1][0], 1.0, out[-1][2])
    return out


def build(road: RoadModel, frame: LocalFrame, step: float = AI_POINT_SPACING_M, *,
          end_short_m: float = END_INSET_M, racing: bool = True) -> AiLine:
    """The AI line, from END_INSET_M after the layout's start to `end_short_m` before its end, with points about
    `step` metres apart: the racing line, or with `racing=False` the centreline."""
    first, last = END_INSET_M, road.length - end_short_m
    e0, n0 = float(road.e[0]), float(road.n[0])
    centre = CubicSpline(road.d, np.column_stack([road.e - e0, road.n - n0]))
    lateral = off = None
    if racing:
        off = racing_line.offsets(road.d, road.e, road.n, road.heading, road.curvature, road.half_left,
                                  road.half_right, hold_until=racing_line.HOLD_START_M,
                                  hold_from=finish_distance(road.length) - racing_line.HOLD_BEFORE_FINISH_M)
        if len(off.s) > 1 and np.any(off.hi > off.lo):
            lateral = CubicSpline(off.s, off.n)

    def at(d):
        """Points on the line at stations `d` (east/north about e0/n0), and their offsets from the centreline."""
        p = centre(d)
        t = centre(d, 1)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        a = np.zeros(len(d))
        if lateral is not None:
            # Exactly on the centreline wherever it's held there; elsewhere the spline through the stations.
            a = np.where(np.interp(d, off.s, off.hi - off.lo) > 0, lateral(d), 0.0)
        return p + np.column_stack([-t[:, 1], t[:, 0]]) * a[:, None], a

    def surface(d, a):
        """The road's height under the line: it falls from the crown to each edge, by `slope_*` per metre."""
        return (np.interp(d, road.d, road.z) - np.where(a >= 0, np.interp(d, road.d, road.slope_left) * a,
                                                         -np.interp(d, road.d, road.slope_right) * a))

    # Trace the line finely for its length and heading.
    d_fine = np.append(np.arange(first, last, FINE_M), last)
    r_fine, a_fine = at(d_fine)
    arc_fine = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(r_fine, axis=0).T))])
    tangent = np.gradient(r_fine, arc_fine, axis=0)
    heading_fine = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    total = float(arc_fine[-1])

    # Speeds, gas, brake, radius and grade every PROFILE_M along the line...
    arc_p = np.append(np.arange(0.0, total, PROFILE_M), total)
    d_p = np.interp(arc_p, arc_fine, d_fine)
    curv_p = np.interp(arc_p, arc_fine, np.gradient(heading_fine, arc_fine))
    a_p = np.interp(arc_p, arc_fine, a_fine)
    grade_p = np.gradient(surface(d_p, a_p), arc_p) if len(arc_p) > 1 else np.zeros(len(arc_p))
    grade_p = gaussian_filter1d(grade_p, SMOOTH_M / PROFILE_M)
    v_p = speed_profile(arc_p, curv_p, grade_p)
    dv = np.gradient(v_p, arc_p) if len(arc_p) > 1 else np.zeros(len(arc_p))  # per metre
    gas_p = np.where(dv >= -0.01, np.clip(0.4 + 6 * dv, 0, 1), 0.0)
    brake_p = np.where(dv < -0.01, np.clip(-4 * dv, 0, 1), 0.0)
    radius_p = radius_of(curv_p)

    # ...and read at the points written, evenly spaced from the line's first station to its last.
    count = max(int(np.ceil(total / step - 1e-9)), 1)
    arc = np.linspace(0.0, total, count + 1)
    d = np.interp(arc, arc_fine, d_fine)
    r, a = at(d)
    heading = np.interp(arc, arc_fine, heading_fine)
    grade = np.interp(arc, arc_p, grade_p)
    pos = frame.to_ac(e0 + r[:, 0], n0 + r[:, 1], surface(d, a) + LIFT_M)

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
