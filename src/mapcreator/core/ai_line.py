"""The AI driving line, `ai/fast_lane.ai` (version 7), computed from the centreline.

Layout (little-endian), as read by Content Manager's AcTools:

    int32 version=7, int32 pointCount N, int32 lapTime=0, int32 sampleCount=0
    N x { f32 pos[3], f32 cumulativeLength, int32 id }
    int32 extraCount=N
    N x 18 f32 { speed, gas, brake, obsoleteLatG, radius, sideLeft, sideRight, camber,
                 direction, normal[3], length, forward[3], tag, grade }
    int32 hasGrid=0  (the game builds and caches the grid itself)

Points sit 0.4 m above the road. The game casts a ray down from each point to
find the surface, and points on the surface crashed it in an open-source
generator's testing. Speeds are m/s from a grip-limited profile with forward
(acceleration) and backward (braking) passes, so the AI brakes before bends
instead of in them.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from mapcreator.core.layout import AI_END_SHORT_M, AI_RUNOFF_RAMP, RoadModel
from mapcreator.core.scene import LocalFrame

LIFT_M = 0.4
END_INSET_M = 1.5
MU = 1.05
G = 9.81
V_MAX = 36.0  # m/s, about 130 km/h
EXTRA_FIELDS = 18


@dataclass
class AiLine:
    pos: np.ndarray  # (n, 3) AC axes
    length: np.ndarray  # cumulative metres
    extra: np.ndarray  # (n, 18)


def speed_profile(d: np.ndarray, curvature: np.ndarray, grade: np.ndarray) -> np.ndarray:
    k = gaussian_filter1d(np.abs(curvature), 2)
    radius = np.minimum(1.0 / np.maximum(k, 1e-6), 1000.0)
    v = np.minimum(np.sqrt(MU * G * radius), V_MAX)
    ds = np.diff(d)
    for i in range(1, len(v)):  # traction-limited acceleration, harder uphill
        a = max(4.5 - G * grade[i], 0.8)
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a * ds[i - 1]))
    for i in range(len(v) - 2, -1, -1):  # braking, easier uphill
        b = max(8.0 + G * grade[i], 3.0)
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * b * ds[i]))
    return v


def runoff_hints(length: float, finish: float, *, end_short_m: float = AI_END_SHORT_M,
                 ramp=AI_RUNOFF_RAMP) -> list[tuple[float, float, float]]:
    """The AI's pace after the finish, as (start, end, pace) sections in fractions of its spline (0 at the start
    of the line, 1 at its end) for a layout of `length` metres with the finish at `finish`. Contiguous, from the
    finish to the end of the line, the pace falling section by section. A run-off too short for a section skips
    it."""
    first, last = END_INSET_M, length - end_short_m
    span = max(last - first, 1.0)
    out, start = [], finish
    for before_end, pace in ramp:
        end = max(start, last - before_end)
        if end > start:
            out.append(((start - first) / span, (end - first) / span, pace))
            start = end
    if out:
        out[-1] = (out[-1][0], 1.0, out[-1][2])
    return out


def build(road: RoadModel, frame: LocalFrame, step: float = 1.0, *, end_short_m: float = END_INSET_M) -> AiLine:
    """The AI line, from END_INSET_M after the layout's start to `end_short_m` before its end."""
    d = np.arange(END_INSET_M, road.length - end_short_m, step)

    def f(a):
        return np.interp(d, road.d, a)

    e, n, z = f(road.e), f(road.n), f(road.z)
    heading = f(np.unwrap(road.heading))
    curv, grade = f(road.curvature), f(road.grade)
    pos = frame.to_ac(e, n, z + LIFT_M)

    v = speed_profile(d, curv, grade)
    dv = np.gradient(v)
    gas = np.where(dv >= -0.01, np.clip(0.4 + 6 * dv, 0, 1), 0.0)
    brake = np.where(dv < -0.01, np.clip(-4 * dv, 0, 1), 0.0)
    radius = np.minimum(1.0 / np.maximum(gaussian_filter1d(np.abs(curv), 2), 1e-6), 1000.0)

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
    extra[:, 0] = v
    extra[:, 1] = gas
    extra[:, 2] = brake
    extra[:, 4] = radius
    extra[:, 5] = f(road.half_left)
    extra[:, 6] = f(road.half_right)
    extra[:, 8] = np.sign(curv)
    extra[:, 9:12] = normal
    extra[:, 12] = length_to_next
    extra[:, 13:16] = fwd
    extra[:, 17] = grade
    return AiLine(pos=pos, length=cumulative, extra=extra)


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
