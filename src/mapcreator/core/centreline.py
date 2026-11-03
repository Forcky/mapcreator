"""Turn a routed polyline into a smooth centreline sampled every metre.

OSM ways are straight lines between nodes (median 13 m apart on C324, but up
to 400 m on straights), so a raw polyline has a kink at every node and the
car feels each one. We fit a quintic smoothing spline in arc length whose
residual is bounded by `sigma_m` per point. That makes curvature continuous,
so bends ramp in and out like a designed road instead of arriving at a node.
We then evaluate it at true 1 m arc-length steps.

Tested on C324 (Nov 2026):
- A Savitzky-Golay filter left curvature spiking at every node, even with a
  50 m window.
- The spline at sigma 0.3 m stays within 1.3 m of OSM, inside OSM's own
  accuracy, and cuts the rate of curvature change roughly tenfold.

Turns at junctions come first. OSM draws a turn from one street into another
as a single corner node, and a spline held within a metre of it turns on the
spot (1.7 m radius at Richmond's Bridge Street into Wellington Street, Nov
2026). `fillet` rounds every such corner to `CORNER_RADIUS_M`, about the path
a car takes through a town junction, before the spline. A turn of more than
`REVERSAL_DEG` is the route going back the way it came, which no radius makes
drivable: it's left for validation to report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep

CORNER_RADIUS_M = 8.0
CORNER_MIN_DEG = 45.0  # a single node turning this much is a corner, not a node along a curve
REVERSAL_DEG = 150.0
ARC_STEP_M = 1.0


@dataclass
class Centreline:
    s: np.ndarray  # (n,) distance along, metres
    xy: np.ndarray  # (n, 2)
    heading: np.ndarray  # (n,) radians, east = 0, counter-clockwise
    curvature: np.ndarray  # (n,) 1/m, positive = turning left
    source_index: np.ndarray  # (n,) index of the input segment each sample came from
    max_deviation_m: float  # largest distance from the input polyline (rounded junction corners included)


def cumulative(xy: np.ndarray) -> np.ndarray:
    d = np.hypot(*np.diff(xy, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample(xy: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points every `step` metres along a polyline, their distances, and their source segment."""
    keep = np.concatenate([[True], np.hypot(*np.diff(xy, axis=0).T) > 1e-6])
    xy = xy[keep]
    seg_ids = np.flatnonzero(keep)[:-1] if keep.sum() > 1 else np.zeros(0, dtype=int)
    s = cumulative(xy)
    n = max(int(np.floor(s[-1] / step)), 1)
    ss = np.linspace(0.0, s[-1], n + 1)
    out = np.column_stack([np.interp(ss, s, xy[:, 0]), np.interp(ss, s, xy[:, 1])])
    seg = np.clip(np.searchsorted(s, ss, side="right") - 1, 0, len(xy) - 2)
    return out, ss, seg_ids[seg] if len(seg_ids) else seg




def distance_to_polyline(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    a = line[:-1]
    d = line[1:] - a
    len2 = np.maximum((d * d).sum(axis=1), 1e-12)
    out = np.full(len(points), np.inf)
    for i in range(0, len(points), 2048):  # chunked: points x segments can be large
        p = points[i : i + 2048, None, :]
        t = np.clip(((p - a) * d).sum(axis=2) / len2, 0, 1)
        proj = a + d * t[..., None]
        out[i : i + 2048] = np.hypot(*(proj - p).transpose(2, 0, 1)).min(axis=1)
    return out


def fillet(xy: np.ndarray, radius: float = CORNER_RADIUS_M, *, min_deg: float = CORNER_MIN_DEG,
           max_deg: float = REVERSAL_DEG) -> tuple[np.ndarray, np.ndarray]:
    """Round each corner node (a turn between `min_deg` and `max_deg` at one node) into a circular arc.

    The arc meets the road `radius * tan(turn / 2)` before and after the node, or less where the next
    corner or an end of the route is closer, in which case it is tighter than `radius`. Returns the new
    polyline and, for each of its segments, the input segment it came from.
    """
    xy = np.asarray(xy, dtype=float)
    keep = np.concatenate([[True], np.hypot(*np.diff(xy, axis=0).T) > 1e-6])
    pts = xy[keep]
    seg_of = np.flatnonzero(keep)[:-1]  # input segment for each segment of `pts`
    if len(pts) < 3:
        return xy, np.arange(max(len(xy) - 1, 0))
    seg = np.diff(pts, axis=0)
    h = np.arctan2(seg[:, 1], seg[:, 0])
    turn = np.abs(np.angle(np.exp(1j * np.diff(h))))  # at pts[1:-1]
    corners = [i + 1 for i in np.flatnonzero((turn >= np.radians(min_deg)) & (turn <= np.radians(max_deg)))]
    if not corners:
        return pts, seg_of
    s = cumulative(pts)
    want = {c: radius * np.tan(turn[c - 1] / 2) for c in corners}
    for a, b in zip(corners, corners[1:]):  # neighbouring corners share the road between them
        gap = s[b] - s[a]
        if want[a] + want[b] > gap:
            want[a], want[b] = min(want[a], gap / 2), min(want[b], gap / 2)
    want[corners[0]] = min(want[corners[0]], s[corners[0]] * 0.9)
    want[corners[-1]] = min(want[corners[-1]], (s[-1] - s[corners[-1]]) * 0.9)

    def at(dist):
        k = int(np.clip(np.searchsorted(s, dist, side="right") - 1, 0, len(pts) - 2))
        f = (dist - s[k]) / max(s[k + 1] - s[k], 1e-12)
        return pts[k] + (pts[k + 1] - pts[k]) * f, k

    out, src = [pts[0]], []
    done_to = 0.0
    for c in corners:
        t = want[c]
        p_in, k_in = at(s[c] - t)
        p_out, _ = at(s[c] + t)
        d_in, d_out = pts[c] - p_in, p_out - pts[c]
        if t < 0.05 or np.linalg.norm(d_in) < 1e-6 or np.linalg.norm(d_out) < 1e-6:
            continue
        d_in, d_out = d_in / np.linalg.norm(d_in), d_out / np.linalg.norm(d_out)
        sweep = float(np.arctan2(d_in[0] * d_out[1] - d_in[1] * d_out[0], d_in @ d_out))  # + = left
        if abs(sweep) < 1e-3:
            continue
        for m in range(1, len(pts)):  # input nodes before the arc
            if done_to < s[m] < s[c] - t - 1e-6:
                out.append(pts[m])
                src.append(seg_of[m - 1])
        out.append(p_in)
        src.append(seg_of[k_in])
        r = t / np.tan(abs(sweep) / 2)
        left = np.array([-d_in[1], d_in[0]]) * np.sign(sweep)
        centre = p_in + left * r
        a0 = np.arctan2(*(p_in - centre)[::-1])
        n = max(int(np.ceil(r * abs(sweep) / ARC_STEP_M)), 2)
        for j in range(1, n + 1):
            a = a0 + sweep * j / n
            out.append(centre + r * np.array([np.cos(a), np.sin(a)]))
            src.append(seg_of[c - 1] if j <= n // 2 else seg_of[c])
        out[-1] = p_out
        done_to = s[c] + t
    for m in range(1, len(pts)):
        if s[m] > done_to + 1e-6:
            out.append(pts[m])
            src.append(seg_of[m - 1])
    return np.array(out), np.array(src, dtype=np.int64)


def smooth(polyline: np.ndarray, *, step: float = 1.0, sigma_m: float = 0.3, dense_m: float = 2.0,
           corner_radius_m: float = CORNER_RADIUS_M) -> Centreline:
    polyline = np.asarray(polyline, dtype=float)
    rounded, rounded_src = fillet(polyline, corner_radius_m)
    dense, u, src = resample(rounded, dense_m)
    src = rounded_src[np.clip(src, 0, len(rounded_src) - 1)]
    if len(dense) < 8:
        raise ValueError("route too short to build a track")
    tck, _ = splprep([dense[:, 0], dense[:, 1]], u=u, s=len(dense) * sigma_m**2, k=5)

    # The spline's parameter is the input's arc length, not the smoothed curve's.
    # Tabulate true arc length on a fine grid, then invert it for exact 1 m steps.
    fine = np.linspace(u[0], u[-1], int(u[-1] / 0.25) + 2)
    fx, fy = splev(fine, tck)
    arc = cumulative(np.column_stack([fx, fy]))
    s = np.arange(0.0, arc[-1], step)
    uu = np.interp(s, arc, fine)

    x, y = splev(uu, tck)
    dx, dy = splev(uu, tck, der=1)
    ddx, ddy = splev(uu, tck, der=2)
    xy = np.column_stack([x, y])
    heading = np.unwrap(np.arctan2(dy, dx))
    curvature = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-12)
    source = src[np.clip(np.searchsorted(u, uu, side="right") - 1, 0, len(src) - 1)]
    dev = float(distance_to_polyline(xy, polyline).max())
    return Centreline(s=s, xy=xy, heading=heading, curvature=curvature, source_index=source, max_deviation_m=dev)


def normals(heading: np.ndarray) -> np.ndarray:
    """Unit vectors pointing to the LEFT of the direction of travel."""
    return np.column_stack([-np.sin(heading), np.cos(heading)])
