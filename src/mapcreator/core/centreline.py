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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splev, splprep


@dataclass
class Centreline:
    s: np.ndarray  # (n,) distance along, metres
    xy: np.ndarray  # (n, 2)
    heading: np.ndarray  # (n,) radians, east = 0, counter-clockwise
    curvature: np.ndarray  # (n,) 1/m, positive = turning left
    source_index: np.ndarray  # (n,) index of the input segment each sample came from
    max_deviation_m: float  # largest distance from the input polyline


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


def smooth(polyline: np.ndarray, *, step: float = 1.0, sigma_m: float = 0.3, dense_m: float = 2.0) -> Centreline:
    polyline = np.asarray(polyline, dtype=float)
    dense, u, src = resample(polyline, dense_m)
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
