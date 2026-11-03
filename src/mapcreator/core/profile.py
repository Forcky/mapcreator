"""Elevation profile and cross-slope along a centreline.

A 2 m DEM carries centimetre noise, and AC's physics feels every centimetre as
a corrugation. So the raw profile goes through a smoothing spline. Its
penalty on curvature acts as a low-pass filter with a set cutoff wavelength:
dips and crests longer than the cutoff survive, and DEM speckle doesn't.

Bridges and tunnels: the DEM is bare earth, so under a bridge it shows the
gully floor. Those stretches are bridged with a straight grade between their
ends before smoothing.

Coarse stretches of a fine grid. Tasmania's statewide 2 m DEM is about 65%
LiDAR; the rest is a coarser model resampled onto the same 2 m grid. There a
road that contours round a steep hillside reads the hillside: 51% over 10 m on
the Lyell Highway in the Franklin-Gordon (Nov 2026). The provider says which
stretches are LiDAR (Tasmania: theLIST's LiDAR index); `fine_stations` is the
fallback, judging by the ground's texture beside the road, which catches most
coarse stretches but takes some steep coarse hillsides for LiDAR. Then
`mixed_profile` treats the coarse stretches as a coarse DEM is treated (harder
smoothing and a grade cap), pinned to the fine profile where they meet it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import grey_closing


@dataclass
class Profile:
    z_raw: np.ndarray
    z: np.ndarray
    grade: np.ndarray  # dz/ds, fraction (0.10 = 10 %)
    vertical_curvature: np.ndarray  # d2z/ds2, 1/m
    filled: np.ndarray  # bool: raw height replaced (bridge, tunnel, no data)


@dataclass
class ProfileStats:
    length_m: float
    z_min: float
    z_max: float
    ascent_m: float
    descent_m: float
    max_grade_10m: float
    max_grade_50m: float
    mean_abs_grade: float
    rms_residual_m: float
    filled_m: float

    def as_dict(self) -> dict:
        return {k: round(v, 3) for k, v in self.__dict__.items()}



class NoElevation(ValueError):
    """The DEM has (almost) nothing under the route: the sea, or past the edge of its coverage."""


def lam_for_cutoff(cutoff_m: float) -> float:
    """Smoothing-spline penalty whose half-power point is at wavelength `cutoff_m`."""
    return (cutoff_m / (2 * math.pi)) ** 4


def fill_runs(s: np.ndarray, z: np.ndarray, mask: np.ndarray, pad_m: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Replace masked (and NaN) stretches, padded by `pad_m`, with a straight line between their ends."""
    bad = mask.copy() | ~np.isfinite(z)
    if pad_m > 0 and bad.any():
        idx = np.flatnonzero(bad)
        step = s[1] - s[0] if len(s) > 1 else 1.0
        pad = int(round(pad_m / step))
        for i in idx:
            bad[max(i - pad, 0) : i + pad + 1] = True
    good = ~bad
    if good.sum() < 2:
        raise NoElevation("there's no elevation data along this road, so it can't be built")
    out = z.copy()
    out[bad] = np.interp(s[bad], s[good], z[good])
    return out, bad


def narrow_dips(s: np.ndarray, z: np.ndarray, *, max_width_m: float = 20.0, min_depth_m: float = 0.4) -> np.ndarray:
    """Stations in dips narrower than `max_width_m` and deeper than `min_depth_m`.

    On a bare-earth DEM these are gullies under bridges that OSM didn't tag.
    C324's 5 m creek bridge (unbridged in OSM) shows as a 0.8 m dip.

    A grey-scale closing fills any dip narrower than its window. A smooth sag
    in the road only "fills" by about curvature x (window/2)² / 2, so at 20 m
    even a sharp 80 m wide, 3 m deep sag stays under the depth threshold.

    The ends are padded by linear extrapolation. Otherwise a slope running off
    the end of the route looks like a dip to the closing.
    """
    step = s[1] - s[0] if len(s) > 1 else 1.0
    size = max(int(round(max_width_m / step)) | 1, 3)
    zz = np.where(np.isfinite(z), z, np.nanmedian(z))
    if len(zz) < 2 * size:
        return np.zeros(len(zz), dtype=bool)
    k = np.arange(1, size + 1)
    left = zz[0] - (zz[size] - zz[0]) / size * k[::-1]
    right = zz[-1] + (zz[-1] - zz[-1 - size]) / size * k
    padded = np.concatenate([left, zz, right])
    closed = grey_closing(padded, size=size, mode="nearest")[size:-size]
    return (closed - zz) > min_depth_m


DEFAULT_CUTOFF_M = 40.0


def smooth_profile(s: np.ndarray, z_raw: np.ndarray, fill_mask: np.ndarray, *,
                   cutoff_m: float = DEFAULT_CUTOFF_M) -> Profile:
    fill_mask = fill_mask | narrow_dips(s, z_raw)
    z_filled, filled = fill_runs(s, z_raw, fill_mask)
    spline = make_smoothing_spline(s, z_filled, lam=lam_for_cutoff(cutoff_m))
    z = spline(s)
    grade = spline.derivative(1)(s)
    vc = spline.derivative(2)(s)
    return Profile(z_raw=z_raw, z=z, grade=grade, vertical_curvature=vc, filled=filled)


COARSE_MAX_GRADE = 0.15

# Ground texture: LiDAR ground is rough at 2 m (ditches, banks, a road's own cut and fill); a coarse model
# resampled onto a 2 m grid is smooth between its own cells. Measured as the RMS Laplacian of 40 m patches
# beside the road on a 2 m lattice (Nov 2026): every patch on Grasstree Hill and Brinktop Road (LiDAR) above
# 0.011 /m, medians 0.021-0.046; three quarters of the Lyell Highway through the Franklin-Gordon under 0.009,
# median 0.005.
FINE_TEXTURE = 0.008
TEXTURE_STEP_M = 100.0
TEXTURE_PATCH_M = 40.0
TEXTURE_GRID_M = 2.0
TEXTURE_OFFSET_M = 15.0  # patch centres this far either side of the centreline: beside the road, not on it
COARSE_LIKE_M = 25.0  # how the coarse parts behave: their smoothing cutoff is 8 cells of this


def limit_grade(s: np.ndarray, z: np.ndarray, max_grade: float = COARSE_MAX_GRADE) -> np.ndarray:
    """Cap the grade while keeping both end heights.

    On a coarse DEM a hillside road samples the hillside, not the road, and
    reads 30%+ grades. Integrate the clipped grade, then spread the height it
    loses or gains back over the stations that weren't clipped, so the climb
    between the ends is unchanged.
    """
    ds = np.diff(s)
    g = np.diff(z) / ds
    clipped = np.clip(g, -max_grade, max_grade)
    for _ in range(20):
        missing = (z[-1] - z[0]) - float(np.sum(clipped * ds))
        free = np.abs(clipped) < max_grade - 1e-9
        if abs(missing) < 1e-6 or not free.any():
            break
        clipped[free] += missing / float(np.sum(ds[free]))
        clipped = np.clip(clipped, -max_grade, max_grade)
    return z[0] + np.concatenate([[0.0], np.cumsum(clipped * ds)])


def ground_texture(sample, x: np.ndarray, y: np.ndarray, *, grid: float = TEXTURE_GRID_M,
                   size: float = TEXTURE_PATCH_M) -> np.ndarray:
    """RMS Laplacian (1/m) of a `size` square patch of the DEM centred on each (x, y), on a `grid` lattice.
    NaN where the patch has no data."""
    g = np.arange(-size / 2, size / 2 + 1e-9, grid)
    gx, gy = np.meshgrid(g, g)
    px, py = np.asarray(x, float)[:, None, None] + gx, np.asarray(y, float)[:, None, None] + gy
    h = np.asarray(sample(px.ravel(), py.ravel()), dtype=float).reshape(px.shape)
    lap = (h[:, 1:-1, :-2] + h[:, 1:-1, 2:] + h[:, :-2, 1:-1] + h[:, 2:, 1:-1] - 4 * h[:, 1:-1, 1:-1]) / grid ** 2
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(lap).any(axis=(1, 2))
        out = np.full(len(h), np.nan)
        out[ok] = np.sqrt(np.nanmean(lap[ok] ** 2, axis=(1, 2)))
    return out


def fine_stations(s: np.ndarray, xy: np.ndarray, heading: np.ndarray, sample, *,
                  step: float = TEXTURE_STEP_M) -> np.ndarray:
    """For each station: does the DEM have real detail here (LiDAR), or is it a coarse model on a fine grid?

    Judged every `step` metres from patches on both sides of the road (the rougher side counts, so a lake or
    a paddock on one side doesn't decide it), then by the majority of each probe and its neighbours.
    """
    probes = np.arange(step / 2, max(float(s[-1]), step / 2) + 1e-9, step)
    idx = np.clip(np.searchsorted(s, probes), 0, len(s) - 1)
    nrm = np.column_stack([-np.sin(heading[idx]), np.cos(heading[idx])]) * TEXTURE_OFFSET_M
    left = ground_texture(sample, *(xy[idx] + nrm).T)
    right = ground_texture(sample, *(xy[idx] - nrm).T)
    with np.errstate(invalid="ignore"):
        fine = np.fmax(left, right) >= FINE_TEXTURE
    if len(fine) >= 3:
        f = np.r_[fine[0], fine, fine[-1]].astype(int)
        fine = (f[:-2] + f[1:-1] + f[2:]) >= 2
    return fine[np.clip((s // step).astype(int), 0, len(fine) - 1)]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where `mask` is true."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def mixed_profile(s: np.ndarray, z_raw: np.ndarray, fill_mask: np.ndarray, fine: np.ndarray, *,
                  coarse_cutoff_m: float = 8.0 * COARSE_LIKE_M, max_grade: float = COARSE_MAX_GRADE) -> Profile:
    """`smooth_profile` where the DEM is fine. Coarse stretches are smoothed harder and grade-capped, with
    their ends pinned to the fine profile (or left where they are at an end of the road), then one light pass
    smooths the joins."""
    base = smooth_profile(s, z_raw, fill_mask)
    if fine.all():
        return base
    heavy = smooth_profile(s, z_raw, fill_mask, cutoff_m=coarse_cutoff_m)
    z = base.z.copy()
    n = len(s)
    for a, b in _runs(~fine):
        lo, hi = max(a - 1, 0), min(b, n - 1)
        if hi - lo < 2:
            continue
        seg = heavy.z[lo:hi + 1].copy()
        t = (s[lo:hi + 1] - s[lo]) / max(s[hi] - s[lo], 1e-9)
        start = base.z[lo] if a > 0 else seg[0]
        end = base.z[hi] if b < n else seg[-1]
        seg += (start - seg[0]) * (1 - t) + (end - seg[-1]) * t
        z[lo:hi + 1] = limit_grade(s[lo:hi + 1], seg, max_grade)
    out = smooth_profile(s, z, np.zeros(n, dtype=bool))
    out.z_raw, out.filled = z_raw, base.filled
    return out


def _windowed_grade(s: np.ndarray, z: np.ndarray, window_m: float) -> float:
    step = s[1] - s[0]
    k = max(int(round(window_m / step)), 1)
    if len(z) <= k:
        return float(abs(z[-1] - z[0]) / max(s[-1] - s[0], 1e-9))
    return float(np.max(np.abs(z[k:] - z[:-k]) / (s[k:] - s[:-k])))


def stats(s: np.ndarray, p: Profile) -> ProfileStats:
    dz = np.diff(p.z)
    resid = (p.z_raw - p.z)[~p.filled & np.isfinite(p.z_raw)]
    step = s[1] - s[0] if len(s) > 1 else 1.0
    return ProfileStats(
        length_m=float(s[-1]),
        z_min=float(p.z.min()),
        z_max=float(p.z.max()),
        ascent_m=float(dz[dz > 0].sum()),
        descent_m=float(-dz[dz < 0].sum()),
        max_grade_10m=_windowed_grade(s, p.z, 10.0),
        max_grade_50m=_windowed_grade(s, p.z, 50.0),
        mean_abs_grade=float(np.mean(np.abs(p.grade))),
        rms_residual_m=float(np.sqrt(np.mean(resid**2))) if len(resid) else 0.0,
        filled_m=float(p.filled.sum() * step),
    )


def cross_slope(
    s: np.ndarray,
    z_left: np.ndarray,
    z_right: np.ndarray,
    half_widths: np.ndarray,
    *,
    cutoff_m: float = 60.0,
    clamp: float = 0.08,
) -> np.ndarray:
    """Cross-fall from heights sampled either side of the road. Positive = left edge higher."""
    raw = (z_left - z_right) / (2 * half_widths)
    ok = np.isfinite(raw)
    if ok.sum() < 4:
        return np.zeros_like(s)
    raw = np.where(ok, raw, np.interp(s, s[ok], raw[ok]))
    smooth = make_smoothing_spline(s, raw, lam=lam_for_cutoff(cutoff_m))(s)
    return np.clip(smooth, -clamp, clamp)
