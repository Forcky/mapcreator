"""The racing line: how far the AI line moves off the centreline, n(s), to take the bends out-in-out.

The maintainer's decision (4 Nov 2026): the AI uses the full width of the road, as on a closed road, not
the left lane.

Method: a minimum-curvature line, after Heilmeier, Wischnewski, Hermansdorfer, Betz, Lienkamp and Lohmann,
"Minimum curvature trajectory planning and control for an autonomous race car", Vehicle System Dynamics
58(10), 2020, pp. 1497-1527, doi:10.1080/00423114.2019.1631455. The line is the centreline moved sideways
by n(s) at stations along the road. Linearised around a known line, its curvature is a linear function
of n, so the summed squared curvature is a quadratic programme in n, bounded by the road's edges less a
margin, with the curvature itself limited too. Solving it again around the line it found shrinks the
linearisation error in tight bends.

Provenance: re-implemented here from the paper's description of the method. TUM's own implementation
(https://github.com/TUMFTM/global_racetrajectory_optimization) is LGPL-3.0, so none of its code was read,
ported or copied.

Where this differs from the paper:
- Curvature is the discrete curvature of the line's points (`STEP_M` apart), not a spline's, and it is
  linearised with its exact derivatives (`_curvature_rows`).
- The curvature limit is the road's own: nowhere may the line bend tighter than the centreline does within
  `LIMIT_NEAR_M` of it. Without it, the least summed squared curvature hugs the inside of a long bend,
  where the line's radius is the road's less the room it moved in by: 48 m on a 50 m bend, a slower apex
  than the centreline's.
- A point-to-point road has ends, and the line is held on the centreline at both: over the grid spots,
  the start gate and the hot-lap spawn (`HOLD_START_M`), and from `HOLD_BEFORE_FINISH_M` before the
  finish gate to the end of the line, where the AI is slowed to a stop (`ai_line.runoff_hints`). The
  bounds widen from 0 over `BLEND_M`, so the line leaves and rejoins the centreline smoothly.
- On a straight, any straight line has no curvature at all. A light pull toward the centre
  (`CENTRE_PULL_M`) picks the middle there, and lets the line drift out only in the run-up to a bend.
- The margin from the road's edges grows with the planned speed (`edge_margin`, #50). AC's AI tracks the
  line less closely the faster it goes: at 127 km/h it ran 1.0-1.3 m outside it (in game, 4-5 Nov), so a
  line 1.2 m from the edge put two wheels off before the turn-in. The speed depends on the line, so the
  caller solves, works out the speeds and the margins, and solves again (`ai_line.build`).
- On the inside of a tight bend the line stays within `INSIDE_FRACTION` of the bend's radius of the
  centreline, so its points never cross the bend's centre.
- It needs no QP library. The curvature limits go into an augmented Lagrangian (multipliers updated
  between rounds), and each round is minimised within the bounds by a projected Newton method (`_newton`)
  on a banded, five-wide Hessian: each step is one banded Cholesky solve, so a long road costs little more
  per station than a short one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solveh_banded
from scipy.ndimage import maximum_filter1d, minimum_filter1d

STEP_M = 2.0  # between optimisation stations
# From the road's edge to the line: half a car's width (0.85 m for a car 1.7 m wide) plus 0.35 m for error. The
# least margin, at or below MARGIN_SLOW_KMH (the maintainer's choice, #44).
EDGE_MARGIN_M = 1.2
# At MARGIN_FAST_KMH and above: the same 1.2 m plus the 1.3 m AC's AI ran outside the line at 127 km/h on
# Grasstree Hill (#50, the most of its measured 1.0-1.3 m). Between the two the margin rises in a straight line.
EDGE_MARGIN_FAST_M = 2.5
MARGIN_SLOW_KMH = 60.0
MARGIN_FAST_KMH = 120.0
HOLD_START_M = 60.0  # the line stays on the centreline this far from the layout's start (grid, start gate)
HOLD_BEFORE_FINISH_M = 30.0  # and from this far before the finish gate to its end (finish gate, run-off)
BLEND_M = 50.0  # over which the bounds widen from the centreline to the road's full width
CENTRE_PULL_M = 80.0  # length scale of the pull back to the centre on a straight
INSIDE_FRACTION = 0.5  # on the inside of a bend, the line stays within this share of its radius
LIMIT_NEAR_M = 20.0  # the line's curvature is limited to the centreline's tightest within this reach
# Bends gentler than this radius aren't limited: AC's speed there is capped anyway (ai_line.V_MAX), and the line
# needs room to curve a little on a straight as it moves across for the next bend.
LIMIT_FROM_RADIUS_M = 150.0
ITERATIONS = 3  # linearisations: the second and third correct the first in tight bends
AL_ROUNDS = 4  # multiplier updates for the curvature limits, per linearisation
AL_PENALTY = 100.0  # the limits' penalty, against 1 for the curvature itself
MAX_NEWTON_STEPS = 400  # per round; Grasstree Hill (12.8 km) takes about 120 in the first, then a handful
NEWTON_TOL_M = 1e-8


@dataclass
class Offsets:
    s: np.ndarray  # stations, metres from the layout's start
    n: np.ndarray  # metres left of the centreline (+) or right (-)
    lo: np.ndarray  # the bounds n was held within
    hi: np.ndarray
    limit: np.ndarray  # the curvature limit (1/m) at each station
    steps: int  # projected Newton steps, all rounds and linearisations together
    margin: np.ndarray | None = None  # the margin from each edge (m) the bounds were drawn with
    hold_until: float = 0.0  # the line is on the centreline up to here and from `hold_from` on
    hold_from: float = np.inf
    lam_hi: np.ndarray | None = None  # the curvature limits' multipliers, for a re-solve to start from
    lam_lo: np.ndarray | None = None


def edge_margin(speed: np.ndarray | float) -> np.ndarray:
    """The margin (m) from each road edge to the line at a planned speed (m/s): EDGE_MARGIN_M at or below
    MARGIN_SLOW_KMH, rising in a straight line to EDGE_MARGIN_FAST_M at MARGIN_FAST_KMH, and that above it."""
    x = (np.asarray(speed, dtype=float) * 3.6 - MARGIN_SLOW_KMH) / (MARGIN_FAST_KMH - MARGIN_SLOW_KMH)
    return EDGE_MARGIN_M + (EDGE_MARGIN_FAST_M - EDGE_MARGIN_M) * np.clip(x, 0.0, 1.0)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def bounds(s: np.ndarray, half_left: np.ndarray, half_right: np.ndarray, curvature: np.ndarray,
           hold_until: float, hold_from: float, step: float,
           margin: np.ndarray | float = EDGE_MARGIN_M) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) for n at each station: the road's edges less `margin` (never past the centreline: a road too
    narrow for it keeps the line in the middle), never past INSIDE_FRACTION of a bend's radius on its inside,
    and 0 (the centreline) before `hold_until` and after `hold_from`."""
    hi = np.maximum(half_left - margin, 0.0)
    lo = -np.maximum(half_right - margin, 0.0)
    k = np.abs(curvature)
    inside = INSIDE_FRACTION / np.maximum(k, 1e-9)
    hi = np.where(curvature > 0, np.minimum(hi, inside), hi)
    lo = np.where(curvature < 0, np.maximum(lo, -inside), lo)
    win = max(int(round(20.0 / step)), 1) | 1  # a bend's neighbours too: its tightest point may fall between stations
    hi = minimum_filter1d(hi, win, mode="nearest")
    lo = -minimum_filter1d(-lo, win, mode="nearest")
    blend = _smoothstep(np.minimum((s - hold_until) / BLEND_M, (hold_from - s) / BLEND_M))
    return lo * blend, hi * blend


def _menger(r: np.ndarray) -> np.ndarray:
    """Signed curvature (+ = turning left) at each interior point of a polyline: one over the radius of the
    circle through it and its two neighbours. Zero at both ends."""
    u, v, w = r[1:-1] - r[:-2], r[2:] - r[1:-1], r[2:] - r[:-2]
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    k = np.zeros(len(r))
    k[1:-1] = 2.0 * cross / np.maximum(np.hypot(*u.T) * np.hypot(*v.T) * np.hypot(*w.T), 1e-12)
    return k


def _curvature_rows(p: np.ndarray, nrm: np.ndarray, n: np.ndarray):
    """The line's curvature at each station as k0 + A n[j-1] + B n[j] + C n[j+1], linearised around the line
    with offsets `n`. The rows are its exact derivatives there, by central differences: each station moves
    one in three at a time, so three pairs of evaluations give every row. (A first version held the line's
    point spacing at the line it linearised around. That gets even the sign wrong for a line moving bodily to
    the inside of a bend, where its spacing shrinks, and such a line couldn't be held to the curvature limit.)"""
    m = len(p)
    k = _menger(p + nrm * n[:, None])
    A, B, C = np.zeros(m), np.zeros(m), np.zeros(m)
    idx = np.arange(m)
    h = 1e-3
    for colour in range(3):
        e = np.where(idx % 3 == colour, h, 0.0)
        dk = (_menger(p + nrm * (n + e)[:, None]) - _menger(p + nrm * (n - e)[:, None])) / (2 * h)
        A[1:] = np.where(idx[:-1] % 3 == colour, dk[1:], A[1:])  # row j moved by station j - 1
        B = np.where(idx % 3 == colour, dk, B)
        C[:-1] = np.where(idx[1:] % 3 == colour, dk[:-1], C[:-1])  # row j moved by station j + 1
    A[[0, -1]] = B[[0, -1]] = C[[0, -1]] = 0.0
    k0 = k - _rows_times(A, B, C, n)
    return k0, A, B, C


def _rows_times(A, B, C, x):
    """D x, D being the rows' tridiagonal matrix (A, B, C)."""
    y = B * x
    y[1:-1] += A[1:-1] * x[:-2] + C[1:-1] * x[2:]
    return y


def _rows_transposed(A, B, C, v):
    """D' v."""
    y = B * v
    y[:-1] += (A * v)[1:]
    y[1:] += (C * v)[:-1]
    return y


def _rows_gram(A, B, C, w):
    """D' diag(w) D as its main diagonal and first and second super-diagonals."""
    wA, wB, wC = w * A, w * B, w * C
    d0 = wB * B
    d0[:-1] += (wA * A)[1:]
    d0[1:] += (wC * C)[:-1]
    d1 = (wA * B)[1:] + (wB * C)[:-1]
    d2 = (wA * C)[1:-1]
    return d0, d1, d2


def _solve_free(d0, d1, d2, idx, rhs):
    """Solve the banded matrix (d0, d1, d2) restricted to the stations `idx` (sorted): still five wide."""
    m = len(idx)
    ab = np.zeros((3, m))  # upper form: ab[2] the diagonal, ab[1, 1:] and ab[0, 2:] the super-diagonals
    ab[2] = d0[idx]
    if m > 1:  # neighbours in idx: one or two stations apart in the road (or further, and then unlinked)
        gap = np.diff(idx)
        near1 = d1[np.minimum(idx[:-1], len(d1) - 1)]
        near2 = d2[np.minimum(idx[:-1], len(d2) - 1)]
        ab[1, 1:] = np.where(gap == 1, near1, np.where(gap == 2, near2, 0.0))
    if m > 2:
        gap2 = idx[2:] - idx[:-2]
        ab[0, 2:] = np.where(gap2 == 2, d2[np.minimum(idx[:-2], len(d2) - 1)], 0.0)
    return solveh_banded(ab, rhs, check_finite=False)


@dataclass
class _Round:
    """One augmented-Lagrangian round: sum(k^2)/2 + eps sum(n^2)/2, k = k0 + D n, with |k| <= limit as
    penalties around the multipliers lam_hi (k <= limit) and lam_lo (k >= -limit)."""
    k0: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    eps: float
    limit: np.ndarray
    lam_hi: np.ndarray
    lam_lo: np.ndarray
    mu: float

    def parts(self, x):
        k = self.k0 + _rows_times(self.A, self.B, self.C, x)
        over = np.maximum(self.lam_hi + self.mu * (k - self.limit), 0.0)
        under = np.maximum(self.lam_lo + self.mu * (-k - self.limit), 0.0)
        return k, over, under

    def value(self, x) -> float:
        k, over, under = self.parts(x)
        return 0.5 * (k @ k + self.eps * (x @ x)) + (over @ over + under @ under) / (2 * self.mu)

    def gradient_and_hessian(self, x):
        k, over, under = self.parts(x)
        g = _rows_transposed(self.A, self.B, self.C, k + over - under) + self.eps * x
        d0, d1, d2 = _rows_gram(self.A, self.B, self.C, 1.0 + self.mu * ((over > 0) | (under > 0)))
        return g, d0 + self.eps, d1, d2


def _newton(problem: _Round, lo, hi, x0) -> tuple[np.ndarray, int]:
    """Minimise `problem` for lo <= x <= hi by Bertsekas's projected Newton method ("Projected Newton methods
    for optimization problems with simple constraints", SIAM J. Control and Optimization 20(2), 1982). Each
    step takes a Newton step on the stations off their bounds (one banded solve) and a scaled gradient step on
    those held at a bound, projects onto the bounds, and backs off until the objective falls enough. Returns
    x, always within the bounds, and the steps taken."""
    fixed = hi - lo < 1e-12
    x = np.clip(x0, lo, hi)
    fx = problem.value(x)
    steps = 0
    for steps in range(1, MAX_NEWTON_STEPS + 1):
        g, d0, d1, d2 = problem.gradient_and_hessian(x)
        w = float(np.abs(x - np.clip(x - g / d0, lo, hi)).max())  # 0 exactly at the optimum
        if w < NEWTON_TOL_M:
            break
        delta = min(0.05, w)  # a station this close to a bound, pushed against it, is held there
        held = fixed | ((x <= lo + delta) & (g > 0)) | ((x >= hi - delta) & (g < 0))
        free = np.flatnonzero(~held)
        step = -g / d0
        if len(free):
            step[free] = _solve_free(d0, d1, d2, free, -g[free])
        a = 1.0
        while True:
            xn = np.clip(x + a * step, lo, hi)
            fn = problem.value(xn)
            # Armijo's rule as Bertsekas states it for this method.
            want = -a * (g[free] @ step[free]) + g[held] @ (x[held] - xn[held])
            if fx - fn >= 1e-4 * want or a < 1e-9:
                break
            a *= 0.5
        if fn > fx:  # no progress possible: stop where we are
            break
        x, fx = xn, fn
    return x, steps


def offsets(d: np.ndarray, e: np.ndarray, n: np.ndarray, heading: np.ndarray, curvature: np.ndarray,
            half_left: np.ndarray, half_right: np.ndarray, *, hold_until: float, hold_from: float,
            step: float = STEP_M, margin: np.ndarray | float = EDGE_MARGIN_M,
            start: Offsets | None = None) -> Offsets:
    """The minimum-curvature line's offsets at stations `step` apart, for a road sampled at distances `d`.

    `hold_until` and `hold_from` (metres along the road) bound the stretch the line may leave the centreline;
    outside it the offset is 0. `margin` is the room kept from each edge, one value or one per sample of the
    road (`edge_margin` of the planned speed there). `start`, a line already found at the same stations, is
    where the solve begins, with its multipliers: a re-solve after the margins change then takes far fewer
    steps (Grasstree Hill: 180 for the first solve, then 49 and 19).
    """
    s = np.arange(d[0], d[-1] + 1e-9, step)
    if len(s) < 5:
        z = np.zeros(len(s))
        return Offsets(s, z, z, z, z, 0, z, hold_until, hold_from)

    def f(a):
        return np.interp(s, d, a)

    h = f(np.unwrap(heading))
    p = np.column_stack([f(e) - e[0], f(n) - n[0]])  # near the origin, for precision
    nrm = np.column_stack([-np.sin(h), np.cos(h)])
    room = f(np.broadcast_to(np.asarray(margin, dtype=float), np.shape(d)))
    lo, hi = bounds(s, f(half_left), f(half_right), f(curvature), hold_until, hold_from, step, room)
    # The centreline's own curvature, measured as the line's is: the limit is its tightest nearby.
    centre = np.abs(_menger(p))
    limit = np.maximum(maximum_filter1d(centre, 2 * max(int(round(LIMIT_NEAR_M / step)), 1) + 1, mode="nearest"),
                       1.0 / LIMIT_FROM_RADIUS_M)
    eps = CENTRE_PULL_M ** -4
    x = np.zeros(len(s))
    lam_hi, lam_lo = np.zeros(len(s)), np.zeros(len(s))
    if start is not None and len(start.n) == len(s):
        # Its multipliers too: the curvature limits bind in the same bends, so the first round starts near them.
        x = np.clip(start.n, lo, hi)
        if start.lam_hi is not None:
            lam_hi, lam_lo = start.lam_hi, start.lam_lo
    total = 0
    if (hi - lo).max() > 1e-9:
        for _ in range(ITERATIONS):
            k0, A, B, C = _curvature_rows(p, nrm, x)
            for _ in range(AL_ROUNDS):
                problem = _Round(k0, A, B, C, eps, limit, lam_hi, lam_lo, AL_PENALTY)
                x, steps = _newton(problem, lo, hi, x)
                total += steps
                k, lam_hi, lam_lo = problem.parts(x)
                if np.all(np.abs(k) <= limit * 1.001):
                    break
    return Offsets(s, x, lo, hi, limit, total, room, hold_until, hold_from, lam_hi, lam_lo)
