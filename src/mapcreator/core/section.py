"""Road cross-section: carriageway width and the slope of each half.

Width comes from OSM: the `width` tag, else `lanes` x a lane width, else a
default for the road class. Tag coverage is patchy. On C324 the longest
8.5 km way has no `lanes` tag, so the defaults matter. Changes in width are
eased over `WIDTH_EASE_M` so a run boundary never makes a step in the road edge.

Slopes are "fall from the centre to that edge" (positive = edge lower). A
straight road keeps a crown of `CROWN` on both halves. Where the DEM shows the
road banked (superelevation `e`, positive = left edge higher), the high
side's slope ramps from +crown to -crown as |e| reaches the crown. Beyond
that, both halves form one plane with slope |e|. This is the usual way of
removing adverse crown in a curve, and every shape along the way is
continuous.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.ndimage import uniform_filter1d

CROWN = 0.025
MAX_SUPERELEVATION = 0.07
WIDTH_EASE_M = 60.0
MIN_WIDTH, MAX_WIDTH = 3.0, 20.0

# Lane width (m) by OSM highway class; two-way roads default to 2 lanes.
LANE_WIDTH = {
    "motorway": 3.5, "trunk": 3.5, "primary": 3.4, "secondary": 3.2, "tertiary": 3.0,
    "unclassified": 2.9, "residential": 2.9, "road": 2.9, "living_street": 2.6,
}


def _parse_metres(value: str | None) -> float | None:
    if not value:
        return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(m|metres?|meters?)?\s*$", value)
    return float(m.group(1)) if m else None


def width_from_tags(tags: dict) -> tuple[float, str]:
    """Carriageway width in metres, and where it came from."""
    w = _parse_metres(tags.get("width"))
    if w and MIN_WIDTH <= w <= MAX_WIDTH:
        return w, "width"
    hw = tags.get("highway", "")
    base = hw.removesuffix("_link")
    lane_w = LANE_WIDTH.get(base, 3.0)
    lanes = tags.get("lanes")
    if lanes and lanes.isdigit() and 1 <= int(lanes) <= 6:
        n = int(lanes)
        # A single-lane one-way slip is still wider than a painted lane.
        return max(n * lane_w, 4.0), "lanes"
    oneway = tags.get("oneway") in ("yes", "true", "1", "-1")
    n = 1 if (oneway and hw.endswith("_link")) else 2
    return max(n * lane_w, 4.0), "class"


def eased(s: np.ndarray, values: np.ndarray, ease_m: float = WIDTH_EASE_M) -> np.ndarray:
    step = s[1] - s[0] if len(s) > 1 else 1.0
    size = max(int(round(ease_m / step)), 1)
    return uniform_filter1d(values.astype(float), size=size, mode="nearest")


def half_slopes(e: np.ndarray, crown: float = CROWN) -> tuple[np.ndarray, np.ndarray]:
    """Fall from centre to the left and right edges, given superelevation e (+ = left edge higher)."""
    e = np.clip(e, -MAX_SUPERELEVATION, MAX_SUPERELEVATION)
    a = np.abs(e)
    high = np.where(a <= crown, crown - 2 * a, -a)
    low = np.where(a <= crown, crown, a)
    left = np.where(e >= 0, high, low)
    right = np.where(e >= 0, low, high)
    return left, right
