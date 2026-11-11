"""Road cross-section: surface, carriageway width and the slope of each half.

**Surface.** Each OSM run is sealed or unsealed by its `surface` tag (`surface_class`). Runs with no tag, or a
value outside both lists, take the class of the longer stretch beside them. Then any stretch shorter than
`MIN_SURFACE_RUN_M` takes its longer neighbour's class (`settle`), so a 10 m concrete culvert on a gravel road
stays gravel. Bridges get no exception. Tasmania tags surfaces well: 98-100% of drivable ways in the Derwent
Valley and the Huon carry one.

**Width.** The `width` tag (raised to `MIN_WIDTH` if under it), else `lanes` x a lane width, else `narrow=yes`,
else a default for the road class, narrower for unsealed roads. Tag coverage is patchy: on C324 the longest
8.5 km way has no `lanes` tag, and about 98% of unsealed ways carry neither `lanes` nor `width`, so the defaults
decide. Changes in width are eased over `WIDTH_EASE_M` so a run boundary never makes a step in the road edge.

**Slopes** are "fall from the centre to that edge" (positive = edge lower). A straight road keeps a crown on
both halves: `CROWN` on a seal and `CROWN_UNSEALED` on gravel, eased over `WIDTH_EASE_M` across a change. Where
the DEM shows the road banked (superelevation `e`, positive = left edge higher), the high side's slope ramps
from +crown to -crown as |e| reaches the crown. Beyond that, both halves form one plane with slope |e|. This is
the usual way of removing adverse crown in a curve, and every shape along the way is continuous.

**Narrow roads** get less camber: at most 4% under 5 m wide, rising to the full 7% at 6.4 m (`camber_limit`),
and they always keep a little crown (`crown_kept`), so a single-lane road never becomes one tilted plane.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter1d

CROWN = 0.025
# NATSPEC AUS-SPEC 0052 (geometric rural road design, unsealed): "Minimum cross-fall: More than 4%".
CROWN_UNSEALED = 0.045
MAX_SUPERELEVATION = 0.07
# The camber limit by carriageway width: NARROW_CAMBER at or under NARROW_CAMBER_WIDTH_M, MAX_SUPERELEVATION at or
# over FULL_CAMBER_WIDTH_M, straight between (`camber_limit`). On a 4 m road the DEM's two camber samples are 3 m
# apart and catch the hillside and the cut and fill beside the road: over a third of Lower Swamp Road read 7% or
# more, and nearly two thirds of it was one tilted plane with no crown. 4% sits under the gravel crown (4.5%), so
# narrow gravel keeps its crown at any camber; 6.4 m is the widest two-lane class default, so roads that wide or
# wider are as they were.
NARROW_CAMBER = 0.04
NARROW_CAMBER_WIDTH_M = 5.0
FULL_CAMBER_WIDTH_M = 6.4
# A narrow road keeps at least this fall either side of its centre at any camber (`half_slopes`' `keep`), fading
# to none at FULL_CAMBER_WIDTH_M: a single-lane seal (crown 2.5%) under the 4% limit keeps a crown rather than
# becoming one tilted plane. At the limit it then has the same section as narrow gravel: 4.5% and -3.5%.
NARROW_CROWN_KEPT = 0.005
WIDTH_EASE_M = 60.0
MIN_WIDTH, MAX_WIDTH = 3.0, 20.0

# Lane width (m) by OSM highway class; two-way roads default to 2 lanes.
LANE_WIDTH = {
    "motorway": 3.5, "trunk": 3.5, "primary": 3.4, "secondary": 3.2, "tertiary": 3.0,
    "unclassified": 2.9, "residential": 2.9, "road": 2.9, "living_street": 2.6,
}

SEALED, UNSEALED = "sealed", "unsealed"
# OSM `surface` values by class. Anything else (`wood` on a bridge, say) counts as untagged and inherits.
SEALED_SURFACES = frozenset({"asphalt", "paved", "concrete", "chipseal", "sealed", "paving_stones", "sett", "metal"})
UNSEALED_SURFACES = frozenset({"unpaved", "gravel", "fine_gravel", "compacted", "pebblestone", "dirt", "earth",
                               "ground", "mud", "sand", "rock", "grass", "woodchips"})
MIN_SURFACE_RUN_M = 50.0

# Unsealed carriageway (the running surface, not the formation) by class, with no `width` or `lanes` tag.
# NATSPEC AUS-SPEC 0052 asks for at least a 6.0 m formation, verges included, for a single-lane unsealed road and
# 7.5 m for two lanes; our ribbon adds 2 m of grass verge outside the carriageway on each side.
UNSEALED_WIDTH = {"motorway": 6.5, "trunk": 6.5, "primary": 6.5, "secondary": 6.0, "tertiary": 5.5,
                  "unclassified": 5.0, "residential": 5.0, "road": 5.0, "track": 4.0}
UNSEALED_DEFAULT_M = 5.0
UNSEALED_LANE_M = 3.0  # per tagged lane on gravel: lanes=2 makes 6.0 m
# `narrow=yes` with no width. OSM's lanes page gives as its example "a narrow two-way road without marked lanes
# and with an estimated width of four metres".
NARROW_M = 4.5
MIN_LANES_WIDTH_M = 4.0  # a single lane, even a one-way slip, is still wider than a painted lane
# Gravel narrower than this, as mapped, with no `lanes` tag, is one lane: a pair of wheel paths and a grass strip.
SINGLE_LANE_BELOW_M = 5.5

# The road's look, by section style code: the material its mesh uses (`mesh.ROAD_TEXTURE` gives the texture).
# Codes 2 and 3 are unsealed: their meshes are 1GRAVEL_### rather than 1ROAD_###.
STYLES = ("road", "road_plain", "gravel", "gravel_narrow")


def style_code(unsealed: bool, single_lane: bool) -> int:
    return 2 * int(unsealed) + int(single_lane)


def is_unsealed(code):
    """Whether a style code (or an array of them) is unsealed."""
    return np.asarray(code) >= 2


def _parse_metres(value: str | None) -> float | None:
    if not value:
        return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(m|metres?|meters?)?\s*$", value)
    return float(m.group(1)) if m else None


def width_from_tags(tags: dict, unsealed: bool = False) -> tuple[float, str]:
    """Carriageway width in metres, and where it came from.

    A `width` under MIN_WIDTH is raised to it (`width=2` is mapped on a gravel road in southern Tasmania).
    Ignoring it, as before, made the road the class default: wider than one mapped as 3 m. One over MAX_WIDTH is
    implausible and is ignored.
    """
    w = _parse_metres(tags.get("width"))
    if w is not None and 0 < w <= MAX_WIDTH:
        return (w, "width") if w >= MIN_WIDTH else (MIN_WIDTH, "width_clamped")
    hw = tags.get("highway", "")
    base = hw.removesuffix("_link")
    lane_w = UNSEALED_LANE_M if unsealed else LANE_WIDTH.get(base, 3.0)
    lanes = tags.get("lanes")
    if lanes and lanes.isdigit() and 1 <= int(lanes) <= 6:
        return max(int(lanes) * lane_w, MIN_LANES_WIDTH_M), "lanes"
    if tags.get("narrow") == "yes":
        return NARROW_M, "narrow"
    oneway = tags.get("oneway") in ("yes", "true", "1", "-1")
    if oneway and hw.endswith("_link"):
        return max(lane_w, MIN_LANES_WIDTH_M), "class"
    if unsealed:
        return UNSEALED_WIDTH.get(base, UNSEALED_DEFAULT_M), "class"
    return max(2 * lane_w, MIN_LANES_WIDTH_M), "class"


def surface_class(tags: dict) -> str | None:
    """SEALED or UNSEALED from OSM's `surface`, or None when it's missing or a value outside both lists."""
    v = str(tags.get("surface") or "").strip().lower()
    if v in SEALED_SURFACES or v.startswith("concrete:"):
        return SEALED
    if v in UNSEALED_SURFACES:
        return UNSEALED
    return None


def _stretches(labels: list, lengths: list[float]) -> list[list]:
    """[label, first run, end run (exclusive), metres] for each stretch of consecutive runs with one label."""
    out: list[list] = []
    for i, (label, m) in enumerate(zip(labels, lengths)):
        if out and out[-1][0] == label:
            out[-1][2] = i + 1
            out[-1][3] += float(m)
        else:
            out.append([label, i, i + 1, float(m)])
    return out


def settle(labels: list, lengths: list[float], fallback: list, min_run_m: float = MIN_SURFACE_RUN_M) -> list:
    """Per-run labels with the gaps filled and the short stretches absorbed.

    1. A stretch of runs labelled None takes the label of the longer stretch beside it. With nothing labelled
       anywhere, each run takes its `fallback`.
    2. Then, shortest first, a stretch under `min_run_m` takes the label of the longer stretch beside it, until
       none is left. With two labels both sides always agree, so the stretch joins them into one.
    """
    out = list(labels)
    st = _stretches(out, lengths)
    if all(s[0] is None for s in st):
        return list(fallback)
    for k, (label, a, b, _) in enumerate(st):
        if label is None:
            beside = [st[j] for j in (k - 1, k + 1) if 0 <= j < len(st)]
            out[a:b] = [max(beside, key=lambda s: s[3])[0]] * (b - a)
    while True:
        st = _stretches(out, lengths)
        short = [k for k, s in enumerate(st) if s[3] < min_run_m]
        if len(st) < 2 or not short:
            return out
        k = min(short, key=lambda j: st[j][3])
        beside = [st[j] for j in (k - 1, k + 1) if 0 <= j < len(st)]
        out[st[k][1]:st[k][2]] = [max(beside, key=lambda s: s[3])[0]] * (st[k][2] - st[k][1])


def single_lane(tags: dict, width_m: float, unsealed: bool) -> bool:
    """Whether a run looks like one lane.

    - On a seal that means no centre line: `lanes=1` or `lane_markings=no`. The edge lines stay.
    - On gravel it means one pair of wheel paths with a grass strip between them: `lanes=1`, `narrow=yes`, or no
      `lanes` tag and narrower than SINGLE_LANE_BELOW_M as mapped (the unclassified default, 5.0 m, is; tertiary,
      5.5 m, isn't). OSM's lanes page likewise assumes one lane for unclassified roads and tracks.
    """
    lanes = str(tags.get("lanes") or "")
    if not unsealed:
        return lanes == "1" or tags.get("lane_markings") == "no"
    if lanes.isdigit():
        return int(lanes) == 1
    return tags.get("narrow") == "yes" or width_m < SINGLE_LANE_BELOW_M


@dataclass(frozen=True)
class RunSection:
    surface: str  # SEALED or UNSEALED, settled
    tagged: bool  # whether the run's own `surface` tag gave it a class (before settling)
    width_m: float  # as mapped, before the job's width option
    width_source: str
    style: int  # index into STYLES


def run_sections(tags: list[dict], lengths: list[float]) -> list[RunSection]:
    """Surface, width and look for each OSM run, given the metres of road each one covers.

    Untagged runs with nothing tagged anywhere are unsealed on `highway=track` and sealed otherwise. Whether a
    run looks like one lane is settled the same way as the surface (stretches under MIN_SURFACE_RUN_M join their
    neighbours), but only within a stretch of one surface, so it never changes a run's surface.
    """
    lengths = [float(m) for m in lengths]
    own = [surface_class(t) for t in tags]
    fallback = [UNSEALED if t.get("highway") == "track" else SEALED for t in tags]
    surface = settle(own, lengths, fallback)
    widths = [width_from_tags(t, s == UNSEALED) for t, s in zip(tags, surface)]
    single = [single_lane(t, w, s == UNSEALED) for t, (w, _), s in zip(tags, widths, surface)]
    for _, a, b, _ in _stretches(surface, lengths):
        single[a:b] = settle(single[a:b], lengths[a:b], single[a:b])
    return [RunSection(s, o is not None, w, src, style_code(s == UNSEALED, one))
            for s, o, (w, src), one in zip(surface, own, widths, single)]


def run_lengths(s: np.ndarray, run: np.ndarray, n_runs: int) -> np.ndarray:
    """Metres of centreline on each run, given each station's run: the surface merge measures the road as built."""
    return np.bincount(np.asarray(run)[:-1], weights=np.diff(s), minlength=n_runs)


def mapped_width(tags: list[dict], s: np.ndarray, run: np.ndarray) -> np.ndarray:
    """Each station's carriageway width as mapped (before the job's width option), from its run's tags."""
    per_run = run_sections(tags, run_lengths(s, run, len(tags)).tolist())
    return np.array([r.width_m for r in per_run])[np.asarray(run)]


def eased(s: np.ndarray, values: np.ndarray, ease_m: float = WIDTH_EASE_M) -> np.ndarray:
    step = s[1] - s[0] if len(s) > 1 else 1.0
    size = max(int(round(ease_m / step)), 1)
    return uniform_filter1d(values.astype(float), size=size, mode="nearest")


def off_ground_factor(s: np.ndarray, mask: np.ndarray, ease_m: float = WIDTH_EASE_M) -> np.ndarray:
    """0 on every station of `mask`, 1 from `ease_m` away, eased between: what the DEM's camber is multiplied by where
    the DEM beside the road isn't the road (a bridge deck, a tunnel, a hump bridged under a crossing road). The mask
    is widened by half the ease before easing, so a short bridge carries no camber at all, not a reduced one."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.ones(len(mask))
    step = s[1] - s[0] if len(s) > 1 else 1.0
    half = max(int(round(ease_m / 2 / step)), 1)
    wide = binary_dilation(mask, structure=np.ones(2 * half + 1, dtype=bool))
    return np.clip(eased(s, (~wide).astype(float), ease_m), 0.0, 1.0)


def narrowness(width) -> np.ndarray:
    """1 at or under NARROW_CAMBER_WIDTH_M, 0 at or over FULL_CAMBER_WIDTH_M, straight between."""
    return np.clip((FULL_CAMBER_WIDTH_M - np.asarray(width, float)) / (FULL_CAMBER_WIDTH_M - NARROW_CAMBER_WIDTH_M),
                   0.0, 1.0)


def camber_limit(width) -> np.ndarray:
    """The largest camber (superelevation) a road this wide gets: 4% under 5 m, rising to 7% at 6.4 m."""
    return MAX_SUPERELEVATION - (MAX_SUPERELEVATION - NARROW_CAMBER) * narrowness(width)


def crown_kept(width) -> np.ndarray:
    """The fall a road this wide keeps either side of its centre at any camber (`half_slopes`' `keep`)."""
    return NARROW_CROWN_KEPT * narrowness(width)


def half_slopes(e: np.ndarray, crown=CROWN, keep=0.0) -> tuple[np.ndarray, np.ndarray]:
    """Fall from centre to the left and right edges, given superelevation e (+ = left edge higher). `crown` and
    `keep` are one number or one per station.

    The two halves meet at a break of 2 x crown on a straight. As |e| grows the high side turns, and the break
    shrinks to nothing at |e| = crown: one plane from there on. With `keep` the break never falls under
    2 x keep, so the road keeps a crown; the extra tilt then goes on both sides, about the centre.
    """
    e = np.clip(e, -MAX_SUPERELEVATION, MAX_SUPERELEVATION)
    a = np.abs(e)
    half_break = np.maximum(crown - a, keep)  # how far the two halves' falls sit above a plane tilted by |e|
    high = half_break - a
    low = half_break + a
    left = np.where(e >= 0, high, low)
    right = np.where(e >= 0, low, high)
    return left, right
