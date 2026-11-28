"""Kerbs and footpaths on town roads, phase 1: from rules, OSM's `sidewalk` tags and the land parcels beside the road.

The kerbs and footpaths research (24 Nov 2026; every question answered "as proposed") measured Hobart's and Glenorchy's
town roads against the councils' own kerb lines. This module draws what it proposed for phase 1, and takes phase 2's
data where it's mirrored: the councils' kerb lines (`council_sides`) set where a kerb stands. Phase 2's street trees,
and parking bays, parking signs along the kerb and pram ramps (phase 3), are later.

**A kerb is its own feature.** The road's centreline, profile, tags and routing stay OpenStreetMap's alone; a kerb and
its footpath are drawn beside the road, like the buildings, and only set where the road's edge is drawn. Nothing here
moves the centreline or merges another road dataset into it.

**Which stretches** (`plan`, the section stage; Q1). Every 5 m, a station is a town road where `maxspeed` is 60 km/h
or less, or untagged, and at least TOWN_BUILDINGS building footprints have their centroid within TOWN_RADIUS_M. The
research found that rule right 94% of the time and finding 85% of the kerbed road in five boxes, and on three rural
routes kerbing only towns (Risdon Vale's edge, Richmond, Dover). Gaps under MAX_GAP_M are closed, then stretches under
MIN_STRETCH_M dropped. Never kerbed: unsealed road, motorways, trunk roads signed 80 km/h or more, the route's bridges,
tunnels and the humps under crossing roads (they keep their own kerb strip or none), roundabouts and slip ramps, each
with BLOCK_MARGIN_M either side. Both sides of a kerbed stretch get a kerb. The job's `kerbs: none` draws none.

**Width** (Q5): kerb face to kerb face, where the run has no `width` tag, its class's median as measured against the
councils' kerb lines (25 Nov 2026, Hobart's and Glenorchy's town roads): 8.0 m on local roads, 10.0 m tertiary,
12.8 m secondary, primary and trunk, 7.3 m on a one-way local street, and 8.3 m on one carriageway of a divided road
whatever its class (outer kerb to the median's, 28 Nov 2026; `divided`, the route stage's, by the ONE WAY signs' own
test, `core.signs.divided_carriageway`); at least 11.0 m on a two-way road with `lanes` tagged (marked lanes: LGAT's
collector, and what those measured); 2.3 m less for each side OSM's `parking:*` says has no on-street parking; where
it's CBD-dense (`cbd_dense`: buildings standing on the reserve's edge), at least 12.5 m (the CBD's measured median),
but not on a divided road's carriageway. The section stage logs the plan's summary in one line (`summary_line`).
Never narrower than the road as it was plus its two channels: a multi-lane road's `lanes` x the lane width (as
`core.section` sets it) is its floor, and adds nothing more (`kerb_to_kerb`). The widening eases in over WIDTH_EASE_M
from each end of a stretch; outside a stretch nothing changes. The road drawn as road (`half_left` and `half_right` in
section.npz, and the ribbon) reaches the channel's lip, CHANNEL_M inside the kerb face: the 450 mm channel is drivable,
part of kerb to kerb.

**The kerb** (Q2): Tasmania's type KC (LGAT TSD-R14-v3), drawn from its dimensions: a 450 mm channel, a 150 mm face
battered 40 mm and a 110 mm top falling 1 in 30, 600 mm overall, in grey concrete. No drawing is copied.

**Physics** (Q3, Q4): AC finds each tyre's ground with one vertical ray, so a true 150 mm face is a one-step spike
(the roundabout island's 150 mm over 0.3 m gave 0.7 g in game, 11 Nov). So the drawn kerb (`KERB_*`) isn't physical;
under it a hidden ramp (`1ROAD_KB*`, ROAD, not drawn) rises RAMP_RISE_M over RAMP_M from the channel's lip to the back
of the kerb. The footpath (`1FOOTWAY_*`) has ROAD's grip but its own surface KEY, FOOTWAY, which isn't valid track
(`core.package`): running on it counts as off the track, as grass does. Each piece shares the road's edge vertices
and the ribbon's outer column exactly, as the bridges' kerb strip does, so there's no seam.

**The footpath** (Q7): from the kerb face to the land parcels' boundary (the road reserve's edge), where that's
PATH_MIN_M to PATH_MAX_M behind the face, pulled in to a building standing over the boundary; else PATH_CBD_M where
it's CBD-dense and PATH_DEFAULT_M elsewhere. Wider than WIDE_VERGE_M outside the CBD, it's a PATH_DEFAULT_M path at the
kerb and mown grass behind (LGAT TSD-R06's first arrangement). It starts level with the kerb's top and rises
PATH_FALL away from the road (TSD-R11). Concrete where CBD-dense, asphalt elsewhere (Hobart's commonest), or as OSM's
`sidewalk:surface` says. On both sides, unless OSM's `sidewalk` says `left`, `right` or `no`; a side without one is a
grass strip at kerb height.

**Junctions** (Q8): where a side road leaves a kerbed stretch its stub's flares follow a kerb return of RETURN_M (or
RETURN_MAJOR_M where either road is secondary or above), the kerb runs round it and STUB_KERB_M up the stub's edges,
and the footpath ends at the return.

**Driveways** (Q9): each driveway, service road, track or private road OSM maps leaving a kerbed stretch gets a KC
vehicular crossing: CROSSOVER_M wide (or the way's own width, if wider) with WING_M transitions, the kerb's face sloped
over CROSSOVER_KC_M from a 20 mm lip and its ramp lower and longer, the footpath across it concrete. Wings are joined
where less than CROSSOVER_JOIN_M of kerb would be left between two. None is inferred from the parcels.

**Furniture and scenery** (Q10): signal posts, blades and signs stand POST_BEHIND_M behind the kerb face on the
footpath; no guide posts; trees TREE_BEHIND_M or more behind the kerb face and none further than TREE_INTO_PATH_M into
the footpath; buildings kept off the footpath rather than BUILDING_CLEAR_M off the road; timing gates GATE_BEHIND_M
behind the kerb face. The AI line's margin is from the kerb face, plus AI_EXTRA_M (Q11).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.ndimage import binary_dilation, uniform_filter1d
from scipy.spatial import cKDTree

from mapcreator.core import section as sec

FORMAT_VERSION = 1

# -- the KC profile (TSD-R14-v3), metres ----------------------------------------------------------------------------
CHANNEL_M = 0.45  # the channel, from its lip (the road's edge as drawn) to the kerb's face
FACE_BATTER_M = 0.04  # the face's batter, in plan
KERB_TOP_M = 0.11  # the kerb's flat top
KC_WIDTH_M = CHANNEL_M + FACE_BATTER_M + KERB_TOP_M  # 0.6 m overall
TOP_FALL = 1.0 / 30.0  # the top falls to the road
CHANNEL_DIP_M = 0.01  # the channel's invert below its lip (drawn only; the physics runs flat to the lip)
# -- physics -------------------------------------------------------------------------------------------------------
RAMP_M = KC_WIDTH_M  # the hidden ramp, from the channel's lip to the kerb's back: 1 in 4
RAMP_RISE_M = 0.15  # the kerb's top above the lip: the footpath starts here
KERB_TAPER_M = 2.0  # a kerb rises from nothing to full height over this at each end
# -- footpaths ------------------------------------------------------------------------------------------------------
PATH_FALL = 0.02  # rising away from the road (TSD-R11 note 3, [B] p. 27)
PATH_MIN_M = 1.2  # TSD-R11 note 2: "Min width 1.2m"
PATH_MAX_M = 6.0  # a reserve edge further than this behind the kerb isn't the footpath's
PATH_DEFAULT_M = 1.5  # TSD-R11: "Generally 'W' = 1.5m when footpath is adjacent to kerb"
PATH_CBD_M = 2.4  # commercial footpaths, preferably ([B] 3.6)
WIDE_VERGE_M = 3.0  # wider than this outside the CBD: a PATH_DEFAULT_M path and grass behind
RAY_START_M = 0.3  # rays for the reserve's edge start this far out from the centreline (the research's)
RAY_REACH_M = 30.0
PATH_SAMPLE_M = 2.0  # the reserve's edge is measured this often
PATH_WINDOW = 9  # samples in the running median that settles it
PATH_SMOOTH_M = 6.0
CONCRETE_SURFACES = frozenset({"concrete", "concrete:plates", "concrete:lanes", "paving_stones", "sett", "paved",
                               "bricks", "cobblestone"})
ASPHALT_SURFACES = frozenset({"asphalt", "chipseal"})
# -- which stretches ------------------------------------------------------------------------------------------------
TOWN_SPEED_KMH = 60.0
TOWN_BUILDINGS = 8
TOWN_RADIUS_M = 80.0
# CBD-dense (`cbd_dense`): building walls on the reserve's edge, as half the CBD's are (estimate).
CBD_SETBACK_M = 1.0
CBD_WINDOW = 13  # samples, PATH_SAMPLE_M apart: about 25 m
CBD_SHARE = 0.5
RULE_STEP_M = 5.0
MIN_STRETCH_M = 150.0  # estimate (the research's Q1)
MAX_GAP_M = 100.0  # estimate
CBD_MIN_M = 50.0  # a CBD-dense stretch shorter than this is the town's, and gaps shorter than this are closed
BLOCK_MARGIN_M = 10.0  # either side of a bridge, tunnel, roundabout or slip ramp, no kerb
RAMP_BLOCK_M = 100.0  # either side of where a slip ramp leaves the route (its taper runs along it)
NEVER_CLASSES = frozenset({"motorway", "motorway_link"})
FAST_TRUNK_KMH = 80.0
# -- width ----------------------------------------------------------------------------------------------------------
# -- the councils' own kerb lines (`providers.councils.kerb_lines`) -----------------------------------------------
COUNCIL_NEAR_M = 2.5  # a kerb line nearer the centreline than this is an island's or a median's, not the edge's
COUNCIL_REACH_M = 15.0  # and further than this, another road's
COUNCIL_PARALLEL_DEG = 45.0
COUNCIL_MIN_HALF_M = 2.0
COUNCIL_EASE_M = 6.0  # a council's kerb follows the street's own changes of width (build-outs), smoothed this much
# -- kerb to kerb from the rule (`kerb_to_kerb`): calibrated 25 Nov 2026 against the councils' kerb lines on Hobart's
# and Glenorchy's town roads (docs/known-issues.md, "Kerbs and footpaths"): each the median measured on a random 70%
# of the streets, to 0.1 m, and tested on the other 30%
LOCAL_KERB_TO_KERB_M = 8.0  # residential, unclassified, living_street, road (LGAT's new local road is 8.9)
CLASS_KERB_TO_KERB_M = {"tertiary": 10.0, "secondary": 12.8, "primary": 12.8,
                        "trunk": 12.8}  # secondary 12.76, primary 12.84; trunk as primary (none two-way in town)
MARKED_KERB_TO_KERB_M = 11.0  # two-way with `lanes` tagged (marked lanes): LGAT TSD-R06-v3's collector, measured 11.0
PARKING_LANE_M = 2.3  # off kerb to kerb for each side OSM's `parking:*` says has no on-street parking (estimate)
NO_PARKING_VALUES = ("no", "separate")
CBD_KERB_TO_KERB_M = 12.5  # at least this where CBD-dense: the CBD's measured median
# Calibrated 28 Nov 2026 on the same streets and the same 70/30 split (docs/known-issues.md, "Kerbs and footpaths"):
DIVIDED_KERB_TO_KERB_M = 8.3  # one carriageway of a divided road, outer kerb to the median's (`divided`), any class
ONE_WAY_LOCAL_KERB_TO_KERB_M = 7.3  # a one-way local street (not a divided road's carriageway)
WIDTH_EASE_M = 30.0
# -- junctions and driveways ---------------------------------------------------------------------------------------
RETURN_M = 5.0  # LGAT 2026 [B] 3.4.7: "a minimum kerb radii of 5m"
RETURN_MAJOR_M = 8.0  # where either road is secondary or above
RETURN_MAJOR_CLASSES = frozenset({"secondary", "primary", "trunk", "motorway"})
STUB_KERB_M = 10.0  # the kerb runs this far up a side road's stub past its return, then tapers off
CROSSOVER_M = 3.6  # TSD-R09-v3, typical and minimum
WING_M = 1.0
CROSSOVER_JOIN_M = 0.5  # wings dropped where less kerb than this would be left between two crossings
CROSSOVER_KC_M = 0.9  # the KC vehicular crossing: 900 wide overall, its face sloped from a 20 mm lip
CROSSOVER_LIP_M = 0.02
CROSSOVER_CLEAR_M = 3.0  # no crossover this near a side road's mouth (the return's kerb meets the road's there)
# -- what stands beside a kerbed road -------------------------------------------------------------------------------
POST_BEHIND_M = 0.6  # signal posts, blades and signs, behind the kerb face (Spec 730: at least 0.5 m)
TREE_BEHIND_M = 0.8
TREE_INTO_PATH_M = 1.0
BUILDING_SLACK_M = 0.1  # a building may stand this much over the footpath's back edge before it's left out
GATE_BEHIND_M = 0.5
AI_EXTRA_M = 0.3  # the AI line's margin is measured from the kerb face, plus this (Q11)
RAIL_BEHIND_M = 0.3  # a guard rail stands this far behind the footpath
# -- meshes ---------------------------------------------------------------------------------------------------------
WINDOW_M = 400.0  # packed by material per this much of the road, as the stubs' meshes are
KERB_TILE_M = 1.0
PATH_TILE_M = 3.0
GRASS_TILE_M = 6.0  # as the verges', so they meet without a seam
KERB_MATERIAL = "kerb"  # the roundabout island's concrete (`core.roundabouts`)
PATH_MATERIALS = {False: "footpath_asphalt", True: "footpath_concrete"}
PATH_TEXTURES = {"footpath_asphalt": "asphalt", "footpath_concrete": "kerb"}
VISUAL = "KERB_"
FOOTWAY = "1FOOTWAY_"
# The hidden ramps: 1ROAD_KB### along the route, 1ROAD_<tag>KB### along a side road's stub and returns (tag J...: a
# stub's, so the checks count it as the stub's road, not the route's).
RAMP_NAME = re.compile(r"^1ROAD_\w*?KB\d{3}")


def is_ramp(name: str) -> bool:
    """A kerb's hidden ramp mesh (`meshes`)."""
    return bool(RAMP_NAME.match(name))


# -- which stretches --------------------------------------------------------------------------------------------------

def maxspeed_kmh(tags: dict) -> float | None:
    """The posted speed (km/h) from `maxspeed`, or None where it's missing or can't be read. `AU:urban` is 50 and
    `AU:rural` 100 (Australia's defaults)."""
    v = str(tags.get("maxspeed") or "").strip().lower()
    if v.endswith(":urban"):
        return 50.0
    if v.endswith(":rural"):
        return 100.0
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?$", v)
    if not m:
        return None
    x = float(m.group(1))
    return x * 1.609344 if m.group(2) == "mph" else x


def _base_class(tags: dict) -> str:
    return str(tags.get("highway") or "").removesuffix("_link")


def never_kerbed(tags: dict) -> bool:
    """A run no kerb is drawn on, whatever's beside it: a motorway, or a trunk road signed FAST_TRUNK_KMH or more."""
    hw = str(tags.get("highway") or "")
    if hw in NEVER_CLASSES:
        return True
    v = maxspeed_kmh(tags)
    return _base_class(tags) == "trunk" and v is not None and v >= FAST_TRUNK_KMH


def town_speed(tags: dict) -> bool:
    """`maxspeed` at TOWN_SPEED_KMH or under, or untagged."""
    v = maxspeed_kmh(tags)
    return v is None or v <= TOWN_SPEED_KMH


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last + 1) of each run of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def stretches(mask: np.ndarray, s: np.ndarray, min_m: float = MIN_STRETCH_M, gap_m: float = MAX_GAP_M,
              hard: np.ndarray | None = None) -> np.ndarray:
    """`mask` with its gaps under `gap_m` closed (between two True stretches, never at the road's ends), then any
    station of `hard` taken out again, then its stretches under `min_m` dropped."""
    s = np.asarray(s, float)
    out = np.asarray(mask, bool).copy()
    runs = _runs(out)
    for (_, b), (a, _) in zip(runs, runs[1:]):  # the gap between two stretches: stations b .. a-1
        if s[a] - s[b - 1] <= gap_m:
            out[b:a] = True
    if hard is not None:
        out &= ~np.asarray(hard, bool)
    for a, b in _runs(out):
        if s[b - 1] - s[a] < min_m:
            out[a:b] = False
    return out


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ramp_in(mask: np.ndarray, s: np.ndarray, metres: float) -> np.ndarray:
    """0 outside `mask`, rising smoothly to 1 over `metres` in from each end of each of its stretches."""
    s = np.asarray(s, float)
    out = np.zeros(len(s))
    for a, b in _runs(mask):
        d = np.minimum(s[a:b] - s[a], s[b - 1] - s[a:b])
        out[a:b] = _smoothstep(d / metres) if metres > 0 else 1.0
    return out


def _dilate(mask: np.ndarray, s: np.ndarray, metres: float) -> np.ndarray:
    mask = np.asarray(mask, bool)
    if not mask.any() or metres <= 0:
        return mask
    step = float(s[1] - s[0]) if len(s) > 1 else 1.0
    k = max(int(round(metres / step)), 1)
    return binary_dilation(mask, structure=np.ones(2 * k + 1, bool))


def building_counts(xy: np.ndarray, s: np.ndarray, centroids: np.ndarray, radius: float = TOWN_RADIUS_M,
                    step_m: float = RULE_STEP_M) -> np.ndarray:
    """The number of building centroids within `radius` of each station, counted every `step_m` and held between."""
    xy, s = np.asarray(xy, float), np.asarray(s, float)
    n = len(s)
    if not len(centroids) or not n:
        return np.zeros(n, int)
    step = float(s[1] - s[0]) if n > 1 else 1.0
    every = max(int(round(step_m / step)), 1)
    idx = np.arange(0, n, every)
    counts = cKDTree(np.asarray(centroids, float)).query_ball_point(xy[idx], r=radius, return_length=True)
    nearest = np.clip(np.round(np.arange(n) / every).astype(int), 0, len(idx) - 1)
    return np.asarray(counts, int)[nearest]


# -- width -------------------------------------------------------------------------------------------------------------

def _lanes(tags: dict) -> int:
    """`lanes` as `section.width_from_tags` reads it (1 to 6), or 0 where it's missing or can't be read."""
    v = str(tags.get("lanes") or "").strip()
    return int(v) if v.isdigit() and 1 <= int(v) <= 6 else 0


def _oneway(tags: dict) -> bool:
    return str(tags.get("oneway") or "") in ("yes", "true", "1", "-1")


def no_parking_sides(tags: dict) -> int:
    """How many sides OSM says have no on-street parking: `parking:both`, `parking:left`, `parking:right` (or the
    older `parking:lane:*`) set to `no` or `separate` (parking off the street)."""
    def v(side):
        return str(tags.get(f"parking:{side}") or tags.get(f"parking:lane:{side}") or "").strip().lower()
    if v("both") in NO_PARKING_VALUES:
        return 2
    return sum(v(side) in NO_PARKING_VALUES for side in ("left", "right"))


def kerb_to_kerb(tags: dict, today_m: float, cbd: bool, divided: bool = False) -> float:
    """A kerbed run's width, kerb face to kerb face (Q5), calibrated against the councils' kerb lines (the constants'
    notes). Its `width` tag where it has one; else its class's measured median (CLASS_KERB_TO_KERB_M on tertiary
    roads and above, LOCAL_KERB_TO_KERB_M on the rest; ONE_WAY_LOCAL_KERB_TO_KERB_M on a one-way local street), at
    least MARKED_KERB_TO_KERB_M on a two-way road with `lanes` tagged, less PARKING_LANE_M for each side tagged with no
    on-street parking; at least CBD_KERB_TO_KERB_M where it's CBD-dense. On one carriageway of a divided road
    (`divided`: `core.signs.divided_carriageway`, found by the route stage) DIVIDED_KERB_TO_KERB_M whatever its class,
    and no CBD floor: the carriageway, outer kerb to the median's, not the road. Never under `today_m` (the road as it
    was, with the job's width option: on a multi-lane road, its `lanes` x the lane width) plus two channels, nor over
    `section.MAX_WIDTH`. A road's extra lanes add nothing more: the lanes tagged on Hobart's multi-lane streets take
    the room of the lanes by the kerb (fitted, an extra lane added 0 m over the road's own width)."""
    w = sec._parse_metres(tags.get("width"))
    if w is not None and 0 < w <= sec.MAX_WIDTH:
        kk = w
    else:
        lanes = _lanes(tags)
        cls = _base_class(tags)
        if divided:
            kk = DIVIDED_KERB_TO_KERB_M
        elif _oneway(tags) and cls not in CLASS_KERB_TO_KERB_M:
            kk = ONE_WAY_LOCAL_KERB_TO_KERB_M
        else:
            kk = CLASS_KERB_TO_KERB_M.get(cls, LOCAL_KERB_TO_KERB_M)
        if lanes >= 2 and not _oneway(tags):
            kk = max(kk, MARKED_KERB_TO_KERB_M)
        kk -= no_parking_sides(tags) * PARKING_LANE_M
    if cbd and not divided:
        kk = max(kk, CBD_KERB_TO_KERB_M)
    return float(min(max(kk, today_m + 2 * CHANNEL_M), sec.MAX_WIDTH))


# -- footpaths ---------------------------------------------------------------------------------------------------------

NO_VALUES = ("no", "none")
YES_VALUES = ("yes", "both", "separate")


def footpath_sides(tags: dict, reversed_: bool = False) -> tuple[bool, bool]:
    """Whether the route's left and right sides have a footpath: both, unless `sidewalk` (or `sidewalk:both`,
    `sidewalk:left`, `sidewalk:right`) says otherwise. A footpath mapped as its own way (`separate`) is one. Left and
    right are the way's; `reversed_` turns them round for a route driving it the other way."""
    left = right = True
    v = str(tags.get("sidewalk") or "").strip().lower()
    if v in NO_VALUES:
        left = right = False
    elif v == "left":
        right = False
    elif v == "right":
        left = False
    both = str(tags.get("sidewalk:both") or "").strip().lower()
    if both in NO_VALUES:
        left = right = False
    elif both in YES_VALUES:
        left = right = True
    for key, side in (("sidewalk:left", "L"), ("sidewalk:right", "R")):
        x = str(tags.get(key) or "").strip().lower()
        if x in NO_VALUES or x in YES_VALUES:
            if side == "L":
                left = x not in NO_VALUES
            else:
                right = x not in NO_VALUES
    return (right, left) if reversed_ else (left, right)


def footpath_surface(tags: dict, side: str, reversed_: bool = False) -> bool | None:
    """True for concrete (or pavers), False for asphalt, from OSM's `sidewalk:<side>:surface`, `sidewalk:both:surface`
    or `sidewalk:surface` on the route's `side` ("L", "R"); None where none says."""
    way_side = {"L": "left", "R": "right"}[side]
    if reversed_:
        way_side = "right" if way_side == "left" else "left"
    for key in (f"sidewalk:{way_side}:surface", "sidewalk:both:surface", "sidewalk:surface"):
        v = str(tags.get(key) or "").strip().lower()
        if v in CONCRETE_SURFACES:
            return True
        if v in ASPHALT_SURFACES:
            return False
    return None


def first_hits(origin: np.ndarray, direction: np.ndarray, geoms: list, start: float = RAY_START_M,
               reach: float = RAY_REACH_M) -> np.ndarray:
    """The distance from each origin, along its direction, to the first boundary of any of `geoms` between `start`
    and `reach`; NaN where there's none."""
    origin, direction = np.asarray(origin, float), np.asarray(direction, float)
    out = np.full(len(origin), np.nan)
    if not len(origin) or not geoms:
        return out
    rays = shapely.linestrings(np.stack([origin + direction * start, origin + direction * reach], axis=1))
    geoms = np.asarray(geoms, dtype=object)
    polygonal = np.isin(shapely.get_type_id(geoms), (3, 6))  # a polygon's edge is its boundary; a line is its own
    edges = np.where(polygonal, shapely.boundary(geoms), geoms)
    tree = shapely.STRtree(edges)
    ri, gi = tree.query(rays, predicate="intersects")
    if not len(ri):
        return out
    hits = shapely.intersection(rays[ri], edges[gi])
    pts, which = shapely.get_coordinates(hits, return_index=True)
    if not len(pts):
        return out
    ray = ri[which]
    d = np.hypot(*(pts[:, :2] - origin[ray]).T)
    np.fmin.at(out, ray, d)
    return out


def _settled(values: np.ndarray, valid: np.ndarray, window: int = PATH_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """A running median of `values` over `window` samples, of the valid ones only, and whether at least half the
    window was valid (else the value isn't used)."""
    n = len(values)
    half = window // 2
    v = np.where(valid, values, np.nan)
    padded = np.concatenate([np.full(half, np.nan), v, np.full(half, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(padded, window)
    count = np.isfinite(win).sum(axis=1)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(win, axis=1)
    ok = count >= max(window // 2, 1)
    return np.where(ok, med, np.nan)[:n], ok[:n]


# -- the plan ----------------------------------------------------------------------------------------------------------

@dataclass
class Plan:
    """Kerbs and footpaths station by station (section.npz's `kerb_*` arrays), and how the road's halves change."""
    kerb: dict  # side ("L", "R") -> 0..1: 1 kerbed, 0 not, between at a kerb's tapered ends
    path: dict  # side -> metres from the kerb face to the footpath's back edge (the reserve's edge), paved or not
    paved: dict  # side -> metres from the kerb face that are paved (the rest is grass)
    concrete: dict  # side -> whether the paved footpath is concrete (else asphalt)
    half: dict  # side -> the road's half width to the channel's lip, as drawn (the section's, widened)
    cbd: np.ndarray  # CBD-dense stations
    summary: dict
    council: set = field(default_factory=set)  # the councils whose kerb lines were used (`providers.councils`)

    def any(self) -> bool:
        return bool(self.kerb["L"].any() or self.kerb["R"].any())


def none_plan(half_l: np.ndarray, half_r: np.ndarray, summary: dict | None = None) -> Plan:
    n = len(half_l)
    z = np.zeros(n)
    return Plan({"L": z, "R": z}, {"L": z, "R": z}, {"L": z, "R": z}, {"L": np.zeros(n, bool), "R": np.zeros(n, bool)},
                {"L": np.asarray(half_l, float), "R": np.asarray(half_r, float)}, np.zeros(n, bool),
                summary or {"kerbed_m": 0.0})


def plan(s, xy, heading, run, run_tags: list[dict], half_l, half_r, *, unsealed=None, blocked=None,
         centroids=(), buildings=(), parcels=None, reversed_runs=None, council=None, divided_runs=None) -> Plan:
    """Kerbs and footpaths along the route (module notes). `run` is each station's OSM run, `run_tags` each run's
    tags, `reversed_runs` whether the route drives each against its drawn direction (for its sides) and `divided_runs`
    whether each is one carriageway of a divided road (for its width, `kerb_to_kerb`). `half_l`,
    `half_r`: the road's half widths as they'd be without kerbs (the section's, eased, with the job's width option).
    `unsealed`: unsealed stations; `blocked`: stations never kerbed (bridges, tunnels, roundabouts, slip ramps: widened
    here by BLOCK_MARGIN_M). `centroids`: building footprints' centroids; `buildings` their polygons (a footpath stops
    at one standing over the reserve's edge); `parcels` the land parcels beside the road reserve, or None (footpaths
    then take PATH_CBD_M or PATH_DEFAULT_M). `council`: a callable giving the councils' kerb lines for a list of boxes
    (`providers.councils.kerb_lines.KerbLines`, or None where there are none to be had): where it has a kerb beside a
    station, the station is kerbed and the kerb stands there; where it marks an edge unkerbed, that side isn't;
    elsewhere the rule decides (`council_sides`)."""
    s, xy, heading = np.asarray(s, float), np.asarray(xy, float), np.asarray(heading, float)
    run = np.asarray(run, int)
    n = len(s)
    half_l, half_r = np.asarray(half_l, float), np.asarray(half_r, float)
    reversed_runs = list(reversed_runs) if reversed_runs is not None else [False] * len(run_tags)
    divided_runs = [bool(v) for v in divided_runs] if divided_runs is not None else [False] * len(run_tags)
    step = float(s[1] - s[0]) if n > 1 else 1.0
    speed_ok = np.array([town_speed(t) for t in run_tags], bool)[run]
    never = np.array([never_kerbed(t) for t in run_tags], bool)[run]
    hard = never.copy()
    if unsealed is not None:
        hard |= np.asarray(unsealed, bool)
    if blocked is not None:
        hard |= _dilate(np.asarray(blocked, bool), s, BLOCK_MARGIN_M)
    count = building_counts(xy, s, np.asarray(centroids, float).reshape(-1, 2))
    town = speed_ok & ~hard & (count >= TOWN_BUILDINGS)
    rule_m = round(float(town.sum() * step), 1)
    # The councils' own kerb lines, where there are any (phase 2's data): they say, and the rule fills in.
    faces, bare, council_sources = _council(s, xy, heading, speed_ok & ~hard, council)
    said = np.isfinite(faces["L"]) | np.isfinite(faces["R"])
    unkerbed = bare["L"] & bare["R"]
    town = (town & ~unkerbed) | said
    kerbed = stretches(town, s, hard=hard)
    if not kerbed.any():
        return none_plan(half_l, half_r, {"kerbed_m": 0.0, "rule_m": rule_m})
    if callable(parcels):  # fetched only now, for the kerbed stretches alone (None if they can't be had)
        parcels = parcels(stretch_boxes(xy, kerbed))
    # The reserve's edge and the first building wall out from the centreline, each side, every PATH_SAMPLE_M.
    nrm = np.column_stack([-np.sin(heading), np.cos(heading)])
    every = max(int(round(PATH_SAMPLE_M / step)), 1)
    idx = np.flatnonzero(kerbed)[::every]
    if len(idx) and np.flatnonzero(kerbed)[-1] != idx[-1]:
        idx = np.append(idx, np.flatnonzero(kerbed)[-1])
    rays = {}
    for side, sign in (("L", 1.0), ("R", -1.0)):
        bound = wall = np.full(len(idx), np.nan)
        if parcels is not None and len(idx):
            bound = first_hits(xy[idx], sign * nrm[idx], list(parcels))
        if len(buildings) and len(idx):
            wall = first_hits(xy[idx], sign * nrm[idx], list(buildings))
        rays[side] = (bound, wall)
    cbd = cbd_dense(s, idx, rays, kerbed)
    # A side the council marks unkerbed (an Edge of Formation, no kerb) gets none.
    k = {side: ramp_in(kerbed & ~bare[side], s, KERB_TAPER_M) for side in "LR"}
    ease = ramp_in(kerbed, s, WIDTH_EASE_M)
    # Kerb to kerb, per run, plain and CBD-dense, then per station.
    today = half_l + half_r
    kk = np.zeros(n)
    for r in np.unique(run[kerbed]):
        on = kerbed & (run == r)
        for dense in (False, True):
            sel = on & (cbd == dense)
            if sel.any():
                kk[sel] = [kerb_to_kerb(run_tags[r], float(t), dense, divided_runs[r]) for t in today[sel]]
    kk = np.where(kerbed, kk, today + 2 * CHANNEL_M)
    # Each half as drawn (to the lip): the same share of the new width as of the old, never narrower than it was.
    share_l = np.where(today > 0, half_l / np.maximum(today, 1e-9), 0.5)
    target_l = np.maximum(share_l * kk - CHANNEL_M, half_l)
    target_r = np.maximum((1 - share_l) * kk - CHANNEL_M, half_r)
    # Where the council has the kerb, the road reaches its channel's lip, each side its own (Q6: the kerb sets where the
    # road's edge is drawn; the centreline stays OSM's), never under COUNCIL_MIN_HALF_M.
    target_l = np.where(np.isfinite(faces["L"]), np.maximum(faces["L"] - CHANNEL_M, COUNCIL_MIN_HALF_M), target_l)
    target_r = np.where(np.isfinite(faces["R"]), np.maximum(faces["R"] - CHANNEL_M, COUNCIL_MIN_HALF_M), target_r)
    # A change of width within a stretch (from one OSM way to the next) eased over WIDTH_EASE_M too, as the road's own
    # changes are (`section.eased`), so the kerb never steps; never narrower than the road was.
    floor_l = np.where(np.isfinite(faces["L"]), 0.0, half_l)  # a council's kerb may stand inside the road as mapped
    floor_r = np.where(np.isfinite(faces["R"]), 0.0, half_r)
    target_l = np.maximum(sec.eased(s, target_l, COUNCIL_EASE_M if council_sources else WIDTH_EASE_M), floor_l)
    target_r = np.maximum(sec.eased(s, target_r, COUNCIL_EASE_M if council_sources else WIDTH_EASE_M), floor_r)
    new_l = half_l + (target_l - half_l) * ease
    new_r = half_r + (target_r - half_r) * ease

    # Footpaths.
    sides = [footpath_sides(t, rv) for t, rv in zip(run_tags, reversed_runs)]
    has = {"L": np.array([a for a, _ in sides], bool)[run], "R": np.array([b for _, b in sides], bool)[run]}
    tagged = {side: np.array([footpath_surface(t, side, rv) for t, rv in zip(run_tags, reversed_runs)],
                             dtype=object)[run] for side in "LR"}
    path, paved, concrete = {}, {}, {}
    sources = {"parcels": 0.0, "standard": 0.0}
    for side, half in (("L", new_l), ("R", new_r)):
        face = half + CHANNEL_M
        bound, wall = rays[side]
        verge = np.full(len(idx), np.nan)
        if parcels is not None and len(idx):
            # The reserve's edge, or a building standing over it, nearer.
            verge = np.where(np.isfinite(bound), np.fmin(bound, wall), np.nan) - face[idx]
        found = np.isfinite(verge) & (verge >= PATH_MIN_M) & (verge <= PATH_MAX_M)
        settled, ok = _settled(verge, found)
        default = np.where(cbd[idx], PATH_CBD_M, PATH_DEFAULT_M)
        sample = np.where(ok, np.clip(settled, PATH_MIN_M, PATH_MAX_M), default)
        sources["parcels"] += float(ok.sum()) * every * step
        sources["standard"] += float((~ok).sum()) * every * step
        width = np.full(n, PATH_DEFAULT_M)
        if len(idx):
            width = np.interp(s, s[idx], sample)
            width = uniform_filter1d(width, max(int(round(PATH_SMOOTH_M / step)), 1), mode="nearest")
        width = np.where(kerbed, width, 0.0)
        wide_suburb = ~cbd & (width > WIDE_VERGE_M)
        pv = np.where(wide_suburb, PATH_DEFAULT_M, width)
        pv = np.where(has[side], pv, 0.0)
        mat = np.array([t if t is not None else None for t in tagged[side]], dtype=object)
        conc = np.where(mat == None, cbd, mat == True).astype(bool)  # noqa: E711 - an object array's None
        path[side], paved[side], concrete[side] = width, np.where(kerbed, pv, 0.0), conc & kerbed

    kerbed_m = float(kerbed.sum() * step)
    summary = {
        "kerbed_m": round(kerbed_m, 1),
        "rule_m": round(float(town.sum() * step), 1),
        "cbd_m": round(float(cbd.sum() * step), 1),
        "stretches": [[round(float(s[a] - s[0]), 1), round(float(s[b - 1] - s[0]), 1)] for a, b in _runs(kerbed)],
        "kerb_to_kerb_m": _stats((new_l + new_r + 2 * CHANNEL_M)[kerbed]),  # as drawn, face to face
        "before_m": _stats(today[kerbed]),
        "wider_m": _stats(((new_l + new_r + 2 * CHANNEL_M) - today)[kerbed]),
        "footpath_m": {side: _stats(path[side][kerbed]) for side in "LR"},
        "footpath_source_m": {k_: round(v, 1) for k_, v in sources.items()},
        "no_footpath_m": {side: round(float((kerbed & ~has[side]).sum() * step), 1) for side in "LR"},
        "concrete_m": {side: round(float((concrete[side] & (paved[side] > 0)).sum() * step), 1) for side in "LR"},
    }
    on_divided = kerbed & np.array(divided_runs, bool)[run] if len(run_tags) else np.zeros(n, bool)
    if on_divided.any():  # kerbed on one carriageway of a divided road (DIVIDED_KERB_TO_KERB_M)
        summary["divided_m"] = round(float(on_divided.sum() * step), 1)
    if council_sources:
        summary["council"] = sorted(council_sources)
        summary["council_m"] = {side: round(float((kerbed & np.isfinite(faces[side])).sum() * step), 1)
                                for side in "LR"}
        summary["council_bare_m"] = {side: round(float((kerbed & bare[side]).sum() * step), 1) for side in "LR"}
    return Plan(k, path, paved, concrete, {"L": new_l, "R": new_r}, cbd, summary, council_sources)


def _council(s, xy, heading, candidates: np.ndarray, council) -> tuple[dict, dict, set]:
    """The councils' kerb faces (distance from the centreline, NaN where none) and unkerbed edges per side and station,
    settled over PATH_WINDOW samples as the footpaths are, along `candidates` (the stations a kerb may stand on); and
    the councils they came from. Nothing (all NaN, none bare) without council lines."""
    n = len(s)
    faces = {side: np.full(n, np.nan) for side in "LR"}
    bare = {side: np.zeros(n, bool) for side in "LR"}
    if council is None or not candidates.any():
        return faces, bare, set()
    got = council(stretch_boxes(xy, candidates))
    if got is None or not got.lines:
        return faces, bare, set()
    step = float(s[1] - s[0]) if n > 1 else 1.0
    every = max(int(round(PATH_SAMPLE_M / step)), 1)
    idx = np.flatnonzero(candidates)[::every]
    sides = council_sides(xy, heading, idx, got.kind("kerb"), got.kind("edge"))
    for side, (face, edge) in sides.items():
        settled, ok = _settled(face, np.isfinite(face))
        sample = np.where(ok, settled, np.nan)
        bare_share = uniform_filter1d(edge.astype(float), PATH_WINDOW, mode="nearest")
        near = np.clip(np.round(np.interp(np.arange(n), idx, np.arange(len(idx)))).astype(int), 0, len(idx) - 1)
        within = np.abs(idx[near] - np.arange(n)) <= every  # held only between samples, not past a candidate's end
        faces[side] = np.where(candidates & within, sample[near], np.nan)
        bare[side] = candidates & within & (bare_share[near] >= 0.5) & ~np.isfinite(faces[side])
    return faces, bare, set(got.sources)


def council_sides(xy: np.ndarray, heading: np.ndarray, idx: np.ndarray, kerb_lines: list,
                  edge_lines: list) -> dict:
    """Per side ("L", "R"), at stations `idx`: the distance from the centreline to a council's kerb face (the first
    Top of Kerb line out, from COUNCIL_NEAR_M to COUNCIL_REACH_M, running within COUNCIL_PARALLEL_DEG of the road: a
    kerb across it is a crossing's or an island's end, not its edge), NaN where none; and whether the council marks
    that edge unkerbed instead (an Edge of Formation line found, nearer than any kerb)."""
    xy, heading = np.asarray(xy, float), np.asarray(heading, float)
    idx = np.asarray(idx, int)
    nrm = np.column_stack([-np.sin(heading), np.cos(heading)])
    out = {}
    for side, sign in (("L", 1.0), ("R", -1.0)):
        face = _parallel_hits(xy[idx], sign * nrm[idx], heading[idx], kerb_lines)
        edge = _parallel_hits(xy[idx], sign * nrm[idx], heading[idx], edge_lines)
        out[side] = (face, np.isfinite(edge) & ~(face < edge))
    return out


def _parallel_hits(origin: np.ndarray, direction: np.ndarray, heading: np.ndarray, lines: list) -> np.ndarray:
    """`first_hits` on lines, counting only a line running within COUNCIL_PARALLEL_DEG of the road where it's hit."""
    out = np.full(len(origin), np.nan)
    if not len(origin) or not lines:
        return out
    lines = np.asarray(lines, dtype=object)
    rays = shapely.linestrings(np.stack([origin + direction * COUNCIL_NEAR_M, origin + direction * COUNCIL_REACH_M],
                                        axis=1))
    ri, gi = shapely.STRtree(lines).query(rays, predicate="intersects")
    if not len(ri):
        return out
    pts, which = shapely.get_coordinates(shapely.intersection(rays[ri], lines[gi]), return_index=True)
    if not len(pts):
        return out
    ray, line = ri[which], lines[gi[which]]
    at = shapely.line_locate_point(line, shapely.points(pts))
    a = shapely.get_coordinates(shapely.line_interpolate_point(line, np.maximum(at - 0.5, 0.0)))
    b = shapely.get_coordinates(shapely.line_interpolate_point(line, at + 0.5))
    d = b - a
    along = np.abs(d[:, 0] * np.cos(heading[ray]) + d[:, 1] * np.sin(heading[ray])) / np.maximum(np.hypot(*d.T), 1e-9)
    keep = along >= np.cos(np.radians(COUNCIL_PARALLEL_DEG))
    dist = np.hypot(*(pts[:, :2] - origin[ray]).T)
    np.fmin.at(out, ray[keep], dist[keep])
    return out


def stretch_boxes(xy: np.ndarray, mask: np.ndarray, chunk_m: float = 600.0,
                  margin_m: float = RAY_REACH_M + 10.0) -> list[tuple[float, float, float, float]]:
    """Boxes (min x, min y, max x, max y) covering the stations of `mask` (about a metre apart) and `margin_m` round
    them, a box per `chunk_m` of each stretch: what's fetched for the footpaths, not the whole route's box."""
    xy = np.asarray(xy, float)
    out = []
    every = max(int(chunk_m), 1)
    for a, b in _runs(mask):
        for c0 in range(a, b, every):
            p = xy[c0:min(c0 + every + 1, b)]
            lo, hi = p.min(axis=0) - margin_m, p.max(axis=0) + margin_m
            out.append((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))
    return out


def cbd_dense(s: np.ndarray, idx: np.ndarray, rays: dict, kerbed: np.ndarray) -> np.ndarray:
    """CBD-dense stations: where on either side, at least CBD_SHARE of the samples within CBD_WINDOW have a building's
    wall within CBD_SETBACK_M of the reserve's edge (or over it), for CBD_MIN_M or more. The research measured half the
    CBD's frontages within 0.5 m of the boundary, and a median setback of 5.5-5.8 m in the suburbs (its section 1c).
    It proposed a count of buildings instead (30 within 80 m), but theLIST's footprints have it backwards: the CBD's
    large buildings make 5-23 within 80 m of its streets, and a suburb's houses and sheds 24-39."""
    s = np.asarray(s, float)
    n = len(s)
    out = np.zeros(n, bool)
    if not len(idx):
        return out
    near = np.zeros(len(idx), bool)
    for bound, wall in rays.values():
        setback = wall - bound
        on = np.isfinite(setback) & (setback <= CBD_SETBACK_M)
        share = uniform_filter1d(on.astype(float), CBD_WINDOW, mode="nearest")
        near |= share >= CBD_SHARE
    hit = np.zeros(n, bool)
    hit[idx[near]] = True
    step = float(s[1] - s[0]) if n > 1 else 1.0
    gap = max(int(round(PATH_SAMPLE_M / step)), 1)
    hit = binary_dilation(hit, structure=np.ones(2 * gap + 1, bool)) & kerbed
    return stretches(hit, s, CBD_MIN_M, CBD_MIN_M) & kerbed


def summary_line(summary: dict) -> str:
    """The kerb plan's summary (kerbs.json) as one line for the build's log, e.g. "kerbs: kerbed 2287 m, kerb to kerb
    median 15.8 m (p10 12.1, p90 18.9), council hobart (ok), bare L 104 m / R 0 m, footpath sides from parcels 2724 m,
    standard widths 1056 m (parcels ok)". A track with none says why: "kerbs: kerbed 0 m (off: job)", or "(no town
    stretch)" where the rule found none."""
    kerbed = float(summary.get("kerbed_m") or 0.0)
    if kerbed <= 0:
        return f"kerbs: kerbed 0 m ({'off: ' + str(summary['off']) if summary.get('off') else 'no town stretch'})"
    parts = [f"kerbs: kerbed {kerbed:.0f} m"]
    kk = summary.get("kerb_to_kerb_m") or {}
    if kk:
        parts.append(f"kerb to kerb median {kk['median']:.1f} m (p10 {kk['p10']:.1f}, p90 {kk['p90']:.1f})")
    if summary.get("divided_m"):
        parts.append(f"divided carriageways {summary['divided_m']:.0f} m")
    parts.append(f"council {', '.join(summary.get('council') or ['none'])} ({summary.get('council_lines', 'none')})")
    bare = summary.get("council_bare_m")
    if bare:
        parts.append(f"bare L {bare.get('L', 0.0):.0f} m / R {bare.get('R', 0.0):.0f} m")
    src = summary.get("footpath_source_m") or {}
    parts.append(f"footpath sides from parcels {src.get('parcels', 0.0):.0f} m, standard widths "
                 f"{src.get('standard', 0.0):.0f} m (parcels {summary.get('parcels', 'none')})")
    return ", ".join(parts)


def _stats(a: np.ndarray) -> dict:
    a = np.asarray(a, float)
    if not len(a):
        return {}
    return {"median": round(float(np.median(a)), 2), "p10": round(float(np.percentile(a, 10)), 2),
            "p90": round(float(np.percentile(a, 90)), 2)}


# -- section.npz -------------------------------------------------------------------------------------------------------

KEYS = ("kerb_left", "kerb_right", "path_left", "path_right", "paved_left", "paved_right", "concrete_left",
        "concrete_right")


def arrays(p: Plan) -> dict[str, np.ndarray]:
    """The plan's arrays as section.npz keeps them."""
    return {"kerb_left": p.kerb["L"], "kerb_right": p.kerb["R"], "path_left": p.path["L"], "path_right": p.path["R"],
            "paved_left": p.paved["L"], "paved_right": p.paved["R"], "concrete_left": p.concrete["L"],
            "concrete_right": p.concrete["R"]}


@dataclass
class Kerbs:
    """Kerbs as the later stages read them from section.npz: per side ("L", "R"), each station's weight (0..1), the
    footpath's width from the kerb face, its paved part and whether that's concrete."""
    k: dict
    path: dict
    paved: dict
    concrete: dict
    placed: set = field(default_factory=set)  # (side, station) of each post put on a footpath (`signals._Road.place`)

    @classmethod
    def from_section(cls, section) -> "Kerbs | None":
        """None for a section without kerbs (built before them, with `kerbs: none`, or with no town stretch)."""
        if "kerb_left" not in section:
            return None
        k = {"L": np.asarray(section["kerb_left"], float), "R": np.asarray(section["kerb_right"], float)}
        if not (k["L"].any() or k["R"].any()):
            return None
        return cls(k, {"L": np.asarray(section["path_left"], float), "R": np.asarray(section["path_right"], float)},
                   {"L": np.asarray(section["paved_left"], float), "R": np.asarray(section["paved_right"], float)},
                   {"L": np.asarray(section["concrete_left"], bool), "R": np.asarray(section["concrete_right"], bool)})

    def kerbed(self, side: str, at: float = 0.5) -> np.ndarray:
        return self.k[side] > at

    def face(self, side: str) -> np.ndarray:
        """Metres from the road's edge as drawn (the lip) to the kerb face, per station: CHANNEL_M where kerbed."""
        return CHANNEL_M * self.k[side]

    def without(self, blocked: dict | None, s: np.ndarray) -> "Kerbs":
        """These kerbs with none on `blocked` stations (per side), tapering off over KERB_TAPER_M either side."""
        if blocked is None:
            return self
        s = np.asarray(s, float)
        k = {}
        for side in "LR":
            b = np.asarray(blocked[side], bool)
            if not b.any():
                k[side] = self.k[side]
                continue
            free = ~_dilate(b, s, KERB_TAPER_M / 2)
            k[side] = np.minimum(self.k[side], ramp_in(free, s, KERB_TAPER_M) if free.any() else 0.0)
            k[side] = np.where(free, k[side], 0.0)
        return Kerbs(k, self.path, self.paved, self.concrete, self.placed)

    def ribbon(self) -> dict:
        """`mesh.build_ribbon`'s `path_l`/`path_r`: where kerbed, the ribbon's outer verge column stands at the
        footpath's back edge (from the lip: the channel and the footpath) and that much above the lip; NaN elsewhere.
        Between a kerb's ends and full height it's blended with the plain verge's 2 m at its fall."""
        from mapcreator.core import mesh

        out = {}
        for side in "LR":
            k = self.k[side]
            full_w = CHANNEL_M + self.path[side]
            full_h = RAMP_RISE_M + PATH_FALL * np.maximum(self.path[side] - (KC_WIDTH_M - CHANNEL_M), 0.0)
            width = k * full_w + (1 - k) * mesh.VERGE_M
            rise = k * full_h - (1 - k) * mesh.VERGE_M * mesh.VERGE_FALL
            use = k > 0
            out[f"path_{side.lower()}"] = (np.where(use, width, np.nan), np.where(use, rise, np.nan))
        return out

    def rail_offsets(self) -> dict:
        """`roadside.guard_rails(behind=)`: where kerbed, a rail stands RAIL_BEHIND_M behind the footpath, on its back
        edge's height, as (metres beyond the road's edge, height above it) per side; blended with a verge's rail
        (`roadside.RAIL_OFFSET_M` out, on the verge's fall) where a kerb tapers; NaN elsewhere."""
        from mapcreator.core import mesh, roadside

        out = {}
        for side in "LR":
            k = self.k[side]
            back = CHANNEL_M + self.path[side]
            full_h = RAMP_RISE_M + PATH_FALL * np.maximum(self.path[side] - (KC_WIDTH_M - CHANNEL_M), 0.0)
            off = k * (back + RAIL_BEHIND_M) + (1 - k) * roadside.RAIL_OFFSET_M
            h = k * full_h - (1 - k) * roadside.RAIL_OFFSET_M * mesh.VERGE_FALL
            out[side] = (np.where(k > 0, off, np.nan), np.where(k > 0, h, np.nan))
        return out

    def quads(self, side: str) -> np.ndarray:
        """The ribbon's quads (between rows i and i + 1) a kerb runs along on `side`."""
        k = self.k[side]
        return (k[:-1] > 0) | (k[1:] > 0)

    def offset(self, side: str, i: int, behind: float) -> float | None:
        """Where a post `behind` the kerb face stands at station i, in metres beyond the lip; None where not kerbed."""
        if self.k[side][i] <= 0.5:
            return None
        return CHANNEL_M + behind

    def rise(self, side: str, i: int, beyond_lip: float) -> float:
        """The footpath's height above the lip, `beyond_lip` metres out, at station i (kerbed)."""
        return RAMP_RISE_M + PATH_FALL * max(beyond_lip - KC_WIDTH_M, 0.0)

    def intervals(self, s: np.ndarray, reverse: bool) -> list[tuple[float, float]]:
        """The kerbed stretches (either side) as (from, to) metres from a layout's start."""
        s = np.asarray(s, float)
        on = (self.k["L"] > 0) | (self.k["R"] > 0)
        out = []
        for a, b in _runs(on):
            lo, hi = float(s[a] - s[0]), float(s[b - 1] - s[0])
            total = float(s[-1] - s[0])
            out.append((total - hi, total - lo) if reverse else (lo, hi))
        return sorted(out)


# -- driveways ---------------------------------------------------------------------------------------------------------

def crossovers(s: np.ndarray, mouths: list, kerbs: Kerbs, clear: dict | None = None) -> tuple[dict, list[dict]]:
    """Each station's crossover weight per side (1 across a KC vehicular crossing, falling to 0 over its wings), for
    the driveways' mouths (`core.junctions.Mouth`s found with no margin) on kerbed stations, and a record of each.
    `clear`, per side: stations no crossover reaches (CROSSOVER_CLEAR_M either side of a side road's mouth)."""
    s = np.asarray(s, float)
    n = len(s)
    x = {"L": np.zeros(n), "R": np.zeros(n)}
    spans: dict = {"L": [], "R": []}
    records = []
    for m in mouths:
        side = m.side
        centre = (m.s0 + m.s1) / 2
        i = int(np.clip(np.searchsorted(s, centre), 0, n - 1))
        if kerbs.k[side][i] < 0.99:
            continue
        width = max(CROSSOVER_M, float(m.s1 - m.s0))
        lo, hi = centre - width / 2, centre + width / 2
        span = (s >= lo - WING_M) & (s <= hi + WING_M)
        if clear is not None and (np.asarray(clear[side], bool) & span).any():
            continue
        if (kerbs.k[side][span] < 0.99).any():
            continue
        spans[side].append((lo, hi))
        records.append({"side": side, "station_m": round(centre, 1), "width_m": round(width, 2),
                        "kind": m.junction.get("kind"), "way_id": m.junction.get("way_id")})
    for side in "LR":
        ordered = sorted(spans[side])
        if not ordered:
            continue
        merged = [list(ordered[0])]
        for lo, hi in ordered[1:]:  # wings dropped where too little kerb is left between two
            if lo - merged[-1][1] < 2 * WING_M + CROSSOVER_JOIN_M:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        # 1 on each crossing, falling linearly to 0 over WING_M either side, on full-height kerb only.
        w = np.zeros(n)
        for lo, hi in merged:
            d = np.maximum(np.maximum(lo - s, s - hi), 0.0)
            w = np.maximum(w, np.clip(1.0 - d / WING_M, 0.0, 1.0))
        x[side] = w * (kerbs.k[side] >= 0.99)
    return x, records


# -- meshes ------------------------------------------------------------------------------------------------------------

@dataclass
class Columns:
    """A kerb's cross-section row by row, in region coordinates (e, n, h), from the road's edge (the channel's lip)
    out to the ribbon's outer column: lip, the channel's invert at the face's toe, the face's top, the kerb's back
    (where the ramp meets the footpath), the paved footpath's edge and the outer column (the reserve's edge)."""
    lip: np.ndarray
    toe: np.ndarray
    top: np.ndarray
    back: np.ndarray
    paved: np.ndarray
    out: np.ndarray


def columns(lip: np.ndarray, out: np.ndarray, k: np.ndarray, x: np.ndarray, paved: np.ndarray) -> Columns:
    """The kerb's columns between `lip` and `out` (rows, 3: e, n, h), with weight `k` (0..1), crossover weight `x` and
    the paved footpath's width from the face (`paved`), per row. Each lies on the line from lip to out in plan; at
    k = 0 they're on that line in height too (the plain verge's plane), so a kerb's tapered end meets a verge with no
    step. Computed element by element, so two pieces given the same row get the same vertices."""
    lip, out = np.asarray(lip, float), np.asarray(out, float)
    k, x, paved = (np.asarray(a, float) for a in (k, x, paved))
    dx, dy = out[:, 0] - lip[:, 0], out[:, 1] - lip[:, 1]
    length = np.sqrt(dx * dx + dy * dy)
    ux, uy = dx / np.maximum(length, 1e-9), dy / np.maximum(length, 1e-9)
    plane = (out[:, 2] - lip[:, 2]) / np.maximum(length, 1e-9)
    room = np.maximum(length - 0.05, 0.0)
    back_off = np.minimum(RAMP_M + (CROSSOVER_KC_M - RAMP_M) * x, room)
    toe_off = np.minimum(CHANNEL_M, back_off)
    top_off = np.minimum((1 - x) * (CHANNEL_M + FACE_BATTER_M) + x * back_off, back_off)
    toe_rel = (1 - x) * -CHANNEL_DIP_M + x * CROSSOVER_LIP_M
    top_rel = RAMP_RISE_M - (1 - x) * KERB_TOP_M * TOP_FALL

    def at(off, rel):
        h = lip[:, 2] + k * rel + (1 - k) * plane * off
        return np.column_stack([lip[:, 0] + ux * off, lip[:, 1] + uy * off, h])

    back = at(back_off, RAMP_RISE_M)
    pe_off = np.clip(CHANNEL_M + paved, back_off, length)
    frac = (pe_off - back_off) / np.maximum(length - back_off, 1e-9)
    pe = np.column_stack([lip[:, 0] + ux * pe_off, lip[:, 1] + uy * pe_off,
                          back[:, 2] + (out[:, 2] - back[:, 2]) * frac])
    pe[pe_off >= length] = out[pe_off >= length]
    return Columns(lip.copy(), at(toe_off, toe_rel), at(top_off, top_rel), back, pe, out.copy())


def _strip(a: np.ndarray, b: np.ndarray, quads: np.ndarray, along: np.ndarray, tile: float,
           planar: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Quads between two columns (rows, 3) for the rows' quads selected: (positions, uvs, triangles), or None."""
    from mapcreator.core import mesh

    rows = len(a)
    if rows < 2 or not np.asarray(quads, bool).any():
        return None
    pos = np.stack([a, b], axis=1).reshape(-1, 3)
    tri = mesh._grid_triangles(rows, 2)
    keep = np.asarray(quads, bool)
    tri = tri[np.concatenate([keep, keep])]
    if planar:
        uv = pos[:, :2] / tile
        uv[:, 1] = -uv[:, 1]
    else:
        across = np.hypot(np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]), b[:, 2] - a[:, 2])
        uv = np.column_stack([np.stack([np.zeros(rows), across], axis=1).ravel(), np.repeat(along, 2)]) / tile
    return pos, uv, tri


def _area_ok(a: np.ndarray, b: np.ndarray, quads: np.ndarray, min_m: float = 0.01) -> np.ndarray:
    """The quads whose width (between the two columns) is at least `min_m` at either row."""
    w = np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
    return np.asarray(quads, bool) & ((w[:-1] >= min_m) | (w[1:] >= min_m))


@dataclass
class Pieces:
    """Kerb pieces by mesh kind and window, before packing: (positions in region e/n/h, uvs, triangles) lists."""
    visual: dict
    ramp: dict
    paths: dict  # (material, window) -> pieces
    grass: dict
    count: dict

    @classmethod
    def empty(cls) -> "Pieces":
        return cls({}, {}, {}, {}, {"rows": 0})


def add_strip(pieces: Pieces, cols: Columns, quads: np.ndarray, along: np.ndarray, window: np.ndarray,
              concrete: np.ndarray) -> None:
    """One side of one ribbon's (or flare's) kerb, its quads `quads` (rows - 1), into `pieces`: the drawn kerb (its
    channel, face and top as three strips, so the face's edges stay hard), the hidden ramp, the paved footpath by
    material (`concrete` per quad) and the grass behind it. `along`: each row's distance along, for texture
    coordinates; `window`: each quad's window number."""
    quads = np.asarray(quads, bool)
    if not quads.any():
        return
    window = np.asarray(window, int)
    for w in np.unique(window[quads]):
        q = quads & (window == w)
        for a, b in ((cols.lip, cols.toe), (cols.toe, cols.top), (cols.top, cols.back)):
            piece = _strip(a, b, q, along, KERB_TILE_M)
            if piece is not None:
                pieces.visual.setdefault(int(w), []).append(piece)
        piece = _strip(cols.lip, cols.back, q, along, KERB_TILE_M)
        if piece is not None:
            pieces.ramp.setdefault(int(w), []).append(piece)
        for conc in (False, True):
            qq = _area_ok(cols.back, cols.paved, q & (np.asarray(concrete, bool) == conc))
            piece = _strip(cols.back, cols.paved, qq, along, PATH_TILE_M, planar=True)
            if piece is not None:
                pieces.paths.setdefault((PATH_MATERIALS[conc], int(w)), []).append(piece)
        qq = _area_ok(cols.paved, cols.out, q)
        piece = _strip(cols.paved, cols.out, qq, along, GRASS_TILE_M, planar=True)
        if piece is not None:
            pieces.grass.setdefault(int(w), []).append(piece)
    pieces.count["rows"] = pieces.count.get("rows", 0) + int(quads.sum())


def ribbon_columns(rb, side: str, k: np.ndarray, x: np.ndarray, paved: np.ndarray) -> Columns:
    """`columns` along a ribbon (`mesh.Ribbon`) on `side`: its road edge and its verge column (the footpath's back
    edge where kerbed)."""
    if side == "L":
        edge, outer = rb.road_cols.start, rb.road_cols.start - 1
    else:
        edge, outer = rb.road_cols.stop - 1, rb.road_cols.stop
    lip = np.column_stack([rb.e[:, edge], rb.n[:, edge], rb.h[:, edge]])
    out = np.column_stack([rb.e[:, outer], rb.n[:, outer], rb.h[:, outer]])
    return columns(lip, out, k, x, paved)


def meshes(pieces: Pieces, frame, tag: str = "") -> list:
    """The kerb pieces as meshes, packed by kind and material per window: `KERB_<tag>###` (drawn, not physical),
    `1ROAD_<tag>KB###` (the hidden ramp), `1FOOTWAY_<tag>###A` or `C` (asphalt or concrete) and `1GRASS_<tag>K###`.
    The ramp's texture coordinates and material are the kerb's, though it's never drawn."""
    from mapcreator.core import mesh

    out = []
    for w, ps in sorted(pieces.visual.items()):
        out += mesh.soup_meshes(f"{VISUAL}{tag}{w:03d}", KERB_MATERIAL, ps, frame)
    for w, ps in sorted(pieces.ramp.items()):
        for m in mesh.soup_meshes(f"1ROAD_{tag}KB{w:03d}", KERB_MATERIAL, ps, frame):
            m.renderable = False
            out.append(m)
    for (material, w), ps in sorted(pieces.paths.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        out += mesh.soup_meshes(f"{FOOTWAY}{tag}{w:03d}{'C' if material.endswith('concrete') else 'A'}", material, ps,
                                frame)
    for w, ps in sorted(pieces.grass.items()):
        out += mesh.soup_meshes(f"1GRASS_{tag}K{w:03d}", "grass", ps, frame)
    return out


def windows(s: np.ndarray) -> np.ndarray:
    """Each quad's window (between rows i and i + 1), by its first row's station."""
    s = np.asarray(s, float)
    return (s[:-1] // WINDOW_M).astype(int)


# -- returns -----------------------------------------------------------------------------------------------------------

def return_radius(main_tags: dict, side_tags: dict) -> float:
    """A kerbed junction's kerb return (Q8): RETURN_MAJOR_M where either road is secondary or above, else RETURN_M."""
    major = _base_class(main_tags) in RETURN_MAJOR_CLASSES or _base_class(side_tags) in RETURN_MAJOR_CLASSES
    return RETURN_MAJOR_M if major else RETURN_M


def stub_weights(d: np.ndarray, kerbed: bool) -> np.ndarray:
    """A side road's stub's kerb weight row by row (`d`: metres along its ribbon from its first row): 1 for its first
    STUB_KERB_M, tapering to 0 over KERB_TAPER_M, where the junction's on a kerbed stretch; none elsewhere."""
    d = np.asarray(d, float)
    if not kerbed:
        return np.zeros(len(d))
    return 1.0 - _smoothstep((d - STUB_KERB_M) / KERB_TAPER_M)


def stub_ribbon(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`mesh.build_ribbon`'s path override for a kerbed stub's rows: the verge column 2 m out (the kerb, and grass at
    the kerb's height behind it), blended with the plain verge where the kerb tapers off; NaN where it's not kerbed."""
    from mapcreator.core import mesh

    k = np.asarray(k, float)
    full_h = RAMP_RISE_M + PATH_FALL * (mesh.VERGE_M - KC_WIDTH_M)
    width = np.full(len(k), mesh.VERGE_M)
    rise = k * full_h - (1 - k) * mesh.VERGE_M * mesh.VERGE_FALL
    use = k > 0
    return np.where(use, width, np.nan), np.where(use, rise, np.nan)



# -- scenery -----------------------------------------------------------------------------------------------------------

def band(xy: np.ndarray, heading: np.ndarray, rows: np.ndarray, side: str, lo: np.ndarray,
         hi: np.ndarray) -> shapely.Geometry:
    """The ground between offsets `lo` and `hi` (per station, metres from the centreline) on `side`, along each run of
    `rows`."""
    xy, heading = np.asarray(xy, float), np.asarray(heading, float)
    nrm = np.column_stack([-np.sin(heading), np.cos(heading)]) * (1.0 if side == "L" else -1.0)
    parts = []
    for a, b in _runs(rows):
        b = min(b + 1, len(xy))  # through the next station, so neighbouring pieces overlap
        if b - a < 2:
            continue
        inner = xy[a:b] + nrm[a:b] * np.asarray(lo, float)[a:b, None]
        outer = xy[a:b] + nrm[a:b] * np.asarray(hi, float)[a:b, None]
        parts.append(shapely.Polygon(np.vstack([inner, outer[::-1]])).buffer(0))
    return shapely.union_all(parts) if parts else shapely.Polygon()


def clear_zones(xy, heading, half_l, half_r, kerbs: Kerbs, tree_beyond: float,
                building_beyond: float) -> tuple[shapely.Geometry, shapely.Geometry]:
    """(where no tree stands, where no building may reach) along a road with kerbed stretches. Off them, as without
    kerbs: the road's widest half there plus `tree_beyond` or `building_beyond`, round the centreline. On them, per
    side: trees TREE_BEHIND_M or more behind the kerb face, and none from TREE_INTO_PATH_M behind it to the footpath's
    back edge (a street tree stands at the kerb, and the footpath stays walkable); buildings off the footpath, give or
    take BUILDING_SLACK_M (in the CBD half the walls stand on the reserve's edge). A side with no kerb where the other
    side has one (a council's Edge of Formation, a taper) is kept clear as a plain road's is, from its own edge: that
    edge is the one drawn, which a council's kerb line or the kerb-to-kerb width may have moved well past the plain
    stretches' widest."""
    xy, heading = np.asarray(xy, float), np.asarray(heading, float)
    half = {"L": np.asarray(half_l, float), "R": np.asarray(half_r, float)}
    kerbed = (kerbs.k["L"] > 0.5) | (kerbs.k["R"] > 0.5)
    plain = ~kerbed
    trees, buildings = [], []
    if plain.any():
        widest = float(max(half["L"][plain].max(), half["R"][plain].max()))
        for a, b in _runs(plain):
            piece = xy[max(a - 1, 0):min(b + 1, len(xy))]
            if len(piece) < 2:
                piece = np.vstack([piece, piece + 1e-3])
            line = shapely.LineString(piece)
            trees.append(line.buffer(widest + tree_beyond))
            buildings.append(line.buffer(widest + building_beyond))
    zero = np.zeros(len(xy))
    for side in "LR":
        on = kerbs.k[side] > 0.5
        face = half[side] + CHANNEL_M
        back = face + kerbs.path[side]
        trees.append(band(xy, heading, on, side, zero, face + TREE_BEHIND_M))
        trees.append(band(xy, heading, on, side, face + TREE_INTO_PATH_M, np.maximum(back, face + TREE_INTO_PATH_M)))
        buildings.append(band(xy, heading, on, side, zero, back - BUILDING_SLACK_M))
        bare = kerbed & ~on  # this side unkerbed, the other kerbed: the plain clearance, from this side's own edge
        trees.append(band(xy, heading, bare, side, zero, half[side] + tree_beyond))
        buildings.append(band(xy, heading, bare, side, zero, half[side] + building_beyond))
    return shapely.union_all(trees), shapely.union_all(buildings)
