"""Traffic signals and street-name blades: phase 1 of the traffic lights research (answered 24 Nov 2026).

**Where the signals come from** (`find`, the route stage): OpenStreetMap's tagged nodes on the roads (the OSM index's
`road_nodes`, layout 4; the roads query's own nodes from Overpass in development). A node with
`highway=traffic_signals`, `crossing=traffic_signals` or `crossing:signals=yes` is a signal node. Those within
`SITE_GROUP_M` of each other are one site, as the research grouped them.
- A site is **at a junction** of the route when a route node where three or more road segments meet is within
  `JUNCTION_REACH_M` of one of its nodes: the one nearest the site's middle. Hobart maps a site two ways, a node per
  approach (with `traffic_signals:direction`) or one node at the junction, and both end up there.
- Otherwise it's **mid-block** (a pedestrian crossing) when one of its nodes is on the route itself.
- Left out (`skipped`): a site off the route (at the next junction along a side road), one on a roundabout, and one
  within `END_CLEAR_M` of the route's first or last point, where the grid, the pits and the end walls are.
- **Legs:** every road segment leaving the site's node, each with which way it runs (`heading`, outwards), whether
  traffic may drive in along it (`inbound`, from `oneway`), its lanes in that direction (`lanes`: `lanes`, or
  `lanes:forward`/`lanes:backward`, or half of `lanes` on a two-way road), whether it's the route before or after the
  site or a side road on the left or right, and whether a crossing is mapped on it (a crossing node within
  `LEG_CROSSING_M` of the site, within `LEG_ANGLE_DEG` of its line).

**What's drawn** (`build`, the mesh stage), after Tasmania's Specification 730 and standard drawings, adapted from
VicRoads, and Austroads' AGTM Part 10 (the research's 4a):
- **Corner posts:** "2T" posts, 114 mm galvanised tubes 3.5 m tall, `POST_OFFSET_M` beyond the road's edge on the
  verge (Spec 730: at least 0.5 m behind the kerb; there is no kerb yet). A junction has a post at each of its four
  corners: on each side of the route, just before and just after the side roads' mouths (`core.junctions.mouths`). A T
  junction's far side has its two posts opposite the side road's mouth. Every post keeps out of side roads', ramps',
  roundabouts' and driveways' mouths and off the route's bridges, as the furniture does: one due in one moves away from
  the junction to the first clear station, by at most `MAX_SHIFT_M`, or isn't put up.
- **Vehicle faces:** each approach (a leg traffic may drive in along) gets three: the primary on the near left post,
  the secondary on the far right and the tertiary on the far left (AGTM10 8.4.1-8.4.3), sharing posts as real sites
  do. Each is a three-aspect 200 mm lantern (red over amber over green) with type 3 visors on a black target board with
  a white border, its top `LANTERN_TOP_M` up (SD-101-202), facing back along the approach.
- **Mast arms** where an approach has `MAST_LANES` or more lanes (the research's Q4): its primary post is a
  `MAST_HEIGHT_M` column with an outreach from `MAST_ARMS_M` reaching over the second lane from the kerb, carrying a
  fourth face whose lantern clears the road by `MAST_CLEARANCE_M`.
- **Pedestrian lanterns and push buttons** on every leg (Q3): at both ends of a crosswalk across each leg, a
  two-aspect lantern (a red standing figure over a green walking one, our own simple figures) aimed at the far end,
  and a button at `BUTTON_M` on the side of the post away from the road (Spec 730 cl 730.14(h)). Where a site has
  crossings mapped, only the legs that have one; where it has none mapped, all of them.
- **Display** (Q1): out of normal service, as Australian signals show when they aren't controlling traffic: every
  vehicle face's amber flashes at 1 Hz, all in step, and the pedestrian lanterns are dark. The amber lenses are a
  material of their own, `AMBER_MATERIAL`, and `package.ext_config_ini` flashes it (`FLASH_*`). Red and green stay
  unlit.
- **Collision** (Q5): every signal and blade post is solid, an invisible box round it from a metre below the verge
  to `COLLIDER_TOP_M` (`1WALL_SIGNAL_k`). Lanterns, boards, arms and blades are visual only.

**Street-name blades** at every junction of two differently named roads (`blades_for`), from `junctions.json` and the
route's own names either side: black lettering on white, both faces, one blade per name, each along the road it
names, on two corner posts. Diagonal corners at a crossroads, the side road's two corners at a T, and where the route
turns (`TURN_DEG`) the corner on the outside of the turn. At a signalised junction they go on top of the signal post
at that corner. The names are OSM's, the street type abbreviated as blades have it (`abbreviate`), in `BLADE_CASE`.
The lettering is Overpass (`core.lettering`), drawn here into the atlas; nothing is traced from a standard's drawing.

**Materials** (one atlas, `atlas`): `MATERIAL` for the posts, housings, visors, buttons, the boards' backs and the
unlit lenses and figures; `RETRO_MATERIAL` for the boards' faces and the blades, which CSP brightens at night as it
does the speed signs (`NIGHT_DIFFUSE`); `AMBER_MATERIAL` for the amber lenses. None has specular on its lettered or
bordered faces (`KN5_PROPERTIES`): the night washout lesson of the closures' signs.

Sizes the sources don't give are marked as estimates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from mapcreator.core import lettering
from mapcreator.core.centreline import cumulative
from mapcreator.core.scene import MAX_VERTICES, Z_SIGN, LocalFrame, Mesh

FORMAT_VERSION = 1

# -- finding the sites (route stage) --------------------------------------------------------------------------------
SITE_GROUP_M = 40.0  # signal nodes this near each other are one site (the research's `analyse.py`)
JUNCTION_REACH_M = 45.0  # a site is at a junction whose node is this near one of its signal nodes (likewise)
ON_ROUTE_M = 1.0  # a mid-block site's node is on the route
END_CLEAR_M = 70.0  # no site, and no blade, this near the route's ends: the grid, pits and hot-lap start, the end walls
LEG_CROSSING_M = 30.0  # a crossing node this near the site's node, along a leg, is that leg's crossing
LEG_ANGLE_DEG = 35.0
STOP_LINE_M = 40.0  # a signal node on an inbound leg this near the site's node is that approach's stop line

# -- drawing (mesh stage) --------------------------------------------------------------------------------------------
POST_OFFSET_M = 0.75  # beyond the road's edge: Spec 730's "at least 500 mm behind the kerb face"; Austroads 0.6-1 m
POST_GAP_M = 0.5  # beyond the side road's mouth
MAX_SHIFT_M = 15.0  # how far a post may move away from the junction to stand clear of another mouth
CORNER_REACH_M = 25.0  # a corner is never further than this from its junction, before such a move
# A driveway's or service way's mouth longer than this along the route is a way running beside it (a lay-by, a bus
# lane), not a mouth: it doesn't keep a junction's corner posts away (the mesh stage leaves it out of their mask).
ALONGSIDE_MOUTH_M = 30.0
JUNCTION_HALF_M = 10.0  # a junction with no side road in junctions.json: its corners this far either side (estimate)
CROSSWALK_HALF_M = 3.0  # a mid-block crossing's posts stand this far either side of its node (estimate)
POST_D = 0.114  # 100 NB tube (Spec 730 cl 730.04)
POST_TOP_M = 3.5  # a 2T post's top above the footpath (3.38 m plus its cap; estimate from SD-101-005)
POST_SIDES = 8
# A vehicle lantern: three 200 mm aspects (below 80 km/h, so every one in a town) in a housing, on a target board.
LANTERN_TOP_M = 3.40  # read off SD-101-202
LANTERN = (0.30, 0.80, 0.22)  # housing width, height, depth (estimate)
BOARD = (0.56, 0.96, 0.02)  # target board width, height, thickness (estimate)
BOARD_GAP_M = 0.005
ASPECT_M = 0.20
VISOR_M = 0.16  # how far a type 3 visor reaches forward (estimate)
BRACKET_M = 0.20  # a lantern's back stands this far from the post's axis
PED_TOP_M = 3.12  # a pedestrian lantern's top (SD-101-202)
PED = (0.30, 0.56, 0.18)
BUTTON_M = 0.9  # the push button's centre above the footpath (Spec 730 cl 730.14(h))
BUTTON = (0.09, 0.17, 0.07)
MAST_LANES = 3  # an approach with this many lanes or more gets a mast arm (Q4)
MAST_HEIGHT_M = 6.6  # the column (estimate: the lantern under the arm clears the road by MAST_CLEARANCE_M)
MAST_D = 0.22
MAST_ARMS_M = (2.5, 3.7, 5.5)  # the outreaches SD-101-010 and SD-101-011 give
MAST_CLEARANCE_M = 5.5
LANE_W_M = 3.2  # to reach over the second lane from the kerb (AGTM10 8.4.6)
ARM_D = 0.14
COLLIDER_W = 0.16  # each post's box collider, square (a mast's: MAST_D + 0.04)
COLLIDER_TOP_M = 2.5
COLLIDER_BELOW_M = 1.0
VERGE_FALL = 0.06  # as `furniture`

# -- street-name blades --------------------------------------------------------------------------------------------
BLADE_H = 0.15  # one name on a 150 mm blade (the ACT's MIS 12, Table 12-5)
BLADE_LETTER_M = 0.10  # its letters
BLADE_MIN_M, BLADE_MAX_M = 0.5, 1.2  # MIS 12: Series C where D would take a blade over 1.2 m
BLADE_THICK = 0.02
BLADE_POST_D = 0.06
BLADE_OFFSET_M = 1.0  # a blade post's place beyond the road's edge, where there's no signal post
BLADE_TOPS_M = (2.95, 2.75, 2.55)  # the blades' tops on their own post, the first name's highest
SIGNAL_BLADE_TOPS_M = (3.95, 3.75, 3.55)  # on a signal post, carried up above its lanterns
TURN_DEG = 30.0  # the route turns at a junction by this much or more
# A side road leaving closer than this to the road ahead or behind (`off_mainline_deg`) gives no blade: a divided
# road's other carriageway, or a lane beside it (Hobart's "Macquarie Street Bus Contra-Lane").
BLADE_MIN_ANGLE_DEG = 35.0
BLADE_SAME_M = 40.0  # two named junctions this close with the same names (a divided side road's two arms): blades once
# The blades' case: "mixed" (as OSM writes the name, "Macquarie St") or "upper" ("MACQUARIE ST"). Mixed until the
# maintainer's own photos of Hobart's blades confirm it (Q6).
BLADE_CASE = "mixed"
# Street types abbreviated on the blades, as Australia Post and AS 4819 abbreviate them (an estimate of what Hobart's
# blades carry: Q6). Only the last word, and never the name's only word or one after "The".
ABBREVIATIONS = {
    "Street": "St", "Road": "Rd", "Avenue": "Ave", "Highway": "Hwy", "Drive": "Dr", "Place": "Pl", "Court": "Ct",
    "Crescent": "Cres", "Parade": "Pde", "Terrace": "Tce", "Close": "Cl", "Grove": "Gr", "Circuit": "Cct",
    "Boulevard": "Bvd", "Esplanade": "Esp", "Square": "Sq",
}

MATERIAL = "signal"
RETRO_MATERIAL = "signal_retro"
AMBER_MATERIAL = "signal_amber"
TEXTURE = "signals"
# The KN5's values by day. No specular on the boards' faces and the blades (their border and lettering), nor on the
# housings: AC adds the specular term on top of the texture, so the headlights lit black legends grey (the closures'
# lesson, `closures.KN5_PROPERTIES`).
KN5_PROPERTIES = {
    MATERIAL: {"ksAmbient": 0.4, "ksDiffuse": 0.6, "ksSpecular": 0.0, "ksSpecularEXP": 10.0},
    RETRO_MATERIAL: {"ksAmbient": 0.4, "ksDiffuse": 0.6, "ksSpecular": 0.0, "ksSpecularEXP": 10.0},
    AMBER_MATERIAL: {"ksAmbient": 0.4, "ksDiffuse": 0.6, "ksSpecular": 0.0, "ksSpecularEXP": 10.0},
}
# ksDiffuse at night: the boards' white border and the blades as the speed signs' faces (`furniture.NIGHT_DIFFUSE`).
NIGHT_DIFFUSE = {RETRO_MATERIAL: 2.0}
# The flashing amber (Q1): Austroads' "flashing yellow at a rate of 1 hertz with equal 'on' and 'off' times" (AGTM10
# 8.3.9), as CSP's own track configs flash "blinking yellow traffic lights" (`config/tracks/leipzig_stadtpark.ini`,
# CONDITION_2 TRAFFICLIGHTSOFF, and `pudsey.ini`, in acc-extension-config): a condition of INPUT=ONE with FLASHING_*
# keys, and a material adjustment of ksEmissive under it. Synced, so every junction flashes in step (a phase offset
# exists only for lights, not material adjustments). The colour and strength are an estimate, to tune in game:
# pudsey's yellow is 250, 250, 0 at 0.025.
FLASH_CONDITION = "SIGNALS_FLASHING"
FLASH_FREQUENCY_HZ = 1.0
FLASH_SMOOTHNESS = 0.1
FLASH_EMISSIVE = "255, 150, 0, 0.025"


def flash_condition() -> dict[str, str]:
    """The flashing condition's keys for `package.ext_config_ini` (as leipzig_stadtpark.ini's CONDITION_2 has them)."""
    return {"NAME": FLASH_CONDITION, "INPUT": "ONE", "FLASHING_FREQUENCY": f"{FLASH_FREQUENCY_HZ:g}",
            "FLASHING_SMOOTHNESS": f"{FLASH_SMOOTHNESS:g}", "FLASHING_SKIP_OFF_STATE": "0",
            "FLASHING_NOISE_AMPLITUDE": "0", "FLASHING_MIN_VALUE": "0.0", "FLASHING_SYNCED": "1"}


# =====================================================================================================================
# Finding the sites
# =====================================================================================================================

def is_signal(tags: dict) -> bool:
    return (tags.get("highway") == "traffic_signals" or tags.get("crossing") == "traffic_signals"
            or tags.get("crossing:signals") == "yes")


def is_crossing(tags: dict) -> bool:
    return tags.get("highway") == "crossing" or tags.get("crossing") not in (None, "no")


def _int(value) -> int | None:
    try:
        n = int(str(value).split(";")[0].strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def lanes_in(tags: dict, along_way: bool) -> int | None:
    """Lanes for traffic driving along the way (`along_way`) or against it, from `lanes`, `lanes:forward` and
    `lanes:backward`: all of `lanes` on a one-way road, half of them on a two-way one."""
    oneway = tags.get("oneway")
    if oneway in ("yes", "true", "1", "-1") or tags.get("junction") in ("roundabout", "circular"):
        return _int(tags.get("lanes"))
    own = _int(tags.get("lanes:forward" if along_way else "lanes:backward"))
    if own is not None:
        return own
    total = _int(tags.get("lanes"))
    return max(total // 2, 1) if total is not None else None


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.hypot(*v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _angle(a, b) -> float:
    """The angle between two directions, degrees."""
    a, b = _unit(a), _unit(b)
    return math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))


def find(g, route_xy: np.ndarray, nodes: Sequence[dict], *, to_lonlat=None) -> dict:
    """The signal sites along the route, from the graph it was routed on (`osm_graph.RoadGraph`), its polyline and the
    tagged road nodes (each `{"id", "xy", "tags"}`, xy in the graph's CRS). Returns `{"version", "rule", "sites",
    "skipped"}` for `signals.json` (docs/ac-format.md)."""
    route_xy = np.asarray(route_xy, float)
    s_route = cumulative(route_xy)
    length = float(s_route[-1])
    line = shapely.LineString(route_xy)
    rule = {"site_group_m": SITE_GROUP_M, "junction_reach_m": JUNCTION_REACH_M, "end_clear_m": END_CLEAR_M,
            "leg_crossing_m": LEG_CROSSING_M, "leg_angle_deg": LEG_ANGLE_DEG, "stop_line_m": STOP_LINE_M}
    out = {"version": FORMAT_VERSION, "rule": rule, "sites": [], "skipped": []}
    by_id = {int(n["id"]): n for n in nodes}
    signals = [n for n in nodes if is_signal(n.get("tags") or {})]
    if not signals or len(route_xy) < 2:
        return out
    sxy = np.array([n["xy"] for n in signals], float)
    near = shapely.distance(line, shapely.points(sxy)) <= JUNCTION_REACH_M
    signals, sxy = [n for n, k in zip(signals, near) if k], sxy[near]
    if not signals:
        return out
    # Groups: single linkage within SITE_GROUP_M.
    pairs = cKDTree(sxy).query_pairs(SITE_GROUP_M, output_type="ndarray")
    n = len(signals)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else \
        coo_matrix((n, n))
    _, label = connected_components(adj, directed=False)

    # The route's nodes: its vertices that are graph nodes (a spliced waypoint isn't one), and where three or more road
    # segments meet.
    tree = cKDTree(g.xy)
    d, gi = tree.query(route_xy)
    on_node = d < 0.01
    degree = np.bincount(np.concatenate([g.seg_a, g.seg_b]), minlength=len(g.xy))
    ring_nodes = {int(x) for w, a, b in zip(g.seg_way, g.seg_a, g.seg_b)
                  if (g.ways[w]["tags"] or {}).get("junction") in ("roundabout", "circular") for x in (a, b)}
    route_k = np.flatnonzero(on_node)
    junction_k = route_k[degree[gi[route_k]] >= 3]

    crossings = [m for m in nodes if is_crossing(m.get("tags") or {})]
    cxy = np.array([m["xy"] for m in crossings], float).reshape(-1, 2)
    used: dict[int, dict] = {}
    for grp in range(int(label.max()) + 1):
        members = [signals[i] for i in np.flatnonzero(label == grp)]
        mxy = np.array([m["xy"] for m in members], float)
        ids = sorted(int(m["id"]) for m in members)
        centre = mxy.mean(axis=0)
        k, kind = None, None
        if len(junction_k):
            jd = np.min(np.hypot(*(route_xy[junction_k][:, None, :] - mxy[None]).transpose(2, 0, 1)), axis=1)
            ok = junction_k[jd <= JUNCTION_REACH_M]
            if len(ok):
                k = int(ok[np.argmin(np.hypot(*(route_xy[ok] - centre).T))])
                kind = "junction"
        if k is None:
            # Mid-block: a signal node that is one of the route's own nodes.
            ids_on = [int(m["id"]) for m in members]
            on_route = [kk for kk in route_k if int(g.osm_node[gi[kk]]) in ids_on]
            if on_route:
                k, kind = int(on_route[0]), "mid_block"
        if k is None:
            out["skipped"].append({"signals": ids, "reason": "off_route"})
            continue
        node = int(gi[k])
        at = float(s_route[k])
        if node in ring_nodes:
            out["skipped"].append({"signals": ids, "node": int(g.osm_node[node]), "reason": "roundabout"})
            continue
        if at < END_CLEAR_M or length - at < END_CLEAR_M:
            out["skipped"].append({"signals": ids, "node": int(g.osm_node[node]), "reason": "route_end"})
            continue
        if node in used:  # two groups at one junction (a divided road's): one site
            used[node]["signals"] = sorted(set(used[node]["signals"]) | set(ids))
            continue
        site = {"node": int(g.osm_node[node]), "kind": kind, "at_m": round(at, 2),
                "xy": [round(float(v), 2) for v in g.xy[node]], "signals": ids}
        if to_lonlat is not None:
            lon, lat = to_lonlat.transform(*g.xy[node])
            site["lonlat"] = [round(float(lon), 7), round(float(lat), 7)]
        site["legs"] = _legs(g, node, route_xy, k)
        _crossings_on_legs(site, g.xy[node], crossings, cxy)
        _stop_lines(site, g.xy[node], members, by_id)
        used[node] = site
    out["sites"] = sorted(used.values(), key=lambda st: st["at_m"])
    for i, st in enumerate(out["sites"]):
        st["id"] = i
    return out


def _legs(g, node: int, route_xy: np.ndarray, k: int) -> list[dict]:
    """Every road segment leaving `node`, as a leg (module docstring)."""
    here = g.xy[node]
    before = _unit(route_xy[k - 1] - route_xy[k]) if k > 0 else None
    after = _unit(route_xy[k + 1] - route_xy[k]) if k + 1 < len(route_xy) else None
    ahead = _unit((after if after is not None else -before) - (before if before is not None else -after))
    legs = []
    for j in np.flatnonzero((g.seg_a == node) | (g.seg_b == node)):
        out_end = int(g.seg_b[j]) if int(g.seg_a[j]) == node else int(g.seg_a[j])
        vec = _unit(g.xy[out_end] - here)
        inbound_along_way = int(g.seg_b[j]) == node  # driving in along the way's own direction
        inbound = bool(g.seg_fwd[j]) if inbound_along_way else bool(g.seg_rev[j])
        tags = g.ways[int(g.seg_way[j])]["tags"] or {}
        route = None
        if before is not None and _angle(vec, before) < 2.0:
            route = "before"
        elif after is not None and _angle(vec, after) < 2.0:
            route = "after"
        side = None
        if route is None:
            side = "L" if ahead[0] * vec[1] - ahead[1] * vec[0] > 0 else "R"
        legs.append({"way_id": int(g.ways[int(g.seg_way[j])]["id"]), "highway": tags.get("highway"),
                     "name": tags.get("name"), "route": route, "side": side,
                     "heading": round(math.atan2(vec[1], vec[0]), 4), "inbound": inbound,
                     "lanes": lanes_in(tags, inbound_along_way), "crossing": False, "stop_m": None})
    return legs


def _leg_for(site: dict, v: np.ndarray) -> dict | None:
    best, angle = None, LEG_ANGLE_DEG
    for leg in site["legs"]:
        a = _angle(v, (math.cos(leg["heading"]), math.sin(leg["heading"])))
        if a < angle:
            best, angle = leg, a
    return best


def _crossings_on_legs(site: dict, here: np.ndarray, crossings: list[dict], cxy: np.ndarray) -> None:
    site["crossings"] = []
    if not len(cxy):
        site["crossings_mapped"] = False
        return
    d = np.hypot(*(cxy - here).T)
    for m, dist, p in zip(crossings, d, cxy):
        if dist > LEG_CROSSING_M or dist < 0.5:
            continue
        leg = _leg_for(site, p - here)
        if leg is not None:
            leg["crossing"] = True
            site["crossings"].append(int(m["id"]))
    site["crossings"].sort()
    site["crossings_mapped"] = bool(site["crossings"])


def _stop_lines(site: dict, here: np.ndarray, members: list[dict], by_id: dict) -> None:
    """Each inbound leg's stop line: the nearest signal node on it within STOP_LINE_M of the site's node."""
    for m in members:
        p = np.asarray(m["xy"], float)
        dist = float(np.hypot(*(p - here)))
        if dist < 0.5 or dist > STOP_LINE_M:
            continue
        leg = _leg_for(site, p - here)
        if leg is not None and leg["inbound"] and (leg["stop_m"] is None or dist < leg["stop_m"]):
            leg["stop_m"] = round(dist, 2)


def pedestrian_legs(site: dict) -> list[dict]:
    """The legs that get pedestrian lanterns and buttons (Q3): those with a crossing mapped, where the site has any
    mapped; every leg where it has none."""
    if site.get("crossings_mapped"):
        return [leg for leg in site["legs"] if leg.get("crossing")]
    return list(site["legs"])


# =====================================================================================================================
# Street-name blades
# =====================================================================================================================

def abbreviate(name: str) -> str:
    """A name as a blade carries it: its street type abbreviated (`ABBREVIATIONS`), in `BLADE_CASE`."""
    words = name.split()
    if len(words) >= 2 and words[-1] in ABBREVIATIONS and words[-2] != "The":
        words[-1] = ABBREVIATIONS[words[-1]]
    text = " ".join(words)
    return text.upper() if BLADE_CASE == "upper" else text


@dataclass
class BladeSite:
    """A junction whose roads have two names or more: where, which names, which way each runs, and its corners."""

    node: int
    station_m: float
    names: list[tuple[str, float]]  # (name as OSM has it, its road's heading at the junction, radians)
    corners: list[tuple[str, str]]  # (side, "before" or "after")


def blades_for(junctions: Sequence[dict], s: np.ndarray, xy: np.ndarray, heading: np.ndarray,
               run_index: np.ndarray, runs: Sequence[dict], station_of) -> list[BladeSite]:
    """The named junctions along the route (module docstring): each junction node with two names or more among the
    route's own either side of it and its side roads'. `station_of(j)` is a junction's station on the centreline."""
    s = np.asarray(s, float)
    groups: dict[int, list[dict]] = {}
    for j in junctions:
        groups.setdefault(int(j["node"]), []).append(j)
    out = []
    for node, js in groups.items():
        at = float(np.mean([station_of(j) for j in js]))
        if at < float(s[0]) + END_CLEAR_M or float(s[-1]) - at < END_CLEAR_M:
            continue
        i0 = int(np.clip(np.searchsorted(s, at - 15.0), 0, len(s) - 1))
        i1 = int(np.clip(np.searchsorted(s, at + 15.0), 0, len(s) - 1))
        names: list[tuple[str, float]] = []

        def add(name, h):
            if name and name not in [n for n, _ in names]:
                names.append((name, float(h)))

        add((runs[int(run_index[i0])].get("tags") or {}).get("name"), heading[i0])
        add((runs[int(run_index[i1])].get("tags") or {}).get("name"), heading[i1])
        ia = int(np.clip(np.searchsorted(s, at), 0, len(s) - 1))
        for j in js:
            if float(j.get("off_mainline_deg", 90.0)) >= BLADE_MIN_ANGLE_DEG:
                add(j.get("name"), heading[ia] + math.radians(float(j.get("angle_deg") or 90.0)))
        if len(names) < 2:
            continue
        if any(abs(o.station_m - at) <= BLADE_SAME_M and {n for n, _ in o.names} == {n for n, _ in names} for o in out):
            continue
        sides = sorted({"L" if j["side"] == "left" else "R" for j in js})
        if len(sides) == 2:
            corners = [("L", "before"), ("R", "after")]
        else:
            corners = [(sides[0], "before"), (sides[0], "after")]
        turn = float(np.mean([float(j.get("route_turn_deg") or 0.0) for j in js]))
        if abs(turn) >= TURN_DEG:
            outside = "R" if turn > 0 else "L"  # turning left, the outside is on the right
            if (outside, "after") not in corners:
                corners = [(outside, "after"), corners[0] if corners[0][0] != outside else corners[1]]
        out.append(BladeSite(node, at, names, corners))
    return sorted(out, key=lambda b: b.station_m)


# =====================================================================================================================
# The texture
# =====================================================================================================================

CELL = 64
ATLAS_W = 512
BLADE_CELL = (ATLAS_W, CELL)  # one name per row, its ink from the left
BLACK, WHITE = (12, 12, 12), (246, 246, 242)


@dataclass
class Atlas:
    image: Image.Image
    cells: dict  # name -> (u0, v0, u1, v1)
    blade_m: dict  # blade text -> its blade's length, metres


def _figure(d: ImageDraw.ImageDraw, box, walking: bool, colour) -> None:
    """Our own simple pedestrian figure: a head and a body, standing (arms at the side) or striding."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    r = 0.09 * h
    d.ellipse([cx - r, y0 + 0.08 * h, cx + r, y0 + 0.08 * h + 2 * r], fill=colour)
    t = max(int(0.09 * w), 2)
    top, hip, foot = y0 + 0.3 * h, y0 + 0.62 * h, y0 + 0.92 * h
    if walking:
        d.line([(cx, top), (cx - 0.02 * w, hip)], fill=colour, width=t + 2)
        d.line([(cx - 0.02 * w, hip), (cx - 0.2 * w, foot)], fill=colour, width=t)
        d.line([(cx - 0.02 * w, hip), (cx + 0.2 * w, foot)], fill=colour, width=t)
        d.line([(cx, top + 0.04 * h), (cx - 0.18 * w, hip - 0.04 * h)], fill=colour, width=t)
        d.line([(cx, top + 0.04 * h), (cx + 0.18 * w, hip - 0.08 * h)], fill=colour, width=t)
    else:
        d.rectangle([cx - 0.1 * w, top, cx + 0.1 * w, hip], fill=colour)
        d.line([(cx - 0.05 * w, hip), (cx - 0.07 * w, foot)], fill=colour, width=t)
        d.line([(cx + 0.05 * w, hip), (cx + 0.07 * w, foot)], fill=colour, width=t)
        d.line([(cx - 0.12 * w, top + 0.02 * h), (cx - 0.16 * w, hip)], fill=colour, width=t)
        d.line([(cx + 0.12 * w, top + 0.02 * h), (cx + 0.16 * w, hip)], fill=colour, width=t)


def atlas(blade_texts: Sequence[str]) -> Atlas:
    """Every face the signals and blades need, in one texture: galvanised steel, the black housing, the target board
    (black with a white border), the lenses (dark: unlit), the pedestrian figures, the button and one row per blade."""
    texts = list(dict.fromkeys(blade_texts))
    rows = 2 + len(texts)
    img = Image.new("RGB", (ATLAS_W, rows * CELL), (40, 40, 40))
    d = ImageDraw.Draw(img)
    cells: dict = {}

    def cell(name, col, row, w=1, h=1):
        x0, y0 = col * CELL, row * CELL
        x1, y1 = x0 + w * CELL, y0 + h * CELL
        cells[name] = (x0 / img.width, y0 / img.height, x1 / img.width, y1 / img.height)
        return x0, y0, x1, y1

    x0, y0, x1, y1 = cell("galv", 0, 0)
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(156, 160, 158))
    x0, y0, x1, y1 = cell("black", 1, 0)
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(22, 22, 22))
    x0, y0, x1, y1 = cell("board", 2, 0, 1, 2)  # 0.56 x 0.96 m: a cell's width, two tall
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=WHITE)
    border = max(int(0.07 / BOARD[0] * CELL), 3)  # a white border of about 70 mm (estimate)
    d.rectangle([x0 + border, y0 + border, x1 - 1 - border, y1 - 1 - border], fill=BLACK)
    x0, y0, x1, y1 = cell("board_back", 3, 0)
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(120, 122, 120))
    for name, col, colour in (("lens_red", 4, (70, 8, 8)), ("lens_green", 5, (6, 52, 34)),
                              ("lens_amber", 6, (96, 58, 4))):
        x0, y0, x1, y1 = cell(name, col, 0)
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(22, 22, 22))
        m = 3
        d.ellipse([x0 + m, y0 + m, x1 - 1 - m, y1 - 1 - m], fill=colour)
    x0, y0, x1, y1 = cell("button", 7, 0)
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(58, 60, 60))
    cx, cy = (x0 + x1) / 2, y0 + 0.62 * CELL
    d.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(176, 178, 176))
    d.polygon([(cx - 8, cy + 3), (cx + 6, cy + 3), (cx + 6, cy + 7), (cx + 12, cy), (cx + 6, cy - 7), (cx + 6, cy - 3),
               (cx - 8, cy - 3)], fill=(40, 40, 40))  # the tactile arrow, pointing the way to walk
    for name, col, walking, colour in (("ped_red", 4, False, (92, 14, 14)), ("ped_green", 5, True, (10, 70, 48))):
        x0, y0, x1, y1 = cell(name, col, 1)
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(18, 18, 18))
        _figure(d, (x0 + 8, y0 + 4, x1 - 8, y1 - 4), walking, colour)
    blade_m = {}
    for i, text in enumerate(texts):
        x0, y0 = 0, (2 + i) * CELL
        x1, y1 = ATLAS_W, y0 + CELL
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=WHITE)
        letter_px = BLADE_LETTER_M / BLADE_H * CELL  # the letters' height on the blade, in pixels
        f = lettering.fit(d, "H", 10_000, letter_px)  # sized by a capital's height
        bx0, by0, bx1, by1 = d.textbbox((0, 0), text, font=f)
        margin = 0.5 * letter_px
        width_px = (bx1 - bx0) + 2 * margin
        if width_px > ATLAS_W:  # longer than the row holds: narrower letters, as Series C on a long blade
            scale = (ATLAS_W - 2 * margin) / (bx1 - bx0)
            f = lettering.font(max(int(f.size * scale), 6))
            bx0, by0, bx1, by1 = d.textbbox((0, 0), text, font=f)
            width_px = ATLAS_W
        width_px = min(max(width_px, BLADE_MIN_M / BLADE_H * CELL), ATLAS_W)
        # Centred from the capitals' tops to the descenders' feet, so a "y" or "q" stays inside the border.
        hx0, hy0, hx1, hy1 = d.textbbox((0, 0), "Hgy", font=f)
        y_text = y0 + (CELL - (hy1 - hy0)) / 2 - hy0
        d.text((x0 + (width_px - (bx1 - bx0)) / 2 - bx0, y_text), text, fill=BLACK, font=f)
        b = 2
        d.rectangle([x0 + b, y0 + b, int(x0 + width_px) - 1 - b, y1 - 1 - b], outline=BLACK, width=2)
        cells[f"blade:{text}"] = (x0 / img.width, y0 / img.height, (x0 + width_px) / img.width, y1 / img.height)
        blade_m[text] = round(float(min(max(width_px / CELL * BLADE_H, BLADE_MIN_M), BLADE_MAX_M)), 3)
    return Atlas(img, cells, blade_m)


# =====================================================================================================================
# Geometry
# =====================================================================================================================

_UP = np.array([0.0, 0.0, 1.0])


def _right(facing) -> np.ndarray:
    """The right hand of someone looking at a face that points along `facing` (2D): (-f_y, f_x), in 3D."""
    f = _unit(facing)
    return np.array([-f[1], f[0], 0.0])


class _Builder:
    """Quads, prisms and boxes in region coordinates (e, n, h), one part per material, and invisible collider boxes."""

    def __init__(self, tex: Atlas, materials: Sequence[str] = (MATERIAL, RETRO_MATERIAL, AMBER_MATERIAL)):
        self.tex = tex
        self.parts = {m: ([], [], [], []) for m in materials}
        self.material = materials[0]  # the plain one: prisms, beams and the colliders take it
        self.colliders: list[tuple] = []  # (e, n, h0, h1, half width)

    def quad(self, c, right, up, w, h, normal, cell, material=None, flip_u=False):
        """A rectangle centred on `c` (3D), `w` along `right` and `h` along `up`, seen from the side `normal` points
        to, with an atlas cell on it."""
        u0, v0, u1, v1 = self.tex.cells[cell]
        if flip_u:
            u0, u1 = u1, u0
        r, u = np.asarray(right, float) * w / 2, np.asarray(up, float) * h / 2
        c = np.asarray(c, float)
        pos, nrm, uv, tri = self.parts[material or self.material]
        i = len(pos)
        pos.extend([c - r - u, c + r - u, c + r + u, c - r + u])
        nrm.extend([np.asarray(normal, float)] * 4)
        uv.extend([(u0, v1), (u1, v1), (u1, v0), (u0, v0)])
        tri.extend([(i, i + 1, i + 2), (i, i + 2, i + 3)])

    def disc(self, c, right, up, radius, normal, cell, material=None, sides=12):
        """A round face (a lens) centred on `c`, a fan of `sides` triangles, its texture the circle inscribed in the
        atlas cell. Round, so a lens lit by ksEmissive glows round: CSP adds the emissive colour over the whole
        face, whatever the texture."""
        u0, v0, u1, v1 = self.tex.cells[cell]
        uc, vc, ur, vr = (u0 + u1) / 2, (v0 + v1) / 2, (u1 - u0) / 2 * 0.9, (v1 - v0) / 2 * 0.9
        right, up, c = np.asarray(right, float), np.asarray(up, float), np.asarray(c, float)
        pos, nrm, uv, tri = self.parts[material or self.material]
        i = len(pos)
        pos.append(c)
        nrm.append(np.asarray(normal, float))
        uv.append((uc, vc))
        for k in range(sides):
            a = 2 * math.pi * k / sides
            pos.append(c + (right * math.cos(a) + up * math.sin(a)) * radius)
            nrm.append(np.asarray(normal, float))
            uv.append((uc + ur * math.cos(a), vc - vr * math.sin(a)))
        for k in range(sides):
            tri.append((i, i + 1 + k, i + 1 + (k + 1) % sides))

    def polygon(self, c, right, up, outline, normal, cell, material=None):
        """A flat convex face (an octagon, a triangle) centred on `c`: `outline`'s points, (x, y) metres along `right`
        and `up`, as a fan from their middle. The atlas cell holds the outline's bounding box."""
        pts = np.asarray(outline, float)
        u0, v0, u1, v1 = self.tex.cells[cell]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
        fx = (pts[:, 0] - lo[0]) / span[0]
        fy = (pts[:, 1] - lo[1]) / span[1]
        right, up, c = np.asarray(right, float), np.asarray(up, float), np.asarray(c, float)
        pos, nrm, uv, tri = self.parts[material or self.material]
        i = len(pos)
        mid = pts.mean(axis=0)
        pos.append(c + right * mid[0] + up * mid[1])
        nrm.append(np.asarray(normal, float))
        uv.append((u0 + (u1 - u0) * (mid[0] - lo[0]) / span[0], v1 - (v1 - v0) * (mid[1] - lo[1]) / span[1]))
        for (x, y), a, b in zip(pts, fx, fy):
            pos.append(c + right * x + up * y)
            nrm.append(np.asarray(normal, float))
            uv.append((u0 + (u1 - u0) * a, v1 - (v1 - v0) * b))
        k = len(pts)
        for j in range(k):
            tri.append((i, i + 1 + j, i + 1 + (j + 1) % k))

    def box(self, c, facing, w, h, depth, cells, materials=None):
        """A box centred on `c` (3D), its front facing `facing` (2D), `w` wide, `h` tall and `depth` deep. `cells`:
        front, back, sides (left and right), top and bottom; `materials` likewise (default `MATERIAL`)."""
        f = np.array([*_unit(facing), 0.0])
        r = _right(facing)
        mats = materials or (self.material,) * 6
        c = np.asarray(c, float)
        front, back, side, top, bottom = cells
        self.quad(c + f * depth / 2, r, _UP, w, h, f, front, mats[0])
        self.quad(c - f * depth / 2, -r, _UP, w, h, -f, back, mats[1])
        self.quad(c + r * w / 2, -f, _UP, depth, h, r, side, mats[2])
        self.quad(c - r * w / 2, f, _UP, depth, h, -r, side, mats[3])
        if top is not None:
            self.quad(c + _UP * h / 2, r, -f, w, depth, _UP, top, mats[4])
        if bottom is not None:
            self.quad(c - _UP * h / 2, r, f, w, depth, -_UP, bottom, mats[5])

    def prism(self, e, n, h0, h1, d, cell="galv", sides=POST_SIDES, cap=True):
        """An upright round post as a prism of `sides` faces, smooth shaded, with a flat cap."""
        u0, v0, u1, v1 = self.tex.cells[cell]
        pos, nrm, uv, tri = self.parts[self.material]
        a = np.linspace(0.0, 2 * math.pi, sides + 1)
        ring = np.column_stack([np.cos(a), np.sin(a)])
        i = len(pos)
        for k, (cx, cy) in enumerate(ring):
            for hh, vv in ((h0, v1), (h1, v0)):
                pos.append(np.array([e + cx * d / 2, n + cy * d / 2, hh]))
                nrm.append(np.array([cx, cy, 0.0]))
                uv.append((u0 + (u1 - u0) * k / sides, vv))
        for k in range(sides):
            a0, a1 = i + 2 * k, i + 2 * (k + 1)
            tri.extend([(a0, a1, a1 + 1), (a0, a1 + 1, a0 + 1)])
        if cap:
            j = len(pos)
            pos.append(np.array([e, n, h1]))
            nrm.append(_UP.copy())
            uv.append(((u0 + u1) / 2, (v0 + v1) / 2))
            for k in range(sides):
                pos.append(np.array([e + ring[k, 0] * d / 2, n + ring[k, 1] * d / 2, h1]))
                nrm.append(_UP.copy())
                uv.append(((u0 + u1) / 2, (v0 + v1) / 2))
            for k in range(sides):
                tri.append((j, j + 1 + k, j + 1 + (k + 1) % sides))

    def beam(self, p0, p1, d, cell="galv"):
        """A horizontal-ish square tube from p0 to p1 (3D), `d` across: four long faces."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        axis = p1 - p0
        length = float(np.linalg.norm(axis))
        ax = axis / length
        side = np.cross(ax, _UP)
        if float(np.linalg.norm(side)) < 1e-6:  # an upright tube (a hanger): any horizontal side will do
            side = np.array([1.0, 0.0, 0.0])
        side /= float(np.linalg.norm(side))
        up = np.cross(side, ax)
        mid = (p0 + p1) / 2
        for nv in (side, -side, up, -up):
            other = np.cross(nv, ax)
            self.quad(mid + nv * d / 2, ax, other, length, d, nv, cell)

    def collider(self, e, n, h0, h1, w):
        self.colliders.append((e, n, h0, h1, w / 2))

    def meshes(self, frame: LocalFrame, prefix: str = "SIGNAL") -> list[Mesh]:
        out = []
        for material, (pos, nrm, uv, tri) in self.parts.items():
            if pos:
                out += _pack(frame, f"{prefix}_", material, np.array(pos, float), np.array(nrm, float),
                             np.array(uv, float), np.array(tri, dtype=np.int64), start=len(out))
        if self.colliders:
            pos, nrm, uv, tri = [], [], [], []
            for e, n, h0, h1, hw in self.colliders:
                c = np.array([e, n, 0.0])
                for nv in (np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]),
                           np.array([0, -1.0, 0])):
                    right = np.array([-nv[1], nv[0], 0.0])
                    base = c + nv * hw
                    i = len(pos)
                    pos += [base - right * hw + _UP * h0, base + right * hw + _UP * h0,
                            base + right * hw + _UP * h1, base - right * hw + _UP * h1]
                    nrm += [nv] * 4
                    uv += [(0, 0)] * 4
                    tri += [(i, i + 1, i + 2), (i, i + 2, i + 3)]
                i = len(pos)
                pos += [c + np.array([-hw, -hw, h1]), c + np.array([hw, -hw, h1]), c + np.array([hw, hw, h1]),
                        c + np.array([-hw, hw, h1])]
                nrm += [_UP] * 4
                uv += [(0, 0)] * 4
                tri += [(i, i + 1, i + 2), (i, i + 2, i + 3)]
            packed = _pack(frame, f"1WALL_{prefix}_", self.material, np.array(pos, float), np.array(nrm, float),
                           np.array(uv, float), np.array(tri, dtype=np.int64), start=0)
            for m in packed:
                m.renderable = False
            out += packed
        return out


def _pack(frame: LocalFrame, prefix: str, material: str, pos, nrm, uv, tri, start: int) -> list[Mesh]:
    """Region-coordinate geometry in AC axes, wound to its normals, split under the vertex limit at quad boundaries
    (a part never straddles two meshes: every part here is built of whole quads or whole prisms, each under the
    limit)."""
    ac = frame.to_ac(pos[:, 0], pos[:, 1], pos[:, 2])
    ac_n = np.column_stack([nrm[:, 0], nrm[:, 2], Z_SIGN * nrm[:, 1]])
    p0, p1, p2 = ac[tri[:, 0]], ac[tri[:, 1]], ac[tri[:, 2]]
    outward = (np.cross(p1 - p0, p2 - p0) * ac_n[tri].mean(axis=1)).sum(axis=1) >= 0
    tri = np.where(outward[:, None], tri, tri[:, [0, 2, 1]])
    out = []
    lo = 0
    owner = tri.min(axis=1)
    while lo < len(ac):
        hi = min(lo + MAX_VERTICES, len(ac))
        if hi < len(ac):
            # Cut where no triangle spans the cut.
            spans = (tri.min(axis=1) < hi) & (tri.max(axis=1) >= hi)
            while spans.any():
                hi = int(tri[spans].min(axis=1).min())
                spans = (tri.min(axis=1) < hi) & (tri.max(axis=1) >= hi)
        sel = (owner >= lo) & (owner < hi)
        out.append(Mesh(f"{prefix}{start + len(out)}", material, ac[lo:hi].astype(np.float32),
                        ac_n[lo:hi].astype(np.float32), uv[lo:hi].astype(np.float32), tri[sel] - lo))
        lo = hi
    return out


# =====================================================================================================================
# Placing and drawing
# =====================================================================================================================

@dataclass
class Post:
    side: str  # "L" or "R"
    end: str  # "before" or "after" the junction, along the stations
    i: int  # its station
    e: float
    n: float
    h: float  # the verge under it
    top: float = POST_TOP_M
    mast: bool = False
    faces: int = 0
    blades: list = field(default_factory=list)


@dataclass
class Built:
    meshes: list[Mesh]
    atlas: Atlas | None
    stats: dict
    records: list = field(default_factory=list)  # each site and named junction as drawn (`04_mesh/signals.json`)
    # Every solid post drawn, signal, mast or blade post, as (e, n, the ground under it): the regulatory signs
    # (`core.signs`) go on one where theirs would stand beside it.
    posts: list = field(default_factory=list)


class _Road:
    """The centreline's stations and what a post needs from them."""

    def __init__(self, s, xy, z, heading, half_l, half_r, slope_l, slope_r):
        self.s, self.xy, self.z = np.asarray(s, float), np.asarray(xy, float), np.asarray(z, float)
        self.heading = np.asarray(heading, float)
        self.half = {"L": np.asarray(half_l, float), "R": np.asarray(half_r, float)}
        self.slope = {"L": np.asarray(slope_l, float), "R": np.asarray(slope_r, float)}

    def index(self, station: float) -> int:
        return int(np.clip(np.searchsorted(self.s, station), 0, len(self.s) - 1))

    def tangent(self, i: int) -> np.ndarray:
        return np.array([math.cos(self.heading[i]), math.sin(self.heading[i])])

    def left(self, i: int) -> np.ndarray:
        return np.array([-math.sin(self.heading[i]), math.cos(self.heading[i])])

    def place(self, i: int, side: str, beyond: float) -> tuple[float, float, float]:
        sign = 1.0 if side == "L" else -1.0
        off = self.half[side][i] + beyond
        e, n = self.xy[i] + self.left(i) * sign * off
        return float(e), float(n), float(self.z[i] - self.slope[side][i] * self.half[side][i] - beyond * VERGE_FALL)


def corner_stations(station: float, mouths_here: dict, half_default: float) -> dict:
    """The stations of a site's four corners, before any move off a mouth: {(side, end): station}. `mouths_here`: the
    mouths (s0, s1) of its side roads per side, where it has any. A side with none takes the other side's (a T's far
    side faces the side road's mouth), and with none on either, `half_default` either side."""
    out = {}
    for side in ("L", "R"):
        span = mouths_here.get(side) or mouths_here.get("R" if side == "L" else "L")
        if span is None:
            span = (station - half_default, station + half_default)
        # A fork or a road running beside the route has a mouth far along it: the corner stays near the junction, and
        # a post that falls in that mouth moves off it or isn't put up (`clear_station`).
        span = (max(span[0], station - CORNER_REACH_M), min(span[1], station + CORNER_REACH_M))
        out[(side, "before")] = span[0] - POST_GAP_M
        out[(side, "after")] = span[1] + POST_GAP_M
    return out


def junction_spans(mouths: Sequence) -> dict[int, dict]:
    """Each junction node's mouths, per side, as one stretch (s0, s1): its side roads' own, and any other side road's
    on the same side that overlaps them (a divided side road's other carriageway, leaving at a node of its own)."""
    out: dict[int, dict] = {}
    for m in mouths:
        span = out.setdefault(int(m.junction["node"]), {})
        lo, hi = span.get(m.side, (m.s0, m.s1))
        span[m.side] = (min(lo, m.s0), max(hi, m.s1))
    stations = {int(m.junction["node"]): m.station_m for m in mouths}
    for node, spans in out.items():
        for side, (lo, hi) in list(spans.items()):
            for m in mouths:
                if m.side == side and abs(m.station_m - stations[node]) <= CORNER_REACH_M and m.s0 <= hi and m.s1 >= lo:
                    lo, hi = min(lo, m.s0), max(hi, m.s1)
            spans[side] = (lo, hi)
    return out


def clear_station(road: _Road, station: float, side: str, end: str, blocked: dict | None) -> int | None:
    """The station index nearest `station` moving away from the junction (`end`), clear of every mouth on `side`, by
    at most MAX_SHIFT_M; None where there's none, or it's off the road's ends."""
    s = road.s
    if station < s[0] or station > s[-1]:
        return None
    # The station on the corner's own side of `station`: never one step into the mouth it stands beside.
    i = int(np.searchsorted(s, station, side="right")) - 1 if end == "before" else road.index(station)
    i = int(np.clip(i, 0, len(s) - 1))
    if blocked is None:
        return i
    mask = blocked[side]
    step = -1 if end == "before" else 1
    while 0 <= i < len(s) and abs(s[i] - station) <= MAX_SHIFT_M:
        if not mask[i]:
            return i
        i += step
    return None


# The faces each approach gets: (which post, in its role) for an approach from the route before the site, after it,
# or a side road on the left or right (module docstring).
FACE_POSTS = {
    ("route", "before"): (("L", "before"), ("R", "after"), ("L", "after")),
    ("route", "after"): (("R", "after"), ("L", "before"), ("R", "before")),
    ("side", "L"): (("L", "after"), ("R", "before"), ("R", "after")),
    ("side", "R"): (("R", "before"), ("L", "after"), ("L", "before")),
}
ROLES = ("primary", "secondary", "tertiary")


def approaches(site: dict) -> list[tuple[tuple[str, str], dict]]:
    """The site's approaches: its inbound legs, one per route side and per side road side (the widest)."""
    out: dict = {}
    for leg in site["legs"]:
        if not leg.get("inbound"):
            continue
        key = ("route", leg["route"]) if leg.get("route") else ("side", leg["side"])
        if key not in out or (leg.get("lanes") or 0) > (out[key].get("lanes") or 0):
            out[key] = leg
    return sorted(out.items())


def face_plan(site: dict, tangent: np.ndarray) -> list[tuple[tuple[str, str], np.ndarray, str, int]]:
    """Every vehicle face a site's approaches get (module docstring): (its post, as (side, "before" or "after"), the
    way it faces (2D, back along its approach), its role, the approach's lanes). `tangent`: the route's direction at
    the site. A route approach faces back along the route; a side road's faces out along that road."""
    tangent = _unit(tangent)
    out = []
    for (kind, which), leg in approaches(site):
        if kind == "route":
            facing = -tangent if which == "before" else tangent
        else:
            facing = np.array([math.cos(leg["heading"]), math.sin(leg["heading"])])
        for role, key in zip(ROLES, FACE_POSTS[(kind, which)]):
            out.append((key, facing, role, int(leg.get("lanes") or 0)))
    return out


def arm_length(lanes: int) -> float:
    """The shortest standard outreach reaching over the second lane from the kerb (estimate: lanes LANE_W_M wide)."""
    need = POST_OFFSET_M + 1.5 * LANE_W_M
    for arm in MAST_ARMS_M:
        if arm >= need:
            return arm
    return MAST_ARMS_M[-1]


def build(sites: Sequence[dict], blade_sites: Sequence[BladeSite], road: _Road, frame: LocalFrame, *,
          mouths: Sequence = (), blocked: dict | None = None, station_of=None) -> Built:
    """The signals and blades as meshes in AC axes, their atlas and a summary. `mouths`: the side roads' mouths
    (`core.junctions.Mouth`), whose ends set the corners; `blocked`: per side, the stations no post stands at (the
    furniture's mouths: side roads', ramps', roundabouts' and driveways', and the route's bridges). `station_of(site)`
    finds a site's station on the centreline."""
    texts = sorted({abbreviate(name) for b in blade_sites for name, _ in b.names})
    tex = atlas(texts)
    b = _Builder(tex)
    stats = {"sites": 0, "junction_sites": 0, "mid_block_sites": 0, "posts": 0, "mast_arms": 0, "vehicle_faces": 0,
             "pedestrian_lanterns": 0, "buttons": 0, "posts_left_out": 0, "faces_left_out": 0,
             "blade_junctions": 0, "blade_posts": 0, "blades": 0, "blades_on_signal_posts": 0}
    by_node = junction_spans(mouths)
    signal_posts: dict[int, dict] = {}  # node -> {(side, end): Post}
    records = []
    for site in sites:
        at = station_of(site) if station_of is not None else float(site["at_m"])
        half = CROSSWALK_HALF_M if site["kind"] == "mid_block" else JUNCTION_HALF_M
        stations = corner_stations(at, by_node.get(int(site["node"]), {}), half)
        posts: dict = {}
        for (side, end), st in stations.items():
            i = clear_station(road, st, side, end, blocked)
            if i is None:
                stats["posts_left_out"] += 1
                continue
            e, n, h = road.place(i, side, POST_OFFSET_M)
            posts[(side, end)] = Post(side, end, i, e, n, h)
        faces = []  # (post key, facing (2D), role, lanes)
        for key, facing, role, lanes in face_plan(site, road.tangent(road.index(at))):
            if key not in posts:
                stats["faces_left_out"] += 1
                continue
            faces.append((key, facing, role, lanes))
        if not posts:
            continue
        stats["sites"] += 1
        stats[f"{site['kind']}_sites"] += 1
        for key, facing, role, lanes in faces:
            p = posts[key]
            if role == "primary" and lanes >= MAST_LANES and not p.mast:
                p.mast = True
                _mast(b, p, facing, arm_length(lanes))
                stats["mast_arms"] += 1
                stats["vehicle_faces"] += 1
            _vehicle_face(b, p, facing)
            p.faces += 1
            stats["vehicle_faces"] += 1
        # Pedestrian lanterns and buttons: a crosswalk across each leg that has one, a lantern and a button at each end.
        walks = set()
        if site["kind"] == "mid_block":  # one crosswalk, between the two approaches' primary posts
            walks.add((("L", "before"), ("R", "after")))
        for leg in pedestrian_legs(site) if site["kind"] == "junction" else ():
            if leg.get("route"):
                walks.add((("L", leg["route"]), ("R", leg["route"])))
            else:
                walks.add(((leg["side"], "before"), (leg["side"], "after")))
        for a_key, b_key in sorted(walks):
            if a_key not in posts or b_key not in posts:
                continue
            for near, far in ((a_key, b_key), (b_key, a_key)):
                p, q = posts[near], posts[far]
                toward = np.array([q.e - p.e, q.n - p.n])
                if float(np.hypot(*toward)) < 0.5:
                    continue
                _pedestrian(b, p, toward, road)
                stats["pedestrian_lanterns"] += 1
                stats["buttons"] += 1
        for p in posts.values():
            _post(b, p)
            stats["posts"] += 1
        signal_posts[int(site["node"])] = posts
        records.append({"site": site["id"], "node": site["node"], "kind": site["kind"], "station_m": round(at, 2),
                        "posts": {f"{side}_{end}": (round(float(road.s[posts[(side, end)].i]), 2)
                                                    if (side, end) in posts else None)
                                  for side, end in sorted(stations)},
                        "mast_arms": sorted(f"{p.side}_{p.end}" for p in posts.values() if p.mast),
                        "vehicle_faces": len(faces) + sum(p.mast for p in posts.values()),
                        "crosswalks": len(walks)})
    for bs in blade_sites:
        stats["blade_junctions"] += 1
        record = {"node": bs.node, "station_m": round(bs.station_m, 2), "names": [n for n, _ in bs.names],
                  "blades": [abbreviate(n) for n, _ in bs.names[:len(BLADE_TOPS_M)]], "posts": []}
        records.append(record)
        on_signals = signal_posts.get(bs.node, {})
        stations = corner_stations(bs.station_m, by_node.get(bs.node, {}), JUNCTION_HALF_M)
        for side, end in bs.corners:
            p = on_signals.get((side, end))
            if p is None:
                i = clear_station(road, stations[(side, end)], side, end, blocked)
                if i is None:
                    record["posts"].append({"corner": f"{side}_{end}", "left_out": "no clear station"})
                    continue
                e, n, h = road.place(i, side, BLADE_OFFSET_M)
                p = Post(side, end, i, e, n, h)
                b.collider(e, n, h - COLLIDER_BELOW_M, h + COLLIDER_TOP_M, COLLIDER_W)
                stats["blade_posts"] += 1
                tops = BLADE_TOPS_M
            else:
                stats["blades_on_signal_posts"] += 1
                tops = SIGNAL_BLADE_TOPS_M
            tops = tops[:len(bs.names)]
            if not p.mast:
                # The post (a blade post's own, or a 2T post carried up) reaches the lowest blade, and a spigot joins
                # each blade to the one above, so no post runs across a name. A mast's column already reaches past
                # them, and its blades stand beside it.
                foot = p.h + p.top if (side, end) in on_signals else p.h - 0.4
                b.prism(p.e, p.n, foot, p.h + tops[-1] - BLADE_H, BLADE_POST_D)
                for upper, lower in zip(tops, tops[1:]):
                    b.prism(p.e, p.n, p.h + lower, p.h + upper - BLADE_H, BLADE_POST_D * 0.6, cap=False)
            record["posts"].append({"corner": f"{side}_{end}", "station_m": round(float(road.s[p.i]), 2),
                                    "on_signal_post": tops is SIGNAL_BLADE_TOPS_M})
            for (name, h_road), top in zip(bs.names, tops):
                _blade(b, p, abbreviate(name), h_road, top, tex)
                stats["blades"] += 1
    return Built(b.meshes(frame), tex if (b.parts[MATERIAL][0] or b.parts[RETRO_MATERIAL][0]) else None, stats,
                 records, [(e, n, h0 + COLLIDER_BELOW_M) for e, n, h0, _, _ in b.colliders])


def _post(b: _Builder, p: Post) -> None:
    if p.mast:
        b.collider(p.e, p.n, p.h - COLLIDER_BELOW_M, p.h + COLLIDER_TOP_M, MAST_D + 0.04)
        return  # the column is drawn with its arm
    b.prism(p.e, p.n, p.h - 0.3, p.h + p.top, POST_D)
    b.collider(p.e, p.n, p.h - COLLIDER_BELOW_M, p.h + COLLIDER_TOP_M, COLLIDER_W)


def _lantern(b: _Builder, centre: np.ndarray, facing: np.ndarray) -> None:
    """A vehicle lantern on its target board: the board, the housing, three visors and three lenses (red, amber,
    green from the top)."""
    f = np.array([*_unit(facing), 0.0])
    r = _right(facing)
    w, h, dep = LANTERN
    bw, bh, bt = BOARD
    b.box(centre - f * (dep / 2 + bt / 2 + BOARD_GAP_M), facing, bw, bh, bt,
          ("board", "board_back", "black", "black", "black"),
          (RETRO_MATERIAL, MATERIAL, MATERIAL, MATERIAL, MATERIAL, MATERIAL))
    b.box(centre, facing, w, h, dep, ("black", "black", "black", "black", "black"))
    front = centre + f * (dep / 2 + 0.003)
    for k, (cell, material) in enumerate((("lens_red", MATERIAL), ("lens_amber", AMBER_MATERIAL),
                                         ("lens_green", MATERIAL))):
        c = front + _UP * (ASPECT_M * (1 - k)) * 1.25
        b.disc(c, r, _UP, ASPECT_M / 2, f, cell, material)
        # The visor: its underside and its two sides, reaching VISOR_M forward over the top of the lens.
        top = c + _UP * (ASPECT_M / 2 + 0.01)
        b.quad(top + f * VISOR_M / 2, r, f, ASPECT_M + 0.02, VISOR_M, -_UP, "black")
        for sgn in (1.0, -1.0):
            b.quad(c + r * sgn * (ASPECT_M / 2 + 0.01) + f * VISOR_M / 2 + _UP * 0.02, f * -sgn, _UP,
                   VISOR_M, ASPECT_M * 0.8, r * sgn, "black")


def _vehicle_face(b: _Builder, p: Post, facing: np.ndarray) -> None:
    """A face on a post: the lantern's back BRACKET_M from the post's axis on the side it faces, its top at
    LANTERN_TOP_M."""
    f = np.array([*_unit(facing), 0.0])
    centre = np.array([p.e, p.n, p.h + LANTERN_TOP_M - LANTERN[1] / 2]) + f * (BRACKET_M + LANTERN[2] / 2)
    _lantern(b, centre, facing)
    b.beam(np.array([p.e, p.n, p.h + LANTERN_TOP_M - LANTERN[1] / 2]),
           centre - f * (LANTERN[2] / 2 + BOARD[2] + BOARD_GAP_M), 0.06)


def _mast(b: _Builder, p: Post, facing: np.ndarray, arm: float) -> None:
    """The primary post as a mast arm: the column, its outreach over the lanes (to the right of the traffic it faces)
    and the overhead face at the outreach's end."""
    f = _unit(facing)
    # Traffic drives along -f. Its right is -f turned clockwise: (-f_y, f_x).
    right_of_traffic = np.array([-f[1], f[0]])
    b.prism(p.e, p.n, p.h - 0.3, p.h + MAST_HEIGHT_M, MAST_D)
    rise = 0.15  # the outreach rises a little towards its end (SD-101-010)
    base = np.array([p.e, p.n, p.h + MAST_HEIGHT_M - 0.25])
    tip = base + np.array([*(right_of_traffic * arm), rise])
    b.beam(base, tip, ARM_D)
    centre = tip - np.array([*(right_of_traffic * 0.4), 0.0])
    centre[2] = p.h + MAST_CLEARANCE_M + LANTERN[1] / 2 + 0.1
    _lantern(b, centre + np.array([*(_unit(facing) * 0.05), 0.0]), facing)
    b.beam(np.array([centre[0], centre[1], tip[2] - 0.02]), np.array([centre[0], centre[1], centre[2] + LANTERN[1] / 2]),
           0.05)


def _pedestrian(b: _Builder, p: Post, toward: np.ndarray, road: _Road) -> None:
    """A pedestrian lantern on the post, aimed along the crosswalk at its far end, and the push button below it on
    the side away from the road."""
    f = _unit(toward)
    f3 = np.array([*f, 0.0])
    w, h, dep = PED
    centre = np.array([p.e, p.n, p.h + PED_TOP_M - h / 2]) + f3 * (BRACKET_M * 0.7 + dep / 2)
    b.box(centre, f, w, h, dep, ("black", "black", "black", "black", "black"))
    r = _right(f)
    front = centre + f3 * (dep / 2 + 0.003)
    b.quad(front + _UP * h / 4, r, _UP, w * 0.8, h / 2 * 0.85, f3, "ped_red")
    b.quad(front - _UP * h / 4, r, _UP, w * 0.8, h / 2 * 0.85, f3, "ped_green")
    b.beam(np.array([p.e, p.n, centre[2]]), centre - f3 * dep / 2, 0.05)
    # The button: on the side of the post away from the road, facing the footpath where the pedestrian waits, a
    # little towards the crosswalk it serves (a post at two crosswalks has two side by side).
    away = road.left(p.i) * (1.0 if p.side == "L" else -1.0)
    c = np.array([p.e, p.n, p.h + BUTTON_M]) + np.array([*away, 0.0]) * (POST_D / 2 + BUTTON[2] / 2)
    c = c + f3 * BUTTON[0] * 0.6
    b.box(c, away, BUTTON[0], BUTTON[1], BUTTON[2], ("button", "black", "black", "black", "black"))


def _blade(b: _Builder, p: Post, text: str, heading: float, top: float, tex: Atlas) -> None:
    """A blade on top of a post, along the road it names, lettered on both faces."""
    along = np.array([math.cos(heading), math.sin(heading), 0.0])
    normal = np.array([-along[1], along[0], 0.0])
    length = tex.blade_m[text]
    centre = np.array([p.e, p.n, p.h + top - BLADE_H / 2])
    if p.mast:  # a mast's column runs on up past them: each blade is carried to one side of it, on its own axis
        centre = centre + along * (length / 2 + MAST_D / 2 + 0.03)
    cell = f"blade:{text}"
    # Each face reads left to right from its own side.
    b.quad(centre + normal * BLADE_THICK / 2, -along, _UP, length, BLADE_H, normal, cell, RETRO_MATERIAL)
    b.quad(centre - normal * BLADE_THICK / 2, along, _UP, length, BLADE_H, -normal, cell, RETRO_MATERIAL)
    for sgn in (1.0, -1.0):
        b.quad(centre + along * sgn * length / 2, normal, _UP, BLADE_THICK, BLADE_H, along * sgn, "board_back")
    b.quad(centre + _UP * BLADE_H / 2, along, normal, length, BLADE_THICK, _UP, "board_back")
