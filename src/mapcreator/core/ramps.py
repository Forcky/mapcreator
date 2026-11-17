"""Slip ramps: ramps (`highway=*_link`) that leave or join the route at a slip angle, under 15 degrees, drawn as a
widened taper, a painted gore, a grassed nose and a 60 m ramp stub past it, closed as a major side road is.

The maintainer's answer to the flyovers research's Q10 (17 Nov 2026) was "yes" to all four: the road widened over the
taper; a 60 m ramp stub past the gore nose; the major closure kit across the ramp (tape, barrier boards, and a police
car under the cap, `core.stub_closures`); and guide posts and rails cleared along the taper. A ramp meeting the route
at 15 degrees or more is a side road like any other (`core.junctions`, `core.stubs`), as before.

Why not a side road's stub: a stub's flare is a circular corner from the road's edge to the side road's, and at a
slip angle its tangent runs past FLARE_MAX_TANGENT_M (`skew`), or the junction finder drops it first (`angle`; under
about 7 degrees `along_route` takes it for the route itself). So the ramp's edges are followed as they peel off the
road's edge instead, with no corner at all.

**Which ramps** (`find`, the route stage; `01_route/ramps.json`): at each node of the route, a `*_link` way leaving
it under MAX_ANGLE_DEG off the mainline (measured as the junction finder measures it), including the ones the finder
can't see. Not one that's private, that has a bridge, tunnel or layer in its first `junctions.GRADE_FREE_M`, that
leads onto a roundabout the route drives, that meets the route again within `junctions.MIN_LEG_M`, or that's within
END_CLEAR_M of the route's first or last point (the end wall closes it). Each is followed out FOLLOW_M. Its `kind`
says how traffic uses it: `diverge` (an off-ramp, one-way away from the route), `merge` (an on-ramp, one-way onto it)
or `two_way`. `select` picks the ways: links only here, but the same code would take a shallow fork of any class.

**The plan** (`plan`, the section stage; `03_section/ramps.json`). Along the road, on the ramp's side:
- **the taper**, from where the ramp's outer edge leaves the road's edge to where its inner edge does (the split):
  the paved area between the road's edge and the ramp's outer edge, so the road widens by the ramp's lane;
- **the gore**, from the split to the nose, where the ramp's inner edge is NOSE_W_M beyond the road's edge: paved,
  and painted where the road is marked (below);
- **the nose island**, from the nose to where the gap is ISLAND_END_W_M: mown grass between the two roads' verges,
  falling from the roads' level at the nose to the ground at its far end;
- **the ramp stub**, the ramp's own ribbon from the nose for STUB_M.
The taper and the gore are one throat, triangulated over the road's own edge vertices, the ramp stub's first row and
the outer edge, sharing them exactly, as a side road's throat is. The road's half widths don't change, so the AI line
and the racing line don't either: the AI keeps to its line and never takes the ramp.

**The paint** (Australian practice: DIT South Australia, *Pavement Marking Manual* v9, 2025, 3.3.3 and 2.1.6.3; Qld
TMR, *MUTCD* Part 2, 2026, which adopts AS 1742.2 clause 5.6; the chevrons' direction read from the drawings in Qld
TMR's *Supplement to Austroads GRD Part 4C*, 2025, figure 11.2): white chevrons in the gore where traffic passes both
sides the same way (a diverge or a merge), diagonal stripes where the gore splits opposing traffic (a two-way ramp);
bars 1 m wide at 45 degrees to the gore's centre line, the first where the gore is 1 m wide, then every 6 m on a road
of 70 km/h or less, 12 m faster, and 24 m off a motorway or a trunk road of 100 km/h or more; the chevrons' points
face approaching traffic. A 100 mm line along the ramp's inner edge outlines the gore, and the
road's edge line carries on along the taper's outer edge, at the ramp texture's own line offset so both meet the ramp
stub's lines. The road's own edge line breaks across the taper and stays solid beside the gore. The paint is cut into
the throat's surface as polygons with texture coordinates on the white of the texture's edge line: one material, no
decal. None on gravel or an unmarked road.

A ramp whose taper, gore or nose can't be drawn, that would overlap a side road's stub, a roundabout or another ramp,
or whose mouth would leave a timing gate nowhere to stand is left out, listed in `skipped` with the reason, and the
road is built there as before. The mesh stage checks each one on the meshes it builds, as it checks stubs, and leaves
out any that fail (`settle`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

from mapcreator.core import centreline as cl
from mapcreator.core import junctions as jn
from mapcreator.core import mesh
from mapcreator.core import roundabouts as rb_core
from mapcreator.core import section as sec
from mapcreator.core import stubs
from mapcreator.core import textures
from mapcreator.core.layout import RoadModel, gate_distances
from mapcreator.core.scene import LocalFrame, Mesh

FORMAT_VERSION = 1

MAX_ANGLE_DEG = jn.MIN_ANGLE_DEG  # 15: a ramp leaving at this or more is a side road (`core.junctions`)
FOLLOW_M = 300.0  # how far each ramp is followed out from the route: far enough for a slow taper and its stub
END_CLEAR_M = 20.0  # a ramp this near the route's first or last point is closed by the end wall
STUB_M = 60.0  # the ramp's own ribbon past the nose (Q10)
START_EASE_M = stubs.START_EASE_M
NOSE_W_M = 2 * mesh.VERGE_M + 0.5  # the gap between the road's edge and the ramp's at the nose: both verges and 0.5 m
ISLAND_END_W_M = 2 * (mesh.VERGE_M + 0.5) + 2.0  # and at the island's far end: room for both skirts, and 2 m between
ISLAND_MIN_END_W_M = 2 * (mesh.VERGE_M + 0.5) + 0.5  # or, with the island to the stub's end, at least this there
NOSE_EASE_M = 5.0  # the verges beside the island fall from the roads' level at the nose to their own over this
TAPER_RATIO = 15.0  # an outer edge already off the road where the ramp starts is tapered in at 1:15 (Qld RPDM 16-22)
EDGE_EPS_M = 0.02  # outer-edge points nearer the road's edge than this are left out of the throat's outline
EDGE_LINE_FROM_M = 0.3  # the road's edge line breaks once the taper is this wide
MAIN_REACH_M = stubs.MAIN_REACH_M
BLEND_M = stubs.BLEND_M
MAX_BLEND_GRADE = stubs.MAX_BLEND_GRADE
DEM_SMOOTH_M = stubs.DEM_SMOOTH_M
MIN_RIBBON_M = stubs.MIN_RIBBON_M  # a ramp with less than this of its own ribbon past the island isn't drawn
LEAD_M = 20.0  # ramps.json keeps the ramp's rows from this far before the nose
CLEAR_M = stubs.CLEAR_M
SKIRT_CLEAR_M = 1.0  # the ramp's skirt stops this short of the road's, so the ground between them is no sliver
RAMP_ID_BASE = 200000  # ramps' ids, apart from junctions' (0 up) and roundabout exits' (100000 up)
# The paint (module notes).
BAR_W_M = 1.0
OUTLINE_GAP_M = 0.1
PAINT_CLEAR_M = 0.05  # paint keeps this far inside the throat's outline, so the outline's vertices stay the road's
SPACING_SLOW_M, SPACING_M, SPACING_FAST_M = 6.0, 12.0, 24.0
SLOW_KMH, FAST_KMH = 70.0, 100.0
CLASS_KMH = {"motorway": 110.0, "trunk": 100.0}  # an untagged road's speed, for the spacing; others count as 80
FAST_CLASSES = frozenset({"motorway", "trunk"})
SEAM_MIN_W_M = 0.3  # the checks sample the seam with the road's edge only where the throat is this wide
KINDS = ("diverge", "merge", "two_way")


def is_ramp(tags: dict) -> bool:
    """A link (`motorway_link` to `tertiary_link`): what `find` takes by default."""
    hw = str(tags.get("highway") or "")
    return hw.endswith("_link") and hw in jn.DRIVABLE


# -- which ramps (the route stage) -------------------------------------------------------------------------------------

def find(net, route_xy: np.ndarray, *, select=is_ramp, to_lonlat=None) -> dict:
    """The slip ramps along a routed polyline: the content of ramps.json. `net` and `route_xy` are as
    `junctions.find` takes them; `select(tags)` picks the ways that count (links). A way it picks that leaves at
    MAX_ANGLE_DEG or more is a side road, the junction finder's, and isn't listed here."""
    nw = jn._Network(net)
    rt = jn._Route(route_xy, nw)
    rings = jn._rings(nw, rt)
    ring_node = {n for ring in rings for n in ring["nodes"]}
    elsewhere = nw.ring_nodes - ring_node
    total = float(rt.s[-1])
    found, skipped = [], []
    for i in range(1, len(rt.xy) - 1):
        node = int(rt.node[i])
        if node < 0:
            continue
        at = float(rt.s[i])
        here = rt.xy[i]
        u_in = here - rt.at(at - jn.ANGLE_AT_M)
        u_out = rt.at(at + jn.ANGLE_AT_M) - here
        u_in /= max(float(np.hypot(*u_in)), 1e-9)
        u_out /= max(float(np.hypot(*u_out)), 1e-9)
        tangent = u_in + u_out if float(np.hypot(*(u_in + u_out))) > 1e-6 else u_out
        for nb, way in nw.adj[node]:
            if frozenset((node, nb)) in rt.pairs or nw.is_ring(way):
                continue
            tags = nw.tags(way)
            if not select(tags):
                continue
            hidden = rt.along_route(here, nw.xy[nb], at)
            if hidden and _spliced(rt, i, nw.xy[nb]):
                continue  # the segment a waypoint was spliced into, running on past the route's end: the route
            leg = nw.walk(node, nb, way, rt.nodes - {node}, FOLLOW_M, elsewhere)
            clip = rt.other_part_m(leg.line, here, at)
            if clip is not None and clip < cl.cumulative(leg.line)[-1]:
                leg.line, leg.ends = jn._cut(leg.line, clip), "meets_route"
            length = float(cl.cumulative(leg.line)[-1])
            direction = jn._along(leg.line, jn.ANGLE_AT_M) - here
            angle = jn._wrap(jn._heading(direction) - jn._heading(tangent))
            ahead = abs(jn._wrap(jn._heading(direction) - jn._heading(u_out)))
            behind = abs(jn._wrap(jn._heading(direction) - jn._heading(-u_in)))
            off = min(ahead, behind)
            if off >= MAX_ANGLE_DEG:
                continue  # a side road (`core.junctions`)
            if jn.is_private(tags):
                reason = "private"
            elif jn._grade_separated_near(nw, leg):
                reason = "grade_separated"
            elif any(n in ring_node for n in leg.nodes[1:]):
                reason = "roundabout"
            elif leg.ends == "meets_route" and length < jn.MIN_LEG_M:
                reason = "meets_route"
            elif at < END_CLEAR_M or at > total - END_CLEAR_M:
                reason = "route_end"
            else:
                reason = None
            record = {"node": int(nw.osm_node[node]), "at_m": round(at, 1), "xy": np.round(here, 2).tolist(),
                      "side": "left" if angle > 0 else "right", "angle_deg": round(angle, 1),
                      "off_mainline_deg": round(off, 1)}
            if reason is not None:
                skipped.append({**record, "way_id": nw.ways[way]["id"], "highway": tags.get("highway"),
                                "name": tags.get("name"), "reason": reason})
                continue
            one_way = nw.one_way.get(frozenset((node, nb)))
            kind = "two_way" if one_way is None else "diverge" if one_way[0] == node else "merge"
            along = "ahead" if ahead <= behind else "behind"
            turn = abs(jn._wrap(jn._heading(u_out) - jn._heading(u_in)))
            found.append({**record, "route_turn_deg": round(turn, 1), "kind": kind, "along": along,
                          "hidden": bool(hidden), "main": _main_road(nw, rt, i, along),
                          **jn._leg_record(nw, leg, way)})
    found.sort(key=lambda r: (r["at_m"], r["side"], r["angle_deg"]))
    ramps = [{"id": RAMP_ID_BASE + k, **r} for k, r in enumerate(found)]
    if to_lonlat is not None:
        for r in ramps:
            lon, lat = to_lonlat.transform(*r["xy"])
            r["lonlat"] = [round(float(lon), 7), round(float(lat), 7)]
    return {"version": FORMAT_VERSION,
            "rule": {"max_angle_deg": MAX_ANGLE_DEG, "angle_at_m": jn.ANGLE_AT_M, "follow_m": FOLLOW_M,
                     "end_clear_m": END_CLEAR_M, "grade_free_m": jn.GRADE_FREE_M, "min_leg_m": jn.MIN_LEG_M},
            "ramps": ramps,
            "skipped": skipped}


def _spliced(rt: jn._Route, i: int, to: np.ndarray) -> bool:
    """Whether the segment from route point `i` to `to` is the one a waypoint next to it was spliced into: it passes
    through that waypoint."""
    for j in (i - 1, i + 1):
        if 0 <= j < len(rt.node) and rt.node[j] < 0 \
                and jn._segment_distance(rt.xy[j], np.vstack([rt.xy[i], to])) < jn.ON_ROUTE_M:
            return True
    return False


def _main_road(nw: jn._Network, rt: jn._Route, i: int, along: str) -> dict:
    """The route's own way beside the ramp at route point `i`: ahead of the node for a ramp running ahead, else
    behind. Its class and speed size the gore's paint."""
    j = i + 1 if along == "ahead" else i - 1
    other = int(rt.node[j]) if 0 <= j < len(rt.node) else -1
    node = int(rt.node[i])
    for nb, way in nw.adj[node]:
        if nb == other:
            tags = nw.tags(way)
            return {"way_id": nw.ways[way]["id"], "highway": tags.get("highway"), "maxspeed": tags.get("maxspeed"),
                    "oneway": tags.get("oneway"), "lane_markings": tags.get("lane_markings")}
    return {}


# -- the plan (the section stage) --------------------------------------------------------------------------------------

def _speed(main: dict) -> float:
    try:
        return float(str(main.get("maxspeed") or "").split()[0])
    except (ValueError, IndexError):
        return CLASS_KMH.get(str(main.get("highway") or ""), 80.0)


def spacing_m(main: dict) -> float:
    """The gore's bars' spacing (module notes) for the road beside it (`ramps.json`'s `main`)."""
    kmh = _speed(main)
    if kmh <= SLOW_KMH:
        return SPACING_SLOW_M
    if kmh >= FAST_KMH and str(main.get("highway") or "") in FAST_CLASSES:
        return SPACING_FAST_M
    return SPACING_M


def _last_rise(g: np.ndarray, upto: int) -> float | None:
    """The last d before index `upto` where g rises through 0, interpolated; None if g is already up at the start."""
    for k in range(min(upto, len(g) - 1), 0, -1):
        if g[k] >= 0 > g[k - 1]:
            return float(k - 1 + (-g[k - 1]) / (g[k] - g[k - 1]))
    return None


def _first(mask: np.ndarray, start: int = 0) -> int | None:
    hit = np.flatnonzero(np.asarray(mask)[start:])
    return int(hit[0]) + start if len(hit) else None


@dataclass
class _Candidate:
    record: dict
    priority: tuple
    throat: shapely.Geometry
    island: shapely.Geometry
    rows_xy: np.ndarray
    rows_n: np.ndarray
    half: float
    first: int  # the nose's row: the ramp's own ribbon starts here
    island_row: int  # and its inner verge comes back here


def _candidate(r: dict, m: stubs._Mainline, dem_sample, width_factor: float) -> tuple[_Candidate | None, str | None]:
    """A ramp's plan, or why there's none."""
    side = jn.SIDE_CODE[r["side"]]
    sign = 1.0 if side == "L" else -1.0
    at = jn._station(m.xy, m.s, np.asarray(r["xy"], float), float(r["at_m"]))
    line = np.asarray(r["line"], float)
    if len(line) < 2 or cl.cumulative(line)[-1] < 3 * MIN_RIBBON_M:
        return None, "short"
    ease = np.clip(1.0 - cl.cumulative(line) / START_EASE_M, 0.0, 1.0)[:, None]
    line = line + (m.at(at) - line[0]) * ease
    try:
        sm = cl.smooth(line)
    except ValueError:
        return None, "short"
    xy, hd, curv = sm.xy, sm.heading, sm.curvature
    nrm = cl.normals(hd)
    near = m.near(at, sm.s[-1] + MAIN_REACH_M)

    # Width, surface and look, by the road's rules, as a side road's stub's.
    tags = {**(r.get("tags") or {}), "highway": r.get("highway")}
    row_at = int(round(m.row(at)))
    surface = sec.surface_class(tags)
    if surface is None:
        surface = sec.UNSEALED if sec.is_unsealed(m.style[row_at]) else sec.SEALED
    unsealed = surface == sec.UNSEALED
    mapped, source = sec.width_from_tags(tags, unsealed=unsealed)
    width = mapped * width_factor
    w = width / 2
    style = sec.style_code(unsealed, sec.single_lane(tags, mapped, unsealed))
    crown = sec.CROWN_UNSEALED if unsealed else sec.CROWN

    # How far each of its edges is beyond the road's edge, along it.
    g_left = near.beyond(xy + w * nrm, side)
    g_right = near.beyond(xy - w * nrm, side)
    g_mid = near.beyond(xy, side)
    off_road = np.flatnonzero(g_mid > w + 0.5)
    if not len(off_road):
        return None, "parallel"
    es_out = 1.0 if float(np.mean((g_left - g_right)[off_road[:20]])) > 0 else -1.0
    es_in = -es_out
    g_out, g_in = (g_left, g_right) if es_out > 0 else (g_right, g_left)
    k = _first(g_in >= NOSE_W_M)
    if k is None:
        return None, "parallel"
    last = int(min(k + STUB_M, len(sm.s) - 1))
    if last - k < MIN_RIBBON_M:
        return None, "short"
    kc = _first(g_in >= ISLAND_END_W_M, k)
    if kc is None or kc > last:
        # A ramp running on beside the road: the island runs the stub's whole length, if both skirts fit at its end.
        if g_in[last] < ISLAND_MIN_END_W_M:
            return None, "parallel"
        kc = last
    d_split = _last_rise(g_in, k)
    if d_split is None:
        return None, "stays_on_road"
    d_taper = _last_rise(g_out, int(math.floor(d_split)))
    sp_mid, _ = near.project(xy)
    dirn = 1 if sp_mid[k] > sp_mid[0] else -1
    outer = xy + es_out * w * nrm
    sp_out, _ = near.project(outer)

    def to_row(s: float, back: bool) -> int:
        x = m.row(s)
        return int(math.floor(x)) if (back == (dirn > 0)) else int(math.ceil(x))

    # The taper: the outer edge from where it leaves the road's edge to the nose's row, every point off the road.
    prefix = []
    if d_taper is not None:
        k0 = int(math.floor(d_taper)) + 1
        lo_pt = np.array([np.interp(d_taper, sm.s, outer[:, 0]), np.interp(d_taper, sm.s, outer[:, 1])])
        s_taper = float(near.project(lo_pt)[0][0])
    else:  # its outer edge is off the road already where it starts: tapered in at 1:15 from the road's edge
        k0 = 0
        g0 = float(g_out[0])
        s_taper = float(sp_out[0] - dirn * TAPER_RATIO * g0)
        rows_between = np.arange(to_row(s_taper, True) + dirn, to_row(float(sp_out[0]), False), dirn)
        for rr in rows_between:
            gg = g0 * abs(m.s[rr] - s_taper) / max(TAPER_RATIO * g0, 1e-9)
            if EDGE_EPS_M < gg < g0:
                prefix.append(m.to_xy(np.array([m.s[rr]]), np.array([sign * (m.half[side][rr] + gg)]))[0])
    taper_row = to_row(s_taper, True)
    chain_rows = [kk for kk in range(k0, k) if g_out[kk] > EDGE_EPS_M]
    if chain_rows:  # contiguous to the nose: an edge that dips back onto the road starts the taper again after it
        breaks = [kk for kk in range(chain_rows[0], k) if g_out[kk] <= EDGE_EPS_M]
        if breaks:
            return None, "taper"
    chain = np.vstack([m.edge(taper_row, side)[None]] + ([np.array(prefix)] if prefix else [])
                      + ([outer[chain_rows]] if chain_rows else []) + [outer[k][None]])
    split_pt = np.array([np.interp(d_split, sm.s, xy[:, 0]), np.interp(d_split, sm.s, xy[:, 1])]) \
        + es_in * w * cl.normals(np.array([np.interp(d_split, sm.s, hd)]))[0]
    s_split = float(near.project(split_pt)[0][0])
    split_row = int(round(m.row(s_split)))
    inner = xy + es_in * w * nrm
    s_nose = float(near.project(inner[k])[0][0])
    nose_row = int(round(m.row(s_nose)))
    s_end = float(near.project(inner[kc])[0][0])
    end_row = int(round(m.row(s_end)))
    order = [taper_row, split_row, nose_row, end_row]
    if dirn < 0:
        order = order[::-1]
    if not (order[0] < order[1] <= order[2] < order[3]):
        return None, "taper"
    lo, hi = min(taper_row, end_row), max(taper_row, end_row)
    if lo < 2 or hi > len(m.s) - 3:
        return None, "road_end"
    # Where the road's edge line breaks: from where the taper is EDGE_LINE_FROM_M wide to the split.
    g_chain = near.beyond(chain, side)
    wide = _first(g_chain >= EDGE_LINE_FROM_M)
    s_wide = float(near.project(chain[wide if wide is not None else len(chain) - 1])[0][0])
    break_rows = sorted([int(round(m.row(s_wide))), split_row])

    # The throat: the road's edge from the taper's start to the nose, across the nose (the road's verge, the ramp's
    # inner verge), the ramp's row at the nose, and back along its outer edge.
    main_rows = list(range(taper_row, nose_row + dirn, dirn))
    main_edge = np.array([m.edge(rr, side) for rr in main_rows])
    main_verge = m.edge(nose_row, side) + sign * mesh.VERGE_M * m.nrm[nose_row]
    inner_verge = inner[k] + es_in * mesh.VERGE_M * nrm[k]
    across = np.array([xy[k] + es_in * t * w * nrm[k] for t in (1.0, 0.5, 0.0, -0.5, -1.0)])  # inner to outer
    ring = np.vstack([main_edge, main_verge[None], inner_verge[None], across, chain[::-1][1:-1]])
    throat = shapely.Polygon(ring)
    road_lo, road_hi = max(lo - 20, 0), min(hi + 20, len(m.s) - 1)
    road = shapely.Polygon([m.edge(rr, "L") for rr in range(road_lo, road_hi + 1)]
                           + [m.edge(rr, "R") for rr in range(road_hi, road_lo - 1, -1)]).buffer(0)
    if not throat.is_valid or throat.area < 1.0 or throat.intersection(road).area > 0.05:
        return None, "throat"
    island_rows = list(range(nose_row, end_row + dirn, dirn))
    island = shapely.Polygon(np.vstack([np.array([m.edge(rr, side) for rr in island_rows]), inner[k:kc + 1][::-1],
                                        inner_verge[None], main_verge[None]]))
    if not island.is_valid or island.intersection(road).area > 0.05 or island.intersection(throat).area > 0.05:
        return None, "island"

    # Heights: the road's cross-fall carried on, blending into the ramp's own profile and crown (as a stub's).
    n_rows = last + 1
    rows_xy, rows_h, rows_k, rows_n = xy[:n_rows], hd[:n_rows], curv[:n_rows], nrm[:n_rows]
    own = dem_sample(rows_xy[:, 0], rows_xy[:, 1])
    if not np.isfinite(own).any():
        own = np.full(n_rows, m.edge_h(row_at, side))
    else:
        ok = np.isfinite(own)
        own = np.interp(np.arange(n_rows), np.flatnonzero(ok), own[ok])
    own = uniform_filter1d(own, max(int(round(DEM_SMOOTH_M)), 1), mode="nearest")
    rise = float(own[k] - near.surface(rows_xy[k:k + 1], side)[0])
    blend = float(np.clip(max(BLEND_M, abs(rise) / MAX_BLEND_GRADE), BLEND_M, max(BLEND_M, 0.8 * (n_rows - 1))))
    row_tree = cKDTree(rows_xy)

    def height(p):
        p = np.atleast_2d(np.asarray(p, float))
        _, kk = row_tree.query(p)
        t = np.column_stack([np.cos(rows_h[kk]), np.sin(rows_h[kk])])
        rel = p - rows_xy[kk]
        d = np.clip(kk + (rel * t).sum(axis=1), 0.0, n_rows - 1)
        lateral = (rel * rows_n[kk]).sum(axis=1)
        ramp_h = np.interp(d, np.arange(n_rows), own) - crown * np.abs(lateral)
        main_h = near.surface(p, side)
        wgt = stubs._smoothstep(near.beyond(p, side) / blend)
        return main_h + wgt * (ramp_h - main_h)

    z = height(rows_xy)
    z_left = height(rows_xy + w * rows_n)
    z_right = height(rows_xy - w * rows_n)
    chain_h = height(chain)
    chain_h[0] = m.edge_h(taper_row, side)
    chain_h[-1] = z_left[k] if es_out > 0 else z_right[k]

    carriageway = shapely.Polygon(np.vstack([rows_xy[k:] + w * rows_n[k:], (rows_xy[k:] - w * rows_n[k:])[::-1]]))
    outline = stubs._paved_outline(throat, rows_xy[k:], rows_n[k:], w)
    if outline is None or not carriageway.is_valid:
        return None, "outline"
    style_name = sec.STYLES[style]
    # Paint where the road is sealed and marked: not gravel, and not tagged `lane_markings=no`. A one-lane carriageway
    # (`road_plain`) has its edge lines, so its gores are painted too.
    main_marked = not sec.is_unsealed(m.style[row_at]) and (r.get("main") or {}).get("lane_markings") != "no"
    paint = "none"
    polygons: list = []
    if main_marked and not unsealed:
        paint = "diagonals" if r.get("kind") == "two_way" else "chevrons"
        polygons = _paint(throat, rows_xy, rows_n, w, es_in, es_out, k, d_split, chain, m, near, side, sign,
                          split_row, nose_row, dirn, style_name, paint, spacing_m(r.get("main") or {}),
                          r.get("kind"), height)
    # How far the ramp has turned away from the road by the nose, against the way the road runs beside it.
    hm = float(np.interp(s_nose, m.s, m.heading)) + (0.0 if dirn > 0 else math.pi)
    nose_angle = abs(math.degrees(math.atan2(math.sin(rows_h[k] - hm), math.cos(rows_h[k] - hm))))

    def r2(a):
        return np.round(np.asarray(a, float), 2).tolist()

    def r3(a):
        return np.round(np.asarray(a, float), 3).tolist()

    # The rows kept, from LEAD_M before the nose (d = 0 there): the closure stands past the nose, and looks for its
    # place from the first row on, measuring from the road near the ramp (`stub_closures`).
    k0 = max(k - int(LEAD_M), 0)
    kept = slice(k0, n_rows)
    n_kept = n_rows - k0
    record = {
        "junction": r["id"], "name": r.get("name"), "highway": r.get("highway"), "closure": r.get("closure"),
        "side": r["side"], "kind": r.get("kind"), "along": "ahead" if dirn > 0 else "behind",
        "node": r.get("node"), "way_id": r.get("way_id"), "junction_angle_deg": r.get("angle_deg"),
        "route_turn_deg": r.get("route_turn_deg", 0.0), "station_m": round(at, 2),
        "off_mainline_deg": r.get("off_mainline_deg"), "nose_angle_deg": round(nose_angle, 1),
        "surface": surface, "style": style_name, "width_m": round(width, 3), "width_mapped_m": round(mapped, 2),
        "width_source": source, "crown": crown, "length_m": float(n_kept - 1), "ends": r.get("ends"),
        "ribbon_from_m": float(k - k0), "rows_from_m": float(k0), "blend_m": round(blend, 1),
        "taper": {"from_m": round(s_taper, 2), "split_m": round(s_split, 2), "nose_m": round(s_nose, 2),
                  "rows": [taper_row, split_row, nose_row], "edge_break_rows": break_rows,
                  "outer_side": "left" if es_out > 0 else "right"},
        "island": {"to_m": round(s_end, 2), "mainline_rows": [nose_row, end_row], "ramp_row": int(kc - k0),
                   "outline": r2(np.asarray(island.exterior.coords)[:-1])},
        "mouth": {"from_m": round(float(m.s[lo]), 2), "to_m": round(float(m.s[hi]), 2), "rows": [lo, hi]},
        "outer": [[*p, h] for p, h in zip(r2(chain), r3(chain_h))],
        "gore": {"paint": paint, "spacing_m": spacing_m(r.get("main") or {}) if paint != "none" else None,
                 "polygons": polygons},
        "rows": {"d": list(range(n_kept)), "xy": r2(rows_xy[kept]), "z": r3(z[kept]), "z_left": r3(z_left[kept]),
                 "z_right": r3(z_right[kept]), "heading": np.round(rows_h[kept], 5).tolist(),
                 "curvature": np.round(rows_k[kept], 5).tolist(),
                 "grade": np.round(np.gradient(z)[kept], 4).tolist(), "half_m": [round(w, 3)] * n_kept},
        "outline": r2(np.asarray(outline.exterior.coords)[:-1]),
    }
    priority = (0 if r.get("closure") == "major" else 1, -(n_rows - 1), r["id"])
    return _Candidate(record, priority, throat.buffer(0), island.buffer(0), rows_xy[kept], rows_n[kept], w, k - k0,
                      kc - k0), None


def _paint(throat, rows_xy, rows_n, w, es_in, es_out, k, d_split, chain, m, near, side, sign, split_row, nose_row,
           dirn, style, paint, spacing, kind, height) -> list:
    """The gore's paint, as polygons inside the throat (module notes): each {"exterior": [[x, y, h], ...]} with
    `holes` where it has any, to the millimetre."""
    tex_w = stubs.TEXTURE_WIDTH_M.get(style)
    if tex_w is None:
        return []
    line_c = textures.EDGE_INSET_M / tex_w * 2 * w  # the ramp texture's edge line: its centre in from the edge
    line_w = textures.LINE_W_M / tex_w * 2 * w
    k_split = max(int(math.floor(d_split)), 0)
    inner = rows_xy + es_in * w * rows_n
    lane = shapely.Polygon(np.vstack([rows_xy[:k + 1] + w * rows_n[:k + 1],
                                      (rows_xy[:k + 1] - w * rows_n[:k + 1])[::-1]])).buffer(0)
    # The gore: the throat between the road's edge and the ramp's inner edge, the piece at the nose.
    pieces = [g for g in shapely.get_parts(throat.difference(lane)) if g.area > 0.05]
    if not pieces:
        return []
    nose_mid = (m.edge(nose_row, side) + inner[k]) / 2
    gore = min(pieces, key=lambda g: g.distance(shapely.Point(nose_mid)))
    # The lines: the ramp's inner edge from the split, and its outer edge from the taper's start, each at the ramp
    # texture's own line offset, inside the carriageway.
    inner_line = shapely.LineString(inner[k_split:k + 1])
    in_strip = inner_line.buffer(line_c + line_w / 2, cap_style="flat").difference(
        inner_line.buffer(max(line_c - line_w / 2, 0.0), cap_style="flat")).intersection(lane)
    outer_line = shapely.LineString(chain)
    out_strip = outer_line.buffer(line_c + line_w / 2, cap_style="flat").difference(
        outer_line.buffer(max(line_c - line_w / 2, 0.0), cap_style="flat")).intersection(throat)
    # The gore's centre line: half way across it at each of the road's rows from the split to the nose, measured in
    # the road's frame; its width there.
    rows = np.arange(split_row, nose_row + dirn, dirn)
    s_in, o_in = near.project(inner[k_split:k + 1])
    g_in = sign * o_in - np.interp(s_in, m.s, m.half[side])
    order = np.argsort(s_in)
    g_at = np.interp(m.s[rows], s_in[order], g_in[order])
    if len(rows) < 3 or g_at.max() < BAR_W_M:
        return []
    axis = m.to_xy(m.s[rows], sign * (m.half[side][rows] + np.maximum(g_at, 0.0) / 2))
    t_axis = cl.cumulative(axis)
    a_hd = np.unwrap(np.arctan2(*np.gradient(axis, axis=0).T[::-1]))
    t1 = float(np.interp(BAR_W_M, g_at, t_axis)) if (np.diff(g_at) >= 0).all() else \
        float(t_axis[_first(g_at >= BAR_W_M)])
    reach = float(g_at.max()) / 2 + 2 * BAR_W_M  # each arm runs out past the gore's edge, to be cut there
    # Diverge: traffic runs from the split to the nose, so the chevrons' points face the split; merge: the other way.
    q = -1.0 if kind == "merge" else 1.0

    def place(local: shapely.Geometry) -> shapely.Geometry:
        local = shapely.segmentize(local, 0.5)

        def fn(p):
            t, u = p[:, 0], p[:, 1]
            x = np.interp(t, t_axis, axis[:, 0])
            y = np.interp(t, t_axis, axis[:, 1])
            h = np.interp(t, t_axis, a_hd)
            return np.column_stack([x - np.sin(h) * u, y + np.cos(h) * u])

        return shapely.transform(local, fn)

    bars = []
    t = t1
    while t < t_axis[-1]:
        if paint == "chevrons":
            centre = shapely.LineString([(t + q * reach, -reach), (t, 0.0), (t + q * reach, reach)])
        else:
            centre = shapely.LineString([(t - reach, -reach), (t + reach, reach)])
        bar = shapely.make_valid(place(centre.buffer(BAR_W_M / 2, cap_style="flat", join_style="mitre")))
        bars.append(bar.intersection(gore.buffer(-OUTLINE_GAP_M)))
        t += spacing
    bars_geom = shapely.unary_union(bars).difference(in_strip.buffer(OUTLINE_GAP_M))
    allowed = throat.buffer(-PAINT_CLEAR_M)
    paint_geom = shapely.unary_union([bars_geom, in_strip, out_strip]).intersection(allowed)
    out = []
    for part in shapely.get_parts(paint_geom):
        if part.geom_type != "Polygon" or part.area < 0.01:
            continue
        part = shapely.Polygon(np.round(np.asarray(part.exterior.coords), 3),
                               [np.round(np.asarray(h.coords), 3) for h in part.interiors])
        if not part.is_valid:
            continue

        def ring3(coords):
            c = np.asarray(coords, float)[:-1]
            return [[*p, round(float(h), 3)] for p, h in zip(c.tolist(), height(c))]

        rec = {"exterior": ring3(part.exterior.coords)}
        if part.interiors:
            rec["holes"] = [ring3(h.coords) for h in part.interiors]
        out.append(rec)
    return out


def _clip(cand: _Candidate, rows: int) -> _Candidate | None:
    """The ramp cut to its first `rows` rows of its own ribbon, or None if that would cut into its island."""
    if cand.first + rows - 1 < cand.island_row + MIN_RIBBON_M:
        return None
    rec = cand.record
    keep = cand.first + rows
    rec["rows"] = {key: v[:keep] for key, v in rec["rows"].items()}
    rec["length_m"] = float(keep - 1)
    rec["ends"] = "clipped"
    cand.rows_xy, cand.rows_n = cand.rows_xy[:keep], cand.rows_n[:keep]
    k = cand.first
    outline = stubs._paved_outline(cand.throat, cand.rows_xy[k:], cand.rows_n[k:], cand.half)
    if outline is None:
        return None
    rec["outline"] = np.round(np.asarray(outline.exterior.coords)[:-1], 2).tolist()
    return cand


def _body(cand: _Candidate) -> shapely.Geometry:
    """The ramp's own ribbon from the nose, its road and verges, as one polygon."""
    k = cand.first
    off = cand.half + mesh.VERGE_M
    left, right = cand.rows_xy[k:] + off * cand.rows_n[k:], cand.rows_xy[k:] - off * cand.rows_n[k:]
    return shapely.Polygon(np.vstack([left, right[::-1]])).buffer(0)


def _footprint(cand: _Candidate) -> shapely.Geometry:
    """Everything the ramp takes beside the road: its throat, its island, its ribbon and verges, and the taper's
    verge."""
    return shapely.unary_union([cand.throat.buffer(mesh.VERGE_M), cand.island, _body(cand)])


def _fit(cand: _Candidate, m: stubs._Mainline, kept: list[_Candidate], taken: list[tuple[str, tuple, object]],
         avoid: shapely.Geometry | None) -> tuple[_Candidate | None, str | None]:
    """The ramp cut short where its ribbon comes within a verge of another part of the route; None, with the reason,
    if too little is left, or it would overlap a roundabout, a side road's stub or a ramp already kept. `taken`: the
    stubs' and rings' (side, mouth rows, footprint)."""
    at = cand.record["station_m"]
    k = cand.first
    dist, idx = m.tree.query(cand.rows_xy[k:])
    # Another part: much further along the road than in a straight line from the junction, as `find` judges it (a
    # ramp runs alongside the road for a long way, so its own distance from the road can't say).
    straight = np.hypot(*(m.xy[idx] - m.at(at)).T)
    other = np.abs(m.s[idx] - at) > jn.OTHER_PART_RATIO * straight + jn.OTHER_PART_SLACK_M
    need = np.maximum(m.half["L"][idx], m.half["R"][idx]) + cand.half + 2 * mesh.VERGE_M + CLEAR_M
    close = np.flatnonzero(other & (dist < need))
    if len(close):
        cand = _clip(cand, int(close[0]) - 2)
        if cand is None:
            return None, "near_route"
    fp = _footprint(cand)
    if avoid is not None and fp.intersects(avoid):
        return None, "roundabout"
    a, b = cand.record["mouth"]["rows"]
    for side, (ka, kb), geom in taken + [(c.record["side"], tuple(c.record["mouth"]["rows"]), _footprint(c))
                                         for c in kept]:
        if (side == cand.record["side"] and a < kb and ka < b) or fp.intersects(geom):
            return None, "overlaps"
    return cand, None


def plan(ramps: list[dict], c, section, dem_sample, *, width_factor: float = 1.0, layouts: tuple[bool, ...] = (False,),
         junctions: list[dict] | None = None, stubs_plan: list[dict] | None = None,
         rings: list[dict] | None = None) -> dict:
    """The ramps' plan: the content of ramps.json (without `crs`). `ramps` are ramps.json's from the route stage; `c`,
    `section` and `dem_sample` as `stubs.plan` takes them. The side roads' stubs (`stubs_plan`, for `junctions`) and
    the roundabouts (`rings`, roundabouts.json's) are already planned: a ramp keeps off them, never the other way, so
    no stub drawn without ramps changes."""
    m = stubs._Mainline(c, section)
    cands, skipped = [], []
    for r in ramps:
        try:
            cand, why = _candidate(r, m, dem_sample, width_factor)
        except shapely.errors.GEOSException:
            cand, why = None, "geometry"
        if cand is None:
            skipped.append({"junction": r["id"], "name": r.get("name"), "reason": why})
        else:
            cands.append(cand)
    taken = []
    for st in stubs_plan or []:
        geom = shapely.Polygon(st["outline"]).buffer(mesh.VERGE_M + CLEAR_M)
        taken.append((st["side"], tuple(st["mouth"]["rows"]), geom))
    for ring in rings or []:
        taken.append((ring["island_side"], tuple(ring["mouth"]["rows"]), shapely.Polygon(ring["outline"]).buffer(0)))
    avoid = rb_core.avoid(rings) if rings else None
    kept: list[_Candidate] = []
    for cand in sorted(cands, key=lambda k: k.priority):
        fitted, reason = _fit(cand, m, kept, taken, avoid)
        if fitted is None:
            skipped.append({"junction": cand.record["junction"], "name": cand.record["name"], "reason": reason})
            continue
        kept.append(fitted)
    kept = _clear_gates(kept, c, section, layouts, junctions or [], stubs_plan or [], rings or [], skipped)
    sound = []
    for k in kept:
        outline = shapely.Polygon(k.record["outline"]).buffer(0.05)
        if outline.contains(shapely.MultiPoint(np.asarray(k.record["rows"]["xy"], float)[k.first:])):
            sound.append(k)
        else:
            skipped.append({"junction": k.record["junction"], "name": k.record["name"], "reason": "outline"})
    sound.sort(key=lambda k: (k.record["station_m"], k.record["junction"]))
    return {"version": FORMAT_VERSION,
            "rule": {"max_angle_deg": MAX_ANGLE_DEG, "stub_m": STUB_M, "nose_w_m": NOSE_W_M,
                     "island_end_w_m": ISLAND_END_W_M, "taper_ratio": TAPER_RATIO, "bar_w_m": BAR_W_M,
                     "spacing_m": [SPACING_SLOW_M, SPACING_M, SPACING_FAST_M], "blend_m": BLEND_M,
                     "max_blend_grade": MAX_BLEND_GRADE, "width_factor": width_factor},
            "ramps": [k.record for k in sound],
            "skipped": sorted(skipped, key=lambda s: s["junction"])}


def _clear_gates(kept: list[_Candidate], c, section, layouts, junctions, stubs_plan, rings, skipped):
    """The ramps less any whose mouth would leave a timing gate nowhere clear to stand, with the side roads' and the
    roundabouts' mouths where they are (`layout.gate_distances`)."""
    s = np.asarray(c["s"], float)
    # Every side road's mouth, with its stub's flares where it has one, as the track stage places the gates.
    mouths = jn.mouths(junctions, s, c["xy"], c["heading"], section["half_left"], section["half_right"],
                       stubs=stubs.by_junction(stubs_plan)) if junctions else []
    while kept:
        drop = None
        for reverse in layouts:
            road = RoadModel.from_arrays(s, c["xy"], c["z"], c["heading"], c["curvature"], c["grade"],
                                         section["half_left"], section["half_right"], section["slope_left"],
                                         section["slope_right"], reverse)
            fixed = jn.layout_intervals(mouths, s, reverse)
            if rings:
                fixed += rb_core.layout_intervals(rings, s, reverse)
            mine = list(zip(layout_intervals([k.record for k in kept], s, reverse, sort=False), kept))
            for d in gate_distances(road, sorted(fixed + [sp for sp, _ in mine])):
                blocking = [k for (lo, hi), k in mine if lo <= d <= hi]
                if blocking:
                    drop = max(blocking, key=lambda k: k.priority)
                    break
            if drop is not None:
                break
        if drop is None:
            return kept
        kept = [k for k in kept if k is not drop]
        skipped.append({"junction": drop.record["junction"], "name": drop.record["name"], "reason": "timing_gate"})
    return kept


# -- reading the plan -------------------------------------------------------------------------------------------------

def junctions_for(ramps: list[dict]) -> list[dict]:
    """The ramps as junctions.json's junctions, for the closures (`stub_closures.place`): the node and way their seeds
    come from, and the angles the overshoot rule reads."""
    return [{"id": r["junction"], "node": r.get("node"), "way_id": r.get("way_id"),
             "angle_deg": r.get("junction_angle_deg") or 0.0, "route_turn_deg": r.get("route_turn_deg") or 0.0,
             "highway": r.get("highway"), "name": r.get("name"), "closure": r.get("closure"), "side": r["side"]}
            for r in ramps]


def centres(ramps: list[dict]) -> np.ndarray:
    """Every ramp's own ribbon's centreline points: other roads' skirts stop short of them (`stubs.side_caps`)."""
    pts = [stubs._centre(r) for r in ramps]
    return np.vstack(pts) if pts else np.zeros((0, 2))


def with_centres(points: np.ndarray | None, ramps: list[dict]) -> np.ndarray | None:
    """`points` (or none) with the ramps' centreline points added; `points` itself without ramps."""
    if not ramps:
        return points
    return centres(ramps) if points is None else np.vstack([np.asarray(points, float).reshape(-1, 2), centres(ramps)])


def masks(ramps: list[dict], s: np.ndarray, margin_m: float = jn.MOUTH_MARGIN_M) -> dict[str, np.ndarray]:
    """The stations each ramp takes on its side of the road, from the taper's start to the island's end, and
    `margin_m` either side: no guard rail, guide post, curve marker or sign stands there."""
    s = np.asarray(s, float)
    out = {"L": np.zeros(len(s), bool), "R": np.zeros(len(s), bool)}
    for r in ramps:
        out[jn.SIDE_CODE[r["side"]]] |= (s >= r["mouth"]["from_m"] - margin_m) & (s <= r["mouth"]["to_m"] + margin_m)
    return out


def with_masks(mouths: dict[str, np.ndarray] | None, ramps: list[dict], s: np.ndarray) -> dict[str, np.ndarray] | None:
    """Mouth masks per side (`junctions.masks`, or None) with the ramps' added (`masks`)."""
    if not ramps:
        return mouths
    mine = masks(ramps, s)
    return mine if mouths is None else {k: mouths[k] | mine[k] for k in "LR"}


def layout_intervals(ramps: list[dict], s: np.ndarray, reverse: bool, margin_m: float = jn.MOUTH_MARGIN_M,
                     sort: bool = True) -> list[tuple[float, float]]:
    """The ramps' stretches of the road as (from, to) metres from a layout's start: no timing gate stands in one."""
    s0, total = float(s[0]), float(s[-1] - s[0])
    spans = [(r["mouth"]["from_m"] - margin_m - s0, r["mouth"]["to_m"] + margin_m - s0) for r in ramps]
    if reverse:
        spans = [(total - b, total - a) for a, b in spans]
    return sorted(spans) if sort else spans


def zone(ramps: list[dict], beyond_m: float) -> shapely.Geometry | None:
    """The ground to keep clear of trees, buildings or power-line supports: `beyond_m` beyond each ramp's paved
    outline, and its nose island, all of it. None with no ramps."""
    if not ramps:
        return None
    parts = [shapely.Polygon(r["outline"]).buffer(beyond_m) for r in ramps]
    parts += [shapely.Polygon(r["island"]["outline"]).buffer(0) for r in ramps]
    return shapely.unary_union(parts)


def paved(r: dict) -> shapely.Geometry:
    return shapely.Polygon(r["outline"]).buffer(0)


def ground(ramps: list[dict]) -> shapely.Geometry:
    """The ground the ramps take: each one's paved outline and its verges, and its nose island."""
    return shapely.unary_union([paved(r).buffer(mesh.VERGE_M) for r in ramps]
                               + [shapely.Polygon(r["island"]["outline"]).buffer(0) for r in ramps])


# -- the meshes (the mesh stage) ---------------------------------------------------------------------------------------

@dataclass
class Built:
    """The ramps' meshes, and what the road and the terrain need to make room for them."""
    meshes: list[Mesh]
    cut: dict[str, np.ndarray]
    edge_break: dict[str, np.ndarray]
    ribbons: list[mesh.Ribbon]
    ribbon_cuts: list
    patches: list[np.ndarray]
    edges: list[tuple[np.ndarray, np.ndarray]]
    stats: dict
    failed: dict = field(default_factory=dict)  # a ramp that couldn't be built, by its id: why


def _cdt(poly: shapely.Polygon) -> np.ndarray:
    """A polygon's constrained triangulation, as (m, 3, 2) corner coordinates (exactly its own vertices)."""
    tris = shapely.get_parts(shapely.constrained_delaunay_triangles(poly))
    return np.array([np.asarray(t.exterior.coords)[:3] for t in tris]).reshape(-1, 3, 2)


def _indexed(tris: np.ndarray, heights: dict) -> tuple[np.ndarray, np.ndarray]:
    """Triangles' corners as points with heights (from `heights`, by exact coordinates) and triangle indices."""
    index: dict = {}
    pts = []
    out = []
    for t in tris.tolist():
        row = []
        for x, y in t:
            key = (x, y)
            if key not in index:
                index[key] = len(pts)
                pts.append((x, y, heights[key]))
            row.append(index[key])
        out.append(row)
    return np.array(pts, float).reshape(-1, 3), np.array(out, np.int64).reshape(-1, 3)


def _strip(inner: np.ndarray, inner_h: np.ndarray, outer: np.ndarray, outer_h: np.ndarray):
    """A strip of quads between two rows of points, as a soup piece."""
    xy = np.stack([inner, outer], axis=1)
    h = np.stack([inner_h, outer_h], axis=1)
    return np.column_stack([xy.reshape(-1, 2), h.ravel()]), None, mesh._grid_triangles(len(inner), 2)


def _verge_along(fxy: np.ndarray, fh: np.ndarray, outer_start: np.ndarray, outer_start_h: float,
                 outer_end: np.ndarray, outer_end_h: float):
    """A mown verge along a curve from one road's verge vertex to another's, as a side road's flare's (`stubs.build`):
    (piece, patch ring, outer edge, its heights)."""
    w0 = float(np.hypot(*(outer_start - fxy[0])))
    w1 = float(np.hypot(*(outer_end - fxy[-1])))
    tang = np.gradient(fxy, axis=0)
    tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
    perp = np.column_stack([-tang[:, 1], tang[:, 0]])
    if perp[0] @ (outer_start - fxy[0]) < 0:
        perp = -perp
    frac = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(fxy, axis=0).T))])
    frac /= max(frac[-1], 1e-9)
    outer = fxy + perp * (w0 + (w1 - w0) * frac)[:, None]
    outer_h = fh - mesh.VERGE_M * mesh.VERGE_FALL
    outer[0], outer_h[0] = outer_start, outer_start_h
    outer[-1], outer_h[-1] = outer_end, outer_end_h
    return _strip(fxy, fh, outer, outer_h), np.vstack([fxy, outer[::-1]]), outer, outer_h


def _ramp_caps(xy, nrm, es_in: float, others: np.ndarray, rb: mesh.Ribbon, side: str):
    """The ramp's skirt caps: nearer its own centreline than any of `others` (`stubs.side_caps`), and on its inner
    side SKIRT_CLEAR_M short of the road's own skirt, wherever that reaches."""
    cap_l, cap_r = stubs.side_caps(xy, nrm, others, stubs.REACH_M)
    col = 0 if side == "L" else rb.e.shape[1] - 1
    skirt = shapely.LineString(np.column_stack([rb.e[:, col], rb.n[:, col]]))
    edge_col = rb.road_cols.start if side == "L" else rb.road_cols.stop - 1
    road_edge = shapely.LineString(np.column_stack([rb.e[:, edge_col], rb.n[:, edge_col]]))
    d_skirt = shapely.distance(skirt, shapely.points(xy))
    d_edge = shapely.distance(road_edge, shapely.points(xy))
    room = np.where(d_skirt < d_edge, d_skirt, 0.0) - SKIRT_CLEAR_M  # the road's skirt between it and the road
    if es_in > 0:
        cap_l = np.minimum(cap_l, np.maximum(room, 0.0))
    else:
        cap_r = np.minimum(cap_r, np.maximum(room, 0.0))
    return cap_l, cap_r


def _build_one(r: dict, rb: mesh.Ribbon, frame: LocalFrame, dem_sample, skirt: bool, others: np.ndarray, tag: str):
    """One ramp's meshes and pieces (see `build`)."""
    side = jn.SIDE_CODE[r["side"]]
    es_out = 1.0 if r["taper"]["outer_side"] == "left" else -1.0
    es_in = -es_out
    taper_row, _, nose_row = r["taper"]["rows"]
    _, end_row = r["island"]["mainline_rows"]
    kc = int(r["island"]["ramp_row"])
    k = int(r["ribbon_from_m"])
    rows = r["rows"]
    xy = np.asarray(rows["xy"], float)
    z, zl, zr = (np.asarray(rows[key], float) for key in ("z", "z_left", "z_right"))
    heading, curv = np.asarray(rows["heading"], float), np.asarray(rows["curvature"], float)
    half = np.asarray(rows["half_m"], float)
    sel = stubs._ribbon_rows(r)
    if kc not in sel:
        sel = np.unique(np.concatenate([sel, [kc]]))
    ic = int(np.flatnonzero(sel == kc)[0])
    cap_l, cap_r = _ramp_caps(xy[sel], cl.normals(heading[sel]), es_in, others, rb, side)
    sb = mesh.build_ribbon(xy[sel], z[sel], heading[sel], curv[sel], half[sel], half[sel],
                           (z[sel] - zl[sel]) / half[sel], (z[sel] - zr[sel]) / half[sel], dem_sample,
                           skirt_cap_l=cap_l, skirt_cap_r=cap_r)
    style = sec.STYLES.index(r["style"])
    nr = sb.e.shape[0]
    own_cut = {"L": np.zeros(nr - 1, bool), "R": np.zeros(nr - 1, bool)}
    in_side = "L" if es_in > 0 else "R"
    own_cut[in_side][:ic] = True
    meshes = mesh.ribbon_meshes(sb, sel.astype(float), frame, style=np.full(nr, style, np.int8), skirt=skirt,
                                cut=own_cut, name_prefix=tag)
    rc = sb.road_cols
    in_col, out_col = (rc.start, rc.stop - 1) if es_in > 0 else (rc.stop - 1, rc.start)
    in_verge, out_verge = (in_col - 1, out_col + 1) if es_in > 0 else (in_col + 1, out_col - 1)
    in_skirt = list(range(in_verge - 1, -1, -1)) if es_in > 0 else list(range(in_verge + 1, sb.e.shape[1]))

    def sbp(row, col):
        return np.array([sb.e[row, col], sb.n[row, col]]), float(sb.h[row, col])

    edge_col = rb.road_cols.start if side == "L" else rb.road_cols.stop - 1
    verge_col = edge_col - 1 if side == "L" else edge_col + 1
    main_skirt = list(range(verge_col - 1, -1, -1)) if side == "L" else list(range(verge_col + 1, rb.e.shape[1]))

    def rbp(row, col):
        return np.array([rb.e[row, col], rb.n[row, col]]), float(rb.h[row, col])

    dirn = 1 if nose_row > taper_row else -1
    material = r["style"]
    tile = textures.GRAVEL_TILE_M if r["surface"] == sec.UNSEALED else textures.ROAD_TILE_M

    # The throat: the road's edge, across the nose, the ribbon's first row, back along the outer edge.
    main_rows = np.arange(taper_row, nose_row + dirn, dirn)
    main_edge = np.column_stack([rb.e[main_rows, edge_col], rb.n[main_rows, edge_col]])
    main_edge_h = rb.h[main_rows, edge_col]
    mv, _ = rbp(nose_row, verge_col)
    iv, _ = sbp(0, in_verge)
    _, in_h = sbp(0, in_col)
    cols = list(range(rc.start, rc.stop))
    if es_in < 0:
        cols = cols[::-1]
    across = np.array([[sb.e[0, cc], sb.n[0, cc], sb.h[0, cc]] for cc in cols])  # inner to outer
    outer = np.asarray(r["outer"], float)
    oxy, oh = outer[:, :2].copy(), outer[:, 2].copy()
    oxy[0], oh[0] = main_edge[0], main_edge_h[0]
    oxy[-1], oh[-1] = across[-1, :2], across[-1, 2]
    oxy, oh = stubs._dedupe(oxy, oh)
    ring = np.vstack([main_edge, mv[None], iv[None], across[:, :2], oxy[::-1][1:-1]])
    ring_h = np.concatenate([main_edge_h, [main_edge_h[-1], in_h], across[:, 2], oh[::-1][1:-1]])
    throat = shapely.Polygon(ring)
    if not throat.is_valid:
        raise ValueError("the throat isn't a simple polygon")
    heights = {(float(x), float(y)): float(h) for (x, y), h in zip(ring.tolist(), ring_h.tolist())}
    o = np.array([sb.e[0, rc.start + 2], sb.n[0, rc.start + 2]])
    h0 = float(heading[k])

    def uv_of(pts, paint_u=None):
        along = (pts - o) @ np.array([math.cos(h0), math.sin(h0)]) + k
        if paint_u is not None:
            return np.column_stack([np.full(len(pts), paint_u), along / tile])
        u0, u1 = stubs.THROAT_U[material]
        band = (u1 - u0) * stubs.TEXTURE_WIDTH_M[material]
        lateral = (pts - o) @ np.array([-math.sin(h0), math.cos(h0)])
        return np.column_stack([u0 + (u1 - u0) * stubs._tri_wave(lateral / band), along / tile])

    road_pieces = []
    painted = _paint_parts(r, throat, heights)
    if painted:
        holes = [p.exterior.coords for p in painted]
        plain = shapely.Polygon(throat.exterior.coords, holes)
        for poly, paint_u in [(plain, None)] + [(p, _paint_u(material)) for p in painted]:
            pts, tri = _indexed(_cdt(poly), heights)
            road_pieces.append((pts, uv_of(pts[:, :2], paint_u), tri))
        for p in painted:  # a hole in a piece of paint is plain road
            for hole in p.interiors:
                pts, tri = _indexed(_cdt(shapely.Polygon(hole)), heights)
                road_pieces.append((pts, uv_of(pts[:, :2]), tri))
    else:
        inner_pts, inner_h = [], []
        shapely.prepare(throat)
        boundary = throat.exterior
        for kk in range(1, k):
            nrm = cl.normals(np.array([heading[kk]]))[0]
            for t in (1.0, 0.5, 0.0, -0.5, -1.0):
                p = xy[kk] + nrm * t * half[kk]
                if throat.contains(shapely.Point(p)) and boundary.distance(shapely.Point(p)) >= stubs.STEINER_CLEAR_M:
                    inner_pts.append(p)
                    inner_h.append(z[kk] + ((zl[kk] if t > 0 else zr[kk]) - z[kk]) * abs(t))
        inner_pts = np.array(inner_pts).reshape(-1, 2)
        tri = stubs._triangulate(ring, inner_pts)
        pts = np.vstack([ring, inner_pts])
        hts = np.concatenate([ring_h, np.asarray(inner_h, float)])
        road_pieces.append((np.column_stack([pts, hts]), uv_of(pts), tri))
    key = "1GRAVEL" if r["surface"] == sec.UNSEALED else "1ROAD"
    meshes += mesh.soup_meshes(f"{key}_{tag}G", material, road_pieces, frame)
    patches = [ring]
    edges = []

    # The taper's verge: along the outer edge from the road's verge vertex at the taper's start to the ramp's outer
    # verge vertex at its first row.
    grass = []
    ms, msh = rbp(taper_row, verge_col)
    ov, ovh = sbp(0, out_verge)
    piece, patch, vout, vout_h = _verge_along(oxy, oh, ms, msh, ov, ovh)
    grass.append(piece)
    patches.append(patch)
    edges.append((vout, vout_h))

    # The nose island: the road's verge along it and the ramp's, falling from the roads' level at the nose, and the
    # ground between them down to the far end, where both skirts come back.
    isl_rows = np.arange(nose_row, end_row + dirn, dirn)
    m_edge = np.column_stack([rb.e[isl_rows, edge_col], rb.n[isl_rows, edge_col]])
    m_edge_h = rb.h[isl_rows, edge_col]
    m_verge = np.column_stack([rb.e[isl_rows, verge_col], rb.n[isl_rows, verge_col]])
    m_dist = np.abs(rb_stations(rb, isl_rows) - rb_stations(rb, isl_rows)[0])
    fall = np.clip(m_dist / NOSE_EASE_M, 0.0, 1.0)
    fall[-1] = 1.0
    m_verge_h = m_edge_h + (rb.h[isl_rows, verge_col] - m_edge_h) * fall
    r_edge = np.column_stack([sb.e[:ic + 1, in_col], sb.n[:ic + 1, in_col]])
    r_edge_h = sb.h[:ic + 1, in_col]
    r_verge = np.column_stack([sb.e[:ic + 1, in_verge], sb.n[:ic + 1, in_verge]])
    r_fall = np.clip((sel[:ic + 1] - sel[0]) / NOSE_EASE_M, 0.0, 1.0)
    r_fall[-1] = 1.0
    r_verge_h = r_edge_h + (sb.h[:ic + 1, in_verge] - r_edge_h) * r_fall
    grass.append(_strip(m_edge, m_edge_h, m_verge, m_verge_h))
    grass.append(_strip(r_edge, r_edge_h, r_verge, r_verge_h))
    far_main = [rbp(end_row, cc) for cc in main_skirt]
    far_ramp = [sbp(ic, cc) for cc in in_skirt][::-1]
    core = np.vstack([m_verge, [p for p, _ in far_main], [p for p, _ in far_ramp], r_verge[::-1]])
    core_h = np.concatenate([m_verge_h, [h for _, h in far_main], [h for _, h in far_ramp], r_verge_h[::-1]])
    core_poly = shapely.Polygon(core)
    if not core_poly.is_valid:
        raise ValueError("the nose island isn't a simple polygon")
    core_heights = {(float(x), float(y)): float(h) for (x, y), h in zip(core.tolist(), core_h.tolist())}
    pts, tri = _indexed(_cdt(core_poly), core_heights)
    grass.append((pts, None, tri))
    island_ring = np.vstack([m_edge, [p for p, _ in far_main], [p for p, _ in far_ramp], r_edge[::-1], iv[None],
                             mv[None]])
    patches.append(island_ring)
    grass = [(p, frame.to_ac(p[:, 0], p[:, 1], p[:, 2])[:, [0, 2]] / 6.0, t) for p, _, t in grass]
    meshes += mesh.soup_meshes(f"1GRASS_{tag}I", "grass", grass, frame)
    lo, hi = r["mouth"]["rows"]
    br0, br1 = r["taper"]["edge_break_rows"]
    return meshes, sb, own_cut, patches, edges, (side, lo, hi, br0, br1)


def rb_stations(rb: mesh.Ribbon, rows: np.ndarray) -> np.ndarray:
    """Distance along the road's centre column to each of `rows`, from its first row."""
    col = rb.road_cols.start + 2
    xy = np.column_stack([rb.e[:, col], rb.n[:, col]])
    s = cl.cumulative(xy)
    return s[np.asarray(rows)]


def _paint_u(material: str) -> float:
    """The u of the middle of a sealed texture's edge line: the white the paint is drawn with."""
    return textures.EDGE_INSET_M / stubs.TEXTURE_WIDTH_M[material]


def _paint_parts(r: dict, throat: shapely.Polygon, heights: dict) -> list[shapely.Polygon]:
    """The gore's paint as built: each polygon inside the throat as built, apart from the others; none at all if any
    isn't (the throat is then plain). Their vertices' heights go into `heights`."""
    if r["gore"]["paint"] == "none" or r["style"] not in ("road", "road_plain"):
        return []
    inside = throat.buffer(-0.01)
    out, found = [], {}
    for rec in r["gore"]["polygons"]:
        ext = np.asarray(rec["exterior"], float)
        holes = [np.asarray(h, float) for h in rec.get("holes", [])]
        poly = shapely.Polygon(ext[:, :2], [h[:, :2] for h in holes])
        if not poly.is_valid or not inside.contains(poly):
            return []
        for ring in [ext] + holes:
            for x, y, h in ring.tolist():
                found[(x, y)] = h
        out.append(poly)
    if out and abs(shapely.unary_union(out).area - sum(p.area for p in out)) > 1e-4:
        return []
    heights.update(found)
    return out


def build(ramps: list[dict], rb: mesh.Ribbon, s: np.ndarray, frame: LocalFrame, dem_sample, *, skirt: bool,
          others: np.ndarray) -> Built:
    """The ramps' meshes from the plan: each ramp's ribbon (`1ROAD_JS<k>_###`, its verges `1GRASS_JS<k>_V###L/R`),
    its throat (`1ROAD_JS<k>_G`: the taper and the gore, with the gore's paint) and its grass (`1GRASS_JS<k>_I`: the
    taper's verge and the nose island); the road's cut and edge break; and the pieces the terrain leaves room for. `rb`
    is the road's ribbon (its skirt held short of the ramps, `centres`), `others` every other centreline point the
    ramps' skirts stop short of. A ramp that can't be built is left out of everything and listed in `failed`, for the
    checks to drop it (`faults`)."""
    rows = len(s)
    cut = {"L": np.zeros(rows - 1, bool), "R": np.zeros(rows - 1, bool)}
    edge_break = {"L": np.zeros(rows, bool), "R": np.zeros(rows, bool)}
    out = Built([], cut, edge_break, [], [], [], [], {}, {})
    others = np.asarray(others, float).reshape(-1, 2)
    for r in ramps:
        tag = f"JS{int(r['junction']) - RAMP_ID_BASE}_"
        rest = centres([q for q in ramps if q is not r])
        try:
            meshes, sb, own_cut, patches, edges, (side, lo, hi, br0, br1) = _build_one(
                r, rb, frame, dem_sample, skirt, np.vstack([others, rest]), tag)
        except (ValueError, KeyError, IndexError, shapely.errors.GEOSException) as e:
            out.failed[r["junction"]] = f"{type(e).__name__}: {e}"
            continue
        cut[side][lo:hi] = True
        edge_break[side][br0:br1 + 1] = True
        out.meshes += meshes
        out.ribbons.append(sb)
        out.ribbon_cuts.append(own_cut)
        out.patches += patches
        out.edges += edges
    built = [r for r in ramps if r["junction"] not in out.failed]
    out.stats = {"ramps": len(built), "meshes": len(out.meshes),
                 "vertices": int(sum(len(m.positions) for m in out.meshes)),
                 "triangles": int(sum(len(m.triangles) for m in out.meshes)),
                 "length_m": round(sum(r["length_m"] - r["ribbon_from_m"] for r in built), 1),
                 "painted": sum(r["gore"]["paint"] != "none" for r in built)}
    return out


def footprint(base: mesh.Footprint | None, built: Built, rows: int) -> mesh.Footprint:
    """The terrain's footprint (`base`: the side roads' stubs and roundabouts', or None) with the ramps' added: their
    ribbons, with their inner verge and skirt cut beside the island, and their throats', verges' and islands' rings."""
    if base is None:
        base = mesh.Footprint({"L": np.zeros(rows - 1, bool), "R": np.zeros(rows - 1, bool)}, [], [], [])
    n = len(base.ribbons)
    cut = {key: base.cut[key] | built.cut[key] for key in "LR"}
    return mesh.Footprint(cut, base.ribbons + built.ribbons, base.patches + built.patches, base.edges + built.edges,
                          ribbon_cuts=(base.ribbon_cuts or [None] * n) + built.ribbon_cuts,
                          open_last=(base.open_last or [False] * n) + [False] * len(built.ribbons),
                          interior=base.interior)


def road_cut(cut: dict | None, edge_break: dict | None, built: Built | None):
    """The road's verge-and-skirt cut and edge-line break with the ramps' added (`mesh.ribbon_meshes`)."""
    if built is None:
        return cut, edge_break
    if cut is None:
        return built.cut, built.edge_break
    return ({key: cut[key] | built.cut[key] for key in "LR"},
            {key: edge_break[key] | built.edge_break[key] for key in "LR"})


# -- checking them as built --------------------------------------------------------------------------------------------

def check(r: dict, c, section, frame: LocalFrame):
    """A ramp as the validator's stub checks see it (`validate.StubCheck`), in AC x/z: its paved outline, points along
    its own ribbon, and pairs either side of its seams with the road's edge (where the throat is SEAM_MIN_W_M wide or
    more) and with its ribbon's first row; and its nose island, with no hole round it."""
    from mapcreator.core import validate

    def ac(p):
        p = np.asarray(p, float).reshape(-1, 2)
        return frame.to_ac(p[:, 0], p[:, 1], np.zeros(len(p)))[:, [0, 2]]

    side = jn.SIDE_CODE[r["side"]]
    sign = 1.0 if side == "L" else -1.0
    half = np.asarray(section["half_left" if side == "L" else "half_right"], float)
    heading, xy = np.asarray(c["heading"], float), np.asarray(c["xy"], float)
    taper_row, _, nose_row = r["taper"]["rows"]
    lo, hi = sorted((taper_row, nose_row))
    rows = np.arange(lo + 2, hi - 1)
    nrm = np.column_stack([-np.sin(heading[rows]), np.cos(heading[rows])]) * sign
    edge = xy[rows] + nrm * half[rows, None]
    paved_poly = shapely.Polygon(r["outline"]).buffer(0)
    wide = shapely.contains_xy(paved_poly, *(edge + nrm * SEAM_MIN_W_M).T)
    edge, nrm = edge[wide], nrm[wide]
    pairs = [np.stack([edge - validate.SEAM_OFFSET_M * nrm, edge + validate.SEAM_OFFSET_M * nrm], axis=1)]
    rr = r["rows"]
    k = int(r["ribbon_from_m"])
    hd = float(rr["heading"][k])
    u, n = np.array([np.cos(hd), np.sin(hd)]), np.array([-np.sin(hd), np.cos(hd)])
    w = float(rr["half_m"][k])
    across = np.asarray(rr["xy"][k], float) + np.outer(np.arange(-0.9 * w, 0.9 * w + 1e-9, 0.5), n)
    pairs.append(np.stack([across - validate.SEAM_OFFSET_M * u, across + validate.SEAM_OFFSET_M * u], axis=1))
    seams = np.concatenate(pairs)
    centre = np.asarray(rr["xy"], float)[k + 1:-1:2]
    ramp = validate.StubCheck(shapely.Polygon(ac(r["outline"])).buffer(0), ac(centre),
                              ac(seams.reshape(-1, 2)).reshape(-1, 2, 2))
    island = validate.StubCheck(shapely.Polygon(ac(r["island"]["outline"])).buffer(0), np.zeros((0, 2)),
                                np.zeros((0, 2, 2)))
    return ramp, island


def checks(ramps: list[dict], c, section, frame: LocalFrame) -> list:
    """Every ramp's checks, keyed `ramp:<id>` and `ramp:<id>:island`, for `stubs.faults(extra=)`: their closures
    count as on a stub, and their faults are theirs (`faults_of`)."""
    out = []
    for r in ramps:
        ramp, island = check(r, c, section, frame)
        out += [(f"ramp:{r['junction']}", ramp), (f"ramp:{r['junction']}:island", island)]
    return out


def is_ramp_key(key) -> bool:
    return isinstance(key, str) and key.startswith("ramp:")


def without_ramps(found: dict) -> dict:
    """`stubs.faults`' result less the ramps' (the side roads' and roundabouts' settling leaves those to `settle`)."""
    return {key: v for key, v in found.items() if not is_ramp_key(key)}


def faults_of(found: dict, built: Built | None = None) -> dict[int, list[str]]:
    """The ramps' faults by their ids, from `stubs.faults`' result, and any ramp `build` couldn't draw."""
    out: dict[int, list[str]] = {}
    for key, v in found.items():
        if is_ramp_key(key) and v:
            rid = int(key.split(":")[1])
            out.setdefault(rid, []).extend(f"island {f}" if key.endswith(":island") else f for f in v)
    for rid, why in (built.failed if built is not None else {}).items():
        out.setdefault(rid, []).append("not_built")
    return out


def settle(state: dict, run, check_ramps, rebuilds: int = 3, log=None):
    """Run the side roads' and roundabouts' settling (`run()`, `stubs.settle` or `roundabouts.settle`) with
    `state["ramps"]`, check the ramps on what it built (`check_ramps(ramps, out)`: faults by ramp id), and run it again
    without any ramp found faulty, until none is; after `rebuilds` rounds every ramp goes, so a ramp never fails a
    build. What it left out goes in `state["dropped"]` as ramps.json's `skipped` records (`build_check`, with their
    `faults`). Returns `run()`'s last result."""
    dropped = state.setdefault("dropped", [])
    for attempt in range(rebuilds + 1):
        out = run()
        ramps = state["ramps"]
        if not ramps:
            return out
        found = check_ramps(ramps, out)
        if not found:
            return out
        for r in ramps:
            if r["junction"] in found or attempt >= rebuilds:
                dropped.append({"junction": r["junction"], "name": r.get("name"), "reason": "build_check",
                                "faults": found.get(r["junction"], [])})
                if log is not None and r["junction"] in found:
                    log.warning("slip ramp left out: %s (%s) at %.0f m: %s", r.get("name") or "an unnamed ramp",
                                r["junction"], r["station_m"], ", ".join(found[r["junction"]]))
        state["ramps"] = [] if attempt >= rebuilds else [r for r in ramps if r["junction"] not in found]
    return run()
