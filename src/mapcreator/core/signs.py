"""Regulatory signs and zebra crossings: phase 2 of the traffic lights research (answered 24 Nov 2026), "regulatory
signs we can stand behind".

**What's drawn, and from what** (`find`, the route stage; `01_route/signs.json`):
- **GIVE WAY (R1-2) and STOP (R1-1) only where OpenStreetMap tags them** (Q8): a road node with `highway=give_way` or
  `highway=stop` (or `traffic_sign` AU:R1-2 or AU:R1-1), on the route or on a side road the route stage found
  (`core.junctions`), within `REACH_M` of the route's junction it's for. Never inferred at an untagged mouth: the
  T-intersection rule leaves many unsigned. Which traffic it's for, so which way it faces:
  - its `direction` (`forward` or `backward`, along its way; `both` gives two), where tagged;
  - else the way its approach runs: towards the route's nearest junction along its road (a side road's into the
    route, the route's own to the junction ahead or behind it), and only a way traffic may drive.
  A sign at a junction's own node, one whose road meets the route nowhere near, or one on a side road the track
  doesn't draw is left out (`skipped`, with the reason).
- **Pedestrian crossing signs (R3-1) and zebra markings** at a crossing node on the route or a drawn side road:
  signs where `crossing:markings=zebra` (or `crossing_ref=zebra`), or `crossing=marked` with no other markings
  tagged; the zebra's stripes only at `crossing:markings=zebra` (or `crossing_ref=zebra`). A signalised crossing is
  the signals' (`core.signals`) and gets neither.
- **ONE WAY (R2-2)** from `oneway`, where the route is one-way through a junction: on the far side of the route from
  each side road whose traffic may drive in to it, facing that side road, its arrow along the route's flow; and
  repeated along a one-way block longer than `ONE_WAY_REPEAT_M`, on the left, both faces. Not on motorways, links or
  roundabouts, nor at a junction where the route turns (`signals.TURN_DEG`): the one-way road isn't "across" a
  corner. Nor on one carriageway of a divided road (a road of the same name or ref one-way the other way within
  `DIVIDED_M`): its one-way is the divided road's, not a one-way street's, and the far side is its median.
- **NO ENTRY (R2-4)** at the exit end of a one-way road (the research's 3): a side road traffic may only drive out of,
  into the route, gets one either side of its mouth, facing the route. A one-way road that carries on past the
  junction, one-way out of it (the same way or the same name: a one-way street crossing the route, or the road the
  route turns onto), isn't at its exit end there (`_carries_on`). And the route itself where it drives against its
  road's `oneway`, which the router never does (it follows `oneway`), so that one is a safety net.
- Side roads leaving under `MIN_ANGLE_DEG` (a divided road's other carriageway, a lane beside it) and links give no
  ONE WAY or NO ENTRY, as they give no blade (`signals.BLADE_MIN_ANGLE_DEG`).
- Not here: Hobart's parking signs along the kerb (Q9: not yet), turn bans, cycling (phase 3).

**Where they stand** (`build`, the mesh stage): real signs where they really stand, on a closed road too. A sign
on a side road (its GIVE WAY or STOP, its NO ENTRY, a crossing on it) faces the side road's traffic as in reality,
so from the route you see its back beside the closure; a closure's own signs and props stay as they are.
- **On the route's verge:** `SIGN_OFFSET_M` beyond the road's edge (on a kerbed town stretch, on the footpath
  `kerbs.POST_BEHIND_M` behind the kerb face, as the signal posts are: `signals._Road.place`), on the left of the
  traffic it's for (ONE WAY: the far side), kept out of side roads', ramps', roundabouts' and driveways' mouths and off
  bridges as the furniture and the signal posts are; one due in one moves upstream (a crossing's either way) by at
  most `MAX_SHIFT_M`, or isn't put up.
- **Beside a side road:** `SIDE_OFFSET_M` beyond its carriageway at the node (further out where its stub's paved
  edge is wider, by at most `SIDE_WIDER_M`), and outside its stub's paved outline, flares included, by
  `SIDE_CLEAR_M`, which keeps it clear of the closure's tape and boards (they reach `stub_closures.VERGE_SPAN_M` onto
  the verge); outside every other side road's and driveway's, and clear of the route's edge by `SIGN_OFFSET_M`. One
  that can't stand there moves out along the side road by at most `MAX_SHIFT_M`. Only on a side road with a stub:
  one left out (or never drawn) gets none, nor a ONE WAY facing it.
- **Posts:** a sign within `SNAP_M` of a signal, mast or blade post (`signals.Built.posts`) goes on that post, as
  real ones do, below its lanterns and blades; signs within `SNAP_M` of each other share one. Otherwise each has its
  own 60 mm galvanised post, solid as the signal posts are (Q5: `1WALL_SIGN_k`). Signs facing opposite ways share a
  height, back to back; others stack.
- **Zebra crossings:** white stripes `ZEBRA_STRIPE_M` wide with `ZEBRA_GAP_M` between them, `ZEBRA_LENGTH_M` along
  the road, across the carriageway inside `ZEBRA_EDGE_M` of each edge (estimates). Drawn as their own visual-only
  mesh `ZEBRA_LIFT_M` above the road's surface, which they follow (its heights sampled under every vertex): the road
  mesh is unchanged, so the physics, the AI line and the road's meshes are byte for byte as without them.

**Faces** (`atlas`): drawn here from plain written descriptions, shapes, colours and words; no AS 1742 artwork and
no Commons file is traced or copied, and the pedestrian on R3-1 is our own (`signals._figure`). Lettering in
Overpass (`core.lettering`). Sizes are estimates (`FACES`). `RETRO_MATERIAL` for the faces, brightened at night as
the speed signs are; `MATERIAL` for the posts and the backs; `PAINT_MATERIAL` for the stripes. None has specular
(`KN5_PROPERTIES`): the night washout lesson of the closures' signs.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from mapcreator.core import junctions as junctions_core
from mapcreator.core import lettering, signals
from mapcreator.core.centreline import cumulative
from mapcreator.core.scene import LocalFrame, Mesh

FORMAT_VERSION = 1

# -- finding them (route stage) ---------------------------------------------------------------------------------------
REACH_M = 30.0  # a tagged sign or crossing this far, along its road, from the route's junction it's for
END_CLEAR_M = signals.END_CLEAR_M  # none this near the route's ends: the grid, pits and end walls
MIN_ANGLE_DEG = signals.BLADE_MIN_ANGLE_DEG  # a side road leaving closer than this gives no ONE WAY or NO ENTRY
NO_ONE_WAY_CLASSES = frozenset({"motorway"})  # and every link and roundabout: not one-way streets
ONE_WAY_REPEAT_M = 200.0  # a one-way block longer than this gets repeaters, about this far apart (estimate)
DIVIDED_M = 40.0  # the same name, one-way the other way, this near: one carriageway of a divided road
STOP_SIGNS = frozenset({"AU:R1-1", "stop"})
GIVE_WAY_SIGNS = frozenset({"AU:R1-2", "give_way"})

# -- drawing (mesh stage) ---------------------------------------------------------------------------------------------
SIGN_OFFSET_M = 1.0  # beyond the route's edge, as the blades' own posts (`signals.BLADE_OFFSET_M`)
SIDE_OFFSET_M = 1.5  # beyond a side road's carriageway
SIDE_CLEAR_M = 1.3  # outside a stub's paved outline: past the closure's lines, which reach 1 m onto the verge
SIDE_WIDER_M = 3.0  # a stub's paved edge may lie this much beyond its mapped width: the sign moves out with it
MAX_SHIFT_M = 15.0  # how far a sign may move to stand clear (`signals.MAX_SHIFT_M`)
STEP_M = 0.5
SNAP_M = 1.0  # a sign this near a post goes on it
SIGN_BOTTOM_M = 1.5  # a sign's lower edge above the ground under it, as the speed signs (`furniture.SIGN_BOTTOM_M`)
STACK_GAP_M = 0.08
SIGNAL_POST_TOP_M = 2.45  # a sign on a signal or blade post stays below its lanterns and blades
OWN_POST_TOP_M = 3.3
POST_D = 0.06  # a sign post (estimate)
PLATE_M = 0.008  # a sign's plate, front face to back face
ZEBRA_STRIPE_M = 0.6  # estimates
ZEBRA_GAP_M = 0.6
ZEBRA_LENGTH_M = 3.0
ZEBRA_EDGE_M = 0.3
ZEBRA_LIFT_M = 0.02
ZEBRA_ROW_M = 0.5  # the stripes' vertices this far apart along the road, so they follow its surface
ZEBRA_MAX_DROP_M = 0.5  # a stripe whose surface is this far from the road's height there isn't on it (a deck above)

# Each face: (cell, width, height, shape) in metres (estimates: AS 1742's sizes aren't public).
FACES = {
    "stop": ("stop", 0.75, 0.75, "octagon"),
    "give_way": ("give_way", 0.9, 0.9 * math.sqrt(3) / 2, "triangle"),
    "one_way": ("one_way_R", 0.9, 0.3, "rect"),
    "no_entry": ("no_entry", 0.6, 0.9, "rect"),
    "pedestrian_crossing": ("pedestrian_crossing", 0.6, 0.6, "rect"),
}

MATERIAL = "sign"
RETRO_MATERIAL = "sign_retro"
PAINT_MATERIAL = "sign_paint"
TEXTURE = "signs"
MESH_PREFIX = "SIGN"  # SIGN_k; the posts' colliders 1WALL_SIGN_k
KN5_PROPERTIES = {
    MATERIAL: {"ksAmbient": 0.4, "ksDiffuse": 0.6, "ksSpecular": 0.0, "ksSpecularEXP": 10.0},
    RETRO_MATERIAL: {"ksAmbient": 0.4, "ksDiffuse": 0.6, "ksSpecular": 0.0, "ksSpecularEXP": 10.0},
    # The road's own values (`track_stage.MATERIAL_PROPS["road"]`), less its sheen, so the paint lights as the
    # road's painted lines do.
    PAINT_MATERIAL: {"ksAmbient": 0.35, "ksDiffuse": 0.42, "ksSpecular": 0.0, "ksSpecularEXP": 28.0},
}
NIGHT_DIFFUSE = {RETRO_MATERIAL: 2.0}  # as the speed signs' faces (`furniture.NIGHT_DIFFUSE`)


# =====================================================================================================================
# Finding them
# =====================================================================================================================

def sign_kind(tags: dict) -> str | None:
    """"stop" or "give_way" for a node OSM tags as one (`highway`, or `traffic_sign` AU:R1-1 or AU:R1-2), else None."""
    hw = tags.get("highway")
    if hw in ("stop", "give_way"):
        return hw
    values = {v.strip() for v in re.split(r"[;,]", str(tags.get("traffic_sign") or "")) if v.strip()}
    if values & STOP_SIGNS:
        return "stop"
    if values & GIVE_WAY_SIGNS:
        return "give_way"
    return None


def crossing_plan(tags: dict) -> dict | None:
    """What a crossing node gets: {"signs", "zebra"}, or {"reason"} for one that might have been and isn't (a
    signalised zebra, a marked crossing with other markings), or None for any other node."""
    if not signals.is_crossing(tags):
        return None
    markings = tags.get("crossing:markings")
    zebra = markings == "zebra" or tags.get("crossing_ref") == "zebra" or tags.get("crossing") == "zebra"
    marked = tags.get("crossing") == "marked"
    if not (zebra or marked):
        return None
    if signals.is_signal(tags):
        return {"reason": "signalised"}
    if zebra:
        return {"signs": True, "zebra": True}
    if markings in (None, ""):
        return {"signs": True, "zebra": False}
    return {"reason": "not_zebra"}


def is_link(tags: dict) -> bool:
    return str(tags.get("highway") or "").endswith("_link")


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.hypot(*v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _heading(v) -> float:
    return round(math.atan2(float(v[1]), float(v[0])), 4)


class _Net:
    """The road graph as the finder walks it: neighbours, who may drive which way, the route's nodes, its junctions'
    side roads."""

    def __init__(self, g, route_xy: np.ndarray, junctions: Sequence[dict]):
        self.g = g
        self.xy = np.asarray(g.xy, float)
        self.route_xy = np.asarray(route_xy, float)
        self.s = cumulative(self.route_xy)
        self.adj: dict[int, list[tuple[int, int]]] = {}
        for j, (a, b) in enumerate(zip(g.seg_a.tolist(), g.seg_b.tolist())):
            self.adj.setdefault(a, []).append((b, j))
            self.adj.setdefault(b, []).append((a, j))
        self.degree = {v: len({nb for nb, _ in nbs}) for v, nbs in self.adj.items()}
        self.idx = {int(o): i for i, o in enumerate(g.osm_node.tolist())}
        d, gi = cKDTree(self.xy).query(self.route_xy)
        self.gi = gi
        self.on_node = d < 0.01
        self.route_k: dict[int, int] = {}
        for k in np.flatnonzero(self.on_node).tolist():
            self.route_k.setdefault(int(gi[k]), k)
        self.ring = {int(x) for w, a, b in zip(g.seg_way, g.seg_a, g.seg_b)
                     if (g.ways[w]["tags"] or {}).get("junction") in ("roundabout", "circular") for x in (a, b)}
        self.leg_of: dict[tuple[int, int], dict] = {}
        for j in junctions:
            jn = self.idx.get(int(j["node"]))
            if jn is None or len(j.get("line") or []) < 2:
                continue
            p = np.asarray(j["line"][1], float)
            nbs = [nb for nb, _ in self.adj.get(jn, [])]
            if nbs:
                nb = min(nbs, key=lambda x: float(np.hypot(*(self.xy[x] - p))))
                self.leg_of[(jn, nb)] = j

    def seg(self, u: int, w: int) -> int | None:
        for nb, j in self.adj.get(u, []):
            if nb == w:
                return j
        return None

    def allowed(self, u: int, w: int) -> bool:
        """May traffic drive from node u to its neighbour w?"""
        j = self.seg(u, w)
        if j is None:
            return False
        return bool(self.g.seg_fwd[j]) if int(self.g.seg_a[j]) == u else bool(self.g.seg_rev[j])

    def one_way(self, u: int, w: int) -> bool:
        return self.allowed(u, w) != self.allowed(w, u)

    def forward(self, u: int, w: int) -> bool | None:
        """Does the way run from u to its neighbour w (its `forward`)?"""
        j = self.seg(u, w)
        return None if j is None else int(self.g.seg_a[j]) == u

    def tags(self, u: int, w: int) -> dict:
        j = self.seg(u, w)
        return {} if j is None else (self.g.ways[int(self.g.seg_way[j])]["tags"] or {})

    def neighbours(self, v: int) -> list[int]:
        return list(dict.fromkeys(nb for nb, _ in self.adj.get(v, [])))

    def walk(self, v: int, nb: int, reach: float) -> tuple[int, int, float] | None:
        """From v through its neighbour nb, along the road through plain nodes (two neighbours) to the first node
        where roads meet or end: (that node, the node before it, the distance), or None past `reach`."""
        prev, cur = v, nb
        dist = float(np.hypot(*(self.xy[cur] - self.xy[v])))
        seen = {v}
        while dist <= reach:
            if self.degree.get(cur, 0) != 2 or cur in seen:
                return cur, prev, dist
            seen.add(cur)
            nxt = [x for x in self.neighbours(cur) if x != prev]
            if not nxt:
                return cur, prev, dist
            prev, cur = cur, nxt[0]
            dist += float(np.hypot(*(self.xy[cur] - self.xy[prev])))
        return None

    def route_junction(self, k: int) -> bool:
        return bool(self.on_node[k]) and self.degree.get(int(self.gi[k]), 0) >= 3

    def route_walk(self, k: int, step: int, reach: float) -> tuple[int, float] | None:
        """Along the route from its vertex k (+1 ahead, -1 behind) to the first junction node within `reach`."""
        i = k + step
        while 0 <= i < len(self.route_xy) and abs(self.s[i] - self.s[k]) <= reach:
            if self.route_junction(i):
                return i, abs(float(self.s[i] - self.s[k]))
            i += step
        return None

    def route_tangent(self, k: int) -> np.ndarray:
        a = self.route_xy[max(k - 1, 0)]
        b = self.route_xy[min(k + 1, len(self.route_xy) - 1)]
        return _unit(b - a)

    def route_allowed(self, k: int, step: int) -> bool:
        """May traffic drive along the route from vertex k to k + step?"""
        i = k + step
        if not (0 <= i < len(self.route_xy)) or not (self.on_node[k] and self.on_node[i]):
            return True  # a spliced waypoint: the way it lies on carries on either side
        return self.allowed(int(self.gi[k]), int(self.gi[i]))


def find(g, route_xy: np.ndarray, nodes: Sequence[dict], junctions: Sequence[dict], *, to_lonlat=None) -> dict:
    """The regulatory signs and zebra crossings along the route (module docstring), from the graph it was routed on
    (`osm_graph.RoadGraph`), its polyline, the tagged road nodes (each `{"id", "xy", "tags"}`, xy in the graph's CRS)
    and the side roads `core.junctions.find` found. Returns `{"version", "rule", "signs", "zebras", "skipped"}` for
    `signs.json` (docs/ac-format.md)."""
    route_xy = np.asarray(route_xy, float)
    rule = {"reach_m": REACH_M, "end_clear_m": END_CLEAR_M, "min_angle_deg": MIN_ANGLE_DEG,
            "turn_deg": signals.TURN_DEG, "one_way_repeat_m": ONE_WAY_REPEAT_M, "divided_m": DIVIDED_M,
            "no_one_way_classes": sorted(NO_ONE_WAY_CLASSES)}
    out = {"version": FORMAT_VERSION, "rule": rule, "signs": [], "zebras": [], "skipped": []}
    if len(route_xy) < 2:
        return out
    net = _Net(g, route_xy, junctions)
    length = float(net.s[-1])
    for n in sorted(nodes, key=lambda m: int(m["id"])):
        tags = n.get("tags") or {}
        kind = sign_kind(tags)
        if kind is not None:
            _tagged_sign(net, n, kind, tags, length, out)
        plan = crossing_plan(tags)
        if plan is not None:
            _crossing(net, n, plan, length, out)
    _one_way_and_no_entry(net, junctions, length, out)
    _repeaters(net, length, out)
    _against_flow(net, length, out)
    for rec in out["signs"] + out["zebras"]:
        rec["xy"] = [round(float(v), 2) for v in rec["xy"]]
        if to_lonlat is not None:
            lon, lat = to_lonlat.transform(*rec["xy"])
            rec["lonlat"] = [round(float(lon), 7), round(float(lat), 7)]
    out["signs"].sort(key=lambda r: (r["at_m"], r["kind"], r.get("node") or 0))
    out["zebras"].sort(key=lambda r: r["at_m"])
    return out


def _skip(out: dict, node, kind: str, reason: str) -> None:
    out["skipped"].append({"node": int(node), "kind": kind, "reason": reason})


def _side_leg(net: _Net, v: int, reach: float) -> list[tuple[int, dict, float, int]]:
    """The side roads node v is on, within `reach` of their junction with the route: (the neighbour towards the
    junction, the junction's record, the distance, the junction's route vertex), nearest first."""
    found = []
    for nb in net.neighbours(v):
        walked = net.walk(v, nb, reach)
        if walked is None:
            continue
        end, prev, dist = walked
        if end in net.route_k and net.degree.get(end, 0) >= 3:
            found.append((nb, net.leg_of.get((end, prev)), dist, net.route_k[end]))
    return sorted(found, key=lambda t: t[2])


def _tagged_sign(net: _Net, n: dict, kind: str, tags: dict, length: float, out: dict) -> None:
    """A GIVE WAY or STOP OSM tags: which traffic it's for and where it stands (module docstring)."""
    v = net.idx.get(int(n["id"]))
    if v is None or v not in net.adj:
        return _skip(out, n["id"], kind, "not_on_a_road")
    if net.degree.get(v, 0) >= 3:
        return _skip(out, n["id"], kind, "at_a_junction_node")
    if v in net.ring:
        return _skip(out, n["id"], kind, "roundabout")
    direction = str(tags.get("direction") or tags.get("traffic_sign:direction") or "").strip().lower()
    nbs = net.neighbours(v)
    if v in net.route_k:
        k = net.route_k[v]
        at = float(net.s[k])
        if at < END_CLEAR_M or length - at < END_CLEAR_M:
            return _skip(out, n["id"], kind, "route_end")
        steps = []
        if direction in ("forward", "backward", "both"):
            ahead = int(net.gi[k + 1]) if k + 1 < len(net.route_xy) and net.on_node[k + 1] else None
            behind = int(net.gi[k - 1]) if k > 0 and net.on_node[k - 1] else None
            fwd = net.forward(v, ahead) if ahead is not None else (
                (not net.forward(v, behind)) if behind is not None else None)
            if fwd is None:
                return _skip(out, n["id"], kind, "direction_unclear")
            if direction in ("forward", "both"):
                steps.append((1 if fwd else -1, "direction_tag"))
            if direction in ("backward", "both"):
                steps.append((-1 if fwd else 1, "direction_tag"))
        else:
            best = None
            for step in (1, -1):
                walked = net.route_walk(k, step, REACH_M)
                if walked is not None and net.route_allowed(k, step) and (best is None or walked[1] < best[1]):
                    best = (step, walked[1])
            if best is None:
                return _skip(out, n["id"], kind, "no_junction_near")
            steps.append((best[0], "approach"))
        for step, source in steps:
            travel = net.route_tangent(k) * step
            out["signs"].append({"kind": kind, "on": "route", "node": int(n["id"]), "at_m": round(at, 2),
                                 "xy": net.xy[v].tolist(), "travel": _heading(travel), "from": source})
        return None
    legs = _side_leg(net, v, REACH_M)
    if not legs:
        return _skip(out, n["id"], kind, "off_route")
    choices = []
    if direction in ("forward", "backward", "both"):
        for nb in nbs:
            fwd = net.forward(v, nb)
            if fwd is None:
                continue
            if (direction in ("forward", "both") and fwd) or (direction in ("backward", "both") and not fwd):
                choices.append((nb, "direction_tag"))
        if not choices:
            return _skip(out, n["id"], kind, "direction_unclear")
    else:
        drivable = [leg for leg in legs if net.allowed(v, leg[0])]
        if not drivable:
            return _skip(out, n["id"], kind, "no_junction_near")
        choices.append((drivable[0][0], "approach"))
    for nb, source in choices:
        leg = next((leg for leg in legs if leg[0] == nb), None)
        if leg is None:
            _skip(out, n["id"], kind, "not_for_a_route_junction")  # it faces traffic leaving towards another junction
            continue
        _, j, dist, k = leg
        if j is None:
            _skip(out, n["id"], kind, "side_road_not_drawn")
            continue
        at = float(net.s[k])
        if at < END_CLEAR_M or length - at < END_CLEAR_M:
            _skip(out, n["id"], kind, "route_end")
            continue
        travel = _unit(net.xy[nb] - net.xy[v])
        out["signs"].append({"kind": kind, "on": "side_road", "node": int(n["id"]), "junction": int(j["id"]),
                             "at_m": round(at, 2), "d_m": round(dist, 2), "xy": net.xy[v].tolist(),
                             "travel": _heading(travel), "from": source})
    return None


def _crossing(net: _Net, n: dict, plan: dict, length: float, out: dict) -> None:
    """A pedestrian crossing's signs and zebra (module docstring), on the route or a drawn side road."""
    if "reason" in plan:
        return _skip(out, n["id"], "pedestrian_crossing", plan["reason"])
    v = net.idx.get(int(n["id"]))
    if v is None or v not in net.adj:
        return _skip(out, n["id"], "pedestrian_crossing", "not_on_a_road")
    if net.degree.get(v, 0) >= 3:
        return _skip(out, n["id"], "pedestrian_crossing", "at_a_junction_node")
    if v in net.ring:
        return _skip(out, n["id"], "pedestrian_crossing", "roundabout")
    if v in net.route_k:
        k = net.route_k[v]
        at = float(net.s[k])
        if at < END_CLEAR_M or length - at < END_CLEAR_M:
            return _skip(out, n["id"], "pedestrian_crossing", "route_end")
        with_route, against = net.route_allowed(k, 1), net.route_allowed(k, -1)
        base = {"on": "route", "node": int(n["id"]), "at_m": round(at, 2), "xy": net.xy[v].tolist(),
                "heading": _heading(net.route_tangent(k))}
    else:
        legs = _side_leg(net, v, REACH_M)
        if not legs:
            return _skip(out, n["id"], "pedestrian_crossing", "off_route")
        nb, j, dist, k = legs[0]
        if j is None:
            return _skip(out, n["id"], "pedestrian_crossing", "side_road_not_drawn")
        at = float(net.s[k])
        if at < END_CLEAR_M or length - at < END_CLEAR_M:
            return _skip(out, n["id"], "pedestrian_crossing", "route_end")
        outward = _unit(net.xy[v] - net.xy[nb])  # away from the route
        far = [x for x in net.neighbours(v) if x != nb]
        with_route = net.allowed(v, far[0]) if far else net.allowed(nb, v)  # "with": driving outwards
        against = net.allowed(v, nb)
        base = {"on": "side_road", "node": int(n["id"]), "junction": int(j["id"]), "at_m": round(at, 2),
                "d_m": round(dist, 2), "xy": net.xy[v].tolist(), "heading": _heading(outward)}
    ways = "both" if with_route and against else ("ahead" if with_route else "back")
    if plan["signs"]:
        out["signs"].append({"kind": "pedestrian_crossing", **base, "ways": ways, "from": "crossing"})
    if plan["zebra"]:
        out["zebras"].append({**base, "ways": ways})
    return None


def _one_way_and_no_entry(net: _Net, junctions: Sequence[dict], length: float, out: dict) -> None:
    """ONE WAY across the route from side roads where the route is one-way; NO ENTRY at the mouths of side roads
    traffic may only leave by (module docstring)."""
    seen = set()
    for j in junctions:
        jn = net.idx.get(int(j["node"]))
        if jn is None or jn not in net.route_k:
            continue
        k = net.route_k[jn]
        at = float(net.s[k])
        if at < END_CLEAR_M or length - at < END_CLEAR_M:
            continue
        nb = next((x for (a, x), rec in net.leg_of.items() if a == jn and rec is j), None)
        if nb is None:
            continue
        tags = j.get("tags") or {}
        if float(j.get("off_mainline_deg", 90.0)) < MIN_ANGLE_DEG or is_link(tags) or nb in net.ring or jn in net.ring:
            continue
        inbound, outbound = net.allowed(nb, jn), net.allowed(jn, nb)
        side = junctions_core.SIDE_CODE[j["side"]]
        if inbound and not outbound and not _carries_on(net, jn, nb):
            out["signs"].append({"kind": "no_entry", "on": "side_road", "node": int(j["node"]),
                                 "junction": int(j["id"]),
                                 "at_m": round(at, 2), "d_m": 0.0, "xy": net.xy[jn].tolist(),
                                 "travel": _heading(net.xy[nb] - net.xy[jn]), "from": "oneway"})
        if not inbound or float(j.get("route_turn_deg") or 0.0) >= signals.TURN_DEG:
            continue
        ahead = int(net.gi[k + 1]) if k + 1 < len(net.route_xy) and net.on_node[k + 1] else None
        behind = int(net.gi[k - 1]) if k > 0 and net.on_node[k - 1] else None
        route_segs = [(behind, jn)] if behind is not None else []
        route_segs += [(jn, ahead)] if ahead is not None else []
        if not route_segs or not all(net.one_way(a, b) and _one_way_class(net.tags(a, b)) for a, b in route_segs):
            continue
        if _divided(net, k):
            continue
        far = "R" if side == "L" else "L"
        if (jn, far) in seen:
            continue
        seen.add((jn, far))
        out["signs"].append({"kind": "one_way", "on": "route", "node": int(j["node"]), "junction": int(j["id"]),
                             "at_m": round(at, 2), "xy": net.xy[jn].tolist(), "side": far,
                             "travel": _heading(net.route_tangent(k)), "from": "oneway"})


def _carries_on(net: _Net, jn: int, nb: int) -> bool:
    """Does the one-way road arriving at junction node `jn` from `nb` carry on past it, one-way out of it (the same
    way, or the same name)? Then the junction isn't its exit end: a one-way street crossing the route, or the road the
    route turns onto, which signs ONE WAY, not NO ENTRY."""
    arriving = net.tags(nb, jn)
    j_in = net.seg(nb, jn)
    way_in = int(net.g.seg_way[j_in]) if j_in is not None else None
    for x in net.neighbours(jn):
        if x == nb or not net.allowed(jn, x):
            continue
        j_out = net.seg(jn, x)
        same_way = j_out is not None and int(net.g.seg_way[j_out]) == way_in
        name = arriving.get("name")
        if same_way or (name and net.tags(jn, x).get("name") == name):
            return True
    return False


def _one_way_class(tags: dict) -> bool:
    """A one-way road that gets ONE WAY signs: not a motorway, trunk road, link or roundabout."""
    return (tags.get("highway") not in NO_ONE_WAY_CLASSES and not is_link(tags)
            and tags.get("junction") not in ("roundabout", "circular"))


def _divided(net: _Net, k: int) -> bool:
    """Is the route at its vertex k one carriageway of a divided road: a road of the same name (or ref), one-way the
    other way, within DIVIDED_M?"""
    pair = next(((int(net.gi[i]), int(net.gi[i + 1])) for i in (k, k - 1)
                 if 0 <= i < len(net.route_xy) - 1 and net.on_node[i] and net.on_node[i + 1]), None)
    if pair is None:
        return False
    return divided_at(net.g, net.route_xy[k], net.route_tangent(k), net.tags(*pair))


def divided_at(g, p, travel, tags: dict, reach: float = DIVIDED_M) -> bool:
    """Is a road with these `tags`, driven along `travel` at the point `p`, one carriageway of a divided road: is
    there a road of the same name (or ref) in the graph `g` (`osm_graph.RoadGraph`), one-way the other way, within
    `reach`? Its ONE WAY signs (`_divided`) and its kerbs' width (`divided_carriageway`, `core.kerbs`) both ask."""
    keys = {(key, tags[key]) for key in ("name", "ref") if tags.get(key)}
    if not keys or not len(g.seg_a):
        return False
    p = np.asarray(p, float)
    t = _unit(travel)
    xy = np.asarray(g.xy, float)
    pa, pb = xy[g.seg_a], xy[g.seg_b]
    ab = pb - pa
    tt = np.clip(((p - pa) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0.0, 1.0)
    near = np.flatnonzero(np.hypot(*(pa + ab * tt[:, None] - p).T) <= reach)
    for j in near.tolist():
        wt = g.ways[int(g.seg_way[j])]["tags"] or {}
        if not any(wt.get(key) == value for key, value in keys):
            continue
        fwd, rev = bool(g.seg_fwd[j]), bool(g.seg_rev[j])
        if fwd == rev:
            continue
        drive = _unit(ab[j]) if fwd else -_unit(ab[j])
        if float(drive @ t) < -0.5:
            return True
    return False


DIVIDED_STEP_M = 20.0  # a way is asked about every this far along it (`divided_carriageway`)
DIVIDED_END_M = 15.0  # but not this near its ends, where it meets its junctions


def divided_carriageway(g, xy, tags: dict) -> bool:
    """Is a way with these `tags`, drawn by `xy` in the order it's driven, one carriageway of a divided road along
    most of it: tagged one-way (not a roundabout), and `divided_at` at half or more of the points DIVIDED_STEP_M
    apart along it (DIVIDED_END_M in from each end, where it's long enough)?"""
    if str(tags.get("oneway") or "") not in ("yes", "true", "1", "-1") or tags.get("junction") in (
            "roundabout", "circular"):
        return False
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return False
    s = cumulative(xy)
    length = float(s[-1])
    if length <= 0:
        return False
    end = DIVIDED_END_M if length > 2 * DIVIDED_END_M + DIVIDED_STEP_M else 0.0
    count = max(int((length - 2 * end) // DIVIDED_STEP_M) + 1, 1)
    at = np.linspace(end, length - end, count) if count > 1 else np.array([length / 2])
    votes = 0
    for m in at.tolist():
        i = int(np.clip(np.searchsorted(s, m, side="right") - 1, 0, len(xy) - 2))
        f = (m - s[i]) / max(s[i + 1] - s[i], 1e-9)
        p = xy[i] + (xy[i + 1] - xy[i]) * f
        votes += divided_at(g, p, xy[i + 1] - xy[i], tags)
    return 2 * votes >= len(at)


def _repeaters(net: _Net, length: float, out: dict) -> None:
    """ONE WAY repeaters along one-way blocks longer than ONE_WAY_REPEAT_M (module docstring)."""
    ks = [k for k in range(len(net.route_xy)) if net.route_junction(k)]
    bounds = [0] + ks + [len(net.route_xy) - 1]
    for k0, k1 in zip(bounds, bounds[1:]):
        if k1 - k0 < 1:
            continue
        s0, s1 = float(net.s[k0]), float(net.s[k1])
        if s1 - s0 < ONE_WAY_REPEAT_M:
            continue
        pairs = [(int(net.gi[k]), int(net.gi[k + 1])) for k in range(k0, k1) if net.on_node[k] and net.on_node[k + 1]]
        if not pairs or not all(net.one_way(a, b) and net.allowed(a, b) and _one_way_class(net.tags(a, b))
                                for a, b in pairs):
            continue
        if _divided(net, (k0 + k1) // 2):
            continue
        count = int((s1 - s0) // ONE_WAY_REPEAT_M)
        for i in range(1, count + 1):
            at = s0 + (s1 - s0) * i / (count + 1)
            if at < END_CLEAR_M or length - at < END_CLEAR_M:
                continue
            k = int(np.clip(np.searchsorted(net.s, at), 0, len(net.s) - 1))
            p = net.route_xy[k]
            out["signs"].append({"kind": "one_way", "on": "route", "node": None, "at_m": round(at, 2),
                                 "xy": p.tolist(), "side": "L", "travel": _heading(net.route_tangent(k)),
                                 "from": "oneway", "repeater": True})


def _against_flow(net: _Net, length: float, out: dict) -> None:
    """NO ENTRY where the route drives into a one-way road against its flow (module docstring)."""
    wrong_before = False
    for k in range(len(net.route_xy) - 1):
        wrong = bool(net.on_node[k] and net.on_node[k + 1]) and not net.route_allowed(k, 1)
        if wrong and not wrong_before:
            at = float(net.s[k])
            if END_CLEAR_M <= at <= length - END_CLEAR_M:
                out["signs"].append({"kind": "no_entry", "on": "route", "node": int(net.g.osm_node[net.gi[k]]),
                                     "at_m": round(at, 2), "xy": net.route_xy[k].tolist(),
                                     "travel": _heading(net.route_tangent(k)), "from": "route_against_oneway"})
        wrong_before = wrong


# =====================================================================================================================
# The faces
# =====================================================================================================================

ATLAS_PX = 512
SCALE = 4  # drawn this much larger, then reduced: smooth edges
RED = (196, 22, 32)
WHITE = (246, 246, 242)
BLACK = (14, 14, 14)
BACK_GREY = (148, 150, 148)
PAINT_WHITE = (212, 212, 205)  # the road texture's line paint (`textures`, 225, 225, 218, worn)
CELLS = {  # (x, y, w, h) in atlas pixels; each face cell holds its shape's bounding box
    "galv": (0, 0, 64, 64), "back": (64, 0, 64, 64), "paint": (128, 0, 64, 64), "black": (192, 0, 64, 64),
    "stop": (0, 64, 128, 128), "give_way": (128, 64, 128, 111), "pedestrian_crossing": (256, 64, 128, 128),
    "no_entry": (384, 64, 128, 192), "one_way_R": (0, 192, 384, 128), "one_way_L": (0, 320, 384, 128),
}


def _octagon(w: float, h: float) -> np.ndarray:
    """A regular octagon with flat top and bottom, across its flats `w`, centred: (8, 2)."""
    r = w / 2 / math.cos(math.pi / 8)
    a = np.pi / 8 + np.arange(8) * np.pi / 4
    return np.column_stack([r * np.cos(a), r * np.sin(a) * h / w])


def _triangle(w: float, h: float) -> np.ndarray:
    """Point down, centred on its bounding box."""
    return np.array([[-w / 2, h / 2], [0.0, -h / 2], [w / 2, h / 2]])


def outline(kind: str) -> np.ndarray:
    _, w, h, shape = FACES[kind]
    if shape == "octagon":
        return _octagon(w, h)
    if shape == "triangle":
        return _triangle(w, h)
    return np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])


def _text(d: ImageDraw.ImageDraw, box, text: str, colour) -> None:
    x0, y0, x1, y1 = box
    f = lettering.fit(d, text, x1 - x0, y1 - y0)
    bx0, by0, bx1, by1 = d.textbbox((0, 0), text, font=f)
    d.text(((x0 + x1) / 2 - (bx0 + bx1) / 2, (y0 + y1) / 2 - (by0 + by1) / 2), text, fill=colour, font=f)


def atlas() -> signals.Atlas:
    """Every face, in one texture (module docstring): drawn SCALE times larger and reduced."""
    big = Image.new("RGB", (ATLAS_PX * SCALE, ATLAS_PX * SCALE), (40, 40, 40))
    d = ImageDraw.Draw(big)
    cells = {name: (x / ATLAS_PX, y / ATLAS_PX, (x + w) / ATLAS_PX, (y + h) / ATLAS_PX)
             for name, (x, y, w, h) in CELLS.items()}

    def box(name):
        x, y, w, h = CELLS[name]
        return x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE

    for name, colour in (("galv", (156, 160, 158)), ("back", BACK_GREY), ("paint", PAINT_WHITE), ("black", BLACK)):
        x0, y0, x1, y1 = box(name)
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=colour)

    def poly(name, pts, fill, inset=0.0):
        """`pts` (metres, y up) of a face FACES names, mapped onto its cell, shrunk by `inset` metres."""
        _, w, h, _ = FACES[name]
        x0, y0, x1, y1 = box(FACES[name][0])
        pts = np.asarray(pts, float)
        if inset:
            c = pts.mean(axis=0)
            poly_m = shapely.Polygon(pts).buffer(-inset, join_style="mitre")
            pts = np.asarray(poly_m.exterior.coords)[:-1] if not poly_m.is_empty else c[None]
        px = [(x0 + (x + w / 2) / w * (x1 - x0), y0 + (h / 2 - y) / h * (y1 - y0)) for x, y in pts]
        d.polygon(px, fill=fill)

    # STOP: a red octagon with a white border and STOP in white.
    x0, y0, x1, y1 = box("stop")
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=WHITE)
    poly("stop", outline("stop"), WHITE)
    poly("stop", outline("stop"), RED, inset=0.025)
    w, h = x1 - x0, y1 - y0
    _text(d, (x0 + 0.17 * w, y0 + 0.34 * h, x1 - 0.17 * w, y0 + 0.66 * h), "STOP", WHITE)
    # GIVE WAY: a triangle, point down, a red border and a white middle, GIVE WAY in black.
    x0, y0, x1, y1 = box("give_way")
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=RED)
    poly("give_way", outline("give_way"), RED)
    poly("give_way", outline("give_way"), WHITE, inset=0.075)
    w, h = x1 - x0, y1 - y0
    _text(d, (x0 + 0.33 * w, y0 + 0.17 * h, x1 - 0.33 * w, y0 + 0.30 * h), "GIVE", BLACK)
    _text(d, (x0 + 0.36 * w, y0 + 0.34 * h, x1 - 0.36 * w, y0 + 0.47 * h), "WAY", BLACK)
    # NO ENTRY: white, a red disc with a white bar across it, NO ENTRY in black below.
    x0, y0, x1, y1 = box("no_entry")
    w, h = x1 - x0, y1 - y0
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=BLACK)
    m = int(0.02 * w)
    d.rectangle([x0 + m, y0 + m, x1 - 1 - m, y1 - 1 - m], fill=WHITE)
    r = 0.36 * w
    cx, cy = (x0 + x1) / 2, y0 + 0.07 * h + r
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED)
    d.rectangle([cx - 0.72 * r, cy - 0.17 * r, cx + 0.72 * r, cy + 0.17 * r], fill=WHITE)
    _text(d, (x0 + 0.18 * w, y0 + 0.66 * h, x1 - 0.18 * w, y0 + 0.76 * h), "NO", BLACK)
    _text(d, (x0 + 0.1 * w, y0 + 0.81 * h, x1 - 0.1 * w, y0 + 0.91 * h), "ENTRY", BLACK)
    # PEDESTRIAN CROSSING: white with a black border, our own walking figure over black stripes.
    x0, y0, x1, y1 = box("pedestrian_crossing")
    w, h = x1 - x0, y1 - y0
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=BLACK)
    m = int(0.03 * w)
    d.rectangle([x0 + m, y0 + m, x1 - 1 - m, y1 - 1 - m], fill=WHITE)
    for i in range(5):
        sx = x0 + 0.14 * w + i * 0.16 * w
        d.polygon([(sx, y0 + 0.88 * h), (sx + 0.09 * w, y0 + 0.88 * h), (sx + 0.12 * w, y0 + 0.76 * h),
                   (sx + 0.03 * w, y0 + 0.76 * h)], fill=BLACK)
    signals._figure(d, (x0 + 0.28 * w, y0 + 0.08 * h, x1 - 0.28 * w, y0 + 0.74 * h), True, BLACK)
    # ONE WAY: black, a white arrow with ONE WAY in black along its shaft; one each way.
    for name, right in (("one_way_R", True), ("one_way_L", False)):
        x0, y0, x1, y1 = box(name)
        w, h = x1 - x0, y1 - y0
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=BLACK)
        shaft = (x0 + 0.04 * w, y0 + 0.24 * h, x0 + 0.78 * w, y0 + 0.76 * h)
        head = [(x0 + 0.74 * w, y0 + 0.08 * h), (x0 + 0.97 * w, y0 + 0.5 * h), (x0 + 0.74 * w, y0 + 0.92 * h)]
        if not right:
            shaft = (x1 - (shaft[2] - x0), shaft[1], x1 - (shaft[0] - x0), shaft[3])
            head = [(x1 - (px - x0), py) for px, py in head]
        d.rectangle(shaft, fill=WHITE)
        d.polygon(head, fill=WHITE)
        inner = (shaft[0] + 0.03 * w, shaft[1] + 0.06 * h, shaft[2] - 0.03 * w, shaft[3] - 0.06 * h)
        _text(d, inner, "ONE WAY", BLACK)
    img = big.resize((ATLAS_PX, ATLAS_PX), Image.Resampling.LANCZOS)
    return signals.Atlas(img, cells, {})


# =====================================================================================================================
# Placing and drawing
# =====================================================================================================================

@dataclass
class _Post:
    e: float
    n: float
    h: float  # the ground under it
    own: bool  # ours, or a signal, mast or blade post (`signals.Built.posts`)
    signs: list = field(default_factory=list)  # (bottom, top, facing (2D)) above the ground

    def slot(self, height: float, facing: np.ndarray) -> float | None:
        """Where a sign `height` tall facing `facing` goes on this post (its bottom, above the ground), or None."""
        cap = OWN_POST_TOP_M if self.own else SIGNAL_POST_TOP_M
        bottom = SIGN_BOTTOM_M
        for b0, b1, f in sorted(self.signs, key=lambda t: t[0]):
            overlaps = bottom < b1 + STACK_GAP_M and bottom + height > b0 - STACK_GAP_M
            if overlaps and float(np.dot(_unit(f), _unit(facing))) > -0.9:
                bottom = b1 + STACK_GAP_M
        return bottom if bottom + height <= cap else None


class _Posts:
    def __init__(self, existing: Sequence[tuple[float, float, float]]):
        self.posts = [_Post(float(e), float(n), float(h), False) for e, n, h in existing]

    def near(self, e: float, n: float) -> _Post | None:
        best, dist = None, SNAP_M
        for p in self.posts:
            d = math.hypot(p.e - e, p.n - n)
            if d <= dist:
                best, dist = p, d
        return best

    def mount(self, e: float, n: float, h: float, height: float, facing) -> tuple[_Post, float] | None:
        """A post for a sign at (e, n): the one within SNAP_M if it has room, else a new one; None where the near one
        has no room."""
        p = self.near(e, n)
        if p is None:
            p = _Post(e, n, h, True)
            bottom = p.slot(height, facing)
            if bottom is None:
                return None
            self.posts.append(p)
        else:
            bottom = p.slot(height, facing)
            if bottom is None:
                return None
        p.signs.append((bottom, bottom + height, np.asarray(facing, float)))
        return p, bottom


@dataclass
class Built:
    meshes: list[Mesh]
    atlas: signals.Atlas | None
    stats: dict
    records: list = field(default_factory=list)  # each sign and zebra as drawn, or why not (`04_mesh/signs.json`)


def _draw_sign(b: signals._Builder, p: _Post, bottom: float, kind: str, facing, arrow_right: bool = True) -> None:
    """A sign's plate on its post: the face, lit at night, and a plain back."""
    cell, w, h, _ = FACES[kind]
    if kind == "one_way":
        cell = "one_way_R" if arrow_right else "one_way_L"
    f = np.array([*_unit(facing), 0.0])
    r = signals._right(facing)
    up = np.array([0.0, 0.0, 1.0])
    post_r = (POST_D if p.own else max(signals.POST_D, signals.MAST_D)) / 2  # a mast's column is the widest
    c = np.array([p.e, p.n, p.h + bottom + h / 2]) + f * (post_r + 0.01 + PLATE_M / 2)
    shape = outline(kind)
    b.polygon(c + f * PLATE_M / 2, r, up, shape, f, cell, RETRO_MATERIAL)
    b.polygon(c - f * PLATE_M / 2, -r, up, shape * np.array([-1.0, 1.0]), -f, "back", MATERIAL)
    # A bracket from the post to the back of the plate.
    b.beam(np.array([p.e, p.n, c[2]]), c - f * PLATE_M, 0.03, cell="galv")


def _arrow_right(facing, flow) -> bool:
    """Does the arrow point right for someone reading a face that points along `facing`, the traffic flowing along
    `flow`? They look along -facing; their right is that turned clockwise."""
    look = -_unit(facing)
    right = np.array([look[1], -look[0]])
    return float(right @ _unit(flow)) > 0


def build(found: dict, road: signals._Road, frame: LocalFrame, *, mouths: dict | None = None,
          all_mouths: Sequence = (), keep_out: Sequence = (), blocked: dict | None = None,
          posts: Sequence[tuple[float, float, float]] = (), height: Callable | None = None,
          road_height: Callable | None = None, station_of: Callable | None = None) -> Built:
    """The signs and zebras of `found` (signs.json) as meshes in AC axes, their atlas, a summary and a record of each.
    `mouths`: the junctions' mouths (`core.junctions.Mouth`) by junction id, only those whose side road has a stub
    (`core.stubs`); `all_mouths`, every junction's, for the corners (`signals.junction_spans`); `keep_out`: geometry a
    sign beside a side road keeps out of (every side road's and driveway's carriageway and stub, `SIDE_CLEAR_M`
    wider); `blocked`: per side, the stations no sign on the route's verge stands at (the furniture's); `posts`: the
    signal, mast and blade posts; `height(e, n)`: the ground; `road_height(e, n)`: the road's surface, for the zebras;
    `station_of(rec)`: a record's station on the centreline."""
    mouths = mouths or {}
    stats = {"signs": 0, "by_kind": {}, "left_out": 0, "left_out_by_reason": {}, "posts": 0,
             "on_signal_posts": 0, "zebras": 0, "zebra_stripes": 0, "zebras_left_out": 0}
    records: list = []
    signs = found.get("signs") or []
    zebras = found.get("zebras") or []
    if not signs and not zebras:
        return Built([], None, stats, records)
    tex = atlas()
    b = signals._Builder(tex, (MATERIAL, RETRO_MATERIAL, PAINT_MATERIAL))
    registry = _Posts(posts)
    spans = signals.junction_spans(all_mouths)
    keep = shapely.unary_union(list(keep_out)) if keep_out else None
    if keep is not None:
        shapely.prepare(keep)
    half_max = np.maximum(road.half["L"], road.half["R"])
    tree = cKDTree(road.xy)

    def station(rec) -> float:
        return station_of(rec) if station_of is not None else float(rec["at_m"])

    def left_out(rec, why):
        stats["left_out"] += 1
        stats["left_out_by_reason"][why] = stats["left_out_by_reason"].get(why, 0) + 1
        records.append({**_ref(rec), "left_out": why})

    def drawn(rec, kind, p: _Post, extra=None):
        stats["signs"] += 1
        stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1
        records.append({**_ref(rec), "e": round(p.e, 2), "n": round(p.n, 2),
                        "post": "own" if p.own else "signal_or_blade_post", **(extra or {})})

    # -- on the route's verge ------------------------------------------------------------------------------------
    def verge_spot(st: float, side: str, end: str, kind: str, facings: list) -> tuple[_Post, float] | None:
        """A post on the verge at station `st` or moved towards `end` ("before" or "after"), clear of the mouths, with
        room for the sign facing each of `facings`; and the sign's bottom."""
        h_sign = FACES[kind][2]
        i = signals.clear_station(road, st, side, end, None)
        if i is None:
            return None
        step = -1 if end == "before" else 1
        while 0 <= i < len(road.s) and abs(road.s[i] - st) <= MAX_SHIFT_M:
            if blocked is None or not blocked[side][i]:
                e, n, h = road.place(i, side, SIGN_OFFSET_M)
                p = registry.near(e, n)
                trial = p or _Post(e, n, h, True)
                if all(trial.slot(h_sign, f) is not None for f in facings):
                    got = registry.mount(e, n, h, h_sign, facings[0])
                    if got is not None:
                        for f in facings[1:]:
                            got[0].signs.append((got[1], got[1] + h_sign, np.asarray(f, float)))
                        return got
            i += step
        return None

    def put(rec, kind, got, facings, arrows=None):
        if got is None:
            return left_out(rec, "no_clear_place")
        p, bottom = got
        for k, f in enumerate(facings):
            _draw_sign(b, p, bottom, kind, f, True if arrows is None else arrows[k])
        if not p.own:
            stats["on_signal_posts"] += 1
        drawn(rec, kind, p, {"faces": len(facings)})
        return None

    for rec in signs:
        kind = rec["kind"]
        if rec["on"] == "route":
            st = station(rec)
            i = road.index(st)
            tangent = road.tangent(i)
            if kind in ("give_way", "stop", "no_entry"):
                travel = np.array([math.cos(rec["travel"]), math.sin(rec["travel"])])
                with_route = float(travel @ tangent) >= 0
                sides = ("L", "R") if kind == "no_entry" else (("L",) if with_route else ("R",))
                facing = -tangent if with_route else tangent
                for side in sides:
                    end = "before" if with_route else "after"
                    put(rec, kind, verge_spot(st, side, end, kind, [facing]), [facing])
            elif kind == "pedestrian_crossing":
                ways = rec.get("ways", "both")
                facings = ([-tangent] if ways in ("both", "ahead") else []) + ([tangent] if ways in ("both", "back")
                                                                                 else [])
                for side, end in (("L", "before"), ("R", "after")):
                    put(rec, kind, verge_spot(st, side, end, kind, facings), facings)
            elif kind == "one_way":
                side = rec["side"]
                towards = road.left(i) * (1.0 if side == "R" else -1.0)  # face back across the road
                if rec.get("repeater"):
                    facings = [towards, -towards]
                    got = verge_spot(st, side, "after", kind, facings)
                    put(rec, kind, got, facings, [_arrow_right(f, tangent) for f in facings])
                    continue
                if rec.get("junction") not in mouths:  # its side road isn't drawn: nobody to face
                    left_out(rec, "side_road_without_stub")
                    continue
                got = None
                if blocked is None or not blocked[side][i]:  # a T: straight across from the side road
                    got = verge_spot(st, side, "after", kind, [towards])
                if got is None:  # a crossroads: the far side has a mouth too; the corner downstream of it
                    corner = signals.corner_stations(st, spans.get(int(rec["node"]), {}), signals.JUNCTION_HALF_M)
                    got = verge_spot(corner[(side, "after")], side, "after", kind, [towards])
                put(rec, kind, got, [towards], [_arrow_right(towards, tangent)])
            continue
        # -- beside a side road ----------------------------------------------------------------------------------
        m = mouths.get(rec.get("junction"))
        if m is None:
            left_out(rec, "side_road_without_stub")
            continue
        line = np.asarray(m.line, float)
        if kind in ("give_way", "stop"):
            places = [(-1.0, [1.0])]  # left of inward traffic: right of the outward line; faces outwards
        elif kind == "no_entry":
            places = [(1.0, [-1.0]), (-1.0, [-1.0])]  # either side, facing the route
        else:  # a crossing: both sides, facing the traffic it has
            ways = rec.get("ways", "both")
            outs = ([-1.0] if ways in ("both", "ahead") else []) + ([1.0] if ways in ("both", "back") else [])
            places = [(1.0, outs), (-1.0, outs)]
        for lateral, faces in places:
            got = _beside(line, float(rec.get("d_m") or 0.0), m.half_m, lateral, kind, faces, registry, keep, tree,
                          half_max, height)
            if got is None:
                left_out(rec, "no_clear_place")
                continue
            p, bottom, tangent = got
            facings = [tangent * sgn for sgn in faces]
            for f in facings:
                _draw_sign(b, p, bottom, kind, f)
            if not p.own:
                stats["on_signal_posts"] += 1
            drawn(rec, kind, p, {"faces": len(facings)})

    # -- the posts --------------------------------------------------------------------------------------------------
    for p in registry.posts:
        if not p.own or not p.signs:
            continue
        top = max(t for _, t, _ in p.signs) + 0.05
        b.prism(p.e, p.n, p.h - 0.3, p.h + top, POST_D)
        b.collider(p.e, p.n, p.h - signals.COLLIDER_BELOW_M, p.h + signals.COLLIDER_TOP_M, signals.COLLIDER_W)
        stats["posts"] += 1

    # -- the zebras -------------------------------------------------------------------------------------------------
    for rec in zebras:
        if rec["on"] == "route":
            st = station(rec)
            i = road.index(st)
            centre = road.xy[i] + road.left(i) * (road.half["L"][i] - road.half["R"][i]) / 2
            along = road.tangent(i)
            width = float(road.half["L"][i] + road.half["R"][i])
            expect = float(road.z[i])
        else:
            m = mouths.get(rec.get("junction"))
            if m is None:
                stats["zebras_left_out"] += 1
                records.append({**_ref(rec, "zebra"), "left_out": "side_road_without_stub"})
                continue
            line = np.asarray(m.line, float)
            centre, along = _at(line, float(rec.get("d_m") or 0.0))
            width = 2 * m.half_m
            expect = None
        stripes = _zebra(b, centre, along, width, road_height, expect)
        if stripes:
            stats["zebras"] += 1
            stats["zebra_stripes"] += stripes
            records.append({**_ref(rec, "zebra"), "stripes": stripes})
        else:
            stats["zebras_left_out"] += 1
            records.append({**_ref(rec, "zebra"), "left_out": "no_road_under_it"})
    meshes = b.meshes(frame, prefix=MESH_PREFIX)
    return Built(meshes, tex if meshes else None, stats, records)


def _ref(rec: dict, kind: str | None = None) -> dict:
    out = {"kind": kind or rec["kind"], "on": rec["on"]}
    for key in ("node", "junction", "at_m", "repeater"):
        if rec.get(key) is not None:
            out[key] = rec[key]
    return out


def _at(line: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    """The point `d` metres along a polyline, and its direction there."""
    s = cumulative(line)
    d = float(np.clip(d, 0.0, s[-1]))
    i = int(np.clip(np.searchsorted(s, d, side="right") - 1, 0, len(line) - 2))
    t = _unit(line[i + 1] - line[i])
    return line[i] + t * (d - s[i]), t


def _beside(line, d0, half, lateral, kind, faces, registry: _Posts, keep, tree, half_max, height):
    """A post beside a side road (module docstring): `lateral` +1 on the left of its outward line, -1 on the right,
    from `d0` metres out, moving out by at most MAX_SHIFT_M; with room for the sign facing each of `faces` (+1
    outwards, -1 back towards the route). Returns (post, bottom, outward direction there) or None."""
    h_sign = FACES[kind][2]
    reach = float(cumulative(line)[-1])
    d = max(d0, 0.0)
    while d <= min(d0 + MAX_SHIFT_M, reach):
        p, t = _at(line, d)
        across = np.array([-t[1], t[0]]) * lateral
        # The nearest place out from the side road's line that's clear of its paved outline (a stub can be wider than
        # its mapped width), by at most SIDE_WIDER_M more.
        q = None
        for x in np.arange(half + SIDE_OFFSET_M, half + SIDE_OFFSET_M + SIDE_WIDER_M + 1e-9, STEP_M / 2):
            c = p + across * x
            if keep is None or not shapely.contains_xy(keep, c[0], c[1]):
                q = c
                break
        ok = q is not None
        if ok:
            gap, i = tree.query(q)
            ok = gap >= half_max[i] + SIGN_OFFSET_M + 0.5
        if ok:
            h = float(height(np.array([q[0]]), np.array([q[1]]))[0]) if height is not None else 0.0
            ok = math.isfinite(h)
        if ok:
            facings = [t * sgn for sgn in faces]
            post = registry.near(q[0], q[1])
            trial = post or _Post(float(q[0]), float(q[1]), h, True)
            if all(trial.slot(h_sign, f) is not None for f in facings):
                got = registry.mount(float(q[0]), float(q[1]), h, h_sign, facings[0])
                if got is not None:
                    for f in facings[1:]:
                        got[0].signs.append((got[1], got[1] + h_sign, np.asarray(f, float)))
                    return got[0], got[1], t
        d += STEP_M
    return None


def _zebra(b: signals._Builder, centre, along, width: float, road_height, expect: float | None) -> int:
    """A zebra's stripes across a carriageway `width` wide centred on `centre`, along `along`: each a grid following
    the road's surface ZEBRA_LIFT_M above it. Returns how many were drawn (a stripe with no road under every vertex,
    or under a deck far above the road, isn't)."""
    if road_height is None:
        return 0
    along = _unit(along)
    across = np.array([-along[1], along[0]])
    usable = width - 2 * ZEBRA_EDGE_M
    count = int((usable + ZEBRA_GAP_M) // (ZEBRA_STRIPE_M + ZEBRA_GAP_M))
    if count < 1:
        return 0
    total = count * ZEBRA_STRIPE_M + (count - 1) * ZEBRA_GAP_M
    rows = int(math.ceil(ZEBRA_LENGTH_M / ZEBRA_ROW_M)) + 1
    a_vals = np.linspace(-ZEBRA_LENGTH_M / 2, ZEBRA_LENGTH_M / 2, rows)
    drawn = 0
    pos, nrm, uv, tri = b.parts[PAINT_MATERIAL]
    u0, v0, u1, v1 = b.tex.cells["paint"]
    for k in range(count):
        x0 = -total / 2 + k * (ZEBRA_STRIPE_M + ZEBRA_GAP_M)
        x_vals = np.linspace(x0, x0 + ZEBRA_STRIPE_M, 3)
        aa, xx = np.meshgrid(a_vals, x_vals, indexing="ij")
        e = centre[0] + along[0] * aa + across[0] * xx
        n = centre[1] + along[1] * aa + across[1] * xx
        z = np.asarray(road_height(e, n), float)
        if not np.isfinite(z).all():
            continue
        if expect is not None and float(np.abs(z - expect).max()) > ZEBRA_MAX_DROP_M:
            continue
        i0 = len(pos)
        for r in range(rows):
            for c in range(3):
                pos.append(np.array([e[r, c], n[r, c], z[r, c] + ZEBRA_LIFT_M]))
                nrm.append(np.array([0.0, 0.0, 1.0]))
                uv.append((u0 + (u1 - u0) * (0.25 + 0.25 * c), v0 + (v1 - v0) * (0.25 + 0.5 * r / (rows - 1))))
        for r in range(rows - 1):
            for c in range(2):
                a = i0 + r * 3 + c
                tri.extend([(a, a + 1, a + 4), (a, a + 4, a + 3)])
        drawn += 1
    return drawn
