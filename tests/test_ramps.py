"""Slip ramps (`core.ramps`): which links count (under 15 degrees, the ones the junction finder can't see, diverge,
merge or two-way), the plan (the taper, the split, the nose, the island and the 60 m stub; the paint and its spacing;
what's left out and why), the meshes (the throat joined to the road's edge and the ramp's first row without a crack
or an overlap, the paint cut into it, the road's verge cut and its edge line broken over the taper only, the terrain
fitted round it with no hole), the closures on it, validation, the masks and gates, and the section stage (the road's
half widths, and so the AI line, unchanged)."""

import math

import numpy as np
import pytest
import shapely
from pyproj import Transformer

from mapcreator import config
from mapcreator.core import ai_line, junctions, mesh, police_car, ramps, section, stub_closures, stubs, validate
from mapcreator.core.layout import RoadModel
from mapcreator.core.scene import LocalFrame
from mapcreator.job import Job
from mapcreator.pipeline import route_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.osm import graph

IDENT = Transformer.from_crs(4326, 4326, always_xy=True)  # the test networks are drawn in metres
FRAME = LocalFrame(0.0, 0.0, 0.0)
HALF = 3.2
GROUND = 10.0
LINK = {"highway": "trunk_link", "oneway": "yes"}


# -- which ramps --------------------------------------------------------------------------------------------------------

def network(ways: list[tuple[int, list, dict]], main: dict | None = None):
    """A trunk road east along y = 0, 0 to 1,000 m with a node every 100 m (ids 0-10), one-way east, and more ways,
    each from the main road's node at x through its points (x, y). Way ids: main 1, the others 100, 101, ..."""
    nodes = {i: (100.0 * i, 0.0) for i in range(11)}
    els = [(1, list(range(11)), main or {"highway": "trunk", "oneway": "yes", "maxspeed": "80"})]
    nid = 1000
    for k, (x, points, tags) in enumerate(ways):
        refs = [x // 100]
        for p in points:
            nodes[nid] = p
            refs.append(nid)
            nid += 1
        els.append((100 + k, refs, tags))
    data = [{"type": "node", "id": i, "lon": x, "lat": y} for i, (x, y) in nodes.items()]
    data += [{"type": "way", "id": w, "nodes": refs, "tags": tags} for w, refs, tags in els]
    return graph.build({"elements": data}, IDENT)


def out_at(x, deg, length=300.0, back=False):
    """Points along a straight line leaving (x, 0) at `deg` (+ left), ahead of it, or behind it with `back`."""
    a = math.radians(deg)
    sgn = -1.0 if back else 1.0
    return [(x + sgn * d * math.cos(a), d * math.sin(a)) for d in np.linspace(length / 10, length, 10)]


def route_of(g, start=10.0, end=990.0):
    return graph.route(g, np.array([[start, 0.0], [end, 0.0]])).xy


def test_a_slip_ramp_the_junction_finder_cant_see_is_found_with_its_kind_and_the_road_beside_it():
    g = network([(300, out_at(300, 6.0), LINK)])
    xy = route_of(g)
    # The junction finder takes a link this shallow for the route itself: it doesn't list it at all.
    found = junctions.find(g, xy)
    assert found["junctions"] == [] and found["skipped"] == []
    res = ramps.find(g, xy)
    (r,) = res["ramps"]
    assert (r["id"], r["at_m"], r["side"], r["kind"], r["along"], r["hidden"]) == (
        ramps.RAMP_ID_BASE, 290.0, "left", "diverge", "ahead", True)
    assert r["off_mainline_deg"] == pytest.approx(6.0, abs=0.5) and r["closure"] == "major"
    assert r["main"] == {"way_id": 1, "highway": "trunk", "maxspeed": "80", "oneway": "yes", "lane_markings": None}
    assert r["length_m"] == pytest.approx(ramps.FOLLOW_M) and r["width_m"] == pytest.approx(4.0)
    assert res["skipped"] == [] and res["rule"]["max_angle_deg"] == 15.0


def test_a_ramp_at_10_degrees_is_the_junction_finder_s_ramp_skip_and_one_at_30_is_its_side_road():
    g = network([(300, out_at(300, -10.0), LINK), (600, out_at(600, 30.0), LINK)])
    xy = route_of(g)
    found = junctions.find(g, xy)
    assert [(s["at_m"], s["reason"]) for s in found["skipped"]] == [(290.0, "ramp")]
    assert [j["at_m"] for j in found["junctions"]] == [590.0]  # 30 degrees: a side road, stubbed as before
    (r,) = ramps.find(g, xy)["ramps"]
    assert (r["at_m"], r["side"], r["hidden"]) == (290.0, "right", False)


def test_merges_two_way_ramps_and_what_isnt_a_ramp():
    g = network([(300, out_at(300, 8.0, back=True), {"highway": "trunk_link"}),
                 (500, out_at(500, -7.0), {"highway": "primary_link", "access": "private"}),
                 (700, out_at(700, 8.0), {"highway": "residential"})])  # a shallow fork, not a link: not now
    res = ramps.find(g, route_of(g))
    assert [(r["at_m"], r["kind"], r["along"]) for r in res["ramps"]] == [(290.0, "two_way", "behind")]
    assert [(s["at_m"], s["reason"]) for s in res["skipped"]] == [(490.0, "private")]
    # A one-way link onto the road is a merge: `oneway=-1` runs it against its nodes, towards the road.
    g = network([(300, out_at(300, 8.0, back=True), {"highway": "trunk_link", "oneway": "-1"})])
    (r,) = ramps.find(g, route_of(g))["ramps"]
    assert (r["kind"], r["along"]) == ("merge", "behind")


def test_a_ramp_at_the_route_s_end_is_left_to_its_end_wall():
    g = network([(100, out_at(100, 6.0), LINK)])
    res = ramps.find(g, route_of(g, start=95.0))
    assert res["ramps"] == [] and [s["reason"] for s in res["skipped"]] == ["route_end"]


# -- the plan -----------------------------------------------------------------------------------------------------------

def road(n=1001, z=GROUND, unsealed=False):
    """A straight road east along y = 0, a station every metre, level at `z`, 6.4 m wide with a crown."""
    s = np.arange(n, dtype=float)
    c = {"s": s, "xy": np.column_stack([s, np.zeros(n)]), "heading": np.zeros(n), "curvature": np.zeros(n),
         "z": np.full(n, z, float), "grade": np.zeros(n)}
    sec = {"half_left": np.full(n, HALF), "half_right": np.full(n, HALF), "slope_left": np.full(n, section.CROWN),
           "slope_right": np.full(n, section.CROWN), "style": np.full(n, 2 if unsealed else 0, np.int8)}
    return c, sec


def ramp(x, deg, *, kind="diverge", rid=ramps.RAMP_ID_BASE, length=300.0, main=None, tags=None, closure="major"):
    """A ramps.json ramp leaving the road's node at x at `deg` (+ left): ahead, or behind for a merge."""
    back = kind == "merge"
    a = math.radians(deg)
    sgn = -1.0 if back else 1.0
    line = [[x + sgn * d * math.cos(a), d * math.sin(a)] for d in np.linspace(0.0, length, 31)]
    return {"id": rid, "node": 7, "way_id": 99, "at_m": x, "xy": [x, 0.0], "side": "left" if deg > 0 else "right",
            "angle_deg": deg, "off_mainline_deg": abs(deg), "route_turn_deg": 0.0, "kind": kind,
            "along": "behind" if back else "ahead", "hidden": abs(deg) < 7,
            "main": main or {"highway": "trunk", "maxspeed": "80"}, "line": line, "length_m": length,
            "ends": "length", "highway": "trunk_link", "closure": closure, "name": None,
            "tags": {"highway": "trunk_link", "oneway": "yes", **(tags or {})}, "width_m": 4.0}


def flat(h=GROUND):
    return lambda x, y: np.full(np.shape(x), h, float)


def plan_one(r, c=None, sec=None, dem=None, **kw):
    if c is None:
        c, sec = road()
    p = ramps.plan([r], c, sec, dem or flat(), **kw)
    return p, (p["ramps"][0] if p["ramps"] else None)


@pytest.mark.parametrize("deg", [4.0, 6.0, 10.0, 14.0])
def test_the_taper_split_nose_and_island_fall_where_the_ramp_s_edges_leave_the_road_s(deg):
    p, r = plan_one(ramp(300.0, deg))
    w, t = 2.0, math.tan(math.radians(deg))
    taper, split, nose = r["taper"]["from_m"], r["taper"]["split_m"], r["taper"]["nose_m"]
    # A straight ramp from the road's centreline: its outer edge leaves the road's edge where it's (HALF - w) out,
    # its inner edge where it's (HALF + w) out, and the nose is where the gap is NOSE_W_M.
    assert taper == pytest.approx(300 + (HALF - w) / t, abs=2.0)
    assert split == pytest.approx(300 + (HALF + w) / t, abs=2.0)
    assert nose == pytest.approx(300 + (HALF + w + ramps.NOSE_W_M) / t, abs=2.5)
    assert r["island"]["to_m"] == pytest.approx(300 + (HALF + w + ramps.ISLAND_END_W_M) / t, abs=3.0)
    assert r["mouth"]["from_m"] == pytest.approx(taper, abs=1.0)
    assert r["mouth"]["to_m"] == pytest.approx(r["island"]["to_m"], abs=1.0)
    # The ramp's own ribbon starts at the nose and runs STUB_M past it; ramps.json keeps its rows from LEAD_M before.
    assert r["length_m"] - r["ribbon_from_m"] == pytest.approx(ramps.STUB_M)
    nose_d = r["rows_from_m"] + r["ribbon_from_m"]  # along the ramp from its node
    assert nose_d == pytest.approx((nose - 300) / math.cos(math.radians(deg)), abs=3.0)
    assert r["ribbon_from_m"] == min(ramps.LEAD_M, nose_d)
    assert (r["kind"], r["along"], r["side"], r["closure"], r["style"]) == ("diverge", "ahead", "left", "major", "road")
    assert r["taper"]["outer_side"] == "left" and p["skipped"] == []
    # The road's edge line breaks over the taper only: to the split, and solid beside the gore.
    br0, br1 = r["taper"]["edge_break_rows"]
    assert taper <= br0 < br1 == r["taper"]["rows"][1]
    # Every point of the throat's outer edge is beyond the road's edge, from the taper's start to the nose.
    outer = np.asarray(r["outer"])
    assert outer[0, 1] == pytest.approx(HALF) and (outer[:, 1] >= HALF - 1e-6).all()


def test_a_merge_runs_behind_the_node_and_a_right_hand_ramp_on_the_right():
    _, r = plan_one(ramp(500.0, 8.0, kind="merge"))
    assert r["along"] == "behind" and r["kind"] == "merge"
    assert r["taper"]["from_m"] > r["taper"]["split_m"] > r["taper"]["nose_m"] > r["island"]["to_m"]
    assert r["mouth"]["rows"] == [round(r["island"]["to_m"]), round(r["taper"]["from_m"])]
    _, right = plan_one(ramp(500.0, -8.0))
    assert right["side"] == "right" and right["taper"]["outer_side"] == "right"
    assert (np.asarray(right["outer"])[:, 1] <= -HALF + 1e-6).all()


def test_a_ramp_wider_than_the_road_s_half_is_tapered_in_at_1_in_15():
    c, sec = road()
    sec = {**sec, "half_left": np.full(len(c["s"]), 1.8), "half_right": np.full(len(c["s"]), 1.8)}
    _, r = plan_one(ramp(300.0, 8.0), c, sec)
    outer = np.asarray(r["outer"])
    # The ramp's outer edge is 0.2 m off the road where it starts: the taper runs 3 m back from the node to the edge.
    assert r["taper"]["from_m"] == pytest.approx(300 - 0.2 * ramps.TAPER_RATIO, abs=1.0)
    assert outer[0, 1] == pytest.approx(1.8) and (np.diff(outer[:, 0]) > 0).all()


@pytest.mark.parametrize("main,spacing", [({"highway": "trunk", "maxspeed": "60"}, 6.0),
                                          ({"highway": "trunk", "maxspeed": "80"}, 12.0),
                                          ({"highway": "primary", "maxspeed": "100"}, 12.0),
                                          ({"highway": "motorway"}, 24.0),
                                          ({"highway": "trunk", "maxspeed": "110"}, 24.0)])
def test_the_gore_s_bars_are_spaced_by_the_road_s_speed(main, spacing):
    assert ramps.spacing_m(main) == spacing
    _, r = plan_one(ramp(300.0, 6.0, main=main))
    assert r["gore"]["spacing_m"] == spacing


def test_chevrons_in_a_diverge_s_gore_point_at_approaching_traffic_and_stay_inside_it():
    _, r = plan_one(ramp(300.0, 5.0, main={"highway": "trunk", "maxspeed": "60"}))
    assert r["gore"]["paint"] == "chevrons"
    throat_paved = shapely.Polygon(r["outline"]).buffer(0)
    polys = [shapely.Polygon(np.asarray(p["exterior"])[:, :2]) for p in r["gore"]["polygons"]]
    assert all(p.is_valid for p in polys) and all(throat_paved.buffer(-ramps.PAINT_CLEAR_M + 1e-3).contains(p)
                                                  for p in polys)
    # Paint keeps off the road's own carriageway and its edge.
    assert all(p.bounds[1] > HALF for p in polys)
    # The chevrons: between the gore's two edge lines (the longest pieces), each a V whose point is towards the split.
    lines = sorted(polys, key=lambda p: -p.length)[:2]
    bars = [p for p in polys if not any(p.equals(q) for q in lines)]
    split, nose = r["taper"]["split_m"], r["taper"]["nose_m"]
    assert len(bars) >= int((nose - split) / 6.0) - 3
    for bar in bars:
        xs = np.asarray(bar.exterior.coords)
        tip = xs[np.argmin(xs[:, 0])]  # the point: its least x, upstream
        assert tip[1] == pytest.approx((HALF + xs[:, 1].max()) / 2, abs=1.2)
    # The first bar is where the gore is about a metre wide.
    first = min(bars, key=lambda p: p.bounds[0])
    assert first.bounds[0] == pytest.approx(split + 1.0 / math.tan(math.radians(5.0)), abs=4.0)


def test_a_two_way_ramp_gets_diagonals_and_a_merge_s_chevrons_point_the_other_way():
    _, two = plan_one(ramp(300.0, 6.0, kind="two_way"))
    assert two["gore"]["paint"] == "diagonals"
    _, merge = plan_one(ramp(600.0, 6.0, kind="merge", main={"highway": "trunk", "maxspeed": "60"}))
    polys = sorted((shapely.Polygon(np.asarray(p["exterior"])[:, :2]) for p in merge["gore"]["polygons"]),
                   key=lambda p: -p.length)[2:]
    for bar in polys:  # traffic runs from the nose (west) to the split (east): the point faces west, towards the nose
        xs = np.asarray(bar.exterior.coords)
        tip = xs[np.argmin(xs[:, 0])]
        assert tip[1] == pytest.approx((HALF + xs[:, 1].max()) / 2, abs=1.2)


def test_no_paint_on_gravel_or_an_unmarked_road():
    c, sec = road(unsealed=True)
    _, gravel = plan_one(ramp(300.0, 6.0), c, sec)
    assert gravel["gore"] == {"paint": "none", "spacing_m": None, "polygons": []} and gravel["surface"] == "unsealed"
    _, unmarked = plan_one(ramp(300.0, 6.0, main={"highway": "trunk", "lane_markings": "no"}))
    assert unmarked["gore"]["paint"] == "none"


def test_heights_carry_the_road_s_cross_fall_then_the_ramp_s_own_profile():
    _, r = plan_one(ramp(300.0, 8.0), dem=flat(GROUND + 1.0))
    outer = np.asarray(r["outer"])
    assert outer[0, 2] == pytest.approx(GROUND - section.CROWN * HALF, abs=1e-3)  # the road's edge
    z = np.asarray(r["rows"]["z"])
    assert z[-1] == pytest.approx(GROUND + 1.0, abs=0.05)  # its own profile, well out


def test_what_is_left_out_and_why():
    c, sec = road()
    # Running beside the road the whole way: no nose.
    assert plan_one(ramp(300.0, 0.5))[1] is None
    assert plan_one(ramp(300.0, 0.5))[0]["skipped"][0]["reason"] == "parallel"
    # Too near the route's end for its taper and island.
    p, r = plan_one(ramp(980.0, 6.0))
    assert r is None and p["skipped"][0]["reason"] in ("road_end", "short", "parallel")
    # A ramp whose mouth shares the road's rows with a side road's stub on its side keeps off it.
    side_road = {"id": 0, "at_m": 360.0, "xy": [360.0, 0.0], "side": "left", "angle_deg": 90.0,
                 "line": [[360.0, d] for d in np.linspace(0, 100, 21)], "length_m": 100.0, "ends": "length",
                 "highway": "residential", "closure": "minor", "name": "Side Road", "tags": {"highway": "residential"},
                 "width_m": 5.8}
    st = stubs.plan([side_road], c, sec, flat())["stubs"]
    p, r = plan_one(ramp(300.0, 6.0), c, sec, junctions=[side_road], stubs_plan=st)
    assert r is None and p["skipped"] == [{"junction": ramps.RAMP_ID_BASE, "name": None, "reason": "overlaps"}]
    # On the other side it fits.
    p, r = plan_one(ramp(300.0, -6.0), c, sec, junctions=[side_road], stubs_plan=st)
    assert r is not None
    # Two ramps on one side whose mouths overlap: the longer one is kept.
    p = ramps.plan([ramp(300.0, 6.0), ramp(330.0, 6.0, rid=ramps.RAMP_ID_BASE + 1)], c, sec, flat())
    assert len(p["ramps"]) == 1 and p["skipped"][0]["reason"] == "overlaps"


def test_a_ramp_keeps_off_the_piers_and_abutments_of_a_bridge_over_the_route():
    c, sec = road()
    _, r = plan_one(ramp(300.0, 6.0))
    # A deck crossing the road square at 380 m, with a pier column on the ramp and abutments 30 m out either side.
    rows = {"xy": [[380.0, y] for y in np.arange(-30.0, 31.0, 2.0)]}
    pier = {"columns": [np.asarray(r["rows"]["xy"])[0].tolist()]}
    deck = {"rows": rows, "deck_half_m": 4.5, "piers": [pier],
            "abutments": [{"end": "start", "face_row": 1, "end_row": 0}, {"end": "end", "face_row": 29, "end_row": 30}]}
    p, r2 = plan_one(ramp(300.0, 6.0), c, sec, crossing={"decks": [deck]})
    assert r2 is None and p["skipped"][0]["reason"] == "bridge"
    # With the pier elsewhere, it passes under the deck.
    deck["piers"] = [{"columns": [[380.0, -6.0]]}]
    assert plan_one(ramp(300.0, 6.0), c, sec, crossing={"decks": [deck]})[1] is not None
    assert ramps.structures(None) is None and ramps.structures({"decks": []}) is None


def test_a_vertex_whose_triangles_have_no_area_points_up():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 0.0, 5.0], [5.0, 0.0, 0.0]])
    tri = np.array([[0, 1, 2], [0, 3, 4]])  # the first three in a line: vertex 1 has nothing else
    n = mesh._vertex_normals(pos, tri)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0) and np.allclose(n[1], [0.0, 1.0, 0.0])
    assert np.allclose(n[3], [0.0, 1.0, 0.0])  # a real triangle's normal is as before


@pytest.mark.parametrize("eps", [0.0, 3.21e-13, 1e-10])  # 3.21e-13: vertex 3 sums to 6.42e-13
def test_a_vertex_whose_triangles_have_only_rounding_error_for_area_points_up(eps):
    # A live build (Grass Tree Hill, 18 Nov 2026): a ground vertex in two triangles of almost no area summed to a
    # normal 6.42e-13 long, and dividing by a 1e-12 floor left it 0.642 long, failing "unit normals".
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, eps], [3.0, 0.0, 0.0], [0.0, 0.0, 5.0],
                    [5.0, 0.0, 0.0]])
    tri = np.array([[0, 1, 2], [1, 3, 2], [0, 4, 5]])  # vertices 1 to 3 almost in a line
    n = mesh._vertex_normals(pos, tri)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
    assert np.allclose(n[1], [0.0, 1.0, 0.0]) and np.allclose(n[4], [0.0, 1.0, 0.0])


def test_a_ramp_whose_mouth_would_leave_a_timing_gate_nowhere_to_stand_is_left_out():
    p, r = plan_one(ramp(20.0, 8.0, kind="merge", length=100.0))  # behind the start, then
    assert r is None
    c, sec = road(n=401)
    p, r = plan_one(ramp(10.0, 6.0), c, sec)  # its mouth over the start gate at 50 m, and all 40 m round it
    assert r is None and p["skipped"][0]["reason"] in ("timing_gate", "road_end")
    p, r = plan_one(ramp(40.0, 14.0), c, sec)
    if r is not None:
        lo, hi = ramps.layout_intervals([r], c["s"], False)[0]
        assert not lo <= 50.0 <= hi


# -- the meshes ---------------------------------------------------------------------------------------------------------

def built(ramp_list, dem_h=GROUND, skirt=True, c=None, sec=None):
    if c is None:
        c, sec = road()
    plan = ramps.plan(ramp_list, c, sec, flat(dem_h))["ramps"]
    caps = stubs.mainline_caps([], c["xy"], c["heading"], extra=ramps.centres(plan))
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], flat(dem_h), skirt_cap_l=caps[0], skirt_cap_r=caps[1])
    b = ramps.build(plan, rb, c["s"], FRAME, flat(dem_h), skirt=skirt, others=c["xy"])
    cut, edge_break = ramps.road_cut(None, None, b)
    meshes = mesh.ribbon_meshes(rb, c["s"], FRAME, style=sec["style"], skirt=skirt, cut=cut,
                                edge_break=edge_break) + b.meshes
    terrain = mesh.terrain_meshes(rb, 150.0, flat(dem_h), FRAME, fold_skirt=not skirt,
                                  extra=ramps.footprint(None, b, len(c["s"])))
    return c, sec, plan, rb, b, meshes, terrain


def triangles(meshes, prefix=None):
    out = [m.positions.astype(float)[m.triangles] for m in meshes if prefix is None or m.name.startswith(prefix)]
    return np.concatenate(out)


def coverage(tris, xz, eps=1e-6):
    """How many triangles cover each AC x/z point (strictly inside; with a negative `eps`, on an edge too)."""
    a, b, c = tris[:, 0, [0, 2]], tris[:, 1, [0, 2]], tris[:, 2, [0, 2]]
    v0, v1 = c - a, b - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    keep = np.abs(den) > 1e-9
    a, v0, v1, den = a[keep], v0[keep], v1[keep], den[keep]
    out = np.zeros(len(xz), int)
    for k, p in enumerate(xz):
        v2 = p - a
        u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        out[k] = int(((u > eps) & (v > eps) & (u + v < 1 - eps)).sum())
    return out


def ac_xz(points):
    p = np.asarray(points, float).reshape(-1, 2)
    return FRAME.to_ac(p[:, 0], p[:, 1], np.zeros(len(p)))[:, [0, 2]]


@pytest.mark.parametrize("deg,kind", [(6.0, "diverge"), (-10.0, "diverge"), (8.0, "merge"), (12.0, "two_way")])
def test_the_throat_joins_the_road_s_edge_and_the_ramp_without_a_crack_or_an_overlap(deg, kind):
    c, sec, plan, rb, b, meshes, _ = built([ramp(400.0, deg, kind=kind)])
    (r,) = plan
    assert b.failed == {} and b.stats["ramps"] == 1
    side = junctions.SIDE_CODE[r["side"]]
    taper_row, _, nose_row = r["taper"]["rows"]
    lo, hi = sorted((taper_row, nose_row))
    # The throat shares the road's edge vertices from the taper's start to the nose, exactly.
    col = rb.road_cols.start if side == "L" else rb.road_cols.stop - 1
    edge = FRAME.to_ac(rb.e[lo:hi + 1, col], rb.n[lo:hi + 1, col], rb.h[lo:hi + 1, col]).astype(np.float32)
    verts = {tuple(v) for v in triangles(b.meshes, "1ROAD_JS0_G").reshape(-1, 3).astype(np.float32).tolist()}
    assert all(tuple(v) in verts for v in edge.tolist())
    # Every point of the paved outline, and of the road a little inside its edges, is road exactly once.
    paved = shapely.Polygon(r["outline"]).buffer(0)
    minx, miny, maxx, maxy = paved.bounds
    xs, ys = np.meshgrid(np.arange(minx + 0.013, maxx, 0.41), np.arange(min(miny, -HALF) + 0.217, maxy, 0.41))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    inside = (shapely.contains_xy(paved.buffer(-0.03), pts[:, 0], pts[:, 1])
              | ((np.abs(pts[:, 1]) < HALF - 0.05) & (pts[:, 0] > minx) & (pts[:, 0] < maxx)))
    road_tris = triangles(meshes, "1ROAD")
    count = coverage(road_tris, ac_xz(pts[inside]))
    assert (count <= 1).all()  # nothing drawn twice
    on_edge = pts[inside][count == 0]  # a sample that falls exactly on an edge two triangles share
    assert len(on_edge) <= 3 and (coverage(road_tris, ac_xz(on_edge), eps=-1e-4) >= 1).all()
    # The throat's heights at the road's edge are the road's.
    tri = triangles(b.meshes, "1ROAD_JS0_G").reshape(-1, 3)
    on_edge = np.isin(tri.astype(np.float32).view([("", np.float32)] * 3).ravel(),
                      edge.view([("", np.float32)] * 3).ravel())
    assert on_edge.sum() >= hi - lo


def test_the_road_s_verge_is_cut_over_the_mouth_and_its_edge_line_breaks_over_the_taper_only():
    c, sec, plan, rb, b, meshes, _ = built([ramp(400.0, 6.0)])
    (r,) = plan
    lo, hi = r["mouth"]["rows"]
    assert b.cut["L"][lo:hi].all() and b.cut["L"].sum() == hi - lo and not b.cut["R"].any()
    br0, br1 = r["taper"]["edge_break_rows"]
    assert b.edge_break["L"][br0:br1 + 1].all() and b.edge_break["L"].sum() == br1 - br0 + 1
    assert br1 == r["taper"]["rows"][1]  # to the split; solid beside the gore
    assert not b.edge_break["R"].any()


def test_the_paint_is_cut_into_the_throat_on_the_white_of_the_edge_line():
    c, sec, plan, rb, b, meshes, _ = built([ramp(400.0, 6.0, main={"highway": "trunk", "maxspeed": "60"})])
    (throat,) = [m for m in b.meshes if m.name == "1ROAD_JS0_G"]
    u = throat.uvs[:, 0][throat.triangles]
    paint = np.all(np.abs(u - ramps._paint_u("road")) < 1e-5, axis=1)
    assert paint.any() and (~paint).any()
    plain = u[~paint]
    lo, hi = stubs.THROAT_U["road"]
    assert ((plain >= lo - 1e-5) & (plain <= hi + 1e-5)).all()  # the rest is plain asphalt
    # The painted area is the plan's.
    p = throat.positions.astype(float)[throat.triangles[paint]]
    area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 2] - p[:, 0, 2])
                        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 2] - p[:, 0, 2])).sum()
    planned = sum(shapely.Polygon(np.asarray(q["exterior"])[:, :2]).area for q in plan[0]["gore"]["polygons"])
    assert area == pytest.approx(planned, rel=1e-3)


@pytest.mark.parametrize("skirt", [True, False])
def test_the_terrain_fits_round_the_ramp_and_its_island_with_no_gap_and_nothing_over_it(skirt):
    c, sec, plan, rb, b, meshes, terrain = built([ramp(400.0, 7.0)], dem_h=GROUND - 0.6, skirt=skirt)
    (r,) = plan
    every = meshes + terrain
    paved = shapely.Polygon(r["outline"]).buffer(0)
    island = shapely.Polygon(r["island"]["outline"]).buffer(0)
    around = validate._grid_in(shapely.union(paved, island).buffer(3.0), 0.43)
    assert (coverage(triangles(every), ac_xz(around)) >= 1).all()  # no hole
    on = validate._grid_in(paved.buffer(-0.05), 0.61)
    assert (coverage(triangles(every, "1GRASS"), ac_xz(on)) == 0).all()  # no ground under or over its road
    on_island = validate._grid_in(island.buffer(-0.3), 0.53)
    assert (coverage(triangles(every, "1GRASS"), ac_xz(on_island)) == 1).all()  # the island: grass, once
    assert (coverage(triangles(every, "1ROAD"), ac_xz(on_island)) == 0).all()
    # The ramp's first row's inner verge and skirt, on the nose between the throat and the island, face no terrain.
    fp = ramps.footprint(None, b, len(c["s"]))
    assert len(b.interior) >= 2 and b.interior <= fp.interior


def test_ramp_meshes_are_named_and_valid():
    c, sec, plan, rb, b, meshes, _ = built([ramp(400.0, 6.0), ramp(700.0, -8.0, rid=ramps.RAMP_ID_BASE + 1)])
    names = sorted(m.name for m in b.meshes)
    assert "1ROAD_JS0_G" in names and "1GRASS_JS0_I" in names and "1ROAD_JS1_000" in names
    assert all("_J" in n for n in names)  # never the route's own road (`validate.is_mainline_road`)
    for m in b.meshes:
        m.validate()
    assert b.stats["ramps"] == 2 and b.stats["painted"] == 2


def test_a_ramp_that_cant_be_built_is_listed_not_raised():
    c, sec = road()
    plan = ramps.plan([ramp(400.0, 6.0)], c, sec, flat())["ramps"]
    outer = np.asarray(plan[0]["outer"])
    outer[1:-1, 1] = -outer[1:-1, 1]  # an outer edge drawn across the road: the throat crosses itself
    broken = [{**plan[0], "outer": outer.tolist()}]
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], flat())
    b = ramps.build(broken, rb, c["s"], FRAME, flat(), skirt=True, others=c["xy"])
    assert ramps.RAMP_ID_BASE in b.failed and b.meshes == [] and not b.cut["L"].any()
    assert ramps.faults_of({}, b) == {ramps.RAMP_ID_BASE: ["not_built"]}


# -- checks, closures, masks and gates ----------------------------------------------------------------------------------

def kn5_nodes(meshes):
    return {"nodes": [{"class": 1, "name": "root"}] + [
        {"class": 2, "name": m.name, "vertices": {"pos": m.positions}, "indices": m.triangles.astype(np.uint16).ravel()}
        for m in meshes]}


def test_the_checks_pass_a_built_ramp_and_find_one_with_its_throat_missing():
    c, sec, plan, rb, b, meshes, terrain = built([ramp(400.0, 6.0)], dem_h=GROUND - 0.4)
    checks = ramps.checks(plan, c, sec, FRAME)
    assert [k for k, _ in checks] == ["ramp:200000", "ramp:200000:island"]
    assert validate.stub_failures(kn5_nodes(meshes + terrain), checks) == {"ramp:200000": [],
                                                                           "ramp:200000:island": []}
    found = validate.stub_failures(kn5_nodes([m for m in meshes + terrain if m.name != "1ROAD_JS0_G"]), checks)
    assert "seam" in found["ramp:200000"]
    assert ramps.faults_of(found) == {ramps.RAMP_ID_BASE: found["ramp:200000"] + [
        f"island {f}" for f in found["ramp:200000:island"]]}
    assert ramps.without_ramps({1: ["seam"], "ramp:3": ["hole"]}) == {1: ["seam"]}


def test_the_major_closure_kit_stands_on_the_ramp_past_the_nose_with_a_police_car_under_the_cap():
    c, sec, plan, rb, b, meshes, terrain = built([ramp(400.0, 9.0)], dem_h=GROUND - 0.4)
    (r,) = plan
    top = stub_closures.TopSurface(meshes + terrain, FRAME, stub_closures.areas(plan))
    closed = stub_closures.place(plan, ramps.junctions_for(plan), c, sec, FRAME, top,
                                 jurisdiction=police_car.DEFAULT_JURISDICTION)
    (rec,) = closed.records
    assert rec["level"] == "major" and rec["obstacle"] == "boards" and not rec["overshoot"]
    assert set(rec["elements"]) == {"tape", "obstacle", "warning"}  # the full kit: 60 m is room for it
    # Past the nose, where the tape keeps 3.5 m from the road.
    assert all(e["d_m"] >= r["ribbon_from_m"] and not e["in_throat"] for e in rec["elements"].values())
    assert closed.stats["cars"] == 1 and rec["car"]
    # Every prop and the car stand on the ramp (within 3.5 m of its paved outline), and nothing solid near the road.
    found = stubs.faults([], meshes + terrain, closed.meshes, c, sec, FRAME, extra=ramps.checks(plan, c, sec, FRAME))
    assert ramps.faults_of(found) == {}


def test_a_long_ramp_s_closure_still_stands_past_its_nose():
    # At 3 degrees the nose is about 190 m from the node. The closure measures from the road within 160 m of the
    # ramp's outline (`stub_closures.REACH_M`): with rows from the node, the road there was out of its reach, so the
    # tape "cleared" it at d = 0, on the road (seen on the Southern Outlet). ramps.json's rows start near the nose.
    c, sec, plan, rb, b, meshes, terrain = built([ramp(300.0, 3.0)], dem_h=GROUND - 0.4)
    (r,) = plan
    assert r["rows_from_m"] > 150.0
    top = stub_closures.TopSurface(meshes + terrain, FRAME, stub_closures.areas(plan))
    closed = stub_closures.place(plan, ramps.junctions_for(plan), c, sec, FRAME, top)
    (rec,) = closed.records
    assert all(e["d_m"] >= r["ribbon_from_m"] for e in rec["elements"].values())
    found = stubs.faults([], meshes + terrain, closed.meshes, c, sec, FRAME, extra=ramps.checks(plan, c, sec, FRAME))
    assert ramps.faults_of(found) == {}


def test_masks_and_gates_keep_clear_of_the_taper_gore_and_island_on_the_ramp_s_side():
    c, _ = road()
    _, r = plan_one(ramp(400.0, 6.0))
    m = ramps.masks([r], c["s"])
    lo, hi = r["mouth"]["from_m"], r["mouth"]["to_m"]
    s = c["s"]
    assert m["L"][(s >= lo - junctions.MOUTH_MARGIN_M) & (s <= hi + junctions.MOUTH_MARGIN_M)].all()
    assert m["L"].sum() == pytest.approx(hi - lo + 2 * junctions.MOUTH_MARGIN_M + 1, abs=1.0) and not m["R"].any()
    assert ramps.layout_intervals([r], s, False) == [(lo - 5.0, hi + 5.0)]
    assert ramps.layout_intervals([r], s, True) == [(1000.0 - hi - 5.0, 1000.0 - lo + 5.0)]
    zone = ramps.zone([r], 4.0)
    assert zone.contains(shapely.Polygon(r["island"]["outline"]).buffer(-0.01))
    assert ramps.zone([], 4.0) is None


def test_settle_leaves_out_a_faulty_ramp_and_builds_again_without_it():
    state = {"ramps": [{"junction": k, "name": None, "station_m": 100.0 * k} for k in (1, 2)]}
    runs = []

    def run():
        runs.append([r["junction"] for r in state["ramps"]])
        return runs[-1]

    out = ramps.settle(state, run, lambda rs, out: {2: ["hole"]} if 2 in out else {})
    assert out == [1] and runs == [[1, 2], [1]]
    assert state["dropped"] == [{"junction": 2, "name": None, "reason": "build_check", "faults": ["hole"]}]
    state = {"ramps": [{"junction": 1, "name": None, "station_m": 0.0}]}
    out = ramps.settle(state, lambda: [r["junction"] for r in state["ramps"]], lambda rs, o: {1: ["seam"]} if o else {},
                       rebuilds=1)
    assert out == [] and state["ramps"] == []


# -- the section stage --------------------------------------------------------------------------------------------------

HOBART = [{"lon": 147.301, "lat": -42.8001}, {"lon": 147.319, "lat": -42.7999}]


class _Dem:
    resolution_m = 2.0

    def load_corridor(self, xy, m):
        return 1

    def sample(self, x, y):
        return np.full(np.shape(x), GROUND + 0.5)


class _Region:
    epsg = 28355

    def elevation(self, settings):
        return _Dem()


def test_the_section_stage_plans_ramps_and_leaves_the_half_widths_and_the_ai_line_byte_for_byte(tmp_path, monkeypatch):
    monkeypatch.setattr(section_stage, "job_region", lambda waypoints: _Region())
    work = WorkDir(tmp_path / "work")
    c, _ = road()
    np.savez_compressed(work.stage("02_profile") / "centreline.npz", run=np.zeros(len(c["s"]), np.int64), **c)
    work.write_json(route_stage.STAGE, "runs.json", [{"tags": {"highway": "trunk", "surface": "asphalt"}}])
    work.write_json(route_stage.STAGE, route_stage.JUNCTIONS, {"version": 1, "junctions": []})
    work.write_json(route_stage.STAGE, route_stage.RAMPS, {"version": 1, "ramps": [ramp(400.0, 6.0)]})
    job = Job(track_id="t", waypoints=HOBART)
    out = section_stage.run(job, config.load(), work)
    assert out["ramps"] == {"ramps": 1, "skipped": {}}
    doc = work.read_json(section_stage.STAGE, section_stage.RAMPS)
    assert doc["version"] == ramps.FORMAT_VERSION and doc["crs"] == "EPSG:28355"
    assert [r["junction"] for r in section_stage.ramps(work, job)] == [ramps.RAMP_ID_BASE]
    with_ramps = dict(np.load(work.root / section_stage.STAGE / "section.npz"))
    none = Job(track_id="t", waypoints=HOBART, junctions="none")
    assert section_stage.ramps(work, none) == [] and route_stage.ramps(work, none) == []
    out = section_stage.run(none, config.load(), work)
    assert "ramps" not in out and not (work.root / section_stage.STAGE / section_stage.RAMPS).exists()
    without = dict(np.load(work.root / section_stage.STAGE / "section.npz"))
    assert all(np.array_equal(with_ramps[k], without[k]) for k in with_ramps)
    files = []
    for sec in (with_ramps, without):
        rm = RoadModel.from_arrays(c["s"], c["xy"], c["z"], c["heading"], c["curvature"], c["grade"],
                                   sec["half_left"], sec["half_right"], sec["slope_left"], sec["slope_right"], False)
        path = tmp_path / f"fast_lane_{len(files)}.ai"
        ai_line.write(path, ai_line.build(rm, FRAME, end_short_m=40.0))
        files.append(path.read_bytes())
    assert files[0] == files[1]


def test_the_later_stages_read_the_ramps_as_built(tmp_path):
    from mapcreator.pipeline import mesh_stage

    work = WorkDir(tmp_path / "work")
    job = Job(track_id="t", waypoints=HOBART)
    work.write_json(route_stage.STAGE, route_stage.RAMPS, {"version": 1, "ramps": [ramp(400.0, 6.0)]})
    work.write_json(section_stage.STAGE, section_stage.RAMPS, {"version": 1, "ramps": [{"junction": 1},
                                                                                        {"junction": 2}]})
    assert [r["junction"] for r in mesh_stage.built_ramps(work, job)] == [1, 2]
    work.write_json(mesh_stage.STAGE, mesh_stage.RAMPS, {"version": 1, "ramps": [{"junction": 1}], "skipped": []})
    assert [r["junction"] for r in mesh_stage.built_ramps(work, job)] == [1]
    assert mesh_stage.built_ramps(work, Job(track_id="t", waypoints=HOBART, junctions="none")) == []
