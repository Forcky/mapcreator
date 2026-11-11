"""Roundabouts the route drives (`core.roundabouts`): the ring recorded in the order it's driven, the rest of it drawn
and joined to the road at both ends, its island, its exits closed, the road's verge and edge line at the island, the
checks that leave a faulty ring out whole, and the preview's warning for a small ring."""

import json

import httpx
import numpy as np
import pytest
import shapely
from pyproj import Transformer

from mapcreator.core import centreline as cl
from mapcreator.core import junctions, mesh, roundabouts, section, stubs, textures, validate
from mapcreator.core.scene import LocalFrame
from mapcreator.job import Waypoint
from mapcreator.providers.osm import graph
from mapcreator.service.routing import GraphHopperRouter

IDENT = Transformer.from_crs(4326, 4326, always_xy=True)  # the test networks are drawn in metres
FRAME = LocalFrame(0.0, 0.0, 0.0)
GROUND = 10.0
HALF = 3.2


def ring_network(radius=30.0, step_deg=15.0, oneway=True, back_to_ring=False, ring_off_route=False):
    """A secondary road from the west (x 0 to 450) into a ring of `radius` round (450 + radius, 0), driven clockwise,
    and on east from its eastern node; a tertiary road north from its top node (on the route's arc) and a residential
    road south from its bottom node (on the rest of the ring). `back_to_ring`: a road from the east road's node 100 m
    past the ring back to the ring's bottom node. `ring_off_route`: a second ring 60 m north of the west road at
    x = 200, reached by a side road."""
    cx = 450.0 + radius
    nodes, ways = {}, []
    west = [(x, 0.0) for x in np.arange(0.0, 450.0 - 1e-6, 50.0)]
    nodes |= {k: p for k, p in enumerate(west)}
    angles = np.radians(np.arange(180.0, -180.0, -step_deg))  # clockwise, from the west node
    ring_ids = [100 + k for k in range(len(angles))]
    nodes |= {i: (cx + radius * np.cos(t), radius * np.sin(t)) for i, t in zip(ring_ids, angles)}
    east = [(cx + radius + d, 0.0) for d in np.arange(50.0, 1000.0 - cx - radius + 1e-6, 50.0)]
    nodes |= {300 + k: p for k, p in enumerate(east)}
    ring_tags = {"highway": "secondary", "junction": "roundabout", **({"oneway": "yes"} if oneway else {})}
    ways.append((1, list(range(len(west))) + [ring_ids[0]], {"highway": "secondary"}))
    ways.append((2, ring_ids + [ring_ids[0]], ring_tags))
    i_east = ring_ids[int(round(180.0 / step_deg))]
    ways.append((3, [i_east] + [300 + k for k in range(len(east))], {"highway": "secondary"}))
    i_top, i_bottom = ring_ids[int(round(90.0 / step_deg))], ring_ids[int(round(270.0 / step_deg))]
    for k, y in enumerate((60.0, 120.0, 180.0)):
        nodes[500 + k] = (cx, radius + y)
        nodes[600 + k] = (cx, -radius - y)
    ways.append((4, [i_top, 500, 501, 502], {"highway": "tertiary", "name": "North Road"}))
    ways.append((5, [i_bottom, 600, 601, 602], {"highway": "residential", "name": "South Road"}))
    if back_to_ring:  # under 100 m long
        nodes[700] = (cx + radius + 10.0, -40.0)
        ways.append((6, [300, 700, i_bottom], {"highway": "unclassified", "name": "Back Lane"}))
    if ring_off_route:
        th = np.radians(np.arange(270.0, -90.0, -30.0))
        far = [800 + k for k in range(len(th))]
        nodes |= {i: (200.0 + 12 * np.cos(t), 72.0 + 12 * np.sin(t)) for i, t in zip(far, th)}
        nodes[900] = (200.0, 30.0)
        nodes[901] = (200.0 + 100.0, 72.0)
        ways.append((7, far + [far[0]], {"highway": "residential", "junction": "roundabout"}))
        ways.append((8, [4, 900, far[0]], {"highway": "residential", "name": "Ring Street"}))
        ways.append((9, [far[3], 901], {"highway": "residential", "name": "Far Street"}))
    els = [{"type": "node", "id": i, "lon": x, "lat": y} for i, (x, y) in nodes.items()]
    els += [{"type": "way", "id": w, "nodes": refs, "tags": tags} for w, refs, tags in ways]
    return graph.build({"elements": els}, IDENT)


def route_xy(g):
    return graph.route(g, np.array([[10.0, 0.0], [990.0, 0.0]])).xy


def arrays(xy):
    """The profile's and section's arrays for a route on flat ground: 6.4 m sealed, crown, no camber."""
    line = cl.smooth(xy)
    n = len(line.s)
    c = {"s": line.s, "xy": line.xy, "heading": line.heading, "curvature": line.curvature,
         "z": np.full(n, GROUND), "grade": np.zeros(n)}
    half = np.full(n, HALF)
    sec = {"half_left": half, "half_right": half, "slope_left": np.full(n, section.CROWN),
           "slope_right": np.full(n, section.CROWN), "style": np.zeros(n, np.int8)}
    return c, sec


def flat(x, y):
    return np.full(np.shape(x), GROUND)


def planned(radius=30.0, **kw):
    g = ring_network(radius, **kw)
    xy = route_xy(g)
    found = junctions.find(g, xy)
    c, sec = arrays(xy)
    return found, c, sec, roundabouts.plan(found["roundabouts"], c, sec, flat)


def built(radius=30.0):
    """The road, its stubs and the ring as the mesh stage builds them, with the terrain; and the checks."""
    found, c, sec, p = planned(radius)
    rings = p["roundabouts"]
    plan = stubs.plan(found["junctions"], c, sec, flat, avoid=roundabouts.avoid(rings))["stubs"]
    caps = stubs.mainline_caps(plan, c["xy"], c["heading"], extra=roundabouts.centres(rings))
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], flat, skirt_cap_l=caps[0], skirt_cap_r=caps[1])
    sb = stubs.build(plan, rb, c["s"], FRAME, flat, skirt=True,
                     main_xy=np.vstack([c["xy"], roundabouts.centres(rings)])) if plan else None
    ring = roundabouts.build(rings, rb, c["s"], FRAME, flat, skirt=True,
                             others=np.vstack([c["xy"]] + [stubs._centre(s) for s in plan]))
    cut = {k: ring.cut[k] | (sb.cut[k] if sb else False) for k in "LR"}
    edge_break = {k: ring.edge_break[k] | (sb.edge_break[k] if sb else False) for k in "LR"}
    meshes = mesh.ribbon_meshes(rb, c["s"], FRAME, cut=cut, edge_break=edge_break) + ring.meshes
    meshes += sb.meshes if sb else []
    meshes += mesh.terrain_meshes(rb, 150.0, flat, FRAME, extra=roundabouts.footprint(sb, ring, cut))
    checks = roundabouts.checks(rings, c, sec, FRAME)
    checks += [(s["junction"], validate.stub_check(s, c["s"], c["xy"], c["heading"], sec["half_left"],
                                                   sec["half_right"], FRAME)) for s in plan]
    return found, c, sec, rings, plan, ring, meshes, checks


# -- what the route stage records --------------------------------------------------------------------------------

def test_the_ring_is_recorded_in_the_order_it_s_driven_with_where_the_route_joins_and_leaves_it():
    g = ring_network()
    found = junctions.find(g, route_xy(g))
    (r,) = found["roundabouts"]
    assert r["oneway"] and r["clockwise"] and r["closed"]
    loop = np.asarray(r["ring"], float)
    a = np.unwrap(np.arctan2(loop[:, 1], loop[:, 0] - 480.0))
    assert np.all(np.diff(a) < 0)  # clockwise all the way round, as the ways are drawn
    entry, leave = r["route_ring_index"]
    assert loop[entry] == pytest.approx([450.0, 0.0], abs=0.01) and loop[leave] == pytest.approx([510.0, 0.0], abs=0.01)
    assert len(r["segment_way"]) == len(loop) - 1 and r["ways"][0]["tags"]["junction"] == "roundabout"
    # North Road leaves where the route is on the ring: a side road of the route's. South Road is the ring's exit.
    assert [(j["name"], j["side"]) for j in found["junctions"]] == [("North Road", "left")]
    assert [(e["name"], e["on_route"]) for e in r["exits"]] == [("North Road", True), ("South Road", False)]


def test_a_ring_drawn_against_its_ways_is_turned_round_and_a_two_way_ring_is_not_one_way():
    g = ring_network()
    found = junctions.find(g, route_xy(g))
    assert found["roundabouts"][0]["ring"][1][1] > 0  # the node after the west one is north of it: clockwise
    g2 = ring_network(oneway=True)
    g2.seg_fwd[g2.seg_way == 1], g2.seg_rev[g2.seg_way == 1] = g2.seg_rev[g2.seg_way == 1], g2.seg_fwd[g2.seg_way == 1]
    two_way = ring_network()
    ring_way = [k for k, w in enumerate(two_way.ways) if w["tags"].get("junction")][0]
    two_way.seg_rev[two_way.seg_way == ring_way] = True
    assert not junctions._driving_order(junctions._Network(two_way), [100, 101, 102, 100], True)[1]


def test_a_side_road_back_onto_the_ring_is_not_a_junction():
    g = ring_network(back_to_ring=True)
    found = junctions.find(g, route_xy(g))
    assert "Back Lane" not in [j["name"] for j in found["junctions"]]
    assert [s["reason"] for s in found["skipped"] if s["name"] == "Back Lane"] == ["roundabout"]


def test_a_side_road_stops_at_a_ring_the_route_doesn_t_drive():
    g = ring_network(ring_off_route=True)
    found = junctions.find(g, route_xy(g))
    (ring_street,) = [j for j in found["junctions"] if j["name"] == "Ring Street"]
    assert ring_street["ends"] == "roundabout" and ring_street["length_m"] == pytest.approx(60.0, abs=0.1)
    assert len(found["roundabouts"]) == 1  # the far ring isn't the route's


# -- the plan --------------------------------------------------------------------------------------------------------

def test_the_rest_of_the_ring_is_planned_with_its_ends_island_and_exit():
    found, c, sec, p = planned()
    assert p["skipped"] == []
    (r,) = p["roundabouts"]
    assert r["island_side"] == "right" and r["clockwise"]
    leaving, joining = r["ends"]
    assert leaving["end"] == "leaving" and joining["end"] == "joining"
    # The joining end's mouth comes first along the road, the leaving end's last; the ring takes the road's right side
    # from the first to the last.
    assert joining["mouth"]["rows"][1] < leaving["mouth"]["rows"][0]
    assert r["mouth"]["rows"] == [joining["mouth"]["rows"][0], leaving["mouth"]["rows"][1]]
    for end in r["ends"]:
        flares = end["flares"]
        assert sorted(f["island"] for f in flares.values()) == [False, True]
        assert all(f["radius_m"] >= stubs.FLARE_MIN_RADIUS_M - 1e-6 for f in flares.values())
    island = shapely.Polygon(r["island"]["outline"])
    assert island.is_valid and island.contains(shapely.Point(480.0, 0.0))
    assert r["island"]["inner_radius_m"] == pytest.approx(30.0 - 3.4, abs=1.0)  # a secondary ring: 6.8 m
    rows = np.asarray(r["rows"]["xy"], float)
    assert np.hypot(rows[:, 0] - 480.0, rows[:, 1]).mean() == pytest.approx(30.0, abs=0.5)
    assert (rows[:, 1] < 0).mean() > 0.6  # the rest of the ring: its southern side
    assert [(x["name"], x["closure"], x["parent"]) for x in r["exits"]] == [("South Road", "minor", "ring:0")]
    assert r["exits"][0]["junction"] == roundabouts.EXIT_ID_BASE + 1


def test_a_ring_only_touched_or_not_one_way_is_left_out():
    g = ring_network()
    found = junctions.find(g, route_xy(g))
    touched = {**found["roundabouts"][0], "route_ring_index": [0, 0]}
    two_way = {**found["roundabouts"][0], "oneway": False}
    c, sec = arrays(route_xy(g))
    p = roundabouts.plan([touched, two_way], c, sec, flat)
    assert p["roundabouts"] == [] and [s["reason"] for s in p["skipped"]] == ["touches", "not_one_way"]


def test_a_small_ring_is_still_drawn_and_its_island_has_no_trees():
    found, c, sec, p = planned(radius=12.0)
    (r,) = p["roundabouts"]
    assert not r["island"]["trees"]
    zone = roundabouts.zone(p["roundabouts"], 4.0)
    assert zone.contains(shapely.Polygon(r["island"]["outline"]).buffer(-0.01))
    big = planned(radius=30.0)[3]["roundabouts"]
    assert big[0]["island"]["trees"]
    assert not roundabouts.zone(big, 4.0).contains(shapely.Point(480.0, 0.0))  # a big island's middle may have trees


def test_side_roads_stubs_keep_off_the_ring():
    found, c, sec, p = planned()
    avoid = roundabouts.avoid(p["roundabouts"])
    plan = stubs.plan(found["junctions"], c, sec, flat, avoid=avoid)
    assert [s["name"] for s in plan["stubs"]] == ["North Road"]
    assert not shapely.Polygon(plan["stubs"][0]["outline"]).intersects(avoid)
    # A stub whose throat would reach the ring is left out.
    near = {**found["junctions"][0], "xy": [480.0, -30.0]}
    assert stubs.plan([near], c, sec, flat, avoid=avoid.buffer(40.0))["skipped"][0]["reason"] == "roundabout"


# -- the meshes and the checks --------------------------------------------------------------------------------------

def test_the_ring_island_and_exit_are_built_and_pass_the_stub_checks():
    found, c, sec, rings, plan, ring, meshes, checks = built()
    names = {m.name for m in meshes}
    assert {"1ROAD_RB0_000", "1ROAD_RB0_T", "1ROAD_RB0_K", "1GRASS_RB0_I", "1GRASS_RB0_V000L", "1ROAD_JR0_000"} <= names
    assert not any(n.startswith("1GRASS_RB0_V") and n.endswith("R") for n in names)  # no verge on the island's side
    faults = validate.stub_failures(validate.mesh_nodes(meshes), checks)
    assert faults and all(v == [] for v in faults.values())
    kerb = next(m for m in meshes if m.name == "1ROAD_RB0_K")
    assert kerb.material == roundabouts.KERB_MATERIAL and "kerb" in textures.LIBRARY


def test_the_island_rises_a_kerb_then_domes_and_no_mesh_but_the_island_s_covers_it():
    found, c, sec, rings, plan, ring, meshes, checks = built()
    island = shapely.Polygon(rings[0]["island"]["outline"])
    inside = island.buffer(-0.5)
    for m in meshes:
        p = m.positions.astype(float)[m.triangles].mean(axis=1)
        e, n, h = FRAME.from_ac(p)
        hit = shapely.contains_xy(inside, e, n)
        if hit.any():
            assert m.name in ("1GRASS_RB0_I", "1ROAD_RB0_K"), m.name
    # The road's edges are its crown below the centreline (8 cm on 3.2 m); the kerb's top is 15 cm above them.
    edge = GROUND - section.CROWN * HALF
    grass = next(m for m in meshes if m.name == "1GRASS_RB0_I")
    e, n, h = FRAME.from_ac(grass.positions.astype(float))
    on = shapely.contains_xy(island.buffer(-roundabouts.ISLAND_KERB_W - 0.01), e, n)
    assert h[on].min() >= edge + roundabouts.ISLAND_KERB_H - 0.03
    assert h[on].max() <= GROUND + roundabouts.ISLAND_KERB_H + roundabouts.DOME_MAX_M + 0.03


def test_the_road_s_island_side_loses_its_verge_and_edge_line_between_the_ring_s_ends_only():
    found, c, sec, rings, plan, ring, meshes, checks = built()
    a, b = rings[0]["mouth"]["rows"]
    assert ring.cut["R"][a:b].all() and not ring.cut["R"][:a].any() and not ring.cut["R"][b:].any()
    assert ring.edge_break["R"][a + 1:b].all() and not ring.edge_break["L"].any()
    road = [m for m in meshes if m.name.startswith("1ROAD_RB0_0")]
    inside = textures.edge_line_u("road_marked") + mesh.EDGE_BREAK_MARGIN_U
    u = np.concatenate([m.uvs[:, 0] for m in road])
    assert np.isclose(u, 1.0 - inside, atol=1e-6).any()  # the ring's island-side edge: no line
    assert np.isclose(u, 0.0, atol=1e-6).any()  # its outer edge keeps the painted line


def test_a_faulty_ring_is_left_out_whole_and_a_faulty_exit_alone():
    found, c, sec, p = planned()
    rings = p["roundabouts"]
    seen = []

    def build(state):
        seen.append((len(state[0]), len(state[1]), sum(len(r["exits"]) for r in state[1])))
        return state

    def check_exit(state, result):
        return {st["junction"]: ["seam"] for r in state[1] for st in r["exits"]}

    stubs_kept, rings_kept, _, dropped, dropped_rings = roundabouts.settle(([], rings), build, check_exit)
    assert len(rings_kept) == 1 and rings_kept[0]["exits"] == []
    assert [d["reason"] for d in dropped_rings] == ["build_check"] and "junction" in dropped_rings[0]

    def check_ring(state, result):
        return {("ring", r["id"], "joining"): ["seam"] for r in state[1]}

    _, rings_kept, _, _, dropped_rings = roundabouts.settle(([], rings), build, check_ring)
    assert rings_kept == [] and dropped_rings[0]["faults"] == ["joining: seam"]


def test_masks_and_gate_intervals_cover_the_ring_s_stretch_of_the_road():
    found, c, sec, p = planned()
    r = p["roundabouts"][0]
    m = roundabouts.masks(p["roundabouts"], c["s"])
    lo, hi = r["mouth"]["from_m"], r["mouth"]["to_m"]
    assert m["R"][(c["s"] >= lo) & (c["s"] <= hi)].all() and not m["L"].any()
    total = float(c["s"][-1])
    (fwd,), (rev,) = roundabouts.layout_intervals(p["roundabouts"], c["s"], False), \
        roundabouts.layout_intervals(p["roundabouts"], c["s"], True)
    assert fwd[0] < lo < hi < fwd[1] and rev == pytest.approx((total - fwd[1], total - fwd[0]))


def test_the_ring_s_exit_closure_measures_its_clearance_from_the_ring():
    from mapcreator.core import stub_closures

    found, c, sec, rings, plan, ring, meshes, checks = built()
    every = plan + roundabouts.exit_stubs(rings)
    top = stub_closures.TopSurface(meshes, FRAME, stub_closures.areas(every))
    placed = stub_closures.place(every, found["junctions"] + roundabouts.exit_junctions(rings), c, sec, FRAME, top,
                                 parents=roundabouts.parents(rings), extra_road=roundabouts.paved(rings))
    south = next(r for r in placed.records if r["name"] == "South Road")
    assert south["elements"]["tape"]["d_m"] >= 4.0
    walls = [m for m in placed.meshes if m.name.startswith("1WALL_CLOSURE")]
    if walls:  # bales: solid, so 5 m clear of the ring's road as of the route's
        xz = np.vstack([m.positions[:, [0, 2]] for m in walls]).astype(float)
        e, n, _ = FRAME.from_ac(np.column_stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]]))
        assert shapely.distance(roundabouts.paved(rings), shapely.points(np.column_stack([e, n]))).min() >= 5.0


# -- the preview's warning ------------------------------------------------------------------------------------------

def test_fit_radius_finds_a_circle_from_an_arc():
    t = np.radians(np.linspace(10.0, 100.0, 7))
    assert roundabouts.fit_radius(np.column_stack([5 + 10.2 * np.cos(t), -3 + 10.2 * np.sin(t)])) == \
        pytest.approx(10.2, abs=1e-6)
    assert roundabouts.fit_radius(np.array([[0.0, 0.0], [1.0, 1.0]])) is None
    assert roundabouts.fit_radius(np.column_stack([np.arange(5.0), np.zeros(5)])) is None


@pytest.mark.parametrize("radius, warns", [(10.0, True), (10.3, True), (12.0, False), (25.0, False)])
def test_the_preview_warns_of_a_ring_of_10_3_m_or_less(radius, warns):
    g = ring_network(radius=30.0)
    xy = route_xy(g)
    notes = roundabouts.preview_notes(xy, [{"from_m": 450.0, "to_m": 500.0, "radius_m": radius}])
    assert bool(notes) == warns
    if warns:
        assert f"({radius:.0f} m radius)" in notes[0]["message"] and "generate it anyway" in notes[0]["message"]


def test_the_tightest_bend_near_a_ring_is_measured_as_validation_measures_the_road():
    # East along y = 0, round a 20 m half circle (tangent at both ends) and back west along y = 40.
    t = np.radians(np.arange(-90.0, 90.1, 5.0))
    xy = np.vstack([[[-200.0, 0.0], [-100.0, 0.0]], np.column_stack([20 * np.cos(t), 20 + 20 * np.sin(t)]),
                    [[-100.0, 40.0], [-200.0, 40.0]]])
    s = cl.cumulative(xy)
    r = roundabouts.tightest_near(xy, float(s[2]), float(s[-3]))
    # As validation's route check reads the profile's line: smoothed, its curvature smoothed over 2 m. The spline
    # overshoots a little where the straight meets the arc, so it's tighter than the drawn 20 m.
    from mapcreator.core import ai_line
    assert r == pytest.approx(float(ai_line.radius_of(cl.smooth(xy).curvature).min()), rel=1e-6)
    assert 14.0 < r < 20.0
    assert roundabouts.tightest_near(xy, 0.0, 10.0) > 100.0  # the straight, far from the bend


def test_the_preview_warns_where_the_route_looks_tighter_than_5_m_whatever_the_ring(monkeypatch):
    g = ring_network(radius=30.0)
    xy = route_xy(g)
    monkeypatch.setattr(roundabouts, "tightest_near", lambda xy, a, b: 3.7)
    (note,) = roundabouts.preview_notes(xy, [{"from_m": 450.0, "to_m": 500.0, "radius_m": None}])
    assert note["tightest_m"] == 3.7 and note["radius_m"] is None and "looks tighter" in note["message"]
    assert "about 3.7 m" in note["message"]


def test_graphhopper_previews_ask_for_the_roundabout_detail_and_warn_of_a_small_ring():
    seen = []
    t = np.radians(np.linspace(180.0, 0.0, 9))
    # Around Hobart, in degrees: a 10 m ring's half, 0.00012 degrees of latitude being about 13 m.
    lon = list(147.30 + np.linspace(-0.004, -0.00012, 6)) + list(147.30 + 0.000122 * np.cos(t)) + \
        list(147.30 + np.linspace(0.00012, 0.004, 6))
    lat = [-42.80] * 6 + list(-42.80 + 0.00009 * np.sin(t)) + [-42.80] * 6
    coords = [[a, b] for a, b in zip(lon, lat)]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"paths": [{"distance": 700.0, "points": {"coordinates": coords},
                                                    "details": {"roundabout": [[0, 6, False], [6, 14, True],
                                                                               [14, 20, False]]}}]})

    router = GraphHopperRouter("http://router.example.org:8989", client=httpx.Client(transport=httpx.MockTransport(handler)))
    preview = router.preview([Waypoint(lon=147.296, lat=-42.80), Waypoint(lon=147.304, lat=-42.80)])
    assert seen[0]["details"] == ["roundabout"]
    (note,) = preview["roundabouts"]
    assert note["radius_m"] == pytest.approx(10.0, abs=0.5) and "small roundabout" in note["message"]


def test_a_graphhopper_without_the_detail_is_asked_again_without_it_and_not_again():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "details" in body:
            return httpx.Response(400, json={"message": "Cannot find the path details: [roundabout]",
                                             "hints": [{"details": "java.lang.IllegalArgumentException"}]})
        return httpx.Response(200, json={"paths": [{"distance": 100.0, "points": {"coordinates": [
            [147.301, -42.8], [147.302, -42.8]]}}]})

    router = GraphHopperRouter("http://router.example.org:8989", client=httpx.Client(transport=httpx.MockTransport(handler)))
    pts = [Waypoint(lon=147.301, lat=-42.8), Waypoint(lon=147.302, lat=-42.8)]
    assert "roundabouts" not in router.preview(pts)
    router.preview(pts)
    assert ["details" in b for b in seen] == [True, False, False]
