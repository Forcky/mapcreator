"""Roads crossing the route, drawn (`core.crossing_roads`): the crossing road followed past its way's ends
(`crossings.road_through`); the plan (a road under the route's bridge, only where the deck stands clear of the ground;
a bridge over the route, only where its underside clears the route by 4.6 m, with its abutments, piers and
approaches, or why not); the ribbons (clear of the route's ground and of each other, the terrain fitted round them
with no hole and nothing over them); the decks (visual, their colliders invisible and clear of the road, everything
facing out); the ground held under a deck; the validator's checks; and the section stage's plan."""

import math

import numpy as np
import pytest
import shapely

from mapcreator import config
from mapcreator.core import crossing_roads as cr
from mapcreator.core import crossings, mesh, section, validate
from mapcreator.core.scene import LocalFrame
from mapcreator.job import Job
from mapcreator.pipeline import route_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.osm import graph

FRAME = LocalFrame(0.0, 0.0, 0.0)
HALF = 3.2


def road(n=1001, z=10.0, z_raw=None):
    """The route: straight east along y = 0, a station every metre, 6.4 m wide with a crown."""
    s = np.arange(n, dtype=float)
    zz = np.full(n, z, float) if np.isscalar(z) else np.asarray(z, float)
    c = {"s": s, "xy": np.column_stack([s, np.zeros(n)]), "heading": np.zeros(n), "curvature": np.zeros(n),
         "z": zz, "z_raw": zz.copy() if z_raw is None else np.asarray(z_raw, float), "grade": np.zeros(n)}
    sec = {"half_left": np.full(n, HALF), "half_right": np.full(n, HALF), "slope_left": np.full(n, section.CROWN),
           "slope_right": np.full(n, section.CROWN), "style": np.zeros(n, np.int8)}
    return c, sec


def crossing(relation, *, x=500.0, deg=90.0, length=150.0, cid=0, way_id=7, highway="secondary", bridge=None,
             road_ways=None, name="Cross Road", tags=None):
    """A crossings.json record: a straight road through (x, 0) at `deg` to the route, `length` either side, its way
    the whole of it unless `road_ways` gives the ways along its line, (from, to, bridge) metres."""
    a = math.radians(deg)
    u = np.array([math.cos(a), math.sin(a)])
    line = [list(np.array([x, 0.0]) + u * d) for d in np.linspace(-length, length, 31)]
    ways = road_ways or [(0.0, 2 * length, bool(bridge))]
    return {"id": cid, "at_m": x, "xy": [x, 0.0], "relation": relation, "angle_deg": deg, "way_id": way_id,
            "highway": highway, "name": name, "level": 1.0 if relation == "over" else 0.0,
            "bridge": bool(bridge), "tunnel": False, "tags": {"highway": highway, **(tags or {})},
            "line": line, "line_at_m": length, "route": {"way_id": 1, "level": 0.0, "bridge": relation == "under",
                                                         "tunnel": False},
            "road": {"line": line, "at_m": length, "ends": ["length", "length"],
                     "ways": [{"way_id": way_id + k, "from_m": f, "to_m": t, "bridge": b, "tunnel": False,
                               "level": 1.0 if b else 0.0, "highway": highway} for k, (f, t, b) in enumerate(ways)]}}


def flat(h):
    return lambda x, y: np.full(np.shape(x), h, float)


def embankment(top=7.0, low=0.0, half_gap=20.0, x=500.0, deg=90.0):
    """Ground at `low`, rising to `top` more than `half_gap` from the route either side, only within 12 m of the
    crossing road through (x, 0) at `deg` (its embankments), so the route's own DEM stays at `low`."""
    a = math.radians(deg)

    def sample(xx, yy):
        xx, yy = np.asarray(xx, float), np.asarray(yy, float)
        off = np.abs(-(xx - x) * math.sin(a) + yy * math.cos(a))
        return np.where((np.abs(yy) >= half_gap) & (off < 12.0), top, low)
    return sample


# -- the crossing road followed past its way's ends -------------------------------------------------------------------

def osm(nodes, ways):
    from pyproj import Transformer

    ident = Transformer.from_crs(4326, 4326, always_xy=True)
    els = [{"type": "node", "id": i, "lon": x, "lat": y} for i, (x, y) in nodes.items()]
    els += [{"type": "way", "id": w, "nodes": refs, "tags": t} for w, refs, t in ways]
    return graph.build({"elements": els}, ident)


def test_a_bridge_over_the_route_is_followed_into_its_approaches_and_straight_on_at_a_junction():
    main = {i: (100.0 * i, 0.0) for i in range(11)}
    nodes = {**main, 100: (450.0, -300.0), 101: (450.0, -25.0), 102: (450.0, 25.0), 103: (450.0, 300.0),
             104: (650.0, 25.0), 105: (452.0, 120.0)}
    ways = [(1, list(range(11)), {"highway": "secondary"}),
            (10, [100, 101], {"highway": "tertiary"}),  # the south approach
            (11, [101, 102], {"highway": "tertiary", "bridge": "yes", "layer": "1"}),  # the bridge
            (12, [102, 105, 103], {"highway": "tertiary"}),  # straight on north
            (13, [102, 104], {"highway": "residential"})]  # turning off east at the bridge's end
    g = osm(nodes, ways)
    r = graph.route(g, np.array([[10.0, 0.0], [990.0, 0.0]]))
    (c,) = crossings.find(g, r.xy, r.seg_way)["crossings"]
    rd = c["road"]
    assert [w["way_id"] for w in rd["ways"]] == [10, 11, 12]  # the straightest way on, not the turn
    assert [w["bridge"] for w in rd["ways"]] == [False, True, False]
    bridge = rd["ways"][1]
    assert bridge["to_m"] - bridge["from_m"] == pytest.approx(50.0, abs=0.01)
    assert rd["at_m"] == pytest.approx(150.0, abs=0.01) and rd["ends"] == ["length", "length"]
    assert crossings.cumulative(np.asarray(rd["line"]))[-1] == pytest.approx(300.0, abs=0.01)
    assert c["line"] == [[450.0, -25.0], [450.0, 25.0]]  # the way's own line, as before


def test_a_crossing_road_stops_at_a_dead_end_and_at_the_route_and_goes_round_a_loop():
    main = {i: (100.0 * i, 0.0) for i in range(11)}
    loop = {200 + k: (500.0 + 30.0 * math.cos(t), 30.0 * math.sin(t)) for k, t in
            enumerate(np.linspace(0, 2 * math.pi, 13)[:-1] + 0.1)}
    nodes = {**main, **loop, 100: (250.0, -40.0), 101: (250.0, 40.0)}
    ways = [(1, list(range(11)), {"highway": "secondary"}),
            (20, list(loop) + [200], {"highway": "secondary", "oneway": "yes", "layer": "-1"}),
            (21, [100, 101], {"highway": "tertiary", "bridge": "yes", "layer": "1"})]
    g = osm(nodes, ways)
    r = graph.route(g, np.array([[10.0, 0.0], [990.0, 0.0]]))
    found = crossings.find(g, r.xy, r.seg_way)["crossings"]
    short = [c for c in found if c["way_id"] == 21][0]["road"]
    assert short["ends"] == ["dead_end", "dead_end"]
    assert [(w["way_id"], w["from_m"], w["to_m"]) for w in short["ways"]] == [(21, 0.0, 80.0)]  # one entry, whole
    assert crossings.cumulative(np.asarray(short["line"]))[-1] == pytest.approx(80.0, abs=0.01)
    looped = [c for c in found if c["way_id"] == 20]
    assert len(looped) == 2  # the loop crosses the route twice
    for c in looped:  # round the loop (186 m) both ways from each crossing, 150 m each way
        assert crossings.cumulative(np.asarray(c["road"]["line"]))[-1] == pytest.approx(300.0, abs=0.01)
        assert c["road"]["ends"] == ["length", "length"] and {w["way_id"] for w in c["road"]["ways"]} == {20}


# -- the plan -----------------------------------------------------------------------------------------------------------

def test_a_road_under_the_route_s_bridge_is_planned_40_m_past_the_deck_either_side_from_the_dem():
    c, sec = road(z=10.0, z_raw=np.where(np.abs(np.arange(1001) - 500) < 30, 4.0, 10.0))  # a 6 m gap under the deck
    p = cr.plan([crossing("under")], c, sec, flat(4.0))
    (r,) = p["roads"]
    assert (r["kind"], r["deck"], r["highway"], r["style"], r["width_m"]) == ("under", None, "secondary", "road", 6.4)
    assert r["gap_m"] == pytest.approx(6.0)
    # From 40 m past the deck's edge (the road's edge and 1.5 m) on one side to the same on the other.
    assert r["length_m"] == pytest.approx(2 * (HALF + cr.DECK_EDGE_M + cr.REACH_M), abs=2.5)
    xy = np.asarray(r["rows"]["xy"])
    assert np.diff(r["rows"]["s"]) == pytest.approx(cr.ROW_M, abs=0.05)
    assert np.abs(xy[:, 0] - 500.0).max() < 0.05 and xy[0, 1] < -40 and xy[-1, 1] > 40
    assert np.asarray(r["rows"]["z"]) == pytest.approx(4.0, abs=0.01)
    assert p["decks"] == [] and p["skipped"] == []


def test_a_road_under_a_bridge_the_dem_shows_no_gap_under_is_left_out_as_predating_it():
    c, sec = road(z=10.0)  # the DEM under the "bridge" is the deck's own height: old ground
    p = cr.plan([crossing("under")], c, sec, flat(10.0))
    assert p["roads"] == []
    (k,) = p["skipped"]
    assert (k["reason"], k["relation"], k["gap_m"]) == ("dem_predates", "under", 0.0)
    (w,) = cr.warnings(p)
    assert w == ("Cross Road (secondary) under the route's bridge at 0.50 km: the elevation data predates this bridge, "
                 "so the road under it is left out (the route's deck stands 0.0 m above the ground under it).")


def over_plan(top=7.0, low=0.0, bridge=(125.0, 175.0), deg=90.0, **kw):
    c, sec = road(z=low)
    ways = [(0.0, bridge[0], False), (bridge[0], bridge[1], True), (bridge[1], 300.0, False)]
    return cr.plan([crossing("over", bridge=True, road_ways=ways, deg=deg, **kw)], c, sec,
                   embankment(top, low, deg=deg))


def test_a_bridge_over_the_route_is_drawn_where_its_underside_clears_the_route_by_4_6_m():
    p = over_plan()
    (d,) = p["decks"]
    assert d["length_m"] == pytest.approx(50.0, abs=1.0)
    top = np.asarray(d["rows"]["z"])
    assert top == pytest.approx(7.0, abs=0.05)  # straight between the road's heights past its ends
    assert d["clearance_m"] == pytest.approx(7.0 - cr.DECK_DEPTH_M, abs=0.1)
    assert d["deck_half_m"] == pytest.approx(3.2 + cr.KERB_M + cr.PARAPET_W)
    # The abutments' faces stand where the embankment comes up under the deck, 20 m either side of the route.
    xy = np.asarray(d["rows"]["xy"])
    faces = [abs(xy[a["face_row"], 1]) for a in d["abutments"]]
    assert faces == pytest.approx([21.0, 21.0], abs=1.5)
    # A 42 m span between them gets a pier, moved off the middle to stand clear of the route's road.
    (pier,) = d["piers"]
    cols = np.asarray(pier["columns"])
    assert len(cols) == 2 and (np.abs(cols[:, 1]) - cr.PIER_R >= HALF + cr.PIER_CLEAR_M).all()
    # The approaches: 40 m past each end, sharing the deck's end rows exactly.
    a0, a1 = p["roads"]
    assert (a0["kind"], a0["deck"], a1["kind"]) == ("approach", 0, "approach")
    assert a0["length_m"] == pytest.approx(40.0) and a1["length_m"] == pytest.approx(40.0)
    assert a0["rows"]["xy"][-1] == d["rows"]["xy"][0] and a0["rows"]["z"][-1] == pytest.approx(d["rows"]["z"][0])
    assert a1["rows"]["xy"][0] == d["rows"]["xy"][-1] and a1["rows"]["z"][0] == pytest.approx(d["rows"]["z"][-1])
    assert d["approaches"] == [a0["id"], a1["id"]]


def test_a_low_bridge_is_left_out_and_one_the_dem_shows_no_gap_under_predates_the_elevation_data():
    low = over_plan(top=5.0)
    assert low["decks"] == [] and low["roads"] == []
    (k,) = low["skipped"]
    assert k["reason"] == "low_clearance" and k["clearance_m"] == pytest.approx(5.0 - 1.3, abs=0.1)
    assert cr.warnings(low) == ["Cross Road (secondary) over the route at 0.50 km: its underside would clear the route "
                                "by only 3.7 m, under the 4.6 m needed, so its deck is left out."]
    old = over_plan(top=0.0)
    (k,) = old["skipped"]
    assert k["reason"] == "dem_predates" and k["dem_gap_m"] == pytest.approx(0.0, abs=0.01)
    (w,) = cr.warnings(old)
    assert w.startswith("Cross Road (secondary) over the route at 0.50 km: the elevation data predates this bridge, so "
                        "its deck is left out (its underside would clear the route by -1.3 m")


def judged(verdict, crossing_ids=(0,), relation="over", separation=-0.7):
    """The profile stage's dem_age.json with one bridge judged (`core.dem_age`)."""
    from mapcreator.core import dem_age

    b = dem_age.Bridge(crossings=list(crossing_ids), relation=relation, way_id=7, route_way_id=1, along_m=500.0,
                       label="Cross Road (secondary) over the route at 0.50 km", verdict=verdict, upper_m=0.0,
                       lower_m=0.7, separation_m=separation)
    return dem_age.Report([b], 0, 0).as_dict()


def test_a_bridge_the_profile_stage_finds_the_elevation_data_doesn_t_hold_is_left_out_on_its_word():
    from mapcreator.core import dem_age

    c, sec = road(z=0.0)
    ways = [(0.0, 125.0, False), (125.0, 175.0, True), (175.0, 300.0, False)]
    found = [crossing("over", bridge=True, road_ways=ways)]
    # Drawn on the DEM's own word...
    assert len(cr.plan(found, c, sec, embankment(), dem_ages=judged("holds", separation=7.0))["decks"]) == 1
    # ...and left out on the profile stage's, without measuring again, worded as it words it.
    for verdict, coarse in (("predates", False), ("coarse", True)):
        p = cr.plan(found, c, sec, embankment(), dem_ages=judged(verdict))
        (k,) = p["skipped"]
        assert p["decks"] == [] and k["reason"] == ("dem_coarse" if coarse else "dem_predates") and k["dem_age"]
        assert "clearance_m" not in k and cr.dem_covered(p) == {0}
        detail = dem_age.Bridge(**judged(verdict)["bridges"][0]).detail()
        assert cr.warnings(p) == [dem_age.warning("Cross Road (secondary) over the route at 0.50 km", coarse=coarse,
                                                  then="its deck is left out", detail=detail)]
    # A low deck the DEM holds is the plan's own call.
    low = cr.plan(found, c, sec, embankment(top=5.0), dem_ages=judged("holds", separation=5.0))
    assert [k["reason"] for k in low["skipped"]] == ["low_clearance"] and cr.dem_covered(low) == set()
    # Without a verdict on it (a bridge it couldn't judge), the plan measures it.
    p = cr.plan(found, c, sec, flat(0.0), dem_ages=judged("unjudged", separation=None))
    assert [k["reason"] for k in p["skipped"]] == ["dem_predates"] and not p["skipped"][0].get("dem_age")


def test_a_highway_s_two_decks_left_out_share_one_line():
    c, sec = road(z=0.0)
    found = [crossing("over", bridge=True, cid=k, way_id=10 * (k + 1), x=500.0 + 3 * k, name="Huon Highway",
                      highway="trunk") for k in range(2)]
    p = cr.plan(found, c, sec, flat(0.0), dem_ages=judged("predates", crossing_ids=(0, 1)))
    assert len(p["skipped"]) == 2 and cr.dem_covered(p) == {0, 1}
    (w,) = cr.warnings(p)  # named as the profile stage names the bridge
    assert w.startswith("Cross Road (secondary) over the route at 0.50 km (2 decks): the elevation data predates this "
                        "bridge, so their decks are left out (it puts the two roads -0.7 m apart")
    # A loop under the route's bridge, crossing it twice, is one bridge: one line, no count.
    loop = [crossing("under", cid=k, way_id=30, x=500.0 + 10 * k, name="Loop Road") for k in range(2)]
    c2, sec2 = road(z=10.0, z_raw=np.where(np.abs(np.arange(1001) - 505) < 30, 4.0, 10.0))
    p = cr.plan(loop, c2, sec2, flat(4.0), dem_ages=judged("predates", crossing_ids=(0, 1), relation="under"))
    (w,) = cr.warnings(p)
    assert ": the elevation data predates this bridge, so the road under it is left out (" in w and "(2 " not in w


def test_the_validate_stage_leaves_a_left_out_deck_to_the_crossing_roads_warning(tmp_path):
    from mapcreator.core import dem_age
    from mapcreator.pipeline import profile_stage, validate_stage

    work = WorkDir(tmp_path / "work")
    deck = dem_age.Bridge(**judged("predates", crossing_ids=(0,))["bridges"][0])
    other = dem_age.Bridge(crossings=[3], relation="under", way_id=9, route_way_id=1, along_m=900.0,
                           label="Other Road (tertiary) under the route's bridge at 0.90 km", verdict="predates",
                           upper_m=0.0, lower_m=0.2, separation_m=-0.2)
    work.write_json(profile_stage.STAGE, dem_age.FILE, dem_age.Report([deck, other], 0, 0).as_dict())
    assert len(validate_stage._dem_age_warnings(work)) == 2
    (w,) = validate_stage._dem_age_warnings(work, {0})
    assert w.startswith("Other Road (tertiary) under the route's bridge at 0.90 km: the elevation data predates")


def test_a_road_over_the_route_that_isn_t_a_bridge_or_runs_past_the_data_is_left_out():
    c, sec = road(z=0.0)
    p = cr.plan([crossing("over", bridge=False)], c, sec, embankment())
    assert [k["reason"] for k in p["skipped"]] == ["not_a_bridge"]
    p = cr.plan([crossing("over", bridge=True)], c, sec, embankment())  # bridge from end to end of the line
    assert [k["reason"] for k in p["skipped"]] == ["long_bridge"]
    p = cr.plan([crossing("over", bridge=True) | {"route": {"tunnel": True}}], c, sec, embankment())
    assert p["decks"] == [] and p["skipped"] == []  # over the route's own tunnel: nothing to draw
    # A minimal record, as from a route stage before roads were followed, doesn't stop the plan.
    p = cr.plan([{"at_m": 500.0, "xy": [500.0, 0.0], "relation": "over", "route": {"tunnel": False}}], c, sec,
                embankment())
    assert p["decks"] == [] and [k["reason"] for k in p["skipped"]] == ["not_a_bridge"]


def test_a_skewed_bridge_measures_its_clearance_over_the_whole_road():
    p = over_plan(deg=50.0, bridge=(110.0, 190.0))
    (d,) = p["decks"]
    assert d["clearance_m"] == pytest.approx(7.0 - 1.3, abs=0.1)
    for pier in d["piers"]:
        cols = np.asarray(pier["columns"])
        assert (np.abs(cols[:, 1]) - cr.PIER_R >= HALF + cr.PIER_CLEAR_M - 1e-6).all()


# -- the ribbons and the terrain ----------------------------------------------------------------------------------------

def route_ground(c, sec, dem):
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], dem)
    return rb, shapely.Polygon(mesh._outline(rb)[0]).buffer(0)


def built_scene(p, c, sec, dem, *, taken=None, skirt=True):
    rb, ground = route_ground(c, sec, dem)
    b = cr.build(p, ground if taken is None else taken, FRAME, dem, skirt=skirt, others_xy=c["xy"])
    fp = cr.footprint(None, b, len(c["s"]))
    meshes = mesh.ribbon_meshes(rb, c["s"], FRAME, skirt=skirt) + b.meshes
    if skirt:
        meshes += mesh.terrain_meshes(rb, 150.0, dem, FRAME, extra=fp)
    else:
        meshes += mesh.terrain_meshes(rb, 150.0, dem, FRAME, extra=fp, fold_skirt=True)
    return rb, b, meshes


def to_ac(q):
    q = np.asarray(q, float)
    return FRAME.to_ac(q[:, 0], q[:, 1], np.zeros(len(q)))[:, [0, 2]]


def checks_for(p, b):
    """Each piece as the validator checks it: drawn along its centreline, and no hole within 3 m of its ribbon."""
    out = []
    by_id = {r["id"]: r for r in p["roads"]}
    for piece in b.pieces:
        a, z = piece["rows"]
        line = np.asarray(by_id[piece["road"]]["rows"]["xy"], float)[a + 1:z]
        ground = shapely.Polygon(to_ac(piece["outline"])).buffer(0)
        out.append((piece["road"], validate.StubCheck(ground, to_ac(line), np.zeros((0, 2, 2)))))
    return out


def holes(meshes, centre, radius):
    """Points of a 0.25 m grid within `radius` of `centre` (east, north) that no road or ground mesh covers."""
    tris = [np.asarray(m.positions, float)[m.triangles] for m in meshes
            if m.name[:1].isdigit() and any(k in m.name for k in ("ROAD", "GRAVEL", "GRASS"))]
    cover = validate._Cover(np.concatenate(tris))
    g = np.arange(-radius, radius + 1e-9, 0.25)
    gx, gy = np.meshgrid(centre[0] + g, centre[1] + g)
    pts = to_ac(np.column_stack([gx.ravel(), gy.ravel()]))
    return int(np.isnan(cover.top(pts)).sum())


@pytest.mark.parametrize("skirt", [True, False])
def test_a_road_under_the_route_s_bridge_stops_short_of_today_s_embankment_with_the_terrain_round_it(skirt):
    c, sec = road(z=10.0, z_raw=np.where(np.abs(np.arange(1001) - 500) < 30, 4.0, 10.0))
    dem = flat(4.0)
    p = cr.plan([crossing("under")], c, sec, dem)
    rb, b, meshes = built_scene(p, c, sec, dem, skirt=skirt)
    assert len(b.pieces) == 2 and b.left_out == []  # one each side of the route's ground
    _, ground = route_ground(c, sec, dem)
    for r in b.ribbons:
        poly = shapely.Polygon(mesh._outline(r)[0])
        assert poly.distance(ground) >= cr.CLEAR_M - 1e-6
    names = {m.name for m in b.meshes}
    assert names == {"1ROAD_X001", "1GRASS_X001"}
    faults = validate.stub_failures(validate.mesh_nodes(meshes), checks_for(p, b))
    assert all(not f for f in faults.values()), faults
    assert holes(meshes, (500.0, 0.0), 60.0) == 0


def test_with_the_route_s_ground_open_under_its_deck_the_road_under_it_is_drawn_whole():
    """Stage 1 takes the route's span out of the terrain's footprint: given ground with room under the deck, the road
    under it carries on in one piece."""
    c, sec = road(z=10.0, z_raw=np.where(np.abs(np.arange(1001) - 500) < 30, 4.0, 10.0))
    dem = flat(4.0)
    p = cr.plan([crossing("under")], c, sec, dem)
    _, ground = route_ground(c, sec, dem)
    open_ground = ground.difference(shapely.box(470.0, -50.0, 530.0, 50.0))
    b = cr.build(p, open_ground, FRAME, dem, skirt=True)
    (piece,) = b.pieces
    assert piece["length_m"] == pytest.approx(p["roads"][0]["length_m"], abs=0.5)


def test_crossing_roads_running_side_by_side_keep_the_major_one_and_say_why_the_other_isn_t_drawn():
    c, sec = road(z=10.0, z_raw=np.where(np.abs(np.arange(1001) - 500) < 30, 4.0, 10.0))
    dem = flat(4.0)
    ramp = crossing("under", x=503.0, cid=1, way_id=20, highway="trunk_link", name=None)
    main = crossing("under", x=500.0, cid=0, way_id=10, highway="trunk", name="Southern Outlet")
    p = cr.plan([main, ramp], c, sec, dem)
    _, ground = route_ground(c, sec, dem)
    b = cr.build(p, ground, FRAME, dem, skirt=True)
    assert {piece["road"] for piece in b.pieces} == {0}
    (gone,) = b.left_out
    assert gone["highway"] == "trunk_link" and gone["why"] == "it runs too close beside another road drawn there"
    built = cr.as_built(p, b)
    assert [r["id"] for r in built["roads"]] == [0] and len(built["roads"][0]["pieces"]) == 2
    assert any(w.endswith(": it runs too close beside another road drawn there, so it isn't drawn.")
               for w in cr.warnings(p, built))


def test_a_bridge_s_approaches_join_the_terrain_with_no_hole_and_nothing_over_them():
    p = over_plan()
    c, sec = road(z=0.0)
    dem = embankment()
    rb, b, meshes = built_scene(p, c, sec, dem)
    assert len(b.pieces) == 2 and all(piece["length_m"] == pytest.approx(40.0) for piece in b.pieces)
    faults = validate.stub_failures(validate.mesh_nodes(meshes), checks_for(p, b))
    assert all(not f for f in faults.values()), faults
    assert holes(meshes, (500.0, 0.0), 70.0) == 0
    # The first row's road vertices are terrain points too (a stub's meet its throat there). Nothing else is added
    # along the outline: the terrain keeps every outline edge as it is (`mesh._constrain`).
    assert len(b.edges) == 2 and all(len(pts) == 5 for pts, _ in b.edges)


def test_the_footprint_keeps_the_stubs_and_rings_pieces_and_adds_the_crossing_roads():
    c, sec = road(z=0.0)
    p = over_plan()
    _, ground = route_ground(c, sec, embankment())
    b = cr.build(p, ground, FRAME, embankment(), skirt=True)
    rows = len(c["s"])
    other = mesh.Footprint({"L": np.zeros(rows - 1, bool), "R": np.zeros(rows - 1, bool)}, [b.ribbons[0]], [],
                           [], ribbon_cuts=[None], open_last=[True])
    fp = cr.footprint(other, b, rows)
    assert len(fp.ribbons) == 3 and fp.ribbon_cuts == [None, None, None] and fp.open_last == [True, False, False]
    assert other.ribbons == [b.ribbons[0]]  # not changed in place
    assert cr.footprint(None, cr.Built([], [], [], [], []), rows) is None


# -- the decks ------------------------------------------------------------------------------------------------------------

def deck_scene():
    p = over_plan()
    c, sec = road(z=0.0)
    ground = cr.GroundUnderDecks(type("Dem", (), {"sample": staticmethod(embankment())})(), p["decks"])
    ms, stats = cr.decks(p, FRAME, ground.sample)
    return p, c, sec, ms, stats


def test_a_deck_is_visual_with_invisible_colliders_on_its_piers_and_abutments():
    p, c, sec, ms, stats = deck_scene()
    names = [m.name for m in ms]
    assert names == ["BRIDGE_X0_ROAD", "BRIDGE_X0_CONCRETE", "BRIDGE_X0_SOFFIT", "BRIDGE_X0_RAIL", "1WALL_PIER_X0",
                     "1WALL_ABUT_X0"]
    assert stats == {"decks": 1, "meshes": 6, "vertices": sum(len(m.positions) for m in ms), "piers": 1}
    for m in ms:
        m.validate()
        assert m.renderable == (not m.name[0].isdigit())
    assert {m.material for m in ms} == {"road", cr.CONCRETE, cr.SOFFIT, cr.RAIL_MATERIAL}
    assert stats["vertices"] < 4000  # about the research's estimate for a 50 m deck


def test_a_deck_s_faces_point_outwards_and_its_underside_down():
    _, _, _, ms, _ = deck_scene()
    by = {m.name: m for m in ms}
    for name, check in (("BRIDGE_X0_SOFFIT", lambda g: g[:, 1] < 0), ("BRIDGE_X0_ROAD", lambda g: g[:, 1] > 0)):
        m = by[name]
        p = m.positions[m.triangles].astype(float)
        g = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        assert check(g).all(), name
    # Every face's winding agrees with its normals.
    for m in ms:
        p = m.positions[m.triangles].astype(float)
        g = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        nrm = m.normals[m.triangles].astype(float).mean(axis=1)
        assert ((g * nrm).sum(axis=1) > 0).all(), m.name


def test_a_deck_clears_the_route_s_road_and_its_colliders_stand_off_it():
    p, c, sec, ms, _ = deck_scene()
    rb, _ = route_ground(c, sec, embankment())
    road_meshes = mesh.ribbon_meshes(rb, c["s"], FRAME)
    data = validate.mesh_nodes(road_meshes + ms)
    rep = validate.Report()
    validate.crossing_checks(rep, "main", data, validate.CrossingCheck((), 1))
    assert rep.failed == [] and any("clear its road by 4.6 m" in s for s in rep.passed), rep.passed
    surface = validate.RoadSurface(data["nodes"])
    for m in ms:
        if m.name.startswith("1WALL"):
            assert not surface.covers(np.asarray(m.positions, float)[:, [0, 2]]).any(), m.name
    # A deck lowered into the route's path fails the check.
    low = [type(m)(m.name, m.material, m.positions - np.array([0, 4.0, 0], np.float32), m.normals, m.uvs,
                   m.triangles, m.renderable) for m in ms]
    rep = validate.Report()
    validate.crossing_checks(rep, "main", validate.mesh_nodes(road_meshes + low), validate.CrossingCheck((), 1))
    assert rep.failed and "clear its road by 4.6 m" in rep.failed[0]


def test_the_ground_is_held_under_a_deck_s_open_span_and_its_ends_and_left_alone_elsewhere():
    p = over_plan()
    d = p["decks"][0]
    dem = embankment(top=7.0)
    ground = cr.GroundUnderDecks(type("Dem", (), {"sample": staticmethod(dem), "resolution_m": 2.0})(), p["decks"])
    assert ground.resolution_m == 2.0  # everything else is the DEM's
    under = 7.0 - cr.DECK_DEPTH_M
    # Under the open span the embankment is cut down below the underside; under the ends, below the top.
    assert ground.sample(np.array([500.0]), np.array([19.0]))[0] == pytest.approx(0.0)
    assert ground.sample(np.array([500.0]), np.array([22.0]))[0] <= under - cr.UNDER_CLEAR_M + 1e-6 + \
        (cr.TOP_CLEAR_M - cr.UNDER_CLEAR_M + cr.DECK_DEPTH_M)
    assert ground.sample(np.array([500.0]), np.array([24.5]))[0] == pytest.approx(7.0 - cr.TOP_CLEAR_M, abs=0.05)
    # Beyond the deck's edges (and the margin), and past its ends, the DEM's own.
    half = d["deck_half_m"]
    assert ground.sample(np.array([500.0 + half + cr.CLAMP_BEYOND_M + 1.0]), np.array([22.0]))[0] == 7.0
    assert ground.sample(np.array([500.0]), np.array([40.0]))[0] == 7.0


def test_the_scenery_zone_takes_in_the_roads_and_the_decks():
    p = over_plan()
    z = cr.zone(p, 4.0)
    assert z.contains(shapely.Point(500.0, 0.0))  # under the deck
    assert z.contains(shapely.Point(500.0 + 3.2 + 3.9, 50.0))  # beside an approach, within the verge and 2 m
    assert not z.contains(shapely.Point(500.0 + 3.2 + 4.5, 50.0))
    assert cr.zone(None, 4.0) is None and cr.centres(None) is None
    assert len(cr.centres(p)) == sum(len(r["rows"]["xy"]) for r in p["roads"])


# -- the section stage ----------------------------------------------------------------------------------------------------

class _Dem:
    resolution_m = 2.0

    def load_corridor(self, xy, m):
        return 1

    def sample(self, x, y):
        return embankment(top=7.0, low=0.0)(x, y)


class _Region:
    epsg = 28355

    def elevation(self, settings):
        return _Dem()


def _work(tmp_path, found):
    work = WorkDir(tmp_path / "work")
    n = 1001
    s = np.arange(n, dtype=float)
    np.savez_compressed(work.stage("02_profile") / "centreline.npz", s=s, xy=np.column_stack([s, np.zeros(n)]),
                        heading=np.zeros(n), curvature=np.zeros(n), z=np.zeros(n), z_raw=np.zeros(n),
                        grade=np.zeros(n), run=np.zeros(n, np.int64), fine=np.ones(n, bool),
                        bridge=np.zeros(n, bool), tunnel=np.zeros(n, bool), under=np.zeros(n, bool))
    work.write_json(route_stage.STAGE, "runs.json",
                    [{"from_m": 0.0, "to_m": 1000.0, "way_id": 1, "tags": {"highway": "secondary"}}])
    np.savez_compressed(work.stage(route_stage.STAGE) / "route.npz", xy=np.column_stack([s, np.zeros(n)]),
                        seg_run=np.zeros(n - 1, np.int64), epsg=28355)
    work.write_json(route_stage.STAGE, route_stage.CROSSINGS, {"version": 1, "crossings": found, "unresolved": []})
    return work


def test_the_section_stage_plans_crossing_roads_whatever_the_junctions_option_and_removes_a_stale_plan(
        tmp_path, monkeypatch):
    monkeypatch.setattr(section_stage, "job_region", lambda waypoints: _Region())
    ways = [(0.0, 125.0, False), (125.0, 175.0, True), (175.0, 300.0, False)]
    found = [crossing("over", bridge=True, road_ways=ways)]
    work = _work(tmp_path, found)
    job = Job(track_id="t", waypoints=[{"lon": 147.301, "lat": -42.8001}, {"lon": 147.319, "lat": -42.7999}],
              junctions="none")
    out = section_stage.run(job, config.load(), work)
    assert out["crossing_roads"] == {"roads": {"under": 0, "approach": 2}, "decks": 1}
    doc = section_stage.crossing_roads(work)
    assert doc["version"] == cr.FORMAT_VERSION and doc["crs"] == "EPSG:28355" and len(doc["decks"]) == 1
    work.write_json(route_stage.STAGE, route_stage.CROSSINGS, {"version": 1, "crossings": [], "unresolved": []})
    out = section_stage.run(job, config.load(), work)
    assert "crossing_roads" not in out and section_stage.crossing_roads(work) is None


# -- a real low bridge: drawn at the height the route is signed for (the maintainer, 18 Nov 2026) ---------------------

def signed_plan(top, maxheight, dem_ages=None):
    c, sec = road(z=0.0)
    ways = [(0.0, 125.0, False), (125.0, 175.0, True), (175.0, 300.0, False)]
    x = crossing("over", bridge=True, road_ways=ways)
    x["route"]["maxheight"] = maxheight
    return cr.plan([x], c, sec, embankment(top, 0.0), dem_ages=dem_ages)


def test_a_low_bridge_the_route_is_signed_for_is_drawn_at_that_height_with_a_thinner_deck():
    p = signed_plan(5.0, "3.8")  # the DEM: 3.7 m under a 1.3 m deck; the sign: 3.8 m
    (d,) = p["decks"]
    assert (d["height_from"], d["signed_m"], d["signed_on"], d["required_clearance_m"]) == ("signed", 3.8, "route", 3.8)
    assert d["clearance_m"] == pytest.approx(3.8, abs=0.01) and d["depth_m"] == pytest.approx(1.2, abs=0.05)
    assert np.asarray(d["rows"]["z"]) == pytest.approx(5.0, abs=0.05)  # its top where the road's heights put it
    assert p["skipped"] == [] and len(p["roads"]) == 2
    (w,) = [w for w in cr.warnings(p) if "drawn over the route" in w]
    assert "3.8 m clear (drawn at its signed height, 3.8 m)" in w
    assert cr.summary(p)["drawn_at_signed_height"] == ["Cross Road (secondary) over the route at 0.50 km, 3.8 m"]


def test_a_low_bridge_the_road_s_heights_put_too_low_is_raised_over_the_route_and_eases_back_by_its_ends():
    p = signed_plan(4.0, "3.8")
    (d,) = p["decks"]
    assert d["clearance_m"] == pytest.approx(3.8, abs=0.01) and d["depth_m"] == cr.MIN_DEPTH_M
    assert d["raised_m"] == pytest.approx(0.3, abs=0.05)
    z = np.asarray(d["rows"]["z"])
    assert z.max() == pytest.approx(4.0 + d["raised_m"], abs=0.01)
    assert z[0] == pytest.approx(4.0, abs=0.05) and z[-1] == pytest.approx(4.0, abs=0.05)
    a0, a1 = p["roads"]  # the road beyond meets it exactly
    assert a0["rows"]["z"][-1] == pytest.approx(z[0]) and a1["rows"]["z"][0] == pytest.approx(z[-1])
    # The DEM too far below the sign for any deck: the data's likely older than the bridge.
    (k,) = signed_plan(1.0, "3.8")["skipped"]
    assert k["reason"] == "dem_predates" and k["signed_m"] == 3.8


def test_a_sign_under_the_floor_is_taken_for_a_slip_and_a_bridge_without_one_is_left_out_as_before():
    p = signed_plan(5.0, "1.8")
    (k,) = p["skipped"]
    assert k["reason"] == "low_clearance" and p["decks"] == []
    assert "signed for only 1.8 m there: under 2 m, a sign is taken for a slip" in cr.warnings(p)[0]
    (k,) = signed_plan(5.0, "default")["skipped"]  # not a height
    assert k["reason"] == "low_clearance" and "signed_m" not in k


def test_a_signed_low_deck_passes_the_validator_at_its_own_height_and_its_piers_colliders_stop_under_it():
    p = signed_plan(4.0, "3.8")
    c, sec = road(z=0.0)
    ground = cr.GroundUnderDecks(type("Dem", (), {"sample": staticmethod(embankment(4.0))})(), p["decks"])
    ms, _ = cr.decks(p, FRAME, ground.sample)
    rb, _ = route_ground(c, sec, embankment(4.0))
    data = validate.mesh_nodes(mesh.ribbon_meshes(rb, c["s"], FRAME) + ms)
    rep = validate.Report()
    validate.crossing_checks(rep, "main", data, validate.CrossingCheck((), 1))
    assert rep.failed  # against 4.6 m
    rep = validate.Report()
    validate.crossing_checks(rep, "main", data, validate.CrossingCheck((), 1), {"BRIDGE_X0_": 3.8})
    assert not rep.failed, rep.failed
    d = p["decks"][0]
    under = np.asarray(d["rows"]["z"]) - d["depth_m"]
    for m in ms:
        if m.name.startswith("1WALL_PIER"):
            assert m.positions[:, 1].max() <= under.max() - cr.HEADSTOCK[1] + 1e-3


# -- a bridge signed 4.6 m or more that the road's heights put too low (the maintainer, 18 Nov 2026: "yes") -----------

def test_a_bridge_signed_over_4_6_m_the_dem_can_t_place_is_raised_to_its_sign_with_its_full_depth():
    p = signed_plan(5.0, "5.2")  # the road's heights: 3.7 m under a 1.3 m deck; the sign: 5.2 m
    (d,) = p["decks"]
    assert (d["height_from"], d["signed_m"], d["signed_on"]) == ("signed", 5.2, "route")
    assert d["clearance_m"] == pytest.approx(5.2, abs=0.01) and d["depth_m"] == cr.DECK_DEPTH_M
    assert d["required_clearance_m"] == cr.CLEARANCE_M and d["raised_m"] == pytest.approx(1.5, abs=0.05)
    z = np.asarray(d["rows"]["z"])
    assert z.max() == pytest.approx(5.0 + d["raised_m"], abs=0.01)
    a0, a1 = p["roads"]  # eased back to the road's own heights by its ends: the road beyond meets it exactly
    assert a0["rows"]["z"][-1] == pytest.approx(z[0]) and a1["rows"]["z"][0] == pytest.approx(z[-1])
    assert z[0] == pytest.approx(5.0, abs=0.05) and z[-1] == pytest.approx(5.0, abs=0.05)
    assert any("(drawn at its signed height, 5.2 m)" in w for w in cr.warnings(p))


def test_a_signed_bridge_too_far_from_the_dem_or_one_the_profile_stage_doesn_t_hold_stays_left_out():
    (k,) = signed_plan(3.5, "5.2")["skipped"]  # 2.9 m to raise, more than MAX_RAISE_M: data older than the bridge
    assert k["reason"] == "dem_predates" and k["signed_m"] == 5.2
    (w,) = cr.warnings(signed_plan(3.5, "5.2"))
    assert "the elevation data predates this bridge, so its deck is left out" in w
    p = signed_plan(5.0, "5.2", dem_ages=judged("predates", separation=3.5))  # left out on the profile stage's word
    assert p["decks"] == [] and p["skipped"][0]["reason"] == "dem_predates" and p["skipped"][0].get("dem_age")


def test_a_bridge_that_already_clears_4_6_m_is_drawn_as_the_dem_gives_it_whatever_its_sign():
    for sign in ("6.5", "4.8", None):
        (d,) = signed_plan(7.0, sign)["decks"]
        assert d["clearance_m"] == pytest.approx(7.0 - cr.DECK_DEPTH_M, abs=0.1) and "height_from" not in d


def test_a_pier_keeps_off_a_side_road_s_stub_and_a_roundabout_s_ring():
    (pier,) = over_plan()["decks"][0]["piers"]
    cols = np.asarray(pier["columns"])
    # A side road's stub paved right where the pier would stand: it moves along the deck, clear of it.
    stub = np.asarray(shapely.box(*(cols.min(axis=0) - 1.0), *(cols.max(axis=0) + 1.0)).exterior.coords)[:-1]
    c, sec = road(z=0.0)
    ways = [(0.0, 125.0, False), (125.0, 175.0, True), (175.0, 300.0, False)]
    p = cr.plan([crossing("over", bridge=True, road_ways=ways)], c, sec, embankment(7.0, 0.0), keep_off=[stub])
    (moved,) = p["decks"][0]["piers"]
    assert moved["row"] != pier["row"]
    clear = shapely.Polygon(stub).buffer(cr.PIER_R + cr.PIER_OFF_ROADS_M)
    assert not shapely.intersects_xy(clear, *np.asarray(moved["columns"]).T).any()
