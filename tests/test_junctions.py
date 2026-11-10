"""Side-road junctions: which side roads count (tags, angle, bridges, private roads, the rest of the route), the
mouths they leave on the mainline, and keeping guard rails, furniture, trees, buildings, power poles and timing gates
out of them; and `junctions: none`, which builds as before."""

import numpy as np
import pytest
import shapely
from pyproj import Transformer

from mapcreator import config
from mapcreator.core import furniture, junctions, layout, power, roadside, scenery, validate
from mapcreator.core.layout import RoadModel, logic_dummies
from mapcreator.core.scene import LocalFrame
from mapcreator.job import Job, Junctions
from mapcreator.pipeline import route_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.osm import graph, overpass
from mapcreator.service.db import job_dict, route_hash

IDENT = Transformer.from_crs(4326, 4326, always_xy=True)  # the test networks are drawn in metres
FRAME = LocalFrame(0.0, 0.0, 0.0)
HALF = 3.2
RESIDENTIAL = {"highway": "residential"}


def osm(nodes: dict[int, tuple[float, float]], ways: list[tuple[int, list[int], dict]]):
    els = [{"type": "node", "id": i, "lon": x, "lat": y} for i, (x, y) in nodes.items()]
    els += [{"type": "way", "id": w, "nodes": refs, "tags": tags} for w, refs, tags in ways]
    return graph.build({"elements": els}, IDENT)


def network(side_roads: list[tuple[int, list, dict]], main: dict | None = None):
    """A secondary road east along y = 0, 0 to 1,000 m with a node every 100 m (ids 0-10), and side roads, each
    leaving the main road's node at x through its points: (x, y) for a new node, or a main road node's id. Way ids:
    main 1, side roads 100, 101, ..."""
    nodes = {i: (100.0 * i, 0.0) for i in range(11)}
    ways = [(1, list(range(11)), main or {"highway": "secondary"})]
    nid = 1000
    for k, (x, points, tags) in enumerate(side_roads):
        refs = [x // 100]
        for p in points:
            if isinstance(p, int):
                refs.append(p)
                continue
            nodes[nid] = p
            refs.append(nid)
            nid += 1
        ways.append((100 + k, refs, tags))
    return osm(nodes, ways)


def find(g, start=10.0, end=990.0):
    return junctions.find(g, graph.route(g, np.array([[start, 0.0], [end, 0.0]])).xy)


def north(x, length=150.0):
    return [(x, length / 3), (x, 2 * length / 3), (x, length)]


def at_angle(x, deg, length=150.0):
    a = np.radians(deg)
    return [(x + d * np.cos(a), d * np.sin(a)) for d in (length / 3, 2 * length / 3, length)]


def extend(g, start, points, tags, way_id=99):
    """The network with one more way, from the node at `start` through new nodes at `points`."""
    a = int(np.flatnonzero((g.xy == start).all(axis=1))[0])
    refs = [a] + list(range(len(g.xy), len(g.xy) + len(points)))
    k = len(refs) - 1
    return graph.RoadGraph(xy=np.vstack([g.xy, np.array(points, float)]),
                           osm_node=np.concatenate([g.osm_node, 5000 + np.arange(len(points))]),
                           seg_a=np.concatenate([g.seg_a, refs[:-1]]), seg_b=np.concatenate([g.seg_b, refs[1:]]),
                           seg_way=np.concatenate([g.seg_way, np.full(k, len(g.ways))]),
                           seg_fwd=np.concatenate([g.seg_fwd, np.ones(k, bool)]),
                           seg_rev=np.concatenate([g.seg_rev, np.ones(k, bool)]),
                           ways=g.ways + [{"id": way_id, "tags": tags}])


# -- which junctions count --------------------------------------------------------------------------------------

def test_the_router_s_classes_are_the_junction_classes():
    assert junctions.DRIVABLE == set(overpass.DRIVABLE.split("|"))


def test_a_public_side_road_of_a_drivable_class_is_a_junction_with_its_side_angle_and_closure():
    res = find(network([(300, north(300), {"highway": "residential", "name": "Kerria Road", "surface": "asphalt"}),
                        (500, [(500.0, -60.0), (500.0, -120.0)], {"highway": "tertiary"})]))
    j = res["junctions"]
    assert [(x["id"], x["at_m"], x["side"], x["closure"]) for x in j] == [(0, 290.0, "left", "minor"),
                                                                         (1, 490.0, "right", "major")]
    assert j[0]["angle_deg"] == pytest.approx(90.0) and j[1]["angle_deg"] == pytest.approx(-90.0)
    assert j[0]["name"] == "Kerria Road" and j[0]["tags"]["surface"] == "asphalt" and j[0]["way_ids"] == [100]
    # Followed out to 100 m. The width is its class default from its tags (section.width_from_tags), as mapped.
    assert j[0]["length_m"] == pytest.approx(junctions.LEG_M) and j[0]["ends"] == "length"
    assert j[0]["line"][0] == [300.0, 0.0] and j[0]["line"][-1] == pytest.approx([300.0, 100.0])
    assert (j[0]["width_m"], j[1]["width_m"]) == (5.8, 6.0)
    assert res["skipped"] == [] and res["roundabouts"] == []


@pytest.mark.parametrize("tags,reason", [
    ({"highway": "residential", "access": "private"}, "private"),
    ({"highway": "residential", "motor_vehicle": "no"}, "private"),
    ({"highway": "residential", "access": "yes", "vehicle": "private"}, "private"),  # the more specific wins
    ({"highway": "service"}, "class"),  # the road data holds none, but it wouldn't be a junction
    ({"highway": "residential", "bridge": "yes"}, "grade_separated"),
    ({"highway": "residential", "tunnel": "culvert"}, "grade_separated"),
    ({"highway": "residential", "layer": "-1"}, "grade_separated"),
])
def test_side_roads_that_are_not_junctions_are_listed_with_the_reason(tags, reason):
    res = find(network([(300, north(300), tags)]))
    assert res["junctions"] == []
    assert [(s["at_m"], s["side"], s["reason"]) for s in res["skipped"]] == [(290.0, "left", reason)]


@pytest.mark.parametrize("tags,closure", [
    ({"highway": "residential", "access": "no", "motor_vehicle": "destination"}, "limited"),  # most specific first
    ({"highway": "unclassified", "access": "permit"}, "limited"),
    ({"highway": "tertiary", "motor_vehicle": "forestry"}, "limited"),
    ({"highway": "secondary_link"}, "major"),
    ({"highway": "trunk"}, "major"),
    ({"highway": "living_street"}, "minor"),
    ({"highway": "residential", "access": "no", "motor_vehicle": "yes"}, "minor"),
])
def test_closure_level_by_access_then_class(tags, closure):
    assert [x["closure"] for x in find(network([(300, north(300), tags)]))["junctions"]] == [closure]


@pytest.mark.parametrize("bridge_from,counts", [(20.0, False), (40.0, True)])
def test_a_bridge_counts_only_within_the_first_30_m(bridge_from, counts):
    # The side road's own way, then a bridge way carrying on north from `bridge_from` metres out.
    g = network([(300, [(300.0, bridge_from)], RESIDENTIAL)])
    g = extend(g, (300.0, bridge_from), [(300.0, bridge_from + 20), (300.0, 150.0)], {**RESIDENTIAL, "bridge": "yes"})
    res = find(g)
    assert bool(res["junctions"]) == counts
    if counts:
        assert res["junctions"][0]["way_ids"] == [100, 99]  # followed on across the bridge
    else:
        assert res["skipped"][0]["reason"] == "grade_separated"


@pytest.mark.parametrize("deg,counts", [(10.0, False), (168.0, False), (20.0, True), (150.0, True)])
def test_a_side_road_must_leave_at_least_15_degrees_off_the_road_ahead_and_behind(deg, counts):
    res = find(network([(500, at_angle(500, deg), {"highway": "unclassified"})]))
    assert bool(res["junctions"]) == counts
    if counts:
        assert res["junctions"][0]["off_mainline_deg"] == pytest.approx(min(deg, 180 - deg), abs=0.5)
    else:
        assert res["skipped"][0]["reason"] == "angle"


def hairpin(gap: float):
    """A road east along y = 0, round a half circle and back west along y = gap, with a residential link straight
    across at x = 200; nodes every 50 m."""
    xs = np.arange(0.0, 501.0, 50.0)
    th = np.radians(np.arange(-80.0, 90.0, 20.0))
    bend = [(500 + gap / 2 * np.cos(t), gap / 2 + gap / 2 * np.sin(t)) for t in th]
    pts = [(x, 0.0) for x in xs] + bend + [(x, gap) for x in xs[::-1]]
    nodes = {i + 1: p for i, p in enumerate(pts)}
    low, high = 1 + 4, len(pts) - 4  # the nodes at x = 200 on each part
    return osm(nodes, [(1, list(nodes), {"highway": "secondary"}), (2, [low, high], RESIDENTIAL)]), gap


@pytest.mark.parametrize("gap,counts", [(20.0, False), (60.0, True)])
def test_a_side_road_is_clear_of_the_rest_of_the_route_or_not_a_junction(gap, counts):
    g, gap = hairpin(gap)
    r = graph.route(g, np.array([[10.0, 0.0], [500.0 + gap / 2, gap / 2], [10.0, gap]]))  # round the hairpin
    res = junctions.find(g, r.xy)
    if not counts:  # it comes within 10 m of the other part 10 m out: under 15 m, so not a junction from either part
        assert res["junctions"] == [] and [s["reason"] for s in res["skipped"]] == ["meets_route"] * 2
    else:  # followed until it comes within 10 m of the other part (tested every metre), from each end
        assert [(j["ends"], j["length_m"]) for j in res["junctions"]] == [("meets_route", pytest.approx(gap - 10, abs=1.0))] * 2


def test_a_dead_end_side_road_is_followed_to_its_end():
    j = find(network([(300, [(300.0, 20.0), (300.0, 40.0)], {"highway": "unclassified"})]))["junctions"]
    assert j[0]["ends"] == "dead_end" and j[0]["length_m"] == pytest.approx(40.0)


def test_the_road_beyond_the_route_s_ends_is_not_a_side_road():
    # The route starts 30 m past one node and ends 5 m short of another: the main road carries on beyond both.
    res = find(network([]), start=130.0, end=895.0)
    assert res["junctions"] == [] and res["skipped"] == []


def test_two_nodes_at_one_place_are_told_apart_by_the_route():
    # An unconnected side road's end node sits exactly on the main road's node at 400 m (an OSM slip): the route's
    # node there is still the main road's, so the real side road at 400 m is found.
    g = network([(200, [(300.0, 50.0), (400.0, 0.0)], {"highway": "unclassified"}), (400, north(400), RESIDENTIAL)])
    assert [j["at_m"] for j in find(g)["junctions"]] == [190.0, 390.0]


def test_a_roundabout_the_route_drives_is_recorded_with_its_exits_and_cleared_of_nothing():
    # A ring of radius 20 m round (500, 20), drawn clockwise from the main road's node at 500 m; one exit north.
    th = np.radians(np.arange(225.0, -90.0, -45.0))
    ring = [(500 + 20 * np.cos(t), 20 + 20 * np.sin(t)) for t in th] + [5]
    g = network([(500, ring, {"highway": "secondary", "junction": "roundabout", "oneway": "yes"})])
    g = extend(g, (500.0, 40.0), [(500.0, 100.0), (500.0, 160.0)], {"highway": "tertiary"})
    res = find(g)
    assert res["junctions"] == []
    (ring_rec,) = res["roundabouts"]
    assert ring_rec["closed"] and ring_rec["on_route"] and ring_rec["radius_m"] == pytest.approx(20.0, abs=0.5)
    assert [(e["highway"], e["on_route"], e["closure"]) for e in ring_rec["exits"]] == [("tertiary", False, "major")]


# -- the mouths ----------------------------------------------------------------------------------------------------

def straight(n=1001):
    s = np.arange(n, dtype=float)
    return s, np.column_stack([s, np.zeros(n)]), np.zeros(n), np.full(n, HALF), np.full(n, HALF)


def junction(x, deg, width=5.8, length=100.0):
    a = np.radians(deg)
    line = [[x + d * np.cos(a), d * np.sin(a)] for d in np.linspace(0, length, 11)]
    return {"at_m": x, "xy": [x, 0.0], "side": "left" if deg > 0 else "right", "width_m": width,
            "length_m": length, "line": line, "highway": "residential", "closure": "minor", "name": "Side Road"}


def test_a_square_junction_s_mouth_is_its_width_plus_the_margin_on_its_own_side():
    s, xy, heading, hl, hr = straight()
    (m,) = junctions.mouths([junction(300.0, 90.0)], s, xy, heading, hl, hr)
    assert m.side == "L" and m.station_m == pytest.approx(300.0) and m.half_m == 2.9
    # The carriageway covers 297.1-302.9 m; stations are a metre apart.
    assert (m.s0, m.s1) == (298.0 - junctions.MOUTH_MARGIN_M, 302.0 + junctions.MOUTH_MARGIN_M)
    mk = junctions.masks([m], s)
    assert mk["L"][293:308].all() and mk["L"].sum() == 15 and not mk["R"].any()


def test_an_acute_junction_s_mouth_reaches_further_along_the_road_ahead():
    s, xy, heading, hl, hr = straight()
    (m,) = junctions.mouths([junction(300.0, -25.0)], s, xy, heading, hl, hr)
    # Its carriageway crosses the band 0-3 m beyond the right edge from 3.2/tan 25 - 2.9/sin 25 to
    # 6.2/tan 25 + 2.9/sin 25 metres past the junction.
    t, sn = np.tan(np.radians(25)), np.sin(np.radians(25))
    assert m.side == "R"
    assert m.s0 == pytest.approx(300 + HALF / t - 2.9 / sn - junctions.MOUTH_MARGIN_M, abs=1.0)
    assert m.s1 == pytest.approx(300 + (HALF + 3) / t + 2.9 / sn + junctions.MOUTH_MARGIN_M, abs=1.0)


def test_layout_intervals_run_from_each_layout_s_start():
    s, xy, heading, hl, hr = straight()
    ms = junctions.mouths([junction(300.0, 90.0)], s, xy, heading, hl, hr)
    ((a, b),) = junctions.layout_intervals(ms, s, reverse=False)
    assert (a, b) == (293.0, 307.0)
    assert junctions.layout_intervals(ms, s, reverse=True) == [(1000.0 - b, 1000.0 - a)]


def test_the_clearing_zone_follows_the_side_road_on_its_own_side_of_the_road():
    s, xy, heading, hl, hr = straight()
    ms = junctions.mouths([junction(300.0, -20.0)], s, xy, heading, hl, hr)
    z = junctions.zone(ms, s, xy, 4.0)
    a = np.radians(20)
    assert z.contains(shapely.Point(300 + 50 * np.cos(a), -50 * np.sin(a) + 6.0))  # 6 m off it: half + 3.1 m
    assert not z.contains(shapely.Point(300 + 50 * np.cos(a), -50 * np.sin(a) + 7.5))
    assert z.bounds[3] <= 1e-6  # nothing left of the centreline, though the side road leaves at only 20 degrees
    assert junctions.zone([], s, xy, 4.0) is None


# -- keeping them clear ----------------------------------------------------------------------------------------------

def test_a_rail_stops_at_a_mouth_and_a_short_piece_beside_it_goes_too():
    s = np.arange(400.0)
    rail = {"L": (s >= 50) & (s <= 250), "R": (s >= 100) & (s <= 140)}
    mouth = {"L": (s >= 200) & (s <= 220), "R": (s >= 110) & (s <= 125)}
    kept, removed = roadside.clear_mouths(rail, mouth, s)
    assert kept["L"][50:200].all() and not kept["L"][200:221].any() and kept["L"][221:251].all()  # 30 m stays
    assert removed["L"] == 21.0
    assert not kept["R"].any() and removed["R"] == 41.0  # 10 m and 15 m pieces either side: too short alone


def test_a_rail_run_that_never_meets_a_mouth_is_left_as_it_was():
    s = np.arange(400.0)
    rail = {"L": s <= 20, "R": np.zeros(400, bool)}  # short, but not cut by a mouth
    kept, removed = roadside.clear_mouths(rail, {"L": (s >= 300) & (s <= 320), "R": np.zeros(400, bool)}, s)
    assert np.array_equal(kept["L"], rail["L"]) and removed == {"L": 0.0, "R": 0.0}


def furn(n=1001, mouth_l=None, mouth_r=None, curvature=None, limits=None):
    s = np.arange(n, dtype=float)
    k = np.zeros(n) if curvature is None else curvature
    heading = np.concatenate([[0.0], np.cumsum(k[:-1])])
    xy = np.column_stack([np.cumsum(np.cos(heading)), np.cumsum(np.sin(heading))]) - [1.0, 0.0]
    flat = np.zeros(n)
    meshes, info, _ = furniture.furniture(s, xy, np.full(n, 100.0), heading, k, np.full(n, HALF), np.full(n, HALF),
                                          flat, flat, flat, np.full(n, 80.0) if limits is None else limits, FRAME,
                                          mouth_l=mouth_l, mouth_r=mouth_r)
    return meshes, info, s


def post_stations(meshes, side):
    """Stations of the guide posts on one side of a straight road east along y = 0: the 1.3 m boxes."""
    pos = np.vstack([m.positions for m in meshes]).astype(float)
    e, n, h = FRAME.from_ac(pos)
    out = set()
    for q in range(0, len(pos), 4):
        if abs((h[q + 2] - h[q]) - (furniture.GUIDE_POST_HEIGHT_M + 0.3)) < 1e-3:
            if (n[q:q + 4].mean() > 0) == (side == "L"):
                out.add(int(round(e[q:q + 4].mean())))
    return sorted(out)


def test_a_guide_post_due_in_a_mouth_goes_up_just_past_it():
    meshes, _, s = furn()
    assert post_stations(meshes, "L") == [5, 155, 305, 455, 605, 755, 905]
    meshes, info, s = furn(mouth_l=(s >= 290) & (s <= 320))
    assert post_stations(meshes, "L")[:4] == [5, 155, 321, 471]
    assert post_stations(meshes, "R") == [5, 155, 305, 455, 605, 755, 905]  # the other side as before
    assert info["side_road_mouths"]["posts_moved"] == 1


def test_a_speed_sign_moves_out_of_a_mouth_and_a_curve_marker_in_one_is_left_out():
    n = 600
    limits = np.full(n, 80.0)
    limits[300:] = 60.0  # a change at 300 m: a sign for each direction
    _, plain, s = furn(n, limits=limits)
    _, info, _ = furn(n, limits=limits, mouth_l=(s >= 295) & (s <= 310))
    assert info["speed_signs"] == plain["speed_signs"] and info["side_road_mouths"]["signs_moved"] == 1
    k = np.zeros(n)
    k[200:300] = 1 / 50.0  # a left-hander: CAMs on its outside, the right
    _, bend, _ = furn(n, curvature=k)
    _, info, _ = furn(n, curvature=k, mouth_r=(s >= 230) & (s <= 260))
    assert info["side_road_mouths"]["cams_left_out"] == 1 and info["cams"] == bend["cams"] - 1


def test_with_no_mouth_furniture_is_exactly_as_before():
    meshes, info, s = furn()
    meshes2, info2, _ = furn(mouth_l=np.zeros(len(s), bool), mouth_r=np.zeros(len(s), bool))
    assert len(meshes) == len(meshes2) and all(np.array_equal(a.positions, b.positions) for a, b in zip(meshes, meshes2))
    assert info == {k: v for k, v in info2.items() if k != "side_road_mouths"}


def test_a_power_support_on_a_side_road_is_dropped_with_its_spans():
    road = np.column_stack([np.arange(0.0, 1001.0, 10.0), np.zeros(101)])
    line = power.Line(np.array([[400.0, 30.0], [500.0, 30.0], [600.0, 30.0], [700.0, 30.0]]), power.DISTRIBUTION)
    side = shapely.LineString([(500.0, 0.0), (500.0, 100.0)]).buffer(2.9 + power.ON_ROAD_MARGIN_M)
    plain = power.plan([line], road, HALF)
    p = power.plan([line], road, HALF, blocked=side)
    assert len(plain.spans) == 3 and p.in_junctions == 1
    assert [(p.supports[s.a].xy[0], p.supports[s.b].xy[0]) for s in p.spans] == [(600.0, 700.0)]
    assert "side_road_supports" not in plain.summary() and p.summary()["side_road_supports"] == 1


def test_a_building_left_out_still_takes_its_draws_so_the_others_are_unchanged():
    houses = [shapely.box(100 * k, 50, 100 * k + 12, 60) for k in range(5)]
    ground = lambda x, y: np.full(len(np.atleast_1d(x)), 20.0)  # noqa: E731
    rng_all, rng_some = np.random.default_rng(7), np.random.default_rng(7)
    every = {m.name: m for m in scenery.building_meshes(houses, ground, FRAME, rng_all)}
    keep = np.array([True, False, True, True, False])
    some = {m.name: m for m in scenery.building_meshes(houses, ground, FRAME, rng_some, keep=keep)}
    assert rng_all.random() == rng_some.random()  # the same draws were taken
    assert set(np.unique(some["1WALL_BUILDINGS"].positions[:, 0] // 100)) == {0, 2, 3}
    for name, m in some.items():  # each kept house has the roof colour it has with all five built
        built = {tuple(p) for p in np.round(every[name].positions, 3)}
        assert all(tuple(p) in built for p in np.round(m.positions, 3))


def test_trees_on_a_side_road_go_and_every_other_tree_keeps_its_model_angle_and_size(tmp_path):
    from mapcreator.pipeline import scenery_stage

    rng = np.random.default_rng(1)
    trees = scenery.Trees(rng.uniform(0, 500, (300, 2)), rng.uniform(6, 40, 300))
    ground = np.full(300, 20.0)
    keep = rng.random(300) > 0.1
    (tmp_path / "all").mkdir()
    (tmp_path / "some").mkdir()
    _, r_all, _ = scenery_stage._tree_fx(trees, ground, FRAME, np.random.default_rng(5), tmp_path / "all")
    _, r_some, _ = scenery_stage._tree_fx(trees, ground, FRAME, np.random.default_rng(5), tmp_path / "some", keep=keep)
    every = np.load(tmp_path / "all" / "treefx" / "placement.npz")
    some = np.load(tmp_path / "some" / "treefx" / "placement.npz")
    assert len(some["pos"]) == keep.sum() and np.allclose(r_some, r_all[keep])
    for key in ("pos", "size", "angle", "width"):
        assert np.array_equal(some[key], every[key][keep])
    assert [str(every["names"][i]) for i in every["model"][keep]] == [str(some["names"][i]) for i in some["model"]]


# -- timing gates ----------------------------------------------------------------------------------------------------

def road_model(length=1000.0):
    s = np.arange(0.0, length + 1)
    zero = np.zeros(len(s))
    return RoadModel.from_arrays(s, np.column_stack([s, zero]), np.full(len(s), 100.0), zero, zero, zero,
                                 np.full(len(s), HALF), np.full(len(s), HALF), np.full(len(s), 0.025),
                                 np.full(len(s), 0.025), False)


def test_a_gate_in_a_mouth_moves_to_the_nearest_clear_place():
    road = road_model()
    assert layout.gate_distances(road) == (50.0, layout.finish_distance(1000.0)) == (50.0, 850.0)
    # Back to 44.5 m would be nearer, but it would put the front grid spot (42 m) less than 4 m behind the line.
    assert layout.gate_distances(road, [(45.0, 60.0), (840.0, 852.0)]) == (60.5, 852.5)
    assert layout.gate_distances(road, [(47.0, 70.0)])[0] == 46.5  # 4.5 m ahead of the front spot
    # Nowhere clear within 40 m: it stays where it was, and the validator says so.
    assert layout.gate_distances(road, [(0.0, 200.0)])[0] == 50.0


def test_only_the_gates_move_the_grid_and_hot_lap_spawn_stay():
    road = road_model()
    plain = {d.name: d.position for d in logic_dummies(road, FRAME)}
    moved = {d.name: d.position for d in logic_dummies(road, FRAME, [(45.0, 56.0)])}
    assert plain["AC_AB_START_L"][0] == pytest.approx(50.0) and moved["AC_AB_START_L"][0] == pytest.approx(56.5)
    assert {k: v for k, v in moved.items() if "AB_START" not in k} == {k: v for k, v in plain.items()
                                                                        if "AB_START" not in k}


# -- validation ------------------------------------------------------------------------------------------------------

def kn5_data(meshes: dict[str, np.ndarray]) -> dict:
    nodes = [{"class": 1, "name": "root"}]
    for name, xz in meshes.items():
        pos = np.column_stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]]).astype(np.float32)
        nodes.append({"class": 2, "name": name, "vertices": {"pos": pos}})
    return {"nodes": nodes}


def start_gate(x: float) -> dict:
    out = {}
    for side, z in (("L", -(HALF + 0.5)), ("R", HALF + 0.5)):  # AC z is minus north: left is -z heading east
        m = np.eye(4)
        m[3, :3] = [x, 0.0, z]
        out[f"AC_AB_START_{side}"] = {"matrix": m}
    return out


def test_the_validator_lists_the_junctions_and_fails_a_rail_or_trunk_on_a_side_road():
    # A side road leaving north (AC -z) at x = 300: its mouth out to 3 m past the edge, and its carriageway.
    mouth = shapely.box(297.1, -(HALF + 3.0), 302.9, 0.0)
    road = shapely.box(297.1, -100.0, 302.9, 0.0)
    check = validate.JunctionCheck("Side Road (residential, minor) at 0.30 km on the left", mouth, road)
    clear = {"RAIL_L000": np.array([[100.0, -4.2], [290.0, -4.2]]), "1WALL_TRUNK_000": np.array([[310.0, -50.0]]),
             "FURNITURE_0": np.array([[315.0, -4.0]])}
    rep = validate.Report()
    validate.junction_checks(rep, "main", kn5_data(clear), start_gate(50.0), [check])
    assert rep.failed == [] and rep.warnings[0].startswith("1 side-road junction found") and "Side Road" in rep.warnings[0]
    assert {"main: no guard rails in a side road's mouth", "main: no guide posts and signs in a side road's mouth",
            "main: no tree colliders on a side road", "main: timing gates clear of side roads' mouths"} <= set(rep.passed)
    rep = validate.Report()
    stuck = {**clear, "1WALL_RAIL_L001": np.array([[299.0, -4.2]]), "1WALL_TRUNK_001": np.array([[300.0, -80.0]])}
    validate.junction_checks(rep, "main", kn5_data(stuck), start_gate(300.0), [check])
    assert sorted(f.split(" (")[0] for f in rep.failed) == ["main: no guard rails in a side road's mouth",
                                                           "main: no tree colliders on a side road"]
    assert any("timing gate stands in a side road's mouth" in w for w in rep.warnings)  # a warning, not a failure


def test_with_no_junctions_the_validator_adds_nothing():
    rep = validate.Report()
    validate.junction_checks(rep, "main", kn5_data({"RAIL_L000": np.zeros((2, 2))}), start_gate(50.0), [])
    assert rep.passed == rep.failed == rep.warnings == []


# -- the route stage, and `junctions: none` ------------------------------------------------------------------------

HOBART = [{"lon": 147.301, "lat": -42.8001}, {"lon": 147.319, "lat": -42.7999}]


def hobart_roads() -> dict:
    """A secondary road east along -42.80, a residential road north from 147.306 and a private one south from
    147.312."""
    nodes = [{"type": "node", "id": i + 1, "lon": 147.300 + i * 0.002, "lat": -42.80} for i in range(11)]
    up = [{"type": "node", "id": 100 + k, "lon": 147.306, "lat": -42.80 + 0.0006 * (k + 1)} for k in range(3)]
    down = [{"type": "node", "id": 200 + k, "lon": 147.312, "lat": -42.80 - 0.0006 * (k + 1)} for k in range(3)]
    ways = [{"type": "way", "id": 1, "nodes": [n["id"] for n in nodes], "tags": {"highway": "secondary"}},
            {"type": "way", "id": 2, "nodes": [4, 100, 101, 102], "tags": {**RESIDENTIAL, "name": "Up Street"}},
            {"type": "way", "id": 3, "nodes": [7, 200, 201, 202], "tags": {**RESIDENTIAL, "access": "private"}}]
    return {"elements": [*nodes, *up, *down, *ways]}


def test_the_route_stage_writes_junctions_json_and_none_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(route_stage.osm_roads, "for_bbox", lambda bbox, settings: hobart_roads())
    settings, work = config.load(), WorkDir(tmp_path / "work")
    job = Job(track_id="hobart", waypoints=HOBART)
    result = route_stage.run(job, settings, work)
    assert (result["junctions"], result["roundabouts"]) == (1, 0)
    doc = work.read_json(route_stage.STAGE, route_stage.JUNCTIONS)
    assert doc["version"] == junctions.FORMAT_VERSION and doc["crs"] == "EPSG:28355"
    (j,) = doc["junctions"]
    assert (j["name"], j["side"], j["closure"], j["node"]) == ("Up Street", "left", "minor", 4)
    assert j["lonlat"] == pytest.approx([147.306, -42.80], abs=1e-6)
    assert [s["reason"] for s in doc["skipped"]] == ["private"]
    assert route_stage.junctions(work, job) == doc["junctions"]
    without = Job(track_id="hobart", waypoints=HOBART, junctions="none")
    assert route_stage.junctions(work, without) == []  # the later stages ignore the file
    result = route_stage.run(without, settings, work)
    assert "junctions" not in result and not (work.root / route_stage.STAGE / route_stage.JUNCTIONS).exists()
    assert route_stage.junctions(work, job) == []  # routed without them: nothing to clear


def test_junctions_default_to_closed_and_stay_out_of_the_track_id():
    job = Job(track_id="a", waypoints=HOBART)
    assert job.junctions == Junctions.CLOSED and "junctions" not in job_dict(job)
    none = Job(track_id="a", waypoints=HOBART, junctions="none")
    assert job_dict(none)["junctions"] == "none" and route_hash(none) != route_hash(job)
    with pytest.raises(ValueError):
        Job(track_id="a", waypoints=HOBART, junctions="open")
