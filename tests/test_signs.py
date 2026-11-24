"""Regulatory signs and zebra crossings (`core.signs`): only where OSM tags them, which way they face, ONE WAY and NO
ENTRY from `oneway`, where they stand, and the zebra's stripes."""

import math

import numpy as np
import pytest
import shapely
from pyproj import Transformer

from mapcreator.core import junctions, signals, signs, validate
from mapcreator.core.scene import LocalFrame
from mapcreator.providers.base import LONLAT_CRS
from mapcreator.providers.osm import graph

TO_XY = Transformer.from_crs(LONLAT_CRS, 28355, always_xy=True)
LAT = -42.8


def _osm(main=None, north=None, south=None, node_tags=None, extra=()):
    """Main Road runs east along LAT through a crossroads at 147.305: North Street (8, 4, 2) comes down from the north,
    South Street (2, 5, 6) goes on south. Nodes 4 and 5 are 22 m from the crossroads, 8 and 6 222 m."""
    pts = {1: (147.300, LAT), 2: (147.305, LAT), 3: (147.310, LAT), 9: (147.315, LAT), 10: (147.320, LAT),
           4: (147.305, LAT + 0.0002), 8: (147.305, LAT + 0.002), 5: (147.305, LAT - 0.0002),
           6: (147.305, LAT - 0.002)}
    for way in extra:
        pts |= way.get("pts", {})
    tags = node_tags or {}
    nodes = [{"type": "node", "id": i, "lon": lon, "lat": lat, **({"tags": tags[i]} if i in tags else {})}
             for i, (lon, lat) in pts.items()]
    ways = [
        {"type": "way", "id": 10, "nodes": [1, 2, 3, 9, 10], "tags": {"highway": "secondary", "name": "Main Road",
                                                                      **(main or {})}},
        {"type": "way", "id": 11, "nodes": [8, 4, 2], "tags": {"highway": "residential", "name": "North Street",
                                                               **(north or {})}},
        {"type": "way", "id": 12, "nodes": [2, 5, 6], "tags": {"highway": "residential", "name": "South Street",
                                                               **(south or {})}},
    ] + [{"type": "way", "id": w["id"], "nodes": w["nodes"], "tags": w["tags"]} for w in extra]
    return {"elements": nodes + ways}


def _find(main=None, north=None, south=None, node_tags=None, extra=(), start=147.3005, end=147.3195):
    data = _osm(main, north, south, node_tags, extra)
    g = graph.build(data, TO_XY)
    route = graph.route(g, np.column_stack(TO_XY.transform([start, end], [LAT, LAT])))
    nodes = [{"id": e["id"], "xy": TO_XY.transform(e["lon"], e["lat"]), "tags": e["tags"]}
             for e in data["elements"] if e["type"] == "node" and e.get("tags")]
    found_junctions = junctions.find(g, route.xy)["junctions"]
    return signs.find(g, route.xy, nodes, found_junctions), found_junctions


def _kinds(found, kind):
    return [s for s in found["signs"] if s["kind"] == kind]


# ----- what OSM tags, and nothing else -----

def test_give_way_and_stop_only_where_osm_tags_them():
    found, _ = _find()
    assert _kinds(found, "give_way") == [] and _kinds(found, "stop") == []  # an untagged crossroads: none inferred
    found, js = _find(node_tags={4: {"highway": "give_way"}, 5: {"traffic_sign": "AU:R1-1"}})
    gw, stop = _kinds(found, "give_way"), _kinds(found, "stop")
    assert len(gw) == 1 and len(stop) == 1
    assert gw[0]["on"] == "side_road" and gw[0]["node"] == 4 and gw[0]["d_m"] == pytest.approx(22.2, abs=0.5)
    north = next(j for j in js if j["name"] == "North Street")
    assert gw[0]["junction"] == north["id"] and gw[0]["from"] == "approach"


def test_tagged_signs_far_from_the_route_or_on_a_road_not_drawn_are_left_out():
    found, _ = _find(node_tags={8: {"highway": "give_way"}})  # 222 m up North Street: another junction's
    assert found["signs"] == [] and found["skipped"] == [{"node": 8, "kind": "give_way", "reason": "off_route"}]
    found, _ = _find(north={"access": "private"}, node_tags={4: {"highway": "stop"}})  # no stub, no junction
    assert found["skipped"] == [{"node": 4, "kind": "stop", "reason": "side_road_not_drawn"}]
    found, _ = _find(node_tags={2: {"highway": "give_way"}})  # on the junction's own node: which way is unknown
    assert found["skipped"][0]["reason"] == "at_a_junction_node"


def test_sign_kinds_from_highway_and_traffic_sign():
    assert signs.sign_kind({"highway": "stop"}) == "stop"
    assert signs.sign_kind({"highway": "give_way"}) == "give_way"
    assert signs.sign_kind({"traffic_sign": "AU:R1-2"}) == "give_way"
    assert signs.sign_kind({"traffic_sign": "AU:R6-23;AU:R1-1"}) == "stop"
    assert signs.sign_kind({"traffic_sign": "maxspeed"}) is None and signs.sign_kind({"highway": "crossing"}) is None


def test_crossings_get_signs_and_stripes_only_where_tagged_as_zebras():
    assert signs.crossing_plan({"highway": "crossing", "crossing:markings": "zebra"}) == {"signs": True, "zebra": True}
    assert signs.crossing_plan({"highway": "crossing", "crossing_ref": "zebra"}) == {"signs": True, "zebra": True}
    # Marked with no markings tagged: the sign, but no stripes we can't stand behind.
    assert signs.crossing_plan({"highway": "crossing", "crossing": "marked"}) == {"signs": True, "zebra": False}
    assert signs.crossing_plan({"highway": "crossing", "crossing": "marked", "crossing:markings": "dashes"}) == {
        "reason": "not_zebra"}
    assert signs.crossing_plan({"highway": "crossing", "crossing": "traffic_signals",
                                "crossing:markings": "zebra"}) == {"reason": "signalised"}
    assert signs.crossing_plan({"highway": "crossing", "crossing": "unmarked"}) is None
    assert signs.crossing_plan({"highway": "give_way"}) is None
    found, _ = _find(node_tags={3: {"highway": "crossing", "crossing:markings": "zebra"},
                                5: {"highway": "crossing", "crossing": "marked"}})
    zebras = found["zebras"]
    assert [(z["on"], z["node"], z["ways"]) for z in zebras] == [("route", 3, "both")]
    assert sorted((s["on"], s["node"]) for s in _kinds(found, "pedestrian_crossing")) == [("route", 3),
                                                                                         ("side_road", 5)]


# ----- which way they face -----

def test_facing_follows_the_direction_tag_or_the_approach():
    # North Street runs 8 -> 4 -> 2, south into the crossroads.
    for tags in ({"highway": "give_way"}, {"highway": "give_way", "direction": "forward"}):
        found, _ = _find(node_tags={4: tags})
        gw = _kinds(found, "give_way")[0]
        assert math.degrees(gw["travel"]) == pytest.approx(-90.0, abs=2.0)  # for traffic heading south, to the route
    found, _ = _find(node_tags={4: {"highway": "give_way", "direction": "forward"}})
    assert _kinds(found, "give_way")[0]["from"] == "direction_tag"
    # Backward: for traffic heading north, away from the route, to another junction: not this track's.
    found, _ = _find(node_tags={4: {"highway": "give_way", "direction": "backward"}})
    assert found["signs"] == [] and found["skipped"][0]["reason"] == "not_for_a_route_junction"
    # On South Street (2 -> 5 -> 6), untagged: for traffic heading north, into the route.
    found, _ = _find(node_tags={5: {"highway": "stop"}})
    assert math.degrees(_kinds(found, "stop")[0]["travel"]) == pytest.approx(90.0, abs=2.0)


def test_a_sign_on_the_route_faces_the_traffic_its_direction_tag_names():
    # Node 3 is on Main Road, 367 m past the crossroads; Main Road runs east, the route's way.
    found, _ = _find(node_tags={3: {"highway": "give_way", "direction": "backward"}})
    gw = _kinds(found, "give_way")[0]
    assert gw["on"] == "route" and abs(math.degrees(gw["travel"])) == pytest.approx(180.0, abs=2.0)
    found, _ = _find(node_tags={3: {"highway": "give_way", "direction": "both"}})
    assert len(_kinds(found, "give_way")) == 2


def test_the_approach_only_follows_a_way_traffic_may_drive():
    # North Street one-way north, away from the route: nobody drives in to the crossroads along it.
    found, _ = _find(north={"oneway": "-1"}, node_tags={4: {"highway": "give_way"}})
    assert found["signs"] == [] or all(s["kind"] != "give_way" for s in found["signs"])
    assert {"node": 4, "kind": "give_way", "reason": "no_junction_near"} in found["skipped"]


# ----- ONE WAY and NO ENTRY -----

def test_one_way_on_the_far_side_from_each_side_road_that_can_drive_in():
    found, js = _find(main={"oneway": "yes"})
    one_way = [s for s in _kinds(found, "one_way") if not s.get("repeater")]
    # North Street (on the left) and South Street (on the right) are two-way: each gets one across from it.
    assert sorted(s["side"] for s in one_way) == ["L", "R"]
    assert all(math.degrees(s["travel"]) == pytest.approx(0.0, abs=2.0) for s in one_way)  # the arrow: east
    # Two-way Main Road: none.
    found, _ = _find()
    assert _kinds(found, "one_way") == []
    # South Street one-way south, out of the junction: nobody drives in from it, so only North Street's.
    found, _ = _find(main={"oneway": "yes"}, south={"oneway": "yes"})
    assert [s["side"] for s in _kinds(found, "one_way") if not s.get("repeater")] == ["R"]


def test_no_entry_at_the_exit_end_of_a_one_way_side_road():
    # South Street one-way north (6 -> 5 -> 2): its exit end is the route, which it may not be entered from.
    found, js = _find(south={"oneway": "-1"})
    no_entry = _kinds(found, "no_entry")
    south = next(j for j in js if j["name"] == "South Street")
    assert [(s["on"], s["junction"]) for s in no_entry] == [("side_road", south["id"])]
    # A one-way street crossing the route, carrying on north as North Street does here (the same name): the
    # crossroads isn't its exit end. ONE WAY signs, not NO ENTRY.
    found, _ = _find(north={"oneway": "-1", "name": "Cross Street"}, south={"oneway": "-1", "name": "Cross Street"})
    assert _kinds(found, "no_entry") == []
    # One-way away from the route: entered the right way.
    found, _ = _find(south={"oneway": "yes"})
    assert _kinds(found, "no_entry") == []


def test_no_one_way_where_the_route_turns_or_on_a_motorway():
    found, _ = _find(main={"oneway": "yes", "highway": "motorway"})
    assert _kinds(found, "one_way") == []
    # Along one-way Main Road, then left up North Street, one-way north: no "across" at a corner.
    data = _osm({"oneway": "yes"}, {"oneway": "-1"})
    g = graph.build(data, TO_XY)
    route = graph.route(g, np.column_stack(TO_XY.transform([147.3005, 147.305], [LAT, LAT + 0.0018])))
    js = junctions.find(g, route.xy)["junctions"]
    assert js and all(j["route_turn_deg"] > 80 for j in js)
    assert [s for s in signs.find(g, route.xy, [], js)["signs"] if not s.get("repeater")] == []


def _long(extra_tags=None, divided=False):
    """Long Road: one way east for 1.6 km with no side road, and, if `divided`, its other carriageway one way west
    20 m north."""
    pts = {i: (147.300 + 0.001 * i, LAT) for i in range(21)}
    nodes = [{"type": "node", "id": i, "lon": lon, "lat": lat} for i, (lon, lat) in pts.items()]
    ways = [{"type": "way", "id": 1, "nodes": list(range(21)),
             "tags": {"highway": "secondary", "name": "Long Road", "oneway": "yes", **(extra_tags or {})}}]
    if divided:
        nodes += [{"type": "node", "id": 100 + i, "lon": 147.300 + 0.001 * i, "lat": LAT + 0.00018} for i in range(21)]
        ways.append({"type": "way", "id": 2, "nodes": [100 + i for i in reversed(range(21))],
                     "tags": {"highway": "secondary", "name": "Long Road", "oneway": "yes"}})
    data = {"elements": nodes + ways}
    g = graph.build(data, TO_XY)
    route = graph.route(g, np.column_stack(TO_XY.transform([147.3005, 147.3195], [LAT, LAT])))
    return signs.find(g, route.xy, [], junctions.find(g, route.xy)["junctions"])


def test_one_way_repeaters_along_a_long_block_but_not_on_a_divided_road():
    reps = [s for s in _kinds(_long(), "one_way") if s.get("repeater")]
    # 1.55 km with no junction: a repeater every ONE_WAY_REPEAT_M or so, none within END_CLEAR_M of the ends.
    assert len(reps) == int(1550 // signs.ONE_WAY_REPEAT_M) and all(s["side"] == "L" for s in reps)
    assert _kinds(_long(divided=True), "one_way") == []
    assert _kinds(_long({"highway": "motorway"}), "one_way") == []


# ----- where they stand -----

def _road(length=400.0):
    s = np.arange(0.0, length + 0.5, 1.0)
    xy = np.column_stack([s, np.zeros_like(s)])
    return signals._Road(s, xy, np.zeros_like(s), np.zeros_like(s), np.full_like(s, 3.5), np.full_like(s, 3.5),
                         np.zeros_like(s), np.zeros_like(s))


def _mouth(side, at, s0, s1, jid=0, half=3.0, length=60.0):
    y = length if side == "L" else -length
    return junctions.Mouth({"id": jid, "node": 2}, side, at, s0, s1, np.array([[at, 0.0], [at, y]]), half)


def _drawn(monkeypatch, found, **kw):
    faces = []
    real = signs._draw_sign

    def spy(b, p, bottom, kind, facing, arrow_right=True):
        faces.append({"kind": kind, "e": p.e, "n": p.n, "facing": np.round(signals._unit(facing), 3).tolist(),
                      "arrow_right": arrow_right, "own": p.own, "bottom": bottom})
        real(b, p, bottom, kind, facing, arrow_right)

    monkeypatch.setattr(signs, "_draw_sign", spy)
    built = signs.build(found, _road(), LocalFrame(0.0, 0.0, 0.0), height=lambda e, n: np.zeros(np.shape(e)), **kw)
    return built, faces


def test_a_route_sign_stands_left_of_its_traffic_facing_it_and_moves_upstream_off_a_mouth(monkeypatch):
    found = {"signs": [{"kind": "give_way", "on": "route", "node": 1, "at_m": 200.0, "xy": [200.0, 0.0],
                        "travel": 0.0}]}
    built, faces = _drawn(monkeypatch, found)
    assert faces == [{"kind": "give_way", "e": 200.0, "n": 4.5, "facing": [-1.0, 0.0], "arrow_right": True,
                      "own": True, "bottom": signs.SIGN_BOTTOM_M}]
    # Against the route: on the right, facing east.
    found["signs"][0]["travel"] = math.pi
    _, faces = _drawn(monkeypatch, found)
    assert (faces[0]["n"], faces[0]["facing"]) == (-4.5, [1.0, 0.0])
    # A driveway's mouth where it would stand: it moves back, upstream, to the first clear station.
    found["signs"][0]["travel"] = 0.0
    blocked = {"L": (np.arange(401) >= 195) & (np.arange(401) <= 205), "R": np.zeros(401, bool)}
    _, faces = _drawn(monkeypatch, found, blocked=blocked)
    assert faces[0]["e"] == 194.0
    # Solid posts, as the signal posts are.
    walls = [m for m in built.meshes if m.name.startswith("1WALL_SIGN_")]
    assert walls and all(not m.renderable for m in walls)
    assert {m.material for m in built.meshes if m.renderable} == {signs.MATERIAL, signs.RETRO_MATERIAL}


def test_a_side_roads_sign_stands_beside_it_clear_of_its_stub_facing_its_traffic(monkeypatch):
    m = _mouth("L", 200.0, 190.0, 210.0)  # North Street, 3 m half width, north from station 200
    found = {"signs": [{"kind": "give_way", "on": "side_road", "node": 4, "junction": 0, "at_m": 200.0, "d_m": 5.0,
                        "xy": [200.0, 5.0], "travel": -math.pi / 2},
                       {"kind": "no_entry", "on": "side_road", "node": 2, "junction": 0, "at_m": 200.0, "d_m": 0.0,
                        "xy": [200.0, 0.0], "travel": math.pi / 2}]}
    keep = [m.corridor(signs.SIDE_CLEAR_M)]
    _, faces = _drawn(monkeypatch, found, mouths={0: m}, all_mouths=[m], keep_out=keep)
    gw = [f for f in faces if f["kind"] == "give_way"]
    # Left of southbound traffic is the east side; it faces north, up the side road, at the traffic.
    assert len(gw) == 1 and gw[0]["e"] == pytest.approx(200.0 + 3.0 + signs.SIDE_OFFSET_M)
    assert gw[0]["facing"] == [0.0, 1.0]
    # It stands clear of the route's edge (3.5 m) by SIGN_OFFSET_M and more: moved out along the side road.
    assert gw[0]["n"] >= 3.5 + signs.SIGN_OFFSET_M
    # NO ENTRY either side of the mouth, facing the route (south).
    ne = [f for f in faces if f["kind"] == "no_entry"]
    assert sorted(round(f["e"], 2) for f in ne) == [round(200 - 3 - signs.SIDE_OFFSET_M, 2),
                                                     round(200 + 3 + signs.SIDE_OFFSET_M, 2)]
    assert all(f["facing"] == [0.0, -1.0] for f in ne)
    # Outside the side road's carriageway and its clearance.
    assert all(not shapely.contains_xy(keep[0], f["e"], f["n"]) for f in faces)
    # A side road with no stub: nothing beside it.
    built, faces = _drawn(monkeypatch, found, mouths={}, all_mouths=[m], keep_out=keep)
    assert faces == [] and built.stats["left_out_by_reason"] == {"side_road_without_stub": 2}


def test_one_way_stands_across_from_the_side_road_its_arrow_along_the_flow(monkeypatch):
    m = _mouth("L", 200.0, 190.0, 210.0)
    found = {"signs": [{"kind": "one_way", "on": "route", "node": 2, "junction": 0, "at_m": 200.0, "xy": [200.0, 0.0],
                        "side": "R", "travel": 0.0}]}
    blocked = {"L": (np.arange(401) >= 190) & (np.arange(401) <= 210), "R": np.zeros(401, bool)}
    _, faces = _drawn(monkeypatch, found, mouths={0: m}, all_mouths=[m], blocked=blocked)
    # Straight across (the right verge, 4.5 m south), facing north at North Street's traffic; flowing east, which is
    # their right as they look south at it.
    assert faces == [{"kind": "one_way", "e": 200.0, "n": -4.5, "facing": [0.0, 1.0], "arrow_right": False,
                      "own": True, "bottom": signs.SIGN_BOTTOM_M}]
    assert signs._arrow_right(np.array([0.0, 1.0]), np.array([1.0, 0.0])) is False
    assert signs._arrow_right(np.array([0.0, -1.0]), np.array([1.0, 0.0])) is True
    # A crossroads: the far side has a mouth too, so it goes to the corner downstream of it.
    m2 = _mouth("R", 200.0, 193.0, 207.0, jid=1)
    blocked["R"] = (np.arange(401) >= 193) & (np.arange(401) <= 207)
    _, faces = _drawn(monkeypatch, found, mouths={0: m, 1: m2}, all_mouths=[m, m2], blocked=blocked)
    assert faces[0]["e"] == pytest.approx(208.0) and faces[0]["n"] == -4.5


def test_a_sign_beside_a_signal_post_goes_on_it(monkeypatch):
    found = {"signs": [{"kind": "give_way", "on": "route", "node": 1, "at_m": 200.0, "xy": [200.0, 0.0],
                        "travel": 0.0}]}
    built, faces = _drawn(monkeypatch, found, posts=[(200.3, 4.4, 0.0)])
    assert faces[0]["own"] is False and (faces[0]["e"], faces[0]["n"]) == (200.3, 4.4)
    assert built.stats["posts"] == 0 and built.stats["on_signal_posts"] == 1
    assert not [m for m in built.meshes if m.name.startswith("1WALL_SIGN_")]  # its post is already solid


def test_a_two_way_crossings_signs_stand_both_sides_back_to_back(monkeypatch):
    found = {"signs": [{"kind": "pedestrian_crossing", "on": "route", "node": 3, "at_m": 200.0, "xy": [200.0, 0.0],
                        "ways": "both"}]}
    _, faces = _drawn(monkeypatch, found)
    assert sorted((f["n"], tuple(f["facing"])) for f in faces) == [(-4.5, (-1.0, 0.0)), (-4.5, (1.0, 0.0)),
                                                                   (4.5, (-1.0, 0.0)), (4.5, (1.0, 0.0))]
    assert {f["bottom"] for f in faces} == {signs.SIGN_BOTTOM_M}  # back to back, at one height


# ----- the zebra -----

def test_zebra_stripes_cross_the_carriageway_along_the_road_on_its_surface():
    tex = signs.atlas()
    b = signals._Builder(tex, (signs.MATERIAL, signs.RETRO_MATERIAL, signs.PAINT_MATERIAL))
    count = signs._zebra(b, np.array([100.0, 0.0]), np.array([1.0, 0.0]), 7.0, lambda e, n: np.full(np.shape(e), 2.0),
                         2.0)
    # 7 m less 0.3 m at each edge: 6.4 m holds five 0.6 m stripes with 0.6 m gaps (5.4 m), centred.
    assert count == 5
    pos = np.array(b.parts[signs.PAINT_MATERIAL][0])
    assert np.allclose(pos[:, 2], 2.0 + signs.ZEBRA_LIFT_M)  # on the road, lifted clear of it
    assert pos[:, 0].min() == pytest.approx(100 - signs.ZEBRA_LENGTH_M / 2)  # along the road, 3 m long
    assert pos[:, 0].max() == pytest.approx(100 + signs.ZEBRA_LENGTH_M / 2)
    assert pos[:, 1].min() == pytest.approx(-2.7) and pos[:, 1].max() == pytest.approx(2.7)  # across it, centred
    xs = np.unique(np.round(pos[:, 1], 3))
    assert len(xs) == 15  # three columns a stripe, so each follows a crowned road's surface
    # A stripe with no road under part of it, or under a deck far above the road, isn't drawn.
    b2 = signals._Builder(tex, (signs.MATERIAL, signs.RETRO_MATERIAL, signs.PAINT_MATERIAL))
    holey = signs._zebra(b2, np.array([100.0, 0.0]), np.array([1.0, 0.0]), 7.0,
                         lambda e, n: np.where(np.asarray(n) > 1.0, np.nan, 0.0), 0.0)
    assert holey == 3
    assert signs._zebra(b2, np.array([0.0, 0.0]), np.array([1.0, 0.0]), 7.0, lambda e, n: np.full(np.shape(e), 9.0),
                        0.0) == 0
    assert signs._zebra(b2, np.array([0.0, 0.0]), np.array([1.0, 0.0]), 1.0, lambda e, n: np.zeros(np.shape(e)),
                        0.0) == 0  # too narrow for a stripe


def test_a_zebra_on_the_route_is_drawn_as_its_own_visual_mesh(monkeypatch):
    found = {"signs": [], "zebras": [{"on": "route", "node": 3, "at_m": 200.0, "xy": [200.0, 0.0], "heading": 0.0,
                                      "ways": "both"}]}
    built, _ = _drawn(monkeypatch, found, road_height=lambda e, n: np.zeros(np.shape(e)))
    assert built.stats["zebras"] == 1 and built.stats["zebra_stripes"] == 5
    paint = [m for m in built.meshes if m.material == signs.PAINT_MATERIAL]
    assert paint and all(m.renderable and m.name.startswith("SIGN_") for m in paint)  # visual only: no surface


def test_the_validator_checks_the_sign_posts_colliders():
    assert ("regulatory sign post", "1WALL_SIGN_") in validate.SCENERY_COLLIDERS
    assert any("1WALL_SIGN_" in prefixes for _, prefixes, _ in validate.JUNCTION_CLEAR)
    # Distinct from the signals' own: neither prefix catches the other's meshes.
    assert not "1WALL_SIGNAL_0".startswith("1WALL_SIGN_") and not "1WALL_SIGN_0".startswith("1WALL_SIGNAL_")
