"""Traffic signals and street-name blades (`core.signals`): finding the sites, placing the posts and faces, the blades,
the flashing amber and the sign font."""

import math

import numpy as np
import pytest
from pyproj import Transformer

from mapcreator.core import furniture, junctions, lettering, package, signals
from mapcreator.core.scene import LocalFrame
from mapcreator.providers.base import LONLAT_CRS
from mapcreator.providers.osm import graph

TO_XY = Transformer.from_crs(LONLAT_CRS, 28355, always_xy=True)
LAT = -42.8


def _osm(main_tags: dict, nodes_tags: dict | None = None) -> dict:
    """Main Road runs east along LAT through a crossroads at 147.305 (North Street one-way into it, South Street one-way
    out of it), with a node at 147.3075 for a mid-block crossing and North Street running on to a far node."""
    pts = {1: (147.300, LAT), 6: (147.3025, LAT), 2: (147.305, LAT), 7: (147.3075, LAT), 3: (147.310, LAT),
           4: (147.305, LAT + 0.001), 8: (147.305, LAT + 0.0035), 5: (147.305, LAT - 0.001)}
    tags = nodes_tags or {}
    nodes = [{"type": "node", "id": i, "lon": lon, "lat": lat, **({"tags": tags[i]} if i in tags else {})}
             for i, (lon, lat) in pts.items()]
    ways = [
        {"type": "way", "id": 10, "nodes": [1, 6, 2, 7, 3], "tags": {"highway": "primary", "name": "Main Road", **main_tags}},
        {"type": "way", "id": 11, "nodes": [8, 4, 2], "tags": {"highway": "secondary", "name": "North Street",
                                                               "oneway": "yes", "lanes": "3"}},
        {"type": "way", "id": 12, "nodes": [2, 5], "tags": {"highway": "secondary", "name": "South Street",
                                                           "oneway": "yes", "lanes": "2"}},
    ]
    return {"elements": nodes + ways}


MAIN = {"lanes": "5", "lanes:forward": "3", "lanes:backward": "2"}
SIGNALS = {2: {"highway": "traffic_signals"}, 7: {"highway": "crossing", "crossing": "traffic_signals"},
           8: {"highway": "traffic_signals"}}


def _found(main=MAIN, node_tags=SIGNALS, start=147.3005):
    data = _osm(main, node_tags)
    g = graph.build(data, TO_XY)
    route = graph.route(g, np.column_stack(TO_XY.transform([start, 147.3095], [LAT, LAT])))
    nodes = []
    for e in data["elements"]:
        if e["type"] == "node" and e.get("tags"):
            nodes.append({"id": e["id"], "xy": TO_XY.transform(e["lon"], e["lat"]), "tags": e["tags"]})
    return g, route, signals.find(g, route.xy, nodes)


# ----- finding the sites -----

def test_a_signalised_crossroads_and_a_mid_block_crossing_are_found_with_their_legs():
    g, route, found = _found()
    sites = found["sites"]
    assert [s["kind"] for s in sites] == ["junction", "mid_block"] and [s["node"] for s in sites] == [2, 7]
    legs = {(leg["route"] or leg["side"]): leg for leg in sites[0]["legs"]}
    assert set(legs) == {"before", "after", "L", "R"}
    assert legs["L"]["name"] == "North Street" and legs["R"]["name"] == "South Street"
    # Who may drive in: both ways on Main Road, into the junction on North Street, never on South Street.
    assert legs["before"]["inbound"] and legs["after"]["inbound"] and legs["L"]["inbound"]
    assert not legs["R"]["inbound"]
    # Lanes in the direction of travel: lanes:forward along Main Road's own direction, lanes:backward against it,
    # and all of a one-way road's.
    assert (legs["before"]["lanes"], legs["after"]["lanes"], legs["L"]["lanes"]) == (3, 2, 3)
    # North Street heads north, out of the junction.
    assert math.degrees(legs["L"]["heading"]) == pytest.approx(90.0, abs=2.0)
    assert sites[0]["signals"] == [2] and not sites[0]["crossings_mapped"]
    # The far signals on North Street, 390 m off the route, are nobody's site.
    assert all(8 not in s["signals"] for s in sites) and found["skipped"] == []


def test_lanes_in_each_direction():
    assert signals.lanes_in({"lanes": "4"}, True) == 2  # two-way, untagged by direction: half each way
    assert signals.lanes_in({"lanes": "4", "oneway": "yes"}, True) == 4
    assert signals.lanes_in({"lanes": "3", "oneway": "-1"}, False) == 3
    assert signals.lanes_in({"lanes": "5", "lanes:forward": "3", "lanes:backward": "2"}, False) == 2
    assert signals.lanes_in({"lanes": "1"}, True) == 1 and signals.lanes_in({}, True) is None
    assert signals.lanes_in({"lanes": "two"}, True) is None


def test_mapped_crossings_choose_the_legs_with_pedestrian_lanterns():
    tags = dict(SIGNALS)
    tags[4] = {"highway": "crossing", "crossing": "traffic_signals"}  # on North Street, 111 m out: too far
    _, _, found = _found(node_tags=tags)
    assert not found["sites"][0]["crossings_mapped"]
    assert len(signals.pedestrian_legs(found["sites"][0])) == 4  # none mapped near: every leg (Q3)
    site = {"legs": [{"route": "before", "crossing": True}, {"route": "after", "crossing": False},
                     {"side": "L", "route": None, "crossing": False}], "crossings_mapped": True}
    assert signals.pedestrian_legs(site) == [site["legs"][0]]


def test_sites_near_the_routes_ends_and_on_roundabouts_are_left_out():
    tags = {6: {"highway": "traffic_signals"}}  # 41 m from the route's start
    _, _, found = _found(node_tags=tags, start=147.302)
    assert found["sites"] == [] and found["skipped"] == [{"signals": [6], "node": 6, "reason": "route_end"}]
    data = _osm({"junction": "roundabout"}, {2: {"highway": "traffic_signals"}})
    g = graph.build(data, TO_XY)
    route = graph.route(g, np.column_stack(TO_XY.transform([147.3005, 147.3095], [LAT, LAT])))
    nodes = [{"id": 2, "xy": TO_XY.transform(147.305, LAT), "tags": {"highway": "traffic_signals"}}]
    assert signals.find(g, route.xy, nodes)["skipped"][0]["reason"] == "roundabout"


def test_no_tagged_nodes_no_sites():
    g, route, _ = _found()
    assert signals.find(g, route.xy, [])["sites"] == []


# ----- placing the posts and faces -----

def test_each_approach_gets_a_primary_secondary_and_tertiary_face_on_the_right_corners():
    _, _, found = _found()
    plan = signals.face_plan(found["sites"][0], np.array([1.0, 0.0]))
    by_role = {(key, role): facing for key, facing, role, _ in plan}
    # Eastbound (the route's own direction, from before the junction): near left, far right, far left, facing west.
    assert np.allclose(by_role[(("L", "before"), "primary")], [-1, 0])
    assert np.allclose(by_role[(("R", "after"), "secondary")], [-1, 0])
    assert np.allclose(by_role[(("L", "after"), "tertiary")], [-1, 0])
    # Westbound: the mirror image, facing east.
    assert np.allclose(by_role[(("R", "after"), "primary")], [1, 0])
    assert np.allclose(by_role[(("L", "before"), "secondary")], [1, 0])
    # Southbound out of North Street (on the left): its primary on its near left, the left side's far corner, facing
    # north up North Street.
    north = by_role[(("L", "after"), "primary")]
    assert north[1] == pytest.approx(1.0, abs=0.01)
    assert (("R", "before"), "secondary") in by_role and (("R", "after"), "tertiary") in by_role
    # South Street is one-way out of the junction: no approach, so 3 approaches and 9 faces.
    assert len(plan) == 9


def _road(length=400.0):
    s = np.arange(0.0, length + 0.5, 1.0)
    xy = np.column_stack([s, np.zeros_like(s)])
    z = np.zeros_like(s)
    return signals._Road(s, xy, z, np.zeros_like(s), np.full_like(s, 3.5), np.full_like(s, 3.5), np.zeros_like(s),
                         np.zeros_like(s))


def _mouth(side, at, s0, s1, node=2):
    return junctions.Mouth({"node": node}, side, at, s0, s1, np.array([[at, 0.0], [at, 50.0]]), 4.0)


def _site(kind="junction", node=2, at=200.0, before_lanes=3, crossings_mapped=False, **legs):
    base = [{"route": "before", "side": None, "inbound": True, "lanes": before_lanes, "heading": math.pi,
             "crossing": False},
            {"route": "after", "side": None, "inbound": True, "lanes": 2, "heading": 0.0, "crossing": False}]
    if kind == "junction":
        base += [{"route": None, "side": "L", "inbound": True, "lanes": 3, "heading": math.pi / 2, "crossing": True},
                 {"route": None, "side": "R", "inbound": False, "lanes": 2, "heading": -math.pi / 2,
                  "crossing": False}]
    return {"id": 0, "node": node, "kind": kind, "at_m": at, "xy": [at, 0.0], "legs": base,
            "crossings_mapped": crossings_mapped}


def test_posts_stand_at_the_corners_beside_the_mouths_and_masts_go_where_three_lanes_approach():
    road = _road()
    frame = LocalFrame(0.0, 0.0, 0.0)
    built = signals.build([_site()], [], road, frame, mouths=[_mouth("L", 200, 190, 212), _mouth("R", 200, 194, 206)])
    rec = built.records[0]
    assert rec["posts"] == {"L_after": 213.0, "L_before": 189.0, "R_after": 207.0, "R_before": 193.0}
    # Three lanes eastbound and three out of North Street: their primaries are mast arms; two westbound: a post.
    assert rec["mast_arms"] == ["L_after", "L_before"]
    st = built.stats
    assert (st["posts"], st["mast_arms"], st["vehicle_faces"]) == (4, 2, 9 + 2)
    # No crossing mapped at the site: lanterns on every leg (4 crosswalks), a lantern and a button at each end.
    assert (st["pedestrian_lanterns"], st["buttons"], rec["crosswalks"]) == (8, 8, 4)
    # Each post stands 0.75 m beyond the road's 3.5 m half width, and is solid; the rest is visual only.
    walls = [m for m in built.meshes if m.name.startswith("1WALL_SIGNAL_")]
    assert walls and all(not m.renderable for m in walls)
    x, z = walls[0].positions[:, 0], walls[0].positions[:, 2]
    assert {round(float(v), 2) for v in np.abs(z)} <= {round(3.5 + 0.75 - w / 2, 2) for w in (signals.COLLIDER_W,
                                                                                    signals.MAST_D + 0.04)} | {
        round(3.5 + 0.75 + w / 2, 2) for w in (signals.COLLIDER_W, signals.MAST_D + 0.04)}
    assert {m.material for m in built.meshes if m.renderable} == {signals.MATERIAL, signals.RETRO_MATERIAL,
                                                                 signals.AMBER_MATERIAL}
    # Two lanes eastbound: no mast arm on that corner.
    assert signals.build([_site(before_lanes=2)], [], road, frame, mouths=[_mouth("L", 200, 190, 212)]
                         ).records[0]["mast_arms"] == ["L_after"]


def test_mapped_crossings_put_lanterns_only_on_their_legs_and_a_mid_block_crossing_has_one():
    road, frame = _road(), LocalFrame(0.0, 0.0, 0.0)
    built = signals.build([_site(crossings_mapped=True)], [], road, frame)
    assert built.records[0]["crosswalks"] == 1 and built.stats["pedestrian_lanterns"] == 2
    mid = signals.build([_site("mid_block", node=7)], [], road, frame)
    rec = mid.records[0]
    # A mid-block crossing's corners: 3 m either side of its node, and one crosswalk across the road.
    assert rec["posts"] == {"L_after": 204.0, "L_before": 196.0, "R_after": 204.0, "R_before": 196.0}
    assert rec["crosswalks"] == 1 and mid.stats["vehicle_faces"] == 6 + 1


def test_a_t_junctions_far_side_takes_the_side_roads_mouth_and_posts_keep_out_of_other_mouths():
    stations = signals.corner_stations(200.0, {"L": (190.0, 212.0)}, signals.JUNCTION_HALF_M)
    assert stations == {("L", "before"): 189.5, ("L", "after"): 212.5, ("R", "before"): 189.5, ("R", "after"): 212.5}
    # No mouth at all: JUNCTION_HALF_M either side. A fork's long mouth: the corner stays within CORNER_REACH_M.
    assert signals.corner_stations(200.0, {}, 10.0)[("R", "after")] == 210.5
    assert signals.corner_stations(200.0, {"R": (195.0, 400.0)}, 10.0)[("R", "after")] == 200 + 25 + 0.5
    road = _road()
    blocked = {"L": np.zeros(len(road.s), bool), "R": np.zeros(len(road.s), bool)}
    blocked["L"][180:190] = True  # a driveway's mouth just before the corner
    assert signals.clear_station(road, 189.5, "L", "before", blocked) == 179  # moved away from the junction
    blocked["L"][150:190] = True
    assert signals.clear_station(road, 189.5, "L", "before", blocked) is None  # more than MAX_SHIFT_M: no post
    assert signals.clear_station(road, 189.5, "R", "before", blocked) == 189
    assert signals.clear_station(road, -5.0, "R", "before", None) is None


def test_divided_side_roads_widen_the_corner_but_a_far_junction_does_not():
    mouths = [_mouth("R", 200, 194, 206), _mouth("R", 208, 202, 222, node=9), _mouth("R", 260, 240, 290, node=11)]
    assert signals.junction_spans(mouths)[2]["R"] == (194, 222)


# ----- street-name blades -----

def _route(length=400.0):
    s = np.arange(0.0, length + 0.5, 1.0)
    return s, np.column_stack([s, np.zeros_like(s)]), np.zeros_like(s), np.zeros(len(s), int)


def _junction(node, side, name, at=200.0, angle=90.0, off=90.0, turn=0.0):
    return {"node": node, "side": side, "name": name, "at_m": at, "xy": [at, 0.0], "angle_deg": angle,
            "off_mainline_deg": off, "route_turn_deg": turn}


def test_blades_go_on_diagonal_corners_at_a_crossroads_and_the_stems_corners_at_a_t():
    s, xy, heading, run = _route()
    runs = [{"tags": {"name": "Main Road"}}]
    station = lambda j: float(j["at_m"])  # noqa: E731
    cross = signals.blades_for([_junction(2, "left", "North Street"), _junction(2, "right", "North Street",
                                                                              angle=-90.0)],
                               s, xy, heading, run, runs, station)
    assert len(cross) == 1 and cross[0].corners == [("L", "before"), ("R", "after")]
    assert [n for n, _ in cross[0].names] == ["Main Road", "North Street"]
    # Each name runs along its own road: Main Road east, North Street north.
    assert cross[0].names[0][1] == pytest.approx(0.0) and cross[0].names[1][1] == pytest.approx(math.pi / 2)
    tee = signals.blades_for([_junction(2, "right", "South Street", angle=-90.0)], s, xy, heading, run, runs, station)
    assert tee[0].corners == [("R", "before"), ("R", "after")]
    # The route turns left: the corner on the outside of the turn (the right) carries one.
    turn = signals.blades_for([_junction(2, "left", "North Street", turn=90.0)], s, xy, heading, run, runs, station)
    assert ("R", "after") in turn[0].corners


def test_no_blades_where_the_names_match_or_the_side_road_runs_beside_the_route():
    s, xy, heading, run = _route()
    station = lambda j: float(j["at_m"])  # noqa: E731
    runs = [{"tags": {"name": "Main Road"}}]
    assert signals.blades_for([_junction(2, "left", "Main Road")], s, xy, heading, run, runs, station) == []
    assert signals.blades_for([_junction(2, "left", None)], s, xy, heading, run, runs, station) == []
    # A bus lane beside the road, leaving it at 28 degrees: no blade.
    lane = _junction(2, "right", "Main Road Bus Contra-Lane", angle=-152.0, off=28.0)
    assert signals.blades_for([lane], s, xy, heading, run, runs, station) == []
    # A divided side road's two arms 20 m apart: one set of blades. Not within 70 m of the route's ends.
    arms = [_junction(2, "left", "Market Place"), _junction(3, "left", "Market Place", at=220.0)]
    assert len(signals.blades_for(arms, s, xy, heading, run, runs, station)) == 1
    assert signals.blades_for([_junction(2, "left", "North Street", at=50.0)], s, xy, heading, run, runs,
                              station) == []


def test_blade_text_is_abbreviated_in_the_chosen_case(monkeypatch):
    assert signals.abbreviate("Macquarie Street") == "Macquarie St"
    assert signals.abbreviate("Brooker Highway") == "Brooker Hwy"
    assert signals.abbreviate("Sandy Bay Road") == "Sandy Bay Rd"
    assert signals.abbreviate("Southern Outlet") == "Southern Outlet"  # not a type it abbreviates
    assert signals.abbreviate("The Avenue") == "The Avenue"  # never after "The"
    assert signals.abbreviate("Street") == "Street"  # nor a name's only word
    assert signals.abbreviate("Street Road") == "Street Rd"  # only the last word
    monkeypatch.setattr(signals, "BLADE_CASE", "upper")
    assert signals.abbreviate("Macquarie Street") == "MACQUARIE ST"


def test_blades_are_built_lettered_on_both_faces_and_on_signal_posts_where_there_are_some():
    road, frame = _road(), LocalFrame(0.0, 0.0, 0.0)
    blade = signals.BladeSite(2, 200.0, [("Main Road", 0.0), ("North Street", math.pi / 2)],
                              [("L", "before"), ("R", "after")])
    alone = signals.build([], [blade], road, frame, mouths=[_mouth("L", 200, 190, 212)])
    assert (alone.stats["blade_posts"], alone.stats["blades"], alone.stats["blades_on_signal_posts"]) == (2, 4, 0)
    assert set(alone.atlas.blade_m) == {"Main Rd", "North St"}
    assert signals.BLADE_MIN_M <= min(alone.atlas.blade_m.values()) <= max(alone.atlas.blade_m.values()) <= 1.2
    with_signals = signals.build([_site()], [blade], road, frame, mouths=[_mouth("L", 200, 190, 212)])
    assert with_signals.stats["blade_posts"] == 0 and with_signals.stats["blades_on_signal_posts"] == 2
    # Lettered faces have no specular (the night washout lesson) and the blades are retroreflective.
    assert all(p["ksSpecular"] == 0.0 for p in signals.KN5_PROPERTIES.values())
    assert signals.NIGHT_DIFFUSE == {signals.RETRO_MATERIAL: 2.0}


def test_a_long_name_fits_its_blade():
    tex = signals.atlas(["Macquarie St", "A Very Long Street Name Indeed Esp"])
    assert tex.blade_m["A Very Long Street Name Indeed Esp"] == pytest.approx(signals.BLADE_MAX_M)
    assert tex.image.size[0] == signals.ATLAS_W and tex.image.size[1] % 4 == 0


# ----- the flashing amber and the font -----

def test_the_amber_flashes_at_one_hertz_through_ext_config():
    text = package.ext_config_ini(grass_materials=["grass"], road_materials=["road"], version="t",
                                  retro_materials={"furniture_retro": (2.0, 0.6)},
                                  flashing=([signals.AMBER_MATERIAL], signals.flash_condition(),
                                            signals.FLASH_EMISSIVE))
    lines = text.split("\r\n")
    cond = lines.index("[CONDITION_...]")
    assert lines[cond + 1:cond + 9] == ["NAME=SIGNALS_FLASHING", "INPUT=ONE", "FLASHING_FREQUENCY=1",
                                        "FLASHING_SMOOTHNESS=0.1", "FLASHING_SKIP_OFF_STATE=0",
                                        "FLASHING_NOISE_AMPLITUDE=0", "FLASHING_MIN_VALUE=0.0", "FLASHING_SYNCED=1"]
    adj = lines.index("[MATERIAL_ADJUSTMENT_1]")  # after the one retro material's
    assert "MATERIALS=signal_amber" in lines[adj:adj + 9] and "KEY_0=ksEmissive" in lines[adj:adj + 9]
    assert "CONDITION=SIGNALS_FLASHING" in lines[adj:adj + 9] and "VALUE_0_OFF=0, 0, 0" in lines[adj:adj + 9]
    plain = package.ext_config_ini(grass_materials=["grass"], road_materials=["road"], version="t")
    assert "CONDITION" not in plain and "ksEmissive" not in plain


def test_the_sign_font_is_overpass_and_the_speed_signs_use_it():
    f = lettering.font(40)
    assert f.getname() == ("Overpass", "SemiBold")
    from importlib import resources
    licence = (resources.files("mapcreator.core") / "fonts" / lettering.LICENCE_FILE).read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE Version 1.1" in licence
    # The speed signs' numerals are drawn in it: the atlas differs from one drawn in Pillow's built-in font.
    from PIL import ImageFont
    tex = furniture.atlas([50])
    drawn = np.asarray(tex.image)
    import mapcreator.core.lettering as lt
    orig = lt.font
    try:
        lt.font = lambda size: ImageFont.load_default(size=size)
        other = np.asarray(furniture.atlas([50]).image)
    finally:
        lt.font = orig
    assert (drawn != other).any()
