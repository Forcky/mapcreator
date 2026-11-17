"""A route that goes back the way it came is refused as the person's to fix, in the build and in the preview."""

import json
import re
from dataclasses import replace

import httpx
import numpy as np
import pytest
from pyproj import Transformer

from mapcreator import config
from mapcreator.core import centreline
from mapcreator.job import Job, Waypoint
from mapcreator.pipeline import route_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.osm import graph
from mapcreator.service import worker
from mapcreator.service.routing import DevRouter, GraphHopperRouter
IDENT = Transformer.from_crs(4326, 4326, always_xy=True)
HOBART = [Waypoint(lon=147.301, lat=-42.8001), Waypoint(lon=147.319, lat=-42.7999)]
OVERPASS = replace(config.load(), osm_index="")  # roads from the fake Overpass below, whatever the local config


def straight_road() -> dict:
    """A secondary road running east along latitude -42.80, nodes every 0.002 degrees (as test_service_routing)."""
    nodes = [{"type": "node", "id": i + 1, "lon": 147.300 + i * 0.002, "lat": -42.80} for i in range(11)]
    way = {"type": "way", "id": 100, "nodes": [n["id"] for n in nodes], "tags": {"highway": "secondary"}}
    return {"elements": [*nodes, way]}


def test_retraced_measures_road_driven_twice_and_where_it_starts():
    assert centreline.retraced(np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])) == (0.0, None)
    # Out to (200, 0) and back to the start: the second half is road already driven, from the far end.
    out_and_back = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [100.0, 0.0], [0.0, 0.0]])
    assert centreline.retraced(out_and_back) == (pytest.approx(200.0), 2)
    # A side trip up a 50 m spur and back, then on.
    spur = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [100.0, 0.0], [200.0, 0.0]])
    assert centreline.retraced(spur) == (pytest.approx(50.0), 2)
    # Round a block twice, the same way: the second lap is road driven twice too.
    block = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]] * 2 + [[0.0, 0.0]])
    length, first = centreline.retraced(block)
    assert length == pytest.approx(400.0) and first == 4


@pytest.mark.parametrize("xy", [
    [[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]],
    [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]],  # a right-angle junction
    [[0.0, 0.0], [100.0, 0.0], [100.0 - 30.6, 94.2]],  # 108 degrees, the sharpest any built route has had
], ids=["straight", "right angle", "108 degrees"])
def test_good_routes_dont_turn_back(xy):
    assert route_stage.turn_back(np.array(xy)) is None


def test_turn_back_finds_road_driven_twice_and_a_reversal_on_the_spot():
    out_and_back = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [100.0, 0.0]])
    assert route_stage.turn_back(out_and_back) == (2, pytest.approx(100.0))
    # Two points on one stretch with no junction between: out to 90 m, back to 50 m, with no node shared.
    on_the_spot = np.array([[10.0, 0.0], [90.0, 0.0], [50.0, 0.0]])
    assert route_stage.turn_back(on_the_spot) == (1, 0.0)


def test_the_message_says_where_and_how_much():
    assert route_stage.turns_back_message(4721.4, 5677.1) == (
        "The route turns back on itself 5.7 km along and drives 4.7 km of road twice. A track can't go back the "
        "way it came: move the point near there onto the road you mean, or finish the track before it turns.")
    assert route_stage.turns_back_message(0.0, 80.0).startswith("The route turns back on itself 80 m along. ")


def test_a_routed_out_and_back_is_a_user_facing_error():
    # A road east along y = 0 with a junction at 100 m; the second point is a side street's end.
    g = graph.build({"elements": [
        *({"type": "node", "id": i, "lon": x, "lat": y} for i, (x, y) in
          {1: (0, 0), 2: (100, 0), 3: (200, 0), 4: (100, 60)}.items()),
        {"type": "way", "id": 10, "nodes": [1, 2, 3], "tags": {"highway": "secondary"}},
        {"type": "way", "id": 11, "nodes": [2, 4], "tags": {"highway": "residential"}},
    ]}, IDENT)
    r = graph.route(g, np.array([[0.0, 0.0], [100.0, 60.0], [200.0, 0.0]]))
    back, retraced_m = route_stage.turn_back(r.xy)
    assert retraced_m == pytest.approx(r.retraced_m) == pytest.approx(60.0)
    assert centreline.cumulative(r.xy)[back] == pytest.approx(160.0)  # at the side street's end
    exc = route_stage.RouteTurnsBack(route_stage.turns_back_message(retraced_m, 160.0))
    assert isinstance(exc, graph.RouteError)
    assert worker.public_error(exc) == str(exc)  # shown as is, not as an internal error


def test_dev_router_preview_says_where_it_turns_back():
    good = DevRouter(settings=OVERPASS, fetch=lambda q, s: straight_road()).preview(HOBART)
    assert "turns_back" not in good
    there_and_back = [HOBART[0], HOBART[1], Waypoint(lon=147.305, lat=-42.80)]
    preview = DevRouter(settings=OVERPASS, fetch=lambda q, s: straight_road()).preview(there_and_back)
    turn = preview["turns_back"]
    assert [turn["lon"], turn["lat"]] == pytest.approx([147.319, -42.80], abs=1e-5)
    assert turn["along_m"] == pytest.approx(good["length_m"], rel=0.01)
    assert turn["message"].startswith(f"The route turns back on itself {turn['along_m'] / 1000:.1f} km along and "
                                      f"drives {turn['retraced_m'] / 1000:.1f} km of road twice.")


def test_graphhopper_preview_says_where_it_turns_back():
    coords = [[147.301, -42.8], [147.31, -42.8], [147.319, -42.8], [147.31, -42.8], [147.305, -42.8]]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"paths": [{"distance": 2500.0,
                                                    "points": {"type": "LineString", "coordinates": coords}}]})

    router = GraphHopperRouter("http://router.example.org:8989/", client=httpx.Client(transport=httpx.MockTransport(handler)))
    turn = router.preview(HOBART)["turns_back"]
    assert [turn["lon"], turn["lat"]] == [147.319, -42.8]
    assert turn["retraced_m"] == pytest.approx(0.009 * 111_320 * np.cos(np.radians(42.8)), rel=0.01)
    assert re.search(r" along and drives 73\d m of road twice\. ", turn["message"])


# -- a U-turn through a divided road's median ------------------------------------------------------------------------

MEDIAN_UTURN = [[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [200.0, -9.6], [180.0, -9.6]]  # east, through a 9.6 m gap, west
TO_LONLAT = Transformer.from_crs(28355, 4326, always_xy=True)
TO_MGA = Transformer.from_crs(4326, 28355, always_xy=True)


def test_a_u_turn_through_a_median_gap_is_found_where_turn_back_sees_nothing():
    # Nov 2026: a point on the far side of a divided road sent a route through a 10 m gap in the median and back down
    # the other carriageway. Two right angles, no road driven twice: turn_back passed it, and the build failed.
    xy = np.array(MEDIAN_UTURN)
    assert route_stage.turn_back(xy) is None
    assert route_stage.u_turn(xy) == 2  # the node where it starts
    # Off the road through the gap (often unnamed) and back onto it: the same, with the roads' names.
    assert route_stage.u_turn(xy, names=["Main Road", "Main Road", None, "Main Road"]) == 2
    assert route_stage.u_turn(xy, names=["Main Road", "Main Road", "Cross Street", "Main Road"]) == 2


def test_a_hairpin_on_one_road_is_left_to_validation():
    # A 3 m hairpin, as some of Tasmania's mountain roads and court ends are drawn: a U-turn by its shape, but on one
    # road, so validation judges it, and a build can still have it on request.
    arc = [[3.0 * np.sin(a), 3.0 * np.cos(a)] for a in np.radians(np.arange(0, 181, 30))]
    xy = np.array([[-50.0, 3.0], *arc, [-50.0, -3.0]])
    assert route_stage.u_turn(xy) is not None
    assert route_stage.u_turn(xy, names=["Saddle Road"] * (len(xy) - 1)) is None
    assert route_stage.u_turn(xy, names=[None] * (len(xy) - 1)) is None  # an unnamed road is one road too


def test_a_turn_into_a_tight_loop_on_another_road_is_left_to_validation():
    # Off a named road into a court's turning circle, or from an unnamed link onto a road at a sharp angle (the survey
    # of Tasmania's roundabouts, Nov 2026): the name changes, but the route doesn't come back onto the road it left.
    arc = [[3.0 * np.sin(a), 3.0 * np.cos(a)] for a in np.radians(np.arange(0, 181, 30))]
    xy = np.array([[-50.0, 3.0], *arc, [-50.0, -3.0]])
    n = len(xy) - 1
    assert route_stage.u_turn(xy, names=["Main Road"] + ["Smith Court"] * (n - 1)) is None
    assert route_stage.u_turn(xy, names=[None] * (n - 1) + ["Main Road"]) is None
    # Into the court and straight back out onto the road it left: a U-turn.
    assert route_stage.u_turn(xy, names=["Main Road"] + ["Smith Court"] * (n - 2) + ["Main Road"]) == 1


@pytest.mark.parametrize("xy", [
    [[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]],
    [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]],  # a right-angle junction
    [[0.0, 0.0], [100.0, 0.0], [100.0 - 30.6, 94.2]],  # 108 degrees, the sharpest any built route has had
    [[0.0, 0.0], [100.0, 0.0], [100.0, -30.0], [0.0, -30.0]],  # round a short block: two right turns 30 m apart
    [[-50.0, 8.0], *[[8.0 * np.sin(a), 8.0 * np.cos(a)] for a in np.radians(np.arange(0, 181, 15))], [-50.0, -8.0]],
], ids=["straight", "right angle", "108 degrees", "round a block", "8 m hairpin"])
def test_good_routes_make_no_u_turn(xy):
    assert route_stage.u_turn(np.array(xy)) is None


def test_the_path_round_a_small_roundabout_is_left_to_validation():
    # In, three quarters of the way round a 4 m ring (a right turn), and out: as hard a turn as a U-turn, but a build
    # can go round it on request, so u_turn leaves the ring's stretch alone.
    ring = [[4.0 * np.cos(a), 4.0 * np.sin(a)] for a in np.radians(np.arange(-90, -360, -15))]
    xy = np.array([[0.0, -60.0], [0.0, -4.0], *ring[1:], [-60.0, 0.0]])
    s = centreline.cumulative(xy)
    assert route_stage.u_turn(xy) is not None
    assert route_stage.u_turn(xy, [(float(s[1]), float(s[len(ring)]))]) is None


def test_ring_spans_are_the_stretches_on_a_roundabout_s_ways():
    xy = np.array([[0.0, 0.0], [50.0, 0.0], [55.0, 5.0], [60.0, 0.0], [110.0, 0.0]])
    r = graph.Route(xy=xy, seg_way=np.array([0, 1, 1, 0]), retraced_m=0.0,
                    ways=[{"id": 1, "tags": {"highway": "primary"}},
                          {"id": 2, "tags": {"highway": "primary", "junction": "roundabout"}}])
    ((a, b),) = route_stage.ring_spans(r)
    assert (a, b) == pytest.approx((50.0, 50.0 + 2 * np.hypot(5.0, 5.0)))


def test_the_u_turn_message_says_where_and_what_to_do():
    assert route_stage.u_turn_message(4883.2) == (
        "The route makes a U-turn 4.9 km along, usually because a point is on the far side of a divided road. A "
        "track can't go back the way it came: move the point near there onto the same side of the road as the "
        "route, or finish the track before it turns.")


def divided_road() -> dict:
    """A divided road east-west along latitude -42.80: the north carriageway one way east, the south one 10 m away
    one way west, and an unnamed gap in the median at 147.306."""
    lons = [147.300 + i * 0.002 for i in range(6)]
    north = [{"type": "node", "id": i + 1, "lon": x, "lat": -42.80} for i, x in enumerate(lons)]
    south = [{"type": "node", "id": i + 11, "lon": x, "lat": -42.80009} for i, x in enumerate(lons)]
    return {"elements": [*north, *south,
                         {"type": "way", "id": 1, "nodes": [n["id"] for n in north],
                          "tags": {"highway": "secondary", "oneway": "yes", "name": "Main Road"}},
                         {"type": "way", "id": 2, "nodes": [n["id"] for n in south][::-1],
                          "tags": {"highway": "secondary", "oneway": "yes", "name": "Main Road"}},
                         {"type": "way", "id": 3, "nodes": [4, 14], "tags": {"highway": "residential"}}]}


ACROSS_THE_MEDIAN = [Waypoint(lon=147.301, lat=-42.80), Waypoint(lon=147.303, lat=-42.80009)]  # the finish: far side


def test_the_build_refuses_a_u_turn_through_the_median_as_the_person_s_to_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(route_stage.osm_roads, "for_bbox", lambda bbox, settings: divided_road())
    with pytest.raises(route_stage.RouteTurnsBack, match=r"^The route makes a U-turn 4\d\d m along") as exc:
        route_stage.run(Job(track_id="median", waypoints=ACROSS_THE_MEDIAN), config.load(), WorkDir(tmp_path / "w"))
    assert worker.public_error(exc.value) == str(exc.value)


def test_dev_router_preview_marks_a_u_turn_through_the_median():
    turn = DevRouter(settings=OVERPASS, fetch=lambda q, s: divided_road()).preview(ACROSS_THE_MEDIAN)["turns_back"]
    assert [turn["lon"], turn["lat"]] == pytest.approx([147.306, -42.80], abs=1e-5)
    assert turn["retraced_m"] == 0.0 and turn["message"].startswith("The route makes a U-turn ")


def _graphhopper(xy, details=None):
    lon, lat = TO_LONLAT.transform(*np.asarray(xy, float).T)
    path = {"distance": float(centreline.cumulative(np.asarray(xy, float))[-1]),
            "points": {"type": "LineString", "coordinates": np.column_stack([lon, lat]).tolist()}}
    if details is not None:
        path["details"] = details

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"paths": [path]})

    return GraphHopperRouter("http://router.example.org:8989/", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_graphhopper_preview_marks_a_u_turn_but_not_a_small_roundabout():
    e, n = TO_MGA.transform(147.301, -42.80)
    xy = np.array(MEDIAN_UTURN) + [e, n]
    names = [[0, 3, "Main Road"], [3, 4, "Cross Street"]]
    turn = _graphhopper(xy, {"street_name": [[0, 2, "Main Road"], [2, 3, ""], [3, 4, "Main Road"]]}).preview(HOBART)
    assert turn["turns_back"]["message"].startswith("The route makes a U-turn 200 m along")
    assert "turns_back" in _graphhopper(xy).preview(HOBART)  # without street names: by its shape alone
    # The same points on one road (a hairpin), or marked as on a roundabout: left to validation, which can build them
    # on request.
    assert "turns_back" not in _graphhopper(xy, {"street_name": [[0, 4, "Main Road"]]}).preview(HOBART)
    assert "turns_back" not in _graphhopper(xy, {"street_name": names,
                                                 "roundabout": [[0, 2, False], [2, 4, True]]}).preview(HOBART)


def test_graphhopper_is_asked_for_roundabouts_and_street_names():
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(json.loads(request.content)["details"])
        return httpx.Response(200, json={"paths": [{"distance": 1500.0, "points": {
            "type": "LineString", "coordinates": [[147.301, -42.8], [147.319, -42.8]]}}]})

    router = GraphHopperRouter("http://router.example.org:8989/", client=httpx.Client(transport=httpx.MockTransport(handler)))
    router.preview(HOBART)
    assert asked == [["roundabout", "street_name"]]
