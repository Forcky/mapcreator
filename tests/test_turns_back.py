"""A route that goes back the way it came is refused as the person's to fix, in the build and in the preview."""

import re
from dataclasses import replace

import httpx
import numpy as np
import pytest
from pyproj import Transformer

from mapcreator import config
from mapcreator.core import centreline
from mapcreator.job import Waypoint
from mapcreator.pipeline import route_stage
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
