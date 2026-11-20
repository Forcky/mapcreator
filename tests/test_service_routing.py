import json
import math
from dataclasses import replace

import httpx
import pytest

from mapcreator import config
from mapcreator.job import Waypoint
from mapcreator.service.routing import (DevRouter, GraphHopperRouter, NoRoute, OutOfArea, RouterUnavailable,
                                        check_region)

HOBART = [Waypoint(lon=147.301, lat=-42.8001), Waypoint(lon=147.319, lat=-42.7999)]
MELBOURNE = Waypoint(lon=144.96, lat=-37.81)
AUCKLAND = Waypoint(lon=174.76, lat=-36.85)
# Roads from (a fake) Overpass, whatever the local configuration says.
OVERPASS = replace(config.load(), osm_index="")


def straight_road() -> dict:
    """A secondary road running east along latitude -42.80, nodes every 0.002 degrees."""
    nodes = [{"type": "node", "id": i + 1, "lon": 147.300 + i * 0.002, "lat": -42.80} for i in range(11)]
    way = {"type": "way", "id": 100, "nodes": [n["id"] for n in nodes], "tags": {"highway": "secondary"}}
    return {"elements": [*nodes, way]}


def test_check_region_names_the_supported_regions():
    assert check_region(HOBART).key == "tas"
    assert check_region([MELBOURNE]).key == "au55"  # the national fallback
    with pytest.raises(OutOfArea, match="Point 2 .*Tasmania, Australia"):
        check_region([HOBART[0], AUCKLAND])
    with pytest.raises(OutOfArea, match="same region"):
        check_region([HOBART[0], MELBOURNE])


def test_dev_router_routes_on_the_osm_graph():
    queries = []

    def fake_fetch(query, settings):
        queries.append(query)
        return straight_road()

    preview = DevRouter(settings=OVERPASS, fetch=fake_fetch).preview(HOBART)
    assert len(queries) == 1 and "highway" in queries[0]
    start, end = preview["coordinates"][0], preview["coordinates"][-1]
    assert start == pytest.approx([147.301, -42.80], abs=1e-5)
    assert end == pytest.approx([147.319, -42.80], abs=1e-5)
    expected = 0.018 * 111_320 * math.cos(math.radians(42.8))
    assert preview["length_m"] == pytest.approx(expected, rel=0.01)


def test_dev_router_errors_are_plain_language():
    far = [HOBART[0], Waypoint(lon=147.31, lat=-42.79)]  # about 1.1 km north of the road
    with pytest.raises(NoRoute, match="^Waypoint 2 is .* m from the nearest drivable road.$"):
        DevRouter(settings=OVERPASS, fetch=lambda q, s: straight_road()).preview(far)

    def offline(query, settings):
        raise httpx.ConnectError("offline")

    with pytest.raises(RouterUnavailable):
        DevRouter(settings=OVERPASS, fetch=offline).preview(HOBART)
    with pytest.raises(NoRoute, match="too far apart"):
        DevRouter(settings=OVERPASS, fetch=offline).preview([HOBART[0], Waypoint(lon=148.3, lat=-41.5)])


def gh(handler) -> GraphHopperRouter:
    return GraphHopperRouter("http://router.example.org:8989/", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_graphhopper_request_and_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"paths": [{
            "distance": 1234.56,
            "points": {"type": "LineString", "coordinates": [[147.301, -42.8, 12.0], [147.31, -42.8, 15.0],
                                                             [147.319, -42.8, 11.0]]},
        }]})

    preview = gh(handler).preview(HOBART)
    assert seen["url"] == "http://router.example.org:8989/route"
    assert seen["body"]["points"] == [[147.301, -42.8001], [147.319, -42.7999]]
    assert seen["body"]["profile"] == "car" and seen["body"]["points_encoded"] is False
    assert preview == {"coordinates": [[147.301, -42.8], [147.31, -42.8], [147.319, -42.8]], "length_m": 1234.6}


@pytest.mark.parametrize("details, index, kind, text", [
    ("com.graphhopper.util.exceptions.PointOutOfBoundsException", 1, OutOfArea, "Point 2 is outside"),
    ("com.graphhopper.util.exceptions.PointNotFoundException", 0, NoRoute, "Point 1 isn't near a road"),
    ("com.graphhopper.util.exceptions.ConnectionNotFoundException", None, NoRoute, "No drivable route"),
    ("java.lang.IllegalArgumentException", None, NoRoute, "couldn't find a route"),
])
def test_graphhopper_errors_map_to_clean_exceptions(details, index, kind, text):
    hint = {"message": "internal detail", "details": details}
    if index is not None:
        hint["point_index"] = index

    def handler(request):
        return httpx.Response(400, json={"message": "internal detail", "hints": [hint]})

    with pytest.raises(kind, match=text) as err:
        gh(handler).preview(HOBART)
    assert "internal detail" not in str(err.value)


def test_graphhopper_outages_are_unavailable():
    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(RouterUnavailable):
        gh(down).preview(HOBART)
    with pytest.raises(RouterUnavailable):
        gh(lambda r: httpx.Response(502, text="bad gateway")).preview(HOBART)
    with pytest.raises(RouterUnavailable):
        gh(lambda r: httpx.Response(200, json={"paths": []})).preview(HOBART)


def test_out_of_area_is_rejected_before_any_request():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    with pytest.raises(OutOfArea):
        gh(handler).preview([HOBART[0], MELBOURNE])
    with pytest.raises(OutOfArea):
        DevRouter(settings=OVERPASS, fetch=lambda q, s: calls.append(q)).preview([MELBOURNE, HOBART[0]])
    assert calls == []


# -- the service's log of previews that fail ---------------------------------------------------------------------

def _router(handler) -> GraphHopperRouter:
    return GraphHopperRouter("http://router.example.org:8989/", client=httpx.Client(transport=httpx.MockTransport(handler)))


def _preview_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == "mapcreator.service.routing" and r.levelname == "WARNING"]


def _raise(error):
    def handler(request: httpx.Request) -> httpx.Response:
        raise error(request)
    return handler


@pytest.mark.parametrize("handler, kind, reason", [
    (lambda req: httpx.Response(503, text="busy"), RouterUnavailable, "router answered HTTP 503"),
    (_raise(lambda req: httpx.ReadTimeout("slow", request=req)), RouterUnavailable, "router: timed out"),
    (_raise(lambda req: httpx.ConnectError("refused", request=req)), RouterUnavailable,
     "router: no answer (ConnectError)"),
    (lambda req: httpx.Response(400, json={"message": "Cannot find point 1: -42.8001,147.301", "hints": [
        {"details": "com.graphhopper.util.exceptions.PointNotFoundException", "point_index": 0}]}),
     NoRoute, "router answered HTTP 400, PointNotFoundException"),
    (lambda req: httpx.Response(200, text="not json"), RouterUnavailable, "router's answer unreadable"),
], ids=["5xx", "timeout", "no connection", "router error", "unreadable"])
def test_a_failed_preview_is_logged_without_the_points(caplog, handler, kind, reason):
    with caplog.at_level("WARNING", logger="mapcreator.service.routing"), pytest.raises(kind):
        _router(handler).preview(HOBART)
    [line] = _preview_warnings(caplog)
    assert line.startswith(f"route preview failed: {kind.__name__} ({reason}")
    assert "; 2 points, 1.5 km point to point" in line
    # Nothing that could place the person: no coordinates (the router's own message carries them), no URL.
    assert not any(c in line for c in ("147.", "42.", "router.example.org"))


def test_a_preview_that_works_isnt_logged(caplog):
    coords = [[147.301, -42.8], [147.319, -42.8]]
    with caplog.at_level("DEBUG", logger="mapcreator.service.routing"):
        _router(lambda req: httpx.Response(200, json={"paths": [{"distance": 1470.0, "points": {
            "type": "LineString", "coordinates": coords}}]})).preview(HOBART)
    assert [r for r in caplog.records if r.name == "mapcreator.service.routing"] == []


def test_dev_router_failures_are_logged_too(caplog):
    def offline(query, settings):
        raise httpx.ConnectError("offline")

    with caplog.at_level("WARNING", logger="mapcreator.service.routing"), pytest.raises(RouterUnavailable):
        DevRouter(settings=OVERPASS, fetch=offline).preview(HOBART)
    [line] = _preview_warnings(caplog)
    assert line.startswith("route preview failed: RouterUnavailable (map data: no answer (ConnectError))")


def test_a_point_out_of_area_is_not_a_router_failure(caplog):
    with caplog.at_level("WARNING", logger="mapcreator.service.routing"), pytest.raises(OutOfArea):
        _router(lambda req: httpx.Response(200)).preview([HOBART[0], AUCKLAND])
    assert _preview_warnings(caplog) == []
