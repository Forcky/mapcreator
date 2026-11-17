"""The preview says when a route goes through a tunnel: tunnels aren't covered yet, but built as an open cutting."""

from dataclasses import replace

import httpx
import numpy as np
import pytest
from pyproj import Transformer

from mapcreator import config
from mapcreator.core.centreline import cumulative
from mapcreator.job import Waypoint
from mapcreator.pipeline import route_stage
from mapcreator.providers.osm import graph
from mapcreator.service import routing
from mapcreator.service.routing import DevRouter, GraphHopperRouter

HOBART = [Waypoint(lon=147.301, lat=-42.8001), Waypoint(lon=147.319, lat=-42.7999)]
OVERPASS = replace(config.load(), osm_index="")
TO_LONLAT = Transformer.from_crs(28355, 4326, always_xy=True)


def road_with_tunnel() -> dict:
    """A secondary road east along latitude -42.80, its middle 0.004 degrees (about 330 m) in a tunnel drawn as two
    ways."""
    nodes = [{"type": "node", "id": i + 1, "lon": 147.300 + i * 0.002, "lat": -42.80} for i in range(11)]
    tags = {"highway": "secondary", "name": "Main Road"}
    return {"elements": [*nodes,
                         {"type": "way", "id": 1, "nodes": [1, 2, 3, 4], "tags": tags},
                         {"type": "way", "id": 2, "nodes": [4, 5], "tags": {**tags, "tunnel": "yes", "layer": "-1"}},
                         {"type": "way", "id": 3, "nodes": [5, 6], "tags": {**tags, "tunnel": "yes", "layer": "-1"}},
                         {"type": "way", "id": 4, "nodes": [6, 7, 8, 9, 10, 11], "tags": tags}]}


def test_tunnel_spans_are_the_stretches_on_tunnel_ways():
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0]])
    r = graph.Route(xy=xy, seg_way=np.array([0, 1, 0]), retraced_m=0.0,
                    ways=[{"id": 1, "tags": {"highway": "primary"}},
                          {"id": 2, "tags": {"highway": "primary", "tunnel": "yes"}}])
    assert route_stage.tunnel_spans(r) == [(100.0, 200.0)]
    r.ways[1]["tags"]["tunnel"] = "no"
    assert route_stage.tunnel_spans(r) == []


def test_the_message_says_where_and_how_long():
    assert routing.tunnel_message(2340.0, 180.0) == (
        "The route goes through a tunnel 2.3 km along (180 m long). Tunnels aren't covered yet: it's built as an "
        "open cutting with grassy banks and no roof.")


def test_dev_router_notes_a_tunnel_drawn_as_two_ways_as_one():
    preview = DevRouter(settings=OVERPASS, fetch=lambda q, s: road_with_tunnel()).preview(HOBART)
    (note,) = preview["tunnels"]
    assert note["length_m"] == pytest.approx(0.004 * 111_320 * np.cos(np.radians(42.8)), rel=0.02)
    assert [note["lon"], note["lat"]] == pytest.approx([147.308, -42.80], abs=1e-4)
    assert "turns_back" not in preview
    plain = {"elements": [e if e["type"] == "node" else {**e, "tags": {"highway": "secondary"}}
                          for e in road_with_tunnel()["elements"]]}
    assert "tunnels" not in DevRouter(settings=OVERPASS, fetch=lambda q, s: plain).preview(HOBART)


def test_graphhopper_notes_a_tunnel_from_its_road_environment_detail():
    xy = np.column_stack([np.linspace(0.0, 1500.0, 16), np.zeros(16)]) + [520000.0, 5260000.0]
    lon, lat = TO_LONLAT.transform(*xy.T)
    path = {"distance": 1500.0, "points": {"type": "LineString", "coordinates": np.column_stack([lon, lat]).tolist()},
            "details": {"road_environment": [[0, 5, "road"], [5, 7, "tunnel"], [7, 9, "bridge"], [9, 15, "road"]]}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"paths": [path]})

    router = GraphHopperRouter("http://router.example.org:8989/", client=httpx.Client(transport=httpx.MockTransport(handler)))
    (note,) = router.preview(HOBART)["tunnels"]
    s = cumulative(xy)
    assert note["along_m"] == pytest.approx(s[5], abs=0.2) and note["length_m"] == pytest.approx(s[7] - s[5], abs=0.2)
    assert note["message"].startswith("The route goes through a tunnel 500 m along (200 m long).")
