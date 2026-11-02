import numpy as np
import pytest
from pyproj import Transformer

from mapcreator.providers.osm import graph

IDENT = Transformer.from_crs(4326, 4326, always_xy=True)


def _osm(nodes: dict[int, tuple[float, float]], ways: list[tuple[int, list[int], dict]]) -> dict:
    els = [{"type": "node", "id": i, "lon": x, "lat": y} for i, (x, y) in nodes.items()]
    els += [{"type": "way", "id": wid, "nodes": refs, "tags": tags} for wid, refs, tags in ways]
    return {"elements": els}


# A square block: a main road along y=0 and a residential detour around the top.
NODES = {1: (0, 0), 2: (100, 0), 3: (200, 0), 4: (0, 100), 5: (100, 100), 6: (200, 100)}


def test_routes_along_main_road_and_starts_exactly_at_the_waypoint():
    g = graph.build(_osm(NODES, [
        (10, [1, 2, 3], {"highway": "secondary"}),
        (11, [1, 4, 5, 6, 3], {"highway": "residential"}),
    ]), IDENT)
    r = graph.route(g, np.array([[30.0, 2.0], [170.0, -1.0]]))
    assert r.xy[0] == pytest.approx([30.0, 0.0])
    assert r.xy[-1] == pytest.approx([170.0, 0.0])
    assert {r.ways[i]["id"] for i in r.seg_way} == {10}
    assert r.retraced_m == 0


def test_one_way_is_respected():
    g = graph.build(_osm(NODES, [
        (10, [3, 2, 1], {"highway": "secondary", "oneway": "yes"}),  # only westbound
        (11, [1, 4, 5, 6, 3], {"highway": "residential"}),
    ]), IDENT)
    r = graph.route(g, np.array([[0.0, 0.0], [200.0, 0.0]]))
    assert {r.ways[i]["id"] for i in r.seg_way} == {11}


def test_two_waypoints_on_one_segment_connect_directly():
    g = graph.build(_osm(NODES, [(10, [1, 3], {"highway": "secondary"})]), IDENT)
    r = graph.route(g, np.array([[50.0, 0.0], [150.0, 0.0]]))
    assert np.hypot(*np.diff(r.xy, axis=0).T).sum() == pytest.approx(100.0)


def test_unreachable_and_far_waypoints_raise():
    g = graph.build(_osm(NODES, [(10, [1, 2], {"highway": "secondary"}), (11, [5, 6], {"highway": "secondary"})]),
                    IDENT)
    with pytest.raises(graph.RouteError, match="no drivable connection"):
        graph.route(g, np.array([[10.0, 0.0], [150.0, 100.0]]))
    with pytest.raises(graph.RouteError, match="from the nearest drivable road"):
        graph.route(g, np.array([[10.0, 0.0], [1000.0, 1000.0]]))


def test_route_too_long_is_a_user_facing_route_error():
    from mapcreator.pipeline.route_stage import RouteTooLong

    assert issubclass(RouteTooLong, graph.RouteError)
