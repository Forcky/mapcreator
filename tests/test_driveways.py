"""Driveways, service roads, tracks and private roads (the junctions research's Q5): which ways leaving the route
count, that they're never routed on or taken for junctions, and that guard rails, roadside furniture and trees keep
out of their mouths, with nothing changed where there's no mouth."""

import numpy as np
import pytest
import shapely

from mapcreator import config
from mapcreator.core import furniture, junctions, roadside, scenery
from mapcreator.core.scene import LocalFrame
from mapcreator.job import Job
from mapcreator.pipeline import route_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.osm import graph
from mapcreator.providers.osm import index as osm_index
from test_junctions import network, north, post_stations

FRAME = LocalFrame(0.0, 0.0, 0.0)
HALF = 3.2
DRIVEWAY = {"highway": "service", "service": "driveway"}


def find(g, start=10.0, end=990.0):
    """The driveways leaving a straight route along the network's main road, found on the network as given."""
    return junctions.find_driveways(g, graph.route(g, np.array([[start, 0.0], [end, 0.0]])).xy)


# -- which ways count ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tags,kind", [
    (DRIVEWAY, "driveway"),
    ({"highway": "service"}, "service"),
    ({"highway": "service", "service": "parking_aisle"}, "service"),
    ({"highway": "track", "tracktype": "grade3"}, "track"),
    ({"highway": "track", "access": "private"}, "track"),
    ({"highway": "residential", "access": "private"}, "private_road"),
    ({"highway": "unclassified", "access": "yes", "motor_vehicle": "no"}, "private_road"),
    ({"highway": "residential"}, None),  # a real side road: a junction (`junctions.find`)
    ({"highway": "residential", "access": "destination"}, None),  # limited, but still a junction
    ({"highway": "footway"}, None),
    ({"highway": "service", "area": "yes"}, None),  # a yard drawn as its outline
])
def test_driveway_kinds(tags, kind):
    assert junctions.driveway_kind(tags) == kind


@pytest.mark.parametrize("tags,width,source", [
    (DRIVEWAY, 3.5, "driveway"),  # the research's 1.75 m half width
    ({**DRIVEWAY, "width": "5"}, 5.0, "width"),
    ({"highway": "track"}, 4.0, "class"),  # unsealed unless tagged sealed
    ({"highway": "track", "surface": "asphalt"}, 6.0, "class"),
    ({"highway": "service", "surface": "gravel"}, 5.0, "class"),
    ({"highway": "residential", "access": "private"}, 5.8, "class"),
])
def test_an_untagged_driveway_is_3_5_m_and_the_rest_take_their_class_default(tags, width, source):
    assert junctions.driveway_width(tags) == (width, source)


def test_driveways_tracks_and_private_roads_are_found_and_real_side_roads_are_not():
    g = network([(300, north(300, 60.0), DRIVEWAY),
                 (500, north(500), {"highway": "residential", "name": "Real Road"}),
                 (600, [(600, -20.0), (600, -40.0)], {"highway": "track"}),
                 (800, north(800), {"highway": "residential", "access": "private"}),
                 (200, north(200), {"highway": "footway"}),
                 (700, north(700), {"highway": "service", "area": "yes"})])
    found = find(g)
    assert found["version"] == junctions.DRIVEWAY_FORMAT_VERSION and found["skipped"] == []
    got = [(d["at_m"], d["kind"], d["side"], d["width_m"], d["length_m"]) for d in found["driveways"]]
    assert got == [(290.0, "driveway", "left", 3.5, 30.0), (590.0, "track", "right", 4.0, 30.0),  # from 10 m
                   (790.0, "private_road", "left", 5.8, 30.0)]
    assert [d["id"] for d in found["driveways"]] == [0, 1, 2]
    # A real side road is a junction, and never a driveway; a driveway is never a junction.
    (j,) = junctions.find(g, graph.route(g, np.array([[10.0, 0.0], [990.0, 0.0]])).xy)["junctions"]
    assert j["name"] == "Real Road"


def test_a_driveway_on_a_bridge_is_skipped_and_a_lay_by_is_found_from_both_ends():
    g = network([(300, north(300, 60.0), {**DRIVEWAY, "bridge": "yes", "layer": "1"}),
                 (500, [(510.0, 4.0), 6], {"highway": "service"})])  # a 100 m lay-by, rejoining the road at 600 m
    found = find(g)
    assert [(s["at_m"], s["reason"]) for s in found["skipped"]] == [(290.0, "grade_separated")]
    assert [(d["at_m"], d["kind"], d["side"], d["length_m"]) for d in found["driveways"]] == [
        (490.0, "service", "left", 30.0), (590.0, "service", "left", 30.0)]
    # Its mouths reach along the road beside it, so nothing stands between the road and the lay-by there.
    s, xy, heading, hl, hr = straight()
    mk = junctions.masks(junctions.mouths(found["driveways"], s, xy, heading, hl, hr), s)
    assert mk["L"][500:531].all() and mk["L"][570:601].all() and not mk["R"].any()


def test_with_no_driveway_nothing_is_found():
    g = network([(500, north(500), {"highway": "residential"})])
    assert find(g)["driveways"] == []


# -- never routed on ---------------------------------------------------------------------------------------------------

HOBART = [{"lon": 147.3005, "lat": -42.8000}, {"lon": 147.3195, "lat": -42.8000}]


def bowed_road() -> dict:
    """A secondary road from 147.300 to 147.320 along -42.80, bowed 500 m north in the middle, with a driveway at its
    fourth node and a track straight across the bow from its second node to its tenth: a shortcut, if it could be
    driven."""
    lat = [-42.80 + 0.0045 * np.sin(np.pi * i / 10) for i in range(11)]
    nodes = [{"type": "node", "id": i + 1, "lon": 147.300 + i * 0.002, "lat": lat[i]} for i in range(11)]
    road = {"type": "way", "id": 1, "nodes": list(range(1, 12)), "tags": {"highway": "secondary"}}
    return {"elements": [*nodes, road]}


def service_ways() -> dict:
    road = {e["id"]: e for e in bowed_road()["elements"] if e["type"] == "node"}
    nodes = [{"type": "node", "id": 100, "lon": road[4]["lon"], "lat": road[4]["lat"] + 0.0005},
             {"type": "node", "id": 200, "lon": 147.310, "lat": road[2]["lat"]}]
    ways = [{"type": "way", "id": 10, "nodes": [4, 100], "tags": DRIVEWAY},
            {"type": "way", "id": 11, "nodes": [2, 200, 10], "tags": {"highway": "track"}}]
    return {"elements": [road[2], road[4], road[10], *nodes, *ways]}


def test_the_route_stage_never_routes_on_a_track_and_writes_the_driveways_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(route_stage.osm_roads, "for_bbox", lambda bbox, settings: bowed_road())
    monkeypatch.setattr(route_stage.osm_roads, "railways_and_paths_for_bbox", lambda bbox, settings: {"elements": []})
    monkeypatch.setattr(route_stage.osm_roads, "service_ways_for_bbox", lambda bbox, settings: service_ways())
    settings, work = config.load(), WorkDir(tmp_path / "work")
    job = Job(track_id="bowed", waypoints=HOBART)
    result = route_stage.run(job, settings, work)
    assert result["ways"] == 1 and result["length_m"] > 1850  # round the bow, not across it (1.55 km)
    assert result["junctions"] == 0 and result["driveways"] == {"driveway": 1, "service": 0, "track": 2,
                                                                 "private_road": 0}
    doc = work.read_json(route_stage.STAGE, route_stage.DRIVEWAYS)
    assert doc["crs"] == "EPSG:28355" and "unavailable" not in doc
    assert [(d["kind"], d["way_id"], d["node"]) for d in doc["driveways"]] == [("track", 11, 2), ("driveway", 10, 4),
                                                                               ("track", 11, 10)]
    assert route_stage.driveways(work, job) == doc["driveways"]
    assert work.read_json(route_stage.STAGE, route_stage.JUNCTIONS)["junctions"] == []
    # Built without junctions: no driveways either, and the file goes.
    without = Job(track_id="bowed", waypoints=HOBART, junctions="none")
    assert route_stage.driveways(work, without) == []
    route_stage.run(without, settings, work)
    assert not (work.root / route_stage.STAGE / route_stage.DRIVEWAYS).exists()


def test_with_an_index_from_before_service_ways_the_route_stage_says_so_and_goes_on(tmp_path, monkeypatch):
    def old_index(bbox, settings):
        raise osm_index.IndexUnavailable("x.sqlite was built before driveways, service roads and tracks were indexed")

    monkeypatch.setattr(route_stage.osm_roads, "for_bbox", lambda bbox, settings: bowed_road())
    monkeypatch.setattr(route_stage.osm_roads, "railways_and_paths_for_bbox", lambda bbox, settings: {"elements": []})
    monkeypatch.setattr(route_stage.osm_roads, "service_ways_for_bbox", old_index)
    work = WorkDir(tmp_path / "work")
    job = Job(track_id="bowed", waypoints=HOBART)
    result = route_stage.run(job, config.load(), work)
    assert "rebuild it" in result["driveways"]["unavailable"]
    assert route_stage.driveways(work, job) == []
    from mapcreator.pipeline import validate_stage
    (note,) = validate_stage.driveway_notes(work)
    assert note.startswith("Driveways, service roads and tracks weren't looked for") and "rebuild it" in note


# -- keeping their mouths clear ----------------------------------------------------------------------------------------

def straight(n=1001):
    s = np.arange(n, dtype=float)
    return s, np.column_stack([s, np.zeros(n)]), np.zeros(n), np.full(n, HALF), np.full(n, HALF)


def driveway_at(x: float, side: str = "left") -> dict:
    """A driveway as find_driveways records it, square off a straight road east along y = 0."""
    g = network([(int(x), north(x, 60.0) if side == "left" else [(x, -20.0), (x, -40.0), (x, -60.0)], DRIVEWAY)])
    (d,) = find(g)["driveways"]
    return d


def test_a_driveway_s_mouth_is_its_3_5_m_plus_the_margin_and_cuts_a_rail_across_it():
    s, xy, heading, hl, hr = straight()
    (m,) = junctions.mouths([driveway_at(400.0)], s, xy, heading, hl, hr)
    assert m.side == "L" and m.half_m == 1.75
    # Its carriageway covers 398.25-401.75 m; stations are a metre apart.
    assert (m.s0, m.s1) == (399.0 - junctions.MOUTH_MARGIN_M, 401.0 + junctions.MOUTH_MARGIN_M)
    mk = junctions.masks([m], s)
    rail = {"L": (s >= 300) & (s <= 500), "R": (s >= 300) & (s <= 500)}
    kept, removed = roadside.clear_mouths(rail, mk, s)
    assert kept["L"][300:394].all() and not kept["L"][394:407].any() and kept["L"][407:501].all()
    assert removed == {"L": 13.0, "R": 0.0}  # a real gap where the driveway is, on its side only
    assert np.array_equal(kept["R"], rail["R"])


def furn(mouth_l=None, mouth_r=None):
    n = 1001
    s = np.arange(n, dtype=float)
    flat = np.zeros(n)
    meshes, info, _ = furniture.furniture(s, np.column_stack([s, flat]), np.full(n, 100.0), flat, flat,
                                          np.full(n, HALF), np.full(n, HALF), flat, flat, flat, np.full(n, 80.0),
                                          FRAME, mouth_l=mouth_l, mouth_r=mouth_r)
    return meshes, info


def test_a_guide_post_due_in_a_driveway_s_mouth_goes_up_past_it():
    s, xy, heading, hl, hr = straight()
    plain, _ = furn()
    assert 305 in post_stations(plain, "R")
    mk = junctions.masks(junctions.mouths([driveway_at(300.0, "right")], s, xy, heading, hl, hr), s)
    meshes, info = furn(mk["L"], mk["R"])
    assert 305 not in post_stations(meshes, "R") and 307 in post_stations(meshes, "R")
    assert post_stations(meshes, "L") == post_stations(plain, "L")
    assert info["side_road_mouths"]["posts_moved"] == 1


def test_trees_stand_off_a_driveway_s_first_30_m_and_nowhere_else():
    s, xy, heading, hl, hr = straight()
    ms = junctions.mouths([driveway_at(400.0)], s, xy, heading, hl, hr)
    lanes = junctions.zone(ms, s, xy, scenery.CLEAR_OF_VERGE_M)
    trees = np.array([[400.0, 20.0],   # on the driveway
                      [403.5, 25.0],   # 1.75 m off its edge: within the 2 m clearance
                      [406.0, 25.0],   # 4.25 m off it
                      [400.0, 35.0],   # past its first 30 m
                      [400.0, -20.0],  # across the road
                      [200.0, 20.0]])  # nowhere near
    assert shapely.contains_xy(lanes, trees[:, 0], trees[:, 1]).tolist() == [True, True, False, False, False, False]


def test_with_no_driveway_there_is_no_mouth_and_nothing_changes():
    s, xy, heading, hl, hr = straight()
    assert junctions.mouths([], s, xy, heading, hl, hr) == []
    assert junctions.zone([], s, xy, scenery.CLEAR_OF_VERGE_M) is None
    meshes, info = furn()
    empty = np.zeros(len(s), bool)
    meshes2, info2 = furn(empty, empty)
    assert all(np.array_equal(a.positions, b.positions) for a, b in zip(meshes, meshes2))
    assert info == {k: v for k, v in info2.items() if k != "side_road_mouths"}


def test_a_driveway_s_mouth_still_keeps_posts_out_on_a_road_with_a_bridge():
    # Brinktop Road and the Lyell Highway have bridges: the parapets' mask must not replace the driveways' mouths.
    from mapcreator.pipeline import mesh_stage

    s = np.arange(100.0)
    side_roads = {"L": s < 10, "R": np.zeros(100, bool)}
    drive = {"L": np.zeros(100, bool), "R": (s >= 40) & (s <= 50)}
    parapet = {"L": (s >= 70) & (s <= 80), "R": (s >= 70) & (s <= 80)}
    got = mesh_stage.furniture_mouths(side_roads, drive, parapet)
    assert np.array_equal(got["L"], (s < 10) | parapet["L"])
    assert np.array_equal(got["R"], drive["R"] | parapet["R"])
    assert mesh_stage.furniture_mouths(None, None, None) is None
    assert np.array_equal(mesh_stage.furniture_mouths(None, drive, None)["R"], drive["R"])

