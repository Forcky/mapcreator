"""Railways and paths crossing the route (stage 2c): the bridge pieces (`core.decks`: every face outward, solids
closed, colliders invisible), what the route stage records (`crossing_lines.json`: the crossing, the whole bridge
and its approaches, the roads it passes over; nothing without the data), what the mesh stage draws over the route
(the clearance rule, the signed-clearance deck, supports off the roads) and under the route's bridges, and the
validator's checks and warnings."""

import math
from dataclasses import replace

import numpy as np
import pytest
import shapely

from mapcreator import config
from mapcreator.core import decks, rail_and_paths, textures, validate
from mapcreator.core.scene import LocalFrame
from mapcreator.job import Job
from mapcreator.pipeline import mesh_stage, route_stage, validate_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.osm import index as osm_index
from mapcreator.providers.osm import overpass, roads

FRAME = LocalFrame(0.0, 0.0, 0.0)
HALF = 3.2
ROAD_Z = 100.0


# -- the pieces --------------------------------------------------------------------------------------------------

def _closed_and_outward(piece: decks.Piece) -> None:
    """Every triangle faces away from the solid's middle (its winding and its vertex normals agree), and every edge
    is shared by exactly two triangles, matched by position: a closed solid, faces with their own vertices."""
    p, t = piece.pos, piece.tri
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1)
    keep = area > 1e-9
    assert (np.einsum("ij,ij->i", cross[keep], piece.nrm[t[keep, 0]]) > 0).all()
    assert np.allclose(np.linalg.norm(piece.nrm, axis=1), 1.0)
    key = np.round(p, 6)
    edges = {}
    for tri in t[keep]:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            e = tuple(sorted((tuple(key[tri[i]]), tuple(key[tri[j]]))))
            edges[e] = edges.get(e, 0) + 1
    assert set(edges.values()) == {2}


def test_a_swept_deck_and_a_box_are_closed_solids_facing_outwards():
    line = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 5.0], [30.0, 5.0]])
    deck = decks.deck(line, np.linspace(10.0, 12.0, 4), 2.3, 0.9)
    _closed_and_outward(deck)
    _closed_and_outward(decks.box(np.array([5.0, 5.0]), np.array([1.0, 1.0]), 1.0, 3.0, 0.0, 4.0))
    _closed_and_outward(decks.ballast(line, np.full(4, 10.0)))
    _closed_and_outward(decks.prism(np.array([3.0, 4.0]), decks.COLUMN_R, 0.0, 6.0))
    wall = decks.collider_prism(np.array([3.0, 4.0]), decks.COLUMN_R, 0.0, 3.0)
    assert len(wall.tri) == 2 * 2 * decks.COLUMN_SIDES and (np.abs(wall.nrm[wall.tri[:, 0], 2]) < 1e-9).all()
    # Mitred at the bends, so the deck keeps its width round them.
    assert np.ptp(decks.offset(line, 1.15)[:, 1] - decks.offset(line, -1.15)[:, 1]) < 2.4


def test_packed_meshes_keep_their_winding_and_normals_in_ac_axes():
    line = np.array([[0.0, 0.0], [10.0, 0.0]])
    (m,) = decks.meshes("FOOTBRIDGE_9_DECK", "bridge_concrete", [decks.deck(line, np.full(2, 5.0), 2.0, 1.0)],
                        LocalFrame(-50.0, 20.0, 1.0))
    p = m.positions.astype(float)[m.triangles]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert (np.einsum("ij,ij->i", cross, m.normals[m.triangles[:, 0]]) > 0).all()
    assert m.renderable and m.positions[:, 1].max() == pytest.approx(4.0)  # 5 m less the frame's 1 m
    (wall,) = decks.meshes("1WALL_PIER_FB9", "rail", [decks.collider_box(np.zeros(2), np.array([1.0, 0.0]), 1.0,
                                                                         1.0, 0.0, 3.0)], FRAME, renderable=False)
    assert not wall.renderable and len(wall.triangles) == 16  # four faces, both windings


def test_the_generic_parapet_and_the_footbridge_railing_stand_at_their_heights():
    line = np.array([[0.0, 0.0], [20.0, 0.0]])
    concrete, steel = decks.barrier(line, np.zeros(2), 5.2)
    assert max(p.pos[:, 2].max() for p in concrete) == pytest.approx(0.8)
    assert max(p.pos[:, 2].max() for p in steel) == pytest.approx(1.1)  # one steel rail on top
    assert min(p.pos[:, 1].min() for p in concrete) == pytest.approx(-2.6)  # flush with the deck's edges
    railing = decks.railing(line, np.zeros(2), 2.3)
    assert max(p.pos[:, 2].max() for p in railing) == pytest.approx(decks.RAILING_H + decks.RAILING_BAR_W / 2)
    rails = decks.running_rails(line, np.zeros(2), 1.067)
    inner = sorted(abs(p.pos[:, 1]).min() for p in rails)
    assert inner == pytest.approx([1.067 / 2, 1.067 / 2])


def test_the_textures_are_our_own():
    for material in (rail_and_paths.CONCRETE, rail_and_paths.BALLAST, "path_asphalt", "path_ground"):
        im = textures.LIBRARY[rail_and_paths.TEXTURES[material]]()
        assert im.mode == "RGB"
    assert textures.LIBRARY["ballast"]().size == (256, 256)


# -- what the route stage records ----------------------------------------------------------------------------

HOBART = [{"lon": 147.301, "lat": -42.8001}, {"lon": 147.319, "lat": -42.7999}]


def hobart_roads() -> dict:
    """A secondary road east along -42.80 (way 1, nodes 1-11, 0.002 degrees apart) and a residential street
    crossing it at grade at 147.314 (way 2), its south end a node of the road."""
    nodes = [{"type": "node", "id": i + 1, "lon": 147.300 + i * 0.002, "lat": -42.80} for i in range(11)]
    street = [{"type": "node", "id": 50, "lon": 147.3141, "lat": -42.7985}]
    ways = [{"type": "way", "id": 1, "nodes": [n["id"] for n in nodes],
             "tags": {"highway": "secondary", "maxheight": "5.3"}},
            {"type": "way", "id": 2, "nodes": [50, 8], "tags": {"highway": "residential"}}]
    return {"elements": [*nodes, *street, *ways]}


def hobart_lines() -> dict:
    """A footbridge over the road at 147.306, mapped as two bridge ways split just north of it (200, 201), with a path
    on from each end (202 south, 203 north: the approaches) and a sharp turn off its north end (204, not part of the
    bridge); a railway crossing the road at grade at 147.310, sharing node 6 (not listed); a footway elsewhere."""
    lon = 147.306
    pts = {300: (lon, -42.8006), 301: (lon, -42.7998), 302: (lon, -42.7994), 303: (lon, -42.8012),
           304: (lon, -42.7988), 305: (lon + 0.0008, -42.7994), 310: (147.310, -42.7990), 311: (147.310, -42.8010),
           320: (147.316, -42.7990), 321: (147.317, -42.7990)}
    nodes = [{"type": "node", "id": i, "lon": x, "lat": y} for i, (x, y) in pts.items()]
    bridge = {"highway": "footway", "bridge": "yes", "layer": "1"}
    ways = [{"type": "way", "id": 200, "nodes": [300, 301], "tags": bridge},
            {"type": "way", "id": 201, "nodes": [301, 302], "tags": bridge},
            {"type": "way", "id": 202, "nodes": [303, 300], "tags": {"highway": "footway"}},
            {"type": "way", "id": 203, "nodes": [302, 304], "tags": {"highway": "footway"}},
            {"type": "way", "id": 204, "nodes": [302, 305], "tags": {"highway": "footway", "bridge": "yes"}},
            {"type": "way", "id": 205, "nodes": [310, 6, 311], "tags": {"railway": "rail", "gauge": "1067"}},
            {"type": "way", "id": 206, "nodes": [320, 321], "tags": {"highway": "footway"}}]
    return {"elements": [*nodes, *ways]}


def _route(tmp_path, monkeypatch, lines=hobart_lines):
    monkeypatch.setattr(route_stage.osm_roads, "for_bbox", lambda bbox, settings: hobart_roads())
    seen = []

    def fetch(bbox, settings):
        seen.append(bbox)
        return lines()

    monkeypatch.setattr(route_stage.osm_roads, "railways_and_paths_for_bbox", fetch)
    work = WorkDir(tmp_path / "work")
    settings = replace(config.load(), osm_index="")  # as from Overpass, whatever the local config
    out = route_stage.run(Job(track_id="hobart", waypoints=HOBART), settings, work)
    return work, out, seen


def test_the_route_stage_lists_a_footbridge_with_its_whole_bridge_approaches_and_roads(tmp_path, monkeypatch):
    work, out, seen = _route(tmp_path, monkeypatch)
    assert out["crossing_lines"] == {"over": 1, "under": 0, "unresolved": 0}  # not the level crossing
    w, s, e, n = seen[0]
    assert w < 147.301 and e > 147.319 and s < -42.8001 and n > -42.7999  # the route's box and a little more
    doc = work.read_json(route_stage.STAGE, route_stage.CROSSING_LINES)
    assert doc["version"] == rail_and_paths.FORMAT_VERSION and doc["crs"] == "EPSG:28355"
    (c,) = route_stage.crossing_lines(work)
    assert (c["relation"], c["kind"], c["railway"], c["bridge"]) == ("over", "path", None, True)
    assert c["route"]["maxheight"] == "5.3"
    st = c["structure"]
    deck = np.array(st["deck"])
    # Both halves of the bridge, about 133 m (0.0012 degrees of latitude), not the sharp turn off its end.
    assert st["ways"] == 2 and float(np.hypot(*np.diff(deck, axis=0).T).sum()) == pytest.approx(133.4, abs=1.0)
    assert st["deck_at_m"] == pytest.approx(66.7, abs=1.0)
    for line in st["approaches"]:
        assert float(np.hypot(*np.diff(np.array(line), axis=0).T).sum()) == pytest.approx(40.0, abs=0.01)
    # The approaches start at the deck's ends, and run on away from it.
    assert np.allclose(st["approaches"][0][0], deck[0]) and np.allclose(st["approaches"][1][0], deck[-1])
    (road,) = c["roads_crossed"]
    assert road["route"] and road["way_id"] == 1 and road["line_at_m"] == pytest.approx(66.7, abs=1.0)
    assert road["angle_deg"] == pytest.approx(90.0, abs=0.5)
    assert road["tags"] == {"highway": "secondary", "maxheight": "5.3"}
    # From Overpass, ATTRIBUTION.txt names the railways' and paths' filters too (ODbL 4.6).
    sel = route_stage.osm_selection(work, config.load())
    assert sel == {"overpass_filter": overpass.ROADS_FILTER, "overpass_lines_filter": roads.LINES_SELECTION}
    note = roads.selection_note(sel, build="test")
    assert overpass.RAILWAYS_FILTER in note and overpass.PATHS_FILTER in note and "index.py" in note


def test_without_railways_and_paths_the_route_stage_says_why_and_goes_on(tmp_path, monkeypatch):
    def old_index():
        raise osm_index.IndexUnavailable("x.sqlite was built before railways and paths were indexed; rebuild it")

    work, out, _ = _route(tmp_path, monkeypatch, lines=old_index)
    assert "rebuild it" in out["crossing_lines"]["unavailable"]
    assert route_stage.crossing_lines(work) == []
    notes = validate_stage.rail_and_paths_notes(work)
    assert notes == ["Footbridges and rail bridges over the route weren't looked for: the OSM index was built before "
                     "railways and paths were kept: rebuild it."]


def test_structure_walks_bridge_ways_only_and_stops_at_a_sharp_turn():
    ways = {1: rail_and_paths.LineWay([1, 2], np.array([[0.0, 0.0], [0.0, 30.0]]), {"railway": "rail",
                                                                                   "bridge": "yes"}),
            2: rail_and_paths.LineWay([2, 3], np.array([[0.0, 30.0], [5.0, 60.0]]), {"railway": "rail",
                                                                                    "bridge": "viaduct"}),
            3: rail_and_paths.LineWay([3, 4], np.array([[5.0, 60.0], [5.0, 200.0]]), {"railway": "rail"}),
            4: rail_and_paths.LineWay([1, 5], np.array([[0.0, 0.0], [-50.0, 0.0]]), {"railway": "rail",
                                                                                    "bridge": "yes"}),
            5: rail_and_paths.LineWay([6, 1], np.array([[0.0, -100.0], [0.0, 0.0]]), {"highway": "footway"})}
    st = rail_and_paths.structure(ways, 1, [0.0, 10.0])
    assert st["ways"] == 2 and st["deck_at_m"] == pytest.approx(10.0)
    assert st["approaches"][0] == []  # a sharp turn (way 4) and a path (way 5) don't carry the railway on
    assert np.allclose(st["approaches"][1][-1], [5.0, 60.0 + (40.0)])
    assert rail_and_paths.structure(ways, 3, [5.0, 100.0]) == {"deck": None}  # not a bridge


def test_gauge_width_and_surface_from_the_tags():
    assert rail_and_paths.gauge_m({"gauge": "1067"}) == pytest.approx(1.067)
    assert rail_and_paths.gauge_m({"gauge": "1600;1435"}) == pytest.approx(1.6)
    assert rail_and_paths.gauge_m({}) == pytest.approx(1.435)
    assert rail_and_paths.path_width_m({"highway": "cycleway"}) == 3.0
    assert rail_and_paths.path_width_m({"highway": "footway", "width": "2.5 m"}) == 2.5
    assert rail_and_paths.path_width_m({"highway": "footway", "width": "40"}) == 2.0  # implausible: the class's
    assert rail_and_paths.path_material({"highway": "footway"}) == rail_and_paths.CONCRETE
    assert rail_and_paths.path_material({"highway": "footway", "surface": "gravel"}) == "path_ground"
    assert rail_and_paths.path_material({"highway": "cycleway"}) == "path_asphalt"


# -- what the mesh stage draws -----------------------------------------------------------------------------------

def route_arrays(n=401):
    """The route east along y = 0 from 0 to 400 m, level at ROAD_Z, 6.4 m wide."""
    s = np.arange(n, dtype=float)
    c = {"s": s, "xy": np.column_stack([s, np.zeros(n)]), "z": np.full(n, ROAD_Z), "heading": np.zeros(n)}
    sec = {"half_left": np.full(n, HALF), "half_right": np.full(n, HALF)}
    return c, sec


def embankment(rise: float):
    """Ground beside the road that rises `rise` m on approach embankments from 10 m to 20 m off the centreline."""
    def h(e, n):
        n = np.asarray(n, float)
        return ROAD_Z + rise * np.clip((np.abs(n) - 10.0) / 10.0, 0.0, 1.0) + 0.0 * np.asarray(e, float)
    return h


def built_road(e, n):
    """The built road's height: on the carriageway only."""
    n = np.asarray(n, float)
    return np.where(np.abs(n) <= HALF, ROAD_Z, np.nan) + 0.0 * np.asarray(e, float)


def built_ground(dem):
    def h(e, n):
        n = np.asarray(n, float)
        return np.where(np.abs(n) > HALF, dem(e, n), np.nan)
    return h


def over_crossing(kind="path", tags=None, maxheight=None, roads=None, x=200.0, half_len=30.0, cid=0):
    """A bridge crossing the route square at x, north to south, `half_len` either side."""
    tags = tags or ({"highway": "footway", "bridge": "yes", "layer": "1"} if kind == "path"
                    else {"railway": "rail", "bridge": "yes", "layer": "1", "gauge": "1067"})
    deck = [[x, -half_len], [x, half_len]]
    route = {"way_id": 1, "level": 0.0, "bridge": False, "tunnel": False}
    if maxheight is not None:
        route["maxheight"] = maxheight
    return {"id": cid, "at_m": x, "xy": [x, 0.0], "relation": "over", "angle_deg": 90.0, "way_id": 500 + cid,
            "highway": tags.get("highway"), "name": None, "level": 1.0, "bridge": True, "tunnel": False,
            "tags": tags, "kind": kind, "railway": tags.get("railway"), "line": deck, "line_at_m": half_len,
            "route": route,
            "structure": {"deck": deck, "deck_at_m": half_len, "ways": 1,
                          "approaches": [[[x, -half_len], [x, -half_len - 40.0]], [[x, half_len], [x, half_len + 40]]]},
            "roads_crossed": roads if roads is not None else [{"line_at_m": half_len, "way_id": 1, "angle_deg": 90.0,
                                                              "route": True, "tags": {"highway": "secondary"}}]}


def _build(crossings, dem, footprint=None):
    c, sec = route_arrays()
    return rail_and_paths.build(crossings, c, sec, FRAME, dem, built_ground(dem), built_road, footprint)


def _deck_vertices(built, name):
    return np.concatenate([m.positions for m in built.meshes if m.name.startswith(name)]).astype(float)


def test_a_footbridge_on_embankments_is_drawn_with_abutments_where_the_ground_meets_it_and_a_pier_off_the_road():
    built = _build([over_crossing()], embankment(6.5))
    (rec,) = built.records
    assert rec["drawn"] and rec["structure"] == "footbridge" and rec["height_from"] == "dem_ends"
    assert rec["clearance_m"] == pytest.approx(6.5 - rail_and_paths.FOOTBRIDGE.depth_m, abs=0.01)
    # The ground comes within 0.3 m of the underside 18.2 m either side of the road: a 36 m span, one pier.
    a, b = rec["supports_m"]
    assert a == pytest.approx(30 - 18.2, abs=1.0) and b == pytest.approx(30 + 18.2, abs=1.0)
    assert rec["piers"] == 1
    names = {m.name: m for m in built.meshes}
    assert {"FOOTBRIDGE_0_DECK", "FOOTBRIDGE_0_STEEL", "1WALL_PIER_FB0"} <= set(names)
    assert not names["1WALL_PIER_FB0"].renderable
    assert "1WALL_ABUT_FB0" not in names  # the embankments come up to the deck: no wall stands out to hit
    # The pier's collider stands clear of the road and its verge, by 1.5 m.
    pier = names["1WALL_PIER_FB0"].positions.astype(float)
    assert np.abs(pier[:, 2]).min() >= HALF + rail_and_paths.VERGE_M + rail_and_paths.ROAD_CLEAR_M - 1e-6
    # Every deck vertex over the road clears it by 4.6 m or more.
    v = _deck_vertices(built, "FOOTBRIDGE_0_DECK")
    over = np.abs(v[:, 2]) <= HALF
    assert (v[over, 1] - ROAD_Z >= rail_and_paths.CLEARANCE_M).all()
    assert built.stats == {"crossings": 1, "drawn": 1, "left_out": 0, "footbridges": 1, "piers": 1}
    # Stage 2b's concrete, its darker underside and the guard rails' steel: one style for every bridge over the route.
    assert built.materials == {"bridge_concrete", "bridge_soffit", "rail"}
    soffit = names["FOOTBRIDGE_0_SOFFIT"]
    assert (soffit.normals[:, 1] < -0.99).all()  # only the faces looking down


def test_a_footbridge_the_dem_cannot_place_is_left_out_unless_the_road_is_signed_for_its_clearance():
    flat = embankment(0.0)
    (rec,) = _build([over_crossing()], flat).records
    assert not rec["drawn"] and "under 4.6 m" in rec["reason"] and "isn't signed" in rec["reason"]
    built = _build([over_crossing(maxheight="5.3")], flat)
    (rec,) = built.records
    assert rec["drawn"] and rec["height_from"] == "maxheight" and rec["clearance_m"] == pytest.approx(5.3)
    assert rec["dem_clearance_m"] < 0
    # Its ends stand clear of the ground (ramps or stairs in the real world): walls a car can hit, so colliders.
    walls = [m for m in built.meshes if m.name == "1WALL_ABUT_FB0"]
    assert len(walls) == 1 and not walls[0].renderable
    # Level over the road and falling at 1:14 beyond it, down to the ground's straight line at its ends.
    v = _deck_vertices(built, "FOOTBRIDGE_0_DECK")
    top = v[:, 1].max()
    assert top == pytest.approx(ROAD_Z + 5.3 + rail_and_paths.FOOTBRIDGE.depth_m, abs=0.01)
    assert rec["top_m"][0] == pytest.approx(top - (30 - HALF - 2) / 14, abs=0.2)
    (low,) = _build([over_crossing(maxheight="3.8")], flat).records
    assert not low["drawn"] and "signed for 3.8 m" in low["reason"]
    # Signed on the other carriageway's way under the same bridge only: the lowest such counts.
    roads = [{"line_at_m": 30.0, "way_id": 1, "angle_deg": 90.0, "route": True, "tags": {"highway": "motorway"}},
             {"line_at_m": 45.0, "way_id": 2, "angle_deg": 90.0, "route": False,
              "tags": {"highway": "motorway", "maxheight": "5.4"}},
             {"line_at_m": 52.0, "way_id": 3, "angle_deg": 90.0, "route": False,
              "tags": {"highway": "motorway_link", "maxheight": "5.2"}}]
    (other,) = _build([over_crossing(roads=roads)], flat).records
    assert other["drawn"] and other["clearance_m"] == pytest.approx(5.2)
    assert other["maxheight_on"] == "another road under the bridge"


def test_a_rail_bridge_carries_ballast_and_rails_and_an_abandoned_one_has_lost_its_rails():
    live = _build([over_crossing("railway")], embankment(7.0))
    (rec,) = live.records
    assert rec["drawn"] and rec["structure"] == "railbridge" and rec["width_m"] == rail_and_paths.RAIL_DECK_M
    assert {"RAILBRIDGE_0_DECK", "RAILBRIDGE_0_STEEL", "RAILBRIDGE_0_BALLAST"} <= {m.name for m in live.meshes}
    gone = _build([over_crossing("railway", tags={"railway": "abandoned", "bridge": "yes"})], embankment(7.0))
    steel = {b: sum(len(m.positions) for m in x.meshes if m.name == "RAILBRIDGE_0_STEEL")
             for b, x in (("live", live), ("gone", gone))}
    assert steel["gone"] < steel["live"]  # the barriers' rails only
    # Its approaches: ballast on the ground, 40 m each way, standing on it.
    assert rec["approach_m"] == pytest.approx(80.0, abs=2.0)
    bed = _deck_vertices(live, "RAILBRIDGE_0_ON_BALLAST")
    assert bed[:, 1].max() == pytest.approx(ROAD_Z + 7.0 + 0.02 + decks.BALLAST[2], abs=0.01)


def test_piers_keep_off_every_road_the_bridge_passes_over():
    # A long rail bridge over the route and a second road 40 m on, along it.
    roads = [{"line_at_m": 60.0, "way_id": 1, "angle_deg": 90.0, "route": True, "tags": {}},
             {"line_at_m": 100.0, "way_id": 2, "angle_deg": 60.0, "route": False,
              "tags": {"highway": "residential"}}]
    cr = over_crossing("railway", roads=roads, half_len=60.0)
    cr["structure"]["deck"] = cr["line"] = [[200.0, -60.0], [200.0, 80.0]]
    n_ends = lambda e, n: ROAD_Z + 7.0 * ((np.asarray(n) < -55.0) | (np.asarray(n) > 75.0)) + 0 * np.asarray(e)  # noqa
    built = _build([cr], n_ends)
    (rec,) = built.records
    assert rec["drawn"] and rec["piers"] >= 3
    pier = [m for m in built.meshes if m.name == "1WALL_PIER_RB0"][0].positions.astype(float)
    along = -pier[:, 2] + 60.0  # deck distance: north is -z in AC axes
    other = (5.8 / 2 + rail_and_paths.VERGE_M + rail_and_paths.ROAD_CLEAR_M) / math.sin(math.radians(60.0))
    assert not ((along > 100.0 - other + 0.01) & (along < 100.0 + other - 0.01)).any()
    assert not ((along > 60.0 - 6.7) & (along < 60.0 + 6.7)).any()


def test_pier_places_split_long_spans_evenly_round_the_keep_out_stretches():
    assert rail_and_paths._piers(0.0, 30.0, 35.0, []) == []
    assert rail_and_paths._piers(0.0, 60.0, 35.0, []) == [30.0]
    assert rail_and_paths._piers(0.0, 60.0, 40.0, [(25.0, 35.0)]) in ([24.0], [36.0])
    assert rail_and_paths._piers(0.0, 60.0, 35.0, [(25.0, 35.0)]) == [20.0, 40.0]  # one wouldn't do: two, even
    assert rail_and_paths._piers(0.0, 50.0, 35.0, [(1.0, 49.0)]) == []  # nowhere to stand: one long span


def test_a_path_under_the_route_s_bridge_is_drawn_beside_it_on_the_ground_out_to_40_m():
    cr = {"id": 3, "at_m": 200.0, "xy": [200.0, 0.0], "relation": "under", "angle_deg": 90.0, "way_id": 700,
          "highway": "footway", "name": "River Walk", "level": -1.0, "bridge": False, "tunnel": False,
          "tags": {"highway": "footway"}, "kind": "path", "railway": None, "line_at_m": 150.0,
          "line": [[200.0, -150.0], [200.0, 150.0]], "route": {"way_id": 1, "level": 1.0, "bridge": True,
                                                                "tunnel": False}}
    ground = lambda e, n: ROAD_Z - 5.0 + 0.0 * np.asarray(n, float)  # noqa: E731 - the route's deck 5 m up
    footprint = shapely.box(0.0, -12.0, 400.0, 12.0)  # the route's road, verges and skirts
    built = _build([cr], ground, footprint)
    (rec,) = built.records
    assert rec["drawn"] and rec["structure"] == "path_under" and rec["gap_m"] == pytest.approx(5.0)
    # 40 m beyond the route's verge (3.2 + 2 m), less the footprint (12 m plus half a metre), both sides.
    assert rec["strip_m"] == pytest.approx(2 * (HALF + 2 + 40 - 12.5), abs=1.5)
    v = _deck_vertices(built, "PATH_UNDER_3")
    assert np.abs(v[:, 2]).min() >= 12.5 - 1e-6 and np.abs(v[:, 2]).max() <= HALF + 2 + 40 + 1e-6
    assert v[:, 1].max() == pytest.approx(ROAD_Z - 5.0 + rail_and_paths.STRIP_LIFT_M, abs=1e-4)
    low = dict(cr, route=dict(cr["route"]))
    (r,) = _build([low], lambda e, n: ROAD_Z - 1.0 + 0.0 * np.asarray(n, float), footprint).records
    assert not r["drawn"] and "only 1.0 m" in r["reason"]
    # A DEM that bridges the ground under the deck (as a narrow underpass's often does): the ground beside counts.
    filled = lambda e, n: np.where(np.abs(np.asarray(n, float)) < 6.0, ROAD_Z, ROAD_Z - 5.0)  # noqa: E731
    (r,) = _build([cr], filled, footprint).records
    assert r["drawn"] and r["gap_m"] == pytest.approx(5.0)
    underpass = dict(cr, route=dict(cr["route"], bridge=False))
    (r,) = _build([underpass], ground, footprint).records
    assert not r["drawn"] and "isn't on a bridge" in r["reason"]


def test_a_way_crossing_the_route_twice_and_two_ways_of_one_bridge_are_one_structure():
    a, b = over_crossing(cid=0), over_crossing(cid=1)
    b["way_id"] = a["way_id"]
    assert len(_build([a, b], embankment(6.5)).records) == 1
    c = over_crossing(cid=2)  # another way of the same bridge, crossing at its split: the same deck
    c["way_id"] = 999
    assert len(_build([over_crossing(cid=0), c], embankment(6.5)).records) == 1


# -- validation ------------------------------------------------------------------------------------------------

def _node(name, pos, tri, renderable=1):
    v = np.zeros(len(pos), dtype=[("pos", "<f4", 3), ("normal", "<f4", 3)])
    v["pos"], v["normal"] = pos, (0.0, 1.0, 0.0)
    return {"class": 2, "name": name, "vertices": v, "indices": np.asarray(tri).ravel(), "renderable": renderable}


@pytest.mark.parametrize("height, ok", [(5.0, True), (4.0, False)])
def test_the_validator_checks_every_deck_over_the_road_clears_it(height, ok):
    road = _node("1ROAD_000", [[-10, 0, -5], [10, 0, -5], [10, 0, 5], [-10, 0, 5]], [[0, 1, 2], [0, 2, 3]])
    deck = _node("FOOTBRIDGE_0_DECK", [[0, height, -1], [1, height, -1], [1, height, 1]], [[0, 1, 2]])
    data = {"nodes": [road, deck]}
    rep = validate.Report()
    validate.overhead_checks(rep, "main", data, validate.RoadSurface(data["nodes"]))
    assert (rep.passed if ok else rep.failed)[0].startswith("main: decks over the road clear it by at least 4.6 m")


def test_left_out_structures_over_the_route_are_in_the_build_s_warnings(tmp_path):
    work = WorkDir(tmp_path / "w")
    work.write_json(route_stage.STAGE, route_stage.CROSSING_LINES, {"version": 1, "crossings": []})
    work.write_json(mesh_stage.STAGE, mesh_stage.RAIL_AND_PATHS, {"version": 1, "structures": [
        {"structure": "railbridge", "drawn": False, "reason": "its underside would clear the road by 3.6 m",
         "name": "South Line", "at_m": 576.1},
        {"structure": "footbridge", "drawn": True, "reason": None, "name": None, "at_m": 300.0},
        {"structure": "path_under", "drawn": False, "reason": "the route isn't on a bridge there", "name": None,
         "at_m": 50.0},
        {"structure": "footbridge", "drawn": False, "reason": "...", "reason_code": "dem", "clearance_m": 1.4,
         "fine": True, "name": None, "at_m": 712.1}]})
    assert validate_stage.rail_and_paths_notes(work) == [
        "South Line (rail bridge) over the route at 0.58 km isn't drawn: its underside would clear the road by 3.6 m.",
        "An unnamed footbridge (footbridge) over the route at 0.71 km: the elevation data predates this bridge, so it "
        "isn't drawn (its deck would clear the road by 1.4 m, under 4.6 m; a footbridge's ramps and stairs aren't in "
        "the bare-earth data either)."]
