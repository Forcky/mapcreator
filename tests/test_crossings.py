"""Roads crossing the route without meeting it (`core.crossings`, `01_route/crossings.json`), and what is built on
them today: humps under a road crossing over are bridged (`core.profile.humps_under`, the profile stage), the route's
own bridges carry crown only (the section stage) and get rails on both sides over a real gap (`roadside.bridge_mask`,
the mesh stage)."""

import numpy as np
import pytest
from pyproj import Transformer

from mapcreator import config
from mapcreator.core import crossings, roadside
from mapcreator.core import profile as pf
from mapcreator.core import section as sec
from mapcreator.core.scene import LocalFrame
from mapcreator.job import Job
from mapcreator.pipeline import profile_stage, route_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.osm import graph

IDENT = Transformer.from_crs(4326, 4326, always_xy=True)  # the test networks are drawn in metres
SECONDARY = {"highway": "secondary"}


def osm(nodes: dict[int, tuple[float, float]], ways: list[tuple[int, list[int], dict]]):
    els = [{"type": "node", "id": i, "lon": x, "lat": y} for i, (x, y) in nodes.items()]
    els += [{"type": "way", "id": w, "nodes": refs, "tags": tags} for w, refs, tags in ways]
    return graph.build({"elements": els}, IDENT)


def network(others: list[tuple[list, dict]], main=None):
    """A road east along y = 0 from 0 to 1,000 m, nodes every 100 m (ids 0-10; way 1, or the ways `main` gives as
    (first node, last node, tags)), and other ways through their points: (x, y) for a new node, or a main road
    node's id. Other ways are 100, 101, ..."""
    nodes = {i: (100.0 * i, 0.0) for i in range(11)}
    ways = [(1, list(range(11)), SECONDARY)] if main is None else [
        (k + 1, list(range(a, b + 1)), tags) for k, (a, b, tags) in enumerate(main)]
    nid = 1000
    for k, (points, tags) in enumerate(others):
        refs = []
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
    r = graph.route(g, np.array([[start, 0.0], [end, 0.0]]))
    return crossings.find(g, r.xy, r.seg_way)


def test_a_bridge_over_the_route_is_listed_over_with_its_tags_and_line():
    bridge = {"highway": "tertiary", "name": "Over Road", "bridge": "yes", "layer": "1", "lanes": "2"}
    doc = find(network([([(450.0, -60.0), (450.0, 60.0)], bridge)]))
    assert doc["version"] == crossings.FORMAT_VERSION and doc["unresolved"] == []
    (c,) = doc["crossings"]
    assert (c["id"], c["relation"], c["way_id"], c["name"], c["highway"]) == (0, "over", 100, "Over Road", "tertiary")
    assert c["at_m"] == pytest.approx(440.0, abs=0.1)  # the route starts 10 m in
    assert c["xy"] == pytest.approx([450.0, 0.0]) and c["angle_deg"] == pytest.approx(90.0)
    assert c["level"] == 1.0 and c["bridge"] and not c["tunnel"]
    assert c["tags"] == {"highway": "tertiary", "name": "Over Road", "bridge": "yes", "layer": "1", "lanes": "2"}
    assert c["way_length_m"] == pytest.approx(120.0) and c["line_at_m"] == pytest.approx(60.0)
    assert c["route"] == {"way_id": 1, "level": 0.0, "bridge": False, "tunnel": False}
    assert crossings.over(doc["crossings"]) == [c]


def test_a_road_under_the_route_s_own_bridge_is_listed_under():
    main = [(0, 4, SECONDARY), (4, 5, {**SECONDARY, "bridge": "yes", "layer": "1"}), (5, 10, SECONDARY)]
    doc = find(network([([(450.0, -60.0), (450.0, 60.0)], {"highway": "residential"})], main=main))
    (c,) = doc["crossings"]
    assert c["relation"] == "under" and c["route"]["bridge"] and c["route"]["level"] == 1.0
    assert crossings.over(doc["crossings"]) == []


def test_a_side_road_meeting_the_route_at_a_node_is_a_junction_not_a_crossing():
    doc = find(network([([3, (300.0, 80.0), (300.0, 160.0)], {"highway": "residential"}),
                        ([(520.0, -50.0), 5, (480.0, 50.0)], {"highway": "residential"})]))  # crosses at node 5
    assert doc["crossings"] == [] and doc["unresolved"] == []


def test_roads_crossing_at_one_level_are_unresolved_and_a_tunnel_is_under():
    doc = find(network([([(250.0, -60.0), (250.0, 60.0)], {"highway": "residential"}),
                        ([(650.0, -60.0), (650.0, 60.0)], {"highway": "residential", "tunnel": "yes"})]))
    (u,) = doc["unresolved"]
    assert u["way_id"] == 100 and "relation" not in u
    (c,) = doc["crossings"]
    assert (c["way_id"], c["relation"], c["level"]) == (101, "under", -1.0)


def test_a_way_crossing_twice_is_listed_twice_and_a_long_line_is_clipped():
    loop = [(300.0, -400.0), (300.0, 400.0), (360.0, 400.0), (360.0, -400.0)]
    doc = find(network([(loop, {"highway": "trunk", "bridge": "yes", "layer": "1"})]))
    assert [c["at_m"] for c in doc["crossings"]] == pytest.approx([290.0, 350.0], abs=0.1)
    first = doc["crossings"][0]
    assert first["way_length_m"] == pytest.approx(1660.0)
    assert np.hypot(*np.diff(np.array(first["line"]), axis=0).T).sum() == pytest.approx(2 * crossings.CLIP_M)


@pytest.mark.parametrize("tags,expected", [
    ({}, 0.0), ({"bridge": "yes"}, 1.0), ({"tunnel": "yes"}, -1.0), ({"bridge": "yes", "layer": "2"}, 2.0),
    ({"layer": "-1"}, -1.0), ({"bridge": "viaduct", "layer": "1;2"}, 1.0), ({"bridge": "yes", "layer": "x"}, 1.0),
    ({"bridge": "no", "layer": "0"}, 0.0), ({"bridge": "low_water_crossing"}, 0.0),
])
def test_a_way_s_level_is_its_layer_else_its_bridge_or_tunnel(tags, expected):
    assert crossings.level(tags) == expected


def test_over_leaves_out_crossings_above_the_route_s_own_tunnel():
    c = {"relation": "over", "route": {"tunnel": True}}
    assert crossings.over([c, {**c, "route": {"tunnel": False}}, {**c, "relation": "under"}]) == [
        {**c, "route": {"tunnel": False}}]


# -- the route stage ------------------------------------------------------------------------------------------------

HOBART = [{"lon": 147.301, "lat": -42.8001}, {"lon": 147.319, "lat": -42.7999}]


def hobart_roads() -> dict:
    """A secondary road east along -42.80, and a trunk bridge crossing it north to south at 147.310, with a node of
    its own right over the road's node there (not shared: a crossing, not a junction)."""
    nodes = [{"type": "node", "id": i + 1, "lon": 147.300 + i * 0.002, "lat": -42.80} for i in range(11)]
    over = [{"type": "node", "id": 100 + k, "lon": 147.310, "lat": -42.7990 - 0.001 * k} for k in range(3)]
    ways = [{"type": "way", "id": 1, "nodes": [n["id"] for n in nodes], "tags": SECONDARY},
            {"type": "way", "id": 2, "nodes": [100, 101, 102],
             "tags": {"highway": "trunk", "name": "High Road", "bridge": "yes", "layer": "1"}}]
    return {"elements": [*nodes, *over, *ways]}


@pytest.mark.parametrize("junctions", ["closed", "none"])
def test_the_route_stage_writes_crossings_json_whatever_the_junctions_option(tmp_path, monkeypatch, junctions):
    monkeypatch.setattr(route_stage.osm_roads, "for_bbox", lambda bbox, settings: hobart_roads())
    work = WorkDir(tmp_path / "work")
    assert route_stage.crossings(work) == []  # before the route stage, or a route stage from before crossings
    out = route_stage.run(Job(track_id="hobart", waypoints=HOBART, junctions=junctions), config.load(), work)
    assert out["crossings"] == {"over": 1, "under": 0, "unresolved": 0}
    doc = work.read_json(route_stage.STAGE, route_stage.CROSSINGS)
    assert doc["version"] == crossings.FORMAT_VERSION and doc["crs"] == "EPSG:28355"
    (c,) = route_stage.crossings(work)
    assert (c["name"], c["relation"]) == ("High Road", "over")
    assert c["lonlat"] == pytest.approx([147.310, -42.80], abs=1e-6)


# -- humps under a road crossing over ------------------------------------------------------------------------------

def _hump(s, at=455.0, height=9.0, width=50.0):
    """A bare-earth hump: a raised cosine `width` across at its foot."""
    x = np.clip((s - at) / (width / 2), -1, 1)
    return height * 0.5 * (1 + np.cos(np.pi * x))


def test_a_hump_beside_an_underpass_is_found_out_to_its_foot_and_bridged():
    s = np.arange(0, 1000.0)
    ground = 30.0 + 0.05 * s
    z = ground + _hump(s)
    mask = pf.humps_under(s, z, [488.0])
    assert mask[440:470].all() and not mask[:420].any() and not mask[490:].any()
    prof = pf.smooth_profile(s, z, mask)
    assert np.abs(prof.z - ground).max() < 0.15  # the embankment no longer lifts the route
    assert np.abs(prof.grade - 0.05).max() < 0.01


def test_a_hump_far_from_any_crossing_and_a_low_one_stay():
    s = np.arange(0, 1000.0)
    z = 30.0 + 0.05 * s + _hump(s)
    assert not pf.humps_under(s, z, [600.0]).any()  # the hump ends at 480 m: more than 40 m from 600 m
    assert not pf.humps_under(s, z, []).any()
    assert not pf.humps_under(s, 30.0 + _hump(s, height=0.8), [455.0]).any()  # under a metre: real ground


@pytest.mark.parametrize("radius", [800.0, 2000.0])
def test_a_real_crest_under_a_bridge_keeps_its_shape(radius):
    s = np.arange(0, 1000.0)
    z = 50.0 - (s - 500.0) ** 2 / (2 * radius)  # a crest of this vertical radius, peaking under the bridge
    assert not pf.humps_under(s, z, [500.0]).any()


# -- the stages that build on it ------------------------------------------------------------------------------------

class _Dem:
    """A DEM rising 5% east, with a 9 m embankment beside an underpass at x = 455 m (a ridge north to south), and
    falling 4% to the north across the road (so camber reads 4%)."""

    resolution_m = 2.0

    def load_corridor(self, xy, m):
        return 1

    def sample(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return 30.0 + 0.05 * x + _hump(x) + 0.04 * y


class _Region:
    epsg = 28355

    def elevation(self, settings):
        return _Dem()


def _route(work, *, over_at=None, bridge=None):
    xy = np.column_stack([np.arange(0, 1001.0, 10.0), np.zeros(101)])
    runs = [{"from_m": 0.0, "to_m": 1000.0, "way_id": 1, "tags": {"highway": "secondary", "surface": "asphalt"}}]
    seg_run = np.zeros(100, np.int64)
    if bridge is not None:  # the route's own bridge over [a, b) metres, as its own run
        a, b = bridge
        runs = [{**runs[0], "to_m": a}, {**runs[0], "from_m": a, "to_m": b, "way_id": 2,
                                         "tags": {**runs[0]["tags"], "bridge": "yes", "layer": "1"}},
                {**runs[0], "from_m": b, "way_id": 3}]
        mid = np.arange(100) * 10.0
        seg_run = np.where(mid < a, 0, np.where(mid < b, 1, 2)).astype(np.int64)
    np.savez_compressed(work.stage(route_stage.STAGE) / "route.npz", xy=xy, seg_run=seg_run, epsg=28355)
    work.write_json(route_stage.STAGE, "runs.json", runs)
    if over_at is not None:
        c = {"at_m": over_at, "xy": [over_at, 0.0], "relation": "over", "route": {"tunnel": False}}
        work.write_json(route_stage.STAGE, route_stage.CROSSINGS, {"version": 1, "crossings": [c], "unresolved": []})


def test_the_profile_stage_bridges_the_hump_under_a_crossing_and_keeps_it_without_one(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_stage, "job_region", lambda waypoints: _Region())
    job = Job(track_id="t", waypoints=HOBART)
    grades = {}
    for name, over_at in (("without", None), ("with", 488.0)):
        work = WorkDir(tmp_path / name)
        _route(work, over_at=over_at)
        out = profile_stage.run(job, config.load(), work)
        c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
        grades[name] = out["max_grade_10m"]
        assert not c["bridge"].any() and not c["tunnel"].any()
        if over_at is None:
            assert not c["under"].any() and out["humps_under_crossings_m"] == 0.0
        else:
            assert c["under"][440:470].all() and not c["under"][:400].any() and not c["under"][520:].any()
            assert out["humps_under_crossings_m"] == pytest.approx(c["under"].sum(), abs=1)
            assert np.abs(c["z"] - (30.0 + 0.05 * c["s"]))[300:700].max() < 0.2
    assert grades["without"] > 0.3 and grades["with"] < 0.06


def test_the_section_stage_gives_the_route_s_bridge_and_the_bridged_hump_crown_only(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_stage, "job_region", lambda waypoints: _Region())
    monkeypatch.setattr(section_stage, "job_region", lambda waypoints: _Region())
    job = Job(track_id="t", waypoints=HOBART)
    work = WorkDir(tmp_path / "work")
    _route(work, over_at=488.0, bridge=(200.0, 220.0))  # a 20 m bridge, and the hump under the crossing
    profile_stage.run(job, config.load(), work)
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    assert c["bridge"][200:220].all() and c["bridge"].sum() == 20
    section_stage.run(job, config.load(), work)
    e = np.load(work.root / section_stage.STAGE / "section.npz")["superelevation"]
    assert np.all(e[200:220] == 0.0) and np.all(e[c["under"]] == 0.0)  # crown only on the deck and the hump
    assert e[100] == pytest.approx(0.04, abs=0.005) and e[800] == pytest.approx(0.04, abs=0.005)  # the DEM's camber


def test_the_off_ground_factor_is_nil_on_a_short_bridge_and_eases_back_over_the_ease():
    s = np.arange(0, 400.0)
    mask = np.zeros(400, bool)
    mask[200:211] = True
    f = sec.off_ground_factor(s, mask)
    assert np.all(f[200:211] == 0.0)
    assert np.all(f[:139] == 1.0) and np.all(f[272:] == 1.0)
    assert np.all(np.diff(f[100:200]) <= 1e-12) and np.all(np.diff(f[211:300]) >= -1e-12)  # no step, no bump
    assert np.all(sec.off_ground_factor(s, np.zeros(400, bool)) == 1.0)


def test_a_bridge_with_a_real_gap_gets_rails_on_both_sides_past_its_ends_and_a_low_one_none():
    s = np.arange(0, 600.0)
    bridge = np.zeros(600, bool)
    bridge[100:130] = True  # a 30 m bridge 3 m over its gully at the middle
    bridge[400:420] = True  # a 20 m bridge on the ground the DEM shows
    gap = np.zeros(600)
    gap[100:130] = 3.0 * np.sin(np.linspace(0, np.pi, 30))
    gap[400:420] = 0.8
    m = roadside.bridge_mask(s, bridge, gap)
    assert m[100:130].all() and m[100 - 8] and m[129 + 8] and not m[100 - 10] and not m[129 + 10]
    assert not m[300:].any()
    nan_gap = np.where(bridge, np.nan, 0.0)  # no DEM under it: over the sea
    assert roadside.bridge_mask(s, bridge, nan_gap)[400:420].all()
    assert not roadside.bridge_mask(s, np.zeros(600, bool), gap).any()


def test_bridge_rails_are_built_on_both_sides_with_their_colliders():
    s = np.arange(0, 400.0)
    n = len(s)
    xy = np.column_stack([500_000 + s, np.full(n, 5_200_000.0)])
    bridge = (s >= 150) & (s < 200)
    on_bridge = roadside.bridge_mask(s, bridge, np.where(bridge, 6.0, 0.0))
    meshes, stats = roadside.guard_rails(s, xy, np.full(n, 50.0), np.zeros(n), np.full(n, 3.2), np.full(n, 3.2),
                                         np.full(n, 0.025), np.full(n, 0.025), lambda x, y: np.full(np.shape(x), 50.0),
                                         LocalFrame(500_000.0, 5_200_000.0, 0.0),
                                         masks={"L": on_bridge, "R": on_bridge})
    assert sorted(m.name for m in meshes) == ["1WALL_RAIL_L000", "1WALL_RAIL_R000", "RAIL_L000", "RAIL_R000"]
    assert stats["L"] == stats["R"] == {"runs": 1, "metres": 66.0}  # 50 m and 8 m past each end
