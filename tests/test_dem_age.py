"""Bridges the elevation data doesn't hold (`core.dem_age`): the shared warning wording, the judgement, theLIST's
survey names over a crossing (`providers.tas_list.dem.surveys`), and the profile and validate stages that write and
repeat the warnings."""

import datetime as dt
from dataclasses import replace

import httpx
import numpy as np
import pytest

from mapcreator import config
from mapcreator.core import dem_age
from mapcreator.job import Job, Waypoint
from mapcreator.pipeline import profile_stage, route_stage, validate_stage
from mapcreator.pipeline.workdir import WorkDir

HOBART = [Waypoint(lon=147.30, lat=-42.97), Waypoint(lon=147.31, lat=-42.98)]
S = np.arange(0.0, 1001.0)  # a route east along y = 0, a station a metre


def over(cid=0, x=500.0, ends=(-25.0, 25.0), way=7, bridge=True, length=None, **extra):
    """Another road's bridge over the route at x, running north from ends[0] to ends[1]."""
    line = [[x, ends[0]], [x, ends[1]]]
    return {"id": cid, "at_m": x, "xy": [x, 0.0], "relation": "over", "way_id": way, "highway": "trunk",
            "name": "Huon Highway", "tags": {"highway": "trunk", "ref": "A6", "bridge": "yes"}, "bridge": bridge,
            "line": line, "line_at_m": -ends[0], "way_length_m": ends[1] - ends[0] if length is None else length,
            "route": {"way_id": 1, "bridge": False, "tunnel": False}, **extra}


def under(cid=0, x=500.0, way=8):
    """Another road under the route's own bridge at x."""
    return {"id": cid, "at_m": x, "xy": [x, 0.0], "relation": "under", "way_id": way, "highway": "secondary",
            "name": None, "tags": {"highway": "secondary"}, "bridge": False,
            "line": [[x, -150.0], [x, 150.0]], "line_at_m": 150.0, "way_length_m": 300.0,
            "route": {"way_id": 1, "bridge": True, "tunnel": False}}


def ground(level=100.0, embankment=0.0):
    """Flat ground at `level`, raised by `embankment` north and south of |y| = 20 m (a bridge's approach fill)."""
    def sample(x, y):
        y = np.asarray(y, dtype=float)
        return level + embankment * (np.abs(y) >= 20.0) + 0.0 * np.asarray(x, dtype=float)
    return sample


def judge(crossings, sample, *, z=None, fine=True, surveys=None, stations=None):
    z_raw = np.asarray(sample(S, np.zeros_like(S)), dtype=float)
    return dem_age.judge(crossings, [c["at_m"] for c in crossings] if stations is None else stations, S, z_raw,
                         z_raw if z is None else z, np.full(len(S), fine), sample, surveys=surveys)


# -- the shared wording -----------------------------------------------------------------------------------------------

def test_one_wording_for_every_stage():
    assert dem_age.warning("A road") == "A road: the elevation data predates this bridge."
    assert dem_age.warning("A road", then="its deck is left out", detail="2011 LiDAR") == (
        "A road: the elevation data predates this bridge, so its deck is left out (2011 LiDAR).")
    assert dem_age.warning("A road", coarse=True) == "A road: the elevation data here is too coarse to show this bridge."


def test_a_label_names_the_road_its_class_and_ref_and_where():
    assert dem_age.label(over(), 420.0) == "Huon Highway (trunk, A6) over the route at 0.42 km"
    assert dem_age.label(under(x=1050.0)) == "an unnamed road (secondary) under the route's bridge at 1.05 km"


# -- the judgement ----------------------------------------------------------------------------------------------------

def test_a_bridge_on_its_approach_fill_holds_and_one_on_old_flat_ground_predates():
    (held,) = judge([over()], ground(embankment=6.5)).bridges
    assert held.verdict == "holds" and held.separation_m == pytest.approx(6.5)
    rep = judge([over()], ground())  # the DEM shows one level: the crossroads before the interchange
    (old,) = rep.bridges
    assert old.verdict == "predates" and old.separation_m == pytest.approx(0.0)
    assert old.upper_m == pytest.approx(100.0) and old.lower_m == pytest.approx(100.0)
    assert rep.warnings() == [dem_age.warning(old.label, detail=old.detail())]
    assert "the elevation data predates this bridge" in rep.warnings()[0]
    assert rep.summary()["bridges_checked"] == 1 and rep.summary()["bridges_dem_predates"] == 1


def test_just_under_the_clearance_is_flagged_and_just_over_holds():
    assert judge([over()], ground(embankment=4.5)).bridges[0].verdict == "predates"
    assert judge([over()], ground(embankment=4.7)).bridges[0].verdict == "holds"


def test_the_lower_road_is_read_at_its_lowest_near_the_crossing():
    """An embankment the DEM kept over a narrow underpass (a hump on the route) doesn't count against the bridge."""
    def sample(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        hump = 6.0 * np.clip(1 - np.abs(x - 500.0) / 15.0, 0, 1) * (np.abs(y) < 5)
        return 100.0 + 6.0 * (np.abs(y) >= 20.0) + hump
    (b,) = judge([over()], sample).bridges
    assert b.lower_m == pytest.approx(100.0) and b.verdict == "holds"


def test_off_lidar_the_reason_is_the_coarse_one():
    rep = judge([over()], ground(), fine=False, surveys=[[]])
    assert rep.bridges[0].verdict == "coarse"
    assert "too coarse to show this bridge" in rep.warnings()[0] and "no LiDAR" in rep.warnings()[0]
    assert rep.summary()["bridges_dem_coarse"] == 1
    # A DEM that doesn't say which surveys it holds (the mainland's) isn't said to have none.
    assert "LiDAR" not in judge([over()], ground(), fine=False).warnings()[0]


def test_the_route_s_own_bridge_is_judged_by_its_profile_over_the_road_below():
    flat = ground()
    raised = np.where(np.abs(S - 500.0) <= 30.0, 106.0, 100.0)  # the deck drawn between its ends
    assert judge([under()], flat, z=raised).bridges[0].verdict == "holds"
    (b,) = judge([under()], flat).bridges  # its ends on the old ground: the deck at the road's own level
    assert b.verdict == "predates" and "under the route's bridge" in b.label


def test_untagged_ways_long_bridges_and_unmatched_crossings_are_not_judged():
    rep = judge([over(bridge=False), over(cid=1, x=700.0, way=9, length=400.0), over(cid=2, x=800.0, way=10)],
                ground(), stations=[500.0, 700.0, None])
    assert rep.not_bridges == 1 and rep.unmatched == 1
    (long_one,) = rep.bridges
    assert long_one.verdict == "unjudged" and "150 m" in long_one.why
    assert rep.warnings() == [] and rep.summary()["bridges_checked"] == 0


def test_two_points_of_the_same_two_ways_are_one_bridge_judged_at_its_worst():
    def sample(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return 100.0 + np.where(x < 600, 6.0, 1.0) * (np.abs(y) >= 20.0)
    rep = judge([over(cid=0, x=400.0), over(cid=1, x=800.0)], sample)
    (b,) = rep.bridges
    assert b.crossings == [0, 1] and b.along_m == 800.0 and b.verdict == "predates"


def test_the_surveys_name_the_dem_s_lidar_and_the_newer_ones():
    surveys = [[{"name": "Mount Wellington 2011", "completed": "2011-01-28", "in_dem": True},
                {"name": "Kingborough 2022", "completed": "2022-02-24", "in_dem": False}]]
    (b,) = judge([over()], ground(), surveys=surveys).bridges
    assert [v["name"] for v in b.surveys] == ["Mount Wellington 2011"]
    assert [v["name"] for v in b.newer_surveys] == ["Kingborough 2022"]
    text = b.warning()
    assert "its newest LiDAR there is Mount Wellington 2011, finished January 2011" in text
    assert "Kingborough 2022 is newer, but builds can't use it yet" in text


def test_a_highway_s_two_decks_over_the_route_share_one_line_at_the_worst():
    def sample(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return 100.0 + np.where(x < 505, 2.0, 1.0) * (np.abs(y) >= 20.0)
    rep = judge([over(cid=0, x=500.0, way=7), over(cid=1, x=510.0, way=8)], sample, stations=[410.0, 410.0])
    assert len(rep.bridges) == 2 and rep.summary()["bridges_dem_predates"] == 2
    (line,) = rep.warnings()
    assert line.startswith("Huon Highway (trunk, A6) over the route at 0.41 km (2 decks): the elevation data predates")
    assert "1.0 m apart" in line


def test_many_flagged_bridges_are_listed_up_to_a_cap_then_counted():
    crossings = [over(cid=k, x=100.0 + 100 * k, way=20 + k) for k in range(8)]
    lines = judge(crossings, ground()).warnings()
    assert len(lines) == dem_age.MAX_LISTED + 1 and lines[-1].startswith("3 more bridges")


# -- theLIST's surveys over a point -----------------------------------------------------------------------------------

def test_the_surveys_over_a_point_are_split_at_the_dem_s_release(tmp_path, monkeypatch):
    from mapcreator.providers.tas_list import dem, lidar_index

    def ms(y, m, d):
        return dt.datetime(y, m, d, tzinfo=dt.UTC).timestamp() * 1000

    def square(x0, y0, x1, y1):  # clockwise, as Esri writes outer rings
        return [[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]]

    answer = {"features": [
        {"attributes": {"PROJ_NAME": "Kingborough 2022", "COMPLETED": ms(2022, 2, 24)},
         "geometry": {"rings": [square(500000, 5250000, 502000, 5251000)]}},
        {"attributes": {"PROJ_NAME": "Mount Wellington 2011", "COMPLETED": ms(2011, 1, 28)},
         "geometry": {"rings": [square(500000, 5250000, 501000, 5251000)]}},
    ]}
    calls = []

    def fake_get(url, params=None, **kw):
        calls.append(params)
        return httpx.Response(200, json=answer, request=httpx.Request("GET", url))

    monkeypatch.setattr(lidar_index.httpx, "get", fake_get)
    settings = replace(config.load(), cache_dir=tmp_path / "cache")
    route = np.array([[500100.0, 5250500.0], [501900.0, 5250500.0]])
    mask = dem.lidar_mask(route, settings.cache_dir, "t")
    got = dem.surveys([[500500, 5250500], [501500, 5250500], [503000, 5250500]], settings.cache_dir, "t",
                      around=route)
    assert len(calls) == 1  # the road's own box: lidar_mask's cached answer
    assert mask.tolist() == [True, False]
    assert got[0] == [{"name": "Mount Wellington 2011", "completed": "2011-01-28", "in_dem": True},
                      {"name": "Kingborough 2022", "completed": "2022-02-24", "in_dem": False}]
    assert got[1] == [{"name": "Kingborough 2022", "completed": "2022-02-24", "in_dem": False}] and got[2] == []
    assert dem.surveys(np.zeros((0, 2)), settings.cache_dir, "t") == []


def test_a_footprint_ring_that_crosses_itself_is_repaired_not_fatal():
    """The server generalises the footprints, and some come back crossing themselves (seen statewide, Nov 2026)."""
    import shapely

    from mapcreator.providers.tas_list import lidar_index

    bowtie = [[0, 0], [0, 10], [10, 0], [10, 10], [0, 0]]  # clockwise-ish, crossing itself at (5, 5)
    good = [[20, 0], [20, 10], [30, 10], [30, 0], [20, 0]]
    features = [{"attributes": {"PROJ_NAME": "Bent", "COMPLETED": 1.0e12}, "geometry": {"rings": [bowtie, good]}},
                {"attributes": {"PROJ_NAME": "Fine", "COMPLETED": 1.1e12}, "geometry": {"rings": [good]}}]
    got = lidar_index.footprints(features)
    assert [name for name, _, _ in got] == ["Bent", "Fine"]
    assert shapely.contains_xy(got[0][2], 25, 5) and got[0][2].is_valid
    assert not lidar_index.coverage(features, dt.date(2030, 1, 1)).is_empty


# -- the stages -------------------------------------------------------------------------------------------------------

class _Dem:
    """Flat ground at 100 m: the old crossroads, with no sign of the bridge over the route."""

    resolution_m = 2.0

    def load_corridor(self, xy, m):
        return 1

    def sample(self, x, y):
        return np.full(np.shape(x), 100.0)

    def surveys(self, xy, *, around=None):
        return [[{"name": "Mount Wellington 2011", "completed": "2011-01-28", "in_dem": True},
                 {"name": "Kingborough 2022", "completed": "2022-02-24", "in_dem": False}] for _ in xy]


class _Region:
    epsg = 28355

    def elevation(self, settings):
        return _Dem()


def _route(work, crossings):
    xy = np.column_stack([np.arange(0, 1001.0, 10.0), np.zeros(101)])
    runs = [{"from_m": 0.0, "to_m": 1000.0, "way_id": 1, "tags": {"highway": "secondary", "surface": "asphalt"}}]
    np.savez_compressed(work.stage(route_stage.STAGE) / "route.npz", xy=xy, seg_run=np.zeros(100, np.int64),
                        epsg=28355)
    work.write_json(route_stage.STAGE, "runs.json", runs)
    work.write_json(route_stage.STAGE, route_stage.CROSSINGS, {"version": 1, "crossings": crossings,
                                                               "unresolved": []})


def test_the_profile_stage_writes_the_judgement_and_validate_repeats_its_warnings(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_stage, "job_region", lambda waypoints: _Region())
    job = Job(track_id="t", waypoints=HOBART)
    work = WorkDir(tmp_path / "work")
    _route(work, [over(x=480.0)])
    out = profile_stage.run(job, config.load(), work)
    doc = work.read_json(profile_stage.STAGE, dem_age.FILE)
    assert doc["version"] == dem_age.FORMAT_VERSION and doc["rule"]["min_clearance_m"] == dem_age.MIN_CLEARANCE_M
    (b,) = doc["bridges"]
    assert b["verdict"] == "predates" and b["crossings"] == [0] and b["newer_surveys"][0]["name"] == "Kingborough 2022"
    assert out["bridges_checked"] == 1 and out["bridges_dem_predates"] == 1
    assert out["dem_warnings"] == doc["warnings"] and "predates this bridge" in doc["warnings"][0]
    assert validate_stage._dem_age_warnings(work) == doc["warnings"]


def test_without_crossings_nothing_is_flagged_and_an_older_profile_stage_gives_no_warnings(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_stage, "job_region", lambda waypoints: _Region())
    work = WorkDir(tmp_path / "work")
    assert validate_stage._dem_age_warnings(work) == []  # no dem_age.json at all
    _route(work, [])
    out = profile_stage.run(Job(track_id="t", waypoints=HOBART), config.load(), work)
    assert out["bridges_checked"] == 0 and out["dem_warnings"] == []
    assert work.read_json(profile_stage.STAGE, dem_age.FILE)["bridges"] == []
