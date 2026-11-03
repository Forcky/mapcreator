"""The on-disk caches keep only whole, good answers."""

from dataclasses import replace

import httpx
import numpy as np
import pytest

from mapcreator import config
from mapcreator.providers import cache
from mapcreator.providers.osm import overpass
from mapcreator.providers.tas_list import buildings


@pytest.fixture
def settings(tmp_path):
    return replace(config.load(), cache_dir=tmp_path / "cache")


def test_atomic_writes_leave_no_temporary_files(tmp_path):
    cache.write_json(tmp_path / "a" / "x.json", {"k": 1})
    cache.save_npy(tmp_path / "a" / "t.npy", np.arange(4))
    assert cache.read_json(tmp_path / "a" / "x.json") == {"k": 1}
    assert np.load(tmp_path / "a" / "t.npy").tolist() == [0, 1, 2, 3]
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == ["t.npy", "x.json"]


def test_a_truncated_json_entry_reads_as_missing(tmp_path):
    (tmp_path / "bad.json").write_text('{"elements": [', encoding="utf-8")
    assert cache.read_json(tmp_path / "bad.json") is None
    assert cache.read_json(tmp_path / "absent.json") is None


def test_an_arcgis_error_is_not_cached_as_no_buildings(settings, monkeypatch):
    calls = []

    def fake_get(url, params=None, **kw):
        calls.append(params["resultOffset"])
        return httpx.Response(200, json={"error": {"code": 400, "message": "Failed to execute query."}},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(buildings.httpx, "get", fake_get)
    monkeypatch.setattr(buildings.time, "sleep", lambda s: None)
    with pytest.raises(buildings.ServiceError):
        buildings.footprints((500000, 5250000, 500100, 5250100), settings)
    assert len(calls) == 4  # retried
    assert not list(settings.cache_dir.rglob("*.json"))  # nothing cached


def test_the_lidar_index_keeps_only_projects_the_dem_can_hold(settings, monkeypatch):
    import datetime as dt

    import shapely

    from mapcreator.providers.tas_list import lidar_index
    from mapcreator.providers.tas_list.dem import DEM_DATE, lidar_mask

    def ms(y, m, d):
        return dt.datetime(y, m, d, tzinfo=dt.UTC).timestamp() * 1000

    def square(x0, y0, x1, y1):  # clockwise, as Esri writes outer rings
        return [[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]]

    answer = {"features": [
        {"attributes": {"PROJ_NAME": "Forestry 2014", "COMPLETED": ms(2014, 8, 30)},
         "geometry": {"rings": [square(500000, 5250000, 501000, 5251000)]}},
        {"attributes": {"PROJ_NAME": "Program 2021", "COMPLETED": ms(2021, 11, 30)},
         "geometry": {"rings": [square(501000, 5250000, 502000, 5251000)]}},
    ]}
    calls = []

    def fake_get(url, params=None, **kw):
        calls.append(params)
        return httpx.Response(200, json=answer, request=httpx.Request("GET", url))

    monkeypatch.setattr(lidar_index.httpx, "get", fake_get)
    area = lidar_index.coverage(lidar_index.projects((500000, 5250000, 502000, 5251000), settings.cache_dir, "t"),
                                DEM_DATE)
    assert shapely.contains_xy(area, 500500, 5250500) and not shapely.contains_xy(area, 501500, 5250500)
    pts = [[500500, 5250500], [500990, 5250500], [501500, 5250500]]  # inside, at the edge, too late
    assert lidar_mask(pts, settings.cache_dir, "t").tolist() == [True, False, False]
    asked = len(calls)
    assert lidar_mask(pts, settings.cache_dir, "t").tolist() == [True, False, False]
    assert len(calls) == asked  # the same box again comes from the cache


def test_an_overpass_answer_with_a_remark_is_used_but_not_kept(settings, monkeypatch):
    answers = [{"elements": [], "remark": "runtime error: Query timed out"}, {"elements": [{"type": "node"}]}]
    monkeypatch.setattr(overpass, "_post", lambda query, s: answers.pop(0))
    assert overpass.fetch("q", settings)["remark"]
    assert overpass.fetch("q", settings) == {"elements": [{"type": "node"}]}  # asked again, then kept
    assert overpass.fetch("q", settings) == {"elements": [{"type": "node"}]}
