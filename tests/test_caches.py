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


def test_an_overpass_answer_with_a_remark_is_used_but_not_kept(settings, monkeypatch):
    answers = [{"elements": [], "remark": "runtime error: Query timed out"}, {"elements": [{"type": "node"}]}]
    monkeypatch.setattr(overpass, "_post", lambda query, s: answers.pop(0))
    assert overpass.fetch("q", settings)["remark"]
    assert overpass.fetch("q", settings) == {"elements": [{"type": "node"}]}  # asked again, then kept
    assert overpass.fetch("q", settings) == {"elements": [{"type": "node"}]}
