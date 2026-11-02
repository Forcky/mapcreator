"""Stable public track ids, retention, and S3-compatible storage."""

import re
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mapcreator.job import Job, public_track_id
from mapcreator.service.api import create_app
from mapcreator.service.db import JobStore, route_hash
from mapcreator.service.settings import ServiceSettings
from mapcreator.service.storage import LocalStorage, S3Storage, storage_for

TOKEN = "w" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
DAY = 86400.0


def job(lon: float = 147.33, track_id: str = "my_road") -> Job:
    return Job.model_validate({"track_id": track_id,
                               "waypoints": [{"lon": lon, "lat": -42.81}, {"lon": 147.42, "lat": -42.74}]})


# -- public ids ------------------------------------------------------------------

def test_public_ids_are_unique_per_road_stable_and_short():
    a, b = job(147.33), job(147.34)
    ida, idb = public_track_id("grasstree_hill", route_hash(a)), public_track_id("grasstree_hill", route_hash(b))
    assert ida != idb  # same name, different road: no collision in a player's AC install
    assert ida == public_track_id("grasstree_hill", route_hash(job(147.33, "other_name")))  # stable
    assert re.fullmatch(r"mc_grasstree_hill_[0-9a-f]{8}", ida)
    long_id = public_track_id("x" * 32, route_hash(a))
    assert len(long_id) <= 32 and Job.model_fields["track_id"].metadata  # still a valid AC folder name
    Job.model_validate({**a.model_dump(), "track_id": long_id})
    assert public_track_id("mc_road", route_hash(a)).count("mc_") == 1
    assert public_track_id("___", route_hash(a)).startswith("mc_road_")


# -- retention -------------------------------------------------------------------

class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def test_prune_deletes_old_finished_jobs_and_blanks_ip_hashes(tmp_path):
    clock = Clock()
    store = JobStore(tmp_path / "jobs.sqlite3", clock=clock)
    done = store.create(job(147.30), ip_hash="ip-a")
    queued = store.create(job(147.31), ip_hash="ip-b")
    claimed = store.claim_next()
    assert claimed.id == done.id
    store.complete(done.id, claimed.worker_id, {"key": f"{done.id}/x.zip"})

    clock.t += 3 * DAY
    assert store.prune(30 * DAY, 2 * DAY) == []  # nothing old enough to delete yet...
    assert store.get(queued.id).ip_hash == ""  # ...but IP hashes older than 2 days are forgotten
    assert store.get(done.id) is not None

    clock.t += 30 * DAY
    assert store.prune(30 * DAY, 2 * DAY) == [{"key": f"{done.id}/x.zip"}]
    assert store.get(done.id) is None
    assert store.get(queued.id) is not None  # unfinished work is never pruned


def test_the_web_app_prunes_old_jobs_and_their_downloads(tmp_path):
    settings = ServiceSettings(db_path=tmp_path / "jobs.sqlite3", storage_dir=tmp_path / "downloads",
                               ip_salt="s", worker_token=TOKEN)
    app = create_app(settings, router=None, storage=LocalStorage(tmp_path / "downloads", "/downloads"))
    clock = Clock()
    app.state.store.clock = clock
    c = TestClient(app, client=("203.0.113.9", 50000))
    job_id = c.post("/api/jobs", json={"job": job().model_dump(mode="json")}).json()["id"]
    lease = c.post("/api/worker/claim", headers=AUTH).json()
    key = f"{job_id}/{lease['job']['track_id']}.zip"
    (tmp_path / "downloads" / job_id).mkdir(parents=True)
    (tmp_path / "downloads" / key).write_bytes(b"PK")
    c.post(f"/api/worker/jobs/{job_id}/complete", headers=AUTH,
           json={"worker_id": lease["worker_id"], "result": {"key": key}})

    clock.t += 31 * DAY
    c.post("/api/worker/claim", headers=AUTH)  # any traffic triggers the hourly prune
    assert c.get(f"/api/jobs/{job_id}").status_code == 404
    assert not (tmp_path / "downloads" / key).exists()


# -- S3-compatible storage ---------------------------------------------------------

class FakeS3:
    def __init__(self) -> None:
        self.puts, self.deletes = [], []

    def put_object(self, **kw):
        kw["Body"] = kw["Body"].read()
        self.puts.append(kw)

    def delete_object(self, **kw):
        self.deletes.append(kw)


def test_s3_storage_uploads_with_the_headers_content_manager_needs(tmp_path):
    zip_path = tmp_path / "t.zip"
    zip_path.write_bytes(b"PK\x03\x04")
    fake = FakeS3()
    s = S3Storage(bucket="tracks", public_base="https://downloads.example.org/", client=fake)
    s.put(zip_path, "abc/mc_road_12345678.zip")
    put = fake.puts[0]
    assert put["Bucket"] == "tracks" and put["Key"] == "abc/mc_road_12345678.zip" and put["Body"] == b"PK\x03\x04"
    assert put["ContentType"] == "application/zip"
    assert put["ContentDisposition"] == 'attachment; filename="mc_road_12345678.zip"'
    assert s.url("abc/mc_road_12345678.zip") == "https://downloads.example.org/abc/mc_road_12345678.zip"
    s.delete("abc/mc_road_12345678.zip")
    assert fake.deletes == [{"Bucket": "tracks", "Key": "abc/mc_road_12345678.zip"}]
    with pytest.raises(ValueError):
        s.put(zip_path, "../escape.zip")


def test_storage_for_picks_the_configured_backend(tmp_path, monkeypatch):
    local = ServiceSettings(db_path=tmp_path / "j.sqlite3", storage_dir=tmp_path / "d", ip_salt="s")
    assert isinstance(storage_for(local), LocalStorage)

    made = {}
    fake_boto3 = types.SimpleNamespace(client=lambda *a, **kw: made.setdefault("kw", kw) or FakeS3())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    s3 = ServiceSettings(db_path=tmp_path / "j.sqlite3", ip_salt="s", storage="s3",
                         s3_endpoint="https://example.r2.example.org", s3_bucket="tracks",
                         s3_access_key_id="id", s3_secret_access_key="secret",
                         download_base="https://downloads.example.org")
    st = storage_for(s3)
    assert isinstance(st, S3Storage) and made["kw"]["endpoint_url"] == "https://example.r2.example.org"


@pytest.mark.parametrize("overrides", [
    {"storage": "ftp"},
    {"storage": "s3", "s3_bucket": "t", "download_base": "https://d.example.org"},  # no endpoint
    {"storage": "s3", "s3_endpoint": "https://e.example.org", "s3_bucket": "t", "download_base": "/downloads"},
    {"keep_days": 0},
])
def test_bad_storage_and_retention_settings_are_refused(tmp_path, overrides):
    with pytest.raises(ValueError):
        ServiceSettings(db_path=tmp_path / "j.sqlite3", ip_salt="s", **overrides)
