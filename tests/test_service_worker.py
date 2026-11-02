import io
import json
import threading
import time
import zipfile
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from mapcreator.providers.osm.graph import RouteError
from mapcreator.service.api import create_app
from mapcreator.service.settings import ServiceSettings
from mapcreator.service.storage import LocalStorage
from mapcreator.service.worker import INTERNAL_ERROR, BuildOutput, run_worker
from mapcreator.job import Job, public_track_id
from mapcreator.service.db import route_hash

TOKEN = "w" * 40
JOB = {"track_id": "test_road", "waypoints": [{"lon": 147.33, "lat": -42.81}, {"lon": 147.42, "lat": -42.74}]}


class FakeRouter:
    def preview(self, waypoints):
        raise AssertionError("not used")


class RecordingStorage(LocalStorage):
    def __init__(self, directory: Path):
        super().__init__(directory, "/downloads")
        self.keys: list[str] = []

    def put(self, local_path: Path, key: str) -> None:
        self.keys.append(key)
        super().put(local_path, key)


@pytest.fixture
def env(tmp_path):
    settings = ServiceSettings(db_path=tmp_path / "jobs.sqlite3", storage_dir=tmp_path / "downloads",
                               download_base="/downloads", worker_token=TOKEN, ip_salt="test-salt")
    storage = RecordingStorage(settings.storage_dir)
    app = create_app(settings, router=FakeRouter(), storage=storage)
    client = TestClient(app, client=("203.0.113.7", 50000))
    return settings, client, storage, app.state.store


def pipeline_that(action=None, seen: dict | None = None):
    """A stand-in for the build: optionally act, then write a small zip."""
    seen = {} if seen is None else seen

    def pipeline(job, work: Path) -> BuildOutput:
        seen["work"] = work
        seen["job_json"] = json.loads((work / "job.json").read_text(encoding="utf-8"))
        if action:
            action()
        zip_path = work / f"{job.track_id}.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr(f"{job.track_id}/ATTRIBUTION.txt", "test")
        return BuildOutput(zip_path, {"title": "Test Road", "length_m": 1000.0})

    return pipeline


def work(env, pipeline, **kwargs) -> int:
    settings, client, storage, _ = env
    return run_worker(settings, "http://testserver", TOKEN, pipeline=pipeline, storage=storage, client=client,
                      exit_when_idle=True, **kwargs)


def submit(env) -> str:
    return env[1].post("/api/jobs", json={"job": JOB}).json()["id"]


def test_worker_builds_uploads_and_reports(env):
    client = env[1]
    job_id = submit(env)
    seen = {}
    assert work(env, pipeline_that(seen=seen)) == 1

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "done", status
    assert status["summary"] == {"title": "Test Road", "length_m": 1000.0}
    tid = public_track_id("test_road", route_hash(Job.model_validate(JOB)))  # hosted builds get a stable id
    assert status["download_url"] == f"http://testserver/downloads/{job_id}/{tid}.zip"
    download = client.get(status["download_url"])
    assert download.status_code == 200
    assert zipfile.ZipFile(io.BytesIO(download.content)).namelist() == [f"{tid}/ATTRIBUTION.txt"]

    assert seen["job_json"]["track_id"] == tid
    assert not seen["work"].exists(), "the temporary work folder is removed"


def test_problems_with_the_road_are_shown_to_the_user(env):
    job_id = submit(env)

    def no_road():
        raise RouteError("waypoint 2 is 900 m from the nearest drivable road")

    work(env, pipeline_that(no_road))
    status = env[1].get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "failed"
    assert status["error"] == "Waypoint 2 is 900 m from the nearest drivable road."
    assert env[2].keys == []


def test_internal_errors_stay_internal(env):
    job_id = submit(env)
    seen = {}

    def broken():
        raise OSError("disk full under /srv/cache/dem")

    work(env, pipeline_that(broken, seen))
    status = env[1].get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "failed" and status["error"] == INTERNAL_ERROR
    assert not seen["work"].exists()


def test_heartbeats_while_building(env):
    store = env[3]
    job_id = submit(env)
    work(env, pipeline_that(lambda: time.sleep(0.4)), heartbeat_s=0.05)
    record = store.get(job_id)
    assert record.status == "done"
    assert record.heartbeat_at > record.started_at


def test_a_job_taken_away_is_neither_uploaded_nor_reported(env):
    store = env[3]
    job_id = submit(env)

    def requeued_meanwhile():
        store.clock = lambda: time.time() + 3600  # the web app decides this worker has gone quiet
        store.requeue_stale(600)
        time.sleep(0.3)  # long enough for a heartbeat to learn of it

    work(env, pipeline_that(requeued_meanwhile), heartbeat_s=0.05, max_jobs=1)
    record = store.get(job_id)
    assert record.status == "queued" and record.error is None
    assert env[2].keys == []


def test_jobs_are_validated_again_on_the_worker():
    """Even a compromised web app can only hand the worker a stretch of road."""
    reports = []

    def api(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        if request.url.path == "/api/worker/claim":
            if reports:
                return httpx.Response(204)
            return httpx.Response(200, json={"id": "j" * 22, "worker_id": "k" * 22, "attempt": 1,
                                             "job": {**JOB, "dem_url": "http://example.org/evil.tif"}})
        reports.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"ok": True})

    def never(job, work):
        raise AssertionError("the pipeline must not run")

    settings = ServiceSettings(worker_token=TOKEN, ip_salt="s")
    n = run_worker(settings, "https://api.example.org", TOKEN, pipeline=never, exit_when_idle=True,
                   client=httpx.Client(transport=httpx.MockTransport(api)))
    assert n == 1
    assert reports == [(f"/api/worker/jobs/{'j' * 22}/fail", {"worker_id": "k" * 22, "error": "The job was not valid."})]


def test_backs_off_while_the_api_is_unreachable():
    calls = []
    stop = threading.Event()

    def down(request):
        calls.append(time.monotonic())
        if len(calls) >= 3:
            stop.set()
        raise httpx.ConnectError("refused")

    settings = ServiceSettings(worker_token=TOKEN, ip_salt="s")
    n = run_worker(settings, "https://api.example.org", TOKEN, client=httpx.Client(transport=httpx.MockTransport(down)),
                   stop=stop, idle_s=0.02, max_backoff_s=1.0)
    assert n == 0 and len(calls) == 3
    assert calls[2] - calls[1] > calls[1] - calls[0]  # waits grow


def test_run_pipeline_runs_every_stage_in_order(monkeypatch, tmp_path):
    from mapcreator.cli import main as cli
    from mapcreator.job import Job
    from mapcreator.service.worker import run_pipeline

    ran = []
    canned = {
        "route": {"length_m": 1234.5, "region": "tas"},
        "track": {"title": "Test Road", "zip_mb": 1.2,
                  "layouts": {"fwd": {"name": "Test Road (N)", "kn5": "x.kn5", "length_m": 1234.5}}},
    }

    def stage(name):
        def run(job, settings, work):
            ran.append((name, settings.work_dir, work.root))
            return canned.get(name, {})
        return run

    monkeypatch.setattr(cli, "STAGES", {name: stage(name) for name in cli.STAGES})
    out = run_pipeline(Job.model_validate(JOB), tmp_path)
    assert [r[0] for r in ran] == ["route", "profile", "section", "mesh", "scenery", "track", "validate"]
    assert all(r[1] == tmp_path and r[2] == tmp_path / "test_road" for r in ran)
    assert out.zip_path == tmp_path / "test_road" / "05_track" / "test_road.zip"
    assert out.summary == {"title": "Test Road", "length_m": 1234.5, "region": "tas",
                           "layouts": [{"name": "Test Road (N)", "length_m": 1234.5}], "zip_mb": 1.2}
