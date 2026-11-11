import _thread
import io
import json
import logging
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from mapcreator.job import Job, public_track_id
from mapcreator.service import worker
from mapcreator.service.api import create_app
from mapcreator.service.db import route_hash
from mapcreator.service.settings import ServiceSettings
from mapcreator.service.storage import LocalStorage
from mapcreator.service.worker import INTERNAL_ERROR, run_worker
from worker_stubs import SUMMARY, Build, end, gone_soon, records

TOKEN = "w" * 40
JOB = {"track_id": "test_road", "waypoints": [{"lon": 147.33, "lat": -42.81}, {"lon": 147.42, "lat": -42.74}]}
OTHER_JOB = {**JOB, "waypoints": [{"lon": 147.34, "lat": -42.81}, JOB["waypoints"][1]]}
WORKER_LOG = "mapcreator.service.worker"


class FakeRouter:
    def preview(self, waypoints):
        raise AssertionError("not used")


@pytest.fixture
def env(tmp_path):
    settings = ServiceSettings(db_path=tmp_path / "jobs.sqlite3", storage_dir=tmp_path / "downloads",
                               download_base="/downloads", worker_token=TOKEN, ip_salt="test-salt")
    # The worker sends its store to each build process, so the tests watch the folder rather than the object.
    storage = LocalStorage(settings.storage_dir, "/downloads")
    app = create_app(settings, router=FakeRouter(), storage=storage)
    client = TestClient(app, client=("203.0.113.7", 50000))
    return settings, client, storage, app.state.store


def work(env, pipeline, **kwargs) -> int:
    settings, client, storage, _ = env
    return run_worker(settings, "http://testserver", TOKEN, pipeline=pipeline, storage=storage, client=client,
                      exit_when_idle=True, **kwargs)


def submit(env, job=JOB) -> str:
    return env[1].post("/api/jobs", json={"job": job}).json()["id"]


def uploaded(env) -> list[str]:
    """The keys of the tracks uploaded so far."""
    root = env[2].directory
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.zip")) if root.exists() else []


def errors(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def test_worker_builds_uploads_and_reports(env, tmp_path):
    client = env[1]
    job_id = submit(env)
    record = tmp_path / "seen.jsonl"
    assert work(env, Build(record=str(record))) == 1

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "done", status
    assert status["summary"] == SUMMARY  # sent back by the build process
    tid = public_track_id("test_road", route_hash(Job.model_validate(JOB)))  # hosted builds get a stable id
    assert status["download_url"].split("?v=")[0] == f"http://testserver/downloads/{job_id}/{tid}.zip"
    download = client.get(status["download_url"])
    assert download.status_code == 200
    assert zipfile.ZipFile(io.BytesIO(download.content)).namelist() == [f"{tid}/ATTRIBUTION.txt"]
    assert uploaded(env) == [f"{job_id}/{tid}.zip"]

    [seen] = records(record)
    assert seen["job_json"]["track_id"] == tid
    assert seen["pid"] != os.getpid(), "the job is built in a process of its own"
    assert not Path(seen["work"]).exists(), "the temporary work folder is removed"


def test_the_web_app_gets_the_same_calls_from_the_worker(tmp_path):
    """The worker API isn't versioned: building in a separate process mustn't change a single call."""
    from mapcreator.pipeline.validate_stage import TightBend

    calls, claims = [], [{"id": c * 22, "worker_id": "k" * 22, "attempt": 1, "job": JOB} for c in "ab"]

    def api(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/worker/claim":
            return httpx.Response(200, json=claims.pop(0)) if claims else httpx.Response(204)
        body = json.loads(request.content)
        if not request.url.path.endswith("/heartbeat"):
            calls.append((request.url.path, body))
        return httpx.Response(200, json={"ok": True})

    tight = (3.2, 442.0, 147.438942, -42.733729, True)
    settings = ServiceSettings(worker_token=TOKEN, ip_salt="s")
    n = run_worker(settings, "https://api.example.org", TOKEN, exit_when_idle=True,
                   pipeline=Build(raises=("tight", tight), once=str(tmp_path / "once")),
                   storage=LocalStorage(tmp_path / "downloads", "/downloads"),
                   client=httpx.Client(transport=httpx.MockTransport(api)))
    assert n == 2
    expected = TightBend(*tight)
    assert calls == [
        (f"/api/worker/jobs/{'a' * 22}/fail", {"worker_id": "k" * 22, "error": worker.public_error(expected),
                                               "code": "tight_bend", "detail": expected.detail()}),
        (f"/api/worker/jobs/{'b' * 22}/complete", {"worker_id": "k" * 22, "result": {
            "key": f"{'b' * 22}/test_road.zip", "summary": SUMMARY}}),
    ]


def test_problems_with_the_road_are_shown_to_the_user(env):
    job_id = submit(env)
    work(env, Build(raises=("route", ("waypoint 2 is 900 m from the nearest drivable road",))))
    status = env[1].get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "failed"
    assert status["error"] == "Waypoint 2 is 900 m from the nearest drivable road."
    assert uploaded(env) == []


def test_road_problems_log_as_warnings_and_internal_ones_as_errors(env, caplog):
    submit(env)
    with caplog.at_level(logging.INFO, logger=WORKER_LOG):
        work(env, Build(raises=("tight", (3.2, 442.0, 147.4, -42.7, True))))
    failures = [r for r in caplog.records if "job" in r.getMessage() and r.levelno >= logging.WARNING]
    assert [r.levelname for r in failures] == ["WARNING"] and "can't be built as drawn" in failures[0].getMessage()
    assert failures[0].name == WORKER_LOG and failures[0].process != os.getpid()  # logged in the build process

    submit(env, {**JOB, "width": "plus_15"})
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=WORKER_LOG):
        work(env, Build(raises=("oserror", ("disk full",))))
    [error] = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error.name == WORKER_LOG and error.getMessage().endswith("failed")
    assert "Traceback" in error.exc_text and "OSError: disk full" in error.exc_text  # the traceback comes too


def test_internal_errors_stay_internal(env, tmp_path):
    job_id = submit(env)
    record = tmp_path / "seen.jsonl"
    work(env, Build(raises=("oserror", ("disk full under /srv/cache/dem",)), record=str(record)))
    status = env[1].get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "failed" and status["error"] == INTERNAL_ERROR
    assert not Path(records(record)[0]["work"]).exists()


def test_a_tight_bend_is_explained_marked_and_can_be_built_anyway(env):
    client = env[1]
    job_id = submit(env)
    tight = Build(raises=("tight", (3.2, 442.0, 147.438942, -42.733729, True)))

    work(env, tight)
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "failed" and "3.2 m radius, about 0.4 km from the start" in status["error"]
    assert status["failure"] == {"code": "tight_bend", "radius_m": 3.2, "km": 0.4, "lon": 147.438942,
                                 "lat": -42.733729}
    assert status["can_override"] is True
    retry = status["retry_job"]
    assert retry == {**JOB, "track_id": "test_road", "width": "as_mapped", "layouts": "both",
                     "max_length_km": 64.0, "allow_tight_bends": True}

    # Building it anyway is a new job with its own track id, and it isn't offered again.
    second = client.post("/api/jobs", json={"job": retry})
    assert second.status_code == 201 and second.json()["id"] != job_id
    work(env, tight)
    again = client.get(f"/api/jobs/{second.json()['id']}").json()
    assert again["failure"] is not None and again["can_override"] is False and again["retry_job"] is None
    first_id = env[3].get(job_id).job["track_id"]
    assert env[3].get(second.json()["id"]).job["track_id"] != first_id


def test_a_bend_below_the_floor_is_explained_but_not_offered(env):
    job_id = submit(env)
    work(env, Build(raises=("tight", (1.1, 2500.0, 147.4, -42.7, False))))
    status = env[1].get(f"/api/jobs/{job_id}").json()
    assert status["failure"]["code"] == "tight_bend" and status["can_override"] is False


def test_other_validation_failures_stay_internal(env):
    job_id = submit(env)
    work(env, Build(raises=("validation", ("main: AC_START_0 sits off the road",))))
    status = env[1].get(f"/api/jobs/{job_id}").json()
    assert status["error"] == INTERNAL_ERROR and status["failure"] is None and status["can_override"] is False


def test_a_build_process_that_crashes_is_an_internal_failure_and_the_worker_carries_on(env, tmp_path, caplog):
    client = env[1]
    crashed, after = submit(env), submit(env, OTHER_JOB)
    with caplog.at_level(logging.INFO, logger=WORKER_LOG):
        assert work(env, Build(exit_code=3, once=str(tmp_path / "once"))) == 2
    status = client.get(f"/api/jobs/{crashed}").json()
    assert status["status"] == "failed" and status["error"] == INTERNAL_ERROR and status["failure"] is None
    assert errors(caplog) == [f"job {crashed} failed: its build process exited with code 3 before sending a result"]
    assert client.get(f"/api/jobs/{after}").json()["status"] == "done"
    assert [k.split("/")[0] for k in uploaded(env)] == [after]


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX signals")
def test_a_build_process_killed_for_memory_is_an_internal_failure(env, caplog):
    job_id = submit(env)
    with caplog.at_level(logging.INFO, logger=WORKER_LOG):
        work(env, Build(kill_self=True))
    status = env[1].get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "failed" and status["error"] == INTERNAL_ERROR
    assert errors(caplog) == [f"job {job_id} failed: its build process was killed by SIGKILL (out of memory?)"]


def test_a_pipeline_that_cant_be_sent_to_the_build_process_fails_the_job_as_internal(env, caplog):
    job_id = submit(env)
    with caplog.at_level(logging.INFO, logger=WORKER_LOG):
        assert work(env, lambda job, work, on_stage: None) == 1  # a closure doesn't pickle
    status = env[1].get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "failed" and status["error"] == INTERNAL_ERROR
    assert errors(caplog) == [f"job {job_id} failed"]


def test_a_build_past_the_time_limit_is_killed_with_what_it_started_and_the_worker_carries_on(env, tmp_path, caplog):
    _, client, _, store = env
    stuck, after = submit(env), submit(env, OTHER_JOB)
    record = tmp_path / "seen.jsonl"
    hangs = Build(stages=("profile",), start_process=True, sleep_s=120, record=str(record),
                  once=str(tmp_path / "once"))  # stands in for a stage that never returns
    t0 = time.monotonic()
    with caplog.at_level(logging.INFO, logger=WORKER_LOG):
        assert work(env, hangs, heartbeat_s=0.05, build_timeout_s=6.0) == 2
    assert time.monotonic() - t0 < 60
    stuck_build = records(record)[0]
    try:
        assert gone_soon(stuck_build["pid"]), "the build process is killed"
        assert gone_soon(stuck_build["started_pid"]), "and so is the process it started"
    finally:
        end(stuck_build["started_pid"])
    job = store.get(stuck)
    assert job.status == "running" and job.stage == "profile"  # nothing reported: it goes stale and is requeued
    assert errors(caplog) == [f"job {stuck} has run past the build time limit; stopping heartbeats"]
    assert client.get(f"/api/jobs/{after}").json()["status"] == "done"
    assert [k.split("/")[0] for k in uploaded(env)] == [after]


def wait_for_record(path: Path, timeout_s: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return records(path)[0]
        except (OSError, IndexError, ValueError):
            time.sleep(0.05)
    raise TimeoutError("the build never started")


def test_aborting_the_worker_kills_its_build(env, tmp_path):
    """A second Ctrl+C or SIGTERM aborts at once: the build mustn't carry on without the worker."""
    store = env[3]
    job_id = submit(env)
    record = tmp_path / "seen.jsonl"
    threading.Thread(target=lambda: (wait_for_record(record), _thread.interrupt_main()), daemon=True).start()
    with pytest.raises(KeyboardInterrupt):
        work(env, Build(sleep_s=120, record=str(record)))
    build_pid = records(record)[0]["pid"]
    try:
        assert gone_soon(build_pid)
    finally:
        end(build_pid)
    assert store.get(job_id).status == "running"  # abandoned, to be requeued when its heartbeat goes stale


def test_a_build_ends_itself_when_its_worker_dies(tmp_path):
    record = tmp_path / "seen.jsonl"
    claim = {"id": "j" * 22, "worker_id": "k" * 22, "attempt": 1, "job": JOB}
    script = f"""
import httpx
from mapcreator.service.settings import ServiceSettings
from mapcreator.service.storage import LocalStorage
from mapcreator.service.worker import run_worker
from worker_stubs import Build

claims = [{claim!r}]

def api(request):
    if request.url.path == "/api/worker/claim":
        return httpx.Response(200, json=claims.pop()) if claims else httpx.Response(204)
    return httpx.Response(200, json={{"ok": True}})

run_worker(ServiceSettings(worker_token={TOKEN!r}, ip_salt="s"), "https://api.example.org", {TOKEN!r},
           pipeline=Build(sleep_s=120, record={str(record)!r}), exit_when_idle=True,
           storage=LocalStorage({str(tmp_path / 'downloads')!r}, "/downloads"),
           client=httpx.Client(transport=httpx.MockTransport(api)))
"""
    worker_process = subprocess.Popen([sys.executable, "-c", script],
                                      env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    build_pid = None
    try:
        build_pid = wait_for_record(record)["pid"]
        worker_process.kill()  # SIGKILL: nothing of the worker gets to clean up
        worker_process.wait(30)
        assert gone_soon(build_pid), "the build process ends itself"
    finally:
        worker_process.kill()
        if build_pid:
            end(build_pid)


def test_each_job_is_built_in_a_process_of_its_own_and_the_worker_keeps_none_of_its_memory(env, tmp_path, caplog):
    client = env[1]
    jobs = [submit(env), submit(env, OTHER_JOB)]
    record = tmp_path / "seen.jsonl"
    before = worker._memory()
    with caplog.at_level(logging.INFO, logger=WORKER_LOG):
        assert work(env, Build(hold_mb=300, record=str(record))) == 2
    after = worker._memory()

    assert [client.get(f"/api/jobs/{j}").json()["status"] for j in jobs] == ["done", "done"]
    pids = [r["pid"] for r in records(record)]
    assert len(set(pids)) == 2 and os.getpid() not in pids
    assert multiprocessing.active_children() == []
    assert not [t for t in threading.enumerate() if t.name.startswith("heartbeat-")]
    peaks = [float(m[1]) for r in caplog.records
             if (m := re.search(r"build process peak ([\d.]+) GiB", r.getMessage()))]
    if before and after:  # where the system says
        assert len(peaks) == 2 and min(peaks) >= 0.29  # each build process held its 300 MiB...
        assert after["rss"] - before["rss"] < 100 * 2**20  # ...and none of it is left in the worker's


def test_the_builds_log_records_come_out_at_the_workers_levels(env, caplog):
    logs = (("mapcreator.pipeline.stub", logging.INFO, "an info line"),
            ("mapcreator.pipeline.stub", logging.WARNING, "a warning"))
    submit(env)
    work(env, Build(logs=logs))  # the root logger's default level, WARNING
    stub = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "mapcreator.pipeline.stub"]
    assert stub == [("WARNING", "a warning")]

    caplog.clear()
    submit(env, OTHER_JOB)
    with caplog.at_level(logging.INFO, logger="mapcreator.pipeline.stub"):
        work(env, Build(logs=logs))
    stub = [r for r in caplog.records if r.name == "mapcreator.pipeline.stub"]
    assert [(r.levelname, r.getMessage()) for r in stub] == [("INFO", "an info line"), ("WARNING", "a warning")]
    assert all(r.process != os.getpid() for r in stub)


def test_the_build_process_gets_the_workers_environment_and_settings(env, tmp_path, monkeypatch):
    values = {"MAPCREATOR_SOURCE_URL": "https://example.org/mapcreator", "MAPCREATOR_NVIS": "/mirror/nvis.tif",
              "MAPCREATOR_SVTM": "/mirror/svtm.tif"}
    for k, v in values.items():
        monkeypatch.setenv(k, v)
    submit(env)
    record = tmp_path / "seen.jsonl"
    work(env, Build(record=str(record)))
    [seen] = records(record)
    assert seen["env"] == values
    assert seen["settings"] == {"source_url": "https://example.org/mapcreator", "nvis_source": "/mirror/nvis.tif",
                                "svtm_source": "/mirror/svtm.tif"}


def test_an_older_web_app_that_refuses_failure_codes_still_gets_the_failure():
    calls = []

    def api(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "code" in body:
            return httpx.Response(422, json={"detail": "extra fields not permitted"})
        return httpx.Response(200, json={"ok": True})

    from mapcreator.service.worker import WorkerApi

    worker_api = WorkerApi(httpx.Client(transport=httpx.MockTransport(api)), "https://api.example.org", TOKEN)
    assert worker_api.fail("j" * 22, "k" * 22, "Too tight.", "tight_bend", {"radius_m": 3.2})
    assert worker_api.fail("j" * 22, "k" * 22, "Too tight.", "tight_bend", {"radius_m": 3.2})
    assert calls[1] == calls[2] == {"worker_id": "k" * 22, "error": "Too tight."}  # straight to the plain form now


def test_reports_each_stage_and_the_job_page_shows_it(env, tmp_path):
    _, client, _, store = env
    job_id = submit(env)
    gate = tmp_path / "gate"
    gate.mkdir()
    builder = threading.Thread(target=work, args=(env, Build(stages=("route", "mesh"), gate=str(gate))),
                               kwargs={"heartbeat_s": 30.0})
    builder.start()
    seen = []
    try:
        for name in ("route", "mesh"):
            # A heartbeat goes out at once with the new stage. Wait for it rather than a fixed time.
            deadline = time.monotonic() + 30.0
            while True:
                stage = client.get(f"/api/jobs/{job_id}").json()["stage"]
                if (stage and stage["name"] == name) or time.monotonic() > deadline:
                    break
                time.sleep(0.02)
            seen.append(stage)
            (gate / name).touch()  # on to the next stage
    finally:
        for name in ("route", "mesh"):
            (gate / name).touch()
        builder.join(60)
    assert seen == [{"name": "route", "number": 1, "of": 7}, {"name": "mesh", "number": 4, "of": 7}]
    done = client.get(f"/api/jobs/{job_id}").json()
    assert done["status"] == "done" and done["stage"] is None and done["builder_online"] is None
    assert store.get(job_id).stage is None


def test_heartbeats_while_building(env):
    store = env[3]
    job_id = submit(env)
    work(env, Build(sleep_s=0.4), heartbeat_s=0.05)
    record = store.get(job_id)
    assert record.status == "done"
    assert record.heartbeat_at > record.started_at


def test_a_job_taken_away_is_neither_uploaded_nor_reported(env, tmp_path):
    store = env[3]
    job_id = submit(env)
    gate = tmp_path / "gate"
    gate.mkdir()
    # Long enough after the stage for a heartbeat to learn the job has gone.
    builder = threading.Thread(target=work, args=(env, Build(stages=("route",), gate=str(gate), sleep_s=0.5)),
                               kwargs={"heartbeat_s": 0.05, "max_jobs": 1})
    builder.start()
    try:
        deadline = time.monotonic() + 30.0
        while store.get(job_id).stage != "route" and time.monotonic() < deadline:
            time.sleep(0.02)
        store.clock = lambda: time.time() + 3600  # the web app decides this worker has gone quiet
        store.requeue_stale(600)
    finally:
        (gate / "route").touch()
        builder.join(60)
    record = store.get(job_id)
    assert record.status == "queued" and record.error is None
    assert uploaded(env) == []


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

    def never(job, work, on_stage):
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
    from mapcreator import buildinfo

    assert out.summary == {"title": "Test Road", "length_m": 1234.5, "region": "tas",
                           "layouts": [{"name": "Test Road (N)", "length_m": 1234.5}], "zip_mb": 1.2,
                           "build": buildinfo.version()}


def test_the_page_can_tell_whether_a_builder_is_online(env):
    _, client, _, store = env
    assert client.get("/api/status").json() == {"builder_online": False, "queued": 0, "typical_build_minutes": None}
    job_id = client.post("/api/jobs", json={"job": JOB}).json()["id"]
    assert client.get(f"/api/jobs/{job_id}").json()["builder_online"] is False
    lease = client.post("/api/worker/claim", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert client.get("/api/status").json() == {"builder_online": True, "queued": 0, "typical_build_minutes": None}
    assert client.get(f"/api/jobs/{job_id}").json()["builder_online"] is True
    bad = client.post(f"/api/worker/jobs/{job_id}/heartbeat", headers={"Authorization": f"Bearer {TOKEN}"},
                      json={"worker_id": lease["worker_id"], "stage": "rm -rf"})
    assert bad.status_code == 422
    store.clock = lambda: time.time() + 3600  # an hour of silence
    assert client.get("/api/status").json()["builder_online"] is False


def test_the_stage_names_match_the_cli():
    from mapcreator.cli.main import STAGES
    from mapcreator.pipeline import STAGE_NAMES

    assert tuple(STAGES) == STAGE_NAMES


def test_an_older_web_app_that_refuses_stages_still_gets_heartbeats():
    beats = []

    def api(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "stage" in body:
            return httpx.Response(422, json={"detail": "extra fields not permitted"})
        beats.append(body)
        return httpx.Response(200, json={"ok": True})

    from mapcreator.service.worker import WorkerApi

    worker_api = WorkerApi(httpx.Client(transport=httpx.MockTransport(api)), "https://api.example.org", TOKEN)
    assert worker_api.heartbeat("j" * 22, "k" * 22, "mesh")
    assert worker_api.heartbeat("j" * 22, "k" * 22, "track")  # straight to the plain form now
    assert beats == [{"worker_id": "k" * 22}, {"worker_id": "k" * 22}]


def test_a_queued_job_gets_a_wait_estimate_once_builds_have_been_timed(env):
    _, client, _, store = env
    auth = {"Authorization": f"Bearer {TOKEN}"}
    first = client.post("/api/jobs", json={"job": JOB}).json()["id"]
    assert client.get("/api/status").json()["typical_build_minutes"] is None
    lease = client.post("/api/worker/claim", headers=auth).json()
    record = store.get(first)
    store.clock = lambda: record.started_at + 150  # a build of two and a half minutes
    client.post(f"/api/worker/jobs/{first}/complete", headers=auth,
                json={"worker_id": lease["worker_id"], "result": {"key": f"{first}/x.zip"}})
    store.clock = lambda: record.started_at + 160
    store.worker_seen()  # a builder is online
    other = {**JOB, "waypoints": [{"lon": 147.34, "lat": -42.81}, JOB["waypoints"][1]]}
    queued = client.post("/api/jobs", json={"job": other}).json()["id"]
    assert client.get("/api/status").json()["typical_build_minutes"] == 3
    assert client.get(f"/api/jobs/{queued}").json()["estimated_wait_minutes"] == 3
