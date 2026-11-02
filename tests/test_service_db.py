import threading

import pytest

from mapcreator.job import Job
from mapcreator.service.db import STALE_ERROR, JobStore, route_hash
from mapcreator.service.settings import ServiceSettings

WP = [{"lon": 147.33, "lat": -42.81}, {"lon": 147.42, "lat": -42.74}]


def job(track_id: str = "t", lon: float = 147.33) -> Job:
    return Job.model_validate({"track_id": track_id, "waypoints": [{"lon": lon, "lat": -42.81}, WP[1]]})


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path, clock) -> JobStore:
    return JobStore(tmp_path / "jobs.sqlite3", clock=clock)


def test_settings_from_env_parses_and_validates(tmp_path):
    s = ServiceSettings.from_env({
        "MAPCREATOR_DB": str(tmp_path / "x.sqlite3"),
        "MAPCREATOR_ROUTER": "GraphHopper",
        "MAPCREATOR_GRAPHHOPPER_URL": "http://router.example.org:8989",
        "MAPCREATOR_JOBS_PER_IP_PER_DAY": "3",
        "MAPCREATOR_TRUSTED_PROXIES": "203.0.113.1, 2001:db8::/64",
        "MAPCREATOR_CLIENT_IP_HEADER": "X-Forwarded-For",
        "MAPCREATOR_WORKER_TOKEN": "",  # empty means unset
    })
    assert s.router == "graphhopper"
    assert s.jobs_per_ip_per_day == 3
    assert s.trusted_proxies == ("203.0.113.1", "2001:db8::/64")
    assert s.worker_token is None
    assert s.max_queued_jobs == 50 and s.previews_per_ip_per_minute == 30
    assert s.serves_downloads


@pytest.mark.parametrize("env", [
    {"MAPCREATOR_ROUTER": "osrm"},
    {"MAPCREATOR_ROUTER": "graphhopper"},  # needs a URL
    {"MAPCREATOR_WORKER_TOKEN": "short"},
    {"MAPCREATOR_JOBS_PER_IP_PER_DAY": "lots"},
    {"MAPCREATOR_MAX_QUEUED_JOBS": "0"},
    {"MAPCREATOR_TRUSTED_PROXIES": "not-an-address"},
    {"MAPCREATOR_CLIENT_IP_HEADER": "X-Bad\r\nHeader"},
    {"MAPCREATOR_TURNSTILE_SECRET": "secret-without-a-site-key"},
    {"MAPCREATOR_DOWNLOAD_BASE": "/api/files"},
    {"MAPCREATOR_DOWNLOAD_BASE": "ftp://files.example.org"},
    {"MAPCREATOR_PUBLIC_URL": "mapcreator.example.org"},
    {"MAPCREATOR_DB": "out/dl/jobs.sqlite3", "MAPCREATOR_STORAGE_DIR": "out/dl"},
])
def test_settings_reject_bad_values(env):
    with pytest.raises(ValueError):
        ServiceSettings.from_env(env)


def test_route_hash_ignores_the_name_only():
    assert route_hash(job("a")) == route_hash(job("b"))
    assert route_hash(job("a")) != route_hash(job("a", lon=147.34))
    wider = Job.model_validate({**job("a").model_dump(), "width": "plus_15"})
    assert route_hash(wider) != route_hash(job("a"))


def test_create_and_get(store):
    rec = store.create(job(), ip_hash="ip1")
    assert rec.status == "queued" and rec.attempts == 0 and rec.worker_id is None
    assert len(rec.id) >= 16
    assert store.get(rec.id).job["track_id"] == "t"
    assert store.get("missing") is None


def test_claim_takes_the_oldest_and_issues_a_fresh_worker_id(store, clock):
    first = store.create(job(lon=147.30), ip_hash="ip")
    clock.t += 1
    second = store.create(job(lon=147.31), ip_hash="ip")
    assert store.queue_position(store.get(second.id)) == 2
    a = store.claim_next()
    b = store.claim_next()
    assert (a.id, b.id) == (first.id, second.id)
    assert a.status == "running" and a.attempts == 1 and a.worker_id and a.worker_id != b.worker_id
    assert store.claim_next() is None
    assert store.queue_position(a) is None


def test_claims_are_atomic_across_threads(store):
    ids = {store.create(job(lon=147.30 + i / 1000), ip_hash="ip").id for i in range(20)}
    claimed: list[str] = []
    lock = threading.Lock()

    def worker():
        while (rec := store.claim_next()) is not None:
            with lock:
                claimed.append(rec.id)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == sorted(ids)


def test_only_the_current_worker_may_report(store, clock):
    store.create(job(), ip_hash="ip")
    rec = store.claim_next()
    assert not store.heartbeat(rec.id, "someone-else")
    clock.t += 30
    assert store.heartbeat(rec.id, rec.worker_id)
    assert store.get(rec.id).heartbeat_at == clock.t
    assert not store.complete(rec.id, "someone-else", {"key": "x"})
    assert store.complete(rec.id, rec.worker_id, {"key": "k", "summary": {"title": "T"}})
    done = store.get(rec.id)
    assert done.status == "done" and done.result["key"] == "k" and done.finished_at == clock.t
    # Finished jobs can't be failed or heartbeated afterwards.
    assert not store.fail(rec.id, rec.worker_id, "late")
    assert not store.heartbeat(rec.id, rec.worker_id)


def test_fail_records_the_error(store):
    store.create(job(), ip_hash="ip")
    rec = store.claim_next()
    assert store.fail(rec.id, rec.worker_id, "no road here")
    failed = store.get(rec.id)
    assert failed.status == "failed" and failed.error == "no road here"


def test_stale_jobs_are_requeued_then_failed(store, clock):
    store.create(job(), ip_hash="ip")
    first = store.claim_next()
    clock.t += 5 * 60
    assert store.requeue_stale(10 * 60) == (0, 0)  # not stale yet
    clock.t += 6 * 60
    assert store.requeue_stale(10 * 60) == (1, 0)
    requeued = store.get(first.id)
    assert requeued.status == "queued" and requeued.worker_id is None
    # The old worker has lost the job.
    assert not store.complete(first.id, first.worker_id, {"key": "x"})

    second = store.claim_next()
    assert second.id == first.id and second.attempts == 2
    clock.t += 11 * 60
    assert store.requeue_stale(10 * 60) == (0, 1)
    gone = store.get(first.id)
    assert gone.status == "failed" and gone.error == STALE_ERROR


def test_counts_and_route_lookups(store, clock):
    a = store.create(job("a"), ip_hash="ip1")
    clock.t += 3600
    store.create(job("b", lon=147.34), ip_hash="ip1")
    store.create(job("c", lon=147.35), ip_hash="ip2")
    assert store.count_queued() == 3
    assert store.count_recent_for_ip("ip1") == 2
    clock.t += 23.5 * 3600  # the first job is now more than a day old
    assert store.count_recent_for_ip("ip1") == 1

    h = route_hash(job("anything"))
    assert store.find_active_by_route_hash(h).id == a.id
    assert store.find_done_by_route_hash(h) is None
    rec = store.claim_next()
    assert rec.id == a.id
    store.complete(rec.id, rec.worker_id, {"key": "k"})
    assert store.find_done_by_route_hash(h).id == a.id
    assert store.find_active_by_route_hash(h) is None


def test_heartbeats_carry_the_stage_and_a_new_claim_clears_it(store, clock):
    store.create(job(), ip_hash="h")
    first = store.claim_next()
    assert store.heartbeat(first.id, first.worker_id, "mesh") and store.get(first.id).stage == "mesh"
    assert store.heartbeat(first.id, first.worker_id) and store.get(first.id).stage == "mesh"  # unchanged
    clock.t += 3600
    assert store.requeue_stale(60) == (1, 0) and store.get(first.id).stage is None
    second = store.claim_next()
    assert second.stage is None and not store.heartbeat(first.id, first.worker_id, "track")


def test_workers_are_remembered_when_they_call(store, clock):
    assert store.worker_last_seen() is None
    store.worker_seen()
    clock.t += 10
    assert store.worker_last_seen() == clock.t - 10
    store.worker_seen()
    assert store.worker_last_seen() == clock.t


def test_a_database_from_before_stage_reporting_is_upgraded_in_place(tmp_path, clock):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, job_json TEXT NOT NULL,
            route_hash TEXT NOT NULL, ip_hash TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, worker_id TEXT,
            created_at REAL NOT NULL, started_at REAL, heartbeat_at REAL, finished_at REAL, result_json TEXT,
            error TEXT);
    """)
    con.execute("INSERT INTO jobs (id, status, job_json, route_hash, ip_hash, created_at) "
                "VALUES ('old-job-0123456789', 'queued', ?, 'r', 'h', 1)", (job().model_dump_json(),))
    con.commit()
    con.close()
    store = JobStore(path, clock=clock)
    record = store.claim_next()
    assert record.id == "old-job-0123456789" and record.stage is None
    assert store.heartbeat(record.id, record.worker_id, "route") and store.get(record.id).stage == "route"
    JobStore(path, clock=clock)  # opening it again changes nothing


def test_simultaneous_submissions_cannot_pass_the_limits(tmp_path):
    store = JobStore(tmp_path / "race.sqlite3")

    def race(make_job, per_ip: int) -> list[str]:
        outcomes, barrier = [], threading.Barrier(20)

        def go(i: int) -> None:
            barrier.wait()
            outcomes.append(store.submit(make_job(i), ip_hash="one", max_queued=50, per_ip_per_day=per_ip)[0])

        threads = [threading.Thread(target=go, args=(i,)) for i in range(20)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        return outcomes

    # Twenty different roads at once from one client allowed one a day: exactly one gets in.
    first = race(lambda i: job(f"t{i}", lon=147.33 + i / 10000), per_ip=1)
    assert first.count("created") == 1 and first.count("ip_limit") == 19
    # The same road twenty times at once: one job, and everyone else joins it.
    second = race(lambda i: job(f"s{i}", lon=147.35), per_ip=100)
    assert second.count("created") == 1 and second.count("existing") == 19
