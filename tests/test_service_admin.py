"""mapcreator-admin: the operator's view of the queue."""

import json

import pytest

from mapcreator.job import Job
from mapcreator.service import admin
from mapcreator.service.db import JobStore

WP = [{"lon": 147.33, "lat": -42.81}, {"lon": 147.42, "lat": -42.74}]


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs.sqlite3")


def make(store, n: int):
    return [store.create(Job.model_validate({"track_id": f"road_{i}", "waypoints": [
        {"lon": 147.33 + i / 1000, "lat": -42.81}, WP[1]]}), ip_hash="secret-hash") for i in range(n)]


def test_stats_list_and_show_never_print_client_hashes(store, capsys):
    jobs = make(store, 3)
    assert admin.main(["stats"], store=store) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["jobs"] == {"queued": 3, "running": 0, "done": 0, "failed": 0} and stats["worker_last_seen"] == "-"
    assert admin.main(["list"], store=store) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and lines[0].startswith(jobs[-1].id)  # newest first
    assert admin.main(["show", jobs[0].id], store=store) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["job"]["track_id"] == "road_0" and "secret-hash" not in out


def test_fail_requeue_and_delete(store, capsys):
    job = make(store, 1)[0]
    assert admin.main(["fail", job.id, "--reason", "Not  a\nreal road."], store=store) == 0
    assert store.get(job.id).status == "failed" and store.get(job.id).error == "Not a real road."
    assert admin.main(["fail", job.id], store=store) == 1  # already failed: nothing to do
    assert admin.main(["requeue", job.id], store=store) == 0
    again = store.get(job.id)
    assert again.status == "queued" and again.attempts == 0 and again.error is None
    claimed = store.claim_next()
    assert admin.main(["requeue", job.id], store=store) == 0  # a running job comes back too...
    assert not store.heartbeat(job.id, claimed.worker_id)  # ...and its old worker has lost it
    assert admin.main(["delete", job.id], store=store) == 0 and store.get(job.id) is None
    assert admin.main(["show", job.id], store=store) == 1
