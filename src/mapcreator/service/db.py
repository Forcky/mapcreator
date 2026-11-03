"""SQLite job store: the queue between the public web front end and the build workers.

One small table, because the queue is small: tens of jobs, built one or two
at a time by workers that poll over HTTPS. SQLite in WAL mode lets the web
process answer status polls while a claim is being written, and every change
that must be atomic (claiming, requeueing) runs inside `BEGIN IMMEDIATE`, so
two workers can never take the same job.

The worker also reports which build stage it's on with each heartbeat, and
any authenticated worker request records when a worker was last heard from,
so the page can say whether jobs are being built at all.

Each claim gets a fresh random `worker_id`. A worker must present it to
heartbeat, complete or fail the job, so a worker that went quiet and had its
job handed to someone else can't overwrite the new attempt's result.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from mapcreator.job import Job

STATUSES = ("queued", "running", "done", "failed")
# A job is tried this many times before a silent worker gets it marked failed.
MAX_ATTEMPTS = 2
STALE_ERROR = "The build stopped responding and was abandoned."

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
    job_json TEXT NOT NULL,
    route_hash TEXT NOT NULL,
    ip_hash TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    worker_id TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    heartbeat_at REAL,
    finished_at REAL,
    result_json TEXT,
    error TEXT,
    stage TEXT,
    failure_json TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);
CREATE INDEX IF NOT EXISTS jobs_route_hash ON jobs (route_hash, status);
CREATE INDEX IF NOT EXISTS jobs_ip_created ON jobs (ip_hash, created_at);
"""


def new_job_id() -> str:
    """A random job id that never starts with "-": one in 64 of token_urlsafe's would, and a command line
    then reads the id as an option (mapcreator-admin show -abc... fails)."""
    while True:
        job_id = secrets.token_urlsafe(16)
        if not job_id.startswith("-"):
            return job_id


# Job fields added later, left out of hashes and stored jobs while they hold their default. Hashes of
# earlier jobs then stay the same (a rebuild keeps its track id), and a worker that predates a field
# (its Job forbids unknown fields) can still build every job that doesn't use it.
LATER_FIELDS = {"allow_tight_bends": False}
# A job's length limit decides whether the road builds, not what it builds, so it isn't part of the road's
# identity. Hashes carry the 30 km every job had before the limit was raised, so earlier ids stay the same.
HASHED_MAX_LENGTH_KM = 30.0


def job_dict(job: Job) -> dict:
    d = job.model_dump(mode="json")
    for name, default in LATER_FIELDS.items():
        if d.get(name) == default:
            d.pop(name, None)
    return d


def job_json(job: Job) -> str:
    return json.dumps(job_dict(job))


def route_hash(job: Job) -> str:
    """Identity of the road a job builds: the canonical job without its name.

    Two people asking for the same road with the same options get the same
    track, so the second request can reuse the first one's build. Asking for
    a tight bend to be built anyway is a different option, so a different track.
    """
    d = job_dict(job)
    d.pop("track_id", None)
    d["max_length_km"] = HASHED_MAX_LENGTH_KM
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class JobRecord:
    id: str
    status: str
    job: dict
    route_hash: str
    ip_hash: str
    attempts: int
    worker_id: str | None
    created_at: float
    started_at: float | None
    heartbeat_at: float | None
    finished_at: float | None
    result: dict | None
    error: str | None
    stage: str | None = None
    failure: dict | None = None  # code and detail of a failure the page can act on (a tight bend)

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> JobRecord:
        return cls(
            id=row["id"], status=row["status"], job=json.loads(row["job_json"]), route_hash=row["route_hash"],
            ip_hash=row["ip_hash"], attempts=row["attempts"], worker_id=row["worker_id"],
            created_at=row["created_at"], started_at=row["started_at"], heartbeat_at=row["heartbeat_at"],
            finished_at=row["finished_at"],
            result=json.loads(row["result_json"]) if row["result_json"] else None, error=row["error"],
            stage=row["stage"],
            failure=json.loads(row["failure_json"]) if row["failure_json"] else None,
        )


class JobStore:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time, busy_timeout_s: float = 5.0):
        self.path = Path(path)
        self.clock = clock
        self.busy_timeout_s = busy_timeout_s
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute("PRAGMA journal_mode = WAL")
            con.executescript(SCHEMA)
            # Databases from before stage reporting and failure codes. ADD COLUMN is cheap and keeps every row.
            have = {r["name"] for r in con.execute("PRAGMA table_info(jobs)")}
            for column in ("stage", "failure_json"):
                if column in have:
                    continue
                try:
                    con.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e):
                        raise  # another process added it between our check and the ALTER: fine

    # A connection per call: cheap for SQLite, and safe from FastAPI's thread pool.
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=self.busy_timeout_s, isolation_level=None)
        try:
            con.row_factory = sqlite3.Row
            con.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_s * 1000)}")
            con.execute("PRAGMA synchronous = NORMAL")
            yield con
        finally:
            con.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
            except BaseException:
                con.execute("ROLLBACK")
                raise
            con.execute("COMMIT")

    def _one(self, sql: str, params: tuple = ()) -> JobRecord | None:
        with self._connect() as con:
            row = con.execute(sql, params).fetchone()
        return JobRecord._from_row(row) if row else None

    def create(self, job: Job, *, ip_hash: str) -> JobRecord:
        job_id = new_job_id()
        with self._connect() as con:
            con.execute(
                "INSERT INTO jobs (id, status, job_json, route_hash, ip_hash, created_at) "
                "VALUES (?, 'queued', ?, ?, ?, ?)",
                (job_id, job_json(job), route_hash(job), ip_hash, self.clock()),
            )
        return self.get(job_id)

    def submit(self, job: Job, *, ip_hash: str, max_queued: int, per_ip_per_day: int) -> tuple[str, JobRecord | None]:
        """Admit a public job, atomically, so simultaneous submissions can't slip past the limits.

        Returns ("existing", the job already done or under way for the same road), ("queue_full", None),
        ("ip_limit", None) or ("created", the new job), checked in that order.
        """
        same = route_hash(job)
        with self._transaction() as con:
            row = (con.execute("SELECT * FROM jobs WHERE route_hash = ? AND status = 'done' "
                               "ORDER BY finished_at DESC LIMIT 1", (same,)).fetchone()
                   or con.execute("SELECT * FROM jobs WHERE route_hash = ? AND status IN ('queued', 'running') "
                                  "ORDER BY created_at LIMIT 1", (same,)).fetchone())
            if row is not None:
                return "existing", JobRecord._from_row(row)
            if con.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()[0] >= max_queued:
                return "queue_full", None
            now = self.clock()
            recent = con.execute("SELECT COUNT(*) FROM jobs WHERE ip_hash = ? AND created_at >= ?",
                                 (ip_hash, now - 86400.0)).fetchone()[0]
            if recent >= per_ip_per_day:
                return "ip_limit", None
            job_id = new_job_id()
            con.execute("INSERT INTO jobs (id, status, job_json, route_hash, ip_hash, created_at) "
                        "VALUES (?, 'queued', ?, ?, ?, ?)", (job_id, job_json(job), same, ip_hash, now))
            created = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return "created", JobRecord._from_row(created)

    def get(self, job_id: str) -> JobRecord | None:
        return self._one("SELECT * FROM jobs WHERE id = ?", (job_id,))

    def claim_next(self) -> JobRecord | None:
        """Hand the oldest queued job to a worker, under a new `worker_id`."""
        with self._transaction() as con:
            row = con.execute(
                "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, rowid LIMIT 1").fetchone()
            if row is None:
                return None
            now = self.clock()
            con.execute(
                "UPDATE jobs SET status = 'running', worker_id = ?, attempts = attempts + 1, started_at = ?, "
                "heartbeat_at = ?, stage = NULL WHERE id = ?",
                (secrets.token_urlsafe(16), now, now, row["id"]),
            )
            claimed = con.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
        return JobRecord._from_row(claimed)

    def _update_running(self, job_id: str, worker_id: str, sets: str, params: tuple) -> bool:
        with self._connect() as con:
            cur = con.execute(
                f"UPDATE jobs SET {sets} WHERE id = ? AND status = 'running' AND worker_id = ?",
                (*params, job_id, worker_id),
            )
        return cur.rowcount == 1

    def heartbeat(self, job_id: str, worker_id: str, stage: str | None = None) -> bool:
        """False when this worker no longer holds the job (it was requeued, or finished)."""
        if stage is None:
            return self._update_running(job_id, worker_id, "heartbeat_at = ?", (self.clock(),))
        return self._update_running(job_id, worker_id, "heartbeat_at = ?, stage = ?", (self.clock(), stage))

    def complete(self, job_id: str, worker_id: str, result: dict) -> bool:
        now = self.clock()
        return self._update_running(job_id, worker_id,
                                    "status = 'done', heartbeat_at = ?, finished_at = ?, result_json = ?, stage = NULL",
                                    (now, now, json.dumps(result)))

    def fail(self, job_id: str, worker_id: str, error: str, failure: dict | None = None) -> bool:
        now = self.clock()
        return self._update_running(
            job_id, worker_id,
            "status = 'failed', heartbeat_at = ?, finished_at = ?, error = ?, stage = NULL, failure_json = ?",
            (now, now, error, json.dumps(failure) if failure else None))

    def requeue_stale(self, stale_after_s: float, *, max_attempts: int = MAX_ATTEMPTS) -> tuple[int, int]:
        """Take running jobs back from workers that stopped heartbeating.

        A job goes back in the queue if it has attempts left, and is failed
        otherwise, so a road that reliably kills workers can't take them all
        down one after another. Returns (requeued, failed).
        """
        now = self.clock()
        cutoff = now - stale_after_s
        with self._transaction() as con:
            failed = con.execute(
                "UPDATE jobs SET status = 'failed', worker_id = NULL, finished_at = ?, error = ?, stage = NULL "
                "WHERE status = 'running' AND heartbeat_at < ? AND attempts >= ?",
                (now, STALE_ERROR, cutoff, max_attempts),
            ).rowcount
            requeued = con.execute(
                "UPDATE jobs SET status = 'queued', worker_id = NULL, started_at = NULL, heartbeat_at = NULL, "
                "stage = NULL "
                "WHERE status = 'running' AND heartbeat_at < ?",
                (cutoff,),
            ).rowcount
        return requeued, failed

    def prune(self, keep_s: float, ip_hash_s: float) -> list[dict]:
        """Forget old work. Returns the results of deleted jobs so their downloads can go too.

        Finished jobs (done or failed) older than `keep_s` are deleted. Every
        job's client-IP hash is blanked once it's older than `ip_hash_s`: it's
        only kept for the daily per-IP limit, so it shouldn't outlive that.
        """
        now = self.clock()
        with self._transaction() as con:
            old = con.execute(
                "SELECT result_json FROM jobs WHERE status IN ('done', 'failed') AND finished_at < ?",
                (now - keep_s,),
            ).fetchall()
            con.execute("DELETE FROM jobs WHERE status IN ('done', 'failed') AND finished_at < ?", (now - keep_s,))
            con.execute("UPDATE jobs SET ip_hash = '' WHERE ip_hash != '' AND created_at < ?", (now - ip_hash_s,))
        return [json.loads(r["result_json"]) for r in old if r["result_json"]]

    # ----- operator commands (mapcreator-admin) -----

    def list_jobs(self, status: str | None = None, limit: int = 50) -> list[JobRecord]:
        """Newest first."""
        if status is not None and status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        where, params = ("WHERE status = ?", (status,)) if status else ("", ())
        with self._connect() as con:
            rows = con.execute(f"SELECT * FROM jobs {where} ORDER BY created_at DESC, rowid DESC LIMIT ?",
                               (*params, limit)).fetchall()
        return [JobRecord._from_row(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._connect() as con:
            rows = con.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        return {s: 0 for s in STATUSES} | {r["status"]: r["n"] for r in rows}

    def requeue(self, job_id: str, *, done_too: bool = False) -> bool:
        """Put a failed or running job back in the queue with its attempts reset. A running job's worker
        loses it (its next heartbeat gets 409).

        With `done_too`, a finished job is built again in place, for a fix that changes what the build makes:
        the same id and download address. Its result stays until the new build replaces it, so if the rebuild
        fails, retention still knows which download to delete.
        """
        statuses = ("failed", "running", "done") if done_too else ("failed", "running")
        with self._connect() as con:
            cur = con.execute(
                "UPDATE jobs SET status = 'queued', worker_id = NULL, attempts = 0, started_at = NULL, "
                "heartbeat_at = NULL, finished_at = NULL, error = NULL, stage = NULL, failure_json = NULL "
                f"WHERE id = ? AND status IN ({', '.join('?' * len(statuses))})", (job_id, *statuses))
        return cur.rowcount == 1

    def cancel(self, job_id: str, reason: str) -> bool:
        """Fail a queued or running job by hand, with a reason the page shows."""
        with self._connect() as con:
            cur = con.execute(
                "UPDATE jobs SET status = 'failed', worker_id = NULL, finished_at = ?, error = ?, stage = NULL "
                "WHERE id = ? AND status IN ('queued', 'running')", (self.clock(), reason, job_id))
        return cur.rowcount == 1

    def delete(self, job_id: str) -> dict | None:
        """Forget a job. Returns its result (so its download can be removed), or None if there was no job."""
        with self._transaction() as con:
            row = con.execute("SELECT result_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            con.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return json.loads(row["result_json"]) if row["result_json"] else {}

    def worker_seen(self) -> None:
        """Record that a worker just made an authenticated request."""
        with self._connect() as con:
            con.execute("INSERT INTO meta (key, value) VALUES ('worker_seen', ?) "
                        "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (self.clock(),))

    def worker_last_seen(self) -> float | None:
        with self._connect() as con:
            row = con.execute("SELECT value FROM meta WHERE key = 'worker_seen'").fetchone()
        return row["value"] if row else None

    def typical_build_s(self, recent: int = 20) -> float | None:
        """Median build time of the most recent finished builds, or None before there are any."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT finished_at - started_at AS took FROM jobs WHERE status = 'done' AND started_at IS NOT NULL "
                "ORDER BY finished_at DESC LIMIT ?", (recent,)).fetchall()
        took = sorted(r["took"] for r in rows if r["took"] is not None and r["took"] >= 0)
        return took[len(took) // 2] if took else None

    def count_queued(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()[0]

    def queue_position(self, record: JobRecord) -> int | None:
        """1 for the next job a worker will take; None if the job isn't queued."""
        if record.status != "queued":
            return None
        with self._connect() as con:
            ahead = con.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND created_at < ?", (record.created_at,)
            ).fetchone()[0]
        return ahead + 1

    def count_recent_for_ip(self, ip_hash: str, *, window_s: float = 86400.0) -> int:
        with self._connect() as con:
            return con.execute(
                "SELECT COUNT(*) FROM jobs WHERE ip_hash = ? AND created_at >= ?",
                (ip_hash, self.clock() - window_s),
            ).fetchone()[0]

    def find_done_by_route_hash(self, route_hash: str) -> JobRecord | None:
        return self._one(
            "SELECT * FROM jobs WHERE route_hash = ? AND status = 'done' ORDER BY finished_at DESC LIMIT 1",
            (route_hash,),
        )

    def find_active_by_route_hash(self, route_hash: str) -> JobRecord | None:
        """A queued or running job for the same road, so a repeat request joins it."""
        return self._one(
            "SELECT * FROM jobs WHERE route_hash = ? AND status IN ('queued', 'running') "
            "ORDER BY created_at LIMIT 1",
            (route_hash,),
        )
