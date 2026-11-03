"""`mapcreator-admin`: look at and fix the job queue by hand.

Run it where the web app's database is, with the web app's settings, for
example inside its container:

    docker exec <web container> mapcreator-admin stats
    docker exec <web container> mapcreator-admin list --status failed
    docker exec <web container> mapcreator-admin requeue <job id>
    docker exec <web container> mapcreator-admin requeue --done <job id>   # rebuild a finished track in place

It works on the database directly, so it needs no token and no network. It
never prints client-address hashes. New job ids never start with "-"; for an
older one that does, put `--` before it: `mapcreator-admin show -- -abc...`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime

from mapcreator.service.db import STATUSES, JobRecord, JobStore
from mapcreator.service.settings import ServiceSettings
from mapcreator.service.storage import Storage, storage_for

CANCELLED = "The build was cancelled by the maintainers."


def _when(t: float | None) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%d %H:%M:%SZ") if t else "-"


def _row(r: JobRecord) -> str:
    track = r.job.get("track_id", "?")
    extra = r.stage or (r.error or "")[:60]
    return f"{r.id}  {r.status:<7}  {_when(r.created_at)}  try {r.attempts}  {track:<32}  {extra}"


def _public(r: JobRecord) -> dict:
    return {"id": r.id, "status": r.status, "job": r.job, "attempts": r.attempts, "stage": r.stage,
            "created_at": _when(r.created_at), "started_at": _when(r.started_at),
            "heartbeat_at": _when(r.heartbeat_at), "finished_at": _when(r.finished_at),
            "result": r.result, "error": r.error}


def main(argv: list[str] | None = None, *, store: JobStore | None = None, storage: Storage | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mapcreator-admin", description="Inspect and fix the mapcreator job queue.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("stats", help="jobs per status, and when a worker was last heard from")
    ls = sub.add_parser("list", help="the newest jobs")
    ls.add_argument("--status", choices=STATUSES)
    ls.add_argument("--limit", type=int, default=30)
    show = sub.add_parser("show", help="one job in full (no client data)")
    show.add_argument("id")
    rq = sub.add_parser("requeue", help="put a failed or running job back in the queue, attempts reset")
    rq.add_argument("id")
    rq.add_argument("--done", action="store_true",
                    help="a finished job too: build it again in place (same id and download address), for a fix "
                         "that changes what the build makes")
    cancel = sub.add_parser("fail", help="fail a queued or running job by hand")
    cancel.add_argument("id")
    cancel.add_argument("--reason", default=CANCELLED, help="shown on the job page (default: %(default)s)")
    rm = sub.add_parser("delete", help="forget a job and delete its download (with public storage, the bucket's "
                                        "lifecycle rule removes it)")
    rm.add_argument("id")
    args = ap.parse_args(argv)

    settings = ServiceSettings.from_env() if store is None or storage is None else None
    store = store or JobStore(settings.db_path)
    storage = storage or storage_for(settings)
    if args.command == "stats":
        seen = store.worker_last_seen()
        print(json.dumps({"jobs": store.counts(),
                          "worker_last_seen": _when(seen),
                          "worker_seconds_ago": round(time.time() - seen) if seen else None}, indent=2))
        return 0
    if args.command == "list":
        for r in store.list_jobs(args.status, args.limit):
            print(_row(r))
        return 0
    record = store.get(args.id)
    if record is None:
        print(f"no job {args.id}", file=sys.stderr)
        return 1
    if args.command == "show":
        print(json.dumps(_public(record), indent=2))
        return 0
    if args.command == "requeue":
        done = store.requeue(args.id, done_too=args.done)
        if done and record.status == "done" and record.result and record.result.get("key"):
            # The new zip replaces the old one at the same address, and a CDN may hold the old one for hours.
            print(f"rebuilding in place: if a CDN caches downloads, purge {storage.url(record.result['key'])} "
                  "once the job is done")
    elif args.command == "fail":
        done = store.cancel(args.id, " ".join(args.reason.split())[:500])
    else:
        result = store.delete(args.id)
        done = result is not None
        if result and result.get("key"):
            try:
                storage.delete(result["key"])  # a no-op with public storage: the lifecycle rule does it
            except Exception as e:  # noqa: BLE001 - the job is gone either way; say what's left behind
                print(f"the job is deleted, but its download {result['key']} couldn't be: {e}", file=sys.stderr)
    if not done:
        print(f"job {args.id} is {record.status}: nothing to do", file=sys.stderr)
        return 1
    print(f"{args.command}: {args.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
