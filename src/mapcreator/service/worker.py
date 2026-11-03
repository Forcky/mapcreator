"""The build worker: pull a job from the web API, build it, upload the track, report back.

The worker only ever makes outbound HTTPS requests. Nothing connects to it,
so it can sit on a machine (or a container network) that accepts no inbound
traffic at all, with an egress allowlist as its only way out.

The job it receives is validated again here as a `mapcreator.job.Job` before
anything runs, so even a compromised web server can only ask for a stretch
of road. Each job runs in a fresh temporary folder that is removed after.

Errors shown to the public are chosen here: messages from exceptions that
describe the user's road (no road near a point, no connection, a bend too
tight to drive) are passed on, and anything else becomes a generic line, with
the traceback kept in the worker's own log. Paths and internals never leave
the machine. A too-tight bend also goes back as a code and its place, so the
page can mark it on the map and offer to build the road anyway.

Stopping: the first Ctrl+C or SIGTERM finishes the current job and exits;
a second one aborts at once, and the abandoned job goes back in the queue
when its heartbeat goes stale.

A build that runs past `build_timeout_s` is taken to be stuck (a hung
download, say): the worker stops heartbeating and exits, the container
restarts it, and the job is requeued once and then failed, as for a worker
that died. Without this, heartbeats would hold a hung job, and with one
worker the whole queue, for ever.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
from pydantic import ValidationError

from mapcreator import buildinfo
from mapcreator.core.profile import NoElevation
from mapcreator.job import Job
from mapcreator.pipeline.validate_stage import TightBend
from mapcreator.providers.osm.graph import RouteError
from mapcreator.regions import OutOfArea
from mapcreator.service.settings import ServiceSettings
from mapcreator.service.storage import Storage, storage_for

log = logging.getLogger(__name__)

INTERNAL_ERROR = "The build failed with an internal error. It has been logged for the maintainers."
# Longest a build may run. A 30 km road takes minutes on a slow server; an hour means something hung.
BUILD_TIMEOUT_S = 3600.0
STUCK_EXIT_CODE = 70
# Exceptions whose message is about the user's road, written for people, and safe to show.
USER_FACING = (RouteError, OutOfArea, NoElevation, TightBend)


@dataclass(frozen=True)
class BuildOutput:
    zip_path: Path
    summary: dict


# (job, work folder, on_stage): on_stage is called with each stage's name as it starts.
Pipeline = Callable[[Job, Path, Callable[[str], None]], BuildOutput]


def run_pipeline(job: Job, work_root: Path,
                 on_stage: Callable[[str], None] = lambda name: None) -> BuildOutput:
    """Every stage of `mapcreator build`, in order, writing under `work_root`.

    Data locations (cache, DEM mirror, Overpass) come from this machine's own
    settings (`mapcreator.config`); only the work folder is per job.
    """
    from mapcreator import config
    from mapcreator.cli.main import STAGES
    from mapcreator.pipeline import track_stage
    from mapcreator.pipeline.runner import run_stages
    from mapcreator.pipeline.workdir import WorkDir

    settings = replace(config.load(), work_dir=work_root)
    work = WorkDir.for_job(work_root, job)
    def started(name: str) -> None:
        log.info("stage %s", name)
        on_stage(name)

    results = run_stages(STAGES, list(STAGES), job, settings, work, on_stage=started)
    track = results["track"]
    summary = {
        "title": track["title"],
        "length_m": results["route"]["length_m"],
        "region": results["route"]["region"],
        "layouts": [{"name": v["name"], "length_m": v["length_m"]} for v in track["layouts"].values()],
        "zip_mb": track["zip_mb"],
    }
    if results.get("validate", {}).get("tight_bend"):
        summary["tight_bend"] = results["validate"]["tight_bend"]
    summary["build"] = buildinfo.version()  # which code made it: the web app reuses only a current build
    return BuildOutput(work.root / track_stage.STAGE / f"{job.track_id}.zip", summary)


def public_error(exc: BaseException) -> str:
    if isinstance(exc, USER_FACING):
        text = " ".join(str(exc).split())
        text = text[:1].upper() + text[1:]
        return text if text.endswith(".") else text + "."
    return INTERNAL_ERROR


def failure_detail(exc: BaseException) -> tuple[str | None, dict | None]:
    """A machine-readable code and detail for failures the page can do more with than show a message."""
    if isinstance(exc, TightBend):
        return "tight_bend", exc.detail()
    return None, None


class ApiError(RuntimeError):
    pass


class WorkerApi:
    """The web app's worker endpoints. Raises ApiError on anything unexpected."""

    def __init__(self, client: httpx.Client, api_base: str, token: str):
        self._client = client
        self._base = api_base.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._send_stage = True  # off if the web app predates stage reporting
        self._send_failure_detail = True  # off if the web app predates failure codes

    def _post(self, path: str, body: dict | None = None) -> httpx.Response:
        try:
            return self._client.post(f"{self._base}/api/worker{path}", json=body or {}, headers=self._headers)
        except httpx.HTTPError as e:
            raise ApiError(f"{path}: {type(e).__name__}") from e

    def _expect(self, r: httpx.Response, *ok: int) -> int:
        if r.status_code not in ok:
            raise ApiError(f"{r.request.url.path}: HTTP {r.status_code}")
        return r.status_code

    def claim(self) -> dict | None:
        r = self._post("/claim")
        return None if self._expect(r, 200, 204) == 204 else r.json()

    def heartbeat(self, job_id: str, worker_id: str, stage: str | None = None) -> bool:
        """False when the job has been taken away from this worker."""
        path = f"/jobs/{job_id}/heartbeat"
        if stage is not None and self._send_stage:
            r = self._post(path, {"worker_id": worker_id, "stage": stage})
            if r.status_code != 422:
                return self._expect(r, 200, 409) == 200
            # An older web app refuses the unknown field. Keep the job alive without it.
            log.warning("the web app doesn't take build stages; heartbeats will leave them out")
            self._send_stage = False
        return self._expect(self._post(path, {"worker_id": worker_id}), 200, 409) == 200

    def complete(self, job_id: str, worker_id: str, result: dict) -> bool:
        r = self._post(f"/jobs/{job_id}/complete", {"worker_id": worker_id, "result": result})
        return self._expect(r, 200, 409) == 200

    def fail(self, job_id: str, worker_id: str, error: str, code: str | None = None,
             detail: dict | None = None) -> bool:
        path = f"/jobs/{job_id}/fail"
        body = {"worker_id": worker_id, "error": error}
        if code is not None and self._send_failure_detail:
            r = self._post(path, {**body, "code": code, "detail": detail})
            if r.status_code != 422:
                return self._expect(r, 200, 409) == 200
            # An older web app refuses the unknown fields. Report the failure without them.
            log.warning("the web app doesn't take failure codes; failures will carry the message only")
            self._send_failure_detail = False
        return self._expect(self._post(path, body), 200, 409) == 200


class Heartbeat(threading.Thread):
    """Tells the web app the job is still alive and which stage it's on, and notices if it was taken away.

    A new stage is reported straight away rather than at the next interval, so the page keeps up.
    """

    def __init__(self, api: WorkerApi, job_id: str, worker_id: str, interval_s: float, *,
                 timeout_s: float = BUILD_TIMEOUT_S, on_timeout: Callable[[], None] | None = None):
        super().__init__(name=f"heartbeat-{job_id}", daemon=True)
        self._api = api
        self._job_id = job_id
        self._worker_id = worker_id
        self._interval_s = interval_s
        self._deadline = time.monotonic() + timeout_s
        self._on_timeout = on_timeout or _abandon_stuck_build
        self._halt = threading.Event()  # not `_stop`: Thread uses that name itself
        self._wake = threading.Event()
        self._stage: str | None = None
        self.lost = False

    def set_stage(self, name: str) -> None:
        self._stage = name
        self._wake.set()

    def run(self) -> None:
        while True:
            self._wake.wait(self._interval_s)
            self._wake.clear()
            if self._halt.is_set():
                return
            if time.monotonic() > self._deadline:
                log.error("job %s has run past the build time limit; stopping heartbeats", self._job_id)
                self.lost = True  # whatever the build produces now, it no longer reports it
                self._on_timeout()
                return
            try:
                if not self._api.heartbeat(self._job_id, self._worker_id, self._stage):
                    log.warning("job %s was taken away from this worker", self._job_id)
                    self.lost = True
                    return
            except ApiError as e:
                log.warning("heartbeat failed (%s); will retry", e)

    def halt(self) -> None:
        self._halt.set()
        self._wake.set()
        self.join(timeout=self._interval_s + 5)


def _abandon_stuck_build() -> None:
    """A stage that never returns can't be stopped from another thread, so end the process. The container
    restarts it; the job's heartbeats have stopped, so it goes back in the queue."""
    logging.shutdown()
    os._exit(STUCK_EXIT_CODE)


def run_worker(settings: ServiceSettings, api_base: str, token: str, *, pipeline: Pipeline | None = None,
               storage: Storage | None = None, client: httpx.Client | None = None,
               stop: threading.Event | None = None, max_jobs: int | None = None, exit_when_idle: bool = False,
               heartbeat_s: float = 30.0, idle_s: float = 15.0, max_backoff_s: float = 300.0,
               scratch_dir: Path | None = None, build_timeout_s: float = BUILD_TIMEOUT_S,
               on_timeout: Callable[[], None] | None = None) -> int:
    """Claim and build jobs until `stop` is set. Returns the number of jobs taken."""
    pipeline = pipeline or run_pipeline
    storage = storage or storage_for(settings)
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=False)
    api = WorkerApi(client, api_base, token)
    stop = stop or threading.Event()
    taken = 0
    backoff = idle_s
    try:
        while not stop.is_set():
            try:
                claim = api.claim()
            except ApiError as e:
                log.warning("can't reach the job API (%s); retrying in %.0f s", e, backoff)
                stop.wait(backoff)
                backoff = min(backoff * 2, max_backoff_s)
                continue
            backoff = idle_s
            if claim is None:
                if exit_when_idle:
                    break
                stop.wait(idle_s)
                continue
            taken += 1
            _build_one(claim, api, pipeline, storage, heartbeat_s, scratch_dir, build_timeout_s, on_timeout)
            if max_jobs is not None and taken >= max_jobs:
                break
    finally:
        if own_client:
            client.close()
    return taken


def _build_one(claim: dict, api: WorkerApi, pipeline: Pipeline, storage: Storage, heartbeat_s: float,
               scratch_dir: Path | None, build_timeout_s: float = BUILD_TIMEOUT_S,
               on_timeout: Callable[[], None] | None = None) -> None:
    job_id, worker_id = str(claim["id"]), str(claim["worker_id"])
    try:
        job = Job.model_validate(claim["job"])
    except (ValidationError, KeyError):
        log.error("job %s is not a valid job; refusing it", job_id)
        _report_failure(api, job_id, worker_id, "The job was not valid.")
        return

    log.info("job %s: building %s (attempt %s)", job_id, job.track_id, claim.get("attempt"))
    work = Path(tempfile.mkdtemp(prefix="mapcreator-job-", dir=scratch_dir))
    beat = Heartbeat(api, job_id, worker_id, min(heartbeat_s, build_timeout_s), timeout_s=build_timeout_s,
                     on_timeout=on_timeout)
    beat.start()
    try:
        (work / "job.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")
        out = pipeline(job, work, beat.set_stage)
        if beat.lost:
            return  # someone else has the job now; their result is the one that counts
        key = f"{job_id}/{job.track_id}.zip"
        storage.put(out.zip_path, key)
    except Exception as exc:
        if isinstance(exc, USER_FACING):
            # About the road as drawn, and the page tells the user why: not an error in the service, so it
            # stays out of the error-level alerts. Internal failures are errors, with the traceback.
            log.warning("job %s: the road can't be built as drawn: %s", job_id, public_error(exc))
        else:
            log.exception("job %s failed", job_id)
        if not beat.lost:
            _report_failure(api, job_id, worker_id, public_error(exc), *failure_detail(exc))
        return
    finally:
        beat.halt()
        shutil.rmtree(work, ignore_errors=True)
    # The track is uploaded: from here on, trouble reaching the API must not turn into a failed job.
    _report_success(api, job_id, worker_id, {"key": key, "summary": out.summary})


def _report_success(api: WorkerApi, job_id: str, worker_id: str, result: dict, *, tries: int = 3) -> None:
    for attempt in range(1, tries + 1):
        try:
            if api.complete(job_id, worker_id, result):
                log.info("job %s: done", job_id)
            else:
                log.warning("job %s: finished, but it had already been taken away", job_id)
            return
        except ApiError as e:
            log.warning("couldn't report job %s as done (%s), try %d of %d", job_id, e, attempt, tries)
            if attempt < tries:
                time.sleep(2.0 * attempt)
    log.error("job %s was built but never reported; it will be requeued when its heartbeat goes stale", job_id)


def _report_failure(api: WorkerApi, job_id: str, worker_id: str, message: str, code: str | None = None,
                    detail: dict | None = None) -> None:
    try:
        api.fail(job_id, worker_id, message, code, detail)
    except ApiError as e:
        log.warning("couldn't report the failure of job %s (%s); it will time out instead", job_id, e)


def main(argv: list[str] | None = None) -> int:
    """`mapcreator-worker`: build jobs from the web API until stopped."""
    ap = argparse.ArgumentParser(prog="mapcreator-worker", description=(
        "Pull jobs from the mapcreator web API, build them and upload the tracks. The worker token comes "
        "from MAPCREATOR_WORKER_TOKEN, never the command line."))
    ap.add_argument("--api", default=os.environ.get("MAPCREATOR_API_BASE"),
                    help="base URL of the web app (default: MAPCREATOR_API_BASE)")
    ap.add_argument("--once", action="store_true", help="build at most one job, then exit")
    ap.add_argument("--exit-when-idle", action="store_true", help="exit when the queue is empty")
    args = ap.parse_args(argv)

    settings = ServiceSettings.from_env()
    if not settings.worker_token:
        ap.error("set MAPCREATOR_WORKER_TOKEN")
    if not args.api or not args.api.startswith(("https://", "http://")):
        ap.error("give the web app's base URL with --api or MAPCREATOR_API_BASE")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stop = threading.Event()

    def on_signal(signum, frame) -> None:
        if stop.is_set():
            raise KeyboardInterrupt
        log.info("stopping after the current job; signal again to abort it")
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        run_worker(settings, args.api, settings.worker_token, stop=stop, max_jobs=1 if args.once else None,
                   exit_when_idle=args.exit_when_idle)
    except KeyboardInterrupt:
        log.warning("aborted; the current job will be requeued when its heartbeat goes stale")
        return 130
    return 0
