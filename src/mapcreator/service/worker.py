"""The build worker: pull a job from the web API, build it, upload the track, report back.

The worker only ever makes outbound HTTPS requests. Nothing connects to it,
so it can sit on a machine (or a container network) that accepts no inbound
traffic at all, with an egress allowlist as its only way out.

The job it receives is validated again here as a `mapcreator.job.Job` before
anything runs, so even a compromised web server can only ask for a stretch
of road. Each job runs in a fresh temporary folder that is removed after.

Each job is also built in a process of its own, started fresh (`spawn`) for
that job and gone when it ends. A long road's build allocates gigabytes, and a
long-lived Python process doesn't hand freed memory back to the system, so
when builds ran in the worker's own process its memory crept up with every
long track. Now the worker process only claims jobs, heartbeats and reports,
and stays small. The build process builds and uploads, and sends back over a
pipe the stage it's on, its log records (logged here as the worker's own, at
their own levels) and a small result: the summary and the uploaded key, or
why it failed. Before uploading it asks whether the job is still this
worker's. The web app sees no difference.

Errors shown to the public are chosen here: messages from exceptions that
describe the user's road (no road near a point, no connection, a bend too
tight to drive) are passed on, and anything else becomes a generic line, with
the traceback kept in the worker's own log. Paths and internals never leave
the machine. A too-tight bend also goes back as a code and its place, so the
page can mark it on the map and offer to build the road anyway. A build
process that dies without a result (it crashed, or the kernel killed it for
memory) is an internal failure, and the worker carries on with the next job.

Stopping: the first Ctrl+C or SIGTERM finishes the current job and exits;
a second one aborts at once, killing the build, and the abandoned job goes
back in the queue when its heartbeat goes stale.

A build that runs past `build_timeout_s` is taken to be stuck (a hung
download, say): the worker stops heartbeating, kills the build process and
anything it started, reports nothing, and carries on. The job goes stale and
is requeued once and then failed, as for a worker that died. Without this,
heartbeats would hold a hung job, and with one worker the whole queue, for
ever. Only if the killed process won't go does the worker exit instead, for
the container to restart it.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from multiprocessing.connection import wait as wait_for
from pathlib import Path

import httpx
from pydantic import ValidationError

from mapcreator import buildinfo
from mapcreator.job import Job
from mapcreator.service.settings import ServiceSettings
from mapcreator.service.storage import Storage, storage_for

log = logging.getLogger(__name__)

INTERNAL_ERROR = "The build failed with an internal error. It has been logged for the maintainers."
# Longest a build may run. A 30 km road takes minutes on a slow server; an hour means something hung.
BUILD_TIMEOUT_S = 3600.0
STUCK_EXIT_CODE = 70
# How long a killed build process may take to go before the worker gives up on it and exits instead.
KILL_WAIT_S = 30.0
# How long a build process may take to exit once it has sent its result.
EXIT_WAIT_S = 30.0
# How often the worker looks up from its build process to check the time limit.
POLL_S = 0.5
# A build process whose worker process has gone ends itself with this code.
ORPHANED_EXIT_CODE = 71

_SPAWN = multiprocessing.get_context("spawn")


def user_facing() -> tuple[type[Exception], ...]:
    """Exceptions whose message is about the user's road, written for people, and safe to show.

    Imported when first asked for: the build process needs them, and the worker process, which only hands
    jobs out, stays free of the pipeline's libraries."""
    from mapcreator.core.profile import NoElevation
    from mapcreator.pipeline.validate_stage import TightBend
    from mapcreator.providers.osm.graph import RouteError
    from mapcreator.regions import OutOfArea

    return RouteError, OutOfArea, NoElevation, TightBend


@dataclass(frozen=True)
class BuildOutput:
    zip_path: Path
    summary: dict


# (job, work folder, on_stage): on_stage is called with each stage's name as it starts. It runs in the build
# process, so the worker sends it there by pickling: a module-level function or class instance, not a closure.
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
    if isinstance(exc, user_facing()):
        text = " ".join(str(exc).split())
        text = text[:1].upper() + text[1:]
        return text if text.endswith(".") else text + "."
    return INTERNAL_ERROR


def failure_detail(exc: BaseException) -> tuple[str | None, dict | None]:
    """A machine-readable code and detail for failures the page can do more with than show a message."""
    from mapcreator.pipeline.validate_stage import TightBend

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
    """The last resort for a stuck build that can't be stopped (its process won't die when killed): end the
    worker. The container restarts it; the job's heartbeats have stopped, so it goes back in the queue."""
    logging.shutdown()
    os._exit(STUCK_EXIT_CODE)


def run_worker(settings: ServiceSettings, api_base: str, token: str, *, pipeline: Pipeline | None = None,
               storage: Storage | None = None, client: httpx.Client | None = None,
               stop: threading.Event | None = None, max_jobs: int | None = None, exit_when_idle: bool = False,
               heartbeat_s: float = 30.0, idle_s: float = 15.0, max_backoff_s: float = 300.0,
               scratch_dir: Path | None = None, build_timeout_s: float = BUILD_TIMEOUT_S) -> int:
    """Claim and build jobs until `stop` is set. Returns the number of jobs taken.

    Each job is built in a fresh process, which is sent `pipeline` and `storage` by pickling. Without
    `storage`, each build process sets up the store from `settings` (an S3 client can't be sent)."""
    pipeline = pipeline or run_pipeline
    if storage is None:
        storage_for(settings)  # a store that can't be set up stops the worker here, at start-up, not every job
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
            _build_one(claim, api, pipeline, storage, settings, heartbeat_s, scratch_dir, build_timeout_s)
            if max_jobs is not None and taken >= max_jobs:
                break
    finally:
        if own_client:
            client.close()
    return taken


def _build_one(claim: dict, api: WorkerApi, pipeline: Pipeline, storage: Storage | None,
               settings: ServiceSettings, heartbeat_s: float, scratch_dir: Path | None,
               build_timeout_s: float = BUILD_TIMEOUT_S) -> None:
    job_id, worker_id = str(claim["id"]), str(claim["worker_id"])
    try:
        job = Job.model_validate(claim["job"])
    except (ValidationError, KeyError):
        log.error("job %s is not a valid job; refusing it", job_id)
        _report_failure(api, job_id, worker_id, "The job was not valid.")
        return

    log.info("job %s: building %s (attempt %s)", job_id, job.track_id, claim.get("attempt"))
    work = Path(tempfile.mkdtemp(prefix="mapcreator-job-", dir=scratch_dir))
    timed_out = threading.Event()
    beat = Heartbeat(api, job_id, worker_id, min(heartbeat_s, build_timeout_s), timeout_s=build_timeout_s,
                     on_timeout=timed_out.set)
    beat.start()
    exit_code = None
    try:
        (work / "job.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")
        result, exit_code = _run_build(job_id, job, work, f"{job_id}/{job.track_id}.zip", pipeline, storage,
                                       settings, beat, timed_out)
    except Exception:
        # Not the build's own failure (it reports those): the scratch folder is full, say, or the build
        # process couldn't be started.
        log.exception("job %s failed", job_id)
        result = {"outcome": "failed", "error": INTERNAL_ERROR}
    finally:
        beat.halt()
        shutil.rmtree(work, ignore_errors=True)

    if result is None:
        if timed_out.is_set():
            return  # stopped at the time limit: nothing is reported, so the job goes stale and is requeued
        log.error("job %s failed: its build process %s", job_id, _exit_description(exit_code))
        result = {"outcome": "failed", "error": INTERNAL_ERROR}
    _log_memory(job_id, result.get("memory"))
    if result["outcome"] == "done":
        # The track is uploaded: from here on, trouble reaching the API must not turn into a failed job.
        _report_success(api, job_id, worker_id, {"key": result["key"], "summary": result["summary"]})
    elif result["outcome"] == "failed" and not beat.lost:
        _report_failure(api, job_id, worker_id, result["error"], *result.get("failure", (None, None)))
    # "dropped": the job was taken away before the upload; someone else's result is the one that counts.


def _run_build(job_id: str, job: Job, work: Path, key: str, pipeline: Pipeline, storage: Storage | None,
               settings: ServiceSettings, beat: Heartbeat,
               timed_out: threading.Event) -> tuple[dict | None, int | None]:
    """Build `job` in a process of its own. Returns the result it sent (None if it sent none: it crashed, was
    killed, or was stopped at the time limit) and its exit code. Nothing of the build outlives this."""
    results, results_w = _SPAWN.Pipe(duplex=False)
    go_r, go = _SPAWN.Pipe(duplex=False)
    proc = _SPAWN.Process(target=_build_process, name=f"build-{job_id}",
                          args=(results_w, go_r, job_id, job, str(work), key, pipeline, storage, settings,
                                _log_levels()))
    try:
        try:
            proc.start()
        finally:
            results_w.close()  # the build process has its own ends; ours would keep the pipe open
            go_r.close()
        return _follow(proc, results, go, beat, timed_out, job_id), proc.exitcode
    finally:
        _stop(proc)
        results.close()
        go.close()


def _follow(proc, results, go, beat: Heartbeat, timed_out: threading.Event, job_id: str) -> dict | None:
    """Relay the build process's messages until it has ended, killing it if it runs past the time limit."""
    result = None
    pipe_open = True
    killed_at = result_at = None
    while True:
        ready = wait_for([results, proc.sentinel] if pipe_open else [proc.sentinel], timeout=POLL_S)
        if results in ready:
            try:
                kind, *body = results.recv()
            except (EOFError, OSError):
                pipe_open = False
            else:
                if kind == "log":
                    _relog(body[0])
                elif kind == "stage":
                    beat.set_stage(body[0])
                elif kind == "built":
                    try:
                        go.send(not beat.lost)  # upload only if the job is still this worker's
                    except OSError:
                        pass  # it has died meanwhile; its exit code says how
                elif kind == "result":
                    result, result_at = body[0], time.monotonic()
                continue  # read everything it sent before looking at whether it has ended
        if proc.exitcode is not None and not (pipe_open and _readable(results)):
            return result
        now = time.monotonic()
        if killed_at is None and (timed_out.is_set() or (result_at is not None and now - result_at > EXIT_WAIT_S)):
            if timed_out.is_set():
                log.warning("job %s: killing its build process", job_id)
            _kill_tree(proc)
            killed_at = now
        elif killed_at is not None and now - killed_at > KILL_WAIT_S:
            log.error("job %s: its build process won't end when killed; exiting for a restart", job_id)
            _abandon_stuck_build()


def _readable(conn) -> bool:
    """Whether a message (or the end of the pipe) is waiting. A pipe on Windows says broken instead of ended."""
    try:
        return conn.poll()
    except OSError:
        return False


def _kill_tree(proc) -> None:
    """Kill the build process and anything it started."""
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # it leads a process group of its own (see _build_process)
        except (ProcessLookupError, PermissionError):
            pass  # gone already, or so new it hasn't made its group yet: the kill below gets it
    elif proc.is_alive():
        try:  # taskkill follows the tree by parent process id
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass
    proc.kill()


def _stop(proc) -> None:
    """Make sure the build process has gone (it has, unless the worker is aborting) and release it."""
    if proc.pid is None:
        return  # never started
    if proc.is_alive():
        _kill_tree(proc)
    proc.join(KILL_WAIT_S)
    if not proc.is_alive():
        proc.close()


def _exit_description(code: int | None) -> str:
    if code is not None and code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        return f"was killed by {name}" + (" (out of memory?)" if -code == 9 else "")  # SIGKILL: the OOM killer's
    return f"exited with code {code} before sending a result"


def _log_levels() -> dict[str, int]:
    """This process's logging levels (the root's, and any logger's set apart), for the build process."""
    levels = {"": logging.getLogger().level}
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger, logging.Logger) and logger.level != logging.NOTSET:
            levels[name] = logger.level
    return levels


def _relog(fields: dict) -> None:
    """Log a record from the build process as this process's own: same logger, same level, same handlers."""
    record = logging.makeLogRecord(fields)
    logger = logging.getLogger(record.name)
    if logger.isEnabledFor(record.levelno):
        logger.handle(record)


def _log_memory(job_id: str, memory: dict | None) -> None:
    if memory:
        own = _memory()
        log.info("job %s: build process peak %.2f GiB; worker process %s", job_id, memory["peak"] / 2**30,
                 f"{own['rss'] / 2**20:.0f} MiB" if own else "unknown")


def _memory() -> dict | None:
    """This process's resident memory now and at its peak, in bytes, where the system says (Linux, Windows)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            c = Counters(cb=ctypes.sizeof(Counters))
            if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return None
            return {"rss": c.WorkingSetSize, "peak": c.PeakWorkingSetSize}
        with open("/proc/self/status", encoding="ascii") as f:
            kb = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in f) if k in ("VmRSS", "VmHWM")}
        return {"rss": kb["VmRSS"] * 1024, "peak": kb["VmHWM"] * 1024}
    except (OSError, KeyError, ValueError, AttributeError):
        return None


# -- the build process --------------------------------------------------------------------------------------------

class _Sender:
    """Sends messages to the worker process. The build's own threads log too, so one at a time."""

    def __init__(self, conn):
        self._conn = conn
        self._lock = threading.Lock()

    def __call__(self, *message) -> None:
        with self._lock:
            try:
                self._conn.send(message)
            except OSError:
                pass  # the worker process has gone, and this one is about to follow it


class _LogForwarder(logging.Handler):
    """Sends each record to the worker process, which logs it as its own. Only plain values cross: the message
    is formatted here, and a traceback goes as text."""

    def __init__(self, send: _Sender):
        super().__init__()
        self._send = send

    def emit(self, record: logging.LogRecord) -> None:
        try:
            fields = dict(record.__dict__, msg=record.getMessage(), args=None, exc_info=None)
            if record.exc_info and not record.exc_text:
                fields["exc_text"] = logging.Formatter().formatException(record.exc_info)
            self._send("log", {k: v if v is None or isinstance(v, (str, int, float, bool)) else str(v)
                               for k, v in fields.items()})
        except Exception:
            self.handleError(record)


def _build_process(results, go, job_id: str, job: Job, work: str, key: str, pipeline: Pipeline,
                   storage: Storage | None, settings: ServiceSettings, levels: dict[str, int]) -> None:
    """The build process: build the job, upload the track if the worker says the job is still its own, and
    send back what happened. Its log records go to the worker process, at the worker's levels."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is the worker's to act on: it finishes the job first
    if hasattr(os, "setsid"):
        os.setsid()  # a process group of its own, so a timeout kills everything the build started
    _end_with_worker()
    send = _Sender(results)
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.addHandler(_LogForwarder(send))
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)

    try:
        out = pipeline(job, Path(work), lambda name: send("stage", name))
        send("built")
        if go.recv():
            (storage if storage is not None else storage_for(settings)).put(out.zip_path, key)
            result = {"outcome": "done", "key": key, "summary": out.summary}
        else:
            result = {"outcome": "dropped"}  # taken away: someone else's result is the one that counts
    except Exception as exc:
        if isinstance(exc, user_facing()):
            # About the road as drawn, and the page tells the user why: not an error in the service, so it
            # stays out of the error-level alerts. Internal failures are errors, with the traceback.
            log.warning("job %s: the road can't be built as drawn: %s", job_id, public_error(exc))
        else:
            log.exception("job %s failed", job_id)
        result = {"outcome": "failed", "error": public_error(exc), "failure": failure_detail(exc)}
    result["memory"] = _memory()
    send("result", result)
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    # Straight out: a thread the build left behind (a hung download, say) mustn't keep the process alive.
    os._exit(0)


def _end_with_worker() -> None:
    """End this build process if the worker process goes (killed, say), rather than build for nobody."""
    worker = multiprocessing.parent_process()
    if worker is None:
        return

    def watch() -> None:
        wait_for([worker.sentinel])
        os._exit(ORPHANED_EXIT_CODE)

    threading.Thread(target=watch, name="watch-worker", daemon=True).start()


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
