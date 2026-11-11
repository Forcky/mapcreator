"""Stand-ins for the build, for the worker tests.

The worker builds each job in a fresh process and sends it the pipeline by pickling, so a stand-in has to be
importable there by name: a class in a module of its own (the test module would drag the web app into every
build process), set up by its fields rather than by a closure.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from mapcreator.service.worker import BuildOutput

SUMMARY = {"title": "Test Road", "length_m": 1000.0}
# Settings the build reads from the worker's environment (through mapcreator.config).
ENV = ("MAPCREATOR_SOURCE_URL", "MAPCREATOR_NVIS", "MAPCREATOR_SVTM")
_HELD: list[bytes] = []  # memory a build keeps hold of, as a module-level cache would


def _exception(kind: str, args: tuple) -> Exception:
    if kind == "route":
        from mapcreator.providers.osm.graph import RouteError

        return RouteError(*args)
    if kind == "tight":
        from mapcreator.pipeline.validate_stage import TightBend

        return TightBend(*args)
    if kind == "validation":
        from mapcreator.pipeline.validate_stage import ValidationFailed

        return ValidationFailed(*args)
    return OSError(*args)


@dataclass(frozen=True)
class Build:
    """Writes a small zip and returns SUMMARY. Before that, as asked: records what it saw, reports stages (after
    each, waiting for a file of that name in `gate`), logs, holds memory, and then (the first build only, with
    `once`) starts a long-lived process, sleeps, and exits, kills itself or raises."""

    raises: tuple[str, tuple] | None = None  # (kind, args): see _exception
    stages: tuple[str, ...] = ()
    gate: str | None = None
    logs: tuple[tuple[str, int, str], ...] = ()  # (logger, level, message)
    hold_mb: int = 0
    start_process: bool = False
    sleep_s: float = 0.0
    exit_code: int | None = None
    kill_self: bool = False  # SIGKILL, as the kernel's OOM killer sends
    record: str | None = None  # a JSON-lines file: each build adds a line of what it saw
    once: str | None = None  # a marker file, made by the first build: later builds just build

    def __call__(self, job, work: Path, on_stage=lambda name: None) -> BuildOutput:
        from mapcreator import config

        first = True
        if self.once:
            first = not Path(self.once).exists()
            Path(self.once).touch()
        settings = config.load()
        seen = {"pid": os.getpid(), "work": str(work),
                "job_json": json.loads((work / "job.json").read_text(encoding="utf-8")),
                "env": {k: os.environ.get(k) for k in ENV},
                "settings": {"source_url": settings.source_url, "nvis_source": settings.nvis_source,
                             "svtm_source": settings.svtm_source}}
        if self.start_process and first:
            seen["started_pid"] = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]).pid
        if self.record:
            with open(self.record, "a", encoding="utf-8") as f:
                f.write(json.dumps(seen) + "\n")
        for name in self.stages:
            on_stage(name)
            if self.gate:
                wait_for_file(Path(self.gate) / name)
        for logger, level, message in self.logs:
            logging.getLogger(logger).log(level, message)
        if self.hold_mb:
            _HELD.append(b"\1" * (self.hold_mb << 20))
        if first:
            time.sleep(self.sleep_s)
            if self.exit_code is not None:
                os._exit(self.exit_code)
            if self.kill_self:
                os.kill(os.getpid(), signal.SIGKILL)
            if self.raises:
                raise _exception(*self.raises)
        zip_path = work / f"{job.track_id}.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr(f"{job.track_id}/ATTRIBUTION.txt", "test")
        return BuildOutput(zip_path, dict(SUMMARY))


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def wait_for_file(path: Path, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"{path.name} never appeared")
        time.sleep(0.02)


def alive(pid: int) -> bool:
    """Whether a process is still running (a zombie waiting to be reaped counts as gone)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def gone_soon(pid: int, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def end(pid: int) -> None:
    """Best-effort clean-up of a process a failing test left running."""
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
