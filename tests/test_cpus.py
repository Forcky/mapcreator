"""A build uses its container's share of the CPUs, not every CPU the host has."""

import multiprocessing
import os
from pathlib import Path

import pytest

from mapcreator import cpus


def cgroup(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return tmp_path


@pytest.mark.parametrize("files, limit", [
    ({"cpu.max": "400000 100000\n"}, 4.0),  # cgroup v2: docker run --cpus 4
    ({"cpu.max": "150000 100000\n"}, 1.5),
    ({"cpu.max": "max 100000\n"}, None),  # no limit
    ({"cpu/cpu.cfs_quota_us": "200000\n", "cpu/cpu.cfs_period_us": "100000\n"}, 2.0),  # cgroup v1
    ({"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}, None),
    ({"cpu.max": "garbled\n"}, None),
    ({}, None),  # no cgroup: Windows, macOS
], ids=["v2 4 CPUs", "v2 1.5 CPUs", "v2 no limit", "v1 2 CPUs", "v1 no limit", "unreadable", "none"])
def test_the_cgroup_s_cpu_limit(tmp_path, files, limit):
    assert cpus.quota(cgroup(tmp_path, files)) == limit


def visible() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def test_the_count_is_the_limit_rounded_up_and_never_more_than_the_cpus_there_are(tmp_path):
    assert cpus.count(cgroup(tmp_path / "a", {"cpu.max": "150000 100000\n"})) == min(2, visible())
    assert cpus.count(cgroup(tmp_path / "b", {"cpu.max": "10000 100000\n"})) == 1  # a tenth of a CPU: still one
    assert cpus.count(cgroup(tmp_path / "c", {"cpu.max": "99900000 100000\n"})) == visible()
    assert cpus.count(cgroup(tmp_path / "d", {})) == visible()


def test_thread_pools_are_sized_unless_already_set():
    env = {"OPENBLAS_NUM_THREADS": "1"}
    cpus.limit_thread_pools(env)
    assert env == {"OMP_NUM_THREADS": str(cpus.count()), "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": str(cpus.count()), "NUMEXPR_MAX_THREADS": str(cpus.count())}


def _threads_setting(out) -> None:
    out.send(os.environ.get("OMP_NUM_THREADS"))


def test_a_build_process_started_afterwards_inherits_the_sizes(monkeypatch):
    # As the worker does: set them, then start each build in a fresh (spawned) process.
    for name in cpus.THREAD_VARS:
        monkeypatch.setenv(name, "")  # so the test leaves them as it found them
        monkeypatch.delenv(name)
    cpus.limit_thread_pools()
    ctx = multiprocessing.get_context("spawn")
    r, w = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_threads_setting, args=(w,))
    proc.start()
    try:
        assert r.recv() == str(cpus.count())
    finally:
        proc.join(30)
