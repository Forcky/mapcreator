"""How many CPUs a build may use: its container's share, not the whole machine's.

A container limited to a few CPUs still sees every CPU on the host. On a 32-CPU host with a 4-CPU limit (Nov
2026), scipy's `workers=-1` and the OpenBLAS and OpenMP thread pools each started a thread per visible CPU, so a
build ran 64-71 threads against 4 CPUs and a 256-process limit. `count()` takes the smallest of the CPUs the
process may run on (its affinity) and the cgroup's CPU limit, rounded up (Linux; there's no cgroup on Windows or
macOS, where it's the CPUs the process can see).
"""

from __future__ import annotations

import math
import os
from collections.abc import MutableMapping
from functools import cache
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup")
# The thread pools numpy and scipy's libraries start when they're first imported, sized by these.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS")


def quota(root: Path = CGROUP) -> float | None:
    """The cgroup's CPU limit, in CPUs (1.5 for a CPU and a half), or None where there's no limit or no cgroup."""
    try:
        q, period = (root / "cpu.max").read_text().split()[:2]  # cgroup v2: "400000 100000", or "max 100000"
        return None if q == "max" else int(q) / int(period)
    except (OSError, ValueError):
        pass
    try:
        q = int((root / "cpu" / "cpu.cfs_quota_us").read_text())  # cgroup v1: -1 for no limit
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
        return q / period if q > 0 and period > 0 else None
    except (OSError, ValueError):
        return None


@cache
def count(root: Path = CGROUP) -> int:
    """The CPUs a build may use: for `workers=` in scipy, and the thread pools (`limit_thread_pools`)."""
    visible = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    limit = quota(root)
    return max(1, min(visible, math.ceil(limit)) if limit else visible)


def limit_thread_pools(environ: MutableMapping[str, str] = os.environ) -> None:
    """Size the libraries' thread pools to `count()` in processes started from here, unless already set. A pool is
    sized when its library is first imported, so this works for a process started afterwards, such as the worker's
    build process, and not for one that has already imported numpy."""
    for name in THREAD_VARS:
        environ.setdefault(name, str(count()))
