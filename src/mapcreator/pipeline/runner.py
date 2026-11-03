"""Run a build's stages: in order, except the mesh and scenery stages, which run side by side.

A build is mostly single-threaded Python, so a worker with more CPUs builds no faster on its own (a 63 km
road kept about one core busy, Nov 2026). The mesh and scenery stages need nothing from each other: both read
the centreline and the cross-section, and scenery works out the same local frame as the mesh stage does. So
they run in two threads, and much of their numpy, shapely and scipy work runs in parallel outside the GIL
(63 km: 70 s together instead of 85 s one after the other).

They share no state in memory. Both write their own stage folder, and the on-disk caches they share write
atomically.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from mapcreator.config import Settings
from mapcreator.job import Job
from mapcreator.pipeline.workdir import WorkDir

Stage = Callable[[Job, Settings, WorkDir], dict]
TOGETHER = ("mesh", "scenery")


def run_stages(stages: dict[str, Stage], names: list[str], job: Job, settings: Settings, work: WorkDir, *,
               on_stage: Callable[[str], None] = lambda name: None,
               on_result: Callable[[str, dict, float], None] = lambda name, result, seconds: None) -> dict[str, dict]:
    """Run `names` (stage names, in order) and return each stage's result. `on_stage` is told when a stage
    starts (for the mesh/scenery pair: mesh, then scenery once mesh is done); `on_result` gets each result and
    its time in seconds."""
    results: dict[str, dict] = {}
    i = 0
    while i < len(names):
        name = names[i]
        if names[i:i + 2] == list(TOGETHER):
            with ThreadPoolExecutor(len(TOGETHER), thread_name_prefix="stage") as ex:
                started = {n: ex.submit(_timed, stages[n], job, settings, work) for n in TOGETHER}
                for n in TOGETHER:
                    on_stage(n)
                    result, seconds = started[n].result()  # the first failure is raised here
                    results[n] = result
                    on_result(n, result, seconds)
            i += len(TOGETHER)
            continue
        on_stage(name)
        result, seconds = _timed(stages[name], job, settings, work)
        results[name] = result
        on_result(name, result, seconds)
        i += 1
    return results


def _timed(stage: Stage, job: Job, settings: Settings, work: WorkDir) -> tuple[dict, float]:
    t0 = time.perf_counter()
    result = stage(job, settings, work)
    return result, time.perf_counter() - t0
