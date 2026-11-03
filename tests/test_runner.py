"""The stage runner: in order, with the mesh and scenery stages side by side."""

import threading
import time

import pytest

from mapcreator.pipeline.runner import run_stages


def test_stages_run_in_order_and_mesh_and_scenery_overlap():
    events, lock = [], threading.Lock()

    def stage(name, wait=0.0):
        def run(job, settings, work):
            with lock:
                events.append(("start", name))
            time.sleep(wait)
            with lock:
                events.append(("end", name))
            return {"name": name}
        return run

    stages = {"route": stage("route"), "mesh": stage("mesh", 0.3), "scenery": stage("scenery", 0.3),
              "track": stage("track")}
    told, results = [], []
    t0 = time.perf_counter()
    out = run_stages(stages, list(stages), None, None, None, on_stage=told.append,
                     on_result=lambda n, r, s: results.append(n))
    assert time.perf_counter() - t0 < 0.55  # 0.3 s each, side by side
    assert out == {n: {"name": n} for n in stages}
    assert told == ["route", "mesh", "scenery", "track"] and results == told
    assert events.index(("start", "scenery")) < events.index(("end", "mesh"))
    assert events.index(("start", "track")) > max(events.index(("end", "mesh")), events.index(("end", "scenery")))


def test_a_failure_in_either_stops_the_build():
    def ok(job, settings, work):
        return {}

    def broken(job, settings, work):
        raise RuntimeError("scenery broke")

    with pytest.raises(RuntimeError, match="scenery broke"):
        run_stages({"mesh": ok, "scenery": broken, "track": ok}, ["mesh", "scenery", "track"], None, None, None)
