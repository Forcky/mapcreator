"""`mapcreator <stage> job.json`: run one stage, or `build` to run them all in order."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from mapcreator import config
from mapcreator.job import load_job
from mapcreator.pipeline import mesh_stage, profile_stage, route_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir

STAGES = {
    "route": route_stage.run,
    "profile": profile_stage.run,
    "section": section_stage.run,
    "mesh": mesh_stage.run,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mapcreator")
    ap.add_argument("stage", choices=[*STAGES, "build"])
    ap.add_argument("job", type=Path)
    args = ap.parse_args(argv)

    settings = config.load()
    job = load_job(args.job)
    work = WorkDir.for_job(settings.work_dir, job)
    names = list(STAGES) if args.stage == "build" else [args.stage]
    for name in names:
        t0 = time.perf_counter()
        result = STAGES[name](job, settings, work)
        print(f"[{name}] {time.perf_counter() - t0:.1f}s {json.dumps(result)}")
    print(f"output: {work.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
