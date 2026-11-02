"""Stage 6: static validation of the assembled track folder."""

from __future__ import annotations

from mapcreator.config import Settings
from mapcreator.core import validate
from mapcreator.job import Job
from mapcreator.pipeline import track_stage
from mapcreator.pipeline.workdir import WorkDir

STAGE = "06_validate"


class ValidationFailed(RuntimeError):
    pass


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    rep = validate.track_folder(work.root / track_stage.STAGE / job.track_id)
    work.write_json(STAGE, "report.json", {"ok": rep.ok, "passed": rep.passed, "failed": rep.failed})
    if not rep.ok:
        raise ValidationFailed("; ".join(rep.failed))
    return {"ok": True, "checks": len(rep.passed)}
