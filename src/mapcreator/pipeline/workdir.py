"""Layout of a track's work folder. Every stage writes under its own subfolder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mapcreator.job import Job


@dataclass(frozen=True)
class WorkDir:
    root: Path

    @classmethod
    def for_job(cls, base: Path, job: Job) -> "WorkDir":
        return cls(base / job.track_id)

    def stage(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, stage: str, name: str, data) -> Path:
        path = self.stage(stage) / name
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def read_json(self, stage: str, name: str):
        return json.loads((self.root / stage / name).read_text(encoding="utf-8"))
