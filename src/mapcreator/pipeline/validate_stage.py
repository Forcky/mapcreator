"""Stage 6: static validation of the assembled track folder.

A build failure raises `ValidationFailed`, which the hosted service shows as an internal error. A bend
tighter than any real road raises `TightBend` instead: a message for the person who drew the route, saying
where the bend is, and whether they can ask for the road to be built anyway.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from mapcreator.config import Settings
from mapcreator.core import validate
from mapcreator.core.scene import Scene
from mapcreator.job import Job
from mapcreator.pipeline import mesh_stage, profile_stage, track_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.regions import job_region

STAGE = "06_validate"


class ValidationFailed(RuntimeError):
    pass


class TightBend(ValidationFailed):
    """The route has a bend tighter than any real road. The message is written for the person who drew it."""

    def __init__(self, radius_m: float, along_m: float, lon: float, lat: float, overridable: bool):
        self.radius_m, self.along_m, self.lon, self.lat, self.overridable = radius_m, along_m, lon, lat, overridable
        where = f"about {along_m / 1000:.1f} km from the start"
        cause = ("That usually means it turns back on itself at a junction, or a waypoint is just off the road. "
                 "Try moving the waypoint near there.")
        if overridable:
            text = (f"This route has a bend tighter than any real hairpin: {radius_m:.1f} m radius, {where}. "
                    f"{cause} If the road really is this tight, you can generate it anyway.")
        else:
            text = f"This route turns almost on the spot ({radius_m:.1f} m radius, {where}), which no car can drive. {cause}"
        super().__init__(text)

    def detail(self) -> dict:
        return {"radius_m": round(self.radius_m, 1), "along_m": round(self.along_m, 1),
                "lon": round(self.lon, 6), "lat": round(self.lat, 6), "overridable": self.overridable}


def _where(job: Job, work: WorkDir, bend: validate.TightBend) -> tuple[float, float, float]:
    """Distance along the route as drawn (whichever layout the bend was found in), longitude, latitude."""
    frame = Scene.load(work.root / mesh_stage.STAGE).frame
    e, n, _ = frame.from_ac(np.array([bend.x, 0.0, bend.z]))
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    _, i = cKDTree(c["xy"]).query([float(e), float(n)])
    lon, lat = job_region(job.waypoints).to_lonlat().transform(float(e), float(n))
    return float(c["s"][i]), float(lon), float(lat)


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    rep = validate.track_folder(work.root / track_stage.STAGE / job.track_id, allow_tight_bends=job.allow_tight_bends)
    work.write_json(STAGE, "report.json", {"ok": rep.ok, "passed": rep.passed, "failed": rep.failed,
                                           "route_failed": rep.route_failed, "warnings": rep.warnings})
    if rep.failed:
        raise ValidationFailed("; ".join(rep.failed + rep.route_failed))
    out: dict = {"ok": True, "checks": len(rep.passed)}
    if rep.bends:
        bend = min(rep.bends, key=lambda b: b.radius_m)
        along, lon, lat = _where(job, work, bend)
        if rep.route_failed:
            raise TightBend(bend.radius_m, along, lon, lat, bend.radius_m >= validate.FLOOR_RADIUS_M)
        out["tight_bend"] = {"radius_m": round(bend.radius_m, 1), "along_m": round(along, 1)}
    return out
