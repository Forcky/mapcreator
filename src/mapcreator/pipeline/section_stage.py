"""Stage 3: carriageway width and camber at every centreline station."""

from __future__ import annotations

import numpy as np

from mapcreator.config import Settings
from mapcreator.core import centreline as cl
from mapcreator.core import profile as pf
from mapcreator.core import section as sec
from mapcreator.job import Job
from mapcreator.pipeline import profile_stage, route_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.regions import region_for

STAGE = "03_section"


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    region = region_for(job.waypoints[0].lon, job.waypoints[0].lat)
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    runs = work.read_json(route_stage.STAGE, "runs.json")

    per_run = [sec.width_from_tags(r["tags"]) for r in runs]
    raw_width = np.array([per_run[i][0] for i in c["run"]]) * job.width.factor
    width = sec.eased(c["s"], raw_width)
    half = width / 2

    # Superelevation from the DEM just inside each edge (2 m cells, so not right at the edge).
    dem = region.elevation(settings)
    dem.load_corridor(c["xy"], profile_stage.CORRIDOR_M)
    n = cl.normals(c["heading"])
    probe = np.maximum(half - 0.5, 1.0)
    zl = dem.sample(*(c["xy"] + n * probe[:, None]).T)
    zr = dem.sample(*(c["xy"] - n * probe[:, None]).T)
    e = pf.cross_slope(c["s"], zl, zr, probe, clamp=sec.MAX_SUPERELEVATION)
    slope_l, slope_r = sec.half_slopes(e)

    np.savez_compressed(work.stage(STAGE) / "section.npz", s=c["s"], half_left=half, half_right=half,
                        slope_left=slope_l, slope_right=slope_r, superelevation=e)
    sources = {}
    for (w, src), r in zip(per_run, runs):
        sources[src] = round(sources.get(src, 0.0) + (r["to_m"] - r["from_m"]), 1)
    return {
        "width_min": round(float(width.min()), 2), "width_max": round(float(width.max()), 2),
        "width_source_m": sources, "max_superelevation": round(float(np.abs(e).max()), 3),
        "banked_share": round(float(np.mean(np.abs(e) > sec.CROWN)), 3),
    }
