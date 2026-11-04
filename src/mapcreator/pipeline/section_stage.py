"""Stage 3: surface, carriageway width and camber at every centreline station."""

from __future__ import annotations

import numpy as np

from mapcreator.config import Settings
from mapcreator.core import centreline as cl
from mapcreator.core import profile as pf
from mapcreator.core import section as sec
from mapcreator.job import Job
from mapcreator.pipeline import profile_stage, route_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.regions import FINE_DEM_M, job_region

STAGE = "03_section"


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    region = job_region(job.waypoints)
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    runs = work.read_json(route_stage.STAGE, "runs.json")

    # Metres of centreline on each run: the surface merge measures stretches of the road as built.
    lengths = np.bincount(c["run"][:-1], weights=np.diff(c["s"]), minlength=len(runs))
    per_run = sec.run_sections([r["tags"] for r in runs], lengths.tolist())
    raw_width = np.array([r.width_m for r in per_run])[c["run"]] * job.width.factor
    width = sec.eased(c["s"], raw_width)
    half = width / 2
    style = np.array([r.style for r in per_run], dtype=np.int8)[c["run"]]
    unsealed = sec.is_unsealed(style)
    crown = sec.eased(c["s"], np.where(unsealed, sec.CROWN_UNSEALED, sec.CROWN))

    # Superelevation from the DEM just inside each edge (2 m cells, so not right at the edge).
    dem = region.elevation(settings)
    dem.load_corridor(c["xy"], profile_stage.CORRIDOR_M)
    n = cl.normals(c["heading"])
    probe = np.maximum(half - 0.5, 1.0)
    zl = dem.sample(*(c["xy"] + n * probe[:, None]).T)
    zr = dem.sample(*(c["xy"] - n * probe[:, None]).T)
    e = pf.cross_slope(c["s"], zl, zr, probe, clamp=sec.MAX_SUPERELEVATION)
    # A coarse DEM (or a coarse stretch of a fine one) can't see a road's camber: crown only there, eased.
    fine = c["fine"] if "fine" in c else np.full(len(c["s"]), getattr(dem, "resolution_m", 2.0) <= FINE_DEM_M)
    if not fine.all():
        e = e * sec.eased(c["s"], fine.astype(float))
    slope_l, slope_r = sec.half_slopes(e, crown)

    np.savez_compressed(work.stage(STAGE) / "section.npz", s=c["s"], half_left=half, half_right=half,
                        slope_left=slope_l, slope_right=slope_r, superelevation=e, crown=crown, style=style,
                        unsealed=unsealed)
    sources: dict[str, float] = {}
    inherited: dict[str, float] = {}
    for r, run, m in zip(per_run, runs, lengths):
        sources[r.width_source] = round(sources.get(r.width_source, 0.0) + float(m), 1)
        if not r.tagged:
            value = str(run["tags"].get("surface") or "none")
            inherited[value] = round(inherited.get(value, 0.0) + float(m), 1)
    ds = np.diff(c["s"])
    return {
        "width_min": round(float(width.min()), 2), "width_max": round(float(width.max()), 2),
        "width_source_m": sources, "max_superelevation": round(float(np.abs(e).max()), 3),
        "banked_share": round(float(np.mean(np.abs(e) > crown)), 3),
        "surface_m": {sec.SEALED: round(float(ds[~unsealed[:-1]].sum()), 1),
                      sec.UNSEALED: round(float(ds[unsealed[:-1]].sum()), 1)},
        # Runs whose own `surface` gave no class (no tag, or a value outside both lists), by that value.
        "surface_inherited_m": inherited,
        "surface_changes": int(np.count_nonzero(np.diff(unsealed.astype(np.int8)))),
        "styles_m": {name: round(float(ds[style[:-1] == k].sum()), 1) for k, name in enumerate(sec.STYLES)
                     if (style[:-1] == k).any()},
    }
