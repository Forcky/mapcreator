"""Stage 3: surface, carriageway width and camber at every centreline station; the side roads' stubs; and the
roundabouts the route drives.

The stubs' plan (`core.stubs`, `stubs.json`) and the roundabouts' (`core.roundabouts`, `roundabouts.json`) are worked
out here, before the mesh and scenery stages, because those two run side by side and both need them: the mesh stage
builds the stubs and rings from them, and the scenery stage keeps trees, buildings and power-line supports off them.
The roundabouts are planned first: a side road's stub keeps off a ring the route drives.
"""

from __future__ import annotations

import numpy as np

from mapcreator.config import Settings
from mapcreator.core import centreline as cl
from mapcreator.core import profile as pf
from mapcreator.core import roundabouts as rb_core
from mapcreator.core import section as sec
from mapcreator.core import stubs as stubs_core
from mapcreator.core.layout import layouts_for
from mapcreator.job import Job
from mapcreator.pipeline import profile_stage, route_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.regions import FINE_DEM_M, job_region

STAGE = "03_section"
# The side roads' stubs (`core.stubs`; the format is in docs/ac-format.md).
STUBS = "stubs.json"
# The roundabouts the route drives: the rest of each ring, its island and its closed exits (`core.roundabouts`).
ROUNDABOUTS = "roundabouts.json"


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    region = job_region(job.waypoints)
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    runs = work.read_json(route_stage.STAGE, "runs.json")

    lengths = sec.run_lengths(c["s"], c["run"], len(runs))
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
    # Less camber on a narrow road, and always a little crown (core/section.py).
    limit = sec.camber_limit(width)
    e = np.clip(e, -limit, limit)
    slope_l, slope_r = sec.half_slopes(e, crown, keep=sec.crown_kept(width))

    np.savez_compressed(work.stage(STAGE) / "section.npz", s=c["s"], half_left=half, half_right=half,
                        slope_left=slope_l, slope_right=slope_r, superelevation=e, crown=crown, style=style,
                        unsealed=unsealed, camber_limit=limit)
    found = route_stage.junctions(work, job)
    rings = route_stage.roundabouts(work, job)
    section = {"half_left": half, "half_right": half, "slope_left": slope_l, "slope_right": slope_r, "style": style}
    layouts = tuple(lay.reverse for lay in layouts_for(job.layouts.value))
    ring_summary, avoid = None, None
    if rings:
        ring_plan = rb_core.plan(rings, c, section, dem.sample, width_factor=job.width.factor, layouts=layouts)
        work.write_json(STAGE, ROUNDABOUTS, {"version": ring_plan.pop("version"), "crs": f"EPSG:{region.epsg}",
                                             **ring_plan})
        avoid = rb_core.avoid(ring_plan["roundabouts"])
        ring_summary = {"drawn": len(ring_plan["roundabouts"]),
                        "exits": len(rb_core.exit_stubs(ring_plan["roundabouts"])),
                        "skipped": {s["reason"]: sum(k["reason"] == s["reason"] for k in ring_plan["skipped"])
                                    for s in ring_plan["skipped"]}}
    else:
        (work.stage(STAGE) / ROUNDABOUTS).unlink(missing_ok=True)
    stub_summary = None
    if found:
        if avoid is None:
            plan = stubs_core.plan(found, c, section, dem.sample, width_factor=job.width.factor, layouts=layouts)
        else:
            plan = stubs_core.plan(found, c, section, dem.sample, width_factor=job.width.factor, layouts=layouts,
                                   avoid=avoid)
        work.write_json(STAGE, STUBS, {"version": plan.pop("version"), "crs": f"EPSG:{region.epsg}", **plan})
        stub_summary = {"stubs": len(plan["stubs"]), "skipped": {}}
        for sk in plan["skipped"]:
            stub_summary["skipped"][sk["reason"]] = stub_summary["skipped"].get(sk["reason"], 0) + 1
    else:
        (work.stage(STAGE) / STUBS).unlink(missing_ok=True)  # no junctions, or built without them
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
        "camber_limit": [round(float(limit.min()), 3), round(float(limit.max()), 3)],
        "at_camber_limit_share": round(float(np.mean(np.abs(e) >= limit - 1e-4)), 3),
        # Where the two halves form one tilted plane, with no crown left.
        "plane_share": round(float(np.mean(slope_l + slope_r < 1e-6)), 3),
        "surface_m": {sec.SEALED: round(float(ds[~unsealed[:-1]].sum()), 1),
                      sec.UNSEALED: round(float(ds[unsealed[:-1]].sum()), 1)},
        # Runs whose own `surface` gave no class (no tag, or a value outside both lists), by that value.
        "surface_inherited_m": inherited,
        "surface_changes": int(np.count_nonzero(np.diff(unsealed.astype(np.int8)))),
        "styles_m": {name: round(float(ds[style[:-1] == k].sum()), 1) for k, name in enumerate(sec.STYLES)
                     if (style[:-1] == k).any()},
    } | ({"junction_stubs": stub_summary} if stub_summary is not None else {})       | ({"roundabouts": ring_summary} if ring_summary is not None else {})


def roundabouts(work: WorkDir, job: Job) -> list[dict]:
    """The roundabouts the section stage planned to draw (`core.roundabouts`), as roundabouts.json lists them: none for
    a job built without junctions, or with no roundabout on its route."""
    path = work.root / STAGE / ROUNDABOUTS
    if not route_stage.roundabouts(work, job) or not path.exists():
        return []
    return work.read_json(STAGE, ROUNDABOUTS)["roundabouts"]


def stubs(work: WorkDir, job: Job) -> list[dict]:
    """The side roads' stubs the section stage planned (`core.stubs`), as stubs.json lists them: none for a job built
    without junctions, or a section stage run before stubs."""
    path = work.root / STAGE / STUBS
    if not route_stage.junctions(work, job) or not path.exists():
        return []
    return work.read_json(STAGE, STUBS)["stubs"]
