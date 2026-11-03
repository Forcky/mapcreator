"""Stage 4: build the road, verges, skirts and terrain as a neutral scene."""

from __future__ import annotations

import logging
import math

import numpy as np

from mapcreator.config import Settings
from mapcreator.core import mesh, roadside
from mapcreator.core import water as water_core
from mapcreator.core.scene import LocalFrame, Material, Scene
from mapcreator.job import Job
from mapcreator.pipeline import profile_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.regions import FINE_DEM_M, job_region

STAGE = "04_mesh"
TERRAIN_CORRIDOR_M = 300.0

log = logging.getLogger(__name__)


def _water(c, rb, dem, settings: Settings, region) -> list[water_core.WaterBody]:
    """Lakes, rivers and sea in the terrain corridor, from ESA WorldCover's permanent water class. Water is
    scenery: if WorldCover can't be read, the track is built without it."""
    try:
        from mapcreator.providers.esa.worldcover import WATER, WorldCover

        cover = WorldCover(settings.cache_dir, region.epsg)
        cover.load_corridor(c["xy"], TERRAIN_CORRIDOR_M)
        corridor, footprint = mesh.terrain_region(rb, TERRAIN_CORRIDOR_M)
        area = corridor.difference(footprint.buffer(0.5))
        return water_core.bodies(lambda x, y: cover.classes(x, y) == WATER, area, dem.sample)
    except Exception as e:  # noqa: BLE001 - scenery, not the road
        log.warning("no water: WorldCover unavailable (%s: %s)", type(e).__name__, e)
        return []


def frame_for(xy: np.ndarray, z: np.ndarray) -> LocalFrame:
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    return LocalFrame(e0=float(round((lo[0] + hi[0]) / 2)), n0=float(round((lo[1] + hi[1]) / 2)),
                      h0=float(math.floor(np.nanmin(z))))


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    region = job_region(job.waypoints)
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    sec = np.load(work.root / section_stage.STAGE / "section.npz")
    dem = region.elevation(settings)
    dem.load_corridor(c["xy"], profile_stage.CORRIDOR_M)

    frame = frame_for(c["xy"], c["z"])
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], dem.sample)
    water = _water(c, rb, dem, settings, region)
    # Inside water the terrain sits just under the surface, so the land never shows through it.
    ground = water_core.lowered(dem.sample, water) if water else dem.sample
    meshes = mesh.ribbon_meshes(rb, c["s"], frame)
    meshes += mesh.terrain_meshes(rb, TERRAIN_CORRIDOR_M, ground, frame)
    meshes += water_core.meshes(water, frame)
    rails, rail_stats = [], {"skipped": "DEM too coarse to place rails"}
    fine = c["fine"] if "fine" in c else np.full(len(c["s"]), getattr(dem, "resolution_m", 2.0) <= FINE_DEM_M)
    if fine.any():  # only where the DEM shows real drops: a coarse one shows hillsides as cliffs
        rails, rail_stats = roadside.guard_rails(c["s"], c["xy"], c["z"], c["heading"], sec["half_left"],
                                                 sec["half_right"], sec["slope_left"], sec["slope_right"],
                                                 dem.sample, frame, allowed=fine)
    meshes += rails
    meshes += roadside.end_walls(c["s"], c["xy"], c["z"], c["heading"], dem.sample, frame)

    scene = Scene(
        frame=frame,
        meshes=meshes,
        materials={
            "road": Material("road", texture="road_marked", uv_scale=1.0),
            "grass": Material("grass", texture="grass"),
            "terrain": Material("terrain", texture="ground"),
            "rail": Material("rail", texture="steel"),
            **({"water": Material("water", texture="water")} if water else {}),
        },
    )
    out = work.stage(STAGE)
    scene.save(out)
    water_summary = water_core.summary(water)
    work.write_json(STAGE, "water.json", water_summary)
    if settings.preview_obj:
        scene.write_obj(out / "preview.obj")

    by_kind: dict[str, list[int]] = {}
    for m in meshes:
        kind = m.name.split("_")[0] + ("_T" if "_T" in m.name else "_V" if "_V" in m.name else "")
        by_kind.setdefault(kind, []).append(len(m.positions))
    return {
        "meshes": len(meshes),
        "vertices": int(sum(len(m.positions) for m in meshes)),
        "triangles": int(sum(len(m.triangles) for m in meshes)),
        "max_vertices_in_a_mesh": int(max(len(m.positions) for m in meshes)),
        "by_kind": {k: {"meshes": len(v), "vertices": int(sum(v))} for k, v in by_kind.items()},
        "frame": [frame.e0, frame.n0, frame.h0],
        "guard_rails": rail_stats,
        "water": water_summary,
    }
