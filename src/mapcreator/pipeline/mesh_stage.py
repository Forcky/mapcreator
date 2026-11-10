"""Stage 4: build the road, verges, skirts and terrain as a neutral scene.

Where the region has a vegetation map for the ground (TASVEG in Tasmania), the terrain and the skirt are split
into ground classes (`core.grass`): a material each, so Grass FX grows each class's own grass and the ground
takes its colour. The map is read out to the terrain's edge. Without one, the terrain is one material as before.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict

import numpy as np

from mapcreator.config import Settings
from mapcreator.core import furniture, grass, mesh, reach, roadside
from mapcreator.core import water as water_core
from mapcreator.core.scene import LocalFrame, Material, Scene
from mapcreator.job import Job
from mapcreator.pipeline import profile_stage, route_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.regions import FINE_DEM_M, job_region

STAGE = "04_mesh"
# Terrain reaches this far either side of the road: the world's edge, 100 m beyond the scenery (`core.reach`).
# Beyond 300 m its lattice is coarse (mesh.OUTER_LATTICE_M). The DEM corridor covers it (profile_stage).
TERRAIN_CORRIDOR_M = reach.TERRAIN_M

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


def _ground(region, c, settings: Settings):
    """The region's ground classes over the terrain corridor (`core.grass.GroundMap`), or None: without a map
    for the region, or if it can't be read (the ground is then one material, as before)."""
    if region.grass is None:
        return None, "none"
    try:
        gmap = region.grass(c["xy"], TERRAIN_CORRIDOR_M + 10, settings)
    except Exception as e:  # noqa: BLE001 - the ground's classes are scenery, not the road
        log.warning("ground classes unavailable (%s: %s); one terrain material", type(e).__name__, e)
        return None, "unavailable"
    return (gmap, "ok") if gmap is not None and len(gmap) else (None, "empty")


def _ground_shares(meshes, frame: LocalFrame, centre_xy: np.ndarray) -> dict:
    """Each ground material's share of the ground beside the road (verge, skirt and terrain, in plan), out to the
    terrain's edge and within 250 m of the centreline."""
    from scipy.spatial import cKDTree

    ground = [m for m in meshes if m.name.startswith("1GRASS")]
    p = np.vstack([m.positions[m.triangles] for m in ground]).astype(float)
    which = np.repeat(np.arange(len(ground)), [len(m.triangles) for m in ground])
    a, b, cc = p[:, 0], p[:, 1], p[:, 2]
    area = np.abs((b[:, 0] - a[:, 0]) * (cc[:, 2] - a[:, 2]) - (cc[:, 0] - a[:, 0]) * (b[:, 2] - a[:, 2])) / 2
    en = np.column_stack([p[:, :, 0].mean(axis=1) + frame.e0, -p[:, :, 2].mean(axis=1) + frame.n0])
    near = cKDTree(centre_xy).query(en, distance_upper_bound=250.0, workers=-1)[0] <= 250.0
    out = {}
    for key, weight in (("all", area), ("within_250m", area * near)):
        per_mesh = np.bincount(which, weights=weight, minlength=len(ground))
        shares: dict[str, float] = {}
        for m, a_m in zip(ground, per_mesh):
            shares[m.material] = shares.get(m.material, 0.0) + float(a_m) / float(weight.sum())
        out[key] = {name: round(v, 4) for name, v in sorted(shares.items(), key=lambda kv: -kv[1])}
    return out


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
    # A section stage from before surfaces has no styles: all sealed and marked, as it was built then.
    style = sec["style"] if "style" in sec else np.zeros(len(c["s"]), dtype=np.int8)
    gmap, ground_note = _ground(region, c, settings)
    # With ground classes the skirt goes with the terrain beside it, so only the 2 m verge is mown grass.
    meshes = mesh.ribbon_meshes(rb, c["s"], frame, style=style, skirt=gmap is None)
    if gmap is None:
        meshes += mesh.terrain_meshes(rb, TERRAIN_CORRIDOR_M, ground, frame)
    else:
        meshes += mesh.terrain_meshes(rb, TERRAIN_CORRIDOR_M, ground, frame, ground=gmap.materials,
                                      classes=grass.MESHES, fold_skirt=True)
    meshes += water_core.meshes(water, frame)
    rails, rail_stats = [], {"skipped": "DEM too coarse to place rails"}
    fine = c["fine"] if "fine" in c else np.full(len(c["s"]), getattr(dem, "resolution_m", 2.0) <= FINE_DEM_M)
    no_rail = np.zeros(len(c["s"]), dtype=bool)
    masks = {"L": no_rail, "R": no_rail}
    if fine.any():  # only where the DEM shows real drops: a coarse one shows hillsides as cliffs
        masks = roadside.rail_masks(c["s"], c["xy"], c["z"], c["heading"], sec["half_left"], sec["half_right"],
                                    sec["slope_left"], sec["slope_right"], dem.sample, allowed=fine)
        rails, rail_stats = roadside.guard_rails(c["s"], c["xy"], c["z"], c["heading"], sec["half_left"],
                                                 sec["half_right"], sec["slope_left"], sec["slope_right"],
                                                 dem.sample, frame, allowed=fine, masks=masks)
    meshes += rails
    out = work.stage(STAGE)
    limits = furniture.speed_limits(c["run"], work.read_json(route_stage.STAGE, "runs.json"))
    furn, furn_stats, atlas = furniture.furniture(
        c["s"], c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
        sec["slope_left"], sec["slope_right"], sec["superelevation"], limits, frame, masks["L"], masks["R"],
        unsealed=sec["unsealed"] if "unsealed" in sec else None)
    meshes += furn
    textures = {}
    if furn:
        textures[furniture.MATERIAL] = str(out / "furniture.png")
        atlas.image.save(textures[furniture.MATERIAL])
    meshes += roadside.end_walls(c["s"], c["xy"], c["z"], c["heading"], dem.sample, frame)

    used = {m.material for m in meshes}
    scene = Scene(
        frame=frame,
        meshes=meshes,
        materials={
            # The road's materials are those its chunks use: road, road_plain, gravel, gravel_narrow.
            **{name: Material(name, texture=mesh.ROAD_TEXTURE[name])
               for name in sorted({m.material for m in meshes if m.material in mesh.ROAD_TEXTURE})},
            "grass": Material("grass", texture="grass"),
            # The terrain's materials: `terrain` alone, or with a vegetation map those of its ground classes.
            **{c.material: Material(c.material, texture=c.texture) for c in grass.CLASSES if c.material in used},
            "rail": Material("rail", texture="steel"),
            **({"water": Material("water", texture="water")} if water else {}),
            # The furniture's materials (plain, reflectors, the boards' faces) all draw on the one atlas.
            **{name: Material(name, furniture.MATERIAL) for name in sorted({m.material for m in furn})},
        },
        textures=textures,
    )
    scene.save(out)
    water_summary = water_core.summary(water)
    work.write_json(STAGE, "water.json", water_summary)
    ground_summary = {"source": " + ".join(a.dataset for a in gmap.attributions) if gmap is not None else ground_note}
    if gmap is not None:
        ground_summary |= {"areas": len(gmap), **_ground_shares(meshes, frame, c["xy"])}
    work.write_json(STAGE, "grass.json", ground_summary)
    work.write_json(STAGE, "attributions.json", [asdict(a) for a in (gmap.attributions if gmap is not None else ())])
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
        "furniture": furn_stats,
        "water": water_summary,
        "ground": ground_summary,
    }
