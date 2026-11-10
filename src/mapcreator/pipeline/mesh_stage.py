"""Stage 4: build the road, verges, skirts and terrain as a neutral scene.

Where the region has a vegetation map for the ground (TASVEG in Tasmania), the terrain and the skirt are split
into ground classes (`core.grass`): a material each, so Grass FX grows each class's own grass and the ground
takes its colour. The map is read out to the terrain's edge. Without one, the terrain is one material as before.

Side roads' stubs (`core.stubs`) come from the section stage's plan: each its own ribbon, joined to the road by its
flares, with the road's verge and skirt cut across its mouth and the terrain leaving room for it. Each is closed off
the stage, with police cars at some of the major closures (`core.stub_closures`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict

import numpy as np

from mapcreator.config import Settings
from mapcreator.core import furniture, grass, junctions, mesh, police_car, reach, roadside, stub_closures, stubs
from mapcreator.core import water as water_core
from mapcreator.core.layout import layouts_for
from mapcreator.core.scene import LocalFrame, Material, Scene
from mapcreator.job import Job
from mapcreator.pipeline import profile_stage, route_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.regions import FINE_DEM_M, job_region

STAGE = "04_mesh"
# Terrain reaches this far either side of the road: the world's edge, 100 m beyond the scenery (`core.reach`).
# Beyond 300 m its lattice is coarse (mesh.OUTER_LATTICE_M). The DEM corridor covers it (profile_stage).
TERRAIN_CORRIDOR_M = reach.TERRAIN_M
# The closures on the side roads' stubs, one record each (`core.stub_closures`; the format is in docs/ac-format.md).
CLOSURES = "closures.json"
# The stubs as built: the section stage's plan less any left out here for failing the checks on the built meshes
# (`stubs.faults`), which join its `skipped` (`built_stubs`). The track and validate stages read this one.
STUBS = "stubs.json"
# A stub left out this way means building the road and ground again without it; after this many tries, all are.
STUB_REBUILDS = 3

log = logging.getLogger(__name__)


def _water(c, rb, dem, settings: Settings, region, extra=None) -> list[water_core.WaterBody]:
    """Lakes, rivers and sea in the terrain corridor, from ESA WorldCover's permanent water class. Water is
    scenery: if WorldCover can't be read, the track is built without it. `extra`: the stubs' ground."""
    try:
        from mapcreator.providers.esa.worldcover import WATER, WorldCover

        cover = WorldCover(settings.cache_dir, region.epsg)
        cover.load_corridor(c["xy"], TERRAIN_CORRIDOR_M)
        corridor, footprint = mesh.terrain_region(rb, TERRAIN_CORRIDOR_M, extra)
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


def built_stubs(work: WorkDir, job: Job) -> list[dict]:
    """The side roads' stubs as the mesh stage built them (`STUBS`): the section stage's plan, less any it left out.
    The section stage's plan where the mesh stage hasn't run, or ran before stubs could be left out."""
    plan = section_stage.stubs(work, job)
    path = work.root / STAGE / STUBS
    if not plan or not path.exists():
        return plan
    return work.read_json(STAGE, STUBS)["stubs"]


def _road_and_ground(plan, c, sec, frame, dem, gmap, settings, region):
    """The road's ribbon, the stubs (`stubs.build`), water, and the road, verge, stub and terrain meshes for `plan`."""
    caps = stubs.mainline_caps(plan, c["xy"], c["heading"]) if plan else (None, None)
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], dem.sample, skirt_cap_l=caps[0], skirt_cap_r=caps[1])
    built = (stubs.build(plan, rb, c["s"], frame, dem.sample, skirt=gmap is None, main_xy=c["xy"])
             if plan else None)
    extra = built.footprint if built is not None else None
    water = _water(c, rb, dem, settings, region, extra)
    # Inside water the terrain sits just under the surface, so the land never shows through it.
    ground = water_core.lowered(dem.sample, water) if water else dem.sample
    # A section stage from before surfaces has no styles: all sealed and marked, as it was built then.
    style = sec["style"] if "style" in sec else np.zeros(len(c["s"]), dtype=np.int8)
    # With ground classes the skirt goes with the terrain beside it, so only the 2 m verge is mown grass.
    if built is None:
        meshes = mesh.ribbon_meshes(rb, c["s"], frame, style=style, skirt=gmap is None)
    else:
        meshes = mesh.ribbon_meshes(rb, c["s"], frame, style=style, skirt=gmap is None, cut=built.cut,
                                    edge_break=built.edge_break)
        meshes += built.meshes
    if gmap is None:
        meshes += mesh.terrain_meshes(rb, TERRAIN_CORRIDOR_M, ground, frame, extra=extra)
    else:
        meshes += mesh.terrain_meshes(rb, TERRAIN_CORRIDOR_M, ground, frame, ground=gmap.materials,
                                      classes=grass.MESHES, fold_skirt=True, extra=extra)
    meshes += water_core.meshes(water, frame)
    return rb, built, water, meshes


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
    gmap, ground_note = _ground(region, c, settings)
    out = work.stage(STAGE)
    found = route_stage.junctions(work, job)
    # Side roads' stubs (`core.stubs`): the road's skirt is held short of them, its verge and skirt cut across their
    # mouths, and its edge lines broken there. Without a plan the road is built exactly as before. Each stub, and its
    # closure, is then checked on the meshes just built (`stubs.faults`, the validator's own checks): one that fails
    # is left out and the road and ground built again without it, so a side road never fails a build.
    plan = section_stage.stubs(work, job)
    lonlat = region.to_lonlat().transform(c["xy"][::50, 0], c["xy"][::50, 1])
    jurisdiction = police_car.jurisdiction_for(region, list(zip(*lonlat)))
    reverses = tuple(lay.reverse for lay in layouts_for(job.layouts.value))

    def build(stubs_now):
        rb, built, water, ground_meshes = _road_and_ground(stubs_now, c, sec, frame, dem, gmap, settings, region)
        closed = None
        if stubs_now:
            # Each stub closed off the stage, and police cars at some of the major closures (`core.stub_closures`),
            # standing on the ground just built.
            top = stub_closures.TopSurface(ground_meshes, frame, stub_closures.areas(stubs_now))
            closed = stub_closures.place(stubs_now, found, c, sec, frame, top, jurisdiction=jurisdiction,
                                         reverses=reverses)
        return rb, built, water, ground_meshes, closed

    def check(stubs_now, result):
        return stubs.faults(stubs_now, result[3], result[4].meshes if result[4] is not None else [], c, sec, frame)

    planned = bool(plan)
    plan, (rb, built, water, meshes, closed), dropped = stubs.settle(plan, build, check, STUB_REBUILDS, log)
    (out / STUBS).unlink(missing_ok=True)
    if planned:
        doc = work.read_json(section_stage.STAGE, section_stage.STUBS)
        work.write_json(STAGE, STUBS, {**doc, "stubs": plan,
                                       "skipped": sorted(doc["skipped"] + dropped, key=lambda d: d["junction"])})
    # Side roads' mouths (`core.junctions`): no guard rail, guide post, curve marker or sign stands in one.
    mouths = junctions.masks(junctions.mouths(found, c["s"], c["xy"], c["heading"], sec["half_left"],
                                              sec["half_right"], stubs=stubs.by_junction(plan)),
                             c["s"]) if found else None
    junction_stats = {"found": len(found)} if found else None
    if built is not None:
        junction_stats["stubs"] = built.stats
    if dropped:
        junction_stats["stubs_left_out"] = len({d["junction"] for d in dropped})
    rails, rail_stats = [], {"skipped": "DEM too coarse to place rails"}
    fine = c["fine"] if "fine" in c else np.full(len(c["s"]), getattr(dem, "resolution_m", 2.0) <= FINE_DEM_M)
    no_rail = np.zeros(len(c["s"]), dtype=bool)
    masks = {"L": no_rail, "R": no_rail}
    if fine.any():  # only where the DEM shows real drops: a coarse one shows hillsides as cliffs
        masks = roadside.rail_masks(c["s"], c["xy"], c["z"], c["heading"], sec["half_left"], sec["half_right"],
                                    sec["slope_left"], sec["slope_right"], dem.sample, allowed=fine)
        if mouths is not None:
            masks, junction_stats["rail_m_removed"] = roadside.clear_mouths(masks, mouths, c["s"])
        rails, rail_stats = roadside.guard_rails(c["s"], c["xy"], c["z"], c["heading"], sec["half_left"],
                                                 sec["half_right"], sec["slope_left"], sec["slope_right"],
                                                 dem.sample, frame, allowed=fine, masks=masks)
    meshes += rails
    limits = furniture.speed_limits(c["run"], work.read_json(route_stage.STAGE, "runs.json"))
    furn, furn_stats, atlas = furniture.furniture(
        c["s"], c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
        sec["slope_left"], sec["slope_right"], sec["superelevation"], limits, frame, masks["L"], masks["R"],
        unsealed=sec["unsealed"] if "unsealed" in sec else None,
        mouth_l=mouths["L"] if mouths is not None else None, mouth_r=mouths["R"] if mouths is not None else None)
    meshes += furn
    textures = {}
    if furn:
        textures[furniture.MATERIAL] = str(out / "furniture.png")
        atlas.image.save(textures[furniture.MATERIAL])
    meshes += roadside.end_walls(c["s"], c["xy"], c["z"], c["heading"], dem.sample, frame)
    # The closures placed and checked above, last, as before.
    (out / CLOSURES).unlink(missing_ok=True)
    if closed is not None:
        meshes += closed.meshes
        for name, image in closed.textures.items():
            textures[name] = str(out / f"{name}.png")
            image.save(textures[name])
        work.write_json(STAGE, CLOSURES, {"version": 1, "closures": closed.records})
        junction_stats["closures"] = closed.stats

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
            # The closures' (closure, closure_retro) and the police cars' (with their own KN5 and night values).
            **(closed.materials if closed is not None else {}),
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
        kind = m.name.split("_")[0] + ("" if not m.name[:1].isdigit() else "_T" if "_T" in m.name else
                                       "_V" if "_V" in m.name else "_J" if "_J" in m.name else "")
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
    } | ({"junctions": junction_stats} if junction_stats is not None else {})
