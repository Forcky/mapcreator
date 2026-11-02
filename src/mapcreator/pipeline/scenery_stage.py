"""Stage 4b: trees and buildings around the road, as a second scene the track stage merges.

Trees come from the region's individual-crown canopy model when one is
configured (Tasmania's 2 m CHM mirror), otherwise from ESA WorldCover tree
cover. Buildings come from the region's footprint layer. Everything here is
visual only.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np
import shapely

from mapcreator.config import Settings
from mapcreator.core import mesh as base_mesh
from mapcreator.core import scenery, sprites
from mapcreator.core.scene import Material, Scene
from mapcreator.job import Job
from mapcreator.pipeline import mesh_stage, profile_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.esa import worldcover
from mapcreator.regions import region_for

STAGE = "04b_scenery"


def _seed(job: Job) -> int:
    return int(hashlib.sha256(job.track_id.encode()).hexdigest()[:8], 16)


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    region = region_for(job.waypoints[0].lon, job.waypoints[0].lat)
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    sec = np.load(work.root / section_stage.STAGE / "section.npz")
    frame = Scene.load(work.root / mesh_stage.STAGE).frame
    rng = np.random.default_rng(_seed(job))
    xy = c["xy"]
    centre = shapely.LineString(xy)
    half = float(max(sec["half_left"].max(), sec["half_right"].max()))
    road_clear = centre.buffer(half + base_mesh.VERGE_M + scenery.CLEAR_OF_VERGE_M)
    attributions = []

    dem = region.elevation(settings)
    dem.load_corridor(xy, profile_stage.CORRIDOR_M)

    # Buildings first: trees must not grow through roofs.
    buildings: list = []
    if region.buildings is not None:
        bbox = centre.buffer(scenery.BUILDING_REACH_M).bounds
        buildings = scenery.select_buildings(region.buildings(bbox, settings), xy, centre.buffer(half + 2.0))
        if buildings:
            attributions.append(region.buildings_attribution)
    built = shapely.unary_union([b.buffer(3.0) for b in buildings]) if buildings else None

    canopy = region.canopy(settings) if region.canopy else None
    tree_region = centre.buffer(scenery.TREE_REACH_M).difference(road_clear)
    if canopy is not None:
        canopy.load_corridor(xy, scenery.TREE_REACH_M + 10)
        cxy, ch = canopy.crowns()
        shapely.prepare(tree_region)
        inside = shapely.contains_xy(tree_region, cxy[:, 0], cxy[:, 1]) if len(cxy) else np.zeros(0, bool)
        trees = scenery.Trees(cxy[inside], np.clip(ch[inside], 4.0, 45.0))
        source = "canopy model"
        attributions.append(_attr("mapcreator.providers.tas_list.canopy"))
    else:
        cover = worldcover.WorldCover(settings.cache_dir, region.epsg)
        cover.load_corridor(xy, scenery.TREE_REACH_M + 20)
        cand = scenery.candidates_from_cover(tree_region, rng)
        is_tree = cover.classes(cand[:, 0], cand[:, 1]) == worldcover.TREE
        cand = cand[is_tree]
        trees = scenery.Trees(cand, scenery.heights_for_cover(len(cand), rng))
        source = "ESA WorldCover"
        attributions.append(worldcover.ATTRIBUTION)

    if built is not None and len(trees):
        shapely.prepare(built)
        trees = trees.subset(~shapely.contains_xy(built, trees.xy[:, 0], trees.xy[:, 1]))
    trees = scenery.thin_by_distance(trees, xy, rng)
    ground = dem.sample(trees.xy[:, 0], trees.xy[:, 1]) if len(trees) else np.zeros(0)
    ok = np.isfinite(ground)
    trees, ground = trees.subset(ok), ground[ok]

    meshes = scenery.tree_meshes(trees, ground, frame, xy, rng, road_half=half)
    meshes += scenery.building_meshes(buildings, dem.sample, frame, rng)

    out = work.stage(STAGE)
    textures = {"trees": out / "trees.png", "walls": out / "walls.png"}
    sprites.tree_atlas().save(textures["trees"])
    sprites.weatherboard().save(textures["walls"])
    materials = {"trees": Material("trees", "trees", kind="alpha_test"), "walls": Material("walls", "walls")}
    for colour, rgb in sprites.ROOF_COLOURS.items():
        textures[f"roof_{colour}"] = out / f"roof_{colour}.png"
        sprites.corrugated(rgb).save(textures[f"roof_{colour}"])
        materials[f"roof_{colour}"] = Material(f"roof_{colour}", f"roof_{colour}")

    scene = Scene(frame=frame, meshes=meshes, materials=materials, textures={k: str(v) for k, v in textures.items()})
    scene.save(out)
    work.write_json(STAGE, "attributions.json", [asdict(a) for a in attributions])
    return {
        "trees": len(trees),
        "tree_source": source,
        "buildings": len(buildings),
        "meshes": len(meshes),
        "vertices": int(sum(len(m.positions) for m in meshes)),
        "triangles": int(sum(len(m.triangles) for m in meshes)),
    }


def _attr(module: str):
    import importlib

    return importlib.import_module(module).ATTRIBUTION
