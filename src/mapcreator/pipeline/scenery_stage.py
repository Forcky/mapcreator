"""Stage 4b: trees, buildings and power lines around the road, as a second scene the track stage merges.

Trees come from the region's individual-crown canopy model when one is
configured (Tasmania's 2 m CHM mirror), and then every crown in the corridor is
kept, at its own height. Otherwise they're made-up positions in ESA WorldCover
tree cover, thinned with distance and capped per km. Buildings come from the
region's footprint layer, and power lines from the region's power source
(`core.power`): towers or poles, and wires, with colliders on the supports a car
can reach.

Trees are 3D models in a Custom Shaders Patch Tree FX list (`treefx/`), which
only CSP reads; the KN5 has no trees of its own. The track stage packs the list
and models into the track's extension folder. Tree FX has no physics, so the
trunks a car can reach get invisible colliders here (`scenery.trunk_colliders`).
Building walls are solid too; roofs are visual only.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict

import numpy as np
import shapely

from mapcreator.config import Settings
from mapcreator.core import mesh as base_mesh
from mapcreator.core import power, scenery, sprites, treefx
from mapcreator.core.scene import Material, Scene
from mapcreator.job import Job
from mapcreator.pipeline import mesh_stage, profile_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.esa import worldcover
from mapcreator.providers.meta import canopy as meta_canopy
from mapcreator.regions import job_region

log = logging.getLogger(__name__)

STAGE = "04b_scenery"


def _seed(job: Job) -> int:
    return int(hashlib.sha256(job.track_id.encode()).hexdigest()[:8], 16)


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    region = job_region(job.waypoints)
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    sec = np.load(work.root / section_stage.STAGE / "section.npz")
    # The mesh stage's frame, worked out the same way, so the two stages can run side by side (runner.py).
    frame = mesh_stage.frame_for(c["xy"], c["z"])
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
    buildings_note = None
    if region.buildings is not None:
        bbox = centre.buffer(scenery.BUILDING_REACH_M).bounds
        try:
            found = region.buildings(bbox, settings)
        except Exception as e:  # noqa: BLE001 - any source failure costs the buildings, not the track
            log.warning("building footprints unavailable (%s: %s); building without them", type(e).__name__, e)
            found, buildings_note = [], "unavailable"
        buildings = scenery.select_buildings(found, xy, centre.buffer(half + 2.0))
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
        source = "ESA WorldCover"
        attributions.append(worldcover.ATTRIBUTION)
        if settings.meta_canopy and is_tree.any():
            # WorldCover's 10 m tree class over-calls forest by about half; Meta/WRI's canopy map trims it to
            # within about 20% of LiDAR (mapcreator#8). Only a mask: it's too coarse to place or size a tree.
            try:
                meta = meta_canopy.MetaCanopy(settings.cache_dir, region.epsg)
                meta.load_corridor(xy, scenery.TREE_REACH_M + 20)
                is_tree &= meta.canopy_near(cand[:, 0], cand[:, 1])
                source = "ESA WorldCover + Meta/WRI canopy"
                attributions.append(meta_canopy.ATTRIBUTION)
            except Exception as e:  # noqa: BLE001 - the mask only refines WorldCover; without it, WorldCover alone
                log.warning("Meta/WRI canopy unavailable (%s: %s); trees from WorldCover alone", type(e).__name__, e)
                source = "ESA WorldCover (Meta/WRI canopy unavailable)"
        cand = cand[is_tree]
        trees = scenery.Trees(cand, scenery.heights_for_cover(len(cand), rng))

    if built is not None and len(trees):
        shapely.prepare(built)
        trees = trees.subset(~shapely.contains_xy(built, trees.xy[:, 0], trees.xy[:, 1]))
    trees = scenery.keep_trees(trees, xy, float(c["s"][-1]), rng, real_crowns=canopy is not None)
    ground = dem.sample(trees.xy[:, 0], trees.xy[:, 1]) if len(trees) else np.zeros(0)
    ok = np.isfinite(ground)
    trees, ground = trees.subset(ok), ground[ok]

    meshes = scenery.building_meshes(buildings, dem.sample, frame, rng)

    out = work.stage(STAGE)
    power_meshes, power_summary = _power(region, settings, xy, half, dem, frame, out, attributions,
                                         c, sec)
    meshes += power_meshes
    fx, trunk_r = _tree_fx(trees, ground, frame, rng, out)
    edge, station = scenery.edge_distance(trees.xy, xy, c["heading"], sec["half_left"], sec["half_right"])
    colliders = scenery.trunk_colliders(trees.xy, ground, trunk_r, edge, c["s"][station], frame)
    meshes += colliders
    textures = {"walls": out / "walls.png"}
    sprites.weatherboard().save(textures["walls"])
    materials = {"walls": Material("walls", "walls")}
    for colour, rgb in sprites.ROOF_COLOURS.items():
        textures[f"roof_{colour}"] = out / f"roof_{colour}.png"
        sprites.corrugated(rgb).save(textures[f"roof_{colour}"])
        materials[f"roof_{colour}"] = Material(f"roof_{colour}", f"roof_{colour}")
    for name in sorted({m.material for m in power_meshes}):
        textures[name] = out / f"{name}.png"
        sprites.plain(sprites.POWER_COLOURS[name]).save(textures[name])
        materials[name] = Material(name, name)

    scene = Scene(frame=frame, meshes=meshes, materials=materials, textures={k: str(v) for k, v in textures.items()})
    scene.save(out)
    work.write_json(STAGE, "attributions.json", [asdict(a) for a in attributions])
    return {
        "trees": len(trees),
        "tree_source": source,
        "tree_fx_models": fx,
        "trunk_colliders": sum(len(m.positions) for m in colliders) // (2 * scenery.TRUNK_SIDES),
        "trunk_collider_meshes": len(colliders),
        "buildings": len(buildings),
        "buildings_source": buildings_note or "ok",
        "power": power_summary,
        "meshes": len(meshes),
        "vertices": int(sum(len(m.positions) for m in meshes)),
        "triangles": int(sum(len(m.triangles) for m in meshes)),
    }


def _power(region, settings: Settings, xy: np.ndarray, half: float, dem, frame, out,
           attributions: list, c, sec) -> tuple[list, dict]:
    """Towers or poles and wires for the region's mapped power lines near the road. Lines from an ODbL source
    are written to power.geojson, which ships with the track."""
    geo = out / "power.geojson"
    geo.unlink(missing_ok=True)
    if region.power is None:
        return [], {"source": "none"}
    source = f"{region.power_attribution.dataset}, {region.power_attribution.holder}"
    reach = mesh_stage.TERRAIN_CORRIDOR_M - 5.0  # supports stand on the terrain mesh
    bbox = shapely.LineString(xy).buffer(reach).bounds
    try:
        lines = region.power(bbox, settings)
    except Exception as e:  # noqa: BLE001 - any source failure costs the power lines, not the track
        log.warning("power lines unavailable (%s: %s); building without them", type(e).__name__, e)
        return [], {"source": f"{source} (unavailable)"}
    p = power.plan(lines, xy, half, reach_m=reach)
    sxy = np.array([s.xy for s in p.supports]).reshape(-1, 2)
    edge, station = scenery.edge_distance(sxy, xy, c["heading"], sec["half_left"], sec["half_right"])
    s_m = c["s"][station] if len(sxy) else np.zeros(0)
    near_grid = (s_m < power.GRID_ZONE_M) | (s_m > float(c["s"][-1]) - power.GRID_ZONE_M)
    built, info = power.meshes(p, dem.sample, frame, edge, near_grid)
    if p.spans:
        attributions.append(region.power_attribution)
        if region.power_attribution.licence.startswith("ODbL"):
            work_json = power.geojson(p, region.to_lonlat(), "ODbL-1.0", region.power_attribution.line())
            geo.write_text(json.dumps(work_json), encoding="utf-8")
    return built, {"source": source, **p.summary(), **info}


def _tree_fx(trees: scenery.Trees, ground: np.ndarray, frame, rng: np.random.Generator,
             out) -> tuple[list[str], np.ndarray]:
    """3D models for Tree FX and a placement for every tree: a model that suits its height, scaled to it.
    Returns the model names and each tree's trunk radius in metres, as Tree FX draws it."""
    fx = out / "treefx"
    if fx.exists():
        for old in fx.iterdir():
            old.unlink()
    fx.mkdir(exist_ok=True)
    models = treefx.library()
    names = list(models)
    tex = treefx.texture_files()
    for name, model in models.items():
        (fx / f"{name}.bin").write_bytes(treefx.model_bin(model, tex))
    n = len(trees)
    model = treefx.assign(trees.height, names, rng)
    own_height = np.array([models[name].box[0] for name in names])
    size = np.clip(trees.height / own_height[model], 0.3, 3.0) if n else np.zeros(0)
    angle = rng.uniform(0, 360, n)
    width = rng.uniform(0.85, 1.15, n)
    np.savez(fx / "placement.npz",
             pos=frame.to_ac(trees.xy[:, 0], trees.xy[:, 1], ground) if n else np.zeros((0, 3)),
             model=model, size=size, angle=angle, width=width, names=np.array(names))
    trunk = np.array([treefx.trunk_radius(models[name]) for name in names])
    return names, trunk[model] * size * width


def _attr(module: str):
    import importlib

    return importlib.import_module(module).ATTRIBUTION
