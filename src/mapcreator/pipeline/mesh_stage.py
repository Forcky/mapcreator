"""Stage 4: build the road, verges, skirts and terrain as a neutral scene.

Where the region has a vegetation map for the ground (TASVEG in Tasmania), the terrain and the skirt are split
into ground classes (`core.grass`): a material each, so Grass FX grows each class's own grass and the ground
takes its colour. The map is read out to the terrain's edge. Without one, the terrain is one material as before.

Side roads' stubs (`core.stubs`) come from the section stage's plan: each its own ribbon, joined to the road by its
flares, with the road's verge and skirt cut across its mouth and the terrain leaving room for it. Each is closed off
the stage, with police cars at some of the major closures (`core.stub_closures`).

Roundabouts the route drives (`core.roundabouts`) come from the section stage's plan too: the rest of each ring, its
throats where it meets the road, its island, and its exits' stubs and closures. The road's verge and skirt come out on
the island's side between the ring's two ends. A ring that fails the checks on the built meshes is left out whole, and
the road through it is built as before; an exit that fails is left out on its own, as a side road's stub is.

The route's own bridges (`core.bridges`) are planned here from the profile's bridge stations: a kerb strip and parapets
on every tagged bridge, and an open deck over the ground, with its abutments and piers, where the gap under it allows.
Untagged fills and bridges' closed stretches get 1:2 batters. An open deck that fails the checks on the built meshes is
built again on the ground (`bridges.faults`).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict

import numpy as np
import shapely

from mapcreator import cpus
from mapcreator.config import Settings
from mapcreator.core import bridges as bridges_core
from mapcreator.core import furniture, grass, junctions, mesh, police_car, reach, roadside, stub_closures, stubs
from mapcreator.core import crossing_roads, rail_and_paths
from mapcreator.core import ramps as ramps_core
from mapcreator.core import roundabouts as rb_core
from mapcreator.core import water as water_core
from mapcreator.core.layout import layouts_for
from mapcreator.core.scene import LocalFrame, Material, Scene
from mapcreator.job import Job
from mapcreator.pipeline import profile_stage, route_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.base import OSM_LINES
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
# The roundabouts as built: the section stage's plan less any ring or exit left out here (`rb_core.settle`).
ROUNDABOUTS = "roundabouts.json"
# The roads crossing the route as built: the section stage's plan, each road's pieces drawn and those left out.
CROSSING_ROADS = "crossing_roads.json"
# The railways and paths crossing the route, each drawn or left out and why (`core.rail_and_paths`).
RAIL_AND_PATHS = "rail_and_paths.json"
# The route's own bridges as built (`core.bridges`; the format is in docs/ac-format.md). The track and validate stages
# read this one.
BRIDGES = "bridges.json"
# The slip ramps and the shallow forks' gore stubs (`core.forks`) as built: the section stage's plan less any left out
# here (`ramps_core.settle`).
RAMPS = "ramps.json"

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
    near = cKDTree(centre_xy).query(en, distance_upper_bound=250.0, workers=cpus.count())[0] <= 250.0
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


def built_roundabouts(work: WorkDir, job: Job) -> list[dict]:
    """The roundabouts as the mesh stage built them (`ROUNDABOUTS`): the section stage's plan, less any ring or exit it
    left out. The section stage's plan where the mesh stage hasn't run."""
    plan = section_stage.roundabouts(work, job)
    path = work.root / STAGE / ROUNDABOUTS
    if not plan or not path.exists():
        return plan
    return work.read_json(STAGE, ROUNDABOUTS)["roundabouts"]


def built_bridges(work: WorkDir) -> list[dict]:
    """The route's own bridges as the mesh stage built them (`BRIDGES`, `core.bridges`): none where it built none, or
    ran before bridges."""
    path = work.root / STAGE / BRIDGES
    return work.read_json(STAGE, BRIDGES)["bridges"] if path.exists() else []


def built_ramps(work: WorkDir, job: Job) -> list[dict]:
    """The slip ramps as the mesh stage built them (`RAMPS`): the section stage's plan, less any it left out. The
    section stage's plan where the mesh stage hasn't run."""
    plan = section_stage.ramps(work, job)
    path = work.root / STAGE / RAMPS
    if not plan or not path.exists():
        return plan
    return work.read_json(STAGE, RAMPS)["ramps"]


def _road_and_ground(plan, c, sec, frame, dem, gmap, settings, region, rings=(), crossing=None, crossed=None,
                     decks=None, ramps=(), ramp_out=None):
    """The road's ribbon, the stubs (`stubs.build`) and roundabouts (`rb_core.build`), water, and the road, verge, stub,
    ring and terrain meshes for `plan` and `rings`. `crossing`: the roads crossing the route to draw
    (`core.crossing_roads`), kept off the ground the rest takes; what was drawn goes in `crossed["built"]`. `decks`
    (`bridges.Decks`): the route's own bridges. `ramps`: the slip ramps (`ramps_core.build`), built before the
    crossing roads, which keep off them too; what was built goes in `ramp_out["built"]`."""
    ring_pts = rb_core.centres(rings) if rings else None
    # The road's skirt is held short of the roads crossing it as well (`core.crossing_roads`), as of a stub's, and of
    # the slip ramps (`core.ramps`).
    cross_pts = crossing_roads.centres(crossing)
    if rings:
        caps = stubs.mainline_caps(plan, c["xy"], c["heading"], extra=ramps_core.with_centres(
            ring_pts if cross_pts is None else np.vstack([ring_pts, cross_pts]), ramps))
    elif cross_pts is not None or ramps:
        caps = stubs.mainline_caps(plan, c["xy"], c["heading"], extra=ramps_core.with_centres(cross_pts, ramps))
    else:
        caps = stubs.mainline_caps(plan, c["xy"], c["heading"]) if plan else (None, None)
    # Bridges: the kerb strip in place of the verge, and 1:2 batters on fills for the skirt.
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], decks.embanked(dem.sample) if decks else dem.sample,
                           skirt_cap_l=caps[0], skirt_cap_r=caps[1], **(decks.verges() if decks else {}))
    main_xy = np.vstack([c["xy"], ring_pts]) if rings else c["xy"]
    main_xy = ramps_core.with_centres(main_xy, ramps)
    built = (stubs.build(plan, rb, c["s"], frame, dem.sample, skirt=gmap is None, main_xy=main_xy)
             if plan else None)
    extra = built.footprint if built is not None else None
    ring_built = None
    if rings:
        stub_pts = [stubs._centre(st) for st in plan]
        ring_built = rb_core.build(rings, rb, c["s"], frame, dem.sample, skirt=gmap is None,
                                   others=np.vstack([c["xy"]] + stub_pts))
        cut = {k: ring_built.cut[k] | (built.cut[k] if built is not None else False) for k in "LR"}
        edge_break = {k: ring_built.edge_break[k] | (built.edge_break[k] if built is not None else False)
                      for k in "LR"}
        extra = rb_core.footprint(built, ring_built, cut)
    if decks:  # open decks leave the terrain's footprint (so a road under one runs on under it), and their
        # abutments' faces join it
        extra = decks.footprint(rb, extra)
    ramp_built = None
    if ramps:
        stub_pts = [stubs._centre(st) for st in plan] + ([ring_pts] if rings else [])
        ramp_built = ramps_core.build(ramps, rb, c["s"], frame, dem.sample, skirt=gmap is None,
                                      others=np.vstack([c["xy"]] + stub_pts))
        extra = ramps_core.footprint(extra, ramp_built, len(c["s"]))
    if ramp_out is not None:
        ramp_out["built"] = ramp_built
    cross_built = None
    if crossing and crossing["roads"]:
        # Roads crossing the route (`core.crossing_roads`): each its own ribbon, clear of the ground taken already.
        taken = mesh.terrain_region(rb, TERRAIN_CORRIDOR_M, extra)[1]
        others = np.vstack([main_xy] + [stubs._centre(st) for st in plan])
        cross_built = crossing_roads.build(crossing, taken, frame, dem.sample, skirt=gmap is None, others_xy=others)
        extra = crossing_roads.footprint(extra, cross_built, len(c["s"]))
        if crossed is not None:
            crossed["built"] = cross_built
    water = _water(c, rb, dem, settings, region, extra)
    # Inside water the terrain sits just under the surface, so the land never shows through it.
    ground = water_core.lowered(dem.sample, water) if water else dem.sample
    if decks:  # batters on fills, the ground kept under open decks, and points under them for the terrain
        ground = decks.ground(ground)
        extra = decks.footprint_points(extra, ground)
    if crossed is not None:
        crossed["footprint"] = extra  # for the railways and paths under the route's bridges (`_rail_and_paths`)
    # A section stage from before surfaces has no styles: all sealed and marked, as it was built then.
    style = sec["style"] if "style" in sec else np.zeros(len(c["s"]), dtype=np.int8)
    deck_kw = {"kerb": decks.kerb()} if decks else {}
    # With ground classes the skirt goes with the terrain beside it, so only the 2 m verge is mown grass.
    if ring_built is None:
        cut, edge_break = (built.cut, built.edge_break) if built is not None else (None, None)
    cut, edge_break = ramps_core.road_cut(cut, edge_break, ramp_built)
    if decks:
        meshes = mesh.ribbon_meshes(rb, c["s"], frame, style=style, skirt=gmap is None, cut=decks.cut(cut),
                                    edge_break=edge_break, **deck_kw)
    else:
        meshes = mesh.ribbon_meshes(rb, c["s"], frame, style=style, skirt=gmap is None, cut=cut,
                                    edge_break=edge_break)
    meshes += (built.meshes if built is not None else []) + (ring_built.meshes if ring_built is not None else [])
    if cross_built is not None:
        meshes += cross_built.meshes
    meshes += ramp_built.meshes if ramp_built is not None else []
    if gmap is None:
        meshes += mesh.terrain_meshes(rb, TERRAIN_CORRIDOR_M, ground, frame, extra=extra)
    else:
        meshes += mesh.terrain_meshes(rb, TERRAIN_CORRIDOR_M, ground, frame, ground=gmap.materials,
                                      classes=grass.MESHES, fold_skirt=True, extra=extra, cut=grass.CUT_MATERIAL)
    meshes += water_core.meshes(water, frame)
    if decks:
        deck_meshes, decks.stats = decks.meshes(rb, frame, ground, cut=cut)
        meshes += deck_meshes
    if rings:
        return rb, (built, ring_built), water, meshes
    return rb, built, water, meshes


def _rail_and_paths(work: WorkDir, c, sec, frame: LocalFrame, dem, meshes, rb, built, cross_built=None,
                    extra=None, ramps=()) -> rail_and_paths.Built:
    """Footbridges and rail bridges over the route, and paths and railways under its own bridges
    (`core.rail_and_paths`), on the ground and road just built, off the route's ribbon, its stubs', the crossing
    roads' (`cross_built`) and the slip ramps' (`ramps`); and `RAIL_AND_PATHS`, what was drawn or left out. `extra`:
    the terrain's footprint as built (`mesh.Footprint`), with the route's open decks out of it (`core.bridges`), so a
    path or railway under one runs on under it; without it, the stubs' alone."""
    (work.stage(STAGE) / RAIL_AND_PATHS).unlink(missing_ok=True)
    found = route_stage.crossing_lines(work)
    if not found:
        return rail_and_paths.Built()
    areas = []
    for cr in found:
        st = cr.get("structure") or {}
        pts = np.vstack([np.asarray(p, float).reshape(-1, 2) for p in
                         [cr["line"], st.get("deck") or [], *(st.get("approaches") or [])] if len(p)])
        lo, hi = pts.min(axis=0) - 10.0, pts.max(axis=0) + 10.0
        areas.append((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))
    ground = stub_closures.TopSurface([m for m in meshes if m.name.startswith("1GRASS")], frame, areas)
    road = stub_closures.TopSurface([m for m in meshes if m.name.startswith(("1ROAD", "1GRAVEL"))], frame, areas)
    if extra is None:
        extra = built.footprint if built is not None else None
    footprint = mesh.terrain_region(rb, 1.0, extra)[1]
    if cross_built is not None and cross_built.pieces:
        footprint = shapely.unary_union([footprint] + [shapely.Polygon(pc["outline"]).buffer(0)
                                                       for pc in cross_built.pieces if len(pc["outline"]) >= 3])
    if ramps:
        footprint = shapely.union(footprint, ramps_core.ground(ramps))
    out = rail_and_paths.build(found, c, sec, frame, dem.sample, ground, road, footprint)
    work.write_json(STAGE, RAIL_AND_PATHS, {"version": rail_and_paths.FORMAT_VERSION, "structures": out.records})
    return out


def _mouths(found, plan, rings, c, sec) -> dict[str, np.ndarray] | None:
    """The side roads' mouths and the roundabouts' island sides as planned, per side (`junctions.masks`,
    `rb_core.masks`): the route's own bridges keep clear of them (`core.bridges`)."""
    out = None
    if found:
        out = junctions.masks(junctions.mouths(found, c["s"], c["xy"], c["heading"], sec["half_left"],
                                               sec["half_right"], stubs=stubs.by_junction(plan)), c["s"])
    if rings:
        ring_masks = rb_core.masks(rings, c["s"])
        out = ring_masks if out is None else {k: out[k] | ring_masks[k] for k in "LR"}
    return out


def furniture_mouths(*masks: dict[str, np.ndarray] | None) -> dict[str, np.ndarray] | None:
    """Where no guide post, curve marker or sign stands, per side: the union of the given masks (side roads', ramps'
    and roundabouts' mouths, driveways' mouths, bridges' parapets), any of them None; None if all are."""
    given = [m for m in masks if m is not None]
    if not given:
        return None
    return {k: np.logical_or.reduce([np.asarray(m[k], bool) for m in given]) for k in "LR"}


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
    # The route's own bridges (`core.bridges`), kept clear of side roads' mouths and roundabouts as planned.
    runs = work.read_json(route_stage.STAGE, "runs.json")
    planned_mouths = _mouths(found, plan, section_stage.roundabouts(work, job), c, sec)
    ramp_plan = section_stage.ramps(work, job)
    if ramp_plan:  # and of the slip ramps' tapers, gores and nose islands (`core.ramps`)
        planned_mouths = ramps_core.with_masks(planned_mouths, ramp_plan, c["s"])
    bridge_plan = bridges_core.plan(c, mouths=planned_mouths, names=[r["tags"].get("name") for r in runs])
    decks = bridges_core.Decks(bridge_plan, c, sec, planned_mouths)
    decks = decks if decks else None  # nothing to build: no bridge, and no fill to batter
    lonlat = region.to_lonlat().transform(c["xy"][::50, 0], c["xy"][::50, 1])
    jurisdiction = police_car.jurisdiction_for(region, list(zip(*lonlat)))
    reverses = tuple(lay.reverse for lay in layouts_for(job.layouts.value))
    # Roads crossing the route (`core.crossing_roads`): the ground under each deck over it is held below its underside.
    crossing = section_stage.crossing_roads(work)
    ground_dem = crossing_roads.GroundUnderDecks(dem, crossing["decks"]) if crossing and crossing["decks"] else dem
    crossed: dict = {}
    # Slip ramps (`core.ramps`): built with the stubs and closed with them, under the same cap on police cars; each
    # checked on the meshes just built as a stub is, and one that fails left out (`ramps_core.settle`).
    ramp_state = {"ramps": ramp_plan, "built": None}

    def build(stubs_now):
        ramps_now = ramp_state["ramps"]
        rb, built, water, ground_meshes = _road_and_ground(stubs_now, c, sec, frame, ground_dem, gmap, settings, region,
                                                           crossing=crossing, crossed=crossed, decks=decks,
                                                           ramps=ramps_now, ramp_out=ramp_state)
        closed = None
        if stubs_now or ramps_now:
            # Each stub closed off the stage, and police cars at some of the major closures (`core.stub_closures`),
            # standing on the ground just built.
            every = stubs_now + ramps_now
            top = stub_closures.TopSurface(ground_meshes, frame, stub_closures.areas(every))
            closed = stub_closures.place(every, found + ramps_core.junctions_for(ramps_now), c, sec, frame, top,
                                         jurisdiction=jurisdiction, reverses=reverses)
        return rb, built, water, ground_meshes, closed

    def all_faults(stubs_now, result, extra=()):
        """`stubs.faults` with the ramps' checks alongside: their closures count as on a stub, their faults theirs."""
        return stubs.faults(stubs_now, result[3], result[4].meshes if result[4] is not None else [], c, sec, frame,
                            extra=list(extra) + ramps_core.checks(ramp_state["ramps"], c, sec, frame))

    def check(stubs_now, result):
        return ramps_core.without_ramps(all_faults(stubs_now, result))

    planned = bool(plan)
    rings = section_stage.roundabouts(work, job)
    ring_built, dropped_rings = None, []
    if rings:
        # Roundabouts (`core.roundabouts`): their rings, islands and exits are built and checked with the stubs, and
        # their exits closed with the side roads; a ring that fails is left out whole (`rb_core.settle`).
        def build_rings(state):
            stubs_now, rings_now = state
            ramps_now = ramp_state["ramps"]
            rb, pair, water, ground_meshes = _road_and_ground(stubs_now, c, sec, frame, ground_dem, gmap, settings,
                                                              region, rings=rings_now, crossing=crossing,
                                                              crossed=crossed, decks=decks, ramps=ramps_now,
                                                              ramp_out=ramp_state)
            closed = None
            every = stubs_now + rb_core.exit_stubs(rings_now) + ramps_now
            if every:
                top = stub_closures.TopSurface(ground_meshes, frame, stub_closures.areas(every))
                closed = stub_closures.place(every, found + rb_core.exit_junctions(rings_now)
                                             + ramps_core.junctions_for(ramps_now), c, sec, frame, top,
                                             jurisdiction=jurisdiction, reverses=reverses,
                                             parents=rb_core.parents(rings_now), extra_road=rb_core.paved(rings_now))
            return rb, pair if isinstance(pair, tuple) else (pair, None), water, ground_meshes, closed

        def check_rings(state, result):
            stubs_now, rings_now = state
            return ramps_core.without_ramps(all_faults(stubs_now, result, rb_core.checks(rings_now, c, sec, frame)))

        def ramp_faults_rings(ramps_now, out):
            return ramps_core.faults_of(all_faults(out[0], out[2], rb_core.checks(out[1], c, sec, frame)),
                                        ramp_state["built"])

        plan, rings, (rb, (built, ring_built), water, meshes, closed), dropped, dropped_rings = ramps_core.settle(
            ramp_state, lambda: rb_core.settle((plan, rings), build_rings, check_rings, STUB_REBUILDS, log),
            ramp_faults_rings, STUB_REBUILDS, log)
    else:
        def ramp_faults(ramps_now, out):
            return ramps_core.faults_of(all_faults(out[0], out[1]), ramp_state["built"])

        plan, (rb, built, water, meshes, closed), dropped = ramps_core.settle(
            ramp_state, lambda: stubs.settle(plan, build, check, STUB_REBUILDS, log), ramp_faults, STUB_REBUILDS, log)
    bridge_faults = bridges_core.faults(decks.records(rb), meshes, frame) if decks and decks.spans else {}
    if bridge_faults:
        # An open deck that fails its checks is built on the ground, with its parapets; the stubs and rings as settled.
        log.warning("bridges built on the ground, their open decks failing checks: %s", bridge_faults)
        decks.close(bridge_faults)
        if rings:
            rb, (built, ring_built), water, meshes, closed = build_rings((plan, rings))
        else:
            rb, built, water, meshes, closed = build(plan)
    ramps_now, ramp_built = ramp_state["ramps"], ramp_state["built"]
    ramps_core.log_paint(ramp_built, ramps_now, log)  # once, for the ramps as settled
    (out / BRIDGES).unlink(missing_ok=True)
    if decks and decks.bridges:
        work.write_json(STAGE, BRIDGES, {"version": bridges_core.FORMAT_VERSION, "crs": f"EPSG:{region.epsg}",
                                         "bridges": decks.records(rb)})
    (out / STUBS).unlink(missing_ok=True)
    if planned:
        doc = work.read_json(section_stage.STAGE, section_stage.STUBS)
        work.write_json(STAGE, STUBS, {**doc, "stubs": plan,
                                       "skipped": sorted(doc["skipped"] + dropped, key=lambda d: d["junction"])})
    (out / RAMPS).unlink(missing_ok=True)
    if ramp_plan:
        doc = work.read_json(section_stage.STAGE, section_stage.RAMPS)
        work.write_json(STAGE, RAMPS, {**doc, "ramps": ramps_now,
                                       "skipped": sorted(doc["skipped"] + ramp_state.get("dropped", []),
                                                         key=lambda d: d["junction"])})
    (out / ROUNDABOUTS).unlink(missing_ok=True)
    if section_stage.roundabouts(work, job):
        doc = work.read_json(section_stage.STAGE, section_stage.ROUNDABOUTS)
        work.write_json(STAGE, ROUNDABOUTS, {**doc, "roundabouts": rings, "skipped": doc["skipped"] + dropped_rings})
    # Side roads' mouths (`core.junctions`): no guard rail, guide post, curve marker or sign stands in one; nor on a
    # roundabout's island side, between its ends (`rb_core.masks`).
    mouths = junctions.masks(junctions.mouths(found, c["s"], c["xy"], c["heading"], sec["half_left"],
                                              sec["half_right"], stubs=stubs.by_junction(plan)),
                             c["s"]) if found else None
    if rings:
        ring_masks = rb_core.masks(rings, c["s"])
        mouths = ring_masks if mouths is None else {k: mouths[k] | ring_masks[k] for k in "LR"}
    if ramps_now:  # nor along a slip ramp's taper, gore and nose island (`ramps_core.masks`)
        ramp_masks = ramps_core.masks(ramps_now, c["s"])
        mouths = ramp_masks if mouths is None else {k: mouths[k] | ramp_masks[k] for k in "LR"}
    # Nor in a driveway's, a service road's, a track's or a private road's mouth (`junctions.find_driveways`), except
    # on the route's own bridges, where nothing meets the road at its level.
    drive = junctions.mouths(route_stage.driveways(work, job), c["s"], c["xy"], c["heading"], sec["half_left"],
                             sec["half_right"])
    drive_masks = junctions.masks(drive, c["s"]) if drive else None
    junction_stats = {"found": len(found)} if found else None
    if ramp_plan:
        junction_stats = junction_stats or {}
        junction_stats["ramps"] = ramp_built.stats if ramp_built is not None and ramps_now else {"ramps": 0}
        if ramp_state.get("dropped"):
            junction_stats["ramps_left_out"] = len({d["junction"] for d in ramp_state["dropped"]})
    if rings:
        junction_stats = junction_stats or {}
        junction_stats["roundabouts"] = ring_built.stats if ring_built is not None else {"rings": 0}
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
    # The route's own bridges: rails on both sides wherever the deck stands clear of the ground, on any DEM.
    on_bridge = roadside.bridge_mask(c["s"], c["bridge"], c["z"] - c["z_raw"]) if "bridge" in c else no_rail
    if on_bridge.any():
        masks = {side: mask | on_bridge for side, mask in masks.items()}
    if decks and decks.bridges:  # on the bridge itself the parapet takes over, sharing the rail's end station
        masks = {side: mask & ~bridges_core.interior(decks.parapet[side]) for side, mask in masks.items()}
    driveway_stats = {"found": len(drive)} if drive else None
    if fine.any() or on_bridge.any():
        if mouths is not None:
            masks, junction_stats["rail_m_removed"] = roadside.clear_mouths(masks, mouths, c["s"])
        if drive_masks is not None:
            before = masks
            masks, driveway_stats["rail_m_removed"] = roadside.clear_mouths(
                masks, {k: drive_masks[k] & ~on_bridge for k in "LR"}, c["s"])
            # The driveways a rail used to run across: a rail now ends either side of each.
            driveway_stats["rail_gaps"] = sum(
                bool((before[m.side] & ~masks[m.side] & (c["s"] >= m.s0) & (c["s"] <= m.s1)).any()) for m in drive)
        rails, rail_stats = roadside.guard_rails(c["s"], c["xy"], c["z"], c["heading"], sec["half_left"],
                                                 sec["half_right"], sec["slope_left"], sec["slope_right"],
                                                 dem.sample, frame, allowed=fine, masks=masks)
        if on_bridge.any():
            rail_stats["on_bridges_m"] = round(float(on_bridge.sum() * (c["s"][1] - c["s"][0])), 1)
    meshes += rails
    limits = furniture.speed_limits(c["run"], runs)
    # Nothing stands on a bridge but its parapets and their reflectors: no guide post, curve marker or sign, as in a
    # side road's mouth (a post due there goes up just past it, a sign moves off it).
    # Nor in a driveway's mouth.
    furn_mouths = furniture_mouths(mouths, drive_masks, decks.parapet if decks and decks.bridges else None)
    furn, furn_stats, atlas = furniture.furniture(
        c["s"], c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
        sec["slope_left"], sec["slope_right"], sec["superelevation"], limits, frame, masks["L"], masks["R"],
        unsealed=sec["unsealed"] if "unsealed" in sec else None,
        mouth_l=furn_mouths["L"] if furn_mouths is not None else None,
        mouth_r=furn_mouths["R"] if furn_mouths is not None else None,
        **({"no_cams": rb_core.cam_mask(rings, c["s"])} if rings else {}))
    if decks and decks.bridges:
        furn += bridges_core.reflectors(decks, rb, frame, atlas)
    meshes += furn
    textures = {}
    if furn:
        textures[furniture.MATERIAL] = str(out / "furniture.png")
        atlas.image.save(textures[furniture.MATERIAL])
    meshes += roadside.end_walls(c["s"], c["xy"], c["z"], c["heading"], dem.sample, frame)
    # Railways and paths crossing the route (`core.rail_and_paths`), on the ground and road built above.
    lines_built = _rail_and_paths(work, c, sec, frame, dem, meshes, rb, built, crossed.get("built"),
                                  extra=crossed.get("footprint"), ramps=ramps_now)
    meshes += lines_built.meshes
    # The closures placed and checked above, last, as before.
    (out / CLOSURES).unlink(missing_ok=True)
    if closed is not None:
        meshes += closed.meshes
        for name, image in closed.textures.items():
            textures[name] = str(out / f"{name}.png")
            image.save(textures[name])
        work.write_json(STAGE, CLOSURES, {"version": 1, "closures": closed.records})
        junction_stats["closures"] = closed.stats

    # The bridges over the route (`core.crossing_roads`), and the roads crossing it as drawn.
    deck_meshes, deck_stats = crossing_roads.decks(crossing, frame, ground_dem.sample)
    meshes += deck_meshes
    (out / CROSSING_ROADS).unlink(missing_ok=True)
    cross_stats = None
    if crossing:
        cross_built = crossed.get("built")
        drawn = crossing_roads.as_built(crossing, cross_built) if cross_built is not None else {
            **crossing, "roads": [], "left_out": []}
        work.write_json(STAGE, CROSSING_ROADS, drawn)
        cross_stats = {**crossing_roads.summary(drawn), "decks": deck_stats,
                       "roads_built": cross_built.stats if cross_built is not None else {"pieces": 0}}

    used = {m.material for m in meshes}
    scene = Scene(
        frame=frame,
        meshes=meshes,
        materials={
            # The road's materials are those its chunks use: road, road_plain, gravel, gravel_narrow.
            **{name: Material(name, texture=mesh.ROAD_TEXTURE[name])
               for name in sorted({m.material for m in meshes if m.material in mesh.ROAD_TEXTURE})},
            "grass": Material("grass", texture="grass"),
            # A roundabout island's kerb (`core.roundabouts`), and a bridge's kerb strip.
            **({rb_core.KERB_MATERIAL: Material(rb_core.KERB_MATERIAL, texture=rb_core.KERB_TEXTURE)}
               if rb_core.KERB_MATERIAL in used else {}),
            # The bridges over the route: their concrete and its darker underside (`core.crossing_roads`).
            **{name: Material(name, texture=crossing_roads.CONCRETE_TEXTURE, properties=props)
               for name, props in crossing_roads.KN5_PROPERTIES.items() if name in used},
            # The terrain's materials: `terrain` alone, or with a vegetation map those of its ground classes.
            **{c.material: Material(c.material, texture=c.texture) for c in grass.CLASSES if c.material in used},
            "rail": Material("rail", texture="steel"),
            # The footbridges' and rail bridges' ballast and the paths' surfaces (`core.rail_and_paths`); their
            # concrete is the crossing roads' bridges' (above), and their steel the guard rails'.
            **{name: Material(name, texture=rail_and_paths.TEXTURES[name])
               for name in sorted(lines_built.materials - {"rail"} - set(crossing_roads.KN5_PROPERTIES))},
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
    credits = list(gmap.attributions if gmap is not None else ())
    if any(r["drawn"] for r in lines_built.records):
        credits.append(OSM_LINES)  # footbridges, rail bridges and the lines under the route's bridges
    work.write_json(STAGE, "attributions.json", [asdict(a) for a in credits])
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
    } | ({"junctions": junction_stats} if junction_stats is not None else {}) | (
        {"driveways": driveway_stats} if driveway_stats is not None else {}) | (
        {"crossing_roads": cross_stats} if cross_stats is not None else {}) | (
        {"rail_and_paths": lines_built.stats} if lines_built.records else {}) | (
        {"bridges": decks.stats} if decks else {})
