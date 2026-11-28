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
from mapcreator.core import crossing_roads as crossing_roads_core
from mapcreator.core import dem_age
from mapcreator.core import forks as forks_core
from mapcreator.core import junctions as jn
from mapcreator.core import kerbs as kerbs_core
from mapcreator.core import profile as pf
from mapcreator.core import ramps as ramps_core
from mapcreator.core import roundabouts as rb_core
from mapcreator.core import section as sec
from mapcreator.core import stubs as stubs_core
from mapcreator.core.layout import layouts_for
from mapcreator.job import Job, Kerbs
from mapcreator.pipeline import profile_stage, route_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.regions import FINE_DEM_M, job_region

STAGE = "03_section"
# The side roads' stubs (`core.stubs`; the format is in docs/ac-format.md).
STUBS = "stubs.json"
# The roundabouts the route drives: the rest of each ring, its island and its closed exits (`core.roundabouts`).
ROUNDABOUTS = "roundabouts.json"
# The roads crossing the route drawn: under its bridges, and over it on their own (`core.crossing_roads`).
CROSSING_ROADS = "crossing_roads.json"
# The slip ramps: their tapers, gores, nose islands and stubs (`core.ramps`); and the shallow forks' gore stubs, drawn
# the same way (`core.forks`, `fork: true`).
RAMPS = "ramps.json"


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
    # Kerbs and footpaths on town roads (`core.kerbs`): the road kerb to kerb there. Elsewhere nothing changes.
    kerb_plan = _kerbs(job, settings, region, work, c, runs, half, unsealed)
    half_l, half_r = kerb_plan.half["L"], kerb_plan.half["R"]
    half = (half_l + half_r) / 2  # the halves stay equal: the DEM's camber is read as before, inside each edge
    width = half_l + half_r

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
    # On the route's own bridges and tunnels, and the humps the profile stage bridged under a crossing road, the DEM
    # beside the road is the ground under the deck, the hill over the tunnel or the embankment: crown only there.
    off_ground = np.zeros(len(c["s"]), dtype=bool)
    for key in ("bridge", "tunnel", "under"):
        if key in c:  # a profile stage from before these were saved: as it was built then
            off_ground |= c[key]
    e = e * sec.off_ground_factor(c["s"], off_ground)
    # Less camber on a narrow road, and always a little crown (core/section.py).
    limit = sec.camber_limit(width)
    e = np.clip(e, -limit, limit)
    slope_l, slope_r = sec.half_slopes(e, crown, keep=sec.crown_kept(width))

    np.savez_compressed(work.stage(STAGE) / "section.npz", s=c["s"], half_left=half_l, half_right=half_r,
                        slope_left=slope_l, slope_right=slope_r, superelevation=e, crown=crown, style=style,
                        unsealed=unsealed, camber_limit=limit,
                        **(kerbs_core.arrays(kerb_plan) if kerb_plan.any() else {}))
    found = route_stage.junctions(work, job)
    rings = route_stage.roundabouts(work, job)
    section = {"half_left": half_l, "half_right": half_r, "slope_left": slope_l, "slope_right": slope_r,
               "style": style}
    kerbed = kerbs_core.Kerbs.from_section(kerbs_core.arrays(kerb_plan)) if kerb_plan.any() else None
    if kerbed is not None and found:
        # A side road leaving a kerbed stretch: its stub's flares follow the kerb return (Q8).
        found = [_with_return(j, c, runs, kerbed) for j in found]
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
        if kerbed is not None:  # its kerb runs round its returns and up its edges (`core.kerbs`)
            for st in plan["stubs"]:
                a, b = st["mouth"]["rows"]
                side = "L" if st["side"] == "left" else "R"
                if min(kerbed.k[side][a], kerbed.k[side][b]) >= 0.99:
                    st["kerb"] = True
        work.write_json(STAGE, STUBS, {"version": plan.pop("version"), "crs": f"EPSG:{region.epsg}", **plan})
        stub_summary = {"stubs": len(plan["stubs"]), "skipped": {}}
        for sk in plan["skipped"]:
            stub_summary["skipped"][sk["reason"]] = stub_summary["skipped"].get(sk["reason"], 0) + 1
        in_clusters = sum(st.get("cluster") is not None for st in plan["stubs"])
        if in_clusters:  # drawn with others as one patch (`stubs._merge`)
            stub_summary["in_clusters"] = in_clusters
    else:
        (work.stage(STAGE) / STUBS).unlink(missing_ok=True)  # no junctions, or built without them
    # Slip ramps (`core.ramps`), planned after the stubs and rings: each keeps off them, so none of those changes.
    ramp_summary, ramp_plan, fork_summary = None, None, None
    ramp_list = route_stage.ramps(work, job)
    if ramp_list:
        ramp_plan = ramps_core.plan(ramp_list, c, section, dem.sample, width_factor=job.width.factor, layouts=layouts,
                                    junctions=found, stubs_plan=plan["stubs"] if found else [],
                                    rings=ring_plan["roundabouts"] if rings else [])
        ramp_summary = {"ramps": len(ramp_plan["ramps"]), "skipped": {}}
        for sk in ramp_plan["skipped"]:
            ramp_summary["skipped"][sk["reason"]] = ramp_summary["skipped"].get(sk["reason"], 0) + 1
    # Shallow forks (`core.forks`): a gore stub, drawn as a slip ramp is, for each side road under 15 degrees and each
    # whose stub was too skewed to flare; keeping off everything above, so none of it changes.
    fork_plan = forks_core.plan(found, plan if found else None, c, section, dem.sample, width_factor=job.width.factor,
                                layouts=layouts, rings=ring_plan["roundabouts"] if rings else [],
                                ramps_planned=ramp_plan["ramps"] if ramp_plan is not None else [])
    doc = forks_core.merged(ramp_plan, fork_plan)
    if doc is not None:
        work.write_json(STAGE, RAMPS, {"version": doc.pop("version"), "crs": f"EPSG:{region.epsg}", **doc})
        if fork_plan is not None:
            fork_summary = forks_core.summary(doc)
    else:
        (work.stage(STAGE) / RAMPS).unlink(missing_ok=True)
    # The roads crossing the route, planned last (19 Nov 2026): the bridges' piers and abutments keep off everything
    # above, so a slip ramp or a gore under a deck is drawn, and the deck stands clear of it.
    cross_summary = _crossing_roads(work, job, region, c, section, dem,
                                    ramps_planned=doc["ramps"] if doc is not None else [])
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
    } | ({"junction_stubs": stub_summary} if stub_summary is not None else {})       | ({"roundabouts": ring_summary} if ring_summary is not None else {}) | (
        {"crossing_roads": cross_summary} if cross_summary is not None else {}) | (
        {"ramps": ramp_summary} if ramp_summary is not None else {}) | (
        {"forks": fork_summary} if fork_summary is not None else {}) | {"kerbs": kerb_plan.summary}


def _crossing_roads(work: WorkDir, job: Job, region, c, section: dict, dem,
                    ramps_planned: list[dict] = ()) -> dict | None:
    """Plan the roads crossing the route to draw (`core.crossing_roads`), whatever the job's junctions option: they
    aren't side roads. Writes crossing_roads.json, or removes a stale one when there's nothing to draw or say.
    `ramps_planned`: the slip ramps and forks' gores (ramps.json's), which the bridges' structures keep off."""
    path = work.stage(STAGE) / CROSSING_ROADS
    found = route_stage.crossings(work)
    if not found:
        path.unlink(missing_ok=True)
        return None
    route_xy = np.load(work.root / route_stage.STAGE / "route.npz")["xy"]
    # The profile stage's verdicts on the bridges (`core.dem_age`): one the elevation data doesn't hold isn't drawn.
    ages = work.root / profile_stage.STAGE / dem_age.FILE
    # Piers and abutments keep off the side roads' stubs, the roundabouts' rings, the slip ramps and the forks' gores
    # (with their nose islands), planned before them.
    keep_off = [np.asarray(x["outline"], float) for x in stubs(work, job) + roundabouts(work, job) if x.get("outline")]
    keep_off += [np.asarray(o, float) for r in ramps_planned
                 for o in (r.get("outline"), (r.get("island") or {}).get("outline")) if o]
    verdicts = work.read_json(profile_stage.STAGE, dem_age.FILE) if ages.exists() else None
    doc = crossing_roads_core.plan(found, c, section, dem.sample, width_factor=job.width.factor,
                                   route_length_m=float(cl.cumulative(route_xy)[-1]), dem_ages=verdicts,
                                   keep_off=keep_off)
    if not (doc["roads"] or doc["decks"] or doc["skipped"]):
        path.unlink(missing_ok=True)
        return None
    work.write_json(STAGE, CROSSING_ROADS, {"version": doc.pop("version"), "crs": f"EPSG:{region.epsg}", **doc})
    return crossing_roads_core.summary(doc)


def crossing_roads(work: WorkDir) -> dict | None:
    """The roads crossing the route the section stage planned to draw, and those left out (`core.crossing_roads`), as
    crossing_roads.json holds them: None where there are none."""
    path = work.root / STAGE / CROSSING_ROADS
    return work.read_json(STAGE, CROSSING_ROADS) if path.exists() else None


def roundabouts(work: WorkDir, job: Job) -> list[dict]:
    """The roundabouts the section stage planned to draw (`core.roundabouts`), as roundabouts.json lists them: none for
    a job built without junctions, or with no roundabout on its route."""
    path = work.root / STAGE / ROUNDABOUTS
    if not route_stage.roundabouts(work, job) or not path.exists():
        return []
    return work.read_json(STAGE, ROUNDABOUTS)["roundabouts"]


def ramps(work: WorkDir, job: Job) -> list[dict]:
    """The slip ramps the section stage planned (`core.ramps`), and the shallow forks' gore stubs (`core.forks`), as
    ramps.json lists them: none for a job built without junctions, or with none on its route."""
    path = work.root / STAGE / RAMPS
    if not (route_stage.ramps(work, job) or route_stage.junctions(work, job)) or not path.exists():
        return []
    return work.read_json(STAGE, RAMPS)["ramps"]


def stubs(work: WorkDir, job: Job) -> list[dict]:
    """The side roads' stubs the section stage planned (`core.stubs`), as stubs.json lists them: none for a job built
    without junctions, or a section stage run before stubs."""
    path = work.root / STAGE / STUBS
    if not route_stage.junctions(work, job) or not path.exists():
        return []
    return work.read_json(STAGE, STUBS)["stubs"]


# The kerb plan's summary and its sources' credits (`core.kerbs`), for the later stages.
KERBS = "kerbs.json"
ATTRIBUTIONS = "attributions.json"


def _kerbs(job: Job, settings: Settings, region, work: WorkDir, c, runs: list[dict], half: np.ndarray,
           unsealed: np.ndarray) -> kerbs_core.Plan:
    """`_plan_kerbs`, and its summary in one INFO line of the build's log (`core.kerbs.summary_line`): the work folder
    holding kerbs.json is deleted when a production job ends, and its log is kept."""
    import logging

    p = _plan_kerbs(job, settings, region, work, c, runs, half, unsealed)
    logging.getLogger(__name__).info("%s", kerbs_core.summary_line(p.summary))
    return p


def _plan_kerbs(job: Job, settings: Settings, region, work: WorkDir, c, runs: list[dict], half: np.ndarray,
                unsealed: np.ndarray) -> kerbs_core.Plan:
    """The town stretches' kerbs and footpaths (`core.kerbs.plan`), from the region's building footprints and land
    parcels. Writes kerbs.json and the sources' credits (attributions.json). Without footprints (none for the region, or
    they can't be read) there's no telling a town, so nothing is kerbed; without parcels, footpaths take the standard
    widths. Either is logged as a warning, and the build goes on."""
    import logging
    from dataclasses import asdict

    log = logging.getLogger(__name__)
    (work.stage(STAGE) / ATTRIBUTIONS).unlink(missing_ok=True)
    (work.stage(STAGE) / KERBS).unlink(missing_ok=True)
    if job.kerbs == Kerbs.NONE:
        return kerbs_core.none_plan(half, half, {"kerbed_m": 0.0, "off": "job"})
    buildings, parcel_source = getattr(region, "buildings", None), getattr(region, "parcels", None)
    if buildings is None:
        return kerbs_core.none_plan(half, half, {"kerbed_m": 0.0, "off": "no_buildings"})
    import shapely

    from mapcreator.core import scenery

    centre = shapely.LineString(c["xy"])
    try:  # the scenery stage's own box, so its later read of the same footprints is a cache hit
        footprints = buildings(centre.buffer(scenery.BUILDING_REACH_M).bounds, settings)
    except Exception as e:  # noqa: BLE001 - any source failure costs the kerbs, not the track
        log.warning("no kerbs: building footprints unavailable (%s: %s)", type(e).__name__, e)
        return kerbs_core.none_plan(half, half, {"kerbed_m": 0.0, "off": "buildings_unavailable"})
    near = centre.buffer(kerbs_core.TOWN_RADIUS_M + kerbs_core.RAY_REACH_M)
    shapely.prepare(near)
    footprints = [p for p in footprints if near.intersects(p)]
    centroids = np.array([(p.centroid.x, p.centroid.y) for p in footprints]).reshape(-1, 2)
    blocked = np.zeros(len(c["s"]), bool)
    for key in ("bridge", "tunnel", "under"):
        if key in c:
            blocked |= np.asarray(c[key], bool)
    blocked |= _ring_and_ramp_stations(work, job, c)
    credits = []
    state = {"parcels": "none"}

    def parcels(boxes):
        if parcel_source is None:
            return None
        try:
            out = [p for box in boxes for p in parcel_source(box, settings)]
        except Exception as e:  # noqa: BLE001 - the footpaths take the standard widths
            log.warning("land parcels unavailable (%s: %s); footpaths at the standard widths", type(e).__name__, e)
            state["parcels"] = "unavailable"
            return None
        state["parcels"] = "ok"
        if getattr(region, "parcels_attribution", None) is not None:
            credits.append(region.parcels_attribution)
        return out

    council_source = getattr(region, "kerb_lines", None)

    def council(boxes):
        """The councils' kerb lines round the candidate stretches (`providers.councils.kerb_lines`). Without them (no
        mirror configured, or it can't be read) the rule decides, with a warning where a council has lines to give."""
        from mapcreator.providers.councils import kerb_lines as kl

        if council_source is None:
            return None
        expected = sorted({a for box in boxes for a in kl.areas_for(box)})
        try:
            got = council_source(boxes, settings)
        except Exception as e:  # noqa: BLE001 - the rule decides instead
            log.warning("councils' kerb lines unavailable (%s: %s); kerbs from the town-road rule",
                        type(e).__name__, e)
            state["council"] = "unavailable"
            return None
        if got is None:
            if expected:
                log.warning("no councils' kerb lines configured (kerb_lines), though the road is in %s's area; kerbs "
                            "from the town-road rule", " and ".join(expected))
            state["council"] = "not_configured" if expected else "none"
            return None
        state["council"] = "ok" if got.lines else "none_here"
        return got

    p = kerbs_core.plan(c["s"], c["xy"], c["heading"], c["run"], [r["tags"] for r in runs], half, half,
                        unsealed=unsealed, blocked=blocked, centroids=centroids, buildings=footprints,
                        parcels=parcels, reversed_runs=[bool(r.get("reversed")) for r in runs], council=council,
                        divided_runs=[bool(r.get("divided")) for r in runs])
    p.summary["parcels"] = state["parcels"]
    p.summary["council_lines"] = state.get("council", "none")
    if p.council:
        from mapcreator.providers.councils import kerb_lines as kl

        credits += [kl.ATTRIBUTIONS[name] for name in sorted(p.council) if name in kl.ATTRIBUTIONS]
    if p.any():
        if getattr(region, "buildings_attribution", None) is not None:
            credits.insert(0, region.buildings_attribution)
        work.write_json(STAGE, KERBS, {"version": kerbs_core.FORMAT_VERSION, **p.summary})
    if credits:
        work.write_json(STAGE, ATTRIBUTIONS, [asdict(a) for a in credits])
    return p


def _ring_and_ramp_stations(work: WorkDir, job: Job, c) -> np.ndarray:
    """The stations no kerb is drawn on for a roundabout or a slip ramp (`core.kerbs`): each roundabout's arc the route
    drives, each slip ramp's junction RAMP_BLOCK_M either side, and each shallow side road's (a fork's gore) as much.
    Found from the route stage's records, before any stub, ring or ramp is planned (they're planned on the widths the
    kerbs set)."""
    s, xy = np.asarray(c["s"], float), np.asarray(c["xy"], float)
    out = np.zeros(len(s), bool)
    route_xy = np.load(work.root / route_stage.STAGE / "route.npz")["xy"]
    route_s = cl.cumulative(route_xy)

    def station(m: float) -> float:
        p = np.array([np.interp(m, route_s, route_xy[:, 0]), np.interp(m, route_s, route_xy[:, 1])])
        return jn._station(xy, s, p, m)

    for r in route_stage.roundabouts(work, job):
        if r.get("route_from_m") is None:
            continue
        a, b = sorted((station(float(r["route_from_m"])), station(float(r["route_to_m"]))))
        out |= (s >= a) & (s <= b)
    shallow = [j for j in route_stage.junctions(work, job) if jn.gore_only(j)]
    for r in route_stage.ramps(work, job) + shallow:
        at = jn._station(xy, s, np.asarray(r["xy"], float), float(r["at_m"]))
        out |= np.abs(s - at) <= kerbs_core.RAMP_BLOCK_M
    return out


def _with_return(j: dict, c, runs: list[dict], kerbed: kerbs_core.Kerbs) -> dict:
    """A junction with its stub's flare radius (`flare_radius_m`, read by `core.stubs`) set to the kerb return where it
    leaves a kerbed stretch: RETURN_M, or RETURN_MAJOR_M where either road is secondary or above."""
    side = jn.SIDE_CODE[j["side"]]
    at = jn._station(c["xy"], c["s"], np.asarray(j["xy"], float), float(j["at_m"]))
    i = int(np.clip(np.searchsorted(c["s"], at), 0, len(c["s"]) - 1))
    if kerbed.k[side][i] < 0.99:
        return j
    main = runs[int(c["run"][i])]["tags"]
    return {**j, "flare_radius_m": kerbs_core.return_radius(main, {"highway": j.get("highway")})}
