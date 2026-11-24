"""Stage 6: static validation of the assembled track folder.

A build failure raises `ValidationFailed`, which the hosted service shows as an internal error. A bend
tighter than any real road raises `TightBend` instead: a message for the person who drew the route, saying
where the bend is, and whether they can ask for the road to be built anyway.
"""

from __future__ import annotations

import numpy as np
import shapely
from scipy.spatial import cKDTree

from mapcreator.config import Settings
from mapcreator.core import ai_line, bridges, dem_age, junctions, rail_and_paths, scenery, stubs, validate
from mapcreator.core import crossing_roads as crossing_core
from mapcreator.core import mesh as base_mesh
from mapcreator.core import ramps as ramps_core
from mapcreator.core import roundabouts as rb_core
from mapcreator.core.layout import AI_END_SHORT_M, RoadModel, layouts_for
from mapcreator.core.scene import Scene
from mapcreator.job import Job, Junctions
from mapcreator.pipeline import mesh_stage, profile_stage, route_stage, section_stage, track_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.regions import job_region

STAGE = "06_validate"


class ValidationFailed(RuntimeError):
    pass


class TightBend(ValidationFailed):
    """The route has a bend tighter than any real road. The message is written for the person who drew it."""

    def __init__(self, radius_m: float, along_m: float, lon: float, lat: float, overridable: bool):
        self.radius_m, self.along_m, self.lon, self.lat, self.overridable = radius_m, along_m, lon, lat, overridable
        where = f"about {along_m / 1000:.1f} km from the start"
        cause = ("That usually means it turns back on itself at a junction, or a waypoint is just off the road. "
                 "Try moving the waypoint near there.")
        if overridable:
            text = (f"This route has a bend tighter than any real hairpin: {radius_m:.1f} m radius, {where}. "
                    f"{cause} If the road really is this tight, you can generate it anyway.")
        else:
            text = f"This route turns almost on the spot ({radius_m:.1f} m radius, {where}), which no car can drive. {cause}"
        super().__init__(text)

    def detail(self) -> dict:
        return {"radius_m": round(self.radius_m, 1), "along_m": round(self.along_m, 1),
                "lon": round(self.lon, 6), "lat": round(self.lat, 6), "overridable": self.overridable}


def _where(job: Job, work: WorkDir, bend: validate.TightBend) -> tuple[float, float, float]:
    """Distance along the route as drawn (whichever layout the bend was found in), longitude, latitude."""
    frame = Scene.load(work.root / mesh_stage.STAGE).frame
    e, n, _ = frame.from_ac(np.array([bend.x, 0.0, bend.z]))
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    _, i = cKDTree(c["xy"]).query([float(e), float(n)])
    lon, lat = job_region(job.waypoints).to_lonlat().transform(float(e), float(n))
    return float(c["s"][i]), float(lon), float(lat)


def road_radius(job: Job, work: WorkDir) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each layout's centreline radius every metre over its AI line, with the AC x/z of each point: what the
    route check judges. The AI file's points are 5 m apart and its radius is the racing line's, so it isn't read
    for this."""
    frame = Scene.load(work.root / mesh_stage.STAGE).frame
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    sec = np.load(work.root / section_stage.STAGE / "section.npz")
    out = {}
    for lay in layouts_for(job.layouts.value):
        road = RoadModel.from_arrays(c["s"], c["xy"], c["z"], c["heading"], c["curvature"], c["grade"],
                                     sec["half_left"], sec["half_right"], sec["slope_left"], sec["slope_right"],
                                     lay.reverse, **track_stage._kerb_arrays(sec))
        d, r = ai_line.road_radius(road, end_short_m=AI_END_SHORT_M)
        xz = frame.to_ac(np.interp(d, road.d, road.e), np.interp(d, road.d, road.n), np.interp(d, road.d, road.z))
        out[lay.key or "main"] = (xz[:, [0, 2]], r)
    return out


def _roundabouts(job: Job, work: WorkDir) -> list[str]:
    """The roundabouts the route passes that weren't drawn whole (`core.roundabouts`), by name, with the reason, as
    the route stage recorded them."""
    path = work.root / route_stage.STAGE / route_stage.JUNCTIONS
    if job.junctions != Junctions.CLOSED or not path.exists():
        return []
    rings = work.read_json(route_stage.STAGE, route_stage.JUNCTIONS).get("roundabouts", [])
    drawn = {r["id"] for r in mesh_stage.built_roundabouts(work, job)}
    why: dict = {}
    for stage in (section_stage.STAGE, mesh_stage.STAGE):
        p = work.root / stage / section_stage.ROUNDABOUTS
        if p.exists():
            why |= {s["roundabout"]: s["reason"] for s in work.read_json(stage, section_stage.ROUNDABOUTS)["skipped"]
                    if "junction" not in s}
    return [f"{rb_core.label(r)}, {r['radius_m']:.0f} m radius ({why.get(r['id'], 'not drawn')})"
            for r in rings if r.get("on_route") and r["id"] not in drawn]


def roundabout_checks(job: Job, work: WorkDir) -> tuple[list[validate.RoundaboutCheck], list[validate.JunctionCheck]]:
    """The roundabouts drawn whole (`core.roundabouts`) as the validator sees them, in AC x/z; and their exits'
    stubs as side roads' junctions, for the stub and closure checks."""
    rings = mesh_stage.built_roundabouts(work, job)
    if not rings:
        return [], []
    frame = Scene.load(work.root / mesh_stage.STAGE).frame
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    sec = np.load(work.root / section_stage.STAGE / "section.npz")
    parts: dict = {}
    exits: dict = {}
    for key, chk in rb_core.checks(rings, c, sec, frame):
        if isinstance(key, tuple):
            parts.setdefault(key[1], []).append((key[2], chk))
        else:
            exits[key] = chk

    def to_ac(geom):
        return shapely.transform(geom, lambda p: frame.to_ac(p[:, 0], p[:, 1], np.zeros(len(p)))[:, [0, 2]])

    out, juncs = [], []
    for r in rings:
        name = r.get("label") or r.get("name") or "an unnamed roundabout"
        road = shapely.unary_union([shapely.Polygon(r["outline"]).buffer(0)]
                                   + [shapely.Polygon(st["outline"]).buffer(0) for st in r.get("exits") or []])
        island = shapely.Polygon(r["island"]["outline"]).buffer(0)
        clear = scenery.CLEAR_OF_VERGE_M + base_mesh.VERGE_M
        bare = island.difference(island.buffer(-clear)) if r["island"]["trees"] else island
        out.append(validate.RoundaboutCheck(f"{name} ({r['radius_m']:.0f} m radius)", tuple(parts.get(r["id"], [])),
                                            to_ac(road), to_ac(bare)))
        by_id = {j["id"]: j for j in r.get("exit_junctions") or []}
        for st in r.get("exits") or []:
            j = by_id.get(st["junction"], {})
            line = shapely.LineString(np.asarray(j.get("line") or st["rows"]["xy"], float))
            road = line.buffer(float(st["width_m"]) / 2, cap_style="flat")
            label = (f"{st.get('name') or 'an unnamed road'} ({st['highway']}, {st['closure']}) off the roundabout "
                     f"({name})")
            juncs.append(validate.JunctionCheck(label, shapely.Polygon(), to_ac(road), exits.get(st["junction"])))
    return out, juncs


def crossing_check(work: WorkDir) -> validate.CrossingCheck | None:
    """The roads crossing the route as the mesh stage drew them (`core.crossing_roads`), in AC x/z, for the validator:
    each piece of road as a stub's ribbon (its paved strip and points along its centreline), and how many decks. None
    where there are none."""
    path = work.root / mesh_stage.STAGE / mesh_stage.CROSSING_ROADS
    if not path.exists():
        return None
    doc = work.read_json(mesh_stage.STAGE, mesh_stage.CROSSING_ROADS)
    if not doc["roads"] and not doc["decks"]:
        return None
    frame = Scene.load(work.root / mesh_stage.STAGE).frame

    def ac(p):
        p = np.asarray(p, float).reshape(-1, 2)
        return frame.to_ac(p[:, 0], p[:, 1], np.zeros(len(p)))[:, [0, 2]]

    roads = []
    for r in doc["roads"]:
        xy = np.asarray(r["rows"]["xy"], float)
        for piece in r["pieces"]:
            a, b = piece["rows"]
            # Its whole ribbon, skirts and all, so the ground round them is checked for holes too.
            ground = shapely.Polygon(ac(piece["outline"])).buffer(0)
            label = f"{crossing_core.describe(r)} ({'under the route' if r['kind'] == 'under' else 'beyond its bridge'})"
            roads.append((label, validate.StubCheck(ground, ac(xy[a + 1:b]), np.zeros((0, 2, 2)))))
    return validate.CrossingCheck(tuple(roads), len(doc["decks"]))


def _stub_check(st: dict, c, sec, frame) -> validate.StubCheck:
    """A stub as the validator sees it (`validate.stub_check`), from the centreline's and cross-section's arrays."""
    return validate.stub_check(st, c["s"], c["xy"], c["heading"], sec["half_left"], sec["half_right"], frame)


def junction_checks(job: Job, work: WorkDir) -> list[validate.JunctionCheck]:
    """The side roads' junctions the route stage found, in AC x/z, for the validator: each mouth's carriageway out to
    MOUTH_BAND_M beyond the road's edge, and the side road's carriageway out to its end; and its stub, if it has
    one (`core.stubs`), whose flares and paved outline the mouth and the side road take in too."""
    found = route_stage.junctions(work, job)
    if not found:
        return []
    frame = Scene.load(work.root / mesh_stage.STAGE).frame
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    sec = np.load(work.root / section_stage.STAGE / "section.npz")
    s, xy = c["s"], c["xy"]
    plan = stubs.by_junction(mesh_stage.built_stubs(work, job))

    def to_ac(geom):
        return shapely.transform(geom, lambda p: frame.to_ac(p[:, 0], p[:, 1], np.zeros(len(p)))[:, [0, 2]])

    out = []
    for m in junctions.mouths(found, s, xy, c["heading"], sec["half_left"], sec["half_right"], stubs=plan):
        near = np.abs(s - m.station_m) <= m.s1 - m.s0 + 30.0
        edge = float(sec["half_left" if m.side == "L" else "half_right"][near].max())
        j = m.junction
        label = (f"{j.get('name') or 'an unnamed road'} ({j['highway']}, {j['closure']}) at "
                 f"{m.station_m / 1000:.2f} km on the {j['side']}")
        mouth = m.corridor().intersection(junctions.side_band(m, s, xy, edge + junctions.MOUTH_BAND_M))
        road = junctions.side_road(m, s, xy)
        stub = plan.get(j["id"])
        out.append(validate.JunctionCheck(label, to_ac(mouth), to_ac(road),
                                          _stub_check(stub, c, sec, frame) if stub is not None else None))
    return out


def _dem_age_warnings(work: WorkDir, covered: set | None = None) -> list[str]:
    """The profile stage's warnings for bridges the DEM doesn't hold (`core.dem_age`); none from a profile stage
    run before it judged them. Bridges at the `covered` crossings are left to the crossing roads' own warnings, which
    say the same and what was left out (`crossing_roads.dem_covered`)."""
    path = work.root / profile_stage.STAGE / dem_age.FILE
    if not path.exists():
        return []
    doc = work.read_json(profile_stage.STAGE, dem_age.FILE)
    if not covered:
        return list(doc.get("warnings") or [])
    bridges = [dem_age.Bridge(**b) for b in doc.get("bridges") or [] if not set(b.get("crossings") or []) & covered]
    return dem_age.Report(bridges, doc.get("not_bridges", 0), doc.get("unmatched", 0)).warnings()


def signed_decks(work: WorkDir) -> dict[str, float]:
    """The decks over the route drawn at the height their road is signed for, under 4.6 m (the maintainer, 18 Nov
    2026: real low bridges), by mesh-name prefix: the clearance each must keep (`validate.deck_clearance`)."""
    out: dict[str, float] = {}
    path = work.root / mesh_stage.STAGE / mesh_stage.CROSSING_ROADS
    if path.exists():
        for d in work.read_json(mesh_stage.STAGE, mesh_stage.CROSSING_ROADS).get("decks") or []:
            need = d.get("required_clearance_m")
            if need is not None and need < crossing_core.CLEARANCE_M:
                out[f"{crossing_core.DECK_MESH}{d['id']}_"] = float(need)
    path = work.root / mesh_stage.STAGE / mesh_stage.RAIL_AND_PATHS
    if path.exists():
        for r in work.read_json(mesh_stage.STAGE, mesh_stage.RAIL_AND_PATHS)["structures"]:
            need = r.get("required_clearance_m")
            if r["drawn"] and need is not None and need < rail_and_paths.CLEARANCE_M:
                prefix = "RAILBRIDGE" if r["structure"] == "railbridge" else "FOOTBRIDGE"
                out[f"{prefix}_{r['crossing']}_"] = float(need)
    return out


STRUCTURE_WORDS = {"footbridge": "footbridge", "railbridge": "rail bridge"}


def rail_and_paths_notes(work: WorkDir) -> list[str]:
    """The build's warnings about railways and paths crossing the route (`core.rail_and_paths`): each footbridge or
    rail bridge over it left out, and why (the maintainer's Q8: say why in the warnings), and none looked for when
    the data had none. A path or railway under the route that isn't drawn is only in `RAIL_AND_PATHS`."""
    out = []
    path = work.root / route_stage.STAGE / route_stage.CROSSING_LINES
    if path.exists():
        why = work.read_json(route_stage.STAGE, route_stage.CROSSING_LINES).get("unavailable")
        if why:
            out.append(f"Footbridges and rail bridges over the route weren't looked for: {why}.")
    path = work.root / mesh_stage.STAGE / mesh_stage.RAIL_AND_PATHS
    if not path.exists():
        return out
    for r in work.read_json(mesh_stage.STAGE, mesh_stage.RAIL_AND_PATHS)["structures"]:
        if r["drawn"] or r["structure"] not in STRUCTURE_WORDS:
            continue
        what = STRUCTURE_WORDS[r["structure"]]
        label = f"{r['name'] or 'An unnamed ' + what} ({what}) over the route at {r['at_m'] / 1000:.2f} km"
        if r.get("reason_code") == "dem":
            # The DEM can't place its deck: `core.dem_age`'s one wording, as for a road's bridge left out.
            out.append(dem_age.warning(label, coarse=not r.get("fine", True), then="it isn't drawn",
                                       detail=f"its deck would clear the road by {r['clearance_m']:.1f} m, under "
                                              f"{rail_and_paths.CLEARANCE_M} m; a footbridge's ramps and stairs "
                                              "aren't in the bare-earth data either"))
        else:
            out.append(f"{label} isn't drawn: {r['reason']}.")
    return out


def driveway_notes(work: WorkDir) -> list[str]:
    """The build's warning when driveways, service roads and tracks weren't looked for (`junctions.find_driveways`):
    an OSM index built before they were kept, or Overpass not answering. Their mouths then aren't kept clear."""
    path = work.root / route_stage.STAGE / route_stage.DRIVEWAYS
    if not path.exists():
        return []
    why = work.read_json(route_stage.STAGE, route_stage.DRIVEWAYS).get("unavailable")
    return [f"Driveways, service roads and tracks weren't looked for, so their mouths aren't kept clear: {why}."] \
        if why else []


def signal_notes(work: WorkDir) -> list[str]:
    """The build's warning when traffic signals weren't looked for (`core.signals.find`): an OSM index built before
    the tagged road nodes were kept, or Overpass not answering. The track then has no signals."""
    path = work.root / route_stage.STAGE / route_stage.SIGNALS
    if not path.exists():
        return []
    why = work.read_json(route_stage.STAGE, route_stage.SIGNALS).get("unavailable")
    return [f"Traffic signals weren't looked for, so none are drawn: {why}."] if why else []


def bridge_checks(work: WorkDir) -> list[validate.BridgeCheck]:
    """The route's own bridges as built (`core.bridges`), as the validator sees them."""
    records = mesh_stage.built_bridges(work)
    if not records:
        return []
    return bridges.checks(records, Scene.load(work.root / mesh_stage.STAGE).frame)


def ramp_checks(job: Job, work: WorkDir) -> list[validate.JunctionCheck]:
    """The slip ramps as built (`core.ramps`) as side roads' junctions, in AC x/z, for the validator: the mouth is the
    ramp's taper, gore and nose island beside the road, out to MOUTH_BAND_M beyond its edge; the side road, the ramp's
    paved outline and island; and its stub check (`ramps_core.check`)."""
    ramps = mesh_stage.built_ramps(work, job)
    if not ramps:
        return []
    frame = Scene.load(work.root / mesh_stage.STAGE).frame
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    sec = np.load(work.root / section_stage.STAGE / "section.npz")

    def to_ac(geom):
        return shapely.transform(geom, lambda p: frame.to_ac(p[:, 0], p[:, 1], np.zeros(len(p)))[:, [0, 2]])

    out = []
    for r in ramps:
        side = "L" if r["side"] == "left" else "R"
        a, b = r["mouth"]["rows"]
        rows = np.arange(a, b + 1)
        half = sec["half_left" if side == "L" else "half_right"][rows]
        sign = 1.0 if side == "L" else -1.0
        nrm = np.column_stack([-np.sin(c["heading"][rows]), np.cos(c["heading"][rows])]) * sign
        edge = c["xy"][rows] + nrm * half[:, None]
        band = shapely.Polygon(np.vstack([edge, (edge + nrm * junctions.MOUTH_BAND_M)[::-1]])).buffer(0)
        road = shapely.union(ramps_core.paved(r), shapely.Polygon(r["island"]["outline"]).buffer(0))
        what = "gore" if r.get("fork") else "slip ramp"  # a shallow fork's gore stub (`core.forks`), or a ramp
        label = (f"{r.get('name') or 'an unnamed ' + ('road' if r.get('fork') else 'ramp')} ({r['highway']}, "
                 f"{r['closure']}, {what}) at {r['station_m'] / 1000:.2f} km on the {r['side']}")
        ramp, _ = ramps_core.check(r, c, sec, frame)
        out.append(validate.JunctionCheck(label, to_ac(road.intersection(band)), to_ac(road), ramp))
    return out


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    ring_checks, ring_exits = roundabout_checks(job, work)
    rep = validate.track_folder(work.root / track_stage.STAGE / job.track_id, allow_tight_bends=job.allow_tight_bends,
                                road_radius=road_radius(job, work),
                                junctions=junction_checks(job, work) + ring_exits + ramp_checks(job, work),
                                roundabouts=ring_checks, crossing=crossing_check(work), bridges=bridge_checks(work),
                                signed_decks=signed_decks(work))
    rings = _roundabouts(job, work)
    if rings:
        rep.warnings.append(f"The route passes {len(rings)} roundabout{'s' if len(rings) != 1 else ''} "
                            f"not drawn whole ({', '.join(rings)}): the route's own arc is built, as before.")
    # Roads crossing the route (`core.crossing_roads`): those drawn, and each one left out with the reason. A bridge
    # left out because the elevation data doesn't hold it is warned of once, saying so.
    plan = section_stage.crossing_roads(work)
    rep.warnings.extend(_dem_age_warnings(work, crossing_core.dem_covered(plan)))
    built_path = work.root / mesh_stage.STAGE / mesh_stage.CROSSING_ROADS
    rep.warnings += crossing_core.warnings(
        plan, work.read_json(mesh_stage.STAGE, mesh_stage.CROSSING_ROADS) if plan and built_path.exists() else None)
    rep.warnings += rail_and_paths_notes(work)
    rep.warnings += driveway_notes(work)
    rep.warnings += signal_notes(work)
    work.write_json(STAGE, "report.json", {"ok": rep.ok, "passed": rep.passed, "failed": rep.failed,
                                           "route_failed": rep.route_failed, "warnings": rep.warnings})
    if rep.failed:
        raise ValidationFailed("; ".join(rep.failed + rep.route_failed))
    out: dict = {"ok": True, "checks": len(rep.passed)}
    if rep.bends:
        bend = min(rep.bends, key=lambda b: b.radius_m)
        along, lon, lat = _where(job, work, bend)
        if rep.route_failed:
            raise TightBend(bend.radius_m, along, lon, lat, bend.radius_m >= validate.FLOOR_RADIUS_M)
        out["tight_bend"] = {"radius_m": round(bend.radius_m, 1), "along_m": round(along, 1)}
    return out
