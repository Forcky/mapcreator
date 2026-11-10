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
from mapcreator.core import ai_line, junctions, stubs, validate
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
                                     lay.reverse)
        d, r = ai_line.road_radius(road, end_short_m=AI_END_SHORT_M)
        xz = frame.to_ac(np.interp(d, road.d, road.e), np.interp(d, road.d, road.n), np.interp(d, road.d, road.z))
        out[lay.key or "main"] = (xz[:, [0, 2]], r)
    return out


def _roundabouts(job: Job, work: WorkDir) -> list[str]:
    """The roundabouts the route passes, by name, as the route stage recorded them."""
    path = work.root / route_stage.STAGE / route_stage.JUNCTIONS
    if job.junctions != Junctions.CLOSED or not path.exists():
        return []
    rings = work.read_json(route_stage.STAGE, route_stage.JUNCTIONS).get("roundabouts", [])
    return [f"{r.get('name') or 'unnamed'}, {r['radius_m']:.0f} m radius" for r in rings if r.get("on_route")]


SEAM_OFFSET_M = 0.05  # the seam checks sample this far either side of where a stub's road meets another's


def _stub_check(st: dict, c, sec, frame) -> validate.StubCheck:
    """A stub (`core.stubs`) as the validator sees it, in AC x/z: its paved outline, points along its centreline, and
    pairs of points either side of its seams with the mainline's edge (away from the flares' tangent points, where
    the flare runs within centimetres of the edge) and with its own ribbon's first row."""
    def ac(p):
        p = np.asarray(p, float).reshape(-1, 2)
        return frame.to_ac(p[:, 0], p[:, 1], np.zeros(len(p)))[:, [0, 2]]

    side = junctions.SIDE_CODE[st["side"]]
    sign = 1.0 if side == "L" else -1.0
    half = sec["half_left" if side == "L" else "half_right"]
    a, b = st["mouth"]["rows"]
    rows = np.arange(a + 2, b - 1)
    nrm = np.column_stack([-np.sin(c["heading"][rows]), np.cos(c["heading"][rows])]) * sign
    edge = c["xy"][rows] + nrm * half[rows, None]
    pairs = [np.stack([edge - SEAM_OFFSET_M * nrm, edge + SEAM_OFFSET_M * nrm], axis=1)]
    r = st["rows"]
    k = int(st["ribbon_from_m"])
    hd = float(r["heading"][k])
    u, n = np.array([np.cos(hd), np.sin(hd)]), np.array([-np.sin(hd), np.cos(hd)])
    w = float(r["half_m"][k])
    across = np.asarray(r["xy"][k], float) + np.outer(np.arange(-0.9 * w, 0.9 * w + 1e-9, 0.5), n)
    pairs.append(np.stack([across - SEAM_OFFSET_M * u, across + SEAM_OFFSET_M * u], axis=1))
    seams = np.concatenate(pairs)
    centre = np.asarray(r["xy"], float)[1:-1:2]  # short of the end row, the mesh's very edge
    paved = shapely.Polygon(ac(st["outline"])).buffer(0)
    return validate.StubCheck(paved, ac(centre), ac(seams.reshape(-1, 2)).reshape(-1, 2, 2))


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
    plan = stubs.by_junction(section_stage.stubs(work, job))

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


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    rep = validate.track_folder(work.root / track_stage.STAGE / job.track_id, allow_tight_bends=job.allow_tight_bends,
                                road_radius=road_radius(job, work), junctions=junction_checks(job, work))
    rings = _roundabouts(job, work)
    if rings:
        rep.warnings.append(f"The route passes {len(rings)} roundabout{'s' if len(rings) != 1 else ''} "
                            f"({', '.join(rings)}): recorded in junctions.json, and built as before.")
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
