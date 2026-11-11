"""Closures on the side roads' stubs: each stub (`core.stubs`) closed off the stage with the props `core.closures`
builds, and police cars (`core.police_car`) parked at some of the major closures.

The maintainer's answers to the intersections research of 10 Nov 2026 this follows:
- **Level** (Q6), the junction's closure: **major** (trunk to tertiary and their links) gets course tape, barrier boards
  across the full width with ROAD CLOSED, and a police car if it's chosen (below); **minor** (unclassified,
  residential, living street, `road`) tape, one obstacle and a sign; **limited** (permit, forestry, destination) tape
  and a sign. No arrows of any kind (Q12).
- **A minor closure's obstacle** is barrier boards or straw bales, about half each, chosen per junction from its OSM
  node and way (`obstacle_for`), so a real junction gets the same one on every track that passes it. Bales go only
  where the whole row keeps SOLID_CLEAR_M from the road's edge (they're solid); elsewhere boards.
- **Layout** (Q8): `closures.layout` from the closure's origin, 100 m stubs: tape at 4 m, obstacle at 12 m, second
  tape and sign at 37 m, and the sign on the obstacle on a short stub.
- **Collision** (Q9, option 2 after the in-game look): straw bales and the police car are solid; boards, trestles,
  signs and tape are passable (`closures.SOLID`). Nothing solid within SOLID_CLEAR_M (5 m) of the road's edge.
- **Overshoot** (Q9): on the stub straight ahead where the route turns (`is_overshoot`), nothing solid in its first
  25 m (`closures.closure(overshoot=True)`).
- **Cars** (Q7): only at major closures, no two within CAR_SPACING_M (2 km) of each other along the route, at most
  CARS_MAX (6) a track. The biggest side roads get one first (`_car_rank`); one jurisdiction a track
  (`police_car.jurisdiction_for`).

**Where a closure stands.** A stub's `d` runs along its centreline from the road's edge line, so on a skewed junction
a line square across the stub 4 m in runs back onto the road on the acute side. The closure's **origin** (its local
z = 0) is the first place along the stub, in ORIGIN_STEP_M steps, where its tape, stake to stake, keeps TAPE_CLEAR_M
from the road's edge: d = 0 on a square junction, so the tape stands 4 m in as Q8 has it; further in on a skewed one.

**How wide.** Inside the throat the paved width is the stub and its flares, so every line of props spans the stub's
paved outline along the line square across it (`span`), and a metre more onto each verge (VERGE_SPAN_M), as the kit's
8 m for a 6 m road; not just the stub's own width, or cars would go round on the flare. A line never reaches nearer
than LINE_CLEAR_M to the road's edge. A span off centre (a flare on one side) moves the props' centre with it, through
the pose `closures.place` takes. Feet stand on the built ground (`TopSurface`: the road, stub and grass meshes).

**A car** stands in its closure's vehicle slot (`closures.Slot`): behind the boards, right of the middle, nose out and
turned in. It keeps the slot's distance in and heading; across the stub it stays within the span where it stands, so a
slot sized to boards across a wide throat still parks it on the side road. Where that would put it within
SOLID_CLEAR_M of the road's edge it tries the mirror image (left of the middle); failing both, that closure gets no
car and the next in line may.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import cKDTree

from mapcreator.core import closures, police_car
from mapcreator.core import mesh as mesh_core
from mapcreator.core import stubs as stubs_core
from mapcreator.core.scene import Z_SIGN, LocalFrame, Material, Mesh

VERGE_SPAN_M = 1.0  # every line of props reaches this far beyond the paved edge, onto the verge
TAPE_CLEAR_M = 3.5  # the tape, stake to stake, keeps this far from the road's edge: 4 m in on a square junction
LINE_CLEAR_M = 1.5  # no line of props reaches nearer the road's edge than this
SOLID_CLEAR_M = 5.0  # Q9: nothing solid (bales, a car) nearer the road's edge than this; the validator checks it
SOLID_SLACK_M = 0.25  # placement keeps this much more, so the check holds with float32 vertices
ORIGIN_STEP_M = 0.5
REACH_M = 160.0  # the road within this of a stub is what its clearances are measured against
CROSS_M = 80.0  # how far either side of the centreline a line of props is looked for in the outline
CAR_ON_STUB_M = mesh_core.VERGE_M  # a car's footprint stays within the stub's paved outline and its verge
OVERSHOOT_TURN_DEG = 30.0  # the route turns at least this much at the junction
OVERSHOOT_AHEAD_DEG = 30.0  # and the side road leaves within this of the way a car arrives (and nearer than the route)
CAR_SPACING_M = 2000.0  # Q7: at most one per 2 km
CARS_MAX = 6  # Q7: and six a track
CLASS_RANK = {"trunk": 0, "primary": 1, "secondary": 2, "tertiary": 3}  # the biggest side roads first; links after
GROUND_PREFIXES = ("1ROAD", "1GRAVEL", "1GRASS")  # the meshes props stand on


# -- the ground the props stand on -----------------------------------------------------------------------------

class TopSurface:
    """Heights of the built ground (road, stub, verge and terrain meshes) under region points: the highest triangle
    over each, NaN where none is. Only triangles inside `areas` ((minx, miny, maxx, maxy), region metres) are kept."""

    CELL_M = 2.0

    def __init__(self, meshes: list[Mesh], frame: LocalFrame, areas: list[tuple[float, float, float, float]]):
        self.frame = frame
        boxes = np.array([(x0 - frame.e0, Z_SIGN * (y1 - frame.n0), x1 - frame.e0, Z_SIGN * (y0 - frame.n0))
                          for x0, y0, x1, y1 in areas]).reshape(-1, 4)
        boxes[:, [1, 3]] = np.sort(boxes[:, [1, 3]], axis=1)
        tris = []
        for m in meshes:
            if not m.name.startswith(GROUND_PREFIXES) or not len(boxes):
                continue
            t = m.positions.astype(float)[np.asarray(m.triangles)]
            lo, hi = t[:, :, [0, 2]].min(axis=1), t[:, :, [0, 2]].max(axis=1)
            near = np.zeros(len(t), bool)
            for b in boxes:
                near |= (lo[:, 0] <= b[2]) & (hi[:, 0] >= b[0]) & (lo[:, 1] <= b[3]) & (hi[:, 1] >= b[1])
            tris.append(t[near])
        self.tri = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
        self.buckets: dict[tuple[int, int], list[int]] = {}
        if len(self.tri):
            xz = self.tri[:, :, [0, 2]]
            lo = np.floor(xz.min(axis=1) / self.CELL_M).astype(int)
            hi = np.floor(xz.max(axis=1) / self.CELL_M).astype(int)
            for k, (a, b, c, d) in enumerate(np.column_stack([lo, hi]).tolist()):
                for i in range(a, c + 1):
                    for j in range(b, d + 1):
                        self.buckets.setdefault((i, j), []).append(k)

    def __call__(self, e, n) -> np.ndarray:
        e, n = np.broadcast_arrays(np.asarray(e, float), np.asarray(n, float))
        shape = e.shape
        p = self.frame.to_ac(e.ravel(), n.ravel(), np.zeros(e.size))[:, [0, 2]]
        out = np.full(len(p), np.nan)
        cells = np.floor(p / self.CELL_M).astype(int)
        keys, inv = np.unique(cells, axis=0, return_inverse=True)
        for k, key in enumerate(keys.tolist()):
            ts = self.buckets.get(tuple(key))
            if not ts:
                continue
            idx = np.flatnonzero(inv.ravel() == k)
            t = self.tri[ts]
            a, b, c = t[:, 0, [0, 2]], t[:, 1, [0, 2]], t[:, 2, [0, 2]]
            q = p[idx][:, None, :]
            v0, v1, v2 = (c - a)[None], (b - a)[None], q - a[None]
            den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]
            good = np.abs(den) > 1e-12
            den = np.where(good, den, 1.0)
            u = (v2[..., 0] * v1[..., 1] - v1[..., 0] * v2[..., 1]) / den
            v = (v0[..., 0] * v2[..., 1] - v2[..., 0] * v0[..., 1]) / den
            inside = good & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
            h = t[None, :, 0, 1] + u * (t[None, :, 2, 1] - t[None, :, 0, 1]) + v * (t[None, :, 1, 1] - t[None, :, 0, 1])
            h = np.where(inside, h, -np.inf).max(axis=1)
            out[idx] = np.where(np.isfinite(h), h + self.frame.h0, np.nan)
        return out.reshape(shape)


# -- the rules ---------------------------------------------------------------------------------------------------

def obstacle_for(junction: dict) -> str:
    """A minor closure's obstacle, "boards" or "bales", about half each: from a hash of the junction's OSM node and
    way, so it never changes between builds, and a real junction gets the same one on every track."""
    key = f"{junction.get('node', junction.get('id'))}:{junction.get('way_id', '')}".encode()
    return "bales" if hashlib.sha256(key).digest()[0] & 1 else "boards"


def seed_for(junction: dict) -> int:
    """The closure's seed (where the tape's tag hangs, the bales' sizes), from the same hash."""
    key = f"{junction.get('node', junction.get('id'))}:{junction.get('way_id', '')}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[1:5], "little")


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def is_overshoot(junction: dict, stub: dict, s: np.ndarray, heading: np.ndarray, reverses=(False,)) -> bool:
    """Whether the stub is the road straight ahead where the route turns (Q9), for a direction the track is driven
    in (`reverses`: whether each layout runs backwards): the route turns at least OVERSHOOT_TURN_DEG there, and the side
    road leaves within OVERSHOOT_AHEAD_DEG of the way a car arrives, nearer straight on than the route goes.

    The angles are junctions.json's: the side road's (`angle_deg`, + left) from the route's mean direction through
    the node, and the route's turn (`route_turn_deg`); the turn's sign comes from the centreline."""
    turn = float(junction.get("route_turn_deg") or 0.0)
    if turn < OVERSHOOT_TURN_DEG:
        return False
    at = float(stub["station_m"])
    hd = np.unwrap(np.asarray(heading, float))
    signed = math.copysign(turn, math.degrees(np.interp(at + 15.0, s, hd) - np.interp(at - 15.0, s, hd)))
    angle = float(junction["angle_deg"])
    for reverse in reverses:
        arrive = signed / 2 + 180.0 if reverse else -signed / 2  # the way a car arrives, from the mean direction
        ahead = abs(_wrap_deg(angle - arrive))
        if ahead <= OVERSHOOT_AHEAD_DEG and ahead < turn:
            return True
    return False


def _car_rank(stub: dict) -> tuple:
    """Biggest side roads first: by class (trunk to tertiary, a link after its road), then width, then length."""
    highway = str(stub.get("highway") or "")
    base = highway.removesuffix("_link")
    return (CLASS_RANK.get(base, len(CLASS_RANK)) + (0.5 if highway.endswith("_link") else 0.0),
            -float(stub.get("width_m") or 0.0), -float(stub.get("length_m") or 0.0), float(stub["station_m"]))


# -- geometry along a stub ---------------------------------------------------------------------------------------

def _road_near(c, section, centre: np.ndarray, tree: cKDTree) -> shapely.Geometry:
    """The road's carriageway within REACH_M of a point, other parts of the route passing by included."""
    s_idx = np.array(sorted(tree.query_ball_point(centre, REACH_M)), dtype=int)
    if len(s_idx) < 2:
        return shapely.Polygon()
    xy, hd = np.asarray(c["xy"], float), np.asarray(c["heading"], float)
    nrm = np.column_stack([-np.sin(hd[s_idx]), np.cos(hd[s_idx])])
    left = xy[s_idx] + nrm * np.asarray(section["half_left"], float)[s_idx, None]
    right = xy[s_idx] - nrm * np.asarray(section["half_right"], float)[s_idx, None]
    keep = np.flatnonzero(np.diff(s_idx) == 1)  # consecutive rows only
    quads = shapely.polygons(np.stack([left[keep], left[keep + 1], right[keep + 1], right[keep]], axis=1))
    return shapely.union_all(shapely.make_valid(quads))


@dataclass
class _Stub:
    """A stub, as its closure needs it."""
    record: dict
    outline: shapely.Geometry
    road: shapely.Geometry  # the carriageway near it
    pose: object  # stubs.pose
    half: float

    def frame_at(self, d: float) -> tuple[np.ndarray, np.ndarray, float]:
        e, n, _, hd = self.pose(d)
        return np.array([e, n]), np.array([-math.sin(hd), math.cos(hd)]), hd

    def cross(self, d: float) -> tuple[float, float]:
        """The paved outline's extent along the line square across the stub `d` in, as lateral offsets from its
        centreline (+ left): the piece of that line inside the outline that holds the centreline."""
        p, nl, _ = self.frame_at(d)
        line = shapely.LineString([p - CROSS_M * nl, p + CROSS_M * nl])
        centre = shapely.Point(p)
        pieces = [g for g in shapely.get_parts(line.intersection(self.outline)) if g.distance(centre) < 0.02]
        if not pieces:
            return -self.half, self.half
        lat = (np.asarray(pieces[0].coords) - p) @ nl
        return float(min(lat.min(), -1e-3)), float(max(lat.max(), 1e-3))

    def clearance(self, d: float, lo: float, hi: float) -> float:
        p, nl, _ = self.frame_at(d)
        return float(shapely.LineString([p + lo * nl, p + hi * nl]).distance(self.road)) if not self.road.is_empty \
            else math.inf

    def span(self, d: float, clear: float = LINE_CLEAR_M) -> tuple[float, float, bool]:
        """Where a line of props `d` in reaches: the outline's extent and VERGE_SPAN_M more each side, less any part
        within `clear` of the road's edge (stakes included). Returns (lo, hi, whole): whole is False where the road
        cut it short."""
        lo, hi = self.cross(d)
        lo, hi = lo - VERGE_SPAN_M, hi + VERGE_SPAN_M
        stake = closures.STAKE_OUT_M + closures.STAKE_W
        if self.road.is_empty or self.clearance(d, lo - stake, hi + stake) >= clear:
            return lo, hi, True
        p, nl, _ = self.frame_at(d)
        seg = shapely.LineString([p + (lo - stake) * nl, p + (hi + stake) * nl])
        left = seg.difference(self.road.buffer(clear, quad_segs=4))
        pieces = [g for g in shapely.get_parts(left) if g.distance(shapely.Point(p)) < 0.02]
        if not pieces:
            return -self.half, self.half, False
        lat = (np.asarray(pieces[0].coords) - p) @ nl
        return float(min(lat.min() + stake, -closures.MIN_WIDTH_M / 2)), \
            float(max(lat.max() - stake, closures.MIN_WIDTH_M / 2)), False

    def origin(self) -> float:
        """The closure's origin: the first d (ORIGIN_STEP_M apart) where the tape keeps TAPE_CLEAR_M from the road."""
        stake = closures.STAKE_OUT_M + closures.STAKE_W
        last = float(self.record["length_m"]) - closures.TAPE_M - 1.0
        d0 = 0.0
        while d0 <= last:
            lo, hi = self.cross(d0 + closures.TAPE_M)
            if self.clearance(d0 + closures.TAPE_M, lo - VERGE_SPAN_M - stake, hi + VERGE_SPAN_M + stake) \
                    >= TAPE_CLEAR_M:
                return d0
            d0 += ORIGIN_STEP_M
        return 0.0


# -- placing -----------------------------------------------------------------------------------------------------

@dataclass
class Placement:
    """The closures and police cars for a track, ready to append to the mesh stage's scene."""
    meshes: list[Mesh]  # merged: CLOSURE_k, 1WALL_CLOSURE_k, POLICE_<CODE>_k, 1WALL_POLICE_k
    materials: dict[str, Material]
    textures: dict[str, object]  # texture name -> PIL image
    records: list[dict]  # one per stub
    stats: dict = field(default_factory=dict)


def _ground_fn(lines: dict[str, float], centre_of, top: TopSurface):
    """`closures.closure`'s ground(x, z): the built ground's height under each local point, relative to the pose's
    height at that point's element line. `closures.place` puts each element rigidly at its line's pose, so a point is
    found in the frame of the line nearest it (`lines`: element -> local z)."""
    zs = np.array(sorted(lines.values())) if lines else np.zeros(1)

    def ground(x, z):
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        line = zs[np.argmin(np.abs(z.ravel()[:, None] - zs[None, :]), axis=1)]
        out = np.empty(line.shape)
        for zl in np.unique(line):
            k = line == zl
            e, n, h, hd = centre_of(float(zl))
            r, f = np.array([math.sin(hd), -math.cos(hd)]), np.array([math.cos(hd), math.sin(hd)])
            px, pz = x.ravel()[k], z.ravel()[k] - zl
            g = top(e + px * r[0] + pz * f[0], n + px * r[1] + pz * f[1])
            out[k] = np.where(np.isfinite(g), g, h) - h
        return out.reshape(x.shape)

    return ground


def _car_pose(st: _Stub, slot: closures.Slot, origin: float, centre_lat: float, mirror: bool):
    """The car's footprint middle (e, n) and heading (radians) for a closure's slot, kept within the span where it
    stands (see the module notes)."""
    a = math.radians(closures.VEHICLE_ANGLE_DEG)
    half_l, half_w = slot.size[0] / 2, slot.size[1] / 2
    reach_x = half_l * math.sin(a) + half_w * math.cos(a)
    d = origin + slot.z
    x, _, _ = slot.position
    fx, _, fz = slot.forward
    lat = centre_lat - x  # local +x is to the right; lateral offsets are + left
    if mirror:
        lat, fx = -lat, -fx
    lo, hi, _ = st.span(d)
    lat = (lo + hi) / 2 if hi - lo < 2 * reach_x else float(np.clip(lat, lo + reach_x, hi - reach_x))
    p, nl, hd = st.frame_at(d)
    en = p + nl * lat
    r, f = np.array([math.sin(hd), -math.cos(hd)]), np.array([math.cos(hd), math.sin(hd)])
    fen = r * fx + f * fz
    return float(en[0]), float(en[1]), math.atan2(fen[1], fen[0])


def _place_car(car: police_car.PoliceCar, e: float, n: float, heading: float, top: TopSurface, frame: LocalFrame,
               index: int) -> list[Mesh]:
    """A copy of the car on the built ground: its height and tilt from the ground under its four wheels."""
    f = np.array([math.cos(heading), math.sin(heading)])
    r = np.array([math.sin(heading), -math.cos(heading)])
    track, base = car.body.width / 2 - 0.1, (car.body.axles[0] - car.body.axles[1]) / 2
    centre_z = (car.body.axles[0] + car.body.axles[1]) / 2
    wheels = [np.array([e, n]) + f * (centre_z + sz * base) + r * sx * track for sz in (1, -1) for sx in (1, -1)]
    wheels = np.array(wheels)
    g = top(wheels[:, 0], wheels[:, 1])
    if not np.isfinite(g).all():
        g = np.where(np.isfinite(g), g, np.nanmean(g) if np.isfinite(g).any() else frame.h0)
    # The ground's plane through the four contact points (least squares), in region axes.
    a = np.column_stack([wheels - [e, n], np.ones(4)])
    (gx, gy, h0), *_ = np.linalg.lstsq(a, g, rcond=None)
    up_en = np.array([-gx, -gy, 1.0])
    up = np.array([up_en[0], up_en[2], Z_SIGN * up_en[1]])
    fwd = np.array([f[0], 0.0, Z_SIGN * f[1]])
    pos = frame.to_ac(e, n, h0)
    return police_car.place(car, pos, fwd, up, index=index)


def _xy(meshes: list[Mesh], frame: LocalFrame) -> np.ndarray:
    if not meshes:
        return np.zeros((0, 2))
    p = np.concatenate([m.positions for m in meshes]).astype(float)
    e, n, _ = frame.from_ac(p)
    return np.column_stack([e, n])


def place(stubs: list[dict], junctions: list[dict], c, section, frame: LocalFrame, top: TopSurface, *,
          jurisdiction: str = police_car.DEFAULT_JURISDICTION, reverses=(False,),
          parents: dict[str, tuple[dict, dict]] | None = None, extra_road: shapely.Geometry | None = None) -> Placement:
    """Close every stub (stubs.json's `stubs`; a junction stubs.json skipped has none, so gets nothing) and park the
    police cars. `junctions` are junctions.json's, `c` and `section` the centreline's and cross-section's arrays,
    `top` the built ground, `jurisdiction` the track's police (`police_car.jurisdiction_for`), `reverses` whether each
    layout runs backwards (for the overshoot rule).

    Roundabouts (`core.roundabouts`): a stub whose `parent` names a ring in `parents` (that ring's arrays, as `c` and
    `section` are the road's) leaves the ring, so its clearances are measured from the ring's carriageway as well as
    the road's; `extra_road` (the rings' paved outlines) counts as road for every stub's clearances."""
    by_id = {j["id"]: j for j in junctions}
    s, heading = np.asarray(c["s"], float), np.asarray(c["heading"], float)
    tree = cKDTree(np.asarray(c["xy"], float))
    parent_trees = {key: cKDTree(np.asarray(pc["xy"], float)) for key, (pc, _) in (parents or {}).items()}

    def road_for(rec, centre):
        road = _road_near(c, section, centre, tree)
        parent = rec.get("parent")
        if parents and parent in parents:
            pc, ps = parents[parent]
            road = shapely.union(road, _road_near(pc, ps, centre, parent_trees[parent]))
        if extra_road is not None and not extra_road.is_empty:
            road = shapely.union(road, extra_road)
        return road

    kits, records, closure_meshes, candidates = [], [], [], []
    clear = SOLID_CLEAR_M + SOLID_SLACK_M
    for k, rec in enumerate(sorted(stubs, key=lambda r: (r["station_m"], r["junction"]))):
        j = by_id.get(rec["junction"], {"id": rec["junction"]})
        outline = shapely.Polygon(rec["outline"]).buffer(0)
        st = _Stub(rec, outline, road_for(rec, np.asarray(outline.centroid.coords[0])),
                   stubs_core.pose(rec), float(rec["rows"]["half_m"][0]))
        level = rec.get("closure") if rec.get("closure") in closures.LEVELS else "minor"
        overshoot = is_overshoot(j, rec, s, heading, reverses) if "angle_deg" in j else False
        origin = st.origin()
        length = float(rec["length_m"]) - origin
        lines = closures.layout(level, length=length, overshoot=overshoot)
        spans = {name: st.span(origin + z) for name, z in lines.items()}
        chosen = obstacle_for(j) if level == "minor" else "boards"
        obstacle, note = chosen, None
        if level == "minor" and chosen == "bales" and "obstacle" in lines:
            lo, hi, whole = spans["obstacle"]
            half_depth = closures.BALE[2] / 2 + closures.BALE_STAGGER_M
            z = origin + lines["obstacle"]
            near = min(st.clearance(z + dz, lo, hi) for dz in (-half_depth, half_depth))
            if not whole or near < clear:
                obstacle, note = "boards", f"bales within {SOLID_CLEAR_M:.0f} m of the road: boards instead"

        def centre_of(z, st=st, spans=spans, lines=lines, origin=origin):
            name = min(lines, key=lambda nm: abs(lines[nm] - z)) if lines else None
            lo, hi, _ = spans[name] if name else (-st.half, st.half, True)
            p, nl, hd = st.frame_at(origin + z)
            q = p + nl * (lo + hi) / 2
            h = float(top(q[0], q[1]))
            return float(q[0]), float(q[1]), h if np.isfinite(h) else float(st.pose(origin + z)[2]), hd

        def width(z, spans=spans, lines=lines):
            name = min(lines, key=lambda nm: abs(lines[nm] - z))
            lo, hi, _ = spans[name]
            return hi - lo

        ground = _ground_fn(lines, centre_of, top)
        kit = closures.closure(level, width, overshoot=overshoot, length=length, obstacle=obstacle, ground=ground,
                               seed=seed_for(j))
        placed = closures.place(kit, centre_of, frame, index=k)
        closure_meshes += placed.meshes + placed.colliders
        first = float(rec.get("ribbon_from_m", 0.0))
        record = {"junction": rec["junction"], "name": rec.get("name"), "highway": rec.get("highway"),
                  "station_m": rec["station_m"], "side": rec.get("side"), "level": level,
                  "obstacle": kit.obstacle, "overshoot": overshoot, "origin_m": round(origin, 1),
                  "elements": {name: {"d_m": round(origin + z, 1), "width_m": round(spans[name][1] - spans[name][0], 2),
                                      "offset_m": round((spans[name][0] + spans[name][1]) / 2, 2),
                                      "in_throat": origin + z < first, "cut_by_road": not spans[name][2]}
                               for name, z in lines.items()},
                  "car": False}
        if note:
            record["note"] = note
        records.append(record)
        if level == "major" and kit.slots:
            ob = spans.get("obstacle")
            centre_lat = (ob[0] + ob[1]) / 2 if ob else 0.0
            candidates.append((rec, st, kit.slots[0], origin, centre_lat, record, k))

    # Police cars: the biggest side roads first, spaced out, within the cap; each where it clears the road.
    car = police_car.police_car(jurisdiction) if candidates else None
    cars, chosen = [], []
    for rec, st, slot, origin, centre_lat, record, k in sorted(candidates, key=lambda cand: _car_rank(cand[0])):
        if len(chosen) >= CARS_MAX:
            break
        if any(abs(rec["station_m"] - other) < CAR_SPACING_M for other in chosen):
            continue
        for mirror in (False, True):
            e, n, hd = _car_pose(st, slot, origin, centre_lat, mirror)
            copy = _place_car(car, e, n, hd, top, frame, len(cars))
            xy = _xy([m for m in copy if not m.renderable], frame)
            pts = shapely.points(xy)
            far = st.road.is_empty or float(shapely.distance(st.road, pts).min()) >= clear
            on_stub = bool(shapely.contains_xy(st.outline.buffer(CAR_ON_STUB_M), xy[:, 0], xy[:, 1]).all())
            if far and on_stub:
                cars += copy
                chosen.append(rec["station_m"])
                record["car"] = {"mirrored": mirror, "d_m": round(origin + slot.z, 1)}
                break

    meshes = closures.merge(closure_meshes) if closure_meshes else []
    materials = dict(closures.materials()) if closure_meshes else {}
    textures = {closures.TEXTURE: closures.atlas().image} if closure_meshes else {}
    if cars:
        meshes += police_car.merge(cars)
        for name, mat in car.materials.items():
            materials[name] = Material(name, mat.texture, properties=dict(car.kn5_properties[name]),
                                       night_diffuse=car.night_diffuse.get(name))
        textures |= car.textures
    visual = [m for m in meshes if m.renderable]
    solid = [m for m in meshes if not m.renderable]
    levels: dict[str, int] = {}
    for r in records:
        key = r["level"] + (f" ({r['obstacle']})" if r["level"] == "minor" and r["obstacle"] else "")
        levels[key] = levels.get(key, 0) + 1
    stats = {"closures": len(records), "by_level": dict(sorted(levels.items())),
             "overshoot": sum(r["overshoot"] for r in records),
             "bales_to_boards": sum("note" in r for r in records),
             "moved_in": sum(r["origin_m"] > 0 for r in records),
             "cars": len(chosen), "jurisdiction": jurisdiction if chosen else None,
             "majors_without_car": sum(r["level"] == "major" and not r["car"] for r in records),
             "meshes": len(meshes), "vertices": int(sum(len(m.positions) for m in visual)),
             "triangles": int(sum(len(m.triangles) for m in visual)),
             "collider_vertices": int(sum(len(m.positions) for m in solid))}
    return Placement(meshes, materials, textures, records, stats)


def areas(stubs: list[dict], margin: float = 10.0) -> list[tuple[float, float, float, float]]:
    """Each stub's outline's bounds, `margin` metres larger: where `TopSurface` needs the ground."""
    out = []
    for rec in stubs:
        xy = np.asarray(rec["outline"], float)
        out.append((float(xy[:, 0].min() - margin), float(xy[:, 1].min() - margin),
                    float(xy[:, 0].max() + margin), float(xy[:, 1].max() + margin)))
    return out
