"""Power lines near the road: towers, poles and the wires between them, as KN5 meshes.

A provider hands over `Line`s: polylines in the region's CRS. On an overhead power line
every vertex is a support, because the wires run straight from one support to the next.
`plan` picks the spans in view of the road and merges supports that lines share, and
`meshes` builds:
- a lattice tower (transmission, 66 kV and up) or a timber pole (anything lower) at each
  support, turned to the line;
- the wires: the phase conductors of each circuit, plus an earth wire over a tower line,
  sagging between supports but kept clear of the ground;
- invisible `1WALL_POWER` colliders on the supports a car can reach from the road.

Only mapped lines get supports: where no line is mapped there are none. A mapped line
whose vertices are further apart than a span can be (sparsely drawn) gets evenly spaced
supports along it (`densify`), and the build reports how many.

Wires sway in the wind under Custom Shaders Patch through their material (`CABLE`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import cKDTree

from mapcreator.core.scene import MAX_VERTICES, Z_SIGN, LocalFrame, Mesh

TRANSMISSION, DISTRIBUTION = "transmission", "distribution"
METAL, TIMBER, CABLE = "power_metal", "power_timber", "power_cable"

SUPPORT_REACH_M = 700.0  # supports stand within this distance of the centreline; the stage loads the DEM under them
VIEW_M = 300.0  # a span is kept if it comes this close to the centreline (the terrain mesh's reach), so a line
# crossing the road keeps its towers on both sides even when one stands beyond the terrain
ON_ROAD_MARGIN_M = 0.5  # a support closer to the centreline than the road's half-width plus this is misplaced
COLLIDE_BEYOND_EDGE_M = 15.0  # supports within this distance of the road's edge get a collider
MERGE_M = 1.0  # supports of different lines closer than this are one support
MAX_SPAN_M = {TRANSMISSION: 650.0, DISTRIBUTION: 120.0}  # `densify`: longer gaps between vertices get supports
TYPICAL_SPAN_M = {TRANSMISSION: 380.0, DISTRIBUTION: 75.0}
CATENARY_M = {TRANSMISSION: 1600.0, DISTRIBUTION: 700.0}  # sag = span^2 / (8 C)
CLEARANCE_M = {TRANSMISSION: 8.0, DISTRIBUTION: 5.5}  # the lowest wire stays this far above the ground
MIN_SAG_M = 0.3
WIRE_STEP_M = 10.0
WIRE_RADIUS_M = {TRANSMISSION: 0.035, DISTRIBUTION: 0.02}
DEFAULT_KV = 110.0  # a transmission line of unknown voltage


@dataclass
class Line:
    xy: np.ndarray  # (n, 2) region CRS; every vertex is a support
    kind: str  # TRANSMISSION or DISTRIBUTION
    voltage_kv: float | None = None
    circuits: int = 1
    ref: str = ""  # the source's id or name, for the summary and power.geojson
    inferred: np.ndarray | None = None  # (n,) True where `densify` placed a support the data doesn't have

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=float).reshape(-1, 2)
        if self.inferred is None:
            self.inferred = np.zeros(len(self.xy), dtype=bool)


def densify(line: Line) -> Line:
    """Add evenly spaced supports where the line's vertices are further apart than a span can be."""
    limit, typical = MAX_SPAN_M[line.kind], TYPICAL_SPAN_M[line.kind]
    pts, inferred = [line.xy[:1]], [line.inferred[:1]]
    for a, b, flag in zip(line.xy[:-1], line.xy[1:], line.inferred[1:]):
        gap = float(np.hypot(*(b - a)))
        if gap > limit:
            k = int(np.ceil(gap / typical))
            t = np.arange(1, k)[:, None] / k
            pts.append(a + t * (b - a))
            inferred.append(np.ones(k - 1, dtype=bool))
        pts.append(b[None])
        inferred.append(np.array([flag]))
    return Line(np.vstack(pts), line.kind, line.voltage_kv, line.circuits, line.ref, np.concatenate(inferred))


def parse_kv(value: str | None) -> float | None:
    """The highest voltage in an OSM `voltage` tag (volts, ';'-separated) or a name like '220 kV', in kV."""
    if not value:
        return None
    import re

    if "kv" in value.lower():
        found = [float(v) for v in re.findall(r"(\d+(?:\.\d+)?)\s*kV", value, flags=re.I)]
        return max(found) if found else None
    volts = []
    for part in value.replace(",", ";").split(";"):
        try:
            volts.append(float(part.strip()))
        except ValueError:
            continue
    return max(volts) / 1000.0 if volts else None


# ----- which supports and spans ---------------------------------------------

@dataclass
class Support:
    xy: np.ndarray  # (2,)
    design: str  # "tower" or "pole"
    template: str  # key into TEMPLATES
    along: np.ndarray  # (2,) unit: the line's direction here (the bisector at a bend)
    distance_m: float  # from the centreline
    inferred: bool = False


@dataclass
class Span:
    a: int  # support indices
    b: int
    kind: str
    template: str  # the line's own support design, which sets where its wires attach
    phases: int  # conductors (3 per circuit)
    earth: bool
    ref: str


@dataclass
class Plan:
    supports: list[Support] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    misplaced: int = 0  # supports dropped for standing on the road

    def summary(self) -> dict:
        lengths = [float(np.hypot(*(self.supports[s.b].xy - self.supports[s.a].xy))) for s in self.spans]
        return {
            "lines": len({s.ref for s in self.spans}),
            "towers": sum(s.design == "tower" for s in self.supports),
            "poles": sum(s.design == "pole" for s in self.supports),
            "inferred_supports": sum(s.inferred for s in self.supports),
            "spans": len(self.spans),
            "km": round(sum(lengths) / 1000.0, 2),
            "misplaced_supports": self.misplaced,
        }


def template_for(kind: str, voltage_kv: float | None) -> str:
    if kind == TRANSMISSION:
        kv = voltage_kv or DEFAULT_KV
        if kv >= 300:
            return "tower_330"
        if kv >= 200:
            return "tower_220"
        if kv >= 100:
            return "tower_110"
        if kv >= 66:
            return "tower_66"
        return "pole_tall"
    return "pole"


def plan(lines: list[Line], centre_xy: np.ndarray, road_half_m: float, *, reach_m: float = SUPPORT_REACH_M,
         view_m: float = VIEW_M) -> Plan:
    """The supports and spans to build: spans with both supports within `reach_m` of the centreline that come
    within `view_m` of it. Supports that lines share are merged, and so are duplicate spans."""
    centre = shapely.LineString(centre_xy)
    tree = cKDTree(_densified(centre_xy, 2.0))
    out = Plan()
    keys: dict[tuple[int, int], int] = {}
    spans: dict[tuple[int, int], Span] = {}
    rank = {name: i for i, name in enumerate(TEMPLATES)}
    for line in lines:
        if len(line.xy) < 2:
            continue
        d, _ = tree.query(line.xy)
        misplaced = d < road_half_m + ON_ROAD_MARGIN_M
        ok = (d <= reach_m) & ~misplaced
        segs = shapely.linestrings(np.stack([line.xy[:-1], line.xy[1:]], axis=1))
        near = shapely.distance(segs, centre) <= view_m
        keep = ok[:-1] & ok[1:] & near & (np.hypot(*np.diff(line.xy, axis=0).T) > 0.5)
        out.misplaced += int(misplaced.sum())
        template = template_for(line.kind, line.voltage_kv)
        design = "tower" if template.startswith("tower") else "pole"
        circuits = max(1, min(int(line.circuits or 1), 2 if design == "tower" else 1))
        for k in np.flatnonzero(keep):
            ends = []
            direction = line.xy[k + 1] - line.xy[k]
            direction = direction / np.linalg.norm(direction)
            for v in (k, k + 1):
                key = (int(round(line.xy[v, 0] / MERGE_M)), int(round(line.xy[v, 1] / MERGE_M)))
                i = keys.get(key)
                if i is None:
                    i = keys[key] = len(out.supports)
                    out.supports.append(Support(line.xy[v].copy(), design, template, np.zeros(2), float(d[v]),
                                                bool(line.inferred[v])))
                s = out.supports[i]
                if rank[template] > rank[s.template]:  # the bigger support stands where lines share one
                    s.template, s.design = template, design
                s.along += direction if float(s.along @ direction) >= 0 else -direction
                ends.append(i)
            span_key = (min(ends), max(ends))
            span = Span(ends[0], ends[1], line.kind, template, 3 * circuits, design == "tower", line.ref)
            if span_key not in spans or span.phases > spans[span_key].phases:
                spans[span_key] = span
    for s in out.supports:
        n = float(np.linalg.norm(s.along))
        s.along = s.along / n if n > 1e-9 else np.array([1.0, 0.0])
    out.spans = list(spans.values())
    return out


def _densified(xy: np.ndarray, step: float) -> np.ndarray:
    seg = np.diff(xy, axis=0)
    n = np.maximum(np.ceil(np.hypot(*seg.T) / step).astype(int), 1)
    parts = [xy[i] + np.arange(k)[:, None] / k * seg[i] for i, k in enumerate(n)]
    return np.vstack(parts + [xy[-1:]])


# ----- support designs -------------------------------------------------------

@dataclass(frozen=True)
class Template:
    """A support in its own axes (x across the line, to the right; y up; z along it), as round members."""

    p0: np.ndarray  # (m, 3)
    p1: np.ndarray  # (m, 3)
    radius: np.ndarray  # (m,)
    sides: int
    attach: np.ndarray  # (k, 3): phases of circuit 1 (right), then circuit 2 (left), then the earth wire if any
    base_half_m: float  # half the base's width, for its collider
    material: str


def _tower(height: float, arm: float, insulator: float) -> Template:
    b, w = 0.11 * height, 0.03 * height + 0.4  # half-widths at the base and from the waist up
    levels = np.array([0.0, 0.12, 0.25, 0.38, 0.5, 0.6, 0.7, 0.8, 0.9]) * height
    half = np.interp(levels, [0.0, 0.6 * height, height], [b, w, w])
    corners = np.array([(1, 1), (1, -1), (-1, -1), (-1, 1)], dtype=float)
    p0, p1, r = [], [], []

    def member(a, c, radius):
        p0.append(a)
        p1.append(c)
        r.append(radius)

    ring = [np.column_stack([corners[:, 0] * h, np.full(4, y), corners[:, 1] * h]) for y, h in zip(levels, half)]
    ring[0] = ring[0] - np.array([0.0, 1.0, 0.0])  # the legs go a metre into the ground, so none floats
    for lo, hi in zip(ring[:-1], ring[1:]):
        for c in range(4):
            member(lo[c], hi[c], 0.16)  # leg
            member(lo[c], hi[(c + 1) % 4], 0.06)  # the face's X bracing
            member(lo[(c + 1) % 4], hi[c], 0.06)
    for rg in ring[1:]:
        for c in range(4):
            member(rg[c], rg[(c + 1) % 4], 0.06)
    top = ring[-1]
    apex = np.array([0.0, height, 0.0])
    for c in range(4):
        member(top[c], apex, 0.07)
    attach_r, attach_l = [], []
    for y, length in zip((0.64, 0.75, 0.86), (arm, 1.15 * arm, 0.9 * arm)):
        y *= height
        for side, out in ((1.0, attach_r), (-1.0, attach_l)):
            root = side * w
            tip = np.array([side * (w + length), y, 0.0])
            member(np.array([root, y, w]), tip, 0.09)
            member(np.array([root, y, -w]), tip, 0.09)
            member(np.array([root, y + 0.35 * length, 0.0]), tip, 0.08)
            hang = tip - np.array([0.0, insulator, 0.0])
            member(tip, hang, 0.12)  # insulator string
            out.append(hang)
    attach = np.array(attach_r + attach_l + [apex])
    return Template(np.array(p0), np.array(p1), np.array(r), 4, attach, b, METAL)


def _pole(height: float) -> Template:
    arm_y = height - 0.7
    p0 = [np.array([0.0, -1.2, 0.0]), np.array([-1.2, arm_y, 0.0])]
    p1 = [np.array([0.0, height, 0.0]), np.array([1.2, arm_y, 0.0])]
    r = [0.14, 0.06]
    attach = []
    for x, y0 in ((1.0, arm_y), (0.0, height), (-1.0, arm_y)):
        a = np.array([x, y0 + 0.06, 0.0])
        top = a + np.array([0.0, 0.3, 0.0])
        p0.append(a)
        p1.append(top)
        r.append(0.05)
        attach.append(top)
    return Template(np.array(p0), np.array(p1), np.array(r), 8, np.array(attach), 0.14, TIMBER)


TEMPLATES: dict[str, Template] = {  # smallest first: where lines share a support, the bigger one stands
    "pole": _pole(11.0),
    "pole_tall": _pole(14.0),
    "tower_66": _tower(26.0, 3.8, 1.2),
    "tower_110": _tower(32.0, 4.8, 1.6),
    "tower_220": _tower(42.0, 6.5, 2.6),
    "tower_330": _tower(50.0, 8.0, 3.5),
}


# ----- meshes ------------------------------------------------------------------

@dataclass
class _Pose:
    e: float
    n: float
    ground: float
    across: np.ndarray  # (2,) unit, to the right of the line
    along: np.ndarray  # (2,) unit

    def place(self, local: np.ndarray) -> np.ndarray:
        """Template axes to region coordinates (e, n, h)."""
        local = np.atleast_2d(local)
        e = self.e + local[:, 0] * self.across[0] + local[:, 2] * self.along[0]
        n = self.n + local[:, 0] * self.across[1] + local[:, 2] * self.along[1]
        return np.column_stack([e, n, self.ground + local[:, 1]])


def meshes(p: Plan, dem_sample, frame: LocalFrame, road_half_m: float) -> tuple[list[Mesh], dict]:
    """The supports, wires and colliders of a plan, and a summary (supports that couldn't stand, low spans)."""
    xy = np.array([s.xy for s in p.supports]) if p.supports else np.zeros((0, 2))
    ground = np.asarray(dem_sample(xy[:, 0], xy[:, 1]), dtype=float) if len(xy) else np.zeros(0)
    poses: dict[int, _Pose] = {}
    for i, (s, g) in enumerate(zip(p.supports, ground)):
        if np.isfinite(g):
            poses[i] = _Pose(float(s.xy[0]), float(s.xy[1]), float(g), np.array([s.along[1], -s.along[0]]), s.along)
    parts: dict[tuple[str, bool], list] = {}

    def add(material: str, renderable: bool, geometry) -> None:
        parts.setdefault((material, renderable), []).append(geometry)

    colliders = 0
    for i, pose in poses.items():
        s = p.supports[i]
        t = TEMPLATES[s.template]
        add(t.material, True, _prisms(pose.place(t.p0), pose.place(t.p1), t.radius, t.sides))
        if s.distance_m <= road_half_m + COLLIDE_BEYOND_EDGE_M:
            colliders += 1
            if s.design == "tower":
                legs = np.array([(1, 1), (1, -1), (-1, -1), (-1, 1)], dtype=float) * t.base_half_m
                feet = np.column_stack([legs[:, 0], np.full(4, -0.5), legs[:, 1]])
                add(METAL, False, _prisms(pose.place(feet), pose.place(feet + [0.0, 4.5, 0.0]), np.full(4, 0.45), 6))
            else:
                add(METAL, False, _prisms(pose.place([0.0, -0.5, 0.0]), pose.place([0.0, 3.0, 0.0]),
                                          np.array([t.base_half_m + 0.06]), 8))
    low = 0
    for span in p.spans:
        if span.a not in poses or span.b not in poses:
            continue
        t = TEMPLATES[span.template]
        which = list(range(min(span.phases, 6 if span.template.startswith("tower") else 3)))
        if span.earth:
            which.append(len(t.attach) - 1)
        a_pts, b_pts = poses[span.a].place(t.attach[which]), poses[span.b].place(t.attach[which])
        radius = WIRE_RADIUS_M[span.kind]
        tubes, ok = [], True
        for a, b in zip(a_pts, b_pts):
            pts, clear = _wire(a, b, span.kind, dem_sample)
            ok &= clear
            tubes.append(_tube(pts, radius))
        low += not ok
        for tube in tubes:
            add(CABLE, True, tube)
    out: list[Mesh] = []
    names = {METAL: "POWER_TOWERS", TIMBER: "POWER_POLES", CABLE: "POWER_WIRES"}
    for (material, renderable), geoms in parts.items():
        name = names[material] if renderable else "1WALL_POWER"
        out.extend(_meshes(name, material, renderable, geoms, frame))
    return out, {"colliders": colliders, "low_spans": low, "unplaced_supports": len(p.supports) - len(poses)}


def _wire(a: np.ndarray, b: np.ndarray, kind: str, dem_sample) -> tuple[np.ndarray, bool]:
    """A sagging wire from a to b (region e, n, h), as points every WIRE_STEP_M. The sag is a parabola (close to
    a catenary at these sags), cut back where the ground rises under the span. Also: whether it stays clear."""
    length = float(np.hypot(*(b[:2] - a[:2])))
    n = max(4, int(np.ceil(length / WIRE_STEP_M)))
    t = np.linspace(0.0, 1.0, n + 1)
    straight = a + t[:, None] * (b - a)
    shape = 4.0 * t * (1.0 - t)
    sag = length ** 2 / (8.0 * CATENARY_M[kind])
    ground = np.asarray(dem_sample(straight[:, 0], straight[:, 1]), dtype=float)
    room = straight[:, 2] - CLEARANCE_M[kind] - ground
    inner = (shape > 1e-6) & np.isfinite(room)
    if inner.any():
        sag = min(sag, float(np.min(room[inner] / shape[inner])))
    sag = max(sag, MIN_SAG_M)
    pts = straight.copy()
    pts[:, 2] -= sag * shape
    clear = bool(np.all(~np.isfinite(ground) | (pts[:, 2] >= ground + CLEARANCE_M[kind] - 1e-6)))
    return pts, clear


def _frame_for(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors at right angles to each axis (m, 3), and to each other."""
    ref = np.where(np.abs(axis[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u, np.cross(axis, u)


def _prisms(p0: np.ndarray, p1: np.ndarray, radius: np.ndarray, sides: int):
    """Open-ended round members from p0 to p1 (region e, n, h), smooth-shaded: positions, normals, uvs, triangles."""
    p0, p1 = np.atleast_2d(p0).astype(float), np.atleast_2d(p1).astype(float)
    axis = p1 - p0
    length = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = axis / np.maximum(length, 1e-9)
    u, v = _frame_for(axis)
    ang = np.arange(sides) * 2 * np.pi / sides
    radial = np.cos(ang)[None, :, None] * u[:, None, :] + np.sin(ang)[None, :, None] * v[:, None, :]  # (m, s, 3)
    r = np.asarray(radius, dtype=float).reshape(-1, 1, 1)
    bottom = p0[:, None, :] + r * radial
    top = p1[:, None, :] + r * radial
    m = len(p0)
    pos = np.concatenate([bottom, top], axis=1).reshape(-1, 3)
    nrm = np.concatenate([radial, radial], axis=1).reshape(-1, 3)
    uv = np.column_stack([np.tile(np.r_[ang, ang] / (2 * np.pi), m), np.repeat(length[:, 0], 2 * sides)
                          * np.tile(np.r_[np.zeros(sides), np.ones(sides)], m) / 4.0])
    k = np.arange(sides)
    k1 = (k + 1) % sides
    quad = np.concatenate([np.column_stack([k, k1, sides + k1]), np.column_stack([k, sides + k1, sides + k])])
    tri = (quad[None, :, :] + (np.arange(m) * 2 * sides)[:, None, None]).reshape(-1, 3)
    return pos, nrm, uv, tri


def _tube(points: np.ndarray, radius: float, sides: int = 3):
    """A thin smooth tube along a polyline (region e, n, h): positions, normals, uvs, triangles."""
    d = np.gradient(points, axis=0)
    axis = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    u, v = _frame_for(axis)
    ang = np.arange(sides) * 2 * np.pi / sides
    radial = np.cos(ang)[None, :, None] * u[:, None, :] + np.sin(ang)[None, :, None] * v[:, None, :]
    pos = (points[:, None, :] + radius * radial).reshape(-1, 3)
    nrm = radial.reshape(-1, 3)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    uv = np.column_stack([np.tile(ang / (2 * np.pi), len(points)), np.repeat(along / 4.0, sides)])
    k = np.arange(sides)
    k1 = (k + 1) % sides
    quad = np.concatenate([np.column_stack([k, k1, sides + k1]), np.column_stack([k, sides + k1, sides + k])])
    tri = (quad[None, :, :] + (np.arange(len(points) - 1) * sides)[:, None, None]).reshape(-1, 3)
    return pos, nrm, uv, tri


def _meshes(name: str, material: str, renderable: bool, geoms: list, frame: LocalFrame) -> list[Mesh]:
    """Pack pieces into meshes under the vertex limit, in AC axes, wound outward."""
    out, batch, count = [], [], 0

    def flush() -> None:
        if not batch:
            return
        pos = np.concatenate([g[0] for g in batch])
        nrm = np.concatenate([g[1] for g in batch])
        uv = np.concatenate([g[2] for g in batch])
        offsets = np.cumsum([0] + [len(g[0]) for g in batch[:-1]])
        tri = np.concatenate([g[3] + o for g, o in zip(batch, offsets)])
        ac_pos = frame.to_ac(pos[:, 0], pos[:, 1], pos[:, 2])
        ac_nrm = np.column_stack([nrm[:, 0], nrm[:, 2], Z_SIGN * nrm[:, 1]])  # region (e, n, h) to AC axes
        p0, p1, p2 = ac_pos[tri[:, 0]], ac_pos[tri[:, 1]], ac_pos[tri[:, 2]]
        outward = (np.cross(p1 - p0, p2 - p0) * ac_nrm[tri].mean(axis=1)).sum(axis=1) >= 0
        tri = np.where(outward[:, None], tri, tri[:, [0, 2, 1]])
        suffix = f"_{len(out)}"
        out.append(Mesh(f"{name}{suffix}", material, ac_pos.astype(np.float32), ac_nrm.astype(np.float32),
                        uv.astype(np.float32), tri.astype(np.int64), renderable=renderable))

    for g in geoms:
        if count + len(g[0]) > MAX_VERTICES:
            flush()
            batch, count = [], 0
        batch.append(g)
        count += len(g[0])
    flush()
    return out


def geojson(p: Plan, to_lonlat, licence: str, attribution: str) -> dict:
    """The spans built, grouped by source line, as GeoJSON in WGS84 (for power.geojson)."""
    by_ref: dict[str, list] = {}
    for s in p.spans:
        a, b = p.supports[s.a].xy, p.supports[s.b].xy
        lon, lat = to_lonlat.transform([a[0], b[0]], [a[1], b[1]])
        by_ref.setdefault(s.ref, []).append([[round(lon[0], 7), round(lat[0], 7)], [round(lon[1], 7), round(lat[1], 7)]])
    features = [{"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": segs},
                 "properties": {"source": ref, "license": licence, "attribution": attribution}}
                for ref, segs in sorted(by_ref.items())]
    return {"type": "FeatureCollection", "features": features}
