"""Scenery: where the trees stand, and buildings as extruded footprints.

Trees and buildings stand within the scenery reach (`core.reach`), as power-line supports do.

Trees:
- Chosen here (positions and heights; made-up ones thinned with distance from the road and, near it, to a
  per-km budget).
- Drawn by Custom Shaders Patch Tree FX as 3D models (see `treefx`). The KN5 has no trees of its own,
  so players without CSP see none.
- Tree FX has no physics. So each trunk a car can reach gets an invisible collider in the KN5
  (`trunk_colliders`): an eight-sided prism the size of the drawn trunk, in `1WALL_TRUNK_*` meshes.

Building walls are a physics `1WALL_*` mesh, so cars stop at houses.

Buildings:
- Near-rectangular footprints get a gable roof along their long axis;
  everything else gets a flat roof.
- Walls start below the lowest ground point, so no building floats on a slope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy.spatial import cKDTree

from mapcreator.core import reach
from mapcreator.core.scene import MAX_VERTICES, LocalFrame, Mesh
from mapcreator.core.sprites import ROOF_COLOURS

# Trees and buildings stand within the scenery reach of the centreline (`core.reach`, 600 m).
# Made-up trees only (tree cover with no crowns, e.g. WorldCover): distance bands from the centreline,
# (up to metres, share of trees kept). Real crowns from a canopy model are all kept (`keep_trees`).
# The far band is sparse, a tree about every 20 m in full cover: far trees are many, and cost frame rate too.
NEAR_M = 250.0
TREE_BANDS = ((60.0, 1.0), (150.0, 0.55), (NEAR_M, 0.25), (reach.SCENERY_M, 0.12))
TREE_REACH_M = TREE_BANDS[-1][0]
COVER_SPACING_M = 7.0  # candidate spacing where only tree *cover* is known
CLEAR_OF_VERGE_M = 2.0
BUILDING_REACH_M = reach.SCENERY_M
STOREY_M = 3.0
SINK_M = 0.4
# Made-up trees within NEAR_M only: keeps their dense forest near the road, where the frame-rate cost is
# (docs/known-issues.md), within a budget (`keep_trees`). Beyond NEAR_M the far band's share is its own bound: at
# most 0.12 x 2 x 350 m x 1,000 m / (7 m x 7 m), about 1,700 trees per km, in unbroken forest both sides.
MAX_TREES_PER_KM = 2500
# Trunk colliders (`trunk_colliders`). The reach is from the road edge: in run-offs leaving the road at 5-30
# degrees, 80-90% of first hits on a trunk were within 20 m of the edge (Grasstree Hill, Esperance Coast Road,
# Dandenong Tourist Rd; 4 Nov 2026), and each 5 m beyond added about a quarter more colliders for 5-7% more.
TRUNK_REACH_M = 20.0
MIN_TRUNK_M = 0.10  # diameter: road-safety guidance counts a tree as a fixed-object hazard above 100 mm (FHWA)
TRUNK_SIDES = 8
TRUNK_COLLIDER = (-1.0, 3.0)  # metres from the tree's ground: below any terrain mesh there, above a bonnet
TRUNK_CHUNK_M = 400.0  # trunks are grouped into meshes by where they stand along the road, like the terrain
TRUNK_MESH = "1WALL_TRUNK"


@dataclass
class Trees:
    xy: np.ndarray  # (n, 2) region CRS
    height: np.ndarray  # (n,)

    def __len__(self) -> int:
        return len(self.height)

    def subset(self, mask: np.ndarray) -> "Trees":
        return Trees(self.xy[mask], self.height[mask])


COVER_BLOCK_M = 512.0  # candidate grid made in blocks this size

def candidates_from_cover(region: shapely.Geometry, rng: np.random.Generator,
                          spacing: float = COVER_SPACING_M) -> np.ndarray:
    """A jittered grid of candidate tree positions inside `region`.

    Made block by block, skipping blocks `region` doesn't touch, so memory follows the corridor's area
    rather than its bounding box's (which a long diagonal road makes huge and mostly empty).
    """
    minx, miny, maxx, maxy = region.bounds
    xs, ys = np.arange(minx, maxx, spacing), np.arange(miny, maxy, spacing)
    per = max(int(COVER_BLOCK_M // spacing), 1)
    shapely.prepare(region)
    out = []
    for i in range(0, len(xs), per):
        bx = xs[i:i + per]
        for j in range(0, len(ys), per):
            by = ys[j:j + per]
            block = shapely.box(bx[0] - spacing, by[0] - spacing, bx[-1] + spacing, by[-1] + spacing)
            if not region.intersects(block):
                continue
            gx, gy = np.meshgrid(bx, by)
            pts = np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-0.45, 0.45, (gx.size, 2)) * spacing
            out.append(pts[shapely.contains_xy(region, pts[:, 0], pts[:, 1])])
    return np.vstack(out) if out else np.zeros((0, 2))


def heights_for_cover(n: int, rng: np.random.Generator) -> np.ndarray:
    """Plausible dry-eucalypt heights where no canopy model exists."""
    return np.clip(rng.normal(18.0, 4.5, n), 8.0, 32.0)


def keep_trees(trees: Trees, centre_xy: np.ndarray, length_m: float, rng: np.random.Generator, *,
               real_crowns: bool, per_km: float = MAX_TREES_PER_KM) -> Trees:
    """The trees that go in the scene.

    Real crowns (a canopy model: each a tree that's there, at its own height) are all kept, so the forest is
    as close to real life as the data allows. Made-up positions (only tree cover is known) are thinned with
    distance from the road (`TREE_BANDS`), and those within NEAR_M capped at `per_km`.

    Each made-up tree gets one random draw and is kept if the draw is under its band's share. Over the cap, the
    trees within NEAR_M are kept nearest first, except that the cap never cuts a tree whose draw is under the far
    band's share. That floor runs all the way in, so a forest dense enough to meet the cap still thins out
    smoothly to the reach, rather than stopping short of NEAR_M and starting again beyond it (a 63 km road
    through forest kept no trees at all 150-250 m out).
    """
    if real_crowns:
        return trees
    d, _ = cKDTree(centre_xy).query(trees.xy, workers=-1)
    draw = rng.random(len(trees))
    keep = draw < band_share(d)
    near = keep & (d <= NEAR_M)
    budget = int(per_km * length_m / 1000)
    if near.sum() > budget:
        floor = near & (draw < TREE_BANDS[-1][1])
        extra = np.flatnonzero(near & ~floor)
        order = extra[np.argsort(d[extra] + rng.uniform(0, 30, len(extra)))]  # by distance, mixed at band edges
        keep[order[max(budget - int(floor.sum()), 0):]] = False
    return trees.subset(keep)


def band_share(d: np.ndarray) -> np.ndarray:
    """The share of made-up trees kept at each distance from the centreline (`TREE_BANDS`; none beyond them)."""
    share = np.zeros(np.shape(d))
    lower = -1.0
    for reach_m, s in TREE_BANDS:
        share[(d > lower) & (d <= reach_m)] = s
        lower = reach_m
    return share


# -- trunk colliders ---------------------------------------------------------

def edge_distance(xy: np.ndarray, centre_xy: np.ndarray, heading: np.ndarray, half_left: np.ndarray,
                  half_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each point's distance beyond the nearer road edge (negative on the road), and its nearest station."""
    if not len(xy):
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    d, i = cKDTree(centre_xy).query(xy, workers=-1)
    nrm = np.column_stack([-np.sin(heading[i]), np.cos(heading[i])])
    left = ((xy - centre_xy[i]) * nrm).sum(axis=1) > 0
    return d - np.where(left, half_left[i], half_right[i]), i


def trunk_colliders(xy: np.ndarray, ground: np.ndarray, radius: np.ndarray, edge_m: np.ndarray,
                    station_m: np.ndarray, frame: LocalFrame) -> list[Mesh]:
    """Invisible walls for the trunks a car can reach: within TRUNK_REACH_M of the road edge, MIN_TRUNK_M thick
    or more.

    Each trunk is a TRUNK_SIDES-sided prism with its corners on the drawn trunk's circle (`radius`), from
    TRUNK_COLLIDER below to above the ground. Faces are wound outwards, as the house walls are (they stop cars
    one-sided); there are no caps, since a car meets a trunk from the side. Trunks are grouped by station into
    `1WALL_TRUNK_###` meshes, split under the vertex limit. The name makes them solid, like the other
    invisible walls.
    """
    keep = (edge_m <= TRUNK_REACH_M) & (2 * radius >= MIN_TRUNK_M) & np.isfinite(ground)
    xy, ground, radius = xy[keep], ground[keep], radius[keep]
    chunk = (np.asarray(station_m)[keep] // TRUNK_CHUNK_M).astype(int)
    per_mesh = MAX_VERTICES // (2 * TRUNK_SIDES)
    out = []
    for k in np.unique(chunk):
        idx = np.flatnonzero(chunk == k)
        starts = range(0, len(idx), per_mesh)
        for p, start in enumerate(starts):
            sel = idx[start:start + per_mesh]
            name = f"{TRUNK_MESH}_{k:03d}" + (f"_{p}" if len(starts) > 1 else "")
            out.append(_prisms(name, xy[sel], ground[sel], radius[sel], frame))
    return out


def _prisms(name: str, xy: np.ndarray, ground: np.ndarray, radius: np.ndarray, frame: LocalFrame) -> Mesh:
    n, s = len(xy), TRUNK_SIDES
    theta = np.arange(s) * 2 * np.pi / s
    ring = np.column_stack([np.cos(theta), np.sin(theta)])  # east, north
    corner = xy[:, None, :] + radius[:, None, None] * ring[None]
    lo = np.repeat((ground + TRUNK_COLLIDER[0])[:, None], s, axis=1)
    hi = np.repeat((ground + TRUNK_COLLIDER[1])[:, None], s, axis=1)
    # Per trunk: the bottom ring, then the top ring.
    pos = np.concatenate([frame.to_ac(corner[..., 0], corner[..., 1], lo),
                          frame.to_ac(corner[..., 0], corner[..., 1], hi)], axis=1).reshape(-1, 3)
    out_dir = np.column_stack([ring[:, 0], np.zeros(s), -ring[:, 1]])  # horizontal, in AC axes
    nrm = np.tile(np.vstack([out_dir, out_dir]), (n, 1))
    uv = np.tile(np.column_stack([np.r_[theta, theta] / (2 * np.pi), np.r_[np.ones(s), np.zeros(s)]]), (n, 1))
    j = np.arange(s)
    j1 = (j + 1) % s
    sides = np.concatenate([np.column_stack([j, j1, s + j1]), np.column_stack([j, s + j1, s + j])])
    tri = (sides[None] + (np.arange(n) * 2 * s)[:, None, None]).reshape(-1, 3).astype(np.int64)
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    flip = (np.cross(p1 - p0, p2 - p0) * nrm[tri].mean(axis=1)).sum(axis=1) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return Mesh(name, "walls", pos.astype(np.float32), nrm.astype(np.float32), uv.astype(np.float32), tri,
                renderable=False)


# -- buildings ---------------------------------------------------------------

def select_buildings(polys: list[shapely.Polygon], centre_xy: np.ndarray, road_clear: shapely.Geometry,
                     reach: float = BUILDING_REACH_M) -> list[shapely.Polygon]:
    line = shapely.LineString(centre_xy)
    shapely.prepare(road_clear)
    out = []
    for p in polys:
        if p.area < 6 or p.area > 5000:
            continue
        if line.distance(p) > reach or road_clear.intersects(p):
            continue
        out.append(shapely.orient_polygons(p.simplify(0.3), exterior_cw=False))
    return out


@dataclass
class _Builder:
    pos: list
    nrm: list
    uv: list
    tri: list
    count: int = 0

    def quad(self, a, b, c, d, normal, uvs) -> None:
        self.pos.extend([a, b, c, d])
        self.nrm.extend([normal] * 4)
        self.uv.extend(uvs)
        i = self.count
        self.tri.extend([[i, i + 1, i + 2], [i, i + 2, i + 3]])
        self.count += 4

    def triangle(self, a, b, c, normal, uvs) -> None:
        self.pos.extend([a, b, c])
        self.nrm.extend([normal] * 3)
        self.uv.extend(uvs)
        i = self.count
        self.tri.append([i, i + 1, i + 2])
        self.count += 3

    def mesh(self, name: str, material: str) -> Mesh:
        pos = np.array(self.pos, dtype=float)
        nrm = np.array(self.nrm, dtype=float)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        tri = np.array(self.tri, dtype=np.int64)
        p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
        flip = (np.cross(p1 - p0, p2 - p0) * nrm[tri].mean(axis=1)).sum(axis=1) < 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        return Mesh(name, material, pos.astype(np.float32), nrm.astype(np.float32),
                    np.array(self.uv, dtype=np.float32), tri)


def building_meshes(polys: list[shapely.Polygon], dem_sample, frame: LocalFrame,
                    rng: np.random.Generator) -> list[Mesh]:
    walls = _Builder([], [], [], [])
    roofs = {c: _Builder([], [], [], []) for c in ROOF_COLOURS}
    colours = list(ROOF_COLOURS)

    def ac(e, n, h):
        return frame.to_ac(e, n, h)

    def ac_dir(de, dn, dh):
        return np.array([de, dh, -dn], dtype=float)

    for p in polys:
        ring = np.asarray(p.exterior.coords)[:-1]
        ground = dem_sample(ring[:, 0], ring[:, 1])
        if not np.isfinite(ground).any():
            continue
        base = float(np.nanmin(ground)) - SINK_M
        storeys = 2 if p.area > 400 and rng.random() < 0.3 else 1
        eave = float(np.nanmax(ground)) + STOREY_M * storeys
        # Walls, outward-facing (exterior ring is counter-clockwise).
        run = 0.0
        for (e0, n0), (e1, n1) in zip(ring, np.roll(ring, -1, axis=0)):
            length = float(np.hypot(e1 - e0, n1 - n0))
            if length < 0.05:
                continue
            out_n = ac_dir((n1 - n0) / length, -(e1 - e0) / length, 0.0)
            hu = (eave - base) / 3.0
            walls.quad(ac(e0, n0, base), ac(e1, n1, base), ac(e1, n1, eave), ac(e0, n0, eave), out_n,
                       [(run / 3.0, hu), ((run + length) / 3.0, hu), ((run + length) / 3.0, 0), (run / 3.0, 0)])
            run += length
        roof = roofs[colours[int(rng.integers(len(colours)))]]
        rect = shapely.minimum_rotated_rectangle(p)
        rc = np.asarray(rect.exterior.coords)[:4]
        sides = np.hypot(*np.diff(np.vstack([rc, rc[:1]]), axis=0).T)
        if p.area / max(rect.area, 1e-6) > 0.88 and min(sides) <= 14.0:
            _gable(roof, walls, rc, sides, eave, ac, ac_dir)
        else:
            _flat(roof, p, eave + 0.15, ac)
    out = []
    if walls.count:
        out.append(walls.mesh("1WALL_BUILDINGS", "walls"))  # solid: cars stop at houses
    for colour, b in roofs.items():
        if b.count:
            out.append(b.mesh(f"BUILDING_ROOF_{colour.upper()}", f"roof_{colour}"))
    return [m for m in _split(out)]


def _gable(roof: _Builder, walls: _Builder, rc: np.ndarray, sides: np.ndarray, eave: float, ac, ac_dir) -> None:
    # Ridge runs along the long axis; long edges are 0-1 and 2-3 after rotation.
    if sides[0] < sides[1]:
        rc = np.roll(rc, -1, axis=0)
        sides = np.roll(sides, -1)
    span = float(sides[1])
    rise = 0.2 * span
    a, b, c, d = rc
    m_bc, m_da = (b + c) / 2, (d + a) / 2
    ridge = eave + rise
    for (p0, p1, q1, q0) in ((a, b, m_bc, m_da), (c, d, m_da, m_bc)):
        edge = p1 - p0
        inward = q0 - p0
        normal_en = -inward / np.linalg.norm(inward)
        slope = rise / max(np.linalg.norm(inward), 1e-6)
        n = ac_dir(normal_en[0] * slope, normal_en[1] * slope, 1.0)
        length = float(np.linalg.norm(edge))
        depth = float(np.hypot(np.linalg.norm(inward), rise))
        roof.quad(ac(*p0, eave), ac(*p1, eave), ac(*q1, ridge), ac(*q0, ridge), n,
                  [(0, depth / 4), (length / 4, depth / 4), (length / 4, 0), (0, 0)])
    centre = rc.mean(axis=0)
    for (p0, p1, m) in ((b, c, m_bc), (d, a, m_da)):
        edge = p1 - p0
        out = np.array([edge[1], -edge[0]]) / max(np.linalg.norm(edge), 1e-6)
        if np.dot(out, (p0 + p1) / 2 - centre) < 0:  # away from the building, whichever way the ring winds
            out = -out
        w = float(np.linalg.norm(edge)) / 3.0
        walls.triangle(ac(*p0, eave), ac(*p1, eave), ac(*m, ridge), ac_dir(out[0], out[1], 0.0),
                       [(0, rise / 3.0), (w, rise / 3.0), (w / 2, 0)])


def _flat(roof: _Builder, poly: shapely.Polygon, height: float, ac) -> None:
    # UVs from the building's own corner: map coordinates (millions of metres) lose their fraction in float32.
    e0, n0 = np.floor(poly.bounds[:2])
    for t in shapely.constrained_delaunay_triangles(poly).geoms:
        c = np.asarray(t.exterior.coords)[:3]
        roof.triangle(*(ac(e, n, height) for e, n in c), np.array([0.0, 1.0, 0.0]),
                      [((e - e0) / 4.0, (n - n0) / 4.0) for e, n in c])


def _split(meshes: list[Mesh]) -> list[Mesh]:
    """Keep every mesh under the vertex limit (buildings use unshared vertices per face)."""
    out = []
    for m in meshes:
        if len(m.positions) <= MAX_VERTICES:
            out.append(m)
            continue
        per = (MAX_VERTICES // 3) * 3
        tri_per = per // 3
        for i, start in enumerate(range(0, len(m.triangles), tri_per)):
            t = m.triangles[start:start + tri_per]
            used = np.unique(t)
            remap = np.full(len(m.positions), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            out.append(Mesh(f"{m.name}_{i}", m.material, m.positions[used], m.normals[used], m.uvs[used], remap[t]))
    return out
