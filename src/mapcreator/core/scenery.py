"""Scenery geometry: trees as crossed cards and buildings as extruded footprints.

Trees are visual only: cars pass through them. That's a known v1 limitation;
trunk colliders can come later. Building walls are a physics `1WALL_*` mesh,
so cars stop at houses.

Trees:
- Each tree is two crossed vertical cards with triangles wound both ways, so
  they read from any angle under AC's back-face culling.
- Normals point straight up, as is usual for foliage cards, so trees light
  softly instead of going black side-on. That also lets both faces share
  vertices: 8 per tree.

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

from mapcreator.core.scene import MAX_VERTICES, LocalFrame, Mesh
from mapcreator.core.sprites import ROOF_COLOURS, TREE_GRID, tree_uv_cell

# Distance bands from the centreline: (up to metres, share of trees kept).
TREE_BANDS = ((60.0, 1.0), (150.0, 0.55), (250.0, 0.25))
TREE_REACH_M = TREE_BANDS[-1][0]
COVER_SPACING_M = 7.0  # candidate spacing where only tree *cover* is known
CLEAR_OF_VERGE_M = 2.0
BUILDING_REACH_M = 250.0
STOREY_M = 3.0
SINK_M = 0.4
CHUNK_M = 300.0
MAX_TREES_PER_KM = 2500  # keeps dense forest within a frame-rate budget


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


def thin_by_distance(trees: Trees, centre_xy: np.ndarray, rng: np.random.Generator) -> Trees:
    d, _ = cKDTree(centre_xy).query(trees.xy, workers=-1)
    keep = np.zeros(len(trees), dtype=bool)
    lower = -1.0
    for reach, share in TREE_BANDS:
        band = (d > lower) & (d <= reach)
        keep |= band & (rng.random(len(trees)) < share)
        lower = reach
    return trees.subset(keep)


def cap_trees(trees: Trees, centre_xy: np.ndarray, length_m: float, rng: np.random.Generator,
              per_km: float = MAX_TREES_PER_KM) -> Trees:
    """Thin to a per-km budget. Nearer trees are kept first; the far bands give way."""
    budget = int(per_km * length_m / 1000)
    if len(trees) <= budget:
        return trees
    d, _ = cKDTree(centre_xy).query(trees.xy, workers=-1)
    order = np.argsort(d + rng.uniform(0, 30, len(d)))  # mostly by distance, mixed a little at band edges
    keep = np.zeros(len(trees), dtype=bool)
    keep[order[:budget]] = True
    return trees.subset(keep)


def tree_meshes(trees: Trees, ground: np.ndarray, frame: LocalFrame, centre_xy: np.ndarray,
                rng: np.random.Generator, material: str = "trees", road_half: float = 3.5) -> list[Mesh]:
    """Tree cards chunked along the road. A card is never wider than twice the tree's distance to
    the road edge, so foliage can reach the edge but never hangs a card across the lanes."""
    if len(trees) == 0:
        return []
    dist, station = cKDTree(centre_xy).query(trees.xy, workers=-1)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(centre_xy, axis=0).T))])
    chunk = (s[station] // CHUNK_M).astype(int)
    per_mesh = MAX_VERTICES // 8
    yaw = rng.uniform(0, np.pi, len(trees))
    width = trees.height * rng.uniform(0.45, 0.65, len(trees))
    width = np.minimum(width, 2 * np.maximum(dist - road_half, 0.5))
    variant = rng.integers(0, TREE_GRID * TREE_GRID, len(trees))
    out = []
    for k in np.unique(chunk):
        idx = np.flatnonzero(chunk == k)
        for part, start in enumerate(range(0, len(idx), per_mesh)):
            sel = idx[start:start + per_mesh]
            name = f"TREES_{k:03d}" + (f"_{part}" if len(idx) > per_mesh else "")
            out.append(_cards(name, material, trees.xy[sel], ground[sel] - SINK_M, trees.height[sel] + SINK_M,
                              width[sel], yaw[sel], variant[sel], frame))
    return out


def _cards(name, material, xy, z0, h, w, yaw, variant, frame: LocalFrame) -> Mesh:
    """Two crossed cards per tree. Each card's 4 vertices serve both faces (one triangle pair per
    winding), which works because foliage normals point straight up: 8 vertices per tree."""
    n = len(xy)
    cells = np.array([tree_uv_cell(int(v)) for v in variant])  # u0, v0, u1, v1 (v = 0 at the top)
    quad_uv = np.stack([np.column_stack([cells[:, 0], cells[:, 3]]), np.column_stack([cells[:, 2], cells[:, 3]]),
                        np.column_stack([cells[:, 2], cells[:, 1]]), np.column_stack([cells[:, 0], cells[:, 1]])],
                       axis=1)
    pos, uv = [], []
    for plane in (0.0, np.pi / 2):
        a = yaw + plane
        d = np.column_stack([np.cos(a), np.sin(a)]) * (w / 2)[:, None]
        corners = [(xy - d, z0), (xy + d, z0), (xy + d, z0 + h), (xy - d, z0 + h)]
        pos.append(np.stack([frame.to_ac(c[:, 0], c[:, 1], hh) for c, hh in corners], axis=1))  # (n, 4, 3)
        uv.append(quad_uv)
    positions = np.concatenate([p.reshape(-1, 3) for p in pos])
    uvs = np.concatenate([q.reshape(-1, 2) for q in uv])
    normals = np.tile([0.0, 1.0, 0.0], (len(positions), 1))
    v = np.arange(2 * n)[:, None] * 4 + np.arange(4)[None, :]  # one row per card
    front = np.concatenate([v[:, [0, 1, 2]], v[:, [0, 2, 3]]])
    back = front[:, [0, 2, 1]]
    return Mesh(name, material, positions.astype(np.float32), normals.astype(np.float32), uvs.astype(np.float32),
                np.concatenate([front, back]).astype(np.int64))


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
