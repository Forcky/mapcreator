"""Geometry: the road ribbon, verges and skirts, and the terrain around them.

Everything is built in region coordinates (E, N, H) and only converted to AC
axes at the end. The pieces share vertices exactly where they meet, so there
are no gaps for a car to fall through:

- **Road:** a cross-section of 5 points every metre, carrying the crown and camber.
- **Verge:** 2 m of grass falling away from the road edge.
- **Skirt:** points beyond the verge that sit on the DEM, so cuttings and
  embankments form by themselves. On the inside of tight bends the skirt is
  shortened before it can fold over itself.
- **Terrain:** a Delaunay surface over nested 4 / 8 / 16 m lattices, out to
  the corridor width. Its inner boundary is the skirt's outer edge, vertex
  for vertex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.spatial import Delaunay, cKDTree

from mapcreator.core.scene import MAX_VERTICES, LocalFrame, Mesh
from mapcreator.core.textures import ROAD_TILE_M

VERGE_M = 2.0
VERGE_FALL = 0.06
SKIRT_STEPS_M = (2.0, 5.0, 10.0)  # beyond the verge
CHUNK_M = 200.0
# Terrain lattice spacing by distance from the centreline: (up to metres, spacing).
LATTICE = ((60.0, 4.0), (140.0, 8.0), (300.0, 16.0))
FOLD_FRACTION = 0.8  # skirt may reach at most this share of the local radius on the inside of a bend
# Road distance over straight distance above which a nearby centreline point is another part of the route.
OTHER_PART_RATIO = 1.4


@dataclass
class Ribbon:
    """A grid of points: rows are stations, columns run left to right across the corridor."""

    e: np.ndarray  # (rows, cols)
    n: np.ndarray
    h: np.ndarray
    road_cols: slice  # columns that are carriageway
    left_cols: slice  # verge + skirt on the left, ordered outside -> road edge
    right_cols: slice  # road edge -> outside


def _neighbour_limits(xy: np.ndarray, nrm: np.ndarray, reach_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Half the distance, on each side, to another part of the route (a parallel leg of a switchback).

    Another part means a centreline point much further away along the road than in a straight line: on the
    same stretch the two are about equal, while across a hairpin the road distance is at least about 1.5
    times the gap. Two legs then share the ground between them instead of each skirt reaching over the
    other carriageway.
    """
    n = len(xy)
    left, right = np.full(n, np.inf), np.full(n, np.inf)
    if n < 3:
        return left, right
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    tree = cKDTree(xy)
    for i, near in enumerate(tree.query_ball_point(xy, r=2 * reach_m)):
        if len(near) < 2:
            continue
        near = np.asarray(near)
        d_vec = xy[near] - xy[i]
        d = np.hypot(d_vec[:, 0], d_vec[:, 1])
        other = (d > 1e-6) & (np.abs(s[near] - s[i]) > OTHER_PART_RATIO * d)
        if not other.any():
            continue
        side = d_vec[other] @ nrm[i]
        half = d[other] / 2
        if (side > 0).any():
            left[i] = half[side > 0].min()
        if (side < 0).any():
            right[i] = half[side < 0].min()
    return left, right


def _side_limits(curvature: np.ndarray, full_m: float, step: float, cap_left=None,
                 cap_right=None) -> tuple[np.ndarray, np.ndarray]:
    """Max lateral reach on each side so offsets never cross the bend's centre of curvature, nor (with the
    caps from `_neighbour_limits`) reach another part of the route."""
    k = curvature
    with np.errstate(divide="ignore"):
        lim_left = np.where(k > 1e-6, FOLD_FRACTION / np.abs(k), np.inf)
        lim_right = np.where(k < -1e-6, FOLD_FRACTION / np.abs(k), np.inf)
    if cap_left is not None:
        lim_left = np.minimum(lim_left, cap_left)
    if cap_right is not None:
        lim_right = np.minimum(lim_right, cap_right)
    win = max(int(round(40.0 / step)), 1)  # neighbouring stations' normals converge too
    lim_left = minimum_filter1d(np.minimum(lim_left, full_m), win)
    lim_right = minimum_filter1d(np.minimum(lim_right, full_m), win)
    return uniform_filter1d(lim_left, win), uniform_filter1d(lim_right, win)


def build_ribbon(xy, z, heading, curvature, half_l, half_r, slope_l, slope_r, dem_sample) -> Ribbon:
    nrm = np.column_stack([-np.sin(heading), np.cos(heading)])  # left
    step = float(np.median(np.hypot(*np.diff(xy, axis=0).T))) if len(xy) > 1 else 1.0
    full = VERGE_M + SKIRT_STEPS_M[-1]
    cap_l, cap_r = _neighbour_limits(np.asarray(xy, dtype=float), nrm, max(half_l.max(), half_r.max()) + full)
    lim_l, lim_r = _side_limits(curvature, half_l.max() + full, step, cap_l, cap_r)

    def side(half, slope, lim):
        edge_h = z - slope * half
        verge_h = edge_h - VERGE_M * VERGE_FALL
        reach = np.clip(lim - half, VERGE_M + 0.5, full)  # how far past the road edge we may go
        offs = [half + VERGE_M * np.minimum(1.0, reach / full)]
        hs = [verge_h]
        for d in SKIRT_STEPS_M:
            frac = (VERGE_M + d) / full
            offs.append(half + reach * frac)
            hs.append(None)  # from the DEM
        return offs, hs

    lo, lh = side(half_l, slope_l, lim_l)
    ro, rh = side(half_r, slope_r, lim_r)

    # Columns left (outermost) -> right (outermost). Offsets positive to the left.
    cols_off = [o for o in reversed(lo)] + [half_l, half_l / 2, np.zeros_like(half_l), -half_r / 2, -half_r] + \
               [-o for o in ro]
    road_h = [z - slope_l * half_l, z - slope_l * half_l / 2, z, z - slope_r * half_r / 2, z - slope_r * half_r]
    cols_h = [h for h in reversed(lh)] + road_h + rh

    off = np.column_stack(cols_off)
    e = xy[:, 0:1] + nrm[:, 0:1] * off
    n = xy[:, 1:2] + nrm[:, 1:2] * off
    h = np.empty_like(off)
    for j, hj in enumerate(cols_h):
        if hj is None:
            sampled = dem_sample(e[:, j], n[:, j])
            # Never let missing DEM punch a hole: fall back to the verge height.
            verge = cols_h[len(lo) - 1] if j < len(lo) else cols_h[len(lo) + 5]
            h[:, j] = np.where(np.isfinite(sampled), sampled, verge)
        else:
            h[:, j] = hj
    nl = len(lo)
    return Ribbon(e=e, n=n, h=h, road_cols=slice(nl, nl + 5), left_cols=slice(0, nl + 1),
                  right_cols=slice(nl + 4, off.shape[1]))


def _grid_triangles(rows: int, cols: int) -> np.ndarray:
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (r * cols + c).ravel()
    b = a + 1
    d = a + cols
    e = d + 1
    return np.concatenate([np.column_stack([a, d, b]), np.column_stack([b, d, e])])


def _vertex_normals(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn *= np.sign(fn[:, 1:2] + 1e-30)  # orient up
    vn = np.zeros_like(pos)
    for k in range(3):
        np.add.at(vn, tri[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    vn[~np.isfinite(vn).all(axis=1)] = (0.0, 1.0, 0.0)
    return vn


def orient_up(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Wind every triangle so its geometric normal (p1-p0)x(p2-p0) points +Y."""
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    flip = np.cross(p1 - p0, p2 - p0)[:, 1] < 0
    out = tri.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


def _mesh(name, material, pos, tri, uv) -> Mesh:
    tri = orient_up(pos, tri)
    return Mesh(name, material, pos.astype(np.float32), _vertex_normals(pos, tri).astype(np.float32),
                uv.astype(np.float32), tri.astype(np.int64))


def ribbon_meshes(rb: Ribbon, s: np.ndarray, frame: LocalFrame, *, road_uv_m: float = ROAD_TILE_M,
                  grass_uv_m: float = 6.0) -> list[Mesh]:
    """Road chunks named 1ROAD_###, verge/skirt chunks named 1GRASS_V###."""
    rows = len(s)
    step = s[1] - s[0] if rows > 1 else 1.0
    per = max(int(round(CHUNK_M / step)), 2)
    out: list[Mesh] = []
    for k, r0 in enumerate(range(0, rows - 1, per)):
        r1 = min(r0 + per, rows - 1) + 1  # include the shared boundary row
        sl = slice(r0, r1)
        for kind, cols in (("road", rb.road_cols), ("verge_l", rb.left_cols), ("verge_r", rb.right_cols)):
            e, n, h = rb.e[sl, cols], rb.n[sl, cols], rb.h[sl, cols]
            nr, nc = e.shape
            pos = frame.to_ac(e, n, h).reshape(-1, 3)
            tri = _grid_triangles(nr, nc)
            if kind == "road":
                u = np.tile(np.linspace(0, 1, nc), nr)
                v = np.repeat(s[sl] / road_uv_m, nc)
                out.append(_mesh(f"1ROAD_{k:03d}", "road", pos, tri, np.column_stack([u, v])))
            else:
                uv = np.column_stack([pos[:, 0], pos[:, 2]]) / grass_uv_m
                out.append(_mesh(f"1GRASS_V{k:03d}{kind[-1].upper()}", "grass", pos, tri, uv))
    return out


def _lattice(region, ring_xy: np.ndarray, centre_xy: np.ndarray) -> np.ndarray:
    """Nested lattice points inside `region`, densest near the road, kept clear of the skirt edge.

    Distances come from KD-trees over the densely sampled centreline and skirt edge.
    Exact shapely distances to a 13k-vertex line took over 10 minutes for C324.
    """
    centre_tree = cKDTree(centre_xy)
    minx, miny, maxx, maxy = region.bounds
    pts = []
    lower = -1.0
    for reach, sp in LATTICE:
        xs = np.arange(np.floor(minx / sp) * sp, maxx + sp, sp)
        ys = np.arange(np.floor(miny / sp) * sp, maxy + sp, sp)
        gx, gy = np.meshgrid(xs, ys)
        cand = np.column_stack([gx.ravel(), gy.ravel()])
        d, _ = centre_tree.query(cand, distance_upper_bound=reach + 1.0, workers=-1)
        keep = (d <= reach) & (d > lower)
        pts.append(cand[keep])
        lower = reach
    pts = np.vstack(pts)
    shapely.prepare(region)
    pts = pts[shapely.contains_xy(region, pts[:, 0], pts[:, 1])]
    # Keep lattice points a little away from the skirt edge, or Delaunay makes slivers.
    d_edge, _ = cKDTree(ring_xy).query(pts, distance_upper_bound=3.0, workers=-1)
    return pts[d_edge > 2.5]


def terrain_meshes(rb: Ribbon, corridor_m: float, dem_sample, frame: LocalFrame, *,
                   uv_m: float = 8.0) -> list[Mesh]:
    """Terrain from the skirt's outer edge out to `corridor_m`, chunked under the vertex limit."""
    # Footprint outline, every ribbon vertex on it: down the left edge, across the last
    # row, back up the right edge, across the first row. Including the end rows in
    # full means the terrain meets the road's ends without a crack.
    rows, cols = rb.e.shape
    idx = ([(r, 0) for r in range(rows)] + [(rows - 1, c) for c in range(1, cols)]
           + [(r, cols - 1) for r in range(rows - 2, -1, -1)] + [(0, c) for c in range(cols - 2, 0, -1)])
    ri, ci = np.array(idx).T
    ring = np.column_stack([rb.e[ri, ci], rb.n[ri, ci]])
    ring_h = rb.h[ri, ci]
    footprint = shapely.Polygon(ring).buffer(0)

    centre_cols = rb.road_cols.start + 2
    centre_xy = np.column_stack([rb.e[:, centre_cols], rb.n[:, centre_cols]])
    corridor = shapely.LineString(centre_xy).buffer(corridor_m, quad_segs=16)
    region = corridor.difference(footprint)

    lattice = _lattice(region, ring, centre_xy)
    lattice_h = dem_sample(lattice[:, 0], lattice[:, 1])
    fill = np.nanmin(ring_h) if np.isfinite(ring_h).any() else 0.0
    lattice_h = np.where(np.isfinite(lattice_h), lattice_h, fill)

    pts = np.vstack([ring, lattice])
    hts = np.concatenate([ring_h, lattice_h])
    tri = Delaunay(pts).simplices
    cen = pts[tri].mean(axis=1)
    # Keep triangles in the corridor but outside the road footprint.
    shapely.prepare(corridor)
    shapely.prepare(footprint)
    keep = shapely.contains_xy(corridor, cen[:, 0], cen[:, 1]) & ~shapely.contains_xy(footprint, cen[:, 0], cen[:, 1])
    tri = tri[keep]

    # Chunk along the road: each triangle goes with its nearest centreline station.
    cxy = centre_xy
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(cxy, axis=0).T))])
    station = cKDTree(cxy).query(pts[tri].mean(axis=1), workers=-1)[1]
    chunk = (s[station] // (CHUNK_M * 2)).astype(int)

    out: list[Mesh] = []
    for k in np.unique(chunk):
        t = tri[chunk == k]
        out.extend(_split_by_vertex_limit(f"1GRASS_T{k:03d}", t, pts, hts, frame, uv_m))
    return out


def _split_by_vertex_limit(name, tri, pts, hts, frame, uv_m) -> list[Mesh]:
    pieces = []
    parts = [tri]
    while parts:
        t = parts.pop()
        if len(np.unique(t)) > MAX_VERTICES:
            half = len(t) // 2
            order = np.argsort(pts[t].mean(axis=1)[:, 0])
            parts.extend([t[order[:half]], t[order[half:]]])
        else:
            pieces.append(t)
    meshes = []
    for i, t in enumerate(pieces):
        used = np.unique(t)
        remap = np.full(len(pts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        pos = frame.to_ac(pts[used, 0], pts[used, 1], hts[used])
        uv = np.column_stack([pos[:, 0], pos[:, 2]]) / uv_m
        suffix = f"_{i}" if len(pieces) > 1 else ""
        meshes.append(_mesh(name + suffix, "terrain", pos, remap[t], uv))
    return meshes
