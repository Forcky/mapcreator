"""Geometry: the road ribbon, verges and skirts, and the terrain around them.

Everything is built in region coordinates (E, N, H) and only converted to AC
axes at the end. The pieces share vertices exactly where they meet, so there
are no gaps for a car to fall through:

- **Road:** a cross-section of 5 points every metre, carrying the crown and camber.
- **Verge:** 2 m of grass falling away from the road edge.
- **Skirt:** points beyond the verge that sit on the DEM, so cuttings and
  embankments form by themselves. On the inside of tight bends the skirt is
  shortened before it can fold over itself.
- **Terrain:** a Delaunay surface over nested 4 / 8 / 16 m lattices, and 32 m
  beyond 300 m, out to the corridor width. One triangulation spans every ring,
  so they join without seams. Its inner boundary is the skirt's outer edge,
  vertex for vertex.

Side roads' stubs (`core.stubs`) add more ground the road takes (`Footprint`): the mainline's verge and skirt are
cut across each mouth, and the terrain leaves room for the stubs, their flares and verges, sharing their outline
vertices as it shares the ribbon's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.spatial import Delaunay, cKDTree

from mapcreator.core import textures
from mapcreator.core.scene import MAX_VERTICES, LocalFrame, Mesh
from mapcreator.core.section import STYLES, is_unsealed
from mapcreator.core.textures import GRAVEL_TILE_M, ROAD_TILE_M

VERGE_M = 2.0
VERGE_FALL = 0.06
SKIRT_STEPS_M = (2.0, 5.0, 10.0)  # beyond the verge
CHUNK_M = 200.0
# Terrain lattice spacing by distance from the centreline: (up to metres, spacing).
LATTICE = ((60.0, 4.0), (140.0, 8.0), (300.0, 16.0))
OUTER_LATTICE_M = 32.0  # spacing beyond the last ring, out to the corridor's edge: distant hills, far towers
LATTICE_BLOCK_M = 512.0  # the lattice is made in blocks this size, skipping those far from the road
FOLD_FRACTION = 0.8  # skirt may reach at most this share of the local radius on the inside of a bend
# Road distance over straight distance above which a nearby centreline point is another part of the route.
OTHER_PART_RATIO = 1.4
# Terrain in ground classes (`terrain_meshes(ground=...)`): a class's mesh gathers its triangles from consecutive
# 400 m chunks until it holds about CLASS_MESH_TRIANGLES, or spans CLASS_MESH_CHUNKS chunks (3.2 km) with at least a
# quarter of that. So the track has about as many terrain meshes as without classes, not a mesh per class per chunk
# (33 -> 167 on Grasstree Hill, and draw calls with them), and a class found in a few small patches is one mesh.
CLASS_MESH_TRIANGLES = 24_000
CLASS_MESH_CHUNKS = 8
# Classed terrain's texture coordinates can be warped by up to about UV_WARP_M, in long, smooth waves of world
# position (`warped_uv`), so a texture's repeat doesn't line up into rows across open ground. Its wavelengths are long
# beside the lattice, and since it's a function of position alone, neighbouring meshes still meet without a seam.
UV_WARP_M = 4.0
UV_WARP_WAVES_M = (71.0, 113.0, 83.0, 127.0)
# Across a side road's mouth the edge line breaks (`ribbon_meshes(edge_break=)`): the outer road column's u moves
# this far inside the painted line's outer edge (`textures.edge_line_u`), so the quad beside it shows plain asphalt.
EDGE_BREAK_MARGIN_U = 0.012


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


def build_ribbon(xy, z, heading, curvature, half_l, half_r, slope_l, slope_r, dem_sample, *,
                 skirt_cap_l=None, skirt_cap_r=None) -> Ribbon:
    """`skirt_cap_l`/`skirt_cap_r`, per station: how far from the centreline the skirt may reach on that side, where
    it shares the ground with something beside the road (a side road's stub, `core.stubs`). The skirt then stops
    there, but always reaches 0.5 m past the verge, and the verge keeps its width. Without them the ribbon is built
    as it always was."""
    nrm = np.column_stack([-np.sin(heading), np.cos(heading)])  # left
    step = float(np.median(np.hypot(*np.diff(xy, axis=0).T))) if len(xy) > 1 else 1.0
    full = VERGE_M + SKIRT_STEPS_M[-1]
    cap_l, cap_r = _neighbour_limits(np.asarray(xy, dtype=float), nrm, max(half_l.max(), half_r.max()) + full)
    lim_l, lim_r = _side_limits(curvature, half_l.max() + full, step, cap_l, cap_r)

    def side(half, slope, lim, skirt_cap):
        edge_h = z - slope * half
        verge_h = edge_h - VERGE_M * VERGE_FALL
        reach = np.clip(lim - half, VERGE_M + 0.5, full)  # how far past the road edge we may go
        verge = VERGE_M * np.minimum(1.0, reach / full)
        offs = [half + verge]
        hs = [verge_h]
        if skirt_cap is None:
            for d in SKIRT_STEPS_M:
                frac = (VERGE_M + d) / full
                offs.append(half + reach * frac)
                hs.append(None)  # from the DEM
            return offs, hs
        skirt = np.minimum(reach, np.maximum(np.asarray(skirt_cap, dtype=float) - half, verge + 0.5)) - verge
        for d in SKIRT_STEPS_M:
            offs.append(half + verge + skirt * (d / SKIRT_STEPS_M[-1]))
            hs.append(None)
        return offs, hs

    lo, lh = side(half_l, slope_l, lim_l, skirt_cap_l)
    ro, rh = side(half_r, slope_r, lim_r, skirt_cap_r)

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


# The texture of each road material (`section.STYLES`).
ROAD_TEXTURE = {"road": "road_marked", "road_plain": "road_plain", "gravel": "gravel_road",
                "gravel_narrow": "gravel_road_narrow"}


def road_chunks(style: np.ndarray, per: int) -> list[tuple[int, int]]:
    """(first row, last row) of each road chunk: a new chunk wherever the style changes, and every `per` rows
    within one style. Neighbouring chunks share their boundary row, so the road has no gap and no step there.

    The quad between rows i-1 and i belongs to the chunk ending at row i, so a change of style at row i (style[i]
    differs from style[i-1]) starts a chunk at row i. A road of one style gets the same chunks as before styles.
    """
    rows = len(style)
    cuts = [0] + [i for i in range(1, rows - 1) if style[i] != style[i - 1]] + [rows - 1]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        out += [(r0, min(r0 + per, b)) for r0 in range(a, b, per)]
    return out


def _verge_strips(rb: Ribbon) -> tuple[slice, slice]:
    """The columns of the mown verge alone, road edge to verge edge, on the left and the right."""
    return slice(rb.road_cols.start - 1, rb.road_cols.start + 1), slice(rb.road_cols.stop - 1, rb.road_cols.stop + 1)


def _skirts(rb: Ribbon) -> tuple[slice, slice]:
    """The columns of the skirt alone, verge edge outwards, on the left and the right."""
    return slice(0, rb.road_cols.start), slice(rb.road_cols.stop, rb.e.shape[1])


def ribbon_meshes(rb: Ribbon, s: np.ndarray, frame: LocalFrame, *, style: np.ndarray | None = None,
                  road_uv_m: float = ROAD_TILE_M, gravel_uv_m: float = GRAVEL_TILE_M,
                  grass_uv_m: float = 6.0, skirt: bool = True, cut: dict[str, np.ndarray] | None = None,
                  edge_break: dict[str, np.ndarray] | None = None, name_prefix: str = "") -> list[Mesh]:
    """Road chunks named 1ROAD_### (sealed) or 1GRAVEL_### (unsealed), verge/skirt chunks named 1GRASS_V###.

    `style` is the section's style code per station (`section.STYLES`); without it the road is all sealed and
    marked. The chunk number ### runs on across both surfaces, so every name is unique.
    With `skirt` False the verge meshes are the 2 m verge alone, and `terrain_meshes(fold_skirt=True)` draws the
    skirt with the ground beside it.
    Side roads' mouths (`core.stubs`): `cut`, per side ("L", "R"), is True for each quad between rows i and i+1 whose
    verge and skirt are taken out, where a stub's flares join the road; `edge_break`, per side, is True for each row
    whose outer road vertex takes a u inside the painted edge line, so a sealed road's line breaks across the mouth.
    `name_prefix` goes after the KEY in every name (a roundabout's ring, `core.roundabouts`: `1ROAD_RB0_000`); a side
    with no verge (a ring's island side) gets no verge mesh.
    """
    rows = len(s)
    step = s[1] - s[0] if rows > 1 else 1.0
    per = max(int(round(CHUNK_M / step)), 2)
    style = np.zeros(rows, dtype=np.int8) if style is None else np.asarray(style)
    left, right = (rb.left_cols, rb.right_cols) if skirt else _verge_strips(rb)
    out: list[Mesh] = []
    for k, (r0, last) in enumerate(road_chunks(style, per)):
        sl = slice(r0, last + 1)  # include the shared boundary row
        material = STYLES[int(style[r0])]
        gravel = bool(is_unsealed(style[r0]))
        for kind, cols in (("road", rb.road_cols), ("verge_l", left), ("verge_r", right)):
            e, n, h = rb.e[sl, cols], rb.n[sl, cols], rb.h[sl, cols]
            nr, nc = e.shape
            pos = frame.to_ac(e, n, h).reshape(-1, 3)
            tri = _grid_triangles(nr, nc)
            if kind == "road":
                u = np.tile(np.linspace(0, 1, nc), nr)
                if edge_break is not None and not gravel:
                    inside = textures.edge_line_u(ROAD_TEXTURE[material]) + EDGE_BREAK_MARGIN_U
                    grid = u.reshape(nr, nc)
                    grid[edge_break["L"][sl], 0] = inside
                    grid[edge_break["R"][sl], nc - 1] = 1.0 - inside
                v = np.repeat(s[sl] / (gravel_uv_m if gravel else road_uv_m), nc)
                name = f"{'1GRAVEL' if gravel else '1ROAD'}_{name_prefix}{k:03d}"
                out.append(_mesh(name, material, pos, tri, np.column_stack([u, v])))
            else:
                if nc < 2:
                    continue
                side = kind[-1].upper()
                uv = np.column_stack([pos[:, 0], pos[:, 2]]) / grass_uv_m
                if cut is not None and cut[side][r0:last].any():
                    keep = np.repeat(~cut[side][r0:last], nc - 1)
                    pos, uv, tri = compact(pos, uv, tri[np.concatenate([keep, keep])])
                    if not len(tri):
                        continue
                out.append(_mesh(f"1GRASS_{name_prefix}V{k:03d}{side}", "grass", pos, tri, uv))
    return out


def compact(pos: np.ndarray, uv: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A mesh's vertices and texture coordinates with only those its triangles use, and the triangles renumbered."""
    used, local = np.unique(tri, return_inverse=True)
    return pos[used], uv[used], local.reshape(tri.shape)


def soup_meshes(name: str, material: str, pieces: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                frame: LocalFrame) -> list[Mesh]:
    """Pieces of one material, each (positions (n, 3) as easting, northing, height; texture coordinates (n, 2);
    triangles (m, 3)), packed in order into meshes under the vertex limit: `name`, then `name_1` and so on. A piece's
    vertices no triangle uses are left out."""
    out, batch, held = [], [], 0
    pieces = [compact(*piece) for piece in pieces]

    def flush():
        if not batch:
            return
        pos = np.vstack([p for p, _, _ in batch])
        uv = np.vstack([u for _, u, _ in batch])
        starts = np.cumsum([0] + [len(p) for p, _, _ in batch[:-1]])
        tri = np.vstack([t + k for (_, _, t), k in zip(batch, starts)])
        suffix = f"_{len(out)}" if out else ""
        out.append(_mesh(name + suffix, material, frame.to_ac(pos[:, 0], pos[:, 1], pos[:, 2]), tri, uv))

    for piece in pieces:
        if held + len(piece[0]) > MAX_VERTICES:
            flush()
            batch, held = [], 0
        batch.append(piece)
        held += len(piece[0])
    flush()
    return out


def _lattice(region, ring_xy: np.ndarray, centre_xy: np.ndarray, reach_m: float = LATTICE[-1][0]) -> np.ndarray:
    """Nested lattice points inside `region`, densest near the road, kept clear of the skirt edge. Past the
    last LATTICE ring, out to `reach_m`, the spacing is OUTER_LATTICE_M.

    Distances come from KD-trees over the densely sampled centreline and skirt edge.
    Exact shapely distances to a 13k-vertex line took over 10 minutes for C324.

    The grid is made block by block, skipping blocks the corridor doesn't reach, so memory follows the
    corridor's area rather than the bounding box's (a long diagonal route has a huge, mostly empty box).
    """
    centre_tree = cKDTree(centre_xy)
    minx, miny, maxx, maxy = region.bounds
    pts = []
    lower = -1.0
    rings = LATTICE + (((reach_m, OUTER_LATTICE_M),) if reach_m > LATTICE[-1][0] else ())
    for reach, sp in rings:
        xs = np.arange(np.floor(minx / sp) * sp, maxx + sp, sp)
        ys = np.arange(np.floor(miny / sp) * sp, maxy + sp, sp)
        per = max(int(LATTICE_BLOCK_M // sp), 1)
        level = []
        for i in range(0, len(xs), per):
            bx = xs[i:i + per]
            for j in range(0, len(ys), per):
                by = ys[j:j + per]
                half_diag = float(np.hypot(bx[-1] - bx[0], by[-1] - by[0])) / 2
                gap, _ = centre_tree.query([(bx[0] + bx[-1]) / 2, (by[0] + by[-1]) / 2])
                if gap - half_diag > reach:
                    continue  # no point in this block is within reach of the road
                gx, gy = np.meshgrid(bx, by)
                cand = np.column_stack([gx.ravel(), gy.ravel()])
                d, _ = centre_tree.query(cand, distance_upper_bound=reach + 1.0, workers=-1)
                level.append(cand[(d <= reach) & (d > lower)])
        level = np.vstack(level) if level else np.zeros((0, 2))
        pts.append(level[np.lexsort((level[:, 0], level[:, 1]))])  # row by row, as one grid would give
        lower = reach
    pts = np.vstack(pts)
    shapely.prepare(region)
    pts = pts[shapely.contains_xy(region, pts[:, 0], pts[:, 1])]
    # Keep lattice points a little away from the skirt edge, or Delaunay makes slivers.
    d_edge, _ = cKDTree(ring_xy).query(pts, distance_upper_bound=3.0, workers=-1)
    return pts[d_edge > 2.5]


def _outline_index(rb: Ribbon) -> tuple[np.ndarray, np.ndarray]:
    """The (row, column) of each ribbon vertex on the footprint's outline, in `_outline`'s order."""
    rows, cols = rb.e.shape
    idx = ([(r, 0) for r in range(rows)] + [(rows - 1, c) for c in range(1, cols)]
           + [(r, cols - 1) for r in range(rows - 2, -1, -1)] + [(0, c) for c in range(cols - 2, 0, -1)])
    ri, ci = np.array(idx).T
    return ri, ci


def _outline(rb: Ribbon) -> tuple[np.ndarray, np.ndarray]:
    """Footprint outline, every ribbon vertex on it: down the left edge, across the last row, back up the
    right edge, across the first row. Including the end rows in full means the terrain meets the road's
    ends without a crack. Returns the ring (easting, northing) and its heights."""
    ri, ci = _outline_index(rb)
    return np.column_stack([rb.e[ri, ci], rb.n[ri, ci]]), rb.h[ri, ci]


@dataclass
class Footprint:
    """Ground the road takes besides its own ribbon: side roads' stubs (`core.stubs`), for the terrain.

    - `cut`: per side ("L", "R"), True for each quad between rows i and i+1 of the road's ribbon whose verge and
      skirt are taken out (`ribbon_meshes(cut=)`).
    - `ribbons`: whole ribbons beside the road (the stubs). Their outlines bound the terrain, less their first row's
      road (which meets the flares), and their skirts join it as the road's does (`terrain_meshes(fold_skirt=True)`).
    - `patches`: other pieces' outlines (n, 2): the throats and the flares' verges, which the terrain stays off.
    - `edges`: the points (n, 2) and heights (n,) of their outlines that face the terrain: the flares' verges' outer
      edges. Only points on the outside of all the pieces together go into the terrain's triangulation: one shared
      by two pieces could join a triangle reaching across one of them, and leave a hole where it's dropped.

    Pieces meet each other only along shared vertices, so the terrain shares the outline points they have in common.
    """

    cut: dict[str, np.ndarray]
    ribbons: list[Ribbon]
    patches: list[np.ndarray]
    edges: list[tuple[np.ndarray, np.ndarray]]
    # A roundabout's ring (`core.roundabouts`), by ribbon: its own verge and skirt quads cut for its exits (as `cut` is
    # for the road), and whether its last row's road meets a throat too, as its first does.
    ribbon_cuts: list[dict[str, np.ndarray] | None] | None = None
    open_last: list[bool] | None = None
    # Outline points that face no terrain (a ring's road edge against its island): never in the triangulation.
    interior: set[tuple[float, float]] | None = None


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last + 1) of each run of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _cut_cells(rb: Ribbon, side: str, a: int, b: int) -> list[tuple[int, int]]:
    """The (row, column) ring of the verge and skirt cut away between rows a and b on one side, road edge included:
    the outer edge from row a to b, row b in to the road's edge, the road's edge back to row a, and row a out."""
    if side == "L":
        edge = rb.road_cols.start
        return ([(r, 0) for r in range(a, b + 1)] + [(b, c) for c in range(1, edge + 1)]
                + [(r, edge) for r in range(b - 1, a - 1, -1)] + [(a, c) for c in range(edge - 1, 0, -1)])
    edge, outer = rb.road_cols.stop - 1, rb.e.shape[1] - 1
    return ([(r, outer) for r in range(a, b + 1)] + [(b, c) for c in range(outer - 1, edge - 1, -1)]
            + [(r, edge) for r in range(b - 1, a - 1, -1)] + [(a, c) for c in range(edge + 1, outer)])


def _footprint(rb: Ribbon, fp: Footprint) -> tuple[dict, np.ndarray, np.ndarray, shapely.Geometry]:
    """The terrain's inner boundary with stubs: every outline vertex of the road's ribbon (less those cut away) and
    of each other piece, each once, as (index by (easting, northing), points, heights), and the ground they all
    cover."""
    ri, ci = _outline_index(rb)
    rows, cols = rb.e.shape
    gone = np.zeros((rows, cols), bool)  # outline vertices of the cut verge and skirt, which no mesh has now
    cells: list[tuple[int, int]] = []  # the cut's edges, rows a and b, from the road's verge outwards
    cut_rings = []
    for side, mask in fp.cut.items():
        edge = rb.road_cols.start if side == "L" else rb.road_cols.stop - 1
        for a, b in _runs(mask):  # quads a .. b-1, so rows a .. b
            ring = _cut_cells(rb, side, a, b)
            cut_rings.append(shapely.Polygon([(rb.e[r, c], rb.n[r, c]) for r, c in ring]).buffer(0))
            gone[a + 1:b, 0 if side == "L" else cols - 1] = True
            cells += [(r, c) for r, c in ring if r in (a, b) and c != edge]
    cells = [(r, c) for r, c in cells if not gone[r, c]]
    index: dict[tuple[float, float], int] = {}
    xy: list[tuple[float, float]] = []
    hs: list[float] = []

    interior = fp.interior or set()

    def add(x, y, h):
        key = (float(x), float(y))
        if key not in index and key not in interior:
            index[key] = len(xy)
            xy.append(key)
            hs.append(float(h))

    for r, c in zip(ri.tolist(), ci.tolist()):
        if not gone[r, c]:
            add(rb.e[r, c], rb.n[r, c], rb.h[r, c])
    for r, c in cells:
        add(rb.e[r, c], rb.n[r, c], rb.h[r, c])
    polygons = []
    for k, other in enumerate(fp.ribbons):
        oi, oc = _outline_index(other)
        road_col = (oc >= other.road_cols.start) & (oc < other.road_cols.stop)
        first_road = (oi == 0) & road_col
        if fp.open_last is not None and fp.open_last[k]:
            first_road |= (oi == other.e.shape[0] - 1) & road_col
        own_cut = fp.ribbon_cuts[k] if fp.ribbon_cuts is not None else None
        if own_cut is None:
            for r, c in zip(oi[~first_road].tolist(), oc[~first_road].tolist()):
                add(other.e[r, c], other.n[r, c], other.h[r, c])
            polygons.append(shapely.Polygon(np.column_stack([other.e[oi, oc], other.n[oi, oc]])).buffer(0))
            continue
        # As the road's own cut above: the cut verge's outline vertices go, its end rows' come in.
        o_rows, o_cols = other.e.shape
        o_gone = np.zeros((o_rows, o_cols), bool)
        o_cells, o_rings = [], []
        for side, mask in own_cut.items():
            edge = other.road_cols.start if side == "L" else other.road_cols.stop - 1
            if (side == "L" and other.road_cols.start == 0) or (side == "R" and other.road_cols.stop == o_cols):
                continue
            for a, b in _runs(mask):
                ring = _cut_cells(other, side, a, b)
                o_rings.append(shapely.Polygon([(other.e[r, c], other.n[r, c]) for r, c in ring]).buffer(0))
                o_gone[a + 1:b, 0 if side == "L" else o_cols - 1] = True
                o_cells += [(r, c) for r, c in ring if r in (a, b) and c != edge]
        for r, c in zip(oi[~first_road].tolist(), oc[~first_road].tolist()):
            if not o_gone[r, c]:
                add(other.e[r, c], other.n[r, c], other.h[r, c])
        for r, c in o_cells:
            if not o_gone[r, c]:
                add(other.e[r, c], other.n[r, c], other.h[r, c])
        poly = shapely.Polygon(np.column_stack([other.e[oi, oc], other.n[oi, oc]])).buffer(0)
        polygons.append(poly.difference(shapely.unary_union(o_rings)) if o_rings else poly)
    for points, heights in fp.edges:
        for (x, y), h in zip(np.asarray(points, float).tolist(), np.asarray(heights, float).tolist()):
            add(x, y, h)
    polygons += [shapely.Polygon(ring).buffer(0) for ring in fp.patches]
    road = shapely.Polygon(np.column_stack([rb.e[ri, ci], rb.n[ri, ci]])).buffer(0)
    if cut_rings:
        road = road.difference(shapely.unary_union(cut_rings))
    return index, np.array(xy), np.array(hs), shapely.unary_union([road, *polygons])


def terrain_region(rb: Ribbon, corridor_m: float, extra: Footprint | None = None):
    """(corridor, footprint): the terrain covers the corridor around the road minus the ribbon's footprint, and with
    `extra` the stubs' too."""
    if extra is None:
        ring, _ = _outline(rb)
        footprint = shapely.Polygon(ring).buffer(0)
    else:
        footprint = _footprint(rb, extra)[3]
    centre_cols = rb.road_cols.start + 2
    centre_xy = np.column_stack([rb.e[:, centre_cols], rb.n[:, centre_cols]])
    return shapely.LineString(centre_xy).buffer(corridor_m, quad_segs=16), footprint


def _skirt_triangles(rb: Ribbon, first: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The skirt's quads as triangles over the terrain's points: its outer edge and the ribbon's end rows are
    already there (the outline, `_outline`, numbered from 0), and its other points are added, numbered from
    `first`. Returns their (easting, northing), heights, and the triangles."""
    rows, cols = rb.e.shape
    ri, ci = _outline_index(rb)
    index = np.full((rows, cols), -1, dtype=np.int64)
    index[ri, ci] = np.arange(len(ri))
    tris, new = [], []
    for cs in _skirts(rb):
        block = index[:, cs]
        fresh = block < 0
        block[fresh] = first + sum(len(n) for n in new) + np.arange(int(fresh.sum()))  # a view: numbers `index`
        new.append(np.argwhere(fresh) + [0, cs.start])
        tris.append(block.ravel()[_grid_triangles(rows, block.shape[1])])
    rc = np.vstack(new)
    return (np.column_stack([rb.e[rc[:, 0], rc[:, 1]], rb.n[rc[:, 0], rc[:, 1]]]), rb.h[rc[:, 0], rc[:, 1]],
            np.vstack(tris))


def _fold_skirts(rb: Ribbon, fp: Footprint, index: dict, first: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`_skirt_triangles` with stubs: the road's skirt less its cut quads, and each stub's, over the terrain's points.
    A vertex already among them (by its easting and northing, `index`) is shared; the others are added, numbered from
    `first`. Returns their (easting, northing), heights, and the triangles."""
    new_xy: list[tuple[float, float]] = []
    new_h: list[float] = []
    tris = []
    cuts = fp.ribbon_cuts if fp.ribbon_cuts is not None else [None] * len(fp.ribbons)
    for ribbon, cut in [(rb, fp.cut)] + list(zip(fp.ribbons, cuts)):
        rows = ribbon.e.shape[0]
        for side, cs in zip("LR", _skirts(ribbon)):
            e, n, h = ribbon.e[:, cs], ribbon.n[:, cs], ribbon.h[:, cs]
            if e.shape[1] < 2:
                continue  # no skirt on this side (a ring's island side)
            ids = np.empty(e.size, np.int64)
            for k, (x, y, z) in enumerate(zip(e.ravel().tolist(), n.ravel().tolist(), h.ravel().tolist())):
                i = index.get((x, y))
                if i is None:
                    i = index[(x, y)] = first + len(new_xy)
                    new_xy.append((x, y))
                    new_h.append(z)
                ids[k] = i
            t = ids[_grid_triangles(rows, e.shape[1])]
            if cut is not None and cut[side].any():
                keep = np.repeat(~cut[side], e.shape[1] - 1)
                t = t[np.concatenate([keep, keep])]
            tris.append(t)
    return np.array(new_xy).reshape(-1, 2), np.array(new_h), np.vstack(tris)


def terrain_meshes(rb: Ribbon, corridor_m: float, dem_sample, frame: LocalFrame, *,
                   uv_m: float = 8.0, ground=None, classes: dict[str, tuple[str, float, bool]] | None = None,
                   fold_skirt: bool = False, extra: Footprint | None = None) -> list[Mesh]:
    """Terrain from the skirt's outer edge out to `corridor_m`, chunked under the vertex limit.

    - `fold_skirt` adds the skirt (verge edge outwards) to the terrain, for verge meshes made without it
      (`ribbon_meshes(skirt=False)`). Its outer edge is the terrain's own boundary, so they share those points.
    - `ground`, if given, picks each triangle's material: called with the triangles' centres (n, 2, easting and
      northing) and their distance from the centreline, it returns a material name for each. `classes` gives each
      material's mesh-name letter, its texture's repeat in metres, and whether its texture coordinates are warped
      (`warped_uv`). Each material's triangles go in meshes of a few 400 m chunks each (`_class_groups`), named by
      their first chunk and the letter (`1GRASS_T012P`). Without it every triangle is `terrain`, in a mesh per chunk
      named `1GRASS_T###`, as before.
    - `extra`: the stubs' ground (`Footprint`). The terrain then leaves room for them too, sharing their outlines.
    """
    if extra is None:
        ring, ring_h = _outline(rb)
        corridor, footprint = terrain_region(rb, corridor_m)
    else:
        index, ring, ring_h, footprint = _footprint(rb, extra)
        corridor, _ = terrain_region(rb, corridor_m)
    centre_cols = rb.road_cols.start + 2
    centre_xy = np.column_stack([rb.e[:, centre_cols], rb.n[:, centre_cols]])
    region = corridor.difference(footprint)

    lattice = _lattice(region, ring, centre_xy, corridor_m)
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
    if fold_skirt:
        skirt_xy, skirt_h, skirt_tri = (_skirt_triangles(rb, len(pts)) if extra is None
                                        else _fold_skirts(rb, extra, index, len(pts)))
        pts, hts, tri = np.vstack([pts, skirt_xy]), np.concatenate([hts, skirt_h]), np.vstack([tri, skirt_tri])

    # Chunk along the road: each triangle goes with its nearest centreline station.
    cxy = centre_xy
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(cxy, axis=0).T))])
    cen = pts[tri].mean(axis=1)
    dist, station = cKDTree(cxy).query(cen, workers=-1)
    chunk = (s[station] // (CHUNK_M * 2)).astype(int)
    material = np.asarray(ground(cen, dist), dtype=object) if ground is not None else None

    out: list[Mesh] = []
    if material is None:
        for k in np.unique(chunk):
            out.extend(_split_by_vertex_limit(f"1GRASS_T{k:03d}", tri[chunk == k], pts, hts, frame, uv_m))
        return out
    # Meshes of each material in `classes` order, each of a few chunks; then sorted along the road.
    rank = {n: i for i, n in enumerate(classes)}
    code = np.fromiter((rank[m] for m in material), np.int64, len(material))
    order = np.argsort(code, kind="stable")
    meshes = []
    for idx in np.split(order, np.flatnonzero(np.diff(code[order])) + 1):
        name = material[idx[0]]
        letter, tile, warp = classes[name]
        for first, mine in _class_groups(chunk[idx]):
            meshes.append((first, rank[name], _split_by_vertex_limit(
                f"1GRASS_T{first:03d}{letter}", tri[idx[mine]], pts, hts, frame, tile, name,
                warp=tile if warp else None)))
    for _, _, ms in sorted(meshes, key=lambda m: m[:2]):
        out.extend(ms)
    return out


def warped_uv(x: np.ndarray, z: np.ndarray, tile_m: float) -> np.ndarray:
    """Texture coordinates for AC x and z (metres), repeating every `tile_m` but shifted by up to about UV_WARP_M in
    two directions by two waves each."""
    a, b, c, d = UV_WARP_WAVES_M
    wx = np.sin(2 * np.pi * z / a + 1.3) + 0.8 * np.sin(2 * np.pi * (x + z) / b + 4.1)
    wz = np.sin(2 * np.pi * x / c + 2.7) + 0.8 * np.sin(2 * np.pi * (x - z) / d + 0.6)
    return np.column_stack([x + UV_WARP_M / 1.8 * wx, z + UV_WARP_M / 1.8 * wz]) / tile_m


def _class_groups(chunk: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """One class's triangles (their 400 m chunk numbers) in meshes: consecutive chunks gathered until a mesh holds
    about CLASS_MESH_TRIANGLES, or spans CLASS_MESH_CHUNKS with a quarter of that. Returns (first chunk, triangle
    indices) per mesh."""
    order = np.argsort(chunk, kind="stable")
    ids, starts, counts = np.unique(chunk[order], return_index=True, return_counts=True)
    out, first, start, held = [], None, 0, 0
    for k, at, n in zip(ids, starts, counts):
        full = held + n > CLASS_MESH_TRIANGLES
        long = k - first >= CLASS_MESH_CHUNKS and 4 * held >= CLASS_MESH_TRIANGLES if first is not None else False
        if first is not None and (full or long):
            out.append((first, order[start:at]))
            first = None
        if first is None:
            first, start, held = int(k), int(at), 0
        held += int(n)
    if first is not None:
        out.append((first, order[start:]))
    return out


def _split_by_vertex_limit(name, tri, pts, hts, frame, uv_m, material: str = "terrain",
                           warp: float | None = None) -> list[Mesh]:
    """Meshes of `tri` under the vertex limit, halved by easting until each fits. UVs repeat every `uv_m`, or with
    `warp` every `warp` metres through `warped_uv`."""
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
        used, local = np.unique(t, return_inverse=True)  # each corner's index among the mesh's own points
        pos = frame.to_ac(pts[used, 0], pts[used, 1], hts[used])
        uv = np.column_stack([pos[:, 0], pos[:, 2]]) / uv_m if warp is None else warped_uv(pos[:, 0], pos[:, 2], warp)
        suffix = f"_{i}" if len(pieces) > 1 else ""
        meshes.append(_mesh(name + suffix, material, pos, local.reshape(t.shape), uv))
    return meshes
