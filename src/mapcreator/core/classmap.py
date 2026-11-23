"""The terrain's ground classes tidied over its triangles, and the cut batters beside the road.

`grass.GroundMap.materials` gives each terrain triangle a class from the vegetation map, looked up at a point
moved by a smooth warp (`grass.warped`), so a mapped edge comes out ragged at 10-40 m rather than ruled. Over the
terrain's own triangles (`terrain_meshes(ground=...)`), `tidy` then:

1. takes the majority class, by area, of the triangles within MAJORITY_M of each triangle's centre, which clears
   the single-triangle speckle the warp leaves where it folds;
2. with a cut material, marks the **cut batters**: ground whose slope, averaged by area over CUT_MEAN_M, is over
   CUT_SLOPE_DEG, standing above the road and within CUT_REACH_M of its edge. Fills (below the road) keep their
   class: they're grassed or planted. The mark is evened out by its own majority over CUT_MEAN_M, and patches and
   holes of it under MIN_REGION_M2 are dropped or filled, so a 4 m blip doesn't make a patch;
3. merges every region of one class under MIN_REGION_M2 into the neighbouring class it shares the longest edge
   with.

So a class's edge is one ragged line following the triangles' edges, not a band of wedges. A cut batter's own
texture coordinates run along the road and up the face (`road_uv`), so its texture doesn't stretch on the slope.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from mapcreator import cpus

MAJORITY_M = 8.0  # each triangle takes the majority class within this of its centre
MIN_REGION_M2 = 100.0  # a region of one class (or a cut patch, or a hole in one) under this is merged away
MERGE_PASSES = 4  # a small region next to only other small ones may need a second look
CUT_SLOPE_DEG = 35.0  # a cut batter's slope, averaged over CUT_MEAN_M
CUT_MEAN_M = 4.0
CUT_REACH_M = 25.0  # how far from the road's edge a cut batter may reach: beyond, a steep gully is natural ground
HEADING_TREND_M = 20.0  # a cut's texture runs along the road from the heading smoothed over this (`Road.beside`)
BLOCK = 50_000  # triangles queried at a time: bounds the neighbour lists' memory on a long road


@dataclass
class Road:
    """The route's centreline and carriageway edges (region coordinates), to say where a point is beside it."""

    xy: np.ndarray  # (n, 2) the centreline
    s: np.ndarray  # (n,) metres along it
    theta: np.ndarray  # (n,) its heading, unwrapped (radians, anticlockwise from east)
    edges: np.ndarray  # (n, 2, 3) each station's left and right carriageway edge: easting, northing, height
    trend: np.ndarray  # (n,) `theta` smoothed over HEADING_TREND_M (`beside`)

    @classmethod
    def from_ribbon(cls, rb) -> "Road":
        """From `mesh.Ribbon`: the centre column, and the first and last carriageway columns."""
        centre = rb.road_cols.start + 2
        xy = np.column_stack([rb.e[:, centre], rb.n[:, centre]]).astype(float)
        sides = [rb.road_cols.start, rb.road_cols.stop - 1]
        edges = np.stack([np.column_stack([rb.e[:, c], rb.n[:, c], rb.h[:, c]]) for c in sides], axis=1)
        return cls.from_line(xy, edges.astype(float))

    @classmethod
    def from_line(cls, xy: np.ndarray, edges: np.ndarray) -> "Road":
        xy = np.asarray(xy, dtype=float)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        t = np.gradient(xy, axis=0)
        theta = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
        step = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
        trend = gaussian_filter1d(theta, HEADING_TREND_M / max(step, 1e-6), mode="nearest")
        return cls(xy, s, theta, np.asarray(edges, dtype=float), trend)

    def beside(self, xy: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For points (easting, northing) at heights `h`: (metres along the road at their offset, metres out from
        the carriageway's edge on their side (0 on the road), height above that edge)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        _, i = cKDTree(self.xy).query(xy, workers=cpus.count())
        tangent = np.column_stack([np.cos(self.theta[i]), np.sin(self.theta[i])])
        left = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        off = xy - self.xy[i]
        along, across = (off * tangent).sum(axis=1), (off * left).sum(axis=1)
        # Each station's two edges, by their offset; a point takes the one on its own side.
        edge_off = ((self.edges[i, :, :2] - self.xy[i][:, None, :]) * left[:, None, :]).sum(axis=2)
        side = np.where(np.sign(edge_off[:, 0]) == np.sign(across), 0, 1)
        k = np.arange(len(xy))
        out = np.clip(np.abs(across) - np.abs(edge_off[k, side]), 0.0, None)
        # Along a parallel of the centreline `across` metres to its left, a metre of road is 1 - curvature * across
        # metres long, so the station there is s - across * (the heading turned through) (plus the little way past
        # station i). But that's from some starting heading, and straight out from the road it shifts the station
        # by the heading's difference from that start for every metre out. From the absolute heading, a cut beside
        # a road heading -2.7 radians (from east) had its texture shifted about 2 m along the road for every metre
        # up the face, its streaks lying nearly flat (Strickland Avenue, 24 Nov 2026); from a cut patch's own mean
        # heading, a long patch on a winding road was still shifted 1.5 m a metre in places (Lyell Highway). So it's
        # measured from the heading's trend over HEADING_TREND_M: the shift is only the road's wobble about that,
        # and on a steady bend the parallel's metres come out as the centreline's, the streaks fanning out along
        # the radii (a metre of face 0.8 m of texture 15 m out on a 60 m bend's outside, 1.33 m on its inside).
        return (self.s[i] + along - across * (self.theta[i] - self.trend[i]), out,
                np.asarray(h, dtype=float) - self.edges[i, side, 2])


def adjacency(tri: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs of triangles sharing an edge (by vertex index), and that edge's length: (i, j, metres)."""
    tri = np.asarray(tri, dtype=np.int64)
    a = tri[:, [0, 1, 2]].ravel()
    b = tri[:, [1, 2, 0]].ravel()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * (int(tri.max(initial=0)) + 1) + hi
    order = np.argsort(key, kind="stable")
    k = key[order]
    same = np.flatnonzero(k[1:] == k[:-1])
    i, j = order[same] // 3, order[same + 1] // 3
    e = order[same]
    length = np.linalg.norm(pos[lo[e]] - pos[hi[e]], axis=1)
    keep = i != j
    return i[keep], j[keep], length[keep]


def _components(label: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Each triangle's region: the triangles joined to it through shared edges, all of its label."""
    same = label[i] == label[j]
    n = len(label)
    graph = coo_matrix((np.ones(int(same.sum()), dtype=np.int8), (i[same], j[same])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _neighbours(tree: cKDTree, at: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
    """(which of `at`, which point of the tree) for every point of the tree within `r` of each of `at`."""
    lists = tree.query_ball_point(at, r, workers=cpus.count())
    counts = np.fromiter((len(x) for x in lists), np.int64, len(lists))
    flat = np.fromiter(itertools.chain.from_iterable(lists), np.int64, int(counts.sum()))
    return np.repeat(np.arange(len(at)), counts), flat


def majority(code: np.ndarray, cen: np.ndarray, area: np.ndarray, i: np.ndarray, j: np.ndarray,
             radius: float = MAJORITY_M) -> np.ndarray:
    """Each triangle's class (`code`, small integers) as the majority by area of the triangles whose centres are
    within `radius` of its own, ties going to its own. Only triangles near a change of class can change, so only
    they are looked at."""
    code = np.asarray(code, dtype=np.int64)
    edge = code[i] != code[j]
    if not edge.any():
        return code.copy()
    tree = cKDTree(cen)
    border = np.unique(np.concatenate([i[edge], j[edge]]))
    # Any triangle whose circle holds another class's centre is within the radius (and a triangle's size) of a
    # triangle on the change: twice the radius covers both.
    near = np.zeros(len(code), bool)
    for b in range(0, len(border), BLOCK):
        near[_neighbours(tree, cen[border[b:b + BLOCK]], 2 * radius)[1]] = True
    look = np.flatnonzero(near)
    k = int(code.max()) + 1
    out = code.copy()
    for b in range(0, len(look), BLOCK):
        mine = look[b:b + BLOCK]
        owner, other = _neighbours(tree, cen[mine], radius)
        votes = np.bincount(owner * k + code[other], weights=area[other], minlength=len(mine) * k).reshape(-1, k)
        votes[np.arange(len(mine)), code[mine]] += 1e-6  # a tie keeps its own class
        out[mine] = votes.argmax(axis=1)
    return out


def drop_small(flag: np.ndarray, area: np.ndarray, i: np.ndarray, j: np.ndarray,
               min_m2: float = MIN_REGION_M2, passes: int = 2) -> np.ndarray:
    """A yes/no mark over the triangles with its patches and holes under `min_m2` turned over."""
    flag = np.asarray(flag, dtype=bool).copy()
    for _ in range(passes):
        region = _components(flag, i, j)
        small = np.bincount(region, weights=area)[region] < min_m2
        if not small.any():
            break
        flag = np.where(small, ~flag, flag)
    return flag


def merge_small(code: np.ndarray, area: np.ndarray, i: np.ndarray, j: np.ndarray, length: np.ndarray,
                min_m2: float = MIN_REGION_M2, passes: int = MERGE_PASSES) -> np.ndarray:
    """Each region of one class under `min_m2` given the class it shares the longest edge with, a region of
    `min_m2` or more before a small one. A region with no neighbour of another class (an island of ground the road
    and its stubs enclose) keeps its class."""
    code = np.asarray(code, dtype=np.int64).copy()
    k = int(code.max(initial=0)) + 1
    for _ in range(passes):
        region = _components(code, i, j)
        size = np.bincount(region, weights=area)
        small = size < min_m2
        if not small[region].any():
            break
        cross = code[i] != code[j]
        a = np.concatenate([i[cross], j[cross]])
        b = np.concatenate([j[cross], i[cross]])
        w = np.concatenate([length[cross], length[cross]])
        mine = small[region[a]]
        a, b, w = a[mine], b[mine], w[mine]
        if not len(a):
            break
        w = w + np.where(small[region[b]], 0.0, 1e6)  # a big neighbour first
        rid, r = np.unique(region[a], return_inverse=True)
        score = np.bincount(r * k + code[b], weights=w, minlength=len(rid) * k).reshape(-1, k)
        best = np.full(len(size), -1, dtype=np.int64)
        best[rid] = score.argmax(axis=1)
        move = best[region] >= 0
        code[move] = best[region][move]
    return code


def slopes(p3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each triangle's slope from level (degrees) and its area (m^2): `p3` (n, 3 corners, easting northing height)."""
    n = np.cross(p3[:, 1] - p3[:, 0], p3[:, 2] - p3[:, 0])
    twice = np.linalg.norm(n, axis=1)
    slope = np.degrees(np.arccos(np.clip(np.abs(n[:, 2]) / np.maximum(twice, 1e-12), 0.0, 1.0)))
    return slope, twice / 2


def cut_batters(slope: np.ndarray, area: np.ndarray, cen: np.ndarray, out: np.ndarray, above: np.ndarray,
                i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Which triangles are a cut batter: the slope averaged by area over CUT_MEAN_M is over CUT_SLOPE_DEG, above the
    road (`above` > 0, metres) and within CUT_REACH_M of its edge (`out`); then the majority of that over
    CUT_MEAN_M, and patches and holes under MIN_REGION_M2 dropped or filled."""
    may = (out <= CUT_REACH_M) & (above > 0)
    flag = np.zeros(len(slope), bool)
    look = np.flatnonzero(may)
    if len(look):
        tree = cKDTree(cen)
        for b in range(0, len(look), BLOCK):
            mine = look[b:b + BLOCK]
            owner, other = _neighbours(tree, cen[mine], CUT_MEAN_M)
            mean = (np.bincount(owner, weights=slope[other] * area[other], minlength=len(mine))
                    / np.maximum(np.bincount(owner, weights=area[other], minlength=len(mine)), 1e-12))
            flag[mine] = mean > CUT_SLOPE_DEG
    if not flag.any():
        return flag
    # The skirt's triangles are a metre along the road and 2-5 m across it, so the mark's edge through them is a comb
    # of single rows: the majority over CUT_MEAN_M evens it out before the small patches go.
    flag = majority(flag.astype(np.int64), cen, area, i, j, CUT_MEAN_M).astype(bool)
    return drop_small(flag, area, i, j)


def tidy(material: np.ndarray, tri: np.ndarray, pts: np.ndarray, hts: np.ndarray, road: Road | None = None,
         cut: str | None = None) -> np.ndarray:
    """Each terrain triangle's material, tidied: the majority within MAJORITY_M, then `cut` (a material name) on the
    cut batters beside `road`, then regions under MIN_REGION_M2 merged. `tri` indexes `pts` (easting, northing) and
    `hts`."""
    material = np.asarray(material, dtype=object)
    if not len(tri):
        return material
    names = sorted(set(material.tolist()) | ({cut} if cut else set()))
    lookup = {n: c for c, n in enumerate(names)}
    code = np.fromiter((lookup[m] for m in material), np.int64, len(material))
    p3 = np.column_stack([pts, hts])[tri]
    slope, area = slopes(p3)
    cen = p3[:, :, :2].mean(axis=1)
    i, j, length = adjacency(tri, np.column_stack([pts, hts]))
    code = majority(code, cen, area, i, j)
    if cut and road is not None:
        _, out, above = road.beside(cen, p3[:, :, 2].mean(axis=1))
        code[cut_batters(slope, area, cen, out, above, i, j)] = lookup[cut]
    code = merge_small(code, area, i, j, length)
    return np.array(names, dtype=object)[code]


def road_uv(xy: np.ndarray, h: np.ndarray, road: Road, tile_m: float) -> np.ndarray:
    """Texture coordinates for a cut batter's points: u along the road at the point's offset, v the distance up
    the face from the carriageway's edge (straight across and up), both repeating every `tile_m`. On a planar
    batter that's the face's own metres, so a 50-degree face isn't stretched 1.6 times up the slope, as the plan
    projection the other ground classes use stretches it; and u doesn't shift with the distance out, so the
    face's streaks run straight up it (`Road.beside`)."""
    along, out, above = road.beside(xy, h)
    return np.column_stack([along, np.hypot(out, above)]) / tile_m
