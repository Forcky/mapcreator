"""The terrain corridor: how far it reaches, how coarse it gets, and that it covers the ground without holes."""

import numpy as np
import pytest
import shapely
from scipy.spatial import Delaunay, cKDTree

from mapcreator.core import centreline as cl
from mapcreator.core import mesh
from mapcreator.core import section as sec
from mapcreator.pipeline import mesh_stage, profile_stage
from mapcreator.pipeline.mesh_stage import frame_for


def hills(e, n):
    """Rolling ground, so a gap or a fold in the surface would show as a missing or doubled area."""
    e, n = np.asarray(e, float), np.asarray(n, float)
    return 100.0 + 15.0 * np.sin(e / 90.0) * np.cos(n / 120.0)


def _terrain(corridor_m: float):
    t = np.linspace(0, 1200, 80)
    line = cl.smooth(np.column_stack([500000 + t, 5250000 + 30 * np.sin(t / 200)]))
    z = hills(line.xy[:, 0], line.xy[:, 1])
    half = np.full(len(line.s), 3.2)
    sl, sr = sec.half_slopes(np.zeros(len(line.s)))
    rb = mesh.build_ribbon(line.xy, z, line.heading, line.curvature, half, half, sl, sr, hills)
    frame = frame_for(line.xy, z)
    meshes = mesh.terrain_meshes(rb, corridor_m, hills, frame)
    corridor, footprint = mesh.terrain_region(rb, corridor_m)
    return line, frame, meshes, corridor.difference(footprint)


def _triangles(meshes, frame) -> np.ndarray:
    """Every terrain triangle in plan, region coordinates: (n, 3, 2)."""
    out = []
    for m in meshes:
        p = m.positions[m.triangles]
        out.append(np.stack([p[..., 0] + frame.e0, -p[..., 2] + frame.n0], axis=-1))
    return np.vstack(out)


def _plan_area(tris: np.ndarray) -> float:
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])).sum() / 2)


def _uncovered(tris: np.ndarray, region, edge_m: float, step: float = 10.0) -> int:
    """Grid points inside `region` (less a band of `edge_m` at its rim, where the lattice can't follow the
    curved edge) that no triangle covers: holes, or seams between the lattice rings."""
    inner = region.buffer(-edge_m)
    x0, y0, x1, y1 = inner.bounds
    gx, gy = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    pts = shapely.points(gx.ravel(), gy.ravel())
    pts = pts[shapely.contains(inner, pts)]
    tree = shapely.STRtree(shapely.polygons(np.concatenate([tris, tris[:, :1]], axis=1)))
    hit = np.zeros(len(pts), bool)
    hit[np.unique(tree.query(pts, predicate="intersects")[0])] = True
    return int((~hit).sum())


def test_the_terrain_reaches_the_corridor_edge_with_a_coarse_outer_ring_and_no_holes():
    reach = mesh_stage.TERRAIN_CORRIDOR_M
    line, frame, meshes, region = _terrain(reach)
    pts = np.vstack([m.positions for m in meshes])
    en = np.column_stack([pts[:, 0] + frame.e0, -pts[:, 2] + frame.n0])  # AC x = east, z = -north
    d, _ = cKDTree(line.xy).query(en)
    assert d.max() == pytest.approx(reach, abs=mesh.OUTER_LATTICE_M)  # out to the world's edge, no further
    # Beyond the 16 m ring the lattice is the coarse outer one: few points for a lot of ground.
    outer = (d > mesh.LATTICE[-1][0] + 5) & (d < reach - 5)
    near_spacing = cKDTree(en[outer]).query(en[outer], k=2)[0][:, 1]
    assert np.median(near_spacing) == pytest.approx(mesh.OUTER_LATTICE_M, rel=0.1)
    # One triangulation spans every ring, so the surface covers the corridor, less the road, with no holes or
    # seams where the 16 m and 32 m lattices meet, and no overlaps.
    tris = _triangles(meshes, frame)
    assert _uncovered(tris, region, mesh.OUTER_LATTICE_M) == 0
    assert _plan_area(tris) <= region.area * 1.001
    assert all(len(m.positions) <= mesh.MAX_VERTICES for m in meshes)
    # The ground follows the DEM out there too.
    h = pts[:, 1] + frame.h0
    assert np.abs(h - hills(en[:, 0], en[:, 1])).max() < 0.5


def test_the_dem_is_read_beyond_the_terrain():
    # Terrain points outside the DEM corridor would get no height and be flattened: keep a margin.
    assert profile_stage.CORRIDOR_M >= mesh_stage.TERRAIN_CORRIDOR_M + 50.0


def test_a_narrower_corridor_needs_no_outer_ring():
    line, frame, meshes, region = _terrain(250.0)
    pts = np.vstack([m.positions for m in meshes])
    en = np.column_stack([pts[:, 0] + frame.e0, -pts[:, 2] + frame.n0])
    d, _ = cKDTree(line.xy).query(en)
    assert d.max() <= 250.0 + 1.0
    tris = _triangles(meshes, frame)
    assert _uncovered(tris, region, mesh.LATTICE[-1][1]) == 0 and _plan_area(tris) <= region.area * 1.001


def _triangulate(ring, others, footprint, corridor):
    """The terrain's two steps over `ring` (the footprint's outline points) and `others` (lattice points): a Delaunay
    triangulation less what lies inside the footprint, then `mesh._constrain`. Returns both."""
    pts = np.vstack([ring, others])
    hts = np.arange(len(pts), dtype=float)
    tri = Delaunay(pts).simplices
    cen = pts[tri].mean(axis=1)
    keep = shapely.contains_xy(corridor, cen[:, 0], cen[:, 1]) & ~shapely.contains_xy(footprint, cen[:, 0], cen[:, 1])
    return pts, tri[keep], mesh._constrain(pts, hts, tri, keep, corridor, footprint)


def test_a_point_near_a_long_outline_edge_no_longer_leaves_a_hole_beside_it():
    # A 10 m outline edge with a lattice point 1 m off its middle: no circle through the edge's ends is empty, so it
    # isn't a Delaunay edge, and the triangles reaching across it to the footprint's far side were dropped.
    ring = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]])
    footprint = shapely.Polygon(ring)
    others = np.array([[5.0, -1.0], [-20.0, -20.0], [30.0, -20.0], [30.0, 24.0], [-20.0, 24.0]])
    corridor = shapely.box(-20.0, -20.0, 30.0, 24.0)
    region = corridor.difference(footprint)
    pts, plain, (pts2, hts2, tri) = _triangulate(ring, others, footprint, corridor)
    assert _plan_area(pts[plain]) < region.area - 1.0  # the hole beside the edge
    assert _plan_area(pts2[tri]) == pytest.approx(region.area)  # covered exactly: no hole, nothing over the footprint
    assert len(pts2) == len(pts) and np.array_equal(hts2, np.arange(len(pts), dtype=float))  # no new points
    edges = {frozenset(e) for t in tri.tolist() for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    assert all(frozenset((k, (k + 1) % 4)) in edges for k in range(4))  # every outline edge kept


def test_where_two_pieces_outlines_cross_the_terrain_meets_both_at_the_crossing():
    # Two overlapping squares: the corner of the footprint's outline where their edges cross is neither's vertex. The
    # terrain takes it as one, at the height of both edges there (the mean), and covers the ground right up to it.
    a = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]])
    b = a + [4.0, 3.0]
    ring = np.vstack([a, b])
    footprint = shapely.union(shapely.Polygon(a), shapely.Polygon(b))
    corridor = shapely.box(-20.0, -20.0, 30.0, 30.0)
    others = np.array([[-20.0, -20.0], [30.0, -20.0], [30.0, 30.0], [-20.0, 30.0]])
    pts, plain, (pts2, hts2, tri) = _triangulate(ring, others, footprint, corridor)
    region = corridor.difference(footprint)
    assert _plan_area(pts2[tri]) == pytest.approx(region.area)
    new = pts2[len(pts):]
    assert sorted(map(tuple, new.tolist())) == [(4.0, 6.0), (6.0, 3.0)]
    # (6, 3) is half way along a's edge from point 1 (height 1) to point 2 (height 2), and a third of the way along b's
    # from point 4 (height 4) to point 5 (height 5).
    at = {tuple(p): h for p, h in zip(new.tolist(), hts2[len(pts):])}
    assert at[(6.0, 3.0)] == pytest.approx((1.5 + 4 + 1 / 3) / 2)


def test_terrain_no_outline_edge_crosses_is_left_as_delaunay_made_it():
    ring = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    footprint = shapely.Polygon(ring)
    corridor = shapely.box(-20.0, -20.0, 24.0, 24.0)
    others = np.array([[-20.0, -20.0], [24.0, -20.0], [24.0, 24.0], [-20.0, 24.0]])
    pts, plain, (pts2, _, tri) = _triangulate(ring, others, footprint, corridor)
    assert len(pts2) == len(pts) and np.array_equal(tri, plain)
