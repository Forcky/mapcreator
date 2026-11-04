"""The terrain corridor: how far it reaches, how coarse it gets, and that it covers the ground without holes."""

import numpy as np
import pytest
import shapely
from scipy.spatial import cKDTree

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
