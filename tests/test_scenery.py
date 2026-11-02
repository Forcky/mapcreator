import io
import struct

import numpy as np
import pytest
import shapely
from PIL import Image

from mapcreator.core import scenery, sprites, textures
from mapcreator.core.scene import LocalFrame
from mapcreator.providers.esa.worldcover import tile_name

FRAME = LocalFrame(500_000.0, 5_200_000.0, 0.0)
CENTRE = np.column_stack([np.linspace(500_000, 501_000, 1001), np.full(1001, 5_200_000.0)])


def test_cover_candidates_stay_in_the_region_and_clear_of_the_road():
    line = shapely.LineString(CENTRE)
    region = line.buffer(250).difference(line.buffer(7.2))
    pts = scenery.candidates_from_cover(region, np.random.default_rng(0))
    d = shapely.distance(shapely.points(pts), line)
    assert d.min() >= 7.2 and d.max() <= 250
    # About one candidate per 7 x 7 m.
    assert len(pts) == pytest.approx(region.area / 49, rel=0.1)


def test_thinning_keeps_everything_near_the_road_and_less_far_away():
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(500_100, 500_900, 20000), 5_200_000 + rng.uniform(-250, 250, 20000)])
    trees = scenery.thin_by_distance(scenery.Trees(pts, np.full(len(pts), 15.0)), CENTRE, rng)
    d = np.abs(trees.xy[:, 1] - 5_200_000)
    near = np.sum(np.abs(pts[:, 1] - 5_200_000) <= 60)
    assert np.sum(d <= 60) == near
    far_kept = np.sum(d > 150) / np.sum(np.abs(pts[:, 1] - 5_200_000) > 150)
    assert far_kept == pytest.approx(0.25, abs=0.03)


def test_buildings_get_walls_and_roofs_and_skip_the_road():
    house = shapely.box(500_300, 5_200_030, 500_312, 5_200_040)  # 12 x 10 m: gable roof
    shed = shapely.Polygon([(500_400, 5_200_050), (500_430, 5_200_050), (500_430, 5_200_060),
                            (500_410, 5_200_060), (500_410, 5_200_080), (500_400, 5_200_080)])  # L-shape: flat roof
    on_road = shapely.box(500_500, 5_199_995, 500_510, 5_200_005)
    line = shapely.LineString(CENTRE)
    keep = scenery.select_buildings([house, shed, on_road], CENTRE, line.buffer(5.0))
    assert len(keep) == 2
    meshes = scenery.building_meshes(keep, lambda x, y: np.full(len(np.atleast_1d(x)), 20.0), FRAME,
                                     np.random.default_rng(3))
    names = {m.name for m in meshes}
    assert "1WALL_BUILDINGS" in names and any(n.startswith("BUILDING_ROOF_") for n in names)
    for m in meshes:
        m.validate()
        assert np.allclose(np.linalg.norm(m.normals, axis=1), 1, atol=1e-5)
    walls = next(m for m in meshes if m.name == "1WALL_BUILDINGS")
    assert walls.positions[:, 1].min() == pytest.approx(20.0 - scenery.SINK_M)
    # Wall faces point away from their building's centre.
    p = walls.positions
    t = walls.triangles
    gn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    assert (gn * walls.normals[t[:, 0]]).sum(axis=1).min() > 0


def test_dxt5_keeps_alpha():
    dds = textures.to_dds(sprites.leaf_sprays().resize((64, 64)), "DXT5")
    assert dds[84:88] == b"DXT5"
    _h, _w, linear, _d, mips = struct.unpack_from("<5I", dds, 12)
    assert linear == 16 * 16 * 16 and mips == 5
    im = Image.open(io.BytesIO(dds))
    im.load()
    alpha = np.asarray(im.convert("RGBA"))[..., 3]
    assert alpha.min() == 0 and alpha.max() == 255


def test_leaf_cells_tile_the_atlas():
    cells = [sprites.leaf_uv_cell(c) for c in range(sprites.LEAF_GRID ** 2)]
    assert all(0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1 for u0, v0, u1, v1 in cells)
    assert len(set(cells)) == len(cells)


@pytest.mark.parametrize("lon,lat,name", [(147.39, -42.78, "S45E147"), (151.2, -33.9, "S36E150"),
                                          (115.86, -31.95, "S33E114"), (-0.1, 51.5, "N51W003")])
def test_worldcover_tile_names(lon, lat, name):
    assert tile_name(lon, lat) == name


def test_tree_cap_keeps_the_nearest_trees():
    rng = np.random.default_rng(4)
    pts = np.column_stack([rng.uniform(500_000, 501_000, 10000), 5_200_000 + rng.uniform(-250, 250, 10000)])
    capped = scenery.cap_trees(scenery.Trees(pts, np.full(len(pts), 15.0)), CENTRE, 1000.0, rng, per_km=2000)
    assert len(capped) == 2000
    d = np.abs(capped.xy[:, 1] - 5_200_000)
    assert np.percentile(d, 95) < 80  # the far bands gave way


def _house_meshes(width: float, depth: float):
    e0, n0 = 500000.0, 5250000.0
    house = shapely.Polygon([(e0 + 50, n0 + 50), (e0 + 50 + width, n0 + 50), (e0 + 50 + width, n0 + 50 + depth),
                             (e0 + 50, n0 + 50 + depth)])
    polys = scenery.select_buildings([house], np.array([[e0, n0], [e0 + 200, n0]]), shapely.Point(e0, n0).buffer(1))
    return scenery.building_meshes(polys, lambda e, n: np.full(np.shape(e), 10.0), LocalFrame(e0, n0, 0.0),
                                   np.random.default_rng(1))


def test_every_wall_and_gable_end_faces_out_of_the_house():
    walls = next(m for m in _house_meshes(12, 8) if m.name.startswith("1WALL"))
    p, t = walls.positions.astype(float), walls.triangles
    centre = p.mean(axis=0)
    tri_centre = p[t].mean(axis=1)
    outward = tri_centre - centre
    outward[:, 1] = 0
    winding = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    assert ((winding * outward).sum(axis=1) > 0).all()  # front faces seen from outside, gable ends included
    assert ((walls.normals.astype(float)[t[:, 0]] * outward).sum(axis=1) > 0).all()


def test_flat_roof_uvs_are_local_so_float32_keeps_them():
    roofs = [m for m in _house_meshes(30, 25) if m.name.startswith("BUILDING_ROOF")]
    uv = np.concatenate([m.uvs for m in roofs])
    assert np.abs(uv).max() < 100  # metres from the building's corner / 4, not map coordinates
