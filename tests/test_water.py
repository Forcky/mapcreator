"""Water beside the road: polygons from a classification, a level from the elevation, flat surfaces."""

import numpy as np
import pytest
import shapely

from mapcreator.core import package, water
from mapcreator.core.scene import LocalFrame

E0, N0 = 500000.0, 5250000.0
LAKE = shapely.Point(E0 + 200, N0 + 150).buffer(80)  # a round lake beside a road along y = N0


def region():
    road = shapely.LineString([(E0 - 400, N0), (E0 + 800, N0)])
    return road.buffer(300).difference(road.buffer(8))


def is_lake(x, y):
    return shapely.contains_xy(LAKE, x, y)


def dem(x, y):
    """Land at 12 m, the lake bed at 5 m with a noisy surface reading around it."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    inside = shapely.contains_xy(LAKE, x, y)
    rng = np.random.default_rng(0)
    return np.where(inside, 5.0 + rng.normal(0, 0.2, x.shape), 12.0)


def test_a_lake_becomes_one_body_at_its_water_level():
    bodies = water.bodies(is_lake, region(), dem)
    assert len(bodies) == 1
    b = bodies[0]
    assert b.polygon.area == pytest.approx(LAKE.area, rel=0.15)  # traced on a 10 m grid, then smoothed
    assert 4.6 < b.level < 5.2
    assert b.kind == "LAKE"


def test_the_ground_inside_sits_below_the_surface_and_the_shore_is_untouched():
    bodies = water.bodies(is_lake, region(), dem)
    ground = water.lowered(dem, bodies)
    centre = ground(np.array([E0 + 200]), np.array([N0 + 150]))
    assert centre[0] <= bodies[0].level - water.DEPTH_M + 1e-9
    far = ground(np.array([E0 - 300]), np.array([N0 + 200]))
    assert far[0] == 12.0


def test_surfaces_are_flat_at_the_level_face_up_and_cover_the_lake():
    bodies = water.bodies(is_lake, region(), dem)
    frame = LocalFrame(E0, N0, 0.0)
    meshes = water.meshes(bodies, frame)
    assert meshes and all(m.name.startswith("WATER_") and m.material == "water" for m in meshes)
    for m in meshes:
        m.validate()
        p = m.positions.astype(float)
        assert np.allclose(p[:, 1], bodies[0].level, atol=1e-3)
        t = m.triangles
        up = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])[:, 1]
        assert (up > 0).all()
    area = 0.0
    for m in meshes:
        a, b, c = (m.positions[m.triangles[:, k]].astype(float)[:, [0, 2]] for k in range(3))
        area += float(0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum())
    assert area == pytest.approx(bodies[0].polygon.area, rel=1e-3)


def test_specks_are_dropped_and_a_dry_corridor_has_no_water():
    speck = shapely.Point(E0 + 100, N0 + 100).buffer(9)
    assert water.bodies(lambda x, y: shapely.contains_xy(speck, x, y), region(), dem) == []
    assert water.bodies(lambda x, y: np.zeros(np.shape(x), bool), region(), dem) == []


def test_the_level_comes_from_the_shore_not_the_bathymetry():
    # A bay whose DEM carries depths: 0.2 m along the shore, 10 m down in the middle, and the corridor's
    # edge cutting across the deep water.
    bay = shapely.box(E0 - 200, N0 + 60, E0 + 600, N0 + 400)

    def deep(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        d = shapely.distance(bay.exterior, shapely.points(x, y))
        return np.where(shapely.contains_xy(bay, x, y), np.where(d < 12, 0.2, -10.0), 4.0)

    bodies = water.bodies(lambda x, y: shapely.contains_xy(bay, x, y), region(), deep)
    assert len(bodies) == 1 and bodies[0].level == pytest.approx(0.2, abs=0.05)


def test_large_water_near_sea_level_is_the_sea():
    bay = shapely.box(E0 - 400, N0 + 40, E0 + 800, N0 + 300)
    bodies = water.bodies(lambda x, y: shapely.contains_xy(bay, x, y), region(), lambda x, y: np.full(np.shape(x), 0.3))
    assert len(bodies) == 1 and bodies[0].kind == "SEA"
    assert water.summary(bodies)["type"] == "SEA"


def test_csp_water_shader_is_configured_only_when_there_is_water():
    with_water = package.ext_config_ini(grass_materials=["grass"], road_materials=["road"], version="t",
                                        water_materials=["water"], water_type="SEA")
    assert "[INCLUDE: common/materials_track.ini]" in with_water
    assert "[Material_Water]" in with_water and "Materials=water" in with_water and "Type=SEA" in with_water
    dry = package.ext_config_ini(grass_materials=["grass"], road_materials=["road"], version="t")
    assert "Material_Water" not in dry and "INCLUDE" not in dry
