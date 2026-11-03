import numpy as np
import pytest

from mapcreator.core import roadside
from mapcreator.core.scene import LocalFrame

FRAME = LocalFrame(500_000.0, 5_200_000.0, 0.0)


def test_mask_ignores_short_dips_joins_gaps_and_extends_runs():
    s = np.arange(0, 600.0)
    drop = np.zeros_like(s)
    drop[100:110] = 5.0  # 10 m spike: too short for a rail
    drop[200:260] = 3.0  # a real drop...
    drop[270:330] = 3.0  # ...with a 10 m gap that should be bridged
    m = roadside.rail_mask(s, drop)
    assert not m[100:110].any()
    assert m[200:330].all()
    assert m[200 - 7] and m[329 + 7]  # extended about 8 m each way
    assert not m[400:].any()


def _road(n=400):
    s = np.arange(n, dtype=float)
    xy = np.column_stack([500_000 + s, np.full(n, 5_200_000.0)])
    return s, xy, np.full(n, 50.0), np.zeros(n), np.full(n, 3.2), np.full(n, 3.2), np.full(n, 0.025), np.full(n, 0.025)


def test_rail_on_the_falling_side_only_with_an_invisible_collider():
    s, xy, z, heading, hl, hr, sl, sr = _road()

    def dem(x, y):
        # Ground falls 6 m on the right (south) side between stations 100 and 300; flat elsewhere.
        x, y = np.asarray(x), np.asarray(y)
        cliff = (y < 5_200_000) & (x > 500_100) & (x < 500_300)
        return np.where(cliff, 44.0, 49.9)

    meshes, stats = roadside.guard_rails(s, xy, z, heading, hl, hr, sl, sr, dem, FRAME)
    assert stats["L"]["runs"] == 0 and stats["R"]["runs"] == 1
    names = {m.name for m in meshes}
    assert names == {"RAIL_R000", "1WALL_RAIL_R000"}
    visual = next(m for m in meshes if m.name == "RAIL_R000")
    wall = next(m for m in meshes if m.name == "1WALL_RAIL_R000")
    assert visual.renderable and not wall.renderable
    for m in meshes:
        m.validate()
    # The rail stands 1 m outside the right edge (south = +Z in AC axes), on the verge.
    e, n, h = FRAME.from_ac(wall.positions.astype(float))
    assert np.allclose(n, 5_200_000 - 3.2 - roadside.RAIL_OFFSET_M, atol=1e-3)
    verge = 50.0 - 0.025 * 3.2 - roadside.RAIL_OFFSET_M * roadside.VERGE_FALL
    assert h.max() == pytest.approx(verge + roadside.COLLIDER[1], abs=1e-3)
    assert e.min() < 500_100 and e.max() > 500_300  # extended past the drop


def test_both_ends_of_the_road_are_closed_by_invisible_walls():
    from mapcreator.core import roadside
    from mapcreator.core.scene import LocalFrame

    s = np.arange(0.0, 500.0)
    xy = np.column_stack([500000 + s, np.full(s.size, 5250000.0)])
    z = np.full(s.size, 100.0)
    walls = roadside.end_walls(s, xy, z, np.zeros(s.size), lambda e, n: np.full(np.shape(e), 98.0),
                               LocalFrame(500000, 5250000, 100))
    assert [w.name for w in walls] == ["1WALL_END_START", "1WALL_END_FINISH"]
    for w in walls:
        w.validate()
        assert not w.renderable
        p = w.positions.astype(float)
        assert p[:, 2].min() <= -roadside.END_WALL_HALF_M + 1e-3 and p[:, 2].max() >= roadside.END_WALL_HALF_M - 1e-3
        assert p[:, 1].min() == pytest.approx(98.0 - 100.0 + roadside.END_WALL[0], abs=1e-3)
        assert p[:, 1].max() == pytest.approx(roadside.END_WALL[1], abs=1e-3)
    assert walls[0].positions[:, 0].mean() == pytest.approx(1.0, abs=1e-3)
    assert walls[1].positions[:, 0].mean() == pytest.approx(498.0, abs=1e-3)
