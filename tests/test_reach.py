"""How far the world reaches beside the road: one set of distances that trees, buildings, power lines, the terrain
and the DEM corridor all follow (`core.reach`)."""

from types import SimpleNamespace

import numpy as np
import pytest
import shapely

from mapcreator.core import power, reach, scenery
from mapcreator.pipeline import mesh_stage, profile_stage, scenery_stage

CENTRE = np.column_stack([np.linspace(500_000, 501_000, 1001), np.full(1001, 5_200_000.0)])


def test_trees_and_buildings_share_one_reach_inside_the_terrain_and_the_dem():
    assert reach.SCENERY_M == 600.0
    assert scenery.TREE_REACH_M == scenery.BUILDING_REACH_M == reach.SCENERY_M
    # Nothing at the scenery's edge stands at the world's edge, and every terrain point has DEM under it.
    assert mesh_stage.TERRAIN_CORRIDOR_M == reach.TERRAIN_M >= reach.SCENERY_M + 100.0
    assert profile_stage.CORRIDOR_M == reach.DEM_M >= reach.TERRAIN_M + 100.0
    # Power-line supports stand anywhere on the terrain, their whole base 5 m inside its edge: past the trees.
    assert reach.POWER_M == reach.TERRAIN_M - 5.0
    # At 600 m the terrain, power and the DEM read stay where they were (#34): only trees and buildings move out.
    assert (reach.TERRAIN_M, reach.POWER_M, reach.DEM_M) == (700.0, 695.0, 800.0)


def test_made_up_tree_bands_keep_the_near_road_as_it_was_and_thin_out_to_the_reach():
    assert scenery.TREE_BANDS == ((60.0, 1.0), (150.0, 0.55), (250.0, 0.25), (reach.SCENERY_M, 0.12))
    reaches = [r for r, _ in scenery.TREE_BANDS]
    shares = [s for _, s in scenery.TREE_BANDS]
    assert reaches == sorted(reaches) and shares == sorted(shares, reverse=True)


def _forest(n: int, seed: int) -> tuple[scenery.Trees, np.ndarray]:
    """Unbroken made-up forest 600 m either side of 1 km of road: (trees, distance from the road)."""
    rng = np.random.default_rng(seed)
    pts = np.column_stack([rng.uniform(500_000, 501_000, n), 5_200_000 + rng.uniform(-600, 600, n)])
    return scenery.Trees(pts, np.full(n, 15.0)), np.abs(pts[:, 1] - 5_200_000)


def _share(kept: np.ndarray, offered: np.ndarray, lo: float, hi: float) -> float:
    return np.sum((kept > lo) & (kept <= hi)) / np.sum((offered > lo) & (offered <= hi))


def test_the_far_band_keeps_about_one_tree_in_eight():
    trees, d = _forest(40000, 3)
    kept = scenery.keep_trees(trees, CENTRE, 1e6, np.random.default_rng(3), real_crowns=False)  # no cap
    k = np.abs(kept.xy[:, 1] - 5_200_000)
    assert _share(k, d, 250, 600) == pytest.approx(0.12, abs=0.01)
    assert _share(k, d, 0, 60) == 1.0 and k.max() <= 600.0


def test_the_per_km_cap_counts_only_the_trees_near_the_road_and_leaves_no_bare_ring():
    # 1 km of road through forest like the Lyell Highway's: about 70% cover, on WorldCover's 7 m candidate grid. The
    # bands alone keep about 3,800 trees within 250 m, over the cap.
    trees, d = _forest(17000, 4)
    kept = scenery.keep_trees(trees, CENTRE, 1000.0, np.random.default_rng(4), real_crowns=False)
    k = np.abs(kept.xy[:, 1] - 5_200_000)
    assert np.sum(k <= scenery.NEAR_M) == scenery.MAX_TREES_PER_KM
    # The nearest trees are kept first (mixed a little at band edges)...
    assert _share(k, d, 0, 60) > 0.98
    assert 0.12 < _share(k, d, 60, 150) < 0.55
    # ...but the cap never cuts below the far band's share, so the forest thins out to the reach with no bare ring
    # 150-250 m out (the old cap left none there on the Lyell Highway).
    assert _share(k, d, 150, 250) == pytest.approx(0.12, abs=0.015)
    # The cap doesn't touch the trees beyond NEAR_M: their band's share bounds them.
    assert _share(k, d, 250, 600) == pytest.approx(0.12, abs=0.01)


def test_buildings_stand_within_the_reach():
    def house(y):
        return shapely.box(500_500, 5_200_000 + y, 500_510, 5_200_000 + y + 8)

    road_clear = shapely.LineString(CENTRE).buffer(5.0)
    kept = scenery.select_buildings([house(30), house(560), house(640)], CENTRE, road_clear)
    assert [round(p.bounds[1] - 5_200_000) for p in kept] == [30, 560]


def test_power_line_supports_reach_the_terrains_edge_past_the_trees(tmp_path, monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def fake_plan(lines, xy, half, *, reach_m, blocked=None):
        seen["reach"], seen["blocked"] = reach_m, blocked
        raise Stop

    def fake_lines(bbox, settings):
        seen["bbox"] = bbox
        return []

    monkeypatch.setattr(power, "plan", fake_plan)
    attribution = SimpleNamespace(dataset="lines", holder="someone", licence="CC BY 4.0")
    region = SimpleNamespace(power=fake_lines, power_attribution=attribution)
    with pytest.raises(Stop):
        scenery_stage._power(region, None, CENTRE, 3.0, None, None, tmp_path, [], None, None)
    assert seen["reach"] == reach.POWER_M == 695.0
    assert seen["blocked"] is None  # no side roads to keep the supports off
    # The lines are asked for over the whole reach.
    assert seen["bbox"][1] <= 5_200_000 - reach.POWER_M and seen["bbox"][3] >= 5_200_000 + reach.POWER_M


def test_grasstree_hills_645_m_tower_stands():
    """Grasstree Hill's farthest TasNetworks tower stands 645 m from the road (its other seven 140-589 m). Supports
    stopping with the trees at 600 m would drop it and its span; at the terrain's edge less 5 m it stays."""
    flat = lambda x, y: np.full(np.shape(x), 100.0)  # noqa: E731
    line = power.Line(np.array([[500_500.0, 5_200_000 + y] for y in (-1200.0, -645.0, -140.0, 232.0, 589.0)]),
                      power.TRANSMISSION, 220.0, 2, "gth")
    p = power.plan([line], CENTRE, 3.2, reach_m=reach.POWER_M)
    built, _ = power.meshes(p, flat, mesh_stage.frame_for(CENTRE, np.zeros(len(CENTRE))),
                            np.array([s.distance_m - 3.2 for s in p.supports]), np.zeros(len(p.supports), bool))
    assert sorted(round(s.distance_m) for s in p.supports) == [140, 232, 589, 645]  # not the one 1,200 m out
    assert len(p.spans) == 3 and built
    # With the trees' reach, the 645 m tower and its span would go.
    narrow = power.plan([line], CENTRE, 3.2, reach_m=reach.SCENERY_M)
    assert sorted(round(s.distance_m) for s in narrow.supports) == [140, 232, 589]
