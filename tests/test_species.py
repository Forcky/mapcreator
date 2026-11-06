import numpy as np
import pytest
import shapely

from mapcreator.core import species, treefx
from mapcreator.core.species import Community
from mapcreator.providers.tas_list import tasveg


def box(x0, y0, x1, y1):
    return shapely.box(500_000 + x0, 5_200_000 + y0, 500_000 + x1, 5_200_000 + y1)


def pts(*xy):
    return np.array(xy, dtype=float) + [500_000.0, 5_200_000.0]


@pytest.mark.parametrize("code,kind", [("DOB", "woody"), ("WNL", "woody"), ("RMT", "woody"), ("FPS", "woody"),
                                       ("FRG", "woody"), ("MBS", "woody"), ("SCH", "woody"), ("FAL", "paddock"),
                                       ("FUR", "paddock"), ("MBW", "treeless"), ("GPL", "treeless"),
                                       ("OAQ", "treeless"), ("FPE", "treeless"), ("", "")])
def test_codes_sort_into_woody_paddock_and_treeless(code, kind):
    assert species.kind(code) == kind


def test_woody_communities_name_their_family_and_bark():
    vmap = species.VegetationMap([Community(box(0, 0, 100, 100), "DOB"), Community(box(100, 0, 200, 100), "DGL"),
                                  Community(box(200, 0, 300, 100), "FPS"), Community(box(300, 0, 400, 100), "WRE")])
    fam = species.families(pts((50, 50), (150, 50), (250, 50), (350, 50)), np.full(4, 20.0), vmap)
    assert fam.tolist() == [species.DRY_ROUGH, species.DRY_SMOOTH, species.PINE, species.WET]


def test_mixed_wet_forest_splits_eucalypt_from_understorey_by_height():
    vmap = species.VegetationMap([Community(box(0, 0, 200, 200), "WNL"), Community(box(200, 0, 400, 200), "WOR")])
    xy = pts((50, 50), (60, 50), (70, 50), (250, 50), (260, 50))
    fam = species.families(xy, np.array([50.0, 32.0, 12.0, 40.0, 10.0]), vmap)
    # 60% of the 50 m tallest is 30 m: 32 m is still the eucalypt, 12 m is the tea-tree understorey.
    assert fam.tolist() == [species.WET, species.WET, species.TEA_TREE, species.WET, species.RAINFOREST]


def test_paddock_trees_take_the_named_species_else_the_nearby_forest_else_a_gum():
    vmap = species.VegetationMap([
        Community(box(0, 0, 100, 100), "FAL", notable="PR", notable_name="Pinus radiata"),
        Community(box(100, 0, 200, 100), "FAL", notable="EP", notable_name="Eucalyptus pulchella"),
        Community(box(200, 0, 300, 100), "FAL", notable="EO", notable_name="Eucalyptus obliqua"),
        Community(box(300, 0, 400, 100), "FAL"),
        Community(box(400, 0, 500, 100), "WOU"),
    ])
    xy = pts((50, 50), (150, 50), (250, 50), (380, 50), (320, 50))
    fam = species.families(xy, np.full(5, 25.0), vmap)
    # 20 m from the wet forest takes its family; 80 m from it is beyond PADDOCK_REACH_M: a smooth gum.
    assert fam.tolist() == [species.PINE, species.DRY_SMOOTH, species.DRY_ROUGH, species.WET, species.DRY_SMOOTH]


def test_treeless_country_keeps_its_crowns_as_shrubs_or_the_nearest_forest():
    vmap = species.VegetationMap([Community(box(0, 0, 500, 100), "MBW"), Community(box(500, 0, 600, 100), "RMT"),
                                  Community(box(1000, 0, 1100, 100), "MBS")])
    xy = pts((490, 50), (100, 50), (100, 60), (1050, 50))
    fam = species.families(xy, np.array([30.0, 5.0, 30.0, 25.0]), vmap)
    # Near the rainforest's edge: rainforest. Out in the moor: a low crown is tea-tree, a tall one the nearest
    # woody family. Buttongrass with emergent shrubs is tea-tree whatever the height.
    assert fam.tolist() == [species.RAINFOREST, species.TEA_TREE, species.RAINFOREST, species.TEA_TREE]


def test_no_map_or_no_trees_is_harmless():
    assert species.families(np.zeros((0, 2)), np.zeros(0), species.VegetationMap([])).tolist() == []
    assert species.families(pts((1, 1)), np.array([10.0]), species.VegetationMap([])).tolist() == [species.DRY_SMOOTH]


def test_made_up_trees_go_from_treeless_country_and_thin_in_paddocks():
    rng = np.random.default_rng(0)
    vmap = species.VegetationMap([
        Community(box(0, 0, 100, 100), "MBW"),                                   # 1 ha moor
        Community(box(100, 0, 200, 100), "FAL", notable="EP", notable_name="Eucalyptus pulchella"),  # 1 ha
        Community(box(200, 0, 300, 100), "FAL"),                                  # 1 ha, no scattered trees named
        Community(box(300, 0, 400, 100), "DOB"),                                  # forest
    ])
    corridor = box(0, 0, 400, 100)
    xy = np.vstack([pts(*[(x0 + 5 + 7 * i, 5 + 7 * j) for i in range(13) for j in range(13)]) for x0 in (0, 100, 200, 300)])
    keep = species.keep_made_up(xy, vmap, corridor, rng, mask_agreed=False)
    per = keep.reshape(4, -1).sum(axis=1)
    assert per.tolist() == [0, round(species.PADDOCK_TREES_PER_HA), 0, 169]
    keep = species.keep_made_up(xy, vmap, corridor, rng, mask_agreed=True)
    assert keep.reshape(4, -1).sum(axis=1).tolist() == [0, 6, 6, 169]


def test_crowns_are_capped_at_90_m_in_forest_and_45_m_elsewhere():
    caps = species.height_caps(np.array(["WRE", "DOB", "RMT", "FPH", "FPS", "FAL", "FUR", "NAR", "SLS", "MBW", ""],
                                        dtype=object))
    assert caps.tolist() == [90.0] * 5 + [45.0] * 6


def test_every_family_has_models_and_every_model_is_built():
    assert set(species.FAMILIES) == set(treefx.FAMILY_BANDS)
    for family, bands in treefx.FAMILY_BANDS.items():
        assert bands[-1][0] == np.inf, family
        for _, shares in bands:
            assert set(shares) <= set(treefx.MODEL_NAMES), family


def test_families_pick_their_own_models_by_height():
    names = list(treefx.MODEL_NAMES)
    rng = np.random.default_rng(3)
    fam = np.array([species.WET] * 200 + [species.TEA_TREE] * 200 + [species.DRY_SMOOTH] * 200, dtype=object)
    height = np.r_[np.full(200, 55.0), np.full(200, 6.0), np.full(200, 30.0)]
    model = np.array(names)[treefx.assign_families(height, fam, names, rng)]
    assert set(model[:200]) == {"euc_wet"} and set(model[200:400]) == {"tea_tree"}
    assert set(model[400:]) <= {"euc_tall", "euc_spread"}


def test_the_tallest_crowns_fit_the_wet_eucalypt():
    # A 90 m crown (the height cap) needs no more than Tree FX's 3x scale of the wet-forest eucalypt.
    assert 90.0 / treefx.library()["euc_wet"].box[0] < 3.0


# -- TASVEG ------------------------------------------------------------------

def test_esri_rings_make_parts_and_holes():
    outer = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]  # clockwise: an exterior
    hole = [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]  # counter-clockwise: a hole
    other = [[20, 0], [20, 5], [25, 5], [25, 0], [20, 0]]
    geom = tasveg.esri_polygon([outer, hole, other])
    assert geom.geom_type == "MultiPolygon" and geom.area == pytest.approx(100 - 4 + 25)
    assert not geom.contains(shapely.Point(3, 3)) and geom.contains(shapely.Point(22, 2))


def test_features_are_deduplicated_across_query_pieces():
    ring = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]
    f = {"attributes": {"OBJECTID": 7, "VEGCODE": "dob ", "NOTABLE_TR": None, "NOTABLE_TD": None, "WEED_TYPE": "PI"},
         "geometry": {"rings": [ring]}}
    got = tasveg.from_features([f, f, {"attributes": {"OBJECTID": 8, "VEGCODE": "FAL"}, "geometry": {"rings": []}}])
    assert len(got) == 1 and got[0].code == "DOB" and got[0].weed == "PI" and got[0].notable == ""


def test_a_long_road_is_queried_in_pieces():
    s = np.linspace(0, 40_000, 400)
    centre = np.column_stack([500_000 + s, np.full(len(s), 5_200_000.0)])
    pieces = tasveg._pieces(centre, 260.0)
    assert len(pieces) == 3
    union = shapely.union_all(pieces)
    assert union.contains(shapely.LineString(centre).buffer(255.0))
    for p in pieces:
        assert len(p.exterior.coords) < 200  # simplified, so a request stays small
