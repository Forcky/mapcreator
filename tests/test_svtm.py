import csv
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from mapcreator.core import species, treefx
from mapcreator.providers.base import Attribution
from mapcreator.providers.nsw import svtm
from mapcreator.providers.nsw.svtm import Pct
from mapcreator.regions import australia

MGA56 = 28356
PAPERBARK, MIXED, GRASS = 3154, 405, 1000
TABLE = [
    (0, "Not classified", "Not classified", "Not classified"),
    (PAPERBARK, "Northern Melaleuca quinquenervia Swamp Forest", "Coastal Swamp Forests", "Forested Wetlands"),
    (MIXED, "Narrow-leaved Ironbark - White Cypress Pine - Buloke tall open forest on lower slopes and flats in the "
            "Pilliga Scrub", "Western Slopes Dry Sclerophyll Forests", "Dry Sclerophyll Forests (Shrubby sub-formation)"),
    (GRASS, "Liverpool Plains grassland mainly on basaltic black earth soils", "Western Slopes Grasslands", "Grasslands"),
]
# The release's formations (C2.0.M2.2): each is typed, or untyped on purpose.
FORMATIONS = {
    "Rainforests", "Wet Sclerophyll Forests (Shrubby sub-formation)", "Wet Sclerophyll Forests (Grassy sub-formation)",
    "Dry Sclerophyll Forests (Shrubby sub-formation)", "Dry Sclerophyll Forests (Shrub/grass sub-formation)",
    "Grassy Woodlands", "Forested Wetlands", "Semi-arid Woodlands (Grassy sub-formation)",
    "Semi-arid Woodlands (Shrubby sub-formation)", "Heathlands", "Arid Shrublands (Acacia sub-formation)",
}
UNTYPED = {"Grasslands", "Freshwater Wetlands", "Saline Wetlands", "Alpine Complex",
           "Arid Shrublands (Chenopod sub-formation)", "Not classified"}


@pytest.fixture
def mirror(tmp_path):
    """A 1 km square SVTM-like raster (5 m, EPSG:3308) around an MGA56 point near Sydney. From west to east:
    200 m of paperbark swamp forest, 600 m cleared, 200 m of the ironbark-cypress-buloke forest; the top 100 m is
    outside NSW."""
    centre = np.array([320_000.0, 6_220_000.0])
    ex, ny = Transformer.from_crs(MGA56, svtm.EPSG, always_xy=True).transform(*centre)
    cells = np.zeros((200, 200), dtype=np.uint16)
    cells[:, :40] = PAPERBARK
    cells[:, 160:] = MIXED
    cells[:20] = svtm.NO_DATA
    path = tmp_path / "svtm.tif"
    with rasterio.open(path, "w", driver="GTiff", width=200, height=200, count=1, dtype="uint16", crs="EPSG:3308",
                       nodata=svtm.NO_DATA, transform=from_origin(ex - 500, ny + 500, 5, 5)) as dst:
        dst.write(cells, 1)
    with open(tmp_path / "svtm.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["PCTID", "PCTName", "vegClass", "vegForm"])
        w.writerows(TABLE)
    return path, centre


def _points(centre, *dxy):
    return centre + np.array(dxy, dtype=float).reshape(-1, 2)


class _Nvis:
    """Just enough of an `NvisMap` for the parts outside NSW."""
    attributions = (Attribution("NVIS", "DCCEEW", "CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),)

    def __len__(self):
        return 2

    def at(self, xy):
        return np.full(len(xy), 58)

    def families(self, xy, height, where=None):
        return np.full(len(xy), species.WATTLE, dtype=object)


def test_pct_names_give_their_trees_families():
    def fams(name, veg_class="", formation="Dry Sclerophyll Forests (Shrubby sub-formation)"):
        return species.svtm_families(Pct(name, veg_class, formation))

    assert fams(TABLE[2][1]) == (species.DRY_ROUGH, species.CONIFER, species.SHE_OAK)
    assert fams("Woronora Plateau Scribbly Gum Woodland") == (species.DRY_SMOOTH,)
    assert fams("Northern Hinterland Grassy River Oak Forest") == (species.SHE_OAK,)
    assert fams("Northern Melaleuca quinquenervia Swamp Forest") == (species.TEA_TREE,)
    # Overlaps go to the first pattern: a rough-barked apple, a snow gum and white cypress pine are one tree each.
    assert fams("River Oak - Rough-barked Apple - red gum - box riparian tall woodland") == (
        species.SHE_OAK, species.DRY_ROUGH, species.DRY_SMOOTH, species.DRY_ROUGH)
    assert fams("Hooked Needlewood - Mulga - Turpentine Bush open shrubland") == (species.WATTLE,)
    # Only the trees before the place count: two bioregions are named after wattles.
    assert fams("Black Box woodland wetland on NSW central and northern floodplains including the Darling Riverine "
                "Plains Bioregion and Brigalow Belt South Bioregion") == (species.DRY_ROUGH,)
    assert fams("Gidgee chenopod woodland on red-brown clays mainly in the Mulga Lands Bioregion") == (species.WATTLE,)
    # Open country that names no tree is treeless, even in a class of trees.
    assert fams("Woollybutt Grass open grassland on red earths of the inland plains", "Sand Plain Mulga Shrublands",
                "Arid Shrublands (Acacia sub-formation)") == ()
    # In wet forest a named eucalypt is wet forest; a rainforest's hoop pine is rainforest.
    assert fams("Kosciuszko Snow Gum-Mountain Gum Moist Forest", formation="Wet Sclerophyll Forests (Grassy "
                "sub-formation)") == (species.SNOW_GUM, species.WET)
    assert fams("Hoop Pine dry rainforest", formation="Rainforests") == (species.RAINFOREST,)
    # Naming no tree we know: the class's family, else the formation's, else none (treeless).
    assert fams("Spinifex linear dune scrub", "Dune Mallee Woodlands", "Semi-arid Woodlands (Shrubby "
                "sub-formation)") == (species.MALLEE,)
    assert fams("Northern Sandplain Wet Heath", "Coastal Heath Swamps", "Freshwater Wetlands") == (species.TEA_TREE,)
    assert fams("Southern Tableland Ranges Boggy Open Woodland", "Montane Bogs and Fens", "Freshwater Wetlands") == ()
    assert fams(TABLE[3][1], TABLE[3][2], TABLE[3][3]) == ()
    assert fams("Not classified", "Not classified", "Not classified") == ()


def test_every_formation_and_class_names_a_family_with_models():
    assert set(species._SVTM_FORMATIONS) == FORMATIONS and not FORMATIONS & UNTYPED
    families = set(species._SVTM_FORMATIONS.values()) | set(species._SVTM_CLASSES.values())
    families |= {f for f, _ in species._NAMED_TREES}
    assert families <= set(treefx.FAMILY_BANDS)
    assert species.SVTM_OUTSIDE == svtm.NO_DATA


def test_reader_loads_only_the_blocks_the_corridor_touches(mirror, monkeypatch):
    path, centre = mirror
    monkeypatch.setattr(svtm, "BLOCK", 16)  # 80 m blocks, so the 1 km square has 13 x 13
    source = svtm.Svtm(str(path), MGA56)
    line = _points(centre, (-450, 0), (450, 0))
    source.load_corridor(line, 40.0, halo_m=200.0)
    assert source.halo == 40
    assert 13 <= len(source.blocks) <= 3 * 13  # a strip of blocks, not the square
    assert all(b.shape == (16 + 80, 16 + 80) for b in source.blocks.values())
    xy = _points(centre, (-400, 0), (-150, 10), (0, -20), (400, 30), (0, 300))
    # Albers-like grid and MGA axes differ a little here, so stay inside each band; the last is off the corridor.
    assert source.pcts(xy[:, 0], xy[:, 1]).tolist() == [PAPERBARK, 0, 0, MIXED, svtm.NO_DATA]


def test_svtm_types_cleared_land_by_its_nearest_forest_and_hands_over_outside_nsw(mirror):
    path, centre = mirror
    source = svtm.Svtm(str(path), MGA56)
    line = _points(centre, (-450, 0), (450, 0), (450, 480))
    source.load_corridor(line, 60.0, halo_m=species.SVTM_NEAREST_M)
    vmap = species.SvtmMap(source, _Nvis(), attributions=(svtm.ATTRIBUTION,))
    assert vmap.attributions == (svtm.ATTRIBUTION,)  # NVIS is credited once it types a tree
    xy = _points(centre, (-400, 0), (-200, 0), (0, 0), (450, 470))
    where = vmap.at(xy)
    assert vmap.codes(where).tolist() == [f"PCT{PAPERBARK}", "PCT0", "PCT0", "MVS58"]
    # 100 m into the clearing takes the paperbark forest's family; 300 m from any forest is a paddock gum; north
    # of the border is NVIS's.
    assert vmap.families(xy, np.full(4, 15.0), where).tolist() == [
        species.TEA_TREE, species.TEA_TREE, species.DRY_SMOOTH, species.WATTLE]
    assert vmap.keep_made_up(xy, None, None, mask_agreed=False).all()
    assert (vmap.height_caps(where) == species.FOREST_HEIGHT_M).all()
    assert len(vmap) == 2 + 2 and vmap.attributions == (svtm.ATTRIBUTION,) + _Nvis.attributions


def test_a_pct_naming_three_trees_draws_each_about_a_third(mirror):
    path, centre = mirror
    source = svtm.Svtm(str(path), MGA56)
    source.load_corridor(_points(centre, (300, -400), (300, 400)), 120.0, halo_m=species.SVTM_NEAREST_M)
    vmap = species.SvtmMap(source)
    rng = np.random.default_rng(3)
    xy = centre + np.column_stack([rng.uniform(380, 480, 3000), rng.uniform(-350, 350, 3000)])
    fams = vmap.families(xy, np.full(len(xy), 15.0))
    share = {f: float(np.mean(fams == f)) for f in (species.DRY_ROUGH, species.CONIFER, species.SHE_OAK)}
    assert sum(share.values()) == pytest.approx(1.0)
    assert all(0.28 < s < 0.39 for s in share.values()), share
    assert vmap.attributions == ()  # no NVIS map, so nothing outside NSW to credit


def test_the_mainland_region_prefers_svtm_in_nsw_only(mirror):
    path, centre = mirror
    vegetation = australia(56).vegetation
    settings = SimpleNamespace(nvis_source="", svtm_source=str(path))
    assert isinstance(vegetation(_points(centre, (-300, 0), (300, 0)), 60.0, settings), species.SvtmMap)
    # A corridor wholly outside NSW (here, off the raster) leaves typing to NVIS: none, without its mirror.
    assert vegetation(_points(centre, (-300, 5000), (300, 5000)), 60.0, settings) is None
    assert vegetation(_points(centre, (-300, 0), (300, 0)), 60.0, SimpleNamespace(nvis_source="", svtm_source="")) \
        is None
