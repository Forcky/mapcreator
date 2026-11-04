import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from mapcreator.core import species, treefx
from mapcreator.providers.dcceew import nvis


@pytest.fixture
def mirror(tmp_path):
    """A 40 x 40 cell NVIS-like raster (100 m, EPSG:9473) around an MGA55 point. From west to east: 1 km of wet
    sclerophyll (3), 2 km of "Cleared, non-native vegetation, buildings" (98), 1 km of Leptospermum forest (58)."""
    centre = np.array([500_000.0, 5_200_000.0])
    ex, ny = Transformer.from_crs(28355, nvis.EPSG, always_xy=True).transform(*centre)
    cells = np.full((40, 40), 98, dtype=np.uint8)
    cells[:, :10] = 3
    cells[:, 30:] = 58
    path = tmp_path / "mvs.tif"
    with rasterio.open(path, "w", driver="GTiff", width=40, height=40, count=1, dtype="uint8", crs=f"EPSG:{nvis.EPSG}",
                       transform=from_origin(ex - 2000, ny + 2000, 100, 100)) as dst:
        dst.write(cells, 1)
    return path, centre


def _points(centre, *dx):
    return np.column_stack([centre[0] + np.array(dx, dtype=float), np.full(len(dx), centre[1])])


def test_subgroups_come_from_the_cell_under_each_point(mirror):
    path, centre = mirror
    source = nvis.Nvis(str(path), 28355)
    source.load_corridor(_points(centre, 0), 1900.0)
    # Albers and MGA55 axes differ by a degree or so here, so stay well inside each band.
    xy = _points(centre, -1500, 0, 1500, 50_000)
    assert source.subgroups(xy[:, 0], xy[:, 1]).tolist() == [3, 98, 58, nvis.NO_DATA]  # the last is outside


def test_nvis_types_trees_but_never_removes_or_caps_them(mirror):
    path, centre = mirror
    source = nvis.Nvis(str(path), 28355)
    source.load_corridor(_points(centre, 0), 1900.0)
    vmap = species.NvisMap(source)
    xy = _points(centre, -1500, -850, 0, 850, 1500)
    where = vmap.at(xy)
    # In the cleared band, a tree 150 m from the forest takes the forest's family; one 1 km from any is a gum.
    assert vmap.families(xy, np.full(5, 20.0), where).tolist() == [
        species.WET, species.WET, species.DRY_SMOOTH, species.TEA_TREE, species.TEA_TREE]
    assert vmap.codes(where).tolist() == ["MVS03", "MVS98", "MVS98", "MVS98", "MVS58"]
    assert vmap.keep_made_up(xy, None, None, mask_agreed=False).all()
    assert (vmap.height_caps(where) == species.FOREST_HEIGHT_M).all()
    assert len(vmap) == 3


def test_every_subgroup_names_one_family_with_models_and_heights():
    numbers = [n for ns in species._SUBGROUPS.values() for n in ns]
    assert len(numbers) == len(set(numbers))  # no subgroup in two families
    assert not set(numbers) & species.NVIS_UNTYPED  # and none both typed and untyped
    assert set(species.NVIS_SUBGROUPS.values()) <= set(species.FAMILIES)
    assert set(species.FAMILIES) == set(species.HEIGHT_M) == set(treefx.FAMILY_BANDS)
    # The key forest subgroups of the report's section 6a.
    assert species.NVIS_SUBGROUPS[3] == species.WET and species.NVIS_SUBGROUPS[1] == species.RAINFOREST
    assert species.NVIS_SUBGROUPS[58] == species.TEA_TREE and species.NVIS_SUBGROUPS[26] == species.SHE_OAK


def test_made_up_trees_get_their_familys_height():
    rng = np.random.default_rng(5)
    fam = np.array([species.TEA_TREE] * 500 + [species.WET] * 500, dtype=object)
    h = species.made_up_heights(fam, rng)
    lo, hi = species.HEIGHT_M[species.TEA_TREE][2:], species.HEIGHT_M[species.WET][2:]
    assert h[:500].min() >= lo[0] and h[:500].max() <= lo[1]
    assert h[500:].min() >= hi[0] and h[500:].max() <= hi[1]
    assert np.median(h[500:]) > 4 * np.median(h[:500])
