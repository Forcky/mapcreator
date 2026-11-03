"""Meta/WRI canopy mask (providers/meta/canopy.py): tile names, the "canopy near" answer, caching and failures."""

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from mapcreator.providers.meta import canopy as mc

MGA55 = 28355
POINT = (530925.0, 5263889.0)  # on Grasstree Hill Road, Tasmania
RES = 1.1943285669558747  # the dataset's pixel size, Web Mercator metres


@pytest.mark.parametrize("lon, lat, qk", [
    (147.37176, -42.78214, "313010023"),  # Grasstree Hill: the tile the M3 comparison read
    (147.03, -43.26, "313010201"),  # Esperance coast road
    (145.9, -42.15, "313001131"),  # Lyell Highway, Mt Arrowsmith
])
def test_quadkeys_name_the_datasets_tiles(lon, lat, qk):
    assert mc.quadkey(*mc.tile_xy(lon, lat)) == qk


def _tile(folder, value_at_point=12, size=600, corrupt=False):
    """A small stand-in for the dataset's GeoTIFF, named by the point's quadkey, with a 4 x 4 px crown on the point."""
    lon, lat = Transformer.from_crs(MGA55, 4326, always_xy=True).transform(*POINT)
    qk = mc.quadkey(*mc.tile_xy(lon, lat))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{qk}.tif"
    if corrupt:
        path.write_bytes(b"not a tiff")
        return path
    mx, my = Transformer.from_crs(MGA55, 3857, always_xy=True).transform(*POINT)
    half = size // 2
    data = np.zeros((size, size), np.uint8)
    data[half - 2:half + 2, half - 2:half + 2] = value_at_point
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=1, dtype="uint8", crs="EPSG:3857",
                       transform=from_origin(mx - half * RES, my + half * RES, RES, RES)) as d:
        d.write(data, 1)
    return path


def _line():
    return np.array([[POINT[0] - 50, POINT[1]], [POINT[0] + 50, POINT[1]]])


def test_canopy_counts_within_reach_and_is_cached(tmp_path):
    path = _tile(tmp_path / "tiles")
    url = str(tmp_path / "tiles" / "{quadkey}.tif")
    x = np.array([POINT[0], POINT[0] + 3.0, POINT[0] + 20.0])
    y = np.full(3, POINT[1])

    meta = mc.MetaCanopy(tmp_path / "cache", MGA55, url_template=url)
    meta.load_corridor(_line(), 30)
    assert meta.canopy_near(x, y).tolist() == [True, True, False]  # on the crown, 3 m off, 20 m off

    path.unlink()  # a rebuild reads only the cache
    again = mc.MetaCanopy(tmp_path / "cache", MGA55, url_template=url)
    again.load_corridor(_line(), 30)
    assert again.canopy_near(x, y).tolist() == [True, True, False]


def test_low_vegetation_is_not_canopy(tmp_path):
    _tile(tmp_path / "tiles", value_at_point=2)  # 2 m: not above CANOPY_M
    meta = mc.MetaCanopy(tmp_path / "cache", MGA55, url_template=str(tmp_path / "tiles" / "{quadkey}.tif"))
    meta.load_corridor(_line(), 30)
    assert not meta.canopy_near([POINT[0]], [POINT[1]])[0]


def test_no_file_for_the_quadkey_means_no_opinion(tmp_path):
    (tmp_path / "tiles").mkdir()
    meta = mc.MetaCanopy(tmp_path / "cache", MGA55, url_template=str(tmp_path / "tiles" / "{quadkey}.tif"))
    meta.load_corridor(_line(), 30)  # the dataset has no tile here (open sea, say): not an error
    assert meta.canopy_near([POINT[0], POINT[0] + 20], [POINT[1], POINT[1]]).all()  # WorldCover's answer stands
    assert meta.missing


def test_a_failed_read_is_an_error_not_an_empty_forest(tmp_path):
    _tile(tmp_path / "tiles", corrupt=True)
    meta = mc.MetaCanopy(tmp_path / "cache", MGA55, url_template=str(tmp_path / "tiles" / "{quadkey}.tif"))
    with pytest.raises(rasterio.errors.RasterioIOError):
        meta.load_corridor(_line(), 30)  # the scenery stage catches this and keeps WorldCover alone


def test_the_mask_can_be_switched_off(monkeypatch):
    from mapcreator import config
    monkeypatch.setenv("MAPCREATOR_META_CANOPY", "0")
    assert config.load().meta_canopy is False
    monkeypatch.setenv("MAPCREATOR_META_CANOPY", "1")
    assert config.load().meta_canopy is True
