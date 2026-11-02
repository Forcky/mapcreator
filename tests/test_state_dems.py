"""State elevation sources: Victoria's tiles, the ArcGIS exportImage fetcher, NSW's dated credit, extents."""

import io
from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from mapcreator.providers import arcgis
from mapcreator.providers.nsw.dem_5m import NswDem5m
from mapcreator.providers.vic import dem_10m

TILE = Path(__file__).parent / "data" / "vicmap_10m_z15_20112_29614.lerc"


def test_a_real_vicmap_tile_decodes_to_256_square_heights():
    a = dem_10m.decode(TILE.read_bytes())
    assert a.shape == (256, 256) and a.dtype == np.float32
    assert np.isfinite(a).all() and 350 < float(a.min()) < float(a.max()) < 600  # the Dandenong Ranges


def test_the_tiled_grid_asks_for_the_right_web_mercator_tile(monkeypatch):
    urls = []

    def fake_get(url, **kw):
        urls.append(url)
        return httpx.Response(200, content=TILE.read_bytes(), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    i, j = 29614 - 2 ** 14, 2 ** 14 - 20112 - 1
    t = dem_10m.TILE_M
    dem_10m.tile_fetch((i * t, j * t, (i + 1) * t, (j + 1) * t), 256)
    assert urls[0].endswith("/tile/15/20112/29614")


def test_vicmap_tiles_with_no_data_read_as_nan(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(404, request=httpx.Request("GET", url)))
    assert np.isnan(dem_10m.tile_fetch((0, 0, dem_10m.TILE_M, dem_10m.TILE_M), 256)).all()


def geotiff(values: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", width=values.shape[1], height=values.shape[0], count=1, dtype="float32",
                      crs="EPSG:4283", transform=from_origin(150.0, -33.0, 0.001, 0.001)) as ds:
            ds.write(values.astype("float32"), 1)
        buf.write(mem.read())
    return buf.getvalue()


def test_export_image_reads_heights_and_turns_no_data_into_nan(monkeypatch):
    values = np.full((4, 4), 123.5)
    values[0, 0] = arcgis.NO_DATA
    seen = {}

    def fake_get(url, params=None, **kw):
        seen.update(params)
        return httpx.Response(200, content=geotiff(values), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    a = arcgis.export_image_fetcher("https://example.org/ImageServer", 4283,
                                    mosaic_rule={"where": "res_value<=1"})((150.0, -33.004, 150.004, -33.0), 4)
    assert np.isnan(a[0, 0]) and (a[np.isfinite(a)] == 123.5).all()
    assert seen["pixelType"] == "F32" and seen["imageSR"] == "4283" and '"res_value<=1"' in seen["mosaicRule"]


def test_export_image_refuses_an_error_dressed_as_a_tiff(monkeypatch):
    # A request the server can't serve comes back HTTP 200, image/tiff, with a short JSON error as the body.
    body = b'{"error":{"code":500,"message":"Error exporting image"}}'
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(
        200, content=body, headers={"content-type": "image/tiff"}, request=httpx.Request("GET", url)))
    with pytest.raises(RuntimeError, match="not a GeoTIFF"):
        arcgis.export_image_fetcher("https://example.org/ImageServer", 4283)((0, 0, 1, 1), 4)


def test_nsw_is_credited_with_the_date_its_tiles_were_extracted(tmp_path):
    from datetime import date

    def fetch(bbox, size):
        return np.full((size, size), 500.0, dtype=np.float32)

    dem = NswDem5m(tmp_path, 28355, fetch=fetch)
    xy = np.array([[690000.0, 6085000.0], [690500.0, 6085300.0]])  # near Canberra
    dem.load_corridor(xy, 50)
    assert dem.attribution.holder == f"© Spatial Services {date.today().isoformat()}"


def test_a_source_is_only_asked_about_roads_inside_its_extent(tmp_path):
    from mapcreator.providers.vic.dem_10m import VicDem10m

    vic = VicDem10m(tmp_path, 28355, fetch=lambda *a: pytest.fail("asked about a road outside Victoria"))
    assert vic.covers(144.9, -37.9, 145.0, -37.8)  # Melbourne
    assert not vic.covers(151.1, -33.9, 151.2, -33.8)  # Sydney
