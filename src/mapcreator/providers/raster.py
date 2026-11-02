"""Read big single-band rasters (cloud-optimised GeoTIFFs or local mirrors) one tile at a time.

Statewide rasters are tens of gigabytes. Reading the bounding box of a
winding road would fetch hundreds of tiles it never uses. Instead we fetch
only the tiles a corridor touches, aligned to the file's own tiling, so each
is one range request. Each tile is cached on disk as .npy, so a rebuild reads
nothing over the network.

Callers work in their region's metric CRS. If the raster uses another CRS
(e.g. a global WGS84 product), points and corridors are reprojected here.
Sampling works across tile edges, so the caller never sees the tiling.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.windows import Window
from shapely.geometry import LineString, box

from mapcreator.providers import cache
from mapcreator.providers.base import LONLAT_CRS

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}


class BlockRaster:
    def __init__(self, source: str, cache_dir: Path, *, region_epsg: int | None = None, block: int | None = None,
                 categorical: bool = False, workers: int = 8, cache_name: str = "raster"):
        self.source = source
        self.workers = workers
        self.categorical = categorical
        self.cache = cache_dir / cache_name / hashlib.sha256(source.encode()).hexdigest()[:12]
        self.cache.mkdir(parents=True, exist_ok=True)
        with rasterio.Env(**GDAL_ENV), rasterio.open(source) as ds:
            t = ds.transform
            if abs(t.b) > 1e-12 or abs(t.d) > 1e-12:
                raise ValueError("rotated raster transforms are not supported")
            self.x0, self.y0, self.res_x, self.res_y = t.c, t.f, t.a, t.e  # res_y is negative
            self.width, self.height = ds.width, ds.height
            self.nodata = ds.nodata
            self.block = block or ds.block_shapes[0][0]
            raster_epsg = ds.crs.to_epsg() if ds.crs else None
        self.to_raster = None
        if raster_epsg == 4326:
            # WGS84 rasters (DEM-S, WorldCover) are GDA2020 to within decimetres: take the same datum shift
            # as OSM coordinates (providers.base.LONLAT_CRS), not PROJ's null shift from GDA94.
            raster_epsg = LONLAT_CRS
        if region_epsg and raster_epsg and region_epsg != raster_epsg:
            self.to_raster = Transformer.from_crs(region_epsg, raster_epsg, always_xy=True)
        self.blocks: dict[tuple[int, int], np.ndarray] = {}

    # -- tiles ---------------------------------------------------------------

    def _read_block(self, key: tuple[int, int]) -> np.ndarray:
        bx, by = key
        path = self.cache / f"{bx}_{by}.npy"
        if path.exists():
            try:
                return np.load(path)
            except (OSError, ValueError, EOFError):
                pass  # cut short by an older writer: read the tile again
        b = self.block
        col0, row0 = bx * b, by * b
        w, h = min(b, self.width - col0), min(b, self.height - row0)
        with rasterio.Env(**GDAL_ENV), rasterio.open(self.source) as ds:
            raw = ds.read(1, window=Window(col0, row0, w, h))
        dtype = raw.dtype if self.categorical else np.float32
        arr = raw.astype(dtype)
        if not self.categorical:
            if self.nodata is not None:
                arr[raw == self.nodata] = np.nan
            arr[arr < -1000] = np.nan  # float-min sentinels seen in the wild
        full = np.zeros((b, b), dtype=dtype) if self.categorical else np.full((b, b), np.nan, dtype=np.float32)
        full[:h, :w] = arr
        cache.save_npy(path, full)
        return full

    def blocks_for(self, geom) -> list[tuple[int, int]]:
        """Tile keys whose footprint intersects a shapely geometry in the raster's CRS."""
        minx, miny, maxx, maxy = geom.bounds
        size_x, size_y = self.block * self.res_x, self.block * -self.res_y
        bx0, bx1 = int((minx - self.x0) // size_x), int((maxx - self.x0) // size_x)
        by0, by1 = int((self.y0 - maxy) // size_y), int((self.y0 - miny) // size_y)
        keys = []
        last_bx = (self.width - 1) // self.block
        last_by = (self.height - 1) // self.block
        for by in range(max(by0, 0), min(by1, last_by) + 1):
            for bx in range(max(bx0, 0), min(bx1, last_bx) + 1):
                x, y = self.x0 + bx * size_x, self.y0 - by * size_y
                if geom.intersects(box(x, y - size_y, x + size_x, y)):
                    keys.append((bx, by))
        return keys

    def _to_raster_geom(self, geom):
        if self.to_raster is None:
            return geom
        return shapely.transform(geom, lambda c: np.column_stack(self.to_raster.transform(c[:, 0], c[:, 1])))

    def load_corridor(self, line_xy, buffer_m: float) -> int:
        """Fetch every tile within `buffer_m` of the polyline (region CRS). Returns tiles held."""
        corridor = self._to_raster_geom(LineString(line_xy).buffer(buffer_m))
        keys = [k for k in self.blocks_for(corridor) if k not in self.blocks]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for key, arr in zip(keys, pool.map(self._read_block, keys)):
                self.blocks[key] = arr
        return len(self.blocks)

    # -- sampling ------------------------------------------------------------

    def _rc(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.to_raster is not None:
            x, y = self.to_raster.transform(x, y)
            x, y = np.asarray(x), np.asarray(y)
        return (y - self.y0) / self.res_y - 0.5, (x - self.x0) / self.res_x - 0.5

    def _cells(self, rows: np.ndarray, cols: np.ndarray, fill) -> np.ndarray:
        dtype = np.int32 if self.categorical else np.float64
        out = np.full(rows.shape, fill, dtype=dtype)
        b = self.block
        bx, by = cols // b, rows // b
        keys = by * 1_000_000 + bx
        for key in np.unique(keys):
            k = (int(key % 1_000_000), int(key // 1_000_000))
            arr = self.blocks.get(k)
            if arr is None:
                continue
            m = keys == key
            out[m] = arr[rows[m] - k[1] * b, cols[m] - k[0] * b]
        return out

    def sample(self, x, y) -> np.ndarray:
        """Bilinear values at region-CRS points. NaN outside loaded tiles or the data."""
        row, col = self._rc(x, y)
        c0, r0 = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
        fc, fr = col - c0, row - r0
        z00 = self._cells(r0, c0, np.nan)
        z01 = self._cells(r0, c0 + 1, np.nan)
        z10 = self._cells(r0 + 1, c0, np.nan)
        z11 = self._cells(r0 + 1, c0 + 1, np.nan)
        return (z00 * (1 - fc) + z01 * fc) * (1 - fr) + (z10 * (1 - fc) + z11 * fc) * fr

    def sample_nearest(self, x, y, fill: int = 0) -> np.ndarray:
        """Nearest-cell values (for class maps). `fill` outside loaded tiles."""
        row, col = self._rc(x, y)
        return self._cells(np.round(row).astype(np.int64), np.round(col).astype(np.int64), fill)
