"""Tasmania 2 m DEM (theLIST, CC BY 3.0 AU), read block by block.

The statewide DEM is a single ~95 GB cloud-optimised GeoTIFF (GDA94 / MGA55,
heights in AHD) with 512 x 512 pixel internal tiles: about 1 km² and ~1 MB
each. Reading the bounding box of a winding road would fetch hundreds of
tiles it never uses. Instead we fetch only the tiles a corridor touches,
aligned to the file's own tiling, so each is one range request. Each tile is
cached on disk as .npy, so a rebuild reads nothing over the network.

Sampling is bilinear and works across tile edges, so the caller never sees
the tiling. Point the source at a local mirror for production builds
(`tas_dem_source` in the config).
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import LineString, box

from mapcreator.providers.base import TAS_DEM

ATTRIBUTION = TAS_DEM
EPSG = 28355
BLOCK = 512
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}


class TasDem:
    def __init__(self, source: str, cache_dir: Path, *, workers: int = 8):
        self.source = source
        self.workers = workers
        self.cache = cache_dir / "tas_dem" / hashlib.sha256(source.encode()).hexdigest()[:12]
        self.cache.mkdir(parents=True, exist_ok=True)
        with rasterio.Env(**GDAL_ENV), rasterio.open(source) as ds:
            t = ds.transform
            if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
                raise ValueError("rotated DEM transforms are not supported")
            self.x0, self.y0, self.res_x, self.res_y = t.c, t.f, t.a, t.e  # res_y is negative
            self.width, self.height = ds.width, ds.height
            self.nodata = ds.nodata
        self.blocks: dict[tuple[int, int], np.ndarray] = {}

    # -- tiles ---------------------------------------------------------------

    def _block_path(self, bx: int, by: int) -> Path:
        return self.cache / f"{bx}_{by}.npy"

    def _read_block(self, key: tuple[int, int]) -> np.ndarray:
        bx, by = key
        path = self._block_path(bx, by)
        if path.exists():
            return np.load(path)
        col0, row0 = bx * BLOCK, by * BLOCK
        w = min(BLOCK, self.width - col0)
        h = min(BLOCK, self.height - row0)
        with rasterio.Env(**GDAL_ENV), rasterio.open(self.source) as ds:
            arr = ds.read(1, window=Window(col0, row0, w, h)).astype(np.float32)
        if self.nodata is not None:
            arr[arr == self.nodata] = np.nan
        arr[arr < -1000] = np.nan  # float min sentinels seen in the wild
        full = np.full((BLOCK, BLOCK), np.nan, dtype=np.float32)
        full[:h, :w] = arr
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, full)
        tmp.replace(path)
        return full

    def blocks_for(self, geom) -> list[tuple[int, int]]:
        """Tile keys whose footprint intersects a shapely geometry in MGA55."""
        minx, miny, maxx, maxy = geom.bounds
        size_x = BLOCK * self.res_x
        size_y = BLOCK * -self.res_y
        bx0 = int((minx - self.x0) // size_x)
        bx1 = int((maxx - self.x0) // size_x)
        by0 = int((self.y0 - maxy) // size_y)
        by1 = int((self.y0 - miny) // size_y)
        keys = []
        for by in range(max(by0, 0), by1 + 1):
            for bx in range(max(bx0, 0), bx1 + 1):
                x = self.x0 + bx * size_x
                y = self.y0 - by * size_y
                if geom.intersects(box(x, y - size_y, x + size_x, y)):
                    keys.append((bx, by))
        return keys

    def load_corridor(self, line_xy: np.ndarray, buffer_m: float) -> int:
        """Fetch every tile within `buffer_m` of the polyline. Returns the number of tiles."""
        corridor = LineString(line_xy).buffer(buffer_m)
        keys = [k for k in self.blocks_for(corridor) if k not in self.blocks]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for key, arr in zip(keys, pool.map(self._read_block, keys)):
                self.blocks[key] = arr
        return len(self.blocks)

    # -- sampling ------------------------------------------------------------

    def _cells(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        out = np.full(rows.shape, np.nan, dtype=np.float64)
        bx, by = cols // BLOCK, rows // BLOCK
        keys = by * 1_000_000 + bx
        for key in np.unique(keys):
            k = (int(key % 1_000_000), int(key // 1_000_000))
            arr = self.blocks.get(k)
            if arr is None:
                continue
            m = keys == key
            out[m] = arr[rows[m] - k[1] * BLOCK, cols[m] - k[0] * BLOCK]
        return out

    def sample(self, x, y) -> np.ndarray:
        """Bilinear heights (m AHD) at MGA55 points. NaN outside the loaded tiles or the data."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        col = (x - self.x0) / self.res_x - 0.5
        row = (y - self.y0) / self.res_y - 0.5
        c0 = np.floor(col).astype(np.int64)
        r0 = np.floor(row).astype(np.int64)
        fc, fr = col - c0, row - r0
        z00 = self._cells(r0, c0)
        z01 = self._cells(r0, c0 + 1)
        z10 = self._cells(r0 + 1, c0)
        z11 = self._cells(r0 + 1, c0 + 1)
        top = z00 * (1 - fc) + z01 * fc
        bot = z10 * (1 - fc) + z11 * fc
        return top * (1 - fr) + bot * fr
