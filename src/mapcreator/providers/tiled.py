"""Elevation fetched from a web service in fixed tiles, cached on disk, and sampled in a region's CRS.

`TiledDem` covers any service that can return a square grid of heights for a box: a WCS GetCoverage, an
ArcGIS ImageServer exportImage, and so on. Each source gives its own CRS, cell size, tile size and a
fetch function. Tiles sit on a fixed grid in the source's CRS (multiples of the tile size), so the cache
is shared by every build that passes the same ground.

Rows run north to south, as GeoTIFFs store them. The grid works the same in lon/lat and in metres.
Samples are cell centres by default; `corner=True` is for sources whose samples sit on the tile edges
(ArcGIS elevation tiles, whose 257th row and column repeat the next tile's first).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import LineString

from mapcreator.providers import cache
from mapcreator.providers.base import Attribution

FETCH_WORKERS = 4

# (west, south, east, north) in the source CRS, and the tile's size in cells -> float32 rows north to south,
# NaN where the source has no data.
Fetcher = Callable[[tuple[float, float, float, float], int], np.ndarray]


class TiledDem:
    def __init__(self, cache_dir: Path, region_epsg: int, *, name: str, crs: int, cell: float, tile: float,
                 fetch: Fetcher, resolution_m: float, attribution: Attribution, corner: bool = False,
                 extent_lonlat: tuple[float, float, float, float] | None = None):
        self.name = name
        self._shift = 0.0 if corner else 0.5
        # Where the source has data at all (west, south, east, north): a road elsewhere never asks it.
        self.extent_lonlat = extent_lonlat
        self.cache = cache_dir / name
        self.to_src = Transformer.from_crs(region_epsg, crs, always_xy=True)
        self.cell, self.tile = cell, tile
        self.tile_px = int(round(tile / cell))
        self.fetch = fetch
        self.resolution_m = resolution_m
        self.attribution = attribution
        self.tiles: dict[tuple[int, int], np.ndarray] = {}
        self.extracted: dict[tuple[int, int], date] = {}  # when each loaded tile was first fetched

    def first_extracted(self) -> str | None:
        """The earliest date any tile in use was fetched, for sources whose credit names it."""
        return min(self.extracted.values()).isoformat() if self.extracted else None

    def _tile(self, key: tuple[int, int]) -> tuple[np.ndarray, date]:
        """A tile and the date it was first fetched, from the cache when it's there."""
        path = self.cache / f"{key[0]}_{key[1]}.npy"
        if path.exists():
            try:
                return np.load(path), date.fromtimestamp(path.stat().st_mtime)
            except (OSError, ValueError, EOFError):
                pass
        w, s = key[0] * self.tile, key[1] * self.tile
        arr = self.fetch((w, s, w + self.tile, s + self.tile), self.tile_px).astype(np.float32)
        if arr.shape != (self.tile_px, self.tile_px):
            raise RuntimeError(f"{self.name} tile {key} came back {arr.shape}, not {self.tile_px} square")
        cache.save_npy(path, arr)
        return arr, date.today()

    def covers(self, lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> bool:
        if self.extent_lonlat is None:
            return True
        w, s, e, n = self.extent_lonlat
        return not (lon_max < w or lon_min > e or lat_max < s or lat_min > n)

    def load_corridor(self, line_xy, buffer_m: float) -> int:
        poly = LineString(line_xy).buffer(buffer_m, quad_segs=4)
        x, y = self.to_src.transform(*np.asarray(poly.exterior.coords).T)
        footprint = shapely.Polygon(np.column_stack([x, y]))
        x0, y0, x1, y1 = footprint.bounds
        shapely.prepare(footprint)
        t = self.tile
        wanted = [(i, j) for i in range(math.floor(x0 / t), math.floor(x1 / t) + 1)
                  for j in range(math.floor(y0 / t), math.floor(y1 / t) + 1)
                  if (i, j) not in self.tiles and footprint.intersects(shapely.box(i * t, j * t, (i + 1) * t, (j + 1) * t))]
        # A few at a time: some services take tens of seconds a tile, and one at a time a long road would
        # spend most of its build waiting. More would be unkind to free public services.
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            for key, (arr, fetched) in zip(wanted, pool.map(self._tile, wanted)):
                self.tiles[key], self.extracted[key] = arr, fetched
        return len(self.tiles)

    def sample(self, x, y) -> np.ndarray:
        """Bilinear heights, any array shape; NaN outside loaded tiles or where the source has no data."""
        shape = np.shape(x)
        sx, sy = self.to_src.transform(np.ravel(np.asarray(x, dtype=float)), np.ravel(np.asarray(y, dtype=float)))
        sx, sy = np.atleast_1d(sx), np.atleast_1d(sy)
        # Continuous cell coordinates over the whole grid (cell centres at half steps).
        col = sx / self.cell - self._shift
        row = -sy / self.cell - self._shift  # rows run north to south
        c0, r0 = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
        fc, fr = col - c0, row - r0
        out = np.zeros(sx.shape)
        for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc), (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
            out += w * self._cells(r0 + dr, c0 + dc)
        return out.reshape(shape) if shape else out

    def _cells(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        px = self.tile_px
        ti = np.floor_divide(cols, px)
        # Tile j spans [j*T, (j+1)*T] northwards; its first row is the northern edge, at global row -(j+1)*px.
        tj = -np.floor_divide(rows, px) - 1
        out = np.full(rows.shape, np.nan)
        for key in set(zip(ti.tolist(), tj.tolist())):
            tile = self.tiles.get(key)
            if tile is None:
                continue
            sel = (ti == key[0]) & (tj == key[1])
            out[sel] = tile[rows[sel] + (key[1] + 1) * px, cols[sel] - key[0] * px]
        return out
