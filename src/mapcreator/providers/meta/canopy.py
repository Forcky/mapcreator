"""Meta and WRI's High Resolution Canopy Height Maps (2024), CC BY 4.0: where the canopy is, worldwide.

Canopy heights estimated by machine learning from Maxar satellite imagery: about 1 m pixels in Web Mercator
(EPSG:3857), uint8 whole metres, one GeoTIFF per zoom-9 quadkey on public cloud storage
(https://registry.opendata.aws/dataforgood-fb-forests). The files are deflated in one-row strips 65,536 px
wide, not in tiles, so a corridor is read through one open file per quadkey, in chunks along the road, and
only the answer this module gives is cached.

Checked against theLIST's 2 m LiDAR canopy model on two Tasmanian roads (mapcreator#8): it matches only
12-13% of real crowns, reads about 0.6 of their height and much less in tall forest, so it never places or
sizes a tree. But where it sees canopy there usually is canopy. Requiring it as well as ESA WorldCover's tree
class brought the forest's extent from about 55% too much to within about 20% of the LiDAR's, and raised the
share of 10 m cells classed right from 0.74-0.77 to 0.81. That mask is its only use.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy.ndimage import maximum_filter
from shapely.geometry import LineString, box

from mapcreator.providers.base import Attribution

ATTRIBUTION = Attribution(
    dataset="High Resolution Canopy Height Maps (canopy extent only)",
    holder="Meta and World Resources Institute (WRI) 2024; source imagery © 2016 Maxar",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
)
URL = "/vsicurl/https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/chm/{quadkey}.tif"
ZOOM = 9
CANOPY_M = 2.0  # a pixel above this is canopy...
REACH_M = 5.0  # ...and counts for every point within this distance (a 10 m square), as checked in #8
CHUNK = 1024  # pixels; read and cached in chunks this size
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    # A chunk's rows are whole strips; neighbouring chunks along the road reuse them from this cache.
    "GDAL_CACHEMAX": 512 * 1024 * 1024,  # bytes
}


def tile_xy(lon: float, lat: float, zoom: int = ZOOM) -> tuple[int, int]:
    """Web Mercator tile column and row (the slippy-map scheme)."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def quadkey(x: int, y: int, zoom: int = ZOOM) -> str:
    digits = []
    for i in range(zoom, 0, -1):
        mask = 1 << (i - 1)
        digits.append(str((1 if x & mask else 0) + (2 if y & mask else 0)))
    return "".join(digits)


def _missing(e: Exception) -> bool:
    """The dataset has no file for this quadkey (open sea, no imagery), as opposed to a failed read."""
    text = str(e)
    return "404" in text or "No such file" in text or "does not exist" in text


class MetaCanopy:
    def __init__(self, cache_dir: Path, region_epsg: int, *, url_template: str = URL):
        self.cache_dir = Path(cache_dir) / "meta_canopy"
        self.url_template = url_template
        self.to_merc = Transformer.from_crs(region_epsg, 3857, always_xy=True)
        self.to_lonlat = Transformer.from_crs(region_epsg, 4326, always_xy=True)
        self.missing: set[str] = set()
        self.tiles: dict[str, tuple] = {}  # quadkey -> (transform, width, height)
        self.chunks: dict[tuple[str, int, int], np.ndarray] = {}  # (quadkey, cx, cy) -> bool "canopy near"

    def _quadkeys(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        return np.array([quadkey(*tile_xy(a, b)) for a, b in zip(np.atleast_1d(lon), np.atleast_1d(lat))])

    def load_corridor(self, line_xy: np.ndarray, buffer_m: float) -> None:
        """Read (or take from the cache) every chunk the corridor touches. Raises if a file can't be read for any
        reason except not existing."""
        poly = LineString(line_xy).buffer(buffer_m + REACH_M)
        minx, miny, maxx, maxy = poly.bounds
        lons, lats = self.to_lonlat.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
        x0, y1 = tile_xy(min(lons), min(lats))  # south-west corner: the highest tile row
        x1, y0 = tile_xy(max(lons), max(lats))
        merc = shapely.transform(poly, lambda c: np.column_stack(self.to_merc.transform(c[:, 0], c[:, 1])))
        shapely.prepare(merc)
        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                self._load_tile(quadkey(tx, ty), merc, (min(lats) + max(lats)) / 2)

    def _load_tile(self, qk: str, merc, mid_lat: float) -> None:
        if qk in self.missing:
            return
        cache = self.cache_dir / qk
        if (cache / "missing").exists():
            self.missing.add(qk)
            return
        meta = cache / "tile.json"
        if meta.exists():
            info = json.loads(meta.read_text(encoding="utf-8"))
            self.tiles[qk] = (Affine(*info["transform"]), info["width"], info["height"])
            wanted = self._wanted(qk, merc)
            cached = {(cx, cy): self._chunk(None, self._window(qk, cx, cy), 0, cache / f"{cx}_{cy}.npy")
                      for cx, cy in wanted}
            if all(v is not None for v in cached.values()):
                for (cx, cy), v in cached.items():  # a rebuild: all cached, nothing read over the network
                    self.chunks[(qk, cx, cy)] = v
                return
        try:
            with rasterio.Env(**GDAL_ENV), rasterio.open(self.url_template.format(quadkey=qk)) as ds:
                self.tiles[qk] = (ds.transform, ds.width, ds.height)
                cache.mkdir(parents=True, exist_ok=True)
                meta.write_text(json.dumps({"transform": list(ds.transform)[:6], "width": ds.width,
                                            "height": ds.height}), encoding="utf-8")
                reach_px = max(1, math.ceil(REACH_M / (abs(ds.transform.a) * math.cos(math.radians(mid_lat)))))
                for cx, cy in self._wanted(qk, merc):  # row by row, so neighbouring chunks share strips
                    self.chunks[(qk, cx, cy)] = self._chunk(ds, self._window(qk, cx, cy), reach_px,
                                                            cache / f"{cx}_{cy}.npy")
        except rasterio.errors.RasterioIOError as e:
            if not _missing(e):
                raise
            self.missing.add(qk)
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "missing").touch()

    def _window(self, qk: str, cx: int, cy: int) -> Window:
        _, w, h = self.tiles[qk]
        return Window(cx * CHUNK, cy * CHUNK, min(CHUNK, w - cx * CHUNK), min(CHUNK, h - cy * CHUNK))

    def _wanted(self, qk: str, merc) -> list[tuple[int, int]]:
        t, w, h = self.tiles[qk]
        out = []
        for cy in range(math.ceil(h / CHUNK)):
            for cx in range(math.ceil(w / CHUNK)):
                if merc.intersects(box(*rasterio.windows.bounds(self._window(qk, cx, cy), t))):
                    out.append((cx, cy))
        return out

    @staticmethod
    def _chunk(ds, win: Window, reach_px: int, path: Path) -> np.ndarray | None:
        """A chunk's "canopy near" mask, from the cache or, with `ds` open, read and cached. None if `ds` is None
        and the cache doesn't hold it."""
        if path.exists():
            try:
                packed = np.load(path)
                return np.unpackbits(packed, count=win.height * win.width).reshape(win.height, win.width).astype(bool)
            except (OSError, ValueError, EOFError):
                pass  # cut short by an older writer: read it again
        if ds is None:
            return None
        # read with a border, so canopy just over a chunk edge still counts
        pad = Window(win.col_off - reach_px, win.row_off - reach_px, win.width + 2 * reach_px, win.height + 2 * reach_px)
        raw = ds.read(1, window=pad, boundless=True, fill_value=0)
        near = maximum_filter(raw > CANOPY_M, size=2 * reach_px + 1)[reach_px:-reach_px, reach_px:-reach_px]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, np.packbits(near))
        tmp.replace(path)
        return near

    def canopy_near(self, x, y) -> np.ndarray:
        """True where Meta sees canopy above CANOPY_M within REACH_M of the point. Where it has no data (no file
        for the quadkey, or a chunk the corridor didn't load), True: the caller's own answer stands there."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        out = np.ones(x.shape, bool)
        if not len(x):
            return out
        lon, lat = self.to_lonlat.transform(x, y)
        mx, my = self.to_merc.transform(x, y)
        mx, my = np.asarray(mx), np.asarray(my)
        qks = self._quadkeys(lon, lat)
        for qk in np.unique(qks):
            if qk not in self.tiles:
                continue
            t, w, h = self.tiles[qk]
            m = np.nonzero(qks == qk)[0]
            col = np.floor((mx[m] - t.c) / t.a).astype(np.int64)
            row = np.floor((my[m] - t.f) / t.e).astype(np.int64)
            inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
            m, row, col = m[inside], row[inside], col[inside]
            keys = (row // CHUNK) * 1_000_000 + col // CHUNK
            for key in np.unique(keys):
                chunk = self.chunks.get((qk, int(key % 1_000_000), int(key // 1_000_000)))
                if chunk is None:
                    continue
                k = keys == key
                out[m[k]] = chunk[row[k] % CHUNK, col[k] % CHUNK]
        return out
