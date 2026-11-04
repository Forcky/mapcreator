"""NSW State Vegetation Type Map (SVTM, NSW DCCEEW, CC BY 4.0): the Plant Community Type of each 5 m cell.

The SVTM's extant map gives every 5 m cell of NSW one Plant Community Type (PCT), in GDA94 / NSW Lambert
(EPSG:3308). 0 is "Not classified", inside NSW but not native vegetation (cleared land, towns, plantations);
no data is outside NSW. Its attribute table names each PCT's vegetation class and formation. SEED publishes it
as a 4.7 GB zip holding a tiled GeoTIFF and the table (`.tif.vat.dbf`), so mirror the raster once as a COG and
the table as a CSV beside it, with the same name:
`gdal_translate -of COG -co COMPRESS=DEFLATE -co BLOCKSIZE=512 -co OVERVIEW_RESAMPLING=NEAREST <tif>
svtm_c2_0_m2_2_pct_3308.tif` and `ogr2ogr -f CSV -select PCTID,PCTName,vegClass,vegForm
svtm_c2_0_m2_2_pct_3308.csv <tif>.vat.dbf`. Only the PCT numbers and the table are read here; `core.species`
turns them into tree families.

A long road's bounding box at 5 m runs to a hundred million cells, so only the blocks the corridor touches
are read, each with a margin (`halo_m`) for searches that look past a block's edge.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

from mapcreator.providers.base import Attribution

ATTRIBUTION = Attribution(
    dataset='"NSW State Vegetation Type Map - SVTM (Extant)" C2.0.M2.2',
    holder="© State Government of NSW and NSW Department of Climate Change, Energy, the Environment and Water 2020",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
)
EPSG = 3308
NO_DATA = 65535  # outside NSW
NOT_NATIVE = 0  # "Not classified": cleared land, towns and plantations
BLOCK = 512  # cells a side


@dataclass(frozen=True)
class Pct:
    name: str
    veg_class: str
    formation: str


def read_table(path: str | Path) -> dict[int, Pct]:
    """PCT number -> its name, vegetation class and formation, from the mirror's CSV."""
    with open(path, encoding="utf-8", newline="") as f:
        return {int(r["PCTID"]): Pct(r["PCTName"], r["vegClass"], r["vegForm"]) for r in csv.DictReader(f)}


class Svtm:
    """The PCT at points in a region's CRS, read from a local COG for one corridor."""

    def __init__(self, source: str, region_epsg: int):
        self.source = source
        self.table = read_table(Path(source).with_suffix(".csv"))
        self._to_svtm = Transformer.from_crs(region_epsg, EPSG, always_xy=True)
        self._blocks: dict[tuple[int, int], np.ndarray] | None = None

    def load_corridor(self, centre_xy: np.ndarray, reach_m: float, halo_m: float = 0.0) -> None:
        """Read the blocks within `reach_m` of the centreline, each with `halo_m` more all round."""
        ex, ny = self._to_svtm.transform(*np.asarray(centre_xy, dtype=float).T)
        corridor = shapely.LineString(np.column_stack([ex, ny])).buffer(reach_m)
        self._blocks = {}
        with rasterio.open(self.source) as src:
            self._transform, self.cell_m = src.transform, abs(src.res[0])
            self.halo = int(np.ceil(halo_m / self.cell_m))
            w, s, e, n = corridor.bounds
            cols, rows = ~src.transform * (np.array([w, e]), np.array([s, n]))
            (c0, c1), (r1, r0) = np.floor(cols / BLOCK).astype(int), np.floor(rows / BLOCK).astype(int)
            for bi in range(r0, r1 + 1):
                for bj in range(c0, c1 + 1):
                    box = shapely.box(*window_bounds(Window(bj * BLOCK, bi * BLOCK, BLOCK, BLOCK), src.transform))
                    if box.intersects(corridor):
                        h = self.halo
                        window = Window(bj * BLOCK - h, bi * BLOCK - h, BLOCK + 2 * h, BLOCK + 2 * h)
                        self._blocks[bi, bj] = src.read(1, window=window, boundless=True, fill_value=NO_DATA)

    @property
    def blocks(self) -> dict[tuple[int, int], np.ndarray]:
        """The loaded blocks (PCT numbers, with their halo), by block row and column."""
        if self._blocks is None:
            raise RuntimeError("load_corridor first")
        return self._blocks

    def sample(self, grids: dict[tuple[int, int], np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """`grids` (shaped like `blocks`) at each point; NO_DATA outside the loaded blocks."""
        ex, ny = self._to_svtm.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        col, row = ~self._transform * (ex, ny)
        col, row = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
        bi, bj = row // BLOCK, col // BLOCK
        out = np.full(len(col), NO_DATA, dtype=np.int64)
        for key in set(zip(bi.tolist(), bj.tolist())):
            if key in grids:
                mine = (bi == key[0]) & (bj == key[1])
                out[mine] = grids[key][row[mine] - key[0] * BLOCK + self.halo, col[mine] - key[1] * BLOCK + self.halo]
        return out

    def pcts(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The PCT number at each point (NOT_NATIVE for cleared land); NO_DATA outside NSW or the loaded blocks."""
        return self.sample(self.blocks, x, y)
