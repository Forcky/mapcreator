"""Tasmania 2 m DEM (theLIST, CC BY 3.0 AU).

The statewide DEM is a single ~95 GB cloud-optimised GeoTIFF (GDA94 / MGA55,
heights in AHD) with 512 x 512 pixel internal tiles: about 1 km² and ~1 MB
each. `BlockRaster` reads only the tiles a corridor touches. Point the source
at a local mirror for production builds (`tas_dem_source` in the config).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import shapely

from mapcreator.providers.base import TAS_DEM
from mapcreator.providers.raster import BlockRaster
from mapcreator.providers.tas_list import lidar_index

ATTRIBUTION = TAS_DEM
EPSG = 28355
BLOCK = 512


# The DEM's release. LiDAR finished after it isn't in it.
DEM_DATE = dt.date(2021, 8, 14)
# Stations this close to the edge of a LiDAR footprint count as coarse: the DEM blends its sources there.
EDGE_M = 20.0


class TasDem(BlockRaster):
    resolution_m = 2.0
    # About 65% LiDAR; the rest is a coarser model on the same 2 m grid (see `lidar_mask`).
    mixed_detail = True

    def __init__(self, source: str, cache_dir: Path, *, workers: int = 8, user_agent: str = "mapcreator"):
        super().__init__(source, cache_dir, region_epsg=EPSG, block=BLOCK, workers=workers, cache_name="tas_dem")
        self._cache_dir = Path(cache_dir)
        self._user_agent = user_agent

    def lidar_mask(self, xy: np.ndarray) -> np.ndarray:
        return lidar_mask(xy, self._cache_dir, self._user_agent)

    def surveys(self, xy: np.ndarray, *, around: np.ndarray | None = None) -> list[list[dict]]:
        return surveys(xy, self._cache_dir, self._user_agent, around=around)


def _box(xy: np.ndarray) -> tuple[float, float, float, float]:
    """The index query's box round some points: the same box for the same points, so its answer comes from the cache."""
    x0, y0 = np.asarray(xy, dtype=float).min(axis=0) - 100
    x1, y1 = np.asarray(xy, dtype=float).max(axis=0) + 100
    return float(x0), float(y0), float(x1), float(y1)


def surveys(xy: np.ndarray, cache_dir: Path, user_agent: str, *, around: np.ndarray | None = None) -> list[list[dict]]:
    """For each point (MGA55): the LiDAR projects whose footprint holds it, from theLIST's LiDAR index, oldest first.
    Each is `{"name", "completed"` (an ISO date)`, "in_dem"}`; `in_dem` is whether it was finished before the
    statewide DEM's release, so the DEM can hold it. A later one is a newer survey the DEM doesn't have.

    `around` (points, e.g. the route's) sets the box asked for, so the answer `lidar_mask` cached for the same road is
    used again."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(xy):
        return []
    features = lidar_index.projects(_box(xy if around is None else around), cache_dir, user_agent)
    out: list[list[dict]] = [[] for _ in range(len(xy))]
    for name, done, shape in lidar_index.footprints(features):
        shapely.prepare(shape)
        for i in np.flatnonzero(shapely.contains_xy(shape, xy[:, 0], xy[:, 1])):
            out[int(i)].append({"name": name, "completed": done.isoformat(), "in_dem": done < DEM_DATE})
    return out


def lidar_mask(xy: np.ndarray, cache_dir: Path, user_agent: str) -> np.ndarray:
    """For each point (MGA55): is the statewide DEM LiDAR-derived there? From theLIST's LiDAR index."""
    xy = np.asarray(xy, dtype=float)
    features = lidar_index.projects(_box(xy), cache_dir, user_agent)
    area = lidar_index.coverage(features, DEM_DATE).buffer(-EDGE_M)
    if area.is_empty:
        return np.zeros(len(xy), dtype=bool)
    shapely.prepare(area)
    return shapely.contains_xy(area, xy[:, 0], xy[:, 1])
