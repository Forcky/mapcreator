"""Which providers serve which part of the map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pyproj import Transformer

from mapcreator.config import Settings
from mapcreator.providers.base import OSM, Attribution


@dataclass(frozen=True)
class Region:
    key: str
    name: str
    epsg: int  # metric CRS the pipeline works in
    bbox: tuple[float, float, float, float]  # west, south, east, north (WGS84)
    elevation: Callable[[Settings], object]
    attributions: tuple[Attribution, ...]

    def contains(self, lon: float, lat: float) -> bool:
        w, s, e, n = self.bbox
        return w <= lon <= e and s <= lat <= n

    def to_xy(self) -> Transformer:
        return Transformer.from_crs(4326, self.epsg, always_xy=True)

    def to_lonlat(self) -> Transformer:
        return Transformer.from_crs(self.epsg, 4326, always_xy=True)


def _tas_dem(settings: Settings):
    from mapcreator.providers.tas_list.dem import TasDem

    return TasDem(settings.tas_dem_source, settings.cache_dir)


def _tas_attributions() -> tuple[Attribution, ...]:
    from mapcreator.providers.tas_list.dem import ATTRIBUTION as DEM

    return (OSM, DEM)


TASMANIA = Region(
    key="tas",
    name="Tasmania",
    epsg=28355,  # GDA94 / MGA zone 55, matching the 2 m DEM
    bbox=(143.5, -43.9, 148.6, -39.3),
    elevation=_tas_dem,
    attributions=_tas_attributions(),
)

REGIONS = (TASMANIA,)


class OutOfArea(ValueError):
    pass


def region_for(lon: float, lat: float) -> Region:
    for region in REGIONS:
        if region.contains(lon, lat):
            return region
    names = ", ".join(r.name for r in REGIONS)
    raise OutOfArea(f"({lon:.4f}, {lat:.4f}) is outside the supported regions: {names}")
