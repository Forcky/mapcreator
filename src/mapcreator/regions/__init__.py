"""Which providers serve which part of the map.

Regions are checked in order, best data first:
- Tasmania has 2 m LiDAR elevation, a canopy model, building footprints and
  TasNetworks' transmission lines and towers.
- Anywhere else in Australia falls back to national data: GA's 5 m LiDAR DEM where it covers
  the whole road (the populated coast, the Murray-Darling and towns), else GA's ~30 m DEM-S,
  ESA WorldCover trees, and OSM building footprints and power lines when a local
  OSM index is configured (none without one). That region works in the MGA zone of the
  route's mean longitude, so a route may cross a zone boundary (the Great
  Ocean Road crosses 144°E); over a route of tens of kilometres the extra
  scale error is a few parts in 10,000.

Every stage asks `job_region(job.waypoints)`, so they all agree on the CRS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from pyproj import Transformer

from mapcreator.config import Settings
from mapcreator.providers.base import LONLAT_CRS, OSM, OSM_BUILDINGS, OSM_POWER, Attribution

# A DEM coarser than this can't resolve a road's camber or the drop beside it.
FINE_DEM_M = 5.0



@dataclass(frozen=True)
class Region:
    key: str
    name: str
    epsg: int  # metric CRS the pipeline works in
    bbox: tuple[float, float, float, float]  # west, south, east, north (WGS84)
    elevation: Callable[[Settings], object]
    attributions: tuple[Attribution, ...]
    # Scenery: individual-crown canopy (or None), and building footprints by bbox (or None).
    canopy: Callable[[Settings], object] | None = None
    buildings: Callable[[tuple, Settings], list] | None = None
    buildings_attribution: Attribution | None = None
    # Overhead power lines (`core.power.Line`s) by bbox in the region's CRS, or None; one source per region.
    power: Callable[[tuple, Settings], list] | None = None
    power_attribution: Attribution | None = None
    country: str = "Australia"
    state: str = ""
    height_datum: str = "AHD"

    def contains(self, lon: float, lat: float) -> bool:
        w, s, e, n = self.bbox
        return w <= lon <= e and s <= lat <= n

    def to_xy(self) -> Transformer:
        return Transformer.from_crs(LONLAT_CRS, self.epsg, always_xy=True)

    def to_lonlat(self) -> Transformer:
        return Transformer.from_crs(self.epsg, LONLAT_CRS, always_xy=True)


def _tas_dem(settings: Settings):
    from mapcreator.providers.tas_list.dem import TasDem

    return TasDem(settings.tas_dem_source, settings.cache_dir, user_agent=settings.user_agent)


def _tas_canopy(settings: Settings):
    if not settings.tas_chm_source:
        return None
    from mapcreator.providers.tas_list.canopy import TasCanopy

    return TasCanopy(settings.tas_chm_source, settings.cache_dir)


def _tas_buildings(bbox, settings: Settings):
    from mapcreator.providers.tas_list.buildings import footprints

    return footprints(bbox, settings)


def _tas_power(bbox, settings: Settings):
    from mapcreator.providers.tas_list.power import lines

    return lines(bbox, settings)


def _tas_power_attribution() -> Attribution:
    from mapcreator.providers.tas_list.power import ATTRIBUTION

    return ATTRIBUTION


def _tas_attributions() -> tuple[Attribution, ...]:
    from mapcreator.providers.tas_list.dem import ATTRIBUTION as DEM

    return (OSM, DEM)


def _tas_buildings_attribution() -> Attribution:
    from mapcreator.providers.tas_list.buildings import ATTRIBUTION

    return ATTRIBUTION


TASMANIA = Region(
    key="tas",
    name="Tasmania",
    epsg=28355,  # GDA94 / MGA zone 55, matching the 2 m DEM
    bbox=(143.5, -43.9, 148.6, -39.3),
    elevation=_tas_dem,
    attributions=_tas_attributions(),
    canopy=_tas_canopy,
    buildings=_tas_buildings,
    buildings_attribution=_tas_buildings_attribution(),
    power=_tas_power,
    power_attribution=_tas_power_attribution(),
    state="Tasmania",
)

# GA's DEM-S coverage (its header: 113 to 154 E, 10 to 44 S). Beyond it there's no national elevation.
AUSTRALIA_BBOX = (113.0, -44.0, 154.0, -10.0)


def _osm_buildings(epsg: int) -> Callable[[tuple, Settings], list]:
    """Footprints from the local OSM index, in the region's CRS. Without an index there are none:
    public Overpass is for light development use, so it isn't asked for buildings at all."""

    def buildings(bbox: tuple, settings: Settings) -> list:
        if not settings.osm_index:
            return []
        from mapcreator.providers.osm import index

        return index.buildings(settings.osm_index, bbox, epsg)

    return buildings


def _osm_power(epsg: int) -> Callable[[tuple, Settings], list]:
    """Overhead power lines from the local OSM index, in the region's CRS; none without an index."""

    def lines(bbox: tuple, settings: Settings) -> list:
        if not settings.osm_index:
            return []
        from mapcreator.providers.osm import index

        return index.power_lines(settings.osm_index, bbox, epsg)

    return lines


def mga_zone(lon: float) -> int:
    """UTM/MGA zone number for a longitude (Australia spans zones 49 to 56)."""
    return int(math.floor((lon + 180) / 6)) + 1


@lru_cache(maxsize=None)
def australia(zone: int) -> Region:
    epsg = 28300 + zone  # GDA94 / MGA zone N

    def elevation(settings: Settings):
        from mapcreator.providers.ga.dem_5m import NationalElevation

        return NationalElevation(settings.cache_dir, epsg)

    from mapcreator.providers.ga.dem_s import ATTRIBUTION as DEM_S

    return Region(key=f"au{zone}", name="Australia", epsg=epsg, bbox=AUSTRALIA_BBOX, elevation=elevation,
                  attributions=(OSM, DEM_S), buildings=_osm_buildings(epsg), buildings_attribution=OSM_BUILDINGS,
                  power=_osm_power(epsg), power_attribution=OSM_POWER,
                  height_datum="AHD (approximately; SRTM-derived)")


# What users are told is supported (names and extents for the map page): Tasmania with
# detailed data, and the rest of Australia on the national fallback.
REGIONS = (TASMANIA, australia(55))


class OutOfArea(ValueError):
    pass


def region_for(lon: float, lat: float) -> Region:
    if TASMANIA.contains(lon, lat):
        return TASMANIA
    w, s, e, n = AUSTRALIA_BBOX
    if w <= lon <= e and s <= lat <= n:
        return australia(mga_zone(lon))
    raise OutOfArea(f"({lon:.4f}, {lat:.4f}) is outside the supported area: Australia")


def same_region(points: list[tuple[float, float]]) -> Region:
    """The region for a set of points. All must be in Tasmania, or all on the national fallback; there,
    the MGA zone is the one for the points' mean longitude, so a route can cross a zone boundary."""
    if not points:
        raise OutOfArea("no points given")
    regions = [region_for(lon, lat) for lon, lat in points]
    tas = [r is TASMANIA for r in regions]
    if all(tas):
        return TASMANIA
    if any(tas):
        raise OutOfArea("all waypoints must be in the same region")
    mean_lon = sum(lon for lon, _ in points) / len(points)
    return australia(mga_zone(mean_lon))


def job_region(waypoints) -> Region:
    """The region a job's pipeline works in, from its waypoints (anything with .lon and .lat)."""
    return same_region([(w.lon, w.lat) for w in waypoints])
