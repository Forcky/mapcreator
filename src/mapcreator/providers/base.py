"""What every data provider declares, so attribution and licence checks are automatic."""

from __future__ import annotations

from dataclasses import dataclass

# Longitudes and latitudes from OSM and the map page are WGS84, which in Australia sits within decimetres of
# GDA2020 and about 1.5 m from GDA94. The work CRSs are GDA94 MGA zones (matching theLIST's DEM and
# footprints and GA's DEM-S), so lon/lat is treated as GDA2020: pyproj then applies the published
# GDA2020 -> GDA94 transformation (7 parameters, 0.01 m), where EPSG:4326 would get a null shift and put
# the road 1.5 m off the elevation and building data.
LONLAT_CRS = 7844


@dataclass(frozen=True)
class Attribution:
    dataset: str
    holder: str
    licence: str
    licence_url: str
    # Whether content derived from this source may ship inside a public track.
    redistributable: bool = True
    note: str = ""

    def line(self) -> str:
        text = f"{self.dataset}: {self.holder}, {self.licence} ({self.licence_url})"
        return f"{text}. {self.note}" if self.note else text


OSM = Attribution(
    dataset="Road centreline and tags",
    holder="© OpenStreetMap contributors",
    licence="ODbL 1.0",
    licence_url="https://www.openstreetmap.org/copyright",
    note="The enriched centreline ships as centreline.geojson under ODbL.",
)

# Footprints from the local OSM index, extruded into scenery (a Produced Work, so attribution is enough).
OSM_BUILDINGS = Attribution(
    dataset="Building footprints",
    holder="© OpenStreetMap contributors",
    licence="ODbL 1.0",
    licence_url="https://www.openstreetmap.org/copyright",
)

TAS_DEM = Attribution(
    dataset='"Tasmania 2 metre DEM" from theLIST',
    holder="© State of Tasmania",
    licence="CC BY 3.0 AU",
    licence_url="https://creativecommons.org/licenses/by/3.0/au/",
)
