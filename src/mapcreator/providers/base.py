"""What every data provider declares, so attribution and licence checks are automatic."""

from __future__ import annotations

from dataclasses import dataclass


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

TAS_DEM = Attribution(
    dataset='"Tasmania 2 metre DEM" from theLIST',
    holder="© State of Tasmania",
    licence="CC BY 3.0 AU",
    licence_url="https://creativecommons.org/licenses/by/3.0/au/",
)
