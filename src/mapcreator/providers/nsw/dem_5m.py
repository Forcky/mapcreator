"""NSW Spatial Services' statewide 5 m DEM, which also covers the ACT, where GA's 5 m LiDAR has holes.

`NSW_5M_Elevation` is "full state cover of 5 metre post spacing elevation derived from stereo imagery"
(AHD), served as an ArcGIS ImageServer. GA's 5 m LiDAR DEM has nothing for all of Canberra, the Blue
Mountains around Katoomba or the Putty Road, and this fills those. It's photogrammetric and smoother than
LiDAR, so it ranks after GA's 5 m.

NSW's 1 m and 2 m LiDAR DEMs are only distributed through ELVIS orders, which a build can't place.

Licence: Spatial Services applies CC BY 3.0 AU and asks to be credited as "© Spatial Services [date of
extraction]"; the date is when a tile was first fetched. Its product sheet adds that its material "should
only be loaded on any external cloud platforms if Spatial Services' Intellectual Property remains
unchanged", so whether derived tracks may be hosted is a judgement to make before turning this on
(see `providers.ga.dem_5m.fine_sources`).
"""

from __future__ import annotations

from pathlib import Path

from mapcreator.providers.arcgis import export_image_fetcher
from mapcreator.providers.base import Attribution
from mapcreator.providers.tiled import Fetcher, TiledDem

URL = "https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_5M_Elevation/ImageServer"
CRS = 4283
CELL_DEG = 0.00005  # about 5 m
TILE_DEG = 0.02
EXTENT_LONLAT = (141.0, -37.51, 154.0, -28.0)  # from the service's own metadata


class NswDem5m(TiledDem):
    def __init__(self, cache_dir: Path, region_epsg: int, *, fetch: Fetcher | None = None):
        super().__init__(cache_dir, region_epsg, name="nsw_dem_5m", crs=CRS, cell=CELL_DEG, tile=TILE_DEG,
                         fetch=fetch or export_image_fetcher(URL, CRS), resolution_m=5.0,
                         attribution=self._credit(None), extent_lonlat=EXTENT_LONLAT)

    @staticmethod
    def _credit(extracted: str | None) -> Attribution:
        return Attribution(
            dataset="NSW 5 metre Digital Elevation Model (NSW_5M_Elevation)",
            holder=f"© Spatial Services {extracted}" if extracted else "© Spatial Services",
            licence="CC BY 3.0 AU",
            licence_url="https://creativecommons.org/licenses/by/3.0/au/",
        )

    def load_corridor(self, line_xy, buffer_m: float) -> int:
        n = super().load_corridor(line_xy, buffer_m)
        self.attribution = self._credit(self.first_extracted())
        return n
