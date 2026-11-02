"""Queensland's 1 m LiDAR DEMs, newest survey first, from the state's DEM time-series ImageServer.

`DEM_TimeSeries_AllUsers` holds 536 public surveys (2006-2024), 518 of them at 1 m or finer, with SRTM
30 m and bathymetry filling everything else. The mosaic rule here asks for surveys at 1 m or finer only,
newest first, so where there's no LiDAR the answer is no data rather than a quiet SRTM fill.

Licence: unresolved. The service and its QSpatial record say "Creative Commons Attribution-Share-Alike
(CC BY-SA)" with an indemnity acknowledgement, while the underlying survey records say CC BY 4.0. Share-
alike would carry into every track built on it, so this stays switched off until Queensland confirms
which applies (see `providers.ga.dem_5m.fine_sources`).
"""

from __future__ import annotations

from pathlib import Path

from mapcreator.providers.arcgis import export_image_fetcher
from mapcreator.providers.base import Attribution
from mapcreator.providers.tiled import Fetcher, TiledDem

URL = "https://spatial-img.information.qld.gov.au/arcgis/rest/services/Elevation/DEM_TimeSeries_AllUsers/ImageServer"
CRS = 4283
CELL_DEG = 0.00001  # about 1.1 m north-south, 1 m east-west in Queensland
TILE_DEG = 0.005  # 500 x 500 cells a request
EXTENT_LONLAT = (137.92, -29.24, 159.99, -9.14)  # from the service's own metadata
MOSAIC_RULE = {"mosaicMethod": "esriMosaicAttribute", "sortField": "zorder", "sortValue": "0", "ascending": True,
               "where": "res_value<=1"}
ATTRIBUTION = Attribution(
    dataset="Queensland LiDAR Digital Elevation Models (DEM_TimeSeries_AllUsers)",
    holder="Includes material © State of Queensland (Department of Natural Resources and Mines, Manufacturing, "
           "and Regional and Rural Development); © Commonwealth of Australia (Geoscience Australia)",
    licence="CC BY-SA (to be confirmed)",
    licence_url="https://creativecommons.org/licenses/by-sa/4.0/",
    note="The State gives no warranty in relation to the data and accepts no liability for any loss, damage or "
         "costs relating to any use of the data.",
)


class QldDem1m(TiledDem):
    def __init__(self, cache_dir: Path, region_epsg: int, *, fetch: Fetcher | None = None):
        super().__init__(cache_dir, region_epsg, name="qld_dem_1m", crs=CRS, cell=CELL_DEG, tile=TILE_DEG,
                         fetch=fetch or export_image_fetcher(URL, CRS, mosaic_rule=MOSAIC_RULE), resolution_m=1.0,
                         attribution=ATTRIBUTION, extent_lonlat=EXTENT_LONLAT)
