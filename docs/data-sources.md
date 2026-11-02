# Data sources and licences

Checked November 2026. Licences change, so recheck a source's own licence statement before adding a provider.

## Rules
- A source is usable only if its licence allows building **and redistributing** derived tracks.
- Being free of charge doesn't change that:
  - NoDerivatives (ND) imagery can't become textures.
  - NonCommercial (NC) terms bind every downstream user.
  - Google, Esri, Bing and Mapbox terms forbid this use outright.
- **OpenStreetMap (ODbL):**
  - A track is a Produced Work, so it needs attribution.
  - The enriched centreline is shipped as `centreline.geojson` under ODbL.
  - OSM is never merged with another road dataset for the same feature.

## Tasmania (theLIST, © State of Tasmania)
Attribution format: `"<dataset name>" from theLIST © State of Tasmania`, with the matching CC licence.

| Need | Dataset | Licence | Notes |
|---|---|---|---|
| Elevation | Tasmania statewide 2 m DEM (2021) | CC BY 3.0 AU | About 65% LiDAR-derived; GDA94 / MGA55, AHD. Mirror locally; it's one ~95 GB COG |
| Road cross-checks | LIST Transport Segments | CC BY 3.0 AU | Surface, class, route number, one-way. No width or lanes |
| Speed limits | State road speed limits | CC BY 3.0 AU | State roads only; not legally authoritative |
| Corridor bound | State road reserve; cadastral road casements | CC BY 3.0 AU | Legal corridor, not the sealed width |
| Trees | Tree Canopy Height 2 m | CC BY 3.0 AU | Includes buildings and other objects. Shipped deflated inside a 26 GB zip, so it can't be range-read: mirror `TAS_CHM_2m.tif` and set `tas_chm_source`. Without it, trees come from ESA WorldCover |
| Vegetation type | TASVEG 5.0 | CC BY 3.0 AU | TASVEG Live is restricted; don't use it |
| Buildings | LIST 2D Building Polygons | CC BY 3.0 AU | Heights are mostly empty; derive them from canopy height |
| Imagery (open) | AerialPhoto2020–2024 basemaps | CC BY 3.0 AU | Coverage varies by year and area |
| Imagery (reference only) | Orthophoto, AerialPhoto2025/2026, Raster 2015–19 | CC BY-NC-ND 3.0 AU | Never textures; don't extract from them |
| Map display | Topographic, Hillshade basemaps | CC BY 3.0 AU | Tasmap25K is NC-ND |

## Australia-wide

| Need | Dataset | Licence | Notes |
|---|---|---|---|
| Roads | OpenStreetMap (Geofabrik extracts) | ODbL | `lanes`, `width`, `surface` coverage is uneven |
| Elevation, national | Geoscience Australia SRTM 1-second DEM-H | CC BY 4.0 | ~30 m; on public cloud storage |
| Elevation, fine | GA 5 m LiDAR DEM; state LiDAR via ELVIS | CC BY (varies by state) | NSW, SA, ACT open; VIC 1 m is paid (10 m free); QLD newer than 3 years is paid; WA is not open |
| Elevation, fallback | Copernicus DEM GLO-30 | Copernicus DEM licence | Surface model (includes trees and buildings); carry its attribution and notice |
| Imagery as texture | NSW Imagery | CC BY | Best open high-resolution option outside Tasmania |
| Imagery as texture | DEA GeoMAD (Landsat) | CC BY 4.0 | 30 m colour only |
| Imagery | QLD public imagery | CC BY-SA | Share-alike carries into derived textures |
| Imagery | VIC Vicmap, WA Landgate | Paid / personal use | Not usable |
| Vegetation | ESA WorldCover 10 m; NVIS 7.0 | CC BY 4.0 | WorldCover tree cover (class 10) is the tree fallback everywhere; read by window from its public COGs |
| Buildings | OSM; Overture (filter sources) | ODbL | Exclude Google Open Buildings from Overture |
| Textures | Poly Haven, ambientCG | CC0 | |
| Map display | OpenFreeMap | OSM / OpenMapTiles attribution | Free, no keys |
