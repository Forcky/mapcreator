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
  - Production reads OSM from a local index built from a Geofabrik extract (below). Public Overpass is for light development use only.

## Tasmania (theLIST, © State of Tasmania)
Attribution format: `"<dataset name>" from theLIST © State of Tasmania`, with the matching CC licence.

| Need | Dataset | Licence | Notes |
|---|---|---|---|
| Elevation | Tasmania statewide 2 m DEM (2021) | CC BY 3.0 AU | About 65% LiDAR-derived; the rest is interpolated from photogrammetric contours, spot heights and the coastline at about 10 m, on the same 2 m grid. GDA94 / MGA55, AHD. Mirror locally; it's one ~95 GB COG |
| Where that DEM is LiDAR | theLIST LiDAR Coverage Index - All Projects (`Public/Indexes/MapServer/97`) | CC BY 3.0 AU | Project footprints with completion dates, queried by box and cached (`providers/tas_list/lidar_index.py`). Projects finished before the DEM's release (14 Aug 2021) mark its fine stretches; elsewhere the profile is treated as coarse (harder smoothing, 15% grade cap), the camber is left flat and no guard rails are placed. Nothing from it ships in a track |
| Road cross-checks | LIST Transport Segments | CC BY 3.0 AU | Surface, class, route number, one-way. No width or lanes |
| Speed limits | State road speed limits | CC BY 3.0 AU | State roads only; not legally authoritative |
| Corridor bound | State road reserve; cadastral road casements | CC BY 3.0 AU | Legal corridor, not the sealed width |
| Trees | Tree Canopy Height 2 m | CC BY 3.0 AU | Whole metres above ground. Each crown is a local maximum over 4 m (touching maxima merged, tiles filtered with their neighbours), and every crown within the 600 m scenery reach (`core.reach`) is kept at its own height, up to 90 m (`CROWN_HEIGHT_M`); in Tasmania, up to 90 m in TASVEG's forest codes (eucalypt, rainforest, plantation) and 45 m elsewhere (`species.height_caps`); footprints mask out roofs. Includes buildings and other objects. Shipped deflated inside a 26 GB zip, so it can't be range-read: mirror `TAS_CHM_2m.tif` and set `tas_chm_source`. Without it, trees come from ESA WorldCover |
| Vegetation type | TASVEG 5.0 (NRE Tas) | CC BY 3.0 AU, credited "TASVEG 5.0 from www.theLIST.tas.gov.au ©State of Tasmania" | Communities (`VEGCODE`, 157 codes in 11 groups), scattered paddock trees by species (`NOTABLE_TR`) and weed trees (`WEED_TYPE`), from `Public/NaturalEnvironmentV2/MapServer/9`. Queried with the corridor itself, in 15 km pieces, generalised to 2 m by the server and cached (`providers/tas_list/tasveg.py`). The licence is in the metadata statement inside the open-data zip; the REST layer's copyright text is empty. Gives each tree its kind (`core.species`): canopy-model crowns are only retyped, never deleted; made-up WorldCover trees are also dropped in treeless codes and thinned in paddocks and towns, after the per-km cap, so treeless country stays open and the total falls. Only class labels reach a track, so polygons drawn from Google Earth or Esri imagery (`SOURCE_TYP`) are fine. "Indicative only and most polygons have not been field verified", at 1:25,000: boundaries are good to tens of metres. TASVEG Live is restricted; don't use it |
| Buildings | LIST 2D Building Polygons | CC BY 3.0 AU | Heights are mostly empty; derive them from canopy height |
| Power lines | TasNetworks Overhead Transmission Line; TasNetworks Tower Locations | CC BY 3.0 AU | Transmission only: a point at every tower, and two kinds of polyline. Route lines run tower to tower; "... kV Transmission Circuit" lines are drawn beside them and give voltage and circuit count. Read from `Public/Infrastructure/MapServer/59` and `/61` by box and cached (`providers/tas_list/power.py`); the open data download of the same datasets states the licence in its `readme.txt`. TasNetworks' distribution poles and low-voltage conductors are also on theLIST but **restricted to emergency services**: never read them |
| Imagery (open) | AerialPhoto2020–2024 basemaps | CC BY 3.0 AU | Coverage varies by year and area |
| Imagery (reference only) | Orthophoto, AerialPhoto2025/2026, Raster 2015–19 | CC BY-NC-ND 3.0 AU | Never textures; don't extract from them |
| Map display | Topographic, Hillshade basemaps | CC BY 3.0 AU | Tasmap25K is NC-ND |

## Australia-wide

| Need | Dataset | Licence | Notes |
|---|---|---|---|
| Roads | OpenStreetMap (Geofabrik extracts) | ODbL | Read from the local OSM index (below). The route keeps each way's `highway`, `ref`, `name`, `lanes`, `width`, `surface`, `narrow`, `lane_markings`, `oneway`, `maxspeed`, `bridge`, `tunnel` and `layer`. `surface` decides sealed or unsealed (`core/section.py`); in Tasmania it's on 98-100% of drivable ways (Derwent Valley and Huon, Nov 2026). `lanes` and `width` are patchy, and on unsealed roads almost never there (17 and 10 of 917 unclassified ways), so the class default sets most widths. `narrow=yes` and `lane_markings=no` are rare |
| Elevation, national | Geoscience Australia 1-second SRTM DEM-S | CC BY 4.0 | ~30 m smoothed, vegetation offsets removed; one 38 GB COG on DEA's public bucket, read by window. The fallback outside Tasmania: too coarse for camber, bridges or rails, so tracks get a 240 m profile filter and a 15% grade cap |
| Elevation, fine (in use) | Geoscience Australia, DEM of Australia derived from LiDAR 5 Metre Grid ([doi:10.26186/89644](https://doi.org/10.26186/89644)) | CC BY 4.0 | About 245,000 km²: the populated coast, the Murray-Darling floodplains and towns (surveys 2001-2015). Read in 0.02° tiles from GA's WCS (`services.ga.gov.au`, GDA94 lon/lat); no LiDAR comes back as 0.0. Used for a whole road when it covers at least 98% of the centreline (`providers/ga/dem_5m.py`), with DEM-S filling small holes; otherwise the road uses DEM-S throughout |
| Elevation, Victoria (in use) | Vicmap Elevation DEM 10m (Vicmap as a Service) | CC BY 4.0 | Statewide, with LiDAR folded in since 2016; ±5 m vertical. Read as ArcGIS LERC elevation tiles (Web Mercator, zoom 15; no data is 0.0). Used for a Victorian road GA's 5 m doesn't cover, such as the Dandenongs (`providers/vic/dem_10m.py`). Victoria's 1 m DEM is licensed for a fee |
| Elevation, Queensland (written, off) | Queensland LiDAR DEMs, `DEM_TimeSeries_AllUsers` ImageServer | CC BY-SA on the service, CC BY 4.0 on the survey records | 1 m from 518 public surveys (2006-2024), newest first. Works (`providers/qld/dem_1m.py`); off until Queensland confirms the licence, because share-alike would carry into every track |
| Elevation, NSW and ACT (written, off) | NSW Spatial Services `NSW_5M_Elevation` ImageServer | CC BY 3.0 AU, credited "© Spatial Services [date of extraction]" | Statewide 5 m from stereo imagery, covering the ACT too: fills GA's holes in Canberra, the Blue Mountains and the Putty Road (`providers/nsw/dem_5m.py`). Off until the product sheet's term about loading its material "on any external cloud platforms" only unchanged is settled |
| Elevation, finer | State LiDAR via ELVIS | CC BY (varies by state) | NSW's 1-2 m, SA's 1 m and the ACT's LiDAR are only distributed through ELVIS orders (email delivery), which a build can't place; its API sits behind a firewall. WA's is personal-use only |
| Elevation, fallback | Copernicus DEM GLO-30 | Copernicus DEM licence | Surface model (includes trees and buildings); carry its attribution and notice |
| Imagery as texture | NSW Imagery | CC BY | Best open high-resolution option outside Tasmania |
| Imagery as texture | DEA GeoMAD (Landsat) | CC BY 4.0 | 30 m colour only |
| Imagery | QLD public imagery | CC BY-SA | Share-alike carries into derived textures |
| Imagery | VIC Vicmap, WA Landgate | Paid / personal use | Not usable |
| Vegetation | ESA WorldCover 10 m | CC BY 4.0 | WorldCover tree cover (class 10) is the tree fallback everywhere; read by window from its public COGs. It gives cover, not crowns, so its made-up trees are thinned with distance (`TREE_BANDS`) and capped per km (`MAX_TREES_PER_KM`) |
| Vegetation type | NVIS 7.0 Major Vegetation Subgroups (DCCEEW), extant | CC BY 4.0, "© Commonwealth of Australia (Department of Climate Change, Energy, the Environment and Water) 2024"; users "are required to acknowledge the Australian Government as the source of the data" | Outside Tasmania it types the made-up trees: each 100 m cell's one dominant subgroup (85 classes) gives the family (`species.NVIS_SUBGROUPS`). Published as a zipped file geodatabase (148 MB) that can't be range-read, and GDAL in the worker image has no OpenFileGDB driver, so it's converted once to a COG (GDAL's own image, `gdal_translate -of COG`, native GDA2020 Albers, EPSG:9473) and mirrored: set `nvis_source` (`providers/dcceew/nvis.py`). Too coarse to delete trees by, and the publisher says its non-native classes "should not be used for analyses", so it only types. Without the mirror, trees are typed by height alone |
| Vegetation type (NSW) | NSW State Vegetation Type Map (SVTM), extant, release C2.0.M2.2 (NSW DCCEEW, via SEED) | CC BY 4.0, "© State Government of NSW and NSW Department of Climate Change, Energy, the Environment and Water 2020" | In NSW it types the made-up trees ahead of NVIS: each 5 m cell's Plant Community Type (1,687 in the table), whose name lists its dominant trees (`species.svtm_families`), else its vegetation class or formation. Published as a 4.7 GB zip holding a tiled GeoTIFF (GDA94 / NSW Lambert, EPSG:3308) and its attribute table, so the raster is converted once to a COG and the table to a CSV beside it, and mirrored: set `svtm_source` (`providers/nsw/svtm.py`). Its REST service only renders the map, with no cell values. Only blocks along the corridor are read. It only types, like NVIS, and outside NSW NVIS takes over |
| Canopy extent | Meta and WRI High Resolution Canopy Height Maps (2024) | CC BY 4.0 | Only as a mask on WorldCover: a tree cell is kept where Meta also sees canopy above 2 m within 5 m. Machine-learning heights from Maxar imagery (about 2016-2020), ~1 m pixels in Web Mercator, one GeoTIFF per zoom-9 quadkey on public S3, deflated in one-row strips. Against theLIST's LiDAR (#8) it matches only 12-13% of real crowns and reads about 0.6 of their height, so it never places or sizes a tree; as a mask it brought trees standing on real canopy from 81% to 88% (Grasstree Hill) and 77% to 87% (Esperance coast road). Attribution: "Meta and World Resources Institute (WRI) 2024; source imagery © 2016 Maxar". `MAPCREATOR_META_CANOPY=0` turns it off |
| Buildings | OSM; Overture (filter sources) | ODbL | OSM footprints (closed `building=*` ways) from the local OSM index are the fallback outside Tasmania; none without an index. If Overture is added, exclude Google Open Buildings |
| Power lines | OSM `power=line` and `power=minor_line` | ODbL | Outside Tasmania, from the local OSM index; none without one. Mapped in patches (about 526,000 poles Australia-wide, Nov 2026), so many roads get none. Every vertex is a support, drawn as the tower or pole its node is tagged (`power=tower`, `power=pole`; the index keeps which vertices are which). A line drawn with vertices further apart than a span can be (250 m distribution, 650 m transmission) gets evenly spaced supports between vertices that aren't both mapped supports, counted as `inferred_supports`. The lines used ship as `power.geojson`. Not used: utility pole data from Queensland and Western Australia (non-commercial or internal use only); Geoscience Australia's Electricity Infrastructure (lines of 66 kV and up) is allowed but not wired in yet |
| Textures | Poly Haven, ambientCG | CC0 | |
| Map display | OpenFreeMap | OSM / OpenMapTiles attribution | Free, no keys |

## OpenStreetMap: the production source

| | |
|---|---|
| Source | Geofabrik's Australia extract, `https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf` (about 0.9 GB) |
| Licence | ODbL 1.0, © OpenStreetMap contributors (`https://www.openstreetmap.org/copyright`) |
| What we keep | Drivable highway ways (the classes in `overpass.DRIVABLE`) with all their tags and node ids; closed `building=*` ways with only their shape and height tags (`building*`, `height`, `min_height`, `roof:*`); `power=line` and `power=minor_line` ways with their kind, voltage, circuit and location tags. Building relations (multipolygons) aren't kept yet. Index layout 2 added the power lines; a layout 1 index still serves roads and buildings |
| Where | A SQLite file built once by `python -m mapcreator.providers.osm.index build` and named by `osm_index` (`MAPCREATOR_OSM_INDEX`). See `docs/dev-handover.md` |
| Freshness | The index records the extract's file name and its OSM data timestamp; `python -m mapcreator.providers.osm.index query` prints them. Refresh by building a new index from a newer extract |

ODbL obligations, checked against the licence text (opendatacommons.org, ODbL 1.0) in November 2026:

- **§4.3, Produced Works.** Publicly using a Produced Work needs a notice that content came from the database and is available under the ODbL. Every track's `ATTRIBUTION.txt` carries the OSM lines: "Road centreline and tags" and, when OSM buildings or power lines were used, "Building footprints" or "Power lines".
- **§4.6, Derivative Databases.** Treat the index as a Derivative Database: it is a filtered, rearranged copy of the extract. Publicly using a Derivative Database, or a Produced Work from one, means offering recipients either the whole Derivative Database or "the method of making the alterations to the Database (such as an algorithm)". The method is `providers/osm/index.py` in this repository, applied to the Geofabrik extract named in the index, and the centreline each track uses ships as `centreline.geojson` (with `power.geojson` for any OSM power lines).
- **Still to do:** tracks don't yet point recipients at that method. Once this repository is published, `ATTRIBUTION.txt` should say where.
- The index itself stays on the build machines and isn't published.
