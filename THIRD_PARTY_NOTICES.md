# Third-party notices

## Code included in this repository

Any ported code carries a provenance header (source URL, licence, commit) and an entry here.

### MapLibre GL JS 5.24.0

- Where: `src/mapcreator/service/static/vendor/maplibre-gl/` (the map page's map library)
- Source: https://cdn.jsdelivr.net/npm/maplibre-gl@5.24.0/dist/ (`maplibre-gl.js`, `maplibre-gl.css`, `LICENSE.txt`)
- Project: https://github.com/maplibre/maplibre-gl-js
- Licence: BSD-3-Clause, full text in `LICENSE.txt` beside the files
- Change: the trailing `//# sourceMappingURL=maplibre-gl.js.map` line was removed from `maplibre-gl.js`, because the source map isn't shipped. Nothing else was changed.
- SHA-256 as downloaded:
  - `maplibre-gl.js` `45a9b07a9189ce56054c620a947ccf41e291e58c95e9b61533b740aaa65ee5cb` (after the change: `8b2c6bfadb8cc2d720aea2816ac368e51ab75dbb81ee3fac2f8b97256ebd1585`)
  - `maplibre-gl.css` `ab1e70d59ec40465bae7e7030da2f3ccf28133fd502e62bd598eefbadfd7a732`
  - `LICENSE.txt` `ee5fc05a0677eaf69601d2c7db0d9ecd6cc27c3abc1d0733bc9ed34707cf8ef2`

To update: download the same three files for the new version, delete the `sourceMappingURL` line, update the version and hashes here and the hash check in `tests/test_service_static.py`.

## Format references

The KN5, `fast_lane.ai`, Tree FX and track-folder writers in `src/mapcreator/core/`, and the checks `core/validate.py` makes on a finished track, were written from a description of the file formats. That description was compiled by reading these projects. No code was copied from them.

| Project | Licence | Used for |
|---|---|---|
| [gottschalkfelix4-source/road-to-assetto](https://github.com/gottschalkfelix4-source/road-to-assetto) | Apache-2.0 | KN5 layout, material values, in-game load-test approach |
| [KinG-InFeT/blender-assetto-corsa-track-generator](https://github.com/KinG-InFeT/blender-assetto-corsa-track-generator) | Apache-2.0 | Multi-layout folder structure |
| [johnzara/ac-track-studio](https://github.com/johnzara/ac-track-studio) | MIT | `fast_lane.ai` v7 layout, AI line height above the road, DDS mip encoding; the `ROAD` and `GRASS` values in `surfaces.ini` (`core/package.py`) are the ones its `pipeline/scripts/export_ac.py` writes |
| [gro-ove/actools](https://github.com/gro-ove/actools) (`AcTools/Kn5File`, `AcTools/AiFile`, `AcTools.Render/Kn5SpecificSpecial/AiLaneObject.cs`, `AcManager/Tools`, `AcTools.ExtraKn5Utils/Helpers/AcTreeModelConverter.cs`) | MS-PL | Cross-checking the KN5 and AI layouts against Content Manager's reader, and what the AI line's side distances mean (CM draws the road's outline from them); CM's install, launch and ui_track.json rules; the layout of Custom Shaders Patch Tree FX models, which `core/treefx.py` re-implements |
| [stormbuster/assetto-corsa-mcp](https://github.com/stormbuster/assetto-corsa-mcp) | MIT | The teleport-and-check approach of the CSP probe app (`harness/csp_app/mc_probe`), written separately from the CSP Lua SDK docs |

## Methods

- **The AI's racing line** (`src/mapcreator/core/racing_line.py`) follows the minimum-curvature method of Heilmeier, Wischnewski, Hermansdorfer, Betz, Lienkamp and Lohmann, "Minimum curvature trajectory planning and control for an autonomous race car", Vehicle System Dynamics 58(10), 2020, pp. 1497-1527, doi:10.1080/00423114.2019.1631455. It was written from the paper. The authors' implementation, [TUMFTM/global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization), is LGPL-3.0; none of its code was read or copied.
- Its solver is the textbook projected Newton method of Bertsekas, "Projected Newton methods for optimization problems with simple constraints", SIAM J. Control and Optimization 20(2), 1982, inside an augmented Lagrangian.
- **xxHash32** (`xxh32` in `src/mapcreator/core/treefx.py`), which Tree FX uses to identify a model's textures, implements Yann Collet's XXH32 algorithm as described in the [xxHash specification](https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md). It is a short Python implementation of the published algorithm; none of xxHash's code (BSD-2-Clause) is included.

## Design references

### Police car liveries and crests

- Where: `src/mapcreator/core/police_car.py` (the police cars parked at closed side roads; see `docs/ac-format.md`, "Police cars")
- What: each Australian police service's marked car is drawn by our own code from shapes: the chequer, the lettering (Pillow's built-in Aileron font, CC0) and a simplified rendition of each crest. No image of a real crest or livery is copied, traced or embedded. The pages below were read, and their photos and badge images viewed, for reference only.
- The services' names, crests and liveries are theirs. mapcreator is not affiliated with or endorsed by any police service.
- Sources (read November 2026; photos dated 2018-2026):
  - **Tasmania Police:** media release "New look police vehicles to spearhead Easter road safety campaign", 17 Apr 2019, https://www.police.tas.gov.au/news-events/media-releases/new-look-police-vehicles-to-spearhead-easter-road-safety-campaign/ ; https://en.wikipedia.org/wiki/Tasmania_Police ; https://pulsetasmania.com.au/news/tasmania-police-upgrades-highway-patrol-fleet-with-sleek-new-bmw ; news photos at https://pulsetasmania.com.au/news/tasmania-police-cars-damaged-as-officers-raid-house-in-derwent-park/ , https://pulsetasmania.com.au/news/police-car-damaged-in-serious-crash-on-tasman-highway-near-sorell and https://pulsetasmania.com.au/news/tasmania-police-arrest-man-after-kettering-campground-incident
  - **NSW Police Force:** https://en.wikipedia.org/wiki/New_South_Wales_Police_Force ; https://brand-genome.github.io/brands/new-south-wales-police-force/ (PMS 287) ; https://www.carexpert.com.au/car-news/petrol-bmw-suvs-to-patrol-new-south-wales-highways ; photos https://commons.wikimedia.org/wiki/File:NSW_General_Duties_Hyundai_Santa_Fe_Active_ES10.jpg and https://commons.wikimedia.org/wiki/File:Code_3_Pursuit_light_bar_NSW_Police_configuration_(front).jpg
  - **Victoria Police:** https://en.wikipedia.org/wiki/Victoria_Police ; https://crwflags.com/fotw/flags/au_vicp.html ; photos https://commons.wikimedia.org/wiki/File:(AUS-Victoria)_Victoria_Police_Toyota_Kluger_VIC-1XN-7MB_2025-12-09.jpg and https://commons.wikimedia.org/wiki/File:(AUS-Victoria)_Victoria_Police_Hyundai_Santa_Fe_VIC-1ZL-2DP_2025-12-08.jpg
  - **Queensland Police Service:** "New extreme high-vis patrol cars join QPS fleet", 18 May 2023, https://fleetautonews.com.au/new-extreme-high-vis-patrol-cars-join-qps-fleet/ ; https://statements.qld.gov.au/statements/101808 ; https://psnews.com.au/?p=105206 ; https://en.wikipedia.org/wiki/Queensland_Police_Service ; photos https://commons.wikimedia.org/wiki/File:AU-Q-Police-vehicle_Toyota-Camry-2024_20260210.jpg and https://commons.wikimedia.org/wiki/File:AU-Q-Police-vehicle_Kia-Sorento-2021_20260210.jpg
  - **South Australia Police:** https://en.wikipedia.org/wiki/South_Australia_Police ; https://crwflags.com/fotw/flags/au_sap.html ; https://www.carexpert.com.au/car-news/police-in-this-australian-state-to-hide-in-plain-sight ; photos https://commons.wikimedia.org/wiki/File:Kia_Sorento_(MQ4)_of_the_South_Australia_Police_,_Adelaide,_24_October_2026_(028A5122).jpg and https://commons.wikimedia.org/wiki/File:Isuzu_D-Max_of_the_South_Australia_Police,_May_2026_(DSCF7207).jpg
  - **Western Australia Police Force:** https://fleetautonews.com.au/wa-police-update-fleet-branding/ ; https://www.insidestategovernment.com.au/wa-police-unveil-new-x5-fleet ; https://www.carexpert.com.au/car-news/hyundai-ioniq-5-ev-toyota-mirai-hydrogen-car-join-wa-police ; photo https://commons.wikimedia.org/wiki/File:AUS_Perth,_Stirling,_The_Esplanade_046.jpg
  - **Northern Territory Police:** https://pfes.nt.gov.au/node/26870 ; https://www.carexpert.com.au/car-news/subaru-wrx-goes-from-hunted-to-hunter-joins-nt-police-fleet ; https://www.insidestategovernment.com.au/nt-road-policing-rolls-out-hi-tech-fleet-additions/ ; https://crwflags.com/fotw/flags/au_ntp.html ; photo https://commons.wikimedia.org/wiki/File:Outback_Cops.jpg
  - **ACT Policing (Australian Federal Police):** "Police cars get decal makeover", 16 May 2019, https://psnews.com.au/police-cars-get-decal-makeover/19474/ ; https://en.wikipedia.org/wiki/ACT_Policing ; https://canberradaily.com.au/two-new-high-tech-bmws-join-act-road-policing-fleet ; photo https://commons.wikimedia.org/wiki/File:ACTPOL_Subaru_Outbck_(2).jpg
  - **Crests:** the badge images on each service's Wikipedia article, viewed for their general form (outline, crown, wreath, rings and emblem) only.

## Test data

### Vicmap Elevation DEM 10m tile

- Where: `tests/data/vicmap_10m_z15_20112_29614.lerc` (one elevation tile of the Dandenong Ranges, which checks the LERC decoding; see `tests/data/README.md`)
- Source: [Vicmap Elevation DEM 10m](https://discover.data.vic.gov.au/dataset/vicmap-elevation-dem-10m), as served by Vicmap as a Service
- Attribution: Copyright (c) The State of Victoria, Department of Energy, Environment and Climate Action
- Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Change: none. The tile is stored as downloaded.

## Data

Built tracks credit their data sources in `ATTRIBUTION.txt` and `ui_track.json`. See `docs/data-sources.md`.
