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

## Test data

### Vicmap Elevation DEM 10m tile

- Where: `tests/data/vicmap_10m_z15_20112_29614.lerc` (one elevation tile of the Dandenong Ranges, which checks the LERC decoding; see `tests/data/README.md`)
- Source: [Vicmap Elevation DEM 10m](https://discover.data.vic.gov.au/dataset/vicmap-elevation-dem-10m), as served by Vicmap as a Service
- Attribution: Copyright (c) The State of Victoria, Department of Energy, Environment and Climate Action
- Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Change: none. The tile is stored as downloaded.

## Data

Built tracks credit their data sources in `ATTRIBUTION.txt` and `ui_track.json`. See `docs/data-sources.md`.
