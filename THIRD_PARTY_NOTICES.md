# Third-party notices

## Code included in this repository

None yet. Any ported code will carry a provenance header (source URL, licence, commit) and an entry here.

## Format references

The KN5, `fast_lane.ai` and track-folder writers in `src/mapcreator/core/` were written from a description of the file formats. That description was compiled by reading these projects. No code was copied from them.

| Project | Licence | Used for |
|---|---|---|
| [gottschalkfelix4-source/road-to-assetto](https://github.com/gottschalkfelix4-source/road-to-assetto) | Apache-2.0 | KN5 layout, material values, in-game load-test approach |
| [KinG-InFeT/blender-assetto-corsa-track-generator](https://github.com/KinG-InFeT/blender-assetto-corsa-track-generator) | Apache-2.0 | Multi-layout folder structure |
| [johnzara/ac-track-studio](https://github.com/johnzara/ac-track-studio) | MIT | `fast_lane.ai` v7 layout, AI line height above the road, DDS mip encoding |
| [gro-ove/actools](https://github.com/gro-ove/actools) (`AcTools/Kn5File`, `AcTools/AiFile`, `AcManager/Tools`) | MS-PL | Cross-checking the KN5 and AI layouts against Content Manager's reader; CM's install, launch and ui_track.json rules |
| [stormbuster/assetto-corsa-mcp](https://github.com/stormbuster/assetto-corsa-mcp) | MIT | The teleport-and-check approach of the CSP probe app (`harness/csp_app/mc_probe`), written separately from the CSP Lua SDK docs |

## Data

Built tracks credit their data sources in `ATTRIBUTION.txt` and `ui_track.json`. See `docs/data-sources.md`.
