# mapcreator: notes for Claude Code

Generates Assetto Corsa tracks from real roads using open data. Read `README.md`, `docs/architecture.md` and `docs/data-sources.md` first.

## This repo is written to be published
- **No private information, ever:** hostnames, IP addresses, subnets, tunnel or account IDs, secret names, personal paths, or anything about the maintainer's own infrastructure. Deployment specifics live in a separate private repo.
- Configuration comes from environment variables or a gitignored `config.local.yaml`. Commit only `*.example` files with placeholders.
- Ported third-party code keeps a provenance header (source URL, licence, commit) and an entry in `THIRD_PARTY_NOTICES.md`. Don't copy code from repos without a licence. For MS-PL sources (e.g. AcTools), re-implement the format; don't copy the code.
- Strip EXIF from any photo before it goes into `docs/`.

## Architecture rules
- `core/` knows nothing about where data comes from. It takes arrays and geometry and returns meshes and files.
- Every data source is a provider under `providers/`. Each provider declares its licence, its attribution text, and whether derived content may be redistributed. A provider that can't be redistributed must never feed textures or geometry into a shipped track.
- `regions/` maps a bounding box to providers. Adding a state means adding providers and a region entry, not editing `core/`.
- Each pipeline stage writes its output to disk and can be rerun on its own.
- The build pipeline must run on Linux without a GPU, Blender, ksEditor or the game. The in-game test harness (`harness/`, Windows) is for development testing only and never part of a public build.
- Job input is coordinates, enums and booleans only, validated with pydantic. No URLs, paths or free text reach a shell or a file parser.

## Data rules
- No Google, Esri, Bing or Mapbox content as build inputs. That includes Street View, satellite tiles, elevation APIs and 3D tiles.
- NoDerivatives (ND) imagery is never turned into textures. NonCommercial data is avoided, because its terms bind every downstream user.
- OpenStreetMap: ship `centreline.geojson` under ODbL with every track, and never merge OSM with another road dataset for the same feature.
- Public Overpass, Nominatim and OSRM demo servers are for light development use only. Use local extracts.

## Conventions
- Python 3.12. Project to a metric CRS for the region (e.g. EPSG:28355 for Tasmania), subtract a local origin, then convert to AC axes (Y up, metres).
- Australian spelling in docs and user-facing text.
- Check AC format details and data licences against primary sources, not memory.
