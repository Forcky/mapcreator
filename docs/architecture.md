# Architecture

## Pipeline

Every stage reads files and writes files, so each one can be tested and rerun on its own. The CLI and the hosted service use the same `job.json`.

```
job.json (waypoints, width option, layouts)
   │
1. route      → route.geojson          snapped OSM centreline + per-segment tags
2. profile    → centreline_3d.csv      DEM sampling, smoothing with a grade-change limit, camber from cross-slope
3. width      → widths per station     OSM width → lanes × lane width → class default; optional +15% / +30%
4. meshgen    → meshes                 road chunks (≤ 65,535 vertices each), verges, adaptive terrain corridor, walls, AC dummies
5. scenery    → scatter.json           trees from canopy height + vegetation type; buildings from footprints
6. kn5        → <track>.kn5            numpy KN5 writer; DDS texture library compressed once at image build
7. ai_line    → ai/fast_lane.ai        computed offline from the centreline with a curvature speed profile
8. package    → <track>.zip            data/*.ini, ui/*, map.png, ATTRIBUTION.txt, centreline.geojson, layouts
9. validate   → report.json            static checks; optional in-game harness during development
```

No stage needs a GPU, Blender, the AC SDK or the game.

## Code layout (planned)

```
src/mapcreator/
  core/        geometry, profile, mesh, kn5, ai_line, package, validate (data-agnostic)
  providers/   Centreline, Elevation, Canopy, Vegetation, Buildings, Imagery, Textures interfaces
               osm/, tas_list/, copernicus/, ...  (one package per source; each declares licence + attribution)
  regions/     bounding box → providers registry; assembles ATTRIBUTION.txt
  cli/         `mapcreator build job.json` and per-stage commands
  worker/      queue client for the hosted service
  web/         API + map front end
harness/       Windows-only in-game test harness for development
deploy/examples/  generic container definitions with placeholder settings
```

## Hosted service (planned shape)

```
browser (map, pick start/via/finish) ─▶ web + API + job queue + car router
                                          ▲ worker pulls jobs (outbound HTTPS only)
build worker ── runs the pipeline against local data mirrors ──▶ object storage ─▶ user download
```

- The worker pulls jobs and pushes results. Nothing calls into the worker.
- The worker runs in a hardened container: read-only root, no capabilities, non-root user, resource limits.
- The worker's only network path is an egress allowlist proxy that refuses private address ranges.
- Abuse controls on the web side: a challenge on submission, per-IP and global quotas, and de-duplication of identical routes.
