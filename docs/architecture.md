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
4b. scenery  → 04b_scenery/scene      trees (canopy model, else ESA WorldCover cover) + buildings from footprints
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
  service/     the hosted service: web API + job queue (api, db), route previews (routing),
               build worker (worker), download storage (storage), map page (static/)
harness/       Windows-only in-game test harness for development
deploy/examples/  generic container definitions with placeholder settings
```

## Hosted service

```
browser (map, pick start/via/finish) ─▶ web + API + job queue + car router
                                          ▲ worker pulls jobs (outbound HTTPS only)
build worker ── runs the pipeline against local data mirrors ──▶ object storage ─▶ user download
```

`src/mapcreator/service/`, two entry points:

- **`mapcreator-web`** (`api.py`) serves the map page, the public job API and the worker API from one FastAPI app. The public submits a `Job` and nothing else. In order: a Turnstile check (when configured), a region check, de-duplication (the same road with the same options returns the existing job), then a global queue cap and a per-IP daily limit. Client addresses are stored only as a keyed hash.
- **The queue** (`db.py`) is one SQLite table in WAL mode. A claim is atomic and hands out a fresh `worker_id` that heartbeat, complete and fail must present. A job whose worker stops heartbeating is requeued, then failed after two attempts.
- **Route previews** (`routing.py`) come from a self-hosted GraphHopper, or for local development from the build's own OSM graph over Overpass. They are only a guide: the worker routes again during the build.
- **`mapcreator-worker`** (`worker.py`) polls the API, validates the job again, runs every stage from `cli.main.STAGES` in a fresh temporary folder, uploads the zip through `storage.py` and reports back. Only errors about the road reach the public; anything else is a generic line.
- **Downloads** are zip attachments with a quoted `<track_id>.zip` filename and support Range requests, so Content Manager can install them from an `acmanager://install?url=…` link on the job page.

Security:

- The worker pulls jobs and pushes results. Nothing calls into the worker.
- The worker API needs a bearer token, checked before routing; without a token configured it doesn't exist (404).
- The map page runs under a content security policy with no inline script or style, loads MapLibre from the app itself, and names the map tile hosts it may fetch from.
- The worker runs in a hardened container: read-only root, no capabilities, non-root user, resource limits.
- The worker's only network path is an egress allowlist proxy that refuses private address ranges. See `deploy/examples/`.
