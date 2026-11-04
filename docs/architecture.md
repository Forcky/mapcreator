# Architecture

## Pipeline

Every stage reads files and writes files, so each one can be tested and rerun on its own. The CLI and the hosted service use the same `job.json`.

```
job.json (waypoints, width option, layouts)
   │
route     01_route     snapped OSM centreline and its ways' tags (roads from the local OSM index)
profile   02_profile   DEM sampling, smoothing with a grade-change limit and grade cap, bridge dips filled
section   03_section   width (OSM width → lanes × lane width → class default; +15% / +30%), camber from the DEM, crown
mesh      04_mesh      road chunks (≤ 65,535 vertices each), verges, adaptive terrain corridor, guard rails, AC dummies
scenery   04b_scenery  trees (canopy model, else ESA WorldCover masked by Meta/WRI canopy), trunk colliders, buildings from footprints
track     05_track     numpy KN5 writer, DDS textures, fast_lane.ai per layout, ini files, ui, ATTRIBUTION.txt,
                       centreline.geojson (ODbL), the zip
validate  06_validate  static checks (report.json); the in-game harness is for development only
```

The stage names are `mapcreator.pipeline.STAGE_NAMES`; `mapcreator <stage> job.json` reruns one, `mapcreator build job.json` runs them all. No stage needs a GPU, Blender, the AC SDK or the game. `data-flows.md` lists each stage's inputs and outputs.

## OSM data

Production reads OpenStreetMap from a local index, not from Overpass (public Overpass is for light development use only):

```
Geofabrik extract (.osm.pbf) ──build once (pyosmium)──▶ index (SQLite) ──query by box──▶ route stage, scenery stage, DevRouter
```

- **`providers/osm/index.py`** makes one pass over the extract and keeps two kinds of way, with node coordinates resolved: drivable roads (the classes in `overpass.DRIVABLE`, with their node ids, because routing needs shared nodes to connect ways) and closed `building=*` ways. Each kind has an R*Tree on its lon/lat boxes.
- **Roads:** `index.query(path, bbox)` returns the same JSON as Overpass (`{"elements": [nodes..., ways...]}`), so `graph.build` doesn't know which source it was given. `providers/osm/roads.py` picks the source: the index when `osm_index` is set, Overpass otherwise. Overpass answers are cached on disk; index answers aren't.
- **Buildings:** `index.buildings(path, bbox, epsg)` returns footprints in the region's CRS. The national fallback regions use them when `osm_index` is set; Tasmania uses theLIST's footprints.
- The index is opened read-only and immutable, so it can sit on a read-only mount. A rebuild writes a new file and moves it into place when complete; nothing updates the index in place.
- Building the index needs pyosmium (the `osm` extra); querying needs only Python's `sqlite3` with the R*Tree module.

## Code layout

```
src/mapcreator/
  core/        centreline, profile, section, mesh, scene, scenery, sprites, textures, roadside, kn5, ai_line,
               layout, package, validate (data-agnostic: arrays and geometry in, meshes and files out)
  providers/   one package per source; each declares its licence and attribution
               base, raster (windowed reads and tile cache)
               tas_list/: 2 m DEM, canopy height model, building footprints (Tasmania)
               ga/: DEM-S (national elevation); esa/: WorldCover (national tree cover); meta/: canopy extent mask
               osm/: graph (routing), index (local SQLite index), roads (index or Overpass), overpass (development)
  regions/     bounding box → providers and CRS; Tasmania, then the national fallback by MGA zone
  pipeline/    one module per stage, wiring providers to core; STAGE_NAMES
  cli/         `mapcreator build job.json` and per-stage commands
  service/     the hosted service: web API + job queue (api, db), route previews (routing), settings,
               Turnstile (turnstile), build worker (worker), download storage (storage), map page (static/)
harness/       Windows-only in-game test harness for development
deploy/examples/  generic container definitions with placeholder settings
deploy/graphhopper/  the car router image and its routing profile
scripts/       mirror.py: resumable downloads of the large Tasmanian rasters
```

## Hosted service

```
browser (map, pick start/via/finish) ─▶ web + API + job queue + car router
                                          ▲ worker pulls jobs (outbound HTTPS only)
build worker ── runs the pipeline against local data mirrors (OSM index, DEM) ──▶ object storage ─▶ user download
```

`src/mapcreator/service/`, two entry points:

- **`mapcreator-web`** (`api.py`) serves the map page, the public job API and the worker API from one FastAPI app. The public submits a `Job` and nothing else. In order: a Turnstile check (when configured), a region check, de-duplication (the same road with the same options returns the existing job), then a global queue cap and a per-IP daily limit. Client addresses are stored only as a keyed hash.
- **The queue** (`db.py`) is one SQLite table in WAL mode. A claim is atomic and hands out a fresh `worker_id` that heartbeat, complete and fail must present. A job whose worker stops heartbeating is requeued, then failed after two attempts.
- **Route previews** (`routing.py`) come from a self-hosted GraphHopper, or for local development from the build's own OSM graph, over the local OSM index when `osm_index` is set and Overpass otherwise. The worker routes again during the build. To keep it on the same road, the page sends points sampled about every kilometre along the current preview as extra waypoints.
- **Track ids** are `mc_<slug>_<first 8 of the route hash>`. Two people's tracks with the same name can't overwrite each other in a player's AC install, and rebuilding the same road keeps its id, so Content Manager treats the new zip as an update.
- **Retention:** every hour, piggy-backed on traffic, the web app deletes finished jobs and their downloads after `MAPCREATOR_KEEP_DAYS` (30) and blanks client-IP hashes after `MAPCREATOR_IP_HASH_DAYS` (2).
- **`mapcreator-worker`** (`worker.py`) polls the API, validates the job again, runs every stage from `cli.main.STAGES` in a fresh temporary folder, uploads the zip through `storage.py` and reports back. Only errors about the road reach the public; anything else is a generic line.
- **Downloads** are zip attachments with a quoted `<track_id>.zip` filename and support Range requests, so Content Manager can install them from an `acmanager://install?url=…` link on the job page. In production they live in an S3-compatible bucket (Cloudflare R2) behind a public custom domain. The worker uploads with a bucket-scoped key (`MAPCREATOR_STORAGE=s3`); the web app holds no key (`public`) and only builds links, and the bucket's lifecycle rule expires old tracks. They're uploaded with the same headers, so the bucket serves exactly what Content Manager needs; locally the web app serves a folder.

Security:

- The worker pulls jobs and pushes results. Nothing calls into the worker.
- The worker API needs a bearer token, checked before routing; without a token configured it doesn't exist (404).
- The map page runs under a content security policy with no inline script or style, loads MapLibre from the app itself, and names the map tile hosts it may fetch from.
- The worker runs in a hardened container: read-only root, no capabilities, non-root user, resource limits.
- The worker's only network path is an egress allowlist proxy that refuses private address ranges. See `deploy/examples/`.
