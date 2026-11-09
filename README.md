# mapcreator

Generate drivable [Assetto Corsa](https://www.assettocorsa.it/) tracks from real roads, using only open data that may legally be redistributed.

**Status:** an early-stage personal project, maintained in spare time. The pipeline builds installable tracks, and they've been driven in game end to end, with AI opponents (Grasstree Hill both ways, the Esperance coast road and the Lyell Highway among them). Expect rough edges, and changes without notice.

**Requirements:** the tracks need Assetto Corsa on PC with [Custom Shaders Patch](https://acstuff.club/patch/) (CSP), which draws their trees and water. They aren't built to work without it (#28). Tested with CSP 0.2.11.

- **Coverage:** Tasmania has detailed data (2 m LiDAR elevation, canopy heights, vegetation types, building footprints and transmission lines). The rest of Australia uses national data: Geoscience Australia's 5 m LiDAR elevation where it covers the road (Victoria's 10 m model, then 30 m, where it doesn't), ESA WorldCover trees checked against Meta and WRI's canopy map, and OSM buildings and power lines when a local OSM index is set up.
- **Hosted service:** running. The map page, job queue and build worker are in `src/mapcreator/service/`; [docs/deploy.md](docs/deploy.md) says how to run your own.

Want to help? See [CONTRIBUTING.md](CONTRIBUTING.md). Developing on a machine with Assetto Corsa: start with [docs/dev-handover.md](docs/dev-handover.md).

## Quick start: build a track on the command line

You need Python 3.12. No GPU, game or account is needed to build.

```sh
python -m venv .venv
. .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install -e .
mapcreator build examples/grasstree_hill.json
```

This builds Grasstree Hill Road in Tasmania (12.8 km, both directions). The first build downloads the road from the public Overpass server and the elevation and scenery data around it; later builds reuse the downloads in `cache/`. The track is written to `out/grasstree_hill/05_track/`: a `grasstree_hill/` folder and `grasstree_hill.zip`. Install it by unzipping it into `assettocorsa/content/tracks`, or by dragging the zip onto Content Manager.

To build another road, copy a job from `examples/` and change its `track_id` and `waypoints` (longitude and latitude, in driving order). `mapcreator <stage> job.json` reruns one stage (`route`, `profile`, `section`, `mesh`, `scenery`, `track`, `validate`). For more than light use, read OSM from a local index rather than public Overpass (below).

## How it works

1. Pick a stretch of road.
2. The pipeline snaps it to OpenStreetMap and drapes it over a LiDAR-derived terrain model.
3. It generates the road, verges, terrain, scenery, timing gates and an AI line.
4. It writes a ready-to-install track: a KN5 model plus the `data/`, `ai/` and `ui/` folders.

Each stage writes its output to disk and can be rerun on its own. See [docs/architecture.md](docs/architecture.md) for the parts, [docs/data-flows.md](docs/data-flows.md) for what moves between them and what is kept, [docs/deploy.md](docs/deploy.md) for running the hosted service, and [docs/known-issues.md](docs/known-issues.md) for what's known not to work yet.

## Data principles

- Only sources whose licences allow building and redistributing derived tracks: OpenStreetMap (ODbL), Creative Commons BY government data, CC0 textures. See [docs/data-sources.md](docs/data-sources.md).
- No Google, Esri, Bing or Mapbox content. Their terms forbid this use whether or not anyone pays.
- Every track carries an `ATTRIBUTION.txt` and a `centreline.geojson` (ODbL) for the road data it was built from.

## Running the hosted service locally

The service is a web app (map page, job API and queue) and a build worker that pulls jobs from it. To try both on one machine:

```sh
python -m venv .venv
. .venv/bin/activate                               # Windows: .venv\Scripts\activate
pip install -e ".[service,dev]"
export MAPCREATOR_WORKER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

mapcreator-web                                      # http://127.0.0.1:8000
mapcreator-worker --api http://127.0.0.1:8000       # in a second terminal, with the same token
```

On PowerShell, set the token with `$env:MAPCREATOR_WORKER_TOKEN = "..."`. Locally, route previews use the public Overpass server (fine for light use, not for the public), finished tracks are served from `out/service/downloads`, and the worker reads its data settings from `config.local.toml` like the CLI. Every service setting is a `MAPCREATOR_*` environment variable; see `src/mapcreator/service/settings.py`. Container examples are in `deploy/examples/`.

**OSM data for anything beyond light local use:** build workers (and dev route previews) should read OpenStreetMap from a local index rather than public Overpass. Build it once from Geofabrik's Australia extract with `python -m mapcreator.providers.osm.index build australia-latest.osm.pbf australia.sqlite` (needs the `osm` extra), then set `MAPCREATOR_OSM_INDEX` (or `osm_index` in `config.local.toml`) to its path. It also gives the rest of Australia OSM building footprints. See `docs/dev-handover.md`.

## Not affiliated

This project is not affiliated with or endorsed by Kunos Simulazioni. Assetto Corsa is a trademark of its owner. Custom tracks work in the original Assetto Corsa on PC only.

## Licence

Apache License 2.0: see `LICENSE`. Third-party material keeps its own licence and notice (see `THIRD_PARTY_NOTICES.md`). Built tracks carry the data licences of their sources (see `docs/data-sources.md`).
