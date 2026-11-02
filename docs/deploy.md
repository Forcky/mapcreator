# Deploying mapcreator

The hosted service has four parts. Each can run on a different machine.

| Part | What | Needs |
|---|---|---|
| **Web app** (`mapcreator-web`) | Map page, job API, queue (SQLite), worker API | Small. Reachable from the internet, normally behind a reverse proxy or tunnel |
| **Build worker** (`mapcreator-worker`) | Pulls jobs, runs the pipeline, uploads zips | CPU and disk: about 2–6 CPU-minutes and 1–2 GB of RAM per 10 km of track. Makes outbound connections only |
| **Object storage** | Finished zips, served to players | Any S3-compatible bucket (e.g. Cloudflare R2) with a public download domain |
| **Router** (GraphHopper, car profile) | Route previews on the map page | About 0.5 GB of RAM to serve Australia. The import needs about 3 GB for about 1.5 h |

The worker never accepts connections. In the example setup it sits on an internal Docker network whose only way out is an allowlisting proxy that refuses private address ranges. See `deploy/examples/`.

None of this needs a GPU, Windows or Assetto Corsa. The game is only for the developer's own testing (`harness/`).

## 1. Prepare data on the worker host

Mirrors make builds fast. Without them the worker reads elevation over HTTPS at about 1.6 s per MB. The canopy model is only usable from a mirror.

| Data | Size | Get it |
|---|---|---|
| Tasmania 2 m DEM | ~95 GB | `python scripts/mirror.py tas-dem <dir>`, or `curl -L -C - -o <dir>/Tasmania_Statewide_2m_DEM_14-08-2021.tif https://data.source.coop/alexgleith/tasmania-dem-2m/Tasmania_Statewide_2m_DEM_14-08-2021.tif` |
| Tasmania canopy height (2 m) | 26 GB zip → ~10 GB TIFF | `python scripts/mirror.py tas-chm <dir>` (downloads, then extracts `CHM_2m/TAS_CHM_2m.tif`) |
| OpenStreetMap, Australia | ~0.9 GB `.osm.pbf` | `https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf` |
| OSM index | SQLite, built from the extract | `python -m mapcreator.providers.osm.index build australia-latest.osm.pbf australia.sqlite` (needs the `osm` extra: pyosmium) |

To build the index without installing Python on the host, use a throwaway container:

```sh
docker run --rm -v <repo>:/src:ro -v <osm dir>:/data python:3.12-slim sh -c \
  "cp -r /src /tmp/src && pip install -q '/tmp/src[osm]' && python -m mapcreator.providers.osm.index build /data/australia-latest.osm.pbf /data/australia.sqlite"
```

pip builds inside the source folder, so the container copies the read-only mount first.

Refresh the extract and rebuild the index now and then (e.g. monthly). The index records the extract's timestamp (`metadata()` in `providers/osm/index.py`). Node locations are held in memory by default. If the build host is short of RAM, add `--locations sparse_file_array,/data/nodes.bin` to keep them on disk.

## 2. Build the worker image and smoke-test it

```sh
docker build -f deploy/examples/Dockerfile.worker -t mapcreator-worker:local .
```

Run the CLI pipeline in that image against the mirrors. This needs no web app, no token and no object storage:

```sh
docker run --rm --entrypoint mapcreator \
  -e MAPCREATOR_WORK_DIR=/out \
  -e MAPCREATOR_TAS_DEM=/mirror/Tasmania_Statewide_2m_DEM_14-08-2021.tif \
  -e MAPCREATOR_TAS_CHM=/mirror/CHM_2m/TAS_CHM_2m.tif \
  -e MAPCREATOR_OSM_INDEX=/osm/australia.sqlite \
  -v <mirror dir>:/mirror:ro -v <osm dir>:/osm:ro -v <repo>/examples:/jobs:ro \
  -v <output dir>:/out -v mapcreator-cache:/cache \
  mapcreator-worker:local build /jobs/grasstree_hill.json
```

It should finish with `[validate] ... "ok": true`. The installable zip is `<output dir>/grasstree_hill/05_track/grasstree_hill.zip`. The scenery stage should report `"tree_source": "canopy model"` when the canopy mirror is mounted.

Bind mounts on Docker Desktop for Windows are much slower than a Docker volume. If the profile or scenery stage is slow, copy the mirrors into a named volume and mount that instead.

## 3. Production

Each step below is a decision or an account for the operator to set up. They aren't automated:

1. **Object storage.**
   - A bucket with a public custom domain for downloads.
   - Two scoped keys: write for the worker, delete for the web app's retention.
   - Settings: `MAPCREATOR_STORAGE=s3`, `MAPCREATOR_S3_*`, `MAPCREATOR_DOWNLOAD_BASE`.
2. **Secrets.**
   - Worker token: `MAPCREATOR_WORKER_TOKEN`, the same on both sides, at least 32 characters.
   - IP salt: `MAPCREATOR_IP_SALT`.
   - Turnstile keys: `MAPCREATOR_TURNSTILE_SECRET` and `MAPCREATOR_TURNSTILE_SITE_KEY`.
   - Keep these in a secrets manager and inject them at deploy time. Never commit them.
3. **Router.**
   - Import a GraphHopper car graph from the same extract and serve it to the web app only.
   - Set `MAPCREATOR_ROUTER=graphhopper` and `MAPCREATOR_GRAPHHOPPER_URL`.
4. **Web app.**
   - Deploy `deploy/examples/compose.web.yaml`, behind TLS on a public hostname (`MAPCREATOR_PUBLIC_URL`).
   - If a proxy sets the client address, configure `MAPCREATOR_TRUSTED_PROXIES` and `MAPCREATOR_CLIENT_IP_HEADER`.
5. **Worker.**
   - Deploy `deploy/examples/compose.worker.yaml` with `MAPCREATOR_API_BASE` set to the web app's public URL.
   - Mount the mirrors and index read-only.
   - Add the API and storage hosts to `egress-allowlist.txt`.
   - Check isolation: from inside the worker container, a private address must be unreachable, and an allowed host must answer through the proxy.
6. **Retention.** Defaults: jobs and downloads 30 days (`MAPCREATOR_KEEP_DAYS`), IP hashes 2 days (`MAPCREATOR_IP_HASH_DAYS`). Add a bucket lifecycle rule as a backstop.

Every setting is a `MAPCREATOR_*` environment variable:
- the service's in `src/mapcreator/service/settings.py`
- the pipeline's in `src/mapcreator/config.py` and `config.local.toml.example`
