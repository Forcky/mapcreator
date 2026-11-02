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

To build the index without installing Python on the host, use the one-shot index image:

```sh
docker build -f deploy/examples/Dockerfile.index -t mapcreator-index:local .
docker run --rm -v <osm dir>:/data mapcreator-index:local \
  /data/australia-latest.osm.pbf /data/australia.sqlite
```

Don't use a bare `python:3.12-slim` container for this. pyosmium's wheel links the system's libexpat, which the slim image doesn't ship, so `import osmium` fails there. The example Dockerfiles install it.

Refresh the extract and rebuild the index now and then (e.g. monthly). The index records the extract's timestamp (`metadata()` in `providers/osm/index.py`). Node locations are held in memory by default. If the build host is short of RAM, add `--locations sparse_file_array,/data/nodes.bin` to keep them on disk.

## 2. Build the worker image and smoke-test it

```sh
docker build -f deploy/examples/Dockerfile.worker -t mapcreator-worker:local .
```

Run the CLI pipeline in that image against the mirrors. This needs no web app, no token and no object storage. The image runs as uid 10001, so on a Linux host make the output folder writable first (`sudo install -d -o 10001 -g 10001 <output dir>`):

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
   - A bucket with a public custom domain for downloads, and a lifecycle rule that deletes objects a few days after `MAPCREATOR_KEEP_DAYS`.
   - One key, for the worker, scoped to that bucket. Worker settings: `MAPCREATOR_STORAGE=s3`, `MAPCREATOR_S3_*`, `MAPCREATOR_DOWNLOAD_BASE`.
   - The web app holds no key: `MAPCREATOR_STORAGE=public` and `MAPCREATOR_DOWNLOAD_BASE`. Most stores can't issue a delete-only key, and a key that can delete can also overwrite downloads, which matters on the internet-facing host. (`s3` on the web side works too, if you'd rather it deleted expired tracks itself.)
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

## 4. Operating the queue

`mapcreator-admin` works on the web app's job database directly. Run it inside the web container, where the database and settings are:

```sh
docker exec <web container> mapcreator-admin stats                    # jobs per status, worker last heard from
docker exec <web container> mapcreator-admin list --status failed     # newest first
docker exec <web container> mapcreator-admin show <job id>            # one job in full, no client data
docker exec <web container> mapcreator-admin requeue <job id>         # failed or running back to queued, attempts reset
docker exec <web container> mapcreator-admin fail <job id> --reason "..."   # shown on the job page
docker exec <web container> mapcreator-admin delete <job id>          # the download goes with the bucket's lifecycle rule
```

`GET /api/status` says whether a worker has been heard from in the last two minutes; the map page warns visitors when it hasn't.
