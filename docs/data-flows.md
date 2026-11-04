# Data flows

What moves between the parts of mapcreator, who holds which credential, what is stored and for how long, and where the trust boundaries are. `architecture.md` describes the parts; `deploy.md` says how to set them up.

## The parts

```
                      ┌──────────────── public internet ─────────────────┐
 browser ─────────────┤ map tiles (OpenFreeMap, theLIST topographic)      │
   │  │               │ Turnstile widget (Cloudflare)                     │
   │  └── download ──▶│ object storage, public custom domain (zip files)  │◀── PUT zip ──┐
   │                  └───────────────────────────────────────────────────┘              │
   ▼ HTTPS (reverse proxy / CDN)                                                         │
 web app (mapcreator-web) ── internal network ──▶ car router (GraphHopper)               │
   │  ▲  └─ HTTPS ──▶ Turnstile siteverify                                               │
   │  │                                                                                  │
   │  └──── /api/worker/* (bearer token, outbound from the worker) ────┐                 │
   ▼                                                                   │                 │
 job database (SQLite)                                build worker (mapcreator-worker) ──┘
                                                         │ via egress allowlist proxy only
                                                         ├─▶ elevation and land-cover hosts (when not mirrored)
                                                         └─ reads local mirrors: OSM index, DEM, canopy (read-only)
```

Two machines, in production:

- **The web host** runs the web app and the router. It is the only part the public reaches.
- **The worker host** runs the build worker and holds the data mirrors. Nothing connects to it: the worker only makes outbound HTTPS requests, through an egress proxy that allows a short list of hosts and refuses private address ranges.

## Network flows

| # | From → to | What | Credential | Started by |
|---|---|---|---|---|
| 1 | browser → web app | the map page, `/api/config`, `/api/status`, route previews, job submission, job status polls | none; a Turnstile token on submission | browser |
| 2 | browser → OpenFreeMap | basemap style, vector tiles, glyphs, sprites | none | browser |
| 3 | browser → theLIST | topographic raster tiles, only when the user ticks the layer | none | browser |
| 4 | browser → Cloudflare | the Turnstile widget | the public site key | browser |
| 5 | web app → Cloudflare siteverify | checks the Turnstile token of each submission | Turnstile secret | web app |
| 6 | web app → router | route previews (`POST /route`) on a private network with no other route out | none (network isolation) | web app |
| 7 | worker → web app `/api/worker/*` | claim a job, heartbeat with the current stage, report done or failed | worker bearer token, plus the per-claim `worker_id` | worker |
| 8 | worker → object storage | upload the finished zip | storage key scoped to one bucket | worker |
| 9 | worker → data hosts | DEM tiles outside Tasmania (GA DEM-S on DEA), ESA WorldCover tiles and Meta/WRI canopy tiles where there's no canopy model, theLIST building footprints and TasNetworks transmission lines and towers, the Tasmanian DEM if it isn't mirrored | none | worker |
| 10 | browser or Content Manager → object storage | download the zip (`acmanager://install?url=…` or a plain link) | none; the URL is public | user |

The web app never connects to the worker, and holds no storage key. Its own outbound traffic is flow 5 only (and flow 6 inside its private network). The worker's job never contains a URL or a path, so flows 8 and 9 go only where the worker's own configuration says.

Operator flows, outside any job: downloading the OSM extract and building the index (`Dockerfile.index`), mirroring the DEM and canopy (`scripts/mirror.py`), importing the router graph from the same extract.

## A job, end to end

1. **Pick a road.** The page sends the points to `POST /api/route-preview` (limit: `MAPCREATOR_PREVIEWS_PER_IP_PER_MINUTE`, default 30). The router returns the route and its length; the page refuses to submit routes over the job's length limit (64 km: Targa Tasmania's longest stage, Mt Arrowsmith at 53 km, plus a fifth).
2. **Submit.** `POST /api/jobs` with the job (waypoints, width option, layouts, track name) and a Turnstile token. The web app, in order:
   - verifies the Turnstile token (when configured);
   - checks every point is in one supported region;
   - looks for the same road with the same options (the route hash): a job in progress, or one finished by the same version of the track-making code (`buildinfo.version()`, recorded in each build's summary), is handed back instead of building it again. A track built by older code is built again as a new job, so fixes reach the next person who asks;
   - refuses when the queue holds `MAPCREATOR_MAX_QUEUED_JOBS` (50) or this client has submitted `MAPCREATOR_JOBS_PER_IP_PER_DAY` (10) today;
   - names the track `mc_<slug>_<first 8 of the route hash>` and stores the job as `queued`. The hash covers the road and its options, not the length limit, so raising the limit kept every track's id.
   The page then polls `GET /api/jobs/<id>` every 5 seconds. The id is in the page's address (`#job=<id>`), so the link can be bookmarked.
3. **Claim.** The worker asks `POST /api/worker/claim` every 15 seconds when idle. The oldest queued job becomes `running`, with a fresh random `worker_id` that every later call must present. Any worker request also records when a worker was last heard from; the page says the builder is offline after 2 minutes of silence.
4. **Build.** The worker validates the job again, then runs the stages in a new temporary folder. It heartbeats every 30 seconds and at the start of each stage, so the page can say "step 4 of 7". A job whose heartbeats stop for `MAPCREATOR_STALE_MINUTES` goes back in the queue, and is failed after two attempts.
5. **Upload and report.** The zip goes to object storage under `<job id>/<track id>.zip`, with the `Content-Type` and quoted `Content-Disposition` Content Manager expects. Then `complete` records the key and a small summary. On failure, only messages about the road itself (no road near a point, no connection, a bend too tight to drive) reach the page; anything else is a generic line, and the traceback stays in the worker's log.
   - A too-tight bend also goes back as `code: tight_bend` with its radius, its distance along the road and its place. The job page marks it on the map.
   - Down to a 2.5 m radius, the job status also carries `can_override` and a `retry_job`: the same road with `allow_tight_bends`. **Generate anyway** submits it as a new job, with its own track id, a fresh Turnstile token and the usual limits.
   - A track built that way says so in its description and on the job page.
6. **Download.** The job page shows a Content Manager install link and a plain download link, both to the public storage URL with the build time as `?v=` (storage ignores it). A track rebuilt in place gets a new link, so a CDN's cached copy of the old zip is never handed out for it.
7. **Expire.** Hourly, on the next request, the web app deletes jobs finished more than `MAPCREATOR_KEEP_DAYS` (30) ago and blanks client-address hashes older than `MAPCREATOR_IP_HASH_DAYS` (2). The bucket's lifecycle rule deletes the zips a few days later. (With `MAPCREATOR_STORAGE=s3` on the web side, the web app deletes them itself.)

## Inside a build

Each stage reads the folders before it and writes its own, so any stage can be rerun alone (`mapcreator <stage> job.json`).

| Stage | Folder | Reads | Writes |
|---|---|---|---|
| route | `01_route` | job waypoints; OSM roads from the local index (Overpass in development) | `route.npz` (centreline in the region's metric CRS, and which run of OSM ways each segment is on), `runs.json` (each run's ways and tags), `route.geojson` |
| profile | `02_profile` | the route; the region's DEM; in Tasmania, theLIST's LiDAR index | `centreline.npz` (smoothed 3D centreline, grades, and which stations have fine elevation; corners at junctions rounded to 8 m first), profile plot when matplotlib is installed |
| section | `03_section` | the centreline; OSM surface, width, lanes, `narrow` and `lane_markings`; the DEM across the road | `section.npz` (half-widths, camber, crown, and each station's surface and look: sealed or unsealed, one lane or two) |
| mesh | `04_mesh` | centreline, section, DEM; WorldCover's permanent water | road, verge, terrain and wall meshes, water surfaces (`water.json`: bodies, levels, LAKE or SEA), AC dummies; roadside furniture (guide posts, CAMs, speed signs) and its texture atlas |
| scenery | `04b_scenery` | centreline, section; canopy model, or WorldCover masked by Meta/WRI canopy; building footprints; power lines (TasNetworks in Tasmania, the OSM index elsewhere) | extruded buildings; towers or poles and wires (`power.geojson` for OSM lines); 3D tree models and every tree's placement for Tree FX (`treefx/`); invisible colliders for the trunks near the road |
| track | `05_track` | everything above | the KN5 model, `fast_lane.ai` per layout, ini files, `ui_track.json`, preview images, `ATTRIBUTION.txt`, `centreline.geojson` (ODbL), `power.geojson` (ODbL, when OSM power lines were used), `extension/` (CSP config, `trees/` with the Tree FX list and models), the zip |
| validate | `06_validate` | the track folder | `report.json`: static checks. A build failure fails the job with the generic message. A bend under 5 m radius is a route failure: the page is told where it is and may offer to build it anyway |

The region (`regions/`) decides the CRS and the providers. Tasmania uses theLIST's 2 m DEM, canopy model, building footprints and TasNetworks' transmission lines in MGA zone 55. Elsewhere in Australia the national fallback uses GA's DEM-S, WorldCover trees, and OSM buildings and power lines, in the MGA zone of the route's mean longitude. The work CRSs are GDA94 MGA zones, to match the elevation and building data; longitudes and latitudes from OSM and the map page (WGS84, within decimetres of GDA2020) go through the GDA2020-to-GDA94 transformation (`providers.base.LONLAT_CRS`), not a null shift, which would leave the road about 1.5 m off the DEM.

## What is stored, where, for how long

| Where | What | How long |
|---|---|---|
| web app, job database | each job: its JSON (coordinates, options, track name), status, timestamps, attempts, current stage, result key and summary or error, and a keyed hash of the client's address (never the address) | 30 days after the job finishes; the address hash is blanked after 2 days |
| web app, memory | per-address preview counts | one minute |
| web app, logs | job ids and state changes. No access log by default (`--access-log` turns one on) | the container's log rotation |
| object storage | `<job id>/<track id>.zip` | until the bucket's lifecycle rule (or the web app's retention) deletes it |
| worker, temporary folder | the work folders of the job being built | deleted when the job ends, success or not |
| worker, cache (`MAPCREATOR_CACHE_DIR`) | downloaded DEM and land-cover tiles, compressed textures, Overpass answers in development | until cleared; holds open data only |
| worker, logs | stage timings and tracebacks | the container's log rotation |
| worker host, mirrors | OSM index, DEM, canopy model | replaced by the operator (refresh monthly) |

No account, email address or name is collected. The track name the user types becomes part of the track id and the AC folder name.

## Trust boundaries

- **Browser → web app.** Everything from the browser is validated with pydantic: coordinates, enums, booleans and a short track name, with body size limits. No URL, path or free text reaches a shell or a file parser. Client addresses come from the socket, or from one configured header when the socket is a configured trusted proxy.
- **Web app → worker.** A job handed to the worker is validated again as a `Job`, so a compromised web app can only ask for a stretch of road. The worker can't be reached at all; it reaches the web app.
- **Worker → web app.** The worker API needs the bearer token, and each call about a job needs that claim's `worker_id`, so a worker that lost its job can't overwrite the new attempt's result. Results are size-limited and storage keys are checked against a strict pattern.
- **If the web host is compromised,** the attacker gets the worker token, the IP salt and the Turnstile secret. They can feed workers jobs (only roads, as above) and read or fake job results. They can't write to object storage, because the web app holds no storage key, so they can't replace a download.
- **If the worker is compromised,** the attacker gets the worker token and the bucket key: they can replace downloads. The container runs read-only, without capabilities, as a non-root user, and its only way out is the egress allowlist, which refuses private ranges, so it can't reach the rest of the worker host's network.

## Configuration

Service settings (`service/settings.py`), all from the environment:

| Variable | Web | Worker | Notes |
|---|---|---|---|
| `MAPCREATOR_WORKER_TOKEN` | yes | yes | the same on both sides, 32+ characters; without it the worker API doesn't exist |
| `MAPCREATOR_IP_SALT` | yes | | keys the client-address hash |
| `MAPCREATOR_TURNSTILE_SITE_KEY`, `MAPCREATOR_TURNSTILE_SECRET` | yes | | both or neither |
| `MAPCREATOR_DB` | yes | | SQLite job database |
| `MAPCREATOR_PUBLIC_URL` | yes | | makes download links absolute when downloads are served by the web app |
| `MAPCREATOR_ROUTER`, `MAPCREATOR_GRAPHHOPPER_URL` | yes | | `graphhopper` in production; the dev router reads the OSM index or Overpass |
| `MAPCREATOR_TRUSTED_PROXIES`, `MAPCREATOR_CLIENT_IP_HEADER` | yes | | the proxy's address as the app sees it, and the header it overwrites |
| `MAPCREATOR_JOBS_PER_IP_PER_DAY`, `MAPCREATOR_MAX_QUEUED_JOBS`, `MAPCREATOR_PREVIEWS_PER_IP_PER_MINUTE` | yes | | 10, 50, 30 |
| `MAPCREATOR_STALE_MINUTES`, `MAPCREATOR_KEEP_DAYS`, `MAPCREATOR_IP_HASH_DAYS` | yes | | requeue silent jobs; retention |
| `MAPCREATOR_STORAGE` | `public` (or `local`, `s3`) | `s3` (or `local`) | where zips go, and whether this side holds a key |
| `MAPCREATOR_DOWNLOAD_BASE` | yes | yes | the public URL (or, for `local`, path) downloads are served from |
| `MAPCREATOR_S3_ENDPOINT`, `MAPCREATOR_S3_BUCKET`, `MAPCREATOR_S3_REGION`, `MAPCREATOR_S3_ACCESS_KEY_ID`, `MAPCREATOR_S3_SECRET_ACCESS_KEY` | only with `s3` | yes | the key scoped to the one bucket |
| `MAPCREATOR_STORAGE_DIR` | only with `local` | only with `local` | |
| `MAPCREATOR_API_BASE` | | yes | the web app's base URL (or `--api`) |

Pipeline settings (`config.py`), from the environment or `config.local.toml`:

| Variable | Notes |
|---|---|
| `MAPCREATOR_OSM_INDEX` | the local OSM index; without it, roads come from Overpass (development only) and the national fallback has no buildings |
| `MAPCREATOR_TAS_DEM` | the Tasmanian 2 m DEM: a local mirror path, or the cloud-optimised GeoTIFF's URL |
| `MAPCREATOR_TAS_CHM` | the Tasmanian canopy height model; without it, trees come from WorldCover |
| `MAPCREATOR_META_CANOPY` | `0` turns off the Meta/WRI canopy mask on WorldCover trees (on by default) |
| `MAPCREATOR_CACHE_DIR`, `MAPCREATOR_WORK_DIR` | tile cache, and where the CLI writes work folders (the worker uses a temporary folder per job) |
| `MAPCREATOR_OVERPASS_URL`, `MAPCREATOR_USER_AGENT` | development only |
