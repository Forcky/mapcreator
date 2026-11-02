# Example deployment

Generic container definitions for the hosted service. Every host name under `example.org`, the `192.0.2.x` addresses and subnet, and the `.env` files are placeholders. A deployment adapted from these files runs in production, but these copies haven't been run exactly as written; treat them as a starting point to review. Real deployment details belong in a private repository. `docs/deploy.md` walks through the whole setup and `docs/data-flows.md` says what moves where.

## Shape

```
browser ── HTTPS ──▶ reverse proxy ──▶ web (mapcreator-web)          DMZ
                                        │  job queue (SQLite), route previews (GraphHopper)
                                        ▲
                                        │ HTTPS, outbound from the worker only
build worker (mapcreator-worker) ── egress proxy (allowlist) ──▶ API, data sources, object storage
```

- **web** serves the map page and the job API from a DMZ, behind your TLS-terminating reverse proxy. It accepts only a `Job` (coordinates, enums, booleans) from the public, and it never connects to a worker.
- **worker** pulls jobs: it polls the web API over HTTPS, builds the track, uploads it and reports back. It accepts no inbound connections at all, so it can run anywhere with outbound HTTPS, including a machine with no published ports.
- **egress proxy** is the worker's only way out. The worker sits on an `internal: true` network shared only with the proxy. The proxy forwards HTTPS to the hosts in `egress-allowlist.txt` and refuses private, loopback and link-local destinations (10/8, 172.16/12, 192.168/16, 169.254/16 and their IPv6 equivalents), even when an allowed name resolves to one.

## Files

| File | What it is |
|---|---|
| `Dockerfile.web` | `python:3.12-slim` pinned by digest, `libexpat1`, non-root uid 10001, `pip install ".[service,s3]"`, imports checked at build time |
| `Dockerfile.worker` | the same base, `pip install ".[s3]"` (the worker needs no web packages), GDAL network timeouts |
| `Dockerfile.index` | a one-shot image that builds the local OSM index from a `.osm.pbf` extract (needs pyosmium, which the other images leave out) |
| `compose.web.yaml` | The web app, published on loopback for the reverse proxy; the GraphHopper router (`deploy/graphhopper`) on an internal network |
| `compose.worker.yaml` | The worker and its Squid egress proxy |
| `squid.conf` | HTTPS-only `CONNECT` to allowlisted hosts; private destinations refused; nothing cached or written to disk |
| `egress-allowlist.txt` | The hosts the pipeline reads from, plus placeholders for your API and object storage |

Build from the repository root, for example `docker build -f deploy/examples/Dockerfile.web .`. The compose files set that up themselves. A `.dockerignore` at the root keeps local configuration, `.env` files and data out of the build context.

## Hardening in the examples

Both services run with a read-only root filesystem, `cap_drop: [ALL]`, `no-new-privileges`, a non-root user, and memory, CPU and process limits. Scratch space is a size-limited `tmpfs`. The worker validates every job again before running it, works in a fresh temporary folder per job, and only reports plain-language errors; tracebacks stay in its own log.

## Secrets and settings

All settings are `MAPCREATOR_*` environment variables (`docs/data-flows.md` lists them, per side). Keep the secrets in `web.env` and `worker.env`, which are gitignored:

- `web.env`:
  - `MAPCREATOR_WORKER_TOKEN`: the same long random value on both sides. Without it the web app's worker API answers 404.
  - `MAPCREATOR_IP_SALT`: client addresses are only stored as a keyed hash; set this so per-IP limits survive restarts.
  - `MAPCREATOR_TURNSTILE_SECRET` and `MAPCREATOR_TURNSTILE_SITE_KEY`: Cloudflare Turnstile on submission. Without them submissions aren't challenged.
- `worker.env`:
  - `MAPCREATOR_WORKER_TOKEN`: the same value as the web app's.
  - `MAPCREATOR_S3_ACCESS_KEY_ID` and `MAPCREATOR_S3_SECRET_ACCESS_KEY`: a key for the download bucket only. Without them every job builds and then fails at upload.

The web app holds no storage key (`MAPCREATOR_STORAGE=public`): it only builds links, and the bucket's lifecycle rule expires old tracks. A key that could delete them could also overwrite live downloads.

Write `.env` files without trailing spaces or carriage returns. Values are trimmed when read, but check each length against your secrets manager.

**Client addresses.** Set `MAPCREATOR_TRUSTED_PROXIES` to your reverse proxy's address as the container sees it, and `MAPCREATOR_CLIENT_IP_HEADER` to the header the proxy overwrites with the client's address. The web app ignores that header from anyone else. With the port published on loopback and the proxy on the same host, the container sees the `edge` network's gateway, which is why the example pins that network's subnet. Get this wrong and every visitor counts as one client, so the per-client limits become one shared limit; the web app logs a warning naming the address the first time it sees the header from an untrusted private address. Your reverse proxy should also cap request bodies and send `Strict-Transport-Security`.

**Bind-mount ownership.** The containers run as uid 10001. On a Linux host, give folders they write to that owner first, for example `sudo install -d -o 10001 -g 10001 router-data` before importing the router graph.

## OSM data

The worker reads OpenStreetMap from a local index, not from Overpass, which isn't on its egress allowlist. Build the index from Geofabrik's Australia extract on another machine (`docs/dev-handover.md` has the steps), copy `australia.sqlite` into `osm/` next to `compose.worker.yaml`, and the example mounts it read-only at `/osm` with `MAPCREATOR_OSM_INDEX` pointing at it. Building needs pyosmium, which the worker image leaves out; reading needs only SQLite. Without the file, every build fails with a generic error and the worker's log names the missing index. To refresh, copy the new index into `osm/` under a temporary name and rename it over the old one, so no build ever opens a half-copied file.

## Still to do

- **DNS.** Containers on an internal Docker network can still resolve names through Docker's resolver. If that matters to you, run the worker host with a resolver that only answers for allowlisted names.
- **Proving the proxy's private-range rule.** A request to a bare private address is refused before that rule is reached (it isn't on the allowlist), so it proves nothing about it. To test the rule itself, run a throwaway copy of the proxy whose allowlist also has a wildcard DNS name that resolves to a private address (such as `.nip.io`), and check that `https://10-0-0-1.nip.io/` (which resolves to a private address) is refused.
