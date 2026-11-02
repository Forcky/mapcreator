# Example deployment

Generic container definitions for the hosted service. Every host name under `example.org`, the `192.0.2.x` address and the `.env` files are placeholders. These files haven't been run as written; treat them as a starting point to review, not a finished deployment. Real deployment details belong in a private repository.

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
| `Dockerfile.web` | `python:3.12-slim`, non-root uid 10001, `pip install ".[service]"` |
| `Dockerfile.worker` | `python:3.12-slim`, non-root uid 10001, `pip install .` (the worker needs no web packages) |
| `compose.web.yaml` | The web app, published on loopback for the reverse proxy; GraphHopper on an internal network |
| `compose.worker.yaml` | The worker and its Squid egress proxy |
| `squid.conf` | HTTPS-only `CONNECT` to allowlisted hosts; private destinations refused; nothing cached or written to disk |
| `egress-allowlist.txt` | The hosts the pipeline reads from, plus placeholders for your API and object storage |

Build from the repository root, for example `docker build -f deploy/examples/Dockerfile.web .`. The compose files set that up themselves. A `.dockerignore` at the root keeps local configuration, `.env` files and data out of the build context.

## Hardening in the examples

Both services run with a read-only root filesystem, `cap_drop: [ALL]`, `no-new-privileges`, a non-root user, and memory, CPU and process limits. Scratch space is a size-limited `tmpfs`. The worker validates every job again before running it, works in a fresh temporary folder per job, and only reports plain-language errors; tracebacks stay in its own log.

## Secrets and settings

All settings are `MAPCREATOR_*` environment variables (see `src/mapcreator/service/settings.py`). Keep the secrets in `web.env` and `worker.env`, which are gitignored:

- `MAPCREATOR_WORKER_TOKEN`: the same long random value on both sides. Without it the web app's worker API answers 404.
- `MAPCREATOR_IP_SALT`: client addresses are only stored as a keyed hash; set this so per-IP limits survive restarts.
- `MAPCREATOR_TURNSTILE_SECRET` and `MAPCREATOR_TURNSTILE_SITE_KEY`: Cloudflare Turnstile on submission. Without them submissions aren't challenged.

Set `MAPCREATOR_TRUSTED_PROXIES` to your reverse proxy's address and `MAPCREATOR_CLIENT_IP_HEADER` to the header it overwrites with the client's address. The web app ignores that header from anyone else. Your reverse proxy should also cap request bodies and send `Strict-Transport-Security`.

## Still to do

- **Object storage.** The worker can only write to a local folder today (`LocalStorage`), so the web app and a worker must share a host until an S3-compatible `Storage` exists. `storage.py` describes what it needs.
- **GraphHopper.** Bring your own image or jar with a `car` profile built from the Geofabrik Australia extract. The example only reserves its place.
- **Local OSM extract.** The pipeline still reads OSM from Overpass, which is for light development use. Once it reads a local Geofabrik extract, remove `overpass-api.de` from the allowlist.
- **DNS.** Containers on an internal Docker network can still resolve names through Docker's resolver. If that matters to you, run the worker host with a resolver that only answers for allowlisted names.
