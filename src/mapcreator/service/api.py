"""The public web API, the worker API and the map page, in one FastAPI app.

The public side accepts a `mapcreator.job.Job` (numbers, enums and booleans)
and nothing else, so nobody on the internet can hand the worker a URL, a path
or free text. Abuse controls sit in front of the queue: a Turnstile challenge,
a global queue cap, a per-IP daily limit, and de-duplication, so asking twice
for the same road returns the first job rather than building it again.

The worker side is guarded by a bearer token in ASGI middleware, ahead of
routing, so a caller without the token learns nothing about it, not even the
request shapes that FastAPI's validation errors would describe. With no token
configured the worker API answers 404 like any unknown path.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mapcreator import __version__
from mapcreator.job import MAX_WAYPOINTS, Job, Waypoint, public_track_id
from mapcreator.pipeline import STAGE_NAMES
from mapcreator.regions import REGIONS
from mapcreator.service.db import JobStore, route_hash
from mapcreator.service.routing import (DevRouter, GraphHopperRouter, OutOfArea, RoutePreview, Router,
                                        RouterUnavailable, RoutingError, check_region)
from mapcreator.service.settings import ServiceSettings
from mapcreator.service.storage import LocalStorage, Storage, check_key, download_headers, storage_for
from mapcreator.service.turnstile import MAX_TOKEN_LENGTH, CloudflareTurnstile, Verifier

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
JOB_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
WORKER_PREFIX = "/api/worker"
# A job is a few hundred bytes; anything near this is not a job.
MAX_BODY_BYTES = 64 * 1024
MAX_ERROR_CHARS = 500
MAX_SUMMARY_BYTES = 16 * 1024
# An idle worker asks for work every 15 s and a busy one heartbeats every 30 s, so a worker that hasn't
# been heard from in this long is offline (stopped, or its machine or network is down).
BUILDER_ONLINE_S = 120.0

# Hosts the map page loads from: the OpenFreeMap style, tiles, glyphs and sprites, and theLIST's
# topographic tiles. Listed by name so nothing else can be fetched, even by injected markup.
MAP_HOSTS = ("https://tiles.openfreemap.org", "https://services.thelist.tas.gov.au")
TURNSTILE_HOST = "https://challenges.cloudflare.com"


# ----- request bodies -----

class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job: Job
    turnstile_token: str | None = Field(default=None, max_length=MAX_TOKEN_LENGTH)


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    waypoints: list[Waypoint] = Field(min_length=2, max_length=MAX_WAYPOINTS)


class WorkerLease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(pattern=r"^[A-Za-z0-9_-]{16,64}$")


class HeartbeatRequest(WorkerLease):
    stage: Literal[STAGE_NAMES] | None = None  # the build stage now running


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    summary: dict[str, Any] = Field(default_factory=dict)

    @field_validator("key")
    @classmethod
    def _key(cls, key: str) -> str:
        return check_key(key)

    @field_validator("summary")
    @classmethod
    def _small(cls, summary: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(summary)) > MAX_SUMMARY_BYTES:
            raise ValueError("summary is too large")
        return summary


class CompleteRequest(WorkerLease):
    result: JobResult


class FailRequest(WorkerLease):
    error: str = Field(max_length=20_000)


# ----- helpers -----

def _iso(t: float | None) -> str | None:
    if t is None:
        return None
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clean_error(text: str) -> str:
    """One line of plain text, short enough to show on the page."""
    text = " ".join(text.split())
    return text if len(text) <= MAX_ERROR_CHARS else text[:MAX_ERROR_CHARS - 1] + "…"


def describe_validation_errors(errors: Sequence[dict]) -> str:
    """Pydantic's errors as one readable sentence, without echoing the submitted values back."""
    parts = []
    for err in errors[:3]:
        where = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        msg = str(err.get("msg", "invalid value"))
        parts.append(f"{where}: {msg}" if where else msg)
    more = f" (and {len(errors) - 3} more)" if len(errors) > 3 else ""
    return "The request wasn't valid. " + "; ".join(parts) + more + "."


class RateLimiter:
    """A sliding-window count per key, kept in memory.

    The web app is meant to run as one process. If it ever runs as several,
    each keeps its own count, which only loosens the limit.
    """

    MAX_KEYS = 10_000

    def __init__(self, limit: int, window_s: float, *, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        cutoff = now - self.window_s
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            if len(self._hits) > self.MAX_KEYS:
                self._hits = {k: q for k, q in self._hits.items() if q and q[-1] > cutoff}
            return True


class ClientIdentity:
    """Who is asking, as a keyed hash of their address. Raw addresses are never stored.

    The address is the socket peer, unless the peer is a configured trusted
    proxy, in which case the configured header names the client (its last
    entry, which is the one that proxy added). IPv6 clients are counted per
    /64, since one connection usually has a whole /64 to rotate through.
    """

    def __init__(self, settings: ServiceSettings):
        self._proxies = [ipaddress.ip_network(p, strict=False) for p in settings.trusted_proxies]
        self._header = settings.client_ip_header
        self._salt = settings.ip_salt.encode()

    def address(self, request: Request) -> str:
        peer = request.client.host if request.client else ""
        if self._header and self._is_trusted(peer):
            claimed = request.headers.get(self._header, "").split(",")[-1].strip()
            try:
                return str(ipaddress.ip_address(claimed))
            except ValueError:
                pass
        return peer

    def _is_trusted(self, peer: str) -> bool:
        try:
            addr = ipaddress.ip_address(peer)
        except ValueError:
            return False
        return any(addr in net for net in self._proxies)

    def key(self, request: Request) -> str:
        raw = self.address(request)
        try:
            addr = ipaddress.ip_address(raw)
        except ValueError:
            subject = raw
        else:
            if addr.version == 6 and addr.ipv4_mapped is not None:
                addr = addr.ipv4_mapped
            subject = str(ipaddress.ip_network(f"{addr}/64", strict=False)) if addr.version == 6 else str(addr)
        return hmac.new(self._salt, subject.encode(), hashlib.sha256).hexdigest()[:32]


# ----- middleware -----

def content_security_policy(*, turnstile: bool) -> str:
    """No inline script or style anywhere; MapLibre's workers start from blob: URLs."""
    challenge = f" {TURNSTILE_HOST}" if turnstile else ""
    frames = TURNSTILE_HOST if turnstile else "'none'"
    hosts = " ".join(MAP_HOSTS)
    return "; ".join([
        "default-src 'none'",
        f"script-src 'self'{challenge}",
        "style-src 'self'",
        f"img-src 'self' data: blob: {hosts}",
        f"connect-src 'self' {hosts}",
        "font-src 'self'",
        "worker-src 'self' blob:",
        "child-src blob:",
        f"frame-src {frames}",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "manifest-src 'self'",
    ])


class SecurityHeaders:
    """Adds the security headers to every response, including errors and static files."""

    def __init__(self, app: ASGIApp, *, csp: str):
        self.app = app
        self.headers = {
            "Content-Security-Policy": csp,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "geolocation=(), camera=(), microphone=(), payment=(), usb=()",
            "X-Frame-Options": "DENY",
            "Cross-Origin-Opener-Policy": "same-origin",
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_api = scope["path"].startswith("/api/")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in self.headers.items():
                    headers[name] = value
                if is_api and "cache-control" not in headers:
                    headers["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_with_headers)


class RequestGuard:
    """Refuses oversized bodies, and anything under /api/worker without the worker token."""

    def __init__(self, app: ASGIApp, *, worker_token: str | None):
        self.app = app
        self._token = worker_token.encode() if worker_token else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            refusal = self._body_refusal(headers) or self._worker_refusal(scope["path"], headers)
            if refusal is not None:
                await refusal(scope, receive, send)
                return
        await self.app(scope, receive, send)

    @staticmethod
    def _body_refusal(headers: Headers) -> Response | None:
        length = headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            return JSONResponse({"detail": "The request is too large."}, status_code=413)
        return None

    def _worker_refusal(self, path: str, headers: Headers) -> Response | None:
        if path != WORKER_PREFIX and not path.startswith(WORKER_PREFIX + "/"):
            return None
        if self._token is None:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        scheme, _, value = headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not value.strip():
            return JSONResponse({"detail": "Missing bearer token."}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
        if not hmac.compare_digest(value.strip().encode(), self._token):
            return JSONResponse({"detail": "Forbidden."}, status_code=403)
        return None


# ----- the app -----

def _default_router(settings: ServiceSettings) -> Router:
    if settings.router == "graphhopper":
        return GraphHopperRouter(settings.graphhopper_url)
    return DevRouter()


PRUNE_EVERY_S = 3600.0


def create_app(settings: ServiceSettings, *, router: Router | None = None, turnstile: Verifier | None = None,
               storage: Storage | None = None) -> FastAPI:
    store = JobStore(settings.db_path)
    router = router or _default_router(settings)
    if turnstile is None and settings.turnstile_secret:
        turnstile = CloudflareTurnstile(settings.turnstile_secret)
    storage = storage or storage_for(settings)
    identity = ClientIdentity(settings)
    previews = RateLimiter(settings.previews_per_ip_per_minute, 60.0)
    max_length_km = Job.model_fields["max_length_km"].default
    last_prune = [float("-inf")]

    def maybe_prune() -> None:
        """Retention, at most hourly, piggy-backed on traffic so no scheduler is needed."""
        now = store.clock()
        if now - last_prune[0] < PRUNE_EVERY_S:
            return
        last_prune[0] = now
        day = 86400.0
        for result in store.prune(settings.keep_days * day, settings.ip_hash_days * day):
            key = result.get("key")
            if not key:
                continue
            try:
                storage.delete(key)
            except Exception:  # an object store lifecycle rule is the backstop; never fail a request on this
                log.warning("could not delete download %s", key)

    app = FastAPI(title="mapcreator", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url="/api/openapi.json")
    app.state.store = store
    # Added innermost first: the security headers wrap everything, including the guard's refusals.
    app.add_middleware(RequestGuard, worker_token=settings.worker_token)
    app.add_middleware(SecurityHeaders, csp=content_security_policy(turnstile=turnstile is not None))

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": describe_validation_errors(exc.errors())}, status_code=422)

    # ----- public -----

    def builder_online() -> bool:
        seen = store.worker_last_seen()
        return seen is not None and store.clock() - seen < BUILDER_ONLINE_S

    @app.get("/api/health")
    def health() -> dict:
        store.count_queued()  # proves the database answers
        return {"status": "ok"}

    @app.get("/api/status")
    def service_status() -> dict:
        """Whether tracks are being built right now, for the page to warn before someone submits."""
        return {"builder_online": builder_online(), "queued": store.count_queued()}

    @app.get("/api/config")
    def client_config() -> dict:
        """What the map page needs to know before it starts."""
        return {
            "regions": [{"key": r.key, "name": r.name, "bbox": list(r.bbox)} for r in REGIONS],
            "max_waypoints": MAX_WAYPOINTS,
            "max_length_km": max_length_km,
            "turnstile_site_key": settings.turnstile_site_key if turnstile is not None else None,
        }

    @app.post("/api/jobs", status_code=201)
    def submit_job(body: JobRequest, request: Request, response: Response) -> dict:
        maybe_prune()
        if turnstile is not None and not turnstile.verify(body.turnstile_token or ""):
            raise HTTPException(403, "The anti-robot check didn't pass. Reload the page and try again.")
        try:
            check_region(body.job.waypoints)
        except OutOfArea as e:
            raise HTTPException(422, str(e)) from None

        # The same road with the same options is the same track: hand back the existing job.
        same = route_hash(body.job)
        existing = store.find_done_by_route_hash(same) or store.find_active_by_route_hash(same)
        if existing is not None:
            response.status_code = 200
            return {"id": existing.id, "status": existing.status}

        if store.count_queued() >= settings.max_queued_jobs:
            raise HTTPException(503, "The build queue is full right now. Please try again later.",
                                headers={"Retry-After": "900"})
        who = identity.key(request)
        if store.count_recent_for_ip(who) >= settings.jobs_per_ip_per_day:
            raise HTTPException(429, f"You've reached the limit of {settings.jobs_per_ip_per_day} tracks a day. "
                                     "Please try again tomorrow.", headers={"Retry-After": "3600"})
        job = body.job.model_copy(update={"track_id": public_track_id(body.job.track_id, same)})
        record = store.create(job, ip_hash=who)
        log.info("job %s queued", record.id)
        return {"id": record.id, "status": record.status}

    def absolute(url: str, request: Request) -> str:
        """Download links leave the page (Content Manager fetches them itself), so they must be absolute."""
        if not url.startswith("/"):
            return url
        return (settings.public_url or str(request.base_url)).rstrip("/") + url

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, request: Request) -> dict:
        record = store.get(job_id) if JOB_ID.match(job_id) else None
        if record is None:
            raise HTTPException(404, "There's no job with that id.")
        out: dict[str, Any] = {
            "id": record.id,
            "status": record.status,
            "created_at": _iso(record.created_at),
            "started_at": _iso(record.started_at),
            "finished_at": _iso(record.finished_at),
            "queue_position": store.queue_position(record),
            "stage": None,
            "builder_online": None,
            "summary": None,
            "download_url": None,
            "error": None,
        }
        if record.status == "running" and record.stage in STAGE_NAMES:
            out["stage"] = {"name": record.stage, "number": STAGE_NAMES.index(record.stage) + 1,
                            "of": len(STAGE_NAMES)}
        if record.status in ("queued", "running"):
            out["builder_online"] = builder_online()
        if record.status == "done" and record.result:
            out["summary"] = record.result.get("summary") or {}
            out["download_url"] = absolute(storage.url(record.result["key"]), request)
        elif record.status == "failed":
            out["error"] = record.error or "The build failed."
        return out

    @app.post("/api/route-preview")
    def route_preview(body: PreviewRequest, request: Request) -> RoutePreview:
        if not previews.allow(identity.key(request)):
            raise HTTPException(429, "That's a lot of route previews. Wait a minute and try again.",
                                headers={"Retry-After": "60"})
        try:
            return router.preview(body.waypoints)
        except OutOfArea as e:
            raise HTTPException(422, str(e)) from None
        except RouterUnavailable as e:
            raise HTTPException(503, str(e)) from None
        except RoutingError as e:
            raise HTTPException(422, str(e)) from None

    # ----- workers (RequestGuard has already checked the token) -----

    @app.post(WORKER_PREFIX + "/claim", include_in_schema=False, response_model=None)
    def worker_claim() -> Response | dict:
        store.worker_seen()
        maybe_prune()
        store.requeue_stale(settings.stale_after_minutes * 60)
        record = store.claim_next()
        if record is None:
            return Response(status_code=204)
        log.info("job %s claimed (attempt %d)", record.id, record.attempts)
        return {"id": record.id, "worker_id": record.worker_id, "attempt": record.attempts, "job": record.job}

    def _lost() -> HTTPException:
        return HTTPException(409, "This worker no longer holds the job.")

    @app.post(WORKER_PREFIX + "/jobs/{job_id}/heartbeat", include_in_schema=False)
    def worker_heartbeat(job_id: str, body: HeartbeatRequest) -> dict:
        store.worker_seen()
        if not store.heartbeat(job_id, body.worker_id, body.stage):
            raise _lost()
        return {"ok": True}

    @app.post(WORKER_PREFIX + "/jobs/{job_id}/complete", include_in_schema=False)
    def worker_complete(job_id: str, body: CompleteRequest) -> dict:
        store.worker_seen()
        if not store.complete(job_id, body.worker_id, body.result.model_dump()):
            raise _lost()
        log.info("job %s done", job_id)
        return {"ok": True}

    @app.post(WORKER_PREFIX + "/jobs/{job_id}/fail", include_in_schema=False)
    def worker_fail(job_id: str, body: FailRequest) -> dict:
        store.worker_seen()
        if not store.fail(job_id, body.worker_id, _clean_error(body.error)):
            raise _lost()
        log.info("job %s failed", job_id)
        return {"ok": True}

    # ----- files -----

    if settings.serves_downloads:
        downloads = settings.storage_dir.resolve()
        downloads.mkdir(parents=True, exist_ok=True)

        @app.api_route(settings.download_base.rstrip("/") + "/{key:path}", methods=["GET", "HEAD"],
                       include_in_schema=False)
        def download(key: str) -> FileResponse:
            """A finished track, as an attachment Content Manager will install. Range requests work."""
            try:
                headers = download_headers(key)
            except ValueError:
                raise HTTPException(404, "There's no download here.") from None
            path = downloads / key
            if not path.is_file():
                raise HTTPException(404, "There's no download here. It may have expired.")
            return FileResponse(path, media_type=headers.pop("Content-Type"), headers=headers)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    """`mapcreator-web`: serve the map page and API with uvicorn."""
    ap = argparse.ArgumentParser(prog="mapcreator-web", description="Run the mapcreator web front end and job API.")
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on (default: %(default)s)")
    ap.add_argument("--port", type=int, default=8000, help="port to listen on (default: %(default)s)")
    ap.add_argument("--access-log", action="store_true",
                    help="log every request; off by default because access logs record client addresses")
    args = ap.parse_args(argv)

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(ServiceSettings.from_env())
    # proxy_headers=False: client addresses come only from ClientIdentity's trusted-proxy rule.
    uvicorn.run(app, host=args.host, port=args.port, proxy_headers=False, server_header=False,
                access_log=args.access_log, log_level="info")
    return 0
