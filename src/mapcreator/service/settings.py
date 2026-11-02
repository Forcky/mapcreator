"""Settings for the hosted service, read from `MAPCREATOR_*` environment variables.

Kept apart from `mapcreator.config` because the web front end and the build
worker run on different machines with different secrets: the web side never
needs the DEM mirror, and the worker never needs the Turnstile secret or the
IP salt. Every value has a local-development default except the secrets,
which default to "off": with no worker token the worker API doesn't exist,
and with no Turnstile secret submissions aren't challenged.
"""

from __future__ import annotations

import ipaddress
import os
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ROUTERS = ("dev", "graphhopper")
# Long enough that guessing is hopeless; `secrets.token_urlsafe(32)` gives 43 characters.
MIN_TOKEN_LENGTH = 32
_HEADER_NAME = re.compile(r"^[A-Za-z0-9-]{1,64}$")


@dataclass(frozen=True)
class ServiceSettings:
    # Paths default to the gitignored `out/` folder under the working directory.
    db_path: Path = Path("out/service/jobs.sqlite3")
    # Bearer token the build workers present. None disables the worker API (it answers 404).
    worker_token: str | None = None
    # Cloudflare Turnstile. No secret means submissions aren't challenged (local development).
    turnstile_secret: str | None = None
    turnstile_site_key: str | None = None
    # Route previews: "dev" asks Overpass through the build's own OSM graph; "graphhopper" a self-hosted server.
    router: str = "dev"
    graphhopper_url: str | None = None
    # Where finished tracks are stored, and the public URL prefix they're downloaded from. A prefix
    # starting with "/" makes the web app serve `storage_dir` itself, which only suits local runs.
    storage_dir: Path = Path("out/service/downloads")
    download_base: str = "/downloads"
    # Abuse limits.
    jobs_per_ip_per_day: int = 5
    max_queued_jobs: int = 50
    previews_per_ip_per_minute: int = 30
    # Client IPs are only ever stored as a keyed hash. Without a configured salt a random one is used,
    # so per-IP limits reset whenever the process restarts.
    ip_salt: str = field(default_factory=lambda: secrets.token_hex(32), repr=False)
    # Peers (addresses or networks) allowed to tell us the client's address, and the header they use.
    trusted_proxies: tuple[str, ...] = ()
    client_ip_header: str | None = None
    # A running job with no heartbeat for this long is handed to another worker (or failed).
    stale_after_minutes: float = 10.0

    def __post_init__(self) -> None:
        if self.router not in ROUTERS:
            raise ValueError(f"MAPCREATOR_ROUTER must be one of {', '.join(ROUTERS)}, not {self.router!r}")
        if self.router == "graphhopper" and not _is_http_url(self.graphhopper_url):
            raise ValueError("MAPCREATOR_GRAPHHOPPER_URL must be an http(s) URL when the router is graphhopper")
        if self.worker_token is not None and len(self.worker_token) < MIN_TOKEN_LENGTH:
            raise ValueError(f"MAPCREATOR_WORKER_TOKEN must be at least {MIN_TOKEN_LENGTH} characters")
        if not self.ip_salt:
            raise ValueError("MAPCREATOR_IP_SALT must not be empty")
        for name in ("jobs_per_ip_per_day", "max_queued_jobs", "previews_per_ip_per_minute"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.stale_after_minutes <= 0:
            raise ValueError("stale_after_minutes must be positive")
        for proxy in self.trusted_proxies:
            ipaddress.ip_network(proxy, strict=False)  # raises ValueError on rubbish
        if self.client_ip_header is not None and not _HEADER_NAME.match(self.client_ip_header):
            raise ValueError("MAPCREATOR_CLIENT_IP_HEADER must be a plain header name")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServiceSettings:
        env = os.environ if env is None else env

        def get(name: str) -> str | None:
            value = env.get(f"MAPCREATOR_{name}", "").strip()
            return value or None

        kwargs: dict = {}
        for attr, name, conv in (
            ("db_path", "DB", Path),
            ("worker_token", "WORKER_TOKEN", str),
            ("turnstile_secret", "TURNSTILE_SECRET", str),
            ("turnstile_site_key", "TURNSTILE_SITE_KEY", str),
            ("router", "ROUTER", str.lower),
            ("graphhopper_url", "GRAPHHOPPER_URL", str),
            ("storage_dir", "STORAGE_DIR", Path),
            ("download_base", "DOWNLOAD_BASE", str),
            ("jobs_per_ip_per_day", "JOBS_PER_IP_PER_DAY", int),
            ("max_queued_jobs", "MAX_QUEUED_JOBS", int),
            ("previews_per_ip_per_minute", "PREVIEWS_PER_IP_PER_MINUTE", int),
            ("ip_salt", "IP_SALT", str),
            ("trusted_proxies", "TRUSTED_PROXIES", _csv),
            ("client_ip_header", "CLIENT_IP_HEADER", str),
            ("stale_after_minutes", "STALE_MINUTES", float),
        ):
            raw = get(name)
            if raw is not None:
                try:
                    kwargs[attr] = conv(raw)
                except ValueError as e:
                    raise ValueError(f"MAPCREATOR_{name}: {e}") from None
        return cls(**kwargs)

    @property
    def serves_downloads(self) -> bool:
        """True when downloads are a path on this web app rather than an object store's URL."""
        return self.download_base.startswith("/")


def _csv(raw: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def _is_http_url(url: str | None) -> bool:
    return bool(url) and url.startswith(("http://", "https://"))
