"""Cloudflare Turnstile: a challenge on job submission, so the queue isn't filled by scripts.

Verification fails closed: if Cloudflare can't be reached, the submission is
refused rather than let through unchallenged. The client's IP address is not
sent along (the `remoteip` field is optional), so no more of it leaves the
service than the widget itself already sees.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

log = logging.getLogger(__name__)

SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
# Cloudflare's tokens are at most 2048 characters.
MAX_TOKEN_LENGTH = 2048


class Verifier(Protocol):
    def verify(self, token: str) -> bool: ...


class CloudflareTurnstile:
    def __init__(self, secret: str, *, client: httpx.Client | None = None, timeout_s: float = 10.0):
        self._secret = secret
        self._client = client or httpx.Client(timeout=timeout_s, follow_redirects=False)

    def verify(self, token: str) -> bool:
        if not token or len(token) > MAX_TOKEN_LENGTH:
            return False
        try:
            r = self._client.post(SITEVERIFY_URL, data={"secret": self._secret, "response": token})
            r.raise_for_status()
            return r.json().get("success") is True
        except (httpx.HTTPError, ValueError, AttributeError):
            log.warning("Turnstile verification could not be completed; refusing the submission")
            return False
