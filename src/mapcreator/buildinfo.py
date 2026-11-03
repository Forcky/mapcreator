"""Which version of the track-making code built a track.

A finished track is handed back when someone asks for the same road again, which saves a build. But a fix that
changes what the build makes (water, 3D trees, a corner rule) should reach the next person who asks. So every
build records `version()`, a hash of the code that makes tracks, and the web app reuses a finished track only
when the same code built it.

The web app and the worker compute the same hash when they run one commit. Line endings are normalised first,
because a checkout on Windows has CRLFs where one on Linux has LFs.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

# The packages and modules a track is made from. The web service, the CLI and the docs don't change the output.
SOURCES = ("core", "pipeline", "providers", "regions")
MODULES = ("job.py", "config.py")


@functools.cache
def version() -> str:
    root = Path(__file__).parent
    paths = sorted(p for d in SOURCES for p in (root / d).rglob("*.py")) + [root / m for m in MODULES]
    h = hashlib.sha256()
    for p in paths:
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]
