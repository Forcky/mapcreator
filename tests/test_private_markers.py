"""This repo is meant to be publishable: fail if anything private gets committed.

Generic patterns live here. Deployment-specific words (host names, project
names) are kept out of this public file: list them one per line in a
gitignored `.private-markers` file, or point MAPCREATOR_PRIVATE_MARKERS at one.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Every tracked file is scanned except known binary formats.
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".ico", ".dds", ".kn5", ".zip", ".npz", ".npy", ".tif", ".tiff",
                   ".pbf", ".pdf", ".woff", ".woff2"}
# The whole reserved blocks may appear (e.g. an egress proxy refusing them); addresses inside them may not.
ALLOWED = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")

GENERIC = [
    (r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "private IPv4 (10/8)"),
    (r"\b192\.168\.\d{1,3}\.\d{1,3}\b", "private IPv4 (192.168/16)"),
    (r"\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b", "private IPv4 (172.16/12)"),
    (r"[A-Za-z]:[\\/]+Users[\\/]+\w+", "personal Windows path"),
    (r"/home/\w+/", "personal Linux path"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (r"\bAKIA[0-9A-Z]{16}\b", "AWS access key id"),
    (r"\bgh[pousr]_[A-Za-z0-9]{30,}\b", "GitHub token"),
]


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    files = [ROOT / line for line in out.splitlines() if line]
    return [f for f in files if f.is_file() and f.suffix.lower() not in BINARY_SUFFIXES]


def _extra_markers() -> list[str]:
    path = Path(os.environ.get("MAPCREATOR_PRIVATE_MARKERS", ROOT / ".private-markers"))
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def test_no_private_information():
    patterns = [(re.compile(p), why) for p, why in GENERIC]
    patterns += [(re.compile(re.escape(m), re.IGNORECASE), "private marker") for m in _extra_markers()]
    hits = []
    for f in _tracked_files():
        if f.resolve() == Path(__file__).resolve():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for block in ALLOWED:
            text = text.replace(block, "")
        for rx, why in patterns:
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{f.relative_to(ROOT)}:{line}: {why}")
    assert not hits, "private information in tracked files:\n" + "\n".join(hits)
