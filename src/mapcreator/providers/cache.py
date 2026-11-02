"""Writing to the on-disk caches safely.

Several builds can share one cache (two workers on one volume, or the CLI next
to a worker). Each write goes to a uniquely named temporary file in the same
folder and is then moved into place in one step, so a reader sees either no
file or a whole one, never a half-written one, even if the writer is killed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def _publish(tmp: str, path: Path) -> None:
    try:
        os.replace(tmp, path)
    except PermissionError:
        # Windows refuses to replace a file another process has open. If that process just wrote the
        # same entry, ours is redundant.
        os.unlink(tmp)
        if not path.exists():
            raise


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        _publish(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path: Path, data) -> None:
    write_bytes(path, json.dumps(data).encode("utf-8"))


def read_json(path: Path):
    """The cached value, or None when there's none or it can't be read (then the caller fetches again)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".npy")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array)
        _publish(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
