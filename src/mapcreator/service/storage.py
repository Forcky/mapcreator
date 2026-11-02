"""Where finished tracks go, and the public URL each one is downloaded from.

Two sides use a `Storage`: the worker calls `put` after a build, and the web
app calls `url` when it reports a finished job. They usually run on different
machines, so in production both point at the same object store and agree on
key names; the database only ever holds the key.

`LocalStorage` writes to a folder. It suits running the web app and a worker
on one machine, where the web app serves that folder itself.

Every download must carry `download_headers(key)`. Content Manager (the usual
AC mod manager) installs from a URL, and it only trusts a quoted
`Content-Disposition` filename ending in `.zip`. It also scrapes small HTML
responses for a link to follow, so a download URL must always answer with
the zip itself (or a plain redirect to it), never an HTML page.

`S3Storage` is the production shape: any S3-compatible store (S3, R2,
MinIO), via boto3 (optional extra `s3`). How it's meant to be used:
- `put` uploads with `download_headers(key)` set as the object's
  Content-Type and Content-Disposition, using credentials scoped to
  write-only on one bucket prefix (the worker should not be able to list or
  delete). Object stores answer Range requests themselves.
- `url` returns either a public bucket URL (`<public base>/<key>`) or a
  short-lived presigned GET; the web app then needs read credentials only.
- A bucket lifecycle rule expires old tracks; the job row keeps the key, so
  an expired download should be reported as such, not as a broken link.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

# `<job id>/<track id>.zip` in practice. Plain segments only, so a key can never climb out of a folder.
_SEGMENT = r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}"
KEY_PATTERN = re.compile(rf"^{_SEGMENT}(/{_SEGMENT}){{0,3}}$")


class Storage(Protocol):
    def put(self, local_path: Path, key: str) -> None: ...

    def url(self, key: str) -> str: ...

    def delete(self, key: str) -> None: ...  # retention; a missing object is not an error


def check_key(key: str) -> str:
    if not KEY_PATTERN.match(key) or any(part.strip(".") == "" for part in key.split("/")):
        raise ValueError(f"not a valid storage key: {key!r}")
    return key


def download_headers(key: str) -> dict[str, str]:
    """What a track download must say about itself, wherever it's served from (see the module notes)."""
    name = check_key(key).rsplit("/", 1)[-1]
    if not name.endswith(".zip"):
        raise ValueError(f"tracks are zip files: {key!r}")
    # Keys are plain ASCII segments, so the name never needs escaping inside the quotes.
    return {"Content-Type": "application/zip", "Content-Disposition": f'attachment; filename="{name}"'}


class LocalStorage:
    def __init__(self, directory: Path, public_base: str):
        self.directory = Path(directory)
        self.public_base = public_base.rstrip("/")

    def put(self, local_path: Path, key: str) -> None:
        target = self.directory / check_key(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy beside the target and rename, so a download never sees half a file.
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".partial-")
        os.close(fd)
        try:
            shutil.copyfile(local_path, tmp)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def url(self, key: str) -> str:
        return f"{self.public_base}/{quote(check_key(key))}"

    def delete(self, key: str) -> None:
        (self.directory / check_key(key)).unlink(missing_ok=True)


class S3Storage:
    """An S3-compatible bucket (Cloudflare R2 in production), with a public base URL for downloads.

    Objects are uploaded with `download_headers` as their Content-Type and
    Content-Disposition, so the bucket serves exactly what Content Manager
    needs. Range requests are answered by the store itself. `client` can be
    injected (tests use a fake); otherwise boto3 is imported lazily, so
    installs without the `s3` extra still work with local storage.
    """

    def __init__(self, *, bucket: str, public_base: str, endpoint: str | None = None, region: str = "auto",
                 access_key_id: str | None = None, secret_access_key: str | None = None, client=None):
        self.bucket = bucket
        self.public_base = public_base.rstrip("/")
        if client is None:
            import boto3  # optional dependency: pip install mapcreator[s3]

            client = boto3.client("s3", endpoint_url=endpoint, region_name=region,
                                  aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
        self.client = client

    def put(self, local_path: Path, key: str) -> None:
        headers = download_headers(key)
        with open(local_path, "rb") as f:
            self.client.put_object(Bucket=self.bucket, Key=check_key(key), Body=f,
                                   ContentType=headers["Content-Type"],
                                   ContentDisposition=headers["Content-Disposition"])

    def url(self, key: str) -> str:
        return f"{self.public_base}/{quote(check_key(key))}"

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=check_key(key))


def storage_for(settings) -> Storage:
    """The storage a service process should use, from its settings."""
    if settings.storage == "s3":
        return S3Storage(bucket=settings.s3_bucket, public_base=settings.download_base,
                         endpoint=settings.s3_endpoint, region=settings.s3_region,
                         access_key_id=settings.s3_access_key_id,
                         secret_access_key=settings.s3_secret_access_key)
    return LocalStorage(settings.storage_dir, settings.download_base)
