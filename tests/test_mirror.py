"""scripts/mirror.py resumes only from the same version of the source file."""

import importlib.util
from pathlib import Path

import httpx
import pytest

SPEC = importlib.util.spec_from_file_location("mirror", Path(__file__).resolve().parents[1] / "scripts" / "mirror.py")
mirror = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mirror)


class Source:
    """A file served with an ETag, Range and If-Range, as an object store would."""

    def __init__(self, body: bytes, etag: str):
        self.body, self.etag = body, etag

    def handler(self, request: httpx.Request) -> httpx.Response:
        headers = {"content-length": str(len(self.body)), "etag": self.etag}
        if request.method == "HEAD":
            return httpx.Response(200, headers=headers)
        rng = request.headers.get("range")
        if rng and request.headers.get("if-range", self.etag) == self.etag:
            start = int(rng.split("=")[1].rstrip("-"))
            return httpx.Response(206, content=self.body[start:], headers={"etag": self.etag})
        return httpx.Response(200, content=self.body, headers={"etag": self.etag})


@pytest.fixture
def serve(monkeypatch):
    def use(source: Source) -> None:
        real = httpx.Client
        monkeypatch.setattr(mirror.httpx, "Client",
                            lambda **kw: real(transport=httpx.MockTransport(source.handler), **kw))

    monkeypatch.setattr(mirror.time, "sleep", lambda s: None)
    return use


def test_a_fresh_download_records_its_version(tmp_path, serve):
    serve(Source(b"A" * 1000, '"v1"'))
    dest = mirror.download("https://data.example.org/f.zip", tmp_path / "f.zip")
    assert dest.read_bytes() == b"A" * 1000
    assert (tmp_path / "f.zip.version").read_text() == '"v1"'
    assert not (tmp_path / "f.zip.part").exists()


def test_a_partial_from_another_version_is_never_spliced(tmp_path, serve):
    # 400 bytes of version 1 were downloaded; the source has since been republished at the same size.
    (tmp_path / "f.zip.part").write_bytes(b"A" * 400)
    (tmp_path / "f.zip.part.version").write_text('"v1"')
    serve(Source(b"B" * 1000, '"v2"'))
    dest = mirror.download("https://data.example.org/f.zip", tmp_path / "f.zip")
    assert dest.read_bytes() == b"B" * 1000


def test_a_partial_from_the_same_version_resumes(tmp_path, serve):
    (tmp_path / "f.zip.part").write_bytes(b"C" * 400)
    (tmp_path / "f.zip.part.version").write_text('"v1"')
    serve(Source(b"C" * 1000, '"v1"'))
    assert mirror.download("https://data.example.org/f.zip", tmp_path / "f.zip").read_bytes() == b"C" * 1000


def test_a_republished_source_replaces_a_complete_mirror(tmp_path, serve):
    (tmp_path / "f.zip").write_bytes(b"A" * 1000)
    (tmp_path / "f.zip.version").write_text('"v1"')
    serve(Source(b"D" * 1000, '"v2"'))
    assert mirror.download("https://data.example.org/f.zip", tmp_path / "f.zip").read_bytes() == b"D" * 1000


def test_a_republish_between_head_and_get_is_followed_not_spliced(tmp_path, serve):
    (tmp_path / "f.zip.part").write_bytes(b"A" * 400)
    (tmp_path / "f.zip.part.version").write_text('"v1"')
    source = Source(b"A" * 1000, '"v1"')
    real_handler = source.handler

    def republished_after_head(request):
        if request.method == "GET":  # the HEAD saw v1; by the GET it's v2, smaller
            source.body, source.etag = b"E" * 700, '"v2"'
        return real_handler(request)

    source.handler = republished_after_head
    serve(source)
    assert mirror.download("https://data.example.org/f.zip", tmp_path / "f.zip").read_bytes() == b"E" * 700
    assert (tmp_path / "f.zip.version").read_text() == '"v2"'
