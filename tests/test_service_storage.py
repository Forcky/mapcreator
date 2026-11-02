import httpx
import pytest

from mapcreator.service.storage import LocalStorage, check_key
from mapcreator.service.turnstile import SITEVERIFY_URL, CloudflareTurnstile


def test_local_storage_copies_and_builds_urls(tmp_path):
    src = tmp_path / "build.zip"
    src.write_bytes(b"PK zip bytes")
    store = LocalStorage(tmp_path / "downloads", "https://downloads.example.org/tracks/")
    store.put(src, "abc123/my_road.zip")
    assert (tmp_path / "downloads" / "abc123" / "my_road.zip").read_bytes() == b"PK zip bytes"
    assert not list((tmp_path / "downloads" / "abc123").glob(".partial-*"))
    assert store.url("abc123/my_road.zip") == "https://downloads.example.org/tracks/abc123/my_road.zip"


@pytest.mark.parametrize("key", ["../x.zip", "a/../../x.zip", "/abs.zip", "a//b.zip", "a\\b.zip", "..", "a/.",
                                 "", "a b.zip", "C:/x.zip", "a/b/c/d/e.zip"])
def test_keys_cannot_escape_the_folder(key, tmp_path):
    with pytest.raises(ValueError):
        check_key(key)
    with pytest.raises(ValueError):
        LocalStorage(tmp_path, "/downloads").put(tmp_path / "unused", key)


def turnstile(handler) -> CloudflareTurnstile:
    return CloudflareTurnstile("secret-value", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_turnstile_posts_the_token_without_the_client_ip():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["form"] = request.content.decode()
        return httpx.Response(200, json={"success": True})

    assert turnstile(handler).verify("tok")
    assert seen["url"] == SITEVERIFY_URL
    assert "response=tok" in seen["form"] and "secret=secret-value" in seen["form"]
    assert "remoteip" not in seen["form"]


def test_turnstile_fails_closed():
    assert not turnstile(lambda r: httpx.Response(200, json={"success": False})).verify("tok")
    assert not turnstile(lambda r: httpx.Response(500)).verify("tok")
    assert not turnstile(lambda r: httpx.Response(200, text="not json")).verify("tok")

    def down(request):
        raise httpx.ConnectError("down")

    assert not turnstile(down).verify("tok")
    assert not turnstile(lambda r: httpx.Response(200, json={"success": True})).verify("")


def test_download_headers_name_a_quoted_zip():
    from mapcreator.service.storage import download_headers

    assert download_headers("abc123/my_road.zip") == {
        "Content-Type": "application/zip",
        "Content-Disposition": 'attachment; filename="my_road.zip"',
    }
    with pytest.raises(ValueError):
        download_headers("abc123/my_road.txt")
    with pytest.raises(ValueError):
        download_headers("../my_road.zip")
