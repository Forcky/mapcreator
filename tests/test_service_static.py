import hashlib
import re
from pathlib import Path

from fastapi.testclient import TestClient

import mapcreator.service
from mapcreator.service.api import create_app
from mapcreator.service.settings import ServiceSettings

STATIC = Path(mapcreator.service.__file__).parent / "static"
VENDOR = STATIC / "vendor"

# Must match THIRD_PARTY_NOTICES.md. A change here means the vendored library changed.
MAPLIBRE_SHA256 = {
    "maplibre-gl.js": "8b2c6bfadb8cc2d720aea2816ac368e51ab75dbb81ee3fac2f8b97256ebd1585",
    "maplibre-gl.css": "ab1e70d59ec40465bae7e7030da2f3ccf28133fd502e62bd598eefbadfd7a732",
    "LICENSE.txt": "ee5fc05a0677eaf69601d2c7db0d9ecd6cc27c3abc1d0733bc9ed34707cf8ef2",
}


def test_vendored_files_have_no_source_map_links():
    """Source maps aren't shipped, and a dangling link makes browsers fetch a file that 404s."""
    files = [p for p in VENDOR.rglob("*") if p.is_file()]
    assert files, "nothing vendored"
    assert [p.name for p in files if b"sourceMappingURL" in p.read_bytes()] == []


def test_maplibre_is_the_recorded_release_with_its_licence():
    folder = VENDOR / "maplibre-gl"
    assert {p.name for p in folder.iterdir()} == set(MAPLIBRE_SHA256)
    for name, digest in MAPLIBRE_SHA256.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name


INDEX = STATIC / "index.html"


def test_page_has_no_inline_script_or_style():
    """The CSP forbids inline script and style, so the page must not depend on any."""
    html = INDEX.read_text(encoding="utf-8")
    scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", html, flags=re.S | re.I)
    assert scripts and all('src="' in attrs and body.strip() == "" for attrs, body in scripts)
    assert not re.search(r"<style\b", html, re.I)
    assert not re.search(r"\sstyle\s*=", html, re.I)
    assert not re.search(r"\son[a-z]+\s*=", html, re.I)  # onclick= and friends
    assert "javascript:" not in html.lower()


def test_page_references_only_files_that_exist():
    html = INDEX.read_text(encoding="utf-8")
    local = [ref for ref in re.findall(r'(?:src|href)="([^"]+)"', html)
             if not ref.startswith(("https://", "#", "data:"))]
    assert local
    assert [ref for ref in local if not (STATIC / ref).is_file()] == []


def test_script_never_parses_api_text_as_html():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for risky in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert risky not in js, risky


def test_static_files_are_served_with_the_security_headers(tmp_path):
    settings = ServiceSettings(db_path=tmp_path / "jobs.sqlite3", storage_dir=tmp_path / "downloads", ip_salt="s")
    client = TestClient(create_app(settings, router=object()))
    page = client.get("/")
    assert page.status_code == 200 and '<div id="map"' in page.text
    assert page.headers["content-type"].startswith("text/html")
    csp = page.headers["Content-Security-Policy"]
    assert "script-src 'self';" in csp and "unsafe-inline" not in csp and "frame-ancestors 'none'" in csp
    assert page.headers["X-Content-Type-Options"] == "nosniff"
    for path, kind in (("/app.js", "javascript"), ("/style.css", "text/css"),
                       ("/vendor/maplibre-gl/maplibre-gl.js", "javascript")):
        r = client.get(path)
        assert r.status_code == 200 and kind in r.headers["content-type"], path
        assert r.headers["Content-Security-Policy"] == csp


ABOUT = STATIC / "about.html"


def test_about_page_runs_no_script_and_links_only_to_what_exists():
    html = ABOUT.read_text(encoding="utf-8")
    assert not re.search(r"<script\b|<style\b|\sstyle\s*=|\son[a-z]+\s*=", html, re.I)
    assert "javascript:" not in html.lower()
    local = [ref for ref in re.findall(r'(?:src|href)="([^"]+)"', html)
             if not ref.startswith(("https://", "#", "data:")) and not ref.endswith("/")]
    assert local and [ref for ref in local if not (STATIC / ref).is_file()] == []
    # The map page links to it, and it says what is kept about people.
    assert 'href="about.html"' in INDEX.read_text(encoding="utf-8")
    for promise in ("no cookies", "30 days", "two days", "Kunos"):
        assert promise in html, promise


def test_about_page_is_served_with_the_security_headers(tmp_path):
    settings = ServiceSettings(db_path=tmp_path / "jobs.sqlite3", storage_dir=tmp_path / "downloads", ip_salt="s")
    client = TestClient(create_app(settings, router=object()))
    page = client.get("/about.html")
    assert page.status_code == 200 and "Privacy" in page.text
    assert "unsafe-inline" not in page.headers["Content-Security-Policy"]


def test_pages_are_never_cached_and_name_assets_by_their_contents(tmp_path):
    import hashlib
    import re as _re

    from fastapi.testclient import TestClient

    from mapcreator.service.api import IMMUTABLE, create_app
    from mapcreator.service.settings import ServiceSettings

    settings = ServiceSettings(db_path=tmp_path / "jobs.sqlite3", storage_dir=tmp_path / "dl", download_base="/dl",
                               worker_token="w" * 40, ip_salt="s")
    client = TestClient(create_app(settings, router=object()))
    static = Path(__file__).parents[1] / "src" / "mapcreator" / "service" / "static"
    v = hashlib.sha256((static / "app.js").read_bytes()).hexdigest()[:10]
    for path in ("/", "/index.html"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
        assert f'src="app.js?v={v}"' in r.text and _re.search(r'href="style\.css\?v=[0-9a-f]{10}"', r.text)
        assert _re.search(r'src="vendor/maplibre-gl/maplibre-gl\.js\?v=[0-9a-f]{10}"', r.text)
    about = client.get("/about.html")
    assert about.headers["cache-control"] == "no-cache" and "style.css?v=" in about.text
    assert "Content-Security-Policy" in about.headers  # the security headers still apply
    assert client.get(f"/app.js?v={v}").headers["cache-control"] == IMMUTABLE
    assert client.get("/app.js").headers["cache-control"] == "no-cache"  # unversioned: always revalidated
    assert client.get("/app.js?v=0000000000").headers["cache-control"] == "no-cache"  # an old name
    assert client.get("/api/health").headers["cache-control"] == "no-store"
