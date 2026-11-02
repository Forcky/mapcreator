import hashlib
from pathlib import Path

import mapcreator.service

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
