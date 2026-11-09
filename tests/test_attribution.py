"""ATTRIBUTION.txt: the data sources, that the data was changed, the ODbL licence, and how the OSM data was selected."""

from mapcreator import config
from mapcreator.pipeline import track_stage
from mapcreator.providers.base import OSM, TAS_DEM
from mapcreator.providers.osm import overpass, roads

ODBL = "https://opendatacommons.org/licenses/odbl/1-0/"
INDEXED = {"index": "australia-latest.osm.pbf", "osm_timestamp": "2026-11-01T20:21:02Z"}


def _attribution(tmp_path, **kw) -> str:
    track_stage._write_attribution(tmp_path, [OSM, TAS_DEM], "Grasstree Hill Road", **kw)
    raw = (tmp_path / "ATTRIBUTION.txt").read_bytes()
    assert b"\n" not in raw.replace(b"\r\n", b"")  # CRLF throughout
    return raw.decode("utf-8")


def test_attribution_marks_the_changes_and_gives_the_odbl_licence(tmp_path):
    text = _attribution(tmp_path)
    assert f"- {OSM.line()}\r\n- {TAS_DEM.line()}\r\n" in text
    # CC BY 4.0 and CC BY 3.0 AU both ask for changes to be marked.
    assert "Changes: mapcreator modified all of the data above to build this track." in text
    assert f"(ODbL) 1.0\r\n({ODBL})." in text
    assert "power.geojson" not in text and "selected" not in text


def test_attribution_says_how_the_osm_data_was_selected(tmp_path):
    note = roads.selection_note(INDEXED, build="0123456789ab", source_url="https://example.org/mapcreator")
    text = _attribution(tmp_path, power_geojson=True, osm_note=note)
    assert "power.geojson holds the OpenStreetMap power lines" in text
    assert f"also under the ODbL 1.0.\r\n{note}\r\n\r\n" in text


def test_the_note_names_the_index_builder_its_source_and_the_build():
    note = roads.selection_note(INDEXED, build="0123456789ab", source_url="https://example.org/mapcreator")
    assert note == ("The OpenStreetMap data was selected with mapcreator's index builder "
                    "(src/mapcreator/providers/osm/index.py at https://example.org/mapcreator, build 0123456789ab) "
                    "from australia-latest.osm.pbf, with data as of 2026-11-01T20:21:02Z.")


def test_the_note_leaves_out_what_isnt_known():
    # No source URL configured, and an index read from XML, which records no OSM timestamp.
    note = roads.selection_note({"index": "fixture.osm", "osm_timestamp": ""}, build="0123456789ab")
    assert note == ("The OpenStreetMap data was selected with mapcreator's index builder "
                    "(src/mapcreator/providers/osm/index.py, build 0123456789ab) from fixture.osm.")
    assert " at " not in note and "<" not in note


def test_without_an_index_the_note_gives_the_overpass_filter(tmp_path):
    sel = roads.selection(config.Settings(cache_dir=tmp_path, work_dir=tmp_path, tas_dem_source="",
                                          tas_chm_source="", overpass_url="", user_agent="", osm_index=""))
    assert sel == {"overpass_filter": overpass.ROADS_FILTER}
    assert overpass.ROADS_FILTER in overpass.query_for_bbox(-42.82, 147.28, -42.78, 147.33)
    note = roads.selection_note(sel, build="0123456789ab")
    assert note == ("The OpenStreetMap data was selected from an Overpass API server with the filter "
                    f"{overpass.ROADS_FILTER} around the route (src/mapcreator/providers/osm/overpass.py, "
                    "build 0123456789ab).")


def test_the_source_url_setting_is_empty_unless_set(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.local.toml")  # none
    monkeypatch.delenv("MAPCREATOR_SOURCE_URL", raising=False)
    assert config.load().source_url == ""
    monkeypatch.setenv("MAPCREATOR_SOURCE_URL", " https://example.org/mapcreator\n")
    assert config.load().source_url == "https://example.org/mapcreator"
    monkeypatch.delenv("MAPCREATOR_SOURCE_URL")
    (tmp_path / "config.local.toml").write_text('source_url = "https://example.org/fork"\n', encoding="utf-8")
    assert config.load().source_url == "https://example.org/fork"
