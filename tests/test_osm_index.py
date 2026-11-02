"""The local OSM index: built from a small hand-written extract, queried like Overpass, routed on."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
import shapely
from pyproj import Transformer

from mapcreator import config
from mapcreator.providers.base import LONLAT_CRS
from mapcreator.job import Job, Waypoint
from mapcreator.pipeline import route_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.osm import graph, index, overpass
from mapcreator.service.routing import DevRouter, RouterUnavailable

EPSG = 28355  # MGA zone 55, the fixture is near Hobart
# A block of road: Main Road runs west only along -42.800; Back Street loops round the north.
# Also: a tertiary road with a node missing from the extract, a footway and a service road
# (not drivable), a road far away, an L-shaped road whose box covers QUERY without entering it,
# a house between the roads, a shed far away, an unclosed "building" and a route relation.
FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="hand-written test fixture">
  <node id="1" lat="-42.8000000" lon="147.3000000"/>
  <node id="2" lat="-42.8000000" lon="147.3050000"><tag k="highway" v="traffic_signals"/></node>
  <node id="3" lat="-42.8000000" lon="147.3100000"/>
  <node id="4" lat="-42.7980000" lon="147.3000000"/>
  <node id="5" lat="-42.7980000" lon="147.3050000"/>
  <node id="6" lat="-42.7980000" lon="147.3100000"/>
  <node id="7" lat="-42.8000000" lon="147.3150000"/>
  <node id="8" lat="-42.8010000" lon="147.3020000"/>
  <node id="9" lat="-42.7990000" lon="147.3050000"/>
  <node id="20" lat="-42.7995000" lon="147.3060000"/>
  <node id="21" lat="-42.7995000" lon="147.3065000"/>
  <node id="22" lat="-42.7992000" lon="147.3065000"/>
  <node id="23" lat="-42.7992000" lon="147.3060000"/>
  <node id="24" lat="-42.7995000" lon="147.4000000"/>
  <node id="25" lat="-42.7995000" lon="147.4005000"/>
  <node id="26" lat="-42.7992000" lon="147.4005000"/>
  <node id="27" lat="-42.7992000" lon="147.4000000"/>
  <node id="28" lat="-42.7996000" lon="147.3080000"/>
  <node id="29" lat="-42.7996000" lon="147.3085000"/>
  <node id="30" lat="-42.7993000" lon="147.3085000"/>
  <node id="40" lat="-42.8000000" lon="147.4000000"/>
  <node id="41" lat="-42.8000000" lon="147.4100000"/>
  <node id="50" lat="-42.8100000" lon="147.2900000"/>
  <node id="51" lat="-42.7900000" lon="147.2900000"/>
  <node id="52" lat="-42.7900000" lon="147.3200000"/>
  <way id="10">
    <nd ref="3"/><nd ref="2"/><nd ref="1"/>
    <tag k="highway" v="secondary"/><tag k="oneway" v="yes"/><tag k="name" v="Main Road"/>
  </way>
  <way id="11">
    <nd ref="1"/><nd ref="4"/><nd ref="5"/><nd ref="6"/><nd ref="3"/>
    <tag k="highway" v="residential"/><tag k="name" v="Back Street"/>
  </way>
  <way id="12">
    <nd ref="3"/><nd ref="7"/><nd ref="999"/>
    <tag k="highway" v="tertiary"/>
  </way>
  <way id="13">
    <nd ref="2"/><nd ref="8"/>
    <tag k="highway" v="footway"/>
  </way>
  <way id="14">
    <nd ref="40"/><nd ref="41"/>
    <tag k="highway" v="primary"/>
  </way>
  <way id="15">
    <nd ref="50"/><nd ref="51"/><nd ref="52"/>
    <tag k="highway" v="unclassified"/>
  </way>
  <way id="16">
    <nd ref="5"/><nd ref="9"/>
    <tag k="highway" v="service"/>
  </way>
  <way id="30">
    <nd ref="20"/><nd ref="21"/><nd ref="22"/><nd ref="23"/><nd ref="20"/>
    <tag k="building" v="house"/><tag k="building:levels" v="2"/><tag k="addr:street" v="Back Street"/>
  </way>
  <way id="31">
    <nd ref="24"/><nd ref="25"/><nd ref="26"/><nd ref="27"/><nd ref="24"/>
    <tag k="building" v="shed"/>
  </way>
  <way id="32">
    <nd ref="28"/><nd ref="29"/><nd ref="30"/>
    <tag k="building" v="yes"/>
  </way>
  <relation id="100">
    <member type="way" ref="10" role=""/><member type="way" ref="11" role=""/>
    <tag k="type" v="route"/><tag k="route" v="road"/>
  </relation>
</osm>
"""
# West, south, east, north: around the block, clear of the far road and inside the L-shaped one.
QUERY = (147.295, -42.805, 147.318, -42.795)
TO_XY = Transformer.from_crs(LONLAT_CRS, EPSG, always_xy=True)


@pytest.fixture(scope="module")
def extract(tmp_path_factory):
    path = tmp_path_factory.mktemp("osm") / "fixture.osm"
    path.write_text(FIXTURE, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def osm_index(extract):
    out = extract.with_name("fixture.sqlite")
    index.build_index(extract, out, reader="xml")
    return out


def _ways_table(path) -> list[tuple]:
    with closing(sqlite3.connect(path)) as con:
        return con.execute("SELECT id, kind, tags_json, coords, node_ids FROM ways ORDER BY id").fetchall()


def test_build_keeps_drivable_roads_and_closed_buildings(extract, tmp_path):
    counts = index.build_index(extract, tmp_path / "out.sqlite", reader="xml")
    assert counts == {"roads": 5, "buildings": 2, "skipped": 3}  # footway, service, unclosed building
    rows = {r[0]: r for r in _ways_table(tmp_path / "out.sqlite")}
    assert {i for i, r in rows.items() if r[1] == "road"} == {10, 11, 12, 14, 15}
    assert {i for i, r in rows.items() if r[1] == "building"} == {30, 31}
    # Road node ids stay in way order with their coordinates; the node missing from the extract is dropped.
    assert np.frombuffer(rows[12][4], dtype="<i8").tolist() == [3, 7]
    assert np.frombuffer(rows[12][3], dtype="<f8").reshape(-1, 2).tolist() == [[147.31, -42.8], [147.315, -42.8]]
    # Buildings keep their shape tags only, and no node ids.
    assert json.loads(rows[30][2]) == {"building": "house", "building:levels": "2"}
    assert rows[30][4] is None
    meta = index.metadata(tmp_path / "out.sqlite")
    assert meta["source"] == "fixture.osm" and meta["roads"] == "5" and meta["schema"] == index.SCHEMA_VERSION
    assert not (tmp_path / "out.sqlite.partial").exists()


def test_query_returns_overpass_json_for_roads_crossing_the_box(osm_index):
    data = index.query(osm_index, QUERY)
    ways = [e for e in data["elements"] if e["type"] == "way"]
    nodes = [e for e in data["elements"] if e["type"] == "node"]
    # Not the far road, and not the L-shaped road whose box covers the query without entering it.
    assert [w["id"] for w in ways] == [10, 11, 12]
    assert ways[0] == {"type": "way", "id": 10, "nodes": [3, 2, 1],
                       "tags": {"highway": "secondary", "oneway": "yes", "name": "Main Road"}}
    assert [n["id"] for n in nodes] == [1, 2, 3, 4, 5, 6, 7]
    assert nodes[1] == {"type": "node", "id": 2, "lat": -42.8, "lon": 147.305}
    assert index.query(osm_index, (147.5, -42.9, 147.6, -42.8))["elements"] == []


def test_the_graph_routes_on_the_index_and_respects_one_way(osm_index):
    g = graph.build(index.query(osm_index, QUERY), TO_XY)
    east = np.column_stack(TO_XY.transform([147.300, 147.310], [-42.800, -42.800]))
    r = graph.route(g, east)  # Main Road is westbound only, so eastbound goes round Back Street
    assert {r.ways[i]["id"] for i in r.seg_way} == {11}
    r = graph.route(g, east[::-1])
    assert {r.ways[i]["id"] for i in r.seg_way} == {10}
    assert np.hypot(*np.diff(r.xy, axis=0).T).sum() == pytest.approx(0.01 * 111_320 * np.cos(np.radians(42.8)),
                                                                      rel=0.01)


def test_buildings_come_back_as_projected_polygons(osm_index):
    box = TO_XY.transform_bounds(*QUERY)
    polys = index.buildings(osm_index, box, EPSG)
    assert len(polys) == 1  # the house; not the far shed or the unclosed way
    x, y = TO_XY.transform([147.3060, 147.3065, 147.3065, 147.3060], [-42.7995, -42.7995, -42.7992, -42.7992])
    expected = shapely.Polygon(np.column_stack([x, y]))
    assert polys[0].is_valid and polys[0].area == pytest.approx(expected.area, rel=1e-9)
    assert polys[0].area == pytest.approx(41 * 33.3, rel=0.02)
    assert index.buildings(osm_index, (box[0], box[1], box[0] + 10, box[1] + 10), EPSG) == []


def test_a_missing_or_foreign_file_is_unavailable(tmp_path):
    with pytest.raises(index.IndexUnavailable, match="no OSM index"):
        index.query(tmp_path / "nope.sqlite", QUERY)
    with closing(sqlite3.connect(tmp_path / "other.sqlite")) as con:
        con.execute("CREATE TABLE t (x)")
    with pytest.raises(index.IndexUnavailable, match="not an OSM index"):
        index.query(tmp_path / "other.sqlite", QUERY)
    with pytest.raises(ValueError, match="west < east"):
        index.query(tmp_path / "other.sqlite", (147.3, -42.8, 147.2, -42.7))


def test_pbf_needs_pyosmium_when_reading_with_the_standard_library(tmp_path):
    (tmp_path / "x.osm.pbf").write_bytes(b"")
    with pytest.raises(RuntimeError, match="needs pyosmium"):
        index.build_index(tmp_path / "x.osm.pbf", tmp_path / "x.sqlite", reader="xml")


@pytest.mark.parametrize("reader", ["auto", "osmium"])
def test_a_pyosmium_that_fails_to_load_is_reported_as_such(reader, tmp_path, monkeypatch):
    # Installed but broken (e.g. a missing system library) must not read as "not installed".
    import builtins

    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "osmium" or name.startswith("osmium."):
            raise ImportError("libexpat.so.1: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    (tmp_path / "x.osm.pbf").write_bytes(b"")
    with pytest.raises(RuntimeError, match="installed but failed to load: libexpat"):
        index.build_index(tmp_path / "x.osm.pbf", tmp_path / "x.sqlite", reader=reader)


def test_pyosmium_builds_the_same_index_from_xml_and_pbf(extract, osm_index, tmp_path):
    osmium = pytest.importorskip("osmium")
    import osmium.io

    pbf = tmp_path / "fixture.osm.pbf"
    header = osmium.io.Header()
    header.set("osmosis_replication_timestamp", "2026-11-01T20:21:02Z")  # as Geofabrik's extracts carry
    # Objects from a FileProcessor, not osmium.apply(): every reader and writer here owns its thread
    # pool and frees it with itself, so nothing is left to tear down when the process exits.
    with osmium.SimpleWriter(str(pbf), header=header) as writer:
        for obj in osmium.FileProcessor(str(extract)):
            writer.add(obj)
    expected = _ways_table(osm_index)
    for source in (extract, pbf):
        out = tmp_path / (source.name + ".sqlite")
        assert index.build_index(source, out, reader="osmium") == {"roads": 5, "buildings": 2, "skipped": 3}
        assert _ways_table(out) == expected
        assert index.metadata(out)["reader"].startswith("pyosmium ")
    assert index.metadata(tmp_path / "fixture.osm.pbf.sqlite")["osm_timestamp"] == "2026-11-01T20:21:02Z"


def test_command_line_builds_and_queries(extract, tmp_path, capsys):
    out = tmp_path / "cli.sqlite"
    assert index.main(["build", str(extract), str(out), "--reader", "xml"]) == 0
    assert "indexed 5 roads and 2 buildings" in capsys.readouterr().out
    assert index.main(["query", str(out), *map(str, QUERY)]) == 0
    text = capsys.readouterr().out
    assert "roads: 3 ways, 7 nodes" in text and "buildings: 1 in EPSG:28355" in text
    assert index.main(["query", str(out), *map(str, QUERY), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["elements"]) == 10


# ----- choosing the source: the index when configured, else Overpass -----

# Westbound along Main Road, the way the one-way allows.
WESTBOUND = [Waypoint(lon=147.3095, lat=-42.8), Waypoint(lon=147.3005, lat=-42.8)]


def _settings(tmp_path, osm_index="") -> config.Settings:
    return replace(config.load(), osm_index=str(osm_index), cache_dir=tmp_path / "cache", work_dir=tmp_path / "out")


def _no_overpass(query, settings):
    raise AssertionError("Overpass must not be asked when an index is configured")


def test_route_stage_reads_the_index_when_configured(osm_index, tmp_path, monkeypatch):
    monkeypatch.setattr(overpass, "fetch", _no_overpass)
    job = Job(track_id="index_road", waypoints=WESTBOUND)
    result = route_stage.run(job, _settings(tmp_path, osm_index), WorkDir(tmp_path / "work"))
    assert result["region"] == "tas" and result["ways"] == 1
    assert result["length_m"] == pytest.approx(0.009 * 111_320 * np.cos(np.radians(42.8)), rel=0.01)
    runs = json.loads((tmp_path / "work" / route_stage.STAGE / "runs.json").read_text(encoding="utf-8"))
    assert runs[0]["way_id"] == 10 and runs[0]["tags"]["name"] == "Main Road"
    assert not (tmp_path / "cache" / "osm").exists()  # index answers aren't cached


def test_route_stage_asks_overpass_without_an_index(osm_index, tmp_path, monkeypatch):
    queries = []

    def fake_overpass(query, settings):
        queries.append(query)
        return index.query(osm_index, (147.2805, -42.82, 147.3295, -42.78))

    monkeypatch.setattr(overpass, "fetch", fake_overpass)
    job = Job(track_id="overpass_road", waypoints=WESTBOUND)
    result = route_stage.run(job, _settings(tmp_path), WorkDir(tmp_path / "work"))
    assert result["ways"] == 1
    # Overpass takes south, west, north, east.
    assert len(queries) == 1 and "(-42.820000,147.280500,-42.780000,147.329500)" in queries[0]


def test_dev_router_reads_the_index_when_configured(osm_index, tmp_path):
    preview = DevRouter(settings=_settings(tmp_path, osm_index), fetch=_no_overpass).preview(WESTBOUND)
    assert preview["coordinates"][0] == pytest.approx([147.3095, -42.8], abs=1e-6)
    assert preview["coordinates"][-1] == pytest.approx([147.3005, -42.8], abs=1e-6)
    with pytest.raises(RouterUnavailable, match="map data isn't available"):
        DevRouter(settings=_settings(tmp_path, tmp_path / "missing.sqlite"), fetch=_no_overpass).preview(WESTBOUND)


def test_the_national_fallback_takes_buildings_from_the_index(osm_index, tmp_path):
    from mapcreator.providers.base import OSM_BUILDINGS
    from mapcreator.regions import TASMANIA, australia

    box = TO_XY.transform_bounds(*QUERY)
    region = australia(55)  # MGA zone 55, the fixture's CRS
    assert len(region.buildings(box, _settings(tmp_path, osm_index))) == 1
    assert region.buildings(box, _settings(tmp_path)) == []  # no index, no buildings (and no Overpass)
    assert region.buildings_attribution == OSM_BUILDINGS and OSM_BUILDINGS.licence == "ODbL 1.0"
    assert TASMANIA.buildings_attribution.holder == "© State of Tasmania"  # Tasmania keeps theLIST's footprints
