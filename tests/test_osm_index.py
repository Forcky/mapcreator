"""The local OSM index: built from a small hand-written extract, queried like Overpass, routed on."""

import json
import shutil
import sqlite3
import xml.etree.ElementTree as ET
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
from mapcreator.providers.osm import graph, index, overpass, roads
from mapcreator.service.routing import DevRouter, RouterUnavailable

EPSG = 28355  # MGA zone 55, the fixture is near Hobart
# A block of road: Main Road runs west only along -42.800; Back Street loops round the north.
# Also: a tertiary road with a node missing from the extract, a footway and a service road
# (not drivable), a road far away, an L-shaped road whose box covers QUERY without entering it,
# a house between the roads, a shed far away, an unclosed "building" and a route relation.
# Power: an 11 kV minor line drawn with vertices 400 m apart, a 220/110 kV double-circuit line, an
# underground line (indexed, never built) and a power=cable way (not indexed).
# Railways and paths (layout 3): a footbridge over Main Road (no shared node), a railway crossing Back Street at grade
# (sharing node 5), a disused railway on a bridge over Main Road, a rail trail (a cycleway with railway=abandoned: a
# path), a platform and a tram line (not indexed), and far away a former station drawn as a closed building with
# railway=disused (a building) and a street with railway=abandoned (a road).
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
  <node id="60" lat="-42.7985000" lon="147.3000000"/>
  <node id="61" lat="-42.7985000" lon="147.3050000"><tag k="power" v="pole"/></node>
  <node id="62" lat="-42.7985000" lon="147.3100000"/>
  <node id="63" lat="-42.8030000" lon="147.3000000"><tag k="power" v="tower"/></node>
  <node id="64" lat="-42.8030000" lon="147.3045000"><tag k="power" v="tower"/></node>
  <node id="65" lat="-42.8040000" lon="147.3000000"/>
  <node id="66" lat="-42.8040000" lon="147.3050000"/>
  <node id="70" lat="-42.7995000" lon="147.3070000"/>
  <node id="71" lat="-42.8005000" lon="147.3070000"/>
  <node id="72" lat="-42.7970000" lon="147.3040000"/>
  <node id="73" lat="-42.7990000" lon="147.3062000"/>
  <node id="74" lat="-42.7990000" lon="147.3080000"/>
  <node id="75" lat="-42.8010000" lon="147.3080000"/>
  <node id="76" lat="-42.8020000" lon="147.3000000"/>
  <node id="77" lat="-42.8020000" lon="147.3100000"/>
  <node id="80" lat="-42.7995000" lon="147.4020000"/>
  <node id="81" lat="-42.7995000" lon="147.4025000"/>
  <node id="82" lat="-42.7992000" lon="147.4025000"/>
  <node id="83" lat="-42.7992000" lon="147.4020000"/>
  <node id="84" lat="-42.8050000" lon="147.4000000"/>
  <node id="85" lat="-42.8050000" lon="147.4100000"/>
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
  <way id="40">
    <nd ref="60"/><nd ref="61"/><nd ref="62"/>
    <tag k="power" v="minor_line"/><tag k="voltage" v="11000"/><tag k="operator" v="Someone"/>
  </way>
  <way id="41">
    <nd ref="63"/><nd ref="64"/>
    <tag k="power" v="line"/><tag k="voltage" v="220000;110000"/><tag k="circuits" v="2"/>
  </way>
  <way id="42">
    <nd ref="65"/><nd ref="66"/>
    <tag k="power" v="line"/><tag k="location" v="underground"/><tag k="voltage" v="66000"/>
  </way>
  <way id="43">
    <nd ref="65"/><nd ref="66"/>
    <tag k="power" v="cable"/>
  </way>
  <way id="50">
    <nd ref="70"/><nd ref="71"/>
    <tag k="highway" v="footway"/><tag k="bridge" v="yes"/><tag k="layer" v="1"/><tag k="name" v="Example Footbridge"/>
    <tag k="surface" v="concrete"/><tag k="lit" v="yes"/><tag k="wheelchair" v="yes"/>
  </way>
  <way id="51">
    <nd ref="72"/><nd ref="5"/><nd ref="73"/>
    <tag k="railway" v="rail"/><tag k="usage" v="main"/><tag k="gauge" v="1067"/><tag k="name" v="Western Line"/>
    <tag k="operator" v="Someone"/>
  </way>
  <way id="52">
    <nd ref="74"/><nd ref="75"/>
    <tag k="railway" v="disused"/><tag k="bridge" v="yes"/><tag k="layer" v="1"/><tag k="source" v="survey"/>
  </way>
  <way id="53">
    <nd ref="76"/><nd ref="77"/>
    <tag k="highway" v="cycleway"/><tag k="railway" v="abandoned"/><tag k="name" v="Rail Trail"/>
    <tag k="bicycle" v="designated"/>
  </way>
  <way id="54">
    <nd ref="72"/><nd ref="73"/>
    <tag k="railway" v="platform"/>
  </way>
  <way id="55">
    <nd ref="80"/><nd ref="81"/><nd ref="82"/><nd ref="83"/><nd ref="80"/>
    <tag k="building" v="yes"/><tag k="railway" v="disused"/><tag k="name" v="Former Station"/>
  </way>
  <way id="56">
    <nd ref="84"/><nd ref="85"/>
    <tag k="highway" v="residential"/><tag k="railway" v="abandoned"/><tag k="name" v="Old Rail Street"/>
  </way>
  <way id="57">
    <nd ref="76"/><nd ref="77"/>
    <tag k="railway" v="tram"/>
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


COUNTS = {"roads": 6, "buildings": 3, "power_lines": 3, "railways": 2, "paths": 3, "skipped": 5}


def test_build_keeps_drivable_roads_and_closed_buildings(extract, tmp_path):
    counts = index.build_index(extract, tmp_path / "out.sqlite", reader="xml")
    # Skipped: the service road, the unclosed building, the power=cable way, the platform and the tram line.
    assert counts == COUNTS
    rows = {r[0]: r for r in _ways_table(tmp_path / "out.sqlite")}
    assert {i for i, r in rows.items() if r[1] == "road"} == {10, 11, 12, 14, 15, 56}
    assert {i for i, r in rows.items() if r[1] == "building"} == {30, 31, 55}
    assert {i for i, r in rows.items() if r[1] == "power"} == {40, 41, 42}
    assert {i for i, r in rows.items() if r[1] == "railway"} == {51, 52}
    assert {i for i, r in rows.items() if r[1] == "path"} == {13, 50, 53}
    # Power lines keep what their model needs, not who runs them, and which vertices are mapped supports.
    assert json.loads(rows[40][2]) == {"power": "minor_line", "voltage": "11000", "mapcreator:supports": "-P-"}
    assert json.loads(rows[41][2])["mapcreator:supports"] == "TT"
    # Road node ids stay in way order with their coordinates; the node missing from the extract is dropped.
    assert np.frombuffer(rows[12][4], dtype="<i8").tolist() == [3, 7]
    assert np.frombuffer(rows[12][3], dtype="<f8").reshape(-1, 2).tolist() == [[147.31, -42.8], [147.315, -42.8]]
    # Buildings keep their shape tags only, and no node ids.
    assert json.loads(rows[30][2]) == {"building": "house", "building:levels": "2"}
    assert rows[30][4] is None
    meta = index.metadata(tmp_path / "out.sqlite")
    assert meta["source"] == "fixture.osm" and meta["roads"] == "6" and meta["schema"] == index.SCHEMA_VERSION == "3"
    assert meta["railways"] == "2" and meta["paths"] == "3" and meta["railway_classes"] == overpass.RAILWAYS
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
        assert index.build_index(source, out, reader="osmium") == COUNTS
        assert _ways_table(out) == expected
        assert index.metadata(out)["reader"].startswith("pyosmium ")
    assert index.metadata(tmp_path / "fixture.osm.pbf.sqlite")["osm_timestamp"] == "2026-11-01T20:21:02Z"


def test_command_line_builds_and_queries(extract, tmp_path, capsys):
    out = tmp_path / "cli.sqlite"
    assert index.main(["build", str(extract), str(out), "--reader", "xml"]) == 0
    assert "indexed 6 roads, 2 railways, 3 paths, 3 buildings and 3 power lines" in capsys.readouterr().out
    assert index.main(["query", str(out), *map(str, QUERY)]) == 0
    text = capsys.readouterr().out
    assert "roads: 3 ways, 7 nodes" in text and "buildings: 1 in EPSG:28355" in text
    assert "railways and paths: 2 railways, 3 paths" in text
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
    # For ATTRIBUTION.txt: the extract the index was built from (XML records no OSM timestamp).
    work, recorded = WorkDir(tmp_path / "work"), {"index": "fixture.osm", "osm_timestamp": ""}
    assert work.read_json(route_stage.STAGE, route_stage.OSM_SELECTION) == recorded
    assert route_stage.osm_selection(work, _settings(tmp_path)) == recorded  # what was recorded, not the settings


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
    work = WorkDir(tmp_path / "work")
    assert route_stage.osm_selection(work, _settings(tmp_path)) == {"overpass_filter": overpass.ROADS_FILTER}
    # A work folder routed before the selection was recorded gets the current settings' answer.
    (tmp_path / "work" / route_stage.STAGE / route_stage.OSM_SELECTION).unlink()
    assert route_stage.osm_selection(work, _settings(tmp_path, osm_index))["index"] == "fixture.osm"


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


def test_power_lines_come_back_in_the_region_crs_with_their_voltage_and_circuits(osm_index):
    from mapcreator.core import power

    box = TO_XY.transform_bounds(*QUERY, densify_pts=21)
    lines = {line.ref: line for line in index.power_lines(osm_index, box, EPSG)}
    assert set(lines) == {"OSM way 40", "OSM way 41"}  # not the underground line
    minor, hv = lines["OSM way 40"], lines["OSM way 41"]
    assert (minor.kind, minor.voltage_kv, minor.circuits) == (power.DISTRIBUTION, 11.0, 1)
    assert (hv.kind, hv.voltage_kv, hv.circuits) == (power.TRANSMISSION, 220.0, 2)
    # The minor line's mapped vertices are about 410 m apart, more than a distribution span can be, so
    # supports were added evenly between them; the mapped vertices stay, marked as not inferred.
    gaps = np.hypot(*np.diff(minor.xy, axis=0).T)
    assert gaps.max() <= power.MAX_SPAN_M[power.DISTRIBUTION]
    assert (~minor.inferred).sum() == 3 and minor.inferred.sum() == len(minor.xy) - 3
    x, y = TO_XY.transform(147.305, -42.7985)
    assert np.hypot(*(minor.xy[~minor.inferred][1] - (x, y))) < 0.01
    assert minor.designs[~minor.inferred].tolist() == ["", "pole", ""]
    # The tower line's two vertices are 370 m apart: a transmission span, so nothing is added.
    assert len(hv.xy) == 2 and not hv.inferred.any() and hv.designs.tolist() == ["tower", "tower"]


def test_an_index_from_before_power_lines_still_routes_but_has_no_power(osm_index, tmp_path):
    import shutil

    old = tmp_path / "old.sqlite"
    shutil.copyfile(osm_index, old)
    with closing(sqlite3.connect(old)) as con:
        con.execute("UPDATE meta SET value = '1' WHERE key = 'schema'")
        con.commit()
    assert [e["id"] for e in index.query(old, QUERY)["elements"] if e["type"] == "way"] == [10, 11, 12]
    with pytest.raises(index.IndexUnavailable, match="before power lines"):
        index.power_lines(old, TO_XY.transform_bounds(*QUERY, densify_pts=21), EPSG)
    with pytest.raises(index.IndexUnavailable, match="before railways and paths"):
        index.railways_and_paths(old, QUERY)


# ----- railways and paths (layout 3): what crosses the route on a bridge, or under the route's bridge -----

def test_railways_and_paths_come_back_as_overpass_json_with_their_node_ids(osm_index):
    data = index.railways_and_paths(osm_index, QUERY)
    ways = {e["id"]: e for e in data["elements"] if e["type"] == "way"}
    assert list(ways) == [13, 50, 51, 52, 53]  # in way id order; not the platform or the tram line
    # What they are, their names and whether they're on a bridge; not who runs them or who may use them.
    assert ways[50]["tags"] == {"highway": "footway", "bridge": "yes", "layer": "1", "name": "Example Footbridge",
                                "surface": "concrete"}
    assert ways[51]["tags"] == {"railway": "rail", "usage": "main", "gauge": "1067", "name": "Western Line"}
    assert ways[52]["tags"] == {"railway": "disused", "bridge": "yes", "layer": "1"}
    assert ways[53]["tags"] == {"highway": "cycleway", "railway": "abandoned", "name": "Rail Trail"}
    # The level crossing shares Back Street's node; the footbridge over Main Road shares none.
    assert ways[51]["nodes"] == [72, 5, 73] and ways[50]["nodes"] == [70, 71]
    nodes = {e["id"]: e for e in data["elements"] if e["type"] == "node"}
    assert set(nodes) == {2, 5, 8, 70, 71, 72, 73, 74, 75, 76, 77}
    assert nodes[5] == {"type": "node", "id": 5, "lat": -42.798, "lon": 147.305}
    # The roads stay apart, so the router is never given a path or a railway.
    assert [e["id"] for e in index.query(osm_index, QUERY)["elements"] if e["type"] == "way"] == [10, 11, 12]
    assert index.railways_and_paths(osm_index, (147.5, -42.9, 147.6, -42.8))["elements"] == []


def _overpass_answer(box, keep) -> dict:
    """What an Overpass server answers for the fixture's ways that `keep(tags)` and that cross `box`: every tag, and
    every node (tagged nodes with their tags)."""
    root = ET.fromstring(FIXTURE)
    nodes = {int(n.get("id")): n for n in root.iter("node")}
    ways = []
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
        refs = [int(nd.get("ref")) for nd in w.iter("nd")]
        points = [(float(nodes[r].get("lon")), float(nodes[r].get("lat"))) for r in refs if r in nodes]
        if keep(tags) and len(points) > 1 and shapely.intersects(shapely.LineString(points), shapely.box(*box)):
            ways.append({"type": "way", "id": int(w.get("id")), "nodes": refs, "tags": tags})
    out = []
    for r in sorted({r for w in ways for r in w["nodes"] if r in nodes}):
        node = {"type": "node", "id": r, "lat": float(nodes[r].get("lat")), "lon": float(nodes[r].get("lon"))}
        tags = {t.get("k"): t.get("v") for t in nodes[r].iter("tag")}
        out.append(node | ({"tags": tags} if tags else {}))
    return {"version": 0.6, "generator": "Overpass API (test)", "elements": out + ways}


def test_overpass_gives_the_same_railways_and_paths_as_the_index(osm_index, tmp_path):
    box = (147.28, -42.82, 147.42, -42.78)  # the whole fixture, with the far street that has railway=abandoned
    answer = _overpass_answer(box, lambda t: t.get("railway") in overpass.RAILWAYS.split("|")
                              or t.get("highway") in overpass.PATHS.split("|"))
    assert 56 in {e["id"] for e in answer["elements"]}  # Overpass's tag filter takes the street
    queries = []

    def fake_overpass(query, settings):
        queries.append(query)
        return answer

    from_overpass = roads.railways_and_paths_for_bbox(box, _settings(tmp_path), fetch=fake_overpass)
    from_index = index.railways_and_paths(osm_index, box)
    assert from_overpass["elements"] == from_index["elements"]
    # One query for both, south, west, north, east, as Overpass takes it.
    assert len(queries) == 1 and overpass.RAILWAYS_FILTER in queries[0] and overpass.PATHS_FILTER in queries[0]
    assert queries[0].count("(-42.820000,147.280000,-42.780000,147.420000)") == 2
    # With an index configured, Overpass isn't asked.
    assert roads.railways_and_paths_for_bbox(box, _settings(tmp_path, osm_index), fetch=_no_overpass) == from_index


def test_the_roads_query_to_overpass_is_unchanged():
    # Its text keys the development cache, so a change would fetch every cached box again.
    assert overpass.query_for_bbox(-42.82, 147.28, -42.78, 147.33) == (
        f"[out:json][timeout:180];\n{overpass.ROADS_FILTER}(-42.820000,147.280000,-42.780000,147.330000);\n"
        "(._;>;);\nout body;\n")


def test_an_index_from_before_railways_and_paths_still_serves_the_rest(osm_index, tmp_path, capsys):
    old = tmp_path / "layout2.sqlite"
    shutil.copyfile(osm_index, old)
    with closing(sqlite3.connect(old)) as con:  # as a layout 2 build leaves it: no railway or path rows or tables
        con.execute("UPDATE meta SET value = '2' WHERE key = 'schema'")
        con.execute("DELETE FROM ways WHERE kind IN ('railway', 'path')")
        con.commit()
        con.executescript("DROP TABLE railway_rtree; DROP TABLE path_rtree;")
    assert [e["id"] for e in index.query(old, QUERY)["elements"] if e["type"] == "way"] == [10, 11, 12]
    assert len(index.buildings(old, TO_XY.transform_bounds(*QUERY), EPSG)) == 1
    assert len(index.power_lines(old, TO_XY.transform_bounds(*QUERY, densify_pts=21), EPSG)) == 2
    with pytest.raises(index.IndexUnavailable, match="before railways and paths were indexed; rebuild it"):
        index.railways_and_paths(old, QUERY)
    with pytest.raises(roads.INDEX_ERRORS):
        roads.railways_and_paths_for_bbox(QUERY, _settings(tmp_path, old), fetch=_no_overpass)
    assert index.main(["query", str(old), *map(str, QUERY)]) == 0
    assert "railways and paths: not in this index" in capsys.readouterr().out
