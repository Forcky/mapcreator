"""The councils' own kerb lines (`providers.councils.kerb_lines`, `core.kerbs.council_sides`): the mirror file, the
GeoPackage's geometry, where they set the kerb, and building on without them."""

import io
import sqlite3
import zipfile

import numpy as np
import pytest
import shapely

from mapcreator.core import kerbs
from mapcreator.providers.councils import kerb_lines as kl
from tests.test_kerbs import plan_of, straight


def lines(kerb_y=(5.0, -5.0), edge_y=(), x0=0.0, x1=1000.0):
    out = kl.KerbLines()
    for y in kerb_y:
        out.lines.append((kl.HOBART, kl.KERB, shapely.LineString([(x0, y), (x1, y)])))
    for y in edge_y:
        out.lines.append((kl.HOBART, kl.EDGE, shapely.LineString([(x0, y), (x1, y)])))
    out.sources.add(kl.HOBART)
    return out


def test_the_mirror_round_trips_and_is_read_by_box(tmp_path):
    path = tmp_path / "kerb_lines.sqlite"
    kl.write_mirror(path, {kl.HOBART: [(kl.KERB, shapely.LineString([(0, 0), (10, 0)])),
                                       (kl.EDGE, shapely.LineString([(500, 0), (510, 0)]))],
                           kl.GLENORCHY: [(kl.KERB, shapely.LineString([(0, 5), (10, 5)]))]}, {})
    got = kl.read(path, (-1, -1, 20, 20))
    assert sorted((s, k) for s, k, _ in got.lines) == [(kl.GLENORCHY, kl.KERB), (kl.HOBART, kl.KERB)]
    assert {a.licence for a in got.attributions()} == {"CC BY 4.0", "CC BY 3.0 AU"}
    assert not kl.read(path, (1000, 1000, 1001, 1001)).lines
    assert not path.with_suffix(".sqlite.part").exists()


def test_a_mirror_of_another_format_is_refused(tmp_path):
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE meta (key TEXT, value TEXT)")
    con.execute("INSERT INTO meta VALUES ('format_version', '0')")
    con.commit()
    con.close()
    with pytest.raises(ValueError):
        kl.read(path, (0, 0, 1, 1))


def test_glenorchy_s_geopackage_geometry_is_read_past_its_header(tmp_path):
    gpkg = tmp_path / "Kerbs.gpkg"
    con = sqlite3.connect(gpkg)
    con.execute("CREATE TABLE gpkg_geometry_columns (table_name TEXT, column_name TEXT)")
    con.execute("INSERT INTO gpkg_geometry_columns VALUES ('Kerbs', 'geom')")
    con.execute('CREATE TABLE "Kerbs" (id INTEGER, geom BLOB)')
    wkb = shapely.to_wkb(shapely.LineString([(1, 2), (3, 4)]))
    header = b"GP" + bytes([0, 0b0000_0011]) + (28355).to_bytes(4, "little") + bytes(32)  # an xy envelope
    con.execute('INSERT INTO "Kerbs" VALUES (1, ?)', (header + wkb,))
    con.commit()
    con.close()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.write(gpkg, "Kerbs.gpkg")
    (kind, line), = kl.glenorchy_lines(buf.getvalue())
    assert kind == kl.KERB and list(line.coords) == [(1.0, 2.0), (3.0, 4.0)]


def test_the_councils_areas():
    assert kl.areas_for((526000, 5251000, 527000, 5252000)) == [kl.HOBART]  # Hobart's CBD
    assert kl.GLENORCHY in kl.areas_for((518000, 5258000, 519000, 5259000))
    assert not kl.areas_for((535000, 5250000, 536000, 5251000))  # Bellerive, across the river: Clarence


def test_council_kerb_lines_set_the_kerb_faces_even_with_no_buildings():
    p = plan_of(centroids=np.zeros((0, 2)), council=lambda boxes: lines(kerb_y=(5.2, -4.4)))
    assert p.any()
    assert p.half["L"][500] + kerbs.CHANNEL_M == pytest.approx(5.2, abs=0.05)
    assert p.half["R"][500] + kerbs.CHANNEL_M == pytest.approx(4.4, abs=0.05)  # unequal halves, as the street is
    assert p.summary["council_m"]["L"] > 900


def test_a_council_s_edge_of_formation_keeps_that_side_bare_in_a_town():
    p = plan_of(council=lambda boxes: lines(kerb_y=(5.0,), edge_y=(-4.0,)))
    assert p.kerb["L"][500] == 1.0 and p.kerb["R"][500] == 0.0
    p = plan_of(council=lambda boxes: lines(kerb_y=(), edge_y=(4.0, -4.0)))
    assert not p.any()  # the council says the road is unkerbed, whatever the buildings say


def test_without_council_lines_the_rule_kerbs_the_town_as_in_phase_1():
    rule = plan_of()
    for council in (None, lambda boxes: None, lambda boxes: kl.KerbLines()):
        p = plan_of(council=council)
        assert np.array_equal(p.kerb["L"], rule.kerb["L"]) and np.allclose(p.half["L"], rule.half["L"])


def test_council_lines_far_from_the_road_or_across_it_aren_t_its_kerb():
    s, xy, heading = straight(11)
    far = kerbs.council_sides(xy, heading, np.arange(11), [shapely.LineString([(0, 20), (10, 20)])], [])
    assert np.isnan(far["L"][0]).all()
    across = kerbs.council_sides(xy, heading, np.arange(11), [shapely.LineString([(5, -3), (5, 3)])], [])
    assert np.isnan(across["L"][0]).all()  # a kerb line square to the road (a crossing's ramp) isn't its edge


def test_without_the_mirror_a_road_in_a_council_s_area_gets_a_warning_and_the_rule(tmp_path, caplog):
    from mapcreator import config
    from mapcreator.job import Job
    from mapcreator.pipeline import route_stage, section_stage
    from mapcreator.pipeline.workdir import WorkDir

    n = 1001
    s = np.arange(n, dtype=float)
    xy = np.column_stack([526000.0 + s, np.full(n, 5251000.0)])  # in Hobart's area
    houses = [shapely.box(526000 + x, 5251015, 526008 + x, 5251025) for x in np.arange(300, 700, 12)]

    class Region:
        buildings = staticmethod(lambda bbox, st: houses)
        parcels = None
        buildings_attribution = parcels_attribution = None
        kerb_lines = staticmethod(lambda boxes, st: None)  # no mirror configured

    work = WorkDir(tmp_path)
    np.savez_compressed(work.stage(route_stage.STAGE) / "route.npz", xy=xy)
    job = Job.model_validate({"track_id": "t", "junctions": "none",
                              "waypoints": [{"lon": 147.3, "lat": -42.8}, {"lon": 147.31, "lat": -42.8}]})
    c = {"s": s, "xy": xy, "heading": np.zeros(n), "run": np.zeros(n, int)}
    p = section_stage._kerbs(job, config.load(), Region, work, c, [{"tags": {"highway": "residential"}}],
                             np.full(n, 3.0), np.zeros(n, bool))
    assert p.any() and not p.council and p.summary["council_lines"] == "not_configured"
    assert "no councils' kerb lines configured" in caplog.text
