"""Power lines: which supports and spans are built, the meshes, wires that stay clear, and TasNetworks' layers."""

import numpy as np
import pytest
from pyproj import Transformer

from mapcreator.core import package, power
from mapcreator.core.scene import MAX_VERTICES, LocalFrame
from mapcreator.providers.tas_list import power as tas_power

ROAD = np.column_stack([np.arange(0.0, 1001.0, 10.0), np.zeros(101)])  # 1 km east along y = 0
HALF = 3.2
FRAME = LocalFrame(0.0, 0.0, 0.0)


def flat(h=100.0):
    return lambda x, y: np.full(np.shape(x), h, dtype=float)


def crossing(kv=220.0, circuits=2):
    """A tower line crossing the road at x = 500, the outer two towers beyond SUPPORT_REACH_M."""
    return power.Line(np.array([[500.0, -800.0], [500.0, -200.0], [500.0, 200.0], [500.0, 800.0]]),
                      power.TRANSMISSION, kv, circuits, "hv")


def roadside(y=9.0):
    """A distribution line beside the road, a pole every 70 m."""
    x = np.arange(20.0, 990.0, 70.0)
    return power.Line(np.column_stack([x, np.full(len(x), y)]), power.DISTRIBUTION, 11.0, 1, "lv")


@pytest.mark.parametrize("value,kv", [("220000;110000", 220.0), ("11000", 11.0), ("WA-LF 1 220 kV Transmission Circuit", 220.0),
                                      (None, None), ("medium", None)])
def test_voltages_parse_from_osm_tags_and_line_names(value, kv):
    assert power.parse_kv(value) == kv


def test_a_sparsely_drawn_line_gets_supports_at_span_length_and_says_so():
    line = power.densify(power.Line(np.array([[0.0, 0.0], [400.0, 0.0], [450.0, 0.0]]), power.DISTRIBUTION))
    gaps = np.hypot(*np.diff(line.xy, axis=0).T)
    assert gaps.max() <= power.MAX_SPAN_M[power.DISTRIBUTION]
    assert line.inferred.tolist() == [False] + [True] * (len(line.xy) - 3) + [False, False]
    hv = power.densify(power.Line(np.array([[0.0, 0.0], [400.0, 0.0]]), power.TRANSMISSION))
    assert len(hv.xy) == 2  # a transmission span can be 400 m


def test_the_plan_keeps_spans_in_view_and_merges_shared_supports():
    branch = power.Line(np.array([[500.0, 200.0], [900.0, 250.0]]), power.TRANSMISSION, 110.0, 1, "branch")
    p = power.plan([crossing(), branch], ROAD, HALF)
    # The crossing's outer towers are 800 m away, beyond SUPPORT_REACH_M; its middle span crosses the road.
    assert [(p.supports[s.a].xy[1], p.supports[s.b].xy[1], s.ref) for s in p.spans if s.ref == "hv"] == [(-200.0, 200.0, "hv")]
    shared = [s for s in p.supports if tuple(s.xy) == (500.0, 200.0)]
    assert len(shared) == 1 and shared[0].template == "tower_220"  # the bigger tower stands where lines meet
    assert p.summary() == {"lines": 2, "towers": 3, "poles": 0, "inferred_supports": 0, "spans": 2, "km": 0.8,
                                "misplaced_supports": 0}


def test_a_support_on_the_road_is_dropped_with_its_spans():
    line = power.Line(np.array([[100.0, 9.0], [170.0, 1.0], [240.0, 9.0], [310.0, 9.0]]), power.DISTRIBUTION)
    p = power.plan([line], ROAD, HALF)
    assert p.misplaced == 1
    assert [(p.supports[s.a].xy[0], p.supports[s.b].xy[0]) for s in p.spans] == [(240.0, 310.0)]


def test_meshes_stand_on_the_ground_with_wires_clear_of_it_and_colliders_by_the_road():
    p = power.plan([crossing(), roadside()], ROAD, HALF)
    meshes, info = power.meshes(p, flat(), FRAME, HALF)
    by = {}
    for m in meshes:
        by.setdefault(m.name.rsplit("_", 1)[0], []).append(m)
        assert len(m.positions) <= MAX_VERTICES
        m.validate()
    assert set(by) == {"POWER_TOWERS", "POWER_POLES", "POWER_WIRES", "1WALL_POWER"}
    assert all(not m.renderable for m in by["1WALL_POWER"]) and all(m.renderable for m in by["POWER_WIRES"])
    towers = np.vstack([m.positions for m in by["POWER_TOWERS"]])
    assert towers[:, 1].max() == pytest.approx(100.0 + 42.0, abs=0.2)  # a 220 kV tower's earth peak
    assert towers[:, 1].min() == pytest.approx(99.0, abs=0.2)  # its legs go a metre into the ground
    wires = np.vstack([m.positions for m in by["POWER_WIRES"]])
    assert wires[:, 1].min() >= 100.0 + power.CLEARANCE_M[power.DISTRIBUTION] - 0.1
    # The crossing's wires stay high over the road: 6 phases and an earth wire, sagging between its towers.
    over_road = wires[np.abs(wires[:, 2]) < 5.0]
    assert over_road[:, 1].min() >= 100.0 + power.CLEARANCE_M[power.TRANSMISSION] - 0.1
    # Every roadside pole is within reach of a car (9 m off the centreline); the towers 200 m away aren't.
    assert info == {"colliders": 14, "low_spans": 0, "unplaced_supports": 0}
    # Outward-facing: each triangle's normal agrees with its vertices' normals.
    for m in meshes:
        p0, p1, p2 = (m.positions[m.triangles[:, k]].astype(float) for k in range(3))
        face = np.cross(p1 - p0, p2 - p0)
        agree = (face * m.normals[m.triangles].mean(axis=1)).sum(axis=1) >= 0
        assert agree.mean() > 0.99, m.name


def test_wires_sag_less_where_the_ground_rises_under_the_span_and_report_when_they_cant_clear_it():
    line = power.Line(np.array([[300.0, 9.0], [370.0, 9.0]]), power.DISTRIBUTION)
    p = power.plan([line], ROAD, HALF)

    def hump(height):
        return lambda x, y: 100.0 + height * np.exp(-((np.asarray(x) - 335.0) / 8.0) ** 2)

    for height, low in ((4.0, 0), (9.0, 1)):
        meshes, info = power.meshes(p, hump(height), FRAME, HALF)
        assert info["low_spans"] == low
        wires = np.vstack([m.positions for m in meshes if m.name.startswith("POWER_WIRES")])
        mid = wires[np.abs(wires[:, 0] - 335.0) < 6.0]  # samples are 10 m apart
        if not low:
            assert mid[:, 1].min() >= 100.0 + height + power.CLEARANCE_M[power.DISTRIBUTION] - 0.15


def test_power_geojson_groups_the_spans_by_source_line():
    p = power.plan([crossing(), roadside()], ROAD, HALF)
    to_lonlat = Transformer.from_crs(28355, 4326, always_xy=True)
    shifted = power.Plan([power.Support(s.xy + (530000.0, 5260000.0), s.design, s.template, s.along, s.distance_m)
                          for s in p.supports], p.spans)
    doc = power.geojson(shifted, to_lonlat, "ODbL-1.0", "Power lines: © OpenStreetMap contributors")
    feats = {f["properties"]["source"]: f for f in doc["features"]}
    assert set(feats) == {"hv", "lv"}
    assert len(feats["lv"]["geometry"]["coordinates"]) == 13  # 14 poles, 13 spans
    lon, lat = feats["hv"]["geometry"]["coordinates"][0][0]
    assert 147.0 < lon < 147.6 and -43.0 < lat < -42.6
    assert feats["lv"]["properties"]["license"] == "ODbL-1.0"


def test_tasnetworks_route_lines_give_the_spans_and_circuit_lines_their_voltage():
    towers = [{"geometry": {"x": x, "y": 0.0}, "attributes": {"LOCATION_ID": str(i)}}
              for i, x in enumerate((0.0, 400.0, 800.0, 1200.0))]
    route = {"attributes": {"LINE_NAME": "Here - There"},
             "geometry": {"paths": [[[0.0, 0.0], [400.0, 0.0], [800.0, 0.0], [1200.0, 0.0]]]}}
    circuits = [{"attributes": {"LINE_NAME": f"HT {k} 220 kV Transmission Circuit"},
                 "geometry": {"paths": [[[0.0, 15.0 * k], [1200.0, 15.0 * k]]]}} for k in (1, 2)]
    elsewhere = {"attributes": {"LINE_NAME": "AB 110 kV Transmission Circuit"},
                 "geometry": {"paths": [[[0.0, 5000.0], [1200.0, 5000.0]]]}}
    lines = tas_power.from_features(towers, [route, *circuits, elsewhere], (-10.0, -100.0, 900.0, 100.0))
    assert len(lines) == 1
    line = lines[0]
    assert line.xy.tolist() == [[0.0, 0.0], [400.0, 0.0], [800.0, 0.0]]  # cut to the towers in the box
    assert (line.kind, line.voltage_kv, line.circuits, line.ref) == (power.TRANSMISSION, 220.0, 2,
                                                                     "TasNetworks: Here - There")


def test_csp_sways_the_wires():
    text = package.ext_config_ini(grass_materials=["grass"], road_materials=["road"], version="t",
                                  cable_materials=["power_cable"])
    assert "[WIND]\r\nDYNAMIC_CABLES=material:power_cable\r\nDYNAMIC_CABLES_THRESHOLD=0.1" in text
    assert "WIND" not in package.ext_config_ini(grass_materials=["grass"], road_materials=["road"], version="t")
