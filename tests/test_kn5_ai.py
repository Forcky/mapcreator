import io
import struct

import numpy as np
import pytest
from PIL import Image

from mapcreator.core import ai_line, kn5, package, textures
from mapcreator.core.layout import RoadModel, layouts_for, logic_dummies
from mapcreator.core.scene import LocalFrame

FRAME = LocalFrame(500_000.0, 5_200_000.0, 10.0)


def _quad(name="1ROAD_000", material=0):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], dtype=np.float32)
    nrm = np.tile([0, 1, 0], (4, 1)).astype(np.float32)
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    tri = np.array([[0, 2, 1], [1, 2, 3]])
    return kn5.Kn5Mesh(name, material, pos, nrm, uv, tri)


def _write(tmp_path, meshes=None, dummies=None, textures_=None, materials=None):
    path = tmp_path / "t.kn5"
    kn5.write(path, root="t", textures=textures_ or {"a.dds": b"DDS fake"},
              materials=materials or [kn5.Kn5Material("road", properties={"ksAmbient": 0.4, "ksEmissive": (0, 0, 0)},
                                                      textures={"txDiffuse": "a.dds"})],
              meshes=meshes or [_quad()], dummies=dummies or [])
    return path


def test_kn5_round_trip(tmp_path):
    m = kn5.dummy_matrix([1, 2, 3], forward=[0, 0, 1])
    path = _write(tmp_path, dummies=[kn5.Kn5Dummy("AC_PIT_0", m)])
    data = kn5.read(path)
    assert data["version"] == 6
    assert data["textures"] == {"a.dds": b"DDS fake"}
    mat = data["materials"][0]
    assert mat["shader"] == "ksPerPixel" and mat["textures"] == {"txDiffuse": "a.dds"}
    assert mat["properties"]["ksAmbient"][0] == pytest.approx(0.4)
    root, mesh, dummy = data["nodes"]
    assert root["class"] == 1 and root["children"] == 2
    assert mesh["name"] == "1ROAD_000" and len(mesh["vertices"]) == 4 and mesh["children"] == 0
    assert mesh["indices"].tolist() == [0, 2, 1, 1, 2, 3]
    assert dummy["matrix"][3, :3] == pytest.approx([1, 2, 3])


def test_vertex_layout_is_44_bytes_pos_normal_uv_tangent(tmp_path):
    data = kn5.read(_write(tmp_path))
    v = data["nodes"][1]["vertices"]
    assert v.dtype.itemsize == 44
    assert v["uv"][3] == pytest.approx([1, 1])
    assert np.allclose(np.linalg.norm(v["tangent"], axis=1), 1)


def test_writer_refuses_known_crash_causes(tmp_path):
    with pytest.raises(ValueError, match="duplicate node"):
        _write(tmp_path, meshes=[_quad("1ROAD_A"), _quad("1ROAD_A")])
    with pytest.raises(ValueError, match="missing texture"):
        _write(tmp_path, materials=[kn5.Kn5Material("m", textures={"txDiffuse": "nope.dds"})])
    big = _quad()
    big.positions = np.zeros((70_000, 3), np.float32)
    with pytest.raises(ValueError, match="vertices"):
        _write(tmp_path, meshes=[big])
    with pytest.raises(ValueError, match="ASCII"):
        _write(tmp_path, meshes=[_quad("1ROAD_é")])


def test_dummy_matrix_is_a_proper_rotation_with_z_forward():
    m = kn5.dummy_matrix([0, 0, 0], forward=[1, 0.2, 0])
    r = m[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3)) and np.linalg.det(r) == pytest.approx(1)
    assert r[1] == pytest.approx([0, 1, 0])
    assert r[2] == pytest.approx([1, 0, 0])


def _road(n=600, reverse=False):
    s = np.arange(n, dtype=float)
    radius = 40.0
    heading = np.where(s < 300, 0.0, (s - 300) / radius)  # straight, then a left-hand arc
    e = np.cumsum(np.cos(heading)) + 500_000
    nn = np.cumsum(np.sin(heading)) + 5_200_000
    curvature = np.where(s < 300, 0.0, 1 / radius)
    z = 20 + 0.03 * s
    half = np.full(n, 3.2)
    slope = np.full(n, 0.025)
    return RoadModel.from_arrays(s, np.column_stack([e, nn]), z, heading, curvature, np.full(n, 0.03),
                                 half, half, slope, slope, reverse)


def test_dummies_left_is_left_and_spawn_is_behind_the_gate():
    road = _road()
    d = {x.name: x for x in logic_dummies(road, FRAME)}
    # Driving east along the straight: left is north, which is -Z in AC axes.
    assert d["AC_AB_START_L"].position[2] < d["AC_AB_START_R"].position[2]
    assert d["AC_START_0"].position[0] < d["AC_AB_START_L"].position[0]
    assert d["AC_HOTLAP_START_0"].position[0] < d["AC_START_0"].position[0]
    assert d["AC_START_0"].forward == pytest.approx([1, 0, 0])


def test_reverse_layout_runs_the_other_way():
    fwd, rev = _road(), _road(reverse=True)
    assert rev.length == pytest.approx(fwd.length)
    assert rev.e[0] == pytest.approx(fwd.e[-1])
    assert rev.curvature[0] == pytest.approx(-fwd.curvature[-1])  # left-hander becomes a right-hander
    assert rev.grade[0] == pytest.approx(-0.03)
    assert [lay.key for lay in layouts_for("both")] == ["forward", "reverse"]
    assert layouts_for("forward")[0].key == ""


def test_ai_line_brakes_before_the_bend_and_round_trips(tmp_path):
    line = ai_line.build(_road(), FRAME)
    v = line.extra[:, 0]
    assert v[150] > v[400]  # straight faster than a 40 m bend
    assert v[400] == pytest.approx(np.sqrt(ai_line.MU * ai_line.G * 40), rel=0.05)
    assert v[280] < v[200]  # already braking before the bend starts at 300 m
    assert np.all(line.pos[:, 1] > 0)
    path = tmp_path / "ai" / "fast_lane.ai"
    ai_line.write(path, line)
    n = len(line.pos)
    assert path.stat().st_size == 16 + 20 * n + 4 + 72 * n + 4
    assert struct.unpack_from("<4i", path.read_bytes())[:2] == (7, n)
    back = ai_line.read(path)
    assert back.pos == pytest.approx(line.pos, abs=1e-3)
    assert back.extra[:, 0] == pytest.approx(v, rel=1e-6)


def test_dds_header_and_mip_chain():
    dds = textures.to_dds_dxt1(textures.asphalt(64))
    assert dds[:4] == b"DDS " and dds[84:88] == b"DXT1"
    h, w, _linear, _depth, mips = struct.unpack_from("<5I", dds, 12)
    assert (w, h, mips) == (64, 64, 5)  # 64, 32, 16, 8, 4
    assert len(dds) == 128 + 8 * (256 + 64 + 16 + 4 + 1)
    im = Image.open(io.BytesIO(dds))
    im.load()
    assert im.size == (64, 64)


def test_textures_tile_seamlessly():
    a = np.asarray(textures.grass(128)).astype(float)
    edge = np.abs(a[0] - a[-1]).mean()
    interior = np.abs(np.diff(a, axis=0)).mean()
    assert edge < 3 * interior


def test_map_ini_maps_world_to_pixels():
    xz = np.array([[-500.0, 200.0], [300.0, -900.0], [800.0, 50.0]])
    im, text = package.map_files(xz)
    vals = dict(line.split("=") for line in text.split("\r\n") if "=" in line)
    sf, xo, zo = float(vals["SCALE_FACTOR"]), float(vals["X_OFFSET"]), float(vals["Z_OFFSET"])
    px = (xz + [xo, zo]) / sf
    assert px.min() == pytest.approx(package.MAP_MARGIN, abs=1)
    assert int(vals["WIDTH"]) == im.size[0] and int(vals["HEIGHT"]) == im.size[1]
    assert (px[:, 0] <= im.size[0]).all() and (px[:, 1] <= im.size[1]).all()


@pytest.mark.parametrize("heading,label", [(0.0, "east"), (np.pi / 2, "north"), (np.pi / 4, "north-east"),
                                           (-np.pi / 4, "south-east"), (np.pi, "west")])
def test_compass(heading, label):
    assert package.compass(heading) == label


def test_the_finish_leaves_a_run_off_and_the_ai_slows_over_it():
    from mapcreator.core import layout, package

    assert layout.finish_distance(12_800.0) == 12_800.0 - layout.RUNOFF_M
    assert layout.finish_distance(1_000.0) == 1_000.0 - 150.0  # a short route gives up a share, not 250 m
    assert layout.finish_distance(200.0) == 200.0 - layout.GATE_FROM_END_M  # never less than before
    # The AI eases off after the finish and is all but stopped where its line ends, short of the end wall.
    length = 12_800.0
    finish = layout.finish_distance(length)
    sections = ai_line.runoff_hints(length, finish)
    assert [pace for *_, pace in sections] == [0.5, 0.2, 0.05]
    assert all(a[1] == b[0] for a, b in zip(sections, sections[1:])) and sections[-1][1] == 1.0
    last = length - layout.AI_END_SHORT_M
    assert sections[0][0] == pytest.approx((finish - ai_line.END_INSET_M) / (last - ai_line.END_INSET_M))
    assert sections[1][0] == pytest.approx((last - 150 - ai_line.END_INSET_M) / (last - ai_line.END_INSET_M))
    text = package.ai_hints_ini(sections)
    assert text.count("[HINT_") == 3 and "[HINT_2]\r\nSTART=" in text and text.rstrip().endswith("VALUE=0.05")
    # A short route's run-off keeps only the sections that fit, still ending at the end of the line.
    short = ai_line.runoff_hints(300.0, layout.finish_distance(300.0))
    assert short and short[-1][1] == 1.0 and all(a < b for a, b, _ in short)


def test_the_ai_line_stops_short_of_the_end_wall():
    s = np.arange(0.0, 1000.0)
    flat = np.zeros(s.size)
    road = RoadModel.from_arrays(s, np.column_stack([s, flat]), flat, flat, flat, flat, np.full(s.size, 3.0),
                                 np.full(s.size, 3.0), flat, flat, False)
    from mapcreator.core import layout

    line = ai_line.build(road, LocalFrame(0.0, 0.0, 0.0), end_short_m=layout.AI_END_SHORT_M)
    assert line.pos[0][0] == pytest.approx(ai_line.END_INSET_M)
    assert line.pos[-1][0] <= 999.0 - layout.AI_END_SHORT_M + 1.0
