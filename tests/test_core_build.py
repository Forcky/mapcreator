"""A whole build from core alone (no data sources): centreline, profile, section, meshes, KN5, AI line,
then the validator's model and layout checks, on a synthetic road over flat ground."""

import numpy as np
import pytest

from mapcreator.core import ai_line, kn5, mesh, validate
from mapcreator.core import centreline as cl
from mapcreator.core import profile as pf
from mapcreator.core import section as sec
from mapcreator.core.layout import RoadModel, logic_dummies
from mapcreator.pipeline.mesh_stage import frame_for


def flat(e, n):
    return 100.0 + 0.0 * np.asarray(e, float)


def gentle() -> np.ndarray:
    t = np.linspace(0, 1000, 60)
    return np.column_stack([500000 + t, 5250000 + 20 * np.sin(t / 150)])


def hairpin_first() -> np.ndarray:
    # A 15 m radius hairpin right at the start, then 400 m straight back.
    th = np.linspace(0, np.pi, 40)
    arc = np.column_stack([15 * np.sin(th), 15 * (1 - np.cos(th))])
    leg = np.column_stack([-np.arange(10, 410, 10.0), np.full(40, 30.0)])
    return np.vstack([arc, leg]) + [500000.0, 5250000.0]


def build_and_check(tmp_path, tags: dict, *, reverse: bool = False, shape=gentle) -> validate.Report:
    line = cl.smooth(shape())
    prof = pf.smooth_profile(line.s, flat(line.xy[:, 0], line.xy[:, 1]), np.zeros(len(line.s), bool))
    w, _ = sec.width_from_tags(tags)
    half = sec.eased(line.s, np.full(len(line.s), w)) / 2
    sl, sr = sec.half_slopes(np.zeros(len(line.s)))
    frame = frame_for(line.xy, prof.z)
    rb = mesh.build_ribbon(line.xy, prof.z, line.heading, line.curvature, half, half, sl, sr, flat)
    meshes = mesh.ribbon_meshes(rb, line.s, frame)
    road = RoadModel.from_arrays(line.s, line.xy, prof.z, line.heading, line.curvature, prof.grade,
                                 half, half, sl, sr, reverse)
    dummies = [kn5.Kn5Dummy(d.name, kn5.dummy_matrix(d.position, d.forward, d.up)) for d in logic_dummies(road, frame)]
    materials = [kn5.Kn5Material(n, textures={"txDiffuse": "a.dds"}) for n in ("road", "grass")]
    index = {"road": 0, "grass": 1}
    km = [kn5.Kn5Mesh(m.name, index[m.material], m.positions, m.normals, m.uvs, m.triangles) for m in meshes]
    model = tmp_path / "t.kn5"
    kn5.write(model, root="t", textures={"a.dds": b"DDS x"}, materials=materials, meshes=km, dummies=dummies)
    ai = tmp_path / "fast_lane.ai"
    ai_line.write(ai, ai_line.build(road, frame))
    rep = validate.Report()
    data, by_name = validate._kn5_checks(rep, model)
    validate._layout_checks(rep, "t", data, by_name, ai)
    return rep


@pytest.mark.parametrize("tags", [{"highway": "secondary"}, {"highway": "unclassified", "lanes": "1"},
                                  {"highway": "service", "width": "3"}])
@pytest.mark.parametrize("reverse", [False, True])
def test_a_synthetic_road_of_any_width_validates(tmp_path, tags, reverse):
    rep = build_and_check(tmp_path, tags, reverse=reverse)
    assert rep.failed == [] and rep.passed


@pytest.mark.parametrize("reverse", [False, True])
def test_a_road_that_starts_or_ends_in_a_hairpin_validates(tmp_path, reverse):
    rep = build_and_check(tmp_path, {"highway": "secondary"}, reverse=reverse, shape=hairpin_first)
    assert rep.failed == []


def test_skirts_stop_halfway_to_a_parallel_leg():
    # A switchback: east along y=0, a tight turn, then west along y=16. Each leg's skirt toward the other
    # may reach at most halfway (8 m), so neither lays grass over the other's carriageway.
    east = np.column_stack([np.arange(0, 301, 10.0), np.zeros(31)])
    th = np.linspace(-np.pi / 2, np.pi / 2, 25)
    turn = np.column_stack([300 + 8 * np.cos(th), 8 + 8 * np.sin(th)])
    west = np.column_stack([np.arange(290, -1, -10.0), np.full(30, 16.0)])
    line = cl.smooth(np.vstack([east, turn[1:-1], west]) + [500000.0, 5250000.0])
    z = np.full(len(line.s), 100.0)
    half = np.full(len(line.s), 3.0)
    sl, sr = sec.half_slopes(np.zeros(len(line.s)))
    rb = mesh.build_ribbon(line.xy, z, line.heading, line.curvature, half, half, sl, sr, flat)
    on_east_leg = (line.xy[:, 0] > 500050) & (line.xy[:, 0] < 500250) & (line.xy[:, 1] < 5250004)
    reach_north = rb.n[on_east_leg, 0] - line.xy[on_east_leg, 1]  # column 0 is the outermost on the left
    assert on_east_leg.sum() > 20 and reach_north.max() <= 8.05  # smoothing leaves the legs ~16 m apart
    # Away from any other leg the skirt still reaches its full width.
    lone = mesh.build_ribbon(line.xy[:40], z[:40], line.heading[:40], line.curvature[:40], half[:40], half[:40],
                             sl[:40], sr[:40], flat)
    assert (lone.n[5:30, 0] - line.xy[5:30, 1]).max() > 12.0
