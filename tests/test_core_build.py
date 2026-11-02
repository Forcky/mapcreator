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
