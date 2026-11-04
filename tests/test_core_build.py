"""A whole build from core alone (no data sources): centreline, profile, section, meshes, KN5, AI line,
then the validator's model and layout checks, on a synthetic road over flat ground."""

import numpy as np
import pytest

from mapcreator.core import ai_line, kn5, mesh, scenery, validate
from mapcreator.core import centreline as cl
from mapcreator.core import profile as pf
from mapcreator.core import section as sec
from mapcreator.core.layout import RoadModel, logic_dummies
from mapcreator.core.power import POWER_COLLIDER
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


def build_and_check(tmp_path, tags: dict, *, reverse: bool = False, shape=gentle, trunks=(),
                    gravel: tuple[float, float] | None = None, collider: str = scenery.TRUNK_MESH) -> validate.Report:
    """`trunks`: (metres along the road, metres left of the centreline) for 0.4 m trunk colliders, named
    `collider` (a power line's are checked the same way). `gravel`: (from, to) metres along the road that are
    unsealed, as 1GRAVEL meshes with the steeper crown."""
    line = cl.smooth(shape())
    prof = pf.smooth_profile(line.s, flat(line.xy[:, 0], line.xy[:, 1]), np.zeros(len(line.s), bool))
    unsealed = np.zeros(len(line.s), bool) if gravel is None else (line.s >= gravel[0]) & (line.s <= gravel[1])
    w, _ = sec.width_from_tags(tags)
    half = sec.eased(line.s, np.full(len(line.s), w)) / 2
    crown = sec.eased(line.s, np.where(unsealed, sec.CROWN_UNSEALED, sec.CROWN))
    sl, sr = sec.half_slopes(np.zeros(len(line.s)), crown)
    style = np.where(unsealed, sec.style_code(True, w < sec.SINGLE_LANE_BELOW_M), sec.style_code(False, False))
    frame = frame_for(line.xy, prof.z)
    rb = mesh.build_ribbon(line.xy, prof.z, line.heading, line.curvature, half, half, sl, sr, flat)
    meshes = mesh.ribbon_meshes(rb, line.s, frame, style=style)
    if trunks:
        i = np.searchsorted(line.s, [s for s, _ in trunks])
        nrm = np.column_stack([-np.sin(line.heading[i]), np.cos(line.heading[i])])
        xy = line.xy[i] + nrm * np.array([off for _, off in trunks])[:, None]
        cols = scenery.trunk_colliders(xy, flat(*xy.T), np.full(len(xy), 0.4), np.zeros(len(xy)), line.s[i], frame)
        for m in cols:
            m.name = m.name.replace(scenery.TRUNK_MESH, collider)
        meshes += cols
    road = RoadModel.from_arrays(line.s, line.xy, prof.z, line.heading, line.curvature, prof.grade,
                                 half, half, sl, sr, reverse)
    dummies = [kn5.Kn5Dummy(d.name, kn5.dummy_matrix(d.position, d.forward, d.up)) for d in logic_dummies(road, frame)]
    names = sorted({m.material for m in meshes})
    materials = [kn5.Kn5Material(n, textures={"txDiffuse": "a.dds"}) for n in names]
    index = {n: i for i, n in enumerate(names)}
    km = [kn5.Kn5Mesh(m.name, index[m.material], m.positions, m.normals, m.uvs, m.triangles, renderable=m.renderable)
          for m in meshes]
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


def test_tree_colliders_must_stay_off_the_road_and_clear_of_the_grid(tmp_path):
    road = {"highway": "secondary"}  # wide enough for the grid's pairs, 2 m either side of the centreline
    half = sec.width_from_tags(road)[0] / 2
    beside = [(300.0, half + 6.0), (520.0, -(half + 6.0))]  # either side of the 400 m chunk boundary
    rep = build_and_check(tmp_path, road, trunks=beside)
    assert rep.failed == []
    assert any("2 tree collider meshes, all invisible" in p for p in rep.passed)
    assert any("tree colliders clear of the road (0 corners" in p for p in rep.passed)
    assert any("tree colliders at least 3 m from every spawn" in p for p in rep.passed)

    rep = build_and_check(tmp_path, road, trunks=beside + [(600.0, 0.0)])  # in the middle of the road
    assert [f for f in rep.failed if "tree colliders clear of the road" in f]
    # Just off the edge, beside the pair of grid spots behind spot 0 (34 m in): off the road, but too close.
    rep = build_and_check(tmp_path, road, trunks=beside + [(34.0, half + 0.6)])
    assert [f for f in rep.failed if "from every spawn" in f]
    assert not [f for f in rep.failed if "clear of the road" in f]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("tags", [{"highway": "unclassified", "lanes": "1"}, {"highway": "secondary"}])
def test_a_road_that_turns_to_gravel_validates_on_both_surfaces(tmp_path, tags, reverse):
    # Gravel for the first 500 m: the forward grid stands on gravel, the reverse grid on the seal, and the AI
    # line crosses the change. Spawn and AI heights are measured against 1ROAD and 1GRAVEL alike.
    rep = build_and_check(tmp_path, tags, reverse=reverse, gravel=(0.0, 500.0))
    assert rep.failed == []
    assert any("each physics mesh matches exactly one surface key" in p for p in rep.passed)
    assert any("AC_START_0 sits 0.30 m above the road" in p for p in rep.passed)
    assert any("AI line" in p and "above the road" in p for p in rep.passed)


def test_an_all_gravel_road_validates_and_is_seen_as_road(tmp_path):
    rep = build_and_check(tmp_path, {"highway": "unclassified", "lanes": "1"}, gravel=(0.0, 1e9))
    assert rep.failed == []
    assert not [p for p in rep.passed if "off the road" in p]


@pytest.mark.parametrize("what,collider", [("tree", scenery.TRUNK_MESH), ("power", POWER_COLLIDER)])
def test_scenery_colliders_are_checked_against_gravel_too(tmp_path, what, collider):
    road = {"highway": "unclassified", "lanes": "1"}  # 4.0 m: single file
    half = sec.width_from_tags(road, unsealed=True)[0] / 2
    beside = [(300.0, half + 6.0), (520.0, -(half + 6.0))]
    all_gravel = (0.0, 1e9)
    rep = build_and_check(tmp_path, road, trunks=beside, gravel=all_gravel, collider=collider)
    assert rep.failed == []
    assert any(f"{what} colliders clear of the road (0 corners" in p for p in rep.passed)
    rep = build_and_check(tmp_path, road, trunks=beside + [(600.0, 0.0)], gravel=all_gravel, collider=collider)
    assert [f for f in rep.failed if f"{what} colliders clear of the road" in f]  # in the middle of the gravel
    # Just off the edge beside the third single-file spot (26 m in): off the road, but too close to the spawn.
    rep = build_and_check(tmp_path, road, trunks=beside + [(26.0, half + 0.6)], gravel=all_gravel, collider=collider)
    assert [f for f in rep.failed if "from every spawn" in f]
    assert not [f for f in rep.failed if "clear of the road" in f]


def test_a_gravel_mesh_needs_gravel_in_surfaces_ini():
    from mapcreator.core import package

    names = ["1ROAD_000", "1GRAVEL_001", "1GRASS_V001L", "1WALL_TRUNK_000", "RAIL_L000"]
    assert validate.undefined_surfaces(names, package.surfaces_ini(gravel=True)) == []
    assert validate.undefined_surfaces(names, package.surfaces_ini()) == ["1GRAVEL_001"]


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


def test_a_very_long_guard_rail_is_split_under_the_vertex_limit():
    from mapcreator.core import roadside
    from mapcreator.core.scene import LocalFrame

    n = 15000  # 30 km at 2 m, with a 10 m drop along the whole left edge
    s = np.arange(n) * 2.0
    xy = np.column_stack([500000 + s, np.full(n, 5250000.0)])
    flat_road = np.zeros(n)
    meshes, stats = roadside.guard_rails(s, xy, np.full(n, 100.0), flat_road, np.full(n, 3.0), np.full(n, 3.0),
                                         flat_road, flat_road, lambda e, nn: np.where(np.asarray(nn) > 5250000, 90.0, 100.0),
                                         LocalFrame(500000, 5250000, 100))
    assert stats["L"]["runs"] == 1
    for m in meshes:
        m.validate()  # raises over 65,535 vertices
    names = [m.name for m in meshes]
    assert names[:2] == ["RAIL_L000", "1WALL_RAIL_L000"] and len(set(names)) == len(names) > 2
