"""Side-road stubs: the plan (where a stub starts, its width, look and heights, its flares sized by the angle, the
mouth, overlaps and timing gates), the meshes (the stub's ribbon, the throat joined to the road's edge and the stub's
first row without a crack, the flares' verges, the road's verge cut and its edge line broken across the mouth), the
terrain fitted round them, the placement functions, validation, and `junctions: none`."""

import json
import math

import numpy as np
import pytest
import shapely

from mapcreator import config
from mapcreator.core import ai_line, junctions, mesh, section, stubs, textures, validate
from mapcreator.core.layout import RoadModel
from mapcreator.core.scene import LocalFrame
from mapcreator.job import Job
from mapcreator.pipeline import route_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir

FRAME = LocalFrame(0.0, 0.0, 0.0)
HALF = 3.2
CROWN = section.CROWN
GROUND = 10.0


def road(n=1001, z=GROUND, unsealed=False):
    """A straight road east along y = 0, a station every metre, level at `z`, 6.4 m wide with a crown."""
    s = np.arange(n, dtype=float)
    c = {"s": s, "xy": np.column_stack([s, np.zeros(n)]), "heading": np.zeros(n), "curvature": np.zeros(n),
         "z": np.full(n, z, float), "grade": np.zeros(n)}
    sec = {"half_left": np.full(n, HALF), "half_right": np.full(n, HALF), "slope_left": np.full(n, CROWN),
           "slope_right": np.full(n, CROWN), "style": np.full(n, 2 if unsealed else 0, np.int8)}
    return c, sec


def junction(x, deg, *, jid=0, length=100.0, tags=None, highway="residential", closure="minor"):
    a = np.radians(deg)
    line = [[x + d * np.cos(a), d * np.sin(a)] for d in np.linspace(0, length, 21)]
    return {"id": jid, "at_m": x, "xy": [x, 0.0], "side": "left" if deg > 0 else "right", "angle_deg": deg,
            "line": line, "length_m": length, "ends": "length", "highway": highway, "closure": closure,
            "name": f"Side Road {jid}", "tags": {"highway": highway, **(tags or {})}, "width_m": 5.8}


def flat(h=GROUND):
    return lambda x, y: np.full(np.shape(x), h, float)


def plan_one(j, c=None, sec=None, dem=None, **kw):
    if c is None:
        c, sec = road()
    p = stubs.plan([j], c, sec, dem or flat(), **kw)
    return p, (p["stubs"][0] if p["stubs"] else None)


# -- the plan ----------------------------------------------------------------------------------------------------------

def test_a_square_stub_starts_at_the_road_s_edge_with_a_flare_of_8_m_either_side():
    p, st = plan_one(junction(300.0, 90.0))
    rows = st["rows"]
    # d = 0 is where its centreline crosses the road's edge, and its rows run a metre apart out to the end.
    assert rows["xy"][0] == pytest.approx([300.0, HALF], abs=0.05)
    assert rows["d"][:3] == [0, 1, 2] and st["length_m"] == pytest.approx(100.0 - HALF, abs=2.0)
    assert (st["side"], st["surface"], st["style"], st["width_m"]) == ("left", "sealed", "road", 5.8)
    for name in ("upstream", "downstream"):
        f = st["flares"][name]
        # Tangent points fall on the road's rows, a metre apart, so the corner can be a little larger than 8 m.
        assert f["radius_m"] == pytest.approx(stubs.FLARE_RADIUS_M, abs=1.0)
        assert f["tangent_m"] == pytest.approx(f["radius_m"], abs=0.05)  # a square corner: tangent = radius
        assert f["angle_deg"] == pytest.approx(90.0, abs=0.5)
    # The mouth: the road's rows between the flares' tangent points, the side road's width plus 8 m either side.
    a, b = st["mouth"]["rows"]
    assert a == pytest.approx(300 - 2.9 - 8, abs=1.0) and b == pytest.approx(300 + 2.9 + 8, abs=1.0)
    assert st["ribbon_from_m"] == pytest.approx(8.0, abs=1.0)
    # Each flare runs from the road's edge to the stub's, beyond the road everywhere.
    for f in st["flares"].values():
        pts = np.asarray(f["points"])
        assert pts[0, 1] == pytest.approx(HALF) and (pts[:, 1] >= HALF - 1e-6).all()
        assert np.hypot(*np.diff(pts[:, :2], axis=0).T).max() <= stubs.FLARE_SAMPLE_M + 1e-6
    assert p["skipped"] == []


def test_an_acute_stub_has_the_longer_flare_on_its_acute_side():
    _, st = plan_one(junction(300.0, -45.0))  # to the right, pointing ahead: the acute corner is downstream
    up, down = st["flares"]["upstream"], st["flares"]["downstream"]
    assert down["angle_deg"] == pytest.approx(45.0, abs=1.0) and up["angle_deg"] == pytest.approx(135.0, abs=1.0)
    assert down["tangent_m"] == pytest.approx(8.0 / math.tan(math.radians(22.5)), abs=1.0)
    assert up["tangent_m"] == pytest.approx(8.0 / math.tan(math.radians(67.5)), abs=1.0)
    assert down["tangent_m"] > 4 * up["tangent_m"]
    # At 30 degrees the acute flare's tangent is capped, and its radius with it.
    _, sharp = plan_one(junction(300.0, 30.0))
    f = sharp["flares"]["downstream"]
    assert f["tangent_m"] == pytest.approx(stubs.FLARE_MAX_M, abs=1.0)
    assert f["radius_m"] == pytest.approx(stubs.FLARE_MAX_M * math.tan(math.radians(15.0)), abs=0.3)
    a, b = sharp["mouth"]["rows"]
    assert b - 300 > 300 - a + 15  # the mouth reaches much further along the road ahead


def test_width_option_surface_and_look_follow_the_road_s_rules():
    _, wide = plan_one(junction(300.0, 90.0), width_factor=1.15)
    assert wide["width_m"] == pytest.approx(5.8 * 1.15) and wide["width_mapped_m"] == 5.8
    _, gravel = plan_one(junction(300.0, 90.0, tags={"surface": "unpaved"}))
    assert (gravel["surface"], gravel["width_m"], gravel["style"]) == ("unsealed", 5.0, "gravel_narrow")
    assert gravel["crown"] == section.CROWN_UNSEALED
    c, sec = road(unsealed=True)
    _, untagged = plan_one(junction(300.0, 90.0), c, sec)  # untagged: as the road is where it meets it
    assert untagged["surface"] == "unsealed"
    _, marked = plan_one(junction(300.0, 90.0, highway="tertiary", closure="major"))
    assert (marked["style"], marked["width_m"]) == ("road", 6.0)


def test_heights_carry_the_road_s_cross_fall_then_the_side_road_s_own_profile():
    edge = GROUND - CROWN * HALF
    _, st = plan_one(junction(300.0, 90.0), dem=flat(GROUND + 1.0))
    z = np.asarray(st["rows"]["z"])
    assert z[0] == pytest.approx(edge, abs=0.01)  # the road's edge
    assert st["blend_m"] == stubs.BLEND_M
    assert z[25:] == pytest.approx(GROUND + 1.0, abs=0.01)  # its own profile, past the blend
    left, right = np.asarray(st["rows"]["z_left"]), np.asarray(st["rows"]["z_right"])
    assert (z - left)[25:] == pytest.approx(CROWN * 2.9, abs=1e-3) and (z - right)[25:] == pytest.approx(CROWN * 2.9,
                                                                                                        abs=1e-3)
    # Well above the road: the blend lengthens so the side road climbs no more than MAX_BLEND_GRADE.
    _, steep = plan_one(junction(300.0, 90.0), dem=flat(GROUND + 8.0))
    zz = np.asarray(steep["rows"]["z"])
    assert steep["blend_m"] == pytest.approx(8.0 / stubs.MAX_BLEND_GRADE, abs=1.0)
    assert np.diff(zz).max() <= 1.6 * stubs.MAX_BLEND_GRADE  # a smoothstep peaks at 1.5 times its mean
    # Each flare starts at the road's edge height.
    assert steep["flares"]["upstream"]["points"][0][2] == pytest.approx(edge, abs=1e-3)


def test_pose_surface_and_ground_for_the_closures():
    _, st = plan_one(junction(300.0, 90.0), dem=flat(GROUND + 1.0))
    at = stubs.pose(st)
    e, n, h, hd = at(12.0)  # the obstacle's distance in from the edge line
    assert (e, n) == pytest.approx((300.0, HALF + 12.0), abs=0.05) and hd == pytest.approx(math.pi / 2, abs=1e-3)
    assert h == pytest.approx(float(np.interp(12.0, st["rows"]["d"], st["rows"]["z"])))
    e, n, h, _ = at(st["length_m"] + 10.0)  # past the end: straight on
    assert n == pytest.approx(HALF + st["length_m"] + 10.0, abs=0.05)
    w = st["width_m"] / 2
    assert stubs.surface_height(st, 37.0, w) == pytest.approx(np.interp(37.0, st["rows"]["d"], st["rows"]["z_left"]))
    assert stubs.surface_height(st, 37.0, 0.0) == pytest.approx(np.interp(37.0, st["rows"]["d"], st["rows"]["z"]))
    verge = stubs.ground_height(st, 37.0, -(w + mesh.VERGE_M))
    assert verge == pytest.approx(stubs.surface_height(st, 37.0, -w) - mesh.VERGE_M * mesh.VERGE_FALL)
    assert stubs.ground_height(st, 37.0, w + 6.0, flat(3.0)) == 3.0  # beyond the verge: the ground


def test_a_stub_overlapping_another_on_the_same_side_tries_smaller_flares_then_is_left_out():
    near = [junction(300.0, 90.0, jid=0, highway="tertiary", closure="major"), junction(318.0, 90.0, jid=1)]
    c, sec = road()
    p = stubs.plan(near, c, sec, flat())
    assert [s["junction"] for s in p["stubs"]] == [0, 1]  # 18 m apart: both fit, with smaller flares between
    assert p["stubs"][1]["flares"]["upstream"]["radius_m"] < stubs.FLARE_RADIUS_M
    on_top = [junction(300.0, 90.0, jid=0, highway="tertiary", closure="major"), junction(306.0, 80.0, jid=1)]
    p = stubs.plan(on_top, c, sec, flat())
    assert [s["junction"] for s in p["stubs"]] == [0]  # the major road is kept
    assert p["skipped"] == [{"junction": 1, "name": "Side Road 1", "reason": "overlaps"}]


def test_a_stub_is_cut_short_before_another_part_of_the_route():
    c, sec = road()
    # A side road leaving north, then turning back to run beside the road 12 m off it.
    line = [[300.0, 0.0], [300.0, 30.0], [330.0, 30.0], [400.0, 30.0], [440.0, 30.0]]
    j = {**junction(300.0, 90.0), "line": line, "length_m": 170.0}
    p = stubs.plan([j], c, sec, flat())
    # Its corner is filleted at 8 m, then it stays 30 m off the road for good: kept whole.
    assert p["stubs"][0]["ends"] == "length"
    hairpin = [[600.0, 0.0], [600.0, 40.0], [660.0, 40.0], [660.0, 9.0]]  # comes back to 9 m off the road
    p = stubs.plan([{**junction(600.0, 90.0), "line": hairpin}], c, sec, flat())
    st = p["stubs"][0]
    assert st["ends"] == "clipped"
    end = np.asarray(st["rows"]["xy"][-1])
    assert end[1] > HALF + 2.9 + 2 * mesh.VERGE_M  # its verge stays off the road's


def test_a_stub_whose_mouth_would_leave_a_timing_gate_nowhere_to_stand_is_left_out():
    c, sec = road(n=131)  # a 130 m road: the start gate may only move between 46 and 65 m
    p = stubs.plan([junction(55.0, 90.0, length=60.0)], c, sec, flat())
    assert p["stubs"] == [] and p["skipped"][0]["reason"] == "timing_gate"
    p = stubs.plan([junction(55.0, 90.0, length=60.0)], *road(), flat())  # on a long road it moves the gate instead
    assert len(p["stubs"]) == 1


def test_the_mouth_reaches_over_the_flares_and_the_side_road_takes_in_the_stub():
    c, sec = road()
    j = junction(300.0, 90.0)
    _, st = plan_one(j)
    (plain,) = junctions.mouths([j], c["s"], c["xy"], c["heading"], sec["half_left"], sec["half_right"])
    (grown,) = junctions.mouths([j], c["s"], c["xy"], c["heading"], sec["half_left"], sec["half_right"],
                                stubs=stubs.by_junction([st]))
    assert grown.s0 == st["mouth"]["from_m"] - junctions.MOUTH_MARGIN_M < plain.s0
    assert grown.s1 == st["mouth"]["to_m"] + junctions.MOUTH_MARGIN_M > plain.s1
    corner = shapely.Point(300.0 + 2.9 + 2.0, HALF + 1.0)  # on the flare, off the mapped carriageway
    assert grown.corridor().contains(corner) and not plain.corridor().contains(corner)


# -- the meshes --------------------------------------------------------------------------------------------------------

def built(junction_list, dem_h=GROUND, gravel=False, skirt=True):
    c, sec = road(unsealed=gravel)
    plan = stubs.plan(junction_list, c, sec, flat(dem_h))["stubs"]
    caps = stubs.mainline_caps(plan, c["xy"], c["heading"])
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], flat(dem_h), skirt_cap_l=caps[0], skirt_cap_r=caps[1])
    b = stubs.build(plan, rb, c["s"], FRAME, flat(dem_h), skirt=skirt, main_xy=c["xy"])
    return c, sec, plan, rb, b


def triangles(meshes, prefix=None):
    out = [m.positions.astype(float)[m.triangles] for m in meshes if prefix is None or m.name.startswith(prefix)]
    return np.concatenate(out)


def coverage(tris, xz):
    """How many triangles cover each AC x/z point (strictly inside)."""
    a, b, c = tris[:, 0, [0, 2]], tris[:, 1, [0, 2]], tris[:, 2, [0, 2]]
    v0, v1 = c - a, b - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    keep = np.abs(den) > 1e-9
    a, v0, v1, den = a[keep], v0[keep], v1[keep], den[keep]
    out = np.zeros(len(xz), int)
    for k, p in enumerate(xz):
        v2 = p - a
        u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        out[k] = int(((u > 1e-6) & (v > 1e-6) & (u + v < 1 - 1e-6)).sum())
    return out


def ac_xz(points):
    p = np.asarray(points, float).reshape(-1, 2)
    return FRAME.to_ac(p[:, 0], p[:, 1], np.zeros(len(p)))[:, [0, 2]]


@pytest.mark.parametrize("deg", [90.0, -45.0, 120.0])
def test_the_throat_joins_the_road_s_edge_and_the_stub_without_a_crack_or_an_overlap(deg):
    c, sec, plan, rb, b = built([junction(300.0, deg)])
    (st,) = plan
    side = junctions.SIDE_CODE[st["side"]]
    a, bb = st["mouth"]["rows"]
    road_meshes = mesh.ribbon_meshes(rb, c["s"], FRAME, cut=b.cut, edge_break=b.edge_break) + b.meshes
    stub_road = triangles(b.meshes, "1ROAD_J")
    # The throat shares the road's edge vertices across the mouth, and the stub ribbon's first row, exactly.
    col = rb.road_cols.start if side == "L" else rb.road_cols.stop - 1
    edge = FRAME.to_ac(rb.e[a:bb + 1, col], rb.n[a:bb + 1, col], rb.h[a:bb + 1, col]).astype(np.float32)
    verts = {tuple(v) for v in stub_road.reshape(-1, 3).astype(np.float32).tolist()}
    assert all(tuple(v) in verts for v in edge.tolist())
    # Every point of the paved outline and a metre either side of the road's edge is road, exactly once.
    paved = shapely.Polygon(st["outline"]).buffer(0)
    # Off the stations and the ribbon's columns, so no point falls on an edge two triangles share.
    xs, ys = np.meshgrid(np.arange(paved.bounds[0] + 0.013, paved.bounds[2], 0.37),
                         np.arange(-HALF + 0.217, paved.bounds[3], 0.37))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    inside = shapely.contains_xy(paved.buffer(-0.02), pts[:, 0], pts[:, 1]) | (np.abs(pts[:, 1]) < HALF - 0.05)
    count = coverage(triangles(road_meshes, "1ROAD"), ac_xz(pts[inside]))
    assert (count == 1).all()


def test_the_road_s_verge_is_cut_across_the_mouth_and_a_mown_verge_follows_each_flare():
    c, sec, plan, rb, b = built([junction(300.0, 90.0)])
    (st,) = plan
    a, bb = st["mouth"]["rows"]
    assert b.cut["L"][a:bb].all() and b.cut["L"].sum() == bb - a and not b.cut["R"].any()
    verges = [m for m in mesh.ribbon_meshes(rb, c["s"], FRAME, cut=b.cut) if m.name.endswith("L") and "_V" in m.name]
    xz = np.concatenate([m.positions[m.triangles].mean(axis=1)[:, [0, 2]] for m in verges])
    assert not ((xz[:, 0] > a + 0.01) & (xz[:, 0] < bb - 0.01)).any()  # no verge triangle across the mouth
    # The flares' verges share the road's verge vertex at each tangent point.
    grass = triangles(b.meshes, "1GRASS_J").reshape(-1, 3).astype(np.float32)
    vg = {tuple(v) for v in grass.tolist()}
    for row in (a, bb):
        col = rb.road_cols.start - 1
        p = FRAME.to_ac(rb.e[row, col], rb.n[row, col], rb.h[row, col]).astype(np.float32)
        assert tuple(p.tolist()) in vg


def test_the_edge_line_breaks_across_the_mouth_by_texture_coordinates():
    c, sec, plan, rb, b = built([junction(300.0, 90.0)])
    a, bb = plan[0]["mouth"]["rows"]
    plain = mesh.ribbon_meshes(rb, c["s"], FRAME)
    broken = mesh.ribbon_meshes(rb, c["s"], FRAME, edge_break=b.edge_break)
    (r0,) = [m for m in plain if m.name == "1ROAD_001"]  # rows 200-400
    (r1,) = [m for m in broken if m.name == "1ROAD_001"]
    u0, u1 = r0.uvs[:, 0].reshape(-1, 5), r1.uvs[:, 0].reshape(-1, 5)
    rows = np.arange(200, 401)
    mouth = (rows > a) & (rows < bb)
    line_outer = textures.edge_line_u("road_marked")
    assert (u1[mouth, 0] > line_outer).all() and (u1[mouth, 0] < 0.1).all()  # inside the line: plain asphalt
    assert (u1[~mouth] == u0[~mouth]).all() and (u1[:, 1:] == u0[:, 1:]).all()  # nothing else moves
    assert np.array_equal(r1.positions, r0.positions)
    # A gravel road has no lines, so nothing changes.
    c, sec, plan, rb, b = built([junction(300.0, 90.0)], gravel=True)
    g0 = mesh.ribbon_meshes(rb, c["s"], FRAME, style=sec["style"])
    g1 = mesh.ribbon_meshes(rb, c["s"], FRAME, style=sec["style"], edge_break=b.edge_break)
    assert all(np.array_equal(m.uvs, n.uvs) for m, n in zip(g0, g1))
    # road_plain's line is further in (it's drawn for 4 m): its break is too.
    assert textures.edge_line_u("road_plain") > line_outer and textures.edge_line_u("gravel_road") == 0.0


@pytest.mark.parametrize("skirt", [True, False])
def test_the_terrain_fits_round_the_stub_with_no_gap_and_nothing_over_it(skirt):
    c, sec, plan, rb, b = built([junction(300.0, 70.0)], dem_h=GROUND - 0.4, skirt=skirt)
    (st,) = plan
    meshes = (mesh.ribbon_meshes(rb, c["s"], FRAME, skirt=skirt, cut=b.cut, edge_break=b.edge_break) + b.meshes
              + mesh.terrain_meshes(rb, 150.0, flat(GROUND - 0.4), FRAME, fold_skirt=not skirt, extra=b.footprint))
    paved = shapely.Polygon(st["outline"]).buffer(0)
    around = validate._grid_in(paved.buffer(3.0), 0.43)
    ground = triangles(meshes)
    count = coverage(ground, ac_xz(around))
    assert (count >= 1).all()  # no hole round it
    on_stub = validate._grid_in(paved.buffer(-0.05), 0.61)
    terrain = triangles(meshes, "1GRASS")
    assert (coverage(terrain, ac_xz(on_stub)) == 0).all()  # no ground under or over its road
    # The terrain shares the stub ribbon's outline vertices.
    sb = b.footprint.ribbons[0]
    tv = {tuple(v) for v in triangles(meshes, "1GRASS_T").reshape(-1, 3).astype(np.float32).tolist()}
    far = FRAME.to_ac(sb.e[-1, 0], sb.n[-1, 0], sb.h[-1, 0]).astype(np.float32)
    assert tuple(far.tolist()) in tv


def test_stub_meshes_are_named_by_surface_and_packed_by_material():
    c, sec, plan, rb, b = built([junction(300.0, 90.0, jid=0),
                                 junction(600.0, 90.0, jid=1, tags={"surface": "gravel"}),
                                 junction(650.0, -90.0, jid=2, tags={"surface": "gravel"})])
    names = sorted(m.name for m in b.meshes)
    assert names == ["1GRASS_J000", "1GRASS_J001", "1GRAVEL_J001N", "1ROAD_J000"]
    assert {m.material for m in b.meshes} == {"road", "gravel_narrow", "grass"}
    assert b.stats["stubs"] == 3 and b.stats["vertices"] == sum(len(m.positions) for m in b.meshes)
    for m in b.meshes:
        m.validate()
        assert np.allclose(np.linalg.norm(m.normals, axis=1), 1.0, atol=1e-3)


def test_ribbon_skirt_caps_hold_the_skirt_short_and_leave_the_verge_alone():
    c, sec = road(n=201)
    args = (c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"], sec["slope_left"],
            sec["slope_right"], flat())
    plain = mesh.build_ribbon(*args)
    cap = np.full(201, np.inf)
    cap[100] = HALF + 5.0
    capped = mesh.build_ribbon(*args, skirt_cap_l=cap, skirt_cap_r=np.full(201, np.inf))
    lc = plain.road_cols.start
    assert capped.n[100, lc - 1] == pytest.approx(plain.n[100, lc - 1])  # the verge keeps its 2 m
    assert capped.n[100, 0] == pytest.approx(HALF + 5.0)  # the skirt stops at the cap
    assert capped.n[50, 0] == pytest.approx(plain.n[50, 0])
    tight = np.full(201, HALF)  # never less than 0.5 m past the verge
    assert mesh.build_ribbon(*args, skirt_cap_l=tight, skirt_cap_r=tight).n[100, 0] == pytest.approx(
        HALF + mesh.VERGE_M + 0.5)


def test_side_caps_keep_each_skirt_nearer_its_own_centreline():
    xy = np.column_stack([np.arange(10.0), np.zeros(10)])
    nrm = np.tile([0.0, 1.0], (10, 1))
    cap_l, cap_r = stubs.side_caps(xy, nrm, np.array([[5.0, 10.0]]), 30.0)
    assert cap_l[5] == pytest.approx(5.0) and np.isinf(cap_r).all()  # half way, straight across
    assert cap_l[0] == pytest.approx((25 + 100) / 20)  # D^2 / 2a, further along


# -- validation --------------------------------------------------------------------------------------------------------

def kn5_nodes(meshes):
    return {"nodes": [{"class": 1, "name": "root"}] + [
        {"class": 2, "name": m.name, "vertices": {"pos": m.positions}, "indices": m.triangles.astype(np.uint16).ravel()}
        for m in meshes]}


def stub_check(st, c, sec):
    from mapcreator.pipeline.validate_stage import _stub_check
    return _stub_check(st, c, sec, FRAME)


def test_the_validator_passes_joined_stubs_and_fails_a_gap_or_ground_over_one():
    c, sec, plan, rb, b = built([junction(300.0, 90.0)], dem_h=GROUND - 0.4)
    (st,) = plan
    meshes = (mesh.ribbon_meshes(rb, c["s"], FRAME, cut=b.cut, edge_break=b.edge_break) + b.meshes
              + mesh.terrain_meshes(rb, 150.0, flat(GROUND - 0.4), FRAME, extra=b.footprint))
    check = validate.JunctionCheck("Side Road 0", shapely.box(0, 0, 1, 1), shapely.box(0, 0, 1, 1),
                                   stub_check(st, c, sec))
    rep = validate.Report()
    validate.stub_checks(rep, "main", kn5_nodes(meshes), [check])
    assert rep.failed == [] and len(rep.passed) == 4
    no_stub = [m for m in meshes if not m.name.startswith("1ROAD_J")]
    rep = validate.Report()
    validate.stub_checks(rep, "main", kn5_nodes(no_stub), [check])
    assert any("drawn as road" in f for f in rep.failed) and any("no gap" in f for f in rep.failed)
    lifted = [m if not m.name.startswith("1GRASS_T") else
              type(m)(m.name, m.material, m.positions + np.float32([0, 2.0, 0]), m.normals, m.uvs, m.triangles)
              for m in meshes]
    extra_ground = mesh.soup_meshes("1GRASS_X", "grass", [(np.array([[290.0, 5.0, 20.0], [310.0, 5.0, 20.0],
                                                                    [300.0, 30.0, 20.0]]),
                                                           np.zeros((3, 2)), np.array([[0, 1, 2]]))], FRAME)
    rep = validate.Report()
    validate.stub_checks(rep, "main", kn5_nodes(lifted + extra_ground), [check])
    assert any("no ground above" in f for f in rep.failed)


# -- the section stage, and `junctions: none` --------------------------------------------------------------------------

HOBART = [{"lon": 147.301, "lat": -42.8001}, {"lon": 147.319, "lat": -42.7999}]


class _Dem:
    resolution_m = 2.0

    def load_corridor(self, xy, m):
        return 1

    def sample(self, x, y):
        return np.full(np.shape(x), GROUND + 0.5)


class _Region:
    epsg = 28355

    def elevation(self, settings):
        return _Dem()


def _work(tmp_path, junction_list):
    work = WorkDir(tmp_path / "work")
    c, _ = road()
    np.savez_compressed(work.stage("02_profile") / "centreline.npz", run=np.zeros(len(c["s"]), np.int64), **c)
    work.write_json(route_stage.STAGE, "runs.json", [{"tags": {"highway": "secondary", "surface": "asphalt"}}])
    work.write_json(route_stage.STAGE, route_stage.JUNCTIONS, {"version": 1, "junctions": junction_list})
    return work


def test_the_section_stage_plans_stubs_and_none_leaves_the_section_and_the_ai_line_byte_for_byte(tmp_path,
                                                                                                   monkeypatch):
    monkeypatch.setattr(section_stage, "job_region", lambda waypoints: _Region())
    settings = config.load()
    work = _work(tmp_path, [junction(300.0, 90.0), junction(700.0, -60.0, jid=1)])
    job = Job(track_id="t", waypoints=HOBART)
    out = section_stage.run(job, settings, work)
    assert out["junction_stubs"] == {"stubs": 2, "skipped": {}}
    doc = work.read_json(section_stage.STAGE, section_stage.STUBS)
    assert doc["version"] == stubs.FORMAT_VERSION and doc["crs"] == "EPSG:28355"
    assert [s["junction"] for s in section_stage.stubs(work, job)] == [0, 1]
    with_stubs = dict(np.load(work.root / section_stage.STAGE / "section.npz"))
    none = Job(track_id="t", waypoints=HOBART, junctions="none")
    assert section_stage.stubs(work, none) == []  # later stages ignore the plan
    out = section_stage.run(none, settings, work)
    assert "junction_stubs" not in out and not (work.root / section_stage.STAGE / section_stage.STUBS).exists()
    without = dict(np.load(work.root / section_stage.STAGE / "section.npz"))
    assert with_stubs.keys() == without.keys() and all(np.array_equal(with_stubs[k], without[k]) for k in with_stubs)
    # The AI line and its hints come from the centreline and the section alone, so they're the same bytes.
    c = road()[0]
    files = []
    for sec in (with_stubs, without):
        rm = RoadModel.from_arrays(c["s"], c["xy"], c["z"], c["heading"], c["curvature"], c["grade"],
                                   sec["half_left"], sec["half_right"], sec["slope_left"], sec["slope_right"], False)
        path = tmp_path / f"fast_lane_{len(files)}.ai"
        ai_line.write(path, ai_line.build(rm, FRAME, end_short_m=40.0))
        files.append(path.read_bytes())
    assert files[0] == files[1]


def test_without_junctions_the_section_stage_writes_no_plan_and_removes_a_stale_one(tmp_path, monkeypatch):
    monkeypatch.setattr(section_stage, "job_region", lambda waypoints: _Region())
    work = _work(tmp_path, [])
    (work.root / section_stage.STAGE).mkdir(parents=True, exist_ok=True)
    (work.root / section_stage.STAGE / section_stage.STUBS).write_text(json.dumps({"stubs": [{"junction": 9}]}))
    job = Job(track_id="t", waypoints=HOBART)
    section_stage.run(job, config.load(), work)
    assert not (work.root / section_stage.STAGE / section_stage.STUBS).exists()
    assert section_stage.stubs(work, job) == [] and stubs.plan([], *road(), flat())["stubs"] == []
