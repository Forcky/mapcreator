"""The route's own bridges (`core.bridges`): the plan (open spans, piers, parapets clear of mouths), the kerb strip in
the ribbon, the open deck with the terrain running on under it, the abutments' seams with the road and the ground, the
parapets' colliders flush with their faces, 1:2 batters on fills, the meshes' names and budget, and the checks."""

import numpy as np
import pytest
import shapely

from mapcreator.core import bridges, mesh, section, validate
from mapcreator.core.scene import LocalFrame

FRAME = LocalFrame(0.0, 0.0, 0.0)
HALF = 3.2
CROWN = section.CROWN
DECK = 20.0
N = 401


def gully(depth=10.0, centre=200.0, width=80.0, angle=0.0):
    """A DEM: level at DECK, with a gully across the road `centre` metres along it, `width` across at the top, `depth`
    deep. The road heads `angle` radians north of east (`road`)."""
    def sample(x, y):
        along = np.asarray(x, float) * np.cos(angle) + np.asarray(y, float) * np.sin(angle)
        d = np.clip(1 - np.abs(along - centre) / (width / 2), 0, 1)
        return DECK - depth * np.sin(np.pi / 2 * d) ** 2
    return sample


def road(dem, rows=(165, 235), n=N, angle=0.0):
    """A straight road from the origin at DECK, `angle` radians north of east (along y = 0 by default), a station
    every metre, 6.4 m wide, tagged a bridge on `rows`."""
    s = np.arange(n, dtype=float)
    xy = np.column_stack([s * np.cos(angle), s * np.sin(angle)])
    bridge = np.zeros(n, bool)
    if rows is not None:
        bridge[rows[0]:rows[1] + 1] = True
    z_raw = dem(xy[:, 0], xy[:, 1])
    filled = bridge.copy()
    if rows is not None:
        filled[rows[0] - 3:rows[1] + 4] = True
    c = {"s": s, "xy": xy, "heading": np.full(n, angle), "curvature": np.zeros(n), "z": np.full(n, DECK),
         "z_raw": z_raw, "bridge": bridge, "filled": filled, "tunnel": np.zeros(n, bool), "under": np.zeros(n, bool),
         "run": np.zeros(n, int), "grade": np.zeros(n)}
    sec = {"half_left": np.full(n, HALF), "half_right": np.full(n, HALF), "slope_left": np.full(n, CROWN),
           "slope_right": np.full(n, CROWN), "style": np.zeros(n, np.int8)}
    return c, sec


def built(dem=None, rows=(165, 235), skirt=True, mouths=None, corridor=120.0, angle=0.0):
    """Everything the mesh stage builds for the road and its bridge, in its order."""
    dem = dem or gully(angle=angle)
    c, sec = road(dem, rows, angle=angle)
    plan = bridges.plan(c, mouths=mouths, names=["Test Road"])
    decks = bridges.Decks(plan, c, sec, mouths)
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], decks.embanked(dem), **decks.verges())
    extra = decks.footprint(rb, None)
    ground = decks.ground(dem)
    extra = decks.footprint_points(extra, ground)
    meshes = mesh.ribbon_meshes(rb, c["s"], FRAME, skirt=skirt, cut=decks.cut(None), kerb=decks.kerb())
    meshes += mesh.terrain_meshes(rb, corridor, ground, FRAME, fold_skirt=not skirt, extra=extra)
    deck_meshes, stats = decks.meshes(rb, FRAME, ground)
    return c, sec, plan, decks, rb, meshes, deck_meshes, stats


def triangles(meshes, prefix=None):
    out = [m.positions.astype(float)[m.triangles] for m in meshes if prefix is None or m.name.startswith(prefix)]
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def ac(points, h=None):
    p = np.asarray(points, float).reshape(-1, 2)
    return FRAME.to_ac(p[:, 0], p[:, 1], np.zeros(len(p)) if h is None else h)


def verts(meshes, prefix=None):
    return {tuple(v) for v in triangles(meshes, prefix).reshape(-1, 3).astype(np.float32).tolist()}


# -- the plan ----------------------------------------------------------------------------------------------------------

def test_a_bridge_over_a_deep_gully_is_an_open_deck_with_piers_about_every_25_m():
    c, sec = road(gully())
    (b,) = bridges.plan(c, names=["Test Road"])
    assert b["rows"] == [165, 235] and b["name"] == "Test Road" and b["look"] == "open"
    a, bb = b["open_rows"]
    gap = c["z"] - c["z_raw"]
    # From the first station whose gap reaches 1.5 m to the last.
    assert gap[a] >= bridges.OPEN_GAP_M > gap[a - 1] and gap[bb] >= bridges.OPEN_GAP_M > gap[bb + 1]
    assert b["gap_max_m"] == pytest.approx(10.0, abs=0.01)
    piers = np.asarray(b["piers_m"])
    assert len(piers) == round((bb - a) / bridges.PIER_SPAN_M) - 1
    assert np.diff(np.concatenate([[a], piers, [bb]])) == pytest.approx((bb - a) / (len(piers) + 1), abs=1.0)


def test_a_shallow_bridge_keeps_the_road_on_the_ground_with_parapets_and_a_short_one_gets_no_piers():
    c, sec = road(gully(depth=1.2))
    (b,) = bridges.plan(c)
    assert b["look"] == "parapets" and b["open_rows"] is None and b["piers_m"] == []
    m = bridges.masks([b], N)
    assert m["L"][165:236].all() and m["R"][165:236].all() and m["L"].sum() == 71
    c, sec = road(gully(depth=4.0, width=24.0), rows=(185, 215))
    (b,) = bridges.plan(c)
    assert b["look"] == "open" and b["piers_m"] == []  # shorter than a span


def test_the_open_span_keeps_clear_of_side_roads_mouths_and_the_parapet_out_of_them():
    c, sec = road(gully())
    mouths = {"L": np.zeros(N, bool), "R": np.zeros(N, bool)}
    mouths["L"][160:172] = True  # a side road meeting the route at the bridge's west end
    (b,) = bridges.plan(c, mouths=mouths)
    a, bb = b["open_rows"]
    assert a >= 172 + bridges.MOUTH_CLEAR_M
    m = bridges.masks([b], N, mouths)
    assert not m["L"][160:172].any() and m["L"][172:236].all() and m["R"][165:236].all()


def test_untagged_bridges_and_tunnels_are_not_bridges():
    c, sec = road(gully(), rows=None)
    assert bridges.plan(c) == []


# -- the ribbon and the ground -----------------------------------------------------------------------------------------

def test_the_kerb_strip_replaces_the_verge_on_the_bridge_in_the_same_ribbon():
    c, sec, plan, decks, rb, meshes, deck_meshes, stats = built()
    k = rb.road_cols.start - 1
    edge_h = DECK - CROWN * HALF
    # On the bridge the verge column is the kerb: KERB_M out, KERB_RISE_M up; beside it the 2 m grass verge as before.
    assert rb.n[200, k] == pytest.approx(HALF + bridges.KERB_M) and rb.h[200, k] == pytest.approx(edge_h + 0.1)
    assert rb.n[100, k] == pytest.approx(HALF + mesh.VERGE_M)
    # The kerb mesh shares the road's edge vertices and the verge's last row exactly, so there's no crack.
    kerb = [m for m in deck_meshes if m.name.startswith("1ROAD_BK")]
    assert sorted(m.name for m in kerb) == ["1ROAD_BK0L", "1ROAD_BK0R"] and {m.material for m in kerb} == {"kerb"}
    kv = verts(kerb)
    road = verts(meshes, "1ROAD_")
    edge = ac(np.column_stack([rb.e[165:236, k + 1], rb.n[165:236, k + 1]]), rb.h[165:236, k + 1]).astype(np.float32)
    assert all(tuple(v) in kv for v in edge.tolist()) and all(tuple(v) in road for v in edge.tolist())
    grass = verts(meshes, "1GRASS_V")
    for row in (165, 235):
        p = ac([[rb.e[row, k], rb.n[row, k]]], [rb.h[row, k]]).astype(np.float32)
        assert tuple(p[0].tolist()) in kv and tuple(p[0].tolist()) in grass
    # No grass verge on the bridge.
    gx = triangles(meshes, "1GRASS_V").mean(axis=1)
    on_deck = (gx[:, 0] > 165.5) & (gx[:, 0] < 234.5) & (np.abs(gx[:, 2]) < HALF + bridges.KERB_M)
    assert not on_deck.any()


@pytest.mark.parametrize("skirt", [True, False])
def test_the_terrain_runs_on_under_the_open_deck_with_no_hole_and_nothing_over_it(skirt):
    c, sec, plan, decks, rb, meshes, deck_meshes, stats = built(skirt=skirt)
    (b,) = plan
    a, bb = b["open_rows"]
    terrain = validate._Cover(triangles(meshes, "1GRASS"))
    # Under the middle of the deck the ground is the gully's floor, not an embankment.
    mid = ac([[200.0, 0.0], [200.0, 2.0], [200.0, -3.0]])[:, [0, 2]]
    assert terrain.top(mid) == pytest.approx(DECK - 10.0, abs=0.3)
    # Over the whole open span, nothing within CLEAR_M of the underside, and no hole anywhere round the bridge.
    deck = shapely.box(a + 1.0, -HALF - 1.0, bb - 1.0, HALF + 1.0)
    pts = validate._grid_in(deck, 0.53)
    under = DECK - CROWN * HALF + bridges.KERB_RISE_M - bridges.DECK_DEPTH_M
    top = terrain.top(ac(pts)[:, [0, 2]])
    assert np.isfinite(top).all() and (top <= under - bridges.CLEAR_M + 1e-6).all()
    around = validate._grid_in(shapely.box(150.0, -30.0, 250.0, 30.0), 0.71)
    covers = validate._Cover(triangles(meshes + deck_meshes))
    assert np.isfinite(covers.top(ac(around)[:, [0, 2]])).all()
    assert validate.bridge_failures(validate.mesh_nodes(meshes + deck_meshes),
                                    [(0, chk) for chk in bridges.checks(decks.records(rb), FRAME)]) == {0: []}


# Lines across the terrain's lattice that left a hole beside an abutment while the terrain wasn't constrained to the
# footprint's outline (`mesh._constrain`).
@pytest.mark.parametrize("angle, shift", [(0.21, 1.3), (1.19, 0.0), (1.68, 0.0), (2.17, 1.3)])
@pytest.mark.parametrize("skirt", [True, False])
def test_no_hole_round_a_bridge_whatever_its_line_across_the_terrain_s_lattice(angle, shift, skirt):
    dem = gully(angle=angle, centre=200.0 + shift)
    c, sec, plan, decks, rb, meshes, deck_meshes, stats = built(dem=dem, angle=angle, skirt=skirt)
    along, across = np.meshgrid(np.arange(150.0, 250.0, 0.43), np.arange(-30.0, 30.0, 0.43))
    a, n = along.ravel(), across.ravel()
    pts = np.column_stack([a * np.cos(angle) - n * np.sin(angle), a * np.sin(angle) + n * np.cos(angle)])
    covers = validate._Cover(triangles(meshes + deck_meshes))
    assert np.isfinite(covers.top(ac(pts)[:, [0, 2]])).all()
    holes, _ = validate.ground_holes(validate.mesh_nodes(meshes + deck_meshes))  # the validator's own check
    assert not len(holes), holes[:5]


def test_each_abutment_s_face_shares_the_road_s_row_and_the_terrain_s_foot():
    c, sec, plan, decks, rb, meshes, deck_meshes, stats = built()
    (b,) = plan
    a, bb = b["open_rows"]
    face = verts(deck_meshes, "BRIDGE_0")
    road_and_kerb = verts(meshes, "1ROAD_") | verts(deck_meshes, "1ROAD_BK")
    terrain = verts(meshes, "1GRASS_T")
    c0, c1 = rb.road_cols.start, rb.road_cols.stop  # the road's columns; the kerb's outer edges either side
    for row in (a, bb):
        top = ac(np.column_stack([rb.e[row, c0:c1], rb.n[row, c0:c1]]), rb.h[row, c0:c1]).astype(np.float32)
        assert all(tuple(v) in face and tuple(v) in road_and_kerb for v in top.tolist())
        foot, h = decks.spans[0].feet[row]
        fv = ac(foot, h).astype(np.float32)
        assert all(tuple(v) in face and tuple(v) in terrain for v in fv.tolist())
        # The foot is under the deck, well below the road: the wall drops to the ground.
        assert (h <= DECK - CROWN * HALF - bridges.DECK_DEPTH_M - bridges.CLEAR_M + 0.1 + 1e-6).all()
        # Its outer corners are the kerb's outer edges, which the kerb and the ground beside the bridge share.
        for col in (c0 - 1, c1):
            p = ac([[rb.e[row, col], rb.n[row, col]]], [rb.h[row, col]]).astype(np.float32)
            assert tuple(p[0].tolist()) in face and tuple(p[0].tolist()) in road_and_kerb
            assert tuple(p[0].tolist()) in terrain | verts(meshes, "1GRASS_V")
    # No hole the validator can find round the bridge: the ground covers everything beyond each open edge.
    holes, _ = validate.ground_holes(validate.mesh_nodes(meshes + deck_meshes))
    assert not len(holes), holes[:5]


def test_the_parapets_colliders_stand_flush_with_their_faces_with_no_gap_and_no_wall_above():
    c, sec, plan, decks, rb, meshes, deck_meshes, stats = built()
    walls = [m for m in deck_meshes if m.name.startswith(bridges.COLLIDER_PREFIX)]
    assert sorted(m.name for m in walls) == ["1WALL_BRIDGE_0L", "1WALL_BRIDGE_0R"]
    assert not any(m.renderable for m in walls)
    for m in walls:
        side = m.name[-1]
        k = rb.road_cols.start - 1 if side == "L" else rb.road_cols.stop
        face = ac(np.column_stack([rb.e[165:236, k], rb.n[165:236, k]]))[:, [0, 2]]
        xz = m.positions[:, [0, 2]].astype(float)
        gap = np.min(np.hypot(*(face[:, None, :] - xz[None, :, :]).transpose(2, 0, 1)), axis=1)
        assert gap.max() < 1e-3  # a vertex on every row of the face
        base = rb.h[165:236, k]
        y = m.positions[:, 1].reshape(-1, 2)
        assert y[:, 0] == pytest.approx(base + bridges.COLLIDER[0], abs=1e-4)
        assert y[:, 1] == pytest.approx(base + bridges.COLLIDER[1], abs=1e-4)
    # The parapet's inner face is the collider's line, and nothing of the deck reaches past its outer face (the
    # abutments' faces, which reach out to the skirt, aside).
    (b,) = plan
    a, bb = b["open_rows"]
    concrete = triangles(deck_meshes, "BRIDGE_0")
    middle = concrete[((concrete[..., 0] > a + 1) & (concrete[..., 0] < bb - 1)).all(axis=1)]
    assert np.abs(middle[..., 2]).max() == pytest.approx(HALF + bridges.KERB_M + bridges.PARAPET_W, abs=1e-3)
    inner = middle.reshape(-1, 3)
    barrier_top = DECK - CROWN * HALF + bridges.KERB_RISE_M + bridges.PARAPET_H
    inner = inner[np.abs(inner[:, 1] - barrier_top) < 0.01]
    assert np.abs(inner[:, 2]).min() == pytest.approx(HALF + bridges.KERB_M, abs=1e-3)
    chk = bridges.checks(decks.records(rb), FRAME)
    assert validate.bridge_failures(validate.mesh_nodes(walls), [(0, chk[0])])[0] == []
    assert validate.bridge_failures(validate.mesh_nodes(walls[:1]), [(0, chk[0])])[0] == ["collider_gap"]


def test_piers_stand_from_below_the_ground_to_the_underside():
    c, sec, plan, decks, rb, meshes, deck_meshes, stats = built()
    (b,) = plan
    assert stats["piers"] == len(b["piers_m"]) >= 1
    concrete = triangles(deck_meshes, "BRIDGE_0").reshape(-1, 3)
    for at in b["piers_m"]:
        under = DECK - CROWN * HALF + bridges.KERB_RISE_M - bridges.DECK_DEPTH_M
        near = concrete[(np.abs(concrete[:, 0] - at) < 1.2) & (concrete[:, 1] < under + 0.01)]  # under the deck
        ground = gully()(np.array([at]), np.array([0.0]))[0]
        assert near[:, 1].min() < ground - 0.3  # buried
        assert near[:, 1].max() == pytest.approx(under, abs=0.01)  # the headstock's top is the underside
        columns = near[near[:, 1] < under - bridges.PIER_HEADSTOCK[0] - 0.01]
        assert len(columns) and np.abs(columns[:, 2]).max() < HALF + bridges.KERB_M  # inside the deck's edges


def test_a_fill_takes_a_1_in_2_batter_from_the_verge():
    dem = gully(depth=3.0, width=16.0)
    c, sec = road(dem, rows=None)
    c["filled"][192:209] = True  # an untagged culvert's fill
    rows = bridges.batter_rows(c, [])
    assert rows[190:212].all() and not rows[100]
    raised = bridges.embanked(dem, c, sec, rows)
    verge_h = DECK - CROWN * HALF - mesh.VERGE_M * mesh.VERGE_FALL
    for off in (2.0, 4.0, 5.0):
        y = HALF + mesh.VERGE_M + off
        h = raised(np.array([200.0, 200.0]), np.array([y, -y]))
        assert h == pytest.approx(np.maximum(verge_h - 0.5 * off, dem(np.array([200.0]), 0)[0]), abs=1e-6)
    assert raised(np.array([100.0]), np.array([20.0]))[0] == dem(np.array([100.0]), 0)[0]  # off the fill: the DEM
    # Not past the road's ends: a fill at the start raises nothing behind it.
    low = gully(depth=3.0, centre=0.0, width=40.0)
    c0, sec0 = road(low, rows=None)
    c0["filled"][:12] = True
    start = bridges.embanked(low, c0, sec0, bridges.batter_rows(c0, []))
    assert start(np.array([-10.0, -10.0]), np.array([0.0, 6.0])) == pytest.approx(low(np.array([-10.0, -10.0]), 0))
    assert start(np.array([1.0]), np.array([6.0]))[0] > low(np.array([1.0]), 0)[0]
    # The ribbon's skirt follows it.
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], raised)
    off = rb.n[200, 0] - (HALF + mesh.VERGE_M)
    assert rb.h[200, 0] == pytest.approx(max(verge_h - 0.5 * off, DECK - 3.0), abs=1e-6)


# -- meshes, names, budget ---------------------------------------------------------------------------------------------

def test_meshes_are_named_by_surface_and_within_the_budget():
    c, sec, plan, decks, rb, meshes, deck_meshes, stats = built()
    names = {m.name for m in deck_meshes}
    assert {"BRIDGE_0", "BRIDGE_0_U", "BRIDGE_0_RAIL", "1WALL_BRIDGE_0L", "1WALL_BRIDGE_0R", "1ROAD_BK0L",
            "1ROAD_BK0R"} <= names
    for m in deck_meshes:
        m.validate()
        assert np.allclose(np.linalg.norm(m.normals, axis=1), 1.0, atol=1e-3)
        if m.name[:1].isdigit():
            assert sum(k in m.name for k in ("ROAD", "GRASS", "GRAVEL", "WALL")) == 1
    # The bridges over the route's concrete (`core.crossing_roads`): one generic style.
    assert {m.material for m in deck_meshes} == {"kerb", "bridge_concrete", "bridge_soffit", "rail"}
    # A 70 m bridge with two piers: about 60 vertices a metre (parapets, rails and posts, colliders, kerbs, underside).
    assert stats["vertices"] == sum(len(m.positions) for m in deck_meshes) < 70 * 65
    # The underside faces down, the parapets' tops up.
    (u,) = [m for m in deck_meshes if m.name == "BRIDGE_0_U"]
    assert (u.normals[:, 1] < -0.99).all()
    p = u.positions.astype(float)[u.triangles]
    assert (np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 1] < 0).all()  # wound to face down too


def test_reflectors_stand_on_the_parapets_every_18_m_on_the_furniture_atlas():
    from mapcreator.core import furniture

    c, sec, plan, decks, rb, meshes, deck_meshes, stats = built()
    tex = furniture.atlas([])
    out = bridges.reflectors(decks, rb, FRAME, tex)
    assert {m.material for m in out} == {furniture.MATERIAL, furniture.REFLECTOR_MATERIAL}
    assert all(m.name.startswith("FURNITURE_BRIDGE_") for m in out)
    refl = np.concatenate([m.positions for m in out if m.material == furniture.REFLECTOR_MATERIAL])
    per_side = len(np.arange(166.0, 234.5, bridges.REFLECTOR_SPACING_M))
    assert len(refl) == 2 * per_side * 2 * 4  # two faces each, both sides
    top = DECK - CROWN * HALF + bridges.KERB_RISE_M + bridges.PARAPET_H
    assert refl[:, 1].min() > top


def test_without_bridges_the_ribbon_and_meshes_are_unchanged():
    dem = gully(depth=3.0)
    c, sec = road(dem, rows=None)
    args = (c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"], sec["slope_left"],
            sec["slope_right"], dem)
    plain = mesh.build_ribbon(*args)
    decks = bridges.Decks([], c, sec)
    same = mesh.build_ribbon(*args, **decks.verges())
    assert np.array_equal(plain.h, same.h) and np.array_equal(plain.n, same.n)
    a = mesh.ribbon_meshes(plain, c["s"], FRAME)
    b = mesh.ribbon_meshes(same, c["s"], FRAME, kerb=decks.kerb(), cut=decks.cut(None))
    assert [m.name for m in a] == [m.name for m in b]
    assert all(np.array_equal(x.positions, y.positions) and np.array_equal(x.triangles, y.triangles)
               for x, y in zip(a, b))


def test_layout_intervals_follow_the_layout_s_direction():
    c, sec = road(gully())
    plan = bridges.plan(c)
    assert bridges.layout_intervals(plan, c["s"], False) == [(165.0, 235.0)]
    assert bridges.layout_intervals(plan, c["s"], True) == [(165.0, 235.0)]  # symmetric here: 400 - 235 = 165
    c, sec = road(gully(centre=100.0, width=40.0), rows=(90, 110))
    assert bridges.layout_intervals(bridges.plan(c), c["s"], True) == [(290.0, 310.0)]


def test_a_timing_gate_on_a_bridge_reaches_past_the_kerb_strip():
    from mapcreator.core.layout import GATE_OUTSIDE_EDGE_M, RoadModel, logic_dummies

    c, sec = road(gully(centre=50.0, width=60.0), rows=(30, 80))
    road_model = RoadModel.from_arrays(c["s"], c["xy"], c["z"], c["heading"], c["curvature"], c["grade"],
                                       sec["half_left"], sec["half_right"], sec["slope_left"], sec["slope_right"],
                                       False)
    plain = {d.name: d for d in logic_dummies(road_model, FRAME)}
    wider = [(a, b, bridges.KERB_M) for a, b in bridges.layout_intervals(bridges.plan(c), c["s"], False)]
    gated = {d.name: d for d in logic_dummies(road_model, FRAME, wider=wider)}
    # The start gate is 50 m in, on the bridge: its ends move out past the kerb; the finish gate, off it, stays.
    for side in "LR":
        assert abs(plain[f"AC_AB_START_{side}"].position[2]) == pytest.approx(HALF + GATE_OUTSIDE_EDGE_M)
        assert abs(gated[f"AC_AB_START_{side}"].position[2]) == pytest.approx(
            HALF + GATE_OUTSIDE_EDGE_M + bridges.KERB_M)
        assert gated[f"AC_AB_FINISH_{side}"].position == plain[f"AC_AB_FINISH_{side}"].position


def test_the_road_surface_takes_the_one_closest_below_where_a_road_runs_under_the_deck():
    def node(name, y):
        pos = np.array([[0.0, y, 0.0], [10.0, y, 0.0], [0.0, y, 10.0]], np.float32)
        return {"class": 2, "name": name, "vertices": {"pos": pos}, "indices": np.array([0, 1, 2])}

    road = validate.RoadSurface([node("1ROAD_000", 10.0), node("1ROAD_X000", 2.0), node("1GRASS_T000", 12.0)])
    assert road.height(2.0, 2.0) == pytest.approx(10.0)  # the highest road, never the grass
    assert road.height(2.0, 2.0, below=10.3) == pytest.approx(10.0)  # a spawn or AI point on the deck
    assert road.height(2.0, 2.0, below=5.0) == pytest.approx(2.0)  # a point under the deck: the road below
    assert road.height(2.0, 2.0, below=11.0, within=2.0) == pytest.approx(10.0)
    assert road.height(2.0, 2.0, below=6.0, within=2.0) is None  # a gate's end over the road below: off the road


def test_a_road_drawn_under_the_open_deck_runs_on_under_it_with_no_hole_round_it():
    """With the deck's span out of the footprint (`core.crossing_roads` clips its roads against it), a road under the
    route's bridge is drawn whole, the points under the deck keep off it, and the deck still passes its checks."""
    import math

    from mapcreator.core import crossing_roads as cr

    dem = gully()
    c, sec = road(dem)
    line = [[200.0, d] for d in np.linspace(-150.0, 150.0, 31)]
    under = {"id": 0, "at_m": 200.0, "xy": [200.0, 0.0], "relation": "under", "angle_deg": 90.0, "way_id": 7,
             "highway": "secondary", "name": "Gully Road", "level": 0.0, "bridge": False, "tunnel": False,
             "tags": {"highway": "secondary"}, "line": line, "line_at_m": 150.0,
             "route": {"way_id": 1, "level": 1.0, "bridge": True, "tunnel": False},
             "road": {"line": line, "at_m": 150.0, "ends": ["length", "length"],
                      "ways": [{"way_id": 7, "from_m": 0.0, "to_m": 300.0, "bridge": False, "tunnel": False,
                                "level": 0.0, "highway": "secondary"}]}}
    p = cr.plan([under], c, sec, dem)
    assert p["roads"], p
    plan = bridges.plan(c)
    decks = bridges.Decks(plan, c, sec)
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], decks.embanked(dem), **decks.verges())
    extra = decks.footprint(rb, None)
    taken = mesh.terrain_region(rb, 120.0, extra)[1]
    b = cr.build(p, taken, FRAME, dem, skirt=True)
    (piece,) = b.pieces
    assert piece["length_m"] == pytest.approx(p["roads"][0]["length_m"], abs=0.5)  # whole, under the deck too
    extra = cr.footprint(extra, b, len(c["s"]))
    assert extra.spans  # the deck's span stays out of the footprint
    ground = decks.ground(dem)
    extra = decks.footprint_points(extra, ground)
    meshes = mesh.ribbon_meshes(rb, c["s"], FRAME, cut=decks.cut(None), kerb=decks.kerb()) + b.meshes
    meshes += mesh.terrain_meshes(rb, 120.0, ground, FRAME, extra=extra)
    deck_meshes, _ = decks.meshes(rb, FRAME, ground)
    road_poly = shapely.Polygon(piece["outline"]).buffer(0)
    mine = np.vstack([np.asarray(pts) for pts, _h in extra.edges[-3:]])  # the abutments' feet and the points under
    assert len(mine) and (np.abs(mine[:, 1]) < 6.0).all()
    assert (shapely.distance(road_poly, shapely.points(mine)) >= bridges.UNDER_CLEAR_OF_M - 1e-6).all()
    around = validate._grid_in(shapely.box(150.0, -30.0, 250.0, 30.0), 0.71)
    covers = validate._Cover(triangles(meshes + deck_meshes))
    assert np.isfinite(covers.top(ac(around)[:, [0, 2]])).all()
    assert validate.bridge_failures(validate.mesh_nodes(meshes + deck_meshes),
                                    [(0, chk) for chk in bridges.checks(decks.records(rb), FRAME)]) == {0: []}
    # Where the two overlap in plan, a point on the deck reads the deck, and one below it the road under it.
    surface = validate.RoadSurface(validate.mesh_nodes(meshes)["nodes"])
    assert surface.height(200.0, 0.0, below=DECK + 0.3) == pytest.approx(DECK, abs=0.01)
    low = surface.height(200.0, 0.0, below=DECK - 5.0)
    assert low is not None and math.isclose(low, DECK - 10.0, abs_tol=0.5)


def test_the_terrain_s_footprint_leaves_the_open_span_out_for_what_runs_under_the_deck():
    """The railways and paths under the route's bridges (`core.rail_and_paths`) and the roads (`core.crossing_roads`)
    keep off `mesh.terrain_region`'s footprint: with the open deck's span out of it, they run on under the deck."""
    dem = gully()
    c, sec = road(dem)
    decks = bridges.Decks(bridges.plan(c), c, sec)
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], decks.embanked(dem), **decks.verges())
    before = mesh.terrain_region(rb, 1.0)[1]
    after = mesh.terrain_region(rb, 1.0, decks.footprint(rb, None))[1]
    under = shapely.points([[200.0, 0.0], [200.0, 4.0], [190.0, -2.0]])
    beside = shapely.points([[100.0, 0.0], [300.0, 3.0], [166.0, 0.0]])  # the road on the ground, a closed end
    assert shapely.contains(before, under).all()  # today's embankment
    assert not shapely.contains(after, under).any() and shapely.contains(after, beside).all()
