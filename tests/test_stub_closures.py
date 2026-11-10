"""Closures on the side roads' stubs (`core.stub_closures`): where each stands (its origin moved in on a skewed
junction, every line spanning the throat's paved outline), the minor obstacle per junction and bales kept off the
road, the overshoot rule, the police cars (which majors, their spacing and cap, the jurisdiction, clear of the road),
the merged meshes, the ground they stand on, the track stage's material values and the validator's checks."""

import math

import numpy as np
import pytest
import shapely

from mapcreator.core import closures, police_car, section, stub_closures, stubs, validate
from mapcreator.core.scene import MAX_VERTICES, LocalFrame, Material, Mesh, Scene
from mapcreator.pipeline import track_stage

FRAME = LocalFrame(0.0, 0.0, 0.0)
HALF = 3.2
GROUND = 10.0


def road(n=1201):
    """A straight road east along y = 0, a station every metre, level, 6.4 m wide with a crown."""
    s = np.arange(n, dtype=float)
    c = {"s": s, "xy": np.column_stack([s, np.zeros(n)]), "heading": np.zeros(n), "curvature": np.zeros(n),
         "z": np.full(n, GROUND), "grade": np.zeros(n)}
    sec = {"half_left": np.full(n, HALF), "half_right": np.full(n, HALF), "slope_left": np.full(n, section.CROWN),
           "slope_right": np.full(n, section.CROWN), "style": np.zeros(n, np.int8)}
    return c, sec


def junction(x, deg, *, jid=0, highway="residential", closure="minor", node=None, way=None, turn=0.0, length=100.0):
    a = np.radians(deg)
    line = [[x + d * np.cos(a), d * np.sin(a)] for d in np.linspace(0, length, 21)]
    return {"id": jid, "node": node if node is not None else 1000 + jid, "way_id": way if way is not None else 5000 + jid,
            "at_m": x, "xy": [x, 0.0], "side": "left" if deg > 0 else "right", "angle_deg": deg,
            "route_turn_deg": turn, "line": line, "length_m": length, "ends": "length", "highway": highway,
            "closure": closure, "name": f"Side Road {jid}", "tags": {"highway": highway}, "width_m": 5.8}


def flat_top(e, n):
    return np.full(np.shape(np.asarray(e, float)), GROUND)


def placed(js, *, c=None, sec=None, **kw):
    if c is None:
        c, sec = road()
    plan = stubs.plan(js, c, sec, lambda x, y: np.full(np.shape(x), GROUND))["stubs"]
    return plan, stub_closures.place(plan, js, c, sec, FRAME, flat_top, **kw)


def xy_of(meshes):
    if not meshes:
        return np.zeros((0, 2))
    p = np.concatenate([m.positions for m in meshes]).astype(float)
    e, n, _ = FRAME.from_ac(p)
    return np.column_stack([e, n])


def bales_obstacle_junction(start=0):
    """A junction whose obstacle is bales (by its hash)."""
    k = start
    while stub_closures.obstacle_for({"node": 1000 + k, "way_id": 5000 + k}) != "bales":
        k += 1
    return k


# -- where a closure stands --------------------------------------------------------------------------------------

def test_a_square_stub_is_closed_4_m_in_with_its_tape_across_the_throat_and_the_obstacle_12_m_in():
    plan, p = placed([junction(300.0, 90.0, closure="major", highway="tertiary")])
    (rec,), (st,) = p.records, plan
    assert rec["origin_m"] == 0.0 and (rec["level"], rec["obstacle"]) == ("major", "boards")
    tape, ob, warn = (rec["elements"][k] for k in ("tape", "obstacle", "warning"))
    assert (tape["d_m"], ob["d_m"], warn["d_m"]) == (4.0, 12.0, 37.0)
    # The tape stands in the throat, where the flares widen the road: it spans them, not just the side road.
    assert tape["in_throat"] and tape["width_m"] > st["width_m"] + 2 * stub_closures.VERGE_SPAN_M + 1.0
    assert tape["offset_m"] == pytest.approx(0.0, abs=0.5)  # a square junction's flares alike, to a row
    # Past the throat the props span the side road and a metre of each verge.
    assert ob["width_m"] == pytest.approx(st["width_m"] + 2 * stub_closures.VERGE_SPAN_M, abs=0.05)
    assert not ob["in_throat"]


def test_a_skewed_stub_s_closure_moves_in_until_its_tape_keeps_clear_of_the_road():
    plan, p = placed([junction(300.0, 35.0)])
    (rec,) = p.records
    assert rec["origin_m"] > 5.0
    assert rec["elements"]["tape"]["d_m"] == rec["origin_m"] + closures.TAPE_M
    # Every prop, stakes included, stands at least TAPE_CLEAR_M beyond the road's edge (y = HALF).
    y = xy_of([m for m in p.meshes if m.name.startswith("CLOSURE_")])[:, 1]
    assert y.min() >= HALF + stub_closures.TAPE_CLEAR_M - 0.05


def test_a_line_in_a_skewed_throat_spans_the_paved_outline_off_centre():
    plan, p = placed([junction(300.0, 50.0)])
    (rec,), (st,) = p.records, plan
    tape = rec["elements"]["tape"]
    assert tape["in_throat"] and abs(tape["offset_m"]) > 0.5
    # The outline's extent along the tape's line lies inside the props' span, which reaches a metre past it.
    pose = stubs.pose(st)
    e, n, _, hd = pose(tape["d_m"])
    nl = np.array([-math.sin(hd), math.cos(hd)])
    cut = shapely.LineString([(e, n) - 60 * nl, (e, n) + 60 * nl]).intersection(shapely.Polygon(st["outline"]))
    (piece,) = [g for g in shapely.get_parts(cut) if g.distance(shapely.Point(e, n)) < 0.02]
    lat = (np.asarray(piece.coords) - (e, n)) @ nl
    lo, hi = tape["offset_m"] - tape["width_m"] / 2, tape["offset_m"] + tape["width_m"] / 2
    assert lo <= lat.min() - stub_closures.VERGE_SPAN_M + 0.05 and hi >= lat.max() + stub_closures.VERGE_SPAN_M - 0.05


# -- the obstacle and the overshoot rule ---------------------------------------------------------------------------

def test_minor_obstacles_are_boards_or_bales_about_half_each_and_fixed_per_junction():
    picks = [stub_closures.obstacle_for({"node": 7_000_000 + k, "way_id": 90_000 + k}) for k in range(400)]
    assert set(picks) == {"boards", "bales"} and 0.4 < picks.count("bales") / len(picks) < 0.6
    assert picks == [stub_closures.obstacle_for({"node": 7_000_000 + k, "way_id": 90_000 + k}) for k in range(400)]
    # Two side roads at one node choose on their own.
    assert len({stub_closures.obstacle_for({"node": 42, "way_id": w}) for w in range(20)}) == 2


def test_bales_stand_solid_across_a_minor_road_and_become_boards_near_the_road(monkeypatch):
    k = bales_obstacle_junction()
    _, p = placed([junction(300.0, 90.0, jid=k)])
    (rec,) = p.records
    assert rec["obstacle"] == "bales" and "note" not in rec
    walls = [m for m in p.meshes if m.name.startswith("1WALL_CLOSURE_")]
    assert walls and all(not m.renderable for m in walls)
    monkeypatch.setattr(stub_closures, "SOLID_CLEAR_M", 30.0)  # as if the road ran close behind the obstacle
    _, p = placed([junction(300.0, 90.0, jid=k)])
    (rec,) = p.records
    assert rec["obstacle"] == "boards" and "boards instead" in rec["note"]
    assert not [m for m in p.meshes if m.name.startswith("1WALL_")]


def turning_road(n=1201, at=600.0, turn_deg=90.0):
    """A road east to `at`, then turning left by `turn_deg` on a tight curve: a junction at the corner."""
    s = np.arange(n, dtype=float)
    hd = np.radians(turn_deg) * np.clip((s - (at - 5.0)) / 10.0, 0.0, 1.0)
    xy = np.column_stack([np.cumsum(np.cos(hd)), np.cumsum(np.sin(hd))]) - [np.cos(hd[0]), np.sin(hd[0])]
    return s, xy, hd


def test_the_stub_straight_ahead_where_the_route_turns_is_an_overshoot_stub():
    s, _, hd = turning_road()
    # The side road carries straight on east where the route turns 90 degrees left: 45 degrees right of the route's
    # mean direction through the node.
    ahead = {"angle_deg": -45.0, "route_turn_deg": 90.0}
    stub = {"station_m": 600.0}
    assert stub_closures.is_overshoot(ahead, stub, s, hd, (False,))
    assert not stub_closures.is_overshoot(ahead, stub, s, hd, (True,))  # driven the other way it's off to the side
    assert stub_closures.is_overshoot(ahead, stub, s, hd, (False, True))
    side = {"angle_deg": 90.0, "route_turn_deg": 90.0}  # north-west: ahead neither way
    assert not stub_closures.is_overshoot(side, stub, s, hd, (False, True))
    straight = {"angle_deg": 80.0, "route_turn_deg": 5.0}
    assert not stub_closures.is_overshoot(straight, stub, s, hd, (False, True))


def test_an_overshoot_stub_has_nothing_solid_in_its_first_25_m(monkeypatch):
    k = bales_obstacle_junction()
    monkeypatch.setattr(stub_closures, "is_overshoot", lambda *a, **kw: True)
    plan, p = placed([junction(300.0, 90.0, jid=k)])
    (rec,), (st,) = p.records, plan
    assert rec["overshoot"] and rec["elements"]["obstacle"]["d_m"] == rec["origin_m"] + closures.OVERSHOOT_OBSTACLE_M
    walls = xy_of([m for m in p.meshes if m.name.startswith("1WALL_")])
    assert len(walls) and walls[:, 1].min() >= HALF + rec["origin_m"] + closures.OVERSHOOT_CLEAR_M


# -- police cars -------------------------------------------------------------------------------------------------

def test_cars_park_in_major_closures_slots_behind_the_boards_clear_of_the_road():
    plan, p = placed([junction(300.0, 90.0, closure="major", highway="tertiary")], jurisdiction="vic")
    (rec,), (st,) = p.records, plan
    assert rec["car"] and p.stats["cars"] == 1 and p.stats["jurisdiction"] == "vic"
    assert {m.material for m in p.meshes if m.name.startswith("POLICE_")} <= set(police_car.material_names("vic").values())
    car = xy_of([m for m in p.meshes if m.name.startswith("1WALL_POLICE")])
    boards = rec["elements"]["obstacle"]["d_m"]
    # Behind the boards (their back feet 0.3 m behind the line), right of the middle looking in (east is right).
    assert car[:, 1].min() > HALF + boards + closures.TRESTLE_FOOT_M
    assert car[:, 0].mean() > 300.0
    assert car[:, 1].min() - HALF >= stub_closures.SOLID_CLEAR_M
    assert shapely.Polygon(st["outline"]).buffer(stub_closures.CAR_ON_STUB_M).contains(shapely.MultiPoint(car))


def test_cars_go_to_the_biggest_major_side_roads_spaced_apart_up_to_the_cap(monkeypatch):
    monkeypatch.setattr(stub_closures, "CAR_SPACING_M", 150.0)
    monkeypatch.setattr(stub_closures, "CARS_MAX", 3)
    classes = ["tertiary", "secondary", "tertiary", "primary", "tertiary", "tertiary", "secondary", "tertiary"]
    js = [junction(200.0 + 100.0 * k, 90.0 if k % 2 else -90.0, jid=k, highway=h, closure="major")
          for k, h in enumerate(classes)]
    js.append(junction(200.0 + 100.0 * 3 + 40.0, 90.0, jid=8, highway="residential", closure="minor"))
    _, p = placed(js)
    with_car = [r for r in p.records if r["car"]]
    stations = sorted(r["station_m"] for r in with_car)
    assert len(with_car) == 3 and min(np.diff(stations)) >= 150.0
    # The primary first, then the secondaries; tertiaries only where those leave room.
    assert {r["highway"] for r in with_car} == {"primary", "secondary"}
    assert all(r["level"] == "major" for r in with_car)


def test_the_cap_and_spacing_hold_on_a_long_road():
    js = [junction(100.0 + 300.0 * k, 90.0, jid=k, highway="tertiary", closure="major") for k in range(40)]
    c, sec = road(12500)
    _, p = placed(js, c=c, sec=sec)
    stations = sorted(r["station_m"] for r in p.records if r["car"])
    assert len(stations) == stub_closures.CARS_MAX
    assert min(np.diff(stations)) >= stub_closures.CAR_SPACING_M


# -- meshes, materials and the ground ------------------------------------------------------------------------------

def test_merged_meshes_carry_the_validator_s_names_under_the_vertex_limit():
    k = bales_obstacle_junction()
    js = [junction(200.0, 90.0, closure="major", highway="secondary"), junction(400.0, -70.0, jid=k),
          junction(600.0, 80.0, jid=2, closure="limited", highway="unclassified")]
    _, p = placed(js)
    names = [m.name for m in p.meshes]
    assert all(n.startswith(("CLOSURE_", "1WALL_CLOSURE_", "POLICE_", "1WALL_POLICE")) for n in names)
    assert all(len(m.positions) <= MAX_VERTICES for m in p.meshes)
    assert all(m.renderable == (not m.name.startswith("1WALL_")) for m in p.meshes)
    assert set(p.materials) == {"closure", "closure_retro", *police_car.material_names("tas").values()}
    assert p.stats["by_level"] == {"limited": 1, "major": 1, "minor (bales)": 1}
    body = p.materials["police_tas"]
    assert body.properties["ksSpecular"] == police_car.PAINT_PROPS["ksSpecular"]
    assert body.night_diffuse == police_car.NIGHT_DIFFUSE_BY_PART["body"]
    assert set(p.textures) == {closures.TEXTURE, "police_tas"}


def test_top_surface_gives_the_highest_ground_under_each_point():
    def quad(name, h, x0, x1):
        pos = FRAME.to_ac(np.array([x0, x1, x1, x0]), np.array([0.0, 0.0, 10.0, 10.0]), np.full(4, h))
        return Mesh(name, "grass", pos.astype(np.float32), np.tile([0, 1, 0], (4, 1)).astype(np.float32),
                    np.zeros((4, 2), np.float32), np.array([[0, 1, 2], [0, 2, 3]]))
    meshes = [quad("1GRASS_T000", 5.0, 0, 10), quad("1ROAD_J000", 5.2, 4, 6), quad("RAIL_000", 9.0, 0, 10)]
    top = stub_closures.TopSurface(meshes, FRAME, [(-1, -1, 11, 11)])
    h = top(np.array([2.0, 5.0, 20.0]), np.array([5.0, 5.0, 5.0]))
    assert h[0] == pytest.approx(5.0) and h[1] == pytest.approx(5.2) and np.isnan(h[2])
    assert np.isnan(stub_closures.TopSurface(meshes, FRAME, [(50, 50, 60, 60)])(2.0, 5.0))


def test_props_stand_on_the_ground_given():
    _, p = placed([junction(300.0, 90.0)])
    y = np.concatenate([m.positions[:, 1] for m in p.meshes if m.name.startswith("CLOSURE_")])
    # Stakes and legs sink a little into it; nothing floats.
    assert y.min() < GROUND - FRAME.h0 and y.min() > GROUND - FRAME.h0 - closures.STAKE_SINK_M - 0.01


def test_the_track_stage_gives_police_materials_their_values_and_closures_and_police_their_night():
    _, p = placed([junction(300.0, 90.0, closure="major", highway="tertiary")])
    scene = Scene(FRAME, meshes=[], materials=p.materials)
    mats = [type("M", (), {"name": n, "properties": dict(m.properties or track_stage.MATERIAL_PROPS["closure"])})()
            for n, m in p.materials.items()]
    night = track_stage.retro_materials([scene], mats)
    assert night["closure_retro"] == (closures.NIGHT_DIFFUSE["closure_retro"], 0.6)
    assert night["police_tas_lens"][0] == police_car.NIGHT_DIFFUSE_BY_PART["lens"]
    assert night["police_tas"][0] == police_car.NIGHT_DIFFUSE_BY_PART["body"]
    assert night["closure"] == (closures.NIGHT_DIFFUSE["closure"], 0.6)


def test_material_values_survive_a_scene_round_trip(tmp_path):
    m = Material("police_vic", "police_vic", properties={"ksDiffuse": 0.6}, night_diffuse=0.35)
    Scene(FRAME, meshes=[], materials={"police_vic": m, "grass": Material("grass", "grass")}).save(tmp_path)
    back = Scene.load(tmp_path).materials
    assert back["police_vic"] == m and back["grass"].properties is None and back["grass"].night_diffuse is None


# -- the validator -------------------------------------------------------------------------------------------------

def kn5(meshes, spawns=((0.0, 0.0),)):
    nodes = [{"class": 1, "name": "root"}]
    nodes += [{"class": 2, "name": m.name, "renderable": int(m.renderable), "vertices": {"pos": m.positions},
               "indices": m.triangles.astype(np.uint16).ravel()} for m in meshes]
    dummies = {}
    for i, (e, n) in enumerate(spawns):
        mtx = np.eye(4)
        mtx[3, :3] = FRAME.to_ac(e, n, GROUND)
        dummies[f"AC_PIT_{i}"] = {"matrix": mtx}
    return {"nodes": nodes}, dummies


def mainline_mesh(c, sec):
    """The road's carriageway as a 1ROAD mesh, as the validator finds the road's edge."""
    left = c["xy"] + np.column_stack([np.zeros(len(c["s"])), sec["half_left"]])
    right = c["xy"] - np.column_stack([np.zeros(len(c["s"])), sec["half_right"]])
    pos = FRAME.to_ac(*np.vstack([left, right]).T, np.full(2 * len(c["s"]), GROUND)).astype(np.float32)
    n = len(c["s"])
    i = np.arange(n - 1)
    tri = np.concatenate([np.column_stack([i, i + 1, n + i]), np.column_stack([i + 1, n + i + 1, n + i])])
    return Mesh("1ROAD_000", "road", pos, np.tile([0, 1, 0], (2 * n, 1)).astype(np.float32),
                np.zeros((2 * n, 2), np.float32), tri)


def checks(js, plan, p, c, sec, spawns=((0.0, 0.0),), extra=()):
    from mapcreator.core import junctions as jn
    by = stubs.by_junction(plan)
    out = []
    for m in jn.mouths(js, c["s"], c["xy"], c["heading"], sec["half_left"], sec["half_right"], stubs=by):
        to_ac = (lambda g: shapely.transform(g, lambda q: FRAME.to_ac(q[:, 0], q[:, 1], np.zeros(len(q)))[:, [0, 2]]))
        mouth = m.corridor().intersection(jn.side_band(m, c["s"], c["xy"], HALF + jn.MOUTH_BAND_M))
        st = by.get(m.junction["id"])
        out.append(validate.JunctionCheck(m.junction["name"], to_ac(mouth), to_ac(jn.side_road(m, c["s"], c["xy"])),
                                          validate.stub_check(st, c["s"], c["xy"], c["heading"], sec["half_left"],
                                                              sec["half_right"], FRAME) if st is not None else None))
    data, dummies = kn5([mainline_mesh(c, sec), *p.meshes, *extra], spawns)
    rep = validate.Report()
    validate.closure_checks(rep, "main", data, dummies, out)
    return rep


def test_the_validator_passes_placed_closures_and_cars():
    k = bales_obstacle_junction()
    js = [junction(300.0, 90.0, closure="major", highway="tertiary"), junction(700.0, -60.0, jid=k)]
    c, sec = road()
    plan, p = placed(js, c=c, sec=sec)
    rep = checks(js, plan, p, c, sec)
    assert rep.failed == [] and len(rep.passed) == 7, rep.failed


def test_the_validator_fails_solid_props_near_the_road_or_a_spawn_and_warns_of_closures_off_their_stubs():
    js = [junction(300.0, 90.0, closure="major", highway="tertiary")]
    c, sec = road()
    plan, p = placed(js, c=c, sec=sec)
    car = next(m for m in p.meshes if m.name.startswith("1WALL_POLICE"))
    e, n, _ = FRAME.from_ac(car.positions.astype(float).mean(axis=0))
    rep = checks(js, plan, p, c, sec, spawns=((e, n),))
    assert any("from every spawn" in f for f in rep.failed)
    # The same collider moved towards the road until it's 2 m from the edge (AC z runs south: z = -northing).
    nearest = float(xy_of([car])[:, 1].min())
    near = Mesh("1WALL_CLOSURE_9", "closure", car.positions + np.float32([0.0, 0.0, nearest - (HALF + 2.0)]),
                car.normals, car.uvs, car.triangles, renderable=False)
    rep = checks(js, plan, p, c, sec, extra=[near])
    assert any("from the road's edge" in f for f in rep.failed)
    # A closure standing on a side road that has no stub (a junction its stub was left out for).
    js2 = js + [junction(800.0, 90.0, jid=1)]
    stray = Mesh("CLOSURE_9", "closure", car.positions + np.float32([500.0, 0, 0]), car.normals, car.uvs,
                 car.triangles)
    rep = checks(js2, plan, p, c, sec, extra=[stray])
    # A side road's dressing: warnings, never failures (the mesh stage leaves out a stub whose closure is off it).
    assert rep.failed == []
    assert any("stand on side roads' stubs" in w for w in rep.warnings)
    assert any("without a stub" in w for w in rep.warnings)
    rep = checks(js, plan, type(p)([m for m in p.meshes if not m.name.startswith("CLOSURE_")], {}, {}, []), c, sec)
    assert rep.failed == [] and any("every stub closed" in w for w in rep.warnings)


def test_with_no_closures_the_validator_adds_nothing():
    c, sec = road()
    data, dummies = kn5([mainline_mesh(c, sec)])
    rep = validate.Report()
    validate.closure_checks(rep, "main", data, dummies, [])
    assert rep.passed == [] and rep.failed == []
