"""Shallow forks (`core.forks`): side roads under 15 degrees are junctions now, with two exceptions (the shallow-Y
research's Q2); one under 25 degrees is followed out 150 m (Q3); the traffic beside its gore picks the paint (Q5, Q6).
The plan: a gore stub (the slip ramps' taper, painted gore, nose island and stub, `core.ramps`) for each shallow side
road and each whose stub was too skewed to flare, ending 100 m in from the road's edge line, planned after everything
else; what's left out and why (Q8). The meshes: the throat joined to the road's edge and the stub's first row without a
crack or an overlap, the terrain round it with no hole, the closure past the nose, the checks; and the section stage,
which leaves the half widths, and so the AI line, as they were."""

import math

import numpy as np
import pytest
import shapely
from test_ramps import (FRAME, GROUND, HALF, HOBART, IDENT, LINK, _Region, ac_xz, coverage, flat, kn5_nodes, network,
                        out_at, road, route_of, triangles)
from test_ramps import ramp as slip_ramp

from mapcreator import config
from mapcreator.core import (ai_line, forks, junctions, mesh, police_car, ramps, stub_closures, stubs, validate)
from mapcreator.core.layout import RoadModel
from mapcreator.job import Job
from mapcreator.pipeline import route_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.osm import graph

ONE_WAY = {"highway": "primary", "oneway": "yes"}
INTO = {"highway": "primary", "oneway": "-1"}  # one-way against its nodes: towards the road
TWO_WAY = {"highway": "primary"}


# -- the junction finder ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("deg,back,tags,main,expect", [
    (8.0, False, ONE_WAY, None, ("diverge", "ahead", "out", False)),
    (8.0, True, INTO, None, ("merge", "behind", "in", False)),
    # A divided road's other carriageway where it splits off: one-way towards the node, beside traffic leaving it.
    (-8.0, False, INTO, None, ("split", "ahead", "out", True)),
    # A two-way fork off a two-way road: its near lane runs in, the road's lane beside the gore out.
    (8.0, False, TWO_WAY, {"highway": "primary"}, ("two_way", "ahead", "out", True)),
    # Off a one-way road to the right, a two-way fork's near lane runs out with the road's traffic: chevrons.
    (-8.0, False, TWO_WAY, None, ("two_way", "ahead", "out", False)),
])
def test_a_shallow_side_road_is_a_junction_with_its_150_m_leg_and_the_traffic_beside_its_gore(deg, back, tags, main,
                                                                                            expect):
    g = network([(300, out_at(300, deg, back=back), tags)], main=main)
    res = junctions.find(g, route_of(g))
    (j,) = res["junctions"]
    assert res["skipped"] == [] and junctions.gore_only(j) and j["off_mainline_deg"] == pytest.approx(8.0, abs=0.5)
    f = j["fork"]
    assert (f["kind"], f["along"], f["flow"], f["opposing"]) == expect
    assert j["length_m"] == pytest.approx(junctions.LEG_M) and f["length_m"] == pytest.approx(junctions.SHALLOW_LEG_M)
    assert f["line"][0] == j["line"][0] and f["ends"] == "length" and f["way_ids"] == [100]
    assert f["main"]["highway"] == (main or {"highway": "trunk"})["highway"]
    assert res["rule"]["shallow_deg"] == junctions.SHALLOW_DEG and res["rule"]["end_clear_m"] == junctions.END_CLEAR_M


def test_a_fork_the_route_hides_is_found_but_not_the_road_the_route_was_spliced_into():
    # At 5 degrees its first 2 m lie within 0.25 m of the route: the finder used to take it for the route itself.
    g = network([(300, out_at(300, 5.0), {"highway": "residential"})])
    (j,) = junctions.find(g, route_of(g))["junctions"]
    assert j["at_m"] == pytest.approx(290.0) and "fork" in j
    assert ramps.find(g, route_of(g))["ramps"] == []  # not a link: no slip ramp
    # The main road carrying on past the route's spliced start isn't one.
    res = junctions.find(g, route_of(g, start=130.0))
    assert [j["at_m"] for j in res["junctions"]] == [pytest.approx(170.0)] and res["skipped"] == []


def test_the_exceptions_route_end_parallel_a_ramp_and_a_ring_the_route_drives():
    # The road carrying on past the route's start (5 m from it): its end wall closes it.
    g = network([(100, out_at(100, 8.0), {"highway": "residential"})])
    res = junctions.find(g, route_of(g, start=95.0))
    assert res["junctions"] == [] and [s["reason"] for s in res["skipped"]] == ["route_end"]
    # A road running on beside the route, 6 m out: its edge never parts from the road's by a nose's width.
    g = network([(300, [(330.0, 6.0), (600.0, 6.0)], {"highway": "residential"})])
    res = junctions.find(g, route_of(g))
    assert res["junctions"] == [] and [s["reason"] for s in res["skipped"]] == ["parallel"]
    # A link is a slip ramp (`core.ramps`).
    g = network([(300, out_at(300, 8.0), LINK)])
    assert [s["reason"] for s in junctions.find(g, route_of(g))["skipped"]] == ["ramp"]


def test_a_shallow_side_road_onto_the_route_s_roundabout_within_150_m_is_the_ring_s():
    # A ring round (500, 20) the route touches at 500 m, and a side road from 400 m that winds onto it 112 m out.
    th = np.radians(np.arange(225.0, -90.0, -45.0))
    ring = [(500 + 20 * np.cos(t), 20 + 20 * np.sin(t)) for t in th] + [5]
    side = [(420.0, 3.0), (440.0, 8.0), (460.0, 14.0), (465.0, 30.0), (470.0, 40.0), (480.0, 20.0)]
    nodes = {i: (100.0 * i, 0.0) for i in range(11)}
    ways = [(1, list(range(11)), {"highway": "secondary"})]
    refs, nid = [], 1000
    for p in ring[:-1]:
        nodes[nid] = p
        refs.append(nid)
        nid += 1
    ways.append((2, [5] + refs + [5], {"highway": "secondary", "junction": "roundabout", "oneway": "yes"}))
    side_refs = [4]
    for p in side[:-1]:
        nodes[nid] = p
        side_refs.append(nid)
        nid += 1
    side_refs.append(1001)  # the ring's node at (480, 20)
    ways.append((3, side_refs, {"highway": "residential"}))
    els = [{"type": "node", "id": i, "lon": x, "lat": y} for i, (x, y) in nodes.items()]
    els += [{"type": "way", "id": w, "nodes": r, "tags": t} for w, r, t in ways]
    g = graph.build({"elements": els}, IDENT)
    res = junctions.find(g, graph.route(g, np.array([[10.0, 0.0], [990.0, 0.0]])).xy)
    assert [(s["at_m"], s["reason"]) for s in res["skipped"]] == [(390.0, "roundabout")]


# -- the plan -----------------------------------------------------------------------------------------------------------

def fork_junction(x, deg, *, jid=0, kind="diverge", flow="out", opposing=False, length=150.0, back=False, line=None,
                  main=None, highway="primary", closure="major", oneway="yes", width=4.0):
    """A junctions.json junction leaving the road's node at x at `deg` (+ left), ahead (or behind with `back`), with
    its `fork`: a straight line, or `line`."""
    a = math.radians(deg)
    sgn = -1.0 if back else 1.0
    if line is None:
        line = [[x + sgn * d * math.cos(a), d * math.sin(a)] for d in np.linspace(0.0, length, 31)]
    line = [list(map(float, p)) for p in line]
    cut = [p for p in line if math.hypot(p[0] - x, p[1]) <= 100.0]
    return {"id": jid, "node": 7 + jid, "way_id": 99 + jid, "at_m": x, "xy": [x, 0.0],
            "side": "left" if deg > 0 else "right", "angle_deg": deg if not back else math.copysign(180 - abs(deg), deg),
            "off_mainline_deg": abs(deg), "route_turn_deg": 0.0, "line": cut, "length_m": 100.0, "ends": "length",
            "highway": highway, "closure": closure, "name": "Fork Road", "width_m": width,
            "tags": {"highway": highway, **({"oneway": oneway} if oneway else {})},
            "fork": {"kind": kind, "along": "behind" if back else "ahead", "flow": flow, "opposing": opposing,
                     "main": main or {"highway": "trunk", "maxspeed": "80"}, "length_m": length, "ends": "length",
                     "way_ids": [99 + jid], "line": line}}


def bent(x, d0, a0, a1, length=150.0):
    """A line leaving (x, 0) at a0 degrees for d0 m, then on at a1 degrees: a road curving away and then running
    nearly alongside, as a highway's diverge does."""
    pts = [[x, 0.0]]
    for d in np.arange(1.0, length + 1e-9, 1.0):
        a = math.radians(a0 if d <= d0 else a1)
        pts.append([pts[-1][0] + math.cos(a), pts[-1][1] + math.sin(a)])
    return pts


def plan_forks(js, c=None, sec=None, **kw):
    if c is None:
        c, sec = road()
    st = stubs.plan(js, c, sec, flat())
    return st, forks.plan(js, st, c, sec, flat(), **kw)


def test_a_shallow_junction_gets_a_gore_stub_ending_100_m_in_from_the_road_s_edge_not_a_stub():
    j = fork_junction(300.0, 6.0)
    st, fp = plan_forks([j])
    assert st["stubs"] == [] and st["skipped"] == [{"junction": 0, "name": "Fork Road", "reason": "shallow"}]
    (r,) = fp["ramps"]
    assert (r["junction"], r["fork"], r["kind"], r["flow"], r["opposing"]) == (0, True, "diverge", "out", False)
    assert r["closure"] == "major" and fp["skipped"] == []
    # The pieces where the edges say (as a slip ramp's): the nose where the gap is 4.5 m.
    w, t = r["width_m"] / 2, math.tan(math.radians(6.0))  # a two-lane one-way primary: 7 m
    assert r["taper"]["split_m"] == pytest.approx(300 + (HALF + w) / t, abs=2.0)
    assert r["taper"]["nose_m"] == pytest.approx(300 + (HALF + w + ramps.NOSE_W_M) / t, abs=2.5)
    # Q3: its stub ends 100 m in from the road's edge line along its centreline, not 60 m past the nose.
    edge_d = HALF / math.sin(math.radians(6.0))
    assert r["rows_from_m"] + r["length_m"] == pytest.approx(edge_d + forks.END_D_M, abs=1.5)
    assert r["length_m"] - r["ribbon_from_m"] != pytest.approx(ramps.STUB_M, abs=1.0)


def bars(r):
    """The gore's bars (all but its two edge lines, the longest pieces of paint), as polygons."""
    polys = [shapely.Polygon(np.asarray(p["exterior"])[:, :2]) for p in r["gore"]["polygons"]]
    return sorted(polys, key=lambda p: -p.length)[2:]


def test_the_paint_follows_the_traffic_beside_the_gore():
    main = {"highway": "trunk", "maxspeed": "60"}
    _, fp = plan_forks([fork_junction(300.0, 6.0, main=main)])
    (diverge,) = fp["ramps"]
    assert diverge["gore"]["paint"] == "chevrons" and diverge["gore"]["spacing_m"] == 6.0
    for bar in bars(diverge):  # traffic runs from the split to the nose: the points face the split (west)
        xs = np.asarray(bar.exterior.coords)
        assert xs[np.argmin(xs[:, 0]), 1] == pytest.approx((HALF + xs[:, 1].max()) / 2, abs=1.2)
    # Q6: a divided road's other carriageway splits opposing traffic: diagonal stripes.
    _, fp = plan_forks([fork_junction(300.0, -6.0, kind="split", flow="out", opposing=True)])
    assert fp["ramps"][0]["gore"]["paint"] == "diagonals" and fp["ramps"][0]["kind"] == "split"
    # Traffic running in beside the gore (a merge, or a two-way fork whose near lane runs in): the points face the nose.
    _, fp = plan_forks([fork_junction(600.0, 6.0, kind="two_way", flow="in", opposing=False, back=True, main=main,
                                      oneway=None)])
    (merge,) = fp["ramps"]
    assert merge["gore"]["paint"] == "chevrons" and merge["along"] == "behind"
    for bar in bars(merge):  # from the nose (west) to the split (east): the point faces west, the nose
        xs = np.asarray(bar.exterior.coords)
        assert xs[np.argmin(xs[:, 0]), 1] == pytest.approx((HALF + xs[:, 1].max()) / 2, abs=1.2)
    # No paint on gravel or an unmarked road.
    _, fp = plan_forks([fork_junction(300.0, 6.0, main={"highway": "trunk", "lane_markings": "no"})])
    assert fp["ramps"][0]["gore"]["paint"] == "none"


def test_a_side_road_too_skewed_to_flare_falls_back_to_a_gore_and_one_that_flares_keeps_its_stub():
    # 20 degrees 15 m out, then on at 6: the finder's angle is over 15, but its edges part at about 6 degrees.
    skewed = fork_junction(400.0, 20.0, line=bent(400.0, 15.0, 20.0, 6.0))
    skewed["off_mainline_deg"] = 20.0
    st, fp = plan_forks([skewed])
    assert st["skipped"] == [{"junction": 0, "name": "Fork Road", "reason": "skew"}]
    (r,) = fp["ramps"]
    assert r["fork"] and r["junction"] == 0 and r["gore"]["paint"] == "chevrons"
    # At 22 degrees straight out its stub flares as today: no gore, and its stub is exactly as without forks.
    plain = fork_junction(400.0, 22.0)
    plain["off_mainline_deg"] = 22.0
    st, fp = plan_forks([plain])
    assert len(st["stubs"]) == 1 and fp is None
    without = {k: v for k, v in plain.items() if k != "fork"}
    assert stubs.plan([without], *road(), flat()) == st
    assert forks.candidates([plain], st["skipped"]) == []


def test_what_is_left_out_and_why():
    c, sec = road()
    # Q8: its nose beyond its 150 m leg: left out, its junction's mouth kept clear as any junction's.
    _, fp = plan_forks([fork_junction(300.0, 2.5)])
    assert fp["ramps"] == [] and fp["skipped"][0]["reason"] in ("parallel", "short") and fp["skipped"][0]["fork"]
    # A side road's stub on the same side, its mouth sharing the road's rows: the stub stays, the gore goes.
    side_road = {"id": 1, "at_m": 360.0, "xy": [360.0, 0.0], "side": "left", "angle_deg": 90.0,
                 "off_mainline_deg": 90.0, "line": [[360.0, d] for d in np.linspace(0, 100, 21)], "length_m": 100.0,
                 "ends": "length", "highway": "residential", "closure": "minor", "name": "Side Road",
                 "tags": {"highway": "residential"}, "width_m": 5.8}
    st, fp = plan_forks([fork_junction(300.0, 6.0), side_road], c, sec)
    assert [s["junction"] for s in st["stubs"]] == [1]
    assert fp["ramps"] == [] and fp["skipped"] == [{"junction": 0, "name": "Fork Road", "reason": "overlaps",
                                                    "fork": True}]
    # A slip ramp already planned there: the ramp stays (`planned`), the gore goes.
    ramp = ramps.plan([slip_ramp(300.0, 6.0)], c, sec, flat())["ramps"]
    assert len(ramp) == 1
    _, fp = plan_forks([fork_junction(320.0, 7.0)], c, sec, ramps_planned=ramp)
    assert fp["ramps"] == [] and fp["skipped"][0]["reason"] == "overlaps"
    # On the other side it fits.
    _, fp = plan_forks([fork_junction(320.0, -7.0)], c, sec, ramps_planned=ramp)
    assert len(fp["ramps"]) == 1


def test_merged_keeps_the_slip_ramps_document_exactly_without_forks():
    c, sec = road()
    ramp_doc = ramps.plan([slip_ramp(600.0, 6.0)], c, sec, flat())
    assert forks.merged(ramp_doc, None) is ramp_doc and forks.merged(None, None) is None
    _, fp = plan_forks([fork_junction(300.0, 6.0)])
    both = forks.merged(ramp_doc, fp)
    assert [r["junction"] for r in both["ramps"]] == [0, ramps.RAMP_ID_BASE]  # in order along the road
    assert both["rule"]["fork"] == {"end_d_m": 100.0, "shallow_deg": 25.0, "shallow_leg_m": 150.0}
    assert {k: v for k, v in both["rule"].items() if k != "fork"} == ramp_doc["rule"]
    alone = forks.merged(None, fp)
    assert [r["junction"] for r in alone["ramps"]] == [0] and forks.summary(alone) == {"gores": 1, "skipped": {}}


# -- the meshes ---------------------------------------------------------------------------------------------------------

def built(js, dem_h=GROUND, skirt=True):
    c, sec = road()
    _, fp = plan_forks(js, c, sec)
    plan = fp["ramps"]
    caps = stubs.mainline_caps([], c["xy"], c["heading"], extra=ramps.centres(plan))
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], flat(dem_h), skirt_cap_l=caps[0], skirt_cap_r=caps[1])
    b = ramps.build(plan, rb, c["s"], FRAME, flat(dem_h), skirt=skirt, others=c["xy"])
    cut, edge_break = ramps.road_cut(None, None, b)
    meshes = mesh.ribbon_meshes(rb, c["s"], FRAME, style=sec["style"], skirt=skirt, cut=cut,
                                edge_break=edge_break) + b.meshes
    terrain = mesh.terrain_meshes(rb, 150.0, flat(dem_h), FRAME, fold_skirt=not skirt,
                                  extra=ramps.footprint(None, b, len(c["s"])))
    return c, sec, plan, rb, b, meshes, terrain


def test_fork_meshes_are_named_by_their_junction_and_valid():
    c, sec, plan, rb, b, meshes, _ = built([fork_junction(400.0, 7.0, jid=3),
                                            fork_junction(700.0, -9.0, jid=4, kind="split", opposing=True)])
    names = sorted(m.name for m in b.meshes)
    assert "1ROAD_JF3_G" in names and "1GRASS_JF3_I" in names and "1ROAD_JF4_000" in names
    assert all("_J" in n for n in names) and not any("_JS" in n for n in names)
    for m in b.meshes:
        m.validate()
    assert b.stats["ramps"] == 2 and b.stats["gores"] == 2 and b.stats["painted"] == 2
    assert ramps.mesh_tag(plan[0]) == "JF3_" and ramps.mesh_tag({"junction": ramps.RAMP_ID_BASE + 1}) == "JS1_"


@pytest.mark.parametrize("deg,kind,back", [(6.0, "diverge", False), (-10.0, "split", False), (8.0, "merge", True)])
def test_the_throat_joins_the_road_s_edge_and_the_stub_without_a_crack_or_an_overlap(deg, kind, back):
    c, sec, plan, rb, b, meshes, _ = built([fork_junction(400.0, deg, kind=kind, back=back,
                                                          flow="in" if back else "out")])
    (r,) = plan
    assert b.failed == {}
    side = junctions.SIDE_CODE[r["side"]]
    taper_row, _, nose_row = r["taper"]["rows"]
    lo, hi = sorted((taper_row, nose_row))
    col = rb.road_cols.start if side == "L" else rb.road_cols.stop - 1
    edge = FRAME.to_ac(rb.e[lo:hi + 1, col], rb.n[lo:hi + 1, col], rb.h[lo:hi + 1, col]).astype(np.float32)
    verts = {tuple(v) for v in triangles(b.meshes, "1ROAD_JF0_G").reshape(-1, 3).astype(np.float32).tolist()}
    assert all(tuple(v) in verts for v in edge.tolist())
    paved = shapely.Polygon(r["outline"]).buffer(0)
    minx, miny, maxx, maxy = paved.bounds
    xs, ys = np.meshgrid(np.arange(minx + 0.013, maxx, 0.41), np.arange(min(miny, -HALF) + 0.217, maxy, 0.41))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    inside = (shapely.contains_xy(paved.buffer(-0.03), pts[:, 0], pts[:, 1])
              | ((np.abs(pts[:, 1]) < HALF - 0.05) & (pts[:, 0] > minx) & (pts[:, 0] < maxx)))
    road_tris = triangles(meshes, "1ROAD")
    count = coverage(road_tris, ac_xz(pts[inside]))
    assert (count <= 1).all()  # nothing drawn twice
    on_edge = pts[inside][count == 0]  # a sample on an edge two triangles share
    assert len(on_edge) <= 3 and (coverage(road_tris, ac_xz(on_edge), eps=-1e-4) >= 1).all()


@pytest.mark.parametrize("skirt", [True, False])
def test_the_terrain_fits_round_the_gore_stub_and_its_nose_with_no_gap_and_nothing_over_it(skirt):
    c, sec, plan, rb, b, meshes, terrain = built([fork_junction(400.0, 7.0)], dem_h=GROUND - 0.6, skirt=skirt)
    (r,) = plan
    every = meshes + terrain
    paved = shapely.Polygon(r["outline"]).buffer(0)
    island = shapely.Polygon(r["island"]["outline"]).buffer(0)
    around = validate._grid_in(shapely.union(paved, island).buffer(3.0), 0.43)
    assert (coverage(triangles(every), ac_xz(around)) >= 1).all()  # no hole
    on = validate._grid_in(paved.buffer(-0.05), 0.61)
    assert (coverage(triangles(every, "1GRASS"), ac_xz(on)) == 0).all()  # no ground under or over its road
    on_nose = validate._grid_in(island.buffer(-0.3), 0.53)
    assert (coverage(triangles(every, "1GRASS"), ac_xz(on_nose)) == 1).all()  # Q4: mown grass at the nose, once
    assert (coverage(triangles(every, "1ROAD"), ac_xz(on_nose)) == 0).all()  # no kerb, no raised island


def test_the_checks_pass_and_the_closure_stands_past_the_nose_with_the_kit_its_class_takes():
    c, sec, plan, rb, b, meshes, terrain = built([fork_junction(400.0, 10.0)], dem_h=GROUND - 0.4)
    (r,) = plan
    checks = ramps.checks(plan, c, sec, FRAME)
    assert [k for k, _ in checks] == ["ramp:0", "ramp:0:island"]
    assert validate.stub_failures(kn5_nodes(meshes + terrain), checks) == {"ramp:0": [], "ramp:0:island": []}
    top = stub_closures.TopSurface(meshes + terrain, FRAME, stub_closures.areas(plan))
    closed = stub_closures.place(plan, ramps.junctions_for(plan), c, sec, FRAME, top,
                                 jurisdiction=police_car.DEFAULT_JURISDICTION)
    (rec,) = closed.records
    assert rec["junction"] == 0 and rec["level"] == "major" and rec["obstacle"] == "boards"
    assert set(rec["elements"]) == {"tape", "obstacle", "warning"}  # 100 m in from the edge leaves room for it all
    assert all(e["d_m"] >= r["ribbon_from_m"] and not e["in_throat"] for e in rec["elements"].values())
    assert closed.stats["cars"] == 1
    found = stubs.faults([], meshes + terrain, closed.meshes, c, sec, FRAME, extra=checks)
    assert ramps.faults_of(found) == {}


# -- the section stage --------------------------------------------------------------------------------------------------

def test_the_section_stage_plans_the_gores_and_leaves_the_half_widths_and_the_ai_line_byte_for_byte(tmp_path,
                                                                                                    monkeypatch):
    monkeypatch.setattr(section_stage, "job_region", lambda waypoints: _Region())
    work = WorkDir(tmp_path / "work")
    c, _ = road()
    np.savez_compressed(work.stage("02_profile") / "centreline.npz", run=np.zeros(len(c["s"]), np.int64), **c)
    work.write_json(route_stage.STAGE, "runs.json", [{"tags": {"highway": "trunk", "surface": "asphalt"}}])
    work.write_json(route_stage.STAGE, route_stage.JUNCTIONS, {"version": 1, "junctions": [fork_junction(400.0, 7.0)]})
    work.write_json(route_stage.STAGE, route_stage.RAMPS, {"version": 1, "ramps": []})
    job = Job(track_id="t", waypoints=HOBART)
    out = section_stage.run(job, config.load(), work)
    assert out["forks"] == {"gores": 1, "skipped": {}} and "ramps" not in out
    assert out["junction_stubs"]["skipped"] == {"shallow": 1}
    doc = work.read_json(section_stage.STAGE, section_stage.RAMPS)
    assert doc["crs"] == "EPSG:28355" and doc["rule"]["fork"]["end_d_m"] == forks.END_D_M
    assert [(r["junction"], r["fork"]) for r in section_stage.ramps(work, job)] == [(0, True)]
    with_forks = dict(np.load(work.root / section_stage.STAGE / "section.npz"))
    none = Job(track_id="t", waypoints=HOBART, junctions="none")
    out = section_stage.run(none, config.load(), work)
    assert "forks" not in out and not (work.root / section_stage.STAGE / section_stage.RAMPS).exists()
    without = dict(np.load(work.root / section_stage.STAGE / "section.npz"))
    assert all(np.array_equal(with_forks[k], without[k]) for k in with_forks)
    files = []
    for sec in (with_forks, without):
        rm = RoadModel.from_arrays(c["s"], c["xy"], c["z"], c["heading"], c["curvature"], c["grade"],
                                   sec["half_left"], sec["half_right"], sec["slope_left"], sec["slope_right"], False)
        path = tmp_path / f"fast_lane_{len(files)}.ai"
        ai_line.write(path, ai_line.build(rm, FRAME, end_short_m=40.0))
        files.append(path.read_bytes())
    assert files[0] == files[1]


def test_junctions_nose_width_is_the_ramps():
    assert junctions.NOSE_W_M == ramps.NOSE_W_M
