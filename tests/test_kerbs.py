"""Kerbs and footpaths on town roads, phase 1 (`core.kerbs`): the town-road rule, the KC profile over its hidden ramp,
the kerb returns, the driveways' crossings, the FOOTWAY surface, and building on with no parcels or footprints."""

import numpy as np
import pytest
import shapely

from mapcreator.core import kerbs, mesh, package, section, signals, stubs, validate
from mapcreator.core.layout import RoadModel
from mapcreator.core.scene import LocalFrame
from mapcreator.job import Job
from mapcreator.service import db

FRAME = LocalFrame(0.0, 0.0, 0.0)
GROUND = 10.0


def straight(n=1001):
    s = np.arange(n, dtype=float)
    return s, np.column_stack([s, np.zeros(n)]), np.zeros(n)


def town(x0, x1, every=12.0, offset=20.0):
    """House centroids both sides of the road from x0 to x1, one every `every` metres."""
    xs = np.arange(x0, x1, every)
    return np.vstack([np.column_stack([xs, np.full(len(xs), offset)]),
                      np.column_stack([xs, np.full(len(xs), -offset)])])


def plan_of(tags=None, centroids=None, n=1001, half=3.0, **kw):
    s, xy, heading = straight(n)
    tags = tags if tags is not None else {"highway": "residential"}
    return kerbs.plan(s, xy, heading, np.zeros(n, int), [tags], np.full(n, half), np.full(n, half),
                      centroids=town(300, 700) if centroids is None else centroids, **kw)


# -- the town-road rule ------------------------------------------------------------------------------------------------

def test_maxspeed_is_read_in_km_h_and_au_defaults():
    assert kerbs.maxspeed_kmh({"maxspeed": "50"}) == 50
    assert kerbs.maxspeed_kmh({"maxspeed": "60 km/h"}) == 60
    assert kerbs.maxspeed_kmh({"maxspeed": "AU:urban"}) == 50
    assert kerbs.maxspeed_kmh({"maxspeed": "AU:rural"}) == 100
    assert kerbs.maxspeed_kmh({"maxspeed": "30 mph"}) == pytest.approx(48.3, abs=0.1)
    assert kerbs.maxspeed_kmh({}) is None
    assert kerbs.town_speed({}) and kerbs.town_speed({"maxspeed": "60"}) and not kerbs.town_speed({"maxspeed": "70"})


def test_motorways_and_fast_trunk_roads_are_never_kerbed():
    assert kerbs.never_kerbed({"highway": "motorway"})
    assert kerbs.never_kerbed({"highway": "trunk", "maxspeed": "80"})
    assert not kerbs.never_kerbed({"highway": "trunk", "maxspeed": "60"})
    assert not kerbs.never_kerbed({"highway": "residential"})


def test_gaps_under_100_m_close_and_stretches_under_150_m_drop():
    s = np.arange(1000, dtype=float)
    m = (s >= 100) & (s < 300) | (s >= 360) & (s < 450) | (s >= 700) & (s < 800)
    out = kerbs.stretches(m, s)
    assert out[100:450].all()  # the 60 m gap closed, so the 90 m piece joins the first
    assert not out[700:800].any()  # 100 m on its own: dropped
    hard = np.zeros(1000, bool)
    hard[320] = True  # a bridge in the gap stays bare, and splits it: the 129 m beyond it is then too short
    out = kerbs.stretches(m, s, hard=hard)
    assert out[100:320].all() and not out[320:450].any()


def test_a_town_stretch_is_kerbed_and_the_country_either_side_is_not():
    p = plan_of()
    k = p.kerb["L"]
    assert k[400:600].min() == 1.0
    assert k[:250].max() == 0.0 and k[760:].max() == 0.0
    assert p.summary["kerbed_m"] > 300


@pytest.mark.parametrize("tags", [{"highway": "residential", "maxspeed": "80"}, {"highway": "motorway"},
                                  {"highway": "trunk", "maxspeed": "100"}])
def test_no_kerb_on_a_fast_road_even_through_a_town(tags):
    assert not plan_of(tags).any()


def test_no_kerb_on_unsealed_road_or_with_too_few_buildings():
    n = 1001
    assert not plan_of(unsealed=np.ones(n, bool)).any()
    assert not plan_of(centroids=town(300, 700, every=60.0)).any()
    assert not plan_of(centroids=np.zeros((0, 2))).any()


def test_a_bridge_and_its_margin_stay_bare_inside_a_town():
    n = 1001
    blocked = np.zeros(n, bool)
    blocked[495:505] = True
    k = plan_of(blocked=blocked).kerb["L"]
    assert k[490:510].max() == 0.0  # the bridge and BLOCK_MARGIN_M either side
    assert k[400:470].min() == 1.0 and k[530:600].min() == 1.0


# -- width -------------------------------------------------------------------------------------------------------------

def k2k(tags, cbd=False):
    """The rule's kerb to kerb for a way with these tags, from the road as `core.section` maps it."""
    return kerbs.kerb_to_kerb(tags, section.width_from_tags(tags)[0], cbd)


@pytest.mark.parametrize("tags, cbd, expected", [
    # each class's measured median (25 Nov 2026, the councils' kerb lines)
    ({"highway": "residential"}, False, 8.0),
    ({"highway": "unclassified"}, False, 8.0),
    ({"highway": "living_street"}, False, 8.0),
    ({"highway": "residential", "oneway": "yes"}, False, 8.0),
    ({"highway": "tertiary"}, False, 10.0),
    ({"highway": "secondary"}, False, 12.8),
    ({"highway": "primary"}, False, 12.8),
    ({"highway": "trunk", "maxspeed": "60"}, False, 12.8),
    # marked lanes on a two-way road: at least a collector's 11.0 m; not on a one-way road
    ({"highway": "residential", "lanes": "2"}, False, 11.0),
    ({"highway": "tertiary", "lanes": "2"}, False, 11.0),
    ({"highway": "secondary", "lanes": "2"}, False, 12.8),
    ({"highway": "tertiary", "lanes": "2", "oneway": "yes"}, False, 10.0),
    # a multi-lane road: its lanes x the lane width plus two channels, and no parking lanes on top (was 18.6 m)
    ({"highway": "trunk", "lanes": "4", "oneway": "yes"}, False, 4 * 3.5 + 0.9),
    ({"highway": "trunk", "lanes": "3", "oneway": "yes"}, False, 12.8),
    ({"highway": "primary", "lanes": "4"}, False, 4 * 3.4 + 0.9),
    # no on-street parking, by side; never narrower than the road plus its channels
    ({"highway": "secondary", "parking:both": "no"}, False, 12.8 - 2 * 2.3),
    ({"highway": "secondary", "parking:left": "separate"}, False, 12.8 - 2.3),
    ({"highway": "secondary", "parking:lane:right": "no"}, False, 12.8 - 2.3),
    ({"highway": "secondary", "parking:both": "lane"}, False, 12.8),
    ({"highway": "residential", "parking:both": "no"}, False, 2 * 2.9 + 0.9),
    # CBD-dense: at least 12.5 m
    ({"highway": "residential"}, True, 12.5),
    ({"highway": "primary"}, True, 12.8),
    # a `width` tag wins: the road is drawn that wide (`core.section`), and its channels are outside it
    ({"highway": "residential", "width": "11"}, False, 11.0 + 0.9),
    ({"highway": "trunk", "lanes": "4", "width": "15"}, False, 15.0 + 0.9),
])
def test_kerb_to_kerb_by_class_and_tags(tags, cbd, expected):
    assert k2k(tags, cbd) == pytest.approx(expected)


def test_kerb_to_kerb_is_never_narrower_than_the_road_was():
    assert kerbs.kerb_to_kerb({"highway": "residential", "width": "11"}, 5.8, False) == pytest.approx(11.0)
    assert kerbs.kerb_to_kerb({"highway": "residential"}, 9.0, False) == pytest.approx(9.9)  # 9 m plus two channels
    assert kerbs.kerb_to_kerb({"highway": "residential"}, 25.0, False) == pytest.approx(section.MAX_WIDTH)


def test_the_rule_meets_the_held_out_streets_measured_medians():
    """The calibration's hold-out: the 30% of streets left out of the fit (tests/data/README.md). Each class with 5 km
    or more of them comes within 0.7 m of its measured median; primary and trunk roads have one street each there."""
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).parent / "data" / "kerb_widths_holdout.json").read_text())
    by_class = {}
    for g in data["groups"]:
        hw = g["tags"]["highway"]
        c = hw if hw in data["classes"] else "local"
        by_class.setdefault(c, []).append((k2k(g["tags"], g["cbd"]), g["metres"]))
    tested = 0
    for c, v in data["classes"].items():
        if v["metres"] < 5000:
            continue
        kk, m = np.array(by_class[c]).T
        order = np.argsort(kk)
        median = kk[order][np.searchsorted(np.cumsum(m[order]), m.sum() / 2)]
        assert abs(median - v["measured_median"]) <= 0.7, (c, median, v["measured_median"])
        tested += 1
    assert tested == 3  # local, tertiary, secondary


def test_the_road_widens_to_kerb_to_kerb_easing_in_and_is_unchanged_outside():
    p = plan_of()
    lip = p.half["L"]
    assert lip[0] == 3.0 and lip[-1] == 3.0  # outside the stretch, as it was
    assert (lip[400:600] + kerbs.CHANNEL_M) * 2 == pytest.approx(kerbs.LOCAL_KERB_TO_KERB_M)  # kerb face to kerb face
    a = int(np.flatnonzero(p.kerb["L"] > 0)[0])
    assert lip[a + 5] < lip[a + 40]  # easing in over WIDTH_EASE_M
    assert np.abs(np.diff(lip)).max() < 0.2  # no step


# -- footpaths ---------------------------------------------------------------------------------------------------------

def test_footpath_sides_from_sidewalk_tags_turned_round_for_a_reversed_way():
    assert kerbs.footpath_sides({}) == (True, True)
    assert kerbs.footpath_sides({"sidewalk": "no"}) == (False, False)
    assert kerbs.footpath_sides({"sidewalk": "left"}) == (True, False)
    assert kerbs.footpath_sides({"sidewalk": "left"}, True) == (False, True)
    assert kerbs.footpath_sides({"sidewalk:right": "no"}) == (True, False)
    assert kerbs.footpath_sides({"sidewalk": "separate"}) == (True, True)
    assert kerbs.footpath_surface({"sidewalk:surface": "paving_stones"}, "L") is True
    assert kerbs.footpath_surface({"sidewalk:left:surface": "asphalt"}, "R", True) is False
    assert kerbs.footpath_surface({}, "L") is None


def parcels_beside(back_m, x0=0.0, x1=1000.0):
    """Lots either side of the road, their boundary `back_m` from the centreline."""
    return [shapely.box(x0, back_m, x1, back_m + 30), shapely.box(x0, -back_m - 30, x1, -back_m)]


def test_the_footpath_reaches_the_parcels_boundary_and_a_wide_suburban_verge_is_a_path_and_grass():
    face = kerbs.LOCAL_KERB_TO_KERB_M / 2
    p = plan_of(parcels=parcels_beside(face + 2.5))  # 2.5 m behind a local road's kerb face
    assert p.path["L"][500] == pytest.approx(2.5, abs=0.05)
    assert p.paved["L"][500] == pytest.approx(2.5, abs=0.05)
    p = plan_of(parcels=parcels_beside(face + 4.0))
    assert p.path["L"][500] == pytest.approx(4.0, abs=0.05)
    assert p.paved["L"][500] == pytest.approx(kerbs.PATH_DEFAULT_M)  # a 1.5 m path at the kerb, grass behind
    p = plan_of(parcels=parcels_beside(face + 9.0))  # too far to be the footpath's
    assert p.path["L"][500] == pytest.approx(kerbs.PATH_DEFAULT_M)


def test_with_no_parcels_footpaths_take_the_standard_width():
    """The graceful fallback: no cadastre (the mainland, or theLIST unreachable) is the rule's standard widths."""
    for parcels in (None, lambda boxes: None):
        p = plan_of(parcels=parcels)
        assert p.any()
        assert p.path["L"][500] == pytest.approx(kerbs.PATH_DEFAULT_M)
        assert p.summary["footpath_source_m"]["parcels"] == 0


def test_the_parcels_are_fetched_only_for_the_kerbed_stretches():
    asked = []

    def parcels(boxes):
        asked.extend(boxes)
        return parcels_beside(7.0)

    plan_of(parcels=parcels)
    assert asked and all(b[0] >= 250 - 50 and b[2] <= 760 + 50 for b in asked)
    asked.clear()
    plan_of(centroids=np.zeros((0, 2)), parcels=parcels)
    assert not asked  # nothing kerbed, nothing fetched


def test_cbd_dense_where_walls_stand_on_the_reserve_s_edge():
    back = 4.45 + 2.8
    walls = [shapely.box(x, back, x + 20, back + 15) for x in np.arange(300, 700, 22)]
    walls += [shapely.box(x, -back - 15, x + 20, -back) for x in np.arange(300, 700, 22)]
    p = plan_of(parcels=parcels_beside(back), buildings=walls, centroids=town(300, 700))
    assert p.cbd[400:600].all()
    assert (p.half["L"][450:550] + kerbs.CHANNEL_M) * 2 == pytest.approx(12.5)
    assert p.concrete["L"][500]
    suburb = [shapely.box(x, back + 6, x + 12, back + 18) for x in np.arange(300, 700, 22)]
    assert not plan_of(parcels=parcels_beside(back), buildings=suburb).cbd.any()


# -- the profile and its ramp ------------------------------------------------------------------------------------------

def rows(k=1.0, x=0.0, paved=1.5, out_m=3.0, n=3):
    lip = np.column_stack([np.arange(n, dtype=float), np.full(n, 5.0), np.full(n, GROUND)])
    out = lip + [0.0, out_m, 0.0]
    out[:, 2] = GROUND + kerbs.RAMP_RISE_M + kerbs.PATH_FALL * (out_m - kerbs.KC_WIDTH_M)
    return kerbs.columns(lip, out, np.full(n, k), np.full(n, x), np.full(n, paved))


def test_the_kc_profile_and_its_hidden_ramp():
    c = rows()
    # The ramp: from the channel's lip to the kerb's back, 150 mm over 600 mm.
    assert c.back[0, 1] - c.lip[0, 1] == pytest.approx(kerbs.RAMP_M)
    assert c.back[0, 2] - c.lip[0, 2] == pytest.approx(kerbs.RAMP_RISE_M)
    # The drawn kerb: a 450 mm channel dipping to the face's toe, a face battered 40 mm up to the 110 mm top.
    assert c.toe[0, 1] - c.lip[0, 1] == pytest.approx(0.45)
    assert c.toe[0, 2] == pytest.approx(GROUND - kerbs.CHANNEL_DIP_M)
    assert c.top[0, 1] - c.toe[0, 1] == pytest.approx(0.04)
    assert c.top[0, 2] - c.toe[0, 2] == pytest.approx(0.15, abs=0.01)
    assert c.back[0, 2] > c.top[0, 2]  # the top falls 1 in 30 to the road
    # The footpath: paved 1.5 m from the face, rising 2% away.
    assert c.paved[0, 1] - c.lip[0, 1] == pytest.approx(0.45 + 1.5)
    assert c.paved[0, 2] - c.back[0, 2] == pytest.approx(0.02 * (1.95 - 0.6), abs=1e-6)


def test_a_tapered_kerb_end_lies_in_the_verge_s_plane():
    c = rows(k=0.0)
    lip, out = c.lip[0], c.out[0]
    for p in (c.toe[0], c.top[0], c.back[0], c.paved[0]):
        t = (p[1] - lip[1]) / (out[1] - lip[1])
        assert p[2] == pytest.approx(lip[2] + t * (out[2] - lip[2]))


def test_a_vehicular_crossing_lowers_the_face_and_lengthens_the_ramp():
    c = rows(x=1.0)
    assert c.back[0, 1] - c.lip[0, 1] == pytest.approx(kerbs.CROSSOVER_KC_M)
    assert c.toe[0, 2] - c.lip[0, 2] == pytest.approx(kerbs.CROSSOVER_LIP_M)
    assert c.back[0, 2] - c.lip[0, 2] == pytest.approx(kerbs.RAMP_RISE_M)  # the footpath still starts at kerb height


def kerbed_road(n=401, half=3.0):
    s, xy, heading = straight(n)
    k = kerbs.ramp_in((s >= 100) & (s <= 300), s, kerbs.KERB_TAPER_M)
    z = np.zeros(n)
    kb = kerbs.Kerbs({"L": k, "R": k}, {"L": np.full(n, 2.5), "R": np.full(n, 2.5)},
                     {"L": np.full(n, 2.5), "R": np.full(n, 2.5)}, {"L": np.zeros(n, bool), "R": np.zeros(n, bool)})
    rb = mesh.build_ribbon(xy, np.full(n, GROUND), heading, z, np.full(n, half), np.full(n, half),
                           np.full(n, section.CROWN), np.full(n, section.CROWN),
                           lambda x, y: np.full(np.shape(x), GROUND - 0.3), **kb.ribbon())
    return s, rb, kb


def test_the_ramp_is_hidden_road_sharing_the_road_s_edge_and_the_footpath_s_vertices():
    s, rb, kb = kerbed_road()
    pieces = kerbs.Pieces.empty()
    for side in "LR":
        cols = kerbs.ribbon_columns(rb, side, kb.k[side], np.zeros(len(s)), kb.paved[side])
        kerbs.add_strip(pieces, cols, kb.quads(side), s, kerbs.windows(s), np.zeros(len(s) - 1, bool))
    out = kerbs.meshes(pieces, FRAME)
    names = {m.name for m in out}
    assert {"KERB_000", "1ROAD_KB000", "1FOOTWAY_000A"} <= names
    ramp = next(m for m in out if m.name == "1ROAD_KB000")
    assert not ramp.renderable and kerbs.is_ramp(ramp.name) and validate.is_road_mesh(ramp.name)
    assert not kerbs.is_ramp("1ROAD_BK0L") and kerbs.is_ramp("1ROAD_JKB003")
    road = mesh.ribbon_meshes(rb, s, FRAME, skirt=True, kerb={k: kb.quads(k) for k in "LR"})
    others = np.vstack([m.positions for m in road + out if m.name != "1ROAD_KB000" and validate.is_ground_mesh(m.name)])
    from scipy.spatial import cKDTree
    gap, _ = cKDTree(others).query(ramp.positions)
    assert gap.max() < 1e-4  # every ramp vertex is the road's edge or the footpath's start
    # No hole round the kerb: the verge quads are taken out where it runs, and it meets the terrain.
    terrain = mesh.terrain_meshes(rb, 60.0, lambda x, y: np.full(np.shape(x), GROUND - 0.3), FRAME)
    assert not validate.ground_holes(validate.mesh_nodes(road + out + terrain), near_m=20.0)[0].size


# -- crossings at driveways ------------------------------------------------------------------------------------------

class _Mouth:
    def __init__(self, side, s0, s1, kind="driveway"):
        self.side, self.s0, self.s1 = side, s0, s1
        self.junction = {"kind": kind, "way_id": 1}


def test_a_driveway_on_a_kerbed_stretch_gets_a_crossing_with_wings_and_close_ones_join():
    s, rb, kb = kerbed_road()
    x, recs = kerbs.crossovers(s, [_Mouth("L", 199, 201), _Mouth("L", 204, 206), _Mouth("R", 50, 52)], kb)
    assert len(recs) == 2  # the right-hand one isn't on a kerbed stretch
    assert x["L"][199] == 1.0 and x["L"][202] == 1.0  # 3.6 m crossings 1.4 m apart: joined, no kerb between
    assert x["L"][198] == pytest.approx(0.8, abs=0.01) and x["L"][197] == 0.0  # a 1 m wing from 198.2 m
    assert not x["R"].any()
    clear = {"L": np.zeros(len(s), bool), "R": np.zeros(len(s), bool)}
    clear["L"][198:202] = True  # a side road's mouth
    x, recs = kerbs.crossovers(s, [_Mouth("L", 199, 201)], kb, clear)
    assert not recs and not x["L"].any()


# -- returns -----------------------------------------------------------------------------------------------------------

def test_kerb_returns_are_5_m_or_8_m_where_either_road_is_secondary_or_above():
    assert kerbs.return_radius({"highway": "residential"}, {"highway": "unclassified"}) == 5.0
    assert kerbs.return_radius({"highway": "tertiary"}, {"highway": "residential"}) == 5.0
    assert kerbs.return_radius({"highway": "secondary"}, {"highway": "residential"}) == 8.0
    assert kerbs.return_radius({"highway": "residential"}, {"highway": "primary_link"}) == 8.0


def test_a_kerbed_junction_s_stub_flares_at_its_return_and_its_kerb_runs_round_it():
    from tests.test_stubs import FRAME as SFRAME, flat, junction, road

    c, sec = road()
    j = {**junction(300.0, 90.0), "flare_radius_m": kerbs.RETURN_M}
    plain = stubs.plan([junction(300.0, 90.0)], c, sec, flat())["stubs"][0]
    st = stubs.plan([j], c, sec, flat())["stubs"][0]
    assert st["flares"]["upstream"]["radius_m"] == pytest.approx(5.0, abs=0.3)
    assert plain["flares"]["upstream"]["radius_m"] == pytest.approx(8.0, abs=0.3)
    n = len(c["s"])
    k = np.ones(n)
    kb = kerbs.Kerbs({"L": k, "R": k}, {"L": np.full(n, 2.5), "R": np.full(n, 2.5)},
                     {"L": np.full(n, 2.5), "R": np.full(n, 2.5)}, {"L": np.zeros(n, bool), "R": np.zeros(n, bool)})
    st["kerb"] = True
    rb = mesh.build_ribbon(c["xy"], c["z"], c["heading"], c["curvature"], sec["half_left"], sec["half_right"],
                           sec["slope_left"], sec["slope_right"], flat(), **kb.ribbon())
    b = stubs.build([st], rb, c["s"], SFRAME, flat(), skirt=True, main_xy=c["xy"], kerbs=kb)
    assert b.stats["kerb_returns"] == 2
    names = {m.name for m in b.meshes}
    assert {"KERB_J000", "1ROAD_JKB000"} <= names
    # The stub's own kerb runs STUB_KERB_M up its edges, then tapers off.
    d = np.arange(30.0)
    w = kerbs.stub_weights(d, True)
    assert w[:10].min() == 1.0 and w[13:].max() == 0.0
    assert not kerbs.stub_weights(d, False).any()


# -- what stands beside a kerbed road ---------------------------------------------------------------------------------

def test_posts_stand_on_the_footpath_behind_the_kerb_face():
    n = 101
    s, xy, heading = straight(n)
    k = np.zeros(n)
    k[50:] = 1.0
    kb = kerbs.Kerbs({"L": k, "R": k}, {"L": np.full(n, 2.5), "R": np.full(n, 2.5)},
                     {"L": np.full(n, 2.5), "R": np.full(n, 2.5)}, {"L": np.zeros(n, bool), "R": np.zeros(n, bool)})
    road = signals._Road(s, xy, np.full(n, GROUND), heading, np.full(n, 4.0), np.full(n, 4.0), np.zeros(n),
                         np.zeros(n), kb)
    e, nn, h = road.place(80, "L", signals.POST_OFFSET_M)
    assert nn == pytest.approx(4.0 + kerbs.CHANNEL_M + kerbs.POST_BEHIND_M)
    assert h > GROUND + kerbs.RAMP_RISE_M - 1e-9
    e, nn, h = road.place(20, "L", signals.POST_OFFSET_M)
    assert nn == pytest.approx(4.0 + signals.POST_OFFSET_M) and h < GROUND


def test_the_ai_line_keeps_its_margin_from_the_kerb_face_plus_0_3_m():
    n = 11
    s, xy, heading = straight(n)
    k = np.ones(n)
    road = RoadModel.from_arrays(s, xy, np.zeros(n), heading, np.zeros(n), np.zeros(n), np.full(n, 4.0),
                                 np.full(n, 4.0), np.zeros(n), np.zeros(n), False, k, k)
    assert road.face_left[0] == pytest.approx(4.45)
    assert road.line_left[0] == pytest.approx(4.45 - 0.3)
    rev = RoadModel.from_arrays(s, xy, np.zeros(n), heading, np.zeros(n), np.zeros(n), np.full(n, 4.0),
                                np.full(n, 4.0), np.zeros(n), np.zeros(n), True, k, np.zeros(n))
    assert rev.face_left[0] == pytest.approx(4.0) and rev.face_right[0] == pytest.approx(4.45)


def test_trees_keep_to_the_kerb_or_behind_the_footpath_and_buildings_off_it():
    n = 401
    s, xy, heading = straight(n)
    k = np.zeros(n)
    k[100:300] = 1.0
    kb = kerbs.Kerbs({"L": k, "R": k}, {"L": np.full(n, 3.0), "R": np.full(n, 3.0)},
                     {"L": np.full(n, 3.0), "R": np.full(n, 3.0)}, {"L": np.zeros(n, bool), "R": np.zeros(n, bool)})
    trees, buildings = kerbs.clear_zones(xy, heading, np.full(n, 4.0), np.full(n, 4.0), kb, 4.0, 2.0)
    face = 4.45
    assert trees.contains(shapely.Point(200, face + 0.5))  # too near the kerb
    assert not trees.contains(shapely.Point(200, face + 0.9))  # a street tree's place
    assert trees.contains(shapely.Point(200, face + 2.0))  # in the footpath
    assert not trees.contains(shapely.Point(200, face + 3.5))  # behind it
    assert trees.contains(shapely.Point(20, 7.5))  # off the kerbed stretch, as before: 4 m past a 4 m half
    assert buildings.contains(shapely.Point(200, face + 2.8)) and not buildings.contains(shapely.Point(200, face + 3.0))


def test_a_side_with_no_kerb_beside_a_kerbed_one_is_kept_clear_from_its_own_wider_edge():
    """The CBD's failing trunk (24 Nov): kerbed on the right, a council's Edge of Formation on the left, whose drawn
    edge (8.85 m out) is far past the plain stretches' widest (4 m). A tree 7.7 m out stood on the road."""
    n = 401
    s, xy, heading = straight(n)
    k_l, k_r = np.zeros(n), np.zeros(n)
    k_r[100:300] = 1.0
    half_l = np.full(n, 4.0)
    half_l[100:300] = 8.85
    kb = kerbs.Kerbs({"L": k_l, "R": k_r}, {"L": np.full(n, 1.5), "R": np.full(n, 3.0)},
                     {"L": np.full(n, 1.5), "R": np.full(n, 3.0)}, {"L": np.zeros(n, bool), "R": np.zeros(n, bool)})
    trees, buildings = kerbs.clear_zones(xy, heading, half_l, np.full(n, 4.0), kb, 4.0, 2.0)
    assert trees.contains(shapely.Point(200, 7.7))  # on the road: cleared
    assert trees.contains(shapely.Point(200, 8.85 + 3.9))  # and the plain clearance past its own edge
    assert not trees.contains(shapely.Point(200, 8.85 + 4.2))
    assert buildings.contains(shapely.Point(200, 8.85 + 1.9)) and not buildings.contains(shapely.Point(200, 8.85 + 2.2))
    assert not trees.contains(shapely.Point(200, -(4.45 + 0.9)))  # the kerbed side keeps its street-tree place


# -- the FOOTWAY surface -----------------------------------------------------------------------------------------------

def test_footway_is_road_s_grip_and_not_valid_track_written_only_when_used():
    plain = package.surfaces_ini()
    assert "FOOTWAY" not in plain
    text = package.surfaces_ini(footway=True)
    blocks = {b.split("KEY=")[1].split("\r\n")[0]: b for b in text.split("[SURFACE_")[1:]}
    assert "IS_VALID_TRACK=0" in blocks["FOOTWAY"] and "FRICTION=0.98" in blocks["FOOTWAY"]
    assert "IS_VALID_TRACK=1" in blocks["ROAD"]
    assert validate.undefined_surfaces(["1FOOTWAY_000A"], plain) == ["1FOOTWAY_000A"]
    assert validate.undefined_surfaces(["1FOOTWAY_000A"], text) == []
    assert validate.is_ground_mesh("1FOOTWAY_000A") and not validate.is_road_mesh("1FOOTWAY_000A")


# -- the job option and the fallbacks ---------------------------------------------------------------------------------

def test_the_kerbs_option_defaults_to_auto_and_leaves_earlier_track_ids_alone():
    base = {"track_id": "x", "waypoints": [{"lon": 147.3, "lat": -42.8}, {"lon": 147.31, "lat": -42.8}]}
    a, b = Job.model_validate(base), Job.model_validate({**base, "kerbs": "auto"})
    assert a.kerbs.value == "auto" and db.route_hash(a) == db.route_hash(b)
    assert db.route_hash(Job.model_validate({**base, "kerbs": "none"})) != db.route_hash(a)
    assert "kerbs" not in db.job_dict(a)


class _Region:
    def __init__(self, buildings=None, parcels=None):
        self.buildings, self.parcels = buildings, parcels
        self.buildings_attribution = self.parcels_attribution = None


def _section_kerbs(tmp_path, region, kerbs_option="auto"):
    from mapcreator import config
    from mapcreator.pipeline import route_stage, section_stage
    from mapcreator.pipeline.workdir import WorkDir

    n = 1001
    s, xy, heading = straight(n)
    work = WorkDir(tmp_path)
    np.savez_compressed(work.stage(route_stage.STAGE) / "route.npz", xy=xy)
    job = Job.model_validate({"track_id": "t", "junctions": "none", "kerbs": kerbs_option,
                              "waypoints": [{"lon": 147.3, "lat": -42.8}, {"lon": 147.31, "lat": -42.8}]})
    c = {"s": s, "xy": xy, "heading": heading, "run": np.zeros(n, int)}
    runs = [{"tags": {"highway": "residential"}}]
    return section_stage._kerbs(job, config.load(), region, work, c, runs, np.full(n, 3.0), np.zeros(n, bool))


def test_no_parcels_source_builds_kerbs_at_the_standard_widths_with_a_warning(tmp_path, caplog):
    houses = [shapely.box(x, 15, x + 8, 25) for x in np.arange(300, 700, 12)]

    def broken(bbox, settings):
        raise ConnectionError("unreachable")

    p = _section_kerbs(tmp_path, _Region(buildings=lambda b, st: houses, parcels=broken))
    assert p.any() and p.summary["parcels"] == "unavailable"
    assert p.path["L"][500] == pytest.approx(kerbs.PATH_DEFAULT_M)
    assert "land parcels unavailable" in caplog.text


def test_no_footprints_means_no_kerbs_and_the_build_goes_on(tmp_path, caplog):
    def broken(bbox, settings):
        raise ConnectionError("unreachable")

    p = _section_kerbs(tmp_path, _Region(buildings=broken))
    assert not p.any() and p.summary["off"] == "buildings_unavailable"
    assert "no kerbs" in caplog.text
    assert not _section_kerbs(tmp_path, _Region(), "none").any()


# -- the safety net before validation ----------------------------------------------------------------------------------

def _road_square(frame, x0=0.0, x1=40.0, y0=-4.0, y1=4.0):
    from mapcreator.core.scene import Mesh

    pos = frame.to_ac(np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1]), np.zeros(4))
    return Mesh("1ROAD_000", "road", pos.astype(np.float32), np.tile([0, 1, 0], (4, 1)).astype(np.float32),
                np.zeros((4, 2), np.float32), np.array([[0, 1, 2], [0, 2, 3]]))


def test_a_tree_or_post_collider_on_the_road_is_dropped_and_the_rest_kept(tmp_path, caplog):
    from mapcreator.core import scenery
    from mapcreator.core.scene import LocalFrame, Scene
    from mapcreator.pipeline import track_stage

    frame = LocalFrame(0.0, 0.0, 0.0)
    road = _road_square(frame)
    xy = np.array([[10.0, 1.0], [20.0, 9.0], [30.0, -3.95]])  # on the road; well off it; its edge over the road's
    (trunks,) = scenery.trunk_colliders(xy, np.zeros(3), np.full(3, 0.2), np.zeros(3), np.zeros(3), frame)
    kept, dropped = validate.drop_colliders_on_road([road, trunks], [road])
    assert [d["what"] for d in dropped] == ["tree", "tree"]
    assert sorted(round(d["x"]) for d in dropped) == [10, 30]
    (left,) = [m for m in kept if m.name == trunks.name]
    assert len(left.positions) == len(trunks.positions) // 3 and left.triangles.max() < len(left.positions)
    nodes = validate.mesh_nodes(kept)["nodes"]
    surface = validate.RoadSurface(nodes)
    assert not surface.covers(left.positions[:, [0, 2]]).any()  # the validator's own check now passes
    # A post's box the same way; a mesh with nothing left goes.
    from mapcreator.core.scene import Mesh

    box = Mesh("1WALL_SIGNAL_0", "x", trunks.positions[:16], trunks.normals[:16], trunks.uvs[:16],
               trunks.triangles[np.all(trunks.triangles < 16, axis=1)])
    kept, dropped = validate.drop_colliders_on_road([road, box], [road])
    assert [d["what"] for d in dropped] == ["signal and blade post"] and [m.name for m in kept] == ["1ROAD_000"]
    # The track stage logs each, and its tree leaves the Tree FX list.
    scene = Scene(frame=frame, meshes=[road, trunks])
    with caplog.at_level("WARNING"):
        out = track_stage._drop_colliders_on_road([scene])
    assert len(out) == 2 and "tree collider dropped: it stood on the road" in caplog.text and "MGA" in caplog.text
    from mapcreator.pipeline import scenery_stage
    from mapcreator.pipeline.workdir import WorkDir

    work = WorkDir(tmp_path)
    fx = work.stage(scenery_stage.STAGE) / "treefx"
    fx.mkdir(parents=True)
    (fx / "t.bin").write_bytes(b"")
    pos = frame.to_ac(xy[:, 0], xy[:, 1], np.zeros(3))
    np.savez(fx / "placement.npz", pos=pos, model=np.zeros(3, np.int64), size=np.ones(3), angle=np.zeros(3),
             width=np.ones(3), names=np.array(["t"]))
    ext = tmp_path / "ext"
    ext.mkdir()
    assert track_stage._write_tree_fx(work, ext, without=[(d["x"], d["z"]) for d in out])[1] == 1


def test_nothing_is_dropped_when_every_collider_is_off_the_road():
    from mapcreator.core import scenery
    from mapcreator.core.scene import LocalFrame

    frame = LocalFrame(0.0, 0.0, 0.0)
    road = _road_square(frame)
    (trunks,) = scenery.trunk_colliders(np.array([[20.0, 9.0]]), np.zeros(1), np.full(1, 0.2), np.zeros(1),
                                        np.zeros(1), frame)
    kept, dropped = validate.drop_colliders_on_road([road, trunks], [road])
    assert not dropped and kept[1] is trunks
