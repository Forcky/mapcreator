"""Roadside furniture: guide post spacing, CAMs on the outside of tight bends, speed signs where the limit changes."""

import numpy as np
import pytest

from mapcreator.core import furniture
from mapcreator.core.scene import MAX_VERTICES, LocalFrame

FRAME = LocalFrame(0.0, 0.0, 0.0)
HALF = 3.2


def road(curvature: np.ndarray, step: float = 1.0):
    """A road from its curvature (1/m, + turning left), heading east from the origin, flat at 100 m."""
    n = len(curvature)
    s = np.arange(n) * step
    heading = np.concatenate([[0.0], np.cumsum(curvature[:-1] * step)])
    xy = np.column_stack([np.cumsum(np.cos(heading)) * step, np.cumsum(np.sin(heading)) * step]) - [step, 0.0]
    flat = np.zeros(n)
    return dict(s=s, xy=xy, z=np.full(n, 100.0), heading=heading, curvature=curvature, half_l=np.full(n, HALF),
                half_r=np.full(n, HALF), slope_l=flat, slope_r=flat, superelevation=flat)


def build(r, limits=None, rail_l=None, rail_r=None, unsealed=None):
    n = len(r["s"])
    limits = np.full(n, 80.0) if limits is None else limits
    return furniture.furniture(r["s"], r["xy"], r["z"], r["heading"], r["curvature"], r["half_l"], r["half_r"],
                               r["slope_l"], r["slope_r"], r["superelevation"], limits, FRAME, rail_l, rail_r,
                               unsealed=unsealed)


def posts_by_side(meshes, r):
    """Each guide post's station and side, from the meshes: posts are the 1.3 m boxes 4 m off the centreline."""
    pos = np.vstack([m.positions for m in meshes]).astype(float)
    e, n, h = FRAME.from_ac(pos)
    out = []
    for quad in range(0, len(pos), 4):
        if not (h[quad + 2] - h[quad] == pytest.approx(furniture.GUIDE_POST_HEIGHT_M + 0.3, abs=1e-3)):
            continue
        p = np.array([e[quad:quad + 4].mean(), n[quad:quad + 4].mean()])
        i = int(np.argmin(np.hypot(*(r["xy"] - p).T)))
        left = np.array([-np.sin(r["heading"][i]), np.cos(r["heading"][i])])
        out.append((round(float(r["s"][i])), "L" if (p - r["xy"][i]) @ left > 0 else "R"))
    return sorted(set(out))  # each post's four sides share a station and side


def test_guide_posts_stand_150_m_apart_on_a_straight_on_both_sides():
    r = road(np.zeros(1001))
    meshes, info, _ = build(r)
    assert info["guide_posts"] == {"L": 7, "R": 7}
    left = [s for s, side in posts_by_side(meshes, r) if side == "L"]
    assert np.diff(left).tolist() == [150] * 6


def test_on_a_curve_the_outside_gets_posts_twice_as_close_as_the_inside():
    r = road(np.full(400, 1 / 150.0))  # turning left at 150 m radius: the outside is on the right
    _, info, _ = build(r, limits=np.full(400, 50.0))  # slow enough that the bend gets no CAMs
    assert info["cams"] == 0
    assert info["guide_posts"]["R"] == pytest.approx(2 * info["guide_posts"]["L"], abs=2)  # 10 m against 20 m


def test_a_tight_bend_gets_cams_on_its_outside_instead_of_guide_posts():
    k = np.zeros(600)
    k[200:300] = 1 / 50.0  # a left-hander at 50 m radius: advisory about 54 km/h, against 80
    r = road(k)
    meshes, info, _ = build(r)
    assert info["cam_curves"] == 1 and info["cams"] >= furniture.CAM_MIN
    # The CAM boards are 0.6 m tall, 2 m beyond the right-hand edge (the outside), around the bend.
    pos = np.vstack([m.positions for m in meshes]).astype(float)
    e, n, h = FRAME.from_ac(pos)
    high = h >= 100.0 + furniture.CAM_BOTTOM_M - 1e-6  # boards and their posts' tops
    i = np.argmin(np.hypot(r["xy"][:, None, 0] - e[high], r["xy"][:, None, 1] - n[high]), axis=0)
    near = (r["s"][i] > 120) & (r["s"][i] < 380)  # not the speed signs near the layouts' starts
    i, pts = i[near], np.column_stack([e[high], n[high]])[near]
    left = np.column_stack([-np.sin(r["heading"][i]), np.cos(r["heading"][i])])
    assert len(i) and (((pts - r["xy"][i]) * left).sum(axis=1) < 0).all()  # all on the right
    assert r["s"][i].min() >= 200 - 40 and r["s"][i].max() <= 300 + 40
    # No guide posts on that side through the bend; they carry on on the inside.
    right = [s for s, sd in posts_by_side(meshes, r) if sd == "R"]
    assert not [s for s in right if 200 <= s <= 300]
    assert [s for s, sd in posts_by_side(meshes, r) if sd == "L" and 200 <= s <= 300]


def test_an_s_bend_gets_cams_on_each_bend_s_own_outside():
    k = np.zeros(800)
    k[200:300], k[300:400] = 1 / 50.0, -1 / 50.0
    _, info, _ = build(road(k))
    assert info["cam_curves"] == 2


def test_a_gentle_curve_gets_no_cams():
    _, info, _ = build(road(np.full(500, 1 / 500.0)))
    assert info["cams"] == 0


def test_no_guide_posts_along_a_guard_rail():
    r = road(np.zeros(1001))
    rail = np.zeros(1001, dtype=bool)
    rail[300:700] = True
    meshes, info, _ = build(r, rail_l=rail)
    left = [s for s, side in posts_by_side(meshes, r) if side == "L"]
    assert not [s for s in left if 300 <= s < 700] and info["guide_posts"]["R"] == 7


def bend_road():
    k = np.zeros(1201)
    k[600:700] = 1 / 50.0  # a left-hander at 50 m radius: advisory about 54 km/h, against 80
    return road(k)


def cam_stations(meshes, r, around=(300.0, 1000.0)):
    """Stations of the CAM boards (and their posts' tops) near the bend: not the speed signs near the ends."""
    pos = np.vstack([m.positions for m in meshes]).astype(float)
    e, n, h = FRAME.from_ac(pos)
    high = h >= 100.0 + furniture.CAM_BOTTOM_M - 1e-6
    i = np.argmin(np.hypot(r["xy"][:, None, 0] - e[high], r["xy"][:, None, 1] - n[high]), axis=0)
    s = r["s"][i]
    return s[(s > around[0]) & (s < around[1])]


def test_gravel_gets_no_guide_posts_and_no_cams_but_keeps_its_speed_signs():
    r = bend_road()
    limits = np.where(r["s"] < 900, 80.0, 60.0)
    unsealed = r["s"] >= 400  # sealed for 400 m, then gravel, with the tight bend on the gravel
    sealed_meshes, sealed, _ = build(r, limits=limits)
    meshes, gravel, _ = build(r, limits=limits, unsealed=unsealed)
    posts = posts_by_side(meshes, r)
    assert posts and all(s < 400 for s, _ in posts)  # posts on the seal only
    assert posts == [p for p in posts_by_side(sealed_meshes, r) if p[0] < 400]  # and just as they were there
    assert sealed["cam_curves"] == 1 and sealed["cams"] >= furniture.CAM_MIN  # sealed, the bend has CAMs
    assert gravel["cam_curves"] == gravel["cams"] == 0  # on gravel, none
    assert gravel["speed_signs"] == sealed["speed_signs"] == 4  # the signs as mapped


def test_a_bend_partly_on_gravel_gets_cams_only_if_its_sealed_part_qualifies_alone():
    r = bend_road()  # tight from about 590 m to 710 m (the curvature is smoothed over 20 m)
    sealed_meshes, sealed, _ = build(r)
    # The seal ends at 680 m: 90 m of the bend is sealed, enough on its own. Its CAMs stand on the seal only.
    meshes, info, _ = build(r, unsealed=r["s"] >= 680)
    at = cam_stations(meshes, r)
    assert info["cam_curves"] == 1 and 0 < info["cams"] < sealed["cams"]
    assert len(at) and at.max() < 680 and at.min() < 600  # one before the bend, as on a seal
    # The seal ends at 600 m: about 10 m of the bend is sealed, under CAM_MIN_CURVE_M. No CAMs at all.
    _, info, _ = build(r, unsealed=r["s"] >= 600)
    assert info["cam_curves"] == info["cams"] == 0


def test_speed_signs_where_the_limit_changes_for_both_directions_and_near_each_start():
    r = road(np.zeros(1001))
    limits = np.where(r["s"] < 500, 60.0, 80.0)
    _, info, tex = build(r, limits=limits)
    # At the change: 80 for forward traffic on its left, 60 for reverse traffic on its left (our right). And one
    # repeater near each layout's start.
    assert info["speed_signs"] == 4 and info["speed_limits"] == [60, 80]
    assert {"speed_60", "speed_80"} <= set(tex.cells)


def test_unknown_limits_make_no_signs():
    r = road(np.zeros(1001))
    limits = np.where(r["s"] < 500, np.nan, 80.0)
    _, info, _ = build(r, limits=limits)
    assert info["speed_signs"] == 1  # only the repeater near the reverse layout's start, inside the 80 zone


def test_speed_limits_come_from_each_station_s_osm_way():
    runs = [{"tags": {"maxspeed": "60"}}, {"tags": {}}, {"tags": {"maxspeed": "80 km/h"}}, {"tags": {"maxspeed": "50 mph"}},
            {"tags": {"maxspeed": "AU:rural"}}]
    got = furniture.speed_limits(np.array([0, 0, 1, 2, 3, 4]), runs)
    assert got[:2].tolist() == [60.0, 60.0] and got[3] == 80.0
    assert np.isnan(got[[2, 4, 5]]).all()  # no tag, a unit our signs don't use, or not a number


def test_meshes_are_valid_visual_only_and_face_outward():
    k = np.zeros(3000)
    k[1000:1100] = 1 / 40.0
    meshes, _, _ = build(road(k))
    for m in meshes:
        m.validate()
        assert m.name.startswith("FURNITURE_") and m.renderable and len(m.positions) <= MAX_VERTICES
        p0, p1, p2 = (m.positions[m.triangles[:, j]].astype(float) for j in range(3))
        agree = (np.cross(p1 - p0, p2 - p0) * m.normals[m.triangles].mean(axis=1)).sum(axis=1) >= 0
        assert agree.all()


def test_cams_need_an_advisory_speed_of_60_or_less():
    k = np.zeros(800)
    k[200:400] = 1 / 90.0  # about 66 km/h advisory: 15 under an 80 limit, but over 60
    _, info, _ = build(road(k))
    assert info["cams"] == 0
    k[200:400] = 1 / 50.0  # about 47 km/h
    _, info, _ = build(road(k))
    assert info["cam_curves"] == 1


def test_a_long_tight_bend_gets_at_most_cam_max_markers():
    k = np.zeros(1200)
    k[200:900] = 1 / 40.0  # a 700 m bend at 40 m radius
    _, info, _ = build(road(k))
    assert info["cam_curves"] == 1 and furniture.CAM_MIN <= info["cams"] <= furniture.CAM_MAX


def test_an_unknown_limit_is_judged_at_the_default():
    k = np.zeros(800)
    k[200:400] = 1 / 70.0  # about 59 km/h advisory: tight against 100 (85), not against 80 (65)... and under 60
    _, unknown, _ = build(road(k), limits=np.full(800, np.nan))
    _, at_80, _ = build(road(k), limits=np.full(800, 80.0))
    assert unknown["cam_curves"] == at_80["cam_curves"]


def _quads(meshes, material):
    """Each quad of the given material: its four corners (region e, n, h)."""
    pos = [m.positions for m in meshes if m.material == material]
    if not pos:
        return np.zeros((0, 4, 3))
    e, n, h = FRAME.from_ac(np.vstack(pos).astype(float))
    return np.column_stack([e, n, h]).reshape(-1, 4, 3)


def test_retroreflective_faces_have_their_own_material():
    k = np.zeros(1001)
    k[400:450] = 1 / 30.0  # a bend with CAMs (advisory about 37 km/h, against 60)
    r = road(k)
    limits = np.where(r["s"] < 700, 60.0, 80.0)
    meshes, info, tex = build(r, limits=limits)
    assert {m.material for m in meshes} == {furniture.MATERIAL, furniture.REFLECTOR_MATERIAL, furniture.RETRO_MATERIAL}
    posts = sum(info["guide_posts"].values())
    cams, signs = info["cams"], info["speed_signs"]
    assert posts and cams and signs
    # Retroreflective: a reflector on each face of a post; both faces of a CAM board and a speed sign's face.
    assert len(_quads(meshes, furniture.REFLECTOR_MATERIAL)) == 2 * posts
    assert len(_quads(meshes, furniture.RETRO_MATERIAL)) == 2 * cams + signs
    # The rest: each post's four sides; each pole's four and a board's two edges, and a speed sign's back.
    assert len(_quads(meshes, furniture.MATERIAL)) == 4 * posts + 6 * cams + 7 * signs
    assert {"post", "reflector_red", "reflector_white"} <= set(tex.cells) and "post_red" not in tex.cells


def test_a_guide_post_reflector_sits_on_its_face_in_the_black_band():
    r = road(np.zeros(400))
    meshes, info, _ = build(r, limits=np.full(400, np.nan))  # no speed signs
    quads = _quads(meshes, furniture.REFLECTOR_MATERIAL)
    assert len(quads) == 2 * sum(info["guide_posts"].values())
    assert not len(_quads(meshes, furniture.RETRO_MATERIAL))  # no CAMs or signs on this road
    tops = quads[:, :, 2].max(axis=1) - quads[:, :, 2].min(axis=1)
    assert np.allclose(tops, furniture.REFLECTOR_H)
    widths = np.hypot(*(quads[:, 1, :2] - quads[:, 0, :2]).T)
    assert np.allclose(widths, furniture.REFLECTOR_W)
    # Each stands just in front of a post's face: half the post's depth and the gap from its centre, along the road.
    post_quads = _quads(meshes, furniture.MATERIAL)
    centres = post_quads[:, :, :2].mean(axis=1)
    for q in quads:
        c = q[:, :2].mean(axis=0)
        assert np.min(np.hypot(*(centres - c).T)) < furniture.GUIDE_POST_D


def test_ext_config_brightens_the_retroreflective_materials_only_at_night():
    from mapcreator.core import package

    retro = {name: (night, 0.6) for name, night in furniture.NIGHT_DIFFUSE.items()}
    text = package.ext_config_ini(grass_materials=["grass"], road_materials=["road"], version="t",
                                  retro_materials=retro)
    lines = text.split("\r\n")
    assert "[INCLUDE: common/conditions.ini]" in lines  # where CSP defines NIGHT_SMOOTH
    found = {}
    for at, line in enumerate(lines):
        if line.startswith("[MATERIAL_ADJUSTMENT_"):
            adj = dict(x.split("=", 1) for x in lines[at + 1:lines.index("", at)])
            assert adj["CONDITION"] == "NIGHT_SMOOTH" and adj["KEY_0"] == "ksDiffuse" and "KEY_1" not in adj
            found[adj["MATERIALS"]] = (float(adj["VALUE_0"]), float(adj["VALUE_0_OFF"]))
    assert found == retro  # brighter at night, the KN5's own value by day
    # Brighter in the headlights (diffuse), not glowing (no emissive); the reflectors more than the boards' faces,
    # whose legends wash out when pushed.
    assert all(night > 0.6 for night in furniture.NIGHT_DIFFUSE.values()) and "Emissive" not in text
    assert furniture.NIGHT_DIFFUSE[furniture.REFLECTOR_MATERIAL] > furniture.NIGHT_DIFFUSE[furniture.RETRO_MATERIAL]
    plain = package.ext_config_ini(grass_materials=["grass"], road_materials=["road"], version="t")
    assert "MATERIAL_ADJUSTMENT" not in plain and "conditions.ini" not in plain
