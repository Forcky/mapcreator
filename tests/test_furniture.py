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


def build(r, limits=None, rail_l=None, rail_r=None):
    n = len(r["s"])
    limits = np.full(n, 80.0) if limits is None else limits
    return furniture.furniture(r["s"], r["xy"], r["z"], r["heading"], r["curvature"], r["half_l"], r["half_r"],
                               r["slope_l"], r["slope_r"], r["superelevation"], limits, FRAME, rail_l, rail_r)


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
