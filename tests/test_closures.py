"""Closure props: the layout along the stub, the local frame, sizes and colours, colliders flush with what's seen,
double-sided signs, no arrows, the budgets, and placing a closure on a track."""

import math

import numpy as np
import pytest

from mapcreator.core import closures as C
from mapcreator.core import furniture, kn5, package, textures
from mapcreator.core.scene import MAX_VERTICES, LocalFrame
from mapcreator.pipeline import track_stage

W = 8.0  # a 6 m carriageway and 1 m of verge either side
KINDS = [("major", "boards"), ("minor", "boards"), ("minor", "bales"), ("limited", "boards")]


def points(meshes) -> np.ndarray:
    return np.vstack([m.positions for m in meshes]).astype(float)


def quads(mesh):
    """(corners, normal, uvs) per quad: the builder adds box faces four vertices at a time."""
    for q in range(0, len(mesh.positions), 4):
        yield mesh.positions[q:q + 4].astype(float), mesh.normals[q].astype(float), mesh.uvs[q:q + 4].astype(float)


def retro(element):
    return next(m for m in element.meshes if m.material == C.RETRO_MATERIAL)


def in_cell(uvs, name) -> bool:
    x0, y0, x1, y1 = C.CELLS[name]
    w, h = C.ATLAS_SIZE
    return bool(((uvs[:, 0] >= x0 / w) & (uvs[:, 0] <= x1 / w) & (uvs[:, 1] >= y0 / h) & (uvs[:, 1] <= y1 / h)).all())


def in_band(uvs, name) -> bool:
    top, bottom = C._band_v(name)
    return bool(np.isin(np.round(uvs[:, 1], 6), np.round([top, bottom], 6)).all())


# -- layout ------------------------------------------------------------------------------------------------------

def test_layout_is_tape_at_4_obstacle_at_12_and_the_second_tape_and_sign_at_37():
    assert C.layout("major") == {"tape": 4.0, "obstacle": 12.0, "warning": 37.0}
    assert C.layout("minor") == {"tape": 4.0, "obstacle": 12.0, "warning": 37.0}
    assert C.layout("limited") == {"tape": 4.0, "sign": 12.0}
    kit = C.closure("major", W)
    warning = kit.element("warning")
    sign_z = np.mean([b.z0 + b.z1 for b in warning.boxes]) / 2
    assert sign_z == pytest.approx(37.0 + C.SIGN_BEHIND_TAPE_M)


def test_a_short_stub_carries_the_sign_on_the_obstacle_then_gets_a_lone_sign():
    assert C.layout("minor", length=45.0) == {"tape": 4.0, "obstacle": 12.0, "warning": 37.0}  # Q8's 45 m
    assert C.layout("minor", length=44.9) == {"tape": 4.0, "obstacle": 12.0}
    assert C.layout("major", length=15.0) == {"tape": 4.0, "obstacle": 12.0}
    assert C.layout("major", length=14.0) == {"tape": 4.0, "sign": 11.0}
    assert C.layout("limited", length=10.0) == {"tape": 4.0, "sign": 7.0}
    assert C.layout("minor", length=8.0) == {"tape": 4.0}
    # The sign goes on the obstacle where the second tape doesn't fit: in the middle of the boards ...
    short = C.closure("minor", W, length=30.0)
    assert {e.name for e in short.elements} == {"tape", "obstacle"}
    assert any(in_cell(uv, "road_closed") for _, _, uv in quads(retro(short.element("obstacle"))))
    # ... and raised behind bales, showing over them.
    bales = C.closure("minor", W, length=30.0, obstacle="bales")
    sign = [b for b in bales.element("obstacle").boxes if b.x1 - b.x0 == pytest.approx(C.SIGN[0])]
    assert len(sign) == 1 and sign[0].y1 > C.BALE[1] + 0.2 and sign[0].z0 >= 12.0 + C.BALE[2] / 2
    # A long stub's minor obstacle carries no sign: it's with the second tape.
    assert not any(in_cell(uv, "road_closed") for _, _, uv in quads(retro(C.closure("minor", W).element("obstacle"))))


@pytest.mark.parametrize("length", [100.0, 60.0, 59.0, 40.0, 30.0, 28.0, 20.0])
@pytest.mark.parametrize("level,obstacle", KINDS)
def test_an_overshoot_stub_has_nothing_solid_in_its_first_25_m(level, obstacle, length):
    kit = C.closure(level, W, overshoot=True, length=length, obstacle=obstacle)
    assert kit.element("tape").z == C.TAPE_M
    for e in kit.elements:
        assert all(b.z0 >= C.OVERSHOOT_CLEAR_M for b in e.boxes)
        if e.name != "tape":
            assert points(e.meshes)[:, 2].min() >= C.OVERSHOOT_CLEAR_M
    for s in kit.slots:
        assert s.z - C.VEHICLE[0] / 2 >= C.OVERSHOOT_CLEAR_M
    if length >= 60.0 and level != "limited":
        assert [(e.name, e.z) for e in kit.elements] == [("tape", 4.0), ("obstacle", 27.0), ("warning", 52.0)]
    if length < 29.0:
        assert [e.name for e in kit.elements] == ["tape"]  # tape alone: no room past the clear zone


# -- the local frame and sizes -----------------------------------------------------------------------------------

@pytest.mark.parametrize("level,obstacle", KINDS)
def test_each_element_stands_across_the_stub_on_its_own_line(level, obstacle):
    kit = C.closure(level, W, obstacle=obstacle)
    for e in kit.elements:
        p = points(e.meshes)
        if e.name == "tape":
            assert p[:, 2] == pytest.approx(np.full(len(p), 4.0), abs=0.03)
        elif e.name == "obstacle":
            reach = C.BALE[2] / 2 if obstacle == "bales" and level == "minor" else C.TRESTLE_FOOT_M + 0.015
            assert p[:, 2].min() == pytest.approx(12.0 - reach, abs=0.02)
            assert p[:, 2].max() == pytest.approx(12.0 + reach, abs=0.02)
        # Across the stub, centred on its centreline (+x right, looking in).
        assert p[:, 0].min() == pytest.approx(-p[:, 0].max(), abs=1e-4)
        assert p[:, 1].min() < 0.0 < p[:, 1].max() < 1.4  # +y up: standing on the ground at y = 0


def test_the_tape_spans_the_width_between_stakes_and_sags_between_0_8_and_1_0_m():
    kit = C.closure("limited", W)
    tape = retro(kit.element("tape"))
    p = tape.positions.astype(float)
    half = W / 2 + C.STAKE_OUT_M
    assert p[:, 0].min() == pytest.approx(-half) and p[:, 0].max() == pytest.approx(half)
    n = C.TAPE_SEGMENTS + 1
    bottom, top = p[:n], p[n:2 * n]
    assert top[:, 1] - bottom[:, 1] == pytest.approx(np.full(n, C.TAPE_W_M))  # 75 mm
    assert 0.8 <= top[:, 1].min() < top[:, 1].max() <= 1.0
    assert top[n // 2, 1] == top[:, 1].min()  # lowest in the middle
    stakes = [m for m in kit.element("tape").meshes if m.material == C.MATERIAL][0].positions.astype(float)
    assert np.abs(stakes[:, 0]).max() == pytest.approx(half + C.STAKE_W / 2, abs=1e-4)
    assert kit.element("tape").colliders == []  # cars break through tape and its stakes


def test_the_tape_has_an_a4_tag_with_both_faces():
    mesh = [m for m in C.closure("major", W).element("tape").meshes if m.material == C.MATERIAL][0]
    tags = [(n[2], uv) for c, n, uv in quads(mesh) if in_cell(uv, "tag_front") or in_cell(uv, "tag_back")]
    assert sorted((z, in_cell(uv, "tag_front")) for z, uv in tags) == [(-1.0, True), (1.0, False)]
    corners = [c for c, n, uv in quads(mesh) if in_cell(uv, "tag_front")][0]
    assert np.ptp(corners[:, 0]) == pytest.approx(C.TAG[0]) and np.ptp(corners[:, 1]) == pytest.approx(C.TAG[1])


@pytest.mark.parametrize("width", [3.5, 8.0, 13.0])
def test_boards_fill_the_width_mirrored_left_and_right_with_stripes_on_both_faces(width):
    for level in ("major", "minor"):
        boards = []
        for c, n, uv in quads(retro(C.closure(level, width).element("obstacle"))):
            if in_band(uv, "stripes"):
                left, right = np.argmin(c[:, 0]), np.argmax(c[:, 0])
                boards.append((c[left, 0], c[right, 0], n[2], uv[left, 0], uv[right, 0]))
                assert np.ptp(c[:, 1]) == pytest.approx(C.BOARD_H)
                assert c[:, 1].max() == pytest.approx(C.BOARD_CENTRE_M + C.BOARD_H / 2)
        fronts = sorted(b for b in boards if b[2] < 0)
        backs = sorted(b for b in boards if b[2] > 0)
        assert [b[:2] for b in fronts] == [b[:2] for b in backs]  # every board striped on both faces
        assert all(x1 - x0 <= C.BOARD_MAX_M + 1e-9 for x0, x1, *_ in fronts)
        for x0, x1, _, u_left, u_right in boards:
            # The band's stripes rise to the right as u grows. Left of the middle u falls to the right, so the stripes
            # fall towards the middle; mirrored on the right. The same along x on both faces, so they fall towards
            # the middle from whichever side they're seen.
            assert (u_right > u_left) == ((x0 + x1) / 2 > 0)
        if level == "minor":
            assert fronts[0][0] == pytest.approx(-width / 2) and fronts[-1][1] == pytest.approx(width / 2)


def test_the_sign_is_t2_4_double_sided_and_reads_left_to_right_from_both_sides():
    faces = [(c, n, uv) for c, n, uv in quads(retro(C.closure("limited", W).element("sign")))]
    assert len(faces) == 2
    for c, n, uv in faces:
        assert in_cell(uv, "road_closed")
        assert np.ptp(c[:, 0]) == pytest.approx(C.SIGN[0]) and np.ptp(c[:, 1]) == pytest.approx(C.SIGN[1])
        # The cell's left edge (smallest u) is on the viewer's left: -x seen from the mainline (normal -z), +x seen
        # from behind (normal +z).
        left = c[np.argmin(uv[:, 0]), 0]
        assert (left < 0) == (n[2] < 0)
    assert sorted(n[2] for _, n, _ in faces) == [-1.0, 1.0]


def test_the_major_barrier_has_its_sign_in_the_middle_of_the_boards():
    faces = list(quads(retro(C.closure("major", W).element("obstacle"))))
    signs = [c for c, n, uv in faces if in_cell(uv, "road_closed")]
    assert len(signs) == 2  # both faces
    assert signs[0][:, 0].min() == pytest.approx(-C.SIGN[0] / 2)
    assert signs[0][:, 2].mean() == pytest.approx(12.0, abs=0.02)


def test_bales_lie_across_the_whole_width():
    kit = C.closure("minor", W, obstacle="bales")
    boxes = kit.element("obstacle").boxes
    assert len(boxes) == math.ceil(W / C.BALE[0])
    assert all(b.x1 - b.x0 <= C.BALE[0] + 1e-9 and b.z1 - b.z0 == pytest.approx(C.BALE[2]) for b in boxes)
    assert all(b.y1 == pytest.approx(C.BALE[1]) for b in boxes)


# -- colours and textures ----------------------------------------------------------------------------------------

def test_the_atlas_colours():
    img = np.asarray(C.atlas().image).astype(int)
    assert C.atlas().image.size == C.ATLAS_SIZE == (1024, 512)

    def share(rows, colour, tol=40):
        return float((np.abs(img[rows] - np.array(colour)).max(axis=-1) < tol).mean())

    row, rows, _ = C.BANDS["tape"]
    tape = slice(row, row + rows)
    assert share(tape, C.TAPE_RED) == pytest.approx(0.5, abs=0.08)
    assert share(tape, C.WHITE) == pytest.approx(0.5, abs=0.08)
    row, rows, _ = C.BANDS["stripes"]
    stripes = slice(row, row + rows)
    assert share(stripes, C.RETRO_YELLOW) == pytest.approx(0.5, abs=0.08)
    assert share(stripes, C.BLACK) == pytest.approx(0.5, abs=0.08)
    x0, y0, x1, y1 = C.CELLS["road_closed"]
    sign = img[y0:y1, x0:x1]
    white = (np.abs(sign - np.array(C.WHITE)).max(axis=-1) < 30).mean()
    black = (np.abs(sign - np.array(C.BLACK)).max(axis=-1) < 30).mean()
    assert white > 0.6 and 0.1 < black < 0.35  # black on white, the legend and its border
    legend = img[y0 + 30:y1 - 30, x0 + 60:x1 - 60]
    assert (np.abs(legend - np.array(C.BLACK)).max(axis=-1) < 30).mean() > 0.15  # the words are there
    x0, y0, x1, y1 = C.CELLS["trestle"]
    assert tuple(img[(y0 + y1) // 2, (x0 + x1) // 2]) == C.TRESTLE_YELLOW
    row, rows, _ = C.BANDS["straw_side"]
    straw = img[row:row + rows].reshape(-1, 3).mean(axis=0)
    assert straw[0] > straw[1] > straw[2] and straw[0] > 150  # golden


def test_repeating_bands_tile_seamlessly():
    img = np.asarray(C.atlas().image).astype(int)
    for name in ("tape", "stripes"):
        row, rows, tile = C.BANDS[name]
        band = img[row:row + rows]
        # The first column continues from the last as the pattern does from one column to the next.
        step = np.abs(band[:, 1] - band[:, 0]).mean()
        assert np.abs(band[:, 0] - band[:, -1]).mean() <= step + 25


def test_texture_size():
    dds = textures.to_dds(C.atlas().image, "DXT1")
    assert len(dds) < 360_000  # 1024 x 512 DXT1 with its mips


def test_no_arrows_chevrons_or_detour_signs():
    names = set(C.CELLS) | set(C.BANDS)
    assert not any(word in n for n in names for word in ("arrow", "chevron", "detour", "cam"))
    assert C.LEGEND == "ROAD CLOSED"
    for level, obstacle in KINDS:
        for overshoot in (False, True):
            kit = C.closure(level, W, obstacle=obstacle, overshoot=overshoot)
            assert {e.name for e in kit.elements} <= {"tape", "obstacle", "warning", "sign"}
            assert not any(w in m.name for m in kit.meshes() + kit.colliders() for w in ("ARROW", "CHEVRON", "DETOUR"))


# -- colliders ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("width", [3.0, 8.0, 13.0])
@pytest.mark.parametrize("level,obstacle", [("major", "boards"), ("minor", "boards"), ("minor", "bales")])
def test_the_obstacles_colliders_are_the_wall_spanning_exactly_the_width_flush_with_its_front(level, obstacle, width):
    el = C.closure(level, width, obstacle=obstacle).element("obstacle")
    front = min(b.z0 for b in el.boxes)
    row = sorted((b for b in el.boxes if b.z0 == front), key=lambda b: b.x0)
    assert row[0].x0 == pytest.approx(-width / 2) and row[-1].x1 == pytest.approx(width / 2)
    assert all(a.x1 == pytest.approx(b.x0) for a, b in zip(row, row[1:]))  # no gap a car could pass
    # Flush with what's seen: the front of the boards' trestle feet, or of the bales.
    assert front == pytest.approx(points(el.meshes)[:, 2].min(), abs=0.015)
    seen_top = points(el.meshes)[np.abs(points(el.meshes)[:, 0]) < width / 2 - 0.5][:, 1]
    assert max(b.y1 for b in row) <= seen_top.max() + 1e-6
    assert all(b.y0 <= -C.COLLIDER_BELOW_M + 1e-6 for b in row)


@pytest.mark.parametrize("level,obstacle", KINDS)
def test_no_collider_stands_in_open_air(level, obstacle):
    """Every collider box is filled by its prop: what's seen inside its footprint spans its width and depth and
    reaches its top."""
    for length in (100.0, 30.0):
        for e in C.closure(level, W, obstacle=obstacle, length=length).elements:
            p = points(e.meshes)
            for b in e.boxes:
                inside = p[(p[:, 0] >= b.x0 - 0.02) & (p[:, 0] <= b.x1 + 0.02) & (p[:, 2] >= b.z0 - 0.02)
                           & (p[:, 2] <= b.z1 + 0.02)]
                assert inside[:, 0].min() == pytest.approx(b.x0, abs=0.02)
                assert inside[:, 0].max() == pytest.approx(b.x1, abs=0.02)
                assert inside[:, 2].min() == pytest.approx(b.z0, abs=0.02)
                assert inside[:, 2].max() == pytest.approx(b.z1, abs=0.02)
                assert np.abs(inside[:, 1] - b.y1).min() < 1e-4  # its top is a top that's seen
                assert b.y1 <= inside[:, 1].max() + 1e-6


def test_colliders_are_invisible_solid_walls_wound_outwards():
    kit = C.closure("major", W)
    cols = kit.colliders()
    assert {m.name for m in cols} == {"1WALL_CLOSURE_OBSTACLE", "1WALL_CLOSURE_WARNING"}
    for m in cols:
        assert not m.renderable
        assert sum(k in m.name for k in package.SURFACE_KEYS + ("WALL",)) == 1  # the validator's physics name rule
    for e in kit.elements:
        for m in e.colliders:
            for k, (c, n, _) in enumerate(quads(m)):  # five faces a box: front, back, left, right, top
                b = e.boxes[k // 5]
                centre = np.array([(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, (b.z0 + b.z1) / 2])
                assert n @ (c.mean(axis=0) - centre) > 0


@pytest.mark.parametrize("level,obstacle", KINDS)
def test_local_triangles_wind_counter_clockwise_seen_from_the_front(level, obstacle):
    """The frame is left-handed, so in its numbers the edges' cross product points against the normal."""
    kit = C.closure(level, W, obstacle=obstacle)
    for m in kit.meshes() + kit.colliders():
        m.validate()
        p = m.positions.astype(float)
        t = m.triangles
        cross = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
        assert ((cross * m.normals[t].mean(axis=1)).sum(axis=1) < 0).all()
        assert np.linalg.norm(m.normals, axis=1) == pytest.approx(np.ones(len(p)), abs=1e-5)


def test_feet_stand_on_the_ground_given():
    def ground(x, z):  # a 2.5% crown and verges falling 6% beyond 3 m
        x = np.abs(x)
        return -0.025 * np.minimum(x, 3.0) - 0.06 * np.maximum(x - 3.0, 0.0)

    kit = C.closure("minor", W, ground=ground)
    tape = [m for m in kit.element("tape").meshes if m.material == C.MATERIAL][0].positions.astype(float)
    stake = tape[np.abs(tape[:, 0]) > W / 2]
    assert stake[:, 1].min() == pytest.approx(float(ground(W / 2 + C.STAKE_OUT_M, 4.0)) - C.STAKE_SINK_M, abs=1e-4)
    for b in kit.element("obstacle").boxes:
        lowest = ground(np.array([b.x0, b.x1]), np.array([b.z0, b.z1])).min()
        assert b.y0 <= lowest - C.COLLIDER_BELOW_M + 0.01
    p = points(kit.element("obstacle").meshes)
    foot = p[np.abs(p[:, 2] - (12.0 - C.TRESTLE_FOOT_M)) < 0.02]
    assert (foot[:, 1] <= ground(foot[:, 0], foot[:, 2]) + 0.001).all()


def test_width_may_change_along_the_stub():
    kit = C.closure("minor", lambda z: 12.0 if z < 6 else 8.0)
    tape = retro(kit.element("tape")).positions.astype(float)
    assert tape[:, 0].max() == pytest.approx(6.0 + C.STAKE_OUT_M)
    obstacle = kit.element("obstacle")
    assert max(b.x1 for b in obstacle.boxes) == pytest.approx(4.0)


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        C.closure("closed", W)
    with pytest.raises(ValueError):
        C.closure("minor", W, obstacle="cones")
    with pytest.raises(ValueError):
        C.closure("minor", 1.5)


# -- the vehicle slot --------------------------------------------------------------------------------------------

def test_a_major_closure_keeps_a_slot_for_the_vehicle_behind_its_boards():
    for width in (3.0, 8.0, 13.0):
        kit = C.closure("major", width)
        (slot,) = kit.slots
        back = max(b.z1 for b in kit.element("obstacle").boxes)
        a = math.radians(C.VEHICLE_ANGLE_DEG)
        reach = C.VEHICLE[0] / 2 * math.cos(a) + C.VEHICLE[1] / 2 * math.sin(a)
        assert slot.z - reach == pytest.approx(back + C.VEHICLE_GAP_M)
        assert slot.forward[2] < 0  # nose out, towards the mainline
        assert slot.z + reach < kit.element("warning").z
    assert C.closure("minor", W).slots == [] and C.closure("limited", W).slots == []


# -- placing -----------------------------------------------------------------------------------------------------

FRAME = LocalFrame(520_000.0, 5_250_000.0, 100.0)


def test_place_maps_the_local_frame_onto_the_track():
    kit = C.closure("major", W)
    e0, n0, h0 = 520_100.0, 5_250_200.0, 112.0
    placed = C.place(kit, (e0, n0, h0, math.pi / 2), FRAME, index=7)  # a stub heading north
    assert {m.name.split("_")[1] for m in placed.meshes} == {"007"}
    obstacle = [m for m in placed.colliders if "OBSTACLE" in m.name][0]
    e, n, h = FRAME.from_ac(obstacle.positions.astype(float))
    assert n.min() == pytest.approx(n0 + 12.0 - C.TRESTLE_FOOT_M, abs=0.01)  # 12 m in, +z north
    assert e.min() == pytest.approx(e0 - W / 2, abs=0.01) and e.max() == pytest.approx(e0 + W / 2, abs=0.01)
    sign = [m for m in placed.meshes if m.material == C.RETRO_MATERIAL and "OBSTACLE" in m.name][0]
    # +x is to the right looking in: east, for a stub heading north. Seen from the mainline (the face whose normal
    # points south: AC +z), the legend's left edge is the west end.
    faces = [(c, nrm, uv) for c, nrm, uv in quads(sign) if in_cell(uv, "road_closed")]
    (c, _, uv), = [f for f in faces if f[1][2] > 0.99]
    ce, _, _ = FRAME.from_ac(c)
    assert ce[np.argmin(uv[:, 0])] == pytest.approx(e0 - C.SIGN[0] / 2, abs=0.01)
    for m in placed.meshes + placed.colliders:
        m.validate()
        p, t = m.positions.astype(float), m.triangles
        cross = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
        assert ((cross * m.normals[t].mean(axis=1)).sum(axis=1) > 0).all()  # as the KN5 winds them
    (slot,) = placed.slots
    assert slot.e > e0 and slot.n > n0 + 12.0  # right of the middle, behind the boards
    assert slot.forward[2] > 0  # nose south (AC z is south), back towards the mainline


def test_place_stands_each_element_across_a_bending_stub_at_its_own_distance():
    def pose(z):  # turning left 1 degree per metre and climbing 5%
        hd = math.radians(z)
        return (520_000.0 + 50 * math.sin(hd), 5_250_000.0 + 50 * (1 - math.cos(hd)), 100.0 + 0.05 * z, hd)

    kit = C.closure("minor", W)
    placed = C.place(kit, pose, FRAME)
    warning = [m for m in placed.meshes if "WARNING" in m.name and m.material == C.RETRO_MATERIAL][0]
    tape = warning.positions.astype(float)[:C.TAPE_SEGMENTS + 1]  # its lower edge, left to right
    e, n, h = FRAME.from_ac(tape)
    pe, pn, ph, hd = pose(37.0)
    across = np.array([e[-1] - e[0], n[-1] - n[0]])
    assert across @ np.array([math.cos(hd), math.sin(hd)]) == pytest.approx(0.0, abs=1e-3)  # square to the stub
    mid = np.array([(e[0] + e[-1]) / 2, (n[0] + n[-1]) / 2])
    assert np.hypot(*(mid - [pe, pn])) == pytest.approx(C.STAKE_W / 2 + 0.002, abs=0.01)
    assert h.min() > ph + 0.7


def test_merge_packs_many_closures_under_the_vertex_limit():
    kits = [C.closure(level, W, obstacle=obstacle, seed=i) for i, (level, obstacle) in enumerate(KINDS * 60)]
    visual, colliders = [], []
    for i, kit in enumerate(kits):
        placed = C.place(kit, (520_000.0 + 50 * i, 5_250_000.0, 100.0, 0.3), FRAME, index=i)
        visual += placed.meshes
        colliders += placed.colliders
    merged = C.merge(visual) + C.merge(colliders)
    assert all(len(m.positions) <= MAX_VERTICES for m in merged)
    assert sum(len(m.positions) for m in merged) == sum(len(m.positions) for m in visual + colliders)
    assert len({m.name for m in merged}) == len(merged)
    assert all(m.name.startswith("1WALL_CLOSURE_") for m in merged if not m.renderable)
    assert len(merged) < 10
    for m in merged:
        m.validate()


def test_a_placed_closure_writes_to_a_kn5_and_reads_back(tmp_path):
    placed = C.place(C.closure("major", W), (520_010.0, 5_250_010.0, 101.0, 1.0), FRAME)
    meshes = C.merge(placed.meshes) + C.merge(placed.colliders)
    mats = list(C.materials().values())
    index = {m.name: i for i, m in enumerate(mats)}
    kmats = [kn5.Kn5Material(m.name, properties={"ksDiffuse": 0.6}, textures={"txDiffuse": f"{m.texture}.dds"})
             for m in mats]
    kmeshes = [kn5.Kn5Mesh(m.name, index[m.material], m.positions, m.normals, m.uvs, m.triangles,
                           tangents=kn5.tangents(m.positions.astype(float), m.normals.astype(float),
                                                 m.uvs.astype(float), m.triangles),
                           cast_shadows=m.renderable, renderable=m.renderable) for m in meshes]
    path = tmp_path / "closure.kn5"
    kn5.write(path, root="closure", textures={f"{C.TEXTURE}.dds": textures.to_dds(C.atlas().image)}, materials=kmats,
              meshes=kmeshes, dummies=[])
    data = kn5.read(path)
    nodes = [n for n in data["nodes"] if n["class"] == 2]
    assert sum(len(n["vertices"]) for n in nodes) == sum(len(m.positions) for m in meshes)
    assert sorted(n["renderable"] for n in nodes) == sorted(int(m.renderable) for m in meshes)


# -- night and budgets -------------------------------------------------------------------------------------------

def test_retroreflective_faces_get_the_furnitures_night_treatment():
    assert C.NIGHT_DIFFUSE == {C.RETRO_MATERIAL: furniture.NIGHT_DIFFUSE[furniture.RETRO_MATERIAL]}
    assert track_stage.NIGHT_DIFFUSE[C.RETRO_MATERIAL] == C.NIGHT_DIFFUSE[C.RETRO_MATERIAL]
    reflector = furniture.REFLECTOR_MATERIAL
    assert track_stage.NIGHT_DIFFUSE[reflector] == furniture.NIGHT_DIFFUSE[reflector]
    ini = package.ext_config_ini(grass_materials=["grass"], road_materials=["road"], version="t",
                                 retro_materials={C.RETRO_MATERIAL: (C.NIGHT_DIFFUSE[C.RETRO_MATERIAL], 0.6)})
    section = ini.split("[MATERIAL_ADJUSTMENT_0]")[1].split("[")[0]
    section = section.replace(" ", "")
    assert "MATERIALS=closure_retro" in section and "CONDITION=NIGHT_SMOOTH" in section
    # The boards' and sign's faces and the tape are retroreflective; frames, stakes, bales and the tag are not.
    kit = C.closure("major", W)
    for name in ("tape", "obstacle", "warning"):
        assert any(m.material == C.RETRO_MATERIAL for m in kit.element(name).meshes)
    for c, n, uv in quads(retro(kit.element("obstacle"))):
        assert in_band(uv, "stripes") or in_cell(uv, "road_closed")
    bales = C.closure("minor", W, obstacle="bales", length=40.0).element("obstacle")
    assert all(in_cell(uv, "road_closed") for _, _, uv in quads(retro(bales)))


@pytest.mark.parametrize("level,obstacle", KINDS)
def test_vertex_budgets(level, obstacle):
    for width in np.linspace(C.MIN_WIDTH_M, C.BUDGET_WIDTH_M, 13):
        for overshoot in (False, True):
            for length in (100.0, 40.0, 20.0):
                kit = C.closure(level, float(width), obstacle=obstacle, overshoot=overshoot, length=length)
                c = kit.counts()
                assert c["vertices"] <= C.BUDGET[level][0] and c["triangles"] <= C.BUDGET[level][1]
                assert c["collider_vertices"] <= C.COLLIDER_BUDGET[level][0]
                assert c["collider_triangles"] <= C.COLLIDER_BUDGET[level][1]
                for e in kit.elements:
                    key = f"obstacle_{kit.obstacle}" if e.name == "obstacle" else e.name
                    v, t = C.BUDGET[key]
                    assert c["elements"][e.name]["vertices"] <= v and c["elements"][e.name]["triangles"] <= t
