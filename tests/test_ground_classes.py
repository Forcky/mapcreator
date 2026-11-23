"""The ground classes over the terrain's triangles (`core.classmap`): the warp, the 8 m majority and the small-region
merge, normals shared across a class edge, and the cut batters above the road with their texture up the face."""

import numpy as np
import pytest
import shapely

from mapcreator.core import centreline as cl
from mapcreator.core import classmap, grass, mesh, package, textures, validate
from mapcreator.core import section as sec
from mapcreator.core.species import Community
from mapcreator.pipeline.mesh_stage import frame_for

E0, N0 = 500_000.0, 5_250_000.0


def box(x0, y0, x1, y1):
    return shapely.box(E0 + x0, N0 + y0, E0 + x1, N0 + y1)


def grid(nx, ny, step=2.0):
    """A plan grid of points `step` apart and its triangles, at height 0."""
    x, y = np.meshgrid(np.arange(nx) * step, np.arange(ny) * step, indexing="ij")
    pts = np.column_stack([x.ravel(), y.ravel()])
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a = (i * ny + j).ravel()
    tri = np.vstack([np.column_stack([a, a + ny, a + 1]), np.column_stack([a + 1, a + ny, a + ny + 1])])
    return pts, np.zeros(len(pts)), tri


def regions(code, tri, pts, hts):
    """(class, area m^2) of each connected region of one class."""
    i, j, _ = classmap.adjacency(tri, np.column_stack([pts, hts]))
    lab = classmap._components(np.asarray(code), i, j)
    _, area = classmap.slopes(np.column_stack([pts, hts])[tri])
    size = np.bincount(lab, weights=area)
    first = np.unique(lab, return_index=True)[1]
    return [(int(np.asarray(code)[f]), float(size[lab[f]])) for f in first]


# -- the warp -----------------------------------------------------------------------------------------------------

def test_the_warp_moves_neighbours_together_within_its_reach():
    rng = np.random.default_rng(1)
    xy = np.column_stack([E0 + rng.uniform(0, 2000, 5000), N0 + rng.uniform(0, 2000, 5000)])
    d = np.full(len(xy), 20.0)
    move = grass.warped(xy, d) - xy
    assert np.abs(move).max() <= grass.EDGE_DITHER_M + 1e-9  # each way
    assert np.hypot(*move.T).mean() > grass.EDGE_DITHER_M / 4  # it does move them
    # A point half a metre away moves nearly the same way (the dither it replaces moved each its own way, up to
    # twice the reach apart), and the same every time.
    near = grass.warped(xy + [0.5, 0.0], d) - (xy + [0.5, 0.0])
    assert np.median(np.hypot(*(near - move).T)) < 0.5
    assert (grass.warped(xy, d) == grass.warped(xy, d)).all()


# -- the majority and the small-region merge ------------------------------------------------------------------------

def test_the_majority_clears_speckle_and_keeps_the_edge():
    pts, hts, tri = grid(60, 60)  # 118 m square, 2 m cells
    cen = pts[tri].mean(axis=1)
    code = (cen[:, 0] > 59).astype(np.int64)  # west 0, east 1
    rng = np.random.default_rng(3)
    speckle = (np.abs(cen[:, 0] - 59) < 6) & (rng.random(len(cen)) < 0.3)  # a band of salt and pepper
    noisy = np.where(speckle, 1 - code, code)
    i, j, _ = classmap.adjacency(tri, np.column_stack([pts, hts]))
    _, area = classmap.slopes(np.column_stack([pts, hts])[tri])
    out = classmap.majority(noisy, cen, area, i, j)
    assert (out != code).mean() < 0.01 and (noisy != code).mean() > 0.02
    # Each side is one region again: no single-triangle islands.
    assert sorted(c for c, _ in regions(out, tri, pts, hts)) == [0, 1]
    # Away from a change of class nothing moves; and with one class there's nothing to do.
    assert (out[np.abs(cen[:, 0] - 59) > 30] == code[np.abs(cen[:, 0] - 59) > 30]).all()
    assert (classmap.majority(np.zeros(len(cen), np.int64), cen, area, i, j) == 0).all()


def test_regions_under_100_m2_merge_into_the_neighbour_they_share_most_edge_with():
    pts, hts, tri = grid(60, 60)
    cen = pts[tri].mean(axis=1)
    code = (cen[:, 0] > 59).astype(np.int64)
    small = (np.abs(cen[:, 0] - 30) < 4) & (np.abs(cen[:, 1] - 60) < 4)  # 8 x 8 m of class 2, in class 0
    big = (np.abs(cen[:, 0] - 90) < 8) & (np.abs(cen[:, 1] - 60) < 8)  # 16 x 16 m of class 2, in class 1
    # 8 x 8 m of class 2 across the edge: 18 m of its edge against class 0, 14 m against 1.
    across = (np.abs(cen[:, 0] - 57) < 4) & (np.abs(cen[:, 1] - 20) < 4)
    code[small | big | across] = 2
    i, j, length = classmap.adjacency(tri, np.column_stack([pts, hts]))
    _, area = classmap.slopes(np.column_stack([pts, hts])[tri])
    out = classmap.merge_small(code, area, i, j, length)
    assert (out[small] == 0).all() and (out[big] == 2).all()
    assert (out[across] == 0).all()
    assert min(a for _, a in regions(out, tri, pts, hts)) >= classmap.MIN_REGION_M2
    # A region with nothing of another class beside it keeps its class, however small.
    alone = np.array([0, 1, 2])
    assert classmap.merge_small(alone[:1], np.ones(1), np.zeros(0, np.int64), np.zeros(0, np.int64),
                                np.zeros(0)).tolist() == [0]


# -- a straight road with a cutting on one side and a fill on the other ---------------------------------------------

CUT_FROM, CUT_TO = 6.0, 18.0  # metres left of the centreline: the cut face, 50 degrees
FACE = np.tan(np.radians(50.0))


def cutting(e, n):
    """Flat at 100 m, but left of the road a 50-degree cut 12 m across (x 200-800 m, easing in over 50 m at each
    end), and right of it a 50-degree fill as deep; a steep natural bank 40-55 m left of the road above it (beyond
    the 25 m reach); and a 3 m knoll at x 1000, 12 m left."""
    e, n = np.asarray(e, float), np.asarray(n, float)
    x, y = e - E0, n - N0
    ends = np.clip((x - 150) / 50, 0, 1) * np.clip((850 - x) / 50, 0, 1)
    h = 100.0 + ends * FACE * (np.clip(y - CUT_FROM, 0, CUT_TO - CUT_FROM) - np.clip(-y - CUT_FROM, 0,
                                                                                         CUT_TO - CUT_FROM))
    h += np.where(y > 0, FACE * np.clip(y - 40, 0, 15), 0.0)  # the bank
    h += np.clip(3.0 - 1.5 * np.hypot(x - 1000, y - 24), 0, None)  # the knoll: 56 degrees, 4 m across
    return h


@pytest.fixture(scope="module")
def cut_road():
    t = np.linspace(0, 1200, 120)
    line = cl.smooth(np.column_stack([E0 + t, np.full(len(t), N0)]))
    z = cutting(line.xy[:, 0], line.xy[:, 1])
    half = np.full(len(line.s), 3.2)
    sl, sr = sec.half_slopes(np.zeros(len(line.s)))
    rb = mesh.build_ribbon(line.xy, z, line.heading, line.curvature, half, half, sl, sr, cutting)
    frame = frame_for(line.xy, z)
    gmap = grass.GroundMap(grass.tasveg_areas([Community(box(-500, -500, 1700, 500), "FAL")]))
    meshes = mesh.terrain_meshes(rb, 120.0, cutting, frame, ground=gmap.materials, classes=grass.MESHES,
                                 fold_skirt=True, cut=grass.CUT_MATERIAL)
    return frame, meshes


def _triangles(meshes, frame, material):
    p = np.vstack([m.positions[m.triangles] for m in meshes if m.material == material]).astype(float)
    cen = p.mean(axis=1)
    e, n, h = frame.from_ac(cen)
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    slope = np.degrees(np.arccos(np.abs(nrm[:, 1]) / np.linalg.norm(nrm, axis=1)))
    return p, e - E0, n - N0, h, slope, np.linalg.norm(nrm, axis=1) / 2


def test_the_cut_batter_is_above_the_road_within_25_m_and_steep(cut_road):
    frame, meshes = cut_road
    assert {m.material for m in meshes} == {"terrain", "terrain_cut"}
    cut = [m for m in meshes if m.material == "terrain_cut"]
    assert all(m.name[11] == "C" for m in cut)
    _, x, y, h, slope, area = _triangles(meshes, frame, "terrain_cut")
    # On the cutting's face, left of the road, all of it above the road, and within 25 m of its edge (3.2 m out).
    assert (y > 0).all() and (h > 100.0).all() and (y <= 3.2 + classmap.CUT_REACH_M).all()
    assert 150 <= x.min() and x.max() <= 850
    # It covers the face where it's full height: 600 m less the easing, 12 m across and up a 50-degree slope.
    full = (x > 210) & (x < 790)
    face = 580 * (CUT_TO - CUT_FROM) / np.cos(np.radians(50))
    assert area[full].sum() == pytest.approx(face, rel=0.15)
    assert np.average(slope, weights=area) > 40
    # The fill below the road keeps its class, the natural bank beyond 25 m too, and the knoll is too small.
    _, x, y, h, slope, area = _triangles(meshes, frame, "terrain")
    fill = (y < -CUT_FROM - 1) & (y > -CUT_TO + 1) & (x > 250) & (x < 750)
    assert fill.any() and np.median(slope[fill]) > 45
    bank = (y > 41) & (y < 54)
    assert bank.any() and (slope[bank] > 45).mean() > 0.5
    knoll = np.hypot(x - 1000, y - 24) < 2
    assert knoll.any()


def test_the_cut_batters_texture_runs_up_the_face_unstretched(cut_road):
    frame, meshes = cut_road
    tile = grass.MESHES[grass.CUT_MATERIAL][1]

    def stretch(m):
        """Per triangle: (most, least) metres of surface per metre of texture."""
        p = m.positions[m.triangles].astype(float)
        uv = m.uvs[m.triangles].astype(float) * tile
        e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        a = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        b = np.cross(np.cross(e1, e2), e1)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        surface = np.stack([np.stack([(e1 * a).sum(1), (e1 * b).sum(1)], 1),
                            np.stack([(e2 * a).sum(1), (e2 * b).sum(1)], 1)], 2)  # columns: e1, e2 in plane
        tex = np.stack([uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]], 2)
        jac = surface @ np.linalg.inv(tex)  # surface metres per texture metre
        s = np.linalg.svd(jac, compute_uv=False)
        return s[:, 0], s[:, 1]

    cut = [m for m in meshes if m.material == "terrain_cut"]
    most, least = np.concatenate([stretch(m)[0] for m in cut]), np.concatenate([stretch(m)[1] for m in cut])
    _, x, _, _, slope, _ = _triangles(cut, frame, "terrain_cut")
    steep = (slope > 45) & (x > 210) & (x < 790)
    assert steep.sum() > 100
    # A metre of texture is a metre of the face, both ways: the plan projection would stretch it 1.56 times.
    assert np.median(most[steep] / least[steep]) == pytest.approx(1.0, abs=0.05)
    assert np.median(most[steep]) == pytest.approx(1.0, abs=0.05)
    assert np.percentile(most[steep] / least[steep], 90) < 1.2
    # v runs up the face: it grows with height on the cutting.
    m = max(cut, key=lambda m: len(m.positions))
    up = np.corrcoef(m.positions[:, 1], m.uvs[:, 1])[0, 1]
    assert up > 0.9


def test_without_a_cut_material_steep_ground_keeps_its_class(cut_road):
    frame, _ = cut_road
    t = np.linspace(0, 1200, 120)
    line = cl.smooth(np.column_stack([E0 + t, np.full(len(t), N0)]))
    z = cutting(line.xy[:, 0], line.xy[:, 1])
    half = np.full(len(line.s), 3.2)
    sl, sr = sec.half_slopes(np.zeros(len(line.s)))
    rb = mesh.build_ribbon(line.xy, z, line.heading, line.curvature, half, half, sl, sr, cutting)
    gmap = grass.GroundMap(grass.tasveg_areas([Community(box(-500, -500, 1700, 500), "WOU")]))
    meshes = mesh.terrain_meshes(rb, 120.0, cutting, frame, ground=gmap.materials, classes=grass.MESHES,
                                 fold_skirt=True)
    assert {m.material for m in meshes} == {"terrain_forest"}


# -- shared normals and tidy class edges on real-looking ground -----------------------------------------------------

def hills(e, n):
    e, n = np.asarray(e, float), np.asarray(n, float)
    return 100.0 + 15.0 * np.sin(e / 90.0) * np.cos(n / 120.0) + 3.0 * np.sin(e / 13.0 + n / 17.0)


@pytest.fixture(scope="module")
def classed():
    t = np.linspace(0, 1200, 80)
    line = cl.smooth(np.column_stack([E0 + t, N0 + 30 * np.sin(t / 200)]))
    z = hills(line.xy[:, 0], line.xy[:, 1])
    half = np.full(len(line.s), 3.2)
    sl, sr = sec.half_slopes(np.zeros(len(line.s)))
    rb = mesh.build_ribbon(line.xy, z, line.heading, line.curvature, half, half, sl, sr, hills)
    frame = frame_for(line.xy, z)
    # Pasture west of x 600 m, forest east, and a strip of tussock 20 m wide across the road at x 300 m: TASVEG
    # draws narrow strips like it along creeks.
    gmap = grass.GroundMap(grass.tasveg_areas([Community(box(290, -800, 310, 800), "GPL"),
                                              Community(box(-800, -800, 600, 800), "FAL"),
                                              Community(box(600, -800, 2000, 800), "WOU")]))
    return frame, mesh.terrain_meshes(rb, 300.0, hills, frame, ground=gmap.materials, classes=grass.MESHES,
                                      fold_skirt=True, cut=grass.CUT_MATERIAL), shapely.LineString(line.xy)


def test_class_meshes_share_their_normals_along_a_class_edge(classed):
    frame, meshes, _ = classed
    pos = np.vstack([m.positions for m in meshes]).astype(float)
    nrm = np.vstack([m.normals for m in meshes]).astype(float)
    which = np.concatenate([[k] * len(m.positions) for k, m in enumerate(meshes)])
    mats = np.array([m.material for m in meshes])
    key, inv = np.unique(np.round(pos, 3), axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    same = np.flatnonzero(inv[order][1:] == inv[order][:-1])
    a, b = order[same], order[same + 1]
    other = mats[which[a]] != mats[which[b]]
    assert other.sum() > 100  # vertices on a class edge, one copy in each class's mesh
    gap = np.degrees(np.arccos(np.clip((nrm[a] * nrm[b]).sum(axis=1), -1, 1)))
    assert gap[other].max() < 0.05  # float32's rounding; a mesh's own normals were 3-12 degrees out (p50-p90)
    # Normals are still unit length and up.
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5) and (nrm[:, 1] > 0).all()


def test_no_class_region_under_100_m2_is_left(classed):
    frame, meshes, line = classed
    terrain = [m for m in meshes if m.name.startswith("1GRASS_T")]
    assert {m.material for m in terrain} >= {"terrain", "terrain_forest", "terrain_tussock"}
    p = np.vstack([m.positions[m.triangles] for m in terrain]).astype(float)
    code = np.concatenate([[k] * len(m.triangles) for k, m in enumerate(terrain)])
    mat = np.array([m.material for m in terrain])[code]
    key, inv = np.unique(np.round(p.reshape(-1, 3), 3), axis=0, return_inverse=True)
    tri = inv.reshape(-1, 3)
    names, label = np.unique(mat, return_inverse=True)
    sizes = regions(label, tri, key[:, [0, 2]], key[:, 1])
    assert min(a for _, a in sizes) >= classmap.MIN_REGION_M2
    # Near the road each class is one region a side, or one where the two sides meet round the road's end (the
    # tussock strip, across the road, is one each side). So no islands: further out the warp grows with the lattice
    # and may break the strip up, and pasture may then join round it.
    e, n, _ = frame.from_ac(key)
    near = shapely.distance(line, shapely.points(e, n)) < 100
    i, j, _ = classmap.adjacency(tri, key)
    lab = classmap._components(label, i, j)
    seen = np.unique(lab[near[tri].all(axis=1)])
    first = {r: label[np.flatnonzero(lab == r)[0]] for r in seen}
    count = {name: sum(1 for c in first.values() if names[c] == name) for name in names}
    assert count["terrain_forest"] <= 2 and count["terrain_tussock"] == 2 and count["terrain"] <= 3, count


# -- the cut's texture and its part of ext_config.ini --------------------------------------------------------------

def test_ground_class_textures_are_1024_px_and_the_mainlands_ground_is_unchanged():
    assert textures.GROUND_COVER_PX == 1024
    for name in [*textures.GROUND_COVERS, "ground_cut"]:
        assert textures.LIBRARY[name]().size == (1024, 1024), name
    assert textures.LIBRARY["ground"]().size == (512, 512)
    assert textures.LIBRARY["grass"]().size == (512, 512)
    assert grass.BY_KEY[grass.CUT].texture == "ground_cut" and grass.BY_KEY[grass.CUT].material == "terrain_cut"


def test_the_cut_texture_is_tileable_bare_earth_with_streaks_down_the_face():
    a = np.asarray(textures.ground_cut()).astype(float)
    inside = np.abs(np.diff(a, axis=0)).mean()
    assert np.abs(a[0] - a[-1]).mean() < 2 * inside + 1 and np.abs(a[:, 0] - a[:, -1]).mean() < 4 * inside + 1
    r, g, b = a.reshape(-1, 3).mean(axis=0)
    assert r > g > b and r > 1.2 * b  # brown, not green
    # Streaks: brightness changes more across the columns (along the road) than down them (up the face).
    lum = a.mean(axis=2)
    assert np.abs(np.diff(lum, axis=1)).mean() > 1.2 * np.abs(np.diff(lum, axis=0)).mean()


def test_the_cut_grows_no_grass_and_is_listed_with_the_bare_ground():
    cut = grass.CUT_MATERIAL
    assert not grass.grows(cut) and grass.BY_KEY[grass.CUT].config is None and not grass.BY_KEY[grass.CUT].warp
    assert cut in grass.PACK_MATERIALS and cut in grass.MESHES
    # One capital letter, and the mesh's surface is GRASS like the rest of the ground.
    assert sum(k in f"1GRASS_T012{grass.BY_KEY[grass.CUT].letter}" for k in package.SURFACE_KEYS + ("WALL",)) == 1
    # As the track stage writes ext_config.ini for a track with forest, pasture and a cut.
    used = ["grass", "terrain", "terrain_forest", cut]
    ground = [c.material for c in grass.CLASSES if c.material in used]
    sections = dict(validate.ini_sections(package.ext_config_ini(
        grass_materials=[n for n in ["grass", *ground] if grass.grows(n)], road_materials=["road"], version="t",
        bare_ground=[n for n in ground if not grass.grows(n)], grass_fx=grass.grass_fx(used),
        tree_list="trees.ini", tree_surfaces=[*ground, "grass"])))
    assert cut not in sections["GRASS_FX"]["GRASS_MATERIALS"].split(", ")
    assert cut in sections["RAIN_FX"]["ROUGH_MATERIALS"].split(", ")
    assert cut in sections["TREES"]["SURFACE_MATERIALS"].split(", ")
    assert cut not in sections["RAIN_FX"]["PUDDLES_MATERIALS"].split(", ")
    for name, values in sections.items():
        if name.startswith("GRASS_FX_ADJUSTMENT_"):
            assert cut not in values["MATERIALS"].split(", ")
