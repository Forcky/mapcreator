"""3D trees for Custom Shaders Patch Tree FX: the model files, read back with an independent parser, and the list."""

import io
import struct
import zipfile

import numpy as np
import pytest

from mapcreator.core import package, reach, sprites, treefx


def read_mesh(data: bytes):
    """Parse a Tree FX .mesh: textures, sub-meshes (diffuse, normal, indices, vertices) and the trunk trailer."""
    off = 0

    def take(fmt):
        nonlocal off
        v = struct.unpack_from(fmt, data, off)
        off += struct.calcsize(fmt)
        return v

    version, zero, count = take("<iBi")
    textures = []
    for _ in range(count):
        h, n = take("<Ii")
        textures.append((h, data[off:off + n].decode("utf-8")))
        off += n
    subs, trunk = [], None
    while off < len(data):
        diffuse, normal = take("<ii")
        if diffuse == -1:
            assert normal == 4
            trunk = take("<f")[0]
            break
        (ni,) = take("<i")
        idx = np.frombuffer(data, "<u2", ni, off).reshape(-1, 3)
        off += 2 * ni
        (nv,) = take("<i")
        verts = np.frombuffer(data, treefx.VERTEX, nv, off)
        off += 44 * nv
        subs.append((diffuse, normal, idx, verts))
    assert off == len(data)
    return version, zero, textures, subs, trunk


@pytest.fixture(scope="module")
def lib():
    return treefx.library()


@pytest.fixture(scope="module")
def tex():
    return treefx.texture_files()


@pytest.mark.parametrize("data,expected", [(b"", 0x02CC5D05), (b"abc", 0x32D153FF),
                                           (b"Nobody inspects the spammish repetition", 0xE2293B2F)])
def test_xxh32_matches_the_reference_vectors(data, expected):
    assert treefx.xxh32(data) == expected


def test_lods_get_lighter_and_stay_under_the_index_limit(lib):
    for model in lib.values():
        tris = model.triangles
        assert tris == sorted(tris, reverse=True) and tris[-1] < tris[0] / 3
        for lod in model.lods:
            assert all(len(g.pos) <= 0xFFFF for g in lod.values())


def test_mesh_files_parse_and_are_normalised_to_the_model_box(lib, tex):
    model = lib["euc_tall"]
    height, half_w = model.box
    names = sorted(tex)
    hashes = [treefx.xxh32(tex[n]) or 1 for n in names]
    lods = [read_mesh(treefx.mesh_file(lod, names, hashes, height, half_w)) for lod in model.lods]
    # Positions are normalised by the box over every LOD (the far LODs' bigger leaf cards reach a bit further).
    every = np.vstack([v["pos"] for *_, subs, _ in lods for *_, v in subs])
    assert np.abs(every[:, [0, 2]]).max() == pytest.approx(1.0, abs=1e-6)
    assert every[:, 1].max() == pytest.approx(1.0, abs=1e-6)
    version, zero, textures, subs, trunk = lods[0]
    assert (version, zero) == (1, 0)
    assert textures == list(zip(hashes, names))
    by_diffuse = {names[d]: (names[n], idx, v) for d, n, idx, v in subs}
    assert set(by_diffuse) == {"bark.dds", "leaves.dds"}
    assert by_diffuse["bark.dds"][0] == "bark_nm.dds" and by_diffuse["leaves.dds"][0] == "leaves_nm.dds"
    allpos = np.vstack([v["pos"] for _, _, v in by_diffuse.values()])
    assert np.abs(allpos).max() <= 1.0 + 1e-6
    assert allpos[:, 1].min() < 0  # the trunk goes a little below the base, so it never floats on a slope
    # Trunk trailer: the base's half-width, normalised by the model's half-width.
    assert trunk == pytest.approx(treefx.KINDS["euc_tall"].radius * 1.25 / half_w, rel=0.1)
    for _, idx, v in by_diffuse.values():
        assert idx.max() < len(v)
        assert (v["extra"] == (255, 255, 255, 0)).all() and (v["tangent"][:, 3] == 0).all()
        assert np.allclose(np.linalg.norm(v["normal"], axis=1), 1, atol=1e-5)
        # Stored the converter's way round: the face's cross product points away from the normal.
        p = v["pos"].astype(float) * [half_w, height, half_w]
        face = np.cross(p[idx[:, 1]] - p[idx[:, 0]], p[idx[:, 2]] - p[idx[:, 0]])
        assert ((face * v["normal"][idx].mean(axis=1)).sum(axis=1) < 0).mean() > 0.99


def test_trunk_radius_is_the_trailer_in_metres(lib, tex):
    """The trunk colliders are sized from the same number the .mesh trailer gives Tree FX."""
    names = sorted(tex)
    for name, model in lib.items():
        height, half_w = model.box
        *_, trunk = read_mesh(treefx.mesh_file(model.lods[0], names, [1] * len(names), height, half_w))
        assert treefx.trunk_radius(model) == pytest.approx(trunk * half_w, rel=1e-6)
        # The base radius with its root flare, seen across the ring's flats.
        kind = treefx.KINDS[treefx.VARIANTS[name][0] if name in treefx.VARIANTS else name]
        assert kind.radius < treefx.trunk_radius(model) < kind.radius * 1.3


def test_a_bin_is_a_zip_with_tree_ini_lods_and_textures(lib, tex):
    model = lib["wattle"]
    with zipfile.ZipFile(io.BytesIO(treefx.model_bin(model, tex))) as z:
        names = set(z.namelist())
        assert names == {"tree.ini", "lod0.mesh", "lod1.mesh", "lod2.mesh", *tex}
        assert z.read("bark.dds") == tex["bark.dds"]
        ini = z.read("tree.ini").decode("utf-8")
    assert not ini.startswith("﻿") and "\r\n" in ini
    height, half_w = model.box
    assert f"HEIGHT={height:.3f}" in ini and f"WIDTH={2 * half_w:.3f}" in ini
    assert "[LOD_0]\r\nMODEL=lod0.mesh" in ini and "[LOD_2]\r\nMODEL=lod2.mesh" in ini
    assert f"DISTANCE={height * 10:.1f}" in ini  # the last LOD reaches ten heights out
    # The same model gives the same bytes, so builds are reproducible.
    assert treefx.model_bin(model, tex) == treefx.model_bin(model, tex)


def test_far_trees_are_left_to_tree_fx_billboards(lib):
    """Past its last LOD, Tree FX draws a tree as a baked billboard of six vertices, not as a mesh (CSP wiki, Tracks:
    Trees; its billboard shader). That's the cheapest far tree, so the meshes stop far inside the scenery reach:
    beyond about 270 m, every tree on a 600 m hillside is a billboard."""
    for model in lib.values():
        height, half_w = model.box
        ini = treefx.tree_ini(height, half_w, ["lod0.mesh", "lod1.mesh", "lod2.mesh"])
        distances = [float(line.split("=")[1]) for line in ini.split("\r\n") if line.startswith("DISTANCE=")]
        assert distances == sorted(distances) and len(distances) == 3
        assert distances[-1] == pytest.approx(height * 10, abs=0.05)  # Content Manager's converter's own rule
        assert distances[-1] < 0.5 * reach.SCENERY_M


def test_leaves_are_greener_than_red_and_bark_is_not():
    """Tree FX tells leaves from bark by green minus red."""
    leaves = np.asarray(sprites.leaf_sprays()).astype(int)
    solid = leaves[..., 3] > 200
    green = leaves[..., 1] - leaves[..., 0]
    assert (green[solid] >= sprites.MIN_LEAF_GREEN - 2).mean() > 0.9  # twigs aside
    for cell in range(sprites.LEAF_GRID ** 2):  # every species' cell, not just the atlas as a whole
        c, r = cell % sprites.LEAF_GRID, cell // sprites.LEAF_GRID
        sl = (slice(r * sprites.LEAF_CELL, (r + 1) * sprites.LEAF_CELL), slice(c * sprites.LEAF_CELL, (c + 1) * sprites.LEAF_CELL))
        assert solid[sl].mean() > 0.05, cell
        assert (green[sl][solid[sl]] >= sprites.MIN_LEAF_GREEN - 2).mean() > 0.85, cell
    bark = np.asarray(sprites.bark_atlas()[0]).astype(int)
    assert (bark[..., 1] <= bark[..., 0]).all()


def test_the_atlases_keep_todays_gum_bark_and_leaves():
    bark, _ = sprites.bark_atlas()
    w, h = sprites.BARK_SIZE
    assert bark.size == (w * sprites.BARK_COLUMNS, h)
    assert np.array_equal(np.asarray(bark)[:, :w], np.asarray(sprites.bark(sprites.BARK_SIZE)))
    # The first four leaf cells are drawn as before (three eucalypt sprays and the wattle), the new ones after.
    assert sprites.LEAF_KINDS[:4] == ("eucalypt", "eucalypt", "eucalypt", "wattle")


def test_every_model_carries_the_same_textures_and_uses_its_own_cells(lib, tex):
    for name, model in lib.items():
        kind = treefx.KINDS[treefx.VARIANTS[name][0] if name in treefx.VARIANTS else name]
        bark = treefx.VARIANTS[name][1] if name in treefx.VARIANTS else kind.bark
        u = np.vstack([lod["bark"].uv for lod in model.lods])[:, 0] * sprites.BARK_COLUMNS
        assert (u > bark).all() and (u < bark + 1).all(), name  # its own bark column, clear of the next
        cells = [sprites.leaf_uv_cell(c) for c in kind.leaf_cells]
        uv = np.vstack([lod["leaves"].uv for lod in model.lods])
        inside = np.zeros(len(uv), dtype=bool)
        for u0, v0, u1, v1 in cells:
            inside |= (uv[:, 0] >= u0 - 1e-6) & (uv[:, 0] <= u1 + 1e-6) & (uv[:, 1] >= v0 - 1e-6) & (uv[:, 1] <= v1 + 1e-6)
        assert inside.all(), name
    with zipfile.ZipFile(io.BytesIO(treefx.model_bin(lib["pine"], tex))) as a, \
            zipfile.ZipFile(io.BytesIO(treefx.model_bin(lib["euc_tall"], tex))) as b:
        for n in tex:
            assert a.read(n) == b.read(n)


def test_rough_bark_variants_keep_their_base_shape(lib):
    for name, (base, _) in treefx.VARIANTS.items():
        for mine, theirs in zip(lib[name].lods, lib[base].lods):
            assert np.array_equal(mine["bark"].pos, theirs["bark"].pos)
            assert np.array_equal(mine["leaves"].pos, theirs["leaves"].pos)
            assert not np.array_equal(mine["bark"].uv, theirs["bark"].uv)


def test_new_species_stay_in_the_triangle_budget_and_the_pine_is_a_cone(lib):
    for name in ("euc_wet", "tea_tree", "myrtle", "pine"):
        tris = lib[name].triangles
        assert tris[0] <= 2200 and tris == sorted(tris, reverse=True), (name, tris)
    # The pine's needles spread less towards the top: a cone, not a ball.
    leaves = lib["pine"].lods[0]["leaves"].pos
    height = lib["pine"].box[0]
    r = np.hypot(leaves[:, 0], leaves[:, 2])
    low = r[(leaves[:, 1] > 0.2 * height) & (leaves[:, 1] < 0.4 * height)].mean()
    assert low > 2 * r[leaves[:, 1] > 0.75 * height].mean()


def test_models_suit_the_tree_height():
    names = list(treefx.KINDS)
    rng = np.random.default_rng(1)
    short = {names[i] for i in treefx.assign(np.full(500, 6.0), names, rng)}
    tall = {names[i] for i in treefx.assign(np.full(500, 30.0), names, rng)}
    assert short <= {"wattle", "euc_young"} and tall <= {"euc_tall", "euc_spread"}


def test_tree_list_lines():
    text = treefx.trees_txt(["euc_tall", "wattle"], np.array([1, 0]), np.array([[1.234, 5.0, -6.789], [10, 2, 3]]),
                            np.array([12.4, 300.0]), np.array([0.5, 1.25]), np.array([1.0, 0.9]))
    assert text.endswith("\r\n")
    lines = text[:-2].split("\r\n")
    # Only configuration and tree lines: CSP took a comment line for a model name and loaded no trees.
    assert all(line.startswith(("configure: ", "tree: ")) for line in lines)
    assert "configure: size variance = 1, 1" in lines
    assert "tree: wattle.bin; pos=1.23, 5.00, -6.79; angle = 12; size = 0.500; width = 1.000" in lines
    assert "tree: euc_tall.bin; pos=10.00, 2.00, 3.00; angle = 300; size = 1.250; width = 0.900" in lines


def test_ext_config_lists_the_trees():
    text = package.ext_config_ini(grass_materials=["grass", "terrain"], road_materials=["road"], version="t",
                                  tree_list="trees/trees.txt", tree_surfaces=["terrain", "grass"])
    assert "[TREES]\r\nLIST_0=trees/trees.txt\r\nSURFACE_MATERIALS=terrain, grass" in text
    assert "MESH_ADJUSTMENT" not in text
    bare = package.ext_config_ini(grass_materials=["grass"], road_materials=["road"], version="t")
    assert "TREES" not in bare


def _fold(counts: dict[str, int], heights: dict[str, float]):
    names = list(treefx.MODEL_NAMES)
    model = np.concatenate([np.full(n, names.index(m)) for m, n in counts.items()])
    height = np.concatenate([np.full(n, heights.get(m, 20.0)) for m, n in counts.items()])
    out, folded = treefx.fold_rare(model, height, names)
    return {m: int((out == names.index(m)).sum()) for m in names if (out == names.index(m)).any()}, folded


def test_a_rarely_drawn_model_folds_into_a_similar_one_that_stays():
    # Lady Wakehurst Drive's three rare models: 5 smooth gums of each outline go to the rough-barked outlines (no
    # smooth model stays), and 7 wattles to tea-tree. 123 young eucalypts are over 0.2% of 31,000, so they stay.
    left, folded = _fold({"euc_wet": 21_000, "myrtle": 4_900, "euc_tall_rough": 3_600, "tea_tree": 1_400,
                          "euc_spread_rough": 450, "euc_young": 123, "wattle": 7, "euc_spread": 5, "euc_tall": 5},
                         {"tea_tree": 6.0, "wattle": 6.0, "euc_young": 10.0, "euc_wet": 45.0})
    assert folded == {"euc_tall": "euc_tall_rough", "euc_spread": "euc_spread_rough", "wattle": "tea_tree"}
    assert left["euc_tall_rough"] == 3_605 and left["euc_spread_rough"] == 455 and left["tea_tree"] == 1_407
    assert len(left) == 6 and sum(left.values()) == 31_490


def test_folding_keeps_pines_scales_and_small_tracks_in_mind():
    # A pine never folds; a rare model whose only similar ones would scale its trees past the clip stays.
    left, folded = _fold({"euc_tall": 5_000, "wattle": 4_000, "pine": 3, "myrtle": 4}, {"myrtle": 25.0})
    assert folded == {} and left["pine"] == 3 and left["myrtle"] == 4  # 25 m myrtles as 7.5 m wattles: 3.3x
    # On a small track the floor applies: fewer than FOLD_MIN trees fold, FOLD_MIN don't.
    assert _fold({"euc_tall": 300, "euc_spread": treefx.FOLD_MIN - 1}, {})[1] == {"euc_spread": "euc_tall"}
    assert _fold({"euc_tall": 300, "euc_spread": treefx.FOLD_MIN}, {})[1] == {}
    # The same trees fold the same way every time.
    assert _fold({"euc_tall": 300, "euc_spread": 2, "euc_young": 2}, {}) == \
        _fold({"euc_tall": 300, "euc_spread": 2, "euc_young": 2}, {})
