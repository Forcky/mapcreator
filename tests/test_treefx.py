"""3D trees for Custom Shaders Patch Tree FX: the model files, read back with an independent parser, and the list."""

import io
import struct
import zipfile

import numpy as np
import pytest

from mapcreator.core import package, sprites, treefx


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


def test_leaves_are_greener_than_red_and_bark_is_not():
    """Tree FX tells leaves from bark by green minus red."""
    leaves = np.asarray(sprites.leaf_sprays()).astype(int)
    solid = leaves[..., 3] > 200
    green = leaves[..., 1] - leaves[..., 0]
    assert (green[solid] >= sprites.MIN_LEAF_GREEN - 2).mean() > 0.9  # twigs aside
    bark = np.asarray(sprites.bark()).astype(int)
    assert (bark[..., 1] <= bark[..., 0]).all()


def test_models_suit_the_tree_height():
    names = list(treefx.KINDS)
    rng = np.random.default_rng(1)
    short = {names[i] for i in treefx.assign(np.full(500, 6.0), names, rng)}
    tall = {names[i] for i in treefx.assign(np.full(500, 30.0), names, rng)}
    assert short <= {"wattle", "euc_young"} and tall <= {"euc_tall", "euc_spread"}


def test_tree_list_lines():
    text = treefx.trees_txt(["euc_tall", "wattle"], np.array([1, 0]), np.array([[1.234, 5.0, -6.789], [10, 2, 3]]),
                            np.array([12.4, 300.0]), np.array([0.5, 1.25]), np.array([1.0, 0.9]),
                            "Test road: 2 trees.\nSecond line.")
    lines = text.split("\r\n")
    assert lines[0] == "; Test road: 2 trees." and lines[1] == "; Second line."
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
