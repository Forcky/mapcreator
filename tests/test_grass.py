"""The grass pack: ground classes from TASVEG, their terrain materials and meshes, the Grass FX part of
ext_config.ini, and the procedural blade atlas."""

import io

import numpy as np
import pytest
import shapely
from PIL import Image

from mapcreator.core import centreline as cl
from mapcreator.core import grass, grassfx, mesh, package, textures, validate
from mapcreator.core import section as sec
from mapcreator.core.species import Community
from mapcreator.pipeline.mesh_stage import frame_for

E0, N0 = 500_000.0, 5_250_000.0


def box(x0, y0, x1, y1):
    return shapely.box(E0 + x0, N0 + y0, E0 + x1, N0 + y1)


# -- TASVEG codes to classes --------------------------------------------------------------------------------------

@pytest.mark.parametrize("codes,key", [
    ("FAL FAC FUM FUR", grass.PASTURE),  # towns too: they give way, as pasture does
    ("AUS ASS ASP HCM", grass.SHORT),
    ("GPL GPH GTL GCL GSL GHC GRP FMG MGH", grass.TUSSOCK),
    ("MBE MBP MBS MBU MBW MBR MDS MRR MSW HSE HSW HHE HHW HCH HUE", grass.MOOR),
    ("ASF ARS AWU FSM AHF AHL AHS", grass.WET),
    ("FPF", grass.BRACKEN),
    ("FRG FWU FPE", grass.ROUGH),
    ("DVG DMW DGW", grass.WOODLAND),
    # Dry and wet eucalypt forest, rainforest, non-eucalypt forest, scrub and plantations.
    ("DOB DGL DRI WNL WRE WOU RMT RML RKP NLA NAV NBS SWW SLS SCH FPS FPH FPU", grass.FOREST),
    ("OSM ORO AAP OAQ", grass.BARE),
])
def test_tasveg_codes_take_the_classes_of_the_design(codes, key):
    assert {grass.tasveg_class(c) for c in codes.split()} == {key}


def test_unlisted_codes_go_by_their_group_and_unknown_codes_by_nothing():
    # A code the table doesn't name takes its group's class (TASVEG's first letter is its group).
    assert grass.tasveg_class("DZZ") == grass.FOREST and grass.tasveg_class("GZZ") == grass.TUSSOCK
    assert grass.tasveg_class("HZZ") == grass.MOOR and grass.tasveg_class("AZZ") == grass.WET
    assert grass.tasveg_class("fal ") == grass.PASTURE  # as the provider reads it, case and spaces aside
    assert {grass.tasveg_class(c) for c in ("", "X", "ZZZ", "FALX")} == {grass.UNKNOWN}


def test_no_grass_in_forest_and_sparse_tussock_in_grassy_woodland():
    # Q5: wet forest, rainforest, plantations and scrub grow no Grass FX; grassy dry woodland grows B at half.
    for code in ("WRE", "RMT", "FPS", "SLS", "DOB"):
        assert grass.BY_KEY[grass.tasveg_class(code)].config is None
    woodland = grass.BY_KEY[grass.tasveg_class("DVG")]
    assert (woodland.config, woodland.weight) == ("B", 0.5)
    # Q3: base pasture, A short, B tussock, C moor and wetland, D bracken and regrowth.
    assert {k: (grass.BY_KEY[k].config, grass.BY_KEY[k].weight) for k in
            (grass.PASTURE, grass.SHORT, grass.TUSSOCK, grass.MOOR, grass.WET, grass.BRACKEN, grass.ROUGH)} == {
        grass.PASTURE: ("", 1.0), grass.SHORT: ("A", 1.0), grass.TUSSOCK: ("B", 1.0), grass.MOOR: ("C", 1.0),
        grass.WET: ("C", 1.0), grass.BRACKEN: ("D", 1.0), grass.ROUGH: ("D", 0.6)}


def test_every_class_has_its_own_material_texture_and_mesh_letter():
    # Pasture is today's ground: the `terrain` material and texture, repeating every 8 m as before.
    assert (grass.BY_KEY[grass.PASTURE].material, grass.BY_KEY[grass.PASTURE].texture) == ("terrain", "ground")
    assert grass.BY_KEY[grass.PASTURE].tile_m == 8.0 and grass.BY_KEY[grass.UNKNOWN] is grass.BY_KEY[grass.PASTURE]
    assert "terrain" not in grass.PACK_MATERIALS
    for field in ("key", "material", "texture", "letter"):
        values = [getattr(c, field) for c in grass.CLASSES]
        assert len(set(values)) == len(values)
    assert all(c.texture in textures.LIBRARY for c in grass.CLASSES)
    # A class letter never makes a terrain mesh match a second surface KEY.
    for c in grass.CLASSES:
        name = f"1GRASS_T012{c.letter}"
        assert sum(k in name for k in package.SURFACE_KEYS + ("WALL",)) == 1


# -- the ground map ------------------------------------------------------------------------------------------------

def test_the_ground_map_gives_each_point_its_area_class_and_far_forest_is_canopy():
    gmap = grass.GroundMap(grass.tasveg_areas([
        Community(box(0, 0, 100, 1000), "FAL"), Community(box(100, 0, 200, 1000), "WOU"),
        Community(box(50, 0, 150, 1000), "MBW"),  # overlaps both: the first area holding a point wins
    ]))
    xy = np.array([[60, 500], [120, 500], [300, 500]], float) + [E0, N0]
    assert gmap.classes(xy).tolist() == [grass.PASTURE, grass.FOREST, grass.UNKNOWN]
    # Pasture and a gap in the map are both today's ground.
    assert gmap.materials(xy, np.full(3, 50.0)).tolist() == ["terrain", "terrain_forest", "terrain"]
    assert grass.GroundMap(grass.tasveg_areas([Community(box(50, 0, 150, 1000), "MBW")])).classes(xy[:2]).tolist() == [
        grass.MOOR, grass.MOOR]
    # Forest past CANOPY_FROM_M is drawn as canopy from afar, dithered over the band around it; nothing else.
    wide = grass.GroundMap(grass.tasveg_areas([Community(box(-2000, -2000, 2000, 2000), "WOU")]))
    forest = np.column_stack([np.full(400, E0 + 175), N0 + np.linspace(1, 999, 400)])
    for d, share in ((grass.CANOPY_FROM_M - grass.CANOPY_BLEND_M / 2 - 1, 0.0), (grass.CANOPY_FROM_M, 0.5),
                     (grass.CANOPY_FROM_M + grass.CANOPY_BLEND_M / 2 + 1, 1.0)):
        canopy = (wide.materials(forest, np.full(400, d)) == "terrain_canopy").mean()
        assert canopy == pytest.approx(share, abs=0.12)
    assert (gmap.materials(xy[:1], np.full(1, 650.0)) == "terrain").all()


def test_a_class_edge_is_dithered_not_ruled():
    # Pasture west of x = 0, forest east of it. Near the road a triangle takes the class of a point up to
    # EDGE_DITHER_M from its centre: so the edge is ragged over about twice that, and plain beyond it.
    gmap = grass.GroundMap(grass.tasveg_areas([Community(box(-500, -500, 0, 500), "FAL"),
                                              Community(box(0, -500, 500, 500), "WOU")]))
    y = N0 + np.linspace(-400, 400, 2000)
    r = grass.EDGE_DITHER_M

    def forest_share(x, dist=20.0):  # forest, or far from the road its canopy
        xy = np.column_stack([np.full(len(y), E0 + x), y])
        return (gmap.materials(xy, np.full(len(y), dist)) != "terrain").mean()

    assert forest_share(-r - 0.5) == 0.0 and forest_share(r + 0.5) == 1.0
    assert 0.2 < forest_share(-r / 4) < 0.5 < forest_share(r / 4) < 0.8
    assert forest_share(0.0) == pytest.approx(0.5, abs=0.1)
    # Further out the dither widens with the lattice, up to five times as much.
    assert 0.0 < forest_share(-2 * r, dist=300.0) < 0.5
    # It's fixed by where a point is, so a build is the same every time.
    xy = np.column_stack([np.full(len(y), E0), y])
    assert (gmap.materials(xy, np.full(len(y), 20.0)) == gmap.materials(xy, np.full(len(y), 20.0))).all()


# -- terrain meshes by class ---------------------------------------------------------------------------------------

def hills(e, n):
    e, n = np.asarray(e, float), np.asarray(n, float)
    return 100.0 + 15.0 * np.sin(e / 90.0) * np.cos(n / 120.0)


@pytest.fixture(scope="module")
def road():
    t = np.linspace(0, 1200, 80)
    line = cl.smooth(np.column_stack([E0 + t, N0 + 30 * np.sin(t / 200)]))
    z = hills(line.xy[:, 0], line.xy[:, 1])
    half = np.full(len(line.s), 3.2)
    sl, sr = sec.half_slopes(np.zeros(len(line.s)))
    rb = mesh.build_ribbon(line.xy, z, line.heading, line.curvature, half, half, sl, sr, hills)
    return line, rb, frame_for(line.xy, z)


def _plan(meshes, frame):
    p = np.vstack([m.positions[m.triangles] for m in meshes]).astype(float)
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 2] - a[:, 2]) - (c[:, 0] - a[:, 0]) * (b[:, 2] - a[:, 2])
    return float(np.abs(cross).sum() / 2)


def test_classed_terrain_takes_the_skirt_and_leaves_the_verge_a_2m_strip(road):
    line, rb, frame = road
    gmap = grass.GroundMap(grass.tasveg_areas([Community(box(-800, -800, 600, 800), "FAL"),
                                              Community(box(600, -800, 2000, 800), "WOU")]))
    before = mesh.terrain_meshes(rb, 700.0, hills, frame) + mesh.ribbon_meshes(rb, line.s, frame)
    after = (mesh.terrain_meshes(rb, 700.0, hills, frame, ground=gmap.materials, classes=grass.MESHES,
                                 fold_skirt=True) + mesh.ribbon_meshes(rb, line.s, frame, skirt=False))
    # The same ground, cut differently: nothing lost or doubled.
    assert _plan(after, frame) == pytest.approx(_plan(before, frame), rel=1e-9)
    terrain = [m for m in after if m.name.startswith("1GRASS_T")]
    assert {m.material for m in terrain} == {"terrain", "terrain_forest", "terrain_canopy"}
    assert all(m.name[11] == grass.MESHES[m.material][0] for m in terrain)
    assert len({m.name for m in terrain}) == len(terrain)
    # No more meshes than without classes, give or take one per class: each gathers chunks rather than splitting
    # every chunk by class.
    assert len(terrain) <= len([m for m in before if m.name.startswith("1GRASS_T")]) + 3
    # Each class's texture coordinates are a function of position alone (at its class's repeat, warped where grass
    # grows), so two meshes of one class meet without a seam. Pasture's are today's.
    for m in terrain:
        x, z = m.positions[:, 0].astype(float), m.positions[:, 2].astype(float)
        _, tile, warp = grass.MESHES[m.material]
        uv = mesh.warped_uv(x, z, tile) if warp else np.column_stack([x, z]) / tile
        assert np.allclose(m.uvs, uv, atol=1e-4)
    assert not grass.MESHES["terrain"][2] and grass.MESHES["terrain"][1] == 8.0
    # The verge meshes now hold only the 2 m mown strip.
    verge = np.vstack([m.positions for m in after if m.name.startswith("1GRASS_V")]).astype(float)
    en = np.column_stack([verge[:, 0] + frame.e0, -verge[:, 2] + frame.n0])
    off = shapely.LineString(line.xy).distance(shapely.points(en))
    assert off.max() <= 3.2 + mesh.VERGE_M + 0.05
    assert all(m.material == "grass" for m in after if m.name.startswith("1GRASS_V"))


def test_without_a_ground_map_the_terrain_is_as_before(road):
    line, rb, frame = road
    plain = mesh.terrain_meshes(rb, 700.0, hills, frame)
    assert {m.material for m in plain} == {"terrain"} and all(m.name[-1].isdigit() for m in plain)
    # The verge meshes keep their skirt, out to 10 m beyond the 2 m verge.
    verge = np.vstack([m.positions for m in mesh.ribbon_meshes(rb, line.s, frame)
                       if m.name.startswith("1GRASS_V")]).astype(float)
    off = shapely.LineString(line.xy).distance(shapely.points(verge[:, 0] + frame.e0, -verge[:, 2] + frame.n0))
    assert off.max() == pytest.approx(3.2 + mesh.VERGE_M + mesh.SKIRT_STEPS_M[-1], abs=0.5)


# -- ext_config.ini ------------------------------------------------------------------------------------------------

# Every key the pack writes, each documented on CSP's Grass FX page (cup.acstuff.club/docs/csp/tracks/grass-fx).
DOCUMENTED = {
    "GRASS_FX": {"GRASS_MATERIALS", "OCCLUDING_MATERIALS", "MASK_BLUR", "TEXTURE", "TEXTURE_GRID", "SHAPE_SIZE",
                 "SHAPE_TIDY", "SHAPE_CUT", "TEXTURE_BASE_CHANCE"} | {f"TEXTURE_GROUP_{g}_CHANCE" for g in range(4)},
    "CONFIGURATION": {"TRIM_PERIOD", "SHAPE_SIZE", "SHAPE_TIDY", "SHAPE_CUT", "MASK_MAIN_THRESHOLD",
                      "MASK_RED_THRESHOLD", "MASK_MIN_LUMINANCE", "MASK_MAX_LUMINANCE", "TEXTURE_BASE_CHANCE"}
                     | {f"TEXTURE_GROUP_{g}_CHANCE" for g in range(4)},
    "GROUP": {f"PIECE_{n}{s}" for n in range(8) for s in ("", "_CHANCE", "_WIND", "_SIZE_MULT")},
    "ADJUSTMENT": {"MATERIALS", "MAP"},
}

EVERY_GROUND = ["grass", "gravel_narrow"] + [c.material for c in grass.CLASSES]


def _pack_ini(materials):
    ground = [c.material for c in grass.CLASSES if c.material in materials]
    narrow = ["gravel_narrow"] if "gravel_narrow" in materials else []
    return validate.ini_sections(package.ext_config_ini(
        grass_materials=[m for m in ["grass", *ground] if m in materials and grass.grows(m)],
        road_materials=["road"], version="t", gravel_materials=narrow, grassy_gravel=narrow,
        bare_ground=[m for m in ground if not grass.grows(m)], grass_fx=grass.grass_fx(materials)))


def test_the_pack_writes_four_configurations_groups_and_an_adjustment_per_class():
    sections = _pack_ini(EVERY_GROUND)
    names = [n for n, _ in sections]
    by = dict(sections)
    assert names.index("GRASS_FX") < names.index("GRASS_FX_TEXTURE_GROUP_0") < names.index("RAIN_FX")
    assert by["GRASS_FX"]["TEXTURE"] == "grass_fx/mapcreator.dds" and by["GRASS_FX"]["TEXTURE_GRID"] == "8, 4"
    assert [n for n in names if n.startswith("GRASS_FX_CONFIGURATION_")] == [
        f"GRASS_FX_CONFIGURATION_{x}" for x in "ABCD"]
    assert [n for n in names if n.startswith("GRASS_FX_TEXTURE_GROUP_")] == [
        f"GRASS_FX_TEXTURE_GROUP_{g}" for g in range(4)]
    # Each material that grows grass is in exactly one adjustment, with its class's configuration and weight; the
    # base classes (pasture, unmapped) and the bare ones are in none.
    maps = {}
    for n, v in sections:
        if n.startswith("GRASS_FX_ADJUSTMENT_"):
            for m in v["MATERIALS"].split(", "):
                assert m not in maps
                maps[m] = v["MAP"]
    assert maps == {"grass": "A", "gravel_narrow": "A", "terrain_short": "A", "terrain_tussock": "B",
                    "terrain_woodland": "B, 0.5", "terrain_moor": "C", "terrain_wet": "C",
                    "terrain_bracken": "D", "terrain_rough": "D, 0.6"}
    grows = by["GRASS_FX"]["GRASS_MATERIALS"].split(", ")
    assert set(grows) == set(maps) | {"terrain"}
    assert not {"terrain_forest", "terrain_canopy", "terrain_bare"} & set(grows)
    # Rain FX treats all the ground as rough, bare or not.
    assert set(EVERY_GROUND) <= set(by["RAIN_FX"]["ROUGH_MATERIALS"].split(", "))


def test_the_pack_uses_documented_keys_only():
    for name, values in _pack_ini(EVERY_GROUND):
        kind = ("GRASS_FX" if name == "GRASS_FX" else "CONFIGURATION" if "_CONFIGURATION_" in name
                else "GROUP" if "_TEXTURE_GROUP_" in name else "ADJUSTMENT" if "_ADJUSTMENT_" in name else None)
        if kind:
            assert set(values) <= DOCUMENTED[kind], (name, set(values) - DOCUMENTED[kind])


def test_adjustments_name_only_the_materials_a_track_has():
    sections = _pack_ini(["grass", "terrain", "terrain_forest", "terrain_moor"])
    named = [m for n, v in sections if n.startswith("GRASS_FX_ADJUSTMENT_") for m in v["MATERIALS"].split(", ")]
    assert sorted(named) == ["grass", "terrain_moor"]
    by = dict(sections)
    assert by["GRASS_FX"]["GRASS_MATERIALS"] == "grass, terrain, terrain_moor"


def _sizes(values):
    return [float(v) for v in values.get("SHAPE_SIZE", "1").split(",")]


def test_pasture_keeps_csps_default_grass_and_nothing_grows_denser_than_it():
    by = dict(_pack_ini(EVERY_GROUND))
    # Pasture and towns (the base configuration) as without the pack: CSP's default shape, and base blades only.
    assert not [k for k in by["GRASS_FX"] if k.startswith("SHAPE_") or k == "TRIM_PERIOD"]
    assert by["GRASS_FX"]["TEXTURE_BASE_CHANCE"] == "1"
    assert all(by["GRASS_FX"][f"TEXTURE_GROUP_{g}_CHANCE"] == "0" for g in range(4))
    # So is the mown verge (A): no pieces, and never under CSP's default size ("Higher size value lowers density").
    a = by["GRASS_FX_CONFIGURATION_A"]
    assert all(a[f"TEXTURE_GROUP_{g}_CHANCE"] == "0" for g in range(4))
    for name in "ABCD":
        values = by[f"GRASS_FX_CONFIGURATION_{name}"]
        assert min(_sizes(values)) >= 1.0, name
        # Every configuration names all four groups' chances: a chance left out isn't documented.
        assert all(f"TEXTURE_GROUP_{g}_CHANCE" in values for g in range(4)), name


def test_every_piece_sits_in_the_atlas_without_overlap():
    taken = np.zeros((grassfx.ROWS, grassfx.COLUMNS), int)
    for p in grassfx.PIECES:
        assert 1 <= p.col and p.col - 1 + p.w <= grassfx.COLUMNS and 0 <= p.row and p.row + p.h <= grassfx.ROWS
        taken[p.row:p.row + p.h, p.col - 1:p.col - 1 + p.w] += 1
    assert taken.max() == 1 and taken[0].tolist() == [1] * grassfx.COLUMNS  # row 0: eight base clumps
    # The groups use pieces of the atlas, never its base row, in CSP's "column (from 1), row (from 0)" form.
    keys = {p.key(): p for p in grassfx.PIECES}
    for _, values in _pack_ini(EVERY_GROUND):
        for k, v in values.items():
            if k.startswith("PIECE_") and k.count("_") == 1:
                assert v in keys and keys[v].row >= 1


def test_a_mainland_track_keeps_todays_grass():
    plain = validate.ini_sections(package.ext_config_ini(grass_materials=["grass", "terrain"], road_materials=["road"],
                                                         version="t"))
    assert dict(plain)["GRASS_FX"] == {"GRASS_MATERIALS": "grass, terrain", "OCCLUDING_MATERIALS": "road"}
    assert not [n for n, _ in plain if n.startswith("GRASS_FX_")]


# -- the atlas ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def atlas():
    return np.asarray(grassfx.atlas()).astype(int)


@pytest.fixture(scope="module")
def dds():
    return grassfx.atlas_dds()


def _cells(a, p):
    x, y, w, h = p.pixels()
    return a[y:y + h, x:x + w]


def test_the_atlas_is_8_by_4_cells_of_256_px(atlas):
    assert atlas.shape == (1024, 2048, 4) and grassfx.SIZE == (2048, 1024)


def test_base_blades_are_green_only_with_alpha_thicker_at_the_root(atlas):
    row = atlas[:grassfx.CELL_PX]
    assert (row[..., 0] == 0).all() and (row[..., 2] == 0).all()  # transparent pixels too: nothing to filter in
    for c in range(grassfx.COLUMNS):
        alpha = row[:, c * grassfx.CELL_PX:(c + 1) * grassfx.CELL_PX, 3] / 255.0
        assert 0.08 < alpha.mean() < 0.35  # CSP's own base clumps cover 13-22% of a cell
        assert alpha[-85:].mean() > 2 * alpha[:85].mean()  # rooted at the bottom edge, as CSP's are
        assert (alpha == 0).mean() > 0.5 and (alpha == 1).any()
    green = row[..., 1][row[..., 3] >= 128]
    assert 170 < np.median(green) < 235  # about as bright as CSP's (median 215), shaded darker at the root


def test_synced_pieces_are_green_only_and_coloured_ones_keep_red_and_blue(atlas):
    for p in grassfx.PIECES:
        cell = _cells(atlas, p)
        solid = cell[cell[..., 3] >= 128]
        coloured = (solid[:, 0] > 0) | (solid[:, 2] > 0)
        if p.colours == grassfx.SYNCED:
            assert (cell[..., 0] == 0).all() and (cell[..., 2] == 0).all(), p.name
        elif p.colours == grassfx.OWN:
            assert coloured.mean() > 0.99, p.name
        else:  # mixed: synced leaves with coloured heads or stalks
            assert 0.01 < coloured.mean() < 0.5, p.name
    for r, c in ((3, 2), (3, 3), (3, 6), (3, 7)):  # the spare cells stay empty
        assert (atlas[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256, 3] == 0).all()


def test_the_atlas_dds_is_dxt5_with_mips_and_stays_green_only_through_compression(dds):
    assert dds[:4] == b"DDS " and dds[84:88] == b"DXT5"
    h, w = np.frombuffer(dds[12:20], "<u4")
    assert (w, h) == grassfx.SIZE and int(np.frombuffer(dds[28:32], "<u4")[0]) == 9  # 2048 x 1024 down to 8 x 4
    assert 2.7e6 < len(dds) < 2.9e6
    decoded = np.asarray(Image.open(io.BytesIO(dds)).convert("RGBA")).astype(int)
    row = decoded[:grassfx.CELL_PX]
    assert (row[..., 0] == 0).all() and (row[..., 2] == 0).all()


def test_the_atlas_is_the_same_every_time():
    p = grassfx.BASE[0]
    a = np.asarray(grassfx.base_clump(p, np.random.default_rng(3)))
    b = np.asarray(grassfx.base_clump(p, np.random.default_rng(3)))
    assert (a == b).all()


def test_mip_levels_are_filled_with_their_pieces_colour():
    level = Image.new("RGBA", (512, 256), (0, 0, 0, 0))  # a mip level with nothing drawn: every cell is bled
    out = np.asarray(grassfx.fill(level)).astype(int)
    assert (out[..., 0] == 0).all() and (out[..., 1] > 0).all() and (out[..., 3] == 0).all()


# -- ground textures -----------------------------------------------------------------------------------------------

def test_each_ground_class_has_its_own_tileable_texture():
    means = {}
    for name in textures.GROUND_COVERS:
        a = np.asarray(textures.LIBRARY[name]()).astype(float)
        assert a.shape == (512, 512, 3)
        # Tileable: the jump across the wrap is no bigger than between neighbouring pixels inside.
        inside = np.abs(np.diff(a, axis=0)).mean()
        assert np.abs(a[0] - a[-1]).mean() < 2 * inside + 1
        means[name] = a.reshape(-1, 3).mean(axis=0)
    # Short herbland is green; tussock and moor run to straw and olive-tan; far forest is darker than all of them.
    assert means["ground_short"][1] > means["ground_short"][0]
    assert means["ground_tussock"][0] > means["ground_tussock"][1] * 0.95
    assert means["ground_canopy"].sum() < 0.6 * means["ground_short"].sum()
    # Pasture and towns draw on today's ground, so their own texture is gone.
    assert "ground_pasture" not in textures.LIBRARY
    # Moorland's peat is dark enough to sit under configuration C's MASK_MIN_LUMINANCE (in linear light).
    moor = np.asarray(textures.LIBRARY["ground_moor"]()).astype(float) / 255.0
    lin = np.where(moor <= 0.04045, moor / 12.92, ((moor + 0.055) / 1.055) ** 2.4) @ [0.2126, 0.7152, 0.0722]
    assert (lin < float(grass.CONFIGURATIONS["C"]["MASK_MIN_LUMINANCE"])).mean() > 0.08


def test_open_ground_doesnt_line_up_in_rows():
    # Over a 256 m plain, how much of each class's pattern (its luminance mask, as configuration C reads it, and its
    # brightness) repeats with the texture: laid at its class's repeat through `warped_uv`, against today's plain
    # 8 m tiling of the same texture. Plain tiling put rows of tussocks across Lyell Highway's moorland in game.
    x, z = np.meshgrid(np.arange(0, 256, 0.5), np.arange(0, 256, 0.5))

    def rows(name, uv, tile):
        tex = np.asarray(textures.LIBRARY[name]()).astype(float)
        img = tex[(np.floor(uv[:, 1] * 512).astype(int)) % 512, (np.floor(uv[:, 0] * 512).astype(int)) % 512]
        out = []
        for m in (img.mean(axis=1) < 60, img.mean(axis=1)):  # dark (bare) patches; brightness
            m = m.reshape(x.shape).astype(float)
            f = np.abs(np.fft.fft2(m - m.mean())) ** 2
            k = int(round(x.shape[0] * 0.5 / tile))
            out.append(sum(f[a * k % x.shape[0], b * k % x.shape[0]] for a in range(-3, 4) for b in range(-3, 4)
                           if (a, b) != (0, 0)) / max(f.sum(), 1e-9))
        return max(out)

    flat = np.column_stack([x.ravel(), z.ravel()])
    # Every class that grows grass, pasture aside, is warped; those that grow none aren't (they deflate better).
    assert {c.material for c in grass.CLASSES if c.warp} == {c.material for c in grass.CLASSES
                                                             if c.config is not None and c.material != "terrain"}
    for c in grass.CLASSES:
        if not c.warp:
            continue
        warped = rows(c.texture, mesh.warped_uv(flat[:, 0], flat[:, 1], c.tile_m), c.tile_m)
        plain = rows(c.texture, flat / 8.0, 8.0)  # what plain 8 m tiling did: 0.2-0.45 of the pattern repeated
        assert warped < 0.1 and warped < plain / 3, c.material
    # The warp moves a texture by no more than about UV_WARP_M, smoothly.
    shift = mesh.warped_uv(flat[:, 0], flat[:, 1], 1.0) - flat
    assert np.abs(shift).max() <= mesh.UV_WARP_M + 1e-6
