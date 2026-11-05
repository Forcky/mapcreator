"""Gravel and narrow roads: surface classes from OSM, widths and crown, the road's chunks and textures, and what
the track files say about them."""

import json

import numpy as np
import pytest

from mapcreator.core import ai_line, layout, mesh, package, textures
from mapcreator.core import section as sec
from mapcreator.core.layout import RoadModel, grid
from mapcreator.core.scene import LocalFrame
from mapcreator.pipeline import track_stage

S, U = sec.SEALED, sec.UNSEALED


# --- Surface classes -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("asphalt", S), ("chipseal", S), ("concrete:plates", S), ("paved", S), ("sett", S),
    ("unpaved", U), ("gravel", U), ("fine_gravel", U), ("compacted", U), ("dirt", U), ("earth", U), ("ground", U),
    ("mud", U), ("sand", U), ("pebblestone", U), ("rock", U), ("grass", U), ("Gravel ", U),
    (None, None), ("", None), ("wood", None), ("asphalt;gravel", None),
])
def test_surface_class(value, expected):
    tags = {"highway": "unclassified"} | ({"surface": value} if value is not None else {})
    assert sec.surface_class(tags) == expected


def test_untagged_runs_take_the_longer_neighbouring_stretch():
    # 300 m sealed, an untagged 80 m bridge, then 2 x 400 m of gravel: the gravel stretch is longer.
    labels = sec.settle([S, None, U, U], [300, 80, 400, 400], [S] * 4)
    assert labels == [S, U, U, U]
    assert sec.settle([None, U, S], [100, 60, 900], [S] * 3) == [U, U, S]  # only one neighbour at the start


def test_with_nothing_tagged_tracks_are_unsealed_and_roads_sealed():
    tags = [{"highway": "track"}, {"highway": "unclassified"}]
    out = sec.run_sections(tags, [500.0, 500.0])
    assert [r.surface for r in out] == [U, S]
    assert not any(r.tagged for r in out)


def test_short_runs_join_their_neighbours():
    # Pelverata Road: 10 m concrete culverts on the gravel, and a 30 m paved apron where the seal ends.
    labels = [S, S, U, S, U, S, U]
    lengths = [1200, 30, 800, 10, 600, 10, 900]
    out = sec.settle(labels, lengths, labels)
    assert out == [S, S, U, U, U, U, U]  # the apron is long enough beside the seal; both culverts go
    # A run under 50 m between two long ones of the other class always goes, whatever its class.
    assert sec.settle([U, S, U], [500, 49, 500], [U, S, U]) == [U, U, U]
    assert sec.settle([U, S, U], [500, 50, 500], [U, S, U]) == [U, S, U]


def test_shortest_short_stretch_goes_first():
    # 30 m of gravel then 40 m of seal between long stretches: the 30 m goes first, which leaves 70 m of seal.
    out = sec.settle([S, U, S, U], [1000, 30, 40, 1000], [S] * 4)
    assert out == [S, S, S, U]


def test_bridges_get_no_exception():
    # An untagged 40 m bridge inside a gravel road, beside a 45 m sealed approach: everything short goes.
    tags = [{"highway": "unclassified", "surface": "gravel"}, {"highway": "unclassified", "bridge": "yes"},
            {"highway": "unclassified", "surface": "asphalt"}, {"highway": "unclassified", "surface": "gravel"}]
    out = sec.run_sections(tags, [800.0, 40.0, 45.0, 800.0])
    assert [r.surface for r in out] == [U, U, U, U]
    assert [r.tagged for r in out] == [True, False, True, True]


def test_looking_like_one_lane_never_changes_the_surface():
    tags = [{"highway": "tertiary", "surface": "asphalt"}, {"highway": "tertiary", "surface": "asphalt", "lanes": "1"},
            {"highway": "tertiary", "surface": "gravel"}]
    out = sec.run_sections(tags, [600.0, 30.0, 600.0])
    assert [r.surface for r in out] == [S, S, U]
    # The 30 m single-lane bit (a bridge, say) keeps the centre line of the road around it.
    assert [sec.STYLES[r.style] for r in out] == ["road", "road", "gravel"]
    out = sec.run_sections(tags, [600.0, 300.0, 600.0])
    assert [sec.STYLES[r.style] for r in out] == ["road", "road_plain", "gravel"]


# --- Widths and crown ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tags,unsealed,expected,source", [
    ({"highway": "secondary"}, True, 6.0, "class"),
    ({"highway": "tertiary"}, True, 5.5, "class"),
    ({"highway": "unclassified"}, True, 5.0, "class"),
    ({"highway": "residential"}, True, 5.0, "class"),
    ({"highway": "track"}, True, 4.0, "class"),
    ({"highway": "primary"}, True, 6.5, "class"),
    ({"highway": "unclassified", "lanes": "1"}, True, 4.0, "lanes"),
    ({"highway": "unclassified", "lanes": "2"}, True, 6.0, "lanes"),
    ({"highway": "unclassified", "narrow": "yes"}, True, 4.5, "narrow"),
    ({"highway": "secondary", "width": "3"}, True, 3.0, "width"),
    ({"highway": "unclassified", "width": "2"}, True, 3.0, "width_clamped"),
    ({"highway": "residential", "width": "1"}, False, 3.0, "width_clamped"),
    ({"highway": "secondary", "width": "75"}, True, 6.0, "class"),  # implausible: still ignored
    # Sealed defaults are unchanged, and a single-lane seal is 4.0 m as before.
    ({"highway": "secondary"}, False, 6.4, "class"),
    ({"highway": "tertiary"}, False, 6.0, "class"),
    ({"highway": "unclassified"}, False, 5.8, "class"),
    ({"highway": "unclassified", "lanes": "1"}, False, 4.0, "lanes"),
    ({"highway": "tertiary", "narrow": "yes"}, False, 4.5, "narrow"),
])
def test_widths(tags, unsealed, expected, source):
    assert sec.width_from_tags(tags, unsealed) == (pytest.approx(expected), source)


@pytest.mark.parametrize("tags,unsealed,single", [
    ({"highway": "secondary", "lanes": "1"}, False, True),
    ({"highway": "secondary", "lane_markings": "no"}, False, True),
    ({"highway": "unclassified"}, False, False),  # a sealed unclassified road keeps its centre line
    ({"highway": "unclassified"}, True, True),  # 5.0 m of gravel: one pair of wheel paths
    ({"highway": "tertiary"}, True, False),  # 5.5 m
    ({"highway": "secondary", "narrow": "yes"}, True, True),
    ({"highway": "unclassified", "lanes": "2"}, True, False),
    ({"highway": "secondary", "lanes": "1"}, True, True),
])
def test_what_looks_like_one_lane(tags, unsealed, single):
    w, _ = sec.width_from_tags(tags, unsealed)
    assert sec.single_lane(tags, w, unsealed) is single


def test_gravel_crown_is_steeper_and_eased_across_a_change():
    s = np.arange(0.0, 600.0)
    unsealed = s >= 300
    crown = sec.eased(s, np.where(unsealed, sec.CROWN_UNSEALED, sec.CROWN))
    assert crown[0] == pytest.approx(0.025) and crown[-1] == pytest.approx(0.045)
    assert np.abs(np.diff(crown)).max() < 0.001  # no step: the edge drops by under 2 mm per metre on 4 m
    left, right = sec.half_slopes(np.zeros(len(s)), crown)
    assert left[-1] == pytest.approx(0.045) and right[0] == pytest.approx(0.025)
    # Banked a little in a bend: the high side flattens, the low side keeps the gravel crown.
    hl, lr = sec.half_slopes(np.array([0.02]), np.array([0.045]))
    assert (hl[0], lr[0]) == pytest.approx((0.005, 0.045))


# --- Mesh chunks -----------------------------------------------------------------------------------------------

def _straight_ribbon(n=701):
    s = np.arange(float(n))
    xy = np.column_stack([500000 + s, np.full(n, 5250000.0)])
    flat = np.zeros(n)
    half = np.full(n, 2.0)
    rb = mesh.build_ribbon(xy, np.full(n, 100.0), flat, flat, half, half, np.full(n, 0.025), np.full(n, 0.025),
                           lambda e, nn: np.full(np.shape(e), 99.0))
    return s, rb


def test_one_style_keeps_the_old_200_m_chunks():
    assert mesh.road_chunks(np.zeros(701, dtype=np.int8), 200) == [(0, 200), (200, 400), (400, 600), (600, 700)]


def test_chunks_split_where_the_surface_changes_and_share_the_boundary_row():
    s, rb = _straight_ribbon()
    style = np.where(s < 250, sec.style_code(False, True), sec.style_code(True, True)).astype(np.int8)
    assert mesh.road_chunks(style, 200) == [(0, 200), (200, 250), (250, 450), (450, 650), (650, 700)]
    frame = LocalFrame(500000.0, 5250000.0, 0.0)
    meshes = mesh.ribbon_meshes(rb, s, frame, style=style)
    roads = [m for m in meshes if m.name[1:].startswith(("ROAD", "GRAVEL"))]
    assert [m.name for m in roads] == ["1ROAD_000", "1ROAD_001", "1GRAVEL_002", "1GRAVEL_003", "1GRAVEL_004"]
    assert [m.material for m in roads] == ["road_plain"] * 2 + ["gravel_narrow"] * 3
    # Verges follow the same chunks, so names stay unique and paired.
    assert {m.name for m in meshes if "_V" in m.name} == {f"1GRASS_V{k:03d}{side}" for k in range(5) for side in "LR"}
    # The seal's last row is the gravel's first row, vertex for vertex: no gap, no step.
    seal, gravel = roads[1], roads[2]
    cols = 5
    assert np.array_equal(seal.positions[-cols:], gravel.positions[:cols])
    # Together the chunks cover every metre of road once.
    assert sum(len(m.positions) // cols - 1 for m in roads) == len(s) - 1
    # Gravel repeats its texture every GRAVEL_TILE_M, the seal every ROAD_TILE_M.
    assert seal.uvs[-1, 1] == pytest.approx(250 / textures.ROAD_TILE_M)
    assert gravel.uvs[0, 1] == pytest.approx(250 / textures.GRAVEL_TILE_M)


def test_every_road_material_has_a_texture():
    assert set(mesh.ROAD_TEXTURE) == set(sec.STYLES)
    assert all(t in textures.LIBRARY for t in mesh.ROAD_TEXTURE.values())


# --- surfaces.ini, ext_config.ini, ui_track.json ---------------------------------------------------------------

def _sections(text: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    name = None
    for line in text.split("\r\n"):
        if line.startswith("["):
            name = line.strip("[]")
            out[name] = {}
        elif "=" in line:
            k, v = line.split("=", 1)
            out[name][k.strip()] = v.strip()
    return out


def test_surfaces_ini_adds_gravel_only_when_asked():
    plain = _sections(package.surfaces_ini())
    assert [v["KEY"] for v in plain.values()] == ["ROAD", "GRASS"]
    text = package.surfaces_ini(gravel=True)
    assert "\n" not in text.replace("\r\n", "")
    gravel = _sections(text)["SURFACE_2"]
    # Friction 0.80, the maintainer's choice: at 0.88 the AI drove the gravel at near-tarmac pace (in game, 5 Nov).
    assert gravel == {
        "KEY": "GRAVEL", "FRICTION": "0.80", "DAMPING": "0.01", "WAV": "gravel.wav", "WAV_PITCH": "0",
        "FF_EFFECT": "NULL", "DIRT_ADDITIVE": "0.8", "IS_VALID_TRACK": "1", "BLACK_FLAG_TIME": "0",
        "SIN_HEIGHT": "0.01", "SIN_LENGTH": "0.5", "IS_PITLANE": "0", "VIBRATION_GAIN": "0.1",
        "VIBRATION_LENGTH": "0.35", "_EXT_SURFACE_TYPE": "GRAVEL", "_EXT_SURFACE_TYPE_MODIFIER": "REGULAR",
        "_EXT_PERLIN_NOISE": "1", "_EXT_PERLIN_OCTAVES": "3", "_EXT_PERLIN_PERSISTENCE": "0.5",
    }
    # Same keys as ROAD and GRASS, plus CSP's: nothing vanilla AC doesn't know outside the _EXT_ ones.
    assert set(gravel) - set(package.SURFACES[0]) == {k for k in gravel if k.startswith("_EXT_")}
    assert "extended" not in text and "KEY=WALL" not in text
    # CSP's SurfacesFX treats it as loose by its numbers too, not only by its type (surface-tweaks).
    assert float(gravel["DIRT_ADDITIVE"]) >= 0.7 and float(gravel["FRICTION"]) < 0.9
    # No key overlaps another, so every mesh name matches exactly one.
    keys = package.SURFACE_KEYS + ("WALL",)
    assert all(a == b or a not in b for a in keys for b in keys)
    assert package.surface_keys(text) == {"ROAD", "GRASS", "GRAVEL"}


def test_ext_config_tyres_fx_and_gravel_materials():
    colours = textures.tyre_dirt_colours()
    text = package.ext_config_ini(grass_materials=["grass", "terrain"], road_materials=["road", "road_plain"],
                                  version="t", gravel_materials=["gravel", "gravel_narrow"],
                                  grassy_gravel=["gravel_narrow"], dirt_colours=colours)
    ini = _sections(text)
    tyres = ini["TYRES_FX"]
    pairs = [f"{k}_COLOR_{ab}" for k in ("DIRT", "GRASS", "SOIL") for ab in "AB"]
    assert list(tyres)[:6] == pairs  # all three pairs: a partial section would drop CSP's generic colours
    for key in pairs:
        assert tyres[key] == f"'{colours[key]}'" and len(colours[key]) == 7 and colours[key].startswith("#")
    assert {tyres[k] for k in ("DIRT_GAIN", "DIRT_FADE_GAIN", "GRASS_GAIN", "GRASS_FADE_GAIN")} == {"1.0"}
    # A is the texture's mean, B darker.
    a, b = (np.array([int(colours[f"DIRT_COLOR_{x}"][i:i + 2], 16) for i in (1, 3, 5)]) for x in "AB")
    assert np.allclose(a, textures.mean_colour(textures.gravel_road()), atol=0.6) and (b < a).all()
    assert ini["GRASS_FX"] == {"GRASS_MATERIALS": "grass, terrain, gravel_narrow",
                               "OCCLUDING_MATERIALS": "road, road_plain, gravel"}
    assert ini["RAIN_FX"] == {"PUDDLES_MATERIALS": "road, road_plain, gravel, gravel_narrow",
                              "SOAKING_MATERIALS": "road, road_plain",
                              "ROUGH_MATERIALS": "grass, terrain, gravel, gravel_narrow"}
    assert "tilingfix" not in text
    sealed = _sections(package.ext_config_ini(grass_materials=["grass"], road_materials=["road"], version="t"))
    assert "TYRES_FX" not in sealed and sealed["GRASS_FX"]["OCCLUDING_MATERIALS"] == "road"


@pytest.mark.parametrize("unsealed,length,tags", [
    (0.0, 6800.0, []), (5200.0, 6800.0, ["gravel", "rally"]), (100.0, 6800.0, ["gravel"]),
    (3400.0, 6800.0, ["gravel"]),  # exactly half: not more than half
])
def test_ui_track_tags(unsealed, length, tags):
    assert package.surface_tags(unsealed, length) == tags
    text = package.ui_track_json(name="x", description="y", tags=["point to point", *package.surface_tags(
        unsealed, length)], lat=-42.0, lon=147.0, country="Australia", city="Tasmania", length_m=length,
        width_m=(4.0, 4.0), run="point-to-point", version="t", year=2026, pitboxes=5)
    assert json.loads(text)["tags"] == ["point to point", *tags]


def test_centreline_surface_runs():
    s = np.arange(0.0, 1001.0)
    runs = track_stage.surface_runs(s, s >= 600)
    assert runs == [{"from_m": 0.0, "to_m": 600.0, "surface_class": "sealed"},
                    {"from_m": 600.0, "to_m": 1000.0, "surface_class": "unsealed"}]
    assert track_stage.surface_runs(s, np.zeros(len(s), bool)) == [
        {"from_m": 0.0, "to_m": 1000.0, "surface_class": "sealed"}]


# --- AI pace on gravel -----------------------------------------------------------------------------------------

def _line(length: float, n: int, arc) -> ai_line.AiLine:
    """A stand-in AI line from END_INSET_M to AI_END_SHORT_M before the end, `arc(station)` metres along itself:
    not `ai_line.build`, so the racing line's own changes can't move these fractions."""
    station = np.linspace(ai_line.END_INSET_M, length - layout.AI_END_SHORT_M, n)
    along = arc(station)
    return ai_line.AiLine(pos=np.zeros((n, 3)), length=along - along[0], extra=np.zeros((n, ai_line.EXTRA_FIELDS)),
                          station=station)


def test_unsealed_runs_are_the_gravel_meshes_in_driving_order():
    s = 100.0 + np.arange(0.0, 3001.0)  # the centreline needn't start at 0
    unsealed = ((s >= 700) & (s < 1500)) | (s >= 2200)
    assert track_stage.unsealed_runs(s, unsealed) == [(600.0, 1400.0), (2100.0, 3000.0)]
    # The reverse layout starts at the far end: its first gravel is the forward layout's last.
    assert track_stage.unsealed_runs(s, unsealed, reverse=True) == [(0.0, 900.0), (1600.0, 2400.0)]
    assert track_stage.unsealed_runs(s, np.zeros(len(s), bool)) == []
    # The same stations the 1GRAVEL_### chunks cover.
    style = np.where(unsealed, 2, 0).astype(np.int8)
    rows = sorted({r for a, b in mesh.road_chunks(style, 200) if sec.is_unsealed(style[a]) for r in range(a, b + 1)})
    starts = [r for r in rows if r - 1 not in rows]
    ends = [r for r in rows if r + 1 not in rows]
    assert [(s[a] - s[0], s[b] - s[0]) for a, b in zip(starts, ends)] == track_stage.unsealed_runs(s, unsealed)


def test_gravel_gets_a_pace_hint_up_to_the_finish():
    length = 6000.0
    finish = layout.finish_distance(length)
    assert finish == 5750.0
    # The racing line cuts the bends of the first 3 km, so it's shorter than the road there.
    line = _line(length, 1200, lambda d: np.where(d < 3000, 0.97 * d, 0.97 * 3000 + (d - 3000)))
    # Gravel from the start, a run in the middle, one across the finish and one wholly after it.
    runs = [(0.0, 400.0), (1000.0, 2500.0), (3200.0, 5800.0), (5870.0, 6000.0)]
    hints = ai_line.pace_hints(length, finish, unsealed=runs, line=line)

    def frac(d):  # a fraction of the line's own length, as the run-off hints use
        return float(np.interp(d, line.station, line.length) / line.length[-1])

    pace = layout.AI_GRAVEL_PACE
    assert pace == 0.8
    gravel, runoff = hints[:3], hints[3:]
    expected = [(0.0, frac(400), pace), (frac(1000), frac(2500), pace), (frac(3200), frac(finish), pace)]
    assert np.array(gravel) == pytest.approx(np.array(expected))
    assert frac(1000) != pytest.approx((1000 - ai_line.END_INSET_M) / (length - layout.AI_END_SHORT_M - 1.5))
    # The run-off ramp is untouched and wins after the finish: the gravel stops where it starts, and the run after
    # the finish gets no hint.
    assert runoff == ai_line.runoff_hints(length, finish, line=line)
    assert gravel[-1][1] == runoff[0][0] == pytest.approx(frac(finish))
    assert all(a[1] <= b[0] and a[0] < a[1] for a, b in zip(hints, hints[1:]))
    text = package.ai_hints_ini(hints)
    ini = _sections(text)
    assert list(ini) == [f"HINT_{i}" for i in range(6)]
    assert [v["VALUE"] for v in ini.values()] == ["0.8", "0.8", "0.8", "0.5", "0.2", "0"]
    starts = [float(v["START"]) for v in ini.values()]
    assert starts == sorted(starts) and ini["HINT_2"]["END"] == ini["HINT_3"]["START"]
    # The reverse layout's runs, in its own driving order, give the same kind of hints.
    s = np.arange(0.0, length + 1)
    unsealed = ((s >= 1000) & (s < 2500)) | (s >= 5600)
    back = track_stage.unsealed_runs(s, unsealed, reverse=True)
    assert back == [(0.0, 400.0), (3500.0, 5000.0)]
    back_hints = ai_line.pace_hints(length, finish, unsealed=back, line=line)
    assert [p for *_, p in back_hints] == [pace, pace, 0.5, 0.2, 0.0]
    assert all(a[1] <= b[0] for a, b in zip(back_hints, back_hints[1:]))


# Written by main before gravel had hints (520b5c9), for the same lines and road.
SEALED_HINTS = {
    "line": "[HINT_0]\r\nSTART=0.9103\r\nEND=0.9359\r\nVALUE=0.5\r\n\r\n[HINT_1]\r\nSTART=0.9359\r\nEND=0.9743\r\n"
            "VALUE=0.2\r\n\r\n[HINT_2]\r\nSTART=0.9743\r\nEND=1.0000\r\nVALUE=0\r\n",
    "road": "[HINT_0]\r\nSTART=0.9110\r\nEND=0.9364\r\nVALUE=0.5\r\n\r\n[HINT_1]\r\nSTART=0.9364\r\nEND=0.9746\r\n"
            "VALUE=0.2\r\n\r\n[HINT_2]\r\nSTART=0.9746\r\nEND=1.0000\r\nVALUE=0\r\n",
}


@pytest.mark.parametrize("which", ["line", "road"])
@pytest.mark.parametrize("reverse", [False, True])
def test_a_sealed_track_writes_the_same_ai_hints_as_before(which, reverse):
    length = 2400.0
    line = _line(length, 470, lambda d: 0.995 * d - 1.5 * np.sin(d / 150.0)) if which == "line" else None
    s = np.arange(0.0, length + 1)
    runs = track_stage.unsealed_runs(s, np.zeros(len(s), bool), reverse=reverse)
    hints = ai_line.pace_hints(length, layout.finish_distance(length), unsealed=runs, line=line)
    assert package.ai_hints_ini(hints) == SEALED_HINTS[which]


# --- Textures --------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gravel_textures():
    return {lanes: np.asarray(textures.gravel_road(lanes)).astype(float) for lanes in (1, 2)}


def _block_jumps(img: np.ndarray, block: int = 32) -> tuple[float, float]:
    """Mean colour of each `block` rows, then the jump from block to block: across the wrap from the last block
    to the first (where the texture repeats along the road), and the largest anywhere else."""
    means = img.reshape(img.shape[0] // block, block, *img.shape[1:]).mean(axis=1)
    jumps = np.abs(np.diff(np.concatenate([means, means[:1]]), axis=0)).mean(axis=(1, 2))
    return float(jumps[-1]), float(jumps[:-1].max())


@pytest.mark.parametrize("lanes", [1, 2])
def test_gravel_tiles_along_the_road(gravel_textures, lanes):
    img = gravel_textures[lanes]
    assert img.shape == (*textures.GRAVEL_SIZE, 3)
    wrap, inside = _block_jumps(img)
    assert wrap <= inside  # the repeat is no more visible than any step within the texture
    # The check has teeth: the same texture cut short of a whole period has a seam the eye would see.
    cut = img[: img.shape[0] * 5 // 8]
    wrap, inside = _block_jumps(cut)
    assert wrap > inside


def _green(img: np.ndarray) -> np.ndarray:
    """Pixels where green clearly leads: what CSP's Grass FX colour mask grows blades on."""
    return img[..., 1] > np.maximum(img[..., 0], img[..., 2]) + 4


def test_single_lane_gravel_has_a_grass_strip_down_the_middle(gravel_textures):
    green = _green(gravel_textures[1]).mean(axis=0)  # share of rows, per column across the road
    cols = len(green)
    x = ((np.arange(cols) + 0.5) / cols - 0.5) * textures.GRAVEL_WIDTH_M[1]
    centre, paths = green[np.abs(x) < 0.08], green[np.abs(np.abs(x) - 0.8) < 0.2]
    assert centre.mean() > 0.4  # patchy, but there
    assert paths.max() == 0.0  # never in the wheel paths
    assert green[np.abs(x) > 2.2].mean() > 0.2  # green fringes at the edges


def test_two_lane_gravel_has_no_green_and_darker_wheel_paths(gravel_textures):
    img = gravel_textures[2]
    assert not _green(img).any()
    cols = img.shape[1]
    x = ((np.arange(cols) + 0.5) / cols - 0.5) * textures.GRAVEL_WIDTH_M[2]
    lum = img.mean(axis=(0, 2))
    paths = np.zeros(cols, bool)
    for c in (-1.5, 1.5):
        paths |= np.abs(np.abs(x - c) - 0.8) < 0.15
    ridges = np.abs(np.abs(x) - 1.5) < 0.1
    assert lum[paths].mean() < 0.97 * lum[ridges].mean()


def test_single_lane_seal_has_edge_lines_and_no_centre_line():
    plain = np.asarray(textures.road_plain()).astype(float).mean(axis=2)
    marked = np.asarray(textures.road_marked(width_m=4.0)).astype(float).mean(axis=2)
    centre = slice(250, 262)
    assert plain[:, centre].max() < 180 < marked[:, centre].max()
    edge = int(round(textures.EDGE_INSET_M / 4.0 * 512))
    assert plain[:, edge - 2:edge + 3].max() > 180


# --- The grid on a narrow road ---------------------------------------------------------------------------------

def _straight(half: float, crown: float, length: float = 1000.0) -> RoadModel:
    s = np.arange(0.0, length)
    flat = np.zeros(s.size)
    return RoadModel.from_arrays(s, np.column_stack([s, flat]), np.full(s.size, 100.0), flat, flat, flat,
                                 np.full(s.size, half), np.full(s.size, half), np.full(s.size, crown),
                                 np.full(s.size, crown), False)


@pytest.mark.parametrize("tags", [{"highway": "unclassified", "lanes": "1"}, {"highway": "unclassified",
                                                                              "narrow": "yes"},
                                  {"highway": "unclassified"}])
def test_a_4_to_5_m_gravel_road_gets_five_spots_in_single_file(tags):
    w, _ = sec.width_from_tags(tags, unsealed=True)
    assert 4.0 <= w <= 5.0
    spots = grid(_straight(w / 2, sec.CROWN_UNSEALED))
    assert spots == [(42.0, 0.0), (34.0, 0.0), (26.0, 0.0), (18.0, 0.0), (10.0, 0.0)]
    assert layout.GATE_FROM_END_M == 50.0
