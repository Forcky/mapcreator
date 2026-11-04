"""The small text and image files around the KN5: ini files, ui_track.json, map and UI images."""

from __future__ import annotations

import json
import math
import re

import numpy as np
from PIL import Image, ImageDraw

MAP_MARGIN = 20
MAP_MAX = 1600
CRLF = "\r\n"

# ROAD/GRASS values follow the Kunos-style set used by open-source generators.
# There is deliberately no WALL entry: defining one is reported to make walls driveable.
SURFACES = [
    {"KEY": "ROAD", "FRICTION": "0.98", "DAMPING": "0", "WAV": "", "WAV_PITCH": "0", "FF_EFFECT": "NULL",
     "DIRT_ADDITIVE": "0", "IS_VALID_TRACK": "1", "BLACK_FLAG_TIME": "0", "SIN_HEIGHT": "0", "SIN_LENGTH": "0",
     "IS_PITLANE": "0", "VIBRATION_GAIN": "0", "VIBRATION_LENGTH": "0"},
    {"KEY": "GRASS", "FRICTION": "0.75", "DAMPING": "0", "WAV": "grass.wav", "WAV_PITCH": "0", "FF_EFFECT": "NULL",
     "DIRT_ADDITIVE": "1", "IS_VALID_TRACK": "0", "BLACK_FLAG_TIME": "0", "SIN_HEIGHT": "0.03", "SIN_LENGTH": "0.5",
     "IS_PITLANE": "0", "VIBRATION_GAIN": "0.2", "VIBRATION_LENGTH": "0.6",
     # Custom Shaders Patch: tell SurfacesFX this is grass (ignored by vanilla AC).
     "_EXT_SURFACE_TYPE": "GRASS"},
]
# Unsealed road (1GRAVEL_### meshes), written only for a track that has some. AC's own loose surfaces (Kunos's
# Laguna Seca and Zandvoort GRAVEL, Black Cat County's OFF: friction 0.7-0.8, dirt 1) are run-off built to slow a
# car down. This is a road to drive at pace, in the range community gravel stages settle on (friction 0.87-0.90,
# damping 0-0.02, dirt 0.8-1; assettocorsamods.net "new CSP +Gravel" and "How to use surfaces in surfaces.ini"):
# - FRICTION 0.88: under 0.89, above which one modder found gravel made the tyres squeal, and under 0.9, below
#   which CSP's ParticlesFX makes "dusty tracks" (CSP 0.2.5 changelog). Dirt on the tyres takes off roughly
#   another 20% (one community skidpad test).
# - DIRT_ADDITIVE 0.8: dirt builds up on the tyres; at least 0.7 makes CSP's SurfacesFX treat it as loose
#   (cup.acstuff.club/docs/csp/tracks/surface-tweaks).
# - WAV gravel.wav, which Kunos ships (Black Cat County's OFF uses it). Never grass.wav: CSP draws the dirt of a
#   grass.wav surface as grass (docs/csp/tracks/dirt-on-tyres).
# - A 1 cm ripple at a 2 m wavelength (SIN_LENGTH is inverse: 0.5 = 2 m), as CSP's Perlin noise instead of a sine.
# - VIBRATION between Black Cat County's OFF (0.05 / 0.35) and Laguna Seca's GRAVEL (0.5 / 0.6).
# - _EXT_* are CSP's (surface-tweaks); vanilla AC ignores them. No `WAV_PITCH=extended-0`: extended track physics
#   isn't needed for these, and vanilla AC would crash on it.
# All of it is to confirm in game.
GRAVEL_SURFACE = {
    "KEY": "GRAVEL", "FRICTION": "0.88", "DAMPING": "0.01", "WAV": "gravel.wav", "WAV_PITCH": "0",
    "FF_EFFECT": "NULL", "DIRT_ADDITIVE": "0.8", "IS_VALID_TRACK": "1", "BLACK_FLAG_TIME": "0",
    "SIN_HEIGHT": "0.01", "SIN_LENGTH": "0.5", "IS_PITLANE": "0", "VIBRATION_GAIN": "0.1", "VIBRATION_LENGTH": "0.35",
    "_EXT_SURFACE_TYPE": "GRAVEL", "_EXT_SURFACE_TYPE_MODIFIER": "REGULAR", "_EXT_PERLIN_NOISE": "1",
    "_EXT_PERLIN_OCTAVES": "3", "_EXT_PERLIN_PERSISTENCE": "0.5",
}
# Every KEY a track of ours can define. A physics mesh name must contain exactly one of these (or WALL).
SURFACE_KEYS = tuple(s["KEY"] for s in SURFACES) + (GRAVEL_SURFACE["KEY"],)
ROAD_KEYS = ("ROAD", "GRAVEL")  # the drivable road: spawns, the AI line and colliders are checked against both


def ini(sections: list[tuple[str, dict]]) -> str:
    lines = []
    for name, values in sections:
        lines.append(f"[{name}]")
        lines.extend(f"{k}={v}" for k, v in values.items())
        lines.append("")
    return CRLF.join(lines)


def surfaces_ini(gravel: bool = False) -> str:
    """`data/surfaces.ini`: ROAD and GRASS, plus GRAVEL as SURFACE_2 for a track with unsealed road."""
    return ini([(f"SURFACE_{i}", s) for i, s in enumerate(SURFACES + ([GRAVEL_SURFACE] if gravel else []))])


def surface_keys(text: str) -> set[str]:
    """The KEYs a `surfaces.ini` defines."""
    return set(re.findall(r"^KEY=(\S+)\s*$", text, re.M))


def models_ini(kn5_files: list[str]) -> str:
    return ini([(f"MODEL_{i}", {"FILE": f, "POSITION": "0,0,0", "ROTATION": "0,0,0"}) for i, f in enumerate(kn5_files)])


def lighting_ini() -> str:
    return ini([("LIGHTING", {"SUN_PITCH_ANGLE": "50", "SUN_HEADING_ANGLE": "16"})])


def groove_ini() -> str:
    return ini([("HEADER", {"GROOVES_NUMBER": "0"})])


def ai_hints_ini(sections: list[tuple[float, float, float]]) -> str:
    """`data/ai_hints.ini`: one `[HINT_n]` per (start, end, pace). START and END are fractions of the AI spline
    (0 at its start, 1 at its end), and VALUE scales the AI's speed between them (the format other tracks use;
    checked against AC modding guides, Nov 2026)."""
    def clamp(v):
        return min(max(v, 0.0), 1.0)

    return ini([(f"HINT_{i}", {"START": f"{clamp(a):.4f}", "END": f"{clamp(b):.4f}", "VALUE": f"{pace:g}"})
                for i, (a, b, pace) in enumerate(sections)])


def crew_ini() -> str:
    return ini([("HEADER", {"SIDE": "1"})])


def compass(heading_rad: float) -> str:
    """Heading (east = 0, counter-clockwise) to a 8-point compass label."""
    bearing = (90 - math.degrees(heading_rad)) % 360
    return ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][
        int((bearing + 22.5) // 45) % 8]


def _drawable(px: np.ndarray, tolerance: float = 0.5) -> list[tuple[float, float]]:
    """A polyline in pixels, simplified to `tolerance` px. Drawing all 63,000 points of a 63 km road with
    rounded joints took 28 s for images at most 1,600 px across; the simplified line looks the same."""
    import shapely

    # Plain Douglas-Peucker: a drawn line may cross itself, and keeping its topology would keep every point.
    line = shapely.LineString(px).simplify(tolerance, preserve_topology=False) if len(px) > 2 else None
    return [tuple(p) for p in (np.asarray(line.coords) if line is not None and not line.is_empty else px)]


def map_files(xz: np.ndarray) -> tuple[Image.Image, str]:
    """map.png and map.ini from the centreline in AC X/Z (north up, as the game draws it)."""
    lo, hi = xz.min(axis=0), xz.max(axis=0)
    span = float(max(hi - lo))
    scale = max(1.0, span / (MAP_MAX - 2 * MAP_MARGIN))  # metres per pixel
    w = int(math.ceil((hi[0] - lo[0]) / scale)) + 2 * MAP_MARGIN
    h = int(math.ceil((hi[1] - lo[1]) / scale)) + 2 * MAP_MARGIN
    x_off = -lo[0] + MAP_MARGIN * scale
    z_off = -lo[1] + MAP_MARGIN * scale
    px = (xz + [x_off, z_off]) / scale
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    pts = _drawable(px)
    draw.line(pts, fill=(0, 0, 0, 255), width=9, joint="curve")
    draw.line(pts, fill=(255, 255, 255, 255), width=5, joint="curve")
    text = ini([("PARAMETERS", {
        "WIDTH": str(w), "HEIGHT": str(h), "MARGIN": str(MAP_MARGIN), "SCALE_FACTOR": f"{scale:.6f}",
        "MAX_SIZE": str(MAP_MAX), "X_OFFSET": f"{x_off:.4f}", "Z_OFFSET": f"{z_off:.4f}", "DRAWING_SIZE": "5"})])
    return im, text


def _fit(xz: np.ndarray, size: tuple[int, int], pad: int) -> np.ndarray:
    lo, hi = xz.min(axis=0), xz.max(axis=0)
    s = min((size[0] - 2 * pad) / max(hi[0] - lo[0], 1e-6), (size[1] - 2 * pad) / max(hi[1] - lo[1], 1e-6))
    centre = (lo + hi) / 2
    return (xz - centre) * s + np.array(size) / 2


def outline_png(xz: np.ndarray, size=(365, 192)) -> Image.Image:
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    pts = _drawable(_fit(xz, size, 12))
    ImageDraw.Draw(im).line(pts, fill=(255, 255, 255, 255), width=4, joint="curve")
    return im


PREVIEW_SIZE = (355, 200)
PREVIEW_BACKGROUND = (34, 38, 42)


def preview_frame(xz: np.ndarray, size=PREVIEW_SIZE, margin: float = 0.08) -> tuple[float, float, float, float]:
    """An AC X/Z window with the preview's aspect ratio that holds the whole route plus a margin."""
    lo, hi = xz.min(axis=0), xz.max(axis=0)
    centre = (lo + hi) / 2
    w, h = (hi - lo) * (1 + 2 * margin)
    aspect = size[0] / size[1]
    if w / max(h, 1e-6) < aspect:
        w = h * aspect
    else:
        h = w / aspect
    return centre[0] - w / 2, centre[0] + w / 2, centre[1] - h / 2, centre[1] + h / 2


def preview_png(shade_rgb: np.ndarray, alpha: np.ndarray, frame: tuple[float, float, float, float],
                xz: np.ndarray, size=PREVIEW_SIZE, supersample: int = 3) -> Image.Image:
    """Hillshaded terrain fading out away from the road, the road in yellow, the start in green.

    `shade_rgb` and `alpha` are rasters covering `frame` (north at the top).
    """
    big_size = (size[0] * supersample, size[1] * supersample)
    terrain = Image.fromarray(shade_rgb).convert("RGB").resize(big_size, Image.Resampling.BILINEAR)
    mask = Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8)).resize(big_size, Image.Resampling.BILINEAR)
    img = Image.new("RGB", big_size, PREVIEW_BACKGROUND)
    img.paste(terrain, (0, 0), mask)
    x0, x1, z0, z1 = frame
    px = (xz - [x0, z0]) * [big_size[0] / (x1 - x0), big_size[1] / (z1 - z0)]
    d = ImageDraw.Draw(img)
    pts = _drawable(px)
    d.line(pts, fill=(20, 20, 20), width=5 * supersample, joint="curve")
    d.line(pts, fill=(250, 210, 60), width=3 * supersample, joint="curve")
    sx, sz = px[0]
    r = 5 * supersample
    d.ellipse([sx - r, sz - r, sx + r, sz + r], fill=(40, 200, 80), outline=(0, 0, 0), width=supersample)
    return img.resize(size, Image.Resampling.LANCZOS)


def hillshade(z: np.ndarray, cell: float) -> np.ndarray:
    """RGB hillshade with a gentle hypsometric tint; NaN becomes dark grey."""
    zz = np.where(np.isfinite(z), z, np.nanmin(z) if np.isfinite(z).any() else 0)
    gy, gx = np.gradient(zz, cell)
    az, alt = math.radians(315), math.radians(45)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)
    t = (zz - zz.min()) / max(np.ptp(zz), 1e-6)
    low, high = np.array([96, 130, 80]), np.array([170, 160, 120])
    rgb = (low * (1 - t[..., None]) + high * t[..., None]) * (0.35 + 0.65 * shade[..., None])
    rgb[~np.isfinite(z)] = (40, 40, 40)
    return np.clip(rgb, 0, 255).astype(np.uint8)


TYRES_FX_GAINS = {"DIRT_GAIN": "1.0", "DIRT_FADE_GAIN": "1.0", "GRASS_GAIN": "1.0", "GRASS_FADE_GAIN": "1.0"}


def ext_config_ini(*, grass_materials: list[str], road_materials: list[str], version: str,
                   gravel_materials: list[str] = (), grassy_gravel: list[str] = (),
                   dirt_colours: dict[str, str] | None = None,
                   water_materials: list[str] | None = None, water_type: str = "LAKE",
                   tree_list: str | None = None, tree_surfaces: list[str] | None = None) -> str:
    """Custom Shaders Patch track config (`extension/ext_config.ini`). Vanilla AC ignores it.

    - Grass FX grows procedural grass on our grass materials. CSP's default colour mask means it follows the
      green patches of the texture. The sealed road occludes it so none grows through the asphalt.
      - Gravel with a grass strip (`grassy_gravel`, single-lane) is a grass material too, so blades grow on the
        strip; gravel tracks in CSP's own configs list their road materials there the same way (karelia). Other
        gravel occludes, as the seal does.
    - Rain FX puddles and soaks the seal and treats grass as rough. Gravel is road, so it gets puddles too
      (PUDDLES_MATERIALS is "the key part for physics for generating puddles"), and it's rough: ROUGH_MATERIALS
      "won't get any reflections", only darken when wet (cup.acstuff.club/docs/csp/tracks/rainfx).
    - `[TYRES_FX]` (with `dirt_colours`, for a track with gravel) colours dust and dirty tyres. A track's own
      section switches off CSP's generic one entirely (CSP 0.2.4 changelog), so it carries all three colour
      pairs, and the generic's four gains at 1.0.
    - No tiling fix (`ksPerPixel_tilingfix`) on the gravel: it offsets the texture at random and would break the
      wheel paths that run along it.
    - Water surfaces get CSP's water shader (waves, reflections), as LAKE or SEA. The shader's template
      comes from CSP's own common/materials_track.ini.
    - Tree FX draws the 3D trees in `tree_list` (relative to the extension folder), standing them on
      `tree_surfaces`. They are the track's only trees.
    - No `[WIND] DYNAMIC_CABLES` for the power line wires: CSP 0.2.11 redraws them as tubes about 8 times thicker
      than the mesh (in game, 4 Nov), and they look right without it.
    """
    water = [("INCLUDE: common/materials_track.ini", {}),
             ("Material_Water", {"Materials": ", ".join(water_materials), "Type": water_type})] if water_materials else []
    trees = [("TREES", {"LIST_0": tree_list, **({"SURFACE_MATERIALS": ", ".join(tree_surfaces)} if tree_surfaces else {})})
             ] if tree_list else []
    tyres = [("TYRES_FX", {**{k: f"'{v}'" for k, v in dirt_colours.items()}, **TYRES_FX_GAINS})] if dirt_colours else []
    occluding = list(road_materials) + [m for m in gravel_materials if m not in grassy_gravel]
    text = ini(water + [
        ("ABOUT", {"AUTHOR": "mapcreator", "VERSION": version}),
        ("GRASS_FX", {"GRASS_MATERIALS": ", ".join(list(grass_materials) + list(grassy_gravel)),
                      "OCCLUDING_MATERIALS": ", ".join(occluding)}),
        ("RAIN_FX", {"PUDDLES_MATERIALS": ", ".join(list(road_materials) + list(gravel_materials)),
                     "SOAKING_MATERIALS": ", ".join(road_materials),
                     "ROUGH_MATERIALS": ", ".join(list(grass_materials) + list(gravel_materials))}),
    ] + tyres + trees)
    return text


RALLY_SHARE = 0.5  # a track more than this share unsealed is tagged `rally` as well as `gravel`


def surface_tags(unsealed_m: float, length_m: float) -> list[str]:
    """`ui_track.json` tags for the road's surface: `gravel` with any unsealed road, and `rally` too when more
    than half of it is unsealed."""
    if unsealed_m <= 0:
        return []
    return ["gravel"] + (["rally"] if unsealed_m > RALLY_SHARE * length_m else [])


def ui_track_json(*, name: str, description: str, tags: list[str], lat: float, lon: float, country: str,
                  city: str, length_m: float, width_m: tuple[float, float], run: str, version: str,
                  year: int, priority: int | None = None, pitboxes: int = 1) -> str:
    w0, w1 = width_m
    width = f"{w0:.1f}" if abs(w1 - w0) < 0.05 else f"{w0:.1f}-{w1:.1f}"
    data = {
        "name": name,
        "description": description,
        "tags": tags,
        "geotags": [f"{lat:.5f}", f"{lon:.5f}"],
        "country": country,
        "city": city,
        # CM parses length with a unit (km / mi / m); a bare number isn't read.
        "length": f"{length_m / 1000:.2f} km",
        "width": width,
        "pitboxes": str(pitboxes),  # one per AC_PIT_k; CM offers this many cars less one as AI opponents
        "run": run,
        "author": "mapcreator",
        "version": version,
        "year": year,
    }
    if priority is not None:
        data["priority"] = priority  # CM picks the highest-priority layout as the default
    return json.dumps(data, indent=2, ensure_ascii=True)
