"""The small text and image files around the KN5: ini files, ui_track.json, map and UI images."""

from __future__ import annotations

import json
import math

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
SURFACE_KEYS = tuple(s["KEY"] for s in SURFACES)


def ini(sections: list[tuple[str, dict]]) -> str:
    lines = []
    for name, values in sections:
        lines.append(f"[{name}]")
        lines.extend(f"{k}={v}" for k, v in values.items())
        lines.append("")
    return CRLF.join(lines)


def surfaces_ini() -> str:
    return ini([(f"SURFACE_{i}", s) for i, s in enumerate(SURFACES)])


def models_ini(kn5_files: list[str]) -> str:
    return ini([(f"MODEL_{i}", {"FILE": f, "POSITION": "0,0,0", "ROTATION": "0,0,0"}) for i, f in enumerate(kn5_files)])


def lighting_ini() -> str:
    return ini([("LIGHTING", {"SUN_PITCH_ANGLE": "50", "SUN_HEADING_ANGLE": "16"})])


def groove_ini() -> str:
    return ini([("HEADER", {"GROOVES_NUMBER": "0"})])


def crew_ini() -> str:
    return ini([("HEADER", {"SIDE": "1"})])


def compass(heading_rad: float) -> str:
    """Heading (east = 0, counter-clockwise) to a 8-point compass label."""
    bearing = (90 - math.degrees(heading_rad)) % 360
    return ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][
        int((bearing + 22.5) // 45) % 8]


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
    pts = [tuple(p) for p in px]
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
    pts = [tuple(p) for p in _fit(xz, size, 12)]
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
    pts = [tuple(p) for p in px]
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


def ext_config_ini(*, grass_materials: list[str], road_materials: list[str], version: str,
                   water_materials: list[str] | None = None, water_type: str = "LAKE",
                   tree_list: str | None = None, tree_surfaces: list[str] | None = None,
                   card_meshes: str | None = None) -> str:
    """Custom Shaders Patch track config (`extension/ext_config.ini`). Vanilla AC ignores it.

    - Grass FX grows procedural grass on our grass materials. CSP's default
      colour mask means it follows the green patches of the texture. The road
      occludes it so none grows through the asphalt.
    - Rain FX puddles and soaks the road and treats grass as rough.
    - Water surfaces get CSP's water shader (waves, reflections), as LAKE or SEA. The shader's template
      comes from CSP's own common/materials_track.ini.
    - Tree FX draws the 3D trees in `tree_list` (relative to the extension folder), standing them on
      `tree_surfaces`. The KN5's tree cards (`card_meshes`) are still drawn as well; the section that
      would hide them under CSP is written commented out, ready to switch on once the 3D trees are
      checked in game.
    """
    water = [("INCLUDE: common/materials_track.ini", {}),
             ("Material_Water", {"Materials": ", ".join(water_materials), "Type": water_type})] if water_materials else []
    trees = [("TREES", {"LIST_0": tree_list, **({"SURFACE_MATERIALS": ", ".join(tree_surfaces)} if tree_surfaces else {})})
             ] if tree_list else []
    text = ini(water + [
        ("ABOUT", {"AUTHOR": "mapcreator", "VERSION": version}),
        ("GRASS_FX", {"GRASS_MATERIALS": ", ".join(grass_materials),
                      "OCCLUDING_MATERIALS": ", ".join(road_materials)}),
        ("RAIN_FX", {"PUDDLES_MATERIALS": ", ".join(road_materials),
                     "SOAKING_MATERIALS": ", ".join(road_materials),
                     "ROUGH_MATERIALS": ", ".join(grass_materials)}),
    ] + trees)
    if tree_list and card_meshes:
        text += CRLF.join([
            "; The flat tree cards in the KN5 are drawn as well as the 3D trees above, for players without",
            "; Custom Shaders Patch. To hide the cards while CSP is running, remove the semicolons below.",
            "; [MESH_ADJUSTMENT_TREE_CARDS]",
            f"; MESHES = {card_meshes}",
            "; IS_RENDERABLE = 0",
            "; CAST_SHADOWS = 0",
            "",
        ])
    return text


def ui_track_json(*, name: str, description: str, tags: list[str], lat: float, lon: float, country: str,
                  city: str, length_m: float, width_m: tuple[float, float], run: str, version: str,
                  year: int, priority: int | None = None) -> str:
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
        "pitboxes": "1",
        "run": run,
        "author": "mapcreator",
        "version": version,
        "year": year,
    }
    if priority is not None:
        data["priority"] = priority  # CM picks the highest-priority layout as the default
    return json.dumps(data, indent=2, ensure_ascii=True)
