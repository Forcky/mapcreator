"""Stage 5: assemble the installable AC track folder (KN5s, AI lines, data and UI files) and zip it."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import zipfile
from collections import Counter

import numpy as np

from mapcreator import __version__, buildinfo
from mapcreator.config import Settings
from mapcreator.core import ai_line, furniture, grass, grassfx, kn5, package, textures, treefx, validate
from mapcreator.core.layout import AI_END_SHORT_M, Layout, RoadModel, finish_distance, layouts_for, logic_dummies
from mapcreator.core.scene import Scene
from mapcreator.job import Job
from mapcreator.pipeline import mesh_stage, profile_stage, route_stage, scenery_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.providers.osm import roads as osm_roads
from mapcreator.regions import job_region

STAGE = "05_track"
# Material values for the base surfaces follow a generator whose tracks pass an in-game load test.
MATERIAL_PROPS = {
    "road": {"ksAmbient": 0.35, "ksDiffuse": 0.42, "ksSpecular": 0.06, "ksSpecularEXP": 28.0},
    # Loose stone and fines: about as matte as the grass, a little less than the asphalt's sheen. road_plain and
    # gravel_narrow take their base material's values (the name before the underscore).
    "gravel": {"ksAmbient": 0.36, "ksDiffuse": 0.44, "ksSpecular": 0.02, "ksSpecularEXP": 15.0},
    "grass": {"ksAmbient": 0.36, "ksDiffuse": 0.44, "ksSpecular": 0.005, "ksSpecularEXP": 18.0},
    "terrain": {"ksAmbient": 0.36, "ksDiffuse": 0.44, "ksSpecular": 0.005, "ksSpecularEXP": 18.0},
    "walls": {"ksAmbient": 0.4, "ksDiffuse": 0.6, "ksSpecular": 0.05, "ksSpecularEXP": 10.0},
    "roof": {"ksAmbient": 0.38, "ksDiffuse": 0.55, "ksSpecular": 0.2, "ksSpecularEXP": 20.0},
    "rail": {"ksAmbient": 0.4, "ksDiffuse": 0.55, "ksSpecular": 0.35, "ksSpecularEXP": 30.0},
    "water": {"ksAmbient": 0.25, "ksDiffuse": 0.35, "ksSpecular": 0.8, "ksSpecularEXP": 90.0},
    "power": {"ksAmbient": 0.4, "ksDiffuse": 0.55, "ksSpecular": 0.25, "ksSpecularEXP": 25.0},
}
ALPHA_REF = 0.5


PREVIEW_REACH_M = 1500.0
PREVIEW_CELL_PX = 2  # raster cells per preview pixel


def _write_text(path, text: str, encoding: str = "ascii") -> None:
    """Write a shipped text file exactly as given. Its CRLFs stay CRLFs: without newline="", Windows would
    turn each into CR CR LF, and the files would differ from the ones the Linux worker writes."""
    path.write_text(text, encoding=encoding, newline="")


def _preview_raster(dem, frame, pframe, road_xy) -> tuple[np.ndarray, np.ndarray]:
    """Hillshade and fade mask over the preview window (north at the top)."""
    from scipy.spatial import cKDTree

    x0, x1, z0, z1 = pframe
    w, h = package.PREVIEW_SIZE[0] * PREVIEW_CELL_PX, package.PREVIEW_SIZE[1] * PREVIEW_CELL_PX
    xs = np.linspace(x0, x1, w)
    zs = np.linspace(z0, z1, h)
    gx, gz = np.meshgrid(xs, zs)
    ge, gn, _ = frame.from_ac(np.stack([gx, np.zeros_like(gx), gz], axis=-1))
    heights = dem.sample(ge, gn)
    cell = float(xs[1] - xs[0])
    d, _ = cKDTree(road_xy[::5]).query(np.column_stack([ge.ravel(), gn.ravel()]), workers=-1)
    fade = np.clip((PREVIEW_REACH_M - 100 - d.reshape(gx.shape)) / 400.0, 0, 1)
    fade[~np.isfinite(heights)] = 0
    return package.hillshade(heights, cell), fade


def _materials_and_textures(work: WorkDir, scenes: list[Scene]):
    """KN5 materials and embedded DDS textures for every material used by the scenes.

    A scene texture is either a PNG the scenery stage wrote, or a name from the
    procedural library. Alpha-tested materials get DXT5 and ksPerPixelAT;
    everything else gets DXT1 and ksPerPixel. DDS files are cached between builds.
    """
    cache = work.stage(STAGE + "_cache")
    tex: dict[str, bytes] = {}
    mats: list[kn5.Kn5Material] = []
    index: dict[str, int] = {}
    for scene in scenes:
        for name, mat in scene.materials.items():
            if name in index:
                continue
            alpha = mat.kind == "alpha_test"
            fmt = "DXT5" if alpha else "DXT1"
            tex_name = f"{mat.texture}.dds"
            if tex_name not in tex:
                cached = cache / tex_name
                src = scene.textures.get(mat.texture)
                if src is not None:
                    from PIL import Image

                    with Image.open(src) as im:
                        tex[tex_name] = textures.to_dds(im, fmt)
                else:
                    if not cached.exists():
                        cached.write_bytes(textures.to_dds(textures.LIBRARY[mat.texture](), fmt))
                    tex[tex_name] = cached.read_bytes()
            props = dict(MATERIAL_PROPS.get(name, MATERIAL_PROPS.get(name.split("_")[0], MATERIAL_PROPS["walls"])))
            props |= {"ksEmissive": (0.0, 0.0, 0.0), "ksAlphaRef": ALPHA_REF if alpha else 0.0}
            index[name] = len(mats)
            mats.append(kn5.Kn5Material(name=name, shader="ksPerPixelAT" if alpha else "ksPerPixel",
                                        properties=props, textures={"txDiffuse": tex_name}, alpha_tested=alpha))
    return tex, mats, index


def _attributions(region, work: WorkDir):
    from mapcreator.providers.base import Attribution

    out = list(region.attributions)
    for stage in (profile_stage.STAGE, scenery_stage.STAGE, mesh_stage.STAGE):
        path = work.root / stage / "attributions.json"
        if path.exists():
            for d in json.loads(path.read_text(encoding="utf-8")):
                a = Attribution(**d)
                if a not in out:
                    out.append(a)
    return out


def _title(runs: list[dict], track_id: str) -> str:
    names = Counter()
    for r in runs:
        if r["tags"].get("name"):
            names[r["tags"]["name"]] += r["to_m"] - r["from_m"]
    return names.most_common(1)[0][0] if names else track_id.replace("_", " ").title()


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    region = job_region(job.waypoints)
    scene = Scene.load(work.root / mesh_stage.STAGE)
    c = np.load(work.root / profile_stage.STAGE / "centreline.npz")
    sec = np.load(work.root / section_stage.STAGE / "section.npz")
    runs = work.read_json(route_stage.STAGE, "runs.json")
    title = _title(runs, job.track_id)
    lonlat = region.to_lonlat()

    out_root = work.stage(STAGE)
    track_dir = out_root / job.track_id
    if track_dir.exists():
        shutil.rmtree(track_dir)
    track_dir.mkdir(parents=True)

    scenes = [scene]
    if (work.root / scenery_stage.STAGE / "scene.json").exists():
        scenes.append(Scene.load(work.root / scenery_stage.STAGE))
    tex, mats, mat_index = _materials_and_textures(work, scenes)
    # Tangents once, shared by every layout's KN5 (the meshes are the same; only the dummies differ).
    meshes = [kn5.Kn5Mesh(m.name, mat_index[m.material], m.positions, m.normals, m.uvs, m.triangles,
                          tangents=kn5.tangents(m.positions.astype(float), m.normals.astype(float),
                                                m.uvs.astype(float), np.asarray(m.triangles, dtype=np.int64)),
                          cast_shadows=m.renderable, renderable=m.renderable)
              for s in scenes for m in s.meshes]
    attributions = _attributions(region, work)

    # Preview background: a hillshade of the DEM around the road, fading out with distance.
    dem = region.elevation(settings)
    dem.load_corridor(c["xy"], PREVIEW_REACH_M)
    xz_all = scene.frame.to_ac(c["xy"][:, 0], c["xy"][:, 1], c["z"])[:, [0, 2]]
    pframe = package.preview_frame(xz_all)
    shade, alpha = _preview_raster(dem, scene.frame, pframe, c["xy"])

    ext = track_dir / "extension"
    ext.mkdir()
    tree_list = _write_tree_fx(work, ext)

    width = sec["half_left"] + sec["half_right"]
    unsealed = _unsealed(sec)
    has_gravel = any(m.name.startswith("1GRAVEL") for m in scene.meshes)
    unsealed_m = _unsealed_m(c["s"], sec)
    surface_tags = package.surface_tags(unsealed_m, float(c["s"][-1] - c["s"][0]))
    unsealed_note = (f" {unsealed_m / 1000:.1f} km of {(c['s'][-1] - c['s'][0]) / 1000:.1f} km unsealed."
                     if unsealed_m > 0 else "")
    layouts = layouts_for(job.layouts.value)
    multi = len(layouts) > 1
    summary = {"title": title, "unsealed_m": round(unsealed_m, 1), "layouts": {}}
    for lay in layouts:
        road = RoadModel.from_arrays(c["s"], c["xy"], c["z"], c["heading"], c["curvature"], c["grade"],
                                     sec["half_left"], sec["half_right"], sec["slope_left"], sec["slope_right"],
                                     lay.reverse)
        dummies = logic_dummies(road, scene.frame)
        pits = sum(1 for d in dummies if d.name.startswith("AC_PIT_"))
        kdummies = [kn5.Kn5Dummy(d.name, kn5.dummy_matrix(d.position, d.forward, d.up)) for d in dummies]
        kn5_name = f"{job.track_id}_{lay.key}.kn5" if multi else f"{job.track_id}.kn5"
        kn5.write(track_dir / kn5_name, root=job.track_id, textures=tex, materials=mats, meshes=meshes,
                  dummies=kdummies)

        layout_dir = track_dir / lay.key if multi else track_dir
        ui_dir = track_dir / "ui" / lay.key if multi else track_dir / "ui"
        for d in (layout_dir / "data", layout_dir / "ai", ui_dir):
            d.mkdir(parents=True, exist_ok=True)
        if multi:
            _write_text(track_dir / f"models_{lay.key}.ini", package.models_ini([kn5_name]), encoding="ascii")

        line = ai_line.build(road, scene.frame, end_short_m=AI_END_SHORT_M)
        ai_line.write(layout_dir / "ai" / "fast_lane.ai", line)
        tight = ""
        bend_d, bend_r = ai_line.road_radius(road, end_short_m=AI_END_SHORT_M)  # the road's bends, as validated
        if job.allow_tight_bends and bend_r.min() < validate.MIN_RADIUS_M:
            i = int(np.argmin(bend_r))
            tight = (f" Built as asked with a {bend_r[i]:.1f} m bend about {bend_d[i] / 1000:.1f} km in: "
                     "some cars can't turn that tightly, and AI cars may stop there.")

        xz = scene.frame.to_ac(road.e, road.n, road.z)[:, [0, 2]]
        map_png, map_ini = package.map_files(xz)
        map_png.save(layout_dir / "map.png")
        data = layout_dir / "data"
        _write_text(data / "map.ini", map_ini, encoding="ascii")
        _write_text(data / "surfaces.ini", package.surfaces_ini(gravel=has_gravel), encoding="ascii")
        _write_text(data / "lighting.ini", package.lighting_ini(), encoding="ascii")
        _write_text(data / "groove.ini", package.groove_ini(), encoding="ascii")
        _write_text(data / "crew.ini", package.crew_ini(), encoding="ascii")
        # The AI at AI_GRAVEL_PACE over unsealed road up to the finish, slower into bends it would take faster than
        # the road allows (a crest, a steep descent, no banking: #63), then ramped down over the run-off after the
        # finish, to all but stopped where its line ends. The hints are fractions of the racing line's own length,
        # which is shorter than the road's. A sealed track with no such bend gets the run-off ramp alone, as before.
        # The bends are found from the line as written, as the validator reads it back. Where the line apexes early
        # and the road is too narrow for the room the AI's slide out of the bend needs, the bend gets a hint too.
        bends = ai_line.bend_hints(ai_line.read(layout_dir / "ai" / "fast_lane.ai"), ai_line.road_banking(road, line))
        bends += ai_line.exit_hints(line)
        hints = ai_line.pace_hints(road.length, finish_distance(road.length), line=line, bends=bends,
                                   unsealed=unsealed_runs(c["s"], unsealed, reverse=lay.reverse))
        _write_text(data / "ai_hints.ini", package.ai_hints_ini(hints), encoding="ascii")

        direction = package.compass(float(np.arctan2(road.n[-1] - road.n[0], road.e[-1] - road.e[0])))
        name = f"{title} ({direction})" if multi else title
        mid = len(road.e) // 2
        lon, lat = lonlat.transform(road.e[mid], road.n[mid])
        country, city = region.country, region.state
        description = (
            f"{title}, {direction}bound, generated from open data by mapcreator.{unsealed_note}{tight} "
            + " ".join(a.line().rstrip(".") + "." for a in attributions)
            + " Requires Custom Shaders Patch (CSP)."
            + " Not affiliated with or endorsed by Kunos Simulazioni."
        )
        _write_text(ui_dir / "ui_track.json", package.ui_track_json(
            name=name, description=description,
            tags=["point to point", "real road", country.lower(), region.name.lower()] + surface_tags,
            lat=float(lat), lon=float(lon), country=country, city=city, length_m=road.length,
            width_m=(float(width.min()), float(width.max())), run="point-to-point", version=__version__,
            year=dt.date.today().year, priority=(1 if multi and not lay.reverse else None), pitboxes=pits),
            encoding="ascii")
        package.outline_png(xz).save(ui_dir / "outline.png")
        package.preview_png(shade, alpha, pframe, xz).save(ui_dir / "preview.png")

        summary["layouts"][lay.key or "main"] = {
            "name": name, "kn5": kn5_name, "length_m": round(road.length, 1), "ai_points": len(line.pos),
            "grid": pits,
            "top_speed_kmh": round(float(line.extra[:, 0].max() * 3.6), 1),
            "slowest_kmh": round(float(line.extra[:, 0].min() * 3.6), 1),
            "ai_tightest_radius_m": round(float(line.extra[:, 4].min()), 1),
            "road_tightest_radius_m": round(float(bend_r.min()), 1),
            # Each risky bend's hint: metres along the AI line where its full pace holds, and the pace.
            "bend_hints": [[round(b.start), round(b.end), b.pace] for b in bends],
            # Each early apex the line keeps room from the outside edge for: metres along the AI line, and whether
            # the road was too narrow for it (then it's hinted above).
            "exit_clearance": [[round(a), round(b), short] for a, b, _, short in line.exits or []],
        }

    power_geo = work.root / scenery_stage.STAGE / "power.geojson"
    if power_geo.exists():
        shutil.copyfile(power_geo, track_dir / "power.geojson")
    osm_note = osm_roads.selection_note(route_stage.osm_selection(work, settings), build=buildinfo.version(),
                                        source_url=settings.source_url)
    _write_attribution(track_dir, attributions, title, power_geojson=power_geo.exists(), osm_note=osm_note)
    water_file = work.root / mesh_stage.STAGE / "water.json"
    water_type = json.loads(water_file.read_text(encoding="utf-8")).get("type") if water_file.exists() else None
    summary["tree_fx_trees"] = tree_list[1] if tree_list else 0
    # The terrain's materials: `terrain`, or with ground classes those of the classes present (`core.grass`).
    ground = [c.material for c in grass.CLASSES if c.material in mat_index]
    pack = any(m in mat_index for m in grass.PACK_MATERIALS)
    if pack:
        summary["grass_fx_atlas_mb"] = _write_grass_atlas(work, ext)
    _write_text(ext / "ext_config.ini", package.ext_config_ini(
        grass_materials=[n for n in ["grass", *ground] if n in mat_index and grass.grows(n)],
        road_materials=[n for n in ("road", "road_plain") if n in mat_index], version=__version__,
        gravel_materials=[n for n in ("gravel", "gravel_narrow") if n in mat_index],
        grassy_gravel=[n for n in ("gravel_narrow",) if n in mat_index],
        bare_ground=[n for n in ground if not grass.grows(n)],
        grass_fx=grass.grass_fx(mat_index) if pack else None,
        dirt_colours=textures.tyre_dirt_colours() if has_gravel else None,
        water_materials=["water"] if "water" in mat_index else [], water_type=water_type or "LAKE",
        tree_list=tree_list[0] if tree_list else None,
        tree_surfaces=[*ground, "grass"] if "grass" in mat_index else ground,
        retro_materials={m.name: (furniture.NIGHT_DIFFUSE[m.name], m.properties["ksDiffuse"])
                         for m in mats if m.name in furniture.NIGHT_DIFFUSE}),
        encoding="ascii")
    _write_centreline(track_dir, job, c, sec, region, lonlat, attributions)

    zip_path = out_root / f"{job.track_id}.zip"
    # Level 3: the same size as the default 6 on a KN5 (83%), in a quarter of the time (1.9 s against 7.2 s
    # for a 90 MB KN5).
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in sorted(track_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out_root).as_posix())
    summary["zip_mb"] = round(zip_path.stat().st_size / 1e6, 1)
    summary["kn5_mb"] = round(sum(p.stat().st_size for p in track_dir.glob("*.kn5")) / 1e6, 1)
    return summary


def _unsealed(sec) -> np.ndarray:
    """Whether each station is unsealed; all sealed for a section stage from before surfaces."""
    return sec["unsealed"].astype(bool) if "unsealed" in sec else np.zeros(len(sec["s"]), dtype=bool)


def _unsealed_m(s: np.ndarray, sec) -> float:
    return float(np.diff(s)[_unsealed(sec)[:-1]].sum())


def _surface_stretches(unsealed: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) station of each stretch of one surface class. A change at station i starts a stretch there,
    as it starts a new chunk of road mesh (`mesh.road_chunks`), so the 1GRAVEL_### meshes cover exactly the
    unsealed stretches."""
    cuts = [0, *np.flatnonzero(np.diff(unsealed.astype(np.int8))) + 1, len(unsealed) - 1]
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def surface_runs(s: np.ndarray, unsealed: np.ndarray) -> list[dict]:
    """Stretches of one surface class along the road, for centreline.geojson."""
    return [{"from_m": round(float(s[a]), 1), "to_m": round(float(s[b]), 1),
             "surface_class": "unsealed" if unsealed[a] else "sealed"} for a, b in _surface_stretches(unsealed)]


def unsealed_runs(s: np.ndarray, unsealed: np.ndarray, *, reverse: bool = False) -> list[tuple[float, float]]:
    """The unsealed stretches as (from, to) metres from a layout's start, in driving order, for its pace hints.
    The reverse layout runs from the centreline's far end, as its RoadModel does."""
    runs = [(float(s[a] - s[0]), float(s[b] - s[0])) for a, b in _surface_stretches(unsealed) if unsealed[a]]
    if not reverse:
        return runs
    total = float(s[-1] - s[0])
    return [(total - b, total - a) for a, b in reversed(runs)]


def _write_grass_atlas(work: WorkDir, ext) -> float:
    """Write the Grass FX atlas (`core.grassfx`) next to ext_config.ini; it's the same for every track, so it's
    drawn once per work folder. Returns its size in MB."""
    cached = work.stage(STAGE + "_cache") / "grass_fx_atlas.dds"
    if not cached.exists():
        cached.write_bytes(grassfx.atlas_dds())
    out = ext / grassfx.FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, out)
    return round(out.stat().st_size / 1e6, 2)


def _write_tree_fx(work: WorkDir, ext) -> tuple[str, int] | None:
    """Copy the 3D tree models into extension/trees and write their list. Returns (list path, trees)."""
    fx = work.root / scenery_stage.STAGE / "treefx"
    placement = fx / "placement.npz"
    if not placement.exists():
        return None
    p = np.load(placement)
    if not len(p["model"]):
        return None
    names = [str(n) for n in p["names"]]
    out = ext / "trees"
    out.mkdir()
    for name in names:
        shutil.copyfile(fx / f"{name}.bin", out / f"{name}.bin")
    _write_text(out / "trees.txt", treefx.trees_txt(names, p["model"], p["pos"], p["angle"], p["size"], p["width"]))
    return "trees/trees.txt", int(len(p["model"]))


def _write_attribution(track_dir, attributions, title: str, *, power_geojson: bool = False,
                       osm_note: str = "") -> None:
    """`osm_note` says how the OpenStreetMap data was selected (`osm_roads.selection_note`, ODbL 4.6)."""
    lines = [
        f"{title}: Assetto Corsa track generated by mapcreator.",
        "",
        "Data sources:",
        *[f"- {a.line()}" for a in attributions],
        "",
        # CC BY 4.0 (3(a)(1)(B)) and CC BY 3.0 AU (4B) ask for changes to the data to be marked.
        "Changes: mapcreator modified all of the data above to build this track. It was resampled, filtered",
        "and smoothed, and turned into 3D geometry.",
        "",
        "centreline.geojson holds the road centreline, heights and widths used to build this track.",
        "It is a derivative of OpenStreetMap data and is available under the Open Database License (ODbL) 1.0",
        "(https://opendatacommons.org/licenses/odbl/1-0/).",
        *(["power.geojson holds the OpenStreetMap power lines used to place this track's towers, poles and wires,",
            "also under the ODbL 1.0."] if power_geojson else []),
        *([osm_note] if osm_note else []),
        "",
        "Textures, 3D tree models and building materials are procedurally generated by mapcreator.",
        "This track requires Custom Shaders Patch (CSP), which draws its trees and water. It isn't built to work without it.",
        "",
        "This track is not affiliated with or endorsed by Kunos Simulazioni.",
        "Assetto Corsa is a trademark of its owner.",
        "",
    ]
    _write_text(track_dir / "ATTRIBUTION.txt", "\r\n".join(lines), encoding="utf-8")


def _write_centreline(track_dir, job: Job, c, sec, region, lonlat, attributions) -> None:
    step = 5
    idx = np.arange(0, len(c["s"]), step)
    lon, lat = lonlat.transform(c["xy"][idx, 0], c["xy"][idx, 1])
    width = (sec["half_left"] + sec["half_right"])[idx]
    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString",
                     "coordinates": [[round(a, 7), round(b, 7), round(float(h), 2)]
                                     for a, b, h in zip(lon, lat, c["z"][idx])]},
        "properties": {
            "track_id": job.track_id,
            "station_m": [round(float(v), 1) for v in c["s"][idx]],
            "width_m": [round(float(v), 2) for v in width],
            "superelevation": [round(float(v), 4) for v in sec["superelevation"][idx]],
            # Sealed or unsealed, from OSM's `surface` with untagged and short runs settled (core/section.py).
            "surface_runs": surface_runs(c["s"], _unsealed(sec)),
            "height_datum": region.height_datum,
            "license": "ODbL-1.0",
            "attribution": "; ".join(a.line() for a in attributions),
        },
    }
    _write_text(track_dir / "centreline.geojson",
                json.dumps({"type": "FeatureCollection", "features": [feature]}), encoding="utf-8")
