"""Stage 5: assemble the installable AC track folder (KN5s, AI lines, data and UI files) and zip it."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import zipfile
from collections import Counter

import numpy as np

from mapcreator import __version__
from mapcreator.config import Settings
from mapcreator.core import ai_line, kn5, package, textures
from mapcreator.core.layout import Layout, RoadModel, layouts_for, logic_dummies
from mapcreator.core.scene import Scene
from mapcreator.job import Job
from mapcreator.pipeline import mesh_stage, profile_stage, route_stage, section_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.regions import region_for

STAGE = "05_track"
TEXTURE_FILES = {"road": "asphalt.dds", "grass": "grass.dds", "terrain": "ground.dds"}
TEXTURE_SOURCE = {"asphalt.dds": "asphalt", "grass.dds": "grass", "ground.dds": "ground"}
# Material values follow a generator whose tracks pass an in-game load test.
MATERIAL_PROPS = {
    "road": {"ksAmbient": 0.35, "ksDiffuse": 0.42, "ksSpecular": 0.06, "ksSpecularEXP": 28.0},
    "grass": {"ksAmbient": 0.36, "ksDiffuse": 0.44, "ksSpecular": 0.005, "ksSpecularEXP": 18.0},
    "terrain": {"ksAmbient": 0.36, "ksDiffuse": 0.44, "ksSpecular": 0.005, "ksSpecularEXP": 18.0},
}
COUNTRY = {"tas": ("Australia", "Tasmania")}


PREVIEW_REACH_M = 1500.0
PREVIEW_CELL_PX = 2  # raster cells per preview pixel


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


def _texture_bytes(work: WorkDir) -> dict[str, bytes]:
    cache = work.stage(STAGE + "_cache")
    out = {}
    for fname, key in TEXTURE_SOURCE.items():
        path = cache / fname
        if not path.exists():
            path.write_bytes(textures.to_dds_dxt1(textures.LIBRARY[key]()))
        out[fname] = path.read_bytes()
    return out


def _materials() -> tuple[list[kn5.Kn5Material], dict[str, int]]:
    mats, index = [], {}
    for name, tex in TEXTURE_FILES.items():
        props = dict(MATERIAL_PROPS[name], ksEmissive=(0.0, 0.0, 0.0), ksAlphaRef=0.0)
        index[name] = len(mats)
        mats.append(kn5.Kn5Material(name=name, properties=props, textures={"txDiffuse": tex}))
    return mats, index


def _title(runs: list[dict], track_id: str) -> str:
    names = Counter()
    for r in runs:
        if r["tags"].get("name"):
            names[r["tags"]["name"]] += r["to_m"] - r["from_m"]
    return names.most_common(1)[0][0] if names else track_id.replace("_", " ").title()


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    region = region_for(job.waypoints[0].lon, job.waypoints[0].lat)
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

    tex = _texture_bytes(work)
    mats, mat_index = _materials()
    meshes = [kn5.Kn5Mesh(m.name, mat_index[m.material], m.positions, m.normals, m.uvs, m.triangles)
              for m in scene.meshes]

    # Preview background: a hillshade of the DEM around the road, fading out with distance.
    dem = region.elevation(settings)
    dem.load_corridor(c["xy"], PREVIEW_REACH_M)
    xz_all = scene.frame.to_ac(c["xy"][:, 0], c["xy"][:, 1], c["z"])[:, [0, 2]]
    pframe = package.preview_frame(xz_all)
    shade, alpha = _preview_raster(dem, scene.frame, pframe, c["xy"])

    width = sec["half_left"] + sec["half_right"]
    layouts = layouts_for(job.layouts.value)
    multi = len(layouts) > 1
    summary = {"title": title, "layouts": {}}
    for lay in layouts:
        road = RoadModel.from_arrays(c["s"], c["xy"], c["z"], c["heading"], c["curvature"], c["grade"],
                                     sec["half_left"], sec["half_right"], sec["slope_left"], sec["slope_right"],
                                     lay.reverse)
        dummies = logic_dummies(road, scene.frame)
        kdummies = [kn5.Kn5Dummy(d.name, kn5.dummy_matrix(d.position, d.forward, d.up)) for d in dummies]
        kn5_name = f"{job.track_id}_{lay.key}.kn5" if multi else f"{job.track_id}.kn5"
        kn5.write(track_dir / kn5_name, root=job.track_id, textures=tex, materials=mats, meshes=meshes,
                  dummies=kdummies)

        layout_dir = track_dir / lay.key if multi else track_dir
        ui_dir = track_dir / "ui" / lay.key if multi else track_dir / "ui"
        for d in (layout_dir / "data", layout_dir / "ai", ui_dir):
            d.mkdir(parents=True, exist_ok=True)
        if multi:
            (track_dir / f"models_{lay.key}.ini").write_text(package.models_ini([kn5_name]), encoding="ascii")

        line = ai_line.build(road, scene.frame)
        ai_line.write(layout_dir / "ai" / "fast_lane.ai", line)

        xz = scene.frame.to_ac(road.e, road.n, road.z)[:, [0, 2]]
        map_png, map_ini = package.map_files(xz)
        map_png.save(layout_dir / "map.png")
        data = layout_dir / "data"
        (data / "map.ini").write_text(map_ini, encoding="ascii")
        (data / "surfaces.ini").write_text(package.surfaces_ini(), encoding="ascii")
        (data / "lighting.ini").write_text(package.lighting_ini(), encoding="ascii")
        (data / "groove.ini").write_text(package.groove_ini(), encoding="ascii")
        (data / "crew.ini").write_text(package.crew_ini(), encoding="ascii")

        direction = package.compass(float(np.arctan2(road.n[-1] - road.n[0], road.e[-1] - road.e[0])))
        name = f"{title} ({direction})" if multi else title
        mid = len(road.e) // 2
        lon, lat = lonlat.transform(road.e[mid], road.n[mid])
        country, city = COUNTRY.get(region.key, ("", region.name))
        description = (
            f"{title}, {direction}bound, generated from open data by mapcreator. "
            + " ".join(a.line().rstrip(".") + "." for a in region.attributions)
            + " Not affiliated with or endorsed by Kunos Simulazioni."
        )
        (ui_dir / "ui_track.json").write_text(package.ui_track_json(
            name=name, description=description,
            tags=["point to point", "real road", country.lower(), region.name.lower()],
            lat=float(lat), lon=float(lon), country=country, city=city, length_m=road.length,
            width_m=(float(width.min()), float(width.max())), run="point to point", version=__version__,
            year=dt.date.today().year), encoding="ascii")
        package.outline_png(xz).save(ui_dir / "outline.png")
        package.preview_png(shade, alpha, pframe, xz).save(ui_dir / "preview.png")

        summary["layouts"][lay.key or "main"] = {
            "name": name, "kn5": kn5_name, "length_m": round(road.length, 1), "ai_points": len(line.pos),
            "top_speed_kmh": round(float(line.extra[:, 0].max() * 3.6), 1),
            "slowest_kmh": round(float(line.extra[:, 0].min() * 3.6), 1),
        }

    _write_attribution(track_dir, region, title)
    _write_centreline(track_dir, job, c, sec, region, lonlat)

    zip_path = out_root / f"{job.track_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(track_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out_root).as_posix())
    summary["zip_mb"] = round(zip_path.stat().st_size / 1e6, 1)
    summary["kn5_mb"] = round(sum(p.stat().st_size for p in track_dir.glob("*.kn5")) / 1e6, 1)
    return summary


def _write_attribution(track_dir, region, title: str) -> None:
    lines = [
        f"{title}: Assetto Corsa track generated by mapcreator.",
        "",
        "Data sources:",
        *[f"- {a.line()}" for a in region.attributions],
        "",
        "centreline.geojson holds the road centreline, heights and widths used to build this track.",
        "It is a derivative of OpenStreetMap data and is available under the Open Database License (ODbL) 1.0.",
        "",
        "Textures are procedurally generated by mapcreator.",
        "",
        "This track is not affiliated with or endorsed by Kunos Simulazioni.",
        "Assetto Corsa is a trademark of its owner.",
        "",
    ]
    (track_dir / "ATTRIBUTION.txt").write_text("\r\n".join(lines), encoding="utf-8")


def _write_centreline(track_dir, job: Job, c, sec, region, lonlat) -> None:
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
            "height_datum": "AHD" if region.key == "tas" else "unknown",
            "license": "ODbL-1.0",
            "attribution": "; ".join(a.line() for a in region.attributions),
        },
    }
    (track_dir / "centreline.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [feature]}), encoding="utf-8")
