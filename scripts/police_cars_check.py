"""Visual check of the police cars (`mapcreator.core.police_car`): renders and a test track for a look in game.

Writes, to `--out` (by default a folder in the system's temp directory, never inside the repository):
- `lineup_right.png` and `lineup_left.png`: all eight cars side by side, seen from each side and a little above,
  drawn by a small software renderer here (textured, lit, back faces culled as AC culls them, so a face wound the
  wrong way shows as a hole);
- `closeup_<code>.png`: each car from the front three-quarter;
- `atlas_<code>.png`: each jurisdiction's texture as the car draws it;
- `mapcreator_police_check/`: an Assetto Corsa track, a straight 600 m road with the eight cars parked on each verge,
  facing the driving direction: those on the left show their right side as you pass, those on the right their left.
  Copy the folder into `assettocorsa/content/tracks/`, pick "Police car check" in Content Manager, and drive. Set
  the time to night (CSP) to see the livery bands and light bars catch the headlights.

Run from the repository root: `PYTHONPATH=src python scripts/police_cars_check.py [--out DIR]`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from mapcreator.core import ai_line, kn5, package, police_car, textures
from mapcreator.core.layout import RoadModel, logic_dummies
from mapcreator.core.scene import LocalFrame, Mesh
from mapcreator.pipeline import track_stage

TRACK = "mapcreator_police_check"
ROAD_M, HALF_W = 600.0, 3.2
FIRST_CAR_M, SPACING_M, VERGE_M = 120.0, 14.0, 4.6  # where the first car stands, the gap, its centre past the edge


# -- a small software renderer -----------------------------------------------------------------------------------


def render(meshes: list[Mesh], images: dict[str, np.ndarray], eye, target, size=(1600, 700), fov=40.0,
           sky=(196, 210, 226)) -> Image.Image:
    """Perspective render of AC-axes meshes: textured (nearest texel), Lambert lit from the upper left, z-buffered,
    and with back faces culled the way AC culls them (a face shows when cross(b - a, c - a) points at the eye)."""
    w, h = size
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = 0.5 * h / np.tan(np.radians(fov) / 2)
    light = np.array([-0.4, 0.8, 0.45])
    light /= np.linalg.norm(light)
    img = np.empty((h, w, 3), float)
    img[:] = sky
    zbuf = np.full((h, w), np.inf)
    for m in meshes:
        if not m.renderable:
            continue
        tex = images[m.material]
        th, tw = tex.shape[:2]
        rel = m.positions.astype(float) - eye
        cam = np.column_stack([rel @ right, rel @ up, rel @ fwd])
        for tri in m.triangles:
            p = m.positions[tri].astype(float)
            if np.cross(p[1] - p[0], p[2] - p[0]) @ (eye - p[0]) <= 0:
                continue
            c = cam[tri]
            if (c[:, 2] < 0.1).any():
                continue
            sx = w / 2 + f * c[:, 0] / c[:, 2]
            sy = h / 2 - f * c[:, 1] / c[:, 2]
            x0, x1 = max(int(np.floor(sx.min())), 0), min(int(np.ceil(sx.max())), w - 1)
            y0, y1 = max(int(np.floor(sy.min())), 0), min(int(np.ceil(sy.max())), h - 1)
            if x0 > x1 or y0 > y1:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            d = (sy[1] - sy[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy[0] - sy[2])
            if abs(d) < 1e-9:
                continue
            l0 = ((sy[1] - sy[2]) * (gx - sx[2]) + (sx[2] - sx[1]) * (gy - sy[2])) / d
            l1 = ((sy[2] - sy[0]) * (gx - sx[2]) + (sx[0] - sx[2]) * (gy - sy[2])) / d
            l2 = 1 - l0 - l1
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not inside.any():
                continue
            iz = l0 / c[0, 2] + l1 / c[1, 2] + l2 / c[2, 2]
            depth = 1 / iz
            yy, xx = gy.astype(int)[inside], gx.astype(int)[inside]
            nearer = depth[inside] < zbuf[yy, xx]
            if not nearer.any():
                continue
            yy, xx = yy[nearer], xx[nearer]
            wts = [(l / c[k, 2])[inside][nearer] / iz[inside][nearer] for k, l in enumerate((l0, l1, l2))]
            uv = sum(wt[:, None] * m.uvs[tri[k]].astype(float) for k, wt in enumerate(wts))
            nrm = sum(wt[:, None] * m.normals[tri[k]].astype(float) for k, wt in enumerate(wts))
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            u = np.clip((uv[:, 0] % 1.0) * tw, 0, tw - 1).astype(int)
            v = np.clip((uv[:, 1] % 1.0) * th, 0, th - 1).astype(int)
            shade = 0.45 + 0.6 * np.clip(nrm @ light, 0, 1)
            img[yy, xx] = tex[v, u] * shade[:, None]
            zbuf[yy, xx] = depth[inside][nearer]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def ground(x0, x1, z0, z1, y=0.0, material="grass", tile=8.0) -> Mesh:
    pos = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], np.float32)
    uv = (pos[:, [0, 2]] / tile).astype(np.float32)
    tri = np.array([[0, 2, 1], [0, 3, 2]])
    p = pos.astype(float)
    if np.cross(p[tri[0, 1]] - p[0], p[tri[0, 2]] - p[0])[1] < 0:
        tri = tri[:, [0, 2, 1]]
    return Mesh("1GRASS_000", material, pos, np.tile([[0, 1, 0]], (4, 1)).astype(np.float32), uv, tri)


def lineup() -> tuple[list[list[Mesh]], list[Mesh], dict[str, np.ndarray], dict[str, Image.Image]]:
    """The eight cars in a row along x, 3 m apart, all facing -z (each car's meshes), and a grey ground."""
    cars, images, atlases = [], {}, {}
    for k, code in enumerate(police_car.JURISDICTIONS):
        car = police_car.police_car(code)
        cars.append(police_car.place(car, ((k - 3.5) * 3.0, 0.0, 0.0), (0.0, 0.0, -1.0), index=k))
        (im,) = car.textures.values()
        atlases[code] = im
        for mat in car.materials.values():
            images[mat.name] = np.asarray(im, float)
    # In 2 m tiles: the renderer drops triangles that reach behind the eye.
    floor = [ground(2 * i, 2 * i + 2, 2 * j, 2 * j + 2, -0.002, material="ground")
             for i in range(-14, 14) for j in range(-14, 14)]
    images["ground"] = np.full((4, 4, 3), 132.0)
    return cars, floor, images, atlases


# -- the test track ----------------------------------------------------------------------------------------------


def road_meshes(frame: LocalFrame) -> list[Mesh]:
    """A flat straight road east from the origin, with grass beside it to 60 m."""
    d = np.arange(0.0, ROAD_M + 1.0, 12.0)
    pos, uv = [], []
    for k, x in enumerate(d):
        for side, u in ((HALF_W, 0.0), (-HALF_W, 1.0)):  # +north (left of travel) first
            pos.append(frame.to_ac(x, side, 0.0))
            uv.append((u, k * 1.0))
    pos = np.array(pos, np.float32)
    tri = []
    for k in range(len(d) - 1):
        a, b, c, e = 2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2
        tri += [(a, b, c), (a, c, e)]
    tri = np.array(tri)
    p = pos.astype(float)
    flip = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])[:, 1] < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    road = Mesh("1ROAD_000", "road", pos, np.tile([[0, 1, 0]], (len(pos), 1)).astype(np.float32),
                np.array(uv, np.float32), tri)
    grass = ground(-60.0, ROAD_M + 60.0, -60.0, 60.0, -0.03)
    return [road, grass]


def track(out: Path) -> Path:
    frame = LocalFrame(0.0, 0.0, 0.0)
    folder = out / TRACK
    if folder.exists():
        shutil.rmtree(folder)
    for sub in ("data", "ai", "ui", "extension"):
        (folder / sub).mkdir(parents=True)
    s = np.arange(0.0, ROAD_M + 0.5, 1.0)
    n = len(s)
    zeros = np.zeros(n)
    road = RoadModel.from_arrays(s, np.column_stack([s, zeros]), zeros, zeros, zeros, zeros, np.full(n, HALF_W),
                                 np.full(n, HALF_W), np.full(n, 0.025), np.full(n, 0.025), False)
    placed = []
    textures_png: dict[str, Image.Image] = {}
    retro: dict[str, tuple[float, float]] = {}
    car_props: dict[str, dict] = {}
    for k, code in enumerate(police_car.JURISDICTIONS):
        car = police_car.police_car(code)
        textures_png.update(car.textures)
        car_props.update(car.kn5_properties)
        for name, night in car.night_diffuse.items():
            retro[name] = (night, car.kn5_properties[name]["ksDiffuse"])
        x = FIRST_CAR_M + k * SPACING_M
        forward = frame.to_ac(1.0, 0.0, 0.0) - frame.to_ac(0.0, 0.0, 0.0)  # east, the driving direction
        for side, north in ((0, HALF_W + VERGE_M), (1, -(HALF_W + VERGE_M))):
            placed += police_car.place(car, frame.to_ac(x, north, 0.0), forward, index=2 * k + side)
    meshes = road_meshes(frame) + police_car.merge(placed)

    tex = {"road_marked.dds": textures.to_dds(textures.LIBRARY["road_marked"]()),
           "grass.dds": textures.to_dds(textures.LIBRARY["grass"]())}
    # The road and grass take the values a built track gives them, so the night exposure is a built track's.
    extra = {"ksEmissive": (0.0, 0.0, 0.0), "ksAlphaRef": 0.0}
    mats = [kn5.Kn5Material(name, properties=track_stage.MATERIAL_PROPS[name] | extra,
                            textures={"txDiffuse": f"{texture}.dds"})
            for name, texture in (("road", "road_marked"), ("grass", "grass"))]
    for name, im in textures_png.items():
        tex[f"{name}.dds"] = textures.to_dds(im)
    for name in sorted(car_props):
        texture = next(t for t in textures_png if name == t or name.startswith(t + "_"))
        mats.append(kn5.Kn5Material(name, properties=car_props[name] | extra, textures={"txDiffuse": f"{texture}.dds"}))
    index = {m.name: i for i, m in enumerate(mats)}
    kmeshes = [kn5.Kn5Mesh(m.name, index[m.material], m.positions, m.normals, m.uvs, m.triangles,
                           cast_shadows=m.renderable, renderable=m.renderable) for m in meshes]
    dummies = [kn5.Kn5Dummy(d.name, kn5.dummy_matrix(d.position, d.forward, d.up)) for d in logic_dummies(road, frame)]
    kn5.write(folder / f"{TRACK}.kn5", root=TRACK, textures=tex, materials=mats, meshes=kmeshes, dummies=dummies)

    ai_line.write(folder / "ai" / "fast_lane.ai", ai_line.build(road, frame, racing=False))
    xz = frame.to_ac(road.e, road.n, road.z)[:, [0, 2]]
    map_png, map_ini = package.map_files(xz)
    map_png.save(folder / "map.png")
    for name, text in (("map.ini", map_ini), ("surfaces.ini", package.surfaces_ini()),
                       ("lighting.ini", package.lighting_ini()), ("groove.ini", package.groove_ini()),
                       ("crew.ini", package.crew_ini())):
        (folder / "data" / name).write_text(text, encoding="ascii", newline="")
    (folder / "ui" / "ui_track.json").write_text(package.ui_track_json(
        name="Police car check", description="Every jurisdiction's police car (mapcreator test track).",
        tags=["point to point", "test"], lat=-42.88, lon=147.33, country="Australia", city="", length_m=ROAD_M,
        width_m=(2 * HALF_W, 2 * HALF_W), run="point-to-point", version="check", year=dt.date.today().year,
        pitboxes=sum(1 for d in dummies if d.name.startswith("AC_PIT_"))), encoding="ascii", newline="")
    package.outline_png(xz).save(folder / "ui" / "outline.png")
    (folder / "extension" / "ext_config.ini").write_text(package.ext_config_ini(
        grass_materials=["grass"], road_materials=["road"], version="check", retro_materials=retro),
        encoding="ascii", newline="")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "mapcreator-police-check")
    args = parser.parse_args()
    out = args.out.resolve()
    repo = Path(__file__).resolve().parents[1]
    if out == repo or repo in out.parents:
        parser.error("--out must be outside the repository")
    out.mkdir(parents=True, exist_ok=True)

    cars, floor, images, atlases = lineup()
    for code, im in atlases.items():
        im.save(out / f"atlas_{code}.png")
    everything = [m for car in cars for m in car] + floor
    # The cars face -z, in a row along x: the front three-quarter from their right, the rear one from their left.
    render(everything, images, eye=(14.0, 5.0, -26.0), target=(0.0, 0.7, 0.0), size=(2000, 760), fov=22
           ).save(out / "lineup_right.png")
    render(everything, images, eye=(-14.0, 5.0, 26.0), target=(0.0, 0.7, 0.0), size=(2000, 760), fov=22
           ).save(out / "lineup_left.png")
    for k, code in enumerate(police_car.JURISDICTIONS):
        x = (k - 3.5) * 3.0
        render(cars[k] + floor, images, eye=(x + 5.5, 1.7, -5.0), target=(x, 0.7, 0.2), size=(900, 600), fov=40
               ).save(out / f"closeup_{code}.png")
        render(cars[k] + floor, images, eye=(x - 5.5, 1.7, 5.0), target=(x, 0.7, -0.2), size=(900, 600), fov=40
               ).save(out / f"closeup_{code}_rear.png")
    folder = track(out)
    car = police_car.police_car("tas")
    print(f"Wrote renders, atlases and the test track to {out}")
    print(f"Track: copy {folder} into assettocorsa/content/tracks/ and pick 'Police car check'.")
    print(f"One car: {car.vertices} vertices, {car.triangles} triangles.")


if __name__ == "__main__":
    main()
