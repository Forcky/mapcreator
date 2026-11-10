"""Look at the closure props (`core/closures.py`) before they're placed on real tracks.

    python scripts/closure_preview.py OUT_DIR                   # renders and a test track
    python scripts/closure_preview.py OUT_DIR --no-track        # renders only
    python scripts/closure_preview.py OUT_DIR --solid bales signs   # other props solid than closures.SOLID

Keep OUT_DIR outside the repository. It gets:
- `renders/<case>.png`: each closure seen from the mainline, three panels: driving up to it, from the mouth, and
  from the mouth with its colliders drawn in red. A small software renderer draws them (back faces culled, as the
  game does), so they show the geometry, the textures and the winding, not the game's lighting.
- `mapcreator_closures/`: a small Assetto Corsa track, a straight 600 m road with a side-road stub every 65 m, each
  closed a different way (the cases below), to drive past by day and at night. On the PC with the game:
  `.venv\\Scripts\\python harness/ac_harness.py install OUT_DIR/mapcreator_closures`, then pick "Closure props
  (test)" in Content Manager. Night needs Custom Shaders Patch: set the time to about 22:00 and check the boards,
  signs and tape light up in the headlights. Drive into a closure to feel its colliders: by default only the bales
  are solid (`closures.SOLID`), and `--solid` builds the track with others. The grey block behind each major
  closure's boards only marks the vehicle slot: the vehicle comes from its own module, and isn't solid here.

Runs from a checkout without installing: it puts the checkout's own `src` first on the path.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapcreator.core import ai_line, closures, kn5, package, roadside, textures  # noqa: E402
from mapcreator.core.layout import AI_END_SHORT_M, RoadModel, finish_distance, logic_dummies  # noqa: E402
from mapcreator.core.scene import LocalFrame, Material, Mesh  # noqa: E402

TRACK_ID = "mapcreator_closures"
ROAD_M = 600.0
HALF = 3.2  # the mainline: a 6.4 m two-lane seal
STUB_W = 6.0  # each stub's carriageway
VERGE = 1.0  # closed beyond the carriageway on each side
FRAME = LocalFrame(0.0, 0.0, 0.0)
EYE = 1.1
# (name, level, obstacle, overshoot, stub length, side: +1 left of the road's direction, -1 right)
CASES = [
    ("major", "major", "boards", False, 100.0, 1),
    ("minor_boards", "minor", "boards", False, 100.0, -1),
    ("minor_bales", "minor", "bales", False, 100.0, 1),
    ("limited", "limited", "boards", False, 100.0, -1),
    ("major_overshoot", "major", "boards", True, 100.0, 1),
    ("minor_bales_30m", "minor", "bales", False, 30.0, -1),
    ("minor_boards_40m", "minor", "boards", False, 40.0, 1),
]
FIRST_STUB_M, STUB_EVERY_M = 100.0, 65.0
MATERIAL_PROPS = {"ksAmbient": 0.4, "ksDiffuse": 0.6, "ksSpecular": 0.05, "ksSpecularEXP": 10.0}
GROUND_PROPS = {"ksAmbient": 0.36, "ksDiffuse": 0.44, "ksSpecular": 0.005, "ksSpecularEXP": 18.0}
ROAD_PROPS = {"ksAmbient": 0.35, "ksDiffuse": 0.42, "ksSpecular": 0.06, "ksSpecularEXP": 28.0}


# -- the scene ----------------------------------------------------------------------------------------------------

def _grid(name, material, e0, e1, n0, n1, cell, uv_of, h=0.0):
    """A flat grid mesh in region coordinates (east e, north n), up = +y."""
    es = np.linspace(e0, e1, max(int(round((e1 - e0) / cell)), 1) + 1)
    ns = np.linspace(n0, n1, max(int(round((n1 - n0) / cell)), 1) + 1)
    ge, gn = np.meshgrid(es, ns)
    pos = FRAME.to_ac(ge.ravel(), gn.ravel(), np.full(ge.size, h))
    uv = uv_of(ge.ravel(), gn.ravel())
    rows, cols = ge.shape
    i = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
    tri = np.concatenate([np.column_stack([i, i + 1, i + cols + 1]), np.column_stack([i, i + cols + 1, i + cols])])
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    up = np.cross(p1 - p0, p2 - p0)[:, 1] >= 0
    tri = np.where(up[:, None], tri, tri[:, [0, 2, 1]])
    nrm = np.tile([0.0, 1.0, 0.0], (len(pos), 1))
    return Mesh(name, material, pos.astype(np.float32), nrm.astype(np.float32), uv.astype(np.float32), tri)


def build_scene(solid=None):
    """The mainline, the ground, each stub and its closure (placed and merged), and the slot markers. `solid` is
    `closures.closure`'s: None for `closures.SOLID`."""
    meshes = [
        _grid("1ROAD_000", "road", 0, ROAD_M, -HALF, HALF, 4.0,
              lambda e, n: np.column_stack([(HALF - n) / (2 * HALF), e / textures.ROAD_TILE_M])),
        _grid("1GRASS_T000", "terrain", -60, ROAD_M + 60, -140, 140, 10.0,
              lambda e, n: np.column_stack([e, n]) / 16.0, h=-0.03),
    ]
    placed_meshes, placed_colliders, stubs = [], [], []
    for k, (name, level, obstacle, overshoot, length, side) in enumerate(CASES):
        e = FIRST_STUB_M + k * STUB_EVERY_M
        n0 = side * HALF
        heading = math.pi / 2 if side > 0 else -math.pi / 2
        lo, hi = sorted([n0, n0 + side * length])
        meshes.append(_grid(f"1ROAD_STUB_{k}", "road_plain", e - STUB_W / 2, e + STUB_W / 2, lo, hi, 3.0,
                            lambda ee, nn, e=e: np.column_stack([(ee - e) / STUB_W + 0.5, nn / textures.ROAD_TILE_M]),
                            h=0.0))
        kit = closures.closure(level, STUB_W + 2 * VERGE, overshoot=overshoot, length=length, obstacle=obstacle,
                               seed=k, solid=solid)
        placed = closures.place(kit, (e, n0, 0.0, heading), FRAME, index=k)
        placed_meshes += placed.meshes
        placed_colliders += placed.colliders
        for s in placed.slots:  # a plain block where the vehicle will stand
            placed_meshes.append(_slot_block(f"SLOT_{k}", s))
        stubs.append((name, kit, e, n0, heading))
    meshes += closures.merge(placed_meshes) + closures.merge(placed_colliders)
    s = np.arange(0.0, ROAD_M + 0.5, 1.0)
    meshes += roadside.end_walls(s, np.column_stack([s, np.zeros_like(s)]), np.zeros_like(s), np.zeros_like(s),
                                 lambda x, y: np.zeros(np.shape(x)), FRAME)
    for m in meshes:
        m.validate()
    return meshes, stubs


def _slot_block(name, s):
    length, width = s.size
    f = np.array([math.cos(s.heading), math.sin(s.heading)])
    r = np.array([math.sin(s.heading), -math.cos(s.heading)])
    pos, nrm, tri = [], [], []
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    ring = [np.array([s.e, s.n]) + f * a * length / 2 + r * b * width / 2 for a, b in corners]
    for i in range(4):
        a, b = ring[i], ring[(i + 1) % 4]
        quad = [(a, s.h), (b, s.h), (b, s.h + 1.4), (a, s.h + 1.4)]
        out = (a + b) / 2 - np.array([s.e, s.n])
        base = len(pos)
        for (en, h) in quad:
            pos.append(FRAME.to_ac(en[0], en[1], h))
            nrm.append([out[0], 0.0, -out[1]])
        tri += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    base = len(pos)
    for en in ring:
        pos.append(FRAME.to_ac(en[0], en[1], s.h + 1.4))
        nrm.append([0.0, 1.0, 0.0])
    tri += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    pos, nrm, tri = np.array(pos), np.array(nrm, float), np.array(tri, dtype=np.int64)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    good = (np.cross(p1 - p0, p2 - p0) * nrm[tri].mean(axis=1)).sum(axis=1) >= 0
    tri = np.where(good[:, None], tri, tri[:, [0, 2, 1]])
    x0, y0, x1, y1 = closures.CELLS["white"]
    uv = np.tile([(x0 + x1) / 2 / closures.ATLAS_SIZE[0], (y0 + y1) / 2 / closures.ATLAS_SIZE[1]], (len(pos), 1))
    return Mesh(name, closures.MATERIAL, pos.astype(np.float32), nrm.astype(np.float32), uv.astype(np.float32), tri)


def scene_materials():
    images = {"road_marked": textures.LIBRARY["road_marked"](), "road_plain": textures.LIBRARY["road_plain"](),
              "ground": textures.LIBRARY["ground"](), closures.TEXTURE: closures.atlas().image}
    mats = {"road": Material("road", "road_marked"), "road_plain": Material("road_plain", "road_plain"),
            "terrain": Material("terrain", "ground"), "rail": Material("rail", "ground"), **closures.materials()}
    return mats, images


# -- the test track -------------------------------------------------------------------------------------------------

def write_track(out: Path, meshes) -> Path:
    import shutil

    mats, images = scene_materials()
    track = out / TRACK_ID
    if track.exists():
        shutil.rmtree(track)
    for sub in ("data", "ai", "ui", "extension"):
        (track / sub).mkdir(parents=True)
    tex = {f"{name}.dds": textures.to_dds(im.convert("RGB"), "DXT1") for name, im in images.items()}
    kmats, index = [], {}
    for name, mat in mats.items():
        props = (ROAD_PROPS if name.startswith("road") else GROUND_PROPS if name == "terrain"
                 else closures.KN5_PROPERTIES if name in closures.materials() else MATERIAL_PROPS)
        index[name] = len(kmats)
        kmats.append(kn5.Kn5Material(name=name, properties={**props, "ksEmissive": (0.0, 0.0, 0.0), "ksAlphaRef": 0.0},
                                     textures={"txDiffuse": f"{mat.texture}.dds"}))
    kmeshes = [kn5.Kn5Mesh(m.name, index[m.material], m.positions, m.normals, m.uvs, m.triangles,
                           tangents=kn5.tangents(m.positions.astype(float), m.normals.astype(float),
                                                 m.uvs.astype(float), m.triangles),
                           cast_shadows=m.renderable, renderable=m.renderable) for m in meshes]

    s = np.arange(0.0, ROAD_M + 0.5, 1.0)
    xy = np.column_stack([s, np.zeros_like(s)])
    zeros = np.zeros_like(s)
    road = RoadModel.from_arrays(s, xy, zeros, zeros, zeros, zeros, np.full_like(s, HALF), np.full_like(s, HALF),
                                 zeros, zeros, False)
    dummies = logic_dummies(road, FRAME)
    kdummies = [kn5.Kn5Dummy(d.name, kn5.dummy_matrix(d.position, d.forward, d.up)) for d in dummies]
    kn5.write(track / f"{TRACK_ID}.kn5", root=TRACK_ID, textures=tex, materials=kmats, meshes=kmeshes, dummies=kdummies)
    kn5.read(track / f"{TRACK_ID}.kn5")  # reads back cleanly, or raises

    line = ai_line.build(road, FRAME, end_short_m=AI_END_SHORT_M)
    ai_line.write(track / "ai" / "fast_lane.ai", line)
    xz = FRAME.to_ac(road.e, road.n, road.z)[:, [0, 2]]
    map_png, map_ini = package.map_files(xz)
    map_png.save(track / "map.png")
    files = {
        "data/map.ini": map_ini, "data/surfaces.ini": package.surfaces_ini(),
        "data/lighting.ini": package.lighting_ini(),
        "data/groove.ini": package.groove_ini(), "data/crew.ini": package.crew_ini(),
        "data/ai_hints.ini": package.ai_hints_ini(ai_line.pace_hints(road.length, finish_distance(road.length),
                                                                     line=line)),
        "ui/ui_track.json": package.ui_track_json(
            name="Closure props (test)", description="Side-road closures for an in-game look, generated by "
            "mapcreator. Not affiliated with or endorsed by Kunos Simulazioni.", tags=["point to point", "test"],
            lat=-42.0, lon=147.0, country="Australia", city="Test", length_m=road.length, width_m=(2 * HALF, 2 * HALF),
            run="point-to-point", version="test", year=2026,
            pitboxes=sum(d.name.startswith("AC_PIT_") for d in dummies)),
        "extension/ext_config.ini": package.ext_config_ini(
            grass_materials=["terrain"], road_materials=["road", "road_plain"], version="test",
            retro_materials={name: (night, closures.KN5_PROPERTIES["ksDiffuse"])
                             for name, night in closures.NIGHT_DIFFUSE.items()}),
        # `harness/ac_harness.py install` only replaces a folder whose ATTRIBUTION.txt says mapcreator made it.
        "ATTRIBUTION.txt": f"Closure props (test): Assetto Corsa track generated by mapcreator.{package.CRLF}"
                           f"No outside data: every texture and mesh is procedural.{package.CRLF}",
    }
    for rel, text in files.items():
        (track / rel).write_text(text, encoding="ascii", newline="")
    package.outline_png(xz).save(track / "ui" / "outline.png")
    Image.new("RGB", (355, 200), (90, 110, 80)).save(track / "ui" / "preview.png")
    return track


# -- a small software renderer --------------------------------------------------------------------------------------

def _camera(pos, target, size, fov_deg):
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 1.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    focal = size[0] / 2 / math.tan(math.radians(fov_deg) / 2)
    return np.asarray(pos, float), r, u, f, focal


def _clip_near(poly, near=0.05):
    """Sutherland-Hodgman against the near plane, on (camera xyz, u, v) rows."""
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ina, inb = a[2] > near, b[2] > near
        if ina:
            out.append(a)
        if ina != inb:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return out


def render(meshes, tex_of, cam, size=(640, 360), lines=()):
    w, h = size
    pos0, r, u, f, focal = cam
    sky = np.linspace([150, 185, 225], [205, 220, 235], h)[:, None, :]
    img = np.broadcast_to(sky, (h, w, 3)).astype(float).copy()
    depth = np.zeros((h, w))  # 1 / camera depth: larger is nearer
    sun = np.array([0.35, 0.8, -0.45])
    sun /= np.linalg.norm(sun)
    ys, xs = np.mgrid[0:h, 0:w] + 0.5
    for m in meshes:
        if not m.renderable:
            continue
        tex = tex_of[m.material]
        th, tw = tex.shape[:2]
        p = m.positions.astype(float) - pos0
        cpos = np.column_stack([p @ r, p @ u, p @ f])
        for tri in m.triangles:
            a, b, c = m.positions[tri].astype(float)
            n = np.cross(b - a, c - a)
            if n @ (pos0 - a) <= 0:  # back face: culled, as the game does
                continue
            n /= max(np.linalg.norm(n), 1e-12)
            shade = 0.5 + 0.5 * max(float(n @ sun), 0.0)
            poly = _clip_near([np.r_[cpos[i], m.uvs[i]] for i in tri])
            if len(poly) < 3:
                continue
            pts = np.array(poly)
            sx = w / 2 + pts[:, 0] / pts[:, 2] * focal
            sy = h / 2 - pts[:, 1] / pts[:, 2] * focal
            iz = 1.0 / pts[:, 2]
            for k in range(1, len(pts) - 1):
                idx = [0, k, k + 1]
                x0, x1 = int(max(math.floor(sx[idx].min()), 0)), int(min(math.ceil(sx[idx].max()), w))
                y0, y1 = int(max(math.floor(sy[idx].min()), 0)), int(min(math.ceil(sy[idx].max()), h))
                if x0 >= x1 or y0 >= y1:
                    continue
                X, Y = xs[y0:y1, x0:x1], ys[y0:y1, x0:x1]
                (ax, bx, cx), (ay, by, cy) = sx[idx], sy[idx]
                area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
                if abs(area) < 1e-9:
                    continue
                l1 = ((X - ax) * (cy - ay) - (cx - ax) * (Y - ay)) / area
                l2 = ((bx - ax) * (Y - ay) - (X - ax) * (by - ay)) / area
                l0 = 1 - l1 - l2
                inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
                if not inside.any():
                    continue
                z = l0 * iz[idx[0]] + l1 * iz[idx[1]] + l2 * iz[idx[2]]
                closer = inside & (z > depth[y0:y1, x0:x1])
                if not closer.any():
                    continue
                uu = (l0 * pts[idx[0], 3] * iz[idx[0]] + l1 * pts[idx[1], 3] * iz[idx[1]]
                      + l2 * pts[idx[2], 3] * iz[idx[2]]) / z
                vv = (l0 * pts[idx[0], 4] * iz[idx[0]] + l1 * pts[idx[1], 4] * iz[idx[1]]
                      + l2 * pts[idx[2], 4] * iz[idx[2]]) / z
                tx = (np.mod(uu, 1.0) * tw).astype(int) % tw
                ty = (np.mod(vv, 1.0) * th).astype(int) % th
                region = img[y0:y1, x0:x1]
                region[closer] = tex[ty[closer], tx[closer]] * shade
                depth[y0:y1, x0:x1][closer] = z[closer]
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    for a, b in lines:  # drawn on top, no depth test
        pa, pb = (np.array([(q - pos0) @ r, (q - pos0) @ u, (q - pos0) @ f]) for q in (a, b))
        if pa[2] < 0.05 or pb[2] < 0.05:
            continue
        d.line([(w / 2 + pa[0] / pa[2] * focal, h / 2 - pa[1] / pa[2] * focal),
                (w / 2 + pb[0] / pb[2] * focal, h / 2 - pb[1] / pb[2] * focal)], fill=(230, 20, 20), width=2)
    return out


def _collider_edges(meshes):
    edges = []
    for m in meshes:
        if m.renderable:
            continue
        p = m.positions.astype(float)
        for q in range(0, len(p), 4):  # each face a quad of 4 vertices
            for i in range(4):
                edges.append((p[q + i], p[q + (i + 1) % 4]))
    return edges


def write_renders(out: Path, meshes, stubs) -> list[Path]:
    mats, images = scene_materials()
    tex_of = {name: np.asarray(images[mat.texture].convert("RGB"), dtype=float) for name, mat in mats.items()}
    (out / "renders").mkdir(parents=True, exist_ok=True)
    paths = []
    for k, (name, kit, e, n0, heading) in enumerate(stubs):
        side = 1 if n0 > 0 else -1
        lane = 1.6  # the left lane, driving east
        zone = 25.0 if kit.overshoot else 0.0
        near = [m for m in meshes if _near(m, e, 140.0)]
        to_ac = FRAME.to_ac
        approach = _camera(to_ac(e - 45.0, lane, EYE), to_ac(e + 5.0, side * 10.0, 1.0), (640, 360), 60)
        mouth = _camera(to_ac(e, -side * lane, EYE), to_ac(e, side * (HALF + 12.0 + zone), 0.8), (640, 360), 55)
        panels = [render(near, tex_of, approach), render(near, tex_of, mouth),
                  render(near, tex_of, mouth, lines=_collider_edges([m for m in near if "CLOSURE" in m.name]))]
        sheet = Image.new("RGB", (640 * 3 + 20, 360 + 28), (32, 32, 32))
        d = ImageDraw.Draw(sheet)
        for i, (panel, label) in enumerate(zip(panels, ("driving up to it", "from the mouth", "colliders in red"))):
            sheet.paste(panel, (i * 650, 28))
            d.text((i * 650 + 6, 8), f"{name}: {label}", fill=(235, 235, 235))
        path = out / "renders" / f"{k}_{name}.png"
        sheet.save(path)
        paths.append(path)
    return paths


def _near(m: Mesh, e: float, reach: float) -> bool:
    x = m.positions[:, 0]
    return bool((x.min() < e + reach) and (x.max() > e - reach))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path, help="output folder, outside the repository")
    p.add_argument("--no-track", action="store_true", help="renders only")
    p.add_argument("--no-renders", action="store_true", help="the test track only")
    p.add_argument("--solid", nargs="*", metavar="NAME",
                   help=f"props with colliders, by kind {closures.SOLID_KINDS} and element {closures.SOLID_ELEMENTS} "
                        f"(default {closures.SOLID}; none at all with an empty list)")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    meshes, stubs = build_scene(args.solid)
    print("solid:", closures.SOLID if args.solid is None else tuple(args.solid))
    for name, kit, *_ in stubs:
        c = kit.counts()
        lines = ", ".join(f"{e.name} at {e.z:g} m" for e in kit.elements)
        print(f"{name:18s} {c['vertices']:4d} vertices, {c['triangles']:4d} triangles; colliders "
              f"{c['collider_vertices']} vertices; {lines}")
    if not args.no_track:
        print("track:", write_track(args.out, meshes))
    if not args.no_renders:
        for path in write_renders(args.out, meshes, stubs):
            print("render:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
