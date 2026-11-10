"""Police cars for closed side roads: one procedural low-poly car per Australian jurisdiction, in its livery.

The maintainer's decision (10 Nov 2026): a police car per state, in that state's full livery, with POLICE and the
crest. Everything is drawn here, from shapes: the body, the chequer, the lettering and each crest are our own
renditions, simplified. No image of a real crest or livery is copied, traced or embedded. The liveries come from
public descriptions and dated photos (sources in `THIRD_PARTY_NOTICES.md`, summary in `docs/ac-format.md`).

This module builds a car; placing it at a closure is the closure stage's job. `police_car(code)` gives meshes,
materials, the atlas and a collider in the car's own frame, and `place` puts a copy in a track's AC axes:
- **Local frame:** the origin is the car's centre on the ground, +z forward, +x to the car's right, +y up, in metres.
  (With x right, y up and z forward the frame is left-handed; `place` maps it onto AC's axes with the right side on
  the right, so nothing is mirrored, and rewinds the triangles to the KN5's rule.)
- **Meshes** are `scene.Mesh`, the type `furniture` hands the KN5 writer: one per material (body, reflective
  livery band, light bar), all drawing on the jurisdiction's one atlas, so up to six cars share one texture.
  `merge` joins placed cars by material.
- **Collider:** the body's own shape in three convex pieces (bonnet, cabin, boot), invisible, in a `1WALL_POLICE_n`
  mesh like the other solid props, so no invisible face stands above the bonnet or boot or beside the windows.
- **Which car:** `jurisdiction_for(region, lonlat)`, one per track.

**Body.** Lofted through cross-sections from the front bumper to the rear: each runs from the left sill up over the
roof and down to the right sill, and splits the side at the livery band, so the band is its own strip of faces.
Two shapes from the same code, a sedan and an SUV. Wheels are ten-sided, standing 5 mm proud of the body side so
they never hide behind it; the wheel arches are painted. Windows are painted on the sides, and the windscreen and
rear window take a glass cell. A static light bar and two mirrors. 550-590 vertices a car.

**Texture.** One 1024 x 1024 atlas per jurisdiction and body: the two sides (the left drawn on its own, so the
lettering reads forwards on both), the top, the front, the rear, a wheel, glass and the light bar. Each face takes
its UVs by projection onto its view. Drawn at twice the size and scaled down, so the edges are smooth. The lettering
uses Pillow's built-in Aileron font (CC0), made bold with a stroke.

**Night.** `night_diffuse` gives each part's ksDiffuse at night, for `package.ext_config_ini`'s `retro_materials`
(by day all three are 0.6). The livery band is retroreflective film on the real cars, so its faces are a material of
their own (`<base>_retro`) and get a boost. The light bar is off: its lenses (`<base>_lens`) are drawn as tinted,
unlit plastic and get a boost too, so they catch the headlights without glowing on their own; the track stage writes
`ksEmissive` 0 for every material, and a steady glow would read as a bar left on that never flashes. The plain body
comes down at night: white paint under the walls' ksDiffuse 0.6 reflects several times what the road does, and in the
headlights at close range it clipped to white. `kn5_properties` gives each part's KN5 values: no specular on the
paint or the band, since AC adds specular untextured, on top of the lettering as much as the paint.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
import shapely
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

from mapcreator.core.scene import MAX_VERTICES, Material, Mesh

# -- geometry ----------------------------------------------------------------------------------------------------

SILL_INSET = 0.05  # the sill sits this far inside the side's widest line
SHOULDER = 0.03  # and the belt (the shoulder under the windows) this far
BAND_BELOW_BELT = 0.03  # the band's top stays this far under the belt; where the belt drops, the band drops with it
WHEEL_SIDES = 10
WHEEL_PROUD = 0.005  # the tyre's outer face stands this far outside the body side, so the side never hides it
MIRROR = (0.16, 0.11, 0.08)  # out from the door, tall, long
COLLIDER_INSET = 0.01  # the collider's sides stand this far inside the side's widest line (the band)


@dataclass(frozen=True)
class Station:
    """A cross-section of the body at `z` (metres forward of the centre): half its width, the heights of its sill
    and belt (the top of the bonnet, the doors or the boot), and where the cabin is over it, the roof's height and
    half width."""

    z: float
    half_w: float
    sill: float
    belt: float
    roof: float | None = None
    roof_half_w: float = 0.0


@dataclass(frozen=True)
class Body:
    name: str
    stations: tuple[Station, ...]  # front to rear; the first and last have no roof
    axles: tuple[float, float]  # z of the front and rear axles
    wheel_r: float
    tyre_w: float
    bar: tuple[float, float, float]  # the light bar's width, depth and height
    bar_z: float  # its centre

    @property
    def length(self) -> float:
        return self.stations[0].z - self.stations[-1].z

    @property
    def width(self) -> float:
        return 2 * max(s.half_w for s in self.stations)

    @property
    def roof(self) -> float:
        return max(s.roof or 0.0 for s in self.stations)

    @property
    def height(self) -> float:
        """To the top of the light bar."""
        return self.roof + self.bar[2]

    @property
    def cabin(self) -> Station:
        """The first station under the roof, which sets the side's proportions (the livery's heights)."""
        return next(s for s in self.stations if s.roof is not None)

    def along(self, prop: str, z):
        """A station property (`sill` or `belt`) at z, linear between stations."""
        zs = [s.z for s in self.stations][::-1]
        return np.interp(z, zs, [getattr(s, prop) for s in self.stations][::-1])


# A mid-size sedan (about a Toyota Camry: 4.89 m long, 1.84 m wide, 1.445 m tall, a 2.825 m wheelbase) and a
# mid-size SUV (about a Kia Sorento: 4.81 x 1.90 x 1.70 m, 2.815 m), the two shapes of Australia's marked fleets.
SEDAN = Body(
    "sedan",
    (Station(2.445, 0.80, 0.30, 0.68),
     Station(2.36, 0.89, 0.26, 0.74),
     Station(2.05, 0.915, 0.24, 0.80),
     Station(1.20, 0.92, 0.22, 0.88),
     Station(0.80, 0.92, 0.20, 0.92),  # the cowl, at the windscreen's foot
     Station(-0.05, 0.92, 0.20, 0.95, 1.445, 0.64),  # the windscreen's top
     Station(-0.95, 0.92, 0.20, 0.98, 1.43, 0.64),
     Station(-1.55, 0.92, 0.22, 1.00),  # the rear window's foot
     Station(-2.25, 0.905, 0.26, 1.00),
     Station(-2.37, 0.88, 0.30, 0.97),
     Station(-2.445, 0.80, 0.34, 0.90)),
    axles=(1.465, -1.36), wheel_r=0.33, tyre_w=0.225, bar=(1.15, 0.30, 0.09), bar_z=-0.40)
SUV = Body(
    "suv",
    (Station(2.405, 0.84, 0.36, 0.80),
     Station(2.32, 0.93, 0.33, 0.88),
     Station(2.00, 0.95, 0.30, 0.95),
     Station(1.20, 0.95, 0.30, 1.03),
     Station(0.78, 0.95, 0.30, 1.06),
     Station(0.00, 0.95, 0.30, 1.08, 1.70, 0.74),
     Station(-2.05, 0.95, 0.30, 1.10, 1.69, 0.72),
     Station(-2.30, 0.94, 0.32, 1.10),
     Station(-2.37, 0.92, 0.36, 1.08),
     Station(-2.405, 0.85, 0.40, 1.02)),
    axles=(1.475, -1.34), wheel_r=0.37, tyre_w=0.235, bar=(1.20, 0.30, 0.09), bar_z=-0.55)
BODIES = {b.name: b for b in (SEDAN, SUV)}


# -- atlas -------------------------------------------------------------------------------------------------------

ATLAS = 1024  # px square, one per jurisdiction and body
SUPERSAMPLE = 2  # drawn at twice the size, then scaled down
PAD_M = 0.08  # each view's margin around the body, so filtering never reaches the next cell
# Cells, in atlas pixels (x0, y0, x1, y1).
CELLS = {
    "side_r": (0, 0, 1024, 384),
    "side_l": (0, 384, 1024, 768),
    "top": (0, 768, 1024, 896),
    "front": (0, 896, 256, 1024),
    "rear": (256, 896, 512, 1024),
    "wheel": (512, 896, 640, 1024),
    "tread": (640, 896, 704, 960),
    "dark": (640, 960, 704, 1024),
    "glass": (704, 896, 768, 960),
    "paint": (704, 960, 768, 1024),
    "bar_front": (768, 896, 1024, 928),  # the light bar from in front: the car's right half on the left
    "bar_back": (768, 928, 1024, 960),  # from behind: its left half on the left, so the lettering reads on both
    "bar_top": (768, 960, 1024, 1024),
}


@dataclass(frozen=True)
class View:
    """Maps a face's two coordinates (metres) to its cell: `a` across the cell, `b` down it."""

    cell: str
    a0: float
    a1: float
    b0: float  # at the cell's top
    b1: float  # at its bottom

    def uv(self, a, b) -> np.ndarray:
        x0, y0, x1, y1 = CELLS[self.cell]
        a, b = np.asarray(a, float), np.asarray(b, float)
        u = (x0 + (a - self.a0) / (self.a1 - self.a0) * (x1 - x0)) / ATLAS
        v = (y0 + (b - self.b0) / (self.b1 - self.b0) * (y1 - y0)) / ATLAS
        return np.stack([u, v], axis=-1)

    def px(self, a, b) -> tuple[float, float]:
        """The point in the cell's own drawing (atlas pixels times SUPERSAMPLE, from the cell's top left)."""
        x0, y0, x1, y1 = CELLS[self.cell]
        return (float((a - self.a0) / (self.a1 - self.a0) * (x1 - x0) * SUPERSAMPLE),
                float((b - self.b0) / (self.b1 - self.b0) * (y1 - y0) * SUPERSAMPLE))


def views(body: Body) -> dict[str, View]:
    """Each cell's projection, fitted to the body with a margin. Seen from outside: on the right side the front is to
    the right, on the left side to the left; the top has the front to the right and the right side at the bottom; the
    front view has the car's right on the viewer's left, the rear view on the viewer's right."""
    front, rear = body.stations[0], body.stations[-1]
    zf, zr = front.z + PAD_M, rear.z - PAD_M
    top, bottom = body.roof + PAD_M, max(min(s.sill for s in body.stations) - PAD_M, 0.0)
    w = body.width / 2 + PAD_M
    fw, rw = front.half_w + PAD_M, rear.half_w + PAD_M
    return {
        "side_r": View("side_r", zr, zf, top, bottom),
        "side_l": View("side_l", zf, zr, top, bottom),
        "top": View("top", zr, zf, -w, w),
        "front": View("front", fw, -fw, front.belt + PAD_M, front.sill - PAD_M),
        "rear": View("rear", -rw, rw, rear.belt + PAD_M, rear.sill - PAD_M),
    }


# -- liveries ----------------------------------------------------------------------------------------------------

RGB = tuple[int, int, int]
# White paint at 224, not 246: near-white under the track stage's ksDiffuse 0.6 reflects several times what the road
# does, and clipped to white in the headlights at night (in game, 10 Nov 2026). Silver at 164-173, darker than first
# drawn (195): in sunlight 195 rendered the same as the white. A whiter white would grey the white cars by day until
# they no longer stand apart from the silver.
WHITE: RGB = (224, 224, 222)
SILVER_PAINT: RGB = (164, 168, 173)
RED: RGB = (214, 24, 32)
BLUE: RGB = (24, 64, 214)


@dataclass(frozen=True)
class Livery:
    """A jurisdiction's marked car, as drawn here.

    Heights are fractions of the side from the sill (0) to the belt (1) at the cabin, and places along the car
    fractions of its length from the front (0) to the rear (1), so one livery fits either body. Sizes are metres."""

    code: str
    service: str  # who polices there
    body: str  # the default body, "sedan" or "suv": the current fleet's usual shape
    colours: tuple[RGB, RGB]  # the chequer's dark and light squares; the dark one leads the top row at the front
    rows: int
    band: tuple[float, float]  # its bottom and top
    word: str  # "Police" or "POLICE", as each service writes it
    word_colour: RGB
    word_side: tuple[float, float, float, float]  # along, height, cap height, longest
    paint: RGB = WHITE
    pattern: str = "sillitoe"  # small squares, or "wedge": large squares rising from front to rear (Queensland)
    wedge: tuple[int, int] = (2, 4)  # rows at the wedge's front (behind the front wheel) and rear (the rear axle)
    wedge_edge: RGB | None = None  # a line along the wedge's top
    front: str = ""  # the band's front end: "scatter" (squares breaking up over the guard) or "enlarge"
    stripes: tuple[tuple[RGB, float, float], ...] = ()  # thin lines along the side: colour, height, thickness
    panels: tuple[tuple[str, float, float], ...] = ()  # plain panels set into the band: lettering, along, width
    extra: tuple[tuple[str, float, float, float], ...] = ()  # more lettering: text, along, height, cap height
    word_stretch: float = 1.1
    word_outline: tuple[tuple[RGB, float], ...] = ()  # keylines round the word, widest first: colour, width (of cap)
    word_bonnet: RGB | None = None  # the bonnet's lettering colour, if it has any
    word_rear: bool = True
    band_rear: bool = True  # the band carries on across the rear
    corner_flash: RGB | None = None  # colour patches on the front bumper's corners
    crest: str = ""  # its drawer in CRESTS
    crest_side: tuple[float, float, float] | None = None  # along, height, size
    crest_bonnet: float = 0.0  # its size on the bonnet (0: none there)
    lightbar: tuple[RGB, RGB] = (RED, BLUE)  # the car's left half, right half
    bar_word: str = ""  # lettering on the bar's centre lens


# -- building ----------------------------------------------------------------------------------------------------


class _Parts:
    """Polygons in the car's frame, by material, with UVs; oriented and smoothed when packed."""

    def __init__(self):
        self.polys: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, str | None]] = []

    def poly(self, material: str, pts, uvs, outward, smooth: str | None = None) -> None:
        """A convex polygon. `outward` is the side its face should look to; `smooth` names a group within which
        normals are averaged where corners meet (None: flat)."""
        pts = np.asarray(pts, float)
        uvs = np.asarray(uvs, float)
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-6:
                keep.append(i)
        if len(keep) > 1 and np.linalg.norm(pts[keep[-1]] - pts[keep[0]]) <= 1e-6:
            keep.pop()
        if len(keep) < 3:
            return
        pts, uvs = pts[keep], uvs[keep]
        normal = np.zeros(3)
        for i in range(1, len(pts) - 1):
            normal += np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])
        if np.linalg.norm(normal) < 1e-8:
            return
        if normal @ np.asarray(outward, float) < 0:
            pts, uvs, normal = pts[::-1], uvs[::-1], -normal
        self.polys.append((material, pts, uvs, normal, smooth))

    def meshes(self, prefix: str, names: dict[str, str] | None = None) -> list[Mesh]:
        """One mesh per material, named `prefix_<names[material]>` (by default the material's name)."""
        # Smoothed normals: the area-weighted face normals of every polygon in the group that has a corner there.
        acc: dict[tuple, np.ndarray] = {}
        for _, pts, _, normal, smooth in self.polys:
            if smooth:
                for p in pts:
                    key = (smooth, *np.round(p, 5))
                    acc[key] = acc.get(key, np.zeros(3)) + normal
        out: dict[str, tuple[list, list, list, list]] = {}
        for material, pts, uvs, normal, smooth in self.polys:
            pos, nrm, uv, tri = out.setdefault(material, ([], [], [], []))
            face = normal / np.linalg.norm(normal)
            base = len(pos)
            for p, t in zip(pts, uvs):
                n = acc[(smooth, *np.round(p, 5))] if smooth else face
                pos.append(p)
                nrm.append(n / np.linalg.norm(n))
                uv.append(t)
            tri.extend((base, base + i, base + i + 1) for i in range(1, len(pts) - 1))
        meshes = []
        for material, (pos, nrm, uv, tri) in out.items():
            suffix = (names or {}).get(material, material).upper()
            meshes.append(Mesh(f"{prefix}_{suffix}", material, np.array(pos, np.float32),
                               np.array(nrm, np.float32), np.array(uv, np.float32), np.array(tri, np.int64)))
        return meshes


def band_heights(body: Body, livery: Livery) -> tuple[float, float]:
    """The band's bottom and top (metres) along the doors."""
    mid = body.cabin
    side = mid.belt - mid.sill
    return mid.sill + livery.band[0] * side, mid.sill + livery.band[1] * side


def _band_at(sill: float, belt: float, band: tuple[float, float], anchored: bool) -> tuple[float, float]:
    """The band's bottom and top where the body has this sill and belt. A band along the shoulder drops with the
    belt towards the nose and tail, keeping its depth; a wedge (`anchored`) keeps its foot and is cut by the belt."""
    b0, b1 = band
    top = min(b1, belt - BAND_BELOW_BELT)
    bottom = max(b0 if anchored else top - (b1 - b0), sill + 0.02)
    return min(bottom, top - 0.02), top


def _section(st: Station, band: tuple[float, float], anchored: bool) -> np.ndarray:
    """The station's cross-section, (10, 2) of x and y: from the left sill up over the top and down to the right
    sill. Points 1-2 and 7-8 bound the livery band; 3-4 and 5-6 the cabin's sides (one point where there's none)."""
    y0, y1 = _band_at(st.sill, st.belt, band, anchored)
    right = [(st.half_w - SILL_INSET, st.sill), (st.half_w, y0), (st.half_w, y1), (st.half_w - SHOULDER, st.belt)]
    right.append((st.roof_half_w, st.roof) if st.roof is not None else right[-1])
    left = [(-x, y) for x, y in right]
    return np.array(left + right[::-1], float)


def _rot_out(d: np.ndarray) -> np.ndarray:
    """Outward from a section's segment direction (x, y): the loop runs left, over the top, right."""
    return np.array([-d[1], d[0], 0.0])


def _body(parts: _Parts, body: Body, livery: Livery, mats: dict[str, str]) -> None:
    band = band_heights(body, livery)
    anchored = livery.pattern == "wedge"
    vw = views(body)
    st = body.stations
    secs = [_section(s, band, anchored) for s in st]
    glass_x = body.width / 2
    for i in range(len(st) - 1):
        a, b = secs[i], secs[i + 1]
        za, zb = st[i].z, st[i + 1].z
        windscreen = st[i].roof is None and st[i + 1].roof is not None
        rear_window = st[i].roof is not None and st[i + 1].roof is None
        for j in range(9):
            pts = np.array([[*a[j], za], [*a[j + 1], za], [*b[j + 1], zb], [*b[j], zb]])
            out = _rot_out(a[j + 1] - a[j]) + _rot_out(b[j + 1] - b[j])
            if j == 4:
                if windscreen or rear_window:
                    # The glass cell across, from the pane's top (v 0) to its foot (v 1).
                    x0, y0, x1, y1 = CELLS["glass"]
                    u = (x0 + 2 + (pts[:, 0] + glass_x) / (2 * glass_x) * (x1 - x0 - 4)) / ATLAS
                    edge = np.array([0, 0, 1, 1]) if rear_window else np.array([1, 1, 0, 0])
                    v = (y0 + 2 + edge * (y1 - y0 - 4)) / ATLAS
                    parts.poly(mats["body"], pts, np.column_stack([u, v]), out, "glass")
                else:
                    parts.poly(mats["body"], pts, vw["top"].uv(pts[:, 2], pts[:, 0]), out, "top")
                continue
            view = vw["side_l"] if j < 4 else vw["side_r"]
            material = mats["retro"] if j in (1, 7) else mats["body"]
            parts.poly(material, pts, view.uv(pts[:, 2], pts[:, 1]), out, view.cell)
    for end, s, sec, sign in (("front", st[0], secs[0], 1.0), ("rear", st[-1], secs[-1], -1.0)):
        wraps = end == "rear" and livery.band_rear
        for k in range(3):
            lk, lk1, rk, rk1 = sec[k], sec[k + 1], sec[9 - k], sec[8 - k]
            pts = np.array([[*lk, s.z], [*lk1, s.z], [*rk1, s.z], [*rk, s.z]])
            material = mats["retro"] if k == 1 and wraps else mats["body"]
            parts.poly(material, pts, vw[end].uv(pts[:, 0], pts[:, 1]), (0.0, 0.0, sign))


def _wheels(parts: _Parts, body: Body, mats: dict[str, str]) -> None:
    hw = max(s.half_w for s in body.stations)
    r = body.wheel_r
    theta = np.linspace(0, 2 * np.pi, WHEEL_SIDES + 1) - np.pi / 2  # a corner at the bottom, on the ground
    x0, y0, x1, y1 = CELLS["tread"]
    fx0, fy0, fx1, fy1 = CELLS["wheel"]
    for z in body.axles:
        for side in (1.0, -1.0):
            xo = side * (hw + WHEEL_PROUD)
            xi = xo - side * body.tyre_w
            ring = np.column_stack([np.sin(theta) * r + r, np.cos(theta) * r + z])  # y, z
            for k in range(WHEEL_SIDES):
                (ya, za), (yb, zb) = ring[k], ring[k + 1]
                pts = [(xo, ya, za), (xi, ya, za), (xi, yb, zb), (xo, yb, zb)]
                u = np.array([x0 + 2, x1 - 2, x1 - 2, x0 + 2]) / ATLAS
                v = (y0 + 2 + (np.array([k, k, k + 1, k + 1]) / WHEEL_SIDES) * (y1 - y0 - 4)) / ATLAS
                mid = (theta[k] + theta[k + 1]) / 2
                parts.poly(mats["body"], pts, np.column_stack([u, v]), (0.0, np.sin(mid), np.cos(mid)),
                           f"tyre{z:+.2f}{side:+.0f}")
            cu, cv = (fx0 + fx1) / 2, (fy0 + fy1) / 2
            half = (fx1 - fx0) / 2 - 2
            # Seen from outside, the face's +z (forward) is to the right on the right side and to the left on the left.
            uv = np.column_stack([(cu + side * np.cos(theta[:-1]) * half) / ATLAS,
                                  (cv - np.sin(theta[:-1]) * half) / ATLAS])
            pts = np.column_stack([np.full(WHEEL_SIDES, xo), ring[:-1, 0], ring[:-1, 1]])
            parts.poly(mats["body"], pts, uv, (side, 0.0, 0.0))


def _box(parts: _Parts, material: str, lo, hi, uv_for, skip=("bottom",)) -> None:
    """An axis-aligned box from `lo` to `hi` (x, y, z); `uv_for(face, pts)` gives each face's UVs."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    faces = {
        "right": ([(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)], (1, 0, 0)),
        "left": ([(x0, y0, z1), (x0, y0, z0), (x0, y1, z0), (x0, y1, z1)], (-1, 0, 0)),
        "top": ([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], (0, 1, 0)),
        "bottom": ([(x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0)], (0, -1, 0)),
        "front": ([(x1, y0, z1), (x0, y0, z1), (x0, y1, z1), (x1, y1, z1)], (0, 0, 1)),
        "rear": ([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], (0, 0, -1)),
    }
    for name, (pts, out) in faces.items():
        if name not in skip:
            pts = np.array(pts, float)
            parts.poly(material, pts, uv_for(name, pts), out)


def _cell_uv(cell: str, a, b, a_range, b_range) -> np.ndarray:
    """UVs in `cell` for coordinates a (across) and b (down) spanning the given ranges, 2 px inside its edges."""
    x0, y0, x1, y1 = CELLS[cell]
    a = (np.asarray(a, float) - a_range[0]) / (a_range[1] - a_range[0])
    b = (np.asarray(b, float) - b_range[0]) / (b_range[1] - b_range[0])
    return np.column_stack([(x0 + 2 + a * (x1 - x0 - 4)) / ATLAS, (y0 + 2 + b * (y1 - y0 - 4)) / ATLAS])


def _light_bar(parts: _Parts, body: Body, mats: dict[str, str]) -> None:
    w, d, h = body.bar
    y0 = body.roof - 0.005
    lo, hi = (-w / 2, y0, body.bar_z - d / 2), (w / 2, y0 + h, body.bar_z + d / 2)
    right_first = (w / 2, -w / 2)

    def uv(face, pts):
        if face == "front":
            return _cell_uv("bar_front", pts[:, 0], pts[:, 1], right_first, (y0 + h, y0))
        if face == "rear":
            return _cell_uv("bar_back", pts[:, 0], pts[:, 1], right_first[::-1], (y0 + h, y0))
        if face == "top":
            return _cell_uv("bar_top", pts[:, 0], pts[:, 2], right_first, (lo[2], hi[2]))
        end = 0.02 if face == "right" else 0.98  # an end takes the colour of its half
        a = np.full(len(pts), right_first[0] + end * (right_first[1] - right_first[0]))
        return _cell_uv("bar_front", a, pts[:, 1], right_first, (y0 + h, y0))

    _box(parts, mats["lens"], lo, hi, uv)


def _mirrors(parts: _Parts, body: Body, mats: dict[str, str]) -> None:
    cowl = body.stations[next(i for i, s in enumerate(body.stations) if s.roof is not None) - 1]
    out, tall, long = MIRROR
    y0 = cowl.belt + 0.06
    z0 = cowl.z - 0.25

    def uv(face, pts):
        return _cell_uv("glass" if face == "rear" else "paint", [0, 1, 1, 0], [1, 1, 0, 0], (0, 1), (0, 1))

    for side in (1.0, -1.0):
        x_in = side * (cowl.half_w - 0.14)  # into the door, so no gap shows above the tumblehome
        x_out = side * (cowl.half_w + out)
        _box(parts, mats["body"], (min(x_in, x_out), y0, z0 - long / 2), (max(x_in, x_out), y0 + tall, z0 + long / 2),
             uv)


def _collider_section(st: Station) -> list[tuple[float, float]]:
    """A station's cross-section for the collider (x, y): straight sides from the sill to the belt, 1 cm inside the
    band's line, then in to the roof's edge where the cabin is over it."""
    x = st.half_w - COLLIDER_INSET
    right = [(x, st.sill), (x, st.belt)] + ([(st.roof_half_w, st.roof)] if st.roof is not None else [])
    return right + [(-px, py) for px, py in right[::-1]]


def collider_pieces(body: Body) -> list[np.ndarray]:
    """The collider's three convex pieces, as the points (x, y, z) they're the hulls of: the bonnet (nose to the
    windscreen's foot), the cabin (to the rear window's foot) and the boot (to the tail)."""
    st = body.stations
    first = next(i for i, s in enumerate(st) if s.roof is not None)
    last = max(i for i, s in enumerate(st) if s.roof is not None)
    spans = ((0, first - 1), (first - 1, last + 1), (last + 1, len(st) - 1))
    return [np.array([(x, y, s.z) for s in st[a:b + 1] for x, y in _collider_section(s)]) for a, b in spans]


def _collider(body: Body, material: str) -> Mesh:
    """The body's shape for collisions, so nothing solid stands where nothing shows: three convex pieces (bonnet,
    cabin, boot), each the hull of the body's own cross-sections, so it follows the bonnet, windscreen, roof, rear
    window and boot lines, the cabin's tumblehome and the narrowing nose and tail. It runs from the sill (the open
    space under the car stays open) to the top. Each piece is closed, its faces wound outwards."""
    pos, nrm, tri = [], [], []
    for pts in collider_pieces(body):
        hull = ConvexHull(pts)
        index = {int(v): len(pos) + k for k, v in enumerate(hull.vertices)}
        acc = np.zeros((len(hull.vertices), 3))
        for (a, b, c), eq in zip(hull.simplices, hull.equations):
            if np.cross(pts[b] - pts[a], pts[c] - pts[a]) @ eq[:3] < 0:
                b, c = c, b
            tri.append((index[a], index[b], index[c]))
            face = np.cross(pts[b] - pts[a], pts[c] - pts[a])
            for v in (a, b, c):
                acc[index[v] - len(pos)] += face
        pos.extend(pts[hull.vertices])
        nrm.extend(acc / np.linalg.norm(acc, axis=1, keepdims=True))
    n = len(pos)
    return Mesh(COLLIDER_MESH, material, np.array(pos, np.float32), np.array(nrm, np.float32),
                np.zeros((n, 2), np.float32), np.array(tri, np.int64), renderable=False)


# -- the car -----------------------------------------------------------------------------------------------------

COLLIDER_MESH = "1WALL_POLICE"
VISUAL_MESH = "POLICE"
# KN5 material values, by part (`PoliceCar.kn5_properties`, for the track stage, which keys its MATERIAL_PROPS by
# the name's first word and would otherwise give every part the walls' values). ksDiffuse stays the walls' 0.6 by day,
# which the night adjustments start from.
#
# No specular on the paint or the band, which carry the lettering and the chequer. AC's ksPerPixel adds its specular
# on top of the texture, not through it: txColor * (ksAmbient * ambient + ksDiffuse * light + ksEmissive) + ksSpecular *
# (N.H)^ksSpecularEXP * light (Content Manager's re-implementation of the shader, `AcTools.Render/Shaders/Includes/
# DarkMaterial.Lighting.Simple.fx`). At night the headlights sit beside the eye, so N.H is near 1 across a side square
# to them, and the specular lifts black lettering as much as white paint: at 0.12 it came to about 40% of the white's
# light at the night 0.35, and the lettering faded to pale lavender (in game, 11 Nov 2026). Tasmania's silver comes
# from its albedo alone. The lenses keep a small, tight glint.
PAINT_PROPS = {"ksAmbient": 0.4, "ksDiffuse": 0.6, "ksSpecular": 0.0, "ksSpecularEXP": 40.0}
LENS_PROPS = {"ksAmbient": 0.4, "ksDiffuse": 0.6, "ksSpecular": 0.05, "ksSpecularEXP": 200.0}
# ksDiffuse at night, by part, for `ext_config_ini(retro_materials=...)` ([MATERIAL_ADJUSTMENT_n], CONDITION =
# NIGHT_SMOOTH; by day each goes back to 0.6). In game the first values (body 0.6, band 2.0, lenses 4.0) washed out at
# 6 m in the beam; with these, and the specular still on, the bodies were as bright as by day but 24-31% near-white,
# and the bars read as off from 12 m. With the specular gone the hot spot loses about a quarter of its light, so the
# body stays at 0.35, the closure props' value, side by side with them. To retest in game.
NIGHT_DIFFUSE_BY_PART = {"body": 0.35, "retro": 1.3, "lens": 2.0}


@dataclass
class PoliceCar:
    jurisdiction: str
    body: Body
    meshes: list[Mesh]  # in the car's frame (see the module notes), one per material
    collider: Mesh  # likewise; invisible
    materials: dict[str, Material]
    textures: dict[str, Image.Image]  # texture name -> its image, drawn here (saved as the stage saves furniture's)
    night_diffuse: dict[str, float] = field(default_factory=dict)  # material -> ksDiffuse at night
    kn5_properties: dict[str, dict[str, float]] = field(default_factory=dict)  # material -> KN5 values by day

    @property
    def vertices(self) -> int:
        return sum(len(m.positions) for m in self.meshes)

    @property
    def triangles(self) -> int:
        return sum(len(m.triangles) for m in self.meshes)


def material_names(code: str, body: str | None = None) -> dict[str, str]:
    """The car's three materials. They share one texture, named like the body material. A body other than the
    jurisdiction's own is named in them, since its atlas differs."""
    base = f"police_{code}" if body in (None, LIVERIES[code].body) else f"police_{code}_{body}"
    return {"body": base, "retro": f"{base}_retro", "lens": f"{base}_lens"}


def police_car(code: str, body: str | None = None) -> PoliceCar:
    """The marked car of jurisdiction `code` (one of JURISDICTIONS), in its own frame, with its default body unless
    `body` ("sedan" or "suv") says otherwise. The meshes are new arrays each call; the atlas is drawn once."""
    if code not in LIVERIES:
        raise ValueError(f"unknown jurisdiction {code!r}; expected one of {', '.join(JURISDICTIONS)}")
    livery = LIVERIES[code]
    shape = BODIES[body or livery.body]
    mats = material_names(code, shape.name)
    parts = _Parts()
    _body(parts, shape, livery, mats)
    _wheels(parts, shape, mats)
    _mirrors(parts, shape, mats)
    _light_bar(parts, shape, mats)
    prefix = f"{VISUAL_MESH}_{code.upper()}" + ("" if shape.name == livery.body else f"_{shape.name.upper()}")
    meshes = parts.meshes(prefix, {m: part for part, m in mats.items()})
    texture = mats["body"]
    return PoliceCar(
        jurisdiction=code, body=shape, meshes=meshes, collider=_collider(shape, mats["body"]),
        materials={m: Material(m, texture) for m in sorted({m.material for m in meshes})},
        textures={texture: atlas(code, shape.name).copy()},
        night_diffuse={mats[part]: v for part, v in NIGHT_DIFFUSE_BY_PART.items()},
        kn5_properties={mats["body"]: dict(PAINT_PROPS), mats["retro"]: dict(PAINT_PROPS),
                        mats["lens"]: dict(LENS_PROPS)})


def place(car: PoliceCar, position, forward, up=(0.0, 1.0, 0.0), index: int = 0) -> list[Mesh]:
    """A copy of `car` in AC axes: its centre on the ground at `position`, facing `forward`, with `up` its up
    (tilted on a slope). Returns the visual meshes and the collider, named with `index` so copies don't clash."""
    up = np.asarray(up, float)
    up /= np.linalg.norm(up)
    fwd = np.asarray(forward, float)
    fwd = fwd - up * (fwd @ up)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)  # in AC's axes (x east, y up, z south), forward x up is the car's right
    basis = np.stack([right, up, fwd])  # rows: where the car's x, y and z go
    origin = np.asarray(position, float)
    out = []
    for m in [*car.meshes, car.collider]:
        pos = m.positions.astype(float) @ basis + origin
        nrm = m.normals.astype(float) @ basis
        tri = m.triangles.copy()
        p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
        flip = (np.cross(p1 - p0, p2 - p0) * nrm[tri].mean(axis=1)).sum(axis=1) < 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        out.append(replace(m, name=f"{m.name}_{index}", positions=pos.astype(np.float32),
                           normals=nrm.astype(np.float32), triangles=tri))
    return out


def merge(meshes: list[Mesh]) -> list[Mesh]:
    """Placed cars' meshes joined by material (and their colliders by material too), under the vertex limit, so six
    cars cost a draw call per material rather than per car."""
    groups: dict[tuple[str, bool], list[Mesh]] = {}
    for m in meshes:
        groups.setdefault((m.material, m.renderable), []).append(m)
    out = []
    count: dict[str, int] = {}
    for (material, renderable), ms in groups.items():
        chunks: list[list[Mesh]] = [[]]
        for m in ms:
            if chunks[-1] and sum(len(c.positions) for c in chunks[-1]) + len(m.positions) > MAX_VERTICES:
                chunks.append([])
            chunks[-1].append(m)
        for chunk in chunks:
            base = np.cumsum([0] + [len(c.positions) for c in chunk[:-1]])
            stem = COLLIDER_MESH if not renderable else material.upper()  # POLICE_TAS, POLICE_TAS_RETRO, ...
            name = f"{stem}_{count.get(stem, 0)}"
            count[stem] = count.get(stem, 0) + 1
            out.append(Mesh(name, material, np.concatenate([c.positions for c in chunk]),
                            np.concatenate([c.normals for c in chunk]), np.concatenate([c.uvs for c in chunk]),
                            np.concatenate([c.triangles + b for c, b in zip(chunk, base)]), renderable))
    return out


# -- which jurisdiction ------------------------------------------------------------------------------------------

JURISDICTIONS = ("tas", "nsw", "vic", "qld", "sa", "wa", "nt", "act")
DEFAULT_JURISDICTION = "tas"  # with no state and no points: the service's first state
STATE_NAMES = {"tasmania": "tas", "new south wales": "nsw", "victoria": "vic", "queensland": "qld",
               "south australia": "sa", "western australia": "wa", "northern territory": "nt",
               "australian capital territory": "act"}
# The borders, coarsely (lon, lat; to within about 10 km, enough to pick a prop). Most are meridians and parallels:
# WA at 129 E; NT south to 26 S and east to 138 E; Queensland's west at 138 E to 26 S, then 141 E to 29 S; SA's east at
# 141 E, then 140.97 E south of the Murray (the surveyed Victorian line). Queensland/NSW: 29 S to the Barwon at
# Mungindi, then the rivers past Goondiwindi and Texas to Wallangarra and the ranges to Point Danger. NSW/Victoria:
# the Murray from the SA border to its source, then the straight line to Cape Howe. West to east, both.
QLD_NSW = ((141.0, -29.0), (148.97, -29.0), (149.45, -28.72), (150.31, -28.56), (150.85, -28.65), (151.17, -28.87),
           (151.55, -28.95), (151.93, -28.90), (152.05, -28.60), (152.40, -28.35), (152.72, -28.34),
           (153.10, -28.30), (153.35, -28.23), (153.55, -28.165))
NSW_VIC = ((140.97, -34.03), (141.91, -34.11), (142.16, -34.17), (142.77, -34.57), (143.17, -34.72),
           (143.35, -35.03), (143.57, -35.33), (144.13, -35.64), (144.75, -36.12), (144.96, -36.02),
           (145.57, -35.82), (146.00, -36.00), (146.39, -36.00), (146.90, -36.10), (147.10, -36.10),
           (147.70, -35.93), (147.98, -36.20), (148.12, -36.55), (148.20, -36.80), (149.976, -37.505))
# The ACT, a coarse outline (lon, lat): Canberra in the north, Namadgi down to its southern tip.
ACT = ((148.98, -35.13), (149.07, -35.12), (149.13, -35.13), (149.25, -35.20), (149.33, -35.26), (149.40, -35.31),
       (149.37, -35.35), (149.27, -35.34), (149.22, -35.33), (149.21, -35.40), (149.20, -35.45), (149.13, -35.55),
       (149.13, -35.70), (149.10, -35.85), (149.05, -35.92), (148.95, -35.85), (148.85, -35.72), (148.80, -35.60),
       (148.77, -35.45), (148.80, -35.35), (148.90, -35.28), (148.95, -35.20))
BASS_STRAIT_LAT = -39.2  # Tasmania and its islands lie south of this


def _lat_on(line, lon: float) -> float:
    xs, ys = zip(*line)
    return float(np.interp(lon, xs, ys))


def _in_polygon(lon: float, lat: float, poly) -> bool:
    inside = False
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        if (y0 > lat) != (y1 > lat) and lon < x0 + (lat - y0) / (y1 - y0) * (x1 - x0):
            inside = not inside
    return inside


def jurisdiction_at(lon: float, lat: float) -> str:
    """The jurisdiction a point (WGS84) falls in, by the coarse borders above. Every point in Australia's box gets
    one; a point at sea goes with the land it faces."""
    if lat < BASS_STRAIT_LAT:
        return "tas"
    if _in_polygon(lon, lat, ACT):
        return "act"
    if lon < 129.0:
        return "wa"
    if lon < 138.0:
        return "nt" if lat > -26.0 else "sa"
    if lon < 140.97:
        return "qld" if lat > -26.0 else "sa"
    if lon < 141.0:  # the sliver between the two SA lines: Victoria south of the Murray
        return "qld" if lat > -26.0 else "vic" if lat < _lat_on(NSW_VIC, lon) else "sa"
    if lat > _lat_on(QLD_NSW, lon):
        return "qld"
    return "nsw" if lat > _lat_on(NSW_VIC, lon) else "vic"


def jurisdiction_for(region=None, lonlat=()) -> str:
    """The jurisdiction whose police car a track gets, one for the whole track.

    `region` is the track's `regions.Region` (anything with a `.state`), `lonlat` the route's points (WGS84). A region
    that names its state decides (Tasmania does). Otherwise the points do: the jurisdiction holding most of them, so a
    route across a border takes the state of most of its length, and a car near the coarse border line matches the
    rest of the track. Ties go to the first of them along the route. With neither, `DEFAULT_JURISDICTION`."""
    state = str(getattr(region, "state", "") or "").strip().lower()
    if state in JURISDICTIONS:
        return state
    if state in STATE_NAMES:
        return STATE_NAMES[state]
    votes: dict[str, int] = {}
    for lon, lat in lonlat:
        code = jurisdiction_at(float(lon), float(lat))
        votes[code] = votes.get(code, 0) + 1
    if not votes:
        return DEFAULT_JURISDICTION
    return max(votes, key=votes.get)  # dicts keep insertion order, so a tie goes to the first seen


# -- drawing -----------------------------------------------------------------------------------------------------

GLASS = (30, 36, 46)
DARK = (24, 24, 26)
TYRE = (28, 28, 30)
RIM = (170, 174, 178)
HEADLIGHT = (214, 218, 222)
TAIL = (176, 22, 26)
PLATE = (220, 220, 212)
HOUSING = (44, 46, 52)
SHUT_LINE_M = 0.006  # door shut lines
B_PILLAR = -0.25  # z of the pillar between the doors, on both bodies


def _font(size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=max(int(size), 4))


def _text(img: Image.Image, text: str, centre, cap_px: float, colour, *, max_len_px: float | None = None,
          len_per_cap: float = 1.0, stretch: float = 1.0, rotate: int = 0, bold: float = 0.07, outlines=()) -> None:
    """`text` with its ink `cap_px` tall, centred on `centre`, turned `rotate` degrees anticlockwise. The cells
    aren't square in metres, so `len_per_cap` is the pixels per metre along the text over those across it; the
    font's proportions are kept in metres (times `stretch`), at most `max_len_px` long. Bold by a stroke of `bold`
    of the size; `outlines` are keylines round it, widest first: (colour, extra stroke as a fraction of the size)."""
    size = 160
    font = _font(size)
    layers = [(tuple(c), int(size * (bold + w))) for c, w in outlines] + [(tuple(colour), int(size * bold))]
    widest = max(s for _, s in layers)
    left, top, right, bottom = font.getbbox(text, stroke_width=widest)
    _, ink_top, _, ink_bottom = font.getbbox(text, stroke_width=layers[-1][1])
    k = cap_px / (ink_bottom - ink_top)
    w, h = (right - left) * k * stretch * len_per_cap, (bottom - top) * k
    if max_len_px is not None and w > max_len_px:
        w, h = max_len_px, h * max_len_px / w
    for fill, stroke in layers:
        mask = Image.new("L", (right - left, bottom - top), 0)
        ImageDraw.Draw(mask).text((-left, -top), text, fill=255, font=font, stroke_width=stroke, stroke_fill=255)
        mask = mask.resize((max(int(round(w)), 1), max(int(round(h)), 1)), Image.Resampling.LANCZOS)
        if rotate:
            mask = mask.rotate(rotate, expand=True, resample=Image.Resampling.BICUBIC)
        x0, y0 = int(round(centre[0] - mask.width / 2)), int(round(centre[1] - mask.height / 2))
        img.paste(Image.new("RGB", mask.size, fill), (x0, y0), mask)


def _scale(view: View, axis: int = 0) -> float:
    """Canvas px per metre along the view's first (0) or second (1) axis."""
    p0 = view.px(0.0, 0.0)
    p1 = view.px(1.0, 0.0) if axis == 0 else view.px(0.0, 1.0)
    return abs(p1[axis] - p0[axis])


def _label(img: Image.Image, view: View, text: str, a: float, b: float, cap_m: float, colour, *,
           max_len_m: float | None = None, stretch: float = 1.0, outlines=(), rotate: int = 0,
           bold: float = 0.07) -> None:
    """Lettering on a view, in metres: `cap_m` tall, at most `max_len_m` long, centred on (a, b). Upright it runs
    along the view's first axis; turned 90 degrees, along its second (the bonnet's, read from in front)."""
    sa, sb = _scale(view, 0), _scale(view, 1)
    cap_s, len_s = (sb, sa) if rotate == 0 else (sa, sb)
    _text(img, text, view.px(a, b), cap_m * cap_s, colour, max_len_px=None if max_len_m is None else max_len_m * len_s,
          len_per_cap=len_s / cap_s, stretch=stretch, rotate=rotate, bold=bold,
          outlines=[(c, w * 0.6) for c, w in outlines])


def _crest(img: Image.Image, view: View, name: str, a: float, b: float, size_m: float, rotate: int = 0) -> None:
    """Crest `name`, `size_m` across, centred on (a, b) of a view."""
    x, y = view.px(a, b)
    draw_crest(img, name, x, y, size_m * _scale(view, 0), size_m * _scale(view, 1), rotate)


def _poly(d: ImageDraw.ImageDraw, view: View, pts, fill) -> None:
    d.polygon([view.px(a, b) for a, b in pts], fill=tuple(fill))


def _rect(d: ImageDraw.ImageDraw, view: View, a0, b0, a1, b1, fill) -> None:
    _poly(d, view, [(a0, b0), (a1, b0), (a1, b1), (a0, b1)], fill)


def _ellipse(d: ImageDraw.ImageDraw, view: View, a, b, ra, rb, fill) -> None:
    (x0, y0), (x1, y1) = view.px(a - ra, b - rb), view.px(a + ra, b + rb)
    d.ellipse([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=tuple(fill))


def _shade(colour, k: float) -> RGB:
    return tuple(int(round(min(c * k, 255))) for c in colour)


def _windows(body: Body) -> list[tuple[float, float]]:
    """The cabin's side outline (z, y): along the roof from the windscreen's foot to the rear window's, then back
    along the belt."""
    st = body.stations
    first = next(i for i, s in enumerate(st) if s.roof is not None)
    last = max(i for i, s in enumerate(st) if s.roof is not None)
    top = [(st[first - 1].z, st[first - 1].belt)] + [(s.z, s.roof) for s in st[first:last + 1]]
    top.append((st[last + 1].z, st[last + 1].belt))
    belt = [(s.z, s.belt) for s in st[first - 1:last + 2]]
    return top + belt[::-1][1:-1]


def _side_frame(body: Body):
    """Functions from a livery's fractions to the side's metres: along the car (0 front, 1 rear) and up its side
    (0 the sill, 1 the belt, at the cabin)."""
    zf, zr = body.stations[0].z, body.stations[-1].z
    mid = body.cabin
    return (lambda f: zf - f * (zf - zr)), (lambda f: mid.sill + f * (mid.belt - mid.sill))


def _square(d, view, a, b, half, deg, fill) -> None:
    t = np.radians(deg)
    corners = [(half * (np.cos(t) * x - np.sin(t) * y), half * (np.sin(t) * x + np.cos(t) * y))
               for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    _poly(d, view, [(a + x, b + y) for x, y in corners], fill)


def _draw_band(d: ImageDraw.ImageDraw, view: View, body: Body, livery: Livery) -> None:
    """The chequer along the side: squares in columns from the nose back, each column following the belt down
    where it drops. At the front a "scatter" breaks the squares up over the guard, an "enlarge" makes them bigger."""
    band = band_heights(body, livery)
    zf, zr = body.stations[0].z, body.stations[-1].z
    dark = livery.colours[0]
    if livery.pattern == "wedge":
        _draw_wedge(d, view, body, livery, band)
        return
    zone = body.axles[0] - body.wheel_r  # the front guard, from about the middle of the front wheel
    rng = np.random.default_rng(7)
    z, k = zf, 0
    while z > zr:
        front = livery.front and z - 0.03 > zone
        rows = livery.rows - 1 if front and livery.front == "enlarge" else livery.rows
        size = (band[1] - band[0]) / rows
        z1 = max(z - size, zr)
        zc = (z + z1) / 2
        y0, y1 = _band_at(float(body.along("sill", zc)), float(body.along("belt", zc)), band, False)
        cell = (y1 - y0) / rows
        for r in range(rows):
            top = y1 - r * cell
            colour = livery.colours[(k + r) % 2]
            if front and livery.front == "scatter":
                fade = (zc - zone) / (zf - zone)
                if colour == dark and rng.random() < 1.0 - 0.75 * fade:
                    _square(d, view, zc + rng.uniform(-0.2, 0.2) * cell, top - cell / 2 + rng.uniform(-0.25, 0.25)
                            * cell, cell * 0.45, rng.uniform(-35, 35), dark)
            else:
                _rect(d, view, z, top - cell, z1, top, colour)
        z, k = z1, k + 1


def _draw_wedge(d, view, body, livery, band) -> None:
    """Queensland's large squares in a wedge on the lower body: `wedge` rows behind the front wheel rising to more at
    the rear axle, its top cut on the slant, with a line along the cut."""
    b0, b1 = band
    q = (b1 - b0) / livery.rows
    z_start = body.axles[0] - body.wheel_r - 0.06
    z_full = body.axles[1]
    zr = body.stations[-1].z

    def top(z):
        rows = np.interp(z, [z_full, z_start], [livery.wedge[1], livery.wedge[0]])
        return b0 + q * rows

    z, k = z_start, 0
    while z > zr:
        z1 = max(z - q, zr)
        for r in range(livery.rows):
            _rect(d, view, z, b0 + r * q, z1, b0 + (r + 1) * q, livery.colours[(k + r + 1) % 2])
        z, k = z1, k + 1
    zs = np.linspace(z_start, zr, 40)
    cut = [(zz, top(zz)) for zz in zs]
    _poly(d, view, cut + [(zr, b1 + q), (z_start, b1 + q)], livery.paint)
    if livery.wedge_edge:
        t = 0.012
        _poly(d, view, [(zz, y + t) for zz, y in cut] + [(zz, y - t) for zz, y in cut[::-1]], livery.wedge_edge)


def _draw_stripes(d, view, body, livery) -> None:
    """Thin lines along the side, dropping with the band where the belt drops."""
    b1 = band_heights(body, livery)[1]
    _, height = _side_frame(body)
    zs = np.linspace(body.stations[0].z, body.stations[-1].z, 60)
    anchored = livery.pattern == "wedge"
    drop = np.zeros_like(zs) if anchored else b1 - np.minimum(b1, body.along("belt", zs) - BAND_BELOW_BELT)
    for colour, frac, thick in livery.stripes:
        y = height(frac) - drop
        _poly(d, view, list(zip(zs, y + thick / 2)) + list(zip(zs[::-1], (y - thick / 2)[::-1])), colour)


def _draw_side(img: Image.Image, d: ImageDraw.ImageDraw, view: View, body: Body, livery: Livery) -> None:
    st = body.stations
    b0, b1 = band_heights(body, livery)
    mid = body.cabin
    along, height = _side_frame(body)

    _draw_band(d, view, body, livery)
    _draw_stripes(d, view, body, livery)
    for text, za, width in livery.panels:
        z = along(za)
        _rect(d, view, z + width / 2, b0, z - width / 2, b1, livery.paint)
        _label(img, view, text, z, (b0 + b1) / 2, 0.5 * (b1 - b0), livery.colours[0], max_len_m=0.85 * width,
               bold=0.09)
    # Shut lines (under the arches, which cut them), the lower trim, the arches and the windows.
    line = _shade(livery.paint, 0.55)
    front_door = st[next(i for i, s in enumerate(st) if s.roof is not None) - 1].z + 0.12
    rear_door = body.axles[1] + 0.22
    for z in (front_door, B_PILLAR - 0.05, B_PILLAR + 0.05, rear_door):
        _rect(d, view, z - SHUT_LINE_M, mid.sill + 0.04, z + SHUT_LINE_M, mid.belt - 0.01, line)
    _rect(d, view, rear_door, mid.belt - 0.012, front_door, mid.belt, line)
    _poly(d, view, [(s.z, s.sill) for s in st] + [(s.z, s.sill + 0.05) for s in st[::-1]], DARK)
    for z in body.axles:
        _ellipse(d, view, z, body.wheel_r, body.wheel_r + 0.065, body.wheel_r + 0.065, DARK)
    outline = shapely.Polygon(_windows(body))
    glass = outline.buffer(-0.035, join_style="mitre")
    _poly(d, view, list(outline.exterior.coords), _shade(livery.paint, 0.86))  # the frame round the glass
    _poly(d, view, list(glass.exterior.coords), GLASS)
    _rect(d, view, B_PILLAR - 0.045, mid.belt, B_PILLAR + 0.045, body.roof, DARK)
    for z in (B_PILLAR - 0.20, rear_door + 0.30):  # handles, near each door's rear edge
        _rect(d, view, z, mid.belt - 0.11, z + 0.14, mid.belt - 0.085, DARK)
    # Lights at the corners: a dark housing round the lens.
    for colour, (a, b, k) in ((HEADLIGHT, (st[0], st[1], 1.0)), (TAIL, (st[-1], st[-2], -1.0))):
        outer = [(a.z + k * 0.02, a.belt - 0.14), (a.z + k * 0.02, a.belt - 0.015), (b.z - k * 0.26, b.belt - 0.02),
                 (b.z - k * 0.18, b.belt - 0.13)]
        _poly(d, view, outer, HOUSING)
        _poly(d, view, list(shapely.Polygon(outer).buffer(-0.015, join_style="mitre").exterior.coords), colour)
    # Lettering and the crest.
    za, frac, cap, longest = livery.word_side
    _label(img, view, livery.word, along(za), height(frac), cap, livery.word_colour, max_len_m=longest,
           stretch=livery.word_stretch, outlines=livery.word_outline)
    for text, za, frac, cap in livery.extra:
        _label(img, view, text, along(za), height(frac), cap, livery.word_colour, bold=0.09)
    if livery.crest and livery.crest_side:
        za, frac, size = livery.crest_side
        _crest(img, view, livery.crest, along(za), height(frac), size)


def _draw_top(img: Image.Image, d: ImageDraw.ImageDraw, view: View, body: Body, livery: Livery) -> None:
    st = body.stations
    first = next(i for i, s in enumerate(st) if s.roof is not None)
    last = max(i for i, s in enumerate(st) if s.roof is not None)
    cowl, front = st[first - 1].z, st[0].z
    _rect(d, view, st[last + 1].z, -1.2, cowl, 1.2, GLASS)  # under the glass cells, should a filter reach
    _rect(d, view, st[first].z, -st[first].roof_half_w, st[last].z, st[first].roof_half_w, livery.paint)
    length = front - cowl
    # Read from in front of the car: the tops of the letters (and the crest) towards the windscreen.
    if livery.word_bonnet:
        _label(img, view, livery.word, cowl + 0.80 * length, 0.0, 0.18, livery.word_bonnet, max_len_m=1.2,
               stretch=livery.word_stretch, outlines=livery.word_outline, rotate=90)
    if livery.crest and livery.crest_bonnet:
        _crest(img, view, livery.crest, cowl + 0.38 * length, 0.0, livery.crest_bonnet, rotate=90)


def _draw_end(img: Image.Image, d: ImageDraw.ImageDraw, view: View, body: Body, livery: Livery, end: str) -> None:
    s = body.stations[0] if end == "front" else body.stations[-1]
    hw, lo, hi = s.half_w, s.sill, s.belt
    y0, y1 = _band_at(lo, hi, band_heights(body, livery), livery.pattern == "wedge")
    if end == "rear" and livery.band_rear:
        q = (y1 - y0) / livery.rows
        for k, x in enumerate(np.arange(-hw, hw, q)):
            for r in range(livery.rows):
                _rect(d, view, x, y1 - (r + 1) * q, min(x + q, hw), y1 - r * q, livery.colours[(k + r) % 2])
        for colour, frac, thick in livery.stripes:
            mid = body.cabin
            y = mid.sill + frac * (mid.belt - mid.sill) - (band_heights(body, livery)[1] - y1)
            _rect(d, view, -hw, y - thick / 2, hw, y + thick / 2, colour)
    _rect(d, view, -hw, lo, hw, lo + 0.06, DARK)
    lamp = HEADLIGHT if end == "front" else TAIL
    top = hi - 0.03
    for side in (1, -1):
        _poly(d, view, [(side * 0.40, top - 0.11), (side * 0.40, top), (side * (hw - 0.01), top + 0.01),
                        (side * (hw - 0.01), top - 0.12)], HOUSING)
        _poly(d, view, [(side * 0.415, top - 0.095), (side * 0.415, top - 0.015), (side * (hw - 0.025), top - 0.005),
                        (side * (hw - 0.025), top - 0.105)], lamp)
    plate_y = lo + (0.08 if end == "front" else 0.12)
    if end == "front":
        _rect(d, view, -0.50, lo + 0.065, 0.50, lo + 0.17, DARK)  # the lower intake, behind the plate
        _rect(d, view, -0.34, lo + 0.20, 0.34, top - 0.04, DARK)  # the grille
        bar = lo + 0.20 + 0.45 * (top - 0.24 - lo)
        _rect(d, view, -0.34, bar, 0.34, bar + 0.02, _shade(livery.paint, 0.7))
        if livery.corner_flash:
            for side in (1, -1):
                _poly(d, view, [(side * hw, lo + 0.07), (side * (hw - 0.22), lo + 0.07), (side * hw, lo + 0.22)],
                      livery.corner_flash)
    _rect(d, view, -0.19, plate_y - 0.01, 0.19, plate_y + 0.12, DARK)
    _rect(d, view, -0.18, plate_y, 0.18, plate_y + 0.11, PLATE)
    if end == "rear" and livery.word_rear:
        cap = 0.075
        y = max(y1 + 0.03 + cap / 2, hi - 0.10) if livery.band_rear else (plate_y + 0.12 + top - 0.12) / 2
        _label(img, view, livery.word, 0.0, y, cap, livery.word_colour, max_len_m=0.70, stretch=livery.word_stretch,
               outlines=livery.word_outline)


def lens_tint(colour) -> RGB:
    """A lens as it looks switched off: its colour darkened and greyed, so in the headlights it reads as tinted
    plastic, not as a lit lamp."""
    return tuple(int(round(0.55 * c + 0.45 * 60)) for c in colour)


def _draw_cells(img: Image.Image, d: ImageDraw.ImageDraw, livery: Livery, body: Body) -> None:
    def box(cell):
        return [v * SUPERSAMPLE for v in CELLS[cell]]

    d.rectangle(box("dark"), fill=DARK)
    d.rectangle(box("paint"), fill=livery.paint)
    x0, y0, x1, y1 = box("glass")
    for k in range(int(y1 - y0)):
        d.line([(x0, y0 + k), (x1, y0 + k)], fill=_shade(GLASS, 1.3 - 0.4 * k / (y1 - y0)))
    x0, y0, x1, y1 = box("tread")
    d.rectangle([x0, y0, x1, y1], fill=TYRE)
    for k in range(8):
        y = y0 + (k + 0.5) * (y1 - y0) / 8
        d.line([(x0, y), (x1, y)], fill=(46, 46, 48), width=2 * SUPERSAMPLE)
    x0, y0, x1, y1 = box("wheel")
    cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=TYRE)
    d.ellipse([cx - 0.66 * r, cy - 0.66 * r, cx + 0.66 * r, cy + 0.66 * r], fill=RIM)
    for k in range(5):
        a0 = k * 72 + 16
        d.pieslice([cx - 0.58 * r, cy - 0.58 * r, cx + 0.58 * r, cy + 0.58 * r], a0, a0 + 40, fill=(60, 62, 66))
    d.ellipse([cx - 0.16 * r, cy - 0.16 * r, cx + 0.16 * r, cy + 0.16 * r], fill=(120, 124, 128))
    # The light bar, switched off: seven tinted lenses with a pale reflector line, a clear one in the middle. From in
    # front the car's right half is on the left.
    left, right = (lens_tint(c) for c in livery.lightbar)
    clear = (136, 140, 146)
    w_m, _, h_m = body.bar
    for cell, order, k in (("bar_front", (right, left), 1.0), ("bar_back", (left, right), 1.0),
                           ("bar_top", (right, left), 0.8)):
        x0, y0, x1, y1 = box(cell)
        d.rectangle([x0, y0, x1, y1], fill=(36, 36, 40))
        w, segs = x1 - x0, 7
        for i in range(segs):
            colour = order[0] if i < segs // 2 else order[1] if i > segs // 2 else clear
            lx0, lx1 = x0 + w * i / segs + 2 * SUPERSAMPLE, x0 + w * (i + 1) / segs - 2 * SUPERSAMPLE
            ly0, ly1 = y0 + 3 * SUPERSAMPLE, y1 - 4 * SUPERSAMPLE
            d.rectangle([lx0, ly0, lx1, ly1], fill=_shade(colour, k))
            d.rectangle([lx0, ly0 + 0.2 * (ly1 - ly0), lx1, ly0 + 0.3 * (ly1 - ly0)], fill=_shade(colour, 1.2 * k))
        if livery.bar_word and cell != "bar_top":
            # The strip is w_m by h_m, so its pixels are wider in metres along than up.
            per_m_along, per_m_up = w / w_m, (y1 - y0) / h_m
            _text(img, livery.bar_word, ((x0 + x1) / 2, (y0 + y1) / 2 - SUPERSAMPLE), 0.4 * (y1 - y0), (20, 30, 70),
                  max_len_px=0.8 * w / segs, len_per_cap=per_m_along / per_m_up)


@lru_cache(maxsize=None)
def atlas(code: str, body: str) -> Image.Image:
    """The jurisdiction's texture for `body`: its sides, top, ends, wheel, glass and light bar (see CELLS)."""
    return draw_atlas(LIVERIES[code], BODIES[body])


def draw_atlas(livery: Livery, shape: Body) -> Image.Image:
    """The texture for any livery on `shape` (`atlas` keeps each jurisdiction's)."""
    canvas = ATLAS * SUPERSAMPLE
    img = Image.new("RGB", (canvas, canvas), livery.paint)
    _draw_cells(img, ImageDraw.Draw(img), livery, shape)
    # Each view in a drawing of its own cell's size, so nothing spills into the next cell.
    for name, draw in (("side_r", _draw_side), ("side_l", _draw_side), ("top", _draw_top),
                       ("front", lambda *a: _draw_end(*a, "front")), ("rear", lambda *a: _draw_end(*a, "rear"))):
        x0, y0, x1, y1 = (v * SUPERSAMPLE for v in CELLS[name])
        cell = Image.new("RGB", (x1 - x0, y1 - y0), livery.paint)
        draw(cell, ImageDraw.Draw(cell), views(shape)[name], shape, livery)
        img.paste(cell, (x0, y0))
    return img.resize((ATLAS, ATLAS), Image.Resampling.LANCZOS)


# -- crests ------------------------------------------------------------------------------------------------------
# Our own simplified renditions, drawn from shapes: each badge's general form (its outline, crown, wreath, rings and
# emblem), drawn big and bold, so that at about 60 px on the door it still reads as that service's from the roadside.
# Nothing is copied or traced. Coordinates are x right and y up, from -1 to 1 across the crest's box; the badge sits
# a little low (CREST_Y) to leave room for the crown above it.

GOLD = (204, 164, 52)
GOLD_LIGHT = (236, 204, 104)
GOLD_DARK = (120, 86, 20)
SILVER = (206, 210, 214)
GREEN = (30, 100, 48)
CROWN_RED = (176, 30, 40)
INK = (16, 24, 40)
CREST_Y = -0.16
CROWN_W = 0.44

# Silhouettes, facing the viewer's left.
LION = ((-0.95, 0.22), (-0.88, 0.42), (-0.70, 0.60), (-0.45, 0.48), (-0.35, 0.32), (0.45, 0.30), (0.68, 0.36),
        (0.80, 0.62), (0.92, 0.70), (0.95, 0.60), (0.86, 0.50), (0.80, 0.28), (0.76, 0.10), (0.80, -0.55),
        (0.64, -0.55), (0.60, -0.08), (0.05, -0.10), (-0.18, -0.55), (-0.32, -0.55), (-0.30, -0.05), (-0.45, -0.02),
        (-0.72, -0.20), (-0.80, -0.12), (-0.55, 0.08), (-0.62, 0.15), (-0.85, 0.12))
_WING = ((0.00, 0.58), (0.09, 0.52), (0.07, 0.40), (0.22, 0.44), (0.45, 0.70), (0.72, 0.90), (0.74, 0.72), (0.90, 0.64),
         (0.72, 0.50), (0.84, 0.38), (0.60, 0.28), (0.64, 0.14), (0.30, 0.10), (0.16, -0.25), (0.30, -0.64),
         (0.00, -0.54))
BIRD = _WING + tuple((-x, y) for x, y in _WING[-2:0:-1])  # wings spread, as on a badge
BEAK = ((-0.08, 0.53), (-0.24, 0.47), (-0.07, 0.43))
SWAN = ((0.85, 0.15), (0.55, 0.12), (0.30, 0.18), (-0.10, 0.12), (-0.35, 0.05), (-0.38, 0.30), (-0.42, 0.50),
        (-0.50, 0.66), (-0.58, 0.74), (-0.68, 0.73), (-0.85, 0.62), (-0.68, 0.64), (-0.58, 0.60), (-0.50, 0.48),
        (-0.47, 0.30), (-0.52, 0.10), (-0.62, -0.05), (-0.55, -0.22), (-0.30, -0.32), (0.20, -0.32), (0.55, -0.22),
        (0.75, 0.0))
KANGAROO = ((-0.42, 0.62), (-0.30, 0.72), (-0.24, 0.90), (-0.16, 0.72), (-0.08, 0.64), (0.06, 0.42), (0.18, 0.08),
            (0.26, -0.30), (0.52, -0.66), (0.72, -0.86), (0.50, -0.82), (0.16, -0.62), (0.10, -0.76), (-0.26, -0.84),
            (-0.26, -0.75), (-0.02, -0.66), (-0.06, -0.40), (-0.12, -0.18), (-0.14, 0.10), (-0.28, 0.16),
            (-0.38, 0.10), (-0.32, 0.24), (-0.22, 0.32), (-0.25, 0.50), (-0.40, 0.55))


class _Pen:
    """A crest's drawing: on a square transparent layer, at several times the size it's pasted at."""

    def __init__(self, size_px: float):
        self.n = max(int(round(size_px)), 16)
        self.img = Image.new("RGBA", (self.n, self.n), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def xy(self, x, y):
        return self.n / 2 * (1 + x), self.n / 2 * (1 - y)

    def width(self, t: float) -> int:
        return max(int(round(t * self.n / 2)), 1)

    def poly(self, pts, fill, outline=None, width=0.0):
        self.d.polygon([self.xy(*p) for p in pts], fill=tuple(fill), outline=tuple(outline) if outline else None,
                       width=self.width(width) if outline else 1)

    def circle(self, x, y, r, fill, outline=None, width=0.0):
        (x0, y0), (x1, y1) = self.xy(x - r, y + r), self.xy(x + r, y - r)
        self.d.ellipse([x0, y0, x1, y1], fill=tuple(fill), outline=tuple(outline) if outline else None,
                       width=self.width(width) if outline else 1)

    def line(self, pts, fill, width):
        self.d.line([self.xy(*p) for p in pts], fill=tuple(fill), width=self.width(width))


def _shape(pts, x, y, scale, mirror=False):
    return [(x + (-px if mirror else px) * scale, y + py * scale) for px, py in pts]


def _star(x, y, r_out, r_in, n, rot=90.0):
    return [(x + (r_out if k % 2 == 0 else r_in) * np.cos(np.radians(rot + k * 180.0 / n)),
             y + (r_out if k % 2 == 0 else r_in) * np.sin(np.radians(rot + k * 180.0 / n))) for k in range(2 * n)]


def _cross_pattee(x, y, r):
    """An eight-pointed cross: four arms widening to notched ends."""
    arm = ((0.18, -0.10), (1.0, -0.56), (0.76, 0.0), (1.0, 0.56), (0.18, 0.10))
    pts = []
    for deg in (90, 180, 270, 0):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        pts += [(x + r * (t * c - w * s), y + r * (t * s + w * c)) for t, w in arm]
    return pts


def _shield(x, y, w, h):
    return [(x - w / 2, y + h / 2), (x + w / 2, y + h / 2), (x + w / 2, y - 0.1 * h), (x, y - h / 2),
            (x - w / 2, y - 0.1 * h)]


def _crown(p: _Pen, w=CROWN_W, gold=GOLD, cap=CROWN_RED, jewel=None, x=0.0, y=None):
    """A simplified crown `w` wide: cap, arches, band, orb and cross. By default at the top of the box."""
    y = 0.98 - 0.98 * w if y is None else y
    jewel = jewel or cap
    edge = _shade(gold, 0.55)
    p.circle(x, y + 0.30 * w, 0.40 * w, cap)
    p.line([(x - 0.5 * w, y + 0.1 * w), (x - 0.42 * w, y + 0.48 * w), (x - 0.2 * w, y + 0.66 * w), (x, y + 0.58 * w),
            (x + 0.2 * w, y + 0.66 * w), (x + 0.42 * w, y + 0.48 * w), (x + 0.5 * w, y + 0.1 * w)], gold, 0.10 * w)
    p.line([(x, y + 0.1 * w), (x, y + 0.6 * w)], gold, 0.09 * w)
    p.poly([(x - 0.5 * w, y - 0.14 * w), (x + 0.5 * w, y - 0.14 * w), (x + 0.5 * w, y + 0.14 * w),
            (x - 0.5 * w, y + 0.14 * w)], gold, edge, 0.02 * w)
    for k in (-0.3, 0.0, 0.3):
        p.circle(x + k * w, y, 0.065 * w, jewel)
    p.circle(x, y + 0.72 * w, 0.10 * w, gold)
    p.poly([(x - 0.04 * w, y + 0.78 * w), (x + 0.04 * w, y + 0.78 * w), (x + 0.04 * w, y + 0.98 * w),
            (x - 0.04 * w, y + 0.98 * w)], gold)
    p.poly([(x - 0.10 * w, y + 0.85 * w), (x + 0.10 * w, y + 0.85 * w), (x + 0.10 * w, y + 0.92 * w),
            (x - 0.10 * w, y + 0.92 * w)], gold)


def _wreath(p: _Pen, r, colour=GREEN, leaves=6, start=-76.0, end=64.0, berries=None, x=0.0, y=CREST_Y):
    """Two sprays of a few big leaves round a circle, from near the bottom up each side; `berries` dots on them."""
    edge = _shade(colour, 0.55)
    for side in (1, -1):
        for i, a in enumerate(np.linspace(start, end, leaves)):
            deg = a if side == 1 else 180 - a
            cx, cy = x + r * np.cos(np.radians(deg)), y + r * np.sin(np.radians(deg))
            t = np.radians(deg + 90 - side * 22)
            ell = [(cx + 0.25 * r * np.cos(u) * np.cos(t) - 0.11 * r * np.sin(u) * np.sin(t),
                    cy + 0.25 * r * np.cos(u) * np.sin(t) + 0.11 * r * np.sin(u) * np.cos(t))
                   for u in np.linspace(0, 2 * np.pi, 14, endpoint=False)]
            p.poly(ell, colour, edge, 0.012)
            if berries and i % 2 == 0:
                bx, by = x + 1.17 * r * np.cos(np.radians(deg)), y + 1.17 * r * np.sin(np.radians(deg))
                p.circle(bx, by, 0.085 * r, berries)


def _scroll(p: _Pen, y, w, h, fill, edge=INK):
    """A plain ribbon with folded ends."""
    back = _shade(fill, 0.6)
    for side in (1, -1):
        p.poly([(side * w / 2, y + h * 0.3), (side * (w / 2 + 0.12), y + h * 0.1), (side * (w / 2 + 0.08), y - h * 0.2),
                (side * (w / 2 + 0.12), y - h * 0.6), (side * w * 0.42, y - h * 0.6)], back, edge, 0.01)
    p.poly([(-w / 2, y + h / 2), (w / 2, y + h / 2), (w / 2, y - h / 2), (-w / 2, y - h / 2)], fill, edge, 0.012)


def _rose(p: _Pen, x, y, r):
    for k in range(5):
        a = np.radians(90 + 72 * k)
        p.circle(x + 0.55 * r * np.cos(a), y + 0.55 * r * np.sin(a), 0.5 * r, (226, 124, 176))
    p.circle(x, y, 0.3 * r, (120, 30, 60))


def _crest_tas(p: _Pen):
    """A rayed gold star under a crown, round a blue disc with a gold lion walking, its forepaw raised."""
    c = CREST_Y
    p.poly(_star(0, c, 0.84, 0.62, 16), GOLD, GOLD_DARK, 0.03)
    p.circle(0, c, 0.50, GOLD_DARK)
    p.circle(0, c, 0.45, (11, 79, 151))
    p.poly(_shape(LION, 0, c - 0.02, 0.40), GOLD_LIGHT, GOLD_DARK, 0.015)
    _crown(p)


def _crest_nsw(p: _Pen):
    """A laurel wreath round a white ring under a crown; in a light blue disc a brown eagle, wings spread, with the
    state's white shield and red cross on its breast; a white scroll below."""
    c = CREST_Y
    blue = (0, 48, 135)
    _wreath(p, 0.66)
    p.circle(0, c, 0.56, WHITE, blue, 0.035)
    p.circle(0, c, 0.43, (140, 190, 228), blue, 0.02)
    p.poly(_shape(BIRD, 0, c - 0.06, 0.44), (110, 70, 34), (50, 30, 12), 0.015)
    p.poly(_shape(BEAK, 0, c - 0.06, 0.44), GOLD)
    sx, sy = 0, c - 0.04
    p.poly(_shield(sx, sy, 0.20, 0.22), WHITE, INK, 0.012)
    p.poly([(sx - 0.025, sy + 0.11), (sx + 0.025, sy + 0.11), (sx + 0.025, sy - 0.09), (sx - 0.025, sy - 0.09)],
           CROWN_RED)
    p.poly([(sx - 0.10, sy + 0.045), (sx + 0.10, sy + 0.045), (sx + 0.10, sy - 0.005), (sx - 0.10, sy - 0.005)],
           CROWN_RED)
    _crown(p)
    _scroll(p, -0.88, 0.86, 0.16, WHITE, blue)


def _crest_vic(p: _Pen, ink=(45, 44, 142)):
    """In one colour, as on the cars: a wreath, a five-pointed star point down, a crown, and the Southern Cross in the
    middle; a scroll below."""
    c = CREST_Y + 0.04
    _wreath(p, 0.70, ink, leaves=6, start=-82, end=70, y=c)
    star = _star(0, c, 0.66, 0.31, 5, rot=-90)
    p.poly(star, ink)
    for x, y in star[::2]:
        p.line([(0, c), (x, y)], WHITE, 0.025)
    p.circle(0, c, 0.27, WHITE)
    p.circle(0, c, 0.21, ink)
    for x, y, r in ((0, 0.12, 0.04), (0.09, 0.02, 0.035), (0, -0.11, 0.04), (-0.09, 0.0, 0.035)):
        p.circle(x, c + y, r, WHITE)
    _crown(p, gold=ink, cap=ink, jewel=WHITE)
    _scroll(p, -0.88, 0.80, 0.15, ink, ink)


def _crest_qld(p: _Pen):
    """A white eight-pointed cross edged in red on a laurel wreath, under a crown; a blue ring round a navy disc with
    a small gold crown; a blue scroll."""
    c = CREST_Y
    blue = (14, 48, 136)
    _wreath(p, 0.70)
    p.poly(_cross_pattee(0, c, 0.76), CROWN_RED)
    p.poly(_cross_pattee(0, c, 0.66), WHITE)
    p.circle(0, c, 0.34, blue, WHITE, 0.02)
    p.circle(0, c, 0.25, (14, 24, 64))
    _crown(p, w=0.26, y=c - 0.07)
    _crown(p)
    _scroll(p, -0.88, 0.90, 0.15, blue)


def _crest_sa(p: _Pen):
    """A golden wattle wreath under a crown; a light blue ring round a white disc with a black and white piping shrike,
    its wings spread."""
    c = CREST_Y
    _wreath(p, 0.70, berries=(242, 205, 40))
    p.circle(0, c, 0.60, (80, 150, 210), INK, 0.025)
    p.circle(0, c, 0.46, WHITE)
    s = 0.46
    p.poly(_shape(BIRD, 0, c - 0.06, s), INK)
    for mirror in (False, True):
        p.poly(_shape(((0.24, 0.44), (0.46, 0.68), (0.56, 0.58), (0.32, 0.34)), 0, c - 0.06, s, mirror), WHITE)
    p.circle(0, c - 0.06 + s * 0.46, 0.045, WHITE)
    p.poly(_shape(BEAK, 0, c - 0.06, s), (150, 150, 150))
    _crown(p)


def _crest_wa(p: _Pen, ink=(36, 52, 138)):
    """In one colour, as on the cars: an eight-pointed star under a crown, a black swan in the middle, a banner
    below."""
    c = CREST_Y
    p.poly(_star(0, c, 0.84, 0.52, 8), ink)
    p.circle(0, c, 0.50, WHITE)
    p.circle(0, c, 0.45, ink)
    p.circle(0, c, 0.36, WHITE)
    p.poly(_shape(SWAN, 0, c - 0.02, 0.33), ink)
    _scroll(p, -0.84, 0.70, 0.15, ink, ink)
    _crown(p, gold=ink, cap=ink, jewel=WHITE)


def _crest_nt(p: _Pen):
    """A laurel wreath with two desert roses, under a crown; an ochre kangaroo standing in the middle; a blue scroll
    below."""
    c = CREST_Y + 0.04
    _wreath(p, 0.66, start=-66, end=60, y=c)
    for deg in (210, 330):
        _rose(p, 0.66 * np.cos(np.radians(deg)), c + 0.66 * np.sin(np.radians(deg)), 0.16)
    p.poly(_shape(KANGAROO, 0.04, c + 0.04, 0.58), (224, 122, 42), (90, 44, 12), 0.022)
    _crown(p)
    _scroll(p, -0.84, 0.90, 0.17, (31, 63, 143))


def _crest_act(p: _Pen):
    """The AFP's: an eight-pointed star under a crown, a navy ring round the Commonwealth arms' shield, a navy scroll.
    Drawn bolder than the badge: the star a darker silver edged in navy and the ring thick, since a pale silver star on
    white paint read only as its gold centre from 6 m (in game, 11 Nov 2026)."""
    c = CREST_Y
    navy = (15, 26, 51)
    p.poly(_star(0, c, 0.86, 0.58, 8), (128, 136, 148), navy, 0.05)
    p.circle(0, c, 0.58, navy)
    p.circle(0, c, 0.36, WHITE)
    p.poly(_shield(0, c, 0.34, 0.40), GOLD, navy, 0.025)
    p.poly([(-0.17, c + 0.20), (0, c + 0.20), (0, c + 0.02), (-0.17, c + 0.02)], CROWN_RED)
    p.poly([(0, c + 0.02), (0.17, c + 0.02), (0.17, c - 0.04), (0, c - 0.15)], (40, 70, 150))
    _crown(p)
    _scroll(p, -0.86, 0.80, 0.16, navy)


CRESTS = {"tas": _crest_tas, "nsw": _crest_nsw, "vic": _crest_vic, "qld": _crest_qld, "sa": _crest_sa,
          "wa": _crest_wa, "nt": _crest_nt, "act": _crest_act}


def draw_crest(img: Image.Image, name: str, cx: float, cy: float, w: float, h: float, rotate: int = 0) -> None:
    """Crest `name` into a `w` by `h` px box centred on (cx, cy) in `img` (a crest is square in metres, and the cells'
    pixels aren't), turned `rotate` degrees anticlockwise."""
    pen = _Pen(4 * max(w, h))
    CRESTS[name](pen)
    layer = pen.img
    if rotate:
        layer = layer.rotate(rotate, resample=Image.Resampling.BICUBIC)
    small = layer.resize((max(int(round(w)), 1), max(int(round(h)), 1)), Image.Resampling.LANCZOS)
    img.paste(small.convert("RGB"), (int(round(cx - small.width / 2)), int(round(cy - small.height / 2))), small)


# -- the liveries ------------------------------------------------------------------------------------------------
# From the services' releases and dated photos, 2019-2026 (THIRD_PARTY_NOTICES.md lists the sources). Where the
# sources leave it open (which side the light bar's red is on, what crosses the rear), a common Australian layout is
# drawn and the docs say so. Crests are drawn 0.30-0.40 m across, larger than on most real cars, so they read from
# the road; Tasmania's moves above the chequer on the rear door to find the room. Where the chequer runs along the
# shoulder (NSW, Victoria, SA, WA, the ACT) the crest goes below it, there being only the windows above.

LIVERIES = {
    # Silver SUVs: a two-row chequer low on the doors with a lime line on top; "Police" above it on the front doors;
    # the crest on the rear door; a POLICE box in the bar (new liveries from April 2019).
    "tas": Livery("tas", "Tasmania Police", "suv", ((11, 79, 151), WHITE), 2, (0.36, 0.53), "Police", (11, 79, 151),
                  (0.43, 0.68, 0.12, 0.9), paint=SILVER_PAINT,
                  stripes=(((212, 220, 0), 0.555, 0.025),), crest="tas", crest_side=(0.655, 0.79, 0.30),
                  bar_word="POLICE"),
    # White SUVs: a two-row chequer along the shoulder, dropping to the headlights; "Police" large and low across both
    # doors, the crest ahead of it; "Police" on the bonnet; "000 EMERGENCY" on the rear quarter.
    "nsw": Livery("nsw", "NSW Police Force", "suv", ((0, 48, 135), WHITE), 2, (0.79, 0.95), "Police", (0, 48, 135),
                  (0.55, 0.32, 0.17, 1.3), word_bonnet=(0, 48, 135), extra=(("000 EMERGENCY", 0.845, 0.72, 0.085),),
                  crest="nsw", crest_side=(0.365, 0.33, 0.36)),
    # White SUVs: a three-row chequer at door-handle height with "131 444" and "000" panels set into it; "Police" low
    # round the B-pillar with the crest, in one colour, ahead of it; "Police" on the bonnet; blue on the driver's side.
    "vic": Livery("vic", "Victoria Police", "suv", ((45, 44, 142), WHITE), 3, (0.70, 0.93), "Police", (45, 44, 142),
                  (0.55, 0.34, 0.15, 1.2), word_bonnet=(45, 44, 142),
                  panels=(("131 444", 0.64, 0.40), ("000", 0.83, 0.24)),
                  crest="vic", crest_side=(0.37, 0.35, 0.33)),
    # White Camrys (RAV4s coming), in the 2023 high-visibility design: large squares in a wedge on the lower body,
    # rising to the rear, an orange-red line along its top and another along the sill; "POLICE" in keylined capitals
    # on the front door; the crest large on the rear door.
    "qld": Livery("qld", "Queensland Police Service", "sedan", ((14, 48, 136), WHITE), 4, (0.12, 0.68), "POLICE",
                  (14, 48, 136), (0.44, 0.63, 0.115, 0.95), pattern="wedge", wedge=(2, 4), wedge_edge=(255, 75, 31),
                  stripes=(((255, 75, 31), 0.085, 0.02),), word_stretch=1.0,
                  word_outline=(((255, 75, 31), 0.10), (WHITE, 0.05)), band_rear=False,
                  crest="qld", crest_side=(0.645, 0.65, 0.38)),
    # White SUVs: a three-row black chequer between yellow lines, breaking up into scattered squares over the front
    # guard; "POLICE" in tall black capitals keylined yellow across both doors, and on the bonnet; red on the driver's
    # side of the bar, with a POLICE box.
    "sa": Livery("sa", "South Australia Police", "suv", ((28, 29, 34), WHITE), 3, (0.62, 0.84), "POLICE", (20, 20, 24),
                 (0.585, 0.35, 0.17, 1.15), front="scatter", stripes=(((242, 215, 0), 0.865, 0.012),
                                                                       ((242, 215, 0), 0.595, 0.03)),
                 word_stretch=0.82, word_outline=(((242, 215, 0), 0.07),), word_bonnet=(20, 20, 24),
                 crest="sa", crest_side=(0.345, 0.36, 0.34), lightbar=(BLUE, RED), bar_word="POLICE"),
    # White SUVs: a two-row chequer along the shoulder with a thin orange-red line above; "Police" on the doors, the
    # crest in one colour near it.
    "wa": Livery("wa", "Western Australia Police Force", "suv", ((36, 52, 138), WHITE), 2, (0.76, 0.92), "Police",
                 (36, 52, 138), (0.55, 0.40, 0.14, 1.0), stripes=(((232, 80, 42), 0.945, 0.008),),
                 crest="wa", crest_side=(0.375, 0.42, 0.32)),
    # White 4WDs: a two-row chequer along the shoulder between ochre lines; "Police" on the front doors below it;
    # "Police" and the crest on the bonnet.
    "nt": Livery("nt", "Northern Territory Police", "suv", ((31, 63, 143), WHITE), 2, (0.72, 0.89), "Police",
                 (31, 63, 143), (0.44, 0.55, 0.13, 0.9), stripes=(((224, 122, 42), 0.915, 0.012),
                                                                   ((224, 122, 42), 0.695, 0.012)),
                 word_bonnet=(31, 63, 143), crest="nt", crest_bonnet=0.50),
    # ACT Policing (AFP), white cars since May 2019 (a Subaru Liberty then, an Outback in the 2022 photos, so the SUV):
    # a three-row near-black chequer between lime lines, bigger squares on the front guard, lime flashes on the bumper;
    # "Police" low on the rear door, ghosted on the bonnet; the AFP's crest on the front door, below the chequer at the
    # other states' height (on the real cars it's smaller and lower); a POLICE sign in the bar.
    "act": Livery("act", "ACT Policing (Australian Federal Police)", "suv", ((15, 26, 51), WHITE), 3, (0.68, 0.90),
                  "Police", (15, 26, 51), (0.645, 0.32, 0.13, 0.85), front="enlarge",
                  stripes=(((200, 220, 0), 0.925, 0.012), ((200, 220, 0), 0.655, 0.012)), word_bonnet=(160, 164, 170),
                  corner_flash=(200, 220, 0), crest="act", crest_side=(0.427, 0.38, 0.40), bar_word="POLICE"),
}
