"""3D trees for Custom Shaders Patch Tree FX: procedural models, the `.bin` model format and the tree list.

Tree FX (CSP 0.1.79 and later) draws instanced 3D trees with LODs and distant billboards, listed in a text
file under the track's `extension/` folder. Vanilla AC ignores it, so the KN5 keeps its tree cards for
players without CSP. With CSP both are drawn unless `ext_config.ini` hides the cards (see
`package.ext_config_ini`).

The models are generated here (eucalypts and a wattle), so there's no licensing question. CSP ships no tree
models of its own.

Model format, re-implemented from the description of Content Manager's tree converter (the reference
writer, MS-PL); no code copied:
- A `.bin` is a zip holding `tree.ini`, one `.mesh` per LOD and the textures (DDS) under their base names.
- `.mesh`, little-endian:
  - int32 version 1, uint8 0, int32 texture count;
  - per texture: uint32 xxHash32 of the file (seed 0, never 0), int32 length, UTF-8 base name;
  - per material sub-mesh until the end: int32 diffuse index, int32 normal-map index, int32 index count
    and uint16 indices, int32 vertex count and 44-byte vertices;
  - an optional trailer (int32 -1, int32 4, float32): the trunk's half-width at its base, normalised.
- Vertex:
  - float32 position, normalised to the model's box: x / half-width, y / height, z / half-width;
  - float32 normal, float32 uv (v = 0 at the top of the image);
  - tangent as 3 unsigned bytes (t * 127.5 + 127.5) and a 0 byte;
  - two float16 (a second uv, 0);
  - bytes 255, 255, 255, 0 (AO, AO, wetness, extra).
- Triangles are stored wound so that cross(b - a, c - a) points away from the vertex normal, as the
  converter writes them. CSP draws trees two-sided and flips the normal on back faces.
- The shader tells leaves from bark by colour: (green - red) * MULT + OFFSET, clamped. So leaf textures
  are greener than red, and bark isn't (see `sprites`).

List (`trees.txt`) line: `tree: <model>.bin; pos=x, y, z; angle = <deg>; size = <mult>; width = <mult>`.
- Position is in AC coordinates, at the tree's base.
- `size` scales the model's own height (tree.ini HEIGHT); `width` scales it horizontally only.
"""

from __future__ import annotations

import io
import struct
import zipfile
from dataclasses import dataclass, field

import numpy as np

from mapcreator.core import package, sprites, textures
from mapcreator.core.kn5 import tangents

UP = np.array([0.0, 1.0, 0.0])
ZIP_DATE = (2020, 1, 1, 0, 0, 0)  # fixed, so the same model gives the same bytes
BARK_TILE_M = (1.2, 2.4)  # bark texture repeat, around and along a branch

VERTEX = np.dtype([("pos", "<f4", 3), ("normal", "<f4", 3), ("uv", "<f4", 2), ("tangent", "u1", 4),
                   ("uv2", "<f2", 2), ("extra", "u1", 4)])
assert VERTEX.itemsize == 44

MATERIALS = {"bark": ("bark.dds", "bark_nm.dds"), "leaves": ("leaves.dds", "leaves_nm.dds")}

# Per LOD: polygon sides for trunk, limbs and twigs (0 = left out), share of the leaf cards kept,
# and how much bigger the kept cards get so the crown keeps its fullness.
LOD_SIDES = ((10, 6, 4), (7, 4, 3), (5, 3, 0))
LOD_CARDS = (1.0, 0.5, 0.22)
LOD_CARD_SCALE = (1.0, 1.3, 1.85)


# -- xxHash32 ------------------------------------------------------------------------

_P1, _P2, _P3, _P4, _P5 = 2654435761, 2246822519, 3266489917, 668265263, 374761393
_M = 0xFFFFFFFF


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & _M


def xxh32(data: bytes, seed: int = 0) -> int:
    """xxHash32, as Tree FX uses to identify a model's textures."""
    n = len(data)
    i = 0
    if n >= 16:
        v = [(seed + _P1 + _P2) & _M, (seed + _P2) & _M, seed & _M, (seed - _P1) & _M]
        stripes = n // 16
        words = struct.unpack_from(f"<{stripes * 4}I", data)
        v1, v2, v3, v4 = v
        for k in range(0, stripes * 4, 4):
            v1 = (_rotl((v1 + words[k] * _P2) & _M, 13) * _P1) & _M
            v2 = (_rotl((v2 + words[k + 1] * _P2) & _M, 13) * _P1) & _M
            v3 = (_rotl((v3 + words[k + 2] * _P2) & _M, 13) * _P1) & _M
            v4 = (_rotl((v4 + words[k + 3] * _P2) & _M, 13) * _P1) & _M
        h = (_rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)) & _M
        i = stripes * 16
    else:
        h = (seed + _P5) & _M
    h = (h + n) & _M
    while i + 4 <= n:
        h = (_rotl((h + struct.unpack_from("<I", data, i)[0] * _P3) & _M, 17) * _P4) & _M
        i += 4
    while i < n:
        h = (_rotl((h + data[i] * _P5) & _M, 11) * _P1) & _M
        i += 1
    h ^= h >> 15
    h = (h * _P2) & _M
    h ^= h >> 13
    h = (h * _P3) & _M
    h ^= h >> 16
    return h


# -- geometry ------------------------------------------------------------------------

@dataclass
class Geometry:
    """One material's triangles in model metres: Y up, the base of the trunk at the origin. Triangles are
    wound counter-clockwise seen from the side their normals face, as everywhere else in mapcreator."""
    pos: np.ndarray
    nrm: np.ndarray
    uv: np.ndarray
    tri: np.ndarray


@dataclass
class _Parts:
    pos: list = field(default_factory=list)
    nrm: list = field(default_factory=list)
    uv: list = field(default_factory=list)
    tri: list = field(default_factory=list)
    count: int = 0

    def add(self, pos, nrm, uv, tri) -> None:
        self.pos.append(pos)
        self.nrm.append(nrm)
        self.uv.append(uv)
        self.tri.append(np.asarray(tri) + self.count)
        self.count += len(pos)

    def geometry(self) -> Geometry:
        pos, nrm = np.vstack(self.pos), np.vstack(self.nrm)
        nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
        tri = np.vstack(self.tri).astype(np.int64)
        p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
        flip = (np.cross(p1 - p0, p2 - p0) * nrm[tri].mean(axis=1)).sum(axis=1) < 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        return Geometry(pos, nrm, np.vstack(self.uv), tri)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def tube(points: np.ndarray, radii: np.ndarray, sides: int, parts: _Parts) -> None:
    """A branch as rings of `sides` vertices along `points`, with bark UVs that tile in metres."""
    t = np.gradient(points, axis=0)
    t = _unit(t)
    a = np.cross(t[0], UP if abs(t[0][1]) < 0.95 else np.array([1.0, 0.0, 0.0]))
    a = a / np.linalg.norm(a)
    frames = []
    for ti in t:  # parallel transport, so the rings don't twist
        a = a - ti * np.dot(a, ti)
        a = a / np.linalg.norm(a)
        frames.append((a, np.cross(ti, a)))
    theta = np.linspace(0, 2 * np.pi, sides + 1)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    u_repeat = max(1, round(2 * np.pi * radii[0] / BARK_TILE_M[0]))
    pos, nrm, uv = [], [], []
    for p, r, (fa, fb), si in zip(points, radii, frames, s):
        d = np.cos(theta)[:, None] * fa + np.sin(theta)[:, None] * fb
        pos.append(p + r * d)
        nrm.append(d)
        uv.append(np.column_stack([theta / (2 * np.pi) * u_repeat, np.full(sides + 1, si / BARK_TILE_M[1])]))
    ring = sides + 1
    tri = []
    for i in range(len(points) - 1):
        for j in range(sides):
            v00, v01, v10, v11 = i * ring + j, i * ring + j + 1, (i + 1) * ring + j, (i + 1) * ring + j + 1
            tri += [[v00, v10, v11], [v00, v11, v01]]
    parts.add(np.vstack(pos), np.vstack(nrm), np.vstack(uv), np.array(tri))


@dataclass(frozen=True)
class Kind:
    height: float  # metres, roughly the model's own height
    trunk: float  # share of the height where the limbs start
    limbs: tuple[int, int]
    limb_angle: tuple[float, float]  # degrees from vertical
    limb_length: float  # share of the height
    radius: float  # trunk radius at the base, metres
    leaf_cells: tuple[int, ...]  # cells of `sprites.leaf_sprays`
    leaf_size: float  # leaf card size, metres
    cards: int  # leaf cards per cluster at LOD 0
    twigs: int  # side branches per limb


KINDS = {
    "euc_tall": Kind(26.0, 0.42, (5, 7), (18, 38), 0.42, 0.42, (0, 1, 2), 2.6, 5, 3),
    "euc_spread": Kind(18.0, 0.33, (4, 7), (32, 58), 0.46, 0.38, (0, 1, 2), 2.3, 5, 3),
    "euc_young": Kind(12.0, 0.25, (5, 8), (15, 32), 0.32, 0.18, (0, 1, 2), 1.8, 4, 2),
    "wattle": Kind(7.5, 0.15, (5, 8), (38, 66), 0.5, 0.14, (sprites.WATTLE_CELL,), 1.6, 6, 3),
}
# Which models suit trees of each height: (up to metres, {model: share}).
HEIGHT_BANDS = (
    (9.0, {"wattle": 0.6, "euc_young": 0.4}),
    (15.0, {"euc_young": 0.5, "euc_spread": 0.35, "wattle": 0.15}),
    (22.0, {"euc_spread": 0.55, "euc_tall": 0.35, "euc_young": 0.1}),
    (np.inf, {"euc_tall": 0.7, "euc_spread": 0.3}),
)


def _path(start, direction, length, n, rng, upcurve):
    pts, d = [np.asarray(start, float)], _unit(direction)
    for _ in range(n):
        d = _unit(d + UP * upcurve + rng.normal(0, 0.06, 3))
        pts.append(pts[-1] + d * length / n)
    return np.array(pts)


def _at(points, radii, f):
    """Point and radius a share `f` of the way along a polyline (by vertex count, near enough)."""
    x = f * (len(points) - 1)
    i = min(int(x), len(points) - 2)
    w = x - i
    return points[i] * (1 - w) + points[i + 1] * w, radii[i] * (1 - w) + radii[i + 1] * w


def skeleton(kind: Kind, rng: np.random.Generator):
    """Branches [(points, radii, depth)] and leaf clusters [(centre, outward)] in model metres."""
    h = kind.height
    top = 0.82 * h
    n = 10
    ys = np.r_[-0.4, np.linspace(0.0, top, n)]
    lean = rng.normal(0, 0.025, 2)
    wig = np.cumsum(rng.normal(0, 0.04 * h / n, (len(ys), 2)), axis=0) * 0.3
    trunk = np.column_stack([lean[0] * ys + wig[:, 0], ys, lean[1] * ys + wig[:, 1]])
    trunk[:, [0, 2]] -= trunk[1, [0, 2]]  # base centred on the origin
    r_trunk = kind.radius * (1 - 0.75 * np.clip(ys / top, 0, 1))
    r_trunk[:2] *= 1.25  # root flare
    branches = [(trunk, r_trunk, 0)]
    clusters = [(trunk[-1] + rng.normal(0, 0.4, 3) * [1, 0.3, 1], _unit(UP + rng.normal(0, 0.3, 3)))
                for _ in range(3)]

    k = int(rng.integers(*kind.limbs))
    heights = np.sort(rng.uniform(kind.trunk * h, 0.78 * h, k))
    az0 = rng.uniform(0, 2 * np.pi)
    for i, y in enumerate(heights):
        az = az0 + i * 2.39996 + rng.normal(0, 0.3)  # golden angle, so limbs spread all round
        el = np.radians(rng.uniform(*kind.limb_angle))
        direction = np.array([np.sin(el) * np.cos(az), np.cos(el), np.sin(el) * np.sin(az)])
        f = np.clip((y + 0.4) / (top + 0.4), 0, 1)
        start, r_at = _at(trunk, r_trunk, f)
        rise = (y - kind.trunk * h) / max(h * (0.78 - kind.trunk), 1e-6)
        length = kind.limb_length * h * rng.uniform(0.75, 1.1) * (1 - 0.35 * rise)
        pts = _path(start, direction, length, 5, rng, 0.08)
        radii = r_at * 0.6 * (1 - 0.8 * np.linspace(0, 1, len(pts)))
        branches.append((pts, radii, 1))
        for f_tip in (0.75, 1.0):
            c, _ = _at(pts, radii, f_tip)
            clusters.append((c, _unit(np.r_[c[0], 0.3 * h, c[2]] - np.r_[0, 0, 0])))
        for _ in range(kind.twigs + int(rng.integers(0, 2))):
            f = rng.uniform(0.35, 0.9)
            p, r = _at(pts, radii, f)
            az2 = az + rng.choice([-1, 1]) * rng.uniform(0.5, 1.2)
            el2 = np.radians(rng.uniform(35, 70))
            d2 = np.array([np.sin(el2) * np.cos(az2), np.cos(el2), np.sin(el2) * np.sin(az2)])
            twig = _path(p, d2, length * rng.uniform(0.35, 0.55), 3, rng, 0.05)
            branches.append((twig, r * 0.55 * (1 - 0.8 * np.linspace(0, 1, len(twig))), 2))
            for f_leaf in (0.45, 0.75, 1.0):
                c, _ = _at(twig, np.zeros(len(twig)), f_leaf)
                clusters.append((c, _unit(np.r_[c[0], 0.3 * h, c[2]])))
    return branches, clusters


def leaf_cards(clusters, kind: Kind, lod: int, rng: np.random.Generator, parts: _Parts) -> None:
    """Leaf-spray cards around each cluster, facing outwards, hanging a little below their twig."""
    per = max(1, round(kind.cards * LOD_CARDS[lod]))
    for c, out in clusters:
        for _ in range(per):
            s = kind.leaf_size * LOD_CARD_SCALE[lod] * rng.uniform(0.8, 1.2)
            normal = _unit(out + rng.normal(0, 0.8, 3))
            up = UP - normal * normal[1]
            up = _unit(up) if np.linalg.norm(up) > 0.2 else _unit(np.cross(normal, [1.0, 0.0, 0.0]))
            tilt = rng.uniform(-0.4, 0.4)
            right = np.cross(up, normal)
            up, right = up * np.cos(tilt) + right * np.sin(tilt), right * np.cos(tilt) - up * np.sin(tilt)
            centre = c + rng.normal(0, 0.35 * kind.leaf_size, 3) * [1, 0.6, 1] - up * s * 0.25
            centre[1] = max(centre[1], s * 0.6)
            u0, v0, u1, v1 = sprites.leaf_uv_cell(int(rng.choice(kind.leaf_cells)))
            hs = s / 2
            pos = np.array([centre - right * hs + up * hs, centre + right * hs + up * hs,
                            centre + right * hs - up * hs, centre - right * hs - up * hs])
            parts.add(pos, np.tile(normal, (4, 1)), np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]),
                      np.array([[0, 1, 2], [0, 2, 3]]))


@dataclass
class TreeModel:
    name: str
    lods: list[dict[str, Geometry]]  # material -> geometry, densest LOD first

    @property
    def box(self) -> tuple[float, float]:
        """(height, half-width) over every LOD: what positions are normalised by."""
        pos = np.vstack([g.pos for lod in self.lods for g in lod.values()])
        return float(pos[:, 1].max()), float(np.abs(pos[:, [0, 2]]).max())

    @property
    def triangles(self) -> list[int]:
        return [sum(len(g.tri) for g in lod.values()) for lod in self.lods]


def build_model(name: str, seed: int) -> TreeModel:
    kind = KINDS[name]
    rng = np.random.default_rng(seed)
    branches, clusters = skeleton(kind, rng)
    lods = []
    for lod, sides in enumerate(LOD_SIDES):
        bark, leaves = _Parts(), _Parts()
        for pts, radii, depth in branches:
            if sides[depth]:
                step = 2 if lod == 2 and depth == 0 else 1
                keep = np.r_[np.arange(0, len(pts) - 1, step), len(pts) - 1]
                tube(pts[keep], radii[keep], sides[depth], bark)
        leaf_cards(clusters, kind, lod, np.random.default_rng(seed * 10 + lod), leaves)
        lods.append({"bark": bark.geometry(), "leaves": leaves.geometry()})
    return TreeModel(name, lods)


def library(seed: int = 5) -> dict[str, TreeModel]:
    return {name: build_model(name, seed + i) for i, name in enumerate(KINDS)}


def texture_files() -> dict[str, bytes]:
    return {
        "bark.dds": textures.to_dds(sprites.bark(), "DXT1"),
        "bark_nm.dds": textures.to_dds(sprites.bark_normal(), "DXT1"),
        "leaves.dds": textures.to_dds(sprites.leaf_sprays(), "DXT5"),
        "leaves_nm.dds": textures.to_dds(sprites.flat_normal(), "DXT1"),
    }


# -- writing --------------------------------------------------------------------------

def mesh_file(lod: dict[str, Geometry], tex_names: list[str], tex_hashes: list[int],
              height: float, half_w: float) -> bytes:
    out = io.BytesIO()
    out.write(struct.pack("<iBi", 1, 0, len(tex_names)))
    for name, h in zip(tex_names, tex_hashes):
        b = name.encode("utf-8")
        out.write(struct.pack("<Ii", h, len(b)) + b)
    scale = np.array([half_w, height, half_w])
    base = []
    for material, g in lod.items():
        if not len(g.tri):
            continue
        if len(g.pos) > 0xFFFF:
            raise ValueError(f"{material}: {len(g.pos)} vertices is over the 16-bit index limit")
        diffuse, normal = MATERIALS[material]
        out.write(struct.pack("<ii", tex_names.index(diffuse), tex_names.index(normal)))
        idx = g.tri[:, [0, 2, 1]].astype("<u2").ravel()  # stored the converter's way round
        out.write(struct.pack("<i", len(idx)) + idx.tobytes())
        v = np.zeros(len(g.pos), VERTEX)
        v["pos"] = g.pos / scale
        v["normal"] = g.nrm
        v["uv"] = g.uv
        t = tangents(g.pos, g.nrm, g.uv, g.tri)
        v["tangent"][:, :3] = np.clip(t * 127.5 + 127.5, 0, 255).astype(np.uint8)
        v["extra"] = (255, 255, 255, 0)
        out.write(struct.pack("<i", len(v)) + v.tobytes())
        m = v["pos"].astype(float)
        base.append(m[(m[:, 1] < 0.002) & (np.abs(m[:, 0]) < 0.2) & (np.abs(m[:, 2]) < 0.2)])
    base = np.vstack(base) if base else np.zeros((0, 3))
    if len(base):
        width = min(base[:, 0].max(), base[:, 2].max(), -base[:, 0].min(), -base[:, 2].min())
        out.write(struct.pack("<iif", -1, 4, width))
    return out.getvalue()


def tree_ini(height: float, half_w: float, mesh_names: list[str]) -> str:
    sections = [
        ("BASIC", {"HEIGHT": f"{height:.3f}", "WIDTH": f"{2 * half_w:.3f}", "CACHE_RESOLUTION": "256,512",
                   "VERTICAL_OFFSET": "0"}),
        ("SHADING", {"SPECULAR": "0.6", "SUBSCATTERING": "1.0", "REFLECTIVITY": "0.6", "BRIGHTNESS": "1.0",
                     "USE_AO_CHANNEL": "0"}),
        ("NORMALS", {"BOOST": "0.9", "SPHERE": "1.0"}),
        ("LEAVES", {"MULT": "20.0", "OFFSET": "0.0"}),
        ("FAKE_SHADOW", {"SIZE": "1.0", "OPACITY": "1.0"}),
    ]
    for i, name in enumerate(mesh_names):
        sections.append((f"LOD_{i}", {"MODEL": name, "DISTANCE": f"{height * 10 * (i + 1) / len(mesh_names):.1f}"}))
    return package.ini(sections)


def model_bin(model: TreeModel, tex: dict[str, bytes]) -> bytes:
    """The model as a Tree FX `.bin`: tree.ini, one .mesh per LOD, and the textures."""
    height, half_w = model.box
    names = sorted(tex)
    hashes = [xxh32(tex[n]) or 1 for n in names]
    meshes = [f"lod{i}.mesh" for i in range(len(model.lods))]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def put(name, data):
            z.writestr(zipfile.ZipInfo(name, ZIP_DATE), data, compress_type=zipfile.ZIP_DEFLATED)

        put("tree.ini", tree_ini(height, half_w, meshes).encode("utf-8"))
        for mesh, lod in zip(meshes, model.lods):
            put(mesh, mesh_file(lod, names, hashes, height, half_w))
        for n in names:
            put(n, tex[n])
    return buf.getvalue()


# -- placing --------------------------------------------------------------------------

def assign(height: np.ndarray, names: list[str], rng: np.random.Generator) -> np.ndarray:
    """A model (index into `names`) for each tree, by its height."""
    height = np.asarray(height, dtype=float)
    out = np.zeros(len(height), dtype=np.int64)
    lower = -np.inf
    for upper, shares in HEIGHT_BANDS:
        band = (height > lower) & (height <= upper)
        options = [n for n in shares if n in names]
        p = np.array([shares[n] for n in options])
        out[band] = [names.index(options[k]) for k in rng.choice(len(options), int(band.sum()), p=p / p.sum())]
        lower = upper
    return out


def trees_txt(names: list[str], model: np.ndarray, pos: np.ndarray, angle: np.ndarray, size: np.ndarray,
              width: np.ndarray, header: str) -> str:
    lines = [f"; {line}" for line in header.splitlines()]
    # Every tree gives its own size, width and angle; no extra randomness on top.
    lines += ["configure: size variance = 1, 1", "configure: width variance = 1, 1",
              "configure: angle variance = 0, 0"]
    for m, p, a, s, w in zip(model, pos, angle, size, width):
        lines.append(f"tree: {names[m]}.bin; pos={p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}; angle = {a:.0f}; "
                     f"size = {s:.3f}; width = {w:.3f}")
    return "\r\n".join(lines) + "\r\n"
