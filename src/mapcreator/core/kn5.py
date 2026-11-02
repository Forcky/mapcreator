"""Write (and read back) Assetto Corsa KN5 model files for static tracks.

The layout below is implemented from a written format description, cross-
checked against several open-source readers and writers (see
THIRD_PARTY_NOTICES.md under "Format references"). No code was copied.

    "sc6969"  int32 version  [int32 extra if version > 5]
    textures:  int32 count; { int32 active, str name, uint32 len, bytes }
    materials: int32 count; { str name, str shader, u8 blend, u8 alphaTested, int32 depthMode,
                              int32 nProps { str name, f32 A, f32 B[2], f32 C[3], f32 D[4] },
                              int32 nMaps  { str sampler, int32 slot, str texture } }
    nodes (depth-first): int32 class, str name, int32 childCount, u8 active, then
        class 1 (dummy): f32 matrix[16], row-major rows = X, Y, Z axes, translation
        class 2 (mesh):  u8 castShadows, u8 visible, u8 transparent,
                         uint32 nVerts { f32 pos[3], normal[3], uv[2], tangent[3] },
                         uint32 nIdx u16[nIdx], uint32 materialId, uint32 layer,
                         f32 lodIn, f32 lodOut, f32 sphereCentre[3], f32 radius, u8 renderable

Little-endian; strings are an int32 byte length then UTF-8 (kept ASCII here).
Mesh nodes never have children: that crashes the game.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"sc6969"
MAX_VERTICES = 65_535
VERTEX_DTYPE = np.dtype([("pos", "<f4", 3), ("normal", "<f4", 3), ("uv", "<f4", 2), ("tangent", "<f4", 3)])
assert VERTEX_DTYPE.itemsize == 44


@dataclass
class Kn5Material:
    name: str
    shader: str = "ksPerPixel"
    properties: dict[str, float | tuple[float, ...]] = field(default_factory=dict)
    textures: dict[str, str] = field(default_factory=dict)  # sampler -> texture name
    blend: int = 0  # 0 opaque, 1 alpha blend, 2 alpha to coverage
    alpha_tested: bool = False
    depth_mode: int = 0


@dataclass
class Kn5Mesh:
    name: str
    material: int
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    triangles: np.ndarray
    tangents: np.ndarray | None = None
    cast_shadows: bool = True
    renderable: bool = True


@dataclass
class Kn5Dummy:
    name: str
    matrix: np.ndarray  # 4x4, rows: X axis, Y axis, Z axis, translation


def dummy_matrix(position, forward, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Orthonormal frame: Z = forward (made perpendicular to up), Y = up, X = Y x Z (det +1)."""
    y = np.asarray(up, dtype=float)
    y /= np.linalg.norm(y)
    z = np.asarray(forward, dtype=float)
    z = z - y * (z @ y)
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = x, y, z, position
    return m


def tangents(pos: np.ndarray, nrm: np.ndarray, uv: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Per-vertex tangent = dP/du, Gram-Schmidt against the normal, with a stable fallback."""
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    t0, t1, t2 = uv[tri[:, 0]], uv[tri[:, 1]], uv[tri[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    ok = np.abs(det) > 1e-12
    r = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    ft = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    acc = np.zeros_like(pos)
    for k in range(3):
        np.add.at(acc, tri[:, k], ft)
    acc -= nrm * (acc * nrm).sum(axis=1, keepdims=True)
    length = np.linalg.norm(acc, axis=1, keepdims=True)
    fallback = np.cross(nrm, np.array([0.0, 1.0, 0.0]))
    fb_len = np.linalg.norm(fallback, axis=1, keepdims=True)
    fallback = np.where(fb_len > 1e-6, fallback / np.maximum(fb_len, 1e-12), np.array([1.0, 0.0, 0.0]))
    return np.where(length > 1e-9, acc / np.maximum(length, 1e-12), fallback)


class _W:
    def __init__(self) -> None:
        self.b = io.BytesIO()

    def raw(self, data: bytes) -> None:
        self.b.write(data)

    def i32(self, v: int) -> None:
        self.b.write(struct.pack("<i", v))

    def u32(self, v: int) -> None:
        self.b.write(struct.pack("<I", v))

    def u8(self, v: int) -> None:
        self.b.write(struct.pack("<B", v))

    def f32(self, *v: float) -> None:
        self.b.write(struct.pack(f"<{len(v)}f", *v))

    def string(self, s: str) -> None:
        data = s.encode("ascii")
        self.i32(len(data))
        self.b.write(data)


def _prop_values(v) -> tuple[float, tuple, tuple, tuple]:
    if isinstance(v, (int, float)):
        f = float(v)
        return f, (f, f), (f, f, f), (f, f, f, f)
    v = tuple(float(x) for x in v)
    a, b, c, d = 0.0, (0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
    if len(v) == 2:
        b = v
    elif len(v) == 3:
        c = v
    elif len(v) == 4:
        d = v
    else:
        raise ValueError(f"property vector of length {len(v)}")
    return a, b, c, d


def _check_name(name: str) -> None:
    if not name or not name.isascii():
        raise ValueError(f"KN5 names must be non-empty ASCII: {name!r}")


def write(path: Path, *, root: str, textures: dict[str, bytes], materials: list[Kn5Material],
          meshes: list[Kn5Mesh], dummies: list[Kn5Dummy], version: int = 6) -> None:
    for group, names in (("texture", list(textures)), ("material", [m.name for m in materials]),
                         ("node", [m.name for m in meshes] + [d.name for d in dummies] + [root])):
        for n in names:
            _check_name(n)
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate {group} names")
    for mat in materials:
        for tex in mat.textures.values():
            if tex not in textures:
                raise ValueError(f"material {mat.name} uses missing texture {tex}")

    w = _W()
    w.raw(MAGIC)
    w.i32(version)
    if version > 5:
        w.i32(0)

    w.i32(len(textures))
    for name, data in textures.items():
        w.i32(1)
        w.string(name)
        w.u32(len(data))
        w.raw(data)

    w.i32(len(materials))
    for mat in materials:
        w.string(mat.name)
        w.string(mat.shader)
        w.u8(mat.blend)
        w.u8(1 if mat.alpha_tested else 0)
        w.i32(mat.depth_mode)
        w.i32(len(mat.properties))
        for pname, val in mat.properties.items():
            a, b, c, d = _prop_values(val)
            w.string(pname)
            w.f32(a, *b, *c, *d)
        w.i32(len(mat.textures))
        for slot, (sampler, tex) in enumerate(mat.textures.items()):
            w.string(sampler)
            w.i32(slot)
            w.string(tex)

    # Root: identity dummy with every mesh and dummy as a flat, childless child.
    w.i32(1)
    w.string(root)
    w.i32(len(meshes) + len(dummies))
    w.u8(1)
    w.f32(*np.eye(4, dtype=np.float32).ravel())

    for m in meshes:
        n = len(m.positions)
        if n > MAX_VERTICES:
            raise ValueError(f"{m.name}: {n} vertices > {MAX_VERTICES}")
        if not 0 <= m.material < len(materials):
            raise ValueError(f"{m.name}: material index {m.material} out of range")
        tri = np.asarray(m.triangles, dtype=np.int64)
        if tri.size == 0 or tri.min() < 0 or tri.max() >= n:
            raise ValueError(f"{m.name}: bad triangle indices")
        verts = np.empty(n, dtype=VERTEX_DTYPE)
        verts["pos"] = m.positions
        verts["normal"] = m.normals
        verts["uv"] = m.uvs
        verts["tangent"] = m.tangents if m.tangents is not None else tangents(
            m.positions.astype(float), m.normals.astype(float), m.uvs.astype(float), tri)
        if not np.isfinite(verts["pos"]).all():
            raise ValueError(f"{m.name}: non-finite positions")
        lo, hi = m.positions.min(axis=0), m.positions.max(axis=0)
        centre = (lo + hi) / 2
        radius = float(np.linalg.norm(m.positions - centre, axis=1).max())

        w.i32(2)
        w.string(m.name)
        w.i32(0)
        w.u8(1)
        w.u8(1 if m.cast_shadows else 0)
        w.u8(1)  # visible
        w.u8(0)  # transparent
        w.u32(n)
        w.raw(verts.tobytes())
        idx = tri.astype("<u2").ravel()
        w.u32(len(idx))
        w.raw(idx.tobytes())
        w.u32(m.material)
        w.u32(0)  # layer
        w.f32(0.0, 0.0)  # lodIn, lodOut: 0/0 = always drawn
        w.f32(*centre, radius)
        w.u8(1 if m.renderable else 0)

    for d in dummies:
        w.i32(1)
        w.string(d.name)
        w.i32(0)
        w.u8(1)
        w.f32(*np.asarray(d.matrix, dtype=np.float32).ravel())

    path.write_bytes(w.b.getvalue())


# -- reader (validation and tests) --------------------------------------------

class _R:
    def __init__(self, data: bytes) -> None:
        self.d = data
        self.o = 0

    def take(self, n: int) -> bytes:
        if self.o + n > len(self.d):
            raise ValueError("unexpected end of file")
        b = self.d[self.o:self.o + n]
        self.o += n
        return b

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u8(self) -> int:
        return self.take(1)[0]

    def f32(self, n: int = 1):
        v = struct.unpack(f"<{n}f", self.take(4 * n))
        return v[0] if n == 1 else v

    def string(self) -> str:
        return self.take(self.i32()).decode("utf-8")


def read(path: Path) -> dict:
    r = _R(path.read_bytes())
    if r.take(6) != MAGIC:
        raise ValueError("not a KN5 file")
    version = r.i32()
    if version > 5:
        r.i32()
    textures = {}
    for _ in range(r.i32()):
        r.i32()
        name = r.string()
        textures[name] = r.take(r.u32())
    materials = []
    for _ in range(r.i32()):
        mat = {"name": r.string(), "shader": r.string(), "blend": r.u8(), "alpha_tested": r.u8(),
               "depth": r.i32(), "properties": {}, "textures": {}}
        for _ in range(r.i32()):
            pname = r.string()
            mat["properties"][pname] = r.f32(10)
        for _ in range(r.i32()):
            sampler, _slot, tex = r.string(), r.i32(), r.string()
            mat["textures"][sampler] = tex
        materials.append(mat)

    nodes = []

    def node(depth: int) -> None:
        cls, name, children, _active = r.i32(), r.string(), r.i32(), r.u8()
        item = {"class": cls, "name": name, "children": children, "depth": depth}
        if cls == 1:
            item["matrix"] = np.array(r.f32(16), dtype=float).reshape(4, 4)
        elif cls == 2:
            item["cast_shadows"], item["visible"], item["transparent"] = r.u8(), r.u8(), r.u8()
            n = r.u32()
            item["vertices"] = np.frombuffer(r.take(n * 44), dtype=VERTEX_DTYPE).copy()
            ni = r.u32()
            item["indices"] = np.frombuffer(r.take(ni * 2), dtype="<u2").copy()
            item["material"], item["layer"] = r.u32(), r.u32()
            item["lod"] = r.f32(2)
            item["sphere"] = r.f32(4)
            item["renderable"] = r.u8()
        else:
            raise ValueError(f"unknown node class {cls}")
        nodes.append(item)
        for _ in range(children):
            node(depth + 1)

    node(0)
    if r.o != len(r.d):
        raise ValueError(f"{len(r.d) - r.o} trailing bytes after the node tree")
    return {"version": version, "textures": textures, "materials": materials, "nodes": nodes}
