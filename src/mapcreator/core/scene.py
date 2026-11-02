"""A neutral 3D scene: meshes, materials and dummies in AC axes, before KN5 encoding.

AC axes: Y up, metres. Region coordinates (easting, northing, height) map to
x = E - E0, y = H - H0, z = Z_SIGN * (N - N0), around a local origin, so
float32 keeps millimetre precision. Z_SIGN and the triangle winding are the
two conventions settled empirically on the first in-game load (milestone M0/M3),
so each lives in one place.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

Z_SIGN = -1.0  # z = -(N - N0): looking down, x east and z south
MAX_VERTICES = 65_535  # KN5 indices are u16


@dataclass(frozen=True)
class LocalFrame:
    e0: float
    n0: float
    h0: float

    def to_ac(self, e, n, h) -> np.ndarray:
        e, n, h = (np.asarray(v, dtype=float) for v in (e, n, h))
        return np.stack([e - self.e0, h - self.h0, Z_SIGN * (n - self.n0)], axis=-1)

    def from_ac(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = np.asarray(p, dtype=float)
        return p[..., 0] + self.e0, p[..., 2] / Z_SIGN + self.n0, p[..., 1] + self.h0


@dataclass
class Mesh:
    name: str
    material: str
    positions: np.ndarray  # (n, 3) float32
    normals: np.ndarray  # (n, 3) float32, unit
    uvs: np.ndarray  # (n, 2) float32
    triangles: np.ndarray  # (m, 3) int, wound so the face normal points up (+Y) when seen from above

    def validate(self) -> None:
        n = len(self.positions)
        if n > MAX_VERTICES:
            raise ValueError(f"{self.name}: {n} vertices exceeds {MAX_VERTICES}")
        if self.triangles.size and (self.triangles.min() < 0 or self.triangles.max() >= n):
            raise ValueError(f"{self.name}: triangle index out of range")
        for arr, w in ((self.normals, 3), (self.uvs, 2)):
            if arr.shape != (n, w):
                raise ValueError(f"{self.name}: attribute shape {arr.shape} != ({n}, {w})")
        if not np.isfinite(self.positions).all():
            raise ValueError(f"{self.name}: non-finite positions")


@dataclass
class Dummy:
    name: str
    position: list[float]  # AC axes
    forward: list[float]  # unit, direction of travel
    up: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])


@dataclass
class Material:
    name: str
    texture: str  # key into Scene.textures
    uv_scale: float = 1.0
    kind: str = "opaque"  # opaque | alpha_test


@dataclass
class Scene:
    frame: LocalFrame
    meshes: list[Mesh] = field(default_factory=list)
    materials: dict[str, Material] = field(default_factory=dict)
    dummies: list[Dummy] = field(default_factory=list)
    textures: dict[str, str] = field(default_factory=dict)  # name -> file path

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for i, m in enumerate(self.meshes):
            m.validate()
            arrays[f"{i}_pos"] = m.positions.astype(np.float32)
            arrays[f"{i}_nrm"] = m.normals.astype(np.float32)
            arrays[f"{i}_uv"] = m.uvs.astype(np.float32)
            arrays[f"{i}_tri"] = m.triangles.astype(np.uint32)
        np.savez_compressed(folder / "scene.npz", **arrays)
        meta = {
            "frame": asdict(self.frame),
            "meshes": [{"name": m.name, "material": m.material} for m in self.meshes],
            "materials": {k: asdict(v) for k, v in self.materials.items()},
            "dummies": [asdict(d) for d in self.dummies],
            "textures": self.textures,
        }
        (folder / "scene.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "Scene":
        meta = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
        data = np.load(folder / "scene.npz")
        meshes = [
            Mesh(m["name"], m["material"], data[f"{i}_pos"], data[f"{i}_nrm"], data[f"{i}_uv"],
                 data[f"{i}_tri"].astype(np.int64))
            for i, m in enumerate(meta["meshes"])
        ]
        return cls(
            frame=LocalFrame(**meta["frame"]),
            meshes=meshes,
            materials={k: Material(**v) for k, v in meta["materials"].items()},
            dummies=[Dummy(**d) for d in meta["dummies"]],
            textures=meta["textures"],
        )

    def write_obj(self, path: Path) -> None:
        """Wavefront OBJ (Y up) for eyeballing in any 3D viewer."""
        with path.open("w", encoding="utf-8") as f:
            base = 1
            for m in self.meshes:
                f.write(f"o {m.name}\n")
                np.savetxt(f, m.positions, fmt="v %.3f %.3f %.3f")
                np.savetxt(f, m.uvs, fmt="vt %.4f %.4f")
                np.savetxt(f, m.normals, fmt="vn %.4f %.4f %.4f")
                t = m.triangles + base
                np.savetxt(f, np.repeat(t, 3, axis=1), fmt="f %d/%d/%d %d/%d/%d %d/%d/%d")
                base += len(m.positions)
