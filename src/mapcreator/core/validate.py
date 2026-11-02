"""Static checks on a finished track folder: everything we can verify without the game.

Each rule comes from a known cause of load failures or crashes (see
docs/ac-format.md). The in-game checks (does it load, are the wheels on the
ground, does the AI finish) live in the Windows test harness.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mapcreator.core import ai_line, kn5
from mapcreator.core.package import SURFACE_KEYS

REQUIRED_DUMMIES = {"AC_PIT_0", "AC_START_0", "AC_HOTLAP_START_0", "AC_AB_START_L", "AC_AB_START_R",
                    "AC_AB_FINISH_L", "AC_AB_FINISH_R"}
KNOWN_DUMMY = re.compile(r"^AC_(PIT_\d+|START_\d+|HOTLAP_START_\d+|AB_(START|FINISH)_[LR]|TIME_\d+_[LR])$")
SPAWNS = ("AC_PIT_0", "AC_START_0", "AC_HOTLAP_START_0")
SPAWN_HEIGHT = (0.15, 0.5)
AI_HEIGHT = (0.3, 0.5)
MIN_GATE_M = 6.0
MIN_RADIUS_M = 5.0  # tighter than any real hairpin: a routing artefact


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def check(self, ok: bool, what: str) -> bool:
        (self.passed if ok else self.failed).append(what)
        return ok

    @property
    def ok(self) -> bool:
        return not self.failed


class RoadSurface:
    """Height of the road (1ROAD meshes) under arbitrary X/Z points."""

    def __init__(self, nodes: list[dict]):
        tris = []
        for nd in nodes:
            if nd["class"] == 2 and "ROAD" in nd["name"]:
                p = nd["vertices"]["pos"].astype(float)
                tris.append(p[nd["indices"].reshape(-1, 3)])
        self.tri = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
        self.tree = cKDTree(self.tri.mean(axis=1)[:, [0, 2]]) if len(self.tri) else None

    def height(self, x: float, z: float) -> float | None:
        if self.tree is None:
            return None
        _, idx = self.tree.query([x, z], k=min(16, len(self.tri)))
        for i in np.atleast_1d(idx):
            a, b, c = self.tri[i][:, [0, 2]]
            v0, v1, v2 = c - a, b - a, np.array([x, z]) - a
            den = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(den) < 1e-12:
                continue
            u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
            v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
            if u >= -1e-6 and v >= -1e-6 and u + v <= 1 + 1e-6:
                ya, yb, yc = self.tri[i][:, 1]
                return float(ya + u * (yc - ya) + v * (yb - ya))
        return None


def _kn5_checks(rep: Report, path: Path) -> tuple[dict, dict] | None:
    try:
        data = kn5.read(path)
    except ValueError as e:
        rep.check(False, f"{path.name}: unreadable ({e})")
        return None
    rep.check(True, f"{path.name}: reads back cleanly (v{data['version']}, no trailing bytes)")
    nodes = data["nodes"]
    names = [n["name"] for n in nodes]
    rep.check(len(set(names)) == len(names), f"{path.name}: node names unique")
    rep.check(nodes[0]["class"] == 1 and all(n["depth"] == 1 for n in nodes[1:]),
              f"{path.name}: flat tree under one root dummy")
    meshes = [n for n in nodes if n["class"] == 2]
    dummies = {n["name"]: n for n in nodes[1:] if n["class"] == 1}
    rep.check(all(m["children"] == 0 for m in meshes), f"{path.name}: no mesh has children")
    rep.check(all(len(m["vertices"]) <= kn5.MAX_VERTICES for m in meshes),
              f"{path.name}: every mesh within {kn5.MAX_VERTICES} vertices")
    rep.check(all(m["indices"].max() < len(m["vertices"]) for m in meshes), f"{path.name}: indices in range")
    rep.check(all(m["material"] < len(data["materials"]) for m in meshes), f"{path.name}: material ids in range")
    bad_keys = [m["name"] for m in meshes if m["name"][:1].isdigit()
                and sum(k in m["name"] for k in SURFACE_KEYS + ("WALL",)) != 1]
    rep.check(not bad_keys, f"{path.name}: each physics mesh matches exactly one surface key"
              + (f" (bad: {bad_keys[:5]})" if bad_keys else ""))
    nrm = np.concatenate([m["vertices"]["normal"] for m in meshes]).astype(float)
    rep.check(np.allclose(np.linalg.norm(nrm, axis=1), 1, atol=1e-3), f"{path.name}: unit normals")
    missing = REQUIRED_DUMMIES - set(dummies)
    rep.check(not missing, f"{path.name}: required AC_ dummies present" + (f" (missing {sorted(missing)})" if missing else ""))
    unknown = [n for n in dummies if n.startswith("AC_") and not KNOWN_DUMMY.match(n)]
    rep.check(not unknown, f"{path.name}: no unknown AC_ names" + (f" ({unknown})" if unknown else ""))
    for name, d in dummies.items():
        m = d["matrix"][:3, :3]
        if not (np.allclose(m @ m.T, np.eye(3), atol=1e-4) and abs(np.linalg.det(m) - 1) < 1e-4):
            rep.check(False, f"{path.name}: {name} matrix is not a proper rotation")
    return data, dummies


def _layout_checks(rep: Report, label: str, data: dict, dummies: dict, ai_path: Path) -> None:
    road = RoadSurface(data["nodes"])
    for name in SPAWNS:
        if name not in dummies:
            continue
        x, y, z = dummies[name]["matrix"][3, :3]
        h = road.height(x, z)
        ok = h is not None and SPAWN_HEIGHT[0] <= y - h <= SPAWN_HEIGHT[1]
        rep.check(ok, f"{label}: {name} sits {('%.2f m' % (y - h)) if h is not None else 'off the road'} above the road")
    for gate in ("AC_AB_START", "AC_AB_FINISH"):
        left, right = dummies.get(f"{gate}_L"), dummies.get(f"{gate}_R")
        if left is None or right is None:
            continue
        pl, pr = left["matrix"][3, :3], right["matrix"][3, :3]
        span = float(np.linalg.norm(pl - pr))
        mid = (pl + pr) / 2
        rep.check(span >= MIN_GATE_M and road.height(mid[0], mid[2]) is not None,
                  f"{label}: {gate} gate spans the road ({span:.1f} m)")

    try:
        line = ai_line.read(ai_path)
    except (ValueError, OSError, struct.error) as e:
        rep.check(False, f"{label}: fast_lane.ai unreadable ({e})")
        return
    rep.check(True, f"{label}: fast_lane.ai v7 reads back ({len(line.pos)} points)")
    rep.check(bool(np.all(np.diff(line.length) > 0)), f"{label}: AI distances strictly increase")
    v = line.extra[:, 0]
    rep.check(bool(np.isfinite(v).all() and v.min() > 5), f"{label}: AI speeds finite and above 18 km/h")
    sample = line.pos[:: max(len(line.pos) // 200, 1)]
    lift = [p[1] - h for p in sample if (h := road.height(p[0], p[2])) is not None]
    rep.check(len(lift) > 0.9 * len(sample) and AI_HEIGHT[0] <= min(lift) and max(lift) <= AI_HEIGHT[1],
              f"{label}: AI line {min(lift):.2f}-{max(lift):.2f} m above the road" if lift else f"{label}: AI line off the road")
    radius = line.extra[:, 4]
    rep.check(bool(radius.min() >= MIN_RADIUS_M),
              f"{label}: tightest bend radius {radius.min():.1f} m (needs >= {MIN_RADIUS_M:.0f} m)")
    if "AC_START_0" in dummies:
        fwd = dummies["AC_START_0"]["matrix"][2, :3]
        ai_dir = line.pos[min(40, len(line.pos) - 1)] - line.pos[0]
        ai_dir /= np.linalg.norm(ai_dir)
        rep.check(float(fwd @ ai_dir) > 0.8, f"{label}: spawn faces the AI line's direction")


def track_folder(track_dir: Path) -> Report:
    rep = Report()
    tid = track_dir.name
    rep.check(bool(re.fullmatch(r"[a-z0-9_]{1,32}", tid)), f"track folder name '{tid}' is valid")
    models = sorted(track_dir.glob("models_*.ini"))
    layouts = [m.stem.removeprefix("models_") for m in models] or [""]
    if models:
        rep.check(not (track_dir / "ui" / "ui_track.json").exists(), "no root ui_track.json in a multi-layout track")
    for lay in layouts:
        label = lay or "main"
        rep.check(bool(re.fullmatch(r"[a-z0-9_]{0,15}", lay)), f"layout id '{label}' is valid")
        base = track_dir / lay if lay else track_dir
        ui = track_dir / "ui" / lay if lay else track_dir / "ui"
        if lay:
            text = (track_dir / f"models_{lay}.ini").read_text(encoding="ascii")
            files = re.findall(r"^FILE=(.+?)\s*$", text, re.M)
        else:
            files = [f"{tid}.kn5"]
        rep.check(bool(files) and all((track_dir / f).exists() for f in files), f"{label}: listed KN5 files exist")
        files = [f for f in files if (track_dir / f).exists()]
        for rel in ("data/surfaces.ini", "data/map.ini", "map.png", "ai/fast_lane.ai"):
            rep.check((base / rel).exists(), f"{label}: {rel} present")
        for rel in ("ui_track.json", "preview.png", "outline.png"):
            rep.check((ui / rel).exists(), f"{label}: ui/{rel} present")
        all_dummies: dict = {}
        datas = []
        for f in files:
            res = _kn5_checks(rep, track_dir / f)
            if res:
                datas.append(res[0])
                all_dummies.update(res[1])
        if (ui / "ui_track.json").exists():
            meta = json.loads((ui / "ui_track.json").read_text(encoding="utf-8"))
            pits = sum(1 for n in all_dummies if re.fullmatch(r"AC_PIT_\d+", n))
            rep.check(meta.get("pitboxes") == str(pits), f"{label}: pitboxes matches {pits} AC_PIT dummies")
            rep.check(isinstance(meta.get("geotags"), list) and all(isinstance(g, str) for g in meta["geotags"]),
                      f"{label}: geotags are strings")
            rep.check(isinstance(meta.get("length"), str) and isinstance(meta.get("year"), int),
                      f"{label}: length is a string and year an integer")
        if datas:
            merged = {"nodes": [n for d in datas for n in d["nodes"]], "materials": datas[0]["materials"]}
            _layout_checks(rep, label, merged, all_dummies, base / "ai" / "fast_lane.ai")
    for rel in ("ATTRIBUTION.txt", "centreline.geojson"):
        rep.check((track_dir / rel).exists(), f"{rel} present")
    return rep
