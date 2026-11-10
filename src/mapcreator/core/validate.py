"""Static checks on a finished track folder: everything we can verify without the game.

Each rule comes from a known cause of load failures or crashes (see
docs/ac-format.md). The in-game checks (does it load, are the wheels on the
ground, does the AI finish) live in the Windows test harness.

Two kinds of failure:
- **Build failures** (`Report.failed`): a malformed KN5, a missing dummy, a
  broken AI file. They're our pipeline's fault, and the person who drew the
  road can't do anything about them.
- **Route failures** (`Report.route_failed`): a bend tighter than any real
  road (`MIN_RADIUS_M`). Usually the route turns back on itself at a junction,
  or a waypoint sits just off the road, and moving a waypoint fixes it. A road
  that really is that tight can be built anyway (`allow_tight_bends`), down to
  `FLOOR_RADIUS_M`; the bend is then a warning.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.spatial import cKDTree

from mapcreator.core import ai_line, kn5, racing_line
from mapcreator.core.package import ROAD_KEYS, SURFACE_KEYS, surface_keys
from mapcreator.core.power import POWER_COLLIDER
from mapcreator.core.scenery import TRUNK_MESH

REQUIRED_DUMMIES = {"AC_PIT_0", "AC_START_0", "AC_HOTLAP_START_0", "AC_AB_START_L", "AC_AB_START_R",
                    "AC_AB_FINISH_L", "AC_AB_FINISH_R"}
KNOWN_DUMMY = re.compile(r"^AC_(PIT_\d+|START_\d+|HOTLAP_START_\d+|AB_(START|FINISH)_[LR]|TIME_\d+_[LR])$")
SPAWN = re.compile(r"^AC_(PIT|START|HOTLAP_START)_\d+$")
GRID_CLEARANCE_M = 3.5  # grid spots closer than this would put two cars in one place (a pair stands 4 m apart)
TRUNK_SPAWN_CLEARANCE_M = 3.0  # a tree or pole collider nearer a spawn than this could be inside or touching the car
SCENERY_COLLIDERS = (("tree", TRUNK_MESH), ("power", POWER_COLLIDER))  # invisible walls the scenery stage adds
SPAWN_HEIGHT = (0.15, 0.5)
AI_HEIGHT = (0.3, 0.5)
# A gate must reach past both road edges; any span is fine as long as it does. This floor only catches
# a degenerate gate (both ends in one place).
MIN_GATE_M = 3.0
# The length as `package.ui_track_json` writes it: kilometres to two places, then the unit. Content Manager ignores
# a bare number, so the unit has to be there.
UI_LENGTH = re.compile(r"^\d+\.\d{2} km$")
MIN_RADIUS_M = 5.0  # tighter than any real hairpin: usually a routing artefact
# Not even on request below this: the route turns on the spot. It also keeps the AI line's slowest speed,
# sqrt(MU * G * r), above the 5 m/s the speed check wants.
FLOOR_RADIUS_M = 2.5
MIN_AI_SPEED = 5.0


@dataclass(frozen=True)
class TightBend:
    """The tightest point of a layout's AI line, where that's under MIN_RADIUS_M."""
    layout: str
    radius_m: float
    x: float  # AC coordinates
    z: float


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)  # build failures: ours
    route_failed: list[str] = field(default_factory=list)  # about the drawn road: fixable, or can be accepted
    warnings: list[str] = field(default_factory=list)
    bends: list[TightBend] = field(default_factory=list)

    def check(self, ok: bool, what: str) -> bool:
        (self.passed if ok else self.failed).append(what)
        return ok

    @property
    def ok(self) -> bool:
        return not self.failed and not self.route_failed


def bend_checks(rep: Report, label: str, line: ai_line.AiLine, *, allow_tight_bends: bool = False,
                radius: np.ndarray | None = None, at: np.ndarray | None = None) -> None:
    """The AI line's speeds and the road's tightest bend: `radius` at AC x/z points `at`, by default the AI line's
    own radius at its points. A slow AI line is put down to the bend when there's one."""
    radius = line.extra[:, 4] if radius is None else np.asarray(radius, dtype=float)
    at = np.asarray(line.pos, dtype=float)[:, [0, 2]] if at is None else np.asarray(at, dtype=float)
    i = int(np.argmin(radius))
    r = float(radius[i])
    v = line.extra[:, 0]
    rep.check(bool(np.isfinite(v).all() and (v.min() > MIN_AI_SPEED or r < MIN_RADIUS_M)),
              f"{label}: AI speeds finite and above 18 km/h")
    what = f"{label}: tightest bend radius {r:.1f} m (needs >= {MIN_RADIUS_M:.0f} m)"
    if r >= MIN_RADIUS_M:
        rep.check(True, what)
        return
    rep.bends.append(TightBend(label, r, float(at[i][0]), float(at[i][1])))
    if allow_tight_bends and r >= FLOOR_RADIUS_M:
        rep.warnings.append(f"{what}; built anyway, as asked")
    else:
        rep.route_failed.append(what)


# The racing line (core/racing_line.py) keeps at least EDGE_MARGIN_M from each road edge; between its stations, at
# the apex of a hairpin, it can come a few centimetres closer. This much closer fails.
AI_EDGE_MIN_M = racing_line.EDGE_MARGIN_M - 0.2
# Where it's fast it keeps more (racing_line.edge_margin, #50), or the middle of a road too narrow for that. This
# much less than that is a warning: the build goes on, as with the line's radius below.
AI_EDGE_SPEED_TOL_M = 0.2
# The line may be this much (a share) tighter than the road nearby before it's warned about. Only a warning, by
# the maintainer's decision (#44): it never stops a build.
LINE_RADIUS_TOL = 0.05
LINE_RADIUS_NEAR_M = 30.0  # either side of a point, the road's tightest radius in this reach is "nearby"
# Above this radius the planned speed is capped anyway, so the line's radius there doesn't matter.
LINE_RADIUS_CARE_M = ai_line.V_MAX ** 2 / (ai_line.MU * ai_line.G)


def _spacing(line: ai_line.AiLine) -> float:
    return float(np.median(np.diff(line.length))) if len(line.length) > 1 else 1.0


def _radius(xz: np.ndarray) -> np.ndarray:
    """Radius at each point of a line of AC x/z points (1 m apart, or 5 m as written now): traced through a spline
    every metre, smoothed as the AI file's own, the tightest within half a spacing of each point."""
    xz = np.asarray(xz, dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xz, axis=0).T))])
    if len(s) < 4 or s[-1] <= 0:
        return np.full(len(s), 1000.0)
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    spline = CubicSpline(s[keep], xz[keep])
    fine = np.append(np.arange(0.0, s[-1], 1.0), s[-1])
    d1, d2 = spline(fine, 1), spline(fine, 2)
    k = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(np.hypot(d1[:, 0], d1[:, 1]) ** 3, 1e-12)
    r = ai_line.radius_of(k, 1.0)
    owner = np.searchsorted((s[1:] + s[:-1]) / 2, fine)  # the nearest point to each metre
    out = np.full(len(s), np.inf)
    np.minimum.at(out, owner, r)
    empty = ~np.isfinite(out)  # points under a metre apart
    out[empty] = np.interp(s[empty], fine, r)
    return out


def road_middle(line: ai_line.AiLine) -> tuple[np.ndarray, np.ndarray]:
    """The middle of the road beside each AI point, from the point and its distances to the two edges (fields 5
    and 6), as AC x/z; and its radius. On a line along the middle of the road, the line itself. From points 5 m
    apart the radius reads a little tight in hairpins (Brinktop Road's 7.2 m corner as 6.9 m), so the route check
    reads the centreline itself when it has it (`track_folder`'s `road_radius`)."""
    fwd = line.extra[:, [13, 15]]
    left = np.column_stack([fwd[:, 1], -fwd[:, 0]]) / np.maximum(np.hypot(*fwd.T), 1e-9)[:, None]
    xz = np.asarray(line.pos, dtype=float)[:, [0, 2]] + left * ((line.extra[:, 5] - line.extra[:, 6]) / 2)[:, None]
    return xz, _radius(xz)


def racing_line_checks(rep: Report, label: str, line: ai_line.AiLine, road: RoadSurface, middle_radius) -> None:
    """The AI line stays on the road, AI_EDGE_MIN_M or more from each edge (by its own edge distances, and on the
    road's triangles): failures. It keeps the margin its planned speed asks for, and nowhere bends tighter than
    the road does nearby: warnings if not."""
    side = np.minimum(line.extra[:, 5], line.extra[:, 6])
    i = int(np.argmin(side))
    rep.check(bool(side[i] >= AI_EDGE_MIN_M),
              f"{label}: AI line at least {AI_EDGE_MIN_M:.1f} m inside both road edges "
              f"({side[i]:.2f} m at the closest, {line.length[i]:.0f} m along it)")
    # The margin for the planned speed, or half the road where it's narrower than twice that.
    need = np.minimum(racing_line.edge_margin(line.extra[:, 0]), (line.extra[:, 5] + line.extra[:, 6]) / 2)
    short = need - side
    j = int(np.argmax(short))
    what = (f"{side[j]:.2f} m from the edge at {line.extra[j, 0] * 3.6:.0f} km/h, {line.length[j]:.0f} m along it, "
            f"against {need[j]:.2f} m")
    if short[j] <= AI_EDGE_SPEED_TOL_M:
        rep.passed.append(f"{label}: AI line keeps its speed's margin from both road edges "
                          f"({racing_line.EDGE_MARGIN_M:.1f} m at {racing_line.MARGIN_SLOW_KMH:.0f} km/h to "
                          f"{racing_line.EDGE_MARGIN_FAST_M:.1f} m at {racing_line.MARGIN_FAST_KMH:.0f} km/h; "
                          f"the closest {what})")
    else:
        rep.warnings.append(f"{label}: the AI line runs close to the edge for its speed: {what}")
    spacing = _spacing(line)
    pick = np.arange(0, len(line.pos), max(int(round(5.0 / spacing)), 1))  # every point, or every 5 m
    fwd = line.extra[pick][:, [13, 15]]
    left = np.column_stack([fwd[:, 1], -fwd[:, 0]]) / np.maximum(np.hypot(*fwd.T), 1e-9)[:, None]
    xz = np.asarray(line.pos, dtype=float)[pick][:, [0, 2]]
    off = ~(road.covers(xz + left * AI_EDGE_MIN_M) & road.covers(xz - left * AI_EDGE_MIN_M))
    short = (f" ({int(off.sum())} of {len(pick)} points short, the first {line.length[pick][off][0]:.0f} m along it)"
             if off.any() else "")
    rep.check(not bool(off.any()),
              f"{label}: the road's triangles reach {AI_EDGE_MIN_M:.1f} m either side of the AI line{short}")
    r_line = _radius(np.asarray(line.pos, dtype=float)[:, [0, 2]])
    w = 2 * max(int(round(LINE_RADIUS_NEAR_M / spacing)), 1) + 1
    nearby = minimum_filter1d(np.asarray(middle_radius, dtype=float), w, mode="nearest")
    ratio = np.where(r_line < LINE_RADIUS_CARE_M, r_line / np.minimum(nearby, LINE_RADIUS_CARE_M), np.inf)
    j = int(np.argmin(ratio))
    if ratio[j] >= 1 - LINE_RADIUS_TOL:
        rep.passed.append(f"{label}: AI line's tightest radius {r_line.min():.1f} m, nowhere tighter than the road "
                          f"nearby (worst {r_line[j]:.1f} m against {nearby[j]:.1f} m, "
                          f"{line.length[j]:.0f} m along it)")
    else:
        rep.warnings.append(f"{label}: the AI line bends tighter than the road near {line.length[j]:.0f} m along it: "
                            f"{r_line[j]:.1f} m against the road's {nearby[j]:.1f} m")


VERTICAL_G_LIMIT = 0.5  # a crest or sag pulling more than this, at the AI's planned speed, gets a warning
VERTICAL_EDGE_M = 20  # metres at each end of the AI line left out of that check


def vertical_checks(rep: Report, label: str, line: ai_line.AiLine) -> None:
    """A warning, never a failure, where a crest or sag is tighter than the AI's planned speed suits: more than
    VERTICAL_G_LIMIT of extra vertical acceleration. Real narrow hill roads have such crests, so the profile
    isn't smoothed to suit; but a tight one can also mean the line reads a cutting beside the road."""
    s = np.asarray(line.length, dtype=float)
    if len(s) < 50:
        return
    grid = np.arange(s[0], s[-1], 1.0)
    h = uniform_filter1d(np.interp(grid, s, np.asarray(line.pos, dtype=float)[:, 1]), 10, mode="nearest")
    k = uniform_filter1d(np.gradient(np.gradient(h)), 10, mode="nearest")  # 1/m: + is a sag, - a crest
    v = np.interp(grid, s, line.extra[:, 0])
    extra = v ** 2 * np.abs(k) / 9.81
    extra[:VERTICAL_EDGE_M] = extra[-VERTICAL_EDGE_M:] = 0.0  # the derivatives aren't reliable at the line's ends
    i = int(np.argmax(extra))
    if extra[i] > VERTICAL_G_LIMIT:
        rep.warnings.append(
            f"{label}: a {'sag' if k[i] > 0 else 'crest'} at {grid[i]:.0f} m along the AI line has a "
            f"{1 / abs(k[i]):.0f} m vertical radius: {extra[i]:.2f} g extra at the AI's planned {v[i] * 3.6:.0f} km/h")


def bend_hint_checks(rep: Report, label: str, line: ai_line.AiLine, road: RoadSurface, hints_ini: Path) -> None:
    """A warning, never a failure, at each bend AC's AI would take faster than the road allows: over a crest, down
    a steep grade, with no banking, or braking for it over a crest (`ai_line.bend_hints`, #63). It says where the
    bend is, the speeds, and what hint `data/ai_hints.ini` holds there. The bends are found from the AI file, as the
    track stage found them, with the banking read from the road's triangles (`_banking`). Only where the car is over
    the grip on a level road: elsewhere even 7% of banking away from the bend leaves the pace over 0.96, nowhere
    near BEND_TRIGGER."""
    level = ai_line.bend_paces(line)
    short = np.interp(np.asarray(line.length, dtype=float), level["s"], level["pace"]) < 1.0
    short[1:] |= short[:-1].copy()
    short[:-1] |= short[1:].copy()
    bends = ai_line.bend_hints(line, _banking(line, road, short))
    if not bends:
        rep.passed.append(f"{label}: no bend the AI would take faster than the road allows")
        return
    written = _hint_ranges(hints_ini)
    s = np.asarray(line.length, dtype=float)
    total = max(float(s[-1]), 1.0)
    for b in bends:
        over = [v for s0, s1, v in written if s0 * total < b.end and s1 * total > b.start]
        if over and min(over) <= b.pace + 1e-6:
            hint = f"a hint of {min(over):g} is written there"
        else:
            hint = f"no hint of {b.pace:g} is written there" + (f" (only {min(over):g})" if over else "")
        why = ([f"over a {b.crest_radius_m:.0f} m crest"] if np.isfinite(b.crest_radius_m) else []) + [
            f"{b.descent * 100:.1f}% downhill", f"{round(b.banking * 100, 1) + 0.0:.1f}% banking"]
        where = f"at {b.start:.0f}-{b.end:.0f} m along the AI line ({', '.join(why)})"
        if b.car_kmh is not None and b.car_kmh > b.ai_kmh + 1.0:
            # Braking over a crest: the car is furthest over on the bend's way in, so the bend is named by its
            # tightest radius over the hint, where the hint holds the AI's corner speed lowest.
            tight = float(np.min(np.interp(np.arange(b.start, b.end + 0.5), s, line.extra[:, 4])))
            corner = np.sqrt(ai_line.BEND_GRIP * ai_line.G * tight) * 3.6
            rep.warnings.append(
                f"{label}: the {tight:.0f} m bend {where}: braking over the crest before it, the car may reach the "
                f"{b.radius_m:.0f} m radius on its way in at {b.car_kmh:.0f} km/h, where AC's AI plans "
                f"{b.ai_kmh:.0f} km/h and the grip left allows {b.road_kmh:.0f} km/h ({b.pace:g} of the "
                f"{corner:.0f} km/h the AI works out for its tightest radius); {hint}")
            continue
        corner = np.sqrt(ai_line.BEND_GRIP * ai_line.G * b.radius_m) * 3.6
        rep.warnings.append(
            f"{label}: the {b.radius_m:.0f} m bend {where}: AC's AI may take it at {b.ai_kmh:.0f} km/h where the "
            f"grip left allows {b.road_kmh:.0f} km/h ({b.pace:g} of the {corner:.0f} km/h it works out for the "
            f"radius); {hint}")


BANKING_INSET_M = 0.25  # the banking is read this far inside each road edge, to be sure of hitting the road


def _banking(line: ai_line.AiLine, road: RoadSurface, where: np.ndarray) -> np.ndarray:
    """The road's banking beside each AI point, as its rise to the left per metre, from its triangles: straight
    across the road's middle (`road_middle`), from just inside its right edge to just inside its left, the same
    distance either side of the crown, so the crown cancels out as in `ai_line.road_banking`. Across the road, not
    the line: in a tight bend the line's heading is several degrees off the road's, and on a steep grade that tilt
    reads as banking. Only at the points `where` says: 0 elsewhere, and where the road isn't found."""
    out = np.zeros(len(line.pos))
    if len(line.pos) < 2:
        return out
    middle, _ = road_middle(line)
    along = np.gradient(middle, axis=0)
    left = np.column_stack([along[:, 1], -along[:, 0]]) / np.maximum(np.hypot(*along.T), 1e-9)[:, None]
    half = (line.extra[:, 5] + line.extra[:, 6]) / 2 - BANKING_INSET_M
    for i in np.flatnonzero(where & (half > 0)):
        up = road.height(*(middle[i] + left[i] * half[i]))
        down = road.height(*(middle[i] - left[i] * half[i]))
        if up is not None and down is not None:
            out[i] = (up - down) / (2 * half[i])
    return out


def _hint_ranges(path: Path) -> list[tuple[float, float, float]]:
    """Each `[HINT_n]` in an ai_hints.ini as (START, END, VALUE); none if there's no file."""
    if not path.exists():
        return []
    out = []
    for block in re.split(r"^\[HINT_\d+\]\s*$", path.read_text(encoding="ascii"), flags=re.M)[1:]:
        values = dict(re.findall(r"^(START|END|VALUE)=([-+.\deE]+)\s*$", block, flags=re.M))
        if len(values) == 3:
            out.append((float(values["START"]), float(values["END"]), float(values["VALUE"])))
    return out


def is_road_mesh(name: str) -> bool:
    """A physics mesh of the drivable road: sealed (1ROAD_###) or unsealed (1GRAVEL_###)."""
    return name[:1].isdigit() and any(k in name for k in ROAD_KEYS)


class RoadSurface:
    """Height of the road (1ROAD and 1GRAVEL meshes) under arbitrary X/Z points."""

    def __init__(self, nodes: list[dict]):
        tris = []
        for nd in nodes:
            if nd["class"] == 2 and is_road_mesh(nd["name"]):
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

    def covers(self, xz: np.ndarray, batch: int = 8192) -> np.ndarray:
        """Whether each X/Z point is over the road: `height` for many points at once."""
        xz = np.asarray(xz, dtype=float).reshape(-1, 2)
        out = np.zeros(len(xz), dtype=bool)
        if self.tree is None:
            return out
        k = min(16, len(self.tri))
        for start in range(0, len(xz), batch):
            pts = xz[start:start + batch]
            _, idx = self.tree.query(pts, k=k, workers=-1)
            t = self.tri[np.asarray(idx).reshape(len(pts), k)][..., [0, 2]]
            a = t[..., 0, :]
            v0, v1, v2 = t[..., 2, :] - a, t[..., 1, :] - a, pts[:, None, :] - a
            den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]
            ok = np.abs(den) > 1e-12
            den = np.where(ok, den, 1.0)
            u = (v2[..., 0] * v1[..., 1] - v1[..., 0] * v2[..., 1]) / den
            v = (v0[..., 0] * v2[..., 1] - v2[..., 0] * v0[..., 1]) / den
            out[start:start + batch] = (ok & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)).any(axis=1)
        return out


def undefined_surfaces(mesh_names: list[str], surfaces_ini: str) -> list[str]:
    """Physics meshes whose surface KEY the layout's `surfaces.ini` doesn't define: a 1GRAVEL mesh on a track
    written without GRAVEL. WALL needs no KEY (it's deliberately left undefined)."""
    defined = surface_keys(surfaces_ini)
    return [n for n in mesh_names if n[:1].isdigit() and "WALL" not in n and not any(k in n for k in defined)]


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
    for what, prefix in SCENERY_COLLIDERS:
        found = [m for m in meshes if m["name"].startswith(prefix)]
        if found:
            rep.check(all(m["renderable"] == 0 for m in found),
                      f"{path.name}: {len(found)} {what} collider meshes, all invisible")
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


def _layout_checks(rep: Report, label: str, data: dict, dummies: dict, ai_path: Path, *,
                   allow_tight_bends: bool = False, road_radius: tuple[np.ndarray, np.ndarray] | None = None) -> None:
    """`road_radius`: the centreline's radius every metre, as (AC x/z points, radii), for the route check. Without
    it (a folder checked on its own) the road's middle is found from the AI file."""
    road = RoadSurface(data["nodes"])
    for name in sorted(n for n in dummies if SPAWN.match(n)):
        x, y, z = dummies[name]["matrix"][3, :3]
        h = road.height(x, z)
        ok = h is not None and SPAWN_HEIGHT[0] <= y - h <= SPAWN_HEIGHT[1]
        rep.check(ok, f"{label}: {name} sits {('%.2f m' % (y - h)) if h is not None else 'off the road'} above the road")
    starts = np.array([d["matrix"][3, :3] for n, d in sorted(dummies.items()) if re.fullmatch(r"AC_START_\d+", n)])
    if len(starts) > 1:
        gaps = np.linalg.norm(starts[:, None, [0, 2]] - starts[None, :, [0, 2]], axis=2)
        np.fill_diagonal(gaps, np.inf)
        rep.check(bool(gaps.min() >= GRID_CLEARANCE_M),
                  f"{label}: {len(starts)} grid spots, at least {GRID_CLEARANCE_M:.0f} m apart ({gaps.min():.1f} m)")
    for what, prefix in SCENERY_COLLIDERS:
        found = [n for n in data["nodes"] if n["class"] == 2 and n["name"].startswith(prefix)]
        if not found:
            continue
        # Every corner, not just the trunk's or pole's centre: a prism standing on the road edge is caught too.
        xz = np.unique(np.concatenate([m["vertices"]["pos"][:, [0, 2]] for m in found]).round(2), axis=0)
        on_road = int(road.covers(xz).sum())
        rep.check(on_road == 0, f"{label}: {what} colliders clear of the road ({on_road} corners over it)")
        spawns = np.array([d["matrix"][3, [0, 2]] for n, d in dummies.items() if SPAWN.match(n)])
        if len(spawns):
            gap = float(cKDTree(xz).query(spawns, workers=-1)[0].min())
            rep.check(gap >= TRUNK_SPAWN_CLEARANCE_M,
                      f"{label}: {what} colliders at least {TRUNK_SPAWN_CLEARANCE_M:.0f} m from every spawn ({gap:.1f} m)")
    for gate in ("AC_AB_START", "AC_AB_FINISH"):
        left, right = dummies.get(f"{gate}_L"), dummies.get(f"{gate}_R")
        if left is None or right is None:
            continue
        pl, pr = left["matrix"][3, :3], right["matrix"][3, :3]
        span = float(np.linalg.norm(pl - pr))
        mid = (pl + pr) / 2
        spans = (span >= MIN_GATE_M and road.height(mid[0], mid[2]) is not None
                 and road.height(pl[0], pl[2]) is None and road.height(pr[0], pr[2]) is None)
        rep.check(spans, f"{label}: {gate} gate spans the road ({span:.1f} m, ends beyond both edges)")

    try:
        line = ai_line.read(ai_path)
    except (ValueError, OSError, struct.error) as e:
        rep.check(False, f"{label}: fast_lane.ai unreadable ({e})")
        return
    rep.check(True, f"{label}: fast_lane.ai v7 reads back ({len(line.pos)} points)")
    rep.check(bool(np.all(np.diff(line.length) > 0)), f"{label}: AI distances strictly increase")
    sample = line.pos[:: max(len(line.pos) // 200, 1)]
    lift = [p[1] - h for p in sample if (h := road.height(p[0], p[2])) is not None]
    rep.check(len(lift) > 0.9 * len(sample) and AI_HEIGHT[0] <= min(lift) and max(lift) <= AI_HEIGHT[1],
              f"{label}: AI line {min(lift):.2f}-{max(lift):.2f} m above the road" if lift else f"{label}: AI line off the road")
    _middle, middle_radius = road_middle(line)
    if road_radius is None:
        bend_checks(rep, label, line, allow_tight_bends=allow_tight_bends, radius=middle_radius)
    else:
        bend_checks(rep, label, line, allow_tight_bends=allow_tight_bends, radius=road_radius[1], at=road_radius[0])
    racing_line_checks(rep, label, line, road, middle_radius)
    vertical_checks(rep, label, line)
    bend_hint_checks(rep, label, line, road, ai_path.parents[1] / "data" / "ai_hints.ini")
    for name in sorted(n for n in dummies if re.fullmatch(r"AC_START_\d+", n)):
        # The AI line's own direction where the car stands: a chord from the line's start would cut across
        # a hairpin there and fail a spawn that faces exactly the right way.
        fwd = dummies[name]["matrix"][2, :3]
        spawn = dummies[name]["matrix"][3, :3]
        i = int(np.argmin(np.linalg.norm((line.pos - spawn)[:, [0, 2]], axis=1)))
        k = max(int(round(2.0 / _spacing(line))), 1)  # about 2 m either side: 2 points 1 m apart, 1 point 5 m apart
        lo, hi = max(i - k, 0), min(i + k, len(line.pos) - 1)
        ai_dir = line.pos[hi] - line.pos[lo]
        ai_dir /= max(float(np.linalg.norm(ai_dir)), 1e-9)
        rep.check(float(fwd @ ai_dir) > 0.8, f"{label}: {name} faces the AI line's direction")


def track_folder(track_dir: Path, *, allow_tight_bends: bool = False,
                 road_radius: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> Report:
    """`road_radius`: by layout label ("main", "forward", "reverse"), the centreline's radius every metre over the
    layout's AI line, as (AC x/z points, radii). The validate stage passes it from the profile, so a bend is judged
    as main's 1 m centreline line judged it."""
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
            length_ok = isinstance(meta.get("length"), str) and UI_LENGTH.match(meta["length"])
            rep.check(bool(length_ok) and isinstance(meta.get("year"), int),
                      f"{label}: length in km with its unit, and year an integer")
        if datas and (base / "data" / "surfaces.ini").exists():
            names = [n["name"] for d in datas for n in d["nodes"] if n["class"] == 2]
            bad = undefined_surfaces(names, (base / "data" / "surfaces.ini").read_text(encoding="ascii"))
            rep.check(not bad, f"{label}: data/surfaces.ini defines every physics mesh's surface"
                      + (f" (not {bad[:5]})" if bad else ""))
        if datas:
            merged = {"nodes": [n for d in datas for n in d["nodes"]], "materials": datas[0]["materials"]}
            _layout_checks(rep, label, merged, all_dummies, base / "ai" / "fast_lane.ai",
                           allow_tight_bends=allow_tight_bends, road_radius=(road_radius or {}).get(label))
    for rel in ("ATTRIBUTION.txt", "centreline.geojson", "extension/ext_config.ini"):
        rep.check((track_dir / rel).exists(), f"{rel} present")
    return rep
