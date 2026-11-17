"""The route's own bridges: open decks, parapets, abutments and piers; and 1:2 batters on fills.

The maintainer's answers to the flyovers research (17 Nov 2026) this follows:
- **Q2:** a tagged bridge whose DEM gap is at least OPEN_GAP_M is an open deck over the real terrain, with concrete
  parapets, abutments, and piers about every PIER_SPAN_M (a guess at a typical beam span) where the gap is over
  PIER_GAP_M. A shallower tagged bridge keeps the road on the ground, with the parapets. Untagged fills
  (`profile.narrow_dips`, mostly culverts) stay embankments, with a 1:2 batter instead of a near-vertical skirt.
- **Q3:** one generic style: a concrete beam deck with a concrete barrier and one steel rail.
- **Q4:** the parapet is solid, and its collider is flush with the barrier's face, so the collision is where the eye
  sees it, with no invisible wall above it: 1.0 m high on the barrier's line, as the guard rails' are
  (`core.roadside`).
- **Q5:** the deck's edges are a concrete kerb and footway strip KERB_M wide instead of the 2 m grass verge, with
  reflectors on the parapet instead of guide posts. The kerb's physics is ROAD: a car that runs wide on a bridge is on
  concrete, not grass.

**Which bridges.** The route's runs tagged `bridge` (anything but `no` or `low_water_crossing`, `crossings.is_bridge`),
as the profile stage marks them per station (`bridge` in `centreline.npz`). Their profile is already a straight grade
between their ends and their camber crown only (both from the profile and section stages). The gap is the profile's
height over the DEM's ground under the centreline, `z - z_raw` (none over the sea: taken as deep).

**The plan** (`plan`, run by the mesh stage; `04_mesh/bridges.json`, its format in docs/ac-format.md):
- **Every tagged bridge** of two stations or more gets the kerb strip and a parapet on both sides, its whole length,
  except across a side road's mouth on that side (`core.junctions`), where neither stands.
- **The open span:** where a bridge's gap reaches OPEN_GAP_M, from the first station where it does to the last, kept
  END_ROWS from the road's ends and MOUTH_CLEAR_M from any side road's mouth. Shorter than MIN_OPEN_M, the bridge is
  built on the ground. Its two end rows are the abutments.
- **Piers** in an open span over PIER_SPAN_M long whose gap is over PIER_GAP_M: the span split into stretches of
  about PIER_SPAN_M, a pier at each split where the gap there is over PIER_GAP_M.

**The road and the ground** (the mesh stage, through `core.mesh`):
- **The kerb strip** is the ribbon's own verge column moved in to KERB_M from the road's edge and raised KERB_RISE_M
  above it (`mesh.build_ribbon(verge_l=, verge_r=)`), so it shares the road's edge vertices and meets the grass verge
  beside the bridge in one ribbon, with no crack. It's drawn here as `1ROAD_BK<k>L/R`, material `kerb` (the
  roundabout island's concrete, `core.roundabouts`); the grass verge leaves it out (`mesh.ribbon_meshes(kerb=)`).
- **The open span's rows leave the terrain's footprint** (`mesh.Footprint.spans`), so the terrain runs on under the
  deck, from the DEM. There's no skirt, so no embankment. At each abutment row an **abutment face** drops from the
  deck's edge row to the ground: a concrete wall whose foot is a copy of the row ABUT_SET_M into the span at the
  ground's height there (`edges`), and the terrain meets that foot, so the wall, the road and the ground share
  every vertex where they meet. Its two outer corners are the skirt's first column, so it closes against the skirt
  beside the bridge's end too.
- **Under the deck** the ground is kept at least CLEAR_M below the underside (`ground`, as `water.lowered` lowers it
  under water), over the deck's plan and GUARD_M beyond. Points under the deck and along its edges join the terrain
  (`edges`), so its triangles follow the ground there, never a long one from an abutment's foot to the other's.
- **Batters** (`ground`, and the ribbon's skirt): on untagged fills (`filled` in `centreline.npz`, less bridges,
  tunnels and the humps under crossing roads), padded by BATTER_PAD_M, and on bridges' closed stretches and their
  approaches, padded by BATTER_APPROACH_M, the ground beside the road is at least `verge - BATTER x distance` past
  the verge's edge, out to BATTER_REACH_M: a 1:2 batter down to where it meets the ground.

**The structure** (`Decks.meshes`), all visual but the colliders, in the generic style the bridges over the route have
(`core.decks`, its barrier, rail, posts, headstocks and columns, and `core.crossing_roads`'s materials):
- the parapet (`BRIDGE_<k>`, material `bridge_concrete`, as the bridges over the route): a concrete barrier PARAPET_W thick and PARAPET_H high on the kerb's
  outer edge, its outer face running on down as the deck's fascia, DECK_DEPTH_M below the kerb on the open span
  (EDGE_BEAM_M on the ground, into the batter); and the abutments' faces and the piers;
- a steel rail on posts along each parapet's top (`BRIDGE_<k>_RAIL`, material `rail`);
- the underside of the open span, flat across, in a darker concrete (`BRIDGE_<k>_U`, material `bridge_soffit`);
- the colliders `1WALL_BRIDGE_<k>L/R`, invisible, on the parapet's face from 0.3 m below to 1.0 m above the kerb;
- reflectors on posts on the parapet's top every REFLECTOR_SPACING_M (`FURNITURE_BRIDGE_<k>`, on the furniture's
  atlas and its retroreflective material, `core.furniture`), red on the left and white on the right.

A bridge whose open deck fails the checks on the built meshes (`faults`: the terrain above its underside, a gap in a
parapet's collider) is built again on the ground with its parapets, as a side road's stub is left out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import shapely
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from mapcreator import cpus
from mapcreator.core import crossing_roads, decks, mesh
from mapcreator.core.scene import LocalFrame, Mesh

FORMAT_VERSION = 1

OPEN_GAP_M = 1.5  # a tagged bridge whose gap reaches this is an open deck where it does (Q2)
PIER_GAP_M = 2.0  # piers only where the gap is over this (Q2)
PIER_SPAN_M = 25.0  # about one pier every this many metres: a guess at a typical beam span (Q2)
MIN_OPEN_M = 6.0  # an open span shorter than this is built on the ground
END_ROWS = 2  # an open span starts and ends at least this many rows from the road's ends
MOUTH_CLEAR_M = 3.0  # and this far from any side road's mouth
KERB_M = 1.0  # the kerb and footway strip, from the road's edge to the parapet's face (Q5)
KERB_RISE_M = 0.1  # its outer edge stands this far above the road's edge: a footway, ramped so a car isn't snagged
PARAPET_W = crossing_roads.PARAPET_W  # the concrete barrier's thickness (the generic style's, `core.decks`)
PARAPET_H = crossing_roads.PARAPET_H  # its height above the kerb
# The steel rail's bottom and top above the kerb, its width, and its posts: the generic style's (`core.decks`, from
# `core.crossing_roads`): about 1.1 m overall, an estimate from common practice.
RAIL = (crossing_roads.RAIL_TOP - crossing_roads.RAIL_SIZE, crossing_roads.RAIL_TOP)
RAIL_W = decks.BARRIER_RAIL[0]
RAIL_POST_W = decks.BARRIER_POST_W
RAIL_POST_SPACING_M = decks.BARRIER_POST_M
RAIL_STRIDE = 2  # the rail has a point every this many rows: a 2 m chord is within 5 mm of a 100 m bend
DECK_DEPTH_M = crossing_roads.DECK_DEPTH_M  # the fascia, from the kerb's top down to the underside (a beam deck)
EDGE_BEAM_M = 0.6  # on the ground the fascia reaches this far below the kerb, into the batter
CLEAR_M = 0.5  # the ground under an open deck stays at least this far below its underside
GUARD_M = 0.5  # and so does the ground this far beyond its fascia
ABUT_SET_M = 0.1  # an abutment's foot is this far into the span from its row
GROUND_STEP_M = 2.0  # points under an open deck: every this many metres along it
GROUND_ACROSS_M = 4.0  # and at most this far apart across it
GROUND_END_M = 2.0  # starting this far from each abutment (the terrain's lattice keeps 2.5 m from the footprint)
UNDER_CLEAR_OF_M = 1.5  # and this far from any other piece of ground under the deck (a road crossing under it)
COLLIDER = (-0.3, 1.0)  # the parapet's collider, below and above the kerb (as `roadside.COLLIDER`)
REFLECTOR_SPACING_M = 18.0  # a reflector on each parapet every this many metres (12-24 m, the research's 4a)
REFLECTOR_POST = (0.08, 0.04, 0.22)  # its post on the parapet's top: across, along, high
REFLECTOR_SIZE = (0.05, 0.12)  # each reflector: wide, high, its top 0.03 m below the post's
BATTER = 0.5  # a fill's side falls 1 m in 2 (Q2)
BATTER_PAD_M = 5.0  # the batter reaches this far along the road past a fill
BATTER_APPROACH_M = 30.0  # and this far past a bridge, whose approaches are usually embankments
BATTER_REACH_M = 40.0  # and at most this far from the centreline
BATTER_ALONG_M = 2.0  # beside its nearest station, never further along the road than this (so not past its ends)
PIER_HEADSTOCK = (1.0, 1.2)  # a pier's headstock: deep (below the underside), along the road
PIER_EDGE_M = 1.5  # its columns stand at least this far in from the deck's edges
PIER_COLUMNS_WIDE_M = 12.0  # three columns under a deck at least this wide, else two
PIER_END_M = 3.0  # no pier within this of an abutment
PIER_BURY_M = 0.5  # a column's foot is this far below the lowest ground round it
CONCRETE_TILE_M = 2.0  # the concrete's texture repeats every this many metres

KERB_MATERIAL = "kerb"  # the roundabout island's concrete kerb (`core.roundabouts`)
# One generic style (Q3): the bridges over the route's concrete, and its darker underside (`core.crossing_roads`, which
# registers both, with their KN5 values), since our vertices carry no baked shade.
MATERIAL = crossing_roads.CONCRETE
UNDERSIDE_MATERIAL = crossing_roads.SOFFIT
RAIL_MATERIAL = "rail"
COLLIDER_PREFIX = "1WALL_BRIDGE_"  # as `validate.BRIDGE_COLLIDER`


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) of each run of True, inclusive."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(mask, bool).astype(int), [0]])))
    return list(zip(edges[::2].tolist(), (edges[1::2] - 1).tolist()))


# -- the plan ----------------------------------------------------------------------------------------------------------

def plan(c, *, mouths: dict[str, np.ndarray] | None = None, names: list[str] | None = None) -> list[dict]:
    """The route's bridges, as bridges.json lists them (without the geometry `Decks.records` adds). `c` is the
    centreline (`centreline.npz`: s, bridge, z, z_raw, run), `mouths` the side roads' mouths per side
    (`junctions.masks`), `names` each run's way name (or None)."""
    s = np.asarray(c["s"], float)
    n = len(s)
    if "bridge" not in c or not np.asarray(c["bridge"]).any():
        return []
    gap = np.nan_to_num(np.asarray(c["z"], float) - np.asarray(c["z_raw"], float), nan=np.inf)
    step = float(s[1] - s[0]) if n > 1 else 1.0
    blocked = np.zeros(n, bool)
    if mouths is not None:
        blocked = binary_dilation(mouths["L"] | mouths["R"], structure=np.ones(2 * int(MOUTH_CLEAR_M / step) + 1))
    out = []
    for i0, i1 in _runs(c["bridge"]):
        if i1 <= i0:
            continue
        g = gap[i0:i1 + 1]
        finite = g[np.isfinite(g)]
        rec = {"id": len(out), "rows": [i0, i1], "from_m": round(float(s[i0]), 1), "to_m": round(float(s[i1]), 1),
               "length_m": round(float(s[i1] - s[i0]), 1),
               "gap_max_m": round(float(finite.max()), 2) if len(finite) else None,
               "look": "parapets", "open_rows": None, "piers_m": []}
        if names is not None and "run" in c:
            rec["name"] = names[int(np.asarray(c["run"])[i0])]
        deep = np.flatnonzero(g >= OPEN_GAP_M)
        if len(deep):
            a, b = max(i0 + int(deep[0]), END_ROWS), min(i0 + int(deep[-1]), n - 1 - END_ROWS)
            # Clear of side roads' mouths: the longest stretch between them.
            free = [(p, q) for p, q in _runs(~blocked[a:b + 1])]
            if free:
                p, q = max(free, key=lambda pq: pq[1] - pq[0])
                a, b = a + p, a + q
                if s[b] - s[a] >= MIN_OPEN_M:
                    rec["look"], rec["open_rows"] = "open", [int(a), int(b)]
                    rec["piers_m"] = _piers(s, gap, a, b)
        out.append(rec)
    return out


def _piers(s: np.ndarray, gap: np.ndarray, a: int, b: int) -> list[float]:
    length = float(s[b] - s[a])
    if length <= PIER_SPAN_M or np.nanmax(gap[a:b + 1]) <= PIER_GAP_M:
        return []
    parts = max(int(round(length / PIER_SPAN_M)), 2)
    out = []
    for k in range(1, parts):
        at = float(s[a]) + length * k / parts
        i = int(np.clip(np.searchsorted(s, at), a, b))
        if gap[i] > PIER_GAP_M and s[i] - s[a] >= PIER_END_M and s[b] - s[i] >= PIER_END_M:
            out.append(round(float(s[i]), 1))
    return out


def masks(bridges: list[dict], n: int, mouths: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Where a parapet (and the kerb strip under it) stands on each side ("L", "R"), station by station: every bridge's
    rows, less side roads' mouths on that side."""
    out = {"L": np.zeros(n, bool), "R": np.zeros(n, bool)}
    for b in bridges:
        i0, i1 = b["rows"]
        for side in "LR":
            out[side][i0:i1 + 1] = True
    if mouths is not None:
        out = {side: m & ~np.asarray(mouths[side], bool) for side, m in out.items()}
    return out


def open_rows(bridges: list[dict], n: int) -> np.ndarray:
    """Stations strictly inside an open span: no ground under the deck there, so no batter."""
    out = np.zeros(n, bool)
    for b in bridges:
        if b["open_rows"] is not None:
            a, bb = b["open_rows"]
            out[a + 1:bb] = True
    return out


def interior(mask: np.ndarray) -> np.ndarray:
    """A run mask less each run's first and last station: where a parapet takes over from a guard rail, the two share
    their end station, so the collision line has no gap."""
    out = np.asarray(mask, bool).copy()
    for i0, i1 in _runs(out):
        out[i0] = out[i1] = False
    return out


def layout_intervals(bridges: list[dict], s: np.ndarray, reverse: bool) -> list[tuple[float, float]]:
    """Each bridge's kerb strip as (from, to) metres from a layout's start, for its timing gates' ends
    (`layout.logic_dummies(kerbs=)`)."""
    s = np.asarray(s, float)
    out = []
    for b in bridges:
        i0, i1 = b["rows"]
        a, bb = float(s[i0] - s[0]), float(s[i1] - s[0])
        out.append((float(s[-1] - s[0]) - bb, float(s[-1] - s[0]) - a) if reverse else (a, bb))
    return sorted(out)


# -- the ground --------------------------------------------------------------------------------------------------------

def _normals(heading: np.ndarray) -> np.ndarray:
    return np.column_stack([-np.sin(heading), np.cos(heading)])


def batter_rows(c, bridges: list[dict]) -> np.ndarray:
    """Stations whose ground beside the road takes a 1:2 batter: untagged fills (`filled`, less bridges, tunnels and
    the humps under crossing roads) padded by BATTER_PAD_M, and bridges' closed stretches and approaches, padded by
    BATTER_APPROACH_M; never inside an open span. Where the road isn't above the ground, a batter changes nothing."""
    s = np.asarray(c["s"], float)
    n = len(s)
    step = float(s[1] - s[0]) if n > 1 else 1.0

    def padded(mask, metres):
        if not mask.any():
            return mask
        return binary_dilation(mask, structure=np.ones(2 * max(int(round(metres / step)), 1) + 1))

    fill = np.asarray(c["filled"], bool).copy() if "filled" in c else np.zeros(n, bool)
    for key in ("bridge", "tunnel", "under"):
        if key in c:
            fill &= ~np.asarray(c[key], bool)
    deck = np.zeros(n, bool)
    for b in bridges:
        i0, i1 = b["rows"]
        deck[i0:i1 + 1] = True
    return (padded(fill, BATTER_PAD_M) | padded(deck, BATTER_APPROACH_M)) & ~open_rows(bridges, n)


def embanked(sample, c, sec, rows: np.ndarray, kerb: dict[str, np.ndarray] | None = None):
    """`sample` (a DEM sampler) with the ground beside the road on `rows` raised to a 1:2 batter from the verge's
    edge: `max(ground, verge - BATTER x distance past the verge's edge)`, out to BATTER_REACH_M from the centreline.
    `kerb`: where the verge is a bridge's kerb strip, per side (its edge KERB_M out, KERB_RISE_M up). `sample` itself
    where `rows` is empty."""
    rows = np.asarray(rows, bool)
    if not rows.any():
        return sample
    xy = np.asarray(c["xy"], float)
    nrm = _normals(np.asarray(c["heading"], float))
    z = np.asarray(c["z"], float)
    tree = cKDTree(xy)
    edge = {}
    for side, half, slope in (("L", sec["half_left"], sec["slope_left"]), ("R", sec["half_right"], sec["slope_right"])):
        half, slope = np.asarray(half, float), np.asarray(slope, float)
        edge_h = z - slope * half
        out, h = half + mesh.VERGE_M, edge_h - mesh.VERGE_M * mesh.VERGE_FALL
        if kerb is not None:
            k = np.asarray(kerb[side], bool)
            out, h = np.where(k, half + KERB_M, out), np.where(k, edge_h + KERB_RISE_M, h)
        edge[side] = (out, h)

    def raised(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = np.array(sample(x, y), dtype=float, copy=True)
        pts = np.column_stack([x.ravel(), y.ravel()])
        d, i = tree.query(pts, distance_upper_bound=BATTER_REACH_M, workers=cpus.count())
        near = np.isfinite(d)
        near[near] = rows[i[near]]
        if not near.any():
            return h
        k = i[near]
        lateral = ((pts[near] - xy[k]) * nrm[k]).sum(axis=1)
        # Beside the road only: past either end of the route the nearest station is its last, straight behind.
        along = np.abs(((pts[near] - xy[k]) * nrm[k][:, ::-1] * [1.0, -1.0]).sum(axis=1))
        beside = along <= BATTER_ALONG_M
        near[np.flatnonzero(near)[~beside]] = False
        k, lateral = k[beside], lateral[beside]
        if not len(k):
            return h
        left = lateral > 0
        out = np.where(left, edge["L"][0][k], edge["R"][0][k])
        top = np.where(left, edge["L"][1][k], edge["R"][1][k])
        batter = top - BATTER * np.maximum(np.abs(lateral) - out, 0.0)
        flat = h.reshape(-1)
        flat[near] = np.fmax(flat[near], batter)
        return flat.reshape(h.shape)

    return raised


def under_decks(sample, decks: list[tuple[shapely.Geometry, np.ndarray, np.ndarray]], inside=None):
    """`sample` with the ground inside each deck's zone (polygon, its stations' points, the highest the ground may
    stand at each) kept at or below the nearest station's height; there from `inside` instead, where given (the
    ground without the batters, which belong to the embankment beside the abutment, not under the deck)."""
    if not decks:
        return sample
    zones = [(zone, cKDTree(pts), cap) for zone, pts, cap in decks]
    for zone, _, _ in zones:
        shapely.prepare(zone)

    def capped(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = np.array(sample(x, y), dtype=float, copy=True)
        flat, xs, ys = h.reshape(-1), x.reshape(-1), y.reshape(-1)
        for zone, tree, cap in zones:
            hit = shapely.contains_xy(zone, xs, ys)
            if hit.any():
                _, i = tree.query(np.column_stack([xs[hit], ys[hit]]))
                below = flat[hit] if inside is None else np.asarray(inside(xs[hit], ys[hit]), dtype=float)
                flat[hit] = np.fmin(below, cap[i])
        return flat.reshape(h.shape)

    return capped


# -- the decks ---------------------------------------------------------------------------------------------------------

@dataclass
class _Span:
    """An open span as built: its bridge, its rows, and the geometry the terrain and the meshes share."""

    bridge: dict
    a: int
    b: int
    top: np.ndarray  # (rows, 2) the kerb's top each side, L and R, per row a..b
    cap: np.ndarray  # (rows,) the highest the ground may stand under the deck, per row a..b
    zone: shapely.Geometry  # where the ground is capped: the deck's plan and GUARD_M beyond
    feet: dict = field(default_factory=dict)  # row -> (points (m, 2), heights (m,)): the abutments' feet


class Decks:
    """The route's bridges for one build: the plan (`plan`), and what the ribbon, the terrain and the meshes need.

    Use in the mesh stage's order: `verges` and `embanked` for the ribbon; then `footprint` (the open spans and the
    abutments' faces, for water and terrain); `ground` over the water-lowered DEM; `footprint_points` (the abutments'
    feet and the points under the decks, which need that ground); then `cut` and `kerb` for the road's meshes, and
    `meshes` for the structure."""

    def __init__(self, bridges: list[dict], c, sec, mouths: dict[str, np.ndarray] | None = None):
        self.bridges = bridges
        self.c, self.sec = c, sec
        self.n = len(c["s"])
        self.parapet = masks(bridges, self.n, mouths)
        self.batter = batter_rows(c, bridges)
        self.spans: list[_Span] = []
        self._rb: mesh.Ribbon | None = None
        self.stats: dict = {}

    def __bool__(self) -> bool:
        """Anything to build: a bridge, or a fill to batter."""
        return bool(self.bridges) or bool(self.batter.any())

    def close(self, faults: dict[int, list[str]]) -> None:
        """Build the bridges in `faults` (by id) on the ground with their parapets, saying why (`on_ground`)."""
        for b in self.bridges:
            if b["id"] in faults:
                b.update(look="parapets", open_rows=None, piers_m=[], on_ground=list(faults[b["id"]]))
        self.batter = batter_rows(self.c, self.bridges)

    # The ribbon.
    def verges(self) -> dict:
        """`mesh.build_ribbon`'s `verge_l`/`verge_r`: the kerb strip on each side where a parapet stands."""
        out = {}
        for side in "LR":
            width = np.where(self.parapet[side], KERB_M, np.nan)
            out[f"verge_{side.lower()}"] = (width, np.full(self.n, KERB_RISE_M))
        return out

    def embanked(self, sample):
        """The DEM sampler with 1:2 batters on fills and bridges' closed stretches (`embanked`)."""
        return embanked(sample, self.c, self.sec, self.batter, self.parapet)

    def kerb(self) -> dict[str, np.ndarray]:
        """`mesh.ribbon_meshes(kerb=)`: the quads whose verge is the kerb strip, per side."""
        return {side: m[:-1] & m[1:] for side, m in self.parapet.items()}

    def cut(self, cut: dict[str, np.ndarray] | None) -> dict[str, np.ndarray] | None:
        """`cut` (the side roads' and rings', or None) with the open spans' quads: no verge or skirt there."""
        if not self.spans:
            return cut
        span = mesh.span_quads(self.n, [(sp.a, sp.b) for sp in self.spans])
        base = cut or {"L": np.zeros(self.n - 1, bool), "R": np.zeros(self.n - 1, bool)}
        return {k: np.asarray(base[k], bool) | span for k in "LR"}

    # The terrain.
    def footprint(self, rb: mesh.Ribbon, extra: mesh.Footprint | None) -> mesh.Footprint | None:
        """`extra` (the stubs' and rings' footprint, or None) with each open span out of it (`Footprint.spans`) and
        its abutments' faces in it (`patches`)."""
        self._rb = rb
        self.spans = []
        for b in self.bridges:
            if b["open_rows"] is None:
                continue
            a, bb = b["open_rows"]
            self.spans.append(self._span(rb, b, a, bb))
        if not self.spans:
            return extra
        base = extra if extra is not None else mesh.Footprint(
            {"L": np.zeros(self.n - 1, bool), "R": np.zeros(self.n - 1, bool)}, [], [], [])
        c0, c1 = rb.road_cols.start - 1, rb.road_cols.stop + 1
        patches = []
        for sp in self.spans:
            for row in (sp.a, sp.b):
                top, foot = self._abutment_plan(rb, sp, row)
                patches.append(np.vstack([top, foot[::-1]]))
        return replace(base, spans=list(base.spans or []) + [(sp.a, sp.b, c0, c1) for sp in self.spans],
                       patches=list(base.patches) + patches)

    def _span(self, rb: mesh.Ribbon, b: dict, a: int, bb: int) -> _Span:
        k0, k1 = rb.road_cols.start - 1, rb.road_cols.stop  # the kerb's outer columns, L and R
        rows = slice(a, bb + 1)
        top = np.column_stack([rb.h[rows, k0], rb.h[rows, k1]])
        cap = top.min(axis=1) - DECK_DEPTH_M - CLEAR_M
        outline = np.vstack([self._outer(rb, "L", GUARD_M)[rows], self._outer(rb, "R", GUARD_M)[rows][::-1]])
        zone = shapely.Polygon(outline).buffer(0.05)
        return _Span(b, a, bb, top, cap, zone)

    def _outer(self, rb: mesh.Ribbon, side: str, beyond: float = 0.0) -> np.ndarray:
        """Each row's point on the parapet's outer face (the fascia), `beyond` further out, on `side`."""
        k = rb.road_cols.start - 1 if side == "L" else rb.road_cols.stop
        e = rb.road_cols.start if side == "L" else rb.road_cols.stop - 1
        base = np.column_stack([rb.e[:, k], rb.n[:, k]])
        out = base - np.column_stack([rb.e[:, e], rb.n[:, e]])
        out /= np.maximum(np.hypot(out[:, 0], out[:, 1]), 1e-9)[:, None]
        return base + out * (PARAPET_W + beyond)

    def _abutment_plan(self, rb: mesh.Ribbon, sp: _Span, row: int) -> tuple[np.ndarray, np.ndarray]:
        """An abutment's face in plan: its top, the row's vertices from the skirt's first column on the left to that
        on the right; and its foot, the kerb's and road's vertices moved ABUT_SET_M into the span."""
        c0, c1 = rb.road_cols.start - 1, rb.road_cols.stop + 1
        top = np.column_stack([rb.e[row, c0 - 1:c1 + 1], rb.n[row, c0 - 1:c1 + 1]])
        h = float(self.c["heading"][row])
        t = np.array([math.cos(h), math.sin(h)]) * (1.0 if row == sp.a else -1.0)
        foot = np.column_stack([rb.e[row, c0:c1], rb.n[row, c0:c1]]) + t * ABUT_SET_M
        return top, foot

    def ground(self, sample):
        """The terrain's sampler: `sample` (the DEM, lowered under water) with the batters, and kept CLEAR_M under
        each open deck's underside."""
        raised = self.embanked(sample)
        if not self.spans:
            return raised
        xy = np.asarray(self.c["xy"], float)
        return under_decks(raised, [(sp.zone, xy[sp.a:sp.b + 1], sp.cap) for sp in self.spans], inside=sample)

    def footprint_points(self, fp: mesh.Footprint | None, ground) -> mesh.Footprint | None:
        """`fp` with the abutments' feet and the points under each open deck (`edges`), their heights from `ground`
        (the terrain's own sampler, `ground`)."""
        if not self.spans or fp is None:
            return fp
        rb = self._rb
        edges = []
        # Other pieces of ground (a road drawn under the deck, `core.crossing_roads`; a side road's stub) keep the points
        # under the deck off them: a point inside one would join a triangle reaching across it, dropped, leaving a hole.
        pieces = [shapely.Polygon(np.column_stack([r.e[i, k], r.n[i, k]])) for r in fp.ribbons
                  for i, k in [mesh._outline_index(r)]] + [shapely.Polygon(ring) for ring in fp.patches]
        blocked = shapely.unary_union([g.buffer(0) for g in pieces]).buffer(UNDER_CLEAR_OF_M) if pieces else None
        if blocked is not None:
            shapely.prepare(blocked)
        for sp in self.spans:
            for row in (sp.a, sp.b):
                _, foot = self._abutment_plan(rb, sp, row)
                cap = sp.cap[row - sp.a]
                h = np.fmin(np.asarray(ground(foot[:, 0], foot[:, 1]), float), cap)
                h = np.where(np.isfinite(h), h, cap)
                sp.feet[row] = (foot, h)
                edges.append((foot, h))
            pts = self._under(rb, sp)
            if len(pts) and blocked is not None:
                pts = pts[~shapely.contains_xy(blocked, pts[:, 0], pts[:, 1])]
            if len(pts):
                h = np.asarray(ground(pts[:, 0], pts[:, 1]), float)
                cap = sp.cap[np.clip(self._nearest_rows(sp, pts) - sp.a, 0, sp.b - sp.a)]
                edges.append((pts, np.where(np.isfinite(h), np.fmin(h, cap), cap)))
        return replace(fp, edges=list(fp.edges) + edges)

    def _nearest_rows(self, sp: _Span, pts: np.ndarray) -> np.ndarray:
        xy = np.asarray(self.c["xy"], float)
        _, i = cKDTree(xy[sp.a:sp.b + 1]).query(pts)
        return sp.a + i

    def _under(self, rb: mesh.Ribbon, sp: _Span) -> np.ndarray:
        """Points under an open deck and along its edges, GUARD_M beyond its fascias: every GROUND_STEP_M along it
        from GROUND_END_M past each abutment, and at most GROUND_ACROSS_M apart across it."""
        s = np.asarray(self.c["s"], float)
        lo, hi = s[sp.a] + GROUND_END_M, s[sp.b] - GROUND_END_M
        if hi < lo:
            return np.zeros((0, 2))
        stations = np.linspace(lo, hi, max(int(round((hi - lo) / GROUND_STEP_M)) + 1, 1))
        rows = np.unique(np.clip(np.searchsorted(s, stations), sp.a + 1, sp.b - 1))
        left, right = self._outer(rb, "L", GUARD_M)[rows], self._outer(rb, "R", GUARD_M)[rows]
        width = float(np.hypot(*(left - right).T).max())
        k = max(int(math.ceil(width / GROUND_ACROSS_M)), 1)
        f = np.linspace(0.0, 1.0, k + 1)
        return (left[:, None, :] * (1 - f)[None, :, None] + right[:, None, :] * f[None, :, None]).reshape(-1, 2)

    # The meshes.
    def meshes(self, rb: mesh.Ribbon, frame: LocalFrame, ground, cut: dict[str, np.ndarray] | None = None
               ) -> tuple[list[Mesh], dict]:
        """The bridges' meshes and a summary. `ground` stands the piers; `cut` (the side roads' and rings') keeps the
        kerb out of their mouths' quads."""
        out: list[Mesh] = []
        step = float(self.c["s"][1] - self.c["s"][0]) if self.n > 1 else 1.0
        stats = {"bridges": len(self.bridges), "open": len(self.spans), "piers": 0,
                 "batter_m": round(float(self.batter.sum() * step), 1),
                 "parapet_m": {side: round(float(m.sum()), 1) for side, m in self.parapet.items()}}
        spans = {sp.bridge["id"]: sp for sp in self.spans}
        kerb = self.kerb()
        if cut is not None:
            kerb = {k: kerb[k] & ~np.asarray(cut[k], bool) for k in "LR"}
        for b in self.bridges:
            k = b["id"]
            i0, i1 = b["rows"]
            sp = spans.get(k)
            concrete, under, steel, colliders = [], [], [], []
            for side in "LR":
                quads = np.zeros(self.n - 1, bool)
                quads[i0:i1] = kerb[side][i0:i1]
                for q0, q1 in _runs(quads):
                    out.append(_kerb_mesh(f"1ROAD_BK{k}{side}" + (f"_{q0}" if len(_runs(quads)) > 1 else ""),
                                          rb, side, q0, q1 + 1, frame))
                rows = np.zeros(self.n, bool)
                rows[i0:i1 + 1] = self.parapet[side][i0:i1 + 1]
                for r0, r1 in _runs(rows):
                    if r1 <= r0:
                        continue
                    face, rail, coll = self._parapet(rb, side, r0, r1, sp)
                    concrete += face
                    steel += rail
                    colliders.append((f"{COLLIDER_PREFIX}{k}{side}" + (f"_{r0}" if len(_runs(rows)) > 1 else ""),
                                      coll))
            if sp is not None:
                under.append(self._underside(rb, sp))
                for row in (sp.a, sp.b):
                    concrete.append(self._abutment(rb, sp, row))
                for at in b["piers_m"]:
                    concrete += self._pier(rb, sp, at, ground)
                    stats["piers"] += 1
            out += decks.meshes(f"BRIDGE_{k}", MATERIAL, concrete, frame)
            out += decks.meshes(f"BRIDGE_{k}_U", UNDERSIDE_MATERIAL, under, frame)
            out += decks.meshes(f"BRIDGE_{k}_RAIL", RAIL_MATERIAL, steel, frame)
            for name, (line, base) in colliders:
                out.append(_collider(name, line, base, frame))
        stats["meshes"] = len(out)
        stats["vertices"] = int(sum(len(m.positions) for m in out))
        return out, stats

    def _parapet(self, rb: mesh.Ribbon, side: str, r0: int, r1: int, sp: _Span | None):
        """One run of parapet on `side`, rows r0..r1: its concrete (the generic barrier swept along the kerb's outer
        edge, `decks.sweep`, and the fascia under its outer face), its rail and posts, and its collider's line and
        base."""
        k = rb.road_cols.start - 1 if side == "L" else rb.road_cols.stop
        sign = 1.0 if side == "L" else -1.0  # the barrier stands outwards: to the left of the road on its left
        rows = np.arange(r0, r1 + 1)
        base = np.column_stack([rb.e[rows, k], rb.n[rows, k]])
        hb = rb.h[rows, k]
        depth = np.full(len(rows), EDGE_BEAM_M)
        if sp is not None:
            depth[(rows >= sp.a) & (rows <= sp.b)] = DECK_DEPTH_M
        s = np.asarray(self.c["s"], float)[rows]
        concrete = [decks.sweep(base, hb, [(sign * u, v) for u, v in decks.BARRIER_PROFILE])]
        # The fascia: the barrier's outer face carried on down below the kerb, the open span's deeper.
        outer = decks.offset(base, sign * PARAPET_W)
        _, left = decks.frames(base)
        out = np.column_stack([decks._unit(left) * sign, np.zeros(len(rows))])
        concrete.append(_face(np.column_stack([outer, hb]), np.column_stack([outer, hb - depth]), out, s,
                              (hb, hb - depth)))
        # The steel rail along the barrier's top, a point every RAIL_STRIDE rows (and the last), on posts.
        j = np.unique(np.append(np.arange(0, len(rows), RAIL_STRIDE), len(rows) - 1))
        mid, w = sign * PARAPET_W / 2, RAIL_W / 2
        steel = [decks.sweep(base[j], hb[j], ((mid - w, RAIL[0]), (mid + w, RAIL[0]), (mid + w, RAIL[1]),
                                               (mid - w, RAIL[1])))] if len(j) > 1 else []
        posts_line = decks.offset(base, mid)
        tangent, _ = decks.frames(base)
        length = s[-1] - s[0]
        count = max(int(round(length / RAIL_POST_SPACING_M)), 1)
        for at in np.linspace(s[0] + 0.2, s[-1] - 0.2, count + 1) if length > 0.5 else []:
            i = int(np.clip(np.searchsorted(s, at), 0, len(s) - 1))
            steel.append(decks.box(posts_line[i], tangent[i], RAIL_POST_W, RAIL_POST_W, float(hb[i] + PARAPET_H),
                                   float(hb[i] + RAIL[0])))
        return concrete, steel, (base, hb)

    def _underside(self, rb: mesh.Ribbon, sp: _Span):
        """The open span's underside, flat across between the fascias' feet (the barriers' outer faces carried on down,
        offset along the whole bridge's kerb lines, as its parapets are)."""
        i0, i1 = sp.bridge["rows"]
        rows = np.arange(sp.a, sp.b + 1)
        edge = {}
        for side, k, sign in (("L", rb.road_cols.start - 1, 1.0), ("R", rb.road_cols.stop, -1.0)):
            line = np.column_stack([rb.e[i0:i1 + 1, k], rb.n[i0:i1 + 1, k]])
            edge[side] = decks.offset(line, sign * PARAPET_W)[rows - i0]
        left, right = edge["L"], edge["R"]
        hl, hr = sp.top[:, 0] - DECK_DEPTH_M, sp.top[:, 1] - DECK_DEPTH_M
        s = np.asarray(self.c["s"], float)[rows]
        down = np.tile([0.0, 0.0, -1.0], (len(rows), 1))
        return _face(np.column_stack([left, hl]), np.column_stack([right, hr]), down, s, None)

    def _abutment(self, rb: mesh.Ribbon, sp: _Span, row: int):
        """The abutment's face at `row`: from the row's vertices, the skirt's first column on each side and the kerb
        and road between, down to its foot (`footprint_points`), facing into the span."""
        c0, c1 = rb.road_cols.start - 1, rb.road_cols.stop + 1
        cols = np.arange(c0 - 1, c1 + 1)
        top = np.column_stack([rb.e[row, cols], rb.n[row, cols], rb.h[row, cols]])
        foot, fh = sp.feet[row]
        bottom = np.vstack([top[:1], np.column_stack([foot, fh]), top[-1:]])
        h = float(self.c["heading"][row])
        t = np.array([math.cos(h), math.sin(h), 0.0]) * (1.0 if row == sp.a else -1.0)
        m = len(top)
        pos = np.vstack([top, bottom])
        # Quads between the top and the bottom, row j to j + 1; the first and last are triangles, since the bottom's
        # first and last points are the top's.
        tri = [(0, 1, m + 1)]
        for j in range(1, m - 2):
            tri += [(j, j + 1, m + j + 1), (j, m + j + 1, m + j)]
        tri.append((m - 2, m - 1, 2 * m - 2))
        nrm = np.tile(t, (len(pos), 1))
        along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(top[:, :2], axis=0).T))])
        uv = np.column_stack([np.concatenate([along, along]) / CONCRETE_TILE_M, pos[:, 2] / CONCRETE_TILE_M])
        return _oriented(pos, nrm, uv, np.array(tri, np.int64))

    def _pier(self, rb: mesh.Ribbon, sp: _Span, at: float, ground) -> list:
        s = np.asarray(self.c["s"], float)
        i = int(np.clip(np.searchsorted(s, at), sp.a + 1, sp.b - 1))
        left, right = self._outer(rb, "L")[i], self._outer(rb, "R")[i]
        width = float(np.hypot(*(left - right)))
        under = float(min(sp.top[i - sp.a]) - DECK_DEPTH_M)
        h = float(self.c["heading"][i])
        across = (left - right) / max(width, 1e-9)
        mid = (left + right) / 2
        along = np.array([math.cos(h), math.sin(h)])
        pieces = [decks.box(mid, along, PIER_HEADSTOCK[1], width - 2 * 0.3, under - PIER_HEADSTOCK[0], under)]
        count = 3 if width >= PIER_COLUMNS_WIDE_M else 2
        reach = max(width / 2 - PIER_EDGE_M, 0.5)
        foot_top = under - PIER_HEADSTOCK[0]
        for off in np.linspace(-reach, reach, count):
            centre = mid + across * off
            g = np.asarray(ground(*_ring(centre, 1.0, 8).T), float)
            low = float(np.nanmin(g)) if np.isfinite(g).any() else foot_top - 2.0
            radius = float(np.clip(0.4 + 0.012 * (foot_top - low), 0.45, 1.0))
            g = np.asarray(ground(*_ring(centre, radius + 0.3, 8).T), float)
            low = min(low, float(np.nanmin(g)) if np.isfinite(g).any() else low)
            if foot_top - low > 0.3:
                pieces.append(decks.prism(centre, radius, low - PIER_BURY_M, foot_top))
        return pieces

    def records(self, rb: mesh.Ribbon) -> list[dict]:
        """The plan with each bridge's geometry for the later stages' checks: its parapets' faces (the kerb's outer
        edge, region coordinates, per side) and, for an open deck, its plan and underside."""
        spans = {sp.bridge["id"]: sp for sp in self.spans}
        out = []
        for b in self.bridges:
            rec = dict(b)
            i0, i1 = b["rows"]
            faces = {}
            for side in "LR":
                k = rb.road_cols.start - 1 if side == "L" else rb.road_cols.stop
                rows = np.flatnonzero(self.parapet[side][i0:i1 + 1]) + i0
                faces[side] = [[round(float(rb.e[r, k]), 3), round(float(rb.n[r, k]), 3), round(float(rb.h[r, k]), 3)]
                               for r in rows]
            rec["parapets"] = faces
            sp = spans.get(b["id"])
            if sp is not None:
                rows = np.arange(sp.a, sp.b + 1)
                left, right = self._outer(rb, "L")[rows], self._outer(rb, "R")[rows]
                rec["deck"] = {
                    "left": [[round(float(x), 3), round(float(y), 3)] for x, y in left],
                    "right": [[round(float(x), 3), round(float(y), 3)] for x, y in right],
                    "underside": [round(float(v), 3) for v in sp.top.min(axis=1) - DECK_DEPTH_M]}
            elif b["look"] == "open":
                rec["look"] = "parapets"
            out.append(rec)
        return out


# -- mesh pieces -------------------------------------------------------------------------------------------------------
# The generic style's pieces are `core.decks`'s (the barrier, rail, posts, headstocks and columns the bridges over the
# route have); these are the shapes only the route's own deck has (its fascia, underside and abutments' faces), as
# `decks.Piece`s, each triangle wound so it faces the way its vertices' normals point.

def _oriented(pos, nrm, uv, tri) -> decks.Piece:
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    flip = (np.cross(p1 - p0, p2 - p0) * nrm[tri].mean(axis=1)).sum(axis=1) < 0
    tri = tri.copy()
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return decks.Piece(pos, nrm, uv, np.asarray(tri, np.int64))


def _face(a: np.ndarray, b: np.ndarray, normal: np.ndarray, s: np.ndarray, heights) -> decks.Piece:
    """A strip along the bridge between two lines of points (n, 3), each pair a row, facing `normal` (n, 3). Its
    texture runs along the bridge by `s`, and across it by `heights` (each line's heights) or by its width."""
    n = len(a)
    pos = np.empty((2 * n, 3))
    pos[0::2], pos[1::2] = a, b
    nrm = np.repeat(np.asarray(normal, float), 2, axis=0)
    if heights is None:
        across = np.hypot(*(b[:, :2] - a[:, :2]).T)
        v = np.column_stack([np.zeros(n), across])
    else:
        v = np.column_stack([np.asarray(heights[0], float), np.asarray(heights[1], float)])
    uv = np.column_stack([np.repeat(s, 2), v.ravel()]) / CONCRETE_TILE_M
    i = np.arange(n - 1) * 2
    tri = np.concatenate([np.column_stack([i, i + 2, i + 1]), np.column_stack([i + 1, i + 2, i + 3])])
    return _oriented(pos, nrm, uv, tri)


def _ring(centre: np.ndarray, radius: float, sides: int) -> np.ndarray:
    a = np.linspace(0.0, 2 * math.pi, sides, endpoint=False)
    return np.asarray(centre, float) + radius * np.column_stack([np.cos(a), np.sin(a)])


def _kerb_mesh(name: str, rb: mesh.Ribbon, side: str, r0: int, r1: int, frame: LocalFrame) -> Mesh:
    """The kerb strip on `side`, rows r0..r1: the ribbon's road edge and verge columns, so it shares their vertices."""
    cols = [rb.road_cols.start - 1, rb.road_cols.start] if side == "L" else [rb.road_cols.stop - 1, rb.road_cols.stop]
    sl = slice(r0, r1 + 1)
    pos = frame.to_ac(rb.e[sl][:, cols], rb.n[sl][:, cols], rb.h[sl][:, cols]).reshape(-1, 3)
    tri = mesh._grid_triangles(r1 - r0 + 1, 2)
    return mesh._mesh(name, KERB_MATERIAL, pos, tri, pos[:, [0, 2]] / CONCRETE_TILE_M)


def _collider(name: str, line: np.ndarray, base: np.ndarray, frame: LocalFrame) -> Mesh:
    """An invisible wall on the parapet's face, COLLIDER below and above the kerb, both windings."""
    n = len(line)
    bottom = frame.to_ac(line[:, 0], line[:, 1], base + COLLIDER[0])
    top = frame.to_ac(line[:, 0], line[:, 1], base + COLLIDER[1])
    pos = np.empty((2 * n, 3))
    pos[0::2], pos[1::2] = bottom, top
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    uv = np.column_stack([np.repeat(dist / 4.0, 2), np.tile([1.0, 0.0], n)])
    i = np.arange(n - 1) * 2
    front = np.concatenate([np.column_stack([i, i + 2, i + 1]), np.column_stack([i + 1, i + 2, i + 3])])
    tri = np.concatenate([front, front[:, [0, 2, 1]]])
    d = np.diff(line, axis=0)
    t = np.vstack([d, d[-1:]])
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    side = np.repeat(np.column_stack([-t[:, 1], np.zeros(n), -t[:, 0]]), 2, axis=0)  # horizontal, in AC axes
    return Mesh(name, RAIL_MATERIAL, pos.astype(np.float32), side.astype(np.float32), uv.astype(np.float32),
                tri.astype(np.int64), renderable=False)


# -- reflectors --------------------------------------------------------------------------------------------------------

def reflectors(decks: Decks, rb: mesh.Ribbon, frame: LocalFrame, atlas) -> list[Mesh]:
    """A reflector on a short post on each parapet's top every REFLECTOR_SPACING_M, facing both ways: red seen by the
    traffic it's on the left of, white by the traffic it's on the right of, as guide posts' (`core.furniture`). On the
    furniture's atlas, with its materials, so CSP brightens them at night as it does the posts'."""
    from mapcreator.core import furniture

    if not decks:
        return []
    builder = furniture._Builder(atlas)
    s = np.asarray(decks.c["s"], float)
    heading = np.asarray(decks.c["heading"], float)
    w, d, hgt = REFLECTOR_POST
    rw, rh = REFLECTOR_SIZE
    for side in "LR":
        k = rb.road_cols.start - 1 if side == "L" else rb.road_cols.stop
        outer = decks._outer(rb, side)
        for r0, r1 in _runs(decks.parapet[side]):
            if s[r1] - s[r0] < 2.0:
                continue
            for at in np.arange(s[r0] + 1.0, s[r1] - 0.5, REFLECTOR_SPACING_M):
                i = int(np.clip(np.searchsorted(s, at), r0, r1))
                e, n = (np.array([rb.e[i, k], rb.n[i, k]]) + outer[i]) / 2
                top = float(rb.h[i, k]) + PARAPET_H
                along = np.array([math.cos(heading[i]), math.sin(heading[i])])
                builder._box(e, n, top, top + hgt, along, w, d, ("white", "white", "white"))
                front, back = (("reflector_red", "reflector_white") if side == "L"
                               else ("reflector_white", "reflector_red"))
                a3 = np.array([along[0], along[1], 0.0])
                x = np.array([-along[1], along[0], 0.0]) * rw / 2
                lo, hi = np.array([0, 0, top + hgt - 0.03 - rh]), np.array([0, 0, top + hgt - 0.03])
                for facing, cell in ((-1.0, front), (1.0, back)):
                    cc = np.array([e, n, 0.0]) + a3 * facing * (d / 2 + furniture.REFLECTOR_GAP_M)
                    sx = x * -facing
                    builder._quad([cc + sx + lo, cc - sx + lo, cc - sx + hi, cc + sx + hi], a3 * facing, cell,
                                  furniture.REFLECTOR_MATERIAL)
    out = builder.meshes(frame)
    for i, m in enumerate(out):
        m.name = f"FURNITURE_BRIDGE_{i}"
    return out


# -- checks ------------------------------------------------------------------------------------------------------------

def checks(records: list[dict], frame: LocalFrame) -> list:
    """Each bridge as the validator's checks see it (`validate.BridgeCheck`), in AC axes."""
    from mapcreator.core import validate

    out = []
    for r in records:
        name = r.get("name") or "an unnamed bridge"
        label = f"{name} at {r['from_m'] / 1000:.2f} km ({r['length_m']:.0f} m)"
        faces = []
        for side in "LR":
            p = np.asarray(r["parapets"][side], float).reshape(-1, 3)
            if len(p) >= 2:
                faces.append((side, frame.to_ac(p[:, 0], p[:, 1], p[:, 2])[:, [0, 2]]))
        deck, under = None, None
        if r.get("deck"):
            left, right = np.asarray(r["deck"]["left"], float), np.asarray(r["deck"]["right"], float)
            under_h = np.asarray(r["deck"]["underside"], float)
            # Less its first and last metre, where the abutment's face and its foot are, and the parapets' own
            # footprint, whose solid hides what's under it.
            if len(left) > 4:
                ring = np.vstack([left[1:-1], right[1:-1][::-1]])
                deck = shapely.Polygon(frame.to_ac(ring[:, 0], ring[:, 1], np.zeros(len(ring)))[:, [0, 2]]
                                       ).buffer(0).buffer(-PARAPET_W - 0.05)
                deck = None if deck.is_empty else deck
                mid = (left + right) / 2
                under = (frame.to_ac(mid[:, 0], mid[:, 1], np.zeros(len(mid)))[:, [0, 2]], under_h - frame.h0)
        out.append(validate.BridgeCheck(label, tuple(faces), deck, under))
    return out


def faults(records: list[dict], ground: list[Mesh], frame: LocalFrame) -> dict[int, list[str]]:
    """What's wrong with each open bridge as built, by its id, for those with anything (`validate.bridge_failures` on
    the meshes just built): the mesh stage builds those on the ground."""
    from mapcreator.core import validate

    chk = checks(records, frame)
    found = validate.bridge_failures(validate.mesh_nodes(ground), [(r["id"], c) for r, c in zip(records, chk)])
    return {k: v for k, v in found.items() if v}
