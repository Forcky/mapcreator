"""Bridge pieces in the one generic style: a concrete beam deck, piers and abutments, a concrete barrier with one
steel rail or a footbridge's steel railing, ballast and rails, and the invisible colliders of piers and abutments.

The maintainer chose one generic style (flyovers research Q3, 17 Nov 2026): a concrete beam deck with a concrete
barrier and one steel rail. Footbridges are narrower and carry a railing instead; rail bridges carry ballast and
rails. Shapes are made in region coordinates (easting, northing, height) as `Piece`s and packed into
`scene.Mesh`es by material (`meshes`).

**Shapes.** Most pieces are a cross-section swept along a line (`sweep`): a deck is a rectangle, a barrier a
tapered block, a rail a small bar. Each face has its own vertices, so every edge is sharp, and every triangle is
wound so it faces outwards (`kn5.py`: front faces wind counter-clockwise, seen from outside). Boxes (columns,
headstocks, abutments) are short sweeps with end caps (`box`). Colliders (`collider_box`) are the four vertical faces
of a box, in both windings like the guard rails' (`roadside._strip`), and `renderable=False`.

Nothing here knows where a structure stands or what it carries: `core.rail_and_paths` decides that. The style's
numbers and materials are stage 2b's (`core.crossing_roads`, which draws a road's bridge over the route), so every
bridge over the route has the same barrier, rail, columns and concrete; its road deck is built there, with its own
cross-section. These pieces are generic, for the route's own bridge (stage 1) too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mapcreator.core import crossing_roads as style
from mapcreator.core.scene import MAX_VERTICES, LocalFrame, Mesh

UV_M = 2.0  # one texture repeat every this many metres, along and across

# The generic style's parts, in metres, as stage 2b's decks have them (`core.crossing_roads`): a concrete barrier
# 0.35 m wide and 0.8 m high, and one steel rail 0.1 m square, its top 1.1 m up, on posts every 2.5 m; columns of
# eight sides under a headstock.
BARRIER_PROFILE = ((0.0, 0.0), (style.PARAPET_W, 0.0), (style.PARAPET_W, style.PARAPET_H),
                   (0.0, style.PARAPET_H))  # (inwards, up), its outer face flush with the deck's edge
BARRIER_RAIL = (style.RAIL_SIZE, style.RAIL_SIZE)  # the one steel rail on top: width, height
BARRIER_RAIL_ABOVE_M = style.RAIL_TOP - style.RAIL_SIZE - style.PARAPET_H  # its underside above the barrier's top
BARRIER_POST_M = style.RAIL_POST_M
BARRIER_POST_W = style.POST_SIZE
COLUMN_R = style.PIER_R
COLUMN_SIDES = style.PIER_SIDES
RAILING_H = 1.3  # a footbridge's railing: Austroads asks for 1.3-1.4 m where cyclists ride (estimate)
RAILING_BARS = (0.1, 0.65, 1.3)  # rails at these heights
RAILING_BAR_W = 0.05
RAILING_POST_M = 2.0
RAILING_POST_W = 0.06
RAIL_PROFILE = (0.07, 0.15)  # a running rail: width, height
BALLAST = (3.0, 4.2, 0.35)  # ballast bed: top width, base width, height


@dataclass
class Piece:
    """A shape in region coordinates: positions and unit normals (n, 3) as easting, northing, height; texture
    coordinates (n, 2); triangles (m, 3) wound to face outwards."""

    pos: np.ndarray
    nrm: np.ndarray
    uv: np.ndarray
    tri: np.ndarray


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def frames(line: np.ndarray, max_miter: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Per point of a plan line (k, 2): the unit tangent and the left offset vector, mitred at the corners (so a
    profile keeps its width round a bend), no longer than `max_miter`."""
    line = np.asarray(line, float)
    seg = _unit(np.diff(line, axis=0))
    t = np.vstack([seg[:1], seg[:-1] + seg[1:], seg[-1:]]) if len(seg) > 1 else np.vstack([seg, seg])
    t = _unit(t)
    left = np.column_stack([-t[:, 1], t[:, 0]])
    seg_left = np.column_stack([-seg[:, 1], seg[:, 0]])
    cos = np.ones(len(line))
    if len(seg) > 1:
        cos[1:-1] = np.einsum("ij,ij->i", left[1:-1], seg_left[1:])
    return t, left / np.clip(cos, 1.0 / max_miter, 1.0)[:, None]


def _quad_sign(a, b, c, outward) -> float:
    return float(np.dot(np.cross(b - a, c - a), outward))


def sweep(line: np.ndarray, base: np.ndarray, profile, *, caps: bool = True, uv_m: float = UV_M) -> Piece:
    """A closed cross-section swept along a plan line.

    `line` (k, 2) is the plan, `base` (k,) the height of the profile's origin at each point. `profile` is a closed
    polygon of (left offset, height above base) points, in either order. Each profile edge becomes a strip of
    quads facing outwards; `caps` closes both ends with the polygon (a fan, so the profile must be convex)."""
    line = np.asarray(line, float)
    base = np.asarray(base, float)
    prof = np.asarray(profile, float)
    prof = prof[np.hypot(*(prof - np.roll(prof, 1, axis=0)).T) > 1e-9]  # no repeated point, closing one included
    t, left = frames(line)
    k, m = len(line), len(prof)
    centroid = prof.mean(axis=0)
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    # The 3D point of profile vertex j at station i.
    pts = np.empty((k, m, 3))
    pts[:, :, :2] = line[:, None, :] + left[:, None, :] * prof[None, :, 0, None]
    pts[:, :, 2] = base[:, None] + prof[None, :, 1]
    flat_left = np.column_stack([-t[:, 1], t[:, 0]])
    pos, nrm, uv, tri = [], [], [], []
    count = 0
    across = 0.0
    for j in range(m):
        j2 = (j + 1) % m
        d = prof[j2] - prof[j]
        length = float(np.hypot(*d))
        if length < 1e-9:
            continue
        n2 = np.array([d[1], -d[0]]) / length
        if np.dot(n2, (prof[j] + prof[j2]) / 2 - centroid) < 0:
            n2 = -n2
        n3 = _unit(np.column_stack([flat_left * n2[0], np.full(k, n2[1])]))
        p = np.concatenate([pts[:, j], pts[:, j2]])
        pos.append(p)
        nrm.append(np.concatenate([n3, n3]))
        uv.append(np.column_stack([np.concatenate([along, along]) / uv_m,
                                   np.concatenate([np.full(k, across), np.full(k, across + length)]) / uv_m]))
        across += length
        i = np.arange(k - 1)
        a, b, c, dd = i, i + 1, k + i + 1, k + i
        quads = np.column_stack([a, b, c, a, c, dd]).reshape(-1, 3)
        # Wind every quad to face its outward normal; a strip's quads all turn the same way, so one test does.
        q = int(np.argmax(np.hypot(*np.diff(line, axis=0).T))) if k > 1 else 0
        if _quad_sign(p[a[q]], p[b[q]], p[c[q]], n3[q]) < 0:
            quads = quads[:, [0, 2, 1]]
        tri.append(quads + count)
        count += 2 * k
    if caps:
        for i, sign in ((0, -1.0), (k - 1, 1.0)):
            n3 = np.array([t[i, 0] * sign, t[i, 1] * sign, 0.0])
            p = pts[i]
            fan = np.column_stack([np.zeros(m - 2, int), np.arange(1, m - 1), np.arange(2, m)])
            # The whole polygon's winding (Newell's normal), so a thin first triangle can't mislead it.
            newell = np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
            if float(newell @ n3) < 0:
                fan = fan[:, [0, 2, 1]]
            pos.append(p)
            nrm.append(np.tile(n3, (m, 1)))
            uv.append(np.column_stack([prof[:, 0], prof[:, 1]]) / uv_m)
            tri.append(fan + count)
            count += m
    return Piece(np.concatenate(pos), np.concatenate(nrm), np.concatenate(uv), np.concatenate(tri).astype(np.int64))


def box(centre: np.ndarray, heading: np.ndarray, length: float, width: float, z0: float, z1: float) -> Piece:
    """A box on plan point `centre`, `length` along the unit plan vector `heading`, `width` across it, from height
    z0 to z1."""
    centre, heading = np.asarray(centre, float), _unit(np.asarray(heading, float))
    line = np.vstack([centre - heading * length / 2, centre + heading * length / 2])
    w = width / 2
    return sweep(line, np.array([z0, z0]), ((-w, 0.0), (w, 0.0), (w, z1 - z0), (-w, z1 - z0)))


def collider_box(centre: np.ndarray, heading: np.ndarray, length: float, width: float, z0: float, z1: float
                 ) -> Piece:
    """The four vertical faces of a box, in both windings, for an invisible wall round a pier or an abutment."""
    centre, h = np.asarray(centre, float), _unit(np.asarray(heading, float))
    side = np.array([-h[1], h[0]])
    corners = [centre + h * a * length / 2 + side * b * width / 2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    pos, nrm, uv, tri = [], [], [], []
    for f in range(4):
        p0, p1 = corners[f], corners[(f + 1) % 4]
        out = _unit(np.array([(p1 - p0)[1], -(p1 - p0)[0]]))
        if np.dot(out, (p0 + p1) / 2 - centre) < 0:
            out = -out
        quad = np.array([[*p0, z0], [*p1, z0], [*p1, z1], [*p0, z1]])
        n = 4 * f
        front = np.array([[0, 1, 2], [0, 2, 3]])
        if _quad_sign(quad[0], quad[1], quad[2], np.array([*out, 0.0])) < 0:
            front = front[:, [0, 2, 1]]
        pos.append(quad)
        nrm.append(np.tile([*out, 0.0], (4, 1)))
        uv.append(np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]))
        tri.append(np.concatenate([front, front[:, [0, 2, 1]]]) + n)
    return Piece(np.concatenate(pos), np.concatenate(nrm), np.concatenate(uv), np.concatenate(tri).astype(np.int64))


def prism(centre: np.ndarray, radius: float, z0: float, z1: float, sides: int = COLUMN_SIDES) -> Piece:
    """A round column (a prism of `sides` sides) on plan point `centre`, from height z0 to z1, closed at both ends."""
    theta = np.arange(sides) * 2 * np.pi / sides
    ring = np.asarray(centre, float) + radius * np.column_stack([np.cos(theta), np.sin(theta)])
    # A vertical sweep is a horizontal one turned up: build it as faces directly.
    pos, nrm, uv, tri, count = [], [], [], [], 0
    for i in range(sides):
        p0, p1 = ring[i], ring[(i + 1) % sides]
        out = _unit(np.array([(p1 - p0)[1], -(p1 - p0)[0]]))
        if np.dot(out, (p0 + p1) / 2 - centre) < 0:
            out = -out
        quad = np.array([[*p0, z0], [*p1, z0], [*p1, z1], [*p0, z1]])
        front = np.array([[0, 1, 2], [0, 2, 3]])
        if _quad_sign(quad[0], quad[1], quad[2], np.array([*out, 0.0])) < 0:
            front = front[:, [0, 2, 1]]
        pos.append(quad)
        nrm.append(np.tile([*out, 0.0], (4, 1)))
        uv.append(np.array([[i, z0], [i + 1, z0], [i + 1, z1], [i, z1]], float) / UV_M)
        tri.append(front + count)
        count += 4
    for z, up in ((z1, 1.0), (z0, -1.0)):
        cap = np.column_stack([ring, np.full(sides, z)])
        fan = np.column_stack([np.zeros(sides - 2, int), np.arange(1, sides - 1), np.arange(2, sides)])
        if _quad_sign(cap[0], cap[1], cap[2], np.array([0.0, 0.0, up])) < 0:
            fan = fan[:, [0, 2, 1]]
        pos.append(cap)
        nrm.append(np.tile([0.0, 0.0, up], (sides, 1)))
        uv.append(ring / UV_M)
        tri.append(fan + count)
        count += sides
    return Piece(np.concatenate(pos), np.concatenate(nrm), np.concatenate(uv), np.concatenate(tri).astype(np.int64))


def collider_prism(centre: np.ndarray, radius: float, z0: float, z1: float, sides: int = COLUMN_SIDES) -> Piece:
    """The sides of a round column, in both windings, for an invisible wall round it (as stage 2b's piers have)."""
    p = prism(centre, radius, z0, z1, sides)
    keep = np.abs(p.nrm[p.tri[:, 0], 2]) < 0.5  # the sides, not the caps
    tri = p.tri[keep]
    return Piece(p.pos, p.nrm, p.uv, np.concatenate([tri, tri[:, [0, 2, 1]]]))


def underside(piece: Piece) -> tuple[Piece, Piece]:
    """A piece split into its faces looking down (a deck's underside, for the darker soffit material) and the rest."""
    down = piece.nrm[piece.tri[:, 0], 2] < -0.99
    return (Piece(piece.pos, piece.nrm, piece.uv, piece.tri[~down]),
            Piece(piece.pos, piece.nrm, piece.uv, piece.tri[down]))


def offset(line: np.ndarray, by: float) -> np.ndarray:
    """A plan line moved `by` metres to its left (negative: right), mitred at the corners."""
    _, left = frames(line)
    return np.asarray(line, float) + left * by


def stations(line: np.ndarray, every_m: float) -> np.ndarray:
    """Points along a plan line about `every_m` apart, its ends and corners included."""
    line = np.asarray(line, float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    if s[-1] <= 0:
        return line[:1]
    extra = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / every_m)), 1) + 1)
    at = np.unique(np.concatenate([s, extra]))
    return np.column_stack([np.interp(at, s, line[:, 0]), np.interp(at, s, line[:, 1])])


def deck(line: np.ndarray, top: np.ndarray, width: float, depth: float) -> Piece:
    """A beam deck: a rectangle `width` across and `depth` deep, its top surface at `top` along the plan `line`."""
    w = width / 2
    return sweep(line, top, ((-w, -depth), (w, -depth), (w, 0.0), (-w, 0.0)))


def barrier(line: np.ndarray, top: np.ndarray, width: float) -> tuple[list[Piece], list[Piece]]:
    """The generic parapet on both edges of a deck: a concrete barrier with one steel rail on posts. Returns the
    (concrete, steel) pieces."""
    concrete, steel = [], []
    for side in (1.0, -1.0):
        # The profile runs inwards from the deck's edge: on the left that's to the right, and the other way round.
        prof = [(side * (width / 2 - u), v) for u, v in BARRIER_PROFILE]
        concrete.append(sweep(line, top, prof))
        bar_w, bar_h = BARRIER_RAIL
        top_h = BARRIER_PROFILE[-1][1]
        mid = side * (width / 2 - style.PARAPET_W / 2)  # centred on the barrier's top
        z0 = top_h + BARRIER_RAIL_ABOVE_M
        steel.append(sweep(line, top, ((mid - bar_w / 2, z0), (mid + bar_w / 2, z0), (mid + bar_w / 2, z0 + bar_h),
                                       (mid - bar_w / 2, z0 + bar_h))))
        steel += _posts(offset(line, mid), top, BARRIER_POST_M, BARRIER_POST_W, top_h, z0)
    return concrete, steel


def railing(line: np.ndarray, top: np.ndarray, width: float) -> list[Piece]:
    """A footbridge's steel railing on both edges of its deck: posts and three rails."""
    out = []
    for side in (1.0, -1.0):
        mid = side * (width / 2 - RAILING_POST_W)
        edge = offset(line, mid)
        for h in RAILING_BARS:
            w = RAILING_BAR_W / 2
            out.append(sweep(edge, top, ((-w, h - w), (w, h - w), (w, h + w), (-w, h + w))))
        out += _posts(edge, top, RAILING_POST_M, RAILING_POST_W, 0.0, RAILING_H)
    return out


def _posts(line: np.ndarray, top: np.ndarray, every_m: float, w: float, z0: float, z1: float) -> list[Piece]:
    """Small square posts along a line, from z0 to z1 above the deck's top there."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    if s[-1] < every_m / 2:
        return []
    at = np.linspace(0.0, s[-1], max(int(round(s[-1] / every_m)), 1) + 1)
    xy = np.column_stack([np.interp(at, s, line[:, 0]), np.interp(at, s, line[:, 1])])
    h = np.interp(at, s, top)
    t, _ = frames(line)
    tx, ty = np.interp(at, s, t[:, 0]), np.interp(at, s, t[:, 1])
    return [box(p, np.array([a, b]), w, w, z + z0, z + z1) for p, a, b, z in zip(xy, tx, ty, h)]


def ballast(line: np.ndarray, base: np.ndarray, *, below_m: float = 0.0) -> Piece:
    """A ballast bed on `base` along the plan line: BALLAST's trapezoid, reaching `below_m` under its base (on the
    ground, so the bed sits in it rather than on it)."""
    top_w, base_w, h = BALLAST
    return sweep(line, base, ((-base_w / 2, -below_m), (base_w / 2, -below_m), (base_w / 2, 0.0),
                              (top_w / 2, h), (-top_w / 2, h), (-base_w / 2, 0.0)))


def running_rails(line: np.ndarray, rail_base: np.ndarray, gauge_m: float) -> list[Piece]:
    """The two running rails, their insides `gauge_m` apart, their foot at `rail_base`."""
    w, h = RAIL_PROFILE
    out = []
    for side in (1.0, -1.0):
        inner = side * gauge_m / 2
        outer = inner + side * w
        lo, hi = sorted((inner, outer))
        out.append(sweep(line, rail_base, ((lo, 0.0), (hi, 0.0), (hi, h), (lo, h))))
    return out


def _compact(piece: Piece) -> Piece:
    """A piece with only the vertices its triangles use."""
    used, local = np.unique(piece.tri, return_inverse=True)
    return Piece(piece.pos[used], piece.nrm[used], piece.uv[used], local.reshape(piece.tri.shape).astype(np.int64))


def merge(pieces: list[Piece]) -> Piece:
    """Pieces as one, their triangles renumbered."""
    pieces = [p for p in pieces if len(p.tri)]
    starts = np.cumsum([0] + [len(p.pos) for p in pieces[:-1]])
    return Piece(np.concatenate([p.pos for p in pieces]), np.concatenate([p.nrm for p in pieces]),
                 np.concatenate([p.uv for p in pieces]),
                 np.concatenate([p.tri + k for p, k in zip(pieces, starts)]).astype(np.int64))


def meshes(name: str, material: str, pieces: list[Piece], frame: LocalFrame, *, renderable: bool = True
           ) -> list[Mesh]:
    """Pieces of one material as meshes under the vertex limit: `name`, then `name_1` and so on. Normals go from
    region axes to AC axes with the positions (`LocalFrame.to_ac` is a rotation, so windings keep their sense)."""
    out, batch, held = [], [], 0

    def flush():
        if not batch:
            return
        p = merge(batch)
        pos = frame.to_ac(p.pos[:, 0], p.pos[:, 1], p.pos[:, 2])
        nrm = np.column_stack([p.nrm[:, 0], p.nrm[:, 2], -p.nrm[:, 1]])
        out.append(Mesh(name + (f"_{len(out)}" if out else ""), material, pos.astype(np.float32),
                        nrm.astype(np.float32), p.uv.astype(np.float32), p.tri, renderable=renderable))

    for piece in pieces:
        if not len(piece.tri):
            continue
        piece = _compact(piece)
        if held + len(piece.pos) > MAX_VERTICES:
            flush()
            batch, held = [], 0
        batch.append(piece)
        held += len(piece.pos)
    flush()
    return out
