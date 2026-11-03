"""Guard rails where the ground falls away steeply beside the road.

Real hill roads have W-beam rails on the outside of drops. Without them a
car that runs wide tumbles down a DEM embankment, so they matter for driving
as well as looks. Placement comes from the DEM:
- Compare the road edge with the ground 7 m beyond it, on each side.
- Where it falls away by more than DROP_M, put up a rail.
- Smooth the readings, close short gaps, drop short runs, and extend each
  run a little, as a road authority would.

Each run gets two meshes:
- `RAIL_*`: the visible beam and posts, visual only.
- `1WALL_RAIL_*`: an invisible, taller collision wall at the same line. Any
  physics mesh named WALL is a solid barrier in AC.

Both ends of the road are closed too (`end_walls`): an invisible wall across
the road and well past its verges, so no car, and no AI that turns round at the
end of its line, drives off the end of the generated world.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_opening, median_filter

from mapcreator.core.scene import LocalFrame, Mesh

PROBE_BEYOND_EDGE_M = 7.0
DROP_M = 2.0
RAIL_OFFSET_M = 1.0  # from the road edge, on the verge
# Sample points per rail mesh: the visual rail has 10 vertices a sample, so this stays well under 65,535.
MAX_RAIL_POINTS = 4000
VERGE_FALL = 0.06
MERGE_GAP_M = 16.0
MIN_RUN_M = 24.0
EXTEND_M = 8.0
STEP_M = 2.0
BEAM = (0.45, 0.75)  # metres above the verge
POST_W = 0.12
COLLIDER = (-0.3, 1.0)
END_WALL_INSET_M = 1.0  # from each end of the road
END_WALL_HALF_M = 30.0  # either side of the centreline: the road, its verges and some terrain
END_WALL = (-1.0, 3.0)  # below the lower and above the higher of road and ground


def rail_mask(s: np.ndarray, drop: np.ndarray) -> np.ndarray:
    step = s[1] - s[0] if len(s) > 1 else 1.0
    d = median_filter(np.nan_to_num(drop, nan=0.0), size=max(int(10 / step) | 1, 3))
    m = d > DROP_M
    m = binary_closing(m, structure=np.ones(max(int(MERGE_GAP_M / step), 1)))
    m = binary_opening(m, structure=np.ones(max(int(MIN_RUN_M / step), 1)))
    return binary_dilation(m, structure=np.ones(2 * max(int(EXTEND_M / step), 1) + 1))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2] - 1))


def guard_rails(s, xy, z, heading, half_l, half_r, slope_l, slope_r, dem_sample, frame: LocalFrame,
                allowed: np.ndarray | None = None) -> tuple[list[Mesh], dict]:
    nrm = np.column_stack([-np.sin(heading), np.cos(heading)])
    meshes: list[Mesh] = []
    stats = {}
    step = s[1] - s[0] if len(s) > 1 else 1.0
    stride = max(int(round(STEP_M / step)), 1)
    for side, sign, half, slope in (("L", 1.0, half_l, slope_l), ("R", -1.0, half_r, slope_r)):
        edge_h = z - slope * half
        probe = xy + nrm * (sign * (half + PROBE_BEYOND_EDGE_M))[:, None]
        drop = edge_h - dem_sample(probe[:, 0], probe[:, 1])
        mask = rail_mask(s, drop)
        if allowed is not None:
            mask &= allowed
        runs = _runs(mask)
        stats[side] = {"runs": len(runs), "metres": round(float(mask.sum() * step), 1)}
        for k, (i0, i1) in enumerate(runs):
            idx = np.arange(i0, i1 + 1, stride)
            if idx[-1] != i1:
                idx = np.append(idx, i1)
            if len(idx) < 2:
                continue
            # A very long run is cut into pieces that share their end points, so each mesh stays under the
            # vertex limit and the rail stays continuous. A run of ordinary length keeps its plain name.
            for p, start in enumerate(range(0, len(idx) - 1, MAX_RAIL_POINTS - 1)):
                part = idx[start:start + MAX_RAIL_POINTS]
                name = f"{side}{k:03d}" + (f"_{p}" if p else "")
                off = (half[part] + RAIL_OFFSET_M) * sign
                line = xy[part] + nrm[part] * off[:, None]
                base = edge_h[part] - RAIL_OFFSET_M * VERGE_FALL
                meshes.append(_rail_visual(f"RAIL_{name}", line, base, frame))
                meshes.append(_collider(f"1WALL_RAIL_{name}", line, base, frame))
    return meshes, stats


def end_walls(s, xy, z, heading, dem_sample, frame: LocalFrame) -> list[Mesh]:
    """Invisible walls across the start and the end of the road (`1WALL_END_START`, `1WALL_END_FINISH`)."""
    out = []
    offs = np.linspace(-END_WALL_HALF_M, END_WALL_HALF_M, 31)
    for name, at in (("1WALL_END_START", s[0] + END_WALL_INSET_M), ("1WALL_END_FINISH", s[-1] - END_WALL_INSET_M)):
        k = int(np.clip(np.searchsorted(s, at), 0, len(s) - 1))
        h = float(heading[k])
        line = xy[k] + offs[:, None] * np.array([-np.sin(h), np.cos(h)])
        ground = np.asarray(dem_sample(line[:, 0], line[:, 1]), dtype=float)
        ground = np.where(np.isfinite(ground), ground, z[k])
        lo = np.minimum(ground, z[k]) + END_WALL[0]
        hi = np.maximum(ground, z[k]) + END_WALL[1]
        pos, uv, tri = _strip(line, lo, hi, frame, 4.0)
        along = np.tile([np.cos(h), 0.0, -np.sin(h)], (len(pos), 1))  # facing along the road, in AC axes
        out.append(Mesh(name, "rail", pos.astype(np.float32), along.astype(np.float32), uv.astype(np.float32), tri,
                        renderable=False))
    return out


def _strip(line: np.ndarray, lo: np.ndarray, hi: np.ndarray, frame: LocalFrame, uv_len: float):
    """A vertical ribbon along `line` between heights lo and hi, both windings, shared vertices."""
    n = len(line)
    bottom = frame.to_ac(line[:, 0], line[:, 1], lo)
    top = frame.to_ac(line[:, 0], line[:, 1], hi)
    pos = np.empty((2 * n, 3))
    pos[0::2], pos[1::2] = bottom, top
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    uv = np.empty((2 * n, 2))
    uv[0::2, 0], uv[1::2, 0] = dist / uv_len, dist / uv_len
    uv[0::2, 1], uv[1::2, 1] = 1.0, 0.0
    i = np.arange(n - 1) * 2
    front = np.concatenate([np.column_stack([i, i + 2, i + 1]), np.column_stack([i + 1, i + 2, i + 3])])
    tri = np.concatenate([front, front[:, [0, 2, 1]]])
    return pos, uv, tri


def _rail_visual(name: str, line: np.ndarray, base: np.ndarray, frame: LocalFrame) -> Mesh:
    pos, uv, tri = _strip(line, base + BEAM[0], base + BEAM[1], frame, 4.0)
    parts_pos, parts_uv, parts_tri = [pos], [uv], [tri]
    count = len(pos)
    # Posts at each sample: a thin crossed pair of quads from just below the verge to the beam top.
    d = np.diff(line, axis=0)
    t = np.vstack([d, d[-1:]])
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    for vec in (t, np.column_stack([-t[:, 1], t[:, 0]])):
        a, b = line - vec * POST_W / 2, line + vec * POST_W / 2
        lo, hi = base - 0.15, base + BEAM[1]
        quad = np.stack([frame.to_ac(a[:, 0], a[:, 1], lo), frame.to_ac(b[:, 0], b[:, 1], lo),
                         frame.to_ac(b[:, 0], b[:, 1], hi), frame.to_ac(a[:, 0], a[:, 1], hi)], axis=1)
        qp = quad.reshape(-1, 3)
        quv = np.tile([[0.0, 1.0], [0.05, 1.0], [0.05, 0.0], [0.0, 0.0]], (len(line), 1))
        v = count + np.arange(len(line))[:, None] * 4 + np.arange(4)[None, :]
        front = np.concatenate([v[:, [0, 1, 2]], v[:, [0, 2, 3]]])
        parts_pos.append(qp)
        parts_uv.append(quv)
        parts_tri.append(np.concatenate([front, front[:, [0, 2, 1]]]))
        count += len(qp)
    positions = np.concatenate(parts_pos)
    normals = np.tile([0.0, 1.0, 0.0], (len(positions), 1))
    return Mesh(name, "rail", positions.astype(np.float32), normals.astype(np.float32),
                np.concatenate(parts_uv).astype(np.float32), np.concatenate(parts_tri).astype(np.int64))


def _collider(name: str, line: np.ndarray, base: np.ndarray, frame: LocalFrame) -> Mesh:
    pos, uv, tri = _strip(line, base + COLLIDER[0], base + COLLIDER[1], frame, 4.0)
    d = np.diff(line, axis=0)
    t = np.vstack([d, d[-1:]])
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    side = np.repeat(np.column_stack([-t[:, 1], np.zeros(len(t)), -t[:, 0]]), 2, axis=0)  # horizontal, in AC axes
    return Mesh(name, "rail", pos.astype(np.float32), side.astype(np.float32), uv.astype(np.float32), tri,
                renderable=False)
