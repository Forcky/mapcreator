"""Water beside the road (lakes, rivers, the sea) as flat surfaces the game draws as water.

Without this, a river beside the road is just terrain at whatever height the elevation model gives the water,
textured like the land. Here:

- `bodies` turns a land-cover classification (ESA WorldCover's permanent water, sampled on its own 10 m grid)
  into smoothed polygons inside the terrain corridor, clear of the road, each with a water level: the median
  elevation just inside its natural shoreline. Not the elevation under the middle, because some DEMs carry
  bathymetry (Tasmania's goes to -10 m in Port Esperance), and not where the corridor's edge cuts across open
  water, for the same reason.
- `lowered` wraps an elevation sampler so the terrain inside a body sits a little below its level, so the
  land never pokes through the surface. Near the shore the terrain keeps its own heights and meets the water.
- `meshes` makes the flat surfaces, named WATER_* with the material "water". They have no collision: a car
  that drives in ends up on the lowered terrain underneath.

Custom Shaders Patch draws the "water" material with its water shader (see `package.ext_config_ini`); vanilla
AC shows it as a glossy blue-green surface.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import shapely

from mapcreator.core.scene import MAX_VERTICES, LocalFrame, Mesh

GRID_M = 10.0  # WorldCover's own cell size
SMOOTH_M = 6.0  # opening and closing radius: drops single-cell specks and rounds the 10 m staircase
MIN_AREA_M2 = 600.0
SHORE_M = 3.0  # the terrain is lowered only this far inside the shoreline
DEPTH_M = 0.6  # how far below the water level the lowered terrain sits
SHORE_SAMPLE_M = 5.0  # spacing of the shoreline samples
UV_M = 16.0
BLOCK_CELLS = 256
SEA_LEVEL_M = 1.5  # a large body at or below this is the sea (or an estuary) rather than a lake
SEA_AREA_M2 = 20_000.0

Sampler = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class WaterBody:
    polygon: shapely.Polygon
    level: float

    @property
    def kind(self) -> str:
        return "SEA" if self.level <= SEA_LEVEL_M and self.polygon.area >= SEA_AREA_M2 else "LAKE"


def mask_polygons(is_water: Callable[[np.ndarray, np.ndarray], np.ndarray], region) -> list[shapely.Polygon]:
    """Smoothed water polygons inside `region`, from a classifier sampled at GRID_M cell centres.

    Only cells inside the region are asked about, block by block, so a long diagonal road doesn't sample
    its whole bounding box.
    """
    from rasterio.features import shapes
    from rasterio.transform import from_origin

    minx, miny, maxx, maxy = region.bounds
    cols, rows = max(1, math.ceil((maxx - minx) / GRID_M)), max(1, math.ceil((maxy - miny) / GRID_M))
    mask = np.zeros((rows, cols), dtype=np.uint8)
    shapely.prepare(region)
    starts = [(r0, c0) for r0 in range(0, rows, BLOCK_CELLS) for c0 in range(0, cols, BLOCK_CELLS)]
    boxes = shapely.box(*np.array([(minx + c0 * GRID_M, maxy - min(rows, r0 + BLOCK_CELLS) * GRID_M,
                                    minx + min(cols, c0 + BLOCK_CELLS) * GRID_M, maxy - r0 * GRID_M)
                                   for r0, c0 in starts]).T)
    touched = shapely.intersects(region, boxes)  # one vectorised test, not one call per block
    for (r0, c0), hit in zip(starts, touched):
        if not hit:
            continue
        r1, c1 = min(rows, r0 + BLOCK_CELLS), min(cols, c0 + BLOCK_CELLS)
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        x = minx + (cc + 0.5) * GRID_M
        y = maxy - (rr + 0.5) * GRID_M
        inside = shapely.contains_xy(region, x, y)
        if not inside.any():
            continue
        wet = np.zeros(x.shape, dtype=bool)
        wet[inside] = is_water(x[inside], y[inside])
        mask[rr[wet], cc[wet]] = 1
    if not mask.any():
        return []
    transform = from_origin(minx, maxy, GRID_M, GRID_M)
    raw = shapely.union_all([shapely.geometry.shape(g) for g, v in shapes(mask, mask=mask.astype(bool),
                                                                          transform=transform) if v == 1])
    smooth = raw.buffer(SMOOTH_M).buffer(-2 * SMOOTH_M).buffer(SMOOTH_M).simplify(1.0)
    smooth = smooth.intersection(region)
    parts = [p for p in getattr(smooth, "geoms", [smooth]) if isinstance(p, shapely.Polygon)]
    return [p for p in parts if p.area >= MIN_AREA_M2]


def _shore_points(poly, region) -> np.ndarray:
    """Points just inside the natural shoreline: the outline moved SHORE_M inwards, minus any stretch along the
    corridor's own edge (a cut across open water, not a shore)."""
    inner = poly.buffer(-SHORE_M)
    line = (inner if not inner.is_empty else poly).boundary
    natural = line.difference(region.boundary.buffer(SHORE_M + GRID_M))
    use = natural if natural.length >= 4 * SHORE_SAMPLE_M else line
    parts = [g for g in getattr(use, "geoms", [use]) if g.length > 0]
    pts = [shapely.line_interpolate_point(g, np.arange(0.0, g.length, SHORE_SAMPLE_M)) for g in parts]
    return shapely.get_coordinates(np.concatenate(pts)) if pts else shapely.get_coordinates(poly.representative_point())


def bodies(is_water, region, dem_sample: Sampler) -> list[WaterBody]:
    out = []
    for poly in mask_polygons(is_water, region):
        pts = _shore_points(poly, region)
        h = np.asarray(dem_sample(pts[:, 0], pts[:, 1]), dtype=float)
        h = h[np.isfinite(h)]
        if h.size:
            out.append(WaterBody(poly, float(np.median(h))))
    return out


def lowered(dem_sample: Sampler, water: list[WaterBody]) -> Sampler:
    """`dem_sample` with the terrain inside each body pushed DEPTH_M below its water level."""
    inners = [(b.polygon.buffer(-SHORE_M), b.level) for b in water]
    inners = [(p, lvl) for p, lvl in inners if not p.is_empty]
    for p, _ in inners:
        shapely.prepare(p)

    def sample(x, y):
        h = np.array(dem_sample(x, y), dtype=float, copy=True)
        for poly, level in inners:
            inside = shapely.contains_xy(poly, x, y)
            if np.any(inside):
                h[inside] = np.fmin(h[inside], level - DEPTH_M)
        return h

    return sample


def meshes(water: list[WaterBody], frame: LocalFrame, material: str = "water") -> list[Mesh]:
    """Flat surfaces, one or more meshes per body (split under the vertex limit), facing up."""
    out: list[Mesh] = []
    for b in water:
        tris = [np.asarray(t.exterior.coords)[:3] for t in shapely.constrained_delaunay_triangles(b.polygon).geoms]
        if not tris:
            continue
        per = MAX_VERTICES // 3
        for start in range(0, len(tris), per):
            chunk = np.asarray(tris[start:start + per])  # (t, 3, 2) easting, northing
            e, n = chunk[..., 0].ravel(), chunk[..., 1].ravel()
            pos = frame.to_ac(e, n, np.full(e.shape, b.level)).astype(np.float32)
            tri = np.arange(len(pos)).reshape(-1, 3)
            p = pos.astype(float)
            up = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])[:, 1]
            tri[up < 0] = tri[up < 0][:, ::-1]
            uv = np.column_stack([(e - frame.e0) / UV_M, (n - frame.n0) / UV_M]).astype(np.float32)
            nrm = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (len(pos), 1))
            out.append(Mesh(f"WATER_{len(out)}", material, pos, nrm, uv, tri))
    return out


def summary(water: list[WaterBody]) -> dict:
    biggest = max(water, key=lambda b: b.polygon.area) if water else None
    return {
        "bodies": len(water),
        "area_m2": round(sum(b.polygon.area for b in water)),
        "levels_m": [round(b.level, 2) for b in water],
        "type": biggest.kind if biggest else None,
    }
