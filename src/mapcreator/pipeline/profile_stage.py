"""Stage 2: smooth the centreline, drape it on the DEM, and smooth the elevation profile."""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np

from mapcreator.config import Settings
from mapcreator.core import centreline as cl
from mapcreator.core import crossings as crossings_core
from mapcreator.core import profile as pf
from mapcreator.core import reach
from mapcreator.core import section as sec
from mapcreator.job import Job
from mapcreator.pipeline import route_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.regions import FINE_DEM_M, job_region

log = logging.getLogger(__name__)

STAGE = "02_profile"
# DEM tiles are fetched this far either side of the road: the terrain corridor and a margin (`core.reach`).
CORRIDOR_M = reach.DEM_M
FILL_TAGS = {"bridge", "tunnel"}
# A crossing (route distance) is matched to the nearest smoothed-centreline station within this of where it falls.
CROSSING_MATCH_M = 50.0


def _fill_mask(run_idx: np.ndarray, runs: list[dict]) -> np.ndarray:
    flagged = np.array(
        [any(r["tags"].get(t, "no") not in ("no", "") for t in FILL_TAGS) for r in runs], dtype=bool
    )
    return flagged[run_idx]


def _tagged(run_idx: np.ndarray, runs: list[dict], test) -> np.ndarray:
    """Per station: is its run's way a bridge (`crossings.is_bridge`) or a tunnel (`crossings.is_tunnel`)?"""
    return np.array([test(r["tags"]) for r in runs], dtype=bool)[run_idx]


def crossing_stations(line: cl.Centreline, route_xy: np.ndarray, found: list[dict]) -> np.ndarray:
    """Each crossing's station along the smoothed centreline: the nearest to its point among those within
    CROSSING_MATCH_M of its distance along the route, scaled to the centreline's length."""
    if not found:
        return np.zeros(0)
    scale = float(line.s[-1]) / max(float(cl.cumulative(route_xy)[-1]), 1e-9)
    out = []
    for c in found:
        near = np.flatnonzero(np.abs(line.s - c["at_m"] * scale) <= CROSSING_MATCH_M)
        if not len(near):
            continue
        d = np.hypot(*(line.xy[near] - np.asarray(c["xy"], dtype=float)).T)
        out.append(float(line.s[near[int(np.argmin(d))]]))
    return np.array(out)


def run(job: Job, settings: Settings, work: WorkDir) -> dict:
    region = job_region(job.waypoints)
    route = np.load(work.root / route_stage.STAGE / "route.npz")
    runs = work.read_json(route_stage.STAGE, "runs.json")

    line = cl.smooth(route["xy"])
    run_idx = route["seg_run"][np.clip(line.source_index, 0, len(route["seg_run"]) - 1)]

    dem = region.elevation(settings)
    tiles = dem.load_corridor(line.xy, CORRIDOR_M)
    z_raw = dem.sample(line.xy[:, 0], line.xy[:, 1])
    # Sources that pick their data per road (the national fallback) say which they used, for ATTRIBUTION.txt.
    if hasattr(dem, "attributions"):
        work.write_json(STAGE, "attributions.json", [asdict(a) for a in dem.attributions])

    # A coarse DEM samples the hillside as much as the road: filter far above its cell size
    # and cap the grade at what a sealed road plausibly has.
    resolution = getattr(dem, "resolution_m", 2.0)
    coarse = resolution > FINE_DEM_M
    cutoff = max(pf.DEFAULT_CUTOFF_M, 8.0 * resolution) if coarse else pf.DEFAULT_CUTOFF_M
    # Where the DEM really has the detail its grid suggests. A fine DEM made of mixed sources is checked
    # along the road; the section and mesh stages read this too (camber, guard rails).
    fine_source = "resolution"
    if coarse:
        fine = np.zeros(len(line.s), dtype=bool)
    elif getattr(dem, "mixed_detail", False):
        try:
            fine, fine_source = dem.lidar_mask(line.xy), "LiDAR index"
        except Exception as e:  # noqa: BLE001 - the index only sharpens the profile; fall back to the ground
            log.warning("LiDAR index unavailable (%s: %s); judging the DEM by its texture", type(e).__name__, e)
            fine, fine_source = pf.fine_stations(line.s, line.xy, line.heading, dem.sample), "texture"
    else:
        fine = np.ones(len(line.s), dtype=bool)
    # Where the mapped line runs up a cutting beside the road, profile the road's own strip instead: a strip as
    # wide as the road is mapped (its width tags, or the class default for its surface; core/section.py).
    width = sec.mapped_width([r["tags"] for r in runs], line.s, run_idx)
    strip = pf.road_strip_heights(line.s, line.xy, line.heading, dem.sample, z_raw, fine,
                                  width=width) if not coarse else None
    # Where a road crosses over the route, a hump the DEM shows there is bridged like the route's own bridges.
    over_at = crossing_stations(line, route["xy"], crossings_core.over(route_stage.crossings(work)))
    under = pf.humps_under(line.s, z_raw if coarse else strip.z, over_at)
    fill = _fill_mask(run_idx, runs) | under
    if coarse:
        prof = pf.smooth_profile(line.s, z_raw, fill, cutoff_m=cutoff)
    else:
        prof = pf.mixed_profile(line.s, strip.z, fill, fine & ~strip.coarse)
        prof.z_raw = z_raw
    if coarse:
        first = prof
        capped = pf.limit_grade(line.s, first.z)
        prof = pf.smooth_profile(line.s, capped, np.zeros(len(line.s), bool), cutoff_m=pf.DEFAULT_CUTOFF_M)
        prof.z_raw, prof.filled = z_raw, first.filled
    st = pf.stats(line.s, prof)

    np.savez_compressed(
        work.stage(STAGE) / "centreline.npz",
        s=line.s, xy=line.xy, heading=line.heading, curvature=line.curvature, run=run_idx,
        z_raw=z_raw, z=prof.z, grade=prof.grade, vc=prof.vertical_curvature, filled=prof.filled,
        # Off-road stretches with no road strip are profiled as coarse, so camber and rails treat them so too.
        fine=fine & ~strip.coarse if strip is not None else fine,
        # Where the profile isn't the ground's: the route's own bridges and tunnels (by their tags), and the humps
        # bridged under a road crossing over (`under`). The section stage gives them crown only, and the mesh stage
        # puts rails along the bridges.
        bridge=_tagged(run_idx, runs, crossings_core.is_bridge),
        tunnel=_tagged(run_idx, runs, crossings_core.is_tunnel),
        under=under,
    )
    summary = st.as_dict() | {
        "dem_tiles": tiles,
        "dem_resolution_m": getattr(dem, "resolution_m", None),
        "dem_lidar_coverage": getattr(dem, "coverage", None),
        "profile_cutoff_m": cutoff,
        "grade_capped": coarse or not fine.all(),
        "dem_fine_share": round(float(fine.mean()), 3),
        "dem_fine_source": fine_source,
        "centreline_max_deviation_m": round(line.max_deviation_m, 2),
        "min_radius_m": round(float(1 / max(np.abs(line.curvature).max(), 1e-9)), 1),
        "off_road_m": round(float(strip.off_road.sum() * (line.s[1] - line.s[0])), 1) if strip is not None else 0.0,
        "off_road_strip_m": round(float(np.isfinite(strip.strip_offset).sum() * (line.s[1] - line.s[0])), 1)
        if strip is not None else 0.0,
        "off_road_coarse_m": round(float(strip.coarse.sum() * (line.s[1] - line.s[0])), 1) if strip is not None else 0.0,
        "humps_under_crossings_m": round(float(under.sum() * (line.s[1] - line.s[0])), 1),
    }
    work.write_json(STAGE, "stats.json", summary)
    _plot(work, line, prof, summary)
    return summary


def _plot(work: WorkDir, line: cl.Centreline, prof: pf.Profile, summary: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:  # plots are optional
        return
    km = line.s / 1000
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True, height_ratios=[3, 1.4, 1.4])
    ax1.plot(km, prof.z_raw, lw=0.6, color="0.6", label="DEM (raw)")
    ax1.plot(km, prof.z, lw=1.4, color="C0", label="smoothed")
    if prof.filled.any():
        ax1.fill_between(km, prof.z.min(), prof.z.max(), where=prof.filled, color="C1", alpha=0.15,
                         label="bridged / no data")
    ax1.set_ylabel("m AHD")
    ax1.legend(loc="upper left")
    ax1.set_title(
        f"{work.root.name}: {summary['length_m'] / 1000:.2f} km, +{summary['ascent_m']:.0f} m / "
        f"-{summary['descent_m']:.0f} m, max grade {summary['max_grade_50m'] * 100:.1f}% (50 m)"
    )
    ax2.plot(km, prof.grade * 100, lw=0.9, color="C2")
    ax2.axhline(0, color="0.5", lw=0.5)
    ax2.set_ylabel("grade %")
    ax3.plot(km, np.clip(np.abs(line.curvature), 0, 0.05), lw=0.8, color="C3")
    ax3.set_ylabel("|curvature| 1/m")
    ax3.set_xlabel("km")
    fig.tight_layout()
    fig.savefig(work.stage(STAGE) / "profile.png", dpi=110)
    plt.close(fig)
