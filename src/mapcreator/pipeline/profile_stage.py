"""Stage 2: smooth the centreline, drape it on the DEM, and smooth the elevation profile."""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np

from mapcreator.config import Settings
from mapcreator.core import centreline as cl
from mapcreator.core import profile as pf
from mapcreator.job import Job
from mapcreator.pipeline import route_stage
from mapcreator.pipeline.workdir import WorkDir
from mapcreator.regions import FINE_DEM_M, job_region

log = logging.getLogger(__name__)

STAGE = "02_profile"
# DEM tiles are fetched this far either side of the road: the terrain corridor (mesh_stage, 700 m) and a margin.
CORRIDOR_M = 800.0
FILL_TAGS = {"bridge", "tunnel"}


def _fill_mask(run_idx: np.ndarray, runs: list[dict]) -> np.ndarray:
    flagged = np.array(
        [any(r["tags"].get(t, "no") not in ("no", "") for t in FILL_TAGS) for r in runs], dtype=bool
    )
    return flagged[run_idx]


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
    if coarse:
        prof = pf.smooth_profile(line.s, z_raw, _fill_mask(run_idx, runs), cutoff_m=cutoff)
    else:
        prof = pf.mixed_profile(line.s, z_raw, _fill_mask(run_idx, runs), fine)
    if coarse:
        first = prof
        capped = pf.limit_grade(line.s, first.z)
        prof = pf.smooth_profile(line.s, capped, np.zeros(len(line.s), bool), cutoff_m=pf.DEFAULT_CUTOFF_M)
        prof.z_raw, prof.filled = z_raw, first.filled
    st = pf.stats(line.s, prof)

    np.savez_compressed(
        work.stage(STAGE) / "centreline.npz",
        s=line.s, xy=line.xy, heading=line.heading, curvature=line.curvature, run=run_idx,
        z_raw=z_raw, z=prof.z, grade=prof.grade, vc=prof.vertical_curvature, filled=prof.filled, fine=fine,
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
