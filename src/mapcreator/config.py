"""Local settings: environment variables first, then an optional `config.local.toml`.

Nothing machine-specific is committed. Copy `config.local.toml.example` to
`config.local.toml` (gitignored) and fill in what applies to your machine.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "config.local.toml"


@dataclass(frozen=True)
class Settings:
    cache_dir: Path
    work_dir: Path
    # Where the Tasmania 2 m DEM is read from: a local mirror path or a /vsicurl/ URL.
    tas_dem_source: str
    # Local mirror of theLIST 2 m canopy height model; empty means fall back to ESA WorldCover.
    tas_chm_source: str
    overpass_url: str
    user_agent: str
    # The local OSM index (providers/osm/index.py, built from a Geofabrik extract) that roads, and
    # buildings outside Tasmania, are read from. Empty means roads come from `overpass_url`, which
    # is for light development use only, and the national fallback has no buildings.
    osm_index: str
    # Write 04_mesh/preview.obj, a debug view of the scene. Off by default: on a 63 km road it took about half
    # the mesh stage, and nothing ships it.
    preview_obj: bool = False
    # Outside a canopy model, keep WorldCover's tree cells only where Meta/WRI's canopy map also sees canopy
    # (providers/meta/canopy.py). On by default; MAPCREATOR_META_CANOPY=0 turns it off.
    meta_canopy: bool = True
    # Local COG mirror of NVIS 7.0's Major Vegetation Subgroups (providers/dcceew/nvis.py); outside Tasmania it
    # types the made-up trees. Empty means they're typed by height alone.
    nvis_source: str = ""


DEFAULT_TAS_DEM = (
    "/vsicurl/https://data.source.coop/alexgleith/tasmania-dem-2m/"
    "Tasmania_Statewide_2m_DEM_14-08-2021.tif"
)


def _file_settings() -> dict:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open("rb") as f:
            return tomllib.load(f)
    return {}


def load() -> Settings:
    f = _file_settings()

    def get(env: str, key: str, default: str) -> str:
        return os.environ.get(env) or f.get(key) or default

    return Settings(
        cache_dir=Path(get("MAPCREATOR_CACHE_DIR", "cache_dir", str(REPO_ROOT / "cache"))),
        work_dir=Path(get("MAPCREATOR_WORK_DIR", "work_dir", str(REPO_ROOT / "out"))),
        tas_dem_source=get("MAPCREATOR_TAS_DEM", "tas_dem_source", DEFAULT_TAS_DEM),
        tas_chm_source=get("MAPCREATOR_TAS_CHM", "tas_chm_source", ""),
        overpass_url=get("MAPCREATOR_OVERPASS_URL", "overpass_url", "https://overpass-api.de/api/interpreter"),
        user_agent=get("MAPCREATOR_USER_AGENT", "user_agent", "mapcreator (development)"),
        osm_index=get("MAPCREATOR_OSM_INDEX", "osm_index", ""),
        preview_obj=str(os.environ.get("MAPCREATOR_PREVIEW_OBJ") or f.get("preview_obj") or "").lower()
        in ("1", "true", "yes"),
        meta_canopy=str(os.environ.get("MAPCREATOR_META_CANOPY") or f.get("meta_canopy", "1")).lower()
        not in ("0", "false", "no", "off"),
        nvis_source=get("MAPCREATOR_NVIS", "nvis_source", ""),
    )
