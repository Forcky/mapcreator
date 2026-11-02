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
    overpass_url: str
    user_agent: str


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
        overpass_url=get("MAPCREATOR_OVERPASS_URL", "overpass_url", "https://overpass-api.de/api/interpreter"),
        user_agent=get("MAPCREATOR_USER_AGENT", "user_agent", "mapcreator (development)"),
    )
