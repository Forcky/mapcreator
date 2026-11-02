"""The build job: the only input the pipeline accepts, from the CLI or the hosted queue.

Deliberately numbers, enums and booleans only. Nothing here may become a URL,
a path or a shell argument, so a job from the public can't steer the worker
anywhere except to a stretch of road.
"""

from __future__ import annotations

import json
import math
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_WAYPOINTS = 25
# A waypoint further than this from any drivable road is refused when the route is built.
MAX_SNAP_M = 150.0
EARTH_RADIUS_M = 6_371_008.8


class WidthOption(StrEnum):
    AS_MAPPED = "as_mapped"
    PLUS_15 = "plus_15"
    PLUS_30 = "plus_30"

    @property
    def factor(self) -> float:
        return {"as_mapped": 1.0, "plus_15": 1.15, "plus_30": 1.30}[self.value]


class Layouts(StrEnum):
    FORWARD = "forward"
    REVERSE = "reverse"
    BOTH = "both"


class Waypoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)


def straight_line_m(waypoints: list[Waypoint]) -> float:
    """Great-circle length of the waypoint sequence: no road through them in order can be shorter,
    give or take the distance each one moves to snap onto a road."""
    total = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        p1, p2 = math.radians(a.lat), math.radians(b.lat)
        dp, dl = p2 - p1, math.radians(b.lon - a.lon)
        h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        total += 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))
    return total


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Folder and KN5 name in AC: lowercase a-z, 0-9 and underscore, at most 32 characters.
    track_id: str = Field(pattern=r"^[a-z0-9_]{1,32}$")
    waypoints: list[Waypoint] = Field(min_length=2, max_length=MAX_WAYPOINTS)
    width: WidthOption = WidthOption.AS_MAPPED
    layouts: Layouts = Layouts.BOTH
    max_length_km: float = Field(default=30.0, gt=0, le=75)

    @field_validator("waypoints")
    @classmethod
    def _distinct(cls, wps: list[Waypoint]) -> list[Waypoint]:
        for a, b in zip(wps, wps[1:]):
            if (a.lon, a.lat) == (b.lon, b.lat):
                raise ValueError("consecutive waypoints must differ")
        return wps

    @model_validator(mode="after")
    def _within_reach(self) -> Job:
        # Refused here, before any build loads the roads in the waypoints' box: points hundreds of km apart
        # would otherwise pull a whole region's roads into memory only to fail the length check after.
        slack_m = 2 * MAX_SNAP_M * (len(self.waypoints) - 1)
        if straight_line_m(self.waypoints) > self.max_length_km * 1000 + slack_m:
            raise ValueError(f"the points are too far apart for a road of at most {self.max_length_km:g} km")
        return self


PUBLIC_PREFIX = "mc_"


def public_track_id(slug: str, route_hash: str) -> str:
    """The id a hosted build ships under: `mc_<slug>_<hash8>`, at most 32 characters.

    The track folder name is global inside a player's AC install, so two people
    naming their roads "grasstree_hill" must not overwrite each other. Rebuilding
    the same road with the same options must keep its id, so Content Manager
    treats the new zip as an update. The route hash gives both.
    """
    base = re.sub(r"[^a-z0-9_]", "", slug.removeprefix(PUBLIC_PREFIX))[:20].strip("_") or "road"
    return f"{PUBLIC_PREFIX}{base}_{route_hash[:8]}"


def load_job(path: Path) -> Job:
    return Job.model_validate(json.loads(path.read_text(encoding="utf-8")))
