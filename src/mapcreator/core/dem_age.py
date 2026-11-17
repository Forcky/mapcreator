"""Bridges the elevation data doesn't hold: where the DEM is older than a structure, or too coarse to show it.

A bare-earth DEM made before a bridge was built shows the ground from before: the old crossroads where an interchange
now stands (Summerleas Road under the Huon Highway: LiDAR from 2011, the interchange from 2016-2018). The build then
runs both roads at one level. Tasmania's statewide DEM (released August 2021) holds only LiDAR finished before it, and
the newer surveys theLIST's index lists (Kingborough 2022 there) aren't published in a form a build can fetch
(`docs/data-sources.md`, "Newer LiDAR surveys"). So the build says so: this module finds those bridges and words the
warning.

**The shared wording.** `PREDATES` and `COARSE` are the reasons, and `warning()` makes the line. The crossing roads'
stage uses the same helper when it leaves an overpass deck out (`then="its deck is left out"`), so the build gives
one reason in one form wherever it comes from.

**Which bridges.** Every road-over-road crossing in `01_route/crossings.json` where the upper road is a tagged bridge:
another road's bridge over the route (`over`), or the route's own bridge over another road (`under`). Untagged upper
ways, and bridges whose ends aren't in the crossing's 150 m clip (over 300 m long), aren't judged.

**The test.** The upper road's level at the crossing is the bridge's own: for another road's bridge, the DEM at its
two ends (the abutments, where the bare earth keeps the approach fill) interpolated to the crossing; for the route's
own bridge, its profile there, which is drawn as a straight grade between the same ends. The lower road's level is the
lowest DEM along it within 40 m of the crossing (so an embankment the DEM kept over a narrow underpass, or a road
dipping under the deck, doesn't count against it). A real road over a road needs 4.6 m of clearance at the least
(Australian practice's lowest, on local roads; main roads have 5.4 m) plus the deck. Where the DEM puts the two less
than 4.6 m apart, it doesn't hold the bridge: `PREDATES` where the DEM is fine there (LiDAR), else `COARSE`.

At Tasmania's 164 road-over-road sites the research (Nov 2026) found 25 where the two roads' straight lines are under
2 m apart, and 58 under 4.5 m; this test, against the lowest ground rather than a straight line, is the gentler one.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field

import numpy as np

FORMAT_VERSION = 1
FILE = "dem_age.json"  # in the profile stage's folder

PREDATES = "the elevation data predates this bridge"
COARSE = "the elevation data here is too coarse to show this bridge"

MIN_CLEARANCE_M = 4.6  # the lowest clearance Australian practice allows (local roads), with no deck at all
NEAR_M = 40.0  # the lower road is read this far either side of the crossing
STEP_M = 2.0  # ... every this many metres
CLIP_SLACK_M = 0.5  # a crossing's line shorter than its way by more than this was clipped: its ends aren't the bridge's
MAX_LISTED = 5  # warnings listed one by one; the rest are counted in one more line


def warning(label: str, *, coarse: bool = False, then: str = "", detail: str = "") -> str:
    """One warning line for a bridge the elevation data doesn't hold, in the one wording every stage uses:
    "`label`: the elevation data predates this bridge[, so `then`][ (`detail`)]." `coarse` gives the coarse reason
    instead."""
    text = f"{label}: {COARSE if coarse else PREDATES}"
    if then:
        text += f", so {then}"
    if detail:
        text += f" ({detail})"
    return text + "."


def label(crossing: dict, along_m: float | None = None) -> str:
    """How a warning names a crossing: "Huon Highway (trunk, A6) over the route at 0.42 km", or "an unnamed road
    (secondary) under the route's bridge at 1.05 km". `along_m` is where along the route (the crossing's `at_m` if
    not given)."""
    tags = crossing.get("tags") or {}
    name = crossing.get("name") or tags.get("name") or "an unnamed road"
    kind = ", ".join(str(v) for v in (crossing.get("highway") or tags.get("highway"), tags.get("ref")) if v)
    where = "over the route" if crossing.get("relation") == "over" else "under the route's bridge"
    km = (crossing.get("at_m", 0.0) if along_m is None else along_m) / 1000
    return f"{name}{f' ({kind})' if kind else ''} {where} at {km:.2f} km"


@dataclass
class Bridge:
    """One bridge over or under the route, judged (one entry for every crossing point of the same two ways)."""

    crossings: list[int]  # the crossing ids (crossings.json) it covers
    relation: str  # over: another road's bridge over the route; under: the route's own bridge over another road
    way_id: int | None  # the crossing way's
    route_way_id: int | None
    along_m: float  # the worst point's distance along the route (the smoothed centreline)
    label: str
    verdict: str  # holds, predates, coarse, or unjudged
    why: str = ""  # for unjudged
    upper_m: float | None = None  # the bridge's own level at the crossing
    lower_m: float | None = None  # the lowest DEM along the lower road within NEAR_M
    separation_m: float | None = None
    surveys: list[dict] = field(default_factory=list)  # the LiDAR the DEM holds there, oldest first (if known)
    newer_surveys: list[dict] = field(default_factory=list)  # LiDAR over it the DEM doesn't hold
    surveys_known: bool = False  # whether the DEM said which surveys lie over it (theLIST's index, in Tasmania)

    @property
    def flagged(self) -> bool:
        return self.verdict in ("predates", "coarse")

    def warning(self) -> str:
        return warning(self.label, coarse=self.verdict == "coarse", detail=self.detail())

    def detail(self) -> str:
        parts = [f"it puts the two roads {self.separation_m:.1f} m apart, where a bridge needs {MIN_CLEARANCE_M:g} m"]
        if self.surveys:
            last = self.surveys[-1]
            parts.append(f"its newest LiDAR there is {last['name']}, finished {_month(last['completed'])}")
        elif self.verdict == "coarse" and self.surveys_known:
            parts.append("there's no LiDAR in it there")
        if self.newer_surveys:
            names = ", ".join(s["name"] for s in self.newer_surveys)
            parts.append(f"{names} {'is' if len(self.newer_surveys) == 1 else 'are'} newer, but builds can't use "
                         f"{'it' if len(self.newer_surveys) == 1 else 'them'} yet")
        return "; ".join(parts)


@dataclass
class Report:
    bridges: list[Bridge]
    not_bridges: int  # crossings whose upper way isn't a tagged bridge: not judged
    unmatched: int  # crossings not found on the smoothed centreline

    def flagged(self) -> list[Bridge]:
        return [b for b in self.bridges if b.flagged]

    def warnings(self) -> list[str]:
        """One line a flagged bridge, up to MAX_LISTED, then one counting the rest. Bridges that read the same (a
        highway's two decks over the route, two carriageways under the route's bridge) share a line, at the worst."""
        same: dict[str, list[Bridge]] = {}
        for b in self.flagged():
            same.setdefault(b.label, []).append(b)
        lines = []
        for group in same.values():
            worst = min(group, key=lambda b: b.separation_m)
            name = worst.label
            if len(group) > 1:
                name += f" ({len(group)} {'decks' if worst.relation == 'over' else 'carriageways'})"
            lines.append(warning(name, coarse=worst.verdict == "coarse", detail=worst.detail()))
        out = lines[:MAX_LISTED]
        if len(lines) > MAX_LISTED:
            more = len(lines) - MAX_LISTED
            out.append(f"{more} more bridge{'s' if more != 1 else ''} the elevation data doesn't hold "
                       f"(02_profile/{FILE})")
        return out

    def summary(self) -> dict:
        count = {v: sum(b.verdict == v for b in self.bridges) for v in ("holds", "predates", "coarse", "unjudged")}
        return {"bridges_checked": count["holds"] + count["predates"] + count["coarse"],
                "bridges_dem_predates": count["predates"], "bridges_dem_coarse": count["coarse"],
                "dem_warnings": self.warnings()}

    def as_dict(self) -> dict:
        return {"version": FORMAT_VERSION,
                "rule": {"min_clearance_m": MIN_CLEARANCE_M, "near_m": NEAR_M,
                         "upper": "the bridge's own level: the DEM at its ends (the route's: its profile)",
                         "lower": "the lowest DEM along the lower road within near_m of the crossing"},
                "bridges": [asdict(b) for b in self.bridges], "not_bridges": self.not_bridges,
                "unmatched": self.unmatched, "warnings": self.warnings()}


def _month(iso: str) -> str:
    try:
        return dt.date.fromisoformat(iso).strftime("%B %Y")
    except ValueError:
        return iso


def _along(line: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))]


def _points(line: np.ndarray, at: float, reach: float) -> np.ndarray:
    """Points every STEP_M along a polyline within `reach` of `at` metres along it."""
    s = _along(line)
    m = np.arange(max(at - reach, 0.0), min(at + reach, float(s[-1])) + 1e-9, STEP_M)
    return np.column_stack([np.interp(m, s, line[:, 0]), np.interp(m, s, line[:, 1])])


def _bridge_ends(c: dict, sample) -> tuple[float | None, str]:
    """Another road's bridge: the DEM at its two ends interpolated to the crossing, or None and why not."""
    line = np.asarray(c.get("line") or [], dtype=float).reshape(-1, 2)
    if len(line) < 2 or "line_at_m" not in c:
        return None, "no line for the bridge"
    s = _along(line)
    if float(c.get("way_length_m", s[-1])) > s[-1] + CLIP_SLACK_M:
        return None, "the bridge runs past the 150 m kept either side of the crossing"
    z = np.asarray(sample(line[[0, -1], 0], line[[0, -1], 1]), dtype=float)
    if not np.isfinite(z).all():
        return None, "no elevation data at the bridge's ends"
    return float(np.interp(float(c["line_at_m"]), [0.0, s[-1]], z)), ""


def _lowest_along(c: dict, sample) -> float | None:
    """The lowest DEM along the crossing way within NEAR_M of the crossing (the road under the route's bridge)."""
    line = np.asarray(c.get("line") or [], dtype=float).reshape(-1, 2)
    if len(line) < 2 or "line_at_m" not in c:
        return None
    pts = _points(line, float(c["line_at_m"]), NEAR_M)
    z = np.asarray(sample(pts[:, 0], pts[:, 1]), dtype=float)
    return float(np.nanmin(z)) if np.isfinite(z).any() else None


def judge(crossings: list[dict], stations: list[float | None], s: np.ndarray, z_raw: np.ndarray, z: np.ndarray,
          fine: np.ndarray, sample, *, surveys: list[list[dict]] | None = None) -> Report:
    """Judge every bridge over or under the route (the module docstring).

    `crossings` are crossings.json's; `stations` each one's station on the smoothed centreline (None if it isn't on
    it). `s`, `z_raw`, `z` and `fine` are the centreline's stations, its raw DEM heights, its profile and where the
    DEM is fine. `sample(x, y)` reads the DEM. `surveys`, if the DEM knows them, are the LiDAR surveys over each
    crossing (`{"name", "completed", "in_dem"}`, oldest first).
    """
    s, z_raw, z = (np.asarray(a, dtype=float) for a in (s, z_raw, z))
    fine = np.asarray(fine, dtype=bool)
    groups: dict[tuple, list[Bridge]] = {}
    not_bridges = unmatched = 0
    for i, c in enumerate(crossings):
        relation = c.get("relation")
        route = c.get("route") or {}
        upper_is_bridge = bool(c.get("bridge")) if relation == "over" else bool(route.get("bridge"))
        if relation not in ("over", "under") or route.get("tunnel") or not upper_is_bridge:
            not_bridges += 1
            continue
        st = stations[i] if i < len(stations) else None
        if st is None or not len(s):
            unmatched += 1
            continue
        b = Bridge(crossings=[int(c.get("id", i))], relation=relation, way_id=c.get("way_id"),
                   route_way_id=route.get("way_id"), along_m=round(float(st), 1), label=label(c, st),
                   verdict="unjudged")
        if surveys is not None and i < len(surveys):
            b.surveys_known = True
            b.surveys = [v for v in surveys[i] if v.get("in_dem")]
            b.newer_surveys = [v for v in surveys[i] if not v.get("in_dem")]
        near = np.abs(s - st) <= NEAR_M
        if relation == "over":
            upper, b.why = _bridge_ends(c, sample)
            lower = float(np.nanmin(z_raw[near])) if np.isfinite(z_raw[near]).any() else None
        else:
            upper = float(np.interp(st, s, z))
            lower = _lowest_along(c, sample)
        if upper is None or lower is None or not np.isfinite(upper):
            b.why = b.why or "no elevation data under the crossing"
        else:
            b.upper_m, b.lower_m = round(upper, 2), round(lower, 2)
            b.separation_m = round(upper - lower, 2)
            on_fine = bool(fine[int(np.argmin(np.abs(s - st)))]) if len(fine) == len(s) else True
            b.verdict = "holds" if upper - lower >= MIN_CLEARANCE_M else "predates" if on_fine else "coarse"
        key = (relation, c.get("way_id"), route.get("way_id")) if c.get("way_id") is not None else (relation, i)
        groups.setdefault(key, []).append(b)
    bridges = [_worst(g) for g in groups.values()]
    bridges.sort(key=lambda b: b.along_m)
    return Report(bridges, not_bridges, unmatched)


def _worst(group: list[Bridge]) -> Bridge:
    """The same two ways crossing at several points (a dual carriageway, a loop under two decks): one bridge, judged
    at its worst point."""
    judged = [b for b in group if b.separation_m is not None]
    worst = min(judged, key=lambda b: b.separation_m) if judged else group[0]
    worst.crossings = sorted(i for b in group for i in b.crossings)
    return worst
