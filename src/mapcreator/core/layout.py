"""Layouts (driving directions) and the AC_* logic dummies for each.

A point-to-point track needs:
- AC_AB_START_L/R and AC_AB_FINISH_L/R: timing gates just outside each road edge
  (L = left of the driving direction).
- AC_PIT_0 (the track won't load without one) and AC_START_0: where the car spawns.
- AC_HOTLAP_START_0: the hot-lap spawn, further back so the car is moving at the gate.

Spawn dummies point +Z along the driving direction with +Y up, and sit 0.3 m
above the road. That's inside the 0.15-0.35 m window one open-source generator
validated in game; others use 0.5 m. Confirmed at M3: the car lands on all four
wheels, facing along the road.

The finish gate leaves a run-off of real road after it (`finish_distance`).
With 50 m, AC's AI crossed the finish, reached the end of its line, turned
round and left the road (M3). The track stage also slows the AI over the
run-off (`package.ai_hints_ini`), and the mesh stage closes both ends of the
road with invisible walls (`roadside.end_walls`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mapcreator.core.scene import Dummy, LocalFrame

GATE_FROM_END_M = 50.0  # the start gate, from the layout's start; also the shortest run-off
RUNOFF_M = 250.0  # road left after the finish gate, to brake and stop on
RUNOFF_SHARE = 0.15  # at most this share of a short route
SPAWN_BEFORE_GATE_M = 8.0
HOTLAP_BEFORE_GATE_M = 30.0
SPAWN_HEIGHT_M = 0.3
GATE_OUTSIDE_EDGE_M = 0.5


@dataclass(frozen=True)
class Layout:
    key: str  # folder id; "" for a single-layout track
    reverse: bool
    label: str


def layouts_for(choice: str) -> list[Layout]:
    if choice == "forward":
        return [Layout("", False, "")]
    if choice == "reverse":
        return [Layout("", True, "")]
    return [Layout("forward", False, "forward"), Layout("reverse", True, "reverse")]


@dataclass
class RoadModel:
    """What the logic needs about the road, in driving order for one layout."""

    d: np.ndarray  # distance from this layout's start
    e: np.ndarray
    n: np.ndarray
    z: np.ndarray  # centreline surface height
    heading: np.ndarray  # direction of travel, radians (east = 0, CCW)
    half_left: np.ndarray  # relative to the direction of travel
    half_right: np.ndarray
    slope_left: np.ndarray
    slope_right: np.ndarray
    curvature: np.ndarray  # + = turning left in the direction of travel
    grade: np.ndarray  # dz/dd in the direction of travel

    @classmethod
    def from_arrays(cls, s, xy, z, heading, curvature, grade, half_l, half_r, slope_l, slope_r,
                    reverse: bool) -> "RoadModel":
        if not reverse:
            return cls(s - s[0], xy[:, 0], xy[:, 1], z, heading, half_l, half_r, slope_l, slope_r, curvature, grade)
        r = slice(None, None, -1)
        return cls(s[-1] - s[r], xy[r, 0], xy[r, 1], z[r], heading[r] + np.pi, half_r[r], half_l[r],
                   slope_r[r], slope_l[r], -curvature[r], -grade[r])

    @property
    def length(self) -> float:
        return float(self.d[-1])

    def at(self, d: float) -> dict:
        def f(a):
            return float(np.interp(d, self.d, a))

        h = float(np.interp(d, self.d, np.unwrap(self.heading)))
        return {"e": f(self.e), "n": f(self.n), "z": f(self.z), "heading": h, "half_left": f(self.half_left),
                "half_right": f(self.half_right), "slope_left": f(self.slope_left),
                "slope_right": f(self.slope_right)}


def finish_distance(length: float) -> float:
    """Where the finish gate goes on a layout of `length` metres: before a run-off of real road."""
    return length - max(GATE_FROM_END_M, min(RUNOFF_M, RUNOFF_SHARE * length))


def _forward_ac(heading: float) -> list[float]:
    # Direction of travel (cos h, sin h) in E/N -> AC (x = east, z = -north).
    return [float(np.cos(heading)), 0.0, float(-np.sin(heading))]


def logic_dummies(road: RoadModel, frame: LocalFrame) -> list[Dummy]:
    if road.length < 2 * GATE_FROM_END_M + 20:
        raise ValueError(f"route of {road.length:.0f} m is too short for start and finish gates")
    out: list[Dummy] = []

    def gate(prefix: str, d: float) -> None:
        p = road.at(d)
        left = np.array([-np.sin(p["heading"]), np.cos(p["heading"])])
        for side, half, slope, sign in (("L", p["half_left"], p["slope_left"], 1.0),
                                        ("R", p["half_right"], p["slope_right"], -1.0)):
            off = half + GATE_OUTSIDE_EDGE_M
            e, n = np.array([p["e"], p["n"]]) + sign * left * off
            h = p["z"] - slope * half
            out.append(Dummy(f"{prefix}_{side}", frame.to_ac(e, n, h).tolist(), _forward_ac(p["heading"])))

    def spawn(name: str, d: float) -> None:
        p = road.at(d)
        pos = frame.to_ac(p["e"], p["n"], p["z"] + SPAWN_HEIGHT_M).tolist()
        out.append(Dummy(name, pos, _forward_ac(p["heading"])))

    start = GATE_FROM_END_M
    finish = finish_distance(road.length)
    gate("AC_AB_START", start)
    gate("AC_AB_FINISH", finish)
    spawn("AC_PIT_0", start - SPAWN_BEFORE_GATE_M)
    spawn("AC_START_0", start - SPAWN_BEFORE_GATE_M)
    spawn("AC_HOTLAP_START_0", start - HOTLAP_BEFORE_GATE_M)
    return out
