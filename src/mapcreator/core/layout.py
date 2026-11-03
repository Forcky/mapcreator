"""Layouts (driving directions) and the AC_* logic dummies for each.

A point-to-point track needs:
- AC_AB_START_L/R and AC_AB_FINISH_L/R: timing gates just outside each road edge
  (L = left of the driving direction).
- AC_PIT_k and AC_START_k: where cars spawn, one pair of dummies per grid spot
  (`grid`). The track won't load without AC_PIT_0. Content Manager offers as many
  AI opponents as there are pit boxes less one, so a single spot meant none.
- AC_HOTLAP_START_0: the hot-lap spawn, further back so the car is moving at the gate.

Spawn dummies point +Z along the driving direction with +Y up, and sit 0.3 m
above the road. That's inside the 0.15-0.35 m window one open-source generator
validated in game; others use 0.5 m. Confirmed at M3: the car lands on all four
wheels, facing along the road.

The finish gate leaves a run-off of real road after it (`finish_distance`).
With 50 m, AC's AI crossed the finish, reached the end of its line, turned
round and left the road (M3). The AI line stops `AI_END_SHORT_M` before the
road's end, the track stage ramps the AI down over the run-off
(`ai_line.runoff_hints`), and the mesh stage closes both ends of the road with
invisible walls (`roadside.end_walls`). With one flat hint of half pace and the
line running to the end, the AI still reached the end wall at about 45 km/h,
and AC put it back at the start (in game, 3 Nov).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mapcreator.core.scene import Dummy, LocalFrame

GATE_FROM_END_M = 50.0  # the start gate, from the layout's start; also the shortest run-off
RUNOFF_M = 250.0  # road left after the finish gate, to brake and stop on
RUNOFF_SHARE = 0.15  # at most this share of a short route
AI_END_SHORT_M = 40.0  # the AI line stops this far before the road's end, short of the end wall
# The AI's pace from the finish on, as (metres before the end of its line, pace): it eases off after the finish
# and is all but stopped by the end of its line.
AI_RUNOFF_RAMP = ((150.0, 0.5), (60.0, 0.2), (0.0, 0.05))
SPAWN_BEFORE_GATE_M = 8.0
HOTLAP_BEFORE_GATE_M = 30.0
SPAWN_HEIGHT_M = 0.3
GATE_OUTSIDE_EDGE_M = 0.5
GRID_MAX = 8  # grid spots (pit boxes) per layout: the player and up to seven AI opponents
GRID_ROW_M = 8.0  # between rows, along the road
GRID_SIDE_M = 2.0  # a pair stands this far either side of the centreline
GRID_CAR_HALF_M = 1.0  # half a car's width, with a margin: a pair needs GRID_SIDE_M + this on both sides
GRID_CLEAR_OF_START_M = 8.0  # the last row stays this far from the road's start (its end wall is 1 m in)


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


def grid(road: RoadModel) -> list[tuple[float, float]]:
    """Grid spots as (distance along the layout, offset from the centreline, + = left), front first.

    Spot 0 is on the centreline, SPAWN_BEFORE_GATE_M before the start gate, where M3 checked the spawn. Behind
    it come pairs, GRID_ROW_M apart, or single file where the road is too narrow for two cars side by side, as
    many as fit before the road's start, up to GRID_MAX.
    """
    front = GATE_FROM_END_M - SPAWN_BEFORE_GATE_M
    spots = [(front, 0.0)]
    d = front
    while len(spots) < GRID_MAX:
        d -= GRID_ROW_M
        if d < GRID_CLEAR_OF_START_M:
            break
        p = road.at(d)
        if min(p["half_left"], p["half_right"]) >= GRID_SIDE_M + GRID_CAR_HALF_M:
            spots += [(d, off) for off in (GRID_SIDE_M, -GRID_SIDE_M)][:GRID_MAX - len(spots)]
        else:
            spots.append((d, 0.0))
    return spots


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

    def spawn(name: str, d: float, offset: float = 0.0) -> None:
        p = road.at(d)
        left = np.array([-np.sin(p["heading"]), np.cos(p["heading"])])
        e, n = np.array([p["e"], p["n"]]) + left * offset
        # The road falls from its crown to each edge: follow it, so an offset spot sits as high above the road.
        fall = p["slope_left"] if offset > 0 else p["slope_right"]
        h = p["z"] - fall * abs(offset) + SPAWN_HEIGHT_M
        out.append(Dummy(name, frame.to_ac(e, n, h).tolist(), _forward_ac(p["heading"])))

    start = GATE_FROM_END_M
    finish = finish_distance(road.length)
    gate("AC_AB_START", start)
    gate("AC_AB_FINISH", finish)
    for k, (d, offset) in enumerate(grid(road)):
        spawn(f"AC_PIT_{k}", d, offset)
        spawn(f"AC_START_{k}", d, offset)
    spawn("AC_HOTLAP_START_0", start - HOTLAP_BEFORE_GATE_M)
    return out
