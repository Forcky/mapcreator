"""Shallow forks: side roads that leave the route at a slip angle and aren't slip ramps, drawn with a gore stub and
closed.

The maintainer accepted every proposal of the shallow-Y research on 18 Nov 2026:
- **Q1, a gore stub, as a fallback only:** for a side road the junction finder used to drop under 15 degrees mid-route
  (now a junction, `core.junctions`), and for one whose stub's corner flare can't be drawn (`skew` in stubs.json:
  under about 10 degrees where its edges leave the road's, whatever its angle 15 m out). Every stub drawn today stays
  as it is, and so do the long acute flares drawn today (Q7).
- **Q3:** such a side road (under `junctions.SHALLOW_DEG`, 25 degrees) is followed out 150 m (junctions.json's
  `fork`), and its gore stub ends END_D_M in from the road's edge line, not 100 m from its node.
- **Q4, the nose:** a painted gore until the gap between the two roads' edges is 4.5 m, then mown grass joining the two
  verges; no kerb, raised island or crash cushion.
- **Q5, the paint, by South Australia's rules:** white chevrons where traffic beside the gore runs the same way on
  both sides (a diverge or a merge); diagonal stripes where it splits opposing traffic (a divided road's other
  carriageway, a two-way fork off a two-way road); 1 m bars at 45 degrees, a 100 mm outline, the first bar where the
  gore is 1 m wide; spaced 6 m at 70 km/h or less, 12 m faster, 24 m off a motorway or a fast trunk road; none on
  gravel or an unmarked road.
- **Q6:** a divided road's other carriageway where it splits off is drawn the same way, with diagonal stripes, and
  closed as a roundabout closes one: its closure by its class, as any side road's.
- **Q8:** one whose gore still can't be drawn is left undrawn, as before: its mouth kept clear (it's a junction) and
  listed in `skipped` with the reason.

**How:** the gore is the slip ramps' (`core.ramps`): the taper, the painted gore, the nose island and the stub's own
ribbon, triangulated over the road's own edge vertices; the same plan, meshes, checks, closures and settling. A fork
is a ramps.json record with `fork: true` and its junction's id (`as_ramp`), planned after the side roads' stubs, the
roundabouts, the roads crossing the route and the slip ramps, keeping off all of them (`plan`), so none of those
changes. Its stub ends `end_d_m` in from the road's edge, and the way traffic runs beside its gore picks the paint and
the chevrons' direction (`flow`, `opposing`: junctions.json's `fork`). Its meshes are `1ROAD_JF<k>_...` (k its
junction's id, `ramps.mesh_tag`). The road's half widths don't change, so neither does the AI line.
"""

from __future__ import annotations

from mapcreator.core import junctions as jn
from mapcreator.core import ramps

END_D_M = 100.0  # Q3: a fork's gore stub ends this far in from the road's edge line (its d, as a stub's)


def wanted(j: dict, skew: set[int]) -> bool:
    """Whether a junction gets a gore stub: one under 15 degrees, or one whose stub was too skewed to flare (`skew`,
    the ids stubs.json skipped so), as long as the route stage followed it out (`fork`)."""
    return jn.gore_only(j) or (j.get("fork") is not None and j["id"] in skew)


def as_ramp(j: dict) -> dict:
    """A junction (junctions.json's) as a ramp for `ramps.plan`: its 150 m line, how traffic uses it, the road beside
    it, where its stub ends, and `fork`."""
    f = j["fork"]
    return {**{k: v for k, v in j.items() if k != "fork"}, "line": f["line"], "length_m": f["length_m"],
            "ends": f["ends"], "way_ids": f["way_ids"], "kind": f["kind"], "along": f["along"], "main": f["main"],
            "flow": f["flow"], "opposing": f["opposing"], "end_d_m": END_D_M, "fork": True}


def candidates(junctions: list[dict], stubs_skipped: list[dict]) -> list[dict]:
    """The junctions to draw with a gore stub, as ramps (`as_ramp`). `stubs_skipped`: stubs.json's `skipped`."""
    skew = {s["junction"] for s in stubs_skipped if s.get("reason") == "skew"}
    return [as_ramp(j) for j in junctions if wanted(j, skew)]


def plan(junctions: list[dict], stubs_doc: dict | None, c, section, dem_sample, *, width_factor: float = 1.0,
         layouts: tuple[bool, ...] = (False,), rings: list[dict] | None = None, crossing: dict | None = None,
         ramps_planned: list[dict] | None = None) -> dict | None:
    """The forks' gore stubs (`ramps.plan`'s content, each record and skip with `fork: true`), or None with none to
    draw. `stubs_doc` is the stubs' plan (stubs.json's content), `ramps_planned` the slip ramps already planned; the
    rest as `ramps.plan` takes them."""
    stubs_doc = stubs_doc or {"stubs": [], "skipped": []}
    found = candidates(junctions, stubs_doc["skipped"])
    if not found:
        return None
    doc = ramps.plan(found, c, section, dem_sample, width_factor=width_factor, layouts=layouts, junctions=junctions,
                     stubs_plan=stubs_doc["stubs"], rings=rings, crossing=crossing, planned=ramps_planned)
    for sk in doc["skipped"]:
        sk["fork"] = True
    return doc


def merged(ramp_doc: dict | None, fork_doc: dict | None) -> dict | None:
    """ramps.json's content with the slip ramps (`ramp_doc`, `ramps.plan`'s) and the forks' gore stubs (`fork_doc`,
    `plan`'s) together, in order along the road; without forks, the slip ramps' exactly. The rule gains `fork`
    (`end_d_m`, `shallow_deg`, `shallow_leg_m`)."""
    if fork_doc is None:
        return ramp_doc
    base = ramp_doc or {**fork_doc, "ramps": [], "skipped": []}
    every = sorted(base["ramps"] + fork_doc["ramps"], key=lambda r: (r["station_m"], r["junction"]))
    return {**base, "rule": {**base["rule"], "fork": {"end_d_m": END_D_M, "shallow_deg": jn.SHALLOW_DEG,
                                                      "shallow_leg_m": jn.SHALLOW_LEG_M}},
            "ramps": every, "skipped": sorted(base["skipped"] + fork_doc["skipped"], key=lambda s: s["junction"])}


def is_fork(r: dict) -> bool:
    """Whether a ramps.json record is a fork's gore stub."""
    return bool(r.get("fork"))


def summary(doc: dict) -> dict:
    """The section stage's summary of the forks' gore stubs in a merged ramps.json (`merged`)."""
    out = {"gores": sum(is_fork(r) for r in doc["ramps"]), "skipped": {}}
    for sk in doc["skipped"]:
        if sk.get("fork"):
            out["skipped"][sk["reason"]] = out["skipped"].get(sk["reason"], 0) + 1
    return out
