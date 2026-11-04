"""Which kind of tree stands where: a model family for every tree, from a vegetation map (TASVEG in Tasmania).

A tree's family comes from the vegetation community it stands in (`Community.code`, TASVEG's VEGCODE):
- **Woody communities** name their family directly: dry eucalypt (smooth or rough bark), wet eucalypt,
  rainforest, tea-tree and paperbark, wattle, plantations. Mixed wet forest names two layers ("eucalypts
  over rainforest", "over Leptospermum", "with broad-leaf shrubs"). There a crown at least `MIXED_SPLIT` of
  the tallest crowns within `MIXED_REACH_M` is the eucalypt, and a lower one the understorey.
- **Paddocks, towns and easements** (`FAL`, `FUR`, `FUM`, `FAC`...) take their scattered trees' species
  (`NOTABLE_TR`) when the map gives it, else the family of the nearest woody community within
  `PADDOCK_REACH_M`, else a smooth dry eucalypt, the common Tasmanian paddock tree.
- **Treeless communities** (grassland, moorland, highland, saltmarsh and wetland, water and sand) hold trees
  only where the canopy model measured one. Within `EDGE_M` of a woody community it takes that family (the
  map's lines are good to tens of metres); else a low crown is tea-tree (emergent shrubs), and a tall one the
  nearest woody community's family.

Real crowns from a canopy model are measured, so the map only retypes them: it never deletes one. Made-up
trees (WorldCover positions) are another matter: `keep_made_up` drops them where the map says there are no
trees, and thins them in paddocks and towns to the map's own scattered-tree cover.

Families are names, not models: `treefx.FAMILY_BANDS` says which models draw each family, and a family
whose own models aren't built yet borrows the nearest ones that are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import maximum_filter

DRY_SMOOTH, DRY_ROUGH, SNOW_GUM = "dry_smooth", "dry_rough", "snow_gum"
WET, RAINFOREST, CONIFER = "wet", "rainforest", "conifer"
TEA_TREE, WATTLE, BLACKWOOD, SHE_OAK, BANKSIA = "tea_tree", "wattle", "blackwood", "she_oak", "banksia"
REGROWTH, PINE, PLANTATION, WILLOW = "regrowth", "pine", "plantation_euc", "willow"
FAMILIES = (DRY_SMOOTH, DRY_ROUGH, SNOW_GUM, WET, RAINFOREST, CONIFER, TEA_TREE, WATTLE, BLACKWOOD, SHE_OAK,
            BANKSIA, REGROWTH, PINE, PLANTATION, WILLOW)

MIXED_SPLIT = 0.6  # share of the local tallest crown above which a crown in mixed wet forest is the eucalypt
MIXED_REACH_M = 50.0
PADDOCK_REACH_M = 50.0
EDGE_M = 20.0
EMERGENT_SHRUB_M = 8.0  # a crown in treeless country lower than this is tea-tree
# TASVEG's scattered trees are "< 5% crown cover": 0.05 x 10,000 m2 / (pi x 5 m x 5 m) is 6.4 trees a
# hectare with 10 m crowns.
PADDOCK_TREES_PER_HA = 6.0

# Woody communities, by TASVEG code (research/TREE_SPECIES.md section 6a). Bark follows each code's dominant
# species: smooth gums against stringybarks, peppermints and the rough-butted ashes.
_CODES = {
    DRY_SMOOTH: "DGL DVG DVC DVF DRI DPU DTD DTG DTO DCR DGW DPE DMO DRO DBA DKW DMW DDP",
    DRY_ROUGH: "DOB DAD DAS DAM DAZ DAC DSC DNI DSO DSG DDE DOV DOW DFP",
    SNOW_GUM: "DPD DPO DCO WSU",
    WET: "WOU WRE WGL WVI WDU WDA WBR WGK WNU",
    RAINFOREST: "RMT RML RMS RMU RCO RSH RFE",
    CONIFER: "RKP RKS RKX RPP RPF RPW RHP",
    TEA_TREE: "NLE NLA NLM NLN NME SLS SLL SLG SMR SMP SRF SWW SWR SSC SCA SED SRE SBR SKA MBS",
    WATTLE: "NAD NBA SAL",
    BLACKWOOD: "NAR NAF",
    SHE_OAK: "NAV NAL",
    BANKSIA: "NBS SBM",
    PINE: "FPS NCR",
    PLANTATION: "FPH FPU",
    REGROWTH: "FRG",
}
WOODY = {code: family for family, codes in _CODES.items() for code in codes.split()}
# Mixed wet forest: the code's middle letter is the eucalypt (O obliqua, N nitida, D delegatensis), the last
# the understorey.
MIXED = {f"W{e}{u}": under for e in "OND" for u, under in (("R", RAINFOREST), ("L", TEA_TREE), ("B", BLACKWOOD))}
PADDOCK = frozenset("FAL FUR FUM FAC FPF FWU".split())
# Modified land with no trees at all: marram grass, spartina marsh and permanent easements.
TREELESS_MODIFIED = frozenset("FMG FSM FPE".split())
TREELESS_GROUPS = frozenset("GMHAO")
# Scattered trees in paddocks, by NOTABLE_TR code; anything else named is taken as a eucalypt.
NOTABLE = {"PR": PINE, "HM": PINE, "AV": SHE_OAK, "MY": RAINFOREST, "AX": BLACKWOOD, "BM": BANKSIA}
ROUGH_EUCALYPTS = ("obliqua", "amygdalina", "ovata", "delegatensis", "sieberi", "nitida", "radiata", "dalrympleana")
WEEDS = {"PI": PINE, "CY": PINE, "WW": WILLOW}


@dataclass(frozen=True)
class Community:
    polygon: shapely.Geometry
    code: str  # TASVEG VEGCODE
    notable: str = ""  # NOTABLE_TR: the species of scattered trees, if mapped
    notable_name: str = ""  # NOTABLE_TD, its description
    weed: str = ""  # WEED_TYPE


def kind(code: str) -> str:
    """'woody', 'paddock', 'treeless', or '' for an unknown or missing code."""
    if code in WOODY or code in MIXED:
        return "woody"
    if code in TREELESS_MODIFIED:
        return "treeless"
    if code in PADDOCK:
        return "paddock"
    if len(code) == 3:
        if code[0] in "DWRNS":  # eucalypt, rainforest, non-eucalypt forest, scrub
            return "woody"
        if code[0] in TREELESS_GROUPS:
            return "treeless"
        if code[0] == "F":
            return "paddock"
    return ""


def _woody_family(code: str) -> str:
    if code in WOODY:
        return WOODY[code]
    if code in MIXED:
        return WET
    return {"D": DRY_SMOOTH, "W": WET, "R": RAINFOREST, "N": WATTLE, "S": TEA_TREE}.get(code[:1], DRY_SMOOTH)


def notable_family(c: Community) -> str | None:
    if c.weed in WEEDS:
        return WEEDS[c.weed]
    if not c.notable:
        return None
    if c.notable in NOTABLE:
        return NOTABLE[c.notable]
    name = c.notable_name.lower()
    return DRY_ROUGH if any(s in name for s in ROUGH_EUCALYPTS) else DRY_SMOOTH


class VegetationMap:
    """Communities with a spatial index: which one each tree stands in, and the nearest woody one."""

    def __init__(self, communities: list[Community]):
        self.communities = list(communities)
        self.kinds = np.array([kind(c.code) for c in self.communities], dtype=object)
        # Per community: its woody family (the eucalypt, in mixed forest), its understorey family in mixed
        # forest, and the family of the scattered trees it names.
        self.family = np.array([_woody_family(c.code) for c in self.communities], dtype=object)
        self.under = np.array([MIXED.get(c.code) for c in self.communities], dtype=object)
        self.named = np.array([notable_family(c) for c in self.communities], dtype=object)
        self._woody = np.flatnonzero(self.kinds == "woody")
        self._woody_tree = shapely.STRtree([self.communities[i].polygon for i in self._woody])

    def __len__(self) -> int:
        return len(self.communities)

    def at(self, xy: np.ndarray) -> np.ndarray:
        """The index of the community each point stands in, or -1 (the first where polygons overlap)."""
        out = np.full(len(xy), -1, dtype=np.int64)
        if not len(xy) or not len(self.communities):
            return out
        # Each polygon against an index of the points, not each point against the polygons: shapely prepares a
        # query's own geometry, and preparing a few hundred detailed polygons is what makes this fast.
        polys, pts = shapely.STRtree(shapely.points(xy)).query([c.polygon for c in self.communities],
                                                               predicate="intersects")
        order = np.lexsort((polys, pts))
        pts, polys = pts[order], polys[order]
        first = np.unique(pts, return_index=True)[1]
        out[pts[first]] = polys[first]
        return out

    def nearest_woody(self, xy: np.ndarray, max_m: float | None = None) -> np.ndarray:
        """The index of the nearest woody community to each point (within `max_m`, if given), or -1."""
        out = np.full(len(xy), -1, dtype=np.int64)
        if not len(xy) or not len(self._woody):
            return out
        pts, polys = self._woody_tree.query_nearest(shapely.points(xy), max_distance=max_m, all_matches=False)
        out[pts] = self._woody[polys]
        return out

    def codes(self, idx: np.ndarray) -> np.ndarray:
        return np.array([self.communities[i].code if i >= 0 else "" for i in idx], dtype=object)


def _local_tallest(xy: np.ndarray, height: np.ndarray, reach_m: float, cell_m: float = 10.0) -> np.ndarray:
    """The tallest crown within about `reach_m` of each tree, on a `cell_m` grid."""
    if not len(xy):
        return np.zeros(0)
    ij = np.floor((xy - xy.min(axis=0)) / cell_m).astype(np.int64)
    grid = np.zeros(tuple(ij.max(axis=0) + 1))
    np.maximum.at(grid, (ij[:, 0], ij[:, 1]), height)
    r = int(np.ceil(reach_m / cell_m))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    grid = maximum_filter(grid, footprint=(xx ** 2 + yy ** 2) <= r * r, mode="constant")
    return grid[ij[:, 0], ij[:, 1]]


def families(xy: np.ndarray, height: np.ndarray, vmap: VegetationMap, where: np.ndarray | None = None) -> np.ndarray:
    """A family (see module docstring) for every tree. `where` is `vmap.at(xy)`, if the caller has it already."""
    xy, height = np.asarray(xy, dtype=float).reshape(-1, 2), np.asarray(height, dtype=float)
    out = np.full(len(xy), DRY_SMOOTH, dtype=object)
    if not len(xy) or not len(vmap):
        return out
    where = vmap.at(xy) if where is None else where
    k = np.where(where >= 0, vmap.kinds[np.maximum(where, 0)], "")
    over = height >= MIXED_SPLIT * _local_tallest(xy, height, MIXED_REACH_M)

    def take(trees: np.ndarray, comm: np.ndarray) -> None:
        """Trees `trees` take the woody family of communities `comm`, the understorey's if they're low."""
        fam = vmap.family[comm].copy()
        low = (vmap.under[comm] != None) & ~over[trees]  # noqa: E711 - elementwise on an object array
        fam[low] = vmap.under[comm][low]
        out[trees] = fam

    woody = np.flatnonzero(k == "woody")
    take(woody, where[woody])
    paddock = np.flatnonzero(k == "paddock")
    named = vmap.named[where[paddock]]
    has = named != None  # noqa: E711
    out[paddock[has]] = named[has]
    rest = paddock[~has]
    near = vmap.nearest_woody(xy[rest], PADDOCK_REACH_M)
    take(rest[near >= 0], near[near >= 0])
    # Treeless, or not mapped at all (the sea's edge, a gap between polygons): the woody side of a nearby edge,
    # else a low crown is an emergent shrub, else the nearest woody community's family.
    other = np.flatnonzero((k != "woody") & (k != "paddock"))
    near = vmap.nearest_woody(xy[other], EDGE_M)
    take(other[near >= 0], near[near >= 0])
    other = other[near < 0]
    shrub = (k[other] == "treeless") & (height[other] < EMERGENT_SHRUB_M)
    out[other[shrub]] = TEA_TREE
    other = other[~shrub]
    near = vmap.nearest_woody(xy[other])
    take(other[near >= 0], near[near >= 0])
    return out


def keep_made_up(xy: np.ndarray, vmap: VegetationMap, corridor: shapely.Geometry, rng: np.random.Generator, *,
                 mask_agreed: bool) -> np.ndarray:
    """Which made-up trees to keep: none in treeless communities, and in paddocks and towns no more than the
    map's own scattered-tree cover. Where the map names scattered trees (`NOTABLE_TR`), up to
    PADDOCK_TREES_PER_HA stay. Elsewhere in those codes a sparse scatter stays only where a second source
    (`mask_agreed`: the Meta/WRI canopy mask) agreed with WorldCover, else none."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    keep = np.ones(len(xy), dtype=bool)
    if not len(xy) or not len(vmap):
        return keep
    where = vmap.at(xy)
    for i in np.unique(where[where >= 0]):
        k = vmap.kinds[i]
        inside = np.flatnonzero(where == i)
        if k == "treeless":
            keep[inside] = False
        elif k == "paddock" and vmap.communities[i].code in ("FAL", "FUR", "FUM"):
            c = vmap.communities[i]
            if not (c.notable or mask_agreed):
                keep[inside] = False
                continue
            cap = int(round(PADDOCK_TREES_PER_HA * c.polygon.intersection(corridor).area / 10_000.0))
            if len(inside) > cap:
                keep[rng.choice(inside, len(inside) - cap, replace=False)] = False
    return keep


def counts(values: np.ndarray) -> dict[str, int]:
    u, n = np.unique(np.asarray(values, dtype=str), return_counts=True)
    return {str(k) or "(none)": int(v) for k, v in sorted(zip(u, n), key=lambda kv: -kv[1])}
