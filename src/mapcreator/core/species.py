"""Which kind of tree stands where: a model family for every tree, from a vegetation map: TASVEG in Tasmania
(`VegetationMap`), NSW's State Vegetation Type Map in NSW (`SvtmMap`), NVIS elsewhere (`NvisMap`).

In Tasmania a tree's family comes from the vegetation community it stands in (`Community.code`, TASVEG's VEGCODE):
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

On the mainland, NVIS 7.0's Major Vegetation Subgroups give each 100 m cell one dominant subgroup, and the
subgroup gives the family (`NVIS_SUBGROUPS`). The cells are too coarse to delete trees by, and the publisher
warns against analysing its non-native classes, so NVIS only types trees (`NvisMap.keep_made_up` keeps all).

In NSW the State Vegetation Type Map gives each 5 m cell a Plant Community Type (PCT), whose name lists its
dominant trees: "Narrow-leaved Ironbark - White Cypress Pine - Buloke tall open forest...", "Woronora Plateau
Scribbly Gum Woodland". A tree there is drawn from the families of the trees its PCT names, in equal shares
(`svtm_families`); a PCT that names none we know takes its vegetation class's family, else its formation's.
Cleared land and treeless PCTs take the nearest typed PCT within `SVTM_NEAREST_M`, as NVIS's cleared cells do.
It only types trees too, so far, and outside NSW the NVIS map takes over.

A made-up tree's height is invented too, so once typed it's drawn again from its family's range (`HEIGHT_M`):
a tea-tree a few metres tall, a wet-forest eucalypt tens of metres. Before, every made-up tree had a dry
eucalypt's height, which the height bands then turned into a eucalypt whatever its family.

Families are names, not models: `treefx.FAMILY_BANDS` says which models draw each family, and a family
whose own models aren't built yet borrows the nearest ones that are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt, maximum_filter

DRY_SMOOTH, DRY_ROUGH, SNOW_GUM = "dry_smooth", "dry_rough", "snow_gum"
WET, RAINFOREST, CONIFER = "wet", "rainforest", "conifer"
TEA_TREE, WATTLE, BLACKWOOD, SHE_OAK, BANKSIA = "tea_tree", "wattle", "blackwood", "she_oak", "banksia"
REGROWTH, PINE, PLANTATION, WILLOW, MALLEE = "regrowth", "pine", "plantation_euc", "willow", "mallee"
FAMILIES = (DRY_SMOOTH, DRY_ROUGH, SNOW_GUM, WET, RAINFOREST, CONIFER, TEA_TREE, WATTLE, BLACKWOOD, SHE_OAK,
            BANKSIA, REGROWTH, PINE, PLANTATION, WILLOW, MALLEE)
# A made-up tree's height by family: (mean, spread, lowest, highest) in metres. Canopy-model crowns keep their
# measured heights.
HEIGHT_M = {
    DRY_SMOOTH: (18.0, 4.5, 8.0, 32.0), DRY_ROUGH: (18.0, 4.5, 8.0, 32.0), SNOW_GUM: (10.0, 3.0, 4.0, 18.0),
    WET: (35.0, 9.0, 15.0, 70.0), RAINFOREST: (22.0, 6.0, 8.0, 40.0), CONIFER: (18.0, 5.0, 8.0, 30.0),
    TEA_TREE: (6.0, 2.0, 3.0, 12.0), WATTLE: (8.0, 2.5, 4.0, 14.0), BLACKWOOD: (14.0, 5.0, 5.0, 30.0),
    SHE_OAK: (9.0, 3.0, 4.0, 16.0), BANKSIA: (6.0, 2.0, 3.0, 12.0), REGROWTH: (10.0, 4.0, 4.0, 22.0),
    PINE: (22.0, 6.0, 8.0, 40.0), PLANTATION: (25.0, 7.0, 10.0, 45.0), WILLOW: (12.0, 3.0, 6.0, 20.0),
    MALLEE: (6.0, 2.0, 3.0, 10.0),
}

MIXED_SPLIT = 0.6  # share of the local tallest crown above which a crown in mixed wet forest is the eucalypt
MIXED_REACH_M = 50.0
PADDOCK_REACH_M = 50.0
EDGE_M = 20.0
EMERGENT_SHRUB_M = 8.0  # a crown in treeless country lower than this is tea-tree
# TASVEG's scattered trees are "< 5% crown cover": 0.05 x 10,000 m2 / (pi x 5 m x 5 m) is 6.4 trees a
# hectare with 10 m crowns.
PADDOCK_TREES_PER_HA = 6.0
# Canopy-model heights are capped by where a crown stands: forest (eucalypt, rainforest, plantation) has real
# 50-60 m trees, while a crown far taller than that in a paddock or town is an artefact (a mast, a cliff edge;
# 100 m and 65 m on the Huon Highway).
FOREST_HEIGHT_M, OTHER_HEIGHT_M = 90.0, 45.0
PLANTATIONS = frozenset("FPS FPH FPU".split())

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

# NVIS 7.0 Major Vegetation Subgroups (the raster's MVS numbers) by family; research/TREE_SPECIES.md section 6a.
# Dry open forests with a shrubby understorey are mostly stringybark and peppermint (rough bark); grassy and open
# woodlands mostly gums and boxes (drawn smooth). Grassland, water, bare ground and the cleared class keep any
# made-up tree there as a paddock gum: NVIS only types.
_SUBGROUPS = {
    RAINFOREST: (1, 2, 6, 11, 62),
    WET: (3, 54, 60),
    DRY_ROUGH: (4,),
    DRY_SMOOTH: (5, 7, 8, 9, 10, 16, 18, 19, 47, 48, 53, 56, 59, 65, 79),
    CONIFER: (12, 70),
    WATTLE: (13, 20, 21, 22, 23, 24, 25, 45, 51, 52),
    BLACKWOOD: (14,),
    TEA_TREE: (15, 28, 30, 32, 40, 49, 57, 58, 75, 80),
    SHE_OAK: (26, 71, 72, 73, 74),
    MALLEE: (27, 29, 55, 61, 66, 67, 68, 69),
    BANKSIA: (50,),
    REGROWTH: (90, 91),
}
NVIS_SUBGROUPS = {mvs: family for family, numbers in _SUBGROUPS.items() for mvs in numbers}
# Subgroups that say nothing about the trees standing in them: grassland, sedgeland, chenopod shrubland, water
# and bare ground, unclassified, unknown, and "Cleared, non-native vegetation, buildings" (towns and gardens:
# 80% of Dandenong Tourist Rd's trees). A tree there takes the family of the nearest cell with trees within
# NVIS_NEAREST_M, as a Tasmanian paddock tree takes its nearest woody community's; else a paddock gum.
NVIS_UNTYPED = frozenset({0, 17, 31, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 46, 63, 64, 92, 93, 97, 98, 99})
NVIS_NEAREST_M = 200.0

# SVTM: each vegetation formation's family, the default for its classes and PCTs. Formations not here
# (grassland, wetland, saltmarsh, chenopod shrubland, the alpine complex, and "Not classified") are untyped.
_SVTM_FORMATIONS = {
    "Rainforests": RAINFOREST,
    "Wet Sclerophyll Forests (Shrubby sub-formation)": WET,
    "Wet Sclerophyll Forests (Grassy sub-formation)": WET,
    # Stringybarks, peppermints and bloodwoods; the grassy forests and woodlands are gums and boxes, as for NVIS.
    "Dry Sclerophyll Forests (Shrubby sub-formation)": DRY_ROUGH,
    "Dry Sclerophyll Forests (Shrub/grass sub-formation)": DRY_SMOOTH,
    "Grassy Woodlands": DRY_SMOOTH,
    "Forested Wetlands": DRY_SMOOTH,  # river red gum
    "Semi-arid Woodlands (Grassy sub-formation)": DRY_ROUGH,  # black box, coolabah, poplar box
    "Semi-arid Woodlands (Shrubby sub-formation)": DRY_ROUGH,
    "Heathlands": TEA_TREE,
    "Arid Shrublands (Acacia sub-formation)": WATTLE,
}
# Vegetation classes whose trees aren't their formation's.
_SVTM_CLASSES = {
    "Subalpine Woodlands": SNOW_GUM,
    "Southern Wattle Dry Sclerophyll Forests": WATTLE,
    "Coastal Swamp Forests": TEA_TREE,  # paperbark and swamp mahogany
    "Coastal Floodplain Wetlands": TEA_TREE,  # paperbark, swamp turpentine and swamp oak
    "Eastern Riverine Forests": SHE_OAK,  # river oak
    "Mangrove Swamps": TEA_TREE,  # low, dense crowns
    "Coastal Heath Swamps": TEA_TREE,  # wet heath: tea-tree, paperbark and banksia
    "Brigalow Clay Plain Woodlands": WATTLE,
    "Riverine Plain Woodlands": WATTLE,  # weeping myall
    "Dune Mallee Woodlands": MALLEE,
    "Sand Plain Mallee Woodlands": MALLEE,
    "Semi-arid Sand Plain Woodlands": SHE_OAK,  # belah and black oak
    "Riverine Sandhill Woodlands": CONIFER,  # white cypress pine
    "Wallum Sand Heaths": BANKSIA,
}
# The trees a PCT's name may list, by family, ignoring case. Where patterns overlap the earlier wins, so a snow
# gum isn't a gum and a rough-barked apple isn't an apple. In wet forest a named eucalypt is the formation's own
# family; elsewhere its bark decides, as for TASVEG's dry forest.
_NAMED_TREES = (
    (SNOW_GUM, "Snow Gum|Sally"),
    (MALLEE, "Mallee"),
    (CONIFER, "Cypress Pine|Cypress|Pine"),  # Callitris; a hoop pine's PCT is rainforest
    (SHE_OAK, "She[- ]?Oak|Oak|Buloke|Belah|Casuarina"),
    (TEA_TREE, "Paperbark|Melaleuca|Tea[- ]?tree|Leptospermum|Kunzea|Broombush|Mangrove"),
    (WATTLE, "Wattle|Acacia|Mulga|Gidgee|Brigalow|Myall|Nelia|Yarran|Currawang|Boree|Lancewood|Ironwood|Coobah|"
             "Dead Finish"),
    (BANKSIA, "Banksia"),
    (DRY_ROUGH, "Rough-barked Apple|Ironbark|Stringybark|Peppermint|Box|Bloodwood|Mahogany|Messmate|Blackbutt|"
                "Tallowwood|Ash|Coolabah|Coolibah|Bangalay|Carbeen|Turpentine(?! Bush)"),
    (DRY_SMOOTH, "Gum|Apple|Angophora"),
)
_NAMED = re.compile("|".join(rf"(?P<{family}>\b(?:{p})\b)" for family, p in _NAMED_TREES), re.IGNORECASE)
# A western PCT's name gives its trees, its structure, then where it grows ("... tall open forest on lower slopes
# in the Pilliga Scrub", "... mainly in the Mulga Lands Bioregion"). Only the part before the place is read, as
# two bioregions are named after trees.
_PLACE = re.compile(r"\s(?:on|of|in|along|including|lining|fringing|mainly|with|from|between|near|the NSW)\s"
                    r"|\b(?:Brigalow Belt|Mulga Lands)\b")
# A name that gives no tree but says it's open country is treeless, whatever its class.
_TREELESS = re.compile(r"\b(?:grassland|herbfield|sedgeland|forbland|fen|bog|saltmarsh|marsh|meadow|lake|lagoon)s?\b",
                       re.IGNORECASE)
SVTM_NEAREST_M = 200.0  # as NVIS_NEAREST_M
SVTM_OUTSIDE = 65535  # the SVTM's no data, outside NSW
_NVIS_CODE = 100_000  # `SvtmMap.at` gives an NVIS subgroup outside NSW as this plus its number

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


def height_caps(codes: np.ndarray) -> np.ndarray:
    """The tallest a crown may be, by the community code it stands in: FOREST_HEIGHT_M in eucalypt forest,
    rainforest and plantations, OTHER_HEIGHT_M anywhere else (paddocks, towns, scrub, treeless, unmapped)."""
    codes = np.asarray(codes, dtype=object)
    forest = np.array([len(c) == 3 and (c[0] in "DWR" or c in PLANTATIONS) for c in codes], dtype=bool)
    return np.where(forest, FOREST_HEIGHT_M, OTHER_HEIGHT_M)


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

    def __init__(self, communities: list[Community], attributions: tuple = ()):
        self.communities = list(communities)
        self.attributions = attributions  # the sources to credit (`providers.base.Attribution`s)
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

    # What the scenery stage asks of any vegetation map (this and `NvisMap`).
    def families(self, xy: np.ndarray, height: np.ndarray, where: np.ndarray | None = None) -> np.ndarray:
        return families(xy, height, self, where)

    def keep_made_up(self, xy: np.ndarray, corridor: shapely.Geometry, rng: np.random.Generator, *,
                     mask_agreed: bool) -> np.ndarray:
        return keep_made_up(xy, self, corridor, rng, mask_agreed=mask_agreed)

    def height_caps(self, where: np.ndarray) -> np.ndarray:
        return height_caps(self.codes(where))


class NvisMap:
    """NVIS subgroups around one corridor (`providers.dcceew.nvis.Nvis`, loaded): each tree's family from the
    subgroup of its 100 m cell. It never removes a tree, and caps no heights: on the mainland every tree is
    made up, with a height drawn from its family."""

    def __init__(self, nvis, attributions: tuple = ()):
        self._nvis = nvis
        self.attributions = attributions
        cells = nvis.cells
        # Each untyped cell takes the subgroup of the nearest cell with trees, if one is within NVIS_NEAREST_M.
        woody = np.isin(cells, list(NVIS_SUBGROUPS))
        self._typed = cells.astype(np.int64)
        if woody.any() and not woody.all():
            dist, (ir, ic) = distance_transform_edt(~woody, return_indices=True)
            near = ~woody & (dist * nvis.cell_m <= NVIS_NEAREST_M)
            self._typed[near] = cells[ir[near], ic[near]]

    def __len__(self) -> int:
        """How many subgroups the corridor's cells hold (not counting no data, 0 and 99)."""
        return len(np.setdiff1d(np.unique(self._nvis.cells), [0, 99]))

    def at(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return self._nvis.subgroups(xy[:, 0], xy[:, 1]) if len(xy) else np.zeros(0, dtype=np.int64)

    def codes(self, where: np.ndarray) -> np.ndarray:
        return np.array([f"MVS{int(n):02d}" for n in where], dtype=object)

    def families(self, xy: np.ndarray, height: np.ndarray, where: np.ndarray | None = None) -> np.ndarray:
        """By the subgroup of each tree's cell, or for an untyped cell its nearest typed neighbour's."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        typed = self._nvis.sample(self._typed, xy[:, 0], xy[:, 1]) if len(xy) else np.zeros(0, dtype=np.int64)
        return np.array([NVIS_SUBGROUPS.get(int(n), DRY_SMOOTH) for n in typed], dtype=object)

    def keep_made_up(self, xy: np.ndarray, corridor, rng, *, mask_agreed: bool) -> np.ndarray:
        return np.ones(len(xy), dtype=bool)

    def height_caps(self, where: np.ndarray) -> np.ndarray:
        return np.full(len(where), FOREST_HEIGHT_M)


def svtm_families(pct) -> tuple[str, ...]:
    """The families a PCT's trees are drawn from, in equal shares (a family named twice gets two), or () for a
    PCT that says nothing about its trees. `pct` is a `providers.nsw.svtm.Pct`."""
    if pct.formation == "Rainforests":
        return (RAINFOREST,)
    wet = pct.formation.startswith("Wet Sclerophyll")
    head = _PLACE.split(pct.name, maxsplit=1)[0]
    named = tuple(WET if wet and m.lastgroup in (DRY_ROUGH, DRY_SMOOTH) else m.lastgroup
                  for m in _NAMED.finditer(head))
    if named or _TREELESS.search(head):
        return named
    base = _SVTM_CLASSES.get(pct.veg_class, _SVTM_FORMATIONS.get(pct.formation))
    return (base,) if base else ()


def _unit_hash(xy: np.ndarray) -> np.ndarray:
    """A number in [0, 1) for each point, fixed by where it is: picks a tree's share without a generator."""
    return np.modf(np.abs(np.sin(xy[:, 0] * 12.9898 + xy[:, 1] * 78.233)) * 43758.5453)[0]


class SvtmMap:
    """SVTM PCTs around one corridor (`providers.nsw.svtm.Svtm`, loaded with an `SVTM_NEAREST_M` halo), with
    `nvis_map` (an `NvisMap`, or None) for any of it outside NSW. Like NVIS it never removes a tree, and caps no
    heights."""

    def __init__(self, svtm, nvis_map: NvisMap | None = None, attributions: tuple = ()):
        self._svtm, self._nvis = svtm, nvis_map
        self._shares = {n: svtm_families(p) for n, p in svtm.table.items()}
        typed = np.zeros(SVTM_OUTSIDE + 1, dtype=bool)
        typed[[n for n, fams in self._shares.items() if fams]] = True
        # Each block's cells, where untyped ones inside NSW take the PCT of the nearest typed cell within
        # SVTM_NEAREST_M; the block's halo lets that search see past its edge.
        self._typed = {}
        counts = np.zeros(SVTM_OUTSIDE + 1, dtype=np.int64)
        for key, cells in svtm.blocks.items():
            counts += np.bincount(cells.ravel(), minlength=SVTM_OUTSIDE + 1)
            woody, out = typed[cells], cells.copy()
            if woody.any() and not woody.all():
                dist, (ir, ic) = distance_transform_edt(~woody, return_indices=True)
                near = ~woody & (cells != SVTM_OUTSIDE) & (dist * svtm.cell_m <= SVTM_NEAREST_M)
                out[near] = cells[ir[near], ic[near]]
            self._typed[key] = out
        self._pcts = int((counts[1:SVTM_OUTSIDE] > 0).sum())
        self._leaves_nsw = bool(counts[SVTM_OUTSIDE]) and nvis_map is not None  # the sea counts
        self._own, self._nvis_typed = attributions, False

    @property
    def attributions(self) -> tuple:
        """The SVTM's, and NVIS's once `families` has typed a tree by it."""
        return self._own + (self._nvis.attributions if self._nvis_typed else ())

    def __len__(self) -> int:
        """How many PCTs the corridor's cells hold (not counting cleared land), and NVIS subgroups if it leaves
        NSW."""
        return self._pcts + (len(self._nvis) if self._leaves_nsw else 0)

    def at(self, xy: np.ndarray) -> np.ndarray:
        """The PCT at each point; outside NSW, _NVIS_CODE plus the NVIS subgroup (or SVTM_OUTSIDE without NVIS)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        where = self._svtm.pcts(xy[:, 0], xy[:, 1]) if len(xy) else np.zeros(0, dtype=np.int64)
        outside = where == SVTM_OUTSIDE
        if self._nvis is not None and outside.any():
            where[outside] = _NVIS_CODE + self._nvis.at(xy[outside])
        return where

    def codes(self, where: np.ndarray) -> np.ndarray:
        return np.array([f"MVS{int(n) - _NVIS_CODE:02d}" if n >= _NVIS_CODE else "" if n == SVTM_OUTSIDE
                         else f"PCT{int(n)}" for n in where], dtype=object)

    def families(self, xy: np.ndarray, height: np.ndarray, where: np.ndarray | None = None) -> np.ndarray:
        """By the PCT of each tree's cell (or for an untyped cell its nearest typed neighbour's), drawing among
        the families it names; outside NSW, NVIS's."""
        xy, height = np.asarray(xy, dtype=float).reshape(-1, 2), np.asarray(height, dtype=float)
        out = np.full(len(xy), DRY_SMOOTH, dtype=object)
        if not len(xy):
            return out
        pct = self._svtm.sample(self._typed, xy[:, 0], xy[:, 1])
        outside = pct == SVTM_OUTSIDE
        if self._nvis is not None and outside.any():
            out[outside] = self._nvis.families(xy[outside], height[outside])
            self._nvis_typed = True
        u = _unit_hash(xy)
        for n in np.unique(pct[~outside]):
            fams = self._shares.get(int(n), ())
            if fams:
                mine = np.flatnonzero(pct == n)
                out[mine] = np.array(fams, dtype=object)[np.minimum((u[mine] * len(fams)).astype(int), len(fams) - 1)]
        return out

    def keep_made_up(self, xy: np.ndarray, corridor, rng, *, mask_agreed: bool) -> np.ndarray:
        return np.ones(len(xy), dtype=bool)

    def height_caps(self, where: np.ndarray) -> np.ndarray:
        return np.full(len(where), FOREST_HEIGHT_M)


def made_up_heights(families: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A height for each made-up tree from its family's range (`HEIGHT_M`)."""
    families = np.asarray(families, dtype=object)
    out = np.empty(len(families))
    for family in sorted(set(families.tolist())):
        mine = families == family
        mean, spread, lo, hi = HEIGHT_M[family]
        out[mine] = np.clip(rng.normal(mean, spread, int(mine.sum())), lo, hi)
    return out


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
