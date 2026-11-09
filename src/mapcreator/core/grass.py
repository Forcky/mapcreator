"""Ground classes: what covers the ground beside the road, from a vegetation map, and how Custom Shaders Patch's
Grass FX draws it.

Each class is a terrain material with its own ground texture (`textures.GROUND_COVERS`), and a Grass FX
configuration assigned by material (`[GRASS_FX_ADJUSTMENT_n] MATERIALS =`), the way most of CSP's shipped track
configs vary their grass. It works on CSP 0.2.11, and the ground's colour changes by class even where no grass is
drawn: Grass FX takes each blade's colour from the ground under it.

| Class | Material | Grass FX |
|---|---|---|
| Mown verge (the 2 m strip), lawn, saltmarsh, cushion moor | `grass`, `terrain_short` | A |
| Pasture | `terrain_pasture` | base `[GRASS_FX]` |
| Native tussock grassland | `terrain_tussock` | B |
| Grassy dry woodland | `terrain_woodland` | B at half weight: sparse tussock |
| Buttongrass moorland, sedgeland; wetland | `terrain_moor`; `terrain_wet` | C (wetland on its own darker ground) |
| Bracken fernland; regrowth, weeds and easements | `terrain_bracken`; `terrain_rough` | D; D at 0.6 |
| Forest, plantation, scrub | `terrain_forest` | none |
| Forest seen from afar | `terrain_canopy` | none |
| Sand, mud, rock, water | `terrain_bare` | none |
| Not mapped (a gap in the map) | `terrain` (today's) | base |

- Only four extra configurations exist ("only up to a D config is possible atm", CSP's Grass FX docs), so wetland
  shares C with moorland, and its own ground tells them apart.
- Forest more than CANOPY_FROM_M from the road is drawn dark canopy green, not leaf litter: out there made-up
  trees thin to a quarter of the cover and fewer (`scenery.TREE_BANDS`), and none stand past the scenery reach,
  so litter-brown ground would show as bare brown hills. The change is dithered over CANOPY_BLEND_M.
- Under the verge's mown strip the class is always A. The skirt beyond it takes the class of the ground it
  stands on (`mesh.terrain_meshes(fold_skirt=True)`), so in forest only the verge is mown grass.
- Without a vegetation map (the mainland so far, or a failed fetch), none of this applies: the terrain is one
  `terrain` material and Grass FX keeps today's defaults.

Tasmania's classes come from TASVEG 5.0 community codes (`TASVEG`), as tree families do (`core.species`).

Grass FX keys (`grass_fx`) are only those CSP documents (cup.acstuff.club/docs/csp/tracks/grass-fx) or names in
its changelogs, each as used in CSP's own shipped track configs (acc-extension-config, commit 433c553, 3 Nov
2026). Every number is a starting point, to tune in game with CSP's Grass FX debug app (`docs/known-issues.md`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import shapely

from mapcreator.core import grassfx, scenery

PASTURE, SHORT, TUSSOCK, WOODLAND, MOOR, WET = "pasture", "short", "tussock", "woodland", "moor", "wet"
BRACKEN, ROUGH, FOREST, CANOPY, BARE, UNKNOWN = "bracken", "rough", "forest", "canopy", "bare", "unknown"


@dataclass(frozen=True)
class GroundClass:
    key: str
    material: str
    texture: str  # `textures.LIBRARY` name
    letter: str  # ends its terrain meshes' names: `1GRASS_T012P`. One capital letter never spells a surface KEY
    config: str | None  # its Grass FX configuration: "" the base [GRASS_FX], "A" to "D", None for no grass
    weight: float = 1.0  # the configuration's share (MAP = X, weight); the rest is the base configuration


CLASSES = (
    GroundClass(PASTURE, "terrain_pasture", "ground_pasture", "P", ""),
    GroundClass(SHORT, "terrain_short", "ground_short", "S", "A"),
    GroundClass(TUSSOCK, "terrain_tussock", "ground_tussock", "T", "B"),
    GroundClass(WOODLAND, "terrain_woodland", "ground_woodland", "G", "B", 0.5),
    GroundClass(MOOR, "terrain_moor", "ground_moor", "M", "C"),
    GroundClass(WET, "terrain_wet", "ground_wet", "W", "C"),
    GroundClass(BRACKEN, "terrain_bracken", "ground_bracken", "B", "D"),
    GroundClass(ROUGH, "terrain_rough", "ground_rough", "R", "D", 0.6),
    GroundClass(FOREST, "terrain_forest", "ground_forest", "F", None),
    GroundClass(CANOPY, "terrain_canopy", "ground_canopy", "K", None),
    GroundClass(BARE, "terrain_bare", "ground_bare", "X", None),
    GroundClass(UNKNOWN, "terrain", "ground", "U", ""),
)
BY_KEY = {c.key: c for c in CLASSES}
BY_MATERIAL = {c.material: c for c in CLASSES}
LETTERS = {c.material: c.letter for c in CLASSES}
# Materials that only exist with a vegetation map: one of these in a track means it has the grass pack.
PACK_MATERIALS = tuple(c.material for c in CLASSES if c.key != UNKNOWN)
# The mown verge strip and a single-lane gravel road's grass strip: configuration A wherever the pack is on.
VERGE_MATERIALS = ("grass", "gravel_narrow")
CANOPY_FROM_M = scenery.NEAR_M  # 250 m: where made-up trees thin to their far bands
CANOPY_BLEND_M = 60.0

# TASVEG 5.0 VEGCODEs by class (157 codes, labels from theLIST's REST renderer). Codes not named here go by their
# group's first letter (`tasveg_class`).
_TASVEG = {
    PASTURE: "FAL FAC FUM",  # agricultural land, pasture with native canopy, extra-urban miscellaneous
    SHORT: "FUR AUS ASS ASP HCM",  # urban lawns, saltmarsh, succulent saline herbland, sphagnum, cushion moor
    # Poa, Themeda and lowland grassland, grassy sedgeland, coastal grass and herbfield, rockplate grassland,
    # marram grass, highland grassy sedgeland.
    TUSSOCK: "GPL GPH GTL GCL GSL GHC GRP FMG MGH",
    # Buttongrass moorland (eastern, pure, emergent shrubs, undifferentiated, western, sparse on slopes),
    # Diplarrena and Restionaceae rushland, western lowland sedgeland, alpine sedgeland, heath and vegetation.
    MOOR: "MBE MBP MBS MBU MBW MBR MDS MRR MSW HSE HSW HHE HHW HCH HUE",
    WET: "ASF ARS AWU FSM AHF AHL AHS",  # sedgeland and rushland, wetland, spartina marsh, aquatic herbland
    BRACKEN: "FPF",  # Pteridium esculentum fernland
    ROUGH: "FRG FWU FPE",  # regenerating cleared land, weed infestation, permanent easements
    WOODLAND: "DVG DMW DGW",  # E. viminalis grassy forest and woodland, Midlands woodland, E. gunnii woodland
    FOREST: "FPS FPH FPU",  # plantations
    BARE: "OSM ORO AAP OAQ",  # sand and mud, lichen lithosere, alkaline pans; water (the water mesh covers it)
}
TASVEG = {code: key for key, codes in _TASVEG.items() for code in codes.split()}
# The rest of each group: dry and wet eucalypt forest, rainforest, non-eucalypt forest and scrub are forest (no
# Grass FX there); then grassland, moorland, highland treeless, wetland, natural bare ground, and modified land.
_GROUPS = {"D": FOREST, "W": FOREST, "R": FOREST, "N": FOREST, "S": FOREST, "G": TUSSOCK, "M": MOOR, "H": MOOR,
           "A": WET, "O": BARE, "F": PASTURE}


def tasveg_class(code: str) -> str:
    """The ground class of a TASVEG VEGCODE: by the code, else by its group's letter, else UNKNOWN."""
    code = (code or "").strip().upper()
    if code in TASVEG:
        return TASVEG[code]
    return _GROUPS.get(code[:1], UNKNOWN) if len(code) == 3 else UNKNOWN


def tasveg_areas(communities: Iterable) -> list[tuple[shapely.Geometry, str]]:
    """(polygon, class) for TASVEG communities (anything with `.polygon` and `.code`, e.g. `species.Community`)."""
    return [(c.polygon, tasveg_class(c.code)) for c in communities]


def _unit_hash(xy: np.ndarray) -> np.ndarray:
    """A number in [0, 1) for each point, fixed by where it is: dithers a change of class without a generator."""
    return np.modf(np.abs(np.sin(xy[:, 0] * 12.9898 + xy[:, 1] * 78.233)) * 43758.5453)[0]


class GroundMap:
    """Ground classes over a corridor, from areas each of one class: the class at any point."""

    def __init__(self, areas: list[tuple[shapely.Geometry, str]], attributions: tuple = ()):
        self.areas = [(g, k) for g, k in areas if not g.is_empty]
        self.attributions = attributions  # the sources to credit (`providers.base.Attribution`s)

    def __len__(self) -> int:
        return len(self.areas)

    def classes(self, xy: np.ndarray) -> np.ndarray:
        """The class key at each point: the first area holding it, else UNKNOWN."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        out = np.full(len(xy), UNKNOWN, dtype=object)
        if not len(xy) or not self.areas:
            return out
        # Each area against an index of the points, as `species.VegetationMap.at` does: shapely prepares the
        # query geometry, so a few hundred detailed polygons against many points is fast.
        polys, pts = shapely.STRtree(shapely.points(xy)).query([g for g, _ in self.areas], predicate="intersects")
        order = np.lexsort((polys, pts))
        pts, polys = pts[order], polys[order]
        first = np.unique(pts, return_index=True)[1]
        keys = np.array([k for _, k in self.areas], dtype=object)
        out[pts[first]] = keys[polys[first]]
        return out

    def materials(self, xy: np.ndarray, dist: np.ndarray) -> np.ndarray:
        """The terrain material at each point, `dist` metres from the centreline: its class's, with forest far
        from the road drawn as canopy (CANOPY_FROM_M, dithered over CANOPY_BLEND_M)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        keys = self.classes(xy)
        far = np.clip((np.asarray(dist, dtype=float) - CANOPY_FROM_M) / CANOPY_BLEND_M + 0.5, 0.0, 1.0)
        keys[(keys == FOREST) & (_unit_hash(xy) < far)] = CANOPY
        found, code = np.unique(keys.astype(str), return_inverse=True)
        return np.array([BY_KEY[k].material for k in found], dtype=object)[code]


# -- Grass FX (extension/ext_config.ini) -------------------------------------------------------------------------
# Keys and their sources (CSP: cup.acstuff.club/docs/csp/tracks/grass-fx; CFG: acc-extension-config at 433c553):
# - TEXTURE, TEXTURE_GRID: "TEXTURE = grass_fx/highlands.dds ; located next to config file", "TEXTURE_GRID = 8, 3
#   ; columns and rows" (CSP, "Custom textures"). A track's own atlas: CFG config/tracks/highforce.ini
#   (TEXTURE = GrassFX_COL.dds, TEXTURE_GRID = 8, 4).
# - MASK_BLUR: "Adjustments will be blurred according to this parameter ... values from 0 to 3" (CSP); 2 as in CFG
#   config/tracks/sakina_hill.ini. It softens the change of grass at a class's edge.
# - SHAPE_SIZE, SHAPE_TIDY, SHAPE_CUT, and TRIM_PERIOD with two values per SHAPE_ key, "a value at the beginning
#   of trim period" and "for the end", the period "duration in days" (CSP, "Shape settings and trimming").
#   "Higher size value lowers density. Default value, 1.0, corresponds to pretty low track grass."
# - MASK_MAIN_THRESHOLD, MASK_RED_THRESHOLD, MASK_MIN_LUMINANCE, MASK_MAX_LUMINANCE (CSP, "Spawn mask settings";
#   defaults 0.5, 0.05, 0.02, 0.35, and "-1, 0, -1, 1" spawns everywhere).
# - [GRASS_FX_CONFIGURATION_A] to _D: "the same parameters as [GRASS_FX]: shape and color mask", and
#   TEXTURE_BASE_CHANCE and TEXTURE_GROUP_n_CHANCE (CSP, "Local adjustments" and "Custom textures").
# - [GRASS_FX_TEXTURE_GROUP_n] with PIECE_n = column, row[, width, height] ("columns index start at 1, but rows
#   index start at 0"), PIECE_n_CHANCE and PIECE_n_WIND ("default value is 1") (CSP, "Custom textures").
# - [GRASS_FX_ADJUSTMENT_n] with MATERIALS and MAP = configuration, intensity ("with 0 intensity, basic [GRASS_FX]
#   configuration will be used. anything in-between will blend values") (CSP, "Local adjustments"). Numbered
#   sections as in CFG config/tracks/sakina_hill.ini ([GRASS_FX_ADJUSTMENT_1]).
# Left out on purpose: TEXTURE_BASE_CHANCE = 0 and COLOR_SAMPLE_MIP_LEVEL (they work only from CSP 0.3.0, and
# 0.2.11 is the baseline); [EXTRA_FX] FORCE_GRASSFX_GBUFFER_PASS (only if tall grass looks see-through against
# water in game); and keys only seen in shipped configs (HEIGHT_MULT, COLOR_FACTOR, EDGE_MULT, [GRASS_FX_AREA_n]).

# The atlas's pieces in four groups (`grassfx` draws them). Each: (piece, chance, wind); None leaves CSP's default.
GROUPS = (
    # 0: tussock: Poa fountains, a Themeda tuft with rust seed heads, rank grass gone to seed.
    ((grassfx.POA, None, None), (grassfx.POA_2, None, None), (grassfx.THEMEDA, 0.3, None), (grassfx.RANK, 0.5, None)),
    # 1: buttongrass: two hummocks, and one in flower on stalks that sway less.
    ((grassfx.BUTTONGRASS, None, None), (grassfx.BUTTONGRASS_2, None, None),
     (grassfx.BUTTONGRASS_FLOWERING, 0.15, 0.3)),
    # 2: rushes and sedges.
    ((grassfx.RUSHES, None, 0.5), (grassfx.SEDGE, None, None)),
    # 3: bracken, green and part browned.
    ((grassfx.BRACKEN, None, 0.6), (grassfx.BRACKEN_BROWN, 0.5, 0.6)),
)

# The base [GRASS_FX] is pasture: medium, untidy, green, with the odd rank tussock. Grass grows where the ground's
# colour passes CSP's default mask (green-ish, mid-dark), so pasture's dry patches stay short of it.
BASE = {"SHAPE_SIZE": "2", "SHAPE_TIDY": "0.3", "SHAPE_CUT": "0", "TEXTURE_BASE_CHANCE": "1",
        "TEXTURE_GROUP_0_CHANCE": "0.05"}
CONFIGURATIONS = {
    # A: the mown verge, lawn and short herbland: cut every six weeks, short and tidy just after.
    "A": {"TRIM_PERIOD": "42", "SHAPE_SIZE": "0.6, 1.2", "SHAPE_CUT": "1, 0.4", "SHAPE_TIDY": "1, 0.5",
          "TEXTURE_BASE_CHANCE": "1"},
    # B: native tussock grassland: tall clumps on straw to blue-green ground, so a mask that takes in straw but
    # not bare soil.
    "B": {"SHAPE_SIZE": "3.5", "SHAPE_TIDY": "0", "MASK_MAIN_THRESHOLD": "0.2", "MASK_RED_THRESHOLD": "0.3",
          "MASK_MAX_LUMINANCE": "0.5", "TEXTURE_BASE_CHANCE": "1", "TEXTURE_GROUP_0_CHANCE": "1"},
    # C: buttongrass moorland, sedgeland and wetland: olive and tan grow too; dark peat stays bare, by luminance.
    "C": {"SHAPE_SIZE": "3", "SHAPE_TIDY": "0", "MASK_MAIN_THRESHOLD": "-1", "MASK_RED_THRESHOLD": "0",
          "MASK_MIN_LUMINANCE": "0.05", "MASK_MAX_LUMINANCE": "1", "TEXTURE_BASE_CHANCE": "0.5",
          "TEXTURE_GROUP_1_CHANCE": "1", "TEXTURE_GROUP_2_CHANCE": "0.3"},
    # D: bracken and rough regrowth: fronds and rank grass.
    "D": {"SHAPE_SIZE": "2.5", "SHAPE_TIDY": "0", "TEXTURE_BASE_CHANCE": "1", "TEXTURE_GROUP_3_CHANCE": "1",
          "TEXTURE_GROUP_0_CHANCE": "0.3"},
}
MASK_BLUR = "2"


def grows(material: str) -> bool:
    """Whether Grass FX grows on a ground material (a class with a configuration, or the verge)."""
    c = BY_MATERIAL.get(material)
    return material in VERGE_MATERIALS if c is None else c.config is not None


def grass_fx(materials: Iterable[str]) -> tuple[dict[str, str], list[tuple[str, dict[str, str]]]]:
    """The grass pack's part of ext_config.ini, for a track whose KN5 has `materials`: the keys [GRASS_FX] gains
    beside its material lists, and the sections after it (texture groups, configurations A to D, and an
    adjustment per configuration and weight, naming the materials present that take it)."""
    present = set(materials)
    keys = {"MASK_BLUR": MASK_BLUR, "TEXTURE": grassfx.FILE,
            "TEXTURE_GRID": f"{grassfx.COLUMNS}, {grassfx.ROWS}", **BASE}
    sections: list[tuple[str, dict[str, str]]] = []
    for g, pieces in enumerate(GROUPS):
        values: dict[str, str] = {}
        for n, (piece, chance, wind) in enumerate(pieces):
            values[f"PIECE_{n}"] = piece.key()
            if chance is not None:
                values[f"PIECE_{n}_CHANCE"] = f"{chance:g}"
            if wind is not None:
                values[f"PIECE_{n}_WIND"] = f"{wind:g}"
        sections.append((f"GRASS_FX_TEXTURE_GROUP_{g}", values))
    sections += [(f"GRASS_FX_CONFIGURATION_{name}", dict(values)) for name, values in CONFIGURATIONS.items()]
    maps: dict[tuple[str, float], list[str]] = {("A", 1.0): [m for m in VERGE_MATERIALS if m in present]}
    for c in CLASSES:
        if c.config and c.material in present:
            maps.setdefault((c.config, c.weight), []).append(c.material)
    adjustments = [(cfg, w, mats) for (cfg, w), mats in sorted(maps.items()) if mats]
    for i, (cfg, w, mats) in enumerate(adjustments):
        sections.append((f"GRASS_FX_ADJUSTMENT_{i}", {"MATERIALS": ", ".join(mats),
                                                       "MAP": cfg if w == 1.0 else f"{cfg}, {w:g}"}))
    return keys, sections
