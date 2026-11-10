# What mapcreator writes for Assetto Corsa

Assetto Corsa has no official spec for its track files. The details below come from open-source readers and writers (see `THIRD_PARTY_NOTICES.md`) and community guides. Items marked **to confirm in game** are settled on the first real load.

## Folder layout

A job with `"layouts": "both"` produces:

```
<track_id>/
  <track_id>_forward.kn5         full geometry + forward logic dummies
  <track_id>_reverse.kn5         full geometry + reverse logic dummies
  models_forward.ini             [MODEL_0] FILE=<track_id>_forward.kn5
  models_reverse.ini
  forward/  data/{surfaces,map,lighting,groove,crew}.ini   ai/fast_lane.ai   map.png
  reverse/  (same)
  ui/forward/{ui_track.json, preview.png, outline.png}
  ui/reverse/(same)
  ATTRIBUTION.txt
  centreline.geojson             ODbL: the centreline, heights, widths and surface runs the track was built from,
                                 and the side roads' stubs
```

- **Single layout:** a job with `forward` or `reverse` gets `<track_id>.kn5` plus `data/`, `ai/`, `map.png` and `ui/` at the track root, with no `models_*.ini`.
- **Shared geometry:** each layout's KN5 carries the full geometry. A shared geometry KN5 plus small per-layout logic KN5s would halve the download, but no generator has shown that working yet. **To confirm in game.**

## KN5

- **Header:** version 6 (followed by an extra int32 of 0), little-endian.
- **Strings:** int32 length, then ASCII.
- **Textures:** DXT1 DDS with a mip chain down to 4 px, generated procedurally (`core/textures.py`).
- **Materials:** `ksPerPixel` with `ksAmbient`, `ksDiffuse`, `ksSpecular`, `ksSpecularEXP`, `ksEmissive` and `ksAlphaRef`, plus one `txDiffuse` sampler.
  - Scalars go in the property's first float, and 3-vectors in its float[3] slot.
- **Node tree:** one identity root dummy, with every mesh and every `AC_*` dummy as a flat, childless child. A mesh with children crashes the game.
- **Meshes:**
  - At most 65,535 vertices each, because indices are u16.
  - Each vertex is 44 bytes: position, normal, uv, tangent.
  - Triangles wind counter-clockwise when seen from above, so face normals point up.
  - lodIn and lodOut are both 0, meaning always drawn.
- **Axes:** AC x = east, y = up, z = −north, in metres around a local origin. That's a rotation, not a mirror, so winding is preserved.
- **Dummy matrices:** rows are the X, Y and Z axes, then the translation. Forward is +Z, up is +Y, and X = Y × Z.

## Names

| Mesh | Surface (`surfaces.ini` KEY) | Material: texture |
|---|---|---|
| `1ROAD_###` (sealed road) | ROAD | `road`: `road_marked` (edge lines and a dashed centre line), or `road_plain`: `road_plain` (edge lines only) on a single-lane seal |
| `1GRAVEL_###` (unsealed road) | GRAVEL | `gravel`: `gravel_road` (two lanes), or `gravel_narrow`: `gravel_road_narrow` (one lane, with a grass strip) |
| `1GRASS_V###L/R` (verge and skirt; with ground classes, the 2 m verge alone) | GRASS | `grass` |
| `1ROAD_J###`, `1ROAD_J###N` (side roads' stubs and their throats, sealed: two lanes, one lane) | ROAD | `road`: `road_marked`; `road_plain`: `road_plain` |
| `1GRAVEL_J###`, `1GRAVEL_J###N` (stubs, unsealed: two lanes, one lane) | GRAVEL | `gravel`: `gravel_road`; `gravel_narrow`: `gravel_road_narrow` |
| `1GRASS_J###` (stubs' verges and their flares' verges; without ground classes, the stubs' skirts too) | GRASS | `grass` |
| `1GRASS_T###` (terrain) | GRASS | `terrain`: `ground` |
| `1GRASS_T###P`, `…S`, `…T` and so on (terrain and skirt, with ground classes: each class's own meshes, named by their first chunk) | GRASS | the class's, e.g. `terrain_moor`: `ground_moor` (see "Grass FX" below) |
| `1WALL_RAIL_L/R###`, `1WALL_END_START/FINISH`, `1WALL_TRUNK_###` and `1WALL_POWER_n` (invisible, `isRenderable` 0), and `1WALL_BUILDINGS` | solid barrier (no KEY: any physics mesh named WALL) | |
| `RAIL_*`, `BUILDING_ROOF_*`, `WATER_*`, `POWER_*`, `FURNITURE_*` | visual only (no leading digit) | |

- Each physics mesh name matches exactly one KEY. There is deliberately no `WALL` KEY, because defining one is reported to make walls driveable. No KEY contains another (`GRAVEL` overlaps neither `ROAD` nor `GRASS`), and the validator checks both rules, and that the layout's `surfaces.ini` defines the KEY of every physics mesh.
- **Road chunks:** the road is cut into chunks of 200 m, and also wherever its look changes (sealed or unsealed, one lane or two). Neighbouring chunks share their boundary row of vertices, so there's no gap and no step where the seal ends. The chunk number `###` runs on across `1ROAD` and `1GRAVEL`, and each chunk's verges carry the same number.
- **Where the road is unsealed** comes from OSM's `surface` (`core/section.py`):
  - sealed: `asphalt`, `paved`, `concrete` (and `concrete:*`), `chipseal`, `sealed`, `paving_stones`, `sett`, `metal`;
  - unsealed: `unpaved`, `gravel`, `fine_gravel`, `compacted`, `pebblestone`, `dirt`, `earth`, `ground`, `mud`, `sand`, `rock`, `grass`, `woodchips`;
  - anything else, or no tag, takes the class of the longer stretch beside it (with nothing tagged at all: unsealed on `highway=track`, sealed otherwise);
  - then any stretch shorter than 50 m takes its longer neighbour's class, shortest first, so a 10 m culvert or an untagged bridge doesn't make a sliver. Bridges get no exception.
- **Cross-section:** the carriageway carries a crown of 2.5% on a seal and 4.5% on gravel (NATSPEC AUS-SPEC 0052, unsealed rural roads: "more than 4%"). Width and crown both change gradually over 60 m.
  - **Camber** (superelevation) comes from the DEM either side of the road. Its limit follows the width: 4% at or under 5.0 m, rising in a straight line to 7% at 6.4 m and wider (`section.camber_limit`). On a 4 m road the DEM's two samples are only 3 m apart and catch the hillside and the cut and fill beside it, so before the limit a third of Lower Swamp Road sat at 7%.
  - **Crown in bends:** as the camber grows, the high side turns until, at a camber equal to the crown, the road is one plane. A road under 6.4 m always keeps some crown: at least 0.5% fall either side of the centre at or under 5 m, fading to none at 6.4 m (`section.crown_kept`). So a single-lane road is never one tilted plane. At its 4% limit, a single-lane road, sealed or gravel, falls 4.5% on its low side and rises 3.5% on its high side.
  - Width comes from the `width` tag (raised to 3 m if under it), else `lanes`, else `narrow=yes` (4.5 m), else the class default. `lanes=1` is 4.0 m on either surface.
  - Sealed class defaults: twice the class lane width (secondary 6.4, tertiary 6.0, unclassified and residential 5.8 m).
  - Unsealed class defaults: trunk and primary 6.5, secondary 6.0, tertiary 5.5, unclassified, residential and road 5.0, track 4.0 m; a tagged lane is 3.0 m. AUS-SPEC 0052 asks for at least a 6.0 m formation, verges included, for a single-lane unsealed road and 7.5 m for two lanes, and our road gets 2 m of grass verge either side. These are guideline minima, to be calibrated against measured roads.
  - The job's width option (+15% or +30%) multiplies all of them.
- **Markings:**
  - A two-lane seal has edge lines and a dashed centre line (`road_marked`).
  - A single-lane seal (`lanes=1` or `lane_markings=no`) has no centre line, and keeps its edge lines (`road_plain`). Most Tasmanian single-lane seals have neither.
  - Gravel has no markings. A gravel run is one lane (`gravel_narrow`) with `lanes=1` or `narrow=yes`, or with no `lanes` tag when it's under 5.5 m wide as mapped (so unclassified and residential gravel at 5.0 m is one lane, and tertiary at 5.5 m is two).
  - Runs shorter than 50 m join their neighbours here too, within one surface, so a short single-lane bridge keeps the road's centre line.
- **Edge lines break across a side road's mouth** (the maintainer's decision, 10 Nov 2026): over the mouth's rows, between its flares' tangent points, the outer road vertex on that side takes a `u` just inside the painted line (`mesh.EDGE_BREAK_MARGIN_U`, 0.012, past the line's outer edge: 0.043 on `road_marked`, drawn for 6.4 m, and 0.062 on `road_plain`, drawn for 4 m; 1 less that on the right). The quad beside it then shows plain asphalt, and the line tapers out over the metre either side. No new texture or material. Gravel has no lines, so nothing changes there.
- **Textures:** `u` runs 0 to 1 across the carriageway, so lines and wheel paths sit in the same place whatever the width. `road_marked` and `road_plain` repeat every 12 m along the road; the gravel textures every 24 m, at 512 x 2048 px (about 1 cm a pixel). The gravel has a pair of darker, compacted wheel paths per lane with fewer stones, loose ridges between and beside them, and a few potholes. The one-lane gravel adds a patchy grass strip about 0.3 m wide down the middle and green fringes at the edges, for CSP's Grass FX. The palette is the mean colours of the CC0 photo texture `gravel_ground_01` (Poly Haven), a grey-brown.
- **Terrain:** `1GRASS_T###` reaches 700 m either side of the road, and the world ends there. It's one Delaunay surface over lattices of 4 m to 60 m out, 8 m to 140 m, 16 m to 300 m, then 32 m. It used to end at 300 m, and power towers further out stood on nothing.
  - **Ground classes:** where the region has a vegetation map for the ground (TASVEG in Tasmania), each terrain triangle takes the class of the ground near its centre (dithered, below), and each class's triangles are gathered from consecutive 400 m chunks into meshes of about 24,000 triangles (or 3.2 km with at least a quarter of that), named by their first chunk and the class's letter (`1GRASS_T012P`). So a track has about as many terrain meshes as without classes: Grasstree Hill 32 (33 without, 167 with a mesh per class per chunk). The skirt beyond the 2 m verge joins the terrain and takes its class too, so under forest only the verge is mown grass. One capital letter never spells a second surface KEY. Without a map, the terrain is one `terrain` mesh per chunk, as before.
  - **Reach:** the distances beside the road are one set (`core.reach`). Trees and buildings stand within 600 m of the centreline (the scenery reach). The terrain reaches 100 m further, so nothing at the scenery's edge stands at the world's edge, and the DEM is read 100 m beyond that (800 m). Power-line supports stand anywhere on the terrain, out to 695 m (see "Power lines").
  - Until the scenery reach, trees and buildings stopped at 250 m and the hills beyond them were bare.
- **Logic dummies:**
  - `AC_AB_START_L/R` and `AC_AB_FINISH_L/R` sit 0.5 m outside each road edge. L is left of the driving direction.
  - The start gate is 50 m from the layout's start. The finish gate leaves a run-off of real road after it: 250 m, or 15% of a short route, never under 50 m. With only 50 m, AC's AI crossed the finish, reached the end of its line, turned round and left the road (M3).
  - A gate that would stand in a side road's mouth moves a few metres along the road to clear it ("Side-road junctions" below).
  - **Grid:** `AC_PIT_k` and `AC_START_k` share each grid spot (`layout.grid`), up to 8 per layout.
    - Spot 0 is on the centreline 8 m before the start gate.
    - Behind it come pairs 8 m apart, 2 m either side of the centreline, where both half-widths are at least 3 m. Otherwise single file.
    - Spots go no closer than 8 m to the road's start.
    - So a road under 6.0 m wide at the start (every `lanes=1` road, and unsealed roads below secondary) gets 5 spots in single file, at 42, 34, 26, 18 and 10 m from its start, and `pitboxes` is 5: four AI opponents. Eight in single file would need the start gate at least 72 m in.
  - `AC_HOTLAP_START_0` sits 30 m before the start gate.
  - Spawns are 0.3 m above the road and face the driving direction. Confirmed in game at M3: the car lands on all four wheels, facing along the road.
- **Pits:** `pitboxes` in `ui_track.json` must equal the number of `AC_PIT_n` dummies (up to 8). Content Manager offers that many cars less one as AI opponents, so with one pit box there were none (M3).
- **Tags:** `ui_track.json` carries `point to point`, `real road`, the country and the region, plus `gravel` on a track with any unsealed road and `rally` when more than half of it is unsealed. The description says how much is unsealed ("5.2 km of 6.8 km unsealed."). How CSP decides a track is a rally track (for its Rally Stage mode, and for picking street tyres) isn't documented: it may be these tags or the surfaces. **To confirm in game.**

## Surfaces (`data/surfaces.ini`)

`ROAD` (friction 0.98) and `GRASS` (0.75, `grass.wav`, dirt 1, not a valid track, `_EXT_SURFACE_TYPE=GRASS`) on every track, and `GRAVEL` as `[SURFACE_2]` only on a track with unsealed road:

```ini
[SURFACE_2]
KEY=GRAVEL
FRICTION=0.80
DAMPING=0.01
WAV=gravel.wav
WAV_PITCH=0
FF_EFFECT=NULL
DIRT_ADDITIVE=0.8
IS_VALID_TRACK=1
BLACK_FLAG_TIME=0
SIN_HEIGHT=0.01
SIN_LENGTH=0.5
IS_PITLANE=0
VIBRATION_GAIN=0.1
VIBRATION_LENGTH=0.35
_EXT_SURFACE_TYPE=GRAVEL
_EXT_SURFACE_TYPE_MODIFIER=REGULAR
_EXT_PERLIN_NOISE=1
_EXT_PERLIN_OCTAVES=3
_EXT_PERLIN_PERSISTENCE=0.5
```

- **One KEY for every unsealed `surface` value** in v1. `unpaved` and `gravel` are 92% of the unsealed length in the Derwent Valley and the Huon. A second, looser `DIRT` key for `dirt`, `earth`, `ground`, `mud` and `sand` can follow if testing shows the difference is worth it.
- **Why these numbers.** Kunos's own loose surfaces are run-off, built to slow a car down: friction 0.7-0.8 with dirt 1 (Laguna Seca's and Zandvoort's `GRAVEL`, Black Cat County's `OFF`, as copied in third-party server repositories). People building gravel road stages settle on friction 0.87-0.90, damping 0-0.02 and dirt 0.8-1 (assettocorsamods.net threads "new CSP +Gravel" and "How to use surfaces in surfaces.ini", 2019-2024). This is a road to drive at pace, so its damping and dirt sit in the second range. Its friction no longer does:
  - **Friction 0.80**, the maintainer's choice (5 Nov 2026), not yet tried by hand. It started at 0.88, in the community range, but in game (5 Nov) AC's AI read that as only 10% less grip than the seal's 0.98: wheel slip as on a seal, about 106 km/h on average over Hartz Road's two-lane gravel (up to 140), and a median of 93-96 km/h on Lower Swamp Road's one lane (up to 156).
  - 0.80 is the top of the range of Kunos's own loose surfaces. Like 0.88, it's under the 0.89 above which one modder found gravel made the tyres squeal, and under 0.9, below which CSP 0.2.5's ParticlesFX makes "dusty tracks". Dirt on the tyres takes off about another 20% (one community skidpad test).
  - The AI also gets a pace hint of 0.8 over unsealed road (see the AI line's hints below). Only an AI re-test is planned for now (`docs/known-issues.md`); how 0.80 feels at the wheel is still to judge.
  - Dirt additive 0.8: dirt builds up on the tyres, progressively. At 0.7 or more, CSP's SurfacesFX treats the surface as loose even without the type (CSP's surface-tweaks page).
  - `gravel.wav` ships with AC (Black Cat County uses it). Never `grass.wav` on a road: CSP draws the dirt of a `grass.wav` surface as grass (CSP's dirt-on-tyres page).
  - `SIN_HEIGHT` 0.01 m and `SIN_LENGTH` 0.5 (it's inverse: 2 m) make a 1 cm ripple, drawn by CSP as Perlin noise (`_EXT_PERLIN_NOISE=1`) instead of a regular washboard.
  - Vibration sits between Black Cat County's `OFF` (0.05 / 0.35) and Laguna Seca's `GRAVEL` (0.5 / 0.6).
  - It's a valid track surface: it's the road, not a cut.
- **CSP's keys** (`_EXT_*`, cup.acstuff.club/docs/csp/tracks/surface-tweaks) say what the surface is for CSP's SurfacesFX, which a player turns on (offline only; off in CSP's shipped config). Vanilla AC ignores them, so the plain numbers above must make a good gravel road on their own.
- **No `WAV_PITCH=extended-0`.** That switches on CSP's extended track physics, on which vanilla AC crashes by design, and none of it is needed here.
- **All of it is to confirm in game** (`docs/known-issues.md`).

## Custom Shaders Patch (`extension/`)

Vanilla AC ignores this folder. `ext_config.ini` holds:

- **Grass FX and Rain FX:** grass grows on the grass and terrain materials, and the road gets puddles. With ground classes (Tasmania), Grass FX grows each class's own grass from our own atlas (see "Grass FX" below), and every ground material, grass or not, is rough for Rain FX.
  - The seal (`road`, `road_plain`) occludes grass, gets puddles and soaks.
  - Gravel gets puddles too and is rough (no reflections, only darker when wet). One-lane gravel (`gravel_narrow`) is also a grass material, so Grass FX grows blades on its grass strip (CSP's default colour mask grows grass only on green, mid-dark pixels); two-lane gravel occludes grass instead.
  - No tiling fix (`ksPerPixel_tilingfix`) on the gravel: it shifts the texture at random and would break the wheel paths.
- **`[TYRES_FX]`**, only on a track with gravel: the colours of dust and of dirt on the tyres, as three pairs (`DIRT_COLOR_A/B` from the gravel texture, `GRASS_COLOR_A/B` from the grass, `SOIL_COLOR_A/B` from the terrain's bare earth). A is the texture's mean colour and B 0.55 of it. A track's own section switches CSP's generic one (`config/tracks/generic/generic.ini`) off entirely (CSP 0.2.4), so ours carries all three pairs and the generic's four gains (`DIRT_GAIN`, `DIRT_FADE_GAIN`, `GRASS_GAIN`, `GRASS_FADE_GAIN`) at 1.0.
- **Water:** `[INCLUDE: common/materials_track.ini]` and `[Material_Water]` give the `water` material CSP's water shader, as `LAKE` or `SEA`. Vanilla AC shows a flat, glossy blue-green surface.
- **Tree FX:** `[TREES]` with `LIST_0=trees/trees.txt` and `SURFACE_MATERIALS=terrain, grass` (with ground classes, every class's material before `grass`), so CSP stands each tree on the ground. These are the track's only trees: the KN5 has none, so vanilla AC shows none.
  - Models, all procedural (`core/treefx.py`): three dry eucalypts, two of them also with rough bark (`euc_tall_rough`, `euc_spread_rough`: the same geometry), a wattle, a tall wet-forest eucalypt (`euc_wet`), a tea-tree (`tea_tree`), a rainforest myrtle (`myrtle`) and a radiata pine (`pine`). A track ships only the models its trees use.
  - Every model carries the same four textures, byte for byte: one bark atlas (four columns: gum, stringybark, pine, grey rainforest bark) and one leaf atlas (3 × 3 sprays), with normal maps. A model picks its cells by UV. Around a branch a bark column is used once, so on thick trunks it's stretched across; it tiles every 2.4 m along the branch.
  - The gaming PC measured about 0.3% of frame rate per extra model (12 models against 4, same trees, 4 Nov 2026), and CSP's tree load time doubling. Whether CSP shares byte-identical textures across models is untested.
  - Where the region has a vegetation map (TASVEG in Tasmania), each tree's family comes from its community (`core.species`), and its model from the family and its height (`treefx.FAMILY_BANDS`). Elsewhere the model comes from the height alone (`HEIGHT_BANDS`).
  - Tree FX trees have no physics, so the KN5 gets an invisible collider for each trunk a car can reach (`1WALL_TRUNK_###`, `scenery.trunk_colliders`). A trunk qualifies if it stands within 20 m of the road edge and is over 10 cm across.
  - Each collider is an eight-sided prism with its corners on the drawn trunk's circle, from 1 m below the ground to 3 m above it. Its radius is the model's trunk trailer in metres, times the tree's `size` and `width`.
  - The faces are wound outwards, as the house walls are. The prisms are grouped into a mesh per 400 m of road, under the vertex limit, at about 200-560 trunks per km.
- **Far trees are billboards.** Each model's three LODs share ten model heights, the rule Content Manager's converter uses (`AcTreeModelConverter`: `DISTANCE = height × 10 × (i + 1) / LODs`). The last LOD is capped at 270 m (`treefx.LOD_REACH_M`, 0.45 of the scenery reach): the three tallest models, the myrtle (27.3 m), the pine (30.1 m) and the wet-forest eucalypt (45.7 m), would otherwise keep meshes to 273 m, 301 m and 457 m, so their LODs shrink in step. LOD 0 ends 28-90 m out and the last 84-270 m, by model.
  - Past its last LOD, Tree FX draws a tree as a billboard baked from the model. CSP's wiki ("Tracks: Trees"): "With LODs, billboards for distant trees and heavy use of instancing CSP should be able to handle hundreds of thousands of trees". CSP's public tree shaders draw each billboard as 6 vertices, from 15 side views and a top view (`flgTree_bb.hlsl`, `flgTree_bb_array_mainobj_vs.fx`).
  - So from about 270 m to the scenery's 600 m, every tree is a billboard, the cheapest tree Tree FX draws. LOD 2, the cheapest mesh, is about 380-400 triangles.
  - Billboards fade in past the last LOD and fade out at a far distance (`gLODOutDistance` in the shader) that CSP's own, closed code sets. Neither CSP's docs nor its configs give it. **To confirm in game:** that billboards still draw 600 m out (`docs/known-issues.md`).
  - The player's `TREES_DISTANCE_MULT` (CSP `graphics_adjustments.ini`, 0.2 to 3.2) scales the LOD distances. Nothing in a track's own config sets them.

`extension/trees/`:

- **`trees.txt`:** one line per tree: `tree: <model>.bin; pos=x, y, z; angle = <deg>; size = <mult>; width = <mult>`.
  - The position is the tree's base, in AC coordinates.
  - `size` scales the model's own height; `width` scales it horizontally only.
  - The `configure:` lines at the top switch off CSP's own random variation.
  - No comment lines. CSP 0.2.11 read a `;` line without a colon as a model name, then loaded no trees at all (`Tree manifest is missing`).
- **`<model>.bin`:** a zip of `tree.ini`, one `.mesh` per LOD and the DDS textures. The byte layout is in `core/treefx.py`. It was re-implemented from Content Manager's tree converter, which is the reference writer.

### Grass FX (`core/grass.py`, `core/grassfx.py`)

Where the region has a vegetation map for the ground (TASVEG in Tasmania), the terrain is split into ground classes, each a material with its own procedural ground texture (`textures.GROUND_COVERS`, 512 px DXT1), and Grass FX gets a configuration per class, assigned by material. That's how most of CSP's shipped track configs vary their grass, and it works on CSP 0.2.11. The ground's colour changes by class even where no grass is drawn, and Grass FX takes each blade's colour from the ground under it. Pasture and towns stay as they are without the pack.

| Class | TASVEG codes | Material | Grass FX |
|---|---|---|---|
| Mown verge (the 2 m strip; and one-lane gravel's grass strip) | | `grass` (`gravel_narrow`) | A |
| Short herbland: saltmarsh, sphagnum, cushion moor | `AUS`, `ASS`, `ASP`, `HCM` | `terrain_short` | A |
| Pasture and towns, and a gap in the map | `FAL`, `FAC`, `FUM`, `FUR` (and other `F**`) | `terrain` (today's, `ground`) | base `[GRASS_FX]`: CSP's defaults |
| Native tussock grassland | `GPL`, `GPH`, `GTL`, `GCL`, `GSL`, `GHC`, `GRP`, `FMG`, `MGH` (and other `G**`) | `terrain_tussock` | B |
| Grassy dry woodland | `DVG`, `DMW`, `DGW` | `terrain_woodland` | B at 0.5 |
| Buttongrass moorland, sedgeland, alpine | `MB*`, `MDS`, `MRR`, `MSW`, `HSE`, `HSW`, `HHE`, `HHW`, `HCH`, `HUE` (and other `M**`, `H**`) | `terrain_moor` | C |
| Wetland | `ASF`, `ARS`, `AWU`, `FSM`, `AHF`, `AHL`, `AHS` (and other `A**`) | `terrain_wet` | C, on its own darker ground |
| Bracken | `FPF` | `terrain_bracken` | D |
| Regrowth, weeds, easements | `FRG`, `FWU`, `FPE` | `terrain_rough` | D at 0.6 |
| Forest, plantation, scrub | other `D**`, `W**`, `R**`, `N**`, `S**`; `FPS`, `FPH`, `FPU` | `terrain_forest`; beyond 250 m, `terrain_canopy` | none |
| Bare | `OSM`, `ORO`, `AAP`, `OAQ` | `terrain_bare` | none |

- **Why these:** CSP allows four configurations besides the base ("only up to a D config is possible atm"), so wetland shares C with moorland, and its ground tells them apart. Wet forest, rainforest, plantations and scrub grow none; grassy dry woodland grows sparse tussock.
- **Far forest:** forest more than 250 m from the road (`grass.CANOPY_FROM_M`, where made-up trees thin to their far bands) is a dark canopy green instead of leaf litter, dithered over 60 m, so far hills don't read as bare brown.
- **Pasture and towns give way** (in game, 9 Nov 2026): they looked almost as without the pack, yet that's where it cost frame rate (Grasstree Hill 5.4% slower, worst on pasture). Their ground was greener, so CSP's colour mask grew grass on about 86% of it against today's 36% (by a proxy of the mask, whose formula isn't published), at `SHAPE_SIZE = 2` with a 5% chance of tussock pieces. Now they're today's `terrain` with CSP's default grass, and towns (`FUR`) are with them.
- **Class edges are dithered:** a triangle takes the class of a point up to 6 m from its centre (`grass.EDGE_DITHER_M`; further out, more with the lattice, up to 30 m), in a direction fixed by where it is. On the axis-aligned lattice a mapped block's edge otherwise stepped into straight lines: a 0.65 ha blue gum patch (`DGL`) beside the road at Grasstree Hill's 11.5 km showed as a brown rectangle on the paddock.
- **No rows:** each class's texture (pasture's aside) repeats every 16 m, with patches small beside the repeat, and where grass grows its texture coordinates are warped by up to 4 m in long, smooth waves of world position (`mesh.warped_uv`). With one big patch every 8 m, the repeat lined tussocks up in faint rows across Lyell Highway's moorland plain. The warp is a function of position alone, so meshes of a class meet without a seam. Forest, far canopy and bare ground grow no grass and aren't warped: warped coordinates deflate worse (about 3 MB more in each of Lyell Highway's KN5s if all its ground were).
- **Without a map** (the mainland so far, or a failed TASVEG read) the terrain is one `terrain` material and Grass FX keeps CSP's defaults: the track is as before.

Keys, all from CSP's Grass FX page (cup.acstuff.club/docs/csp/tracks/grass-fx) and used as in CSP's shipped track configs (acc-extension-config at commit `433c553`):

- `[GRASS_FX]` gains `MASK_BLUR = 2` ("Adjustments will be blurred according to this parameter", 0-3; as `sakina_hill.ini`; it blends configurations across a class's edge, and changes nothing inside pasture), `TEXTURE = grass_fx/mapcreator.dds` and `TEXTURE_GRID = 8, 4` (a track's own atlas, "located next to config file", as `highforce.ini`'s), and for pasture's base configuration only `TEXTURE_BASE_CHANCE = 1` and every `TEXTURE_GROUP_n_CHANCE = 0`: CSP's default shape and its base blades, no pieces.
- `[GRASS_FX_TEXTURE_GROUP_0]` to `_3` (tussock, buttongrass, rushes and sedges, bracken) list the atlas's pieces as `PIECE_n = column, row[, width, height]`, "columns index start at 1, but rows index start at 0", with `PIECE_n_CHANCE` and `PIECE_n_WIND` where they differ from CSP's defaults.
- `[GRASS_FX_CONFIGURATION_A]` to `_D` set `SHAPE_SIZE`, `SHAPE_TIDY`, `SHAPE_CUT`, `TRIM_PERIOD` (A: cut every 42 days, two values per key for the start and end of the period), the colour mask (`MASK_*`: B takes in straw, C olive and tan but not dark peat), `TEXTURE_BASE_CHANCE` and all four `TEXTURE_GROUP_n_CHANCE` (those unused as 0: what CSP does with one left out isn't documented). No `SHAPE_SIZE` is under CSP's default of 1, since "Higher size value lowers density": A, the mown verge beside the car, is 1 to 1.3.
- `[GRASS_FX_ADJUSTMENT_n]` give each configuration (and weight, as `MAP = B, 0.5`) the materials that take it. Numbered, as `sakina_hill.ini`'s are.
- **Not used:** `TEXTURE_BASE_CHANCE = 0` and `COLOR_SAMPLE_MIP_LEVEL`, which work only from CSP 0.3.0 (0.2.11 is the baseline); `[EXTRA_FX] FORCE_GRASSFX_GBUFFER_PASS`, until tall grass is seen to look see-through against water; keys only seen in shipped configs (`HEIGHT_MULT`, `COLOR_FACTOR`, `EDGE_MULT`, `[GRASS_FX_AREA_n]`). No seasons: heights are fixed.
- Every number is a starting point, to tune in game with CSP's Grass FX debug app (`docs/known-issues.md`).

The atlas, `extension/grass_fx/mapcreator.dds` (`core/grassfx.py`): 8 x 4 cells of 256 px, 2048 x 1024, DXT5 with 9 mips, 2.8 MB, the same for every track.

| Cells (column from 1, row from 0; size) | Piece | Colour |
|---|---|---|
| 1-8, 0 | Base blades: eight clumps of 12-20 | Green only |
| 1, 1 and 3, 1 (2 x 1) | *Poa* tussock: a fountain of fine arching leaves | Green only |
| 5, 1 | *Themeda* tuft with rust seed heads | Leaves green only, heads coloured |
| 6, 1 | Rank grass gone to seed | Green only |
| 7, 1 (2 x 1) | Bracken fronds | Coloured |
| 1, 2 and 3, 2 (2 x 1) | Buttongrass hummock: a dense dome of wiry leaves | Green only |
| 5, 2 (1 x 2) | Buttongrass hummock with flower stalks and buttons | Hummock green only, stalks and buttons coloured |
| 6, 2 (1 x 2) | Tall rush clump | Green only |
| 7, 2 (2 x 1) | Bracken, part browned | Coloured |
| 1, 3 (2 x 1) | Sedge clump | Green only |
| 3-4, 3 and 7-8, 3 | Spare | |

- **Green only** is how CSP's own base blades are made: red and blue zero, a green shading over alpha, and CSP colours them from the ground. A piece with any red or blue keeps its own colour instead (CSP 0.3.0 changelog: "Customly colorized pieces (if color has red or blue component disabling surface color sync) don't fade to green with distance", so on 0.2.11 they do). In the pieces with coloured heads or stalks, every pixel that's leaf is made green only, so the antialiased edge where a stalk crosses a leaf doesn't keep its own bright green.
- Blades are tapered strips, drawn at twice the size and scaled down with alpha premultiplied, rooted on each piece's bottom edge as CSP's are, and shaded darker towards the root (CSP's own run from about 205 at the root to 230 at the tip). Each mip level's transparent pixels take their piece's colour (green only where it's synced), so a blade's edge never filters towards black.
- Our own generator, after the idea of CSP's MIT-licensed `ac-grass-gen`; no code from it is used.

## AI line (`ai/fast_lane.ai`, version 7)

- **Header:** `7, N, 0, 0`.
- **Points:** N × (position[3], cumulative length, id).
- **Extras:** an int32 N, then N × 18 floats:
  - speed (m/s), gas, brake
  - obsolete lateral g (0), radius
  - side left, side right, camber (0), direction (±1)
  - normal[3], length to the next point
  - forward[3], tag (0), grade
  - Side left and side right are the distances from the line to each road edge, not the road's half-widths: Content Manager draws the road's outline as each point plus and minus its side vector times them (AcTools.Render, `Kn5SpecificSpecial/AiLaneObject.cs`). So they sum to the road's width, and on the racing line they differ.
  - Radius, direction, speed, gas, brake, grade and forward are all the racing line's own, not the centreline's.
- **Grid:** `hasGrid = 0`; the game builds and caches the grid itself.
- **End:** the AI line stops 40 m before the road's end (`AI_END_SHORT_M`), short of the invisible end wall.
- **Hints (`data/ai_hints.ini`):** AC's AI sets its own speed (M3), so hints are the way to slow it. Three `[HINT_n]` sections run from the finish gate to the end of the line, as fractions of the spline: pace 0.5, then 0.2 from 150 m before the line's end, then **0** for its last 60 m. On a track with unsealed road or a risky bend, their hints come before them (below). The sections are numbered from `HINT_0` in order along the line.
  - The fractions are of the line's own length, measured from the points written. The racing line is shorter than the road (Grasstree Hill: 12,737 m against 12,772 m), so a place on the road falls at a later fraction of it than of the centreline.
  - A hint's VALUE scales the speed AC works out from the line's curvature, not the car's speed. Measured in game (4 Nov): AC's own normalised position matches our line exactly in both layouts, and the hints fall where we put them. Yet down Grasstree Hill's straight reverse run-off (radius 287-1,000 m) 0.2 and 0.05 left the AI at about 100 km/h, while forward, where the run-off bends (radius 19-210 m), the same hints slowed it to 12-15 km/h.
  - Kunos's shipped hints fit that reading: they're all short ranges around corners, 0.6 to 0.92 (`ks_laguna_seca`'s Corkscrew, `ks_black_cat_county`, `ks_nurburgring`), and none sits on a straight or near 0. There's no Kunos or CSP documentation of the file.
  - So only 0 stops a car on a straight. With 0.05 the AI crawled past the end of its line at about 6 km/h and off the road; with 0 it stops on the road about 60 m before the end of its line, about 100 m short of the end wall, in both layouts. On the straight reverse run-off the 0.2 stretch still does nothing, and the AI brakes from about 100 km/h in the 45 m before the 0 hint: hard, but straight and on the road.
  - Several cars queue behind the first one stopped. That leaves them short of the 0 hint, where the pace is still 0.2, so only AC's car-following holds them, creeping at about 1 km/h. In two of the four multi-car runs that reached the run-off (Brinktop Road, and a later race on Grasstree Hill), one crept into the car ahead (slight damage to both, 1.3). A community tutorial on AI lines gives the same advice: a 0 hint near the end of a point-to-point line, which should end after the last timing gate (Overtake.gg, "Tutorial on making 2-way traffic (or other) AI lines and layouts").
  - Earlier: with a single hint of 0.5 and the line running to the road's end, the AI reached the end wall at about 45 km/h, and AC put it back at the start (3 Nov).
  - Kunos's own point-to-point track, `ks_drag`, sidesteps the problem: one straight AI line runs 2,469 m, past every layout's finish (`AC_OPEN_FINISH_L/R`, 204 m to 2,004 m along it), so even the 2,000 m layout leaves 465 m of line after its finish.
- **Hints on gravel:** one `[HINT_n]` of pace **0.8** (`layout.AI_GRAVEL_PACE`) over each unsealed run before the finish, by the maintainer's decision of 5 Nov 2026. With no hint, and GRAVEL at friction 0.88, AC's AI overdrove the gravel in game (5 Nov), at about tarmac pace (Surfaces above). On Lower Swamp Road in reverse, all 5 cars jumped the crest the validator warns about, about 1.8 km in (a 175 m vertical radius), and took about 70 damage each.
  - The runs are the stretches the `1GRAVEL_###` meshes cover (the section stage's `unsealed`, cut where the surface changes), in each layout's own driving order. The hints are in the same fractions of the line's own length as the run-off's.
  - Each is cut at the finish gate: from there the run-off ramp sets the pace, so the two never overlap, and a run wholly after the finish gets none. Unsealed stretches under 50 m have already joined their neighbours (`section.settle`), so there's no separate minimum.
  - A sealed track's `ai_hints.ini` is exactly as before: the run-off ramp alone.
  - **It may not slow the AI on a straight.** A hint scales the speed AC allows for the line's curvature (above): on the straight reverse run-off, 0.2 and 0.05 did nothing, and only 0 stopped a car. So 0.8 should slow the AI into gravel bends, but may do little on a straight, or over a straight crest like Lower Swamp Road's. GRAVEL's lower friction is the other lever. To check in game (`docs/known-issues.md`).
- **Hints at risky bends:** a bend AC's AI would take faster than the road allows gets a hint of its own (`ai_line.bend_hints`). The AI sets its corner speed from the line's radius and the car's grip, about 0.9 g in an Abarth 500 (110-120 km/h in Lyell Highway's 109-135 m left-hander at 0.7 km, in game on 9 Nov), and doesn't allow for a crest or a descent. There, on lap 1, most of an 8-car field left the road: the bend starts just over a crest and runs down an 8.5% grade, with no banking. Nor does it allow for a crest in its braking zone. On Lyell Highway reverse at 14.0 km (in game on 10 Nov) a 44-66 m left-hander starts 40 m past a 220-280 m crest and runs down 15%: the AI plans to brake from about 130 km/h at the crest to its 71 km/h corner speed, but with the tyres carrying half the car over the crest it can't, and all 8 cars came into the bend at 86-99 km/h and left the road.
  - **The rule,** every metre of the line as written to `fast_lane.ai`. The AI's likely speed v_AI is a speed profile at 0.9 g (`BEND_GRIP`) up to 200 km/h, with its braking and acceleration. The car follows it, except over a crest in a braking zone (`arrival_speeds`): where the AI brakes, the car brakes at the straight-line rate the profile plans with (8 m/s², `BRAKE`) times the share of its weight the tyres still carry, 1 - v²/(R_v g) at its own speed v (R_v the crest's vertical radius), plus g times the grade, so it comes off the crest faster than planned. At the car's speed the grip left is 0.9 g less two things and plus one: a crest takes load off the tyres, leaving 1 - v²/(R_v g) of it; a descent takes g times its grade, to hold the speed on the brakes; banking into the bend adds g times its fall. Where the grip left falls short of v²/R, the car should be slower by the square root of the shortfall. A hint scales the speed the AI works out from the line's curvature, sqrt(0.9 g R), so the pace is the speed the grip allows over that one; and where the car comes in faster than the AI planned, lower again by v_AI/v, since the AI's lower target still leaves the car overshooting it by about as much.
  - **Where the car keeps to the plan, nothing changes.** With no crest in the braking zone the car brakes at least as hard as the plan does, so its speed is the AI's exactly and the pace is what it was before; so is every `ai_hints.ini` without such a bend. The car's braking isn't shared with cornering, as the plan's is, so in a bend's entry it can catch the plan up: the overshoot is the least the crest costs, and with the hint applied the car comes in within what the grip allows (`tests/test_bend_hints.py`).
  - **Only where it matters:** where the pace falls under 0.9 for at least a second at the car's speed (`BEND_TRIGGER`, `BEND_MIN_S`). Smaller or shorter, the shortfall slides a car out by less than 0.84 m, inside the 1.3 m the line's margin keeps for the AI's own drift at speed. Only in bends of up to 250 m: on the run-off the hints slowed the AI in bends of 19-210 m and did nothing on a straight of 287-1,000 m. And never below the bend's advisory speed, sqrt(127 R (e + 0.35)) km/h (as for the curve markers), which is 0.62 of the AI's on an unbanked bend.
  - **The hint** holds the lowest pace in that stretch (to 0.01, rounded down) and eases in and out over 50 m either side, in two steps of a third each. Where a crest cuts the braking, the stretch starts where the car first comes into the bend faster than the grip allows, so it covers the bend's way in and its tightest part, and the ramp in falls on the crest, where it does nothing (hints don't act on a straight).
  - **Where it lands:** the distances are along each layout's own AI line, as read back from its `fast_lane.ai`, and the fractions are of that line's length, so a reverse layout's hints fall on its own bends (a test drives a synthetic road both ways). In game the forward hint at 0.7 km fell at 692-732 m along the road, where it was put (10 Nov).
  - **The banking** is the superelevation, the crown left out: the crown falls into a bend on its inside half and away from it on its outside half, and the AI and the cars beside it use both. On coarse DEM stretches it's 0 (`docs/known-issues.md`).
  - **With the other hints:** where a bend's hint and the gravel's overlap, the lower pace holds there; they're never applied on top of each other. Up to the finish only, as the gravel's. A track with no such bend writes exactly the `ai_hints.ini` it did before.
  - **On the example roads:** Lyell Highway forward 3 bends (0.82-0.87; at 0.7 km 0.87 over 688-730 m along the line, easing in from 638 m and out to 780 m, as before), reverse 9 (0.65-0.87, two of them braking over a crest: 0.65 at 13.93-13.98 km and 0.73 at 29.73-29.78 km), the Esperance Coast Road 2 (0.78 and 0.79, one of them banked 7% the wrong way by the DEM), and none on Grasstree Hill, Brinktop Road, Lower Swamp Road or Mount Dandenong Tourist Road. On Lyell Highway that's 23-66 m at the full pace each (bends of 43-159 m at their tightest, over or just past crests, 3-15% downhill), and roughly 0.7 s (forward) and 3.9 s (reverse) on a 26-minute run, reckoned as the AI's likely speeds capped by the hints. **To check in game** (`docs/known-issues.md`).
- **The line is a racing line** across the road's full width (`core/racing_line.py`), by the maintainer's decision of 4 Nov 2026: out-in-out through the bends, as on a closed road, not kept to the left lane.
  - It's a minimum-curvature line after Heilmeier et al., "Minimum curvature trajectory planning and control for an autonomous race car" (Vehicle System Dynamics 58(10), 2020, doi:10.1080/00423114.2019.1631455), re-implemented from the paper. TUM's implementation is LGPL-3.0 and wasn't used.
  - Offsets from the centreline at stations 2 m apart, chosen to minimise the line's summed squared curvature, linearised around the line found and solved again, three times.
  - **Its margin from each road edge grows with the speed** (`racing_line.edge_margin`): 1.2 m at or below 60 km/h (`EDGE_MARGIN_M`, to the line: a car 1.7 m wide has 0.35 m to spare), rising in straight lines to 2.5 m at 120 km/h (`EDGE_MARGIN_FAST_M`) and 3.3 m at 200 km/h (`EDGE_MARGIN_TOP_M`), and held there above it. On the inside of a tight bend it also stays within half the bend's radius.
    - Why: AC's AI tracks the line less closely the faster it goes. On Grasstree Hill forward (in game, 4-5 Nov) the line entered the right-hander at about 5.95 km at 127 km/h, 1.2 m from the outside edge, and the AI ran 1.0-1.3 m outside it, so two wheels were off before the turn-in: alone it crashed out, and in an 8-car race every car put two tyres off there and 4 crashed out. On Brinktop Road at about 5.6 km, a near-straight (radius 650-700 m) taken at 130 km/h, the line came within 1.2 m of the edge too, and 3 cars of 8 put two tyres off.
    - So 2.5 m is the slow margin plus the most the AI was measured outside the line at speed: 1.2 + 1.3 m. In the slow bends the 1.2 m held, so it stays there: Lyell Highway's 15.8 m and 13.9 m hairpins at 45 and 40 km/h with no tyre off, and Brinktop Road's 6.1 m hairpin at 29-42 km/h with at most two tyres off for a moment.
    - The speed is the one the cars may reach, not the planned speed. The planned speed stops at 130 km/h, but AC's AI doesn't: in game on 5 Nov it ran up to 220 km/h. On Brinktop Road's kink at 9.95-10.0 km, taken at 190-200 km/h, the cars ran 2.5-3.4 m outside a line 2.5 m from the edge, and 5 of 8 put wheels off. So the margins read the same speed profile capped at 200 km/h instead of 130 (`_Traced.margin_speeds`): its acceleration pass still keeps the margin narrow out of a slow bend.
    - 3.3 m at 200 km/h is the maintainer's choice. Room for all the AI's drift there (1.2 + 3.4 m) is more than a two-lane road has; 3.3 m holds the line on the centreline of a 6.4 m road, where the cars drove anyway. On Brinktop Road's kink the line now keeps 3.05 m from the edge, 0.15 m off the centreline, where it kept 2.50 m, 0.70 m off; its 5.6 km bend is on the centreline. Grasstree Hill's 6 km bend, at 90-130 km/h, is unchanged.
    - On a road too narrow for the margin (under 5 m wide at 120 km/h, 6.6 m at 200 km/h), the bounds stop at the centreline, and the line keeps to the middle.
    - The speed depends on the line, so the margins are first set from the centreline's speeds, the line solved, its own speeds worked out, the margins set from those and the line solved again from where it was. At most two re-solves; the last only widens margins. On Grasstree Hill, Brinktop Road and Lower Swamp Road that takes both, and the line written then comes within 2 cm of the margin its own speed asks for wherever that margin holds it.
    - On a 6.4 m road at 120 km/h and over that leaves the line 0.7 m either side of the centreline, so it can't swing far at speed: on Grasstree Hill forward, the most it moves across the road in 50 m above 110 km/h went from 3.98 m to 1.47 m. So there's no separate limit on how fast it swings across; that's the next step if cars still run wide in game.
  - Its curvature is limited to the centreline's tightest within 20 m, so it never takes a bend tighter than the road does. Without that limit it hugged the inside of long bends, 48 m on a 50 m bend.
  - **Room for the AI's slide out of a bend where the line would apex early** (`ai_line.exit_stretches`, `exit_clearance` in the track summary). In game on 10 Nov, AC's AI came into three of Lyell Highway reverse's bends, at 18.5, 24.4 and 27.25 km, 13-17% over the line's corner speed and slid up to 2.9 m outside the line as it shed that speed. That's harmless where the line leaves room: at 18.5 km it runs on the inside, 4.05 m from the outside edge, and all 26 passes stayed on. At 27.25 km the line apexed early: it crossed to the outside half 10 m into the bend's tight part (68-73 m for 90 m, down 4-5%) and kept 2.0-2.5 m from the edge, and 18 of the 22 cars that got there left the road at its outside.
    - **Where:** a bend's tight part is where its radius is within 10% of its least, and its closing part where the radius has come within 1.5 times its least before that. On a descent over the tight part, with the AI's corner speed at the least radius over 60 km/h (slower, the AI keeps close to its line, as the slow margin assumes), the line apexes early in either of two ways. It starts the tight part on the inside half, and the tight part runs on for 50 m or more from where the line first reaches the outside half (27.25 km). Or it comes off the inside half in the closing part, before the tight part starts, and holds the outside half unbroken for 50 m or more (61.9 km: inside at 61.89-61.90 km, out from 61.915 to 61.966 km, the tight part from 61.93 km). The stretch runs from the tight part, or from where the line reached the outside half if that's earlier, to 30 m past it, or to where the bend opens past 1.5 times its least radius.
    - **The room:** 3.85 m from the line to the outside edge there, the most the cars slid on the road (3.0 m, 2.9 m at both 18.5 and 27.25 km) plus half the car (0.85 m), kept from 20 m before the stretch to 20 m after it. The line is solved once more with that room (`racing_line.bounds`' `clearance`, past the centreline if need be), from where it was: a later apex. The margins for speed are unchanged, so the line still keeps them from both edges.
    - **Not where the AI sheds its overspeed, or the line apexes late:** at 24.4 km the line keeps as little (2.1-2.6 m) on the way in, but the road turns uphill there and the cars slid only 0.9 m; its tight part (64-70 m) is uphill, so it isn't counted. Nor is a late apex (14.0, 14.9 and 18.5 km: the line inside through the tight part), or a brief one (reverse 12.9 km, where the line reaches the outside half 4 m before the bend opens; driven cleanly over 30 times).
    - **On a road too narrow for the room** and the inside margin both, the inside margin holds, and the bend gets a hint of 0.86 (1 over 1.15, rounded down) over the stretch, so the AI comes in at about its corner speed with no overspeed to shed (`ai_line.exit_hints`). None of the example roads needs one.
    - **On the example roads:** only Lyell Highway. In reverse the line moves in at seven bends of 68-147 m that it apexed early on a descent: 15.39-15.48 km (up to 0.88 m), 27.16-27.28 km (1.96 m), 29.84-29.93 km (1.28 m), 32.47-32.59 km (1.12 m), 51.31-51.39 km (0.97 m), 57.95-58.06 km (0.57 m) and 61.92-62.04 km (1.51 m). Through 27.17-27.28 km the line is now 0.35-0.82 m on the inside, 3.85-4.3 m from the outside edge, where it was up to 1.48 m on the outside, 2.0 m from it; through 61.92-61.96 km it's 0.37-0.87 m inside, 3.87-4.37 m from the edge, where it was up to 1.13 m outside, 2.37 m from it. Forward it moves in at one, 34.50-34.58 km (48 m, 2.07 m; 3.85 m from the edge, where it came within 1.82 m). Elsewhere the re-solve moves it by a few centimetres (0.17 m at most, at hairpins). The hints keep their places and paces but one: the reverse line is 1.3 m shorter, which re-spaces its 5 m points by up to 0.3 m, and 14.0 km's lowest pace falls from 0.651 to 0.642, so its hint is 0.64 where it was 0.65 (its ramps 0.76 and 0.88). The planned runs are 1.1 s (reverse) and 0.35 s (forward) longer. Every other layout's `fast_lane.ai` and `ai_hints.ini` is byte-identical. **To check in game** (`docs/known-issues.md`).
  - It keeps to the centreline for the first 60 m of each layout (the grid spots, the start gate and the hot-lap spawn) and from 30 m before the finish gate to its end (the run-off, where the AI is stopped). It leaves and rejoins the centreline over 50 m.
  - Each layout gets its own line: the reverse isn't the forward one backwards.
  - Built on Grasstree Hill (6.4 m wide), the tightest radius outside the held stretches went from 26.2 m to 30.3 m and the planned run from 386 s to 375 s. On Brinktop Road, 7.1 m to 16.1 m, its slowest planned speed from 31 to 46 km/h. On Lyell Highway (63 km), 13.9 m to 17.3 m.
  - With the margins for speed and the braking below, Grasstree Hill's planned run is 381.5 s forward and 382.7 s reverse (381.7 s and 382.9 s with the margins for speeds up to 200 km/h; Brinktop Road 291.0 s), against the centreline line's 389.0 s and 390.7 s with the same speed rules (2% quicker; it was 3%). Brinktop Road: 290.9 s against 292.2 s. Lower Swamp Road (4.0 m wide), where the line now keeps to the middle above about 97 km/h: 223.0 s against 224.7 s forward.
- **Placement:** points about 5 m apart along the line itself (`AI_POINT_SPACING_M`), evenly from its start to its end, each 0.4 m above the road surface where it stands, crown and banking included. Points on the surface have crashed the game in another generator.
  - Why 5 m: AC measures the spline from the points in float32, and the rounding inflates the length it reports (`TRACK LENGTH` in `log.txt`) by about 1/spacing². On Lyell Highway (63.06 km, reaching ±21.7 km east-west from the local origin), measured in game on 4 Nov by swapping only `fast_lane.ai`: 1 m points made it 83,747 m (+32.8%), 2 m +8.8%, 5 m +1.3% (positions within 203 m everywhere, 26 m at the end), 10 m +0.3%. Grasstree Hill's 1 m line read 1.1% long.
  - The inflation grows with distance from the origin, so every spline fraction (the run-off hints, AC's progress) moved with it on a long track.
  - 10 m points cut the chord 0.8 m inside a 15 m hairpin. At 5 m the AI drove Lyell's 15.8 m and 13.9 m hairpins at 40 and 36 km/h, within 1.0 m of the line: as good as the 1 m line or better.
  - Speeds, gas, brake, radius and grade are worked out every metre along the line, then read at each point.
- **Speed:** grip-limited on the line's own curvature, `v = sqrt(μ g R)` with μ = 1.05, capped at 130 km/h. It then gets a forward pass for acceleration and a backward pass for braking, both adjusted for grade. AC's AI finished the forward layout at M3, but it ignores this speed and set its own, up to 214 km/h in an Abarth 500.
  - **Braking shares the grip with cornering**: turning at lateral acceleration a, the backward pass brakes at only `sqrt(1 − (a / μg)²)` of the straight-line rate (8 m/s², adjusted for grade), the friction ellipse. Before, a bend that tightens was planned at the grip limit for the radius the car was on, braking for the tighter one ahead with no grip left: on Grasstree Hill's 5.95 km right-hander, 127 km/h on the 130 m radius, then 95.5 km/h at the 68 m apex, turning at 99% of the grip at 5,936 m while braking at the full rate (1.98 on the ellipse, where 1 is the limit). Now the line slows before the bend tightens: 117 km/h at 5,926 m, and nowhere on the forward line more than 1.06.
  - Acceleration isn't limited by cornering the same way; it's the traction limit alone, as before.
  - **AC's AI doesn't drive these speeds** (in game, 5 Nov). In Grasstree Hill's 5.95 km right-hander the cars arrived at 122-123 km/h on both lines, where the line planned 127 km/h before the braking change and 116.8 after it, and they ran up to 220 km/h where the speeds stop at 130. The AI works out its own speeds; its apex speed matching the plan (90.7 against 90.3 km/h) is most likely because both come from the same curvature. Holding the cars to a planned speed would take `ai_hints.ini`. The braking change stays: it costs nothing, and the margins are set from these speeds.

## Validation

`mapcreator validate` reads every file back and checks the rules above, including measured heights of spawns and AI points against the actual road triangles. The road is every `1ROAD` and `1GRAVEL` mesh: spawns, the AI line, and the tree and power colliders' clearance are all measured against both. The in-game checks (does it load, are the wheels on the ground, can the AI drive it) live in `harness/`. With side-road junctions, it also lists them, checks nothing stands on them, and checks their stubs are drawn and joined to the road with no gap and no ground over them ("Side-road junctions" below). The stubs are road meshes too, so they count as road for every check above.

For the AI line it also checks:
- distances strictly increase, and each spawn faces along the line;
- every point is at least 1.0 m (the 1.2 m margin less 0.2 m) inside both edges: by its side distances, and on the road's triangles either side of it;
- every point keeps the margin for its planned speed (1.2 m at 60 km/h to 2.5 m at 120 km/h, or half the road where it's narrower than twice that), less 0.2 m: a warning, never a failure, as for the line's radius. The line is built for the faster speeds the cars may reach (up to 3.3 m at 200 km/h), but the AI file's speeds stop at 130 km/h, so the check can't see that part;
- the line is nowhere more than 5% tighter than the road within 30 m of it: a warning, never a failure (the maintainer's decision). The road's middle is found from each point and its two side distances, and both radii come from a spline through the points.
- no early apex leaves less than 3.85 m from the outside edge (the exit clearance above): a warning, never a failure, found from the AI file's radius, turn, grade and side distances as the track stage finds it, saying what hint `data/ai_hints.ini` holds there. A line the track stage wrote shows none unless the road was too narrow for the room.
- a bend the AI would take faster than the road allows (the bend hints above): a warning, never a failure, saying where it is, why (crest, grade, banking) and what hint `data/ai_hints.ini` holds there. The bends are found from the AI file as the track stage found them, with the banking read from the road's triangles straight across the road.
- The route check (a bend under 5 m) judges the road, not the racing line: the centreline's radius every metre over the AI line's stretch, from the profile (`ai_line.road_radius`), exactly as main's centreline line carried it. The AI file can't give it: its radius is the racing line's, and the road's middle found from points 5 m apart reads hairpins tight (Brinktop Road's 7.2 m corner as 6.9 m). Only a folder validated on its own, without the build's profile, falls back to that.

## Power lines (`core/power.py`)

- **Meshes:** `POWER_TOWERS_n` (lattice towers, material `power_metal`), `POWER_POLES_n` (timber poles, `power_timber`) and `POWER_WIRES_n` (`power_cable`). They're visual only; they're split to stay under the 65,535-vertex limit.
- **Supports:** a lattice tower or a timber pole, whichever the data maps at that support (OSM's `power=tower` or `power=pole` on the node).
  - Where the data doesn't say, the support takes the majority design of the line's mapped supports.
  - With nothing mapped, it's a pole on distribution lines, under 66 kV, or where the line's vertices are under 200 m apart; otherwise a tower.
  - A lattice tower is 26-50 m tall by voltage, with three levels of cross-arms carrying one circuit on each side and an earth wire at the peak.
  - A pole is 11 m of timber (14 m on lines tagged `power=line`) with a cross-arm and three insulators, carrying one circuit.
  - Each support is turned to the bisector of the spans on either side of it. Wires attach to whatever stands at each end.
  - Every leg and pole goes 2.5 m below the ground at its own foot (`LEG_SINK_M`), so a downhill leg on a slope, or a support on coarse terrain cells, never shows a gap.
- **Wires:**
  - A parabolic sag of span² / (8 C), with C 1,600 m for transmission and 700 m for distribution.
  - Where the DEM rises under a span, the sag is cut back so the wire stays 8 m (transmission) or 5.5 m (distribution) above the ground. Spans that still can't clear it are counted as `low_spans`.
- **Collision:** supports within 15 m of the road's edge get an invisible `1WALL_POWER_n` mesh: the pole, or the tower's four legs, up to 3-4.5 m.
  - No collider where it would come within 0.3 m of the road, or within 2.5 m of it in the 60 m at either end of the road, where the grid spots stand side by side.
  - The validator checks them as it does the trunk colliders: invisible, clear of the road, and at least 3 m from every spawn.
- **Reach:** supports stand only on the terrain mesh, with their whole base within 5 m of its edge: 695 m of the centreline, since the terrain reaches 700 m (`core.reach.POWER_M`; it was 300 m). That's past the trees and buildings (600 m): towers are tall, few and cheap to draw, so they reach as far as the ground does (the maintainer's decision, 4 Nov).
  - Towers beyond the terrain stood in the sky, their bases floating past the ground's edge (in game, 4 Nov), so a span with a support beyond the reach isn't built.
  - With the 300 m terrain, Grasstree Hill's two TasNetworks lines were left out. Widening the terrain to 700 m brought back 8 towers, 140-645 m out.
- **No wind:** CSP's `[WIND] DYNAMIC_CABLES` (undocumented; CSP's own Laguna Seca config uses it) does act on the wires: its log shows "Preparing cables". But it redraws them as tubes about 8 times thicker than the 3.5 cm mesh. Without it they draw as thin lines and look right (in game, 4 Nov), so `ext_config.ini` leaves it out.

## Roadside furniture (`core/furniture.py`)

Built in the mesh stage, visual only, in `FURNITURE_n` meshes and a texture atlas the stage draws (`furniture.png`).

- **Three materials on the one atlas**, with the same values by day:
  - `furniture_reflector`: the guide post reflectors, a small quad on each face of the post, 2 mm proud of it;
  - `furniture_retro`: both faces of a CAM board and a speed sign's face, which are retroreflective sheeting on the real thing;
  - `furniture`: everything else (posts, poles, board edges, the backs of speed signs).
- **At night** CSP brightens the two reflective materials through `ext_config.ini`, so they throw back the headlights the way real reflectors do:
  - a `[MATERIAL_ADJUSTMENT_n]` per material with `CONDITION = NIGHT_SMOOTH` raises its `ksDiffuse` (`furniture.NIGHT_DIFFUSE`: 12 for the reflectors, 2 for the boards' faces, against 0.6 by day), and `VALUE_0_OFF` puts the KN5's value back by day;
  - the file includes CSP's `common/conditions.ini`, which defines `NIGHT_SMOOTH` (sun above 88 degrees);
  - this is how CSP's own Imola config makes "curbs + start markers reflective at night" (`extension/config/tracks/loaded/imola.ini`, `MATERIAL_ADJUSTMENT_15`);
  - it raises diffuse, not `ksEmissive`, so the faces light up only where the headlights fall and don't glow by themselves from afar. The ambient term (the night sky) stays as it is;
  - the boards' faces get less: in the beam a CAM already reaches the top of CSP's tone curve unboosted, so more only bleaches it (yellow to nearly white from 2.5), and from about 6 a speed sign's legend washes out at 35 m; at 12 one 90 m away reads as a light (in game, 5 Nov);
  - vanilla AC ignores the file, so there the faces look as by day.


- **Guide posts** stand on both sides, 0.8 m beyond the road's edge and about 1 m tall, with a black band and a reflector on each face.
  - Red faces the traffic for which the post is on the left; white faces the traffic for which it's on the right.
  - Spacing follows Main Roads WA's "Design of Guide Posts" (citing AS 1742.2-2009 clause 4.2.4.4): 150 m on straights. On curves it's closer, and closer still on the outside: 6/12 m (outside/inside) below 100 m radius, 10/20 m to 200 m, 15/30 m to 300 m, 20/40 m to 400 m, then 40, 50, 60 and 90 m both sides out to 2,000 m.
  - The spacing is checked at every station, so posts close up as soon as a curve begins.
  - None where a guard rail runs, and none on unsealed road.
  - None in a side road's mouth: a post due there goes up just past it ("Side-road junctions" below). A CAM there is left out, and a speed sign moves to the nearest place clear of it.
- **Curve alignment markers** (CAMs: black chevron on yellow, 0.45 x 0.6 m, 1.2 m up, 2 m beyond the edge) go on the outside of a bend whose advisory speed is at least 15 km/h below the speed limit and no more than 60 km/h.
  - Advisory speed is v = sqrt(127 R (e + 0.35)) km/h. The limit comes from OSM; where it's unknown, 80 km/h. (At 100 km/h and with no 60 km/h ceiling, the Esperance Coast Road got 723 CAMs over 20 km.)
  - Three to six per bend, spaced at twice the outside guide post spacing (12-40 m), starting one spacing before the bend and ending one after.
  - Boards stand back to back, so each direction sees its own chevron.
  - Each bend of an S-bend is treated on its own.
  - None on unsealed road. A bend partly on gravel counts only by its sealed part: that must qualify on its own (tight, and at least 15 m long), and any of its markers whose place falls on the gravel, such as the one after the bend, isn't put up.
- **On gravel** there are no guide posts and no CAMs (the maintainer's decision), and speed signs go up as mapped.
  - The spacing is an approximation: the AS 1742.2 table isn't openly published.
- **Speed limit signs** (white, red ring, black numerals; 0.45 x 0.6 m, 1.5 m up, 2 m beyond the edge) go on the left for each direction of travel.
  - One where OSM's `maxspeed` changes between two known values.
  - One near each layout's start (70 m in), where the limit there is known.
  - Unknown or non-km/h limits make no sign. The numerals use Pillow's built-in Aileron font (CC0).

## Side-road junctions (`core/junctions.py`)

The route stage finds the real side roads that meet the route and writes them to `01_route/junctions.json` (below). The section stage plans a stub of each, its first 100 m (`core/stubs.py`, `03_section/stubs.json`, below), and the mesh stage draws it as road, joined to the mainline by a corner flare either side ("Side roads' stubs" below). Guard rails, roadside furniture, trees, buildings and power-line supports are kept out of the side roads' mouths and off their stubs, and the timing gates move clear of them. Each stub is closed off the stage, with police cars at some of the major closures ("Closures on the stubs" below). A job with `"junctions": "none"` builds without any of it, exactly as before (`Job.junctions`, `closed` by default, and left out of the track id while it holds its default).

- **Which side roads count** (the maintainer's decision, 10 Nov 2026). At each node of the route, a way leaving it is a junction when:
  - its way is a class the router uses (`overpass.DRIVABLE`: motorway to living_street, `road` and the links). The road data holds no other kind, so driveways, other service roads and tracks never count;
  - it isn't private: the most specific of `motor_vehicle`, `vehicle` and `access` that's tagged isn't `private` or `no`;
  - it leaves at least 15° off the mainline, against both the road ahead and the road behind, measured to its point 15 m out;
  - it's clear of the rest of the route. It's followed until it comes within 10 m of another part of the route (one where the road distance is more than 1.4 times the straight distance, plus 5 m), and one that does so within 15 m isn't a junction;
  - it has no `bridge`, `tunnel` or `layer` other than 0 in its first 30 m: a road passing over or under isn't a junction.
- **How far:** each is followed out to 100 m, the stubs' length to come. It follows its own way, then, at the way's end, the straightest way that turns less than 60°. It stops sooner at a dead end, at a branch, or where it meets the route. A side road at the route's very first or last point isn't one: the end wall closes it.
- **Closure level** ("Closures on the stubs" below): `limited` where the most specific access tag is `permit`, `forestry`, `agricultural`, `destination`, `delivery`, `customers` or `military`; else `major` (motorway, trunk, primary, secondary, tertiary and their links) or `minor` (unclassified, residential, living_street, road).
- **On the examples:** Grasstree Hill 8, Brinktop Road 18, Esperance Coast Road 10, Lower Swamp Road 3, Lyell Highway 4 (Burns Dam Road, `access=permit`, is limited) and Mount Dandenong Tourist Road 73.
- **Roundabouts** the route drives or touches (`junction=roundabout` or `circular`) are recorded with their ring and exits, and built as before: the route drives only its arc of the ring, its exits aren't junctions, and nothing is cleared at them. The validator names them. None of the examples passes one.
- **The mouth** of a junction is the stretch of the mainline where the side road's carriageway crosses the band from the road's edge to 3 m beyond it (`MOUTH_BAND_M`: guide posts stand at 0.8 m, rails at 1 m, signs at 2 m), plus 5 m along the road either side (`MOUTH_MARGIN_M`). The carriageway is its line as `junctions.json` has it, at its tagged or class-default width as mapped (`section.width_from_tags`). In it:
  - no guard rail or rail collider. The rail mask is cut after `rail_mask` has closed its gaps and extended its runs, so a mouth stays open, and a piece left shorter than 24 m beside it goes too;
  - no guide post (one due there goes up at the first station past it), no CAM (left out) and no speed sign (it moves to the nearest station clear of the mouth);
  - no timing gate (below).
  - Where the side road has a stub, the mouth reaches over its flares too (from one flare's tangent point to the other's, plus the margin), and the side road is the stub's paved outline (throat and carriageway) as well as its mapped line, whole: a stub keeps to its own side of the road by construction.
- **Along the side road**, for its first 100 m and on its own side of the mainline, the mainline's own clearances apply from the side road's edge: no tree within 4 m (the 2 m verge and 2 m more), and so no trunk collider; no building within 2 m; no power-line support whose base comes within 0.5 m (it's dropped with its spans, as one on the road is).
  - They're taken out after everything else is chosen, and every other tree and building keeps its random draws, so the rest of the scenery is as without junctions. Trees also stay out of the 3 m round a building taken out, as round every building.
- **What doesn't change:** the mainline's road surface and its half widths, so the AI line and the racing line, `ai_hints.ini`, the grid, the spawns and the run-off. On the six examples every `fast_lane.ai` and `ai_hints.ini` is byte for byte as without junctions. The AI drives only its line, so it never meets a stub. A car that runs wide can drive into one: the closures stop it only where they're solid, at straw bales or a police car ("Closures on the stubs" below).
- **Timing gates:** a gate whose place falls in a mouth moves to the nearest place clear of every mouth, 0.5 m past its end, by at most 40 m (`layout.gate_distances`). A start gate moved back stays at least 4 m ahead of the grid's front spot (so at 46 m or more), and the finish gate leaves at least 50 m of run-off. With nowhere clear, it stays, and the validator warns; a stub whose mouth would do that is left out instead (a gate end on a stub would fail "gate spans the road"). The grid, the hot-lap spawn, the AI line and its hints keep to the gate's usual place, so the timed length changes by those few metres. On the examples only Brinktop Road's start gate moves, 0.5 m back to 49.5 m, clear of Percy Street's flare.
- **Validation:** a warning listing the junctions (name, class, closure level and where), and checks that no guard rail or rail collider, guide post or sign has a vertex on a side road's carriageway or flares within 3 m of the road's edge, and no tree or power collider has one on a side road's carriageway or stub in its first 100 m. A timing gate's end in a mouth is a warning. For the stubs (`validate.stub_checks`), failures unless:
  - each is road (a `1ROAD` or `1GRAVEL` mesh) along its centreline;
  - its road meets the mainline's and its own first full row with no gap and no step over 5 cm, sampled 5 cm either side of the mainline's edge across the mouth (away from the flares' tangent points, where a flare runs within centimetres of the edge) and of its first row;
  - there's no hole in the ground within 3 m of it: road, verge, skirt or terrain under every point, sampled every metre;
  - no ground (verge, skirt or terrain) stands more than 2 cm above its road.

### Side roads' stubs (`core/stubs.py`)

Each junction's side road is drawn as road for its first 100 m (the maintainer's decision, 10 Nov 2026, "so it doesn't look weird when driving past"), or to where it ends or meets the route. Service roads, tracks and driveways get none (they aren't junctions).

- **Line:** the side road's line from `junctions.json`, its start moved onto the smoothed centreline (eased over 20 m) and then filleted and smoothed as the route is (`centreline.smooth`). Its rows are a metre apart from where it crosses the mainline's edge (d = 0), so `d` is metres in from the mainline's edge line.
- **Width, surface and look,** by the mainline's rules (`core/section.py`): the tagged or class-default width as mapped, times the job's width option; sealed or unsealed by its own `surface`, or, untagged, as the mainline is where it meets it; one lane or two (`section.single_lane`). Crown 2.5% sealed and 4.5% unsealed, and no camber.
- **Heights:** beside the mainline the surface carries on the mainline's own cross-fall; further out it's the side road's own profile, the DEM along it smoothed over 15 m, with its crown. The two blend (smoothly) over 15 m from the mainline's edge, or longer where the side road sits well above or below the road, so the blend climbs or falls no more than 12%.
- **Flares:** a circular arc either side from the mainline's edge to the stub's, tangent to both, so a car can turn in on it. It's drawn in the mainline's own frame (station along it, offset from it), where its edge is a straight line, so the arc never dips into the road however the road bends.
  - The radius is 8 m, the route's own corner radius ("about the path a car takes through a town junction"). So the flare on the acute side of a skewed junction runs longer: its tangent is 8 m / tan(angle / 2), 3.3 m in a 135° corner and 19 m in a 45° one.
  - The tangent is capped at 25 m (so in a 30° corner the radius drops to 6.7 m) and the radius kept at 3.5 m or more. A corner needing a tangent over 40 m gets no flare, and the side road no stub. Tangent points fall on the road's and the stub's rows.
  - A flare that comes out wrong is drawn again at half the radius, then a quarter: one that folds over itself, far out on the inside of a tight bend where the road's frame does, or one that leaves the stub's own carriageway outside the throat, where the side road bends hard just past the junction.
- **The throat:** the road between the mainline's edge, the two flares and the stub's first full row. It's triangulated over the mainline's edge vertices across the mouth, the flares' points and the first row's vertices, sharing them exactly, with the stub's own cross-section points inside it, so its surface is the stub's. It's plain surface: its texture coordinates stay inside a band of the stub's texture with no markings (sealed) or between the wheel paths (gravel), mirrored back and forth across it.
- **The stub's ribbon** (`mesh.build_ribbon`): road, 2 m verge and skirt, like the mainline's, from its first full row to its end, a row every metre for its first 20 m (where the closures stand) and every 2 m beyond. The road stops square at its end, and the side road carries on as terrain.
- **The flares' verges:** 2 m of mown grass along each flare, from the mainline's verge vertex at the tangent point to the stub's verge vertex at its first row.
- **The mainline at the mouth:** its verge and skirt are cut across the mouth's rows. Its skirt is held short of each stub's, and each stub's of the mainline's and of other stubs': each stops where the ground is nearer the other road's centreline than its own (`stubs.side_caps`), but always reaches 0.5 m past its verge. The verge keeps its width.
- **Terrain:** the triangulation's boundary is every outline vertex of the mainline's ribbon (less those cut away), of each stub's ribbon (less its first row's road, which meets the throat) and of each flare's verge's outer edge, each once. Only points on the outside of the pieces together go in, so no triangle reaches across a piece and leaves a hole where it's dropped. The terrain keeps out of all of them; with ground classes, the stubs' skirts join it as the mainline's does.
- **Meshes:** `1ROAD_J###` (sealed) or `1GRAVEL_J###` (unsealed), with an `N` after the number for one lane, and `1GRASS_J###` for their verges: packed by material per 400 m of the mainline (`###`), under the vertex limit. Their materials are the mainline's, so Grass FX occlusion, Rain FX and the gravel's `[TYRES_FX]` follow, and a track with an unsealed stub has `GRAVEL` in `surfaces.ini`: a sealed road with a gravel side road gains the gravel material and texture.
- **Left out,** and listed in `stubs.json`'s `skipped` with the reason: one whose flares can't be drawn (`flare`, `corner`, `skew`, `stays_on_road`), too near the road's end for its flares (`road_end`), with less than 10 m of ribbon past its throat (`short`), coming within a verge of another part of the route right from its first row (`near_route`), whose mouth would share the road's rows with a stub already kept on the same side, or its throat overlap one, even with flares half the size (`overlaps`: major roads, then longer ones, are kept first), or whose mouth would leave a timing gate nowhere clear to stand (`timing_gate`). Further out, a stub that would come within a verge of another part of the route, or overlap a kept stub, is cut short before it (`ends: clipped`).
- **On the examples:** Grasstree Hill 8 of 8, Brinktop Road 14 of 18 (Henry Street and the Tasman Highway at the route's ends, Bridge Street's one-way arm into its other arm, and Charles Street leaving Gunning Street's node on the same side), Esperance Coast Road 10 of 10, Lower Swamp Road 3 of 3, Lyell Highway 3 of 4 (a 33 m dead-end spur), Mount Dandenong Tourist Road 66 of 73 (6 in its clusters, and School Lane, beside another part of the route).
- **For the closures:** `stubs.pose(stub)` gives (easting, northing, height, heading into the stub) on its centreline at d metres in from the mainline's edge line, carrying on straight past its end; `stubs.surface_height(stub, d, lateral)` and `stubs.ground_height(stub, d, lateral, dem)` give its road's surface and the ground beside it.

### `03_section/stubs.json`

Written by the section stage when the route has junctions, removed when it has none or the job says `none`. Coordinates are in the region's metric CRS (`crs`), to the centimetre; heights in metres on the DEM's datum, to the millimetre; headings in radians, east 0, counter-clockwise. A new field may be added at any time; a change to an existing field's meaning raises `version`.

- `version`: 1. `crs`: e.g. `EPSG:28355`.
- `rule`: `flare_radius_m` (8), `flare_max_m` (25), `flare_min_radius_m` (3.5), `blend_m` (15), `max_blend_grade` (0.12) and `width_factor` (the job's width option).
- `stubs`, in order along the road, each:
  - `junction` (its `id` in `junctions.json`), `name`, `highway`, `closure` and `side`;
  - `station_m`: the junction on the smoothed centreline; `angle_deg`: the stub's heading where it leaves the mainline's edge, against the road's at the junction, positive to the left;
  - `surface` (`sealed` or `unsealed`), `style` (its material: `road`, `road_plain`, `gravel` or `gravel_narrow`), `width_m` (with the job's width option), `width_mapped_m`, `width_source` and `crown`;
  - `length_m`: d at its last row; `ends`: as in `junctions.json`, or `clipped`;
  - `ribbon_from_m`: the d where its own ribbon starts (nearer the road is the throat); `blend_m`: over how far its heights blend from the mainline's cross-fall to its own profile;
  - `mouth`: `from_m` and `to_m`, the stations of the flares' tangent points on the mainline, and `rows`, the centreline's rows there;
  - `flares`: `upstream` (nearer the road's start) and `downstream`, each with `stub_side` (`left` or `right`, looking out along the stub), `radius_m`, `tangent_m`, `angle_deg` (the corner's), `mainline_row`, `stub_row`, and `points`: [easting, northing, height] from the mainline's edge to the stub's edge at its first full row, a metre apart or less;
  - `rows`: its centreline every metre from d = 0: `d`, `xy`, `z` (its road's surface on the centreline), `z_left` and `z_right` (at its left and right edges, looking out along it; the surface runs straight from the centreline to each edge), `heading`, `curvature`, `grade` (dz/dd) and `half_m`;
  - `outline`: its paved area, the throat and carriageway, as one ring.
- `skipped`: `junction`, `name` and `reason` (above).

### `01_route/junctions.json`

Coordinates are in the region's metric CRS (`crs`), as `route.npz`'s are, rounded to the centimetre; distances in metres. A new field may be added at any time; a change to an existing field's meaning raises `version`.

- `version`: 1. `crs`: e.g. `EPSG:28355`.
- `rule`: the rule's numbers as built: `leg_m` (100), `min_angle_deg` (15), `angle_at_m` (15), `min_leg_m` (15), `grade_free_m` (30), and the `classes`, `private` and `limited_access` values.
- `junctions`, in order along the route, each:
  - `id`: its index; `node`: the OSM node where it leaves the route;
  - `at_m`: where, along `route.npz`'s polyline from its start. The later stages find the junction on the smoothed centreline by projecting `xy` onto it, with `at_m` only breaking a tie where the road passes the node twice (`junctions.mouths`): the smoothed centreline's stations drift from the routed line's (67 m by 53 km on Lyell Highway);
  - `xy`: the node; `lonlat`: the same in longitude and latitude, to find it on a map;
  - `side`: `left` or `right` of the route's direction (the forward layout);
  - `angle_deg`: its direction to its point 15 m out, against the route's direction there (the bisector of the road ahead and behind), positive to the left; `off_mainline_deg`: the smaller angle to the road ahead or the road behind, which the 15° rule reads; `route_turn_deg`: how much the route itself turns there;
  - `way_id`: its first way; `way_ids`: every way it was followed along;
  - `highway`, `name`, `closure` (`major`, `minor` or `limited`) and `access` (the most specific access tag, or null);
  - `tags`: those of `highway`, `name`, `ref`, `surface`, `lanes`, `width`, `narrow`, `lane_markings`, `oneway`, `maxspeed`, `access`, `motor_vehicle`, `vehicle` and `junction` its first way has;
  - `surface_class`: `sealed`, `unsealed`, or null where its `surface` gives neither;
  - `width_m` and `width_source`: its carriageway from `section.width_from_tags`, as mapped (before the job's width option); `unsealed` widths where `surface_class` is unsealed;
  - `length_m`: how far it was followed; `ends`: why it stopped: `length` (100 m), `dead_end`, `branches` (no way on within 60°), `joins` (only ways already followed go on) or `meets_route`;
  - `line`: its centreline from the junction node outwards, `length_m` long.
- `roundabouts`, each: `id`, `way_ids` (the ring's ways), `highway`, `name`, `tags`, `ring` (its nodes in order round it, the first repeated at the end when it closes), `closed`, `centre`, `radius_m` (the nodes' mean distance from their centre), `on_route`, `route_from_m` and `route_to_m` (where the route is on it, as `at_m`), and `exits`: each drivable, public way leaving the ring that isn't the route, with `node`, `xy`, `on_route` (whether it leaves the ring where the route is on it) and the junctions' fields from `way_id` to `line`.
- `skipped`: the side roads that leave the route but aren't junctions, for checking the rule: `node`, `at_m`, `xy`, `side`, `angle_deg`, `off_mainline_deg`, `way_id`, `highway`, `name`, and `reason`: `class`, `private`, `grade_separated`, `roundabout` (an entry or exit of a roundabout the route drives), `angle` or `meets_route`. On the examples only Brinktop Road has any: Bridge Street behind its start and the Tasman Highway beyond its finish, each the road carrying straight on past the route's end (`angle`).

## Closure props (`core/closures.py`)

The objects that close a side road off the stage, after Motorsport Australia's stage-rally rules (2026 Special Stage Rally standing regulations, art. 11.10 "Road Closure Standards"). The mesh stage places them on every side road's stub ("Closures on the stubs" below).

- **Levels**, by the side road's class:
  - **major** (trunk to tertiary): course tape at the mouth; barrier boards on yellow trestles across the full width, with a ROAD CLOSED sign in the middle and a slot behind them for a vehicle (its own module builds it); 25 m further in, a second tape and another ROAD CLOSED sign.
  - **minor** (unclassified, residential, living street, `road`): the tape; one obstacle across the road, barrier boards or big square straw bales; the second tape and sign.
  - **limited** (permit, forestry, destination): the tape and a sign.
  - No arrows, chevrons or DETOUR signs (the maintainer's decision).
- **Layout** (`closures.layout`), in metres from the mainline's edge line:

  | | Tape | Obstacle | Second tape | ROAD CLOSED sign |
  |---|---|---|---|---|
  | major, minor | 4 | 12 | 37 | 37.6, behind the second tape; a major closure's boards carry one too |
  | limited | 4 | | | 12 |
  | overshoot stub (straight ahead where the route turns) | 4 | 27 | 52 | 52.6 (limited: 27) |

  - MA asks for "a minimum of 25m" between the obstacle and the tape and sign.
  - A stub under 45 m (60 m on an overshoot stub) has no second tape: the sign goes on the obstacle, in the middle of the boards or raised behind the bales. Under 15 m (30 m) there's no obstacle, only the tape and a lone sign as far in as fits.
  - No obstacle or sign stands in an overshoot stub's first 25 m. A lone sign never stands nearer than 6 m (26 m overshooting); where it can't fit, there's only the tape.
- **The local frame:** origin on the stub's centreline at the mainline's edge line; +z into the stub, +x to the right looking in, +y up; metres.
  - `closures.closure(level, width, overshoot=, length=, obstacle=, ground=, seed=, solid=)` returns each element (`tape`, `obstacle`, `warning`: the second tape and its sign, `sign`: a lone sign) at its distance in, with its visual meshes (one per material), its props' extents, and a collider mesh where any of them is solid, plus the vehicle slot.
  - `width` is the width to close, centred on the centreline: normally the carriageway and its verges. It's a number, or a function of the distance in. `ground(x, z)` gives the cross-section (crown and verge fall) for the feet to stand on.
  - The frame is left-handed, so `closures.place(kit, pose, frame, index)` maps it onto AC's axes by a mirror and rewinds the triangles. `pose` is the stub's centreline at the mouth (easting, northing, height, heading into the stub), or a function of the distance in, so on a bending or climbing stub each element stands square across the stub at its own distance.
  - `closures.merge` packs many closures' meshes into a few under the vertex limit, one run per material.
- **Collision:** straw bales are solid; barrier boards, trestles and signs are passable, like the tape (the maintainer's decision, 10 Nov 2026). The vehicle, from its own module, is solid too.
  - Why: in game (10 Nov) the solid props felt like what the car met at about 20 km/h, but at 60 km/h they were a rigid wall. Boards on trestles, which would fold over, stopped the car like concrete, and two of four cars glanced along the face and spun off its end. Bales are believable as a hard stop.
  - So a major closure's boards don't stop a car, which drives on down the stub; a minor closure built with bales does.
  - No invisible wall in open air (the maintainer's concern): each bale has its own box collider, exactly its visible extent, from 1 m below the lowest ground under it to its top. The row spans exactly the width, 6 cm between bales, each collider flush with its own bale's face.
  - **One setting:** `closures.SOLID` (or `closure`'s `solid`) names the solid props by kind (`boards`, `bales`, `signs`) and element (`obstacle`, `warning`, `sign`); a prop is solid when both are named, naming none of either meaning all. Today it's `("bales",)`. Made solid, a board's or sign's collider runs from its trestles' front feet to their back feet.
  - Colliders are `1WALL_CLOSURE_*`: invisible, solid by the name, four sides and a top per box, wound outwards.
- **Sizes and colours** (the code marks the guesses):
  - Tape: 75 mm, red diagonal stripes on white (MA's Tape A), tied 1.0 m up on star pickets 0.3 m beyond the width and sagging up to 0.15 m, with an A4 warning tag.
  - Barrier boards: up to 2.5 m long, cut equal to fill the width; 0.2 m deep, centred 1.0 m up; yellow and black stripes at 45 degrees on both faces, 0.1 m wide, mirrored either side of the middle and falling towards it. AS 1742.3's rule for the direction is paywalled; the US MUTCD (2009, section 6F.68) asks for stripes falling towards the middle "where no turns are intended" (`STRIPES_FALL_TO_MIDDLE`).
  - Trestles: yellow (the NSW roads authority's G10 specification), open A-frames: on each side of the board two slim legs (tubes about 40 mm across, 0.48 m apart at the feet and 0.34 m under the board) and a crossbar about 0.3 m up, the feet 0.3 m either side of the board. Along a row the legs stand at most about 1.4 m apart. The first version's solid 0.45 m leg panels read as blocks in game.
  - ROAD CLOSED: AS 1742.3's T2-4, 1800 x 300 mm, black on white, double-sided so it reads from the stage.
  - Straw bales: big squares, 0.9 m high, 1.2 m deep, about 2.4 m long at most, twines along the top and down the ends. They stand 6 cm apart, each up to 5 cm out of line, 3 cm taller or lower and 8% longer or shorter. Each face shows one bale, with darker, ragged edges, in one of two tints (one a little weathered), either way round. In game the first row, with no joins, read as one 8 m block.
- **Two materials on one atlas** (`closures.atlas`: 1024 x 512, 0.35 MB as DXT1 with mips):
  - `closure`: trestles, stakes, bales, edges and the tag;
  - `closure_retro`: the boards' and signs' faces and the tape.
  - The tape and the stripes are bands the atlas's full width, so their texture coordinates run on past 1 and wrap. Each straw band holds two bales' faces side by side.
  - Every texture is drawn by our own code; the legend uses Pillow's built-in Aileron font (CC0), as the speed signs do.
- **At night** `closure_retro` gets ksDiffuse **0.9** and `closure` comes down to **0.35** (both 0.6 by day), each a `[MATERIAL_ADJUSTMENT_n]` as above; the track stage's `NIGHT_DIFFUSE` includes them. The plain material's 0.35 is the police car body's.
  - In game at 22:00 with the furniture boards' 2.0, the closures glinted like the reflectors from 25 m and more. But at 8-12 m, square to the beam, the sign burnt out (93% near white, the legend unreadable at 12 m), the stripes washed pale, and the straw bales glowed near white (83%) though they're not retroreflective.
  - At 1.3 on the faces, with the plain material unadjusted, the sign was still 79% near white with pale grey letters, the stripes faded mid-beam, the middle bales glowed (70%), and even a plain grey block came up from 106 by day to 141 at night. CSP's night exposure is set for the dark road (30 in the beam), so any light surface square to the headlights at 10 m clips. From 25 m and more the faces still glinted like the reflectors.
  - The legend, the sign's border and the black stripes are pure black: ksDiffuse multiplies the texture, so no boost lifts them. At 18 (7%) the boost and the near beam had lifted them into the top of CSP's tone curve.
  - The sign's white is a light grey, 200 (it was 244), so the white round the letters doesn't clip and bloom into them; the police car's white paint came down to 224 for the same reason.
  - **No specular** on either material (`closures.KN5_PROPERTIES`, which the track stage gives both: the walls' values with `ksSpecular` 0). On Brinktop Road at night (11 Nov), from the front at 12-15 m with the beam's hot spot on the boards, the sign was 91% near white with no dark letter pixels at all, and the stripes' yellow paled in the middle of the beam; from behind at 8 m it read crisp.
    - The cause: AC's `ksPerPixel` adds the specular term on top of the texture, not through it (Content Manager's re-implementation of Kunos's shaders has `txColor * (ambient + diffuse) + specular`), so even a pure-black texel gets `ksSpecular` x (N.H)^`ksSpecularEXP` x the light. The headlights sit beside the eye, so a face square to them has N.H close to 1 across its width (0.99-0.93 over the middle 4 m at 13 m, at the walls' exponent of 10).
    - The walls' 0.05 put a floor under the black of about 7-10% of the sign's white at its night ksDiffuse of 0.9. The furniture's boards are at 2.6%, since their 2.0 dwarfs it. Each night cut to ksDiffuse (2.0, then 1.3, then 0.9) and to the white made that floor a larger share. Where CSP's auto-exposure overexposes the face about tenfold, the floor alone reaches white. White light added to yellow also turns it pale.
    - Not the texture: decoded from the KN5's DXT1, the legend keeps exactly black stroke cores down to mip 3, which the game samples for the sign out to about 25 m on a 1080-row screen. 18-24% of its texels are under a fifth of the white there; they grey only from mip 4, about 30 m.
  - The straw is about 0.72 as light as before (mean luminance 118 against 164). A bale's face is big, flat and square to the beam where it's strongest, and the straw had been nearly as light as the sign's white (208 by day in game, against the sign's 194).
- **Budgets:** every element and level is held to `closures.BUDGET` and `COLLIDER_BUDGET` at any width up to 14 m. The vehicle isn't counted. The trestles' tubes have a triangle for every corner, so triangles run closer to vertices than on boxes.

  | Closure, 8 m wide (14 m) | Vertices | Triangles | Collider vertices |
  |---|---|---|---|
  | major | 536 (848) | 440 (740) | 0 |
  | minor, boards | 588 (756) | 502 (658) | 0 |
  | minor, bales | 324 (364) | 226 (246) | 80 (120) |
  | limited | 168 | 134 | 0 |

  Per element: the tape 76 vertices, the second tape and its sign 168, a lone sign 92. In game the first version cost no measurable frame rate (164 fps facing a major closure against 160 facing open grass).
- **Names when placed:** visual meshes `CLOSURE_*`, colliders `1WALL_CLOSURE_*` (validation: "Closures on the stubs" below).
- **To look at them:** `python scripts/closure_preview.py OUT_DIR` renders each case from the mainline (its colliders drawn in red) and writes a small test track, `mapcreator_closures`, to install with the harness; `--solid` builds it with other props solid. Confirmed in game (10 Nov): the bales' and trestles' looks, passing through the boards and stopping at the bales, and no frame-rate cost. **To confirm in game:** the night at 8-12 m and from 25 m.

## Police cars (`core/police_car.py`)

A procedural low-poly police car for each Australian jurisdiction, in that jurisdiction's livery with POLICE and its crest: the maintainer's decision for the closed side roads (10 Nov 2026). The mesh stage parks one at some major closures, at most one per 2 km and six per track ("Closures on the stubs" below). This module builds the car.

- **Drawn here, from shapes.** The body, the chequer, the lettering and each crest are our own renditions, simplified. No image of a real crest or livery is copied, traced or embedded. The liveries were researched from the services' releases, news reports and dated photos; images were viewed for reference only. The sources are in `THIRD_PARTY_NOTICES.md` ("Design references").
- **Entry point:** `police_car(code, body=None)` returns a `PoliceCar` with:
  - `meshes`: `scene.Mesh`, the type furniture hands the KN5 writer, one per material, in the car's own frame;
  - `collider`, `materials` (`scene.Material`), `textures` (texture name to its Pillow image, to save as a PNG the way the mesh stage saves `furniture.png`), `kn5_properties` (each material's KN5 values) and `night_diffuse` (each material's ksDiffuse at night).
- **Local frame:** the origin is the car's centre on the ground, +z forward, +x to the car's right, +y up, metres. The tyres touch y = 0 and the light bar is the top.
  - With x right, y up and z forward the frame is left-handed. `place(car, position, forward, up=(0, 1, 0), index=0)` maps it onto AC's axes with the car's right on its right (right = forward × up), so nothing comes out mirrored, rewinds each triangle to the KN5's rule, and names the copy `..._<index>`. `up` tilts the car on a slope.
  - `merge(meshes)` joins placed cars by material, under the vertex limit, so six cars draw as three meshes and one collider mesh.
- **Codes:** `tas`, `nsw`, `vic`, `qld`, `sa`, `wa`, `nt`, `act` (ACT Policing, part of the AFP).

### Meshes, materials and the collider

| Mesh | Material | What |
|---|---|---|
| `POLICE_<CODE>_BODY` | `police_<code>` | body, glass, wheels, mirrors |
| `POLICE_<CODE>_RETRO` | `police_<code>_retro` | the livery band's strip of faces, both sides and across the rear |
| `POLICE_<CODE>_LENS` | `police_<code>_lens` | the light bar |
| `1WALL_POLICE` | (`police_<code>`, invisible) | the collider |

- All three materials draw on one texture, `police_<code>`, so up to six cars of a track share one atlas. A body other than the jurisdiction's own adds its name (`police_nsw_sedan`), since its atlas differs.
- **The collider is the body's own shape, so nothing solid stands where nothing shows.** It's a `WALL` physics mesh (no surface KEY) with `isRenderable` 0, so the car carries its own collision, as the closure answers ask (no wall in open air).
  - **Three convex pieces:** the bonnet (nose to the windscreen's foot), the cabin (to the rear window's foot) and the boot (to the tail). `collider_pieces` gives each piece's points.
  - **Each piece** is the convex hull of the body's own cross-sections at its stations. So it follows the bonnet, windscreen, roof, rear-window and boot lines, the cabin's tumblehome and the narrowing nose and tail.
  - **Height:** from the sill up. The open space under the car stays open.
  - **Fit:** the collider's sides stand 1 cm inside the band's line. The drawn body is at most 1 cm outside the collider; the collider is at most 4 cm outside the drawn body, along the sill's lower edge. The mirrors (0.16 m) and the light bar (9 cm above the roof) are left out.
  - Each piece is closed, its faces wound outwards like the trunk colliders.
  - The first collider was a box the body's size, up to the roof. In game it stood above the bonnet and boot, where a car riding up would meet an invisible face.
- **KN5 values** (`kn5_properties`, by material; the track stage keys its `MATERIAL_PROPS` by a name's first word, so the mesh stage hands these over in each `scene.Material`, "Closures on the stubs" below):

  | Part | ksAmbient | ksDiffuse (by day) | ksSpecular | ksSpecularEXP |
  |---|---|---|---|---|
  | body (white or silver paint) and the band | 0.4 | 0.6 | 0 | 40 |
  | lenses | 0.4 | 0.6 | 0.05 | 200 |

  - ksDiffuse by day is the walls' 0.6, which a stage that doesn't look these up gives them anyway; the night adjustments start from it.
  - **No specular on the paint or the band**, which carry the lettering and the chequer. AC's `ksPerPixel` adds its specular on top of the texture, not through it, so it lifts black lettering as much as white paint (see "Night"). A stage that doesn't look these up gives them the walls' 0.05.
  - Tasmania's silver comes from its albedo alone (164-173, against the white's 224).
  - The lenses keep a small, tight glint.

### Night

`night_diffuse` gives each material's ksDiffuse at night. It goes through the furniture's `retro_materials` path in `ext_config.ini`: a `[MATERIAL_ADJUSTMENT_n]` with `CONDITION = NIGHT_SMOOTH`, and `VALUE_0_OFF` 0.6 by day.

| Part | Night ksDiffuse | First check track |
|---|---|---|
| body (`police_<code>`) | 0.35 | 0.6 (no adjustment) |
| band (`_retro`) | 1.3 | 2.0 |
| lenses (`_lens`) | 2.0 | 4.0 |

- **Why these values.** In game (10 Nov 2026), side-on in the beam at about 6 m, the first values washed out:
  - the bodies clipped to white over 36-47% of their area (1-18% by day);
  - the chequers turned pale or vanished, and the lettering went with them;
  - the lenses burned to white boxes, and from 12 m the bars showed steady red and blue, like a bar left on.
- **The causes, shared with any prop:**
  - **A near-white texture under the walls' ksDiffuse 0.6.** Every material outside the track's own surfaces takes those fallback values.
    - White at 246 under 0.6 reflects 3.7 times what the road does (95 under 0.42), taking the texture values as they are, or 11.5 times if the game decodes them from sRGB first.
    - CSP's night exposure suits the dark road, so at close range in the beam the props go past the top of the tone curve: white clips, and saturated colours pale towards white.
  - **Specular, added untextured.** `ksPerPixel` gives `txColor * (ksAmbient * ambient + ksDiffuse * light + ksEmissive) + ksSpecular * (N·H)^ksSpecularEXP * light`. That's Content Manager's re-implementation of the shader, `AcTools.Render/Shaders/Includes/DarkMaterial.Lighting.Simple.fx`, which the closure props' work found.
    - At night the headlights sit beside the eye, so N·H is near 1 across a face square to them, and the specular lifts every texel alike, black lettering as much as white paint.
    - At 0.12 and exponent 40 it came to about 40% of the white's light at the night 0.35, so the lettering's contrast fell from about 11:1 to under 3:1. Lowering the night diffuse made that share larger.
- **Round 2** (night 0.35, 1.3 and 2.0, specular 0.12; in game 11 Nov): the bodies were as bright as by day but 24-31% near-white at 6 m, the chequers kept their dark squares but pale, and most lettering faded to pale lavender. The bars read as off from 12 m.
- **What changed:**
  - white paint at 224 (was 246), and Tasmania's silver at 164-173;
  - the body comes down to 0.35 at night. Out of the beam it's lit by its ambient, which stays. That's the closure props' value too, so the two look alike side by side;
  - the band and the lenses get smaller boosts;
  - no specular on the paint and band (0.3, then 0.12), and a small, tight glint on the lenses (0.25 at exponent 60, now 0.05 at 200). Without the specular, the hot spot on a white side loses about a quarter of its light;
  - the lenses are drawn as tinted, unlit plastic: each colour darkened and greyed, nothing near white.
- White paint stays at 224 rather than the 200-210 the round 2 report suggested: whiter paint than that by day and the silver car no longer stands apart.
- **The light bar is off, not emissive.** The track stage writes ksEmissive 0 for every material, so a glow would need a new property path. A steady glow would read as a bar left on that never flashes, and whether a track config can make a material flash is unverified.
- The ini's description line still says "retroreflective furniture" for every entry, including the body's darker one.
- All of it is to retest in game (`docs/known-issues.md`). Vanilla AC ignores the file, so there the car looks as by day.

### Body and texture

- **Body.** Lofted through 10 or 11 cross-sections from the front bumper to the rear. Each section runs from the left sill up over the roof and down to the right sill, and splits the side at the band's heights, so the band is its own strip of faces.
  - Two shapes from the same code: a sedan (about a Toyota Camry: 4.89 x 1.84 x 1.445 m) and an SUV (about a Kia Sorento: 4.81 x 1.90 x 1.70 m).
  - Wheels are ten-sided with a corner on the ground. They stand 5 mm proud of the body side, so the side never hides them; the arches are painted.
  - Windows are painted on the sides. The windscreen and rear window take a glass cell.
  - A static light bar on the roof, and two mirrors.
- **Atlas:** 1024 x 1024 per jurisdiction and body. Each face takes its UVs by projection onto its view:
  - the right side (1024 x 384, about 5 mm a pixel), the left side drawn on its own so the lettering reads forwards on both;
  - the top (bonnet lettering and crest read from in front of the car), the front and the rear;
  - a wheel, the tread, glass, plain paint, and the light bar from in front, from behind (so its lettering reads both ways) and from above.
  - Each view is drawn in a drawing of its cell's size, so nothing spills into the next cell, at twice the size and scaled down for smooth edges. The lettering is Pillow's built-in Aileron (CC0), made bold with a stroke. Crests are drawn on a layer at four times their size.
  - **Cells aren't square in metres** (the sedan's side is about 405 px a metre along and 547 up; the front view 291 and 474). Lettering and crests are sized by each axis, so a word or crest has its proportions and size in metres in any view. The first atlases sized them by the horizontal scale alone: crests came out squashed into small ellipses, side lettering about a quarter short of its height, and bonnet lettering stretched about three and a half times across the car.
- **Crests** are drawn 0.30-0.40 m across (0.50 m on the NT's bonnet), bigger than on most real cars, so they read at parking distance. Bold shapes with dark edges, a few big leaves to a wreath, no small lettering. In game the first ones (0.16-0.19 m) read only as roundels, apart from Queensland's (0.30 m). Tasmania's moves above the chequer on the rear door, where there's room for it.
  - **Where they sit.** Tasmania's and Queensland's chequers run low, so their crests go above them. Where the chequer runs along the shoulder (NSW, Victoria, SA, WA, the ACT), only the windows are above it, so the crest goes on the door below it, 0.56-0.63 m up on the SUVs.
  - **The ACT's** star is drawn a darker silver edged in navy, with a thick navy ring round a bigger shield. Its pale silver star on white paint read only as its gold centre from 6 m. It's 0.40 m, at the other SUV states' height on the front door.
    - On the ACT sedan it sat lowest of the eight (centred 0.52 m up, against 0.56-0.63 m), and in game it read as a small shape low on the door. Traced from the built KN5's UVs, it was where it was meant to be; the sedan's lower door was the trouble. The ACT now takes the SUV body (see "Bodies" below).
  - **The tests:**
    - each crest stands clearly off its paint over at least 27% of its box (the ACT's 39%, against 27% before);
    - each crest, traced from the atlas through the mesh's UVs onto the car, is on both sides at the same place, at its size and height, and off the chequer.
- **The band follows the shoulder.** Along the doors it runs between the livery's heights. Towards the nose and tail it drops with the belt line, keeping its depth (NSW's "sweeps down towards the headlight"), and the geometry's strip drops with it. Queensland's wedge keeps its foot instead.

### Budget

| | Sedan | SUV |
|---|---|---|
| Vertices | 584 | 556 |
| Triangles | 302 | 288 |
| Collider | 52 vertices, 92 triangles | 48 vertices, 84 triangles |

- **Vertices:** under the intersections report's estimate of about 700. Each face keeps its own vertices (its UVs come from its view), so the four wheels are a third of them (50 each), the body about 320, the mirrors 40 and the light bar 20.
- **Texture:** one 1024 x 1024 DXT1 atlas with its mips, 699 KB, shared by every car on the track.
- **Six cars on a track** (the cap), sharing the atlas, merged: an 873 KB KN5, of which 174 KB is geometry (3,624 vertices in four meshes, colliders included). Zipped, 105 KB, because the atlas is mostly flat colour. One car: 729 KB, 75 KB zipped.

### Liveries

Current marked general-duties cars, from releases and dated photos (2019-2026). Colours are approximate unless a source gave an official one (NSW's PMS 287).

| Code | Service | Body | Paint and band | Lettering | Crest (our rendition), where | Light bar |
|---|---|---|---|---|---|---|
| `tas` | Tasmania Police | SUV | Silver (164-173). Two-row blue/white chequer low on the doors, a lime line on top, across the tailgate | "Police" above the line on the front doors; tailgate | Rayed gold star under a crown; blue disc with a gold lion walking. Rear door, above the chequer (on the real cars smaller, below it) | blue and red, POLICE centre |
| `nsw` | NSW Police Force | SUV | White. Two-row blue/white chequer along the shoulder, dropping to the headlight | "Police" large and low across both doors; bonnet; "000 EMERGENCY" (0.085 m) on the rear quarter | Laurel wreath and white ring under a crown; brown eagle with the state's shield in a light blue disc; scroll. Front door, ahead of "Police" | blue, clear, red |
| `vic` | Victoria Police | SUV | White. Three-row blue/white chequer at door-handle height, "131 444" and "000" panels set into it | "Police" low round the B-pillar; bonnet | In one colour, as on the cars: wreath, five-pointed star point down, crown, Southern Cross. Front door, ahead of "Police" | blue on the driver's side, red |
| `qld` | Queensland Police Service | sedan | White. The 2023 design: large blue/white squares in a wedge on the lower body, two rows behind the front wheel rising to four at the rear axle; orange-red lines along its top and the sill | "POLICE" in capitals, blue keylined white and orange-red, on the front door | White eight-pointed cross edged red on a wreath, under a crown; blue ring, navy disc with a small crown; scroll. Large, rear door | blue and red |
| `sa` | South Australia Police | SUV | White. Three-row black/white chequer between yellow lines, breaking into scattered squares over the front guard | "POLICE" in tall black capitals keylined yellow across both doors; bonnet | Wattle wreath under a crown; light blue ring, white disc with a black and white piping shrike, wings spread. Front door | red on the driver's side, blue; POLICE centre |
| `wa` | WA Police Force | SUV | White. Two-row blue/white chequer along the shoulder, a thin orange-red line above | "Police" on the doors | In one colour, as on the cars: eight-pointed star under a crown, banners, a black swan. Front door, near "Police" | red and blue |
| `nt` | NT Police | SUV | White. Two-row blue/white chequer along the shoulder between ochre lines | "Police" on the front doors below the band; bonnet | Wreath with two desert roses under a crown; an ochre kangaroo standing; blue and gold scrolls. Bonnet | red and blue |
| `act` | ACT Policing (AFP) | SUV | White. Three-row near-black/white chequer along the shoulder between lime lines, bigger squares on the front guard, lime flashes on the bumper | "Police" low on the rear door; ghosted grey on the bonnet | The AFP's: an eight-pointed star under a crown, a navy ring round the Commonwealth arms' shield, a scroll; drawn bolder than the badge (a darker silver star edged in navy, a thick ring). 0.40 m on the front door, below the chequer, at the other SUV states' height (smaller and lower on the real cars) | red and blue, POLICE centre |

- **What the sources leave open**, drawn as a common Australian layout:
  - which side the red is on, except Victoria (blue on the driver's side) and SA (red);
  - the rear, except Tasmania: the band across it and "Police" above it (no band on Queensland's wedge cars).
- **Least certain:** WA's general-duties livery after 2023, the NT's after 2018, the ACT's current models, and when SA's chequer changed from blue to black. Tasmania's silver comes from photos only.
- **Bodies** follow the current fleets: SUVs almost everywhere (Santa Fe, Sorento, Kluger, RAV4, Tiguan; HiLux and LandCruiser in the NT), and Camrys in Queensland (RAV4s replacing them). The ACT's Subaru Liberty of 2019 gave way to the Outback in the 2022 photos, which the SUV fits better. Either body can be asked for.
- Call signs aren't drawn: they belong to individual cars.

### Which jurisdiction

`jurisdiction_for(region, lonlat)` gives one jurisdiction for a whole track:

- A region that names its state decides (`Region.state`; Tasmania does).
- Otherwise the route's points do, by `jurisdiction_at(lon, lat)`, and the jurisdiction holding most of them wins. So a route across a border takes the state of most of its length, and a car near the coarse border line matches the rest of the track. A tie goes to the first along the route.
- With neither, `tas`.
- The borders are coarse, to within about 10 km. Most are meridians and parallels (WA at 129° E, the NT to 26° S and 138° E, Queensland's west at 138° E and 141° E, SA's east at 141° E, then 140.97° E south of the Murray). Queensland/NSW follows 29° S, the rivers past Goondiwindi and the ranges to Point Danger. NSW/Victoria follows the Murray and the line from its source to Cape Howe. The ACT is an outline of about 20 points. Every point in Australia's box gets a jurisdiction.

### Visual check

`PYTHONPATH=src python scripts/police_cars_check.py [--out DIR]` writes, by default to a folder in the system's temp directory (never inside the repository):

- `lineup_right.png`, `lineup_left.png` and a close-up of each car from the front and the rear, from a small software renderer that culls back faces as AC does (a face wound the wrong way shows as a hole);
- each atlas as a PNG;
- `mapcreator_police_check/`, a test track: a straight 600 m road with the eight cars parked on each verge, so you see each car's right side on the left verge and its left side on the right. Copy it to `assettocorsa/content/tracks/`, pick "Police car check", and set a night time (CSP) to see the bands and light bars catch the headlights. Its `ext_config.ini` carries the cars' night adjustments, as a built track's would.

## Closures on the stubs (`core/stub_closures.py`)

The mesh stage closes every side road's stub off the stage with the closure props, and parks police cars at some of the major closures: the maintainer's answers to the intersections research (10 Nov 2026). A junction whose stub was left out (`stubs.json`'s `skipped`) gets nothing, and `"junctions": "none"` gives neither stubs nor closures.

- **Level**, the junction's closure: **major** gets course tape, barrier boards across the full width with ROAD CLOSED, and a police car if it's chosen (below); **minor** gets tape, one obstacle and a sign; **limited** gets tape and a sign. No arrows of any kind.
- **A minor closure's obstacle:** barrier boards or straw bales, about half each, from a hash of the junction's OSM node and way (`stub_closures.obstacle_for`). So it never changes between builds, and a real junction gets the same one on every track that passes it. Bales go only where the whole row keeps 5 m from the road's edge (they're solid); elsewhere boards. None needed switching on the examples.
- **Layout:** `closures.layout` from the closure's origin: tape at 4 m, the obstacle at 12 m, the second tape and sign at 37 m; on a stub under 45 m the sign stands on the obstacle.
- **The origin.** A stub's `d` runs along its centreline from the road's edge line, so on a skewed junction a line square across the stub 4 m in runs back onto the road on its acute side. The closure's origin (its local z = 0) is the first place along the stub, half a metre at a time, where its tape keeps 3.5 m from the road's edge, stake to stake. On a square junction that's d = 0 and the tape stands 4 m in, as the research's Q8 has it; on a skewed one the whole closure moves in, up to 26 m on Mount Dandenong Tourist Road's shallowest junctions.
- **Width.** In the throat the paved width is the stub and its flares. So each line of props spans the stub's paved outline (`stubs.json`'s `outline`) along the line square across the stub, and 1 m more onto each verge, as the kit's 8 m does for a 6 m road. Otherwise cars would go round on the flare. Where one flare is longer, the props' centre moves off the stub's centreline with the span. No line reaches nearer the road's edge than 1.5 m.
- **Ground.** Feet stand on the built ground: the highest road, stub, verge or terrain triangle under each point (`stub_closures.TopSurface`). `closures.place` stands each line of props rigidly at its own distance in, so the ground under a prop is read in its line's frame.
- **Overshoot** (straight ahead where the route turns): the route turns at least 30 degrees at the junction, and the side road leaves within 30 degrees of the way a car arrives, nearer straight on than the route goes, in a direction one of the track's layouts drives. Its closure has nothing solid in its first 25 m (`closures.closure(overshoot=True)`): the obstacle stands at 27 m. On the examples there's one: Gunning Street, on Brinktop Road.
- **Police cars:** only at major closures with a vehicle slot, no two within 2 km of each other along the route, and six a track at most (Q7). The biggest side roads come first: by class (trunk, primary, secondary, tertiary; a link after its road), then width, then length, then station. One jurisdiction a track (`police_car.jurisdiction_for`: the region's state, else most of the route's points).
  - A car keeps its slot's distance in and heading: behind the boards, right of the middle, nose out and turned 30 degrees in. Across the stub it stays within the span where it stands, so a slot sized to boards across a wide throat still parks it on the side road. Its height and tilt come from the ground under its four wheels.
  - Where its collider would come within 5 m of the road's edge or leave the stub and its verge, it tries the mirror image (left of the middle). Failing both, that closure gets no car and the next in line may.
- **Collision:** straw bales and the police cars are solid; boards, trestles, signs and tape are passable (Q9, option 2). Nothing solid stands within 5 m of the road's edge (placement keeps 5.25 m).
- **Meshes:** every closure merged by `closures.merge`: `CLOSURE_k` (one per material) and `1WALL_CLOSURE_k`. The cars, by `police_car.merge`: `POLICE_<CODE>_k`, `POLICE_<CODE>_RETRO_k`, `POLICE_<CODE>_LENS_k` and `1WALL_POLICE_k`. The atlases are saved beside the scene as `closure.png` and `police_<code>.png`, as `furniture.png` is.
- **Materials:** `closure` and `closure_retro` take the walls' values (`track_stage.MATERIAL_PROPS["closure"]`, as their check track had them), and `closure_retro` goes to 1.3 at night (`track_stage.NIGHT_DIFFUSE`). A police material brings its own KN5 values and night value in its `scene.Material` (`properties` and `night_diffuse`, from `PoliceCar.kn5_properties` and `.night_diffuse`). The track stage writes those as given and adds the night values to `ext_config.ini`'s adjustments (`track_stage.retro_materials`).
- **What doesn't change:** the mainline, its half widths, the AI line and racing line, `ai_hints.ini`, the grid, the spawns, the gates and the run-off. On the six examples every `fast_lane.ai`, `ai_hints.ini`, `surfaces.ini`, `map.ini`, `map.png`, `ui_track.json` and `centreline.geojson` is byte for byte the stubs' build's, and so is every KN5 dummy and every mesh that build had.
- **`04_mesh/closures.json`** (version 1): `closures`, one per stub in order along the route, each with `junction`, `name`, `highway`, `station_m`, `side`, `level`, `obstacle` (`boards`, `bales`, or null on a limited closure), `overshoot`, `origin_m`, `elements` (`tape`, `obstacle`, `warning` for the second tape and its sign, `sign` for a lone sign; each with `d_m` metres in from the road's edge line, `width_m`, `offset_m` the props' centre left of the stub's centreline, `in_throat` and `cut_by_road`), `car` (false, or `mirrored` and `d_m`), and `note` where bales became boards.
- **Validation** (`validate.closure_checks`), failures unless:
  - the `1WALL_CLOSURE_*` and `1WALL_POLICE*` colliders are invisible;
  - none has a vertex within 5 m of the road's edge (its own `1ROAD` and `1GRAVEL` meshes, not the stubs') or within 3 m of a spawn, in every layout;
  - every `CLOSURE_*` and `POLICE_*` vertex lies within 3.5 m of a stub's paved outline, every stub carries closure props, and none stands on a side road that has no stub.
- **Cost:** `docs/known-issues.md`, "Side-road junctions".
