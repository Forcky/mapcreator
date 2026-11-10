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
  centreline.geojson             ODbL: the centreline, heights, widths and surface runs the track was built from
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

`mapcreator validate` reads every file back and checks the rules above, including measured heights of spawns and AI points against the actual road triangles. The road is every `1ROAD` and `1GRAVEL` mesh: spawns, the AI line, and the tree and power colliders' clearance are all measured against both. The in-game checks (does it load, are the wheels on the ground, can the AI drive it) live in `harness/`. With side-road junctions, it also lists them and checks nothing stands on them ("Side-road junctions" below).

For the AI line it also checks:
- distances strictly increase, and each spawn faces along the line;
- every point is at least 1.0 m (the 1.2 m margin less 0.2 m) inside both edges: by its side distances, and on the road's triangles either side of it;
- every point keeps the margin for its planned speed (1.2 m at 60 km/h to 2.5 m at 120 km/h, or half the road where it's narrower than twice that), less 0.2 m: a warning, never a failure, as for the line's radius. The line is built for the faster speeds the cars may reach (up to 3.3 m at 200 km/h), but the AI file's speeds stop at 130 km/h, so the check can't see that part;
- the line is nowhere more than 5% tighter than the road within 30 m of it: a warning, never a failure (the maintainer's decision). The road's middle is found from each point and its two side distances, and both radii come from a spline through the points.
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

The route stage finds the real side roads that meet the route and writes them to `01_route/junctions.json` (below). Nothing is drawn for them yet: guard rails, roadside furniture, trees, buildings and power-line supports are kept out of their mouths, and the timing gates move clear of them. The side roads' stubs and closures are to build on the file. A job with `"junctions": "none"` builds without any of it, exactly as before (`Job.junctions`, `closed` by default, and left out of the track id while it holds its default).

- **Which side roads count** (the maintainer's decision, 10 Nov 2026). At each node of the route, a way leaving it is a junction when:
  - its way is a class the router uses (`overpass.DRIVABLE`: motorway to living_street, `road` and the links). The road data holds no other kind, so driveways, other service roads and tracks never count;
  - it isn't private: the most specific of `motor_vehicle`, `vehicle` and `access` that's tagged isn't `private` or `no`;
  - it leaves at least 15° off the mainline, against both the road ahead and the road behind, measured to its point 15 m out;
  - it's clear of the rest of the route. It's followed until it comes within 10 m of another part of the route (one where the road distance is more than 1.4 times the straight distance, plus 5 m), and one that does so within 15 m isn't a junction;
  - it has no `bridge`, `tunnel` or `layer` other than 0 in its first 30 m: a road passing over or under isn't a junction.
- **How far:** each is followed out to 100 m, the stubs' length to come. It follows its own way, then, at the way's end, the straightest way that turns less than 60°. It stops sooner at a dead end, at a branch, or where it meets the route. A side road at the route's very first or last point isn't one: the end wall closes it.
- **Closure level**, for the closures to come: `limited` where the most specific access tag is `permit`, `forestry`, `agricultural`, `destination`, `delivery`, `customers` or `military`; else `major` (motorway, trunk, primary, secondary, tertiary and their links) or `minor` (unclassified, residential, living_street, road).
- **On the examples:** Grasstree Hill 8, Brinktop Road 18, Esperance Coast Road 10, Lower Swamp Road 3, Lyell Highway 4 (Burns Dam Road, `access=permit`, is limited) and Mount Dandenong Tourist Road 73.
- **Roundabouts** the route drives or touches (`junction=roundabout` or `circular`) are recorded with their ring and exits, and built as before: the route drives only its arc of the ring, its exits aren't junctions, and nothing is cleared at them. The validator names them. None of the examples passes one.
- **The mouth** of a junction is the stretch of the mainline where the side road's carriageway crosses the band from the road's edge to 3 m beyond it (`MOUTH_BAND_M`: guide posts stand at 0.8 m, rails at 1 m, signs at 2 m), plus 5 m along the road either side (`MOUTH_MARGIN_M`). The carriageway is its line as `junctions.json` has it, at its tagged or class-default width as mapped (`section.width_from_tags`). In it:
  - no guard rail or rail collider. The rail mask is cut after `rail_mask` has closed its gaps and extended its runs, so a mouth stays open, and a piece left shorter than 24 m beside it goes too;
  - no guide post (one due there goes up at the first station past it), no CAM (left out) and no speed sign (it moves to the nearest station clear of the mouth);
  - no timing gate (below).
- **Along the side road**, for its first 100 m and on its own side of the mainline, the mainline's own clearances apply from the side road's edge: no tree within 4 m (the 2 m verge and 2 m more), and so no trunk collider; no building within 2 m; no power-line support whose base comes within 0.5 m (it's dropped with its spans, as one on the road is).
  - They're taken out after everything else is chosen, and every other tree and building keeps its random draws, so the rest of the scenery is as without junctions. Trees also stay out of the 3 m round a building taken out, as round every building.
- **What doesn't change:** the road, the verges and the terrain, the AI line and the racing line, `ai_hints.ini`, the grid and the spawns. On the six examples every road, verge, terrain and wall mesh, every `fast_lane.ai` and `ai_hints.ini`, and `centreline.geojson` are byte for byte as without junctions.
- **Timing gates:** a gate whose place falls in a mouth moves to the nearest place clear of every mouth, 0.5 m past its end, by at most 40 m (`layout.gate_distances`). A start gate moved back stays at least 4 m ahead of the grid's front spot (so at 46 m or more), and the finish gate leaves at least 50 m of run-off. With nowhere clear, it stays, and the validator warns. The grid, the hot-lap spawn, the AI line and its hints keep to the gate's usual place, so the timed length changes by those few metres. No gate moves on the examples: the nearest mouth to a gate is Percy Street's on Brinktop Road, from 58 m, 8 m past its start gate.
- **Validation:** a warning listing the junctions (name, class, closure level and where), and checks that no guard rail or rail collider, guide post or sign has a vertex on a side road's carriageway within 3 m of the road's edge, and no tree or power collider has one on a side road's carriageway in its first 100 m. A timing gate's end on a mouth's carriageway is a warning: the stubs to come would make it a failure ("gate spans the road").

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
