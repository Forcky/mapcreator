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
| `1GRASS_V###L/R` (verge and skirt) | GRASS | `grass` |
| `1GRASS_T###` (terrain) | GRASS | `terrain`: `ground` |
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
- **Terrain:** `1GRASS_T###` reaches 700 m either side of the road, and the world ends there. It's one Delaunay surface over lattices of 4 m to 60 m out, 8 m to 140 m, 16 m to 300 m, then 32 m. Until #34 it ended at 300 m, and power towers further out stood on nothing.
  - **Reach:** the distances beside the road are one set (`core.reach`). Trees and buildings stand within 600 m of the centreline (the scenery reach). The terrain reaches 100 m further, so nothing at the scenery's edge stands at the world's edge, and the DEM is read 100 m beyond that (800 m). Power-line supports stand anywhere on the terrain, out to 695 m (see "Power lines").
  - Until the scenery reach, trees and buildings stopped at 250 m and the hills beyond them were bare.
- **Logic dummies:**
  - `AC_AB_START_L/R` and `AC_AB_FINISH_L/R` sit 0.5 m outside each road edge. L is left of the driving direction.
  - The start gate is 50 m from the layout's start. The finish gate leaves a run-off of real road after it: 250 m, or 15% of a short route, never under 50 m. With only 50 m, AC's AI crossed the finish, reached the end of its line, turned round and left the road (M3).
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

- **Grass FX and Rain FX:** grass grows on the grass and terrain materials, and the road gets puddles.
  - The seal (`road`, `road_plain`) occludes grass, gets puddles and soaks.
  - Gravel gets puddles too and is rough (no reflections, only darker when wet). One-lane gravel (`gravel_narrow`) is also a grass material, so Grass FX grows blades on its grass strip (CSP's default colour mask grows grass only on green, mid-dark pixels); two-lane gravel occludes grass instead.
  - No tiling fix (`ksPerPixel_tilingfix`) on the gravel: it shifts the texture at random and would break the wheel paths.
- **`[TYRES_FX]`**, only on a track with gravel: the colours of dust and of dirt on the tyres, as three pairs (`DIRT_COLOR_A/B` from the gravel texture, `GRASS_COLOR_A/B` from the grass, `SOIL_COLOR_A/B` from the terrain's bare earth). A is the texture's mean colour and B 0.55 of it. A track's own section switches CSP's generic one (`config/tracks/generic/generic.ini`) off entirely (CSP 0.2.4), so ours carries all three pairs and the generic's four gains (`DIRT_GAIN`, `DIRT_FADE_GAIN`, `GRASS_GAIN`, `GRASS_FADE_GAIN`) at 1.0.
- **Water:** `[INCLUDE: common/materials_track.ini]` and `[Material_Water]` give the `water` material CSP's water shader, as `LAKE` or `SEA`. Vanilla AC shows a flat, glossy blue-green surface.
- **Tree FX:** `[TREES]` with `LIST_0=trees/trees.txt` and `SURFACE_MATERIALS=terrain, grass`, so CSP stands each tree on the ground. These are the track's only trees: the KN5 has none, so vanilla AC shows none.
  - Tree FX trees have no physics, so the KN5 gets an invisible collider for each trunk a car can reach (`1WALL_TRUNK_###`, `scenery.trunk_colliders`). A trunk qualifies if it stands within 20 m of the road edge and is over 10 cm across.
  - Each collider is an eight-sided prism with its corners on the drawn trunk's circle, from 1 m below the ground to 3 m above it. Its radius is the model's trunk trailer in metres, times the tree's `size` and `width`.
  - The faces are wound outwards, as the house walls are. The prisms are grouped into a mesh per 400 m of road, under the vertex limit, at about 200-560 trunks per km.
- **Far trees are billboards.** Each model's three LODs share ten model heights, the rule Content Manager's converter uses (`AcTreeModelConverter`: `DISTANCE = height × 10 × (i + 1) / LODs`). LOD 0 ends 28-89 m out and the last 85-267 m, by model.
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
- **Hints (`data/ai_hints.ini`):** AC's AI sets its own speed (M3), so hints are the way to slow it. Three `[HINT_n]` sections run from the finish gate to the end of the line, as fractions of the spline: pace 0.5, then 0.2 from 150 m before the line's end, then **0** for its last 60 m. On a track with unsealed road, the gravel's hints come before them (below). The sections are numbered from `HINT_0` in order along the line.
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
- **The line is a racing line** across the road's full width (`core/racing_line.py`), by the maintainer's decision of 4 Nov 2026: out-in-out through the bends, as on a closed road, not kept to the left lane.
  - It's a minimum-curvature line after Heilmeier et al., "Minimum curvature trajectory planning and control for an autonomous race car" (Vehicle System Dynamics 58(10), 2020, doi:10.1080/00423114.2019.1631455), re-implemented from the paper. TUM's implementation is LGPL-3.0 and wasn't used.
  - Offsets from the centreline at stations 2 m apart, chosen to minimise the line's summed squared curvature, linearised around the line found and solved again, three times.
  - It stays 1.2 m from each road edge (`EDGE_MARGIN_M`, to the line: a car 1.7 m wide has 0.35 m to spare), and on the inside of a tight bend within half the bend's radius.
  - Its curvature is limited to the centreline's tightest within 20 m, so it never takes a bend tighter than the road does. Without that limit it hugged the inside of long bends, 48 m on a 50 m bend.
  - It keeps to the centreline for the first 60 m of each layout (the grid spots, the start gate and the hot-lap spawn) and from 30 m before the finish gate to its end (the run-off, where the AI is stopped). It leaves and rejoins the centreline over 50 m.
  - Each layout gets its own line: the reverse isn't the forward one backwards.
  - Built on Grasstree Hill (6.4 m wide), the tightest radius outside the held stretches went from 26.2 m to 30.3 m and the planned run from 386 s to 375 s. On Brinktop Road, 7.1 m to 16.1 m, its slowest planned speed from 31 to 46 km/h. On Lyell Highway (63 km), 13.9 m to 17.3 m.
- **Placement:** points about 5 m apart along the line itself (`AI_POINT_SPACING_M`), evenly from its start to its end, each 0.4 m above the road surface where it stands, crown and banking included. Points on the surface have crashed the game in another generator.
  - Why 5 m: AC measures the spline from the points in float32, and the rounding inflates the length it reports (`TRACK LENGTH` in `log.txt`) by about 1/spacing². On Lyell Highway (63.06 km, reaching ±21.7 km east-west from the local origin), measured in game on 4 Nov by swapping only `fast_lane.ai`: 1 m points made it 83,747 m (+32.8%), 2 m +8.8%, 5 m +1.3% (positions within 203 m everywhere, 26 m at the end), 10 m +0.3%. Grasstree Hill's 1 m line read 1.1% long.
  - The inflation grows with distance from the origin, so every spline fraction (the run-off hints, AC's progress) moved with it on a long track.
  - 10 m points cut the chord 0.8 m inside a 15 m hairpin. At 5 m the AI drove Lyell's 15.8 m and 13.9 m hairpins at 40 and 36 km/h, within 1.0 m of the line: as good as the 1 m line or better.
  - Speeds, gas, brake, radius and grade are worked out every metre along the line, then read at each point.
- **Speed:** grip-limited on the line's own curvature, `v = sqrt(μ g R)` with μ = 1.05, capped at 130 km/h. It then gets a forward pass for acceleration and a backward pass for braking, both adjusted for grade. AC's AI finished the forward layout at M3, but it ignores this speed and set its own, up to 214 km/h in an Abarth 500.

## Validation

`mapcreator validate` reads every file back and checks the rules above, including measured heights of spawns and AI points against the actual road triangles. The road is every `1ROAD` and `1GRAVEL` mesh: spawns, the AI line, and the tree and power colliders' clearance are all measured against both. The in-game checks (does it load, are the wheels on the ground, can the AI drive it) live in `harness/`.

For the AI line it also checks:
- distances strictly increase, and each spawn faces along the line;
- every point is at least 1.0 m (the 1.2 m margin less 0.2 m) inside both edges: by its side distances, and on the road's triangles either side of it;
- the line is nowhere more than 5% tighter than the road within 30 m of it: a warning, never a failure (the maintainer's decision, #44). The road's middle is found from each point and its two side distances, and both radii come from a spline through the points.
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
- **Reach:** supports stand only on the terrain mesh, with their whole base within 5 m of its edge: 695 m of the centreline, since the terrain reaches 700 m (`core.reach.POWER_M`; #34, it was 300 m). That's past the trees and buildings (600 m): towers are tall, few and cheap to draw, so they reach as far as the ground does (the maintainer's decision, 4 Nov).
  - Towers beyond the terrain stood in the sky, their bases floating past the ground's edge (in game, 4 Nov), so a span with a support beyond the reach isn't built.
  - With the 300 m terrain, Grasstree Hill's two TasNetworks lines were left out. #34 widened the terrain to 700 m, which brought back 8 towers, 140-645 m out.
- **No wind:** CSP's `[WIND] DYNAMIC_CABLES` (undocumented; CSP's own Laguna Seca config uses it) does act on the wires: its log shows "Preparing cables". But it redraws them as tubes about 8 times thicker than the 3.5 cm mesh. Without it they draw as thin lines and look right (in game, 4 Nov), so `ext_config.ini` leaves it out.

## Roadside furniture (`core/furniture.py`)

Built in the mesh stage, visual only, in `FURNITURE_n` meshes with one material (`furniture`) and a texture atlas the stage draws (`furniture.png`).

- **Guide posts** stand on both sides, 0.8 m beyond the road's edge and about 1 m tall, with a black band and a reflector on each face.
  - Red faces the traffic for which the post is on the left; white faces the traffic for which it's on the right.
  - Spacing follows Main Roads WA's "Design of Guide Posts" (citing AS 1742.2-2009 clause 4.2.4.4): 150 m on straights. On curves it's closer, and closer still on the outside: 6/12 m (outside/inside) below 100 m radius, 10/20 m to 200 m, 15/30 m to 300 m, 20/40 m to 400 m, then 40, 50, 60 and 90 m both sides out to 2,000 m.
  - The spacing is checked at every station, so posts close up as soon as a curve begins.
  - None where a guard rail runs, and none on unsealed road.
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
