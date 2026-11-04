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
  centreline.geojson             ODbL: the centreline, heights and widths the track was built from
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

| Mesh | Surface (`surfaces.ini` KEY) |
|---|---|
| `1ROAD_###` | ROAD |
| `1GRASS_V###L/R` (verge and skirt) | GRASS |
| `1GRASS_T###` (terrain) | GRASS |
| `1WALL_RAIL_L/R###`, `1WALL_END_START/FINISH` and `1WALL_TRUNK_###` (invisible, `isRenderable` 0), and `1WALL_BUILDINGS` | solid barrier (no KEY: any physics mesh named WALL) |
| `RAIL_*`, `BUILDING_ROOF_*`, `WATER_*` | visual only (no leading digit) |

- Each physics mesh name matches exactly one KEY. There is deliberately no `WALL` KEY, because defining one is reported to make walls driveable.
- **Terrain:** `1GRASS_T###` reaches 700 m either side of the road (`mesh_stage.TERRAIN_CORRIDOR_M`), and the world ends there. It's one Delaunay surface over lattices of 4 m to 60 m out, 8 m to 140 m, 16 m to 300 m, then 32 m. The DEM is read 100 m further (`profile_stage.CORRIDOR_M`). Trees and buildings stop at 250 m, so the ground beyond is bare. Until #34 it ended at 300 m, and power towers further out stood on nothing.
- **Logic dummies:**
  - `AC_AB_START_L/R` and `AC_AB_FINISH_L/R` sit 0.5 m outside each road edge. L is left of the driving direction.
  - The start gate is 50 m from the layout's start. The finish gate leaves a run-off of real road after it: 250 m, or 15% of a short route, never under 50 m. With only 50 m, AC's AI crossed the finish, reached the end of its line, turned round and left the road (M3).
  - **Grid:** `AC_PIT_k` and `AC_START_k` share each grid spot (`layout.grid`), up to 8 per layout.
    - Spot 0 is on the centreline 8 m before the start gate.
    - Behind it come pairs 8 m apart, 2 m either side of the centreline, where both half-widths are at least 3 m. Otherwise single file.
    - Spots go no closer than 8 m to the road's start.
  - `AC_HOTLAP_START_0` sits 30 m before the start gate.
  - Spawns are 0.3 m above the road and face the driving direction. Confirmed in game at M3: the car lands on all four wheels, facing along the road.
- **Pits:** `pitboxes` in `ui_track.json` must equal the number of `AC_PIT_n` dummies (up to 8). Content Manager offers that many cars less one as AI opponents, so with one pit box there were none (M3).

## Custom Shaders Patch (`extension/`)

Vanilla AC ignores this folder. `ext_config.ini` holds:

- **Grass FX and Rain FX:** grass grows on the grass and terrain materials, and the road gets puddles.
- **Water:** `[INCLUDE: common/materials_track.ini]` and `[Material_Water]` give the `water` material CSP's water shader, as `LAKE` or `SEA`. Vanilla AC shows a flat, glossy blue-green surface.
- **Tree FX:** `[TREES]` with `LIST_0=trees/trees.txt` and `SURFACE_MATERIALS=terrain, grass`, so CSP stands each tree on the ground. These are the track's only trees: the KN5 has none, so vanilla AC shows none.
  - Tree FX trees have no physics, so the KN5 gets an invisible collider for each trunk a car can reach (`1WALL_TRUNK_###`, `scenery.trunk_colliders`). A trunk qualifies if it stands within 20 m of the road edge and is over 10 cm across.
  - Each collider is an eight-sided prism with its corners on the drawn trunk's circle, from 1 m below the ground to 3 m above it. Its radius is the model's trunk trailer in metres, times the tree's `size` and `width`.
  - The faces are wound outwards, as the house walls are. The prisms are grouped into a mesh per 400 m of road, under the vertex limit, at about 200-560 trunks per km.

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
- **Grid:** `hasGrid = 0`; the game builds and caches the grid itself.
- **End:** the AI line stops 40 m before the road's end (`AI_END_SHORT_M`), short of the invisible end wall.
- **Hints (`data/ai_hints.ini`):** AC's AI sets its own speed (M3), so hints are the way to slow it. Three `[HINT_n]` sections run from the finish gate to the end of the line, as fractions of the spline: pace 0.5, then 0.2 from 150 m before the line's end, then **0** for its last 60 m.
  - A hint's VALUE scales the speed AC works out from the line's curvature, not the car's speed. Measured in game (4 Nov): AC's own normalised position matches our line exactly in both layouts, and the hints fall where we put them. Yet down Grasstree Hill's straight reverse run-off (radius 287-1,000 m) 0.2 and 0.05 left the AI at about 100 km/h, while forward, where the run-off bends (radius 19-210 m), the same hints slowed it to 12-15 km/h.
  - Kunos's shipped hints fit that reading: they're all short ranges around corners, 0.6 to 0.92 (`ks_laguna_seca`'s Corkscrew, `ks_black_cat_county`, `ks_nurburgring`), and none sits on a straight or near 0. There's no Kunos or CSP documentation of the file.
  - So only 0 stops a car on a straight. With 0.05 the AI crawled past the end of its line at about 6 km/h and off the road; with 0 it stops on the road about 60 m before the end of its line, about 100 m short of the end wall, in both layouts. On the straight reverse run-off the 0.2 stretch still does nothing, and the AI brakes from about 100 km/h in the 45 m before the 0 hint: hard, but straight and on the road.
  - Several cars queue behind the first one stopped. That leaves them short of the 0 hint, where the pace is still 0.2, so only AC's car-following holds them, creeping at about 1 km/h. In two of the four multi-car runs that reached the run-off (Brinktop Road, and a later race on Grasstree Hill), one crept into the car ahead (slight damage to both, 1.3). A community tutorial on AI lines gives the same advice: a 0 hint near the end of a point-to-point line, which should end after the last timing gate (Overtake.gg, "Tutorial on making 2-way traffic (or other) AI lines and layouts").
  - Earlier: with a single hint of 0.5 and the line running to the road's end, the AI reached the end wall at about 45 km/h, and AC put it back at the start (3 Nov).
  - Kunos's own point-to-point track, `ks_drag`, sidesteps the problem: one straight AI line runs 2,469 m, past every layout's finish (`AC_OPEN_FINISH_L/R`, 204 m to 2,004 m along it), so even the 2,000 m layout leaves 465 m of line after its finish.
- **Placement:** points every 1 m along the centreline, 0.4 m above the road. Points on the surface have crashed the game in another generator.
- **Speed:** grip-limited, `v = sqrt(μ g R)` with μ = 1.05, capped at 130 km/h. It then gets a forward pass for acceleration and a backward pass for braking, both adjusted for grade. AC's AI finished the forward layout at M3, but it ignores this speed and set its own, up to 214 km/h in an Abarth 500.

## Validation

`mapcreator validate` reads every file back and checks the rules above, including measured heights of spawns and AI points against the actual road triangles. The in-game checks (does it load, are the wheels on the ground, can the AI drive it) live in `harness/`.

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
- **Reach:** supports stand only on the terrain mesh, with their whole base within 5 m of its edge: 695 m of the centreline, since the terrain reaches 700 m (#34; it was 300 m).
  - Towers further out stood in the sky, their bases floating past the ground's edge (in game, 4 Nov), so a span with a support beyond it isn't built.
  - On Grasstree Hill that leaves out both TasNetworks lines; widening the terrain (#34) would bring them back.
- **No wind:** CSP's `[WIND] DYNAMIC_CABLES` (undocumented; CSP's own Laguna Seca config uses it) does act on the wires: its log shows "Preparing cables". But it redraws them as tubes about 8 times thicker than the 3.5 cm mesh. Without it they draw as thin lines and look right (in game, 4 Nov), so `ext_config.ini` leaves it out.
