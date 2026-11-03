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
| `1WALL_RAIL_L/R###` and `1WALL_END_START/FINISH` (invisible, `isRenderable` 0), and `1WALL_BUILDINGS` | solid barrier (no KEY: any physics mesh named WALL) |
| `RAIL_*`, `BUILDING_ROOF_*`, `WATER_*` | visual only (no leading digit) |

- Each physics mesh name matches exactly one KEY. There is deliberately no `WALL` KEY, because defining one is reported to make walls driveable.
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
- **Hints (`data/ai_hints.ini`):** AC's AI sets its own speed (M3), so hints are the way to slow it. Three `[HINT_n]` sections run from the finish gate to the end of the line, as fractions of the spline: pace 0.5, then 0.2 from 150 m before the line's end, then 0.05 for its last 60 m.
  - With a single hint of 0.5 and the line running to the road's end, the AI slowed to 40-75 km/h but still reached the end wall at about 45 km/h, and AC put it back at the start (in game, 3 Nov).
  - **To confirm in game (M4):** that the ramp stops AI cars on the road before the wall.
- **Placement:** points every 1 m along the centreline, 0.4 m above the road. Points on the surface have crashed the game in another generator.
- **Speed:** grip-limited, `v = sqrt(μ g R)` with μ = 1.05, capped at 130 km/h. It then gets a forward pass for acceleration and a backward pass for braking, both adjusted for grade. AC's AI finished the forward layout at M3, but it ignores this speed and set its own, up to 214 km/h in an Abarth 500.

## Validation

`mapcreator validate` reads every file back and checks the rules above, including measured heights of spawns and AI points against the actual road triangles. The in-game checks (does it load, are the wheels on the ground, can the AI drive it) live in `harness/`.
