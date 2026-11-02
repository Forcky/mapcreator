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
| `1WALL_RAIL_L/R###` (invisible, `isRenderable` 0) and `1WALL_BUILDINGS` | solid barrier (no KEY: any physics mesh named WALL) |
| `RAIL_*`, `TREES_*`, `BUILDING_ROOF_*` | visual only (no leading digit) |

- Each physics mesh name matches exactly one KEY. There is deliberately no `WALL` KEY, because defining one is reported to make walls driveable.
- **Logic dummies:**
  - `AC_AB_START_L/R` and `AC_AB_FINISH_L/R` sit 0.5 m outside each road edge, 50 m from each end. L is left of the driving direction.
  - `AC_PIT_0` and `AC_START_0` sit 8 m before the start gate; `AC_HOTLAP_START_0` sits 30 m before it.
  - Spawns are 0.3 m above the road and face the driving direction. **To confirm in game:** height and facing.
- **Pits:** `pitboxes` in `ui_track.json` must equal the number of `AC_PIT_n` dummies (1).

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
- **Placement:** points every 1 m along the centreline, 0.4 m above the road. Points on the surface have crashed the game in another generator.
- **Speed:** grip-limited, `v = sqrt(μ g R)` with μ = 1.05, capped at 130 km/h. It then gets a forward pass for acceleration and a backward pass for braking, both adjusted for grade. **To confirm in game:** that AI cars finish.

## Validation

`mapcreator validate` reads every file back and checks the rules above, including measured heights of spawns and AI points against the actual road triangles. The in-game checks (does it load, are the wheels on the ground, can the AI drive it) live in `harness/`.
