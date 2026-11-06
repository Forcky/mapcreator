"""How far the world reaches either side of the road, from the centreline: one set, so the stages agree.

- **Scenery** (`SCENERY_M`): trees and buildings stand within it.
- **Terrain** (`TERRAIN_M`): the ground mesh reaches `TERRAIN_MARGIN_M` further, and the world ends there. So no
  tree or house at the scenery's edge stands at the edge of the world.
- **Power** (`POWER_M`): power-line supports stand anywhere on the terrain, their whole base within
  `POWER_MARGIN_M` of its edge. Towers are tall, few and cheap to draw, so they reach past the trees (the
  maintainer's decision, 4 Nov 2026).
- **DEM** (`DEM_M`): elevation tiles are read `DEM_MARGIN_M` beyond the terrain, so every terrain point is sampled
  inside them.

With 600 m of scenery, the terrain stays at 700 m, power at 695 m and the DEM at 800 m, where #34 put them.
"""

from __future__ import annotations

SCENERY_M = 600.0
TERRAIN_MARGIN_M = 100.0
POWER_MARGIN_M = 5.0
DEM_MARGIN_M = 100.0
TERRAIN_M = SCENERY_M + TERRAIN_MARGIN_M
POWER_M = TERRAIN_M - POWER_MARGIN_M
DEM_M = TERRAIN_M + DEM_MARGIN_M
