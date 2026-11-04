# Developer handover: first drive on a machine with Assetto Corsa

Read this when picking up development on a Windows machine with Assetto Corsa, Content Manager and Custom Shaders Patch installed.

Everything that can be built and checked without the game is done. The next milestone (M3) is the first in-game load and drive. The private deployment repo has the full plan, the research notes and the progress log. Record findings there, not here.

## Where things stand

| Stage | Command | What it does | State |
|---|---|---|---|
| route | `mapcreator route job.json` | Snaps waypoints to OSM; writes `01_route/` | Done |
| profile | `mapcreator profile job.json` | Smooth centreline + DEM profile, bridges filled; `02_profile/profile.png` | Done |
| section | `mapcreator section job.json` | Width from OSM tags, camber from the DEM | Done |
| mesh | `mapcreator mesh job.json` | Road, verges, skirts, terrain; `04_mesh/preview.obj` with `MAPCREATOR_PREVIEW_OBJ=1` (or `preview_obj = true` in `config.local.toml`) | Done |
| scenery | `mapcreator scenery job.json` | Trees (canopy model or WorldCover) and buildings (LIST in Tasmania; OSM elsewhere, with an OSM index); solid house walls and trunk colliders near the road | Done |
| track | `mapcreator track job.json` | KN5s, AI lines, layouts, data and UI files, zip | Done, **never loaded in game** |
| validate | `mapcreator validate job.json` | 73 static checks | Done |
| `harness/` | `python harness/ac_harness.py …` | CM showroom, install and launch; log, results and shared-memory checks | Written, **untested** |

`mapcreator build job.json` runs every stage. The reference job is `examples/grasstree_hill.json`: Grasstree Hill Road (C324), Tasmania, 12.8 km, two layouts.

Changes since the code was last run by hand that M3 should look out for (all checked by tests, none yet in game):

- OSM and map coordinates now get the GDA2020-to-GDA94 datum shift, so the road sits about 1.5 m from where it did, onto the DEM and theLIST buildings.
- Verge skirts stop halfway to a parallel leg of the road on switchbacks.
- Gable-end walls of houses face outward; before, they were see-through from outside.
- Roads under 5 m wide now pass validation; the timing gates still sit 0.5 m past each edge.
- A build has an hour's time limit in the hosted worker, and the CLI is unchanged.

## Set up

```powershell
git clone <repo url> mapcreator
cd mapcreator
# main has everything; the feature branches are history
py -3 -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
```

Then:

1. Copy the private marker list from the deployment repo to `.private-markers` in this folder. It's gitignored, and the privacy test uses it.
2. Create `config.local.toml` from `config.local.toml.example`. Set `ac_root` (the folder that contains `acs.exe`) and `cm_path` (`Content Manager.exe`).
3. Run `.venv\Scripts\python -m pytest -q`. Every test should pass.
4. Run `.venv\Scripts\mapcreator build examples\grasstree_hill.json`. The first run downloads OSM data from Overpass (about 45 s; nothing with an OSM index, below) and DEM tiles (about 30 MB). Later runs take about 30 s.

The output is `out\grasstree_hill\05_track\grasstree_hill\` and `grasstree_hill.zip`.

## OSM data: build the local index once

Public Overpass is fine for a handful of development builds. For anything more, such as running the hosted service, a batch of builds, or buildings outside Tasmania, read OSM from a local index built once from Geofabrik's Australia extract (ODbL; see `docs/data-sources.md`):

```powershell
.venv\Scripts\python -m pip install -e .[osm]      # pyosmium: needed to build the index, not to use it
mkdir data\osm                                      # data\ is gitignored
# Download https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf (about 0.9 GB) into data\osm\
.venv\Scripts\python -m mapcreator.providers.osm.index build data\osm\australia-latest.osm.pbf data\osm\australia.sqlite
```

Then set `osm_index` in `config.local.toml` (or `MAPCREATOR_OSM_INDEX`) to the index's full path. The route stage, the dev route previews and the national fallback's buildings all read it, and Overpass is no longer asked.

- **Check it:** `.venv\Scripts\python -m mapcreator.providers.osm.index query data\osm\australia.sqlite 147.30 -42.82 147.34 -42.79` prints the source file, the OSM data timestamp, and how many roads and buildings are in the box.
- **Memory:** node locations are kept in memory while building. On a small machine, add `--locations sparse_file_array,data\osm\nodes.cache` to keep them in a file (delete it afterwards).
- **Refreshing:** download a newer extract and build again. The new index is written as `<name>.partial` and replaces the old one only when it's complete.
- **Not yet measured:** build time, peak memory and index size for the full Australia extract. On synthetic data the builder handles about 13,000 ways a second on a laptop, so expect it to take tens of minutes.
- **Windows:** a Python process that had called `osmium.apply()` could take many minutes to exit, or not exit at all. The builder and the tests don't use it; don't use it in scripts either.

## First drive (M3)

Follow `harness/README.md`. It goes through Content Manager, which is what players use. In short:

1. `showroom`: look at the KN5 in CM's Custom Showroom before driving.
2. `install-cm`: install the zip through CM, the same way users will.
3. `launch`: start practice through `acmanager://race/config`, then press Drive.
4. While driving, `probe`: all four wheels should be loaded at the spawn.
5. Afterwards, `results` and `check-log`.
6. Repeat for the reverse layout, then add an AI car (M4).

If the track won't load:

1. Load an open-source generator's demo track (road-to-assetto) as a control.
2. Compare the two KN5 files with `mapcreator.core.kn5.read`.

## Example roads and what each one tests in game

| Job | Road | What to look at |
|---|---|---|
| `examples/grasstree_hill.json` | Grasstree Hill Road (C324), 12.8 km, both directions, all LiDAR | the reference: spawns, timing gates, AI, guard rails, houses |
| `examples/brinktop_road.json` | Brinktop Road from Richmond, 10.4 km | the 108° turn from Bridge Street into Wellington Street, 0.4 km in, rounded to an 8 m corner: drivable, AI slows to about 31 km/h |
| `examples/esperance_coast_road.json` | Esperance coast road, 20 km | water: the sea beside the road (CSP's water shader with CSP, a glossy flat surface without); 43,000 3D trees (Tree FX) |
| `examples/lyell_highway_mt_arrowsmith.json` | Lyell Highway from Bradshaw Bridge, 63 km, both directions | the longest allowed road: load time, frame rate and memory with two 97 MB KN5s and 157,000 trees; grades stay at or under 15% on the non-LiDAR stretches |

Trees are 3D models drawn by Custom Shaders Patch (Tree FX, CSP 0.1.79+); without CSP a track has none.

## What Content Manager and CSP expect, and what we already do

Taken from reading CM's source and CSP's docs (Nov 2026). None of it has been tried in game yet.

- **Zip:** `<track_id>/` at the zip root.
- **Layouts:** each layout needs both `models_<layout>.ini` and `ui/<layout>/ui_track.json`, or CM silently drops it. There's no root `ui/ui_track.json`. The forward layout carries `"priority": 1`, so it's the default.
- **`ui_track.json`:**
  - `geotags` must be two strings.
  - `length` must carry a unit ("12.80 km"); CM ignores a bare number.
  - CM also warns when the preview, outline or map is missing. We ship all of them.
- **`extension/ext_config.ini`:**
  - CSP's Grass FX grows grass on our grass and terrain materials, with the road occluding it. On a one-lane gravel road it also grows on the grass strip.
  - Rain FX puddles and soaks the road.
  - On a track with gravel, `[TYRES_FX]` sets the dust and tyre-dirt colours from our textures.
  - Vanilla AC ignores it.
- **`surfaces.ini`:** GRASS carries `_EXT_SURFACE_TYPE=GRASS` for CSP. A track with unsealed road adds GRAVEL (`1GRAVEL_###` meshes) with `_EXT_SURFACE_TYPE=GRAVEL`; see `docs/ac-format.md`.
- **Public install link:** `acmanager://install?url=<encoded zip URL>`. CM always asks the user to confirm.
- **Download endpoint:**
  - `application/zip`
  - a quoted `filename="<id>.zip"`
  - `Accept-Ranges`
  - never an HTML page in between

## Questions only the game can answer

These are written down as defaults in the code. Change them in one place if the game disagrees.

| Question | Default | Where |
|---|---|---|
| Do spawned cars face the driving direction (+Z)? | +Z forward | `core/kn5.py` `dummy_matrix`, `core/layout.py` |
| Spawn height above the road | 0.3 m | `core/layout.py` `SPAWN_HEIGHT_M` |
| Does the AI drive the computed line? | computed, 0.4 m above the road | `core/ai_line.py` |
| Is AC's z = −north (rotation, not mirror)? | yes | `core/scene.py` `Z_SIGN` |
| Is the DEM camber believable? (14% of C324 reads above 7%, capped at 7%) | capped at 7% | `core/section.py` |
| Frame rate with a ~570k-triangle KN5 (terrain 565k + 251 buildings) plus ~36k Tree FX trees on a GTX 1660 Ti | unknown | `core/scenery.py` `TREE_BANDS` and `MAX_TREES_PER_KM` thin made-up (WorldCover) trees; canopy-model builds keep every crown (#9), so they're the dense case to measure |
| Does water show as water: CSP's shader with CSP, a glossy flat surface without? | it should | `core/water.py`, `core/package.py` `ext_config_ini` |
| Do the Tree FX 3D trees load (CSP 0.1.79+), stand on the ground and face the light the right way? | they should | `core/treefx.py` (winding, `size`) |
| Do guard rails stop the car? (invisible `1WALL_RAIL_*` colliders, 5.4 km on C324) | they should | `core/roadside.py` |
| Do houses stop the car? (`1WALL_BUILDINGS`) | they should | `core/scenery.py` |
| Do tree trunks stop the car at the bark? (invisible `1WALL_TRUNK_*` prisms, trunks within 20 m of the road edge) | they should | `core/scenery.py` `trunk_colliders` |
| Do both layouts show in Content Manager with names and previews? | they should | `pipeline/track_stage.py` |

## Rules that still apply

- **Nothing private in this repo:** no hostnames, IPs, personal paths or account details. `tests/test_private_markers.py` enforces this.
- **The game and its tools are for testing only.** AC, Content Manager and the AC SDK never run as part of a public build, because the game's licence is personal use only.
- **No Google, Esri, Bing or Mapbox content.** See `docs/data-sources.md`.
- **One commit per reviewable step.** Run the tests before each commit.

## After M3 and M4

- **M5 scenery:** trees and buildings from open data, plus CC0 textures.
- **M6 build worker:** a hardened container.
- **M7 web front end and queue.**

`docs/architecture.md` has the shape. Work on these may already be on other branches; check `git branch -a`.
