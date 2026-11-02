# Developer handover: first drive on a machine with Assetto Corsa

Read this when picking up development on a Windows machine with Assetto Corsa, Content Manager and Custom Shaders Patch installed.

Everything that can be built and checked without the game is done. The next milestone (M3) is the first in-game load and drive. The private deployment repo has the full plan, the research notes and the progress log. Record findings there, not here.

## Where things stand

| Stage | Command | What it does | State |
|---|---|---|---|
| route | `mapcreator route job.json` | Snaps waypoints to OSM; writes `01_route/` | Done |
| profile | `mapcreator profile job.json` | Smooth centreline + DEM profile, bridges filled; `02_profile/profile.png` | Done |
| section | `mapcreator section job.json` | Width from OSM tags, camber from the DEM | Done |
| mesh | `mapcreator mesh job.json` | Road, verges, skirts, terrain; `04_mesh/preview.obj` | Done |
| scenery | `mapcreator scenery job.json` | Trees (canopy model or WorldCover) and LIST buildings, visual only | Done, branch `scenery` |
| track | `mapcreator track job.json` | KN5s, AI lines, layouts, data and UI files, zip | Done, **never loaded in game** |
| validate | `mapcreator validate job.json` | 70 static checks | Done |
| `harness/` | `python harness/ac_harness.py …` | Install, race.ini, log and physics checks | Written, **untested** |

`mapcreator build job.json` runs every stage. The reference job is `examples/grasstree_hill.json`: Grasstree Hill Road (C324), Tasmania, 12.8 km, two layouts.

## Set up

```powershell
git clone <repo url> mapcreator
cd mapcreator
git checkout pipeline-foundations        # or main, once merged
py -3 -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
```

Then:

1. Copy the private marker list from the deployment repo to `.private-markers` in this folder. It's gitignored, and the privacy test uses it.
2. Create `config.local.toml` from `config.local.toml.example` and set `ac_root` to the folder that contains `acs.exe`.
3. Run `.venv\Scripts\python -m pytest -q`. Every test should pass.
4. Run `.venv\Scripts\mapcreator build examples\grasstree_hill.json`. The first run downloads OSM data (about 45 s) and DEM tiles (about 30 MB). Later runs take about 30 s.

The output is `out\grasstree_hill\05_track\grasstree_hill\` and `grasstree_hill.zip`.

## First drive (M3)

Follow `harness/README.md`. In short:

1. **Look before driving.** In Content Manager, open `grasstree_hill_forward.kn5` in the Custom Showroom. You should see the road, green verges and brown terrain, with no holes or spikes. `out\grasstree_hill\04_mesh\preview.obj` opens in any 3D viewer too.
2. **Install:** run `python harness\ac_harness.py install out\grasstree_hill\05_track\grasstree_hill`, or drag the zip onto Content Manager.
3. **Drive:** in Content Manager, Drive → Quick Drive → Practice, track "Grasstree Hill Road", layout north-east (forward), a road car.
4. **While spawned:** run `python harness\ac_harness.py probe`. All four wheel loads should be above 100 N.
5. **Drive start to finish.** Check that the timer starts and stops at the gates.
6. **After quitting AC:** run `python harness\ac_harness.py check-log grasstree_hill`.
7. **Repeat for the reverse layout,** then add an AI car (M4).

If the track won't load: load an open-source generator's demo track (road-to-assetto) as a control, and compare with `mapcreator.core.kn5.read`.

## Questions only the game can answer

These are written down as defaults in the code. Change them in one place if the game disagrees.

| Question | Default | Where |
|---|---|---|
| Do spawned cars face the driving direction (+Z)? | +Z forward | `core/kn5.py` `dummy_matrix`, `core/layout.py` |
| Spawn height above the road | 0.3 m | `core/layout.py` `SPAWN_HEIGHT_M` |
| Does the AI drive the computed line? | computed, 0.4 m above the road | `core/ai_line.py` |
| Is AC's z = −north (rotation, not mirror)? | yes | `core/scene.py` `Z_SIGN` |
| Is the DEM camber believable? (14% of C324 reads above 7%, capped at 7%) | capped at 7% | `core/section.py` |
| Frame rate with ~860k triangles (terrain 565k + 36k tree cards + 251 buildings) on a GTX 1660 Ti | unknown | `core/scenery.py` `TREE_BANDS` thins trees |
| Do the alpha-tested tree cards look right (ksPerPixelAT, DXT5)? | they should | `core/sprites.py`, `pipeline/track_stage.py` |
| Do guard rails stop the car? (invisible `1WALL_RAIL_*` colliders, 5.4 km on C324) | they should | `core/roadside.py` |
| Do houses stop the car? (`1WALL_BUILDINGS`) | they should | `core/scenery.py` |
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
