# In-game test harness (development only, Windows)

Assetto Corsa's licence covers personal use, so the game never runs as part of a public build. This harness is for the developer's own PC.

It goes through **Content Manager (CM)**, which most players use to install and launch AC. Set `ac_root` and `cm_path` in `config.local.toml`.

**Status:** written but **untested**. It gets its first run at milestone M3.

## First drive checklist (M3)

1. Build and validate: `mapcreator build examples/grasstree_hill.json`.
2. **Look first:** `python harness/ac_harness.py showroom out/grasstree_hill/05_track/grasstree_hill/grasstree_hill_forward.kn5` opens the model in CM's Custom Showroom.
   - Check for the road, green verges, terrain, tree cards, houses and guard rails.
   - There should be no holes, spikes or pink or black textures.
3. **Install** one of two ways:
   - `python harness/ac_harness.py install out/grasstree_hill/05_track/grasstree_hill` copies the folder and clears AC's AI cache.
   - `python harness/ac_harness.py install-cm out/grasstree_hill/05_track/grasstree_hill.zip` makes CM ask to install it, the way users will.

   CM should list "Grasstree Hill Road (north-east)" and "(south-west)". The north-east layout should be the default, with preview and outline and no red warnings.
4. **Drive:** `python harness/ac_harness.py launch grasstree_hill --layout forward`, or pick it in CM's Quick Drive. Press Drive in the pits menu.
5. **While on track:** `python harness/ac_harness.py probe` should show:
   - status 2 (live)
   - all four wheel loads above 100 N
   - the car's position
6. **Drive to the finish.** Watch for:
   - Does the car spawn facing up the road?
   - Does the A→B timer start and stop at the gates?
   - Do guard rails and houses stop the car?
   - Any gaps, bumps or seams? Is the camber believable?
   - With CSP: is there Grass FX grass on the verges and none on the road?
7. **After the session:** `python harness/ac_harness.py results` lists the times. It's still unconfirmed whether A→B runs appear as laps.
8. **After quitting AC:** `python harness/ac_harness.py check-log grasstree_hill` checks `log.txt` and the CSP log.
9. Repeat for `--layout reverse`, then add an AI opponent in Quick Drive and check it reaches the finish (M4).

If the track doesn't load at all:

1. Load road-to-assetto's demo track as a control. If that also fails, the problem is the environment, not our writer.
2. Then compare the two KN5 files with `mapcreator.core.kn5.read`.

## What to record

Write these into the private deployment notes:

- the spawn facing (+Z or −Z) and height
- whether the A→B timing shows in `race_out.json`
- whether the AI completes the run
- frame rate on the GTX 1660 Ti, with and without CSP
- every log error

## Automated probe (CSP Lua app, untested)

`csp_app/mc_probe/` is a Custom Shaders Patch Lua app. It presses Drive, teleports the car along the track and records, at each point:
- wheel contact, and whether the surface is valid track
- a downward raycast, which finds holes in the collision mesh
- the lateral offset on the AI spline

To run it:

```powershell
python harness/ac_harness.py probe-install                     # once: copy the app into AC's apps/lua
python harness/ac_harness.py probe-points out/grasstree_hill/05_track/grasstree_hill --layout forward --every 200 --quit --screenshots-every 10
python harness/ac_harness.py launch grasstree_hill --layout forward
# ...the app drives itself through every point, writes results.json and quits AC
python harness/ac_harness.py probe-report                      # PASS or a list of problem points
```

- Results and screenshots land in `Documents/Assetto Corsa/mapcreator_probe/`.
- A teleport the game ignores is reported as a harness problem, not a track problem.
- Teleporting is only known to work in offline practice.
