# In-game test harness (development only, Windows)

Assetto Corsa's licence covers personal use, so the game never runs as part of a public build. This harness is for the developer's own PC.

It goes through **Content Manager (CM)**, which most players use to install and launch AC. Set `ac_root` and `cm_path` in `config.local.toml`.

**Status:** first run at milestone M3, with CSP 0.2.11 and Content Manager's default starter. The launch, the shared-memory probe, the log check and the automated probe all worked.

## First drive checklist (M3)

1. Build and validate: `.venv\Scripts\mapcreator build examples/grasstree_hill.json`. Run everything here from the repository root with the venv's own `python` (or activate the venv first): a bare `python` is often the Microsoft Store stub, without numpy or scipy.
2. **Look first:** `.venv\Scripts\python harness/ac_harness.py showroom out/grasstree_hill/05_track/grasstree_hill/grasstree_hill_forward.kn5` opens the model in CM's Custom Showroom.
   - Check for the road, green verges, terrain, trees (with Custom Shaders Patch), houses and guard rails.
   - There should be no holes, spikes or pink or black textures.
3. **Install** one of two ways:
   - `.venv\Scripts\python harness/ac_harness.py install out/grasstree_hill/05_track/grasstree_hill` copies the folder and clears AC's AI cache.
   - `.venv\Scripts\python harness/ac_harness.py install-cm out/grasstree_hill/05_track/grasstree_hill.zip` makes CM ask to install it, the way users will.

   CM should list "Grasstree Hill Road (north-east)" and "(south-west)". The north-east layout should be the default, with preview and outline and no red warnings.
4. **Drive:** `.venv\Scripts\python harness/ac_harness.py launch grasstree_hill --layout forward`, or pick it in CM's Quick Drive. Press Drive in the pits menu.
   - AC opens a UDP telemetry port, so Windows Firewall may ask about `acs.exe` on the first launch. Answer it before scripting any input: the prompt keeps the keyboard and mouse focus, and synthetic clicks never reach AC.
   - With no wheel to hand, press Ctrl+C in the session: AC's AI then drives your car along the AI line to the finish. Ctrl+G toggles the automatic gearbox. Content Manager rewrites `assists.ini` on every launch, so set assists in CM or in the session, not in the file.
5. **While on track:** `.venv\Scripts\python harness/ac_harness.py probe` should show:
   - status 2 (live)
   - all four wheel loads above 100 N
   - the car's position
6. **Drive to the finish.** Watch for:
   - Does the car spawn facing up the road?
   - Does the A→B timer start and stop at the gates?
   - Do guard rails and houses stop the car?
   - Any gaps, bumps or seams? Is the camber believable?
   - With CSP: is there Grass FX grass on the verges and none on the road?
7. **After the session:** `.venv\Scripts\python harness/ac_harness.py results` lists the times. An A→B run appears as one lap with its time. A lap AC's AI drove for you (Ctrl+C in a session) is recorded as time −1 with a cut.
8. **After quitting AC:** `.venv\Scripts\python harness/ac_harness.py check-log grasstree_hill` checks `log.txt` and the CSP log.
9. Repeat for `--layout reverse`, then race AI opponents (below).

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

## Automated probe (CSP Lua app)

`csp_app/mc_probe/` is a Custom Shaders Patch Lua app. It presses Drive, teleports the car along the track and records, at each point:
- wheel contact, and whether the surface is valid track
- a downward raycast, which finds holes in the collision mesh
- the lateral offset on the AI spline

To run it:

```powershell
.venv\Scripts\python harness/ac_harness.py probe-install                     # once: copy the app into AC's apps/lua
.venv\Scripts\python harness/ac_harness.py probe-points out/grasstree_hill/05_track/grasstree_hill --layout forward --every 200 --quit --screenshots-every 10
.venv\Scripts\python harness/ac_harness.py launch grasstree_hill --layout forward
# ...the app drives itself through every point, writes results.json and quits AC
.venv\Scripts\python harness/ac_harness.py probe-report                      # PASS or a list of problem points
```

- Results and screenshots land in `Documents/Assetto Corsa/mapcreator_probe/`.
- A teleport the game ignores is reported as a harness problem, not a track problem.
- Teleporting is only known to work in offline practice.

## AI opponents (CSP Lua app)

`csp_app/mc_watch/` logs every car in the session: its position, heading, speed, wheels, laps and damage, and the frame rate. It samples 4 times a second for the first 15 s, then once a second. It runs only when the harness switches it on, for one session.

```powershell
.venv\Scripts\python harness/ac_harness.py race grasstree_hill --layout reverse --cars 8 --session race   # or --session practice
# press Drive, then Ctrl+C to have AC's AI drive your car too
.venv\Scripts\python harness/ac_harness.py watch-report "<AC>/content/tracks/grasstree_hill" --layout reverse
```

The report checks the grid, which fails if any car spawned with a wheel unloaded or off the road, facing more than 10° off the road, or outside it. It then gives, for each car:
- the furthest point it reached, and when it finished
- where it left the road, if it did
- where it came to rest near the end of the line

- **A race won't show the run-off.** It ends when the last car still running finishes, and AC then puts every car back on the grid. Use `--session practice` to watch the cars stop. In practice, AC sends a stopped AI car back to its pit box after about 20–25 s.
- **In practice the AI leave their pit boxes at random times.** The pit boxes are the grid spots on the road, so cars can bump the ones still standing ahead.
- **Press Drive with real mouse input.** AC's menus follow raw mouse input, so a scripted click needs absolute mouse moves (`mouse_event` with `MOUSEEVENTF_ABSOLUTE`), not just `SetCursorPos`.
