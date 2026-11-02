# In-game test harness (development only, Windows)

Assetto Corsa's licence covers personal use, so the game never runs as part of a public build. This harness is for the developer's own PC.

**Status:** written but **untested**. It gets its first run at milestone M3.

## First drive checklist (M3)

1. Build and validate: `mapcreator build examples/grasstree_hill.json` (this runs every stage, `validate` included).
2. Install: `python harness/ac_harness.py install out/grasstree_hill/05_track/grasstree_hill`.
3. In Content Manager, pick the track and the **forward** layout, a road car such as the Abarth 500, and a practice session.
   - Or run `python harness/ac_harness.py prepare grasstree_hill --layout forward` and start AC from the launcher.
4. While on track: `python harness/ac_harness.py probe` should report all four wheel loads above 100 N at the spawn.
5. Drive to the finish. Things to watch for:
   - does the car spawn facing up the road?
   - does the timer start at the first gate and stop at the second?
   - any gaps, bumps or seams?
   - is the camber believable?
6. Quit AC, then run `python harness/ac_harness.py check-log grasstree_hill`.
7. Repeat for the **reverse** layout.
8. Add an AI opponent and check it drives to the finish (M4).
9. Run `python harness/ac_harness.py restore` if you used `prepare`.

If the track doesn't load at all:

1. Load road-to-assetto's demo track as a control. If that fails too, the problem is the environment, not our writer.
2. Then compare the two KN5 files with `mapcreator.core.kn5.read`.

## What to record

Write these into the private deployment notes:

- the spawn facing (+Z or −Z) and height
- whether the AI completes the run
- frame rate on the GTX 1660 Ti
- every `log.txt` error
