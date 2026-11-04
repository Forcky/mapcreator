# Known issues and open questions

Found in a code review in November 2026 and not fixed yet, or only answerable in the game. Each says what it affects, so it can be weighed against other work.

## Service

- **The worker API isn't versioned.** A newer worker copes with an older web app (it drops build stages from its heartbeats when refused), but deploy both from the same commit.

## Only the game can answer

These need the in-game harness on a machine with Assetto Corsa (`docs/dev-handover.md`):

- what `map.ini`'s `SCALE_FACTOR` means to the game's track map
- what the AI line's `direction` field does
- whether the AI drives the generated `fast_lane.ai` sensibly through hairpins (Grasstree Hill has none)
- in the Tree FX model files: the meaning of the trailer's `4`, and whether `size` scales exactly as assumed (model height × size)
- CSP rewrites each tree `.bin` the first time it loads it: the same entries, re-packed in another order and about 9 KB larger. Harmless so far, but it suggests our zip differs from the layout CSP's own writer uses.
- what race.ini's `[AUTOSPAWN]` does
- AI opponents on a road too narrow for pairs, where the grid goes single file: none of the example roads is that narrow

Answered at milestone M3 (Grasstree Hill, both layouts; CSP 0.2.11, Content Manager's default starter):

- **Rendering:** triangle winding and the front-face convention are right. The road, terrain, verges, guard rails, houses and water all draw correctly.
- **Spawn:** the car spawns on the road with all four wheels loaded, facing along it.
- **Without CSP:** not supported, by the maintainer's decision (#28). Plain AC wouldn't load multi-layout tracks (it wants a `<track>.kn5`), built the AI grid on every load (4 minutes for 12.8 km), and its AI crashed on our open AI line. The website, the track description and ATTRIBUTION.txt say CSP is required.
- **Tree FX:** the models load, and trees stand on the ground with surface alignment on. A comment line in `trees.txt` made CSP load none (fixed: the list has no comments now).
- **Frame rate:** at about 2,500 trees per km, Tree FX costs about 60% of the frame rate on a GTX 1660 Ti at 1440p, mostly from the trees near the road. Canopy-model builds keep every real crown (#9), with no distance bands or cap, and are faster: Grasstree Hill 90 / 63 fps (median / slowest) against 68 / 45 for the capped WorldCover build, because the near-road trees halve. WorldCover builds keep the bands and the cap.
- **AI:** AC's AI drives both layouts from the spawn to the finish. It ignores the 130 km/h cap in the line's speed data. After the finish it eases off and comes to rest on the road about 60 m before the end of its line, about 100 m short of the end wall (#30).
  - The last hint is 0. A hint scales the speed AC allows for the line's curvature, so at 0.05 the AI crawled off the end of its line, and on a straight run-off it barely slowed at all (`docs/ac-format.md`).
  - At M3 the line ended 50 m past the finish gate, and the AI turned round there and left the road.
- **Timing:** the A→B timer starts and stops at the gates. `race_out.json` records the run as a lap with its time. Laps the AI drove for the player are recorded as time −1 with a cut.
- **Harness:** the probe app loads with its manifest, `ac.worldCoordinateToTrack` and `sim.isInMainMenu` work, and the CSP log is `custom_shaders_patch.log`.

Answered at M4 (AI opponents; Grasstree Hill, both layouts, 8 Abarth 500s at AI level 100, the player's car driven by AC's AI; same PC and CSP):

- **Grid:** all 8 cars spawn on the road with every wheel loaded (2.1–3.6 kN), facing along it (within 1.2°), in pairs 4 m apart and rows 8 m apart, none overlapping. Checked with `harness/ac_harness.py race` and `watch-report`.
- **Race:** the start is clean in both layouts.
  - Reverse: all 8 finished within 6 s of each other.
  - Forward: 6 of 8 finished. Two cars side by side ran wide on a fast bend at 1.5 km (radius about 185 m) and left the 6.4 m road. They didn't touch each other; one picked up light damage off the road.
- **Run-off:** every car that reached the end came to rest on the road, 56–74 m before the end of the line, with no tyre off and no damage (6 cars forward, 8 reverse). Up to four stood there at once, the closest 2 m apart side by side. No pile-up.
- **AC's own behaviour:**
  - A race ends when the last car still running finishes, and AC puts everyone back on the grid. To watch the run-off, use practice. There, AC sends a stopped AI car back to its pit box after about 20–25 s.
  - In practice the AI leave their pit boxes at random times. The pit boxes are our grid spots on the road, so in one of the two practice runs (forward) cars bumped the ones still standing ahead. The damage was light, but two of the bumped cars went back to their boxes and stayed there for the rest of the session.
- **Frame rate:** 7 AI opponents cost about 1% on the GTX 1660 Ti at 1440p. On the reverse layout, the median was 63 fps with 8 cars against 64 alone (slowest 10%: 48 against 49), and each kilometre's median stayed within 7%. The worst single samples dipped further: 26 fps against 41.
