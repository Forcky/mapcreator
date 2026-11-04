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
- whether the tree trunk colliders (#15) stop a car at the bark: their radius assumes Tree FX scales a model by `size` all round and by `width` horizontally on top. And what 200-560 colliders per km cost in frame rate.
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

Answered at M4 (AI opponents; Grasstree Hill, both layouts, and Brinktop Road; 8 Abarth 500s at AI level 100, the player's car driven by AC's AI; same PC and CSP):

- **Grid:** all 8 cars spawn on the road with every wheel loaded (2.1–3.6 kN), facing along it (within 1.2°), in pairs 4 m apart and rows 8 m apart, none overlapping. Checked with `harness/ac_harness.py race` and `watch-report`.
- **Race:** the start is clean every time.
  - Forward: 6 of 8 finished. Two cars side by side ran wide on a fast bend at 1.5 km (radius about 185 m) and left the 6.4 m road. They didn't touch each other; one picked up light damage off the road.
  - Reverse: all 8 finished within 6 s of each other.
  - Brinktop Road: all 8 finished. Two left the road briefly mid-race and rejoined.
- **Run-off:** every car that reached the end came to rest on the road, 56–78 m before the end of the line, with no tyre off.
  - Grasstree Hill: 6 cars forward and 8 reverse, with no damage. Up to four stood there at once, the closest 2 m apart side by side.
  - Brinktop Road: 6 cars stood there at once, 3–6 m apart. One crept into the one ahead at about 1 km/h, with slight damage to both (1.3), so not quite no contact.
  - Why they creep: the cars behind the first one stop short of the 0 hint, where the pace is still 0.2, and AC's own car-following holds them there.
  - No pile-up.
- **AC's own behaviour:**
  - A race ends when the last car still running finishes, and AC puts everyone back on the grid. With a close field, as on Grasstree Hill (all within 6 s), that comes before the leaders have stopped. Practice always shows the run-off. There, AC sends a stopped AI car back to its pit box after about 20–25 s.
  - In practice the AI leave their pit boxes at random times. The pit boxes are our grid spots on the road, so in one of the two practice runs (forward) cars bumped the ones still standing ahead. The damage was light, but two of the bumped cars went back to their boxes and stayed there for the rest of the session.
- **Frame rate:** 7 AI opponents cost about 1% on the GTX 1660 Ti at 1440p. On the reverse layout, the median was 63 fps with 8 cars against 64 alone (slowest 10%: 48 against 49), and each kilometre's median stayed within 7%. The worst single samples dipped further: 26 fps against 41.

## Power lines

- **Answered in game (4 Nov, Grasstree Hill, with towers then allowed 700 m out):**
  - The wires sag sensibly and cross the road 22-57 m above it.
  - Towers beyond the 300 m terrain stood in the sky. Supports are now built only on the terrain, and since #34 that reaches 700 m, so Grasstree Hill's two lines are back: 8 towers, the farthest 650 m out.
  - CSP's `[WIND] DYNAMIC_CABLES` redrew the wires about 8 times thicker, so it isn't used.
- **Still to check in game:**
  - towers turned along their line, seen from near;
  - roadside poles on a mainland road with mapped poles (e.g. Stratford-Maffra Road, Victoria);
  - a car stopping at a pole;
  - frame rate on a road with many poles.
- **Tasmania has transmission lines only:** TasNetworks' distribution poles are restricted to emergency services, and OpenStreetMap maps none along Grasstree Hill Road, so roadside poles there are missing.
- **Inferred supports:** an OSM line drawn with vertices further apart than a span can be (250 m distribution, 650 m transmission) gets evenly spaced supports along it, at a typical span, except between two mapped supports (a long real span). The line is real; those supports' exact positions aren't.
