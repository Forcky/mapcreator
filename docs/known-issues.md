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
- AI opponents, now that each layout has up to 8 grid spots: do they spawn at them and race to the finish, on both layouts?
- whether AI cars now stop in the run-off after the finish. The setup: 250 m of road after the finish, the AI line ending 40 m before the end wall, and the pace ramping 0.5, 0.2, 0.05. A single half-pace hint still let the AI hit the end wall at about 45 km/h (3 Nov).

Answered at milestone M3 (Grasstree Hill, both layouts; CSP 0.2.11, Content Manager's default starter):

- **Rendering:** triangle winding and the front-face convention are right. The road, terrain, verges, guard rails, houses and water all draw correctly.
- **Spawn:** the car spawns on the road with all four wheels loaded, facing along it.
- **Without CSP:** not supported, by the maintainer's decision (#28). Plain AC wouldn't load multi-layout tracks (it wants a `<track>.kn5`), built the AI grid on every load (4 minutes for 12.8 km), and its AI crashed on our open AI line. The website, the track description and ATTRIBUTION.txt say CSP is required.
- **Tree FX:** the models load, and trees stand on the ground with surface alignment on. A comment line in `trees.txt` made CSP load none (fixed: the list has no comments now).
- **Frame rate:** at about 2,500 trees per km, Tree FX costs about 60% of the frame rate on a GTX 1660 Ti at 1440p, mostly from the trees near the road. Canopy-model builds now keep every real crown (#9), with no distance bands or cap, so their frame rate is still to be measured. WorldCover builds keep the bands and the cap.
- **AI:** AC's AI drove the forward layout from the spawn to the finish. It ignores the 130 km/h cap in the line's speed data. The line ends 50 m past the finish gate, and the AI turns round there and leaves the road.
- **Timing:** the A→B timer starts and stops at the gates. `race_out.json` records the run as a lap with its time. Laps the AI drove for the player are recorded as time −1 with a cut.
- **Harness:** the probe app loads with its manifest, `ac.worldCoordinateToTrack` and `sim.isInMainMenu` work, and the CSP log is `custom_shaders_patch.log`.

## Power lines

- **To check in game:**
  - how towers and wires look from the road;
  - towers standing beyond the 300 m terrain mesh (up to 700 m out, so a crossing line keeps both its towers): their bases may show against the sky;
  - whether `[WIND] DYNAMIC_CABLES` sways the wires;
  - frame rate (Grasstree Hill: 6 towers and 1.9 km of line, about 11,000 vertices);
  - a car stopping at a roadside pole.
- **Tasmania has transmission lines only:** TasNetworks' distribution poles are restricted to emergency services, and OpenStreetMap maps none along Grasstree Hill Road, so roadside poles there are missing.
- **Inferred supports:** an OSM line drawn with vertices further apart than a span can be gets evenly spaced supports along it, at a typical span. The line is real; those supports' exact positions aren't.
