# Known issues and open questions

Found in a code review in November 2026 and not fixed yet, or only answerable in the game. Each says what it affects, so it can be weighed against other work.

## Service

- **The worker API isn't versioned.** A newer worker copes with an older web app (it drops build stages from its heartbeats when refused), but deploy both from the same commit.

## Only the game can answer

These need the in-game harness on a machine with Assetto Corsa (`docs/dev-handover.md`):

- triangle winding and the front-face convention as AC renders them (the code is self-consistent and matches the KN5 references, but nothing has been drawn in game yet)
- what `map.ini`'s `SCALE_FACTOR` means to the game's track map
- what the AI line's `direction` field does
- the Tree FX model files: written to the converter's layout and read back by our own parser, but not yet loaded by CSP. Still unknown:
  - the meaning of the trailer's `4`;
  - how CSP treats a list position's height when surface alignment is on;
  - whether `size` scales exactly as assumed (model height × size).
- whether the AI drives the generated `fast_lane.ai` sensibly through hairpins
- in the harness (`harness/`), checked against Content Manager's source and the CSP SDK but never run:
  - whether the probe app loads with its manifest (it now declares a small window, as every CSP app found does)
  - whether `ac.worldCoordinateToTrack` exists (the probe copes if it doesn't, but then reports no lateral offset)
  - whether `sim.isInMainMenu` is the right signal to press Drive
  - the CSP log's file name (`custom_shaders_patch.log`) and what race.ini's `[AUTOSPAWN]` does
