# Known issues and open questions

Found in a code review in November 2026 and not fixed yet, or only answerable in the game. Each says what it affects, so it can be weighed against other work.

## Geometry

- **Guard-rail visual meshes aren't split.** One continuous rail run longer than about 13 km would exceed the 65,535-vertex limit and fail the build. Real runs are far shorter.

## Service

- **Turnstile tokens aren't checked for hostname or action.** Verification requires `success`, and fails closed on errors and timeouts. A token minted for another site using the same widget's site key would pass, which needs the secret, so it's of little use to anyone.
- **The worker API isn't versioned.** A newer worker copes with an older web app (it drops build stages from its heartbeats when refused), but deploy both from the same commit.

## Only the game can answer

These need the in-game harness on a machine with Assetto Corsa (`docs/dev-handover.md`):

- triangle winding and the front-face convention as AC renders them (the code is self-consistent and matches the KN5 references, but nothing has been drawn in game yet)
- what `map.ini`'s `SCALE_FACTOR` means to the game's track map
- what the AI line's `direction` field does
- whether the AI drives the generated `fast_lane.ai` sensibly through hairpins
