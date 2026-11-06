# Known issues and open questions

Found in a code review in November 2026 and not fixed yet, or only answerable in the game. Each says what it affects, so it can be weighed against other work.

## Service

- **The worker API isn't versioned.** A newer worker copes with an older web app (it drops build stages from its heartbeats when refused), but deploy both from the same commit.

## Only the game can answer

These need the in-game harness on a machine with Assetto Corsa (`docs/dev-handover.md`):

- what `map.ini`'s `SCALE_FACTOR` means to the game's track map
- what the AI line's `direction` field does
- whether the AI drives the racing line sensibly through hairpins: it drove Lyell Highway's 15.8 m and 13.9 m hairpins on a centreline line with points 5 m apart, but not yet on the racing line
- in the Tree FX model files: the meaning of the trailer's `4`, and whether `size` scales exactly as assumed (model height × size)
- CSP rewrites each tree `.bin` the first time it loads it: the same entries, re-packed in another order and about 9 KB larger. Harmless so far, but it suggests our zip differs from the layout CSP's own writer uses.
- what race.ini's `[AUTOSPAWN]` does
- whether the tree trunk colliders (#15) stop a car at the bark: their radius assumes Tree FX scales a model by `size` all round and by `width` horizontally on top. And what 200-560 colliders per km cost in frame rate.
- AI opponents on a road too narrow for pairs, where the grid goes single file: Lower Swamp Road (`examples/lower_swamp_road.json`, 4.0 m) is the first example that narrow (see "Gravel and narrow roads" below)
- whether trees typed by TASVEG look right: tall wet eucalypts over tea-tree or rainforest understorey, tea-tree in scrub and buttongrass, myrtle in rainforest, pine in plantations. Some families still borrow other models until batch 2: plantation eucalypt (`euc_young`), she-oak (`wattle`), blackwood (`wattle`, then `myrtle` above 12 m), snow gum (`euc_spread`), banksia (`tea_tree`), the highland conifers (`pine`)
- whether NVIS-typed trees on the mainland look right: one subgroup per 100 m cell, so a stretch of road takes its family from the dominant vegetation around it, and mallee borrows `euc_young` until it has its own multi-stemmed model. The mainland's dry forests are drawn rough-barked where NVIS calls them open forest with a shrubby understorey, and smooth in grassy forests and woodlands: a rule of thumb, not species data
- whether SVTM-typed trees in NSW look right: a tree takes one of the trees its Plant Community Type's name lists, in equal shares, so an ironbark-cypress-buloke forest is a third of each. Names are read for common names ("Snow Gum", "Cypress Pine", "Buloke", "Paperbark", "Mulga"); the eucalypts' bark is a rule of thumb (stringybarks, ironbarks, boxes and bloodwoods rough, gums and apples smooth). A PCT that names none takes its vegetation class's family. Cleared land and plantations are 0 in the SVTM, so plantation pines are typed by the native vegetation nearest them
- whether the bark atlas holds up close: a column goes once around a branch, so a thick trunk's bark is stretched across (about 2.6 times on a 1 m-wide trunk)
- a few canopy-model crowns in paddocks and towns are far taller than any forest beside them (100 m and 65 m on the Huon Highway, where the tallest wet-forest crowns are 57-58 m). They're probably artefacts in the canopy model (a mast, a cliff edge). Crowns outside forest codes are capped at 45 m, so they stand as 45 m trees rather than vanish (`species.height_caps`)
- whether Tree FX draws trees 600 m out, and what they cost in frame rate (see "Scenery reach" below)

Answered at milestone M3 (Grasstree Hill, both layouts; CSP 0.2.11, Content Manager's default starter):

- **Rendering:** triangle winding and the front-face convention are right. The road, terrain, verges, guard rails, houses and water all draw correctly.
- **Spawn:** the car spawns on the road with all four wheels loaded, facing along it.
- **Without CSP:** not supported, by the maintainer's decision (#28). Plain AC wouldn't load multi-layout tracks (it wants a `<track>.kn5`), built the AI grid on every load (4 minutes for 12.8 km), and its AI crashed on our open AI line. The website, the track description and ATTRIBUTION.txt say CSP is required.
- **Tree FX:** the models load, and trees stand on the ground with surface alignment on. A comment line in `trees.txt` made CSP load none (fixed: the list has no comments now).
- **Frame rate:** at about 2,500 trees per km, Tree FX costs about 60% of the frame rate on a GTX 1660 Ti at 1440p, mostly from the trees near the road. Canopy-model builds keep every real crown (#9), with no distance bands or cap, and are faster: Grasstree Hill 90 / 63 fps (median / slowest) against 68 / 45 for the capped WorldCover build, because the near-road trees halve. WorldCover builds keep the bands and the cap, which counts only the trees within 250 m since the scenery reach went to 600 m.
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
  - Grasstree Hill, in practice: 6 cars forward and 8 reverse, with no damage. Up to four stood there at once, the closest 2 m apart side by side.
  - Grasstree Hill, a later 8-car race (forward): six came to rest in the run-off before the reset. Two of them touched (1.3 each), as below.
  - Brinktop Road: 6 cars stood there at once, 3–6 m apart. One crept into the one ahead at about 1 km/h, with slight damage to both (1.3), so not quite no contact.
  - Why they creep: the cars behind the first one stop short of the 0 hint, where the pace is still 0.2, and AC's own car-following holds them there.
  - No pile-up.
- **AC's own behaviour:**
  - A race ends when the last car still running finishes, and AC puts everyone back on the grid. With a close field, as on Grasstree Hill (all within 6 s), that comes before the leaders have stopped. Practice always shows the run-off. There, AC sends a stopped AI car back to its pit box after about 20–25 s.
  - In practice the AI leave their pit boxes at random times. The pit boxes are our grid spots on the road, so in one of the two practice runs (forward) cars bumped the ones still standing ahead. The damage was light, but two of the bumped cars went back to their boxes and stayed there for the rest of the session.
- **Frame rate:** 7 AI opponents cost about 1% on the GTX 1660 Ti at 1440p. On the reverse layout, the median was 63 fps with 8 cars against 64 alone (slowest 10%: 48 against 49), and each kilometre's median stayed within 7%. The worst single samples dipped further: 26 fps against 41.

## Gravel and narrow roads

Nothing about gravel has been loaded in game yet. The `GRAVEL` surface's numbers (`docs/ac-format.md`) are a starting point from community gravel stages, to be tuned by feel.

- **To check in game** (CSP 0.2.11):
  - **Lower Swamp Road** (`examples/lower_swamp_road.json`, both layouts): one lane at 4.0 m, sealed for 1.6 km and then unsealed for 5.2 km.
    - All 5 single-file spots spawn on the road with every wheel loaded, and `pitboxes` is 5. The forward grid stands on the seal and the reverse grid on the gravel.
    - 4 AI opponents start and drive in file.
    - Guide posts and CAMs on the seal only; none on the gravel.
    - The change from seal to gravel at 1.6 km is felt, with no bump.
    - Dust behind the cars, gravel skidmarks, and dirt carried onto the seal. Dust and tyre dirt should match the road (`[TYRES_FX]` colours from our textures).
    - The grass strip down the middle grows Grass FX blades, and only there.
    - `custom_shaders_patch.log` is clean.
    - A Lua probe reads `wheels[i].surfaceSectionIndex` 2 and `surfaceExtendedType` (CSP 0.2.8) on the gravel: that shows `_EXT_SURFACE_TYPE=GRAVEL` takes effect without extended physics.
  - **Hartz Road** (C632, 13.2 km of two-lane gravel at 6.0 m, a 700 m climb): gravel at speed.
  - **Tuning (Q3 of the gravel research):** friction 0.88, damping 0.01, dirt additive 0.8, `gravel.wav`, 1 cm Perlin bumps, on Lower Swamp Road and Hartz Road. With CSP's Surfaces FX **off** first, because it's off by default and offline only, so the plain numbers must feel right on their own; then **on**.
  - **AI on gravel:** whether AC's AI copes with friction 0.88 plus dirty tyres, or overdrives. There are no pace hints on gravel; if it overdrives, try about 0.8 over the unsealed stretches.
  - **CSP's Rally Stage mode** (Quick Drive, CSP 0.2.4) and its Rally Copilot app with our point-to-point layouts, and whether CSP picks street tyres for the AI on a track tagged `rally`. Their requirements aren't documented.
  - The gravel texture's colour (a grey-brown from a CC0 photo; Tasmanian dolerite gravel runs from blue-grey to brown-grey), and whether its 24 m repeat shows.
  - A narrow seal with no centre line at 4.0 m (Allens Rivulet Road, `lanes=1`).
- **Camber on narrow roads is capped, to be tuned in game.** The section stage reads camber from the DEM 0.5 m inside each edge, but never closer than 1 m to the centreline, so on a 4 m road its two samples are 3 m apart and catch the hillside and the cut and fill beside the road.
  - Before: 37% of Lower Swamp Road sat at the 7% limit (55% before the road-strip profile), and 63% of it was one tilted plane with no crown.
  - Now the limit follows the width (4% at or under 5 m, 7% at 6.4 m) and narrow roads keep some crown (`docs/ac-format.md`). Lower Swamp Road: 67% at its 4% limit, none of it one plane, at least 0.5% of crown everywhere. Hartz Road (6.0 m, limit 6.1%): 36% at its limit, against 33% at 7% before; its crown kept fades to 0.14% at that width, so in banked bends it's still all but a plane, as before.
  - **To check in game:** whether 4% feels right on a single-lane road, or whether the DEM's camber should be trusted less still (a probe further from the edges, or no camber at all under 5 m).
- **Widths are guesses.** About 98% of unsealed ways in OSM carry neither `lanes` nor `width`, so the class default (5.0-6.0 m) decides. The defaults follow NATSPEC AUS-SPEC 0052's minimum formations, not measurements. Measuring about 20 unsealed roads in theLIST's CC BY aerial photos (AerialPhoto2020-2024; never the NC-ND imagery) would calibrate them.
- **Not routable:** `highway=track`, so most forestry roads (and Rally Tasmania's stages) can't be built yet. Lower Swamp Road carries on as a track past where its example ends.
- **One surface for all unsealed values.** `dirt`, `earth`, `ground`, `mud` and `sand` drive like `gravel`. A looser `DIRT` surface can follow if testing shows it's worth it.

## Racing line

The AI line runs out-in-out over the whole road (`core/racing_line.py`, 4 Nov). Static checks pass on Grasstree Hill (both layouts) and Brinktop Road. To check in game, against the same track built from main:
- **AI lap time:** the AI's time from gate to gate, both layouts. Its planned run is about 3% quicker on Grasstree Hill (375 s against 386 s), but AC sets its own speed.
- **On the road:** the AI stays on the road through the tight bends in both layouts, and doesn't run wide on the exit where the line runs 1.2 m from the edge. The probe (`harness/ac_harness.py probe-points`) teleports a car onto the line every 200 m and checks all four wheels.
- **8-car races:** the cars now share one line that crosses the whole road. Whether they still race clean, side by side and passing, and get away from the grid in pairs as before.
- **Run-off:** as before (#30), every car should come to rest on the road about 60 m before the end of the line.
- **Spawns:** the grid and the hot-lap spawn are unchanged; the line stays on the centreline over them, and the AI should leave the grid as before.
- **Track length:** `log.txt`'s `TRACK LENGTH` on Grasstree Hill and Lyell Highway against the line's real length (the last cumulative length in `fast_lane.ai`; Grasstree Hill forward: 12,737 m). With points 5 m apart, Lyell should read about 1.3% long, against 32.8% with 1 m points.

## Scenery reach (600 m)

Trees and buildings now stand within 600 m of the road (`core.reach`); they stopped at 250 m. Power-line supports still reach the terrain's edge less 5 m (695 m). None of it has been seen in game yet.

- **To check in game** (CSP 0.2.11, GTX 1660 Ti, 1440p):
  - **Do far trees draw?** Past its last LOD (85-267 m out, by model), Tree FX draws each tree as a billboard. The billboard fades out at a distance that CSP's own code sets and doesn't document (`docs/ac-format.md`). If that's under 600 m, the far trees never show.
  - **What do they cost?** Compare frame rates against a 250 m build, at the same probe points, both along the road and looking out to the side (`probe-points --yaw 90` and `--yaw -90`). Near the road, the cost was mostly the trees within 40 m. What a billboard 250-600 m out costs isn't known.
  - CSP's tree loading time, with 2.4 times as many trees in a canopy-model build.
- **What it adds** (laptop builds, made-up WorldCover trees with the Meta/WRI mask):
  - Made-up trees keep 12% of the tree cover 250-600 m out: 9,000 more on Grasstree Hill (28,125 → 37,196), 10,800 on Esperance Coast Road and 68,700 on the 63 km Lyell Highway.
  - The per-km cap counts only the trees within 250 m. It also never cuts below that 12%, so a forest dense enough to meet the cap thins out evenly to 600 m. Before, the Lyell Highway had no trees at all 150-250 m out.
  - Canopy-model builds keep every crown, so their count grows with the tree-covered area: about 2.4 times the 250 m count (Grasstree Hill: an estimated 84,000, against 34,850).
  - Buildings: Grasstree Hill 718 (was 253), Esperance 347 (was 246).
  - Build: the terrain and the DEM read don't change. KN5s grow under 3% (buildings), zips 0.3-1.5 MB and `trees.txt` 32-44%. Build time and peak memory barely move (Lyell Highway: 104 → 109 s, 1.85 → 1.86 GiB, on a laptop).

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

## Roadside furniture

- **To check in game:**
  - guide posts and their reflectors at night;
  - CAMs read as chevrons pointing the right way, in both directions;
  - speed signs readable;
  - frame rate (Grasstree Hill: about 450 guide posts and 300 CAMs, about 25,000 vertices; each KN5 grows 2 MB).
- **Speed limits come only from OSM.** Many rural ways have no `maxspeed` (only 24% of Grasstree Hill Road has one), so those stretches get no signs. CAMs assume 100 km/h there. Tasmania's State Road speed limits (theLIST, CC BY) cover state roads fully, but using them with an OSM centreline would combine two road datasets for the same feature.

## Elevation

- **A mapped line beside the road reads the cutting, not the road** (the Esperance Coast Road dip, 4 Nov).
  - Cause: OSM's line ran 9-14 m up a 25-59% batter for 360 m, so its heights were up to 5.3 m high. That made a 79 m sag, about 1 g at the AI's speed, and a 22% pinch the road doesn't have.
  - Fix: the profile stage now finds such stretches on LiDAR ground and takes the height of the flat road strip beside the line (`profile.road_strip_heights`), or profiles them as coarse where there's no strip.
  - The strip is as wide as the road is mapped: 7 one-metre samples (6 m between the outer two) on a 6-7.9 m road, as before, 5 (4 m) on a narrower one, and more on a wider one (`profile.strip_half`). With a fixed 7, a 4 m road's strip never lay flat, and the steepness test reached over its batters: on Lower Swamp Road 1,022 m was profiled as coarse and 449 m from a strip. Now 629 m is coarse and 651 m from a strip, and 191 m fewer are taken for off the road at all.
  - Esperance after the fix: tightest vertical radius 73 m → 184 m, steepest 10 m grade 22.4% → 13%.
  - The line itself stays where OSM draws it. Realigning OSM way 23248489 there would fix it at the source.
- **The validator warns** (it never fails a build) where a crest or sag would pull more than 0.5 g at the AI's planned speed.
