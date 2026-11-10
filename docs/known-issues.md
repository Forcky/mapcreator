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
- the police cars (`core/police_car.py`) after the first in-game check: whether the night values (ksDiffuse 0.35 for the body, 1.3 for the livery band, 2.0 for the light bar's lenses, against 0.6 by day) stop the wash to white at 6 m in the beam and keep the bar looking off at 12 m; whether Tasmania's darker silver reads as silver in sunlight; whether the larger crests read at parking distance. `scripts/police_cars_check.py` builds a test track with all eight. The same near-white wash may affect any prop under the walls' ksDiffuse 0.6 (see `ac-format.md`, "Police cars", "Night")
- whether the tree trunk colliders stop a car at the bark: their radius assumes Tree FX scales a model by `size` all round and by `width` horizontally on top. And what 200-560 colliders per km cost in frame rate.
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
- **Without CSP:** not supported, by the maintainer's decision. Plain AC wouldn't load multi-layout tracks (it wants a `<track>.kn5`), built the AI grid on every load (4 minutes for 12.8 km), and its AI crashed on our open AI line. The website, the track description and ATTRIBUTION.txt say CSP is required.
- **Tree FX:** the models load, and trees stand on the ground with surface alignment on. A comment line in `trees.txt` made CSP load none (fixed: the list has no comments now).
- **Frame rate:** at about 2,500 trees per km, Tree FX costs about 60% of the frame rate on a GTX 1660 Ti at 1440p, mostly from the trees near the road. Canopy-model builds keep every real crown, with no distance bands or cap, and are faster: Grasstree Hill 90 / 63 fps (median / slowest) against 68 / 45 for the capped WorldCover build, because the near-road trees halve. WorldCover builds keep the bands and the cap, which counts only the trees within 250 m since the scenery reach went to 600 m.
- **AI:** AC's AI drives both layouts from the spawn to the finish. It ignores the 130 km/h cap in the line's speed data. After the finish it eases off and comes to rest on the road about 60 m before the end of its line, about 100 m short of the end wall.
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

Lower Swamp Road and Hartz Road have been raced by the AI in game (5 Nov), but nobody has driven them by hand yet. The `GRAVEL` surface's numbers (`docs/ac-format.md`) started from community gravel stages and are to be tuned by feel.

- **The AI overdrove the gravel (5 Nov), so its friction is lower and it has a pace hint, both untested.** With friction 0.88 and no hints, the AI drove the gravel at near-tarmac pace, with wheel slip as on a seal: about 106 km/h on average over Hartz Road (up to 140), and a median of 93-96 km/h on Lower Swamp Road's one lane (up to 156). In reverse on Lower Swamp Road, all 5 cars jumped the crest the validator warns about, about 1.8 km in, landed on the steep slope beyond it and took about 70 damage each.
  - The maintainer's decisions: GRAVEL friction 0.80 (chosen without a test by hand), and a pace hint of 0.8 over each unsealed run up to the finish (`docs/ac-format.md`).
  - **The hint may not slow the AI over the crest.** An earlier test at the finish showed a hint scales AC's corner speed: on a straight run-off, 0.05 to 0.2 did nothing, and only 0 stopped a car. So 0.8 may do nothing on a straight, and the crest on Lower Swamp Road is on one. The lower friction is the other lever.
  - **To check in game (the AI re-test):** Lower Swamp Road in reverse, over the crest about 1.8 km in: whether the cars still leave the ground there, and still take damage. Then the gravel speeds on both roads against those above, and that nothing breaks: no car off the narrow road or stuck, the run-off stop as before, the grid and start as before.
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
  - **Tuning by hand (Q3 of the gravel research):** friction 0.80, damping 0.01, dirt additive 0.8, `gravel.wav`, 1 cm Perlin bumps, on Lower Swamp Road and Hartz Road. With CSP's Surfaces FX **off** first, because it's off by default and offline only, so the plain numbers must feel right on their own; then **on**.
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
- **Running wide at speed:** in game on 4-5 Nov, AC's AI ran 1.0-1.3 m outside the line at 127 km/h. Where the line came within 1.2 m of the edge at that speed, cars left the road: Grasstree Hill forward at 5.95-6.03 km (alone it crashed out; in an 8-car race every car put two tyres off and 4 crashed out) and Brinktop Road at about 5.6 km (two tyres off for 3 cars of 8).
  - The fix: the line's margin from the edges grows with its planned speed, from 1.2 m at 60 km/h to 2.5 m at 120 km/h, and its planned braking shares the grip with cornering, so it slows before a bend tightens (`docs/ac-format.md`). Through Grasstree Hill's bend, at 5,926 m: 2.52 m from the edge at 117 km/h, where it was 1.20 m at 127 km/h. At Brinktop Road's 5,619 m: 2.50 m at 130 km/h, where it was 1.20 m.
  - **Re-tested in game (5 Nov):** fixed. Grasstree Hill forward alone: no tyre off; 8 cars: all finished, none off in the bend. Brinktop Road at 5.6 km: no tyre off. AC doesn't use the planned speeds (`docs/ac-format.md`): the wider margin did it.
  - **A new weak spot, Brinktop Road's kink at 9.95-10.0 km,** taken at 190-200 km/h: 5 of 8 cars put 2-4 tyres off (no damage, all finished). The margin stopped at 2.5 m because the planned speed stops at 130 km/h. It now follows the speed the cars may reach, up to 3.3 m at 200 km/h, and the line there keeps 3.05 m from the edge.
  - **To check in game:** Brinktop Road's kink with 8 cars, and Grasstree Hill forward's 6 km bend with 8 cars (unchanged by the last step, as a check).
  - The cost: the planned run is about 2% quicker than the centreline line's on Grasstree Hill, where it was 3%. On a 4 m road (Lower Swamp Road) the line keeps to the middle above about 97 km/h.
  - If cars still run wide, the next step is a limit on how fast the line may move across the road at speed.
- **AI lap time:** the AI's time from gate to gate, both layouts. Its planned run is about 2% quicker on Grasstree Hill (381.5 s against 389.0 s), but AC sets its own speed. In game, before the fix for running wide, the racing line's cars were about 7% quicker than the centreline line's (reverse: 332 s alone against 351.3 s).
- **On the road:** the AI stays on the road through the tight bends in both layouts, and doesn't run wide on the exits. The line now comes within 1.2 m of the edge only at or below 60 km/h. The probe (`harness/ac_harness.py probe-points`) teleports a car onto the line every 200 m and checks all four wheels.
- **8-car races:** the cars now share one line that crosses the whole road. Whether they still race clean, side by side and passing, and get away from the grid in pairs as before.
- **Run-off:** as before, every car should come to rest on the road about 60 m before the end of the line.
- **Spawns:** the grid and the hot-lap spawn are unchanged; the line stays on the centreline over them, and the AI should leave the grid as before.
- **Track length:** `log.txt`'s `TRACK LENGTH` on Grasstree Hill and Lyell Highway against the line's real length (the last cumulative length in `fast_lane.ai`; Grasstree Hill forward: 12,737 m). With points 5 m apart, Lyell should read about 1.3% long, against 32.8% with 1 m points.

## Crest bends

On Lyell Highway forward, AC's AI put most of an 8-car field off the road at 0.72-0.80 km on lap 1, in all three races (9 Nov): 3-6 cars of 8, at 110-120 km/h. The bend is a 109-135 m left-hander that starts just over a crest (a 556-673 m vertical radius) and runs down an 8.5% grade, with no banking. The cars drifted right through it, 2-4 m right of the centreline at 720 m and 9-12 m at 785 m, and came to rest 8-18 m off the road at 790-800 m; the two or three that kept to the left got through. The AI line kept 2.6-3.2 m from both edges there.

- **The fix:** a pace hint at such bends (`docs/ac-format.md`): 0.87 over 688-730 m along the line, easing in from 638 m and out to 780 m. The physics says 0.87; the issue suggested about 0.8 over 0.65-0.85 km.
- **In game (10 Nov), forward: fixed.** No car damaged at 0.72-0.80 km in three races (main: 4-5 of 8 each time). The cars reach 720-730 m at 99-108 km/h, against 110-120 before; one or two still ran wide for a moment and recovered.
- **Reverse: two bends still put the field off** (one race each, 8 cars). On main no car finished, all 8 off at reverse 14.0 km; with the hints 1 of 8 finished, 3 off at 14.0 km and 4 at 27.25 km. With the hint below, three more races (10 Nov): none off at 14.0 km (0 of 8, 0 of 7, 0 of 8), but 18 of the 22 cars that reached 27.25 km left the road there.
  - **Reverse 14.0 km: braking over a crest.** A 44-66 m left-hander at 13.94-14.01 km starts 40 m past a 220-280 m crest and runs down 15%, with no banking. The hint written for it before sat at 14.09-14.17 km, but that was the next bend's (a 115-143 m left-hander over a 450-560 m crest), and in the right place: nothing was wrong with the placement. This bend had none, because the rule took the AI to reach its own 71 km/h corner speed, and there the pace only touched 0.9 for 7 m. It doesn't: the AI plans its braking from about 130 km/h at the crest at the rate it counts on, but over the crest the tyres carry 45-65% of the car, and it came in at 86-99 km/h (76-80 with the next bend's hint). The rule now works out the car's speed braking with the grip it has (`docs/ac-format.md`): it comes in at about 117 km/h where the AI plans 92, and the bend gets 0.65 over 13.93-13.98 km, easing in from 13.88 km over the crest and out to 14.03 km. The tester suggested about 0.75 from 13.88 to 14.04 km, from the cars that got through at 70-77 km/h. In game it held every car (at most two tyres off), though at 98 km/h at 13.95 km and 78 km/h at 13.98 km, 10-20 km/h over the model. From the crest the cars slowed at about 4.5 m/s² in all three builds (main, the first hints and this one); the hint only moved where they started.
  - **Reverse 27.25 km: still no hint; nothing in the rule tells it apart.** A 66-90 m left-hander at 27.13-27.27 km, down 3-5%, just past a gentle crest (a 1,300-2,900 m vertical radius, where the tyres keep 92-98% of the car), unbanked. In game (10 Nov, three races) 18 of the 22 cars that reached it left the road, 15 of them alone: all at 27.19-27.21 km, on the outside of the exit, at 82-93 km/h, 3.5-5.5 m from the centreline on a road 3.5 m either side. Those at 79-87 km/h stayed on. They came in at 129 km/h at 27.10 km, 118 at 27.13, 107 at 27.16 and 88-93 at 27.19, and ran 2.5-4 m outside the line, which keeps 2.0-2.4 m from the outside edge through the exit.
    - **The rule's pace there is 0.957** (the AI's 89 km/h corner speed against the 85 km/h the grip allows) for about 4 s, over the trigger. The braking-zone fix doesn't reach it: the crest is too gentle to cut the braking. Bends as short of grip and longer are driven cleanly: reverse 18.46-18.65 km (0.93 for 5 s) and 14.88-14.96 km (0.946), over 20 passes each on 10 Nov. A trigger that caught 0.957 would also hint 3 bends on Lower Swamp Road reverse and 6 on Mount Dandenong Tourist Road, where the AI is fine.
    - **Braking carried into the bend** (the AI braking as it turns in; the friction circle with g times the descent) doesn't single it out. With the AI slowing at 4.5 m/s² into every radius, the grip it asks for peaks at 1.20 times what it has at 27.25 km, against 1.30 at 18.46 km and 1.28 at 14.88 km. With it braking only for each bend's slowest point, 1.18 against 1.34 at 14.9 km.
    - **The crown under the line** (`road_banking` leaves it out): through 27.25 km's exit the line is on the outside half, where the 2.5% crown falls away from the bend; at 18.5 and 14.9 km it's on the inside half. With the descent on the friction circle as well, that's the one thing that sets 27.25 km apart from those two: 3-5% short of grip through the exit, where they have it in hand. But it's too small to slide a car 2.5-4 m wide, and the same reckoning finds 3 bends on Mount Dandenong Tourist Road, 9 on Lower Swamp Road and 33 on Lyell Highway as bad or worse.
    - **The in-game speeds don't fit the grip at all.** On the line's radius, 107 km/h at 27.16 km is 1.18 g, where the cars go off; 98 and 91 km/h at 13.95-13.96 km on 14.0 km's hint is 1.33-1.38 g, where they stay on. So the cars don't drive the line's radius, or the Abarth holds more than 0.9 g, and no measure worked out from the line ranks 27.25 km above 14.0 km. What 14.0 km has that 27.25 km doesn't is a bend that opens straight after it (65 m at 14.00 km, 139 m at 14.02 km), where 27.25 km stays at 70-74 m to 27.25 km with the line at its outside margin.
    - **The maintainer's decision (10 Nov): no design superelevation on coarse stretches**, so the fix has to be the AI's: a hint or the line, from a general rule. No rule is added until in-game data tells 27.25 km apart: the cars' speed and offset from the line at 14.85-14.96 km and 18.43-18.65 km in the same races, against 27.10-27.26 km.
- **No banking on coarse DEM stretches.** Lyell Highway is 92% coarse (not LiDAR), and those stretches get the crown and no superelevation (`section_stage`), because a coarse DEM can't see a road's camber. Real curves that tight are superelevated: Queensland's road design manual calls superelevating all horizontal curves "normal practice", wants curves under 2,000 m superelevated on a 100 km/h rural road (desirably; tighter ones may go without only with justification), and asks for curves in mountainous terrain to be fully superelevated (TMR, *Road Planning and Design Manual*, 2nd edition, Volume 3, Supplement to Austroads Guide to Road Design Part 3, July 2025: 7.7, 7.8 and Table 7.8). A design superelevation from the curvature (TMR's 2002 manual, chapter 11, equation 11.5, the linear method it says Austroads favours) would give that bend 6-7%, and the rule's pace there 0.91-0.92: no hint. The maintainer decided against it (10 Nov): the terrain isn't changed to suit the AI.
- **To check in game** (8 Abarth 500s, AI level 100):
  - from the 10 Nov reverse runs, if they're still to hand: the cars' speed and offset from the line through 14.85-14.96 km and 18.43-18.65 km, against 27.10-27.26 km, to tell 27.25 km apart;
  - Lyell Highway reverse and forward otherwise as on 10 Nov: 14.0 km, 29.73-29.78 km and both forward hints clean.

## Scenery reach (600 m)

Trees and buildings now stand within 600 m of the road (`core.reach`); they stopped at 250 m. Power-line supports still reach the terrain's edge less 5 m (695 m). None of it has been seen in game yet.

- **To check in game** (CSP 0.2.11, GTX 1660 Ti, 1440p):
  - **Do far trees draw?** Past its last LOD (84-270 m out, by model), Tree FX draws each tree as a billboard. The billboard fades out at a distance that CSP's own code sets and doesn't document (`docs/ac-format.md`). If that's under 600 m, the far trees never show.
  - **What do they cost?** Compare frame rates against a 250 m build, at the same probe points, both along the road and looking out to the side (`probe-points --yaw 90` and `--yaw -90`). Near the road, the cost was mostly the trees within 40 m. What a billboard 250-600 m out costs isn't known.
  - CSP's tree loading time, with 2.4 times as many trees in a canopy-model build.
- **What it adds** (laptop builds, made-up WorldCover trees with the Meta/WRI mask):
  - Made-up trees keep 12% of the tree cover 250-600 m out: 9,000 more on Grasstree Hill (28,125 → 37,196), 10,800 on Esperance Coast Road and 68,700 on the 63 km Lyell Highway.
  - The per-km cap counts only the trees within 250 m. It also never cuts below that 12%, so a forest dense enough to meet the cap thins out evenly to 600 m. Before, the Lyell Highway had no trees at all 150-250 m out.
  - Canopy-model builds keep every crown, so their count grows with the tree-covered area: about 2.4 times the 250 m count (Grasstree Hill: an estimated 84,000, against 34,850).
  - Buildings: Grasstree Hill 718 (was 253), Esperance 347 (was 246).
  - Build: the terrain and the DEM read don't change. KN5s grow under 3% (buildings), zips 0.3-1.5 MB and `trees.txt` 32-44%. Build time and peak memory barely move (Lyell Highway: 104 → 109 s, 1.85 → 1.86 GiB, on a laptop).

## Grass pack (ground classes and Grass FX)

In Tasmania the terrain is split into ground classes from TASVEG, each with its own ground texture and Grass FX configuration, and Grass FX draws our own atlas (`docs/ac-format.md`, "Grass FX"). Every number is a starting point.

- **In game, 9 Nov 2026** (CSP 0.2.11, Grass FX Medium, interleaved with main in one session, set up as for main's frame-rate baseline):
  - **Frame rate:** Lyell Highway 1.5% faster than main (its forest, 63% of the ground within 250 m, no longer grows grass); Grasstree Hill 5.4% slower, just over its 5% budget, worst at 9.8-12.5 km, up to 11% at single points. The slowest 10% was within budget on both.
  - **Looks, kept:** buttongrass moorland (straw and olive-tan tussocks, the verge still mown green), the brown forest floor, grassy woodland and regrowth reading differently, canopy-dark far hillsides, the sandy band on lake shores. Pasture and town lawn looked almost as on main.
  - **Problems:** a straight-edged brown block on the paddock at Grasstree Hill's 11.5 km, and faint rows of tussocks on Lyell's moorland plain at 15.3 km. The heights table was left for a viewpoint with the guide post in frame.
  - **Tuned after it** (the maintainer's Q9: "tune until it fits"): pasture and towns back to today's ground and CSP's default grass, with no atlas pieces (the 9.8-12.5 km stretch is 98-100% pasture within 60 m); the mown verge (A) never denser than CSP's default; terrain gathered into about as many meshes as main (Grasstree Hill 32, main 33, the first test 167); class edges dithered; the open classes' textures repeat every 16 m, warped. To re-measure, interleaved with main as before.

- **To check in game** (CSP 0.2.11, then the visual checks again on 0.3.1; Grass FX quality Medium, also Very low and High on Grasstree Hill; the Grass FX debug app on, `APP_GRASSFX_DEBUG` in CSP's `gui.ini`). Build each road on main and on the pack with the same job:
  - **Loads cleanly:** `ac_harness.py check-log`: no CSP texture or config errors. The grass shows our atlas, not CSP's default: the tussocks, buttongrass and bracken are recognisable. Tree FX trees still stand on every new material, and Rain FX puts no puddles on the ground classes.
  - **Frame rate,** at fixed probe points and in an AI race, against main with Grass FX off and on, run interleaved in one session (the gaming PC loses about 2.4% a run as it warms, so numbers from another session don't compare): the budget is the median within 5% of main's on Grasstree Hill and 10% on Lyell Highway, and the slowest 10% within 10% / 15% (the maintainer's, 5 Nov 2026). Over it: lower the heights first, then the pieces' chances, then the number of tall pieces. Also a moorland `SHAPE_SIZE` sweep (2, 3, 4, 6) on Lyell.
  - **Heights:** what `SHAPE_SIZE` is in metres, beside a guide post (1.0 m tall, 0.8 m off the edge), then set: verge about 0.05-0.15 m, pasture 0.1-0.3 m, tussock and buttongrass 0.5-1 m (stalks to 1.5 m), bracken 0.5-1.5 m.
  - **Colour:** do the blades take each class's ground (straw, olive-tan, dark green)? Is CSP's colour sync per pixel, so the buttongrass stalks stay brown on green-only hummocks in one piece? How far out do coloured pieces (bracken, the stalks and Themeda's heads) fade to green on 0.2.11? Does the colour mask leave moorland's peat and the woodland's litter bare, and grow on the rest? (The mask's formula isn't documented: whether its luminance is linear or not decides whether the peat is dark enough.) Record the `MASK_*` values that work.
  - **Class edges:** the ground's colour changes at a class's edge by triangle, dithered over about 12 m near the road. Does the 11.5 km blue gum patch on Grasstree Hill still read as a block? Try `MASK_BLUR` 1, 2 and 3. If the edges still look bad, the answer is a multilayer ground with a blended class mask (the design's mechanism B), a PR of its own. So is the dithered change to canopy green 220-280 m out in forest.
  - **Rows:** are the tussocks on Lyell's moorland plain at 15.3 km irregular now? If rows remain, they're CSP's own placement, not our ground.
  - **Near the car:** on 0.2.11 CSP spawns its base blades nearby in every class ("custom types of grass only affect third generation pass onwards"): thin blades in moorland and tussock colours, acceptable?
  - **Near water:** on Lyell's lake shores, with Extra FX on, does tall grass look see-through against CSP's water? If so, try `[EXTRA_FX] FORCE_GRASSFX_GBUFFER_PASS = 1` and measure what it costs.
  - **Road safety:** the 2 m verge stays mown everywhere, so the tall classes never hide a guide post or CAM; the gravel's centre strip stays short (A); no grass through the seal or the gravel.
  - **Not documented:** how a part weight (`MAP = D, 0.6` on regrowth, `MAP = B, 0.5` in grassy woodland) mixes the texture groups' chances: is regrowth sparser in bracken than bracken fernland?
- **The ground textures and the atlas are procedural.** If a piece looks poor in game (bracken most likely), a CC0 photo card (ambientCG's `Foliage` atlases, Poly Haven's `fern_02`) can replace it, its luminance in green for the synced pieces.
- **One weight per class.** TASVEG codes the design gives a lower weight within a class (coastal grass and herbfield, rockplate grassland, sparse buttongrass on slopes) take their class's full weight: one material per class. A part-weight material can follow if they look too dense.
- **The mainland keeps today's grass:** NVIS and WorldCover classes are for a later PR, as is a mainland atlas group for spinifex.

## Side-road junctions

The route stage finds the real side roads that meet the route; each gets its first 100 m drawn as road, joined to the mainline by a corner flare either side, and nothing stands in its mouth or on its stub (`docs/ac-format.md`, "Side-road junctions" and "Side roads' stubs"). Each stub is closed off the stage, with police cars at some of the major closures ("Closures on the stubs").

- **To check in game** (Grasstree Hill, Brinktop Road, and an acute junction):
  - **the seams:** drive slowly across each mouth, on and off the edge, along the mainline's edge across the mouth, across the stub's first full row (about 8-30 m in) and along each flare's verge. No bump, no gap to fall through, no ground showing through the asphalt, no flicker where two surfaces meet. Grasstree Hill: Sugarloaf Road (1.34 km, right), the unnamed gravel road at 5.51 km (right, 54°: a long flare on its acute side, closing round a grass gore), Back Tea Tree Road (9.50 km, left, 76°) and Cold Blow Road (12.48 km, left, gravel); Brinktop Road: the Franklin and Edward Street crossroads (0.18 km), Bridge Street (0.50 km, right), Parramore Street (0.84 km, right);
  - **the edge lines** break across each mouth, from one flare's tangent point to the other's, and taper back over a metre either side; the throat is plain asphalt (or loose gravel), and the stub has its own lines from its first full row;
  - **the terrain at the mouths:** the ground meets the flares' verges and the stubs' skirts without a step or a cliff, and a stub climbing or falling away from the road (Sugarloaf Road drops about 12%) reads as a road, not a ramp;
  - **driving into a stub:** a car that runs wide can turn in on the flare and drive the stub to its end, where the asphalt stops square and the side road carries on as ground. Only straw bales and police cars stop it now (below); whether the stub's end shows past the closures is to judge;
  - the timing gates: only Brinktop Road's start gate moves, 0.5 m back, clear of Percy Street's flare;
  - as before: nothing stands in a side road's mouth or on its stub, the guide posts either side of a mouth read naturally, and a rail ending at a mouth doesn't look cut off.
- **Flares are 8 m corners,** the route's own corner radius, and longer on the acute side. Real rural Tasmanian junctions may be wider (heavy vehicles) or tighter (minor roads). The mouth's margin (5 m either side of the flares) and the 3 m band are as before.
- **Side roads' widths are class defaults:** no side road on the examples has a `width` tag and few have `lanes`, so stubs are as wide as their class default, times the job's width option.
- **A stub's heights come from the DEM along its line,** smoothed over 15 m and blended from the mainline's cross-fall over 15 m or more. Victoria's 10 m DEM blurs a side road's formation, so Mount Dandenong Tourist Road's stubs follow the hillside more than the road.
- **Stubs left out:** where a side road's mouth would overlap another's on the same side (Brinktop Road's Charles Street, which leaves Gunning Street's node; Mount Dandenong Tourist Road's clusters, 6 of them), at the route's very ends (Henry Street and the Tasman Highway on Brinktop Road), beside another part of the route (School Lane on Mount Dandenong Tourist Road), too short for a ribbon past its flares (a 33 m dead-end spur on Lyell Highway), or where Bridge Street's one-way arm runs into its other arm. They keep their mouths clear as before. Drawing a cluster as one patch, a shared throat, is the obvious next step for town roads.
- **Flares on bending roads:** a flare is drawn in the mainline's frame, which folds over itself far out on the inside of a tight bend, and a side road can bend hard inside a long flare. Either way the flare is drawn again at half the size, then a quarter, or the stub is left out. Two did on the examples (Perrins Creek Road on Mount Dandenong Tourist Road, Gunning Street on Brinktop Road). A side road that runs alongside the route after leaving it (Bella Vista Crescent) gets a long paved gore before its flare closes it.
- **Slivers in the terrain:** its triangulation isn't constrained, so where a stub's skirt meets the mainline's, a terrain triangle can overlap a skirt by a few square centimetres: 6 points of a 0.3 m grid across Grasstree Hill's 8 mouths, against none without stubs. Too small to see, probably; a constrained triangulation would remove them.
- **A gravel side road off a sealed road** brings the gravel material, its 2048 x 512 texture and the `GRAVEL` surface into the track: most of Grasstree Hill's 1 MB growth.
- **Driveways, service roads and tracks** aren't junctions (the maintainer's decision, 10 Nov 2026): the road data doesn't hold them. Grasstree Hill has 37 driveways and Lower Swamp Road 39, and guide posts, rails and trunk colliders still stand in some of their mouths. Clearing them needs the OSM index to keep those ways too, a PR of its own.
- **Roundabouts** are recorded and built as before: only the route's arc of the ring is road, and nothing is cleared at their exits. A later PR draws them.
- **Mount Dandenong Tourist Road** has 73 junctions in 14.6 km, 8 clusters within 30 m of each other: its guide posts fall from 1,535 to 1,443, and 1,053 trees (420 trunk colliders) come off its side roads.
- **Cost** (laptop builds, made-up WorldCover trees; mesh-stage vertices and triangles, and each layout's KN5, without stubs and with them):

| Road | Junctions | Stubs | Stubs' own meshes, vertices | Mesh-stage vertices | Triangles | KN5, each layout |
|---|---|---|---|---|---|---|
| Grasstree Hill | 8 | 8 | 4,802 | 419,430 → 425,301 (+1.4%) | 636,289 → 644,078 (+1.2%) | 29.2 → 30.2 MB (+3.4%) |
| Brinktop Road | 18 | 14 | 8,062 | 307,052 → 317,207 (+3.3%) | 492,450 → 505,965 (+2.7%) | 21.2 → 22.4 MB (+5.8%) |
| Esperance Coast Road | 10 | 10 | 5,937 | 699,050 → 706,449 (+1.1%) | 1,001,332 → 1,010,602 (+0.9%) | 43.6 → 45.4 MB (+4.1%) |
| Lower Swamp Road | 3 | 3 | 1,757 | 208,271 → 210,416 (+1.0%) | 331,503 → 334,205 (+0.8%) | 15.5 → 15.6 MB (+0.7%) |
| Lyell Highway | 4 | 3 | 1,387 | 1,891,998 → 1,893,671 (+0.1%) | 2,919,669 → 2,921,694 (+0.1%) | 121.9 → 122.7 MB (+0.6%) |
| Mount Dandenong Tourist Road | 73 | 66 | 54,272 | 431,150 → 472,967 (+9.7%) | 682,085 → 736,832 (+8.0%) | 30.6 → 33.6 MB (+9.9%) |

  - The mesh stage grows by less than the stubs' own meshes: the terrain gives up its lattice under them. The KN5s grow more than their vertices suggest where a stub brings a material the road didn't have: a gravel side road off a seal (every example but Lower Swamp Road, which is gravel already) or a one-lane seal (Mount Dandenong Tourist Road).
  - The stubs' rows are a metre apart for their first 20 m and 2 m beyond: about 40% fewer stub vertices than 1 m rows all the way (Grasstree Hill: 4,800 against 7,500 with ground classes, 7,000 against 11,600 without).
  - Further savings if it's too heavy: rows 4 m apart past 40 m (the stubs are nearly straight out there), no skirt on the stubs (the terrain fills to their verges), or 60 m stubs on dense town roads.
  - Frame rate: the stubs add a few meshes per 400 m (packed by material), and fewer vertices than the trees do. To measure with the harness's `probe-points` at Olinda (about 6.8 km on Mount Dandenong Tourist Road) against a `junctions: none` build.
- **The closures and police cars** (`core/stub_closures.py`), to check in game, by day and at night (CSP, headlights on):
  - **looks:** the tape across each mouth, spanning the throat where the flares widen it; the boards or bales a metre onto each verge; the second tape and sign behind; on a skewed junction the whole closure further in, its tape at least 3.5 m from the edge. At night the boards, signs and tape glint like the reflectors from 25 m and more without burning out at 8-12 m (`closure_retro` 1.3), and the cars don't wash out (their own night values, `ac-format.md`, "Police cars");
  - **a major closure at 20 and 60 km/h:** the car passes the tape and the boards (passable) and stops at the police car behind them, at its body;
  - **bales:** at 20 and 60 km/h the car stops at the straw, and doesn't slip through a 6 cm gap or glance off round the end onto the flare;
  - **the overshoot stub:** Gunning Street (Brinktop Road, 0.44 km, left), straight ahead where the route turns about 107 degrees: nothing solid before 27 m;
  - **the AI:** it never leaves its line, so it never meets a closure; check it drives Brinktop Road and Mount Dandenong Tourist Road as before;
  - **frame rate:** at Olinda (Mount Dandenong Tourist Road, about 6.8 km), against a `junctions: none` build. The closure props cost nothing measurable on their check track (164 fps facing a major closure against 160), nor 16 cars on theirs.
- **Which majors get a car** is a greedy pick: biggest road first, then no other within 2 km. Mount Dandenong Tourist Road's 13 major closures get 5 cars, not the cap's 6: the 2 km rule leaves no room for a sixth. The other majors keep their boards and their empty slot.
- **A closure moves in on a skewed junction** (61 of Mount Dandenong Tourist Road's 66), so its tape keeps 3.5 m from the edge. On the shallowest junctions (under about 35 degrees off the road) that's 18-26 m in, past the flare: whether that reads naturally from the road is to judge in game.
- **Closures' cost** (the same laptop builds, on top of the stubs; vertices are visual and collider, KN5 each layout):

| Road | Closures: major / minor with bales / minor with boards / limited | Cars (jurisdiction) | Added vertices (collider) | Mesh-stage vertices | KN5, each layout |
|---|---|---|---|---|---|
| Grasstree Hill | 8: 3 / 3 / 2 / 0 | 2 (Tas) | 5,136 (316) | 425,301 → 430,753 (+1.3%) | 30.2 → 31.5 MB (+4.4%) |
| Brinktop Road | 14: 3 / 5 / 6 / 0, one overshoot | 2 (Tas) | 7,676 (496) | 317,207 → 325,379 (+2.6%) | 22.4 → 23.9 MB (+6.5%) |
| Esperance Coast Road | 10: 1 / 4 / 5 / 0 | 1 (Tas) | 5,288 (328) | 706,449 → 712,065 (+0.8%) | 45.4 → 46.8 MB (+2.9%) |
| Lower Swamp Road | 3: 0 / 3 / 0 / 0 | none | 912 (180) | 210,416 → 211,508 (+0.5%) | 15.6 → 16.0 MB (+2.6%) |
| Lyell Highway | 3: 0 / 0 / 2 / 1 | none | 1,124 (0) | 1,893,671 → 1,894,795 (+0.1%) | 122.7 → 123.1 MB (+0.3%) |
| Mount Dandenong Tourist Road | 66: 13 / 21 / 32 / 0 | 5 (Vic) | 36,012 (1,800) | 472,967 → 510,779 (+8.0%) | 33.6 → 36.5 MB (+8.6%) |

  - Most of a small road's growth is the two textures: the closures' atlas (0.35 MB) and, with a car, the jurisdiction's (0.7 MB). The rest is geometry, about 550 vertices a closure and 560 a car.
  - With the stubs, Mount Dandenong Tourist Road comes to 431,150 → 510,779 mesh-stage vertices (+18%) and 30.6 → 36.5 MB, under the research's estimate (+22%, 5-7 MB).

## Power lines

- **Answered in game (4 Nov, Grasstree Hill, with towers then allowed 700 m out):**
  - The wires sag sensibly and cross the road 22-57 m above it.
  - Towers beyond the 300 m terrain stood in the sky. Supports are now built only on the terrain, and that has since been widened to 700 m, so Grasstree Hill's two lines are back: 8 towers, the farthest 650 m out.
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
  - the reflectors, CAMs and speed signs at night, at other times of night and in other weather (the night adjustment was tuned at 22:00, clear);
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
