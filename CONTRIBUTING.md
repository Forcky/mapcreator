# Contributing to mapcreator

Thanks for taking a look. mapcreator is an early-stage personal project, maintained in spare time. Small fixes are welcome as pull requests straight away. For anything bigger, open an issue first, so we can agree on the approach before you spend time on it.

## Set up

You need Python 3.12. The build pipeline runs on Linux, macOS and Windows, without a GPU or the game.

```sh
python -m venv .venv
. .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install -e ".[service,s3,osm,dev]"   # the same extras CI installs
python -m pytest -q
```

Tests that need the public data services are marked `network` and skipped by default. Run them with `python -m pytest -q -m network`, sparingly: those services are shared.

To build a track, see the quick start in `README.md`. To develop on a Windows PC with Assetto Corsa, read `docs/dev-handover.md` and `harness/README.md`. For how the code fits together, read `docs/architecture.md` and `docs/data-flows.md`.

## Before you open a pull request

- Run `python -m pytest -q`. CI runs the same suite on Linux, and scans every commit for secrets with [gitleaks](https://github.com/gitleaks/gitleaks).
- To run gitleaks on each commit before it's made, install the hook once: `pip install pre-commit && pre-commit install`.
- Keep each commit to one reviewable step, and say in its message what changed and why.
- If the change alters what a track looks like or how it drives, say how you checked it: static validation (`mapcreator validate`), or in game (which CSP version).

## Nothing private

This repository is written to be published, and `tests/test_private_markers.py` checks every tracked and untracked file. It fails on private IP addresses, personal paths, email addresses, cloud account ids, private keys and tokens.

- Settings come from `MAPCREATOR_*` environment variables or a gitignored `config.local.toml`. Commit only `*.example` files, with placeholders.
- Keep logs, scratch files and build output out of the repository folder, or in the gitignored `out/` and `cache/`.
- Strip EXIF data from any photo before it goes into `docs/`.
- Maintainers also keep a private list of deployment-specific words, which the same test reads from a gitignored `.private-markers` file (or the file `MAPCREATOR_PRIVATE_MARKERS` names). You don't need one.

## Data and licences

Every track is built only from data that may be redistributed. `docs/data-sources.md` has the rules and the sources in use. In short:

- Each data source is a provider under `src/mapcreator/providers/`. It declares its licence, its attribution text, and whether derived content may ship in a track.
- No Google, Esri, Bing or Mapbox content. No NoDerivatives (ND) imagery as textures, and no NonCommercial data.
- Check licences, attribution wording and file formats against primary sources: the licence text, the dataset's own page, the format's reference. Say in the pull request where you checked.
- Code ported from another project keeps a provenance header (source URL, licence, commit) and gets an entry in `THIRD_PARTY_NOTICES.md`. Don't copy code from a repository without a licence. For MS-PL sources such as Content Manager's AcTools, re-implement the format from a description; don't copy the code.

## Writing style

Docs, comments and anything a user reads are in plain language, with Australian spelling (licence as a noun, metre, colour). Use short sentences, say what something does and why, and give numbers with their units. `CLAUDE.md` has the project's rules in brief.

## Licence of contributions

mapcreator is under the Apache License 2.0 (`LICENSE`). As its section 5 says, unless you explicitly state otherwise, anything you submit for inclusion is under the same licence, without additional terms. There's no contributor licence agreement and no sign-off to add.

## Conduct

Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md). To report a security problem, see [SECURITY.md](SECURITY.md).
