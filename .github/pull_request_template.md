## What and why

<!-- What this changes, and why. Link the issue it fixes, if there is one. -->

## How it was checked

<!-- Tests added or changed. For a change to a track's look or driving: static validation (`mapcreator validate`), or in game with which CSP version. -->

## Checklist

- [ ] `python -m pytest -q` passes, including `tests/test_private_markers.py`.
- [ ] Nothing private: no hostnames, addresses, personal paths, account ids or secrets (`CONTRIBUTING.md`).
- [ ] A new data source or ported code: its licence is checked against the source, and it's credited (provider attribution, `docs/data-sources.md` or `THIRD_PARTY_NOTICES.md`).
- [ ] Docs and user-facing text: plain language, Australian spelling.
