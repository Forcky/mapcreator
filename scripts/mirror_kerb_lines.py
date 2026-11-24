"""Mirror the councils' kerb lines into one local file for the build worker (`providers/councils/kerb_lines.py`).

    python scripts/mirror_kerb_lines.py D:/mirror/kerb_lines.sqlite

Then point the worker at it (`kerb_lines` in config.local.toml, or `MAPCREATOR_KERB_LINES`). Run it from a host
that may reach the councils' servers, not from the worker: the worker reads the file and needs no egress for it.
Monthly is plenty (Hobart's layer was last edited 17 Nov 2026; Glenorchy's file 8 Nov 2026).

It reads:
- the City of Hobart's Road Formation, layer 0 of
  https://services1.arcgis.com/NHqdsnvwfSTg42I8/arcgis/rest/services/Road_Formation/FeatureServer, its Top of Kerb
  (`roatkb`) and Edge of Formation (`roaeof`) lines (about 50,000), 1,000 a page, in EPSG:28355 (CC BY 4.0);
- Glenorchy City Council's kerbs, https://maps.gcc.tas.gov.au/OpenData/gpkg/kerbs.zip (about 2.7 MB; 3,456 lines,
  EPSG:28355, CC BY 3.0 AU).

The file is written beside the target as `.part` and moved into place when complete, so a build never reads half of
one. A source that fails leaves the old file as it was.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from mapcreator.providers.councils import kerb_lines as kl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path, help="the mirror file to write (SQLite)")
    args = ap.parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    hobart = kl.hobart_lines()
    print(f"Hobart: {len(hobart)} lines", flush=True)
    glenorchy = kl.glenorchy_lines()
    print(f"Glenorchy: {len(glenorchy)} lines", flush=True)
    kl.write_mirror(args.out, {kl.HOBART: hobart, kl.GLENORCHY: glenorchy}, {
        kl.HOBART: {"url": kl.HOBART_LAYER, "licence": "CC BY 4.0", "fetched": now},
        kl.GLENORCHY: {"url": kl.GLENORCHY_ZIP, "licence": "CC BY 3.0 AU", "fetched": now},
    })
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
