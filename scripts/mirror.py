"""Mirror the big statewide rasters onto local disk for a build host. Resumable.

    python scripts/mirror.py tas-dem  D:/mirror     # Tasmania 2 m DEM, ~95 GB, one COG
    python scripts/mirror.py tas-chm  D:/mirror     # Tasmania canopy height: 26 GB zip -> ~10 GB TIFF

Then point the config at the files (`tas_dem_source`, `tas_chm_source`).
Builds read the DEM over HTTP without a mirror, but each build spends most of
its time downloading tiles. The canopy model can't be read remotely at all:
its TIFF is deflate-compressed inside the zip.

Downloads resume with HTTP Range requests after an interruption. Each file is
written to `<name>.part` and renamed only once its size matches the server's.
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import httpx

SOURCES = {
    "tas-dem": {
        "url": "https://data.source.coop/alexgleith/tasmania-dem-2m/Tasmania_Statewide_2m_DEM_14-08-2021.tif",
        "file": "Tasmania_Statewide_2m_DEM_14-08-2021.tif",
    },
    "tas-chm": {
        "url": "https://listdata.thelist.tas.gov.au/opendata/data/CHM_2M_STATEWIDE.zip",
        "file": "CHM_2M_STATEWIDE.zip",
        "extract": "CHM_2m/TAS_CHM_2m.tif",
    },
}
CHUNK = 8 * 1024 * 1024
HEADERS = {"User-Agent": "mapcreator mirror"}


def download(url: str, dest: Path) -> Path:
    part = dest.with_name(dest.name + ".part")
    with httpx.Client(headers=HEADERS, timeout=120, follow_redirects=True) as client:
        total = int(client.head(url).headers["content-length"])
        if dest.exists() and dest.stat().st_size == total:
            print(f"{dest.name}: already complete ({total / 1e9:.2f} GB)")
            return dest
        for attempt in range(1000):
            have = part.stat().st_size if part.exists() else 0
            if have >= total:
                break
            try:
                with client.stream("GET", url, headers={"Range": f"bytes={have}-"}) as r:
                    if r.status_code not in (200, 206):
                        raise httpx.HTTPStatusError("bad status", request=r.request, response=r)
                    if r.status_code == 200 and have:
                        have = 0  # server ignored the range; start again
                        part.unlink()
                    t0, done = time.time(), 0
                    with part.open("ab") as f:
                        for chunk in r.iter_bytes(CHUNK):
                            f.write(chunk)
                            done += len(chunk)
                            if done % (CHUNK * 64) < len(chunk):
                                rate = done / max(time.time() - t0, 1e-6) / 1e6
                                print(f"{dest.name}: {(have + done) / total:6.1%} of {total / 1e9:.2f} GB "
                                      f"at {rate:.1f} MB/s", flush=True)
            except (httpx.HTTPError, OSError) as e:
                wait = min(60, 2 ** min(attempt, 6))
                print(f"{dest.name}: {e!r}; resuming in {wait} s", flush=True)
                time.sleep(wait)
        if part.stat().st_size != total:
            sys.exit(f"{dest.name}: size mismatch after download")
        part.replace(dest)
    print(f"{dest.name}: complete")
    return dest


def extract(zip_path: Path, member: str, out_dir: Path) -> Path:
    target = out_dir / Path(member).name
    with zipfile.ZipFile(zip_path) as z:
        info = z.getinfo(member)
        if target.exists() and target.stat().st_size == info.file_size:
            print(f"{target.name}: already extracted")
            return target
        tmp = target.with_name(target.name + ".part")
        with z.open(member) as src, tmp.open("wb") as dst:
            while chunk := src.read(CHUNK):
                dst.write(chunk)
        tmp.replace(target)
    print(f"{target.name}: extracted ({target.stat().st_size / 1e9:.2f} GB)")
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mirror")
    ap.add_argument("source", choices=sorted(SOURCES))
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args(argv)
    src = SOURCES[args.source]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = download(src["url"], args.out_dir / src["file"])
    if "extract" in src:
        path = extract(path, src["extract"], args.out_dir)
    print(f"done: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
