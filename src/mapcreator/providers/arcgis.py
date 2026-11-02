"""Heights from an ArcGIS ImageServer's exportImage, for `TiledDem`.

exportImage returns any box at any size in the CRS asked for, as a float32 GeoTIFF. Two traps, both
handled here: a request the server can't serve still comes back HTTP 200 as `image/tiff`, with a short
JSON error as the body; and cells with no data come back as the noData value asked for.
"""

from __future__ import annotations

import io
import json

import numpy as np

from mapcreator.providers.tiled import Fetcher

NO_DATA = -9999.0
TIFF_MAGIC = (b"II*\x00", b"MM\x00*")


def export_image_fetcher(url: str, crs: int, *, mosaic_rule: dict | None = None, timeout_s: float = 120.0,
                         user_agent: str = "mapcreator") -> Fetcher:
    def fetch(bbox: tuple[float, float, float, float], size: int) -> np.ndarray:
        import httpx
        import rasterio

        params = {"bbox": ",".join(f"{v:.8f}" for v in bbox), "bboxSR": str(crs), "imageSR": str(crs),
                  "size": f"{size},{size}", "format": "tiff", "pixelType": "F32", "noData": str(NO_DATA),
                  "interpolation": "RSP_BilinearInterpolation", "f": "image"}
        if mosaic_rule is not None:
            params["mosaicRule"] = json.dumps(mosaic_rule, separators=(",", ":"))
        r = httpx.get(f"{url}/exportImage", params=params, timeout=timeout_s, headers={"User-Agent": user_agent})
        r.raise_for_status()
        if not r.content.startswith(TIFF_MAGIC):
            raise RuntimeError(f"exportImage answered {r.content[:200]!r}, not a GeoTIFF")
        with rasterio.MemoryFile(io.BytesIO(r.content)) as mem, mem.open() as ds:
            a = ds.read(1).astype(np.float32)
            if ds.nodata is not None:
                a[a == ds.nodata] = np.nan
        a[a <= NO_DATA + 1] = np.nan
        a[a < -1000] = np.nan
        return a

    return fetch
