"""The sign font: Overpass SemiBold, bundled in `core/fonts/` with its SIL Open Font License (`OFL.txt` beside it).

Overpass (Red Hat, github.com/RedHatOfficial/Overpass, release 3.0.5) is "inspired by Highway Gothic", the US family
Australia's AS 1744 sign alphabets follow. It isn't a faithful Series, and there's no open AS 1744 font (the traffic
lights research, 24 Nov 2026, Q7). Signs are drawn in it into textures: a track carries the rendered letters, never
the font file. The street-name blades (`core.signals`) and the speed signs (`core.furniture`) use it. The closures'
ROAD CLOSED sign and the police cars' lettering keep Pillow's built-in Aileron (CC0).
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from PIL import ImageFont

FONT_FILE = "Overpass-SemiBold.ttf"
LICENCE_FILE = "OFL.txt"


@lru_cache(maxsize=32)
def font(size: int) -> ImageFont.FreeTypeFont:
    """Overpass SemiBold at `size` pixels (the em)."""
    with resources.as_file(resources.files("mapcreator.core") / "fonts" / FONT_FILE) as path:
        return ImageFont.truetype(str(path), size=max(int(size), 1))


def fit(draw, text: str, width: float, height: float, *, start: int | None = None) -> ImageFont.FreeTypeFont:
    """The largest size whose rendering of `text` fits `width` x `height` pixels (its ink, not its em)."""
    size = int(start or height * 1.6)
    while size > 6:
        f = font(size)
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=f)
        if x1 - x0 <= width and y1 - y0 <= height:
            return f
        size -= 1
    return font(6)
