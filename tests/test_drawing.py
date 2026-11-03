"""The map, outline and preview images draw a simplified centreline."""

import numpy as np


def test_drawn_lines_are_simplified_but_keep_their_shape():
    import shapely

    from mapcreator.core import package

    s = np.linspace(0, 1, 63_000)
    px = np.column_stack([800 + 700 * np.cos(4 * s), 800 + 700 * np.sin(6 * s)])
    pts = package._drawable(px)
    assert len(pts) < 2000
    assert shapely.LineString(pts).hausdorff_distance(shapely.LineString(px)) <= 0.5 + 1e-9
    assert pts[0] == tuple(px[0]) and pts[-1] == tuple(px[-1])
