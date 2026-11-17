"""Tests for the command that draws an isochrone map."""

import numpy as np
import pandas as pd

from isochrone.make_map import load_or_compute_access


def test_load_or_compute_access(small_graph, tmp_path):
    """The first call writes the cache; the second reads the same legs without the graph."""
    platforms = pd.DataFrame({"stop_id": ["a", "b"], "x": [200.0, 290.0], "y": [0.0, 0.0]})
    points = np.array([[140.0, 25.0], [0.0, -20.0], [300.0, 0.0], [100.0, 0.0]])
    cache_path = tmp_path / "processed" / "access.npz"

    computed = load_or_compute_access(cache_path, small_graph, points, platforms, 300.0)
    assert cache_path.exists()
    assert len(computed) > 0

    cached = load_or_compute_access(cache_path, None, None, None, 300.0)
    pd.testing.assert_frame_equal(cached, computed)
