"""Shared fixtures for the test suite."""

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_array

from isochrone.download import RAW_DATA_DIR
from isochrone.osm import WalkingGraph


@pytest.fixture(scope="session")
def mvv_feed() -> Path:
    """Return the most recent MVV GTFS feed in the raw data folder.

    Raw data are not stored in the repository, so tests that need the feed are skipped
    when no feed has been downloaded (e.g. on GitHub Actions).
    """
    feeds = sorted(RAW_DATA_DIR.glob("mvv_gtfs_*.zip"))
    if not feeds:
        pytest.skip("MVV GTFS feed not downloaded: run python -m isochrone.download")
    return feeds[-1]


@pytest.fixture
def small_graph() -> WalkingGraph:
    """Path 0-1-2-3 along the x axis (100 m edges) and a separate edge 4-5 near node 1."""
    x = np.array([0.0, 100.0, 200.0, 300.0, 150.0, 160.0])
    y = np.array([0.0, 0.0, 0.0, 0.0, 30.0, 30.0])
    u = np.array([0, 1, 2, 4])
    v = np.array([1, 2, 3, 5])
    length = np.hypot(x[u] - x[v], y[u] - y[v])
    lengths = csr_array(
        (np.concatenate([length, length]), (np.concatenate([u, v]), np.concatenate([v, u]))),
        shape=(6, 6),
    )
    return WalkingGraph(np.arange(10, 16), x, y, lengths)
