"""Shared fixtures for the test suite."""

from pathlib import Path

import pytest

from isochrone.download import RAW_DATA_DIR


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
