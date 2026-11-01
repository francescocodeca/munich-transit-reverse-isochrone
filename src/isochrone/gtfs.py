"""Read tables of a GTFS feed stored as a ZIP file."""

import zipfile
from pathlib import Path

import pandas as pd


def read_gtfs_table(
    feed_path: Path, name: str, usecols: list[str] | None = None
) -> pd.DataFrame:
    """Read one table of a GTFS feed directly from the ZIP file.

    All columns are read as strings: GTFS identifiers may look like numbers but must
    keep leading zeros, and times can exceed 24:00:00, so conversions are done
    explicitly later. Some files of the MVV feed start with a byte order mark (BOM):
    the "utf-8-sig" encoding states explicitly that it must be removed.

    Args:
        feed_path: Path of the GTFS ZIP file.
        name: Table name without the ".txt" extension, e.g. "stops".
        usecols: Columns to read. If None, all columns are read.

    Returns:
        The table as a DataFrame where every column has string values.
    """
    with zipfile.ZipFile(feed_path) as feed, feed.open(f"{name}.txt") as f:
        return pd.read_csv(f, dtype=str, usecols=usecols, encoding="utf-8-sig")
