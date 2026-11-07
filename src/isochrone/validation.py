"""Compare the travel times of the isochrone map with those of r5py (R5 routing engine)."""

import shutil
import zipfile
from pathlib import Path


def write_feed_without_empty_tables(feed_path: Path, out_path: Path) -> list[str]:
    """Copy a GTFS feed, leaving out the tables that have a header but no rows.

    R5 refuses a feed with an empty table (e.g. the MVV feed has a shapes.txt with only the
    header), while an absent optional table is allowed. The other tables are copied unchanged.

    Args:
        feed_path: Path of the GTFS zip file.
        out_path: Path of the zip file to write; its folder is created if needed.

    Returns:
        The names of the tables left out, e.g. ["shapes.txt"].
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    removed = []
    with zipfile.ZipFile(feed_path) as source, zipfile.ZipFile(
        out_path, "w", zipfile.ZIP_DEFLATED
    ) as target:
        for info in source.infolist():
            with source.open(info) as table:
                table.readline()
                if not table.readline().strip():
                    removed.append(info.filename)
                    continue
            with source.open(info) as table, target.open(info.filename, "w") as copy:
                shutil.copyfileobj(table, copy)
    return removed
