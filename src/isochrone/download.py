"""Download raw input data (GTFS feeds) into the local data directory."""

import shutil
import urllib.request
import zipfile
from datetime import date
from pathlib import Path


def download_file(url: str, dest: Path) -> Path:
    """Download a file from a URL and save it to a local path.

    The file is first written to a temporary ".part" file and renamed only when the
    download is complete, so an interrupted download never leaves a truncated file
    under the final name.

    Args:
        url: Address of the file to download.
        dest: Local path where the file is saved. Missing parent folders are created.

    Returns:
        The path of the downloaded file.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dest)
    return dest


MVV_GTFS_URL = (
    "https://www.mvv-muenchen.de/fileadmin/mediapool/developer/opendata/gesamt_gtfs.zip"
)
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_mvv_gtfs(url: str = MVV_GTFS_URL, data_dir: Path = RAW_DATA_DIR) -> Path:
    """Download the MVV "Gesamt" GTFS feed and check that it is a valid ZIP file.

    The download date is written in the file name, because the MVV license (CC BY 4.0)
    requires it in the attribution, and because the feed changes every few weeks.

    Args:
        url: Address of the feed. Defaults to the official MVV download link.
        data_dir: Folder where the feed is saved. Defaults to data/raw in the repository.

    Returns:
        The path of the downloaded feed, e.g. data/raw/mvv_gtfs_2026-11-01.zip.

    Raises:
        ValueError: If the downloaded file is not a ZIP archive.
    """
    dest = Path(data_dir) / f"mvv_gtfs_{date.today().isoformat()}.zip"
    download_file(url, dest)
    if not zipfile.is_zipfile(dest):
        raise ValueError(f"{dest} is not a ZIP file: check the download URL.")
    return dest


if __name__ == "__main__":
    path = download_mvv_gtfs()
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
