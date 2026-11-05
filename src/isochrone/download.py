"""Download raw input data (GTFS feeds, OpenStreetMap extracts) into the local data directory."""

import shutil
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import osmium


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


def osm_timestamp(pbf_path: Path) -> str | None:
    """Return the time up to which an OpenStreetMap extract contains data.

    Args:
        pbf_path: Path of the .osm.pbf file.

    Returns:
        The replication timestamp in the file header, e.g. "2026-11-03T21:24:23Z", or None if
        the header has none.

    Raises:
        RuntimeError: If the file is not a valid PBF file (raised by osmium).
    """
    reader = osmium.io.Reader(str(pbf_path), osmium.osm.osm_entity_bits.NOTHING)
    timestamp = reader.header().get("osmosis_replication_timestamp")
    reader.close()
    return timestamp or None


GEOFABRIK_URL = "https://download.geofabrik.de/europe/germany/bayern/oberbayern"


def download_osm_extract(
    base_url: str = GEOFABRIK_URL, data_dir: Path = RAW_DATA_DIR, name: str = "oberbayern"
) -> tuple[Path, Path]:
    """Download a Geofabrik OpenStreetMap extract and the polygon that bounds it.

    The download date is written in the file names, because the ODbL attribution should state
    the date of the data, and because Geofabrik publishes a new extract every day.

    Args:
        base_url: Address of the extract without extension; "-latest.osm.pbf" and ".poly" are
            appended. Defaults to the Geofabrik extract of Oberbayern.
        data_dir: Folder where the files are saved. Defaults to data/raw in the repository.
        name: Prefix of the saved file names.

    Returns:
        The paths of the extract and of the polygon, e.g.
        data/raw/oberbayern_2026-11-05.osm.pbf and data/raw/oberbayern_2026-11-05.poly.

    Raises:
        RuntimeError: If the extract is not a valid PBF file (raised by osmium).
        ValueError: If the extract has no replication timestamp in its header, i.e. it is not
            a Geofabrik extract.
    """
    stem = f"{name}_{date.today().isoformat()}"
    pbf_path = download_file(f"{base_url}-latest.osm.pbf", Path(data_dir) / f"{stem}.osm.pbf")
    poly_path = download_file(f"{base_url}.poly", Path(data_dir) / f"{stem}.poly")
    if osm_timestamp(pbf_path) is None:
        raise ValueError(f"{pbf_path} has no replication timestamp: check the download URL.")
    return pbf_path, poly_path


if __name__ == "__main__":
    targets = sys.argv[1:] or ["gtfs", "osm"]
    paths = []
    if "gtfs" in targets:
        paths.append(download_mvv_gtfs())
    if "osm" in targets:
        paths.extend(download_osm_extract())
        print(f"OpenStreetMap data up to {osm_timestamp(paths[-2])}")
    for path in paths:
        print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
