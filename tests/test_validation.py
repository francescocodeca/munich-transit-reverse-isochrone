"""Tests for the comparison with r5py."""

import zipfile

from isochrone.validation import write_feed_without_empty_tables


def test_write_feed_without_empty_tables(tmp_path):
    """Tables with only a header are left out; the others are copied byte for byte."""
    feed_path = tmp_path / "feed.zip"
    tables = {
        "stops.txt": "\ufeffstop_id,stop_name\r\nA,Marienplatz\r\n".encode(),
        "shapes.txt": "\ufeffshape_id,shape_pt_lat\r\n".encode(),
        "transfers.txt": b"",
    }
    with zipfile.ZipFile(feed_path, "w") as feed:
        for name, content in tables.items():
            feed.writestr(name, content)

    out_path = tmp_path / "derived" / "feed_r5.zip"
    removed = write_feed_without_empty_tables(feed_path, out_path)
    assert removed == ["shapes.txt", "transfers.txt"]
    with zipfile.ZipFile(out_path) as copy:
        assert copy.namelist() == ["stops.txt"]
        assert copy.read("stops.txt") == tables["stops.txt"]
