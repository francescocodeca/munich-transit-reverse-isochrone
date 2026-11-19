"""Tests for the command that draws the map of two destinations."""

import argparse
import datetime as dt

import pytest

from isochrone.make_pair_map import Destination, parse_args, parse_destination

GARCHING = "Garching,48.2649,11.6712,09:00"
POOL = "Pool,48.1736,11.5514,19:00"


def test_parse_destination_reads_four_fields():
    """Spaces around the fields are dropped; commas in the name are kept."""
    assert parse_destination("Garching,48.2649,11.6712,09:00") == Destination(
        "Garching", 48.2649, 11.6712, dt.time(9, 0)
    )
    pool = parse_destination("Pool, Olympiapark, 48.1736, 11.5514, 19:00")
    assert pool.name == "Pool, Olympiapark"
    assert pool.arrival == dt.time(19, 0)


@pytest.mark.parametrize(
    "text",
    [
        "48.2649,11.6712,09:00",
        ",48.2649,11.6712,09:00",
        "Garching,north,11.6712,09:00",
        "Garching,48.2649,11.6712,9h",
    ],
)
def test_parse_destination_rejects_invalid_text(text):
    """A missing field, an empty name, a coordinate or a time that is not valid is rejected."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse_destination(text)


def test_parse_args_reads_two_destinations_with_default_window():
    """Two --dest in order, a window of 60 min by default, optional paths None."""
    args = parse_args(["--dest", GARCHING, "--dest", POOL, "--date", "2026-12-09"])
    assert [dest.name for dest in args.dest] == ["Garching", "Pool"]
    assert args.date == dt.date(2026, 12, 9)
    assert args.window == 60
    assert args.feed is None and args.osm is None and args.output is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--dest", GARCHING, "--date", "2026-12-09"],
        ["--dest", GARCHING, "--dest", POOL, "--dest", POOL, "--date", "2026-12-09"],
        ["--dest", GARCHING, "--dest", POOL, "--date", "2026-12-09", "--window", "-10"],
        ["--dest", "Garching,48.2649,11.6712", "--dest", POOL, "--date", "2026-12-09"],
    ],
)
def test_parse_args_rejects_invalid_arguments(argv, capsys):
    """One or three destinations, a negative window or a bad destination stop the program."""
    with pytest.raises(SystemExit) as stop:
        parse_args(argv)
    assert stop.value.code == 2
    assert "usage:" in capsys.readouterr().err
