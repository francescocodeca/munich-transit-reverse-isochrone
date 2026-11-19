"""Command-line entry point that draws the cells reaching two destinations within two limits."""

import argparse
import datetime as dt
from pathlib import Path
from typing import NamedTuple

from isochrone.make_map import (
    MAPS_DIR,
    arrival_label,
    arrival_times_s,
    load_inputs,
    median_minutes,
    projected_destination,
    travel_minutes,
)
from isochrone.maps import band_pair_polygons, pair_map


class Destination(NamedTuple):
    """A destination of the map and the latest arrival time there.

    Attributes:
        name: Label written next to the slider of the destination.
        lat: Latitude of the destination.
        lon: Longitude of the destination.
        arrival: Latest arrival time at the destination (the centre of the window).
    """

    name: str
    lat: float
    lon: float
    arrival: dt.time


def parse_destination(text: str) -> Destination:
    """Read a destination written as NAME,LAT,LON,HH:MM.

    The name may contain commas: only the last three commas separate the fields.

    Args:
        text: The destination as given on the command line, e.g. "Garching,48.26,11.67,09:00".

    Returns:
        The destination.

    Raises:
        argparse.ArgumentTypeError: If text does not have four fields or a field is not valid.
    """
    fields = text.rsplit(",", 3)
    if len(fields) != 4 or not fields[0].strip():
        raise argparse.ArgumentTypeError(f"expected NAME,LAT,LON,HH:MM, got {text!r}")
    name, lat, lon, arrival = (field.strip() for field in fields)
    try:
        return Destination(name, float(lat), float(lon), dt.time.fromisoformat(arrival))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid destination {text!r}: {error}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line arguments.

    Args:
        argv: Arguments without the program name; None reads them from sys.argv.

    Returns:
        Namespace with attributes dest (list of two Destination), date (datetime.date),
        window (int, minutes), and feed, osm and output (Path, or None when not given).

    Raises:
        SystemExit: If the arguments are not valid, with a message on the standard error.
    """
    parser = argparse.ArgumentParser(
        prog="python -m isochrone.make_pair_map",
        description="Draw a map of the places in the Munich area from which two destinations "
        "can be reached by public transport and on foot within two limits chosen on the map.",
    )
    parser.add_argument(
        "--dest",
        type=parse_destination,
        action="append",
        required=True,
        metavar="NAME,LAT,LON,HH:MM",
        help="destination and latest arrival time there; give it exactly twice",
    )
    parser.add_argument(
        "--date",
        type=dt.date.fromisoformat,
        required=True,
        metavar="YYYY-MM-DD",
        help="service day",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        metavar="MIN",
        help="width of the window of arrival times centred on each arrival time, one per "
        "minute; the map uses the median travel time (default: 60; 0 for the arrival time only)",
    )
    parser.add_argument(
        "--feed", type=Path, help="GTFS feed (default: the most recent in data/raw)"
    )
    parser.add_argument(
        "--osm",
        type=Path,
        help="OpenStreetMap .osm.pbf extract (default: the most recent in data/raw)",
    )
    parser.add_argument(
        "--output", type=Path, help="HTML file (default: a name from the arguments, in data/maps)"
    )
    args = parser.parse_args(argv)
    if len(args.dest) != 2:
        parser.error(f"--dest must be given exactly twice, got {len(args.dest)}")
    if args.window < 0:
        parser.error(f"--window must not be negative, got {args.window}")
    return args


def main(argv: list[str] | None = None) -> Path:
    """Draw the map of the two destinations and arrival times given on the command line.

    Both destinations are checked before the travel times are computed, so that a wrong
    second destination does not cost the computation of the first. For each destination the
    travel time of a cell is its median over the window of arrival times (see make_map.main).

    Args:
        argv: Arguments without the program name, as for parse_args.

    Returns:
        Path of the HTML map.

    Raises:
        SystemExit: If the arguments are not valid, the date is outside the feed or a
            destination is outside the area of the OpenStreetMap extract.
    """
    args = parse_args(argv)
    inputs = load_inputs(args.feed, args.osm, args.date)
    destinations_xy = [projected_destination(inputs, dest.lon, dest.lat) for dest in args.dest]

    minutes, labels = [], []
    for dest, destination_xy in zip(args.dest, destinations_xy):
        arrivals_s = arrival_times_s(dest.arrival, args.window)
        travel = travel_minutes(
            inputs.graph,
            inputs.grid,
            inputs.platforms,
            inputs.access,
            inputs.network,
            destination_xy,
            arrivals_s,
        )
        minutes.append(median_minutes(travel))
        labels.append(f"{dest.name} ({arrival_label(arrivals_s)})")
    pairs = band_pair_polygons(inputs.grid, minutes[0], minutes[1])
    print(f"{pairs['n_cells'].sum()} of {len(inputs.grid)} grid cells reach both destinations")

    first, second = args.dest
    map_name = (
        f"pair_{first.lat:.4f}_{first.lon:.4f}_{first.arrival:%H%M}"
        f"_{second.lat:.4f}_{second.lon:.4f}_{second.arrival:%H%M}"
        f"_{args.date}_w{args.window}.html"
    )
    output = args.output or MAPS_DIR / map_name
    output.parent.mkdir(parents=True, exist_ok=True)
    destinations_lonlat = tuple((dest.lon, dest.lat) for dest in args.dest)
    pair_map(pairs, destinations_lonlat, tuple(labels)).save(output)
    print(f"map saved to {output}")
    return output


if __name__ == "__main__":
    main()
