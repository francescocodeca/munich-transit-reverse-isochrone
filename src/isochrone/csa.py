"""Find latest departures to a destination with the reverse Connection Scan Algorithm."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Network:
    """Timetable and transfers of one service day, in the form used by the scan.

    Stops and trips are numbered 0, 1, ...; the connection lists are parallel (position k of
    each list describes connection k) and sorted in scan order.

    Attributes:
        stop_ids: stop_id of each stop index.
        stop_index: Stop index of each stop_id.
        n_trips: Number of trips.
        trip_ids: trip_id of each trip index.
        dep_stop: Stop index of the departure of each connection.
        arr_stop: Stop index of the arrival of each connection.
        trip: Trip index of each connection.
        dep_time: Departure time of each connection, seconds after midnight.
        arr_time: Arrival time of each connection, seconds after midnight.
        can_board: Whether boarding is allowed at dep_stop.
        can_alight: Whether alighting is allowed at arr_stop.
        incoming: For each stop u, the pairs (w, duration_s) of the transfers w -> u.
    """

    stop_ids: list[str]
    stop_index: dict[str, int]
    n_trips: int
    trip_ids: list[str]
    dep_stop: list[int]
    arr_stop: list[int]
    trip: list[int]
    dep_time: list[int]
    arr_time: list[int]
    can_board: list[bool]
    can_alight: list[bool]
    incoming: list[list[tuple[int, int]]]


def build_network(connections: pd.DataFrame, transfers: pd.DataFrame) -> Network:
    """Convert connections and transfers into the structures used by the reverse scan.

    Connections are sorted by decreasing dep_time and, for equal dep_time, by decreasing
    stop_sequence, so that the connections of one trip with equal times are scanned from the
    last to the first.

    Args:
        connections: DataFrame as returned by build_connections.
        transfers: DataFrame as returned by build_transfers.

    Returns:
        The Network of the given connections and transfers.
    """
    stop_ids = sorted(
        set(connections["dep_stop"])
        | set(connections["arr_stop"])
        | set(transfers["from_stop"])
        | set(transfers["to_stop"])
    )
    stop_index = {stop_id: i for i, stop_id in enumerate(stop_ids)}

    order = np.lexsort(
        (-connections["stop_sequence"].to_numpy(), -connections["dep_time"].to_numpy())
    )
    scan = connections.iloc[order]
    trip_codes, trip_ids = pd.factorize(scan["trip_id"])

    incoming = [[] for _ in stop_ids]
    for w, u, duration in zip(
        transfers["from_stop"].map(stop_index),
        transfers["to_stop"].map(stop_index),
        transfers["duration_s"],
    ):
        incoming[u].append((w, duration))

    return Network(
        stop_ids=stop_ids,
        stop_index=stop_index,
        n_trips=len(trip_ids),
        trip_ids=trip_ids.tolist(),
        dep_stop=scan["dep_stop"].map(stop_index).tolist(),
        arr_stop=scan["arr_stop"].map(stop_index).tolist(),
        trip=trip_codes.tolist(),
        dep_time=scan["dep_time"].tolist(),
        arr_time=scan["arr_time"].tolist(),
        can_board=scan["can_board"].tolist(),
        can_alight=scan["can_alight"].tolist(),
        incoming=incoming,
    )


@dataclass
class Scan:
    """Result of a reverse scan, with what is needed to rebuild the journeys.

    Attributes:
        latest_departure: Latest departure time from each stop index, seconds after midnight;
            -inf for stops from which no target can be reached.
        boarded: For each stop index, the connection (position in the scan order) whose
            departure is the latest departure from the stop; -1 if there is none.
        exit: For each trip index, a connection of the trip at whose arrival stop one can
            alight in time to continue; -1 if the trip reaches no target in time.
        next_stop: For each stop index, the stop whose departure set its latest arrival, to be
            reached with the transfer between the two; -1 if the latest arrival is the one
            given in the targets, or if there is none.
    """

    latest_departure: list[float]
    boarded: list[int]
    exit: list[int]
    next_stop: list[int]


def scan_latest_departures(
    network: Network, targets: dict[str, int], max_duration_s: int = 7200
) -> Scan:
    """Run the reverse Connection Scan and record how each latest departure is obtained.

    Connections are scanned by decreasing departure time. A connection is useful if its trip
    is already known to reach a target in time, or if one can alight at its arrival stop early
    enough to continue; the first such connection of a trip is recorded as its exit. The first
    useful connection boarded at a stop fixes the latest departure from that stop.

    Args:
        network: Network of the service day, as returned by build_network.
        targets: Latest arrival time (seconds after midnight) at each target stop_id, e.g.
            the arrival time minus the walking time from the stop to the destination.
        max_duration_s: Connections departing more than max_duration_s before the latest
            target time are not scanned.

    Returns:
        The Scan with the latest departures and the records of the journeys.

    Raises:
        ValueError: If targets is empty.
    """
    if not targets:
        raise ValueError("targets must contain at least one stop")

    n_stops = len(network.stop_ids)
    latest_arrival = [-math.inf] * n_stops
    for stop_id, time in targets.items():
        latest_arrival[network.stop_index[stop_id]] = time
    latest_departure = [-math.inf] * n_stops
    boarded = [-1] * n_stops
    next_stop = [-1] * n_stops
    exit = [-1] * network.n_trips

    end_time = max(targets.values())
    start_time = end_time - max_duration_s

    for k, (dep_stop, arr_stop, trip, dep_time, arr_time, can_board, can_alight) in enumerate(
        zip(
            network.dep_stop,
            network.arr_stop,
            network.trip,
            network.dep_time,
            network.arr_time,
            network.can_board,
            network.can_alight,
        )
    ):
        if dep_time > end_time:
            continue
        if dep_time < start_time:
            break
        if exit[trip] < 0:
            if not (can_alight and arr_time <= latest_arrival[arr_stop]):
                continue
            exit[trip] = k
        if not can_board or boarded[dep_stop] >= 0:
            continue
        boarded[dep_stop] = k
        latest_departure[dep_stop] = dep_time
        for from_stop, duration in network.incoming[dep_stop]:
            if dep_time - duration > latest_arrival[from_stop]:
                latest_arrival[from_stop] = dep_time - duration
                next_stop[from_stop] = dep_stop

    return Scan(latest_departure, boarded, exit, next_stop)


def latest_departures(
    network: Network, targets: dict[str, int], max_duration_s: int = 7200
) -> dict[str, int]:
    """Return, for each stop, the latest departure that reaches a target in time.

    Runs scan_latest_departures and keeps only the latest departures.

    Args:
        network: Network of the service day, as returned by build_network.
        targets: Latest arrival time (seconds after midnight) at each target stop_id, e.g.
            the arrival time minus the walking time from the stop to the destination.
        max_duration_s: Connections departing more than max_duration_s before the latest
            target time are not scanned.

    Returns:
        Latest departure time (seconds after midnight) from each stop_id from which a target
        can be reached; stops from which no target can be reached are not included.

    Raises:
        ValueError: If targets is empty.
    """
    scan = scan_latest_departures(network, targets, max_duration_s)
    return {
        network.stop_ids[stop]: scan.latest_departure[stop]
        for stop in range(len(network.stop_ids))
        if scan.boarded[stop] >= 0
    }


def journey(network: Network, scan: Scan, stop_id: str) -> pd.DataFrame:
    """Rebuild the journey that leaves stop_id at its latest departure.

    Starting from the connection that fixed the latest departure, the journey rides the trip
    to its exit, walks the transfer to the next stop, and repeats until the latest arrival is
    one given in the targets. Transfers that stay at the same stop are not listed as walks.

    Args:
        network: Network used for the scan.
        scan: Scan returned by scan_latest_departures for network.
        stop_id: stop_id where the journey starts.

    Returns:
        DataFrame with one row per leg and columns kind ("ride" or "walk"), from_stop,
        to_stop, dep_time, arr_time (seconds after midnight) and trip_id (missing for walks).
        The arr_time of a walk is the arrival plus the duration of the transfer.

    Raises:
        ValueError: If no target can be reached from stop_id.
    """
    stop = network.stop_index[stop_id]
    if scan.boarded[stop] < 0:
        raise ValueError(f"no target can be reached from {stop_id}")

    ids = network.stop_ids
    legs = []
    while stop >= 0:
        board = scan.boarded[stop]
        trip = network.trip[board]
        alight = scan.exit[trip]
        end = network.arr_stop[alight]
        arrival = network.arr_time[alight]
        legs.append(
            ("ride", ids[stop], ids[end], network.dep_time[board], arrival, network.trip_ids[trip])
        )
        stop = scan.next_stop[end]
        if stop >= 0 and stop != end:
            duration = dict(network.incoming[stop])[end]
            legs.append(("walk", ids[end], ids[stop], arrival, arrival + duration, None))

    columns = ["kind", "from_stop", "to_stop", "dep_time", "arr_time", "trip_id"]
    return pd.DataFrame(legs, columns=columns)
