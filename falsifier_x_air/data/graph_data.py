"""Dynamic, observed BTS graph snapshots; no causal claims or synthetic resources."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GraphBuildReport:
    nodes: int
    edges: int
    edge_type_counts: dict[str, int]
    temporal_violations: int


def build_dynamic_edges(flights: pd.DataFrame, minimum_rotation_minutes: int, maximum_rotation_minutes: int, airport_propagation_minutes: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    ordered = flights.drop_duplicates("flight_id").sort_values("scheduled_departure_utc")
    for row in ordered.itertuples(index=False):
        records.extend((
            {"source": row.flight_id, "target": f"airport:{row.origin_airport}", "edge_type": "FLIGHT_DEPARTS_FROM", "source_time": row.scheduled_departure_utc, "target_time": row.scheduled_departure_utc},
            {"source": row.flight_id, "target": f"airport:{row.destination_airport}", "edge_type": "FLIGHT_ARRIVES_AT", "source_time": row.scheduled_arrival_utc, "target_time": row.scheduled_arrival_utc},
        ))
    for _, group in ordered.dropna(subset=["tail_number", "actual_arrival_utc"]).groupby("tail_number", sort=False):
        prior = None
        for row in group.sort_values("scheduled_departure_utc").itertuples(index=False):
            if prior is not None and prior.destination_airport == row.origin_airport:
                gap = (row.scheduled_departure_utc - prior.actual_arrival_utc).total_seconds() / 60
                if minimum_rotation_minutes <= gap <= maximum_rotation_minutes:
                    records.append({"source": prior.flight_id, "target": row.flight_id, "edge_type": "AIRCRAFT_ROTATION", "source_time": prior.actual_arrival_utc, "target_time": row.scheduled_departure_utc})
            prior = row
    arrivals = ordered.dropna(subset=["actual_arrival_utc"])
    for airport, departures in ordered.groupby("origin_airport", sort=False):
        incoming = arrivals[arrivals.destination_airport == airport].sort_values("actual_arrival_utc")
        for departure in departures.sort_values("scheduled_departure_utc").itertuples(index=False):
            previous = incoming[(incoming.actual_arrival_utc < departure.scheduled_departure_utc) & ((departure.scheduled_departure_utc - incoming.actual_arrival_utc).dt.total_seconds() <= airport_propagation_minutes * 60)]
            if not previous.empty:
                arrival = previous.iloc[-1]
                if arrival.flight_id != departure.flight_id:
                    records.append({"source": arrival.flight_id, "target": departure.flight_id, "edge_type": "TEMPORAL_AIRPORT_PROPAGATION", "source_time": arrival.actual_arrival_utc, "target_time": departure.scheduled_departure_utc})
    return pd.DataFrame(records).drop_duplicates(["source", "target", "edge_type"]) if records else pd.DataFrame(columns=["source", "target", "edge_type", "source_time", "target_time"])


def graph_report(flights: pd.DataFrame, edges: pd.DataFrame) -> GraphBuildReport:
    nodes = set(flights.flight_id) | {value for value in edges.source if str(value).startswith("airport:")} | {value for value in edges.target if str(value).startswith("airport:")}
    temporal = edges[edges.edge_type.isin(["AIRCRAFT_ROTATION", "TEMPORAL_AIRPORT_PROPAGATION"])]
    violations = int((temporal.source_time >= temporal.target_time).sum())
    return GraphBuildReport(len(nodes), len(edges), {key: int(value) for key, value in edges.edge_type.value_counts().items()}, violations)


def chronological_split(flights: pd.DataFrame, train_end: str, validation_end: str) -> pd.Series:
    """Chronological labels; targets must never define the split."""
    timestamp = pd.to_datetime(flights["scheduled_departure_utc"], utc=True)
    train_cutoff = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1)
    validation_cutoff = pd.Timestamp(validation_end, tz="UTC") + pd.Timedelta(days=1)
    labels = pd.Series("test", index=flights.index, dtype="string")
    labels.loc[timestamp < validation_cutoff] = "validation"
    labels.loc[timestamp < train_cutoff] = "train"
    return labels.rename("split")
