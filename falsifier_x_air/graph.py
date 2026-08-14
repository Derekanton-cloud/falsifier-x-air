"""Construction and residual analysis for the observable aviation graph."""

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np

from .schema import Flight


@dataclass(frozen=True)
class GraphSnapshot:
    graph: nx.MultiDiGraph
    flight_ids: tuple[str, ...]


class AviationGraph:
    """Semantic graph of scheduled operational dependencies, not hidden causality."""

    @classmethod
    def build(cls, flights: Iterable[Flight]) -> GraphSnapshot:
        flights = tuple(flights)
        graph = nx.MultiDiGraph()
        for flight in flights:
            graph.add_node(flight.flight_id, node_type="flight", aircraft_id=flight.aircraft_id)
            for airport, relation in ((flight.origin, "DEPARTS_FROM"), (flight.destination, "ARRIVES_AT")):
                graph.add_node(airport, node_type="airport")
                graph.add_edge(flight.flight_id, airport, relation=relation)
            if flight.resource_id is not None:
                resource_node = f"resource:{flight.resource_id}"
                graph.add_node(resource_node, node_type="resource", resource_id=flight.resource_id)
                graph.add_edge(flight.flight_id, resource_node, relation="USES_RESOURCE")

        by_aircraft: dict[str, list[Flight]] = {}
        for flight in sorted(flights, key=lambda item: (item.aircraft_id, item.scheduled_time)):
            by_aircraft.setdefault(flight.aircraft_id, []).append(flight)
        for sequence in by_aircraft.values():
            for previous, following in zip(sequence, sequence[1:]):
                graph.add_edge(previous.flight_id, following.flight_id, relation="AIRCRAFT_ROTATION")
        by_resource: dict[str, list[Flight]] = {}
        for flight in sorted(flights, key=lambda item: (item.resource_id or "", item.scheduled_time)):
            if flight.resource_id is not None:
                by_resource.setdefault(flight.resource_id, []).append(flight)
        for sequence in by_resource.values():
            for previous, following in zip(sequence, sequence[1:]):
                graph.add_edge(previous.flight_id, following.flight_id, relation="RESOURCE_DEPENDENCY")
        return GraphSnapshot(graph=graph, flight_ids=tuple(f.flight_id for f in flights))

    @staticmethod
    def localize(snapshot: GraphSnapshot, residuals: np.ndarray, quantile: float = 0.75) -> tuple[str, ...]:
        """Return high-residual flights plus their one-hop flight neighbours."""
        threshold = float(np.quantile(np.abs(residuals), quantile))
        seeds = [flight_id for flight_id, residual in zip(snapshot.flight_ids, residuals) if abs(residual) >= threshold]
        selected = set(seeds)
        for flight_id in seeds:
            selected.update(node for node in snapshot.graph.successors(flight_id) if snapshot.graph.nodes[node].get("node_type") == "flight")
            selected.update(node for node in snapshot.graph.predecessors(flight_id) if snapshot.graph.nodes[node].get("node_type") == "flight")
        return tuple(sorted(selected))

    @staticmethod
    def residual_concentration(snapshot: GraphSnapshot, residuals: np.ndarray) -> float:
        """Fraction of residual mass aligned with observable operational dependencies."""
        values = dict(zip(snapshot.flight_ids, np.abs(residuals)))
        total = float(sum(values.values()))
        if total == 0:
            return 0.0
        connected = 0.0
        for source, target, data in snapshot.graph.edges(data=True):
            if data.get("relation") in {"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY"}:
                connected += min(values.get(source, 0.0), values.get(target, 0.0))
        return float(np.clip(connected / total, 0.0, 1.0))

    @staticmethod
    def relation_types(snapshot: GraphSnapshot, flight_ids: Iterable[str]) -> set[str]:
        node_set = set(flight_ids)
        return {
            data["relation"] for source, target, data in snapshot.graph.edges(data=True)
            if source in node_set or target in node_set
        }
