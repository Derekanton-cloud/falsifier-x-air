import networkx as nx
from dataclasses import dataclass

@dataclass
class AviationGraph:
    graph: nx.MultiDiGraph

    @classmethod
    def build(cls, flights, include_hidden_mechanism=False):
        g = nx.MultiDiGraph()
        for f in flights:
            g.add_node(f.flight_id, type="flight",
                       origin=f.origin, destination=f.destination,
                       aircraft_id=f.aircraft_id)
        by_aircraft = {}
        for f in sorted(flights, key=lambda x: x.scheduled_time):
            by_aircraft.setdefault(f.aircraft_id, []).append(f)
        if include_hidden_mechanism:
            for seq in by_aircraft.values():
                for a, b in zip(seq, seq[1:]):
                    g.add_edge(a.flight_id, b.flight_id,
                               relation="AIRCRAFT_ROTATION")
        return cls(g)

    def local_subgraph(self, flight_ids):
        return self.graph.subgraph(flight_ids).copy()
