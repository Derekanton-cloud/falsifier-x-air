from dataclasses import dataclass
import numpy as np

@dataclass
class TwinState:
    delays: dict

class AviationDigitalTwin:
    def __init__(self, flights, hidden_mechanisms=None, seed=7):
        self.flights = flights
        self.rng = np.random.default_rng(seed)
        self.hidden_mechanisms = set(
            hidden_mechanisms or {"AIRCRAFT_ROTATION"}
        )

    def run(self, weather=0.0, capacity=1.0, interventions=None):
        interventions = interventions or {}
        delays = {}
        for f in self.flights:
            noise = self.rng.normal(0, 1.5)
            base = 8.0 * weather + 12.0 * max(0.0, 1.0 - capacity)
            delays[f.flight_id] = max(0.0, base + noise)

        if "AIRCRAFT_ROTATION" in self.hidden_mechanisms:
            if not interventions.get("disable_aircraft_rotation", False):
                by_aircraft = {}
                for f in sorted(self.flights, key=lambda x: x.scheduled_time):
                    by_aircraft.setdefault(f.aircraft_id, []).append(f)
                for seq in by_aircraft.values():
                    for prev, nxt in zip(seq, seq[1:]):
                        delays[nxt.flight_id] += 0.85 * delays[prev.flight_id]

        if interventions.get("increase_capacity", False):
            for k in delays:
                delays[k] *= 0.65

        if interventions.get("disable_resource_dependency", False):
            for k in delays:
                delays[k] *= 0.97

        return TwinState(delays)

    def total_delay(self, **kwargs):
        return float(sum(self.run(**kwargs).delays.values()))
