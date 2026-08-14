"""Synthetic aviation environment with a strict observation/intervention boundary."""

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .schema import Flight, NetworkObservation


@dataclass(frozen=True)
class TwinScenario:
    scenario_id: str
    weather: float
    capacity: float
    seed: int


class AviationDigitalTwin:
    """Controlled generative environment. Hidden mechanisms are never exposed to learners.

    ``observe`` and ``counterfactual`` are the only interfaces needed by the
    investigation system. Ground truth deliberately has no public accessor.
    """

    def __init__(self, flights: Iterable[Flight], hidden_mechanisms: Iterable[str] | None = None, seed: int = 7) -> None:
        self.flights = tuple(flights)
        self._hidden_mechanisms = frozenset(hidden_mechanisms or {"AIRCRAFT_ROTATION"})
        self.seed = seed

    def observe(self, scenario: TwinScenario, interventions: Mapping[str, bool] | None = None) -> NetworkObservation:
        interventions = interventions or {}
        rng = np.random.default_rng(scenario.seed)
        base = 8.0 * scenario.weather + 12.0 * max(0.0, 1.0 - scenario.capacity)
        delays = {flight.flight_id: max(0.0, base + rng.normal(0.0, 1.5)) for flight in self.flights}

        if "AIRCRAFT_ROTATION" in self._hidden_mechanisms and not interventions.get("disable_aircraft_rotation", False):
            by_aircraft: dict[str, list[Flight]] = {}
            for flight in sorted(self.flights, key=lambda item: (item.aircraft_id, item.scheduled_time)):
                by_aircraft.setdefault(flight.aircraft_id, []).append(flight)
            for sequence in by_aircraft.values():
                for previous, following in zip(sequence, sequence[1:]):
                    delays[following.flight_id] += 0.85 * delays[previous.flight_id]
        # These are latent generative components. Their parameters remain an
        # environment implementation detail; observations expose delays only.
        if "AIRPORT_CAPACITY" in self._hidden_mechanisms and not interventions.get("increase_capacity", False):
            congestion = 6.0 * max(0.0, 1.0 - scenario.capacity)
            delays = {key: value + congestion for key, value in delays.items()}
        if "RESOURCE_DEPENDENCY" in self._hidden_mechanisms and not interventions.get("relieve_resource_dependency", False):
            for flight in self.flights:
                if flight.scheduled_time > min(item.scheduled_time for item in self.flights):
                    delays[flight.flight_id] += 3.0
        return NetworkObservation(self.flights, delays, scenario.weather, scenario.capacity, scenario.scenario_id)

    def counterfactual(self, scenario: TwinScenario, intervention_id: str) -> NetworkObservation:
        return self.observe(scenario, {intervention_id: True})

    # Compatibility helpers retained for the v0.1 API.
    def run(self, weather: float = 0.0, capacity: float = 1.0, interventions: Mapping[str, bool] | None = None):
        observation = self.observe(TwinScenario("legacy", weather, capacity, self.seed), interventions)
        return type("TwinState", (), {"delays": observation.delays})()

    def total_delay(self, **kwargs: object) -> float:
        return float(sum(self.run(**kwargs).delays.values()))

    def held_out_scenarios(self, count: int, seed: int) -> tuple[TwinScenario, ...]:
        rng = np.random.default_rng(seed)
        return tuple(TwinScenario(f"held-out-{index}", float(rng.uniform(0.2, 1.0)), float(rng.uniform(0.55, 0.95)),
                                  int(rng.integers(0, 2**31 - 1))) for index in range(count))
