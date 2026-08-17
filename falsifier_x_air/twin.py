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
    family: str = "STANDARD"
    noise_scale: float = 1.5
    mechanism_strength: float = 1.0
    regime: int = 0
    # Phase-8 extensions — None/default values reproduce Phase-7 behaviour exactly.
    transient_after: int | None = None      # mechanism only applies to scheduled_time >= this
    rotation_coefficient: float = 0.85     # propagation multiplier for AIRCRAFT_ROTATION
    resource_coefficient: float = 0.60     # propagation multiplier for RESOURCE_DEPENDENCY


class AviationDigitalTwin:
    """Controlled generative environment. Hidden mechanisms are never exposed to learners.

    ``observe`` and ``counterfactual`` are the only interfaces needed by the
    investigation system. Ground truth deliberately has no public accessor.
    """

    def __init__(self, flights: Iterable[Flight], hidden_mechanisms: Iterable[str] | None = None, seed: int = 7) -> None:
        self.flights = tuple(flights)
        self._hidden_mechanisms = frozenset(hidden_mechanisms or ())
        self.seed = seed

    def observe(self, scenario: TwinScenario, interventions: Mapping[str, bool] | None = None) -> NetworkObservation:
        interventions = interventions or {}
        rng = np.random.default_rng(scenario.seed)
        base = 8.0 * scenario.weather + 12.0 * max(0.0, 1.0 - scenario.capacity)
        delays = {flight.flight_id: max(0.0, base + rng.normal(0.0, scenario.noise_scale)) for flight in self.flights}

        if "AIRCRAFT_ROTATION" in self._hidden_mechanisms and not interventions.get("disable_aircraft_rotation", False):
            by_aircraft: dict[str, list[Flight]] = {}
            for flight in sorted(self.flights, key=lambda item: (item.aircraft_id, item.scheduled_time)):
                by_aircraft.setdefault(flight.aircraft_id, []).append(flight)
            for sequence in by_aircraft.values():
                for previous, following in zip(sequence, sequence[1:]):
                    if scenario.transient_after is None or following.scheduled_time >= scenario.transient_after:
                        delays[following.flight_id] += scenario.rotation_coefficient * scenario.mechanism_strength * delays[previous.flight_id]
        # These are latent generative components. Their parameters remain an
        # environment implementation detail; observations expose delays only.
        if "AIRPORT_CAPACITY" in self._hidden_mechanisms and not interventions.get("increase_capacity", False):
            congestion = 20.0 * scenario.mechanism_strength * max(0.0, 1.0 - scenario.capacity)
            delays = {key: value + congestion for key, value in delays.items()}
        if "RESOURCE_DEPENDENCY" in self._hidden_mechanisms and not interventions.get("relieve_resource_dependency", False):
            by_resource: dict[str, list[Flight]] = {}
            for flight in sorted(self.flights, key=lambda item: (item.resource_id or "", item.scheduled_time)):
                if flight.resource_id is not None:
                    by_resource.setdefault(flight.resource_id, []).append(flight)
            for sequence in by_resource.values():
                for previous, following in zip(sequence, sequence[1:]):
                    if scenario.transient_after is None or following.scheduled_time >= scenario.transient_after:
                        delays[following.flight_id] += scenario.resource_coefficient * scenario.mechanism_strength * delays[previous.flight_id]
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
                                  int(rng.integers(0, 2**31 - 1)), family="HELD_OUT") for index in range(count))

    def _benchmark_truth(self) -> Mapping[str, object]:
        """Private benchmark-only oracle. Learner modules must never call this."""
        affected = set()
        edges = set()
        ordered = sorted(self.flights, key=lambda item: (item.aircraft_id, item.scheduled_time))
        if "AIRCRAFT_ROTATION" in self._hidden_mechanisms:
            by_key: dict[str, list[Flight]] = {}
            for flight in ordered:
                by_key.setdefault(flight.aircraft_id, []).append(flight)
            for sequence in by_key.values():
                for left, right in zip(sequence, sequence[1:]):
                    affected.update((left.flight_id, right.flight_id)); edges.add((left.flight_id, right.flight_id))
        if "RESOURCE_DEPENDENCY" in self._hidden_mechanisms:
            by_key = {}
            for flight in sorted(self.flights, key=lambda item: (item.resource_id or "", item.scheduled_time)):
                if flight.resource_id:
                    by_key.setdefault(flight.resource_id, []).append(flight)
            for sequence in by_key.values():
                for left, right in zip(sequence, sequence[1:]):
                    affected.update((left.flight_id, right.flight_id)); edges.add((left.flight_id, right.flight_id))
        if "AIRPORT_CAPACITY" in self._hidden_mechanisms:
            affected.update(f.flight_id for f in self.flights)
        return {"mechanisms": self._hidden_mechanisms, "affected_nodes": frozenset(affected), "affected_edges": frozenset(edges)}
