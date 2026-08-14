"""Constrained mechanism registry and candidate generation."""

from .graph import AviationGraph, GraphSnapshot
from .schema import CounterfactualIntervention, Mechanism, MechanismEvidence


MECHANISM_REGISTRY = (
    Mechanism("AIRCRAFT_ROTATION", "flight", "AIRCRAFT_ROTATION", "flight",
              CounterfactualIntervention("disable_aircraft_rotation", "Break a scheduled rotation dependency", 1.0),
              "Delay propagates from one leg to the next leg of an aircraft rotation."),
    Mechanism("AIRPORT_CAPACITY", "airport", "DEPARTS_FROM", "flight",
              CounterfactualIntervention("increase_capacity", "Temporarily increase airport processing capacity", 1.5),
              "Airport congestion contributes to flight-delay accumulation."),
    Mechanism("RESOURCE_DEPENDENCY", "resource", "RESOURCE_DEPENDENCY", "flight",
              CounterfactualIntervention("relieve_resource_dependency", "Relieve a represented shared operational resource", 2.0),
              "A represented shared operational resource propagates disruption."),
)


def generate_candidates(snapshot: GraphSnapshot, affected_flights: tuple[str, ...]) -> tuple[MechanismEvidence, ...]:
    """Use local graph semantics to avoid unconstrained latent-cause proposals."""
    relations = AviationGraph.relation_types(snapshot, affected_flights)
    allowed = []
    for mechanism in MECHANISM_REGISTRY:
        if mechanism.relation_type in relations:
            allowed.append(MechanismEvidence(mechanism.identifier))
    return tuple(allowed)


def mechanism_by_id(identifier: str) -> Mechanism:
    return next(mechanism for mechanism in MECHANISM_REGISTRY if mechanism.identifier == identifier)
