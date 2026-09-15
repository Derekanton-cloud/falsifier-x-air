"""Phase 10 causal model repair using recovered mechanisms."""

import numpy as np

from .schema import NetworkObservation, ExperimentResult
from .graph import GraphSnapshot, AviationGraph
from .evaluation import run_case, scenario_family, _prediction
from .falsifier import FalsifierXAir, InvestigationContext
from .twin import AviationDigitalTwin, TwinScenario
from .mechanisms import mechanism_by_id

# Estimated mechanism coefficients (frozen after discovery)
_ESTIMATED_COEFFICIENTS: dict[str, float] = {}

def get_observable_feature(mechanism_id: str, observation: NetworkObservation, graph: GraphSnapshot) -> np.ndarray:
    """Return the per-flight observable feature added by the mechanism."""
    size = len(graph.flight_ids)
    feature = np.zeros(size)
    if mechanism_id == "AIRCRAFT_ROTATION":
        for source, target, data in graph.graph.edges(data=True):
            if data.get("relation") == "AIRCRAFT_ROTATION":
                idx = graph.flight_ids.index(target)
                feature[idx] += observation.delays[source]
    elif mechanism_id == "RESOURCE_DEPENDENCY":
        for source, target, data in graph.graph.edges(data=True):
            if data.get("relation") == "RESOURCE_DEPENDENCY":
                idx = graph.flight_ids.index(target)
                feature[idx] += observation.delays[source]
    elif mechanism_id == "AIRPORT_CAPACITY":
        feature += max(0.0, 1.0 - observation.capacity)
    return feature

def get_generic_feature(observation: NetworkObservation, graph: GraphSnapshot) -> np.ndarray:
    """Return a generic non-causal feature (node in-degree) to test augmentation."""
    size = len(graph.flight_ids)
    feature = np.zeros(size)
    for i, flight_id in enumerate(graph.flight_ids):
        feature[i] = graph.graph.in_degree(flight_id)
    return feature

def apply_repair(prediction_mean: np.ndarray, mechanism_id: str, coeff: float, observation: NetworkObservation, graph: GraphSnapshot) -> np.ndarray:
    """Repair the prediction using the specified mechanism and coefficient."""
    feature = get_observable_feature(mechanism_id, observation, graph)
    return prediction_mean + coeff * feature

def apply_generic_repair(prediction_mean: np.ndarray, coeff: float, observation: NetworkObservation, graph: GraphSnapshot) -> np.ndarray:
    """Repair using a generic feature that carries no causal claim (node degree)."""
    feature = get_generic_feature(observation, graph)
    return prediction_mean + coeff * feature

def estimate_coefficients() -> dict[str, float]:
    """Estimate coefficients from discovery seeds 100-109."""
    global _ESTIMATED_COEFFICIENTS
    if _ESTIMATED_COEFFICIENTS:
        return _ESTIMATED_COEFFICIENTS

    from .evaluation import DISCOVERY_SEEDS, SCENARIO_FAMILIES
    
    # mechanism_id -> [(feature_sum, effect)]
    data = {"AIRCRAFT_ROTATION": [], "RESOURCE_DEPENDENCY": [], "AIRPORT_CAPACITY": [], "GENERIC": []}
    
    for seed in DISCOVERY_SEEDS:
        for family in SCENARIO_FAMILIES:
            twin, scenario = scenario_family(family, seed)
            observation = twin.observe(scenario)
            graph = AviationGraph.build(observation.flights)
            prediction = _prediction(twin, scenario)
            ood_score = 5.0 if family == "OOD" else 0.0
            
            context = InvestigationContext(
                observation, graph, prediction, prediction.mean, ood_score, 3, scenario,
            )
            result = FalsifierXAir().investigate(twin, context)
            
            if result.recovered_mechanism and result.identifiability_reason == "LEGITIMATELY_IDENTIFIED":
                mech_id = result.recovered_mechanism
                mech = mechanism_by_id(mech_id)
                intervention_id = mech.intervention.identifier
                
                # Find the executed experiment effect
                effect = None
                for exp in result.experiment_results:
                    if exp.intervention.identifier == intervention_id:
                        effect = exp.effect
                        break
                if effect is None:
                    continue
                
                feature = get_observable_feature(mech_id, observation, graph)
                feature_sum = feature.sum()
                data[mech_id].append((feature_sum, effect))
                
                gen_feature = get_generic_feature(observation, graph)
                data["GENERIC"].append((gen_feature.sum(), effect))
                
    coeffs = {}
    for mech_id, points in data.items():
        if not points:
            coeffs[mech_id] = 0.0
            continue
        X = np.array([p[0] for p in points])
        Y = np.array([p[1] for p in points])
        # Least squares without intercept: coeff = sum(X*Y) / sum(X^2)
        denom = np.sum(X**2)
        coeffs[mech_id] = np.sum(X * Y) / denom if denom > 1e-9 else 0.0
        
    _ESTIMATED_COEFFICIENTS = coeffs
    return coeffs
