"""Read-only JSON bridge over existing public FALSIFIER-X investigation paths.

It returns observations, observable graph structure and public counterfactual
outcomes only. Benchmark truth, hidden mechanisms and evaluator certificates
never cross this process boundary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from falsifier_x_air.demo import make_flights
from falsifier_x_air.falsifier import FalsifierXAir, InvestigationContext
from falsifier_x_air.graph import AviationGraph
from falsifier_x_air.identifiability import EPSILON_IDENT, response_distance
from falsifier_x_air.mechanisms import mechanism_by_id
from falsifier_x_air.phase8_worlds import world_confounded
from falsifier_x_air.phase9_evaluation import _prediction as phase9_prediction
from falsifier_x_air.predictor import BaselinePredictor
from falsifier_x_air.twin import AviationDigitalTwin, TwinScenario


def _demo_context() -> tuple[AviationDigitalTwin, TwinScenario, InvestigationContext]:
    """The existing six-flight public demonstration from falsifier_x_air.demo."""
    rng = np.random.default_rng(7)
    features = np.column_stack([rng.uniform(0, 1, 400), rng.uniform(0.6, 1, 400)])
    targets = 8 * features[:, 0] + 12 * (1 - features[:, 1]) + rng.normal(0, 1.5, 400)
    predictor = BaselinePredictor().fit(features, targets)
    alternative = BaselinePredictor(alpha=5.0).fit(features, targets)
    scenario = TwinScenario("console-demo", weather=0.9, capacity=0.7, seed=21)
    twin = AviationDigitalTwin(make_flights(), hidden_mechanisms={"AIRCRAFT_ROTATION"}, seed=21)
    observation = twin.observe(scenario)
    graph = AviationGraph.build(observation.flights)
    observed_features = np.tile([scenario.weather, scenario.capacity], (len(graph.flight_ids), 1))
    prediction = predictor.predict_with_uncertainty(observed_features)
    context = InvestigationContext(observation, graph, prediction,
                                   alternative.predict_with_uncertainty(observed_features).mean,
                                   float(np.mean(predictor.feature_distance(observed_features))), 3, scenario)
    return twin, scenario, context


def _confounded_context() -> tuple[AviationDigitalTwin, TwinScenario, InvestigationContext]:
    """Existing Phase-8 confounded world, run only through learner-facing APIs."""
    twin, scenario, _certificate = world_confounded(3000)
    observation = twin.observe(scenario)
    graph = AviationGraph.build(observation.flights)
    prediction = phase9_prediction(twin, scenario)
    context = InvestigationContext(observation, graph, prediction, prediction.mean, 0.0, 3, scenario)
    return twin, scenario, context


def _trace(case: str) -> dict[str, object]:
    twin, scenario, context = _confounded_context() if case == "indistinguishable" else _demo_context()
    falsifier = FalsifierXAir()
    result = falsifier.investigate(twin, context)
    observation, graph, prediction = context.observation, context.graph, context.prediction
    residuals = observation.delay_vector(graph.flight_ids) - prediction.mean
    residual_by_flight = dict(zip(graph.flight_ids, residuals.tolist()))
    prediction_by_flight = dict(zip(graph.flight_ids, prediction.mean.tolist()))

    nodes = []
    for node_id, attributes in graph.graph.nodes(data=True):
        node_type = attributes.get("node_type", "unknown")
        entry: dict[str, object] = {"id": node_id, "type": node_type}
        if node_type == "flight":
            entry.update({"aircraft": attributes.get("aircraft_id", ""),
                          "delay": observation.delays.get(node_id, 0.0),
                          "prediction": prediction_by_flight.get(node_id, 0.0),
                          "residual": residual_by_flight.get(node_id, 0.0),
                          "time": next(f.scheduled_time for f in observation.flights if f.flight_id == node_id)})
        nodes.append(entry)
    edges = [{"source": source, "target": target, "relation": attributes.get("relation", "")}
             for source, target, attributes in graph.graph.edges(data=True)]

    experiments = [{"id": experiment.experiment_id, "intervention": experiment.intervention.identifier,
                    "description": experiment.intervention.description, "effect": experiment.effect,
                    "baseline": experiment.baseline_delay, "counterfactual": experiment.intervention_delay}
                   for experiment in result.experiment_results]
    effects = {item["intervention"]: float(item["effect"]) for item in experiments}
    survivor = result.recovered_mechanism or "AIRCRAFT_ROTATION"
    survivor_action = mechanism_by_id(survivor).intervention.identifier
    d_values = []
    for candidate in result.candidates:
        if candidate.mechanism_id == survivor:
            continue
        action = mechanism_by_id(candidate.mechanism_id).intervention.identifier
        if survivor_action in effects and action in effects:
            d_values.append({"survivor": survivor, "competitor": candidate.mechanism_id,
                             "distance": response_distance(effects[survivor_action], effects[action], float(prediction.epistemic_std.mean()))})

    return {"source": "public-digital-twin", "case": case, "scenario": scenario.scenario_id,
            "adequacy": {"state": result.adequacy.state, "reason": result.adequacy.reason,
                         "evidence": result.adequacy.evidence_score,
                         "standardized_residual": result.adequacy.standardized_residual,
                         "residual_z_threshold": falsifier.detector.config.residual_z_threshold,
                         "interval_miscoverage": result.adequacy.interval_miscoverage,
                         "persistence": result.adequacy.persistence_fraction,
                         "graph_concentration": result.adequacy.structural_concentration,
                         "ood_score": result.adequacy.ood_score, "channels": result.adequacy.channels},
            "graph": {"nodes": nodes, "edges": edges}, "affected_flights": result.affected_flights,
            "candidates": [{"id": item.mechanism_id, "status": item.status.name, "observations": item.observations,
                            "log_evidence": item.log_evidence} for item in result.candidates],
            "experiments": experiments, "recovered_mechanism": result.recovered_mechanism,
            "outcome": result.outcome,
            "identifiability": {"epsilon": EPSILON_IDENT, "reason": result.identifiability_reason, "distances": d_values}}


if __name__ == "__main__":
    requested_case = sys.argv[1] if len(sys.argv) > 1 else "identifiable"
    if requested_case not in {"identifiable", "indistinguishable"}:
        raise SystemExit("case must be identifiable or indistinguishable")
    print(json.dumps(_trace(requested_case)))
