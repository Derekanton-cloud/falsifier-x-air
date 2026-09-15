"""Concise synthetic demonstration of the complete research loop."""

import numpy as np

from .falsifier import FalsifierXAir, InvestigationContext
from .graph import AviationGraph
from .predictor import BaselinePredictor
from .recovery import choose_recovery, evaluate_selected_action
from .schema import Flight
from .twin import AviationDigitalTwin, TwinScenario


def make_flights() -> tuple[Flight, ...]:
    return (
        Flight("F1", "A", "B", "AC1", 0), Flight("F2", "B", "C", "AC1", 1), Flight("F3", "C", "D", "AC1", 2),
        Flight("F4", "A", "C", "AC2", 0), Flight("F5", "C", "D", "AC2", 1), Flight("F6", "D", "A", "AC2", 2),
    )


def main() -> None:
    rng = np.random.default_rng(7)
    features = np.column_stack([rng.uniform(0, 1, 400), rng.uniform(0.6, 1, 400)])
    targets = 8 * features[:, 0] + 12 * (1 - features[:, 1]) + rng.normal(0, 1.5, 400)
    predictor = BaselinePredictor().fit(features, targets)
    alternative = BaselinePredictor(alpha=5.0).fit(features, targets)
    scenario = TwinScenario("demo", weather=0.9, capacity=0.7, seed=21)
    twin = AviationDigitalTwin(make_flights(), hidden_mechanisms={"AIRCRAFT_ROTATION"}, seed=21)
    observation = twin.observe(scenario)
    graph = AviationGraph.build(observation.flights)
    x_observation = np.tile([scenario.weather, scenario.capacity], (len(graph.flight_ids), 1))
    prediction = predictor.predict_with_uncertainty(x_observation)
    context = InvestigationContext(observation, graph, prediction,
                                   alternative.predict_with_uncertainty(x_observation).mean,
                                   float(np.mean(predictor.feature_distance(x_observation))), 3, scenario)
    result = FalsifierXAir().investigate(twin, context)
    print("\n=== FALSIFIER-X AIR v0.1 research foundation ===")
    print(f"Adequacy: {result.adequacy.state} (evidence={result.adequacy.evidence_score:.2f})")
    print("Affected flights:", ", ".join(result.affected_flights) or "none")
    print("Candidates:", ", ".join(item.mechanism_id for item in result.candidates) or "none")
    for experiment in result.experiment_results:
        print(f"Experiment {experiment.experiment_id}: {experiment.intervention.identifier}, effect={experiment.effect:.1f}")
    print("Recovered mechanism:", result.recovered_mechanism or "inconclusive")
    predicted_total = float(prediction.mean.sum())
    action_without_mechanism = choose_recovery(predicted_total, None)
    recovered_evidence = next((item for item in result.candidates if item.mechanism_id == result.recovered_mechanism), None)
    action_with_recovered_mechanism = choose_recovery(predicted_total, result.recovered_mechanism, recovered_evidence)
    print(f"Recovery choice (action without mechanism): {action_without_mechanism.action.identifier}")
    print(f"Recovery choice (action with recovered mechanism): {action_with_recovered_mechanism.action.identifier}")
    print(f"Environment outcome (action without mechanism choice): {evaluate_selected_action(twin, scenario, action_without_mechanism).realized_total_delay:.1f}")
    print(f"Environment outcome (action with recovered mechanism choice): {evaluate_selected_action(twin, scenario, action_with_recovered_mechanism).realized_total_delay:.1f}")


if __name__ == "__main__":
    main()
