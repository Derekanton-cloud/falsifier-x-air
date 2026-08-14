import inspect

import numpy as np

from falsifier_x_air.adequacy import StructuralAdequacyDetector
from falsifier_x_air.experiments import ActiveExperimentSelector
from falsifier_x_air.falsifier import FalsifierXAir, InvestigationContext
from falsifier_x_air.graph import AviationGraph
from falsifier_x_air.mechanisms import generate_candidates
from falsifier_x_air.predictor import BaselinePredictor
from falsifier_x_air.recovery import choose_recovery, evaluate_selected_action
from falsifier_x_air.schema import Flight, PredictionOutput
from falsifier_x_air.twin import AviationDigitalTwin, TwinScenario


def flights():
    return (Flight("F1", "A", "B", "AC1", 0), Flight("F2", "B", "C", "AC1", 1), Flight("F3", "C", "D", "AC1", 2))


def test_graph_has_semantic_rotation_and_airport_edges():
    graph = AviationGraph.build(flights()).graph
    relations = {data["relation"] for _, _, data in graph.edges(data=True)}
    assert {"AIRCRAFT_ROTATION", "DEPARTS_FROM", "ARRIVES_AT"} <= relations


def test_interval_and_predictor_protocol_output():
    rng = np.random.default_rng(1)
    features = rng.random((50, 2))
    model = BaselinePredictor().fit(features, 8 * features[:, 0] + 12 * (1 - features[:, 1]))
    prediction = model.predict_with_uncertainty(features[:3])
    assert len(prediction.mean) == 3 and np.all(prediction.upper >= prediction.lower)


def test_ood_gate_prevents_structural_claim():
    detector = StructuralAdequacyDetector()
    prediction = PredictionOutput(np.array([20.0]), np.array([10.0]), np.array([30.0]), np.array([1.0]))
    result = detector.evaluate(np.array([100.0]), prediction, 5.0, 10, 1.0, np.array([20.0]))
    assert result.state == "ADEQUATE"


def test_candidate_generation_is_constrained_by_observable_graph():
    snapshot = AviationGraph.build(flights())
    candidates = generate_candidates(snapshot, ("F1", "F2"))
    assert {candidate.mechanism_id for candidate in candidates} == {"AIRCRAFT_ROTATION", "AIRPORT_CAPACITY"}


def test_active_selector_returns_one_untried_intervention():
    snapshot = AviationGraph.build(flights())
    scenario = TwinScenario("test", 0.8, 0.7, 3)
    twin = AviationDigitalTwin(flights(), {"AIRCRAFT_ROTATION"})
    observation = twin.observe(scenario)
    candidates = generate_candidates(snapshot, snapshot.flight_ids)
    selected = ActiveExperimentSelector().select(candidates, snapshot, observation, np.zeros(3), set())
    assert selected in {"disable_aircraft_rotation", "increase_capacity"}


def test_sequential_investigation_respects_budget_and_updates_evidence():
    snapshot = AviationGraph.build(flights())
    scenario = TwinScenario("test", 0.9, 0.7, 1)
    twin = AviationDigitalTwin(flights(), {"AIRCRAFT_ROTATION"})
    observation = twin.observe(scenario)
    prediction = PredictionOutput(np.zeros(3), np.ones(3), np.ones(3), np.ones(3))
    result = FalsifierXAir().investigate(twin, InvestigationContext(observation, snapshot, prediction, np.zeros(3), 0.0, 3, scenario))
    assert result.adequacy.state == "STRUCTURALLY_SUSPICIOUS"
    assert 1 <= len(result.experiment_results) <= 3
    assert any(item.observations for item in result.candidates)


def test_twin_hidden_ground_truth_is_not_learner_api():
    twin = AviationDigitalTwin(flights(), {"AIRCRAFT_ROTATION"})
    assert not hasattr(twin, "hidden_mechanisms")
    assert "hidden_mechanisms" not in inspect.signature(choose_recovery).parameters


def test_recovery_decision_is_model_based_then_evaluated():
    twin = AviationDigitalTwin(flights(), {"AIRCRAFT_ROTATION"})
    scenario = TwinScenario("test", 0.9, 0.7, 1)
    decision = choose_recovery(100.0, "AIRCRAFT_ROTATION")
    assert decision.action.identifier == "DISABLE_AIRCRAFT_ROTATION"
    evaluated = evaluate_selected_action(twin, scenario, decision)
    assert evaluated.realized_total_delay is not None
