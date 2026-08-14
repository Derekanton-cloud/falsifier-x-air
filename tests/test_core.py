import inspect

import numpy as np
import pytest

from falsifier_x_air.adequacy import StructuralAdequacyDetector
from falsifier_x_air.experiments import ActiveExperimentSelector
from falsifier_x_air.falsifier import FalsifierXAir, InvestigationContext
from falsifier_x_air.graph import AviationGraph
from falsifier_x_air.mechanisms import generate_candidates
from falsifier_x_air.predictor import BaselinePredictor
from falsifier_x_air.recovery import choose_recovery, evaluate_selected_action
from falsifier_x_air.schema import Flight, MechanismEvidence, PredictionOutput
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
    evidence = MechanismEvidence("AIRCRAFT_ROTATION", observed_effects=[40.0], baseline_delays=[100.0],
                                 experiment_interventions=["disable_aircraft_rotation"])
    decision = choose_recovery(100.0, "AIRCRAFT_ROTATION", evidence)
    assert decision.action.identifier == "DISABLE_AIRCRAFT_ROTATION"
    evaluated = evaluate_selected_action(twin, scenario, decision)
    assert evaluated.realized_total_delay is not None


def test_resource_dependency_candidate_requires_observable_shared_resource_relation():
    resource_flights = (
        Flight("F1", "A", "B", "AC1", 0, "GATE-1"),
        Flight("F2", "B", "C", "AC2", 1, "GATE-1"),
    )
    snapshot = AviationGraph.build(resource_flights)
    candidates = generate_candidates(snapshot, snapshot.flight_ids)
    assert "RESOURCE_DEPENDENCY" in {candidate.mechanism_id for candidate in candidates}

    no_resource_snapshot = AviationGraph.build(tuple(Flight(flight.flight_id, flight.origin, flight.destination,
                                                    flight.aircraft_id, flight.scheduled_time) for flight in resource_flights))
    assert "RESOURCE_DEPENDENCY" not in {candidate.mechanism_id for candidate in generate_candidates(no_resource_snapshot, no_resource_snapshot.flight_ids)}


def test_resource_dependency_is_sequentially_investigated_and_recovered():
    resource_flights = (
        Flight("F1", "A", "B", "AC1", 0, "CREW-1"),
        Flight("F2", "B", "C", "AC2", 1, "CREW-1"),
    )
    snapshot = AviationGraph.build(resource_flights)
    scenario = TwinScenario("resource", 0.9, 0.7, 1)
    twin = AviationDigitalTwin(resource_flights, {"RESOURCE_DEPENDENCY"})
    observation = twin.observe(scenario)
    prediction = PredictionOutput(np.zeros(2), np.ones(2), np.ones(2), np.ones(2))
    result = FalsifierXAir().investigate(twin, InvestigationContext(observation, snapshot, prediction, np.zeros(2), 0.0, 3, scenario))
    assert any(experiment.intervention.identifier == "relieve_resource_dependency" for experiment in result.experiment_results)
    assert result.recovered_mechanism == "RESOURCE_DEPENDENCY"


def test_recovery_uses_measured_effect_not_default_assumption():
    evidence = MechanismEvidence("RESOURCE_DEPENDENCY", observed_effects=[72.0], baseline_delays=[100.0],
                                 experiment_interventions=["relieve_resource_dependency"])
    decision = choose_recovery(100.0, "RESOURCE_DEPENDENCY", evidence)
    assert decision.action.estimated_effect_fraction == 0.72
    assert decision.estimated_total_delay == pytest.approx(28.0)
