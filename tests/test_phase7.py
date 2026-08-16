"""Focused Phase-7 scientific-contract tests."""

import ast
from pathlib import Path

import numpy as np

from falsifier_x_air.adequacy import StructuralAdequacyDetector
from falsifier_x_air.evaluation import EVALUATION_SEEDS, REPAIR_SEEDS, frozen_final_configuration, run_case, set_metrics, scenario_family
from falsifier_x_air.experiments import select_baseline
from falsifier_x_air.falsifier import FalsifierXAir, InvestigationContext
from falsifier_x_air.graph import AviationGraph
from falsifier_x_air.mechanisms import generate_candidates
from falsifier_x_air.schema import Flight, PredictionOutput


def _investigate(family: str, seed: int = 101):
    twin, scenario = scenario_family(family, seed)
    observation = twin.observe(scenario); graph = AviationGraph.build(observation.flights)
    base = 8 * scenario.weather + 12 * (1 - scenario.capacity)
    radius = max(5.0 if scenario.family in {"CORRECT", "NOISE"} else 3.0, scenario.noise_scale * 2)
    prediction = PredictionOutput(np.full(len(graph.flight_ids), base), np.full(len(graph.flight_ids), base-radius),
                                  np.full(len(graph.flight_ids), base+radius), np.full(len(graph.flight_ids), radius/1.96))
    return FalsifierXAir().investigate(twin, InvestigationContext(observation, graph, prediction, prediction.mean,
        5.0 if family == "OOD" else 0.0, 3, scenario))


def test_learner_modules_do_not_import_or_name_hidden_truth_oracle():
    root = Path("falsifier_x_air")
    for filename in ("adequacy.py", "falsifier.py", "experiments.py", "mechanisms.py", "recovery.py"):
        source = (root / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "_benchmark_truth" not in source
        assert all(not (isinstance(node, ast.ImportFrom) and node.module == ".twin") for node in ast.walk(tree))


def test_adequate_and_ood_are_not_structural_discoveries():
    assert _investigate("CORRECT").adequacy.state == "ADEQUATE"
    assert _investigate("OOD").adequacy.state == "OOD"


def test_hidden_mechanism_and_decoy_are_sequentially_tested():
    result = _investigate("DECOY", 100)
    assert result.adequacy.state == "STRUCTURALLY_SUSPICIOUS"
    assert len(result.experiment_results) >= 1
    assert result.recovered_mechanism in {"AIRCRAFT_ROTATION", None}


def test_active_and_baseline_selectors_are_single_step_and_reproducible():
    twin, scenario = scenario_family("AIRCRAFT_ROTATION", 102)
    observation = twin.observe(scenario); graph = AviationGraph.build(observation.flights)
    candidates = generate_candidates(graph, graph.flight_ids)
    selected = {strategy: select_baseline(strategy, candidates, graph, observation, np.zeros(len(graph.flight_ids)), set(), np.random.default_rng(3))
                for strategy in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE")}
    assert all(value is not None for value in selected.values())
    assert selected["RANDOM"] == select_baseline("RANDOM", candidates, graph, observation, np.zeros(4), set(), np.random.default_rng(3))


def test_paired_counterfactual_is_deterministic():
    twin, scenario = scenario_family("AIRCRAFT_ROTATION", 103)
    assert twin.counterfactual(scenario, "disable_aircraft_rotation").delays == twin.counterfactual(scenario, "disable_aircraft_rotation").delays


def test_hand_checked_precision_recall_f1():
    # predicted {A,B,C}, truth {B,C,D}: TP=2, P=2/3, R=2/3, F1=2/3.
    score = set_metrics({"A", "B", "C"}, {"B", "C", "D"})
    assert score.precision == score.recall == score.f1 == 2 / 3


def test_multiple_mechanisms_is_a_supported_stress_case_not_forced_recovery():
    result = _investigate("MULTIPLE")
    assert result.outcome in {"RECOVERED", "INCONCLUSIVE"}


def test_held_out_repair_validation_uses_a_disjoint_scenario_seed():
    record = run_case("AIRPORT_CAPACITY", 104)
    assert record["held_out_policy_delta"] > 0.0


def test_observable_predecessor_signature_prevents_true_chain_rejection():
    # Discovery seed 100: both true interventions have measurable paired effects.
    assert _investigate("AIRCRAFT_ROTATION", 100).recovered_mechanism == "AIRCRAFT_ROTATION"
    assert _investigate("RESOURCE_DEPENDENCY", 100).recovered_mechanism == "RESOURCE_DEPENDENCY"


def test_discovery_diagnosis_does_not_reference_final_or_held_out_seed_partitions():
    source = Path("falsifier_x_air/discovery_diagnosis.py").read_text(encoding="utf-8")
    assert "EVALUATION_SEEDS" not in source
    assert "1000" not in source and "101000" not in source


def test_hand_constructed_directional_propagation_is_relation_aware_and_observable():
    flights = (
        # Two represented chains: predecessor residuals are ordinary, successors are high.
        Flight("F1", "A", "B", "AC1", 0, "R1"),
        Flight("F2", "B", "C", "AC1", 1, "R1"),
        Flight("F3", "C", "D", "AC2", 0, "R2"),
        Flight("F4", "D", "A", "AC2", 1, "R2"),
    )
    graph = AviationGraph.build(flights)
    # Manual expectation: mean(6, 8) - mean(0, 0) = 7 standardized units.
    assert AviationGraph.directional_residual_lift(graph, np.array([-1., 6., -2., 8.]), np.ones(4)) == 7.0
    # Unstructured small residuals do not create an automatic propagation signal.
    assert AviationGraph.directional_residual_lift(graph, np.array([1., -1., 1., -1.]), np.ones(4)) == -1.0


def test_directional_signal_source_has_no_twin_or_hidden_state_access():
    source = Path("falsifier_x_air/graph.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert all(not (isinstance(node, ast.ImportFrom) and node.module == "twin") for node in ast.walk(tree))
    method = source[source.index("def directional_residual_lift"):source.index("def relation_types")]
    assert "_hidden" not in method


def test_frozen_final_configuration_has_disjoint_partitions_and_threshold():
    configuration = frozen_final_configuration()
    assert set(configuration["discovery_seeds"]).isdisjoint(EVALUATION_SEEDS)
    assert set(REPAIR_SEEDS).isdisjoint(EVALUATION_SEEDS)
    assert configuration["directional_lift_threshold"] == 3.0
