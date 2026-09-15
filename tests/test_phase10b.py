"""Isolation and semantics tests for the additive Phase-10B benchmark."""
import ast
from pathlib import Path

import numpy as np
import pytest

from falsifier_x_air.graph import AviationGraph
from falsifier_x_air.phase10b_evaluation import (
    INTERVENTION_NOISE_SD, PHASE10B_BLIND_SEEDS, PHASE10B_DISCOVERY_SEEDS,
    PURE_FAMILIES, paired_intervention_observation, phase10b_flights,
    phase10b_world, topology_certificate,
)
from falsifier_x_air.phase10b_repair import WRONG_MECHANISM, observable_feature


def test_phase10b_seed_partitions_are_disjoint():
    assert set(PHASE10B_DISCOVERY_SEEDS).isdisjoint(PHASE10B_BLIND_SEEDS)
    assert min(PHASE10B_DISCOVERY_SEEDS) == 700000
    assert max(PHASE10B_DISCOVERY_SEEDS) == 700099
    assert min(PHASE10B_BLIND_SEEDS) == 700100
    assert max(PHASE10B_BLIND_SEEDS) == 700299


def test_topology_is_disjoint_and_observable_features_differ():
    graph = AviationGraph.build(phase10b_flights())
    certificate = topology_certificate(graph)
    assert certificate["edge_overlap"] == []
    assert certificate["target_overlap"] == []
    assert certificate["source_overlap"] != []  # Identical sources
    assert not certificate["features_identical_by_structure"]
    twin, scenario = phase10b_world("AIRCRAFT_ROTATION", 700000)
    observation = twin.observe(scenario)
    assert not np.array_equal(observable_feature("AIRCRAFT_ROTATION", observation, graph),
                              observable_feature("RESOURCE_DEPENDENCY", observation, graph))


def test_paired_world_and_noise_semantics_are_reproducible_non_cancelling():
    twin, scenario = phase10b_world("AIRCRAFT_ROTATION", 700000)
    first = paired_intervention_observation(twin, scenario, "AIRCRAFT_ROTATION")
    second = paired_intervention_observation(twin, scenario, "AIRCRAFT_ROTATION")
    assert first == second
    # The twin itself uses the same scenario RNG for factual and counterfactual.
    factual = twin.observe(scenario)
    assert factual.delays == twin.observe(scenario).delays
    assert np.isclose(first["observed_effect"] - first["true_effect"], first["epsilon_intervention"])
    assert first["epsilon_intervention"] != 0.0
    other_twin, other_scenario = phase10b_world("AIRCRAFT_ROTATION", 700001)
    other = paired_intervention_observation(other_twin, other_scenario, "AIRCRAFT_ROTATION")
    assert other["observed_effect"] != first["observed_effect"]
    assert INTERVENTION_NOISE_SD == 2.0


def test_controls_are_predefined():
    assert WRONG_MECHANISM == {"AIRCRAFT_ROTATION": "RESOURCE_DEPENDENCY",
                               "RESOURCE_DEPENDENCY": "AIRCRAFT_ROTATION",
                               "AIRPORT_CAPACITY": "AIRCRAFT_ROTATION"}


def test_inconclusive_identification_blocks_phase10b_repair(monkeypatch):
    import falsifier_x_air.phase10b_evaluation as evaluation
    frozen = {m: {"coefficient": 1.0, "sample_count": 1} for m in PURE_FAMILIES}
    frozen["GENERIC"] = {"coefficient": 1.0, "sample_count": 1}
    monkeypatch.setattr(evaluation, "_identified", lambda *args: None)
    monkeypatch.setattr(evaluation, "apply_repair", lambda *args: pytest.fail("repair must be gated"))
    result = evaluation._case("AIRCRAFT_ROTATION", 700100, frozen)
    assert result is not None
    assert result["repaired"] is False
    assert result["e2e_mae"] == result["orig_mae"]


def test_wrong_identification_passed_to_repair_not_oracle_family(monkeypatch):
    import falsifier_x_air.phase10b_evaluation as evaluation
    frozen = {m: {"coefficient": 1.0, "sample_count": 1} for m in PURE_FAMILIES}
    frozen["GENERIC"] = {"coefficient": 1.0, "sample_count": 1}
    
    # Force falsifier to return the WRONG mechanism.
    monkeypatch.setattr(evaluation, "_identified", lambda *args: "AIRPORT_CAPACITY")
    
    calls = []
    def mock_repair(pred, mech, coeff, obs, graph):
        calls.append(mech)
        return pred
        
    monkeypatch.setattr(evaluation, "apply_repair", mock_repair)
    evaluation._case("AIRCRAFT_ROTATION", 700100, frozen)
    
    # The first call to apply_repair (for correct_mae) must use the actual RECOVERED mechanism.
    # The final call (oracle_mae) uses the TRUE family.
    assert calls[0] == "AIRPORT_CAPACITY"
    assert calls[-1] == "AIRCRAFT_ROTATION"


def test_learner_source_has_no_evaluator_truth_or_blind_access():
    source = (Path(__file__).parent.parent / "falsifier_x_air" / "phase10b_repair.py").read_text(encoding="utf-8")
    forbidden = ("_benchmark_truth", "_hidden_mechanisms", "TwinScenario", "AviationDigitalTwin",
                 "PHASE10B_BLIND_SEEDS", "_oracle_coefficient", "rotation_coefficient", "resource_coefficient")
    assert not any(token in source for token in forbidden)
    ast.parse(source)
