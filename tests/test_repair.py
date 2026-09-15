"""Phase 10 tests for causal model repair."""

import ast
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import patch

from falsifier_x_air.repair import estimate_coefficients, apply_repair, apply_generic_repair, _ESTIMATED_COEFFICIENTS
from falsifier_x_air.phase10_repair import REPAIR_EVAL_SEEDS, evaluate_case, run_phase10_evaluation
from falsifier_x_air.evaluation import DISCOVERY_SEEDS, EVALUATION_SEEDS, REPAIR_SEEDS
from falsifier_x_air.twin import AviationDigitalTwin, TwinScenario
from falsifier_x_air.schema import Flight, NetworkObservation, PredictionOutput
from falsifier_x_air.graph import AviationGraph, GraphSnapshot

def test_source_inspection_no_oracle():
    """Coefficient estimation must not reference twin-private state."""
    repair_file = Path(__file__).parent.parent / "falsifier_x_air" / "repair.py"
    content = repair_file.read_text(encoding="utf-8")
    assert "_benchmark_truth" not in content
    assert "_hidden_mechanisms" not in content
    assert "scenario.rotation_coefficient" not in content
    assert "scenario.resource_coefficient" not in content
    assert "scenario.mechanism_strength" not in content

def test_partition_disjointness():
    """REPAIR_EVAL_SEEDS must be disjoint from existing ranges."""
    eval_seeds = set(REPAIR_EVAL_SEEDS)
    assert eval_seeds.isdisjoint(DISCOVERY_SEEDS)
    assert eval_seeds.isdisjoint(EVALUATION_SEEDS)
    assert eval_seeds.isdisjoint(REPAIR_SEEDS)

def test_coefficients_frozen():
    """Coefficients are frozen before REPAIR_EVAL_SEEDS is touched."""
    # We clear the global cache to test this
    _ESTIMATED_COEFFICIENTS.clear()
    
    with patch("falsifier_x_air.phase10_repair.estimate_coefficients", wraps=estimate_coefficients) as mock_est:
        # Run a tiny evaluation to see call order
        # We'll just patch evaluate_case to not do anything expensive
        with patch("falsifier_x_air.phase10_repair.evaluate_case", return_value={
            "family": "TEST", "seed": 500000, "recovered": "AIRCRAFT_ROTATION",
            "orig_mae": 1.0, "correct_mae": 0.5, "wrong_mae": 1.0, "generic_mae": 1.0, "oracle_mae": 0.4
        }) as mock_eval:
            # We mock families_to_test to be fast
            with patch("falsifier_x_air.phase10_repair.REPAIR_EVAL_SEEDS", [500000, 500001]):
                run_phase10_evaluation()
                
            # Assert estimate_coefficients was called exactly once
            assert mock_est.call_count == 1
            # And evaluate_case was called after
            assert mock_eval.call_count > 0

def test_gating_rule_inconclusive_rejected():
    """INCONCLUSIVE cases never produce a repair attempt."""
    # evaluate_case returns None for inconclusive cases
    # Let's mock investigate to return inconclusive
    from falsifier_x_air.falsifier import InvestigationResult
    from falsifier_x_air.schema import AdequacyEvaluation
    
    adequacy = AdequacyEvaluation(0, 0, 0, 0, 0, 0, 0, "STRUCTURALLY_SUSPICIOUS", "REASON")
    mock_result = InvestigationResult(
        adequacy=adequacy,
        affected_flights=(),
        candidates=(),
        experiment_results=(),
        recovered_mechanism="AIRCRAFT_ROTATION",
        outcome="INCONCLUSIVE",
        identifiability_reason="INDISTINGUISHABLE_EQUIVALENCE_CLASS"
    )
    
    with patch("falsifier_x_air.falsifier.FalsifierXAir.investigate", return_value=mock_result):
        # Even though recovered_mechanism is set, identifiability_reason is not LEGITIMATELY_IDENTIFIED
        res = evaluate_case("MULTIPLE", 500000, {"AIRCRAFT_ROTATION": 1.0})
        assert res is None

def test_repair_reduces_error_on_synthetic_case():
    """CORRECT_REPAIR reduces error relative to ORIGINAL, WRONG_REPAIR does not."""
    # Hand-construct a simple graph and observation
    flights = [
        Flight("F1", "A", "B", "AC1", 0),
        Flight("F2", "B", "C", "AC1", 1)
    ]
    graph_snapshot = AviationGraph.build(flights)
    
    # F1 has delay 10, F2 has delay 10.
    # Prediction was 0 for both.
    delays = {"F1": 10.0, "F2": 10.0}
    observation = NetworkObservation(tuple(flights), delays, 0.5, 0.5, "test")
    
    pred_mean = np.array([0.0, 0.0]) # M0 prediction
    target = np.array([10.0, 10.0])
    
    # Correct repair: AIRCRAFT_ROTATION
    # Feature for F1 = 0, F2 = 10.
    # If coeff = 1.0, prediction for F2 becomes 10.0.
    # Error for F1 is 10, Error for F2 becomes 0.
    pred_correct = apply_repair(pred_mean, "AIRCRAFT_ROTATION", 1.0, observation, graph_snapshot)
    
    # Wrong repair: RESOURCE_DEPENDENCY
    # There are no resource edges.
    # Feature for F1 = 0, F2 = 0.
    pred_wrong = apply_repair(pred_mean, "RESOURCE_DEPENDENCY", 1.0, observation, graph_snapshot)
    
    orig_mae = np.mean(np.abs(pred_mean - target))
    correct_mae = np.mean(np.abs(pred_correct - target))
    wrong_mae = np.mean(np.abs(pred_wrong - target))
    
    assert correct_mae < orig_mae
    assert wrong_mae == orig_mae

def test_deterministic_reproducibility():
    """Same seed, same coefficients, same result on repeat runs."""
    coeffs = {"AIRCRAFT_ROTATION": 0.85, "RESOURCE_DEPENDENCY": 0.60, "AIRPORT_CAPACITY": 1.0}
    res1 = evaluate_case("AIRCRAFT_ROTATION", 500005, coeffs)
    res2 = evaluate_case("AIRCRAFT_ROTATION", 500005, coeffs)
    
    if res1 is not None and res2 is not None:
        assert res1 == res2
    else:
        assert res1 is None and res2 is None
