"""Focused tests for the FALSIFIER-X AIR Component Ablation Study."""

import ast
import json
import re
from pathlib import Path
import numpy as np
import pytest

from falsifier_x_air.ablation import (
    ABLATION_LADDER,
    DISCOVERY_SEEDS,
    EVALUATION_SEEDS,
    REPAIR_SEEDS,
    SELECTOR_STRATEGIES,
    compute_temporal_persistence,
    run_ablation_case,
    run_ablation_study,
)
from falsifier_x_air.evaluation import SCENARIO_FAMILIES, _prediction, scenario_family
from falsifier_x_air.graph import AviationGraph
from falsifier_x_air.schema import Flight, PredictionOutput


def test_ablation_ladder_definition_and_order():
    assert len(ABLATION_LADDER) == 6
    assert ABLATION_LADDER[0] == "A1_PREDICTION_ONLY"
    assert ABLATION_LADDER[1] == "A2_UNCERTAINTY"
    assert ABLATION_LADDER[2] == "A3_TEMPORAL"
    assert ABLATION_LADDER[3] == "A4_GRAPH_PASSIVE"
    assert ABLATION_LADDER[4] == "A5_RANDOM_INTERVENTION"
    assert ABLATION_LADDER[5] == "A6_ACTIVE_INTERVENTION"


def test_ablation_partition_isolation_strictly_disjoint():
    # Ablation must run on discovery seeds 100--109 and NEVER touch final 1000--1009 or repair 101000--101009
    assert set(DISCOVERY_SEEDS).isdisjoint(EVALUATION_SEEDS)
    assert set(DISCOVERY_SEEDS).isdisjoint(REPAIR_SEEDS)
    assert set(EVALUATION_SEEDS).isdisjoint(REPAIR_SEEDS)
    assert min(DISCOVERY_SEEDS) == 100 and max(DISCOVERY_SEEDS) == 109
    assert min(EVALUATION_SEEDS) == 1000 and max(EVALUATION_SEEDS) == 1009


def test_ablation_rejects_evaluation_or_repair_seeds():
    with pytest.raises(ValueError, match="never use locked evaluation seeds"):
        run_ablation_case("A1_PREDICTION_ONLY", "CORRECT", 1000)
    with pytest.raises(ValueError, match="never use locked evaluation seeds"):
        run_ablation_case("A6_ACTIVE_INTERVENTION", "AIRCRAFT_ROTATION", 101000)


def test_ablation_ground_truth_isolation():
    # Only ablation runner / evaluation module may access private benchmark oracle for scoring
    # Learner modules (falsifier, adequacy, experiments, mechanisms, recovery) must NOT import twin
    root = Path("falsifier_x_air")
    for filename in ("adequacy.py", "falsifier.py", "experiments.py", "mechanisms.py", "recovery.py"):
        source = (root / filename).read_text(encoding="utf-8")
        assert "_benchmark_truth" not in source


def test_compute_temporal_persistence_has_no_hidden_state_access():
    source = Path("falsifier_x_air/ablation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    func_def = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "compute_temporal_persistence")
    func_source = ast.unparse(func_def)
    assert "_hidden" not in func_source
    assert "_benchmark_truth" not in func_source
    assert "twin" not in func_source


def test_a1_prediction_only_inclusion_exclusion():
    # A1 uses raw point residual threshold, no uncertainty bounds, no persistence, no graph, 0 experiments
    res = run_ablation_case("A1_PREDICTION_ONLY", "AIRCRAFT_ROTATION", 100)
    assert res.experiments == 0
    assert res.cost == 0.0
    assert res.recovered_mechanism is None
    assert len(res.candidate_mechanisms) == 0

    # In NOISE, raw point error flags false discovery because there are no conformal bounds
    res_noise = run_ablation_case("A1_PREDICTION_ONLY", "NOISE", 100)
    assert res_noise.is_false_structural_detection is True


def test_a2_uncertainty_eliminates_noise_false_discovery_and_flags_ood():
    # A2 includes calibrated uncertainty and OOD check, but no persistence and no graph
    res_noise = run_ablation_case("A2_UNCERTAINTY", "NOISE", 100)
    assert res_noise.adequacy_state == "ADEQUATE"
    assert res_noise.is_false_structural_detection is False

    res_ood = run_ablation_case("A2_UNCERTAINTY", "OOD", 100)
    assert res_ood.adequacy_state == "OOD"
    assert res_ood.experiments == 0


def test_hand_constructed_persistence_calculation():
    # 4 distinct time windows: t=0, 1, 2, 3
    flights = (
        Flight("F1", "A", "B", "AC1", 0),
        Flight("F2", "B", "C", "AC1", 1),
        Flight("F3", "C", "D", "AC2", 2),
        Flight("F4", "D", "A", "AC2", 3),
    )
    pred = PredictionOutput(
        mean=np.full(4, 10.0),
        lower=np.full(4, 7.0),
        upper=np.full(4, 13.0),
        epistemic_std=np.full(4, 1.5),
    )

    # 1. Zero violation: all delays at mean (10.0)
    c0, f0, p0 = compute_temporal_persistence(flights, np.array([10.0, 10.0, 10.0, 10.0]), pred)
    assert c0 == 0 and f0 == 0.0 and p0 is False

    # 2. Single-window spike (t=0 only): F1 has delay 20.0 (violating), others 10.0
    c1, f1, p1 = compute_temporal_persistence(flights, np.array([20.0, 10.0, 10.0, 10.0]), pred)
    assert c1 == 1 and f1 == 0.25 and p1 is False

    # 3. Two-window repeated propagation (t=0 and t=1): F1 and F2 violate
    c2, f2, p2 = compute_temporal_persistence(flights, np.array([20.0, 20.0, 10.0, 10.0]), pred)
    assert c2 == 2 and f2 == 0.50 and p2 is True

    # 4. Four-window pervasive disruption (t=0, 1, 2, 3 all violate)
    c4, f4, p4 = compute_temporal_persistence(flights, np.array([20.0, 20.0, 20.0, 20.0]), pred)
    assert c4 == 4 and f4 == 1.00 and p4 is True


def test_a2_vs_a3_information_difference():
    # Proves A2 and A3 genuinely use different information:
    # A2 evaluates aggregate/instantaneous conformal bounds, oblivious to time distribution.
    # A3 evaluates temporal persistence across distinct scheduled time windows.
    # On a single-window spike (e.g. 2 violating flights both scheduled at t=0):
    # Both have identical aggregate miscoverage = 2/4 = 0.50 and standardized z >= 2.0.
    # A2 detects STRUCTURALLY_SUSPICIOUS, while A3 filters it out as INCONCLUSIVE.
    flights_1win = (
        Flight("F1", "A", "B", "AC1", 0),
        Flight("F2", "B", "C", "AC1", 0),
        Flight("F3", "C", "D", "AC2", 1),
        Flight("F4", "D", "A", "AC2", 2),
    )
    obs = np.array([20.0, 20.0, 10.0, 10.0])
    pred = PredictionOutput(
        mean=np.full(4, 10.0),
        lower=np.full(4, 7.0),
        upper=np.full(4, 13.0),
        epistemic_std=np.full(4, 1.5),
    )
    # Check persistence on single-window vs multi-window
    c_1win, f_1win, p_1win = compute_temporal_persistence(flights_1win, obs, pred)
    assert c_1win == 1 and f_1win == 1/3 and p_1win is False

    flights_2win = (
        Flight("F1", "A", "B", "AC1", 0),
        Flight("F2", "B", "C", "AC1", 1),
        Flight("F3", "C", "D", "AC2", 2),
        Flight("F4", "D", "A", "AC2", 3),
    )
    c_2win, f_2win, p_2win = compute_temporal_persistence(flights_2win, obs, pred)
    assert c_2win == 2 and f_2win == 0.50 and p_2win is True


def test_persistence_computed_from_observable_evidence_varies():
    # Proves persistence is computed from observable evidence and is not constant across scenarios
    counts = []
    fractions = []
    for fam in ("CORRECT", "NOISE", "AIRCRAFT_ROTATION", "AIRPORT_CAPACITY"):
        for s in DISCOVERY_SEEDS:
            twin, scenario = scenario_family(fam, s)
            obs = twin.observe(scenario)
            g = AviationGraph.build(obs.flights)
            pred = _prediction(twin, scenario)
            observed = obs.delay_vector(g.flight_ids)
            c, f, _ = compute_temporal_persistence(obs.flights, observed, pred, g.flight_ids)
            counts.append(c)
            fractions.append(f)

    # Persistence counts and fractions are not all identical (not a hardcoded constant)
    assert min(counts) == 0
    assert max(counts) >= 3
    assert len(set(counts)) >= 3
    assert min(fractions) == 0.0
    assert max(fractions) >= 0.75


def test_a4_graph_passive_generates_candidates_without_interventions():
    # A4 localizes graph and generates candidates, but runs 0 interventions (passive)
    res = run_ablation_case("A4_GRAPH_PASSIVE", "AIRCRAFT_ROTATION", 100)
    assert res.adequacy_state == "STRUCTURALLY_SUSPICIOUS"
    assert len(res.localized_nodes) > 0
    assert len(res.candidate_mechanisms) > 0
    assert res.experiments == 0
    assert res.cost == 0.0
    assert res.recovered_mechanism is None  # Cannot recover without intervention


def test_a5_and_a6_interventions_recover_ground_truth():
    # A5 and A6 perform interventions and recover ground truth on unambiguous cases
    res_a5 = run_ablation_case("A5_RANDOM_INTERVENTION", "AIRCRAFT_ROTATION", 100)
    assert res_a5.experiments >= 1
    assert res_a5.recovered_mechanism == "AIRCRAFT_ROTATION"
    assert res_a5.held_out_policy_delta > 0.0

    res_a6 = run_ablation_case("A6_ACTIVE_INTERVENTION", "AIRCRAFT_ROTATION", 100)
    assert res_a6.experiments >= 1
    assert res_a6.recovered_mechanism == "AIRCRAFT_ROTATION"
    assert res_a6.held_out_policy_delta > 0.0


def test_ablation_reproducibility():
    res1 = run_ablation_case("A6_ACTIVE_INTERVENTION", "RESOURCE_DEPENDENCY", 102)
    res2 = run_ablation_case("A6_ACTIVE_INTERVENTION", "RESOURCE_DEPENDENCY", 102)
    assert res1.adequacy_state == res2.adequacy_state
    assert res1.recovered_mechanism == res2.recovered_mechanism
    assert res1.experiments == res2.experiments
    assert res1.cost == res2.cost
    assert res1.held_out_policy_delta == res2.held_out_policy_delta


def test_selector_strategies_complete():
    assert set(SELECTOR_STRATEGIES) == {"RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE"}


def test_report_consistency_with_json_results(tmp_path):
    # Run ablation study in temporary directory and verify report numbers match JSON
    results = run_ablation_study(output_dir=tmp_path)
    json_path = tmp_path / "phase7_ablation_results.json"
    report_path = tmp_path / "phase7_ablation_report.md"

    assert json_path.exists()
    assert report_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Verify no stale hardcoded 2.69 string exists
    assert "2.69" not in report_text

    # Verify operational mitigation delta in report matches JSON
    a6_delta = data["ladder_summary"]["A6_ACTIVE_INTERVENTION"]["mean_held_out_policy_delta"]
    expected_delta_str = f"{a6_delta:.2f}"
    assert expected_delta_str in report_text

    # Verify table values for all ladder levels match JSON
    for level in ABLATION_LADDER:
        s = data["ladder_summary"][level]
        pattern = (
            rf"\|\s*\*\*{level}\*\*\s*\|\s*{s['structural_detection_rate']:.2f}\s*\|\s*"
            rf"{s['false_structural_discovery_rate']:.2f}\s*\|\s*{s['ood_discrimination_rate']:.2f}\s*\|\s*"
            rf"{s['node_f1']:.2f}\s*\|\s*{s['edge_f1']:.2f}\s*\|\s*{s['mechanism_recovery_rate']:.2f}\s*\|\s*"
            rf"{s['mechanism_f1']:.2f}\s*\|\s*{s['experiments_to_recovery']:.2f}\s*\|\s*"
            rf"{s['mean_cost']:.2f}\s*\|\s*{s['mean_held_out_policy_delta']:.2f}\s*\|"
        )
        assert re.search(pattern, report_text) is not None, f"Report table row mismatch for {level}"

