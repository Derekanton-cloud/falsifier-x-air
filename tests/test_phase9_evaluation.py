"""Phase-9 evaluation test suite.

Validates:
- Partition cleanliness: seeds 3000-3019 are disjoint from all earlier phases.
- Evaluation harness outputs: phase9_eval_results.json and phase9_eval_report.md.
- 100% abstention correctness on confounded worlds without oracle leakage.
- Preservation of legitimate recovery on distinguishable worlds (scales 8, 16, 32).
- Zero false recovery across evaluated suites.
- D-distribution margin around frozen epsilon=0.20.
- AST isolation: no _benchmark_truth access in learner code.
"""

import ast
import json
from pathlib import Path

import pytest

from falsifier_x_air.phase9_evaluation import (
    P9_EVALUATION_SEEDS,
    _FORBIDDEN_SEEDS,
    _assert_p9_clean,
    run_phase9_evaluation,
)


class TestPhase9PartitionIsolation:
    def test_p9_evaluation_seeds_disjoint_from_all_prior(self):
        assert set(P9_EVALUATION_SEEDS).isdisjoint(_FORBIDDEN_SEEDS), \
            "Phase-9 evaluation seeds must be completely disjoint from prior phases"

    def test_forbidden_seeds_raise_in_p9_eval(self):
        for s in (100, 1000, 101000, 200, 2000):
            with pytest.raises(ValueError, match="Phase-9 evaluation must not use prior seed"):
                _assert_p9_clean(s)


class TestPhase9EvaluationExecution:
    @pytest.fixture(scope="class")
    def eval_output(self, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("p9_eval_test")
        # Run evaluation on subset of evaluation seeds for fast testing
        return run_phase9_evaluation(output_dir=out_dir, seeds=(3000, 3001, 3002))

    def test_results_structure(self, eval_output):
        assert "metadata" in eval_output
        assert "confounded_summary" in eval_output
        assert "scale_summary" in eval_output
        assert "strength_summary" in eval_output
        assert "budget_summary" in eval_output
        assert "unseen_topology_summary" in eval_output
        assert "d_distribution_summary" in eval_output
        assert "records" in eval_output

    def test_confounded_abstention_correctness_is_100_percent(self, eval_output):
        conf = eval_output["confounded_summary"]
        assert conf["abstention_correctness"] == 1.0, \
            f"Expected 100% abstention correctness on confounded worlds, got {conf['abstention_correctness']}"
        assert conf["inconclusive_rate"] == 1.0
        assert conf["false_recovery_rate"] == 0.0

    def test_scale_generalization_recovery_preserved(self, eval_output):
        scale = eval_output["scale_summary"]
        for k, v in scale.items():
            assert v["recovered_rate"] == 1.0, f"Expected 100% recovery for {k}, got {v['recovered_rate']}"
            assert v["false_recovery_rate"] == 0.0

    def test_unseen_topology_recovery_preserved(self, eval_output):
        unseen = eval_output["unseen_topology_summary"]
        assert unseen["recovered_rate"] == 1.0
        assert unseen["false_recovery_rate"] == 0.0

    def test_d_distribution_margins(self, eval_output):
        d_dist = eval_output["d_distribution_summary"]
        margin = d_dist["margin_around_epsilon"]
        assert margin["confounded_max_to_epsilon"] > 0.0, "Confounded D max must be strictly below epsilon"
        assert margin["epsilon_to_distinguishable_min"] > 0.0, "Distinguishable D min must be strictly above epsilon"
        assert d_dist["confounded_survivor_vs_competitor_d"]["max"] == pytest.approx(0.0)
        assert d_dist["distinguishable_survivor_vs_competitor_d"]["min"] == pytest.approx(1.0)


class TestOracleIsolationAST:
    def test_phase9_evaluation_is_only_module_with_benchmark_truth(self):
        learner_files = [
            "falsifier_x_air/adequacy.py",
            "falsifier_x_air/falsifier.py",
            "falsifier_x_air/experiments.py",
            "falsifier_x_air/identifiability.py",
            "falsifier_x_air/mechanisms.py",
            "falsifier_x_air/recovery.py",
            "falsifier_x_air/graph.py",
            "falsifier_x_air/twin.py",
        ]
        for fpath in learner_files:
            src = Path(fpath).read_text(encoding="utf-8")
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "_benchmark_truth":
                    # Only twin.py defines the method; no other module should call or define it
                    if fpath != "falsifier_x_air/twin.py":
                        pytest.fail(f"{fpath} must not access _benchmark_truth")
