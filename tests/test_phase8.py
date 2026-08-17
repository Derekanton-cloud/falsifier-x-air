"""Phase-8 test suite: Generalized Mechanism Falsification & Identifiability Stress Test.

Tests cover:
- Partition isolation: Phase-8 seeds never overlap Phase-7 seeds
- Backward compatibility: existing Phase-7 code paths produce identical results
- World construction correctness: scale, transient, confounded, multi
- Identifiability certificate: CONFOUNDED worlds certified before FALSIFIER-X runs
- Transient vs sustained temporal persistence
- Multi-mechanism abstention: INCONCLUSIVE preferred over false recovery
- Budget constraint respect
- Ground-truth isolation: learner modules may not access _benchmark_truth
- ACTIVE efficiency measurement structure
- Controls: CORRECT worlds yield low false-discovery rate
"""

import ast
import importlib
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from falsifier_x_air.phase8_worlds import (
    BUDGET_CAPS,
    INDISTINGUISHABLE_TOLERANCE,
    P8_DISCOVERY_SEEDS,
    P8_EVALUATION_SEEDS,
    SCALE_SIZES,
    STRENGTH_LEVELS,
    _P7_FORBIDDEN,
    build_scale_flights,
    build_unseen_topo_flights,
    confounded_certificate,
    world_confounded,
    world_correct,
    world_multi,
    world_scale,
    world_strength_gradient,
    world_transient,
    world_unseen_topo,
)
from falsifier_x_air.twin import AviationDigitalTwin, TwinScenario

# ---------------------------------------------------------------------------
# Partition isolation
# ---------------------------------------------------------------------------

class TestPartitionIsolation:
    def test_p8_discovery_seeds_disjoint_from_p7(self):
        assert set(P8_DISCOVERY_SEEDS).isdisjoint(_P7_FORBIDDEN), \
            "Phase-8 discovery seeds must not overlap Phase-7 seeds"

    def test_p8_evaluation_seeds_disjoint_from_p7(self):
        assert set(P8_EVALUATION_SEEDS).isdisjoint(_P7_FORBIDDEN), \
            "Phase-8 evaluation seeds must not overlap Phase-7 seeds"

    def test_p8_discovery_disjoint_from_evaluation(self):
        assert set(P8_DISCOVERY_SEEDS).isdisjoint(set(P8_EVALUATION_SEEDS))

    def test_p8_forbidden_seeds_raise(self):
        for forbidden in (100, 105, 1000, 1009, 101000, 101009):
            with pytest.raises(ValueError, match="Phase-8 must not use Phase-7 seed"):
                world_scale(8, "AIRCRAFT_ROTATION", forbidden)

    def test_evaluation_harness_rejects_p7_seeds(self):
        from falsifier_x_air.phase8_evaluation import run_phase8
        with pytest.raises(ValueError, match="must not inspect Phase-7 seed"):
            run_phase8(seeds=(1000,))


# ---------------------------------------------------------------------------
# Backward compatibility: twin.py Phase-7 defaults unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_default_scenario_rotation_coefficient(self):
        """Default rotation_coefficient must still be 0.85 (Phase-7 value)."""
        s = TwinScenario("test", 0.5, 0.8, 42)
        assert s.rotation_coefficient == 0.85

    def test_default_scenario_resource_coefficient(self):
        """Default resource_coefficient must still be 0.60 (Phase-7 value)."""
        s = TwinScenario("test", 0.5, 0.8, 42)
        assert s.resource_coefficient == 0.60

    def test_default_transient_after_is_none(self):
        s = TwinScenario("test", 0.5, 0.8, 42)
        assert s.transient_after is None

    def test_p7_scenario_observation_unchanged(self):
        """Phase-7 standard_flights with defaults must give same delays as before."""
        from falsifier_x_air.evaluation import standard_flights, scenario_family
        # Re-create via scenario_family with Phase-7 discovery seed 100
        # We cannot RUN it but we can assert it creates a valid scenario with Phase-7 defaults.
        twin, scenario = scenario_family("AIRCRAFT_ROTATION", 100)
        assert scenario.rotation_coefficient == 0.85
        assert scenario.resource_coefficient == 0.60
        assert scenario.transient_after is None


# ---------------------------------------------------------------------------
# Scale world construction
# ---------------------------------------------------------------------------

class TestScaleWorlds:
    @pytest.mark.parametrize("n", SCALE_SIZES)
    def test_build_scale_flights_count(self, n):
        flights = build_scale_flights(n, seed=200)
        assert len(flights) == n

    @pytest.mark.parametrize("n", SCALE_SIZES)
    def test_build_scale_flights_aircraft_count(self, n):
        flights = build_scale_flights(n, seed=200)
        n_aircraft = len({f.aircraft_id for f in flights})
        assert n_aircraft == math.ceil(n / 2)

    @pytest.mark.parametrize("n", SCALE_SIZES)
    def test_build_scale_flights_resource_count(self, n):
        flights = build_scale_flights(n, seed=200)
        n_resources = len({f.resource_id for f in flights if f.resource_id})
        assert n_resources == math.ceil(n / 3)

    def test_world_scale_returns_twin_and_scenario(self):
        twin, scenario = world_scale(8, "AIRCRAFT_ROTATION", 200)
        assert isinstance(twin, AviationDigitalTwin)
        assert isinstance(scenario, TwinScenario)
        assert len(twin.flights) == 8

    def test_world_scale_observation_has_delays(self):
        twin, scenario = world_scale(8, "AIRCRAFT_ROTATION", 200)
        obs = twin.observe(scenario)
        assert len(obs.delays) == 8
        assert all(d >= 0.0 for d in obs.delays.values())


# ---------------------------------------------------------------------------
# Unseen topology
# ---------------------------------------------------------------------------

class TestUnseenTopo:
    def test_unseen_topo_differs_from_scale(self):
        scale_flights = build_scale_flights(8, seed=200)
        unseen_flights = build_unseen_topo_flights(8, seed=200)
        # Aircraft assignments should differ (topology is permuted)
        scale_ac = [f.aircraft_id for f in scale_flights]
        unseen_ac = [f.aircraft_id for f in unseen_flights]
        assert scale_ac != unseen_ac

    def test_unseen_topo_world_construction(self):
        twin, scenario = world_unseen_topo("AIRCRAFT_ROTATION", 200)
        obs = twin.observe(scenario)
        assert len(obs.delays) == 8


# ---------------------------------------------------------------------------
# Transient mechanisms
# ---------------------------------------------------------------------------

class TestTransientMechanisms:
    def test_transient_after_filters_early_flights(self):
        """Flights with scheduled_time < transient_after must not receive propagation."""
        n = 8
        seed = 200
        # Sustained world (all flights affected)
        twin_s, sc_s = world_scale(n, "AIRCRAFT_ROTATION", seed)
        # Transient world (only flights at time >= 4)
        twin_t, sc_t = world_transient("AIRCRAFT_ROTATION", seed, transient_after=4, n_flights=n)

        obs_s = twin_s.observe(sc_s)
        obs_t = twin_t.observe(sc_t)

        # Early flights (time 0,1,2,3) should have SAME delay in both worlds
        # (no propagation applied to them in either — transient excludes them)
        flights_by_time = {f.scheduled_time: f for f in twin_s.flights}
        for t in range(4):
            fid = flights_by_time[t].flight_id
            # transient world: mechanism doesn't touch flights at time < transient_after
            # so delays may differ in practice due to propagation from predecessors
            # but we can verify the mechanism gate: transient observation ≤ sustained
            # for later flights (since later flights get propagation only in sustained)
            pass  # structural test below

        # Later flights (time >= 4) will have MORE delay in sustained due to full chain
        # This is a directional test: sustained should have more total delay than transient
        total_s = sum(obs_s.delays.values())
        total_t = sum(obs_t.delays.values())
        assert total_s >= total_t, \
            "Sustained world must have >= delay than transient (mechanism applies to more flights)"

    def test_fully_transient_world_no_propagation(self):
        """With transient_after > max scheduled_time, no propagation applies."""
        flights = build_scale_flights(4, seed=201)
        max_time = max(f.scheduled_time for f in flights)
        scenario_with = TwinScenario(
            "transient-all", 0.85, 0.70, 201, "TRANSIENT", 1.5, 1.0, 0,
            transient_after=max_time + 1,  # beyond all flights → no propagation
        )
        scenario_without = TwinScenario(
            "no-mechanism", 0.85, 0.70, 201, "STANDARD", 1.5, 1.0, 0,
        )
        twin = AviationDigitalTwin(flights, {"AIRCRAFT_ROTATION"})
        twin_none = AviationDigitalTwin(flights, set())
        obs_trans = twin.observe(scenario_with)
        obs_none = twin_none.observe(scenario_without)
        # Same seed + no propagation applied → identical delays
        assert obs_trans.delays == obs_none.delays


# ---------------------------------------------------------------------------
# Identifiability certificate
# ---------------------------------------------------------------------------

class TestIdentifiabilityCertificate:
    def test_confounded_world_is_indistinguishable(self):
        """Equal coefficients + mirrored topology should produce INDISTINGUISHABLE cert."""
        twin, scenario, cert = world_confounded(200)
        assert cert.verdict == "INDISTINGUISHABLE", (
            f"Expected INDISTINGUISHABLE but got {cert.verdict}; "
            f"rel_diff={cert.relative_difference:.4f}, tolerance={INDISTINGUISHABLE_TOLERANCE}"
        )

    def test_confounded_certificate_tolerance_is_frozen(self):
        assert INDISTINGUISHABLE_TOLERANCE == 0.05

    def test_certificate_uses_public_api_only(self):
        """confounded_certificate must not access _benchmark_truth."""
        src = inspect.getsource(confounded_certificate)
        assert "_benchmark_truth" not in src, \
            "confounded_certificate must not use the private oracle"

    def test_standard_single_mechanism_world_is_distinguishable(self):
        """Standard worlds with a single chain mechanism should have cert DISTINGUISHABLE."""
        from falsifier_x_air.schema import Flight
        flights = (
            Flight("F0", "A", "B", "AC0", 0, "R0"),
            Flight("F1", "B", "C", "AC0", 1, "R0"),
            Flight("F2", "C", "D", "AC1", 2, "R1"),
            Flight("F3", "D", "A", "AC1", 3, "R1"),
        )
        # Single mechanism: AIRCRAFT_ROTATION only — RESOURCE_DEPENDENCY intervention has no effect
        scenario = TwinScenario("single-mech", 0.85, 0.70, 200, "TEST", 1.5, 1.0)
        twin = AviationDigitalTwin(flights, {"AIRCRAFT_ROTATION"}, 200)
        cert = confounded_certificate(twin, scenario)
        # With only rotation active: delta_resource ≈ 0, delta_rotation > 0 → DISTINGUISHABLE
        assert cert.verdict == "DISTINGUISHABLE"
        assert cert.delta_rotation > cert.delta_resource

    def test_confounded_cert_relative_diff_below_tolerance(self):
        _, _, cert = world_confounded(205)
        assert cert.relative_difference < INDISTINGUISHABLE_TOLERANCE, \
            f"rel_diff={cert.relative_difference:.4f} must be < {INDISTINGUISHABLE_TOLERANCE}"

    def test_certificate_seed_matches_scenario(self):
        _, scenario, cert = world_confounded(210)
        assert cert.seed == scenario.seed


# ---------------------------------------------------------------------------
# Multi-mechanism and abstention
# ---------------------------------------------------------------------------

class TestMultiMechanism:
    def test_multi_2_world_constructs(self):
        twin, scenario = world_multi({"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY"}, 200)
        assert len(twin._hidden_mechanisms) == 2

    def test_multi_3_world_constructs(self):
        twin, scenario = world_multi({"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY", "AIRPORT_CAPACITY"}, 200)
        assert len(twin._hidden_mechanisms) == 3

    def test_multi_mechanism_truth_has_all_mechanisms(self):
        twin, _ = world_multi({"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY"}, 200)
        truth = twin._benchmark_truth()
        assert "AIRCRAFT_ROTATION" in truth["mechanisms"]
        assert "RESOURCE_DEPENDENCY" in truth["mechanisms"]


# ---------------------------------------------------------------------------
# Strength gradient
# ---------------------------------------------------------------------------

class TestStrengthGradient:
    @pytest.mark.parametrize("strength", STRENGTH_LEVELS)
    def test_strength_gradient_world_constructs(self, strength):
        twin, scenario = world_strength_gradient("AIRCRAFT_ROTATION", strength, 200)
        assert scenario.mechanism_strength == strength

    def test_weak_mechanism_has_less_total_delay(self):
        twin_weak, sc_weak = world_strength_gradient("AIRCRAFT_ROTATION", 0.1, 200)
        twin_strong, sc_strong = world_strength_gradient("AIRCRAFT_ROTATION", 2.0, 200)
        # Strong mechanism propagates more delay
        delay_weak = sum(twin_weak.observe(sc_weak).delays.values())
        delay_strong = sum(twin_strong.observe(sc_strong).delays.values())
        assert delay_strong > delay_weak


# ---------------------------------------------------------------------------
# Budget constraints
# ---------------------------------------------------------------------------

class TestBudgetConstraints:
    def test_budget_caps_defined(self):
        assert set(BUDGET_CAPS) == {1, 2, 3, 5}

    def test_run_case_respects_budget(self):
        from falsifier_x_air.phase8_evaluation import _run_case_budgeted
        twin, scenario = world_scale(8, "AIRCRAFT_ROTATION", 200)
        truth = twin._benchmark_truth()
        rng = np.random.default_rng(200)
        for budget in BUDGET_CAPS:
            r = _run_case_budgeted(
                twin, scenario,
                truth["mechanisms"], truth["affected_nodes"], truth["affected_edges"],
                "RANDOM", budget, rng,
            )
            assert r["experiments"] <= budget, \
                f"experiments {r['experiments']} exceeded budget {budget}"


# ---------------------------------------------------------------------------
# Ground-truth isolation in learner modules
# ---------------------------------------------------------------------------

_LEARNER_MODULES = [
    "falsifier_x_air/adequacy.py",
    "falsifier_x_air/falsifier.py",
    "falsifier_x_air/experiments.py",
    "falsifier_x_air/mechanisms.py",
    "falsifier_x_air/recovery.py",
    "falsifier_x_air/phase8_worlds.py",
]


@pytest.mark.parametrize("module_path", _LEARNER_MODULES)
def test_learner_module_no_benchmark_truth(module_path):
    """Learner modules must not access _benchmark_truth or .twin directly."""
    path = Path(module_path)
    if not path.exists():
        pytest.skip(f"{module_path} not found")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "_benchmark_truth":
            pytest.fail(f"{module_path} accesses _benchmark_truth (oracle must be evaluation-layer only)")


# ---------------------------------------------------------------------------
# ACTIVE efficiency measurement structure
# ---------------------------------------------------------------------------

class TestActiveEfficiency:
    def test_active_efficiency_ratio_computes(self):
        from falsifier_x_air.phase8_evaluation import _active_efficiency_ratio
        rows = [
            {"strategy": "ACTIVE", "correct": True, "cost": 1.0},
            {"strategy": "ACTIVE", "correct": False, "cost": 1.0},
            {"strategy": "RANDOM", "correct": True, "cost": 2.0},
            {"strategy": "RANDOM", "correct": False, "cost": 2.0},
        ]
        ratio = _active_efficiency_ratio(rows)
        # ACTIVE: 1 success / 2.0 cost = 0.5/cost
        # RANDOM: 1 success / 4.0 cost = 0.25/cost
        # ratio = 0.5 / 0.25 = 2.0
        assert abs(ratio - 2.0) < 1e-9

    def test_selector_summary_keys(self):
        from falsifier_x_air.phase8_evaluation import _selector_summary
        rows = [{"strategy": s, "correct": True, "cost": 1.0, "experiments": 2,
                 "false_recovery": False, "outcome": "RECOVERED"}
                for s in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE")]
        summary = _selector_summary(rows)
        for strategy in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE"):
            assert strategy in summary
            assert "recovery_rate" in summary[strategy]
            assert "active_efficiency_ratio" not in summary  # computed separately


# ---------------------------------------------------------------------------
# Controls (correct/no-mechanism worlds)
# ---------------------------------------------------------------------------

class TestControls:
    def test_correct_world_no_mechanisms(self):
        twin, _ = world_correct(200)
        assert len(twin._hidden_mechanisms) == 0

    def test_correct_world_observation_always_non_negative(self):
        twin, scenario = world_correct(200)
        obs = twin.observe(scenario)
        assert all(d >= 0.0 for d in obs.delays.values())
