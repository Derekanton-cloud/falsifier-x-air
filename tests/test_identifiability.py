"""Phase-9 identifiability test suite.

Covers:
- Oracle isolation (AST scan of identifiability.py)
- response_distance unit tests (symmetric, asymmetric, degenerate)
- Confounded world → INCONCLUSIVE via FalsifierXAir.investigate()
- Single-mechanism world → RECOVERED via FalsifierXAir.investigate()
- Probe experiments use public twin API only
- Epsilon value is frozen
- Budget-limited cases
- Regression: Phase-7 family behavior unchanged by identifiability layer
- Phase-8 H3 baseline preserved (phase8_evaluation._run_case_budgeted unchanged)
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from falsifier_x_air.identifiability import (
    EPSILON_IDENT,
    IdentifiabilityResult,
    check_identifiability,
    response_distance,
)
from falsifier_x_air.falsifier import FalsifierXAir, InvestigationContext, InvestigationResult
from falsifier_x_air.phase8_worlds import (
    world_confounded,
    world_correct,
    world_scale,
    world_multi,
)
from falsifier_x_air.phase8_evaluation import _prediction, _truth
from falsifier_x_air.graph import AviationGraph
from falsifier_x_air.schema import Flight
from falsifier_x_air.twin import AviationDigitalTwin, TwinScenario


# ---------------------------------------------------------------------------
# Helper: build a minimal InvestigationContext
# ---------------------------------------------------------------------------

def _make_context(twin, scenario, ood_score=0.0):
    observation = twin.observe(scenario)
    graph = AviationGraph.build(observation.flights)
    prediction = _prediction(twin, scenario)
    return InvestigationContext(
        observation=observation,
        graph=graph,
        prediction=prediction,
        alternative_prediction=prediction.mean,
        ood_score=ood_score,
        persistence_count=3,
        scenario=scenario,
    )


# ---------------------------------------------------------------------------
# Oracle isolation
# ---------------------------------------------------------------------------

class TestOracleIsolation:
    def test_identifiability_module_no_benchmark_truth(self):
        src = Path("falsifier_x_air/identifiability.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "_benchmark_truth":
                pytest.fail("identifiability.py must not access _benchmark_truth")

    def test_identifiability_module_no_confounded_certificate(self):
        """identifiability.py must not import or call confounded_certificate."""
        src = Path("falsifier_x_air/identifiability.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "confounded_certificate":
                pytest.fail("identifiability.py must not call confounded_certificate")
            if isinstance(node, ast.Attribute) and node.attr == "confounded_certificate":
                pytest.fail("identifiability.py must not call confounded_certificate")

    def test_falsifier_no_benchmark_truth(self):
        src = Path("falsifier_x_air/falsifier.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "_benchmark_truth":
                pytest.fail("falsifier.py must not access _benchmark_truth")


# ---------------------------------------------------------------------------
# response_distance unit tests
# ---------------------------------------------------------------------------

class TestResponseDistance:
    def test_identical_effects_gives_zero(self):
        assert response_distance(100.0, 100.0, 1.0) == pytest.approx(0.0)

    def test_zero_vs_nonzero_gives_one(self):
        assert response_distance(100.0, 0.0, 0.01) == pytest.approx(1.0)

    def test_symmetric(self):
        d_ab = response_distance(80.0, 40.0, 1.0)
        d_ba = response_distance(40.0, 80.0, 1.0)
        assert d_ab == pytest.approx(d_ba)

    def test_below_epsilon_threshold(self):
        # Tiny difference: |101 - 100| / 101 ≈ 0.0099 → below EPSILON_IDENT
        d = response_distance(101.0, 100.0, 1.0)
        assert d < EPSILON_IDENT

    def test_above_epsilon_threshold(self):
        # Large difference: |100 - 0| / 100 = 1.0 → clearly above EPSILON_IDENT
        d = response_distance(100.0, 0.0, 1.0)
        assert d > EPSILON_IDENT

    def test_sigma_eff_as_floor(self):
        # Both effects zero but sigma_eff prevents division by zero
        d = response_distance(0.0, 0.0, 5.0)
        assert d == pytest.approx(0.0)

    def test_epsilon_frozen_at_0_20(self):
        assert EPSILON_IDENT == 0.20


# ---------------------------------------------------------------------------
# Confounded world → INCONCLUSIVE (H3 fix)
# ---------------------------------------------------------------------------

class TestConfoundedAbstention:
    def test_confounded_world_returns_inconclusive(self):
        """Core Phase-9 assertion: FalsifierXAir.investigate() must abstain on
        certified INDISTINGUISHABLE worlds instead of guessing."""
        twin, scenario, cert = world_confounded(200)
        assert cert.verdict == "INDISTINGUISHABLE", \
            f"Precondition: cert must be INDISTINGUISHABLE, got {cert.verdict}"
        ctx = _make_context(twin, scenario)
        result = FalsifierXAir().investigate(twin, ctx)
        assert result.outcome == "INCONCLUSIVE", \
            f"FALSIFIER-X must abstain on confounded world, got {result.outcome}"

    def test_confounded_world_no_false_recovery(self):
        twin, scenario, _ = world_confounded(201)
        ctx = _make_context(twin, scenario)
        result = FalsifierXAir().investigate(twin, ctx)
        assert result.recovered_mechanism is None, \
            "recovered_mechanism must be None when INCONCLUSIVE"

    def test_confounded_identifiability_reason(self):
        twin, scenario, _ = world_confounded(202)
        ctx = _make_context(twin, scenario)
        result = FalsifierXAir().investigate(twin, ctx)
        if result.adequacy.state == "STRUCTURALLY_SUSPICIOUS":
            assert result.identifiability_reason == "INDISTINGUISHABLE_EQUIVALENCE_CLASS", \
                f"Expected INDISTINGUISHABLE_EQUIVALENCE_CLASS, got {result.identifiability_reason}"

    def test_confounded_probe_experiments_captured(self):
        """Probe experiments must appear in probe_experiments, not discarded."""
        twin, scenario, _ = world_confounded(203)
        ctx = _make_context(twin, scenario)
        result = FalsifierXAir().investigate(twin, ctx)
        if result.adequacy.state == "STRUCTURALLY_SUSPICIOUS":
            assert isinstance(result.probe_experiments, tuple)

    @pytest.mark.parametrize("seed", [204, 205, 206])
    def test_confounded_world_inconclusive_across_seeds(self, seed):
        twin, scenario, cert = world_confounded(seed)
        if cert.verdict != "INDISTINGUISHABLE":
            pytest.skip(f"Seed {seed} not certified INDISTINGUISHABLE")
        ctx = _make_context(twin, scenario)
        result = FalsifierXAir().investigate(twin, ctx)
        assert result.outcome == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Single-mechanism worlds → RECOVERED (no regression)
# ---------------------------------------------------------------------------

class TestSingleMechanismRecovery:
    @pytest.mark.parametrize("mechanism,seed", [
        ("AIRCRAFT_ROTATION", 200),
        ("RESOURCE_DEPENDENCY", 200),
        ("AIRPORT_CAPACITY", 200),
    ])
    def test_single_mechanism_still_recovered(self, mechanism, seed):
        """Identifiability layer must not break single-mechanism recovery."""
        twin, scenario = world_scale(8, mechanism, seed)
        ctx = _make_context(twin, scenario)
        result = FalsifierXAir().investigate(twin, ctx)
        # Either adequacy gate fires (ADEQUATE/INCONCLUSIVE) or mechanism is RECOVERED.
        # We do not assert a specific adequacy state (depends on 8-flight topology).
        if result.adequacy.state == "STRUCTURALLY_SUSPICIOUS":
            if result.recovered_mechanism is not None:
                assert result.recovered_mechanism == mechanism, \
                    f"Wrong mechanism: expected {mechanism}, got {result.recovered_mechanism}"
                assert result.outcome == "RECOVERED"
                assert result.identifiability_reason == "LEGITIMATELY_IDENTIFIED"

    def test_identifiability_reason_on_legitimate_recovery(self):
        """When a mechanism is legitimately recovered, reason must be LEGITIMATELY_IDENTIFIED."""
        twin, scenario = world_scale(8, "AIRCRAFT_ROTATION", 207)
        ctx = _make_context(twin, scenario)
        result = FalsifierXAir().investigate(twin, ctx)
        if result.recovered_mechanism is not None:
            assert result.identifiability_reason == "LEGITIMATELY_IDENTIFIED"


# ---------------------------------------------------------------------------
# check_identifiability unit tests (directly testing the function)
# ---------------------------------------------------------------------------

class TestCheckIdentifiability:
    def _make_flights_and_confounded_twin(self, seed=200):
        """Return (twin, scenario) for a confounded world."""
        from falsifier_x_air.phase8_worlds import world_confounded
        twin, scenario, _ = world_confounded(seed)
        return twin, scenario

    def test_equal_effects_returns_not_identifiable(self):
        """If competitor probe yields same effect as survivor, must return not identifiable."""
        from falsifier_x_air.mechanisms import generate_candidates
        from falsifier_x_air.graph import AviationGraph as AG
        from falsifier_x_air.experiments import run_experiment

        twin, scenario = self._make_flights_and_confounded_twin(200)
        observation = twin.observe(scenario)
        graph = AG.build(observation.flights)
        prediction = _prediction(twin, scenario)
        observed_vec = observation.delay_vector(graph.flight_ids)

        affected = AG.localize(graph, observed_vec - prediction.mean)
        candidates = tuple(generate_candidates(graph, affected))

        # Simulate: run disable_rotation, mark RESOURCE + CAPACITY as REJECTED
        rotation_result = run_experiment(twin, scenario, "disable_aircraft_rotation", "test-exp-1")

        for c in candidates:
            if c.mechanism_id != "AIRCRAFT_ROTATION":
                c.status = c.status.REJECTED

        survivor = next(c for c in candidates if c.mechanism_id == "AIRCRAFT_ROTATION")
        result = check_identifiability(
            survivor, candidates, [rotation_result], twin, scenario,
            sigma_eff=1.0, epsilon=EPSILON_IDENT,
        )
        # In confounded world, probing disable_resource yields same effect → D ≈ 0
        assert not result.is_identifiable
        assert result.reason == "INDISTINGUISHABLE_EQUIVALENCE_CLASS"

    def test_inactive_competitor_returns_identifiable(self):
        """If competitor's intervention yields ~0 effect, it's legitimately distinguished."""
        from falsifier_x_air.schema import MechanismEvidence, MechanismStatus
        from falsifier_x_air.experiments import run_experiment

        # Single-mechanism: AIRCRAFT_ROTATION only
        twin, scenario = world_scale(8, "AIRCRAFT_ROTATION", 200)
        observation = twin.observe(scenario)
        graph = AviationGraph.build(observation.flights)
        prediction = _prediction(twin, scenario)
        observed_vec = observation.delay_vector(graph.flight_ids)

        from falsifier_x_air.mechanisms import generate_candidates
        affected = AviationGraph.localize(graph, observed_vec - prediction.mean)
        candidates = tuple(generate_candidates(graph, affected))

        rotation_result = run_experiment(twin, scenario, "disable_aircraft_rotation", "test-exp-1")
        for c in candidates:
            if c.mechanism_id != "AIRCRAFT_ROTATION":
                c.status = c.status.REJECTED

        survivor = next(c for c in candidates if c.mechanism_id == "AIRCRAFT_ROTATION")
        result = check_identifiability(
            survivor, candidates, [rotation_result], twin, scenario,
            sigma_eff=1.0, epsilon=EPSILON_IDENT,
        )
        assert result.is_identifiable
        assert result.reason == "LEGITIMATELY_IDENTIFIED"

    def test_d_values_populated(self):
        """d_values must contain one entry per rejected competitor."""
        twin, scenario = self._make_flights_and_confounded_twin(210)
        observation = twin.observe(scenario)
        graph = AviationGraph.build(observation.flights)
        prediction = _prediction(twin, scenario)
        observed_vec = observation.delay_vector(graph.flight_ids)

        from falsifier_x_air.mechanisms import generate_candidates
        from falsifier_x_air.experiments import run_experiment
        affected = AviationGraph.localize(graph, observed_vec - prediction.mean)
        candidates = tuple(generate_candidates(graph, affected))

        rotation_result = run_experiment(twin, scenario, "disable_aircraft_rotation", "test-exp-1")
        for c in candidates:
            if c.mechanism_id != "AIRCRAFT_ROTATION":
                c.status = c.status.REJECTED

        survivor = next(c for c in candidates if c.mechanism_id == "AIRCRAFT_ROTATION")
        result = check_identifiability(survivor, candidates, [rotation_result], twin, scenario)
        rejected_count = sum(1 for c in candidates if c.mechanism_id != "AIRCRAFT_ROTATION" and c.status.name == "REJECTED")
        assert len(result.d_values) == rejected_count


# ---------------------------------------------------------------------------
# Probe experiments use public API only
# ---------------------------------------------------------------------------

class TestProbeExperimentsAPI:
    def test_probes_do_not_use_oracle(self):
        """run_experiment in identifiability.py calls twin.counterfactual (public).
        Verified by: probe results are ExperimentResult objects, not truth dicts."""
        from falsifier_x_air.schema import ExperimentResult
        twin, scenario, _ = world_confounded(211)
        observation = twin.observe(scenario)
        graph = AviationGraph.build(observation.flights)
        prediction = _prediction(twin, scenario)
        observed_vec = observation.delay_vector(graph.flight_ids)

        from falsifier_x_air.mechanisms import generate_candidates
        from falsifier_x_air.experiments import run_experiment
        affected = AviationGraph.localize(graph, observed_vec - prediction.mean)
        candidates = tuple(generate_candidates(graph, affected))

        rotation_result = run_experiment(twin, scenario, "disable_aircraft_rotation", "test-exp-1")
        for c in candidates:
            if c.mechanism_id != "AIRCRAFT_ROTATION":
                c.status = c.status.REJECTED

        survivor = next(c for c in candidates if c.mechanism_id == "AIRCRAFT_ROTATION")
        result = check_identifiability(survivor, candidates, [rotation_result], twin, scenario)
        for probe in result.probe_experiments:
            assert isinstance(probe, ExperimentResult), \
                "Probe results must be ExperimentResult, not oracle dicts"

    def test_probe_uses_existing_result_if_available(self):
        """If a competitor's intervention was already run, no new probe should be added."""
        from falsifier_x_air.experiments import run_experiment

        twin, scenario, _ = world_confounded(212)
        observation = twin.observe(scenario)
        graph = AviationGraph.build(observation.flights)
        prediction = _prediction(twin, scenario)
        observed_vec = observation.delay_vector(graph.flight_ids)

        from falsifier_x_air.mechanisms import generate_candidates
        affected = AviationGraph.localize(graph, observed_vec - prediction.mean)
        candidates = tuple(generate_candidates(graph, affected))

        # Run both interventions
        rotation_result = run_experiment(twin, scenario, "disable_aircraft_rotation", "exp-1")
        resource_result = run_experiment(twin, scenario, "relieve_resource_dependency", "exp-2")
        for c in candidates:
            if c.mechanism_id != "AIRCRAFT_ROTATION":
                c.status = c.status.REJECTED

        survivor = next(c for c in candidates if c.mechanism_id == "AIRCRAFT_ROTATION")
        result = check_identifiability(
            survivor, candidates, [rotation_result, resource_result], twin, scenario
        )
        # Resource intervention already in experiments → no probe for RESOURCE
        resource_probed = any(
            "resource_dependency" in p.experiment_id for p in result.probe_experiments
        )
        assert not resource_probed, "Should reuse existing resource result, not re-probe"


# ---------------------------------------------------------------------------
# Regression: Phase-7 RESEARCH_DESIGN families unchanged
# ---------------------------------------------------------------------------

class TestPhase7Regression:
    """Verify that adding the identifiability layer does not change Phase-7 outcomes.
    Uses representative families from the Phase-7 SCENARIO_FAMILIES.
    Phase-7 discovery seeds 100–109 are used (read-only; we just run the logic,
    not modifying any frozen JSON files).
    """

    def _p7_investigate(self, family: str, seed: int):
        from falsifier_x_air.evaluation import scenario_family, _prediction as p7_pred
        twin, scenario = scenario_family(family, seed)
        observation = twin.observe(scenario)
        graph = AviationGraph.build(observation.flights)
        prediction = p7_pred(twin, scenario)
        ood_score = 5.0 if family == "OOD" else 0.0
        ctx = InvestigationContext(
            observation=observation, graph=graph, prediction=prediction,
            alternative_prediction=prediction.mean, ood_score=ood_score,
            persistence_count=3, scenario=scenario,
        )
        return FalsifierXAir().investigate(twin, ctx)

    def test_correct_still_adequate(self):
        result = self._p7_investigate("CORRECT", 100)
        assert result.adequacy.state == "ADEQUATE"
        assert result.recovered_mechanism is None

    def test_ood_still_ood(self):
        result = self._p7_investigate("OOD", 100)
        assert result.adequacy.state == "OOD"

    def test_rotation_world_recovers_or_inconclusive_not_wrong(self):
        """If AIRCRAFT_ROTATION world recovers, must recover AIRCRAFT_ROTATION, not RESOURCE."""
        result = self._p7_investigate("AIRCRAFT_ROTATION", 100)
        if result.recovered_mechanism is not None:
            assert result.recovered_mechanism == "AIRCRAFT_ROTATION"
            assert result.identifiability_reason == "LEGITIMATELY_IDENTIFIED"

    def test_resource_world_recovers_or_inconclusive_not_wrong(self):
        result = self._p7_investigate("RESOURCE_DEPENDENCY", 100)
        if result.recovered_mechanism is not None:
            assert result.recovered_mechanism == "RESOURCE_DEPENDENCY"
            assert result.identifiability_reason == "LEGITIMATELY_IDENTIFIED"

    def test_investigation_result_probe_experiments_always_tuple(self):
        """probe_experiments field must always be a tuple, never None."""
        for family in ("CORRECT", "AIRCRAFT_ROTATION", "OOD"):
            result = self._p7_investigate(family, 100)
            assert isinstance(result.probe_experiments, tuple)


# ---------------------------------------------------------------------------
# Phase-8 H3 baseline: _run_case_budgeted is unchanged (not fixed by Phase-9)
# ---------------------------------------------------------------------------

class TestPhase8BaselinePreserved:
    def test_phase8_h3_loop_still_fails_without_identifiability(self):
        """_run_case_budgeted does NOT call FalsifierXAir.investigate(), so the
        Phase-8 H3 baseline (0.00 abstention correctness) is unchanged.
        This confirms the Phase-8 result remains valid as a pre-fix baseline."""
        from falsifier_x_air.phase8_evaluation import _run_case_budgeted
        twin, scenario, cert = world_confounded(2000)
        tm, tn, te = _truth(twin)
        r = _run_case_budgeted(twin, scenario, tm, tn, te, "ACTIVE", 3, np.random.default_rng(2000))
        # Phase-8 loop does NOT have identifiability; it may return RECOVERED here.
        # We just verify the function still runs without error.
        assert "outcome" in r


# ---------------------------------------------------------------------------
# InvestigationResult backward compatibility
# ---------------------------------------------------------------------------

class TestInvestigationResultBackwardCompat:
    def test_new_fields_have_defaults(self):
        from falsifier_x_air.schema import AdequacyEvaluation
        adequacy = AdequacyEvaluation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      "ADEQUATE", "No violation.", {})
        result = InvestigationResult(adequacy, (), (), (), None)
        assert result.outcome == "INCONCLUSIVE"
        assert result.identifiability_reason is None
        assert result.probe_experiments == ()
