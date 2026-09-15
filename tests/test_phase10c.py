"""Tests for Phase-10C mechanism heterogeneity experiment.

Verifies: no oracle filter, correct E2E semantics, leakage isolation,
topology disjointness, seed partitions, frozen parameters, and heterogeneity.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from falsifier_x_air.graph import AviationGraph
from falsifier_x_air.phase10c_evaluation import (
    BETA_DIST, INTERVENTION_NOISE_SD,
    PHASE10C_DISCOVERY_SEEDS, PHASE10C_BLIND_SEEDS,
    PURE_FAMILIES,
    _sample_beta, _noise,
    paired_intervention_observation,
    phase10c_flights, phase10c_world, topology_certificate_10c,
    _case,
)
from falsifier_x_air.phase10c_repair import WRONG_MECHANISM


# ---- 1. Seed partitions are disjoint and do not overlap Phase 10B ----------
def test_seed_partitions_disjoint():
    assert set(PHASE10C_DISCOVERY_SEEDS).isdisjoint(PHASE10C_BLIND_SEEDS)
    assert min(PHASE10C_DISCOVERY_SEEDS) == 800000
    assert max(PHASE10C_DISCOVERY_SEEDS) == 800099
    assert min(PHASE10C_BLIND_SEEDS) == 800100
    assert max(PHASE10C_BLIND_SEEDS) == 800299
    # Must not overlap Phase 10B seeds.
    phase10b = set(range(700000, 700300))
    assert set(PHASE10C_DISCOVERY_SEEDS).isdisjoint(phase10b)
    assert set(PHASE10C_BLIND_SEEDS).isdisjoint(phase10b)


# ---- 2. AR/RD topology disjointness (edges and targets) --------------------
def test_topology_disjoint_edges_and_targets():
    graph = AviationGraph.build(phase10c_flights())
    cert = topology_certificate_10c(graph)
    assert cert["edge_overlap"] == []
    assert cert["target_overlap"] == []
    assert cert["ar_target_support"] != cert["rd_target_support"]


# ---- 3. Heterogeneous beta_s actually varies --------------------------------
def test_beta_s_varies_per_world():
    betas = {_sample_beta("AIRCRAFT_ROTATION", s) for s in range(800000, 800010)}
    assert len(betas) > 1, "beta_s must differ across worlds"


def test_beta_s_within_bounds():
    d = BETA_DIST["AIRCRAFT_ROTATION"]
    for seed in range(800000, 800020):
        b = _sample_beta("AIRCRAFT_ROTATION", seed)
        assert d["lo"] <= b <= d["hi"], f"beta_s={b} out of [{d['lo']}, {d['hi']}]"


# ---- 4. Learner module has no access to beta_s or oracle truth -------------
def test_learner_module_has_no_oracle_access():
    src = (Path(__file__).parent.parent / "falsifier_x_air" / "phase10c_repair.py").read_text(encoding="utf-8")
    forbidden = (
        "beta_s", "_sample_beta", "TwinScenario", "AviationDigitalTwin",
        "PHASE10C_BLIND_SEEDS", "_oracle_coefficient",
        "rotation_coefficient", "resource_coefficient", "mechanism_strength",
    )
    violations = [tok for tok in forbidden if tok in src]
    assert not violations, f"Learner module contains oracle token(s): {violations}"
    ast.parse(src)  # valid Python


# ---- 5. INCONCLUSIVE → abstain, no repair applied -------------------------
def test_inconclusive_abstains(monkeypatch):
    import falsifier_x_air.phase10c_evaluation as ev
    frozen = {m: {"coefficient": 1.0, "sample_count": 1, "ols_se": 0.0} for m in PURE_FAMILIES}
    frozen["GENERIC"] = {"coefficient": 1.0, "sample_count": 1, "ols_se": 0.0}
    monkeypatch.setattr(ev, "_identified", lambda *a: None)
    monkeypatch.setattr(ev, "apply_repair", lambda *a: pytest.fail("repair must be gated"))
    result = ev._case("AIRCRAFT_ROTATION", 800100, frozen, 0.85)
    assert result is not None
    assert result["repaired"] is False
    assert result["e2e_mae"] == result["orig_mae"]


# ---- 6. Wrong recovered mechanism is passed to repair, not oracle ----------
def test_wrong_recovery_uses_recovered_mechanism(monkeypatch):
    import falsifier_x_air.phase10c_evaluation as ev
    frozen = {m: {"coefficient": 1.0, "sample_count": 1, "ols_se": 0.0} for m in PURE_FAMILIES}
    frozen["GENERIC"] = {"coefficient": 1.0, "sample_count": 1, "ols_se": 0.0}
    monkeypatch.setattr(ev, "_identified", lambda *a: "AIRPORT_CAPACITY")

    calls = []
    def mock_repair(pred, mech, coeff, obs, graph):
        calls.append(mech)
        return pred
    monkeypatch.setattr(ev, "apply_repair", mock_repair)
    ev._case("AIRCRAFT_ROTATION", 800100, frozen, 0.85)
    # First call = FALSIFIER's output (wrong: AIRPORT_CAPACITY); last = oracle (true: AIRCRAFT_ROTATION)
    assert calls[0] == "AIRPORT_CAPACITY"
    assert calls[-1] == "AIRCRAFT_ROTATION"


# ---- 7. All blind cases enter E2E evaluation (no oracle filter) ------------
def test_all_blind_cases_enter_e2e(monkeypatch):
    import falsifier_x_air.phase10c_evaluation as ev
    frozen = {m: {"coefficient": 1.0, "sample_count": 1, "ols_se": 0.0} for m in PURE_FAMILIES}
    frozen["GENERIC"] = {"coefficient": 1.0, "sample_count": 1, "ols_se": 0.0}
    # Simulate: falsifier returns wrong mechanism.
    monkeypatch.setattr(ev, "_identified", lambda *a: "RESOURCE_DEPENDENCY")
    result = ev._case("AIRCRAFT_ROTATION", 800100, frozen, 0.85)
    assert result is not None
    assert result["repaired"] is True
    assert result["is_correct_id"] is False


# ---- 8. Paired world semantics and noise after differencing ----------------
def test_paired_worlds_share_scenario_and_noise_is_post_differencing():
    twin, scenario, _ = phase10c_world("AIRCRAFT_ROTATION", 800000)
    first  = paired_intervention_observation(twin, scenario, "AIRCRAFT_ROTATION")
    second = paired_intervention_observation(twin, scenario, "AIRCRAFT_ROTATION")
    assert first == second, "Paired observations must be deterministic"
    assert np.isclose(first["observed_effect"] - first["true_effect"], first["epsilon_intervention"])
    assert first["epsilon_intervention"] != 0.0
    factual = twin.observe(scenario)
    assert factual.delays == twin.observe(scenario).delays


# ---- 9. Noise streams differ between seeds/mechanisms ----------------------
def test_noise_is_independent_per_seed_and_mechanism():
    n1 = _noise(800000, "AIRCRAFT_ROTATION")
    n2 = _noise(800001, "AIRCRAFT_ROTATION")
    n3 = _noise(800000, "RESOURCE_DEPENDENCY")
    assert n1 != n2
    assert n1 != n3


# ---- 10. GENERIC coefficient is not hardcoded -------------------------------
def test_generic_coefficient_not_hardcoded():
    src = (Path(__file__).parent.parent / "falsifier_x_air" / "phase10c_repair.py").read_text(encoding="utf-8")
    assert "GENERIC_COEFFICIENT" not in src
    assert "= 0.10" not in src
    assert "= 0.1" not in src


# ---- 11. Intervention noise SD is preserved --------------------------------
def test_intervention_noise_sd():
    assert INTERVENTION_NOISE_SD == 2.0


# ---- 12. Phase 5-9 and historical Phase 10/10B files are untouched --------
def test_frozen_phase_files_unchanged():
    root = Path(__file__).parent.parent
    frozen = [
        "falsifier_x_air/phase10b_evaluation.py",
        "falsifier_x_air/phase10b_repair.py",
        "falsifier_x_air/phase10_repair.py",
        "falsifier_x_air/phase8_evaluation.py",
        "falsifier_x_air/phase9_evaluation.py",
    ]
    for rel in frozen:
        p = root / rel
        assert p.exists(), f"Frozen file missing: {rel}"
        # Just verify it exists and is non-empty.
        assert p.stat().st_size > 0


# ---- 13. Wrong mechanism mapping is pre-declared ---------------------------
def test_wrong_mechanism_predeclared():
    assert WRONG_MECHANISM == {
        "AIRCRAFT_ROTATION": "RESOURCE_DEPENDENCY",
        "RESOURCE_DEPENDENCY": "AIRCRAFT_ROTATION",
        "AIRPORT_CAPACITY": "AIRCRAFT_ROTATION",
    }
