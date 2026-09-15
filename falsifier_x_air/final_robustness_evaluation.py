"""Final Robustness and External Validation block for FALSIFIER-X AIR.

Addresses six scientific weaknesses:
  W1: Synthetic benchmark dependence  -> held-out causal world generalization
  W2: Narrow mechanism class          -> topology/context variants
  W3: Low AR/RD coverage              -> coverage-reliability tradeoff K=1,2,3
  W4: Linear repair only              -> nonlinear threshold mechanism
  W5: External baselines              -> residual-shift, random-mechanism
  W6: Novelty                         -> novelty_and_baseline_analysis.md

Seeds: disjoint from Phase 10B (700000-700299) and Phase 10C (800000-800299).
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

from .evaluation import _prediction
from .falsifier import FalsifierXAir, InvestigationContext
from .experiments import ExperimentConfig, select_baseline
from .graph import AviationGraph
from .schema import Flight, NetworkObservation
from .twin import AviationDigitalTwin, TwinScenario
from .final_robustness_repair import (
    MECHANISMS, WRONG_MECHANISM,
    observable_feature, generic_feature, nonlinear_feature,
    estimate_from_discovery, apply_repair, apply_generic_repair,
    apply_nonlinear_repair, residual_magnitude_repair, random_mechanism_repair,
)

# ---------------------------------------------------------------------------
# Seed partitions -- disjoint from all prior phases
# ---------------------------------------------------------------------------
DISC_SEEDS     = tuple(range(900000, 900100))
BLIND_SEEDS    = tuple(range(900100, 900400))
NL_DISC_SEEDS  = tuple(range(900400, 900450))
NL_BLIND_SEEDS = tuple(range(900450, 900550))

_ALL_SEEDS = set(DISC_SEEDS) | set(BLIND_SEEDS) | set(NL_DISC_SEEDS) | set(NL_BLIND_SEEDS)
assert _ALL_SEEDS.isdisjoint(range(700000, 700300)), "Phase 10B overlap"
assert _ALL_SEEDS.isdisjoint(range(800000, 800300)), "Phase 10C overlap"
assert set(DISC_SEEDS).isdisjoint(BLIND_SEEDS)
assert set(NL_DISC_SEEDS).isdisjoint(NL_BLIND_SEEDS)

INTERVENTION_NOISE_SD = 2.0
NL_TAU = 5.0   # Pre-declared before any result is observed; never tuned.
NL_COEFF = 3.0  # True nonlinear coefficient; evaluator-only.


# ---------------------------------------------------------------------------
# Topology definitions
# ---------------------------------------------------------------------------
def _base_flights():
    """Phase 10B/10C frozen topology (10 flights)."""
    return (
        Flight("F1", "A", "B", "AC1", 0, "R1"), Flight("F2", "B", "C", "AC1", 1, "R2"),
        Flight("F3", "C", "D", "AC1", 2, "R3"), Flight("F4", "E", "F", "AC2", 0, "R4"),
        Flight("F5", "F", "G", "AC2", 1, "R5"), Flight("F6", "G", "H", "AC2", 2, "R6"),
        Flight("F7", "I", "J", "AC3", 1, "R1"), Flight("F8", "K", "L", "AC4", 1, "R4"),
        Flight("F9", "M", "N", "AC5", 2, "R2"), Flight("F10", "O", "P", "AC6", 2, "R5"),
    )


def _heldout_flights():
    """Held-out topology: 12 flights, longer AR chains (4-leg), denser RD pools.
    Structurally distinct: longer dependency paths, more shared resources."""
    return (
        Flight("F1",  "A", "B", "AC1", 0, "R1"), Flight("F2",  "B", "C", "AC1", 1, "R2"),
        Flight("F3",  "C", "D", "AC1", 2, "R3"), Flight("F4",  "D", "E", "AC1", 3, "R4"),
        Flight("F5",  "F", "G", "AC2", 0, "R1"), Flight("F6",  "G", "H", "AC2", 1, "R2"),
        Flight("F7",  "H", "I", "AC2", 2, "R3"), Flight("F8",  "I", "J", "AC2", 3, "R4"),
        Flight("F9",  "K", "L", "AC3", 1, "R1"), Flight("F10", "L", "M", "AC4", 1, "R2"),
        Flight("F11", "N", "O", "AC5", 2, "R3"), Flight("F12", "P", "Q", "AC6", 2, "R4"),
    )


# ---------------------------------------------------------------------------
# World factories (evaluator-only)
# ---------------------------------------------------------------------------
def _noise(seed: int, mechanism: str) -> float:
    code = {m: i for i, m in enumerate(MECHANISMS)}[mechanism]
    return float(np.random.default_rng(
        np.random.SeedSequence([seed, 10_030, code])).normal(0.0, INTERVENTION_NOISE_SD))


def _nl_noise(seed: int) -> float:
    return float(np.random.default_rng(
        np.random.SeedSequence([seed, 10_031, 99])).normal(0.0, INTERVENTION_NOISE_SD))


def base_world(family: str, seed: int):
    """Standard development-configuration world (same as Phase 10B/10C)."""
    scenario = TwinScenario(
        f"fr-base-{family.lower()}-{seed}", weather=1.00, capacity=0.55, seed=seed,
        family="FR_BASE", noise_scale=1.5, mechanism_strength=1.0,
        rotation_coefficient=0.85, resource_coefficient=0.60,
    )
    return AviationDigitalTwin(_base_flights(), {family}, seed), scenario


def heldout_world(family: str, seed: int):
    """Held-out world: longer chains, denser resources, higher weather stress.
    Coefficients drawn from a range shifted from the development configuration."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 77_777, hash(family) & 0xFFFF]))
    if family == "AIRCRAFT_ROTATION":
        rot = float(np.clip(rng.normal(0.90, 0.10), 0.70, 1.10))
        res, strength = 0.60, 1.0
    elif family == "RESOURCE_DEPENDENCY":
        res = float(np.clip(rng.normal(0.60, 0.08), 0.45, 0.75))
        rot, strength = 0.85, 1.0
    else:
        rot, res = 0.85, 0.60
        strength = float(np.clip(rng.normal(1.1, 0.10), 0.85, 1.35))
    scenario = TwinScenario(
        f"fr-held-{family.lower()}-{seed}", weather=1.40, capacity=0.45, seed=seed,
        family="FR_HELD", noise_scale=2.0, mechanism_strength=strength,
        rotation_coefficient=rot, resource_coefficient=res,
    )
    return AviationDigitalTwin(_heldout_flights(), {family}, seed), scenario


def nl_world(seed: int):
    """Nonlinear (threshold) world. The twin has NO AR mechanism active.
    Nonlinear effect is applied externally at evaluation level."""
    scenario = TwinScenario(
        f"fr-nl-{seed}", weather=1.00, capacity=0.55, seed=seed,
        family="FR_NL", noise_scale=1.5, mechanism_strength=0.0,
        rotation_coefficient=0.0, resource_coefficient=0.0,
    )
    twin = AviationDigitalTwin(_base_flights(), set(), seed)
    return twin, scenario


def _nl_observation(twin, scenario, disable: bool = False) -> NetworkObservation:
    """Simulate nonlinear threshold AR effect externally."""
    base_obs = twin.observe(scenario)
    delays = dict(base_obs.delays)
    graph = AviationGraph.build(twin.flights)
    if not disable:
        for source, target, data in graph.graph.edges(data=True):
            if data.get("relation") == "AIRCRAFT_ROTATION":
                delays[target] = delays.get(target, 0.0) + NL_COEFF * max(
                    0.0, delays.get(source, 0.0) - NL_TAU)
    return NetworkObservation(
        twin.flights, delays, scenario.weather, scenario.capacity, scenario.scenario_id)


# ---------------------------------------------------------------------------
# FALSIFIER-X runner helpers
# ---------------------------------------------------------------------------
def _run_falsifier(twin, scenario, max_experiments: int = 3) -> str | None:
    obs = twin.observe(scenario)
    graph = AviationGraph.build(obs.flights)
    pred = _prediction(twin, scenario)
    cfg = ExperimentConfig(maximum_experiments=max_experiments)
    result = FalsifierXAir(experiment_config=cfg).investigate(
        twin, InvestigationContext(obs, graph, pred, pred.mean, 0.0, 3, scenario))
    return (result.recovered_mechanism
            if result.identifiability_reason == "LEGITIMATELY_IDENTIFIED" else None)


def _run_falsifier_ablation(twin, scenario, strategy: str, rng_seed: int,
                            max_experiments: int = 3) -> str | None:
    """Run FALSIFIER-X with alternative intervention strategy."""
    from .mechanisms import generate_candidates
    from .experiments import update_evidence, run_experiment, ActiveExperimentSelector
    from .adequacy import StructuralAdequacyDetector
    from .identifiability import EPSILON_IDENT, check_identifiability

    obs = twin.observe(scenario)
    graph = AviationGraph.build(obs.flights)
    pred = _prediction(twin, scenario)
    cfg = ExperimentConfig(maximum_experiments=max_experiments)
    detector = StructuralAdequacyDetector()
    observed = obs.delay_vector(graph.flight_ids)
    residuals = observed - pred.mean
    adequacy = detector.evaluate(
        observed, pred, 0.0, 3,
        AviationGraph.residual_concentration(graph, residuals), pred.mean,
        AviationGraph.directional_residual_lift(graph, residuals, pred.epistemic_std),
    )
    if adequacy.state != "STRUCTURALLY_SUSPICIOUS":
        return None
    affected = AviationGraph.localize(graph, residuals)
    candidates = generate_candidates(graph, affected)
    tried: set[str] = set()
    results = []
    rng = np.random.default_rng(rng_seed)
    selector = ActiveExperimentSelector()

    for idx in range(0 if strategy == "PASSIVE" else cfg.maximum_experiments):
        plausible = [c for c in candidates if c.status != c.status.REJECTED]
        if not plausible:
            break
        intervention = select_baseline(strategy, tuple(plausible), graph, obs, pred.mean, tried, rng)
        if intervention is None:
            break
        expected = selector.predicted_effects(tuple(candidates), graph, obs, pred.mean, intervention)
        result = run_experiment(twin, scenario, intervention, f"{obs.scenario_id}-abl-{idx}")
        update_evidence(tuple(candidates), expected, result, cfg)
        results.append(result)
        tried.add(intervention)

    survivors = [c for c in candidates if c.status != c.status.REJECTED]
    if len(survivors) != 1:
        return None
    candidate = survivors[0]
    if (candidate.observations < cfg.minimum_confirmation_experiments
            or candidate.log_evidence < cfg.support_log_evidence):
        return None
    ident_result = check_identifiability(
        candidate, tuple(candidates), results, twin, scenario,
        scenario_id_prefix=f"{obs.scenario_id}-abl-ident",
        sigma_eff=float(pred.epistemic_std.mean()), epsilon=EPSILON_IDENT,
    )
    return candidate.mechanism_id if ident_result.is_identifiable else None


# ---------------------------------------------------------------------------
# Discovery builders
# ---------------------------------------------------------------------------
def _paired_effect(twin, scenario, mechanism: str, noise_fn):
    imap = {"AIRCRAFT_ROTATION": "disable_aircraft_rotation",
            "RESOURCE_DEPENDENCY": "relieve_resource_dependency",
            "AIRPORT_CAPACITY": "increase_capacity"}
    factual = twin.observe(scenario)
    cf = twin.observe(scenario, {imap[mechanism]: True})
    true_effect = float(sum(factual.delays.values()) - sum(cf.delays.values()))
    eps = noise_fn(scenario.seed, mechanism)
    return {"true_effect": true_effect, "observed_effect": true_effect + eps}


def build_discovery(seeds, world_fn) -> list[dict]:
    records = []
    for seed in seeds:
        for family in MECHANISMS:
            twin, scenario = world_fn(family, seed)
            recovered = _run_falsifier(twin, scenario)
            if recovered is None:
                continue
            obs = twin.observe(scenario)
            graph = AviationGraph.build(obs.flights)
            paired = _paired_effect(twin, scenario, recovered, _noise)
            records.append({
                "seed": seed, "true_family": family, "mechanism": recovered,
                "feature_sum": float(observable_feature(recovered, obs, graph).sum()),
                "generic_feature_sum": float(generic_feature(obs, graph).sum()),
                **paired,
            })
    return records


def build_nl_discovery(seeds) -> list[dict]:
    records = []
    for seed in seeds:
        twin, scenario = nl_world(seed)
        obs_fact = _nl_observation(twin, scenario, disable=False)
        obs_cf   = _nl_observation(twin, scenario, disable=True)
        true_effect = float(sum(obs_fact.delays.values()) - sum(obs_cf.delays.values()))
        eps = _nl_noise(seed)
        graph = AviationGraph.build(twin.flights)
        records.append({
            "seed": seed, "true_family": "NL_AR", "mechanism": "AIRCRAFT_ROTATION",
            "feature_sum": float(observable_feature("AIRCRAFT_ROTATION", obs_fact, graph).sum()),
            "generic_feature_sum": float(generic_feature(obs_fact, graph).sum()),
            "nl_feature_sum": float(nonlinear_feature(obs_fact, graph, NL_TAU).sum()),
            "observed_effect": true_effect + eps, "true_effect": true_effect,
        })
    return records


# ---------------------------------------------------------------------------
# Blind evaluation helpers
# ---------------------------------------------------------------------------
def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _oracle_coeff(scenario, family: str) -> float:
    if family == "AIRCRAFT_ROTATION":   return scenario.rotation_coefficient
    if family == "RESOURCE_DEPENDENCY": return scenario.resource_coefficient
    return 20.0 * scenario.mechanism_strength


def _eval_case(family: str, seed: int, frozen: dict, world_fn) -> dict:
    twin, scenario = world_fn(family, seed)
    recovered = _run_falsifier(twin, scenario)
    obs = twin.observe(scenario)
    graph = AviationGraph.build(obs.flights)
    pred = _prediction(twin, scenario).mean
    target = obs.delay_vector(graph.flight_ids)
    result = {
        "family": family, "seed": seed, "recovered": recovered,
        "is_correct_id": recovered == family,
        "orig_mae": _mae(pred, target), "repaired": False,
    }
    if recovered is None or recovered not in frozen:
        result["e2e_mae"] = result["orig_mae"]
        return result
    result["repaired"] = True
    cc = float(frozen[recovered]["coefficient"])
    wm = WRONG_MECHANISM[recovered]
    wc = float(frozen[wm]["coefficient"])
    gc = float(frozen["GENERIC"]["coefficient"])
    oc = _oracle_coeff(scenario, family)
    result["correct_mae"]  = _mae(apply_repair(pred, recovered, cc, obs, graph), target)
    result["wrong_mae"]    = _mae(apply_repair(pred, wm, wc, obs, graph), target)
    result["generic_mae"]  = _mae(apply_generic_repair(pred, gc, obs, graph), target)
    result["oracle_mae"]   = _mae(apply_repair(pred, family, oc, obs, graph), target)
    # External baselines (W5)
    result["residual_mae"] = _mae(residual_magnitude_repair(pred, None, obs, graph), target)
    result["random_mae"]   = _mae(random_mechanism_repair(pred, obs, graph, cc, rng_seed=seed), target)
    result["e2e_mae"]      = result["correct_mae"]
    return result


def _eval_nl_case(seed: int, nl_frozen: dict) -> dict:
    twin, scenario = nl_world(seed)
    obs_fact = _nl_observation(twin, scenario, disable=False)
    graph = AviationGraph.build(twin.flights)
    pred = _prediction(twin, scenario).mean
    target = obs_fact.delay_vector(graph.flight_ids)
    lin_cc = float(nl_frozen.get("AIRCRAFT_ROTATION", {}).get("coefficient", 0.0))
    nl_cc  = float(nl_frozen.get("NONLINEAR", {}).get("coefficient", 0.0))
    gc     = float(nl_frozen.get("GENERIC", {}).get("coefficient", 0.0))
    return {
        "seed": seed,
        "orig_mae":    _mae(pred, target),
        "linear_mae":  _mae(apply_repair(pred, "AIRCRAFT_ROTATION", lin_cc, obs_fact, graph), target),
        "nl_mae":      _mae(apply_nonlinear_repair(pred, nl_cc, obs_fact, graph, NL_TAU), target),
        "generic_mae": _mae(apply_generic_repair(pred, gc, obs_fact, graph), target),
        "oracle_mae":  _mae(apply_nonlinear_repair(pred, NL_COEFF, obs_fact, graph, NL_TAU), target),
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def _summary(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(x.mean()) if n else 0.0
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    z = mean / se if se else (float("inf") if mean > 0 else float("-inf"))
    p = float(math.erfc(abs(z) / math.sqrt(2))) if math.isfinite(z) else 0.0
    p_str = "< 1e-300" if p < 1e-300 else p
    return {"n": n, "mean": mean,
            "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
            "z": z, "p_two_sided": p_str}


def _aggregate_blind(rows: list[dict]) -> dict:
    out = {}
    for fam in MECHANISMS:
        r = [x for x in rows if x["family"] == fam]
        if not r:
            continue
        n = len(r)
        n_ok  = sum(1 for x in r if x.get("is_correct_id"))
        n_wr  = sum(1 for x in r if x.get("recovered") and not x.get("is_correct_id"))
        n_inc = sum(1 for x in r if x.get("recovered") is None)
        cond  = [x for x in r if x.get("repaired") and x.get("is_correct_id")]
        id_block = {
            "n_total": n, "n_correct": n_ok, "n_wrong": n_wr, "n_inconclusive": n_inc,
            "accuracy": n_ok / n, "false_recovery_rate": n_wr / n, "abstention_rate": n_inc / n,
        }
        e2e_block = {
            "orig_mae": float(np.mean([x["orig_mae"] for x in r])),
            "e2e_mae":  float(np.mean([x["e2e_mae"]  for x in r])),
            "n_repaired": sum(1 for x in r if x.get("repaired")),
            "n_abstained": n_inc,
            "improvement": _summary([x["orig_mae"] - x["e2e_mae"] for x in r]),
        }
        cond_block: dict = {}
        if cond:
            keys = ["orig_mae", "correct_mae", "wrong_mae", "generic_mae", "oracle_mae"]
            cond_block = {
                "n": len(cond),
                **{k: float(np.mean([x[k] for x in cond])) for k in keys if all(k in x for x in cond)},
                "oracle_gap":            float(np.mean([x["correct_mae"] - x["oracle_mae"]  for x in cond])),
                "correct_minus_wrong":   _summary([x["correct_mae"] - x["wrong_mae"]   for x in cond]),
                "correct_minus_generic": _summary([x["correct_mae"] - x["generic_mae"] for x in cond]),
            }
            if all("residual_mae" in x and "random_mae" in x for x in cond):
                cond_block["correct_minus_residual"] = _summary(
                    [x["correct_mae"] - x["residual_mae"] for x in cond])
                cond_block["correct_minus_random"] = _summary(
                    [x["correct_mae"] - x["random_mae"] for x in cond])
        out[fam] = {"identification": id_block, "e2e": e2e_block, "conditional": cond_block}
    return out


# ---------------------------------------------------------------------------
# Coverage-reliability tradeoff (W3)
# ---------------------------------------------------------------------------
def coverage_reliability(disc_records: list[dict], frozen: dict, n_blind: int = 100) -> dict:
    results = {}
    for K in (1, 2, 3):
        rows = []
        for family in MECHANISMS:
            for seed in BLIND_SEEDS[:n_blind]:
                twin, scenario = base_world(family, seed)
                recovered = _run_falsifier(twin, scenario, max_experiments=K)
                obs = twin.observe(scenario)
                graph = AviationGraph.build(obs.flights)
                pred = _prediction(twin, scenario).mean
                target = obs.delay_vector(graph.flight_ids)
                orig_mae = _mae(pred, target)
                repaired = recovered is not None and recovered in frozen
                e2e_mae = (_mae(apply_repair(pred, recovered,
                                             float(frozen[recovered]["coefficient"]),
                                             obs, graph), target)
                           if repaired else orig_mae)
                rows.append({"family": family, "seed": seed, "recovered": recovered,
                             "is_correct_id": recovered == family, "orig_mae": orig_mae,
                             "e2e_mae": e2e_mae, "repaired": repaired})
        fam_summary = {}
        for fam in MECHANISMS:
            fr = [x for x in rows if x["family"] == fam]
            n = len(fr)
            n_rep = sum(1 for x in fr if x["repaired"])
            n_ok  = sum(1 for x in fr if x["is_correct_id"])
            n_wr  = sum(1 for x in fr if x["recovered"] and not x["is_correct_id"])
            cond_mae = (float(np.mean([x["e2e_mae"] for x in fr
                                       if x["is_correct_id"] and x["repaired"]]))
                        if n_ok else None)
            fam_summary[fam] = {
                "coverage": n_rep / n if n else 0.0,
                "accuracy_among_repaired": n_ok / n_rep if n_rep else 0.0,
                "false_recovery_rate": n_wr / n if n else 0.0,
                "conditional_mae": cond_mae,
                "e2e_improvement_mean": float(np.mean([x["orig_mae"] - x["e2e_mae"] for x in fr])),
            }
        results[f"K={K}"] = fam_summary
    return results


# ---------------------------------------------------------------------------
# Ablation study (W3/W5)
# ---------------------------------------------------------------------------
def ablation_study(frozen: dict, n: int = 60) -> dict:
    strategies = ["PASSIVE", "RANDOM", "ACTIVE"]
    rows = []
    for family in MECHANISMS:
        for seed in BLIND_SEEDS[:n]:
            twin, scenario = base_world(family, seed)
            obs = twin.observe(scenario)
            graph = AviationGraph.build(obs.flights)
            pred = _prediction(twin, scenario).mean
            target = obs.delay_vector(graph.flight_ids)
            orig_mae = _mae(pred, target)
            row = {"family": family, "seed": seed, "orig_mae": orig_mae}
            for strategy in strategies:
                rec = _run_falsifier_ablation(twin, scenario, strategy, rng_seed=seed)
                if rec and rec in frozen:
                    r_pred = apply_repair(pred, rec, float(frozen[rec]["coefficient"]), obs, graph)
                else:
                    r_pred = pred
                row[f"{strategy}_recovered"] = rec
                row[f"{strategy}_mae"] = _mae(r_pred, target)
            rows.append(row)
    out = {}
    for fam in MECHANISMS:
        fr = [x for x in rows if x["family"] == fam]
        if not fr:
            continue
        out[fam] = {}
        for strategy in strategies:
            key = f"{strategy}_mae"
            n_rep = sum(1 for x in fr if x.get(f"{strategy}_recovered"))
            out[fam][strategy] = {
                "mean_mae": float(np.mean([x[key] for x in fr])),
                "coverage": n_rep / len(fr),
            }
    return out


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def run_final_robustness_evaluation(output_dir: str = "data/processed") -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    docs = Path("docs")
    docs.mkdir(exist_ok=True)

    # --- Development discovery (base topology + coefficients) ---
    print("[1/7] Building base discovery...")
    disc = build_discovery(DISC_SEEDS, base_world)
    correct_disc = [r for r in disc if r["mechanism"] == r["true_family"]]
    frozen = estimate_from_discovery(correct_disc)
    frozen_snap = json.dumps(frozen, sort_keys=True)

    # --- Base blind evaluation ---
    print("[2/7] Base blind evaluation...")
    base_blind = []
    for family in MECHANISMS:
        for seed in BLIND_SEEDS:
            base_blind.append(_eval_case(family, seed, frozen, base_world))
    assert frozen_snap == json.dumps(frozen, sort_keys=True), "Mutation during base blind!"

    # --- Held-out world blind evaluation (W1) ---
    print("[3/7] Held-out world evaluation (W1)...")
    heldout_blind = []
    for family in MECHANISMS:
        for seed in BLIND_SEEDS[:100]:
            heldout_blind.append(_eval_case(family, seed, frozen, heldout_world))
    assert frozen_snap == json.dumps(frozen, sort_keys=True), "Mutation during heldout!"

    # --- Coverage-reliability tradeoff (W3) ---
    print("[4/7] Coverage-reliability tradeoff (W3)...")
    cov_rel = coverage_reliability(correct_disc, frozen)

    # --- Nonlinear experiment (W4) ---
    print("[5/7] Nonlinear experiment (W4)...")
    nl_disc = build_nl_discovery(NL_DISC_SEEDS)
    nl_frozen = estimate_from_discovery(nl_disc)
    nl_frozen_snap = json.dumps(nl_frozen, sort_keys=True)
    nl_blind = [_eval_nl_case(seed, nl_frozen) for seed in NL_BLIND_SEEDS]
    assert nl_frozen_snap == json.dumps(nl_frozen, sort_keys=True), "NL mutation!"

    # --- Ablation study (W5) ---
    print("[6/7] Ablation study (W5)...")
    ablations = ablation_study(frozen)

    # --- Aggregate ---
    print("[7/7] Aggregating results...")
    base_agg    = _aggregate_blind(base_blind)
    heldout_agg = _aggregate_blind(heldout_blind)

    disc_counts = {}
    for fam in MECHANISMS:
        all_r  = [r for r in disc if r["true_family"] == fam]
        corr   = [r for r in all_r if r["mechanism"] == fam]
        disc_counts[fam] = {
            "seeds": len(DISC_SEEDS), "identified": len(all_r),
            "correct": len(corr), "wrong": len(all_r) - len(corr),
            "inconclusive": len(DISC_SEEDS) - len(all_r),
            "used_for_estimation": len(corr),
        }

    nl_agg = {
        "disc_n": len(nl_disc),
        "nl_beta_hat": nl_frozen.get("NONLINEAR", {}).get("coefficient", 0.0),
        "nl_ols_se":   nl_frozen.get("NONLINEAR", {}).get("ols_se", 0.0),
        "linear_beta_hat": nl_frozen.get("AIRCRAFT_ROTATION", {}).get("coefficient", 0.0),
        "true_nl_coeff": NL_COEFF, "tau": NL_TAU,
        "orig_mae":    float(np.mean([r["orig_mae"]    for r in nl_blind])),
        "linear_mae":  float(np.mean([r["linear_mae"]  for r in nl_blind])),
        "nl_mae":      float(np.mean([r["nl_mae"]      for r in nl_blind])),
        "generic_mae": float(np.mean([r["generic_mae"] for r in nl_blind])),
        "oracle_mae":  float(np.mean([r["oracle_mae"]  for r in nl_blind])),
        "nl_minus_linear":  _summary([r["nl_mae"] - r["linear_mae"]  for r in nl_blind]),
        "nl_minus_generic": _summary([r["nl_mae"] - r["generic_mae"] for r in nl_blind]),
        "nl_minus_oracle":  _summary([r["nl_mae"] - r["oracle_mae"]  for r in nl_blind]),
    }

    output = {
        "metadata": {
            "disc_seeds": [DISC_SEEDS[0], DISC_SEEDS[-1]],
            "blind_seeds": [BLIND_SEEDS[0], BLIND_SEEDS[-1]],
            "nl_disc_seeds": [NL_DISC_SEEDS[0], NL_DISC_SEEDS[-1]],
            "nl_blind_seeds": [NL_BLIND_SEEDS[0], NL_BLIND_SEEDS[-1]],
            "noise_sd": INTERVENTION_NOISE_SD,
            "nl_tau": NL_TAU, "nl_coeff": NL_COEFF,
            "frozen_before_blind": True,
        },
        "discovery_counts": disc_counts,
        "estimation": {m: {"beta_hat": frozen[m]["coefficient"],
                            "ols_se":   frozen[m]["ols_se"],
                            "n":        frozen[m]["sample_count"]} for m in frozen},
        "base_blind": base_agg,
        "heldout_blind": heldout_agg,
        "coverage_reliability": cov_rel,
        "nonlinear": nl_agg,
        "ablations": ablations,
    }

    (out_path / "final_robustness_results.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")
    _write_report(output, docs)
    return output


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def _write_report(D: dict, docs: Path) -> None:
    lines = [
        "# FALSIFIER-X AIR: Final Robustness and External Validation",
        "",
        "## 1. Discovery Counts (Base Configuration)",
    ]
    for fam, dc in D["discovery_counts"].items():
        lines.append(
            f"- {fam}: {dc['seeds']} seeds | {dc['identified']} identified | "
            f"{dc['correct']} correct | {dc['wrong']} wrong | "
            f"{dc['inconclusive']} inconclusive | {dc['used_for_estimation']} for estimation"
        )

    lines += ["", "## 2. Parameter Estimation"]
    for m, e in D["estimation"].items():
        lines.append(f"- {m}: beta_hat={e['beta_hat']:.4f}, OLS SE={e['ols_se']:.4f}, n={e['n']}")

    def _section(title: str, agg: dict, section_num: int) -> list[str]:
        s = ["", f"## {section_num}. {title}"]
        for fam, data in agg.items():
            im = data["identification"]
            e2 = data["e2e"]
            s.append(f"### {fam}")
            s.append(
                f"  ID: {im['n_correct']}/{im['n_total']} correct ({im['accuracy']:.1%}) | "
                f"{im['n_wrong']} wrong ({im['false_recovery_rate']:.1%}) | "
                f"{im['n_inconclusive']} inconclusive ({im['abstention_rate']:.1%})")
            imp = e2["improvement"]
            s.append(
                f"  E2E: orig={e2['orig_mae']:.4f} -> e2e={e2['e2e_mae']:.4f} | "
                f"D={imp['mean']:.4f} [{imp['ci95_low']:.4f},{imp['ci95_high']:.4f}] "
                f"p={imp['p_two_sided']} | repaired={e2['n_repaired']} abstained={e2['n_abstained']}")
            c = data.get("conditional", {})
            if c:
                s.append(
                    f"  Cond (N={c['n']}): orig={c.get('orig_mae',0):.3f} "
                    f"correct={c.get('correct_mae',0):.3f} wrong={c.get('wrong_mae',0):.3f} "
                    f"generic={c.get('generic_mae',0):.3f} oracle={c.get('oracle_mae',0):.3f} "
                    f"oracle_gap={c.get('oracle_gap',0):.3f}")
                for label, key in [("correct-wrong", "correct_minus_wrong"),
                                    ("correct-generic", "correct_minus_generic"),
                                    ("correct-residual", "correct_minus_residual"),
                                    ("correct-random", "correct_minus_random")]:
                    v = c.get(key)
                    if v:
                        s.append(f"  D {label}: {v['mean']:.4f} [{v['ci95_low']:.4f},{v['ci95_high']:.4f}] p={v['p_two_sided']}")
        return s

    lines += _section("Base Blind Evaluation (Development Config)", D["base_blind"], 3)
    lines += _section("Held-Out Causal World Evaluation (W1)", D["heldout_blind"], 4)

    lines += ["", "## 5. Coverage-Reliability Tradeoff (W3)"]
    for K_key, fam_data in D["coverage_reliability"].items():
        lines.append(f"### {K_key}")
        for fam, fd in fam_data.items():
            lines.append(
                f"  {fam}: coverage={fd['coverage']:.2f} "
                f"acc_repaired={fd['accuracy_among_repaired']:.2f} "
                f"false_rec={fd['false_recovery_rate']:.2f} "
                f"e2e_improvement={fd['e2e_improvement_mean']:.4f}")

    nl = D["nonlinear"]
    lines += [
        "", "## 6. Nonlinear Repair Experiment (W4)",
        f"  Pre-declared threshold tau={nl['tau']}, true NL coefficient={nl['true_nl_coeff']}",
        f"  Discovery records: {nl['disc_n']}",
        f"  NL beta_hat={nl['nl_beta_hat']:.4f} (OLS SE={nl['nl_ols_se']:.4f}) | "
        f"Linear beta_hat={nl['linear_beta_hat']:.4f}",
        f"  MAE: orig={nl['orig_mae']:.4f} linear={nl['linear_mae']:.4f} "
        f"NL={nl['nl_mae']:.4f} generic={nl['generic_mae']:.4f} oracle={nl['oracle_mae']:.4f}",
        f"  D NL-linear:  {nl['nl_minus_linear']['mean']:.4f} "
        f"[{nl['nl_minus_linear']['ci95_low']:.4f},{nl['nl_minus_linear']['ci95_high']:.4f}] "
        f"p={nl['nl_minus_linear']['p_two_sided']}",
        f"  D NL-generic: {nl['nl_minus_generic']['mean']:.4f} "
        f"[{nl['nl_minus_generic']['ci95_low']:.4f},{nl['nl_minus_generic']['ci95_high']:.4f}] "
        f"p={nl['nl_minus_generic']['p_two_sided']}",
        f"  D NL-oracle:  {nl['nl_minus_oracle']['mean']:.4f} "
        f"(oracle gap = cost of estimating NL coefficient)",
    ]

    lines += ["", "## 7. Ablation Study (W3/W5)"]
    for fam, strats in D.get("ablations", {}).items():
        lines.append(f"### {fam}")
        for strategy, vals in strats.items():
            lines.append(
                f"  {strategy}: mean_MAE={vals['mean_mae']:.4f} coverage={vals['coverage']:.2f}")

    lines += [
        "", "## 8. Weakness Assessment",
        "W1 Synthetic dependence: Tested via held-out topology (12-flight, 4-leg chains).",
        "W2 Narrow mechanisms: Three mechanisms evaluated across base + held-out variants.",
        "W3 Low AR/RD coverage: Coverage-reliability tradeoff with K=1,2,3 reported.",
        "W4 Linear repair: Threshold nonlinear mechanism with tau=5.0 tested.",
        "W5 External baselines: Residual-shift and random-mechanism baselines compared.",
        "W6 Novelty: See docs/novelty_and_baseline_analysis.md.",
    ]

    (docs / "final_robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run_final_robustness_evaluation()
    print("Done.")
