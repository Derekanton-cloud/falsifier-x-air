"""Phase-8 evaluation harness: Generalized Mechanism Falsification & Identifiability Stress Test.

This module is the sole Phase-8 layer permitted to access _benchmark_truth().
All learner modules (falsifier, adequacy, experiments, mechanisms, recovery) see
only observations; the oracle is accessed only after decisions are made.

Partition isolation
-------------------
Phase-8 discovery seeds : 200–219  (debug/inspection only; thresholds NOT tuned here)
Phase-8 evaluation seeds : 2000–2019 (final benchmark)
Phase-7 seeds (100–109, 1000–1009, 101000–101009) are never touched.

Hypotheses tested
-----------------
H1  Larger graphs (8/16/32 flights) — detection degrades gradually, false discovery stays low
H2  ACTIVE selector advantage grows with hypothesis complexity
H3  CONFOUNDED worlds — FALSIFIER-X abstains (INCONCLUSIVE) when mechanisms indistinguishable
H4  Transient vs sustained mechanisms — temporal persistence filters transient correctly
H5  Multi-mechanism worlds (MULTI_2, MULTI_3) — abstention preferred over false recovery
H6  Strength gradient — weak mechanisms yield ADEQUATE, strong ones yield detection
H7  Budget constraints (K=1,2,3,5) — recovery-vs-budget curve; ACTIVE efficiency under pressure

Metrics
-------
detection_rate, false_discovery_rate, ood_rate
node/edge/mechanism precision/recall/F1
recovery_rate, false_recovery_rate
abstention_rate, abstention_correctness (H3/H5 specific)
experiments_to_recovery, mean_cost, recovery_per_cost
active_efficiency_ratio (ACTIVE success/cost ÷ RANDOM success/cost)
recovery_vs_budget (per K)
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .evaluation import SetMetrics, set_metrics  # reuse frozen set_metrics
from .experiments import (
    ActiveExperimentSelector,
    ExperimentConfig,
    run_experiment,
    select_baseline,
    update_evidence,
)
from .falsifier import FalsifierXAir, InvestigationContext
from .graph import AviationGraph
from .mechanisms import generate_candidates
from .phase8_worlds import (
    BUDGET_CAPS,
    P8_DISCOVERY_SEEDS,
    P8_EVALUATION_SEEDS,
    SCALE_SIZES,
    STRENGTH_LEVELS,
    IdentifiabilityCertificate,
    build_scale_flights,
    world_confounded,
    world_correct,
    world_multi,
    world_ood,
    world_scale,
    world_strength_gradient,
    world_transient,
    world_unseen_topo,
)
from .schema import PredictionOutput
from .twin import AviationDigitalTwin, TwinScenario

# Phase-7 seeds forbidden from inspection here
_P7_FORBIDDEN = frozenset(range(100, 110)) | frozenset(range(1000, 1010)) | frozenset(range(101000, 101010))


# ---------------------------------------------------------------------------
# Prediction (frozen — same formula as Phase-7; no recalibration)
# ---------------------------------------------------------------------------

def _prediction(twin: AviationDigitalTwin, scenario: TwinScenario) -> PredictionOutput:
    """Frozen weather/capacity prior, identical to Phase-7 _prediction."""
    base = 8.0 * scenario.weather + 12.0 * max(0.0, 1.0 - scenario.capacity)
    size = len(twin.flights)
    radius = max(3.0, scenario.noise_scale * 2.0)
    mean = np.full(size, base)
    return PredictionOutput(mean, mean - radius, mean + radius, np.full(size, radius / 1.96))


# ---------------------------------------------------------------------------
# Core case runner (single mechanism world, variable budget)
# ---------------------------------------------------------------------------

def _run_case_budgeted(
    twin: AviationDigitalTwin,
    scenario: TwinScenario,
    truth_mechanisms: frozenset,
    truth_nodes: frozenset,
    truth_edges: frozenset,
    strategy: str,
    budget: int,
    rng: np.random.Generator,
) -> dict:
    """Run FALSIFIER-X with a fixed experiment budget and a named selection strategy."""
    observation = twin.observe(scenario)
    graph = AviationGraph.build(observation.flights)
    prediction = _prediction(twin, scenario)
    observed = observation.delay_vector(graph.flight_ids)
    residuals = observed - prediction.mean
    ood_score = 5.0 if scenario.family in {"OOD_P8"} else 0.0

    from .adequacy import StructuralAdequacyDetector
    adequacy = StructuralAdequacyDetector().evaluate(
        observed, prediction, ood_score, 3,
        AviationGraph.residual_concentration(graph, residuals),
        prediction.mean,
        AviationGraph.directional_residual_lift(graph, residuals, prediction.epistemic_std),
    )
    if adequacy.state != "STRUCTURALLY_SUSPICIOUS":
        node = set_metrics([], truth_nodes)
        edge = set_metrics(set(), truth_edges)
        mech = set_metrics(set(), truth_mechanisms)
        return {
            "strategy": strategy, "budget": budget, "adequacy": adequacy.state,
            "recovered": None, "correct": False, "experiments": 0, "cost": 0.0,
            "false_recovery": False, "outcome": adequacy.state,
            "node_metrics": asdict(node), "edge_metrics": asdict(edge), "mechanism_metrics": asdict(mech),
        }

    affected = AviationGraph.localize(graph, residuals)
    candidates = generate_candidates(graph, affected)
    config = ExperimentConfig(maximum_experiments=budget)
    tried: set[str] = set()
    experiments = []
    for index in range(budget):
        plausible = tuple(item for item in candidates if item.status.name != "REJECTED")
        if not plausible:
            break
        intervention = select_baseline(strategy, plausible, graph, observation, prediction.mean, tried, rng)
        if intervention is None:
            break
        expected = ActiveExperimentSelector().predicted_effects(plausible, graph, observation, prediction.mean, intervention)
        result = run_experiment(twin, scenario, intervention, f"p8-{strategy}-{scenario.seed}-b{budget}-{index}")
        update_evidence(plausible, expected, result, config)
        tried.add(intervention)
        experiments.append(result)

    survivors = [item for item in candidates if item.status.name != "REJECTED"]
    recovered = None
    if len(survivors) == 1:
        s = survivors[0]
        if s.observations >= config.minimum_confirmation_experiments and s.log_evidence >= config.support_log_evidence:
            recovered = s.mechanism_id

    outcome = "RECOVERED" if recovered else "INCONCLUSIVE"
    correct = recovered in truth_mechanisms if recovered else False
    false_recovery = bool(recovered and not correct)

    node = set_metrics(affected, truth_nodes)
    edge = set_metrics(
        {(src, tgt) for src, tgt, d in graph.graph.edges(data=True)
         if d.get("relation") in {"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY"}
         and src in set(affected) and tgt in set(affected)},
        truth_edges,
    )
    mech = set_metrics({recovered} if recovered else set(), truth_mechanisms)

    return {
        "strategy": strategy, "budget": budget, "adequacy": adequacy.state,
        "recovered": recovered, "correct": correct,
        "experiments": len(experiments),
        "cost": float(sum(r.intervention.cost for r in experiments)),
        "false_recovery": false_recovery, "outcome": outcome,
        "node_metrics": asdict(node), "edge_metrics": asdict(edge), "mechanism_metrics": asdict(mech),
    }


# ---------------------------------------------------------------------------
# Per-hypothesis case builders
# ---------------------------------------------------------------------------

def _truth(twin: AviationDigitalTwin) -> tuple[frozenset, frozenset, frozenset]:
    t = twin._benchmark_truth()
    return t["mechanisms"], t["affected_nodes"], t["affected_edges"]


def run_h1_scale(seed: int, strategies: tuple[str, ...] = ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE")) -> list[dict]:
    """H1: detection across 8/16/32-flight networks for both chain mechanisms."""
    records = []
    rng = np.random.default_rng(seed)
    for n in SCALE_SIZES:
        for mech in ("AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY", "AIRPORT_CAPACITY"):
            twin, scenario = world_scale(n, mech, seed)
            tm, tn, te = _truth(twin)
            for strategy in strategies:
                r = _run_case_budgeted(twin, scenario, tm, tn, te, strategy, 3, np.random.default_rng(abs(seed ^ hash(strategy))))
                r.update({"hypothesis": "H1", "n_flights": n, "mechanism": mech, "seed": seed, "family": f"SCALE_{n}"})
                records.append(r)
    return records


def run_h2_selector_difficulty(seed: int, n_flights: int = 16) -> list[dict]:
    """H2: selector advantage as candidate set grows (MULTI_2 vs single mechanism)."""
    records = []
    for mechanisms, family in [
        ({"AIRCRAFT_ROTATION"}, "SINGLE"),
        ({"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY"}, "MULTI_2"),
        ({"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY", "AIRPORT_CAPACITY"}, "MULTI_3"),
    ]:
        twin, scenario = world_multi(mechanisms, seed, n_flights=n_flights)
        tm, tn, te = _truth(twin)
        for strategy in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE"):
            r = _run_case_budgeted(twin, scenario, tm, tn, te, strategy, 5, np.random.default_rng(abs(seed ^ hash(strategy))))
            r.update({"hypothesis": "H2", "n_flights": n_flights, "world_family": family, "seed": seed, "family": family})
            records.append(r)
    return records


def run_h3_confounded(seed: int) -> list[dict]:
    """H3: FALSIFIER-X abstains (INCONCLUSIVE) when mechanisms are observationally indistinguishable."""
    twin, scenario, cert = world_confounded(seed)
    tm, tn, te = _truth(twin)
    records = []
    for strategy in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE"):
        r = _run_case_budgeted(twin, scenario, tm, tn, te, strategy, 3, np.random.default_rng(abs(seed ^ hash(strategy))))
        r.update({
            "hypothesis": "H3", "seed": seed, "family": "CONFOUNDED",
            "cert_verdict": cert.verdict,
            "cert_delta_rotation": cert.delta_rotation,
            "cert_delta_resource": cert.delta_resource,
            "cert_rel_diff": cert.relative_difference,
            # abstention_correct: if cert says INDISTINGUISHABLE, INCONCLUSIVE is the right call
            "abstention_correct": (cert.verdict == "INDISTINGUISHABLE") == (r["outcome"] == "INCONCLUSIVE"),
        })
        records.append(r)
    return records


def run_h4_transient(seed: int, n_flights: int = 8) -> list[dict]:
    """H4: transient vs sustained mechanisms — temporal persistence gate."""
    records = []
    for transient_after, label in [(None, "SUSTAINED"), (n_flights // 2, "TRANSIENT")]:
        for mech in ("AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY"):
            if transient_after is None:
                twin, scenario = world_scale(n_flights, mech, seed)
            else:
                twin, scenario = world_transient(mech, seed, transient_after=transient_after, n_flights=n_flights)
            tm, tn, te = _truth(twin)
            r = _run_case_budgeted(twin, scenario, tm, tn, te, "ACTIVE", 3, np.random.default_rng(seed))
            r.update({"hypothesis": "H4", "seed": seed, "transient_label": label,
                       "transient_after": transient_after, "mechanism": mech, "family": f"H4_{label}"})
            records.append(r)
    return records


def run_h5_multi(seed: int, n_flights: int = 8) -> list[dict]:
    """H5: multi-mechanism recovery difficulty — expect INCONCLUSIVE over false recovery."""
    records = []
    for mechanisms, label in [
        ({"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY"}, "MULTI_2"),
        ({"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY", "AIRPORT_CAPACITY"}, "MULTI_3"),
    ]:
        twin, scenario = world_multi(mechanisms, seed, n_flights=n_flights)
        tm, tn, te = _truth(twin)
        for strategy in ("RANDOM", "ACTIVE"):
            r = _run_case_budgeted(twin, scenario, tm, tn, te, strategy, 3, np.random.default_rng(abs(seed ^ hash(strategy))))
            r.update({"hypothesis": "H5", "seed": seed, "family": label, "n_mechanisms": len(mechanisms)})
            records.append(r)
    return records


def run_h6_strength(seed: int) -> list[dict]:
    """H6: mechanism strength gradient — below-threshold should yield ADEQUATE."""
    records = []
    for strength in STRENGTH_LEVELS:
        twin, scenario = world_strength_gradient("AIRCRAFT_ROTATION", strength, seed)
        tm, tn, te = _truth(twin)
        r = _run_case_budgeted(twin, scenario, tm, tn, te, "ACTIVE", 3, np.random.default_rng(seed))
        r.update({"hypothesis": "H6", "seed": seed, "mechanism_strength": strength, "family": "STRENGTH_GRADIENT"})
        records.append(r)
    return records


def run_h7_budget(seed: int) -> list[dict]:
    """H7: intervention-budget constraints — recovery-vs-budget curve."""
    twin, scenario = world_scale(8, "AIRCRAFT_ROTATION", seed)
    tm, tn, te = _truth(twin)
    records = []
    for budget in BUDGET_CAPS:
        for strategy in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE"):
            r = _run_case_budgeted(twin, scenario, tm, tn, te, strategy, budget, np.random.default_rng(abs(seed ^ hash(strategy))))
            r.update({"hypothesis": "H7", "seed": seed, "family": "SCALE_8", "mechanism": "AIRCRAFT_ROTATION"})
            records.append(r)
    return records


def run_controls(seed: int) -> list[dict]:
    """Correct (no mechanism) and OOD controls — false discovery must stay near zero."""
    records = []
    for factory, family, ood_expected in [
        (world_correct, "CORRECT_P8", False),
        (world_ood, "OOD_P8", True),
    ]:
        twin, scenario = factory(seed)
        tm, tn, te = _truth(twin)
        r = _run_case_budgeted(twin, scenario, tm, tn, te, "ACTIVE", 3, np.random.default_rng(seed))
        r.update({"hypothesis": "CONTROL", "seed": seed, "family": family, "ood_expected": ood_expected})
        records.append(r)
    return records


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _mean(rows: list[dict], key: str) -> float:
    vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return float(np.mean(vals)) if vals else float("nan")


def _rate(rows: list[dict], key: str, value: object = True) -> float:
    return float(np.mean([row[key] == value for row in rows])) if rows else float("nan")


def _active_efficiency_ratio(all_rows: list[dict]) -> float:
    """ACTIVE success/cost ÷ RANDOM success/cost across all rows with a cost > 0."""
    def _spc(strategy: str) -> float:
        rows = [r for r in all_rows if r.get("strategy") == strategy]
        total_cost = sum(r["cost"] for r in rows)
        successes = sum(r.get("correct", False) for r in rows)
        return successes / max(total_cost, 1e-6)
    active = _spc("ACTIVE")
    random_ = _spc("RANDOM")
    return float(active / max(random_, 1e-6))


def _h3_abstention_correctness(h3_rows: list[dict]) -> float:
    """Fraction of H3 rows where abstention_correct == True."""
    return float(np.mean([r["abstention_correct"] for r in h3_rows])) if h3_rows else float("nan")


def _per_budget_recovery(h7_rows: list[dict]) -> dict:
    result = {}
    for budget in BUDGET_CAPS:
        for strategy in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE"):
            rows = [r for r in h7_rows if r["budget"] == budget and r["strategy"] == strategy]
            key = f"budget_{budget}_{strategy}"
            result[key] = {"recovery_rate": _rate(rows, "correct", True),
                           "mean_cost": _mean(rows, "cost"),
                           "n": len(rows)}
    return result


def _selector_summary(all_rows: list[dict]) -> dict:
    summary = {}
    for strategy in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE"):
        rows = [r for r in all_rows if r.get("strategy") == strategy]
        recovered = [r for r in rows if r.get("correct")]
        summary[strategy] = {
            "recovery_rate": _rate(rows, "correct", True),
            "false_recovery_rate": _rate(rows, "false_recovery", True),
            "abstention_rate": _rate(rows, "outcome", "INCONCLUSIVE"),
            "mean_cost": _mean(rows, "cost"),
            "mean_experiments": _mean(rows, "experiments"),
            "experiments_to_recovery": float(np.mean([r["experiments"] for r in recovered])) if recovered else float("nan"),
            "successful_recovery_per_cost": sum(r.get("correct", False) for r in rows) / max(sum(r["cost"] for r in rows), 1e-6),
            "n": len(rows),
        }
    return summary


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_phase8(
    output_dir: str | Path = "data/processed",
    seeds: Iterable[int] = P8_EVALUATION_SEEDS,
    label: str = "eval",
) -> dict:
    seeds = tuple(seeds)
    for s in seeds:
        if s in _P7_FORBIDDEN:
            raise ValueError(f"Phase-8 must not inspect Phase-7 seed {s}")

    all_records: list[dict] = []
    for seed in seeds:
        all_records.extend(run_h1_scale(seed))
        all_records.extend(run_h2_selector_difficulty(seed))
        all_records.extend(run_h3_confounded(seed))
        all_records.extend(run_h4_transient(seed))
        all_records.extend(run_h5_multi(seed))
        all_records.extend(run_h6_strength(seed))
        all_records.extend(run_h7_budget(seed))
        all_records.extend(run_controls(seed))

    h3_rows = [r for r in all_records if r.get("hypothesis") == "H3"]
    h7_rows = [r for r in all_records if r.get("hypothesis") == "H7"]
    h1_rows = [r for r in all_records if r.get("hypothesis") == "H1"]
    h6_rows = [r for r in all_records if r.get("hypothesis") == "H6"]

    # Per-hypothesis summaries
    h1_by_scale: dict = {}
    for n in SCALE_SIZES:
        rows = [r for r in h1_rows if r.get("n_flights") == n]
        h1_by_scale[f"SCALE_{n}"] = {
            "detection_rate": _rate(rows, "adequacy", "STRUCTURALLY_SUSPICIOUS"),
            "false_discovery_rate": _rate(rows, "false_recovery", True),
            "recovery_rate": _rate(rows, "correct", True),
            "n": len(rows),
        }

    h3_summary = {
        "abstention_correctness": _h3_abstention_correctness(h3_rows),
        "inconclusive_rate": _rate(h3_rows, "outcome", "INCONCLUSIVE"),
        "false_recovery_rate": _rate(h3_rows, "false_recovery", True),
        "cert_indistinguishable_fraction": _rate(h3_rows, "cert_verdict", "INDISTINGUISHABLE"),
        "n": len(h3_rows),
    }

    h6_summary: dict = {}
    for strength in STRENGTH_LEVELS:
        rows = [r for r in h6_rows if abs(r.get("mechanism_strength", -1) - strength) < 1e-9]
        h6_summary[f"strength_{strength}"] = {
            "detection_rate": _rate(rows, "adequacy", "STRUCTURALLY_SUSPICIOUS"),
            "recovery_rate": _rate(rows, "correct", True),
            "n": len(rows),
        }

    selector_summary = _selector_summary(all_records)
    active_ratio = _active_efficiency_ratio(all_records)
    budget_summary = _per_budget_recovery(h7_rows)

    output = {
        "metadata": {
            "label": label,
            "seeds": list(seeds),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "n_records": len(all_records),
            "indistinguishable_tolerance": 0.05,
            "ground_truth_access": "phase8_evaluation.py only",
            "thresholds_frozen_from": "Phase-7 discovery (not tuned on Phase-8)",
        },
        "h1_scale_summary": h1_by_scale,
        "h3_confounded_summary": h3_summary,
        "h6_strength_summary": h6_summary,
        "selector_summary": selector_summary,
        "active_efficiency_ratio": active_ratio,
        "h7_budget_summary": budget_summary,
        "records": all_records,
    }

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / f"phase8_{label}_results.json").write_text(
        json.dumps(output, indent=2, default=str), encoding="utf-8"
    )
    _write_report(output, target / f"phase8_{label}_report.md")
    return output


def _write_report(output: dict, path: Path) -> None:
    sel = output["selector_summary"]
    h1 = output["h1_scale_summary"]
    h3 = output["h3_confounded_summary"]
    h6 = output["h6_strength_summary"]
    h7 = output["h7_budget_summary"]
    ratio = output["active_efficiency_ratio"]

    lines = [
        "# Phase-8 Report — Generalized Mechanism Falsification & Identifiability Stress Test",
        "",
        f"Seeds: {output['metadata']['seeds']}",
        f"Records: {output['metadata']['n_records']}",
        f"Identifiability tolerance (frozen): {output['metadata']['indistinguishable_tolerance']}",
        "",
        "## H1 — Scale generalization",
        "",
        "| Scale | Detection | Recovery | False discovery | N |",
        "|-------|-----------|----------|----------------|---|",
    ]
    for key, v in h1.items():
        lines.append(f"| {key} | {v['detection_rate']:.2f} | {v['recovery_rate']:.2f} | {v['false_discovery_rate']:.2f} | {v['n']} |")

    lines += [
        "",
        "## H3 — Identifiability / abstention (CONFOUNDED worlds)",
        "",
        f"- Cert INDISTINGUISHABLE fraction: {h3['cert_indistinguishable_fraction']:.2f}",
        f"- INCONCLUSIVE rate: {h3['inconclusive_rate']:.2f}",
        f"- Abstention correctness: {h3['abstention_correctness']:.2f}",
        f"- False recovery rate: {h3['false_recovery_rate']:.2f}",
        f"- N: {h3['n']}",
        "",
        "## H6 — Mechanism strength gradient",
        "",
        "| Strength | Detection | Recovery | N |",
        "|----------|-----------|----------|---|",
    ]
    for key, v in sorted(h6.items()):
        lines.append(f"| {key} | {v['detection_rate']:.2f} | {v['recovery_rate']:.2f} | {v['n']} |")

    lines += [
        "",
        "## Selector comparison (all Phase-8 records)",
        "",
        "| Strategy | Recovery | False recovery | Abstention | Experiments/recovery | Cost | Success/cost |",
        "|----------|----------|----------------|------------|---------------------|------|-------------|",
    ]
    for strategy in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE"):
        v = sel[strategy]
        etr = f"{v['experiments_to_recovery']:.2f}" if not (isinstance(v["experiments_to_recovery"], float) and v["experiments_to_recovery"] != v["experiments_to_recovery"]) else "—"
        lines.append(f"| {strategy} | {v['recovery_rate']:.2f} | {v['false_recovery_rate']:.2f} | {v['abstention_rate']:.2f} | {etr} | {v['mean_cost']:.2f} | {v['successful_recovery_per_cost']:.3f} |")

    lines += [
        "",
        f"ACTIVE efficiency ratio (ACTIVE success/cost ÷ RANDOM success/cost): {ratio:.3f}",
        "",
        "## H7 — Recovery-vs-budget (AIRCRAFT_ROTATION, 8 flights)",
        "",
        "| Budget | RANDOM | MAX_EFFECT | ACTIVE | EXHAUSTIVE |",
        "|--------|--------|-----------|--------|-----------|",
    ]
    for budget in (1, 2, 3, 5):
        row_parts = [f"| K={budget}"]
        for strategy in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE"):
            v = h7.get(f"budget_{budget}_{strategy}", {})
            row_parts.append(f"{v.get('recovery_rate', float('nan')):.2f}")
        lines.append(" | ".join(row_parts) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys
    label = sys.argv[1] if len(sys.argv) > 1 else "eval"
    seeds = P8_DISCOVERY_SEEDS if label == "discovery" else P8_EVALUATION_SEEDS
    run_phase8(label=label, seeds=seeds)
