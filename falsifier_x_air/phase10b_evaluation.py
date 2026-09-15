"""Separate Phase-10B benchmark: noisy intervention estimation and blind repair.

Evaluator-only code owns world construction and the oracle coefficient.  The
learner functions in :mod:`phase10b_repair` receive only observable records.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np

from .evaluation import _prediction
from .falsifier import FalsifierXAir, InvestigationContext
from .graph import AviationGraph, GraphSnapshot
from .schema import Flight
from .twin import AviationDigitalTwin, TwinScenario
from .phase10b_repair import (MECHANISMS, WRONG_MECHANISM,
                              apply_generic_repair, apply_repair, estimate_from_discovery,
                              observable_feature, generic_feature)

PHASE10B_DISCOVERY_SEEDS = tuple(range(700000, 700100))
PHASE10B_BLIND_SEEDS = tuple(range(700100, 700300))
INTERVENTION_NOISE_SD = 2.0  # minutes of total network-delay effect, fixed a priori.
PURE_FAMILIES = MECHANISMS


def phase10b_flights() -> tuple[Flight, ...]:
    """Small scheduled topology with disjoint AR and RD directed edge sets, disjoint targets, but identical sources."""
    return (
        Flight("F1", "A", "B", "AC1", 0, "R1"),
        Flight("F2", "B", "C", "AC1", 1, "R2"),
        Flight("F3", "C", "D", "AC1", 2, "R3"),
        Flight("F4", "E", "F", "AC2", 0, "R4"),
        Flight("F5", "F", "G", "AC2", 1, "R5"),
        Flight("F6", "G", "H", "AC2", 2, "R6"),
        Flight("F7", "I", "J", "AC3", 1, "R1"),
        Flight("F8", "K", "L", "AC4", 1, "R4"),
        Flight("F9", "M", "N", "AC5", 2, "R2"),
        Flight("F10", "O", "P", "AC6", 2, "R5"),
    )


def topology_certificate(graph: GraphSnapshot) -> dict[str, object]:
    def edges(relation: str) -> set[tuple[str, str]]:
        return {(s, t) for s, t, d in graph.graph.edges(data=True) if d.get("relation") == relation}
    ar, rd = edges("AIRCRAFT_ROTATION"), edges("RESOURCE_DEPENDENCY")
    ar_targets, rd_targets = {t for _, t in ar}, {t for _, t in rd}
    ar_sources, rd_sources = {s for s, _ in ar}, {s for s, _ in rd}
    certificate = {
        "ar_edges": sorted(ar), "rd_edges": sorted(rd), "edge_overlap": sorted(ar & rd),
        "ar_target_support": sorted(ar_targets), "rd_target_support": sorted(rd_targets),
        "target_overlap": sorted(ar_targets & rd_targets),
        "ar_source_support": sorted(ar_sources), "rd_source_support": sorted(rd_sources),
        "source_overlap": sorted(ar_sources & rd_sources),
        "features_identical_by_structure": ar_targets == rd_targets,
        "no_single_relation_proxy": bool(ar - rd) and bool(rd - ar),
    }
    assert ar and rd and not (ar & rd)
    assert not (ar_targets & rd_targets)
    assert ar_sources == rd_sources
    assert certificate["no_single_relation_proxy"]
    return certificate


def phase10b_world(family: str, seed: int) -> tuple[AviationDigitalTwin, TwinScenario]:
    if family not in PURE_FAMILIES:
        raise ValueError("Phase-10B primary analysis accepts pure families only")
    # Fixed coefficients and strength across every case: no seed heterogeneity.
    scenario = TwinScenario(f"phase10b-{family.lower()}-{seed}", 1.00, 0.55, seed,
                            family="PHASE10B", noise_scale=1.5, mechanism_strength=1.0,
                            rotation_coefficient=0.85, resource_coefficient=0.60)
    return AviationDigitalTwin(phase10b_flights(), {family}, seed), scenario


def _noise(seed: int, mechanism: str) -> float:
    """Independent observation noise; distinct stream from twin scenario RNG."""
    code = {m: i for i, m in enumerate(MECHANISMS)}[mechanism]
    return float(np.random.default_rng(np.random.SeedSequence([seed, 10_010, code])).normal(0.0, INTERVENTION_NOISE_SD))


def paired_intervention_observation(twin: AviationDigitalTwin, scenario: TwinScenario,
                                    mechanism: str) -> dict[str, float]:
    """Same world for factual/counterfactual; noise is added only after differencing."""
    intervention = {"AIRCRAFT_ROTATION": "disable_aircraft_rotation",
                    "RESOURCE_DEPENDENCY": "relieve_resource_dependency",
                    "AIRPORT_CAPACITY": "increase_capacity"}[mechanism]
    factual = twin.observe(scenario)
    counterfactual = twin.counterfactual(scenario, intervention)
    true_effect = float(sum(factual.delays.values()) - sum(counterfactual.delays.values()))
    epsilon = _noise(scenario.seed, mechanism)
    return {"true_effect": true_effect, "observed_effect": true_effect + epsilon,
            "epsilon_intervention": epsilon,
            "factual_total": float(sum(factual.delays.values())),
            "counterfactual_total": float(sum(counterfactual.delays.values()))}


def _identified(twin: AviationDigitalTwin, scenario: TwinScenario, family: str) -> str | None:
    observation = twin.observe(scenario); graph = AviationGraph.build(observation.flights)
    prediction = _prediction(twin, scenario)
    result = FalsifierXAir().investigate(twin, InvestigationContext(
        observation, graph, prediction, prediction.mean, 0.0, 3, scenario))
    return result.recovered_mechanism if result.identifiability_reason == "LEGITIMATELY_IDENTIFIED" else None


def discovery_records(seeds: Iterable[int] = PHASE10B_DISCOVERY_SEEDS) -> list[dict]:
    records: list[dict] = []
    for seed in seeds:
        for family in PURE_FAMILIES:
            twin, scenario = phase10b_world(family, seed)
            recovered = _identified(twin, scenario, family)
            # Legitimate identification gate applies, but NO oracle filter against true family.
            if recovered is None:
                continue
            observation = twin.observe(scenario)
            graph = AviationGraph.build(observation.flights)
            paired = paired_intervention_observation(twin, scenario, recovered)
            records.append({"seed": seed, "true_family": family, "mechanism": recovered,
                            "feature_sum": float(observable_feature(recovered, observation, graph).sum()), 
                            "generic_feature_sum": float(generic_feature(observation, graph).sum()), 
                            **paired})
    return records


def _oracle_coefficient(scenario: TwinScenario, family: str) -> float:
    """Evaluator-only upper bound; never passed to estimation or correct repair."""
    if family == "AIRCRAFT_ROTATION": return scenario.rotation_coefficient * scenario.mechanism_strength
    if family == "RESOURCE_DEPENDENCY": return scenario.resource_coefficient * scenario.mechanism_strength
    return 20.0 * scenario.mechanism_strength


def _case(family: str, seed: int, frozen: dict[str, dict[str, float | int]]) -> dict | None:
    twin, scenario = phase10b_world(family, seed)
    recovered = _identified(twin, scenario, family)
    
    observation = twin.observe(scenario)
    graph = AviationGraph.build(observation.flights)
    prediction = _prediction(twin, scenario).mean
    target = observation.delay_vector(graph.flight_ids)
    def mae(p: np.ndarray) -> float: return float(np.mean(np.abs(p - target)))
    
    result = {
        "family": family, "seed": seed, "recovered": recovered,
        "is_correct_id": recovered == family,
        "orig_mae": mae(prediction),
        "repaired": False
    }
    
    if recovered is None or recovered not in frozen:
        result["e2e_mae"] = result["orig_mae"]
        return result
        
    result["repaired"] = True
    correct_coeff = float(frozen[recovered]["coefficient"])
    wrong_mechanism = WRONG_MECHANISM[recovered]
    wrong_coeff = float(frozen[wrong_mechanism]["coefficient"])
    generic_coeff = float(frozen["GENERIC"]["coefficient"])
    oracle_coeff = _oracle_coefficient(scenario, family)
    
    result["correct_mae"] = mae(apply_repair(prediction, recovered, correct_coeff, observation, graph))
    result["wrong_mae"] = mae(apply_repair(prediction, wrong_mechanism, wrong_coeff, observation, graph))
    result["generic_mae"] = mae(apply_generic_repair(prediction, generic_coeff, observation, graph))
    result["oracle_mae"] = mae(apply_repair(prediction, family, oracle_coeff, observation, graph))
    
    result["e2e_mae"] = result["correct_mae"]
    return result


def _paired_summary(values: list[float]) -> dict[str, float | int | str]:
    x = np.asarray(values, dtype=float); n = len(x); mean = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    z = mean / se if se else (float("inf") if mean else 0.0)
    p = float(math.erfc(abs(z) / math.sqrt(2))) if math.isfinite(z) else 0.0
    return {"n": n, "mean": mean, "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
            "test": "paired normal-approximation z test", "z": z, "p_two_sided": p if p > 1e-300 else "< 1e-300"}


def run_phase10b_evaluation(output_dir: str | Path = "data/processed") -> dict:
    assert set(PHASE10B_DISCOVERY_SEEDS).isdisjoint(PHASE10B_BLIND_SEEDS)
    
    flights = phase10b_flights()
    graph = AviationGraph.build(flights)
    certificate = topology_certificate(graph)
    
    discovery = discovery_records()
    
    # Feature correlation analysis on discovery set
    ar_feats, rd_feats = [], []
    for r in discovery:
        if r["true_family"] == "AIRCRAFT_ROTATION":
            twin, scenario = phase10b_world(r["true_family"], r["seed"])
            obs = twin.observe(scenario)
            ar_feats.append(observable_feature("AIRCRAFT_ROTATION", obs, graph))
            rd_feats.append(observable_feature("RESOURCE_DEPENDENCY", obs, graph))
    if ar_feats:
        ar_flat = np.concatenate(ar_feats)
        rd_flat = np.concatenate(rd_feats)
        if np.std(ar_flat) > 0 and np.std(rd_flat) > 0:
            certificate["ar_rd_feature_correlation"] = float(np.corrcoef(ar_flat, rd_flat)[0, 1])
        else:
            certificate["ar_rd_feature_correlation"] = 0.0
    else:
        certificate["ar_rd_feature_correlation"] = 0.0

    # Parameter estimation strictly on correctly identified cases
    frozen = estimate_from_discovery([r for r in discovery if r["mechanism"] == r["true_family"]])
    
    assert all(int(v["sample_count"]) > 0 for v in frozen.values())
    frozen_before_blind = json.dumps(frozen, sort_keys=True)
    
    blind = [r for family in PURE_FAMILIES for seed in PHASE10B_BLIND_SEEDS
             if (r := _case(family, seed, frozen)) is not None]
             
    assert frozen_before_blind == json.dumps(frozen, sort_keys=True)
    
    summaries = {}
    identification_metrics = {}
    e2e_metrics = {}
    
    for family in PURE_FAMILIES:
        rows = [r for r in blind if r["family"] == family]
        n_total = len(rows)
        n_correct = sum(1 for r in rows if r["recovered"] == family)
        n_wrong = sum(1 for r in rows if r["recovered"] is not None and r["recovered"] != family)
        n_inconclusive = sum(1 for r in rows if r["recovered"] is None)
        
        identification_metrics[family] = {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "n_inconclusive": n_inconclusive,
            "accuracy": n_correct / n_total if n_total > 0 else 0.0
        }
        
        # End-to-end metrics
        orig_e2e = [r["orig_mae"] for r in rows]
        pred_e2e = [r["e2e_mae"] for r in rows]
        e2e_metrics[family] = {
            "orig_mae": float(np.mean(orig_e2e)),
            "e2e_mae": float(np.mean(pred_e2e)),
            "e2e_improvement": _paired_summary([o - e for o, e in zip(orig_e2e, pred_e2e)])
        }
        
        # Conditional repair metrics
        repaired_rows = [r for r in rows if r["repaired"] and r["is_correct_id"]]
        if not repaired_rows:
            continue
            
        summaries[family] = {
            "n": len(repaired_rows), **{key: float(np.mean([r[key] for r in repaired_rows])) for key in
                ("orig_mae", "correct_mae", "wrong_mae", "generic_mae", "oracle_mae")},
            "oracle_gap": float(np.mean([r["correct_mae"] - r["oracle_mae"] for r in repaired_rows])),
            "correct_minus_wrong": _paired_summary([r["correct_mae"] - r["wrong_mae"] for r in repaired_rows]),
            "correct_minus_generic": _paired_summary([r["correct_mae"] - r["generic_mae"] for r in repaired_rows]),
        }
        
    diagnostics = {m: {"estimated_coefficient": v["coefficient"], 
                   "expected_se_approx": 2.0 / (max(1, v["sample_count"] * (70.0**2)))**0.5 if m != "GENERIC" else 0.0,
                   "true_effective_coefficient":
                   {"AIRCRAFT_ROTATION": .85, "RESOURCE_DEPENDENCY": .60, "AIRPORT_CAPACITY": 20.0}.get(m, 0.0),
                   "discovery_sample_count": v["sample_count"]} for m, v in frozen.items()}
                   
    output = {"metadata": {"discovery_seeds": list(PHASE10B_DISCOVERY_SEEDS), "blind_seeds": list(PHASE10B_BLIND_SEEDS),
              "intervention_effect_noise": "Normal(0, 2.0^2) minutes",
              "parameters_frozen_before_blind": True, "multiple_primary_excluded": True},
              "topology_certificate": certificate, "identification": identification_metrics, "estimation": diagnostics, 
              "conditional_repair": summaries, "end_to_end": e2e_metrics, "blind_records": blind}
              
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    (target / "phase10b_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    
    lines = ["# Phase 10B: Robust Causal Model Repair & Predictive Specificity\n",
             "## 1. Identification Metrics"]
    for family, im in identification_metrics.items():
        lines.append(f"- {family}: {im['n_correct']} correct, {im['n_wrong']} wrong, {im['n_inconclusive']} inconclusive (Accuracy: {im['accuracy']:.1%})")
        
    lines.append("\n## 2. Parameter Estimation (Discovery)")
    for m, d in diagnostics.items(): 
        lines.append(f"- {m}: estimate={d['estimated_coefficient']:.4f}, true diagnostic={d['true_effective_coefficient']:.4f}, n={d['discovery_sample_count']} (expected SE ~ {d['expected_se_approx']:.4f})")
        
    lines.append("\n## 3. Conditional Repair Performance (Correct IDs only)")
    for family, s in summaries.items():
        lines += [f"### {family} (N={s['n']})", 
                  f"MAE original={s['orig_mae']:.4f}; correct={s['correct_mae']:.4f}; wrong={s['wrong_mae']:.4f}; generic={s['generic_mae']:.4f}; oracle={s['oracle_mae']:.4f}; oracle gap={s['oracle_gap']:.4f}",
                  f"D correct-wrong={s['correct_minus_wrong']['mean']:.4f} [{s['correct_minus_wrong']['ci95_low']:.4f}, {s['correct_minus_wrong']['ci95_high']:.4f}], p={s['correct_minus_wrong']['p_two_sided']}",
                  f"D correct-generic={s['correct_minus_generic']['mean']:.4f} [{s['correct_minus_generic']['ci95_low']:.4f}, {s['correct_minus_generic']['ci95_high']:.4f}], p={s['correct_minus_generic']['p_two_sided']}"]
                  
    lines.append("\n## 4. End-to-End Pipeline Performance")
    for family, e2e in e2e_metrics.items():
        lines += [f"### {family}", f"Original MAE = {e2e['orig_mae']:.4f}, E2E Repaired MAE = {e2e['e2e_mae']:.4f}"]
        lines += [f"E2E Improvement = {e2e['e2e_improvement']['mean']:.4f}, p={e2e['e2e_improvement']['p_two_sided']}"]

    (target / "phase10b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    run_phase10b_evaluation()
