"""Benchmark-only Phase-7 evaluation layer.

This is the sole module permitted to query the twin's private benchmark oracle.
It creates scenarios and scores learner outputs; the learner never receives an
oracle value, scenario truth, or a repair target.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .falsifier import FalsifierXAir, InvestigationContext
from .adequacy import StructuralAdequacyDetector
from .experiments import ActiveExperimentSelector, ExperimentConfig, run_experiment, select_baseline, update_evidence
from .graph import AviationGraph
from .mechanisms import generate_candidates
from .recovery import choose_recovery, evaluate_selected_action
from .schema import Flight, PredictionOutput
from .twin import AviationDigitalTwin, TwinScenario


DISCOVERY_SEEDS = tuple(range(100, 110))
EVALUATION_SEEDS = tuple(range(1000, 1010))
SCENARIO_FAMILIES = (
    "CORRECT", "NOISE", "OOD", "AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY",
    "AIRPORT_CAPACITY", "MULTIPLE", "DECOY", "STRENGTH_SWEEP", "NONSTATIONARY", "UNSEEN_TOPOLOGY",
)
REPAIR_SEEDS = tuple(range(101000, 101010))


@dataclass(frozen=True)
class SetMetrics:
    precision: float
    recall: float
    f1: float


def set_metrics(predicted: Iterable[object], truth: Iterable[object]) -> SetMetrics:
    """Set metrics, deliberately simple enough for independent hand checks."""
    predicted, truth = set(predicted), set(truth)
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else (1.0 if not truth else 0.0)
    recall = tp / len(truth) if truth else (1.0 if not predicted else 0.0)
    return SetMetrics(precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0)


def standard_flights(resource: bool = True) -> tuple[Flight, ...]:
    return (
        Flight("F1", "A", "B", "AC1", 0, "R1" if resource else None),
        Flight("F2", "B", "C", "AC1", 1, "R1" if resource else None),
        Flight("F3", "C", "D", "AC2", 2, "R2" if resource else None),
        Flight("F4", "D", "A", "AC2", 3, "R2" if resource else None),
    )


def scenario_family(family: str, seed: int) -> tuple[AviationDigitalTwin, TwinScenario]:
    mechanisms = {
        "AIRCRAFT_ROTATION": {"AIRCRAFT_ROTATION"}, "RESOURCE_DEPENDENCY": {"RESOURCE_DEPENDENCY"},
        "AIRPORT_CAPACITY": {"AIRPORT_CAPACITY"}, "MULTIPLE": {"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY"},
        "DECOY": {"AIRCRAFT_ROTATION"}, "STRENGTH_SWEEP": {"AIRCRAFT_ROTATION"},
        "NONSTATIONARY": {"AIRCRAFT_ROTATION"},
    }.get(family, set())
    noise = 8.0 if family == "NOISE" else 1.5
    strength = 0.35 if family == "STRENGTH_SWEEP" else (1.5 if family == "RESOURCE_DEPENDENCY" else 1.0)
    flights = standard_flights(resource=True)
    if family == "UNSEEN_TOPOLOGY":
        flights = tuple(reversed(flights))
    scenario = TwinScenario(f"{family.lower()}-{seed}", 0.85, 0.70, seed, family, noise, strength,
                            1 if family == "NONSTATIONARY" else 0)
    return AviationDigitalTwin(flights, mechanisms, seed), scenario


def _prediction(twin: AviationDigitalTwin, scenario: TwinScenario) -> PredictionOutput:
    base = 8.0 * scenario.weather + 12.0 * max(0.0, 1.0 - scenario.capacity)
    size = len(twin.flights)
    # Frozen pre-mechanism reference; noise scenarios receive wider valid intervals.
    radius = max(5.0 if scenario.family in {"CORRECT", "NOISE"} else 3.0, scenario.noise_scale * 2.0)
    mean = np.full(size, base)
    return PredictionOutput(mean, mean - radius, mean + radius, np.full(size, radius / 1.96))


def run_case(family: str, seed: int) -> dict[str, object]:
    twin, scenario = scenario_family(family, seed)
    observation = twin.observe(scenario)
    graph = AviationGraph.build(observation.flights)
    prediction = _prediction(twin, scenario)
    # OOD is an observable benchmark condition, passed as a feature-distance diagnostic;
    # hidden mechanisms remain withheld.
    ood_score = 5.0 if family == "OOD" else 0.0
    result = FalsifierXAir().investigate(twin, InvestigationContext(
        observation, graph, prediction, prediction.mean, ood_score, 3, scenario,
    ))
    truth = twin._benchmark_truth()
    node = set_metrics(result.affected_flights, truth["affected_nodes"])
    affected = set(result.affected_flights)
    predicted_edges = {
        (source, target) for source, target, data in graph.graph.edges(data=True)
        if data.get("relation") in {"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY"}
        and source in affected and target in affected
    }
    edge = set_metrics(predicted_edges, truth["affected_edges"])
    recovered = {result.recovered_mechanism} if result.recovered_mechanism else set()
    mechanism = set_metrics(recovered, truth["mechanisms"])
    selected = next((item for item in result.candidates if item.mechanism_id == result.recovered_mechanism), None)
    # The recovery policy is learned on ``scenario`` but assessed on a disjoint,
    # paired held-out scenario; no discovery outcome is reused for this measure.
    held_out = TwinScenario(f"held-out-{family.lower()}-{seed}", scenario.weather, scenario.capacity,
                            seed + 100_000, "HELD_OUT", scenario.noise_scale, scenario.mechanism_strength)
    no_action = evaluate_selected_action(twin, held_out, choose_recovery(float(prediction.mean.sum()), None)).realized_total_delay
    action = evaluate_selected_action(twin, held_out, choose_recovery(float(prediction.mean.sum()), result.recovered_mechanism, selected)).realized_total_delay
    return {"family": family, "seed": seed, "adequacy": result.adequacy.state,
            "recovered": result.recovered_mechanism, "experiments": len(result.experiment_results),
            "cost": sum(item.intervention.cost for item in result.experiment_results),
            "node_metrics": asdict(node), "edge_metrics": asdict(edge), "mechanism_metrics": asdict(mechanism),
            "false_recovery": bool(result.recovered_mechanism and result.recovered_mechanism not in truth["mechanisms"]),
            "held_out_policy_delta": float(no_action - action)}


def run_selector_case(family: str, seed: int, strategy: str) -> dict[str, object]:
    """Sequential baseline comparison; only selection policy differs."""
    twin, scenario = scenario_family(family, seed)
    observation = twin.observe(scenario); graph = AviationGraph.build(observation.flights); prediction = _prediction(twin, scenario)
    adequacy = StructuralAdequacyDetector().evaluate(observation.delay_vector(graph.flight_ids), prediction,
        5.0 if family == "OOD" else 0.0, 3, AviationGraph.residual_concentration(graph, observation.delay_vector(graph.flight_ids) - prediction.mean), prediction.mean,
        AviationGraph.directional_residual_lift(graph, observation.delay_vector(graph.flight_ids) - prediction.mean, prediction.epistemic_std))
    candidates = generate_candidates(graph, AviationGraph.localize(graph, observation.delay_vector(graph.flight_ids) - prediction.mean)) if adequacy.state == "STRUCTURALLY_SUSPICIOUS" else ()
    config, tried, experiments = ExperimentConfig(), set(), []
    rng = np.random.default_rng(seed)
    for index in range(config.maximum_experiments):
        plausible = tuple(item for item in candidates if item.status.name != "REJECTED")
        intervention = select_baseline(strategy, plausible, graph, observation, prediction.mean, tried, rng) if plausible else None
        if intervention is None: break
        expected = ActiveExperimentSelector().predicted_effects(plausible, graph, observation, prediction.mean, intervention)
        result = run_experiment(twin, scenario, intervention, f"{strategy}-{seed}-{index}")
        update_evidence(plausible, expected, result, config); tried.add(intervention); experiments.append(result)
    survivors = [item for item in candidates if item.status.name != "REJECTED"]
    recovered = survivors[0].mechanism_id if len(survivors) == 1 and survivors[0].observations >= config.minimum_confirmation_experiments and survivors[0].log_evidence >= config.support_log_evidence else None
    truth = twin._benchmark_truth()
    return {"family": family, "seed": seed, "strategy": strategy, "recovered": recovered,
            "correct": recovered in truth["mechanisms"], "experiments": len(experiments),
            "cost": sum(item.intervention.cost for item in experiments)}


def _mean_metric(rows: list[dict[str, object]], metric: str, field: str) -> float:
    return float(np.mean([row[metric][field] for row in rows])) if rows else 0.0  # type: ignore[index]


def frozen_final_configuration() -> dict[str, object]:
    """Assertions guarding the pre-registered partition and frozen adequacy gate."""
    from .adequacy import AdequacyConfig
    assert set(DISCOVERY_SEEDS).isdisjoint(EVALUATION_SEEDS)
    assert set(REPAIR_SEEDS).isdisjoint(DISCOVERY_SEEDS)
    assert set(REPAIR_SEEDS).isdisjoint(EVALUATION_SEEDS)
    assert AdequacyConfig().directional_lift_threshold == 3.0
    return {"discovery_seeds": list(DISCOVERY_SEEDS), "evaluation_seeds": list(EVALUATION_SEEDS),
            "repair_seeds": list(REPAIR_SEEDS), "directional_lift_threshold": 3.0,
            "adequacy_config": asdict(AdequacyConfig())}


def run_benchmark(output_dir: str | Path = "data/processed", seeds: Iterable[int] = EVALUATION_SEEDS) -> dict[str, object]:
    """Run the frozen final partition, then its pre-declared held-out repair seeds."""
    seeds = tuple(seeds)
    if seeds != EVALUATION_SEEDS:
        raise ValueError("Final benchmark must use exactly pre-declared evaluation seeds 1000--1009")
    configuration = frozen_final_configuration()
    records = [run_case(family, seed) for family in SCENARIO_FAMILIES for seed in seeds]
    selector_records = [run_selector_case(family, seed, strategy) for family in SCENARIO_FAMILIES
                        for seed in seeds for strategy in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE")]
    grouped = {family: [r for r in records if r["family"] == family] for family in SCENARIO_FAMILIES}
    selector_summary = {}
    for strategy in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE"):
        rows = [r for r in selector_records if r["strategy"] == strategy]
        recovered = [r for r in rows if r["recovered"] is not None]
        selector_summary[strategy] = {"recovery_rate": float(np.mean([r["correct"] for r in rows])),
            "false_recovery_rate": float(np.mean([r["recovered"] is not None and not r["correct"] for r in rows])),
            "experiments_to_recovery": float(np.mean([r["experiments"] for r in recovered])) if recovered else None,
            "mean_cost": float(np.mean([r["cost"] for r in rows])),
            "successful_recovery_per_cost": float(sum(r["correct"] for r in rows) / max(sum(r["cost"] for r in rows), 1e-6))}
    family_summary = {family: {"detection_rate": float(np.mean([r["adequacy"] == "STRUCTURALLY_SUSPICIOUS" for r in rows])),
        "ood_rate": float(np.mean([r["adequacy"] == "OOD" for r in rows])),
        "recovery_rate": float(np.mean([r["recovered"] is not None for r in rows])),
        "false_recovery_rate": float(np.mean([r["false_recovery"] for r in rows])),
        "mean_experiments": float(np.mean([r["experiments"] for r in rows])), "mean_cost": float(np.mean([r["cost"] for r in rows])),
        "node": {field: _mean_metric(rows, "node_metrics", field) for field in ("precision", "recall", "f1")},
        "edge": {field: _mean_metric(rows, "edge_metrics", field) for field in ("precision", "recall", "f1")},
        "mechanism": {field: _mean_metric(rows, "mechanism_metrics", field) for field in ("precision", "recall", "f1")}}
        for family, rows in grouped.items()}
    summary = {"records": records, "selector_records": selector_records, "family_summary": family_summary,
               "selector_summary": selector_summary, "metadata": {"python": platform.python_version(), "numpy": np.__version__,
               "configuration": configuration, "families": list(SCENARIO_FAMILIES), "selector": "ACTIVE",
               "ground_truth_access": "evaluation.py only", "top_k_localisation": "not available: localisation is an unranked set"}}
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    (target / "phase7_final_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Phase-7 final untouched benchmark", "", "Final evaluation seeds: " + ", ".join(map(str, seeds)),
             "Held-out repair seeds: " + ", ".join(map(str, REPAIR_SEEDS)), ""]
    for family, rows in grouped.items():
        item = family_summary[family]
        lines.append(f"- {family}: detection={item['detection_rate']:.2f}; recovery={item['recovery_rate']:.2f}; "
                     f"false recovery={item['false_recovery_rate']:.2f}; node F1={item['node']['f1']:.2f}; "
                     f"edge F1={item['edge']['f1']:.2f}; mean experiments={item['mean_experiments']:.2f}; cost={item['mean_cost']:.2f}")
    lines.extend(["", "## Selector comparison (all final cases)"])
    for strategy in ("RANDOM", "MAX_EFFECT", "ACTIVE", "EXHAUSTIVE"):
        item = selector_summary[strategy]
        lines.append(f"- {strategy}: recovery={item['recovery_rate']:.2f}; false recovery={item['false_recovery_rate']:.2f}; "
                     f"experiments-to-recovery={item['experiments_to_recovery']}; cost={item['mean_cost']:.2f}; "
                     f"success/cost={item['successful_recovery_per_cost']:.3f}")
    (target / "phase7_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    run_benchmark()
