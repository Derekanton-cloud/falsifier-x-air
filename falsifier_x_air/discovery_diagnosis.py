"""Discovery-partition-only diagnostic traces for Phase 7.

This evaluator module may read private twin truth solely to label the trace after
the learner has acted. It deliberately accepts only the sealed discovery range.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .evaluation import DISCOVERY_SEEDS, _prediction, scenario_family
from .experiments import run_experiment
from .falsifier import FalsifierXAir, InvestigationContext
from .graph import AviationGraph
from .mechanisms import mechanism_by_id


def diagnose_family(family: str, seed: int) -> dict[str, object]:
    if seed not in DISCOVERY_SEEDS:
        raise ValueError("Discovery diagnosis accepts only seeds 100--109")
    twin, scenario = scenario_family(family, seed)
    observation = twin.observe(scenario)
    graph = AviationGraph.build(observation.flights)
    prediction = _prediction(twin, scenario)
    residual = observation.delay_vector(graph.flight_ids) - prediction.mean
    result = FalsifierXAir().investigate(twin, InvestigationContext(
        observation, graph, prediction, prediction.mean, 5.0 if family == "OOD" else 0.0, 3, scenario,
    ))
    # Oracle values label this completed trace; no learner call receives them.
    truth = twin._benchmark_truth()
    intervention_effects = {
        mechanism.identifier: run_experiment(twin, scenario, mechanism.intervention.identifier, f"diagnostic-{seed}").effect
        for mechanism in (mechanism_by_id("AIRCRAFT_ROTATION"), mechanism_by_id("RESOURCE_DEPENDENCY"), mechanism_by_id("AIRPORT_CAPACITY"))
    }
    return {
        "family": family, "seed": seed, "hidden_mechanisms": sorted(truth["mechanisms"]),
        "observable_total_delay": float(sum(observation.delays.values())),
        "residual": [round(float(value), 4) for value in residual],
        "adequacy": {"state": result.adequacy.state, "channels": dict(result.adequacy.channels),
                     "standardized": result.adequacy.standardized_residual,
                     "miscoverage": result.adequacy.interval_miscoverage,
                     "concentration": result.adequacy.structural_concentration,
                     "directional_lift": AviationGraph.directional_residual_lift(graph, residual, prediction.epistemic_std)},
        "localized_flights": list(result.affected_flights),
        "truth_affected_flights": sorted(truth["affected_nodes"]),
        "generated_candidates": [item.mechanism_id for item in result.candidates],
        "intervention_effects": intervention_effects,
        "evidence": [{"mechanism": item.mechanism_id, "status": item.status.value, "log_evidence": item.log_evidence,
                      "expected": item.expected_effects, "observed": item.observed_effects,
                      "interventions": item.experiment_interventions} for item in result.candidates],
        "decision": result.recovered_mechanism or "INCONCLUSIVE",
    }


def run_discovery_diagnosis(output_dir: str | Path = "data/processed") -> dict[str, object]:
    records = [diagnose_family(family, seed) for family in ("AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY", "CORRECT", "NOISE", "OOD", "DECOY") for seed in DISCOVERY_SEEDS]
    payload = {"partition": "discovery only", "seeds": list(DISCOVERY_SEEDS), "records": records}
    path = Path(output_dir); path.mkdir(parents=True, exist_ok=True)
    (path / "phase7_discovery_diagnosis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Phase-7 discovery-only diagnosis", "", "Seeds: 100--109. Final and held-out partitions were not run.", ""]
    for family in ("AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY", "CORRECT", "NOISE", "OOD", "DECOY"):
        rows = [row for row in records if row["family"] == family]
        states = {state: sum(row["adequacy"]["state"] == state for row in rows) for state in
                  ("ADEQUATE", "OOD", "STRUCTURALLY_SUSPICIOUS", "INCONCLUSIVE")}
        recovered = sum(row["decision"] != "INCONCLUSIVE" for row in rows)
        lines.append(f"- {family}: states={states}; recovery={recovered}/{len(rows)}")
    (path / "phase7_discovery_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    run_discovery_diagnosis()
