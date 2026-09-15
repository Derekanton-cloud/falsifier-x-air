"""Phase 10 evaluation of causal model repair."""

import json
from pathlib import Path
import numpy as np

from .evaluation import scenario_family, _prediction
from .falsifier import FalsifierXAir, InvestigationContext
from .graph import AviationGraph
from .repair import estimate_coefficients, apply_repair, apply_generic_repair

REPAIR_EVAL_SEEDS = range(500000, 500100)

def evaluate_case(family: str, seed: int, coeffs: dict[str, float]) -> dict | None:
    twin, scenario = scenario_family(family, seed)
    observation = twin.observe(scenario)
    graph = AviationGraph.build(observation.flights)
    prediction = _prediction(twin, scenario)
    ood_score = 5.0 if family == "OOD" else 0.0
    
    context = InvestigationContext(
        observation, graph, prediction, prediction.mean, ood_score, 3, scenario,
    )
    result = FalsifierXAir().investigate(twin, context)
    
    # Gating rule - MUST be a hard branch
    if result.recovered_mechanism is None or result.identifiability_reason != "LEGITIMATELY_IDENTIFIED":
        return None
        
    recovered = result.recovered_mechanism
    
    # Ground truth
    truth = twin._benchmark_truth()
    
    # Target
    target = observation.delay_vector(graph.flight_ids)
    
    # 1. ORIGINAL
    pred_orig = prediction.mean.copy()
    
    # 2. CORRECT_REPAIR
    pred_correct = apply_repair(prediction.mean, recovered, coeffs[recovered], observation, graph)
    
    # 3. WRONG_REPAIR
    # Choose decoy
    if recovered == "AIRCRAFT_ROTATION":
        decoy = "RESOURCE_DEPENDENCY"
    elif recovered == "RESOURCE_DEPENDENCY":
        decoy = "AIRCRAFT_ROTATION"
    else:
        decoy = "AIRCRAFT_ROTATION"
    pred_wrong = apply_repair(prediction.mean, decoy, coeffs[decoy], observation, graph)
    
    # 4. GENERIC_AUGMENTATION
    pred_generic = apply_generic_repair(prediction.mean, coeffs.get("GENERIC", 0.0), observation, graph)
    
    # 5. ORACLE_REPAIR
    # Evaluator-only reference condition using the twin's TRUE coefficient.
    if recovered == "AIRCRAFT_ROTATION":
        oracle_coeff = scenario.rotation_coefficient * scenario.mechanism_strength
    elif recovered == "RESOURCE_DEPENDENCY":
        oracle_coeff = scenario.resource_coefficient * scenario.mechanism_strength
    elif recovered == "AIRPORT_CAPACITY":
        # Congestion is 20.0 * scenario.mechanism_strength * max(0.0, 1.0 - scenario.capacity)
        # And we use signal max(0.0, 1.0 - scenario.capacity) in the feature.
        oracle_coeff = 20.0 * scenario.mechanism_strength
    else:
        oracle_coeff = 0.0
        
    pred_oracle = apply_repair(prediction.mean, recovered, oracle_coeff, observation, graph)
    
    # Metrics
    def metrics(pred):
        mae = np.mean(np.abs(pred - target))
        rmse = np.sqrt(np.mean((pred - target)**2))
        return mae, rmse
        
    orig_mae, orig_rmse = metrics(pred_orig)
    correct_mae, correct_rmse = metrics(pred_correct)
    wrong_mae, wrong_rmse = metrics(pred_wrong)
    generic_mae, generic_rmse = metrics(pred_generic)
    oracle_mae, oracle_rmse = metrics(pred_oracle)
    
    return {
        "family": family,
        "seed": seed,
        "recovered": recovered,
        "true_mechanisms": list(truth["mechanisms"]),
        "orig_mae": orig_mae, "orig_rmse": orig_rmse,
        "correct_mae": correct_mae, "correct_rmse": correct_rmse,
        "wrong_mae": wrong_mae, "wrong_rmse": wrong_rmse,
        "generic_mae": generic_mae, "generic_rmse": generic_rmse,
        "oracle_mae": oracle_mae, "oracle_rmse": oracle_rmse,
    }

def run_phase10_evaluation() -> None:
    # 1. Estimate coefficients on discovery seeds ONLY
    coeffs = estimate_coefficients()
    
    # Ensure REPAIR_EVAL_SEEDS is disjoint from discovery, evaluation, repair seeds
    from .evaluation import DISCOVERY_SEEDS, EVALUATION_SEEDS, REPAIR_SEEDS
    assert set(REPAIR_EVAL_SEEDS).isdisjoint(DISCOVERY_SEEDS)
    assert set(REPAIR_EVAL_SEEDS).isdisjoint(EVALUATION_SEEDS)
    assert set(REPAIR_EVAL_SEEDS).isdisjoint(REPAIR_SEEDS)
    
    # 2. Evaluate
    results = []
    from .evaluation import SCENARIO_FAMILIES
    
    # We only care about cases where a mechanism is recovered. Let's just run over typical causal families
    families_to_test = ["AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY", "AIRPORT_CAPACITY", "MULTIPLE"]
    
    for seed in REPAIR_EVAL_SEEDS:
        for family in families_to_test:
            res = evaluate_case(family, seed, coeffs)
            if res is not None:
                results.append(res)
                
    # Aggregate metrics
    report = [
        "# Phase 10: Causal Model Repair & Blind Improvement\n",
        "## Coefficient Estimation (Discovery Seeds 100-109)"
    ]
    
    # Actually I should also fetch how many discovery cases each came from, but it's fine just printing the coeffs.
    for m, c in coeffs.items():
        report.append(f"- {m}: {c:.3f}")
        
    report.append("\n## Blind Repair Evaluation (Seeds 500000-500099)")
    
    all_mechs = set(r["recovered"] for r in results)
    
    def agg_metrics(subset, name):
        if not subset:
            return []
        
        orig_mae = np.mean([r["orig_mae"] for r in subset])
        correct_mae = np.mean([r["correct_mae"] for r in subset])
        wrong_mae = np.mean([r["wrong_mae"] for r in subset])
        gen_mae = np.mean([r["generic_mae"] for r in subset])
        oracle_mae = np.mean([r["oracle_mae"] for r in subset])
        
        c_imp = np.mean([1 if r["correct_mae"] < r["orig_mae"] else 0 for r in subset])
        w_imp = np.mean([1 if r["wrong_mae"] < r["orig_mae"] else 0 for r in subset])
        
        lines = [
            f"### {name} (N={len(subset)})",
            f"MAE: ORIGINAL={orig_mae:.3f} | CORRECT={correct_mae:.3f} | WRONG={wrong_mae:.3f} | GENERIC={gen_mae:.3f} | ORACLE={oracle_mae:.3f}",
            f"Delta (Repair - Orig): CORRECT={correct_mae - orig_mae:.3f} | WRONG={wrong_mae - orig_mae:.3f} | GENERIC={gen_mae - orig_mae:.3f}",
            f"Fraction Improved: CORRECT={c_imp:.2f} | WRONG={w_imp:.2f}",
            f"Distance to Oracle (CORRECT - ORACLE): {correct_mae - oracle_mae:.3f}"
        ]
        return lines

    report.extend(agg_metrics(results, "AGGREGATED"))
    
    for m in all_mechs:
        subset = [r for r in results if r["recovered"] == m]
        report.extend(agg_metrics(subset, f"Mechanism: {m}"))
        
    report.append("\n## Conclusion")
    # Determine support
    correct_mae = np.mean([r["correct_mae"] for r in results])
    wrong_mae = np.mean([r["wrong_mae"] for r in results])
    orig_mae = np.mean([r["orig_mae"] for r in results])
    generic_mae = np.mean([r["generic_mae"] for r in results])
    
    if correct_mae < orig_mae and correct_mae < wrong_mae and correct_mae < generic_mae:
        report.append("The results support the claim: 'the recovered mechanism is functionally useful, not merely correlated'.")
    else:
        report.append("The results DO NOT support the claim. The recovered mechanism does not reliably outperform decoy or generic augmentations.")
        
    Path("data/processed/phase10_repair_results.json").write_text(json.dumps(results, indent=2))
    Path("data/processed/phase10_repair_report.md").write_text("\n".join(report))

if __name__ == "__main__":
    run_phase10_evaluation()
