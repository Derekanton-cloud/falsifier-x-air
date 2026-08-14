"""Active, sequential experiment selection based on candidate model disagreement."""

from dataclasses import dataclass

import numpy as np

from .graph import AviationGraph, GraphSnapshot
from .mechanisms import mechanism_by_id
from .schema import ExperimentResult, MechanismEvidence, NetworkObservation


@dataclass(frozen=True)
class ExperimentConfig:
    maximum_experiments: int = 3
    evidence_tolerance: float = 0.35
    rejection_log_evidence: float = -2.0


class ActiveExperimentSelector:
    """Selects one untried intervention, never evaluates a candidate catalogue eagerly."""

    def predicted_effects(
        self, candidates: tuple[MechanismEvidence, ...], snapshot: GraphSnapshot, observation: NetworkObservation,
        prediction_mean: np.ndarray, intervention_id: str,
    ) -> np.ndarray:
        residual = observation.delay_vector(snapshot.flight_ids) - prediction_mean
        positive = np.maximum(residual, 0.0)
        total = float(positive.sum())
        effects = []
        for evidence in candidates:
            mechanism = mechanism_by_id(evidence.mechanism_id)
            if mechanism.intervention.identifier != intervention_id:
                effects.append(0.0)
                continue
            if mechanism.identifier == "AIRCRAFT_ROTATION":
                concentration = AviationGraph.residual_concentration(snapshot, residual)
                effects.append(total * concentration)
            else:
                effects.append(total)
        return np.asarray(effects, dtype=float)

    def select(
        self, candidates: tuple[MechanismEvidence, ...], snapshot: GraphSnapshot, observation: NetworkObservation,
        prediction_mean: np.ndarray, tried: set[str],
    ) -> str | None:
        interventions = {mechanism_by_id(item.mechanism_id).intervention.identifier for item in candidates} - tried
        if not interventions:
            return None
        # Weighted outcome disagreement / intervention cost: an explicit non-Bayesian EIG approximation.
        weights = np.exp(np.asarray([item.log_evidence for item in candidates], dtype=float))
        weights /= weights.sum()
        def utility(intervention: str) -> float:
            effects = self.predicted_effects(candidates, snapshot, observation, prediction_mean, intervention)
            mean = float(np.dot(weights, effects))
            disagreement = float(np.sqrt(np.dot(weights, (effects - mean) ** 2)))
            cost = next(mechanism_by_id(item.mechanism_id).intervention.cost for item in candidates
                        if mechanism_by_id(item.mechanism_id).intervention.identifier == intervention)
            return disagreement / cost
        return max(interventions, key=utility)


def run_experiment(twin, scenario, intervention_id: str, experiment_id: str) -> ExperimentResult:
    baseline = twin.observe(scenario)
    counterfactual = twin.counterfactual(scenario, intervention_id)
    intervention = next(mechanism_by_id(identifier).intervention for identifier in
                        ("AIRCRAFT_ROTATION", "AIRPORT_CAPACITY", "RESOURCE_DEPENDENCY")
                        if mechanism_by_id(identifier).intervention.identifier == intervention_id)
    return ExperimentResult(experiment_id, intervention, sum(baseline.delays.values()), sum(counterfactual.delays.values()), scenario.scenario_id)


def update_evidence(candidates: tuple[MechanismEvidence, ...], expected: np.ndarray, result: ExperimentResult,
                    config: ExperimentConfig) -> None:
    """Likelihood-shaped evidence update; it is documented as an approximation, not posterior probability."""
    scale = max(abs(result.effect), 1.0)
    for item, expected_effect in zip(candidates, expected):
        error = abs(result.effect - expected_effect) / scale
        item.expected_effects.append(float(expected_effect))
        item.observed_effects.append(result.effect)
        item.observations += 1
        item.log_evidence -= error / config.evidence_tolerance
        if item.log_evidence <= config.rejection_log_evidence:
            item.status = item.status.REJECTED
