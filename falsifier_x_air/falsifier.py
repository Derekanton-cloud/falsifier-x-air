"""Closed-loop adequacy, localization, active falsification and model-repair proposal."""

from dataclasses import dataclass

import numpy as np

from .adequacy import StructuralAdequacyDetector
from .experiments import ActiveExperimentSelector, ExperimentConfig, run_experiment, update_evidence
from .graph import AviationGraph, GraphSnapshot
from .mechanisms import generate_candidates
from .schema import AdequacyEvaluation, ExperimentResult, MechanismEvidence, NetworkObservation, PredictionOutput


@dataclass(frozen=True)
class InvestigationContext:
    observation: NetworkObservation
    graph: GraphSnapshot
    prediction: PredictionOutput
    alternative_prediction: np.ndarray
    ood_score: float
    persistence_count: int
    scenario: object


@dataclass(frozen=True)
class InvestigationResult:
    adequacy: AdequacyEvaluation
    affected_flights: tuple[str, ...]
    candidates: tuple[MechanismEvidence, ...]
    experiment_results: tuple[ExperimentResult, ...]
    recovered_mechanism: str | None


class FalsifierXAir:
    def __init__(self, detector: StructuralAdequacyDetector | None = None, experiment_config: ExperimentConfig | None = None) -> None:
        self.detector = detector or StructuralAdequacyDetector()
        self.experiment_config = experiment_config or ExperimentConfig()
        self.selector = ActiveExperimentSelector()

    def investigate(self, twin, context: InvestigationContext) -> InvestigationResult:
        observed = context.observation.delay_vector(context.graph.flight_ids)
        residuals = observed - context.prediction.mean
        adequacy = self.detector.evaluate(
            observed, context.prediction, context.ood_score, context.persistence_count,
            AviationGraph.residual_concentration(context.graph, residuals), context.alternative_prediction,
        )
        if adequacy.state != "STRUCTURALLY_SUSPICIOUS":
            return InvestigationResult(adequacy, (), (), (), None)
        affected = AviationGraph.localize(context.graph, residuals)
        candidates = generate_candidates(context.graph, affected)
        tried: set[str] = set()
        results: list[ExperimentResult] = []
        for index in range(self.experiment_config.maximum_experiments):
            plausible = tuple(item for item in candidates if item.status != item.status.REJECTED)
            if not plausible:
                break
            if len(plausible) == 1:
                survivor = plausible[0]
                survivor_intervention = self.selector.select(
                    plausible, context.graph, context.observation, context.prediction.mean, tried,
                )
                if survivor_intervention is None:
                    break
                # A single survivor is a provisional hypothesis: confirm it with
                # its own intervention before declaring a recovered mechanism.
                intervention = survivor_intervention
            else:
                intervention = self.selector.select(plausible, context.graph, context.observation, context.prediction.mean, tried)
            if intervention is None:
                break
            expected = self.selector.predicted_effects(candidates, context.graph, context.observation, context.prediction.mean, intervention)
            result = run_experiment(twin, context.scenario, intervention, f"{context.observation.scenario_id}-exp-{index + 1}")
            update_evidence(candidates, expected, result, self.experiment_config)
            results.append(result)
            tried.add(intervention)
        survivors = [item for item in candidates if item.status != item.status.REJECTED]
        recovered = max(survivors, key=lambda item: item.log_evidence).mechanism_id if len(survivors) == 1 else None
        if recovered:
            survivors[0].status = survivors[0].status.SUPPORTED
        return InvestigationResult(adequacy, affected, candidates, tuple(results), recovered)
