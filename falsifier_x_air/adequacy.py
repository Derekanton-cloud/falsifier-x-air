"""Evidence-based model-adequacy assessment (not a causal significance test)."""

from dataclasses import dataclass

import numpy as np

from .schema import AdequacyEvaluation, PredictionOutput


@dataclass(frozen=True)
class AdequacyConfig:
    persistence_required: int = 3
    ood_threshold: float = 3.0
    residual_z_threshold: float = 2.0
    minimum_evidence_fraction: float = 0.6


class StructuralAdequacyDetector:
    """Combines explicit diagnostic gates into a transparent evidence score.

    Thresholds are configuration values to calibrate and freeze on validation data;
    this score is only an adequacy signal, never a causal p-value.
    """

    def __init__(self, config: AdequacyConfig | None = None) -> None:
        self.config = config or AdequacyConfig()

    def evaluate(
        self,
        observed: np.ndarray,
        prediction: PredictionOutput,
        ood_score: float,
        persistence_count: int,
        structural_concentration: float,
        alternative_prediction: np.ndarray,
    ) -> AdequacyEvaluation:
        residual = observed - prediction.mean
        standardized = float(np.mean(np.abs(residual) / np.maximum(prediction.epistemic_std, 1e-6)))
        miscoverage = float(np.mean((observed < prediction.lower) | (observed > prediction.upper)))
        persistence = min(1.0, persistence_count / self.config.persistence_required)
        residual_agreement = float(np.corrcoef(residual, observed - alternative_prediction)[0, 1]) if len(observed) > 1 else 0.0
        if not np.isfinite(residual_agreement):
            residual_agreement = 0.0
        evidence = np.mean([
            standardized >= self.config.residual_z_threshold,
            miscoverage >= 0.5,
            persistence >= 1.0,
            structural_concentration >= 0.25,
            residual_agreement >= 0.5,
        ])
        if ood_score >= self.config.ood_threshold:
            state, reason = "ADEQUATE", "Distribution shift is a stronger explanation than missing structure."
        elif persistence < 1.0:
            state, reason = "ADEQUATE", "Residual pattern has not persisted long enough."
        elif evidence >= self.config.minimum_evidence_fraction:
            state, reason = "STRUCTURALLY_SUSPICIOUS", "Persistent, interval-violating residuals are graph-structured."
        else:
            state, reason = "INVESTIGATE", "Residual evidence is unusual but does not clear all adequacy gates."
        return AdequacyEvaluation(standardized, miscoverage, persistence, structural_concentration,
                                  residual_agreement, ood_score, float(evidence), state, reason)
