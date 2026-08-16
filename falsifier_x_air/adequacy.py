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
    minimum_structural_channels: int = 3
    directional_lift_threshold: float = 3.0


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
        directional_lift: float = 0.0,
    ) -> AdequacyEvaluation:
        residual = observed - prediction.mean
        standardized = float(np.mean(np.abs(residual) / np.maximum(prediction.epistemic_std, 1e-6)))
        miscoverage = float(np.mean((observed < prediction.lower) | (observed > prediction.upper)))
        persistence = min(1.0, persistence_count / self.config.persistence_required)
        # A duplicate alternative does not constitute independent residual structure.
        residual_agreement = (float(np.corrcoef(residual, observed - alternative_prediction)[0, 1])
                              if len(observed) > 1 and not np.allclose(prediction.mean, alternative_prediction) else 0.0)
        if not np.isfinite(residual_agreement):
            residual_agreement = 0.0
        channels = {
            "prediction_uncertainty_violation": standardized >= self.config.residual_z_threshold and miscoverage >= 0.5,
            "persistence": persistence >= 1.0,
            "graph_concentration": structural_concentration >= 0.25,
            "residual_structure": residual_agreement >= 0.5,
            "directional_propagation": directional_lift >= self.config.directional_lift_threshold,
            "ood": ood_score >= self.config.ood_threshold,
        }
        structural_count = sum(channels[name] for name in (
            "prediction_uncertainty_violation", "persistence", "graph_concentration",
            "residual_structure", "directional_propagation"
        ))
        evidence = structural_count / 5.0
        if ood_score >= self.config.ood_threshold:
            state, reason = "OOD", "Observed feature distribution is outside the frozen reference range."
        elif (channels["prediction_uncertainty_violation"] and channels["persistence"]
              and (channels["directional_propagation"] or channels["graph_concentration"])):
            state, reason = "STRUCTURALLY_SUSPICIOUS", "Persistent interval violations have observable graph-structural support."
        elif not channels["prediction_uncertainty_violation"]:
            state, reason = "ADEQUATE", "No prediction/uncertainty violation is present."
        else:
            state, reason = "INCONCLUSIVE", "Some diagnostic channels are unusual, but structural evidence is insufficient."
        return AdequacyEvaluation(standardized, miscoverage, persistence, structural_concentration,
                                  residual_agreement, ood_score, float(evidence), state, reason, channels)
