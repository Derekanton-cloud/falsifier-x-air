from dataclasses import dataclass
import numpy as np

@dataclass
class AdequacyResult:
    residual_score: float
    uncertainty_violation: float
    persistence_score: float
    structural_score: float
    cross_model_score: float
    mis_score: float
    state: str
    reason: str

class StructuralAdequacyDetector:
    def __init__(self, persistence_required=3, mis_threshold=0.65,
                 ood_threshold=3.0):
        self.persistence_required = persistence_required
        self.mis_threshold = mis_threshold
        self.ood_threshold = ood_threshold

    @staticmethod
    def norm(x):
        return float(np.clip(x, 0.0, 1.0))

    def evaluate(self, y_true, prediction, ood_score,
                 persistence_count, graph_concentration, cross_model_gap):
        residual = np.abs(y_true-prediction.mean)
        scale = np.maximum(np.abs(prediction.mean)+5.0, 5.0)
        r = self.norm(np.mean(residual/scale))
        outside = (y_true < prediction.lower) | (y_true > prediction.upper)
        u = self.norm(np.mean(outside))
        p = self.norm(persistence_count/max(self.persistence_required, 1))
        s = self.norm(graph_concentration)
        c = self.norm(cross_model_gap)

        mis = 0.25*r + 0.20*u + 0.20*p + 0.20*s + 0.15*c

        if ood_score >= self.ood_threshold:
            state, reason = "ADEQUATE", "OOD is a stronger alternative explanation."
        elif persistence_count < self.persistence_required:
            state, reason = "ADEQUATE", "Insufficient persistence."
        elif u < 0.5:
            state, reason = "ADEQUATE", "Behaviour remains inside uncertainty."
        elif mis >= self.mis_threshold:
            state, reason = "STRUCTURALLY_SUSPICIOUS", (
                "Persistent structured deviation remains after uncertainty/OOD gates."
            )
        else:
            state, reason = "INVESTIGATE", "Evidence is unusual but not conclusive."

        return AdequacyResult(r,u,p,s,c,mis,state,reason)
