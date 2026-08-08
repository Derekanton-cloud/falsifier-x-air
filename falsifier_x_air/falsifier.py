from dataclasses import dataclass
from .adequacy import StructuralAdequacyDetector
from .experiments import ActiveExperimentSelector, run_experiment
from .mechanisms import generate_candidates

@dataclass
class InvestigationResult:
    adequacy: object
    candidates: list
    selected_experiment: str | None
    experiment_results: dict
    selected_mechanism: str | None

class FalsifierXAir:
    def __init__(self):
        self.detector = StructuralAdequacyDetector()
        self.selector = ActiveExperimentSelector()

    def investigate(self, twin, y_true, prediction, ood_score,
                    persistence_count, graph_concentration,
                    cross_model_gap, local_context):
        adequacy = self.detector.evaluate(
            y_true, prediction, ood_score, persistence_count,
            graph_concentration, cross_model_gap
        )
        if adequacy.state != "STRUCTURALLY_SUSPICIOUS":
            return InvestigationResult(
                adequacy, [], None, {}, None
            )

        candidates = generate_candidates(local_context)
        selected_experiment = self.selector.select(candidates)
        results = {}

        for c in candidates:
            exp = self.selector.experiment_for(c)
            before, after = run_experiment(twin, exp)
            results[c.name] = {
                "intervention": exp,
                "before": before,
                "after": after,
                "effect": before-after,
            }

        selected = max(results, key=lambda k: results[k]["effect"])
        return InvestigationResult(
            adequacy, candidates, selected_experiment,
            results, selected
        )
