"""Evidence-driven recovery choice; the twin evaluates only after a choice is made."""

from statistics import median

from .mechanisms import mechanism_by_id
from .schema import MechanismEvidence, RecoveryAction, RecoveryEvaluation


def measured_effect_fraction(mechanism_id: str, evidence: MechanismEvidence) -> float | None:
    """Estimate intervention benefit from matching, observed counterfactual trials.

    The estimate is the median paired delay reduction divided by its baseline
    delay. It intentionally does not infer an effect when the investigation has
    not actually measured the recovered mechanism's intervention.
    """
    intervention_id = mechanism_by_id(mechanism_id).intervention.identifier
    fractions = [effect / baseline for effect, baseline, experiment_id in zip(
        evidence.observed_effects, evidence.baseline_delays, evidence.experiment_interventions
    ) if experiment_id == intervention_id and baseline > 0.0]
    return float(median(fractions)) if fractions else None


def candidate_actions(recovered_mechanism: str | None, evidence: MechanismEvidence | None = None) -> tuple[RecoveryAction, ...]:
    actions = [RecoveryAction("NO_ACTION", None, 0.0)]
    if recovered_mechanism is not None and evidence is not None:
        effect_fraction = measured_effect_fraction(recovered_mechanism, evidence)
        if effect_fraction is not None:
            intervention = mechanism_by_id(recovered_mechanism).intervention
            actions.append(RecoveryAction(intervention.identifier.upper(), intervention, effect_fraction))
    return tuple(actions)


def choose_recovery(
    predicted_total_delay: float, recovered_mechanism: str | None, evidence: MechanismEvidence | None = None,
) -> RecoveryEvaluation:
    """Choose using prediction and learned experiment evidence, with no twin argument."""
    evaluations = [RecoveryEvaluation(action, predicted_total_delay * (1.0 - action.estimated_effect_fraction))
                   for action in candidate_actions(recovered_mechanism, evidence)]
    return min(evaluations, key=lambda evaluation: evaluation.estimated_total_delay)


def evaluate_selected_action(twin, scenario, decision: RecoveryEvaluation) -> RecoveryEvaluation:
    """Environment-only evaluation step, intentionally separate from decision generation."""
    observation = twin.counterfactual(scenario, decision.action.intervention.identifier) if decision.action.intervention else twin.observe(scenario)
    return RecoveryEvaluation(decision.action, decision.estimated_total_delay, sum(observation.delays.values()))
