"""Model-based recovery choice; the twin evaluates only after a choice is made."""

from .mechanisms import mechanism_by_id
from .schema import CounterfactualIntervention, RecoveryAction, RecoveryEvaluation


def candidate_actions(recovered_mechanism: str | None) -> tuple[RecoveryAction, ...]:
    actions = [RecoveryAction("NO_ACTION", None, 0.0)]
    if recovered_mechanism:
        intervention = mechanism_by_id(recovered_mechanism).intervention
        actions.append(RecoveryAction(intervention.identifier.upper(), intervention, 0.35))
    return tuple(actions)


def choose_recovery(predicted_total_delay: float, recovered_mechanism: str | None) -> RecoveryEvaluation:
    """Choose using repaired/unrepaired model estimates, with no environment argument."""
    evaluations = [RecoveryEvaluation(action, predicted_total_delay * (1.0 - action.estimated_effect_fraction))
                   for action in candidate_actions(recovered_mechanism)]
    return min(evaluations, key=lambda evaluation: evaluation.estimated_total_delay)


def evaluate_selected_action(twin, scenario, decision: RecoveryEvaluation) -> RecoveryEvaluation:
    """Environment-only evaluation step, intentionally separate from decision generation."""
    observation = twin.counterfactual(scenario, decision.action.intervention.identifier) if decision.action.intervention else twin.observe(scenario)
    return RecoveryEvaluation(decision.action, decision.estimated_total_delay, sum(observation.delays.values()))
