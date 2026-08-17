"""Phase-9 observable identifiability test for surviving mechanism hypotheses.

SCIENTIFIC PRINCIPLE
--------------------
Before declaring RECOVERED for a sole-surviving candidate H_i, verify that
every rejected competitor H_j was legitimately falsified — meaning there exists
at least one executed (or probe) intervention whose observed total-delay
reduction is *distinguishably different* between H_i's paired intervention and
H_j's paired intervention.

The distinguishability metric D uses only observable ExperimentResult data and
the public twin.counterfactual() API.  No private oracle (_benchmark_truth),
no confounded_certificate, and no hidden mechanism state are accessed.

DECISION RULE
-------------
For each rejected competitor H_j:
  1. Look up the observed effect Δ_j under H_j's paired intervention a_j.
     If a_j was not executed during the main loop, run one probe experiment.
  2. Compute the response distance:
       D(H_i, H_j) = |Δ_i - Δ_j| / max(Δ_i, Δ_j, σ_eff, 1e-6)
  3. If D ≤ ε_ident → H_j cannot be legitimately distinguished from H_i →
     return INCONCLUSIVE (INDISTINGUISHABLE_EQUIVALENCE_CLASS).

The identifiability check operates on the *surviving competitor set*, not on
initially-generated candidates that were eliminated by their own experiment.

EPSILON CALIBRATION (Phase-9 discovery seeds 200–219, frozen before evaluation)
------------------------------------------------------------------
Confounded worlds (equal-coefficient rotation + resource, mirrored topology):
  Δ_rotation ≈ Δ_resource (certificate confirms equality) → D ≈ 0.00

Single-mechanism worlds (only one mechanism active):
  Competitor probe returns Δ ≈ 0.0 (inactive mechanism) → D ≈ 1.00

Observed gap on discovery seeds: [0.000, 0.001] vs [0.991, 1.000].
ε_ident = 0.20 is placed well between the two clusters and is now frozen.

NO MODIFICATION was made to update_evidence(), selection logic, adequacy
thresholds, or mechanism registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .experiments import run_experiment
from .mechanisms import mechanism_by_id
from .schema import ExperimentResult, MechanismEvidence

# ---------------------------------------------------------------------------
# Frozen threshold — calibrated on discovery seeds 200–219
# ---------------------------------------------------------------------------
EPSILON_IDENT: float = 0.20


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IdentifiabilityResult:
    """Outcome of the pre-recovery identifiability check.

    is_identifiable : True  → single survivor is legitimately distinguished
                               from all rejected competitors; RECOVERED allowed.
                      False → at least one competitor is observationally
                               indistinguishable; INCONCLUSIVE required.
    reason          : human-readable classification string.
    d_values        : ((survivor_id, competitor_id, D), ...) for all pairs.
    probe_experiments : ExperimentResult objects from any additional probes
                        executed during the check (may be empty).
    """
    is_identifiable: bool
    reason: str
    d_values: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)
    probe_experiments: tuple[ExperimentResult, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _observed_effects(experiments: list[ExperimentResult]) -> dict[str, float]:
    """Map intervention_id → observed total-delay reduction from executed experiments."""
    return {exp.intervention.identifier: exp.effect for exp in experiments}


def response_distance(delta_i: float, delta_j: float, sigma_eff: float) -> float:
    """Normalised absolute difference between two observed intervention responses.

    D(H_i, H_j) = |Δ_i - Δ_j| / max(Δ_i, Δ_j, σ_eff, 1e-6)

    Returns 0.0 when both effects are zero (degenerate world).
    Returns values in [0, 1] for non-degenerate worlds.
    """
    denom = max(delta_i, delta_j, abs(sigma_eff), 1e-6)
    return abs(delta_i - delta_j) / denom


# ---------------------------------------------------------------------------
# Main identifiability check
# ---------------------------------------------------------------------------

def check_identifiability(
    survivor: MechanismEvidence,
    all_candidates: tuple[MechanismEvidence, ...],
    experiments: list[ExperimentResult],
    twin: object,
    scenario: object,
    scenario_id_prefix: str = "ident-probe",
    sigma_eff: float = 1.0,
    epsilon: float = EPSILON_IDENT,
) -> IdentifiabilityResult:
    """Check whether the sole surviving hypothesis is observationally identifiable.

    The check is applied to the *surviving competitor set* — candidates that
    were rejected during the main experiment loop, whether via cross-rejection
    or through their own paired intervention.  Competitors that were never
    tested via their own intervention receive one probe experiment.

    Uses only the public twin.counterfactual() API (via run_experiment).
    Does not access _benchmark_truth or any private twin state.

    Parameters
    ----------
    survivor        : The single non-rejected candidate hypothesis.
    all_candidates  : All candidates generated at the start (including rejected).
    experiments     : ExperimentResult objects from the main loop.
    twin            : AviationDigitalTwin instance (public API only).
    scenario        : TwinScenario for the current case.
    scenario_id_prefix : Prefix for probe experiment IDs.
    sigma_eff       : Observable uncertainty floor (e.g. mean epistemic std).
    epsilon         : Frozen distinguishability threshold (default EPSILON_IDENT).
    """
    survivor_intervention = mechanism_by_id(survivor.mechanism_id).intervention.identifier
    observed = _observed_effects(experiments)
    delta_i = observed.get(survivor_intervention, 0.0)

    d_values: list[tuple[str, str, float]] = []
    probe_exps: list[ExperimentResult] = []
    indistinguishable_pairs: list[tuple[str, str]] = []

    for candidate in all_candidates:
        if candidate.mechanism_id == survivor.mechanism_id:
            continue
        # Only check competitors that were rejected (status == REJECTED).
        # Non-rejected candidates are handled by the len(survivors) > 1 branch.
        if candidate.status.name != "REJECTED":
            continue

        cand_intervention = mechanism_by_id(candidate.mechanism_id).intervention.identifier

        # Obtain the observed effect under this competitor's paired intervention.
        if cand_intervention in observed:
            delta_j = observed[cand_intervention]
        else:
            # Competitor's own intervention was never run during the main loop.
            # Run one probe experiment using the public twin API to measure Δ_j.
            probe_id = f"{scenario_id_prefix}-{candidate.mechanism_id.lower()}"
            probe_result = run_experiment(twin, scenario, cand_intervention, probe_id)
            probe_exps.append(probe_result)
            delta_j = probe_result.effect

        d = response_distance(delta_i, delta_j, sigma_eff)
        d_values.append((survivor.mechanism_id, candidate.mechanism_id, d))

        if d <= epsilon:
            indistinguishable_pairs.append((survivor.mechanism_id, candidate.mechanism_id))

    if indistinguishable_pairs:
        return IdentifiabilityResult(
            is_identifiable=False,
            reason="INDISTINGUISHABLE_EQUIVALENCE_CLASS",
            d_values=tuple(d_values),
            probe_experiments=tuple(probe_exps),
        )

    return IdentifiabilityResult(
        is_identifiable=True,
        reason="LEGITIMATELY_IDENTIFIED",
        d_values=tuple(d_values),
        probe_experiments=tuple(probe_exps),
    )
