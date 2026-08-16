"""Typed domain contracts shared across the research prototype."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


class MechanismStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    TESTING = "TESTING"
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class Flight:
    flight_id: str
    origin: str
    destination: str
    aircraft_id: str
    scheduled_time: int
    resource_id: str | None = None


@dataclass(frozen=True)
class NetworkObservation:
    """Observable state from one synthetic scenario; no hidden twin state."""
    flights: tuple[Flight, ...]
    delays: Mapping[str, float]
    weather: float
    capacity: float
    scenario_id: str

    def delay_vector(self, flight_ids: Sequence[str]) -> NDArray[np.float64]:
        return np.asarray([self.delays[flight_id] for flight_id in flight_ids], dtype=float)


@dataclass(frozen=True)
class PredictionOutput:
    mean: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    epistemic_std: NDArray[np.float64]


@dataclass(frozen=True)
class CounterfactualIntervention:
    identifier: str
    description: str
    cost: float


@dataclass(frozen=True)
class Mechanism:
    identifier: str
    source_type: str
    relation_type: str
    target_type: str
    intervention: CounterfactualIntervention
    description: str
    expected_effect_signature: str = ""
    falsification_condition: str = ""


@dataclass
class MechanismEvidence:
    mechanism_id: str
    status: MechanismStatus = MechanismStatus.CANDIDATE
    log_evidence: float = 0.0
    observations: int = 0
    expected_effects: list[float] = field(default_factory=list)
    observed_effects: list[float] = field(default_factory=list)
    experiment_interventions: list[str] = field(default_factory=list)
    baseline_delays: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    intervention: CounterfactualIntervention
    baseline_delay: float
    intervention_delay: float
    scenario_id: str

    @property
    def effect(self) -> float:
        return self.baseline_delay - self.intervention_delay


@dataclass(frozen=True)
class AdequacyEvaluation:
    standardized_residual: float
    interval_miscoverage: float
    persistence_fraction: float
    structural_concentration: float
    residual_agreement: float
    ood_score: float
    evidence_score: float
    state: str
    reason: str
    channels: Mapping[str, float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryAction:
    identifier: str
    intervention: CounterfactualIntervention | None
    estimated_effect_fraction: float


@dataclass(frozen=True)
class RecoveryEvaluation:
    action: RecoveryAction
    estimated_total_delay: float
    realized_total_delay: float | None = None
