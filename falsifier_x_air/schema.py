from dataclasses import dataclass

@dataclass(frozen=True)
class Flight:
    flight_id: str
    origin: str
    destination: str
    aircraft_id: str
    scheduled_time: int

@dataclass(frozen=True)
class Mechanism:
    name: str
    source_type: str
    relation_type: str
    target_type: str
    expected_effect: str
    description: str

@dataclass
class ExperimentResult:
    mechanism: str
    intervention: str
    baseline_delay: float
    intervention_delay: float

    @property
    def effect(self):
        return self.baseline_delay - self.intervention_delay
