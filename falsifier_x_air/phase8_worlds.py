"""Phase-8 world generators for generalization and identifiability stress tests.

Each public function returns (AviationDigitalTwin, TwinScenario) for a named
world family.  Prediction is computed separately in phase8_evaluation.py using
the same frozen prediction function as Phase-7 (weather/capacity prior only).

Phase-8 seed partitions
-----------------------
Discovery / calibration : 200–219   (used for debugging; not used to tune thresholds)
Final evaluation        : 2000–2019 (inspected only after discovery is complete)
Phase-7 seeds (100–109, 1000–1009, 101000–101009) are never touched here.

IDENTIFIABILITY CERTIFICATE (H3)
---------------------------------
For CONFOUNDED worlds this module computes an *identifiability certificate*
before any learner runs.  The certificate checks whether the complete observable
intervention-response vector (total delay reduction under each available
intervention) is distinguishable between competing hypotheses.

Criterion: relative_difference = |delta_A - delta_B| / max(delta_A, delta_B, 1e-6)
If relative_difference < INDISTINGUISHABLE_TOLERANCE the world is certified
INDISTINGUISHABLE and the expected learner output is INCONCLUSIVE.
If relative_difference >= INDISTINGUISHABLE_TOLERANCE the world is NOT confounded
and we record it as DISTINGUISHABLE (FALSIFIER-X should recover one mechanism).

The tolerance is frozen at 0.05 (5%) and must not be tuned on evaluation seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .schema import Flight
from .twin import AviationDigitalTwin, TwinScenario

# ----- frozen constants ---------------------------------------------------
INDISTINGUISHABLE_TOLERANCE: float = 0.05  # frozen before any evaluation seed is inspected
P8_DISCOVERY_SEEDS = tuple(range(200, 220))
P8_EVALUATION_SEEDS = tuple(range(2000, 2020))
_P7_FORBIDDEN = frozenset(range(100, 110)) | frozenset(range(1000, 1010)) | frozenset(range(101000, 101010))

STRENGTH_LEVELS = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0)
SCALE_SIZES = (8, 16, 32)
BUDGET_CAPS = (1, 2, 3, 5)


# ----- helpers ------------------------------------------------------------

def _assert_seed_clean(seed: int) -> None:
    if seed in _P7_FORBIDDEN:
        raise ValueError(f"Phase-8 must not use Phase-7 seed {seed}")


def _airports_for_size(n: int) -> list[str]:
    """Return enough airport labels so every flight has a distinct origin/destination pair."""
    return [chr(ord("A") + i % 26) + ("" if i < 26 else str(i // 26)) for i in range(n + 1)]


def build_scale_flights(n_flights: int, seed: int) -> tuple[Flight, ...]:
    """Build an N-flight network with ceil(N/2) aircraft and ceil(N/3) resources.

    Aircraft are assigned so each has a sequential chain: AC0→F0→F1, AC1→F2→F3, ...
    Resources are distributed round-robin across all flights.
    Topological structure is purely determined by n_flights; seed is not used here
    (randomness enters via TwinScenario.seed in observe()).
    """
    n_aircraft = math.ceil(n_flights / 2)
    n_resources = math.ceil(n_flights / 3)
    airports = _airports_for_size(n_flights)
    flights = []
    for i in range(n_flights):
        aircraft_id = f"AC{i % n_aircraft}"
        resource_id = f"R{i % n_resources}"
        origin = airports[i % len(airports)]
        dest = airports[(i + 1) % len(airports)]
        flights.append(Flight(f"F{i}", origin, dest, aircraft_id, i, resource_id))
    return tuple(flights)


def build_unseen_topo_flights(n_flights: int, seed: int) -> tuple[Flight, ...]:
    """Build a network whose aircraft/resource assignment is seed-randomised.

    This is an *unseen topology*: aircraft sequences and resource groups are not
    the standard sequential pattern.  The airport layout is also permuted.
    """
    rng = np.random.default_rng(seed + 10_000)
    airports = _airports_for_size(n_flights)
    airport_perm = list(airports[:n_flights + 1])
    rng.shuffle(airport_perm)
    n_aircraft = math.ceil(n_flights / 2)
    n_resources = math.ceil(n_flights / 3)
    aircraft_ids = [f"AC{rng.integers(0, n_aircraft)}" for _ in range(n_flights)]
    resource_ids = [f"R{rng.integers(0, n_resources)}" for _ in range(n_flights)]
    flights = []
    for i in range(n_flights):
        origin = airport_perm[i % len(airport_perm)]
        dest = airport_perm[(i + 1) % len(airport_perm)]
        flights.append(Flight(f"F{i}", origin, dest, aircraft_ids[i], i, resource_ids[i]))
    return tuple(flights)


def _standard_scenario(family: str, seed: int, mechanisms: set[str],
                        flights: tuple[Flight, ...],
                        noise_scale: float = 1.5,
                        mechanism_strength: float = 1.0,
                        transient_after: int | None = None,
                        rotation_coefficient: float = 0.85,
                        resource_coefficient: float = 0.60) -> tuple[AviationDigitalTwin, TwinScenario]:
    _assert_seed_clean(seed)
    scenario = TwinScenario(
        f"p8-{family.lower()}-{seed}", 0.85, 0.70, seed, family,
        noise_scale, mechanism_strength, 0,
        transient_after, rotation_coefficient, resource_coefficient,
    )
    twin = AviationDigitalTwin(flights, mechanisms, seed)
    return twin, scenario


# ----- world family builders ----------------------------------------------

def world_scale(n_flights: int, mechanism: str, seed: int) -> tuple[AviationDigitalTwin, TwinScenario]:
    """H1: larger graph with a single mechanism (AIRCRAFT_ROTATION, RESOURCE_DEPENDENCY, or AIRPORT_CAPACITY)."""
    flights = build_scale_flights(n_flights, seed)
    return _standard_scenario(f"SCALE_{n_flights}", seed, {mechanism}, flights)


def world_unseen_topo(mechanism: str, seed: int, n_flights: int = 8) -> tuple[AviationDigitalTwin, TwinScenario]:
    """H1/H2: unseen topology — aircraft/resource chains randomised by seed."""
    flights = build_unseen_topo_flights(n_flights, seed)
    return _standard_scenario("UNSEEN_TOPO", seed, {mechanism}, flights)


def world_transient(mechanism: str, seed: int, transient_after: int = 2, n_flights: int = 8) -> tuple[AviationDigitalTwin, TwinScenario]:
    """H4: mechanism applies only to flights with scheduled_time >= transient_after.

    With transient_after=2 and 8 flights, only the second half of the schedule
    is affected, producing partial temporal persistence.
    """
    flights = build_scale_flights(n_flights, seed)
    return _standard_scenario("TRANSIENT", seed, {mechanism}, flights, transient_after=transient_after)


def world_multi(mechanisms: set[str], seed: int, n_flights: int = 8) -> tuple[AviationDigitalTwin, TwinScenario]:
    """H5: multiple simultaneous mechanisms."""
    n = len(mechanisms)
    family = f"MULTI_{n}"
    flights = build_scale_flights(n_flights, seed)
    return _standard_scenario(family, seed, mechanisms, flights)


def world_strength_gradient(mechanism: str, strength: float, seed: int) -> tuple[AviationDigitalTwin, TwinScenario]:
    """H6: mechanism present at varying strengths from very weak to very strong."""
    flights = build_scale_flights(8, seed)
    return _standard_scenario("STRENGTH_GRADIENT", seed, {mechanism}, flights, mechanism_strength=strength)


def world_correct(seed: int, n_flights: int = 8) -> tuple[AviationDigitalTwin, TwinScenario]:
    """Control: no hidden mechanisms; model should be ADEQUATE."""
    flights = build_scale_flights(n_flights, seed)
    return _standard_scenario("CORRECT_P8", seed, set(), flights)


def world_ood(seed: int, n_flights: int = 8) -> tuple[AviationDigitalTwin, TwinScenario]:
    """Control: OOD — same as correct but the evaluation layer supplies ood_score=5.0."""
    flights = build_scale_flights(n_flights, seed)
    return _standard_scenario("OOD_P8", seed, set(), flights)


# ----- CONFOUNDED worlds and identifiability certificate ------------------

@dataclass(frozen=True)
class IdentifiabilityCertificate:
    """Computed before FALSIFIER-X runs; documents whether two mechanisms are distinguishable."""
    seed: int
    delta_rotation: float       # total delay reduction under disable_aircraft_rotation
    delta_resource: float       # total delay reduction under relieve_resource_dependency
    relative_difference: float  # |delta_A - delta_B| / max(delta_A, delta_B, 1e-6)
    verdict: str                # "INDISTINGUISHABLE" or "DISTINGUISHABLE"
    tolerance: float = INDISTINGUISHABLE_TOLERANCE


def confounded_certificate(twin: AviationDigitalTwin, scenario: TwinScenario) -> IdentifiabilityCertificate:
    """Compute the identifiability certificate for a CONFOUNDED world.

    Observes baseline and each single-intervention counterfactual to measure
    the total delay reduction.  Uses only the public twin API (observe/counterfactual);
    no private oracle is accessed.
    """
    baseline = sum(twin.observe(scenario).delays.values())
    cf_rotation = sum(twin.counterfactual(scenario, "disable_aircraft_rotation").delays.values())
    cf_resource = sum(twin.counterfactual(scenario, "relieve_resource_dependency").delays.values())
    delta_rotation = max(0.0, baseline - cf_rotation)
    delta_resource = max(0.0, baseline - cf_resource)
    denom = max(delta_rotation, delta_resource, 1e-6)
    rel_diff = abs(delta_rotation - delta_resource) / denom
    verdict = "INDISTINGUISHABLE" if rel_diff < INDISTINGUISHABLE_TOLERANCE else "DISTINGUISHABLE"
    return IdentifiabilityCertificate(scenario.seed, delta_rotation, delta_resource, rel_diff, verdict)


def world_confounded(seed: int, n_flights: int = 8) -> tuple[AviationDigitalTwin, TwinScenario, IdentifiabilityCertificate]:
    """H3: both AIRCRAFT_ROTATION and RESOURCE_DEPENDENCY active with equal coefficients.

    Setting rotation_coefficient = resource_coefficient = 0.75 and constructing
    flight schedules where every flight is in both a rotation chain AND a resource
    chain with the same predecessor structure makes the two intervention-response
    vectors numerically identical (within floating point).

    The certificate is computed and returned; the evaluation layer uses it to
    decide whether INCONCLUSIVE is the *correct* outcome rather than a failure.

    Note: with equal coefficients the mechanisms are structurally indistinguishable
    given the available interventions.  This is the genuine identifiability boundary.
    """
    _assert_seed_clean(seed)
    # Construct flights where EVERY flight shares aircraft and resource chains
    # with identical predecessor structure: AC0→F0→F2→F4→F6, R0→F0→F2→F4→F6
    # (all even-indexed flights share AC0 and R0; odd-indexed share AC1 and R1)
    # With rotation_coefficient == resource_coefficient, disabling either yields
    # the same propagation removal.
    n = n_flights
    flights = []
    for i in range(n):
        # alternating aircraft ensures 2 chains; resource mirrors aircraft assignment
        aircraft_id = f"AC{i % 2}"
        resource_id = f"R{i % 2}"
        origin = chr(ord("A") + i % 26)
        dest = chr(ord("A") + (i + 1) % 26)
        flights.append(Flight(f"F{i}", origin, dest, aircraft_id, i, resource_id))
    flights = tuple(flights)

    # equal coefficients → identical intervention signatures
    coeff = 0.75
    scenario = TwinScenario(
        f"p8-confounded-{seed}", 0.85, 0.70, seed, "CONFOUNDED",
        1.5, 1.0, 0, None, coeff, coeff,
    )
    twin = AviationDigitalTwin(flights, {"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY"}, seed)
    cert = confounded_certificate(twin, scenario)
    return twin, scenario, cert
