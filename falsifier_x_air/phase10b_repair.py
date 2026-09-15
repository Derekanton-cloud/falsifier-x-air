"""Learner-side Phase-10B repair primitives.

This module intentionally has no access to twin truth, scenarios, or blind
outcomes.  It consumes only observable features and noisy discovery effects.
"""
from __future__ import annotations

import numpy as np

from .graph import GraphSnapshot
from .schema import NetworkObservation


MECHANISMS = ("AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY", "AIRPORT_CAPACITY")
WRONG_MECHANISM = {
    "AIRCRAFT_ROTATION": "RESOURCE_DEPENDENCY",
    "RESOURCE_DEPENDENCY": "AIRCRAFT_ROTATION",
    "AIRPORT_CAPACITY": "AIRCRAFT_ROTATION",
}


def observable_feature(mechanism: str, observation: NetworkObservation, graph: GraphSnapshot) -> np.ndarray:
    """Feature available at prediction time for one specified structural family."""
    values = np.zeros(len(graph.flight_ids), dtype=float)
    index = {flight_id: i for i, flight_id in enumerate(graph.flight_ids)}
    if mechanism in {"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY"}:
        for source, target, data in graph.graph.edges(data=True):
            if data.get("relation") == mechanism:
                values[index[target]] += observation.delays[source]
    elif mechanism == "AIRPORT_CAPACITY":
        values += max(0.0, 1.0 - observation.capacity)
    else:
        raise ValueError(f"unknown mechanism: {mechanism}")
    return values


def generic_feature(observation: NetworkObservation, graph: GraphSnapshot) -> np.ndarray:
    """Predefined non-mechanistic schedule-position feature, with no fitted term."""
    del observation
    return np.asarray([graph.graph.in_degree(flight_id) for flight_id in graph.flight_ids], dtype=float)


def estimate_from_discovery(records: list[dict]) -> dict[str, dict[str, float | int]]:
    """No-intercept least squares using discovery effect/feature pairs only."""
    estimates: dict[str, dict[str, float | int]] = {}
    for mechanism in MECHANISMS:
        rows = [r for r in records if r["mechanism"] == mechanism]
        x = np.asarray([r["feature_sum"] for r in rows], dtype=float)
        y = np.asarray([r["observed_effect"] for r in rows], dtype=float)
        coefficient = float(np.dot(x, y) / np.dot(x, x)) if len(x) and np.dot(x, x) > 1e-12 else 0.0
        estimates[mechanism] = {"coefficient": coefficient, "sample_count": len(rows)}
        
    x_gen = np.asarray([r["generic_feature_sum"] for r in records], dtype=float)
    y_gen = np.asarray([r["observed_effect"] for r in records], dtype=float)
    generic_coefficient = float(np.dot(x_gen, y_gen) / np.dot(x_gen, x_gen)) if len(x_gen) and np.dot(x_gen, x_gen) > 1e-12 else 0.0
    estimates["GENERIC"] = {"coefficient": generic_coefficient, "sample_count": len(records)}
    return estimates


def apply_repair(prediction: np.ndarray, mechanism: str, coefficient: float,
                 observation: NetworkObservation, graph: GraphSnapshot) -> np.ndarray:
    return prediction + coefficient * observable_feature(mechanism, observation, graph)


def apply_generic_repair(prediction: np.ndarray, coefficient: float, observation: NetworkObservation, graph: GraphSnapshot) -> np.ndarray:
    return prediction + coefficient * generic_feature(observation, graph)
