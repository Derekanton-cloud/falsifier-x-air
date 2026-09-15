"""Learner-side Phase-10C repair primitives.

This module is intentionally separate from phase10b_repair to preserve Phase 10B isolation.
No access to twin truth, per-world mechanism strength, scenarios, or blind outcomes.
Consumes only observable features and noisy discovery intervention effects.
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

def observable_feature(mechanism, observation, graph):
    values = np.zeros(len(graph.flight_ids), dtype=float)
    index = {fid: i for i, fid in enumerate(graph.flight_ids)}
    if mechanism in {"AIRCRAFT_ROTATION", "RESOURCE_DEPENDENCY"}:
        for source, target, data in graph.graph.edges(data=True):
            if data.get("relation") == mechanism:
                values[index[target]] += observation.delays[source]
    elif mechanism == "AIRPORT_CAPACITY":
        values += max(0.0, 1.0 - observation.capacity)
    else:
        raise ValueError(f"unknown mechanism: {mechanism}")
    return values

def generic_feature(observation, graph):
    del observation
    return np.asarray([graph.graph.in_degree(fid) for fid in graph.flight_ids], dtype=float)

def estimate_from_discovery(records):
    estimates = {}
    for mechanism in MECHANISMS:
        rows = [r for r in records if r["mechanism"] == mechanism]
        x = np.asarray([r["feature_sum"] for r in rows], dtype=float)
        y = np.asarray([r["observed_effect"] for r in rows], dtype=float)
        if len(x) > 1 and np.dot(x, x) > 1e-12:
            coeff = float(np.dot(x, y) / np.dot(x, x))
            resid = y - coeff * x
            se = float(np.std(resid, ddof=1)) / float(np.sqrt(np.dot(x, x)))
        elif len(x) == 1 and np.dot(x, x) > 1e-12:
            coeff = float(np.dot(x, y) / np.dot(x, x))
            se = 0.0
        else:
            coeff = 0.0
            se = 0.0
        estimates[mechanism] = {"coefficient": coeff, "sample_count": len(rows), "ols_se": se}
    x_gen = np.asarray([r["generic_feature_sum"] for r in records], dtype=float)
    y_gen = np.asarray([r["observed_effect"] for r in records], dtype=float)
    if len(x_gen) > 1 and np.dot(x_gen, x_gen) > 1e-12:
        gc = float(np.dot(x_gen, y_gen) / np.dot(x_gen, x_gen))
        resid_g = y_gen - gc * x_gen
        se_g = float(np.std(resid_g, ddof=1)) / float(np.sqrt(np.dot(x_gen, x_gen)))
    else:
        gc = 0.0
        se_g = 0.0
    estimates["GENERIC"] = {"coefficient": gc, "sample_count": len(records), "ols_se": se_g}
    return estimates

def apply_repair(prediction, mechanism, coefficient, observation, graph):
    return prediction + coefficient * observable_feature(mechanism, observation, graph)

def apply_generic_repair(prediction, coefficient, observation, graph):
    return prediction + coefficient * generic_feature(observation, graph)
