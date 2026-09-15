"""Learner-side repair module for the Final Robustness block.

No oracle truth, no beta_s, no blind data, no hidden twin state.
Consumes only observable features and noisy discovery effects.
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
                if target in index:
                    values[index[target]] += observation.delays.get(source, 0.0)
    elif mechanism == "AIRPORT_CAPACITY":
        values += max(0.0, 1.0 - observation.capacity)
    else:
        raise ValueError(f"unknown mechanism: {mechanism}")
    return values

def generic_feature(observation, graph):
    del observation
    return np.asarray([graph.graph.in_degree(fid) for fid in graph.flight_ids], dtype=float)

def nonlinear_feature(observation, graph, tau=5.0):
    """Threshold feature: max(0, source_delay - tau) for each AR target."""
    values = np.zeros(len(graph.flight_ids), dtype=float)
    index = {fid: i for i, fid in enumerate(graph.flight_ids)}
    for source, target, data in graph.graph.edges(data=True):
        if data.get("relation") == "AIRCRAFT_ROTATION" and target in index:
            values[index[target]] += max(0.0, observation.delays.get(source, 0.0) - tau)
    return values

def estimate_from_discovery(records):
    """No-intercept OLS. Returns coefficient, sample_count, ols_se per mechanism."""
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

    x_gen = np.asarray([r.get("generic_feature_sum", 0.0) for r in records], dtype=float)
    y_gen = np.asarray([r["observed_effect"] for r in records], dtype=float)
    if len(x_gen) > 1 and np.dot(x_gen, x_gen) > 1e-12:
        gc = float(np.dot(x_gen, y_gen) / np.dot(x_gen, x_gen))
        resid_g = y_gen - gc * x_gen
        se_g = float(np.std(resid_g, ddof=1)) / float(np.sqrt(np.dot(x_gen, x_gen)))
    else:
        gc = 0.0; se_g = 0.0
    estimates["GENERIC"] = {"coefficient": gc, "sample_count": len(records), "ols_se": se_g}

    # Nonlinear threshold estimator (used only for NL experiment)
    nl_rows = [r for r in records if r.get("nl_feature_sum") is not None]
    x_nl = np.asarray([r["nl_feature_sum"] for r in nl_rows], dtype=float)
    y_nl = np.asarray([r["observed_effect"] for r in nl_rows], dtype=float)
    if len(x_nl) > 1 and np.dot(x_nl, x_nl) > 1e-12:
        nl_coeff = float(np.dot(x_nl, y_nl) / np.dot(x_nl, x_nl))
        nl_resid = y_nl - nl_coeff * x_nl
        nl_se = float(np.std(nl_resid, ddof=1)) / float(np.sqrt(np.dot(x_nl, x_nl)))
    else:
        nl_coeff = 0.0; nl_se = 0.0
    estimates["NONLINEAR"] = {"coefficient": nl_coeff, "sample_count": len(nl_rows), "ols_se": nl_se}

    return estimates

def apply_repair(prediction, mechanism, coefficient, observation, graph):
    return prediction + coefficient * observable_feature(mechanism, observation, graph)

def apply_generic_repair(prediction, coefficient, observation, graph):
    return prediction + coefficient * generic_feature(observation, graph)

def apply_nonlinear_repair(prediction, coefficient, observation, graph, tau=5.0):
    return prediction + coefficient * nonlinear_feature(observation, graph, tau)

# --- External baselines (learner-accessible) ---

def residual_magnitude_repair(prediction, target_mean, observation, graph):
    """Passive residual-based repair: shifts prediction toward observed mean.
    This is a data-driven, non-causal, non-intervention baseline."""
    obs_mean = float(np.mean(list(observation.delays.values())))
    pred_mean = float(np.mean(prediction))
    shift = obs_mean - pred_mean
    return prediction + shift * 0.5  # conservative shrinkage

def random_mechanism_repair(prediction, observation, graph, coefficient, rng_seed=0):
    """Applies a randomly selected mechanism repair without identification."""
    rng = np.random.default_rng(rng_seed)
    mech = rng.choice(list(MECHANISMS))
    return apply_repair(prediction, mech, coefficient, observation, graph)
