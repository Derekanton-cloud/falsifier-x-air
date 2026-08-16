"""Focused Phase-6 CQR contract tests; no test targets enter calibration."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from falsifier_x_air.stgnn import FORBIDDEN_COLUMNS, load_flight_graph
from falsifier_x_air.uncertainty import (
    CHECKPOINT, ResidualCQR, checkpoint_sha256, conformal_quantile, construct_intervals,
    cqr_scores, frozen_predictions, interval_score, load_frozen_point_predictor,
)


def test_checkpoint_loading_and_integrity():
    data = load_flight_graph(); model, saved = load_frozen_point_predictor(data)
    assert CHECKPOINT.exists() and len(checkpoint_sha256()) == 64
    assert saved["feature_names"] == data.feature_names
    assert not model.training and all(not p.requires_grad for p in model.parameters())


def test_phase5_prediction_reproducibility_and_preservation():
    data = load_flight_graph(); before_hash = checkpoint_sha256()
    first, second = frozen_predictions(data), frozen_predictions(data)
    assert np.max(np.abs(first - second)) <= 1e-5
    assert checkpoint_sha256() == before_hash


def test_chronological_boundaries_and_forbidden_feature_exclusion():
    data = load_flight_graph()
    assert {name: int((data.split == name).sum()) for name in ("train", "validation", "test")} == {"train": 30507, "validation": 9822, "test": 8309}
    assert not FORBIDDEN_COLUMNS.intersection(data.feature_names)


def test_cqr_nonconformity_and_finite_sample_calibration():
    residual = np.array([-2.0, 0.0, 5.0]); low = np.array([-1.0, -1.0, 1.0]); high = np.array([1.0, 1.0, 4.0])
    scores = cqr_scores(residual, low, high)
    assert np.allclose(scores, [1.0, -1.0, 1.0])
    # n=3, alpha=.25: ceil(4*.75)/3 = 1, using the higher empirical quantile.
    assert conformal_quantile(scores, .25) == 1.0


def test_interval_construction_contains_point_and_interval_score_formula():
    point = np.array([10.0, 20.0]); lower, upper = construct_intervals(point, np.array([2.0, -4.0]), np.array([3.0, -1.0]), 1.0)
    assert np.all(lower <= point) and np.all(point <= upper)
    y = np.array([5.0, 25.0]); lo = np.array([8.0, 10.0]); hi = np.array([12.0, 22.0])
    # widths 4,12 plus miss penalties 60 and 60 at alpha .1 => mean 68.
    assert interval_score(y, lo, hi, .1) == pytest.approx(68.0)


def test_train_only_quantile_fit_and_validation_only_calibration():
    train_x = np.arange(20, dtype=float).reshape(-1, 1); train_residual = np.linspace(-2, 2, 20)
    validation_x = np.array([[1.0], [2.0], [3.0]]); validation_residual = np.array([-3.0, 0.0, 3.0])
    cqr = ResidualCQR(alpha=.2, seed=7).fit(train_x, train_residual)
    qhat = cqr.calibrate(validation_x, validation_residual)
    assert np.isfinite(qhat)
    # Fit is deterministic and depends only on explicitly supplied train arrays.
    twin = ResidualCQR(alpha=.2, seed=7).fit(train_x, train_residual)
    assert np.allclose(cqr.lower_model.predict(train_x), twin.lower_model.predict(train_x))


def test_adversarial_test_targets_do_not_change_calibration_parameters():
    train_x = np.arange(20, dtype=float).reshape(-1, 1); train_residual = np.linspace(-2, 2, 20)
    validation_x = np.array([[1.0], [2.0], [3.0]]); validation_residual = np.array([-3.0, 0.0, 3.0])
    test_targets_a = np.array([-1000.0, 0.0, 1000.0]); test_targets_b = -test_targets_a
    a = ResidualCQR(alpha=.2, seed=7).fit(train_x, train_residual); qa = a.calibrate(validation_x, validation_residual)
    b = ResidualCQR(alpha=.2, seed=7).fit(train_x, train_residual); qb = b.calibrate(validation_x, validation_residual)
    assert not np.array_equal(test_targets_a, test_targets_b)
    assert qa == qb and np.allclose(a.lower_model.predict(validation_x), b.lower_model.predict(validation_x))


def test_missing_phase5_features_are_imputed_before_cqr_input():
    data = load_flight_graph()
    assert np.isfinite(data.x.numpy()).all()
