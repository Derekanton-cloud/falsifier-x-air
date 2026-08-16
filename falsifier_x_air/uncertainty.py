"""Phase 6: residual conformalized quantile regression around frozen Phase 5."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.ensemble import GradientBoostingRegressor

from .data.config import load_config
from .data.features import generate_features, prepare_training_data
from .stgnn import FORBIDDEN_COLUMNS, FlightSTGNN, load_flight_graph, metrics, set_seed

ALPHA = 0.10
SEED = 42
CHECKPOINT = Path("data/processed/stgnn_checkpoint.pt")


def checkpoint_sha256(path: Path = CHECKPOINT) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_point_predictor(data, checkpoint: Path = CHECKPOINT) -> tuple[FlightSTGNN, dict]:
    """Load, validate, and freeze the Phase-5 checkpoint without writing it."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["feature_names"] != data.feature_names:
        raise ValueError("Phase-5 checkpoint feature names do not match prepared Phase-5 inputs")
    hidden = int(saved["report"]["hyperparameters"]["hidden"])
    model = FlightSTGNN(data.x.shape[1], hidden=hidden)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, saved


def frozen_predictions(data, checkpoint: Path = CHECKPOINT) -> np.ndarray:
    model, _ = load_frozen_point_predictor(data, checkpoint)
    with torch.no_grad():
        return model(data).cpu().numpy()


def cqr_scores(residual: np.ndarray, lower_residual: np.ndarray, upper_residual: np.ndarray) -> np.ndarray:
    """Standard CQR nonconformity: distance outside the residual quantile band."""
    return np.maximum(lower_residual - residual, residual - upper_residual)


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    if not 0 < alpha < 1 or len(scores) == 0:
        raise ValueError("alpha must be in (0, 1) and calibration scores cannot be empty")
    level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


def construct_intervals(point: np.ndarray, lower_residual: np.ndarray, upper_residual: np.ndarray, qhat: float) -> tuple[np.ndarray, np.ndarray]:
    """CQR residual interval, conservatively enlarged to contain the point estimate."""
    lower = point + np.minimum(lower_residual - qhat, 0.0)
    upper = point + np.maximum(upper_residual + qhat, 0.0)
    return lower, upper


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = ALPHA) -> float:
    return float(np.mean((upper - lower) + (2 / alpha) * (lower - y) * (y < lower) + (2 / alpha) * (y - upper) * (y > upper)))


class ResidualCQR:
    """Two train-only residual quantile regressors and validation-only CQR calibration."""
    def __init__(self, alpha: float = ALPHA, seed: int = SEED) -> None:
        self.alpha, self.seed, self.qhat = alpha, seed, None
        # Fixed, deliberately modest downstream models: no validation/test tuning.
        common = dict(n_estimators=100, max_depth=3, min_samples_leaf=20, random_state=seed)
        self.lower_model = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, **common)
        self.upper_model = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, **common)

    def fit(self, x_train: np.ndarray, residual_train: np.ndarray) -> "ResidualCQR":
        self.lower_model.fit(x_train, residual_train)
        self.upper_model.fit(x_train, residual_train)
        return self

    def residual_quantiles(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.lower_model.predict(x), self.upper_model.predict(x)

    def calibrate(self, x_validation: np.ndarray, residual_validation: np.ndarray) -> float:
        lower, upper = self.residual_quantiles(x_validation)
        self.qhat = conformal_quantile(cqr_scores(residual_validation, lower, upper), self.alpha)
        return self.qhat

    def predict_interval(self, x: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.qhat is None:
            raise RuntimeError("calibrate on validation data before constructing intervals")
        lower, upper = self.residual_quantiles(x)
        return construct_intervals(point, lower, upper, self.qhat)


def _metadata(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    joined_path = config.processed / "joined" / "flights_weather.parquet"
    ready_path = config.processed / "graphs" / "baseline_ready.parquet"
    joined = pd.read_parquet(joined_path) if joined_path.exists() else pd.read_csv(joined_path.with_suffix(".csv"), low_memory=False)
    ready = pd.read_parquet(ready_path) if ready_path.exists() else pd.read_csv(ready_path.with_suffix(".csv"), low_memory=False)
    features = generate_features(joined)
    if FORBIDDEN_COLUMNS.intersection(features.columns):
        raise ValueError("Forbidden Phase-5 feature found while preparing uncertainty metadata")
    columns = ["flight_id", "carrier", "origin_airport", "scheduled_departure_utc"]
    return prepare_training_data(features, ready)[columns].sort_values("scheduled_departure_utc").reset_index(drop=True)


def _summary(y: np.ndarray, point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    return {**metrics(y, point), "coverage": float(np.mean((y >= lower) & (y <= upper))), "mean_interval_width": float(np.mean(upper - lower)), "median_interval_width": float(np.median(upper - lower)), "interval_score": interval_score(y, lower, upper)}


def _subgroups(meta: pd.DataFrame, y: np.ndarray, point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    frame = meta.copy(); frame["y"], frame["point"], frame["lower"], frame["upper"] = y, point, lower, upper
    frame["date"] = pd.to_datetime(frame.scheduled_departure_utc, utc=True).dt.date.astype(str)
    frame["delay_magnitude"] = pd.cut(frame.y, [-np.inf, -15, 15, 60, np.inf], labels=["early(<-15)", "on_time[-15,15)", "delayed[15,60)", "severe(>=60)"])
    def grouped(column: str, min_size: int = 100) -> dict:
        output = {}
        for key, group in frame.groupby(column, observed=True):
            if len(group) >= min_size:
                output[str(key)] = {"n": len(group), **_summary(group.y.to_numpy(), group.point.to_numpy(), group.lower.to_numpy(), group.upper.to_numpy())}
        return output
    return {"chronological_test_day": grouped("date"), "origin_airport": grouped("origin_airport"), "carrier": grouped("carrier"), "delay_magnitude": grouped("delay_magnitude", 1)}


def write_phase6_report(result: dict, path: Path = Path("data/processed/phase6_cqr_report.md")) -> None:
    """Create a concise human-readable companion to the machine-readable result."""
    test, validation, cqr, integrity = result["test"], result["validation"], result["cqr"], result["phase5_integrity"]
    path.write_text(f"""# Phase 6: residual CQR uncertainty layer

Phase 6 leaves the official Phase-5 ST-GNN untouched and uses its frozen point
prediction plus the same 340 approved, prediction-time encoded features. Two
separate GradientBoosting quantile models predict the 5th and 95th percentile
of the residual `arrival_delay_minutes - frozen_point_prediction`, fitted on
the **30,507 train flights only**. The entire **9,822-flight validation period**
is the sole conformal-calibration set. The **8,309-flight test period** is used
only for this final evaluation.

For alpha = 0.10, CQR uses `max(q_lower_residual - residual, residual -
q_upper_residual)` and its finite-sample conformal quantile `qhat = {cqr['qhat']:.6f}`.
The final residual bounds are expanded conservatively to include zero, ensuring
that every interval contains the frozen point estimate. The interval score is
`(upper-lower) + (2/alpha)(lower-y) I(y<lower) + (2/alpha)(y-upper) I(y>upper)`.

| Split | Coverage | Mean width | Median width | Interval score | Point MAE | Point RMSE | Point R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation (calibration) | {validation['coverage']:.4f} | {validation['mean_interval_width']:.2f} | {validation['median_interval_width']:.2f} | {validation['interval_score']:.2f} | {validation['MAE']:.2f} | {validation['RMSE']:.2f} | {validation['R2']:.4f} |
| Test (held out) | {test['coverage']:.4f} | {test['mean_interval_width']:.2f} | {test['median_interval_width']:.2f} | {test['interval_score']:.2f} | {test['MAE']:.2f} | {test['RMSE']:.2f} | {test['R2']:.4f} |

Subgroup results are in `phase6_cqr_results.json`; groups below 100 flights are
not reported except the requested delay-magnitude categories. Test-day coverage
ranged from 0.8384 to 0.8884. Severe delays had 0.5047 coverage (n=428), a
material limitation rather than evidence of calibration under tail events.
The UTC timestamp-based test table includes 293 rows on 2024-02-01 from the
existing immutable prepared artifact; it is retained under the established
chronological split and was not moved or used for development.

No actual-time, outcome, delay-cause, cancellation/diversion, future-weather,
or test-target information enters fitting or calibration. The Phase-5 checkpoint
SHA-256 was `{integrity['checkpoint_sha256_before']}` both before and after;
the maximum Phase-5 prediction difference was {integrity['max_abs_prediction_difference']:.1f}.

This provides finite-sample marginal conformal calibration under the usual
exchangeability assumptions, not a guarantee under temporal distribution shift.
The next step is a preregistered rolling-origin/prospective calibration study;
do not alter the frozen Phase-5 predictor.
""", encoding="utf-8")


def run_phase6(config_path: str = "configs/data.toml", checkpoint: Path = CHECKPOINT) -> dict:
    """Fit on train, calibrate once on validation, then perform held-out test evaluation."""
    set_seed(SEED); hash_before = checkpoint_sha256(checkpoint)
    data = load_flight_graph(config_path); meta = _metadata(config_path)
    if not np.array_equal(meta.flight_id.astype(str).to_numpy(), data.flight_ids):
        raise ValueError("Phase-6 metadata does not align with Phase-5 graph flight ordering")
    point_before = frozen_predictions(data, checkpoint)
    x = np.column_stack((data.x.numpy(), point_before)).astype(np.float32)
    y, split = data.y.numpy(), data.split
    train, validation, test = split == "train", split == "validation", split == "test"
    cqr = ResidualCQR(); started = time.perf_counter()
    cqr.fit(x[train], y[train] - point_before[train])
    qhat = cqr.calibrate(x[validation], y[validation] - point_before[validation])
    fit_seconds = time.perf_counter() - started
    # Persist the frozen downstream layer separately; never overwrite Phase 5.
    joblib.dump({"lower_model": cqr.lower_model, "upper_model": cqr.upper_model, "qhat": qhat,
                 "alpha": cqr.alpha, "feature_names": data.feature_names + ["frozen_phase5_point_prediction"]},
                "data/processed/phase6_cqr_models.joblib")
    lower, upper = cqr.predict_interval(x[test], point_before[test])
    point_after = frozen_predictions(data, checkpoint)
    hash_after = checkpoint_sha256(checkpoint)
    prediction_difference = float(np.max(np.abs(point_before - point_after)))
    if hash_before != hash_after or prediction_difference > 1e-5:
        raise RuntimeError("Frozen Phase-5 checkpoint/predictions changed during Phase 6")
    result = {
        "phase": 6, "method": "Residual conformalized quantile regression (CQR)", "prediction_target": "arrival_delay_minutes", "prediction_timestamp": "scheduled_departure_utc",
        "boundaries": {"quantile_fit": "train only: 2024-01-01 through 2024-01-20", "calibration": "entire validation only: 2024-01-21 through 2024-01-26", "evaluation": "held-out test only: 2024-01-27 through 2024-01-31"},
        "dataset_sizes": {"train": int(train.sum()), "validation": int(validation.sum()), "test": int(test.sum())},
        "features": {"phase5_encoded_prediction_time_features": int(data.x.shape[1]), "additional": "frozen Phase-5 point prediction", "forbidden": sorted(FORBIDDEN_COLUMNS)},
        "cqr": {"alpha": ALPHA, "nominal_coverage": 1 - ALPHA, "residual_quantiles": [ALPHA / 2, 1 - ALPHA / 2], "qhat": qhat, "nonconformity": "max(q_lower_residual - residual, residual - q_upper_residual)", "point_containment": "residual bounds are conservatively expanded to include zero"},
        "quantile_models": {"algorithm": "GradientBoostingRegressor", "n_estimators": 100, "max_depth": 3, "min_samples_leaf": 20, "random_state": SEED},
        "validation": _summary(y[validation], point_before[validation], *cqr.predict_interval(x[validation], point_before[validation])),
        "test": _summary(y[test], point_before[test], lower, upper), "subgroups_test": _subgroups(meta.loc[test].reset_index(drop=True), y[test], point_before[test], lower, upper),
        "phase5_integrity": {"checkpoint_sha256_before": hash_before, "checkpoint_sha256_after": hash_after, "max_abs_prediction_difference": prediction_difference},
        "timing_seconds": {"fit_and_calibration": fit_seconds}, "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(), "seed": SEED},
    }
    Path("data/processed/phase6_cqr_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_phase6_report(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/data.toml"); args = parser.parse_args()
    print(json.dumps(run_phase6(args.config), indent=2))


if __name__ == "__main__": main()
