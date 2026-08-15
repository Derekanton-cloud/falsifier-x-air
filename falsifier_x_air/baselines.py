"""Baseline predictors for aviation delay: Ridge, Random Forest, Gradient Boosting, Naive Mean."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data.config import load_config
from .data.features import generate_features, prepare_training_data


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R² from true and predicted arrays."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def run_baselines(config_path: str):
    config = load_config(config_path)

    # 1. Load Data
    joined_path = config.processed / "joined" / "flights_weather.csv"
    baseline_ready_path = config.processed / "graphs" / "baseline_ready.csv"

    if not joined_path.exists():
        # Fallback to parquet if csv doesn't exist
        joined_path = config.processed / "joined" / "flights_weather.parquet"
        baseline_ready_path = config.processed / "graphs" / "baseline_ready.parquet"

    df_joined = pd.read_csv(joined_path, low_memory=False) if joined_path.suffix == ".csv" else pd.read_parquet(joined_path)
    df_ready = pd.read_csv(baseline_ready_path) if baseline_ready_path.suffix == ".csv" else pd.read_parquet(baseline_ready_path)

    # 2. Generate Features
    print("Generating features...")
    features = generate_features(df_joined)
    data = prepare_training_data(features, df_ready)

    # 3. Define Columns
    categorical_cols = ["carrier", "origin_airport", "destination_airport"]
    numerical_cols = [
        "scheduled_duration_minutes", "distance",
        "dep_hour", "dep_day_of_week",
        "origin_weather_temperature_c", "origin_weather_dew_point_c",
        "origin_weather_pressure_hpa", "origin_weather_visibility_m",
        "origin_weather_wind_direction_degrees", "origin_weather_wind_speed_mps",
        "prev_aircraft_delay", "airport_congestion"
    ]
    target_col = "target_arrival_delay_minutes"

    # 4. Split — chronological, no random split
    train_df = data[data["split"] == "train"]
    val_df = data[data["split"] == "validation"]
    test_df = data[data["split"] == "test"]

    X_train = train_df[categorical_cols + numerical_cols]
    y_train = train_df[target_col].to_numpy()
    X_val = val_df[categorical_cols + numerical_cols]
    y_val = val_df[target_col].to_numpy()
    X_test = test_df[categorical_cols + numerical_cols]
    y_test = test_df[target_col].to_numpy()

    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)

    print(f"Train samples: {n_train}")
    print(f"Val samples:   {n_val}")
    print(f"Test samples:  {n_test}")
    print(f"Input feature count (pre-OHE): {len(categorical_cols) + len(numerical_cols)}")

    # 5. Naive Mean Baseline — fit only on train
    naive_train_mean = float(y_train.mean())
    naive_val_preds = np.full(n_val, naive_train_mean)
    naive_test_preds = np.full(n_test, naive_train_mean)
    naive_val_metrics = _evaluate(y_val, naive_val_preds)
    naive_test_metrics = _evaluate(y_test, naive_test_preds)
    print(
        f"NaiveMean  train_mean={naive_train_mean:.2f}  "
        f"val: MAE={naive_val_metrics['MAE']:.2f} RMSE={naive_val_metrics['RMSE']:.2f} R2={naive_val_metrics['R2']:.4f}  "
        f"test: MAE={naive_test_metrics['MAE']:.2f} RMSE={naive_test_metrics['RMSE']:.2f} R2={naive_test_metrics['R2']:.4f}"
    )

    # 6. Preprocessing Pipeline
    # Imputer strategy="mean" is fitted only on training data inside Pipeline.fit().
    # StandardScaler is also fitted only on training data.
    # Fitting happens via pipeline.fit(X_train, y_train) for each model separately.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()

        # Clone the preprocessor configuration for each model so each pipeline
        # fits its own preprocessor independently on X_train.
        from sklearn.base import clone as sklearn_clone
        pipeline = Pipeline([
            ("preprocessor", sklearn_clone(preprocessor)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Compute post-OHE feature count from the fitted pipeline
        fitted_preprocessor = pipeline.named_steps["preprocessor"]
        n_features_out = fitted_preprocessor.transform(X_train[:1]).shape[1]

        val_preds = pipeline.predict(X_val)
        test_preds = pipeline.predict(X_test)

        val_m = _evaluate(y_val, val_preds)
        test_m = _evaluate(y_test, test_preds)

        results[name] = {
            "validation": val_m,
            "test": test_m,
            "training_time_seconds": round(train_time, 3),
            "input_feature_count": len(categorical_cols) + len(numerical_cols),
            "post_ohe_feature_count": n_features_out,
        }
        print(
            f"{name:18s}  val:  MAE={val_m['MAE']:.2f} RMSE={val_m['RMSE']:.2f} R2={val_m['R2']:.4f}"
            f"  test: MAE={test_m['MAE']:.2f} RMSE={test_m['RMSE']:.2f} R2={test_m['R2']:.4f}"
        )

    # 7. Environment and report
    import sys
    import importlib.metadata as importlib_metadata
    import platform

    env = {
        "python_version": sys.version.split()[0],
        "pandas_version": importlib_metadata.version("pandas"),
        "numpy_version": importlib_metadata.version("numpy"),
        "scikit_learn_version": importlib_metadata.version("scikit-learn"),
        "platform": platform.platform(),
        "random_seeds": {"RandomForest": 42, "GradientBoosting": 42},
    }

    report = {
        "prediction_target": "arrival_delay_minutes",
        "prediction_timestamp": "scheduled_departure_utc",
        "split_boundaries": {
            "train": "2024-01-01 through 2024-01-20",
            "validation": "2024-01-21 through 2024-01-26",
            "test": "2024-01-27 through 2024-01-31",
        },
        "dataset_sizes": {
            "train": n_train,
            "validation": n_val,
            "test": n_test,
        },
        "features": categorical_cols + numerical_cols,
        "missing_value_handling": "SimpleImputer(strategy='mean'), fitted on training data only",
        "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
        "naive_mean_baseline": {
            "train_mean": naive_train_mean,
            "validation": naive_val_metrics,
            "test": naive_test_metrics,
        },
        "model_results": results,
        "environment": env,
    }

    output_path = Path("data/processed/baseline_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.toml")
    args = parser.parse_args()
    run_baselines(args.config)
