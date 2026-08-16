"""Causal, flight-level spatio-temporal GNN for the prepared BTS/NOAA data.

The graph is deliberately restricted to observed, historical flight-to-flight
relations.  Its convolution applies an in-model ``source_time < target_time``
mask, independently of the graph builder, so malformed edges cannot inject an
equal-time or future representation.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data.config import load_config
from .data.features import generate_features, prepare_training_data

RELATIONS = ("AIRCRAFT_ROTATION", "TEMPORAL_AIRPORT_PROPAGATION")
CATEGORICAL_COLUMNS = ("carrier", "origin_airport", "destination_airport")
NUMERICAL_COLUMNS = (
    "scheduled_duration_minutes", "distance", "dep_hour", "dep_day_of_week",
    "origin_weather_temperature_c", "origin_weather_dew_point_c",
    "origin_weather_pressure_hpa", "origin_weather_visibility_m",
    "origin_weather_wind_direction_degrees", "origin_weather_wind_speed_mps",
    "prev_aircraft_delay", "airport_congestion",
)
FORBIDDEN_COLUMNS = frozenset((
    "actual_arrival_utc", "arrival_delay_minutes", "actual_departure_utc",
    "departure_delay_minutes", "cancelled", "diverted", "cancellation_code",
    "carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay",
))


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.read_csv(path.with_suffix(".csv"), low_memory=False)


@dataclass
class FlightGraphData:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    source_time: torch.Tensor
    target_time: torch.Tensor
    split: np.ndarray
    flight_ids: np.ndarray
    feature_names: list[str]
    graph_report: dict[str, int]

    def mask(self, name: str) -> torch.Tensor:
        return torch.tensor(self.split == name, dtype=torch.bool)


class TrainOnlyFeatureEncoder:
    """Mean-impute/standardise numeric values and one-hot categories from train only."""
    def fit_transform(self, frame: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
        self.fit(frame.loc[train_mask])
        return self.transform(frame), self.names

    def fit(self, frame: pd.DataFrame) -> "TrainOnlyFeatureEncoder":
        self.means = frame.loc[:, NUMERICAL_COLUMNS].apply(pd.to_numeric, errors="coerce").mean().fillna(0.0)
        std = frame.loc[:, NUMERICAL_COLUMNS].apply(pd.to_numeric, errors="coerce").std().fillna(1.0)
        self.stds = std.mask(std == 0, 1.0)
        self.categories = {c: sorted(frame[c].fillna("__MISSING__").astype(str).unique()) for c in CATEGORICAL_COLUMNS}
        self.names = list(NUMERICAL_COLUMNS) + [f"{c}={v}" for c in CATEGORICAL_COLUMNS for v in self.categories[c]]
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = frame.loc[:, NUMERICAL_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(self.means)
        blocks = [((numeric - self.means) / self.stds).to_numpy(dtype=np.float32)]
        for column in CATEGORICAL_COLUMNS:
            values = frame[column].fillna("__MISSING__").astype(str).to_numpy()
            cats = self.categories[column]
            blocks.append(np.column_stack([(values == cat).astype(np.float32) for cat in cats]))
        return np.concatenate(blocks, axis=1)


def load_flight_graph(config_path: str = "configs/data.toml") -> FlightGraphData:
    config = load_config(config_path)
    joined = _read(config.processed / "joined" / "flights_weather.parquet")
    ready = _read(config.processed / "graphs" / "baseline_ready.parquet")
    edges = _read(config.processed / "graphs" / "edges.parquet")
    features = generate_features(joined)
    data = prepare_training_data(features, ready).sort_values("scheduled_departure_utc").reset_index(drop=True)
    if FORBIDDEN_COLUMNS.intersection(features.columns):
        raise ValueError("Forbidden post-outcome feature found in ST-GNN inputs")
    if data.empty or not set(("train", "validation", "test")).issubset(data.split.unique()):
        raise ValueError("Prepared data does not contain all chronological splits")
    encoder = TrainOnlyFeatureEncoder()
    x, names = encoder.fit_transform(data, (data.split == "train").to_numpy())
    ids = data.flight_id.astype(str).to_numpy()
    index = {fid: i for i, fid in enumerate(ids)}
    temporal = edges[edges.edge_type.isin(RELATIONS)].copy()
    temporal["source_time"] = pd.to_datetime(temporal.source_time, utc=True)
    temporal["target_time"] = pd.to_datetime(temporal.target_time, utc=True)
    temporal = temporal[temporal.source.astype(str).isin(index) & temporal.target.astype(str).isin(index)]
    src = temporal.source.astype(str).map(index).to_numpy(dtype=np.int64)
    dst = temporal.target.astype(str).map(index).to_numpy(dtype=np.int64)
    edge_type = temporal.edge_type.map({name: i for i, name in enumerate(RELATIONS)}).to_numpy(dtype=np.int64)
    seconds = lambda s: pd.to_datetime(s, utc=True).astype("int64").to_numpy(dtype=np.float64) / 1e9
    return FlightGraphData(
        x=torch.from_numpy(x), y=torch.tensor(data.target_arrival_delay_minutes.to_numpy(dtype=np.float32)),
        edge_index=torch.tensor(np.vstack([src, dst]), dtype=torch.long), edge_type=torch.tensor(edge_type, dtype=torch.long),
        source_time=torch.tensor(seconds(temporal.source_time), dtype=torch.float64), target_time=torch.tensor(seconds(temporal.target_time), dtype=torch.float64),
        split=data.split.astype(str).to_numpy(), flight_ids=ids, feature_names=names,
        graph_report={"flight_nodes": len(data), "message_edges": len(temporal), **{k: int(v) for k, v in temporal.edge_type.value_counts().items()}},
    )


class CausalMessagePassing(torch.nn.Module):
    """One-hop relation-specific message passing with strict temporal filtering."""
    def __init__(self, hidden: int, relations: int = len(RELATIONS)) -> None:
        super().__init__()
        self.relation = torch.nn.Embedding(relations, hidden)
        self.time = torch.nn.Sequential(torch.nn.Linear(1, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden))
        self.message = torch.nn.Sequential(torch.nn.Linear(hidden * 3, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden))
        self.update = torch.nn.Sequential(torch.nn.Linear(hidden * 2, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor) -> torch.Tensor:
        valid = source_time < target_time  # model-level leakage boundary
        src, dst = edge_index[:, valid]
        if src.numel() == 0:
            return h
        age_hours = ((target_time[valid] - source_time[valid]).float() / 3600).clamp(max=24 * 14).unsqueeze(1)
        msg = self.message(torch.cat((h[src], self.relation(edge_type[valid]), self.time(torch.log1p(age_hours))), dim=1))
        aggregate = torch.zeros_like(h); aggregate.index_add_(0, dst, msg)
        counts = torch.zeros((h.size(0), 1), dtype=h.dtype, device=h.device)
        counts.index_add_(0, dst, torch.ones((len(dst), 1), dtype=h.dtype, device=h.device))
        return h + self.update(torch.cat((h, aggregate / counts.clamp_min(1)), dim=1))


class FlightSTGNN(torch.nn.Module):
    def __init__(self, features: int, hidden: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(features, hidden), torch.nn.SiLU(), torch.nn.LayerNorm(hidden), torch.nn.Dropout(dropout))
        self.convolution = CausalMessagePassing(hidden)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, hidden // 2), torch.nn.SiLU(), torch.nn.Linear(hidden // 2, 1))

    def forward(self, data: FlightGraphData) -> torch.Tensor:
        h = self.encoder(data.x)
        h = self.convolution(h, data.edge_index, data.edge_type, data.source_time, data.target_time)
        return self.head(h).squeeze(-1)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"MAE": float(mean_absolute_error(y, pred)), "RMSE": float(np.sqrt(mean_squared_error(y, pred))), "R2": float(r2_score(y, pred))}


def train_stgnn(config_path: str = "configs/data.toml", seed: int = 42, epochs: int = 100, patience: int = 12, hidden: int = 64) -> dict:
    set_seed(seed); data = load_flight_graph(config_path); model = FlightSTGNN(data.x.shape[1], hidden=hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    train_mask, val_mask, test_mask = data.mask("train"), data.mask("validation"), data.mask("test")
    best_state, best_mae, stale = None, float("inf"), 0; started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(); pred = model(data); loss = torch.nn.functional.smooth_l1_loss(pred[train_mask], data.y[train_mask]); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_mae = float(torch.mean(torch.abs(model(data)[val_mask] - data.y[val_mask])))
        if val_mae < best_mae:
            best_mae, stale = val_mae, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if stale >= patience: break
    training_seconds = time.perf_counter() - started; model.load_state_dict(best_state); model.eval()
    started = time.perf_counter()
    with torch.no_grad(): prediction = model(data).cpu().numpy()
    inference_seconds = time.perf_counter() - started
    result = {
        "prediction_target": "arrival_delay_minutes", "prediction_timestamp": "scheduled_departure_utc", "architecture": "one-hop causal relation-aware temporal message passing (64 hidden units)",
        "relations": list(RELATIONS), "causality": "CausalMessagePassing filters every edge with source_time < target_time at forward time; invalid edges are ignored.",
        "dataset_sizes": {name: int(mask.sum()) for name, mask in (("train", train_mask), ("validation", val_mask), ("test", test_mask))},
        "graph": data.graph_report, "feature_dimensions": {"raw_prediction_features": len(CATEGORICAL_COLUMNS) + len(NUMERICAL_COLUMNS), "encoded": int(data.x.shape[1])},
        "features": list(CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS), "normalization": "numeric imputation, means, standard deviations, and categorical vocabularies fit on train only",
        "hyperparameters": {"seed": seed, "hidden": hidden, "max_epochs": epochs, "patience": patience, "optimizer": "AdamW(lr=0.002, weight_decay=0.0001)", "loss": "SmoothL1"},
        "best_epoch": epoch - stale, "parameter_count": int(sum(p.numel() for p in model.parameters())), "training_time_seconds": training_seconds, "inference_time_seconds": inference_seconds,
        "validation": metrics(data.y[val_mask].numpy(), prediction[val_mask.numpy()]), "test": metrics(data.y[test_mask].numpy(), prediction[test_mask.numpy()]),
        "environment": {"python_version": sys.version.split()[0], "pytorch_version": torch.__version__, "pyg_version": None, "platform": platform.platform(), "seed": seed},
    }
    output = Path("data/processed/stgnn_results.json"); output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    checkpoint = Path("data/processed/stgnn_checkpoint.pt"); torch.save({"state_dict": model.state_dict(), "feature_names": data.feature_names, "report": result}, checkpoint)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/data.toml"); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--patience", type=int, default=12); args = parser.parse_args()
    print(json.dumps(train_stgnn(args.config, args.seed, args.epochs, args.patience), indent=2))


if __name__ == "__main__": main()
